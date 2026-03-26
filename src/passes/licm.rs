//! Loop-Invariant Code Motion (LICM)
//!
//! Hoists computations out of loops when their operands do not change within
//! the loop body, reducing redundant work per iteration.
//!
//! ## Algorithm
//!
//! 1. **Detect natural loops**: Find back-edges in the CFG (edges where the
//!    target dominates the source). For each back-edge, compute the loop body
//!    as the set of blocks from which the back-edge source is reachable without
//!    going through the loop header.
//! 2. **Identify invariant instructions**: An instruction in a loop is invariant
//!    if all its operands are either:
//!    - Defined outside the loop
//!    - Defined by another invariant instruction
//!    - Constants
//! 3. **Hoist invariant instructions**: Move them to the loop's preheader
//!    (the block that dominates the loop header).
//!
//! ## Constraints
//!
//! Only side-effect-free instructions are hoisted (no stores, calls, or
//! volatile operations). Instructions that may trap (division) are not
//! hoisted to preserve exception semantics.
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};

/// Represents a natural loop in the CFG.
struct NaturalLoop {
    /// Loop header block index.
    #[allow(dead_code)]
    header: usize,
    /// Set of block indices in the loop body (includes header).
    body: FxHashSet<usize>,
    /// Preheader block index (unique predecessor of header outside loop).
    /// None if no unique preheader exists.
    preheader: Option<usize>,
}

/// Finds all natural loops in the function.
///
/// A natural loop is defined by a back-edge (source → header) where the
/// header dominates the source. The loop body is the set of blocks that
/// can reach the source without going through the header.
fn find_loops(func: &IrFunction) -> Vec<NaturalLoop> {
    let num_blocks = func.blocks().len();
    if num_blocks == 0 {
        return vec![];
    }

    // Compute dominators using simple iterative algorithm
    let rpo = func.reverse_postorder();
    let mut doms: Vec<Option<usize>> = vec![None; num_blocks];
    doms[0] = Some(0); // Entry dominates itself

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            if b == 0 {
                continue;
            }
            let preds = func.predecessors(b);
            let mut new_idom: Option<usize> = None;
            for &p in preds {
                if doms[p].is_some() {
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect_doms(&doms, &rpo, cur, p),
                    });
                }
            }
            if new_idom != doms[b] {
                doms[b] = new_idom;
                changed = true;
            }
        }
    }

    // Find back-edges: b → h where h dominates b
    let mut loops = Vec::new();
    for b in 0..num_blocks {
        let succs = func.successors(b);
        for &h in succs.iter() {
            if dominates(&doms, h, b) {
                // Found back-edge b → h
                let mut body = FxHashSet::default();
                body.insert(h);
                collect_loop_body(func, b, &mut body);

                // Find preheader: unique predecessor of header not in loop
                let header_preds = func.predecessors(h);
                let outside_preds: Vec<usize> = header_preds
                    .iter()
                    .filter(|p| !body.contains(p))
                    .copied()
                    .collect();
                let preheader = if outside_preds.len() == 1 {
                    Some(outside_preds[0])
                } else {
                    None
                };

                loops.push(NaturalLoop {
                    header: h,
                    body,
                    preheader,
                });
            }
        }
    }

    loops
}

/// Intersect two dominators in the dominator tree.
fn intersect_doms(doms: &[Option<usize>], rpo: &[usize], mut a: usize, mut b: usize) -> usize {
    // Build RPO position map
    let mut rpo_pos = vec![0usize; doms.len()];
    for (pos, &block) in rpo.iter().enumerate() {
        if block < rpo_pos.len() {
            rpo_pos[block] = pos;
        }
    }

    while a != b {
        while rpo_pos.get(a).copied().unwrap_or(0) > rpo_pos.get(b).copied().unwrap_or(0) {
            a = doms[a].unwrap_or(0);
        }
        while rpo_pos.get(b).copied().unwrap_or(0) > rpo_pos.get(a).copied().unwrap_or(0) {
            b = doms[b].unwrap_or(0);
        }
    }
    a
}

/// Checks if `a` dominates `b`.
fn dominates(doms: &[Option<usize>], a: usize, b: usize) -> bool {
    let mut cur = b;
    let mut steps = 0;
    while cur != a {
        match doms.get(cur).and_then(|d| *d) {
            Some(d) if d != cur => {
                cur = d;
                steps += 1;
                if steps > doms.len() {
                    return false; // Safety net
                }
            }
            _ => return false,
        }
    }
    true
}

/// Collect all blocks in the loop body using reverse DFS from the back-edge source.
fn collect_loop_body(func: &IrFunction, source: usize, body: &mut FxHashSet<usize>) {
    if body.contains(&source) {
        return;
    }
    body.insert(source);
    let preds = func.predecessors(source);
    for &p in preds {
        if !body.contains(&p) {
            collect_loop_body(func, p, body);
        }
    }
}

/// Checks if an instruction is safe to hoist (no side effects, no trapping).
fn is_hoistable(inst: &Instruction) -> bool {
    match inst {
        // Pure computation — safe to hoist
        Instruction::BinOp { op, .. } => {
            // Don't hoist divisions (may trap on divide-by-zero)
            !matches!(
                op,
                crate::ir::instructions::BinOp::SDiv
                    | crate::ir::instructions::BinOp::UDiv
                    | crate::ir::instructions::BinOp::SRem
                    | crate::ir::instructions::BinOp::URem
            )
        }
        Instruction::ICmp { .. }
        | Instruction::FCmp { .. }
        | Instruction::ZExt { .. }
        | Instruction::SExt { .. }
        | Instruction::Trunc { .. }
        | Instruction::BitCast { .. }
        | Instruction::GetElementPtr { .. }
        | Instruction::IntToPtr { .. }
        | Instruction::PtrToInt { .. } => true,

        // NOT safe to hoist
        Instruction::Store { .. }
        | Instruction::Load { .. }
        | Instruction::Call { .. }
        | Instruction::Alloca { .. }
        | Instruction::StackAlloc { .. }
        | Instruction::StackSave { .. }
        | Instruction::StackRestore { .. }
        | Instruction::Phi { .. }
        | Instruction::InlineAsm { .. }
        | Instruction::Branch { .. }
        | Instruction::CondBranch { .. }
        | Instruction::Switch { .. }
        | Instruction::IndirectBranch { .. }
        | Instruction::Return { .. }
        | Instruction::BlockAddress { .. } => false,
    }
}

/// Runs LICM on a single function.
///
/// Returns `true` if any instruction was hoisted out of a loop.
pub fn run_licm(func: &mut IrFunction) -> bool {
    // Rebuild CFG edges first
    func.rebuild_cfg_edges();

    let loops = find_loops(func);
    if loops.is_empty() {
        return false;
    }

    // Build def map: which block defines each value
    let mut def_block: FxHashMap<Value, usize> = FxHashMap::default();
    for (block_idx, block) in func.blocks().iter().enumerate() {
        for inst in block.instructions() {
            if let Some(result) = inst.result() {
                def_block.insert(result, block_idx);
            }
        }
    }

    let mut changed = false;

    for lp in &loops {
        let preheader = match lp.preheader {
            Some(ph) => ph,
            None => continue, // No preheader — can't hoist
        };

        // Iteratively find invariant instructions
        let mut invariant_vals: FxHashSet<Value> = FxHashSet::default();
        let mut found_new = true;
        let mut iterations = 0;

        while found_new && iterations < 100 {
            found_new = false;
            iterations += 1;

            for &block_idx in &lp.body {
                if block_idx >= func.blocks().len() {
                    continue;
                }
                let block = &func.blocks()[block_idx];
                for inst in block.instructions() {
                    if !is_hoistable(inst) {
                        continue;
                    }
                    let result = match inst.result() {
                        Some(r) => r,
                        None => continue,
                    };
                    if invariant_vals.contains(&result) {
                        continue;
                    }

                    // Check if all operands are loop-invariant
                    let all_invariant = inst.operands().iter().all(|&op| {
                        // Defined outside loop
                        if let Some(&def_b) = def_block.get(&op) {
                            !lp.body.contains(&def_b) || invariant_vals.contains(&op)
                        } else {
                            true // Function parameter or constant — invariant
                        }
                    });

                    if all_invariant {
                        invariant_vals.insert(result);
                        found_new = true;
                    }
                }
            }
        }

        if invariant_vals.is_empty() {
            continue;
        }

        // Hoist invariant instructions to preheader
        // Collect instructions to move
        let mut to_hoist: Vec<Instruction> = Vec::new();

        for &block_idx in &lp.body {
            if block_idx >= func.blocks().len() {
                continue;
            }
            let block = func.get_block_mut(block_idx).unwrap();
            let mut remaining = Vec::new();
            for inst in block.instructions.drain(..) {
                if let Some(result) = inst.result() {
                    if invariant_vals.contains(&result) {
                        to_hoist.push(inst);
                        changed = true;
                        continue;
                    }
                }
                remaining.push(inst);
            }
            block.instructions = remaining;
        }

        // Insert hoisted instructions at end of preheader (before terminator)
        if !to_hoist.is_empty() && preheader < func.blocks().len() {
            let ph_block = func.get_block_mut(preheader).unwrap();
            let term_pos = ph_block
                .instructions
                .iter()
                .position(|i| i.is_terminator())
                .unwrap_or(ph_block.instructions.len());
            for (i, inst) in to_hoist.into_iter().enumerate() {
                ph_block.instructions.insert(term_pos + i, inst);
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
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{BlockId, Instruction};
    use crate::ir::types::IrType;

    #[test]
    fn test_licm_no_loops() {
        let mut func = IrFunction::new("test_licm".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        let changed = run_licm(&mut func);
        assert!(!changed);
    }

    #[test]
    fn test_find_loops_simple() {
        // Build a simple loop: bb0 → bb1, bb1 → bb1 (self-loop)
        let mut func = IrFunction::new("test_loop".to_string(), vec![], IrType::Void);

        // bb0: branch to bb1
        func.blocks[0].instructions.push(Instruction::Branch {
            target: BlockId(1),
            span: Span::dummy(),
        });

        // bb1: branch to bb1 (self-loop)
        let bb1_idx = func.add_block(BasicBlock::new(1));
        func.blocks[bb1_idx].instructions.push(Instruction::Branch {
            target: BlockId(1),
            span: Span::dummy(),
        });

        func.rebuild_cfg_edges();
        let loops = find_loops(&func);
        assert!(!loops.is_empty());
        assert!(loops[0].body.contains(&1));
    }

    #[test]
    fn test_dominates() {
        // Simple chain: 0 → 1 → 2
        let doms = vec![Some(0), Some(0), Some(1)];
        assert!(dominates(&doms, 0, 2)); // 0 dominates 2
        assert!(dominates(&doms, 0, 1)); // 0 dominates 1
        assert!(dominates(&doms, 1, 2)); // 1 dominates 2
        assert!(!dominates(&doms, 2, 0)); // 2 does not dominate 0
    }
}
