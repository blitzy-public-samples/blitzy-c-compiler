//! Dominance Frontier Computation
//!
//! Computes the dominance frontier (DF) for each basic block in the control flow graph.
//! The dominance frontier of block B is the set of blocks Y such that:
//! - B dominates a predecessor of Y, BUT
//! - B does NOT strictly dominate Y
//!
//! The dominance frontier determines where phi nodes need to be placed during
//! SSA construction (mem2reg). For each variable definition in block B, phi nodes
//! must be inserted at every block in DF(B).
//!
//! This module also provides the **iterated dominance frontier** (IDF) computation,
//! which extends the basic DF to handle the transitive closure: phi nodes at DF(B)
//! create new definitions that may require additional phi nodes at DF(DF(B)), etc.
//!
//! ## Algorithm
//!
//! The dominance frontier computation uses the efficient algorithm from
//! Cooper, Harvey, and Kennedy ("A Simple, Fast Dominance Algorithm", 2001).
//! For each join point (block with 2+ predecessors), the algorithm walks up the
//! dominator tree from each predecessor, adding the join point to the dominance
//! frontier of every block encountered along the way until the immediate
//! dominator of the join point is reached.
//!
//! The overall complexity is O(|edges| × α(|nodes|)), where α is the inverse
//! Ackermann function — effectively linear in the size of the CFG.
//!
//! ## Iterated Dominance Frontier (IDF)
//!
//! The IDF extends the basic DF to compute the fixed-point:
//!   IDF = DF(def_blocks ∪ IDF)
//!
//! This is necessary because placing a phi node at a DF location creates a new
//! "definition" that may in turn require phi nodes at DF(that location), and so
//! on transitively. The worklist-based algorithm terminates in O(|blocks|) since
//! each block is processed at most once.
//!
//! Uses the dominator tree from [`super::dominator_tree::DominatorTree`].

use super::dominator_tree::DominatorTree;
use crate::common::fx_hash::FxHashSet;
use crate::ir::function::IrFunction;

// ===========================================================================
// DominanceFrontiers — the per-block frontier data structure
// ===========================================================================

/// Dominance frontiers for all basic blocks in a function.
///
/// Maps each block index to the set of blocks in its dominance frontier.
/// The dominance frontier of block X is the set of blocks Y where X dominates
/// a predecessor of Y but does not strictly dominate Y itself.
///
/// This is the key data structure driving phi-node placement in the mem2reg
/// SSA construction pass. For each promoted variable with a definition in
/// block X, phi nodes must be inserted at every block in DF(X).
///
/// # Construction
///
/// Use [`compute_dominance_frontiers`] to build this from an [`IrFunction`]
/// and its [`DominatorTree`].
///
/// # Example
///
/// ```ignore
/// use bcc::ir::mem2reg::dominator_tree::DominatorTree;
/// use bcc::ir::mem2reg::dominance_frontier::{compute_dominance_frontiers, DominanceFrontiers};
///
/// let dom_tree = DominatorTree::build(&func);
/// let df = compute_dominance_frontiers(&func, &dom_tree);
///
/// // Check which blocks are in the dominance frontier of block 0
/// for &block in df.frontier(0).iter() {
///     println!("Block {} is in DF(0)", block);
/// }
/// ```
pub struct DominanceFrontiers {
    /// `frontiers[block_index]` = set of block indices in the dominance
    /// frontier of that block.
    ///
    /// Indexed by block index within the parent function's block list.
    /// Each entry is an `FxHashSet<usize>` for O(1) membership testing
    /// during the iterated dominance frontier computation.
    frontiers: Vec<FxHashSet<usize>>,
}

// ===========================================================================
// DominanceFrontiers — query methods
// ===========================================================================

impl DominanceFrontiers {
    /// Get the dominance frontier of a specific block.
    ///
    /// Returns a reference to the set of block indices that comprise the
    /// dominance frontier of the block at `block_index`.
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the block whose frontier is requested.
    ///
    /// # Panics
    ///
    /// Panics if `block_index >= self.block_count()`.
    #[inline]
    pub fn frontier(&self, block_index: usize) -> &FxHashSet<usize> {
        &self.frontiers[block_index]
    }

    /// Get the number of blocks for which frontiers are stored.
    ///
    /// This matches the block count of the function from which the
    /// frontiers were computed. Includes both reachable and unreachable
    /// blocks (unreachable blocks simply have empty frontiers).
    #[inline]
    pub fn block_count(&self) -> usize {
        self.frontiers.len()
    }

    /// Check if block `y` is in the dominance frontier of block `x`.
    ///
    /// Returns `true` if and only if `y` ∈ DF(`x`).
    ///
    /// # Parameters
    ///
    /// - `x`: Block whose dominance frontier to query.
    /// - `y`: Block to check for membership in DF(`x`).
    ///
    /// # Returns
    ///
    /// `false` if either `x` or `y` is out of bounds, or if `y` is not
    /// in the dominance frontier of `x`.
    #[inline]
    pub fn contains(&self, x: usize, y: usize) -> bool {
        if x >= self.frontiers.len() {
            return false;
        }
        self.frontiers[x].contains(&y)
    }
}

// ===========================================================================
// Dominance Frontier Computation — Cooper-Harvey-Kennedy Algorithm
// ===========================================================================

/// Compute dominance frontiers for all blocks in the function.
///
/// Implements the efficient algorithm from Cooper, Harvey, and Kennedy
/// ("A Simple, Fast Dominance Algorithm", 2001):
///
/// ```text
/// For each block B in the function:
///   If B has >= 2 predecessors (it is a "join point"):
///     For each predecessor P of B:
///       runner = P
///       While runner != idom(B):
///         DF(runner) += {B}
///         runner = idom(runner)
/// ```
///
/// This walks up the dominator tree from each predecessor of each join
/// point, adding the join point to the dominance frontier of every block
/// encountered until the immediate dominator of the join point is reached.
///
/// ## Complexity
///
/// O(|edges| × depth_of_dominator_tree). For practical CFGs (including
/// Linux kernel functions), the dominator tree depth is logarithmic in
/// the number of blocks, making this effectively O(|edges| × log |blocks|).
///
/// ## Correctness Properties
///
/// - The entry block (block 0) has no idom — join points whose idom is
///   the entry block use `None` as the sentinel, causing the walk to stop
///   at the entry block itself.
/// - Single-predecessor blocks contribute nothing to any frontier — only
///   join points (blocks with 2+ predecessors) produce frontier entries.
/// - Self-loops are handled correctly: a block can appear in its own
///   dominance frontier if it has a back edge to itself.
/// - Irreducible control flow is handled correctly — the algorithm works
///   on any CFG structure.
///
/// # Parameters
///
/// - `func`: The function whose CFG frontiers are to be computed.
/// - `dom_tree`: The dominator tree for `func`, computed by
///   [`DominatorTree::build`] or [`DominatorTree::build_simple`].
///
/// # Returns
///
/// A [`DominanceFrontiers`] structure containing the frontier set for
/// every block in the function.
pub fn compute_dominance_frontiers(
    func: &IrFunction,
    dom_tree: &DominatorTree,
) -> DominanceFrontiers {
    let block_count = func.block_count();

    // Initialize empty frontier sets for each block.
    let mut frontiers: Vec<FxHashSet<usize>> = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        frontiers.push(FxHashSet::default());
    }

    // For each block B in the function, if B is a join point (has 2+
    // predecessors), walk up the dominator tree from each predecessor
    // and add B to the frontier of every block along the way.
    let blocks = func.blocks();
    for (b_idx, block) in blocks.iter().enumerate() {
        // Only process join points — blocks with 2 or more predecessors.
        // Blocks with 0 or 1 predecessors cannot contribute to any
        // dominance frontier through this algorithm.
        if block.predecessor_count() < 2 {
            continue;
        }

        // Get the immediate dominator of B. If B is the entry block
        // (idom returns None), we use a sentinel that no runner will
        // equal, causing the walk to stop at the entry block.
        let idom_b = dom_tree.idom(b_idx);

        for &pred in block.predecessors() {
            // Walk up the dominator tree from the predecessor `pred`
            // toward (but not including) idom(B).
            let mut runner = pred;

            // The loop terminates when the runner reaches idom(B).
            // For the entry block case (idom_b == None), the loop
            // terminates when runner has no idom (it is the entry block).
            loop {
                // If runner equals idom(B), stop — idom(B) is NOT
                // included in the walk (it strictly dominates B).
                match idom_b {
                    Some(idom_val) if runner == idom_val => break,
                    None => {
                        // B's idom is None (B is entry or unreachable).
                        // If B is the entry block, no block can be in DF
                        // because every block is dominated by entry.
                        // However, the entry block having 2+ predecessors
                        // is unusual (back edges to entry). In this case,
                        // we still walk up but stop when runner has no idom.
                        // Add B to DF(runner) first, then try to walk up.
                        frontiers[runner].insert(b_idx);
                        match dom_tree.idom(runner) {
                            Some(next) => runner = next,
                            None => break, // runner is also root/unreachable
                        }
                        continue;
                    }
                    _ => {}
                }

                // Add B to the dominance frontier of runner.
                frontiers[runner].insert(b_idx);

                // Walk up: move runner to its immediate dominator.
                match dom_tree.idom(runner) {
                    Some(next) => runner = next,
                    None => break, // runner is the entry block or unreachable
                }
            }
        }
    }

    DominanceFrontiers { frontiers }
}

// ===========================================================================
// Iterated Dominance Frontier (IDF) — Phi-Node Placement
// ===========================================================================

/// Compute the iterated dominance frontier (IDF) for a set of definition blocks.
///
/// The IDF is the fixed-point of:
///   IDF = DF(def_blocks ∪ IDF)
///
/// This is needed because placing a phi node at a DF location creates a new
/// "definition" that may require additional phi nodes at DF(that location).
/// The worklist-based algorithm converges to the exact set of blocks requiring
/// phi nodes for a single promoted variable.
///
/// ## Algorithm (Worklist-Based)
///
/// 1. Initialize worklist = def_blocks
/// 2. Initialize visited = def_blocks (to avoid reprocessing)
/// 3. Initialize idf_result = empty
/// 4. While worklist is not empty:
///    - Pop block B from worklist
///    - For each block F in DF(B):
///      - If F not already in idf_result:
///        - Add F to idf_result
///        - If F not in visited:
///          - Add F to visited
///          - Add F to worklist (new phi = new definition)
/// 5. Return idf_result
///
/// ## Termination
///
/// The algorithm always terminates because:
/// - The function has a finite number of blocks.
/// - Each block is added to the worklist at most once (via the visited set).
/// - Therefore the main loop executes at most |blocks| iterations.
///
/// # Parameters
///
/// - `def_blocks`: Set of block indices containing definitions (stores) for
///   a single promoted variable.
/// - `dominance_frontiers`: Pre-computed dominance frontiers from
///   [`compute_dominance_frontiers`].
///
/// # Returns
///
/// Set of block indices where phi nodes should be inserted for this variable.
pub fn compute_iterated_dominance_frontier(
    def_blocks: &FxHashSet<usize>,
    dominance_frontiers: &DominanceFrontiers,
) -> FxHashSet<usize> {
    let mut idf_result: FxHashSet<usize> = FxHashSet::default();
    let mut visited: FxHashSet<usize> = FxHashSet::default();
    let mut worklist: Vec<usize> = Vec::with_capacity(def_blocks.len());

    // Seed the worklist and visited set with all definition blocks.
    for &def_block in def_blocks.iter() {
        worklist.push(def_block);
        visited.insert(def_block);
    }

    // Process worklist until empty.
    while let Some(b) = worklist.pop() {
        // Skip blocks that are out of range for the frontier data.
        if b >= dominance_frontiers.block_count() {
            continue;
        }

        // For each block F in DF(B):
        for &f in dominance_frontiers.frontier(b).iter() {
            // If F is not already in the IDF result set, add it.
            if idf_result.insert(f) {
                // F was newly added. If we haven't visited F yet,
                // add it to the worklist so its DF is also processed
                // (the phi node at F creates a new definition).
                if visited.insert(f) {
                    worklist.push(f);
                }
            }
        }
    }

    idf_result
}

// ===========================================================================
// Store Frontiers in BasicBlocks
// ===========================================================================

/// Store computed dominance frontiers into the BasicBlock structures.
///
/// For each block in the function, writes the dominance frontier as a sorted
/// `Vec<usize>` into the block's `dominance_frontier` field via
/// [`BasicBlock::set_dominance_frontier`].
///
/// This allows subsequent passes to access the frontier from the block itself
/// without needing a reference to the [`DominanceFrontiers`] structure.
///
/// # Parameters
///
/// - `func`: Mutable reference to the function whose blocks will be updated.
/// - `frontiers`: Pre-computed dominance frontiers from
///   [`compute_dominance_frontiers`].
pub fn store_frontiers_in_blocks(func: &mut IrFunction, frontiers: &DominanceFrontiers) {
    let block_count = func.block_count();
    let frontier_count = frontiers.block_count();
    let count = block_count.min(frontier_count);

    let blocks = func.blocks_mut();
    for (i, block) in blocks.iter_mut().enumerate().take(count) {
        // Collect the frontier set into a sorted Vec for deterministic
        // iteration in subsequent passes.
        let mut frontier_vec: Vec<usize> = frontiers.frontier(i).iter().copied().collect();
        frontier_vec.sort_unstable();
        block.set_dominance_frontier(frontier_vec);
    }
}

// ===========================================================================
// Utility — Strict Dominance Query
// ===========================================================================

/// Check if block `a` strictly dominates block `b`.
///
/// A strictly dominates B if A dominates B and A ≠ B.
///
/// This is a convenience wrapper that delegates to
/// [`DominatorTree::strictly_dominates`]. It is provided as a free function
/// for use in contexts where calling the method directly is less ergonomic.
///
/// # Parameters
///
/// - `dom_tree`: The dominator tree for the function.
/// - `a`: Block index to test as the dominator.
/// - `b`: Block index to test as the dominated block.
///
/// # Returns
///
/// `true` if `a` strictly dominates `b`, `false` otherwise.
/// Returns `false` if `a == b` (strict dominance excludes self-dominance).
/// Returns `false` if either block is unreachable or out of bounds.
#[inline]
pub fn strictly_dominates(dom_tree: &DominatorTree, a: usize, b: usize) -> bool {
    dom_tree.strictly_dominates(a, b)
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{BlockId, Instruction, Value};
    use crate::ir::mem2reg::dominator_tree::DominatorTree;
    use crate::ir::types::IrType;

    /// Helper: build a function with the specified number of blocks and
    /// the given CFG edges. Each block gets a simple return terminator
    /// (except as overridden by edge setup).
    fn build_cfg(num_blocks: usize, edges: &[(usize, usize)]) -> IrFunction {
        let mut func = IrFunction::new("test".to_string(), vec![], IrType::Void);

        // Remove the default entry block and rebuild from scratch.
        func.blocks.clear();
        for i in 0..num_blocks {
            let label = format!("bb{}", i);
            let bb = BasicBlock::with_label(i, label);
            func.blocks.push(bb);
        }

        // Add edges: for each (src, dst), add terminator-implied edges.
        for &(src, dst) in edges {
            func.blocks[src].add_successor(dst);
            func.blocks[dst].add_predecessor(src);
        }

        let dummy_span = Span::dummy();

        // Add return terminators to blocks without successors.
        for block in &mut func.blocks {
            if block.successors().is_empty() {
                block.push_instruction(Instruction::Return {
                    value: None,
                    span: dummy_span,
                });
            } else if block.successors().len() == 1 {
                let target = BlockId(block.successors()[0] as u32);
                block.push_instruction(Instruction::Branch {
                    target,
                    span: dummy_span,
                });
            } else if block.successors().len() >= 2 {
                let succs = block.successors().to_vec();
                let then_block = BlockId(succs[0] as u32);
                let else_block = BlockId(succs[1] as u32);
                block.push_instruction(Instruction::CondBranch {
                    condition: Value(0),
                    then_block,
                    else_block,
                    span: dummy_span,
                });
            }
        }

        func
    }

    #[test]
    fn test_single_block_no_frontiers() {
        // Single block, no edges → empty frontiers.
        let func = build_cfg(1, &[]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert_eq!(df.block_count(), 1);
        assert!(df.frontier(0).is_empty());
    }

    #[test]
    fn test_linear_chain_no_frontiers() {
        // Linear: 0 → 1 → 2 → 3
        // No join points, so all frontiers are empty.
        let func = build_cfg(4, &[(0, 1), (1, 2), (2, 3)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert_eq!(df.block_count(), 4);
        for i in 0..4 {
            assert!(
                df.frontier(i).is_empty(),
                "Block {} should have empty DF, got {:?}",
                i,
                df.frontier(i)
            );
        }
    }

    #[test]
    fn test_diamond_cfg() {
        // Diamond:
        //     0
        //    / \
        //   1   2
        //    \ /
        //     3
        //
        // Block 3 is a join point with predecessors 1 and 2.
        // idom(3) = 0, idom(1) = 0, idom(2) = 0.
        //
        // For pred=1: runner=1, idom(3)=0, so DF(1) += {3}, runner=idom(1)=0, stop.
        // For pred=2: runner=2, idom(3)=0, so DF(2) += {3}, runner=idom(2)=0, stop.
        //
        // Expected: DF(0)={}, DF(1)={3}, DF(2)={3}, DF(3)={}
        let func = build_cfg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert_eq!(df.block_count(), 4);
        assert!(df.frontier(0).is_empty(), "DF(0) should be empty");
        assert!(df.contains(1, 3), "3 should be in DF(1)");
        assert_eq!(
            df.frontier(1).len(),
            1,
            "DF(1) should have exactly 1 element"
        );
        assert!(df.contains(2, 3), "3 should be in DF(2)");
        assert_eq!(
            df.frontier(2).len(),
            1,
            "DF(2) should have exactly 1 element"
        );
        assert!(df.frontier(3).is_empty(), "DF(3) should be empty");
    }

    #[test]
    fn test_if_then_else_nested() {
        // Nested if-then-else:
        //       0
        //      / \
        //     1   2
        //     |  / \
        //     | 3   4
        //     | |   |
        //     \ |  /
        //      \|/
        //       5
        //
        // Edges: 0→1, 0→2, 1→5, 2→3, 2→4, 3→5, 4→5
        // Block 5 has predecessors [1, 3, 4].
        // idom(5)=0, idom(1)=0, idom(2)=0, idom(3)=2, idom(4)=2
        //
        // For pred=1: runner=1, idom(5)=0, DF(1)+={5}, runner=0, stop.
        // For pred=3: runner=3, DF(3)+={5}, runner=idom(3)=2, DF(2)+={5}, runner=idom(2)=0, stop.
        // For pred=4: runner=4, DF(4)+={5}, runner=idom(4)=2, DF(2) already has 5, runner=0, stop.
        //
        // Expected: DF(0)={}, DF(1)={5}, DF(2)={5}, DF(3)={5}, DF(4)={5}, DF(5)={}
        let func = build_cfg(6, &[(0, 1), (0, 2), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert_eq!(df.block_count(), 6);
        assert!(df.frontier(0).is_empty());
        assert!(df.contains(1, 5));
        assert!(df.contains(2, 5));
        assert!(df.contains(3, 5));
        assert!(df.contains(4, 5));
        assert!(df.frontier(5).is_empty());
    }

    #[test]
    fn test_simple_loop() {
        // Simple loop:
        //   0 → 1 → 2
        //       ^   |
        //       +---+
        //
        // Edges: 0→1, 1→2, 2→1
        // Block 1 has predecessors [0, 2] → join point.
        // idom(1)=0, idom(2)=1
        //
        // For pred=0: runner=0, idom(1)=0, stop. (entry is idom of 1, no DF added)
        // For pred=2: runner=2, idom(1)=0, DF(2)+={1}, runner=idom(2)=1, DF(1)+={1}, runner=idom(1)=0, stop.
        //
        // Expected: DF(0)={}, DF(1)={1}, DF(2)={1}
        let func = build_cfg(3, &[(0, 1), (1, 2), (2, 1)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert_eq!(df.block_count(), 3);
        assert!(df.frontier(0).is_empty(), "DF(0) should be empty");
        assert!(
            df.contains(1, 1),
            "1 should be in DF(1) (self-loop back edge)"
        );
        assert!(df.contains(2, 1), "1 should be in DF(2)");
    }

    #[test]
    fn test_iterated_dominance_frontier_diamond() {
        // Diamond: 0→1, 0→2, 1→3, 2→3
        // DF(1)={3}, DF(2)={3}
        //
        // If def_blocks = {1}, IDF = DF(1) = {3}.
        // Since DF(3) = {}, no further expansion → IDF = {3}.
        let func = build_cfg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        let mut def_blocks = FxHashSet::default();
        def_blocks.insert(1);

        let idf = compute_iterated_dominance_frontier(&def_blocks, &df);
        assert!(idf.contains(&3), "3 should be in IDF({{1}})");
        assert_eq!(idf.len(), 1, "IDF({{1}}) should have exactly 1 element");
    }

    #[test]
    fn test_iterated_dominance_frontier_chain() {
        // Chain with multiple merge points:
        //     0
        //    / \
        //   1   2
        //    \ /
        //     3
        //    / \
        //   4   5
        //    \ /
        //     6
        //
        // DF(1)={3}, DF(2)={3}, DF(4)={6}, DF(5)={6}
        //
        // If def_blocks = {1, 4}, IDF:
        //   - Process 1: DF(1)={3}, add 3 to IDF, add 3 to worklist
        //   - Process 4: DF(4)={6}, add 6 to IDF, add 6 to worklist
        //   - Process 3: DF(3)={} (3 is a join point but no further frontier)
        //   - Process 6: DF(6)={} (similar)
        //   → IDF = {3, 6}
        let func = build_cfg(
            7,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 6),
                (5, 6),
            ],
        );
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        let mut def_blocks = FxHashSet::default();
        def_blocks.insert(1);
        def_blocks.insert(4);

        let idf = compute_iterated_dominance_frontier(&def_blocks, &df);
        assert!(idf.contains(&3), "3 should be in IDF({{1, 4}})");
        assert!(idf.contains(&6), "6 should be in IDF({{1, 4}})");
        assert_eq!(idf.len(), 2);
    }

    #[test]
    fn test_iterated_dominance_frontier_empty_defs() {
        // No definition blocks → IDF should be empty.
        let func = build_cfg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        let def_blocks = FxHashSet::default();
        let idf = compute_iterated_dominance_frontier(&def_blocks, &df);
        assert!(idf.is_empty(), "IDF of empty def_blocks should be empty");
    }

    #[test]
    fn test_store_frontiers_in_blocks() {
        // Diamond: 0→1, 0→2, 1→3, 2→3
        let mut func = build_cfg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        store_frontiers_in_blocks(&mut func, &df);

        // Verify that the frontiers are stored in the blocks.
        assert!(func.blocks()[0].dominance_frontier().is_empty());
        assert_eq!(func.blocks()[1].dominance_frontier(), &[3]);
        assert_eq!(func.blocks()[2].dominance_frontier(), &[3]);
        assert!(func.blocks()[3].dominance_frontier().is_empty());
    }

    #[test]
    fn test_strictly_dominates_basic() {
        // Linear: 0 → 1 → 2
        let func = build_cfg(3, &[(0, 1), (1, 2)]);
        let dom_tree = DominatorTree::build(&func);

        // 0 strictly dominates 1 and 2.
        assert!(strictly_dominates(&dom_tree, 0, 1));
        assert!(strictly_dominates(&dom_tree, 0, 2));
        // 1 strictly dominates 2.
        assert!(strictly_dominates(&dom_tree, 1, 2));
        // Self-dominance is NOT strict.
        assert!(!strictly_dominates(&dom_tree, 0, 0));
        assert!(!strictly_dominates(&dom_tree, 1, 1));
        // 2 does not dominate 0 or 1.
        assert!(!strictly_dominates(&dom_tree, 2, 0));
        assert!(!strictly_dominates(&dom_tree, 2, 1));
    }

    #[test]
    fn test_contains_out_of_bounds() {
        let func = build_cfg(2, &[(0, 1)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        // Out-of-bounds queries should return false, not panic.
        assert!(!df.contains(100, 0));
        assert!(!df.contains(0, 100));
    }

    #[test]
    fn test_self_loop_dominance_frontier() {
        // Self-loop: 0 → 1 → 1 (and 1 → 2 for exit)
        //
        //   0 → 1 ←─┐
        //       |    |
        //       +────+ (self-loop)
        //       |
        //       v
        //       2
        //
        // Block 1 has predecessors [0, 1] → join point.
        // idom(1) = 0
        //
        // For pred=0: runner=0, idom(1)=0, stop.
        // For pred=1: runner=1, idom(1)=0, DF(1)+={1}, runner=idom(1)=0, stop.
        //
        // DF(1) = {1} — block 1 is in its own dominance frontier.
        let func = build_cfg(3, &[(0, 1), (1, 1), (1, 2)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert!(
            df.contains(1, 1),
            "Block 1 should be in its own DF (self-loop)"
        );
    }

    #[test]
    fn test_empty_function() {
        // A function with 0 blocks (unusual but should not panic).
        let mut func = IrFunction::new("empty".to_string(), vec![], IrType::Void);
        func.blocks.clear();
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert_eq!(df.block_count(), 0);
    }

    #[test]
    fn test_while_loop_with_body() {
        // While loop pattern:
        //   0 (entry)
        //   |
        //   v
        //   1 (loop header) ←──┐
        //  / \                  |
        // 2   3 (exit)          |
        // |                     |
        // v                     |
        // 4 (loop body) ───────┘
        //
        // Edges: 0→1, 1→2, 1→3, 2→4, 4→1
        // Block 1 has predecessors [0, 4] → join point.
        // idom(1)=0, idom(2)=1, idom(3)=1, idom(4)=2
        //
        // For block 1 (join point, preds=[0, 4]):
        //   pred=0: runner=0, idom(1)=0, stop.
        //   pred=4: runner=4, DF(4)+={1}, runner=idom(4)=2, DF(2)+={1}, runner=idom(2)=1, DF(1)+={1}, runner=idom(1)=0, stop.
        //
        // Expected: DF(0)={}, DF(1)={1}, DF(2)={1}, DF(3)={}, DF(4)={1}
        let func = build_cfg(5, &[(0, 1), (1, 2), (1, 3), (2, 4), (4, 1)]);
        let dom_tree = DominatorTree::build(&func);
        let df = compute_dominance_frontiers(&func, &dom_tree);

        assert!(df.frontier(0).is_empty());
        assert!(df.contains(1, 1), "DF(1) should contain 1");
        assert!(df.contains(2, 1), "DF(2) should contain 1");
        assert!(df.frontier(3).is_empty(), "DF(3) should be empty");
        assert!(df.contains(4, 1), "DF(4) should contain 1");
    }
}
