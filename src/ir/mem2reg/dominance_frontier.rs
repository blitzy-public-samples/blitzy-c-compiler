//! Dominance frontier computation for phi-node placement during SSA construction.
//!
//! This module implements the dominance frontier (DF) computation algorithm
//! from Cytron et al. 1991 and the iterated dominance frontier (IDF)
//! algorithm for determining transitive phi-node placement locations.
//!
//! # Dominance Frontier Definition
//!
//! The dominance frontier of a block X is the set of all blocks Y where
//! X's dominance "ends":
//!
//! ```text
//! DF(X) = { Y | ∃ predecessor Z of Y such that X dominates Z
//!               but X does NOT strictly dominate Y }
//! ```
//!
//! Intuitively, values defined in X reach blocks in DF(X), but other
//! definitions can also reach those blocks through alternate CFG paths.
//! This is exactly where phi nodes must be inserted to merge the
//! different reaching definitions into a single SSA value.
//!
//! # Walk-Up Algorithm
//!
//! The computation uses the efficient walk-up formulation:
//!
//! ```text
//! for each block B in the CFG:
//!     for each predecessor P of B:
//!         runner = P
//!         while runner ≠ idom(B):
//!             DF(runner) ∪= { B }
//!             runner = idom(runner)
//! ```
//!
//! Each walk starts at a predecessor of B and ascends the dominator tree
//! until reaching B's immediate dominator. Every node visited along the
//! walk has B in its dominance frontier — it dominates the predecessor
//! (by transitivity through the dominator tree) but does not strictly
//! dominate B (since idom(B) is the first strict dominator).
//!
//! # Iterated Dominance Frontier (IDF)
//!
//! A phi node at block Y introduces a new definition of a variable.
//! This new definition may itself require additional phi nodes at Y's
//! own dominance frontiers. The iterated dominance frontier computes
//! the transitive closure of this process using a worklist algorithm,
//! yielding the complete set of phi-node placement locations.
//!
//! # Complexity
//!
//! - **DF computation:** O(|E| · depth(domtree)) where |E| is the number
//!   of CFG edges. Nearly linear for structured control flow (shallow trees).
//! - **IDF computation:** O(|DF_total|) where |DF_total| is the sum of
//!   all dominance frontier sizes traversed.
//!
//! # References
//!
//! R. Cytron, J. Ferrante, B. K. Rosen, M. N. Wegman, and F. K. Zadeck,
//! "Efficiently Computing Static Single Assignment Form and the Control
//! Dependence Graph," *ACM TOPLAS*, 13(4):451–490, 1991.

use crate::common::fx_hash::{fx_hash_map, fx_hash_set, FxHashMap, FxHashSet};
use crate::ir::basic_block::{BasicBlock, BasicBlockId};
use crate::ir::function::IrFunction;
use crate::ir::mem2reg::dominator_tree::DominatorTree;

// ====================================================================
// DominanceFrontier — public struct
// ====================================================================

/// Dominance frontier sets for all basic blocks in a function's CFG.
///
/// Each block X has an associated frontier set DF(X) containing the
/// blocks where X's dominance ends. These sets drive phi-node placement
/// during SSA construction (Phase 7 — mem2reg).
///
/// # Construction
///
/// Use [`DominanceFrontier::compute()`] to build frontier sets from an
/// [`IrFunction`] and its pre-computed [`DominatorTree`].
///
/// # Query Methods
///
/// | Method | Description | Complexity |
/// |--------|-------------|------------|
/// | [`frontier_of()`](Self::frontier_of) | DF(block) | O(1) |
/// | [`is_in_frontier()`](Self::is_in_frontier) | Membership test | O(1) expected |
/// | [`verify()`](Self::verify) | Correctness check | O(V²·E) |
///
/// # Usage
///
/// ```ignore
/// let dom_tree = DominatorTree::compute(&func);
/// let df = DominanceFrontier::compute(&func, &dom_tree);
///
/// // Query individual block frontiers
/// let frontier = df.frontier_of(block_id);
///
/// // Compute phi placements for a variable's definition blocks
/// let phi_blocks = compute_iterated_frontier(&df, &def_blocks);
/// ```
pub struct DominanceFrontier {
    /// Dominance frontier set for each block, indexed by `block.id.0`.
    /// `frontiers[b.0]` contains all blocks Y in DF(b).
    frontiers: Vec<FxHashSet<BasicBlockId>>,

    /// Number of index slots in the frontiers vector. Equal to
    /// `max_block_id + 1`, ensuring all block IDs are valid indices.
    num_slots: usize,

    /// Pre-allocated empty set returned by [`frontier_of()`] for
    /// out-of-range block IDs. Avoids the need for Option returns
    /// or panics on invalid lookups.
    empty: FxHashSet<BasicBlockId>,
}

// ====================================================================
// DominanceFrontier — construction
// ====================================================================

impl DominanceFrontier {
    /// Computes the dominance frontier for every basic block in `func`
    /// using the pre-computed dominator tree.
    ///
    /// Implements the walk-up formulation of the Cytron et al. 1991 DF
    /// algorithm: for each block B in the CFG, for each predecessor P
    /// of B, walk up the dominator tree from P until reaching idom(B),
    /// adding B to the frontier of each visited node.
    ///
    /// # Arguments
    ///
    /// * `func`     — The IR function whose CFG to analyse. Provides
    ///   access to all basic blocks and their predecessor/successor
    ///   edges via [`IrFunction::blocks()`] and [`IrFunction::get_block()`].
    /// * `dom_tree` — Pre-computed dominator tree for `func`, providing
    ///   [`DominatorTree::idom()`] for the walk-up termination condition.
    ///
    /// # Returns
    ///
    /// A `DominanceFrontier` with frontier sets populated for all
    /// reachable blocks. Unreachable blocks have empty frontier sets.
    ///
    /// # Complexity
    ///
    /// O(|E| · depth(domtree)) where |E| is the number of CFG edges.
    /// In practice this is nearly linear for structured control flow
    /// graphs with shallow dominator trees (e.g., most C functions).
    ///
    /// # Panics
    ///
    /// Panics if `func.get_block()` is called with a block ID that does
    /// not exist in the function (indicates a malformed CFG with edges
    /// to non-existent blocks).
    pub fn compute(func: &IrFunction, dom_tree: &DominatorTree) -> DominanceFrontier {
        let blocks = func.blocks();
        let total_blocks = func.block_count();

        // Handle empty functions — defensive guard for degenerate input.
        if blocks.is_empty() {
            return DominanceFrontier {
                frontiers: Vec::new(),
                num_slots: 0,
                empty: fx_hash_set(),
            };
        }

        // Record the entry block ID — used for reachability checks below.
        // A block is reachable iff the entry block dominates it; this
        // avoids introducing spurious frontier entries for unreachable
        // predecessors that are not part of the dominator tree.
        let entry_block = func.entry_block();
        let entry_id = entry_block.id;

        // Determine array sizing from the maximum block ID present in
        // the function. Block IDs may not be contiguous (blocks can be
        // removed by optimization passes), so we size based on the
        // maximum rather than the count.
        let max_id = blocks.iter().map(|b| b.id.0).max().unwrap_or(0) as usize;
        let num_slots = max_id + 1;

        // Pre-allocate frontier sets for each block slot. The number of
        // slots is derived from max_block_id; total_blocks provides a
        // useful sanity bound (there are never more blocks than slots).
        debug_assert!(
            total_blocks <= num_slots + 1,
            "block_count ({}) exceeds slot capacity ({})",
            total_blocks,
            num_slots,
        );
        let mut frontiers: Vec<FxHashSet<BasicBlockId>> =
            (0..num_slots).map(|_| fx_hash_set()).collect();

        // ---- Walk-Up DF Algorithm ------------------------------------------
        //
        // For each block B, for each predecessor P of B:
        //   Start at runner = P.
        //   While runner ≠ idom(B):
        //     Add B to DF(runner).
        //     runner = idom(runner).
        //
        // Termination:
        //   - Normal: runner reaches idom(B) (which strictly dominates B).
        //   - Entry:  B is the entry block (idom(B) = None). The walk
        //     continues until runner's own idom is None (runner = root),
        //     at which point we stop after adding B to DF(root).
        //   - Unreachable: runner's idom is None because it is unreachable.
        //     We break to avoid infinite loops.
        //
        // The comparison `Some(runner) == idom_b` handles both cases
        // uniformly: when idom_b is Some(x), we stop at x; when idom_b
        // is None, we stop when runner's idom becomes None (walked past
        // the root).

        for block in blocks {
            let b = block.id;

            // Skip blocks with no predecessors — the walk-up has no
            // starting points and cannot contribute to any frontier.
            let preds = block.predecessors();
            if preds.is_empty() {
                continue;
            }

            // Cache idom(B) for all predecessor walk-ups of this block.
            let idom_b: Option<BasicBlockId> = dom_tree.idom(b);

            for &pred in preds {
                // Skip unreachable predecessors — they are not part of
                // the dominator tree and would produce spurious frontier
                // entries that fail verification. A block is reachable
                // iff the entry block dominates it (dominates returns
                // true for self, false for unreachable blocks).
                if !dom_tree.dominates(entry_id, pred) {
                    continue;
                }

                // Start the walk-up from this predecessor.
                let mut runner = pred;

                loop {
                    // Termination: if runner equals idom(B), the walk
                    // has reached the immediate dominator of B. Since
                    // idom(B) strictly dominates B, B is NOT in the
                    // frontier of idom(B) via this particular walk.
                    if Some(runner) == idom_b {
                        break;
                    }

                    // Add B to the dominance frontier of runner.
                    // Runner dominates pred (or IS pred), and pred is a
                    // predecessor of B, so runner dominates a predecessor
                    // of B. Since runner ≠ idom(B), runner does not
                    // strictly dominate B. Hence B ∈ DF(runner).
                    let runner_idx = runner.0 as usize;
                    if runner_idx < num_slots {
                        frontiers[runner_idx].insert(b);
                    }

                    // Ascend the dominator tree.
                    match dom_tree.idom(runner) {
                        Some(parent) => runner = parent,
                        // Runner is the root (entry block) or unreachable.
                        // The walk terminates here.
                        None => break,
                    }
                }
            }
        }

        DominanceFrontier {
            frontiers,
            num_slots,
            empty: fx_hash_set(),
        }
    }
}

// ====================================================================
// DominanceFrontier — query methods
// ====================================================================

impl DominanceFrontier {
    /// Returns the dominance frontier set of `block`.
    ///
    /// DF(block) is the set of all blocks Y where `block`'s dominance
    /// ends — `block` dominates a predecessor of Y but does not strictly
    /// dominate Y.
    ///
    /// Returns an empty set if `block` has an out-of-range ID, was
    /// unreachable, or simply has no blocks in its frontier.
    ///
    /// # Complexity
    ///
    /// O(1) — direct array lookup.
    #[inline]
    pub fn frontier_of(&self, block: BasicBlockId) -> &FxHashSet<BasicBlockId> {
        let idx = block.0 as usize;
        if idx < self.num_slots {
            &self.frontiers[idx]
        } else {
            &self.empty
        }
    }

    /// Returns `true` if `frontier_member` is in the dominance frontier
    /// of `block`.
    ///
    /// Equivalent to `self.frontier_of(block).contains(&frontier_member)`
    /// but expressed as a single method call for readability at call sites.
    ///
    /// # Complexity
    ///
    /// O(1) expected — FxHashSet membership test.
    #[inline]
    pub fn is_in_frontier(
        &self,
        block: BasicBlockId,
        frontier_member: BasicBlockId,
    ) -> bool {
        self.frontier_of(block).contains(&frontier_member)
    }

    /// Verifies that the computed dominance frontiers satisfy the formal
    /// DF definition for every block in the function.
    ///
    /// Performs three independent verification passes:
    ///
    /// 1. **Soundness:** Every Y in DF(X) has a predecessor Z such that
    ///    X dominates Z but X does not strictly dominate Y.
    /// 2. **DF-local completeness:** For every block X and successor Y of X,
    ///    if idom(Y) ≠ X, then Y must be in DF(X).
    /// 3. **DF-up completeness:** For every block X and dominator tree
    ///    child C of X, every Y in DF(C) that is not strictly dominated
    ///    by X must also be in DF(X).
    ///
    /// # Arguments
    ///
    /// * `func`     — The IR function whose CFG was analysed.
    /// * `dom_tree` — The dominator tree used during frontier computation.
    ///
    /// # Returns
    ///
    /// `true` if all frontiers are correct, `false` if any inconsistency
    /// is detected.
    ///
    /// # Complexity
    ///
    /// O(|V|² · |E|) — verification is expensive and intended for debug
    /// builds and testing only. Do not call in release-mode hot paths.
    pub fn verify(
        &self,
        func: &IrFunction,
        dom_tree: &DominatorTree,
    ) -> bool {
        let blocks = func.blocks();

        // ---- Pass 1: Soundness ----
        //
        // For each block X, for each Y in DF(X):
        //   There must exist a predecessor Z of Y such that
        //   X dominates Z AND X does NOT strictly dominate Y.
        for block in blocks {
            let x = block.id;
            for &y in self.frontier_of(x) {
                let y_block: &BasicBlock = func.get_block(y);
                let has_valid_pred = y_block.predecessors().iter().any(|&z| {
                    dom_tree.dominates(x, z) && !dom_tree.strictly_dominates(x, y)
                });
                if !has_valid_pred {
                    return false;
                }
            }
        }

        // ---- Pass 2: DF-local Completeness ----
        //
        // For each block X, for each successor S of X:
        //   If idom(S) ≠ X, then S must be in DF(X).
        //
        // This checks the "local" contribution: X is a predecessor of S
        // (since S is a successor of X), X dominates itself, and if
        // idom(S) ≠ X then X does not strictly dominate S, so S ∈ DF(X).
        for block in blocks {
            let x = block.id;
            for &succ in block.successors() {
                // Skip successors that are not reachable (their idom is
                // undefined and they are excluded from DF computation).
                if dom_tree.idom(succ).is_none() && succ != func.entry_block().id {
                    continue;
                }
                if dom_tree.idom(succ) != Some(x) {
                    if !self.is_in_frontier(x, succ) {
                        return false;
                    }
                }
            }
        }

        // ---- Pass 3: DF-up Completeness ----
        //
        // For each block X with dominator tree children:
        //   For each child C of X in the dominator tree:
        //     For each Y in DF(C):
        //       If X does NOT strictly dominate Y, then Y ∈ DF(X).
        //
        // This checks the "up" contribution: if a child C of X has Y in
        // its frontier, and X doesn't strictly dominate Y, then X also
        // has Y in its frontier (the dominance of X "ends" at Y too).
        for block in blocks {
            let x = block.id;
            for &child in dom_tree.children(x) {
                for &y in self.frontier_of(child) {
                    if !dom_tree.strictly_dominates(x, y) {
                        if !self.is_in_frontier(x, y) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Returns the number of blocks that have non-empty dominance frontiers.
    ///
    /// Useful for diagnostic output, performance metrics, and understanding
    /// the CFG complexity (more non-empty frontiers generally means more
    /// phi nodes will be needed during SSA construction).
    pub fn non_empty_count(&self) -> usize {
        self.frontiers.iter().filter(|s| !s.is_empty()).count()
    }

    /// Returns the total number of (block, frontier_member) pairs across
    /// all dominance frontier sets.
    ///
    /// This metric indicates the total "phi pressure" — larger values mean
    /// more potential phi-node insertion points and higher SSA construction
    /// cost. Useful for memory usage estimation and compilation time
    /// prediction.
    pub fn total_frontier_size(&self) -> usize {
        self.frontiers.iter().map(|s| s.len()).sum()
    }

    /// Returns a map representation of the non-empty dominance frontiers.
    ///
    /// Useful for serialization, debugging output, and interoperability
    /// with algorithms that expect map-based frontier representations.
    /// Only blocks with non-empty frontiers are included in the map.
    pub fn to_map(&self) -> FxHashMap<BasicBlockId, FxHashSet<BasicBlockId>> {
        let mut map: FxHashMap<BasicBlockId, FxHashSet<BasicBlockId>> = fx_hash_map();
        for (idx, frontier) in self.frontiers.iter().enumerate() {
            if !frontier.is_empty() {
                map.insert(BasicBlockId(idx as u32), frontier.clone());
            }
        }
        map
    }

    /// Returns `true` if all dominance frontier sets are empty.
    ///
    /// This occurs when the function has a trivial CFG (single block,
    /// linear chain, or no join points), meaning no phi nodes will be
    /// needed during SSA construction.
    pub fn is_empty(&self) -> bool {
        self.frontiers.iter().all(|s| s.is_empty())
    }
}

// ====================================================================
// Iterated Dominance Frontier (IDF) — free functions
// ====================================================================

/// Computes the iterated dominance frontier (IDF) for a set of
/// definition blocks.
///
/// The IDF is the transitive closure of the dominance frontier applied
/// to the given definition blocks. Starting from `def_blocks`, the
/// algorithm discovers all blocks where phi nodes must be inserted:
///
/// 1. A variable is defined in blocks `def_blocks`.
/// 2. Phi nodes are needed at each block in DF(def_blocks).
/// 3. Each phi node is itself a new definition, requiring additional
///    phi nodes at _its_ dominance frontiers.
/// 4. The process repeats until no new blocks are discovered.
///
/// # Algorithm
///
/// Uses a worklist-based approach:
///
/// ```text
/// worklist = def_blocks
/// idf = ∅
/// while worklist ≠ ∅:
///     B = worklist.pop()
///     for each Y in DF(B):
///         if Y ∉ idf:
///             idf ∪= { Y }
///             worklist.push(Y)
/// return idf
/// ```
///
/// # Arguments
///
/// * `df`         — Pre-computed dominance frontiers for the function.
/// * `def_blocks` — Set of blocks containing definitions of a variable
///   (typically blocks containing stores to an alloca).
///
/// # Returns
///
/// The set of blocks where phi nodes must be inserted for the variable.
/// This set may overlap with `def_blocks` (a definition block can also
/// be a phi-node location if it is in its own iterated frontier).
///
/// # Complexity
///
/// O(|IDF_traversal|) — each block is added to the worklist at most
/// once (tracked by the `in_worklist` set), and each frontier set is
/// iterated at most once. In practice this is very fast for typical
/// C function CFGs.
pub fn compute_iterated_frontier(
    df: &DominanceFrontier,
    def_blocks: &FxHashSet<BasicBlockId>,
) -> FxHashSet<BasicBlockId> {
    let mut idf_result: FxHashSet<BasicBlockId> = fx_hash_set();

    // Worklist of blocks whose dominance frontiers have not yet been
    // examined. Initialised with the original definition blocks.
    let mut worklist: Vec<BasicBlockId> = Vec::with_capacity(def_blocks.len());
    worklist.extend(def_blocks.iter().copied());

    // Track which blocks have already been added to the worklist to
    // prevent redundant processing. A block can appear in multiple
    // frontier sets but only needs to be processed once.
    let mut in_worklist: FxHashSet<BasicBlockId> = def_blocks.clone();

    while let Some(block) = worklist.pop() {
        for &frontier_block in df.frontier_of(block) {
            // If frontier_block is new to the IDF result, add it and
            // schedule it for processing (its own frontiers may
            // contribute further phi-node locations).
            if idf_result.insert(frontier_block) {
                // Only add to worklist if not previously scheduled.
                if in_worklist.insert(frontier_block) {
                    worklist.push(frontier_block);
                }
            }
        }
    }

    idf_result
}

/// Computes phi-node placement locations for each alloca slot.
///
/// For each alloca variable (identified by its index in the
/// `alloca_def_blocks` slice), this function computes the iterated
/// dominance frontier of that variable's definition blocks. The result
/// is a parallel vector telling the SSA builder exactly where to insert
/// phi nodes for each promoted alloca.
///
/// # Arguments
///
/// * `df`                — Pre-computed dominance frontiers.
/// * `alloca_def_blocks` — For each alloca slot index `i`, the set of
///   blocks that contain stores to (definitions of) alloca `i`.
///
/// # Returns
///
/// A `Vec` parallel to `alloca_def_blocks` where entry `i` contains the
/// set of blocks requiring phi nodes for alloca slot `i`.
///
/// # Example
///
/// ```ignore
/// // During mem2reg Phase 7:
/// let mut def_blocks = vec![fx_hash_set(); alloca_count];
/// // ... scan IR to populate def_blocks[i] with blocks containing
/// //     stores to alloca i ...
///
/// let phi_placements = compute_phi_placements(&df, &def_blocks);
///
/// // phi_placements[i] contains blocks where phi nodes for alloca i
/// // must be inserted before SSA renaming.
/// for (alloca_idx, phi_blocks) in phi_placements.iter().enumerate() {
///     for &block_id in phi_blocks {
///         insert_phi_node(block_id, alloca_idx);
///     }
/// }
/// ```
///
/// # Complexity
///
/// O(sum of |IDF(def_blocks[i])| for all i). Each alloca is processed
/// independently, so the total cost is proportional to the sum of all
/// iterated frontier sizes.
pub fn compute_phi_placements(
    df: &DominanceFrontier,
    alloca_def_blocks: &[FxHashSet<BasicBlockId>],
) -> Vec<FxHashSet<BasicBlockId>> {
    alloca_def_blocks
        .iter()
        .map(|def_blocks| compute_iterated_frontier(df, def_blocks))
        .collect()
}

// ====================================================================
// Display implementation for diagnostic output
// ====================================================================

impl std::fmt::Debug for DominanceFrontier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DominanceFrontier {{")?;
        for (idx, frontier) in self.frontiers.iter().enumerate() {
            if !frontier.is_empty() {
                let mut members: Vec<u32> = frontier.iter().map(|b| b.0).collect();
                members.sort_unstable();
                write!(f, "  DF(bb{}) = {{ ", idx)?;
                for (i, m) in members.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "bb{}", m)?;
                }
                writeln!(f, " }}")?;
            }
        }
        writeln!(f, "}}")
    }
}

impl std::fmt::Display for DominanceFrontier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let non_empty = self.non_empty_count();
        let total = self.total_frontier_size();
        write!(
            f,
            "DominanceFrontier({} blocks with non-empty DF, {} total entries)",
            non_empty, total,
        )
    }
}

// ====================================================================
// Unit Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::{BasicBlock, BasicBlockId};
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::Instruction;
    use crate::ir::mem2reg::dominator_tree::DominatorTree;
    use crate::ir::types::IrType;

    /// Helper: build a function with the given number of blocks and
    /// edges, compute its dominator tree and dominance frontiers.
    fn build_diamond_cfg() -> (IrFunction, DominatorTree, DominanceFrontier) {
        // Diamond CFG:
        //       bb0 (entry)
        //      /   \
        //    bb1   bb2
        //      \   /
        //       bb3
        let mut func = IrFunction::new("diamond".into(), IrType::Void, vec![]);

        let bb1 = BasicBlock::new(BasicBlockId(1), Some("then".into()));
        let bb2 = BasicBlock::new(BasicBlockId(2), Some("else".into()));
        let bb3 = BasicBlock::new(BasicBlockId(3), Some("merge".into()));

        func.add_basic_block(bb1);
        func.add_basic_block(bb2);
        func.add_basic_block(bb3);

        // Add edges: bb0 -> bb1, bb0 -> bb2
        func.get_block_mut(BasicBlockId(0)).add_successor(BasicBlockId(1));
        func.get_block_mut(BasicBlockId(0)).add_successor(BasicBlockId(2));
        func.get_block_mut(BasicBlockId(1)).add_predecessor(BasicBlockId(0));
        func.get_block_mut(BasicBlockId(2)).add_predecessor(BasicBlockId(0));

        // Add edges: bb1 -> bb3, bb2 -> bb3
        func.get_block_mut(BasicBlockId(1)).add_successor(BasicBlockId(3));
        func.get_block_mut(BasicBlockId(2)).add_successor(BasicBlockId(3));
        func.get_block_mut(BasicBlockId(3)).add_predecessor(BasicBlockId(1));
        func.get_block_mut(BasicBlockId(3)).add_predecessor(BasicBlockId(2));

        // Add terminators to make the CFG well-formed.
        func.get_block_mut(BasicBlockId(0)).set_terminator(
            Instruction::CondBranch {
                condition: crate::ir::instructions::ValueId(0),
                true_block: BasicBlockId(1),
                false_block: BasicBlockId(2),
            },
        );
        func.get_block_mut(BasicBlockId(1)).set_terminator(
            Instruction::Branch {
                target: BasicBlockId(3),
            },
        );
        func.get_block_mut(BasicBlockId(2)).set_terminator(
            Instruction::Branch {
                target: BasicBlockId(3),
            },
        );
        func.get_block_mut(BasicBlockId(3)).set_terminator(
            Instruction::Return { value: None },
        );

        let dom_tree = DominatorTree::compute(&func);
        let df = DominanceFrontier::compute(&func, &dom_tree);

        (func, dom_tree, df)
    }

    #[test]
    fn test_diamond_df() {
        let (func, dom_tree, df) = build_diamond_cfg();

        // In a diamond:
        //   DF(bb0) = {} (entry dominates everything)
        //   DF(bb1) = {bb3} (bb1 dominates itself, predecessor of bb3,
        //                     but does not strictly dominate bb3)
        //   DF(bb2) = {bb3} (same reasoning)
        //   DF(bb3) = {} (no successors, no frontiers)
        assert!(df.frontier_of(BasicBlockId(0)).is_empty());
        assert!(df.is_in_frontier(BasicBlockId(1), BasicBlockId(3)));
        assert!(df.is_in_frontier(BasicBlockId(2), BasicBlockId(3)));
        assert!(df.frontier_of(BasicBlockId(3)).is_empty());

        // Verify correctness.
        assert!(df.verify(&func, &dom_tree));
    }

    #[test]
    fn test_single_block_df() {
        // Single-block function: no edges, empty frontiers.
        let func = IrFunction::new("trivial".into(), IrType::Void, vec![]);

        // Add a return terminator to the entry block.
        func.blocks()[0].clone(); // verify block exists
        let mut func = func;
        func.get_block_mut(BasicBlockId(0)).set_terminator(
            Instruction::Return { value: None },
        );

        let dom_tree = DominatorTree::compute(&func);
        let df = DominanceFrontier::compute(&func, &dom_tree);

        assert!(df.is_empty());
        assert!(df.verify(&func, &dom_tree));
    }

    #[test]
    fn test_iterated_frontier_diamond() {
        let (_func, _dom_tree, df) = build_diamond_cfg();

        // Variable defined only in bb1:
        // IDF({bb1}) = DF(bb1) = {bb3}
        let mut def_blocks: FxHashSet<BasicBlockId> = fx_hash_set();
        def_blocks.insert(BasicBlockId(1));

        let idf = compute_iterated_frontier(&df, &def_blocks);
        assert!(idf.contains(&BasicBlockId(3)));
        assert_eq!(idf.len(), 1);
    }

    #[test]
    fn test_phi_placements() {
        let (_func, _dom_tree, df) = build_diamond_cfg();

        // Two allocas:
        //   alloca 0: defined in bb1
        //   alloca 1: defined in bb1 and bb2
        let mut defs_0: FxHashSet<BasicBlockId> = fx_hash_set();
        defs_0.insert(BasicBlockId(1));

        let mut defs_1: FxHashSet<BasicBlockId> = fx_hash_set();
        defs_1.insert(BasicBlockId(1));
        defs_1.insert(BasicBlockId(2));

        let alloca_defs = vec![defs_0, defs_1];
        let placements = compute_phi_placements(&df, &alloca_defs);

        assert_eq!(placements.len(), 2);
        // alloca 0: phi at bb3
        assert!(placements[0].contains(&BasicBlockId(3)));
        // alloca 1: phi at bb3
        assert!(placements[1].contains(&BasicBlockId(3)));
    }

    #[test]
    fn test_to_map() {
        let (_func, _dom_tree, df) = build_diamond_cfg();

        let map = df.to_map();
        // bb1 and bb2 should have non-empty frontiers.
        assert!(map.contains_key(&BasicBlockId(1)));
        assert!(map.contains_key(&BasicBlockId(2)));
        // bb0 and bb3 should not be in the map (empty frontiers).
        assert!(!map.contains_key(&BasicBlockId(0)));
        assert!(!map.contains_key(&BasicBlockId(3)));
    }

    #[test]
    fn test_empty_def_blocks_idf() {
        let (_func, _dom_tree, df) = build_diamond_cfg();

        // No definition blocks → empty IDF.
        let empty_defs: FxHashSet<BasicBlockId> = fx_hash_set();
        let idf = compute_iterated_frontier(&df, &empty_defs);
        assert!(idf.is_empty());
    }

    #[test]
    fn test_display_and_debug() {
        let (_func, _dom_tree, df) = build_diamond_cfg();

        // Ensure Display and Debug don't panic.
        let display = format!("{}", df);
        assert!(display.contains("DominanceFrontier"));

        let debug = format!("{:?}", df);
        assert!(debug.contains("DF("));
    }
}
