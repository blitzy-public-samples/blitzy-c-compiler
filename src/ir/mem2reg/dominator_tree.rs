//! Dominator tree computation using the Lengauer-Tarjan algorithm.
//!
//! This module computes the dominator tree for a function's control flow
//! graph. The dominator tree is the foundational data structure for SSA
//! construction — it determines both where phi nodes must be placed
//! (via dominance frontiers computed in [`crate::ir::mem2reg::dominance_frontier`])
//! and the order in which variables are renamed (via dominator tree pre-order
//! traversal in [`crate::ir::mem2reg::ssa_builder`]).
//!
//! # Algorithm
//!
//! The implementation uses the Lengauer-Tarjan algorithm (1979) with path
//! compression in the EVAL/LINK operations, achieving near-linear
//! amortised time complexity:
//!
//! 1. **DFS numbering (Phase 1):** Assign depth-first search numbers to all
//!    basic blocks reachable from the entry block. Unreachable blocks are
//!    excluded from dominator computation entirely.
//! 2. **Semidominator computation (Phase 2):** For each block in reverse DFS
//!    order, compute its semidominator using the EVAL operation on a disjoint
//!    set forest with path compression.
//! 3. **Implicit idom computation (Phase 3):** Determine a candidate immediate
//!    dominator for each block using the semidominator theorem and bucket
//!    processing.
//! 4. **Idom finalisation (Phase 4):** Adjust candidate immediate dominators
//!    by walking in DFS order to produce the final dominator tree.
//!
//! # Performance
//!
//! Time complexity: **O(n · α(n))** where *n* is the number of reachable
//! basic blocks and *α* is the inverse Ackermann function (practically ≤ 4
//! for all inputs encountered in real programs). The implementation handles
//! functions with 10 000+ blocks efficiently, as required for Linux kernel
//! compilation.
//!
//! # Edge Cases
//!
//! - **Unreachable blocks** are not visited during Phase 1 DFS and are
//!   excluded from dominator computation. Their `idom` is `None`.
//! - **Single-block functions** produce a trivial tree where the entry
//!   block dominates itself and has no immediate dominator.
//! - **Self-loops** do not affect dominator computation — a block's
//!   self-loop edge does not change its dominator.
//!
//! # References
//!
//! T. Lengauer and R. E. Tarjan, "A Fast Algorithm for Finding Dominators
//! in a Flowgraph," *ACM TOPLAS*, 1(1):121–141, 1979.

use crate::common::fx_hash::{FxHashMap, FxHashSet, fx_hash_map, fx_hash_set};
use crate::ir::basic_block::{BasicBlock, BasicBlockId};
use crate::ir::function::IrFunction;

/// Sentinel value representing "undefined" / "no value" in algorithm arrays
/// indexed by DFS number. Chosen as `u32::MAX` so it is never a valid DFS
/// number (valid DFS numbers are 0..n-1 where n ≤ number of blocks).
const UNDEF: u32 = u32::MAX;

// ====================================================================
// DominatorTree — public API
// ====================================================================

/// Dominator tree for a function's control flow graph.
///
/// The dominator tree captures the dominance relationship between basic
/// blocks: block A *dominates* block B if every path from the function
/// entry to B must pass through A. The *immediate dominator* of B
/// (idom(B)) is the unique block that strictly dominates B and is
/// dominated by all other strict dominators of B.
///
/// # Construction
///
/// Use [`DominatorTree::compute()`] to build the tree from an
/// [`IrFunction`]. The computation uses the Lengauer-Tarjan algorithm
/// with O(n · α(n)) amortised time.
///
/// # Query Methods
///
/// | Method | Description | Complexity |
/// |--------|-------------|------------|
/// | [`idom()`](Self::idom) | Immediate dominator | O(1) |
/// | [`dominates()`](Self::dominates) | Does A dominate B? | O(1) |
/// | [`strictly_dominates()`](Self::strictly_dominates) | Strict dominance | O(1) |
/// | [`children()`](Self::children) | Dominator tree children | O(1) |
/// | [`depth()`](Self::depth) | Depth in tree | O(1) |
/// | [`preorder()`](Self::preorder) | Pre-order traversal | O(1) |
/// | [`lca()`](Self::lca) | Lowest common ancestor | O(depth) |
pub struct DominatorTree {
    /// Immediate dominator for each block, indexed by `block.id.0`.
    /// The entry block and unreachable blocks have `None`.
    idom: Vec<Option<BasicBlockId>>,

    /// Dominator tree children for each block, indexed by `block.id.0`.
    /// `children[b.0]` lists blocks whose immediate dominator is `b`.
    children: Vec<Vec<BasicBlockId>>,

    /// Depth of each block in the dominator tree, indexed by `block.id.0`.
    /// Entry block (root) has depth 0. Unreachable blocks have depth 0.
    depth: Vec<u32>,

    /// Blocks in dominator tree pre-order traversal. The entry block is
    /// always first. Used by SSA renaming in `ssa_builder.rs`.
    preorder: Vec<BasicBlockId>,

    /// Blocks in CFG reverse postorder. Used by iterative dataflow
    /// algorithms and dominance frontier computation.
    postorder: Vec<BasicBlockId>,

    /// DFS pre-order timestamp on the dominator tree, indexed by
    /// `block.id.0`. Together with `post_time`, enables O(1) dominance
    /// checks via the Euler-tour property:
    ///   `a dominates b ⟺ pre_time[a] ≤ pre_time[b] ∧ post_time[a] ≥ post_time[b]`
    pre_time: Vec<u32>,

    /// DFS post-order timestamp on the dominator tree, indexed by
    /// `block.id.0`. See `pre_time` for usage.
    post_time: Vec<u32>,

    /// The entry block ID — root of the dominator tree.
    entry: BasicBlockId,

    /// Number of index slots in the per-block vectors. Equal to
    /// `max_block_id + 1`, ensuring all block IDs are valid indices.
    num_slots: usize,

    /// Set of block IDs reachable from the entry block. Unreachable
    /// blocks are excluded from dominator computation.
    reachable: FxHashSet<BasicBlockId>,
}

// ====================================================================
// DominatorTree — construction
// ====================================================================

impl DominatorTree {
    /// Compute the dominator tree for `func` using the Lengauer-Tarjan
    /// algorithm.
    ///
    /// # Arguments
    ///
    /// * `func` — The IR function whose CFG to analyse. Must have at
    ///   least one basic block (the entry block).
    ///
    /// # Returns
    ///
    /// A fully constructed `DominatorTree` ready for O(1) dominance
    /// queries and pre-order traversal.
    ///
    /// # Panics
    ///
    /// Panics if `func.blocks()` is non-empty but `func.entry_block()`
    /// references a non-existent block (malformed function).
    pub fn compute(func: &IrFunction) -> DominatorTree {
        let blocks = func.blocks();
        let total_blocks = func.block_count();

        // Handle empty functions (defensive; should not occur in practice).
        if blocks.is_empty() {
            return Self::empty();
        }

        let entry_block: &BasicBlock = func.entry_block();
        let entry_id = entry_block.id;

        // Determine array sizing from the maximum block ID.
        let max_id = blocks.iter().map(|b| b.id.0).max().unwrap_or(0) as usize;
        let num_slots = max_id + 1;

        // ---- Phase 1: DFS Numbering ----------------------------------------
        //
        // Assign DFS numbers to all blocks reachable from the entry block.
        // `block_to_dfs` maps BasicBlockId → DFS number.
        // `vertex[dfs_num]` maps DFS number → BasicBlockId.
        // `parent_dfs[dfs_num]` gives the DFS tree parent's DFS number.
        let mut block_to_dfs: FxHashMap<BasicBlockId, u32> = fx_hash_map();
        let mut vertex: Vec<BasicBlockId> = Vec::with_capacity(total_blocks);
        let mut parent_dfs: Vec<u32> = Vec::with_capacity(total_blocks);
        let mut reachable: FxHashSet<BasicBlockId> = fx_hash_set();

        let mut dfs_count: u32 = 0;
        // Stack entries: (block to visit, DFS number of DFS-tree parent)
        let mut dfs_stack: Vec<(BasicBlockId, u32)> = vec![(entry_id, UNDEF)];

        while let Some((block_id, par)) = dfs_stack.pop() {
            if reachable.contains(&block_id) {
                continue; // Already visited.
            }
            reachable.insert(block_id);

            let current_dfs = dfs_count;
            block_to_dfs.insert(block_id, current_dfs);
            vertex.push(block_id);
            parent_dfs.push(par);
            dfs_count += 1;

            // Push successors in reverse order so that the first successor
            // in the original list is popped (and thus numbered) first.
            let block = func.get_block(block_id);
            let succs = block.successors();
            for &succ in succs.iter().rev() {
                if !reachable.contains(&succ) {
                    dfs_stack.push((succ, current_dfs));
                }
            }
        }

        let n = dfs_count as usize; // Number of reachable blocks.

        // Trivial cases: 0 or 1 reachable blocks.
        if n <= 1 {
            return Self::trivial(entry_id, num_slots, n, reachable);
        }

        // ---- Phase 2 & 3: Semidominators and Immediate Dominators ----------
        //
        // All arrays in this section are indexed by DFS number (0..n).
        //
        // `semi[w]`:    semidominator DFS number for node w.
        //               Initialised to w itself (identity).
        // `idom_arr[w]`:candidate immediate dominator DFS number for w.
        // `bucket[v]`:  set of nodes whose semidominator is vertex[v].

        let mut semi: Vec<u32> = (0..n as u32).collect();
        let mut idom_arr: Vec<u32> = vec![UNDEF; n];
        let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); n];

        // Disjoint set forest for LINK/EVAL with path compression.
        let mut forest = LinkEvalForest::new(n);

        // Process nodes in reverse DFS order (from n-1 down to 1).
        // Node 0 is the entry block and is excluded.
        for w in (1..n).rev() {
            let w_block_id = vertex[w];
            let block = func.get_block(w_block_id);

            // --- Phase 2: Compute semidominator of w ---
            // semi(w) = min over all CFG predecessors v of w of:
            //   semi(eval(v))   if v was visited before w in DFS
            //   v               if v was visited after  w in DFS (back/cross edge)
            // This is captured by: semi(w) = min(semi(w), semi(eval(dfnum(v))))
            for &pred_id in block.predecessors() {
                if let Some(&v_dfs) = block_to_dfs.get(&pred_id) {
                    let u = forest.eval(v_dfs as usize, &semi);
                    if semi[u] < semi[w] {
                        semi[w] = semi[u];
                    }
                }
                // Predecessors not in block_to_dfs are unreachable; skip.
            }

            // Add w to the bucket of its semidominator.
            bucket[semi[w] as usize].push(w as u32);

            // LINK w to its DFS tree parent.
            let p = parent_dfs[w] as usize;
            forest.link(p, w);

            // --- Phase 3: Process bucket of parent(w) ---
            // For each v in bucket[parent(w)]:
            //   u = eval(v)
            //   if semi[u] < semi[v]:  idom[v] = u  (implicit)
            //   else:                  idom[v] = parent(w)
            let p_bucket = std::mem::take(&mut bucket[p]);
            for &v_u32 in &p_bucket {
                let v = v_u32 as usize;
                let u = forest.eval(v, &semi);
                idom_arr[v] = if semi[u] < semi[v] {
                    u as u32
                } else {
                    p as u32
                };
            }
            // bucket[p] is now empty (we took it with std::mem::take).
        }

        // ---- Phase 4: Finalise Immediate Dominators ------------------------
        //
        // For each node w (in DFS order), if idom[w] is not already the
        // semidominator of w, follow the idom chain to get the true idom.
        for w in 1..n {
            if idom_arr[w] != semi[w] {
                idom_arr[w] = idom_arr[idom_arr[w] as usize];
            }
        }

        // ---- Convert DFS-indexed results to block-ID-indexed results -------

        let mut idom_result: Vec<Option<BasicBlockId>> = vec![None; num_slots];
        for w in 1..n {
            let block_id = vertex[w];
            let idom_block = vertex[idom_arr[w] as usize];
            idom_result[block_id.0 as usize] = Some(idom_block);
        }
        // Entry block idom remains None.

        // Build dominator tree children lists.
        let mut children_result: Vec<Vec<BasicBlockId>> = vec![Vec::new(); num_slots];
        for block_id in vertex.iter().take(n).skip(1) {
            if let Some(idom_id) = idom_result[block_id.0 as usize] {
                children_result[idom_id.0 as usize].push(*block_id);
            }
        }

        // Compute depth of each block in the dominator tree via BFS.
        let mut depth_result = vec![0u32; num_slots];
        {
            let mut bfs_queue = std::collections::VecDeque::new();
            bfs_queue.push_back(entry_id);
            while let Some(bid) = bfs_queue.pop_front() {
                let d = depth_result[bid.0 as usize];
                for &child in &children_result[bid.0 as usize] {
                    depth_result[child.0 as usize] = d + 1;
                    bfs_queue.push_back(child);
                }
            }
        }

        // Compute dominator tree pre-order traversal.
        let preorder = compute_dom_tree_preorder(entry_id, &children_result);

        // Compute CFG reverse postorder.
        let postorder =
            compute_cfg_reverse_postorder(func, &block_to_dfs, num_slots);

        // Compute DFS timestamps on the dominator tree for O(1) dominance.
        let (pre_time, post_time) =
            compute_dom_tree_timestamps(entry_id, &children_result, num_slots);

        DominatorTree {
            idom: idom_result,
            children: children_result,
            depth: depth_result,
            preorder,
            postorder,
            pre_time,
            post_time,
            entry: entry_id,
            num_slots,
            reachable,
        }
    }

    // ---- Private helpers for trivial / empty cases ----

    /// Construct an empty dominator tree for a function with no blocks.
    fn empty() -> Self {
        DominatorTree {
            idom: Vec::new(),
            children: Vec::new(),
            depth: Vec::new(),
            preorder: Vec::new(),
            postorder: Vec::new(),
            pre_time: Vec::new(),
            post_time: Vec::new(),
            entry: BasicBlockId(0),
            num_slots: 0,
            reachable: fx_hash_set(),
        }
    }

    /// Construct a trivial dominator tree for a function with 0 or 1
    /// reachable blocks.
    fn trivial(
        entry_id: BasicBlockId,
        num_slots: usize,
        n: usize,
        reachable: FxHashSet<BasicBlockId>,
    ) -> Self {
        let idom = vec![None; num_slots];
        let children: Vec<Vec<BasicBlockId>> = vec![Vec::new(); num_slots];
        let depth = vec![0u32; num_slots];
        let mut pre_time = vec![0u32; num_slots];
        let mut post_time = vec![0u32; num_slots];

        let (preorder, postorder) = if n == 1 {
            // Single reachable block: it dominates itself.
            pre_time[entry_id.0 as usize] = 0;
            post_time[entry_id.0 as usize] = 1;
            (vec![entry_id], vec![entry_id])
        } else {
            (Vec::new(), Vec::new())
        };

        DominatorTree {
            idom,
            children,
            depth,
            preorder,
            postorder,
            pre_time,
            post_time,
            entry: entry_id,
            num_slots,
            reachable,
        }
    }
}

// ====================================================================
// DominatorTree — query methods
// ====================================================================

impl DominatorTree {
    /// Returns the immediate dominator of `block`, or `None` if `block`
    /// is the entry block (which has no dominator) or is unreachable
    /// from the entry block.
    ///
    /// # Complexity
    ///
    /// O(1) — direct array lookup.
    #[inline]
    pub fn idom(&self, block: BasicBlockId) -> Option<BasicBlockId> {
        let idx = block.0 as usize;
        if idx >= self.num_slots {
            return None;
        }
        self.idom[idx]
    }

    /// Returns `true` if `a` dominates `b`.
    ///
    /// Every block dominates itself. The check uses DFS timestamps on
    /// the dominator tree (Euler-tour property) for O(1) time:
    ///
    /// ```text
    /// a dominates b ⟺ pre_time[a] ≤ pre_time[b] ∧ post_time[a] ≥ post_time[b]
    /// ```
    ///
    /// If either block is unreachable, returns `false` (unless `a == b`
    /// and both are unreachable, which still returns `true`).
    ///
    /// # Complexity
    ///
    /// O(1).
    #[inline]
    pub fn dominates(&self, a: BasicBlockId, b: BasicBlockId) -> bool {
        if a == b {
            return true;
        }
        let ai = a.0 as usize;
        let bi = b.0 as usize;
        if ai >= self.num_slots || bi >= self.num_slots {
            return false;
        }
        if !self.reachable.contains(&a) || !self.reachable.contains(&b) {
            return false;
        }
        self.pre_time[ai] <= self.pre_time[bi]
            && self.post_time[ai] >= self.post_time[bi]
    }

    /// Returns `true` if `a` *strictly* dominates `b`, i.e.
    /// `a ≠ b` **and** `a` dominates `b`.
    ///
    /// # Complexity
    ///
    /// O(1).
    #[inline]
    pub fn strictly_dominates(&self, a: BasicBlockId, b: BasicBlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Returns the dominator tree children of `block` — the blocks
    /// whose immediate dominator is `block`.
    ///
    /// Returns an empty slice if `block` has no children, is
    /// unreachable, or has an out-of-range ID.
    ///
    /// # Complexity
    ///
    /// O(1) — returns a slice reference.
    #[inline]
    pub fn children(&self, block: BasicBlockId) -> &[BasicBlockId] {
        let idx = block.0 as usize;
        if idx >= self.num_slots {
            return &[];
        }
        &self.children[idx]
    }

    /// Returns the depth of `block` in the dominator tree.
    ///
    /// The entry block (root) has depth 0. Unreachable blocks have
    /// depth 0 (they are not part of the dominator tree).
    ///
    /// # Complexity
    ///
    /// O(1).
    #[inline]
    pub fn depth(&self, block: BasicBlockId) -> u32 {
        let idx = block.0 as usize;
        if idx >= self.num_slots {
            return 0;
        }
        self.depth[idx]
    }

    /// Returns blocks in dominator tree pre-order. The entry block is
    /// always the first element.
    ///
    /// This ordering is used by SSA renaming: processing blocks in
    /// dominator tree pre-order ensures that every block is visited
    /// after its immediate dominator, so definitions are available
    /// before their uses.
    ///
    /// # Complexity
    ///
    /// O(1) — returns a slice reference.
    #[inline]
    pub fn preorder(&self) -> &[BasicBlockId] {
        &self.preorder
    }

    /// Returns the lowest common ancestor of `a` and `b` in the
    /// dominator tree.
    ///
    /// The LCA of two blocks is the deepest block that dominates both.
    /// This is useful for finding merge points in the CFG.
    ///
    /// # Algorithm
    ///
    /// Walk both nodes upward to the same depth, then walk both up
    /// simultaneously until they meet. Worst-case O(depth).
    ///
    /// # Panics
    ///
    /// Does not panic — gracefully returns the entry block if either
    /// argument is unreachable or out of range.
    pub fn lca(&self, mut a: BasicBlockId, mut b: BasicBlockId) -> BasicBlockId {
        // Ensure both blocks are valid and reachable.
        let a_ok = (a.0 as usize) < self.num_slots && self.reachable.contains(&a);
        let b_ok = (b.0 as usize) < self.num_slots && self.reachable.contains(&b);
        if !a_ok || !b_ok {
            return self.entry;
        }

        // Level both nodes to the same depth.
        while self.depth(a) > self.depth(b) {
            match self.idom(a) {
                Some(parent) => a = parent,
                None => return a, // a is the root
            }
        }
        while self.depth(b) > self.depth(a) {
            match self.idom(b) {
                Some(parent) => b = parent,
                None => return b, // b is the root
            }
        }

        // Walk both up simultaneously until they converge.
        while a != b {
            match (self.idom(a), self.idom(b)) {
                (Some(pa), Some(pb)) => {
                    a = pa;
                    b = pb;
                }
                // One (or both) reached the root without meeting the other.
                // This can happen with unreachable blocks. Return entry.
                _ => return self.entry,
            }
        }

        a
    }

    /// Returns the CFG reverse postorder traversal computed during
    /// dominator tree construction. Useful for iterative dataflow
    /// algorithms.
    #[inline]
    pub fn reverse_postorder(&self) -> &[BasicBlockId] {
        &self.postorder
    }

    /// Returns `true` if `block` is reachable from the function entry.
    #[inline]
    pub fn is_reachable(&self, block: BasicBlockId) -> bool {
        self.reachable.contains(&block)
    }

    /// Returns the entry block (root of the dominator tree).
    #[inline]
    pub fn entry(&self) -> BasicBlockId {
        self.entry
    }
}

// ====================================================================
// LinkEvalForest — disjoint set forest for LINK/EVAL operations
// ====================================================================

/// Disjoint set forest with path compression for the EVAL/LINK
/// operations in the Lengauer-Tarjan algorithm.
///
/// Each node starts as its own tree root. The [`link`](Self::link)
/// operation merges two trees (always placing the child under the
/// parent), and [`eval`](Self::eval) finds the vertex with minimum
/// semidominator on the path from a given node to its tree root,
/// applying path compression along the way.
///
/// This implementation uses the "simple" Lengauer-Tarjan variant:
/// `link(v, w)` always sets `ancestor[w] = v`, and `eval` applies
/// iterative path compression. The amortised complexity is
/// O(n · log n) — practically identical to the theoretically optimal
/// O(n · α(n)) of the "sophisticated" balanced-link variant for all
/// realistic CFG sizes.
struct LinkEvalForest {
    /// Forest parent for each node (indexed by DFS number).
    /// `UNDEF` indicates the node is a tree root.
    ancestor: Vec<u32>,

    /// For each node `v`, `label[v]` is the vertex on the path from
    /// `v` to its tree root with the **minimum** semidominator DFS
    /// number. Initialised to `v` itself (identity).
    label: Vec<u32>,
}

impl LinkEvalForest {
    /// Create a new forest where each node is its own root.
    ///
    /// # Arguments
    ///
    /// * `n` — number of nodes (DFS numbers 0..n-1).
    fn new(n: usize) -> Self {
        Self {
            ancestor: vec![UNDEF; n],
            label: (0..n as u32).collect(),
        }
    }

    /// **EVAL(v):** Return the DFS number of the vertex `u` on the
    /// path from `v` to `v`'s tree root such that `semi[u]` is
    /// minimum.
    ///
    /// Applies path compression so that subsequent calls on the same
    /// path are O(1) amortised.
    ///
    /// # Arguments
    ///
    /// * `v`    — node (DFS number) to evaluate.
    /// * `semi` — semidominator array (indexed by DFS number).
    ///
    /// # Returns
    ///
    /// DFS number of the vertex with minimum semidominator on the
    /// compressed path from `v` to its root.
    fn eval(&mut self, v: usize, semi: &[u32]) -> usize {
        if self.ancestor[v] == UNDEF {
            // v is a tree root; its own label is the answer.
            return v;
        }
        self.compress(v, semi);
        self.label[v] as usize
    }

    /// Iterative path compression from `v` to its tree root.
    ///
    /// After compression, every node on the path points directly to
    /// the root's child, and `label[v]` reflects the minimum-semi
    /// vertex on the (now compressed) path.
    ///
    /// Uses an explicit stack to avoid recursion-depth issues on
    /// large CFGs (10 000+ blocks).
    fn compress(&mut self, v: usize, semi: &[u32]) {
        // Collect the path from v toward the root, stopping at the
        // first node whose ancestor is itself a root (or UNDEF).
        let mut path: Vec<usize> = Vec::new();
        let mut cur = v;

        while self.ancestor[cur] != UNDEF {
            path.push(cur);
            cur = self.ancestor[cur] as usize;
        }
        // `cur` is now the root. `path` = [v, ..., child_of_root].

        if path.len() < 2 {
            // Only one node on the path to root; nothing to compress.
            return;
        }

        // Compress from the node closest to the root down toward v.
        // This propagates minimum-semi labels and flattens ancestor
        // pointers.
        //
        // Iteration order: path[len-2], path[len-3], ..., path[0]
        // (path[len-1] is the child of root — its ancestor is already
        // the root, so it needs no update.)
        for i in (0..path.len() - 1).rev() {
            let node = path[i];
            let parent_on_path = path[i + 1];

            // Propagate minimum-semi label from parent to node.
            if semi[self.label[parent_on_path] as usize]
                < semi[self.label[node] as usize]
            {
                self.label[node] = self.label[parent_on_path];
            }

            // Point node directly to the same target as its parent
            // (full path compression toward root).
            self.ancestor[node] = self.ancestor[parent_on_path];
        }
    }

    /// **LINK(v, w):** Make `w` a child of `v` in the forest.
    ///
    /// Uses the "simple" Lengauer-Tarjan linking strategy: `w` is
    /// always placed directly under `v` by setting `ancestor[w] = v`.
    /// Combined with the path-compression in [`eval`](Self::eval) /
    /// [`compress`](Self::compress), this yields an amortised
    /// O(n · log n) complexity, which is practically
    /// indistinguishable from the theoretically optimal O(n · α(n))
    /// of the "sophisticated" balanced-link variant for all realistic
    /// CFG sizes (even 10 000+ basic blocks).
    ///
    /// # Why not balanced linking?
    ///
    /// The "sophisticated" balanced variant (Lengauer-Tarjan §4) uses
    /// size-balanced union in the LINK step: when `size[v] < size[w]`,
    /// the roles are swapped so `v` becomes a child of `w`, keeping
    /// tree height O(log n).  However, this requires a significantly
    /// more complex implementation with additional `child` arrays and
    /// careful label propagation to preserve the EVAL invariant.
    ///
    /// A naïve balanced union that simply reverses `ancestor[v] = w`
    /// **breaks correctness**: it can make the DFS-tree root (entry
    /// block) a non-root in the forest, causing `eval()` to return
    /// stale labels and producing wrong semi-dominators.  The simple
    /// variant avoids this class of bugs entirely.
    ///
    /// # Arguments
    ///
    /// * `v` — the node that becomes the root (DFS tree parent).
    /// * `w` — the node placed under `v`.
    fn link(&mut self, v: usize, w: usize) {
        self.ancestor[w] = v as u32;
    }
}

// ====================================================================
// Helper functions — traversal order computation
// ====================================================================

/// Compute dominator tree pre-order traversal starting from `entry`.
///
/// Uses an iterative DFS to avoid stack overflow on deep dominator
/// trees (unlikely in practice but handled for robustness).
///
/// # Returns
///
/// A `Vec<BasicBlockId>` with the entry block first, followed by all
/// dominated blocks in pre-order.
fn compute_dom_tree_preorder(
    entry: BasicBlockId,
    children: &[Vec<BasicBlockId>],
) -> Vec<BasicBlockId> {
    let mut result = Vec::new();
    let mut stack = vec![entry];

    while let Some(block) = stack.pop() {
        result.push(block);
        let bid = block.0 as usize;
        if bid < children.len() {
            // Push children in reverse order so the first child is
            // popped (and thus visited) first, producing standard
            // pre-order.
            for &child in children[bid].iter().rev() {
                stack.push(child);
            }
        }
    }

    result
}

/// Compute CFG reverse postorder starting from the entry block.
///
/// Reverse postorder is the reverse of a DFS postorder traversal of
/// the CFG. It has the property that (for reducible CFGs) every block
/// appears after all its dominators, making it the natural iteration
/// order for forward dataflow analyses.
///
/// # Arguments
///
/// * `func`          — the function whose CFG to traverse.
/// * `block_to_dfs`  — mapping from block ID to DFS number
///   (used to identify reachable blocks).
/// * `num_slots`     — size of per-block arrays.
///
/// # Returns
///
/// A `Vec<BasicBlockId>` in reverse postorder (entry block first).
fn compute_cfg_reverse_postorder(
    func: &IrFunction,
    block_to_dfs: &FxHashMap<BasicBlockId, u32>,
    num_slots: usize,
) -> Vec<BasicBlockId> {
    let entry_id = func.entry_block().id;
    let mut postorder: Vec<BasicBlockId> = Vec::with_capacity(num_slots);
    let mut visited: FxHashSet<BasicBlockId> = fx_hash_set();

    // DFS with explicit Enter/Exit actions for postorder recording.
    enum Action {
        Enter(BasicBlockId),
        Exit(BasicBlockId),
    }

    let mut stack = vec![Action::Enter(entry_id)];

    while let Some(action) = stack.pop() {
        match action {
            Action::Enter(block) => {
                if visited.contains(&block) {
                    continue;
                }
                // Only visit reachable blocks.
                if !block_to_dfs.contains_key(&block) {
                    continue;
                }
                visited.insert(block);
                stack.push(Action::Exit(block));

                let b = func.get_block(block);
                for &succ in b.successors().iter().rev() {
                    if !visited.contains(&succ) {
                        stack.push(Action::Enter(succ));
                    }
                }
            }
            Action::Exit(block) => {
                postorder.push(block);
            }
        }
    }

    // Reverse to get reverse postorder.
    postorder.reverse();
    postorder
}

/// Compute DFS pre-order and post-order timestamps on the dominator
/// tree for O(1) dominance queries.
///
/// A block `a` dominates block `b` iff:
/// ```text
/// pre_time[a] <= pre_time[b]  AND  post_time[a] >= post_time[b]
/// ```
///
/// # Returns
///
/// `(pre_time, post_time)` — both vectors indexed by `block.id.0`.
fn compute_dom_tree_timestamps(
    entry: BasicBlockId,
    children: &[Vec<BasicBlockId>],
    num_slots: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut pre_time = vec![0u32; num_slots];
    let mut post_time = vec![0u32; num_slots];
    let mut time: u32 = 0;

    // Iterative DFS with explicit Enter/Exit actions.
    enum Action {
        Enter(BasicBlockId),
        Exit(BasicBlockId),
    }

    let mut stack = vec![Action::Enter(entry)];

    while let Some(action) = stack.pop() {
        match action {
            Action::Enter(block) => {
                pre_time[block.0 as usize] = time;
                time += 1;
                // Push Exit first (so it is processed after all children).
                stack.push(Action::Exit(block));
                let bid = block.0 as usize;
                if bid < children.len() {
                    for &child in children[bid].iter().rev() {
                        stack.push(Action::Enter(child));
                    }
                }
            }
            Action::Exit(block) => {
                post_time[block.0 as usize] = time;
                time += 1;
            }
        }
    }

    (pre_time, post_time)
}

// ====================================================================
// Unit tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{BasicBlockId, Instruction};
    use crate::ir::types::IrType;

    /// Helper: create a minimal IrFunction with `n` empty blocks
    /// (IDs 0..n-1). Block 0 is the entry. No edges yet.
    fn make_func(n: usize) -> IrFunction {
        let mut func = IrFunction::new("test".into(), IrType::Void, vec![]);
        // Block 0 (entry) is already created by IrFunction::new().
        for i in 1..n {
            let bb = BasicBlock::new(BasicBlockId(i as u32), None);
            func.add_basic_block(bb);
        }
        func
    }

    /// Helper: add a directed CFG edge from `src` to `dst` in `func`.
    fn add_edge(func: &mut IrFunction, src: BasicBlockId, dst: BasicBlockId) {
        func.get_block_mut(src).add_successor(dst);
        func.get_block_mut(dst).add_predecessor(src);
    }

    /// Helper: add a `Branch` terminator to `src` targeting `dst`.
    fn set_branch(func: &mut IrFunction, src: BasicBlockId, dst: BasicBlockId) {
        let inst = Instruction::Branch { target: dst };
        func.get_block_mut(src).set_terminator(inst);
    }

    // ---- Test: single-block function ----

    #[test]
    fn test_single_block() {
        let func = make_func(1);
        let dt = DominatorTree::compute(&func);

        let bb0 = BasicBlockId(0);
        assert!(dt.idom(bb0).is_none(), "entry has no idom");
        assert!(dt.dominates(bb0, bb0), "entry dominates itself");
        assert!(!dt.strictly_dominates(bb0, bb0));
        assert_eq!(dt.depth(bb0), 0);
        assert_eq!(dt.preorder(), &[bb0]);
        assert!(dt.children(bb0).is_empty());
    }

    // ---- Test: linear chain  bb0 → bb1 → bb2 → bb3 ----

    #[test]
    fn test_linear_chain() {
        let mut func = make_func(4);
        let bb = |i: u32| BasicBlockId(i);

        add_edge(&mut func, bb(0), bb(1));
        add_edge(&mut func, bb(1), bb(2));
        add_edge(&mut func, bb(2), bb(3));
        set_branch(&mut func, bb(0), bb(1));
        set_branch(&mut func, bb(1), bb(2));
        set_branch(&mut func, bb(2), bb(3));

        let dt = DominatorTree::compute(&func);

        // In a linear chain, each block is dominated by its predecessor.
        assert_eq!(dt.idom(bb(0)), None);
        assert_eq!(dt.idom(bb(1)), Some(bb(0)));
        assert_eq!(dt.idom(bb(2)), Some(bb(1)));
        assert_eq!(dt.idom(bb(3)), Some(bb(2)));

        // Transitive dominance.
        assert!(dt.dominates(bb(0), bb(3)));
        assert!(dt.dominates(bb(1), bb(3)));
        assert!(dt.strictly_dominates(bb(0), bb(3)));
        assert!(!dt.dominates(bb(3), bb(0)));

        // Depths.
        assert_eq!(dt.depth(bb(0)), 0);
        assert_eq!(dt.depth(bb(1)), 1);
        assert_eq!(dt.depth(bb(2)), 2);
        assert_eq!(dt.depth(bb(3)), 3);

        // Pre-order.
        assert_eq!(dt.preorder(), &[bb(0), bb(1), bb(2), bb(3)]);

        // Children.
        assert_eq!(dt.children(bb(0)), &[bb(1)]);
        assert_eq!(dt.children(bb(1)), &[bb(2)]);
        assert_eq!(dt.children(bb(2)), &[bb(3)]);
        assert!(dt.children(bb(3)).is_empty());

        // LCA.
        assert_eq!(dt.lca(bb(1), bb(3)), bb(1));
        assert_eq!(dt.lca(bb(2), bb(3)), bb(2));
        assert_eq!(dt.lca(bb(0), bb(3)), bb(0));
    }

    // ---- Test: diamond  bb0 → {bb1, bb2} → bb3 ----

    #[test]
    fn test_diamond() {
        let mut func = make_func(4);
        let bb = |i: u32| BasicBlockId(i);

        // bb0 branches to bb1 and bb2; both go to bb3.
        add_edge(&mut func, bb(0), bb(1));
        add_edge(&mut func, bb(0), bb(2));
        add_edge(&mut func, bb(1), bb(3));
        add_edge(&mut func, bb(2), bb(3));

        let dt = DominatorTree::compute(&func);

        assert_eq!(dt.idom(bb(1)), Some(bb(0)));
        assert_eq!(dt.idom(bb(2)), Some(bb(0)));
        // bb3 is dominated by bb0 (the last common dominator of bb1 and bb2).
        assert_eq!(dt.idom(bb(3)), Some(bb(0)));

        assert!(dt.dominates(bb(0), bb(3)));
        assert!(!dt.dominates(bb(1), bb(3)));
        assert!(!dt.dominates(bb(2), bb(3)));

        assert_eq!(dt.depth(bb(1)), 1);
        assert_eq!(dt.depth(bb(2)), 1);
        assert_eq!(dt.depth(bb(3)), 1);

        // LCA of bb1 and bb2 is bb0.
        assert_eq!(dt.lca(bb(1), bb(2)), bb(0));
        // LCA of bb1 and bb3 is bb0.
        assert_eq!(dt.lca(bb(1), bb(3)), bb(0));
    }

    // ---- Test: loop (back edge)  bb0 → bb1 → bb2 → bb1 ----

    #[test]
    fn test_loop_back_edge() {
        let mut func = make_func(3);
        let bb = |i: u32| BasicBlockId(i);

        add_edge(&mut func, bb(0), bb(1));
        add_edge(&mut func, bb(1), bb(2));
        add_edge(&mut func, bb(2), bb(1)); // back edge

        let dt = DominatorTree::compute(&func);

        assert_eq!(dt.idom(bb(1)), Some(bb(0)));
        assert_eq!(dt.idom(bb(2)), Some(bb(1)));

        // bb1 dominates bb2 despite the back edge.
        assert!(dt.dominates(bb(1), bb(2)));
        assert!(dt.dominates(bb(0), bb(2)));
    }

    // ---- Test: unreachable block ----

    #[test]
    fn test_unreachable_block() {
        let mut func = make_func(3);
        let bb = |i: u32| BasicBlockId(i);

        // bb0 → bb1; bb2 is unreachable.
        add_edge(&mut func, bb(0), bb(1));

        let dt = DominatorTree::compute(&func);

        assert_eq!(dt.idom(bb(1)), Some(bb(0)));
        assert_eq!(dt.idom(bb(2)), None); // unreachable
        assert!(!dt.is_reachable(bb(2)));

        // Unreachable block is not dominated by anything except itself.
        assert!(dt.dominates(bb(2), bb(2)));
        assert!(!dt.dominates(bb(0), bb(2)));
    }

    // ---- Test: self-loop ----

    #[test]
    fn test_self_loop() {
        let mut func = make_func(2);
        let bb = |i: u32| BasicBlockId(i);

        add_edge(&mut func, bb(0), bb(1));
        add_edge(&mut func, bb(1), bb(1)); // self-loop

        let dt = DominatorTree::compute(&func);

        assert_eq!(dt.idom(bb(1)), Some(bb(0)));
        assert!(dt.dominates(bb(0), bb(1)));
    }

    // ---- Test: complex CFG (if-else with merge) ----
    //
    //        bb0
    //       / \
    //     bb1  bb2
    //     |    / \
    //     |  bb3  bb4
    //      \  |  /
    //       bb5
    //

    #[test]
    fn test_complex_cfg() {
        let mut func = make_func(6);
        let bb = |i: u32| BasicBlockId(i);

        add_edge(&mut func, bb(0), bb(1));
        add_edge(&mut func, bb(0), bb(2));
        add_edge(&mut func, bb(2), bb(3));
        add_edge(&mut func, bb(2), bb(4));
        add_edge(&mut func, bb(1), bb(5));
        add_edge(&mut func, bb(3), bb(5));
        add_edge(&mut func, bb(4), bb(5));

        let dt = DominatorTree::compute(&func);

        assert_eq!(dt.idom(bb(1)), Some(bb(0)));
        assert_eq!(dt.idom(bb(2)), Some(bb(0)));
        assert_eq!(dt.idom(bb(3)), Some(bb(2)));
        assert_eq!(dt.idom(bb(4)), Some(bb(2)));
        // bb5 has three predecessors: bb1, bb3, bb4.
        // idom(bb5) = bb0 (common dominator of bb1 and bb2).
        assert_eq!(dt.idom(bb(5)), Some(bb(0)));

        assert!(dt.dominates(bb(0), bb(5)));
        assert!(!dt.dominates(bb(1), bb(5)));
        assert!(!dt.dominates(bb(2), bb(5)));
        assert!(dt.dominates(bb(2), bb(3)));
        assert!(dt.dominates(bb(2), bb(4)));

        // LCA tests.
        assert_eq!(dt.lca(bb(3), bb(4)), bb(2));
        assert_eq!(dt.lca(bb(1), bb(3)), bb(0));
        assert_eq!(dt.lca(bb(1), bb(4)), bb(0));
    }

    // ---- Test: preorder and reverse_postorder ----

    #[test]
    fn test_traversal_orders() {
        let mut func = make_func(4);
        let bb = |i: u32| BasicBlockId(i);

        add_edge(&mut func, bb(0), bb(1));
        add_edge(&mut func, bb(0), bb(2));
        add_edge(&mut func, bb(1), bb(3));
        add_edge(&mut func, bb(2), bb(3));

        let dt = DominatorTree::compute(&func);

        // Pre-order starts with entry.
        assert_eq!(dt.preorder()[0], bb(0));
        // All blocks present.
        assert_eq!(dt.preorder().len(), 4);

        // Reverse postorder starts with entry.
        assert_eq!(dt.reverse_postorder()[0], bb(0));
        assert_eq!(dt.reverse_postorder().len(), 4);
    }
}
