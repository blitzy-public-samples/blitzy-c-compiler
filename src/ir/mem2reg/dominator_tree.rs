//! Dominator Tree Computation — Lengauer-Tarjan Algorithm
//!
//! Computes the dominator tree for a function's control flow graph.
//! A block A dominates block B if every path from the entry block to B
//! must pass through A. The immediate dominator (idom) of B is the
//! closest strict dominator of B.
//!
//! ## Algorithm
//!
//! Uses the Lengauer-Tarjan algorithm, which runs in O(n × α(n)) time
//! where n is the number of basic blocks and α is the inverse Ackermann
//! function (effectively constant). This efficiency is critical for
//! compiling large kernel functions with hundreds of basic blocks.
//!
//! An alternative Cooper-Harvey-Kennedy iterative algorithm is also provided
//! via [`DominatorTree::build_simple`] for verification and small functions.
//!
//! ## References
//!
//! - Lengauer, T. and Tarjan, R.E. "A fast algorithm for finding dominators
//!   in a flowgraph." ACM TOPLAS 1(1), 1979.
//! - Cooper, Harvey, Kennedy. "A Simple, Fast Dominance Algorithm." TR, 2001.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;

/// Sentinel value indicating an unreachable block or uninitialized data.
/// Used in `idom`, `dfs_number`, and internal working arrays.
const SENTINEL: usize = usize::MAX;

// ===========================================================================
// UnionFind — EVAL/LINK forest for Lengauer-Tarjan
// ===========================================================================

/// Union-Find forest for the Lengauer-Tarjan algorithm.
///
/// Supports path compression for near-O(1) amortized EVAL operations.
/// All arrays are indexed by DFS number (not block index).
///
/// The forest tracks, for each vertex, the vertex with the minimum
/// semidominator value on the path from that vertex to the root of its
/// tree. This is the key data structure enabling efficient semidominator
/// computation.
struct UnionFind {
    /// Parent of each vertex in the forest.
    /// `ancestor[v] == SENTINEL` means v is a root of its tree.
    ancestor: Vec<usize>,

    /// Label for each vertex — the vertex (DFS number) with the minimum
    /// semidominator on the path from this vertex to the root of its
    /// tree in the forest. Updated during path compression.
    label: Vec<usize>,

    /// Semidominator value for each vertex, stored as a DFS number.
    /// `semi[v]` is the DFS number of the semidominator of vertex v.
    /// Initially `semi[v] = v` (each vertex is its own semidominator).
    semi: Vec<usize>,
}

impl UnionFind {
    /// Create a new union-find forest for `n` vertices (DFS numbers 0..n-1).
    ///
    /// Initially each vertex is a root of its own tree, with
    /// `label[v] = v` and `semi[v] = v`.
    fn new(n: usize) -> Self {
        let mut semi = Vec::with_capacity(n);
        let mut label = Vec::with_capacity(n);
        for i in 0..n {
            semi.push(i);
            label.push(i);
        }
        UnionFind {
            ancestor: vec![SENTINEL; n],
            label,
            semi,
        }
    }

    /// EVAL operation: find the vertex with minimum semidominator
    /// on the path from `v` to the root of its tree in the forest.
    ///
    /// Applies path compression to flatten the tree, ensuring
    /// amortized near-O(1) performance (O(α(n)) with the simple
    /// LINK implementation used here).
    ///
    /// # Parameters
    ///
    /// - `v`: DFS number of the vertex to evaluate.
    ///
    /// # Returns
    ///
    /// DFS number of the vertex with minimum `semi` value on the
    /// path from `v` to the root.
    fn eval(&mut self, v: usize) -> usize {
        // If v is a root, return its label directly.
        if self.ancestor[v] == SENTINEL {
            return self.label[v];
        }

        // Collect all non-root vertices on the path from v to the root.
        // We will compress this path afterward.
        let mut path = Vec::new();
        let mut u = v;
        while self.ancestor[u] != SENTINEL {
            path.push(u);
            u = self.ancestor[u];
        }
        // `u` is now the root of the tree containing `v`.

        // Path compression: process from the vertex closest to the root
        // back toward `v`. For each vertex on the path, update its label
        // to reflect the minimum semidominator encountered, and point it
        // directly to the tree root.
        //
        // We skip the last element of `path` (closest to root) because
        // its ancestor IS the root — the base case of the recursive
        // compress operation.
        if path.len() >= 2 {
            for i in (0..path.len() - 1).rev() {
                let w = path[i];
                let a = self.ancestor[w];
                // If the ancestor's label has a smaller semidominator,
                // propagate it down to w.
                if self.semi[self.label[a]] < self.semi[self.label[w]] {
                    self.label[w] = self.label[a];
                }
                // Point w directly to its ancestor's ancestor (path compression).
                // After the previous iteration, ancestor[a] has been updated
                // to point closer to (or at) the root.
                self.ancestor[w] = self.ancestor[a];
            }
        }

        self.label[v]
    }

    /// LINK operation: add an edge from `v` to `w` in the forest.
    ///
    /// Makes `v` the parent (ancestor) of `w`. This is called as
    /// `LINK(parent[w], w)` in the Lengauer-Tarjan algorithm, linking
    /// each vertex to its DFS parent.
    ///
    /// # Parameters
    ///
    /// - `v`: DFS number of the parent vertex.
    /// - `w`: DFS number of the child vertex.
    #[inline]
    fn link(&mut self, v: usize, w: usize) {
        self.ancestor[w] = v;
    }
}

// ===========================================================================
// DFS result — output of the depth-first search phase
// ===========================================================================

/// Result of the DFS traversal from the entry block.
///
/// Contains all the numbering information needed by both the
/// Lengauer-Tarjan algorithm and the dominator tree construction.
struct DfsResult {
    /// DFS preorder mapping: `vertex[dfs_number] = block_index`.
    /// Only contains reachable blocks (length = `dfs_count`).
    vertex: Vec<usize>,

    /// Reverse mapping: `dfnum[block_index] = dfs_number`.
    /// Unreachable blocks have `dfnum[block] == SENTINEL`.
    dfnum: Vec<usize>,

    /// DFS tree parent for each vertex: `parent[dfs_number] = parent_dfs_number`.
    /// The root (DFS number 0) has `parent[0] == SENTINEL`.
    parent: Vec<usize>,

    /// Number of blocks reachable from the entry via DFS.
    dfs_count: usize,

    /// Reverse postorder of the CFG (block indices).
    /// The entry block is always first. Only reachable blocks are included.
    rpo: Vec<usize>,
}

/// Perform a depth-first search from the entry block (block 0), computing
/// DFS numbering, parent relationships, and reverse postorder.
///
/// Uses an explicit stack (not recursion) to handle arbitrarily large
/// functions without risking stack overflow — critical for compiling
/// Linux kernel functions with hundreds of basic blocks.
///
/// # Parameters
///
/// - `func`: The function whose CFG is being traversed.
///
/// # Returns
///
/// A [`DfsResult`] containing all DFS-derived data.
fn dfs_from_entry(func: &IrFunction) -> DfsResult {
    let n = func.block_count();

    let mut dfnum = vec![SENTINEL; n];
    let mut vertex = Vec::with_capacity(n);
    let mut parent = Vec::with_capacity(n);
    let mut postorder = Vec::with_capacity(n);

    if n == 0 {
        return DfsResult {
            vertex,
            dfnum,
            parent,
            dfs_count: 0,
            rpo: Vec::new(),
        };
    }

    let mut counter: usize = 0;

    // Assign DFS number 0 to the entry block.
    dfnum[0] = counter;
    vertex.push(0);
    parent.push(SENTINEL); // root has no parent
    counter += 1;

    // Explicit DFS stack: each frame tracks the current block and
    // the index of the next successor to explore.
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n);
    stack.push((0, 0));

    while let Some((block_idx, succ_pos)) = stack.last_mut() {
        let succs = func.successors(*block_idx);
        if *succ_pos < succs.len() {
            let next_block = succs[*succ_pos];
            *succ_pos += 1;

            // Only visit unvisited, valid blocks.
            if next_block < n && dfnum[next_block] == SENTINEL {
                let parent_dfs = dfnum[*block_idx];
                dfnum[next_block] = counter;
                vertex.push(next_block);
                parent.push(parent_dfs);
                counter += 1;
                stack.push((next_block, 0));
            }
        } else {
            // All successors explored — record in postorder.
            let finished = *block_idx;
            stack.pop();
            postorder.push(finished);
        }
    }

    // Reverse postorder: reverse the postorder list.
    postorder.reverse();

    DfsResult {
        vertex,
        dfnum,
        parent,
        dfs_count: counter,
        rpo: postorder,
    }
}

// ===========================================================================
// DominatorTree — the public dominator tree data structure
// ===========================================================================

/// Dominator tree for a function's control flow graph.
///
/// Stores the immediate dominator (idom) for each block and provides
/// tree traversal utilities (children, ancestors, dominance queries).
///
/// The entry block (block 0) is the root of the dominator tree and has
/// no immediate dominator. Unreachable blocks (not reachable from the
/// entry via the CFG) have `idom == SENTINEL` and are treated as having
/// no dominator.
///
/// # Construction
///
/// Use [`DominatorTree::build`] for the efficient Lengauer-Tarjan algorithm
/// (O(n × α(n))), or [`DominatorTree::build_simple`] for the simpler
/// iterative Cooper-Harvey-Kennedy algorithm (O(n²) worst case).
///
/// # Usage in SSA Construction
///
/// The dominator tree is a foundational data structure for SSA construction:
/// - **Dominance frontiers** are computed from the dominator tree to
///   determine where phi nodes must be placed.
/// - **SSA renaming** walks the dominator tree in preorder to maintain
///   reaching-definition stacks.
pub struct DominatorTree {
    /// Immediate dominator for each block index.
    ///
    /// - `idom[entry]` = entry (self-loop for the root).
    /// - `idom[b]` = block index of b's immediate dominator for reachable blocks.
    /// - `idom[b]` = `SENTINEL` for unreachable blocks.
    idom: Vec<usize>,

    /// Children in the dominator tree.
    ///
    /// `children[i]` contains the block indices that are immediately
    /// dominated by block `i` (i.e., blocks whose idom is `i`).
    children: Vec<Vec<usize>>,

    /// Total number of blocks in the function (including unreachable ones).
    block_count: usize,

    /// Reverse postorder of the CFG (block indices).
    ///
    /// The entry block is always first. Only reachable blocks are included.
    /// Used by [`reverse_postorder`](DominatorTree::reverse_postorder) and
    /// as the canonical iteration order for dataflow analyses.
    dfs_order: Vec<usize>,

    /// DFS preorder number for each block index.
    ///
    /// `dfs_number[block_index]` = DFS number assigned during the
    /// Lengauer-Tarjan DFS traversal. Unreachable blocks have
    /// `dfs_number[block] == SENTINEL`.
    dfs_number: Vec<usize>,
}

impl DominatorTree {
    // ===================================================================
    // Construction — primary and alternative algorithms
    // ===================================================================

    /// Build the dominator tree for a function using the Lengauer-Tarjan algorithm.
    ///
    /// This is the preferred construction method for production use.
    /// It runs in O(n × α(n)) time where n is the number of basic blocks
    /// and α is the inverse Ackermann function (effectively constant).
    ///
    /// # Algorithm Steps
    ///
    /// 1. **DFS numbering** — Perform depth-first search from the entry block
    ///    to assign DFS numbers and identify reachable blocks.
    /// 2. **Semidominator computation** — Process blocks in reverse DFS order,
    ///    using the EVAL/LINK union-find operations to efficiently compute
    ///    semidominators.
    /// 3. **Implicit dominator computation** — Derive tentative dominators
    ///    from semidominators using the forest structure.
    /// 4. **Explicit dominator computation** — Finalize dominators in a
    ///    forward DFS pass.
    /// 5. **Children construction** — Build the tree's child lists from
    ///    the computed idom array.
    ///
    /// # Parameters
    ///
    /// - `func`: The function whose CFG is being analyzed.
    ///
    /// # Returns
    ///
    /// The computed [`DominatorTree`].
    pub fn build(func: &IrFunction) -> Self {
        let n = func.block_count();
        if n == 0 {
            return Self::empty(0);
        }

        // ---------------------------------------------------------------
        // Step 1: DFS numbering from the entry block.
        // ---------------------------------------------------------------
        let dfs_result = dfs_from_entry(func);
        let dfs_count = dfs_result.dfs_count;

        // Edge case: no reachable blocks (degenerate function).
        if dfs_count == 0 {
            return Self::empty(n);
        }

        // Edge case: single block — entry dominates itself.
        if dfs_count == 1 {
            let mut idom_arr = vec![SENTINEL; n];
            idom_arr[0] = 0;
            return Self::finalize(idom_arr, n, dfs_result.dfnum, dfs_result.rpo);
        }

        let vertex = &dfs_result.vertex;
        let dfnum = &dfs_result.dfnum;
        let parent_dfs = &dfs_result.parent;

        // ---------------------------------------------------------------
        // Step 2-3: Compute semidominators and implicit dominators.
        //
        // All working arrays below are indexed by DFS number (0..dfs_count).
        // ---------------------------------------------------------------
        let mut uf = UnionFind::new(dfs_count);
        let mut dom_dfs = vec![SENTINEL; dfs_count];
        let mut bucket: Vec<Vec<usize>> = vec![Vec::new(); dfs_count];

        // Process vertices in reverse DFS order (highest DFS number first,
        // skipping the root at DFS number 0).
        for d in (1..dfs_count).rev() {
            let w_block = vertex[d]; // block index of the vertex with DFS number d

            // --- Step 2: Compute semidominator of d ---
            // The semidominator is the vertex with the smallest DFS number
            // reachable via a path of tree and cross/forward edges.
            for &pred_block in func.predecessors(w_block) {
                // Skip unreachable predecessors (should be rare in well-formed CFGs).
                let pred_d = dfnum[pred_block];
                if pred_d == SENTINEL {
                    continue;
                }

                let u = uf.eval(pred_d);
                if uf.semi[u] < uf.semi[d] {
                    uf.semi[d] = uf.semi[u];
                }
            }

            // Add d to the bucket of its semidominator.
            bucket[uf.semi[d]].push(d);

            // LINK: make parent[d] the ancestor of d in the forest.
            let p = parent_dfs[d];
            uf.link(p, d);

            // --- Step 3: Implicitly compute dominators ---
            // Process the bucket of parent[d]: for each vertex v whose
            // semidominator equals parent[d], determine v's dominator.
            let bucket_p = std::mem::take(&mut bucket[p]);
            for v in bucket_p {
                let u = uf.eval(v);
                if uf.semi[u] < uf.semi[v] {
                    // The semidominator of u is strictly less than that of v,
                    // so v's dominator is u (not parent[d]).
                    dom_dfs[v] = u;
                } else {
                    // v's dominator is parent[d] (the semidominator vertex).
                    dom_dfs[v] = p;
                }
            }
        }

        // ---------------------------------------------------------------
        // Step 4: Explicitly compute dominators.
        //
        // For vertices whose tentative dominator differs from their
        // semidominator vertex, chase the idom chain to the correct value.
        // ---------------------------------------------------------------
        dom_dfs[0] = 0; // Root dominates itself.
        for d in 1..dfs_count {
            if dom_dfs[d] != uf.semi[d] {
                dom_dfs[d] = dom_dfs[dom_dfs[d]];
            }
        }

        // ---------------------------------------------------------------
        // Step 5: Convert from DFS-number space to block-index space.
        // ---------------------------------------------------------------
        let mut idom_arr = vec![SENTINEL; n];
        for d in 0..dfs_count {
            let block_idx = vertex[d];
            let idom_dfs_num = dom_dfs[d];
            idom_arr[block_idx] = vertex[idom_dfs_num];
        }

        Self::finalize(idom_arr, n, dfs_result.dfnum, dfs_result.rpo)
    }

    /// Build the dominator tree using the simple iterative algorithm
    /// (Cooper, Harvey, Kennedy 2001).
    ///
    /// This algorithm is O(n²) in the worst case but simpler to implement
    /// and understand. It is useful for:
    ///
    /// - **Small functions** (most functions in practice have few blocks).
    /// - **Verification** — comparing results against the Lengauer-Tarjan
    ///   implementation in debug/test builds.
    ///
    /// # Algorithm
    ///
    /// 1. Compute reverse postorder (RPO) of the CFG.
    /// 2. Initialize `idom[entry] = entry`.
    /// 3. Iterate over blocks in RPO (excluding entry):
    ///    - For each block, intersect the idom chains of all processed
    ///      predecessors to find the new idom.
    ///    - Repeat until no idom values change (fixpoint).
    ///
    /// # Parameters
    ///
    /// - `func`: The function whose CFG is being analyzed.
    ///
    /// # Returns
    ///
    /// The computed [`DominatorTree`].
    pub fn build_simple(func: &IrFunction) -> Self {
        let n = func.block_count();
        if n == 0 {
            return Self::empty(0);
        }

        // Perform DFS to obtain RPO and DFS numbering.
        let dfs_result = dfs_from_entry(func);
        let rpo = &dfs_result.rpo;

        if rpo.is_empty() {
            return Self::empty(n);
        }

        // RPO position for each block (lower number = earlier in RPO).
        let mut rpo_pos = vec![SENTINEL; n];
        for (pos, &block) in rpo.iter().enumerate() {
            rpo_pos[block] = pos;
        }

        // Initialize idom array. Entry block (block 0) dominates itself.
        let mut idom_arr = vec![SENTINEL; n];
        idom_arr[rpo[0]] = rpo[0]; // entry block

        // Iterate until fixpoint.
        let mut changed = true;
        while changed {
            changed = false;

            // Process all blocks in RPO, skipping the entry.
            for &b in rpo.iter().skip(1) {
                let preds = func.predecessors(b);
                let mut new_idom = SENTINEL;

                // Find first processed predecessor.
                for &p in preds {
                    if idom_arr[p] != SENTINEL {
                        if new_idom == SENTINEL {
                            new_idom = p;
                        } else {
                            new_idom = intersect_blocks(&idom_arr, &rpo_pos, new_idom, p);
                        }
                    }
                }

                if new_idom != SENTINEL && idom_arr[b] != new_idom {
                    idom_arr[b] = new_idom;
                    changed = true;
                }
            }
        }

        Self::finalize(idom_arr, n, dfs_result.dfnum, dfs_result.rpo)
    }

    // ===================================================================
    // Internal construction helpers
    // ===================================================================

    /// Create an empty dominator tree for a function with `n` blocks.
    /// Used when the function has no blocks or no reachable blocks.
    fn empty(n: usize) -> Self {
        DominatorTree {
            idom: vec![SENTINEL; n],
            children: vec![Vec::new(); n],
            block_count: n,
            dfs_order: Vec::new(),
            dfs_number: vec![SENTINEL; n],
        }
    }

    /// Finalize the dominator tree from a computed idom array.
    ///
    /// Builds the children lists from the idom relationships and
    /// packages all data into a [`DominatorTree`].
    fn finalize(idom_arr: Vec<usize>, n: usize, dfnum: Vec<usize>, rpo: Vec<usize>) -> Self {
        // Build children lists from the idom array.
        let mut children = vec![Vec::new(); n];
        for (block, &dom) in idom_arr.iter().enumerate() {
            // Add to children if:
            // - The block is reachable (idom != SENTINEL)
            // - The block is NOT the entry (entry's idom is itself)
            if dom != SENTINEL && dom != block {
                children[dom].push(block);
            }
        }

        DominatorTree {
            idom: idom_arr,
            children,
            block_count: n,
            dfs_order: rpo,
            dfs_number: dfnum,
        }
    }

    // ===================================================================
    // Query methods — dominator tree inspection
    // ===================================================================

    /// Get the immediate dominator of a block.
    ///
    /// Returns `None` for the entry block (root of the dominator tree)
    /// and for unreachable blocks. Returns `Some(idom_block)` for all
    /// other reachable blocks.
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.block_count`.
    #[inline]
    pub fn idom(&self, block: usize) -> Option<usize> {
        let dom = self.idom[block];
        if dom == SENTINEL || dom == block {
            // Entry block (dom == block) or unreachable (dom == SENTINEL).
            None
        } else {
            Some(dom)
        }
    }

    /// Get the children of a block in the dominator tree.
    ///
    /// These are the blocks immediately dominated by the given block
    /// (i.e., blocks whose immediate dominator is `block`).
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.block_count`.
    #[inline]
    pub fn children(&self, block: usize) -> &[usize] {
        &self.children[block]
    }

    /// Check if block `a` dominates block `b`.
    ///
    /// A block dominates itself (reflexive dominance). The entry block
    /// dominates all reachable blocks. Unreachable blocks are not
    /// dominated by any block (returns `false` if `b` is unreachable,
    /// except for the degenerate case `a == b` where both are unreachable).
    ///
    /// This walks up the idom chain from `b`, so it is O(depth) in the
    /// dominator tree. For most practical CFGs (including Linux kernel
    /// functions), the tree depth is logarithmic in the number of blocks.
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        if a >= self.block_count || b >= self.block_count {
            return false;
        }

        // Reflexive: every block dominates itself.
        if a == b {
            return true;
        }

        // If b is unreachable, a cannot dominate b (unless a == b, handled above).
        if self.idom[b] == SENTINEL {
            return false;
        }

        // If a is unreachable, a cannot dominate anything except itself.
        if self.idom[a] == SENTINEL {
            return false;
        }

        // Walk up the idom chain from b until we reach a or the root.
        let mut current = b;
        loop {
            let dom = self.idom[current];
            if dom == current {
                // Reached the root without finding a.
                return false;
            }
            if dom == a {
                return true;
            }
            current = dom;
        }
    }

    /// Check if block `a` strictly dominates block `b`.
    ///
    /// A strictly dominates B if A dominates B and A ≠ B.
    #[inline]
    pub fn strictly_dominates(&self, a: usize, b: usize) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Return the total number of blocks in the function.
    ///
    /// This includes both reachable and unreachable blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Return blocks in preorder of the dominator tree.
    ///
    /// This is the order needed by SSA renaming, which processes blocks
    /// from the root (entry) outward through the dominator tree. Each
    /// block is visited before any of its children in the dominator tree.
    ///
    /// Only reachable blocks are included.
    pub fn preorder(&self) -> Vec<usize> {
        if self.block_count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.block_count);
        let mut stack = Vec::with_capacity(self.block_count);

        // Find the root (entry block — block whose idom is itself).
        // For well-formed functions, this is always block 0.
        let root = self.find_root();
        if root == SENTINEL {
            return Vec::new();
        }

        stack.push(root);
        while let Some(block) = stack.pop() {
            result.push(block);
            // Push children in reverse order so they are visited in
            // forward order (stack is LIFO).
            for &child in self.children[block].iter().rev() {
                stack.push(child);
            }
        }

        result
    }

    /// Return blocks in reverse postorder of the CFG.
    ///
    /// The entry block is always first. This is the canonical iteration
    /// order for forward dataflow analyses. Only reachable blocks are
    /// included.
    #[inline]
    pub fn reverse_postorder(&self) -> &[usize] {
        &self.dfs_order
    }

    /// Get the depth of a block in the dominator tree.
    ///
    /// The entry block has depth 0. Each step up the idom chain adds 1.
    /// Unreachable blocks return 0 (since they have no dominator).
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.block_count`.
    pub fn depth(&self, block: usize) -> usize {
        let mut d = 0;
        let mut current = block;

        // Walk up the idom chain counting steps.
        loop {
            let dom = self.idom[current];
            if dom == SENTINEL || dom == current {
                // Reached the root or an unreachable block.
                break;
            }
            d += 1;
            current = dom;
        }

        d
    }

    // ===================================================================
    // Mutation — store results into BasicBlock structures
    // ===================================================================

    /// Store computed immediate dominators into BasicBlock structures.
    ///
    /// For each reachable non-entry block, calls `block.set_idom(idom_block)`.
    /// The entry block's idom is set to `None` (it has no dominator).
    /// Unreachable blocks are left with their existing idom value.
    ///
    /// This allows other passes (dominance frontier computation, SSA
    /// renaming) to access dominator information directly from blocks
    /// without needing a reference to the `DominatorTree`.
    pub fn store_idom_in_blocks(&self, func: &mut IrFunction) {
        let count = func.block_count().min(self.block_count);

        for i in 0..count {
            let dom = self.idom[i];
            if dom == SENTINEL {
                // Unreachable block — leave idom unchanged.
                continue;
            }
            if dom == i {
                // Entry block — no dominator.
                // BasicBlock.idom is Option<usize>, so we don't set it
                // (or we could explicitly clear it if needed).
                continue;
            }
            // Use get_block_mut() for safe, bounds-checked mutable access.
            if let Some(block) = func.get_block_mut(i) {
                block.set_idom(dom);
            }
        }
    }

    // ===================================================================
    // Debug and validation
    // ===================================================================

    /// Format the dominator tree as a human-readable string for debugging.
    ///
    /// Displays the tree structure with indentation showing parent-child
    /// relationships, along with DFS numbers and idom values.
    pub fn dump(&self) -> String {
        let mut output = String::with_capacity(self.block_count * 40);
        output.push_str("DominatorTree {\n");
        output.push_str(&format!("  block_count: {}\n", self.block_count));
        output.push_str(&format!("  reachable: {}\n", self.dfs_order.len()));

        // Print idom relationships.
        output.push_str("  idom: [");
        for i in 0..self.block_count {
            if i > 0 {
                output.push_str(", ");
            }
            let dom = self.idom[i];
            if dom == SENTINEL {
                output.push_str("UNREACH");
            } else if dom == i {
                output.push_str("ROOT");
            } else {
                output.push_str(&dom.to_string());
            }
        }
        output.push_str("]\n");

        // Print tree structure with indentation.
        let root = self.find_root();
        if root != SENTINEL {
            output.push_str("  tree:\n");
            self.dump_subtree(root, 2, &mut output);
        }

        output.push_str("}\n");
        output
    }

    /// Validate the dominator tree for correctness.
    ///
    /// Checks the following properties:
    /// - The entry block's idom is itself (root of the tree).
    /// - Every reachable non-entry block has a valid idom.
    /// - No cycles exist in the idom chain (except the entry self-loop).
    /// - The idom of every reachable block dominates that block (i.e.,
    ///   every path from entry to the block passes through its idom).
    /// - The children lists are consistent with the idom array.
    ///
    /// Returns `true` if the tree is valid, `false` otherwise.
    pub fn validate(&self, func: &IrFunction) -> bool {
        let n = func.block_count();
        if n != self.block_count {
            return false;
        }

        if n == 0 {
            return true;
        }

        // Track reachable blocks using FxHashSet (as required by
        // the zero-dependency mandate for hash-based structures).
        let mut reachable = FxHashSet::default();
        for &block in &self.dfs_order {
            reachable.insert(block);
        }

        // Track idom depth for cycle detection using FxHashMap.
        let mut depth_cache: FxHashMap<usize, usize> = FxHashMap::default();

        // Check 1: Entry block should be root (idom = self).
        if self.idom.is_empty() {
            return n == 0;
        }
        let entry = 0;
        if reachable.contains(&entry) && self.idom[entry] != entry {
            return false;
        }

        // Check 2: Every reachable non-entry block must have a valid idom.
        for &block in &reachable {
            if block == entry {
                continue;
            }
            let dom = self.idom[block];
            if dom == SENTINEL {
                return false; // Reachable but no idom.
            }
            if dom >= n {
                return false; // idom out of range.
            }
            if !reachable.contains(&dom) {
                return false; // idom is unreachable (inconsistent).
            }
        }

        // Check 3: No cycles in the idom chain (except entry self-loop).
        // Walk from each reachable block up the idom chain; it must reach
        // the entry within `n` steps.
        for &block in &reachable {
            if block == entry {
                depth_cache.insert(entry, 0);
                continue;
            }
            let mut current = block;
            let mut steps = 0;
            loop {
                if let Some(&cached) = depth_cache.get(&current) {
                    depth_cache.insert(block, steps + cached);
                    break;
                }
                let dom = self.idom[current];
                if dom == current {
                    // Reached the root.
                    depth_cache.insert(block, steps);
                    break;
                }
                if dom == SENTINEL {
                    return false; // Chain ends at unreachable block.
                }
                steps += 1;
                if steps > n {
                    return false; // Cycle detected.
                }
                current = dom;
            }
        }

        // Check 4: Children lists are consistent with the idom array.
        // Every child's idom should point back to its parent.
        for block in 0..n {
            for &child in &self.children[block] {
                if child >= n {
                    return false;
                }
                if self.idom[child] != block {
                    return false; // Child's idom doesn't match parent.
                }
            }
        }

        // Check 5: Every reachable non-entry block appears in exactly
        // one parent's children list.
        let mut child_count = vec![0usize; n];
        for block in 0..n {
            for &child in &self.children[block] {
                if child < n {
                    child_count[child] += 1;
                }
            }
        }
        for &block in &reachable {
            if block == entry {
                // Entry should not be a child of anyone.
                if child_count[block] != 0 {
                    return false;
                }
            } else if child_count[block] != 1 {
                // Every other reachable block should be a child exactly once.
                return false;
            }
        }

        // Check 6: For join points (blocks with multiple predecessors),
        // verify the idom is a common dominator of all predecessors.
        let blocks: &[BasicBlock] = func.blocks();
        for &block in &reachable {
            if block == entry {
                continue;
            }
            let dom = self.idom[block];
            let bb = &blocks[block];

            // Use predecessor_count() for efficient join-point detection.
            if bb.predecessor_count() > 1 {
                // The idom of a join point must dominate all predecessors.
                for &pred in bb.predecessors() {
                    if reachable.contains(&pred) && !self.dominates(dom, pred) {
                        return false;
                    }
                }
            }

            // Sanity check: successor_count() should be consistent with
            // the successors() slice length for all reachable blocks.
            if bb.successor_count() != bb.successors().len() {
                return false;
            }
        }

        true
    }

    // ===================================================================
    // Internal helper methods
    // ===================================================================

    /// Find the root of the dominator tree (block whose idom is itself).
    /// Returns `SENTINEL` if no root is found.
    fn find_root(&self) -> usize {
        // In well-formed functions, the root is always block 0 (entry).
        if !self.idom.is_empty() && self.idom[0] == 0 {
            return 0;
        }
        // Fallback: search all blocks.
        for i in 0..self.block_count {
            if self.idom[i] == i {
                return i;
            }
        }
        SENTINEL
    }

    /// Recursively dump a subtree of the dominator tree with indentation.
    fn dump_subtree(&self, block: usize, indent: usize, output: &mut String) {
        let prefix: String = " ".repeat(indent);
        let dfs_num = if block < self.dfs_number.len() && self.dfs_number[block] != SENTINEL {
            format!("dfs={}", self.dfs_number[block])
        } else {
            "unreachable".to_string()
        };
        output.push_str(&format!("{}BB{} ({})\n", prefix, block, dfs_num));

        if block < self.children.len() {
            for &child in &self.children[block] {
                self.dump_subtree(child, indent + 2, output);
            }
        }
    }
}

// ===========================================================================
// Free functions — helpers for the Cooper-Harvey-Kennedy algorithm
// ===========================================================================

/// Intersect two blocks' idom chains to find their nearest common dominator.
///
/// This is the core operation of the Cooper-Harvey-Kennedy algorithm.
/// It walks up the idom chains of both blocks using RPO positions as
/// the ordering criterion, converging to the nearest common ancestor
/// in the dominator tree.
///
/// # Parameters
///
/// - `idom`: The current idom array (block index → idom block index).
/// - `rpo_pos`: RPO position for each block (lower = earlier in RPO).
/// - `a`, `b`: The two blocks to intersect.
///
/// # Returns
///
/// The nearest common dominator of `a` and `b`.
fn intersect_blocks(idom: &[usize], rpo_pos: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        // Walk up the chain of the block with the larger RPO position
        // (i.e., the block that appears later in RPO).
        while rpo_pos[a] > rpo_pos[b] {
            let next = idom[a];
            if next == SENTINEL || next == a {
                // Safety: reached an unprocessed block or root. Return b.
                return b;
            }
            a = next;
        }
        while rpo_pos[b] > rpo_pos[a] {
            let next = idom[b];
            if next == SENTINEL || next == b {
                // Safety: reached an unprocessed block or root. Return a.
                return a;
            }
            b = next;
        }
    }
    a
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::BasicBlock as BB;
    use crate::ir::function::IrFunction;
    use crate::ir::types::IrType;

    /// Helper: build a minimal function with the given CFG edges.
    /// `edges` is a list of (from_block, to_block) pairs.
    /// Returns a function with `num_blocks` blocks.
    fn build_test_function(num_blocks: usize, edges: &[(usize, usize)]) -> IrFunction {
        let mut func = IrFunction::new("test".to_string(), Vec::new(), IrType::Void);

        // The constructor creates block 0 (entry). Add remaining blocks.
        for i in 1..num_blocks {
            let block = BB::new(i);
            func.add_block(block);
        }

        // Add CFG edges.
        for &(from, to) in edges {
            let blocks = func.blocks_mut();
            blocks[from].add_successor(to);
            blocks[to].add_predecessor(from);
        }

        func
    }

    #[test]
    fn test_single_block() {
        let func = build_test_function(1, &[]);
        let dt = DominatorTree::build(&func);
        assert_eq!(dt.block_count(), 1);
        assert_eq!(dt.idom(0), None); // Entry is root
        assert!(dt.dominates(0, 0)); // Reflexive
        assert!(!dt.strictly_dominates(0, 0));
        assert_eq!(dt.depth(0), 0);
        assert!(dt.validate(&func));
    }

    #[test]
    fn test_linear_chain() {
        // 0 → 1 → 2 → 3
        let func = build_test_function(4, &[(0, 1), (1, 2), (2, 3)]);
        let dt = DominatorTree::build(&func);

        assert_eq!(dt.idom(0), None);
        assert_eq!(dt.idom(1), Some(0));
        assert_eq!(dt.idom(2), Some(1));
        assert_eq!(dt.idom(3), Some(2));

        assert!(dt.dominates(0, 3));
        assert!(dt.dominates(1, 3));
        assert!(dt.dominates(2, 3));
        assert!(!dt.dominates(3, 0));

        assert!(dt.strictly_dominates(0, 3));
        assert!(!dt.strictly_dominates(3, 3));

        assert_eq!(dt.depth(0), 0);
        assert_eq!(dt.depth(1), 1);
        assert_eq!(dt.depth(2), 2);
        assert_eq!(dt.depth(3), 3);

        assert!(dt.validate(&func));
    }

    #[test]
    fn test_diamond() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3
        let func = build_test_function(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let dt = DominatorTree::build(&func);

        assert_eq!(dt.idom(0), None);
        assert_eq!(dt.idom(1), Some(0));
        assert_eq!(dt.idom(2), Some(0));
        assert_eq!(dt.idom(3), Some(0)); // 0 is the idom of the join point

        assert!(dt.dominates(0, 3));
        assert!(!dt.dominates(1, 3));
        assert!(!dt.dominates(2, 3));

        assert_eq!(dt.children(0).len(), 3); // 1, 2, 3

        assert!(dt.validate(&func));
    }

    #[test]
    fn test_self_loop() {
        // 0 → 1, 1 → 1 (self-loop), 1 → 2
        let func = build_test_function(3, &[(0, 1), (1, 1), (1, 2)]);
        let dt = DominatorTree::build(&func);

        assert_eq!(dt.idom(1), Some(0));
        assert_eq!(dt.idom(2), Some(1));
        assert!(dt.dominates(0, 2));
        assert!(dt.dominates(1, 2));

        assert!(dt.validate(&func));
    }

    #[test]
    fn test_loop_with_back_edge() {
        // 0 → 1, 1 → 2, 2 → 1 (back edge), 2 → 3
        let func = build_test_function(4, &[(0, 1), (1, 2), (2, 1), (2, 3)]);
        let dt = DominatorTree::build(&func);

        assert_eq!(dt.idom(1), Some(0));
        assert_eq!(dt.idom(2), Some(1));
        assert_eq!(dt.idom(3), Some(2));

        assert!(dt.validate(&func));
    }

    #[test]
    fn test_unreachable_block() {
        // 0 → 1, block 2 is unreachable
        let func = build_test_function(3, &[(0, 1)]);
        let dt = DominatorTree::build(&func);

        assert_eq!(dt.idom(0), None);
        assert_eq!(dt.idom(1), Some(0));
        assert_eq!(dt.idom(2), None); // Unreachable
        assert!(!dt.dominates(0, 2)); // Can't dominate unreachable
        assert!(!dt.dominates(2, 0)); // Unreachable can't dominate

        assert!(dt.validate(&func));
    }

    #[test]
    fn test_preorder() {
        // 0 → 1, 0 → 2, 1 → 3
        let func = build_test_function(4, &[(0, 1), (0, 2), (1, 3)]);
        let dt = DominatorTree::build(&func);

        let pre = dt.preorder();
        // Root (0) must be first.
        assert_eq!(pre[0], 0);
        // 3's parent is 1, so 1 must appear before 3.
        let pos_1 = pre.iter().position(|&x| x == 1).unwrap();
        let pos_3 = pre.iter().position(|&x| x == 3).unwrap();
        assert!(pos_1 < pos_3);
        // All reachable blocks present.
        assert_eq!(pre.len(), 4);

        assert!(dt.validate(&func));
    }

    #[test]
    fn test_build_simple_matches_lt() {
        // Complex CFG: 0→1, 0→2, 1→3, 2→3, 3→4, 3→5, 4→6, 5→6
        let edges = &[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
        ];
        let func = build_test_function(7, edges);
        let dt_lt = DominatorTree::build(&func);
        let dt_chk = DominatorTree::build_simple(&func);

        // Both algorithms should agree on all idom values.
        for i in 0..7 {
            assert_eq!(
                dt_lt.idom(i),
                dt_chk.idom(i),
                "idom mismatch for block {}",
                i
            );
        }

        assert!(dt_lt.validate(&func));
        assert!(dt_chk.validate(&func));
    }

    #[test]
    fn test_store_idom_in_blocks() {
        let edges = &[(0, 1), (0, 2), (1, 3), (2, 3)];
        let mut func = build_test_function(4, edges);
        let dt = DominatorTree::build(&func);

        dt.store_idom_in_blocks(&mut func);

        // Verify the idom values were stored in the blocks.
        assert_eq!(func.blocks()[1].idom(), Some(0));
        assert_eq!(func.blocks()[2].idom(), Some(0));
        assert_eq!(func.blocks()[3].idom(), Some(0));
    }

    #[test]
    fn test_dump_does_not_panic() {
        let func = build_test_function(3, &[(0, 1), (1, 2)]);
        let dt = DominatorTree::build(&func);
        let dump = dt.dump();
        assert!(dump.contains("DominatorTree"));
        assert!(dump.contains("block_count: 3"));
    }

    #[test]
    fn test_empty_function() {
        // Create a function — IrFunction::new creates one entry block.
        let func = IrFunction::new("empty".to_string(), Vec::new(), IrType::Void);
        // IrFunction::new creates one entry block, so block_count = 1.
        let dt = DominatorTree::build(&func);
        assert_eq!(dt.block_count(), 1);
        assert_eq!(dt.idom(0), None);
        assert!(dt.validate(&func));
    }

    #[test]
    fn test_reverse_postorder() {
        let func = build_test_function(4, &[(0, 1), (1, 2), (2, 3)]);
        let dt = DominatorTree::build(&func);
        let rpo = dt.reverse_postorder();
        // Entry must be first.
        assert_eq!(rpo[0], 0);
        // In a linear chain, RPO = 0, 1, 2, 3.
        assert_eq!(rpo, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_complex_cfg_with_nested_loops() {
        // Outer loop: 0→1, 1→2, 2→1 (back), 2→3
        // Inner loop in 1: 1→4, 4→1 (back)
        // Exit: 3→5
        let edges = &[(0, 1), (1, 2), (2, 1), (2, 3), (1, 4), (4, 1), (3, 5)];
        let func = build_test_function(6, edges);
        let dt = DominatorTree::build(&func);

        // 0 dominates everything.
        for i in 0..6 {
            assert!(dt.dominates(0, i), "0 should dominate {}", i);
        }
        // 1 dominates 2, 3, 4, 5 (loop header).
        assert!(dt.dominates(1, 2));
        assert!(dt.dominates(1, 4));

        assert!(dt.validate(&func));
    }
}
