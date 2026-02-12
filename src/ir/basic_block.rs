//! Basic block representation for the BCC IR control flow graph.
//!
//! A [`BasicBlock`] is the fundamental unit of control flow in the
//! intermediate representation. Each basic block satisfies two invariants:
//!
//! - **Single entry:** Execution always begins at the first instruction.
//! - **Single exit:** The terminator instruction (branch, return, or switch)
//!   is the only way to leave the block.
//!
//! # Control Flow Graph (CFG)
//!
//! Basic blocks are connected by directed predecessor/successor edges,
//! forming a CFG. These edge sets are maintained explicitly so that every
//! consumer — lowering, mem2reg, optimization passes, and the backend —
//! can traverse the graph efficiently without recomputing edges.
//!
//! # Dominator Tree Support
//!
//! Each block carries lazily-populated dominator tree fields used during
//! SSA construction (Phase 7, mem2reg):
//!
//! | Field             | Populated by                                     |
//! |-------------------|--------------------------------------------------|
//! | `idom`            | Lengauer-Tarjan in `mem2reg::dominator_tree`      |
//! | `dom_frontier`    | `mem2reg::dominance_frontier`                     |
//! | `dom_tree_children` | Dominator tree builder                          |
//!
//! These are `None`/empty until the dominator analysis pass runs.
//!
//! # Phi Node Invariant
//!
//! [`Instruction::Phi`] nodes must always appear as a contiguous prefix
//! at the start of the instruction list, before any non-phi instruction.
//! The [`phi_nodes()`](BasicBlock::phi_nodes) and
//! [`phi_count()`](BasicBlock::phi_count) methods rely on this ordering.
//!
//! # Module Re-exports
//!
//! [`BasicBlockId`] is defined in [`crate::ir::instructions`] (because
//! instruction variants such as `Branch`, `CondBranch`, and `Phi`
//! reference block IDs directly) and re-exported from this module for
//! ergonomic access by downstream consumers that work primarily with
//! basic blocks.

use std::fmt;

// Import the Instruction enum (element type for the instruction list)
// and BasicBlockId (the block identifier newtype).
use crate::ir::instructions::Instruction;

/// Re-export [`BasicBlockId`] from this module so that consumers working
/// with basic blocks can import it from either `crate::ir::instructions`
/// or `crate::ir::basic_block` — whichever is more natural in context.
pub use crate::ir::instructions::BasicBlockId;

// ---------------------------------------------------------------------------
// BasicBlock — core struct
// ---------------------------------------------------------------------------

/// A basic block in the BCC intermediate representation.
///
/// Each `BasicBlock` owns an ordered [`Vec<Instruction>`] and maintains
/// bookkeeping for CFG edges and dominator tree information. The struct
/// is `Clone + Debug` to support IR cloning during optimization and
/// diagnostic dumps.
///
/// # Instruction Ordering
///
/// ```text
/// ┌──────────────────────────────────┐
/// │  Phi nodes (0 or more)           │  ← phi_nodes() / phi_count()
/// ├──────────────────────────────────┤
/// │  Regular instructions (0 or more)│
/// ├──────────────────────────────────┤
/// │  Terminator (exactly 1)          │  ← terminator()
/// └──────────────────────────────────┘
/// ```
///
/// During construction the terminator may be absent; a well-formed block
/// must have one before code generation.
///
/// # Example (IR textual form)
///
/// ```text
/// bb0:    ; preds: bb2, bb3
///     %v5 = phi i32, [%v1, bb2], [%v3, bb3]
///     %v6 = add i32, %v5, %v4
///     br bb1
///     ; succs: bb1
/// ```
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /// Unique identifier for this block within the containing function.
    pub id: BasicBlockId,

    /// Optional human-readable label for debugging and IR dump display.
    ///
    /// When present, used in `Display` output instead of the numeric ID.
    /// Typical values: `"entry"`, `"if.then"`, `"while.body"`, `"return"`.
    pub name: Option<String>,

    /// Ordered list of instructions in this block.
    ///
    /// **Invariants maintained by mutation methods:**
    /// - Phi nodes (if any) form a contiguous prefix.
    /// - At most one terminator instruction, always the last element.
    instructions: Vec<Instruction>,

    /// Blocks that transfer control *to* this block (incoming CFG edges).
    predecessors: Vec<BasicBlockId>,

    /// Blocks that this block may transfer control *to* (outgoing CFG edges).
    successors: Vec<BasicBlockId>,

    /// Index into `instructions` of the terminator instruction.
    ///
    /// `None` while the block is under construction and no terminator has
    /// been added yet. When `Some(idx)`, `instructions[idx].is_terminator()`
    /// is guaranteed to be `true`.
    terminator: Option<usize>,

    /// Immediate dominator in the dominator tree.
    ///
    /// `None` for the entry block (which has no dominator) or before the
    /// dominator tree has been computed by the Lengauer-Tarjan algorithm.
    idom: Option<BasicBlockId>,

    /// Dominance frontier — the set of blocks where this block's dominance
    /// "ends" and phi nodes may be needed.
    ///
    /// Empty until [`crate::ir::mem2reg::dominance_frontier`] computes it.
    dom_frontier: Vec<BasicBlockId>,

    /// Children of this block in the dominator tree.
    ///
    /// Block C is a child of this block in the dominator tree iff this
    /// block is C's immediate dominator. Empty until the tree is built.
    dom_tree_children: Vec<BasicBlockId>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl BasicBlock {
    /// Creates a new, empty basic block.
    ///
    /// The block starts with no instructions, no CFG edges, and no
    /// dominator tree information. Populate it incrementally using the
    /// instruction-management and edge-management methods.
    ///
    /// # Arguments
    ///
    /// * `id`   — Unique block identifier within the containing function.
    /// * `name` — Optional label for debug display. Pass `None` for
    ///   anonymous blocks (the numeric ID is used in output).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let entry = BasicBlock::new(BasicBlockId(0), Some("entry".into()));
    /// let bb1   = BasicBlock::new(BasicBlockId(1), None);
    /// ```
    pub fn new(id: BasicBlockId, name: Option<String>) -> Self {
        BasicBlock {
            id,
            name,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
            terminator: None,
            idom: None,
            dom_frontier: Vec::new(),
            dom_tree_children: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction management
// ---------------------------------------------------------------------------

impl BasicBlock {
    /// Appends an instruction to the end of this block's instruction list.
    ///
    /// If the instruction satisfies [`Instruction::is_terminator()`], the
    /// internal `terminator` index is updated to point to it.
    ///
    /// # Arguments
    ///
    /// * `inst` — The instruction to append.
    pub fn add_instruction(&mut self, inst: Instruction) {
        if inst.is_terminator() {
            self.terminator = Some(self.instructions.len());
        }
        self.instructions.push(inst);
    }

    /// Inserts an instruction at position `index`, shifting all subsequent
    /// instructions one position to the right.
    ///
    /// The existing terminator index (if any) is adjusted when the
    /// insertion point is at or before it. If the inserted instruction
    /// is itself a terminator, the `terminator` field points to `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.instructions.len()`.
    pub fn insert_instruction(&mut self, index: usize, inst: Instruction) {
        let is_new_terminator = inst.is_terminator();
        self.instructions.insert(index, inst);

        // Adjust the existing terminator index if it was shifted right.
        if let Some(ref mut term_idx) = self.terminator {
            if !is_new_terminator && index <= *term_idx {
                *term_idx += 1;
            }
        }

        if is_new_terminator {
            self.terminator = Some(index);
        }
    }

    /// Removes and returns the instruction at `index`, shifting all
    /// subsequent instructions one position to the left.
    ///
    /// If the removed instruction was the terminator, the `terminator`
    /// field is cleared. If the removal is *before* the terminator, its
    /// tracked index is decremented.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.instructions.len()`.
    pub fn remove_instruction(&mut self, index: usize) -> Instruction {
        let inst = self.instructions.remove(index);

        match self.terminator {
            Some(term_idx) if term_idx == index => {
                // The terminator itself was removed.
                self.terminator = None;
            }
            Some(ref mut term_idx) if index < *term_idx => {
                // Removed before the terminator — adjust index.
                *term_idx -= 1;
            }
            _ => {}
        }

        inst
    }

    /// Replaces the instruction at `index` with `inst`, returning nothing.
    ///
    /// Terminator tracking is updated:
    /// - If the old instruction was the terminator and the new one is not,
    ///   the `terminator` field is cleared.
    /// - If the new instruction is a terminator, the `terminator` field
    ///   is set to `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.instructions.len()`.
    pub fn replace_instruction(&mut self, index: usize, inst: Instruction) {
        let was_terminator = self.terminator == Some(index);
        let is_terminator = inst.is_terminator();

        self.instructions[index] = inst;

        if was_terminator && !is_terminator {
            self.terminator = None;
        } else if is_terminator {
            self.terminator = Some(index);
        }
    }

    /// Returns an immutable slice of all instructions in program order.
    ///
    /// The slice honours the phi-first invariant: phi nodes appear at the
    /// front, followed by regular instructions, with the terminator last.
    #[inline]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Returns a mutable slice of all instructions.
    ///
    /// **Callers must preserve the phi-first invariant** and must not
    /// relocate the terminator without calling
    /// [`set_terminator()`](BasicBlock::set_terminator) to keep the
    /// internal index consistent.
    #[inline]
    pub fn instructions_mut(&mut self) -> &mut [Instruction] {
        &mut self.instructions
    }

    /// Returns a reference to the terminator instruction, or `None` if
    /// the block has no terminator yet (block under construction).
    ///
    /// A well-formed block always has a terminator; `None` is valid only
    /// during incremental IR construction.
    #[inline]
    pub fn terminator(&self) -> Option<&Instruction> {
        self.terminator.map(|idx| &self.instructions[idx])
    }

    /// Sets or replaces the block's terminator instruction.
    ///
    /// - If the block already has a terminator it is replaced **in-place**
    ///   at the same index.
    /// - If there is no terminator the instruction is **appended** and
    ///   the index recorded.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `inst` is actually a terminator
    /// ([`Instruction::is_terminator()`] returns `true`).
    pub fn set_terminator(&mut self, inst: Instruction) {
        debug_assert!(
            inst.is_terminator(),
            "set_terminator: instruction is not a terminator: {:?}",
            inst,
        );

        if let Some(term_idx) = self.terminator {
            // Replace existing terminator in-place.
            self.instructions[term_idx] = inst;
        } else {
            // No existing terminator — append and record.
            self.terminator = Some(self.instructions.len());
            self.instructions.push(inst);
        }
    }

    /// Returns `true` if this block has a terminator instruction.
    ///
    /// A well-formed block always has a terminator. During construction
    /// the block may temporarily lack one.
    #[inline]
    pub fn has_terminator(&self) -> bool {
        self.terminator.is_some()
    }

    /// Returns `true` if the block contains no instructions at all —
    /// no phi nodes, no regular instructions, and no terminator.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CFG edge management
// ---------------------------------------------------------------------------

impl BasicBlock {
    /// Adds a predecessor edge — records that block `pred` may transfer
    /// control to this block.
    ///
    /// Duplicate entries are permitted (e.g., when a `CondBranch` has
    /// both targets pointing here).
    #[inline]
    pub fn add_predecessor(&mut self, pred: BasicBlockId) {
        self.predecessors.push(pred);
    }

    /// Adds a successor edge — records that this block may transfer
    /// control to block `succ`.
    ///
    /// Duplicate entries are permitted.
    #[inline]
    pub fn add_successor(&mut self, succ: BasicBlockId) {
        self.successors.push(succ);
    }

    /// Removes the **first** occurrence of `pred` from the predecessor
    /// list.
    ///
    /// If `pred` is not present the call is a no-op. Removing only the
    /// first occurrence preserves correct edge multiplicity when a block
    /// has multiple edges from the same predecessor.
    pub fn remove_predecessor(&mut self, pred: BasicBlockId) {
        if let Some(pos) = self.predecessors.iter().position(|&p| p == pred) {
            self.predecessors.remove(pos);
        }
    }

    /// Removes the **first** occurrence of `succ` from the successor
    /// list.
    ///
    /// If `succ` is not present the call is a no-op.
    pub fn remove_successor(&mut self, succ: BasicBlockId) {
        if let Some(pos) = self.successors.iter().position(|&s| s == succ) {
            self.successors.remove(pos);
        }
    }

    /// Returns an immutable slice of predecessor block IDs.
    #[inline]
    pub fn predecessors(&self) -> &[BasicBlockId] {
        &self.predecessors
    }

    /// Returns an immutable slice of successor block IDs.
    #[inline]
    pub fn successors(&self) -> &[BasicBlockId] {
        &self.successors
    }

    /// Returns `true` if `pred` appears in this block's predecessor list.
    #[inline]
    pub fn has_predecessor(&self, pred: BasicBlockId) -> bool {
        self.predecessors.contains(&pred)
    }

    /// Returns the number of predecessor edges.
    ///
    /// Zero predecessors typically means this is the entry block or an
    /// unreachable block.
    #[inline]
    pub fn predecessor_count(&self) -> usize {
        self.predecessors.len()
    }

    /// Returns the number of successor edges.
    ///
    /// Zero successors typically means this block ends with a `Return`.
    #[inline]
    pub fn successor_count(&self) -> usize {
        self.successors.len()
    }
}

// ---------------------------------------------------------------------------
// Dominator tree fields
// ---------------------------------------------------------------------------

impl BasicBlock {
    /// Sets the immediate dominator of this block in the dominator tree.
    ///
    /// Populated by the Lengauer-Tarjan algorithm in
    /// [`crate::ir::mem2reg::dominator_tree`].
    #[inline]
    pub fn set_idom(&mut self, idom: BasicBlockId) {
        self.idom = Some(idom);
    }

    /// Returns the immediate dominator, or `None` if not yet computed
    /// or if this is the entry block.
    #[inline]
    pub fn idom(&self) -> Option<BasicBlockId> {
        self.idom
    }

    /// Adds `block` to this block's dominance frontier.
    ///
    /// The dominance frontier of block A is the set of blocks B where A
    /// dominates a predecessor of B but does not strictly dominate B.
    /// Phi nodes are placed at dominance frontier boundaries during SSA
    /// construction.
    #[inline]
    pub fn add_dom_frontier(&mut self, block: BasicBlockId) {
        self.dom_frontier.push(block);
    }

    /// Returns an immutable slice of this block's dominance frontier.
    ///
    /// Empty until computed by
    /// [`crate::ir::mem2reg::dominance_frontier`].
    #[inline]
    pub fn dom_frontier(&self) -> &[BasicBlockId] {
        &self.dom_frontier
    }

    /// Adds `child` to this block's dominator tree children.
    ///
    /// Block C is a dominator-tree child of this block iff this block is
    /// C's immediate dominator.
    #[inline]
    pub fn add_dom_child(&mut self, child: BasicBlockId) {
        self.dom_tree_children.push(child);
    }

    /// Returns an immutable slice of this block's dominator tree children.
    ///
    /// Empty until the dominator tree has been constructed.
    #[inline]
    pub fn dom_children(&self) -> &[BasicBlockId] {
        &self.dom_tree_children
    }
}

// ---------------------------------------------------------------------------
// Phi node access
// ---------------------------------------------------------------------------

impl BasicBlock {
    /// Returns an iterator over the phi nodes at the beginning of this
    /// block.
    ///
    /// By invariant, phi nodes form a contiguous prefix. The iterator
    /// yields [`Instruction`] references from the front and stops at the
    /// first non-phi instruction (or at the end if the block contains
    /// only phi nodes, which would be malformed but handled gracefully).
    pub fn phi_nodes(&self) -> impl Iterator<Item = &Instruction> {
        self.instructions.iter().take_while(|inst| inst.is_phi())
    }

    /// Returns the count of phi nodes at the beginning of this block.
    ///
    /// Equivalent to `self.phi_nodes().count()` but expressed as a
    /// standalone method for clarity at call sites.
    pub fn phi_count(&self) -> usize {
        self.instructions
            .iter()
            .take_while(|inst| inst.is_phi())
            .count()
    }
}

// ---------------------------------------------------------------------------
// Display implementation
// ---------------------------------------------------------------------------

impl fmt::Display for BasicBlock {
    /// Renders the basic block in a human-readable IR textual form.
    ///
    /// # Output Format
    ///
    /// ```text
    /// bb0:    ; preds: bb2, bb3
    ///     %v5 = phi i32, [%v1, bb2], [%v3, bb3]
    ///     %v6 = add i32, %v5, %v4
    ///     br bb1
    ///     ; succs: bb1
    /// ```
    ///
    /// - The block label is the `name` (if set) or the numeric ID.
    /// - Predecessor list is appended as a comment after the label.
    /// - Each instruction is indented with four spaces.
    /// - Successor list is appended as a trailing comment.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // -- Block header: label and predecessors --
        match &self.name {
            Some(label) => write!(f, "{}:", label)?,
            None => write!(f, "{}:", self.id)?,
        }

        if !self.predecessors.is_empty() {
            write!(f, "    ; preds:")?;
            for (i, pred) in self.predecessors.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, " {}", pred)?;
            }
        }
        writeln!(f)?;

        // -- Instructions (indented) --
        for inst in &self.instructions {
            writeln!(f, "    {}", inst)?;
        }

        // -- Successor list --
        if !self.successors.is_empty() {
            write!(f, "    ; succs:")?;
            for (i, succ) in self.successors.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, " {}", succ)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::{BasicBlockId, Instruction, ValueId};
    use crate::ir::types::IrType;

    /// Helper: create a simple `Branch` terminator to `target`.
    fn branch(target: BasicBlockId) -> Instruction {
        Instruction::Branch { target }
    }

    /// Helper: create a simple `Return` terminator.
    fn ret(value: Option<ValueId>) -> Instruction {
        Instruction::Return { value }
    }

    /// Helper: create an `Alloca` instruction.
    fn alloca(result: u32) -> Instruction {
        Instruction::Alloca {
            result: ValueId(result),
            ty: IrType::I32,
            alignment: 4,
        }
    }

    /// Helper: create a `Phi` instruction with given incoming edges.
    fn phi(result: u32, incoming: Vec<(u32, u32)>) -> Instruction {
        Instruction::Phi {
            result: ValueId(result),
            ty: IrType::I32,
            incoming: incoming
                .into_iter()
                .map(|(v, b)| (ValueId(v), BasicBlockId(b)))
                .collect(),
        }
    }

    // -- Construction -------------------------------------------------------

    #[test]
    fn new_block_is_empty() {
        let bb = BasicBlock::new(BasicBlockId(0), None);
        assert!(bb.is_empty());
        assert!(!bb.has_terminator());
        assert_eq!(bb.predecessor_count(), 0);
        assert_eq!(bb.successor_count(), 0);
        assert!(bb.idom().is_none());
        assert!(bb.dom_frontier().is_empty());
        assert!(bb.dom_children().is_empty());
        assert_eq!(bb.phi_count(), 0);
    }

    #[test]
    fn new_block_with_name() {
        let bb = BasicBlock::new(BasicBlockId(5), Some("entry".into()));
        assert_eq!(bb.id, BasicBlockId(5));
        assert_eq!(bb.name.as_deref(), Some("entry"));
    }

    // -- Instruction management ---------------------------------------------

    #[test]
    fn add_instruction_appends() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(alloca(1));
        assert_eq!(bb.instructions().len(), 2);
        assert!(!bb.is_empty());
    }

    #[test]
    fn add_terminator_tracks_index() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(branch(BasicBlockId(1)));
        assert!(bb.has_terminator());
        assert!(bb.terminator().unwrap().is_terminator());
    }

    #[test]
    fn insert_instruction_shifts_terminator() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(branch(BasicBlockId(1)));
        // Terminator is at index 1.  Insert before it.
        bb.insert_instruction(1, alloca(2));
        // Terminator should now be at index 2.
        assert_eq!(bb.instructions().len(), 3);
        assert!(bb.terminator().unwrap().is_terminator());
        // The instruction at index 1 should be the newly inserted alloca.
        assert!(bb.instructions()[1].is_alloca());
    }

    #[test]
    fn remove_instruction_clears_terminator() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(branch(BasicBlockId(1)));
        // Remove the terminator (index 1).
        let removed = bb.remove_instruction(1);
        assert!(removed.is_terminator());
        assert!(!bb.has_terminator());
    }

    #[test]
    fn remove_before_terminator_adjusts_index() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(alloca(1));
        bb.add_instruction(branch(BasicBlockId(1)));
        // Terminator at index 2.  Remove index 0.
        bb.remove_instruction(0);
        // Terminator should now be at index 1.
        assert!(bb.has_terminator());
        assert!(bb.terminator().unwrap().is_terminator());
        assert_eq!(bb.instructions().len(), 2);
    }

    #[test]
    fn replace_instruction_updates_terminator() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(branch(BasicBlockId(1)));
        // Replace the terminator with a different terminator.
        bb.replace_instruction(1, ret(None));
        assert!(bb.has_terminator());
        // Replace the terminator with a non-terminator → clears.
        bb.replace_instruction(1, alloca(2));
        assert!(!bb.has_terminator());
    }

    #[test]
    fn set_terminator_appends_when_missing() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.set_terminator(branch(BasicBlockId(1)));
        assert_eq!(bb.instructions().len(), 2);
        assert!(bb.terminator().unwrap().is_terminator());
    }

    #[test]
    fn set_terminator_replaces_existing() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.set_terminator(branch(BasicBlockId(1)));
        bb.set_terminator(ret(None));
        // Length unchanged — replacement was in-place.
        assert_eq!(bb.instructions().len(), 2);
        // Terminator is now a Return.
        match bb.terminator().unwrap() {
            Instruction::Return { .. } => {}
            other => panic!("expected Return, got {:?}", other),
        }
    }

    #[test]
    fn instructions_mut_allows_modification() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(alloca(1));
        let slice = bb.instructions_mut();
        assert_eq!(slice.len(), 2);
        // Mutation through slice is permitted.
        slice[0] = alloca(99);
        assert_eq!(bb.instructions()[0].result(), Some(ValueId(99)));
    }

    // -- CFG edge management ------------------------------------------------

    #[test]
    fn add_and_query_predecessors() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_predecessor(BasicBlockId(1));
        bb.add_predecessor(BasicBlockId(2));
        assert_eq!(bb.predecessor_count(), 2);
        assert!(bb.has_predecessor(BasicBlockId(1)));
        assert!(bb.has_predecessor(BasicBlockId(2)));
        assert!(!bb.has_predecessor(BasicBlockId(3)));
        assert_eq!(bb.predecessors(), &[BasicBlockId(1), BasicBlockId(2)]);
    }

    #[test]
    fn add_and_query_successors() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_successor(BasicBlockId(3));
        bb.add_successor(BasicBlockId(4));
        assert_eq!(bb.successor_count(), 2);
        assert_eq!(bb.successors(), &[BasicBlockId(3), BasicBlockId(4)]);
    }

    #[test]
    fn remove_predecessor_removes_first_only() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_predecessor(BasicBlockId(1));
        bb.add_predecessor(BasicBlockId(1)); // duplicate edge
        bb.remove_predecessor(BasicBlockId(1));
        assert_eq!(bb.predecessor_count(), 1);
        assert!(bb.has_predecessor(BasicBlockId(1)));
    }

    #[test]
    fn remove_successor_noop_when_absent() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_successor(BasicBlockId(1));
        bb.remove_successor(BasicBlockId(99)); // not present
        assert_eq!(bb.successor_count(), 1);
    }

    // -- Dominator tree fields ----------------------------------------------

    #[test]
    fn idom_default_none() {
        let bb = BasicBlock::new(BasicBlockId(0), None);
        assert!(bb.idom().is_none());
    }

    #[test]
    fn set_and_get_idom() {
        let mut bb = BasicBlock::new(BasicBlockId(1), None);
        bb.set_idom(BasicBlockId(0));
        assert_eq!(bb.idom(), Some(BasicBlockId(0)));
    }

    #[test]
    fn dom_frontier_operations() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_dom_frontier(BasicBlockId(2));
        bb.add_dom_frontier(BasicBlockId(3));
        assert_eq!(bb.dom_frontier(), &[BasicBlockId(2), BasicBlockId(3)]);
    }

    #[test]
    fn dom_children_operations() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_dom_child(BasicBlockId(1));
        bb.add_dom_child(BasicBlockId(4));
        assert_eq!(bb.dom_children(), &[BasicBlockId(1), BasicBlockId(4)]);
    }

    // -- Phi node access ----------------------------------------------------

    #[test]
    fn phi_count_with_no_phis() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(alloca(0));
        bb.add_instruction(branch(BasicBlockId(1)));
        assert_eq!(bb.phi_count(), 0);
        assert_eq!(bb.phi_nodes().count(), 0);
    }

    #[test]
    fn phi_count_with_phis() {
        let mut bb = BasicBlock::new(BasicBlockId(2), None);
        bb.add_instruction(phi(10, vec![(1, 0), (2, 1)]));
        bb.add_instruction(phi(11, vec![(3, 0), (4, 1)]));
        bb.add_instruction(alloca(12));
        bb.add_instruction(branch(BasicBlockId(3)));
        assert_eq!(bb.phi_count(), 2);
        let phis: Vec<_> = bb.phi_nodes().collect();
        assert_eq!(phis.len(), 2);
        assert!(phis[0].is_phi());
        assert!(phis[1].is_phi());
    }

    #[test]
    fn phi_nodes_empty_block() {
        let bb = BasicBlock::new(BasicBlockId(0), None);
        assert_eq!(bb.phi_count(), 0);
        assert_eq!(bb.phi_nodes().count(), 0);
    }

    // -- Display ------------------------------------------------------------

    #[test]
    fn display_unnamed_block() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_predecessor(BasicBlockId(1));
        bb.add_instruction(alloca(0));
        bb.add_instruction(branch(BasicBlockId(2)));
        bb.add_successor(BasicBlockId(2));

        let text = format!("{}", bb);
        assert!(text.contains("bb0:"));
        assert!(text.contains("; preds: bb1"));
        assert!(text.contains("; succs: bb2"));
    }

    #[test]
    fn display_named_block() {
        let bb = BasicBlock::new(BasicBlockId(0), Some("entry".into()));
        let text = format!("{}", bb);
        assert!(text.starts_with("entry:"));
    }

    #[test]
    fn display_no_preds_no_succs() {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        bb.add_instruction(ret(None));
        let text = format!("{}", bb);
        assert!(!text.contains("; preds:"));
        assert!(!text.contains("; succs:"));
    }
}
