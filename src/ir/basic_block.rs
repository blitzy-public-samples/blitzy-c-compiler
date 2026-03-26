//! # Basic Block Representation
//!
//! This module defines the [`BasicBlock`] type — the fundamental building block
//! of BCC's control flow graph (CFG).
//!
//! ## Overview
//!
//! A basic block is a straight-line sequence of IR instructions with exactly:
//! - **One entry point** — control flow enters only at the top of the block.
//! - **One exit point** — a *terminator* instruction (Branch, CondBranch,
//!   Switch, or Return) at the bottom that transfers control elsewhere.
//!
//! ## Control Flow Graph
//!
//! Basic blocks are the nodes of the CFG.  Each block maintains lists of
//! predecessor and successor block indices that represent the directed edges
//! of the graph.  These edges are critical for:
//! - **Dominance analysis** — the Lengauer-Tarjan algorithm traverses the CFG
//!   to compute the dominator tree, stored in the [`idom`](BasicBlock::idom)
//!   field.
//! - **SSA construction** — the mem2reg pass uses dominance frontiers (stored
//!   in [`dominance_frontier`](BasicBlock::dominance_frontier)) to determine
//!   where phi nodes must be placed.
//!
//! ## Phi Node Convention
//!
//! Phi instructions (`Instruction::Phi`) must always appear at the **beginning**
//! of a basic block, before any non-phi instructions.  The [`add_phi`](BasicBlock::add_phi)
//! method enforces this by inserting phi nodes after existing phi nodes but
//! before the first non-phi instruction.  The [`phi_instructions`](BasicBlock::phi_instructions)
//! method provides an iterator over exactly the leading phi nodes.
//!
//! ## Block Identification
//!
//! Blocks are identified by their index within the parent function's block list.
//! The [`index`](BasicBlock::index) field stores this position and is set when
//! the block is added to a function.  An optional [`label`](BasicBlock::label)
//! provides human-readable names for IR dumps (e.g., `"entry"`, `"if.then"`,
//! `"loop.header"`).
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::ir::instructions::Instruction` and the
//! Rust standard library — no external crates are used.

use crate::ir::instructions::Instruction;

// ===========================================================================
// BasicBlock — CFG node containing a sequence of IR instructions
// ===========================================================================

/// A basic block in the control flow graph (CFG).
///
/// Contains an ordered sequence of [`Instruction`]s terminated by exactly one
/// terminator instruction (Branch, CondBranch, Switch, or Return).  Maintains
/// predecessor/successor edge lists for CFG traversal and dominator tree
/// information for SSA construction.
///
/// # Invariants
///
/// - The last instruction (if the block is complete) **must** be a terminator.
/// - Non-terminator instructions **must not** appear after the terminator.
/// - Phi nodes **must** appear at the beginning of the block, before any
///   non-phi instruction.
/// - Predecessor and successor lists contain no duplicate entries.
///
/// # Example
///
/// ```ignore
/// use bcc::ir::basic_block::BasicBlock;
/// use bcc::ir::instructions::Instruction;
///
/// let mut entry = BasicBlock::with_label(0, "entry".to_string());
/// // ... add instructions ...
/// assert_eq!(entry.index, 0);
/// assert_eq!(entry.label.as_deref(), Some("entry"));
/// ```
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Optional label name for this block (e.g., `"entry"`, `"if.then"`,
    /// `"loop.header"`).
    ///
    /// Labels are purely for human readability in IR dumps and diagnostic
    /// messages.  They do not affect semantics — blocks are identified by
    /// their [`index`](BasicBlock::index) within the parent function.
    pub label: Option<String>,

    /// Ordered list of instructions in this block.
    ///
    /// Instruction order defines execution semantics — the first instruction
    /// executes first, the last instruction (the terminator) executes last
    /// and transfers control flow.  Phi nodes, if present, always occupy the
    /// leading positions before any non-phi instruction.
    pub instructions: Vec<Instruction>,

    /// Predecessor block indices — blocks whose terminators can transfer
    /// control *to* this block.
    ///
    /// Maintained as a duplicate-free list.  Predecessors are essential for
    /// phi node operand resolution (each phi incoming edge corresponds to
    /// exactly one predecessor) and dominance computation.
    pub predecessors: Vec<usize>,

    /// Successor block indices — blocks to which this block's terminator
    /// can transfer control.
    ///
    /// Maintained as a duplicate-free list.  Successors mirror the target
    /// information in the terminator instruction and are used for CFG
    /// traversal during optimization and analysis passes.
    pub successors: Vec<usize>,

    /// Index of this block within the parent function's block list.
    ///
    /// Set when the block is added to a function.  Used as the canonical
    /// block identifier throughout the IR pipeline, register allocator,
    /// and code generator.  Corresponds to [`BlockId`](crate::ir::instructions::BlockId)
    /// values used in terminator instructions and phi nodes.
    pub index: usize,

    /// Immediate dominator block index, computed by the Lengauer-Tarjan
    /// dominator tree algorithm in the mem2reg pass.
    ///
    /// `None` until dominance analysis has been run.  The entry block of
    /// a function has no immediate dominator (`None` even after analysis).
    /// For all other blocks, this points to the closest block that
    /// dominates this one on every path from the entry.
    pub idom: Option<usize>,

    /// Dominance frontier — the set of blocks where this block's dominance
    /// ends and phi nodes may need to be placed.
    ///
    /// Empty until the dominance frontier computation pass is run.  A block
    /// `Y` is in the dominance frontier of block `X` if `X` dominates a
    /// predecessor of `Y` but does not strictly dominate `Y` itself.
    ///
    /// This is the key data structure driving phi-node placement in the
    /// mem2reg SSA construction pass.
    pub dominance_frontier: Vec<usize>,
}

// ===========================================================================
// BasicBlock — construction
// ===========================================================================

impl BasicBlock {
    /// Creates a new empty basic block with the given index.
    ///
    /// The block is initialized with no instructions, no predecessors or
    /// successors, no label, and no dominance information.
    ///
    /// # Parameters
    ///
    /// - `index`: The block's position within its parent function's block list.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bb = BasicBlock::new(0);
    /// assert!(bb.is_empty());
    /// assert_eq!(bb.index, 0);
    /// assert!(bb.label.is_none());
    /// ```
    #[inline]
    pub fn new(index: usize) -> Self {
        BasicBlock {
            label: None,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
            index,
            idom: None,
            dominance_frontier: Vec::new(),
        }
    }

    /// Creates a new labeled basic block with the given index and label.
    ///
    /// Labels provide human-readable names for IR dumps and diagnostics
    /// (e.g., `"entry"`, `"if.then"`, `"while.cond"`, `"loop.header"`).
    /// They do not affect execution semantics.
    ///
    /// # Parameters
    ///
    /// - `index`: The block's position within its parent function's block list.
    /// - `label`: A descriptive name for the block.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bb = BasicBlock::with_label(0, "entry".to_string());
    /// assert_eq!(bb.label.as_deref(), Some("entry"));
    /// assert_eq!(bb.index, 0);
    /// ```
    #[inline]
    pub fn with_label(index: usize, label: String) -> Self {
        BasicBlock {
            label: Some(label),
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
            index,
            idom: None,
            dominance_frontier: Vec::new(),
        }
    }
}

// ===========================================================================
// BasicBlock — instruction management
// ===========================================================================

impl BasicBlock {
    /// Appends an instruction to the end of the block.
    ///
    /// In debug builds, this asserts that no instruction is added after a
    /// terminator — doing so would violate the basic block invariant that
    /// the terminator is always the last instruction.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug mode if the block already has a terminator instruction
    /// as its last instruction.  In release builds, the instruction is
    /// appended unconditionally for maximum performance.
    ///
    /// # Parameters
    ///
    /// - `inst`: The instruction to append.
    #[inline]
    pub fn push_instruction(&mut self, inst: Instruction) {
        debug_assert!(
            !self.has_terminator(),
            "Cannot push instruction after terminator in block {} (label: {:?})",
            self.index,
            self.label
        );
        self.instructions.push(inst);
    }

    /// Inserts an instruction at a specific position in the block.
    ///
    /// All instructions at `index` and beyond are shifted to the right.
    /// This is used by optimization passes that need to insert instructions
    /// mid-block (e.g., spill code insertion by the register allocator).
    ///
    /// # Panics
    ///
    /// Panics if `index > self.instructions.len()`.
    ///
    /// # Parameters
    ///
    /// - `index`: The position at which to insert (0-based).
    /// - `inst`: The instruction to insert.
    #[inline]
    pub fn insert_instruction(&mut self, index: usize, inst: Instruction) {
        self.instructions.insert(index, inst);
    }

    /// Pushes an alloca instruction into the entry block, ensuring it is
    /// placed before any terminator instruction.
    ///
    /// During IR lowering of GCC statement expressions containing ternary
    /// operators, the entry block may already have a terminator (e.g., a
    /// `CondBranch` from the first ternary) by the time a subsequent
    /// statement expression needs to allocate stack space.  Using the
    /// normal `push_instruction` would place the alloca *after* the
    /// terminator, corrupting the block structure.
    ///
    /// This method inserts before the terminator (if one exists) or
    /// appends to the end (if no terminator is present).
    #[inline]
    pub fn push_alloca(&mut self, inst: Instruction) {
        if self.has_terminator() {
            // Insert just before the terminator (last instruction).
            let pos = self.instructions.len() - 1;
            self.instructions.insert(pos, inst);
        } else {
            self.instructions.push(inst);
        }
    }

    /// Returns an immutable slice over all instructions in the block.
    ///
    /// Instructions are returned in execution order — phi nodes first
    /// (if any), then regular instructions, with the terminator last.
    #[inline]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Returns a mutable reference to the instruction vector.
    ///
    /// Provides direct mutable access for passes that need to rewrite
    /// instructions in place (e.g., SSA renaming in mem2reg, instruction
    /// lowering in code generation).
    ///
    /// # Safety Note
    ///
    /// Callers must maintain the basic block invariants:
    /// - Terminator remains the last instruction.
    /// - Phi nodes remain at the beginning.
    #[inline]
    pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
        &mut self.instructions
    }

    /// Returns the number of instructions in the block.
    ///
    /// Includes phi nodes, regular instructions, and the terminator
    /// (if present).
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` if the block contains no instructions.
    ///
    /// An empty block is typically an intermediate state during IR
    /// construction — every completed block must have at least a
    /// terminator instruction.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

// ===========================================================================
// BasicBlock — terminator access
// ===========================================================================

impl BasicBlock {
    /// Returns a reference to the terminator instruction, if present.
    ///
    /// The terminator is always the last instruction in a well-formed block.
    /// Returns `None` if the block is empty or if the last instruction is
    /// not a terminator (Branch, CondBranch, Switch, or Return).
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(term) = bb.terminator() {
    ///     // term is &Instruction — a Branch, CondBranch, Switch, or Return
    /// }
    /// ```
    #[inline]
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last().filter(|inst| inst.is_terminator())
    }

    /// Returns a mutable reference to the terminator instruction, if present.
    ///
    /// Enables in-place modification of the terminator, e.g., to retarget a
    /// branch during CFG simplification or to replace a conditional branch
    /// with an unconditional one after constant folding.
    #[inline]
    pub fn terminator_mut(&mut self) -> Option<&mut Instruction> {
        self.instructions
            .last_mut()
            .filter(|inst| inst.is_terminator())
    }

    /// Returns `true` if the block has a terminator instruction.
    ///
    /// Every completed basic block must have a terminator.  This method is
    /// used for validation during IR construction and by optimization passes
    /// that need to verify block well-formedness before analysis.
    #[inline]
    pub fn has_terminator(&self) -> bool {
        self.instructions
            .last()
            .map_or(false, |inst| inst.is_terminator())
    }

    /// Sets the terminator instruction for this block.
    ///
    /// If the block already has a terminator (the last instruction is a
    /// terminator), it is **replaced**.  Otherwise, the new terminator is
    /// **appended** to the end of the instruction list.
    ///
    /// This is used for control flow rewiring during optimization — for
    /// example, replacing a conditional branch with an unconditional one
    /// when the condition is a compile-time constant.
    ///
    /// # Parameters
    ///
    /// - `inst`: The new terminator instruction.  Should be one of Branch,
    ///   CondBranch, Switch, or Return.
    pub fn set_terminator(&mut self, inst: Instruction) {
        if self.has_terminator() {
            // Replace the existing terminator (the last instruction).
            let last_idx = self.instructions.len() - 1;
            self.instructions[last_idx] = inst;
        } else {
            // No existing terminator — append.
            self.instructions.push(inst);
        }
    }
}

// ===========================================================================
// BasicBlock — CFG edge management
// ===========================================================================

impl BasicBlock {
    /// Adds a predecessor block index if not already present.
    ///
    /// Predecessors are blocks whose terminators can branch to this block.
    /// Duplicate entries are silently ignored to maintain the invariant that
    /// each predecessor appears at most once.
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the predecessor block within the parent function.
    #[inline]
    pub fn add_predecessor(&mut self, block_index: usize) {
        if !self.predecessors.contains(&block_index) {
            self.predecessors.push(block_index);
        }
    }

    /// Adds a successor block index if not already present.
    ///
    /// Successors are blocks to which this block's terminator can transfer
    /// control.  Duplicate entries are silently ignored.
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the successor block within the parent function.
    #[inline]
    pub fn add_successor(&mut self, block_index: usize) {
        if !self.successors.contains(&block_index) {
            self.successors.push(block_index);
        }
    }

    /// Removes a predecessor block index.
    ///
    /// If the predecessor is not present, this is a no-op.  Used during
    /// CFG simplification when edges are removed (e.g., eliminating
    /// unreachable predecessors).
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the predecessor to remove.
    #[inline]
    pub fn remove_predecessor(&mut self, block_index: usize) {
        self.predecessors.retain(|&idx| idx != block_index);
    }

    /// Removes a successor block index.
    ///
    /// If the successor is not present, this is a no-op.  Used during
    /// CFG simplification when edges are removed (e.g., removing dead
    /// branch targets).
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the successor to remove.
    #[inline]
    pub fn remove_successor(&mut self, block_index: usize) {
        self.successors.retain(|&idx| idx != block_index);
    }

    /// Returns an immutable slice over predecessor block indices.
    ///
    /// Predecessors are blocks whose terminators can branch to this block.
    /// The order reflects insertion order via [`add_predecessor`](BasicBlock::add_predecessor).
    #[inline]
    pub fn predecessors(&self) -> &[usize] {
        &self.predecessors
    }

    /// Returns an immutable slice over successor block indices.
    ///
    /// Successors are blocks to which this block's terminator can transfer
    /// control.  The order reflects insertion order via [`add_successor`](BasicBlock::add_successor).
    #[inline]
    pub fn successors(&self) -> &[usize] {
        &self.successors
    }

    /// Returns the number of predecessor blocks.
    ///
    /// A block with zero predecessors is either the entry block of the
    /// function or an unreachable block.
    #[inline]
    pub fn predecessor_count(&self) -> usize {
        self.predecessors.len()
    }

    /// Returns the number of successor blocks.
    ///
    /// A block with zero successors has a `Return` terminator (exits the
    /// function).  Unconditional branches have 1 successor, conditional
    /// branches have 2, and switch instructions can have many.
    #[inline]
    pub fn successor_count(&self) -> usize {
        self.successors.len()
    }
}

// ===========================================================================
// BasicBlock — dominance information
// ===========================================================================

impl BasicBlock {
    /// Sets the immediate dominator for this block.
    ///
    /// The immediate dominator (idom) is the closest block that dominates
    /// this one on every path from the function entry.  Computed by the
    /// Lengauer-Tarjan algorithm in `src/ir/mem2reg/dominator_tree.rs`.
    ///
    /// # Parameters
    ///
    /// - `idom`: Index of the immediate dominator block within the parent function.
    #[inline]
    pub fn set_idom(&mut self, idom: usize) {
        self.idom = Some(idom);
    }

    /// Returns the immediate dominator block index, if computed.
    ///
    /// Returns `None` if dominance analysis has not been run yet, or for the
    /// function entry block (which has no dominator).
    #[inline]
    pub fn idom(&self) -> Option<usize> {
        self.idom
    }

    /// Sets the dominance frontier for this block.
    ///
    /// The dominance frontier of block X is the set of blocks Y where X
    /// dominates a predecessor of Y but does not strictly dominate Y.
    /// This determines where phi nodes must be placed during SSA construction.
    ///
    /// Computed by the dominance frontier algorithm in
    /// `src/ir/mem2reg/dominance_frontier.rs`.
    ///
    /// # Parameters
    ///
    /// - `frontier`: Vector of block indices in this block's dominance frontier.
    #[inline]
    pub fn set_dominance_frontier(&mut self, frontier: Vec<usize>) {
        self.dominance_frontier = frontier;
    }

    /// Returns an immutable slice over the dominance frontier block indices.
    ///
    /// Empty until the dominance frontier computation pass has been run.
    #[inline]
    pub fn dominance_frontier(&self) -> &[usize] {
        &self.dominance_frontier
    }
}

// ===========================================================================
// BasicBlock — phi node support
// ===========================================================================

impl BasicBlock {
    /// Returns an iterator over phi instructions at the start of the block.
    ///
    /// By convention, phi nodes always appear at the beginning of a basic
    /// block, before any non-phi instructions.  This iterator yields
    /// references to exactly those leading phi nodes, stopping at the first
    /// non-phi instruction.
    ///
    /// Used by the mem2reg pass for phi-node operand resolution and by
    /// the phi-elimination pass for converting phi nodes to copy operations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for phi in bb.phi_instructions() {
    ///     // phi is &Instruction::Phi { .. }
    /// }
    /// ```
    #[inline]
    pub fn phi_instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.instructions.iter().take_while(|inst| inst.is_phi())
    }

    /// Inserts a phi instruction at the correct position in the block.
    ///
    /// Phi nodes are always placed at the beginning of the block, after any
    /// existing phi nodes but before the first non-phi instruction.  This
    /// maintains the invariant that all phi nodes form a contiguous prefix
    /// of the instruction list.
    ///
    /// # Parameters
    ///
    /// - `phi`: The phi instruction to insert.  Should be an
    ///   `Instruction::Phi { .. }` variant.
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, asserts that the provided instruction is actually
    /// a phi node.
    pub fn add_phi(&mut self, phi: Instruction) {
        debug_assert!(
            phi.is_phi(),
            "add_phi called with non-phi instruction in block {} (label: {:?})",
            self.index,
            self.label
        );

        // Find the insertion point: after all existing phi nodes.
        let insert_pos = self
            .instructions
            .iter()
            .position(|inst| !inst.is_phi())
            .unwrap_or(self.instructions.len());

        self.instructions.insert(insert_pos, phi);
    }
}

// ===========================================================================
// Display implementation
// ===========================================================================

impl std::fmt::Display for BasicBlock {
    /// Formats the basic block in human-readable IR text form.
    ///
    /// Output includes the block label (or index if unlabeled), followed
    /// by each instruction on its own line with indentation.
    ///
    /// # Example output
    ///
    /// ```text
    /// bb0 (entry):
    ///   %0 = alloca i32
    ///   store i32 %1, ptr %0
    ///   br label %bb1
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Block header: label or index-based name.
        if let Some(ref label) = self.label {
            write!(f, "bb{} ({}):", self.index, label)?;
        } else {
            write!(f, "bb{}:", self.index)?;
        }
        writeln!(f)?;

        // Each instruction indented by two spaces.
        for inst in &self.instructions {
            writeln!(f, "  {}", inst)?;
        }

        Ok(())
    }
}
