//! Pass scheduling and execution framework for the BCC compiler's Phase 8
//! optimization pipeline.
//!
//! This module orchestrates a fixed-order pipeline of optimization passes that
//! runs iteratively until a fixpoint is reached (no pass makes changes):
//!
//! **constant folding → dead code elimination → CFG simplification**
//!
//! # Architecture
//!
//! The [`PassManager`] is the entry point called by the main compilation
//! pipeline between SSA construction (Phase 7 — mem2reg) and phi elimination
//! (Phase 9). It accepts an [`IrModule`] and runs all registered passes on
//! each function.
//!
//! # Fixpoint Iteration
//!
//! Passes compose synergistically: constant folding may expose dead code,
//! dead code elimination may expose redundant branches, and CFG
//! simplification may expose further constant-folding opportunities. The
//! fixpoint loop runs until:
//!
//! 1. No pass reports any changes during an iteration, **or**
//! 2. The `max_iterations` limit is reached (default: 10).
//!
//! # Pass Ordering
//!
//! The pipeline ordering is **fixed and intentional**:
//!
//! | Position | Pass               | Rationale                                       |
//! |----------|--------------------|-------------------------------------------------|
//! | 1        | Constant folding   | Produces constants that DCE and CFG simplify    |
//! | 2        | DCE                | Removes code exposed as dead by constant fold   |
//! | 3        | CFG simplification | Cleans up branches exposed by folding and DCE    |
//!
//! # SSA Invariant Preservation
//!
//! All passes operate on SSA-form IR (with phi nodes) and must preserve
//! SSA invariants. The phi-elimination pass (Phase 9) runs **after** the
//! pass manager completes.
//!
//! # Error Handling
//!
//! Optimization passes should not produce errors — they only simplify
//! correct IR. If a pass encounters unexpected IR patterns, it leaves them
//! unchanged (conservative approach). No diagnostic emission occurs from
//! optimization passes.

use crate::ir::function::IrFunction;
use crate::ir::module::IrModule;
use crate::passes::{constant_folding, dead_code_elimination, simplify_cfg};

// ---------------------------------------------------------------------------
// Pass enum
// ---------------------------------------------------------------------------

/// Registered optimization pass types.
///
/// Each variant dispatches to the corresponding pass function. The pass
/// manager iterates over a `Vec<Pass>` and calls each in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// Phase 8a: Evaluate compile-time-constant operations, fold branches
    /// with known conditions, propagate constants through SSA chains.
    ConstantFolding,

    /// Phase 8b: Remove instructions with unused results and no side
    /// effects; remove unreachable basic blocks.
    DeadCodeElimination,

    /// Phase 8c: Merge blocks with single predecessor/successor, eliminate
    /// empty blocks, simplify branch chains, and thread branches.
    SimplifyCFG,
}

impl Pass {
    /// Runs this pass on a single function.
    ///
    /// Returns `true` if the pass made any changes to the function.
    fn run(&self, func: &mut IrFunction) -> bool {
        match self {
            Pass::ConstantFolding => constant_folding::run_constant_folding(func),
            Pass::DeadCodeElimination => dead_code_elimination::run_dead_code_elimination(func),
            Pass::SimplifyCFG => simplify_cfg::run_simplify_cfg(func),
        }
    }

    /// Returns the human-readable name of this pass (for diagnostics/logging).
    #[allow(dead_code)]
    fn name(&self) -> &'static str {
        match self {
            Pass::ConstantFolding => "ConstantFolding",
            Pass::DeadCodeElimination => "DeadCodeElimination",
            Pass::SimplifyCFG => "SimplifyCFG",
        }
    }
}

// ---------------------------------------------------------------------------
// PassStats — aggregate statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics collected during pass execution.
///
/// Tracks the amount of work performed across all functions in the module.
/// Useful for debugging, performance tuning, and verification that the
/// optimization pipeline is effective.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Number of functions processed by the pass manager.
    pub functions_processed: usize,

    /// Total number of fixpoint iterations across all functions.
    pub total_iterations: usize,

    /// Number of instructions removed (approximate — measured by change flags).
    pub instructions_removed: usize,

    /// Number of basic blocks removed (approximate).
    pub blocks_removed: usize,

    /// Number of constants folded (approximate).
    pub constants_folded: usize,

    /// Number of branches simplified (approximate).
    pub branches_simplified: usize,
}

impl PassStats {
    /// Creates a new, zeroed statistics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if any changes were made during pass execution.
    pub fn has_changes(&self) -> bool {
        self.instructions_removed > 0
            || self.blocks_removed > 0
            || self.constants_folded > 0
            || self.branches_simplified > 0
    }
}

// ---------------------------------------------------------------------------
// PassManager
// ---------------------------------------------------------------------------

/// The optimization pass manager and scheduler.
///
/// Owns the registered passes, controls fixpoint iteration, and collects
/// aggregate statistics. The standard usage is:
///
/// ```ignore
/// let mut pm = PassManager::default_pipeline();
/// let stats = pm.run_on_module(&mut ir_module);
/// ```
pub struct PassManager {
    /// Maximum number of fixpoint iterations per function.
    ///
    /// If the pipeline has not converged (no changes) within this many
    /// iterations, we stop to avoid pathological cases. A value of 10
    /// is sufficient for virtually all real-world functions.
    max_iterations: usize,

    /// Registered passes in execution order.
    passes: Vec<Pass>,

    /// Cumulative statistics from the most recent `run_on_module` invocation.
    stats: PassStats,
}

impl PassManager {
    // ----- Construction -------------------------------------------------

    /// Creates a new `PassManager` with the default fixed pipeline and
    /// a default iteration limit of 10.
    ///
    /// The default pipeline is:
    /// 1. `ConstantFolding`
    /// 2. `DeadCodeElimination`
    /// 3. `SimplifyCFG`
    pub fn new() -> Self {
        Self::default_pipeline()
    }

    /// Creates the standard optimization pipeline.
    ///
    /// This is the pipeline used for `-O0` and above. Even at `-O0`, basic
    /// cleanup is valuable for code quality (removing trivially dead code
    /// and simplifying the CFG).
    pub fn default_pipeline() -> Self {
        Self {
            max_iterations: 10,
            passes: vec![
                Pass::ConstantFolding,
                Pass::DeadCodeElimination,
                Pass::SimplifyCFG,
            ],
            stats: PassStats::new(),
        }
    }

    /// Creates a `PassManager` with a custom fixpoint iteration limit.
    ///
    /// # Parameters
    ///
    /// - `max`: Maximum number of fixpoint iterations per function.
    ///          Must be at least 1.
    pub fn with_max_iterations(max: usize) -> Self {
        let max = if max == 0 { 1 } else { max };
        Self {
            max_iterations: max,
            passes: vec![
                Pass::ConstantFolding,
                Pass::DeadCodeElimination,
                Pass::SimplifyCFG,
            ],
            stats: PassStats::new(),
        }
    }

    // ----- Pass registration --------------------------------------------

    /// Adds a pass to the end of the pipeline.
    ///
    /// This allows future extension with additional passes without changing
    /// the core framework.
    pub fn add_pass(&mut self, pass: Pass) {
        self.passes.push(pass);
    }

    /// Removes all registered passes.
    ///
    /// Useful for testing or for building a custom pipeline from scratch.
    pub fn clear_passes(&mut self) {
        self.passes.clear();
    }

    // ----- Execution ----------------------------------------------------

    /// Runs all registered passes on every function in the module until
    /// each function reaches fixpoint.
    ///
    /// Returns aggregate statistics for the entire module.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// for each function in module.functions:
    ///     run_on_function(function)
    ///     update aggregate stats
    /// return stats
    /// ```
    pub fn run_on_module(&mut self, module: &mut IrModule) -> PassStats {
        self.stats = PassStats::new();

        for func in module.functions.iter_mut() {
            self.run_on_function(func);
            self.stats.functions_processed += 1;
        }

        self.stats.clone()
    }

    /// Runs the fixpoint optimization loop on a single function.
    ///
    /// Returns `true` if any pass made changes during any iteration.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// for iteration in 0..max_iterations:
    ///     changed_this_iter = false
    ///     for pass in passes:
    ///         changed_this_iter |= pass.run(func)
    ///     if !changed_this_iter:
    ///         break  // fixpoint reached
    /// ```
    pub fn run_on_function(&mut self, func: &mut IrFunction) -> bool {
        let mut changed_overall = false;

        for _iteration in 0..self.max_iterations {
            let mut changed_this_iter = false;

            for pass in &self.passes {
                let pass_changed = pass.run(func);
                changed_this_iter |= pass_changed;

                // Update approximate statistics based on which pass ran.
                if pass_changed {
                    match pass {
                        Pass::ConstantFolding => {
                            self.stats.constants_folded += 1;
                        }
                        Pass::DeadCodeElimination => {
                            self.stats.instructions_removed += 1;
                        }
                        Pass::SimplifyCFG => {
                            self.stats.branches_simplified += 1;
                        }
                    }
                }
            }

            self.stats.total_iterations += 1;
            changed_overall |= changed_this_iter;

            if !changed_this_iter {
                break; // Fixpoint reached — no pass made changes.
            }
        }

        changed_overall
    }

    // ----- Accessors ----------------------------------------------------

    /// Returns the collected statistics from the most recent run.
    pub fn stats(&self) -> &PassStats {
        &self.stats
    }

    /// Returns the current maximum iteration count.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns a reference to the registered passes.
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}
