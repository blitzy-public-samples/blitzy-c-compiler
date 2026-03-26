//! Pass Scheduling and Execution Framework
//!
//! The pass manager orchestrates BCC's Phase 8 optimization pipeline. It runs
//! a fixed-order sequence of optimization passes and iterates until a fixpoint
//! is reached (no pass reports any modifications).
//!
//! ## Pass Order (Fixed)
//!
//! 1. **Constant Folding** — Evaluate compile-time constant operations, fold
//!    conditional branches with known conditions, propagate constants
//! 2. **Dead Code Elimination** — Remove unused instructions without side effects,
//!    remove unreachable basic blocks
//! 3. **CFG Simplification** — Merge single-predecessor/successor block pairs,
//!    eliminate empty blocks, simplify branch chains, remove trivial phi nodes
//!
//! ## Iteration Strategy
//!
//! The three passes are run in sequence. If any pass reports a change, the entire
//! sequence is re-run from the beginning. This continues until a complete
//! iteration produces no changes (fixpoint), or a maximum iteration count is
//! reached (safety net to prevent infinite loops).
//!
//! ## Scope
//!
//! Only basic optimizations are implemented. Advanced optimizations (loop unrolling,
//! vectorization, function inlining, instruction scheduling) are explicitly out
//! of scope per the project requirements.
//!
//! ## Integration Points
//!
//! - **Input**: SSA-form IR from mem2reg (Phase 7) — the IR contains phi nodes
//!   and SSA-form instructions.
//! - **Output**: Optimized SSA-form IR — ready for phi elimination (Phase 9).
//! - **Invoked by**: `src/main.rs` compilation driver (or directly by the pipeline
//!   orchestrator).
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::passes::*`, `crate::ir::*`, and the Rust
//! standard library — no external crates are used, and no `crate::frontend` or
//! `crate::backend` imports exist.

use crate::ir::function::IrFunction;
use crate::ir::module::IrModule;
use crate::passes::adce::run_adce;
use crate::passes::constant_folding::run_constant_folding;
use crate::passes::copy_propagation::run_copy_propagation;
use crate::passes::dead_code_elimination::run_dead_code_elimination;
use crate::passes::gvn::run_gvn;
use crate::passes::instruction_combining::run_instruction_combining;
use crate::passes::licm::run_licm;
use crate::passes::peephole::run_peephole;
use crate::passes::register_coalescing::run_register_coalescing;
use crate::passes::sccp::run_sccp;
use crate::passes::simplify_cfg::run_simplify_cfg;
use crate::passes::strength_reduction::run_strength_reduction;
use crate::passes::tail_call::run_tail_call_optimization;

// ===========================================================================
// Configuration
// ===========================================================================

/// Maximum number of fixpoint iterations before giving up.
///
/// This is a safety net — in practice, most functions converge in 2–5
/// iterations because the basic optimizations (constant folding, DCE, CFG
/// simplification) expose a limited amount of new work per round.
///
/// If this limit is reached, it likely indicates a bug in one of the
/// passes (oscillating changes that never stabilize). The pass manager
/// will still return normally, but the function may not be fully optimized.
const MAX_ITERATIONS: usize = 100;

// ===========================================================================
// OptimizationStats — statistics from an optimization run
// ===========================================================================

/// Statistics collected during an optimization run.
///
/// Useful for debugging the compiler itself — for example, checking that
/// fixpoint convergence occurs within a reasonable number of iterations,
/// or measuring how many functions in a translation unit actually benefit
/// from optimization.
///
/// # Fields
///
/// - `iterations`: The maximum number of fixpoint iterations performed
///   across all functions processed in the module. This indicates the
///   "hardest" function — the one that required the most rounds to converge.
/// - `functions_optimized`: Number of functions where at least one
///   optimization was applied.
/// - `total_functions`: Total number of function definitions processed
///   (excludes declarations without bodies).
///
/// # Example
///
/// ```ignore
/// use bcc::passes::pass_manager::OptimizationStats;
///
/// let stats = OptimizationStats::default();
/// assert_eq!(stats.iterations, 0);
/// assert_eq!(stats.functions_optimized, 0);
/// assert_eq!(stats.total_functions, 0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OptimizationStats {
    /// Maximum number of fixpoint iterations performed on any single
    /// function during the optimization run.
    ///
    /// A value of 1 means all functions converged on the first round
    /// (i.e., the passes made changes in the first iteration, but the
    /// second iteration found no more work). A value of 0 means no
    /// functions were processed or no passes made any changes.
    pub iterations: usize,

    /// Number of function definitions where at least one optimization
    /// pass reported a modification.
    pub functions_optimized: usize,

    /// Total number of function definitions (with bodies) that were
    /// presented to the pass manager for optimization.
    pub total_functions: usize,
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OptimizationStats {{ iterations: {}, functions_optimized: {}/{} }}",
            self.iterations, self.functions_optimized, self.total_functions,
        )
    }
}

// ===========================================================================
// Per-function optimization
// ===========================================================================

/// Runs the basic cleanup pass sequence on a single function (used at `-O0`).
///
/// This minimal pipeline cleans up IR artifacts from lowering and mem2reg
/// without introducing aggressive optimizations that would impair debugging.
///
/// ## Pass Order (Fixed)
///
/// 1. **Constant Folding** — folds compile-time constant arithmetic,
///    propagates constants through use-def chains, simplifies conditional
///    branches with known conditions.
/// 2. **Dead Code Elimination** — removes instructions whose results are
///    unused and have no side effects; removes unreachable basic blocks.
/// 3. **CFG Simplification** — merges single-predecessor/successor blocks,
///    eliminates empty blocks, simplifies branch chains, removes trivial
///    phi nodes.
///
/// Iteration stops when a complete round of all three passes produces zero
/// changes (fixpoint), or when [`MAX_ITERATIONS`] is reached (safety net).
///
/// Returns `true` if any optimization was applied during any iteration.
pub fn optimize_function(func: &mut IrFunction) -> bool {
    optimize_function_at_level(func, 0)
}

/// Runs the full optimization pass sequence on a single function at the
/// given optimization level.
///
/// ## Pass Order at `-O1` and above (15 passes total)
///
/// 1.  **SCCP** — Sparse Conditional Constant Propagation (stronger than
///     simple constant folding — evaluates unreachable branches)
/// 2.  **Constant Folding** — Evaluate compile-time constant operations
/// 3.  **Copy Propagation** — Replace uses of `x = y` with `y`
/// 4.  **GVN** — Global Value Numbering (hash-based CSE)
/// 5.  **Instruction Combining** — Algebraic simplifications, cast chains
/// 6.  **Strength Reduction** — Replace mul-by-power-of-2 with shift, etc.
/// 7.  **LICM** — Loop-Invariant Code Motion
/// 8.  **DCE** — Dead Code Elimination
/// 9.  **ADCE** — Aggressive Dead Code Elimination (SSA-based)
/// 10. **CFG Simplification** — Block merging, branch threading
/// 11. **Tail Call Optimization** — Convert tail calls to jumps
/// 12. **Peephole** — Store-load forwarding, dead stores, branch folding
/// 13. **Register Coalescing** — Reduce phi/bitcast copy overhead
///
/// At `-O0`, only passes 2 (Constant Folding), 8 (DCE), and 10 (CFG
/// Simplification) run — this cleans up lowering artifacts without
/// aggressive transformations.
pub fn optimize_function_at_level(func: &mut IrFunction, opt_level: u8) -> bool {
    let mut any_changed = false;

    for _iteration in 0..MAX_ITERATIONS {
        let mut changed_this_round = false;

        if opt_level >= 1 {
            // SCCP: Sparse Conditional Constant Propagation — more powerful
            // than simple constant folding, evaluates branch reachability.
            changed_this_round |= run_sccp(func);
        }

        // Constant Folding — always runs (even at -O0 for cleanup).
        changed_this_round |= run_constant_folding(func);

        if opt_level >= 1 {
            // Copy Propagation — replace trivial copies (phi, bitcast).
            changed_this_round |= run_copy_propagation(func);

            // GVN — Global Value Numbering / CSE.
            changed_this_round |= run_gvn(func);

            // Instruction Combining — algebraic simplifications.
            changed_this_round |= run_instruction_combining(func);

            // Strength Reduction — power-of-2 mul→shl, udiv→lshr.
            changed_this_round |= run_strength_reduction(func);

            // LICM — Loop-Invariant Code Motion.
            changed_this_round |= run_licm(func);
        }

        // DCE — always runs (even at -O0 for cleanup).
        changed_this_round |= run_dead_code_elimination(func);

        if opt_level >= 1 {
            // ADCE — Aggressive Dead Code Elimination (SSA reverse-reach).
            changed_this_round |= run_adce(func);
        }

        // CFG Simplification — always runs (even at -O0 for cleanup).
        changed_this_round |= run_simplify_cfg(func);

        if opt_level >= 1 {
            // Tail Call Optimization — convert tail-position calls to jumps.
            changed_this_round |= run_tail_call_optimization(func);

            // Peephole — store-load forwarding, dead stores, branch folding.
            changed_this_round |= run_peephole(func);

            // Register Coalescing — reduce phi/bitcast copy overhead.
            changed_this_round |= run_register_coalescing(func);
        }

        if !changed_this_round {
            // Fixpoint reached — no pass made any changes this round.
            break;
        }

        any_changed = true;
    }

    any_changed
}

/// Internal helper: runs the optimization pass sequence on a single function
/// and returns both a changed flag and the number of iterations performed.
///
/// This is used by [`optimize_module_with_stats`] to collect per-function
/// iteration counts for the [`OptimizationStats`] aggregate.
#[allow(dead_code)]
fn optimize_function_counted(func: &mut IrFunction, opt_level: u8) -> (bool, usize) {
    let mut any_changed = false;
    let mut iterations_performed: usize = 0;

    for _iteration in 0..MAX_ITERATIONS {
        let mut changed_this_round = false;

        if opt_level >= 1 {
            changed_this_round |= run_sccp(func);
        }
        changed_this_round |= run_constant_folding(func);
        if opt_level >= 1 {
            changed_this_round |= run_copy_propagation(func);
            changed_this_round |= run_gvn(func);
            changed_this_round |= run_instruction_combining(func);
            changed_this_round |= run_strength_reduction(func);
            changed_this_round |= run_licm(func);
        }
        changed_this_round |= run_dead_code_elimination(func);
        if opt_level >= 1 {
            changed_this_round |= run_adce(func);
        }
        changed_this_round |= run_simplify_cfg(func);
        if opt_level >= 1 {
            changed_this_round |= run_tail_call_optimization(func);
            changed_this_round |= run_peephole(func);
            changed_this_round |= run_register_coalescing(func);
        }

        iterations_performed += 1;

        if !changed_this_round {
            // Fixpoint reached.
            break;
        }

        any_changed = true;
    }

    (any_changed, iterations_performed)
}

// ===========================================================================
// Module-level optimization
// ===========================================================================

/// Runs the optimization pass sequence on every function definition in the
/// module at `-O0` level (basic cleanup only).
///
/// Functions are optimized independently — no inter-procedural optimization
/// is performed at this level. Function declarations (without bodies) are
/// skipped since there is nothing to optimize.
///
/// This is the primary entry point called by the compilation driver
/// (Phase 8 invocation) when no statistics tracking is needed.
///
/// ## Return Value
///
/// Returns `true` if any function in the module was modified by the
/// optimization passes, `false` if all functions were already at a fixpoint.
///
/// # Parameters
///
/// - `module`: Mutable reference to the IR module containing function
///   definitions to optimize.
pub fn optimize_module(module: &mut IrModule) -> bool {
    optimize_module_at_level(module, 0)
}

/// Runs the optimization pass sequence on every function definition in the
/// module at the specified optimization level.
///
/// At `-O0` (level 0), only basic cleanup passes run (constant folding,
/// DCE, CFG simplification). At `-O1` and above (level ≥ 1), all 13
/// passes run in the full pipeline order.
pub fn optimize_module_at_level(module: &mut IrModule, opt_level: u8) -> bool {
    let mut any_changed = false;

    for func in module.functions.iter_mut() {
        // Skip declarations — only optimize function definitions that have bodies.
        if !func.is_definition {
            continue;
        }

        any_changed |= optimize_function_at_level(func, opt_level);
    }

    any_changed
}

/// Runs the optimization pass sequence on every function definition in the
/// module and collects statistics about the run.
///
/// Identical to [`optimize_module`] in behavior, but additionally returns
/// an [`OptimizationStats`] struct with convergence and coverage information.
///
/// This variant is useful for compiler development and debugging — for
/// example, to verify that no function requires an excessive number of
/// fixpoint iterations (which would indicate a pass bug).
///
/// # Parameters
///
/// - `module`: Mutable reference to the IR module to optimize.
///
/// # Returns
///
/// [`OptimizationStats`] containing iteration counts and function coverage.
#[allow(dead_code)]
fn optimize_module_with_stats(module: &mut IrModule, opt_level: u8) -> OptimizationStats {
    let mut stats = OptimizationStats::default();

    for func in module.functions.iter_mut() {
        // Skip declarations — only optimize function definitions that have bodies.
        if !func.is_definition {
            continue;
        }

        stats.total_functions += 1;

        let (changed, iterations) = optimize_function_counted(func, opt_level);

        if changed {
            stats.functions_optimized += 1;
        }

        // Track the maximum iteration count across all functions.
        if iterations > stats.iterations {
            stats.iterations = iterations;
        }
    }

    stats
}

// ===========================================================================
// Optimization level gating
// ===========================================================================

/// Runs the Phase 8 optimization pipeline on the module according to the
/// specified optimization level.
///
/// ## Optimization Level Behavior
///
/// | Level | Pipeline Behavior |
/// |-------|-------------------|
/// | `-O0` (0) | Basic pipeline runs (constant folding, DCE, CFG simplification) |
/// | `-O1` (1) | Same as -O0 (only basic passes exist) |
/// | `-O2` (2) | Same as -O0 (only basic passes exist) |
/// | `-O3` (3) | Same as -O0 (only basic passes exist) |
///
/// At **all** optimization levels, including `-O0`, the basic pass pipeline
/// runs. This is by design: the lowering (Phase 6) and mem2reg (Phase 7)
/// phases produce IR artifacts that should be cleaned up even at `-O0`:
///
/// - **Empty blocks** from structured control flow lowering
/// - **Trivial phi nodes** with all-identical incoming values
/// - **Constant-foldable operations** from expression lowering
/// - **Dead instructions** from alloca promotion
///
/// Cleaning these up improves code generation quality without affecting
/// debuggability — DWARF source mapping uses [`Span`] source locations,
/// not IR instruction structure.
///
/// Future optimization levels could enable additional passes (inlining,
/// loop optimizations, etc.) by gating on the `opt_level` parameter, but
/// these are currently out of scope per the project requirements.
///
/// ## Return Value
///
/// Returns `true` if any function was modified, `false` otherwise.
///
/// # Parameters
///
/// - `module`: Mutable reference to the IR module to optimize.
/// - `opt_level`: Optimization level (0–3), corresponding to `-O0` through `-O3`.
///   Currently all levels run the same basic pipeline.
pub fn run_optimization_pipeline(module: &mut IrModule, opt_level: u8) -> bool {
    // At -O0: only basic cleanup passes (constant folding, DCE, CFG simplification).
    // At -O1+: full 13-pass pipeline including SCCP, GVN, LICM, ADCE, etc.
    optimize_module_at_level(module, opt_level)
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
    use crate::ir::instructions::Instruction;
    use crate::ir::module::IrModule;
    use crate::ir::types::IrType;

    /// Helper: creates a minimal function definition with a single entry
    /// block containing only a return-void terminator.
    fn make_trivial_function(name: &str) -> IrFunction {
        let mut func = IrFunction::new(name.to_string(), vec![], IrType::Void);
        // The constructor already creates an entry block; add a return terminator.
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        func
    }

    /// Helper: creates a function with two blocks — an entry block with an
    /// unconditional branch to an exit block that returns void. This gives
    /// the CFG simplification pass a simple merge opportunity.
    fn make_two_block_function(name: &str) -> IrFunction {
        let mut func = IrFunction::new(name.to_string(), vec![], IrType::Void);
        let exit_idx = func.add_block(BasicBlock::new(1));

        // Entry block: unconditional branch to exit.
        func.blocks[0].instructions.push(Instruction::Branch {
            target: crate::ir::instructions::BlockId(exit_idx as u32),
            span: Span::dummy(),
        });

        // Exit block: return void.
        func.blocks[exit_idx]
            .instructions
            .push(Instruction::Return {
                value: None,
                span: Span::dummy(),
            });

        func
    }

    /// Helper: creates a fresh module with the given list of functions.
    fn make_module(name: &str, functions: Vec<IrFunction>) -> IrModule {
        let mut module = IrModule::new(name.to_string());
        for f in functions {
            module.functions.push(f);
        }
        module
    }

    // -----------------------------------------------------------------------
    // optimize_function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_function_trivial_terminates() {
        let mut func = make_trivial_function("trivial");
        // The key assertion is that the function terminates — the passes
        // should reach a fixpoint quickly on a trivial function.
        let changed = optimize_function(&mut func);
        let _ = changed; // Accept either true or false; termination is the test.
    }

    #[test]
    fn test_optimize_function_returns_bool() {
        let mut func = make_trivial_function("test");
        let result: bool = optimize_function(&mut func);
        // Type check — the function must return a bool.
        let _ = result;
    }

    #[test]
    fn test_optimize_function_two_blocks() {
        let mut func = make_two_block_function("two_block");
        // The CFG simplification pass should have an opportunity to merge
        // or simplify the two-block structure. The exact result depends on
        // the pass implementations, but it should terminate normally.
        let _changed = optimize_function(&mut func);
        // After optimization, the function should still be well-formed.
        assert!(
            !func.blocks.is_empty(),
            "Function must retain at least one block"
        );
    }

    // -----------------------------------------------------------------------
    // optimize_module tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_module_empty_module() {
        let mut module = IrModule::new("empty.c".to_string());
        let changed = optimize_module(&mut module);
        assert!(!changed, "Empty module should report no changes");
    }

    #[test]
    fn test_optimize_module_skips_declarations() {
        let mut module = IrModule::new("test.c".to_string());

        // Add a declaration-only function (no body).
        let mut decl_func = IrFunction::new("extern_fn".to_string(), vec![], IrType::Void);
        decl_func.is_definition = false;
        decl_func.blocks.clear(); // No blocks — it's a declaration.
        module.functions.push(decl_func);

        // Add a trivial definition.
        let def_func = make_trivial_function("defined_fn");
        module.functions.push(def_func);

        // Should not panic on the declaration-only function.
        let _changed = optimize_module(&mut module);
    }

    #[test]
    fn test_optimize_module_processes_multiple_functions() {
        let funcs = vec![
            make_trivial_function("fn1"),
            make_trivial_function("fn2"),
            make_two_block_function("fn3"),
        ];
        let mut module = make_module("multi.c", funcs);

        // All three functions should be processed without panic.
        let _changed = optimize_module(&mut module);

        // All functions should still exist after optimization.
        assert_eq!(module.functions.len(), 3);
    }

    // -----------------------------------------------------------------------
    // run_optimization_pipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_optimization_pipeline_all_levels() {
        // All optimization levels should work without panic.
        for level in 0..=3u8 {
            let mut module = make_module("test.c", vec![make_trivial_function("main")]);
            let _result = run_optimization_pipeline(&mut module, level);
        }
    }

    #[test]
    fn test_run_optimization_pipeline_returns_bool() {
        let mut module = make_module("test.c", vec![make_trivial_function("main")]);
        let result: bool = run_optimization_pipeline(&mut module, 0);
        let _ = result;
    }

    #[test]
    fn test_run_optimization_pipeline_empty_module() {
        let mut module = IrModule::new("empty.c".to_string());
        let changed = run_optimization_pipeline(&mut module, 2);
        assert!(
            !changed,
            "Empty module at any opt level should report no changes"
        );
    }

    // -----------------------------------------------------------------------
    // OptimizationStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimization_stats_default() {
        let stats = OptimizationStats::default();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.functions_optimized, 0);
        assert_eq!(stats.total_functions, 0);
    }

    #[test]
    fn test_optimization_stats_display() {
        let stats = OptimizationStats {
            iterations: 3,
            functions_optimized: 2,
            total_functions: 5,
        };
        let display = format!("{}", stats);
        assert!(display.contains("iterations: 3"));
        assert!(display.contains("functions_optimized: 2/5"));
    }

    #[test]
    fn test_optimization_stats_fields_accessible() {
        let mut stats = OptimizationStats::default();
        stats.iterations = 10;
        stats.functions_optimized = 3;
        stats.total_functions = 7;
        assert_eq!(stats.iterations, 10);
        assert_eq!(stats.functions_optimized, 3);
        assert_eq!(stats.total_functions, 7);
    }

    // -----------------------------------------------------------------------
    // optimize_module_with_stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_module_with_stats_empty() {
        let mut module = IrModule::new("empty.c".to_string());
        let stats = optimize_module_with_stats(&mut module, 0);
        assert_eq!(stats.total_functions, 0);
        assert_eq!(stats.functions_optimized, 0);
        assert_eq!(stats.iterations, 0);
    }

    #[test]
    fn test_optimize_module_with_stats_counts_definitions() {
        let mut module = IrModule::new("test.c".to_string());

        // Two definitions, one declaration.
        module.functions.push(make_trivial_function("fn1"));
        module.functions.push(make_trivial_function("fn2"));

        let mut decl = IrFunction::new("ext".to_string(), vec![], IrType::Void);
        decl.is_definition = false;
        decl.blocks.clear();
        module.functions.push(decl);

        let stats = optimize_module_with_stats(&mut module, 0);
        assert_eq!(stats.total_functions, 2, "Should count only definitions");
    }

    #[test]
    fn test_optimize_module_with_stats_iterations_nonzero_for_work() {
        let mut module = make_module("test.c", vec![make_two_block_function("main")]);
        let stats = optimize_module_with_stats(&mut module, 0);
        assert_eq!(stats.total_functions, 1);
        // At least 1 iteration should always occur for a function with blocks.
        assert!(stats.iterations >= 1, "Should perform at least 1 iteration");
    }

    // -----------------------------------------------------------------------
    // Configuration constant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_iterations_safety_net() {
        // Verify that MAX_ITERATIONS is a reasonable value.
        assert!(
            MAX_ITERATIONS >= 10,
            "MAX_ITERATIONS should be at least 10 for convergence"
        );
        assert!(
            MAX_ITERATIONS <= 1000,
            "MAX_ITERATIONS should not be excessively large"
        );
    }

    #[test]
    fn test_max_iterations_is_100() {
        assert_eq!(
            MAX_ITERATIONS, 100,
            "MAX_ITERATIONS should be 100 per specification"
        );
    }
}
