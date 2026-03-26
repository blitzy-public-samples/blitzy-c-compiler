//! # BCC Optimization Passes (Phase 8)
//!
//! This module implements the Phase 8 optimization pipeline for BCC's intermediate
//! representation. The pipeline consists of **fifteen** optimization passes orchestrated
//! by a pass manager that iterates until fixpoint.
//!
//! ## Pass Pipeline
//!
//! ### Core Passes (run at all optimization levels):
//!
//! 1. [`copy_propagation`] — Replace uses of copy-like instructions (trivial phis,
//!    same-type bitcasts) with the original source value
//! 2. [`constant_folding`] — Evaluate compile-time constant operations, fold conditional
//!    branches with known conditions, propagate constants through use-def chains
//! 3. [`sccp`] — Sparse conditional constant propagation — lattice-based analysis
//!    that finds more constants than simple forward analysis
//! 4. [`instruction_combining`] — Algebraic simplifications: identity operations,
//!    cast chain elimination, boolean simplifications
//! 5. [`strength_reduction`] — Replace expensive operations with cheaper equivalents
//!    (multiply → shift, divide → shift, modulo → AND)
//! 6. [`gvn`] — Global value numbering — eliminates redundant computations by
//!    assigning value numbers, subsumes local CSE
//! 7. [`peephole`] — Pattern-matching optimizations on instruction sequences:
//!    store-load forwarding, dead store elimination, branch simplification
//! 8. [`licm`] — Loop-invariant code motion — hoists computations out of loops
//!    when their operands do not change within the loop body
//! 9. [`dead_code_elimination`] — Remove instructions with unused results and no side
//!    effects, remove unreachable basic blocks
//! 10. [`adce`] — Aggressive dead code elimination — reverse reachability from
//!     essential instructions removes entire dead computation chains
//! 11. [`simplify_cfg`] — Merge single-predecessor/successor block pairs, eliminate
//!     empty blocks, simplify branch chains, remove trivial phi nodes
//! 12. [`register_coalescing`] — Merge values connected through copy-like operations
//!     when their live ranges do not interfere
//! 13. [`tail_call`] — Identify function calls in tail position for jump optimization
//!
//! ## Entry Point
//!
//! The main entry point is [`pass_manager::optimize_module`] (or
//! [`pass_manager::run_optimization_pipeline`] with optimization level), which runs
//! the full pipeline on every function in an IR module.
//!
//! ## Dependencies
//!
//! - Depends on `crate::ir` for IR instruction, basic block, and function types
//! - Depends on `crate::common` for FxHashMap/FxHashSet and diagnostics
//! - Does NOT depend on `crate::frontend` or `crate::backend`
//!
//! ## Architectural Notes
//!
//! - All passes operate on SSA-form IR (produced by mem2reg, Phase 7)
//! - All passes must preserve SSA invariants
//! - The pipeline iterates until fixpoint to maximize optimization at minimal complexity
//! - Each pass is individually toggleable and has its own unit tests

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Pass scheduling and execution framework — orchestrates the multi-pass
/// optimization pipeline with fixpoint iteration.
pub mod pass_manager;

/// Copy propagation — replaces uses of copy-like instructions (trivial phis,
/// same-type bitcasts, identity casts) with original source values.
pub mod copy_propagation;

/// Constant folding and propagation pass — evaluates compile-time constant
/// operations, folds conditional branches with known conditions, and
/// propagates constants through use-def chains.
pub mod constant_folding;

/// Sparse conditional constant propagation (SCCP) — lattice-based analysis
/// that simultaneously propagates constants and eliminates unreachable code.
pub mod sccp;

/// Instruction combining — algebraic simplifications including identity
/// elimination, cast chain cancellation, and boolean simplification.
pub mod instruction_combining;

/// Strength reduction — replaces expensive arithmetic with cheaper equivalents
/// (multiply-by-power-of-2 → shift, unsigned-divide-by-power-of-2 → shift, etc.).
pub mod strength_reduction;

/// Global value numbering (GVN) — assigns value numbers to expressions and
/// eliminates redundant computations, subsuming local CSE.
pub mod gvn;

/// Peephole optimizer — pattern-matching optimizations on instruction sequences
/// including store-load forwarding, dead store elimination, and branch simplification.
pub mod peephole;

/// Loop-invariant code motion (LICM) — hoists computations out of loops when
/// their operands do not change within the loop body.
pub mod licm;

/// Dead code elimination pass — removes instructions with unused results
/// and no side effects, and removes unreachable basic blocks.
pub mod dead_code_elimination;

/// Aggressive dead code elimination (ADCE) — removes entire dead computation
/// chains using reverse reachability from essential instructions.
pub mod adce;

/// Control-flow graph simplification pass — merges single-predecessor/successor
/// block pairs, eliminates empty blocks, simplifies branch chains, and removes
/// trivial phi nodes.
pub mod simplify_cfg;

/// Register coalescing — merges values connected through copy-like operations
/// when their live ranges do not interfere, reducing register pressure.
pub mod register_coalescing;

/// Tail call optimization — identifies function calls in tail position and
/// marks them for the code generator to emit jumps instead of calls.
pub mod tail_call;

// ---------------------------------------------------------------------------
// Re-exports — main entry points for the optimization pipeline
// ---------------------------------------------------------------------------

/// Re-export of [`pass_manager::optimize_module`] for convenient access.
///
/// Runs the multi-pass optimization sequence on every function definition
/// in the module at `-O0`. Returns `true` if any function was modified.
pub use pass_manager::optimize_module;

/// Re-export of [`pass_manager::optimize_module_at_level`] for level-gated access.
///
/// Runs the multi-pass optimization sequence on every function definition
/// in the module at the specified optimization level (0–3).
pub use pass_manager::optimize_module_at_level;

/// Re-export of [`pass_manager::optimize_function`] for convenient access.
///
/// Runs the multi-pass optimization sequence on a single function until
/// fixpoint at `-O0`. Returns `true` if any optimization was applied.
pub use pass_manager::optimize_function;

/// Re-export of [`pass_manager::optimize_function_at_level`] for level-gated access.
///
/// Runs the multi-pass optimization sequence on a single function at the
/// specified optimization level (0–3).
pub use pass_manager::optimize_function_at_level;

/// Re-export of [`pass_manager::run_optimization_pipeline`] for convenient access.
///
/// Runs the Phase 8 optimization pipeline on the module according to the
/// specified optimization level (0–3). Returns `true` if any function was modified.
pub use pass_manager::run_optimization_pipeline;
