//! Optimization passes for the BCC compiler — Phase 8 of the compilation pipeline.
//!
//! This module provides the optimization passes that run between SSA construction
//! (Phase 7 / mem2reg) and phi elimination (Phase 9). The fixed-order pipeline is:
//!
//! 1. **Constant folding** ([`constant_folding`]) — evaluate compile-time-constant
//!    operations, fold conditional branches with known conditions, propagate
//!    constants through use-def chains.
//! 2. **Dead code elimination** ([`dead_code_elimination`]) — remove instructions
//!    whose results are unused and have no side effects, remove unreachable blocks.
//! 3. **CFG simplification** ([`simplify_cfg`]) — merge blocks, eliminate empty
//!    blocks, simplify branch chains, thread branches.
//!
//! All passes preserve SSA form: phi nodes are correctly maintained when blocks
//! are merged, edges are redirected, or instructions are removed.
//!
//! The [`pass_manager`] module orchestrates pass execution with fixpoint iteration —
//! passes run in the fixed order above, repeating until no pass reports any change.

// Submodule declarations — each pass is a separate module.
// The pass manager orchestrates pass execution.

pub mod constant_folding;
pub mod dead_code_elimination;
pub mod pass_manager;
pub mod simplify_cfg;

// Convenience re-exports of the primary public API types.
pub use pass_manager::{PassManager, PassStats};
