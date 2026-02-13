//! Memory-to-register promotion (mem2reg) subsystem — Phases 7 & 9 of the BCC pipeline.
//!
//! This module implements the SSA construction and elimination passes that form
//! the core of the mandated **alloca-then-promote** architecture:
//!
//! 1. **Phase 6** (in `crate::ir::lowering`) emits all local variables as `alloca`
//!    instructions in the function's entry block.
//! 2. **Phase 7** (this module) promotes eligible allocas to SSA virtual registers
//!    via dominance-frontier-based phi-node insertion and dominator-tree-ordered
//!    variable renaming.
//! 3. **Phase 9** (`phi_eliminate`) converts SSA phi nodes back to copy operations
//!    at predecessor block terminators for consumption by the register allocator.
//!
//! # Processing Pipeline
//!
//! ```text
//! identify_promotable_allocas  →  DominatorTree::compute  →  DominanceFrontier::compute
//!         →  insert phi nodes (IDF)  →  SSA rename  →  cleanup alloca/load/store
//! ```
//!
//! # Submodules
//!
//! - [`dominator_tree`] — Lengauer-Tarjan dominator tree (O(n·α(n)))
//! - `dominance_frontier` — Dominance frontier and iterated DF computation
//! - `ssa_builder` — SSA variable renaming via reaching-definition stacks
//! - `phi_eliminate` — Phase 9 phi-node elimination to copies

// ── Submodule declarations ──────────────────────────────────────────────────
// Only modules with source files on disk are declared.  Additional sub-modules
// (dominance_frontier, ssa_builder, phi_eliminate) will be declared as their
// source files are created by the generation pipeline.

/// Dominator tree computation using the Lengauer-Tarjan algorithm.
/// Provides [`DominatorTree`] which is the foundational data structure for
/// both dominance frontier computation and SSA renaming traversal order.
pub mod dominator_tree;

/// Phase 9 phi-node elimination — converts SSA phi nodes back to copy
/// operations at predecessor block terminators for consumption by the
/// register allocator and backend code generator.
pub mod phi_eliminate;

// ── Convenience re-exports ──────────────────────────────────────────────────

pub use dominator_tree::DominatorTree;
pub use phi_eliminate::{eliminate_phis, verify_no_phis};
