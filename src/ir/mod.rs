//! Intermediate Representation (IR) — Middle-end of the BCC compilation pipeline.
//!
//! This module implements the core IR data structures and transformations
//! spanning Phases 6, 7, and 9 of the BCC compilation pipeline:
//!
//! - **Phase 6 — AST-to-IR lowering** (`lowering/`): Converts the
//!   semantically validated AST into a flat IR with alloca instructions for
//!   every local variable (the "alloca" phase of alloca-then-promote).
//! - **Phase 7 — SSA construction** (`mem2reg/`): Promotes eligible allocas
//!   to SSA virtual registers using dominance frontier computation, inserting
//!   phi nodes at join points.
//! - **Phase 9 — Phi elimination** (`mem2reg/phi_eliminate.rs`): Converts
//!   phi nodes back to copy operations at predecessor block terminators,
//!   producing a form suitable for register allocation.
//!
//! # Module Dependencies
//!
//! Per the import architecture rules (Section 0.3.2):
//! - IR modules depend on `crate::common` (types, target, diagnostics)
//! - IR modules depend on `crate::frontend` (for AST types during lowering)
//! - IR is consumed by `crate::passes` (optimization)
//! - IR is consumed by `crate::backend` (code generation)
//!
//! # Submodules
//!
//! - [`types`]: IR type system bridging C types to machine register classes
//! - `instructions`: IR instruction definitions (alloca, load, store, etc.)
//! - `basic_block`: Basic block representation for control flow graphs
//! - `function`: Function-level IR container with SSA value registry
//! - `module`: Top-level IR container for compilation units
//! - `builder`: IR construction API with insertion point tracking
//! - `lowering`: AST-to-IR lowering (Phase 6)
//! - `mem2reg`: SSA construction and phi elimination (Phases 7 & 9)

// ── Submodule declarations ──────────────────────────────────────────────────
// Only modules with corresponding source files on disk are declared here.
// Other submodules (instructions, basic_block, function, module, builder,
// lowering, mem2reg) will be added as their source files are created.

/// IR type system — defines [`IrType`] for machine-level type representation
/// used throughout the intermediate representation.  Bridges C language types
/// from the frontend to machine register classes in the backend.
pub mod types;

/// IR instruction definitions — the central [`Instruction`] enum representing
/// all intermediate representation operations. Also defines [`ValueId`] and
/// [`BasicBlockId`] handle types, along with [`BinOp`], [`ICmpPredicate`],
/// and [`FCmpPredicate`] supporting enums.
pub mod instructions;

// ── Convenience re-exports ──────────────────────────────────────────────────
// Re-export the most commonly used types so that other modules can write
// `use crate::ir::IrType` rather than `use crate::ir::types::IrType`.

pub use types::IrType;

// Re-export instruction-layer types for convenient access.
pub use instructions::{
    BasicBlockId, BinOp, FCmpPredicate, ICmpPredicate, Instruction, ValueId,
};
