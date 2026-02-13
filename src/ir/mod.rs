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

/// Basic block representation — [`BasicBlock`] is the fundamental unit of
/// control flow in the IR. Each block contains an ordered instruction list,
/// CFG predecessor/successor edges, and lazily-populated dominator tree fields
/// used during SSA construction (Phase 7, mem2reg).
pub mod basic_block;

/// IR function representation — [`IrFunction`] is the primary container for
/// function-level IR in the BCC pipeline. Holds basic blocks (CFG), SSA value
/// registry, calling convention, linkage, and GCC function attributes. The
/// entry block is the alloca insertion point for the alloca-then-promote
/// architecture (Phase 6 lowering → Phase 7 mem2reg).
pub mod function;

/// IR module representation — [`IrModule`] is the top-level container for
/// compilation-unit-level entities: global variables, function definitions,
/// external declarations, string literal pool, and inline assembly blocks.
/// The `IrModule` is the data structure passed from Phase 6 (IR lowering)
/// through optimisation passes to Phase 10 (code generation backend).
pub mod module;

/// IR builder API — [`IrBuilder`] provides typed instruction creation
/// methods and insertion point tracking for constructing the intermediate
/// representation.  Used by the AST-to-IR lowering phase to build IR
/// functions one instruction at a time with automatic SSA numbering.
pub mod builder;

/// Memory-to-register promotion (mem2reg) — SSA construction (Phase 7) and
/// phi-node elimination (Phase 9).  Contains the Lengauer-Tarjan dominator
/// tree, dominance frontier computation, SSA renaming, and phi elimination
/// sub-modules.
pub mod mem2reg;

// ── Convenience re-exports ──────────────────────────────────────────────────
// Re-export the most commonly used types so that other modules can write
// `use crate::ir::IrType` rather than `use crate::ir::types::IrType`.

pub use types::IrType;

// Re-export instruction-layer types for convenient access.
pub use instructions::{BasicBlockId, BinOp, FCmpPredicate, ICmpPredicate, Instruction, ValueId};

// Re-export basic block types for convenient access.
pub use basic_block::BasicBlock;

// Re-export function-layer types for convenient access.
pub use function::{
    CallingConvention, FunctionAttributes, IrFunction, Linkage, Parameter, ValueInfo, Visibility,
};

// Re-export module-layer types for convenient access.
pub use module::{Constant, FunctionDecl, GlobalVariable, InlineAsmBlock, IrModule, StringLiteral};

// Re-export builder types for convenient access.
pub use builder::{InsertPosition, IrBuilder};
