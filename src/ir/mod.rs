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

/// AST-to-IR lowering (Phase 6).  Converts the semantically validated and
/// type-annotated AST into the initial IR form where every local variable
/// lives in an `alloca` instruction ("alloca-then-promote" architecture).
/// Contains expression, statement, declaration, and inline-assembly
/// lowering sub-modules coordinated by the `LoweringContext` driver.
pub mod lowering;

// ── Convenience re-exports ──────────────────────────────────────────────────
// Re-export the most commonly used types so that downstream consumers can
// write `use crate::ir::IrType` rather than `use crate::ir::types::IrType`.
// The grouping follows the logical ownership of each type:
//
//   types        → IrType
//   instructions → Instruction, BinOp, ICmpPredicate, FCmpPredicate
//   basic_block  → BasicBlock, BasicBlockId
//   function     → IrFunction, ValueId, Parameter, FunctionAttributes,
//                   Visibility, Linkage, ValueInfo
//   module       → IrModule, GlobalVariable, FunctionDecl, Constant,
//                   StringLiteral, CallingConvention, InlineAsmBlock
//   builder      → IrBuilder, InsertPosition

/// IR type system — the most widely referenced IR type across all consumers.
pub use types::IrType;

/// Core instruction vocabulary — Instruction enum and supporting operation /
/// predicate enums used by every phase after AST lowering.
pub use instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction};

/// Basic block types — the fundamental control-flow-graph unit and its
/// lightweight identifier handle.  `BasicBlockId` is *defined* in
/// `instructions` but canonically re-exported via `basic_block` for
/// consumers working with the CFG.
pub use basic_block::{BasicBlock, BasicBlockId};

/// Function-level IR types — the central function container, SSA value
/// handle, parameter descriptor, function attributes, ELF visibility and
/// linkage enums, and per-value metadata.  `ValueId` is *defined* in
/// `instructions` but canonically re-exported via `function` for consumers
/// working with function IR.
pub use function::{
    FunctionAttributes, IrFunction, Linkage, Parameter, ValueId, ValueInfo, Visibility,
};

/// Module-level IR types — top-level container for a compilation unit,
/// global variables, external declarations, compile-time constants, string
/// literal pool, calling convention enum, and module-level inline assembly.
/// `CallingConvention` is *defined* in `function` but canonically
/// re-exported via `module` for consumers working with module-level
/// constructs.
pub use module::{
    CallingConvention, Constant, FunctionDecl, GlobalVariable, InlineAsmBlock, IrModule,
    StringLiteral,
};

/// IR builder API — the primary construction interface used by AST-to-IR
/// lowering (Phase 6), and the insertion position enum controlling where
/// new instructions are placed within a basic block.
pub use builder::{InsertPosition, IrBuilder};
