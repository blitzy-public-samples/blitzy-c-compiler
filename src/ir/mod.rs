//! # BCC Intermediate Representation
//!
//! This module defines BCC's intermediate representation (IR) and the transformations
//! that operate on it:
//!
//! ## Core IR Definitions
//! - [`types`] — IR type system (Void, I1–I128, F32/F64/F80, Ptr, Array, Struct, Function)
//! - [`instructions`] — IR instruction definitions (Alloca, Load, Store, BinOp, ICmp, FCmp,
//!   Branch, CondBranch, Switch, Call, Return, Phi, GetElementPtr, casts, InlineAsm)
//! - [`basic_block`] — Basic block representation with CFG edges and dominance info
//! - [`function`] — IR function representation with ordered block list
//! - [`module`] — IR module (translation unit) with globals, functions, string pool
//! - [`builder`] — IR builder API for instruction creation with automatic SSA numbering
//!
//! ## Transformations
//! - [`lowering`] — Phase 6: AST-to-IR lowering with "alloca-first" pattern
//! - [`mem2reg`] — Phase 7: SSA construction via alloca promotion (dominance frontiers)
//!   and Phase 9: Phi-node elimination
//!
//! ## Architecture
//!
//! The IR follows the "alloca-then-promote" SSA construction strategy:
//! 1. Phase 6 (lowering): All local variables are initially placed as `alloca` instructions
//!    in the function's entry block
//! 2. Phase 7 (mem2reg): Eligible allocas (scalar, non-address-taken) are promoted to
//!    SSA virtual registers using dominance frontier computation
//! 3. Phase 8 (optimization): Passes operate on SSA-form IR
//! 4. Phase 9 (phi elimination): Phi nodes are eliminated, converting back to copy operations
//!    for register allocation
//!
//! ## Dependencies
//! - Depends on `crate::common` (FxHash, types, diagnostics, source_map)
//! - Depends on `crate::frontend` (AST types consumed during lowering)
//! - Does NOT depend on `crate::backend` or `crate::passes`

// ============================================================================
// Core IR definition submodules
// ============================================================================

/// IR type system — [`IrType`] enum (Void, I1–I128, F32/F64/F80, Ptr, Array,
/// Struct, Function) and [`StructType`] for aggregate field layout.
pub mod types;

/// IR instruction definitions — [`Instruction`] enum covering memory operations
/// (Alloca, Load, Store), arithmetic ([`BinOp`]), comparisons ([`ICmpOp`],
/// [`FCmpOp`]), control flow (Branch, CondBranch, Switch, Return), SSA (Phi),
/// pointer operations (GetElementPtr), casts, and inline assembly.
pub mod instructions;

/// Basic block representation — [`BasicBlock`] struct with ordered instruction
/// list, CFG predecessor/successor edges, and dominator tree fields for SSA
/// construction.
pub mod basic_block;

/// IR function representation — [`IrFunction`] struct with parameter list,
/// ordered basic block sequence, return type, calling convention, linkage,
/// visibility, and attribute-derived flags.
pub mod function;

/// IR module (translation unit) — [`IrModule`] struct containing global
/// variables ([`GlobalVariable`]), function definitions, function declarations,
/// string literal pool, and type metadata.  Also defines [`Constant`],
/// [`Linkage`], and [`Visibility`] for module-level symbol attributes.
pub mod module;

/// IR builder API — [`IrBuilder`] struct providing typed instruction creation
/// methods with automatic SSA value numbering and basic block ID allocation.
/// Used by AST-to-IR lowering (Phase 6).
pub mod builder;

// ============================================================================
// IR transformation submodules
// ============================================================================

/// Phase 6: AST-to-IR lowering — orchestrates the "alloca-first" pattern,
/// converting the semantically-validated AST into IR with all local variables
/// as alloca instructions.  Contains expression, statement, declaration, and
/// inline assembly lowering submodules.
pub mod lowering;

/// Phase 7 (SSA construction) and Phase 9 (phi elimination) — promotes
/// eligible allocas to SSA virtual registers via dominance frontier
/// computation, and later eliminates phi nodes for register allocation.
/// Contains dominator tree, dominance frontier, SSA builder, and phi
/// elimination submodules.
pub mod mem2reg;

// ============================================================================
// Re-export core IR types for convenient access
//
// These re-exports allow consumers to write `use crate::ir::IrType` instead
// of the more verbose `use crate::ir::types::IrType`.  Only the most
// commonly used types are re-exported here.
// ============================================================================

pub use types::IrType;
pub use types::StructType;

pub use instructions::{BinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};

pub use basic_block::BasicBlock;

pub use function::IrFunction;

pub use module::{Constant, GlobalVariable, IrModule, Linkage, Visibility};

pub use builder::IrBuilder;

pub use function::{CallingConvention, FunctionParam};
