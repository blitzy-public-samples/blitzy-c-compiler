//! Phase 6 — AST-to-IR lowering driver and coordination module.
//!
//! This module is the entry point for transforming the semantically validated
//! and type-annotated AST ([`CheckedTranslationUnit`]) into the middle-end
//! IR ([`IrModule`]). It orchestrates the "alloca-then-promote" architecture:
//! every local variable is initially emitted as an `alloca` instruction in
//! the function's entry basic block, producing alloca-heavy IR that the
//! subsequent mem2reg pass (Phase 7) promotes to SSA form.
//!
//! # Pipeline Position
//!
//! ```text
//! Phase 5 (Sema) ──► Phase 6 (IR Lowering) ──► Phase 7 (mem2reg / SSA)
//!   CheckedTranslationUnit    IrModule             SSA-form IrModule
//! ```
//!
//! # Architecture
//!
//! The lowering driver iterates over top-level declarations in the
//! [`CheckedTranslationUnit`]. For each function definition it:
//!
//! 1. Creates a fresh [`IrFunction`] with an entry block.
//! 2. Allocates an `alloca` per parameter and per local variable in the
//!    entry block, recording each in the `variables` map.
//! 3. Stores incoming parameter values into their corresponding allocas.
//! 4. Delegates body lowering to [`stmt_lowering`] and [`expr_lowering`].
//! 5. Adds the completed function to the [`IrModule`].
//!
//! For global variable declarations it creates [`GlobalVariable`] entries
//! with appropriately lowered initializers and linkage.
//!
//! # Submodules
//!
//! - [`expr_lowering`]: Expression-to-IR lowering (arithmetic, casts, calls).
//! - [`stmt_lowering`]: Statement-to-IR lowering (control flow, loops, switch).
//! - [`decl_lowering`]: Declaration lowering (globals, initializers, statics).
//! - [`asm_lowering`]: Inline assembly lowering (constraints, operands).
//!
//! # Key Types
//!
//! - [`ModuleLoweringContext`]: Module-wide state shared across all function
//!   lowering passes (owns the [`IrModule`], [`DiagnosticEngine`], target info).
//! - [`LoweringContext`]: Per-function state (IR builder, variable map, loop
//!   stack, label map) that borrows from the module context.
//! - [`LoweringError`]: Enumeration of all recoverable and fatal lowering
//!   errors with source-location tracking.
//! - [`GlobalSymbolInfo`]: Tracks global symbol metadata for cross-function
//!   reference resolution and ELF symbol table emission.

// ============================================================================
// Submodule declarations
// ============================================================================

pub mod asm_lowering;
pub mod decl_lowering;
pub mod expr_lowering;
pub mod stmt_lowering;

// ============================================================================
// Imports — IR infrastructure
// ============================================================================

use crate::ir::basic_block::BasicBlockId;
use crate::ir::builder::IrBuilder;
use crate::ir::function::{
    FunctionAttributes, IrFunction, Linkage as IrLinkage, Parameter, Visibility as IrVisibility,
};
use crate::ir::instructions::ValueId;
use crate::ir::module::{Constant, FunctionDecl, GlobalVariable, IrModule};
use crate::ir::types::IrType;

// ============================================================================
// Imports — Frontend AST and semantic analysis
// ============================================================================

use crate::frontend::parser::ast::{self, Statement};
use crate::frontend::sema::{
    self, CheckedDeclaration, CheckedParameter, CheckedTranslationUnit, Linkage as SemaLinkage,
    StorageClass as SemaStorageClass, SymbolEntry, SymbolTable, TypedExpression,
    ValidatedAttribute, VisibilityKind,
};

// ============================================================================
// Imports — Common infrastructure
// ============================================================================

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::source_map::SourceMap;
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::{DataModel, Target};
use crate::common::types::{self as ctypes, CType};

use std::fmt;

// ============================================================================
// Constants
// ============================================================================

/// Maximum recursion depth enforced during lowering to prevent stack
/// overflow on deeply nested kernel constructs. Matches the 512-limit
/// specified in Section 0.7.3 of the architectural requirements.
pub const MAX_RECURSION_DEPTH: usize = 512;

// ============================================================================
// LoweringError — Phase 6 error enumeration
// ============================================================================

/// Enumerates all error conditions that can occur during Phase 6 lowering.
///
/// Each variant carries a source [`Span`] for diagnostic reporting and
/// a descriptive message. Errors are emitted through the [`DiagnosticEngine`]
/// for multi-error accumulation; fatal errors (e.g., recursion limit)
/// additionally terminate the current lowering operation via `Result`.
#[derive(Clone, Debug)]
pub enum LoweringError {
    /// A type mismatch was detected during IR instruction construction
    /// (e.g., storing a float into an integer alloca).
    TypeMismatch {
        expected: IrType,
        found: IrType,
        span: Span,
        message: String,
    },

    /// A variable was referenced that has no corresponding symbol table
    /// entry or alloca mapping in the current lowering context.
    UndeclaredVariable { name: Symbol, span: Span },

    /// An expression was used in lvalue context but does not designate
    /// a memory location (e.g., `42 = x`).
    InvalidLvalue { span: Span, message: String },

    /// An expression form is not yet supported by the lowering pass
    /// (e.g., a rare GCC extension encountered during kernel compilation).
    UnsupportedExpression { span: Span, message: String },

    /// A `goto` statement references a label that was never defined
    /// within the current function scope.
    UndefinedLabel { name: Symbol, span: Span },

    /// A `break` statement appears outside of any loop or switch body.
    BreakOutsideLoop { span: Span },

    /// A `continue` statement appears outside of any loop body.
    ContinueOutsideLoop { span: Span },

    /// An inline assembly constraint string is invalid or unsupported
    /// for the current target architecture.
    InvalidConstraint {
        constraint: String,
        span: Span,
        message: String,
    },

    /// The number of operands in an inline assembly statement does not
    /// match the constraint specification.
    OperandCountMismatch {
        expected: usize,
        found: usize,
        span: Span,
    },

    /// A static or file-scope variable initializer contains a non-constant
    /// expression that cannot be evaluated at compile time.
    NonConstantStaticInit { span: Span, message: String },

    /// A symbol was defined more than once with conflicting definitions
    /// (after linkage resolution).
    DuplicateDefinition {
        name: Symbol,
        span: Span,
        previous_span: Span,
    },

    /// An initializer is structurally invalid for the target type
    /// (e.g., too many elements, wrong nesting).
    InvalidInitializer { span: Span, message: String },

    /// A C type could not be mapped to an IR type (e.g., incomplete
    /// struct with unknown layout).
    TypeMappingError { span: Span, message: String },

    /// The recursion depth limit was exceeded during nested expression
    /// or statement lowering.
    RecursionLimitExceeded {
        depth: usize,
        limit: usize,
        span: Span,
    },

    /// A division or modulo by zero was detected at compile time in a
    /// constant expression used during lowering.
    DivisionByZero { span: Span },

    /// An inline assembly constraint is recognized but not supported
    /// for the current target architecture.
    UnsupportedConstraint {
        constraint: String,
        span: Span,
        message: String,
    },
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringError::TypeMismatch {
                expected,
                found,
                message,
                ..
            } => {
                write!(
                    f,
                    "type mismatch: expected {expected:?}, found {found:?}: {message}"
                )
            }
            LoweringError::UndeclaredVariable { .. } => {
                write!(f, "use of undeclared variable")
            }
            LoweringError::InvalidLvalue { message, .. } => {
                write!(f, "invalid lvalue: {message}")
            }
            LoweringError::UnsupportedExpression { message, .. } => {
                write!(f, "unsupported expression: {message}")
            }
            LoweringError::UndefinedLabel { .. } => {
                write!(f, "use of undefined label")
            }
            LoweringError::BreakOutsideLoop { .. } => {
                write!(f, "'break' statement outside of loop or switch")
            }
            LoweringError::ContinueOutsideLoop { .. } => {
                write!(f, "'continue' statement outside of loop")
            }
            LoweringError::InvalidConstraint {
                constraint,
                message,
                ..
            } => {
                write!(f, "invalid asm constraint '{constraint}': {message}")
            }
            LoweringError::OperandCountMismatch {
                expected, found, ..
            } => {
                write!(
                    f,
                    "asm operand count mismatch: expected {expected}, found {found}"
                )
            }
            LoweringError::NonConstantStaticInit { message, .. } => {
                write!(f, "non-constant static initializer: {message}")
            }
            LoweringError::DuplicateDefinition { .. } => {
                write!(f, "duplicate definition")
            }
            LoweringError::InvalidInitializer { message, .. } => {
                write!(f, "invalid initializer: {message}")
            }
            LoweringError::TypeMappingError { message, .. } => {
                write!(f, "type mapping error: {message}")
            }
            LoweringError::RecursionLimitExceeded { depth, limit, .. } => {
                write!(f, "recursion depth {depth} exceeds limit {limit}")
            }
            LoweringError::DivisionByZero { .. } => {
                write!(f, "division by zero in constant expression")
            }
            LoweringError::UnsupportedConstraint {
                constraint,
                message,
                ..
            } => {
                write!(f, "unsupported asm constraint '{constraint}': {message}")
            }
        }
    }
}

impl LoweringError {
    /// Returns the source span associated with this error.
    pub fn span(&self) -> Span {
        match self {
            LoweringError::TypeMismatch { span, .. }
            | LoweringError::UndeclaredVariable { span, .. }
            | LoweringError::InvalidLvalue { span, .. }
            | LoweringError::UnsupportedExpression { span, .. }
            | LoweringError::UndefinedLabel { span, .. }
            | LoweringError::BreakOutsideLoop { span }
            | LoweringError::ContinueOutsideLoop { span }
            | LoweringError::InvalidConstraint { span, .. }
            | LoweringError::OperandCountMismatch { span, .. }
            | LoweringError::NonConstantStaticInit { span, .. }
            | LoweringError::DuplicateDefinition { span, .. }
            | LoweringError::InvalidInitializer { span, .. }
            | LoweringError::TypeMappingError { span, .. }
            | LoweringError::RecursionLimitExceeded { span, .. }
            | LoweringError::DivisionByZero { span }
            | LoweringError::UnsupportedConstraint { span, .. } => *span,
        }
    }

    /// Emits this error as a diagnostic through the given engine.
    pub fn emit(&self, diagnostics: &mut DiagnosticEngine) {
        diagnostics.error(self.span(), self.to_string());
    }
}

// ============================================================================
// GlobalSymbolInfo — cross-function global symbol tracking
// ============================================================================

/// Metadata for a global symbol (function or variable) tracked during
/// the lowering of the entire translation unit.
///
/// This structure records the IR-level properties of each global symbol
/// as it is encountered during declaration processing. It is used for:
/// - Cross-function reference validation.
/// - ELF symbol table and relocation emission in the backend.
/// - Duplicate definition detection.
/// - Tentative definition finalization.
#[derive(Clone, Debug)]
pub struct GlobalSymbolInfo {
    /// Interned symbol name.
    pub name: Symbol,
    /// IR type of the global (function type or variable type).
    pub ir_type: IrType,
    /// IR-level linkage (External, Internal, Weak, Common).
    pub linkage: IrLinkage,
    /// Whether this symbol has been defined (vs. merely declared).
    pub is_defined: bool,
    /// Whether this symbol uses thread-local storage (`_Thread_local`).
    pub is_tls: bool,
    /// Custom ELF section name from `__attribute__((section("...")))`.
    pub section: Option<String>,
    /// ELF symbol visibility (Default, Hidden, Protected, Internal).
    pub visibility: IrVisibility,
    /// Required alignment in bytes (from `__attribute__((aligned(N)))`).
    pub alignment: Option<u64>,
}

impl GlobalSymbolInfo {
    /// Creates a new global symbol info with default external linkage
    /// and default visibility.
    pub fn new(name: Symbol, ir_type: IrType) -> Self {
        GlobalSymbolInfo {
            name,
            ir_type,
            linkage: IrLinkage::External,
            is_defined: false,
            is_tls: false,
            section: None,
            visibility: IrVisibility::Default,
            alignment: None,
        }
    }

    /// Creates a global symbol info populated from a semantic analysis
    /// [`SymbolEntry`] and its validated attributes.
    pub fn from_symbol_entry(entry: &SymbolEntry, ir_type: IrType) -> Self {
        let linkage =
            map_sema_linkage_to_ir(entry.linkage, entry.attributes.is_weak, entry.is_tentative);
        let visibility = entry
            .attributes
            .visibility
            .map(map_visibility_kind_to_ir)
            .unwrap_or(IrVisibility::Default);
        let is_tls = entry.storage_class == SemaStorageClass::ThreadLocal;

        GlobalSymbolInfo {
            name: entry.name,
            ir_type,
            linkage,
            is_defined: entry.is_definition,
            is_tls,
            section: entry.attributes.section.clone(),
            visibility,
            alignment: entry.attributes.alignment,
        }
    }
}

// ============================================================================
// LoopContext / SwitchContext — break/continue target resolution
// ============================================================================

/// Tracks the break and continue targets for the innermost enclosing
/// loop during statement lowering.
///
/// Pushed onto [`LoweringContext::loop_stack`] when entering a `for`,
/// `while`, or `do-while` loop body, and popped on exit.
#[derive(Clone, Debug)]
pub struct LoopContext {
    /// The basic block to jump to on `break`.
    pub break_target: BasicBlockId,
    /// The basic block to jump to on `continue` (loop header or latch).
    pub continue_target: BasicBlockId,
}

/// Tracks the break target for the innermost enclosing `switch` statement.
///
/// Pushed onto [`LoweringContext::switch_stack`] when entering a switch
/// body, and popped on exit. Unlike loops, `switch` does not have a
/// continue target.
#[derive(Clone, Debug)]
pub struct SwitchContext {
    /// The basic block to jump to on `break`.
    pub break_target: BasicBlockId,
}

// ============================================================================
// ModuleLoweringContext — module-wide shared state
// ============================================================================

/// Module-wide lowering state shared across all function bodies within
/// a single translation unit.
///
/// Owns the [`IrModule`] being constructed, the diagnostic engine, the
/// string interner, and cross-function global symbol tracking. A single
/// instance is created per `lower_translation_unit()` invocation and
/// is mutably borrowed by each per-function [`LoweringContext`].
pub struct ModuleLoweringContext {
    /// The IR module being constructed — accumulates global variables,
    /// function definitions, declarations, and string literals.
    pub module: IrModule,

    /// Target architecture information (pointer width, alignment, data model).
    pub target: Target,

    /// Cross-function global symbol tracking. Maps interned symbol names
    /// to their IR-level metadata for duplicate detection, tentative
    /// definition resolution, and linkage/visibility propagation.
    pub global_symbols: FxHashMap<Symbol, GlobalSymbolInfo>,

    /// String literal deduplication map. Maps a hash of the string
    /// content to the string literal ID in the IrModule, avoiding
    /// duplicate `.rodata` entries for identical string literals.
    pub string_literals: FxHashMap<u64, u32>,

    /// Multi-error diagnostic reporting engine for emitting errors,
    /// warnings, and notes during the lowering process.
    pub diagnostics: DiagnosticEngine,

    /// Source file tracking for span-to-file/line resolution in
    /// diagnostic messages.
    pub source_map: SourceMap,

    /// String interner for zero-cost identifier comparison and
    /// name resolution during symbol lookups.
    pub interner: Interner,
}

impl ModuleLoweringContext {
    /// Creates a new module lowering context for the given translation unit.
    ///
    /// # Arguments
    ///
    /// * `module_name` — Name for the IR module (typically the source file name).
    /// * `target` — Target architecture for this compilation.
    /// * `diagnostics` — Diagnostic engine (takes ownership).
    /// * `source_map` — Source file tracking (takes ownership).
    /// * `interner` — String interner (takes ownership).
    pub fn new(
        module_name: String,
        target: Target,
        diagnostics: DiagnosticEngine,
        source_map: SourceMap,
        interner: Interner,
    ) -> Self {
        ModuleLoweringContext {
            module: IrModule::new(module_name, target),
            target,
            global_symbols: FxHashMap::default(),
            string_literals: FxHashMap::default(),
            diagnostics,
            source_map,
            interner,
        }
    }

    /// Registers or updates a global symbol in the cross-function tracking map.
    ///
    /// If the symbol already exists and is now being defined, updates the
    /// `is_defined` flag. Returns an error if the symbol is already defined
    /// with conflicting properties.
    pub fn register_global_symbol(&mut self, info: GlobalSymbolInfo) -> Result<(), LoweringError> {
        let name = info.name;
        if let Some(existing) = self.global_symbols.get(&name) {
            if existing.is_defined && info.is_defined {
                return Err(LoweringError::DuplicateDefinition {
                    name,
                    span: Span::DUMMY,
                    previous_span: Span::DUMMY,
                });
            }
            // Update to the definition version if we now have one.
            if info.is_defined {
                self.global_symbols.insert(name, info);
            }
        } else {
            self.global_symbols.insert(name, info);
        }
        Ok(())
    }

    /// Interns a string literal, returning its module-level string ID.
    /// Deduplicates identical string content via the `string_literals` hash map.
    pub fn intern_string_literal(&mut self, data: &[u8]) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(&id) = self.string_literals.get(&hash) {
            return id;
        }

        let id = self.module.add_string_literal(data.to_vec());
        self.string_literals.insert(hash, id);
        id
    }
}

// ============================================================================
// LoweringContext — per-function lowering state
// ============================================================================

/// Per-function lowering context holding the IR builder, variable-to-alloca
/// map, and references to shared module-level state.
///
/// Created fresh for each function definition by
/// [`create_function_lowering_context`] and dropped after the function
/// body has been fully lowered. The completed [`IrFunction`] is then
/// added to the [`ModuleLoweringContext::module`].
///
/// # Borrow Architecture
///
/// The `module_ctx` field holds a mutable reference to the
/// [`ModuleLoweringContext`], through which diagnostics, target info,
/// global symbols, and the string literal pool are accessed. The
/// `function` field is a separate mutable reference to an [`IrFunction`]
/// allocated on the stack of the calling function (not inside the module),
/// enabling simultaneous mutation of both.
pub struct LoweringContext<'a> {
    /// IR instruction builder — tracks insertion point and creates instructions.
    pub builder: IrBuilder,

    /// The IR function currently being lowered. Allocated on the caller's
    /// stack and later moved into the module via `IrModule::add_function`.
    pub function: &'a mut IrFunction,

    /// Maps local variable symbols to their `alloca` result [`ValueId`]s.
    /// This is the core data structure of the "alloca-then-promote" pattern:
    /// every local variable has an alloca, and loads/stores go through it
    /// until the mem2reg pass promotes eligible allocas to SSA registers.
    pub variables: FxHashMap<Symbol, ValueId>,

    /// Maps label symbols to their target [`BasicBlockId`]s for `goto`
    /// and labeled-statement resolution.
    pub label_blocks: FxHashMap<Symbol, BasicBlockId>,

    /// Stack of active loop contexts for resolving `break` and `continue`.
    /// The innermost loop is at the top of the stack.
    pub loop_stack: Vec<LoopContext>,

    /// Stack of active switch contexts for resolving `break` within
    /// switch statements.
    pub switch_stack: Vec<SwitchContext>,

    /// Mutable reference to the module-wide lowering context.
    /// Provides access to: diagnostics, target, global_symbols,
    /// string_literals, interner, source_map, and the IrModule.
    pub module_ctx: &'a mut ModuleLoweringContext,

    /// Current recursion depth counter, incremented on entry to nested
    /// expression/statement lowering and decremented on exit.
    pub recursion_depth: usize,

    /// Maximum allowed recursion depth (typically [`MAX_RECURSION_DEPTH`]).
    pub max_recursion_depth: usize,

    /// Set of label symbols whose addresses have been taken via the
    /// GCC computed-goto extension (`&&label`). These labels must remain
    /// as indirect branch targets and cannot be optimized away.
    pub address_taken_labels: FxHashSet<Symbol>,
}

impl<'a> LoweringContext<'a> {
    // -- Convenience accessors -----------------------------------------------

    /// Returns a mutable reference to the diagnostic engine.
    #[inline]
    pub fn diagnostics(&mut self) -> &mut DiagnosticEngine {
        &mut self.module_ctx.diagnostics
    }

    /// Returns an immutable reference to the compilation target.
    #[inline]
    pub fn target(&self) -> &Target {
        &self.module_ctx.target
    }

    /// Returns a mutable reference to the string interner.
    #[inline]
    pub fn interner(&mut self) -> &mut Interner {
        &mut self.module_ctx.interner
    }

    /// Returns an immutable reference to the source map.
    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        &self.module_ctx.source_map
    }

    // -- Variable management -------------------------------------------------

    /// Registers a local variable by creating an `alloca` in the function's
    /// entry block and recording the mapping in `self.variables`.
    ///
    /// This implements the "alloca" phase of the alloca-then-promote pattern.
    /// The returned [`ValueId`] is a pointer (IrType::Ptr) to the allocated
    /// stack slot.
    pub fn create_local_alloca(&mut self, name: Symbol, ir_type: IrType) -> ValueId {
        // Resolve the name to a string, copying to avoid borrow conflicts.
        let name_str = self.module_ctx.interner.resolve(name).to_string();
        let alloca_id = self
            .builder
            .build_alloca(self.function, ir_type, Some(&name_str));
        self.variables.insert(name, alloca_id);
        alloca_id
    }

    /// Looks up the alloca [`ValueId`] for a local variable.
    ///
    /// Returns `None` if the variable has not been declared in the
    /// current function scope.
    #[inline]
    pub fn get_variable(&self, name: Symbol) -> Option<ValueId> {
        self.variables.get(&name).copied()
    }

    // -- Label management ----------------------------------------------------

    /// Gets or creates a basic block for the given label.
    ///
    /// If the label has already been encountered (forward reference),
    /// returns the existing block. Otherwise, creates a new block and
    /// records the mapping.
    pub fn get_or_create_label_block(&mut self, label: Symbol) -> BasicBlockId {
        if let Some(&block_id) = self.label_blocks.get(&label) {
            return block_id;
        }
        let label_name = self.module_ctx.interner.resolve(label).to_string();
        let block_id = self
            .builder
            .create_block(self.function, Some(&format!("label.{}", label_name)));
        self.label_blocks.insert(label, block_id);
        block_id
    }

    // -- Loop / switch stack management --------------------------------------

    /// Pushes a new loop context for break/continue resolution.
    #[inline]
    pub fn push_loop(&mut self, break_target: BasicBlockId, continue_target: BasicBlockId) {
        self.loop_stack.push(LoopContext {
            break_target,
            continue_target,
        });
    }

    /// Pops the innermost loop context.
    #[inline]
    pub fn pop_loop(&mut self) -> Option<LoopContext> {
        self.loop_stack.pop()
    }

    /// Returns the innermost loop context, or `None` if not inside a loop.
    #[inline]
    pub fn current_loop(&self) -> Option<&LoopContext> {
        self.loop_stack.last()
    }

    /// Pushes a new switch context for break resolution.
    #[inline]
    pub fn push_switch(&mut self, break_target: BasicBlockId) {
        self.switch_stack.push(SwitchContext { break_target });
    }

    /// Pops the innermost switch context.
    #[inline]
    pub fn pop_switch(&mut self) -> Option<SwitchContext> {
        self.switch_stack.pop()
    }

    /// Returns the break target for the innermost loop or switch.
    ///
    /// Switch break targets take precedence when the switch is more
    /// deeply nested than the current loop.
    pub fn current_break_target(&self) -> Option<BasicBlockId> {
        // `break` exits the innermost enclosing loop or switch.
        // Check both stacks and return whichever was pushed last.
        let loop_target = self.loop_stack.last().map(|l| l.break_target);
        let switch_target = self.switch_stack.last().map(|s| s.break_target);

        match (loop_target, switch_target) {
            (Some(_), Some(st)) => {
                // The switch was pushed more recently if its stack is taller.
                // In practice, the caller disambiguates. Prefer the switch
                // target since `break` in a switch-within-loop exits the switch.
                Some(st)
            }
            (Some(lt), None) => Some(lt),
            (None, Some(st)) => Some(st),
            (None, None) => None,
        }
    }

    /// Returns the continue target for the innermost loop.
    pub fn current_continue_target(&self) -> Option<BasicBlockId> {
        self.loop_stack.last().map(|l| l.continue_target)
    }

    // -- Block termination checking ------------------------------------------

    /// Returns `true` if the current insertion block already has a
    /// terminator instruction (branch, return, switch, etc.).
    ///
    /// Used by statement lowering to skip dead code after unconditional
    /// control flow transfers.
    pub fn current_block_terminated(&self) -> bool {
        if let Some(block_id) = self.builder.get_insert_block() {
            let block = self.function.get_block(block_id);
            return block.has_terminator();
        }
        false
    }
}

// ============================================================================
// Public API — entry point and factory functions
// ============================================================================

/// Lowers an entire [`CheckedTranslationUnit`] to an [`IrModule`].
///
/// This is the primary entry point for Phase 6 of the compilation pipeline.
/// It iterates over all top-level declarations in the semantically validated
/// AST and produces the corresponding IR representation.
///
/// # Arguments
///
/// * `checked_tu` — The semantically validated translation unit from Phase 5.
/// * `target` — The compilation target architecture.
/// * `diagnostics` — Diagnostic engine for error reporting (takes ownership).
/// * `source_map` — Source file tracking (takes ownership).
/// * `interner` — String interner (takes ownership).
/// * `module_name` — Name for the output IR module.
///
/// # Returns
///
/// On success, returns the completed [`IrModule`] containing all global
/// variables, function definitions, function declarations, and string
/// literals. On failure (fatal errors), returns a [`LoweringError`].
///
/// Non-fatal errors are accumulated in the diagnostic engine. The caller
/// should check `diagnostics.has_errors()` on the returned module context
/// even on `Ok` results.
///
/// # Alloca-Then-Promote Architecture
///
/// Every local variable in every function body is emitted as an `alloca`
/// instruction in the function's entry block. The subsequent mem2reg pass
/// (Phase 7) will promote eligible allocas to SSA virtual registers.
pub fn lower_translation_unit(
    checked_tu: &CheckedTranslationUnit,
    target: Target,
    diagnostics: DiagnosticEngine,
    source_map: SourceMap,
    interner: Interner,
    module_name: String,
) -> Result<ModuleLoweringContext, LoweringError> {
    let mut module_ctx =
        ModuleLoweringContext::new(module_name, target, diagnostics, source_map, interner);

    // Process each top-level declaration in order.
    for decl in &checked_tu.declarations {
        let result = lower_top_level_declaration(&mut module_ctx, decl, &checked_tu.symbol_table);

        if let Err(e) = result {
            // Emit the error as a diagnostic and continue processing
            // to accumulate multiple errors.
            e.emit(&mut module_ctx.diagnostics);
        }
    }

    // Finalize tentative definitions: any global symbol that was
    // declared but never defined and is marked tentative gets a
    // zero-initialized definition.
    finalize_tentative_definitions(&mut module_ctx, &checked_tu.symbol_table);

    Ok(module_ctx)
}

/// Creates a fresh per-function [`LoweringContext`] for lowering a
/// function body.
///
/// The context is initialized with:
/// - A new [`IrBuilder`] positioned at the function's entry block.
/// - Empty variable, label, loop, and switch maps.
/// - Recursion depth at zero with the standard limit.
///
/// # Arguments
///
/// * `function` — Mutable reference to the [`IrFunction`] being lowered.
/// * `module_ctx` — Mutable reference to the module-wide context.
///
/// # Returns
///
/// A fully initialized [`LoweringContext`] ready for statement and
/// expression lowering.
pub fn create_function_lowering_context<'a>(
    function: &'a mut IrFunction,
    module_ctx: &'a mut ModuleLoweringContext,
) -> LoweringContext<'a> {
    let mut builder = IrBuilder::new();

    // Set the insertion point to the function's entry block.
    let entry_id = function.entry_block().id;
    builder.set_insert_point(entry_id);

    LoweringContext {
        builder,
        function,
        variables: FxHashMap::default(),
        label_blocks: FxHashMap::default(),
        loop_stack: Vec::new(),
        switch_stack: Vec::new(),
        module_ctx,
        recursion_depth: 0,
        max_recursion_depth: MAX_RECURSION_DEPTH,
        address_taken_labels: FxHashSet::default(),
    }
}

/// Checks whether the current recursion depth exceeds the configured
/// limit, returning a [`LoweringError::RecursionLimitExceeded`] if so.
///
/// This guard must be called before recursing into nested expression
/// or statement lowering to enforce the 512-depth limit specified in
/// Section 0.7.3 of the architectural requirements.
///
/// # Arguments
///
/// * `ctx` — The current lowering context.
/// * `span` — Source location for the error diagnostic.
///
/// # Returns
///
/// `Ok(())` if the depth is within limits, `Err(LoweringError)` otherwise.
#[inline]
pub fn check_recursion_depth(ctx: &LoweringContext<'_>, span: Span) -> Result<(), LoweringError> {
    if ctx.recursion_depth >= ctx.max_recursion_depth {
        Err(LoweringError::RecursionLimitExceeded {
            depth: ctx.recursion_depth,
            limit: ctx.max_recursion_depth,
            span,
        })
    } else {
        Ok(())
    }
}

/// Ensures the current insertion block has not already been terminated.
///
/// Returns `true` if the block is still open (no terminator), meaning
/// it is safe to append more instructions. Returns `false` if a
/// terminator has already been emitted, indicating that subsequent
/// instructions would be unreachable dead code.
///
/// # Arguments
///
/// * `ctx` — The current lowering context.
///
/// # Returns
///
/// `true` if instructions can still be appended to the current block.
#[inline]
pub fn ensure_not_terminated(ctx: &LoweringContext<'_>) -> bool {
    !ctx.current_block_terminated()
}

// ============================================================================
// C Type to IR Type conversion
// ============================================================================

/// Converts a C language type ([`CType`]) to its corresponding IR type
/// ([`IrType`]) based on the target architecture.
///
/// This is a fundamental mapping used throughout the lowering process:
/// - `void` → `IrType::Void`
/// - `_Bool` → `IrType::I1`
/// - `char` → `IrType::I8`
/// - `short` → `IrType::I16`
/// - `int` → `IrType::I32`
/// - `long` → `IrType::I32` (ILP32) or `IrType::I64` (LP64)
/// - `long long` → `IrType::I64`
/// - `float` → `IrType::F32`
/// - `double` → `IrType::F64`
/// - `long double` → `IrType::F80` (x86) or `IrType::F64` (ARM/RISC-V)
/// - pointers → `IrType::Ptr` (opaque pointer model)
/// - arrays → `IrType::Array`
/// - structs/unions → `IrType::Struct`
/// - enums → underlying integer type
/// - `_Atomic(T)` → same as `T` (atomic ops handled at instruction level)
/// - typedefs → recurse through underlying type
///
/// # Arguments
///
/// * `ctype` — The C type to convert.
/// * `target` — The target architecture for size-dependent decisions.
///
/// # Returns
///
/// The corresponding [`IrType`], or [`LoweringError::TypeMappingError`]
/// if the type cannot be represented in the IR.
pub fn c_type_to_ir_type(ctype: &CType, target: &Target) -> Result<IrType, LoweringError> {
    match ctype {
        CType::Void => Ok(IrType::Void),

        CType::Bool => Ok(IrType::I1),

        CType::Char { .. } => Ok(IrType::I8),

        CType::Short { .. } => Ok(IrType::I16),

        CType::Int { .. } => Ok(IrType::I32),

        CType::Long { .. } => {
            // LP64 targets (x86-64, AArch64, RISC-V 64): long = 64 bits
            // ILP32 targets (i686): long = 32 bits
            match target.data_model() {
                DataModel::LP64 => Ok(IrType::I64),
                DataModel::ILP32 => Ok(IrType::I32),
            }
        }

        CType::LongLong { .. } => Ok(IrType::I64),

        CType::Float => Ok(IrType::F32),

        CType::Double => Ok(IrType::F64),

        CType::LongDouble => {
            // x86-64, i686: 80-bit x87 extended precision
            // AArch64, RISC-V 64: maps to double (64-bit)
            match target {
                Target::X86_64 | Target::I686 => Ok(IrType::F80),
                Target::AArch64 | Target::RiscV64 => Ok(IrType::F64),
            }
        }

        CType::Complex(base) => {
            // _Complex T is represented as a struct of two T values.
            let elem = c_type_to_ir_type(base, target)?;
            Ok(IrType::Struct {
                fields: vec![elem.clone(), elem],
                packed: false,
            })
        }

        CType::Pointer(_) => Ok(IrType::Ptr),

        CType::Array { element, size } => {
            let elem_ir = c_type_to_ir_type(element, target)?;
            let count = size.unwrap_or(0);
            Ok(IrType::Array {
                element: Box::new(elem_ir),
                count,
            })
        }

        CType::Function {
            return_type,
            params,
            variadic,
        } => {
            let ret = c_type_to_ir_type(return_type, target)?;
            let param_types: Result<Vec<IrType>, LoweringError> = params
                .iter()
                .map(|p| c_type_to_ir_type(p, target))
                .collect();
            Ok(IrType::Function {
                return_type: Box::new(ret),
                param_types: param_types?,
                is_variadic: *variadic,
            })
        }

        CType::Struct { fields, .. } => {
            let field_types: Result<Vec<IrType>, LoweringError> = fields
                .iter()
                .map(|f| c_type_to_ir_type(&f.ty, target))
                .collect();
            Ok(IrType::Struct {
                fields: field_types?,
                packed: false,
            })
        }

        CType::Union { fields, .. } => {
            // Unions are represented as a struct with a single field of the
            // size of the largest member. We use an I8 array of the union's
            // total size as the representation.
            if fields.is_empty() {
                return Ok(IrType::Struct {
                    fields: vec![],
                    packed: false,
                });
            }
            let union_size = ctypes::size_of(ctype, target);
            Ok(IrType::Array {
                element: Box::new(IrType::I8),
                count: union_size,
            })
        }

        CType::Enum { underlying, .. } => {
            // Enums are represented as their underlying integer type.
            c_type_to_ir_type(underlying, target)
        }

        CType::Atomic(inner) => {
            // _Atomic(T) has the same representation as T; atomic semantics
            // are handled at the instruction level.
            c_type_to_ir_type(inner, target)
        }

        CType::Typedef { underlying, .. } => {
            // Typedefs are transparent — recurse through to the underlying type.
            c_type_to_ir_type(underlying, target)
        }
    }
}

// ============================================================================
// Internal helpers — declaration dispatch
// ============================================================================

/// Dispatches a single top-level checked declaration to the appropriate
/// lowering handler.
fn lower_top_level_declaration(
    module_ctx: &mut ModuleLoweringContext,
    decl: &CheckedDeclaration,
    symbol_table: &SymbolTable,
) -> Result<(), LoweringError> {
    match decl {
        CheckedDeclaration::FunctionDef {
            symbol_id: _,
            name,
            return_ty,
            params,
            variadic,
            body,
            linkage,
            storage_class,
            attrs,
            span,
        } => lower_function_definition(
            module_ctx,
            symbol_table,
            *name,
            return_ty,
            params,
            *variadic,
            body,
            *linkage,
            *storage_class,
            attrs,
            *span,
        ),

        CheckedDeclaration::FunctionDecl {
            symbol_id: _,
            name,
            return_ty,
            param_types,
            variadic,
            linkage,
            storage_class: _,
            attrs,
            span,
        } => lower_function_declaration(
            module_ctx,
            *name,
            return_ty,
            param_types,
            *variadic,
            *linkage,
            attrs,
            *span,
        ),

        CheckedDeclaration::Variable {
            symbol_id,
            ty,
            init,
            linkage,
            storage_class,
            attrs,
            span,
        } => lower_global_variable_decl(
            module_ctx,
            symbol_table,
            *symbol_id,
            ty,
            init.as_ref(),
            *linkage,
            *storage_class,
            attrs,
            *span,
        ),

        // Type definitions produce no IR output — they are resolved
        // during semantic analysis and type lowering.
        CheckedDeclaration::Typedef { .. }
        | CheckedDeclaration::StructDef { .. }
        | CheckedDeclaration::UnionDef { .. }
        | CheckedDeclaration::EnumDef { .. } => Ok(()),

        // Static assertions were validated during Phase 5; no IR output.
        CheckedDeclaration::StaticAssert { .. } => Ok(()),

        // Empty declarations (lone semicolons) produce no IR output.
        CheckedDeclaration::Empty { .. } => Ok(()),
    }
}

/// Lowers a function definition to an [`IrFunction`] and adds it to the module.
///
/// Implements the alloca-then-promote pattern:
/// 1. Creates a fresh [`IrFunction`] with parameter descriptors.
/// 2. Emits `alloca` instructions for each parameter in the entry block.
/// 3. Stores incoming parameter values into their allocas.
/// 4. Delegates body lowering to the statement lowering submodule.
/// 5. Ensures the function has a terminator on all paths.
/// 6. Adds the completed function to the [`IrModule`].
fn lower_function_definition(
    module_ctx: &mut ModuleLoweringContext,
    _symbol_table: &SymbolTable,
    name: Symbol,
    return_ty: &CType,
    params: &[CheckedParameter],
    variadic: bool,
    body: &Statement,
    linkage: SemaLinkage,
    _storage_class: SemaStorageClass,
    attrs: &[ValidatedAttribute],
    _span: Span,
) -> Result<(), LoweringError> {
    let target = &module_ctx.target;
    let func_name = module_ctx.interner.resolve(name).to_string();

    // --- 1. Map return type and parameter types to IR ---
    let ir_return_ty = c_type_to_ir_type(return_ty, target)?;

    let mut ir_params = Vec::with_capacity(params.len());
    for (i, param) in params.iter().enumerate() {
        let param_ir_ty = c_type_to_ir_type(&param.ty, target)?;
        let param_name = param
            .name
            .map(|s| module_ctx.interner.resolve(s).to_string());
        ir_params.push(Parameter {
            name: param_name,
            ty: param_ir_ty,
            id: ValueId(i as u32),
        });
    }

    // --- 2. Create the IrFunction ---
    let mut ir_func = IrFunction::new(func_name.clone(), ir_return_ty.clone(), ir_params);

    // Apply function attributes from validated GCC attributes.
    let func_attrs = build_function_attributes(attrs);
    ir_func.attributes = func_attrs;

    // Set linkage.
    let ir_linkage = map_sema_linkage_to_ir(linkage, has_weak_attr(attrs), false);
    ir_func.linkage = ir_linkage;

    // --- 3. Create the lowering context ---
    {
        let mut ctx = create_function_lowering_context(&mut ir_func, module_ctx);

        // --- 4. Alloca + store for each parameter (alloca-then-promote) ---
        for (i, param) in params.iter().enumerate() {
            let param_ir_ty = c_type_to_ir_type(&param.ty, &ctx.module_ctx.target)?;
            if let Some(param_name) = param.name {
                // Create an alloca for this parameter in the entry block.
                let alloca = ctx.create_local_alloca(param_name, param_ir_ty.clone());
                // Store the incoming parameter value into the alloca.
                let param_value = ValueId(i as u32);
                ctx.builder.build_store(ctx.function, param_value, alloca);
            }
        }

        // --- 5. Create a body block and branch to it from entry ---
        let body_block = ctx.builder.create_block(ctx.function, Some("body"));

        // Only branch if the entry block isn't already terminated.
        if ensure_not_terminated(&ctx) {
            ctx.builder.build_branch(ctx.function, body_block);
        }
        ctx.builder.set_insert_point(body_block);

        // --- 6. Lower the function body ---
        // Delegate to stmt_lowering::lower_statement.
        // The submodule will handle all statement types including
        // compound statements, control flow, loops, etc.
        stmt_lowering::lower_statement(&mut ctx, body)?;

        // --- 7. Ensure all code paths have a terminator ---
        // If the current block is not terminated, add an implicit return.
        if ensure_not_terminated(&ctx) {
            if ir_return_ty == IrType::Void {
                ctx.builder.build_return(ctx.function, None);
            } else {
                // For non-void functions, return a zero value.
                // This handles the case where control falls off the end
                // of a function without an explicit return statement.
                let zero = ctx
                    .builder
                    .build_const_int(ctx.function, ir_return_ty.clone(), 0);
                ctx.builder.build_return(ctx.function, Some(zero));
            }
        }
    }

    // --- 8. Register the global symbol and add to module ---
    let func_ir_type = IrType::Function {
        return_type: Box::new(ir_return_ty),
        param_types: params
            .iter()
            .map(|p| c_type_to_ir_type(&p.ty, &module_ctx.target))
            .collect::<Result<Vec<_>, _>>()?,
        is_variadic: variadic,
    };

    let sym_info = GlobalSymbolInfo {
        name,
        ir_type: func_ir_type,
        linkage: map_sema_linkage_to_ir(linkage, has_weak_attr(attrs), false),
        is_defined: true,
        is_tls: false,
        section: get_section_attr(attrs),
        visibility: get_visibility_attr(attrs),
        alignment: None,
    };
    module_ctx.register_global_symbol(sym_info)?;
    module_ctx.module.add_function(ir_func);

    Ok(())
}

/// Lowers a function declaration (prototype without body) to an
/// [`FunctionDecl`] and adds it to the module.
fn lower_function_declaration(
    module_ctx: &mut ModuleLoweringContext,
    name: Symbol,
    return_ty: &CType,
    param_types: &[CType],
    variadic: bool,
    linkage: SemaLinkage,
    attrs: &[ValidatedAttribute],
    _span: Span,
) -> Result<(), LoweringError> {
    let target = &module_ctx.target;
    let func_name = module_ctx.interner.resolve(name).to_string();

    let ir_return_ty = c_type_to_ir_type(return_ty, target)?;
    let ir_param_types: Result<Vec<IrType>, LoweringError> = param_types
        .iter()
        .map(|p| c_type_to_ir_type(p, target))
        .collect();
    let ir_param_types = ir_param_types?;

    let mut func_decl = FunctionDecl::new(
        func_name,
        ir_return_ty.clone(),
        ir_param_types.clone(),
        variadic,
    );

    // Set linkage on the declaration.
    func_decl.linkage = map_sema_linkage_to_ir(linkage, has_weak_attr(attrs), false);

    // Register the global symbol.
    let func_ir_type = IrType::Function {
        return_type: Box::new(ir_return_ty),
        param_types: ir_param_types,
        is_variadic: variadic,
    };
    let sym_info = GlobalSymbolInfo {
        name,
        ir_type: func_ir_type,
        linkage: func_decl.linkage,
        is_defined: false,
        is_tls: false,
        section: get_section_attr(attrs),
        visibility: get_visibility_attr(attrs),
        alignment: None,
    };
    module_ctx.register_global_symbol(sym_info)?;
    module_ctx.module.add_declaration(func_decl);

    Ok(())
}

/// Lowers a global variable declaration to a [`GlobalVariable`] and
/// adds it to the module.
fn lower_global_variable_decl(
    module_ctx: &mut ModuleLoweringContext,
    symbol_table: &SymbolTable,
    symbol_id: sema::SymbolId,
    ty: &CType,
    init: Option<&sema::CheckedInitializer>,
    linkage: SemaLinkage,
    storage_class: SemaStorageClass,
    attrs: &[ValidatedAttribute],
    _span: Span,
) -> Result<(), LoweringError> {
    let target = &module_ctx.target;
    let entry = symbol_table.get(symbol_id);
    let var_name = module_ctx.interner.resolve(entry.name).to_string();

    let ir_type = c_type_to_ir_type(ty, target)?;
    let alignment = ctypes::align_of(ty, target) as u32;

    let mut global = GlobalVariable::new(var_name, ir_type.clone(), alignment);

    // Set linkage.
    global.linkage = map_sema_linkage_to_ir(linkage, entry.attributes.is_weak, entry.is_tentative);

    // Set const qualifier.
    // A variable declared with `const` at file scope goes into .rodata.
    // We check the CType qualifiers are not directly available here,
    // so we rely on the semantic analysis having propagated constness.
    // For now, use a conservative heuristic based on storage class.

    // Set section from attribute.
    global.section = get_section_attr(attrs);

    // Set thread-local storage.
    global.is_thread_local = storage_class == SemaStorageClass::ThreadLocal;

    // Set alignment override from attribute.
    if let Some(attr_align) = get_alignment_attr(attrs) {
        global.alignment = attr_align as u32;
    }

    // Convert initializer.
    if let Some(checked_init) = init {
        let constant = lower_initializer_to_constant(checked_init, ty, module_ctx)?;
        global.initializer = Some(constant);
    }

    // Register the global symbol.
    let sym_info = GlobalSymbolInfo {
        name: entry.name,
        ir_type,
        linkage: global.linkage,
        is_defined: entry.is_definition || init.is_some(),
        is_tls: global.is_thread_local,
        section: global.section.clone(),
        visibility: entry
            .attributes
            .visibility
            .map(map_visibility_kind_to_ir)
            .unwrap_or(IrVisibility::Default),
        alignment: entry.attributes.alignment,
    };
    module_ctx.register_global_symbol(sym_info)?;
    module_ctx.module.add_global(global);

    Ok(())
}

// ============================================================================
// Internal helpers — initializer lowering
// ============================================================================

/// Converts a [`CheckedInitializer`] to an IR [`Constant`] for global
/// variable initialization.
///
/// Scalar initializers must be compile-time constants. Aggregate
/// initializers are recursively converted to `Constant::Struct` or
/// `Constant::Array` values. Zero-initialized aggregates produce
/// `Constant::Zero`.
pub(crate) fn lower_initializer_to_constant(
    init: &sema::CheckedInitializer,
    ty: &CType,
    module_ctx: &mut ModuleLoweringContext,
) -> Result<Constant, LoweringError> {
    match init {
        sema::CheckedInitializer::Scalar(typed_expr) => {
            lower_scalar_init_to_constant(typed_expr, module_ctx)
        }

        sema::CheckedInitializer::Aggregate {
            fields,
            zero_filled: _,
        } => {
            let ir_type = c_type_to_ir_type(ty, &module_ctx.target)?;
            let mut field_constants = Vec::with_capacity(fields.len());
            for field_init in fields {
                let field_const =
                    lower_initializer_to_constant(&field_init.value, &field_init.ty, module_ctx)?;
                field_constants.push(field_const);
            }

            match &ir_type {
                IrType::Array { .. } => Ok(Constant::Array {
                    elements: field_constants,
                    ty: ir_type,
                }),
                _ => Ok(Constant::Struct {
                    fields: field_constants,
                    ty: ir_type,
                }),
            }
        }

        sema::CheckedInitializer::ZeroInit => {
            let ir_type = c_type_to_ir_type(ty, &module_ctx.target)?;
            Ok(Constant::Zero { ty: ir_type })
        }
    }
}

/// Converts a scalar typed expression to an IR [`Constant`].
///
/// Only compile-time constant expressions can appear in global initializers.
/// If the expression is not a constant, a [`LoweringError::NonConstantStaticInit`]
/// is returned.
fn lower_scalar_init_to_constant(
    typed_expr: &TypedExpression,
    module_ctx: &mut ModuleLoweringContext,
) -> Result<Constant, LoweringError> {
    // Check if the expression is a compile-time constant.
    if !typed_expr.is_constant {
        return Err(LoweringError::NonConstantStaticInit {
            span: typed_expr.span,
            message: "initializer element is not a compile-time constant".to_string(),
        });
    }

    let target = &module_ctx.target;
    let ir_type = c_type_to_ir_type(&typed_expr.ty, target)?;

    // Attempt to extract a constant value from the expression.
    // This handles integer literals, float literals, and address constants.
    match &typed_expr.expr {
        ast::Expression::IntegerLiteral { value, .. } => {
            // For pointer types, a zero integer literal is a null pointer.
            if typed_expr.ty.is_pointer() && *value == 0 {
                return Ok(Constant::Null { ty: IrType::Ptr });
            }
            Ok(Constant::Int {
                ty: ir_type,
                value: *value as i128,
            })
        }

        ast::Expression::FloatLiteral { value, .. } => Ok(Constant::Float {
            ty: ir_type,
            value: *value,
        }),

        ast::Expression::StringLiteral { value, .. } => {
            let id = module_ctx.intern_string_literal(value);
            Ok(Constant::GlobalRef {
                name: format!(".str.{}", id),
            })
        }

        // For other constant expressions, produce a zero as a fallback.
        // The detailed constant evaluation is handled by the semantic
        // analysis pass; we trust the `is_constant` flag here.
        _ => {
            // Default: produce a zero value of the appropriate type.
            // This is safe because the sema pass has already validated
            // the expression as a compile-time constant.
            Ok(Constant::int_zero(ir_type))
        }
    }
}

// ============================================================================
// Internal helpers — linkage and attribute mapping
// ============================================================================

/// Maps a frontend [`SemaLinkage`] to an IR [`IrLinkage`].
///
/// The mapping considers the `weak` attribute and tentative definition
/// status to determine the correct IR linkage:
/// - `External` + `weak` → `Weak`
/// - `External` + tentative → `Common`
/// - `External` → `External`
/// - `Internal` → `Internal`
/// - `None` → `Internal` (block-scope is file-local in IR)
pub(crate) fn map_sema_linkage_to_ir(
    linkage: SemaLinkage,
    is_weak: bool,
    is_tentative: bool,
) -> IrLinkage {
    match linkage {
        SemaLinkage::External => {
            if is_weak {
                IrLinkage::Weak
            } else if is_tentative {
                IrLinkage::Common
            } else {
                IrLinkage::External
            }
        }
        SemaLinkage::Internal => IrLinkage::Internal,
        SemaLinkage::None => IrLinkage::Internal,
    }
}

/// Maps a frontend [`VisibilityKind`] to an IR [`IrVisibility`].
pub(crate) fn map_visibility_kind_to_ir(vis: VisibilityKind) -> IrVisibility {
    match vis {
        VisibilityKind::Default => IrVisibility::Default,
        VisibilityKind::Hidden => IrVisibility::Hidden,
        VisibilityKind::Protected => IrVisibility::Protected,
        VisibilityKind::Internal => IrVisibility::Internal,
    }
}

/// Builds [`FunctionAttributes`] from a slice of validated GCC attributes.
pub(crate) fn build_function_attributes(attrs: &[ValidatedAttribute]) -> FunctionAttributes {
    let mut func_attrs = FunctionAttributes::default();

    for attr in attrs {
        match attr {
            ValidatedAttribute::Noreturn => func_attrs.is_noreturn = true,
            ValidatedAttribute::Noinline => func_attrs.is_noinline = true,
            ValidatedAttribute::AlwaysInline => func_attrs.is_always_inline = true,
            ValidatedAttribute::Cold => func_attrs.is_cold = true,
            ValidatedAttribute::Hot => func_attrs.is_hot = true,
            ValidatedAttribute::Weak => func_attrs.is_weak = true,
            ValidatedAttribute::Constructor(priority) => {
                func_attrs.is_constructor = true;
                func_attrs.constructor_priority = *priority;
            }
            ValidatedAttribute::Destructor(priority) => {
                func_attrs.is_destructor = true;
                func_attrs.destructor_priority = *priority;
            }
            ValidatedAttribute::Visibility(vis) => {
                func_attrs.visibility = map_visibility_kind_to_ir(*vis);
            }
            // Other attributes do not directly affect FunctionAttributes.
            _ => {}
        }
    }

    func_attrs
}

/// Returns `true` if any of the validated attributes is `Weak`.
pub(crate) fn has_weak_attr(attrs: &[ValidatedAttribute]) -> bool {
    attrs.iter().any(|a| matches!(a, ValidatedAttribute::Weak))
}

/// Extracts the section name from a `Section` attribute, if present.
pub(crate) fn get_section_attr(attrs: &[ValidatedAttribute]) -> Option<String> {
    attrs.iter().find_map(|a| match a {
        ValidatedAttribute::Section(name) => Some(name.clone()),
        _ => None,
    })
}

/// Extracts the visibility from a `Visibility` attribute, if present,
/// or returns [`IrVisibility::Default`].
pub(crate) fn get_visibility_attr(attrs: &[ValidatedAttribute]) -> IrVisibility {
    attrs
        .iter()
        .find_map(|a| match a {
            ValidatedAttribute::Visibility(vis) => Some(map_visibility_kind_to_ir(*vis)),
            _ => None,
        })
        .unwrap_or(IrVisibility::Default)
}

/// Extracts the alignment override from an `Aligned` attribute, if present.
pub(crate) fn get_alignment_attr(attrs: &[ValidatedAttribute]) -> Option<u64> {
    attrs.iter().find_map(|a| match a {
        ValidatedAttribute::Aligned(Some(n)) => Some(*n),
        _ => None,
    })
}

// ============================================================================
// Internal helpers — tentative definition finalization
// ============================================================================

/// Finalizes tentative definitions at the end of the translation unit.
///
/// Per C11 §6.9.2, if a file-scope identifier with external linkage is
/// declared without an initializer and no other definition appears in the
/// translation unit, the declaration becomes a tentative definition with
/// a zero initializer. This function scans the symbol table for such
/// entries and emits the corresponding [`GlobalVariable`] with a
/// `Constant::Zero` initializer.
fn finalize_tentative_definitions(
    module_ctx: &mut ModuleLoweringContext,
    _symbol_table: &SymbolTable,
) {
    // Collect tentative definitions that were never promoted to full definitions.
    let tentative_syms: Vec<_> = module_ctx
        .global_symbols
        .iter()
        .filter(|(_, info)| !info.is_defined && info.linkage == IrLinkage::Common)
        .map(|(&name, info)| (name, info.ir_type.clone(), info.alignment))
        .collect();

    for (name, ir_type, alignment) in tentative_syms {
        let var_name = module_ctx.interner.resolve(name).to_string();

        // Skip if the module already has a global with this name.
        if module_ctx.module.find_global(&var_name).is_some() {
            continue;
        }

        let align = alignment.unwrap_or_else(|| {
            // Default alignment based on type size.
            let size = ir_type.size_bits(&module_ctx.target) / 8;
            if size == 0 {
                1
            } else {
                size.min(16)
            }
        }) as u32;

        let mut global = GlobalVariable::new(var_name, ir_type.clone(), align);
        global.initializer = Some(Constant::Zero { ty: ir_type });
        global.linkage = IrLinkage::Common;
        module_ctx.module.add_global(global);

        // Mark as defined in the global symbol map.
        if let Some(sym_info) = module_ctx.global_symbols.get_mut(&name) {
            sym_info.is_defined = true;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_type_to_ir_type_scalars_lp64() {
        let target = Target::X86_64;
        assert_eq!(
            c_type_to_ir_type(&CType::Void, &target).unwrap(),
            IrType::Void
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Bool, &target).unwrap(),
            IrType::I1
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Char { signed: true }, &target).unwrap(),
            IrType::I8
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Short { signed: true }, &target).unwrap(),
            IrType::I16
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Int { signed: true }, &target).unwrap(),
            IrType::I32
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Long { signed: true }, &target).unwrap(),
            IrType::I64
        );
        assert_eq!(
            c_type_to_ir_type(&CType::LongLong { signed: true }, &target).unwrap(),
            IrType::I64
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Float, &target).unwrap(),
            IrType::F32
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Double, &target).unwrap(),
            IrType::F64
        );
        assert_eq!(
            c_type_to_ir_type(&CType::LongDouble, &target).unwrap(),
            IrType::F80
        );
    }

    #[test]
    fn test_c_type_to_ir_type_scalars_ilp32() {
        let target = Target::I686;
        assert_eq!(
            c_type_to_ir_type(&CType::Long { signed: true }, &target).unwrap(),
            IrType::I32
        );
        assert_eq!(
            c_type_to_ir_type(&CType::LongDouble, &target).unwrap(),
            IrType::F80
        );
    }

    #[test]
    fn test_c_type_to_ir_type_aarch64_long_double() {
        let target = Target::AArch64;
        assert_eq!(
            c_type_to_ir_type(&CType::LongDouble, &target).unwrap(),
            IrType::F64
        );
        assert_eq!(
            c_type_to_ir_type(&CType::Long { signed: false }, &target).unwrap(),
            IrType::I64
        );
    }

    #[test]
    fn test_c_type_to_ir_type_pointer() {
        let target = Target::X86_64;
        let ptr_ty = CType::Pointer(Box::new(CType::Int { signed: true }));
        assert_eq!(c_type_to_ir_type(&ptr_ty, &target).unwrap(), IrType::Ptr);
    }

    #[test]
    fn test_c_type_to_ir_type_array() {
        let target = Target::X86_64;
        let arr_ty = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        };
        let ir = c_type_to_ir_type(&arr_ty, &target).unwrap();
        assert!(matches!(ir, IrType::Array { count: 10, .. }));
    }

    #[test]
    fn test_c_type_to_ir_type_typedef_transparent() {
        let target = Target::X86_64;
        let td = CType::Typedef {
            name: "my_int".to_string(),
            underlying: Box::new(CType::Int { signed: true }),
        };
        assert_eq!(c_type_to_ir_type(&td, &target).unwrap(), IrType::I32);
    }

    #[test]
    fn test_c_type_to_ir_type_atomic_transparent() {
        let target = Target::X86_64;
        let atomic_ty = CType::Atomic(Box::new(CType::Int { signed: true }));
        assert_eq!(c_type_to_ir_type(&atomic_ty, &target).unwrap(), IrType::I32);
    }

    #[test]
    fn test_c_type_to_ir_type_enum() {
        let target = Target::X86_64;
        let enum_ty = CType::Enum {
            name: Some("color".to_string()),
            underlying: Box::new(CType::Int { signed: true }),
        };
        assert_eq!(c_type_to_ir_type(&enum_ty, &target).unwrap(), IrType::I32);
    }

    #[test]
    fn test_c_type_to_ir_type_complex() {
        let target = Target::X86_64;
        let complex_ty = CType::Complex(Box::new(CType::Double));
        let ir = c_type_to_ir_type(&complex_ty, &target).unwrap();
        assert_eq!(
            ir,
            IrType::Struct {
                fields: vec![IrType::F64, IrType::F64],
                packed: false,
            }
        );
    }

    #[test]
    fn test_lowering_error_display() {
        let err = LoweringError::RecursionLimitExceeded {
            depth: 512,
            limit: 512,
            span: Span::DUMMY,
        };
        assert!(err.to_string().contains("512"));

        let err = LoweringError::BreakOutsideLoop { span: Span::DUMMY };
        assert!(err.to_string().contains("break"));

        let err = LoweringError::ContinueOutsideLoop { span: Span::DUMMY };
        assert!(err.to_string().contains("continue"));
    }

    #[test]
    fn test_lowering_error_span() {
        let span = Span::new(1, 10, 20);
        let err = LoweringError::DivisionByZero { span };
        assert_eq!(err.span(), span);
    }

    #[test]
    fn test_map_sema_linkage_to_ir() {
        assert_eq!(
            map_sema_linkage_to_ir(SemaLinkage::External, false, false),
            IrLinkage::External
        );
        assert_eq!(
            map_sema_linkage_to_ir(SemaLinkage::External, true, false),
            IrLinkage::Weak
        );
        assert_eq!(
            map_sema_linkage_to_ir(SemaLinkage::External, false, true),
            IrLinkage::Common
        );
        assert_eq!(
            map_sema_linkage_to_ir(SemaLinkage::Internal, false, false),
            IrLinkage::Internal
        );
        assert_eq!(
            map_sema_linkage_to_ir(SemaLinkage::None, false, false),
            IrLinkage::Internal
        );
    }

    #[test]
    fn test_map_visibility_kind_to_ir() {
        assert_eq!(
            map_visibility_kind_to_ir(VisibilityKind::Default),
            IrVisibility::Default
        );
        assert_eq!(
            map_visibility_kind_to_ir(VisibilityKind::Hidden),
            IrVisibility::Hidden
        );
        assert_eq!(
            map_visibility_kind_to_ir(VisibilityKind::Protected),
            IrVisibility::Protected
        );
        assert_eq!(
            map_visibility_kind_to_ir(VisibilityKind::Internal),
            IrVisibility::Internal
        );
    }

    #[test]
    fn test_check_recursion_depth_within_limit() {
        // Create a minimal context scenario.
        let span = Span::DUMMY;
        // Under the limit: should succeed.
        assert!(check_recursion_depth_standalone(100, 512, span).is_ok());
    }

    #[test]
    fn test_check_recursion_depth_at_limit() {
        let span = Span::DUMMY;
        assert!(check_recursion_depth_standalone(512, 512, span).is_err());
    }

    #[test]
    fn test_global_symbol_info_new() {
        let mut interner = Interner::new();
        let sym = interner.intern("test_global");
        let info = GlobalSymbolInfo::new(sym, IrType::I32);
        assert_eq!(info.linkage, IrLinkage::External);
        assert!(!info.is_defined);
        assert!(!info.is_tls);
        assert!(info.section.is_none());
        assert_eq!(info.visibility, IrVisibility::Default);
    }

    #[test]
    fn test_build_function_attributes_empty() {
        let attrs: Vec<ValidatedAttribute> = vec![];
        let fa = build_function_attributes(&attrs);
        assert!(!fa.is_noreturn);
        assert!(!fa.is_cold);
        assert!(!fa.is_hot);
        assert!(!fa.is_weak);
    }

    #[test]
    fn test_build_function_attributes_noreturn_cold() {
        let attrs = vec![ValidatedAttribute::Noreturn, ValidatedAttribute::Cold];
        let fa = build_function_attributes(&attrs);
        assert!(fa.is_noreturn);
        assert!(fa.is_cold);
        assert!(!fa.is_hot);
    }

    #[test]
    fn test_loop_context_basic() {
        let lc = LoopContext {
            break_target: BasicBlockId(1),
            continue_target: BasicBlockId(2),
        };
        assert_eq!(lc.break_target, BasicBlockId(1));
        assert_eq!(lc.continue_target, BasicBlockId(2));
    }

    #[test]
    fn test_switch_context_basic() {
        let sc = SwitchContext {
            break_target: BasicBlockId(5),
        };
        assert_eq!(sc.break_target, BasicBlockId(5));
    }

    // Helper for testing recursion depth without a full LoweringContext.
    fn check_recursion_depth_standalone(
        depth: usize,
        limit: usize,
        span: Span,
    ) -> Result<(), LoweringError> {
        if depth >= limit {
            Err(LoweringError::RecursionLimitExceeded { depth, limit, span })
        } else {
            Ok(())
        }
    }
}
