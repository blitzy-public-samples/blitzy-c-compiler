//! # IR Function Representation
//!
//! This module defines [`IrFunction`] — the central data structure representing
//! a single C function at the IR level.  An `IrFunction` contains the function's
//! name, parameter list (with SSA value numbers), return type, an ordered
//! sequence of [`BasicBlock`]s, and metadata such as calling convention,
//! linkage, visibility, and attribute-derived flags.
//!
//! ## Entry Block and the Alloca-Then-Promote Pattern
//!
//! The first basic block (`blocks[0]`) is always the **entry block**.  During
//! Phase 6 (AST-to-IR lowering), every local variable is emitted as an
//! [`Instruction::Alloca`] instruction placed in this entry block.  The
//! subsequent mem2reg pass (Phase 7) promotes eligible allocas — those that
//! are scalar, non-address-taken, and non-volatile — to SSA virtual registers,
//! inserting phi nodes at dominance frontiers.  This mirrors the LLVM approach
//! to SSA construction and is a non-negotiable architectural mandate.
//!
//! ## Block Ordering
//!
//! Block order within the `blocks` vector is significant:
//! - `blocks[0]` is the function entry point.
//! - Iteration order determines the layout of emitted machine code.
//! - The [`reverse_postorder`](IrFunction::reverse_postorder) traversal
//!   computes a canonical visitation order used by the dominator tree
//!   algorithm and optimization passes.
//!
//! ## Calling Convention
//!
//! The [`CallingConvention`] enum annotates which ABI the function uses.
//! Currently only the standard C calling convention is supported.  The
//! actual ABI rules (register assignment, struct passing, etc.) are
//! implemented in the architecture-specific `abi.rs` files in the backend.
//!
//! ## Linkage and Visibility
//!
//! [`Linkage`] controls symbol resolution semantics (external, internal,
//! weak, common) while [`Visibility`] controls ELF symbol visibility
//! (default, hidden, protected).  These directly map to the corresponding
//! C storage class specifiers and GCC `__attribute__((visibility(...)))`.
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::ir::basic_block`, `crate::ir::types`,
//! `crate::ir::instructions`, and the Rust standard library — no external
//! crates are used.

use crate::ir::basic_block::BasicBlock;
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// CallingConvention — function ABI annotation
// ===========================================================================

/// Calling convention annotation for an IR function.
///
/// Specifies which ABI rules govern parameter passing, return values, and
/// register usage for the function.  The actual ABI logic (register
/// classification, struct decomposition, stack alignment) is implemented
/// by the architecture-specific `abi.rs` modules in the backend.
///
/// Currently only the standard C calling convention is supported, which
/// maps to System V AMD64 on x86-64, cdecl on i686, AAPCS64 on AArch64,
/// and LP64D on RISC-V 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    /// Standard C calling convention.
    ///
    /// This is the default for all functions.  Architecture-specific
    /// details are handled by the backend's ABI module.
    C,
}

impl Default for CallingConvention {
    #[inline]
    fn default() -> Self {
        CallingConvention::C
    }
}

impl std::fmt::Display for CallingConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallingConvention::C => write!(f, "ccc"),
        }
    }
}

// ===========================================================================
// Linkage — symbol resolution semantics
// ===========================================================================

/// Symbol linkage kind — controls how the linker resolves references to
/// the function across translation units.
///
/// Maps directly to C storage class specifiers and GCC attributes:
/// - `External` — visible to other translation units (default for non-`static` functions)
/// - `Internal` — file-scoped (`static` functions)
/// - `Weak` — `__attribute__((weak))`, can be overridden by a strong definition
/// - `Common` — tentative definition semantics (for uninitialized globals)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage {
    /// Externally visible — the default for non-`static` functions.
    /// The symbol is exported and can be referenced from other translation units.
    External,

    /// Internal linkage — `static` functions/variables.
    /// The symbol is local to the current translation unit and is not
    /// visible to the linker outside of the defining object file.
    Internal,

    /// Weak linkage — `__attribute__((weak))`.
    /// The symbol can be overridden by a strong (non-weak) definition
    /// from another translation unit.  If no strong definition exists,
    /// the weak definition is used.
    Weak,

    /// Common linkage — tentative definition semantics.
    /// Used for uninitialized global variables that may have multiple
    /// tentative definitions across translation units.  The linker merges
    /// them into a single definition with the largest size.
    Common,
}

impl Default for Linkage {
    #[inline]
    fn default() -> Self {
        Linkage::External
    }
}

impl std::fmt::Display for Linkage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Linkage::External => write!(f, "external"),
            Linkage::Internal => write!(f, "internal"),
            Linkage::Weak => write!(f, "weak"),
            Linkage::Common => write!(f, "common"),
        }
    }
}

// ===========================================================================
// Visibility — ELF symbol visibility
// ===========================================================================

/// ELF symbol visibility — controls how the dynamic linker resolves
/// references to the symbol in shared libraries.
///
/// Maps to GCC's `__attribute__((visibility("...")))`:
/// - `Default` — normal visibility, can be preempted by another definition
/// - `Hidden` — not visible outside the shared object
/// - `Protected` — visible but cannot be preempted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Default visibility — the symbol is visible to other shared objects
    /// and can be preempted (overridden) by a definition in another
    /// shared object or the main executable.
    Default,

    /// Hidden visibility — the symbol is not exported from the shared
    /// object.  References within the same shared object are resolved
    /// directly, bypassing the GOT/PLT.  This is the most restrictive
    /// visibility and enables the best code generation for PIC.
    Hidden,

    /// Protected visibility — the symbol is visible to other shared
    /// objects but cannot be preempted.  References within the defining
    /// shared object are resolved directly, but the symbol still appears
    /// in the dynamic symbol table.
    Protected,
}

impl Default for Visibility {
    #[inline]
    fn default() -> Self {
        Visibility::Default
    }
}

impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Default => write!(f, "default"),
            Visibility::Hidden => write!(f, "hidden"),
            Visibility::Protected => write!(f, "protected"),
        }
    }
}

// ===========================================================================
// FunctionParam — function parameter with SSA value
// ===========================================================================

/// A function parameter at the IR level.
///
/// Each parameter carries its name (for debug info and diagnostics),
/// its IR type, and the SSA [`Value`] number assigned to it.  Parameter
/// values are the first SSA values allocated in a function — they are
/// "defined" at function entry and used as initial reaching definitions
/// during SSA construction.
///
/// # Example
///
/// For `int add(int a, int b)`:
/// - Parameter 0: `FunctionParam { name: "a", ty: IrType::I32, value: Value(0) }`
/// - Parameter 1: `FunctionParam { name: "b", ty: IrType::I32, value: Value(1) }`
#[derive(Debug, Clone)]
pub struct FunctionParam {
    /// Parameter name from the C source (for debug info and diagnostics).
    pub name: String,

    /// IR type of the parameter (resolved from the C type by the lowering phase).
    pub ty: IrType,

    /// SSA value number assigned to this parameter.
    ///
    /// Parameter values are numbered sequentially starting from 0.  These
    /// values serve as the initial definitions for the parameter variables
    /// in the SSA form constructed by mem2reg.
    pub value: Value,
}

impl FunctionParam {
    /// Creates a new function parameter.
    ///
    /// # Parameters
    ///
    /// - `name`: Parameter name from the C source.
    /// - `ty`: IR type of the parameter.
    /// - `value`: SSA value number for this parameter.
    #[inline]
    pub fn new(name: String, ty: IrType, value: Value) -> Self {
        FunctionParam { name, ty, value }
    }
}

impl std::fmt::Display for FunctionParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.ty, self.value)
    }
}

// ===========================================================================
// LocalVarDebugInfo — debug metadata for local variables
// ===========================================================================

/// Debug information for a local variable, threaded from IR lowering to
/// the DWARF emitter so that `DW_TAG_variable` entries can be generated.
///
/// Each entry corresponds to one user-declared local variable in the
/// original C source.  The `alloca_index` records the variable's
/// sequential position among the function's alloca instructions (0-based),
/// which the DWARF emitter uses to compute a `DW_OP_fbreg` stack offset.
#[derive(Debug, Clone)]
pub struct LocalVarDebugInfo {
    /// Variable name as written in the C source.
    pub name: String,
    /// IR type of the variable (after C-to-IR type mapping).
    pub ir_type: IrType,
    /// Zero-based index of this variable among the function's allocas.
    /// Used to derive a stack frame offset for the DWARF location.
    pub alloca_index: u32,
    /// Source line number where the variable is declared (1-based).
    pub decl_line: u32,
}

// ===========================================================================
// IrFunction — the central IR function representation
// ===========================================================================

/// A single C function at the IR level.
///
/// `IrFunction` is the primary container for a function's IR — it holds
/// the function signature (name, parameters, return type), the ordered
/// list of basic blocks forming the control flow graph, calling convention
/// annotation, linkage/visibility metadata, and attribute-derived flags.
///
/// # Entry Block Convention
///
/// `blocks[0]` is **always** the entry block.  During Phase 6 (AST-to-IR
/// lowering), all `alloca` instructions for local variables are placed in
/// this entry block.  The mem2reg pass (Phase 7) then promotes eligible
/// allocas to SSA virtual registers.
///
/// # Block Ordering
///
/// The order of blocks in the `blocks` vector is significant:
/// - `blocks[0]` is the function entry point.
/// - Code generation emits blocks in vector order.
/// - [`reverse_postorder`](IrFunction::reverse_postorder) provides the
///   canonical traversal order for analysis passes.
///
/// # SSA Value Numbering
///
/// SSA values are numbered sequentially starting from 0.  The first
/// `param_count()` values are assigned to function parameters.  The
/// `value_count` field tracks the total number of SSA values allocated,
/// serving as the next available value number when creating new
/// instructions.
#[derive(Debug, Clone)]
pub struct IrFunction {
    /// Function name (symbol name in the output object file).
    pub name: String,

    /// Function parameters with their names, types, and SSA values.
    pub params: Vec<FunctionParam>,

    /// Return type of the function.
    pub return_type: IrType,

    /// Ordered list of basic blocks forming the control flow graph.
    ///
    /// `blocks[0]` is always the entry block where alloca instructions
    /// are placed during IR lowering.  Block order is significant for
    /// code generation — blocks are emitted in vector order.
    pub blocks: Vec<BasicBlock>,

    /// Calling convention annotation.
    ///
    /// Specifies which ABI rules apply to this function.  The actual
    /// register assignment and struct passing logic is in the backend.
    pub calling_convention: CallingConvention,

    /// Whether the function accepts a variable number of arguments
    /// (C variadic function with `...` in the parameter list).
    pub is_variadic: bool,

    /// Symbol linkage — controls cross-TU visibility and resolution.
    pub linkage: Linkage,

    /// ELF symbol visibility — controls dynamic linker behavior.
    pub visibility: Visibility,

    /// Whether the function has `__attribute__((noreturn))` or `_Noreturn`.
    ///
    /// Functions marked noreturn are guaranteed to never return to their
    /// caller.  This enables the backend to omit epilogue code and
    /// informs optimization passes that code after a call to this
    /// function is unreachable.
    pub is_noreturn: bool,

    /// Optional section override from `__attribute__((section("...")))`.
    ///
    /// When set, the function is placed in the named ELF section instead
    /// of the default `.text` section.  Used heavily by the Linux kernel
    /// for `__init`, `__exit`, and other special sections.
    pub section: Option<String>,

    /// Optional alignment override for the function entry point.
    ///
    /// When set, the function's first instruction is aligned to this
    /// byte boundary in the output object file.  Must be a power of two.
    pub alignment: Option<usize>,

    /// Number of local variables (alloca instructions in the entry block).
    ///
    /// Tracked during IR lowering for mem2reg analysis — the mem2reg pass
    /// uses this to pre-allocate data structures for alloca promotion.
    pub local_count: u32,

    /// Total number of SSA values allocated in this function.
    ///
    /// Serves as the "next available value number" — when creating a new
    /// instruction that produces a result, the builder increments this
    /// counter and assigns the current value as the result.
    pub value_count: u32,

    /// Whether this is a function definition (has a body) or just a
    /// declaration (forward declaration or extern).
    ///
    /// Declaration-only functions have no basic blocks and serve as
    /// placeholders for call target resolution during linking.
    pub is_definition: bool,

    /// Compile-time integer constant map: SSA `Value` → constant integer.
    ///
    /// Populated during IR lowering (Phase 6) by `emit_int_const`.  Each
    /// integer constant materialised as a sentinel `BinOp(Add, V, V, UNDEF)`
    /// records its value here so that the backend can resolve constants
    /// without fragile positional matching against global `.Lconst.i.*`
    /// variables.
    pub constant_values: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, i64>,

    /// Compile-time float constant map: SSA `Value` → (global name, f64).
    ///
    /// Populated during IR lowering (Phase 6) by `emit_float_const`.
    /// Records the global name for RIP-relative SSE loads and the raw
    /// float value for potential constant-folding.
    pub float_constant_values:
        crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, (String, f64)>,

    /// Per-function map from SSA `Value` to function name for function
    /// references (function pointer decay).
    ///
    /// This is scoped to this function — unlike `IrModule::func_ref_map`
    /// which is global and suffers from Value-ID collisions across
    /// different functions (since each function starts numbering from 0).
    /// The backend uses this to distinguish function address loads (LEA)
    /// from normal value moves (MOV) during call argument setup.
    pub func_ref_map: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,

    /// Per-function map from SSA `Value` to global variable name.
    ///
    /// Scoped to this function for the same reason as `func_ref_map`.
    pub global_var_refs: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,

    /// Debug metadata for local variables — used by the DWARF emitter to
    /// generate `DW_TAG_variable` entries with names, types, and locations.
    ///
    /// Populated during Phase 6 (IR lowering) from the `local_vars` map.
    /// Each entry records the variable name, IR type, alloca index, and
    /// source line number.  The DWARF emitter iterates this list to
    /// produce variable DIEs inside each `DW_TAG_subprogram`.
    pub local_var_debug_info: Vec<LocalVarDebugInfo>,
}

// ===========================================================================
// IrFunction — construction
// ===========================================================================

impl IrFunction {
    /// Creates a new IR function with the given name, parameters, and return type.
    ///
    /// The function is initialized with:
    /// - One empty entry block (`blocks[0]`) labeled `"entry"`
    /// - Calling convention set to [`CallingConvention::C`]
    /// - Linkage set to [`Linkage::External`]
    /// - Visibility set to [`Visibility::Default`]
    /// - `is_definition` set to `true` (a function body is expected)
    /// - `value_count` initialized to the number of parameters (each parameter
    ///   occupies one SSA value)
    ///
    /// # Parameters
    ///
    /// - `name`: Function name (symbol name in the output object file).
    /// - `params`: Function parameters with names, types, and SSA values.
    /// - `return_type`: The IR type returned by the function.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bcc::ir::function::{IrFunction, FunctionParam};
    /// use bcc::ir::types::IrType;
    /// use bcc::ir::instructions::Value;
    ///
    /// let params = vec![
    ///     FunctionParam::new("a".to_string(), IrType::I32, Value(0)),
    ///     FunctionParam::new("b".to_string(), IrType::I32, Value(1)),
    /// ];
    /// let func = IrFunction::new("add".to_string(), params, IrType::I32);
    ///
    /// assert_eq!(func.name, "add");
    /// assert_eq!(func.param_count(), 2);
    /// assert_eq!(func.block_count(), 1); // entry block
    /// assert_eq!(func.value_count, 2);   // two parameters
    /// ```
    pub fn new(name: String, params: Vec<FunctionParam>, return_type: IrType) -> Self {
        let value_count = params.len() as u32;

        // Create the entry block (index 0) with a descriptive label.
        let entry = BasicBlock::with_label(0, "entry".to_string());

        IrFunction {
            name,
            params,
            return_type,
            blocks: vec![entry],
            calling_convention: CallingConvention::C,
            is_variadic: false,
            linkage: Linkage::External,
            visibility: Visibility::Default,
            is_noreturn: false,
            section: None,
            alignment: None,
            local_count: 0,
            value_count,
            is_definition: true,
            constant_values: crate::common::fx_hash::FxHashMap::default(),
            float_constant_values: crate::common::fx_hash::FxHashMap::default(),
            func_ref_map: crate::common::fx_hash::FxHashMap::default(),
            global_var_refs: crate::common::fx_hash::FxHashMap::default(),
            local_var_debug_info: Vec::new(),
        }
    }
}

// ===========================================================================
// IrFunction — entry block access
// ===========================================================================

impl IrFunction {
    /// Returns an immutable reference to the entry block (`blocks[0]`).
    ///
    /// The entry block is special — all `alloca` instructions for local
    /// variables are placed here during IR lowering (Phase 6).  The mem2reg
    /// pass (Phase 7) scans this block to identify promotable allocas.
    ///
    /// # Panics
    ///
    /// Panics if the function has no basic blocks.  Every well-formed
    /// `IrFunction` created via [`new`](IrFunction::new) has at least one
    /// block, so this should only occur for manually constructed functions
    /// with an empty block list.
    #[inline]
    pub fn entry_block(&self) -> &BasicBlock {
        &self.blocks[0]
    }

    /// Returns a mutable reference to the entry block (`blocks[0]`).
    ///
    /// Used during IR lowering to append alloca instructions to the entry
    /// block and during optimization passes that need to modify entry
    /// block contents.
    ///
    /// # Panics
    ///
    /// Panics if the function has no basic blocks.
    #[inline]
    pub fn entry_block_mut(&mut self) -> &mut BasicBlock {
        &mut self.blocks[0]
    }
}

// ===========================================================================
// IrFunction — block management
// ===========================================================================

impl IrFunction {
    /// Appends a new basic block to the function and returns its index.
    ///
    /// The block's [`index`](BasicBlock::index) field is updated to match
    /// its position in the function's block list.  The returned index can
    /// be used as a [`BlockId`] for branch targets and phi node references.
    ///
    /// # Parameters
    ///
    /// - `block`: The basic block to append.
    ///
    /// # Returns
    ///
    /// The index of the newly added block within `self.blocks`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut func = IrFunction::new("f".to_string(), vec![], IrType::Void);
    /// let bb1 = BasicBlock::new(0); // index will be corrected by add_block
    /// let idx = func.add_block(bb1);
    /// assert_eq!(idx, 1); // entry block is 0, new block is 1
    /// ```
    pub fn add_block(&mut self, mut block: BasicBlock) -> usize {
        let index = self.blocks.len();
        block.index = index;
        self.blocks.push(block);
        index
    }

    /// Ensures that the blocks vector contains a block for the given `BlockId`.
    ///
    /// If the blocks vector is not large enough, empty `BasicBlock` entries
    /// are pushed until `block_id.0` is a valid index. This is used during
    /// label pre-scanning to pre-allocate blocks for forward references
    /// (e.g., `asm goto` labels that reference labels defined later in the
    /// function body).
    ///
    /// # Parameters
    ///
    /// - `block_id`: The block ID that must become a valid index.
    pub fn ensure_block(&mut self, block_id: crate::ir::instructions::BlockId) {
        let idx = block_id.0 as usize;
        while self.blocks.len() <= idx {
            let new_idx = self.blocks.len();
            self.blocks.push(BasicBlock::new(new_idx));
        }
    }

    /// Returns an immutable reference to a basic block by its index.
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `index`: The block index (position in the `blocks` vector).
    #[inline]
    pub fn get_block(&self, index: usize) -> Option<&BasicBlock> {
        self.blocks.get(index)
    }

    /// Returns a mutable reference to a basic block by its index.
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `index`: The block index (position in the `blocks` vector).
    #[inline]
    pub fn get_block_mut(&mut self, index: usize) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(index)
    }

    /// Returns an immutable slice over all basic blocks in order.
    ///
    /// Block order is significant — `blocks[0]` is the entry block and
    /// code generation emits blocks in this order.
    #[inline]
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    /// Returns a mutable slice over all basic blocks.
    ///
    /// Provides mutable access for optimization passes that need to
    /// rewrite blocks (e.g., CFG simplification, block merging).
    #[inline]
    pub fn blocks_mut(&mut self) -> &mut [BasicBlock] {
        &mut self.blocks
    }

    /// Returns the number of basic blocks in the function.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of function parameters.
    #[inline]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

// ===========================================================================
// IrFunction — CFG navigation
// ===========================================================================

impl IrFunction {
    /// Returns the predecessor block indices for the block at `block_index`.
    ///
    /// Delegates to the [`BasicBlock::predecessors()`] method on the
    /// specified block.  Predecessors are blocks whose terminators can
    /// transfer control to this block.
    ///
    /// # Panics
    ///
    /// Panics if `block_index` is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the block to query.
    #[inline]
    pub fn predecessors(&self, block_index: usize) -> &[usize] {
        self.blocks[block_index].predecessors()
    }

    /// Returns the successor block indices for the block at `block_index`.
    ///
    /// Delegates to the [`BasicBlock::successors()`] method on the
    /// specified block.  Successors are blocks to which this block's
    /// terminator can transfer control.
    ///
    /// # Panics
    ///
    /// Panics if `block_index` is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the block to query.
    #[inline]
    pub fn successors(&self, block_index: usize) -> &[usize] {
        self.blocks[block_index].successors()
    }

    /// Computes a reverse-postorder traversal of the control flow graph.
    ///
    /// Reverse postorder (RPO) is the standard traversal order for
    /// dataflow analysis — it guarantees that every block is visited
    /// after all of its dominators.  This property makes RPO the
    /// canonical iteration order for:
    ///
    /// - **Dominator tree construction** (Lengauer-Tarjan algorithm)
    /// - **SSA construction** (mem2reg reaching-definition propagation)
    /// - **Dataflow analysis** (constant propagation, liveness)
    /// - **Optimization passes** (dead code elimination, value numbering)
    ///
    /// # Algorithm
    ///
    /// 1. Perform a depth-first traversal starting from the entry block.
    /// 2. Append each block to the postorder list *after* visiting all
    ///    its successors.
    /// 3. Reverse the postorder list to get reverse postorder.
    ///
    /// # Returns
    ///
    /// A vector of block indices in reverse-postorder.  The entry block
    /// (`blocks[0]`) is always the first element.  Unreachable blocks
    /// (not reachable from the entry) are excluded.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let rpo = func.reverse_postorder();
    /// assert_eq!(rpo[0], 0); // entry block is always first
    /// ```
    pub fn reverse_postorder(&self) -> Vec<usize> {
        if self.blocks.is_empty() {
            return Vec::new();
        }

        let num_blocks = self.blocks.len();
        let mut visited = vec![false; num_blocks];
        let mut postorder = Vec::with_capacity(num_blocks);

        // Iterative DFS using an explicit stack to avoid deep recursion.
        // Each stack frame is (block_index, successor_iter_position).
        // When all successors of a block have been visited, we push the
        // block onto the postorder list.
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(num_blocks);

        visited[0] = true;
        stack.push((0, 0));

        while let Some((block_idx, succ_pos)) = stack.last_mut() {
            let succs = self.blocks[*block_idx].successors();
            if *succ_pos < succs.len() {
                let next_succ = succs[*succ_pos];
                *succ_pos += 1;

                if next_succ < num_blocks && !visited[next_succ] {
                    visited[next_succ] = true;
                    stack.push((next_succ, 0));
                }
            } else {
                // All successors visited — this block is done.
                let done_block = *block_idx;
                stack.pop();
                postorder.push(done_block);
            }
        }

        // Reverse postorder.
        postorder.reverse();
        postorder
    }
}

// ===========================================================================
// IrFunction — instruction analysis utilities
// ===========================================================================

impl IrFunction {
    /// Computes successor block indices for a block by inspecting its
    /// terminator instruction directly.
    ///
    /// Unlike [`successors`](IrFunction::successors), which reads the
    /// pre-computed `BasicBlock::successors` field, this method derives
    /// successor information from the actual terminator instruction using
    /// [`Instruction::is_terminator`] and [`Instruction::successors`].
    /// This is useful for validating or rebuilding the CFG edge lists
    /// after IR transformations.
    ///
    /// Returns an empty vector if the block has no terminator or if
    /// `block_index` is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `block_index`: Index of the block to analyze.
    pub fn compute_successors_from_terminator(&self, block_index: usize) -> Vec<usize> {
        let Some(block) = self.blocks.get(block_index) else {
            return Vec::new();
        };

        // Find the terminator instruction (last instruction if it is a terminator).
        let Some(last_inst) = block.instructions.last() else {
            return Vec::new();
        };

        if !Instruction::is_terminator(last_inst) {
            return Vec::new();
        }

        // Use Instruction::successors() to get BlockId targets, then
        // convert to usize indices via BlockId::index().
        let targets: Vec<BlockId> = Instruction::successors(last_inst);
        targets.into_iter().map(BlockId::index).collect()
    }

    /// Counts the number of alloca instructions in the entry block.
    ///
    /// Scans the entry block's instructions using [`Instruction::is_alloca`]
    /// to count stack allocation instructions.  This is used during mem2reg
    /// analysis to pre-allocate data structures for alloca promotion and
    /// to update the [`local_count`](IrFunction::local_count) field.
    ///
    /// # Returns
    ///
    /// The number of `Alloca` instructions in the entry block, or 0
    /// if the function has no blocks.
    pub fn count_entry_allocas(&self) -> u32 {
        if self.blocks.is_empty() {
            return 0;
        }
        self.blocks[0]
            .instructions
            .iter()
            .filter(|inst| Instruction::is_alloca(inst))
            .count() as u32
    }

    /// Rebuilds the predecessor and successor edge lists for all blocks
    /// by inspecting each block's terminator instruction.
    ///
    /// This is a repair operation that recomputes the CFG edges from
    /// scratch using [`Instruction::is_terminator`], [`Instruction::successors`],
    /// and [`BlockId::index`].  Call this after IR transformations that
    /// may have invalidated the cached edge lists (e.g., after replacing
    /// terminators, splitting blocks, or eliminating dead blocks).
    ///
    /// After this method returns, every block's `predecessors` and
    /// `successors` fields are consistent with the terminator instructions.
    pub fn rebuild_cfg_edges(&mut self) {
        let num_blocks = self.blocks.len();

        // Clear all existing edges.
        for block in &mut self.blocks {
            block.predecessors.clear();
            block.successors.clear();
        }

        // Recompute from terminators.  We collect successor info first
        // to avoid borrow conflicts.
        let mut edges: Vec<(usize, Vec<usize>)> = Vec::with_capacity(num_blocks);

        for block in &self.blocks {
            let succs = if let Some(last_inst) = block.instructions.last() {
                if Instruction::is_terminator(last_inst) {
                    let targets: Vec<BlockId> = Instruction::successors(last_inst);
                    targets.into_iter().map(BlockId::index).collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            edges.push((block.index, succs));
        }

        // Apply collected edges.
        for (src_idx, succ_indices) in &edges {
            for &dst_idx in succ_indices {
                if dst_idx < num_blocks {
                    self.blocks[*src_idx].successors.push(dst_idx);
                    self.blocks[dst_idx].predecessors.push(*src_idx);
                }
            }
        }
    }

    /// Returns the next available SSA [`Value`] number and increments
    /// the counter.
    ///
    /// This is a convenience method used during IR construction to
    /// allocate unique SSA value numbers for new instructions.
    ///
    /// # Returns
    ///
    /// A fresh [`Value`] that has not been used by any other instruction
    /// in this function.
    #[inline]
    pub fn next_value(&mut self) -> Value {
        let v = Value(self.value_count);
        self.value_count += 1;
        v
    }
}

// ===========================================================================
// IrFunction — Display implementation
// ===========================================================================

impl std::fmt::Display for IrFunction {
    /// Formats the function in a human-readable IR text form.
    ///
    /// Output includes the function signature (linkage, visibility, calling
    /// convention, return type, name, parameters) followed by each basic
    /// block.
    ///
    /// # Example output
    ///
    /// ```text
    /// define external ccc i32 @add(i32 %0, i32 %1) {
    /// bb0 (entry):
    ///   %2 = add i32 %0, %1
    ///   ret i32 %2
    /// }
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Function header: define/declare, linkage, calling convention, return type, name, params
        if self.is_definition {
            write!(f, "define ")?;
        } else {
            write!(f, "declare ")?;
        }

        write!(
            f,
            "{} {} {} @{}(",
            self.linkage, self.calling_convention, self.return_type, self.name
        )?;

        // Parameters
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param)?;
        }

        if self.is_variadic {
            if !self.params.is_empty() {
                write!(f, ", ")?;
            }
            write!(f, "...")?;
        }

        write!(f, ")")?;

        // Attributes
        if self.is_noreturn {
            write!(f, " noreturn")?;
        }
        if let Some(ref sec) = self.section {
            write!(f, " section(\"{}\")", sec)?;
        }
        if let Some(align) = self.alignment {
            write!(f, " align({})", align)?;
        }
        if self.visibility != Visibility::Default {
            write!(f, " visibility({})", self.visibility)?;
        }

        if self.is_definition {
            writeln!(f, " {{")?;
            for block in &self.blocks {
                write!(f, "{}", block)?;
            }
            write!(f, "}}")
        } else {
            Ok(())
        }
    }
}
