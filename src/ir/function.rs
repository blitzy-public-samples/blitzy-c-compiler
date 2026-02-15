//! IR function representation for the BCC compiler.
//!
//! This module defines [`IrFunction`], the primary container for function-level
//! intermediate representation in the BCC compilation pipeline. Each `IrFunction`
//! holds a list of [`BasicBlock`]s forming its control flow graph (CFG), a
//! registry of SSA values, and metadata such as calling convention, linkage,
//! and function attributes.
//!
//! # Alloca-Then-Promote Architecture
//!
//! The entry basic block is designated as the **alloca insertion point**.
//! During Phase 6 (IR lowering), all local variable allocations are placed
//! in the entry block as [`Alloca`](crate::ir::instructions::Instruction::Alloca)
//! instructions. Phase 7 (`mem2reg`) then promotes eligible allocas to SSA
//! virtual registers using dominance frontier computation. This architecture
//! mirrors the LLVM approach to SSA construction and is mandated by the
//! project requirements (Section 0.7.2).
//!
//! # Value Numbering
//!
//! Each function maintains its own SSA value numbering namespace via
//! [`new_value()`](IrFunction::new_value). Values are tracked in the
//! `local_values` registry, which stores the type and optional name for
//! each value. [`ValueId`] handles are only meaningful within the function
//! that allocated them.
//!
//! # Function Attributes
//!
//! GCC function attributes (`__attribute__((...))`) are captured in the
//! [`FunctionAttributes`] struct. These influence code generation decisions:
//!
//! | Attribute                  | Effect                                         |
//! |----------------------------|------------------------------------------------|
//! | `noreturn` / `_Noreturn`   | Omit epilogue, mark in ELF symbol table        |
//! | `noinline`                 | Prevent inlining by optimization passes         |
//! | `always_inline`            | Force inlining when called                      |
//! | `cold` / `hot`             | Branch prediction hints, section placement      |
//! | `visibility`               | ELF symbol visibility (default/hidden/etc.)     |
//! | `weak`                     | Weak ELF symbol binding                         |
//! | `constructor`/`destructor` | Place in `.init_array` / `.fini_array`           |
//!
//! # Supporting Types
//!
//! This module also defines several supporting types used throughout the IR:
//!
//! - [`CallingConvention`] — calling convention enum (C, Fast, Cold, Custom)
//! - [`Linkage`] — symbol linkage (External, Internal, Weak, Common)
//! - [`Visibility`] — ELF symbol visibility (Default, Hidden, Protected, Internal)
//! - [`Parameter`] — function parameter with name, type, and value ID
//! - [`ValueInfo`] — metadata for each SSA value in the function
//!
//! # Module Re-exports
//!
//! [`ValueId`] is defined in [`crate::ir::instructions`] and re-exported
//! from this module for ergonomic access by consumers that work primarily
//! with function-level IR constructs.

use std::fmt;

use crate::ir::basic_block::{BasicBlock, BasicBlockId};
use crate::ir::types::IrType;

/// Re-export [`ValueId`] from this module so that consumers working with
/// function IR can import it from `crate::ir::function` alongside
/// [`IrFunction`], [`Parameter`], and [`ValueInfo`].
pub use crate::ir::instructions::ValueId;

// ---------------------------------------------------------------------------
// CallingConvention — function calling convention
// ---------------------------------------------------------------------------

/// Calling convention for a function.
///
/// The calling convention determines parameter passing rules, register
/// save/restore responsibilities, and stack frame layout. Most C functions
/// use the [`C`](CallingConvention::C) convention, which maps to the
/// platform-specific ABI:
///
/// - **x86-64:** System V AMD64 ABI (RDI, RSI, RDX, RCX, R8, R9)
/// - **i686:** cdecl (all args on stack)
/// - **AArch64:** AAPCS64 (X0–X7)
/// - **RISC-V 64:** LP64D (a0–a7)
///
/// This enum is defined here (in the function module) rather than in
/// `module.rs` to avoid circular import dependencies — functions reference
/// their calling convention directly, and the module aggregates functions.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum CallingConvention {
    /// Standard C calling convention — platform ABI default.
    C,

    /// Fast calling convention — hints that the function is a leaf or
    /// internal helper. The backend may use additional caller-saved
    /// registers for parameter passing to reduce spill pressure.
    Fast,

    /// Cold calling convention — marks the function as rarely executed.
    /// The backend may optimise the caller's code path at the expense
    /// of this function's prologue/epilogue overhead.
    Cold,

    /// Custom calling convention — reserved for future extension or
    /// architecture-specific conventions not covered by the above.
    Custom,
}

impl Default for CallingConvention {
    /// The default calling convention is [`C`](CallingConvention::C),
    /// matching the platform ABI for standard C function calls.
    #[inline]
    fn default() -> Self {
        CallingConvention::C
    }
}

impl fmt::Display for CallingConvention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallingConvention::C => f.write_str("ccc"),
            CallingConvention::Fast => f.write_str("fastcc"),
            CallingConvention::Cold => f.write_str("coldcc"),
            CallingConvention::Custom => f.write_str("customcc"),
        }
    }
}

// ---------------------------------------------------------------------------
// Linkage — symbol linkage type
// ---------------------------------------------------------------------------

/// Symbol linkage type for functions and global variables.
///
/// Linkage determines the visibility and binding of a symbol in the ELF
/// object file and how the linker resolves references to it across
/// translation units.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Linkage {
    /// External linkage — the symbol is visible across translation units.
    /// Corresponds to `STB_GLOBAL` in ELF.
    External,

    /// Internal linkage — the symbol is file-local (`static` in C).
    /// Corresponds to `STB_LOCAL` in ELF.
    Internal,

    /// Weak linkage — the symbol can be overridden by a strong definition.
    /// Corresponds to `STB_WEAK` in ELF.
    Weak,

    /// Common linkage — used for tentative definitions of uninitialised
    /// global variables. The linker merges multiple common symbols,
    /// choosing the largest size.
    Common,
}

impl Default for Linkage {
    /// The default linkage is [`External`](Linkage::External), matching
    /// the default visibility of non-`static` C functions and variables.
    #[inline]
    fn default() -> Self {
        Linkage::External
    }
}

impl fmt::Display for Linkage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Linkage::External => f.write_str("external"),
            Linkage::Internal => f.write_str("internal"),
            Linkage::Weak => f.write_str("weak"),
            Linkage::Common => f.write_str("common"),
        }
    }
}

// ---------------------------------------------------------------------------
// Visibility — ELF symbol visibility
// ---------------------------------------------------------------------------

/// ELF symbol visibility for functions and global variables.
///
/// Visibility controls how the dynamic linker handles the symbol during
/// shared library loading. It is orthogonal to linkage — a symbol can be
/// externally linked but hidden from the dynamic symbol table.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Visibility {
    /// Default visibility — the symbol is exported in the dynamic symbol
    /// table and can be preempted by another shared library.
    /// Corresponds to `STV_DEFAULT`.
    Default,

    /// Hidden visibility — the symbol is not exported in the dynamic
    /// symbol table. References within the same shared object use direct
    /// PC-relative addressing (no GOT/PLT indirection).
    /// Corresponds to `STV_HIDDEN`.
    Hidden,

    /// Protected visibility — the symbol is exported but cannot be
    /// preempted. The defining shared object always resolves its own
    /// references to this symbol.
    /// Corresponds to `STV_PROTECTED`.
    Protected,

    /// Internal visibility — like hidden, but additionally the compiler
    /// may assume the symbol is never referenced from another module.
    /// Corresponds to `STV_INTERNAL`.
    Internal,
}

impl Default for Visibility {
    /// The default visibility is [`Default`](Visibility::Default),
    /// matching the standard ELF symbol export behaviour.
    #[inline]
    fn default() -> Self {
        Visibility::Default
    }
}

impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Visibility::Default => f.write_str("default"),
            Visibility::Hidden => f.write_str("hidden"),
            Visibility::Protected => f.write_str("protected"),
            Visibility::Internal => f.write_str("internal"),
        }
    }
}

// ---------------------------------------------------------------------------
// FunctionAttributes — GCC function attributes
// ---------------------------------------------------------------------------

/// Collection of GCC function attributes that influence code generation
/// and linker behaviour.
///
/// Each flag corresponds to a `__attribute__((...))` annotation on the
/// function declaration or definition. The semantic analyser
/// (`crate::frontend::sema::attribute_handler`) populates these fields
/// during AST traversal.
///
/// # Mutually Exclusive Attributes
///
/// - `is_noinline` and `is_always_inline` are semantically contradictory.
///   If both are set, `is_always_inline` takes precedence (matching GCC
///   behaviour).
/// - `is_cold` and `is_hot` are contradictory; the last-specified wins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionAttributes {
    /// `__attribute__((noreturn))` or `_Noreturn` — the function never
    /// returns to its caller (e.g., `exit()`, `abort()`, infinite loops).
    pub is_noreturn: bool,

    /// `__attribute__((noinline))` — prevent the optimiser from inlining
    /// this function at call sites.
    pub is_noinline: bool,

    /// `__attribute__((always_inline))` — force inlining at every call
    /// site, regardless of optimiser cost model.
    pub is_always_inline: bool,

    /// `__attribute__((cold))` — the function is rarely executed. The
    /// backend may place it in a `.text.cold` section and bias branch
    /// prediction away from calls to it.
    pub is_cold: bool,

    /// `__attribute__((hot))` — the function is frequently executed. The
    /// backend may place it in a `.text.hot` section and optimise its
    /// code layout for instruction cache locality.
    pub is_hot: bool,

    /// Symbol visibility in the ELF dynamic symbol table.
    ///
    /// Set by `__attribute__((visibility("...")))` where the argument is
    /// one of `"default"`, `"hidden"`, `"protected"`, or `"internal"`.
    pub visibility: Visibility,

    /// `__attribute__((weak))` — weak ELF symbol binding. The linker
    /// resolves undefined references to a weak symbol only if no strong
    /// definition is available.
    pub is_weak: bool,

    /// `__attribute__((constructor))` — the function is called during
    /// program initialisation, before `main()`. Placed in `.init_array`.
    pub is_constructor: bool,

    /// `__attribute__((destructor))` — the function is called during
    /// program termination, after `main()` returns or `exit()` is called.
    /// Placed in `.fini_array`.
    pub is_destructor: bool,

    /// Priority for constructor ordering (lower = earlier). `None` means
    /// default priority (65535). Valid range: 0–65535.
    pub constructor_priority: Option<u32>,

    /// Priority for destructor ordering (lower = earlier). `None` means
    /// default priority (65535). Valid range: 0–65535.
    pub destructor_priority: Option<u32>,
}

impl Default for FunctionAttributes {
    /// Returns function attributes with all flags disabled and default
    /// visibility — the baseline for an ordinary C function with no
    /// `__attribute__` annotations.
    fn default() -> Self {
        FunctionAttributes {
            is_noreturn: false,
            is_noinline: false,
            is_always_inline: false,
            is_cold: false,
            is_hot: false,
            visibility: Visibility::Default,
            is_weak: false,
            is_constructor: false,
            is_destructor: false,
            constructor_priority: None,
            destructor_priority: None,
        }
    }
}

impl fmt::Display for FunctionAttributes {
    /// Renders non-default attributes in a compact annotation format.
    ///
    /// Only attributes that differ from their default value are shown.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let mut attr = |s: &str, out: &mut fmt::Formatter<'_>| -> fmt::Result {
            if !first {
                out.write_str(" ")?;
            }
            first = false;
            out.write_str(s)
        };

        if self.is_noreturn {
            attr("noreturn", f)?;
        }
        if self.is_noinline {
            attr("noinline", f)?;
        }
        if self.is_always_inline {
            attr("alwaysinline", f)?;
        }
        if self.is_cold {
            attr("cold", f)?;
        }
        if self.is_hot {
            attr("hot", f)?;
        }
        if self.visibility != Visibility::Default {
            attr(&format!("visibility({})", self.visibility), f)?;
        }
        if self.is_weak {
            attr("weak", f)?;
        }
        if self.is_constructor {
            match self.constructor_priority {
                Some(p) => attr(&format!("constructor({})", p), f)?,
                None => attr("constructor", f)?,
            }
        }
        if self.is_destructor {
            match self.destructor_priority {
                Some(p) => attr(&format!("destructor({})", p), f)?,
                None => attr("destructor", f)?,
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parameter — function parameter
// ---------------------------------------------------------------------------

/// A formal parameter of an IR function.
///
/// Each parameter has an SSA [`ValueId`] assigned during IR lowering,
/// an [`IrType`] representing its machine-level type, and an optional
/// name for debug information and IR dumps.
///
/// # Examples
///
/// ```ignore
/// let param = Parameter {
///     name: Some("argc".into()),
///     ty: IrType::I32,
///     id: ValueId(0),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Optional parameter name from the source code. `None` for unnamed
    /// parameters (e.g., `void f(int, int)`).
    pub name: Option<String>,

    /// The IR type of this parameter, determined by the C-type-to-IR-type
    /// conversion during lowering.
    pub ty: IrType,

    /// SSA value ID for this parameter within the function. Used to
    /// reference the parameter value in instructions.
    pub id: ValueId,
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.ty, self.id)?;
        if let Some(ref name) = self.name {
            write!(f, " ; {}", name)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ValueInfo — SSA value metadata
// ---------------------------------------------------------------------------

/// Metadata for an SSA value defined within a function.
///
/// The function's `local_values` registry maps each [`ValueId`] to a
/// `ValueInfo` record, enabling type lookups and debug name resolution
/// without traversing the instruction list.
///
/// # Examples
///
/// ```ignore
/// let info = ValueInfo {
///     id: ValueId(5),
///     ty: IrType::I32,
///     name: Some("x".into()),
/// };
/// assert_eq!(info.id, ValueId(5));
/// ```
#[derive(Clone, Debug)]
pub struct ValueInfo {
    /// The unique SSA value identifier.
    pub id: ValueId,

    /// The IR type of this value.
    pub ty: IrType,

    /// Optional human-readable name for debug output and IR dumps.
    /// Derived from the source-level variable name during lowering.
    pub name: Option<String>,
}

impl fmt::Display for ValueInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.ty)?;
        if let Some(ref name) = self.name {
            write!(f, " ({})", name)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IrFunction — function-level IR container
// ---------------------------------------------------------------------------

/// Function-level intermediate representation container.
///
/// `IrFunction` is the central data structure for representing a single
/// C function in the BCC IR. It owns:
///
/// - An ordered list of [`BasicBlock`]s forming the control flow graph
/// - A registry of all SSA values ([`ValueInfo`]) with monotonic numbering
/// - Function signature (return type, parameters, variadic flag)
/// - Metadata (linkage, calling convention, attributes, section, alignment)
///
/// # Entry Block and Alloca Insertion
///
/// The entry block (at index 0 by convention, tracked by `entry_block_id`)
/// is the designated alloca insertion point. During Phase 6 lowering, all
/// local variable `Alloca` instructions are placed at the beginning of
/// this block before any other instructions. This ensures that:
///
/// 1. All allocas dominate their uses (since the entry block dominates
///    all other blocks).
/// 2. The `mem2reg` pass (Phase 7) can efficiently identify and promote
///    eligible allocas by scanning only the entry block.
///
/// # Value Numbering
///
/// SSA values are numbered sequentially starting from 0. Parameter values
/// receive the first IDs, followed by instruction results. The
/// [`new_value()`](IrFunction::new_value) method allocates the next ID
/// and registers it in `local_values`.
///
/// # Lifecycle
///
/// ```text
/// IrFunction::new(...)
///   │
///   ├── Phase 6: IR lowering adds blocks, instructions, allocas
///   │
///   ├── Phase 7: mem2reg promotes allocas, inserts phi nodes
///   │
///   ├── Phase 8: Optimisation passes (constant folding, DCE, CFG simp)
///   │
///   ├── Phase 9: Phi elimination converts phi nodes to copies
///   │
///   └── Phase 10: Code generation lowers to machine instructions
/// ```
#[derive(Clone, Debug)]
pub struct IrFunction {
    /// Function name (symbol name in the object file).
    pub name: String,

    /// Return type of the function.
    pub return_type: IrType,

    /// Formal parameters with their types and SSA value IDs.
    pub params: Vec<Parameter>,

    /// Ordered list of basic blocks forming the control flow graph.
    ///
    /// The entry block is always at index 0 after construction (tracked
    /// by `entry_block_id`). Block ordering is significant for code
    /// layout — the backend emits machine code in this order.
    pub basic_blocks: Vec<BasicBlock>,

    /// ID of the entry basic block — the alloca insertion point.
    ///
    /// This is always `BasicBlockId(0)` immediately after construction,
    /// but may change if blocks are reorganised by optimisation passes.
    pub entry_block_id: BasicBlockId,

    /// Calling convention for this function.
    pub calling_convention: CallingConvention,

    /// Symbol linkage type (external, internal, weak, common).
    pub linkage: Linkage,

    /// `true` if the function accepts variadic arguments (`...`).
    pub is_variadic: bool,

    /// GCC function attributes collected during semantic analysis.
    pub attributes: FunctionAttributes,

    /// Registry of all SSA values defined in this function.
    ///
    /// Indexed by `ValueId::index()` — the i-th entry corresponds to
    /// `ValueId(i)`. Contains type and optional name metadata.
    pub local_values: Vec<ValueInfo>,

    /// Next value ID to allocate. Monotonically increasing.
    pub next_value_id: u32,

    /// Function alignment in bytes for section placement.
    ///
    /// Typically 16 for x86-64 functions, 4 for AArch64/RISC-V.
    /// May be overridden by `__attribute__((aligned(N)))`.
    pub alignment: u32,

    /// Custom section name from `__attribute__((section("...")))`.
    ///
    /// When `Some`, the function is placed in the named section instead
    /// of the default `.text` section. Used by the Linux kernel for
    /// `__init`, `__exit`, and other special sections.
    pub section: Option<String>,

    /// `true` if this function has a body (definition), `false` for
    /// forward declarations (extern prototypes).
    pub is_definition: bool,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl IrFunction {
    /// Creates a new IR function with the given name, return type, and
    /// parameters.
    ///
    /// The function is initialised with:
    /// - An empty entry basic block (named `"entry"`, ID 0)
    /// - Parameters registered as the first SSA values
    /// - Default calling convention ([`CallingConvention::C`])
    /// - External linkage ([`Linkage::External`])
    /// - Default function attributes (all flags disabled)
    /// - 16-byte alignment (suitable for x86-64)
    /// - Marked as a definition (`is_definition = true`)
    ///
    /// # Arguments
    ///
    /// * `name` — function symbol name
    /// * `return_type` — the function's return type
    /// * `params` — formal parameters (their value IDs are preserved as-is)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let params = vec![
    ///     Parameter { name: Some("argc".into()), ty: IrType::I32, id: ValueId(0) },
    ///     Parameter { name: Some("argv".into()), ty: IrType::Ptr, id: ValueId(1) },
    /// ];
    /// let func = IrFunction::new("main".into(), IrType::I32, params);
    /// assert_eq!(func.block_count(), 1); // entry block created automatically
    /// assert_eq!(func.name, "main");
    /// ```
    pub fn new(name: String, return_type: IrType, params: Vec<Parameter>) -> Self {
        // Create the entry basic block — the alloca insertion point.
        let entry = BasicBlock::new(BasicBlockId(0), Some("entry".into()));
        let entry_block_id = entry.id;

        // Register each parameter as an SSA value in the local_values registry.
        let mut local_values = Vec::with_capacity(params.len());
        let mut next_value_id: u32 = 0;

        for param in &params {
            local_values.push(ValueInfo {
                id: param.id,
                ty: param.ty.clone(),
                name: param.name.clone(),
            });
            // Ensure next_value_id is always greater than any parameter ID
            // to avoid ID collisions when new values are created later.
            if param.id.0 >= next_value_id {
                next_value_id = param.id.0 + 1;
            }
        }

        IrFunction {
            name,
            return_type,
            params,
            basic_blocks: vec![entry],
            entry_block_id,
            calling_convention: CallingConvention::C,
            linkage: Linkage::External,
            is_variadic: false,
            attributes: FunctionAttributes::default(),
            local_values,
            next_value_id,
            alignment: 16,
            section: None,
            is_definition: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Basic block management
// ---------------------------------------------------------------------------

impl IrFunction {
    /// Adds a basic block to the function's CFG and returns its
    /// [`BasicBlockId`].
    ///
    /// The block is appended to the end of the `basic_blocks` vector.
    /// Its existing `id` field is preserved; callers are responsible for
    /// assigning unique IDs before calling this method.
    ///
    /// # Arguments
    ///
    /// * `block` — the basic block to add.
    ///
    /// # Returns
    ///
    /// The [`BasicBlockId`] of the newly added block.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut func = IrFunction::new("foo".into(), IrType::Void, vec![]);
    /// let bb1 = BasicBlock::new(BasicBlockId(1), Some("then".into()));
    /// let id = func.add_basic_block(bb1);
    /// assert_eq!(id, BasicBlockId(1));
    /// assert_eq!(func.block_count(), 2); // entry + bb1
    /// ```
    pub fn add_basic_block(&mut self, block: BasicBlock) -> BasicBlockId {
        let id = block.id;
        self.basic_blocks.push(block);
        id
    }

    /// Returns an immutable reference to the basic block with the given ID.
    ///
    /// Block lookup is performed by scanning the block list and matching
    /// on the `id` field. For functions with a small number of blocks
    /// (the common case), this linear scan is efficient.
    ///
    /// # Panics
    ///
    /// Panics if no block with the given ID exists in the function.
    pub fn get_block(&self, id: BasicBlockId) -> &BasicBlock {
        self.basic_blocks
            .iter()
            .find(|bb| bb.id == id)
            .unwrap_or_else(|| panic!("BasicBlock {:?} not found in function '{}'", id, self.name))
    }

    /// Returns an immutable reference to the basic block with the given ID,
    /// or `None` if no such block exists.
    ///
    /// This is the non-panicking variant of [`get_block`] — useful for
    /// defensive code in optimization passes where stale block references
    /// may exist in successor/predecessor lists.
    pub fn try_get_block(&self, id: BasicBlockId) -> Option<&BasicBlock> {
        self.basic_blocks.iter().find(|bb| bb.id == id)
    }

    /// Returns `true` if a basic block with the given ID exists in this
    /// function.
    pub fn has_block(&self, id: BasicBlockId) -> bool {
        self.basic_blocks.iter().any(|bb| bb.id == id)
    }

    /// Returns a mutable reference to the basic block with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if no block with the given ID exists in the function.
    pub fn get_block_mut(&mut self, id: BasicBlockId) -> &mut BasicBlock {
        let func_name = self.name.clone();
        self.basic_blocks
            .iter_mut()
            .find(|bb| bb.id == id)
            .unwrap_or_else(|| panic!("BasicBlock {:?} not found in function '{}'", id, func_name))
    }

    /// Returns an immutable reference to the entry basic block.
    ///
    /// The entry block is the alloca insertion point — all local variable
    /// `Alloca` instructions reside here during Phase 6 lowering. The
    /// entry block dominates every other reachable block in the function's
    /// CFG, which is why allocas placed there are visible to all uses.
    ///
    /// # Panics
    ///
    /// Panics if the entry block ID does not correspond to a valid block
    /// (should never happen in a well-formed function).
    #[inline]
    pub fn entry_block(&self) -> &BasicBlock {
        self.get_block(self.entry_block_id)
    }

    /// Returns an immutable slice of all basic blocks in program order.
    ///
    /// The first element is the entry block. Subsequent blocks follow the
    /// order in which they were added via [`add_basic_block()`].
    #[inline]
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.basic_blocks
    }

    /// Returns a mutable slice of all basic blocks.
    ///
    /// **Callers must preserve the entry block invariant** — the block
    /// referenced by `entry_block_id` must remain in the vector after
    /// any modifications. Removing or reordering the entry block without
    /// updating `entry_block_id` leads to panics in [`entry_block()`].
    #[inline]
    pub fn blocks_mut(&mut self) -> &mut [BasicBlock] {
        &mut self.basic_blocks
    }

    /// Returns the number of basic blocks in the function.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.basic_blocks.len()
    }

    /// Removes the basic block with the given ID from the function.
    ///
    /// **Warning:** This does not update CFG edges in other blocks.
    /// Callers are responsible for removing predecessor/successor
    /// references to the deleted block from all remaining blocks.
    ///
    /// # Panics
    ///
    /// - Panics if `id` equals `entry_block_id` — the entry block
    ///   cannot be removed because it is the alloca insertion point.
    /// - Panics if no block with the given ID exists.
    pub fn remove_block(&mut self, id: BasicBlockId) {
        assert_ne!(
            id, self.entry_block_id,
            "Cannot remove the entry block (id={:?}) from function '{}'",
            id, self.name,
        );
        let pos = self
            .basic_blocks
            .iter()
            .position(|bb| bb.id == id)
            .unwrap_or_else(|| {
                panic!(
                    "BasicBlock {:?} not found in function '{}' for removal",
                    id, self.name
                )
            });
        self.basic_blocks.remove(pos);
    }
}

// ---------------------------------------------------------------------------
// SSA value management
// ---------------------------------------------------------------------------

impl IrFunction {
    /// Allocates a new SSA value ID and registers it in the function's
    /// value registry.
    ///
    /// The value is assigned the next sequential ID and recorded with
    /// its type and optional debug name.
    ///
    /// # Arguments
    ///
    /// * `ty`   — the IR type of the new value.
    /// * `name` — optional human-readable name for debug output.
    ///
    /// # Returns
    ///
    /// The newly allocated [`ValueId`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut func = IrFunction::new("f".into(), IrType::Void, vec![]);
    /// let v0 = func.new_value(IrType::I32, Some("x".into()));
    /// let v1 = func.new_value(IrType::Ptr, None);
    /// assert_eq!(v0, ValueId(0));
    /// assert_eq!(v1, ValueId(1));
    /// ```
    pub fn new_value(&mut self, ty: IrType, name: Option<String>) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        self.local_values.push(ValueInfo { id, ty, name });
        id
    }

    /// Returns the IR type of the SSA value with the given ID.
    ///
    /// Uses an optimistic fast-path: if IDs were allocated sequentially
    /// starting from 0, the value resides at index `id.0` in the
    /// `local_values` vector. Falls back to a linear scan for
    /// non-contiguous ID spaces (e.g., after block removal and
    /// value renumbering).
    ///
    /// # Panics
    ///
    /// Panics if no value with the given ID exists in this function's
    /// value registry.
    pub fn get_value_type(&self, id: ValueId) -> &IrType {
        // Optimistic fast path: sequential allocation places ValueId(n) at
        // index n.  This is O(1) for the common case.
        let idx = id.index() as usize;
        if idx < self.local_values.len() && self.local_values[idx].id == id {
            return &self.local_values[idx].ty;
        }
        // Fallback: linear search handles non-contiguous ID spaces that may
        // arise after value renumbering during optimisation passes.
        self.local_values
            .iter()
            .find(|vi| vi.id == id)
            .map(|vi| &vi.ty)
            .unwrap_or_else(|| {
                panic!(
                    "ValueId {:?} not found in function '{}' value registry",
                    id, self.name,
                )
            })
    }

    /// Returns the [`ValueInfo`] for the SSA value with the given ID,
    /// or `None` if the ID is not registered.
    ///
    /// This is the non-panicking alternative to [`get_value_type()`].
    pub fn get_value_info(&self, id: ValueId) -> Option<&ValueInfo> {
        let idx = id.index() as usize;
        if idx < self.local_values.len() && self.local_values[idx].id == id {
            return Some(&self.local_values[idx]);
        }
        self.local_values.iter().find(|vi| vi.id == id)
    }

    /// Returns the total number of SSA values registered in this function,
    /// including parameter values.
    #[inline]
    pub fn value_count(&self) -> usize {
        self.local_values.len()
    }
}

// ---------------------------------------------------------------------------
// Convenience query methods
// ---------------------------------------------------------------------------

impl IrFunction {
    /// Returns `true` if the function's return type is [`IrType::Void`].
    #[inline]
    pub fn returns_void(&self) -> bool {
        self.return_type.is_void()
    }

    /// Returns `true` if the function has weak linkage or the weak
    /// attribute set in its [`FunctionAttributes`].
    #[inline]
    pub fn is_weak(&self) -> bool {
        self.linkage == Linkage::Weak || self.attributes.is_weak
    }

    /// Returns `true` if the function has internal (file-local) linkage.
    #[inline]
    pub fn is_internal(&self) -> bool {
        self.linkage == Linkage::Internal
    }

    /// Returns `true` if the function has the `noreturn` attribute.
    #[inline]
    pub fn is_noreturn(&self) -> bool {
        self.attributes.is_noreturn
    }

    /// Creates a new basic block with a unique ID, adds it to the
    /// function, and returns its [`BasicBlockId`].
    ///
    /// The block ID is derived from the current block count, ensuring
    /// uniqueness for sequential allocation patterns. For non-sequential
    /// ID schemes, use [`add_basic_block()`] directly.
    ///
    /// # Arguments
    ///
    /// * `name` — optional label for the new block.
    pub fn create_block(&mut self, name: Option<String>) -> BasicBlockId {
        let id = BasicBlockId(self.basic_blocks.len() as u32);
        let block = BasicBlock::new(id, name);
        self.add_basic_block(block)
    }
}

// ---------------------------------------------------------------------------
// Display implementation
// ---------------------------------------------------------------------------

impl fmt::Display for IrFunction {
    /// Renders the function in a human-readable IR textual form.
    ///
    /// # Output Format
    ///
    /// ```text
    /// define external ccc i32 @main(i32 %v0, ptr %v1) {
    /// entry:
    ///     %v2 = alloca i32, align 4
    ///     ...
    ///     ret i32 %v3
    /// }
    /// ```
    ///
    /// Declarations (no body) render as:
    ///
    /// ```text
    /// declare external ccc i32 @printf(ptr %v0, ...)
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Keyword: define (has body) or declare (prototype only).
        if self.is_definition {
            f.write_str("define")?;
        } else {
            f.write_str("declare")?;
        }

        // Linkage and calling convention.
        write!(f, " {} {}", self.linkage, self.calling_convention)?;

        // Return type and function name.
        write!(f, " {} @{}", self.return_type, self.name)?;

        // Parameter list.
        f.write_str("(")?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{} {}", param.ty, param.id)?;
        }
        if self.is_variadic {
            if !self.params.is_empty() {
                f.write_str(", ")?;
            }
            f.write_str("...")?;
        }
        f.write_str(")")?;

        // Function-level attributes on the declaration line.
        if self.attributes.is_noreturn {
            f.write_str(" noreturn")?;
        }
        if self.attributes.is_noinline {
            f.write_str(" noinline")?;
        }
        if self.attributes.is_always_inline {
            f.write_str(" alwaysinline")?;
        }
        if self.attributes.is_cold {
            f.write_str(" cold")?;
        }
        if self.attributes.is_hot {
            f.write_str(" hot")?;
        }

        // Section annotation.
        if let Some(ref sec) = self.section {
            write!(f, " section \"{}\"", sec)?;
        }

        // Alignment annotation.
        if self.alignment != 16 {
            write!(f, " align {}", self.alignment)?;
        }

        // Function body (only for definitions).
        if self.is_definition {
            writeln!(f, " {{")?;
            for bb in &self.basic_blocks {
                write!(f, "{}", bb)?;
            }
            f.write_str("}")?;
        }

        Ok(())
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::instructions::{BasicBlockId, Instruction, ValueId};
    use crate::ir::types::IrType;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Creates a simple `Return` terminator instruction.
    fn ret(value: Option<ValueId>) -> Instruction {
        Instruction::Return { value }
    }

    /// Creates a simple `Branch` terminator to `target`.
    fn branch(target: BasicBlockId) -> Instruction {
        Instruction::Branch { target }
    }

    /// Creates an `Alloca` instruction for an i32 value.
    fn alloca(result: u32) -> Instruction {
        Instruction::Alloca {
            result: ValueId(result),
            ty: IrType::I32,
            alignment: 4,
        }
    }

    /// Creates a `Phi` instruction with given incoming edges.
    fn phi(result: u32, incoming: Vec<(u32, u32)>) -> Instruction {
        Instruction::Phi {
            result: ValueId(result),
            ty: IrType::I32,
            incoming: incoming
                .into_iter()
                .map(|(v, b)| (ValueId(v), BasicBlockId(b)))
                .collect(),
        }
    }

    /// Builds a simple function with the given name and no parameters.
    fn make_simple_func(name: &str) -> IrFunction {
        IrFunction::new(name.into(), IrType::I32, vec![])
    }

    /// Builds a function with two integer parameters.
    fn make_two_param_func() -> IrFunction {
        let params = vec![
            Parameter {
                name: Some("a".into()),
                ty: IrType::I32,
                id: ValueId(0),
            },
            Parameter {
                name: Some("b".into()),
                ty: IrType::I32,
                id: ValueId(1),
            },
        ];
        IrFunction::new("add".into(), IrType::I32, params)
    }

    // -----------------------------------------------------------------------
    // CallingConvention tests
    // -----------------------------------------------------------------------

    #[test]
    fn calling_convention_default_is_c() {
        assert_eq!(CallingConvention::default(), CallingConvention::C);
    }

    #[test]
    fn calling_convention_display() {
        assert_eq!(format!("{}", CallingConvention::C), "ccc");
        assert_eq!(format!("{}", CallingConvention::Fast), "fastcc");
        assert_eq!(format!("{}", CallingConvention::Cold), "coldcc");
        assert_eq!(format!("{}", CallingConvention::Custom), "customcc");
    }

    #[test]
    fn calling_convention_equality() {
        assert_eq!(CallingConvention::C, CallingConvention::C);
        assert_ne!(CallingConvention::C, CallingConvention::Fast);
    }

    // -----------------------------------------------------------------------
    // Linkage tests
    // -----------------------------------------------------------------------

    #[test]
    fn linkage_default_is_external() {
        assert_eq!(Linkage::default(), Linkage::External);
    }

    #[test]
    fn linkage_display() {
        assert_eq!(format!("{}", Linkage::External), "external");
        assert_eq!(format!("{}", Linkage::Internal), "internal");
        assert_eq!(format!("{}", Linkage::Weak), "weak");
        assert_eq!(format!("{}", Linkage::Common), "common");
    }

    #[test]
    fn linkage_copy_and_clone() {
        let l = Linkage::Weak;
        let l2 = l; // Copy
        #[allow(clippy::clone_on_copy)]
        let l3 = l.clone(); // Clone (verified Copy + Clone derivation)
        assert_eq!(l, l2);
        assert_eq!(l, l3);
    }

    // -----------------------------------------------------------------------
    // Visibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn visibility_default_is_default() {
        assert_eq!(Visibility::default(), Visibility::Default);
    }

    #[test]
    fn visibility_display() {
        assert_eq!(format!("{}", Visibility::Default), "default");
        assert_eq!(format!("{}", Visibility::Hidden), "hidden");
        assert_eq!(format!("{}", Visibility::Protected), "protected");
        assert_eq!(format!("{}", Visibility::Internal), "internal");
    }

    #[test]
    fn visibility_all_variants_distinct() {
        let variants = [
            Visibility::Default,
            Visibility::Hidden,
            Visibility::Protected,
            Visibility::Internal,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // FunctionAttributes tests
    // -----------------------------------------------------------------------

    #[test]
    fn function_attributes_default_all_false() {
        let attrs = FunctionAttributes::default();
        assert!(!attrs.is_noreturn);
        assert!(!attrs.is_noinline);
        assert!(!attrs.is_always_inline);
        assert!(!attrs.is_cold);
        assert!(!attrs.is_hot);
        assert_eq!(attrs.visibility, Visibility::Default);
        assert!(!attrs.is_weak);
        assert!(!attrs.is_constructor);
        assert!(!attrs.is_destructor);
        assert!(attrs.constructor_priority.is_none());
        assert!(attrs.destructor_priority.is_none());
    }

    #[test]
    fn function_attributes_display_default_is_empty() {
        let attrs = FunctionAttributes::default();
        assert_eq!(format!("{}", attrs), "");
    }

    #[test]
    fn function_attributes_display_noreturn() {
        let attrs = FunctionAttributes {
            is_noreturn: true,
            ..Default::default()
        };
        assert_eq!(format!("{}", attrs), "noreturn");
    }

    #[test]
    fn function_attributes_display_multiple() {
        let attrs = FunctionAttributes {
            is_noreturn: true,
            is_cold: true,
            is_weak: true,
            ..Default::default()
        };
        let display = format!("{}", attrs);
        assert!(display.contains("noreturn"));
        assert!(display.contains("cold"));
        assert!(display.contains("weak"));
    }

    #[test]
    fn function_attributes_display_constructor_with_priority() {
        let attrs = FunctionAttributes {
            is_constructor: true,
            constructor_priority: Some(100),
            ..Default::default()
        };
        let display = format!("{}", attrs);
        assert!(display.contains("constructor(100)"));
    }

    #[test]
    fn function_attributes_display_destructor_without_priority() {
        let attrs = FunctionAttributes {
            is_destructor: true,
            ..Default::default()
        };
        let display = format!("{}", attrs);
        assert!(display.contains("destructor"));
        assert!(!display.contains("destructor("));
    }

    #[test]
    fn function_attributes_display_visibility_hidden() {
        let attrs = FunctionAttributes {
            visibility: Visibility::Hidden,
            ..Default::default()
        };
        let display = format!("{}", attrs);
        assert!(display.contains("visibility(hidden)"));
    }

    #[test]
    fn function_attributes_equality() {
        let a = FunctionAttributes::default();
        let b = FunctionAttributes::default();
        assert_eq!(a, b);

        let c = FunctionAttributes {
            is_noreturn: true,
            ..Default::default()
        };
        assert_ne!(a, c);
    }

    // -----------------------------------------------------------------------
    // Parameter tests
    // -----------------------------------------------------------------------

    #[test]
    fn parameter_with_name() {
        let param = Parameter {
            name: Some("argc".into()),
            ty: IrType::I32,
            id: ValueId(0),
        };
        assert_eq!(param.name.as_deref(), Some("argc"));
        assert_eq!(param.id, ValueId(0));
    }

    #[test]
    fn parameter_without_name() {
        let param = Parameter {
            name: None,
            ty: IrType::Ptr,
            id: ValueId(1),
        };
        assert!(param.name.is_none());
        assert_eq!(param.id, ValueId(1));
    }

    #[test]
    fn parameter_display() {
        let param = Parameter {
            name: Some("x".into()),
            ty: IrType::I32,
            id: ValueId(0),
        };
        let display = format!("{}", param);
        assert!(display.contains("i32"));
        assert!(display.contains("%v0"));
    }

    // -----------------------------------------------------------------------
    // ValueInfo tests
    // -----------------------------------------------------------------------

    #[test]
    fn value_info_construction() {
        let info = ValueInfo {
            id: ValueId(5),
            ty: IrType::I64,
            name: Some("counter".into()),
        };
        assert_eq!(info.id, ValueId(5));
        assert_eq!(info.name.as_deref(), Some("counter"));
    }

    #[test]
    fn value_info_display() {
        let info = ValueInfo {
            id: ValueId(3),
            ty: IrType::F64,
            name: Some("temp".into()),
        };
        let display = format!("{}", info);
        assert!(display.contains("%v3"));
        assert!(display.contains("f64"));
        assert!(display.contains("temp"));
    }

    #[test]
    fn value_info_display_no_name() {
        let info = ValueInfo {
            id: ValueId(7),
            ty: IrType::I8,
            name: None,
        };
        let display = format!("{}", info);
        assert!(display.contains("%v7"));
        assert!(display.contains("i8"));
        assert!(!display.contains("("));
    }

    // -----------------------------------------------------------------------
    // IrFunction::new tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_function_has_entry_block() {
        let func = make_simple_func("test");
        assert_eq!(func.block_count(), 1);
        assert_eq!(func.entry_block_id, BasicBlockId(0));
    }

    #[test]
    fn new_function_entry_block_named_entry() {
        let func = make_simple_func("test");
        let entry = func.entry_block();
        assert_eq!(entry.name.as_deref(), Some("entry"));
        assert_eq!(entry.id, BasicBlockId(0));
    }

    #[test]
    fn new_function_defaults() {
        let func = make_simple_func("test");
        assert_eq!(func.name, "test");
        assert_eq!(func.calling_convention, CallingConvention::C);
        assert_eq!(func.linkage, Linkage::External);
        assert!(!func.is_variadic);
        assert!(func.is_definition);
        assert_eq!(func.alignment, 16);
        assert!(func.section.is_none());
        assert_eq!(func.attributes, FunctionAttributes::default());
    }

    #[test]
    fn new_function_with_params_registers_values() {
        let func = make_two_param_func();
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.local_values.len(), 2);
        assert_eq!(func.next_value_id, 2);
    }

    #[test]
    fn new_function_param_value_ids_tracked() {
        let func = make_two_param_func();
        // Parameter values should be accessible via get_value_type.
        let ty0 = func.get_value_type(ValueId(0));
        assert_eq!(*ty0, IrType::I32);
        let ty1 = func.get_value_type(ValueId(1));
        assert_eq!(*ty1, IrType::I32);
    }

    #[test]
    fn new_function_return_type_stored() {
        let func = IrFunction::new("void_fn".into(), IrType::Void, vec![]);
        assert!(func.return_type.is_void());
        assert!(func.returns_void());
    }

    // -----------------------------------------------------------------------
    // Block management tests
    // -----------------------------------------------------------------------

    #[test]
    fn add_basic_block_returns_id() {
        let mut func = make_simple_func("f");
        let bb1 = BasicBlock::new(BasicBlockId(1), Some("then".into()));
        let id = func.add_basic_block(bb1);
        assert_eq!(id, BasicBlockId(1));
        assert_eq!(func.block_count(), 2);
    }

    #[test]
    fn add_multiple_blocks() {
        let mut func = make_simple_func("f");
        for i in 1..=5 {
            let bb = BasicBlock::new(BasicBlockId(i), None);
            func.add_basic_block(bb);
        }
        assert_eq!(func.block_count(), 6); // entry + 5 added
    }

    #[test]
    fn get_block_by_id() {
        let mut func = make_simple_func("f");
        let bb1 = BasicBlock::new(BasicBlockId(1), Some("target".into()));
        func.add_basic_block(bb1);

        let block = func.get_block(BasicBlockId(1));
        assert_eq!(block.id, BasicBlockId(1));
        assert_eq!(block.name.as_deref(), Some("target"));
    }

    #[test]
    fn get_block_mut_allows_modification() {
        let mut func = make_simple_func("f");
        let bb1 = BasicBlock::new(BasicBlockId(1), None);
        func.add_basic_block(bb1);

        // Add an instruction to the block through the mutable reference.
        let block = func.get_block_mut(BasicBlockId(1));
        block.add_instruction(alloca(10));
        assert_eq!(block.instructions().len(), 1);

        // Verify the instruction persists when accessed immutably.
        let block_immut = func.get_block(BasicBlockId(1));
        assert_eq!(block_immut.instructions().len(), 1);
    }

    #[test]
    fn entry_block_returns_correct_block() {
        let func = make_simple_func("f");
        let entry = func.entry_block();
        assert_eq!(entry.id, func.entry_block_id);
    }

    #[test]
    fn blocks_returns_all_in_order() {
        let mut func = make_simple_func("f");
        func.add_basic_block(BasicBlock::new(BasicBlockId(1), None));
        func.add_basic_block(BasicBlock::new(BasicBlockId(2), None));

        let blocks = func.blocks();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].id, BasicBlockId(0)); // entry
        assert_eq!(blocks[1].id, BasicBlockId(1));
        assert_eq!(blocks[2].id, BasicBlockId(2));
    }

    #[test]
    fn blocks_mut_allows_bulk_modification() {
        let mut func = make_simple_func("f");
        func.add_basic_block(BasicBlock::new(BasicBlockId(1), None));

        for bb in func.blocks_mut() {
            bb.add_predecessor(BasicBlockId(99));
        }

        // Verify modifications persisted.
        for bb in func.blocks() {
            assert!(bb.predecessors().contains(&BasicBlockId(99)));
        }
    }

    #[test]
    fn remove_block_succeeds() {
        let mut func = make_simple_func("f");
        func.add_basic_block(BasicBlock::new(BasicBlockId(1), None));
        func.add_basic_block(BasicBlock::new(BasicBlockId(2), None));
        assert_eq!(func.block_count(), 3);

        func.remove_block(BasicBlockId(1));
        assert_eq!(func.block_count(), 2);
        // Block 2 should still be present.
        let _ = func.get_block(BasicBlockId(2));
    }

    #[test]
    #[should_panic(expected = "Cannot remove the entry block")]
    fn remove_entry_block_panics() {
        let mut func = make_simple_func("f");
        func.remove_block(BasicBlockId(0));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_nonexistent_block_panics() {
        let func = make_simple_func("f");
        let _ = func.get_block(BasicBlockId(999));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn remove_nonexistent_block_panics() {
        let mut func = make_simple_func("f");
        func.remove_block(BasicBlockId(42));
    }

    // -----------------------------------------------------------------------
    // SSA value management tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_value_allocates_sequential_ids() {
        let mut func = make_simple_func("f");
        let v0 = func.new_value(IrType::I32, Some("x".into()));
        let v1 = func.new_value(IrType::I64, None);
        let v2 = func.new_value(IrType::Ptr, Some("p".into()));
        assert_eq!(v0, ValueId(0));
        assert_eq!(v1, ValueId(1));
        assert_eq!(v2, ValueId(2));
    }

    #[test]
    fn new_value_with_params_continues_numbering() {
        let mut func = make_two_param_func(); // params get ValueId(0), ValueId(1)
        let v2 = func.new_value(IrType::I32, Some("local".into()));
        assert_eq!(v2, ValueId(2));
        assert_eq!(func.next_value_id, 3);
    }

    #[test]
    fn get_value_type_fast_path() {
        let mut func = make_simple_func("f");
        func.new_value(IrType::I32, None);
        func.new_value(IrType::F64, None);

        assert_eq!(*func.get_value_type(ValueId(0)), IrType::I32);
        assert_eq!(*func.get_value_type(ValueId(1)), IrType::F64);
    }

    #[test]
    fn get_value_type_for_params() {
        let func = make_two_param_func();
        assert_eq!(*func.get_value_type(ValueId(0)), IrType::I32);
        assert_eq!(*func.get_value_type(ValueId(1)), IrType::I32);
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_value_type_nonexistent_panics() {
        let func = make_simple_func("f");
        let _ = func.get_value_type(ValueId(999));
    }

    #[test]
    fn get_value_info_returns_some() {
        let mut func = make_simple_func("f");
        func.new_value(IrType::I16, Some("short_var".into()));

        let info = func.get_value_info(ValueId(0)).unwrap();
        assert_eq!(info.id, ValueId(0));
        assert_eq!(info.ty, IrType::I16);
        assert_eq!(info.name.as_deref(), Some("short_var"));
    }

    #[test]
    fn get_value_info_returns_none_for_missing() {
        let func = make_simple_func("f");
        assert!(func.get_value_info(ValueId(42)).is_none());
    }

    #[test]
    fn value_count_tracks_all_values() {
        let mut func = make_two_param_func(); // 2 param values
        func.new_value(IrType::I32, None);
        func.new_value(IrType::I32, None);
        assert_eq!(func.value_count(), 4); // 2 params + 2 locals
    }

    // -----------------------------------------------------------------------
    // Convenience method tests
    // -----------------------------------------------------------------------

    #[test]
    fn returns_void_true_for_void() {
        let func = IrFunction::new("void_fn".into(), IrType::Void, vec![]);
        assert!(func.returns_void());
    }

    #[test]
    fn returns_void_false_for_int() {
        let func = make_simple_func("int_fn");
        assert!(!func.returns_void());
    }

    #[test]
    fn is_weak_from_linkage() {
        let mut func = make_simple_func("f");
        func.linkage = Linkage::Weak;
        assert!(func.is_weak());
    }

    #[test]
    fn is_weak_from_attribute() {
        let mut func = make_simple_func("f");
        func.attributes.is_weak = true;
        assert!(func.is_weak());
    }

    #[test]
    fn is_internal_detection() {
        let mut func = make_simple_func("f");
        assert!(!func.is_internal());
        func.linkage = Linkage::Internal;
        assert!(func.is_internal());
    }

    #[test]
    fn is_noreturn_detection() {
        let mut func = make_simple_func("f");
        assert!(!func.is_noreturn());
        func.attributes.is_noreturn = true;
        assert!(func.is_noreturn());
    }

    #[test]
    fn create_block_auto_ids() {
        let mut func = make_simple_func("f");
        let id1 = func.create_block(Some("loop.header".into()));
        let id2 = func.create_block(Some("loop.body".into()));
        assert_eq!(id1, BasicBlockId(1));
        assert_eq!(id2, BasicBlockId(2));
        assert_eq!(func.block_count(), 3);
    }

    // -----------------------------------------------------------------------
    // Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn display_simple_function() {
        let mut func = make_simple_func("main");
        // Add a return to the entry block.
        let entry_id = func.entry_block_id;
        func.get_block_mut(entry_id)
            .add_instruction(ret(Some(ValueId(0))));

        let text = format!("{}", func);
        assert!(text.contains("define"));
        assert!(text.contains("external"));
        assert!(text.contains("ccc"));
        assert!(text.contains("i32"));
        assert!(text.contains("@main"));
        assert!(text.contains("{"));
        assert!(text.contains("}"));
    }

    #[test]
    fn display_function_with_params() {
        let func = make_two_param_func();
        let text = format!("{}", func);
        assert!(text.contains("@add"));
        assert!(text.contains("i32 %v0"));
        assert!(text.contains("i32 %v1"));
    }

    #[test]
    fn display_variadic_function() {
        let params = vec![Parameter {
            name: Some("fmt".into()),
            ty: IrType::Ptr,
            id: ValueId(0),
        }];
        let mut func = IrFunction::new("printf".into(), IrType::I32, params);
        func.is_variadic = true;
        func.is_definition = false;

        let text = format!("{}", func);
        assert!(text.contains("declare"));
        assert!(text.contains("..."));
        assert!(!text.contains("{"));
    }

    #[test]
    fn display_function_with_attributes() {
        let mut func = make_simple_func("abort");
        func.attributes.is_noreturn = true;
        func.attributes.is_cold = true;

        let text = format!("{}", func);
        assert!(text.contains("noreturn"));
        assert!(text.contains("cold"));
    }

    #[test]
    fn display_function_with_section() {
        let mut func = make_simple_func("init_fn");
        func.section = Some(".init.text".into());

        let text = format!("{}", func);
        assert!(text.contains("section \".init.text\""));
    }

    #[test]
    fn display_internal_linkage() {
        let mut func = make_simple_func("helper");
        func.linkage = Linkage::Internal;

        let text = format!("{}", func);
        assert!(text.contains("internal"));
    }

    // -----------------------------------------------------------------------
    // Integration: blocks with instructions
    // -----------------------------------------------------------------------

    #[test]
    fn function_with_cfg_edges() {
        let mut func = make_simple_func("f");
        let mut bb1 = BasicBlock::new(BasicBlockId(1), Some("then".into()));
        let mut bb2 = BasicBlock::new(BasicBlockId(2), Some("else".into()));
        let mut bb3 = BasicBlock::new(BasicBlockId(3), Some("merge".into()));

        // Set up successors on entry block.
        func.get_block_mut(BasicBlockId(0))
            .add_successor(BasicBlockId(1));
        func.get_block_mut(BasicBlockId(0))
            .add_successor(BasicBlockId(2));

        // Set up predecessors and successors.
        bb1.add_predecessor(BasicBlockId(0));
        bb1.add_successor(BasicBlockId(3));
        bb1.add_instruction(branch(BasicBlockId(3)));

        bb2.add_predecessor(BasicBlockId(0));
        bb2.add_successor(BasicBlockId(3));
        bb2.add_instruction(branch(BasicBlockId(3)));

        bb3.add_predecessor(BasicBlockId(1));
        bb3.add_predecessor(BasicBlockId(2));
        bb3.add_instruction(ret(None));

        func.add_basic_block(bb1);
        func.add_basic_block(bb2);
        func.add_basic_block(bb3);

        assert_eq!(func.block_count(), 4);

        // Verify CFG structure.
        let entry = func.entry_block();
        assert_eq!(entry.successors().len(), 2);
        let merge = func.get_block(BasicBlockId(3));
        assert_eq!(merge.predecessors().len(), 2);
        assert!(merge.terminator().unwrap().is_terminator());
    }

    #[test]
    fn function_with_phi_nodes() {
        let mut func = make_simple_func("phi_test");

        // Create a merge block with phi nodes.
        let mut merge = BasicBlock::new(BasicBlockId(1), Some("merge".into()));
        merge.add_instruction(phi(10, vec![(1, 0), (2, 0)]));
        merge.add_instruction(ret(Some(ValueId(10))));
        func.add_basic_block(merge);

        // Verify phi node query.
        let merge_block = func.get_block(BasicBlockId(1));
        assert_eq!(merge_block.phi_nodes().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Various IrType usage in values (ensures all type variants are covered)
    // -----------------------------------------------------------------------

    #[test]
    fn value_types_cover_all_variants() {
        let mut func = make_simple_func("type_test");

        // Exercise all IrType variants through new_value.
        let void_id = func.new_value(IrType::Void, None);
        let i1_id = func.new_value(IrType::I1, None);
        let i8_id = func.new_value(IrType::I8, None);
        let i16_id = func.new_value(IrType::I16, None);
        let i32_id = func.new_value(IrType::I32, None);
        let i64_id = func.new_value(IrType::I64, None);
        let i128_id = func.new_value(IrType::I128, None);
        let f32_id = func.new_value(IrType::F32, None);
        let f64_id = func.new_value(IrType::F64, None);
        let f80_id = func.new_value(IrType::F80, None);
        let ptr_id = func.new_value(IrType::Ptr, None);
        let arr_id = func.new_value(
            IrType::Array {
                element: Box::new(IrType::I32),
                count: 10,
            },
            None,
        );
        let struct_id = func.new_value(
            IrType::Struct {
                fields: vec![IrType::I32, IrType::Ptr],
                packed: false,
            },
            None,
        );
        let fn_id = func.new_value(
            IrType::Function {
                return_type: Box::new(IrType::I32),
                param_types: vec![IrType::I32, IrType::Ptr],
                is_variadic: false,
            },
            None,
        );

        // Verify each type is retrievable.
        assert_eq!(*func.get_value_type(void_id), IrType::Void);
        assert_eq!(*func.get_value_type(i1_id), IrType::I1);
        assert_eq!(*func.get_value_type(i8_id), IrType::I8);
        assert_eq!(*func.get_value_type(i16_id), IrType::I16);
        assert_eq!(*func.get_value_type(i32_id), IrType::I32);
        assert_eq!(*func.get_value_type(i64_id), IrType::I64);
        assert_eq!(*func.get_value_type(i128_id), IrType::I128);
        assert_eq!(*func.get_value_type(f32_id), IrType::F32);
        assert_eq!(*func.get_value_type(f64_id), IrType::F64);
        assert_eq!(*func.get_value_type(f80_id), IrType::F80);
        assert_eq!(*func.get_value_type(ptr_id), IrType::Ptr);
        assert!(func.get_value_type(arr_id).is_array());
        assert!(func.get_value_type(struct_id).is_struct());
        assert!(func.get_value_type(fn_id).is_function());
    }

    // -----------------------------------------------------------------------
    // Clone test
    // -----------------------------------------------------------------------

    #[test]
    fn function_clone_is_independent() {
        let mut func = make_two_param_func();
        func.new_value(IrType::I32, Some("v".into()));

        let mut cloned = func.clone();
        cloned.name = "add_clone".into();
        cloned.new_value(IrType::I64, None);

        // Original unchanged.
        assert_eq!(func.name, "add");
        assert_eq!(func.value_count(), 3);
        // Clone has extra value.
        assert_eq!(cloned.name, "add_clone");
        assert_eq!(cloned.value_count(), 4);
    }
}
