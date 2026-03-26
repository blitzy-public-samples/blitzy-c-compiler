//! # IR Module — Translation Unit Representation
//!
//! This module defines [`IrModule`], the top-level IR data structure for BCC.
//! An `IrModule` represents an entire C translation unit — the IR equivalent
//! of a single compiled `.c` file.
//!
//! ## Contents
//!
//! An IR module contains:
//!
//! - **Global variables** ([`GlobalVariable`]) — file-scope variables with
//!   optional constant initializers, linkage, visibility, alignment, and
//!   section placement attributes.
//! - **Function definitions** ([`IrFunction`]) — functions with bodies
//!   (basic blocks, instructions, SSA values).
//! - **Function declarations** ([`FunctionDeclaration`]) — extern function
//!   signatures without bodies (resolved at link time).
//! - **String literal pool** ([`StringLiteral`]) — deduplicated byte
//!   sequences for string constants placed in `.rodata`.
//! - **Top-level inline assembly** ([`InlineAsmBlock`]) — file-scope `asm`
//!   directives emitted verbatim into the assembly output.
//! - **Type metadata** ([`TypeMetadata`]) — type name/IR type associations
//!   used for DWARF debug information generation.
//!
//! ## Backend Consumption
//!
//! The backend's code generation driver (`src/backend/generation.rs`, Phase 10)
//! consumes an `IrModule` to produce an ELF object file (`.o`), executable
//! (`ET_EXEC`), or shared object (`ET_DYN`). The module provides fast
//! O(1) name-based lookups via [`FxHashMap`]-backed index maps for globals,
//! functions, and string literals.
//!
//! ## String Pool Deduplication
//!
//! The [`IrModule::intern_string`] method deduplicates string constants by
//! byte content. Identical byte sequences (including null terminators) share
//! a single `.rodata` entry, reducing output binary size. Each interned
//! string receives a unique numeric ID and a generated label (e.g.,
//! `.L.str.0`, `.L.str.1`).
//!
//! ## Linkage and Visibility
//!
//! [`Linkage`] controls how the static linker resolves symbol references
//! across translation units (external, internal, weak, common). [`Visibility`]
//! controls how the dynamic linker handles symbol resolution in shared
//! libraries (default, hidden, protected). These map directly to C storage
//! class specifiers and GCC `__attribute__((visibility(...)))`.
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::ir::function`, `crate::ir::types`,
//! `crate::common::fx_hash`, and the Rust standard library — no external
//! crates are used.

use crate::common::fx_hash::FxHashMap;
use crate::ir::function::IrFunction;
use crate::ir::instructions::Value;
use crate::ir::types::IrType;

// ===========================================================================
// Linkage — symbol resolution semantics for module-level symbols
// ===========================================================================

/// Symbol linkage kind — controls how the linker resolves references to
/// the symbol across translation units.
///
/// Applies to both global variables and function declarations at the
/// module level. Maps directly to C storage class specifiers and GCC
/// attributes:
///
/// - `External` — visible to other translation units (default for non-`static` symbols)
/// - `Internal` — file-scoped (`static` symbols)
/// - `Weak` — `__attribute__((weak))`, can be overridden by a strong definition
/// - `Common` — tentative definition semantics (uninitialized globals)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage {
    /// Externally visible — the default for non-`static` symbols.
    /// The symbol is exported and can be referenced from other translation units.
    External,

    /// Internal linkage — `static` variables/functions.
    /// The symbol is local to the current translation unit and is not
    /// visible to the linker outside of the defining object file.
    Internal,

    /// Weak linkage — `__attribute__((weak))`.
    /// The symbol can be overridden by a strong (non-weak) definition
    /// from another translation unit. If no strong definition exists,
    /// the weak definition is used.
    Weak,

    /// Common linkage — tentative definition semantics.
    /// Used for uninitialized global variables that may have multiple
    /// tentative definitions across translation units. The linker merges
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
// Visibility — ELF symbol visibility for module-level symbols
// ===========================================================================

/// ELF symbol visibility — controls how the dynamic linker resolves
/// references to the symbol in shared libraries.
///
/// Maps to GCC's `__attribute__((visibility("...")))`:
///
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
    /// object. References within the same shared object are resolved
    /// directly, bypassing the GOT/PLT. This is the most restrictive
    /// visibility and enables the best code generation for PIC.
    Hidden,

    /// Protected visibility — the symbol is visible to other shared
    /// objects but cannot be preempted. References within the defining
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
// Constant — initializer values for global variables
// ===========================================================================

/// Constant value for global variable initializers.
///
/// Represents all possible compile-time constant values that can appear
/// as initializers for global variables, static locals, and entries in
/// the string literal pool. These map to the C11 constant expression
/// categories:
///
/// - **Integer and floating-point literals** — direct numeric values
/// - **String literals** — raw byte sequences (including null terminator)
/// - **Aggregate initializers** — struct and array initializers with
///   per-member/element constants (supporting designated initializers)
/// - **Address constants** — references to other global symbols
///   (`GlobalRef`) and null pointers (`Null`)
/// - **Zero-initialization** — BSS-eligible zero-filled storage
/// - **Undefined** — uninitialized storage (behavior is undefined if read)
///
/// # Long Double Representation
///
/// The `LongDouble` variant stores raw 80-bit extended precision bytes
/// (10 bytes) in little-endian format, matching the x87 FPU internal
/// representation. This avoids precision loss from conversion through
/// `f64` and maintains byte-exact fidelity required by the PUA encoding
/// mandate.
#[derive(Debug, Clone)]
pub enum Constant {
    /// Integer constant — covers all integer types from `_Bool` through
    /// `__int128`. The `i128` type provides sufficient range for all
    /// C integer constant expressions.
    Integer(i128),

    /// IEEE 754 double-precision floating-point constant.
    /// Used for `float` and `double` initializers (float values are
    /// widened to double for storage).
    Float(f64),

    /// 80-bit extended precision floating-point constant (`long double`
    /// on x86 platforms). Stored as raw bytes in little-endian format
    /// to preserve exact bit patterns without conversion through `f64`.
    LongDouble([u8; 10]),

    /// Raw byte sequence — typically a string literal including its
    /// null terminator. Bytes are stored exactly as they appear in the
    /// source (with PUA decoding applied), preserving byte-exact fidelity
    /// for non-UTF-8 content.
    String(Vec<u8>),

    /// Zero-initialized storage — the global variable is placed in BSS
    /// (or equivalent) and filled with zero bytes. This is the implicit
    /// initializer for C globals without an explicit initializer.
    ZeroInit,

    /// Struct aggregate initializer — ordered list of per-field constant
    /// values. The field order matches the struct's declaration order,
    /// with designated initializer reordering already resolved by the
    /// semantic analysis phase.
    Struct(Vec<Constant>),

    /// Array aggregate initializer — ordered list of per-element constant
    /// values. Trailing unspecified elements are implicitly zero-initialized
    /// by the semantic analysis phase before IR lowering.
    Array(Vec<Constant>),

    /// Reference to another global symbol (address constant).
    /// The string is the name of the referenced global variable or
    /// function. Used for initializers like `int *p = &global_var;`.
    GlobalRef(String),

    /// Reference to a global symbol with a byte offset (address constant
    /// with addend).  Used for initializers like
    /// `const char *p = &array[42];` where the result is the symbol
    /// address plus a compile-time byte offset.
    GlobalRefOffset(String, i64),

    /// Null pointer constant — `(void *)0` or any integer constant
    /// expression with value 0 cast to a pointer type.
    Null,

    /// Undefined/uninitialized value — represents storage that has no
    /// defined initial value. Reading from undefined storage is UB in C.
    /// Used as a placeholder during IR construction for variables that
    /// are assigned before use.
    Undefined,
}

impl std::fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant::Integer(v) => write!(f, "{}", v),
            Constant::Float(v) => write!(f, "{:e}", v),
            Constant::LongDouble(bytes) => {
                write!(f, "long_double(0x")?;
                for b in bytes.iter().rev() {
                    write!(f, "{:02x}", b)?;
                }
                write!(f, ")")
            }
            Constant::String(bytes) => {
                write!(f, "c\"")?;
                for &b in bytes {
                    if b.is_ascii_graphic() || b == b' ' {
                        write!(f, "{}", b as char)?;
                    } else {
                        write!(f, "\\{:02x}", b)?;
                    }
                }
                write!(f, "\"")
            }
            Constant::ZeroInit => write!(f, "zeroinit"),
            Constant::Struct(fields) => {
                write!(f, "{{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, " }}")
            }
            Constant::Array(elems) => {
                write!(f, "[ ")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, " ]")
            }
            Constant::GlobalRef(name) => write!(f, "@{}", name),
            Constant::GlobalRefOffset(name, off) => write!(f, "@{}+{}", name, off),
            Constant::Null => write!(f, "null"),
            Constant::Undefined => write!(f, "undef"),
        }
    }
}

// ===========================================================================
// GlobalVariable — file-scope variable with initializer
// ===========================================================================

/// A global variable at the IR level.
///
/// Represents a file-scope variable declaration or definition. Global
/// variables may have constant initializers (evaluated at compile time),
/// explicit linkage, visibility, section placement, alignment, and
/// thread-local storage attributes.
///
/// # Placement in ELF sections
///
/// - `is_constant == true` → placed in `.rodata` (read-only data)
/// - `initializer == Some(Constant::ZeroInit)` → placed in `.bss`
/// - `initializer == Some(_)` → placed in `.data`
/// - `section == Some(name)` → placed in the named section (overrides above)
/// - `is_thread_local == true` → placed in `.tdata` or `.tbss`
///
/// # Alignment
///
/// When `alignment` is `Some(n)`, the variable is aligned to `n` bytes
/// in the output section. When `None`, the backend uses the natural
/// alignment for the variable's IR type on the target architecture.
#[derive(Debug, Clone)]
pub struct GlobalVariable {
    /// Symbol name of the global variable.
    pub name: String,

    /// IR type of the global variable — determines size and alignment.
    pub ty: IrType,

    /// Optional constant initializer expression.
    ///
    /// - `Some(Constant::ZeroInit)` — zero-initialized (BSS)
    /// - `Some(constant)` — initialized with the given constant value
    /// - `None` — external declaration (defined in another translation unit)
    pub initializer: Option<Constant>,

    /// Whether this is a definition (has storage) or an external declaration
    /// (resolved at link time).
    ///
    /// - `true` — this translation unit provides the definition
    /// - `false` — declared `extern`, definition is elsewhere
    pub is_definition: bool,

    /// Symbol linkage — controls cross-TU resolution behavior.
    pub linkage: Linkage,

    /// Whether the variable uses thread-local storage (`_Thread_local`).
    ///
    /// Thread-local variables are placed in `.tdata`/`.tbss` ELF sections
    /// and accessed via the TLS ABI (e.g., `fs` segment on x86-64).
    pub is_thread_local: bool,

    /// Optional section name from `__attribute__((section("...")))`.
    ///
    /// Overrides the default section placement (`.data`, `.rodata`, `.bss`).
    /// Used extensively by the Linux kernel for init/exit data sections.
    pub section: Option<String>,

    /// Optional explicit alignment from `__attribute__((aligned(N)))`.
    ///
    /// When `Some(n)`, the variable is aligned to `n` bytes. Must be a
    /// power of two. When `None`, natural alignment is used.
    pub alignment: Option<usize>,

    /// Whether the global is a constant (read-only after initialization).
    ///
    /// Constant globals are placed in `.rodata` instead of `.data`,
    /// enabling the loader to map the page as read-only for security.
    pub is_constant: bool,
}

impl GlobalVariable {
    /// Creates a new global variable with the given name, type, and initializer.
    ///
    /// Defaults: definition=true, external linkage, not thread-local,
    /// no section override, no alignment override, not constant.
    ///
    /// # Parameters
    ///
    /// - `name`: Symbol name for the variable.
    /// - `ty`: IR type determining size and natural alignment.
    /// - `initializer`: Optional constant initializer value.
    pub fn new(name: String, ty: IrType, initializer: Option<Constant>) -> Self {
        GlobalVariable {
            name,
            ty,
            initializer,
            is_definition: true,
            linkage: Linkage::External,
            is_thread_local: false,
            section: None,
            alignment: None,
            is_constant: false,
        }
    }
}

impl std::fmt::Display for GlobalVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{} = {} ", self.name, self.linkage)?;
        if self.is_constant {
            write!(f, "constant ")?;
        } else {
            write!(f, "global ")?;
        }
        write!(f, "{}", self.ty)?;
        if let Some(ref init) = self.initializer {
            write!(f, " {}", init)?;
        }
        if let Some(align) = self.alignment {
            write!(f, ", align {}", align)?;
        }
        if let Some(ref sec) = self.section {
            write!(f, ", section \"{}\"", sec)?;
        }
        if self.is_thread_local {
            write!(f, " thread_local")?;
        }
        Ok(())
    }
}

// ===========================================================================
// FunctionDeclaration — extern function signature (no body)
// ===========================================================================

/// An external function declaration at the IR level.
///
/// Represents a function that is declared but not defined in this
/// translation unit — typically an `extern` function whose body resides
/// in another object file or shared library. The linker resolves the
/// symbol at link time.
///
/// # Distinction from [`IrFunction`]
///
/// - `FunctionDeclaration` — signature only, no basic blocks, no body.
///   Used for calls to external functions (e.g., `printf`, `malloc`).
/// - [`IrFunction`] — complete function definition with basic blocks,
///   instructions, and SSA values.
#[derive(Debug, Clone)]
pub struct FunctionDeclaration {
    /// Symbol name of the declared function.
    pub name: String,

    /// Return type of the function.
    pub return_type: IrType,

    /// Parameter types (positional, no names needed for declarations).
    pub param_types: Vec<IrType>,

    /// Whether the function accepts variadic arguments (`...`).
    pub is_variadic: bool,

    /// Symbol linkage for the declaration.
    pub linkage: Linkage,

    /// ELF symbol visibility for the declaration.
    pub visibility: Visibility,
}

impl FunctionDeclaration {
    /// Creates a new function declaration with the given signature.
    ///
    /// Defaults: not variadic, external linkage, default visibility.
    ///
    /// # Parameters
    ///
    /// - `name`: Symbol name of the function.
    /// - `return_type`: The IR type returned by the function.
    /// - `param_types`: Ordered list of parameter IR types.
    pub fn new(name: String, return_type: IrType, param_types: Vec<IrType>) -> Self {
        FunctionDeclaration {
            name,
            return_type,
            param_types,
            is_variadic: false,
            linkage: Linkage::External,
            visibility: Visibility::Default,
        }
    }
}

impl std::fmt::Display for FunctionDeclaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "declare {} @{}(", self.return_type, self.name)?;
        for (i, pt) in self.param_types.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", pt)?;
        }
        if self.is_variadic {
            if !self.param_types.is_empty() {
                write!(f, ", ")?;
            }
            write!(f, "...")?;
        }
        write!(f, ")")
    }
}

// ===========================================================================
// StringLiteral — deduplicated string constant pool entry
// ===========================================================================

/// A string literal entry in the module's string constant pool.
///
/// String literals are deduplicated by byte content — identical byte
/// sequences (including null terminators) share a single pool entry.
/// Each entry receives a unique numeric ID and a generated label used
/// by the backend to reference the string in `.rodata`.
///
/// # Byte-Exact Fidelity
///
/// The `bytes` field stores the exact byte sequence from the C source
/// (after PUA decoding), including the null terminator. This preserves
/// non-UTF-8 content that the Linux kernel embeds in string literals
/// and inline assembly operands.
#[derive(Debug, Clone)]
pub struct StringLiteral {
    /// Unique numeric identifier for this string literal.
    /// Assigned sequentially (0, 1, 2, ...) by [`IrModule::intern_string`].
    pub id: u32,

    /// Raw bytes of the string literal, including null terminator.
    /// Byte-exact content preserved through PUA encoding/decoding.
    pub bytes: Vec<u8>,

    /// Generated label name for referencing this string in assembly/ELF.
    /// Format: `.L.str.N` where N is the string's ID.
    pub label: String,
}

impl std::fmt::Display for StringLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = ", self.label)?;
        write!(f, "c\"")?;
        for &b in &self.bytes {
            if b.is_ascii_graphic() || b == b' ' {
                write!(f, "{}", b as char)?;
            } else {
                write!(f, "\\{:02x}", b)?;
            }
        }
        write!(f, "\"")
    }
}

// ===========================================================================
// InlineAsmBlock — top-level file-scope assembly
// ===========================================================================

/// A top-level (file-scope) inline assembly block.
///
/// Represents `asm("...")` or `__asm__("...")` statements at file scope
/// (outside any function body). These are emitted verbatim into the
/// assembly output before any code generation, used by the Linux kernel
/// for `.pushsection`/`.popsection` directives and other assembler-level
/// constructs.
///
/// This is distinct from inline assembly *within* function bodies, which
/// is represented as an [`Instruction::InlineAsm`] in the IR instruction
/// stream.
#[derive(Debug, Clone)]
pub struct InlineAsmBlock {
    /// Raw assembly text to emit verbatim.
    pub assembly: String,
}

impl InlineAsmBlock {
    /// Creates a new inline assembly block with the given text.
    #[inline]
    pub fn new(assembly: String) -> Self {
        InlineAsmBlock { assembly }
    }
}

impl std::fmt::Display for InlineAsmBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "module asm \"{}\"", self.assembly)
    }
}

// ===========================================================================
// TypeMetadata — type information for debug/reflection
// ===========================================================================

/// Type metadata entry for DWARF debug information generation.
///
/// Associates a human-readable type name with its IR-level type
/// representation. Used by the DWARF emitter (`src/backend/dwarf/`)
/// to generate `.debug_info` type DIEs when `-g` is specified.
///
/// # Examples
///
/// - `TypeMetadata { name: "struct point", ir_type: IrType::Struct(...) }`
/// - `TypeMetadata { name: "size_t", ir_type: IrType::I64 }`
#[derive(Debug, Clone)]
pub struct TypeMetadata {
    /// Human-readable C type name (e.g., `"struct point"`, `"size_t"`).
    pub name: String,

    /// Corresponding IR type for layout and encoding information.
    pub ir_type: IrType,
}

impl TypeMetadata {
    /// Creates a new type metadata entry.
    #[inline]
    pub fn new(name: String, ir_type: IrType) -> Self {
        TypeMetadata { name, ir_type }
    }
}

impl std::fmt::Display for TypeMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!type {} = {}", self.name, self.ir_type)
    }
}

// ===========================================================================
// IrModule — top-level IR container for a translation unit
// ===========================================================================

/// Top-level IR container representing an entire C translation unit.
///
/// `IrModule` is the primary data structure consumed by the backend's
/// code generation driver (Phase 10). It aggregates all IR entities
/// produced by the frontend and middle-end passes:
///
/// - Global variables with initializers
/// - Function definitions (with bodies and SSA-form basic blocks)
/// - External function declarations (signatures without bodies)
/// - A deduplicated string literal pool
/// - Top-level inline assembly blocks
/// - Type metadata for DWARF debug information
///
/// ## Lookup Performance
///
/// Three [`FxHashMap`]-backed index maps provide O(1) name-based lookups:
///
/// - `global_map`: global variable name → index in `globals`
/// - `function_map`: function name → index in `functions`
/// - `string_map`: byte content → index in `string_pool`
///
/// These use the FxHash algorithm (fast, non-cryptographic Fibonacci
/// hashing) as mandated by the project's zero-dependency architecture
/// for performance-critical symbol table operations.
///
/// ## Construction Pattern
///
/// ```ignore
/// use bcc::ir::module::{IrModule, GlobalVariable, Linkage, Constant};
/// use bcc::ir::types::IrType;
///
/// let mut module = IrModule::new("hello.c".to_string());
///
/// // Add a global string constant
/// let str_id = module.intern_string(b"Hello, World!\n\0".to_vec());
///
/// // Add a global variable
/// let var = GlobalVariable::new(
///     "message".to_string(),
///     IrType::Ptr,
///     Some(Constant::String(b"Hello, World!\n\0".to_vec())),
/// );
/// module.add_global(var);
/// ```
pub struct IrModule {
    /// Translation unit name (typically the source filename, e.g., `"hello.c"`).
    pub name: String,

    /// Global variable definitions and declarations, in declaration order.
    pub globals: Vec<GlobalVariable>,

    /// Function definitions with bodies (basic blocks and instructions).
    pub functions: Vec<IrFunction>,

    /// External function declarations (signatures without bodies).
    pub declarations: Vec<FunctionDeclaration>,

    /// Deduplicated string literal pool.
    ///
    /// Each unique byte sequence is stored exactly once. The backend
    /// emits these as labeled constants in the `.rodata` section.
    pub string_pool: Vec<StringLiteral>,

    /// Top-level (file-scope) inline assembly blocks.
    ///
    /// Emitted verbatim before any generated code, preserving the
    /// declaration order from the source file.
    pub inline_asm_blocks: Vec<InlineAsmBlock>,

    /// Type metadata entries for DWARF debug information.
    ///
    /// Populated during IR lowering when `-g` is active.
    pub type_metadata: Vec<TypeMetadata>,

    /// Fast lookup: global variable name → index in `globals`.
    global_map: FxHashMap<String, usize>,

    /// Fast lookup: function name → index in `functions`.
    function_map: FxHashMap<String, usize>,

    /// Fast lookup: string byte content → index in `string_pool`.
    /// Used by [`intern_string`](IrModule::intern_string) for deduplication.
    string_map: FxHashMap<Vec<u8>, usize>,

    /// Map from IR `Value` to the function name it references.
    /// Populated during IR lowering when `lower_identifier` resolves
    /// a name to a known function definition or declaration.
    pub func_ref_map: FxHashMap<Value, String>,

    /// Maps IR `Value`s that represent the *address* of a global variable
    /// to the corresponding global variable name.  Populated during IR
    /// lowering so the backend can emit RIP-relative (x86-64) or
    /// ADRP/LDR (AArch64) or LUI/LD (RISC-V) global accesses.
    pub global_var_refs: FxHashMap<Value, String>,

    /// Maps function name to its C-level return type.  Populated during
    /// IR lowering of function declarations and definitions so that
    /// `lower_function_call` can recover the correct C return type even
    /// when the callee IR value is typed as an opaque `Pointer(Void)`.
    pub func_c_return_types: FxHashMap<String, crate::common::types::CType>,

    /// Maps function name to its C-level parameter types.  Populated during
    /// IR lowering of function declarations and definitions so that
    /// `lower_function_call` can insert correct implicit conversions
    /// (e.g. sign-extending `int` → `long long`) when the callee IR value
    /// is typed as an opaque `Pointer(Void)`.
    pub func_c_param_types: FxHashMap<String, Vec<crate::common::types::CType>>,

    /// Maps global variable name to its original C-level type.  Populated
    /// during `lower_global_variable` so that expression lowering can recover
    /// the full C type (including struct field names and function pointer
    /// signatures) instead of relying on the lossy `ir_type_to_approx_ctype`
    /// conversion from the stored `IrType`.
    pub global_c_types: FxHashMap<String, crate::common::types::CType>,

    /// GCC global register variables: `register T *name __asm__("reg");`
    ///
    /// Maps the variable name to (register_name, C type).  These
    /// declarations bind a C identifier to a specific hardware register
    /// (e.g. RISC-V `tp` for `current`).  No storage is allocated — reads
    /// emit an inline-asm register read; writes emit a register store.
    pub register_globals: FxHashMap<String, (String, crate::common::types::CType)>,

    /// Function-level alignment overrides from `__attribute__((aligned(N)))`.
    ///
    /// Populated during both function declaration and definition lowering.
    /// A declaration `void f(void) __attribute__((aligned(256)));` followed
    /// by a definition `void f(void) {}` stores `("f", 256)`.  This allows
    /// `__alignof__(f)` to return the correct value even when the definition
    /// itself does not repeat the attribute (attribute inheritance from
    /// prior declarations — standard C behaviour).
    pub func_alignments: FxHashMap<String, usize>,

    /// Symbol aliases: `__attribute__((alias("target")))`.
    ///
    /// Maps the alias symbol name to the target symbol name. Both functions
    /// and variables can be aliased. The ELF writer emits the alias as a
    /// symbol that shares the same address as the target.
    pub symbol_aliases: Vec<(String, String)>,
}

// ===========================================================================
// IrModule — construction
// ===========================================================================

impl IrModule {
    /// Creates a new, empty IR module for the given translation unit.
    ///
    /// All collections are initialized empty and the lookup maps are
    /// default-constructed (empty `FxHashMap`s).
    ///
    /// # Parameters
    ///
    /// - `name`: Translation unit identifier (typically the source filename).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let module = IrModule::new("main.c".to_string());
    /// assert!(module.globals().is_empty());
    /// assert!(module.functions().is_empty());
    /// assert!(module.declarations().is_empty());
    /// assert!(module.string_pool().is_empty());
    /// ```
    pub fn new(name: String) -> Self {
        IrModule {
            name,
            globals: Vec::new(),
            functions: Vec::new(),
            declarations: Vec::new(),
            string_pool: Vec::new(),
            inline_asm_blocks: Vec::new(),
            type_metadata: Vec::new(),
            global_map: FxHashMap::default(),
            function_map: FxHashMap::default(),
            string_map: FxHashMap::default(),
            func_ref_map: FxHashMap::default(),
            global_var_refs: FxHashMap::default(),
            func_c_return_types: FxHashMap::default(),
            func_c_param_types: FxHashMap::default(),
            global_c_types: FxHashMap::default(),
            register_globals: FxHashMap::default(),
            func_alignments: FxHashMap::default(),
            symbol_aliases: Vec::new(),
        }
    }
}

// ===========================================================================
// IrModule — global variable management
// ===========================================================================

impl IrModule {
    /// Adds a global variable to the module.
    ///
    /// The variable is appended to the `globals` list and indexed in
    /// `global_map` for O(1) name-based retrieval. If a global with
    /// the same name already exists, the new entry replaces the old
    /// one in the lookup map (the old entry remains in the vector but
    /// is shadowed for lookups).
    ///
    /// # Parameters
    ///
    /// - `global`: The global variable to add.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut module = IrModule::new("test.c".to_string());
    /// let var = GlobalVariable::new("x".to_string(), IrType::I32, Some(Constant::Integer(42)));
    /// module.add_global(var);
    /// assert!(module.get_global("x").is_some());
    /// ```
    pub fn add_global(&mut self, global: GlobalVariable) {
        // Deduplicate by name: if a global with the same name already exists,
        // a definition takes priority over a declaration (extern).  This
        // correctly merges the common C pattern where the same extern
        // declaration appears in multiple included headers within one TU.
        if let Some(&existing_idx) = self.global_map.get(&global.name) {
            let existing = &self.globals[existing_idx];
            if global.is_definition && !existing.is_definition {
                // New entry is a definition replacing an extern declaration —
                // overwrite the existing slot so that the definition's storage
                // attributes (initializer, linkage, section, alignment) are
                // used for code generation.
                self.globals[existing_idx] = global;
            } else if global.is_definition && existing.is_definition {
                // Both are definitions (tentative definitions in the same TU).
                // C allows multiple tentative definitions; the latest one's
                // type and initializer are canonical.  Replace to match GCC.
                self.globals[existing_idx] = global;
            }
            // Otherwise the new entry is a plain extern declaration and the
            // existing entry already has equal or greater information — skip.
            return;
        }
        let index = self.globals.len();
        self.global_map.insert(global.name.clone(), index);
        self.globals.push(global);
    }

    /// Looks up a global variable by name.
    ///
    /// Returns `Some(&GlobalVariable)` if a global with the given name
    /// exists, or `None` otherwise. Lookup is O(1) via `FxHashMap`.
    ///
    /// # Parameters
    ///
    /// - `name`: The symbol name of the global variable to find.
    #[inline]
    pub fn get_global(&self, name: &str) -> Option<&GlobalVariable> {
        self.global_map
            .get(name)
            .and_then(|&idx| self.globals.get(idx))
    }

    /// Returns a slice of all global variables in declaration order.
    #[inline]
    pub fn globals(&self) -> &[GlobalVariable] {
        &self.globals
    }
}

// ===========================================================================
// IrModule — function definition management
// ===========================================================================

impl IrModule {
    /// Adds a function definition to the module.
    ///
    /// The function is appended to the `functions` list and indexed in
    /// `function_map` for O(1) name-based retrieval. Uses
    /// [`IrFunction.name`] for the lookup key.
    ///
    /// # Parameters
    ///
    /// - `func`: The IR function definition to add.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut module = IrModule::new("test.c".to_string());
    /// let func = IrFunction::new("main".to_string(), vec![], IrType::I32);
    /// module.add_function(func);
    /// assert!(module.get_function("main").is_some());
    /// ```
    pub fn add_function(&mut self, func: IrFunction) {
        let index = self.functions.len();
        self.function_map.insert(func.name.clone(), index);
        self.functions.push(func);
    }

    /// Looks up a function definition by name.
    ///
    /// Returns `Some(&IrFunction)` if a function with the given name
    /// exists, or `None` otherwise. Lookup is O(1) via `FxHashMap`.
    ///
    /// # Parameters
    ///
    /// - `name`: The symbol name of the function to find.
    #[inline]
    pub fn get_function(&self, name: &str) -> Option<&IrFunction> {
        self.function_map
            .get(name)
            .and_then(|&idx| self.functions.get(idx))
    }

    /// Looks up a function definition by name, returning a mutable reference.
    ///
    /// Used by optimization passes and transformations that need to modify
    /// a function's basic blocks, instructions, or metadata in-place.
    ///
    /// # Parameters
    ///
    /// - `name`: The symbol name of the function to find.
    #[inline]
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut IrFunction> {
        self.function_map
            .get(name)
            .copied()
            .and_then(move |idx| self.functions.get_mut(idx))
    }

    /// Returns a slice of all function definitions in insertion order.
    #[inline]
    pub fn functions(&self) -> &[IrFunction] {
        &self.functions
    }
}

// ===========================================================================
// IrModule — function declaration management
// ===========================================================================

impl IrModule {
    /// Adds an external function declaration to the module.
    ///
    /// Declarations represent functions that are referenced but not defined
    /// in this translation unit. They provide the signature information
    /// needed for correct calling convention handling during code generation.
    ///
    /// # Parameters
    ///
    /// - `decl`: The function declaration to add.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut module = IrModule::new("test.c".to_string());
    /// let decl = FunctionDeclaration::new(
    ///     "printf".to_string(),
    ///     IrType::I32,
    ///     vec![IrType::Ptr],
    /// );
    /// module.add_declaration(decl);
    /// ```
    pub fn add_declaration(&mut self, decl: FunctionDeclaration) {
        self.declarations.push(decl);
    }

    /// Returns a slice of all external function declarations.
    #[inline]
    pub fn declarations(&self) -> &[FunctionDeclaration] {
        &self.declarations
    }
}

// ===========================================================================
// IrModule — string literal pool management
// ===========================================================================

impl IrModule {
    /// Interns a string literal into the module's string pool.
    ///
    /// If the exact byte sequence already exists in the pool, returns the
    /// existing entry's ID without creating a duplicate. Otherwise, creates
    /// a new [`StringLiteral`] entry with a generated label (`.L.str.N`)
    /// and returns its assigned ID.
    ///
    /// # Deduplication
    ///
    /// Deduplication is performed by byte content using [`FxHashMap`].
    /// Two string literals with identical byte sequences (including null
    /// terminators) share a single pool entry and `.rodata` allocation.
    ///
    /// # Parameters
    ///
    /// - `bytes`: Raw byte content of the string literal, including the
    ///   null terminator.
    ///
    /// # Returns
    ///
    /// The unique `u32` ID for this string literal. The ID can be used
    /// to reference the string in IR instructions and is stable for the
    /// lifetime of the module.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut module = IrModule::new("test.c".to_string());
    /// let id1 = module.intern_string(b"hello\0".to_vec());
    /// let id2 = module.intern_string(b"hello\0".to_vec());
    /// assert_eq!(id1, id2); // deduplicated — same ID
    ///
    /// let id3 = module.intern_string(b"world\0".to_vec());
    /// assert_ne!(id1, id3); // different content — different ID
    /// ```
    pub fn intern_string(&mut self, bytes: Vec<u8>) -> u32 {
        // Check if this exact byte sequence already exists in the pool.
        if let Some(&existing_idx) = self.string_map.get(&bytes) {
            return self.string_pool[existing_idx].id;
        }

        // Create a new string literal entry with a generated label.
        let id = self.string_pool.len() as u32;
        let label = format!(".L.str.{}", id);
        let index = self.string_pool.len();

        self.string_map.insert(bytes.clone(), index);
        self.string_pool.push(StringLiteral { id, bytes, label });

        id
    }

    /// Returns a slice of all interned string literals.
    #[inline]
    pub fn string_pool(&self) -> &[StringLiteral] {
        &self.string_pool
    }
}

// ===========================================================================
// IrModule — inline assembly management
// ===========================================================================

impl IrModule {
    /// Adds a top-level (file-scope) inline assembly block.
    ///
    /// The assembly text is emitted verbatim by the backend before any
    /// generated code. Multiple blocks are emitted in insertion order.
    ///
    /// # Parameters
    ///
    /// - `asm`: The raw assembly text to emit.
    pub fn add_inline_asm(&mut self, asm: String) {
        self.inline_asm_blocks.push(InlineAsmBlock::new(asm));
    }
}

// ===========================================================================
// IrModule — Display implementation for IR dump
// ===========================================================================

impl std::fmt::Display for IrModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "; ModuleID = '{}'", self.name)?;
        writeln!(f)?;

        // Emit top-level inline assembly
        for asm_block in &self.inline_asm_blocks {
            writeln!(f, "{}", asm_block)?;
        }
        if !self.inline_asm_blocks.is_empty() {
            writeln!(f)?;
        }

        // Emit string pool
        for lit in &self.string_pool {
            writeln!(f, "{}", lit)?;
        }
        if !self.string_pool.is_empty() {
            writeln!(f)?;
        }

        // Emit global variables
        for global in &self.globals {
            writeln!(f, "{}", global)?;
        }
        if !self.globals.is_empty() {
            writeln!(f)?;
        }

        // Emit function declarations
        for decl in &self.declarations {
            writeln!(f, "{}", decl)?;
        }
        if !self.declarations.is_empty() {
            writeln!(f)?;
        }

        // Emit type metadata
        for meta in &self.type_metadata {
            writeln!(f, "{}", meta)?;
        }

        Ok(())
    }
}

// ===========================================================================
// IrModule — Debug implementation
// ===========================================================================

impl std::fmt::Debug for IrModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrModule")
            .field("name", &self.name)
            .field("globals_count", &self.globals.len())
            .field("functions_count", &self.functions.len())
            .field("declarations_count", &self.declarations.len())
            .field("string_pool_count", &self.string_pool.len())
            .field("inline_asm_count", &self.inline_asm_blocks.len())
            .field("type_metadata_count", &self.type_metadata.len())
            .finish()
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a simple IrFunction for testing.
    fn make_test_function(name: &str, is_def: bool) -> IrFunction {
        use crate::ir::basic_block::BasicBlock;
        use crate::ir::function::{
            CallingConvention, Linkage as FnLinkage, Visibility as FnVisibility,
        };

        IrFunction {
            name: name.to_string(),
            params: Vec::new(),
            return_type: IrType::I32,
            blocks: vec![BasicBlock::with_label(0, "entry".to_string())],
            calling_convention: CallingConvention::C,
            is_variadic: false,
            linkage: FnLinkage::External,
            visibility: FnVisibility::Default,
            is_noreturn: false,
            section: None,
            alignment: None,
            local_count: 0,
            value_count: 0,
            is_definition: is_def,
            constant_values: crate::common::fx_hash::FxHashMap::default(),
            float_constant_values: crate::common::fx_hash::FxHashMap::default(),
            func_ref_map: crate::common::fx_hash::FxHashMap::default(),
            global_var_refs: crate::common::fx_hash::FxHashMap::default(),
            local_var_debug_info: Vec::new(),
        }
    }

    #[test]
    fn test_new_module_is_empty() {
        let m = IrModule::new("test.c".to_string());
        assert_eq!(m.name, "test.c");
        assert!(m.globals().is_empty());
        assert!(m.functions().is_empty());
        assert!(m.declarations().is_empty());
        assert!(m.string_pool().is_empty());
        assert!(m.inline_asm_blocks.is_empty());
        assert!(m.type_metadata.is_empty());
    }

    #[test]
    fn test_add_and_get_global() {
        let mut m = IrModule::new("test.c".to_string());
        let var = GlobalVariable::new("x".to_string(), IrType::I32, Some(Constant::Integer(42)));
        m.add_global(var);

        assert_eq!(m.globals().len(), 1);
        let g = m.get_global("x").expect("global 'x' should exist");
        assert_eq!(g.name, "x");
        assert!(matches!(g.initializer, Some(Constant::Integer(42))));
        assert!(m.get_global("nonexistent").is_none());
    }

    #[test]
    fn test_add_and_get_function() {
        let mut m = IrModule::new("test.c".to_string());
        let func = make_test_function("main", true);
        m.add_function(func);

        assert_eq!(m.functions().len(), 1);
        let f = m
            .get_function("main")
            .expect("function 'main' should exist");
        assert_eq!(f.name, "main");
        assert!(f.is_definition);
        assert!(m.get_function("missing").is_none());
    }

    #[test]
    fn test_get_function_mut() {
        let mut m = IrModule::new("test.c".to_string());
        let func = make_test_function("foo", true);
        m.add_function(func);

        let f = m
            .get_function_mut("foo")
            .expect("function 'foo' should exist");
        f.is_noreturn = true;

        let f2 = m.get_function("foo").unwrap();
        assert!(f2.is_noreturn);
    }

    #[test]
    fn test_intern_string_deduplication() {
        let mut m = IrModule::new("test.c".to_string());
        let id1 = m.intern_string(b"hello\0".to_vec());
        let id2 = m.intern_string(b"hello\0".to_vec());
        assert_eq!(id1, id2, "identical strings should yield same ID");
        assert_eq!(m.string_pool().len(), 1, "pool should have one entry");

        let id3 = m.intern_string(b"world\0".to_vec());
        assert_ne!(id1, id3, "different strings should yield different IDs");
        assert_eq!(m.string_pool().len(), 2, "pool should have two entries");
    }

    #[test]
    fn test_intern_string_labels() {
        let mut m = IrModule::new("test.c".to_string());
        m.intern_string(b"first\0".to_vec());
        m.intern_string(b"second\0".to_vec());

        assert_eq!(m.string_pool()[0].label, ".L.str.0");
        assert_eq!(m.string_pool()[1].label, ".L.str.1");
    }

    #[test]
    fn test_add_declaration() {
        let mut m = IrModule::new("test.c".to_string());
        let decl = FunctionDeclaration::new("printf".to_string(), IrType::I32, vec![IrType::Ptr]);
        m.add_declaration(decl);

        assert_eq!(m.declarations().len(), 1);
        assert_eq!(m.declarations()[0].name, "printf");
        assert!(m.declarations()[0].is_variadic == false);
    }

    #[test]
    fn test_add_inline_asm() {
        let mut m = IrModule::new("test.c".to_string());
        m.add_inline_asm(".section .init".to_string());
        m.add_inline_asm(".previous".to_string());

        assert_eq!(m.inline_asm_blocks.len(), 2);
        assert_eq!(m.inline_asm_blocks[0].assembly, ".section .init");
        assert_eq!(m.inline_asm_blocks[1].assembly, ".previous");
    }

    #[test]
    fn test_linkage_variants() {
        assert_eq!(Linkage::default(), Linkage::External);
        assert_ne!(Linkage::Internal, Linkage::External);
        assert_ne!(Linkage::Weak, Linkage::Common);

        // Display formatting
        assert_eq!(format!("{}", Linkage::External), "external");
        assert_eq!(format!("{}", Linkage::Internal), "internal");
        assert_eq!(format!("{}", Linkage::Weak), "weak");
        assert_eq!(format!("{}", Linkage::Common), "common");
    }

    #[test]
    fn test_visibility_variants() {
        assert_eq!(Visibility::default(), Visibility::Default);
        assert_ne!(Visibility::Hidden, Visibility::Protected);

        // Display formatting
        assert_eq!(format!("{}", Visibility::Default), "default");
        assert_eq!(format!("{}", Visibility::Hidden), "hidden");
        assert_eq!(format!("{}", Visibility::Protected), "protected");
    }

    #[test]
    fn test_constant_variants() {
        // Verify all variant constructions compile and display
        let constants: Vec<Constant> = vec![
            Constant::Integer(42),
            Constant::Float(3.14),
            Constant::LongDouble([0u8; 10]),
            Constant::String(b"test\0".to_vec()),
            Constant::ZeroInit,
            Constant::Struct(vec![Constant::Integer(1), Constant::Integer(2)]),
            Constant::Array(vec![Constant::Integer(10), Constant::Integer(20)]),
            Constant::GlobalRef("other_var".to_string()),
            Constant::Null,
            Constant::Undefined,
        ];

        for c in &constants {
            let _ = format!("{}", c);
        }

        assert_eq!(constants.len(), 10);
    }

    #[test]
    fn test_global_variable_defaults() {
        let var = GlobalVariable::new("g".to_string(), IrType::I64, None);
        assert_eq!(var.name, "g");
        assert!(var.is_definition);
        assert_eq!(var.linkage, Linkage::External);
        assert!(!var.is_thread_local);
        assert!(var.section.is_none());
        assert!(var.alignment.is_none());
        assert!(!var.is_constant);
    }

    #[test]
    fn test_function_declaration_defaults() {
        let decl = FunctionDeclaration::new("malloc".to_string(), IrType::Ptr, vec![IrType::I64]);
        assert_eq!(decl.name, "malloc");
        assert!(!decl.is_variadic);
        assert_eq!(decl.linkage, Linkage::External);
        assert_eq!(decl.visibility, Visibility::Default);
    }

    #[test]
    fn test_type_metadata() {
        let meta = TypeMetadata::new("struct point".to_string(), IrType::I64);
        assert_eq!(meta.name, "struct point");
        let _ = format!("{}", meta);
    }

    #[test]
    fn test_module_display() {
        let mut m = IrModule::new("display_test.c".to_string());
        m.add_global(GlobalVariable::new(
            "counter".to_string(),
            IrType::I32,
            Some(Constant::Integer(0)),
        ));
        m.intern_string(b"hello\0".to_vec());
        m.add_inline_asm(".text".to_string());
        let decl = FunctionDeclaration::new("puts".to_string(), IrType::I32, vec![IrType::Ptr]);
        m.add_declaration(decl);

        let output = format!("{}", m);
        assert!(output.contains("ModuleID = 'display_test.c'"));
        assert!(output.contains("@counter"));
        assert!(output.contains(".L.str.0"));
        assert!(output.contains("module asm"));
        assert!(output.contains("declare"));
    }

    #[test]
    fn test_module_debug() {
        let m = IrModule::new("debug_test.c".to_string());
        let debug_output = format!("{:?}", m);
        assert!(debug_output.contains("IrModule"));
        assert!(debug_output.contains("debug_test.c"));
    }

    #[test]
    fn test_multiple_globals_ordering() {
        let mut m = IrModule::new("test.c".to_string());
        m.add_global(GlobalVariable::new("a".to_string(), IrType::I32, None));
        m.add_global(GlobalVariable::new("b".to_string(), IrType::I64, None));
        m.add_global(GlobalVariable::new("c".to_string(), IrType::Ptr, None));

        let globals = m.globals();
        assert_eq!(globals.len(), 3);
        assert_eq!(globals[0].name, "a");
        assert_eq!(globals[1].name, "b");
        assert_eq!(globals[2].name, "c");
    }

    #[test]
    fn test_multiple_functions_ordering() {
        let mut m = IrModule::new("test.c".to_string());
        m.add_function(make_test_function("first", true));
        m.add_function(make_test_function("second", true));
        m.add_function(make_test_function("third", false));

        let funcs = m.functions();
        assert_eq!(funcs.len(), 3);
        assert_eq!(funcs[0].name, "first");
        assert_eq!(funcs[1].name, "second");
        assert_eq!(funcs[2].name, "third");
        assert!(!funcs[2].is_definition);
    }

    #[test]
    fn test_intern_string_empty_bytes() {
        let mut m = IrModule::new("test.c".to_string());
        let id = m.intern_string(Vec::new());
        assert_eq!(id, 0);
        assert_eq!(m.string_pool()[0].bytes.len(), 0);
    }

    #[test]
    fn test_intern_string_non_utf8() {
        let mut m = IrModule::new("test.c".to_string());
        // Non-UTF-8 bytes (PUA-decoded content from Linux kernel)
        let bytes = vec![0x80, 0xFF, 0x00];
        let id = m.intern_string(bytes.clone());
        assert_eq!(m.string_pool()[id as usize].bytes, bytes);
    }

    #[test]
    fn test_global_variable_display() {
        let mut var = GlobalVariable::new(
            "my_const".to_string(),
            IrType::I32,
            Some(Constant::Integer(100)),
        );
        var.is_constant = true;
        var.alignment = Some(16);
        var.section = Some(".mydata".to_string());

        let output = format!("{}", var);
        assert!(output.contains("@my_const"));
        assert!(output.contains("constant"));
        assert!(output.contains("100"));
        assert!(output.contains("align 16"));
        assert!(output.contains("section \".mydata\""));
    }

    #[test]
    fn test_function_declaration_variadic_display() {
        let mut decl =
            FunctionDeclaration::new("printf".to_string(), IrType::I32, vec![IrType::Ptr]);
        decl.is_variadic = true;

        let output = format!("{}", decl);
        assert!(output.contains("declare"));
        assert!(output.contains("@printf"));
        assert!(output.contains("..."));
    }

    #[test]
    fn test_global_replace_in_map() {
        let mut m = IrModule::new("test.c".to_string());
        m.add_global(GlobalVariable::new(
            "x".to_string(),
            IrType::I32,
            Some(Constant::Integer(1)),
        ));
        m.add_global(GlobalVariable::new(
            "x".to_string(),
            IrType::I64,
            Some(Constant::Integer(2)),
        ));

        // Deduplication: same-name globals merge into a single entry.
        // The latest definition replaces the earlier one.
        let g = m.get_global("x").unwrap();
        assert!(matches!(g.initializer, Some(Constant::Integer(2))));
        assert_eq!(m.globals().len(), 1); // deduplicated to single entry
    }
}
