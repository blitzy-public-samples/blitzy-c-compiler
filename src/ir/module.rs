//! IR module representation for the BCC compiler.
//!
//! This module defines [`IrModule`], the top-level container for all
//! intermediate representation produced during the AST-to-IR lowering
//! phase (Phase 6). An `IrModule` corresponds to a single C translation
//! unit and aggregates:
//!
//! - **Global variables** ([`GlobalVariable`]) — with type, initializer,
//!   linkage, alignment, and optional custom section placement.
//! - **Function definitions** ([`IrFunction`]) — functions with bodies
//!   (basic blocks, instructions, SSA values).
//! - **External function declarations** ([`FunctionDecl`]) — prototypes
//!   for functions defined in other translation units (e.g., `printf`).
//! - **String literal pool** ([`StringLiteral`]) — deduplicated string
//!   constants destined for the `.rodata` section.
//! - **Module-level inline assembly** ([`InlineAsmBlock`]) — top-level
//!   `asm()` blocks that are emitted verbatim into the assembly output.
//!
//! # Pipeline Position
//!
//! ```text
//! Frontend (AST)
//!       │
//!       ▼
//!   Phase 6: IR Lowering ──► IrModule (this struct)
//!       │
//!       ▼
//!   Phase 7: mem2reg (SSA construction)
//!       │
//!       ▼
//!   Phase 8: Optimisation passes
//!       │
//!       ▼
//!   Phase 9: Phi elimination
//!       │
//!       ▼
//!   Phase 10: Code generation ──► ELF binary
//! ```
//!
//! The `IrModule` is the data structure passed from the middle-end to the
//! backend. It carries the target architecture information needed by the
//! code generator to select the correct `ArchCodegen` implementation.
//!
//! # Global Initializers
//!
//! Global variable initial values are represented by the [`Constant`] enum,
//! which supports integer, float, string reference, null pointer, array,
//! struct, global address reference, and zero-initializer constants. The
//! backend uses the constant kind to decide section placement:
//!
//! - `Constant::Zero` → `.bss` (zero-initialized, no file space)
//! - Other constants with `is_const = true` → `.rodata`
//! - Other constants with `is_const = false` → `.data`
//!
//! # Re-exports
//!
//! [`Linkage`] and [`CallingConvention`] are defined in
//! [`crate::ir::function`] and re-exported from this module so that
//! consumers dealing with module-level constructs (global variables,
//! function declarations) can import them from a single location.

use std::fmt;

use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::types::IrType;

// Re-export Linkage and CallingConvention from the function module.
// These enums are used pervasively in module-level data structures
// (GlobalVariable.linkage, FunctionDecl.linkage, FunctionDecl.calling_convention)
// and are re-exported here for ergonomic access by consumers that work
// primarily with module-level IR constructs.
pub use crate::ir::function::{CallingConvention, Linkage};

// ---------------------------------------------------------------------------
// Constant — compile-time constant values for global initializers
// ---------------------------------------------------------------------------

/// A compile-time constant value used as a global variable initializer.
///
/// Constants form a recursive tree structure: array and struct constants
/// contain nested `Constant` children representing element/field values.
/// This tree is consumed by the ELF writer to emit initialized data
/// into `.data` or `.rodata` sections.
///
/// # Variants
///
/// | Variant      | Section Placement | Description                              |
/// |--------------|-------------------|------------------------------------------|
/// | `Int`        | `.data`/`.rodata` | Integer constant with explicit type      |
/// | `Float`      | `.data`/`.rodata` | Floating-point constant                  |
/// | `String`     | `.rodata`         | Reference to string literal pool entry   |
/// | `Null`       | `.data`/`.rodata` | Typed null pointer (all zero bytes)      |
/// | `Array`      | `.data`/`.rodata` | Array of element constants               |
/// | `Struct`     | `.data`/`.rodata` | Struct with field constants              |
/// | `GlobalRef`  | `.data`           | Address of another global or function    |
/// | `Zero`       | `.bss`            | Zero-initialized (occupies no file space)|
///
/// # Examples
///
/// ```ignore
/// // Integer constant: int x = 42;
/// let c = Constant::Int { value: 42, ty: IrType::I32 };
///
/// // Null pointer: void *p = 0;
/// let null = Constant::Null { ty: IrType::Ptr };
///
/// // Array: int arr[] = {1, 2, 3};
/// let arr = Constant::Array {
///     elements: vec![
///         Constant::Int { value: 1, ty: IrType::I32 },
///         Constant::Int { value: 2, ty: IrType::I32 },
///         Constant::Int { value: 3, ty: IrType::I32 },
///     ],
///     ty: IrType::Array { element: Box::new(IrType::I32), count: 3 },
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    /// Integer constant with a 128-bit value and explicit IR type.
    ///
    /// The `value` field uses `i128` to accommodate all C integer widths
    /// from `_Bool` (1 bit) through `__int128` (128 bits). The `ty` field
    /// specifies the actual width (`I1`, `I8`, `I16`, `I32`, `I64`, `I128`).
    Int {
        /// The integer value, sign-extended to 128 bits.
        value: i128,
        /// The IR type determining the storage width.
        ty: IrType,
    },

    /// IEEE 754 floating-point constant.
    ///
    /// The `value` is stored as `f64`, which can exactly represent all
    /// `float` values and most `double` values. For `long double` (F80),
    /// the f64 approximation is used during constant folding; the backend
    /// emits the exact bit pattern from the source literal.
    Float {
        /// The floating-point value.
        value: f64,
        /// The IR type (`F32`, `F64`, or `F80`).
        ty: IrType,
    },

    /// Reference to a string literal in the module's string pool.
    ///
    /// The `id` indexes into [`IrModule::string_literals`]. At link time,
    /// this becomes a pointer to the string's location in `.rodata`.
    String {
        /// Index into the module's string literal pool.
        id: u32,
    },

    /// Typed null pointer constant.
    ///
    /// Emitted as all-zero bytes with the width determined by the target's
    /// pointer size. The `ty` field is always [`IrType::Ptr`] but is kept
    /// explicit for consistency with other variants.
    Null {
        /// The pointer type (always `IrType::Ptr`).
        ty: IrType,
    },

    /// Array constant — a fixed-size sequence of element constants.
    ///
    /// The `elements` vector must have exactly as many entries as the
    /// array's `count` in the `ty` field. Each element must have the
    /// array's element type.
    Array {
        /// Element constants in index order.
        elements: Vec<Constant>,
        /// The array type (`IrType::Array { element, count }`).
        ty: IrType,
    },

    /// Struct constant — an ordered sequence of field constants.
    ///
    /// The `fields` vector must have exactly as many entries as the
    /// struct's field list in the `ty` field. Each field constant must
    /// match the corresponding field type.
    Struct {
        /// Field constants in declaration order.
        fields: Vec<Constant>,
        /// The struct type (`IrType::Struct { fields, packed }`).
        ty: IrType,
    },

    /// Reference to another global variable or function by name.
    ///
    /// At link time, this resolves to the address of the referenced
    /// symbol. Used for global pointer initializers such as:
    /// ```c
    /// int x;
    /// int *p = &x;       // GlobalRef { name: "x" }
    /// void (*fp)(void) = foo;  // GlobalRef { name: "foo" }
    /// ```
    GlobalRef {
        /// The symbol name of the referenced global or function.
        name: String,
    },

    /// Zero-initialized constant — no explicit data in the object file.
    ///
    /// Globals initialized with `Zero` are placed in the `.bss` section,
    /// which occupies no space in the ELF file. The runtime loader
    /// guarantees that `.bss` memory is zeroed before program start.
    ///
    /// Corresponds to `= {0}` or implicit zero-initialization in C.
    Zero {
        /// The type determining the size of the zero-filled region.
        ty: IrType,
    },
}

impl Constant {
    /// Returns the IR type of this constant value.
    ///
    /// For most variants the type is stored directly. For [`String`] and
    /// [`GlobalRef`], the type is [`IrType::Ptr`] since they represent
    /// addresses.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let c = Constant::Int { value: 42, ty: IrType::I32 };
    /// assert_eq!(c.get_type(), IrType::I32);
    ///
    /// let s = Constant::String { id: 0 };
    /// assert_eq!(s.get_type(), IrType::Ptr);
    /// ```
    pub fn get_type(&self) -> IrType {
        match self {
            Constant::Int { ty, .. } => ty.clone(),
            Constant::Float { ty, .. } => ty.clone(),
            Constant::String { .. } => IrType::Ptr,
            Constant::Null { ty } => ty.clone(),
            Constant::Array { ty, .. } => ty.clone(),
            Constant::Struct { ty, .. } => ty.clone(),
            Constant::GlobalRef { .. } => IrType::Ptr,
            Constant::Zero { ty } => ty.clone(),
        }
    }

    /// Returns `true` if this constant is all-zero (either an explicit
    /// [`Zero`](Constant::Zero) or a recursively zero aggregate).
    ///
    /// Used by the backend to determine `.bss` eligibility: a global
    /// variable whose initializer is entirely zero can be placed in `.bss`
    /// to avoid wasting file space in the ELF image.
    pub fn is_zero(&self) -> bool {
        match self {
            Constant::Zero { .. } => true,
            Constant::Int { value, .. } => *value == 0,
            Constant::Float { value, .. } => *value == 0.0 && !value.is_sign_negative(),
            Constant::Null { .. } => true,
            Constant::Array { elements, .. } => elements.iter().all(|e| e.is_zero()),
            Constant::Struct { fields, .. } => fields.iter().all(|f| f.is_zero()),
            Constant::String { .. } | Constant::GlobalRef { .. } => false,
        }
    }

    /// Returns `true` if this constant is a simple scalar (integer, float,
    /// null, or zero with a scalar type) rather than an aggregate.
    pub fn is_scalar(&self) -> bool {
        match self {
            Constant::Int { .. }
            | Constant::Float { .. }
            | Constant::String { .. }
            | Constant::Null { .. }
            | Constant::GlobalRef { .. } => true,
            Constant::Zero { ty } => ty.is_scalar(),
            Constant::Array { .. } | Constant::Struct { .. } => false,
        }
    }

    /// Creates an integer zero constant of the given type.
    ///
    /// Convenience constructor for the common case of zero-initialized
    /// integer globals and struct padding fields.
    #[inline]
    pub fn int_zero(ty: IrType) -> Self {
        Constant::Int { value: 0, ty }
    }

    /// Creates a null pointer constant.
    #[inline]
    pub fn null_ptr() -> Self {
        Constant::Null { ty: IrType::Ptr }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Int { value, ty } => write!(f, "{} {}", ty, value),
            Constant::Float { value, ty } => write!(f, "{} {:.17e}", ty, value),
            Constant::String { id } => write!(f, "string @.str.{}", id),
            Constant::Null { ty } => write!(f, "{} null", ty),
            Constant::Array { elements, ty } => {
                write!(f, "{} [", ty)?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Constant::Struct { fields, ty } => {
                write!(f, "{} {{", ty)?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, "}}")
            }
            Constant::GlobalRef { name } => write!(f, "ptr @{}", name),
            Constant::Zero { ty } => write!(f, "{} zeroinitializer", ty),
        }
    }
}

// ---------------------------------------------------------------------------
// StringLiteral — string constant for .rodata
// ---------------------------------------------------------------------------

/// A string literal constant stored in the module's string pool.
///
/// String literals are collected during IR lowering and emitted into the
/// `.rodata` section of the output ELF file. Each literal has a unique
/// `id` used to reference it from [`Constant::String`] initializers and
/// from IR instructions that load string addresses.
///
/// # Encoding
///
/// The `data` field contains the raw byte sequence of the string,
/// including any embedded null bytes and escape sequences that were
/// resolved during lexing. Non-UTF-8 bytes survive the pipeline via
/// PUA encoding (Section 0.7.9): bytes 0x80–0xFF are mapped to
/// U+E080–U+E0FF during source reading and decoded back to exact
/// bytes during code generation, ensuring byte-exact fidelity.
///
/// # Null Termination
///
/// C string literals are null-terminated by language specification.
/// The `null_terminated` flag indicates whether `data` includes the
/// trailing `\0` byte. When `true`, the backend emits `data` as-is.
/// When `false` (rare — used for raw byte arrays), no extra null is
/// appended.
///
/// # Examples
///
/// ```ignore
/// // "Hello, World!\n" → 15 bytes including \n and \0
/// let lit = StringLiteral {
///     id: 0,
///     data: b"Hello, World!\n\0".to_vec(),
///     null_terminated: true,
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StringLiteral {
    /// Unique identifier for this string literal within the module.
    /// Used by [`Constant::String`] to reference pool entries.
    pub id: u32,

    /// Raw byte content of the string literal.
    ///
    /// For null-terminated strings, this includes the trailing `\0`.
    /// For wide/unicode strings, this contains the encoded byte sequence
    /// (UTF-8 for `u8""`, UTF-16LE for `u""`, UTF-32LE for `U""`).
    pub data: Vec<u8>,

    /// Whether this string literal is null-terminated.
    ///
    /// `true` for standard C string literals (`"..."`).
    /// `false` for raw byte data (e.g., initializer byte arrays).
    pub null_terminated: bool,
}

impl StringLiteral {
    /// Returns the size of this string literal in bytes, including any
    /// null terminator present in `data`.
    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the string literal contains no data bytes
    /// (an empty string `""` still has a null terminator if
    /// `null_terminated` is `true`, so this checks `data.is_empty()`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl fmt::Display for StringLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@.str.{} = ", self.id)?;
        if self.null_terminated {
            write!(f, "c\"")?;
        } else {
            write!(f, "[\"")?;
        }
        // Print data as escaped ASCII for readability.
        for &byte in &self.data {
            match byte {
                b'\n' => write!(f, "\\n")?,
                b'\r' => write!(f, "\\r")?,
                b'\t' => write!(f, "\\t")?,
                b'\0' => write!(f, "\\00")?,
                b'\\' => write!(f, "\\\\")?,
                b'"' => write!(f, "\\\"")?,
                0x20..=0x7E => write!(f, "{}", byte as char)?,
                _ => write!(f, "\\{:02x}", byte)?,
            }
        }
        if self.null_terminated {
            write!(f, "\"")
        } else {
            write!(f, "\"]")
        }
    }
}

// ---------------------------------------------------------------------------
// InlineAsmBlock — module-level inline assembly
// ---------------------------------------------------------------------------

/// A module-level inline assembly block.
///
/// Module-level `asm()` statements (also written as `__asm__()`) are
/// emitted verbatim into the assembly output at the top level, outside
/// any function. They are used in the Linux kernel for:
///
/// - Section directives (`.pushsection` / `.popsection`)
/// - Symbol definitions and annotations
/// - Architecture-specific assembly sequences
/// - Alternative instruction patching tables
///
/// # Extended Fields
///
/// While module-level asm typically consists only of a template string,
/// the extended fields (`constraints`, `operands`, `clobbers`,
/// `goto_labels`) are provided for completeness and to support GCC's
/// extended asm syntax at the top level. Most module-level asm blocks
/// will have empty constraint/operand/clobber/label vectors.
///
/// # Examples
///
/// ```ignore
/// // Simple module-level asm
/// let asm = InlineAsmBlock {
///     template: ".pushsection .note.GNU-stack,\"\",@progbits\n.popsection".into(),
///     constraints: vec![],
///     operands: vec![],
///     clobbers: vec![],
///     is_volatile: true,
///     has_side_effects: true,
///     goto_labels: vec![],
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlineAsmBlock {
    /// The assembly template string in AT&T syntax.
    ///
    /// May contain multiple lines separated by `\n` and assembler
    /// directives such as `.section`, `.global`, `.align`, etc.
    pub template: String,

    /// Operand constraints in GCC inline asm format.
    ///
    /// Each string is a constraint specification like `"=r"`, `"r"`,
    /// `"m"`, `"i"`, etc. For module-level asm this is typically empty.
    pub constraints: Vec<String>,

    /// Operand expressions as string representations.
    ///
    /// For module-level asm, operands reference global symbols by name.
    /// The vector is parallel to `constraints` — `operands[i]` is
    /// constrained by `constraints[i]`.
    pub operands: Vec<String>,

    /// Clobber list — registers and memory clobbered by the asm block.
    ///
    /// Common clobbers include `"memory"`, `"cc"`, and specific register
    /// names. For module-level asm this is typically empty.
    pub clobbers: Vec<String>,

    /// Whether this asm block is marked `volatile`.
    ///
    /// Volatile asm blocks must not be optimized away or reordered
    /// relative to other volatile operations. Module-level asm is
    /// implicitly volatile.
    pub is_volatile: bool,

    /// Whether this asm block has observable side effects.
    ///
    /// When `true`, the optimizer must not eliminate this block even
    /// if its outputs appear unused. Module-level asm inherently has
    /// side effects (it modifies the assembly output).
    pub has_side_effects: bool,

    /// Labels that `asm goto` can jump to.
    ///
    /// For module-level asm this is typically empty. Present for
    /// consistency with function-level inline asm representation.
    pub goto_labels: Vec<String>,
}

impl InlineAsmBlock {
    /// Creates a simple module-level asm block with just a template string.
    ///
    /// The block is marked volatile with side effects (the default for
    /// module-level asm), and all extended fields are empty.
    ///
    /// # Arguments
    ///
    /// * `template` — the assembly template in AT&T syntax.
    pub fn new(template: String) -> Self {
        InlineAsmBlock {
            template,
            constraints: Vec::new(),
            operands: Vec::new(),
            clobbers: Vec::new(),
            is_volatile: true,
            has_side_effects: true,
            goto_labels: Vec::new(),
        }
    }

    /// Returns `true` if this asm block has no constraints, operands,
    /// clobbers, or goto labels — i.e., it is a simple template-only block.
    pub fn is_simple(&self) -> bool {
        self.constraints.is_empty()
            && self.operands.is_empty()
            && self.clobbers.is_empty()
            && self.goto_labels.is_empty()
    }
}

impl fmt::Display for InlineAsmBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module asm")?;
        if self.is_volatile {
            write!(f, " volatile")?;
        }
        write!(f, " \"{}\"", self.template.replace('\n', "\\n"))?;
        if !self.constraints.is_empty() {
            write!(f, " : {}", self.constraints.join(", "))?;
        }
        if !self.clobbers.is_empty() {
            write!(f, " clobber({})", self.clobbers.join(", "))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GlobalVariable — global variable definition
// ---------------------------------------------------------------------------

/// A global variable definition in the IR module.
///
/// Global variables reside in the module's symbol table and are emitted
/// into ELF sections based on their properties:
///
/// | Property                  | Section     | Notes                               |
/// |---------------------------|-------------|-------------------------------------|
/// | `is_const = true`         | `.rodata`   | Read-only data                      |
/// | `initializer = Some(Zero)`| `.bss`      | Zero-initialized, no file space     |
/// | `initializer = None`      | `.bss`      | Extern declaration (tentative def)  |
/// | `is_thread_local = true`  | `.tdata`/`.tbss` | Thread-local storage           |
/// | `section = Some(name)`    | `name`      | Custom section override             |
/// | Otherwise                 | `.data`     | Read-write initialized data         |
///
/// # Linkage and Visibility
///
/// The `linkage` field controls ELF symbol binding:
/// - [`Linkage::External`] → `STB_GLOBAL` — visible across translation units
/// - [`Linkage::Internal`] → `STB_LOCAL` — file-scoped (`static` in C)
/// - [`Linkage::Weak`] → `STB_WEAK` — overridable by strong definitions
/// - [`Linkage::Common`] → common symbol, merged by linker (tentative defs)
///
/// # Examples
///
/// ```ignore
/// // static const int MAGIC = 0xDEAD;
/// let global = GlobalVariable {
///     name: "MAGIC".into(),
///     ty: IrType::I32,
///     initializer: Some(Constant::Int { value: 0xDEAD, ty: IrType::I32 }),
///     is_const: true,
///     linkage: Linkage::Internal,
///     alignment: 4,
///     section: None,
///     is_thread_local: false,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct GlobalVariable {
    /// Symbol name of the global variable.
    pub name: String,

    /// IR type of the global variable, determining its size and alignment
    /// in the output section.
    pub ty: IrType,

    /// Optional initial value. `None` for `extern` declarations without
    /// a definition in this translation unit. `Some(Constant::Zero { .. })`
    /// for implicit zero-initialization (`.bss` placement).
    pub initializer: Option<Constant>,

    /// Whether this global is `const`-qualified.
    ///
    /// When `true`, the variable is placed in `.rodata` (or the custom
    /// section if `section` is set). Writes to const globals are undefined
    /// behaviour in C.
    pub is_const: bool,

    /// Symbol linkage type controlling visibility and binding in the
    /// ELF symbol table.
    pub linkage: Linkage,

    /// Required alignment in bytes.
    ///
    /// Defaults to the natural alignment of `ty` for the target
    /// architecture. May be overridden by `__attribute__((aligned(N)))`.
    /// Must be a power of two.
    pub alignment: u32,

    /// Custom section name from `__attribute__((section("...")))`.
    ///
    /// When `Some`, the global is placed in the named section instead of
    /// the default `.data`/`.rodata`/`.bss`. Used extensively in the Linux
    /// kernel for `__initdata`, `__read_mostly`, and similar annotations.
    pub section: Option<String>,

    /// Whether this global uses thread-local storage (`_Thread_local`).
    ///
    /// Thread-local globals are placed in `.tdata` (initialized) or
    /// `.tbss` (zero-initialized) and accessed via TLS mechanisms
    /// (e.g., `fs` segment on x86-64).
    pub is_thread_local: bool,
}

impl GlobalVariable {
    /// Creates a new global variable with external linkage and default settings.
    ///
    /// # Arguments
    ///
    /// * `name` — symbol name
    /// * `ty` — IR type
    /// * `alignment` — alignment in bytes (must be power of two)
    pub fn new(name: String, ty: IrType, alignment: u32) -> Self {
        GlobalVariable {
            name,
            ty,
            initializer: None,
            is_const: false,
            linkage: Linkage::External,
            alignment,
            section: None,
            is_thread_local: false,
        }
    }

    /// Returns `true` if this global should be placed in the `.bss` section.
    ///
    /// A global qualifies for `.bss` if it has no initializer (extern
    /// tentative definition), or its initializer is entirely zero-valued,
    /// and it is not `const` (const zero globals go in `.rodata`).
    pub fn is_bss(&self) -> bool {
        if self.is_const {
            return false;
        }
        match &self.initializer {
            None => true,
            Some(c) => c.is_zero(),
        }
    }

    /// Returns `true` if this global has common linkage.
    ///
    /// Common symbols are used for tentative definitions of uninitialized
    /// globals. The linker merges multiple common symbols, choosing the
    /// largest size.
    #[inline]
    pub fn is_common(&self) -> bool {
        self.linkage == Linkage::Common
    }

    /// Returns `true` if this global is externally visible.
    ///
    /// External and weak symbols are visible across translation units;
    /// internal symbols are file-local.
    #[inline]
    pub fn is_externally_visible(&self) -> bool {
        matches!(self.linkage, Linkage::External | Linkage::Weak | Linkage::Common)
    }

    /// Returns the section name this global should be placed in.
    ///
    /// If a custom section is specified via `__attribute__((section(...)))`,
    /// that name is returned. Otherwise, the default section is determined
    /// by the global's properties (const, bss, thread-local).
    pub fn effective_section(&self) -> &str {
        if let Some(ref section) = self.section {
            return section;
        }
        if self.is_thread_local {
            if self.is_bss() {
                return ".tbss";
            }
            return ".tdata";
        }
        if self.is_const {
            return ".rodata";
        }
        if self.is_bss() {
            return ".bss";
        }
        ".data"
    }
}

impl fmt::Display for GlobalVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{} = {} ", self.name, self.linkage)?;
        if self.is_const {
            write!(f, "constant ")?;
        } else {
            write!(f, "global ")?;
        }
        write!(f, "{}", self.ty)?;
        if let Some(ref init) = self.initializer {
            write!(f, " {}", init)?;
        }
        write!(f, ", align {}", self.alignment)?;
        if self.is_thread_local {
            write!(f, ", thread_local")?;
        }
        if let Some(ref section) = self.section {
            write!(f, ", section \"{}\"", section)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FunctionDecl — external function declaration (no body)
// ---------------------------------------------------------------------------

/// An external function declaration — a function prototype for a symbol
/// defined in another translation unit.
///
/// Function declarations inform the IR lowering and code generation
/// phases about the signature of externally-defined functions (e.g.,
/// `printf`, `memcpy`, `__stack_chk_fail`). They are emitted as
/// undefined symbols in the ELF object file, resolved by the linker.
///
/// # Distinction from IrFunction
///
/// - [`IrFunction`] represents a function **definition** (has a body
///   with basic blocks and instructions).
/// - [`FunctionDecl`] represents a function **declaration** (prototype
///   only, no body). It is a lighter-weight structure that stores only
///   the information needed for call-site code generation.
///
/// # Examples
///
/// ```ignore
/// // extern int printf(const char *fmt, ...);
/// let printf_decl = FunctionDecl {
///     name: "printf".into(),
///     return_type: IrType::I32,
///     param_types: vec![IrType::Ptr],
///     is_variadic: true,
///     linkage: Linkage::External,
///     calling_convention: CallingConvention::C,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct FunctionDecl {
    /// Function symbol name.
    pub name: String,

    /// Return type of the function.
    pub return_type: IrType,

    /// Parameter types in declaration order.
    ///
    /// For variadic functions, this contains the types of the fixed
    /// parameters only. Variadic arguments are indicated by `is_variadic`.
    pub param_types: Vec<IrType>,

    /// Whether this function accepts variadic arguments (`...`).
    pub is_variadic: bool,

    /// Symbol linkage type.
    ///
    /// Declarations are typically [`Linkage::External`] (referencing
    /// a symbol defined elsewhere) or [`Linkage::Weak`] (overridable
    /// default implementation).
    pub linkage: Linkage,

    /// Calling convention for this function.
    ///
    /// Most C functions use [`CallingConvention::C`] (the platform ABI
    /// default). Internal helpers may use [`CallingConvention::Fast`]
    /// for reduced call overhead, [`CallingConvention::Cold`] for
    /// rarely-invoked error paths, or [`CallingConvention::Custom`]
    /// for architecture-specific conventions.
    pub calling_convention: CallingConvention,
}

impl FunctionDecl {
    /// Creates a new external function declaration with the C calling
    /// convention and external linkage.
    ///
    /// # Arguments
    ///
    /// * `name` — function symbol name
    /// * `return_type` — return type
    /// * `param_types` — parameter types in order
    /// * `is_variadic` — whether the function is variadic
    pub fn new(
        name: String,
        return_type: IrType,
        param_types: Vec<IrType>,
        is_variadic: bool,
    ) -> Self {
        FunctionDecl {
            name,
            return_type,
            param_types,
            is_variadic,
            linkage: Linkage::External,
            calling_convention: CallingConvention::C,
        }
    }

    /// Returns the function type as an [`IrType::Function`].
    ///
    /// Constructs the function type from the declaration's return type,
    /// parameter types, and variadic flag.
    pub fn function_type(&self) -> IrType {
        IrType::Function {
            return_type: Box::new(self.return_type.clone()),
            param_types: self.param_types.clone(),
            is_variadic: self.is_variadic,
        }
    }

    /// Returns the number of fixed (non-variadic) parameters.
    #[inline]
    pub fn param_count(&self) -> usize {
        self.param_types.len()
    }
}

impl fmt::Display for FunctionDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "declare {} {} @{}(", self.linkage, self.return_type, self.name)?;
        for (i, ty) in self.param_types.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", ty)?;
        }
        if self.is_variadic {
            if !self.param_types.is_empty() {
                write!(f, ", ")?;
            }
            write!(f, "...")?;
        }
        write!(f, ")")?;
        if self.calling_convention != CallingConvention::C {
            write!(f, " {}", self.calling_convention)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IrModule — top-level IR container
// ---------------------------------------------------------------------------

/// Top-level IR container representing a single C translation unit.
///
/// `IrModule` is the central data structure passed from the IR lowering
/// phase (Phase 6) through optimisation passes (Phase 8) to the code
/// generation backend (Phase 10). It aggregates all compilation-unit-level
/// entities: global variables, function definitions, external declarations,
/// string literals, and module-level inline assembly.
///
/// # Target Architecture
///
/// The `target` field carries the compilation target ([`Target::X86_64`],
/// [`Target::I686`], [`Target::AArch64`], or [`Target::RiscV64`]) through
/// the entire pipeline. Optimisation passes may query the target for
/// architecture-specific decisions (e.g., pointer width for constant
/// folding), and the code generation backend uses it to dispatch to the
/// correct `ArchCodegen` implementation.
///
/// # String Literal Pool
///
/// String literals are deduplicated at the module level and referenced by
/// ID from [`Constant::String`] values. The pool is populated during IR
/// lowering and emitted into the `.rodata` section by the backend.
///
/// # Lifecycle
///
/// ```text
/// IrModule::new("hello.c", Target::X86_64)
///   │
///   ├── Phase 6: IR lowering populates globals, functions, declarations
///   │
///   ├── Phase 7: mem2reg transforms function bodies (SSA construction)
///   │
///   ├── Phase 8: Optimisation passes transform function bodies
///   │
///   ├── Phase 9: Phi elimination prepares for register allocation
///   │
///   └── Phase 10: Code generation consumes the module, produces ELF
/// ```
///
/// # Examples
///
/// ```ignore
/// use crate::common::target::Target;
/// use crate::ir::module::{IrModule, GlobalVariable, FunctionDecl, Constant};
/// use crate::ir::types::IrType;
///
/// let mut module = IrModule::new("hello.c".into(), Target::X86_64);
///
/// // Add a string literal
/// let str_id = module.add_string_literal(b"Hello, World!\n\0".to_vec());
///
/// // Add a global variable
/// let hello_str = GlobalVariable {
///     name: "hello_msg".into(),
///     ty: IrType::Ptr,
///     initializer: Some(Constant::String { id: str_id }),
///     is_const: true,
///     linkage: Linkage::Internal,
///     alignment: 8,
///     section: None,
///     is_thread_local: false,
/// };
/// module.add_global(hello_str);
///
/// // Add an external function declaration
/// let printf_decl = FunctionDecl::new(
///     "printf".into(),
///     IrType::I32,
///     vec![IrType::Ptr],
///     true,
/// );
/// module.add_declaration(printf_decl);
/// ```
#[derive(Clone, Debug)]
pub struct IrModule {
    /// Module name — typically the source file name (e.g., `"hello.c"`).
    ///
    /// Used for diagnostic messages, DWARF compilation unit identification,
    /// and ELF section naming.
    pub name: String,

    /// Target architecture for this compilation unit.
    ///
    /// Carried through the pipeline for architecture-specific decisions
    /// in optimisation passes and code generation dispatch.
    pub target: Target,

    /// Global variable definitions.
    ///
    /// Includes both initialized globals (with `initializer = Some(...)`)
    /// and tentative definitions (with `initializer = None` or
    /// `Constant::Zero`).
    pub globals: Vec<GlobalVariable>,

    /// Function definitions with bodies (basic blocks and instructions).
    ///
    /// Each [`IrFunction`] contains the complete control flow graph and
    /// SSA value registry for one C function.
    pub functions: Vec<IrFunction>,

    /// External function declarations (prototypes without bodies).
    ///
    /// These generate undefined symbol references in the ELF object,
    /// resolved by the linker against other translation units or libraries.
    pub declarations: Vec<FunctionDecl>,

    /// String literal pool for the `.rodata` section.
    ///
    /// Strings are added via [`add_string_literal()`] and referenced by
    /// their `id` from [`Constant::String`] values. The pool is append-only
    /// — string IDs are stable once assigned.
    pub string_literals: Vec<StringLiteral>,

    /// Module-level inline assembly blocks.
    ///
    /// Emitted verbatim into the assembly output in the order they appear.
    /// Used by the Linux kernel for section annotations, symbol definitions,
    /// and architecture-specific assembly sequences.
    pub inline_asm_blocks: Vec<InlineAsmBlock>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl IrModule {
    /// Creates a new, empty IR module for the given source file and target.
    ///
    /// All collections (globals, functions, declarations, string literals,
    /// inline assembly blocks) are initialized empty.
    ///
    /// # Arguments
    ///
    /// * `name` — module name (typically the source file name)
    /// * `target` — target architecture for code generation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let module = IrModule::new("kernel/main.c".into(), Target::RiscV64);
    /// assert_eq!(module.name, "kernel/main.c");
    /// assert_eq!(module.target, Target::RiscV64);
    /// assert!(module.globals.is_empty());
    /// assert!(module.functions.is_empty());
    /// ```
    pub fn new(name: String, target: Target) -> Self {
        IrModule {
            name,
            target,
            globals: Vec::new(),
            functions: Vec::new(),
            declarations: Vec::new(),
            string_literals: Vec::new(),
            inline_asm_blocks: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Modification operations
// ---------------------------------------------------------------------------

impl IrModule {
    /// Adds a global variable definition to the module.
    ///
    /// The global is appended to the end of the `globals` vector.
    /// No duplicate-name checking is performed — the caller (IR lowering)
    /// is responsible for ensuring unique names within the module.
    ///
    /// # Arguments
    ///
    /// * `global` — the global variable to add.
    #[inline]
    pub fn add_global(&mut self, global: GlobalVariable) {
        self.globals.push(global);
    }

    /// Adds a function definition to the module.
    ///
    /// The function is appended to the end of the `functions` vector.
    /// The function must have `is_definition = true` and contain at least
    /// an entry basic block.
    ///
    /// # Arguments
    ///
    /// * `func` — the function definition to add.
    #[inline]
    pub fn add_function(&mut self, func: IrFunction) {
        self.functions.push(func);
    }

    /// Adds an external function declaration to the module.
    ///
    /// The declaration is appended to the end of the `declarations` vector.
    /// Duplicate declarations (same name) are permitted — the linker
    /// resolves them to the same symbol.
    ///
    /// # Arguments
    ///
    /// * `decl` — the function declaration to add.
    #[inline]
    pub fn add_declaration(&mut self, decl: FunctionDecl) {
        self.declarations.push(decl);
    }

    /// Adds a string literal to the module's string pool and returns its ID.
    ///
    /// The returned ID can be used in [`Constant::String`] to reference
    /// this literal from global variable initializers or IR instructions.
    /// String IDs are assigned sequentially starting from 0.
    ///
    /// The string is stored as-is with null termination assumed by default.
    /// For C string literals the data should include the trailing `\0` byte.
    ///
    /// # Arguments
    ///
    /// * `data` — raw byte content of the string literal.
    ///
    /// # Returns
    ///
    /// The unique `u32` identifier for the newly added string literal.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut module = IrModule::new("test.c".into(), Target::X86_64);
    /// let id = module.add_string_literal(b"hello\0".to_vec());
    /// assert_eq!(id, 0);
    /// let id2 = module.add_string_literal(b"world\0".to_vec());
    /// assert_eq!(id2, 1);
    /// ```
    pub fn add_string_literal(&mut self, data: Vec<u8>) -> u32 {
        let id = self.string_literals.len() as u32;
        self.string_literals.push(StringLiteral {
            id,
            data,
            null_terminated: true,
        });
        id
    }

    /// Adds a string literal with explicit null-termination control.
    ///
    /// Unlike [`add_string_literal()`], this method allows the caller to
    /// specify whether the data is null-terminated. Useful for byte arrays
    /// or strings from wide/unicode literal contexts.
    ///
    /// # Arguments
    ///
    /// * `data` — raw byte content.
    /// * `null_terminated` — whether the data is null-terminated.
    ///
    /// # Returns
    ///
    /// The unique `u32` identifier for the newly added string literal.
    pub fn add_string_literal_with_termination(
        &mut self,
        data: Vec<u8>,
        null_terminated: bool,
    ) -> u32 {
        let id = self.string_literals.len() as u32;
        self.string_literals.push(StringLiteral {
            id,
            data,
            null_terminated,
        });
        id
    }

    /// Adds a module-level inline assembly block.
    ///
    /// The block is appended to the end of the `inline_asm_blocks` vector
    /// and will be emitted in order during code generation.
    ///
    /// # Arguments
    ///
    /// * `asm_block` — the inline assembly block to add.
    #[inline]
    pub fn add_inline_asm(&mut self, asm_block: InlineAsmBlock) {
        self.inline_asm_blocks.push(asm_block);
    }
}

// ---------------------------------------------------------------------------
// Query operations
// ---------------------------------------------------------------------------

impl IrModule {
    /// Looks up a global variable by name.
    ///
    /// Returns `None` if no global with the given name exists in the module.
    /// The search is linear; for modules with many globals, consider using
    /// an `FxHashMap`-based index for O(1) lookups.
    ///
    /// # Arguments
    ///
    /// * `name` — the symbol name to search for.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let module = /* ... */;
    /// if let Some(global) = module.find_global("errno") {
    ///     assert!(global.is_thread_local);
    /// }
    /// ```
    pub fn find_global(&self, name: &str) -> Option<&GlobalVariable> {
        self.globals.iter().find(|g| g.name == name)
    }

    /// Looks up a global variable by name, returning a mutable reference.
    ///
    /// Allows modifying the global's initializer, linkage, or section
    /// after initial creation (e.g., during optimization or linker script
    /// processing).
    pub fn find_global_mut(&mut self, name: &str) -> Option<&mut GlobalVariable> {
        self.globals.iter_mut().find(|g| g.name == name)
    }

    /// Looks up a function definition by name.
    ///
    /// Searches only function **definitions** (with bodies), not external
    /// declarations. Returns `None` if no function with the given name
    /// exists.
    ///
    /// # Arguments
    ///
    /// * `name` — the function symbol name to search for.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(func) = module.find_function("main") {
    ///     println!("main returns: {}", func.return_type);
    ///     println!("main has {} params", func.params.len());
    /// }
    /// ```
    pub fn find_function(&self, name: &str) -> Option<&IrFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Looks up a function definition by name, returning a mutable reference.
    ///
    /// Allows modifying the function body (e.g., during optimization passes
    /// that transform instructions or restructure the CFG).
    pub fn find_function_mut(&mut self, name: &str) -> Option<&mut IrFunction> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    /// Looks up an external function declaration by name.
    ///
    /// Searches only external declarations (prototypes without bodies).
    /// Returns `None` if no declaration with the given name exists.
    pub fn find_declaration(&self, name: &str) -> Option<&FunctionDecl> {
        self.declarations.iter().find(|d| d.name == name)
    }

    /// Looks up a string literal by its ID.
    ///
    /// Returns `None` if the ID does not correspond to any string in the pool.
    pub fn find_string_literal(&self, id: u32) -> Option<&StringLiteral> {
        self.string_literals.iter().find(|s| s.id == id)
    }

    /// Returns the total number of symbols (globals + functions + declarations)
    /// in the module.
    pub fn symbol_count(&self) -> usize {
        self.globals.len() + self.functions.len() + self.declarations.len()
    }

    /// Returns `true` if the module contains no definitions — no globals,
    /// functions, or inline assembly.
    pub fn is_empty(&self) -> bool {
        self.globals.is_empty()
            && self.functions.is_empty()
            && self.declarations.is_empty()
            && self.string_literals.is_empty()
            && self.inline_asm_blocks.is_empty()
    }

    /// Returns an iterator over all function names defined in this module.
    pub fn function_names(&self) -> impl Iterator<Item = &str> {
        self.functions.iter().map(|f| f.name.as_str())
    }

    /// Returns an iterator over all global variable names in this module.
    pub fn global_names(&self) -> impl Iterator<Item = &str> {
        self.globals.iter().map(|g| g.name.as_str())
    }

    /// Checks whether a function or global with the given name exists.
    ///
    /// Searches globals, function definitions, and external declarations.
    pub fn has_symbol(&self, name: &str) -> bool {
        self.find_global(name).is_some()
            || self.find_function(name).is_some()
            || self.find_declaration(name).is_some()
    }
}

// ---------------------------------------------------------------------------
// Display — human-readable IR dump
// ---------------------------------------------------------------------------

impl fmt::Display for IrModule {
    /// Renders the module as a human-readable IR text representation.
    ///
    /// The output format resembles LLVM IR for familiarity:
    ///
    /// ```text
    /// ; ModuleID = 'hello.c'
    /// ; Target: x86-64
    ///
    /// @.str.0 = c"Hello, World!\n\00"
    ///
    /// @counter = external global i32, align 4
    ///
    /// declare external i32 @printf(ptr, ...)
    ///
    /// define external i32 @main() {
    ///   ; ... function body ...
    /// }
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Module header
        writeln!(f, "; ModuleID = '{}'", self.name)?;
        writeln!(f, "; Target: {}", self.target)?;
        writeln!(f)?;

        // String literal pool
        for lit in &self.string_literals {
            writeln!(f, "{}", lit)?;
        }
        if !self.string_literals.is_empty() {
            writeln!(f)?;
        }

        // Global variables
        for global in &self.globals {
            writeln!(f, "{}", global)?;
        }
        if !self.globals.is_empty() {
            writeln!(f)?;
        }

        // External declarations
        for decl in &self.declarations {
            writeln!(f, "{}", decl)?;
        }
        if !self.declarations.is_empty() {
            writeln!(f)?;
        }

        // Function definitions
        for (i, func) in self.functions.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            // Function header: linkage, calling convention, return type, name, params
            write!(f, "define {} ", func.linkage)?;
            if func.calling_convention != CallingConvention::C {
                write!(f, "{} ", func.calling_convention)?;
            }
            write!(f, "{} @{}(", func.return_type, func.name)?;
            for (j, param) in func.params.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param)?;
            }
            if func.is_variadic {
                if !func.params.is_empty() {
                    write!(f, ", ")?;
                }
                write!(f, "...")?;
            }
            writeln!(f, ") {{")?;
            // Emit basic blocks
            for block in func.blocks() {
                writeln!(f, "  {}:", block.id)?;
                for inst in block.instructions() {
                    writeln!(f, "    {}", inst)?;
                }
            }
            writeln!(f, "}}")?;
        }

        // Module-level inline assembly
        if !self.inline_asm_blocks.is_empty() {
            writeln!(f)?;
            for asm_block in &self.inline_asm_blocks {
                writeln!(f, "{}", asm_block)?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a new module is created empty with the correct name and target.
    #[test]
    fn test_new_module() {
        let module = IrModule::new("test.c".into(), Target::X86_64);
        assert_eq!(module.name, "test.c");
        assert_eq!(module.target, Target::X86_64);
        assert!(module.globals.is_empty());
        assert!(module.functions.is_empty());
        assert!(module.declarations.is_empty());
        assert!(module.string_literals.is_empty());
        assert!(module.inline_asm_blocks.is_empty());
        assert!(module.is_empty());
        assert_eq!(module.symbol_count(), 0);
    }

    /// Verify module creation for each target architecture.
    #[test]
    fn test_all_targets() {
        let targets = [
            (Target::X86_64, "x86_64"),
            (Target::I686, "i686"),
            (Target::AArch64, "aarch64"),
            (Target::RiscV64, "riscv64"),
        ];
        for (target, name) in &targets {
            let module = IrModule::new(name.to_string(), *target);
            assert_eq!(module.target, *target);
        }
    }

    /// Verify add_global and find_global operations.
    #[test]
    fn test_add_and_find_global() {
        let mut module = IrModule::new("test.c".into(), Target::X86_64);

        let global = GlobalVariable {
            name: "counter".into(),
            ty: IrType::I32,
            initializer: Some(Constant::Int {
                value: 0,
                ty: IrType::I32,
            }),
            is_const: false,
            linkage: Linkage::External,
            alignment: 4,
            section: None,
            is_thread_local: false,
        };
        module.add_global(global);

        assert_eq!(module.globals.len(), 1);
        let found = module.find_global("counter");
        assert!(found.is_some());
        assert_eq!(found.unwrap().alignment, 4);

        assert!(module.find_global("nonexistent").is_none());
    }

    /// Verify add_declaration and find_declaration.
    #[test]
    fn test_add_and_find_declaration() {
        let mut module = IrModule::new("test.c".into(), Target::AArch64);

        let decl = FunctionDecl::new(
            "printf".into(),
            IrType::I32,
            vec![IrType::Ptr],
            true,
        );
        module.add_declaration(decl);

        assert_eq!(module.declarations.len(), 1);
        let found = module.find_declaration("printf");
        assert!(found.is_some());
        let d = found.unwrap();
        assert!(d.is_variadic);
        assert_eq!(d.calling_convention, CallingConvention::C);
        assert_eq!(d.linkage, Linkage::External);
    }

    /// Verify string literal pool management.
    #[test]
    fn test_string_literal_pool() {
        let mut module = IrModule::new("test.c".into(), Target::I686);

        let id0 = module.add_string_literal(b"hello\0".to_vec());
        assert_eq!(id0, 0);

        let id1 = module.add_string_literal(b"world\0".to_vec());
        assert_eq!(id1, 1);

        assert_eq!(module.string_literals.len(), 2);

        let lit0 = module.find_string_literal(0);
        assert!(lit0.is_some());
        assert_eq!(lit0.unwrap().data, b"hello\0");
        assert!(lit0.unwrap().null_terminated);

        let lit1 = module.find_string_literal(1);
        assert!(lit1.is_some());
        assert_eq!(lit1.unwrap().data, b"world\0");
    }

    /// Verify non-null-terminated string literal.
    #[test]
    fn test_string_literal_raw() {
        let mut module = IrModule::new("test.c".into(), Target::RiscV64);

        let id = module.add_string_literal_with_termination(
            vec![0x80, 0xFF, 0x00, 0x42],
            false,
        );
        assert_eq!(id, 0);

        let lit = module.find_string_literal(id).unwrap();
        assert!(!lit.null_terminated);
        assert_eq!(lit.size(), 4);
    }

    /// Verify Constant type inference and zero detection.
    #[test]
    fn test_constant_get_type() {
        let c_int = Constant::Int {
            value: 42,
            ty: IrType::I32,
        };
        assert_eq!(c_int.get_type(), IrType::I32);
        assert!(!c_int.is_zero());

        let c_zero = Constant::Int {
            value: 0,
            ty: IrType::I64,
        };
        assert_eq!(c_zero.get_type(), IrType::I64);
        assert!(c_zero.is_zero());

        let c_float = Constant::Float {
            value: 3.14,
            ty: IrType::F64,
        };
        assert_eq!(c_float.get_type(), IrType::F64);
        assert!(!c_float.is_zero());

        let c_str = Constant::String { id: 0 };
        assert_eq!(c_str.get_type(), IrType::Ptr);
        assert!(!c_str.is_zero());

        let c_null = Constant::Null { ty: IrType::Ptr };
        assert_eq!(c_null.get_type(), IrType::Ptr);
        assert!(c_null.is_zero());

        let c_ref = Constant::GlobalRef {
            name: "foo".into(),
        };
        assert_eq!(c_ref.get_type(), IrType::Ptr);
        assert!(!c_ref.is_zero());

        let c_bss = Constant::Zero { ty: IrType::I32 };
        assert_eq!(c_bss.get_type(), IrType::I32);
        assert!(c_bss.is_zero());
    }

    /// Verify aggregate constant zero detection.
    #[test]
    fn test_aggregate_zero_detection() {
        let zero_array = Constant::Array {
            elements: vec![
                Constant::Int {
                    value: 0,
                    ty: IrType::I32,
                },
                Constant::Int {
                    value: 0,
                    ty: IrType::I32,
                },
            ],
            ty: IrType::Array {
                element: Box::new(IrType::I32),
                count: 2,
            },
        };
        assert!(zero_array.is_zero());

        let nonzero_array = Constant::Array {
            elements: vec![
                Constant::Int {
                    value: 0,
                    ty: IrType::I32,
                },
                Constant::Int {
                    value: 1,
                    ty: IrType::I32,
                },
            ],
            ty: IrType::Array {
                element: Box::new(IrType::I32),
                count: 2,
            },
        };
        assert!(!nonzero_array.is_zero());

        let zero_struct = Constant::Struct {
            fields: vec![
                Constant::Null { ty: IrType::Ptr },
                Constant::Zero { ty: IrType::I32 },
            ],
            ty: IrType::Struct {
                fields: vec![IrType::Ptr, IrType::I32],
                packed: false,
            },
        };
        assert!(zero_struct.is_zero());
    }

    /// Verify GlobalVariable section placement logic.
    #[test]
    fn test_global_section_placement() {
        // Const global → .rodata
        let const_global = GlobalVariable {
            name: "PI".into(),
            ty: IrType::F64,
            initializer: Some(Constant::Float {
                value: 3.14159265358979,
                ty: IrType::F64,
            }),
            is_const: true,
            linkage: Linkage::Internal,
            alignment: 8,
            section: None,
            is_thread_local: false,
        };
        assert_eq!(const_global.effective_section(), ".rodata");
        assert!(!const_global.is_bss());

        // Zero-initialized non-const → .bss
        let bss_global = GlobalVariable {
            name: "buffer".into(),
            ty: IrType::Array {
                element: Box::new(IrType::I8),
                count: 4096,
            },
            initializer: Some(Constant::Zero {
                ty: IrType::Array {
                    element: Box::new(IrType::I8),
                    count: 4096,
                },
            }),
            is_const: false,
            linkage: Linkage::External,
            alignment: 16,
            section: None,
            is_thread_local: false,
        };
        assert_eq!(bss_global.effective_section(), ".bss");
        assert!(bss_global.is_bss());

        // Thread-local with initializer → .tdata
        let tls_global = GlobalVariable {
            name: "errno".into(),
            ty: IrType::I32,
            initializer: Some(Constant::Int {
                value: 0,
                ty: IrType::I32,
            }),
            is_const: false,
            linkage: Linkage::External,
            alignment: 4,
            section: None,
            is_thread_local: true,
        };
        // Zero value + thread_local → .tbss
        assert_eq!(tls_global.effective_section(), ".tbss");

        // Custom section overrides default
        let custom_section = GlobalVariable {
            name: "init_data".into(),
            ty: IrType::I32,
            initializer: Some(Constant::Int {
                value: 1,
                ty: IrType::I32,
            }),
            is_const: false,
            linkage: Linkage::External,
            alignment: 4,
            section: Some(".init.data".into()),
            is_thread_local: false,
        };
        assert_eq!(custom_section.effective_section(), ".init.data");
    }

    /// Verify GlobalVariable linkage helpers.
    #[test]
    fn test_global_linkage_helpers() {
        let external = GlobalVariable::new("ext".into(), IrType::I32, 4);
        assert!(external.is_externally_visible());
        assert!(!external.is_common());

        let mut internal = GlobalVariable::new("int".into(), IrType::I32, 4);
        internal.linkage = Linkage::Internal;
        assert!(!internal.is_externally_visible());

        let mut weak = GlobalVariable::new("wk".into(), IrType::I32, 4);
        weak.linkage = Linkage::Weak;
        assert!(weak.is_externally_visible());

        let mut common = GlobalVariable::new("cm".into(), IrType::I32, 4);
        common.linkage = Linkage::Common;
        assert!(common.is_common());
        assert!(common.is_externally_visible());
    }

    /// Verify FunctionDecl construction and function_type().
    #[test]
    fn test_function_decl() {
        let decl = FunctionDecl::new(
            "memcpy".into(),
            IrType::Ptr,
            vec![IrType::Ptr, IrType::Ptr, IrType::I64],
            false,
        );
        assert_eq!(decl.name, "memcpy");
        assert_eq!(decl.param_count(), 3);
        assert!(!decl.is_variadic);

        let ft = decl.function_type();
        match ft {
            IrType::Function {
                return_type,
                param_types,
                is_variadic,
            } => {
                assert_eq!(*return_type, IrType::Ptr);
                assert_eq!(param_types.len(), 3);
                assert!(!is_variadic);
            }
            _ => panic!("Expected Function type"),
        }
    }

    /// Verify InlineAsmBlock creation.
    #[test]
    fn test_inline_asm_block() {
        let asm = InlineAsmBlock::new(
            ".pushsection .note.GNU-stack,\"\",@progbits\n.popsection".into(),
        );
        assert!(asm.is_volatile);
        assert!(asm.has_side_effects);
        assert!(asm.is_simple());
        assert!(asm.constraints.is_empty());
        assert!(asm.operands.is_empty());
        assert!(asm.clobbers.is_empty());
        assert!(asm.goto_labels.is_empty());
    }

    /// Verify InlineAsmBlock with extended fields.
    #[test]
    fn test_inline_asm_block_extended() {
        let asm = InlineAsmBlock {
            template: "movq %0, %%rax".into(),
            constraints: vec!["=r".into()],
            operands: vec!["some_global".into()],
            clobbers: vec!["memory".into(), "cc".into()],
            is_volatile: true,
            has_side_effects: true,
            goto_labels: vec!["error_label".into()],
        };
        assert!(!asm.is_simple());
        assert_eq!(asm.constraints.len(), 1);
        assert_eq!(asm.clobbers.len(), 2);
        assert_eq!(asm.goto_labels.len(), 1);
    }

    /// Verify Constant scalar check.
    #[test]
    fn test_constant_is_scalar() {
        assert!(Constant::Int {
            value: 1,
            ty: IrType::I32
        }
        .is_scalar());

        assert!(Constant::Float {
            value: 1.0,
            ty: IrType::F64
        }
        .is_scalar());

        assert!(Constant::String { id: 0 }.is_scalar());

        assert!(Constant::Null { ty: IrType::Ptr }.is_scalar());

        assert!(Constant::GlobalRef {
            name: "x".into()
        }
        .is_scalar());

        assert!(Constant::Zero { ty: IrType::I32 }.is_scalar());

        assert!(!Constant::Zero {
            ty: IrType::Array {
                element: Box::new(IrType::I32),
                count: 4
            }
        }
        .is_scalar());

        assert!(!Constant::Array {
            elements: vec![],
            ty: IrType::Array {
                element: Box::new(IrType::I8),
                count: 0
            }
        }
        .is_scalar());
    }

    /// Verify convenience constructors.
    #[test]
    fn test_constant_constructors() {
        let zero = Constant::int_zero(IrType::I32);
        assert!(zero.is_zero());
        assert_eq!(zero.get_type(), IrType::I32);

        let null = Constant::null_ptr();
        assert!(null.is_zero());
        assert_eq!(null.get_type(), IrType::Ptr);
    }

    /// Verify has_symbol searches all collections.
    #[test]
    fn test_has_symbol() {
        let mut module = IrModule::new("test.c".into(), Target::X86_64);

        let global = GlobalVariable::new("my_global".into(), IrType::I32, 4);
        module.add_global(global);

        let decl = FunctionDecl::new("my_func".into(), IrType::Void, vec![], false);
        module.add_declaration(decl);

        assert!(module.has_symbol("my_global"));
        assert!(module.has_symbol("my_func"));
        assert!(!module.has_symbol("nonexistent"));
        assert_eq!(module.symbol_count(), 2);
        assert!(!module.is_empty());
    }

    /// Verify Display output does not panic for a populated module.
    #[test]
    fn test_module_display() {
        let mut module = IrModule::new("display_test.c".into(), Target::RiscV64);

        // Add a string literal
        let str_id = module.add_string_literal(b"test\0".to_vec());

        // Add a global variable
        let global = GlobalVariable {
            name: "greeting".into(),
            ty: IrType::Ptr,
            initializer: Some(Constant::String { id: str_id }),
            is_const: true,
            linkage: Linkage::Internal,
            alignment: 8,
            section: None,
            is_thread_local: false,
        };
        module.add_global(global);

        // Add a declaration
        let decl = FunctionDecl::new("puts".into(), IrType::I32, vec![IrType::Ptr], false);
        module.add_declaration(decl);

        // Add an inline asm block
        let asm = InlineAsmBlock::new(".section .rodata".into());
        module.add_inline_asm(asm);

        // Render and verify it doesn't panic
        let output = format!("{}", module);
        assert!(output.contains("ModuleID = 'display_test.c'"));
        assert!(output.contains("@greeting"));
        assert!(output.contains("declare"));
        assert!(output.contains("puts"));
        assert!(output.contains("module asm"));
    }

    /// Verify Display output for Constant variants.
    #[test]
    fn test_constant_display() {
        let int_c = Constant::Int {
            value: 42,
            ty: IrType::I32,
        };
        let s = format!("{}", int_c);
        assert!(s.contains("42"));

        let null_c = Constant::Null { ty: IrType::Ptr };
        let s = format!("{}", null_c);
        assert!(s.contains("null"));

        let zero_c = Constant::Zero { ty: IrType::I64 };
        let s = format!("{}", zero_c);
        assert!(s.contains("zeroinitializer"));

        let ref_c = Constant::GlobalRef {
            name: "my_sym".into(),
        };
        let s = format!("{}", ref_c);
        assert!(s.contains("@my_sym"));
    }

    /// Verify StringLiteral Display with escape sequences.
    #[test]
    fn test_string_literal_display() {
        let lit = StringLiteral {
            id: 0,
            data: b"hello\n\0".to_vec(),
            null_terminated: true,
        };
        let s = format!("{}", lit);
        assert!(s.contains("@.str.0"));
        assert!(s.contains("hello"));
        assert!(s.contains("\\n"));
    }

    /// Verify GlobalVariable Display.
    #[test]
    fn test_global_display() {
        let g = GlobalVariable {
            name: "x".into(),
            ty: IrType::I32,
            initializer: Some(Constant::Int {
                value: 10,
                ty: IrType::I32,
            }),
            is_const: false,
            linkage: Linkage::External,
            alignment: 4,
            section: None,
            is_thread_local: false,
        };
        let s = format!("{}", g);
        assert!(s.contains("@x"));
        assert!(s.contains("global"));
        assert!(s.contains("align 4"));
    }

    /// Verify FunctionDecl Display for variadic function.
    #[test]
    fn test_function_decl_display() {
        let decl = FunctionDecl::new(
            "printf".into(),
            IrType::I32,
            vec![IrType::Ptr],
            true,
        );
        let s = format!("{}", decl);
        assert!(s.contains("declare"));
        assert!(s.contains("@printf"));
        assert!(s.contains("..."));
    }

    /// Verify FunctionDecl with non-default calling convention.
    #[test]
    fn test_function_decl_custom_cc() {
        let mut decl = FunctionDecl::new(
            "fast_helper".into(),
            IrType::Void,
            vec![IrType::I64],
            false,
        );
        decl.calling_convention = CallingConvention::Fast;
        let s = format!("{}", decl);
        assert!(s.contains("fastcc"));

        let mut cold_decl = FunctionDecl::new(
            "error_handler".into(),
            IrType::Void,
            vec![],
            false,
        );
        cold_decl.calling_convention = CallingConvention::Cold;
        let s = format!("{}", cold_decl);
        assert!(s.contains("coldcc"));

        let mut custom_decl = FunctionDecl::new(
            "arch_specific".into(),
            IrType::Void,
            vec![],
            false,
        );
        custom_decl.calling_convention = CallingConvention::Custom;
        let s = format!("{}", custom_decl);
        assert!(s.contains("customcc"));
    }
}
