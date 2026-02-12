// src/frontend/parser/ast.rs
//
// Complete Abstract Syntax Tree (AST) node definitions for the BCC C11 parser.
//
// This module defines the entire AST hierarchy used throughout the compiler
// pipeline — from the parser through semantic analysis, IR lowering, and
// diagnostics. Every node carries a `Span` for precise source location
// tracking, enabling accurate error messages and debug information.
//
// The AST covers:
//   - C11 language constructs (declarations, statements, expressions)
//   - GCC extensions (statement expressions, typeof, computed goto, case
//     ranges, label address, conditional omission, inline assembly)
//   - Full declaration syntax (declarators, parameter lists, initializers,
//     designators, bitfields)
//   - Attributes (__attribute__((...))) with all argument forms
//   - Inline assembly (asm/asm volatile/asm goto with operands and clobbers)
//
// Design decisions:
//   - All enum variants use named fields (not tuple variants) for clarity
//     except where single-value wrapping is clearer (e.g., Initializer::Expression)
//   - Box<T> is used for recursive types to ensure finite struct size
//   - Vec<u8> is used for string/char literal values to preserve PUA-encoded
//     non-UTF-8 bytes with byte-exact fidelity
//   - Symbol (4-byte interned handle) is used for all identifiers, labels,
//     and names — zero-cost comparison via integer equality
//   - All types derive Clone and Debug; PartialEq is included for testing
//
// Integration points:
//   - Produced by: src/frontend/parser/ (recursive descent parser)
//   - Consumed by: src/frontend/sema/ (semantic analysis)
//   - Consumed by: src/ir/lowering/ (AST-to-IR lowering)
//   - Span type flows through to: src/common/diagnostics.rs

// ---------------------------------------------------------------------------
// Re-exports from the lexer token module
// ---------------------------------------------------------------------------
// These re-exports provide a single import point for downstream modules
// (parser, sema, IR lowering) so they can write:
//   use crate::frontend::parser::ast::{Span, IntegerSuffix, ...};
// rather than reaching into the lexer module directly.

/// Source location span — re-exported from `crate::frontend::lexer::token`.
/// Canonical definition lives in `crate::common::diagnostics`.
pub use crate::frontend::lexer::token::Span;

/// Integer literal suffix — re-exported from `crate::frontend::lexer::token`.
pub use crate::frontend::lexer::token::IntegerSuffix;

/// Floating-point literal suffix — re-exported from `crate::frontend::lexer::token`.
pub use crate::frontend::lexer::token::FloatSuffix;

/// String literal encoding prefix — re-exported from `crate::frontend::lexer::token`.
pub use crate::frontend::lexer::token::StringPrefix;

/// Character literal encoding prefix — re-exported from `crate::frontend::lexer::token`.
pub use crate::frontend::lexer::token::CharPrefix;

// ---------------------------------------------------------------------------
// Internal imports
// ---------------------------------------------------------------------------

use crate::common::string_interner::Symbol;

// ===========================================================================
// Type Qualifiers
// ===========================================================================

/// C11 type qualifiers (§6.7.3): `const`, `volatile`, `restrict`, `_Atomic`.
///
/// Stored as a bitfield-style struct for compact representation and fast
/// merging. The `merge` method OR-combines qualifier sets, used when
/// accumulating qualifiers from declaration specifier lists.
///
/// # Default
///
/// All qualifiers are `false` by default, representing an unqualified type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeQualifiers {
    /// `const` qualifier — object may not be modified after initialization.
    pub is_const: bool,
    /// `volatile` qualifier — accesses are not optimizable side-effect-free.
    pub is_volatile: bool,
    /// `restrict` qualifier (pointer only) — no aliasing through other pointers.
    pub is_restrict: bool,
    /// `_Atomic` qualifier — atomic access semantics (C11 §6.7.2.4).
    pub is_atomic: bool,
}

impl Default for TypeQualifiers {
    fn default() -> Self {
        TypeQualifiers {
            is_const: false,
            is_volatile: false,
            is_restrict: false,
            is_atomic: false,
        }
    }
}

impl TypeQualifiers {
    /// Merges another set of qualifiers into this one by OR-combining each flag.
    ///
    /// This is used when accumulating qualifiers from a declaration specifier
    /// list where qualifiers may appear in any order and may repeat.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut q = TypeQualifiers::default();
    /// q.merge(&TypeQualifiers { is_const: true, ..Default::default() });
    /// q.merge(&TypeQualifiers { is_volatile: true, ..Default::default() });
    /// assert!(q.is_const && q.is_volatile);
    /// ```
    pub fn merge(&mut self, other: &TypeQualifiers) {
        self.is_const |= other.is_const;
        self.is_volatile |= other.is_volatile;
        self.is_restrict |= other.is_restrict;
        self.is_atomic |= other.is_atomic;
    }

    /// Returns `true` if no qualifiers are set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.is_const && !self.is_volatile && !self.is_restrict && !self.is_atomic
    }
}

// ===========================================================================
// Function Specifiers
// ===========================================================================

/// C11 function specifiers: `inline` and `_Noreturn`.
///
/// These appear in declaration specifiers and apply only to function
/// declarations/definitions. `inline` hints the compiler to inline the
/// function body. `_Noreturn` indicates the function never returns to
/// its caller (e.g., `exit()`, `abort()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionSpecifiers {
    /// `inline` or `__inline__` — hint for inlining.
    pub is_inline: bool,
    /// `_Noreturn` or `__attribute__((noreturn))` — function never returns.
    pub is_noreturn: bool,
}

impl Default for FunctionSpecifiers {
    fn default() -> Self {
        FunctionSpecifiers {
            is_inline: false,
            is_noreturn: false,
        }
    }
}

// ===========================================================================
// Storage Class
// ===========================================================================

/// C11 storage-class specifiers (§6.7.1).
///
/// At most one storage-class specifier may appear in a declaration's specifiers
/// (except `_Thread_local` which can combine with `static` or `extern`).
/// The parser enforces this constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// `auto` — automatic storage duration (block scope default).
    Auto,
    /// `register` — automatic storage with optimization hint.
    Register,
    /// `static` — static storage duration, internal linkage at file scope.
    Static,
    /// `extern` — external linkage, declaration without definition.
    Extern,
    /// `typedef` — not a real storage class, but parsed in the same position.
    Typedef,
    /// `_Thread_local` — thread-local storage duration (C11 §6.7.1).
    ThreadLocal,
}

// ===========================================================================
// Alignas Specifier
// ===========================================================================

/// C11 `_Alignas` specifier (§6.7.5).
///
/// Specifies an alignment requirement for an object or type. Can be either
/// a type name (alignment of that type) or a constant expression (explicit
/// byte alignment value, must be a power of two).
#[derive(Clone, Debug, PartialEq)]
pub enum AlignasSpecifier {
    /// `_Alignas(type-name)` — align to the alignment of the given type.
    TypeName(Box<TypeName>),
    /// `_Alignas(constant-expression)` — align to the given byte count.
    Expression(Box<Expression>),
}

// ===========================================================================
// Operators
// ===========================================================================

/// Binary operators for `Expression::BinaryOp`.
///
/// Covers all C11 binary operators including arithmetic, bitwise, logical,
/// relational, equality, assignment, and compound assignment operators.
/// Operator precedence is handled by the parser's expression parsing logic
/// (precedence climbing); the AST records only the already-structured tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    // --- Arithmetic ---
    /// `+` addition
    Add,
    /// `-` subtraction
    Sub,
    /// `*` multiplication
    Mul,
    /// `/` division
    Div,
    /// `%` modulus (remainder)
    Mod,

    // --- Bitwise ---
    /// `&` bitwise AND
    BitAnd,
    /// `|` bitwise OR
    BitOr,
    /// `^` bitwise XOR
    BitXor,
    /// `<<` left shift
    Shl,
    /// `>>` right shift
    Shr,

    // --- Logical ---
    /// `&&` logical AND
    LogAnd,
    /// `||` logical OR
    LogOr,

    // --- Relational and equality ---
    /// `==` equal
    Eq,
    /// `!=` not equal
    Ne,
    /// `<` less than
    Lt,
    /// `>` greater than
    Gt,
    /// `<=` less than or equal
    Le,
    /// `>=` greater than or equal
    Ge,

    // --- Assignment ---
    /// `=` simple assignment
    Assign,
    /// `+=` addition assignment
    AddAssign,
    /// `-=` subtraction assignment
    SubAssign,
    /// `*=` multiplication assignment
    MulAssign,
    /// `/=` division assignment
    DivAssign,
    /// `%=` modulus assignment
    ModAssign,
    /// `&=` bitwise AND assignment
    BitAndAssign,
    /// `|=` bitwise OR assignment
    BitOrAssign,
    /// `^=` bitwise XOR assignment
    BitXorAssign,
    /// `<<=` left shift assignment
    ShlAssign,
    /// `>>=` right shift assignment
    ShrAssign,
}

/// Unary operators for `Expression::UnaryOp`.
///
/// Covers all C11 prefix and postfix unary operators. The `is_postfix` field
/// on `Expression::UnaryOp` distinguishes prefix from postfix forms for
/// increment and decrement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOperator {
    /// `+` unary plus
    Plus,
    /// `-` unary negation
    Neg,
    /// `~` bitwise complement
    BitNot,
    /// `!` logical NOT
    LogNot,
    /// `++` prefix increment
    PreInc,
    /// `--` prefix decrement
    PreDec,
    /// `++` postfix increment
    PostInc,
    /// `--` postfix decrement
    PostDec,
    /// `&` address-of (also represented as `Expression::AddressOf`)
    AddressOf,
    /// `*` dereference/indirection (also represented as `Expression::Dereference`)
    Deref,
}

// ===========================================================================
// Attribute AST
// ===========================================================================

/// A GCC `__attribute__((...))` attribute.
///
/// Attributes are attached to declarations, declarators, statements (labels),
/// and types. The parser extracts the attribute name and argument list; the
/// semantic analyzer validates and interprets them.
///
/// # Examples
///
/// - `__attribute__((aligned(16)))` → name="aligned", args=[Integer(16)]
/// - `__attribute__((section(".init")))` → name="section", args=[String(b".init")]
/// - `__attribute__((format(printf, 1, 2)))` → name="format", args=[Identifier("printf"), Integer(1), Integer(2)]
/// - `__attribute__((unused))` → name="unused", args=[]
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    /// Attribute name (e.g., "aligned", "packed", "section", "visibility").
    pub name: Symbol,
    /// Attribute arguments — may be empty for flag-style attributes.
    pub args: Vec<AttributeArg>,
    /// Source location of the entire attribute specification.
    pub span: Span,
}

/// An argument to a GCC `__attribute__`.
///
/// Attributes accept a heterogeneous set of argument forms: integer constants,
/// string literals, identifiers (for format-style attributes), and general
/// constant expressions.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeArg {
    /// Integer constant argument (e.g., alignment value `16`).
    Integer(i64),
    /// String literal argument (e.g., section name `".init.text"`).
    /// Stored as raw bytes to preserve PUA-encoded non-UTF-8 content.
    String(Vec<u8>),
    /// Identifier argument (e.g., `printf` in `format(printf, 1, 2)`).
    Identifier(Symbol),
    /// General expression argument (e.g., `sizeof(int)` in `aligned(sizeof(int))`).
    Expression(Box<Expression>),
}

// ===========================================================================
// Inline Assembly AST
// ===========================================================================

/// An inline assembly statement (`asm` / `__asm__`).
///
/// Supports the full GCC extended assembly syntax including:
/// - Volatile flag (`asm volatile`)
/// - Goto flag (`asm goto` with jump labels)
/// - Template strings (multiple string literal fragments concatenated)
/// - Output operands with constraints
/// - Input operands with constraints
/// - Clobber registers/flags
/// - Named operands (`[name] "constraint" (expr)`)
///
/// # Assembly Syntax
///
/// ```c
/// asm [volatile] [goto] (
///     "template"           // template strings
///     : "=r"(out)          // output operands
///     : "r"(in)            // input operands
///     : "memory", "cc"     // clobbers
///     : label1, label2     // goto labels (asm goto only)
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AsmStatement {
    /// `true` if `volatile` / `__volatile__` keyword is present.
    /// Prevents the compiler from optimizing away or reordering the assembly.
    pub is_volatile: bool,
    /// `true` if `goto` keyword is present, enabling jump to C labels.
    pub is_goto: bool,
    /// Assembly template string fragments. Multiple adjacent string literals
    /// are concatenated during parsing. Stored as raw bytes to preserve
    /// PUA-encoded content for `.pushsection`/`.popsection` directives.
    pub template: Vec<Vec<u8>>,
    /// Output operands — expressions written to by the assembly.
    pub outputs: Vec<AsmOperand>,
    /// Input operands — expressions read by the assembly.
    pub inputs: Vec<AsmOperand>,
    /// Clobber list — registers and special flags ("memory", "cc") that the
    /// assembly may modify beyond the declared outputs.
    pub clobbers: Vec<String>,
    /// Goto labels — C labels that the assembly may jump to (asm goto only).
    pub goto_labels: Vec<Symbol>,
    /// Source location spanning the entire asm statement.
    pub span: Span,
}

/// An operand in an inline assembly statement (input or output).
///
/// Each operand binds an assembly constraint string to a C expression.
/// Optionally, a symbolic name can be given for use in the template
/// string via `%[name]` syntax.
///
/// # Constraint Syntax
///
/// - Output: `"=r"`, `"=m"`, `"+r"` (read-write)
/// - Input: `"r"`, `"i"`, `"n"`, `"m"`, digit (matching constraint)
#[derive(Clone, Debug, PartialEq)]
pub struct AsmOperand {
    /// Optional symbolic name for `%[name]` reference in the template.
    pub name: Option<Symbol>,
    /// Constraint string (e.g., `"=r"`, `"+m"`, `"i"`).
    pub constraint: String,
    /// The C expression bound to this operand.
    pub expression: Box<Expression>,
    /// Source location of this operand specification.
    pub span: Span,
}

// ===========================================================================
// Type Specifiers
// ===========================================================================

/// Operand of `typeof` / `__typeof__` — either an expression or a type name.
///
/// GCC extension: `typeof(expr)` yields the type of the expression;
/// `typeof(type-name)` is an identity (useful in macros).
#[derive(Clone, Debug, PartialEq)]
pub enum TypeofOperand {
    /// `typeof(expression)` — the type is inferred from the expression.
    Expression(Box<Expression>),
    /// `typeof(type-name)` — the type is explicitly specified.
    TypeName(Box<TypeName>),
}

/// C11 type specifiers (§6.7.2) with GCC extensions.
///
/// Type specifiers appear in declaration specifiers and specifier-qualifier
/// lists. Multiple specifiers can be combined (e.g., `unsigned long long`).
/// The semantic analyzer resolves the combined specifier list into a concrete
/// C type.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeSpecifier {
    /// `void`
    Void,
    /// `_Bool`
    Bool,
    /// `char`
    Char,
    /// `short`
    Short,
    /// `int`
    Int,
    /// `long` — may appear once (long) or twice (long long)
    Long,
    /// `float`
    Float,
    /// `double`
    Double,
    /// `signed`
    Signed,
    /// `unsigned`
    Unsigned,
    /// `_Complex` (C11 §6.7.2) — complex floating-point modifier
    Complex,
    /// `_Atomic(type-name)` — atomic type specifier (C11 §6.7.2.4)
    Atomic(Box<TypeName>),
    /// `struct [name] [{ fields }]` — struct type specifier
    Struct {
        /// Optional tag name (anonymous structs have `None`).
        name: Option<Symbol>,
        /// Field list — `Some` for definitions, `None` for forward references.
        fields: Option<Vec<FieldDeclaration>>,
        /// Attributes applied to the struct.
        attrs: Vec<Attribute>,
        /// Source location of the struct specifier.
        span: Span,
    },
    /// `union [name] [{ fields }]` — union type specifier
    Union {
        /// Optional tag name.
        name: Option<Symbol>,
        /// Field list — `Some` for definitions, `None` for forward references.
        fields: Option<Vec<FieldDeclaration>>,
        /// Attributes applied to the union.
        attrs: Vec<Attribute>,
        /// Source location of the union specifier.
        span: Span,
    },
    /// `enum [name] [{ enumerators }]` — enum type specifier
    Enum {
        /// Optional tag name.
        name: Option<Symbol>,
        /// Enumerator list — `Some` for definitions, `None` for forward references.
        enumerators: Option<Vec<Enumerator>>,
        /// Attributes applied to the enum.
        attrs: Vec<Attribute>,
        /// Source location of the enum specifier.
        span: Span,
    },
    /// A typedef name used as a type specifier (e.g., `size_t`, `uint32_t`).
    TypedefName {
        /// The interned name of the typedef.
        name: Symbol,
        /// Source location of the typedef name token.
        span: Span,
    },
    /// `typeof(operand)` / `__typeof__(operand)` — GCC extension.
    Typeof {
        /// The operand — either an expression or a type name.
        operand: TypeofOperand,
        /// Source location of the typeof specifier.
        span: Span,
    },
}

// ===========================================================================
// Type Name, Abstract Declarator, Specifier-Qualifier List
// ===========================================================================

/// A specifier-qualifier list — type specifiers and qualifiers without
/// storage class or function specifiers.
///
/// Used in `TypeName` (casts, sizeof, _Alignof, _Atomic, _Generic),
/// struct/union member declarations, and typeof.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecifierQualifierList {
    /// Type specifiers (e.g., `unsigned`, `long`, `int`, struct/union/enum).
    pub specifiers: Vec<TypeSpecifier>,
    /// Type qualifiers (const, volatile, restrict, _Atomic).
    pub qualifiers: TypeQualifiers,
    /// Source location spanning the entire specifier-qualifier list.
    pub span: Span,
}

/// An abstract declarator — a declarator without a name, used in type names
/// (casts, sizeof, parameter declarations with unnamed parameters).
///
/// Contains only derived declarator parts (pointer, array, function).
#[derive(Clone, Debug, PartialEq)]
pub struct AbstractDeclarator {
    /// Derived declarator modifiers (pointers, arrays, function parameter lists).
    pub derived: Vec<DerivedDeclarator>,
    /// Source location of the abstract declarator.
    pub span: Span,
}

/// A type name — specifier-qualifier list plus optional abstract declarator.
///
/// Used in cast expressions, sizeof/alignof operands, compound literals,
/// _Atomic(type-name), _Generic type associations, and _Alignas(type-name).
///
/// # Examples
///
/// - `int` → specifiers=[Int], declarator=None
/// - `const int *` → specifiers=[Int], qualifiers={const}, declarator=Some(Pointer)
/// - `void (*)(int)` → specifiers=[Void], declarator=Some(Pointer→Function)
#[derive(Clone, Debug, PartialEq)]
pub struct TypeName {
    /// The specifier-qualifier list portion of the type name.
    pub specifiers: SpecifierQualifierList,
    /// Optional abstract declarator for pointer/array/function types.
    pub declarator: Option<AbstractDeclarator>,
    /// Source location spanning the entire type name.
    pub span: Span,
}

// ===========================================================================
// Declaration Specifiers
// ===========================================================================

/// A complete set of declaration specifiers (C11 §6.7).
///
/// Aggregates all specifiers that can appear before a declarator list in a
/// declaration: storage class, type specifiers, type qualifiers, function
/// specifiers, alignment, and GCC attributes.
///
/// The `has_extension` flag tracks whether `__extension__` preceded this
/// declaration, suppressing certain GCC warnings.
#[derive(Clone, Debug, PartialEq)]
pub struct DeclarationSpecifiers {
    /// Storage class specifier (at most one, except _Thread_local combinations).
    pub storage_class: Option<StorageClass>,
    /// Type specifiers — accumulated from the declaration specifier list.
    /// Multiple specifiers combine (e.g., `unsigned long long` = [Unsigned, Long, Long]).
    pub type_specifiers: Vec<TypeSpecifier>,
    /// Type qualifiers (const, volatile, restrict, _Atomic).
    pub type_qualifiers: TypeQualifiers,
    /// Function specifiers (inline, _Noreturn).
    pub function_specifiers: FunctionSpecifiers,
    /// Optional `_Alignas` alignment specifier.
    pub alignment: Option<AlignasSpecifier>,
    /// GCC `__attribute__` annotations attached to the declaration specifiers.
    pub attrs: Vec<Attribute>,
    /// `true` if `__extension__` keyword preceded this declaration.
    pub has_extension: bool,
    /// Source location spanning all declaration specifiers.
    pub span: Span,
}

// ===========================================================================
// Declarators
// ===========================================================================

/// A derived declarator — pointer, array, or function modifiers that compose
/// with a base declarator to form the complete declared type.
///
/// In C's "declaration mirrors use" syntax, derived declarators build outward
/// from the identifier: `int *a[10]` → name=a, derived=[Pointer, Array(10)].
/// Reading from innermost (name) outward: a is a pointer to array of 10 ints.
///
/// The parser stores these in the order they are parsed (outside-in for
/// pointers, inside-out for postfix), and the semantic analyzer reads them
/// to construct the full type.
#[derive(Clone, Debug, PartialEq)]
pub enum DerivedDeclarator {
    /// `* [qualifiers]` — pointer derivation with optional qualifiers.
    Pointer {
        /// Qualifiers applied to the pointer itself (e.g., `* const`).
        qualifiers: TypeQualifiers,
    },
    /// `[size]` or `[static size]` or `[*]` — array derivation.
    Array {
        /// Array size expression — `None` for `[]` (incomplete array) or `[*]` (VLA).
        size: Option<Box<Expression>>,
        /// `true` if the `static` keyword appears in the array declarator
        /// (C11 §6.7.6.3, parameter array notation).
        is_static: bool,
        /// Qualifiers inside the array brackets (C11 §6.7.6.2).
        qualifiers: TypeQualifiers,
    },
    /// `(params)` — function derivation with parameter list.
    Function {
        /// The function's parameter list (possibly variadic).
        params: ParameterList,
    },
}

/// A named declarator — the core of C declaration syntax.
///
/// Combines an optional identifier name with zero or more derived declarator
/// modifiers (pointers, arrays, function parameter lists) and GCC attributes.
///
/// # Examples
///
/// - `x` → name=Some("x"), derived=[]
/// - `*p` → name=Some("p"), derived=[Pointer]
/// - `arr[10]` → name=Some("arr"), derived=[Array(10)]
/// - `(*fptr)(int, int)` → name=Some("fptr"), derived=[Pointer, Function([int, int])]
#[derive(Clone, Debug, PartialEq)]
pub struct Declarator {
    /// The declared identifier — `None` for abstract declarators in parameter
    /// declarations where the name is omitted.
    pub name: Option<Symbol>,
    /// Derived declarator modifiers building the type outward from the name.
    pub derived: Vec<DerivedDeclarator>,
    /// GCC attributes attached to this declarator.
    pub attrs: Vec<Attribute>,
    /// Source location of the declarator.
    pub span: Span,
}

/// A declarator paired with an optional initializer.
///
/// Used in variable declarations: `int x = 5, y, z = {1, 2};`
/// Each init-declarator is one entry in the comma-separated list.
#[derive(Clone, Debug, PartialEq)]
pub struct InitDeclarator {
    /// The declarator (name + type modifiers).
    pub declarator: Declarator,
    /// Optional initializer expression or brace-enclosed list.
    pub initializer: Option<Initializer>,
    /// Source location spanning the declarator and initializer.
    pub span: Span,
}

// ===========================================================================
// Parameter List
// ===========================================================================

/// A function parameter list.
///
/// Contains the list of parameters and a flag indicating whether the function
/// is variadic (ends with `, ...`).
///
/// # Special Cases
///
/// - `()` → empty params, variadic=false (K&R-style, or no-parameter in C)
/// - `(void)` → single Void parameter, variadic=false (explicitly no parameters)
/// - `(int, ...)` → one int param, variadic=true
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterList {
    /// The parameter declarations.
    pub params: Vec<Parameter>,
    /// `true` if the parameter list ends with `, ...` (variadic function).
    pub variadic: bool,
    /// Source location spanning the entire parameter list including parentheses.
    pub span: Span,
}

/// A single function parameter declaration.
///
/// In a full prototype, each parameter has declaration specifiers and an
/// optional declarator (the name may be omitted in prototypes).
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    /// Declaration specifiers for this parameter's type.
    pub specifiers: DeclarationSpecifiers,
    /// Optional declarator — `None` when the parameter name is omitted
    /// (e.g., `void foo(int, char *)` — the `int` parameter has no name).
    pub declarator: Option<Declarator>,
    /// Source location of this parameter.
    pub span: Span,
}

// ===========================================================================
// Struct/Union Field Declarations
// ===========================================================================

/// A struct or union field (member) declaration.
///
/// A single field declaration line can declare multiple members:
/// `int x, y:3, z;` → one FieldDeclaration with three FieldDeclarators.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldDeclaration {
    /// Type specifiers and qualifiers for this field declaration.
    pub specifiers: DeclarationSpecifiers,
    /// Individual field declarators (possibly with bitfield widths).
    pub declarators: Vec<FieldDeclarator>,
    /// GCC attributes applied to this field declaration.
    pub attrs: Vec<Attribute>,
    /// Source location of the entire field declaration line.
    pub span: Span,
}

/// A single field declarator within a struct/union field declaration.
///
/// The declarator may be absent for anonymous bitfields (`int :3;`)
/// or anonymous struct/union members.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldDeclarator {
    /// The field declarator — `None` for anonymous bitfields.
    pub declarator: Option<Declarator>,
    /// Bitfield width expression — `None` for regular (non-bitfield) members.
    pub bit_width: Option<Box<Expression>>,
    /// Source location of this field declarator.
    pub span: Span,
}

// ===========================================================================
// Enumerator
// ===========================================================================

/// An enumerator constant within an `enum` definition.
///
/// Each enumerator has a name and an optional explicit value expression.
/// When the value is `None`, the enumerator's value is one greater than
/// the previous enumerator (or 0 for the first).
///
/// # Examples
///
/// ```c
/// enum color { RED, GREEN = 5, BLUE };  // RED=0, GREEN=5, BLUE=6
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Enumerator {
    /// The enumerator constant name.
    pub name: Symbol,
    /// Optional explicit value expression.
    pub value: Option<Box<Expression>>,
    /// GCC attributes on the enumerator.
    pub attrs: Vec<Attribute>,
    /// Source location of this enumerator.
    pub span: Span,
}

// ===========================================================================
// Initializers
// ===========================================================================

/// A designator in a designated initializer.
///
/// C11 §6.7.9 allows designators to specify which struct field or array
/// index an initializer value applies to.
///
/// # Examples
///
/// - `.field` → `Designator::Field(field)`
/// - `[3]` → `Designator::Index(3)`
/// - `.field.subfield[2]` → [Field(field), Field(subfield), Index(2)]
#[derive(Clone, Debug, PartialEq)]
pub enum Designator {
    /// `.field` — struct/union field designator.
    Field(Symbol),
    /// `[index]` — array index designator.
    Index(Box<Expression>),
}

/// A single item in a brace-enclosed initializer list.
///
/// Each item consists of zero or more designators followed by an initializer
/// value. When the designator list is empty, the item initializes the next
/// member/element in sequence.
///
/// # Examples
///
/// - `5` → designators=[], initializer=Expression(5)
/// - `.x = 10` → designators=[Field(x)], initializer=Expression(10)
/// - `[2] = {1, 2}` → designators=[Index(2)], initializer=List([1, 2])
#[derive(Clone, Debug, PartialEq)]
pub struct InitializerItem {
    /// Designator chain — empty for positional initialization.
    pub designators: Vec<Designator>,
    /// The initializer value for this item.
    pub initializer: Initializer,
    /// Source location of this initializer item.
    pub span: Span,
}

/// An initializer — either a single expression or a brace-enclosed list.
///
/// C11 §6.7.9: Initializers provide values for declared objects. Simple
/// initializers are single expressions; aggregate/union initializers use
/// brace-enclosed lists with optional designated initializers.
///
/// # Examples
///
/// - `int x = 5;` → Initializer::Expression(5)
/// - `int a[] = {1, 2, 3};` → Initializer::List([Item(1), Item(2), Item(3)])
/// - `struct s = {.x = 1, .y = 2};` → Initializer::List with designators
#[derive(Clone, Debug, PartialEq)]
pub enum Initializer {
    /// A single expression initializer (e.g., `= 5`).
    Expression(Box<Expression>),
    /// A brace-enclosed initializer list (e.g., `= {1, 2, .x = 3}`).
    List {
        /// The initializer items within the braces.
        items: Vec<InitializerItem>,
        /// Source location spanning the braces and their contents.
        span: Span,
    },
}

// ===========================================================================
// sizeof / alignof Operand Types
// ===========================================================================

/// Operand of `sizeof` — either a parenthesized expression or a type name.
///
/// - `sizeof(expr)` and `sizeof expr` → `SizeofOperand::Expression`
/// - `sizeof(type-name)` → `SizeofOperand::TypeName`
#[derive(Clone, Debug, PartialEq)]
pub enum SizeofOperand {
    /// `sizeof expression` — size of the expression's type.
    Expression(Box<Expression>),
    /// `sizeof(type-name)` — size of the named type.
    TypeName(Box<TypeName>),
}

/// Operand of `_Alignof` / `__alignof__` — can be either a type name
/// (per C11 standard) or an expression (GCC extension).
///
/// - `_Alignof(type-name)` → `AlignofOperand::TypeName`
/// - `__alignof__(expr)` → `AlignofOperand::Expression` (GCC extension)
#[derive(Clone, Debug, PartialEq)]
pub enum AlignofOperand {
    /// `_Alignof(type-name)` — alignment of the named type.
    TypeName(Box<TypeName>),
    /// `__alignof__(expression)` — alignment of the expression's type (GCC).
    Expression(Box<Expression>),
}

/// A `_Generic` type association (C11 §6.5.1.1).
///
/// Maps a type to an expression in a `_Generic` selection. When `type_name`
/// is `None`, this is the `default:` association.
///
/// # Examples
///
/// ```c
/// _Generic(x,
///     int: "integer",        // type_name = Some(int), expression = "integer"
///     double: "floating",    // type_name = Some(double), expression = "floating"
///     default: "other"       // type_name = None, expression = "other"
/// )
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct GenericAssociation {
    /// The associated type — `None` for the `default` association.
    pub type_name: Option<TypeName>,
    /// The expression to evaluate when this association is selected.
    pub expression: Box<Expression>,
    /// Source location of this association.
    pub span: Span,
}

// ===========================================================================
// Expression Nodes
// ===========================================================================

/// An expression in the C11 language with GCC extensions.
///
/// This enum covers every form of C expression the parser can produce.
/// All variants carry a `Span` for source location tracking. Recursive
/// sub-expressions are boxed to ensure finite enum size.
///
/// GCC extensions included:
/// - `StatementExpression` — `({ ... })` compound statement in expression context
/// - `LabelAddress` — `&&label` for computed goto targets
/// - `Conditional` with `then_expr: None` — GCC `x ?: y` conditional omission
///
/// Operator precedence is fully resolved during parsing — the AST tree
/// structure directly encodes associativity and precedence.
#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    /// Integer constant: `42`, `0xFF`, `0b1010`, `42ULL`.
    IntegerLiteral {
        /// The integer value (up to 128 bits for maximum precision).
        value: u128,
        /// Integer suffix determining the C type.
        suffix: IntegerSuffix,
        /// Source location of the literal token.
        span: Span,
    },

    /// Floating-point constant: `3.14`, `1e10`, `0x1.0p10`, `3.14f`.
    FloatLiteral {
        /// The floating-point value (stored as f64; long double uses
        /// software arithmetic during constant evaluation).
        value: f64,
        /// Float suffix determining the C type (float, double, long double).
        suffix: FloatSuffix,
        /// Source location of the literal token.
        span: Span,
    },

    /// String literal: `"hello"`, `L"wide"`, `u8"utf8"`.
    /// Adjacent string literals are concatenated during parsing.
    StringLiteral {
        /// Raw byte content of the string (including PUA-encoded non-UTF-8 bytes).
        /// Does NOT include the null terminator — that is added during IR lowering.
        value: Vec<u8>,
        /// Encoding prefix determining the element type.
        prefix: StringPrefix,
        /// Source location spanning all concatenated string tokens.
        span: Span,
    },

    /// Character constant: `'A'`, `L'\n'`, `'\x41'`.
    CharLiteral {
        /// The character value as a Unicode code point (or raw byte value).
        value: u32,
        /// Encoding prefix determining the type.
        prefix: CharPrefix,
        /// Source location of the character literal token.
        span: Span,
    },

    /// Identifier reference: `x`, `printf`, `my_var`.
    Identifier {
        /// The interned identifier name.
        name: Symbol,
        /// Source location of the identifier token.
        span: Span,
    },

    /// Binary operation: `a + b`, `x = y`, `p && q`.
    BinaryOp {
        /// The binary operator.
        op: BinaryOperator,
        /// Left-hand side expression.
        left: Box<Expression>,
        /// Right-hand side expression.
        right: Box<Expression>,
        /// Source location spanning the entire binary expression.
        span: Span,
    },

    /// Unary operation: `-x`, `!flag`, `++i`, `p--`.
    UnaryOp {
        /// The unary operator.
        op: UnaryOperator,
        /// The operand expression.
        operand: Box<Expression>,
        /// `true` for postfix operators (PostInc, PostDec), `false` for prefix.
        is_postfix: bool,
        /// Source location spanning the operator and operand.
        span: Span,
    },

    /// Conditional (ternary) expression: `a ? b : c`.
    ///
    /// GCC extension: `a ?: c` — when `then_expr` is `None`, the condition
    /// value `a` is used as the "then" value if truthy.
    Conditional {
        /// The condition expression.
        condition: Box<Expression>,
        /// The "then" expression — `None` for GCC `x ?: y` omission.
        then_expr: Option<Box<Expression>>,
        /// The "else" expression.
        else_expr: Box<Expression>,
        /// Source location spanning the entire conditional.
        span: Span,
    },

    /// Function call: `printf("hello")`, `(*fptr)(x, y)`.
    FunctionCall {
        /// The callee expression (function name, pointer dereference, etc.).
        callee: Box<Expression>,
        /// Argument expressions passed to the function.
        args: Vec<Expression>,
        /// Source location spanning callee through closing parenthesis.
        span: Span,
    },

    /// Array subscript: `a[i]`.
    ArraySubscript {
        /// The array or pointer expression.
        array: Box<Expression>,
        /// The index expression.
        index: Box<Expression>,
        /// Source location spanning the expression and brackets.
        span: Span,
    },

    /// Struct/union member access via `.`: `s.field`.
    MemberAccess {
        /// The struct/union object expression.
        object: Box<Expression>,
        /// The member name.
        member: Symbol,
        /// Source location spanning the expression, dot, and member name.
        span: Span,
    },

    /// Struct/union member access via `->`: `p->field`.
    ArrowAccess {
        /// The pointer expression.
        pointer: Box<Expression>,
        /// The member name.
        member: Symbol,
        /// Source location spanning the expression, arrow, and member name.
        span: Span,
    },

    /// Explicit type cast: `(int)x`, `(void *)p`.
    Cast {
        /// The target type.
        type_name: Box<TypeName>,
        /// The expression being cast.
        operand: Box<Expression>,
        /// Source location spanning the cast and operand.
        span: Span,
    },

    /// `sizeof` operator: `sizeof(int)`, `sizeof x`.
    Sizeof {
        /// The operand — expression or type name.
        operand: SizeofOperand,
        /// Source location of the sizeof expression.
        span: Span,
    },

    /// `_Alignof` / `__alignof__` operator.
    Alignof {
        /// The operand — type name (C11) or expression (GCC extension).
        operand: AlignofOperand,
        /// Source location of the alignof expression.
        span: Span,
    },

    /// Compound literal: `(int[]){1, 2, 3}`, `(struct point){.x=1, .y=2}`.
    CompoundLiteral {
        /// The type of the compound literal.
        type_name: Box<TypeName>,
        /// The brace-enclosed initializer.
        initializer: Initializer,
        /// Source location spanning the type and initializer.
        span: Span,
    },

    /// Comma expression: `a, b, c` — evaluates all, yields the last.
    Comma {
        /// The sequence of sub-expressions (at least two).
        expressions: Vec<Expression>,
        /// Source location spanning the entire comma expression.
        span: Span,
    },

    /// C11 `_Generic` selection expression (§6.5.1.1).
    Generic {
        /// The controlling expression whose type selects the association.
        controlling: Box<Expression>,
        /// Type-expression associations (including optional `default`).
        associations: Vec<GenericAssociation>,
        /// Source location of the _Generic expression.
        span: Span,
    },

    /// GCC statement expression: `({ int t = a; a = b; b = t; t; })`.
    ///
    /// The value of a statement expression is the value of the last
    /// expression-statement in the compound statement.
    StatementExpression {
        /// The block items (declarations and statements) within the `({ })`.
        body: Vec<BlockItem>,
        /// Source location spanning `({` through `})`.
        span: Span,
    },

    /// GCC label address: `&&label` for computed goto.
    ///
    /// Takes the address of a label as a `void *` value, used with
    /// `goto *expr` (computed goto) for jump tables.
    LabelAddress {
        /// The label name whose address is taken.
        label: Symbol,
        /// Source location of the `&&label` expression.
        span: Span,
    },

    /// Address-of operator: `&x`.
    ///
    /// This is a distinct node from `UnaryOp { op: AddressOf, .. }` for
    /// clarity in IR lowering, which handles address-of differently from
    /// other unary operators.
    AddressOf {
        /// The operand whose address is taken (must be an lvalue).
        operand: Box<Expression>,
        /// Source location of the `&` operator and operand.
        span: Span,
    },

    /// Dereference (indirection) operator: `*p`.
    ///
    /// This is a distinct node from `UnaryOp { op: Deref, .. }` for
    /// clarity in IR lowering.
    Dereference {
        /// The pointer operand to dereference.
        operand: Box<Expression>,
        /// Source location of the `*` operator and operand.
        span: Span,
    },

    /// Error recovery placeholder — used when the parser encounters a syntax
    /// error in an expression context and needs to continue parsing.
    Error {
        /// Source location of the erroneous token(s).
        span: Span,
    },
}

impl Expression {
    /// Returns the source `Span` of this expression, regardless of variant.
    ///
    /// Every expression variant carries a span, so this method provides
    /// uniform access without matching on all variants.
    pub fn span(&self) -> Span {
        match self {
            Expression::IntegerLiteral { span, .. }
            | Expression::FloatLiteral { span, .. }
            | Expression::StringLiteral { span, .. }
            | Expression::CharLiteral { span, .. }
            | Expression::Identifier { span, .. }
            | Expression::BinaryOp { span, .. }
            | Expression::UnaryOp { span, .. }
            | Expression::Conditional { span, .. }
            | Expression::FunctionCall { span, .. }
            | Expression::ArraySubscript { span, .. }
            | Expression::MemberAccess { span, .. }
            | Expression::ArrowAccess { span, .. }
            | Expression::Cast { span, .. }
            | Expression::Sizeof { span, .. }
            | Expression::Alignof { span, .. }
            | Expression::CompoundLiteral { span, .. }
            | Expression::Comma { span, .. }
            | Expression::Generic { span, .. }
            | Expression::StatementExpression { span, .. }
            | Expression::LabelAddress { span, .. }
            | Expression::AddressOf { span, .. }
            | Expression::Dereference { span, .. }
            | Expression::Error { span, .. } => *span,
        }
    }
}

// ===========================================================================
// Statement Nodes
// ===========================================================================

/// A block item — either a declaration or a statement.
///
/// C11 §6.8.2: Compound statements contain a list of block items. This
/// allows declarations and statements to be freely intermixed (unlike C89
/// which required declarations before statements).
#[derive(Clone, Debug, PartialEq)]
pub enum BlockItem {
    /// A declaration within a block (variable, typedef, struct/union/enum, etc.).
    Declaration(Declaration),
    /// A statement within a block.
    Statement(Statement),
}

/// Initializer for the first clause of a `for` statement.
///
/// C99/C11 allows either a declaration (`for (int i = 0; ...)`) or an
/// expression (`for (i = 0; ...)`) in the initializer position.
#[derive(Clone, Debug, PartialEq)]
pub enum ForInit {
    /// A declaration in the for-init position (e.g., `int i = 0`).
    Declaration(Box<Declaration>),
    /// An expression in the for-init position (e.g., `i = 0`).
    Expression(Box<Expression>),
}

/// A C11 statement with GCC extensions.
///
/// This enum covers every form of statement the parser can produce. All
/// variants carry a `Span` for source location tracking. GCC extensions
/// include `CaseRange`, `ComputedGoto`, and `Asm`.
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    /// Compound statement (block): `{ decl_or_stmt* }`.
    Compound {
        /// The block items (declarations and statements).
        items: Vec<BlockItem>,
        /// Source location spanning the braces.
        span: Span,
    },

    /// `if (condition) then_branch [else else_branch]`.
    If {
        /// The condition expression (parenthesized in source, unwrapped in AST).
        condition: Box<Expression>,
        /// The "then" branch statement.
        then_branch: Box<Statement>,
        /// Optional "else" branch statement.
        else_branch: Option<Box<Statement>>,
        /// Source location from `if` keyword through the end of both branches.
        span: Span,
    },

    /// `while (condition) body`.
    While {
        /// The loop condition.
        condition: Box<Expression>,
        /// The loop body statement.
        body: Box<Statement>,
        /// Source location from `while` keyword through body.
        span: Span,
    },

    /// `do body while (condition);`.
    DoWhile {
        /// The loop body statement.
        body: Box<Statement>,
        /// The loop condition (evaluated after each iteration).
        condition: Box<Expression>,
        /// Source location from `do` through `;`.
        span: Span,
    },

    /// `for (init; condition; increment) body`.
    For {
        /// Optional initialization (declaration or expression).
        init: Option<ForInit>,
        /// Optional loop condition — `None` means infinite loop.
        condition: Option<Box<Expression>>,
        /// Optional increment expression.
        increment: Option<Box<Expression>>,
        /// The loop body statement.
        body: Box<Statement>,
        /// Source location from `for` through body.
        span: Span,
    },

    /// `switch (expression) body`.
    Switch {
        /// The controlling expression.
        expression: Box<Expression>,
        /// The switch body (typically a compound statement with case labels).
        body: Box<Statement>,
        /// Source location from `switch` through body.
        span: Span,
    },

    /// `case value: body` — a case label in a switch statement.
    Case {
        /// The case value (must be an integer constant expression).
        value: Box<Expression>,
        /// The statement following the case label.
        body: Box<Statement>,
        /// Source location from `case` through the end of the body.
        span: Span,
    },

    /// `case low ... high: body` — GCC case range extension.
    ///
    /// Matches any value in the inclusive range `[low, high]`.
    /// Both `low` and `high` must be integer constant expressions.
    CaseRange {
        /// The lower bound of the range (inclusive).
        low: Box<Expression>,
        /// The upper bound of the range (inclusive).
        high: Box<Expression>,
        /// The statement following the case range label.
        body: Box<Statement>,
        /// Source location from `case` through body.
        span: Span,
    },

    /// `default: body` — default label in a switch statement.
    Default {
        /// The statement following the default label.
        body: Box<Statement>,
        /// Source location from `default` through body.
        span: Span,
    },

    /// `goto label;` — unconditional jump to a named label.
    Goto {
        /// The target label name.
        label: Symbol,
        /// Source location from `goto` through `;`.
        span: Span,
    },

    /// `goto *expression;` — GCC computed goto.
    ///
    /// Jumps to the address held in the expression (typically obtained via
    /// `&&label`). Used to implement jump tables in performance-critical code.
    ComputedGoto {
        /// The expression evaluating to a `void *` label address.
        target: Box<Expression>,
        /// Source location from `goto` through `;`.
        span: Span,
    },

    /// `break;` — exit the innermost loop or switch.
    Break {
        /// Source location of the `break` statement.
        span: Span,
    },

    /// `continue;` — jump to the next iteration of the innermost loop.
    Continue {
        /// Source location of the `continue` statement.
        span: Span,
    },

    /// `return [expression];` — return from the current function.
    Return {
        /// Optional return value expression.
        value: Option<Box<Expression>>,
        /// Source location from `return` through `;`.
        span: Span,
    },

    /// `label: statement` — a named label (target of `goto`).
    Labeled {
        /// The label name.
        label: Symbol,
        /// GCC attributes on the label (e.g., `__attribute__((unused))`).
        attrs: Vec<Attribute>,
        /// The statement following the label.
        body: Box<Statement>,
        /// Source location from the label through body.
        span: Span,
    },

    /// An expression statement: `expr;`.
    Expression {
        /// The expression being evaluated as a statement.
        expr: Box<Expression>,
        /// Source location from expression through `;`.
        span: Span,
    },

    /// Null statement: `;` (empty statement).
    Null {
        /// Source location of the semicolon.
        span: Span,
    },

    /// Inline assembly statement.
    Asm(Box<AsmStatement>),

    /// Error recovery placeholder — used when the parser encounters a syntax
    /// error in a statement context.
    Error {
        /// Source location of the erroneous token(s).
        span: Span,
    },
}

impl Statement {
    /// Returns the source `Span` of this statement, regardless of variant.
    pub fn span(&self) -> Span {
        match self {
            Statement::Compound { span, .. }
            | Statement::If { span, .. }
            | Statement::While { span, .. }
            | Statement::DoWhile { span, .. }
            | Statement::For { span, .. }
            | Statement::Switch { span, .. }
            | Statement::Case { span, .. }
            | Statement::CaseRange { span, .. }
            | Statement::Default { span, .. }
            | Statement::Goto { span, .. }
            | Statement::ComputedGoto { span, .. }
            | Statement::Break { span, .. }
            | Statement::Continue { span, .. }
            | Statement::Return { span, .. }
            | Statement::Labeled { span, .. }
            | Statement::Expression { span, .. }
            | Statement::Null { span, .. }
            | Statement::Error { span, .. } => *span,
            Statement::Asm(asm) => asm.span,
        }
    }
}

// ===========================================================================
// Declaration Nodes
// ===========================================================================

/// A top-level or block-scope declaration.
///
/// This enum covers all forms of C11 declarations that can appear at file
/// scope (translation unit level) or within compound statements. GCC
/// attribute annotations are attached to declaration variants that support
/// them.
///
/// # Variants
///
/// - `Variable` — variable/object declarations with optional initializers
/// - `FunctionDef` — function definition with a body
/// - `FunctionDecl` — forward function declaration (prototype)
/// - `Typedef` — type alias declaration
/// - `StructDef` / `UnionDef` / `EnumDef` — tag type definitions (standalone)
/// - `StaticAssert` — compile-time assertion (`_Static_assert`)
/// - `Empty` — lone semicolon (C11 allows empty declarations)
/// - `Error` — parser error recovery placeholder
#[derive(Clone, Debug, PartialEq)]
pub enum Declaration {
    /// Variable or object declaration with optional initializers.
    ///
    /// # Examples
    ///
    /// - `int x;` — single declarator, no initializer
    /// - `int x = 5, y = 10;` — multiple init-declarators
    /// - `static const char *name = "hello";` — with storage class and qualifiers
    Variable {
        /// Declaration specifiers (type, storage class, qualifiers, attributes).
        specifiers: DeclarationSpecifiers,
        /// List of declarator-initializer pairs.
        declarators: Vec<InitDeclarator>,
        /// GCC attributes attached to the declaration.
        attrs: Vec<Attribute>,
        /// Source location spanning the entire declaration.
        span: Span,
    },

    /// Function definition with a body.
    ///
    /// # Example
    ///
    /// ```c
    /// int main(int argc, char **argv) { return 0; }
    /// ```
    FunctionDef {
        /// Declaration specifiers for the return type and storage class.
        specifiers: DeclarationSpecifiers,
        /// The function declarator (name and parameter list).
        declarator: Declarator,
        /// GCC attributes attached to the function definition.
        attrs: Vec<Attribute>,
        /// The function body (compound statement).
        body: Box<Statement>,
        /// Source location from specifiers through closing brace.
        span: Span,
    },

    /// Forward function declaration (prototype without body).
    ///
    /// # Example
    ///
    /// ```c
    /// extern int printf(const char *fmt, ...);
    /// ```
    FunctionDecl {
        /// Declaration specifiers for the return type.
        specifiers: DeclarationSpecifiers,
        /// The function declarator (name and parameter list).
        declarator: Declarator,
        /// GCC attributes.
        attrs: Vec<Attribute>,
        /// Source location spanning the declaration.
        span: Span,
    },

    /// Typedef declaration — creates a type alias.
    ///
    /// # Example
    ///
    /// ```c
    /// typedef unsigned long size_t;
    /// typedef struct node { int val; struct node *next; } Node;
    /// ```
    Typedef {
        /// Declaration specifiers (must include StorageClass::Typedef).
        specifiers: DeclarationSpecifiers,
        /// The declarators being aliased (one or more).
        declarators: Vec<Declarator>,
        /// GCC attributes.
        attrs: Vec<Attribute>,
        /// Source location spanning the typedef declaration.
        span: Span,
    },

    /// Standalone struct definition.
    ///
    /// # Example
    ///
    /// ```c
    /// struct point { int x; int y; };
    /// ```
    StructDef {
        /// Optional struct tag name.
        name: Option<Symbol>,
        /// Struct field declarations.
        fields: Vec<FieldDeclaration>,
        /// GCC attributes.
        attrs: Vec<Attribute>,
        /// Source location.
        span: Span,
    },

    /// Standalone union definition.
    UnionDef {
        /// Optional union tag name.
        name: Option<Symbol>,
        /// Union field declarations.
        fields: Vec<FieldDeclaration>,
        /// GCC attributes.
        attrs: Vec<Attribute>,
        /// Source location.
        span: Span,
    },

    /// Standalone enum definition.
    ///
    /// # Example
    ///
    /// ```c
    /// enum color { RED, GREEN, BLUE };
    /// ```
    EnumDef {
        /// Optional enum tag name.
        name: Option<Symbol>,
        /// Enumerator constant list.
        enumerators: Vec<Enumerator>,
        /// GCC attributes.
        attrs: Vec<Attribute>,
        /// Source location.
        span: Span,
    },

    /// `_Static_assert(condition, message);` — compile-time assertion (C11).
    ///
    /// The condition must be an integer constant expression. If it evaluates
    /// to zero, the compiler emits an error including the message string.
    StaticAssert {
        /// The constant expression condition.
        condition: Box<Expression>,
        /// The error message string literal (raw bytes for PUA fidelity).
        message: Vec<u8>,
        /// Source location from `_Static_assert` through `;`.
        span: Span,
    },

    /// Empty declaration — a lone semicolon at file or block scope.
    ///
    /// C11 §6.7 permits empty declarations. These are typically harmless
    /// and produce no semantic effect.
    Empty {
        /// Source location of the semicolon.
        span: Span,
    },

    /// Error recovery placeholder — produced when the parser encounters an
    /// unrecoverable syntax error in a declaration context and must skip
    /// ahead to continue parsing.
    Error {
        /// Source location of the erroneous token(s).
        span: Span,
    },
}

impl Declaration {
    /// Returns the source `Span` of this declaration, regardless of variant.
    pub fn span(&self) -> Span {
        match self {
            Declaration::Variable { span, .. }
            | Declaration::FunctionDef { span, .. }
            | Declaration::FunctionDecl { span, .. }
            | Declaration::Typedef { span, .. }
            | Declaration::StructDef { span, .. }
            | Declaration::UnionDef { span, .. }
            | Declaration::EnumDef { span, .. }
            | Declaration::StaticAssert { span, .. }
            | Declaration::Empty { span, .. }
            | Declaration::Error { span, .. } => *span,
        }
    }
}

// ===========================================================================
// Translation Unit (Top-Level)
// ===========================================================================

/// The root AST node — represents an entire C translation unit (source file).
///
/// A translation unit is a sequence of external declarations: function
/// definitions, variable declarations, type definitions, and preprocessor-
/// level constructs that survived macro expansion.
///
/// # Usage
///
/// The parser produces a single `TranslationUnit` per input file. The
/// semantic analyzer processes each declaration in order. The IR lowering
/// phase iterates over function definitions to produce IR functions.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationUnit {
    /// The sequence of top-level declarations in the translation unit.
    pub declarations: Vec<Declaration>,
    /// Source location spanning the entire file content.
    pub span: Span,
}

// ===========================================================================
// Utility Implementations
// ===========================================================================

impl TranslationUnit {
    /// Creates a new empty translation unit with a dummy span.
    pub fn new() -> Self {
        TranslationUnit {
            declarations: Vec::new(),
            span: Span::DUMMY,
        }
    }

    /// Creates a new translation unit with the given declarations and span.
    pub fn with_declarations(declarations: Vec<Declaration>, span: Span) -> Self {
        TranslationUnit {
            declarations,
            span,
        }
    }
}

impl Default for TranslationUnit {
    fn default() -> Self {
        TranslationUnit::new()
    }
}

impl DeclarationSpecifiers {
    /// Creates a new empty set of declaration specifiers with a dummy span.
    pub fn new() -> Self {
        DeclarationSpecifiers {
            storage_class: None,
            type_specifiers: Vec::new(),
            type_qualifiers: TypeQualifiers::default(),
            function_specifiers: FunctionSpecifiers::default(),
            alignment: None,
            attrs: Vec::new(),
            has_extension: false,
            span: Span::DUMMY,
        }
    }
}

impl Default for DeclarationSpecifiers {
    fn default() -> Self {
        DeclarationSpecifiers::new()
    }
}

impl Declarator {
    /// Creates a simple named declarator with no derived parts or attributes.
    pub fn simple(name: Symbol, span: Span) -> Self {
        Declarator {
            name: Some(name),
            derived: Vec::new(),
            attrs: Vec::new(),
            span,
        }
    }

    /// Creates an abstract (unnamed) declarator.
    pub fn abstract_decl(derived: Vec<DerivedDeclarator>, span: Span) -> Self {
        Declarator {
            name: None,
            derived,
            attrs: Vec::new(),
            span,
        }
    }
}

impl ParameterList {
    /// Creates an empty, non-variadic parameter list.
    pub fn empty(span: Span) -> Self {
        ParameterList {
            params: Vec::new(),
            variadic: false,
            span,
        }
    }
}

impl SpecifierQualifierList {
    /// Creates a new empty specifier-qualifier list with a dummy span.
    pub fn new() -> Self {
        SpecifierQualifierList {
            specifiers: Vec::new(),
            qualifiers: TypeQualifiers::default(),
            span: Span::DUMMY,
        }
    }
}

impl Default for SpecifierQualifierList {
    fn default() -> Self {
        SpecifierQualifierList::new()
    }
}

impl BinaryOperator {
    /// Returns `true` if this is an assignment operator (simple or compound).
    pub fn is_assignment(&self) -> bool {
        matches!(
            self,
            BinaryOperator::Assign
                | BinaryOperator::AddAssign
                | BinaryOperator::SubAssign
                | BinaryOperator::MulAssign
                | BinaryOperator::DivAssign
                | BinaryOperator::ModAssign
                | BinaryOperator::BitAndAssign
                | BinaryOperator::BitOrAssign
                | BinaryOperator::BitXorAssign
                | BinaryOperator::ShlAssign
                | BinaryOperator::ShrAssign
        )
    }

    /// Returns `true` if this is a comparison (relational or equality) operator.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOperator::Eq
                | BinaryOperator::Ne
                | BinaryOperator::Lt
                | BinaryOperator::Gt
                | BinaryOperator::Le
                | BinaryOperator::Ge
        )
    }

    /// Returns `true` if this is a logical operator (`&&` or `||`).
    pub fn is_logical(&self) -> bool {
        matches!(self, BinaryOperator::LogAnd | BinaryOperator::LogOr)
    }
}

impl std::fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinaryOperator::Add => "+",
            BinaryOperator::Sub => "-",
            BinaryOperator::Mul => "*",
            BinaryOperator::Div => "/",
            BinaryOperator::Mod => "%",
            BinaryOperator::BitAnd => "&",
            BinaryOperator::BitOr => "|",
            BinaryOperator::BitXor => "^",
            BinaryOperator::Shl => "<<",
            BinaryOperator::Shr => ">>",
            BinaryOperator::LogAnd => "&&",
            BinaryOperator::LogOr => "||",
            BinaryOperator::Eq => "==",
            BinaryOperator::Ne => "!=",
            BinaryOperator::Lt => "<",
            BinaryOperator::Gt => ">",
            BinaryOperator::Le => "<=",
            BinaryOperator::Ge => ">=",
            BinaryOperator::Assign => "=",
            BinaryOperator::AddAssign => "+=",
            BinaryOperator::SubAssign => "-=",
            BinaryOperator::MulAssign => "*=",
            BinaryOperator::DivAssign => "/=",
            BinaryOperator::ModAssign => "%=",
            BinaryOperator::BitAndAssign => "&=",
            BinaryOperator::BitOrAssign => "|=",
            BinaryOperator::BitXorAssign => "^=",
            BinaryOperator::ShlAssign => "<<=",
            BinaryOperator::ShrAssign => ">>=",
        };
        write!(f, "{}", s)
    }
}

impl std::fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            UnaryOperator::Plus => "+",
            UnaryOperator::Neg => "-",
            UnaryOperator::BitNot => "~",
            UnaryOperator::LogNot => "!",
            UnaryOperator::PreInc | UnaryOperator::PostInc => "++",
            UnaryOperator::PreDec | UnaryOperator::PostDec => "--",
            UnaryOperator::AddressOf => "&",
            UnaryOperator::Deref => "*",
        };
        write!(f, "{}", s)
    }
}
