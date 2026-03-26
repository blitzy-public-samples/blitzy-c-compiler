//! Abstract Syntax Tree (AST) node definitions for the BCC C11 parser.
//!
//! This is the **foundational** file of the parser module — every other parser
//! file, the semantic analyzer (`sema`), the IR lowering phase, and downstream
//! pipeline stages depend on the types defined here.
//!
//! # Design Principles
//!
//! - **Every AST node carries a [`Span`]** for source-accurate diagnostic
//!   reporting throughout the entire compilation pipeline.
//! - **All identifiers use [`Symbol`]** from the string interner for zero-cost
//!   O(1) comparison by integer handle instead of string comparison.
//! - **String literals use `Vec<u8>`** for PUA-encoded byte fidelity — non-UTF-8
//!   bytes (0x80–0xFF) survive the pipeline as Private Use Area code points and
//!   are decoded back to exact bytes during code generation.
//! - **Integer literals use `u128`** for full-range coverage of all C integer
//!   constant widths including `unsigned long long`.
//!
//! # GCC Extension Coverage
//!
//! The AST represents every GCC extension required for Linux kernel compilation:
//! - Statement expressions `({ ... })`
//! - `typeof` / `__typeof__` type deduction
//! - Computed gotos `goto *expr` and address-of-label `&&label`
//! - Case ranges `case low ... high:`
//! - Conditional operand omission `x ?: y`
//! - `__attribute__((...))` with 21+ attributes
//! - Inline assembly (AT&T syntax, constraints, clobbers, named operands, `asm goto`)
//! - Local labels `__label__`
//! - Zero-length arrays, flexible array members
//! - `_Atomic`, `_Alignas`, `_Alignof`, `_Generic`, `_Noreturn`, `_Thread_local`, `_Complex`
//! - `_Static_assert`
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules. No external crates.

use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;

// ===========================================================================
// Top-Level AST Node — TranslationUnit
// ===========================================================================

/// The root AST node representing an entire C translation unit (source file).
///
/// A translation unit is a sequence of external declarations — function
/// definitions, global variable declarations, type definitions, inline
/// assembly statements, and `_Static_assert` declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationUnit {
    /// The list of top-level declarations in source order.
    pub declarations: Vec<ExternalDeclaration>,
    /// Span covering the entire translation unit.
    pub span: Span,
}

/// A single top-level (external) declaration in a translation unit.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalDeclaration {
    /// A function definition with a body (boxed to reduce enum size).
    FunctionDefinition(Box<FunctionDefinition>),
    /// A declaration: global variable, typedef, struct/union/enum, or `_Static_assert`
    /// (boxed to reduce enum size).
    Declaration(Box<Declaration>),
    /// A top-level inline assembly statement (basic asm at file scope).
    AsmStatement(AsmStatement),
    /// An empty declaration (stray semicolon at file scope).
    Empty,
}

// ===========================================================================
// Declaration Nodes
// ===========================================================================

/// A C declaration: specifiers followed by zero or more init-declarators.
///
/// Covers variable declarations, typedef declarations, struct/union/enum
/// forward declarations, and `_Static_assert`.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    /// Declaration specifiers (storage class, type specifiers, qualifiers, etc.).
    pub specifiers: DeclarationSpecifiers,
    /// Zero or more declarators, each optionally with an initializer.
    pub declarators: Vec<InitDeclarator>,
    /// Optional `_Static_assert` data.  When present, this declaration
    /// represents a compile-time assertion rather than a variable/type
    /// declaration.  The `specifiers` and `declarators` fields are empty
    /// in this case; the assertion condition and message live here.
    pub static_assert: Option<StaticAssert>,
    /// Source span covering the entire declaration.
    pub span: Span,
}

/// A declarator paired with an optional initializer: `x = 5` or just `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct InitDeclarator {
    /// The declarator (name and type shape).
    pub declarator: Declarator,
    /// Optional initializer expression or initializer list.
    pub initializer: Option<Initializer>,
    /// GCC explicit register variable binding: `register int x asm("eax")`.
    /// Contains the register name string (e.g., "a0", "eax").
    pub asm_register: Option<String>,
    /// Source span.
    pub span: Span,
}

/// Declaration specifiers: storage class, type specifiers, qualifiers,
/// function specifiers, alignment, and GCC attributes.
///
/// These can appear in any order in C source (e.g., `static const int`
/// is the same as `int static const`).
#[derive(Debug, Clone, PartialEq)]
pub struct DeclarationSpecifiers {
    /// Optional storage class specifier (at most one, except `_Thread_local`
    /// which can combine with `static` or `extern`).
    pub storage_class: Option<StorageClass>,
    /// Type specifiers: `int`, `long`, `struct foo`, `typeof(x)`, etc.
    pub type_specifiers: Vec<TypeSpecifier>,
    /// Type qualifiers: `const`, `volatile`, `restrict`, `_Atomic`.
    pub type_qualifiers: Vec<TypeQualifier>,
    /// Function specifiers: `inline`, `_Noreturn`.
    pub function_specifiers: Vec<FunctionSpecifier>,
    /// Optional alignment specifier: `_Alignas(N)` or `_Alignas(type)`.
    pub alignment_specifier: Option<AlignmentSpecifier>,
    /// GCC `__attribute__((...))` specifiers.
    pub attributes: Vec<Attribute>,
    /// Source span covering all specifiers.
    pub span: Span,
}

/// Storage class specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// `auto` — automatic storage duration (default for locals).
    Auto,
    /// `register` — hint for register allocation.
    Register,
    /// `static` — internal linkage or static storage duration.
    Static,
    /// `extern` — external linkage.
    Extern,
    /// `typedef` — introduces a type alias.
    Typedef,
    /// `_Thread_local` — thread-local storage duration.
    ThreadLocal,
}

/// Function specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionSpecifier {
    /// `inline` — hint for function inlining.
    Inline,
    /// `_Noreturn` — function does not return to its caller.
    Noreturn,
}

/// Alignment specifier: `_Alignas(type)` or `_Alignas(constant-expression)`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentSpecifier {
    /// The alignment argument — either a type name or a constant expression.
    pub arg: AlignasArg,
    /// Source span.
    pub span: Span,
}

/// Argument to `_Alignas`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlignasArg {
    /// `_Alignas(type-name)` — alignment of the given type.
    Type(TypeName),
    /// `_Alignas(constant-expression)` — explicit alignment value.
    Expression(Box<Expression>),
}

// ===========================================================================
// Type Specifier Nodes
// ===========================================================================

/// A type specifier within a declaration or type name.
///
/// Multiple type specifiers combine per C11 §6.7.2 (e.g., `unsigned long long`).
#[derive(Debug, Clone, PartialEq)]
pub enum TypeSpecifier {
    /// `void`
    Void,
    /// `char`
    Char,
    /// `short`
    Short,
    /// `int`
    Int,
    /// `long` — may appear twice for `long long`.
    Long,
    /// `float`
    Float,
    /// `double`
    Double,
    /// `signed`
    Signed,
    /// `unsigned`
    Unsigned,
    /// `_Bool`
    Bool,
    /// `_Complex`
    Complex,
    /// `__int128` — GCC 128-bit integer extension.
    Int128,
    /// `_Float128` / `__float128` — GCC/C23 128-bit float.
    Float128,
    /// `_Float16` — IEC 60559 half-precision float.
    Float16,
    /// `_Float32` / `_Float32x` — IEC 60559 32-bit float.
    Float32,
    /// `_Float64` / `_Float64x` — IEC 60559 64-bit float.
    Float64,
    /// `struct { ... }` or `struct tag`
    Struct(StructOrUnionSpecifier),
    /// `union { ... }` or `union tag`
    Union(StructOrUnionSpecifier),
    /// `enum { ... }` or `enum tag`
    Enum(EnumSpecifier),
    /// A typedef name — an identifier previously declared with `typedef`.
    TypedefName(Symbol),
    /// `_Atomic(type-name)` — atomic type specifier form.
    Atomic(Box<TypeName>),
    /// `typeof(expr)` or `typeof(type-name)` — GCC type deduction extension.
    Typeof(TypeofArg),
    /// `__auto_type` — GCC automatic type inference (similar to C++ `auto`).
    /// The actual type is inferred from the initializer expression.
    AutoType,
}

/// Argument to `typeof` / `__typeof__`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeofArg {
    /// `typeof(expression)` — deduce type from an expression.
    Expression(Box<Expression>),
    /// `typeof(type-name)` — explicit type name (identity operation, useful in macros).
    TypeName(Box<TypeName>),
}

/// A `struct` or `union` specifier with optional tag and member list.
#[derive(Debug, Clone, PartialEq)]
pub struct StructOrUnionSpecifier {
    /// Whether this is a `struct` or `union`.
    pub kind: StructOrUnion,
    /// Optional tag name (e.g., `struct foo` → `Some(foo)`).
    pub tag: Option<Symbol>,
    /// Member declarations — `None` for forward declarations (`struct foo;`).
    pub members: Option<Vec<StructMember>>,
    /// GCC attributes (e.g., `__attribute__((packed))`).
    pub attributes: Vec<Attribute>,
    /// Source span.
    pub span: Span,
}

/// Discriminant for struct vs. union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructOrUnion {
    /// `struct`
    Struct,
    /// `union`
    Union,
}

/// A single member declaration within a struct or union body.
///
/// May contain multiple declarators (e.g., `int x, y;`), bitfields,
/// or anonymous structs/unions.
#[derive(Debug, Clone, PartialEq)]
pub struct StructMember {
    /// Specifier-qualifier list for this member.
    pub specifiers: SpecifierQualifierList,
    /// Declarators (may include bitfield widths).
    pub declarators: Vec<StructDeclarator>,
    /// GCC attributes on this member.
    pub attributes: Vec<Attribute>,
    /// Source span.
    pub span: Span,
}

/// A struct/union member declarator, optionally with a bitfield width.
#[derive(Debug, Clone, PartialEq)]
pub struct StructDeclarator {
    /// The declarator — `None` for anonymous bitfields (e.g., `int : 3;`).
    pub declarator: Option<Declarator>,
    /// Bitfield width — `None` for non-bitfield members.
    pub bit_width: Option<Box<Expression>>,
    /// Source span.
    pub span: Span,
}

/// An `enum` specifier with optional tag and enumerator list.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumSpecifier {
    /// Optional tag name.
    pub tag: Option<Symbol>,
    /// Enumerator list — `None` for forward references (`enum foo`).
    pub enumerators: Option<Vec<Enumerator>>,
    /// GCC attributes.
    pub attributes: Vec<Attribute>,
    /// Source span.
    pub span: Span,
}

/// A single enumerator within an enum definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Enumerator {
    /// The enumerator name.
    pub name: Symbol,
    /// Optional explicit value (e.g., `FOO = 42`).
    pub value: Option<Box<Expression>>,
    /// Source span.
    pub span: Span,
}

/// A specifier-qualifier list — type specifiers and qualifiers without
/// storage class specifiers. Used in struct member declarations and type names.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecifierQualifierList {
    /// Type specifiers.
    pub type_specifiers: Vec<TypeSpecifier>,
    /// Type qualifiers.
    pub type_qualifiers: Vec<TypeQualifier>,
    /// GCC attributes.
    pub attributes: Vec<Attribute>,
    /// Source span.
    pub span: Span,
}

// ===========================================================================
// Type Qualifier and Type Name Nodes
// ===========================================================================

/// A type qualifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeQualifier {
    /// `const` — object is read-only.
    Const,
    /// `volatile` — accesses are observable side effects.
    Volatile,
    /// `restrict` — pointer has no aliases (C99).
    Restrict,
    /// `_Atomic` — atomic qualifier form (without parenthesized type argument).
    Atomic,
}

/// A type name: specifier-qualifier list with an optional abstract declarator.
///
/// Used in cast expressions `(type)expr`, `sizeof(type)`, `_Alignof(type)`,
/// `_Generic` associations, compound literals `(type){init}`, and `typeof(type)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeName {
    /// Specifier-qualifier list (type specifiers + qualifiers).
    pub specifier_qualifiers: SpecifierQualifierList,
    /// Optional abstract declarator (pointer, array, function shapes without a name).
    pub abstract_declarator: Option<AbstractDeclarator>,
    /// Source span.
    pub span: Span,
}

/// An abstract declarator — a declarator without a name, used in type names.
///
/// Examples: `*` (pointer to), `*const` (const pointer to),
/// `(*)(int, float)` (pointer to function).
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractDeclarator {
    /// Optional pointer chain.
    pub pointer: Option<Pointer>,
    /// Optional direct abstract declarator (array/function shapes).
    pub direct: Option<DirectAbstractDeclarator>,
    /// Source span.
    pub span: Span,
}

/// A direct abstract declarator — array or function shape without a name.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectAbstractDeclarator {
    /// Parenthesized abstract declarator: `(abstract-declarator)`.
    Parenthesized(Box<AbstractDeclarator>),
    /// Array abstract declarator: `[size]`, `[*]`, `[static size]`.
    Array {
        /// The base direct-abstract-declarator being arrayed, if this is a
        /// suffix chain (e.g., the `(*)` in `(*)[10]`).
        base: Option<Box<DirectAbstractDeclarator>>,
        /// Optional array size expression.
        size: Option<Box<Expression>>,
        /// Type qualifiers on the array (e.g., `[const 10]`).
        qualifiers: Vec<TypeQualifier>,
        /// Whether `static` appears in the array declarator.
        is_static: bool,
    },
    /// Function abstract declarator: `(parameter-list)`.
    Function {
        /// The base direct-abstract-declarator this function suffix is
        /// applied to (e.g., the `(*)` in `(*)(int)`). `None` for
        /// standalone function type names like `void(int)`.
        base: Option<Box<DirectAbstractDeclarator>>,
        /// Parameter declarations.
        params: Vec<ParameterDeclaration>,
        /// Whether the parameter list ends with `...` (variadic).
        is_variadic: bool,
    },
}

// ===========================================================================
// Declarator Nodes
// ===========================================================================

/// A declarator: the "name and shape" part of a C declaration.
///
/// Declarators are recursive and build types "inside-out":
/// `int *(*fp)(int)` → `fp` is a pointer to a function(int) returning pointer-to-int.
#[derive(Debug, Clone, PartialEq)]
pub struct Declarator {
    /// Optional pointer chain (e.g., `*const *volatile`).
    pub pointer: Option<Pointer>,
    /// The direct declarator (identifier, array, function, or parenthesized).
    pub direct: DirectDeclarator,
    /// GCC attributes attached to this declarator.
    pub attributes: Vec<Attribute>,
    /// Source span.
    pub span: Span,
}

impl Declarator {
    /// Returns `true` if the outermost direct-declarator suffix is a function
    /// parameter list — i.e. this declarator has the shape `name(...)`.
    pub fn has_function_suffix(&self) -> bool {
        matches!(self.direct, DirectDeclarator::Function { .. })
    }
}

/// A pointer declarator component: `*` optionally followed by type qualifiers
/// and another pointer level.
#[derive(Debug, Clone, PartialEq)]
pub struct Pointer {
    /// Qualifiers on this pointer level (`const`, `volatile`, `restrict`, `_Atomic`).
    pub qualifiers: Vec<TypeQualifier>,
    /// Inner pointer for multi-level pointers (`**`).
    pub inner: Option<Box<Pointer>>,
    /// Source span.
    pub span: Span,
}

/// A direct declarator: the non-pointer part of a declarator.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectDeclarator {
    /// A simple identifier: `x`, `main`, `my_var`.
    Identifier(Symbol, Span),
    /// A parenthesized declarator: `(declarator)` for function pointer syntax.
    Parenthesized(Box<Declarator>),
    /// An array declarator: `name[size]`, `name[*]`, `name[static size]`.
    Array {
        /// The base declarator being arrayed.
        base: Box<DirectDeclarator>,
        /// Optional array size expression — `None` for `[]`.
        size: Option<Box<Expression>>,
        /// Type qualifiers (e.g., `[const N]`).
        qualifiers: Vec<TypeQualifier>,
        /// Whether `static` appears.
        is_static: bool,
        /// Whether `*` appears (VLA with unspecified size).
        is_star: bool,
        /// Source span of the array suffix.
        span: Span,
    },
    /// A function declarator: `name(params...)`.
    Function {
        /// The base declarator being called.
        base: Box<DirectDeclarator>,
        /// Parameter declarations.
        params: Vec<ParameterDeclaration>,
        /// Whether the parameter list ends with `...` (variadic).
        is_variadic: bool,
        /// Source span of the function suffix.
        span: Span,
    },
}

/// A parameter declaration in a function declarator's parameter list.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDeclaration {
    /// Declaration specifiers for the parameter.
    pub specifiers: DeclarationSpecifiers,
    /// Optional named declarator (e.g., `int x`).
    pub declarator: Option<Declarator>,
    /// Optional abstract declarator for unnamed parameters (e.g., `int *`).
    pub abstract_declarator: Option<AbstractDeclarator>,
    /// Source span.
    pub span: Span,
}

// ===========================================================================
// Function Definition Node
// ===========================================================================

/// A function definition: declaration specifiers, declarator, optional
/// K&R-style parameter declarations, and the function body.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    /// Declaration specifiers (return type, storage class, etc.).
    pub specifiers: DeclarationSpecifiers,
    /// The function declarator (name and parameter list).
    pub declarator: Declarator,
    /// K&R-style (old-style) parameter declarations between the declarator
    /// and the opening brace.
    pub old_style_params: Vec<Declaration>,
    /// The function body.
    pub body: CompoundStatement,
    /// GCC attributes on the function definition.
    pub attributes: Vec<Attribute>,
    /// Source span covering the entire function definition.
    pub span: Span,
}

// ===========================================================================
// Statement Nodes
// ===========================================================================

/// A C statement.
///
/// Covers all C11 statement forms plus GCC extensions (computed gotos, case
/// ranges, local labels, inline assembly).
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Compound statement (block): `{ ... }`.
    Compound(CompoundStatement),
    /// Expression statement: `expr;` — `None` for empty statement `;`.
    Expression(Option<Box<Expression>>),
    /// `if (condition) then_branch [else else_branch]`.
    If {
        condition: Box<Expression>,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
        span: Span,
    },
    /// `switch (condition) body`.
    Switch {
        condition: Box<Expression>,
        body: Box<Statement>,
        span: Span,
    },
    /// `while (condition) body`.
    While {
        condition: Box<Expression>,
        body: Box<Statement>,
        span: Span,
    },
    /// `do body while (condition);`.
    DoWhile {
        body: Box<Statement>,
        condition: Box<Expression>,
        span: Span,
    },
    /// `for (init; condition; increment) body`.
    For {
        init: Option<ForInit>,
        condition: Option<Box<Expression>>,
        increment: Option<Box<Expression>>,
        body: Box<Statement>,
        span: Span,
    },
    /// `goto label;` — standard goto.
    Goto { label: Symbol, span: Span },
    /// `goto *expr;` — GCC computed goto extension.
    ComputedGoto { target: Box<Expression>, span: Span },
    /// `continue;`.
    Continue { span: Span },
    /// `break;`.
    Break { span: Span },
    /// `return [expr];`.
    Return {
        value: Option<Box<Expression>>,
        span: Span,
    },
    /// `label: statement` — labeled statement.
    Labeled {
        label: Symbol,
        attributes: Vec<Attribute>,
        statement: Box<Statement>,
        span: Span,
    },
    /// `case value: statement`.
    Case {
        value: Box<Expression>,
        statement: Box<Statement>,
        span: Span,
    },
    /// `case low ... high: statement` — GCC case range extension.
    CaseRange {
        low: Box<Expression>,
        high: Box<Expression>,
        statement: Box<Statement>,
        span: Span,
    },
    /// `default: statement`.
    Default {
        statement: Box<Statement>,
        span: Span,
    },
    /// A declaration within a compound statement (C99/C11, boxed to reduce enum size).
    Declaration(Box<Declaration>),
    /// Inline assembly statement.
    Asm(AsmStatement),
    /// GCC `__label__` local label declaration.
    LocalLabel(Vec<Symbol>, Span),
}

/// A compound statement (block): `{ block-item* }`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundStatement {
    /// The block items (declarations and statements) in source order.
    pub items: Vec<BlockItem>,
    /// Source span from `{` to `}`.
    pub span: Span,
}

/// A block item within a compound statement — either a declaration or a statement.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    /// A declaration within a block (boxed to reduce enum size).
    Declaration(Box<Declaration>),
    /// A statement within a block.
    Statement(Statement),
}

/// The initialization clause of a `for` loop.
#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    /// A declaration in the init clause: `for (int i = 0; ...)` (boxed to reduce enum size).
    Declaration(Box<Declaration>),
    /// An expression in the init clause: `for (i = 0; ...)`.
    Expression(Box<Expression>),
}

// ===========================================================================
// Expression Nodes
// ===========================================================================

/// A C expression — covers all C11 expression forms plus GCC extensions.
///
/// Every variant carries a [`Span`] for source-accurate diagnostics.
/// Integer literals use `u128` for full range. String literals use `Vec<u8>`
/// for PUA-encoded byte fidelity.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    // -- Primary expressions -----------------------------------------------
    /// Integer constant: `42`, `0xFF`, `0b1010`, `100ULL`.
    IntegerLiteral {
        value: u128,
        suffix: IntegerSuffix,
        /// True when the literal was written in hex (0x) or octal (0) notation.
        /// C11 §6.4.4.1: hex/octal literals may have unsigned types even
        /// without a `u`/`U` suffix, while decimal literals cannot.
        is_hex_or_octal: bool,
        span: Span,
    },
    /// Floating-point constant: `3.14`, `1.0f`, `2.0L`.
    FloatLiteral {
        value: f64,
        suffix: FloatSuffix,
        span: Span,
    },
    /// String literal (may be adjacent-concatenated): `"hello" " world"`.
    StringLiteral {
        segments: Vec<StringSegment>,
        prefix: StringPrefix,
        span: Span,
    },
    /// Character constant: `'a'`, `L'\x00'`, `U'\U0001F600'`.
    CharLiteral {
        value: u32,
        prefix: CharPrefix,
        span: Span,
    },
    /// Identifier reference: variable, function, or enumerator name.
    Identifier { name: Symbol, span: Span },
    /// Parenthesized expression: `(expr)`.
    Parenthesized { inner: Box<Expression>, span: Span },

    // -- Postfix expressions -----------------------------------------------
    /// Array subscript: `arr[index]`.
    ArraySubscript {
        base: Box<Expression>,
        index: Box<Expression>,
        span: Span,
    },
    /// Function call: `callee(arg1, arg2, ...)`.
    FunctionCall {
        callee: Box<Expression>,
        args: Vec<Expression>,
        span: Span,
    },
    /// Struct/union member access: `object.member`.
    MemberAccess {
        object: Box<Expression>,
        member: Symbol,
        span: Span,
    },
    /// Pointer member access: `ptr->member`.
    PointerMemberAccess {
        object: Box<Expression>,
        member: Symbol,
        span: Span,
    },
    /// Postfix increment: `x++`.
    PostIncrement {
        operand: Box<Expression>,
        span: Span,
    },
    /// Postfix decrement: `x--`.
    PostDecrement {
        operand: Box<Expression>,
        span: Span,
    },

    // -- Unary expressions -------------------------------------------------
    /// Prefix increment: `++x`.
    PreIncrement {
        operand: Box<Expression>,
        span: Span,
    },
    /// Prefix decrement: `--x`.
    PreDecrement {
        operand: Box<Expression>,
        span: Span,
    },
    /// Unary operator: `&x`, `*x`, `+x`, `-x`, `~x`, `!x`.
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expression>,
        span: Span,
    },
    /// `sizeof expr` — size of an expression's type.
    SizeofExpr {
        operand: Box<Expression>,
        span: Span,
    },
    /// `sizeof(type-name)` — size of a type.
    SizeofType {
        type_name: Box<TypeName>,
        span: Span,
    },
    /// `_Alignof(type-name)` — alignment of a type.
    AlignofType {
        type_name: Box<TypeName>,
        span: Span,
    },
    /// `__alignof__(expr)` — GCC extension: alignment of an expression's type.
    AlignofExpr { expr: Box<Expression>, span: Span },

    // -- Cast expression ---------------------------------------------------
    /// `(type-name) expr` — explicit type cast.
    Cast {
        type_name: Box<TypeName>,
        operand: Box<Expression>,
        span: Span,
    },

    // -- Binary expression -------------------------------------------------
    /// Binary operator: `left op right`.
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
        span: Span,
    },

    // -- Ternary / Conditional ---------------------------------------------
    /// `condition ? then_expr : else_expr`.
    ///
    /// `then_expr` is `Option` to support the GCC conditional operand omission
    /// extension: `x ?: y` is equivalent to `x ? x : y`.
    Conditional {
        condition: Box<Expression>,
        then_expr: Option<Box<Expression>>,
        else_expr: Box<Expression>,
        span: Span,
    },

    // -- Assignment --------------------------------------------------------
    /// `target op= value` — assignment (simple or compound).
    Assignment {
        op: AssignOp,
        target: Box<Expression>,
        value: Box<Expression>,
        span: Span,
    },

    // -- Comma expression --------------------------------------------------
    /// `expr1, expr2, ...` — comma operator.
    Comma { exprs: Vec<Expression>, span: Span },

    // -- Compound literal (C11) -------------------------------------------
    /// `(type-name){ initializer-list }` — compound literal.
    CompoundLiteral {
        type_name: Box<TypeName>,
        initializer: Initializer,
        span: Span,
    },

    // -- GCC Statement expression -----------------------------------------
    /// `({ ... })` — GCC statement expression extension.
    ///
    /// The value of the expression is the value of the last
    /// expression-statement in the compound statement.
    StatementExpression {
        compound: CompoundStatement,
        span: Span,
    },

    // -- GCC Builtin calls ------------------------------------------------
    /// A call to a GCC builtin requiring special parsing.
    ///
    /// E.g., `__builtin_offsetof(type, member)`,
    /// `__builtin_choose_expr(const_expr, e1, e2)`,
    /// `__builtin_types_compatible_p(type1, type2)`.
    BuiltinCall {
        builtin: BuiltinKind,
        args: Vec<Expression>,
        span: Span,
    },

    // -- C11 _Generic selection -------------------------------------------
    /// `_Generic(controlling-expr, type: expr, ..., default: expr)`.
    Generic {
        controlling: Box<Expression>,
        associations: Vec<GenericAssociation>,
        span: Span,
    },

    // -- GCC Address-of-label ---------------------------------------------
    /// `&&label` — take the address of a label for computed gotos.
    AddressOfLabel { label: Symbol, span: Span },
}

// ===========================================================================
// Expression Support Types
// ===========================================================================

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// `&` — address-of.
    AddressOf,
    /// `*` — dereference (indirection).
    Deref,
    /// `+` — unary plus.
    Plus,
    /// `-` — unary minus (negation).
    Negate,
    /// `~` — bitwise NOT (one's complement).
    BitwiseNot,
    /// `!` — logical NOT.
    LogicalNot,
    /// `__real__` — extract real part of `_Complex` (GCC extension).
    RealPart,
    /// `__imag__` — extract imaginary part of `_Complex` (GCC extension).
    ImagPart,
}

/// Binary operators (arithmetic, bitwise, logical, relational, equality).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `+` — addition.
    Add,
    /// `-` — subtraction.
    Sub,
    /// `*` — multiplication.
    Mul,
    /// `/` — division.
    Div,
    /// `%` — remainder (modulo).
    Mod,
    /// `&` — bitwise AND.
    BitwiseAnd,
    /// `|` — bitwise OR.
    BitwiseOr,
    /// `^` — bitwise XOR.
    BitwiseXor,
    /// `<<` — left shift.
    ShiftLeft,
    /// `>>` — right shift.
    ShiftRight,
    /// `&&` — logical AND.
    LogicalAnd,
    /// `||` — logical OR.
    LogicalOr,
    /// `==` — equality.
    Equal,
    /// `!=` — inequality.
    NotEqual,
    /// `<` — less than.
    Less,
    /// `>` — greater than.
    Greater,
    /// `<=` — less than or equal.
    LessEqual,
    /// `>=` — greater than or equal.
    GreaterEqual,
}

/// Assignment operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignOp {
    /// `=` — simple assignment.
    Assign,
    /// `+=`
    AddAssign,
    /// `-=`
    SubAssign,
    /// `*=`
    MulAssign,
    /// `/=`
    DivAssign,
    /// `%=`
    ModAssign,
    /// `&=`
    AndAssign,
    /// `|=`
    OrAssign,
    /// `^=`
    XorAssign,
    /// `<<=`
    ShlAssign,
    /// `>>=`
    ShrAssign,
}

/// Integer literal suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerSuffix {
    /// No suffix — type is determined by value.
    None,
    /// `u` or `U` — unsigned.
    U,
    /// `l` or `L` — long.
    L,
    /// `ul` or `UL` — unsigned long.
    UL,
    /// `ll` or `LL` — long long.
    LL,
    /// `ull` or `ULL` — unsigned long long.
    ULL,
}

/// Floating-point literal suffix.
/// GCC extension: `i` / `j` suffix for imaginary constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSuffix {
    /// No suffix — `double`.
    None,
    /// `f` or `F` — `float`.
    F,
    /// `l` or `L` — `long double`.
    L,
    /// `i` or `j` — `_Complex double` imaginary literal (GCC extension).
    I,
    /// `fi` / `if` / `fj` / `jf` — `_Complex float` imaginary literal.
    FI,
    /// `li` / `il` / `lj` / `jl` — `_Complex long double` imaginary literal.
    LI,
}

/// String literal encoding prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringPrefix {
    /// No prefix — `char` string.
    None,
    /// `L` — wide string (`wchar_t`).
    L,
    /// `u8` — UTF-8 string.
    U8,
    /// `u` — UTF-16 string (`char16_t`).
    U16,
    /// `U` — UTF-32 string (`char32_t`).
    U32,
}

/// Character literal encoding prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CharPrefix {
    /// No prefix — `char`.
    None,
    /// `L` — wide character (`wchar_t`).
    L,
    /// `u` — UTF-16 character (`char16_t`).
    U16,
    /// `U` — UTF-32 character (`char32_t`).
    U32,
}

/// A segment of a (possibly concatenated) string literal.
///
/// Raw bytes are stored as `Vec<u8>` to preserve PUA-encoded non-UTF-8
/// bytes with exact fidelity through the compilation pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct StringSegment {
    /// Raw byte content of this string segment (PUA-decoded).
    pub value: Vec<u8>,
    /// Source span of this segment.
    pub span: Span,
}

/// A `_Generic` association: `type-name: expression` or `default: expression`.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericAssociation {
    /// The type name — `None` for the `default:` association.
    pub type_name: Option<TypeName>,
    /// The associated expression.
    pub expression: Box<Expression>,
    /// Source span.
    pub span: Span,
}

/// GCC builtin function kind.
///
/// Enumerates the ~25 GCC builtins that require special parsing or
/// compile-time evaluation (as specified in §4.3 of the AAP).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinKind {
    /// `__builtin_expect(expr, expected_value)`
    Expect,
    /// `__builtin_unreachable()`
    Unreachable,
    /// `__builtin_constant_p(expr)` — compile-time constant predicate.
    ConstantP,
    /// `__builtin_offsetof(type, member)` — byte offset of a struct member.
    Offsetof,
    /// `__builtin_types_compatible_p(type1, type2)` — type compatibility check.
    TypesCompatibleP,
    /// `__builtin_choose_expr(const_expr, expr1, expr2)` — compile-time selection.
    ChooseExpr,
    /// `__builtin_clz(x)` — count leading zeros.
    Clz,
    /// `__builtin_clzl(x)` — count leading zeros (long).
    ClzL,
    /// `__builtin_clzll(x)` — count leading zeros (long long).
    ClzLL,
    /// `__builtin_ctz(x)` — count trailing zeros.
    Ctz,
    /// `__builtin_ctzl(x)` — count trailing zeros (long).
    CtzL,
    /// `__builtin_ctzll(x)` — count trailing zeros (long long).
    CtzLL,
    /// `__builtin_popcount(x)` — population count (number of set bits).
    Popcount,
    /// `__builtin_popcountl(x)` — population count (long).
    PopcountL,
    /// `__builtin_popcountll(x)` — population count (long long).
    PopcountLL,
    /// `__builtin_bswap16(x)` — 16-bit byte swap.
    Bswap16,
    /// `__builtin_bswap32(x)` — 32-bit byte swap.
    Bswap32,
    /// `__builtin_bswap64(x)` — 64-bit byte swap.
    Bswap64,
    /// `__builtin_ffs(x)` — find first set bit.
    Ffs,
    /// `__builtin_ffsll(x)` — find first set bit (64-bit long long).
    Ffsll,
    /// `__builtin_va_start(ap, last_named)` — initialize va_list.
    VaStart,
    /// `__builtin_va_end(ap)` — clean up va_list.
    VaEnd,
    /// `__builtin_va_arg(ap, type)` — fetch next variadic argument.
    VaArg,
    /// `__builtin_va_copy(dest, src)` — copy va_list.
    VaCopy,
    /// `__builtin_frame_address(level)` — frame pointer at call depth.
    FrameAddress,
    /// `__builtin_return_address(level)` — return address at call depth.
    ReturnAddress,
    /// `__builtin_trap()` — abnormal program termination.
    Trap,
    /// `__builtin_assume_aligned(ptr, align)` — pointer alignment hint.
    AssumeAligned,
    /// `__builtin_add_overflow(a, b, result)` — checked addition.
    AddOverflow,
    /// `__builtin_sub_overflow(a, b, result)` — checked subtraction.
    SubOverflow,
    /// `__builtin_mul_overflow(a, b, result)` — checked multiplication.
    MulOverflow,
    /// `__builtin_prefetch(addr, ...)` — data prefetch hint.
    PrefetchData,
    /// `__builtin_object_size(ptr, type)` — object size at compile time.
    ObjectSize,
    /// `__builtin_extract_return_addr(addr)` — extract return address
    /// (no-op identity on most architectures).
    ExtractReturnAddr,
    /// `__builtin_abort()` — call `abort()` to terminate the program.
    Abort,
    /// `__builtin_exit(status)` — call `exit(status)` to terminate the program.
    Exit,
    /// `__builtin_memcpy(dst, src, n)` — inline memory copy.
    Memcpy,
    /// `__builtin_memset(dst, c, n)` — inline memory set.
    Memset,
    /// `__builtin_memcmp(s1, s2, n)` — inline memory compare.
    Memcmp,
    /// `__builtin_strlen(s)` — inline string length.
    Strlen,
    /// `__builtin_strcmp(s1, s2)` — inline string compare.
    Strcmp,
    /// `__builtin_strncmp(s1, s2, n)` — inline string compare with limit.
    Strncmp,
    /// `__builtin_abs(x)` — absolute value of int.
    Abs,
    /// `__builtin_labs(x)` — absolute value of long.
    Labs,
    /// `__builtin_llabs(x)` — absolute value of long long.
    Llabs,
    /// `__builtin_alloca(size)` — allocate stack memory.
    Alloca,
    /// `__builtin_inf()` — positive infinity (double).
    Inf,
    /// `__builtin_inff()` — positive infinity (float).
    Inff,
    /// `__builtin_infl()` — positive infinity (long double).
    Infl,
    /// `__builtin_nan(str)` — quiet NaN (double).
    Nan,
    /// `__builtin_nanf(str)` — quiet NaN (float).
    Nanf,
    /// `__builtin_nanl(str)` — quiet NaN (long double).
    Nanl,
    /// `__builtin_huge_val()` — positive infinity (double), alias for inf.
    HugeVal,
    /// `__builtin_huge_valf()` — positive infinity (float).
    HugeValf,
    /// `__builtin_huge_vall()` — positive infinity (long double).
    HugeVall,
    /// `__builtin_signbit(x)` — nonzero if sign bit is set (works for all float types).
    Signbit,
    /// `__builtin_isnan(x)` — nonzero if argument is NaN.
    Isnan,
    /// `__builtin_isinf(x)` — nonzero if argument is infinity.
    Isinf,
    /// `__builtin_isfinite(x)` — nonzero if argument is finite.
    Isfinite,
    /// `__builtin_isinf_sign(x)` — -1, 0, or 1 depending on inf sign.
    IsinfSign,
    /// `__builtin_copysign(x, y)` — copy sign of y to x (double).
    Copysign,
    /// `__builtin_copysignf(x, y)` — copy sign (float).
    Copysignf,
    /// `__builtin_copysignl(x, y)` — copy sign (long double).
    Copysignl,
    /// `__builtin_fabs(x)` — absolute value (double).
    Fabs,
    /// `__builtin_fabsf(x)` — absolute value (float).
    Fabsf,
    /// `__builtin_fabsl(x)` — absolute value (long double).
    Fabsl,
    /// `__builtin_classify_type(expr)` — return enum for type class.
    ClassifyType,
    /// `__builtin_constant_p` applied to a whole expression tree (for macro tricks).
    /// Maps to ConstantP semantics but can be distinguished if needed.
    BuiltinConstantP,
    /// `__sync_val_compare_and_swap(ptr, old, new)` — atomic CAS returning old val.
    SyncValCompareAndSwap,
    /// `__sync_bool_compare_and_swap(ptr, old, new)` — atomic CAS returning bool.
    SyncBoolCompareAndSwap,
    /// `__sync_fetch_and_add(ptr, val)` — atomic fetch-and-add.
    SyncFetchAndAdd,
    /// `__sync_fetch_and_sub(ptr, val)` — atomic fetch-and-sub.
    SyncFetchAndSub,
    /// `__sync_fetch_and_and(ptr, val)` — atomic fetch-and-and.
    SyncFetchAndAnd,
    /// `__sync_fetch_and_or(ptr, val)` — atomic fetch-and-or.
    SyncFetchAndOr,
    /// `__sync_fetch_and_xor(ptr, val)` — atomic fetch-and-xor.
    SyncFetchAndXor,
    /// `__sync_lock_test_and_set(ptr, val)` — atomic exchange.
    SyncLockTestAndSet,
    /// `__sync_lock_release(ptr)` — atomic store zero with release semantics.
    SyncLockRelease,
    /// `__sync_synchronize()` — full memory barrier.
    SyncSynchronize,
    /// `__atomic_load_n(ptr, order)` — C11 atomic load.
    AtomicLoadN,
    /// `__atomic_store_n(ptr, val, order)` — C11 atomic store.
    AtomicStoreN,
    /// `__atomic_exchange_n(ptr, val, order)` — C11 atomic exchange.
    AtomicExchangeN,
    /// `__atomic_compare_exchange_n(ptr, expected, desired, weak, succ, fail)` — C11 CAS.
    AtomicCompareExchangeN,
    /// `__atomic_fetch_add(ptr, val, order)` — C11 atomic fetch-and-add.
    AtomicFetchAdd,
    /// `__atomic_fetch_sub(ptr, val, order)` — C11 atomic fetch-and-sub.
    AtomicFetchSub,
    /// `__atomic_fetch_and(ptr, val, order)` — C11 atomic fetch-and-and.
    AtomicFetchAnd,
    /// `__atomic_fetch_or(ptr, val, order)` — C11 atomic fetch-and-or.
    AtomicFetchOr,
    /// `__atomic_fetch_xor(ptr, val, order)` — C11 atomic fetch-and-xor.
    AtomicFetchXor,
    /// `__builtin_mul_overflow_p(a, b, c)` — overflow check without storing result.
    MulOverflowP,
    /// `__builtin_add_overflow_p(a, b, c)` — overflow check without storing result.
    AddOverflowP,
    /// `__builtin_sub_overflow_p(a, b, c)` — overflow check without storing result.
    SubOverflowP,
}

// ===========================================================================
// Initializer Nodes
// ===========================================================================

/// An initializer for a variable or compound literal.
#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    /// A simple expression initializer: `= expr`.
    Expression(Box<Expression>),
    /// A brace-enclosed initializer list: `= { ... }`.
    List {
        /// The list of (possibly designated) initializers.
        designators_and_initializers: Vec<DesignatedInitializer>,
        /// Whether a trailing comma appears after the last initializer.
        trailing_comma: bool,
        /// Source span.
        span: Span,
    },
}

/// A single entry in an initializer list, optionally with designators.
///
/// Positional initializers have an empty `designators` vector.
/// Designated initializers have one or more designators:
/// `.field = value`, `[index] = value`, `.field.subfield = value`.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignatedInitializer {
    /// Designator chain — empty for positional initializers.
    pub designators: Vec<Designator>,
    /// The initializer value (expression or nested list).
    pub initializer: Initializer,
    /// Source span.
    pub span: Span,
}

/// A designator in a designated initializer.
#[derive(Debug, Clone, PartialEq)]
pub enum Designator {
    /// `.member` — struct/union field designation.
    Field(Symbol, Span),
    /// `[index]` — array index designation.
    Index(Box<Expression>, Span),
    /// `[low ... high]` — GCC array index range designation.
    IndexRange(Box<Expression>, Box<Expression>, Span),
}

// ===========================================================================
// Attribute Nodes
// ===========================================================================

/// A GCC `__attribute__` specification.
///
/// Represents a single attribute within `__attribute__((...))`.
/// The attribute name is interned as a [`Symbol`] for zero-cost comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// The attribute name (with `__` prefix/suffix stripped, e.g.,
    /// `__aligned__` → `aligned`).
    pub name: Symbol,
    /// Arguments to the attribute (may be empty for simple attributes).
    pub args: Vec<AttributeArg>,
    /// Source span.
    pub span: Span,
}

/// An argument to a GCC attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeArg {
    /// An identifier argument (e.g., `printf` in `format(printf, 1, 2)`).
    Identifier(Symbol, Span),
    /// An expression argument (e.g., `16` in `aligned(16)`).
    Expression(Box<Expression>),
    /// A string argument (e.g., `".data"` in `section(".data")`).
    String(Vec<u8>, Span),
    /// A type argument (used in some attributes).
    Type(TypeName),
}

// ===========================================================================
// Inline Assembly Nodes
// ===========================================================================

/// An inline assembly statement: `asm [volatile] [goto] ( template : outputs : inputs : clobbers [: goto_labels] );`.
///
/// Supports the full range of GCC inline assembly syntax required for
/// Linux kernel compilation: AT&T syntax, constraints, clobber lists,
/// named operands, `asm volatile`, and `asm goto`.
#[derive(Debug, Clone, PartialEq)]
pub struct AsmStatement {
    /// Whether `volatile` / `__volatile__` qualifier is present.
    pub is_volatile: bool,
    /// Whether `goto` qualifier is present (asm may transfer control to labels).
    pub is_goto: bool,
    /// The assembly template string (raw bytes, PUA-decoded).
    pub template: Vec<u8>,
    /// Output operands: `[name] "constraint" (expression)`.
    pub outputs: Vec<AsmOperand>,
    /// Input operands: `[name] "constraint" (expression)`.
    pub inputs: Vec<AsmOperand>,
    /// Clobber list: `"memory"`, `"cc"`, register names.
    pub clobbers: Vec<AsmClobber>,
    /// Goto labels (only if `is_goto` is true).
    pub goto_labels: Vec<Symbol>,
    /// Source span.
    pub span: Span,
}

/// An operand in an inline assembly statement (output or input).
#[derive(Debug, Clone, PartialEq)]
pub struct AsmOperand {
    /// Optional symbolic name: `[name]` for named operand references.
    pub symbolic_name: Option<Symbol>,
    /// Constraint string: `"=r"`, `"+m"`, `"i"`, etc. (raw bytes).
    pub constraint: Vec<u8>,
    /// The C expression providing/receiving the value.
    pub expression: Box<Expression>,
    /// Source span.
    pub span: Span,
}

/// A clobber entry in an inline assembly statement.
#[derive(Debug, Clone, PartialEq)]
pub struct AsmClobber {
    /// The clobber register/resource name: `"memory"`, `"cc"`, `"rax"`, etc.
    pub register: Vec<u8>,
    /// Source span.
    pub span: Span,
}

// ===========================================================================
// _Static_assert Node
// ===========================================================================

/// A `_Static_assert` declaration: compile-time assertion.
///
/// `_Static_assert(condition, "message");`
///
/// The message is optional in C23 / GCC lenient mode.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticAssert {
    /// The compile-time condition expression (must be an integer constant expression).
    pub condition: Box<Expression>,
    /// Optional diagnostic message string (raw bytes).
    pub message: Option<Vec<u8>>,
    /// Source span.
    pub span: Span,
}

// ===========================================================================
// Convenience Implementations
// ===========================================================================

impl Expression {
    /// Returns the [`Span`] of this expression, regardless of variant.
    ///
    /// This is a convenience method used throughout the pipeline for
    /// attaching source locations to diagnostics generated from expressions.
    pub fn span(&self) -> Span {
        match self {
            Expression::IntegerLiteral { span, .. }
            | Expression::FloatLiteral { span, .. }
            | Expression::StringLiteral { span, .. }
            | Expression::CharLiteral { span, .. }
            | Expression::Identifier { span, .. }
            | Expression::Parenthesized { span, .. }
            | Expression::ArraySubscript { span, .. }
            | Expression::FunctionCall { span, .. }
            | Expression::MemberAccess { span, .. }
            | Expression::PointerMemberAccess { span, .. }
            | Expression::PostIncrement { span, .. }
            | Expression::PostDecrement { span, .. }
            | Expression::PreIncrement { span, .. }
            | Expression::PreDecrement { span, .. }
            | Expression::UnaryOp { span, .. }
            | Expression::SizeofExpr { span, .. }
            | Expression::SizeofType { span, .. }
            | Expression::AlignofType { span, .. }
            | Expression::AlignofExpr { span, .. }
            | Expression::Cast { span, .. }
            | Expression::Binary { span, .. }
            | Expression::Conditional { span, .. }
            | Expression::Assignment { span, .. }
            | Expression::Comma { span, .. }
            | Expression::CompoundLiteral { span, .. }
            | Expression::StatementExpression { span, .. }
            | Expression::BuiltinCall { span, .. }
            | Expression::Generic { span, .. }
            | Expression::AddressOfLabel { span, .. } => *span,
        }
    }
}

impl Statement {
    /// Returns the [`Span`] of this statement, regardless of variant.
    ///
    /// For variants that embed the span differently (e.g., `Expression` where
    /// the span is derived from the inner expression), this method ensures
    /// a consistent interface.
    pub fn span(&self) -> Span {
        match self {
            Statement::Compound(cs) => cs.span,
            Statement::Expression(Some(expr)) => expr.span(),
            Statement::Expression(None) => Span::dummy(),
            Statement::If { span, .. }
            | Statement::Switch { span, .. }
            | Statement::While { span, .. }
            | Statement::DoWhile { span, .. }
            | Statement::For { span, .. }
            | Statement::Goto { span, .. }
            | Statement::ComputedGoto { span, .. }
            | Statement::Continue { span }
            | Statement::Break { span }
            | Statement::Return { span, .. }
            | Statement::Labeled { span, .. }
            | Statement::Case { span, .. }
            | Statement::CaseRange { span, .. }
            | Statement::Default { span, .. } => *span,
            Statement::Declaration(decl) => decl.span,
            Statement::Asm(asm) => asm.span,
            Statement::LocalLabel(_, span) => *span,
        }
    }
}

impl TranslationUnit {
    /// Creates a new, empty translation unit with a dummy span.
    pub fn new() -> Self {
        TranslationUnit {
            declarations: Vec::new(),
            span: Span::dummy(),
        }
    }
}

impl Default for TranslationUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl DeclarationSpecifiers {
    /// Creates empty declaration specifiers with a dummy span.
    pub fn empty() -> Self {
        DeclarationSpecifiers {
            storage_class: None,
            type_specifiers: Vec::new(),
            type_qualifiers: Vec::new(),
            function_specifiers: Vec::new(),
            alignment_specifier: None,
            attributes: Vec::new(),
            span: Span::dummy(),
        }
    }
}

impl CompoundStatement {
    /// Creates an empty compound statement with a dummy span.
    pub fn empty() -> Self {
        CompoundStatement {
            items: Vec::new(),
            span: Span::dummy(),
        }
    }
}

impl SpecifierQualifierList {
    /// Creates an empty specifier-qualifier list with a dummy span.
    pub fn empty() -> Self {
        SpecifierQualifierList {
            type_specifiers: Vec::new(),
            type_qualifiers: Vec::new(),
            attributes: Vec::new(),
            span: Span::dummy(),
        }
    }
}
