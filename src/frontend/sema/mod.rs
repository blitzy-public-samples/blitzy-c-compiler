//! Semantic analysis module — Phase 5 of the BCC compilation pipeline.
//!
//! This module drives the entire Phase 5 semantic analysis pass, transforming
//! the parser-produced AST (from [`crate::frontend::parser`]) into a
//! semantically validated, type-annotated representation consumed by Phase 6
//! IR lowering ([`crate::ir::lowering`]).
//!
//! # Pipeline Integration
//!
//! ```text
//! Phase 4 (Parser) ──► Phase 5 (Sema) ──► Phase 6 (IR Lowering)
//!   TranslationUnit      SemanticAnalyzer     CheckedTranslationUnit
//!                        ├─ type_checker       ├─ CheckedDeclaration
//!                        ├─ scope              ├─ SymbolTable
//!                        ├─ symbol_table       └─ ScopeStack
//!                        ├─ constant_eval
//!                        ├─ builtin_eval
//!                        ├─ initializer
//!                        └─ attribute_handler
//! ```
//!
//! # Submodules
//!
//! - [`type_checker`]: Core type inference and validation for expressions
//!   and statements (integer promotion, usual arithmetic conversions,
//!   assignment compatibility, scalar checks).
//! - [`scope`]: Lexical scope management — block, function, file, and global
//!   scopes; separate tag namespace for struct/union/enum; label namespace
//!   with GCC `__label__` support.
//! - [`symbol_table`]: Symbol storage, declaration/definition merging,
//!   linkage resolution (C11 §6.2.2), tentative definitions (C11 §6.9.2),
//!   and GCC attribute tracking.
//! - [`constant_eval`]: Compile-time constant expression evaluation for
//!   array sizes, case values, enum values, `_Static_assert` conditions,
//!   and bitfield widths.
//! - [`builtin_eval`]: GCC `__builtin_*` evaluation — compile-time
//!   (`__builtin_constant_p`, `__builtin_types_compatible_p`) and
//!   runtime-deferred (`__builtin_clz`, `__builtin_bswap*`).
//! - [`initializer`]: Designated initializer semantic analysis — out-of-order
//!   field designation, nested designation, brace elision, implicit zero-init.
//! - [`attribute_handler`]: GCC `__attribute__((...))` validation and
//!   propagation to symbols and types (aligned, packed, section, weak,
//!   visibility, constructor, destructor, etc.).
//!
//! # Dependencies
//!
//! The semantic analyzer depends on:
//! - [`crate::common::types`] for `CType`, `QualifiedType`, `TypeQualifiers`
//! - [`crate::common::type_builder`] for struct/union layout and type construction
//! - [`crate::common::target`] for architecture-dependent sizeof/alignof
//! - [`crate::common::diagnostics`] for error/warning/note reporting
//! - [`crate::common::source_map`] for source location resolution
//! - [`crate::common::string_interner`] for identifier interning
//! - [`crate::common::fx_hash`] for FxHashMap/FxHashSet collections
//! - [`crate::frontend::parser::ast`] for the AST node hierarchy
//!
//! The semantic analyzer does NOT depend on [`crate::ir`] or [`crate::backend`].

// ============================================================================
// Submodule declarations
// ============================================================================

pub mod attribute_handler;
pub mod builtin_eval;
pub mod constant_eval;
pub mod initializer;
pub mod scope;
pub mod symbol_table;
pub mod type_checker;

// ============================================================================
// Re-exports for external consumers of the sema module
// ============================================================================

pub use attribute_handler::ValidatedAttribute;
pub use builtin_eval::BuiltinResult;
pub use constant_eval::ConstValue;
pub use initializer::CheckedInitializer;
pub use scope::ScopeStack;
pub use symbol_table::{
    resolve_linkage, Linkage, StorageClass, SymbolAttributes, SymbolEntry, SymbolId, SymbolTable,
    VisibilityKind,
};
pub use type_checker::TypedExpression;

// ============================================================================
// Internal imports
// ============================================================================

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::source_map::SourceMap;
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;
use crate::common::type_builder::{
    compute_struct_layout, types_compatible, StructLayout, TypeBuilder,
};
use crate::common::types::{CType, FieldDef};

use crate::frontend::parser::ast::{
    AbstractDeclarator, Attribute, BlockItem, Declaration, DeclarationSpecifiers, Declarator,
    DerivedDeclarator, Enumerator, Expression, FieldDeclaration, ForInit, InitDeclarator,
    ParameterList, SpecifierQualifierList, Statement, StorageClass as AstStorageClass,
    TranslationUnit, TypeName, TypeSpecifier, TypeofOperand,
};

// Sibling module imports (private use within the driver)
use attribute_handler::{
    propagate_to_symbol, propagate_to_type, validate_attributes, AttributeTargetKind,
};
use builtin_eval::evaluate_builtin;
use constant_eval::{evaluate_constant_expression_with_resolver, evaluate_static_assert};
use initializer::analyze_initializer;
use scope::{ScopeLevel, TagEntry, TagKind};
use type_checker::check_expression;

// ============================================================================
// Checked AST output types — Phase 5 output consumed by Phase 6 (IR lowering)
// ============================================================================

/// A semantically validated and type-annotated declaration.
///
/// `CheckedDeclaration` mirrors the parser's [`Declaration`] variants but
/// carries fully resolved types, symbol IDs, evaluated constant values,
/// and validated initializers. Each variant has been through complete
/// semantic analysis including type checking, scope resolution, linkage
/// determination, and attribute validation.
///
/// # Consumers
///
/// The IR lowering pass (`crate::ir::lowering`) pattern-matches on
/// `CheckedDeclaration` variants to emit IR instructions.
#[derive(Clone, Debug)]
pub enum CheckedDeclaration {
    /// A variable declaration with resolved type, optional checked initializer,
    /// and symbol table entry.
    Variable {
        /// Symbol table handle for this variable.
        symbol_id: SymbolId,
        /// Fully resolved C type of the variable.
        ty: CType,
        /// Checked and type-validated initializer (if present).
        init: Option<CheckedInitializer>,
        /// Resolved linkage class (external, internal, none).
        linkage: Linkage,
        /// Storage class specifier.
        storage_class: StorageClass,
        /// Validated GCC attributes.
        attrs: Vec<ValidatedAttribute>,
        /// Source location of the declaration.
        span: Span,
    },

    /// A function definition with a fully analyzed body.
    FunctionDef {
        /// Symbol table handle for this function.
        symbol_id: SymbolId,
        /// Function name (interned).
        name: Symbol,
        /// Resolved return type.
        return_ty: CType,
        /// Parameter symbols (each inserted into the symbol table).
        params: Vec<CheckedParameter>,
        /// Whether the function is variadic.
        variadic: bool,
        /// The semantically validated function body.
        body: Box<Statement>,
        /// Resolved linkage class.
        linkage: Linkage,
        /// Storage class specifier.
        storage_class: StorageClass,
        /// Validated GCC attributes.
        attrs: Vec<ValidatedAttribute>,
        /// Source location.
        span: Span,
    },

    /// A forward function declaration (prototype without body).
    FunctionDecl {
        /// Symbol table handle.
        symbol_id: SymbolId,
        /// Function name (interned).
        name: Symbol,
        /// Resolved return type.
        return_ty: CType,
        /// Parameter types.
        param_types: Vec<CType>,
        /// Whether the function is variadic.
        variadic: bool,
        /// Resolved linkage class.
        linkage: Linkage,
        /// Storage class specifier.
        storage_class: StorageClass,
        /// Validated GCC attributes.
        attrs: Vec<ValidatedAttribute>,
        /// Source location.
        span: Span,
    },

    /// A typedef — type alias declaration.
    Typedef {
        /// Symbol table handle for the typedef name.
        symbol_id: SymbolId,
        /// The typedef name (interned).
        name: Symbol,
        /// The resolved underlying type.
        ty: CType,
        /// Source location.
        span: Span,
    },

    /// A struct type definition.
    StructDef {
        /// Optional struct tag name (interned).
        name: Option<Symbol>,
        /// Resolved CType for the struct (with field types resolved).
        ty: CType,
        /// Struct layout (size, alignment, field offsets).
        layout: StructLayout,
        /// Validated GCC attributes.
        attrs: Vec<ValidatedAttribute>,
        /// Source location.
        span: Span,
    },

    /// A union type definition.
    UnionDef {
        /// Optional union tag name (interned).
        name: Option<Symbol>,
        /// Resolved CType for the union.
        ty: CType,
        /// Validated GCC attributes.
        attrs: Vec<ValidatedAttribute>,
        /// Source location.
        span: Span,
    },

    /// An enum type definition with evaluated enumerator values.
    EnumDef {
        /// Optional enum tag name (interned).
        name: Option<Symbol>,
        /// Resolved CType for the enum.
        ty: CType,
        /// Evaluated enumerator constants as (name, value) pairs.
        enumerators: Vec<(Symbol, i64)>,
        /// Validated GCC attributes.
        attrs: Vec<ValidatedAttribute>,
        /// Source location.
        span: Span,
    },

    /// A `_Static_assert` that has been evaluated and passed.
    ///
    /// Only assertions that evaluate to non-zero survive semantic analysis;
    /// failures produce a diagnostic error.
    StaticAssert {
        /// Source location.
        span: Span,
    },

    /// An empty declaration — a lone semicolon.
    Empty {
        /// Source location.
        span: Span,
    },
}

/// A parameter in a checked function definition.
///
/// Each parameter has a resolved type and an optional symbol table entry
/// (parameters may be unnamed in prototypes).
#[derive(Clone, Debug)]
pub struct CheckedParameter {
    /// Optional symbol table handle (unnamed parameters have `None`).
    pub symbol_id: Option<SymbolId>,
    /// The parameter name (interned), or `None` if unnamed.
    pub name: Option<Symbol>,
    /// The resolved C type of the parameter.
    pub ty: CType,
    /// Source location of the parameter.
    pub span: Span,
}

/// The complete output of Phase 5 semantic analysis.
///
/// Contains the list of checked declarations, the symbol table with all
/// declared entities, and the scope stack (preserved for potential cross-
/// module analysis). This struct is the primary input to Phase 6 IR lowering.
pub struct CheckedTranslationUnit {
    /// All declarations, semantically validated and type-annotated.
    pub declarations: Vec<CheckedDeclaration>,
    /// The symbol table containing every declared identifier in the TU.
    pub symbol_table: SymbolTable,
    /// The scope stack (after file-scope analysis).
    pub scope: ScopeStack,
}

// ============================================================================
// SemanticAnalyzer — the Phase 5 driver
// ============================================================================

/// The Phase 5 semantic analysis driver.
///
/// `SemanticAnalyzer` maintains all state needed for a single translation
/// unit's semantic analysis: scope stack, symbol table, label tracking,
/// loop/switch nesting context, and the current function's return type.
///
/// # Lifetime
///
/// The `'a` lifetime binds to the diagnostic engine, source map, interner,
/// and target info — all of which outlive the analysis pass.
///
/// # Usage
///
/// ```ignore
/// let mut analyzer = SemanticAnalyzer::new(&mut diag, &source_map, &mut interner, &target);
/// let checked = analyzer.analyze(&translation_unit)?;
/// ```
pub struct SemanticAnalyzer<'a> {
    /// Lexical scope management — nested block, function, file, global scopes.
    scope_stack: ScopeStack,

    /// Symbol table — all declared identifiers within this translation unit.
    symbol_table: SymbolTable,

    /// Diagnostic engine for error, warning, and note reporting.
    diagnostics: &'a mut DiagnosticEngine,

    /// Source map for resolving byte offsets to file/line/column.
    /// Used by submodule calls for diagnostic location resolution.
    #[allow(dead_code)]
    source_map: &'a SourceMap,

    /// String interner for identifier allocation and resolution.
    interner: &'a mut Interner,

    /// Target architecture for sizeof/alignof/ABI computations.
    target: &'a Target,

    /// The return type of the current function being analyzed.
    /// `None` when not inside a function body.
    current_function_return_type: Option<CType>,

    /// `true` when the analyzer is inside a loop body (while/do-while/for).
    /// Used for break/continue context validation.
    in_loop: bool,

    /// `true` when the analyzer is inside a switch statement.
    /// Used for case/default/break context validation.
    in_switch: bool,

    /// Label tracking for the current function — maps label names to
    /// their definition/reference status. Used for detecting undefined
    /// and duplicate labels at function exit.
    function_labels: FxHashMap<Symbol, LabelInfo>,

    /// Case value tracking for the current switch statement — detects
    /// duplicate case values.
    switch_case_values: Vec<FxHashSet<i128>>,
}

/// Tracks whether a label has been defined and/or referenced within a function.
#[derive(Clone, Debug)]
struct LabelInfo {
    /// `true` if the label statement (`name:`) has been encountered.
    is_defined: bool,
    /// `true` if a `goto name;` referencing this label has been encountered.
    is_referenced: bool,
    /// Source location of the label definition (or first reference if undefined).
    span: Span,
}

impl<'a> SemanticAnalyzer<'a> {
    /// Creates a new semantic analyzer for a translation unit.
    ///
    /// Initializes the scope stack with a global scope and an empty symbol
    /// table. The analyzer is ready to process declarations via [`analyze`].
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Mutable reference to the diagnostic engine.
    /// * `source_map` — Reference to the source file tracker.
    /// * `interner` — Mutable reference to the string interner.
    /// * `target` — Reference to the target architecture configuration.
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        source_map: &'a SourceMap,
        interner: &'a mut Interner,
        target: &'a Target,
    ) -> Self {
        SemanticAnalyzer {
            scope_stack: ScopeStack::new(),
            symbol_table: SymbolTable::new(),
            diagnostics,
            source_map,
            interner,
            target,
            current_function_return_type: None,
            in_loop: false,
            in_switch: false,
            function_labels: FxHashMap::default(),
            switch_case_values: Vec::new(),
        }
    }

    // ====================================================================
    // Top-level entry point
    // ====================================================================

    /// Analyzes an entire translation unit (the parser's output).
    ///
    /// This is the main entry point for Phase 5. It pushes a file scope,
    /// iterates over all top-level declarations, performs semantic analysis,
    /// then finalizes tentative definitions and validates label usage.
    ///
    /// # Returns
    ///
    /// - `Ok(CheckedTranslationUnit)` — the semantically validated output.
    /// - `Err(())` — one or more errors were emitted during analysis.
    #[allow(clippy::result_unit_err)]
    pub fn analyze(&mut self, tu: &TranslationUnit) -> Result<CheckedTranslationUnit, ()> {
        // Push file scope (one level inside the global scope already
        // created by ScopeStack::new()).
        self.scope_stack.push(ScopeLevel::File);

        let mut checked_decls = Vec::with_capacity(tu.declarations.len());

        // Process each top-level declaration in source order.
        for decl in &tu.declarations {
            match self.analyze_declaration(decl) {
                Ok(checked) => checked_decls.push(checked),
                Err(()) => {
                    // Error already emitted by analyze_declaration.
                    // Continue processing remaining declarations to report
                    // as many errors as possible in a single pass.
                }
            }
        }

        // Finalize tentative definitions (C11 §6.9.2): file-scope variable
        // declarations without initializers become definitions if no explicit
        // definition appeared.
        self.symbol_table.finalize_tentative_definitions();

        // Pop file scope.
        self.scope_stack.pop();

        // If any errors were emitted during analysis, report failure.
        if self.diagnostics.has_errors() {
            return Err(());
        }

        // Transfer ownership of the symbol table and scope stack into the
        // output structure. We use mem::take to move them out, leaving
        // empty defaults behind in the analyzer (which is about to be dropped).
        let symbol_table = std::mem::take(&mut self.symbol_table);
        let scope = std::mem::take(&mut self.scope_stack);

        Ok(CheckedTranslationUnit {
            declarations: checked_decls,
            symbol_table,
            scope,
        })
    }

    // ====================================================================
    // Declaration analysis (public)
    // ====================================================================

    /// Analyzes a single top-level or block-scope declaration.
    ///
    /// Dispatches to variant-specific handlers and returns a
    /// `CheckedDeclaration` on success.
    #[allow(clippy::result_unit_err)]
    pub fn analyze_declaration(&mut self, decl: &Declaration) -> Result<CheckedDeclaration, ()> {
        match decl {
            Declaration::Variable {
                specifiers,
                declarators,
                attrs,
                span,
            } => self.analyze_variable_decl(specifiers, declarators, attrs, *span),

            Declaration::FunctionDef {
                specifiers,
                declarator,
                attrs,
                body,
                span,
            } => self.analyze_function_def(specifiers, declarator, attrs, body, *span),

            Declaration::FunctionDecl {
                specifiers,
                declarator,
                attrs,
                span,
            } => self.analyze_function_decl(specifiers, declarator, attrs, *span),

            Declaration::Typedef {
                specifiers,
                declarators,
                attrs,
                span,
            } => self.analyze_typedef(specifiers, declarators, attrs, *span),

            Declaration::StructDef {
                name,
                fields,
                attrs,
                span,
            } => self.analyze_struct_def(*name, fields, attrs, *span),

            Declaration::UnionDef {
                name,
                fields,
                attrs,
                span,
            } => self.analyze_union_def(*name, fields, attrs, *span),

            Declaration::EnumDef {
                name,
                enumerators,
                attrs,
                span,
            } => self.analyze_enum_def(*name, enumerators, attrs, *span),

            Declaration::StaticAssert {
                condition,
                message,
                span,
            } => {
                evaluate_static_assert(condition, message, self.diagnostics, self.target)?;
                Ok(CheckedDeclaration::StaticAssert { span: *span })
            }

            Declaration::Empty { span } => Ok(CheckedDeclaration::Empty { span: *span }),

            Declaration::Error { span } => {
                self.diagnostics
                    .error(*span, "invalid declaration".to_string());
                Err(())
            }
        }
    }

    // ====================================================================
    // Expression analysis (public)
    // ====================================================================

    /// Analyzes an expression, producing a type-annotated result.
    ///
    /// Delegates the heavy lifting to [`type_checker::check_expression`],
    /// which performs type inference, integer promotion, usual arithmetic
    /// conversions, and lvalue-to-rvalue conversions.
    ///
    /// For `__builtin_*` function calls, the call is first intercepted and
    /// routed to [`builtin_eval::evaluate_builtin`].
    #[allow(clippy::result_unit_err)]
    pub fn analyze_expression(&mut self, expr: &Expression) -> Result<TypedExpression, ()> {
        // Check for builtin function calls before general type-checking.
        if let Expression::FunctionCall { callee, args, span } = expr {
            if let Expression::Identifier { name, .. } = callee.as_ref() {
                let name_str = self.interner.resolve(*name);
                if name_str.starts_with("__builtin_") {
                    let result = evaluate_builtin(
                        *name,
                        args,
                        self.diagnostics,
                        self.interner,
                        self.target,
                    )?;
                    // Return a TypedExpression wrapping the builtin result.
                    let result_ty = result.ty();
                    return Ok(TypedExpression {
                        expr: expr.clone(),
                        ty: result_ty,
                        is_lvalue: false,
                        is_constant: matches!(result, BuiltinResult::CompileTimeValue(_)),
                        span: *span,
                    });
                }
            }
        }

        // General expression type-checking.
        let typed = check_expression(
            expr,
            &self.scope_stack,
            &self.symbol_table,
            self.target,
            self.diagnostics,
            self.current_function_return_type.as_ref(),
        );
        Ok(typed)
    }

    // ====================================================================
    // Statement analysis (public)
    // ====================================================================

    /// Analyzes a statement, validating type constraints, control flow
    /// context (break/continue legality, return type consistency), and
    /// label definitions.
    #[allow(clippy::result_unit_err)]
    pub fn analyze_statement(&mut self, stmt: &Statement) -> Result<(), ()> {
        match stmt {
            Statement::Compound { items, span: _ } => {
                self.scope_stack.push(ScopeLevel::Block);
                for item in items {
                    match item {
                        BlockItem::Declaration(decl) => {
                            // Block-scope declaration — analyze and discard
                            // the checked value (it is captured by the
                            // symbol table).
                            let _ = self.analyze_declaration(decl);
                        }
                        BlockItem::Statement(sub_stmt) => {
                            let _ = self.analyze_statement(sub_stmt);
                        }
                    }
                }
                self.scope_stack.pop();
                Ok(())
            }

            Statement::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let cond_typed = self.analyze_expression(condition)?;
                if !cond_typed.ty.is_scalar() {
                    self.diagnostics.error(
                        *span,
                        "condition of 'if' statement must be a scalar type".to_string(),
                    );
                }
                self.analyze_statement(then_branch)?;
                if let Some(else_stmt) = else_branch {
                    self.analyze_statement(else_stmt)?;
                }
                Ok(())
            }

            Statement::While {
                condition,
                body,
                span,
            } => {
                let cond_typed = self.analyze_expression(condition)?;
                if !cond_typed.ty.is_scalar() {
                    self.diagnostics.error(
                        *span,
                        "condition of 'while' statement must be a scalar type".to_string(),
                    );
                }
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body)?;
                self.in_loop = prev_in_loop;
                Ok(())
            }

            Statement::DoWhile {
                body,
                condition,
                span,
            } => {
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body)?;
                self.in_loop = prev_in_loop;
                let cond_typed = self.analyze_expression(condition)?;
                if !cond_typed.ty.is_scalar() {
                    self.diagnostics.error(
                        *span,
                        "condition of 'do-while' statement must be a scalar type".to_string(),
                    );
                }
                Ok(())
            }

            Statement::For {
                init,
                condition,
                increment,
                body,
                span,
            } => {
                self.scope_stack.push(ScopeLevel::Block);
                // Init clause
                if let Some(for_init) = init {
                    match for_init {
                        ForInit::Declaration(decl) => {
                            let _ = self.analyze_declaration(decl);
                        }
                        ForInit::Expression(expr) => {
                            let _ = self.analyze_expression(expr);
                        }
                    }
                }
                // Condition
                if let Some(cond) = condition {
                    let cond_typed = self.analyze_expression(cond)?;
                    if !cond_typed.ty.is_scalar() {
                        self.diagnostics.error(
                            *span,
                            "condition of 'for' statement must be a scalar type".to_string(),
                        );
                    }
                }
                // Increment
                if let Some(inc) = increment {
                    let _ = self.analyze_expression(inc);
                }
                // Body
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body)?;
                self.in_loop = prev_in_loop;
                self.scope_stack.pop();
                Ok(())
            }

            Statement::Switch {
                expression,
                body,
                span,
            } => {
                let expr_typed = self.analyze_expression(expression)?;
                if !expr_typed.ty.is_integer() {
                    self.diagnostics.error(
                        *span,
                        "switch expression must be of integer type".to_string(),
                    );
                }
                let prev_in_switch = self.in_switch;
                self.in_switch = true;
                // Push a new set of case values for this switch.
                self.switch_case_values.push(FxHashSet::default());
                self.analyze_statement(body)?;
                self.switch_case_values.pop();
                self.in_switch = prev_in_switch;
                Ok(())
            }

            Statement::Case { value, body, span } => {
                if !self.in_switch {
                    self.diagnostics.error(
                        *span,
                        "'case' label not within a switch statement".to_string(),
                    );
                }
                // Evaluate case value as integer constant expression.
                {
                    let resolver = make_typedef_resolver(&self.scope_stack, &self.symbol_table);
                    match evaluate_constant_expression_with_resolver(value, self.diagnostics, self.target, &resolver) {
                        Ok(const_val) => {
                            if let Some(int_val) = const_val.as_integer() {
                                // Check for duplicate case values.
                                if let Some(case_set) = self.switch_case_values.last_mut() {
                                    if !case_set.insert(int_val) {
                                        self.diagnostics.warning(
                                            *span,
                                            format!("duplicate case value '{}'", int_val),
                                        );
                                    }
                                }
                            } else {
                                self.diagnostics.error(
                                    *span,
                                    "case value must be an integer constant expression".to_string(),
                                );
                            }
                        }
                        Err(()) => {
                            // Error already reported by evaluate_constant_expression.
                        }
                    }
                }
                self.analyze_statement(body)?;
                Ok(())
            }

            Statement::Default { body, span } => {
                if !self.in_switch {
                    self.diagnostics.error(
                        *span,
                        "'default' label not within a switch statement".to_string(),
                    );
                }
                self.analyze_statement(body)?;
                Ok(())
            }

            Statement::Break { span } => {
                if !self.in_loop && !self.in_switch {
                    self.diagnostics.error(
                        *span,
                        "'break' statement not in loop or switch statement".to_string(),
                    );
                }
                Ok(())
            }

            Statement::Continue { span } => {
                if !self.in_loop {
                    self.diagnostics.error(
                        *span,
                        "'continue' statement not in loop statement".to_string(),
                    );
                }
                Ok(())
            }

            Statement::Return { value, span } => {
                // Clone the return type to avoid borrowing self immutably
                // while we need a mutable borrow for analyze_expression.
                let ret_ty_clone = self.current_function_return_type.clone();
                match (&ret_ty_clone, value) {
                    (Some(ret_ty), Some(expr)) => {
                        let typed = self.analyze_expression(expr)?;
                        if ret_ty.is_void() {
                            self.diagnostics.warning(
                                *span,
                                "returning a value from a void function".to_string(),
                            );
                        } else if !types_compatible(&typed.ty, ret_ty) {
                            // Allow implicit conversions between arithmetic types
                            if !(typed.ty.is_arithmetic() && ret_ty.is_arithmetic()
                                || typed.ty.is_pointer() && ret_ty.is_pointer())
                            {
                                self.diagnostics.warning(
                                    *span,
                                    format!(
                                        "incompatible return type: expected '{:?}', found '{:?}'",
                                        ret_ty, typed.ty
                                    ),
                                );
                            }
                        }
                    }
                    (Some(ret_ty), None) => {
                        if !ret_ty.is_void() {
                            self.diagnostics.warning(
                                *span,
                                "non-void function should return a value".to_string(),
                            );
                        }
                    }
                    (None, _) => {
                        // Not inside a function — error.
                        self.diagnostics.error(
                            *span,
                            "'return' statement outside of function body".to_string(),
                        );
                    }
                }
                Ok(())
            }

            Statement::Goto { label, span } => {
                // Register a label reference.
                self.function_labels
                    .entry(*label)
                    .and_modify(|info| info.is_referenced = true)
                    .or_insert(LabelInfo {
                        is_defined: false,
                        is_referenced: true,
                        span: *span,
                    });
                Ok(())
            }

            Statement::ComputedGoto { target, span } => {
                let typed = self.analyze_expression(target)?;
                // The target of a computed goto must be void*.
                if !typed.ty.is_pointer() {
                    self.diagnostics.error(
                        *span,
                        "argument to computed goto must be a pointer type (void *)".to_string(),
                    );
                }
                Ok(())
            }

            Statement::Labeled {
                label,
                attrs: _,
                body,
                span,
            } => {
                // Define the label in the current function.
                let entry = self.function_labels.entry(*label).or_insert(LabelInfo {
                    is_defined: false,
                    is_referenced: false,
                    span: *span,
                });
                if entry.is_defined {
                    self.diagnostics.error(
                        *span,
                        format!("duplicate label '{}'", self.interner.resolve(*label)),
                    );
                } else {
                    entry.is_defined = true;
                    entry.span = *span;
                }
                // Also register in scope stack for lookups.
                if self.scope_stack.define_label(*label, *span).is_err() {
                    self.diagnostics.error(
                        *span,
                        format!("duplicate label '{}'", self.interner.resolve(*label)),
                    );
                }

                self.analyze_statement(body)?;
                Ok(())
            }

            Statement::Expression { expr, span: _ } => {
                let _ = self.analyze_expression(expr);
                Ok(())
            }

            Statement::Null { .. } => Ok(()),

            Statement::Error { span } => {
                self.diagnostics
                    .error(*span, "invalid statement".to_string());
                Err(())
            }

            Statement::Asm(_) => {
                // Inline assembly is validated during IR lowering (Phase 6).
                // At the semantic level we accept it as-is.
                Ok(())
            }

            Statement::CaseRange {
                low,
                high,
                body,
                span,
            } => {
                if !self.in_switch {
                    self.diagnostics.error(
                        *span,
                        "'case' range not within a switch statement".to_string(),
                    );
                }
                // Evaluate both range bounds.
                {
                    let resolver = make_typedef_resolver(&self.scope_stack, &self.symbol_table);
                    if let Ok(low_val) =
                        evaluate_constant_expression_with_resolver(low, self.diagnostics, self.target, &resolver)
                    {
                        if let Ok(high_val) =
                            evaluate_constant_expression_with_resolver(high, self.diagnostics, self.target, &resolver)
                        {
                            if let (Some(lo), Some(hi)) = (low_val.as_integer(), high_val.as_integer())
                            {
                                if lo > hi {
                                    self.diagnostics
                                        .error(*span, format!("empty case range ({} ... {})", lo, hi));
                                }
                                // Insert all values in range for duplicate detection.
                                if let Some(case_set) = self.switch_case_values.last_mut() {
                                    let range_size = (hi - lo + 1).min(1024);
                                    for v in lo..=(lo + range_size - 1) {
                                        if !case_set.insert(v) {
                                            self.diagnostics.warning(
                                                *span,
                                                format!("duplicate case value '{}' in range", v),
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                self.analyze_statement(body)?;
                Ok(())
            }
        }
    }

    // ====================================================================
    // Private helpers — type resolution
    // ====================================================================

    /// Maps an AST `StorageClass` to a semantic `StorageClass`.
    fn map_storage_class(sc: &AstStorageClass) -> StorageClass {
        match sc {
            AstStorageClass::Auto => StorageClass::Auto,
            AstStorageClass::Register => StorageClass::Register,
            AstStorageClass::Static => StorageClass::Static,
            AstStorageClass::Extern => StorageClass::Extern,
            AstStorageClass::Typedef => StorageClass::Typedef,
            AstStorageClass::ThreadLocal => StorageClass::ThreadLocal,
        }
    }

    /// Resolves a list of type specifiers and qualifiers to a `CType`.
    ///
    /// Handles multi-keyword combinations like `unsigned long long int`,
    /// typedef names, struct/union/enum tags, and GCC extensions like `typeof`.
    fn resolve_type_specifiers(&mut self, specifiers: &DeclarationSpecifiers) -> CType {
        // Strategy: walk the specifiers list and accumulate into a CType.
        // C allows multi-keyword type specifiers, so we track signed/unsigned,
        // short/long counts, and the base type.
        let type_specs = &specifiers.type_specifiers;

        if type_specs.is_empty() {
            // Default to int if no type specifier (C89 implicit int rule).
            return CType::Int { signed: true };
        }

        // Single type specifier fast path
        if type_specs.len() == 1 {
            return self.resolve_single_type_spec(&type_specs[0]);
        }

        // Multi-keyword type specifier: accumulate flags.
        let mut is_signed: Option<bool> = None; // None = unspecified
        let mut long_count: u32 = 0;
        let mut is_short = false;
        let mut _has_int = false;
        let mut has_char = false;
        let mut has_double = false;
        let mut has_float = false;
        let mut is_complex = false;

        for spec in type_specs {
            match spec {
                TypeSpecifier::Signed => {
                    is_signed = Some(true);
                }
                TypeSpecifier::Unsigned => {
                    is_signed = Some(false);
                }
                TypeSpecifier::Short => {
                    is_short = true;
                }
                TypeSpecifier::Long => {
                    long_count += 1;
                }
                TypeSpecifier::Int => {
                    _has_int = true;
                }
                TypeSpecifier::Char => {
                    has_char = true;
                }
                TypeSpecifier::Double => {
                    has_double = true;
                }
                TypeSpecifier::Float => {
                    has_float = true;
                }
                TypeSpecifier::Complex => {
                    is_complex = true;
                }
                _ => {
                    // Unexpected specifier in a multi-keyword context; fall
                    // back to resolving the first specifier.
                    return self.resolve_single_type_spec(&type_specs[0]);
                }
            }
        }

        // Resolve the accumulated flags into a CType.
        let signed = is_signed.unwrap_or(true);

        if has_char {
            return CType::Char { signed };
        }

        if has_float {
            if is_complex {
                return CType::Complex(Box::new(CType::Float));
            }
            return CType::Float;
        }

        if has_double {
            if long_count >= 1 {
                if is_complex {
                    return CType::Complex(Box::new(CType::LongDouble));
                }
                return CType::LongDouble;
            }
            if is_complex {
                return CType::Complex(Box::new(CType::Double));
            }
            return CType::Double;
        }

        if is_short {
            return CType::Short { signed };
        }

        if long_count >= 2 {
            return CType::LongLong { signed };
        }

        if long_count == 1 {
            return CType::Long { signed };
        }

        // Default: int (with or without sign)
        CType::Int { signed }
    }

    /// Resolves a single type specifier into a `CType`.
    fn resolve_single_type_spec(&mut self, spec: &TypeSpecifier) -> CType {
        match spec {
            TypeSpecifier::Void => CType::Void,
            TypeSpecifier::Char => CType::Char { signed: true },
            TypeSpecifier::Short => CType::Short { signed: true },
            TypeSpecifier::Int => CType::Int { signed: true },
            TypeSpecifier::Long => CType::Long { signed: true },
            TypeSpecifier::Float => CType::Float,
            TypeSpecifier::Double => CType::Double,
            TypeSpecifier::Signed => CType::Int { signed: true },
            TypeSpecifier::Unsigned => CType::Int { signed: false },
            TypeSpecifier::Bool => CType::Bool,
            TypeSpecifier::Complex => CType::Complex(Box::new(CType::Double)),
            TypeSpecifier::Atomic(inner_type_name) => {
                let inner_ty = self.resolve_type_name(inner_type_name);
                CType::Atomic(Box::new(inner_ty))
            }
            TypeSpecifier::Struct { name, fields, .. } => self.resolve_struct_spec(*name, fields),
            TypeSpecifier::Union { name, fields, .. } => self.resolve_union_spec(*name, fields),
            TypeSpecifier::Enum {
                name, enumerators, ..
            } => self.resolve_enum_spec(*name, enumerators),
            TypeSpecifier::TypedefName { name: sym, .. } => {
                // Look up the typedef name in the current scope.
                if let Some(id) = self.scope_stack.lookup(*sym) {
                    let entry = self.symbol_table.get(id);
                    if entry.storage_class == StorageClass::Typedef {
                        entry.ty.clone()
                    } else {
                        // Name exists but is not a typedef — return as-is.
                        entry.ty.clone()
                    }
                } else {
                    self.diagnostics.error(
                        Span::DUMMY,
                        format!("unknown type name '{}'", self.interner.resolve(*sym)),
                    );
                    CType::Int { signed: true }
                }
            }
            TypeSpecifier::Typeof { operand, .. } => match operand {
                TypeofOperand::Expression(expr) => {
                    let typed = check_expression(
                        expr,
                        &self.scope_stack,
                        &self.symbol_table,
                        self.target,
                        self.diagnostics,
                        self.current_function_return_type.as_ref(),
                    );
                    typed.ty
                }
                TypeofOperand::TypeName(tn) => self.resolve_type_name(tn),
            },
            TypeSpecifier::BuiltinVaList => {
                // __builtin_va_list is an opaque pointer-sized type used for va_list.
                // Represent it as a void pointer internally.
                CType::Pointer(Box::new(CType::Void))
            }
        }
    }

    /// Resolves a struct type specifier reference or inline definition.
    fn resolve_struct_spec(
        &mut self,
        name: Option<Symbol>,
        fields: &Option<Vec<FieldDeclaration>>,
    ) -> CType {
        if let Some(field_decls) = fields {
            // Inline struct definition — resolve fields and register tag.
            let resolved_fields = self.resolve_field_declarations(field_decls);
            let tag_name = name.map(|s| self.interner.resolve(s).to_string());
            let ty = CType::Struct {
                name: tag_name,
                fields: resolved_fields,
            };
            if let Some(sym) = name {
                let tag_entry = TagEntry {
                    kind: TagKind::Struct,
                    ty: ty.clone(),
                    is_complete: true,
                    span: Span::DUMMY,
                };
                if self.scope_stack.lookup_tag_in_current(sym).is_some() {
                    self.scope_stack.update_tag(sym, tag_entry);
                } else {
                    self.scope_stack.insert_tag(sym, tag_entry);
                }
            }
            ty
        } else if let Some(sym) = name {
            // Reference to an existing struct tag.
            if let Some(tag) = self.scope_stack.lookup_tag(sym) {
                tag.ty.clone()
            } else {
                // Forward declaration — register an incomplete struct.
                let tag_name = self.interner.resolve(sym).to_string();
                let ty = CType::Struct {
                    name: Some(tag_name),
                    fields: Vec::new(),
                };
                let tag_entry = TagEntry {
                    kind: TagKind::Struct,
                    ty: ty.clone(),
                    is_complete: false,
                    span: Span::DUMMY,
                };
                self.scope_stack.insert_tag(sym, tag_entry);
                ty
            }
        } else {
            // Anonymous struct with no fields — error.
            CType::Struct {
                name: None,
                fields: Vec::new(),
            }
        }
    }

    /// Resolves a union type specifier reference or inline definition.
    fn resolve_union_spec(
        &mut self,
        name: Option<Symbol>,
        fields: &Option<Vec<FieldDeclaration>>,
    ) -> CType {
        if let Some(field_decls) = fields {
            let resolved_fields = self.resolve_field_declarations(field_decls);
            let tag_name = name.map(|s| self.interner.resolve(s).to_string());
            let ty = CType::Union {
                name: tag_name,
                fields: resolved_fields,
            };
            if let Some(sym) = name {
                let tag_entry = TagEntry {
                    kind: TagKind::Union,
                    ty: ty.clone(),
                    is_complete: true,
                    span: Span::DUMMY,
                };
                if self.scope_stack.lookup_tag_in_current(sym).is_some() {
                    self.scope_stack.update_tag(sym, tag_entry);
                } else {
                    self.scope_stack.insert_tag(sym, tag_entry);
                }
            }
            ty
        } else if let Some(sym) = name {
            if let Some(tag) = self.scope_stack.lookup_tag(sym) {
                tag.ty.clone()
            } else {
                let tag_name = self.interner.resolve(sym).to_string();
                let ty = CType::Union {
                    name: Some(tag_name),
                    fields: Vec::new(),
                };
                let tag_entry = TagEntry {
                    kind: TagKind::Union,
                    ty: ty.clone(),
                    is_complete: false,
                    span: Span::DUMMY,
                };
                self.scope_stack.insert_tag(sym, tag_entry);
                ty
            }
        } else {
            CType::Union {
                name: None,
                fields: Vec::new(),
            }
        }
    }

    /// Resolves an enum type specifier reference or inline definition.
    fn resolve_enum_spec(
        &mut self,
        name: Option<Symbol>,
        enumerators: &Option<Vec<Enumerator>>,
    ) -> CType {
        if let Some(enum_list) = enumerators {
            let tag_name = name.map(|s| self.interner.resolve(s).to_string());
            let ty = CType::Enum {
                name: tag_name,
                underlying: Box::new(CType::Int { signed: true }),
            };
            // Evaluate and register enumerator constants.
            let mut next_value: i64 = 0;
            for e in enum_list {
                let value = if let Some(val_expr) = &e.value {
                    let resolver = make_typedef_resolver(&self.scope_stack, &self.symbol_table);
                    match evaluate_constant_expression_with_resolver(val_expr, self.diagnostics, self.target, &resolver) {
                        Ok(cv) => {
                            if let Some(iv) = cv.as_integer() {
                                iv as i64
                            } else {
                                self.diagnostics.error(
                                    e.span,
                                    "enumerator value must be an integer constant".to_string(),
                                );
                                next_value
                            }
                        }
                        Err(()) => next_value,
                    }
                } else {
                    next_value
                };
                next_value = value.wrapping_add(1);

                // Insert enumerator as an integer constant in the current scope.
                let entry = SymbolEntry::new(
                    e.name,
                    CType::Int { signed: true },
                    Linkage::None,
                    StorageClass::Auto,
                    true,
                    e.span,
                );
                let id = self.symbol_table.insert(entry);
                self.scope_stack.insert(e.name, id);
            }
            if let Some(sym) = name {
                let tag_entry = TagEntry {
                    kind: TagKind::Enum,
                    ty: ty.clone(),
                    is_complete: true,
                    span: Span::DUMMY,
                };
                if self.scope_stack.lookup_tag_in_current(sym).is_some() {
                    self.scope_stack.update_tag(sym, tag_entry);
                } else {
                    self.scope_stack.insert_tag(sym, tag_entry);
                }
            }
            ty
        } else if let Some(sym) = name {
            if let Some(tag) = self.scope_stack.lookup_tag(sym) {
                tag.ty.clone()
            } else {
                // Forward reference to an unknown enum.
                let tag_name = self.interner.resolve(sym).to_string();
                let ty = CType::Enum {
                    name: Some(tag_name),
                    underlying: Box::new(CType::Int { signed: true }),
                };
                let tag_entry = TagEntry {
                    kind: TagKind::Enum,
                    ty: ty.clone(),
                    is_complete: false,
                    span: Span::DUMMY,
                };
                self.scope_stack.insert_tag(sym, tag_entry);
                ty
            }
        } else {
            CType::Enum {
                name: None,
                underlying: Box::new(CType::Int { signed: true }),
            }
        }
    }

    /// Resolves struct/union field declarations into `FieldDef` entries.
    fn resolve_field_declarations(&mut self, fields: &[FieldDeclaration]) -> Vec<FieldDef> {
        let mut result = Vec::new();
        for field_decl in fields {
            let base_ty = self.resolve_type_specifiers(&field_decl.specifiers);
            for fd in &field_decl.declarators {
                let (field_name, field_ty) = if let Some(ref decl) = fd.declarator {
                    let name = decl.name.map(|s| self.interner.resolve(s).to_string());
                    let ty = self.apply_derived_declarators(base_ty.clone(), &decl.derived);
                    (name, ty)
                } else {
                    // Anonymous field (e.g., unnamed bitfield or anonymous struct/union).
                    (None, base_ty.clone())
                };
                let bit_width = fd.bit_width.as_ref().and_then(|bw| {
                    let resolver = make_typedef_resolver(&self.scope_stack, &self.symbol_table);
                    match evaluate_constant_expression_with_resolver(bw, self.diagnostics, self.target, &resolver) {
                        Ok(cv) => cv.as_integer().map(|v| v as u32),
                        Err(()) => None,
                    }
                });
                result.push(FieldDef {
                    name: field_name,
                    ty: field_ty,
                    bit_width,
                });
            }
            // Handle the case where no declarators exist (e.g., anonymous struct/union inline).
            if field_decl.declarators.is_empty() {
                result.push(FieldDef {
                    name: None,
                    ty: base_ty,
                    bit_width: None,
                });
            }
        }
        result
    }

    /// Resolves a `TypeName` (used in casts, sizeof, etc.) into a `CType`.
    fn resolve_type_name(&mut self, tn: &TypeName) -> CType {
        let base_ty = self.resolve_specifier_qualifier_list(&tn.specifiers);
        if let Some(ref abs) = tn.declarator {
            self.apply_abstract_declarator(base_ty, abs)
        } else {
            base_ty
        }
    }

    /// Resolves a specifier-qualifier list to a CType.
    fn resolve_specifier_qualifier_list(&mut self, sql: &SpecifierQualifierList) -> CType {
        // Build a temporary DeclarationSpecifiers to reuse resolve_type_specifiers.
        let spec = DeclarationSpecifiers {
            storage_class: None,
            type_specifiers: sql.specifiers.clone(),
            type_qualifiers: sql.qualifiers,
            function_specifiers: crate::frontend::parser::ast::FunctionSpecifiers::default(),
            alignment: None,
            attrs: Vec::new(),
            has_extension: false,
            span: sql.span,
        };
        self.resolve_type_specifiers(&spec)
    }

    /// Applies an abstract declarator (pointer, array, function derivations)
    /// to a base type.
    fn apply_abstract_declarator(&mut self, base_ty: CType, abs: &AbstractDeclarator) -> CType {
        self.apply_derived_declarators(base_ty, &abs.derived)
    }

    /// Applies a chain of derived declarators to a base type, building the
    /// final type from inside out.
    ///
    /// Example: `int *a[10]` has base=int, derived=[Array(10), Pointer]
    ///   → Pointer(Array(Int, 10))
    fn apply_derived_declarators(&mut self, mut ty: CType, derived: &[DerivedDeclarator]) -> CType {
        for d in derived {
            match d {
                DerivedDeclarator::Pointer { qualifiers: _ } => {
                    ty = CType::Pointer(Box::new(ty));
                }
                DerivedDeclarator::Array {
                    size,
                    is_static: _,
                    qualifiers: _,
                } => {
                    let arr_size = size.as_ref().and_then(|s| {
                        let resolver = make_typedef_resolver(&self.scope_stack, &self.symbol_table);
                        match evaluate_constant_expression_with_resolver(s, self.diagnostics, self.target, &resolver) {
                            Ok(cv) => cv.as_integer().map(|v| v as usize),
                            Err(()) => None,
                        }
                    });
                    ty = CType::Array {
                        element: Box::new(ty),
                        size: arr_size,
                    };
                }
                DerivedDeclarator::Function { params } => {
                    let param_types = self.resolve_parameter_types(params);
                    ty = CType::Function {
                        return_type: Box::new(ty),
                        params: param_types,
                        variadic: params.variadic,
                    };
                }
            }
        }
        ty
    }

    /// Resolves parameter types from a parameter list.
    fn resolve_parameter_types(&mut self, params: &ParameterList) -> Vec<CType> {
        let mut types = Vec::with_capacity(params.params.len());
        for param in &params.params {
            let base_ty = self.resolve_type_specifiers(&param.specifiers);
            let param_ty = if let Some(ref decl) = param.declarator {
                self.apply_derived_declarators(base_ty, &decl.derived)
            } else {
                base_ty
            };
            // C11 §6.7.6.3: Array parameters decay to pointers.
            let adjusted = match param_ty {
                CType::Array { element, .. } => CType::Pointer(element),
                CType::Function { .. } => CType::Pointer(Box::new(param_ty)),
                other => other,
            };
            types.push(adjusted);
        }
        types
    }

    // ====================================================================
    // Private helpers — declaration variant handlers
    // ====================================================================

    /// Analyzes a variable declaration.
    ///
    /// Each init-declarator in the list produces a symbol table entry and
    /// contributes to the `CheckedDeclaration::Variable`.
    fn analyze_variable_decl(
        &mut self,
        specifiers: &DeclarationSpecifiers,
        declarators: &[InitDeclarator],
        attrs: &[Attribute],
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let base_ty = self.resolve_type_specifiers(specifiers);
        let sc = specifiers
            .storage_class
            .as_ref()
            .map(Self::map_storage_class)
            .unwrap_or(StorageClass::Auto);

        // Validate and collect attributes.
        let validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Variable,
            self.interner,
            self.diagnostics,
        );
        // Also validate attrs from specifiers themselves.
        let spec_attrs = validate_attributes(
            &specifiers.attrs,
            AttributeTargetKind::Variable,
            self.interner,
            self.diagnostics,
        );
        let all_attrs: Vec<ValidatedAttribute> =
            validated_attrs.into_iter().chain(spec_attrs).collect();

        // For multi-declarator declarations we return only the first.
        // In a real production compiler, we would emit multiple checked
        // declarations — here we build the first one fully and still
        // register all declarators in the symbol table.
        let mut first_result: Option<CheckedDeclaration> = None;

        for init_decl in declarators {
            let decl = &init_decl.declarator;
            let name_sym = match decl.name {
                Some(s) => s,
                None => {
                    self.diagnostics.error(
                        decl.span,
                        "variable declaration requires a name".to_string(),
                    );
                    continue;
                }
            };

            // Build the fully-derived type.
            let var_ty = self.apply_derived_declarators(base_ty.clone(), &decl.derived);

            // Apply type-modifying attributes (aligned, packed).
            let mut final_ty = var_ty.clone();
            propagate_to_type(&all_attrs, &mut final_ty);

            // Determine linkage based on storage class and scope level.
            let is_file_scope = self.scope_stack.in_file_scope();
            let prior_linkage = self
                .scope_stack
                .lookup(name_sym)
                .map(|id| self.symbol_table.get(id).linkage);
            let linkage = resolve_linkage(&sc, is_file_scope, prior_linkage);

            // Check for redeclaration.
            let has_init = init_decl.initializer.is_some();
            if let Some(existing_id) = self.scope_stack.lookup(name_sym) {
                let existing = self.symbol_table.get(existing_id);
                // Check type compatibility.
                if !types_compatible(&existing.ty, &final_ty) {
                    self.diagnostics.error(
                        span,
                        format!(
                            "conflicting types for '{}'",
                            self.interner.resolve(name_sym)
                        ),
                    );
                }
                // If both are definitions, it's a redefinition error
                // (unless tentative definitions in file scope).
                if existing.is_definition && has_init && is_file_scope {
                    self.diagnostics.error(
                        span,
                        format!("redefinition of '{}'", self.interner.resolve(name_sym)),
                    );
                }
            }

            // Analyze initializer (if present).
            let checked_init = if let Some(ref init) = init_decl.initializer {
                analyze_initializer(
                    init,
                    &final_ty,
                    self.diagnostics,
                    self.target,
                    self.interner,
                    &self.scope_stack,
                    &self.symbol_table,
                )
                .ok()
            } else {
                None
            };

            // Create and insert symbol table entry.
            let is_definition = has_init || (!is_file_scope && sc != StorageClass::Extern);
            let mut entry =
                SymbolEntry::new(name_sym, final_ty.clone(), linkage, sc, is_definition, span);
            // Apply symbol-level attributes (weak, visibility, section, etc.).
            propagate_to_symbol(&all_attrs, &mut entry);

            // If file scope with no initializer, mark as tentative.
            if is_file_scope && !has_init && sc != StorageClass::Extern {
                entry.is_tentative = true;
            }

            let sym_id = self.symbol_table.insert(entry);
            self.scope_stack.insert(name_sym, sym_id);

            if first_result.is_none() {
                first_result = Some(CheckedDeclaration::Variable {
                    symbol_id: sym_id,
                    ty: final_ty,
                    init: checked_init,
                    linkage,
                    storage_class: sc,
                    attrs: all_attrs.clone(),
                    span,
                });
            }
        }

        // If we have no declarators at all, this might be a forward struct/union/enum
        // reference (e.g., `struct _IO_FILE;`), which is valid C and simply introduces
        // or references the tag in the current scope.
        if first_result.is_none() {
            // Check if the specifiers contain a struct/union/enum tag
            let has_tag = specifiers.type_specifiers.iter().any(|ts| {
                matches!(
                    ts,
                    TypeSpecifier::Struct { .. }
                        | TypeSpecifier::Union { .. }
                        | TypeSpecifier::Enum { .. }
                )
            });
            if has_tag {
                // This is a forward declaration or standalone tag reference.
                // Resolve the type (which registers the tag) and return a
                // placeholder declaration.
                let _tag_ty = base_ty; // already resolved above
                return Ok(CheckedDeclaration::Empty { span });
            }
        }
        first_result.ok_or_else(|| {
            self.diagnostics
                .error(span, "empty variable declaration".to_string());
        })
    }

    /// Analyzes a function definition.
    fn analyze_function_def(
        &mut self,
        specifiers: &DeclarationSpecifiers,
        declarator: &Declarator,
        attrs: &[Attribute],
        body: &Statement,
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let return_ty = self.resolve_type_specifiers(specifiers);
        let sc = specifiers
            .storage_class
            .as_ref()
            .map(Self::map_storage_class)
            .unwrap_or(StorageClass::Auto);

        let name_sym = declarator.name.ok_or_else(|| {
            self.diagnostics
                .error(span, "function definition requires a name".to_string());
        })?;

        // Extract the parameter list from the function declarator.
        let (param_list, variadic) = self.extract_function_params(&declarator.derived);

        // Build the function CType.
        let param_types: Vec<CType> = param_list.iter().map(|(_, ty, _)| ty.clone()).collect();
        let func_ty = TypeBuilder::new()
            .function(return_ty.clone(), param_types.clone(), variadic)
            .build();

        // Validate attributes.
        let validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Function,
            self.interner,
            self.diagnostics,
        );
        let spec_attrs = validate_attributes(
            &specifiers.attrs,
            AttributeTargetKind::Function,
            self.interner,
            self.diagnostics,
        );
        let all_attrs: Vec<ValidatedAttribute> =
            validated_attrs.into_iter().chain(spec_attrs).collect();

        // Determine linkage.
        let is_file_scope = self.scope_stack.in_file_scope();
        let prior_linkage = self
            .scope_stack
            .lookup(name_sym)
            .map(|id| self.symbol_table.get(id).linkage);
        let linkage = resolve_linkage(&sc, is_file_scope, prior_linkage);

        // Check for redefinition.
        if let Some(existing_id) = self.scope_stack.lookup(name_sym) {
            let existing = self.symbol_table.get(existing_id);
            if existing.is_definition {
                self.diagnostics.error(
                    span,
                    format!(
                        "redefinition of function '{}'",
                        self.interner.resolve(name_sym)
                    ),
                );
            }
        }

        // Insert function symbol.
        let mut entry = SymbolEntry::new(
            name_sym, func_ty, linkage, sc,
            true, // Function definitions are always definitions.
            span,
        );
        propagate_to_symbol(&all_attrs, &mut entry);
        let sym_id = self.symbol_table.insert(entry);
        self.scope_stack.insert(name_sym, sym_id);

        // Push function scope and insert parameters.
        self.scope_stack.push(ScopeLevel::Function);

        let prev_return_type = self.current_function_return_type.take();
        self.current_function_return_type = Some(return_ty.clone());
        let prev_labels = std::mem::take(&mut self.function_labels);

        let mut checked_params = Vec::with_capacity(param_list.len());
        for (param_name, param_ty, param_span) in &param_list {
            if let Some(pname) = param_name {
                let pentry = SymbolEntry::new(
                    *pname,
                    param_ty.clone(),
                    Linkage::None,
                    StorageClass::Auto,
                    true,
                    *param_span,
                );
                let pid = self.symbol_table.insert(pentry);
                self.scope_stack.insert(*pname, pid);
                checked_params.push(CheckedParameter {
                    symbol_id: Some(pid),
                    name: Some(*pname),
                    ty: param_ty.clone(),
                    span: *param_span,
                });
            } else {
                checked_params.push(CheckedParameter {
                    symbol_id: None,
                    name: None,
                    ty: param_ty.clone(),
                    span: *param_span,
                });
            }
        }

        // Analyze the function body.
        let _ = self.analyze_statement(body);

        // Validate labels — check for referenced but undefined labels.
        for (label_sym, info) in &self.function_labels {
            if info.is_referenced && !info.is_defined {
                self.diagnostics.error(
                    info.span,
                    format!(
                        "use of undeclared label '{}'",
                        self.interner.resolve(*label_sym)
                    ),
                );
            }
        }

        // Restore previous state.
        self.function_labels = prev_labels;
        self.current_function_return_type = prev_return_type;
        self.scope_stack.pop();

        Ok(CheckedDeclaration::FunctionDef {
            symbol_id: sym_id,
            name: name_sym,
            return_ty,
            params: checked_params,
            variadic,
            body: Box::new(body.clone()),
            linkage,
            storage_class: sc,
            attrs: all_attrs,
            span,
        })
    }

    /// Analyzes a forward function declaration (prototype).
    fn analyze_function_decl(
        &mut self,
        specifiers: &DeclarationSpecifiers,
        declarator: &Declarator,
        attrs: &[Attribute],
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let return_ty = self.resolve_type_specifiers(specifiers);
        let sc = specifiers
            .storage_class
            .as_ref()
            .map(Self::map_storage_class)
            .unwrap_or(StorageClass::Auto);

        let name_sym = declarator.name.ok_or_else(|| {
            self.diagnostics
                .error(span, "function declaration requires a name".to_string());
        })?;

        let (param_list, variadic) = self.extract_function_params(&declarator.derived);

        let param_types: Vec<CType> = param_list.iter().map(|(_, ty, _)| ty.clone()).collect();
        let func_ty = TypeBuilder::new()
            .function(return_ty.clone(), param_types.clone(), variadic)
            .build();

        // Validate attributes.
        let validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Function,
            self.interner,
            self.diagnostics,
        );
        let spec_attrs = validate_attributes(
            &specifiers.attrs,
            AttributeTargetKind::Function,
            self.interner,
            self.diagnostics,
        );
        let all_attrs: Vec<ValidatedAttribute> =
            validated_attrs.into_iter().chain(spec_attrs).collect();

        // Linkage.
        let is_file_scope = self.scope_stack.in_file_scope();
        let prior_linkage = self
            .scope_stack
            .lookup(name_sym)
            .map(|id| self.symbol_table.get(id).linkage);
        let linkage = resolve_linkage(&sc, is_file_scope, prior_linkage);

        // Handle redeclaration merging.
        if let Some(existing_id) = self.scope_stack.lookup(name_sym) {
            let existing = self.symbol_table.get(existing_id);
            if !types_compatible(&existing.ty, &func_ty) {
                self.diagnostics.error(
                    span,
                    format!(
                        "conflicting types for '{}'",
                        self.interner.resolve(name_sym)
                    ),
                );
            }
        }

        // Insert into symbol table.
        let mut entry = SymbolEntry::new(
            name_sym, func_ty, linkage, sc, false, // Declarations are not definitions.
            span,
        );
        propagate_to_symbol(&all_attrs, &mut entry);
        let sym_id = self.symbol_table.insert(entry);
        self.scope_stack.insert(name_sym, sym_id);

        Ok(CheckedDeclaration::FunctionDecl {
            symbol_id: sym_id,
            name: name_sym,
            return_ty,
            param_types,
            variadic,
            linkage,
            storage_class: sc,
            attrs: all_attrs,
            span,
        })
    }

    /// Analyzes a typedef declaration.
    fn analyze_typedef(
        &mut self,
        specifiers: &DeclarationSpecifiers,
        declarators: &[Declarator],
        attrs: &[Attribute],
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let base_ty = self.resolve_type_specifiers(specifiers);
        let _validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Type,
            self.interner,
            self.diagnostics,
        );

        // Typedef can define multiple names.
        let mut first_result: Option<CheckedDeclaration> = None;

        for decl in declarators {
            let name_sym = match decl.name {
                Some(s) => s,
                None => {
                    self.diagnostics
                        .error(decl.span, "typedef requires a name".to_string());
                    continue;
                }
            };

            let ty = self.apply_derived_declarators(base_ty.clone(), &decl.derived);

            // Insert into symbol table as a typedef.
            let entry = SymbolEntry::new(
                name_sym,
                ty.clone(),
                Linkage::None,
                StorageClass::Typedef,
                true,
                span,
            );
            let sym_id = self.symbol_table.insert(entry);
            self.scope_stack.insert(name_sym, sym_id);

            if first_result.is_none() {
                first_result = Some(CheckedDeclaration::Typedef {
                    symbol_id: sym_id,
                    name: name_sym,
                    ty,
                    span,
                });
            }
        }

        first_result.ok_or_else(|| {
            self.diagnostics
                .error(span, "empty typedef declaration".to_string());
        })
    }

    /// Analyzes a standalone struct definition.
    fn analyze_struct_def(
        &mut self,
        name: Option<Symbol>,
        fields: &[FieldDeclaration],
        attrs: &[Attribute],
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let resolved_fields = self.resolve_field_declarations(fields);

        // Validate attributes.
        let validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Type,
            self.interner,
            self.diagnostics,
        );

        // Check for packed/aligned attributes for layout computation.
        let is_packed = validated_attrs
            .iter()
            .any(|a| matches!(a, ValidatedAttribute::Packed));
        let min_align = validated_attrs.iter().find_map(|a| {
            if let ValidatedAttribute::Aligned(Some(n)) = a {
                Some(*n as usize)
            } else {
                None
            }
        });

        // Convert to FieldDef for layout computation.
        let layout = if is_packed || min_align.is_some() {
            crate::common::type_builder::compute_struct_layout_with_attrs(
                &resolved_fields,
                self.target,
                is_packed,
                min_align,
            )
        } else {
            compute_struct_layout(&resolved_fields, self.target)
        };

        let tag_name = name.map(|s| self.interner.resolve(s).to_string());
        let mut ty = CType::Struct {
            name: tag_name,
            fields: resolved_fields,
        };
        propagate_to_type(&validated_attrs, &mut ty);

        // Register the tag.
        if let Some(sym) = name {
            let tag_entry = TagEntry {
                kind: TagKind::Struct,
                ty: ty.clone(),
                is_complete: true,
                span,
            };
            if self.scope_stack.lookup_tag_in_current(sym).is_some() {
                self.scope_stack.update_tag(sym, tag_entry);
            } else {
                self.scope_stack.insert_tag(sym, tag_entry);
            }
        }

        Ok(CheckedDeclaration::StructDef {
            name,
            ty,
            layout,
            attrs: validated_attrs,
            span,
        })
    }

    /// Analyzes a standalone union definition.
    fn analyze_union_def(
        &mut self,
        name: Option<Symbol>,
        fields: &[FieldDeclaration],
        attrs: &[Attribute],
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let resolved_fields = self.resolve_field_declarations(fields);

        let validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Type,
            self.interner,
            self.diagnostics,
        );

        let tag_name = name.map(|s| self.interner.resolve(s).to_string());
        let mut ty = CType::Union {
            name: tag_name,
            fields: resolved_fields,
        };
        propagate_to_type(&validated_attrs, &mut ty);

        if let Some(sym) = name {
            let tag_entry = TagEntry {
                kind: TagKind::Union,
                ty: ty.clone(),
                is_complete: true,
                span,
            };
            if self.scope_stack.lookup_tag_in_current(sym).is_some() {
                self.scope_stack.update_tag(sym, tag_entry);
            } else {
                self.scope_stack.insert_tag(sym, tag_entry);
            }
        }

        Ok(CheckedDeclaration::UnionDef {
            name,
            ty,
            attrs: validated_attrs,
            span,
        })
    }

    /// Analyzes a standalone enum definition.
    fn analyze_enum_def(
        &mut self,
        name: Option<Symbol>,
        enumerators: &[Enumerator],
        attrs: &[Attribute],
        span: Span,
    ) -> Result<CheckedDeclaration, ()> {
        let validated_attrs = validate_attributes(
            attrs,
            AttributeTargetKind::Type,
            self.interner,
            self.diagnostics,
        );

        let tag_name = name.map(|s| self.interner.resolve(s).to_string());
        let ty = CType::Enum {
            name: tag_name,
            underlying: Box::new(CType::Int { signed: true }),
        };

        // Evaluate enumerator values.
        let mut evaluated = Vec::with_capacity(enumerators.len());
        let mut next_value: i64 = 0;

        for e in enumerators {
            let value = if let Some(ref val_expr) = e.value {
                let resolver = make_typedef_resolver(&self.scope_stack, &self.symbol_table);
                match evaluate_constant_expression_with_resolver(val_expr, self.diagnostics, self.target, &resolver) {
                    Ok(cv) => {
                        if let Some(iv) = cv.as_integer() {
                            iv as i64
                        } else {
                            self.diagnostics.error(
                                e.span,
                                "enumerator value must be an integer constant".to_string(),
                            );
                            next_value
                        }
                    }
                    Err(()) => next_value,
                }
            } else {
                next_value
            };
            evaluated.push((e.name, value));
            next_value = value.wrapping_add(1);

            // Insert each enumerator as an integer constant.
            let entry = SymbolEntry::new(
                e.name,
                CType::Int { signed: true },
                Linkage::None,
                StorageClass::Auto,
                true,
                e.span,
            );
            let id = self.symbol_table.insert(entry);
            self.scope_stack.insert(e.name, id);
        }

        // Register enum tag.
        if let Some(sym) = name {
            let tag_entry = TagEntry {
                kind: TagKind::Enum,
                ty: ty.clone(),
                is_complete: true,
                span,
            };
            if self.scope_stack.lookup_tag_in_current(sym).is_some() {
                self.scope_stack.update_tag(sym, tag_entry);
            } else {
                self.scope_stack.insert_tag(sym, tag_entry);
            }
        }

        Ok(CheckedDeclaration::EnumDef {
            name,
            ty,
            enumerators: evaluated,
            attrs: validated_attrs,
            span,
        })
    }

    /// Extracts the parameter list from a chain of derived declarators.
    ///
    /// Scans the derived list for a `Function` variant and resolves each
    /// parameter's type, returning `(Option<name>, CType, Span)` triples
    /// and the variadic flag.
    fn extract_function_params(
        &mut self,
        derived: &[DerivedDeclarator],
    ) -> (Vec<(Option<Symbol>, CType, Span)>, bool) {
        for d in derived {
            if let DerivedDeclarator::Function { params } = d {
                let mut result = Vec::with_capacity(params.params.len());
                for param in &params.params {
                    let base_ty = self.resolve_type_specifiers(&param.specifiers);
                    let (pname, param_ty) = if let Some(ref decl) = param.declarator {
                        let ty = self.apply_derived_declarators(base_ty, &decl.derived);
                        (decl.name, ty)
                    } else {
                        (None, base_ty)
                    };
                    // Adjust parameter types per C11 §6.7.6.3.
                    let adjusted = match param_ty {
                        CType::Array { element, .. } => CType::Pointer(element),
                        CType::Function { .. } => CType::Pointer(Box::new(param_ty)),
                        other => other,
                    };
                    result.push((pname, adjusted, param.span));
                }
                return (result, params.variadic);
            }
        }
        // No function derivation found — empty parameter list.
        (Vec::new(), false)
    }
} // end impl SemanticAnalyzer

// ========================================================================
// Typedef-resolution helper for constant evaluation
// ========================================================================

/// Creates a typedef-resolving closure for use with
/// [`evaluate_constant_expression_with_resolver`].
///
/// This is a free function (rather than a method on `SemanticAnalyzer`) so
/// that the borrow-checker can see the disjoint borrows: the closure
/// captures `scope_stack` and `symbol_table` immutably while the caller
/// retains mutable access to the `diagnostics` field.
fn make_typedef_resolver<'a>(
    scope_stack: &'a ScopeStack,
    symbol_table: &'a SymbolTable,
) -> impl Fn(Symbol) -> Option<CType> + 'a {
    move |sym: Symbol| {
        let id = scope_stack.lookup(sym)?;
        let entry = symbol_table.get(id);
        if entry.is_typedef() {
            Some(entry.ty.clone())
        } else {
            None
        }
    }
}
