//! # Statement Lowering
//!
//! Converts AST statements into IR control flow graph (CFG) structures.
//! This is the primary module responsible for creating the basic block
//! topology of IR functions.
//!
//! ## Control Flow Patterns
//!
//! - **If/Else**: Creates `then_block`, `else_block`, `merge_block`
//! - **While loop**: Creates `header_block` (condition), `body_block`, `exit_block`
//! - **Do-While loop**: Creates `body_block`, `cond_block`, `exit_block`
//! - **For loop**: Creates `cond_block`, `body_block`, `incr_block`, `exit_block`
//! - **Switch**: Creates a `dispatch_block` + one block per case + `default_block` + `exit_block`
//! - **Computed goto**: Creates an indirect branch instruction with possible targets
//!
//! ## Recursion Limit
//!
//! A hard 512-depth recursion limit is enforced for deeply nested statements
//! (kernel macros can produce deeply nested compound statements). When the
//! limit is exceeded a diagnostic is emitted and lowering returns early.
//!
//! ## Dependencies
//! - `crate::ir::*` — Core IR types (instructions, basic_block, function, builder)
//! - `crate::frontend::parser::ast` — AST statement node types
//! - `crate::common::*` — Types, diagnostics, target
//! - Calls into `expr_lowering` for expression evaluation within statements
//! - Calls into `asm_lowering` for inline assembly statements

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::CType;
use crate::frontend::parser::ast;
use crate::ir::basic_block::BasicBlock;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::lowering::expr_lowering::{
    insert_implicit_conversion, lower_expression, lower_expression_typed, lower_lvalue,
    ExprLoweringContext, TypedValue,
};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

use super::asm_lowering;
use super::decl_lowering;
#[allow(unused_imports)]
use super::expr_lowering;

// ===========================================================================
// Constants
// ===========================================================================

/// Maximum recursion depth for nested statement lowering.
/// This limit protects against deeply nested macro expansions
/// encountered in the Linux kernel source tree.
const MAX_RECURSION_DEPTH: u32 = 512;

// ===========================================================================
// LoopContext — break/continue target tracking
// ===========================================================================

/// Tracks the current control flow environment for break/continue targets.
///
/// Each time a loop construct is entered (`while`, `do-while`, `for`), a new
/// `LoopContext` is pushed onto `StmtLoweringContext::loop_stack`. When the
/// loop body ends the context is popped. `break` and `continue` statements
/// use the topmost context to determine their branch target.
#[derive(Clone)]
pub struct LoopContext {
    /// Block to branch to on `break`.
    pub break_target: BlockId,
    /// Block to branch to on `continue`.
    pub continue_target: BlockId,
}

// ===========================================================================
// SwitchContext — case/default collection during switch lowering
// ===========================================================================

/// Tracks switch statement context for case/default lowering.
///
/// Created when a `switch` statement is entered and consumed when the switch
/// body has been fully lowered. Case and default labels register their blocks
/// here so the switch dispatch instruction can be built after the body.
pub struct SwitchContext {
    /// Block to branch to after the switch (break target).
    pub break_target: BlockId,
    /// Map from case values to their blocks.
    pub case_blocks: Vec<(i64, BlockId)>,
    /// Case ranges (low, high, target) for large `case low ... high:` ranges
    /// that are too large to enumerate individually.
    pub case_ranges: Vec<(i64, i64, BlockId)>,
    /// Default case block (if present).
    pub default_block: Option<BlockId>,
}

// ===========================================================================
// StmtLoweringContext — per-function lowering state
// ===========================================================================

/// Context for statement lowering within a function.
///
/// Holds mutable references to the IR construction machinery and tracks the
/// control flow state required by break/continue/case/goto.
///
/// # Alloca-First Pattern
///
/// All local variables are stored as allocas in the entry block. Statement
/// lowering reads and writes variables through loads and stores to those
/// allocas. The subsequent mem2reg pass (Phase 7) promotes eligible allocas
/// to SSA virtual registers.
pub struct StmtLoweringContext<'a> {
    /// IR builder for instruction creation.
    pub builder: &'a mut IrBuilder,
    /// Current function being lowered.
    pub function: &'a mut IrFunction,
    /// IR module for global references.
    pub module: &'a mut IrModule,
    /// Target architecture.
    pub target: &'a Target,
    /// Diagnostic engine.
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Local variable name → alloca Value mapping.
    pub local_vars: &'a mut FxHashMap<String, Value>,
    /// Label name → block ID mapping (for goto/label).
    pub label_blocks: &'a mut FxHashMap<String, BlockId>,
    /// Stack of active loop contexts (for break/continue).
    pub loop_stack: Vec<LoopContext>,
    /// Current switch context (for case/default).
    pub switch_ctx: Option<SwitchContext>,
    /// Current recursion depth (enforced to ≤512).
    pub recursion_depth: u32,

    // --- Fields required for expression lowering integration ---
    /// Type builder for sizeof/alignof/struct layout queries.
    pub type_builder: &'a TypeBuilder,
    /// Function parameter name → Value mapping.
    pub param_values: &'a FxHashMap<String, Value>,
    /// Symbol index → interned name table.
    pub name_table: &'a [String],
    /// Local variable name → declared C type.
    pub local_types: &'a FxHashMap<String, CType>,
    /// Enum constant name → integer value mapping.
    pub enum_constants: &'a FxHashMap<String, i128>,
    /// Static local variable name → mangled global name mapping.
    pub static_locals: &'a mut FxHashMap<String, String>,
    /// Struct/union tag → full CType definition registry.
    pub struct_defs: &'a FxHashMap<String, CType>,
    /// Name of the function currently being lowered (for `__func__`).
    pub current_function_name: Option<&'a str>,
    /// Scope-local type overrides for block-scoped variable shadowing.
    /// When a compound statement declares a variable with the same name as
    /// one in an outer scope but with a different type, this map holds the
    /// CURRENT scope's type, overriding `local_types`.
    pub scope_type_overrides: FxHashMap<String, CType>,
    /// C-level return type of the enclosing function.  Used by `lower_return`
    /// to insert an implicit conversion (SExt/ZExt/Trunc) when the return
    /// expression's C type differs from the function's declared return type.
    /// Without this, returning a `signed char` from an `int`-returning
    /// function would omit the sign-extension, producing incorrect values
    /// for negative numbers.
    pub return_ctype: Option<CType>,
    /// Cache for computed struct/union layouts, keyed by struct tag name.
    /// Shared with `ExprLoweringContext` to avoid recomputing full layout
    /// for every member access on large structs.
    pub layout_cache: FxHashMap<String, crate::common::type_builder::StructLayout>,
    /// VLA variable name → IR Value holding the total byte size (runtime).
    /// When sizeof is applied to a VLA variable, this value is emitted
    /// instead of the compile-time sizeof (which would be 8 = pointer size).
    pub vla_sizes: FxHashMap<String, Value>,
    /// Stack pointer save-point for VLA deallocation.  When a VLA is first
    /// encountered, the stack pointer is saved here.  Before each subsequent
    /// VLA allocation in the same scope, the stack is restored to this point
    /// so that VLAs in loops don't leak stack space.
    pub vla_stack_save: Option<Value>,
    /// Set of variable names whose first actual declaration has been
    /// processed in the current or an enclosing scope.  Used by
    /// `ensure_allocas_for_declaration` to detect variable shadowing:
    /// if a name is already "claimed" when a new declaration for it is
    /// encountered, the new declaration shadows an outer variable and
    /// needs its own alloca.  Saved/restored with compound-statement
    /// scope boundaries so that sequential (non-overlapping) scopes
    /// can safely share the pre-scan alloca.
    pub claimed_vars: FxHashSet<String>,
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Lower a single AST statement to IR.
///
/// Returns `Some(block_id)` indicating the block where execution continues
/// after this statement, or `None` if the statement terminates the current
/// block (return, break, continue, goto — no fall-through).
///
/// # Recursion Limit
///
/// Increments `ctx.recursion_depth` on entry and decrements on exit.
/// If the depth exceeds [`MAX_RECURSION_DEPTH`] a diagnostic error is emitted
/// and `None` is returned immediately.
pub fn lower_statement(
    ctx: &mut StmtLoweringContext<'_>,
    stmt: &ast::Statement,
) -> Option<BlockId> {
    // Enforce 512-depth recursion limit.
    ctx.recursion_depth += 1;
    if ctx.recursion_depth > MAX_RECURSION_DEPTH {
        let span = statement_span(stmt);
        ctx.diagnostics.emit(Diagnostic::error(
            span,
            format!(
                "statement nesting depth exceeds maximum of {} (deeply nested macros?)",
                MAX_RECURSION_DEPTH,
            ),
        ));
        ctx.recursion_depth -= 1;
        return None;
    }

    let result = match stmt {
        ast::Statement::Compound(compound) => lower_compound_statement(ctx, compound),

        ast::Statement::Expression(opt_expr) => {
            if let Some(expr) = opt_expr {
                lower_expression_statement(ctx, expr);
            }
            // Expression statements (including empty `;`) fall through.
            ctx.builder.get_insert_block()
        }

        ast::Statement::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => lower_if_statement(ctx, condition, then_branch, else_branch.as_deref(), *span),

        ast::Statement::While {
            condition,
            body,
            span,
        } => lower_while_loop(ctx, condition, body, *span),

        ast::Statement::DoWhile {
            body,
            condition,
            span,
        } => lower_do_while_loop(ctx, body, condition, *span),

        ast::Statement::For {
            init,
            condition,
            increment,
            body,
            span,
        } => lower_for_loop(
            ctx,
            init.as_ref(),
            condition.as_deref(),
            increment.as_deref(),
            body,
            *span,
        ),

        ast::Statement::Switch {
            condition,
            body,
            span,
        } => lower_switch(ctx, condition, body, *span),

        ast::Statement::Case {
            value,
            statement,
            span,
        } => lower_case(ctx, value, statement, *span),

        ast::Statement::CaseRange {
            low,
            high,
            statement,
            span,
        } => lower_case_range(ctx, low, high, statement, *span),

        ast::Statement::Default { statement, span } => lower_default(ctx, statement, *span),

        ast::Statement::Break { span } => lower_break(ctx, *span),

        ast::Statement::Continue { span } => lower_continue(ctx, *span),

        ast::Statement::Return { value, span } => lower_return(ctx, value.as_deref(), *span),

        ast::Statement::Goto { label, span } => lower_goto(ctx, label, *span),

        ast::Statement::ComputedGoto { target, span } => lower_computed_goto(ctx, target, *span),

        ast::Statement::Labeled {
            label,
            statement,
            span,
            ..
        } => lower_label(ctx, label, statement, *span),

        ast::Statement::Asm(asm_stmt) => lower_asm_dispatch(ctx, asm_stmt),

        ast::Statement::LocalLabel(labels, span) => {
            for sym in labels {
                register_local_label(ctx, sym, *span);
            }
            ctx.builder.get_insert_block()
        }

        ast::Statement::Declaration(decl) => {
            // Process declaration initializers: emit Store instructions
            // for each declarator with an initializer.
            lower_declaration_initializers(ctx, decl);
            ctx.builder.get_insert_block()
        }
    };

    ctx.recursion_depth -= 1;
    result
}

// ===========================================================================
// Compound Statement
// ===========================================================================

/// Lowers a compound statement (block) `{ ... }`.
///
/// Processes each block item sequentially. After a terminator statement
/// (return, goto, break, continue) remaining items are unreachable and
/// are silently skipped.
fn lower_compound_statement(
    ctx: &mut StmtLoweringContext<'_>,
    compound: &ast::CompoundStatement,
) -> Option<BlockId> {
    // ── Block Scope: save local variable bindings ────────────────────────
    //
    // C11 §6.2.1: Each compound statement (block) introduces a new scope.
    // Variables declared inside the block shadow identically-named variables
    // from outer scopes (including function parameters).  When the block
    // ends, the outer bindings must be restored.
    //
    // We snapshot the current `local_vars` and `scope_type_overrides` maps
    // before processing the block's items.  After lowering, we restore both
    // maps so that names declared inside the block don't leak into the
    // enclosing scope and any shadowed outer names are recovered.
    let saved_local_vars = ctx.local_vars.clone();
    let saved_scope_type_overrides = ctx.scope_type_overrides.clone();
    let saved_claimed_vars = ctx.claimed_vars.clone();

    let mut current_block = ctx.builder.get_insert_block();

    for item in &compound.items {
        // If the current block is terminated (e.g. by a goto, return, break),
        // non-label items are unreachable.  BUT labeled statements are always
        // reachable (via goto / computed goto) — they create new blocks.
        // So we must NOT break; instead skip only non-label items.
        let terminated = current_block.is_none() || is_block_terminated(ctx);

        match item {
            ast::BlockItem::Statement(stmt) => {
                // Labeled statements (including case/default) start new
                // blocks and must always be lowered, even after a
                // terminator.  All other statements are unreachable.
                if terminated && !is_label_like(stmt) {
                    continue;
                }
                current_block = lower_statement(ctx, stmt);
            }
            ast::BlockItem::Declaration(decl) => {
                if terminated {
                    continue;
                }
                // Ensure allocas exist for ALL declarators in this
                // declaration.  The top-level function lowering driver
                // pre-scans the AST for locals, but cannot see into
                // GCC statement expressions, so variables declared
                // inside nested compound statements (e.g. inside
                // `do { } while(0)` in percpu macros) may be missing.
                // We create allocas on demand here before processing
                // initializers so that every variable has a slot.
                ensure_allocas_for_declaration(ctx, decl);
                lower_declaration_initializers(ctx, decl);
            }
        }
    }

    // ── Block Scope: restore outer variable bindings ─────────────────────
    //
    // Restore the local_vars and scope_type_overrides maps to their state
    // before this compound statement.  This ensures that any shadowing
    // declarations inside the block do not persist into the enclosing scope.
    *ctx.local_vars = saved_local_vars;
    ctx.scope_type_overrides = saved_scope_type_overrides;
    ctx.claimed_vars = saved_claimed_vars;

    current_block
}

/// Returns `true` if the statement introduces or **contains** a new block
/// (label-like). These must be lowered even when the current block is
/// terminated because they are reachable through jumps (goto, computed goto,
/// switch dispatch).
///
/// This check is recursive: `if (0) { L: x = 0; }` is NOT itself a Labeled
/// statement, but it contains one, so it must still be lowered (the label is
/// a goto target that would otherwise be lost).
fn is_label_like(stmt: &ast::Statement) -> bool {
    match stmt {
        ast::Statement::Labeled { .. }
        | ast::Statement::Case { .. }
        | ast::Statement::Default { .. }
        | ast::Statement::CaseRange { .. } => true,
        // Recurse into control-flow structures that may contain labels.
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => is_label_like(then_branch) || else_branch.as_ref().map_or(false, |e| is_label_like(e)),
        ast::Statement::While { body, .. }
        | ast::Statement::DoWhile { body, .. }
        | ast::Statement::For { body, .. } => is_label_like(body),
        ast::Statement::Switch { body, .. } => is_label_like(body),
        ast::Statement::Compound(compound) => compound.items.iter().any(|item| match item {
            ast::BlockItem::Statement(s) => is_label_like(s),
            _ => false,
        }),
        _ => false,
    }
}

// ===========================================================================
// Declaration Initializer Lowering
// ===========================================================================

/// Process a declaration's initializers, emitting Store instructions for each
/// declarator that has an initializer.
///
/// This bridges the gap between the alloca-first pattern (where all allocas
/// are pre-created in the entry block by `decl_lowering::allocate_local_variables`)
/// and the actual initialization that must happen at the point of declaration.
///
/// For each declarator in the declaration:
/// 1. Extract the variable name from the declarator
/// 2. Look up its alloca in `ctx.local_vars`
/// 3. If an initializer is present, call `decl_lowering::lower_local_initializer`
///    to emit the Store instruction(s)
///
/// Declarations without initializers (e.g., `int x;`) are no-ops — the alloca
/// already exists and the value is undefined until a later assignment.
/// Ensure every declarator in `decl` has an alloca in `ctx.local_vars`.
///
/// The top-level function lowering pre-scans the AST body to collect all
/// local variables, but that scan cannot see through expression boundaries
/// into GCC statement expressions.  Variables declared in nested compound
/// statements (e.g. `do { const void *__vpp_verify = …; } while(0)` inside
/// a `VERIFY_PERCPU_PTR` macro expansion) are therefore missing.
///
/// Infer the size of an unsized array declaration from its initializer.
///
/// For declarations like `char data[] = "hello"` or `int arr[] = {1,2,3}`,
/// the C type has `Array(elem, None)`. This function counts elements in
/// the initializer and returns a sized `Array(elem, Some(n))`.
///
/// If the type is not an unsized array, or there's no initializer, the
/// original type is returned unchanged.
fn infer_array_size_from_init(
    c_type: &CType,
    initializer: Option<&ast::Initializer>,
    name_table: &[String],
) -> CType {
    let _ = name_table;
    match c_type {
        CType::Array(elem, None) => {
            if let Some(init) = initializer {
                match init {
                    ast::Initializer::Expression(expr) => {
                        // String initializer: `char d[] = "hello"`
                        // Count the string length (including null terminator).
                        if let Some(len) = string_literal_length(expr) {
                            CType::Array(elem.clone(), Some(len))
                        } else {
                            c_type.clone()
                        }
                    }
                    ast::Initializer::List {
                        designators_and_initializers,
                        ..
                    } => {
                        // Brace-init list: `int a[] = {1,2,3}` → size=3
                        CType::Array(elem.clone(), Some(designators_and_initializers.len()))
                    }
                }
            } else {
                c_type.clone()
            }
        }
        _ => c_type.clone(),
    }
}

/// Count characters in a string literal expression (including null terminator).
fn string_literal_length(expr: &ast::Expression) -> Option<usize> {
    match expr {
        ast::Expression::StringLiteral {
            segments, prefix, ..
        } => {
            let char_width: usize = match prefix {
                crate::frontend::parser::ast::StringPrefix::None
                | crate::frontend::parser::ast::StringPrefix::U8 => 1,
                crate::frontend::parser::ast::StringPrefix::L
                | crate::frontend::parser::ast::StringPrefix::U32 => 4,
                crate::frontend::parser::ast::StringPrefix::U16 => 2,
            };
            if char_width == 1 {
                let mut total: usize = 0;
                for seg in segments {
                    total += seg.value.len();
                }
                Some(total + 1)
            } else {
                // Wide string: decode UTF-8 to count code points.
                let mut raw_bytes = Vec::new();
                for seg in segments {
                    raw_bytes.extend_from_slice(&seg.value);
                }
                let codepoints = super::decl_lowering::decode_bytes_to_codepoints(&raw_bytes);
                Some(codepoints.len() + 1) // +1 for null terminator
            }
        }
        ast::Expression::Parenthesized { inner, .. } => string_literal_length(inner),
        _ => None,
    }
}

/// This function creates allocas on demand for any declarator that is not
/// already present in `ctx.local_vars`, so that subsequent initializer
/// lowering and identifier lookups can find every variable.
/// Compare two CTypes for the purpose of scope-shadowing detection.
///
/// This handles the common case where `resolve_declaration_type` produces
/// a named struct/union with an empty field list (forward-reference style)
/// while the pre-scanned `local_types` entry has the full definition with
/// fields populated.  Named structs/unions with the same tag name are
/// treated as equivalent regardless of whether their field lists differ.
/// For compound types (arrays, pointers), the comparison recurses into
/// element/pointee types.  Scalar and other types fall through to an
/// IR-type comparison that ignores minor CType representation differences
/// (e.g. `int` vs `typedef int my_int`).
fn ctypes_compatible_for_scope_shadowing(a: &CType, b: &CType, target: &Target) -> bool {
    // Unwrap typedefs before comparison so that a resolved typedef
    // (with full struct fields) matches an unresolved one (empty fields)
    // when they share the same typedef name or underlying struct tag.
    let a = unwrap_typedef_for_compat(a);
    let b = unwrap_typedef_for_compat(b);
    match (a, b) {
        // Named struct — same tag means same type regardless of field resolution.
        (CType::Struct { name: Some(na), .. }, CType::Struct { name: Some(nb), .. }) => na == nb,
        // Named union — same tag means same type.
        (CType::Union { name: Some(na), .. }, CType::Union { name: Some(nb), .. }) => na == nb,
        // Array — recurse into element type and compare sizes.
        // When one side has None (unsized array, e.g. `T arr[]`), treat
        // it as compatible with any sized variant — the pre-scan may have
        // already inferred the size from the initializer while
        // resolve_declaration_type preserves the raw unsized form.
        (CType::Array(ea, sa), CType::Array(eb, sb)) => {
            let sizes_compatible = match (sa, sb) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            };
            sizes_compatible && ctypes_compatible_for_scope_shadowing(ea, eb, target)
        }
        // Pointer — recurse into pointee type.
        (CType::Pointer(pa, _), CType::Pointer(pb, _)) => {
            ctypes_compatible_for_scope_shadowing(pa, pb, target)
        }
        // For scalar integer types that map to the same IR width (e.g.
        // `unsigned char` vs `signed char` → both I8, `unsigned int` vs
        // `signed int` → both I32), signedness MUST be distinguished.
        // The code generator uses different widening operations (ZExt vs
        // SExt) depending on the C-level signedness, so reusing the same
        // alloca for variables of different signedness produces incorrect
        // results (e.g. `unsigned char 0xFF` → 255 vs `signed char 0xFF` → -1).
        _ if crate::common::types::is_integer(a)
            && crate::common::types::is_integer(b)
            && crate::common::types::is_unsigned(a) != crate::common::types::is_unsigned(b) =>
        {
            false
        }
        // For all other types, compare via IR types which normalises
        // typedefs and minor representation differences.
        _ => IrType::from_ctype(a, target) == IrType::from_ctype(b, target),
    }
}

/// Unwrap typedef layers to reach the underlying concrete type.
/// This ensures that `typedef struct S S;` comparisons look through
/// the typedef wrapper and compare the underlying struct tags,
/// preventing false "scope shadowing" when the pre-scan resolved
/// the struct fields but the AST-level type still has empty fields.
fn unwrap_typedef_for_compat(ty: &CType) -> &CType {
    match ty {
        CType::Typedef { underlying, .. } => unwrap_typedef_for_compat(underlying),
        other => other,
    }
}

pub fn ensure_allocas_for_declaration(ctx: &mut StmtLoweringContext<'_>, decl: &ast::Declaration) {
    // Skip typedef declarations — they don't produce any runtime code.
    if matches!(
        decl.specifiers.storage_class,
        Some(ast::StorageClass::Typedef)
    ) {
        return;
    }

    // Handle extern declarations: register as external globals so they can
    // be referenced by name during identifier lowering.
    if matches!(
        decl.specifiers.storage_class,
        Some(ast::StorageClass::Extern)
    ) {
        for init_decl in &decl.declarators {
            let var_name = match extract_decl_name(&init_decl.declarator, ctx.name_table) {
                Some(name) => name,
                None => continue,
            };
            // C11 §6.2.2p4: a block-scope `extern` declaration refers to
            // the external definition, NOT to any local variable of the
            // same name in an enclosing scope.  Remove the local mapping
            // so that `lower_identifier` falls through to the global lookup.
            ctx.local_vars.remove(&var_name);

            // If already known as a global, skip adding again.
            if ctx.module.get_global(&var_name).is_some() {
                continue;
            }
            let c_type = decl_lowering::resolve_declaration_type(
                &decl.specifiers,
                &init_decl.declarator,
                ctx.target,
                ctx.name_table,
            );
            let ir_type = IrType::from_ctype(&c_type, ctx.target);
            let mut global =
                crate::ir::module::GlobalVariable::new(var_name.clone(), ir_type, None);
            global.linkage = crate::ir::module::Linkage::External;
            global.is_definition = false;
            ctx.module.add_global(global);
        }
        return;
    }

    // Handle static/thread-local declarations: create mangled globals (like
    // lower_static_local in decl_lowering). This handles static variables
    // declared inside statement expressions that aren't pre-scanned.
    if matches!(
        decl.specifiers.storage_class,
        Some(ast::StorageClass::Static) | Some(ast::StorageClass::ThreadLocal)
    ) {
        for init_decl in &decl.declarators {
            let var_name = match extract_decl_name(&init_decl.declarator, ctx.name_table) {
                Some(name) => name,
                None => continue,
            };
            // If already registered as a static local, skip.
            if ctx.static_locals.contains_key(&var_name) {
                continue;
            }
            let c_type = decl_lowering::resolve_declaration_type(
                &decl.specifiers,
                &init_decl.declarator,
                ctx.target,
                ctx.name_table,
            );
            let ir_type = IrType::from_ctype(&c_type, ctx.target);
            let func_name = &ctx.function.name;
            let mangled_name = format!("{}.{}", func_name, var_name);

            // Evaluate constant initializer if present.  Delegate to the
            // full evaluate_initializer_constant from decl_lowering so that
            // function pointers, address-of expressions, designated initializers,
            // and other non-trivial constant expressions are handled correctly
            // — matching how global variable initializers are processed.
            let constant = if let Some(ref init) = init_decl.initializer {
                super::decl_lowering::evaluate_initializer_constant(
                    init,
                    &c_type,
                    ctx.target,
                    ctx.type_builder,
                    ctx.diagnostics,
                    ctx.name_table,
                    ctx.enum_constants,
                )
            } else {
                Some(crate::ir::module::Constant::ZeroInit)
            };

            // Infer array size from initializer when declaration uses [].
            let ir_type = match (&c_type, &constant) {
                (CType::Array(elem, None), Some(crate::ir::module::Constant::String(bytes))) => {
                    let elem_byte_size = super::decl_lowering::wide_char_elem_size(elem);
                    let fixed = CType::Array(elem.clone(), Some(bytes.len() / elem_byte_size));
                    IrType::from_ctype(&fixed, ctx.target)
                }
                (CType::Array(elem, None), Some(crate::ir::module::Constant::Array(elems))) => {
                    let fixed = CType::Array(elem.clone(), Some(elems.len()));
                    IrType::from_ctype(&fixed, ctx.target)
                }
                _ => ir_type,
            };

            let mut global =
                crate::ir::module::GlobalVariable::new(mangled_name.clone(), ir_type, constant);
            global.linkage = crate::ir::module::Linkage::Internal;
            global.is_definition = true;
            ctx.module.add_global(global);
            ctx.static_locals
                .insert(var_name.clone(), mangled_name.clone());
        }
        return;
    }

    // Regular (auto/register) declarations: create allocas on-demand.
    for init_decl in &decl.declarators {
        let var_name = match extract_decl_name(&init_decl.declarator, ctx.name_table) {
            Some(name) => name,
            None => continue,
        };

        // Resolve the C type for this declaration.
        let c_type = decl_lowering::resolve_declaration_type(
            &decl.specifiers,
            &init_decl.declarator,
            ctx.target,
            ctx.name_table,
        );

        // C11 §6.7.1p7 / §6.2.2p5: A function declaration at block scope
        // (without `static`) has external linkage. Treat it like `extern`.
        // For example, `float fx();` inside main() declares fx as an
        // externally-defined function — it must NOT get a stack alloca.
        if matches!(c_type, CType::Function { .. }) {
            ctx.local_vars.remove(&var_name);
            // Only add a function declaration if the function is not
            // already known to the module (as a definition or declaration).
            let already_known = ctx.module.get_function(&var_name).is_some()
                || ctx.module.declarations().iter().any(|d| d.name == var_name);
            if !already_known {
                // Extract return type and param types from the CType::Function.
                let (ret_cty, param_ctypes) = match &c_type {
                    CType::Function {
                        return_type,
                        params,
                        ..
                    } => {
                        let ret_ir = IrType::from_ctype(return_type, ctx.target);
                        let param_irs: Vec<IrType> = params
                            .iter()
                            .map(|p| IrType::from_ctype(p, ctx.target))
                            .collect();
                        (ret_ir, param_irs)
                    }
                    _ => (IrType::I32, vec![]),
                };
                let decl = crate::ir::module::FunctionDeclaration::new(
                    var_name.clone(),
                    ret_cty,
                    param_ctypes,
                );
                ctx.module.add_declaration(decl);
            }
            continue;
        }

        // If the variable already has an alloca, check if the type matches.
        // In C, different block scopes can declare variables with the same
        // name but different types (e.g. `u32 t` in one case block and
        // `void *t` in another within the same switch statement).  When the
        // declared type differs from the pre-scanned type, we must create a
        // new alloca with the correct IR type and update the scope overrides.
        if ctx.local_vars.contains_key(&var_name) {
            // VLA variables: local_types has Pointer(elem, _) but
            // resolve_declaration_type returns Array(elem, None).  These
            // are intentionally different (the alloca holds a *pointer*
            // to the dynamically-allocated VLA memory).  Skip the
            // scope-shadowing re-alloca to preserve the Pointer type.
            if matches!(c_type, CType::Array(_, None)) {
                if let Some(CType::Pointer(_, _)) = ctx.local_types.get(&var_name) {
                    continue;
                }
            }

            // --- Block-scope variable shadowing (C11 §6.2.1) ---
            //
            // Detect whether this declaration shadows an outer variable.
            // A name is "claimed" the first time its actual source-level
            // declaration is lowered.  If a name is already claimed when a
            // new declaration for it is encountered, that new declaration
            // must shadow the outer one and needs its own alloca.
            //
            // Similarly, if the name belongs to a function parameter, a
            // new alloca is always required to avoid overwriting the
            // parameter's storage.
            //
            // For sequential (non-overlapping) scopes that re-use the same
            // variable name, the `claimed_vars` set is restored on scope
            // exit, so the name becomes "unclaimed" again and the pre-scan
            // alloca is safely reused.  This avoids the code-size blowup
            // that would occur if every nested declaration got a fresh
            // alloca unconditionally.
            let is_param_shadow = ctx.param_values.contains_key(&var_name)
                && ctx.param_values.get(&var_name) == ctx.local_vars.get(&var_name);
            let is_local_shadow = ctx.claimed_vars.contains(&var_name);

            if is_param_shadow || is_local_shadow {
                // Shadowing an outer variable — create a new alloca.
                let sized_c_type = infer_array_size_from_init(
                    &c_type,
                    init_decl.initializer.as_ref(),
                    ctx.name_table,
                );
                let resolved_c_type =
                    decl_lowering::resolve_sizeof_struct_ref_pub(sized_c_type.clone(), ctx.target);
                let ir_type = IrType::from_ctype(&resolved_c_type, ctx.target);
                let alloca_ir_type = {
                    let sz = ir_type.size_bytes(ctx.target);
                    if sz > 0 && sz <= 8 && !sz.is_power_of_two() {
                        IrType::Array(Box::new(IrType::I8), sz.next_power_of_two())
                    } else {
                        ir_type
                    }
                };
                let (alloca_val, alloca_inst) = ctx.builder.build_alloca(alloca_ir_type, decl.span);
                ctx.function.entry_block_mut().push_alloca(alloca_inst);
                ctx.local_vars.insert(var_name.clone(), alloca_val);
                // Store the RESOLVED type (with full struct field definitions)
                // so that lower_declaration_initializers can properly emit
                // struct field stores.  Using sized_c_type here would store a
                // forward-declared struct with empty fields, causing
                // initializers to be silently skipped.
                ctx.scope_type_overrides
                    .insert(var_name.clone(), resolved_c_type);
            } else {
                // First encounter of this variable — "claim" it and
                // reuse the pre-scan alloca if the type matches.
                ctx.claimed_vars.insert(var_name.clone());

                let effective_type = ctx
                    .scope_type_overrides
                    .get(&var_name)
                    .or_else(|| ctx.local_types.get(&var_name));
                let types_match = effective_type.map_or(false, |et| {
                    ctypes_compatible_for_scope_shadowing(et, &c_type, ctx.target)
                });
                if !types_match {
                    // Different type — create new alloca even for first encounter.
                    let sized_c_type = infer_array_size_from_init(
                        &c_type,
                        init_decl.initializer.as_ref(),
                        ctx.name_table,
                    );
                    let resolved_c_type = decl_lowering::resolve_sizeof_struct_ref_pub(
                        sized_c_type.clone(),
                        ctx.target,
                    );
                    let ir_type = IrType::from_ctype(&resolved_c_type, ctx.target);
                    let alloca_ir_type = {
                        let sz = ir_type.size_bytes(ctx.target);
                        if sz > 0 && sz <= 8 && !sz.is_power_of_two() {
                            IrType::Array(Box::new(IrType::I8), sz.next_power_of_two())
                        } else {
                            ir_type
                        }
                    };
                    let (alloca_val, alloca_inst) =
                        ctx.builder.build_alloca(alloca_ir_type, decl.span);
                    ctx.function.entry_block_mut().push_alloca(alloca_inst);
                    ctx.local_vars.insert(var_name.clone(), alloca_val);
                    // Store the RESOLVED type (with full struct field
                    // definitions) so that lower_declaration_initializers
                    // can properly emit struct field stores.
                    ctx.scope_type_overrides
                        .insert(var_name.clone(), resolved_c_type);
                }
            }
            continue;
        }
        // For unsized arrays (e.g., `char data[] = "hello"` or `int a[] = {1,2,3}`),
        // infer the size from the initializer BEFORE creating the alloca so that
        // the frame layout allocates the correct number of bytes.
        // Also resolve forward-referenced struct/union tags so the alloca
        // gets the correct struct field layout and size.
        let c_type =
            infer_array_size_from_init(&c_type, init_decl.initializer.as_ref(), ctx.name_table);
        let resolved_c_type =
            decl_lowering::resolve_sizeof_struct_ref_pub(c_type.clone(), ctx.target);
        let ir_type = IrType::from_ctype(&resolved_c_type, ctx.target);
        // Round up non-power-of-2 struct allocas so whole-struct register
        // stores (e.g. 3-byte struct stored via 4-byte movl) don't overflow.
        let alloca_ir_type = {
            let sz = ir_type.size_bytes(ctx.target);
            if sz > 0 && sz <= 8 && !sz.is_power_of_two() {
                IrType::Array(Box::new(IrType::I8), sz.next_power_of_two())
            } else {
                ir_type
            }
        };
        let (alloca_val, alloca_inst) = ctx.builder.build_alloca(alloca_ir_type, decl.span);
        ctx.function.entry_block_mut().push_alloca(alloca_inst);
        ctx.local_vars.insert(var_name.clone(), alloca_val);
    }
}

pub fn lower_declaration_initializers(ctx: &mut StmtLoweringContext<'_>, decl: &ast::Declaration) {
    // Skip extern declarations — they don't allocate stack storage.
    if matches!(
        decl.specifiers.storage_class,
        Some(ast::StorageClass::Extern)
    ) {
        return;
    }
    // Skip static declarations — they were lowered as globals.
    // Exception: static locals whose initializers contain label addresses
    // (&&label — computed goto) are downgraded to stack-allocated and need
    // runtime initialization.
    if matches!(
        decl.specifiers.storage_class,
        Some(ast::StorageClass::Static) | Some(ast::StorageClass::ThreadLocal)
    ) {
        let has_label_addr = decl.declarators.iter().any(|init_decl| {
            init_decl
                .initializer
                .as_ref()
                .map_or(false, decl_lowering::initializer_contains_label_address_pub)
        });
        if !has_label_addr {
            return;
        }
    }
    // Skip typedef declarations — but handle VLA typedefs specially.
    if matches!(
        decl.specifiers.storage_class,
        Some(ast::StorageClass::Typedef)
    ) {
        // For VLA typedefs like `typedef int c[i+2]`, we need to compute
        // the runtime total byte size so that `sizeof(c)` works.
        for init_decl in &decl.declarators {
            let td_name = match extract_decl_name(&init_decl.declarator, ctx.name_table) {
                Some(n) => n,
                None => continue,
            };
            if let Some(vla_expr) = decl_lowering::extract_vla_size_expr_pub(&init_decl.declarator)
            {
                // Determine the element type from the declaration specifiers.
                let elem_ty = {
                    use super::decl_lowering::resolve_base_type_from_sqlist;
                    let sqlist = crate::frontend::parser::ast::SpecifierQualifierList {
                        type_specifiers: decl.specifiers.type_specifiers.clone(),
                        type_qualifiers: decl.specifiers.type_qualifiers.clone(),
                        attributes: Vec::new(),
                        span: decl.span,
                    };
                    resolve_base_type_from_sqlist(&sqlist)
                };
                let span = decl.span;
                let elem_size = crate::common::types::sizeof_ctype(&elem_ty, ctx.target) as i64;
                let ptr_ty = if ctx.target.pointer_width() == 8 {
                    IrType::I64
                } else {
                    IrType::I32
                };
                // Build an expression context to evaluate the size expression.
                let mut expr_ctx = ExprLoweringContext {
                    builder: ctx.builder,
                    function: ctx.function,
                    module: ctx.module,
                    target: ctx.target,
                    type_builder: ctx.type_builder,
                    diagnostics: ctx.diagnostics,
                    local_vars: ctx.local_vars,
                    param_values: ctx.param_values,
                    name_table: ctx.name_table,
                    local_types: ctx.local_types,
                    enum_constants: ctx.enum_constants,
                    static_locals: ctx.static_locals,
                    struct_defs: ctx.struct_defs,
                    label_blocks: ctx.label_blocks,
                    current_function_name: ctx.current_function_name,
                    enclosing_loop_stack: ctx.loop_stack.clone(),
                    scope_type_overrides: &ctx.scope_type_overrides,
                    last_bitfield_info: None,
                    layout_cache: &mut ctx.layout_cache,
                    vla_sizes: &ctx.vla_sizes,
                };
                // Evaluate the VLA size expression at runtime.
                let tv = super::expr_lowering::lower_expression_typed(&mut expr_ctx, &vla_expr);
                let n_val = tv.value;
                // ZExt the count to pointer width if needed.
                let n_wide = {
                    let (v, inst) =
                        expr_ctx
                            .builder
                            .build_zext_from(n_val, IrType::I32, ptr_ty.clone(), span);
                    push_inst_to_block(&mut expr_ctx, inst);
                    v
                };
                // Emit element-size constant.
                let elem_const = {
                    use crate::ir::instructions::BinOp as IrBinOp;
                    use crate::ir::module as ir_module;
                    use crate::ir::module::{Constant, GlobalVariable};
                    let const_id = expr_ctx.module.globals().len();
                    let gname = format!(".Lconst.i.{}", const_id);
                    let mut gv = GlobalVariable::new(
                        gname,
                        ptr_ty.clone(),
                        Some(Constant::Integer(elem_size as i128)),
                    );
                    gv.linkage = ir_module::Linkage::Internal;
                    gv.is_constant = true;
                    expr_ctx.module.add_global(gv);
                    let result = expr_ctx.builder.fresh_value();
                    let inst = Instruction::BinOp {
                        result,
                        op: IrBinOp::Add,
                        lhs: result,
                        rhs: Value::UNDEF,
                        ty: ptr_ty.clone(),
                        span,
                    };
                    push_inst_to_block(&mut expr_ctx, inst);
                    expr_ctx.function.constant_values.insert(result, elem_size);
                    result
                };
                let (total, mul_inst) =
                    expr_ctx.builder.build_mul(n_wide, elem_const, ptr_ty, span);
                push_inst_to_block(&mut expr_ctx, mul_inst);
                drop(expr_ctx);
                // Store the total byte size in vla_sizes so sizeof(c) works.
                ctx.vla_sizes.insert(td_name, total);
            }
        }
        return;
    }

    for init_decl in &decl.declarators {
        // Extract the variable name from the declarator.
        let var_name = match extract_decl_name(&init_decl.declarator, ctx.name_table) {
            Some(name) => name,
            None => continue,
        };

        // ----- VLA dynamic allocation -----
        // Detect VLA declarations by checking the declarator for a
        // non-constant array size expression.  We cannot check local_types
        // here because VLA variables were overridden to CType::Pointer
        // during allocate_local_variables.
        if let Some(vla_expr) = decl_lowering::extract_vla_size_expr_pub(&init_decl.declarator) {
            // VLA variable detected — emit runtime stack allocation.
            //
            // Multi-dimensional VLA support: for `int a[n][m]`, we need to
            // compute `total = n * m * sizeof(int)` at runtime.  We extract
            // ALL array dimension expressions from the declarator and
            // multiply them together with the base element size.
            //
            // Determine the innermost non-array element type by peeling
            // through Pointer(Array(...)) layers in the local type.
            let base_elem = {
                let vty = ctx
                    .local_types
                    .get(&var_name)
                    .cloned()
                    .unwrap_or(CType::Int);
                let mut cur = match vty {
                    CType::Pointer(inner, _) => (*inner).clone(),
                    _ => CType::Int,
                };
                // Peel off Array layers to find the innermost element type.
                // For `int a[n][m]`, type is Pointer(Array(Int, None)):
                //   first peel: Array(Int, None) → inner = Int
                // For `int a[n][m][k]`, type is Pointer(Array(Array(Int, None), None)):
                //   first peel: Array(Array(Int, None), None) → Array(Int, None)
                //   second peel: Array(Int, None) → Int
                while let CType::Array(inner, _) = cur {
                    cur = (*inner).clone();
                }
                cur
            };
            // Look up the pointer-sized alloca for this VLA.
            if let Some(&alloca_val) = ctx.local_vars.get(&var_name) {
                let span = decl.span;
                let base_elem_size =
                    crate::common::types::sizeof_ctype(&base_elem, ctx.target) as i64;
                // Build an expression context to evaluate size expressions.
                let mut expr_ctx = ExprLoweringContext {
                    builder: ctx.builder,
                    function: ctx.function,
                    module: ctx.module,
                    target: ctx.target,
                    type_builder: ctx.type_builder,
                    diagnostics: ctx.diagnostics,
                    local_vars: ctx.local_vars,
                    param_values: ctx.param_values,
                    name_table: ctx.name_table,
                    local_types: ctx.local_types,
                    enum_constants: ctx.enum_constants,
                    static_locals: ctx.static_locals,
                    struct_defs: ctx.struct_defs,
                    label_blocks: ctx.label_blocks,
                    current_function_name: ctx.current_function_name,
                    enclosing_loop_stack: ctx.loop_stack.clone(),
                    scope_type_overrides: &ctx.scope_type_overrides,
                    last_bitfield_info: None,
                    layout_cache: &mut ctx.layout_cache,
                    vla_sizes: &ctx.vla_sizes,
                };
                let ptr_ty = if ctx.target.pointer_width() == 8 {
                    IrType::I64
                } else {
                    IrType::I32
                };
                // Extract ALL array dimension expressions from the
                // declarator.  For `int a[n][m]` this gives [n, m].
                let all_dims = decl_lowering::extract_all_vla_dims(&init_decl.declarator);
                // Start with base element size as the running product.
                let mut running_total = {
                    use crate::ir::instructions::BinOp as IrBinOp;
                    use crate::ir::module as ir_module;
                    use crate::ir::module::{Constant, GlobalVariable};
                    let const_id = expr_ctx.module.globals().len();
                    let gname = format!(".Lconst.i.{}", const_id);
                    let mut gv = GlobalVariable::new(
                        gname,
                        ptr_ty.clone(),
                        Some(Constant::Integer(base_elem_size as i128)),
                    );
                    gv.linkage = ir_module::Linkage::Internal;
                    gv.is_constant = true;
                    expr_ctx.module.add_global(gv);
                    let result = expr_ctx.builder.fresh_value();
                    let inst = Instruction::BinOp {
                        result,
                        op: IrBinOp::Add,
                        lhs: result,
                        rhs: Value::UNDEF,
                        ty: ptr_ty.clone(),
                        span,
                    };
                    push_inst_to_block(&mut expr_ctx, inst);
                    expr_ctx
                        .function
                        .constant_values
                        .insert(result, base_elem_size);
                    result
                };
                // Multiply each dimension expression into the running total.
                // For `int a[n][m]`: total = base_elem_size * n * m
                // We use all_dims if non-empty, otherwise fall back to the
                // single vla_expr extracted earlier.
                let dim_exprs: Vec<crate::frontend::parser::ast::Expression> =
                    if all_dims.is_empty() {
                        vec![*vla_expr]
                    } else {
                        all_dims
                    };
                // Save the base element size Value before the loop
                // modifies running_total — needed for stride computation.
                let base_elem_size_val = running_total;
                // Collect lowered dimension Values for stride computation.
                let mut dim_lowered_vals: Vec<Value> = Vec::new();
                for dim_expr in &dim_exprs {
                    let tv = super::expr_lowering::lower_expression_typed(&mut expr_ctx, dim_expr);
                    let dim_val = tv.value;
                    // ZExt to pointer width if needed.
                    let dim_wide = {
                        let (v, inst) = expr_ctx.builder.build_zext_from(
                            dim_val,
                            IrType::I32,
                            ptr_ty.clone(),
                            span,
                        );
                        push_inst_to_block(&mut expr_ctx, inst);
                        v
                    };
                    dim_lowered_vals.push(dim_wide);
                    let (product, mul_inst) =
                        expr_ctx
                            .builder
                            .build_mul(running_total, dim_wide, ptr_ty.clone(), span);
                    push_inst_to_block(&mut expr_ctx, mul_inst);
                    running_total = product;
                }
                let total = running_total;
                // Compute per-dimension row strides for multi-dimensional
                // VLA subscript lowering. For `int a[n][m]`:
                //   stride_0 = m * sizeof(int)  (row stride for a[i])
                //   stride_1 = sizeof(int)      (element stride for a[i][j])
                // Strides are computed right-to-left from the innermost
                // dimension outward.
                let mut stride_entries: Vec<(String, Value)> = Vec::new();
                if dim_lowered_vals.len() > 1 {
                    let mut row_stride = base_elem_size_val;
                    for d in (0..dim_lowered_vals.len()).rev() {
                        let key = format!("{}\0stride_{}", var_name, d);
                        stride_entries.push((key, row_stride));
                        if d > 0 {
                            let (product, mul_inst) = expr_ctx.builder.build_mul(
                                row_stride,
                                dim_lowered_vals[d],
                                ptr_ty.clone(),
                                span,
                            );
                            push_inst_to_block(&mut expr_ctx, mul_inst);
                            row_stride = product;
                        }
                    }
                }
                // VLA stack management: save/restore RSP so that VLAs
                // inside loops or goto-loops don't exhaust the stack.
                //
                // Strategy: emit StackSave in the ENTRY BLOCK so it
                // executes exactly once, then emit StackRestore before
                // every StackAlloc to reclaim previous VLA space.
                if ctx.vla_stack_save.is_none() {
                    // Insert StackSave at the END of the entry block
                    // (block 0), just before its terminator.
                    let (sp_val, save_inst) = expr_ctx.builder.build_stack_save(span);
                    // Insert into block 0 (entry block) before
                    // the terminator.
                    let entry_block = expr_ctx.function.get_block_mut(0).unwrap();
                    let insert_pos =
                        if entry_block.has_terminator() && !entry_block.instructions.is_empty() {
                            entry_block.instructions.len() - 1
                        } else {
                            entry_block.instructions.len()
                        };
                    entry_block.instructions.insert(insert_pos, save_inst);
                    ctx.vla_stack_save = Some(sp_val);
                }
                // Restore to the saved stack pointer before allocating.
                let saved_sp = ctx.vla_stack_save.unwrap();
                let restore_inst = expr_ctx.builder.build_stack_restore(saved_sp, span);
                push_inst_to_block(&mut expr_ctx, restore_inst);
                // Emit StackAlloc.
                let (ptr_val, sa_inst) = expr_ctx.builder.build_stack_alloc(total, span);
                push_inst_to_block(&mut expr_ctx, sa_inst);
                // Store the pointer into the alloca.
                let store_inst = expr_ctx.builder.build_store(ptr_val, alloca_val, span);
                push_inst_to_block(&mut expr_ctx, store_inst);
                // Record the total byte size so that sizeof(vla) works at runtime.
                drop(expr_ctx);
                ctx.vla_sizes.insert(var_name.clone(), total);
                // Store per-dimension strides for subscript lowering.
                for (key, val) in stride_entries {
                    ctx.vla_sizes.insert(key, val);
                }
            }
            continue; // VLA handled, skip normal init path.
        }

        // Only process declarators with initializers.
        let initializer = match &init_decl.initializer {
            Some(init) => init,
            None => continue,
        };

        // Look up the alloca Value for this variable.
        let alloca_val = match ctx.local_vars.get(&var_name) {
            Some(&val) => val,
            None => continue, // Variable not found — possibly a redeclaration or error.
        };

        // Determine the C type for this variable (needed for aggregate initializers).
        // Check scope_type_overrides FIRST — when a variable is shadowed in a
        // nested scope with a different type (e.g. `long long x` in the if-branch
        // and `double x` in the else-branch), ensure_allocas_for_declaration
        // updates scope_type_overrides with the correct type for the current
        // scope.  The immutable local_types map may have the WRONG type because
        // the pre-scan's HashMap overwrites earlier entries with later ones when
        // the same variable name appears in different branches.
        let var_type = ctx
            .scope_type_overrides
            .get(&var_name)
            .cloned()
            .or_else(|| ctx.local_types.get(&var_name).cloned())
            .unwrap_or(CType::Int);

        // Build an ExprLoweringContext and call decl_lowering::lower_local_initializer
        // to emit the Store instruction(s).
        let mut expr_ctx = ExprLoweringContext {
            builder: ctx.builder,
            function: ctx.function,
            module: ctx.module,
            target: ctx.target,
            type_builder: ctx.type_builder,
            diagnostics: ctx.diagnostics,
            local_vars: ctx.local_vars,
            param_values: ctx.param_values,
            name_table: ctx.name_table,
            local_types: ctx.local_types,
            enum_constants: ctx.enum_constants,
            static_locals: ctx.static_locals,
            struct_defs: ctx.struct_defs,
            label_blocks: ctx.label_blocks,
            current_function_name: ctx.current_function_name,
            enclosing_loop_stack: ctx.loop_stack.clone(),
            scope_type_overrides: &ctx.scope_type_overrides,
            last_bitfield_info: None,
            layout_cache: &mut ctx.layout_cache,
            vla_sizes: &ctx.vla_sizes,
        };
        decl_lowering::lower_local_initializer(alloca_val, initializer, &var_type, &mut expr_ctx);
    }
}

/// Extract the variable name from a declarator by walking into the
/// DirectDeclarator tree. Returns `None` if no identifier is found.
pub fn extract_decl_name(declarator: &ast::Declarator, name_table: &[String]) -> Option<String> {
    extract_dd_name(&declarator.direct, name_table)
}

/// Recursively extract the identifier name from a DirectDeclarator.
fn extract_dd_name(dd: &ast::DirectDeclarator, name_table: &[String]) -> Option<String> {
    match dd {
        ast::DirectDeclarator::Identifier(sym, _) => {
            let idx = sym.as_u32() as usize;
            if idx < name_table.len() {
                Some(name_table[idx].clone())
            } else {
                None
            }
        }
        ast::DirectDeclarator::Parenthesized(inner) => extract_decl_name(inner, name_table),
        ast::DirectDeclarator::Array { base, .. } => extract_dd_name(base, name_table),
        ast::DirectDeclarator::Function { base, .. } => extract_dd_name(base, name_table),
    }
}

// ===========================================================================
// Expression Statement
// ===========================================================================

/// Lowers an expression statement `expr;`.
///
/// Evaluates the expression for its side effects. The resulting value is
/// discarded. Uses a minimal [`ExprLoweringContext`] constructed from the
/// statement lowering context.
fn lower_expression_statement(ctx: &mut StmtLoweringContext<'_>, expr: &ast::Expression) {
    // We need to construct an ExprLoweringContext to call lower_expression.
    // Since we only need the side effects, we discard the returned Value.
    let _value = lower_expr_via_context(ctx, expr);
}

// ===========================================================================
// If/Else Statement
// ===========================================================================

/// Lowers an if/else statement.
///
/// Creates the following block structure:
/// ```text
///   [current]
///       |
///   CondBranch(cond, then_block, else_block_or_merge)
///      / \
/// [then]  [else]  (else may not exist)
///      \  /
///    [merge]
/// ```
fn lower_if_statement(
    ctx: &mut StmtLoweringContext<'_>,
    condition: &ast::Expression,
    then_body: &ast::Statement,
    else_body: Option<&ast::Statement>,
    span: Span,
) -> Option<BlockId> {
    // Create blocks.
    let then_block = create_block(ctx, "if.then");
    let merge_block = create_block(ctx, "if.end");
    let else_block = if else_body.is_some() {
        create_block(ctx, "if.else")
    } else {
        merge_block
    };

    // Lower condition expression.
    let cond_val = lower_expr_to_i1(ctx, condition, span);

    // Emit conditional branch.
    let cond_branch = ctx
        .builder
        .build_cond_branch(cond_val, then_block, else_block, span);
    emit_instruction_to_current_block(ctx, cond_branch);

    // Add CFG edges from current block.
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, then_block.index());
    add_cfg_edge(ctx, current_idx, else_block.index());

    // --- Then block ---
    ctx.builder.set_insert_point(then_block);
    let then_result = lower_statement(ctx, then_body);
    let then_terminated = then_result.is_none() || is_block_terminated(ctx);
    if !then_terminated {
        let br = ctx.builder.build_branch(merge_block, span);
        emit_instruction_to_current_block(ctx, br);
        let then_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(then_block.index());
        add_cfg_edge(ctx, then_end_idx, merge_block.index());
    }

    // --- Else block ---
    if let Some(else_stmt) = else_body {
        ctx.builder.set_insert_point(else_block);
        let else_result = lower_statement(ctx, else_stmt);
        let else_terminated = else_result.is_none() || is_block_terminated(ctx);
        if !else_terminated {
            let br = ctx.builder.build_branch(merge_block, span);
            emit_instruction_to_current_block(ctx, br);
            let else_end_idx = ctx
                .builder
                .get_insert_block()
                .map(|b| b.index())
                .unwrap_or(else_block.index());
            add_cfg_edge(ctx, else_end_idx, merge_block.index());
        }
    }

    // Set insert point to merge block.
    ctx.builder.set_insert_point(merge_block);
    Some(merge_block)
}

// ===========================================================================
// While Loop
// ===========================================================================

/// Lowers a `while (condition) body` loop.
///
/// Block structure:
/// ```text
/// [current] → Branch(header)
/// [header]  → CondBranch(cond, body, exit)
/// [body]    → ... → Branch(header)
/// [exit]    → (continuation)
/// ```
fn lower_while_loop(
    ctx: &mut StmtLoweringContext<'_>,
    condition: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let header_block = create_block(ctx, "while.cond");
    let body_block = create_block(ctx, "while.body");
    let exit_block = create_block(ctx, "while.end");

    // Branch from current to header.
    let br = ctx.builder.build_branch(header_block, span);
    emit_instruction_to_current_block(ctx, br);
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, header_block.index());

    // --- Header block: condition check ---
    ctx.builder.set_insert_point(header_block);
    let cond_val = lower_expr_to_i1(ctx, condition, span);
    let cond_br = ctx
        .builder
        .build_cond_branch(cond_val, body_block, exit_block, span);
    emit_instruction_to_current_block(ctx, cond_br);
    // Use actual current block (may differ from header_block when the
    // condition contains `&&`/`||` that create short-circuit blocks).
    let actual_header_end = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(header_block.index());
    add_cfg_edge(ctx, actual_header_end, body_block.index());
    add_cfg_edge(ctx, actual_header_end, exit_block.index());

    // --- Body block ---
    ctx.builder.set_insert_point(body_block);
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: header_block,
    });

    let body_result = lower_statement(ctx, body);
    let body_terminated = body_result.is_none() || is_block_terminated(ctx);
    if !body_terminated {
        let br = ctx.builder.build_branch(header_block, span);
        emit_instruction_to_current_block(ctx, br);
        let body_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(body_block.index());
        add_cfg_edge(ctx, body_end_idx, header_block.index());
    }

    ctx.loop_stack.pop();

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// Do-While Loop
// ===========================================================================

/// Lowers a `do body while (condition);` loop.
///
/// Block structure:
/// Returns `true` when `expr` is a compile-time constant that evaluates
/// to false (zero).  This is used to optimise `do { ... } while(0)` — the
/// canonical C multi-statement-macro wrapper — by eliding the condition
/// block entirely so the register allocator never clobbers live variables.
fn is_constant_false_expr(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => *value == 0,
        ast::Expression::FloatLiteral { value, .. } => *value == 0.0,
        // Cast of zero is still zero: `(int)0`, `(void*)0`, etc.
        ast::Expression::Cast { operand: inner, .. } => is_constant_false_expr(inner),
        // Parenthesised expression
        ast::Expression::Parenthesized { inner, .. } => is_constant_false_expr(inner),
        _ => false,
    }
}

/// ```text
/// [current] → Branch(body)
/// [body]    → ... → Branch(cond)
/// [cond]    → CondBranch(cond, body, exit)
/// [exit]    → (continuation)
/// ```
fn lower_do_while_loop(
    ctx: &mut StmtLoweringContext<'_>,
    body: &ast::Statement,
    condition: &ast::Expression,
    span: Span,
) -> Option<BlockId> {
    // Optimisation for `do { ... } while(0)` — the most common C idiom
    // for multi-statement macros.  When the condition is a constant zero
    // the body executes exactly once and no loop-back edge is needed.
    // Emitting the condition block would waste a register and, because
    // the register allocator might assign that register to a variable
    // that is live across the body, could silently corrupt locals.
    let condition_is_constant_false = is_constant_false_expr(condition);

    let body_block = create_block(ctx, "dowhile.body");
    let cond_block = if condition_is_constant_false {
        // Dummy — unused, but we still need a BlockId for LoopContext
        // (a `continue` inside the body jumps to exit when while(0)).
        body_block // placeholder, will redirect to exit below
    } else {
        create_block(ctx, "dowhile.cond")
    };
    let exit_block = create_block(ctx, "dowhile.end");

    // Branch to body block.
    let br = ctx.builder.build_branch(body_block, span);
    emit_instruction_to_current_block(ctx, br);
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, body_block.index());

    // --- Body block ---
    ctx.builder.set_insert_point(body_block);
    // For `do { ... } while(0)`, `continue` should jump directly
    // to the exit block (there is no next iteration).
    let continue_target = if condition_is_constant_false {
        exit_block
    } else {
        cond_block
    };
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target,
    });

    let body_result = lower_statement(ctx, body);
    let body_terminated = body_result.is_none() || is_block_terminated(ctx);
    if !body_terminated {
        if condition_is_constant_false {
            // `while(0)` — body falls through directly to exit.
            let br = ctx.builder.build_branch(exit_block, span);
            emit_instruction_to_current_block(ctx, br);
            let body_end_idx = ctx
                .builder
                .get_insert_block()
                .map(|b| b.index())
                .unwrap_or(body_block.index());
            add_cfg_edge(ctx, body_end_idx, exit_block.index());
        } else {
            let br = ctx.builder.build_branch(cond_block, span);
            emit_instruction_to_current_block(ctx, br);
            let body_end_idx = ctx
                .builder
                .get_insert_block()
                .map(|b| b.index())
                .unwrap_or(body_block.index());
            add_cfg_edge(ctx, body_end_idx, cond_block.index());
        }
    }

    ctx.loop_stack.pop();

    if !condition_is_constant_false {
        // --- Condition block (only when condition is not constant false) ---
        ctx.builder.set_insert_point(cond_block);
        let cond_val = lower_expr_to_i1(ctx, condition, span);
        let cond_br = ctx
            .builder
            .build_cond_branch(cond_val, body_block, exit_block, span);
        emit_instruction_to_current_block(ctx, cond_br);
        // CRITICAL: `lower_expr_to_i1` may create intermediate blocks for
        // short-circuit `&&` / `||` operators. The conditional branch above
        // is emitted in the *actual* current block (which may be a merge
        // block created by `&&`/`||`), NOT necessarily `cond_block`. We
        // must record CFG edges from the actual block that holds the branch,
        // otherwise phi-node predecessors at `body_block` (the do-while
        // header) will reference the wrong block, causing phi-elimination
        // to place back-edge copies in a block that doesn't directly
        // branch to the loop header — making loop variables stale.
        let actual_cond_end = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(cond_block.index());
        add_cfg_edge(ctx, actual_cond_end, body_block.index());
        add_cfg_edge(ctx, actual_cond_end, exit_block.index());
    }

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// For Loop
// ===========================================================================

/// Lowers a `for (init; cond; incr) body` loop.
///
/// Block structure:
/// ```text
/// [current] → lower init → Branch(cond)
/// [cond]    → CondBranch(cond, body, exit) or Branch(body) if no condition
/// [body]    → ... → Branch(incr)
/// [incr]    → lower increment → Branch(cond)
/// [exit]    → (continuation)
/// ```
fn lower_for_loop(
    ctx: &mut StmtLoweringContext<'_>,
    init: Option<&ast::ForInit>,
    condition: Option<&ast::Expression>,
    increment: Option<&ast::Expression>,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    // C11 §6.8.5p5: The for-loop's init-clause declaration has scope
    // limited to the for statement (including body).  Save local_vars
    // so that any variable declared in the init clause doesn't leak
    // into the enclosing scope after the for-loop completes.
    let saved_local_vars = ctx.local_vars.clone();
    let saved_scope_type_overrides = ctx.scope_type_overrides.clone();
    let saved_claimed_vars = ctx.claimed_vars.clone();

    // Lower init clause in the current block.
    if let Some(for_init) = init {
        match for_init {
            ast::ForInit::Expression(expr) => {
                let _val = lower_expr_via_context(ctx, expr);
            }
            ast::ForInit::Declaration(decl) => {
                // Ensure allocas exist (may be missing for statement
                // expression scopes) and emit Store instructions.
                ensure_allocas_for_declaration(ctx, decl);
                lower_declaration_initializers(ctx, decl);
            }
        }
    }

    let cond_block = create_block(ctx, "for.cond");
    let body_block = create_block(ctx, "for.body");
    let incr_block = create_block(ctx, "for.inc");
    let exit_block = create_block(ctx, "for.end");

    // Branch from current to condition.
    let br = ctx.builder.build_branch(cond_block, span);
    emit_instruction_to_current_block(ctx, br);
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, cond_block.index());

    // --- Condition block ---
    ctx.builder.set_insert_point(cond_block);
    if let Some(cond_expr) = condition {
        let cond_val = lower_expr_to_i1(ctx, cond_expr, span);
        let cond_br = ctx
            .builder
            .build_cond_branch(cond_val, body_block, exit_block, span);
        emit_instruction_to_current_block(ctx, cond_br);
        // Use actual current block — `lower_expr_to_i1` may create
        // short-circuit blocks for `&&`/`||` in the condition.
        let actual_cond_end = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(cond_block.index());
        add_cfg_edge(ctx, actual_cond_end, body_block.index());
        add_cfg_edge(ctx, actual_cond_end, exit_block.index());
    } else {
        // Infinite loop (no condition): always branch to body.
        let br = ctx.builder.build_branch(body_block, span);
        emit_instruction_to_current_block(ctx, br);
        add_cfg_edge(ctx, cond_block.index(), body_block.index());
    }

    // --- Body block ---
    ctx.builder.set_insert_point(body_block);
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: incr_block,
    });

    let body_result = lower_statement(ctx, body);
    let body_terminated = body_result.is_none() || is_block_terminated(ctx);
    if !body_terminated {
        let br = ctx.builder.build_branch(incr_block, span);
        emit_instruction_to_current_block(ctx, br);
        let body_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(body_block.index());
        add_cfg_edge(ctx, body_end_idx, incr_block.index());
    }

    ctx.loop_stack.pop();

    // --- Increment block ---
    ctx.builder.set_insert_point(incr_block);
    if let Some(incr_expr) = increment {
        let _val = lower_expr_via_context(ctx, incr_expr);
    }
    let br = ctx.builder.build_branch(cond_block, span);
    emit_instruction_to_current_block(ctx, br);
    add_cfg_edge(ctx, incr_block.index(), cond_block.index());

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);

    // Restore local variable scope — variables declared in the for-loop's
    // init clause must not be visible after the loop.
    *ctx.local_vars = saved_local_vars;
    ctx.scope_type_overrides = saved_scope_type_overrides;
    ctx.claimed_vars = saved_claimed_vars;

    Some(exit_block)
}

// ===========================================================================
// Switch Statement
// ===========================================================================

/// Lowers a `switch (value) { ... }` statement.
///
/// The switch is lowered in two phases:
/// 1. **Body lowering**: The body (typically a compound statement containing
///    case/default labels) is lowered. Each `case`/`default` label registers
///    its block in the `SwitchContext`.
/// 2. **Dispatch construction**: After the body is fully lowered, the collected
///    case/default information is used to build a `Switch` IR instruction in the
///    dispatch block.
///
/// Block structure:
/// ```text
/// [current] -> dispatch block saves position
/// [body]    -> case blocks register themselves
/// [dispatch] -> Switch(val, default, cases) built after body
/// [exit]    -> continuation
/// ```
/// Emit an integer constant value from within a `StmtLoweringContext`.
/// Uses the same Add+UNDEF sentinel pattern as `emit_int_const` in
/// `expr_lowering`, but works with the stmt-level context directly.
fn emit_switch_range_const(
    ctx: &mut StmtLoweringContext<'_>,
    value: i128,
    ir_ty: IrType,
    span: Span,
) -> Value {
    let const_id = ctx.module.globals().len();
    let gname = format!(".Lconst.i.{}", const_id);
    let mut gv = crate::ir::module::GlobalVariable::new(
        gname,
        ir_ty.clone(),
        Some(crate::ir::module::Constant::Integer(value)),
    );
    gv.linkage = crate::ir::module::Linkage::Internal;
    gv.is_constant = true;
    ctx.module.add_global(gv);

    let result = ctx.builder.fresh_value();
    let inst = Instruction::BinOp {
        result,
        op: crate::ir::instructions::BinOp::Add,
        lhs: result,
        rhs: Value::UNDEF,
        ty: ir_ty,
        span,
    };
    emit_instruction_to_current_block(ctx, inst);
    ctx.function.constant_values.insert(result, value as i64);
    result
}

fn lower_switch(
    ctx: &mut StmtLoweringContext<'_>,
    condition: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let exit_block = create_block(ctx, "switch.end");

    // Lower the switch value expression with type information.
    // Per C11 §6.8.4.2p5, the integer promotions are performed on the
    // controlling expression.  This matters for signed sub-int types
    // (signed char, short): `(signed char)(-1)` must promote to int
    // as -1 (sign-extended), not 255 (zero-extended).
    let switch_tv = lower_expr_typed_via_context(ctx, condition);
    let promoted_ctype = crate::common::types::integer_promotion(&switch_tv.ty);
    let switch_val = if promoted_ctype != switch_tv.ty {
        // Need to promote: sign-extend or zero-extend based on the
        // signedness of the *original* type.
        let from_ir = crate::ir::lowering::expr_lowering::ctype_to_ir(&switch_tv.ty, ctx.target);
        let to_ir = crate::ir::lowering::expr_lowering::ctype_to_ir(&promoted_ctype, ctx.target);
        let is_signed = crate::common::types::is_signed(&switch_tv.ty);
        if is_signed {
            let (v, inst) = ctx.builder.build_sext(switch_tv.value, to_ir.clone(), span);
            let mut patched = inst;
            if let Instruction::SExt {
                ref mut from_type, ..
            } = patched
            {
                *from_type = from_ir;
            }
            emit_instruction_to_current_block(ctx, patched);
            v
        } else {
            let (v, inst) = ctx.builder.build_zext(switch_tv.value, to_ir.clone(), span);
            let mut patched = inst;
            if let Instruction::ZExt {
                ref mut from_type, ..
            } = patched
            {
                *from_type = from_ir;
            }
            emit_instruction_to_current_block(ctx, patched);
            v
        }
    } else {
        switch_tv.value
    };

    // Remember the dispatch block — we will emit the Switch instruction
    // here after processing the body.
    let dispatch_block = ctx.builder.get_insert_block().unwrap_or(BlockId(0));

    // Save previous switch context (for nested switches).
    let prev_switch = ctx.switch_ctx.take();

    // Create new switch context.
    ctx.switch_ctx = Some(SwitchContext {
        break_target: exit_block,
        case_blocks: Vec::new(),
        case_ranges: Vec::new(),
        default_block: None,
    });

    // Create a body entry block (the first case statement will set the
    // actual insert point, but we need a block in case the body starts
    // with non-case statements).
    let body_entry = create_block(ctx, "switch.body");

    // Don't emit a branch from dispatch to body_entry yet — the Switch
    // instruction will handle dispatch. We need to be in body_entry
    // for the body lowering to work.
    ctx.builder.set_insert_point(body_entry);

    // Push break target for the switch (break in switch goes to exit).
    // continue_target is u32::MAX to signal that continue passes through
    // the switch to an enclosing loop.
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: BlockId(u32::MAX),
    });

    // Lower the switch body.
    let _body_result = lower_statement(ctx, body);

    // If the body doesn't terminate, add fallthrough to exit.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(exit_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), exit_block.index());
        }
    }

    ctx.loop_stack.pop();

    // Extract collected case/default info.
    let switch_info = ctx.switch_ctx.take().unwrap_or(SwitchContext {
        break_target: exit_block,
        case_blocks: Vec::new(),
        case_ranges: Vec::new(),
        default_block: None,
    });

    // Restore previous switch context.
    ctx.switch_ctx = prev_switch;

    // Determine default target.
    let default_target = switch_info.default_block.unwrap_or(exit_block);

    // Build the Switch instruction in the dispatch block.
    ctx.builder.set_insert_point(dispatch_block);

    // Emit range checks BEFORE the cascaded Switch comparisons.
    // For each case range (low, high, target), emit:
    //   if (val >= low && val <= high) goto target
    // This handles large ranges that cannot be enumerated.
    let mut current_dispatch = dispatch_block;
    for &(low, high, target) in &switch_info.case_ranges {
        let ir_ty = crate::ir::types::IrType::I64;

        // Use unsigned range check: (val - low) <=u (high - low).
        // This works for both signed and unsigned switch values and
        // avoids signed overflow issues when case values exceed i64 max.
        let low_val = emit_switch_range_const(ctx, low as i128, ir_ty.clone(), span);
        let (sub_val, sub_inst) = ctx
            .builder
            .build_sub(switch_val, low_val, ir_ty.clone(), span);
        emit_instruction_to_current_block(ctx, sub_inst);

        let range_width = (high as u64).wrapping_sub(low as u64);
        let range_val =
            emit_switch_range_const(ctx, range_width as i64 as i128, ir_ty.clone(), span);
        let (in_range, cmp_inst) = ctx.builder.build_icmp(
            crate::ir::instructions::ICmpOp::Ule,
            sub_val,
            range_val,
            span,
        );
        emit_instruction_to_current_block(ctx, cmp_inst);

        // Create a fallthrough block for the next check or the Switch.
        let next_block = create_block(ctx, "switch.range_next");
        let cond_br = ctx
            .builder
            .build_cond_branch(in_range, target, next_block, span);
        emit_instruction_to_current_block(ctx, cond_br);

        add_cfg_edge(ctx, current_dispatch.index(), target.index());
        add_cfg_edge(ctx, current_dispatch.index(), next_block.index());

        ctx.builder.set_insert_point(next_block);
        current_dispatch = next_block;
    }

    let switch_inst = ctx.builder.build_switch(
        switch_val,
        default_target,
        switch_info.case_blocks.clone(),
        span,
    );
    emit_instruction_to_current_block(ctx, switch_inst);

    // Add CFG edges from current dispatch to all case targets and default.
    add_cfg_edge(ctx, current_dispatch.index(), default_target.index());
    for &(_val, target) in &switch_info.case_blocks {
        add_cfg_edge(ctx, current_dispatch.index(), target.index());
    }

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// Case Label
// ===========================================================================

/// Lowers a `case value: statement` label within a switch body.
///
/// Creates a new block for this case, registers the (value, block_id) pair
/// in the current SwitchContext, and lowers the body in the new block.
/// If the current block has no terminator, a fallthrough branch is emitted.
fn lower_case(
    ctx: &mut StmtLoweringContext<'_>,
    value_expr: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let case_block = create_block(ctx, "switch.case");

    // Evaluate the case value as a compile-time constant.
    let case_val = eval_case_constant(value_expr, ctx.name_table, ctx.enum_constants);

    // Register in the switch context.
    if let Some(ref mut sctx) = ctx.switch_ctx {
        sctx.case_blocks.push((case_val, case_block));
    }

    // Emit fallthrough from current block if it is not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(case_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), case_block.index());
        }
    }

    // Switch to case block and lower the body.
    ctx.builder.set_insert_point(case_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Case Range (GCC Extension)
// ===========================================================================

/// Lowers a `case low ... high: statement` (GCC case range extension).
///
/// For small ranges (≤256 values), registers each value individually in
/// the SwitchContext.  For large ranges, stores as a
/// `(low, high, target)` range entry that the switch dispatch will emit
/// as a pair of comparisons: `val >= low && val <= high`.
fn lower_case_range(
    ctx: &mut StmtLoweringContext<'_>,
    low_expr: &ast::Expression,
    high_expr: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let case_block = create_block(ctx, "switch.case_range");

    let low_val = eval_case_constant(low_expr, ctx.name_table, ctx.enum_constants);
    let high_val = eval_case_constant(high_expr, ctx.name_table, ctx.enum_constants);

    // Register in the switch context.
    if let Some(ref mut sctx) = ctx.switch_ctx {
        let range_size = (high_val as u64)
            .wrapping_sub(low_val as u64)
            .wrapping_add(1);
        if range_size <= 256 {
            // Small range: enumerate individual values.
            for i in 0..range_size {
                sctx.case_blocks
                    .push((low_val.wrapping_add(i as i64), case_block));
            }
        } else {
            // Large range: store as a range entry for comparison-based
            // dispatch instead of enumerating all values.
            sctx.case_ranges.push((low_val, high_val, case_block));
        }
    }

    // Fallthrough from current block if not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(case_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), case_block.index());
        }
    }

    ctx.builder.set_insert_point(case_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Default Label
// ===========================================================================

/// Lowers a `default: statement` label within a switch body.
fn lower_default(
    ctx: &mut StmtLoweringContext<'_>,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let default_block = create_block(ctx, "switch.default");

    // Register in the switch context.
    if let Some(ref mut sctx) = ctx.switch_ctx {
        if sctx.default_block.is_some() {
            ctx.diagnostics.emit(Diagnostic::warning(
                span,
                "duplicate 'default' label in switch statement",
            ));
        }
        sctx.default_block = Some(default_block);
    }

    // Fallthrough from current block if not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(default_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), default_block.index());
        }
    }

    ctx.builder.set_insert_point(default_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Break Statement
// ===========================================================================

/// Lowers a `break;` statement.
///
/// Branches to the innermost loop or switch break target. Returns `None`
/// because the current block is terminated after the branch.
fn lower_break(ctx: &mut StmtLoweringContext<'_>, span: Span) -> Option<BlockId> {
    // The loop_stack contains both loop and switch break targets. The topmost
    // entry is the innermost enclosing construct that accepts break.
    let target = if let Some(loop_ctx) = ctx.loop_stack.last() {
        loop_ctx.break_target
    } else if let Some(ref sctx) = ctx.switch_ctx {
        sctx.break_target
    } else {
        ctx.diagnostics.emit(Diagnostic::error(
            span,
            "'break' statement not within loop or switch",
        ));
        return None;
    };

    let br = ctx.builder.build_branch(target, span);
    emit_instruction_to_current_block(ctx, br);
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), target.index());
    }
    None // Block is terminated.
}

// ===========================================================================
// Continue Statement
// ===========================================================================

/// Lowers a `continue;` statement.
///
/// Branches to the innermost loop's continue target. Returns `None`
/// because the current block is terminated after the branch.
fn lower_continue(ctx: &mut StmtLoweringContext<'_>, span: Span) -> Option<BlockId> {
    // Find the innermost loop context (skip switch-only contexts whose
    // continue_target is the sentinel u32::MAX).
    let continue_target = ctx
        .loop_stack
        .iter()
        .rev()
        .find(|lc| lc.continue_target != BlockId(u32::MAX))
        .map(|lc| lc.continue_target);

    let target = match continue_target {
        Some(t) => t,
        None => {
            ctx.diagnostics.emit(Diagnostic::error(
                span,
                "'continue' statement not within a loop",
            ));
            return None;
        }
    };

    let br = ctx.builder.build_branch(target, span);
    emit_instruction_to_current_block(ctx, br);
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), target.index());
    }
    None // Block is terminated.
}

// ===========================================================================
// Return Statement
// ===========================================================================

/// Lowers a `return [expr];` statement.
///
/// If the return expression is present, it is evaluated first. Returns `None`
/// because the current block is terminated by the Return instruction.
///
/// Type checking: If the function returns `CType::Void` and a value is
/// provided, or vice versa, a diagnostic is emitted. The `IrType::Void`
/// return type is used to determine this.
fn lower_return(
    ctx: &mut StmtLoweringContext<'_>,
    value: Option<&ast::Expression>,
    span: Span,
) -> Option<BlockId> {
    let return_val = if let Some(expr) = value {
        // Verify function does not have a void return type.
        if ctx.function.return_type == IrType::Void {
            ctx.diagnostics.emit(Diagnostic::warning(
                span,
                "returning a value from a void function",
            ));
        }
        // Use typed lowering to get both the Value and the C type of
        // the expression.  This enables correct implicit conversion to
        // the function's return type (e.g., `signed char` → `int`
        // needs SExt, not zero-extension).
        let tv = lower_expr_typed_via_context(ctx, expr);
        let mut val = tv.value;

        // Insert implicit conversion if the function has a known C
        // return type and the expression's C type differs.
        if let Some(ref ret_cty) = ctx.return_ctype {
            let mut expr_ctx = ExprLoweringContext {
                builder: ctx.builder,
                function: ctx.function,
                module: ctx.module,
                target: ctx.target,
                type_builder: ctx.type_builder,
                diagnostics: ctx.diagnostics,
                local_vars: ctx.local_vars,
                param_values: ctx.param_values,
                name_table: ctx.name_table,
                local_types: ctx.local_types,
                enum_constants: ctx.enum_constants,
                static_locals: ctx.static_locals,
                struct_defs: ctx.struct_defs,
                label_blocks: ctx.label_blocks,
                current_function_name: ctx.current_function_name,
                enclosing_loop_stack: ctx.loop_stack.clone(),
                scope_type_overrides: &ctx.scope_type_overrides,
                last_bitfield_info: None,
                layout_cache: &mut ctx.layout_cache,
                vla_sizes: &ctx.vla_sizes,
            };
            val = insert_implicit_conversion(&mut expr_ctx, val, &tv.ty, ret_cty, span);
        }
        Some(val)
    } else {
        // Verify function has a void return type when no value is returned.
        if ctx.function.return_type != IrType::Void {
            // This is valid in C (implicit return of undefined value) but
            // we emit a warning for correctness.
            ctx.diagnostics.emit(Diagnostic::warning(
                span,
                "non-void function should return a value",
            ));
        }
        None
    };

    let ret = ctx.builder.build_return(return_val, span);
    emit_instruction_to_current_block(ctx, ret);
    None // Block is terminated.
}

// ===========================================================================
// Goto Statement
// ===========================================================================

/// Lowers a `goto label;` statement.
///
/// Lazily creates the target block if the label has not yet been seen
/// (forward reference). Returns `None` because the current block is
/// terminated by the branch.
fn lower_goto(
    ctx: &mut StmtLoweringContext<'_>,
    label: &crate::common::string_interner::Symbol,
    span: Span,
) -> Option<BlockId> {
    let label_name = format!("label_{}", label.as_u32());
    let target = get_or_create_label_block(ctx, &label_name);

    let br = ctx.builder.build_branch(target, span);
    emit_instruction_to_current_block(ctx, br);
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), target.index());
    }
    None // Block is terminated.
}

// ===========================================================================
// Computed Goto (GCC Extension)
// ===========================================================================

/// Lowers a `goto *expr;` (GCC computed goto extension).
///
/// The expression evaluates to a pointer (address of a label obtained via
/// `&&label`).  We emit an `IndirectBranch` instruction whose target is
/// the loaded code-address value.  The `possible_targets` list (for CFG
/// purposes) is the set of all label blocks in the current function.
fn lower_computed_goto(
    ctx: &mut StmtLoweringContext<'_>,
    target_expr: &ast::Expression,
    span: Span,
) -> Option<BlockId> {
    // Evaluate the target expression — typically `j[i]` where `j` was
    // initialized with `&&label` values.  This produces a `void*` that
    // is the actual code address of the target label.
    let addr_val = lower_expr_via_context(ctx, target_expr);

    // Collect all known label blocks as possible indirect-branch targets
    // (for CFG construction and phi-node placement).
    let possible_targets: Vec<crate::ir::instructions::BlockId> =
        ctx.label_blocks.values().copied().collect();

    // Build the indirect branch instruction.
    let ibr = ctx
        .builder
        .build_indirect_branch(addr_val, possible_targets.clone(), span);
    emit_instruction_to_current_block(ctx, ibr);

    // Add CFG edges to all possible targets.
    if let Some(current) = ctx.builder.get_insert_block() {
        for &target in &possible_targets {
            add_cfg_edge(ctx, current.index(), target.index());
        }
    }

    None // Block is terminated.
}

// ===========================================================================
// Label Statement
// ===========================================================================

/// Lowers a `label: statement` (labeled statement).
///
/// Creates (or looks up) the block for this label, emits a fallthrough
/// branch from the current block if needed, and lowers the labeled
/// statement in the label block.
fn lower_label(
    ctx: &mut StmtLoweringContext<'_>,
    label: &crate::common::string_interner::Symbol,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let label_name = format!("label_{}", label.as_u32());
    let label_block = get_or_create_label_block(ctx, &label_name);

    // Emit fallthrough from current block if not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(label_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), label_block.index());
        }
    }

    ctx.builder.set_insert_point(label_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Local Label (GCC Extension)
// ===========================================================================

/// Registers a GCC `__label__` local label declaration.
///
/// Creates a block for the local label and registers it in the label map.
/// Local labels have block scope (not function scope) per the GCC extension
/// semantics.
fn register_local_label(
    ctx: &mut StmtLoweringContext<'_>,
    label: &crate::common::string_interner::Symbol,
    _span: Span,
) {
    let label_name = format!("local_label_{}", label.as_u32());
    if !ctx.label_blocks.contains_key(&label_name) {
        let block = create_block(ctx, &label_name);
        ctx.label_blocks.insert(label_name, block);
    }
}

// ===========================================================================
// Inline Assembly Dispatch
// ===========================================================================

/// Dispatches an inline assembly statement to the asm_lowering module.
///
/// Constructs the operand arrays required by `asm_lowering::lower_inline_asm`
/// and calls into it.
fn lower_asm_dispatch(
    ctx: &mut StmtLoweringContext<'_>,
    asm_stmt: &ast::AsmStatement,
) -> Option<BlockId> {
    let span = asm_stmt.span;

    // Build output pointer values (from output operand expressions).
    // Outputs are LVALUES — we need the ADDRESS (alloca pointer) so the
    // asm result can be stored there, not the loaded value.
    let output_ptrs: Vec<Value> = asm_stmt
        .outputs
        .iter()
        .map(|op| lower_lvalue_via_context(ctx, &op.expression))
        .collect();

    // Build input values (from input operand expressions).
    let input_vals: Vec<Value> = asm_stmt
        .inputs
        .iter()
        .map(|op| lower_expr_via_context(ctx, &op.expression))
        .collect();

    // Build goto label names (for asm goto).
    let goto_label_names: Vec<String> = asm_stmt
        .goto_labels
        .iter()
        .map(|sym| format!("label_{}", sym.as_u32()))
        .collect();

    // Build named operand strings (symbolic names for operands).
    // Resolve Symbol handles to actual interned strings via name_table.
    let mut named_operand_strs: Vec<Option<String>> = Vec::new();
    for op in &asm_stmt.outputs {
        named_operand_strs.push(op.symbolic_name.as_ref().and_then(|s| {
            let idx = s.as_u32() as usize;
            if idx < ctx.name_table.len() {
                Some(ctx.name_table[idx].clone())
            } else {
                None
            }
        }));
    }
    for op in &asm_stmt.inputs {
        named_operand_strs.push(op.symbolic_name.as_ref().and_then(|s| {
            let idx = s.as_u32() as usize;
            if idx < ctx.name_table.len() {
                Some(ctx.name_table[idx].clone())
            } else {
                None
            }
        }));
    }

    // Call asm_lowering module.
    let _result = asm_lowering::lower_inline_asm(
        asm_stmt,
        ctx.builder,
        ctx.function,
        ctx.target,
        ctx.diagnostics,
        ctx.local_vars,
        ctx.label_blocks,
        &output_ptrs,
        &input_vals,
        &goto_label_names,
        &named_operand_strs,
        span,
    );

    ctx.builder.get_insert_block()
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Creates a new basic block with the given label, adds it to the function,
/// and returns its `BlockId`.
fn create_block(ctx: &mut StmtLoweringContext<'_>, label: &str) -> BlockId {
    let block_id = ctx.builder.create_block();
    let block = BasicBlock::with_label(block_id.index(), label.to_string());
    ctx.function.add_block(block);
    block_id
}

/// Checks if the current insertion block already has a terminator instruction.
fn is_block_terminated(ctx: &StmtLoweringContext<'_>) -> bool {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        if let Some(block) = ctx.function.get_block(block_id.index()) {
            return block.has_terminator();
        }
    }
    false
}

/// If the current block has no terminator, emit `Branch(target)`.
/// Prevents unterminated blocks which would be invalid IR.
#[allow(dead_code)]
fn ensure_terminator(ctx: &mut StmtLoweringContext<'_>, target: BlockId) {
    if !is_block_terminated(ctx) {
        let br = ctx.builder.build_branch(target, Span::dummy());
        emit_instruction_to_current_block(ctx, br);
        if let Some(current) = ctx.builder.get_insert_block() {
            add_cfg_edge(ctx, current.index(), target.index());
        }
    }
}

/// Pushes an instruction into the current block using an `ExprLoweringContext`.
/// Used by VLA allocation code that operates within an expression context.
fn push_inst_to_block(ctx: &mut ExprLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let idx = block_id.index();
        if let Some(block) = ctx.function.blocks.get_mut(idx) {
            block.push_instruction(inst);
        }
    }
}

/// Emits an instruction into the current insertion block.
fn emit_instruction_to_current_block(ctx: &mut StmtLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let idx = block_id.index();
        if let Some(block) = ctx.function.blocks.get_mut(idx) {
            block.push_instruction(inst);
        }
    }
}

/// Adds a CFG edge (predecessor/successor) between two blocks.
fn add_cfg_edge(ctx: &mut StmtLoweringContext<'_>, from_idx: usize, to_idx: usize) {
    // Add successor to the `from` block.
    if let Some(from_block) = ctx.function.blocks.get_mut(from_idx) {
        from_block.add_successor(to_idx);
    }
    // Add predecessor to the `to` block.
    if let Some(to_block) = ctx.function.blocks.get_mut(to_idx) {
        to_block.add_predecessor(from_idx);
    }
}

/// Gets or lazily creates a label block for goto/label resolution.
///
/// If the label has already been created (either by a previous `goto` forward
/// reference or by the label definition itself), the existing block is returned.
/// Otherwise a new block is created and registered.
fn get_or_create_label_block(ctx: &mut StmtLoweringContext<'_>, label_name: &str) -> BlockId {
    if let Some(&block_id) = ctx.label_blocks.get(label_name) {
        return block_id;
    }
    let block_id = ctx.builder.create_block();
    let block = BasicBlock::with_label(block_id.index(), label_name.to_string());
    ctx.function.add_block(block);
    ctx.label_blocks.insert(label_name.to_string(), block_id);
    block_id
}

/// Extracts the source span from a statement AST node.
fn statement_span(stmt: &ast::Statement) -> Span {
    match stmt {
        ast::Statement::Compound(c) => c.span,
        ast::Statement::Expression(_) => Span::dummy(),
        ast::Statement::If { span, .. } => *span,
        ast::Statement::Switch { span, .. } => *span,
        ast::Statement::While { span, .. } => *span,
        ast::Statement::DoWhile { span, .. } => *span,
        ast::Statement::For { span, .. } => *span,
        ast::Statement::Goto { span, .. } => *span,
        ast::Statement::ComputedGoto { span, .. } => *span,
        ast::Statement::Continue { span } => *span,
        ast::Statement::Break { span } => *span,
        ast::Statement::Return { span, .. } => *span,
        ast::Statement::Labeled { span, .. } => *span,
        ast::Statement::Case { span, .. } => *span,
        ast::Statement::CaseRange { span, .. } => *span,
        ast::Statement::Default { span, .. } => *span,
        ast::Statement::Declaration(decl) => decl.span,
        ast::Statement::Asm(asm) => asm.span,
        ast::Statement::LocalLabel(_, span) => *span,
    }
}

/// Evaluates a case constant expression to an i64 value.
///
/// For the IR lowering phase this extracts the compile-time integer value
/// from the expression. Full constant evaluation is done by the semantic
/// analyzer; here we handle the common cases that survive into the AST.
fn eval_case_constant(
    expr: &ast::Expression,
    name_table: &[String],
    enum_constants: &crate::common::fx_hash::FxHashMap<String, i128>,
) -> i64 {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => *value as i64,
        ast::Expression::CharLiteral { value, .. } => *value as i64,
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Negate,
            operand,
            ..
        } => -eval_case_constant(operand, name_table, enum_constants),
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::BitwiseNot,
            operand,
            ..
        } => !eval_case_constant(operand, name_table, enum_constants),
        ast::Expression::Parenthesized { inner, .. } => {
            eval_case_constant(inner, name_table, enum_constants)
        }
        ast::Expression::Cast { operand: inner, .. } => {
            eval_case_constant(inner, name_table, enum_constants)
        }
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let l = eval_case_constant(left, name_table, enum_constants);
            let r = eval_case_constant(right, name_table, enum_constants);
            match op {
                ast::BinaryOp::Add => l.wrapping_add(r),
                ast::BinaryOp::Sub => l.wrapping_sub(r),
                ast::BinaryOp::Mul => l.wrapping_mul(r),
                ast::BinaryOp::Div if r != 0 => l.wrapping_div(r),
                ast::BinaryOp::Mod if r != 0 => l.wrapping_rem(r),
                ast::BinaryOp::BitwiseAnd => l & r,
                ast::BinaryOp::BitwiseOr => l | r,
                ast::BinaryOp::BitwiseXor => l ^ r,
                ast::BinaryOp::ShiftLeft => l.wrapping_shl(r as u32),
                ast::BinaryOp::ShiftRight => l.wrapping_shr(r as u32),
                _ => 0,
            }
        }
        ast::Expression::Identifier { name, .. } => {
            // Resolve enum constants using the name table and enum constant map.
            let name_str = &name_table[name.as_u32() as usize];
            if let Some(&val) = enum_constants.get(name_str.as_str()) {
                val as i64
            } else {
                0
            }
        }
        ast::Expression::SizeofExpr { .. } | ast::Expression::SizeofType { .. } => {
            // sizeof in case expressions — rare but legal.
            // Would need target info; return 0 as fallback.
            0
        }
        _ => {
            // For non-trivial constant expressions the semantic analyzer
            // should have folded them before they reach IR lowering.
            // Return 0 as a safe fallback.
            0
        }
    }
}

/// Lowers an expression using the expression lowering subsystem.
///
/// This function bridges between `StmtLoweringContext` and the
/// `expr_lowering` module. It constructs a minimal `ExprLoweringContext`
/// from the available statement lowering state and delegates to
/// `expr_lowering::lower_expression`.
///
/// The parent lowering driver (`src/ir/lowering/mod.rs`) is responsible
/// for ensuring the full context (including `type_builder`, `param_values`,
/// `name_table`, `local_types`) is available when the driver invokes
/// statement lowering.
fn lower_expr_via_context(ctx: &mut StmtLoweringContext<'_>, expr: &ast::Expression) -> Value {
    // Construct a full ExprLoweringContext from the StmtLoweringContext
    // and delegate to expr_lowering::lower_expression.
    let mut expr_ctx = ExprLoweringContext {
        builder: ctx.builder,
        function: ctx.function,
        module: ctx.module,
        target: ctx.target,
        type_builder: ctx.type_builder,
        diagnostics: ctx.diagnostics,
        local_vars: ctx.local_vars,
        param_values: ctx.param_values,
        name_table: ctx.name_table,
        local_types: ctx.local_types,
        enum_constants: ctx.enum_constants,
        static_locals: ctx.static_locals,
        struct_defs: ctx.struct_defs,
        label_blocks: ctx.label_blocks,
        current_function_name: ctx.current_function_name,
        enclosing_loop_stack: ctx.loop_stack.clone(),
        scope_type_overrides: &ctx.scope_type_overrides,
        last_bitfield_info: None,
        layout_cache: &mut ctx.layout_cache,
        vla_sizes: &ctx.vla_sizes,
    };
    lower_expression(&mut expr_ctx, expr)
}

/// Lower an expression via a `StmtLoweringContext`, returning both the
/// IR `Value` and the expression's C-level type.  Used by `lower_return`
/// so that the caller can insert a correct implicit conversion (SExt for
/// signed types, ZExt for unsigned) to the function's return type.
fn lower_expr_typed_via_context(
    ctx: &mut StmtLoweringContext<'_>,
    expr: &ast::Expression,
) -> TypedValue {
    let mut expr_ctx = ExprLoweringContext {
        builder: ctx.builder,
        function: ctx.function,
        module: ctx.module,
        target: ctx.target,
        type_builder: ctx.type_builder,
        diagnostics: ctx.diagnostics,
        local_vars: ctx.local_vars,
        param_values: ctx.param_values,
        name_table: ctx.name_table,
        local_types: ctx.local_types,
        enum_constants: ctx.enum_constants,
        static_locals: ctx.static_locals,
        struct_defs: ctx.struct_defs,
        label_blocks: ctx.label_blocks,
        current_function_name: ctx.current_function_name,
        enclosing_loop_stack: ctx.loop_stack.clone(),
        scope_type_overrides: &ctx.scope_type_overrides,
        last_bitfield_info: None,
        layout_cache: &mut ctx.layout_cache,
        vla_sizes: &ctx.vla_sizes,
    };
    lower_expression_typed(&mut expr_ctx, expr)
}

/// Lower an expression as an **lvalue** via a `StmtLoweringContext`.
///
/// Returns the ADDRESS (alloca pointer) of the expression, not its value.
/// Used for inline assembly output operands which need a pointer to store
/// the result into.
fn lower_lvalue_via_context(ctx: &mut StmtLoweringContext<'_>, expr: &ast::Expression) -> Value {
    let mut expr_ctx = ExprLoweringContext {
        builder: ctx.builder,
        function: ctx.function,
        module: ctx.module,
        target: ctx.target,
        type_builder: ctx.type_builder,
        diagnostics: ctx.diagnostics,
        local_vars: ctx.local_vars,
        param_values: ctx.param_values,
        name_table: ctx.name_table,
        local_types: ctx.local_types,
        enum_constants: ctx.enum_constants,
        static_locals: ctx.static_locals,
        struct_defs: ctx.struct_defs,
        label_blocks: ctx.label_blocks,
        current_function_name: ctx.current_function_name,
        enclosing_loop_stack: ctx.loop_stack.clone(),
        scope_type_overrides: &ctx.scope_type_overrides,
        last_bitfield_info: None,
        layout_cache: &mut ctx.layout_cache,
        vla_sizes: &ctx.vla_sizes,
    };
    lower_lvalue(&mut expr_ctx, expr)
}

/// Converts an expression result to an I1 (boolean) value for use as a
/// branch condition.
///
/// Convert an expression to an I1 boolean value for use as a branch condition.
///
/// If the expression is already a comparison operator (==, !=, <, >, <=, >=,
/// ||, &&, !) it naturally produces an I1, so the lowered result is returned
/// directly.  Otherwise a `value != 0` comparison is emitted to produce the
/// boolean (C truthiness rule: any non-zero scalar is true).
fn lower_expr_to_i1(
    ctx: &mut StmtLoweringContext<'_>,
    expr: &ast::Expression,
    span: Span,
) -> Value {
    // Detect expressions that already produce an I1 boolean.
    let already_i1 = match expr {
        ast::Expression::Binary { op, .. } => matches!(
            op,
            ast::BinaryOp::Equal
                | ast::BinaryOp::NotEqual
                | ast::BinaryOp::Less
                | ast::BinaryOp::Greater
                | ast::BinaryOp::LessEqual
                | ast::BinaryOp::GreaterEqual
                | ast::BinaryOp::LogicalAnd
                | ast::BinaryOp::LogicalOr
        ),
        ast::Expression::UnaryOp { op, .. } => matches!(op, ast::UnaryOp::LogicalNot),
        _ => false,
    };

    // Lower the expression with type information so we can detect float conditions.
    let tv = lower_expr_typed_via_context(ctx, expr);
    let val = tv.value;
    let is_float = matches!(tv.ty, CType::Float | CType::Double | CType::LongDouble);

    if already_i1 {
        // Expression is a comparison/logical op — already produces 0 or 1.
        return val;
    }

    if is_float {
        // Float-to-bool: emit FCmp One (ordered not equal) against 0.0
        // Create a float zero constant using the FAdd sentinel pattern.
        let ir_ty = match tv.ty {
            CType::Float => IrType::F32,
            _ => IrType::F64,
        };
        let const_id = ctx.module.globals().len();
        let gname = format!(".Lconst.f.{}", const_id);
        let gname_clone = gname.clone();
        let mut gv = crate::ir::module::GlobalVariable::new(
            gname,
            ir_ty.clone(),
            Some(crate::ir::module::Constant::Float(0.0)),
        );
        gv.linkage = crate::ir::module::Linkage::Internal;
        gv.is_constant = true;
        ctx.module.add_global(gv);

        let fzero = ctx.builder.fresh_value();
        let fzero_inst = Instruction::BinOp {
            result: fzero,
            op: crate::ir::instructions::BinOp::FAdd,
            lhs: fzero,
            rhs: Value::UNDEF,
            ty: ir_ty,
            span,
        };
        emit_instruction_to_current_block(ctx, fzero_inst);
        ctx.function
            .float_constant_values
            .insert(fzero, (gname_clone, 0.0));

        let cmp_result = ctx.builder.fresh_value();
        let cmp_inst = Instruction::FCmp {
            result: cmp_result,
            op: crate::ir::instructions::FCmpOp::One,
            lhs: val,
            rhs: fzero,
            span,
        };
        emit_instruction_to_current_block(ctx, cmp_inst);
        return cmp_result;
    }

    // Complex types: truthy if real != 0 OR imag != 0.
    // We need a pointer to the aggregate to extract real/imag via GEP.
    // Try getting the lvalue first (works for identifiers); if the expression
    // is not a simple lvalue (e.g. assignment `c = z`, function call result),
    // store the aggregate value to a temporary alloca and use that instead.
    if let CType::Complex(ref base) = tv.ty {
        let elem_ir = crate::ir::lowering::expr_lowering::ctype_to_ir(base.as_ref(), ctx.target);
        let elem_sz: i64 = match &elem_ir {
            IrType::F32 => 4,
            IrType::F64 => 8,
            IrType::F80 => 16,
            _ => 8,
        };

        // Determine complex aggregate IR type and get a pointer to it.
        let complex_ir = crate::ir::lowering::expr_lowering::ctype_to_ir(&tv.ty, ctx.target);
        // Use the already-lowered value (`tv.value`): for complex types the
        // lowering result is the alloca pointer produced by lower_expr_inner.
        // However for some expression forms (function calls, assignment
        // expressions) the "value" is actually the loaded first-word of the
        // aggregate — we need an alloca pointer.  Check the value's type:
        // if we can determine the expression is a simple identifier we re-
        // lower as lvalue; otherwise we store the aggregate to a temporary.
        let complex_ptr = match expr {
            ast::Expression::Identifier { .. }
            | ast::Expression::MemberAccess { .. }
            | ast::Expression::PointerMemberAccess { .. }
            | ast::Expression::ArraySubscript { .. }
            | ast::Expression::UnaryOp { .. } => {
                // These are guaranteed lvalues.
                lower_lvalue_via_context(ctx, expr)
            }
            _ => {
                // For non-lvalue expressions (assignments, function calls,
                // casts, etc.), create a temp alloca and store the aggregate.
                let (alloca_val, alloca_inst) = ctx.builder.build_alloca(complex_ir.clone(), span);
                emit_instruction_to_current_block(ctx, alloca_inst);
                let store_inst = ctx.builder.build_store(tv.value, alloca_val, span);
                emit_instruction_to_current_block(ctx, store_inst);
                alloca_val
            }
        };

        // Load real part (offset 0) directly from alloca pointer.
        let (real_val, load_r) = ctx.builder.build_load(complex_ptr, elem_ir.clone(), span);
        emit_instruction_to_current_block(ctx, load_r);

        // GEP to get pointer to imaginary part (offset elem_sz), then load.
        let imag_idx = ctx.builder.fresh_value();
        let imag_idx_inst = Instruction::BinOp {
            result: imag_idx,
            op: crate::ir::instructions::BinOp::Add,
            lhs: imag_idx,
            rhs: Value::UNDEF,
            ty: IrType::I64,
            span,
        };
        emit_instruction_to_current_block(ctx, imag_idx_inst);
        ctx.function.constant_values.insert(imag_idx, elem_sz);

        let imag_ptr = ctx.builder.fresh_value();
        let gep_i = Instruction::GetElementPtr {
            result: imag_ptr,
            base: complex_ptr,
            indices: vec![imag_idx],
            result_type: IrType::Ptr,
            in_bounds: true,
            span,
        };
        emit_instruction_to_current_block(ctx, gep_i);

        let (imag_val, load_i) = ctx.builder.build_load(imag_ptr, elem_ir.clone(), span);
        emit_instruction_to_current_block(ctx, load_i);

        // Compare each part to zero.
        let (real_nz, imag_nz) = if elem_ir.is_float() {
            // Create float zero constants for real and imaginary comparisons.
            let fzero_gid = ctx.module.globals().len();
            let fzero_name = format!(".Lconst.f.{}", fzero_gid);
            let fzero_name_c = fzero_name.clone();
            let mut gv = crate::ir::module::GlobalVariable::new(
                fzero_name,
                elem_ir.clone(),
                Some(crate::ir::module::Constant::Float(0.0)),
            );
            gv.linkage = crate::ir::module::Linkage::Internal;
            gv.is_constant = true;
            ctx.module.add_global(gv);

            let fzero_r = ctx.builder.fresh_value();
            let fzero_r_inst = Instruction::BinOp {
                result: fzero_r,
                op: crate::ir::instructions::BinOp::FAdd,
                lhs: fzero_r,
                rhs: Value::UNDEF,
                ty: elem_ir.clone(),
                span,
            };
            emit_instruction_to_current_block(ctx, fzero_r_inst);
            ctx.function
                .float_constant_values
                .insert(fzero_r, (fzero_name_c.clone(), 0.0));

            let fzero_i = ctx.builder.fresh_value();
            let fzero_i_inst = Instruction::BinOp {
                result: fzero_i,
                op: crate::ir::instructions::BinOp::FAdd,
                lhs: fzero_i,
                rhs: Value::UNDEF,
                ty: elem_ir.clone(),
                span,
            };
            emit_instruction_to_current_block(ctx, fzero_i_inst);
            ctx.function
                .float_constant_values
                .insert(fzero_i, (fzero_name_c, 0.0));

            let rnz = ctx.builder.fresh_value();
            let cmp_r = Instruction::FCmp {
                result: rnz,
                op: crate::ir::instructions::FCmpOp::One,
                lhs: real_val,
                rhs: fzero_r,
                span,
            };
            emit_instruction_to_current_block(ctx, cmp_r);

            let inz = ctx.builder.fresh_value();
            let cmp_i = Instruction::FCmp {
                result: inz,
                op: crate::ir::instructions::FCmpOp::One,
                lhs: imag_val,
                rhs: fzero_i,
                span,
            };
            emit_instruction_to_current_block(ctx, cmp_i);
            (rnz, inz)
        } else {
            let izero_r = ctx.builder.fresh_value();
            let izero_r_inst = Instruction::BinOp {
                result: izero_r,
                op: crate::ir::instructions::BinOp::Add,
                lhs: izero_r,
                rhs: Value::UNDEF,
                ty: elem_ir.clone(),
                span,
            };
            emit_instruction_to_current_block(ctx, izero_r_inst);
            ctx.function.constant_values.insert(izero_r, 0);

            let izero_i = ctx.builder.fresh_value();
            let izero_i_inst = Instruction::BinOp {
                result: izero_i,
                op: crate::ir::instructions::BinOp::Add,
                lhs: izero_i,
                rhs: Value::UNDEF,
                ty: elem_ir.clone(),
                span,
            };
            emit_instruction_to_current_block(ctx, izero_i_inst);
            ctx.function.constant_values.insert(izero_i, 0);

            let rnz = ctx.builder.fresh_value();
            let cmp_r = Instruction::ICmp {
                result: rnz,
                op: crate::ir::instructions::ICmpOp::Ne,
                lhs: real_val,
                rhs: izero_r,
                span,
            };
            emit_instruction_to_current_block(ctx, cmp_r);

            let inz = ctx.builder.fresh_value();
            let cmp_i = Instruction::ICmp {
                result: inz,
                op: crate::ir::instructions::ICmpOp::Ne,
                lhs: imag_val,
                rhs: izero_i,
                span,
            };
            emit_instruction_to_current_block(ctx, cmp_i);
            (rnz, inz)
        };

        // OR the two boolean results: truthy if either part is non-zero.
        let result = ctx.builder.fresh_value();
        let or_inst = Instruction::BinOp {
            result,
            op: crate::ir::instructions::BinOp::Or,
            lhs: real_nz,
            rhs: imag_nz,
            ty: IrType::I1,
            span,
        };
        emit_instruction_to_current_block(ctx, or_inst);
        return result;
    }

    // Non-comparison, non-float expression: emit `val != 0` to convert to boolean.
    // We create a proper zero constant in the module's global pool so the
    // backend can resolve it correctly during constant cache population.
    let const_id = ctx.module.globals().len();
    let gname = format!(".Lconst.i.{}", const_id);
    let mut gv = crate::ir::module::GlobalVariable::new(
        gname,
        IrType::I32,
        Some(crate::ir::module::Constant::Integer(0)),
    );
    gv.linkage = crate::ir::module::Linkage::Internal;
    gv.is_constant = true;
    ctx.module.add_global(gv);

    let zero = ctx.builder.fresh_value();
    let zero_inst = Instruction::BinOp {
        result: zero,
        op: crate::ir::instructions::BinOp::Add,
        lhs: zero,
        rhs: Value::UNDEF,
        ty: IrType::I32,
        span,
    };
    emit_instruction_to_current_block(ctx, zero_inst);
    // Register the zero constant in the function's constant_values map so
    // the backend resolves it correctly (Bug 35: without this, the backend
    // falls through to select_binop which produces garbage values instead
    // of immediate 0).
    ctx.function.constant_values.insert(zero, 0);

    let cmp_result = ctx.builder.fresh_value();
    let cmp_inst = Instruction::ICmp {
        result: cmp_result,
        op: crate::ir::instructions::ICmpOp::Ne,
        lhs: val,
        rhs: zero,
        span,
    };
    emit_instruction_to_current_block(ctx, cmp_inst);
    cmp_result
}
