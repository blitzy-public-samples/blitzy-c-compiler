//! Statement-to-IR lowering module.
//!
//! Translates C statement AST nodes into IR basic-block control flow graphs.
//! This is the critical module for building the control flow graph of the IR —
//! every C control flow construct (`if`/`else`, `while`, `do-while`, `for`,
//! `switch`, `goto`, computed `goto`, `break`, `continue`, `return`) is
//! lowered here into basic blocks connected by branch, conditional-branch,
//! switch, and return terminator instructions.
//!
//! # Architecture
//!
//! The public entry point [`lower_statement`] dispatches on the
//! [`Statement`] enum variant, delegating to private handler functions for
//! each construct. Loop/switch context is maintained via stacks in the
//! [`LoweringContext`] for resolving `break` and `continue`.
//!
//! ## Loop Lowering Patterns
//!
//! - **while:** `header_bb` (condition) → `body_bb` → back-edge to `header_bb`;
//!   condition false → `exit_bb`.
//! - **do-while:** `body_bb` → `latch_bb` (condition) → back-edge to `body_bb`;
//!   condition false → `exit_bb`.
//! - **for:** init in current block → `header_bb` (condition) → `body_bb` →
//!   `latch_bb` (increment) → back-edge to `header_bb`; condition false → `exit_bb`.
//!
//! ## Switch Lowering
//!
//! A two-pass approach is used:
//! 1. **Collect pass:** recursively scan the switch body to find all `case`,
//!    `case range`, and `default` labels, evaluate their constant values, and
//!    create pre-allocated basic blocks.
//! 2. **Emit pass:** emit the `Switch` IR instruction in the dispatch block,
//!    then lower the switch body, transitioning to the appropriate case block
//!    on each case label and handling fall-through semantics.
//!
//! ## Computed Goto
//!
//! Since the IR does not have an `IndirectBranch` instruction, computed goto
//! (`goto *expr`) is lowered as a `Switch` over all address-taken labels,
//! mapping integer label addresses to their target blocks.

// ============================================================================
// Imports
// ============================================================================

use crate::common::diagnostics::Span;
use crate::common::fx_hash::FxHashMap;
use crate::common::string_interner::Symbol;
use crate::common::types::CType;
use crate::frontend::parser::ast::{
    BlockItem, Declaration, Expression, ForInit, Initializer, Statement,
};
use crate::frontend::sema::constant_eval::evaluate_integer_constant;
use crate::ir::basic_block::BasicBlockId;
use crate::ir::instructions::ValueId;
use crate::ir::types::IrType;

use super::asm_lowering::lower_asm_statement;
use super::expr_lowering::lower_expression;
use super::{check_recursion_depth, ensure_not_terminated, LoweringContext, LoweringError};

// ============================================================================
// Public API
// ============================================================================

/// Lowers a single C [`Statement`] to IR instructions and basic blocks.
///
/// This is the main dispatcher called for any statement node. It pattern-
/// matches on the [`Statement`] variant and delegates to the appropriate
/// private handler function. Each handler is responsible for creating any
/// needed basic blocks, emitting instructions via `ctx.builder`, and
/// updating loop/switch/label context stacks as appropriate.
///
/// After a terminator instruction (branch, return, switch) has been emitted
/// in the current block, subsequent statements are lowered into a new
/// unreachable block to maintain well-formed IR. The dead-code elimination
/// pass will later remove any unreachable blocks.
///
/// # Errors
///
/// Returns [`LoweringError`] on:
/// - `break` outside of loop or switch context
/// - `continue` outside of loop context
/// - Undefined goto labels (detected at function-level post-verification)
/// - Recursion depth exceeded
/// - Expression lowering failures (propagated from [`lower_expression`])
pub fn lower_statement(
    ctx: &mut LoweringContext<'_>,
    stmt: &Statement,
) -> Result<(), LoweringError> {
    check_recursion_depth(ctx, stmt.span())?;
    ctx.recursion_depth += 1;
    let result = lower_statement_inner(ctx, stmt);
    ctx.recursion_depth -= 1;
    result
}

// ============================================================================
// Internal Dispatcher
// ============================================================================

/// Core statement dispatch — routes each [`Statement`] variant to its handler.
fn lower_statement_inner(
    ctx: &mut LoweringContext<'_>,
    stmt: &Statement,
) -> Result<(), LoweringError> {
    // If the current block is already terminated (e.g., after a return),
    // create a new unreachable block for any subsequent code. The
    // dead-code elimination pass will clean up unreachable blocks later.
    if !ensure_not_terminated(ctx) {
        let dead_bb = ctx.builder.create_block(ctx.function, Some("unreachable"));
        ctx.builder.set_insert_point(dead_bb);
    }

    // DEBUG: trace statement lowering

    match stmt {
        Statement::Compound { items, span } => lower_compound_stmt(ctx, items, *span),

        Statement::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => lower_if_stmt(ctx, condition, then_branch, else_branch.as_deref(), *span),

        Statement::While {
            condition,
            body,
            span,
        } => lower_while_stmt(ctx, condition, body, *span),

        Statement::DoWhile {
            body,
            condition,
            span,
        } => lower_do_while_stmt(ctx, body, condition, *span),

        Statement::For {
            init,
            condition,
            increment,
            body,
            span,
        } => lower_for_stmt(
            ctx,
            init.as_ref(),
            condition.as_deref(),
            increment.as_deref(),
            body,
            *span,
        ),

        Statement::Switch {
            expression,
            body,
            span,
        } => lower_switch_stmt(ctx, expression, body, *span),

        // Case/CaseRange/Default outside of switch context: these are
        // handled specially during switch body lowering. If encountered
        // at the top level, they are erroneous but we lower the body anyway.
        Statement::Case { body, .. }
        | Statement::CaseRange { body, .. }
        | Statement::Default { body, .. } => lower_statement(ctx, body),

        Statement::Goto { label, span } => lower_goto_stmt(ctx, *label, *span),

        Statement::ComputedGoto { target, span } => lower_computed_goto(ctx, target, *span),

        Statement::Break { span } => lower_break(ctx, *span),

        Statement::Continue { span } => lower_continue(ctx, *span),

        Statement::Return { value, span } => lower_return_stmt(ctx, value.as_deref(), *span),

        Statement::Labeled {
            label, body, span, ..
        } => lower_labeled_stmt(ctx, *label, body, *span),

        Statement::Expression { expr, .. } => lower_expr_stmt(ctx, expr),

        // Null statement — no-op.
        Statement::Null { .. } => Ok(()),

        // Inline assembly — delegate to asm_lowering module.
        Statement::Asm(asm_stmt) => lower_asm_statement(ctx, asm_stmt),

        // Error recovery node — skip silently.
        Statement::Error { .. } => Ok(()),
    }
}

// ============================================================================
// Compound Statement (Block)
// ============================================================================

/// Lowers a compound statement `{ item1; item2; ... }`.
///
/// Iterates through block items in order. Declarations within the block
/// are handled inline by creating allocas and optionally storing initializer
/// values. Statements are recursively dispatched through [`lower_statement`].
fn lower_compound_stmt(
    ctx: &mut LoweringContext<'_>,
    items: &[BlockItem],
    _span: Span,
) -> Result<(), LoweringError> {
    // eprintln!("[COMPOUND_DBG] Lowering compound statement with {} items", items.len());
    for (_idx, item) in items.iter().enumerate() {
        // eprintln!("[COMPOUND_DBG]   Item {}: {:?}", idx, match item {
        // BlockItem::Statement(s) => format!("Statement({:?})", std::mem::discriminant(s)),
        // BlockItem::Declaration(_) => "Declaration".to_string(),
        // });
        match item {
            BlockItem::Statement(stmt) => {
                lower_statement(ctx, stmt)?;
            }
            BlockItem::Declaration(decl) => {
                lower_block_declaration(ctx, decl)?;
            }
        }
        // eprintln!("[COMPOUND_DBG]   After item {}: insert_block={:?}, terminated={}",
        // idx,
        // ctx.builder.get_insert_block(),
        // ctx.current_block_terminated());
    }
    Ok(())
}

// ============================================================================
// If/Else → Conditional Branch
// ============================================================================

/// Lowers `if (condition) then_branch [else else_branch]`.
///
/// Creates a conditional branch structure:
/// ```text
///   current_block:
///     %cond = <lower condition>
///     cond_branch %cond, then_bb, else_bb/merge_bb
///
///   then_bb:
///     <lower then_branch>
///     branch merge_bb
///
///   else_bb (if present):
///     <lower else_branch>
///     branch merge_bb
///
///   merge_bb:
///     <subsequent code>
/// ```
fn lower_if_stmt(
    ctx: &mut LoweringContext<'_>,
    condition: &Expression,
    then_branch: &Statement,
    else_branch: Option<&Statement>,
    _span: Span,
) -> Result<(), LoweringError> {
    // Lower the condition to an I1 boolean value.
    let cond_val = lower_expression(ctx, condition)?;

    // Create blocks.
    let then_bb = ctx.builder.create_block(ctx.function, Some("if.then"));
    let merge_bb = ctx.builder.create_block(ctx.function, Some("if.merge"));
    let else_bb = if else_branch.is_some() {
        ctx.builder.create_block(ctx.function, Some("if.else"))
    } else {
        merge_bb
    };

    // Emit conditional branch: true → then_bb, false → else_bb/merge_bb.
    ctx.builder
        .build_cond_branch(ctx.function, cond_val, then_bb, else_bb);

    // Lower "then" branch.
    ctx.builder.set_insert_point(then_bb);
    lower_statement(ctx, then_branch)?;
    // Fall through to merge block if the then block is not terminated.
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, merge_bb);
    }

    // Lower "else" branch, if present.
    if let Some(else_stmt) = else_branch {
        ctx.builder.set_insert_point(else_bb);
        lower_statement(ctx, else_stmt)?;
        // Fall through to merge block.
        if ensure_not_terminated(ctx) {
            ctx.builder.build_branch(ctx.function, merge_bb);
        }
    }

    // Continue emitting code at the merge block.
    ctx.builder.set_insert_point(merge_bb);
    Ok(())
}

// ============================================================================
// While Loop → Header/Body/Exit
// ============================================================================

/// Lowers `while (condition) body`.
///
/// Creates the standard loop pattern:
/// ```text
///   current_block:
///     branch header_bb
///
///   header_bb:
///     %cond = <lower condition>
///     cond_branch %cond, body_bb, exit_bb
///
///   body_bb:
///     <lower body>
///     branch header_bb  (back-edge)
///
///   exit_bb:
///     <subsequent code>
/// ```
fn lower_while_stmt(
    ctx: &mut LoweringContext<'_>,
    condition: &Expression,
    body: &Statement,
    _span: Span,
) -> Result<(), LoweringError> {
    let header_bb = ctx.builder.create_block(ctx.function, Some("while.header"));
    let body_bb = ctx.builder.create_block(ctx.function, Some("while.body"));
    let exit_bb = ctx.builder.create_block(ctx.function, Some("while.exit"));

    // Branch from current block to header.
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, header_bb);
    }

    // Lower condition in header block.
    ctx.builder.set_insert_point(header_bb);
    let cond_val = lower_expression(ctx, condition)?;
    ctx.builder
        .build_cond_branch(ctx.function, cond_val, body_bb, exit_bb);

    // Push loop context for break/continue resolution.
    // break → exit_bb, continue → header_bb.
    ctx.push_loop(exit_bb, header_bb);

    // Lower loop body.
    ctx.builder.set_insert_point(body_bb);
    lower_statement(ctx, body)?;
    // Back-edge: branch to header for next iteration.
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, header_bb);
    }

    ctx.pop_loop();
    ctx.builder.set_insert_point(exit_bb);
    Ok(())
}

// ============================================================================
// Do-While Loop → Body/Latch/Exit
// ============================================================================

/// Lowers `do body while (condition);`.
///
/// ```text
///   current_block:
///     branch body_bb
///
///   body_bb:
///     <lower body>
///     branch latch_bb
///
///   latch_bb:
///     %cond = <lower condition>
///     cond_branch %cond, body_bb, exit_bb  (back-edge on true)
///
///   exit_bb:
///     <subsequent code>
/// ```
fn lower_do_while_stmt(
    ctx: &mut LoweringContext<'_>,
    body: &Statement,
    condition: &Expression,
    _span: Span,
) -> Result<(), LoweringError> {
    let body_bb = ctx.builder.create_block(ctx.function, Some("dowhile.body"));
    let latch_bb = ctx
        .builder
        .create_block(ctx.function, Some("dowhile.latch"));
    let exit_bb = ctx.builder.create_block(ctx.function, Some("dowhile.exit"));

    // Branch from current block into the body (first iteration always executes).
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, body_bb);
    }

    // Push loop context: break → exit_bb, continue → latch_bb.
    ctx.push_loop(exit_bb, latch_bb);

    // Lower body.
    ctx.builder.set_insert_point(body_bb);
    lower_statement(ctx, body)?;
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, latch_bb);
    }

    // Lower condition in latch block.
    ctx.builder.set_insert_point(latch_bb);
    let cond_val = lower_expression(ctx, condition)?;
    ctx.builder
        .build_cond_branch(ctx.function, cond_val, body_bb, exit_bb);

    ctx.pop_loop();
    ctx.builder.set_insert_point(exit_bb);
    Ok(())
}

// ============================================================================
// For Loop → Init/Header/Body/Latch/Exit
// ============================================================================

/// Lowers `for (init; condition; increment) body`.
///
/// ```text
///   current_block:
///     <lower init>
///     branch header_bb
///
///   header_bb:
///     %cond = <lower condition>  (or unconditional if None)
///     cond_branch %cond, body_bb, exit_bb
///
///   body_bb:
///     <lower body>
///     branch latch_bb
///
///   latch_bb:
///     <lower increment>
///     branch header_bb  (back-edge)
///
///   exit_bb:
///     <subsequent code>
/// ```
fn lower_for_stmt(
    ctx: &mut LoweringContext<'_>,
    init: Option<&ForInit>,
    condition: Option<&Expression>,
    increment: Option<&Expression>,
    body: &Statement,
    _span: Span,
) -> Result<(), LoweringError> {
    // Lower the initializer in the current block.
    if let Some(for_init) = init {
        match for_init {
            ForInit::Expression(expr) => {
                // Lower expression for side effects, discard result.
                let _ = lower_expression(ctx, expr)?;
            }
            ForInit::Declaration(decl) => {
                lower_block_declaration(ctx, decl)?;
            }
        }
    }

    let header_bb = ctx.builder.create_block(ctx.function, Some("for.header"));
    let body_bb = ctx.builder.create_block(ctx.function, Some("for.body"));
    let latch_bb = ctx.builder.create_block(ctx.function, Some("for.latch"));
    let exit_bb = ctx.builder.create_block(ctx.function, Some("for.exit"));

    // Branch from current block to header.
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, header_bb);
    }

    // Header: evaluate condition (or unconditional entry if absent).
    ctx.builder.set_insert_point(header_bb);
    if let Some(cond_expr) = condition {
        let cond_val = lower_expression(ctx, cond_expr)?;
        ctx.builder
            .build_cond_branch(ctx.function, cond_val, body_bb, exit_bb);
    } else {
        // No condition → infinite loop (equivalent to `for(;;)`).
        ctx.builder.build_branch(ctx.function, body_bb);
    }

    // Push loop context: break → exit_bb, continue → latch_bb.
    ctx.push_loop(exit_bb, latch_bb);

    // Lower loop body.
    ctx.builder.set_insert_point(body_bb);
    lower_statement(ctx, body)?;
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, latch_bb);
    }

    // Latch: evaluate increment, then branch back to header.
    ctx.builder.set_insert_point(latch_bb);
    if let Some(inc_expr) = increment {
        let _ = lower_expression(ctx, inc_expr)?;
    }
    ctx.builder.build_branch(ctx.function, header_bb);

    ctx.pop_loop();
    ctx.builder.set_insert_point(exit_bb);
    Ok(())
}

// ============================================================================
// Switch Statement
// ============================================================================

/// Information about case labels collected during the first pass over a
/// switch body.
struct SwitchCaseInfo {
    /// (case_value, target_block) pairs for the Switch IR instruction.
    cases: Vec<(i64, BasicBlockId)>,
    /// The default target block. If no `default:` label exists, this
    /// points to the exit block.
    default_block: BasicBlockId,
    /// Maps each case constant value to its block, for lookup during
    /// the body-lowering pass.
    value_to_block: FxHashMap<i64, BasicBlockId>,
    /// The block allocated for the `default:` label (if any).
    /// `None` if no default label was found.
    has_default: bool,
}

/// Lowers `switch (expression) body`.
///
/// Two-pass algorithm:
/// 1. Scan the body for all `case`, `case range`, and `default` labels.
///    Evaluate constant values and pre-allocate a block for each.
/// 2. Emit the `Switch` IR instruction, then lower the body with case
///    label transitions.
fn lower_switch_stmt(
    ctx: &mut LoweringContext<'_>,
    expression: &Expression,
    body: &Statement,
    _span: Span,
) -> Result<(), LoweringError> {
    // Lower the switch controlling expression.
    let switch_val = lower_expression(ctx, expression)?;

    // Create the exit block for break targets.
    let exit_bb = ctx.builder.create_block(ctx.function, Some("switch.exit"));

    // --- Pass 1: Collect all case labels and allocate blocks ---
    let mut case_info = SwitchCaseInfo {
        cases: Vec::new(),
        default_block: exit_bb, // default falls through to exit if none specified
        value_to_block: FxHashMap::default(),
        has_default: false,
    };
    collect_switch_cases(ctx, body, &mut case_info)?;

    // Emit the Switch IR instruction in the current block.
    ctx.builder.build_switch(
        ctx.function,
        switch_val,
        case_info.default_block,
        case_info.cases.clone(),
    );

    // Push switch context for break resolution.
    ctx.push_switch(exit_bb);

    // --- Pass 2: Lower the switch body with case transitions ---
    // Create an initial body block for any code before the first case label.
    let body_start_bb = ctx
        .builder
        .create_block(ctx.function, Some("switch.body.start"));
    ctx.builder.set_insert_point(body_start_bb);

    lower_switch_body(ctx, body, &case_info)?;

    ctx.pop_switch();

    // If current block is not terminated, fall through to exit.
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, exit_bb);
    }

    ctx.builder.set_insert_point(exit_bb);
    Ok(())
}

/// Recursively scans a switch body to collect all `case`, `case range`,
/// and `default` labels, evaluating their constant values and creating
/// basic blocks for each.
fn collect_switch_cases(
    ctx: &mut LoweringContext<'_>,
    stmt: &Statement,
    info: &mut SwitchCaseInfo,
) -> Result<(), LoweringError> {
    match stmt {
        Statement::Case { value, body, .. } => {
            let case_val = try_eval_case_value(ctx, value)?;
            let block = ctx
                .builder
                .create_block(ctx.function, Some(&format!("case.{}", case_val)));
            info.cases.push((case_val, block));
            info.value_to_block.insert(case_val, block);
            // Recursively scan the case body for nested case labels.
            collect_switch_cases(ctx, body, info)?;
        }

        Statement::CaseRange {
            low, high, body, ..
        } => {
            let low_val = try_eval_case_value(ctx, low)?;
            let high_val = try_eval_case_value(ctx, high)?;
            // All values in the range map to the same block.
            let block = ctx.builder.create_block(
                ctx.function,
                Some(&format!("case.range.{}_{}", low_val, high_val)),
            );
            // Clamp range to prevent pathological expansions.
            let range_size = (high_val - low_val + 1).min(10_000);
            for i in 0..range_size {
                let val = low_val + i;
                info.cases.push((val, block));
                info.value_to_block.insert(val, block);
            }
            collect_switch_cases(ctx, body, info)?;
        }

        Statement::Default { body, .. } => {
            let block = ctx.builder.create_block(ctx.function, Some("default"));
            info.default_block = block;
            info.has_default = true;
            collect_switch_cases(ctx, body, info)?;
        }

        Statement::Compound { items, .. } => {
            for item in items {
                if let BlockItem::Statement(s) = item {
                    collect_switch_cases(ctx, s, info)?;
                }
            }
        }

        // Scan into nested control-flow statements — case labels can
        // technically appear inside if/else, loops, etc. (legal in C).
        Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_switch_cases(ctx, then_branch, info)?;
            if let Some(else_br) = else_branch {
                collect_switch_cases(ctx, else_br, info)?;
            }
        }

        Statement::While { body, .. }
        | Statement::DoWhile { body, .. }
        | Statement::For { body, .. } => {
            collect_switch_cases(ctx, body, info)?;
        }

        Statement::Labeled { body, .. } => {
            collect_switch_cases(ctx, body, info)?;
        }

        // Other statement types cannot contain case labels.
        _ => {}
    }
    Ok(())
}

/// Lowers the switch body, transitioning to pre-allocated case blocks
/// when case labels are encountered and handling fall-through semantics.
fn lower_switch_body(
    ctx: &mut LoweringContext<'_>,
    stmt: &Statement,
    case_info: &SwitchCaseInfo,
) -> Result<(), LoweringError> {
    match stmt {
        Statement::Case { value, body, .. } => {
            let case_val = try_eval_case_value(ctx, value)?;
            if let Some(&target_block) = case_info.value_to_block.get(&case_val) {
                // If the current block is not terminated, emit a fall-through
                // branch to the case block.
                if ensure_not_terminated(ctx) {
                    ctx.builder.build_branch(ctx.function, target_block);
                }
                ctx.builder.set_insert_point(target_block);
            }
            // Lower the case body.
            lower_switch_body(ctx, body, case_info)?;
            Ok(())
        }

        Statement::CaseRange { low, body, .. } => {
            let low_val = try_eval_case_value(ctx, low)?;
            if let Some(&target_block) = case_info.value_to_block.get(&low_val) {
                if ensure_not_terminated(ctx) {
                    ctx.builder.build_branch(ctx.function, target_block);
                }
                ctx.builder.set_insert_point(target_block);
            }
            lower_switch_body(ctx, body, case_info)?;
            Ok(())
        }

        Statement::Default { body, .. } => {
            if case_info.has_default {
                if ensure_not_terminated(ctx) {
                    ctx.builder
                        .build_branch(ctx.function, case_info.default_block);
                }
                ctx.builder.set_insert_point(case_info.default_block);
            }
            lower_switch_body(ctx, body, case_info)?;
            Ok(())
        }

        Statement::Compound { items, .. } => {
            for item in items {
                match item {
                    BlockItem::Statement(s) => {
                        lower_switch_body(ctx, s, case_info)?;
                    }
                    BlockItem::Declaration(decl) => {
                        lower_block_declaration(ctx, decl)?;
                    }
                }
            }
            Ok(())
        }

        // For statements that are not case labels, fall back to the
        // normal statement lowering path.
        _ => lower_statement(ctx, stmt),
    }
}

/// Evaluates a case label expression to a compile-time integer constant.
///
/// Uses the semantic analysis constant evaluator when available, with a
/// fallback to direct literal extraction for simple cases.
fn try_eval_case_value(
    ctx: &mut LoweringContext<'_>,
    expr: &Expression,
) -> Result<i64, LoweringError> {
    // Try the full constant evaluator first. We reborrow the module
    // context to allow simultaneous mutable access to diagnostics and
    // immutable access to target.
    {
        let mc = &mut *ctx.module_ctx;
        if let Ok(val) = evaluate_integer_constant(expr, &mut mc.diagnostics, &mc.target) {
            return Ok(val as i64);
        }
    }

    // Fallback: handle simple literal forms directly.
    match expr {
        Expression::IntegerLiteral { value, .. } => Ok(*value as i64),
        Expression::CharLiteral { value, .. } => Ok(*value as i64),
        Expression::UnaryOp { op, operand, .. } => {
            use crate::frontend::parser::ast::UnaryOperator;
            match op {
                UnaryOperator::Neg => {
                    let inner = try_eval_case_value(ctx, operand)?;
                    Ok(-inner)
                }
                UnaryOperator::BitNot => {
                    let inner = try_eval_case_value(ctx, operand)?;
                    Ok(!inner)
                }
                _ => Err(LoweringError::UnsupportedExpression {
                    message: "non-constant case expression".to_string(),
                    span: expr.span(),
                }),
            }
        }
        _ => Err(LoweringError::UnsupportedExpression {
            message: "unable to evaluate case value as compile-time constant".to_string(),
            span: expr.span(),
        }),
    }
}

// ============================================================================
// Goto and Label Lowering
// ============================================================================

/// Lowers `goto label;` — unconditional jump to a named label.
///
/// If the target label's block has already been created (forward or
/// backward reference), the existing block is used. Otherwise, a new
/// block is created as a forward reference and will be bound when the
/// labeled statement is encountered.
fn lower_goto_stmt(
    ctx: &mut LoweringContext<'_>,
    label: Symbol,
    _span: Span,
) -> Result<(), LoweringError> {
    let target_block = ctx.get_or_create_label_block(label);

    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, target_block);
    }

    // After an unconditional goto, subsequent code is unreachable.
    // Create a new block for any following statements.
    let after_bb = ctx.builder.create_block(ctx.function, Some("after.goto"));
    ctx.builder.set_insert_point(after_bb);
    Ok(())
}

/// Lowers `label: statement` — a named label target for `goto`.
///
/// If the label's block was already created by a preceding `goto`
/// (forward reference), the insert point is set to that block.
/// Otherwise, a new block is created and registered in the label map.
fn lower_labeled_stmt(
    ctx: &mut LoweringContext<'_>,
    label: Symbol,
    body: &Statement,
    _span: Span,
) -> Result<(), LoweringError> {
    let label_block = ctx.get_or_create_label_block(label);

    // If the current block is not terminated, emit a fall-through
    // branch to the label's block.
    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, label_block);
    }

    ctx.builder.set_insert_point(label_block);

    // Lower the body statement at the label's block.
    lower_statement(ctx, body)
}

// ============================================================================
// Computed Goto → Switch-Based Dispatch
// ============================================================================

/// Lowers `goto *expression;` — GCC computed goto extension.
///
/// Since the IR lacks a native `IndirectBranch` instruction, computed goto
/// is implemented as a `Switch` instruction that dispatches over all labels
/// whose addresses have been taken (via `&&label` expressions). Each label
/// is assigned a small integer address, and the `Switch` compares the
/// target pointer (cast to integer) against these values.
///
/// If no address-taken labels are known, an unconditional branch to the
/// current block is emitted as a fallback (this case is degenerate and
/// would be a programming error in the original C).
fn lower_computed_goto(
    ctx: &mut LoweringContext<'_>,
    target: &Expression,
    _span: Span,
) -> Result<(), LoweringError> {
    // Lower the target expression to a pointer value (a block address).
    let target_val = lower_expression(ctx, target)?;

    // Collect all address-taken labels and their blocks.
    let taken_labels: Vec<Symbol> = ctx.address_taken_labels.iter().copied().collect();

    if taken_labels.is_empty() {
        // No address-taken labels — degenerate case.  Emit an
        // unreachable-style branch to a dead block.
        let dead_bb = ctx
            .builder
            .create_block(ctx.function, Some("computed.goto.dead"));
        if ensure_not_terminated(ctx) {
            ctx.builder.build_branch(ctx.function, dead_bb);
        }
        ctx.builder.set_insert_point(dead_bb);
        return Ok(());
    }

    // Gather the basic blocks of all address-taken labels.  These are
    // the possible targets of the indirect branch.
    let mut possible_targets: Vec<crate::ir::basic_block::BasicBlockId> = Vec::new();
    for label in &taken_labels {
        let block = ctx.get_or_create_label_block(*label);
        possible_targets.push(block);
    }

    // Emit IndirectBranch — a true indirect jump through the pointer.
    if ensure_not_terminated(ctx) {
        ctx.builder
            .build_indirect_branch(ctx.function, target_val, possible_targets);
    }

    // After computed goto, subsequent code is unreachable.
    let after_bb = ctx
        .builder
        .create_block(ctx.function, Some("after.computed.goto"));
    ctx.builder.set_insert_point(after_bb);
    Ok(())
}

// ============================================================================
// Break and Continue
// ============================================================================

/// Lowers `break;` — exits the innermost loop or switch.
///
/// Emits an unconditional branch to the break target at the top of the
/// combined loop/switch break stack.
fn lower_break(ctx: &mut LoweringContext<'_>, span: Span) -> Result<(), LoweringError> {
    let break_target = ctx
        .current_break_target()
        .ok_or(LoweringError::BreakOutsideLoop { span })?;

    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, break_target);
    }

    // Create unreachable block for subsequent dead code.
    let after_bb = ctx.builder.create_block(ctx.function, Some("after.break"));
    ctx.builder.set_insert_point(after_bb);
    Ok(())
}

/// Lowers `continue;` — jumps to the next iteration of the innermost loop.
///
/// Emits an unconditional branch to the continue target (header or latch)
/// of the innermost enclosing loop.
fn lower_continue(ctx: &mut LoweringContext<'_>, span: Span) -> Result<(), LoweringError> {
    let continue_target = ctx
        .current_continue_target()
        .ok_or(LoweringError::ContinueOutsideLoop { span })?;

    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, continue_target);
    }

    // Create unreachable block for subsequent dead code.
    let after_bb = ctx
        .builder
        .create_block(ctx.function, Some("after.continue"));
    ctx.builder.set_insert_point(after_bb);
    Ok(())
}

// ============================================================================
// Return Statement
// ============================================================================

/// Lowers `return [expression];`.
///
/// For non-void returns, the expression is lowered to a value and passed
/// to the `Return` IR instruction. For void returns, `Return(None)` is
/// emitted. A new unreachable block is created for any subsequent code.
fn lower_return_stmt(
    ctx: &mut LoweringContext<'_>,
    value: Option<&Expression>,
    _span: Span,
) -> Result<(), LoweringError> {
    let ret_val = if let Some(expr) = value {
        Some(lower_expression(ctx, expr)?)
    } else {
        None
    };

    if ensure_not_terminated(ctx) {
        ctx.builder.build_return(ctx.function, ret_val);
    }

    // After return, subsequent code is unreachable.
    let after_bb = ctx.builder.create_block(ctx.function, Some("after.return"));
    ctx.builder.set_insert_point(after_bb);
    Ok(())
}

// ============================================================================
// Expression Statement
// ============================================================================

/// Lowers an expression statement `expr;`.
///
/// The expression is lowered for its side effects; the result value
/// is discarded.
fn lower_expr_stmt(ctx: &mut LoweringContext<'_>, expr: &Expression) -> Result<(), LoweringError> {
    // Lower the expression; discard the result.
    let _ = lower_expression(ctx, expr)?;
    Ok(())
}

// ============================================================================
// Block-Scope Declaration Lowering
// ============================================================================

/// Lowers a block-scope declaration (variable declaration within a
/// compound statement).
///
/// This handles `Declaration::Variable` by creating allocas for each
/// declarator and optionally lowering initializer expressions. Other
/// declaration kinds (typedefs, struct/union/enum definitions, static
/// asserts) are no-ops at the IR level since they only affect the type
/// system and symbol table.
///
/// # Note on Type Resolution
///
/// Since we are working with raw parser AST declarations (not the
/// semantically checked `CheckedDeclaration` type), type resolution is
/// done with a best-effort approach. The basic strategy is to use
/// `IrType::I32` as a default for simple `int` declarations, `IrType::Ptr`
/// for pointers, and `IrType::I8` arrays for character arrays. Full type
/// resolution depends on the semantic analysis layer; this inline handler
/// covers the common cases needed for control flow lowering.
pub(crate) fn lower_block_declaration(
    ctx: &mut LoweringContext<'_>,
    decl: &Declaration,
) -> Result<(), LoweringError> {
    match decl {
        Declaration::Variable {
            specifiers,
            declarators,
            ..
        } => {
            // Register struct/union field names from the type specifiers
            // before resolving the IR type, so that the field name→index
            // mapping is available for member access expressions.
            // For anonymous structs, this returns a synthetic tag symbol.
            let anon_tag = register_struct_fields_from_specifiers(ctx, specifiers);

            // Track whether this declaration uses unsigned integer types.
            let is_unsigned = is_unsigned_from_specifiers(ctx, specifiers);

            // Resolve the base IR type from declaration specifiers.
            let base_ir_type = resolve_base_ir_type_from_specifiers(ctx, specifiers);

            // Detect the struct/union tag name for variable→struct mapping.
            // Prefer the explicit tag from the source, fall back to the
            // synthetic tag generated for anonymous structs.
            let struct_tag = extract_struct_tag_from_specifiers(specifiers).or(anon_tag);

            for init_decl in declarators {
                // Extract the variable name from the declarator.
                let var_name = match init_decl.declarator.name {
                    Some(name) => name,
                    None => continue, // Anonymous declarator — skip.
                };

                // Adjust the type based on derived declarators (pointers, arrays).
                let mut ir_type =
                    apply_derived_declarators(&base_ir_type, &init_decl.declarator.derived);

                // For unsized arrays, infer the element count from the
                // initializer when possible.
                if let IrType::Array { count: 0, .. } = ir_type {
                    if let Some(ref initializer) = init_decl.initializer {
                        // Special case: char arrays with string literal
                        // initializers use the string length.
                        let elem_cloned = if let IrType::Array { ref element, .. } = ir_type {
                            Some(element.clone())
                        } else {
                            None
                        };

                        if let Some(ref elem) = elem_cloned {
                            if matches!(**elem, IrType::I8) {
                                if let Some(str_len) = get_string_literal_array_size(initializer) {
                                    ir_type = IrType::Array {
                                        element: elem.clone(),
                                        count: str_len,
                                    };
                                }
                            }
                        }

                        // General case: for list initializers like
                        //   `int arr[] = {1, 2, 3}`  or
                        //   `void *table[] = {&&l1, &&l2, &&l3}`
                        // count the top-level initializer items to infer size.
                        if let IrType::Array { count: 0, .. } = ir_type {
                            if let Some(ref elem) = elem_cloned {
                                if let Initializer::List { items, .. } = initializer {
                                    if !items.is_empty() {
                                        ir_type = IrType::Array {
                                            element: elem.clone(),
                                            count: items.len(),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }

                // Create an alloca for this variable.
                let alloca = ctx.create_local_alloca(var_name, ir_type.clone());

                // Track unsigned variables for correct comparison predicates.
                if is_unsigned {
                    ctx.variable_unsigned.insert(var_name);
                }

                // If this variable is a pointer, record what type is
                // produced when it is dereferenced.  This enables correct
                // code generation for loads through multi-level pointers
                // (e.g. `int **pp`  →  `*pp` produces `int *` = Ptr).
                if ir_type == IrType::Ptr {
                    let mut pointee =
                        compute_pointee_type(&base_ir_type, &init_decl.declarator.derived);

                    // When the base type is already Ptr (e.g. from a typeof
                    // or typedef that evaluates to a pointer type) AND there
                    // are no pointer-derivation tokens in the declarator,
                    // `compute_pointee_type` returns Ptr (the base type
                    // itself) which loses the actual pointee information.
                    // Recover it from the typeof/typedef expression.
                    if pointee == IrType::Ptr {
                        let has_ptr_deriv = init_decl.declarator.derived.iter().any(|d| {
                            matches!(
                                d,
                                crate::frontend::parser::ast::DerivedDeclarator::Pointer { .. }
                            )
                        });
                        if !has_ptr_deriv {
                            if let Some(better) = resolve_pointee_from_specifiers(ctx, specifiers) {
                                pointee = better;
                            }
                        }
                    }

                    ctx.variable_pointee_types.insert(var_name, pointee.clone());

                    // For multi-level pointers (e.g. `int **pp`), also
                    // record the "deep" pointee type — the type obtained
                    // after fully dereferencing all pointer layers.
                    //
                    // `int **pp`  →  pointee = Ptr, deep_pointee = base = I32
                    // `int ***ppp` → pointee = Ptr, deep_pointee = base = I32
                    //
                    // This is used by `infer_dereference_type` to correctly
                    // type the innermost dereference (e.g. `**pp` → I32).
                    if pointee == IrType::Ptr {
                        // First, try to inherit the deep_pointee from the
                        // source variable when declared via typeof.
                        //   typeof(pp) pp2  →  inherit deep_pointee["pp"]
                        let inherited = resolve_deep_pointee_from_typeof(ctx, specifiers);
                        if let Some(deep) = inherited {
                            ctx.variable_deep_pointee_types.insert(var_name, deep);
                        } else if base_ir_type != IrType::Ptr {
                            // The base type is the actual element type
                            // (e.g. I32 for `int **pp`).
                            ctx.variable_deep_pointee_types
                                .insert(var_name, base_ir_type.clone());
                        }
                    }
                }

                // Track the C type for this variable. This is essential for
                // _Generic selection and member access resolution.
                if let Some(tag) = struct_tag {
                    let tag_name_str = ctx.module_ctx.interner.resolve(tag).to_string();
                    // Determine if this is a union or struct from the specifiers.
                    let is_union_type = is_union_specifier(specifiers);
                    if is_union_type {
                        ctx.variable_ctypes.insert(
                            var_name,
                            CType::Union {
                                name: Some(tag_name_str),
                                fields: Vec::new(),
                            },
                        );
                    } else {
                        ctx.variable_ctypes.insert(
                            var_name,
                            CType::Struct {
                                name: Some(tag_name_str),
                                fields: Vec::new(),
                            },
                        );
                    }
                } else if let Some(ref_ctype) = extract_typeof_ctype(ctx, specifiers) {
                    // typeof-derived variable: propagate the CType so that
                    // member access can resolve field indices correctly.
                    ctx.variable_ctypes.insert(var_name, ref_ctype);
                } else {
                    // For all other types (int, unsigned int, float, char *,
                    // etc.), resolve the C type from the specifiers and
                    // derived declarators so that _Generic can correctly
                    // match controlling expression types.
                    if let Some(ctype) =
                        specifiers_to_ctype(ctx, specifiers, &init_decl.declarator.derived)
                    {
                        ctx.variable_ctypes.insert(var_name, ctype);
                    }
                }

                // If there is an initializer, lower it and store the value.
                if let Some(ref initializer) = init_decl.initializer {
                    lower_variable_initializer(ctx, alloca, initializer, &ir_type, is_unsigned)?;
                }
            }
            Ok(())
        }

        // Typedef declarations record the type mapping so that subsequent
        // variable declarations using the typedef name resolve correctly.
        Declaration::Typedef {
            specifiers,
            declarators,
            ..
        } => {
            let base_ir_type = resolve_base_ir_type_from_specifiers(ctx, specifiers);
            for decl in declarators {
                if let Some(name) = decl.name {
                    let resolved_type = apply_derived_declarators(&base_ir_type, &decl.derived);
                    ctx.typedef_types.insert(name, resolved_type);
                }
            }
            Ok(())
        }

        // Struct/union definitions register field name mappings.
        Declaration::StructDef { name, fields, .. } => {
            if let Some(tag) = name {
                register_struct_field_names(ctx, *tag, fields, false);
            }
            Ok(())
        }
        Declaration::UnionDef { name, fields, .. } => {
            if let Some(tag) = name {
                register_struct_field_names(ctx, *tag, fields, true);
            }
            Ok(())
        }
        // Enum definitions and static asserts are type-system-only constructs.
        Declaration::EnumDef { .. }
        | Declaration::StaticAssert { .. }
        | Declaration::Empty { .. }
        | Declaration::Error { .. } => Ok(()),

        // Function definitions and declarations at block scope are legal
        // in C (local function declarations). They don't produce IR in
        // the current function body.
        Declaration::FunctionDef { .. } | Declaration::FunctionDecl { .. } => Ok(()),
    }
}

/// Resolves the base IR type from declaration specifiers.
///
/// This is a best-effort resolution that handles the common type specifier
/// patterns. Complex cases fall back to `IrType::I32`.
fn resolve_base_ir_type_from_specifiers(
    ctx: &LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> IrType {
    use crate::frontend::parser::ast::TypeSpecifier;

    // Scan the type specifiers to determine the base type.
    let mut has_void = false;
    let mut has_char = false;
    let mut has_short = false;
    let mut has_long = false;
    let mut long_count = 0u32;
    let mut has_float = false;
    let mut has_double = false;
    let mut has_bool = false;
    // `signed` / `unsigned` / `int` are tracked but only influence
    // the width when combined with other specifiers like `short` or
    // `long`.  On their own they resolve to I32.
    let mut _has_unsigned = false;
    let mut _has_signed = false;
    let mut _has_int = false;

    for spec in &specifiers.type_specifiers {
        match spec {
            TypeSpecifier::Void => has_void = true,
            TypeSpecifier::Char => has_char = true,
            TypeSpecifier::Short => has_short = true,
            TypeSpecifier::Int => _has_int = true,
            TypeSpecifier::Long => {
                has_long = true;
                long_count += 1;
            }
            TypeSpecifier::Float => has_float = true,
            TypeSpecifier::Double => has_double = true,
            TypeSpecifier::Unsigned => _has_unsigned = true,
            TypeSpecifier::Signed => _has_signed = true,
            TypeSpecifier::Bool => has_bool = true,
            // Struct/union — resolve to IrType::Struct with fields.
            TypeSpecifier::Struct { name, fields, .. } => {
                return resolve_struct_or_union_ir_type(
                    ctx,
                    name.as_ref().copied(),
                    fields.as_deref(),
                    false,
                );
            }
            TypeSpecifier::Union { name, fields, .. } => {
                return resolve_struct_or_union_ir_type(
                    ctx,
                    name.as_ref().copied(),
                    fields.as_deref(),
                    true,
                );
            }
            TypeSpecifier::Enum { .. } => {
                return IrType::I32;
            }
            // __builtin_va_list — always a pointer type (void *).
            TypeSpecifier::BuiltinVaList => {
                return IrType::Ptr;
            }
            // typeof(expr) / typeof(type-name) — GCC extension.
            TypeSpecifier::Typeof { operand, .. } => {
                return resolve_typeof_ir_type(ctx, operand);
            }
            // _Atomic(type-name)
            TypeSpecifier::Atomic(inner_tn) => {
                // Map _Atomic(T) to the IR type for T (ignoring atomic qualifier
                // at the IR level — atomics are lowered to atomic load/store ops).
                let inner_specs = crate::frontend::parser::ast::DeclarationSpecifiers {
                    type_specifiers: inner_tn.specifiers.specifiers.clone(),
                    type_qualifiers: inner_tn.specifiers.qualifiers.clone(),
                    storage_class: None,
                    function_specifiers: Default::default(),
                    alignment: None,
                    attrs: Vec::new(),
                    has_extension: false,
                    span: crate::common::diagnostics::Span::DUMMY,
                };
                let base = resolve_base_ir_type_from_specifiers(ctx, &inner_specs);
                if let Some(ref decl) = inner_tn.declarator {
                    return apply_derived_declarators(&base, &decl.derived);
                }
                return base;
            }
            // Complex type specifier
            TypeSpecifier::Complex => {
                // _Complex alone (without float/double) defaults to _Complex double
                // Handled via has_float/has_double flags in the combined path below.
                // If standalone, treat as complex double.
                return IrType::Struct {
                    fields: vec![IrType::F64, IrType::F64],
                    packed: false,
                };
            }
            // Typedef names — look up in the typedef map populated
            // during lowering.  Falls back to I32 if not found.
            TypeSpecifier::TypedefName { name, .. } => {
                if let Some(resolved) = ctx.typedef_types.get(name) {
                    return resolved.clone();
                }
                // Fallback: check if name resolves to a known built-in
                // typedef in the interner.
                let name_str = ctx.module_ctx.interner.resolve(*name);
                match name_str {
                    "size_t" | "uintptr_t" | "ptrdiff_t" | "ssize_t" | "intptr_t" => {
                        return match ctx.module_ctx.target.data_model() {
                            crate::common::target::DataModel::LP64 => IrType::I64,
                            crate::common::target::DataModel::ILP32 => IrType::I32,
                        };
                    }
                    "uint8_t" | "int8_t" => return IrType::I8,
                    "uint16_t" | "int16_t" => return IrType::I16,
                    "uint32_t" | "int32_t" => return IrType::I32,
                    "uint64_t" | "int64_t" => return IrType::I64,
                    _ => return IrType::I32,
                }
            }
        }
    }

    if has_void {
        return IrType::Void;
    }
    if has_bool {
        return IrType::I1;
    }
    if has_char {
        return IrType::I8;
    }
    if has_float {
        return IrType::F32;
    }
    if has_double && has_long {
        return IrType::F80;
    }
    if has_double {
        return IrType::F64;
    }
    if has_short {
        return IrType::I16;
    }
    if long_count >= 2 {
        return IrType::I64;
    }
    if has_long {
        // Target-dependent: LP64 → I64, ILP32 → I32.
        // Use I64 as the common case for long on 64-bit targets.
        return IrType::I64;
    }
    // Default: int / signed / unsigned / (empty) → I32
    IrType::I32
}

/// Resolves `typeof(expr)` or `typeof(type-name)` to an IR type.
///
/// For expression-based typeof: looks up the variable in the lowering
/// context to determine its type. For type-name-based typeof: resolves
/// the specifiers and declarators recursively.
fn resolve_typeof_ir_type(
    ctx: &LoweringContext<'_>,
    operand: &crate::frontend::parser::ast::TypeofOperand,
) -> IrType {
    use crate::frontend::parser::ast::TypeofOperand;

    match operand {
        TypeofOperand::Expression(expr) => {
            // Try to determine the type from the expression.
            match expr.as_ref() {
                Expression::Identifier { name, .. } => {
                    // Look up the variable's IR element type.
                    // `variable_ir_types` stores the element type of the alloca
                    // (e.g., I32 for `int x`), unlike `get_value_type` which
                    // always returns Ptr for alloca results.
                    if let Some(ir_ty) = ctx.variable_ir_types.get(name) {
                        return ir_ty.clone();
                    }
                    // Fallback: check typedef types.
                    if let Some(ir_ty) = ctx.typedef_types.get(name) {
                        return ir_ty.clone();
                    }
                    // Not found as a local variable — could be a parameter
                    // or global. Default to I32 as fallback.
                    IrType::I32
                }
                Expression::Dereference { operand, .. } => {
                    // typeof(*ptr) — dereference a pointer. Resolve the
                    // inner expression's type, and if it's Ptr, we need
                    // context to know the pointee type. For now, default I32.
                    let inner_operand = TypeofOperand::Expression(operand.clone());
                    let inner_ty = resolve_typeof_ir_type(ctx, &inner_operand);
                    match inner_ty {
                        IrType::Ptr => IrType::I32, // approximate — pointee type unknown at IR level
                        _ => inner_ty,
                    }
                }
                Expression::UnaryOp { operand, op, .. } => {
                    use crate::frontend::parser::ast::UnaryOperator;
                    match op {
                        UnaryOperator::AddressOf => {
                            // typeof(&x) → pointer
                            IrType::Ptr
                        }
                        _ => {
                            // Propagate the operand type for most unary ops.
                            let inner = TypeofOperand::Expression(operand.clone());
                            resolve_typeof_ir_type(ctx, &inner)
                        }
                    }
                }
                Expression::AddressOf { .. } => IrType::Ptr,
                Expression::IntegerLiteral { suffix, .. } => {
                    use crate::frontend::lexer::token::IntegerSuffix;
                    match suffix {
                        IntegerSuffix::ULL | IntegerSuffix::LL => IrType::I64,
                        IntegerSuffix::UL | IntegerSuffix::L => IrType::I64,
                        _ => IrType::I32,
                    }
                }
                Expression::FloatLiteral { suffix, .. } => {
                    use crate::frontend::lexer::token::FloatSuffix;
                    match suffix {
                        FloatSuffix::F => IrType::F32,
                        FloatSuffix::L => IrType::F64, // long double → F64 approx
                        FloatSuffix::None => IrType::F64,
                    }
                }
                Expression::StringLiteral { .. } => IrType::Ptr,
                Expression::FunctionCall { callee, .. } => {
                    // typeof(func(args)) — resolve the function's return type.
                    // Extract the callee name and look up its signature in
                    // global symbols.
                    if let Expression::Identifier { name, .. } = callee.as_ref() {
                        if let Some(info) = ctx.module_ctx.global_symbols.get(name) {
                            if let IrType::Function {
                                ref return_type, ..
                            } = info.ir_type
                            {
                                return (**return_type).clone();
                            }
                        }
                    }
                    // For indirect calls through function pointers, we
                    // cannot easily determine the return type — default to I32.
                    IrType::I32
                }
                Expression::Cast { type_name, .. } => {
                    // typeof((int *)expr) — resolve the cast target type.
                    let inner_specs = crate::frontend::parser::ast::DeclarationSpecifiers {
                        type_specifiers: type_name.specifiers.specifiers.clone(),
                        type_qualifiers: type_name.specifiers.qualifiers.clone(),
                        storage_class: None,
                        function_specifiers: Default::default(),
                        alignment: None,
                        attrs: Vec::new(),
                        has_extension: false,
                        span: crate::common::diagnostics::Span::DUMMY,
                    };
                    let base = resolve_base_ir_type_from_specifiers(ctx, &inner_specs);
                    if let Some(ref decl) = type_name.declarator {
                        return apply_derived_declarators(&base, &decl.derived);
                    }
                    base
                }
                Expression::BinaryOp {
                    left, right, op, ..
                } => {
                    // typeof(a + b) — use usual arithmetic conversion rules.
                    // For pointer arithmetic, result is a pointer.
                    use crate::frontend::parser::ast::BinaryOperator;
                    let left_ty =
                        resolve_typeof_ir_type(ctx, &TypeofOperand::Expression(left.clone()));
                    let right_ty =
                        resolve_typeof_ir_type(ctx, &TypeofOperand::Expression(right.clone()));
                    match op {
                        BinaryOperator::Add | BinaryOperator::Sub => {
                            // Pointer arithmetic: ptr + int = ptr
                            if left_ty == IrType::Ptr || right_ty == IrType::Ptr {
                                return IrType::Ptr;
                            }
                            // Float promotion: int + double = double
                            if left_ty == IrType::F64 || right_ty == IrType::F64 {
                                return IrType::F64;
                            }
                            if left_ty == IrType::F32 || right_ty == IrType::F32 {
                                return IrType::F32;
                            }
                            // Integer promotion: at least I32
                            if left_ty == IrType::I64 || right_ty == IrType::I64 {
                                return IrType::I64;
                            }
                            IrType::I32
                        }
                        _ => {
                            // Most other binary ops: result is promoted type
                            if left_ty == IrType::F64 || right_ty == IrType::F64 {
                                return IrType::F64;
                            }
                            if left_ty == IrType::I64 || right_ty == IrType::I64 {
                                return IrType::I64;
                            }
                            IrType::I32
                        }
                    }
                }
                Expression::MemberAccess { object, member, .. } => {
                    // typeof(s.field) — resolve the struct field type.
                    // 1. Get the struct's IR type from the object variable.
                    // 2. Look up the field name → index mapping.
                    // 3. Return the field's IR type from the struct.
                    if let Expression::Identifier { name, .. } = object.as_ref() {
                        if let Some(ir_ty) = ctx.variable_ir_types.get(name) {
                            if let IrType::Struct { ref fields, .. } = ir_ty {
                                // Try to find field index via struct_field_names
                                if let Some(ctype) = ctx.variable_ctypes.get(name) {
                                    if let CType::Struct {
                                        name: Some(ref tag_name),
                                        ..
                                    } = ctype
                                    {
                                        if let Some(tag_sym) =
                                            ctx.module_ctx.interner.lookup(tag_name)
                                        {
                                            if let Some(field_names) =
                                                ctx.module_ctx.struct_field_names.get(&tag_sym)
                                            {
                                                for (idx, opt_name) in
                                                    field_names.iter().enumerate()
                                                {
                                                    if let Some(fname) = opt_name {
                                                        if *fname == *member {
                                                            if idx < fields.len() {
                                                                return fields[idx].clone();
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Fallback if struct field resolution fails.
                    IrType::I32
                }
                Expression::ArrowAccess {
                    pointer, member, ..
                } => {
                    // typeof(p->field) — resolve the pointed-to struct field type.
                    if let Expression::Identifier { name, .. } = pointer.as_ref() {
                        if let Some(ctype) = ctx.variable_ctypes.get(name) {
                            if let CType::Struct {
                                name: Some(ref tag_name),
                                ..
                            } = ctype
                            {
                                if let Some(tag_sym) = ctx.module_ctx.interner.lookup(tag_name) {
                                    // Get the struct's IR type from the module-level struct defs
                                    if let Some(field_names) =
                                        ctx.module_ctx.struct_field_names.get(&tag_sym)
                                    {
                                        // Look up the actual struct IR type from the pointee
                                        if let Some(pointee_ty) =
                                            ctx.variable_pointee_types.get(name)
                                        {
                                            if let IrType::Struct { ref fields, .. } = pointee_ty {
                                                for (idx, opt_name) in
                                                    field_names.iter().enumerate()
                                                {
                                                    if let Some(fname) = opt_name {
                                                        if *fname == *member {
                                                            if idx < fields.len() {
                                                                return fields[idx].clone();
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Fallback if field resolution fails.
                    IrType::I32
                }
                Expression::ArraySubscript { array, .. } => {
                    // typeof(arr[i]) — dereference the array element type.
                    let arr_ty =
                        resolve_typeof_ir_type(ctx, &TypeofOperand::Expression(array.clone()));
                    match arr_ty {
                        IrType::Array { ref element, .. } => (**element).clone(),
                        IrType::Ptr => IrType::I32, // pointer subscript defaults
                        _ => arr_ty,
                    }
                }
                Expression::Conditional { then_expr, .. } => {
                    // typeof(cond ? a : b) — use the `then` branch type.
                    if let Some(then_e) = then_expr {
                        resolve_typeof_ir_type(ctx, &TypeofOperand::Expression(then_e.clone()))
                    } else {
                        IrType::I32
                    }
                }
                _ => {
                    // For complex expressions, attempt to lower the expression
                    // to determine its type. This is a best-effort approach.
                    IrType::I32
                }
            }
        }
        TypeofOperand::TypeName(type_name) => {
            // Resolve from the type name's specifiers.
            let inner_specs = crate::frontend::parser::ast::DeclarationSpecifiers {
                type_specifiers: type_name.specifiers.specifiers.clone(),
                type_qualifiers: type_name.specifiers.qualifiers.clone(),
                storage_class: None,
                function_specifiers: Default::default(),
                alignment: None,
                attrs: Vec::new(),
                has_extension: false,
                span: crate::common::diagnostics::Span::DUMMY,
            };
            let base = resolve_base_ir_type_from_specifiers(ctx, &inner_specs);
            // Apply abstract declarator modifiers (pointers, arrays).
            if let Some(ref decl) = type_name.declarator {
                return apply_derived_declarators(&base, &decl.derived);
            }
            base
        }
    }
}

/// Extracts the struct/union tag name from declaration specifiers.
/// Returns `true` if the declaration specifiers indicate a union type.
/// Resolve declaration specifiers (and optional derived declarators) to a CType.
/// This is used by _Generic and other features that need to know the full C
/// type of a variable at the IR level.
fn specifiers_to_ctype(
    ctx: &LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
    derived: &[crate::frontend::parser::ast::DerivedDeclarator],
) -> Option<CType> {
    use crate::frontend::parser::ast::{DerivedDeclarator, TypeSpecifier};

    let specs = &specifiers.type_specifiers;

    // Determine base type from specifiers.
    let has_unsigned = specs.iter().any(|s| matches!(s, TypeSpecifier::Unsigned));
    let has_signed = specs.iter().any(|s| matches!(s, TypeSpecifier::Signed));
    let long_count = specs
        .iter()
        .filter(|s| matches!(s, TypeSpecifier::Long))
        .count();
    let has_short = specs.iter().any(|s| matches!(s, TypeSpecifier::Short));
    let has_char = specs.iter().any(|s| matches!(s, TypeSpecifier::Char));
    let has_int = specs.iter().any(|s| matches!(s, TypeSpecifier::Int));
    let has_float = specs.iter().any(|s| matches!(s, TypeSpecifier::Float));
    let has_double = specs.iter().any(|s| matches!(s, TypeSpecifier::Double));
    let has_void = specs.iter().any(|s| matches!(s, TypeSpecifier::Void));
    let has_bool = specs.iter().any(|s| matches!(s, TypeSpecifier::Bool));

    let base = if has_void {
        CType::Void
    } else if has_bool {
        CType::Bool
    } else if has_float && !has_double {
        CType::Float
    } else if has_double {
        if long_count > 0 {
            CType::LongDouble
        } else {
            CType::Double
        }
    } else if has_char {
        CType::Char {
            signed: !has_unsigned,
        }
    } else if has_short {
        CType::Short {
            signed: !has_unsigned,
        }
    } else if long_count >= 2 {
        CType::LongLong {
            signed: !has_unsigned,
        }
    } else if long_count == 1 {
        CType::Long {
            signed: !has_unsigned,
        }
    } else if has_unsigned {
        CType::Int { signed: false }
    } else if has_int || has_signed {
        CType::Int { signed: true }
    } else {
        // Check for struct/union/enum/typedef
        for spec in specs {
            match spec {
                TypeSpecifier::Struct { name, .. } => {
                    let n = name.map(|s| ctx.module_ctx.interner.resolve(s).to_string());
                    return Some(CType::Struct {
                        name: n,
                        fields: Vec::new(),
                    });
                }
                TypeSpecifier::Union { name, .. } => {
                    let n = name.map(|s| ctx.module_ctx.interner.resolve(s).to_string());
                    return Some(CType::Union {
                        name: n,
                        fields: Vec::new(),
                    });
                }
                TypeSpecifier::Enum { .. } => {
                    return Some(CType::Int { signed: true });
                }
                TypeSpecifier::TypedefName { name, .. } => {
                    // Try to resolve the typedef to a known CType.
                    // We only have IR types for typedefs, not C types, so use best-effort.
                    if let Some(ir_ty) = ctx.typedef_types.get(name) {
                        return Some(ir_type_to_rough_ctype(ir_ty));
                    }
                    return Some(CType::Int { signed: true }); // fallback
                }
                TypeSpecifier::Typeof { operand, .. } => {
                    if let crate::frontend::parser::ast::TypeofOperand::Expression(expr) = operand {
                        if let crate::frontend::parser::ast::Expression::Identifier {
                            name, ..
                        } = expr.as_ref()
                        {
                            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                                return Some(ctype.clone());
                            }
                        }
                    }
                    return None;
                }
                _ => {}
            }
        }
        return None;
    };

    // Apply derived declarators (pointers, arrays).
    let mut result = base;
    for d in derived {
        match d {
            DerivedDeclarator::Pointer { .. } => {
                result = CType::Pointer(Box::new(result));
            }
            DerivedDeclarator::Array { size, .. } => {
                let count = match size {
                    Some(expr) => {
                        // Try to extract constant size.
                        if let crate::frontend::parser::ast::Expression::IntegerLiteral {
                            value,
                            ..
                        } = expr.as_ref()
                        {
                            *value as usize
                        } else {
                            0
                        }
                    }
                    None => 0,
                };
                result = CType::Array {
                    element: Box::new(result),
                    size: Some(count),
                };
            }
            DerivedDeclarator::Function { .. } => {
                // Function pointer: wrap in Function type then in Pointer.
                result = CType::Function {
                    return_type: Box::new(result),
                    params: Vec::new(),
                    variadic: false,
                };
            }
        }
    }

    Some(result)
}

/// Rough conversion from IR type back to CType (best-effort for _Generic, etc.).
fn ir_type_to_rough_ctype(ir_ty: &IrType) -> CType {
    match ir_ty {
        IrType::Void => CType::Void,
        IrType::I1 => CType::Bool,
        IrType::I8 => CType::Char { signed: true },
        IrType::I16 => CType::Short { signed: true },
        IrType::I32 => CType::Int { signed: true },
        IrType::I64 => CType::Long { signed: true },
        IrType::I128 => CType::LongLong { signed: true },
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void)),
        _ => CType::Int { signed: true },
    }
}

fn is_union_specifier(specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers) -> bool {
    use crate::frontend::parser::ast::TypeSpecifier;
    for spec in &specifiers.type_specifiers {
        match spec {
            TypeSpecifier::Union { .. } => return true,
            TypeSpecifier::Struct { .. } => return false,
            _ => {}
        }
    }
    false
}

fn extract_struct_tag_from_specifiers(
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> Option<Symbol> {
    use crate::frontend::parser::ast::{TypeSpecifier, TypeofOperand};
    for spec in &specifiers.type_specifiers {
        match spec {
            TypeSpecifier::Struct { name, .. } | TypeSpecifier::Union { name, .. } => {
                return *name;
            }
            // Handle typeof(struct tag) — peel through typeof to find
            // the struct tag in the wrapped type specifiers.
            TypeSpecifier::Typeof { operand, .. } => {
                if let TypeofOperand::TypeName(type_name) = operand {
                    for inner_spec in &type_name.specifiers.specifiers {
                        match inner_spec {
                            TypeSpecifier::Struct { name, .. }
                            | TypeSpecifier::Union { name, .. } => {
                                return *name;
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Extracts the CType for a typeof-declared variable by tracing back to
/// the referenced expression or type.  When `typeof(p)` is used and `p`
/// has a known CType, that CType is returned so it can be propagated to
/// the new variable.  This enables member access field resolution even
/// when the type is expressed through typeof indirection.
fn extract_typeof_ctype(
    ctx: &LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> Option<CType> {
    use crate::frontend::parser::ast::{Expression, TypeSpecifier, TypeofOperand};
    for spec in &specifiers.type_specifiers {
        if let TypeSpecifier::Typeof { operand, .. } = spec {
            match operand {
                TypeofOperand::Expression(expr) => {
                    // typeof(variable) — look up the variable's CType
                    match expr.as_ref() {
                        Expression::Identifier { name, .. } => {
                            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                                return Some(ctype.clone());
                            }
                        }
                        // typeof(expr.field) or typeof(expr->field) — try
                        // to get the struct type from the inner expression
                        Expression::MemberAccess { object, .. }
                        | Expression::ArrowAccess {
                            pointer: object, ..
                        } => {
                            if let Expression::Identifier { name, .. } = object.as_ref() {
                                if let Some(ctype) = ctx.variable_ctypes.get(name) {
                                    return Some(ctype.clone());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                TypeofOperand::TypeName(_) => {
                    // typeof(struct pair) — the struct tag is already
                    // handled by extract_struct_tag_from_specifiers above.
                    // No extra CType propagation needed here.
                }
            }
        }
    }
    None
}

/// Registers struct/union field names from declaration specifiers into the
/// module-level registry.  Called before `resolve_base_ir_type_from_specifiers`
/// so that the field mapping is available immediately.
/// Registers field name→index mappings for any struct/union type specifiers
/// found in the declaration.  Returns an optional synthetic tag for anonymous
/// Checks whether a declaration's type specifiers indicate an unsigned
/// integer type.  Returns `true` for `unsigned`, `unsigned int`,
/// `unsigned long`, `unsigned short`, `unsigned char`, `unsigned long long`.
/// Also returns `true` for `typeof(unsigned_var)` when the typeof operand
/// refers to a known-unsigned variable.
fn is_unsigned_from_specifiers(
    ctx: &LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> bool {
    use crate::frontend::parser::ast::TypeSpecifier;
    for spec in &specifiers.type_specifiers {
        match spec {
            TypeSpecifier::Unsigned => {
                return true;
            }
            TypeSpecifier::Typeof { operand, .. } => {
                // typeof(var) — check if the source variable is unsigned.
                if let crate::frontend::parser::ast::TypeofOperand::Expression(expr) = operand {
                    if let crate::frontend::parser::ast::Expression::Identifier { name, .. } =
                        expr.as_ref()
                    {
                        if ctx.variable_unsigned.contains(name) {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// structs so the caller can associate variables with it for field resolution.
fn register_struct_fields_from_specifiers(
    ctx: &mut LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> Option<Symbol> {
    use crate::frontend::parser::ast::TypeSpecifier;
    let mut anon_tag = None;
    for spec in &specifiers.type_specifiers {
        match spec {
            TypeSpecifier::Struct {
                name: Some(tag),
                fields: Some(fields),
                ..
            } => {
                register_struct_field_names(ctx, *tag, fields, false);
            }
            TypeSpecifier::Union {
                name: Some(tag),
                fields: Some(fields),
                ..
            } => {
                register_struct_field_names(ctx, *tag, fields, true);
            }
            // Handle anonymous struct: no tag name but has inline fields.
            TypeSpecifier::Struct {
                name: None,
                fields: Some(fields),
                ..
            } => {
                let id = ctx.anon_struct_counter;
                ctx.anon_struct_counter += 1;
                let synthetic_name = format!("__anon_struct_{}", id);
                let sym = ctx.module_ctx.interner.intern(&synthetic_name);
                register_struct_field_names(ctx, sym, fields, false);
                anon_tag = Some(sym);
            }
            TypeSpecifier::Union {
                name: None,
                fields: Some(fields),
                ..
            } => {
                let id = ctx.anon_struct_counter;
                ctx.anon_struct_counter += 1;
                let synthetic_name = format!("__anon_union_{}", id);
                let sym = ctx.module_ctx.interner.intern(&synthetic_name);
                register_struct_field_names(ctx, sym, fields, true);
                anon_tag = Some(sym);
            }
            _ => {}
        }
    }
    anon_tag
}

/// Registers the field name→index mapping for a struct/union definition
/// in the module-level registry. This mapping is used by `infer_field_index`
/// to resolve member names (e.g. `.x`, `.y`) to their positional index.
fn register_struct_field_names(
    ctx: &mut LoweringContext<'_>,
    tag: Symbol,
    fields: &[crate::frontend::parser::ast::FieldDeclaration],
    is_union: bool,
) {
    use crate::common::types::{CType, FieldDef};
    let mut field_names: Vec<Option<Symbol>> = Vec::new();
    let mut ctype_fields: Vec<FieldDef> = Vec::new();

    for fd in fields {
        // Convert the AST specifiers to an IR type, then approximate
        // the corresponding CType for the struct_defs registry.
        let base_ir = resolve_base_ir_type_from_specifiers(ctx, &fd.specifiers);

        if fd.declarators.is_empty() {
            field_names.push(None);
            // Anonymous field (e.g. anonymous struct/union)
            ctype_fields.push(FieldDef {
                name: None,
                ty: ir_type_to_ctype_approx(&base_ir, &ctx.module_ctx.target),
                bit_width: None,
            });
        } else {
            for fdecl in &fd.declarators {
                let name_sym = fdecl.declarator.as_ref().and_then(|d| d.name);
                field_names.push(name_sym);

                let field_ir = if let Some(ref decl) = fdecl.declarator {
                    apply_derived_declarators(&base_ir, &decl.derived)
                } else {
                    base_ir.clone()
                };

                let name_str = name_sym.map(|s| ctx.module_ctx.interner.resolve(s).to_string());
                ctype_fields.push(FieldDef {
                    name: name_str,
                    ty: ir_type_to_ctype_approx(&field_ir, &ctx.module_ctx.target),
                    bit_width: None,
                });
            }
        }
    }
    ctx.module_ctx.struct_field_names.insert(tag, field_names);

    // Also register in struct_defs so that infer_base_struct_type can
    // find function-local struct definitions and resolve field IR types
    // correctly (e.g. float vs int).
    if !ctx.module_ctx.struct_defs.contains_key(&tag) {
        let tag_str = ctx.module_ctx.interner.resolve(tag).to_string();
        let ctype = if is_union {
            CType::Union {
                name: Some(tag_str),
                fields: ctype_fields,
            }
        } else {
            CType::Struct {
                name: Some(tag_str),
                fields: ctype_fields,
            }
        };
        ctx.module_ctx.struct_defs.insert(tag, ctype);
    }
}

/// Approximates a CType from an IrType for struct_defs registration.
/// This reverse mapping is not perfect but preserves the essential type
/// information (int/float/pointer/array sizes) needed for field resolution.
fn ir_type_to_ctype_approx(ir: &IrType, _target: &crate::common::target::Target) -> CType {
    use crate::common::types::{CType, FieldDef};
    match ir {
        IrType::I1 => CType::Bool,
        IrType::I8 => CType::Char { signed: true },
        IrType::I16 => CType::Short { signed: true },
        IrType::I32 => CType::Int { signed: true },
        IrType::I64 => CType::LongLong { signed: true },
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::F80 => CType::LongDouble,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void)),
        IrType::Array { element, count } => CType::Array {
            element: Box::new(ir_type_to_ctype_approx(element, _target)),
            size: Some(*count),
        },
        IrType::Struct { fields, .. } => CType::Struct {
            name: None,
            fields: fields
                .iter()
                .map(|f| FieldDef {
                    name: None,
                    ty: ir_type_to_ctype_approx(f, _target),
                    bit_width: None,
                })
                .collect(),
        },
        _ => CType::Int { signed: true },
    }
}

/// Resolves a struct or union type specifier to its IR type.
///
/// For structs: produces `IrType::Struct { fields, packed: false }`.
/// For unions:  produces `IrType::Array { I8, max_field_size }` (all
///              fields share the same starting address).
///
/// Also registers the field name→index mapping in the module context
/// so that `infer_field_index` can resolve member names.
fn resolve_struct_or_union_ir_type(
    ctx: &LoweringContext<'_>,
    tag_name: Option<Symbol>,
    inline_fields: Option<&[crate::frontend::parser::ast::FieldDeclaration]>,
    is_union: bool,
) -> IrType {
    // Helper to convert a single field's specifiers to IR type.
    let field_spec_to_ir = |specs: &crate::frontend::parser::ast::DeclarationSpecifiers| -> IrType {
        resolve_base_ir_type_from_specifiers(ctx, specs)
    };

    // Helper to extract field list from either inline or looked-up definition.
    #[allow(unused_variables)]
    let build_fields = |field_decls: &[crate::frontend::parser::ast::FieldDeclaration]|
        -> (Vec<IrType>, Vec<Option<Symbol>>) {
        let mut ir_fields = Vec::new();
        let mut field_names: Vec<Option<Symbol>> = Vec::new();
        for fd in field_decls {
            let base_ty = field_spec_to_ir(&fd.specifiers);
            if fd.declarators.is_empty() {
                // Anonymous field (e.g. anonymous struct/union member)
                ir_fields.push(base_ty);
                field_names.push(None);
            } else {
                for fdecl in &fd.declarators {
                    let ty = if let Some(ref decl) = fdecl.declarator {
                        apply_derived_declarators(&base_ty, &decl.derived)
                    } else {
                        base_ty.clone()
                    };
                    let name = fdecl.declarator.as_ref().and_then(|d| d.name);
                    ir_fields.push(ty);
                    field_names.push(name);
                }
            }
        }
        (ir_fields, field_names)
    };

    // Try inline fields first, then look up by tag name from registries.
    let ir_fields = if let Some(fields) = inline_fields {
        let (f, _names) = build_fields(fields);
        f
    } else if let Some(tag) = tag_name {
        // Strategy: check both registries.
        // struct_field_names is populated for function-body defined structs.
        // struct_defs is populated for file-scope defined structs via CheckedDeclaration.
        // We prefer struct_defs because it has the full CType with field types.

        if let Some(ctype) = ctx.module_ctx.struct_defs.get(&tag) {
            // File-scope struct definition available — convert fields from CType.
            match ctype {
                CType::Struct {
                    fields: cfields, ..
                } => cfields
                    .iter()
                    .map(|f| {
                        super::c_type_to_ir_type(&f.ty, &ctx.module_ctx.target)
                            .unwrap_or(IrType::I32)
                    })
                    .collect(),
                CType::Union {
                    fields: cfields, ..
                } => cfields
                    .iter()
                    .map(|f| {
                        super::c_type_to_ir_type(&f.ty, &ctx.module_ctx.target)
                            .unwrap_or(IrType::I32)
                    })
                    .collect(),
                _ => vec![IrType::I32],
            }
        } else if let Some(names) = ctx.module_ctx.struct_field_names.get(&tag) {
            // Function-body struct with field names but no CType —
            // rebuild IR fields from field count with I32 default.
            names.iter().map(|_| IrType::I32).collect()
        } else {
            // Not registered in either registry — return a placeholder.
            return IrType::Struct {
                fields: vec![IrType::I32],
                packed: false,
            };
        }
    } else {
        // Anonymous struct with no fields visible.
        return IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
    };

    if is_union {
        // Union: all fields share offset 0. The IR type is an I8 array
        // sized to the largest field, as per the union representation convention.
        let target = &ctx.module_ctx.target;
        let max_size = ir_fields
            .iter()
            .map(|f| f.size_bytes(target))
            .max()
            .unwrap_or(4);
        IrType::Array {
            element: Box::new(IrType::I8),
            count: max_size as usize,
        }
    } else {
        IrType::Struct {
            fields: ir_fields,
            packed: false,
        }
    }
}

/// Adjusts an IR type based on derived declarators (pointers, arrays, etc.).
/// Returns the array size for a `char arr[] = "..."` initializer.
///
/// When a string literal is used to initialize an unsized character array,
/// the array size is the string length plus the null terminator (1 byte).
/// Returns `None` if the initializer is not a plain string literal.
fn get_string_literal_array_size(
    init: &crate::frontend::parser::ast::Initializer,
) -> Option<usize> {
    use crate::frontend::parser::ast::Initializer;
    match init {
        Initializer::Expression(expr) => match expr.as_ref() {
            Expression::StringLiteral { value, prefix, .. } => {
                let null_size = match prefix {
                    crate::frontend::parser::ast::StringPrefix::None
                    | crate::frontend::parser::ast::StringPrefix::U8 => 1,
                    crate::frontend::parser::ast::StringPrefix::L
                    | crate::frontend::parser::ast::StringPrefix::BigU => 4,
                    crate::frontend::parser::ast::StringPrefix::SmallU => 2,
                };
                Some(value.len() + null_size)
            }
            _ => None,
        },
        _ => None,
    }
}

fn apply_derived_declarators(
    base_type: &IrType,
    derived: &[crate::frontend::parser::ast::DerivedDeclarator],
) -> IrType {
    use crate::frontend::parser::ast::DerivedDeclarator;

    let mut current_type = base_type.clone();

    for d in derived {
        match d {
            DerivedDeclarator::Pointer { .. } => {
                // Pointer to anything → IrType::Ptr (opaque pointer model).
                current_type = IrType::Ptr;
            }
            DerivedDeclarator::Array { size, .. } => {
                // Array of current_type with given size.
                let count = size
                    .as_ref()
                    .and_then(|s| match s.as_ref() {
                        Expression::IntegerLiteral { value, .. } => Some(*value as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                current_type = IrType::Array {
                    element: Box::new(current_type),
                    count,
                };
            }
            DerivedDeclarator::Function { .. } => {
                // Function declarator at block scope → function pointer type.
                current_type = IrType::Ptr;
            }
        }
    }

    current_type
}

/// Computes the IR type produced when a pointer variable is dereferenced
/// once.  Given the base type and derived declarator chain, this function
/// applies one fewer pointer indirection than the full chain.
///
/// Examples:
/// - `int *p`   (base=I32, derived=[Pointer])       → pointee = I32
/// - `int **pp` (base=I32, derived=[Pointer, Pointer]) → pointee = Ptr
/// - `char *s`  (base=I8,  derived=[Pointer])        → pointee = I8
/// - `int (*a)[10]` (base=I32, derived=[Pointer, Array(10)]) → pointee = Array(I32, 10)
fn compute_pointee_type(
    base_type: &IrType,
    derived: &[crate::frontend::parser::ast::DerivedDeclarator],
) -> IrType {
    use crate::frontend::parser::ast::DerivedDeclarator;

    // Count how many pointer derivations exist.
    let ptr_count = derived
        .iter()
        .filter(|d| matches!(d, DerivedDeclarator::Pointer { .. }))
        .count();

    if ptr_count <= 1 {
        // Single pointer: dereferencing gives the base type (possibly
        // modified by array/function derivations that precede the pointer).
        // Apply all non-pointer derivations to get the element type.
        let mut ty = base_type.clone();
        for d in derived {
            match d {
                DerivedDeclarator::Pointer { .. } => {
                    // Skip the single pointer — we're computing what's behind it.
                    break;
                }
                DerivedDeclarator::Array { size, .. } => {
                    let count = size
                        .as_ref()
                        .and_then(|s| match s.as_ref() {
                            Expression::IntegerLiteral { value, .. } => Some(*value as usize),
                            _ => None,
                        })
                        .unwrap_or(0);
                    ty = IrType::Array {
                        element: Box::new(ty),
                        count,
                    };
                }
                DerivedDeclarator::Function { .. } => {
                    ty = IrType::Ptr;
                    break;
                }
            }
        }
        ty
    } else {
        // Multiple pointers: dereferencing removes one layer, result is
        // still a pointer.
        IrType::Ptr
    }
}

/// Resolves the pointee type for a pointer variable whose pointer-ness
/// originates from the declaration specifiers (typeof, typedef) rather
/// than from derived declarators (`*`).
///
/// When we have `typeof(p) q` where `p` is `int*`, the base IR type is
/// already `Ptr` and there are no pointer derivations.
/// `compute_pointee_type(Ptr, [])` incorrectly returns `Ptr` (the base
/// itself).  This function looks into the typeof / typedef expression to
/// determine the real pointee.
/// Resolves the "deep" pointee type from a `typeof(var)` specifier by
/// inheriting the deep_pointee recorded for the referenced variable.
///
/// For `typeof(pp) pp2` where `pp` is `int**`, `pp` has
/// `deep_pointee = I32`.  This function returns `Some(I32)`.
fn resolve_deep_pointee_from_typeof(
    ctx: &LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> Option<IrType> {
    use crate::frontend::parser::ast::{Expression, TypeSpecifier, TypeofOperand};
    for spec in &specifiers.type_specifiers {
        if let TypeSpecifier::Typeof { operand, .. } = spec {
            if let TypeofOperand::Expression(expr) = operand {
                if let Expression::Identifier { name, .. } = expr.as_ref() {
                    // Inherit the deep_pointee from the source variable.
                    if let Some(deep) = ctx.variable_deep_pointee_types.get(name) {
                        return Some(deep.clone());
                    }
                }
            }
        }
    }
    None
}

fn resolve_pointee_from_specifiers(
    ctx: &LoweringContext<'_>,
    specifiers: &crate::frontend::parser::ast::DeclarationSpecifiers,
) -> Option<IrType> {
    use crate::frontend::parser::ast::TypeSpecifier;
    for spec in &specifiers.type_specifiers {
        match spec {
            TypeSpecifier::Typeof { operand, .. } => {
                return resolve_typeof_pointee(ctx, operand);
            }
            TypeSpecifier::TypedefName { name, .. } => {
                // If the typedef expands to a pointer type, check whether
                // we recorded a pointee for it.  Otherwise fall back to I32
                // which is the most common element type for C pointer-typed
                // typedefs (e.g. `typedef int *intptr;`).
                if let Some(resolved) = ctx.typedef_types.get(name) {
                    if *resolved == IrType::Ptr {
                        return Some(IrType::I32);
                    }
                }
                return None;
            }
            _ => {}
        }
    }
    None
}

/// Given a `typeof(...)` operand that is known to evaluate to a pointer
/// type, determine the type it points to.
///
/// Examples:
///  - `typeof(&val)` where `val: int`   → pointee is I32
///  - `typeof(p)` where `p: int*`       → pointee from `variable_pointee_types[p]`
///  - `typeof(func(...))` returning ptr  → I32 (default)
///  - `typeof(typeof(x) *)` with x: int → pointee is I32
fn resolve_typeof_pointee(
    ctx: &LoweringContext<'_>,
    operand: &crate::frontend::parser::ast::TypeofOperand,
) -> Option<IrType> {
    use crate::frontend::parser::ast::{
        DerivedDeclarator, Expression, TypeofOperand, UnaryOperator,
    };

    match operand {
        TypeofOperand::Expression(expr) => match expr.as_ref() {
            // typeof(&x) → pointer to typeof(x).  Pointee = typeof(x).
            Expression::UnaryOp {
                op: UnaryOperator::AddressOf,
                operand: inner,
                ..
            }
            | Expression::AddressOf { operand: inner, .. } => {
                let inner_op = TypeofOperand::Expression(inner.clone());
                Some(resolve_typeof_ir_type(ctx, &inner_op))
            }

            // typeof(var) → look up var's known pointee type.
            Expression::Identifier { name, .. } => ctx.variable_pointee_types.get(name).cloned(),

            // typeof(*ptr) resolves to the pointee type of ptr.  If that
            // pointee is itself a Ptr (e.g. ptr is `int**`, *ptr is `int*`),
            // the pointee-of-the-result is one level further → default I32.
            Expression::Dereference { .. } => Some(IrType::I32),

            // typeof(func(...)) returning a pointer — no richer type info.
            Expression::FunctionCall { .. } => Some(IrType::I32),

            // Any other expression producing Ptr — conservative default.
            _ => Some(IrType::I32),
        },

        TypeofOperand::TypeName(type_name) => {
            // typeof(T *)   → Ptr, pointee is T
            // typeof(T **)  → Ptr, pointee is Ptr
            let specs = crate::frontend::parser::ast::DeclarationSpecifiers {
                type_specifiers: type_name.specifiers.specifiers.clone(),
                type_qualifiers: type_name.specifiers.qualifiers.clone(),
                storage_class: None,
                function_specifiers: Default::default(),
                alignment: None,
                attrs: Vec::new(),
                has_extension: false,
                span: crate::common::diagnostics::Span::DUMMY,
            };
            let base = resolve_base_ir_type_from_specifiers(ctx, &specs);

            if let Some(ref decl) = type_name.declarator {
                let ptr_count = decl
                    .derived
                    .iter()
                    .filter(|d| matches!(d, DerivedDeclarator::Pointer { .. }))
                    .count();
                if ptr_count == 1 {
                    // typeof(T *) — pointee is T (the base).
                    return Some(base);
                } else if ptr_count > 1 {
                    // typeof(T **) — pointee is still a pointer.
                    return Some(IrType::Ptr);
                }
            }
            // typeof(T) where T itself is Ptr (e.g. via nested typeof).
            Some(IrType::I32)
        }
    }
}

/// Lowers a variable initializer and stores the result into the alloca.
/// Resolves a designator list (e.g., `.z`) to the actual struct field
/// index by looking up the field name in the struct_field_names registry.
///
/// For a designator `.z` on a struct with fields `[x, y, z]`, this returns
/// `Some(2)` because `z` is at index 2.
/// Evaluate a constant array index expression at compile time.
/// For cases like `[3] = 30`, the expression is typically an integer literal.
fn eval_const_array_index(
    _ctx: &LoweringContext<'_>,
    expr: &crate::frontend::parser::ast::Expression,
) -> usize {
    use crate::frontend::parser::ast::Expression;
    match expr {
        Expression::IntegerLiteral { value, .. } => *value as usize,
        Expression::UnaryOp { op, operand, .. } => {
            let inner = eval_const_array_index(_ctx, operand);
            match op {
                crate::frontend::parser::ast::UnaryOperator::Neg => (-(inner as i64)) as usize,
                _ => inner,
            }
        }
        _ => 0, // fallback
    }
}

fn resolve_designator_field_index(
    ctx: &LoweringContext<'_>,
    designators: &[crate::frontend::parser::ast::Designator],
    ir_type: &IrType,
) -> Option<usize> {
    use crate::frontend::parser::ast::Designator;

    // We only handle a single Field designator for now (the common case).
    if designators.len() != 1 {
        return None;
    }
    let field_name = match &designators[0] {
        Designator::Field(sym) => *sym,
        _ => return None,
    };

    // Look up the struct tag associated with this IR type.
    // Try all struct tags in struct_field_names and find one whose fields
    // include the designated field name.
    let field_name_str = ctx.module_ctx.interner.resolve(field_name);

    for (_tag, field_names) in ctx.module_ctx.struct_field_names.iter() {
        for (idx, opt_name) in field_names.iter().enumerate() {
            if let Some(name_sym) = opt_name {
                let name_str = ctx.module_ctx.interner.resolve(*name_sym);
                if name_str == field_name_str {
                    // Verify the index is valid for the IR type
                    if let IrType::Struct { fields, .. } = ir_type {
                        if idx < fields.len() {
                            return Some(idx);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Resolves a multi-level designator chain (e.g., `.origin.z` or `.items[0]`)
/// by emitting GEPs step by step.  Returns `(final_ptr, final_ir_type, first_field_idx)`
/// where `first_field_idx` is the top-level struct field index (for pos_index tracking).
fn resolve_nested_designator_chain(
    ctx: &mut LoweringContext<'_>,
    base_ptr: ValueId,
    base_ir_type: &IrType,
    designators: &[crate::frontend::parser::ast::Designator],
) -> Option<(ValueId, IrType, usize)> {
    use crate::frontend::parser::ast::Designator;

    if designators.is_empty() {
        return None;
    }

    let mut current_ptr = base_ptr;
    let mut current_type = base_ir_type.clone();
    let mut first_idx: Option<usize> = None;

    for (_d_idx, desig) in designators.iter().enumerate() {
        match desig {
            Designator::Field(sym) => {
                // We need to find the field index for this name in current_type
                let field_name_str = ctx.module_ctx.interner.resolve(*sym).to_string();

                match &current_type {
                    IrType::Struct { fields, .. } => {
                        // Find field index by searching struct_field_names
                        let mut found_idx = None;
                        for (_tag, field_names) in ctx.module_ctx.struct_field_names.iter() {
                            // Check this struct's field names match the current type's field count
                            if field_names.len() == fields.len() {
                                for (fidx, opt_name) in field_names.iter().enumerate() {
                                    if let Some(name_sym) = opt_name {
                                        let ns = ctx.module_ctx.interner.resolve(*name_sym);
                                        if ns == field_name_str && fidx < fields.len() {
                                            found_idx = Some(fidx);
                                            break;
                                        }
                                    }
                                }
                                if found_idx.is_some() {
                                    break;
                                }
                            }
                        }

                        let fidx = found_idx.unwrap_or(0);
                        if first_idx.is_none() {
                            first_idx = Some(fidx);
                        }

                        // Emit GEP [0, fidx]
                        let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                        let fi =
                            ctx.builder
                                .build_const_int(ctx.function, IrType::I32, fidx as i64);
                        let field_ptr = ctx.builder.build_gep(
                            ctx.function,
                            current_ptr,
                            vec![zero, fi],
                            current_type.clone(),
                            true,
                        );
                        current_ptr = field_ptr;
                        current_type = fields[fidx].clone();
                    }
                    _ => return None,
                }
            }
            Designator::Index(expr) => {
                let idx_val = eval_const_array_index(ctx, expr);
                if first_idx.is_none() {
                    first_idx = Some(idx_val);
                }

                match &current_type {
                    IrType::Array { element, .. } => {
                        let index =
                            ctx.builder
                                .build_const_int(ctx.function, IrType::I64, idx_val as i64);
                        let elem_ptr = ctx.builder.build_gep(
                            ctx.function,
                            current_ptr,
                            vec![index],
                            current_type.clone(),
                            true,
                        );
                        current_ptr = elem_ptr;
                        current_type = (**element).clone();
                    }
                    _ => return None,
                }
            }
            Designator::IndexRange(_, _) => {
                // Range designators are handled separately
                return None;
            }
        }
    }

    first_idx.map(|fi| (current_ptr, current_type, fi))
}

fn lower_variable_initializer(
    ctx: &mut LoweringContext<'_>,
    alloca: ValueId,
    initializer: &crate::frontend::parser::ast::Initializer,
    ir_type: &IrType,
    is_unsigned: bool,
) -> Result<(), LoweringError> {
    use crate::frontend::parser::ast::Initializer;

    match initializer {
        Initializer::Expression(expr) => {
            // Special case: string literal initializing a character array.
            // Instead of storing a pointer to .rodata, expand the string
            // bytes directly into the stack-allocated array.
            if let IrType::Array { ref element, count } = ir_type {
                if matches!(**element, IrType::I8) {
                    if let Expression::StringLiteral { value, prefix, .. } = expr.as_ref() {
                        let null_size = match prefix {
                            crate::frontend::parser::ast::StringPrefix::None
                            | crate::frontend::parser::ast::StringPrefix::U8 => 1,
                            crate::frontend::parser::ast::StringPrefix::L
                            | crate::frontend::parser::ast::StringPrefix::BigU => 4,
                            crate::frontend::parser::ast::StringPrefix::SmallU => 2,
                        };
                        let total = if *count > 0 {
                            *count
                        } else {
                            value.len() + null_size
                        };
                        // Array type for reference (GEP uses element type).
                        let _array_ty = IrType::Array {
                            element: Box::new(IrType::I8),
                            count: total,
                        };
                        // Emit individual byte stores for each character.
                        for i in 0..total {
                            let byte_val = if i < value.len() {
                                value[i] as i64
                            } else {
                                0i64 // null terminator or zero-padding
                            };
                            let val =
                                ctx.builder
                                    .build_const_int(ctx.function, IrType::I8, byte_val);
                            let idx =
                                ctx.builder
                                    .build_const_int(ctx.function, IrType::I64, i as i64);
                            let elem_ptr = ctx.builder.build_gep(
                                ctx.function,
                                alloca,
                                vec![idx],
                                IrType::I8,
                                true,
                            );
                            ctx.builder.build_store(ctx.function, val, elem_ptr);
                        }
                        return Ok(());
                    }
                }
            }
            // Lower the initializer expression and store into the alloca.
            let init_val = lower_expression(ctx, expr)?;
            // Coerce the initializer value to the alloca's type to prevent
            // width mismatches (e.g., storing an I32 literal `16` into an
            // I64 alloca for `unsigned long`, which would leave the upper
            // 32 bits as stack garbage).
            let init_val = coerce_init_to_alloca_type(ctx, init_val, ir_type, is_unsigned);
            ctx.builder.build_store(ctx.function, init_val, alloca);
            Ok(())
        }
        Initializer::List { items, .. } => {
            // Brace-enclosed initializer list: lower each element using GEP
            // to compute the address of each array/struct element.
            use crate::frontend::parser::ast::Designator;

            let is_struct = matches!(ir_type, IrType::Struct { .. });
            let is_array = matches!(ir_type, IrType::Array { .. });

            // Determine element type for GEP
            let elem_ir_type = match ir_type {
                IrType::Array { element, .. } => (**element).clone(),
                IrType::Struct { fields, .. } => {
                    if !fields.is_empty() {
                        fields[0].clone()
                    } else {
                        IrType::I32
                    }
                }
                _ => ir_type.clone(),
            };

            // Detect if ANY item uses designators (designated init).
            // If so, we should zero-initialize the entire aggregate first
            // because designated initializers may skip elements/fields.
            let has_any_designator = items.iter().any(|it| !it.designators.is_empty());

            // Also zero-init if the number of initializer items is fewer
            // than the aggregate size (partial initialization).
            let is_partial = match ir_type {
                IrType::Array { count, .. } => items.len() < *count,
                IrType::Struct { fields, .. } => items.len() < fields.len(),
                _ => false,
            };

            if has_any_designator || is_partial {
                // Zero-initialize the entire aggregate, then selectively
                // overwrite with designated values.
                let type_size = ir_type.size_bytes(&ctx.module_ctx.target);
                if type_size > 0 {
                    // Use a series of 8-byte stores to zero the memory.
                    let zero_8 = ctx.builder.build_const_int(ctx.function, IrType::I64, 0);
                    let zero_4 = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                    let zero_1 = ctx.builder.build_const_int(ctx.function, IrType::I8, 0);
                    let mut offset = 0u64;
                    while offset + 8 <= type_size {
                        let off_val =
                            ctx.builder
                                .build_const_int(ctx.function, IrType::I64, offset as i64);
                        let ptr = ctx.builder.build_gep(
                            ctx.function,
                            alloca,
                            vec![off_val],
                            IrType::Array {
                                element: Box::new(IrType::I8),
                                count: type_size as usize,
                            },
                            true,
                        );
                        ctx.builder.build_store(ctx.function, zero_8, ptr);
                        offset += 8;
                    }
                    while offset + 4 <= type_size {
                        let off_val =
                            ctx.builder
                                .build_const_int(ctx.function, IrType::I64, offset as i64);
                        let ptr = ctx.builder.build_gep(
                            ctx.function,
                            alloca,
                            vec![off_val],
                            IrType::Array {
                                element: Box::new(IrType::I8),
                                count: type_size as usize,
                            },
                            true,
                        );
                        ctx.builder.build_store(ctx.function, zero_4, ptr);
                        offset += 4;
                    }
                    while offset < type_size {
                        let off_val =
                            ctx.builder
                                .build_const_int(ctx.function, IrType::I64, offset as i64);
                        let ptr = ctx.builder.build_gep(
                            ctx.function,
                            alloca,
                            vec![off_val],
                            IrType::Array {
                                element: Box::new(IrType::I8),
                                count: type_size as usize,
                            },
                            true,
                        );
                        ctx.builder.build_store(ctx.function, zero_1, ptr);
                        offset += 1;
                    }
                }
            }

            // ===== Brace Elision Detection =====
            // When initializing an array of structs with flat scalars
            // (no inner braces), distribute the values across struct fields.
            // Example: struct point pts[3] = { 10, 20, 30, 40, 50, 60 };
            // => pts[0].x=10, pts[0].y=20, pts[0].z=30, pts[1].x=40, ...
            if is_array && !has_any_designator {
                if let IrType::Array { element, count } = ir_type {
                    if let IrType::Struct {
                        fields: struct_fields,
                        ..
                    } = element.as_ref()
                    {
                        let fields_per_struct = struct_fields.len();
                        if fields_per_struct > 0 {
                            // Check if items are all scalar (no nested lists)
                            // and item count doesn't match array count but does
                            // match total field count.
                            let all_scalar = items
                                .iter()
                                .all(|it| matches!(it.initializer, Initializer::Expression(_)));
                            let total_flat_fields = count * fields_per_struct;
                            if all_scalar
                                && items.len() != *count
                                && items.len() <= total_flat_fields
                            {
                                // Zero-init the entire array first so that
                                // unfilled elements/fields are guaranteed
                                // to be zero.
                                if items.len() < total_flat_fields {
                                    let type_size = ir_type.size_bytes(&ctx.module_ctx.target);
                                    if type_size > 0 {
                                        let zero_8 = ctx.builder.build_const_int(
                                            ctx.function,
                                            IrType::I64,
                                            0,
                                        );
                                        let zero_4 = ctx.builder.build_const_int(
                                            ctx.function,
                                            IrType::I32,
                                            0,
                                        );
                                        let zero_1 = ctx.builder.build_const_int(
                                            ctx.function,
                                            IrType::I8,
                                            0,
                                        );
                                        let mut off = 0u64;
                                        while off + 8 <= type_size {
                                            let o = ctx.builder.build_const_int(
                                                ctx.function,
                                                IrType::I64,
                                                off as i64,
                                            );
                                            let p = ctx.builder.build_gep(
                                                ctx.function,
                                                alloca,
                                                vec![o],
                                                IrType::Array {
                                                    element: Box::new(IrType::I8),
                                                    count: type_size as usize,
                                                },
                                                true,
                                            );
                                            ctx.builder.build_store(ctx.function, zero_8, p);
                                            off += 8;
                                        }
                                        while off + 4 <= type_size {
                                            let o = ctx.builder.build_const_int(
                                                ctx.function,
                                                IrType::I64,
                                                off as i64,
                                            );
                                            let p = ctx.builder.build_gep(
                                                ctx.function,
                                                alloca,
                                                vec![o],
                                                IrType::Array {
                                                    element: Box::new(IrType::I8),
                                                    count: type_size as usize,
                                                },
                                                true,
                                            );
                                            ctx.builder.build_store(ctx.function, zero_4, p);
                                            off += 4;
                                        }
                                        while off < type_size {
                                            let o = ctx.builder.build_const_int(
                                                ctx.function,
                                                IrType::I64,
                                                off as i64,
                                            );
                                            let p = ctx.builder.build_gep(
                                                ctx.function,
                                                alloca,
                                                vec![o],
                                                IrType::Array {
                                                    element: Box::new(IrType::I8),
                                                    count: type_size as usize,
                                                },
                                                true,
                                            );
                                            ctx.builder.build_store(ctx.function, zero_1, p);
                                            off += 1;
                                        }
                                    }
                                }
                                // Brace-elided flat initialization.
                                let mut item_pos = 0usize;
                                for arr_idx in 0..*count {
                                    let arr_index = ctx.builder.build_const_int(
                                        ctx.function,
                                        IrType::I64,
                                        arr_idx as i64,
                                    );
                                    let struct_ptr = ctx.builder.build_gep(
                                        ctx.function,
                                        alloca,
                                        vec![arr_index],
                                        ir_type.clone(),
                                        true,
                                    );
                                    for (field_idx, field_ty) in struct_fields.iter().enumerate() {
                                        if item_pos < items.len() {
                                            if let Initializer::Expression(expr) =
                                                &items[item_pos].initializer
                                            {
                                                let val = lower_expression(ctx, expr)?;
                                                // Coerce to struct field type to
                                                // prevent store-width mismatch.
                                                let val = coerce_init_to_alloca_type(
                                                    ctx, val, field_ty, false,
                                                );
                                                let zero = ctx.builder.build_const_int(
                                                    ctx.function,
                                                    IrType::I32,
                                                    0,
                                                );
                                                let fi = ctx.builder.build_const_int(
                                                    ctx.function,
                                                    IrType::I32,
                                                    field_idx as i64,
                                                );
                                                let field_ptr = ctx.builder.build_gep(
                                                    ctx.function,
                                                    struct_ptr,
                                                    vec![zero, fi],
                                                    element.as_ref().clone(),
                                                    true,
                                                );
                                                ctx.builder.build_store(
                                                    ctx.function,
                                                    val,
                                                    field_ptr,
                                                );
                                            }
                                            item_pos += 1;
                                        }
                                        // else: zero-init (already done above)
                                    }
                                }
                                return Ok(());
                            }
                        }
                    }
                }
            }

            // Track the "current positional index" for items without
            // designators.  This is separate from `idx` (iteration index)
            // because designated items can reset or skip the position.
            let mut pos_index: usize = 0;

            for (_idx, item) in items.iter().enumerate() {
                // ==== Handle multi-level designators (e.g., .origin.z or .items[0]) ====
                if item.designators.len() > 1 {
                    if let Some((final_ptr, final_type, first_fi)) =
                        resolve_nested_designator_chain(ctx, alloca, ir_type, &item.designators)
                    {
                        pos_index = first_fi + 1;
                        match &item.initializer {
                            Initializer::Expression(expr) => {
                                let val = lower_expression(ctx, expr)?;
                                let val = coerce_init_to_alloca_type(ctx, val, &final_type, false);
                                ctx.builder.build_store(ctx.function, val, final_ptr);
                            }
                            Initializer::List { .. } => {
                                lower_variable_initializer(
                                    ctx,
                                    final_ptr,
                                    &item.initializer,
                                    &final_type,
                                    false,
                                )?;
                            }
                        }
                        continue;
                    }
                    // Fallthrough to standard logic if resolution failed
                }

                // Resolve the actual target index from designators.
                let actual_idx = if !item.designators.is_empty() {
                    if is_struct {
                        let resolved =
                            resolve_designator_field_index(ctx, &item.designators, ir_type)
                                .unwrap_or(pos_index);
                        pos_index = resolved + 1;
                        resolved
                    } else if is_array {
                        // Array index designator: resolve [N] or [lo...hi].
                        match &item.designators[0] {
                            Designator::Index(expr) => {
                                let idx_val = eval_const_array_index(ctx, expr);
                                pos_index = idx_val + 1;
                                idx_val
                            }
                            Designator::IndexRange(lo, hi) => {
                                let lo_val = eval_const_array_index(ctx, lo);
                                let hi_val = eval_const_array_index(ctx, hi);
                                // Store the value at each index in [lo, hi].
                                if let Initializer::Expression(expr) = &item.initializer {
                                    let val = lower_expression(ctx, expr)?;
                                    let val =
                                        coerce_init_to_alloca_type(ctx, val, &elem_ir_type, false);
                                    for range_idx in lo_val..=hi_val {
                                        let index = ctx.builder.build_const_int(
                                            ctx.function,
                                            IrType::I64,
                                            range_idx as i64,
                                        );
                                        let elem_ptr = ctx.builder.build_gep(
                                            ctx.function,
                                            alloca,
                                            vec![index],
                                            ir_type.clone(),
                                            true,
                                        );
                                        ctx.builder.build_store(ctx.function, val, elem_ptr);
                                    }
                                }
                                pos_index = hi_val + 1;
                                continue; // Already handled range stores
                            }
                            Designator::Field(_sym) => {
                                // Field designator on an array? Shouldn't
                                // happen, but use pos_index as fallback.
                                let r =
                                    resolve_designator_field_index(ctx, &item.designators, ir_type)
                                        .unwrap_or(pos_index);
                                pos_index = r + 1;
                                r
                            }
                        }
                    } else {
                        let p = pos_index;
                        pos_index += 1;
                        p
                    }
                } else {
                    let p = pos_index;
                    pos_index += 1;
                    p
                };

                // Determine the element type for this index
                let field_type = match ir_type {
                    IrType::Struct { fields, .. } => {
                        if actual_idx < fields.len() {
                            fields[actual_idx].clone()
                        } else {
                            elem_ir_type.clone()
                        }
                    }
                    _ => elem_ir_type.clone(),
                };

                match &item.initializer {
                    Initializer::Expression(expr) => {
                        let val = lower_expression(ctx, expr)?;
                        // Coerce the initializer value to the target field/element
                        // type.  Without this, an integer literal typed as I32
                        // would be stored with a 4-byte `mov` into an I8 slot,
                        // corrupting adjacent stack variables on i686 and other
                        // backends that choose store width from the value type.
                        let val = coerce_init_to_alloca_type(ctx, val, &field_type, false);
                        if !is_struct && !is_array && items.len() == 1 && actual_idx == 0 {
                            // Single-element scalar init: store directly.
                            ctx.builder.build_store(ctx.function, val, alloca);
                        } else if is_struct {
                            // Struct field: two-index GEP [0, field_idx].
                            let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                            let field_idx_val = ctx.builder.build_const_int(
                                ctx.function,
                                IrType::I32,
                                actual_idx as i64,
                            );
                            let elem_ptr = ctx.builder.build_gep(
                                ctx.function,
                                alloca,
                                vec![zero, field_idx_val],
                                ir_type.clone(),
                                true,
                            );
                            ctx.builder.build_store(ctx.function, val, elem_ptr);
                        } else {
                            // Array element: single-index GEP [actual_idx].
                            let index = ctx.builder.build_const_int(
                                ctx.function,
                                IrType::I64,
                                actual_idx as i64,
                            );
                            let elem_ptr = ctx.builder.build_gep(
                                ctx.function,
                                alloca,
                                vec![index],
                                ir_type.clone(),
                                true,
                            );
                            ctx.builder.build_store(ctx.function, val, elem_ptr);
                        }
                    }
                    Initializer::List { .. } => {
                        // Nested initializer list: compute sub-aggregate
                        // address and recurse.
                        let indices = if is_struct {
                            let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                            let field_idx_val = ctx.builder.build_const_int(
                                ctx.function,
                                IrType::I32,
                                actual_idx as i64,
                            );
                            vec![zero, field_idx_val]
                        } else {
                            let index = ctx.builder.build_const_int(
                                ctx.function,
                                IrType::I64,
                                actual_idx as i64,
                            );
                            vec![index]
                        };
                        let elem_ptr = ctx.builder.build_gep(
                            ctx.function,
                            alloca,
                            indices,
                            ir_type.clone(),
                            true,
                        );
                        lower_variable_initializer(
                            ctx,
                            elem_ptr,
                            &item.initializer,
                            &field_type,
                            false,
                        )?;
                    }
                }
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Initializer value coercion
// ---------------------------------------------------------------------------

/// Coerces an initializer value to match the alloca's target type.
///
/// This prevents width mismatches where, e.g., an integer literal typed as
/// `I32` is stored into an `I64` alloca.  Without this coercion the backend
/// emits a 32-bit store (`movl`) into a 64-bit stack slot, leaving the
/// upper 32 bits as whatever garbage was previously on the stack.
///
/// The `is_unsigned` flag controls whether widening uses zero-extension
/// (for unsigned C types) or sign-extension (for signed C types).
fn coerce_init_to_alloca_type(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    target_ty: &IrType,
    is_unsigned: bool,
) -> ValueId {
    let val_ty = ctx.function.get_value_type(val).clone();
    if val_ty == *target_ty {
        return val;
    }

    // Integer-to-integer coercion: widen or narrow as needed.
    if val_ty.is_integer() && target_ty.is_integer() {
        let from_bits = ir_int_bits(&val_ty);
        let to_bits = ir_int_bits(target_ty);
        if from_bits < to_bits {
            // Widen: unsigned types use zero-extension, signed use sign-extension.
            if is_unsigned {
                return ctx.builder.build_zext(ctx.function, val, target_ty.clone());
            } else {
                return ctx.builder.build_sext(ctx.function, val, target_ty.clone());
            }
        } else if from_bits > to_bits {
            return ctx
                .builder
                .build_trunc(ctx.function, val, target_ty.clone());
        }
    }

    // Integer to pointer (e.g., `void *p = 0;`)
    if val_ty.is_integer() && target_ty.is_pointer() {
        return ctx
            .builder
            .build_int_to_ptr(ctx.function, val, target_ty.clone());
    }

    // Pointer to integer (rare, but legal in C)
    if val_ty.is_pointer() && target_ty.is_integer() {
        return ctx
            .builder
            .build_ptr_to_int(ctx.function, val, target_ty.clone());
    }

    // Fall through: no coercion needed or types are incompatible
    // (e.g., struct, array, float).  Return as-is and let the backend
    // handle it.
    val
}

/// Returns the bit width of an integer IR type, or 0 for non-integer types.
fn ir_int_bits(ty: &IrType) -> u32 {
    match ty {
        IrType::I1 => 1,
        IrType::I8 => 8,
        IrType::I16 => 16,
        IrType::I32 => 32,
        IrType::I64 => 64,
        IrType::I128 => 128,
        _ => 0,
    }
}
