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
use crate::frontend::parser::ast::{
    BlockItem, Declaration, Expression, ForInit, Statement,
};
use crate::frontend::sema::constant_eval::evaluate_integer_constant;
use crate::ir::basic_block::BasicBlockId;
use crate::ir::instructions::ValueId;
use crate::ir::types::IrType;

use super::{
    LoweringContext, LoweringError,
    check_recursion_depth, ensure_not_terminated,
};
use super::expr_lowering::lower_expression;
use super::asm_lowering::lower_asm_statement;

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

    match stmt {
        Statement::Compound { items, span } => {
            lower_compound_stmt(ctx, items, *span)
        }

        Statement::If { condition, then_branch, else_branch, span } => {
            lower_if_stmt(ctx, condition, then_branch, else_branch.as_deref(), *span)
        }

        Statement::While { condition, body, span } => {
            lower_while_stmt(ctx, condition, body, *span)
        }

        Statement::DoWhile { body, condition, span } => {
            lower_do_while_stmt(ctx, body, condition, *span)
        }

        Statement::For { init, condition, increment, body, span } => {
            lower_for_stmt(ctx, init.as_ref(), condition.as_deref(), increment.as_deref(), body, *span)
        }

        Statement::Switch { expression, body, span } => {
            lower_switch_stmt(ctx, expression, body, *span)
        }

        // Case/CaseRange/Default outside of switch context: these are
        // handled specially during switch body lowering. If encountered
        // at the top level, they are erroneous but we lower the body anyway.
        Statement::Case { body, .. }
        | Statement::CaseRange { body, .. }
        | Statement::Default { body, .. } => {
            lower_statement(ctx, body)
        }

        Statement::Goto { label, span } => {
            lower_goto_stmt(ctx, *label, *span)
        }

        Statement::ComputedGoto { target, span } => {
            lower_computed_goto(ctx, target, *span)
        }

        Statement::Break { span } => {
            lower_break(ctx, *span)
        }

        Statement::Continue { span } => {
            lower_continue(ctx, *span)
        }

        Statement::Return { value, span } => {
            lower_return_stmt(ctx, value.as_deref(), *span)
        }

        Statement::Labeled { label, body, span, .. } => {
            lower_labeled_stmt(ctx, *label, body, *span)
        }

        Statement::Expression { expr, .. } => {
            lower_expr_stmt(ctx, expr)
        }

        // Null statement — no-op.
        Statement::Null { .. } => Ok(()),

        // Inline assembly — delegate to asm_lowering module.
        Statement::Asm(asm_stmt) => {
            lower_asm_statement(ctx, asm_stmt)
        }

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
    for item in items {
        match item {
            BlockItem::Statement(stmt) => {
                lower_statement(ctx, stmt)?;
            }
            BlockItem::Declaration(decl) => {
                lower_block_declaration(ctx, decl)?;
            }
        }
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
    ctx.builder.build_cond_branch(ctx.function, cond_val, then_bb, else_bb);

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
    ctx.builder.build_cond_branch(ctx.function, cond_val, body_bb, exit_bb);

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
    let latch_bb = ctx.builder.create_block(ctx.function, Some("dowhile.latch"));
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
    ctx.builder.build_cond_branch(ctx.function, cond_val, body_bb, exit_bb);

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
        ctx.builder.build_cond_branch(ctx.function, cond_val, body_bb, exit_bb);
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
    let body_start_bb = ctx.builder.create_block(ctx.function, Some("switch.body.start"));
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
            let block = ctx.builder.create_block(
                ctx.function,
                Some(&format!("case.{}", case_val)),
            );
            info.cases.push((case_val, block));
            info.value_to_block.insert(case_val, block);
            // Recursively scan the case body for nested case labels.
            collect_switch_cases(ctx, body, info)?;
        }

        Statement::CaseRange { low, high, body, .. } => {
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
        Statement::If { then_branch, else_branch, .. } => {
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
                    ctx.builder.build_branch(ctx.function, case_info.default_block);
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
    // Lower the target expression to a pointer value.
    let target_val = lower_expression(ctx, target)?;

    // Collect all address-taken labels and their blocks.
    let taken_labels: Vec<Symbol> = ctx.address_taken_labels.iter().copied().collect();

    if taken_labels.is_empty() {
        // No address-taken labels — degenerate case. Emit an unreachable-
        // style branch to a dead block.
        let dead_bb = ctx.builder.create_block(ctx.function, Some("computed.goto.dead"));
        if ensure_not_terminated(ctx) {
            ctx.builder.build_branch(ctx.function, dead_bb);
        }
        ctx.builder.set_insert_point(dead_bb);
        return Ok(());
    }

    // Build case entries: map each label to a small integer address.
    // The convention is that &&label produces the block's index as an
    // integer (PtrToInt). We use the BasicBlockId's internal index.
    let mut cases: Vec<(i64, BasicBlockId)> = Vec::new();
    let mut first_block = None;
    for label in &taken_labels {
        let block = ctx.get_or_create_label_block(*label);
        // Use the block ID's numeric value as the switch case value.
        // This must match the value produced by Expression::LabelAddress
        // in expr_lowering.
        cases.push((block.0 as i64, block));
        if first_block.is_none() {
            first_block = Some(block);
        }
    }

    // The default target of the switch is the first label (or a dead block).
    let default_target = first_block.unwrap_or_else(|| {
        ctx.builder.create_block(ctx.function, Some("computed.goto.default"))
    });

    if ensure_not_terminated(ctx) {
        ctx.builder.build_switch(ctx.function, target_val, default_target, cases);
    }

    // After computed goto, subsequent code is unreachable.
    let after_bb = ctx.builder.create_block(ctx.function, Some("after.computed.goto"));
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
fn lower_break(
    ctx: &mut LoweringContext<'_>,
    span: Span,
) -> Result<(), LoweringError> {
    let break_target = ctx.current_break_target().ok_or(
        LoweringError::BreakOutsideLoop { span }
    )?;

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
fn lower_continue(
    ctx: &mut LoweringContext<'_>,
    span: Span,
) -> Result<(), LoweringError> {
    let continue_target = ctx.current_continue_target().ok_or(
        LoweringError::ContinueOutsideLoop { span }
    )?;

    if ensure_not_terminated(ctx) {
        ctx.builder.build_branch(ctx.function, continue_target);
    }

    // Create unreachable block for subsequent dead code.
    let after_bb = ctx.builder.create_block(ctx.function, Some("after.continue"));
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
fn lower_expr_stmt(
    ctx: &mut LoweringContext<'_>,
    expr: &Expression,
) -> Result<(), LoweringError> {
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
fn lower_block_declaration(
    ctx: &mut LoweringContext<'_>,
    decl: &Declaration,
) -> Result<(), LoweringError> {
    match decl {
        Declaration::Variable { specifiers, declarators, .. } => {
            // Resolve the base IR type from declaration specifiers.
            let base_ir_type = resolve_base_ir_type_from_specifiers(ctx, specifiers);

            for init_decl in declarators {
                // Extract the variable name from the declarator.
                let var_name = match init_decl.declarator.name {
                    Some(name) => name,
                    None => continue, // Anonymous declarator — skip.
                };

                // Adjust the type based on derived declarators (pointers, arrays).
                let ir_type = apply_derived_declarators(
                    &base_ir_type,
                    &init_decl.declarator.derived,
                );

                // Create an alloca for this variable.
                let alloca = ctx.create_local_alloca(var_name, ir_type.clone());

                // If there is an initializer, lower it and store the value.
                if let Some(ref initializer) = init_decl.initializer {
                    lower_variable_initializer(ctx, alloca, initializer, &ir_type)?;
                }
            }
            Ok(())
        }

        // Typedef, struct/union/enum definitions, and static asserts are
        // type-system-only constructs that do not produce IR instructions.
        Declaration::Typedef { .. }
        | Declaration::StructDef { .. }
        | Declaration::UnionDef { .. }
        | Declaration::EnumDef { .. }
        | Declaration::StaticAssert { .. }
        | Declaration::Empty { .. }
        | Declaration::Error { .. } => Ok(()),

        // Function definitions and declarations at block scope are legal
        // in C (local function declarations). They don't produce IR in
        // the current function body.
        Declaration::FunctionDef { .. }
        | Declaration::FunctionDecl { .. } => Ok(()),
    }
}

/// Resolves the base IR type from declaration specifiers.
///
/// This is a best-effort resolution that handles the common type specifier
/// patterns. Complex cases fall back to `IrType::I32`.
fn resolve_base_ir_type_from_specifiers(
    _ctx: &LoweringContext<'_>,
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
            // Struct, union, enum, typedef name — use a pointer-width
            // default for struct/union, or I32 for enum/typedef.
            TypeSpecifier::Struct { .. } | TypeSpecifier::Union { .. } => {
                return IrType::Ptr;
            }
            TypeSpecifier::Enum { .. } | TypeSpecifier::TypedefName { .. } => {
                return IrType::I32;
            }
            _ => {}
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

/// Adjusts an IR type based on derived declarators (pointers, arrays, etc.).
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
                let count = size.as_ref().and_then(|s| {
                    match s.as_ref() {
                        Expression::IntegerLiteral { value, .. } => Some(*value as usize),
                        _ => None,
                    }
                }).unwrap_or(0);
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

/// Lowers a variable initializer and stores the result into the alloca.
fn lower_variable_initializer(
    ctx: &mut LoweringContext<'_>,
    alloca: ValueId,
    initializer: &crate::frontend::parser::ast::Initializer,
    _ir_type: &IrType,
) -> Result<(), LoweringError> {
    use crate::frontend::parser::ast::Initializer;

    match initializer {
        Initializer::Expression(expr) => {
            // Lower the initializer expression and store into the alloca.
            let init_val = lower_expression(ctx, expr)?;
            ctx.builder.build_store(ctx.function, init_val, alloca);
            Ok(())
        }
        Initializer::List { items, .. } => {
            // Brace-enclosed initializer list: lower each element.
            // For simple cases (e.g., arrays), store element-by-element.
            // For complex cases, this is a simplification — full designated
            // initializer support requires the semantic layer.
            for (idx, item) in items.iter().enumerate() {
                if let Some(expr) = extract_initializer_expression(item) {
                    let val = lower_expression(ctx, expr)?;
                    // For array elements, compute the GEP address.
                    // Simplified: store directly for single-element initializers.
                    if idx == 0 && items.len() == 1 {
                        ctx.builder.build_store(ctx.function, val, alloca);
                    }
                    // Multi-element initialization requires GEP — the
                    // expr_lowering module handles the full GEP pattern.
                    // For now, store the first element as a common case.
                }
            }
            Ok(())
        }
    }
}

/// Extracts the expression from an initializer item (skipping designators).
fn extract_initializer_expression(
    item: &crate::frontend::parser::ast::InitializerItem,
) -> Option<&Expression> {
    match &item.initializer {
        crate::frontend::parser::ast::Initializer::Expression(e) => Some(e),
        _ => None,
    }
}
