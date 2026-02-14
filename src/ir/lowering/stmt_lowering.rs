//! Statement to IR lowering module.
//!
//! Translates C statement AST nodes into IR basic-block control flow graphs.
//! Manages the if/else, while, for, do-while, switch, goto (including computed),
//! label, break, continue, and return lowering.
//!
//! **Note:** Full implementation pending — this stub enables compilation of
//! the lowering driver module (`mod.rs`).

use crate::ir::instructions::ValueId;
use super::{LoweringContext, LoweringError};
use crate::frontend::parser::ast::Statement;

/// Lowers a single C statement to IR.
///
/// Dispatches on the statement variant and emits the corresponding basic
/// blocks and instructions via the `LoweringContext`.
pub fn lower_statement(
    _ctx: &mut LoweringContext<'_>,
    _stmt: &Statement,
) -> Result<(), LoweringError> {
    Err(LoweringError::UnsupportedExpression {
        message: "stmt_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}

/// Lowers a compound statement (block) — the body of functions and `{ ... }`.
pub fn lower_compound_statement(
    _ctx: &mut LoweringContext<'_>,
    _stmts: &[Statement],
) -> Result<(), LoweringError> {
    Err(LoweringError::UnsupportedExpression {
        message: "stmt_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}

/// Lowers a function body — creates the initial entry block, emits allocas
/// for all local variables, then lowers the body statements.
pub fn lower_function_body(
    _ctx: &mut LoweringContext<'_>,
    _body: &Statement,
) -> Result<(), LoweringError> {
    Err(LoweringError::UnsupportedExpression {
        message: "stmt_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}

/// Helper to lower a function body for use by the mod.rs driver.
/// This variant accepts a `ValueId` for future use by the driver.
#[allow(unused_variables)]
pub fn lower_function_body_with_return(
    ctx: &mut LoweringContext<'_>,
    body: &Statement,
    return_value: Option<ValueId>,
) -> Result<(), LoweringError> {
    lower_function_body(ctx, body)
}
