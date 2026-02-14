//! Expression to IR lowering module.
//!
//! Translates C expression AST nodes into IR instructions via the IrBuilder.
//! This module is the primary value-producing lowering component — every C
//! expression form flows through here.
//!
//! **Note:** Full implementation pending — this stub enables compilation of
//! the lowering driver module (`mod.rs`).

use crate::ir::instructions::ValueId;
use super::{LoweringContext, LoweringError};
use crate::frontend::sema::TypedExpression;

/// Lowers a C expression to an IR rvalue (loads from memory if needed).
///
/// This is the main entry point called from statement lowering for conditions,
/// return values, expression statements, and operand evaluation.
pub fn lower_expression(
    _ctx: &mut LoweringContext<'_>,
    _expr: &TypedExpression,
) -> Result<ValueId, LoweringError> {
    Err(LoweringError::UnsupportedExpression {
        message: "expr_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}

/// Lowers a C expression to an IR lvalue (address/pointer).
///
/// Used for assignment targets, address-of operands, and inline assembly
/// output operands.
pub fn lower_lvalue(
    _ctx: &mut LoweringContext<'_>,
    _expr: &TypedExpression,
) -> Result<ValueId, LoweringError> {
    Err(LoweringError::InvalidLvalue {
        message: "expr_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}
