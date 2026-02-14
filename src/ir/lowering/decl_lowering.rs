//! Declaration to IR lowering module.
//!
//! Handles global variable initializers, function prologue/epilogue generation,
//! static local variables, thread-local storage, and the `c_type_to_ir_type()`
//! mapping that converts C language types to IR types.
//!
//! **Note:** Full implementation pending — this stub enables compilation of
//! the lowering driver module (`mod.rs`).

use super::{LoweringContext, LoweringError, ModuleLoweringContext};
use crate::frontend::sema::CheckedDeclaration;

/// Lowers a global variable declaration into the IR module.
///
/// Handles static initializers, zero-initialization, extern declarations,
/// and thread-local storage annotations.
pub fn lower_global_declaration(
    _module_ctx: &mut ModuleLoweringContext,
    _decl: &CheckedDeclaration,
) -> Result<(), LoweringError> {
    Err(LoweringError::UnsupportedExpression {
        message: "decl_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}

/// Lowers a local variable declaration within a function body.
///
/// Creates an alloca instruction in the entry block and optionally emits
/// a store for the initializer.
pub fn lower_local_declaration(
    _ctx: &mut LoweringContext<'_>,
    _decl: &CheckedDeclaration,
) -> Result<(), LoweringError> {
    Err(LoweringError::UnsupportedExpression {
        message: "decl_lowering not yet implemented".to_string(),
        span: crate::common::diagnostics::Span::DUMMY,
    })
}
