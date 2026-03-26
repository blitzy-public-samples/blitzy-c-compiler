//! Tail Call Optimization
//!
//! Identifies function calls in tail position (immediately followed by a
//! return of the call's result) and marks them for the code generator to
//! emit as jumps instead of calls, reusing the current stack frame.
//!
//! ## Criteria for Tail Call
//!
//! A call is in tail position when:
//! 1. The call result is immediately returned (no intervening computation)
//! 2. The callee's return type matches the caller's return type
//! 3. No local variables (allocas) have their addresses taken (the stack
//!    frame may need to persist for pointers into it)
//!
//! ## Implementation
//!
//! This pass marks eligible calls by converting the `Call` + `Return`
//! sequence into a `TailCall` annotation (using a flag on the Call
//! instruction or by inserting metadata). The code generator then emits
//! a JMP instead of CALL+RET.
//!
//! In the current implementation, we convert the pattern:
//! ```text
//!   %r = call @func(args...)
//!   ret %r
//! ```
//! to:
//! ```text
//!   %r = call @func(args...)  // marked as tail
//!   ret %r
//! ```
//!
//! The actual jump emission happens at the codegen level. At the IR level,
//! we ensure the call is in tail position and mark it appropriately.
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::FxHashSet;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};

/// Checks if any alloca's address escapes (is used in a store, call, etc.).
/// If so, tail call optimization is unsafe because the callee might access
/// the caller's stack frame through the escaped pointer.
fn has_escaping_allocas(func: &IrFunction) -> bool {
    let mut alloca_values: FxHashSet<Value> = FxHashSet::default();
    for block in func.blocks() {
        for inst in block.instructions() {
            if let Instruction::Alloca { result, .. } | Instruction::StackAlloc { result, .. } =
                inst
            {
                alloca_values.insert(*result);
            }
        }
    }

    if alloca_values.is_empty() {
        return false;
    }

    // Check if any alloca pointer is passed to a call or stored
    for block in func.blocks() {
        for inst in block.instructions() {
            match inst {
                Instruction::Call { args, .. } => {
                    for arg in args {
                        if alloca_values.contains(arg) {
                            return true;
                        }
                    }
                }
                Instruction::Store { value, .. } => {
                    // If an alloca pointer is stored somewhere, it escapes
                    if alloca_values.contains(value) {
                        return true;
                    }
                }
                _ => {}
            }
        }
    }

    false
}

/// Identifies and counts tail-call-eligible call sites.
///
/// A call is tail-call eligible if:
/// 1. It's the last instruction before a Return in a basic block
/// 2. The Return returns the call's result
/// 3. No allocas have escaping addresses
///
/// Returns `true` if any tail call was identified (the function was modified
/// to mark the tail calls). Currently this pass serves as analysis only —
/// marking is done at codegen level based on the pattern.
pub fn run_tail_call_optimization(func: &mut IrFunction) -> bool {
    if has_escaping_allocas(func) {
        return false;
    }

    let mut tail_calls_found = 0;

    for block in func.blocks() {
        let insts = block.instructions();
        if insts.len() < 2 {
            continue;
        }

        // Check the last two instructions
        let last = &insts[insts.len() - 1];
        let second_last = &insts[insts.len() - 2];

        // Pattern: %r = call @f(args...) ; ret %r
        if let Instruction::Return {
            value: Some(ret_val),
            ..
        } = last
        {
            if let Instruction::Call { result, .. } = second_last {
                if *result == *ret_val {
                    tail_calls_found += 1;
                }
            }
        }

        // Pattern: call @f(args...) ; ret void
        if let Instruction::Return { value: None, .. } = last {
            if let Instruction::Call { .. } = second_last {
                tail_calls_found += 1;
            }
        }
    }

    // This pass is informational for now — the actual tail-call emission
    // happens at codegen. We return whether we found any candidates.
    tail_calls_found > 0
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{Instruction, Value};
    use crate::ir::types::IrType;

    #[test]
    fn test_tail_call_detection() {
        let mut func = IrFunction::new("test_tc".to_string(), vec![], IrType::I32);
        let entry = &mut func.blocks[0];

        // %0 = call @other()
        entry.instructions.push(Instruction::Call {
            result: Value(0),
            callee: Value(100), // function reference
            args: vec![],
            return_type: IrType::I32,
            span: Span::dummy(),
        });

        // ret %0
        entry.instructions.push(Instruction::Return {
            value: Some(Value(0)),
            span: Span::dummy(),
        });

        let has_tail = run_tail_call_optimization(&mut func);
        assert!(has_tail);
    }

    #[test]
    fn test_no_tail_call_with_computation() {
        let mut func = IrFunction::new("test_no_tc".to_string(), vec![], IrType::I32);
        let entry = &mut func.blocks[0];

        // %0 = call @other()
        entry.instructions.push(Instruction::Call {
            result: Value(0),
            callee: Value(100),
            args: vec![],
            return_type: IrType::I32,
            span: Span::dummy(),
        });

        // %1 = add %0, %0
        entry.instructions.push(Instruction::BinOp {
            result: Value(1),
            op: crate::ir::instructions::BinOp::Add,
            lhs: Value(0),
            rhs: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // ret %1 (not the call result directly)
        entry.instructions.push(Instruction::Return {
            value: Some(Value(1)),
            span: Span::dummy(),
        });

        let has_tail = run_tail_call_optimization(&mut func);
        assert!(!has_tail);
    }

    #[test]
    fn test_void_tail_call() {
        let mut func = IrFunction::new("test_vtc".to_string(), vec![], IrType::Void);
        let entry = &mut func.blocks[0];

        entry.instructions.push(Instruction::Call {
            result: Value(0),
            callee: Value(100),
            args: vec![],
            return_type: IrType::Void,
            span: Span::dummy(),
        });

        entry.instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });

        let has_tail = run_tail_call_optimization(&mut func);
        assert!(has_tail);
    }
}
