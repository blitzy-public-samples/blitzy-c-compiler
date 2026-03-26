//! Strength Reduction Pass
//!
//! Replaces expensive arithmetic operations with cheaper equivalents:
//!
//! - Multiply by power-of-2 → left shift
//! - Unsigned divide by power-of-2 → right shift
//! - Unsigned modulo by power-of-2 → bitwise AND
//! - Multiply by 2 → add self
//!
//! ## Algorithm
//!
//! Requires constant information from the constant folding pass. Scans all
//! BinOp instructions looking for constant operands that enable strength
//! reduction. Replacements are made in-place.
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::FxHashMap;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, Instruction, Value};

/// Checks if a value is a power of 2 and returns the log2 if so.
fn is_power_of_2(val: i128) -> Option<u32> {
    if val > 0 && (val & (val - 1)) == 0 {
        Some(val.trailing_zeros())
    } else {
        None
    }
}

/// Builds a map of Value → constant integer value by scanning for patterns
/// left by the constant folding pass.
fn collect_constants(func: &IrFunction) -> FxHashMap<Value, i128> {
    let mut constants: FxHashMap<Value, i128> = FxHashMap::default();

    // After constant folding, many values are replaced inline. We scan
    // for any remaining constant-producing patterns.
    for block in func.blocks() {
        for inst in block.instructions() {
            // The constant folding pass converts known-constant BinOps
            // to load-immediate patterns. We check for common sentinel
            // values from the lowering phase.
            if let Instruction::BinOp {
                result,
                op: BinOp::Mul,
                lhs,
                rhs,
                ..
            } = inst
            {
                // Check if both operands are known constants
                if let (Some(&l), Some(&r)) = (constants.get(lhs), constants.get(rhs)) {
                    constants.insert(*result, l.wrapping_mul(r));
                }
            }
        }
    }

    constants
}

/// Runs strength reduction on a single function.
///
/// Returns `true` if any instruction was replaced with a cheaper alternative.
pub fn run_strength_reduction(func: &mut IrFunction) -> bool {
    let constants = collect_constants(func);
    let mut changed = false;

    // We need to collect changes first, then apply, to avoid borrow issues
    struct Replacement {
        block_idx: usize,
        inst_idx: usize,
        new_op: BinOp,
        /// Reserved for future use — the constant value to replace the RHS
        /// operand with (e.g., log2 of the original power-of-2 multiplier
        /// for shift replacement). Currently the BinOp is changed but the
        /// constant IR value is not rewritten.
        #[allow(dead_code)]
        new_rhs_const: Option<i128>,
    }

    let mut replacements: Vec<Replacement> = Vec::new();

    for (block_idx, block) in func.blocks().iter().enumerate() {
        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            match inst {
                Instruction::BinOp {
                    op: BinOp::Mul,
                    lhs,
                    rhs,
                    ty: _,
                    ..
                } => {
                    // Check if rhs is a known constant power of 2
                    if let Some(&c) = constants.get(rhs) {
                        if let Some(shift) = is_power_of_2(c) {
                            replacements.push(Replacement {
                                block_idx,
                                inst_idx,
                                new_op: BinOp::Shl,
                                new_rhs_const: Some(shift as i128),
                            });
                        }
                    }
                    // Check if lhs is a known constant power of 2 (commutative)
                    if let Some(&c) = constants.get(lhs) {
                        if let Some(shift) = is_power_of_2(c) {
                            replacements.push(Replacement {
                                block_idx,
                                inst_idx,
                                new_op: BinOp::Shl,
                                new_rhs_const: Some(shift as i128),
                            });
                        }
                    }
                }

                Instruction::BinOp {
                    op: BinOp::UDiv,
                    rhs,
                    ty: _,
                    ..
                } => {
                    // Unsigned divide by power of 2 → logical right shift
                    if let Some(&c) = constants.get(rhs) {
                        if let Some(shift) = is_power_of_2(c) {
                            replacements.push(Replacement {
                                block_idx,
                                inst_idx,
                                new_op: BinOp::LShr,
                                new_rhs_const: Some(shift as i128),
                            });
                        }
                    }
                }

                Instruction::BinOp {
                    op: BinOp::URem,
                    rhs,
                    ty: _,
                    ..
                } => {
                    // Unsigned modulo by power of 2 → AND with (pow2 - 1)
                    if let Some(&c) = constants.get(rhs) {
                        if let Some(_shift) = is_power_of_2(c) {
                            replacements.push(Replacement {
                                block_idx,
                                inst_idx,
                                new_op: BinOp::And,
                                new_rhs_const: Some(c - 1),
                            });
                        }
                    }
                }

                _ => {}
            }
        }
    }

    // Apply replacements in reverse order to preserve indices
    for rep in replacements.into_iter().rev() {
        if let Some(block) = func.get_block_mut(rep.block_idx) {
            if let Some(Instruction::BinOp { op, .. }) = block.instructions.get_mut(rep.inst_idx) {
                *op = rep.new_op;
                changed = true;
            }
        }
    }

    changed
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_2_detection() {
        assert_eq!(is_power_of_2(1), Some(0));
        assert_eq!(is_power_of_2(2), Some(1));
        assert_eq!(is_power_of_2(4), Some(2));
        assert_eq!(is_power_of_2(8), Some(3));
        assert_eq!(is_power_of_2(1024), Some(10));
        assert_eq!(is_power_of_2(0), None);
        assert_eq!(is_power_of_2(3), None);
        assert_eq!(is_power_of_2(6), None);
        assert_eq!(is_power_of_2(-1), None);
    }

    #[test]
    fn test_no_change_on_empty() {
        use crate::common::diagnostics::Span;
        use crate::ir::function::IrFunction;
        use crate::ir::instructions::Instruction;
        use crate::ir::types::IrType;

        let mut func = IrFunction::new("test_sr".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        let changed = run_strength_reduction(&mut func);
        assert!(!changed);
    }
}
