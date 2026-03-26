//! Register Coalescing
//!
//! Reduces unnecessary register-to-register moves at the IR level by
//! merging values that are related through copy-like operations (phi nodes,
//! bitcasts, same-type casts) when their live ranges do not interfere.
//!
//! ## Algorithm
//!
//! 1. Identify "coalescing candidates" — pairs of values connected by
//!    copy-like instructions (phi incoming values, bitcasts, identity casts)
//! 2. Check for interference: two values interfere if they are both live
//!    at the same program point (approximated by checking if one value
//!    is used after the other is defined in the same block)
//! 3. If no interference, merge the two values by replacing all uses of
//!    one with the other
//!
//! ## Scope
//!
//! This is an IR-level coalescing pass that reduces the number of distinct
//! virtual registers, making the backend register allocator's job easier.
//! It complements the copy propagation pass by handling cases where live
//! ranges need to be checked.
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::FxHashMap;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};

/// Represents a coalescing candidate: a pair of values that could be merged.
struct CoalescePair {
    /// Value to replace (the "copy destination").
    dst: Value,
    /// Value to keep (the "copy source").
    src: Value,
}

/// Finds coalescing candidates from phi nodes and copy-like instructions.
fn find_candidates(func: &IrFunction) -> Vec<CoalescePair> {
    let mut candidates = Vec::new();

    for block in func.blocks() {
        for inst in block.instructions() {
            match inst {
                // Phi nodes: each incoming (value, pred) is a coalescing candidate
                Instruction::Phi {
                    result, incoming, ..
                } => {
                    for (val, _) in incoming {
                        if *val != *result {
                            candidates.push(CoalescePair {
                                dst: *result,
                                src: *val,
                            });
                        }
                    }
                }

                // BitCast (same type) is a pure copy
                Instruction::BitCast { result, value, .. } => {
                    candidates.push(CoalescePair {
                        dst: *result,
                        src: *value,
                    });
                }

                _ => {}
            }
        }
    }

    candidates
}

/// Approximates live range interference between two values.
///
/// Two values interfere if both are "alive" at the same program point.
/// This is approximated by checking if one value is used in an instruction
/// where the other is also used or defined.
fn values_interfere(func: &IrFunction, a: Value, b: Value) -> bool {
    // Simple interference check: values interfere if they are both used
    // in the same instruction (suggesting they need distinct registers)
    for block in func.blocks() {
        for inst in block.instructions() {
            let ops = inst.operands();
            let result = inst.result();

            let a_used = ops.contains(&a) || result == Some(a);
            let b_used = ops.contains(&b) || result == Some(b);

            // If both appear in the same instruction (not as src/dst of a copy),
            // they may interfere
            if a_used && b_used {
                // Exception: if this is the very copy we're trying to coalesce
                match inst {
                    Instruction::Phi { result: r, .. } if *r == a || *r == b => continue,
                    Instruction::BitCast { result: r, .. } if *r == a || *r == b => continue,
                    _ => return true,
                }
            }
        }
    }

    false
}

/// Runs register coalescing on a single function.
///
/// Returns `true` if any values were coalesced.
pub fn run_register_coalescing(func: &mut IrFunction) -> bool {
    let candidates = find_candidates(func);
    if candidates.is_empty() {
        return false;
    }

    let mut coalesced: FxHashMap<Value, Value> = FxHashMap::default();
    let mut changed = false;

    for pair in &candidates {
        // Resolve transitive coalescing
        let mut src = pair.src;
        while let Some(&mapped) = coalesced.get(&src) {
            if mapped == src {
                break;
            }
            src = mapped;
        }
        let mut dst = pair.dst;
        while let Some(&mapped) = coalesced.get(&dst) {
            if mapped == dst {
                break;
            }
            dst = mapped;
        }

        if src == dst {
            continue; // Already coalesced
        }

        if !values_interfere(func, src, dst) {
            coalesced.insert(dst, src);
        }
    }

    if coalesced.is_empty() {
        return false;
    }

    // Resolve transitive chains
    let keys: Vec<Value> = coalesced.keys().copied().collect();
    for key in keys {
        let mut val = coalesced[&key];
        let mut steps = 0;
        while let Some(&next) = coalesced.get(&val) {
            if next == val {
                break;
            }
            val = next;
            steps += 1;
            if steps > 1000 {
                break;
            }
        }
        coalesced.insert(key, val);
    }

    // Apply coalescing
    for block in func.blocks_mut() {
        for inst in block.instructions_mut().iter_mut() {
            for operand in inst.operands_mut() {
                if let Some(&replacement) = coalesced.get(operand) {
                    if *operand != replacement {
                        *operand = replacement;
                        changed = true;
                    }
                }
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
    use crate::common::diagnostics::Span;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{Instruction, Value};
    use crate::ir::types::IrType;

    #[test]
    fn test_coalescing_empty() {
        let mut func = IrFunction::new("test_coal".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        let changed = run_register_coalescing(&mut func);
        assert!(!changed);
    }

    #[test]
    fn test_coalescing_bitcast() {
        let mut func = IrFunction::new("test_coal2".to_string(), vec![], IrType::Void);
        let entry = &mut func.blocks[0];

        // %1 = bitcast %0
        entry.instructions.push(Instruction::BitCast {
            result: Value(1),
            value: Value(0),
            to_type: IrType::I32,
            source_unsigned: false,
            span: Span::dummy(),
        });

        // %2 = add %1, %1
        entry.instructions.push(Instruction::BinOp {
            result: Value(2),
            op: crate::ir::instructions::BinOp::Add,
            lhs: Value(1),
            rhs: Value(1),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        entry.instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });

        let changed = run_register_coalescing(&mut func);
        assert!(changed);
        // %1 should be replaced with %0
        if let Instruction::BinOp { lhs, rhs, .. } = &func.blocks[0].instructions[1] {
            assert_eq!(*lhs, Value(0));
            assert_eq!(*rhs, Value(0));
        }
    }
}
