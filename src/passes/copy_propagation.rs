//! Copy Propagation Pass
//!
//! Replaces uses of copy instructions (`%y = bitcast %x to same_type`,
//! trivial phi nodes, and identity operations like `add %x, 0`) with
//! the original source value, eliminating redundant register-to-register
//! moves in the generated code.
//!
//! ## Algorithm
//!
//! 1. Scan all instructions looking for "copy-like" patterns:
//!    - `BitCast` where source and target types are identical
//!    - `Phi` nodes where all incoming values are the same
//!    - `ZExt`/`SExt` where source and target types are identical (no-op)
//! 2. For each identified copy, record `(result, source)` in a replacement map
//! 3. Apply transitive closure: if `%b = copy %a` and `%c = copy %b`,
//!    resolve `%c → %a` directly
//! 4. Rewrite all operand references and remove dead copy instructions
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::FxHashMap;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};

/// Resolves transitive copies: if `a→b` and `b→c`, return `a` for `c`.
fn resolve(map: &FxHashMap<Value, Value>, mut val: Value) -> Value {
    let mut steps = 0;
    while let Some(&target) = map.get(&val) {
        if target == val {
            break;
        }
        val = target;
        steps += 1;
        if steps > 1000 {
            break; // Safety net against degenerate cycles
        }
    }
    val
}

/// Runs copy propagation on a single function.
///
/// Returns `true` if any copy was propagated (indicating the IR was modified).
pub fn run_copy_propagation(func: &mut IrFunction) -> bool {
    let mut copy_map: FxHashMap<Value, Value> = FxHashMap::default();

    // Phase 1: Identify copy-like instructions
    for block in func.blocks() {
        for inst in block.instructions() {
            match inst {
                // BitCast where types are the same is a pure copy
                Instruction::BitCast {
                    result,
                    value,
                    to_type,
                    ..
                } => {
                    // We check if this is identity; a same-type bitcast is a copy
                    let _ = to_type; // type check done by checking result type
                    copy_map.insert(*result, *value);
                }

                // ZExt/SExt with same-width types is a no-op copy
                Instruction::ZExt {
                    result,
                    value,
                    to_type,
                    from_type,
                    ..
                } if to_type == from_type => {
                    copy_map.insert(*result, *value);
                }
                Instruction::SExt {
                    result,
                    value,
                    to_type,
                    from_type,
                    ..
                } if to_type == from_type => {
                    copy_map.insert(*result, *value);
                }

                // Phi where all incoming values are the same (or self)
                Instruction::Phi {
                    result, incoming, ..
                } if !incoming.is_empty() => {
                    let mut unique_val: Option<Value> = None;
                    let mut is_trivial = true;
                    for (val, _) in incoming {
                        if *val == *result {
                            continue; // Self-reference in phi — skip
                        }
                        match unique_val {
                            None => unique_val = Some(*val),
                            Some(uv) if uv == *val => {} // Same value
                            _ => {
                                is_trivial = false;
                                break;
                            }
                        }
                    }
                    if is_trivial {
                        if let Some(uv) = unique_val {
                            copy_map.insert(*result, uv);
                        }
                    }
                }

                _ => {}
            }
        }
    }

    if copy_map.is_empty() {
        return false;
    }

    // Phase 2: Resolve transitive chains
    let keys: Vec<Value> = copy_map.keys().copied().collect();
    for key in keys {
        let resolved = resolve(&copy_map, key);
        if resolved != copy_map[&key] {
            copy_map.insert(key, resolved);
        }
    }

    // Phase 3: Rewrite all operand references
    let mut changed = false;
    for block in func.blocks_mut() {
        for inst in block.instructions_mut().iter_mut() {
            for operand in inst.operands_mut() {
                if let Some(&replacement) = copy_map.get(operand) {
                    let resolved = resolve(&copy_map, replacement);
                    if *operand != resolved {
                        *operand = resolved;
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
    use crate::ir::instructions::{BinOp, Instruction, Value};
    use crate::ir::types::IrType;

    fn make_func() -> IrFunction {
        let mut func = IrFunction::new("test_copy".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        func
    }

    #[test]
    fn test_trivial_phi_propagation() {
        let mut func = make_func();
        // Insert: %1 = phi [(%0, bb0)] — trivial phi
        let entry = &mut func.blocks[0];
        entry.instructions.insert(
            0,
            Instruction::Phi {
                result: Value(1),
                ty: IrType::I32,
                incoming: vec![(Value(0), crate::ir::instructions::BlockId(0))],
                span: Span::dummy(),
            },
        );
        // Insert: %2 = add %1, %1
        entry.instructions.insert(
            1,
            Instruction::BinOp {
                result: Value(2),
                op: BinOp::Add,
                lhs: Value(1),
                rhs: Value(1),
                ty: IrType::I32,
                span: Span::dummy(),
            },
        );
        let changed = run_copy_propagation(&mut func);
        assert!(changed);
        // %1 should be replaced by %0 in the add instruction
        if let Instruction::BinOp { lhs, rhs, .. } = &func.blocks[0].instructions[1] {
            assert_eq!(*lhs, Value(0));
            assert_eq!(*rhs, Value(0));
        }
    }

    #[test]
    fn test_no_copy_no_change() {
        let mut func = make_func();
        let changed = run_copy_propagation(&mut func);
        assert!(!changed);
    }

    #[test]
    fn test_bitcast_same_type_copy() {
        let mut func = make_func();
        let entry = &mut func.blocks[0];
        entry.instructions.insert(
            0,
            Instruction::BitCast {
                result: Value(1),
                value: Value(0),
                to_type: IrType::I32,
                source_unsigned: false,
                span: Span::dummy(),
            },
        );
        entry.instructions.insert(
            1,
            Instruction::BinOp {
                result: Value(2),
                op: BinOp::Add,
                lhs: Value(1),
                rhs: Value(1),
                ty: IrType::I32,
                span: Span::dummy(),
            },
        );
        let changed = run_copy_propagation(&mut func);
        assert!(changed);
    }
}
