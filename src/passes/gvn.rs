//! Global Value Numbering (GVN)
//!
//! Assigns a "value number" to each computation, treating expressions with
//! the same value number as identical. When two instructions compute the
//! same value (same operation, same operand value numbers), the later one
//! is replaced by the earlier one, eliminating redundant computation.
//!
//! GVN subsumes local common subexpression elimination (CSE) and also handles
//! redundancies across basic blocks within the same dominance region.
//!
//! ## Algorithm
//!
//! Uses hash-based value numbering:
//! 1. Walk blocks in dominator-tree preorder (RPO approximation)
//! 2. For each instruction, compute a hash key from (opcode, operand value numbers)
//! 3. If the key exists in the value table, replace the instruction's result
//!    with the existing value
//! 4. Otherwise, add the new mapping to the value table
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::FxHashMap;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

/// A hash key representing an expression for value numbering.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExprKey {
    BinOp {
        op: u8,
        lhs: u32,
        rhs: u32,
    },
    ICmp {
        op: u8,
        lhs: u32,
        rhs: u32,
    },
    FCmp {
        op: u8,
        lhs: u32,
        rhs: u32,
    },
    ZExt {
        value: u32,
        to_bits: u8,
    },
    SExt {
        value: u32,
        to_bits: u8,
    },
    Trunc {
        value: u32,
        to_bits: u8,
    },
    BitCast {
        value: u32,
    },
    /// GEP with base and indices
    Gep {
        base: u32,
        indices: Vec<u32>,
    },
}

/// Encode a BinOp variant as a u8 for hashing.
fn binop_code(op: &BinOp) -> u8 {
    match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::SDiv => 3,
        BinOp::UDiv => 4,
        BinOp::SRem => 5,
        BinOp::URem => 6,
        BinOp::And => 7,
        BinOp::Or => 8,
        BinOp::Xor => 9,
        BinOp::Shl => 10,
        BinOp::AShr => 11,
        BinOp::LShr => 12,
        BinOp::FAdd => 13,
        BinOp::FSub => 14,
        BinOp::FMul => 15,
        BinOp::FDiv => 16,
        BinOp::FRem => 17,
    }
}

fn icmp_code(op: &ICmpOp) -> u8 {
    match op {
        ICmpOp::Eq => 0,
        ICmpOp::Ne => 1,
        ICmpOp::Slt => 2,
        ICmpOp::Sle => 3,
        ICmpOp::Sgt => 4,
        ICmpOp::Sge => 5,
        ICmpOp::Ult => 6,
        ICmpOp::Ule => 7,
        ICmpOp::Ugt => 8,
        ICmpOp::Uge => 9,
    }
}

fn fcmp_code(op: &FCmpOp) -> u8 {
    match op {
        FCmpOp::Oeq => 0,
        FCmpOp::One => 1,
        FCmpOp::Olt => 2,
        FCmpOp::Ole => 3,
        FCmpOp::Ogt => 4,
        FCmpOp::Oge => 5,
        FCmpOp::Uno => 6,
        FCmpOp::Ord => 7,
    }
}

fn type_bits(ty: &IrType) -> u8 {
    match ty {
        IrType::I1 => 1,
        IrType::I8 => 8,
        IrType::I16 => 16,
        IrType::I32 => 32,
        IrType::I64 => 64,
        IrType::I128 => 128,
        IrType::F32 => 32,
        IrType::F64 => 64,
        _ => 0,
    }
}

/// Value number for each SSA Value — maps to a canonical representative.
fn get_vn(vn_map: &FxHashMap<Value, Value>, val: Value) -> u32 {
    // If a value has been renumbered, use its canonical form
    if let Some(&canonical) = vn_map.get(&val) {
        canonical.0
    } else {
        val.0
    }
}

/// Try to build an ExprKey for the instruction.
fn instruction_key(
    inst: &Instruction,
    vn_map: &FxHashMap<Value, Value>,
) -> Option<(Value, ExprKey)> {
    match inst {
        Instruction::BinOp {
            result,
            op,
            lhs,
            rhs,
            ..
        } => {
            let l = get_vn(vn_map, *lhs);
            let r = get_vn(vn_map, *rhs);
            // For commutative ops, normalize operand order
            let (l, r) = match op {
                BinOp::Add
                | BinOp::Mul
                | BinOp::And
                | BinOp::Or
                | BinOp::Xor
                | BinOp::FAdd
                | BinOp::FMul => {
                    if l > r {
                        (r, l)
                    } else {
                        (l, r)
                    }
                }
                _ => (l, r),
            };
            Some((
                *result,
                ExprKey::BinOp {
                    op: binop_code(op),
                    lhs: l,
                    rhs: r,
                },
            ))
        }

        Instruction::ICmp {
            result,
            op,
            lhs,
            rhs,
            ..
        } => Some((
            *result,
            ExprKey::ICmp {
                op: icmp_code(op),
                lhs: get_vn(vn_map, *lhs),
                rhs: get_vn(vn_map, *rhs),
            },
        )),

        Instruction::FCmp {
            result,
            op,
            lhs,
            rhs,
            ..
        } => Some((
            *result,
            ExprKey::FCmp {
                op: fcmp_code(op),
                lhs: get_vn(vn_map, *lhs),
                rhs: get_vn(vn_map, *rhs),
            },
        )),

        Instruction::ZExt {
            result,
            value,
            to_type,
            ..
        } => Some((
            *result,
            ExprKey::ZExt {
                value: get_vn(vn_map, *value),
                to_bits: type_bits(to_type),
            },
        )),

        Instruction::SExt {
            result,
            value,
            to_type,
            ..
        } => Some((
            *result,
            ExprKey::SExt {
                value: get_vn(vn_map, *value),
                to_bits: type_bits(to_type),
            },
        )),

        Instruction::Trunc {
            result,
            value,
            to_type,
            ..
        } => Some((
            *result,
            ExprKey::Trunc {
                value: get_vn(vn_map, *value),
                to_bits: type_bits(to_type),
            },
        )),

        Instruction::BitCast { result, value, .. } => Some((
            *result,
            ExprKey::BitCast {
                value: get_vn(vn_map, *value),
            },
        )),

        Instruction::GetElementPtr {
            result,
            base,
            indices,
            ..
        } => {
            let idx_vns: Vec<u32> = indices.iter().map(|i| get_vn(vn_map, *i)).collect();
            Some((
                *result,
                ExprKey::Gep {
                    base: get_vn(vn_map, *base),
                    indices: idx_vns,
                },
            ))
        }

        _ => None,
    }
}

/// Runs global value numbering on a single function.
///
/// Returns `true` if any redundant computation was eliminated.
pub fn run_gvn(func: &mut IrFunction) -> bool {
    let rpo = func.reverse_postorder();

    // Expression table: maps ExprKey → canonical Value
    let mut expr_table: FxHashMap<ExprKey, Value> = FxHashMap::default();
    // Value number map: maps each Value to its canonical representative
    let mut vn_map: FxHashMap<Value, Value> = FxHashMap::default();

    // Phase 1: Assign value numbers in RPO
    for &block_idx in &rpo {
        if block_idx >= func.blocks().len() {
            continue;
        }
        let block = &func.blocks()[block_idx];
        for inst in block.instructions() {
            if let Some((result, key)) = instruction_key(inst, &vn_map) {
                if let Some(&existing) = expr_table.get(&key) {
                    // This expression already computed — map result to existing value
                    vn_map.insert(result, existing);
                } else {
                    // New expression — record it
                    expr_table.insert(key, result);
                }
            }
        }
    }

    if vn_map.is_empty() {
        return false;
    }

    // Phase 2: Replace all uses of redundant values with canonical values
    let mut changed = false;
    for block in func.blocks_mut() {
        for inst in block.instructions_mut().iter_mut() {
            for operand in inst.operands_mut() {
                if let Some(&canonical) = vn_map.get(operand) {
                    if *operand != canonical {
                        *operand = canonical;
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

    #[test]
    fn test_gvn_eliminates_cse() {
        let mut func = IrFunction::new("test_gvn".to_string(), vec![], IrType::I32);
        let entry = &mut func.blocks[0];

        // %1 = add i32 %0, %0
        entry.instructions.push(Instruction::BinOp {
            result: Value(1),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // %2 = add i32 %0, %0  — same computation as %1
        entry.instructions.push(Instruction::BinOp {
            result: Value(2),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // %3 = add i32 %1, %2  — uses both, but %2 should be replaced by %1
        entry.instructions.push(Instruction::BinOp {
            result: Value(3),
            op: BinOp::Add,
            lhs: Value(1),
            rhs: Value(2),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        entry.instructions.push(Instruction::Return {
            value: Some(Value(3)),
            span: Span::dummy(),
        });

        let changed = run_gvn(&mut func);
        assert!(changed);

        // %2 should be replaced by %1 in the third add
        if let Instruction::BinOp { lhs, rhs, .. } = &func.blocks[0].instructions[2] {
            assert_eq!(*lhs, Value(1));
            assert_eq!(*rhs, Value(1)); // %2 replaced with %1
        }
    }

    #[test]
    fn test_gvn_commutative() {
        let mut func = IrFunction::new("test_gvn2".to_string(), vec![], IrType::I32);
        let entry = &mut func.blocks[0];

        // %2 = add i32 %0, %1
        entry.instructions.push(Instruction::BinOp {
            result: Value(2),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(1),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // %3 = add i32 %1, %0  — same computation (commutative)
        entry.instructions.push(Instruction::BinOp {
            result: Value(3),
            op: BinOp::Add,
            lhs: Value(1),
            rhs: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // %4 = add %2, %3
        entry.instructions.push(Instruction::BinOp {
            result: Value(4),
            op: BinOp::Add,
            lhs: Value(2),
            rhs: Value(3),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        entry.instructions.push(Instruction::Return {
            value: Some(Value(4)),
            span: Span::dummy(),
        });

        let changed = run_gvn(&mut func);
        assert!(changed);
    }

    #[test]
    fn test_gvn_no_change() {
        let mut func = IrFunction::new("test_gvn3".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        let changed = run_gvn(&mut func);
        assert!(!changed);
    }
}
