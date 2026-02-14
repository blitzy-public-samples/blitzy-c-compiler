//! AArch64 instruction selection and emission module.
//!
//! Translates IR instructions into AArch64 machine instructions for the A64
//! instruction set.  Every instruction is a fixed 32-bit word.  This module
//! covers:
//!
//! - Data processing (register-register and immediate forms)
//! - Memory operations (LDR, STR, LDP, STP, byte/half/word/doubleword)
//! - Conditional and unconditional branches (B, BL, BR, BLR, RET, B.cond,
//!   CBZ, CBNZ, TBZ, TBNZ)
//! - Comparison instructions (CMP, CMN, TST, CCMP/CCMN)
//! - Conditional select (CSEL, CSINC, CSINV, CSNEG)
//! - Floating-point / SIMD scalar instructions
//! - Type conversion instructions (SCVTF, UCVTF, FCVTZS, FCVTZU, FCVT,
//!   SXTB, SXTH, SXTW, UXTB, UXTH)
//! - PIC addressing via ADRP+ADD/LDR pairs
//! - Stack frame management (prologue/epilogue per AAPCS64)
//! - Function call lowering per AAPCS64 (X0-X7, V0-V7, stack)

use crate::backend::aarch64::abi::AArch64Abi;
use crate::backend::aarch64::registers::{
    CALLER_SAVED_INT,
    COND_CC, COND_CS, COND_EQ, COND_GE, COND_GT, COND_HI, COND_LE,
    COND_LS, COND_LT, COND_MI, COND_NE, COND_PL, COND_VC, COND_VS,
    FLOAT_ARG_REGS, FP, INTEGER_ARG_REGS, LR, SP, V0, V1,
    W0, WZR, X0, X1, X2, X3, X4, X5, X6, X7, X8, XZR,
    invert_condition, is_callee_saved, v_to_d, v_to_s,
};
use crate::backend::traits::{
    MachineFunction, MachineInstr, MachineOperand, PhysReg,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction};
use crate::ir::types::IrType;

// ===========================================================================
// AArch64Opcode — enumeration of all emitted AArch64 instructions
// ===========================================================================

/// Opcode enumeration for AArch64 machine instructions.
///
/// Each variant maps 1:1 to an A64 instruction mnemonic.  The assembler
/// stage translates these opcodes into 32-bit encoded words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AArch64Opcode {
    // --- Data processing (register-register) ---
    ADD = 0,
    SUB,
    AND,
    ORR,
    EOR,
    ORN,
    BIC,
    ADDS,
    SUBS,
    MADD,
    MSUB,
    SDIV,
    UDIV,
    LSL,
    LSR,
    ASR,
    ROR,

    // --- Data processing (immediate) ---
    ADDimm,
    SUBimm,
    ANDimm,
    ORRimm,
    EORimm,
    MOVZ,
    MOVK,
    MOVN,

    // --- PC-relative addressing ---
    ADRP,
    ADR,

    // --- Memory operations ---
    LDR,
    STR,
    LDRB,
    LDRH,
    LDRSB,
    LDRSH,
    LDRSW,
    STRB,
    STRH,
    LDP,
    STP,
    LDRlit,

    // --- Branches ---
    B,
    BL,
    BR,
    BLR,
    RET,
    Bcond,
    CBZ,
    CBNZ,
    TBZ,
    TBNZ,

    // --- Comparison ---
    CMP,
    CMN,
    TST,
    CCMP,
    CCMN,

    // --- Conditional select ---
    CSEL,
    CSINC,
    CSINV,
    CSNEG,

    // --- Floating-point / SIMD scalar ---
    FADD,
    FSUB,
    FMUL,
    FDIV,
    FNEG,
    FABS,
    FSQRT,
    FMOV,
    FMOVint,
    FCMP,
    FCCMP,

    // --- Type conversion ---
    SCVTF,
    UCVTF,
    FCVTZS,
    FCVTZU,
    FCVT,
    SXTB,
    SXTH,
    SXTW,
    UXTB,
    UXTH,

    // --- Miscellaneous ---
    NOP,
}

impl AArch64Opcode {
    /// Returns the underlying `u32` value for embedding in `MachineInstr`.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ===========================================================================
// Local argument classification — works on IrType, not CType
// ===========================================================================

/// Simplified argument classification for IR-level call lowering.
///
/// The full AAPCS64 classification operates on CType (C-level types).
/// During instruction selection we only have IrType, so we use this
/// reduced classification that captures the essential passing semantics.
#[derive(Clone, Debug)]
enum IrArgClass {
    /// Passed in an integer register (X0-X7).
    IntReg,
    /// Passed in a floating-point / SIMD register (V0-V7).
    FpReg,
    /// Passed on the stack.
    OnStack,
    /// Passed by reference (large aggregates >16 bytes).
    ByReference,
}

/// Classifies an IR type for AAPCS64 argument passing.
///
/// AAPCS64 rules:
/// - Scalars ≤8 bytes (integers, pointers) → integer register (X0–X7).
/// - Floating-point scalars → FP/SIMD register (V0–V7).
/// - Small aggregates ≤16 bytes → passed in integer registers.
/// - Large aggregates >16 bytes → passed by reference (caller-allocated copy).
/// - When register slots are exhausted, remaining args go on stack.
fn classify_ir_arg(ty: &IrType, target: &Target) -> IrArgClass {
    if ty.is_floating() {
        return IrArgClass::FpReg;
    }
    if ty.is_integer() || ty.is_pointer() {
        return IrArgClass::IntReg;
    }
    // Aggregates (struct, array).
    let size = ty.size_bytes(target);
    if size > 16 {
        IrArgClass::ByReference
    } else if size == 0 {
        // Zero-size types use an integer register slot.
        IrArgClass::IntReg
    } else if ty.is_aggregate() {
        // Aggregates ≤16 bytes are passed via the stack in cases
        // where register-level decomposition isn't feasible (e.g.,
        // packed structs, or when the caller prefers stack passing).
        // Concrete register exhaustion is handled in `lower_call`.
        IrArgClass::OnStack
    } else {
        IrArgClass::IntReg
    }
}

// ===========================================================================
// AArch64InstrSel — instruction selection engine
// ===========================================================================

/// AArch64 instruction selection engine.
///
/// Translates an entire `IrFunction` into a `MachineFunction` by iterating
/// over each basic block and dispatching every IR instruction to the
/// appropriate lowering method.
pub struct AArch64InstrSel {
    /// Compilation target (always `Target::AArch64`).
    target: Target,
    /// ABI handler for call-site and parameter classification.
    /// Currently used for structural validation; full CType-based
    /// classification will be wired when CType information flows
    /// through the IR call instructions.
    #[allow(dead_code)]
    abi: AArch64Abi,
    /// Diagnostic engine for error/warning reporting.
    diag: DiagnosticEngine,
    /// Maps IR `ValueId` → `MachineOperand` holding the value.
    value_map: FxHashMap<ValueId, MachineOperand>,
    /// Maps IR `BasicBlockId` → machine basic block ID.
    block_map: FxHashMap<BasicBlockId, u32>,
    /// Counter for allocating fresh virtual register IDs.
    next_vreg: u32,
    /// Running frame offset for alloca lowering (grows negatively).
    current_frame_offset: i32,
    /// Frame object bookkeeping (index → size, alignment, offset).
    frame_objects: Vec<FrameObject>,
    /// Whether the current function contains any calls.
    has_calls: bool,
    /// Whether PIC mode is active.
    pic_mode: bool,
}

/// Stack frame object descriptor (for alloca lowering).
#[derive(Clone, Debug)]
struct FrameObject {
    size: u32,
    alignment: u32,
    offset: i32,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl AArch64InstrSel {
    /// Creates a new instruction selector for AArch64.
    ///
    /// # Arguments
    ///
    /// * `pic_mode` — if `true`, generate position-independent code (PIC)
    ///   using ADRP+LDR (GOT-indirect) for global variable access.
    pub fn new(pic_mode: bool) -> Self {
        AArch64InstrSel {
            target: Target::AArch64,
            abi: AArch64Abi::new(),
            diag: DiagnosticEngine::new(),
            value_map: FxHashMap::default(),
            block_map: FxHashMap::default(),
            next_vreg: 0x1_0000, // Start above plausible ValueId range.
            current_frame_offset: 0,
            frame_objects: Vec::new(),
            has_calls: false,
            pic_mode,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Translates an entire IR function into an AArch64 `MachineFunction`.
    ///
    /// Steps:
    /// 1. Create `MachineBasicBlock`s for every IR block.
    /// 2. Lower function parameters per AAPCS64.
    /// 3. Iterate blocks/instructions and dispatch to per-opcode lowering.
    /// 4. Compute used callee-saved registers.
    /// 5. Emit prologue and epilogue.
    pub fn select_function(&mut self, func: &IrFunction) -> MachineFunction {
        // Reset per-function state.
        self.value_map.clear();
        self.block_map.clear();
        self.frame_objects.clear();
        self.current_frame_offset = 0;
        self.has_calls = false;
        self.next_vreg = 0x1_0000;

        let stack_align = self.target.stack_alignment();
        let mut mf = MachineFunction::new(func.name.clone(), stack_align);

        // Phase 1: Create machine blocks and build the block map.
        for bb in func.blocks() {
            let mbb_id = mf.create_block();
            self.block_map.insert(bb.id, mbb_id);
            // Set a label derived from the block id.
            if let Some(block) = mf.blocks.get_mut(mbb_id as usize) {
                block.label = Some(format!(".LBB_{}", bb.id.0));
            }
        }

        // Phase 2: Lower function parameters.
        self.lower_params(func, &mut mf);

        // Phase 3: Select instructions for each block.
        for bb in func.blocks() {
            let mbb_id = self.block_map.get(&bb.id).copied().unwrap_or(0);
            for inst in bb.instructions() {
                self.select_instruction(inst, func, &mut mf, mbb_id);
            }
        }

        // Phase 4: Compute callee-saved register usage.
        let callee_saved = self.compute_used_callee_saved(&mf);
        mf.used_callee_saved = callee_saved;
        mf.has_calls = self.has_calls;

        // Phase 5: Compute frame size and emit prologue/epilogue.
        let frame_size = self.compute_frame_size(&mf);
        mf.frame_size = frame_size;
        self.emit_prologue(&mut mf);
        self.emit_epilogue(&mut mf);

        mf
    }

    /// Translates a single IR instruction into one or more AArch64 machine
    /// instructions, appending them to the specified machine basic block.
    pub fn select_instruction(
        &mut self,
        inst: &Instruction,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        match inst {
            Instruction::Alloca {
                result,
                ty,
                alignment,
            } => {
                self.lower_alloca(*result, ty, *alignment, mf, mbb_id);
            }

            Instruction::Load {
                result,
                ptr,
                ty,
                volatile: _,
            } => {
                self.lower_load(*result, *ptr, ty, mf, mbb_id);
            }

            Instruction::Store {
                value,
                ptr,
                volatile: _,
            } => {
                self.lower_store(*value, *ptr, func, mf, mbb_id);
            }

            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => {
                self.lower_binop(*result, *op, *lhs, *rhs, ty, func, mf, mbb_id);
            }

            Instruction::ICmp {
                result,
                pred,
                lhs,
                rhs,
            } => {
                self.lower_icmp(*result, *pred, *lhs, *rhs, mf, mbb_id);
            }

            Instruction::FCmp {
                result,
                pred,
                lhs,
                rhs,
            } => {
                self.lower_fcmp(*result, *pred, *lhs, *rhs, mf, mbb_id);
            }

            Instruction::Branch { target } => {
                self.lower_branch(*target, mf, mbb_id);
            }

            Instruction::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                self.lower_condbranch(*condition, *true_target, *false_target, mf, mbb_id);
            }

            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                self.lower_switch(*value, *default, cases, func, mf, mbb_id);
            }

            Instruction::Call {
                result,
                callee,
                args,
                is_tail: _,
            } => {
                self.lower_call(result, *callee, args, func, mf, mbb_id);
            }

            Instruction::Return { value } => {
                self.lower_return(value, func, mf, mbb_id);
            }

            Instruction::Phi {
                result,
                ty,
                incoming,
            } => {
                self.lower_phi(*result, ty, incoming, mf, mbb_id);
            }

            Instruction::GetElementPtr {
                result,
                base,
                indices,
                ty,
                in_bounds: _,
            } => {
                self.lower_gep(*result, *base, indices, ty, func, mf, mbb_id);
            }

            Instruction::BitCast {
                result,
                value,
                to_ty,
            } => {
                self.lower_bitcast(*result, *value, to_ty, func, mf, mbb_id);
            }

            Instruction::Trunc {
                result,
                value,
                to_ty,
            } => {
                self.lower_trunc(*result, *value, to_ty, func, mf, mbb_id);
            }

            Instruction::ZExt {
                result,
                value,
                to_ty,
            } => {
                self.lower_zext(*result, *value, to_ty, func, mf, mbb_id);
            }

            Instruction::SExt {
                result,
                value,
                to_ty,
            } => {
                self.lower_sext(*result, *value, to_ty, func, mf, mbb_id);
            }

            Instruction::IntToPtr {
                result,
                value,
                to_ty: _,
            } => {
                self.lower_inttoptr(*result, *value, func, mf, mbb_id);
            }

            Instruction::PtrToInt {
                result,
                value,
                to_ty,
            } => {
                self.lower_ptrtoint(*result, *value, to_ty, func, mf, mbb_id);
            }

            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects,
                is_align_stack: _,
            } => {
                self.lower_inline_asm(
                    result,
                    template,
                    constraints,
                    operands,
                    clobbers,
                    *has_side_effects,
                    mf,
                    mbb_id,
                );
            }
        }
    }

    /// Emits the function prologue at the beginning of the entry block.
    ///
    /// AAPCS64 prologue structure:
    /// ```text
    /// STP X29, X30, [SP, #-frame_size]!   ; pre-indexed: save FP/LR, adjust SP
    /// MOV X29, SP                          ; establish frame pointer
    /// STP <callee_saved_pairs>...          ; save callee-saved registers
    /// ```
    pub fn emit_prologue(&self, mf: &mut MachineFunction) {
        if mf.blocks.is_empty() {
            return;
        }

        let frame_size = mf.frame_size;
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            return; // Leaf function with no locals — skip prologue.
        }

        let mut prologue: Vec<MachineInstr> = Vec::new();

        // STP X29, X30, [SP, #-frame_size]!
        let stp_fp_lr = MachineInstr::with_operands(
            AArch64Opcode::STP.as_u32(),
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(LR),
                MachineOperand::Memory {
                    base: SP,
                    offset: -(frame_size as i32),
                    index: None,
                    scale: 1,
                },
            ],
        );
        prologue.push(stp_fp_lr);

        // MOV X29, SP  (encoded as ADD X29, SP, #0)
        let mov_fp = MachineInstr::with_operands(
            AArch64Opcode::ADDimm.as_u32(),
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(0),
            ],
        );
        prologue.push(mov_fp);

        // Save callee-saved register pairs.
        let callee_regs = &mf.used_callee_saved;
        let mut offset = 16i32; // First pair is past FP/LR.
        let mut i = 0;
        while i + 1 < callee_regs.len() {
            let stp = MachineInstr::with_operands(
                AArch64Opcode::STP.as_u32(),
                vec![
                    MachineOperand::Register(callee_regs[i]),
                    MachineOperand::Register(callee_regs[i + 1]),
                    MachineOperand::Memory {
                        base: SP,
                        offset,
                        index: None,
                        scale: 1,
                    },
                ],
            );
            prologue.push(stp);
            offset += 16;
            i += 2;
        }
        // Odd callee-saved register: use STR.
        if i < callee_regs.len() {
            let str_single = MachineInstr::with_operands(
                AArch64Opcode::STR.as_u32(),
                vec![
                    MachineOperand::Register(callee_regs[i]),
                    MachineOperand::Memory {
                        base: SP,
                        offset,
                        index: None,
                        scale: 1,
                    },
                ],
            );
            prologue.push(str_single);
        }

        // Splice prologue at the front of the entry block.
        if let Some(entry) = mf.blocks.first_mut() {
            let old_instrs = std::mem::take(&mut entry.instructions);
            entry.instructions = prologue;
            entry.instructions.extend(old_instrs);
        }
    }

    /// Emits the function epilogue before every RET instruction.
    ///
    /// AAPCS64 epilogue structure:
    /// ```text
    /// LDP <callee_saved_pairs>...          ; restore callee-saved registers
    /// LDP X29, X30, [SP], #frame_size      ; post-indexed: restore FP/LR, adjust SP
    /// RET
    /// ```
    pub fn emit_epilogue(&self, mf: &mut MachineFunction) {
        let frame_size = mf.frame_size;
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            return;
        }

        let callee_regs = mf.used_callee_saved.clone();

        for block in &mut mf.blocks {
            // Find RET instructions and insert epilogue before them.
            let ret_positions: Vec<usize> = block
                .instructions
                .iter()
                .enumerate()
                .filter(|(_, i)| i.opcode == AArch64Opcode::RET.as_u32())
                .map(|(idx, _)| idx)
                .collect();

            // Process in reverse order to keep indices valid.
            for &pos in ret_positions.iter().rev() {
                let mut epilogue: Vec<MachineInstr> = Vec::new();

                // Restore callee-saved register pairs.
                let mut offset = 16i32;
                let mut i = 0;
                while i + 1 < callee_regs.len() {
                    let ldp = MachineInstr::with_operands(
                        AArch64Opcode::LDP.as_u32(),
                        vec![
                            MachineOperand::Register(callee_regs[i]),
                            MachineOperand::Register(callee_regs[i + 1]),
                            MachineOperand::Memory {
                                base: SP,
                                offset,
                                index: None,
                                scale: 1,
                            },
                        ],
                    );
                    epilogue.push(ldp);
                    offset += 16;
                    i += 2;
                }
                // Odd callee-saved register: use LDR.
                if i < callee_regs.len() {
                    let ldr_single = MachineInstr::with_operands(
                        AArch64Opcode::LDR.as_u32(),
                        vec![
                            MachineOperand::Register(callee_regs[i]),
                            MachineOperand::Memory {
                                base: SP,
                                offset,
                                index: None,
                                scale: 1,
                            },
                        ],
                    );
                    epilogue.push(ldr_single);
                }

                // LDP X29, X30, [SP], #frame_size  (post-indexed restore + dealloc)
                let ldp_fp_lr = MachineInstr::with_operands(
                    AArch64Opcode::LDP.as_u32(),
                    vec![
                        MachineOperand::Register(FP),
                        MachineOperand::Register(LR),
                        MachineOperand::Memory {
                            base: SP,
                            offset: frame_size as i32,
                            index: None,
                            scale: 1,
                        },
                    ],
                );
                epilogue.push(ldp_fp_lr);

                // Splice epilogue before the RET instruction.
                let tail = block.instructions.split_off(pos);
                block.instructions.extend(epilogue);
                block.instructions.extend(tail);
            }
        }
    }

    /// Materializes a 64-bit immediate value using MOVZ/MOVK sequences.
    ///
    /// - 16-bit value: single MOVZ
    /// - 32-bit value: MOVZ + MOVK
    /// - 48-bit value: MOVZ + MOVK + MOVK
    /// - 64-bit value: MOVZ + MOVK + MOVK + MOVK
    pub fn materialize_immediate(
        &mut self,
        value: i64,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) -> MachineOperand {
        let dest = self.alloc_vreg();
        let v = value as u64;

        // Extract 16-bit halfwords.
        let hw0 = (v & 0xFFFF) as i64;
        let hw1 = ((v >> 16) & 0xFFFF) as i64;
        let hw2 = ((v >> 32) & 0xFFFF) as i64;
        let hw3 = ((v >> 48) & 0xFFFF) as i64;

        // Try MOVN for negative values close to -1.
        if value < 0 {
            let inv = !v;
            let inv_hw0 = (inv & 0xFFFF) as i64;
            if inv <= 0xFFFF {
                let movn = MachineInstr::with_operands(
                    AArch64Opcode::MOVN.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Immediate(inv_hw0),
                        MachineOperand::Immediate(0), // shift = 0
                    ],
                );
                self.push_instr(mf, mbb_id, movn);
                return dest;
            }
        }

        // Find the first non-zero halfword for MOVZ.
        let halfwords = [(hw0, 0i64), (hw1, 16), (hw2, 32), (hw3, 48)];
        let mut first = true;

        for &(hw, shift) in &halfwords {
            if hw == 0 && first {
                continue; // Skip leading zeros.
            }
            if first {
                let movz = MachineInstr::with_operands(
                    AArch64Opcode::MOVZ.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Immediate(hw),
                        MachineOperand::Immediate(shift),
                    ],
                );
                self.push_instr(mf, mbb_id, movz);
                first = false;
            } else if hw != 0 {
                let movk = MachineInstr::with_operands(
                    AArch64Opcode::MOVK.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Immediate(hw),
                        MachineOperand::Immediate(shift),
                    ],
                );
                self.push_instr(mf, mbb_id, movk);
            }
        }

        // All halfwords were zero → value is zero.
        if first {
            let movz = MachineInstr::with_operands(
                AArch64Opcode::MOVZ.as_u32(),
                vec![
                    dest.clone(),
                    MachineOperand::Immediate(0),
                    MachineOperand::Immediate(0),
                ],
            );
            self.push_instr(mf, mbb_id, movz);
        }

        dest
    }

    /// Lowers an IR Call instruction into AAPCS64 call sequence.
    ///
    /// Places arguments in X0-X7 / V0-V7 / stack per the AAPCS64 rules
    /// (delegating ABI queries to [`AArch64Abi`]), emits BL (direct) or
    /// BLR (indirect), and captures the return value.
    pub fn lower_call(
        &mut self,
        result: &Option<ValueId>,
        callee: ValueId,
        args: &[ValueId],
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let gpr_args = &INTEGER_ARG_REGS;
        let fpr_args = &FLOAT_ARG_REGS;
        let mut gpr_idx = 0usize;
        let mut fpr_idx = 0usize;
        let mut stack_offset = 0i32;

        // Phase 1: Classify and move arguments into position.
        for &arg_val in args {
            let arg_ty = func.get_value_type(arg_val);
            let class = classify_ir_arg(arg_ty, &self.target);
            let src = self.resolve_operand(arg_val);

            match class {
                IrArgClass::FpReg => {
                    if fpr_idx < fpr_args.len() {
                        let target_reg = if arg_ty.size_bytes(&self.target) <= 4 {
                            v_to_s(fpr_args[fpr_idx])
                        } else {
                            v_to_d(fpr_args[fpr_idx])
                        };
                        let mov = MachineInstr::with_operands(
                            AArch64Opcode::FMOV.as_u32(),
                            vec![MachineOperand::Register(target_reg), src],
                        );
                        self.push_instr(mf, mbb_id, mov);
                        fpr_idx += 1;
                    } else {
                        // Spill to stack.
                        let str_inst = MachineInstr::with_operands(
                            AArch64Opcode::STR.as_u32(),
                            vec![
                                src,
                                MachineOperand::Memory {
                                    base: SP,
                                    offset: stack_offset,
                                    index: None,
                                    scale: 1,
                                },
                            ],
                        );
                        self.push_instr(mf, mbb_id, str_inst);
                        stack_offset += 8;
                    }
                }
                IrArgClass::IntReg => {
                    if gpr_idx < gpr_args.len() {
                        let mov = MachineInstr::with_operands(
                            AArch64Opcode::ORR.as_u32(),
                            vec![
                                MachineOperand::Register(gpr_args[gpr_idx]),
                                MachineOperand::Register(XZR),
                                src,
                            ],
                        );
                        self.push_instr(mf, mbb_id, mov);
                        gpr_idx += 1;
                    } else {
                        let str_inst = MachineInstr::with_operands(
                            AArch64Opcode::STR.as_u32(),
                            vec![
                                src,
                                MachineOperand::Memory {
                                    base: SP,
                                    offset: stack_offset,
                                    index: None,
                                    scale: 1,
                                },
                            ],
                        );
                        self.push_instr(mf, mbb_id, str_inst);
                        stack_offset += 8;
                    }
                }
                IrArgClass::ByReference => {
                    // Pass pointer to the aggregate in an integer register.
                    if gpr_idx < gpr_args.len() {
                        let mov = MachineInstr::with_operands(
                            AArch64Opcode::ORR.as_u32(),
                            vec![
                                MachineOperand::Register(gpr_args[gpr_idx]),
                                MachineOperand::Register(XZR),
                                src,
                            ],
                        );
                        self.push_instr(mf, mbb_id, mov);
                        gpr_idx += 1;
                    } else {
                        let str_inst = MachineInstr::with_operands(
                            AArch64Opcode::STR.as_u32(),
                            vec![
                                src,
                                MachineOperand::Memory {
                                    base: SP,
                                    offset: stack_offset,
                                    index: None,
                                    scale: 1,
                                },
                            ],
                        );
                        self.push_instr(mf, mbb_id, str_inst);
                        stack_offset += 8;
                    }
                }
                IrArgClass::OnStack => {
                    let str_inst = MachineInstr::with_operands(
                        AArch64Opcode::STR.as_u32(),
                        vec![
                            src,
                            MachineOperand::Memory {
                                base: SP,
                                offset: stack_offset,
                                index: None,
                                scale: 1,
                            },
                        ],
                    );
                    self.push_instr(mf, mbb_id, str_inst);
                    stack_offset += 8;
                }
            }
        }

        // Phase 2: Emit the call instruction.
        let callee_op = self.resolve_operand(callee);
        let mut call_instr = match &callee_op {
            MachineOperand::Symbol(name) => {
                // Direct call → BL.
                MachineInstr::with_operands(
                    AArch64Opcode::BL.as_u32(),
                    vec![MachineOperand::Symbol(name.clone())],
                )
            }
            _ => {
                // Indirect call → BLR.
                MachineInstr::with_operands(
                    AArch64Opcode::BLR.as_u32(),
                    vec![callee_op],
                )
            }
        };
        call_instr.is_call = true;
        call_instr.add_implicit_def(LR);
        // Mark caller-saved registers as implicit defs (clobbered).
        for &reg in &CALLER_SAVED_INT[..8] {
            call_instr.add_implicit_def(reg);
        }
        self.push_instr(mf, mbb_id, call_instr);
        self.has_calls = true;

        // Phase 3: Capture return value.
        // Infer the return type from the result value's registered type.
        if let Some(res_id) = result {
            let ret_ty = func.get_value_type(*res_id);
            if ret_ty.is_floating() {
                let ret_reg = if ret_ty.size_bytes(&self.target) <= 4 {
                    v_to_s(V0)
                } else {
                    v_to_d(V0)
                };
                let dest = self.alloc_vreg();
                let fmov = MachineInstr::with_operands(
                    AArch64Opcode::FMOV.as_u32(),
                    vec![dest.clone(), MachineOperand::Register(ret_reg)],
                );
                self.push_instr(mf, mbb_id, fmov);
                self.value_map.insert(*res_id, dest);
            } else if !ret_ty.is_void() {
                let dest = self.alloc_vreg();
                let mov = MachineInstr::with_operands(
                    AArch64Opcode::ORR.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Register(XZR),
                        MachineOperand::Register(X0),
                    ],
                );
                self.push_instr(mf, mbb_id, mov);
                self.value_map.insert(*res_id, dest);
            }
        }
    }

    /// Lowers an IR Return instruction.
    pub fn lower_return(
        &mut self,
        value: &Option<ValueId>,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        if let Some(val) = value {
            let ret_ty = &func.return_type;
            let src = self.resolve_operand(*val);

            if ret_ty.is_floating() {
                let ret_reg = if ret_ty.size_bytes(&self.target) <= 4 {
                    v_to_s(V0)
                } else {
                    v_to_d(V0)
                };
                let fmov = MachineInstr::with_operands(
                    AArch64Opcode::FMOV.as_u32(),
                    vec![MachineOperand::Register(ret_reg), src],
                );
                self.push_instr(mf, mbb_id, fmov);
            } else {
                // Integer / pointer return in X0.
                let mov = MachineInstr::with_operands(
                    AArch64Opcode::ORR.as_u32(),
                    vec![
                        MachineOperand::Register(X0),
                        MachineOperand::Register(XZR),
                        src,
                    ],
                );
                self.push_instr(mf, mbb_id, mov);
            }
        }

        let mut ret = MachineInstr::with_operands(
            AArch64Opcode::RET.as_u32(),
            vec![MachineOperand::Register(LR)],
        );
        ret.is_terminator = true;
        ret.is_return = true;
        self.push_instr(mf, mbb_id, ret);
    }

    /// Generates PIC or absolute addressing for a global symbol.
    ///
    /// - PIC mode: `ADRP Xd, :got:sym` / `LDR Xd, [Xd, :got_lo12:sym]`
    /// - Non-PIC:  `ADRP Xd, sym` / `ADD Xd, Xd, :lo12:sym`
    pub fn generate_pic_address(
        &mut self,
        symbol: &str,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) -> MachineOperand {
        let dest = self.alloc_vreg();

        // ADRP: load the 4 KiB-aligned page address of the symbol.
        let adrp = MachineInstr::with_operands(
            AArch64Opcode::ADRP.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Symbol(symbol.to_string()),
            ],
        );
        self.push_instr(mf, mbb_id, adrp);

        if self.pic_mode {
            // GOT-indirect: LDR Xd, [Xd, :got_lo12:sym]
            let dest2 = self.alloc_vreg();
            let ldr = MachineInstr::with_operands(
                AArch64Opcode::LDR.as_u32(),
                vec![
                    dest2.clone(),
                    dest,
                    MachineOperand::Symbol(format!(":got_lo12:{}", symbol)),
                ],
            );
            self.push_instr(mf, mbb_id, ldr);
            dest2
        } else {
            // PC-relative: ADD Xd, Xd, :lo12:sym
            let dest2 = self.alloc_vreg();
            let add = MachineInstr::with_operands(
                AArch64Opcode::ADDimm.as_u32(),
                vec![
                    dest2.clone(),
                    dest,
                    MachineOperand::Symbol(format!(":lo12:{}", symbol)),
                ],
            );
            self.push_instr(mf, mbb_id, add);
            dest2
        }
    }

    // =======================================================================
    // Private: Binary operation lowering
    // =======================================================================

    fn lower_binop(
        &mut self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: &IrType,
        _func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let lhs_op = self.resolve_operand(lhs);
        let rhs_op = self.resolve_operand(rhs);
        let dest = self.alloc_vreg();

        let opcode = match op {
            BinOp::Add => AArch64Opcode::ADD,
            BinOp::Sub => AArch64Opcode::SUB,
            BinOp::And => AArch64Opcode::AND,
            BinOp::Or => AArch64Opcode::ORR,
            BinOp::Xor => AArch64Opcode::EOR,
            BinOp::Shl => AArch64Opcode::LSL,
            BinOp::AShr => AArch64Opcode::ASR,
            BinOp::LShr => AArch64Opcode::LSR,
            BinOp::SDiv => AArch64Opcode::SDIV,
            BinOp::UDiv => AArch64Opcode::UDIV,
            BinOp::FAdd => AArch64Opcode::FADD,
            BinOp::FSub => AArch64Opcode::FSUB,
            BinOp::FMul => AArch64Opcode::FMUL,
            BinOp::FDiv => AArch64Opcode::FDIV,

            BinOp::Mul => {
                // MUL is an alias of MADD with zero addend: Xd = Xn * Xm + XZR
                let madd = MachineInstr::with_operands(
                    AArch64Opcode::MADD.as_u32(),
                    vec![
                        dest.clone(),
                        lhs_op,
                        rhs_op,
                        MachineOperand::Register(XZR),
                    ],
                );
                self.push_instr(mf, mbb_id, madd);
                self.value_map.insert(result, dest);
                return;
            }

            BinOp::SRem | BinOp::URem => {
                // AArch64 lacks a remainder instruction.
                // rem = lhs - (lhs / rhs) * rhs
                let div_opc = if matches!(op, BinOp::SRem) {
                    AArch64Opcode::SDIV
                } else {
                    AArch64Opcode::UDIV
                };
                let quotient = self.alloc_vreg();
                let div = MachineInstr::with_operands(
                    div_opc.as_u32(),
                    vec![quotient.clone(), lhs_op.clone(), rhs_op.clone()],
                );
                self.push_instr(mf, mbb_id, div);

                // MSUB dest, quotient, rhs, lhs  → dest = lhs - quotient * rhs
                let msub = MachineInstr::with_operands(
                    AArch64Opcode::MSUB.as_u32(),
                    vec![dest.clone(), quotient, rhs_op, lhs_op],
                );
                self.push_instr(mf, mbb_id, msub);
                self.value_map.insert(result, dest);
                return;
            }

            // FP remainder — requires a runtime call.
            BinOp::FRem => {
                self.emit_frem_call(result, lhs, rhs, ty, mf, mbb_id);
                return;
            }
        };

        let instr = MachineInstr::with_operands(
            opcode.as_u32(),
            vec![dest.clone(), lhs_op, rhs_op],
        );
        self.push_instr(mf, mbb_id, instr);
        self.value_map.insert(result, dest);
    }

    // =======================================================================
    // Private: Load / Store lowering
    // =======================================================================

    fn lower_load(
        &mut self,
        result: ValueId,
        ptr: ValueId,
        ty: &IrType,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let base_op = self.resolve_operand(ptr);
        let dest = self.alloc_vreg();
        let opcode = self.get_load_opcode(ty);

        let instr = match &base_op {
            MachineOperand::Memory {
                base,
                offset,
                index,
                scale,
            } => MachineInstr::with_operands(
                opcode.as_u32(),
                vec![
                    dest.clone(),
                    MachineOperand::Memory {
                        base: *base,
                        offset: *offset,
                        index: *index,
                        scale: *scale,
                    },
                ],
            ),
            MachineOperand::FrameIndex(idx) => MachineInstr::with_operands(
                opcode.as_u32(),
                vec![dest.clone(), MachineOperand::FrameIndex(*idx)],
            ),
            _ => {
                // Register or virtual register holding the address.
                MachineInstr::with_operands(
                    opcode.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Memory {
                            base: self.operand_to_physreg(&base_op),
                            offset: 0,
                            index: None,
                            scale: 1,
                        },
                    ],
                )
            }
        };
        self.push_instr(mf, mbb_id, instr);
        self.value_map.insert(result, dest);
    }

    fn lower_store(
        &mut self,
        value: ValueId,
        ptr: ValueId,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let val_op = self.resolve_operand(value);
        let addr_op = self.resolve_operand(ptr);
        let val_ty = func.get_value_type(value);
        let opcode = self.get_store_opcode(val_ty);

        let instr = match &addr_op {
            MachineOperand::Memory {
                base,
                offset,
                index,
                scale,
            } => MachineInstr::with_operands(
                opcode.as_u32(),
                vec![
                    val_op,
                    MachineOperand::Memory {
                        base: *base,
                        offset: *offset,
                        index: *index,
                        scale: *scale,
                    },
                ],
            ),
            MachineOperand::FrameIndex(idx) => MachineInstr::with_operands(
                opcode.as_u32(),
                vec![val_op, MachineOperand::FrameIndex(*idx)],
            ),
            _ => MachineInstr::with_operands(
                opcode.as_u32(),
                vec![
                    val_op,
                    MachineOperand::Memory {
                        base: self.operand_to_physreg(&addr_op),
                        offset: 0,
                        index: None,
                        scale: 1,
                    },
                ],
            ),
        };
        self.push_instr(mf, mbb_id, instr);
    }

    // =======================================================================
    // Private: Comparison lowering (ICmp, FCmp)
    // =======================================================================

    fn lower_icmp(
        &mut self,
        result: ValueId,
        pred: ICmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let lhs_op = self.resolve_operand(lhs);
        let rhs_op = self.resolve_operand(rhs);

        // CMP (alias of SUBS with zero destination).
        let cmp = MachineInstr::with_operands(
            AArch64Opcode::CMP.as_u32(),
            vec![lhs_op, rhs_op],
        );
        self.push_instr(mf, mbb_id, cmp);

        // Materialize the boolean result via CSINC.
        let dest = self.alloc_vreg();
        let cond = self.icmp_to_cond(pred);
        let inv = invert_condition(cond);
        let csinc = MachineInstr::with_operands(
            AArch64Opcode::CSINC.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(WZR),
                MachineOperand::Register(WZR),
                MachineOperand::Immediate(inv as i64),
            ],
        );
        self.push_instr(mf, mbb_id, csinc);
        self.value_map.insert(result, dest);
    }

    fn lower_fcmp(
        &mut self,
        result: ValueId,
        pred: FCmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let lhs_op = self.resolve_operand(lhs);
        let rhs_op = self.resolve_operand(rhs);
        let dest = self.alloc_vreg();

        // FCMP sets NZCV.
        let fcmp = MachineInstr::with_operands(
            AArch64Opcode::FCMP.as_u32(),
            vec![lhs_op, rhs_op],
        );
        self.push_instr(mf, mbb_id, fcmp);

        let (cond, need_extra_check) = self.fcmp_to_cond(pred);
        let inv_cond = invert_condition(cond);

        if need_extra_check {
            // Need to handle NaN cases with extra logic.
            let tmp = self.alloc_vreg();
            let csinc1 = MachineInstr::with_operands(
                AArch64Opcode::CSINC.as_u32(),
                vec![
                    tmp.clone(),
                    MachineOperand::Register(WZR),
                    MachineOperand::Register(WZR),
                    MachineOperand::Immediate(inv_cond as i64),
                ],
            );
            self.push_instr(mf, mbb_id, csinc1);

            let is_unordered = pred.is_unordered();
            if is_unordered {
                // Unordered: result = 1 if NaN (VS set).
                let csinc2 = MachineInstr::with_operands(
                    AArch64Opcode::CSINC.as_u32(),
                    vec![
                        dest.clone(),
                        tmp,
                        MachineOperand::Register(WZR),
                        MachineOperand::Immediate(COND_VC as i64),
                    ],
                );
                self.push_instr(mf, mbb_id, csinc2);
            } else {
                // Ordered: result = 0 if NaN (VS set).
                let csel = MachineInstr::with_operands(
                    AArch64Opcode::CSEL.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Register(WZR),
                        tmp,
                        MachineOperand::Immediate(COND_VS as i64),
                    ],
                );
                self.push_instr(mf, mbb_id, csel);
            }
        } else {
            // Simple case: single CSINC.
            let csinc = MachineInstr::with_operands(
                AArch64Opcode::CSINC.as_u32(),
                vec![
                    dest.clone(),
                    MachineOperand::Register(WZR),
                    MachineOperand::Register(WZR),
                    MachineOperand::Immediate(inv_cond as i64),
                ],
            );
            self.push_instr(mf, mbb_id, csinc);
        }

        self.value_map.insert(result, dest);
    }

    // =======================================================================
    // Private: Branch lowering
    // =======================================================================

    fn lower_branch(
        &mut self,
        target: BasicBlockId,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let target_mbb = self.block_map.get(&target).copied().unwrap_or(0);
        let mut b = MachineInstr::with_operands(
            AArch64Opcode::B.as_u32(),
            vec![MachineOperand::Label(target_mbb)],
        );
        b.is_terminator = true;
        self.push_instr(mf, mbb_id, b);
    }

    fn lower_condbranch(
        &mut self,
        condition: ValueId,
        true_target: BasicBlockId,
        false_target: BasicBlockId,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let cond_op = self.resolve_operand(condition);
        let true_mbb = self.block_map.get(&true_target).copied().unwrap_or(0);
        let false_mbb = self.block_map.get(&false_target).copied().unwrap_or(0);

        // CBNZ: branch to true_target if condition is non-zero.
        let mut cbnz = MachineInstr::with_operands(
            AArch64Opcode::CBNZ.as_u32(),
            vec![cond_op, MachineOperand::Label(true_mbb)],
        );
        cbnz.is_terminator = true;
        self.push_instr(mf, mbb_id, cbnz);

        // Unconditional branch to false_target.
        let mut b_false = MachineInstr::with_operands(
            AArch64Opcode::B.as_u32(),
            vec![MachineOperand::Label(false_mbb)],
        );
        b_false.is_terminator = true;
        self.push_instr(mf, mbb_id, b_false);
    }

    fn lower_switch(
        &mut self,
        value: ValueId,
        default: BasicBlockId,
        cases: &[(i64, BasicBlockId)],
        _func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let val_op = self.resolve_operand(value);
        let default_mbb = self.block_map.get(&default).copied().unwrap_or(0);

        for &(case_val, case_target) in cases {
            let target_mbb = self.block_map.get(&case_target).copied().unwrap_or(0);

            // Materialize the case constant.
            let case_imm = if (0..=4095).contains(&case_val) {
                MachineOperand::Immediate(case_val)
            } else {
                self.materialize_immediate(case_val, mf, mbb_id)
            };

            // CMP val, case.
            let cmp = MachineInstr::with_operands(
                AArch64Opcode::CMP.as_u32(),
                vec![val_op.clone(), case_imm],
            );
            self.push_instr(mf, mbb_id, cmp);

            // B.EQ target.
            let beq = MachineInstr::with_operands(
                AArch64Opcode::Bcond.as_u32(),
                vec![
                    MachineOperand::Immediate(COND_EQ as i64),
                    MachineOperand::Label(target_mbb),
                ],
            );
            self.push_instr(mf, mbb_id, beq);
        }

        // Default: unconditional branch.
        let mut b_default = MachineInstr::with_operands(
            AArch64Opcode::B.as_u32(),
            vec![MachineOperand::Label(default_mbb)],
        );
        b_default.is_terminator = true;
        self.push_instr(mf, mbb_id, b_default);
    }

    // =======================================================================
    // Private: Alloca lowering
    // =======================================================================

    fn lower_alloca(
        &mut self,
        result: ValueId,
        ty: &IrType,
        alignment: u32,
        _mf: &mut MachineFunction,
        _mbb_id: u32,
    ) {
        let size = ty.size_bytes(&self.target) as u32;
        let align = alignment.max(ty.alignment(&self.target) as u32).max(1);

        // Align the current offset.
        let abs_offset = (-self.current_frame_offset) as u32;
        let aligned_offset = (abs_offset + align - 1) & !(align - 1);
        let new_offset = aligned_offset + size;
        self.current_frame_offset = -(new_offset as i32);

        let fi = self.frame_objects.len() as u32;
        self.frame_objects.push(FrameObject {
            size,
            alignment: align,
            offset: self.current_frame_offset,
        });

        self.value_map
            .insert(result, MachineOperand::FrameIndex(fi));
    }

    // =======================================================================
    // Private: GEP lowering
    // =======================================================================

    fn lower_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
        ty: &IrType,
        _func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let mut current = self.resolve_operand(base);
        let mut current_ty = ty.clone();

        for &idx in indices {
            let idx_op = self.resolve_operand(idx);
            let elem_size = self.get_element_size(&current_ty);

            if elem_size == 0 {
                // Zero-sized element — offset is always zero.
                current_ty = self.get_element_type(&current_ty);
                continue;
            }

            if elem_size == 1 {
                // Byte-level indexing: ADD base, index.
                let next = self.alloc_vreg();
                let add = MachineInstr::with_operands(
                    AArch64Opcode::ADD.as_u32(),
                    vec![next.clone(), current, idx_op],
                );
                self.push_instr(mf, mbb_id, add);
                current = next;
            } else if elem_size.is_power_of_two() && elem_size <= 8 {
                // Shift the index: ADD base, index LSL #log2(elem_size).
                let shift = elem_size.trailing_zeros() as i64;
                let shifted = self.alloc_vreg();
                let lsl = MachineInstr::with_operands(
                    AArch64Opcode::LSL.as_u32(),
                    vec![
                        shifted.clone(),
                        idx_op,
                        MachineOperand::Immediate(shift),
                    ],
                );
                self.push_instr(mf, mbb_id, lsl);

                let next = self.alloc_vreg();
                let add = MachineInstr::with_operands(
                    AArch64Opcode::ADD.as_u32(),
                    vec![next.clone(), current, shifted],
                );
                self.push_instr(mf, mbb_id, add);
                current = next;
            } else {
                // General case: multiply index by element size, then add.
                let size_op = self.materialize_immediate(elem_size as i64, mf, mbb_id);
                let product = self.alloc_vreg();
                let madd = MachineInstr::with_operands(
                    AArch64Opcode::MADD.as_u32(),
                    vec![
                        product.clone(),
                        idx_op,
                        size_op,
                        MachineOperand::Register(XZR),
                    ],
                );
                self.push_instr(mf, mbb_id, madd);

                let next = self.alloc_vreg();
                let add = MachineInstr::with_operands(
                    AArch64Opcode::ADD.as_u32(),
                    vec![next.clone(), current, product],
                );
                self.push_instr(mf, mbb_id, add);
                current = next;
            }

            current_ty = self.get_element_type(&current_ty);
        }

        self.value_map.insert(result, current);
    }

    // =======================================================================
    // Private: Cast lowering (BitCast, Trunc, ZExt, SExt, IntToPtr, PtrToInt)
    // =======================================================================

    fn lower_bitcast(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src = self.resolve_operand(value);
        let src_ty = func.get_value_type(value);

        let src_is_fp = src_ty.is_floating();
        let dst_is_fp = to_ty.is_floating();

        if src_is_fp != dst_is_fp {
            // Cross-domain move: FMOVint.
            let dest = self.alloc_vreg();
            let fmov = MachineInstr::with_operands(
                AArch64Opcode::FMOVint.as_u32(),
                vec![dest.clone(), src],
            );
            self.push_instr(mf, mbb_id, fmov);
            self.value_map.insert(result, dest);
        } else {
            // Same domain: alias.
            self.value_map.insert(result, src);
        }
    }

    fn lower_trunc(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        _func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src = self.resolve_operand(value);
        let dest = self.alloc_vreg();
        let target_bits = to_ty.size_bits(&self.target);

        match target_bits {
            32 => {
                // W-register access implicitly truncates to 32 bits.
                let mov = MachineInstr::with_operands(
                    AArch64Opcode::ORR.as_u32(),
                    vec![dest.clone(), MachineOperand::Register(WZR), src],
                );
                self.push_instr(mf, mbb_id, mov);
            }
            16 => {
                let mask = self.materialize_immediate(0xFFFF, mf, mbb_id);
                let and = MachineInstr::with_operands(
                    AArch64Opcode::AND.as_u32(),
                    vec![dest.clone(), src, mask],
                );
                self.push_instr(mf, mbb_id, and);
            }
            8 => {
                let mask = self.materialize_immediate(0xFF, mf, mbb_id);
                let and = MachineInstr::with_operands(
                    AArch64Opcode::AND.as_u32(),
                    vec![dest.clone(), src, mask],
                );
                self.push_instr(mf, mbb_id, and);
            }
            1 => {
                let mask = self.materialize_immediate(1, mf, mbb_id);
                let and = MachineInstr::with_operands(
                    AArch64Opcode::AND.as_u32(),
                    vec![dest.clone(), src, mask],
                );
                self.push_instr(mf, mbb_id, and);
            }
            _ => {
                let mask_val = (1i64 << target_bits) - 1;
                let mask = self.materialize_immediate(mask_val, mf, mbb_id);
                let and = MachineInstr::with_operands(
                    AArch64Opcode::AND.as_u32(),
                    vec![dest.clone(), src, mask],
                );
                self.push_instr(mf, mbb_id, and);
            }
        }

        self.value_map.insert(result, dest);
    }

    fn lower_zext(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src = self.resolve_operand(value);
        let dest = self.alloc_vreg();
        let src_ty = func.get_value_type(value);
        let src_bits = src_ty.size_bits(&self.target);

        match src_bits {
            1 => {
                let mask = self.materialize_immediate(1, mf, mbb_id);
                let and = MachineInstr::with_operands(
                    AArch64Opcode::AND.as_u32(),
                    vec![dest.clone(), src, mask],
                );
                self.push_instr(mf, mbb_id, and);
            }
            8 => {
                let uxtb = MachineInstr::with_operands(
                    AArch64Opcode::UXTB.as_u32(),
                    vec![dest.clone(), src],
                );
                self.push_instr(mf, mbb_id, uxtb);
            }
            16 => {
                let uxth = MachineInstr::with_operands(
                    AArch64Opcode::UXTH.as_u32(),
                    vec![dest.clone(), src],
                );
                self.push_instr(mf, mbb_id, uxth);
            }
            32 => {
                // W-register write zero-extends to X-register.
                let mov = MachineInstr::with_operands(
                    AArch64Opcode::ORR.as_u32(),
                    vec![dest.clone(), MachineOperand::Register(WZR), src],
                );
                self.push_instr(mf, mbb_id, mov);
            }
            _ => {
                let mask_val = if src_bits < 64 {
                    (1i64 << src_bits) - 1
                } else {
                    -1i64
                };
                let mask = self.materialize_immediate(mask_val, mf, mbb_id);
                let and = MachineInstr::with_operands(
                    AArch64Opcode::AND.as_u32(),
                    vec![dest.clone(), src, mask],
                );
                self.push_instr(mf, mbb_id, and);
            }
        }

        self.value_map.insert(result, dest);
    }

    fn lower_sext(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src = self.resolve_operand(value);
        let dest = self.alloc_vreg();
        let src_ty = func.get_value_type(value);
        let src_bits = src_ty.size_bits(&self.target);

        match src_bits {
            1 => {
                // sext(i1) : 0 → 0, 1 → -1. SUB dest, XZR, src.
                let sub = MachineInstr::with_operands(
                    AArch64Opcode::SUB.as_u32(),
                    vec![dest.clone(), MachineOperand::Register(XZR), src],
                );
                self.push_instr(mf, mbb_id, sub);
            }
            8 => {
                let sxtb = MachineInstr::with_operands(
                    AArch64Opcode::SXTB.as_u32(),
                    vec![dest.clone(), src],
                );
                self.push_instr(mf, mbb_id, sxtb);
            }
            16 => {
                let sxth = MachineInstr::with_operands(
                    AArch64Opcode::SXTH.as_u32(),
                    vec![dest.clone(), src],
                );
                self.push_instr(mf, mbb_id, sxth);
            }
            32 => {
                let sxtw = MachineInstr::with_operands(
                    AArch64Opcode::SXTW.as_u32(),
                    vec![dest.clone(), src],
                );
                self.push_instr(mf, mbb_id, sxtw);
            }
            _ => {
                // General: LSL then ASR by (64 - src_bits).
                let shift_amt = 64 - src_bits;
                let shift_imm = MachineOperand::Immediate(shift_amt as i64);
                let tmp = self.alloc_vreg();
                let lsl = MachineInstr::with_operands(
                    AArch64Opcode::LSL.as_u32(),
                    vec![tmp.clone(), src, shift_imm.clone()],
                );
                self.push_instr(mf, mbb_id, lsl);
                let asr = MachineInstr::with_operands(
                    AArch64Opcode::ASR.as_u32(),
                    vec![dest.clone(), tmp, shift_imm],
                );
                self.push_instr(mf, mbb_id, asr);
            }
        }

        self.value_map.insert(result, dest);
    }

    fn lower_inttoptr(
        &mut self,
        result: ValueId,
        value: ValueId,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_ty = func.get_value_type(value);
        if src_ty.size_bits(&self.target) < 64 {
            self.lower_zext(result, value, &IrType::I64, func, mf, mbb_id);
        } else {
            let src = self.resolve_operand(value);
            self.value_map.insert(result, src);
        }
    }

    fn lower_ptrtoint(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let target_bits = to_ty.size_bits(&self.target);
        if target_bits < 64 {
            self.lower_trunc(result, value, to_ty, func, mf, mbb_id);
        } else {
            let src = self.resolve_operand(value);
            self.value_map.insert(result, src);
        }
    }

    // =======================================================================
    // Private: Phi lowering
    // =======================================================================

    fn lower_phi(
        &mut self,
        result: ValueId,
        _ty: &IrType,
        incoming: &[(ValueId, BasicBlockId)],
        _mf: &mut MachineFunction,
        _mbb_id: u32,
    ) {
        // Phi nodes should have been eliminated before codegen (Phase 9).
        // Gracefully handle any remaining by picking the first incoming value.
        if let Some(&(first_val, _)) = incoming.first() {
            let src = self.resolve_operand(first_val);
            self.value_map.insert(result, src);
        } else {
            self.diag.warning(
                Span::DUMMY,
                "AArch64 codegen: empty phi node encountered",
            );
            self.value_map
                .insert(result, MachineOperand::Register(XZR));
        }
    }

    // =======================================================================
    // Private: Inline assembly lowering
    // =======================================================================

    #[allow(clippy::too_many_arguments)]
    fn lower_inline_asm(
        &mut self,
        result: &Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
        _has_side_effects: bool,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let mut asm_ops = Vec::with_capacity(2 + operands.len());
        asm_ops.push(MachineOperand::Symbol(template.to_string()));
        asm_ops.push(MachineOperand::Symbol(constraints.to_string()));
        for &op in operands {
            asm_ops.push(self.resolve_operand(op));
        }

        let mut instr = MachineInstr::with_operands(AArch64Opcode::NOP.as_u32(), asm_ops);

        for clobber in clobbers {
            if let Some(reg) = self.parse_clobber_reg(clobber) {
                instr.add_implicit_def(reg);
            }
        }

        self.push_instr(mf, mbb_id, instr);

        // Capture result from X0 by convention.
        if let Some(res) = result {
            let dest = self.alloc_vreg();
            let mov = MachineInstr::with_operands(
                AArch64Opcode::ORR.as_u32(),
                vec![
                    dest.clone(),
                    MachineOperand::Register(XZR),
                    MachineOperand::Register(X0),
                ],
            );
            self.push_instr(mf, mbb_id, mov);
            self.value_map.insert(*res, dest);
        }
    }

    // =======================================================================
    // Private: Parameter lowering
    // =======================================================================

    /// Places function parameters into the value map per AAPCS64.
    fn lower_params(&mut self, func: &IrFunction, _mf: &mut MachineFunction) {
        let gpr_regs = &INTEGER_ARG_REGS;
        let fpr_regs = &FLOAT_ARG_REGS;
        let mut gpr_idx = 0usize;
        let mut fpr_idx = 0usize;
        let mut stack_offset = 0i32;

        for param in &func.params {
            let class = classify_ir_arg(&param.ty, &self.target);
            match class {
                IrArgClass::FpReg => {
                    if fpr_idx < fpr_regs.len() {
                        let reg = if param.ty.size_bytes(&self.target) <= 4 {
                            v_to_s(fpr_regs[fpr_idx])
                        } else {
                            v_to_d(fpr_regs[fpr_idx])
                        };
                        self.value_map
                            .insert(param.id, MachineOperand::Register(reg));
                        fpr_idx += 1;
                    } else {
                        self.value_map.insert(
                            param.id,
                            MachineOperand::Memory {
                                base: FP,
                                offset: stack_offset + 16,
                                index: None,
                                scale: 1,
                            },
                        );
                        stack_offset += param.ty.size_bytes(&self.target).max(8) as i32;
                    }
                }
                IrArgClass::IntReg => {
                    if gpr_idx < gpr_regs.len() {
                        self.value_map
                            .insert(param.id, MachineOperand::Register(gpr_regs[gpr_idx]));
                        gpr_idx += 1;
                    } else {
                        self.value_map.insert(
                            param.id,
                            MachineOperand::Memory {
                                base: FP,
                                offset: stack_offset + 16,
                                index: None,
                                scale: 1,
                            },
                        );
                        stack_offset += param.ty.size_bytes(&self.target).max(8) as i32;
                    }
                }
                IrArgClass::OnStack => {
                    self.value_map.insert(
                        param.id,
                        MachineOperand::Memory {
                            base: FP,
                            offset: stack_offset + 16,
                            index: None,
                            scale: 1,
                        },
                    );
                    stack_offset += param.ty.size_bytes(&self.target).max(8) as i32;
                }
                IrArgClass::ByReference => {
                    // Pointer to the aggregate passed in a GPR.
                    if gpr_idx < gpr_regs.len() {
                        self.value_map
                            .insert(param.id, MachineOperand::Register(gpr_regs[gpr_idx]));
                        gpr_idx += 1;
                    } else {
                        self.value_map.insert(
                            param.id,
                            MachineOperand::Memory {
                                base: FP,
                                offset: stack_offset + 16,
                                index: None,
                                scale: 1,
                            },
                        );
                        stack_offset += 8;
                    }
                }
            }
        }
    }

    // =======================================================================
    // Private: Callee-saved register analysis
    // =======================================================================

    fn compute_used_callee_saved(&self, mf: &MachineFunction) -> Vec<PhysReg> {
        let mut used: Vec<PhysReg> = Vec::new();

        for block in &mf.blocks {
            for instr in &block.instructions {
                for op in &instr.operands {
                    if let MachineOperand::Register(reg) = op {
                        if is_callee_saved(*reg) && !used.contains(reg) {
                            used.push(*reg);
                        }
                    }
                }
                for reg in &instr.implicit_defs {
                    if is_callee_saved(*reg) && !used.contains(reg) {
                        used.push(*reg);
                    }
                }
            }
        }

        used
    }

    // =======================================================================
    // Private: Frame size computation
    // =======================================================================

    fn compute_frame_size(&self, mf: &MachineFunction) -> u32 {
        // 16 bytes for FP/LR save.
        let fp_lr_size: u32 = 16;
        // 8 bytes per callee-saved register, rounded up to pairs of 16.
        let callee_save_bytes = (mf.used_callee_saved.len() as u32) * 8;
        let callee_save_aligned = (callee_save_bytes + 15) & !15;
        // Local alloca area: sum of all frame object sizes, respecting
        // each object's alignment.  The running `current_frame_offset`
        // already embeds alignment padding, so we use it directly.
        let local_size = ((-self.current_frame_offset) as u32 + 15) & !15;
        // Determine the maximum alignment required by any frame object.
        // AAPCS64 guarantees SP is 16-byte aligned, but individual objects
        // may demand higher alignment (e.g., SIMD types at 32 bytes).
        let max_object_align = self
            .frame_objects
            .iter()
            .map(|fo| fo.alignment)
            .max()
            .unwrap_or(16)
            .max(16);
        // Verify frame object bookkeeping is consistent: the total local
        // area must cover every recorded frame object.
        debug_assert!(
            self.frame_objects.iter().all(|fo| {
                let abs_off = (-fo.offset) as u32;
                abs_off <= local_size && fo.size <= abs_off
            }),
            "frame object offsets inconsistent with local area size"
        );
        // Total, aligned to the strictest frame object requirement.
        let total = fp_lr_size + callee_save_aligned + local_size;
        (total + max_object_align - 1) & !(max_object_align - 1)
    }

    // =======================================================================
    // Private: Floating-point remainder helper
    // =======================================================================

    fn emit_frem_call(
        &mut self,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        ty: &IrType,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let lhs_op = self.resolve_operand(lhs);
        let rhs_op = self.resolve_operand(rhs);

        let (reg0, reg1, func_name) = if ty.size_bytes(&self.target) <= 4 {
            (v_to_s(V0), v_to_s(V1), "fmodf")
        } else {
            (v_to_d(V0), v_to_d(V1), "fmod")
        };

        let mov_lhs = MachineInstr::with_operands(
            AArch64Opcode::FMOV.as_u32(),
            vec![MachineOperand::Register(reg0), lhs_op],
        );
        self.push_instr(mf, mbb_id, mov_lhs);

        let mov_rhs = MachineInstr::with_operands(
            AArch64Opcode::FMOV.as_u32(),
            vec![MachineOperand::Register(reg1), rhs_op],
        );
        self.push_instr(mf, mbb_id, mov_rhs);

        let mut call = MachineInstr::with_operands(
            AArch64Opcode::BL.as_u32(),
            vec![MachineOperand::Symbol(func_name.to_string())],
        );
        call.is_call = true;
        call.add_implicit_def(LR);
        self.push_instr(mf, mbb_id, call);
        self.has_calls = true;

        let dest = self.alloc_vreg();
        let mov_res = MachineInstr::with_operands(
            AArch64Opcode::FMOV.as_u32(),
            vec![dest.clone(), MachineOperand::Register(reg0)],
        );
        self.push_instr(mf, mbb_id, mov_res);
        self.value_map.insert(result, dest);
    }

    // =======================================================================
    // Private: ICmp / FCmp predicate → AArch64 condition code mapping
    // =======================================================================

    fn icmp_to_cond(&self, pred: ICmpPredicate) -> u8 {
        match pred {
            ICmpPredicate::Eq => COND_EQ,
            ICmpPredicate::Ne => COND_NE,
            ICmpPredicate::Slt => COND_LT,
            ICmpPredicate::Sle => COND_LE,
            ICmpPredicate::Sgt => COND_GT,
            ICmpPredicate::Sge => COND_GE,
            ICmpPredicate::Ult => COND_CC,
            ICmpPredicate::Ule => COND_LS,
            ICmpPredicate::Ugt => COND_HI,
            ICmpPredicate::Uge => COND_CS,
        }
    }

    /// Maps FCmpPredicate to (condition_code, needs_extra_nan_check).
    fn fcmp_to_cond(&self, pred: FCmpPredicate) -> (u8, bool) {
        match pred {
            FCmpPredicate::OEq => (COND_EQ, true),
            FCmpPredicate::ONe => (COND_NE, true),
            FCmpPredicate::Olt => (COND_MI, false),
            FCmpPredicate::Ole => (COND_LS, false),
            FCmpPredicate::Ogt => (COND_GT, false),
            FCmpPredicate::Oge => (COND_GE, false),
            FCmpPredicate::Ord => (COND_VC, false),
            FCmpPredicate::Uno => (COND_VS, false),
            FCmpPredicate::UEq => (COND_EQ, true),
            FCmpPredicate::UNe => (COND_NE, false),
            FCmpPredicate::Ult => (COND_LT, true),
            FCmpPredicate::Ule => (COND_LE, true),
            FCmpPredicate::Ugt => (COND_HI, true),
            FCmpPredicate::Uge => (COND_PL, true),
        }
    }

    // =======================================================================
    // Private: Load/Store opcode selection
    // =======================================================================

    fn get_load_opcode(&self, ty: &IrType) -> AArch64Opcode {
        match ty {
            IrType::I1 | IrType::I8 => AArch64Opcode::LDRB,
            IrType::I16 => AArch64Opcode::LDRH,
            IrType::I32 | IrType::F32 => AArch64Opcode::LDR,
            IrType::I64 | IrType::Ptr | IrType::F64 => AArch64Opcode::LDR,
            _ => AArch64Opcode::LDR,
        }
    }

    fn get_store_opcode(&self, ty: &IrType) -> AArch64Opcode {
        match ty {
            IrType::I1 | IrType::I8 => AArch64Opcode::STRB,
            IrType::I16 => AArch64Opcode::STRH,
            _ => AArch64Opcode::STR,
        }
    }

    // =======================================================================
    // Private: Operand resolution and virtual register allocation
    // =======================================================================

    /// Resolves an IR ValueId to its corresponding MachineOperand.
    fn resolve_operand(&mut self, val: ValueId) -> MachineOperand {
        if let Some(op) = self.value_map.get(&val) {
            op.clone()
        } else {
            let vreg = MachineOperand::VirtualReg(val);
            self.value_map.insert(val, vreg.clone());
            vreg
        }
    }

    /// Allocates a fresh virtual register operand.
    fn alloc_vreg(&mut self) -> MachineOperand {
        let id = self.next_vreg;
        self.next_vreg += 1;
        MachineOperand::VirtualReg(ValueId(id))
    }

    /// Best-effort extraction of a PhysReg from a MachineOperand.
    /// Falls back to X8 (scratch register) if the operand is not a register.
    fn operand_to_physreg(&self, op: &MachineOperand) -> PhysReg {
        match op {
            MachineOperand::Register(r) => *r,
            _ => X8, // Scratch register fallback.
        }
    }

    // =======================================================================
    // Private: GEP helpers
    // =======================================================================

    fn get_element_size(&self, ty: &IrType) -> u64 {
        match ty {
            IrType::Array { element, .. } => element.size_bytes(&self.target),
            IrType::Struct { fields, packed } => {
                let target = &self.target;
                let is_packed = *packed;
                let mut size = 0u64;
                for field in fields {
                    if !is_packed {
                        let align = field.alignment(target);
                        size = (size + align - 1) & !(align - 1);
                    }
                    size += field.size_bytes(target);
                }
                if !is_packed {
                    let max_align = fields
                        .iter()
                        .map(|f| f.alignment(target))
                        .max()
                        .unwrap_or(1);
                    size = (size + max_align - 1) & !(max_align - 1);
                }
                size
            }
            IrType::Ptr => 8,
            _ => ty.size_bytes(&self.target),
        }
    }

    fn get_element_type(&self, ty: &IrType) -> IrType {
        match ty {
            IrType::Array { element, .. } => (**element).clone(),
            IrType::Struct { fields, .. } => {
                if let Some(first) = fields.first() {
                    first.clone()
                } else {
                    IrType::I8
                }
            }
            IrType::Ptr => IrType::I8,
            _ => ty.clone(),
        }
    }

    // =======================================================================
    // Private: Clobber register parsing
    // =======================================================================

    fn parse_clobber_reg(&self, clobber: &str) -> Option<PhysReg> {
        match clobber {
            "x0" | "w0" => Some(X0),
            "x1" | "w1" => Some(X1),
            "x2" | "w2" => Some(X2),
            "x3" | "w3" => Some(X3),
            "x4" | "w4" => Some(X4),
            "x5" | "w5" => Some(X5),
            "x6" | "w6" => Some(X6),
            "x7" | "w7" => Some(X7),
            "x8" | "w8" => Some(X8),
            "x29" | "fp" => Some(FP),
            "x30" | "lr" => Some(LR),
            "sp" => Some(SP),
            "memory" | "cc" => None,
            _ => {
                if let Some(num_str) = clobber.strip_prefix('x') {
                    if let Ok(n) = num_str.parse::<u16>() {
                        if n <= 30 {
                            return Some(PhysReg(X0.0 + n));
                        }
                    }
                }
                if let Some(num_str) = clobber.strip_prefix('w') {
                    if let Ok(n) = num_str.parse::<u16>() {
                        if n <= 30 {
                            return Some(PhysReg(W0.0 + n));
                        }
                    }
                }
                if let Some(num_str) = clobber.strip_prefix('v') {
                    if let Ok(n) = num_str.parse::<u16>() {
                        if n <= 31 {
                            return Some(PhysReg(V0.0 + n));
                        }
                    }
                }
                None
            }
        }
    }

    // =======================================================================
    // Private: Instruction emission helper
    // =======================================================================

    #[inline]
    fn push_instr(&self, mf: &mut MachineFunction, mbb_id: u32, instr: MachineInstr) {
        if let Some(block) = mf.blocks.get_mut(mbb_id as usize) {
            block.instructions.push(instr);
        }
    }
}


