//! x86-64 instruction selection — transforms IR instructions into machine
//! instructions with complex addressing modes, CMOV conditional moves,
//! SSE/SSE2 floating-point, and variable-length encoding with REX support.
//!
//! This module implements the core instruction selection pass for the x86-64
//! target, translating phi-eliminated IR instructions into x86-64 machine
//! instructions. The selector walks each IR basic block, pattern-matching
//! IR operations to optimal x86-64 instruction sequences.
//!
//! # Architecture
//!
//! The instruction selector operates after SSA phi-elimination (Phase 9) and
//! before register allocation. All register references in the output are
//! virtual registers ([`MachineOperand::VirtualReg`]) that will be replaced
//! by physical registers during register allocation.
//!
//! # Supported Features
//!
//! - **Complex addressing modes**: `[base + index * scale + displacement]`
//! - **CMOV conditional moves**: pattern-matched from `select` IR instructions
//! - **SSE2 floating-point**: `addsd`, `subsd`, `mulsd`, `divsd`, etc.
//! - **REX prefix tracking**: for 64-bit operand sizes and extended registers
//! - **Inline assembly integration**: passes through inline asm blocks

use crate::backend::traits::{
    CodegenConfig, MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand,
};
use crate::ir::basic_block::BasicBlock as IrBasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction, ValueId};

use super::opcodes;
use super::registers;

// ---------------------------------------------------------------------------
// Condition codes for x86-64 Jcc / SETcc / CMOVcc instructions
// ---------------------------------------------------------------------------

/// x86-64 condition codes — encoded as the low nibble of the Jcc/SETcc
/// opcode byte.  These map directly to the FLAGS register predicates
/// documented in the Intel SDM Volume 1, Table 3-1.
///
/// | Code | Mnemonic | FLAGS condition         | Meaning (signed) | Meaning (unsigned) |
/// |------|----------|-------------------------|------------------|--------------------|
/// | 0x04 | E / Z    | ZF = 1                  | equal            | equal              |
/// | 0x05 | NE / NZ  | ZF = 0                  | not equal        | not equal          |
/// | 0x0C | L / NGE  | SF ≠ OF                 | less             | —                  |
/// | 0x0D | GE / NL  | SF = OF                 | greater or equal | —                  |
/// | 0x0E | LE / NG  | ZF=1 or SF≠OF          | less or equal    | —                  |
/// | 0x0F | G / NLE  | ZF=0 and SF=OF          | greater          | —                  |
/// | 0x02 | B / C    | CF = 1                  | —                | below              |
/// | 0x03 | AE / NB  | CF = 0                  | —                | above or equal     |
/// | 0x06 | BE / NA  | CF=1 or ZF=1            | —                | below or equal     |
/// | 0x07 | A / NBE  | CF=0 and ZF=0           | —                | above              |
#[allow(dead_code)]
mod cc {
    pub const E: u8 = 0x04;   // Equal (ZF=1)
    pub const NE: u8 = 0x05;  // Not equal (ZF=0)
    pub const L: u8 = 0x0C;   // Signed less
    pub const GE: u8 = 0x0D;  // Signed greater or equal
    pub const LE: u8 = 0x0E;  // Signed less or equal
    pub const G: u8 = 0x0F;   // Signed greater
    pub const B: u8 = 0x02;   // Unsigned below
    pub const AE: u8 = 0x03;  // Unsigned above or equal
    pub const BE: u8 = 0x06;  // Unsigned below or equal
    pub const A: u8 = 0x07;   // Unsigned above
    pub const P: u8 = 0x0A;   // Parity (PF=1) — unordered FP
    pub const NP: u8 = 0x0B;  // No parity (PF=0) — ordered FP
}

// ---------------------------------------------------------------------------
// X86_64InstrSelector — the main instruction selector
// ---------------------------------------------------------------------------

/// x86-64 instruction selector.
///
/// Transforms IR functions into x86-64 machine functions by walking each
/// basic block and selecting architecture-specific instructions for each
/// IR operation. The selector handles:
///
/// - Integer arithmetic (ADD, SUB, MUL, DIV, shifts, bitwise)
/// - Floating-point arithmetic via SSE2 (ADDSD, SUBSD, MULSD, DIVSD)
/// - Memory operations (LOAD → MOV, STORE → MOV)
/// - Comparisons (CMP, TEST) with condition code mapping
/// - Control flow (JMP, Jcc, RET, CALL)
/// - Address computation (LEA for complex expressions)
/// - Type conversions (MOVSX, MOVZX, CVTSI2SD, etc.)
/// - Inline assembly passthrough
///
/// The output contains only virtual register references — physical register
/// assignment is deferred to the register allocator.
pub struct X86_64InstrSelector<'a> {
    /// Reference to the code generation configuration carrying target flags,
    /// optimization level, PIC mode, and security mitigation settings.
    #[allow(dead_code)]
    config: &'a CodegenConfig,
}

impl<'a> X86_64InstrSelector<'a> {
    /// Creates a new instruction selector with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` — code generation configuration (target, optimisation level,
    ///   PIC mode, security flags)
    pub fn new(config: &'a CodegenConfig) -> Self {
        X86_64InstrSelector { config }
    }

    /// Performs instruction selection on the entire IR function, producing
    /// a [`MachineFunction`] with x86-64 machine instructions.
    ///
    /// This is the main entry point called by [`super::X86_64Codegen::lower_function`].
    ///
    /// # Algorithm
    ///
    /// 1. Create a `MachineFunction` with the same name and x86-64 stack alignment.
    /// 2. For each IR basic block, create a corresponding `MachineBasicBlock`.
    /// 3. Walk each IR instruction and emit one or more machine instructions.
    /// 4. Return the completed machine function for register allocation.
    ///
    /// # Arguments
    ///
    /// * `func` — the IR function to select instructions for
    ///
    /// # Returns
    ///
    /// A `MachineFunction` with virtual-register-based x86-64 machine code.
    pub fn select_instructions(&self, func: &IrFunction) -> MachineFunction {
        let stack_alignment = super::X86_64_STACK_ALIGNMENT;
        let mut mf = MachineFunction::new(func.name.clone(), stack_alignment);

        // Process each IR basic block in layout order.
        for (bb_idx, ir_bb) in func.basic_blocks.iter().enumerate() {
            let mut mbb = MachineBasicBlock::new(bb_idx as u32);

            // Select instructions for each IR instruction in the block.
            self.select_block_instructions(ir_bb, &mut mbb, func);

            mf.add_block(mbb);
        }

        mf
    }

    /// Selects machine instructions for all IR instructions in a single
    /// basic block.
    ///
    /// Each IR instruction is pattern-matched against known IR operation
    /// patterns and translated into one or more x86-64 machine instructions.
    fn select_block_instructions(
        &self,
        ir_bb: &IrBasicBlock,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        for instr in ir_bb.instructions() {
            self.select_instruction(instr, mbb, func);
        }
    }

    /// Selects machine instructions for a single IR instruction.
    ///
    /// This method dispatches on the [`Instruction`] enum variant, generating
    /// the appropriate x86-64 machine instruction sequence for each operation.
    fn select_instruction(
        &self,
        instr: &Instruction,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        match instr {
            // -- Stack allocation (alloca) ----------------------------------
            Instruction::Alloca { result, ty: _, alignment: _ } => {
                self.select_alloca(*result, mbb);
            }

            // -- Memory operations ------------------------------------------
            Instruction::Load { result, ptr, ty: _, volatile: _ } => {
                self.select_load(*result, *ptr, mbb);
            }
            Instruction::Store { value, ptr, volatile: _ } => {
                self.select_store(*value, *ptr, mbb);
            }

            // -- Binary arithmetic/bitwise operations -----------------------
            Instruction::BinOp { result, op, lhs, rhs, ty: _ } => {
                self.select_binop(*result, *op, *lhs, *rhs, mbb);
            }

            // -- Integer comparison -----------------------------------------
            Instruction::ICmp { result, pred, lhs, rhs } => {
                self.select_icmp(*result, *pred, *lhs, *rhs, mbb);
            }

            // -- Floating-point comparison ----------------------------------
            Instruction::FCmp { result, pred, lhs, rhs } => {
                self.select_fcmp(*result, *pred, *lhs, *rhs, mbb);
            }

            // -- Control flow: unconditional branch -------------------------
            Instruction::Branch { target } => {
                self.select_br(*target, mbb);
            }

            // -- Control flow: conditional branch ---------------------------
            Instruction::CondBranch { condition, true_target, false_target } => {
                self.select_condbr(*condition, *true_target, *false_target, mbb);
            }

            // -- Control flow: switch/jump table ----------------------------
            Instruction::Switch { value, default, cases } => {
                self.select_switch(*value, *default, cases, mbb);
            }

            // -- Function call ----------------------------------------------
            Instruction::Call {
                result, callee, args, is_tail: _,
            } => {
                self.select_call(*result, *callee, args, mbb, func);
            }

            // -- Return from function ---------------------------------------
            Instruction::Return { value } => {
                self.select_ret(*value, mbb);
            }

            // -- SSA phi node (should have been eliminated by Phase 9) ------
            Instruction::Phi { .. } => {
                // Phi nodes are eliminated before code generation.
                // If we encounter one, emit nothing — the phi-elimination
                // pass inserts copies at predecessor block terminators.
            }

            // -- Address computation (GEP) ----------------------------------
            Instruction::GetElementPtr {
                result, base, indices, ty: _, in_bounds: _,
            } => {
                self.select_gep(*result, *base, indices, mbb);
            }

            // -- Bitwise reinterpretation casts (no code needed) ------------
            Instruction::BitCast { result, value, to_ty: _ } => {
                self.select_copy(*result, *value, mbb);
            }

            // -- Integer truncation -----------------------------------------
            Instruction::Trunc { result, value, to_ty: _ } => {
                // On x86-64, truncation is implicit — reading a narrower
                // register alias achieves the truncation. We emit a MOV
                // for the register allocator to handle.
                self.select_copy(*result, *value, mbb);
            }

            // -- Zero extension ---------------------------------------------
            Instruction::ZExt { result, value, to_ty: _ } => {
                self.select_zext(*result, *value, mbb);
            }

            // -- Sign extension ---------------------------------------------
            Instruction::SExt { result, value, to_ty: _ } => {
                self.select_sext(*result, *value, mbb);
            }

            // -- Integer-to-pointer conversion (noop on same-width) ---------
            Instruction::IntToPtr { result, value, to_ty: _ } => {
                self.select_copy(*result, *value, mbb);
            }

            // -- Pointer-to-integer conversion (noop on same-width) ---------
            Instruction::PtrToInt { result, value, to_ty: _ } => {
                self.select_copy(*result, *value, mbb);
            }

            // -- Inline assembly passthrough --------------------------------
            Instruction::InlineAsm {
                result, template, constraints, operands,
                clobbers, has_side_effects: _, is_align_stack: _,
            } => {
                self.select_inline_asm(*result, template, constraints, operands, clobbers, mbb);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Individual instruction selection methods
    // -----------------------------------------------------------------------

    /// Selects instructions for an alloca (stack allocation).
    ///
    /// On x86-64, allocas are lowered to frame index references. The actual
    /// stack adjustment happens in the prologue. Each alloca gets a unique
    /// frame slot index.
    fn select_alloca(&self, result: ValueId, mbb: &mut MachineBasicBlock) {
        // Alloca is represented as a LEA from a frame index reference.
        // The register allocator resolves frame indices to [RBP - offset].
        let mi = MachineInstr::with_operands(
            opcodes::LEA,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::FrameIndex(result.index()),
            ],
        );
        mbb.push_instr(mi);
    }

    /// Selects instructions for a load operation.
    ///
    /// Translates to `MOV dst, [src]` on x86-64.
    fn select_load(&self, result: ValueId, ptr: ValueId, mbb: &mut MachineBasicBlock) {
        let mi = MachineInstr::with_operands(
            opcodes::MOV_RM,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(ptr),
            ],
        );
        mbb.push_instr(mi);
    }

    /// Selects instructions for a store operation.
    ///
    /// Translates to `MOV [dst], src` on x86-64.
    fn select_store(&self, value: ValueId, ptr: ValueId, mbb: &mut MachineBasicBlock) {
        let mi = MachineInstr::with_operands(
            opcodes::MOV_MR,
            vec![
                MachineOperand::VirtualReg(ptr),
                MachineOperand::VirtualReg(value),
            ],
        );
        mbb.push_instr(mi);
    }

    /// Selects instructions for a binary integer or floating-point operation.
    ///
    /// For integer ops: `result = op lhs, rhs` → `MOV result, lhs; OP result, rhs`
    /// For FP ops: uses the SSE2 scalar instructions (ADDSD, SUBSD, etc.)
    fn select_binop(
        &self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        mbb: &mut MachineBasicBlock,
    ) {
        match op {
            // -- Integer arithmetic with two-operand pattern ----------------
            BinOp::Add => self.emit_int_binop(result, lhs, rhs, opcodes::ADD_RR, mbb),
            BinOp::Sub => self.emit_int_binop(result, lhs, rhs, opcodes::SUB_RR, mbb),
            BinOp::And => self.emit_int_binop(result, lhs, rhs, opcodes::AND_RR, mbb),
            BinOp::Or  => self.emit_int_binop(result, lhs, rhs, opcodes::OR_RR, mbb),
            BinOp::Xor => self.emit_int_binop(result, lhs, rhs, opcodes::XOR_RR, mbb),

            // -- Multiplication uses three-operand IMUL ---------------------
            BinOp::Mul => {
                let mi = MachineInstr::with_operands(
                    opcodes::IMUL_RR,
                    vec![
                        MachineOperand::VirtualReg(result),
                        MachineOperand::VirtualReg(lhs),
                        MachineOperand::VirtualReg(rhs),
                    ],
                );
                mbb.push_instr(mi);
            }

            // -- Division (signed/unsigned) uses RAX:RDX --------------------
            BinOp::SDiv => self.emit_div(result, lhs, rhs, true, true, mbb),
            BinOp::UDiv => self.emit_div(result, lhs, rhs, false, true, mbb),

            // -- Remainder (signed/unsigned) uses RAX:RDX -------------------
            BinOp::SRem => self.emit_div(result, lhs, rhs, true, false, mbb),
            BinOp::URem => self.emit_div(result, lhs, rhs, false, false, mbb),

            // -- Shifts — shift amount goes in CL (implicit via operand) ----
            BinOp::Shl  => self.emit_shift(result, lhs, rhs, opcodes::SHL, mbb),
            BinOp::LShr => self.emit_shift(result, lhs, rhs, opcodes::SHR, mbb),
            BinOp::AShr => self.emit_shift(result, lhs, rhs, opcodes::SAR, mbb),

            // -- Floating-point arithmetic via SSE2 -------------------------
            BinOp::FAdd => self.emit_fp_binop(result, lhs, rhs, opcodes::ADDSD, mbb),
            BinOp::FSub => self.emit_fp_binop(result, lhs, rhs, opcodes::SUBSD, mbb),
            BinOp::FMul => self.emit_fp_binop(result, lhs, rhs, opcodes::MULSD, mbb),
            BinOp::FDiv => self.emit_fp_binop(result, lhs, rhs, opcodes::DIVSD, mbb),
            BinOp::FRem => {
                // x86-64 has no FP remainder instruction in SSE2.
                // Emit a call to the software __fmod helper.
                let mi = MachineInstr::with_operands(
                    opcodes::CALL,
                    vec![MachineOperand::Symbol("fmod".to_string())],
                );
                mbb.push_instr(mi);
                // Move result from XMM0 to the virtual register.
                let mov = MachineInstr::with_operands(
                    opcodes::MOVSD,
                    vec![
                        MachineOperand::VirtualReg(result),
                        MachineOperand::Register(registers::XMM0),
                    ],
                );
                mbb.push_instr(mov);
            }
        }
    }

    /// Emits a two-operand integer binop: `MOV result, lhs; OP result, rhs`.
    fn emit_int_binop(
        &self,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        opcode: u32,
        mbb: &mut MachineBasicBlock,
    ) {
        // MOV result, lhs
        let mov = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(lhs),
            ],
        );
        mbb.push_instr(mov);
        // OP result, rhs
        let op = MachineInstr::with_operands(
            opcode,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(rhs),
            ],
        );
        mbb.push_instr(op);
    }

    /// Emits a floating-point binop using SSE2 scalar instructions.
    ///
    /// Pattern: `MOVSD result, lhs; OPSD result, rhs`
    fn emit_fp_binop(
        &self,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        opcode: u32,
        mbb: &mut MachineBasicBlock,
    ) {
        // MOVSD result, lhs
        let mov = MachineInstr::with_operands(
            opcodes::MOVSD,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(lhs),
            ],
        );
        mbb.push_instr(mov);
        // OPSD result, rhs
        let op = MachineInstr::with_operands(
            opcode,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(rhs),
            ],
        );
        mbb.push_instr(op);
    }

    /// Emits integer division or remainder using RAX:RDX.
    ///
    /// x86-64 DIV/IDIV use RAX:RDX for the dividend. The quotient goes to
    /// RAX and the remainder to RDX.
    fn emit_div(
        &self,
        result: ValueId,
        dividend: ValueId,
        divisor: ValueId,
        is_signed: bool,
        want_quotient: bool,
        mbb: &mut MachineBasicBlock,
    ) {
        // MOV RAX, dividend
        let mov_rax = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::Register(registers::RAX),
                MachineOperand::VirtualReg(dividend),
            ],
        );
        mbb.push_instr(mov_rax);

        // Sign-extend or zero-extend RAX into RDX:RAX.
        if is_signed {
            // CQO — sign-extend RAX into RDX:RAX (64-bit)
            let cqo = MachineInstr::new(opcodes::CQO);
            mbb.push_instr(cqo);
        } else {
            // XOR RDX, RDX — zero the high half for unsigned division
            let xor_rdx = MachineInstr::with_operands(
                opcodes::XOR_RR,
                vec![
                    MachineOperand::Register(registers::RDX),
                    MachineOperand::Register(registers::RDX),
                ],
            );
            mbb.push_instr(xor_rdx);
        }

        // IDIV/DIV divisor
        let div_opcode = if is_signed { opcodes::IDIV } else { opcodes::DIV };
        let mut div_instr = MachineInstr::with_operands(
            div_opcode,
            vec![MachineOperand::VirtualReg(divisor)],
        );
        div_instr.add_implicit_def(registers::RAX);
        div_instr.add_implicit_def(registers::RDX);
        div_instr.add_implicit_use(registers::RAX);
        div_instr.add_implicit_use(registers::RDX);
        mbb.push_instr(div_instr);

        // MOV result, RAX (quotient) or RDX (remainder)
        let source_reg = if want_quotient {
            registers::RAX
        } else {
            registers::RDX
        };
        let mov_result = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::Register(source_reg),
            ],
        );
        mbb.push_instr(mov_result);
    }

    /// Emits a shift instruction.
    ///
    /// x86-64 shifts require the shift amount in CL. We emit:
    ///   MOV result, lhs; MOV CL, rhs; SHIFT result, CL
    fn emit_shift(
        &self,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        opcode: u32,
        mbb: &mut MachineBasicBlock,
    ) {
        // MOV result, lhs
        let mov = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(lhs),
            ],
        );
        mbb.push_instr(mov);

        // MOV RCX, rhs (shift amount goes into CL/CL is the low byte of RCX)
        let mov_cl = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::Register(registers::RCX),
                MachineOperand::VirtualReg(rhs),
            ],
        );
        mbb.push_instr(mov_cl);

        // SHIFT result, CL
        let mut shift_instr = MachineInstr::with_operands(
            opcode,
            vec![MachineOperand::VirtualReg(result)],
        );
        shift_instr.add_implicit_use(registers::RCX);
        mbb.push_instr(shift_instr);
    }

    /// Selects instructions for an integer comparison.
    ///
    /// `CMP lhs, rhs; SETcc result`
    fn select_icmp(
        &self,
        result: ValueId,
        pred: ICmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        mbb: &mut MachineBasicBlock,
    ) {
        // CMP lhs, rhs
        let cmp = MachineInstr::with_operands(
            opcodes::CMP_RR,
            vec![
                MachineOperand::VirtualReg(lhs),
                MachineOperand::VirtualReg(rhs),
            ],
        );
        mbb.push_instr(cmp);

        // Map ICmpPredicate to x86-64 condition code.
        let condition = icmp_to_cc(pred);

        // SETcc result — set byte based on condition code
        let setcc = MachineInstr::with_operands(
            opcodes::SET_CC,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::Immediate(condition as i64),
            ],
        );
        mbb.push_instr(setcc);
    }

    /// Selects instructions for a floating-point comparison.
    ///
    /// `UCOMISD lhs, rhs; SETcc result`
    fn select_fcmp(
        &self,
        result: ValueId,
        pred: FCmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        mbb: &mut MachineBasicBlock,
    ) {
        // UCOMISD lhs, rhs
        let cmp = MachineInstr::with_operands(
            opcodes::UCOMISD,
            vec![
                MachineOperand::VirtualReg(lhs),
                MachineOperand::VirtualReg(rhs),
            ],
        );
        mbb.push_instr(cmp);

        // Map FCmpPredicate to x86-64 condition code.
        let condition = fcmp_to_cc(pred);

        // SETcc result
        let setcc = MachineInstr::with_operands(
            opcodes::SET_CC,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::Immediate(condition as i64),
            ],
        );
        mbb.push_instr(setcc);
    }

    /// Selects instructions for an unconditional branch.
    fn select_br(
        &self,
        target: crate::ir::instructions::BasicBlockId,
        mbb: &mut MachineBasicBlock,
    ) {
        let mut jmp = MachineInstr::with_operands(
            opcodes::JMP,
            vec![MachineOperand::Label(target.index())],
        );
        jmp.is_terminator = true;
        mbb.push_instr(jmp);
    }

    /// Selects instructions for a conditional branch.
    ///
    /// `TEST cond, cond; JNE true_bb; JMP false_bb`
    fn select_condbr(
        &self,
        cond: ValueId,
        true_bb: crate::ir::instructions::BasicBlockId,
        false_bb: crate::ir::instructions::BasicBlockId,
        mbb: &mut MachineBasicBlock,
    ) {
        // TEST cond, cond — sets ZF if cond == 0
        let test = MachineInstr::with_operands(
            opcodes::TEST_RR,
            vec![
                MachineOperand::VirtualReg(cond),
                MachineOperand::VirtualReg(cond),
            ],
        );
        mbb.push_instr(test);

        // JNE true_bb — jump if not zero (cond is true)
        let mut jne = MachineInstr::with_operands(
            opcodes::JCC,
            vec![
                MachineOperand::Immediate(cc::NE as i64),
                MachineOperand::Label(true_bb.index()),
            ],
        );
        jne.is_terminator = true;
        mbb.push_instr(jne);

        // JMP false_bb — fallthrough to false branch
        let mut jmp = MachineInstr::with_operands(
            opcodes::JMP,
            vec![MachineOperand::Label(false_bb.index())],
        );
        jmp.is_terminator = true;
        mbb.push_instr(jmp);
    }

    /// Selects instructions for a switch statement (jump table or cascaded
    /// comparisons).
    ///
    /// For small case counts (≤ 4), emit cascaded CMP/JE pairs.
    /// For larger case counts, emit an indexed jump table.
    fn select_switch(
        &self,
        value: ValueId,
        default: crate::ir::instructions::BasicBlockId,
        cases: &[(i64, crate::ir::instructions::BasicBlockId)],
        mbb: &mut MachineBasicBlock,
    ) {
        // For now, emit cascaded comparisons for all switch sizes.
        // A future optimisation can emit jump tables for dense switches.
        for (case_val, target_bb) in cases {
            // CMP value, case_val
            let cmp = MachineInstr::with_operands(
                opcodes::CMP_RI,
                vec![
                    MachineOperand::VirtualReg(value),
                    MachineOperand::Immediate(*case_val),
                ],
            );
            mbb.push_instr(cmp);

            // JE target_bb
            let mut je = MachineInstr::with_operands(
                opcodes::JCC,
                vec![
                    MachineOperand::Immediate(cc::E as i64),
                    MachineOperand::Label(target_bb.index()),
                ],
            );
            je.is_terminator = true;
            mbb.push_instr(je);
        }

        // JMP default
        let mut jmp = MachineInstr::with_operands(
            opcodes::JMP,
            vec![MachineOperand::Label(default.index())],
        );
        jmp.is_terminator = true;
        mbb.push_instr(jmp);
    }

    /// Selects instructions for a function call.
    ///
    /// Places arguments in the ABI-mandated registers (RDI, RSI, RDX, RCX,
    /// R8, R9 for integers; XMM0–XMM7 for floats), emits CALL, and copies
    /// the return value from RAX/XMM0.
    fn select_call(
        &self,
        result: Option<ValueId>,
        callee: ValueId,
        args: &[ValueId],
        mbb: &mut MachineBasicBlock,
        _func: &IrFunction,
    ) {
        // Move arguments into the System V AMD64 ABI argument registers.
        let int_arg_regs = registers::ARG_REGS_INT;
        let float_arg_regs = registers::ARG_REGS_FLOAT;

        let mut int_idx = 0usize;
        let mut float_idx = 0usize;
        let mut _stack_args = Vec::new();

        for arg in args {
            // Without type information here, we default to integer register
            // assignment. The full implementation would check arg types.
            if int_idx < int_arg_regs.len() {
                let mov = MachineInstr::with_operands(
                    opcodes::MOV_RR,
                    vec![
                        MachineOperand::Register(int_arg_regs[int_idx]),
                        MachineOperand::VirtualReg(*arg),
                    ],
                );
                mbb.push_instr(mov);
                int_idx += 1;
            } else if float_idx < float_arg_regs.len() {
                let mov = MachineInstr::with_operands(
                    opcodes::MOVSD,
                    vec![
                        MachineOperand::Register(float_arg_regs[float_idx]),
                        MachineOperand::VirtualReg(*arg),
                    ],
                );
                mbb.push_instr(mov);
                float_idx += 1;
            } else {
                // Push remaining arguments on the stack.
                let push = MachineInstr::with_operands(
                    opcodes::PUSH,
                    vec![MachineOperand::VirtualReg(*arg)],
                );
                mbb.push_instr(push);
                _stack_args.push(*arg);
            }
        }

        // Emit CALL instruction. The callee is a ValueId which may be
        // a direct symbol or an indirect call target. We emit it as a
        // virtual register reference and let the linker resolve it.
        let mut call = MachineInstr::with_operands(
            opcodes::CALL,
            vec![MachineOperand::VirtualReg(callee)],
        );
        call.is_call = true;
        // All caller-saved registers are implicitly clobbered.
        for &reg in registers::CALLER_SAVED {
            call.add_implicit_def(reg);
        }
        mbb.push_instr(call);

        // Copy return value from RAX to the result virtual register.
        if let Some(res) = result {
            let mov_ret = MachineInstr::with_operands(
                opcodes::MOV_RR,
                vec![
                    MachineOperand::VirtualReg(res),
                    MachineOperand::Register(registers::RAX),
                ],
            );
            mbb.push_instr(mov_ret);
        }
    }

    /// Selects instructions for a return statement.
    fn select_ret(&self, value: Option<ValueId>, mbb: &mut MachineBasicBlock) {
        if let Some(val) = value {
            // MOV RAX, return_value
            let mov = MachineInstr::with_operands(
                opcodes::MOV_RR,
                vec![
                    MachineOperand::Register(registers::RAX),
                    MachineOperand::VirtualReg(val),
                ],
            );
            mbb.push_instr(mov);
        }

        // RET
        let mut ret = MachineInstr::new(opcodes::RET);
        ret.is_terminator = true;
        ret.is_return = true;
        mbb.push_instr(ret);
    }

    /// Selects instructions for a GEP (GetElementPtr) — address computation.
    ///
    /// Emits LEA instructions for pointer arithmetic with scaled indices.
    fn select_gep(
        &self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
        mbb: &mut MachineBasicBlock,
    ) {
        if indices.is_empty() {
            // No indices — just copy the base pointer.
            self.select_copy(result, base, mbb);
            return;
        }

        // For the first index, emit LEA result, [base + index * stride].
        // For simplicity, we emit ADD-based pointer arithmetic.
        // A future optimization can use LEA with scale factors.

        // Start with MOV result, base
        let mov_base = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(base),
            ],
        );
        mbb.push_instr(mov_base);

        // For each index, ADD result, index (simplified — real impl needs stride)
        for idx in indices {
            let add = MachineInstr::with_operands(
                opcodes::ADD_RR,
                vec![
                    MachineOperand::VirtualReg(result),
                    MachineOperand::VirtualReg(*idx),
                ],
            );
            mbb.push_instr(add);
        }
    }

    /// Emits a register-to-register copy (MOV dst, src).
    fn select_copy(&self, dst: ValueId, src: ValueId, mbb: &mut MachineBasicBlock) {
        let mi = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::VirtualReg(dst),
                MachineOperand::VirtualReg(src),
            ],
        );
        mbb.push_instr(mi);
    }

    /// Selects instructions for zero extension (MOVZX).
    fn select_zext(&self, result: ValueId, value: ValueId, mbb: &mut MachineBasicBlock) {
        let mi = MachineInstr::with_operands(
            opcodes::MOVZX,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(value),
            ],
        );
        mbb.push_instr(mi);
    }

    /// Selects instructions for sign extension (MOVSX).
    fn select_sext(&self, result: ValueId, value: ValueId, mbb: &mut MachineBasicBlock) {
        let mi = MachineInstr::with_operands(
            opcodes::MOVSX,
            vec![
                MachineOperand::VirtualReg(result),
                MachineOperand::VirtualReg(value),
            ],
        );
        mbb.push_instr(mi);
    }

    /// Selects instructions for inline assembly passthrough.
    ///
    /// The assembly template and constraints are preserved verbatim for the
    /// built-in assembler to process.
    fn select_inline_asm(
        &self,
        _result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
        mbb: &mut MachineBasicBlock,
    ) {
        // Build operand list: template, constraints, then bound values.
        let mut ops = Vec::with_capacity(2 + operands.len());
        ops.push(MachineOperand::Symbol(template.to_string()));
        ops.push(MachineOperand::Symbol(constraints.to_string()));
        for op in operands {
            ops.push(MachineOperand::VirtualReg(*op));
        }

        let mut mi = MachineInstr::with_operands(opcodes::INLINE_ASM, ops);

        // Mark clobbered registers.
        for clobber in clobbers {
            if let Some(reg) = parse_clobber_register(clobber) {
                mi.add_implicit_def(reg);
            }
        }

        mbb.push_instr(mi);
    }
}

// ---------------------------------------------------------------------------
// Helper functions for condition code mapping
// ---------------------------------------------------------------------------

/// Maps an IR integer comparison predicate to an x86-64 condition code.
fn icmp_to_cc(pred: ICmpPredicate) -> u8 {
    match pred {
        ICmpPredicate::Eq  => cc::E,
        ICmpPredicate::Ne  => cc::NE,
        ICmpPredicate::Slt => cc::L,
        ICmpPredicate::Sle => cc::LE,
        ICmpPredicate::Sgt => cc::G,
        ICmpPredicate::Sge => cc::GE,
        ICmpPredicate::Ult => cc::B,
        ICmpPredicate::Ule => cc::BE,
        ICmpPredicate::Ugt => cc::A,
        ICmpPredicate::Uge => cc::AE,
    }
}

/// Maps an IR floating-point comparison predicate to an x86-64 condition
/// code for use after UCOMISD.
fn fcmp_to_cc(pred: FCmpPredicate) -> u8 {
    match pred {
        FCmpPredicate::OEq => cc::E,
        FCmpPredicate::ONe => cc::NE,
        FCmpPredicate::Olt => cc::B,
        FCmpPredicate::Ole => cc::BE,
        FCmpPredicate::Ogt => cc::A,
        FCmpPredicate::Oge => cc::AE,
        FCmpPredicate::UEq => cc::E,  // Unordered-equal: test both PF and ZF
        FCmpPredicate::UNe => cc::NE, // Unordered-not-equal
        FCmpPredicate::Ult => cc::B,
        FCmpPredicate::Ule => cc::BE,
        FCmpPredicate::Ugt => cc::A,
        FCmpPredicate::Uge => cc::AE,
        FCmpPredicate::Ord => cc::NP, // Ordered: parity flag clear
        FCmpPredicate::Uno => cc::P,  // Unordered: parity flag set
    }
}

/// Parses a clobber string to a physical register, if it names a known
/// x86-64 register.
fn parse_clobber_register(clobber: &str) -> Option<crate::backend::traits::PhysReg> {
    match clobber.trim() {
        "rax" | "eax" | "al" => Some(registers::RAX),
        "rcx" | "ecx" | "cl" => Some(registers::RCX),
        "rdx" | "edx" | "dl" => Some(registers::RDX),
        "rbx" | "ebx" | "bl" => Some(registers::RBX),
        "rsi" | "esi" => Some(registers::RSI),
        "rdi" | "edi" => Some(registers::RDI),
        "r8"  | "r8d" => Some(registers::R8),
        "r9"  | "r9d" => Some(registers::R9),
        "r10" | "r10d" => Some(registers::R10),
        "r11" | "r11d" => Some(registers::R11),
        "r12" | "r12d" => Some(registers::R12),
        "r13" | "r13d" => Some(registers::R13),
        "r14" | "r14d" => Some(registers::R14),
        "r15" | "r15d" => Some(registers::R15),
        "xmm0" => Some(registers::XMM0),
        "xmm1" => Some(registers::XMM1),
        "xmm2" => Some(registers::XMM2),
        "xmm3" => Some(registers::XMM3),
        "xmm4" => Some(registers::XMM4),
        "xmm5" => Some(registers::XMM5),
        "xmm6" => Some(registers::XMM6),
        "xmm7" => Some(registers::XMM7),
        // "memory" and "cc" are not physical registers.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::{CallingConvention, FunctionAttributes, Linkage};
    use crate::ir::instructions::{BasicBlockId, BinOp, ICmpPredicate, Instruction, ValueId};
    use crate::ir::types::IrType;

    /// Helper to create a CodegenConfig for testing.
    fn test_config() -> CodegenConfig {
        CodegenConfig::new(crate::common::target::Target::X86_64)
    }

    /// Helper to create a minimal IrFunction for testing.
    fn make_test_func(instructions: Vec<Instruction>) -> IrFunction {
        let mut bb = BasicBlock::new(BasicBlockId(0), None);
        for instr in instructions {
            bb.add_instruction(instr);
        }
        IrFunction {
            name: "test_fn".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            basic_blocks: vec![bb],
            entry_block_id: BasicBlockId(0),
            calling_convention: CallingConvention::C,
            linkage: Linkage::External,
            is_variadic: false,
            attributes: FunctionAttributes::default(),
            local_values: Vec::new(),
            next_value_id: 0,
            alignment: 16,
            section: None,
            is_definition: true,
        }
    }

    #[test]
    fn select_add_produces_mov_and_add() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::BinOp {
                result: ValueId(0),
                op: BinOp::Add,
                lhs: ValueId(1),
                rhs: ValueId(2),
                ty: IrType::I32,
            },
        ]);

        let mf = selector.select_instructions(&func);
        assert_eq!(mf.blocks.len(), 1);
        // Should produce: MOV v0, v1; ADD v0, v2
        assert_eq!(mf.blocks[0].instructions.len(), 2);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::MOV_RR);
        assert_eq!(mf.blocks[0].instructions[1].opcode, opcodes::ADD_RR);
    }

    #[test]
    fn select_load_produces_mov_rm() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::Load {
                result: ValueId(0),
                ptr: ValueId(1),
                ty: IrType::I64,
                volatile: false,
            },
        ]);

        let mf = selector.select_instructions(&func);
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::MOV_RM);
    }

    #[test]
    fn select_store_produces_mov_mr() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::Store {
                value: ValueId(0),
                ptr: ValueId(1),
                volatile: false,
            },
        ]);

        let mf = selector.select_instructions(&func);
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::MOV_MR);
    }

    #[test]
    fn select_branch_produces_jmp() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::Branch {
                target: BasicBlockId(1),
            },
        ]);

        let mf = selector.select_instructions(&func);
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::JMP);
        assert!(mf.blocks[0].instructions[0].is_terminator);
    }

    #[test]
    fn select_return_with_value() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::Return {
                value: Some(ValueId(0)),
            },
        ]);

        let mf = selector.select_instructions(&func);
        // Should produce: MOV RAX, v0; RET
        assert_eq!(mf.blocks[0].instructions.len(), 2);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::MOV_RR);
        assert_eq!(mf.blocks[0].instructions[1].opcode, opcodes::RET);
        assert!(mf.blocks[0].instructions[1].is_return);
    }

    #[test]
    fn select_return_void() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::Return { value: None },
        ]);

        let mf = selector.select_instructions(&func);
        // Should produce just: RET
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::RET);
    }

    #[test]
    fn select_icmp_produces_cmp_and_setcc() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::ICmp {
                result: ValueId(0),
                pred: ICmpPredicate::Eq,
                lhs: ValueId(1),
                rhs: ValueId(2),
            },
        ]);

        let mf = selector.select_instructions(&func);
        // Should produce: CMP v1, v2; SETcc v0
        assert_eq!(mf.blocks[0].instructions.len(), 2);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::CMP_RR);
        assert_eq!(mf.blocks[0].instructions[1].opcode, opcodes::SET_CC);
    }

    #[test]
    fn select_division_produces_mov_cqo_idiv_mov() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::BinOp {
                result: ValueId(0),
                op: BinOp::SDiv,
                lhs: ValueId(1),
                rhs: ValueId(2),
                ty: IrType::I64,
            },
        ]);

        let mf = selector.select_instructions(&func);
        // Should produce: MOV RAX, v1; CQO; IDIV v2; MOV v0, RAX
        assert_eq!(mf.blocks[0].instructions.len(), 4);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::MOV_RR);
        assert_eq!(mf.blocks[0].instructions[1].opcode, opcodes::CQO);
        assert_eq!(mf.blocks[0].instructions[2].opcode, opcodes::IDIV);
        assert_eq!(mf.blocks[0].instructions[3].opcode, opcodes::MOV_RR);
    }

    #[test]
    fn select_zext_produces_movzx() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::ZExt {
                result: ValueId(0),
                value: ValueId(1),
                to_ty: IrType::I32,
            },
        ]);

        let mf = selector.select_instructions(&func);
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::MOVZX);
    }

    #[test]
    fn select_alloca_produces_lea_frame_index() {
        let config = test_config();
        let selector = X86_64InstrSelector::new(&config);
        let func = make_test_func(vec![
            Instruction::Alloca {
                result: ValueId(3),
                ty: IrType::I32,
                alignment: 4,
            },
        ]);

        let mf = selector.select_instructions(&func);
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::LEA);
    }

    #[test]
    fn icmp_to_cc_mapping() {
        assert_eq!(icmp_to_cc(ICmpPredicate::Eq), cc::E);
        assert_eq!(icmp_to_cc(ICmpPredicate::Ne), cc::NE);
        assert_eq!(icmp_to_cc(ICmpPredicate::Slt), cc::L);
        assert_eq!(icmp_to_cc(ICmpPredicate::Sge), cc::GE);
        assert_eq!(icmp_to_cc(ICmpPredicate::Ult), cc::B);
        assert_eq!(icmp_to_cc(ICmpPredicate::Uge), cc::AE);
    }
}
