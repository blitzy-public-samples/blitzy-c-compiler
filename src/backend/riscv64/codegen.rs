//! RISC-V 64-bit instruction selection and emission module.
//!
//! This module implements the core code generation logic for the RV64IMAFDC ISA,
//! translating IR instructions into RISC-V 64 machine instructions. The instruction
//! selector ([`RiscV64InstrSel`]) handles all IR operation types and produces
//! [`MachineFunction`] output containing architecture-specific instructions.
//!
//! # Instruction Formats
//!
//! RISC-V uses six core instruction formats, all 32 bits wide:
//!
//! | Format | Purpose                          | Examples                    |
//! |--------|----------------------------------|-----------------------------|
//! | R-type | Register-register arithmetic     | ADD, SUB, MUL, FADD.D      |
//! | I-type | Immediate arithmetic, loads      | ADDI, LW, LD, JALR         |
//! | S-type | Stores                           | SW, SD, FSW, FSD            |
//! | B-type | Conditional branches             | BEQ, BNE, BLT, BGE         |
//! | U-type | Upper immediate                  | LUI, AUIPC                  |
//! | J-type | Unconditional jumps              | JAL                         |
//!
//! # Large Immediate Materialization
//!
//! RISC-V immediate fields are limited to 12 bits (I-type) or 20 bits (U-type).
//! For larger constants, multi-instruction sequences are used:
//!
//! - **12-bit signed range [-2048, 2047]:** Single `ADDI rd, x0, imm`
//! - **32-bit:** `LUI rd, upper20` + `ADDI rd, rd, lower12`
//! - **64-bit:** Multi-step `LUI`+`ADDI`+`SLLI`+`ADDI` sequences
//!
//! # PIC Addressing
//!
//! Position-independent code uses `AUIPC`+`LD` for GOT-relative global access
//! and `AUIPC`+`JALR` for PLT-relative function calls.

use crate::backend::riscv64::abi::{
    ArgClassification, ReturnClassification, RiscV64Abi, StackLayout,
};
use crate::backend::riscv64::registers;
use crate::backend::traits::{
    MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand, PhysReg,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::{CallingConvention, IrFunction, Parameter};
use crate::ir::instructions::{
    BasicBlockId, BinOp, FCmpPredicate, ICmpPredicate, Instruction, ValueId,
};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// RISC-V 64 instruction opcode constants
// ---------------------------------------------------------------------------
// MachineInstr.opcode is u32. We use sequential constants grouped by format.

// R-type integer arithmetic
pub const RV_ADD: u32 = 0;
pub const RV_SUB: u32 = 1;
pub const RV_AND: u32 = 2;
pub const RV_OR: u32 = 3;
pub const RV_XOR: u32 = 4;
pub const RV_SLL: u32 = 5;
pub const RV_SRL: u32 = 6;
pub const RV_SRA: u32 = 7;
pub const RV_SLT: u32 = 8;
pub const RV_SLTU: u32 = 9;

// R-type word operations (32-bit on RV64)
pub const RV_ADDW: u32 = 10;
pub const RV_SUBW: u32 = 11;
pub const RV_SLLW: u32 = 12;
pub const RV_SRLW: u32 = 13;
pub const RV_SRAW: u32 = 14;

// M-extension (multiply/divide) — 64-bit
pub const RV_MUL: u32 = 15;
pub const RV_MULH: u32 = 16;
pub const RV_MULHU: u32 = 17;
pub const RV_MULHSU: u32 = 18;
pub const RV_DIV: u32 = 19;
pub const RV_DIVU: u32 = 20;
pub const RV_REM: u32 = 21;
pub const RV_REMU: u32 = 22;

// M-extension word operations (32-bit on RV64)
pub const RV_MULW: u32 = 23;
pub const RV_DIVW: u32 = 24;
pub const RV_DIVUW: u32 = 25;
pub const RV_REMW: u32 = 26;
pub const RV_REMUW: u32 = 27;

// I-type immediate arithmetic
pub const RV_ADDI: u32 = 28;
pub const RV_ANDI: u32 = 29;
pub const RV_ORI: u32 = 30;
pub const RV_XORI: u32 = 31;
pub const RV_SLTI: u32 = 32;
pub const RV_SLTIU: u32 = 33;
pub const RV_ADDIW: u32 = 34;

// I-type immediate shifts
pub const RV_SLLI: u32 = 35;
pub const RV_SRLI: u32 = 36;
pub const RV_SRAI: u32 = 37;

// Load instructions (I-type)
pub const RV_LB: u32 = 38;
pub const RV_LBU: u32 = 39;
pub const RV_LH: u32 = 40;
pub const RV_LHU: u32 = 41;
pub const RV_LW: u32 = 42;
pub const RV_LWU: u32 = 43;
pub const RV_LD: u32 = 44;

// Store instructions (S-type)
pub const RV_SB: u32 = 45;
pub const RV_SH: u32 = 46;
pub const RV_SW: u32 = 47;
pub const RV_SD: u32 = 48;

// Branch instructions (B-type)
pub const RV_BEQ: u32 = 49;
pub const RV_BNE: u32 = 50;
pub const RV_BLT: u32 = 51;
pub const RV_BGE: u32 = 52;
pub const RV_BLTU: u32 = 53;
pub const RV_BGEU: u32 = 54;

// Upper immediate (U-type)
pub const RV_LUI: u32 = 55;
pub const RV_AUIPC: u32 = 56;

// Jump instructions (J-type / I-type)
pub const RV_JAL: u32 = 57;
pub const RV_JALR: u32 = 58;

// F-extension — single-precision floating-point
pub const RV_FLW: u32 = 59;
pub const RV_FSW: u32 = 60;
pub const RV_FADD_S: u32 = 61;
pub const RV_FSUB_S: u32 = 62;
pub const RV_FMUL_S: u32 = 63;
pub const RV_FDIV_S: u32 = 64;
pub const RV_FSQRT_S: u32 = 65;
pub const RV_FMIN_S: u32 = 66;
pub const RV_FMAX_S: u32 = 67;
pub const RV_FEQ_S: u32 = 68;
pub const RV_FLT_S: u32 = 69;
pub const RV_FLE_S: u32 = 70;
pub const RV_FCLASS_S: u32 = 71;
pub const RV_FCVT_W_S: u32 = 72;
pub const RV_FCVT_WU_S: u32 = 73;
pub const RV_FCVT_L_S: u32 = 74;
pub const RV_FCVT_LU_S: u32 = 75;
pub const RV_FCVT_S_W: u32 = 76;
pub const RV_FCVT_S_WU: u32 = 77;
pub const RV_FCVT_S_L: u32 = 78;
pub const RV_FCVT_S_LU: u32 = 79;
pub const RV_FMV_X_W: u32 = 80;
pub const RV_FMV_W_X: u32 = 81;

// D-extension — double-precision floating-point
pub const RV_FLD: u32 = 82;
pub const RV_FSD: u32 = 83;
pub const RV_FADD_D: u32 = 84;
pub const RV_FSUB_D: u32 = 85;
pub const RV_FMUL_D: u32 = 86;
pub const RV_FDIV_D: u32 = 87;
pub const RV_FSQRT_D: u32 = 88;
pub const RV_FMIN_D: u32 = 89;
pub const RV_FMAX_D: u32 = 90;
pub const RV_FEQ_D: u32 = 91;
pub const RV_FLT_D: u32 = 92;
pub const RV_FLE_D: u32 = 93;
pub const RV_FCLASS_D: u32 = 94;
pub const RV_FCVT_W_D: u32 = 95;
pub const RV_FCVT_WU_D: u32 = 96;
pub const RV_FCVT_L_D: u32 = 97;
pub const RV_FCVT_LU_D: u32 = 98;
pub const RV_FCVT_D_W: u32 = 99;
pub const RV_FCVT_D_WU: u32 = 100;
pub const RV_FCVT_D_L: u32 = 101;
pub const RV_FCVT_D_LU: u32 = 102;
pub const RV_FCVT_S_D: u32 = 103;
pub const RV_FCVT_D_S: u32 = 104;
pub const RV_FMV_X_D: u32 = 105;
pub const RV_FMV_D_X: u32 = 106;

// Pseudo-instructions used during selection (expanded by the assembler)
pub const RV_NOP: u32 = 200;
pub const RV_MV: u32 = 201;
pub const RV_NEG: u32 = 202;
pub const RV_LI: u32 = 203;
pub const RV_CALL: u32 = 204;
pub const RV_TAIL: u32 = 205;
pub const RV_RET: u32 = 206;
pub const RV_LA: u32 = 207;
pub const RV_FMOV_S: u32 = 208;
pub const RV_FMOV_D: u32 = 209;
pub const RV_FNEG_S: u32 = 210;
pub const RV_FNEG_D: u32 = 211;
pub const RV_INLINE_ASM: u32 = 250;

// ---------------------------------------------------------------------------
// Frame object — tracking stack-allocated locals
// ---------------------------------------------------------------------------

/// Describes a single stack-allocated frame object (from an IR `Alloca`).
/// The register allocator and prologue/epilogue emitter use these to compute
/// final stack offsets.
#[derive(Clone, Debug)]
struct FrameObject {
    /// Size in bytes of the allocated storage.
    size: u32,
    /// Required alignment in bytes.
    alignment: u32,
    /// Offset from the frame pointer (resolved during prologue emission).
    offset: i32,
}

// ---------------------------------------------------------------------------
// RiscV64InstrSel — instruction selection engine
// ---------------------------------------------------------------------------

/// RISC-V 64-bit instruction selection engine.
///
/// Translates a complete IR function into a [`MachineFunction`] containing
/// RISC-V 64 machine instructions. Maintains per-function state for value
/// mapping (IR SSA values → machine operands), block label resolution,
/// and stack frame construction.
///
/// # Usage
///
/// ```ignore
/// let mut isel = RiscV64InstrSel::new(/*pic_enabled=*/ false);
/// let machine_func = isel.select_function(&ir_function);
/// ```
pub struct RiscV64InstrSel {
    /// Whether position-independent code generation is enabled (`-fPIC`).
    pic_enabled: bool,
    /// Maps IR SSA ValueIds to their corresponding machine operands.
    /// Populated during instruction selection as each IR instruction is lowered.
    value_map: FxHashMap<ValueId, MachineOperand>,
    /// Maps IR BasicBlockIds to machine basic block label indices.
    block_map: FxHashMap<BasicBlockId, u32>,
    /// Next virtual register ID for temporaries created during selection.
    /// Starts from the IR function's next_value_id to avoid collisions.
    next_vreg_id: u32,
    /// Stack frame objects tracking alloca-created locals.
    frame_objects: Vec<FrameObject>,
    /// Next frame index for alloca stack slots.
    next_frame_index: u32,
    /// Running total of frame space consumed by frame objects (before alignment).
    frame_locals_size: u32,
    /// The ABI classifier instance used for call/return lowering.
    abi: RiscV64Abi,
    /// Set of callee-saved registers actually used in the current function.
    used_callee_saved: Vec<PhysReg>,
    /// Whether the current function contains any calls (affects RA save).
    has_calls: bool,
}

// ---------------------------------------------------------------------------
// Construction and internal helpers
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Creates a new RISC-V 64 instruction selector.
    ///
    /// # Arguments
    ///
    /// * `pic_enabled` — whether to generate position-independent code. When
    ///   `true`, global accesses use GOT-relative addressing (`AUIPC`+`LD`)
    ///   and function calls use PLT stubs (`AUIPC`+`JALR`).
    pub fn new(pic_enabled: bool) -> Self {
        Self {
            pic_enabled,
            value_map: FxHashMap::default(),
            block_map: FxHashMap::default(),
            next_vreg_id: 0,
            frame_objects: Vec::new(),
            next_frame_index: 0,
            frame_locals_size: 0,
            abi: RiscV64Abi::new(),
            used_callee_saved: Vec::new(),
            has_calls: false,
        }
    }

    // -- Private helpers --

    /// Allocates a fresh virtual register ID, returning it as a `ValueId`.
    /// These temporaries do not collide with IR-produced ValueIds because
    /// `next_vreg_id` is seeded from the IR function's `next_value_id`.
    fn alloc_vreg(&mut self) -> ValueId {
        let id = ValueId(self.next_vreg_id);
        self.next_vreg_id += 1;
        id
    }

    /// Returns the machine operand corresponding to an IR `ValueId`.
    ///
    /// If the value has already been lowered (present in `value_map`), the
    /// mapped operand is returned. Otherwise, a `VirtualReg` wrapper is
    /// created and cached, assuming the register allocator will resolve it.
    fn operand_for_value(&mut self, val: ValueId) -> MachineOperand {
        if let Some(op) = self.value_map.get(&val) {
            return op.clone();
        }
        let op = MachineOperand::VirtualReg(val);
        self.value_map.insert(val, op.clone());
        op
    }

    /// Returns the machine basic-block label index for an IR block ID.
    /// Panics if the block was not registered during initial block mapping.
    fn block_label(&self, bb: BasicBlockId) -> u32 {
        *self
            .block_map
            .get(&bb)
            .unwrap_or_else(|| panic!("BasicBlockId {:?} not found in block_map", bb))
    }

    /// Emits a machine instruction into the specified machine basic block.
    fn emit_to_block(block: &mut MachineBasicBlock, instr: MachineInstr) {
        block.instructions.push(instr);
    }

    /// Creates a simple R-type instruction: `op rd, rs1, rs2`.
    fn make_rrr(
        opcode: u32,
        rd: MachineOperand,
        rs1: MachineOperand,
        rs2: MachineOperand,
    ) -> MachineInstr {
        MachineInstr::with_operands(opcode, vec![rd, rs1, rs2])
    }

    /// Creates a simple I-type instruction: `op rd, rs1, imm`.
    fn make_rri(
        opcode: u32,
        rd: MachineOperand,
        rs1: MachineOperand,
        imm: i64,
    ) -> MachineInstr {
        MachineInstr::with_operands(opcode, vec![rd, rs1, MachineOperand::Immediate(imm)])
    }

    /// Creates a load instruction: `op rd, offset(base)`.
    fn make_load(
        opcode: u32,
        rd: MachineOperand,
        base: MachineOperand,
        offset: i64,
    ) -> MachineInstr {
        MachineInstr::with_operands(opcode, vec![rd, base, MachineOperand::Immediate(offset)])
    }

    /// Creates a store instruction: `op rs2, offset(base)`.
    fn make_store(
        opcode: u32,
        src: MachineOperand,
        base: MachineOperand,
        offset: i64,
    ) -> MachineInstr {
        MachineInstr::with_operands(opcode, vec![src, base, MachineOperand::Immediate(offset)])
    }

    /// Creates a branch instruction: `op rs1, rs2, label`.
    fn make_branch(
        opcode: u32,
        rs1: MachineOperand,
        rs2: MachineOperand,
        label: u32,
    ) -> MachineInstr {
        let mut instr =
            MachineInstr::with_operands(opcode, vec![rs1, rs2, MachineOperand::Label(label)]);
        instr.is_terminator = true;
        instr
    }

    /// Determines the appropriate load opcode based on IR type and signedness.
    /// For integer loads below 64-bit, `signed` selects sign-extending vs
    /// zero-extending variants (e.g., LB vs LBU).
    fn load_opcode_for_type(ty: &IrType, signed: bool) -> u32 {
        match ty {
            IrType::I1 | IrType::I8 => {
                if signed {
                    RV_LB
                } else {
                    RV_LBU
                }
            }
            IrType::I16 => {
                if signed {
                    RV_LH
                } else {
                    RV_LHU
                }
            }
            IrType::I32 => {
                if signed {
                    RV_LW
                } else {
                    RV_LWU
                }
            }
            IrType::I64 | IrType::Ptr => RV_LD,
            IrType::F32 => RV_FLW,
            IrType::F64 => RV_FLD,
            _ => RV_LD, // Default to doubleword for aggregates/pointers
        }
    }

    /// Determines the appropriate store opcode based on IR type.
    fn store_opcode_for_type(ty: &IrType) -> u32 {
        match ty {
            IrType::I1 | IrType::I8 => RV_SB,
            IrType::I16 => RV_SH,
            IrType::I32 => RV_SW,
            IrType::I64 | IrType::Ptr => RV_SD,
            IrType::F32 => RV_FSW,
            IrType::F64 => RV_FSD,
            _ => RV_SD,
        }
    }

    /// Returns `true` if the given IR type should be held in a floating-point
    /// register.
    fn is_fp_type(ty: &IrType) -> bool {
        matches!(ty, IrType::F32 | IrType::F64 | IrType::F80)
    }

    /// Returns `true` if the given immediate fits in a signed 12-bit field.
    fn fits_in_simm12(val: i64) -> bool {
        val >= -2048 && val <= 2047
    }

    /// Returns `true` if the given IR type is a 32-bit integer requiring
    /// W-suffix operations on RV64 to produce correctly sign-extended results.
    fn is_word_type(ty: &IrType) -> bool {
        matches!(ty, IrType::I32)
    }

    /// Resets per-function state. Called at the start of `select_function`.
    fn reset(&mut self) {
        self.value_map.clear();
        self.block_map.clear();
        self.frame_objects.clear();
        self.next_frame_index = 0;
        self.frame_locals_size = 0;
        self.used_callee_saved.clear();
        self.has_calls = false;
    }
}

// ---------------------------------------------------------------------------
// select_function — top-level function selection
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Translates an entire IR function into a [`MachineFunction`].
    ///
    /// This is the main entry point for RISC-V 64 instruction selection.
    /// It performs the following steps:
    ///
    /// 1. Reset per-function state and seed virtual register counter.
    /// 2. Create machine basic blocks and populate the block label map.
    /// 3. Map function parameters to their ABI locations (registers/stack).
    /// 4. Iterate over every IR basic block and select machine instructions
    ///    for each IR instruction within.
    /// 5. Emit prologue/epilogue sequences.
    /// 6. Return the completed `MachineFunction`.
    pub fn select_function(&mut self, func: &IrFunction) -> MachineFunction {
        self.reset();
        self.next_vreg_id = func.next_value_id;

        // Create the output MachineFunction with 16-byte stack alignment.
        let mut mf = MachineFunction::new(func.name.clone(), 16);

        // Step 1: Create machine blocks and populate block_map.
        for (idx, bb) in func.basic_blocks.iter().enumerate() {
            let label = idx as u32;
            self.block_map.insert(bb.id, label);
            let mbb = MachineBasicBlock {
                id: label,
                instructions: Vec::new(),
                label: bb.name.clone(),
            };
            mf.add_block(mbb);
        }

        // Step 2: Lower function parameters into the value map.
        self.lower_parameters(func, &mut mf);

        // Step 3: Select instructions for each basic block.
        for bb in &func.basic_blocks {
            let block_label = self.block_label(bb.id);
            let instrs = bb.instructions();
            // Collect selected instructions into a temporary vector, then
            // append them to the machine basic block.
            let mut selected: Vec<MachineInstr> = Vec::new();
            for ir_inst in instrs {
                self.select_instruction_into(ir_inst, func, &mut selected);
            }
            // Find the machine basic block and append instructions.
            if let Some(mbb) = mf.blocks.iter_mut().find(|b| b.id == block_label) {
                mbb.instructions.extend(selected);
            }
        }

        // Step 4: Determine callee-saved register usage and frame size.
        self.compute_callee_saved(&mf);
        let locals_size = self.frame_locals_size;
        let callee_save_size = (self.used_callee_saved.len() as u32) * 8;
        // RA is always saved if the function makes calls.
        let ra_save = if self.has_calls { 8u32 } else { 0 };
        let raw_frame = locals_size + callee_save_size + ra_save;
        // Align to 16 bytes.
        let frame_size = (raw_frame + 15) & !15;

        mf.frame_size = frame_size;
        mf.has_calls = self.has_calls;
        mf.used_callee_saved = self.used_callee_saved.clone();
        mf.stack_alignment = 16;

        // Step 5: Emit prologue and epilogue.
        self.emit_prologue(&mut mf, frame_size);
        self.emit_epilogue(&mut mf, frame_size);

        mf
    }

    /// Lowers function parameters into the value map according to the LP64D
    /// ABI. Integer arguments go in a0–a7, float arguments go in fa0–fa7,
    /// and excess arguments are passed on the stack.
    ///
    /// Uses [`registers::is_float_reg`] and [`registers::is_integer_reg`] to
    /// validate register class assignments after ABI classification.
    fn lower_parameters(&mut self, func: &IrFunction, _mf: &mut MachineFunction) {
        let mut int_idx: usize = 0;
        let mut fp_idx: usize = 0;
        let mut stack_offset: i32 = 0;

        for param in &func.params {
            let ty = &param.ty;
            if Self::is_fp_type(ty) && fp_idx < registers::FLOAT_ARG_REGS.len() {
                // Pass in floating-point argument register.
                let reg = registers::FLOAT_ARG_REGS[fp_idx];
                debug_assert!(
                    registers::is_float_reg(reg),
                    "ABI float arg reg must be a float register: {}",
                    registers::reg_name(reg),
                );
                self.value_map
                    .insert(param.id, MachineOperand::Register(reg));
                fp_idx += 1;
            } else if !Self::is_fp_type(ty) && int_idx < registers::INTEGER_ARG_REGS.len() {
                // Pass in integer argument register.
                let reg = registers::INTEGER_ARG_REGS[int_idx];
                debug_assert!(
                    registers::is_integer_reg(reg),
                    "ABI int arg reg must be an integer register: {}",
                    registers::reg_name(reg),
                );
                self.value_map
                    .insert(param.id, MachineOperand::Register(reg));
                int_idx += 1;
            } else {
                // Spill to stack. The caller places excess args above the
                // callee's frame pointer. Offset is relative to incoming SP.
                self.value_map.insert(
                    param.id,
                    MachineOperand::Memory {
                        base: registers::FP,
                        offset: stack_offset,
                        index: None,
                        scale: 1,
                    },
                );
                stack_offset += 8; // All stack slots are 8-byte aligned on RV64.
            }
        }
    }

    /// Scans emitted machine instructions for physical register uses to
    /// determine which callee-saved registers need to be preserved.
    ///
    /// Iterates the canonical callee-saved register sets from
    /// [`registers::CALLEE_SAVED_INT`] and [`registers::CALLEE_SAVED_FP`]
    /// to collect only those that are actually referenced in the emitted code.
    fn compute_callee_saved(&mut self, mf: &MachineFunction) {
        // Build the set of all registers actually used in emitted instructions.
        let mut referenced: Vec<PhysReg> = Vec::new();
        for block in &mf.blocks {
            for instr in &block.instructions {
                if instr.is_call {
                    self.has_calls = true;
                }
                for op in &instr.operands {
                    if let MachineOperand::Register(reg) = op {
                        if !referenced.contains(reg) {
                            referenced.push(*reg);
                        }
                    }
                }
                for reg in &instr.implicit_defs {
                    if !referenced.contains(reg) {
                        referenced.push(*reg);
                    }
                }
                for reg in &instr.implicit_uses {
                    if !referenced.contains(reg) {
                        referenced.push(*reg);
                    }
                }
            }
        }

        // Intersect referenced registers with the canonical callee-saved
        // sets. This ensures we only preserve registers that are both
        // callee-saved and actually used by the function body.
        let mut used = Vec::new();
        for &reg in registers::CALLEE_SAVED_INT.iter() {
            debug_assert!(registers::is_callee_saved(reg));
            if referenced.contains(&reg) && !used.contains(&reg) {
                used.push(reg);
            }
        }
        for &reg in registers::CALLEE_SAVED_FP.iter() {
            debug_assert!(registers::is_callee_saved(reg));
            if referenced.contains(&reg) && !used.contains(&reg) {
                used.push(reg);
            }
        }
        self.used_callee_saved = used;
    }
}

// ---------------------------------------------------------------------------
// select_instruction — single-instruction dispatch
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Dispatches a single IR instruction to the appropriate selection method.
    ///
    /// This is the public single-instruction entry point used by the
    /// `select_function` driver. Each IR instruction variant is matched and
    /// delegated to a specialised handler that emits one or more RISC-V
    /// machine instructions.
    pub fn select_instruction(
        &mut self,
        ir_inst: &Instruction,
        func: &IrFunction,
        mf: &mut MachineFunction,
    ) {
        let mut out = Vec::new();
        self.select_instruction_into(ir_inst, func, &mut out);
        // Append to the last block — used when called externally for single
        // instruction insertion.
        if let Some(last_block) = mf.blocks.last_mut() {
            last_block.instructions.extend(out);
        }
    }

    /// Internal instruction dispatch. Appends emitted machine instructions
    /// to `out` rather than directly to a machine block.
    fn select_instruction_into(
        &mut self,
        ir_inst: &Instruction,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        match ir_inst {
            Instruction::Alloca { result, ty, alignment } => {
                self.select_alloca(*result, ty, *alignment, out);
            }
            Instruction::Load { result, ptr, ty, volatile: _ } => {
                self.select_load(*result, *ptr, ty, func, out);
            }
            Instruction::Store { value, ptr, volatile: _ } => {
                self.select_store(*value, *ptr, func, out);
            }
            Instruction::BinOp { result, op, lhs, rhs, ty } => {
                self.select_binary_op(*result, *op, *lhs, *rhs, ty, out);
            }
            Instruction::ICmp { result, pred, lhs, rhs } => {
                self.select_comparison(*result, *pred, *lhs, *rhs, func, out);
            }
            Instruction::FCmp { result, pred, lhs, rhs } => {
                self.select_fp_comparison(*result, *pred, *lhs, *rhs, func, out);
            }
            Instruction::Branch { target } => {
                self.select_branch(*target, out);
            }
            Instruction::CondBranch { condition, true_target, false_target } => {
                self.select_cond_branch(*condition, *true_target, *false_target, out);
            }
            Instruction::Switch { value, default, cases } => {
                self.select_switch(*value, *default, cases, out);
            }
            Instruction::Call { result, callee, args, is_tail } => {
                self.select_call(*result, *callee, args, *is_tail, func, out);
            }
            Instruction::Return { value } => {
                self.select_return(*value, func, out);
            }
            Instruction::Phi { result, ty, incoming } => {
                self.select_phi(*result, ty, incoming, out);
            }
            Instruction::GetElementPtr { result, base, indices, ty, in_bounds: _ } => {
                self.select_gep(*result, *base, indices, ty, func, out);
            }
            Instruction::BitCast { result, value, .. } => {
                // BitCast is a no-op at the machine level — same bits, new type.
                // Just alias the operand.
                let src = self.operand_for_value(*value);
                self.value_map.insert(*result, src);
            }
            Instruction::Trunc { result, value, to_ty } => {
                self.select_cast(*result, *value, to_ty, CastKind::Trunc, func, out);
            }
            Instruction::ZExt { result, value, to_ty } => {
                self.select_cast(*result, *value, to_ty, CastKind::ZExt, func, out);
            }
            Instruction::SExt { result, value, to_ty } => {
                self.select_cast(*result, *value, to_ty, CastKind::SExt, func, out);
            }
            Instruction::IntToPtr { result, value, .. } => {
                // IntToPtr is a no-op on RV64 — pointers are 64-bit integers.
                let src = self.operand_for_value(*value);
                self.value_map.insert(*result, src);
            }
            Instruction::PtrToInt { result, value, .. } => {
                // PtrToInt is a no-op on RV64 — pointers are 64-bit integers.
                let src = self.operand_for_value(*value);
                self.value_map.insert(*result, src);
            }
            Instruction::InlineAsm {
                result, template, constraints, operands,
                clobbers, has_side_effects, ..
            } => {
                self.select_inline_asm(
                    result.as_ref().copied(),
                    template,
                    constraints,
                    operands,
                    clobbers,
                    *has_side_effects,
                    out,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cast kind helper enum
// ---------------------------------------------------------------------------

/// Internal enum distinguishing truncation, zero-extension, and sign-extension
/// casts during instruction selection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CastKind {
    Trunc,
    ZExt,
    SExt,
}

// ---------------------------------------------------------------------------
// select_binary_op — arithmetic and bitwise operations
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Selects RISC-V instructions for an IR binary operation.
    ///
    /// Handles integer arithmetic, bitwise ops, shifts, multiply/divide (M
    /// extension), and floating-point arithmetic (F/D extensions). On RV64,
    /// 32-bit integer operations use W-suffix variants (ADDW, SUBW, etc.)
    /// to produce correctly sign-extended 64-bit results.
    pub fn select_binary_op(
        &mut self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: &IrType,
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);
        let rs1 = self.operand_for_value(lhs);
        let rs2 = self.operand_for_value(rhs);
        let use_word = Self::is_word_type(ty);

        let instr = match op {
            // -- Integer arithmetic --
            BinOp::Add => {
                if use_word {
                    Self::make_rrr(RV_ADDW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_ADD, rd.clone(), rs1, rs2)
                }
            }
            BinOp::Sub => {
                if use_word {
                    Self::make_rrr(RV_SUBW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_SUB, rd.clone(), rs1, rs2)
                }
            }
            BinOp::Mul => {
                if use_word {
                    Self::make_rrr(RV_MULW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_MUL, rd.clone(), rs1, rs2)
                }
            }
            BinOp::SDiv => {
                if use_word {
                    Self::make_rrr(RV_DIVW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_DIV, rd.clone(), rs1, rs2)
                }
            }
            BinOp::UDiv => {
                if use_word {
                    Self::make_rrr(RV_DIVUW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_DIVU, rd.clone(), rs1, rs2)
                }
            }
            BinOp::SRem => {
                if use_word {
                    Self::make_rrr(RV_REMW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_REM, rd.clone(), rs1, rs2)
                }
            }
            BinOp::URem => {
                if use_word {
                    Self::make_rrr(RV_REMUW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_REMU, rd.clone(), rs1, rs2)
                }
            }

            // -- Shifts --
            BinOp::Shl => {
                if use_word {
                    Self::make_rrr(RV_SLLW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_SLL, rd.clone(), rs1, rs2)
                }
            }
            BinOp::LShr => {
                if use_word {
                    Self::make_rrr(RV_SRLW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_SRL, rd.clone(), rs1, rs2)
                }
            }
            BinOp::AShr => {
                if use_word {
                    Self::make_rrr(RV_SRAW, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_SRA, rd.clone(), rs1, rs2)
                }
            }

            // -- Bitwise --
            BinOp::And => Self::make_rrr(RV_AND, rd.clone(), rs1, rs2),
            BinOp::Or => Self::make_rrr(RV_OR, rd.clone(), rs1, rs2),
            BinOp::Xor => Self::make_rrr(RV_XOR, rd.clone(), rs1, rs2),

            // -- Single-precision floating-point --
            BinOp::FAdd => {
                if matches!(ty, IrType::F32) {
                    Self::make_rrr(RV_FADD_S, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_FADD_D, rd.clone(), rs1, rs2)
                }
            }
            BinOp::FSub => {
                if matches!(ty, IrType::F32) {
                    Self::make_rrr(RV_FSUB_S, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_FSUB_D, rd.clone(), rs1, rs2)
                }
            }
            BinOp::FMul => {
                if matches!(ty, IrType::F32) {
                    Self::make_rrr(RV_FMUL_S, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_FMUL_D, rd.clone(), rs1, rs2)
                }
            }
            BinOp::FDiv => {
                if matches!(ty, IrType::F32) {
                    Self::make_rrr(RV_FDIV_S, rd.clone(), rs1, rs2)
                } else {
                    Self::make_rrr(RV_FDIV_D, rd.clone(), rs1, rs2)
                }
            }
            BinOp::FRem => {
                // RISC-V has no FREM instruction. Implement via runtime call
                // to __fmodl / fmodf / fmod. Emit a CALL pseudo.
                let fname = if matches!(ty, IrType::F32) {
                    "fmodf"
                } else {
                    "fmod"
                };
                // Move operands to fa0, fa1 (float arg regs).
                out.push(MachineInstr::with_operands(
                    RV_FMOV_D,
                    vec![MachineOperand::Register(registers::FA0), rs1],
                ));
                out.push(MachineInstr::with_operands(
                    RV_FMOV_D,
                    vec![MachineOperand::Register(registers::FA1), rs2],
                ));
                let mut call_instr = MachineInstr::with_operands(
                    RV_CALL,
                    vec![MachineOperand::Symbol(fname.to_string())],
                );
                call_instr.is_call = true;
                self.has_calls = true;
                out.push(call_instr);
                // Result is in fa0.
                let move_result = MachineInstr::with_operands(
                    RV_FMOV_D,
                    vec![rd.clone(), MachineOperand::Register(registers::FA0)],
                );
                out.push(move_result);
                self.value_map.insert(result, rd);
                return;
            }
        };

        out.push(instr);
        self.value_map.insert(result, rd);
    }

    // -----------------------------------------------------------------------
    // select_comparison — integer comparisons (ICmp)
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for an IR integer comparison.
    ///
    /// RISC-V has only BEQ/BNE/BLT/BGE/BLTU/BGEU for branches, and SLT/SLTU
    /// for set-on-less-than. Comparisons not directly supported are lowered
    /// by swapping operands or using SLT+XORI sequences.
    ///
    /// The result is an I1 value (0 or 1) placed in a virtual register.
    pub fn select_comparison(
        &mut self,
        result: ValueId,
        pred: ICmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);
        let rs1 = self.operand_for_value(lhs);
        let rs2 = self.operand_for_value(rhs);

        match pred {
            ICmpPredicate::Eq => {
                // rd = (rs1 == rs2): XOR then SLTIU against 1
                let tmp = self.alloc_vreg();
                let tmp_op = MachineOperand::VirtualReg(tmp);
                out.push(Self::make_rrr(RV_XOR, tmp_op.clone(), rs1, rs2));
                out.push(Self::make_rri(RV_SLTIU, rd.clone(), tmp_op, 1));
            }
            ICmpPredicate::Ne => {
                // rd = (rs1 != rs2): XOR then SLTU x0 against result
                let tmp = self.alloc_vreg();
                let tmp_op = MachineOperand::VirtualReg(tmp);
                out.push(Self::make_rrr(RV_XOR, tmp_op.clone(), rs1, rs2));
                out.push(Self::make_rrr(
                    RV_SLTU,
                    rd.clone(),
                    MachineOperand::Register(registers::ZERO),
                    tmp_op,
                ));
            }
            ICmpPredicate::Slt => {
                out.push(Self::make_rrr(RV_SLT, rd.clone(), rs1, rs2));
            }
            ICmpPredicate::Sge => {
                // rd = !(rs1 < rs2) = SLT then XORI with 1
                let tmp = self.alloc_vreg();
                let tmp_op = MachineOperand::VirtualReg(tmp);
                out.push(Self::make_rrr(RV_SLT, tmp_op.clone(), rs1, rs2));
                out.push(Self::make_rri(RV_XORI, rd.clone(), tmp_op, 1));
            }
            ICmpPredicate::Sgt => {
                // rd = (rs2 < rs1) — swap operands
                out.push(Self::make_rrr(RV_SLT, rd.clone(), rs2, rs1));
            }
            ICmpPredicate::Sle => {
                // rd = !(rs2 < rs1) — swap operands + invert
                let tmp = self.alloc_vreg();
                let tmp_op = MachineOperand::VirtualReg(tmp);
                out.push(Self::make_rrr(RV_SLT, tmp_op.clone(), rs2, rs1));
                out.push(Self::make_rri(RV_XORI, rd.clone(), tmp_op, 1));
            }
            ICmpPredicate::Ult => {
                out.push(Self::make_rrr(RV_SLTU, rd.clone(), rs1, rs2));
            }
            ICmpPredicate::Uge => {
                let tmp = self.alloc_vreg();
                let tmp_op = MachineOperand::VirtualReg(tmp);
                out.push(Self::make_rrr(RV_SLTU, tmp_op.clone(), rs1, rs2));
                out.push(Self::make_rri(RV_XORI, rd.clone(), tmp_op, 1));
            }
            ICmpPredicate::Ugt => {
                out.push(Self::make_rrr(RV_SLTU, rd.clone(), rs2, rs1));
            }
            ICmpPredicate::Ule => {
                let tmp = self.alloc_vreg();
                let tmp_op = MachineOperand::VirtualReg(tmp);
                out.push(Self::make_rrr(RV_SLTU, tmp_op.clone(), rs2, rs1));
                out.push(Self::make_rri(RV_XORI, rd.clone(), tmp_op, 1));
            }
        }

        self.value_map.insert(result, rd);
    }

    // -----------------------------------------------------------------------
    // select_fp_comparison — floating-point comparisons (FCmp)
    // -----------------------------------------------------------------------

    /// Selects RISC-V instructions for an IR floating-point comparison.
    ///
    /// RISC-V provides FEQ, FLT, FLE for ordered comparisons. Unordered
    /// comparisons are implemented by checking for NaN via FCLASS and
    /// combining results. The result is an I1 integer (0 or 1).
    pub fn select_fp_comparison(
        &mut self,
        result: ValueId,
        pred: FCmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);
        let rs1 = self.operand_for_value(lhs);
        let rs2 = self.operand_for_value(rhs);

        // Determine if single or double precision based on operand type.
        let lhs_ty = func.get_value_type(lhs);
        let is_single = matches!(lhs_ty, IrType::F32);

        let (feq, flt, fle, fclass) = if is_single {
            (RV_FEQ_S, RV_FLT_S, RV_FLE_S, RV_FCLASS_S)
        } else {
            (RV_FEQ_D, RV_FLT_D, RV_FLE_D, RV_FCLASS_D)
        };

        match pred {
            // -- Ordered comparisons (false if NaN) --
            FCmpPredicate::OEq => {
                out.push(Self::make_rrr(feq, rd.clone(), rs1, rs2));
            }
            FCmpPredicate::Olt => {
                out.push(Self::make_rrr(flt, rd.clone(), rs1, rs2));
            }
            FCmpPredicate::Ole => {
                out.push(Self::make_rrr(fle, rd.clone(), rs1, rs2));
            }
            FCmpPredicate::Ogt => {
                // a > b  ↔  b < a
                out.push(Self::make_rrr(flt, rd.clone(), rs2, rs1));
            }
            FCmpPredicate::Oge => {
                // a >= b  ↔  b <= a
                out.push(Self::make_rrr(fle, rd.clone(), rs2, rs1));
            }
            FCmpPredicate::ONe => {
                // Ordered-not-equal: !(a==b) AND ordered.
                // FLT(a,b) | FLT(b,a) gives 1 iff a!=b and both are not NaN.
                let t1 = self.alloc_vreg();
                let t2 = self.alloc_vreg();
                let t1_op = MachineOperand::VirtualReg(t1);
                let t2_op = MachineOperand::VirtualReg(t2);
                out.push(Self::make_rrr(flt, t1_op.clone(), rs1.clone(), rs2.clone()));
                out.push(Self::make_rrr(flt, t2_op.clone(), rs2, rs1));
                out.push(Self::make_rrr(RV_OR, rd.clone(), t1_op, t2_op));
            }
            FCmpPredicate::Ord => {
                // Ordered: feq(a,a) & feq(b,b) — NaN != NaN is false.
                let t1 = self.alloc_vreg();
                let t2 = self.alloc_vreg();
                let t1_op = MachineOperand::VirtualReg(t1);
                let t2_op = MachineOperand::VirtualReg(t2);
                out.push(Self::make_rrr(feq, t1_op.clone(), rs1.clone(), rs1));
                out.push(Self::make_rrr(feq, t2_op.clone(), rs2.clone(), rs2));
                out.push(Self::make_rrr(RV_AND, rd.clone(), t1_op, t2_op));
            }
            FCmpPredicate::Uno => {
                // Unordered: !(feq(a,a)) | !(feq(b,b))
                let t1 = self.alloc_vreg();
                let t2 = self.alloc_vreg();
                let t3 = self.alloc_vreg();
                let t4 = self.alloc_vreg();
                let t1_op = MachineOperand::VirtualReg(t1);
                let t2_op = MachineOperand::VirtualReg(t2);
                let t3_op = MachineOperand::VirtualReg(t3);
                let t4_op = MachineOperand::VirtualReg(t4);
                out.push(Self::make_rrr(feq, t1_op.clone(), rs1.clone(), rs1));
                out.push(Self::make_rri(RV_XORI, t2_op.clone(), t1_op, 1));
                out.push(Self::make_rrr(feq, t3_op.clone(), rs2.clone(), rs2));
                out.push(Self::make_rri(RV_XORI, t4_op.clone(), t3_op, 1));
                out.push(Self::make_rrr(RV_OR, rd.clone(), t2_op, t4_op));
            }

            // -- Unordered comparisons (true if NaN) --
            // Implement as: !ordered_opposite
            FCmpPredicate::UEq => {
                // !(a<b) & !(b<a)  → but this gives Uge∧Ule. Actually UEq = !ONe.
                let t = self.alloc_vreg();
                let t_op = MachineOperand::VirtualReg(t);
                // ONe = flt(a,b)|flt(b,a)
                let t1 = self.alloc_vreg();
                let t2 = self.alloc_vreg();
                let t1_op = MachineOperand::VirtualReg(t1);
                let t2_op = MachineOperand::VirtualReg(t2);
                out.push(Self::make_rrr(flt, t1_op.clone(), rs1.clone(), rs2.clone()));
                out.push(Self::make_rrr(flt, t2_op.clone(), rs2, rs1));
                out.push(Self::make_rrr(RV_OR, t_op.clone(), t1_op, t2_op));
                out.push(Self::make_rri(RV_XORI, rd.clone(), t_op, 1));
            }
            FCmpPredicate::UNe => {
                // !OEq = !feq(a,b)
                let t = self.alloc_vreg();
                let t_op = MachineOperand::VirtualReg(t);
                out.push(Self::make_rrr(feq, t_op.clone(), rs1, rs2));
                out.push(Self::make_rri(RV_XORI, rd.clone(), t_op, 1));
            }
            FCmpPredicate::Ult => {
                // !Oge = !(b<=a) → !(fle(b,a))
                let t = self.alloc_vreg();
                let t_op = MachineOperand::VirtualReg(t);
                out.push(Self::make_rrr(fle, t_op.clone(), rs2, rs1));
                out.push(Self::make_rri(RV_XORI, rd.clone(), t_op, 1));
            }
            FCmpPredicate::Ule => {
                // !Ogt = !(b<a) → !(flt(b,a))
                let t = self.alloc_vreg();
                let t_op = MachineOperand::VirtualReg(t);
                out.push(Self::make_rrr(flt, t_op.clone(), rs2, rs1));
                out.push(Self::make_rri(RV_XORI, rd.clone(), t_op, 1));
            }
            FCmpPredicate::Ugt => {
                // !Ole = !(fle(a,b))
                let t = self.alloc_vreg();
                let t_op = MachineOperand::VirtualReg(t);
                out.push(Self::make_rrr(fle, t_op.clone(), rs1, rs2));
                out.push(Self::make_rri(RV_XORI, rd.clone(), t_op, 1));
            }
            FCmpPredicate::Uge => {
                // !Olt = !(flt(a,b))
                let t = self.alloc_vreg();
                let t_op = MachineOperand::VirtualReg(t);
                out.push(Self::make_rrr(flt, t_op.clone(), rs1, rs2));
                out.push(Self::make_rri(RV_XORI, rd.clone(), t_op, 1));
            }
        }

        self.value_map.insert(result, rd);
    }
}

// ---------------------------------------------------------------------------
// Control-flow: branches, conditional branches
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Emits an unconditional jump to a basic block label.
    pub fn select_branch(&mut self, target: BasicBlockId, out: &mut Vec<MachineInstr>) {
        let label = self.block_label(target);
        let mut instr =
            MachineInstr::with_operands(RV_JAL, vec![
                MachineOperand::Register(registers::ZERO),
                MachineOperand::Label(label),
            ]);
        instr.is_terminator = true;
        out.push(instr);
    }

    /// Emits a conditional branch: if `condition` is non-zero, jump to
    /// `true_target`, otherwise fall through/jump to `false_target`.
    ///
    /// RISC-V conditional branches compare two registers. Since the IR's
    /// `CondBranch` only provides a boolean condition value, we compare it
    /// against `x0` with `BNE` for the true branch and `JAL` for the false
    /// fallthrough.
    fn select_cond_branch(
        &mut self,
        condition: ValueId,
        true_target: BasicBlockId,
        false_target: BasicBlockId,
        out: &mut Vec<MachineInstr>,
    ) {
        let cond_op = self.operand_for_value(condition);
        let true_label = self.block_label(true_target);
        let false_label = self.block_label(false_target);

        // BNE cond, x0, true_label — branch if condition is non-zero
        out.push(Self::make_branch(
            RV_BNE,
            cond_op,
            MachineOperand::Register(registers::ZERO),
            true_label,
        ));
        // JAL x0, false_label — unconditional jump to false block
        let mut fallthrough =
            MachineInstr::with_operands(RV_JAL, vec![
                MachineOperand::Register(registers::ZERO),
                MachineOperand::Label(false_label),
            ]);
        fallthrough.is_terminator = true;
        out.push(fallthrough);
    }
}

// ---------------------------------------------------------------------------
// Memory operations: load, store, alloca
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Selects a load instruction. Determines the correct load width from the
    /// target type and emits the appropriate LB/LH/LW/LD/FLW/FLD.
    pub fn select_load(
        &mut self,
        result: ValueId,
        ptr: ValueId,
        ty: &IrType,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);
        let base = self.operand_for_value(ptr);

        // Check if the base operand already contains a memory offset
        // (e.g. from a stack slot). If so, use that offset directly.
        let (actual_base, offset) = match &base {
            MachineOperand::Memory { base: b, offset: o, .. } => {
                (MachineOperand::Register(*b), *o as i64)
            }
            MachineOperand::FrameIndex(idx) => {
                // Frame index will be resolved later by the prologue emitter.
                // For now, emit a load relative to FP.
                let fo = self.frame_objects.get(*idx as usize);
                let off = fo.map(|f| f.offset as i64).unwrap_or(0);
                (MachineOperand::Register(registers::FP), off)
            }
            _ => (base, 0i64),
        };

        let opcode = Self::load_opcode_for_type(ty, true);
        out.push(Self::make_load(opcode, rd.clone(), actual_base, offset));
        self.value_map.insert(result, rd);
    }

    /// Selects a store instruction. Determines the correct store width from
    /// the value's type and emits SB/SH/SW/SD/FSW/FSD.
    pub fn select_store(
        &mut self,
        value: ValueId,
        ptr: ValueId,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        let src = self.operand_for_value(value);
        let base = self.operand_for_value(ptr);
        let ty = func.get_value_type(value);

        let (actual_base, offset) = match &base {
            MachineOperand::Memory { base: b, offset: o, .. } => {
                (MachineOperand::Register(*b), *o as i64)
            }
            MachineOperand::FrameIndex(idx) => {
                let fo = self.frame_objects.get(*idx as usize);
                let off = fo.map(|f| f.offset as i64).unwrap_or(0);
                (MachineOperand::Register(registers::FP), off)
            }
            _ => (base, 0i64),
        };

        let opcode = Self::store_opcode_for_type(ty);
        out.push(Self::make_store(opcode, src, actual_base, offset));
    }

    /// Selects an alloca instruction. Allocates space on the stack frame and
    /// maps the result ValueId to a frame index operand.
    pub fn select_alloca(
        &mut self,
        result: ValueId,
        ty: &IrType,
        alignment: u32,
        out: &mut Vec<MachineInstr>,
    ) {
        let size = ty.size_bytes() as u32;
        let align = if alignment > 0 {
            alignment
        } else {
            std::cmp::max(size.next_power_of_two(), 1)
        };

        // Align frame_locals_size to the object's alignment.
        self.frame_locals_size = (self.frame_locals_size + align - 1) & !(align - 1);
        let offset = self.frame_locals_size as i32;
        self.frame_locals_size += size;

        let idx = self.next_frame_index;
        self.next_frame_index += 1;
        self.frame_objects.push(FrameObject {
            size,
            alignment: align,
            offset: -(offset as i32 + size as i32), // Negative offset from FP
        });

        // Map the alloca result to a frame index. The prologue/epilogue will
        // resolve this to an actual FP-relative address.
        let op = MachineOperand::FrameIndex(idx);
        self.value_map.insert(result, op);
    }

    /// Selects a GetElementPtr instruction. Computes the address of a
    /// sub-element of an aggregate type.
    ///
    /// For each index, the element size is multiplied by the index and added
    /// to the base pointer. On RISC-V, this involves `SLLI` (for power-of-2
    /// multiplies) or `MUL` plus `ADD`.
    pub fn select_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
        ty: &IrType,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);
        let mut current = self.operand_for_value(base);

        // Walk the type and indices to compute the final offset.
        let mut current_ty = ty.clone();
        for &idx_val in indices {
            let idx_op = self.operand_for_value(idx_val);

            // Determine the element size based on the current aggregate type.
            let elem_size = match &current_ty {
                IrType::Ptr => 8u64, // Pointer-to-anything, use pointee size
                IrType::Array(elem_ty, _) => {
                    let sz = elem_ty.size_bytes() as u64;
                    current_ty = (**elem_ty).clone();
                    sz
                }
                IrType::Struct(fields) => {
                    // For structs, the index selects a specific field.
                    // This should be a constant index in well-formed IR.
                    // We handle it by computing the cumulative field offset.
                    // For now, treat as byte offset.
                    let sz = current_ty.size_bytes() as u64;
                    sz
                }
                _ => current_ty.size_bytes() as u64,
            };

            if elem_size == 0 {
                continue;
            }

            // Multiply index by element size.
            let offset_reg = self.alloc_vreg();
            let offset_op = MachineOperand::VirtualReg(offset_reg);

            if elem_size == 1 {
                // No multiplication needed; just add the index.
                let next = self.alloc_vreg();
                let next_op = MachineOperand::VirtualReg(next);
                out.push(Self::make_rrr(RV_ADD, next_op.clone(), current, idx_op));
                current = next_op;
            } else if elem_size.is_power_of_two() {
                // Use shift for power-of-2 element sizes.
                let shift = elem_size.trailing_zeros() as i64;
                out.push(Self::make_rri(RV_SLLI, offset_op.clone(), idx_op, shift));
                let next = self.alloc_vreg();
                let next_op = MachineOperand::VirtualReg(next);
                out.push(Self::make_rrr(RV_ADD, next_op.clone(), current, offset_op));
                current = next_op;
            } else {
                // General case: multiply by loading the size into a register.
                let size_reg = self.alloc_vreg();
                let size_op = MachineOperand::VirtualReg(size_reg);
                self.materialize_immediate_into(elem_size as i64, size_op.clone(), out);
                out.push(Self::make_rrr(RV_MUL, offset_op.clone(), idx_op, size_op));
                let next = self.alloc_vreg();
                let next_op = MachineOperand::VirtualReg(next);
                out.push(Self::make_rrr(RV_ADD, next_op.clone(), current, offset_op));
                current = next_op;
            }
        }

        self.value_map.insert(result, current);
    }
}

// ---------------------------------------------------------------------------
// Cast / conversion instructions
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Selects instructions for integer/floating-point cast operations.
    ///
    /// Handles truncation, zero-extension, sign-extension, and float↔int
    /// conversions. RISC-V lacks dedicated zero-extension or sign-extension
    /// instructions for sub-64-bit values; instead we use shift pairs:
    ///
    /// - **Sign-ext 8→64:** `SLLI rd, rs, 56` then `SRAI rd, rd, 56`
    /// - **Zero-ext 8→64:** `ANDI rd, rs, 0xFF` (for 8-bit)
    /// - **Zero-ext 16→64:** `SLLI rd, rs, 48` then `SRLI rd, rd, 48`
    /// - **Trunc 64→32:** `ADDIW rd, rs, 0` (sign-extends low 32 bits)
    pub fn select_cast(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        kind: CastKind,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);
        let src = self.operand_for_value(value);
        let from_ty = func.get_value_type(value);

        // Float ↔ Float conversions.
        if from_ty.is_floating() && to_ty.is_floating() {
            let opc = match (from_ty, to_ty) {
                (IrType::F32, IrType::F64) => RV_FCVT_D_S,
                (IrType::F64, IrType::F32) => RV_FCVT_S_D,
                _ => {
                    // Same type or F80 — move bits.
                    self.value_map.insert(result, src);
                    return;
                }
            };
            out.push(MachineInstr::with_operands(opc, vec![rd.clone(), src]));
            self.value_map.insert(result, rd);
            return;
        }

        // Float → Int conversions.
        if from_ty.is_floating() && to_ty.is_integer() {
            let to_bits = to_ty.size_bits();
            let is_single = matches!(from_ty, IrType::F32);
            let opc = match (is_single, to_bits > 32) {
                (true, false) => RV_FCVT_W_S,
                (true, true) => RV_FCVT_L_S,
                (false, false) => RV_FCVT_W_D,
                (false, true) => RV_FCVT_L_D,
            };
            out.push(MachineInstr::with_operands(opc, vec![rd.clone(), src]));
            self.value_map.insert(result, rd);
            return;
        }

        // Int → Float conversions.
        if from_ty.is_integer() && to_ty.is_floating() {
            let from_bits = from_ty.size_bits();
            let is_single = matches!(to_ty, IrType::F32);
            let opc = match (is_single, from_bits > 32) {
                (true, false) => RV_FCVT_S_W,
                (true, true) => RV_FCVT_S_L,
                (false, false) => RV_FCVT_D_W,
                (false, true) => RV_FCVT_D_L,
            };
            out.push(MachineInstr::with_operands(opc, vec![rd.clone(), src]));
            self.value_map.insert(result, rd);
            return;
        }

        // Integer ↔ Integer conversions.
        let from_bits = from_ty.size_bits();
        let to_bits = to_ty.size_bits();

        match kind {
            CastKind::Trunc => {
                // Truncation on RV64: for truncating to 32-bit, ADDIW sign-
                // extends the lower 32 bits. For smaller, mask with AND or
                // use shift pair.
                if to_bits == 32 {
                    out.push(Self::make_rri(RV_ADDIW, rd.clone(), src, 0));
                } else if to_bits == 16 {
                    // SLLI rd, src, 48; SRLI rd, rd, 48
                    out.push(Self::make_rri(RV_SLLI, rd.clone(), src, 48));
                    out.push(Self::make_rri(RV_SRLI, rd.clone(), rd.clone(), 48));
                } else if to_bits == 8 {
                    out.push(Self::make_rri(RV_ANDI, rd.clone(), src, 0xFF));
                } else if to_bits == 1 {
                    out.push(Self::make_rri(RV_ANDI, rd.clone(), src, 1));
                } else {
                    // Generic: mask with (1 << to_bits) - 1
                    let mask = (1i64 << to_bits) - 1;
                    if Self::fits_in_simm12(mask) {
                        out.push(Self::make_rri(RV_ANDI, rd.clone(), src, mask));
                    } else {
                        let mask_reg = self.alloc_vreg();
                        let mask_op = MachineOperand::VirtualReg(mask_reg);
                        self.materialize_immediate_into(mask, mask_op.clone(), out);
                        out.push(Self::make_rrr(RV_AND, rd.clone(), src, mask_op));
                    }
                }
            }
            CastKind::ZExt => {
                // Zero-extension: clear upper bits.
                if from_bits == 1 {
                    out.push(Self::make_rri(RV_ANDI, rd.clone(), src, 1));
                } else if from_bits == 8 {
                    out.push(Self::make_rri(RV_ANDI, rd.clone(), src, 0xFF));
                } else if from_bits == 16 {
                    let shift_amt = 64 - from_bits as i64;
                    out.push(Self::make_rri(RV_SLLI, rd.clone(), src, shift_amt));
                    out.push(Self::make_rri(RV_SRLI, rd.clone(), rd.clone(), shift_amt));
                } else if from_bits == 32 {
                    // Zero-extend 32→64: SLLI rd, src, 32; SRLI rd, rd, 32
                    out.push(Self::make_rri(RV_SLLI, rd.clone(), src, 32));
                    out.push(Self::make_rri(RV_SRLI, rd.clone(), rd.clone(), 32));
                } else {
                    // Same width or wider — just move.
                    self.value_map.insert(result, src);
                    return;
                }
            }
            CastKind::SExt => {
                // Sign-extension: replicate the sign bit into upper positions.
                if from_bits == 1 {
                    // Boolean: negate (0→0, 1→-1) then mask if needed.
                    // Or simply: SUB rd, x0, src (NEG)
                    out.push(Self::make_rrr(
                        RV_SUB,
                        rd.clone(),
                        MachineOperand::Register(registers::ZERO),
                        src,
                    ));
                } else if from_bits == 8 {
                    let shift_amt = 56i64;
                    out.push(Self::make_rri(RV_SLLI, rd.clone(), src, shift_amt));
                    out.push(Self::make_rri(RV_SRAI, rd.clone(), rd.clone(), shift_amt));
                } else if from_bits == 16 {
                    let shift_amt = 48i64;
                    out.push(Self::make_rri(RV_SLLI, rd.clone(), src, shift_amt));
                    out.push(Self::make_rri(RV_SRAI, rd.clone(), rd.clone(), shift_amt));
                } else if from_bits == 32 {
                    // ADDIW sign-extends the lower 32 bits on RV64.
                    out.push(Self::make_rri(RV_ADDIW, rd.clone(), src, 0));
                } else {
                    self.value_map.insert(result, src);
                    return;
                }
            }
        }

        self.value_map.insert(result, rd);
    }
}

// ---------------------------------------------------------------------------
// select_phi — Phi-node handling
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Handles PHI nodes during instruction selection.
    ///
    /// At this stage, phi nodes are represented as placeholders. The actual
    /// copy insertion at predecessor block ends is performed by the phi
    /// elimination pass (Phase 9). Here we simply create the virtual register
    /// mapping for the phi result so that downstream instructions can
    /// reference it.
    pub fn select_phi(
        &mut self,
        result: ValueId,
        _ty: &IrType,
        incoming: &[(ValueId, BasicBlockId)],
        out: &mut Vec<MachineInstr>,
    ) {
        let rd = MachineOperand::VirtualReg(result);

        // Emit a PHI-like pseudo-instruction that the phi-elimination pass
        // will replace with copies at predecessor block ends. The operands
        // encode (value, block) pairs.
        let mut operands = Vec::with_capacity(incoming.len() * 2);
        for &(val, bb) in incoming {
            operands.push(self.operand_for_value(val));
            operands.push(MachineOperand::Label(self.block_label(bb)));
        }
        // Prepend the destination.
        let mut all_ops = vec![rd.clone()];
        all_ops.extend(operands);

        // Use opcode 0xFFFF_FFFF as a sentinel for PHI pseudo-instructions.
        // The phi-elimination pass recognizes and removes these.
        let phi_pseudo = 0xFFFF_FFFF_u32;
        out.push(MachineInstr::with_operands(phi_pseudo, all_ops));

        self.value_map.insert(result, rd);
    }
}

// ---------------------------------------------------------------------------
// select_switch — Multi-way branch
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Selects instructions for a switch/multi-way branch.
    ///
    /// Generates a cascaded chain of `BEQ` comparisons for each case value,
    /// with a final unconditional jump to the default block. For very large
    /// switch tables, a jump table could be more efficient, but the cascaded
    /// approach is correct for all sizes and simpler to emit.
    pub fn select_switch(
        &mut self,
        value: ValueId,
        default: BasicBlockId,
        cases: &[(i64, BasicBlockId)],
        out: &mut Vec<MachineInstr>,
    ) {
        let val_op = self.operand_for_value(value);

        for &(case_val, target_bb) in cases {
            let target_label = self.block_label(target_bb);

            // Materialize the case constant into a temporary register.
            let case_reg = self.alloc_vreg();
            let case_op = MachineOperand::VirtualReg(case_reg);
            self.materialize_immediate_into(case_val, case_op.clone(), out);

            // BEQ val, case_reg, target_label
            out.push(Self::make_branch(
                RV_BEQ,
                val_op.clone(),
                case_op,
                target_label,
            ));
        }

        // Fall through to the default block.
        let default_label = self.block_label(default);
        let mut jmp = MachineInstr::with_operands(
            RV_JAL,
            vec![
                MachineOperand::Register(registers::ZERO),
                MachineOperand::Label(default_label),
            ],
        );
        jmp.is_terminator = true;
        out.push(jmp);
    }
}

// ---------------------------------------------------------------------------
// Function call lowering and return
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Selects instructions for a function call.
    ///
    /// Implements the RISC-V LP64D calling convention:
    ///
    /// 1. Move integer arguments into a0–a7, float arguments into fa0–fa7.
    /// 2. Spill excess arguments onto the stack.
    /// 3. Emit `JAL` (direct) or `JALR` (indirect) to invoke the callee.
    /// 4. Move the return value from a0/fa0 into the result virtual register.
    pub fn select_call(
        &mut self,
        result: Option<ValueId>,
        callee: ValueId,
        args: &[ValueId],
        is_tail: bool,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        self.has_calls = true;

        let mut int_idx: usize = 0;
        let mut fp_idx: usize = 0;
        let mut stack_args: Vec<(MachineOperand, i64)> = Vec::new();
        let mut stack_offset: i64 = 0;

        // Step 1: Classify and place arguments.
        for &arg_val in args {
            let arg_op = self.operand_for_value(arg_val);
            let arg_ty = func.get_value_type(arg_val);

            if Self::is_fp_type(arg_ty) {
                if fp_idx < registers::FLOAT_ARG_REGS.len() {
                    let reg = registers::FLOAT_ARG_REGS[fp_idx];
                    let mov_opc = if matches!(arg_ty, IrType::F32) {
                        RV_FMOV_S
                    } else {
                        RV_FMOV_D
                    };
                    out.push(MachineInstr::with_operands(
                        mov_opc,
                        vec![MachineOperand::Register(reg), arg_op],
                    ));
                    fp_idx += 1;
                } else if int_idx < registers::INTEGER_ARG_REGS.len() {
                    // Out of FP regs — pass in integer register (bit-moved).
                    let reg = registers::INTEGER_ARG_REGS[int_idx];
                    let mov_opc = if matches!(arg_ty, IrType::F32) {
                        RV_FMV_X_W
                    } else {
                        RV_FMV_X_D
                    };
                    out.push(MachineInstr::with_operands(
                        mov_opc,
                        vec![MachineOperand::Register(reg), arg_op],
                    ));
                    int_idx += 1;
                } else {
                    stack_args.push((arg_op, stack_offset));
                    stack_offset += 8;
                }
            } else {
                if int_idx < registers::INTEGER_ARG_REGS.len() {
                    let reg = registers::INTEGER_ARG_REGS[int_idx];
                    out.push(MachineInstr::with_operands(
                        RV_MV,
                        vec![MachineOperand::Register(reg), arg_op],
                    ));
                    int_idx += 1;
                } else {
                    stack_args.push((arg_op, stack_offset));
                    stack_offset += 8;
                }
            }
        }

        // Step 2: Spill excess arguments onto the stack.
        if !stack_args.is_empty() {
            // Adjust SP for the stack argument area (must be 16-byte aligned).
            let aligned_stack = ((stack_offset + 15) & !15) as i64;
            out.push(Self::make_rri(
                RV_ADDI,
                MachineOperand::Register(registers::SP),
                MachineOperand::Register(registers::SP),
                -aligned_stack,
            ));

            for (arg_op, off) in &stack_args {
                out.push(Self::make_store(
                    RV_SD,
                    arg_op.clone(),
                    MachineOperand::Register(registers::SP),
                    *off,
                ));
            }
        }

        // Step 3: Emit the call instruction.
        let callee_op = self.operand_for_value(callee);
        let call_opcode = if is_tail { RV_TAIL } else { RV_CALL };

        match &callee_op {
            MachineOperand::Symbol(name) => {
                // Direct call — use JAL or CALL pseudo.
                if self.pic_enabled {
                    // PIC: AUIPC t1, %got_pcrel_hi(sym); JALR ra, t1
                    let t1 = MachineOperand::Register(registers::T1);
                    out.push(MachineInstr::with_operands(
                        RV_AUIPC,
                        vec![t1.clone(), MachineOperand::Symbol(name.clone())],
                    ));
                    let mut jalr = MachineInstr::with_operands(
                        RV_JALR,
                        vec![
                            MachineOperand::Register(registers::RA),
                            t1,
                            MachineOperand::Immediate(0),
                        ],
                    );
                    jalr.is_call = true;
                    out.push(jalr);
                } else {
                    let mut call_instr = MachineInstr::with_operands(
                        call_opcode,
                        vec![MachineOperand::Symbol(name.clone())],
                    );
                    call_instr.is_call = true;
                    out.push(call_instr);
                }
            }
            _ => {
                // Indirect call through function pointer in a register.
                // Move pointer to t1 if not already a register, then JALR.
                let ptr_reg = match &callee_op {
                    MachineOperand::Register(_) => callee_op.clone(),
                    _ => {
                        let t = MachineOperand::Register(registers::T1);
                        out.push(MachineInstr::with_operands(
                            RV_MV,
                            vec![t.clone(), callee_op.clone()],
                        ));
                        t
                    }
                };
                let mut jalr = MachineInstr::with_operands(
                    RV_JALR,
                    vec![
                        MachineOperand::Register(registers::RA),
                        ptr_reg,
                        MachineOperand::Immediate(0),
                    ],
                );
                jalr.is_call = true;
                out.push(jalr);
            }
        }

        // Step 4: Mark caller-saved registers as implicitly defined (clobbered)
        // by the call. This is necessary for correct register allocation — the
        // allocator must know which registers are destroyed across the call.
        if let Some(last) = out.last_mut() {
            if last.is_call {
                for &reg in registers::CALLER_SAVED_INT.iter() {
                    last.implicit_defs.push(reg);
                }
                for &reg in registers::CALLER_SAVED_FP.iter() {
                    last.implicit_defs.push(reg);
                }
            }
        }

        // Step 5: Restore SP if stack arguments were placed.
        if !stack_args.is_empty() {
            let aligned_stack = ((stack_offset + 15) & !15) as i64;
            out.push(Self::make_rri(
                RV_ADDI,
                MachineOperand::Register(registers::SP),
                MachineOperand::Register(registers::SP),
                aligned_stack,
            ));
        }

        // Step 6: Move return value into the result register.
        if let Some(res) = result {
            let rd = MachineOperand::VirtualReg(res);
            // Determine return type. Default to integer register (a0).
            let ret_ty = func.get_value_type(res);
            if Self::is_fp_type(ret_ty) {
                let mov_opc = if matches!(ret_ty, IrType::F32) {
                    RV_FMOV_S
                } else {
                    RV_FMOV_D
                };
                out.push(MachineInstr::with_operands(
                    mov_opc,
                    vec![rd.clone(), MachineOperand::Register(registers::FA0)],
                ));
            } else {
                out.push(MachineInstr::with_operands(
                    RV_MV,
                    vec![rd.clone(), MachineOperand::Register(registers::A0)],
                ));
            }
            self.value_map.insert(res, rd);
        }
    }

    /// Selects instructions for a function return.
    ///
    /// If a return value is present, moves it into a0 (integer) or fa0
    /// (floating-point), then emits the RET pseudo-instruction (JALR x0, ra, 0).
    pub fn select_return(
        &mut self,
        value: Option<ValueId>,
        func: &IrFunction,
        out: &mut Vec<MachineInstr>,
    ) {
        if let Some(val) = value {
            let src = self.operand_for_value(val);
            let val_ty = func.get_value_type(val);

            if Self::is_fp_type(val_ty) {
                let mov_opc = if matches!(val_ty, IrType::F32) {
                    RV_FMOV_S
                } else {
                    RV_FMOV_D
                };
                out.push(MachineInstr::with_operands(
                    mov_opc,
                    vec![MachineOperand::Register(registers::FA0), src],
                ));
            } else {
                out.push(MachineInstr::with_operands(
                    RV_MV,
                    vec![MachineOperand::Register(registers::A0), src],
                ));
            }
        }

        // Emit RET (pseudo for JALR x0, ra, 0).
        let mut ret = MachineInstr::with_operands(RV_RET, vec![]);
        ret.is_terminator = true;
        ret.is_return = true;
        ret.implicit_uses.push(registers::RA);
        out.push(ret);
    }

    /// Selects instructions for inline assembly.
    ///
    /// Inline assembly is passed through as a special pseudo-instruction
    /// containing the template string and operand bindings. The assembler
    /// module processes these during final emission.
    pub fn select_inline_asm(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
        has_side_effects: bool,
        out: &mut Vec<MachineInstr>,
    ) {
        let mut ops: Vec<MachineOperand> = Vec::new();

        // If there is a result, the first operand is the destination.
        if let Some(res) = result {
            ops.push(MachineOperand::VirtualReg(res));
        }

        // Map input operands.
        for &op_val in operands {
            ops.push(self.operand_for_value(op_val));
        }

        // Encode the template and constraints as Symbol operands so the
        // assembler phase can extract them from the instruction.
        ops.push(MachineOperand::Symbol(template.to_string()));
        if !constraints.is_empty() {
            ops.push(MachineOperand::Symbol(format!("constraints:{}", constraints)));
        }

        let mut instr = MachineInstr::with_operands(RV_INLINE_ASM, ops);

        // Mark clobbered registers as implicit defs.
        for clobber in clobbers {
            match clobber.as_str() {
                "memory" => { /* memory clobber — barrier only, no reg */ }
                "cc" => { /* condition codes — not a RISC-V register */ }
                name => {
                    // Try to resolve the clobber name to a physical register.
                    // Common names: "ra", "t0", "a0", etc.
                    if let Some(reg) = clobber_name_to_reg(name) {
                        // Verify the register encoding is valid (0..31 for
                        // both integer and float register files).
                        debug_assert!(
                            registers::encoding(reg) < 32,
                            "Clobber register has invalid encoding: {}",
                            registers::reg_name(reg),
                        );
                        instr.implicit_defs.push(reg);
                    }
                }
            }
        }

        if has_side_effects {
            // Prevent reordering across this instruction.
            instr.is_call = true;
        }

        out.push(instr);

        // Map the result, if any.
        if let Some(res) = result {
            self.value_map
                .insert(res, MachineOperand::VirtualReg(res));
        }
    }
}

/// Resolves an inline-assembly clobber name to a physical RISC-V register.
///
/// Accepts both ABI names (e.g., "a0", "ra", "sp") and register numbers
/// (e.g., "x1", "x10"). Returns `None` for unrecognized names or special
/// clobbers like "memory" / "cc".
///
/// After resolution, [`registers::encoding`] can be used to obtain the 5-bit
/// hardware encoding of the resolved register for binary instruction emission.
fn clobber_name_to_reg(name: &str) -> Option<PhysReg> {
    match name {
        "zero" | "x0" => Some(registers::ZERO),
        "ra" | "x1" => Some(registers::RA),
        "sp" | "x2" => Some(registers::SP),
        "gp" | "x3" => Some(registers::GP),
        "tp" | "x4" => Some(registers::TP),
        "t0" | "x5" => Some(registers::T0),
        "t1" | "x6" => Some(registers::T1),
        "t2" | "x7" => Some(registers::T2),
        "s0" | "fp" | "x8" => Some(registers::S0),
        "s1" | "x9" => Some(registers::S1),
        "a0" | "x10" => Some(registers::A0),
        "a1" | "x11" => Some(registers::A1),
        "a2" | "x12" => Some(registers::A2),
        "a3" | "x13" => Some(registers::A3),
        "a4" | "x14" => Some(registers::A4),
        "a5" | "x15" => Some(registers::A5),
        "a6" | "x16" => Some(registers::A6),
        "a7" | "x17" => Some(registers::A7),
        "s2" | "x18" => Some(registers::S2),
        "s3" | "x19" => Some(registers::S3),
        "s4" | "x20" => Some(registers::S4),
        "s5" | "x21" => Some(registers::S5),
        "s6" | "x22" => Some(registers::S6),
        "s7" | "x23" => Some(registers::S7),
        "s8" | "x24" => Some(registers::S8),
        "s9" | "x25" => Some(registers::S9),
        "s10" | "x26" => Some(registers::S10),
        "s11" | "x27" => Some(registers::S11),
        "t3" | "x28" => Some(registers::T3),
        "t4" | "x29" => Some(registers::T4),
        "t5" | "x30" => Some(registers::T5),
        "t6" | "x31" => Some(registers::T6),
        "fa0" => Some(registers::FA0),
        "fa1" => Some(registers::FA1),
        "fa2" => Some(registers::FA2),
        "fa3" => Some(registers::FA3),
        "fa4" => Some(registers::FA4),
        "fa5" => Some(registers::FA5),
        "fa6" => Some(registers::FA6),
        "fa7" => Some(registers::FA7),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Prologue and epilogue emission
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Emits the function prologue into the entry (first) machine block.
    ///
    /// The RISC-V standard prologue:
    ///
    /// ```asm
    /// addi  sp, sp, -framesize    # allocate stack frame
    /// sd    ra, framesize-8(sp)   # save return address (if has_calls)
    /// sd    s0, framesize-16(sp)  # save frame pointer
    /// sd    s1, ...               # save callee-saved registers
    /// addi  s0, sp, framesize     # set up frame pointer
    /// ```
    ///
    /// All offsets are computed relative to the adjusted SP after allocation.
    pub fn emit_prologue(&self, mf: &mut MachineFunction, frame_size: u32) {
        if frame_size == 0 && self.used_callee_saved.is_empty() && !self.has_calls {
            return; // Leaf function with no locals — no prologue needed.
        }

        let mut prologue: Vec<MachineInstr> = Vec::new();
        let sp = MachineOperand::Register(registers::SP);
        let fs = frame_size as i64;

        // Allocate stack frame. If framesize > 2047, use a temporary.
        if Self::fits_in_simm12(-fs) {
            prologue.push(Self::make_rri(RV_ADDI, sp.clone(), sp.clone(), -fs));
        } else {
            // Materialize -framesize into t0, then ADD sp, sp, t0.
            let t0 = MachineOperand::Register(registers::T0);
            Self::materialize_immediate_static(-fs, t0.clone(), &mut prologue);
            prologue.push(Self::make_rrr(RV_ADD, sp.clone(), sp.clone(), t0));
        }

        // Save return address if function makes calls.
        let mut save_offset = fs - 8;
        if self.has_calls {
            prologue.push(Self::make_store(
                RV_SD,
                MachineOperand::Register(registers::RA),
                sp.clone(),
                save_offset,
            ));
            save_offset -= 8;
        }

        // Save callee-saved registers.
        for &reg in &self.used_callee_saved {
            prologue.push(Self::make_store(
                RV_SD,
                MachineOperand::Register(reg),
                sp.clone(),
                save_offset,
            ));
            save_offset -= 8;
        }

        // Set up frame pointer: s0 = sp + framesize
        // This allows FP-relative access to both locals and spilled args.
        if !self.used_callee_saved.is_empty() || self.has_calls || self.frame_locals_size > 0 {
            let fp = MachineOperand::Register(registers::FP);
            if Self::fits_in_simm12(fs) {
                prologue.push(Self::make_rri(RV_ADDI, fp, sp.clone(), fs));
            } else {
                let t0 = MachineOperand::Register(registers::T0);
                Self::materialize_immediate_static(fs, t0.clone(), &mut prologue);
                prologue.push(Self::make_rrr(RV_ADD, fp, sp.clone(), t0));
            }
        }

        // Prepend prologue to the first block.
        if let Some(entry) = mf.blocks.first_mut() {
            let mut combined = prologue;
            combined.append(&mut entry.instructions);
            entry.instructions = combined;
        }
    }

    /// Emits the function epilogue before every return instruction.
    ///
    /// The RISC-V standard epilogue:
    ///
    /// ```asm
    /// ld    s11, ...(sp)          # restore callee-saved registers
    /// ld    s0, framesize-16(sp)  # restore frame pointer
    /// ld    ra, framesize-8(sp)   # restore return address
    /// addi  sp, sp, framesize     # deallocate stack frame
    /// ret
    /// ```
    pub fn emit_epilogue(&self, mf: &mut MachineFunction, frame_size: u32) {
        if frame_size == 0 && self.used_callee_saved.is_empty() && !self.has_calls {
            return;
        }

        let sp = MachineOperand::Register(registers::SP);
        let fs = frame_size as i64;

        // Find all return instructions and insert epilogue before each one.
        for block in &mut mf.blocks {
            let mut new_instrs: Vec<MachineInstr> = Vec::new();
            for instr in block.instructions.drain(..) {
                if instr.is_return {
                    // Restore callee-saved registers in reverse order.
                    let mut restore_offset = fs - 8;
                    if self.has_calls {
                        restore_offset -= 8; // Skip to callee-saved area
                    }

                    // Restore callee-saved registers (reverse order).
                    for (i, &reg) in self.used_callee_saved.iter().enumerate() {
                        let off = fs - 8 - (if self.has_calls { 8 } else { 0 })
                            - (i as i64) * 8;
                        new_instrs.push(Self::make_load(
                            RV_LD,
                            MachineOperand::Register(reg),
                            sp.clone(),
                            off,
                        ));
                    }

                    // Restore return address.
                    if self.has_calls {
                        new_instrs.push(Self::make_load(
                            RV_LD,
                            MachineOperand::Register(registers::RA),
                            sp.clone(),
                            fs - 8,
                        ));
                    }

                    // Deallocate stack frame.
                    if Self::fits_in_simm12(fs) {
                        new_instrs.push(Self::make_rri(
                            RV_ADDI,
                            sp.clone(),
                            sp.clone(),
                            fs,
                        ));
                    } else {
                        let t0 = MachineOperand::Register(registers::T0);
                        Self::materialize_immediate_static(fs, t0.clone(), &mut new_instrs);
                        new_instrs.push(Self::make_rrr(
                            RV_ADD,
                            sp.clone(),
                            sp.clone(),
                            t0,
                        ));
                    }

                    // Finally, the actual return instruction.
                    new_instrs.push(instr);
                } else {
                    new_instrs.push(instr);
                }
            }
            block.instructions = new_instrs;
        }
    }
}

// ---------------------------------------------------------------------------
// Immediate materialization and PIC address generation
// ---------------------------------------------------------------------------

impl RiscV64InstrSel {
    /// Materializes a 64-bit immediate value into a virtual register.
    ///
    /// This is the public API for immediate materialization. Returns the
    /// machine operand holding the materialized value.
    ///
    /// Strategy selection based on value range:
    ///
    /// | Range                           | Sequence                          |
    /// |---------------------------------|-----------------------------------|
    /// | [-2048, 2047]                   | `ADDI rd, x0, imm`               |
    /// | Fits in upper 20 + lower 12     | `LUI rd, hi20` + `ADDI rd, lo12` |
    /// | Full 64-bit                     | Multi-step shift+add sequence     |
    pub fn materialize_immediate(&mut self, value: i64, out: &mut Vec<MachineInstr>) -> MachineOperand {
        let rd_id = self.alloc_vreg();
        let rd = MachineOperand::VirtualReg(rd_id);
        self.materialize_immediate_into(value, rd.clone(), out);
        rd
    }

    /// Materializes a 64-bit immediate into the specified destination operand.
    fn materialize_immediate_into(
        &mut self,
        value: i64,
        rd: MachineOperand,
        out: &mut Vec<MachineInstr>,
    ) {
        Self::materialize_immediate_static(value, rd, out);
    }

    /// Static version of immediate materialization (no `&mut self` needed).
    /// Used by prologue/epilogue emission which holds an immutable reference
    /// to `self`.
    fn materialize_immediate_static(
        value: i64,
        rd: MachineOperand,
        out: &mut Vec<MachineInstr>,
    ) {
        let zero = MachineOperand::Register(registers::ZERO);

        // Case 1: Fits in signed 12-bit immediate.
        if value >= -2048 && value <= 2047 {
            out.push(Self::make_rri(RV_ADDI, rd, zero, value));
            return;
        }

        // Case 2: Fits in 32-bit signed range.
        let value_32 = value as i32;
        if value as i64 == value_32 as i64 {
            // LUI loads bits [31:12] with sign-extension to 64 bits.
            let lo12 = ((value_32 as u32) & 0xFFF) as i32;
            let mut hi20 = ((value_32 as u32) >> 12) as i32;
            // If low 12 bits are negative (bit 11 set), add 1 to high to
            // compensate for the sign-extension of ADDI.
            if lo12 >= 0x800_i32 {
                hi20 = hi20.wrapping_add(1);
            }
            let lo12_signed = if lo12 >= 0x800 {
                lo12 - 0x1000
            } else {
                lo12
            };

            out.push(MachineInstr::with_operands(
                RV_LUI,
                vec![rd.clone(), MachineOperand::Immediate(hi20 as i64)],
            ));
            if lo12_signed != 0 {
                out.push(Self::make_rri(RV_ADDI, rd.clone(), rd, lo12_signed as i64));
            }
            return;
        }

        // Case 3: Full 64-bit constant. Build from the top down using
        // shifts and adds. We split the 64-bit value into chunks that fit
        // in 12-bit signed immediates combined with shifts.
        //
        // Algorithm:
        // 1. Materialize the high 32 bits.
        // 2. SLLI by 32.
        // 3. Add in the low 32 bits (using LUI+ADDI for the low portion).
        let hi32 = (value >> 32) as i32;
        let lo32 = value as i32;

        // Materialize high 32 bits.
        let hi_lo12 = ((hi32 as u32) & 0xFFF) as i32;
        let mut hi_hi20 = ((hi32 as u32) >> 12) as i32;
        if hi_lo12 >= 0x800 {
            hi_hi20 = hi_hi20.wrapping_add(1);
        }
        let hi_lo12_signed = if hi_lo12 >= 0x800 {
            hi_lo12 - 0x1000
        } else {
            hi_lo12
        };

        if hi_hi20 != 0 {
            out.push(MachineInstr::with_operands(
                RV_LUI,
                vec![rd.clone(), MachineOperand::Immediate(hi_hi20 as i64)],
            ));
            if hi_lo12_signed != 0 {
                out.push(Self::make_rri(
                    RV_ADDI,
                    rd.clone(),
                    rd.clone(),
                    hi_lo12_signed as i64,
                ));
            }
        } else {
            // High part fits in 12 bits.
            out.push(Self::make_rri(
                RV_ADDI,
                rd.clone(),
                zero.clone(),
                hi_lo12_signed as i64,
            ));
        }

        // Shift left by 32 to position the high bits.
        out.push(Self::make_rri(RV_SLLI, rd.clone(), rd.clone(), 32));

        // Add the low 32 bits. Similar LUI+ADDI decomposition.
        let lo_lo12 = ((lo32 as u32) & 0xFFF) as i32;
        let mut lo_hi20 = ((lo32 as u32) >> 12) as i32;
        if lo_lo12 >= 0x800 {
            lo_hi20 = lo_hi20.wrapping_add(1);
        }
        let lo_lo12_signed = if lo_lo12 >= 0x800 {
            lo_lo12 - 0x1000
        } else {
            lo_lo12
        };

        if lo_hi20 != 0 {
            // We need a temporary to hold the LUI result, then OR/ADD it in.
            // Use SLLI+ADDI on rd itself:
            // rd already has high bits. We need to add (lo_hi20 << 12) + lo_lo12.
            //
            // Approach: SLLI rd, rd, 12; ADDI rd, rd, lo_hi20-chunk; repeat
            // But this is complex. Simpler: use a second temp if needed.
            //
            // Actually, since rd already has high32 << 32, we can:
            // 1. If lo_hi20 != 0: shift rd left by 12 more (total 44),
            //    ADDI with hi20-portion of low, shift back... This gets messy.
            //
            // Alternative: build low32 in pieces and OR them in.
            // Split the remaining SLLI into two parts.
            // First, add upper 20 bits of lo32:
            //   SLLI rd, rd, 12 (from 32 -> was already 32, we overwrite)
            // No — rd has value << 32 already. Just add lo32 directly.
            //
            // Simplest correct approach for arbitrary lo32:
            // Since we need to combine hi32<<32 with lo32 as unsigned:
            //   After SLLI 32, rd = hi32 << 32 (as i64)
            //   We want rd | (lo32 as u32 as u64).
            //
            // For the low portion, shift right approach won't work because of
            // sign extension. Instead, break the SLLI 32 into smaller shifts
            // and interleave ADDI operations.
            //
            // Better 64-bit materialization using iterative shift+add:

            // Undo the shift-32 above — we'll redo with a better sequence.
            // Actually, let's just use the simpler approach: materialize the
            // full value iteratively. Pop the last SLLI we added and redo.
            out.pop(); // Remove the SLLI 32

            // Rebuild using the standard approach for 64-bit:
            // Start with the high portion already in rd.
            // Now shift and add chunks from MSB to LSB.

            // We have hi_part in rd. Now shift in the upper portion of lo32.
            // Split lo32 into: bits [31:20], bits [19:8], bits [7:0]
            // to keep each chunk under 12 bits for ADDI.

            let chunk2 = ((lo32 as u32 >> 20) & 0xFFF) as i64;
            let chunk1 = ((lo32 as u32 >> 8) & 0xFFF) as i64;
            let chunk0 = ((lo32 as u32) & 0xFF) as i64;

            // Shift hi-part by 12 and add chunk2
            out.push(Self::make_rri(RV_SLLI, rd.clone(), rd.clone(), 12));
            if chunk2 != 0 {
                out.push(Self::make_rri(RV_ADDI, rd.clone(), rd.clone(), chunk2));
            }

            // Shift by 12 and add chunk1
            out.push(Self::make_rri(RV_SLLI, rd.clone(), rd.clone(), 12));
            if chunk1 != 0 {
                out.push(Self::make_rri(RV_ADDI, rd.clone(), rd.clone(), chunk1));
            }

            // Shift by 8 and add chunk0
            out.push(Self::make_rri(RV_SLLI, rd.clone(), rd.clone(), 8));
            if chunk0 != 0 {
                out.push(Self::make_rri(RV_ADDI, rd.clone(), rd.clone(), chunk0));
            }
        } else if lo_lo12_signed != 0 {
            // Low 32 bits fit in 12 bits. Just ADDI.
            out.push(Self::make_rri(
                RV_ADDI,
                rd.clone(),
                rd,
                lo_lo12_signed as i64,
            ));
        }
    }

    /// Generates a PIC-mode address for a global symbol.
    ///
    /// For PIC code, global variable access uses:
    /// ```asm
    /// auipc  rd, %got_pcrel_hi(symbol)
    /// ld     rd, %pcrel_lo(.Ltmp)(rd)     # load GOT entry
    /// ```
    ///
    /// For non-PIC code:
    /// ```asm
    /// lui    rd, %hi(symbol)
    /// addi   rd, rd, %lo(symbol)
    /// ```
    pub fn generate_pic_address(
        &mut self,
        symbol: &str,
        out: &mut Vec<MachineInstr>,
    ) -> MachineOperand {
        let rd_id = self.alloc_vreg();
        let rd = MachineOperand::VirtualReg(rd_id);

        if self.pic_enabled {
            // PIC addressing: AUIPC + LD through GOT.
            out.push(MachineInstr::with_operands(
                RV_AUIPC,
                vec![rd.clone(), MachineOperand::Symbol(symbol.to_string())],
            ));
            out.push(Self::make_load(
                RV_LD,
                rd.clone(),
                rd.clone(),
                0, // Relocation will fill in the actual GOT offset.
            ));
        } else {
            // Absolute addressing: LUI + ADDI.
            out.push(MachineInstr::with_operands(
                RV_LUI,
                vec![rd.clone(), MachineOperand::Symbol(symbol.to_string())],
            ));
            out.push(Self::make_rri(RV_ADDI, rd.clone(), rd.clone(), 0));
        }

        rd
    }
}
