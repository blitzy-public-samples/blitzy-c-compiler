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
    invert_condition, is_callee_saved, v_to_d, v_to_s, CALLER_SAVED_INT, COND_CC, COND_CS, COND_EQ,
    COND_GE, COND_GT, COND_HI, COND_LE, COND_LS, COND_LT, COND_MI, COND_NE, COND_PL, COND_VC,
    COND_VS, FLOAT_ARG_REGS, FP, INTEGER_ARG_REGS, LR, SP, V0, V1, V2, W0, W1, W10, W11, W12, W13,
    W14, W15, W16, W17, W19, W2, W20, W21, W3, W4, W5, W6, W7, W8, W9, WZR, X0, X1, X10, X11, X12,
    X13, X14, X15, X16, X17, X18, X19, X2, X20, X21, X22, X23, X24, X25, X26, X27, X28, X3, X4, X5,
    X6, X7, X8, X9, XZR,
};
use crate::backend::traits::{MachineFunction, MachineInstr, MachineOperand, PhysReg};
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
    // --- Data processing (register-register) --- range 0x0100..=0x01FF
    ADD = 0x0100,  // OP_ADD_REG
    SUB = 0x0101,  // OP_SUB_REG
    ADDS = 0x0102, // OP_ADDS_REG
    SUBS = 0x0103, // OP_SUBS_REG
    AND = 0x0104,  // OP_AND_REG
    ORR = 0x0105,  // OP_ORR_REG
    EOR = 0x0106,  // OP_EOR_REG
    ORN = 0x0107,  // OP_ORN_REG
    BIC = 0x0108,  // OP_BIC_REG
    MADD = 0x0109, // OP_MADD
    MSUB = 0x010A, // OP_MSUB
    SDIV = 0x010D, // OP_SDIV
    UDIV = 0x010E, // OP_UDIV
    LSL = 0x010F,  // OP_LSLV (register shift)
    LSR = 0x0110,  // OP_LSRV (register shift)
    ASR = 0x0111,  // OP_ASRV (register shift)
    ROR = 0x0112,  // OP_RORV (register shift)

    // --- Data processing (immediate) --- range 0x0000..=0x00FF
    ADDimm = 0x0001, // OP_ADD_IMM
    SUBimm = 0x0002, // OP_SUB_IMM
    ANDimm = 0x0005, // OP_AND_IMM
    ORRimm = 0x0006, // OP_ORR_IMM
    EORimm = 0x0007, // OP_EOR_IMM
    MOVZ = 0x0009,   // OP_MOVZ
    MOVK = 0x000A,   // OP_MOVK
    MOVN = 0x000B,   // OP_MOVN

    // --- PC-relative addressing ---
    ADRP = 0x0010, // OP_ADRP
    ADR = 0x000F,  // OP_ADR

    // --- Memory operations --- range 0x0200..=0x02FF
    LDR = 0x0200,     // OP_LDR_IMM
    STR = 0x0201,     // OP_STR_IMM
    LDRB = 0x0202,    // OP_LDRB_IMM
    LDRH = 0x0203,    // OP_LDRH_IMM
    LDRSB = 0x0204,   // OP_LDRSB_IMM
    LDRSH = 0x0205,   // OP_LDRSH_IMM
    LDRSW = 0x0206,   // OP_LDRSW_IMM
    STRB = 0x0207,    // OP_STRB_IMM
    STRH = 0x0208,    // OP_STRH_IMM
    LDP = 0x020F,     // OP_LDP (signed offset)
    STP = 0x0210,     // OP_STP (signed offset)
    LdpPre = 0x0211,  // OP_LDP_PRE (pre-indexed)
    StpPre = 0x0212,  // OP_STP_PRE (pre-indexed)
    LdpPost = 0x0213, // OP_LDP_POST (post-indexed)
    StpPost = 0x0214, // OP_STP_POST (post-indexed)
    LDRlit = 0x0215,  // OP_LDR_LITERAL
    /// 32-bit word load — forces `size=0b10` in the encoder regardless
    /// of the physical register name so that I32 values are loaded with
    /// the correct 4-byte transfer size (zero-extended to 64 bits).
    LDRW = 0x021A, // OP_LDR_W32
    /// 32-bit word store — forces `size=0b10` in the encoder.
    STRW = 0x021B, // OP_STR_W32

    // --- Branches --- range 0x0300..=0x037F
    B = 0x0300,     // OP_B
    BL = 0x0301,    // OP_BL
    Bcond = 0x0302, // OP_B_COND
    CBZ = 0x0303,   // OP_CBZ
    CBNZ = 0x0304,  // OP_CBNZ
    TBZ = 0x0305,   // OP_TBZ
    TBNZ = 0x0306,  // OP_TBNZ
    BR = 0x0307,    // OP_BR
    BLR = 0x0308,   // OP_BLR
    RET = 0x0309,   // OP_RET

    // --- Comparison / Conditional --- range 0x0380..=0x03FF
    CMP = 0x0386,   // pseudo: SUBS with XZR dest (new OP_CMP_REG)
    CMN = 0x0387,   // pseudo: ADDS with XZR dest (new OP_CMN_REG)
    TST = 0x0388,   // pseudo: ANDS with XZR dest (new OP_TST_REG)
    CCMP = 0x0380,  // OP_CCMP_REG
    CCMN = 0x0389,  // new OP_CCMN_REG
    CSEL = 0x0382,  // OP_CSEL
    CSINC = 0x0383, // OP_CSINC
    CSINV = 0x0384, // OP_CSINV
    CSNEG = 0x0385, // OP_CSNEG

    // --- Floating-point / SIMD scalar --- range 0x0400..=0x04FF
    FADD = 0x0400,      // OP_FADD
    FSUB = 0x0401,      // OP_FSUB
    FMUL = 0x0402,      // OP_FMUL
    FDIV = 0x0403,      // OP_FDIV
    FNEG = 0x0404,      // OP_FNEG
    FABS = 0x0405,      // OP_FABS
    FSQRT = 0x0406,     // OP_FSQRT
    FCMP = 0x0407,      // OP_FCMP
    FMOV = 0x0409,      // OP_FMOV_REG
    FMOVint = 0x040B,   // OP_FMOV_FROM_GPR  (GPR→FP)
    FMOVtoGPR = 0x040A, // OP_FMOV_TO_GPR (FP→GPR)
    FCCMP = 0x0412,     // new OP_FCCMP
    SCVTF = 0x040D,     // OP_SCVTF
    UCVTF = 0x040E,     // OP_UCVTF
    FCVTZS = 0x040F,    // OP_FCVTZS
    FCVTZU = 0x0410,    // OP_FCVTZU
    FCVT = 0x0411,      // OP_FCVT

    // --- Type extension (mapped to SBFM/UBFM) --- range 0x0000..=0x00FF
    SXTB = 0x0011, // new: SBFM alias (imms=7)
    SXTH = 0x0012, // new: SBFM alias (imms=15)
    SXTW = 0x0013, // new: SBFM alias (imms=31)
    UXTB = 0x0014, // new: UBFM alias (imms=7)
    UXTH = 0x0015, // new: UBFM alias (imms=15)

    // --- Miscellaneous / System --- range 0x0500..=0x05FF
    NOP = 0x0500, // OP_NOP
    // BRK = 0x0501, SVC = 0x0502, DMB = 0x0503, DSB = 0x0504,
    // ISB = 0x0505, MRS = 0x0506, MSR = 0x0507 (defined in encoder.rs)

    // --- Bit manipulation ---
    CLZ = 0x0508,       // OP_CLZ  (Count Leading Zeros)
    RBIT = 0x0509,      // OP_RBIT (Reverse Bits)
    REV = 0x050A,       // OP_REV  (Byte Reverse)
    REV16 = 0x050B,     // OP_REV16
    InlineAsm = 0x050C, // Pseudo: raw inline assembly
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
    /// Whether the current function is variadic (accepts `...` args).
    /// When true, the prologue spills X0-X7 into a 64-byte register
    /// save area at [SP+16..SP+80], and frame layout computations
    /// account for this extra region.
    is_variadic_func: bool,
    /// Tracks the IR type associated with each `ValueId`.
    /// Populated during instruction selection to support type-aware
    /// operations (e.g. sign-extending I32 values before signed
    /// comparisons on 64-bit registers).
    value_types: FxHashMap<ValueId, IrType>,
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
            is_variadic_func: false,
            value_types: FxHashMap::default(),
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
        self.value_types.clear();
        self.current_frame_offset = 0;
        self.has_calls = false;
        self.is_variadic_func = func.is_variadic;
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

        // Phase 1a-extra: Collect IR types for every ValueId.
        //
        // We scan all instructions to record the type each result value
        // has in the IR.  This is used by lower_icmp to determine whether
        // operands are I32 (and thus need sign-extension on 64-bit
        // registers for signed comparisons) vs I64/Ptr.
        for bb in func.blocks() {
            for instr in bb.instructions() {
                match instr {
                    Instruction::Alloca { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::Load { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BinOp { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::ICmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::FCmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::Call { result, .. } => {
                        if let Some(r) = result {
                            self.value_types.insert(*r, IrType::I64);
                        }
                    }
                    Instruction::Phi { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::GetElementPtr { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::BitCast { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::Trunc { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::ZExt { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::SExt { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::IntToPtr { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::PtrToInt { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::SIToFP { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::UIToFP { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::FPToSI { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::FPToUI { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::FPExt { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    Instruction::FPTrunc { result, to_ty, .. } => {
                        self.value_types.insert(*result, to_ty.clone());
                    }
                    _ => {}
                }
            }
        }
        // Also record types for function parameters.
        for p in &func.params {
            self.value_types.insert(p.id, p.ty.clone());
        }

        // Phase 1b: Pre-populate value_map for special IR values.
        //
        // The IR builder encodes certain value kinds (global references,
        // integer constants, float constants, null pointers) purely by
        // naming convention in the ValueInfo table rather than emitting
        // dedicated instructions.  The codegen must recognize these names
        // and map them to the appropriate MachineOperand *before*
        // instruction selection begins — otherwise resolve_operand()
        // returns a VirtualReg which produces incorrect code (e.g.
        // `blr x0` instead of `bl <symbol>`).
        //
        // Naming conventions (see ir::builder):
        //   "global.<name>"        → Symbol("<name>")  — external function / global var
        //   "const.int.<value>"    → Immediate(<value>) — integer constant
        //   "const.float.<value>"  → Immediate(f64 bits) — float constant
        //   "const.null"           → Immediate(0)        — null pointer
        for vi in &func.local_values {
            if let Some(ref n) = vi.name {
                if let Some(sym_name) = n.strip_prefix("global.") {
                    self.value_map
                        .insert(vi.id, MachineOperand::Symbol(sym_name.to_string()));
                } else if let Some(int_str) = n.strip_prefix("const.int.") {
                    if let Ok(val) = int_str.parse::<i64>() {
                        self.value_map.insert(vi.id, MachineOperand::Immediate(val));
                    }
                } else if let Some(flt_str) = n.strip_prefix("const.float.") {
                    if let Ok(val) = flt_str.parse::<f64>() {
                        // Store the bit pattern at the correct width for the
                        // value's IR type.  F32 values must use f32 bit
                        // patterns (32-bit), not f64 (64-bit), so that FMOV
                        // from GPR to the S-register view sees the right bits.
                        let bits = if matches!(vi.ty, IrType::F32) {
                            (val as f32).to_bits() as i64
                        } else {
                            val.to_bits() as i64
                        };
                        self.value_map
                            .insert(vi.id, MachineOperand::Immediate(bits));
                    }
                } else if n == "const.null" {
                    self.value_map.insert(vi.id, MachineOperand::Immediate(0));
                }
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

        // Phase 4: Store frame metadata in MachineFunction for deferred
        // prologue/epilogue emission (runs AFTER register allocation).
        //
        // We do NOT emit prologue/epilogue here because register allocation
        // hasn't happened yet.  Virtual registers are still in use, so
        // `compute_used_callee_saved()` would see zero callee-saved regs.
        // The correct sequence is: isel → regalloc → emit_prologue/epilogue.
        mf.has_calls = self.has_calls;
        mf.is_variadic = self.is_variadic_func;
        mf.frame_offset_watermark = self.current_frame_offset;
        mf.frame_objects = self
            .frame_objects
            .iter()
            .map(|fo| (fo.size, fo.alignment, fo.offset))
            .collect();

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
                self.lower_fcmp(*result, *pred, *lhs, *rhs, func, mf, mbb_id);
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
                ..
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

            // --- Floating-point conversion instructions ---
            Instruction::SIToFP {
                result,
                value,
                to_ty,
            } => {
                self.lower_si_to_fp(*result, *value, to_ty, func, mf, mbb_id);
            }
            Instruction::UIToFP {
                result,
                value,
                to_ty,
            } => {
                self.lower_ui_to_fp(*result, *value, to_ty, func, mf, mbb_id);
            }
            Instruction::FPToSI {
                result,
                value,
                to_ty,
            } => {
                self.lower_fp_to_si(*result, *value, to_ty, func, mf, mbb_id);
            }
            Instruction::FPToUI {
                result,
                value,
                to_ty,
            } => {
                self.lower_fp_to_ui(*result, *value, to_ty, func, mf, mbb_id);
            }
            Instruction::FPExt {
                result,
                value,
                to_ty,
            } => {
                self.lower_fp_ext(*result, *value, to_ty, func, mf, mbb_id);
            }
            Instruction::FPTrunc {
                result,
                value,
                to_ty,
            } => {
                self.lower_fp_trunc(*result, *value, to_ty, func, mf, mbb_id);
            }

            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects,
                is_align_stack: _,
                goto_targets: _,
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

            // -- Computed goto --
            Instruction::BlockAddress { result, block } => {
                // ADR Xd, label — load PC-relative address of block label.
                let dst = self.alloc_vreg();
                self.value_map.insert(*result, dst.clone());
                // Map IR block ID → machine block ID via block_map
                let target_id = self.block_map.get(block).copied().unwrap_or(0);
                let mbb = &mut mf.blocks[mbb_id as usize];
                let mut mi = MachineInstr::new(AArch64Opcode::ADR.as_u32());
                mi.add_operand(dst);
                mi.add_operand(MachineOperand::Label(target_id));
                mbb.push_instr(mi);
            }

            Instruction::IndirectBranch {
                addr,
                possible_targets: _,
            } => {
                // BR Xn — unconditional branch to address in register.
                let op = self.resolve_operand(*addr);
                let mbb = &mut mf.blocks[mbb_id as usize];
                let mut mi = MachineInstr::new(AArch64Opcode::BR.as_u32());
                mi.add_operand(op);
                mi.set_terminator();
                mbb.push_instr(mi);
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
    #[allow(dead_code)]
    pub fn emit_prologue(&self, mf: &mut MachineFunction) {
        if mf.blocks.is_empty() {
            return;
        }

        let frame_size = mf.frame_size;
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            return; // Leaf function with no locals — skip prologue.
        }

        let mut prologue: Vec<MachineInstr> = Vec::new();

        // STP X29, X30, [SP, #-frame_size]!  (pre-indexed: save FP/LR, adjust SP)
        let stp_fp_lr = MachineInstr::with_operands(
            AArch64Opcode::StpPre.as_u32(),
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

        // For variadic functions, spill argument registers X0-X7 into
        // the 64-byte register save area at [SP+16..SP+80].  This allows
        // __builtin_va_start / va_arg to walk the arguments sequentially.
        if self.is_variadic_func {
            for reg_idx in 0u32..8 {
                let arg_reg = INTEGER_ARG_REGS[reg_idx as usize];
                let save_offset = 16 + (reg_idx as i32) * 8;
                let str_arg = MachineInstr::with_operands(
                    AArch64Opcode::STR.as_u32(),
                    vec![
                        MachineOperand::Register(arg_reg),
                        MachineOperand::Memory {
                            base: SP,
                            offset: save_offset,
                            index: None,
                            scale: 1,
                        },
                    ],
                );
                prologue.push(str_arg);
            }
        }

        // Save callee-saved register pairs.
        // For variadic functions, the first 64 bytes after FP/LR are the
        // arg save area, so callee-saved regs start at offset 80.
        let callee_regs = &mf.used_callee_saved;
        let callee_start = if self.is_variadic_func { 80i32 } else { 16i32 };
        let mut offset = callee_start;
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
    #[allow(dead_code)]
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
                // For variadic functions, callee-saved regs start at 80
                // (past the 64-byte arg save area).
                let callee_start = if self.is_variadic_func { 80i32 } else { 16i32 };
                let mut offset = callee_start;
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
                    AArch64Opcode::LdpPost.as_u32(),
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

    /// Zero-extend a 32-bit value in a 64-bit X register.
    ///
    /// AArch64 ALU instructions always operate on the full 64-bit X register,
    /// so I32 operations like MUL, ADD, SUB, SHL can leave significant data in
    /// the upper 32 bits.  This helper emits `UBFM Xd, Xn, #0, #31` (the
    /// canonical UXTW alias) to clear the upper 32 bits when the IR type is
    /// I32 or smaller (but not I64 or pointer types which need full width).
    ///
    /// For non-I32 types (I64, pointers, floats) this is a no-op and returns
    /// the input operand unchanged.
    fn maybe_truncate_i32(
        &mut self,
        src: MachineOperand,
        ty: &IrType,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) -> MachineOperand {
        let needs_trunc = matches!(ty, IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32);
        if !needs_trunc {
            return src;
        }
        // UBFM Xd, Xn, #0, #31  →  zero-extends bits [31:0] to 64 bits
        let truncated = self.alloc_vreg();
        let ubfm = MachineInstr::with_operands(
            0x000D, // OP_UBFM
            vec![
                truncated.clone(),
                src,
                MachineOperand::Immediate(0),  // immr = 0
                MachineOperand::Immediate(31), // imms = 31
            ],
        );
        self.push_instr(mf, mbb_id, ubfm);
        truncated
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

        // === TWO-PHASE ARGUMENT SETUP ===
        //
        // Phase 1A: Materialize all arguments into virtual registers and
        // classify them.  This MUST happen before any physical register
        // moves so that address materialization (ADRP+ADD) does not clobber
        // physical argument registers set up by earlier arguments.
        //
        // Phase 1B: Copy the materialized vregs into physical argument
        // registers (or spill to the stack).

        // Collect (class, materialized_vreg, arg_ty) for each argument.
        enum ArgSlot {
            GprReg(PhysReg, MachineOperand),
            FprReg(PhysReg, MachineOperand), // target FP reg alias, src
            Stack(MachineOperand, i32),
        }

        let mut slots: Vec<ArgSlot> = Vec::with_capacity(args.len());

        for &arg_val in args {
            let arg_ty = func.get_value_type(arg_val);
            let class = classify_ir_arg(arg_ty, &self.target);
            let src = self.resolve_operand(arg_val);
            // Phase 1A: Materialize into a virtual register.
            let src = self.materialize_to_register(src, mf, mbb_id);

            match class {
                IrArgClass::FpReg => {
                    if fpr_idx < fpr_args.len() {
                        let target_reg = if arg_ty.size_bytes(&self.target) <= 4 {
                            v_to_s(fpr_args[fpr_idx])
                        } else {
                            v_to_d(fpr_args[fpr_idx])
                        };
                        slots.push(ArgSlot::FprReg(target_reg, src));
                        fpr_idx += 1;
                    } else {
                        slots.push(ArgSlot::Stack(src, stack_offset));
                        stack_offset += 8;
                    }
                }
                IrArgClass::IntReg | IrArgClass::ByReference => {
                    if gpr_idx < gpr_args.len() {
                        slots.push(ArgSlot::GprReg(gpr_args[gpr_idx], src));
                        gpr_idx += 1;
                    } else {
                        slots.push(ArgSlot::Stack(src, stack_offset));
                        stack_offset += 8;
                    }
                }
                IrArgClass::OnStack => {
                    slots.push(ArgSlot::Stack(src, stack_offset));
                    stack_offset += 8;
                }
            }
        }

        // Phase 1B: Now emit the physical register moves and stack stores.
        // All materialization is already done, so these MOV/STR instructions
        // won't clobber each other's source vregs.
        //
        // If any arguments spill to the stack, we reserve space below the
        // current SP with `SUB SP, SP, #size` so that the stores land in
        // freshly allocated space instead of overwriting the saved FP/LR
        // or callee-saved registers.  After the call we restore SP with
        // a matching `ADD SP, SP, #size`.
        //
        // Because all frame-relative accesses (locals, spills) now use FP
        // (X29) as their base register, temporarily lowering SP is safe.
        let outgoing_stack_size = if stack_offset > 0 {
            // AArch64 requires 16-byte SP alignment.
            let aligned = ((stack_offset as u32 + 15) & !15) as i64;
            let sub_sp = MachineInstr::with_operands(
                AArch64Opcode::SUBimm.as_u32(),
                vec![
                    MachineOperand::Register(SP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(aligned),
                ],
            );
            self.push_instr(mf, mbb_id, sub_sp);
            aligned
        } else {
            0
        };

        for slot in slots {
            match slot {
                ArgSlot::GprReg(phys, src) => {
                    let mov = MachineInstr::with_operands(
                        AArch64Opcode::ORR.as_u32(),
                        vec![
                            MachineOperand::Register(phys),
                            MachineOperand::Register(XZR),
                            src,
                        ],
                    );
                    self.push_instr(mf, mbb_id, mov);
                }
                ArgSlot::FprReg(phys, src) => {
                    // Cross-domain move: GP virtual register → FP physical register.
                    // Use FMOVint (GPR→FP) instead of FMOV (FP→FP).
                    let ftype: i64 = if phys.0 >= 130 { 1 } else { 0 }; // D-reg ⇒ double(1), S-reg ⇒ float(0)
                    let mov = MachineInstr::with_operands(
                        AArch64Opcode::FMOVint.as_u32(),
                        vec![
                            MachineOperand::Register(phys),
                            src,
                            MachineOperand::Immediate(ftype),
                        ],
                    );
                    self.push_instr(mf, mbb_id, mov);
                }
                ArgSlot::Stack(src, off) => {
                    // Stores go to the just-reserved outgoing area.
                    let str_inst = MachineInstr::with_operands(
                        AArch64Opcode::STR.as_u32(),
                        vec![
                            src,
                            MachineOperand::Memory {
                                base: SP,
                                offset: off,
                                index: None,
                                scale: 1,
                            },
                        ],
                    );
                    self.push_instr(mf, mbb_id, str_inst);
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
                MachineInstr::with_operands(AArch64Opcode::BLR.as_u32(), vec![callee_op])
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

        // Restore SP after the call if we reserved outgoing argument space.
        if outgoing_stack_size > 0 {
            let add_sp = MachineInstr::with_operands(
                AArch64Opcode::ADDimm.as_u32(),
                vec![
                    MachineOperand::Register(SP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(outgoing_stack_size),
                ],
            );
            self.push_instr(mf, mbb_id, add_sp);
        }

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
                let ftype: i64 = if ret_ty.size_bytes(&self.target) <= 4 {
                    0
                } else {
                    1
                };
                let dest = self.alloc_vreg();
                // Cross-domain: FP return register → GP virtual register.
                let fmov = MachineInstr::with_operands(
                    AArch64Opcode::FMOVtoGPR.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Register(ret_reg),
                        MachineOperand::Immediate(ftype),
                    ],
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
            let src_raw = self.resolve_operand(*val);
            let src = self.materialize_to_register(src_raw, mf, mbb_id);

            if ret_ty.is_floating() {
                let ret_reg = if ret_ty.size_bytes(&self.target) <= 4 {
                    v_to_s(V0)
                } else {
                    v_to_d(V0)
                };
                let ftype: i64 = if ret_ty.size_bytes(&self.target) <= 4 {
                    0
                } else {
                    1
                };
                // Cross-domain: GP virtual register → FP return register.
                let fmov = MachineInstr::with_operands(
                    AArch64Opcode::FMOVint.as_u32(),
                    vec![
                        MachineOperand::Register(ret_reg),
                        src,
                        MachineOperand::Immediate(ftype),
                    ],
                );
                self.push_instr(mf, mbb_id, fmov);
            } else {
                // Integer / pointer return in X0.
                let src_reg = src;
                let mov = MachineInstr::with_operands(
                    AArch64Opcode::ORR.as_u32(),
                    vec![
                        MachineOperand::Register(X0),
                        MachineOperand::Register(XZR),
                        src_reg,
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
            vec![dest.clone(), MachineOperand::Symbol(symbol.to_string())],
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
        let lhs_raw = self.resolve_operand(lhs);
        let rhs_raw = self.resolve_operand(rhs);

        // AArch64 register-form ALU instructions require both sources in
        // registers.  Materialize immediates, frame-indices and symbols
        // before emitting the instruction.
        let lhs_op = self.materialize_to_register(lhs_raw, mf, mbb_id);
        let rhs_op = self.materialize_to_register(rhs_raw, mf, mbb_id);
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
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv => {
                // FP binary ops need cross-domain moves because all vregs are
                // GPRs but FP instructions require FP registers.
                let is_double = matches!(ty, IrType::F64 | IrType::F80);
                let ftype: i64 = if is_double { 1 } else { 0 };
                let fp_lhs = if is_double { v_to_d(V0) } else { v_to_s(V0) };
                let fp_rhs = if is_double { v_to_d(V1) } else { v_to_s(V1) };
                let fp_dst = if is_double { v_to_d(V2) } else { v_to_s(V2) };
                // Move LHS GPR → FP
                let fmov_l = MachineInstr::with_operands(
                    AArch64Opcode::FMOVint.as_u32(),
                    vec![
                        MachineOperand::Register(fp_lhs),
                        lhs_op,
                        MachineOperand::Immediate(ftype),
                    ],
                );
                self.push_instr(mf, mbb_id, fmov_l);
                // Move RHS GPR → FP
                let fmov_r = MachineInstr::with_operands(
                    AArch64Opcode::FMOVint.as_u32(),
                    vec![
                        MachineOperand::Register(fp_rhs),
                        rhs_op,
                        MachineOperand::Immediate(ftype),
                    ],
                );
                self.push_instr(mf, mbb_id, fmov_r);
                // FP operation
                let fp_opcode = match op {
                    BinOp::FAdd => AArch64Opcode::FADD,
                    BinOp::FSub => AArch64Opcode::FSUB,
                    BinOp::FMul => AArch64Opcode::FMUL,
                    BinOp::FDiv => AArch64Opcode::FDIV,
                    _ => unreachable!(),
                };
                let fp_op = MachineInstr::with_operands(
                    fp_opcode.as_u32(),
                    vec![
                        MachineOperand::Register(fp_dst),
                        MachineOperand::Register(fp_lhs),
                        MachineOperand::Register(fp_rhs),
                        MachineOperand::Immediate(ftype),
                    ],
                );
                self.push_instr(mf, mbb_id, fp_op);
                // Move result FP → GPR
                let fmov_out = MachineInstr::with_operands(
                    AArch64Opcode::FMOVtoGPR.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Register(fp_dst),
                        MachineOperand::Immediate(ftype),
                    ],
                );
                self.push_instr(mf, mbb_id, fmov_out);
                self.value_map.insert(result, dest);
                return;
            }

            BinOp::Mul => {
                // MUL is an alias of MADD with zero addend: Xd = Xn * Xm + XZR
                let madd = MachineInstr::with_operands(
                    AArch64Opcode::MADD.as_u32(),
                    vec![dest.clone(), lhs_op, rhs_op, MachineOperand::Register(XZR)],
                );
                self.push_instr(mf, mbb_id, madd);
                // Truncate to 32 bits if this is an I32 operation — the 64-bit
                // MUL leaves significant upper-32-bit data that corrupts later
                // shifts / masks.
                let final_dest = self.maybe_truncate_i32(dest, ty, mf, mbb_id);
                self.value_map.insert(result, final_dest);
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
                let final_dest = self.maybe_truncate_i32(dest, ty, mf, mbb_id);
                self.value_map.insert(result, final_dest);
                return;
            }

            // FP remainder — requires a runtime call.
            BinOp::FRem => {
                self.emit_frem_call(result, lhs, rhs, ty, mf, mbb_id);
                return;
            }
        };

        let instr =
            MachineInstr::with_operands(opcode.as_u32(), vec![dest.clone(), lhs_op, rhs_op]);
        self.push_instr(mf, mbb_id, instr);
        // Truncate to 32 bits for I32 operations — AArch64 ALU instructions
        // operate on 64-bit X registers, so ADD/SUB/SHL/etc. can leave
        // significant data in the upper 32 bits that corrupts subsequent ops.
        let final_dest = self.maybe_truncate_i32(dest, ty, mf, mbb_id);
        self.value_map.insert(result, final_dest);
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
                // The operand might be a Symbol (global variable) or a
                // VirtualReg.  Materialize it into a register first.
                // We emit the operands as [dest, VirtualReg, Imm(0)] — after
                // register allocation the VirtualReg becomes Register(phys),
                // and `extract_base_offset` handles Register+Immediate layout.
                let base_reg_op = self.materialize_to_register(base_op, mf, mbb_id);
                MachineInstr::with_operands(
                    opcode.as_u32(),
                    vec![dest.clone(), base_reg_op, MachineOperand::Immediate(0)],
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

        // AArch64 STR requires the value in a register — materialize
        // immediates and symbols before encoding the store.
        let val_op = self.materialize_to_register(val_op, mf, mbb_id);

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
            _ => {
                // The address might be a Symbol (global variable) or a
                // VirtualReg.  Materialize it into a register first.
                // Emit as [val_reg, VirtualReg, Imm(0)] — register allocator
                // converts VirtualReg → Register, which extract_base_offset
                // handles in its Register+Immediate layout.
                let addr_reg_op = self.materialize_to_register(addr_op, mf, mbb_id);
                MachineInstr::with_operands(
                    opcode.as_u32(),
                    vec![val_op, addr_reg_op, MachineOperand::Immediate(0)],
                )
            }
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
        let lhs_raw = self.resolve_operand(lhs);
        let rhs_raw = self.resolve_operand(rhs);

        // CMP is a register-form instruction; materialize non-register operands.
        let mut lhs_op = self.materialize_to_register(lhs_raw, mf, mbb_id);
        let mut rhs_op = self.materialize_to_register(rhs_raw, mf, mbb_id);

        // For *signed* comparisons on I32 values we must sign-extend
        // the operands before the 64-bit CMP.  AArch64 loads use LDRW
        // which zero-extends to 64 bits — a negative I32 like -3
        // (0xFFFFFFFD) becomes 0x00000000_FFFFFFFD in the X register,
        // which is a large positive number in 64-bit signed arithmetic.
        // SXTW corrects this by replicating bit 31 into bits 32-63.
        let is_signed = matches!(
            pred,
            ICmpPredicate::Slt | ICmpPredicate::Sle | ICmpPredicate::Sgt | ICmpPredicate::Sge
        );
        if is_signed {
            let lhs_ty = self.value_types.get(&lhs).cloned();
            let rhs_ty = self.value_types.get(&rhs).cloned();
            let need_sext = |ty: &Option<IrType>| {
                matches!(ty, Some(IrType::I8) | Some(IrType::I16) | Some(IrType::I32))
            };
            if need_sext(&lhs_ty) {
                let ext = self.alloc_vreg();
                let sxtw = MachineInstr::with_operands(
                    AArch64Opcode::SXTW.as_u32(),
                    vec![ext.clone(), lhs_op],
                );
                self.push_instr(mf, mbb_id, sxtw);
                lhs_op = ext;
            }
            if need_sext(&rhs_ty) {
                let ext = self.alloc_vreg();
                let sxtw = MachineInstr::with_operands(
                    AArch64Opcode::SXTW.as_u32(),
                    vec![ext.clone(), rhs_op],
                );
                self.push_instr(mf, mbb_id, sxtw);
                rhs_op = ext;
            }
        }

        // CMP (alias of SUBS with zero destination).
        let cmp = MachineInstr::with_operands(AArch64Opcode::CMP.as_u32(), vec![lhs_op, rhs_op]);
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
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let lhs_raw = self.resolve_operand(lhs);
        let rhs_raw = self.resolve_operand(rhs);
        let lhs_op = self.materialize_to_register(lhs_raw, mf, mbb_id);
        let rhs_op = self.materialize_to_register(rhs_raw, mf, mbb_id);
        let dest = self.alloc_vreg();

        // Determine float precision from LHS type.
        let lhs_ty = func.get_value_type(lhs);
        let is_double = matches!(lhs_ty, IrType::F64 | IrType::F80);
        let ftype: i64 = if is_double { 1 } else { 0 };
        let fp_lhs = if is_double { v_to_d(V0) } else { v_to_s(V0) };
        let fp_rhs = if is_double { v_to_d(V1) } else { v_to_s(V1) };
        // Move GPR values to FP registers before FCMP.
        let fmov_l = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(fp_lhs),
                lhs_op,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov_l);
        let fmov_r = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(fp_rhs),
                rhs_op,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov_r);
        // FCMP sets NZCV using FP register operands.
        let fcmp = MachineInstr::with_operands(
            AArch64Opcode::FCMP.as_u32(),
            vec![
                MachineOperand::Register(fp_lhs),
                MachineOperand::Register(fp_rhs),
                MachineOperand::Immediate(ftype),
            ],
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

    fn lower_branch(&mut self, target: BasicBlockId, mf: &mut MachineFunction, mbb_id: u32) {
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
        let cond_raw = self.resolve_operand(condition);
        let cond_op = self.materialize_to_register(cond_raw, mf, mbb_id);
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
        let val_raw = self.resolve_operand(value);
        let val_op = self.materialize_to_register(val_raw, mf, mbb_id);
        let default_mbb = self.block_map.get(&default).copied().unwrap_or(0);

        for &(case_val, case_target) in cases {
            let target_mbb = self.block_map.get(&case_target).copied().unwrap_or(0);

            // Materialize the case constant into a register (CMP is register-form).
            let case_reg = self.materialize_immediate(case_val, mf, mbb_id);

            // CMP val, case.
            let cmp = MachineInstr::with_operands(
                AArch64Opcode::CMP.as_u32(),
                vec![val_op.clone(), case_reg],
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
        // On AArch64 our codegen uses X-registers (64-bit) for all integer
        // types, so STR X0 writes 8 bytes even for I32 values.  We must
        // guarantee that *every* alloca slot:
        //   (a) occupies at least 8 bytes (to avoid clobbering adjacent data)
        //   (b) is 8-byte aligned (so that SP-relative offsets are multiples
        //       of 8, which the unsigned-offset LDR/STR encoding requires).
        let raw_size = ty.size_bytes(&self.target) as u32;
        let size = raw_size.max(8);
        let align = alignment.max(ty.alignment(&self.target) as u32).max(8);

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
        let base_raw = self.resolve_operand(base);
        let mut current = self.materialize_to_register(base_raw, mf, mbb_id);
        let mut current_ty = ty.clone();

        // LLVM-style GEP semantics:
        //   - The FIRST index is always an array-level index: it multiplies
        //     by the full size of the pointee type (the base type).
        //     For struct pointers this means "which struct in an array of
        //     structs" — typically 0 for simple member access.
        //   - SUBSEQUENT indices drill into nested types:
        //     * If the current type is a struct, the index selects a field
        //       and we compute the cumulative byte offset to that field
        //       (respecting alignment padding between fields).
        //     * If the current type is an array, the index selects an element.
        for (idx_pos, &idx) in indices.iter().enumerate() {
            let idx_raw = self.resolve_operand(idx);
            let is_first_index = idx_pos == 0;

            // Handle struct field access (subsequent index on a struct type).
            // The index MUST be a constant selecting a specific field.
            // We compute the cumulative byte offset and add it directly.
            if !is_first_index {
                if let IrType::Struct { ref fields, packed } = current_ty {
                    if let MachineOperand::Immediate(field_idx) = &idx_raw {
                        let fi = *field_idx as usize;
                        if fi < fields.len() {
                            // Compute the byte offset of the requested field,
                            // respecting alignment padding between fields.
                            let is_packed = packed;
                            let mut offset: u64 = 0;
                            for field in fields.iter().take(fi) {
                                let f_size = field.size_bytes(&self.target);
                                if !is_packed {
                                    let f_align = field.alignment(&self.target);
                                    offset = (offset + f_align - 1) & !(f_align - 1);
                                }
                                offset += f_size;
                            }
                            if !is_packed {
                                let field_align = fields[fi].alignment(&self.target);
                                offset = (offset + field_align - 1) & !(field_align - 1);
                            }

                            // Add the byte offset to the current address.
                            if offset > 0 {
                                let off_op = self.materialize_immediate(offset as i64, mf, mbb_id);
                                let next = self.alloc_vreg();
                                let add = MachineInstr::with_operands(
                                    AArch64Opcode::ADD.as_u32(),
                                    vec![next.clone(), current, off_op],
                                );
                                self.push_instr(mf, mbb_id, add);
                                current = next;
                            }
                            current_ty = fields[fi].clone();
                            continue;
                        }
                    }
                    // Non-constant struct index fallback (shouldn't happen
                    // in well-formed IR) — fall through to generic handling.
                }
            }

            // For non-struct types (arrays, pointers) or the first index
            // on a struct (array-of-structs indexing): multiply index by
            // element size and add to base.
            let idx_op = self.materialize_to_register(idx_raw, mf, mbb_id);
            let elem_size = match &current_ty {
                IrType::Array { element, .. } => {
                    let sz = element.size_bytes(&self.target);
                    current_ty = (**element).clone();
                    sz
                }
                IrType::Struct { .. } => {
                    // First index on struct: array-of-structs indexing.
                    // current_ty stays as the struct for subsequent field
                    // indices.
                    self.get_element_size(&current_ty)
                }
                _ => current_ty.size_bytes(&self.target),
            };

            if elem_size == 0 {
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
                let shift_reg = self.materialize_immediate(shift, mf, mbb_id);
                let lsl = MachineInstr::with_operands(
                    AArch64Opcode::LSL.as_u32(),
                    vec![shifted.clone(), idx_op, shift_reg],
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
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        let src_ty = func.get_value_type(value);

        let _src_is_fp = src_ty.is_floating();
        let _dst_is_fp = to_ty.is_floating();

        // Use get_or_alloc_vreg so that when the same ValueId is defined
        // in multiple basic blocks (phi-like BitCast copies for ternary
        // expressions), all definitions target the SAME virtual register.
        // At runtime only one path executes, writing the correct value.
        let dest = self.get_or_alloc_vreg(result);
        // All values in the current backend live in GP registers,
        // so BitCast (same-width type reinterpretation) is always a
        // simple GPR→GPR copy regardless of float↔int domain crossing.
        {
            // Same domain: MOV via ORR Xd, XZR, Xn.
            let mov = MachineInstr::with_operands(
                AArch64Opcode::ORR.as_u32(),
                vec![dest.clone(), MachineOperand::Register(XZR), src],
            );
            self.push_instr(mf, mbb_id, mov);
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
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
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
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        let dest = self.alloc_vreg();
        let src_ty = func.get_value_type(value);
        let src_bits = src_ty.size_bits(&self.target);

        match src_bits {
            1 => {
                // I1 values are already 0 or 1 in full-width registers on AArch64
                // (produced by cset/csinc). Use UXTB to mask to low byte without
                // needing a separate mask register, avoiding register allocation
                // conflicts where the materialized mask could overwrite the source.
                let uxtb = MachineInstr::with_operands(
                    AArch64Opcode::UXTB.as_u32(),
                    vec![dest.clone(), src],
                );
                self.push_instr(mf, mbb_id, uxtb);
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
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
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
                let shift_reg = self.materialize_immediate(shift_amt as i64, mf, mbb_id);
                let tmp = self.alloc_vreg();
                let lsl = MachineInstr::with_operands(
                    AArch64Opcode::LSL.as_u32(),
                    vec![tmp.clone(), src, shift_reg.clone()],
                );
                self.push_instr(mf, mbb_id, lsl);
                let asr = MachineInstr::with_operands(
                    AArch64Opcode::ASR.as_u32(),
                    vec![dest.clone(), tmp, shift_reg],
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

    // -----------------------------------------------------------------------
    // Floating-point conversion lowering (AArch64)
    // -----------------------------------------------------------------------

    /// SIToFP: signed integer → float via SCVTF.
    ///
    /// Because all virtual registers live in GPRs, we emit:
    ///   SCVTF <Sd|Dd>, <Xn|Wn>    ; convert int (GPR) → FP result (in FP reg)
    ///   FMOVtoGPR <Xd>, <Sd|Dd>   ; move FP bits back to GPR dest
    fn lower_si_to_fp(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        _func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        // Determine target float precision: 0=S, 1=D
        let is_double = matches!(to_ty, IrType::F64 | IrType::F80);
        let ftype: i64 = if is_double { 1 } else { 0 };
        let fp_tmp = if is_double { v_to_d(V0) } else { v_to_s(V0) };
        // SCVTF <fp_tmp>, <src_gpr>  (integer → float, result in FP register)
        let scvtf = MachineInstr::with_operands(
            AArch64Opcode::SCVTF.as_u32(),
            vec![
                MachineOperand::Register(fp_tmp),
                src,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, scvtf);
        // FMOVtoGPR dest, <fp_tmp>  (FP → GPR)
        let dest = self.alloc_vreg();
        let fmov = MachineInstr::with_operands(
            AArch64Opcode::FMOVtoGPR.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(fp_tmp),
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov);
        self.value_map.insert(result, dest);
    }

    /// UIToFP: unsigned integer → float via UCVTF.
    fn lower_ui_to_fp(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        _func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        let is_double = matches!(to_ty, IrType::F64 | IrType::F80);
        let ftype: i64 = if is_double { 1 } else { 0 };
        let fp_tmp = if is_double { v_to_d(V0) } else { v_to_s(V0) };
        let ucvtf = MachineInstr::with_operands(
            AArch64Opcode::UCVTF.as_u32(),
            vec![
                MachineOperand::Register(fp_tmp),
                src,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, ucvtf);
        let dest = self.alloc_vreg();
        let fmov = MachineInstr::with_operands(
            AArch64Opcode::FMOVtoGPR.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(fp_tmp),
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov);
        self.value_map.insert(result, dest);
    }

    /// FPToSI: float → signed integer via FCVTZS.
    ///
    /// Emit:
    ///   FMOVint <Sd|Dd>, <Xn|Wn>  ; move float bits from GPR to FP reg
    ///   FCVTZS <Xd|Wd>, <Sd|Dd>   ; convert FP → signed int (result in GPR)
    fn lower_fp_to_si(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        // Determine source float precision from the IR value type.
        let src_ty = func.get_value_type(value);
        let is_double = matches!(src_ty, IrType::F64 | IrType::F80);
        let ftype: i64 = if is_double { 1 } else { 0 };
        let fp_tmp = if is_double { v_to_d(V0) } else { v_to_s(V0) };
        // FMOVint <fp_tmp>, <src_gpr>  (GPR → FP)
        let fmov = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(fp_tmp),
                src,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov);
        // FCVTZS <dest_gpr>, <fp_tmp>  (FP → signed int, result in GPR)
        let dest = self.alloc_vreg();
        let fcvtzs = MachineInstr::with_operands(
            AArch64Opcode::FCVTZS.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(fp_tmp),
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fcvtzs);
        self.value_map.insert(result, dest);
    }

    /// FPToUI: float → unsigned integer via FCVTZU.
    fn lower_fp_to_ui(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        let src_ty = func.get_value_type(value);
        let is_double = matches!(src_ty, IrType::F64 | IrType::F80);
        let ftype: i64 = if is_double { 1 } else { 0 };
        let fp_tmp = if is_double { v_to_d(V0) } else { v_to_s(V0) };
        let fmov = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(fp_tmp),
                src,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov);
        let dest = self.alloc_vreg();
        let fcvtzu = MachineInstr::with_operands(
            AArch64Opcode::FCVTZU.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(fp_tmp),
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fcvtzu);
        self.value_map.insert(result, dest);
    }

    /// FPExt: float widening (e.g. F32 → F64) via FCVT.
    ///
    /// Emit:
    ///   FMOVint S0, Wn    ; move F32 bits from GPR to FP
    ///   FCVT D1, S0       ; widen single → double
    ///   FMOVtoGPR Xd, D1  ; move F64 bits back to GPR
    fn lower_fp_ext(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        // Source is narrower (F32), destination is wider (F64).
        let src_ty = func.get_value_type(value);
        let src_is_double = matches!(src_ty, IrType::F64 | IrType::F80);
        let dst_is_double = matches!(to_ty, IrType::F64 | IrType::F80);
        let src_ftype: i64 = if src_is_double { 1 } else { 0 };
        let dst_ftype: i64 = if dst_is_double { 1 } else { 0 };
        let src_fp = if src_is_double {
            v_to_d(V0)
        } else {
            v_to_s(V0)
        };
        let dst_fp = if dst_is_double {
            v_to_d(V1)
        } else {
            v_to_s(V1)
        };
        // Step 1: GPR → FP (source)
        let fmov_in = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(src_fp),
                src,
                MachineOperand::Immediate(src_ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov_in);
        // Step 2: FCVT <dst_fp>, <src_fp>  (precision conversion)
        let fcvt = MachineInstr::with_operands(
            AArch64Opcode::FCVT.as_u32(),
            vec![
                MachineOperand::Register(dst_fp),
                MachineOperand::Register(src_fp),
                MachineOperand::Immediate(dst_ftype), // dst ftype
                MachineOperand::Immediate(src_ftype), // src ftype
            ],
        );
        self.push_instr(mf, mbb_id, fcvt);
        // Step 3: FP → GPR (result)
        let dest = self.alloc_vreg();
        let fmov_out = MachineInstr::with_operands(
            AArch64Opcode::FMOVtoGPR.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(dst_fp),
                MachineOperand::Immediate(dst_ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov_out);
        self.value_map.insert(result, dest);
    }

    /// FPTrunc: float narrowing (e.g. F64 → F32) via FCVT.
    fn lower_fp_trunc(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        let src_raw = self.resolve_operand(value);
        let src = self.materialize_to_register(src_raw, mf, mbb_id);
        let src_ty = func.get_value_type(value);
        let src_is_double = matches!(src_ty, IrType::F64 | IrType::F80);
        let dst_is_double = matches!(to_ty, IrType::F64 | IrType::F80);
        let src_ftype: i64 = if src_is_double { 1 } else { 0 };
        let dst_ftype: i64 = if dst_is_double { 1 } else { 0 };
        let src_fp = if src_is_double {
            v_to_d(V0)
        } else {
            v_to_s(V0)
        };
        let dst_fp = if dst_is_double {
            v_to_d(V1)
        } else {
            v_to_s(V1)
        };
        // Step 1: GPR → FP (source)
        let fmov_in = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(src_fp),
                src,
                MachineOperand::Immediate(src_ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov_in);
        // Step 2: FCVT <dst_fp>, <src_fp>  (narrowing)
        let fcvt = MachineInstr::with_operands(
            AArch64Opcode::FCVT.as_u32(),
            vec![
                MachineOperand::Register(dst_fp),
                MachineOperand::Register(src_fp),
                MachineOperand::Immediate(dst_ftype),
                MachineOperand::Immediate(src_ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fcvt);
        // Step 3: FP → GPR (result)
        let dest = self.alloc_vreg();
        let fmov_out = MachineInstr::with_operands(
            AArch64Opcode::FMOVtoGPR.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(dst_fp),
                MachineOperand::Immediate(dst_ftype),
            ],
        );
        self.push_instr(mf, mbb_id, fmov_out);
        self.value_map.insert(result, dest);
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
            self.diag
                .warning(Span::DUMMY, "AArch64 codegen: empty phi node encountered");
            self.value_map.insert(result, MachineOperand::Register(XZR));
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
        // Detect builtin asm ($N operands) vs GCC user asm (%N operands).
        let is_gcc_user_asm = template.contains('%');

        if is_gcc_user_asm {
            self.lower_gcc_user_inline_asm(
                result,
                template,
                constraints,
                operands,
                clobbers,
                mf,
                mbb_id,
            );
        } else {
            self.lower_builtin_inline_asm(
                result,
                template,
                constraints,
                operands,
                clobbers,
                mf,
                mbb_id,
            );
        }
    }

    /// Lowers compiler-generated builtin inline assembly with $N operands.
    ///
    /// These are templates generated by `expr_lowering.rs` for builtins
    /// like `__builtin_clz`, `__builtin_ctz`, `__builtin_popcount`, etc.
    /// The templates use `$0`, `$1` etc. as operand placeholders.
    fn lower_builtin_inline_asm(
        &mut self,
        result: &Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        _clobbers: &[String],
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        // Parse constraints: "=r,r" → output is $0, inputs start at $1.
        // "=r" means output-only, "=r,r" means output + 1 input, etc.
        let constraint_parts: Vec<&str> = constraints.split(',').collect();
        let _has_output = constraint_parts.first().map_or(false, |c| c.contains('='));

        // Allocate output vreg.
        let out_vreg = if let Some(res) = result {
            let v = self.alloc_vreg();
            self.value_map.insert(*res, v.clone());
            v
        } else {
            self.alloc_vreg()
        };

        // Resolve input operands into vregs.
        let mut input_vregs: Vec<MachineOperand> = Vec::new();
        for &op in operands {
            let resolved = self.resolve_operand(op);
            let vreg = self.ensure_in_register(resolved, mf, mbb_id);
            input_vregs.push(vreg);
        }

        // Build operand map: $0 = output, $1 = first input, $2 = second input, ...
        let mut operand_map: Vec<MachineOperand> = Vec::new();
        operand_map.push(out_vreg.clone());
        operand_map.extend(input_vregs.iter().cloned());

        // Parse template into lines.
        let lines: Vec<&str> = template
            .split('\n')
            .flat_map(|l| l.split('\t'))
            .flat_map(|l| l.split(';'))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for line in &lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            // Skip labels like "1:", "2f:", etc.
            if line.ends_with(':') && !line.contains(' ') {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, |c: char| c.is_whitespace()).collect();
            let mnemonic = parts[0].to_lowercase();
            let args_str = if parts.len() > 1 { parts[1].trim() } else { "" };

            // Parse comma-separated operands.
            let args: Vec<&str> = if args_str.is_empty() {
                vec![]
            } else {
                args_str.split(',').map(|s| s.trim()).collect()
            };

            self.emit_aarch64_asm_instr(&mnemonic, &args, &operand_map, mf, mbb_id);
        }
    }

    /// Resolves a single AArch64 inline asm operand string.
    ///
    /// Handles:
    /// - `$N` — operand map reference
    /// - `#N` — immediate
    /// - Physical register names (`x0`, `x29`, `w0`, `d0`, `s0`, etc.)
    fn resolve_asm_operand_str(&self, s: &str, operand_map: &[MachineOperand]) -> MachineOperand {
        let s = s.trim();
        if s.starts_with('$') {
            if let Ok(idx) = s[1..].parse::<usize>() {
                if idx < operand_map.len() {
                    return operand_map[idx].clone();
                }
            }
            return MachineOperand::Immediate(0);
        }
        if s.starts_with('#') {
            if let Ok(v) = s[1..].parse::<i64>() {
                return MachineOperand::Immediate(v);
            }
            return MachineOperand::Immediate(0);
        }
        if s.starts_with("$$") {
            if let Ok(v) = s[2..].parse::<i64>() {
                return MachineOperand::Immediate(v);
            }
        }
        // Physical register names.
        if let Some(reg) = self.parse_aarch64_register_name(s) {
            return MachineOperand::Register(reg);
        }
        MachineOperand::Immediate(0)
    }

    /// Parses an AArch64 register name to a PhysReg.
    fn parse_aarch64_register_name(&self, name: &str) -> Option<PhysReg> {
        let name = name.trim().to_lowercase();
        match name.as_str() {
            "x0" => Some(X0),
            "x1" => Some(X1),
            "x2" => Some(X2),
            "x3" => Some(X3),
            "x4" => Some(X4),
            "x5" => Some(X5),
            "x6" => Some(X6),
            "x7" => Some(X7),
            "x8" => Some(X8),
            "x9" => Some(X9),
            "x10" => Some(X10),
            "x11" => Some(X11),
            "x12" => Some(X12),
            "x13" => Some(X13),
            "x14" => Some(X14),
            "x15" => Some(X15),
            "x16" => Some(X16),
            "x17" => Some(X17),
            "x18" => Some(X18),
            "x19" => Some(X19),
            "x20" => Some(X20),
            "x21" => Some(X21),
            "x22" => Some(X22),
            "x23" => Some(X23),
            "x24" => Some(X24),
            "x25" => Some(X25),
            "x26" => Some(X26),
            "x27" => Some(X27),
            "x28" => Some(X28),
            "x29" | "fp" => Some(FP),
            "x30" | "lr" => Some(LR),
            "sp" => Some(SP),
            "xzr" => Some(XZR),
            "w0" => Some(W0),
            "w1" => Some(W1),
            "w2" => Some(W2),
            "w3" => Some(W3),
            "w4" => Some(W4),
            "w5" => Some(W5),
            "w6" => Some(W6),
            "w7" => Some(W7),
            "w8" => Some(W8),
            "w9" => Some(W9),
            "w10" => Some(W10),
            "w11" => Some(W11),
            "w12" => Some(W12),
            "w13" => Some(W13),
            "w14" => Some(W14),
            "w15" => Some(W15),
            "w16" | "ip0" => Some(W16),
            "w17" | "ip1" => Some(W17),
            "w19" => Some(W19),
            "w20" => Some(W20),
            "w21" => Some(W21),
            "wzr" => Some(WZR),
            _ => None,
        }
    }

    /// Emits a single AArch64 machine instruction from parsed asm.
    fn emit_aarch64_asm_instr(
        &mut self,
        mnemonic: &str,
        args: &[&str],
        operand_map: &[MachineOperand],
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        match mnemonic {
            "clz" | "clzw" => {
                // CLZ: "clz" = 64-bit, "clzw" = 32-bit.
                // Pass sf flag as third operand immediate (1=64-bit, 0=32-bit).
                if args.len() >= 2 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src = self.resolve_asm_operand_str(args[1], operand_map);
                    let sf_flag = if mnemonic == "clzw" { 0i64 } else { 1i64 };
                    let instr = MachineInstr::with_operands(
                        AArch64Opcode::CLZ.as_u32(),
                        vec![dst, src, MachineOperand::Immediate(sf_flag)],
                    );
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "rbit" | "rbitw" => {
                // RBIT: "rbit" = 64-bit, "rbitw" = 32-bit.
                if args.len() >= 2 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src = self.resolve_asm_operand_str(args[1], operand_map);
                    let sf_flag = if mnemonic == "rbitw" { 0i64 } else { 1i64 };
                    let instr = MachineInstr::with_operands(
                        AArch64Opcode::RBIT.as_u32(),
                        vec![dst, src, MachineOperand::Immediate(sf_flag)],
                    );
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "rev" | "revw" => {
                // REV: "rev" = 64-bit, "revw" = 32-bit.
                if args.len() >= 2 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src = self.resolve_asm_operand_str(args[1], operand_map);
                    let sf_flag = if mnemonic == "revw" { 0i64 } else { 1i64 };
                    let instr = MachineInstr::with_operands(
                        AArch64Opcode::REV.as_u32(),
                        vec![dst, src, MachineOperand::Immediate(sf_flag)],
                    );
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "add" => {
                // ADD Xd, Xn, #imm or ADD Xd, Xn, Xm
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src1 = self.resolve_asm_operand_str(args[1], operand_map);
                    let src2 = self.resolve_asm_operand_str(args[2], operand_map);
                    let opcode = match &src2 {
                        MachineOperand::Immediate(_) => AArch64Opcode::ADDimm,
                        _ => AArch64Opcode::ADD,
                    };
                    let instr = MachineInstr::with_operands(opcode.as_u32(), vec![dst, src1, src2]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "sub" => {
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src1 = self.resolve_asm_operand_str(args[1], operand_map);
                    let src2 = self.resolve_asm_operand_str(args[2], operand_map);
                    let opcode = match &src2 {
                        MachineOperand::Immediate(_) => AArch64Opcode::SUBimm,
                        _ => AArch64Opcode::SUB,
                    };
                    let instr = MachineInstr::with_operands(opcode.as_u32(), vec![dst, src1, src2]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "mov" => {
                // MOV Xd, Xn (alias for ORR Xd, XZR, Xn)
                if args.len() >= 2 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src = self.resolve_asm_operand_str(args[1], operand_map);
                    match &src {
                        MachineOperand::Immediate(v) => {
                            let instr = MachineInstr::with_operands(
                                AArch64Opcode::MOVZ.as_u32(),
                                vec![dst, MachineOperand::Immediate(*v)],
                            );
                            self.push_instr(mf, mbb_id, instr);
                        }
                        _ => {
                            let instr = MachineInstr::with_operands(
                                AArch64Opcode::ORR.as_u32(),
                                vec![dst, MachineOperand::Register(XZR), src],
                            );
                            self.push_instr(mf, mbb_id, instr);
                        }
                    }
                }
            }
            "and" | "ands" => {
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let s1 = self.resolve_asm_operand_str(args[1], operand_map);
                    let s2 = self.resolve_asm_operand_str(args[2], operand_map);
                    let opcode = match &s2 {
                        MachineOperand::Immediate(_) => AArch64Opcode::ANDimm,
                        _ => AArch64Opcode::AND,
                    };
                    let instr = MachineInstr::with_operands(opcode.as_u32(), vec![dst, s1, s2]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "orr" => {
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let s1 = self.resolve_asm_operand_str(args[1], operand_map);
                    let s2 = self.resolve_asm_operand_str(args[2], operand_map);
                    let instr =
                        MachineInstr::with_operands(AArch64Opcode::ORR.as_u32(), vec![dst, s1, s2]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "eor" => {
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let s1 = self.resolve_asm_operand_str(args[1], operand_map);
                    let s2 = self.resolve_asm_operand_str(args[2], operand_map);
                    let instr =
                        MachineInstr::with_operands(AArch64Opcode::EOR.as_u32(), vec![dst, s1, s2]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "lsr" | "lsl" | "asr" => {
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let s1 = self.resolve_asm_operand_str(args[1], operand_map);
                    let s2 = self.resolve_asm_operand_str(args[2], operand_map);
                    let opcode = match mnemonic {
                        "lsl" => AArch64Opcode::LSL,
                        "lsr" => AArch64Opcode::LSR,
                        "asr" => AArch64Opcode::ASR,
                        _ => AArch64Opcode::LSR,
                    };
                    let instr = MachineInstr::with_operands(opcode.as_u32(), vec![dst, s1, s2]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "li" => {
                // RISC-V style li pseudo (appears in generic templates).
                // We translate to MOVZ for AArch64.
                if args.len() >= 2 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let imm = self.resolve_asm_operand_str(args[1], operand_map);
                    let instr =
                        MachineInstr::with_operands(AArch64Opcode::MOVZ.as_u32(), vec![dst, imm]);
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "addi" => {
                // RISC-V style addi → ADD imm on AArch64
                if args.len() >= 3 {
                    let dst = self.resolve_asm_operand_str(args[0], operand_map);
                    let src = self.resolve_asm_operand_str(args[1], operand_map);
                    let imm = self.resolve_asm_operand_str(args[2], operand_map);
                    let instr = MachineInstr::with_operands(
                        AArch64Opcode::ADDimm.as_u32(),
                        vec![dst, src, imm],
                    );
                    self.push_instr(mf, mbb_id, instr);
                }
            }
            "fmov" => {
                // FMOV: either SIMD-to-GPR or GPR-to-SIMD.
                // For now, emit as a NOP and handle POPCOUNT differently.
                self.push_instr(mf, mbb_id, MachineInstr::new(AArch64Opcode::NOP.as_u32()));
            }
            "cnt" | "addv" => {
                // SIMD popcount instructions - emit NOP (handled by software fallback).
                self.push_instr(mf, mbb_id, MachineInstr::new(AArch64Opcode::NOP.as_u32()));
            }
            "leaq" | "leal" => {
                // x86 LEA in AArch64 context - shouldn't happen, but handle gracefully.
                self.push_instr(mf, mbb_id, MachineInstr::new(AArch64Opcode::NOP.as_u32()));
            }
            _ => {
                // Unknown mnemonic: emit as raw INLINE_ASM NOP to avoid crashes.
                self.push_instr(mf, mbb_id, MachineInstr::new(AArch64Opcode::NOP.as_u32()));
            }
        }
    }

    /// Lowers GCC-style user inline assembly (%N operands) on AArch64.
    fn lower_gcc_user_inline_asm(
        &mut self,
        result: &Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) {
        // For now, handle GCC user asm similarly to builtin but with %N operands.
        // Convert %N to $N and delegate to builtin handler.
        let converted = template
            .replace("%0", "$0")
            .replace("%1", "$1")
            .replace("%2", "$2")
            .replace("%3", "$3")
            .replace("%4", "$4")
            .replace("%5", "$5");
        self.lower_builtin_inline_asm(
            result,
            &converted,
            constraints,
            operands,
            clobbers,
            mf,
            mbb_id,
        );
    }

    /// Ensures an operand is in a register, loading from memory if needed.
    fn ensure_in_register(
        &mut self,
        op: MachineOperand,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) -> MachineOperand {
        match &op {
            MachineOperand::Register(_) | MachineOperand::VirtualReg(_) => op,
            MachineOperand::Immediate(v) => {
                // Use materialize_immediate (MOVZ/MOVK sequence) for ALL
                // immediates.  A bare MOVZ only holds 16 bits and silently
                // truncates larger values — this was the root cause of
                // incorrect CLZ / popcount results for values like 0x80000000.
                self.materialize_immediate(*v, mf, mbb_id)
            }
            MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. } => {
                let vreg = self.alloc_vreg();
                let instr = MachineInstr::with_operands(
                    AArch64Opcode::LDR.as_u32(),
                    vec![vreg.clone(), op],
                );
                self.push_instr(mf, mbb_id, instr);
                vreg
            }
            MachineOperand::Symbol(ref name) => {
                let name_clone = name.clone();
                self.generate_pic_address(&name_clone, mf, mbb_id)
            }
            _ => op,
        }
    }

    // =======================================================================
    // Private: Parameter lowering
    // =======================================================================

    /// Places function parameters into the value map per AAPCS64.
    fn lower_params(&mut self, func: &IrFunction, mf: &mut MachineFunction) {
        let gpr_regs = &INTEGER_ARG_REGS;
        let fpr_regs = &FLOAT_ARG_REGS;
        let mut gpr_idx = 0usize;
        let mut fpr_idx = 0usize;
        let mut stack_offset = 0i32;

        // Determine the entry block in the machine function.
        let entry_mbb = self
            .block_map
            .get(&func.entry_block_id)
            .copied()
            .unwrap_or(0);

        for param in &func.params {
            let class = classify_ir_arg(&param.ty, &self.target);
            match class {
                IrArgClass::FpReg => {
                    if fpr_idx < fpr_regs.len() {
                        let phys = if param.ty.size_bytes(&self.target) <= 4 {
                            v_to_s(fpr_regs[fpr_idx])
                        } else {
                            v_to_d(fpr_regs[fpr_idx])
                        };
                        let ftype: i64 = if param.ty.size_bytes(&self.target) <= 4 {
                            0
                        } else {
                            1
                        };
                        // Cross-domain: FP physical register → GP virtual register.
                        let vreg = self.alloc_vreg();
                        let copy = MachineInstr::with_operands(
                            AArch64Opcode::FMOVtoGPR.as_u32(),
                            vec![
                                vreg.clone(),
                                MachineOperand::Register(phys),
                                MachineOperand::Immediate(ftype),
                            ],
                        );
                        self.push_instr(mf, entry_mbb, copy);
                        self.value_map.insert(param.id, vreg);
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
                        // CRITICAL: Copy from physical ABI register (X0-X7) to
                        // a virtual register at the top of the entry block.
                        // If we map params directly to physical registers, any
                        // subsequent function call will clobber X0-X7 before
                        // the param value is consumed.
                        //
                        // AArch64 "MOV Xd, Xm" is encoded as "ORR Xd, XZR, Xm".
                        let phys = gpr_regs[gpr_idx];
                        let vreg = self.alloc_vreg();
                        let copy = MachineInstr::with_operands(
                            AArch64Opcode::ORR.as_u32(),
                            vec![
                                vreg.clone(),
                                MachineOperand::Register(XZR),
                                MachineOperand::Register(phys),
                            ],
                        );
                        self.push_instr(mf, entry_mbb, copy);
                        self.value_map.insert(param.id, vreg);
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
                    // Pointer to the aggregate passed in a GPR — also needs
                    // copying to a vreg.
                    if gpr_idx < gpr_regs.len() {
                        let phys = gpr_regs[gpr_idx];
                        let vreg = self.alloc_vreg();
                        let copy = MachineInstr::with_operands(
                            AArch64Opcode::ORR.as_u32(),
                            vec![
                                vreg.clone(),
                                MachineOperand::Register(XZR),
                                MachineOperand::Register(phys),
                            ],
                        );
                        self.push_instr(mf, entry_mbb, copy);
                        self.value_map.insert(param.id, vreg);
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn compute_frame_size(&self, mf: &MachineFunction) -> u32 {
        // 16 bytes for FP/LR save.
        let fp_lr_size: u32 = 16;
        // For variadic functions: 64-byte register save area for X0-X7.
        let va_save_size: u32 = if self.is_variadic_func { 64 } else { 0 };
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
        let total = fp_lr_size + va_save_size + callee_save_aligned + local_size;
        (total + max_object_align - 1) & !(max_object_align - 1)
    }

    // =======================================================================
    // Private: Frame index resolution
    // =======================================================================

    /// Resolves all `FrameIndex(i)` operands in the machine function to
    /// concrete `Memory { base: SP, offset: sp_relative }` operands.
    ///
    /// Stack frame layout (low addresses at top):
    /// ```text
    /// [SP + 0]                        saved X29 (FP)
    /// [SP + 8]                        saved X30 (LR)
    /// [SP + 16]                       callee-saved registers (if any)
    /// [SP + 16 + callee_save_aligned] start of local variables area
    /// [SP + frame_size - 1]           top of frame (old SP)
    /// ```
    ///
    /// The `frame_objects[i].offset` is a negative value tracking the
    /// allocation pointer within the local area. The SP-relative offset is:
    /// `local_base + local_size + frame_objects[i].offset`
    #[allow(dead_code)]
    fn resolve_frame_indices(&self, mf: &mut MachineFunction) {
        if self.frame_objects.is_empty() {
            return;
        }

        // Compute local_base: bytes reserved for FP/LR + va_save + callee-saved regs
        let va_save_size: u32 = if self.is_variadic_func { 64 } else { 0 };
        let callee_save_bytes = (mf.used_callee_saved.len() as u32) * 8;
        let callee_save_aligned = (callee_save_bytes + 15) & !15;
        let local_base = 16u32 + va_save_size + callee_save_aligned;

        // local_size: total bytes used by allocas, 16-byte aligned
        let local_size = ((-self.current_frame_offset) as u32 + 15) & !15;

        // Pre-compute SP-relative offset for each frame object
        let mut fi_offsets: Vec<i32> = Vec::with_capacity(self.frame_objects.len());
        for fo in &self.frame_objects {
            // fo.offset is negative; local_size + fo.offset gives offset from
            // start of local area
            let within_local = local_size as i32 + fo.offset;
            let sp_off = local_base as i32 + within_local;
            fi_offsets.push(sp_off);
        }

        // Walk all instructions and replace FrameIndex operands.
        //
        // Context-sensitive replacement:
        //   • If the FrameIndex appears as operand[2] of an ADD whose
        //     operand[1] is Register(SP), it is an **address computation**
        //     (LEA-equivalent).  Replace with Immediate(sp_offset).
        //   • Otherwise, the FrameIndex is a memory reference.
        //     Replace with Memory { base: SP, offset: sp_offset }.
        for block in &mut mf.blocks {
            for instr in &mut block.instructions {
                // Detect the "ADD vreg, SP, FrameIndex(idx)" pattern.
                let is_add_sp_fi = instr.operands.len() >= 3
                    && (instr.opcode == AArch64Opcode::ADDimm.as_u32()
                        || instr.opcode == AArch64Opcode::ADD.as_u32())
                    && matches!(instr.operands[1], MachineOperand::Register(r) if r == SP)
                    && matches!(instr.operands[2], MachineOperand::FrameIndex(_));

                if is_add_sp_fi {
                    // Address computation: replace FrameIndex with Immediate.
                    if let MachineOperand::FrameIndex(idx) = instr.operands[2] {
                        let fi = idx as usize;
                        let sp_off = if fi < fi_offsets.len() {
                            fi_offsets[fi]
                        } else {
                            idx as i32
                        };
                        instr.operands[2] = MachineOperand::Immediate(sp_off as i64);
                    }
                } else {
                    // General case: replace FrameIndex with Memory.
                    for op in &mut instr.operands {
                        if let MachineOperand::FrameIndex(idx) = op {
                            let fi = *idx as usize;
                            let sp_off = if fi < fi_offsets.len() {
                                fi_offsets[fi]
                            } else {
                                *idx as i32
                            };
                            *op = MachineOperand::Memory {
                                base: SP,
                                offset: sp_off,
                                index: None,
                                scale: 1,
                            };
                        }
                    }
                }
            }
        }
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
        let lhs_raw = self.resolve_operand(lhs);
        let rhs_raw = self.resolve_operand(rhs);
        let lhs_op = self.materialize_to_register(lhs_raw, mf, mbb_id);
        let rhs_op = self.materialize_to_register(rhs_raw, mf, mbb_id);

        let is_single = ty.size_bytes(&self.target) <= 4;
        let (reg0, reg1, func_name) = if is_single {
            (v_to_s(V0), v_to_s(V1), "fmodf")
        } else {
            (v_to_d(V0), v_to_d(V1), "fmod")
        };
        let ftype: i64 = if is_single { 0 } else { 1 };

        // GP→FP for arguments
        let mov_lhs = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(reg0),
                lhs_op,
                MachineOperand::Immediate(ftype),
            ],
        );
        self.push_instr(mf, mbb_id, mov_lhs);

        let mov_rhs = MachineInstr::with_operands(
            AArch64Opcode::FMOVint.as_u32(),
            vec![
                MachineOperand::Register(reg1),
                rhs_op,
                MachineOperand::Immediate(ftype),
            ],
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

        // FP→GP for result
        let dest = self.alloc_vreg();
        let mov_res = MachineInstr::with_operands(
            AArch64Opcode::FMOVtoGPR.as_u32(),
            vec![
                dest.clone(),
                MachineOperand::Register(reg0),
                MachineOperand::Immediate(ftype),
            ],
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
            // I32 and F32 use the dedicated 32-bit load opcode so that the
            // encoder emits `size=0b10` (4-byte transfer).  Using the
            // regular `LDR` would produce a 64-bit transfer because the
            // register allocator assigns X-registers to virtual regs,
            // causing 8 bytes to be read — the upper 4 bytes contain the
            // adjacent memory slot, corrupting comparisons.
            IrType::I32 | IrType::F32 => AArch64Opcode::LDRW,
            IrType::I64 | IrType::Ptr | IrType::F64 => AArch64Opcode::LDR,
            _ => AArch64Opcode::LDR,
        }
    }

    fn get_store_opcode(&self, ty: &IrType) -> AArch64Opcode {
        match ty {
            IrType::I1 | IrType::I8 => AArch64Opcode::STRB,
            IrType::I16 => AArch64Opcode::STRH,
            // I32 and F32 use a dedicated 32-bit store opcode for
            // consistency with the load side.
            IrType::I32 | IrType::F32 => AArch64Opcode::STRW,
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

    /// Ensures that `op` is in a register (physical or virtual).
    ///
    /// - `Register` / `VirtualReg` — returned as-is.
    /// - `Immediate` — materialized via `MOVZ` (small) or `MOVZ`+`MOVK` sequence.
    /// - `Symbol` — materialized via `ADRP` + `ADD` (or GOT load in PIC mode).
    /// - Other operand kinds — returned as-is (caller decides).
    fn materialize_to_register(
        &mut self,
        op: MachineOperand,
        mf: &mut MachineFunction,
        mbb_id: u32,
    ) -> MachineOperand {
        match op {
            MachineOperand::Register(_) | MachineOperand::VirtualReg(_) => op,
            MachineOperand::Symbol(ref name) => {
                let name_clone = name.clone();
                self.generate_pic_address(&name_clone, mf, mbb_id)
            }
            MachineOperand::Immediate(val) => {
                if val == 0 {
                    // Zero is best represented as XZR (reads as zero).
                    // But we need it in a GP register, so MOV Xd, XZR.
                    let dest = self.alloc_vreg();
                    let mov = MachineInstr::with_operands(
                        AArch64Opcode::ORR.as_u32(),
                        vec![
                            dest.clone(),
                            MachineOperand::Register(XZR),
                            MachineOperand::Register(XZR),
                        ],
                    );
                    self.push_instr(mf, mbb_id, mov);
                    dest
                } else {
                    // Use the full MOVZ/MOVK sequence for values > 16 bits
                    // to avoid silent truncation.  `materialize_immediate`
                    // handles the multi-halfword case correctly.
                    self.materialize_immediate(val, mf, mbb_id)
                }
            }
            MachineOperand::FrameIndex(idx) => {
                // A FrameIndex used as a *value* means we need the ADDRESS
                // of the frame slot, not its contents.  Emit
                //     ADDimm vreg, SP, FrameIndex(idx)
                // The `resolve_frame_indices` pass will later replace the
                // FrameIndex operand with `Immediate(sp_offset)`, yielding
                //     ADDimm vreg, SP, #offset
                // CRITICAL: Must use ADDimm (not ADD) so that register 31
                // is treated as SP, not XZR, in the encoder.
                let dest = self.alloc_vreg();
                let add = MachineInstr::with_operands(
                    AArch64Opcode::ADDimm.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Register(SP),
                        MachineOperand::FrameIndex(idx),
                    ],
                );
                self.push_instr(mf, mbb_id, add);
                dest
            }
            MachineOperand::Memory { base, offset, .. } => {
                // Memory in value position — compute the effective address
                // via ADDimm dest, base, #offset.
                let dest = self.alloc_vreg();
                let add = MachineInstr::with_operands(
                    AArch64Opcode::ADDimm.as_u32(),
                    vec![
                        dest.clone(),
                        MachineOperand::Register(base),
                        MachineOperand::Immediate(offset as i64),
                    ],
                );
                self.push_instr(mf, mbb_id, add);
                dest
            }
            other => other,
        }
    }

    /// Allocates a fresh virtual register operand.
    fn alloc_vreg(&mut self) -> MachineOperand {
        let id = self.next_vreg;
        self.next_vreg += 1;
        MachineOperand::VirtualReg(ValueId(id))
    }

    /// Returns an existing virtual register for `result`, or allocates a
    /// fresh one and records it in `value_map`.  This is essential for
    /// instructions where the same `ValueId` is defined in multiple
    /// basic blocks (phi-like copies produced by `BitCast` for ternary
    /// expressions) — all definitions must target the **same** virtual
    /// register so that only the executed path's write takes effect at
    /// runtime.
    fn get_or_alloc_vreg(&mut self, result: ValueId) -> MachineOperand {
        if let Some(existing) = self.value_map.get(&result) {
            existing.clone()
        } else {
            let vreg = self.alloc_vreg();
            self.value_map.insert(result, vreg.clone());
            vreg
        }
    }

    /// Best-effort extraction of a PhysReg from a MachineOperand.
    /// Falls back to X8 (scratch register) if the operand is not a register.
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
