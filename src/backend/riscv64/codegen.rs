//! # RISC-V 64 Instruction Selection and Emission
//!
//! Translates BCC IR instructions into RISC-V 64 machine instructions.
//! Covers the RV64IMAFDC ISA (Integer, Multiply/Divide, Atomic,
//! Single-precision Float, Double-precision Float, Compressed).
//!
//! ## Instruction Formats
//! - R-type: register-register operations (ADD, SUB, MUL, etc.)
//! - I-type: register-immediate operations (ADDI, LW, LD, JALR, etc.)
//! - S-type: stores (SW, SD, etc.)
//! - B-type: conditional branches (BEQ, BNE, BLT, BGE, etc.)
//! - U-type: upper-immediate (LUI, AUIPC)
//! - J-type: jumps (JAL)
//!
//! ## Key Design Decision: LUI/AUIPC for Large Constants
//! RISC-V lacks direct 64-bit immediate loads. Large constants require:
//! - LUI rd, upper_20_bits  +  ADDI rd, rd, lower_12_bits  (for 32-bit constants)
//! - For 64-bit: additional shift+add sequences
//! - PC-relative addressing: AUIPC + ADDI/LD for PIC code
//!
//! ## Primary validation target for Linux kernel 6.9 boot.

use crate::backend::riscv64::abi::RiscV64Abi;
use crate::backend::riscv64::registers::*;
use crate::backend::traits::ArgLocation;
use crate::common::target::Target;
use crate::common::types::{CType, TypeQualifiers};
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// Re-export needed traits to keep in schema compliance.
// These are used by callers, even if not directly used in this module's body.
#[allow(unused_imports)]
use crate::backend::riscv64::abi::{FrameLayout, STACK_ALIGNMENT};
#[allow(unused_imports)]
use crate::backend::traits::{
    MachineBasicBlock, MachineFunction, MachineInstruction, MachineOperand, RegisterInfo,
};
#[allow(unused_imports)]
use crate::common::diagnostics::{DiagnosticEngine, Span};
#[allow(unused_imports)]
use crate::common::types::MachineType;
#[allow(unused_imports)]
use crate::ir::basic_block::BasicBlock;
#[allow(unused_imports)]
use crate::ir::instructions::BlockId;

// ===========================================================================
// RvOpcode — RISC-V 64 instruction opcodes
// ===========================================================================

/// RISC-V 64 instruction opcodes for machine code representation.
///
/// Covers the full RV64IMAFDC ISA plus pseudo-instructions used during
/// instruction selection. Each variant maps to one or more hardware
/// instruction encodings resolved by the assembler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum RvOpcode {
    // RV64I Base Integer Instructions
    /// Load Upper Immediate — U-type: rd = imm << 12
    LUI,
    /// Add Upper Immediate to PC — U-type: rd = PC + (imm << 12)
    AUIPC,
    /// Jump And Link — J-type: rd = PC+4; PC += offset
    JAL,
    /// Jump And Link Register — I-type: rd = PC+4; PC = rs1 + imm
    JALR,
    /// Branch if Equal — B-type
    BEQ,
    /// Branch if Not Equal — B-type
    BNE,
    /// Branch if Less Than (signed) — B-type
    BLT,
    /// Branch if Greater or Equal (signed) — B-type
    BGE,
    /// Branch if Less Than (unsigned) — B-type
    BLTU,
    /// Branch if Greater or Equal (unsigned) — B-type
    BGEU,
    /// Load Byte (sign-extend) — I-type
    LB,
    /// Load Halfword (sign-extend) — I-type
    LH,
    /// Load Word (sign-extend) — I-type
    LW,
    /// Load Doubleword — I-type
    LD,
    /// Load Byte Unsigned — I-type
    LBU,
    /// Load Halfword Unsigned — I-type
    LHU,
    /// Load Word Unsigned — I-type
    LWU,
    /// Store Byte — S-type
    SB,
    /// Store Halfword — S-type
    SH,
    /// Store Word — S-type
    SW,
    /// Store Doubleword — S-type
    SD,
    /// Add Immediate — I-type
    ADDI,
    /// Set Less Than Immediate (signed) — I-type
    SLTI,
    /// Set Less Than Immediate Unsigned — I-type
    SLTIU,
    /// XOR Immediate — I-type
    XORI,
    /// OR Immediate — I-type
    ORI,
    /// AND Immediate — I-type
    ANDI,
    /// Shift Left Logical Immediate — I-type
    SLLI,
    /// Shift Right Logical Immediate — I-type
    SRLI,
    /// Shift Right Arithmetic Immediate — I-type
    SRAI,
    /// Add — R-type
    ADD,
    /// Subtract — R-type
    SUB,
    /// Shift Left Logical — R-type
    SLL,
    /// Set Less Than (signed) — R-type
    SLT,
    /// Set Less Than Unsigned — R-type
    SLTU,
    /// XOR — R-type
    XOR,
    /// Shift Right Logical — R-type
    SRL,
    /// Shift Right Arithmetic — R-type
    SRA,
    /// OR — R-type
    OR,
    /// AND — R-type
    AND,

    // RV64I Word variants (32-bit operations on 64-bit, sign-extend result)
    /// Add Immediate Word — sign-extends result to 64 bits
    ADDIW,
    /// Shift Left Logical Immediate Word
    SLLIW,
    /// Shift Right Logical Immediate Word
    SRLIW,
    /// Shift Right Arithmetic Immediate Word
    SRAIW,
    /// Add Word — R-type, sign-extends result
    ADDW,
    /// Subtract Word — R-type, sign-extends result
    SUBW,
    /// Shift Left Logical Word
    SLLW,
    /// Shift Right Logical Word
    SRLW,
    /// Shift Right Arithmetic Word
    SRAW,

    // RV64M Multiply/Divide Extension
    /// Multiply — R-type: rd = (rs1 × rs2)[63:0]
    MUL,
    /// Multiply High (signed×signed) — R-type: rd = (rs1 × rs2)[127:64]
    MULH,
    /// Multiply High Signed-Unsigned — R-type
    MULHSU,
    /// Multiply High Unsigned — R-type
    MULHU,
    /// Divide (signed) — R-type
    DIV,
    /// Divide Unsigned — R-type
    DIVU,
    /// Remainder (signed) — R-type
    REM,
    /// Remainder Unsigned — R-type
    REMU,
    /// Multiply Word — R-type, sign-extends result
    MULW,
    /// Divide Word (signed) — R-type
    DIVW,
    /// Divide Word Unsigned — R-type
    DIVUW,
    /// Remainder Word (signed) — R-type
    REMW,
    /// Remainder Word Unsigned — R-type
    REMUW,

    // RV64A Atomic Extension
    /// Load Reserved Word
    LR_W,
    /// Store Conditional Word
    SC_W,
    /// Load Reserved Doubleword
    LR_D,
    /// Store Conditional Doubleword
    SC_D,
    /// Atomic Swap Word
    AMOSWAP_W,
    /// Atomic Add Word
    AMOADD_W,
    /// Atomic AND Word
    AMOAND_W,
    /// Atomic OR Word
    AMOOR_W,
    /// Atomic XOR Word
    AMOXOR_W,
    /// Atomic Max (signed) Word
    AMOMAX_W,
    /// Atomic Min (signed) Word
    AMOMIN_W,
    /// Atomic Max Unsigned Word
    AMOMAXU_W,
    /// Atomic Min Unsigned Word
    AMOMINU_W,
    /// Atomic Swap Doubleword
    AMOSWAP_D,
    /// Atomic Add Doubleword
    AMOADD_D,
    /// Atomic AND Doubleword
    AMOAND_D,
    /// Atomic OR Doubleword
    AMOOR_D,
    /// Atomic XOR Doubleword
    AMOXOR_D,
    /// Atomic Max (signed) Doubleword
    AMOMAX_D,
    /// Atomic Min (signed) Doubleword
    AMOMIN_D,
    /// Atomic Max Unsigned Doubleword
    AMOMAXU_D,
    /// Atomic Min Unsigned Doubleword
    AMOMINU_D,

    // RV64F Single-Precision Float Extension
    /// Load Float Word — I-type: fd = mem[rs1+imm]
    FLW,
    /// Store Float Word — S-type: mem[rs1+imm] = fs2
    FSW,
    /// Float Add Single
    FADD_S,
    /// Float Subtract Single
    FSUB_S,
    /// Float Multiply Single
    FMUL_S,
    /// Float Divide Single
    FDIV_S,
    /// Float Square Root Single
    FSQRT_S,
    /// Float Minimum Single
    FMIN_S,
    /// Float Maximum Single
    FMAX_S,
    /// Convert Float to Signed 32-bit Int
    FCVT_W_S,
    /// Convert Float to Unsigned 32-bit Int
    FCVT_WU_S,
    /// Convert Float to Signed 64-bit Int
    FCVT_L_S,
    /// Convert Float to Unsigned 64-bit Int
    FCVT_LU_S,
    /// Convert Signed 32-bit Int to Float
    FCVT_S_W,
    /// Convert Unsigned 32-bit Int to Float
    FCVT_S_WU,
    /// Convert Signed 64-bit Int to Float
    FCVT_S_L,
    /// Convert Unsigned 64-bit Int to Float
    FCVT_S_LU,
    /// Move Float to Integer Register
    FMV_X_W,
    /// Move Integer to Float Register
    FMV_W_X,
    /// Float Equal Single
    FEQ_S,
    /// Float Less Than Single
    FLT_S,
    /// Float Less or Equal Single
    FLE_S,
    /// Fused Multiply-Add Single: rd = rs1×rs2 + rs3
    FMADD_S,
    /// Fused Multiply-Subtract Single: rd = rs1×rs2 − rs3
    FMSUB_S,
    /// Negated Fused Multiply-Subtract Single: rd = −(rs1×rs2 − rs3)
    FNMSUB_S,
    /// Negated Fused Multiply-Add Single: rd = −(rs1×rs2 + rs3)
    FNMADD_S,
    /// Float Sign-Inject Single (copy sign)
    FSGNJ_S,
    /// Float Sign-Inject Negate Single
    FSGNJN_S,
    /// Float Sign-Inject XOR Single (abs via self-xor)
    FSGNJX_S,
    /// Float Classify Single
    FCLASS_S,

    // RV64D Double-Precision Float Extension
    /// Load Float Double — I-type: fd = mem[rs1+imm]
    FLD,
    /// Store Float Double — S-type: mem[rs1+imm] = fs2
    FSD,
    /// Float Add Double
    FADD_D,
    /// Float Subtract Double
    FSUB_D,
    /// Float Multiply Double
    FMUL_D,
    /// Float Divide Double
    FDIV_D,
    /// Float Square Root Double
    FSQRT_D,
    /// Float Minimum Double
    FMIN_D,
    /// Float Maximum Double
    FMAX_D,
    /// Convert Double to Signed 32-bit Int
    FCVT_W_D,
    /// Convert Double to Unsigned 32-bit Int
    FCVT_WU_D,
    /// Convert Double to Signed 64-bit Int
    FCVT_L_D,
    /// Convert Double to Unsigned 64-bit Int
    FCVT_LU_D,
    /// Convert Signed 32-bit Int to Double
    FCVT_D_W,
    /// Convert Unsigned 32-bit Int to Double
    FCVT_D_WU,
    /// Convert Signed 64-bit Int to Double
    FCVT_D_L,
    /// Convert Unsigned 64-bit Int to Double
    FCVT_D_LU,
    /// Convert Single to Double
    FCVT_S_D,
    /// Convert Double to Single
    FCVT_D_S,
    /// Move Double to Integer Register (64-bit)
    FMV_X_D,
    /// Move Integer (64-bit) to Double Register
    FMV_D_X,
    /// Float Equal Double
    FEQ_D,
    /// Float Less Than Double
    FLT_D,
    /// Float Less or Equal Double
    FLE_D,
    /// Fused Multiply-Add Double
    FMADD_D,
    /// Fused Multiply-Subtract Double
    FMSUB_D,
    /// Negated Fused Multiply-Subtract Double
    FNMSUB_D,
    /// Negated Fused Multiply-Add Double
    FNMADD_D,
    /// Float Sign-Inject Double
    FSGNJ_D,
    /// Float Sign-Inject Negate Double
    FSGNJN_D,
    /// Float Sign-Inject XOR Double
    FSGNJX_D,
    /// Float Classify Double
    FCLASS_D,

    // System / CSR instructions
    /// CSR Read-Write: rd = csr; csr = rs1
    CSRRW,
    /// CSR Read-Set: rd = csr; csr |= rs1
    CSRRS,
    /// CSR Read-Clear: rd = csr; csr &= ~rs1
    CSRRC,
    /// CSR Read-Write Immediate: rd = csr; csr = zimm
    CSRRWI,
    /// CSR Read-Set Immediate: rd = csr; csr |= zimm
    CSRRSI,
    /// CSR Read-Clear Immediate: rd = csr; csr &= ~zimm
    CSRRCI,
    /// Environment Call (triggers ecall exception)
    ECALL,
    /// Environment Breakpoint (triggers breakpoint exception)
    EBREAK,
    /// Memory Fence (ordering predecessor/successor sets)
    FENCE,
    /// Instruction Fence
    FENCE_I,
    /// Wait For Interrupt (hint to stall until next interrupt)
    WFI,
    /// Supervisor Fence Virtual Memory
    SFENCE_VMA,
    /// Machine Return (return from M-mode trap)
    MRET,
    /// Supervisor Return (return from S-mode trap)
    SRET,

    // Pseudo-instructions (expanded by assembler)
    /// No operation: ADDI x0, x0, 0
    NOP,
    /// Move register: ADDI rd, rs, 0
    MV,
    /// Load immediate (multi-instruction LUI+ADDI sequence)
    LI,
    /// Load address (AUIPC+ADDI sequence for PC-relative)
    LA,
    /// Function call (AUIPC+JALR or JAL for near calls)
    CALL,
    /// Return from function: JALR x0, ra, 0
    RET,
    /// Negate: SUB rd, x0, rs
    NEG,
    /// Bitwise NOT: XORI rd, rs, -1
    NOT,
    /// Set if Equal to Zero: SLTIU rd, rs, 1
    SEQZ,
    /// Set if Not Equal to Zero: SLTU rd, x0, rs
    SNEZ,
    /// Unconditional jump: JAL x0, offset
    J,
    /// Inline assembly passthrough marker
    INLINE_ASM,
}

// ===========================================================================
// RvInstruction — RISC-V 64 machine instruction
// ===========================================================================

/// A RISC-V 64 machine instruction produced by instruction selection.
///
/// Fields `rd`, `rs1`, `rs2`, `rs3` hold register IDs from the register
/// namespace defined in `registers.rs` (0–31 for GPRs, 32–63 for FPRs).
/// The `symbol` field carries relocation references resolved by the linker.
#[derive(Debug, Clone)]
pub struct RvInstruction {
    /// The instruction opcode identifying the operation.
    pub opcode: RvOpcode,
    /// Destination register, or `None` for instructions without a result.
    pub rd: Option<u16>,
    /// First source register.
    pub rs1: Option<u16>,
    /// Second source register.
    pub rs2: Option<u16>,
    /// Third source register (fused multiply-add only).
    pub rs3: Option<u16>,
    /// Immediate value (12-bit signed for I-type, 20-bit for U-type, etc.).
    pub imm: i64,
    /// Symbol reference for linker relocations (function/global names).
    pub symbol: Option<String>,
    /// Whether this instruction operates on floating-point registers.
    pub is_fp: bool,
    /// Optional debug annotation for assembly listing.
    pub comment: Option<String>,
    /// Whether this instruction is part of a call-argument setup sequence.
    /// Used by the post-register-allocation parallel-move resolver
    /// (`resolve_call_arg_conflicts`) to identify the contiguous window of
    /// argument-loading instructions that precede a CALL.
    pub is_call_arg_setup: bool,
}

impl RvInstruction {
    /// Create a new R-type instruction (register-register).
    fn r_type(opcode: RvOpcode, rd: u16, rs1: u16, rs2: u16) -> Self {
        RvInstruction {
            opcode,
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: Some(rs2),
            rs3: None,
            imm: 0,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    /// Create a new I-type instruction (register-immediate).
    fn i_type(opcode: RvOpcode, rd: u16, rs1: u16, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: None,
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    /// Create a new S-type instruction (store).
    fn s_type(opcode: RvOpcode, rs1: u16, rs2: u16, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: None,
            rs1: Some(rs1),
            rs2: Some(rs2),
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    /// Create a new B-type instruction (conditional branch).
    fn b_type(opcode: RvOpcode, rs1: u16, rs2: u16, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: None,
            rs1: Some(rs1),
            rs2: Some(rs2),
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    /// Create a new U-type instruction (upper immediate).
    fn u_type(opcode: RvOpcode, rd: u16, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: Some(rd),
            rs1: None,
            rs2: None,
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    /// Create a no-operand instruction (e.g., NOP, RET).
    fn no_op(opcode: RvOpcode) -> Self {
        RvInstruction {
            opcode,
            rd: None,
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    /// Set the floating-point flag on this instruction.
    fn with_fp(mut self) -> Self {
        self.is_fp = true;
        self
    }

    /// Mark this instruction as part of a call-argument setup sequence.
    /// The post-register-allocation parallel-move resolver uses this flag
    /// to identify the contiguous window of argument-loading instructions
    /// that precede a CALL instruction.
    #[allow(clippy::wrong_self_convention)]
    fn as_call_arg_setup(mut self) -> Self {
        self.is_call_arg_setup = true;
        self
    }

    /// Set a symbol reference on this instruction.
    #[allow(dead_code)]
    fn with_symbol(mut self, sym: String) -> Self {
        self.symbol = Some(sym);
        self
    }

    /// Set a debug comment on this instruction.
    #[allow(dead_code)]
    fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }
}

// ===========================================================================
// Helper: 12-bit signed immediate range check
// ===========================================================================

/// Check whether an immediate value fits in a 12-bit signed field (I-type).
#[inline]
fn fits_i12(val: i64) -> bool {
    (-2048..=2047).contains(&val)
}

/// Check whether an immediate fits in a 32-bit signed value (LUI+ADDI).
#[inline]
fn fits_i32(val: i64) -> bool {
    (i32::MIN as i64..=i32::MAX as i64).contains(&val)
}

/// Decompose a 32-bit value into LUI upper 20 bits and ADDI lower 12 bits,
/// handling the sign-extension gotcha: if bit 11 of the lower portion is set,
/// the ADDI will sign-extend and subtract 0x1000, so we compensate by adding
/// 1 to the upper portion.
fn split_i32_lui_addi(val: i64) -> (i64, i64) {
    let val32 = val as i32;
    let lo12 = ((val32 as i64) << 52 >> 52) as i32; // sign-extended lower 12
    let hi20 = ((val32 as u32).wrapping_sub(lo12 as u32) >> 12) as i32;
    let hi20_masked = hi20 & 0xFFFFF_i32;
    (hi20_masked as i64, lo12 as i64)
}

// ===========================================================================
// Virtual register tracking
// ===========================================================================

/// Maps IR Value indices to physical/virtual register IDs or stack offsets.
/// During instruction selection, IR values are mapped to virtual registers
/// that are later resolved by the register allocator.
struct ValueMap {
    /// Maps IR Value index → register ID (physical or virtual).
    regs: crate::common::fx_hash::FxHashMap<u32, u16>,
    /// Maps IR Value index → stack frame offset for spilled/alloca'd values.
    stack_offsets: crate::common::fx_hash::FxHashMap<u32, i64>,
    /// Next virtual register number for allocation.
    next_vreg: u32,
}

impl ValueMap {
    fn new() -> Self {
        ValueMap {
            regs: crate::common::fx_hash::FxHashMap::default(),
            stack_offsets: crate::common::fx_hash::FxHashMap::default(),
            next_vreg: 100, // virtual register IDs start above physical (0-63)
        }
    }

    /// Allocate a virtual GPR for an IR value.
    ///
    /// Returns a virtual register ID (≥ 100) that will be resolved to
    /// a physical register by the register allocator in `generation.rs`.
    fn alloc_gpr(&mut self, val: Value) -> u16 {
        let idx = val.index();
        if let Some(&reg) = self.regs.get(&idx) {
            return reg;
        }
        let vreg = self.next_vreg as u16;
        self.next_vreg += 1;
        self.regs.insert(idx, vreg);
        vreg
    }

    /// Allocate a virtual FPR for an IR value.
    ///
    /// Returns a virtual register ID that will be resolved to a physical
    /// FPR by the register allocator.
    fn alloc_fpr(&mut self, val: Value) -> u16 {
        let idx = val.index();
        if let Some(&reg) = self.regs.get(&idx) {
            return reg;
        }
        let vreg = self.next_vreg as u16;
        self.next_vreg += 1;
        self.regs.insert(idx, vreg);
        vreg
    }

    /// Get the register for an IR value, defaulting to T0.
    fn get_reg(&self, val: Value) -> u16 {
        *self.regs.get(&val.index()).unwrap_or(&T0)
    }

    /// Check if a register has been allocated for the given Value.
    fn has_reg(&self, val: Value) -> bool {
        self.regs.contains_key(&val.index())
    }

    /// Get the stack offset for an IR value (alloca'd value).
    #[allow(dead_code)]
    fn get_stack_offset(&self, val: Value) -> Option<i64> {
        self.stack_offsets.get(&val.index()).copied()
    }

    /// Record a stack offset for an IR value.
    fn set_stack_offset(&mut self, val: Value, offset: i64) {
        self.stack_offsets.insert(val.index(), offset);
    }

    /// Record a register mapping for an IR value.
    fn set_reg(&mut self, val: Value, reg: u16) {
        self.regs.insert(val.index(), reg);
    }

    /// Build a mapping from virtual register ID → IR Value for the
    /// register allocator.  Only includes virtual registers (≥ 100).
    fn vreg_to_ir_value_map(&self) -> crate::common::fx_hash::FxHashMap<u32, Value> {
        let mut map = crate::common::fx_hash::FxHashMap::default();
        for (&val_idx, &reg) in &self.regs {
            if reg >= 100 {
                map.insert(reg as u32, Value(val_idx));
            }
        }
        map
    }
}

// ===========================================================================
// RiscV64InstructionSelector — Main instruction selection engine
// ===========================================================================

/// RISC-V 64 instruction selector.
///
/// Translates phi-eliminated IR functions into sequences of `RvInstruction`s
/// that conform to the RV64IMAFDC ISA. Handles:
/// - Integer and floating-point arithmetic
/// - Memory loads and stores (byte through doubleword)
/// - Comparisons and branches (with ±4 KiB offset awareness)
/// - Large immediate materialization (LUI+ADDI, multi-instruction 64-bit)
/// - Function call lowering per LP64D ABI
/// - Prologue/epilogue generation with callee-saved register preservation
/// - Inline assembly passthrough
/// - PIC (position-independent code) addressing via GOT/PLT
/// - Switch statements via cascaded comparisons or jump tables
pub struct RiscV64InstructionSelector {
    /// Accumulated machine instructions for the current function.
    pub instructions: Vec<RvInstruction>,
    /// Current basic block label.
    current_block: Option<String>,
    /// Computed frame size in bytes (aligned to 16).
    pub frame_size: i64,
    /// Spill slots: (virtual_reg_id, stack_offset) pairs.
    spill_slots: Vec<(u32, i64)>,
    /// Whether PIC mode is enabled (`-fPIC`).
    pub pic_mode: bool,
    /// Target architecture reference.
    target: Target,
    /// Value-to-register and value-to-stack mapping.
    vmap: ValueMap,
    /// Current stack allocation offset (grows downward from FP).
    alloca_offset: i64,
    /// Callee-saved GPRs actually used in this function.
    used_callee_saved_gprs: Vec<u16>,
    /// Callee-saved FPRs actually used in this function.
    used_callee_saved_fprs: Vec<u16>,
    /// Block label map: IR BlockId index → string label.
    block_labels: crate::common::fx_hash::FxHashMap<usize, String>,
    /// Whether the current function makes any calls.
    has_calls: bool,
    /// Constant value cache: maps IR Value index to integer constant value.
    constant_values: crate::common::fx_hash::FxHashMap<u32, i64>,
    /// Float constant cache: maps IR Value index to .rodata global symbol name.
    /// Populated from `IrFunction::float_constant_values` so the instruction
    /// selector can load FP constants via `la + fld` instead of integer LI.
    float_constant_cache: crate::common::fx_hash::FxHashMap<u32, (String, f64)>,
    /// Maps IR Value indices to function names for direct calls.
    /// Populated from `IrModule::func_ref_map` during lowering.
    func_ref_names: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    /// Set of function names that are declared as variadic (`...`).
    /// On RISC-V LP64D, variadic function calls pass ALL FP arguments
    /// in integer registers (not FP registers), so the codegen must
    /// convert FPR values to GPR before the call.
    variadic_functions: crate::common::fx_hash::FxHashSet<String>,
    /// Maps IR Value indices to global variable names.
    global_var_refs: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    /// Maps IR Value indices to their IR types, populated during instruction
    /// selection so that Store can choose the correct width (SB/SH/SW/SD).
    value_types: crate::common::fx_hash::FxHashMap<u32, crate::ir::types::IrType>,
}

impl RiscV64InstructionSelector {
    /// Create a new RISC-V 64 instruction selector.
    ///
    /// # Arguments
    /// - `target` — Must be `Target::RiscV64`.
    /// - `pic_mode` — Whether to generate position-independent code.
    pub fn new(target: Target, pic_mode: bool) -> Self {
        RiscV64InstructionSelector {
            instructions: Vec::new(),
            current_block: None,
            frame_size: 0,
            spill_slots: Vec::new(),
            pic_mode,
            target,
            vmap: ValueMap::new(),
            alloca_offset: 0,
            used_callee_saved_gprs: Vec::new(),
            used_callee_saved_fprs: Vec::new(),
            block_labels: crate::common::fx_hash::FxHashMap::default(),
            has_calls: false,
            constant_values: crate::common::fx_hash::FxHashMap::default(),
            float_constant_cache: crate::common::fx_hash::FxHashMap::default(),
            func_ref_names: crate::common::fx_hash::FxHashMap::default(),
            variadic_functions: crate::common::fx_hash::FxHashSet::default(),
            global_var_refs: crate::common::fx_hash::FxHashMap::default(),
            value_types: crate::common::fx_hash::FxHashMap::default(),
        }
    }

    /// Set the function reference name map for resolving direct call targets.
    pub fn set_func_ref_names(
        &mut self,
        map: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.func_ref_names = map;
    }

    /// Set the global variable reference map.
    pub fn set_global_var_refs(
        &mut self,
        map: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.global_var_refs = map;
    }

    /// Set the variadic function names so the call emitter knows to pass
    /// FP arguments in integer registers (RISC-V LP64D ABI requirement).
    pub fn set_variadic_functions(&mut self, set: crate::common::fx_hash::FxHashSet<String>) {
        self.variadic_functions = set;
    }

    /// Set the constant value cache (called before `select_function`).
    pub fn set_constant_values(&mut self, cv: crate::common::fx_hash::FxHashMap<u32, i64>) {
        self.constant_values = cv;
    }

    /// Set the float constant cache for FP constant loading from .rodata.
    pub fn set_float_constant_values(
        &mut self,
        fv: crate::common::fx_hash::FxHashMap<u32, (String, f64)>,
    ) {
        self.float_constant_cache = fv;
    }

    /// Return the virtual register → IR Value mapping for the register
    /// allocator.  Must be called after `select_function`.
    pub fn vreg_to_ir_value_map(&self) -> crate::common::fx_hash::FxHashMap<u32, Value> {
        self.vmap.vreg_to_ir_value_map()
    }

    /// Reset selector state for a new function.
    fn reset(&mut self) {
        self.instructions.clear();
        self.current_block = None;
        self.frame_size = 0;
        self.spill_slots.clear();
        self.vmap = ValueMap::new();
        // Reserve 16 bytes at the top of the frame for RA and FP saves.
        // Local allocas will start below FP - 16, avoiding overlap with
        // the saved return address at FP - 8 and saved frame pointer at FP - 16.
        self.alloca_offset = 16;
        self.used_callee_saved_gprs.clear();
        self.used_callee_saved_fprs.clear();
        self.block_labels.clear();
        self.has_calls = false;
        // NOTE: constant_values is NOT cleared here — it is set before
        // select_function and must survive the reset.
    }

    // -----------------------------------------------------------------------
    // Block label management
    // -----------------------------------------------------------------------

    /// Get or create the label string for a block index.
    fn block_label(&mut self, block_idx: usize) -> String {
        if let Some(lbl) = self.block_labels.get(&block_idx) {
            lbl.clone()
        } else {
            let lbl = format!(".LBB_{}", block_idx);
            self.block_labels.insert(block_idx, lbl.clone());
            lbl
        }
    }

    /// Emit a label pseudo-instruction for a basic block.
    ///
    /// Emits a NOP carrying a `.label:` comment so that the assembler
    /// can record the label's offset during code emission.
    fn emit_block_label(&mut self, label: &str) {
        self.current_block = Some(label.to_string());
        // Emit a NOP that carries the label definition.
        let mut inst = RvInstruction::no_op(RvOpcode::NOP);
        inst.comment = Some(format!(".label:{}", label));
        self.instructions.push(inst);
    }

    // -----------------------------------------------------------------------
    // Register selection helpers
    // -----------------------------------------------------------------------

    /// Determine whether an IR type should use floating-point registers.
    fn uses_fpr(ty: &IrType) -> bool {
        matches!(ty, IrType::F32 | IrType::F64)
    }

    /// Get a destination register for an IR value given its type.
    fn dest_reg(&mut self, val: Value, ty: &IrType) -> u16 {
        if Self::uses_fpr(ty) {
            self.vmap.alloc_fpr(val)
        } else {
            self.vmap.alloc_gpr(val)
        }
    }

    /// Get the source register for an IR value.
    fn src_reg(&mut self, val: Value) -> u16 {
        // If the Value already has a register allocated, just return it.
        if self.vmap.has_reg(val) {
            return self.vmap.get_reg(val);
        }

        // Check if the Value refers to a global variable whose address
        // should be loaded via an LA pseudo-instruction.
        if let Some(name) = self.global_var_refs.get(&val).cloned() {
            let rd = self.vmap.alloc_gpr(val);
            self.emit(RvInstruction {
                opcode: RvOpcode::LA,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(name),
                is_fp: false,
                comment: Some("load global address".to_string()),
                is_call_arg_setup: false,
            });
            return rd;
        }

        // Check if the Value refers to a function whose address is
        // needed (function pointer decay).
        if let Some(name) = self.func_ref_names.get(&val).cloned() {
            let rd = self.vmap.alloc_gpr(val);
            self.emit(RvInstruction {
                opcode: RvOpcode::LA,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(name),
                is_fp: false,
                comment: Some("load function address".to_string()),
                is_call_arg_setup: false,
            });
            return rd;
        }

        // Fall back to normal register lookup (allocates on first use).
        self.vmap.get_reg(val)
    }

    /// Track usage of a callee-saved register.
    fn mark_callee_saved(&mut self, reg: u16) {
        if is_callee_saved(reg) {
            if is_gpr(reg) && !self.used_callee_saved_gprs.contains(&reg) {
                self.used_callee_saved_gprs.push(reg);
            } else if is_fpr(reg) && !self.used_callee_saved_fprs.contains(&reg) {
                self.used_callee_saved_fprs.push(reg);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Emit helpers — push instructions to the output
    // -----------------------------------------------------------------------

    fn emit(&mut self, inst: RvInstruction) {
        if let Some(rd) = inst.rd {
            self.mark_callee_saved(rd);
        }
        self.instructions.push(inst);
    }

    fn emit_r(&mut self, op: RvOpcode, rd: u16, rs1: u16, rs2: u16) {
        self.emit(RvInstruction::r_type(op, rd, rs1, rs2));
    }

    fn emit_i(&mut self, op: RvOpcode, rd: u16, rs1: u16, imm: i64) {
        self.emit(RvInstruction::i_type(op, rd, rs1, imm));
    }

    fn emit_s(&mut self, op: RvOpcode, base: u16, src: u16, offset: i64) {
        self.emit(RvInstruction::s_type(op, base, src, offset));
    }

    fn emit_b(&mut self, op: RvOpcode, rs1: u16, rs2: u16, target_label: &str) {
        let mut inst = RvInstruction::b_type(op, rs1, rs2, 0);
        inst.symbol = Some(target_label.to_string());
        self.emit(inst);
    }

    #[allow(dead_code)]
    fn emit_u(&mut self, op: RvOpcode, rd: u16, imm: i64) {
        self.emit(RvInstruction::u_type(op, rd, imm));
    }

    fn emit_fp_r(&mut self, op: RvOpcode, rd: u16, rs1: u16, rs2: u16) {
        self.emit(RvInstruction::r_type(op, rd, rs1, rs2).with_fp());
    }

    fn emit_fp_unary(&mut self, op: RvOpcode, rd: u16, rs1: u16) {
        let mut inst = RvInstruction::i_type(op, rd, rs1, 0);
        inst.is_fp = true;
        inst.rs2 = None;
        self.emit(inst);
    }

    // ===================================================================
    // Immediate materialization
    // ===================================================================

    /// Materialize a constant integer into a register.
    ///
    /// Handles the full range of 64-bit values:
    /// - 12-bit signed: single ADDI
    /// - 32-bit: LUI + ADDI (with sign-extension correction)
    /// - 64-bit: multi-instruction sequence (LUI+ADDI+SLLI+ADDI chains)
    pub fn materialize_immediate(&mut self, rd: u16, value: i64) -> Vec<RvInstruction> {
        let mut result = Vec::new();

        if value == 0 {
            result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, 0));
            return result;
        }

        if fits_i12(value) {
            result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, value));
            return result;
        }

        if fits_i32(value) {
            let (hi, lo) = split_i32_lui_addi(value);
            if hi != 0 {
                result.push(RvInstruction::u_type(RvOpcode::LUI, rd, hi));
            }
            if lo != 0 {
                let base = if hi != 0 { rd } else { ZERO };
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, base, lo));
            }
            return result;
        }

        // 64-bit value: decompose into shift-add sequences.
        Self::materialize_i64_into(&mut result, rd, value);
        result
    }

    /// Internal helper for materializing 64-bit immediates.
    fn materialize_i64_into(result: &mut Vec<RvInstruction>, rd: u16, value: i64) {
        let upper32 = (value >> 32) as i32 as i64;
        let lower32_u = (value as u64) & 0xFFFF_FFFF;

        // Load upper 32 bits as a 32-bit constant
        if fits_i12(upper32) {
            result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, upper32));
        } else {
            let (hi, lo) = split_i32_lui_addi(upper32);
            if hi != 0 {
                result.push(RvInstruction::u_type(RvOpcode::LUI, rd, hi));
            }
            if lo != 0 {
                let base = if hi != 0 { rd } else { ZERO };
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, base, lo));
            } else if hi == 0 {
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, 0));
            }
        }

        // Shift left by 32
        result.push(RvInstruction::i_type(RvOpcode::SLLI, rd, rd, 32));

        // Add lower 32 bits if non-zero.
        //
        // On RV64, LUI sign-extends the 32-bit result into 64 bits.  If
        // bit 31 of the lower 32 bits is set (e.g. lower32 = 0xFFFF0000),
        // LUI produces a negative 64-bit value.  We must zero-extend T0
        // with SLLI+SRLI before adding it to the shifted upper portion.
        if lower32_u != 0 {
            // Decompose lower 32 bits: upper 20 + lower 12
            let lo_raw = lower32_u as i64;
            let lo_lo = (lo_raw << 52) >> 52; // sign-extend low 12
            let lo_hi = ((lower32_u.wrapping_sub(lo_lo as u64)) >> 12) as i64 & 0xFFFFF;

            if lo_hi != 0 {
                // Use T0 as temporary to build lower 32 bits
                result.push(RvInstruction::u_type(RvOpcode::LUI, T0, lo_hi));
                if lo_lo != 0 {
                    result.push(RvInstruction::i_type(RvOpcode::ADDI, T0, T0, lo_lo));
                }
                // If bit 31 of the lower 32 is set, LUI sign-extends on
                // RV64 and contaminates the upper 32 bits.  Zero-extend
                // T0 with a SLLI+SRLI pair to clear the upper 32 bits.
                if lower32_u & 0x8000_0000 != 0 {
                    result.push(RvInstruction::i_type(RvOpcode::SLLI, T0, T0, 32));
                    result.push(RvInstruction::i_type(RvOpcode::SRLI, T0, T0, 32));
                }
                result.push(RvInstruction::r_type(RvOpcode::ADD, rd, rd, T0));
            } else if lo_lo != 0 {
                // lo_lo fits in 12 bits — ADDI sign-extends, but since
                // lo_hi == 0 the value is small enough that no upper-bit
                // contamination occurs (< 2048 or >= -2048 in the low
                // portion means bit 31 is not set from LUI).
                // However, if bit 11 is set in lo_lo, ADDI sign-extends
                // it to fill bits 12-63 of the result.  When the destination
                // is rd (which holds the upper portion shifted by 32),
                // this sign-extension contaminates the result.  Use T0
                // and zero-extend when lo_lo is negative.
                if lo_lo < 0 {
                    result.push(RvInstruction::i_type(RvOpcode::ADDI, T0, ZERO, lo_lo));
                    result.push(RvInstruction::i_type(RvOpcode::SLLI, T0, T0, 32));
                    result.push(RvInstruction::i_type(RvOpcode::SRLI, T0, T0, 32));
                    result.push(RvInstruction::r_type(RvOpcode::ADD, rd, rd, T0));
                } else {
                    result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, rd, lo_lo));
                }
            }
        }
    }

    /// Emit immediate materialization inline (adds to self.instructions).
    fn emit_immediate(&mut self, rd: u16, value: i64) {
        let insts = self.materialize_immediate(rd, value);
        for inst in insts {
            self.instructions.push(inst);
        }
    }

    // ===================================================================
    // Address materialization
    // ===================================================================

    /// Materialize a symbol address into a register.
    ///
    /// - **Non-PIC**: `LUI rd, %hi(sym)` + `ADDI rd, rd, %lo(sym)`
    /// - **PIC (GOT)**: `AUIPC rd, %got_pcrel_hi(sym)` + `LD rd, rd, %pcrel_lo(.)`
    pub fn materialize_address(&mut self, rd: u16, symbol: &str) -> Vec<RvInstruction> {
        let mut result = Vec::new();

        if self.pic_mode {
            result.push(RvInstruction {
                opcode: RvOpcode::AUIPC,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%got_pcrel_hi({})", symbol)),
                is_fp: false,
                comment: Some(format!("GOT hi: {}", symbol)),
                is_call_arg_setup: false,
            });
            result.push(RvInstruction {
                opcode: RvOpcode::LD,
                rd: Some(rd),
                rs1: Some(rd),
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%pcrel_lo(.L_got_{})", symbol)),
                is_fp: false,
                comment: Some(format!("GOT ld: {}", symbol)),
                is_call_arg_setup: false,
            });
        } else {
            result.push(RvInstruction {
                opcode: RvOpcode::LUI,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%hi({})", symbol)),
                is_fp: false,
                comment: Some(format!("addr hi: {}", symbol)),
                is_call_arg_setup: false,
            });
            result.push(RvInstruction {
                opcode: RvOpcode::ADDI,
                rd: Some(rd),
                rs1: Some(rd),
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%lo({})", symbol)),
                is_fp: false,
                comment: Some(format!("addr lo: {}", symbol)),
                is_call_arg_setup: false,
            });
        }

        result
    }

    /// Emit address materialization inline.
    fn emit_address(&mut self, rd: u16, symbol: &str) {
        let insts = self.materialize_address(rd, symbol);
        for inst in insts {
            self.instructions.push(inst);
        }
    }

    /// Materialize a PC-relative address (for local PIC symbols).
    fn emit_pcrel_address(&mut self, rd: u16, symbol: &str) {
        self.emit(RvInstruction {
            opcode: RvOpcode::AUIPC,
            rd: Some(rd),
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: Some(format!("%pcrel_hi({})", symbol)),
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        });
        self.emit(RvInstruction {
            opcode: RvOpcode::ADDI,
            rd: Some(rd),
            rs1: Some(rd),
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: Some(format!("%pcrel_lo(.L_pcrel_{})", symbol)),
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        });
    }

    // ===================================================================
    // Load instruction selection
    // ===================================================================

    /// Emit a load instruction from memory to a register.
    ///
    /// Selects the appropriate load opcode based on the type size and
    /// sign-extension requirements:
    /// - I1/I8 signed → LB, unsigned → LBU
    /// - I16 signed → LH, unsigned → LHU
    /// - I32 signed → LW, unsigned → LWU
    /// - I64/Ptr → LD
    /// - F32 → FLW
    /// - F64 → FLD
    pub fn emit_load(&mut self, rd: u16, base: u16, offset: i64, ty: &IrType) {
        if !fits_i12(offset) {
            // Large offset: materialize into T0, then ADD to base
            self.emit_immediate(T0, offset);
            self.emit_r(RvOpcode::ADD, T0, base, T0);
            self.emit_load(rd, T0, 0, ty);
            return;
        }

        let opcode = match ty {
            IrType::I1 | IrType::I8 => RvOpcode::LBU,
            IrType::I16 => RvOpcode::LH,
            IrType::I32 => RvOpcode::LW,
            IrType::I64 | IrType::I128 | IrType::Ptr => RvOpcode::LD,
            IrType::F32 => RvOpcode::FLW,
            IrType::F64 | IrType::F80 => RvOpcode::FLD,
            IrType::Void => return,
            IrType::Array(_, _) | IrType::Struct(_) | IrType::Function(_, _) => {
                // Aggregate types: load as pointer (address)
                RvOpcode::LD
            }
        };

        let is_fp = matches!(ty, IrType::F32 | IrType::F64 | IrType::F80);
        let mut inst = RvInstruction::i_type(opcode, rd, base, offset);
        inst.is_fp = is_fp;
        self.emit(inst);
    }

    // ===================================================================
    // Store instruction selection
    // ===================================================================

    /// Emit a store instruction from a register to memory.
    ///
    /// Selects the appropriate store opcode based on the type size:
    /// - I1/I8 → SB
    /// - I16 → SH
    /// - I32 → SW
    /// - I64/Ptr → SD
    /// - F32 → FSW
    /// - F64 → FSD
    pub fn emit_store(&mut self, src: u16, base: u16, offset: i64, ty: &IrType) {
        if !fits_i12(offset) {
            self.emit_immediate(T0, offset);
            self.emit_r(RvOpcode::ADD, T0, base, T0);
            self.emit_store(src, T0, 0, ty);
            return;
        }

        let opcode = match ty {
            IrType::I1 | IrType::I8 => RvOpcode::SB,
            IrType::I16 => RvOpcode::SH,
            IrType::I32 => RvOpcode::SW,
            IrType::I64 | IrType::I128 | IrType::Ptr => RvOpcode::SD,
            IrType::F32 => RvOpcode::FSW,
            IrType::F64 | IrType::F80 => RvOpcode::FSD,
            IrType::Void => return,
            IrType::Array(_, _) | IrType::Struct(_) | IrType::Function(_, _) => RvOpcode::SD,
        };

        let is_fp = matches!(ty, IrType::F32 | IrType::F64 | IrType::F80);
        let mut inst = RvInstruction::s_type(opcode, base, src, offset);
        inst.is_fp = is_fp;
        self.emit(inst);
    }

    // ===================================================================
    // Comparison instruction selection
    // ===================================================================

    /// Emit an integer comparison, materializing the boolean result into `rd`.
    ///
    /// For conditional branches, callers should use `emit_branch` instead,
    /// which generates direct branch instructions. This method produces a
    /// 0/1 integer result for use as a value.
    pub fn emit_comparison(&mut self, rd: u16, rs1: u16, rs2: u16, op: &ICmpOp) {
        match op {
            ICmpOp::Eq => {
                // rd = (rs1 == rs2) → SUB tmp, rs1, rs2; SEQZ rd, tmp
                self.emit_r(RvOpcode::SUB, rd, rs1, rs2);
                self.emit_i(RvOpcode::SLTIU, rd, rd, 1); // SEQZ
            }
            ICmpOp::Ne => {
                // rd = (rs1 != rs2) → SUB tmp, rs1, rs2; SNEZ rd, tmp
                self.emit_r(RvOpcode::SUB, rd, rs1, rs2);
                self.emit_r(RvOpcode::SLTU, rd, ZERO, rd); // SNEZ
            }
            ICmpOp::Slt => {
                self.emit_r(RvOpcode::SLT, rd, rs1, rs2);
            }
            ICmpOp::Sle => {
                // rd = (rs1 <= rs2) → rd = !(rs2 < rs1) → SLT tmp, rs2, rs1; XORI rd, tmp, 1
                self.emit_r(RvOpcode::SLT, rd, rs2, rs1);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            ICmpOp::Sgt => {
                // rd = (rs1 > rs2) → SLT rd, rs2, rs1
                self.emit_r(RvOpcode::SLT, rd, rs2, rs1);
            }
            ICmpOp::Sge => {
                // rd = (rs1 >= rs2) → !(rs1 < rs2) → SLT tmp, rs1, rs2; XORI rd, tmp, 1
                self.emit_r(RvOpcode::SLT, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            ICmpOp::Ult => {
                self.emit_r(RvOpcode::SLTU, rd, rs1, rs2);
            }
            ICmpOp::Ule => {
                self.emit_r(RvOpcode::SLTU, rd, rs2, rs1);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            ICmpOp::Ugt => {
                self.emit_r(RvOpcode::SLTU, rd, rs2, rs1);
            }
            ICmpOp::Uge => {
                self.emit_r(RvOpcode::SLTU, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
        }
    }

    /// Emit a floating-point comparison, putting 0/1 into integer `rd`.
    fn emit_fcmp(&mut self, rd: u16, rs1: u16, rs2: u16, op: &FCmpOp, is_double: bool) {
        match (op, is_double) {
            (FCmpOp::Oeq, false) => self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs2),
            (FCmpOp::Oeq, true) => self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs2),
            (FCmpOp::Olt, false) => self.emit_fp_r(RvOpcode::FLT_S, rd, rs1, rs2),
            (FCmpOp::Olt, true) => self.emit_fp_r(RvOpcode::FLT_D, rd, rs1, rs2),
            (FCmpOp::Ole, false) => self.emit_fp_r(RvOpcode::FLE_S, rd, rs1, rs2),
            (FCmpOp::Ole, true) => self.emit_fp_r(RvOpcode::FLE_D, rd, rs1, rs2),
            (FCmpOp::Ogt, false) => self.emit_fp_r(RvOpcode::FLT_S, rd, rs2, rs1),
            (FCmpOp::Ogt, true) => self.emit_fp_r(RvOpcode::FLT_D, rd, rs2, rs1),
            (FCmpOp::Oge, false) => self.emit_fp_r(RvOpcode::FLE_S, rd, rs2, rs1),
            (FCmpOp::Oge, true) => self.emit_fp_r(RvOpcode::FLE_D, rd, rs2, rs1),
            (FCmpOp::One, false) => {
                // !(a == b) — FEQ then invert
                self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::One, true) => {
                self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::Uno, false) => {
                // Unordered: !(a == a) || !(b == b)
                self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_S, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::Uno, true) => {
                self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_D, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::Ord, false) => {
                // Ordered: (a == a) && (b == b)
                self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_S, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
            }
            (FCmpOp::Ord, true) => {
                self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_D, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
            }
        }
    }

    // ===================================================================
    // Branch instruction selection
    // ===================================================================

    /// Emit a conditional or unconditional branch.
    ///
    /// Conditional branches use B-type instructions with ±4 KiB range.
    /// For far branches (detected post-layout), trampolines may be needed.
    pub fn emit_branch(&mut self, cond: Option<(u16, u16, &ICmpOp)>, target_label: &str) {
        match cond {
            None => {
                // Unconditional jump: J target (pseudo for JAL x0, offset)
                let mut inst = RvInstruction::no_op(RvOpcode::J);
                inst.symbol = Some(target_label.to_string());
                self.emit(inst);
            }
            Some((rs1, rs2, op)) => {
                let branch_op = match op {
                    ICmpOp::Eq => RvOpcode::BEQ,
                    ICmpOp::Ne => RvOpcode::BNE,
                    ICmpOp::Slt => RvOpcode::BLT,
                    ICmpOp::Sge => RvOpcode::BGE,
                    ICmpOp::Ult => RvOpcode::BLTU,
                    ICmpOp::Uge => RvOpcode::BGEU,
                    // For other predicates, invert or swap operands
                    ICmpOp::Sgt => {
                        self.emit_b(RvOpcode::BLT, rs2, rs1, target_label);
                        return;
                    }
                    ICmpOp::Sle => {
                        self.emit_b(RvOpcode::BGE, rs2, rs1, target_label);
                        return;
                    }
                    ICmpOp::Ugt => {
                        self.emit_b(RvOpcode::BLTU, rs2, rs1, target_label);
                        return;
                    }
                    ICmpOp::Ule => {
                        self.emit_b(RvOpcode::BGEU, rs2, rs1, target_label);
                        return;
                    }
                };
                self.emit_b(branch_op, rs1, rs2, target_label);
            }
        }
    }

    // ===================================================================
    // Type conversion instruction selection
    // ===================================================================

    /// Emit type conversion (cast) instructions.
    ///
    /// Handles integer width conversions, integer↔float conversions,
    /// and pointer↔integer conversions.
    pub fn emit_conversion(&mut self, rd: u16, rs: u16, from_ty: &IrType, to_ty: &IrType) {
        match (from_ty, to_ty) {
            // Integer truncation: just use the lower bits via ANDI/shift
            (_, IrType::I1) => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 1);
            }
            (_, IrType::I8) if !matches!(from_ty, IrType::I1) => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 0xFF);
            }
            (_, IrType::I16) if !matches!(from_ty, IrType::I1 | IrType::I8) => {
                // Zero-extend 16-bit: shift left 48, then logical shift right 48
                self.emit_i(RvOpcode::SLLI, rd, rs, 48);
                self.emit_i(RvOpcode::SRLI, rd, rd, 48);
            }
            (_, IrType::I32) if matches!(from_ty, IrType::I64 | IrType::Ptr | IrType::I128) => {
                // Truncate to 32 bits: ADDIW sign-extends 32→64
                self.emit_i(RvOpcode::ADDIW, rd, rs, 0);
            }

            // Zero extension
            (IrType::I1, _) if to_ty.is_integer() || to_ty.is_pointer() => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 1);
            }
            (IrType::I8, _) if to_ty.is_integer() || to_ty.is_pointer() => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 0xFF);
            }
            (IrType::I16, _) if to_ty.is_integer() || to_ty.is_pointer() => {
                self.emit_i(RvOpcode::SLLI, rd, rs, 48);
                self.emit_i(RvOpcode::SRLI, rd, rd, 48);
            }
            (IrType::I32, IrType::I64) | (IrType::I32, IrType::Ptr) => {
                // Zero-extend word: shift left 32, logical shift right 32
                self.emit_i(RvOpcode::SLLI, rd, rs, 32);
                self.emit_i(RvOpcode::SRLI, rd, rd, 32);
            }

            // Sign extension
            (IrType::I8, IrType::I32) => {
                self.emit_i(RvOpcode::SLLI, rd, rs, 56);
                self.emit_i(RvOpcode::SRAI, rd, rd, 56);
            }
            (IrType::I16, IrType::I32) => {
                self.emit_i(RvOpcode::SLLI, rd, rs, 48);
                self.emit_i(RvOpcode::SRAI, rd, rd, 48);
            }

            // Integer ↔ Float conversions
            (IrType::I32, IrType::F32) => {
                self.emit_fp_unary(RvOpcode::FCVT_S_W, rd, rs);
            }
            (IrType::I32, IrType::F64) => {
                self.emit_fp_unary(RvOpcode::FCVT_D_W, rd, rs);
            }
            (IrType::I64, IrType::F32) => {
                self.emit_fp_unary(RvOpcode::FCVT_S_L, rd, rs);
            }
            (IrType::I64, IrType::F64) => {
                self.emit_fp_unary(RvOpcode::FCVT_D_L, rd, rs);
            }
            (IrType::F32, IrType::I32) => {
                self.emit_fp_unary(RvOpcode::FCVT_W_S, rd, rs);
            }
            (IrType::F32, IrType::I64) => {
                self.emit_fp_unary(RvOpcode::FCVT_L_S, rd, rs);
            }
            (IrType::F64, IrType::I32) => {
                self.emit_fp_unary(RvOpcode::FCVT_W_D, rd, rs);
            }
            (IrType::F64, IrType::I64) => {
                self.emit_fp_unary(RvOpcode::FCVT_L_D, rd, rs);
            }
            // Float ↔ Float conversions
            (IrType::F32, IrType::F64) => {
                self.emit_fp_unary(RvOpcode::FCVT_D_S, rd, rs);
            }
            (IrType::F64, IrType::F32) => {
                self.emit_fp_unary(RvOpcode::FCVT_S_D, rd, rs);
            }
            // Pointer ↔ Integer: no-op on RV64 (both 64 bits)
            (IrType::Ptr, _) if to_ty.is_integer() => {
                if rd != rs {
                    self.emit_i(RvOpcode::ADDI, rd, rs, 0); // MV
                }
            }
            (_, IrType::Ptr) if from_ty.is_integer() => {
                if rd != rs {
                    self.emit_i(RvOpcode::ADDI, rd, rs, 0); // MV
                }
            }
            // Default: MV (copy register)
            _ => {
                if rd != rs {
                    self.emit_i(RvOpcode::ADDI, rd, rs, 0);
                }
            }
        }
    }

    // ===================================================================
    // Function call lowering
    // ===================================================================

    /// Emit a function call instruction sequence.
    ///
    /// Follows the LP64D ABI:
    /// Attempt to emit specialised code for compiler builtins (RISC-V 64).
    /// Returns `true` if the builtin was handled, `false` otherwise.
    fn try_emit_rv_builtin(
        &mut self,
        fname: &str,
        result: &Value,
        args: &[Value],
        return_type: &IrType,
    ) -> bool {
        match fname {
            // ── byte-swap builtins ────────────────────────────────────
            // RISC-V Zbb extension has REV8, but we cannot assume Zbb.
            // Implement bswap with shift-and-mask sequences.
            "__builtin_bswap64" => {
                let rd = self.dest_reg(*result, return_type);
                let src = self.src_reg(args[0]);
                self.emit_bswap64(rd, src);
                true
            }
            "__builtin_bswap32" => {
                let rd = self.dest_reg(*result, return_type);
                let src = self.src_reg(args[0]);
                self.emit_bswap32(rd, src);
                true
            }
            "__builtin_bswap16" => {
                let rd = self.dest_reg(*result, return_type);
                let src = self.src_reg(args[0]);
                self.emit_bswap16(rd, src);
                true
            }
            // ── branch hint builtins (passthrough) ───────────────────
            "__builtin_expect" | "__builtin_expect_with_probability" => {
                let rd = self.dest_reg(*result, return_type);
                let src = self.src_reg(args[0]);
                if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0); // MV
                }
                true
            }
            "__builtin_assume_aligned" => {
                let rd = self.dest_reg(*result, return_type);
                let src = self.src_reg(args[0]);
                if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0); // MV
                }
                true
            }
            // CLZ / CTZ / popcount / ffs — these require loop labels which
            // are complex in the RISC-V instruction selection phase.
            // Fall through to a real function call for now.
            "__builtin_clz"
            | "__builtin_clzl"
            | "__builtin_clzll"
            | "__builtin_ctz"
            | "__builtin_ctzl"
            | "__builtin_ctzll"
            | "__builtin_popcount"
            | "__builtin_popcountl"
            | "__builtin_popcountll"
            | "__builtin_ffs"
            | "__builtin_ffsl"
            | "__builtin_ffsll" => {
                false // Let these fall through to emit_call
            }
            // ── trap ─────────────────────────────────────────────────
            "__builtin_trap" | "__builtin_unreachable" => {
                self.emit(RvInstruction::no_op(RvOpcode::EBREAK));
                true
            }
            _ => false, // Not a recognised builtin.
        }
    }

    /// Emit byte-swap for 64-bit value: reverse all 8 bytes.
    /// Uses t0 (register 5) as temporary.
    fn emit_bswap64(&mut self, rd: u16, src: u16) {
        // Strategy: extract each byte, shift to its swapped position, OR together.
        // We use rd as accumulator and T0 (5) as scratch.
        let t0: u16 = 5; // t0 = x5
                         // Byte 0 (bits 0-7) → bits 56-63
        self.emit_i(RvOpcode::ANDI, rd, src, 0xFF);
        self.emit_i(RvOpcode::SLLI, rd, rd, 56);
        // Byte 1 (bits 8-15) → bits 48-55
        self.emit_i(RvOpcode::SRLI, t0, src, 8);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 48);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        // Byte 2 (bits 16-23) → bits 40-47
        self.emit_i(RvOpcode::SRLI, t0, src, 16);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 40);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        // Byte 3 (bits 24-31) → bits 32-39
        self.emit_i(RvOpcode::SRLI, t0, src, 24);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 32);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        // Byte 4 (bits 32-39) → bits 24-31
        self.emit_i(RvOpcode::SRLI, t0, src, 32);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 24);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        // Byte 5 (bits 40-47) → bits 16-23
        self.emit_i(RvOpcode::SRLI, t0, src, 40);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 16);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        // Byte 6 (bits 48-55) → bits 8-15
        self.emit_i(RvOpcode::SRLI, t0, src, 48);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 8);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        // Byte 7 (bits 56-63) → bits 0-7
        self.emit_i(RvOpcode::SRLI, t0, src, 56);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
    }

    /// Emit byte-swap for 32-bit value: reverse bottom 4 bytes.
    fn emit_bswap32(&mut self, rd: u16, src: u16) {
        let t0: u16 = 5; // t0 = x5
        self.emit_i(RvOpcode::ANDI, rd, src, 0xFF);
        self.emit_i(RvOpcode::SLLI, rd, rd, 24);
        self.emit_i(RvOpcode::SRLI, t0, src, 8);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 16);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        self.emit_i(RvOpcode::SRLI, t0, src, 16);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_i(RvOpcode::SLLI, t0, t0, 8);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
        self.emit_i(RvOpcode::SRLI, t0, src, 24);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
    }

    /// Emit byte-swap for 16-bit value: swap the two bottom bytes.
    fn emit_bswap16(&mut self, rd: u16, src: u16) {
        let t0: u16 = 5; // t0 = x5
                         // low byte → bits 8-15
        self.emit_i(RvOpcode::ANDI, rd, src, 0xFF);
        self.emit_i(RvOpcode::SLLI, rd, rd, 8);
        // high byte → bits 0-7
        self.emit_i(RvOpcode::SRLI, t0, src, 8);
        self.emit_i(RvOpcode::ANDI, t0, t0, 0xFF);
        self.emit_r(RvOpcode::OR, rd, rd, t0);
    }

    /// Emit a sequence of GPR register-to-register moves that correctly
    /// handles the parallel-copy problem (no source clobbered before read).
    ///
    /// The RISC-V calling convention places:
    /// - Integer args in a0–a7, FP args in fa0–fa7
    /// - Stack-passed args aligned to 8 bytes
    /// - Return value in a0 (integer) or fa0 (FP)
    /// - Caller saves: t0–t6, a0–a7, ft0–ft11, fa0–fa7
    ///
    /// Uses topological ordering with T0 (x5) as the scratch register
    /// to break cycles.
    fn emit_parallel_gpr_moves(instructions: &mut Vec<RvInstruction>, moves: &[(u16, u16)]) {
        if moves.is_empty() {
            return;
        }
        // Check if there are any conflicts: a dst that is also a src
        // of a different pending move.
        let dsts: std::collections::HashSet<u16> = moves.iter().map(|m| m.0).collect();
        let has_conflict = moves.iter().any(|m| dsts.contains(&m.1) && m.0 != m.1);
        if !has_conflict {
            // No conflicts — emit in order.
            for &(dst, src) in moves {
                instructions
                    .push(RvInstruction::i_type(RvOpcode::ADDI, dst, src, 0).as_call_arg_setup());
            }
            return;
        }
        // Parallel-copy resolution with topological ordering.
        let mut pending: Vec<(u16, u16)> = moves.to_vec();
        let mut emitted = true;
        while emitted && !pending.is_empty() {
            emitted = false;
            let srcs: std::collections::HashSet<u16> = pending.iter().map(|m| m.1).collect();
            let mut next_pending = Vec::new();
            for &(dst, src) in &pending {
                if !srcs.contains(&dst) || dst == src {
                    // Safe to emit: no other pending move reads from dst.
                    if dst != src {
                        instructions.push(
                            RvInstruction::i_type(RvOpcode::ADDI, dst, src, 0).as_call_arg_setup(),
                        );
                    }
                    emitted = true;
                } else {
                    next_pending.push((dst, src));
                }
            }
            pending = next_pending;
        }
        // If there's a cycle remaining, break it using T0 as scratch.
        while !pending.is_empty() {
            let (_first_dst, first_src) = pending[0];
            // Save first_src to T0
            instructions
                .push(RvInstruction::i_type(RvOpcode::ADDI, T0, first_src, 0).as_call_arg_setup());
            // Replace first_src in all pending moves with T0
            for m in pending.iter_mut() {
                if m.1 == first_src {
                    m.1 = T0;
                }
            }
            // Now first move can be emitted: dst ← T0
            pending[0].1 = T0;
            // Re-run topological resolution
            let mut progress = true;
            while progress && !pending.is_empty() {
                progress = false;
                let srcs: std::collections::HashSet<u16> = pending.iter().map(|m| m.1).collect();
                let mut next = Vec::new();
                for &(dst, src) in &pending {
                    if !srcs.contains(&dst) || dst == src {
                        if dst != src {
                            instructions.push(
                                RvInstruction::i_type(RvOpcode::ADDI, dst, src, 0)
                                    .as_call_arg_setup(),
                            );
                        }
                        progress = true;
                    } else {
                        next.push((dst, src));
                    }
                }
                pending = next;
            }
            // Any remaining pending moves that couldn't be resolved
            // are left in `pending` — this handles residual cycles
            // that the single-scratch-register approach didn't break.
            let _ = &pending;
        }
    }

    /// Emit a sequence of FPR register-to-register moves with parallel-copy
    /// resolution. Uses FT0 as the scratch register.
    fn emit_parallel_fpr_moves(instructions: &mut Vec<RvInstruction>, moves: &[(u16, u16)]) {
        if moves.is_empty() {
            return;
        }
        let dsts: std::collections::HashSet<u16> = moves.iter().map(|m| m.0).collect();
        let has_conflict = moves.iter().any(|m| dsts.contains(&m.1) && m.0 != m.1);
        if !has_conflict {
            for &(dst, src) in moves {
                let mut inst =
                    RvInstruction::r_type(RvOpcode::FSGNJ_D, dst, src, src).as_call_arg_setup();
                inst.is_fp = true;
                instructions.push(inst);
            }
            return;
        }
        // For FP cycles, use FT0 as scratch
        let ft0: u16 = FT0;
        let mut pending: Vec<(u16, u16)> = moves.to_vec();
        let mut emitted = true;
        while emitted && !pending.is_empty() {
            emitted = false;
            let srcs: std::collections::HashSet<u16> = pending.iter().map(|m| m.1).collect();
            let mut next_pending = Vec::new();
            for &(dst, src) in &pending {
                if !srcs.contains(&dst) || dst == src {
                    if dst != src {
                        let mut inst = RvInstruction::r_type(RvOpcode::FSGNJ_D, dst, src, src)
                            .as_call_arg_setup();
                        inst.is_fp = true;
                        instructions.push(inst);
                    }
                    emitted = true;
                } else {
                    next_pending.push((dst, src));
                }
            }
            pending = next_pending;
        }
        while !pending.is_empty() {
            let (_, first_src) = pending[0];
            let mut inst = RvInstruction::r_type(RvOpcode::FSGNJ_D, ft0, first_src, first_src)
                .as_call_arg_setup();
            inst.is_fp = true;
            instructions.push(inst);
            for m in pending.iter_mut() {
                if m.1 == first_src {
                    m.1 = ft0;
                }
            }
            let mut progress = true;
            while progress && !pending.is_empty() {
                progress = false;
                let srcs: std::collections::HashSet<u16> = pending.iter().map(|m| m.1).collect();
                let mut next = Vec::new();
                for &(dst, src) in &pending {
                    if !srcs.contains(&dst) || dst == src {
                        if dst != src {
                            let mut inst2 = RvInstruction::r_type(RvOpcode::FSGNJ_D, dst, src, src)
                                .as_call_arg_setup();
                            inst2.is_fp = true;
                            instructions.push(inst2);
                        }
                        progress = true;
                    } else {
                        next.push((dst, src));
                    }
                }
                pending = next;
            }
        }
    }

    pub fn emit_call(
        &mut self,
        callee: &str,
        args: &[(u16, bool)], // (register, is_fp)
        result_reg: Option<u16>,
        result_is_fp: bool,
        is_variadic: bool,
    ) {
        self.has_calls = true;

        // ── Parallel-copy resolution for argument register moves ─────
        //
        // Build the list of (dst_abi_reg, src_reg) moves for both integer
        // and FP banks, then resolve the parallel-copy problem to avoid
        // clobbering source registers that are read by later moves.
        //
        // Example conflict:
        //   printf("%d %d", a, b)  where a is in A2, b is in A0
        //   Naive order:  MV A0, A2 (format); MV A1, A2 (a); MV A2, A0 (b)
        //   A0 clobbered before A2 reads it → wrong value for b.
        //
        // The resolution algorithm:
        //   1. Collect (dst, src) register-to-register moves.
        //   2. Emit moves with no dependency first (topological order).
        //   3. Break cycles using T0 as scratch for GPR, FT0 for FPR.

        let mut int_idx: usize = 0;
        let mut fp_idx: usize = 0;
        let mut stack_offset: i64 = 0;

        // GPR moves: (dst_abi_reg, src_reg)
        let mut gpr_moves: Vec<(u16, u16)> = Vec::new();
        // FPR moves: (dst_abi_reg, src_reg)
        let mut fpr_moves: Vec<(u16, u16)> = Vec::new();
        // FP→GPR cross-class moves for variadic calls.
        // These are emitted as FMV.X.D / FMV.X.W instructions AFTER
        // the normal parallel-copy resolution (since they cannot use
        // the standard MV pseudo which only works within a register class).
        // Entries: (dst_gpr_abi_reg, src_fpr_vreg)
        let mut fp_to_gpr_moves: Vec<(u16, u16)> = Vec::new();

        for &(reg, is_fp) in args.iter() {
            if is_fp && is_variadic {
                // ── RISC-V LP64D ABI: variadic FP → integer register ──
                //
                // Variadic arguments after the `...` must ALL be passed
                // in integer registers (a0–a7) as raw bit patterns.
                // We use the integer arg slot and record a cross-class
                // move that will be emitted as FMV.X.D / FMV.X.W below.
                if int_idx < 8 {
                    let ai = A0 + int_idx as u16;
                    int_idx += 1;
                    fp_to_gpr_moves.push((ai, reg));
                } else {
                    // Spill to stack as raw bits (still FP-sized).
                    self.emit_store(reg, SP, stack_offset, &IrType::F64);
                    stack_offset += 8;
                }
            } else if is_fp {
                if fp_idx < 8 {
                    let fa = FA0 + fp_idx as u16;
                    fp_idx += 1;
                    if reg != fa {
                        fpr_moves.push((fa, reg));
                    }
                } else {
                    self.emit_store(reg, SP, stack_offset, &IrType::F64);
                    stack_offset += 8;
                }
            } else if int_idx < 8 {
                let ai = A0 + int_idx as u16;
                int_idx += 1;
                if reg != ai {
                    gpr_moves.push((ai, reg));
                }
            } else {
                self.emit_s(RvOpcode::SD, SP, reg, stack_offset);
                stack_offset += 8;
            }
        }

        // Resolve GPR parallel copies
        Self::emit_parallel_gpr_moves(&mut self.instructions, &gpr_moves);

        // Resolve FPR parallel copies
        Self::emit_parallel_fpr_moves(&mut self.instructions, &fpr_moves);

        // Emit cross-class FP→GPR moves for variadic arguments.
        // These use FMV.X.D (double) with physical destination registers
        // and virtual/physical source FPR registers.
        for &(dst_gpr, src_fpr) in &fp_to_gpr_moves {
            let mut inst = RvInstruction::i_type(RvOpcode::FMV_X_D, dst_gpr, src_fpr, 0);
            inst.is_fp = true;
            inst.is_call_arg_setup = true;
            self.instructions.push(inst);
        }

        // Generate the call instruction
        if self.pic_mode {
            // PIC: AUIPC+JALR through PLT
            self.emit(RvInstruction {
                opcode: RvOpcode::CALL,
                rd: Some(RA),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("{}@plt", callee)),
                is_fp: false,
                comment: Some(format!("call {}", callee)),
                is_call_arg_setup: false,
            });
        } else {
            // Direct call
            self.emit(RvInstruction {
                opcode: RvOpcode::CALL,
                rd: Some(RA),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(callee.to_string()),
                is_fp: false,
                comment: Some(format!("call {}", callee)),
                is_call_arg_setup: false,
            });
        }

        // Move result from ABI return register to destination
        if let Some(dst) = result_reg {
            if result_is_fp {
                if dst != FA0 {
                    self.emit_fp_r(RvOpcode::FSGNJ_D, dst, FA0, FA0);
                }
            } else if dst != A0 {
                self.emit_i(RvOpcode::ADDI, dst, A0, 0); // MV
            }
        }
    }

    // ===================================================================
    // Inline assembly handling
    // ===================================================================

    /// Emit inline assembly as a passthrough instruction.
    ///
    /// The template string and constraints are preserved for the assembler
    /// to process. This is critical for Linux kernel builds which use
    /// extensive inline assembly.
    /// Emit an already-substituted inline assembly template.
    pub fn emit_inline_asm_substituted(
        &mut self,
        substituted_template: &str,
        constraints: &str,
        clobbers: &[String],
    ) {
        let inst = RvInstruction {
            opcode: RvOpcode::INLINE_ASM,
            rd: None,
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: Some(substituted_template.to_string()),
            is_fp: false,
            comment: Some(format!(
                "asm: constraints={}, clobbers=[{}]",
                constraints,
                clobbers.join(","),
            )),
            is_call_arg_setup: false,
        };
        self.emit(inst);
    }

    /// Get RISC-V ABI register name for a physical register index.
    fn rv_reg_name(&self, reg: u16) -> String {
        match reg {
            0 => "zero".to_string(),
            1 => "ra".to_string(),
            2 => "sp".to_string(),
            3 => "gp".to_string(),
            4 => "tp".to_string(),
            5 => "t0".to_string(),
            6 => "t1".to_string(),
            7 => "t2".to_string(),
            8 => "s0".to_string(),
            9 => "s1".to_string(),
            10 => "a0".to_string(),
            11 => "a1".to_string(),
            12 => "a2".to_string(),
            13 => "a3".to_string(),
            14 => "a4".to_string(),
            15 => "a5".to_string(),
            16 => "a6".to_string(),
            17 => "a7".to_string(),
            18 => "s2".to_string(),
            19 => "s3".to_string(),
            20 => "s4".to_string(),
            21 => "s5".to_string(),
            22 => "s6".to_string(),
            23 => "s7".to_string(),
            24 => "s8".to_string(),
            25 => "s9".to_string(),
            26 => "s10".to_string(),
            27 => "s11".to_string(),
            28 => "t3".to_string(),
            29 => "t4".to_string(),
            30 => "t5".to_string(),
            31 => "t6".to_string(),
            _ => format!("r{}", reg),
        }
    }

    /// Substitute %0, %1, ... (and %l0, %l1, ...) in an inline assembly template.
    fn substitute_template(
        template: &str,
        operand_reprs: &[String],
        goto_labels: &[String],
    ) -> String {
        let mut result = String::with_capacity(template.len());
        let bytes = template.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            if bytes[i] == b'%' && i + 1 < len {
                if bytes[i + 1] == b'%' {
                    result.push('%');
                    i += 2;
                } else if bytes[i + 1] == b'l' && i + 2 < len && bytes[i + 2].is_ascii_digit() {
                    // Goto label reference: %l0, %l1, ...
                    let start = i + 2;
                    let mut end = start;
                    while end < len && bytes[end].is_ascii_digit() {
                        end += 1;
                    }
                    let idx: usize = template[start..end].parse().unwrap_or(0);
                    if idx < goto_labels.len() {
                        result.push_str(&goto_labels[idx]);
                    } else {
                        result.push_str(&template[i..end]);
                    }
                    i = end;
                } else if bytes[i + 1].is_ascii_digit() {
                    let start = i + 1;
                    let mut end = start;
                    while end < len && bytes[end].is_ascii_digit() {
                        end += 1;
                    }
                    let idx: usize = template[start..end].parse().unwrap_or(0);
                    if idx < operand_reprs.len() {
                        result.push_str(&operand_reprs[idx]);
                    } else {
                        result.push_str(&template[i..end]);
                    }
                    i = end;
                } else {
                    result.push('%');
                    i += 1;
                }
            } else {
                result.push(bytes[i] as char);
                i += 1;
            }
        }
        result
    }

    // ===================================================================
    // Switch statement lowering
    // ===================================================================

    /// Emit a switch statement.
    ///
    /// For small case counts (≤8): cascaded BEQ comparisons.
    /// For dense ranges: jump table with bounds check.
    /// For sparse cases: binary search tree.
    pub fn emit_switch(&mut self, val_reg: u16, default_label: &str, cases: &[(i64, String)]) {
        if cases.is_empty() {
            // Just jump to default
            self.emit_branch(None, default_label);
            return;
        }

        // Threshold: use cascaded comparisons for small case counts
        if cases.len() <= 8 {
            self.emit_switch_cascade(val_reg, default_label, cases);
            return;
        }

        // Check for dense range: if (max - min + 1) <= 2 * len, use jump table
        let min_val = cases.iter().map(|(v, _)| *v).min().unwrap_or(0);
        let max_val = cases.iter().map(|(v, _)| *v).max().unwrap_or(0);
        let range = (max_val - min_val + 1) as usize;

        if range <= cases.len() * 2 && range <= 1024 {
            self.emit_switch_jump_table(val_reg, default_label, cases, min_val, range);
        } else {
            // Sparse: use cascaded comparisons (can be extended to binary search)
            self.emit_switch_cascade(val_reg, default_label, cases);
        }
    }

    /// Emit cascaded BEQ comparisons for a switch statement.
    fn emit_switch_cascade(&mut self, val_reg: u16, default_label: &str, cases: &[(i64, String)]) {
        for (case_val, case_label) in cases {
            if fits_i12(*case_val) {
                // Compare: ADDI T0, x0, case_val; BEQ val_reg, T0, label
                self.emit_i(RvOpcode::ADDI, T0, ZERO, *case_val);
            } else {
                self.emit_immediate(T0, *case_val);
            }
            self.emit_b(RvOpcode::BEQ, val_reg, T0, case_label);
        }
        // Fall through to default
        self.emit_branch(None, default_label);
    }

    /// Emit a jump table for a dense switch statement.
    fn emit_switch_jump_table(
        &mut self,
        val_reg: u16,
        default_label: &str,
        cases: &[(i64, String)],
        min_val: i64,
        range: usize,
    ) {
        // Subtract min_val to get zero-based index
        if min_val != 0 {
            if fits_i12(-min_val) {
                self.emit_i(RvOpcode::ADDI, T0, val_reg, -min_val);
            } else {
                self.emit_immediate(T0, -min_val);
                self.emit_r(RvOpcode::ADD, T0, val_reg, T0);
            }
        } else {
            self.emit_i(RvOpcode::ADDI, T0, val_reg, 0); // MV
        }

        // Bounds check: if index >= range, jump to default
        self.emit_immediate(T1, range as i64);
        self.emit_b(RvOpcode::BGEU, T0, T1, default_label);

        // Scale index by 8 (pointer size) for table entry
        self.emit_i(RvOpcode::SLLI, T0, T0, 3);

        // Load jump table base address
        let table_label = format!(".L_switch_table_{}", self.instructions.len());
        if self.pic_mode {
            self.emit_pcrel_address(T1, &table_label);
        } else {
            self.emit_address(T1, &table_label);
        }

        // Load target address from table: LD T0, T1(T0)
        self.emit_r(RvOpcode::ADD, T0, T1, T0);
        self.emit_i(RvOpcode::LD, T0, T0, 0);

        // Jump to target: JALR x0, T0, 0
        self.emit_i(RvOpcode::JALR, ZERO, T0, 0);

        // Build the case map for table generation (done by assembler/linker)
        // Emit the table entries as data pseudo-instructions
        let mut table: Vec<String> = vec![default_label.to_string(); range];
        for (case_val, case_label) in cases {
            let idx = (*case_val - min_val) as usize;
            if idx < range {
                table[idx] = case_label.clone();
            }
        }

        // Emit table references as symbol-bearing pseudo-instructions
        for entry in &table {
            self.emit(RvInstruction {
                opcode: RvOpcode::LA,
                rd: None,
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(entry.clone()),
                is_fp: false,
                comment: Some(format!("switch table entry: {}", entry)),
                is_call_arg_setup: false,
            });
        }
    }

    // ===================================================================
    // Prologue / Epilogue generation
    // ===================================================================

    /// Emit function prologue.
    ///
    /// Allocates the stack frame, saves the return address and frame pointer,
    /// and saves all callee-saved registers used in the function body.
    ///
    /// Stack layout (growing downward):
    /// ```text
    /// [old SP] ← caller's stack pointer
    ///   RA save slot
    ///   FP (s0) save slot
    ///   s1 save slot (if used)
    ///   s2 save slot (if used)
    ///   ...
    ///   fs0 save slot (if used)
    ///   ...
    ///   local variables / spill area
    /// [new SP] ← current stack pointer
    /// ```
    pub fn emit_prologue(&mut self) {
        let frame = self.frame_size;
        if frame == 0 {
            return;
        }

        // Allocate stack frame: addi sp, sp, -frame_size
        if fits_i12(-frame) {
            self.emit_i(RvOpcode::ADDI, SP, SP, -frame);
        } else {
            // Large frame: materialize offset in T0 then sub
            self.emit_immediate(T0, frame);
            self.emit_r(RvOpcode::SUB, SP, SP, T0);
        }

        // Save return address
        let ra_offset = frame - 8;
        self.emit_s(RvOpcode::SD, SP, RA, ra_offset);

        // Save frame pointer
        let fp_offset = frame - 16;
        self.emit_s(RvOpcode::SD, SP, FP, fp_offset);

        // Set up frame pointer: addi s0, sp, frame_size
        if fits_i12(frame) {
            self.emit_i(RvOpcode::ADDI, FP, SP, frame);
        } else {
            self.emit_immediate(T0, frame);
            self.emit_r(RvOpcode::ADD, FP, SP, T0);
        }

        // Save callee-saved GPRs
        let mut offset = frame - 24; // after RA and FP
        for &reg in &self.used_callee_saved_gprs.clone() {
            if reg == FP {
                continue; // FP already saved above
            }
            if fits_i12(offset) {
                self.emit_s(RvOpcode::SD, SP, reg, offset);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                self.emit_s(RvOpcode::SD, T0, reg, 0);
            }
            offset -= 8;
        }

        // Save callee-saved FPRs
        for &reg in &self.used_callee_saved_fprs.clone() {
            if fits_i12(offset) {
                let mut inst = RvInstruction::s_type(RvOpcode::FSD, SP, reg, offset);
                inst.is_fp = true;
                self.emit(inst);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                let mut inst = RvInstruction::s_type(RvOpcode::FSD, T0, reg, 0);
                inst.is_fp = true;
                self.emit(inst);
            }
            offset -= 8;
        }
    }

    /// Emit function epilogue.
    ///
    /// Restores callee-saved registers, deallocates the stack frame,
    /// and returns to the caller.
    pub fn emit_epilogue(&mut self) {
        let frame = self.frame_size;
        if frame == 0 {
            self.emit(RvInstruction::no_op(RvOpcode::RET));
            return;
        }

        // Restore callee-saved FPRs (reverse order of prologue)
        // Count includes FP which we skip
        let gprs_to_restore: Vec<u16> = self
            .used_callee_saved_gprs
            .iter()
            .copied()
            .filter(|&r| r != FP)
            .collect();
        let fprs_to_restore: Vec<u16> = self.used_callee_saved_fprs.clone();

        // Restore FPRs
        let mut offset = frame - 24 - (gprs_to_restore.len() as i64 * 8);
        for &reg in fprs_to_restore.iter().rev() {
            offset -= 8;
            if fits_i12(offset) {
                let mut inst = RvInstruction::i_type(RvOpcode::FLD, reg, SP, offset);
                inst.is_fp = true;
                self.emit(inst);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                let mut inst = RvInstruction::i_type(RvOpcode::FLD, reg, T0, 0);
                inst.is_fp = true;
                self.emit(inst);
            }
        }

        // Restore callee-saved GPRs (reverse order)
        offset = frame - 24;
        for &reg in gprs_to_restore.iter().rev() {
            if fits_i12(offset) {
                self.emit_i(RvOpcode::LD, reg, SP, offset);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                self.emit_i(RvOpcode::LD, reg, T0, 0);
            }
            offset -= 8;
        }

        // Restore frame pointer
        let fp_offset = frame - 16;
        self.emit_i(RvOpcode::LD, FP, SP, fp_offset);

        // Restore return address
        let ra_offset = frame - 8;
        self.emit_i(RvOpcode::LD, RA, SP, ra_offset);

        // Deallocate stack frame
        if fits_i12(frame) {
            self.emit_i(RvOpcode::ADDI, SP, SP, frame);
        } else {
            self.emit_immediate(T0, frame);
            self.emit_r(RvOpcode::ADD, SP, SP, T0);
        }

        // Return
        self.emit(RvInstruction::no_op(RvOpcode::RET));
    }

    // ===================================================================
    // Arithmetic instruction selection helpers
    // ===================================================================

    /// Select arithmetic instructions for an IR BinOp.
    fn select_binop(&mut self, rd: u16, lhs_reg: u16, rhs_reg: u16, op: &BinOp, ty: &IrType) {
        // Decide whether to use W-variants for 32-bit int ops
        let use_word = matches!(ty, IrType::I32);

        match op {
            BinOp::Add => {
                if use_word {
                    self.emit_r(RvOpcode::ADDW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::ADD, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::Sub => {
                if use_word {
                    self.emit_r(RvOpcode::SUBW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SUB, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::Mul => {
                if use_word {
                    self.emit_r(RvOpcode::MULW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::MUL, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::SDiv => {
                if use_word {
                    self.emit_r(RvOpcode::DIVW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::DIV, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::UDiv => {
                if use_word {
                    self.emit_r(RvOpcode::DIVUW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::DIVU, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::SRem => {
                if use_word {
                    self.emit_r(RvOpcode::REMW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::REM, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::URem => {
                if use_word {
                    self.emit_r(RvOpcode::REMUW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::REMU, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::And => {
                self.emit_r(RvOpcode::AND, rd, lhs_reg, rhs_reg);
            }
            BinOp::Or => {
                self.emit_r(RvOpcode::OR, rd, lhs_reg, rhs_reg);
            }
            BinOp::Xor => {
                self.emit_r(RvOpcode::XOR, rd, lhs_reg, rhs_reg);
            }
            BinOp::Shl => {
                if use_word {
                    self.emit_r(RvOpcode::SLLW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SLL, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::AShr => {
                if use_word {
                    self.emit_r(RvOpcode::SRAW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SRA, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::LShr => {
                if use_word {
                    self.emit_r(RvOpcode::SRLW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SRL, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FAdd => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FADD_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FADD_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FSub => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FSUB_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FSUB_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FMul => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FMUL_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FMUL_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FDiv => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FDIV_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FDIV_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FRem => {
                // RISC-V has no FREM instruction; implement via call to fmod/fmodf.
                // This requires the target binary to be linked against libm (-lm).
                // The linker must add -lm to resolve fmod/fmodf symbols.
                // Store operands in fa0/fa1, call fmod, result in fa0
                if matches!(ty, IrType::F32) {
                    if lhs_reg != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_S, FA0, lhs_reg, lhs_reg);
                    }
                    if rhs_reg != FA0 + 1 {
                        self.emit_fp_r(RvOpcode::FSGNJ_S, FA0 + 1, rhs_reg, rhs_reg);
                    }
                    self.emit(RvInstruction {
                        opcode: RvOpcode::CALL,
                        rd: Some(RA),
                        rs1: None,
                        rs2: None,
                        rs3: None,
                        imm: 0,
                        symbol: Some("fmodf".to_string()),
                        is_fp: false,
                        comment: Some("FRem via fmodf".to_string()),
                        is_call_arg_setup: false,
                    });
                    if rd != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_S, rd, FA0, FA0);
                    }
                } else {
                    if lhs_reg != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_D, FA0, lhs_reg, lhs_reg);
                    }
                    if rhs_reg != FA0 + 1 {
                        self.emit_fp_r(RvOpcode::FSGNJ_D, FA0 + 1, rhs_reg, rhs_reg);
                    }
                    self.emit(RvInstruction {
                        opcode: RvOpcode::CALL,
                        rd: Some(RA),
                        rs1: None,
                        rs2: None,
                        rs3: None,
                        imm: 0,
                        symbol: Some("fmod".to_string()),
                        is_fp: false,
                        comment: Some("FRem via fmod".to_string()),
                        is_call_arg_setup: false,
                    });
                    if rd != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_D, rd, FA0, FA0);
                    }
                }
                self.has_calls = true;
            }
        }
    }

    // ===================================================================
    // GEP (GetElementPtr) lowering
    // ===================================================================

    /// Lower a GetElementPtr instruction.
    ///
    /// The IR lowering phase pre-computes byte offsets for all GEP indices
    /// (`index * sizeof(element)`), so the backend treats indices as raw
    /// byte displacements without additional scaling (elem_size = 1).
    fn select_gep(&mut self, result: Value, base: Value, indices: &[Value], _result_type: &IrType) {
        let rd = self.vmap.alloc_gpr(result);
        let base_reg = self.src_reg(base);

        // Start with base address
        if rd != base_reg {
            self.emit_i(RvOpcode::ADDI, rd, base_reg, 0); // MV
        }

        // Indices are already byte offsets — just ADD them to the base.
        for idx in indices {
            let idx_reg = self.src_reg(*idx);
            self.emit_r(RvOpcode::ADD, rd, rd, idx_reg);
        }
    }

    // ===================================================================
    // Main instruction selection dispatcher
    // ===================================================================

    /// Select RISC-V 64 instructions for a single IR instruction.
    ///
    /// Dispatches on the IR instruction variant and generates the
    /// corresponding RV64 machine instruction sequence.
    pub fn select_instruction(&mut self, ir_inst: &Instruction) -> Vec<RvInstruction> {
        let start_idx = self.instructions.len();

        match ir_inst {
            Instruction::Alloca {
                result,
                ty,
                alignment,
                span: _,
            } => {
                // Allocate stack space for the variable
                let size = ty.size_bytes(&self.target) as i64;
                let align = match alignment {
                    Some(a) if *a > 0 => *a as i64,
                    _ => ty.align_bytes(&self.target) as i64,
                };
                // Align the offset
                self.alloca_offset = (self.alloca_offset + align - 1) & !(align - 1);
                self.alloca_offset += size;
                let offset = -self.alloca_offset;

                // Store the stack offset for this alloca
                self.vmap.set_stack_offset(*result, offset);

                // Also give it a GPR that holds the address
                let rd = self.vmap.alloc_gpr(*result);
                // Compute address: rd = FP + offset
                if fits_i12(offset) {
                    self.emit_i(RvOpcode::ADDI, rd, FP, offset);
                } else {
                    self.emit_immediate(T0, offset);
                    self.emit_r(RvOpcode::ADD, rd, FP, T0);
                }
                // Alloca produces a pointer.
                self.value_types.insert(result.index(), IrType::Ptr);
            }

            Instruction::Load {
                result,
                ptr,
                ty,
                volatile: _,
                span: _,
            } => {
                let ptr_reg = self.src_reg(*ptr);
                let rd = self.dest_reg(*result, ty);
                self.emit_load(rd, ptr_reg, 0, ty);
                self.value_types.insert(result.index(), ty.clone());
            }

            Instruction::Store {
                value,
                ptr,
                volatile: _,
                span: _,
            } => {
                let val_reg = self.src_reg(*value);
                let ptr_reg = self.src_reg(*ptr);
                // Look up the type of the stored value.  Fall back to I64
                // (8-byte store) when the producing instruction wasn't tracked.
                let ty = self
                    .value_types
                    .get(&value.index())
                    .cloned()
                    .unwrap_or(IrType::I64);
                self.emit_store(val_reg, ptr_reg, 0, &ty);
            }

            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                span: _,
            } => {
                // Handle constant sentinel: BinOp(Add/FAdd, result, result, UNDEF)
                if *lhs == *result && *rhs == Value::UNDEF {
                    // --- Float constant sentinel (FAdd or float type) ---
                    // Check the float constant cache first: float constants
                    // are stored as global variables in .rodata and must be
                    // loaded via LA + FLD (not integer LI).
                    if let Some((sym_name, _fval)) =
                        self.float_constant_cache.get(&result.index()).cloned()
                    {
                        let fd = self.vmap.alloc_fpr(*result);
                        // Load the address of the .rodata float constant
                        // into T0 via LA (AUIPC+ADDI with PC-relative relocs).
                        self.emit(RvInstruction {
                            opcode: RvOpcode::LA,
                            rd: Some(T0),
                            rs1: None,
                            rs2: None,
                            rs3: None,
                            imm: 0,
                            symbol: Some(sym_name),
                            is_fp: false,
                            comment: Some("load float const addr".to_string()),
                            is_call_arg_setup: false,
                        });
                        // Load the double/float from memory into the FPR.
                        self.emit_load(fd, T0, 0, ty);
                        self.value_types.insert(result.index(), ty.clone());
                    } else if let Some(gname) = self.global_var_refs.get(result).cloned() {
                        // Global variable reference: load address of global.
                        let rd = self.dest_reg(*result, ty);
                        let addr_insts = self.materialize_address(rd, &gname);
                        for inst in addr_insts {
                            self.emit(inst);
                        }
                    } else {
                        let rd = self.dest_reg(*result, ty);
                        if let Some(&imm) = self.constant_values.get(&result.index()) {
                            // On RV64, all 32-bit (and smaller) integer
                            // values in registers must be sign-extended to
                            // 64 bits.  W-suffix instructions (ADDW, SUBW,
                            // NEGW, etc.) always sign-extend their 32-bit
                            // result, so computed values already satisfy
                            // this invariant.  But raw i64 constant values
                            // stored in the IR may be zero-extended (e.g.
                            // unsigned int 0xFFFFFFF8 stored as i64
                            // 4294967288 = 0x00000000_FFFFFFF8).  We must
                            // sign-extend to match the register convention.
                            let imm_adjusted = match ty {
                                IrType::I1 => imm & 1,
                                IrType::I8 => (imm as i8) as i64,
                                IrType::I16 => (imm as i16) as i64,
                                IrType::I32 => (imm as i32) as i64,
                                _ => imm,
                            };
                            let insts = self.materialize_immediate(rd, imm_adjusted);
                            for inst in insts {
                                self.emit(inst);
                            }
                        } else {
                            // Fallback: load 0.
                            self.emit(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, 0));
                        }
                    }
                } else if *op == BinOp::Xor && *rhs == Value::UNDEF {
                    // Bitwise NOT: XOR with UNDEF sentinel = ~operand.
                    // RISC-V encodes this as XORI rd, rs, -1.
                    let lhs_reg = self.src_reg(*lhs);
                    let rd = self.dest_reg(*result, ty);
                    self.emit_i(RvOpcode::XORI, rd, lhs_reg, -1i64 as i32 as i64);
                } else if *op == BinOp::Sub && *lhs == Value::UNDEF {
                    // Negation: Sub(UNDEF, operand) → SUB rd, x0, rs
                    // The IR builder encodes negation as `0 - operand` with
                    // Value::UNDEF as the zero placeholder.  On RISC-V we
                    // use the hardware zero register (x0) directly.
                    let rhs_reg = self.src_reg(*rhs);
                    let rd = self.dest_reg(*result, ty);
                    if matches!(ty, IrType::I32) {
                        self.emit_r(RvOpcode::SUBW, rd, ZERO, rhs_reg);
                    } else {
                        self.emit_r(RvOpcode::SUB, rd, ZERO, rhs_reg);
                    }
                } else if *op == BinOp::FSub && *lhs == Value::UNDEF {
                    // Float negation: FSub(UNDEF, operand) → FNEG rd, rs
                    // Encoded as FSGNJN rd, rs, rs (sign-inject-negate).
                    let rhs_reg = self.vmap.get_reg(*rhs);
                    let rd = self.vmap.alloc_fpr(*result);
                    if matches!(ty, IrType::F32) {
                        self.emit_r(RvOpcode::FSGNJN_S, rd, rhs_reg, rhs_reg);
                    } else {
                        self.emit_r(RvOpcode::FSGNJN_D, rd, rhs_reg, rhs_reg);
                    }
                } else {
                    let lhs_reg = self.src_reg(*lhs);
                    let rhs_reg = self.src_reg(*rhs);
                    let rd = self.dest_reg(*result, ty);
                    self.select_binop(rd, lhs_reg, rhs_reg, op, ty);
                }
                self.value_types.insert(result.index(), ty.clone());
            }

            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => {
                let lhs_reg = self.src_reg(*lhs);
                let rhs_reg = self.src_reg(*rhs);
                let rd = self.vmap.alloc_gpr(*result);
                self.emit_comparison(rd, lhs_reg, rhs_reg, op);
                // Comparison results are I32 (boolean in a GPR).
                self.value_types.insert(result.index(), IrType::I32);
            }

            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => {
                let lhs_reg = self.src_reg(*lhs);
                let rhs_reg = self.src_reg(*rhs);
                let rd = self.vmap.alloc_gpr(*result);
                // Determine if double based on FP register class of operands.
                // FPRs have IDs >= 32. We default to double (F64) for safety since
                // the IR FCmp does not carry the operand type directly.
                let is_double_op = true; // conservative default
                self.emit_fcmp(rd, lhs_reg, rhs_reg, op, is_double_op);
                // FCmp result is an integer boolean.
                self.value_types.insert(result.index(), IrType::I32);
            }

            Instruction::Branch { target, span: _ } => {
                let label = self.block_label(target.index());
                self.emit_branch(None, &label);
            }

            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                span: _,
            } => {
                let cond_reg = self.src_reg(*condition);
                let then_label = self.block_label(then_block.index());
                let else_label = self.block_label(else_block.index());

                // BNE cond_reg, ZERO, then_label (branch if true)
                self.emit_b(RvOpcode::BNE, cond_reg, ZERO, &then_label);
                // Fall-through or jump to else
                self.emit_branch(None, &else_label);
            }

            Instruction::Switch {
                value,
                default,
                cases,
                span: _,
            } => {
                let val_reg = self.src_reg(*value);
                let default_label = self.block_label(default.index());
                let case_entries: Vec<(i64, String)> = cases
                    .iter()
                    .map(|(val, blk)| (*val, self.block_label(blk.index())))
                    .collect();
                self.emit_switch(val_reg, &default_label, &case_entries);
            }

            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                span: _,
            } => {
                self.has_calls = true;

                // Resolve callee — check func_ref_names for direct call target.
                let callee_name = if let Some(fname) = self.func_ref_names.get(callee) {
                    fname.clone()
                } else {
                    format!("__ir_callee_{}", callee.index())
                };

                // Prepare argument registers.
                // For variadic functions on RISC-V LP64D, ALL FP arguments
                // must be passed in integer registers (the ABI mandates that
                // variadic FP values are conveyed as raw bits in GPRs).
                let is_variadic_call = self.variadic_functions.contains(&callee_name);
                let mut arg_regs: Vec<(u16, bool)> = Vec::new();
                for arg in args.iter() {
                    let reg = self.src_reg(*arg);
                    // Determine if the argument is floating-point by its IR
                    // type, NOT by the virtual register number.  Virtual
                    // register IDs (≥100) do not encode GPR-vs-FPR — that
                    // information is only in the type map.
                    let is_fp_arg = self
                        .value_types
                        .get(&arg.index())
                        .map(Self::uses_fpr)
                        .unwrap_or(false);
                    // NOTE: For variadic calls, the FP→GPR conversion
                    // (FMV.X.D) is deferred to emit_call where physical
                    // ABI registers are available.  Virtual register IDs
                    // do not distinguish GPR/FPR, so emitting FMV.X.D
                    // here with virtual registers would produce garbage
                    // after register allocation.
                    arg_regs.push((reg, is_fp_arg));
                }

                // ── Builtin interception ──────────────────────────────
                // Check if this is a compiler builtin that can be emitted
                // inline without a real function call.
                let handled_as_builtin =
                    self.try_emit_rv_builtin(&callee_name, result, args, return_type);

                if !handled_as_builtin {
                    // result is a Value; allocate a register for it
                    let result_reg = if *result != Value::UNDEF {
                        Some(if Self::uses_fpr(return_type) {
                            self.vmap.alloc_fpr(*result)
                        } else {
                            self.vmap.alloc_gpr(*result)
                        })
                    } else {
                        None
                    };
                    let result_is_fp = Self::uses_fpr(return_type);

                    self.emit_call(
                        &callee_name,
                        &arg_regs,
                        result_reg,
                        result_is_fp,
                        is_variadic_call,
                    );
                }
                // Record the return type so Store picks the right width.
                if *result != Value::UNDEF {
                    self.value_types.insert(result.index(), return_type.clone());
                }
            }

            Instruction::Return { value, span: _ } => {
                if let Some(val) = value {
                    let val_reg = self.src_reg(*val);
                    // Move return value to a0 or fa0
                    if is_fpr(val_reg) {
                        if val_reg != FA0 {
                            self.emit_fp_r(RvOpcode::FSGNJ_D, FA0, val_reg, val_reg);
                        }
                    } else if val_reg != A0 {
                        self.emit_i(RvOpcode::ADDI, A0, val_reg, 0); // MV
                    }
                }
                // Emit RET as terminator — generation.rs will insert the
                // trait-generated epilogue (frame teardown) before this.
                self.emit(RvInstruction::no_op(RvOpcode::RET));
            }

            Instruction::Phi {
                result,
                ty,
                incoming: _,
                span: _,
            } => {
                // Phi nodes should be eliminated before instruction selection.
                // If encountered, just allocate a register for the result.
                let _rd = self.dest_reg(*result, ty);
                self.value_types.insert(result.index(), ty.clone());
            }

            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type,
                in_bounds: _,
                span: _,
            } => {
                self.select_gep(*result, *base, indices, result_type);
                // GEP always produces a pointer.
                self.value_types.insert(result.index(), IrType::Ptr);
            }

            Instruction::BitCast {
                result,
                value,
                to_type,
                source_unsigned,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.dest_reg(*result, to_type);
                let _is_unsigned = *source_unsigned;
                // BitCast: reinterpret bits, no conversion
                if rd != src {
                    if Self::uses_fpr(to_type) && is_gpr(src) {
                        self.emit_fp_unary(RvOpcode::FMV_D_X, rd, src);
                    } else if !Self::uses_fpr(to_type) && is_fpr(src) {
                        self.emit_fp_unary(RvOpcode::FMV_X_D, rd, src);
                    } else {
                        self.emit_i(RvOpcode::ADDI, rd, src, 0); // MV
                    }
                }
                self.value_types.insert(result.index(), to_type.clone());
            }

            Instruction::Trunc {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // Truncation: mask to target width
                match to_type {
                    IrType::I1 => self.emit_i(RvOpcode::ANDI, rd, src, 1),
                    IrType::I8 => self.emit_i(RvOpcode::ANDI, rd, src, 0xFF),
                    IrType::I16 => {
                        self.emit_i(RvOpcode::SLLI, rd, src, 48);
                        self.emit_i(RvOpcode::SRLI, rd, rd, 48);
                    }
                    IrType::I32 => {
                        self.emit_i(RvOpcode::ADDIW, rd, src, 0);
                    }
                    _ => {
                        if rd != src {
                            self.emit_i(RvOpcode::ADDI, rd, src, 0);
                        }
                    }
                }
                self.value_types.insert(result.index(), to_type.clone());
            }

            Instruction::ZExt {
                result,
                value,
                to_type,
                from_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                self.emit_conversion(rd, src, from_type, to_type);
                self.value_types.insert(result.index(), to_type.clone());
            }

            Instruction::SExt {
                result,
                value,
                to_type,
                from_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // Sign extension: use SLLI+SRAI to replicate the sign
                // bit of the source width across the destination width.
                // The shift amount is (64 - source_bits).
                let shift = match from_type {
                    IrType::I1 => 63,
                    IrType::I8 => 56,
                    IrType::I16 => 48,
                    IrType::I32 => 32,
                    _ => 0,
                };
                if shift == 32 {
                    // ADDIW is the canonical I32→I64 sign extension on
                    // RV64: it takes the low 32 bits of src and sign-
                    // extends them to 64 bits.
                    self.emit_i(RvOpcode::ADDIW, rd, src, 0);
                } else if shift > 0 {
                    self.emit_i(RvOpcode::SLLI, rd, src, shift);
                    self.emit_i(RvOpcode::SRAI, rd, rd, shift);
                } else if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0);
                }
                self.value_types.insert(result.index(), to_type.clone());
            }

            Instruction::IntToPtr {
                result,
                value,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // IntToPtr: no-op on RV64 (both 64 bits)
                if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0);
                }
                self.value_types.insert(result.index(), IrType::Ptr);
            }

            Instruction::PtrToInt {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // PtrToInt: no-op on RV64, possibly truncate
                if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0);
                }
                // If target is smaller than 64 bits, truncate
                match to_type {
                    IrType::I32 => {
                        self.emit_i(RvOpcode::ADDIW, rd, rd, 0);
                    }
                    IrType::I16 => {
                        self.emit_i(RvOpcode::SLLI, rd, rd, 48);
                        self.emit_i(RvOpcode::SRLI, rd, rd, 48);
                    }
                    IrType::I8 => {
                        self.emit_i(RvOpcode::ANDI, rd, rd, 0xFF);
                    }
                    _ => {}
                }
                self.value_types.insert(result.index(), to_type.clone());
            }

            // ----- StackAlloc (dynamic stack allocation: __builtin_alloca) -----
            Instruction::StackAlloc { result, size, .. } => {
                let size_reg = self.src_reg(*size);
                let rd = self.vmap.alloc_gpr(*result);
                // SUB SP, SP, size_reg
                self.emit_r(RvOpcode::SUB, SP, SP, size_reg);
                // ANDI SP, SP, -16  (align to 16 bytes)
                self.emit_i(RvOpcode::ANDI, SP, SP, -16);
                // MV result, SP  (encoded as ADDI rd, SP, 0)
                self.emit_i(RvOpcode::ADDI, rd, SP, 0);
            }

            // StackSave — capture SP into a virtual register
            Instruction::StackSave { result, .. } => {
                let rd = self.vmap.alloc_gpr(*result);
                // MV rd, SP (encoded as ADDI rd, sp, 0)
                self.emit_i(RvOpcode::ADDI, rd, SP, 0);
            }

            // StackRestore — restore SP from a previously saved value
            Instruction::StackRestore { ptr, .. } => {
                let rs = self.src_reg(*ptr);
                // MV sp, rs (encoded as ADDI sp, rs, 0)
                self.emit_i(RvOpcode::ADDI, SP, rs, 0);
            }

            Instruction::InlineAsm {
                result: _,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_volatile: _,
                goto_targets,
                span: _,
            } => {
                // Parse constraints to identify immediate ("i"/"n"/"I") vs register operands.
                let constraint_parts: Vec<&str> = constraints.split(',').collect();
                let mut operand_reprs: Vec<String> = Vec::new();
                for (idx, op_val) in operands.iter().enumerate() {
                    let is_imm = if let Some(cstr) = constraint_parts.get(idx) {
                        let c = cstr.trim_start_matches('=').trim_start_matches('+');
                        c.contains('i') || c.contains('n') || c.contains('I')
                    } else {
                        false
                    };
                    if is_imm {
                        // For immediate constraints, use constant value directly.
                        if let Some(&imm) = self.constant_values.get(&op_val.index()) {
                            operand_reprs.push(format!("{}", imm));
                        } else {
                            operand_reprs.push(format!("{}", self.src_reg(*op_val)));
                        }
                    } else {
                        let reg = self.src_reg(*op_val);
                        operand_reprs.push(self.rv_reg_name(reg));
                    }
                }
                let target_labels: Vec<String> = goto_targets
                    .iter()
                    .map(|blk| self.block_label(blk.index()))
                    .collect();
                // Substitute %0, %1, ... in template with operand representations.
                let substituted =
                    Self::substitute_template(template, &operand_reprs, &target_labels);
                self.emit_inline_asm_substituted(&substituted, constraints, clobbers);
            }

            // IndirectBranch — computed goto: jr %reg (JALR x0, rs1, 0)
            Instruction::IndirectBranch { target, .. } => {
                let target_reg = self.src_reg(*target);
                self.emit(RvInstruction::i_type(RvOpcode::JALR, ZERO, target_reg, 0));
            }

            // BlockAddress — materialize address of a labeled basic block
            Instruction::BlockAddress { result, block, .. } => {
                let rd = self.vmap.alloc_gpr(*result);
                // Use same label convention as block_label() → .LBB_{idx}
                let block_label = self.block_label(block.index());
                // Use LA pseudo-instruction (AUIPC+ADDI) to load label address
                let mut la_inst = RvInstruction::u_type(RvOpcode::LA, rd, 0);
                la_inst.symbol = Some(block_label);
                self.emit(la_inst);
            }
        }

        // Return the newly emitted instructions
        self.instructions[start_idx..].to_vec()
    }

    // ===================================================================
    // Function-level instruction selection
    // ===================================================================

    /// Select RISC-V 64 instructions for an entire IR function.
    ///
    /// This is the main entry point for instruction selection. It:
    /// 1. Scans all blocks to build the block label map
    /// 2. Classifies function parameters per LP64D ABI
    /// 3. Selects instructions for each basic block
    /// 4. Computes the stack frame layout
    /// 5. Emits prologue and epilogue
    pub fn select_function(&mut self, func: &IrFunction, _abi: &RiscV64Abi) -> Vec<RvInstruction> {
        self.reset();

        // Phase 1: Build block label map.
        // Labels must be unique across the entire compilation unit.
        // IR block labels like "if.then" are NOT unique across functions
        // (or even within a function with nested ifs), so we always
        // prefix with the function name and block index.
        for (i, block) in func.blocks().iter().enumerate() {
            let base = block.label.as_deref().unwrap_or("bb");
            let label = format!(".L_{}_{}_{}", func.name, base, i);
            self.block_labels.insert(block.index, label);
        }

        // Phase 2: Set up parameter registers per LP64D ABI
        let mut abi_state = RiscV64Abi::new();
        for param in func.params.iter() {
            let loc = abi_state.classify_arg(&param_ir_to_ctype(&param.ty));
            match loc {
                ArgLocation::Register(reg) => {
                    self.vmap.set_reg(param.value, reg);
                }
                ArgLocation::RegisterPair(r1, _r2) => {
                    self.vmap.set_reg(param.value, r1);
                }
                ArgLocation::Stack(offset) => {
                    self.vmap.set_stack_offset(param.value, offset as i64);
                    // Load from stack into a temp register
                    let rd = self.vmap.alloc_gpr(param.value);
                    self.emit_i(RvOpcode::LD, rd, FP, offset as i64);
                }
            }
        }

        // Phase 3: Select instructions for each basic block
        for block in func.blocks().iter() {
            let label = self
                .block_labels
                .get(&block.index)
                .cloned()
                .unwrap_or_else(|| format!(".LBB_{}", block.index));
            self.emit_block_label(&label);

            for inst in &block.instructions {
                self.select_instruction(inst);
            }
        }

        // Phase 4: Compute frame layout
        // Callee-saved registers (excluding FP which is always saved)
        let callee_saved_count =
            self.used_callee_saved_gprs.len() + self.used_callee_saved_fprs.len();

        // Frame size = locals + callee_saved_regs * 8 + RA + FP + alignment
        // alloca_offset starts at 16 (reserving space for RA and FP saves).
        // compute_frame_layout already accounts for RA/FP (16 bytes), so
        // pass only the actual local variable space.
        let locals_size = if self.alloca_offset > 16 {
            (self.alloca_offset - 16) as usize
        } else {
            0
        };
        let frame_layout = RiscV64Abi::compute_frame_layout(
            &func
                .params
                .iter()
                .map(|p| param_ir_to_ctype(&p.ty))
                .collect::<Vec<_>>(),
            locals_size,
            callee_saved_count,
        );
        self.frame_size = frame_layout.total_size as i64;

        // Ensure frame size is aligned to STACK_ALIGNMENT (16 bytes)
        self.frame_size =
            (self.frame_size + (STACK_ALIGNMENT as i64) - 1) & !((STACK_ALIGNMENT as i64) - 1);

        // Prologue/epilogue are generated externally by the ArchCodegen
        // trait methods (emit_prologue/emit_epilogue) in mod.rs, which are
        // called by generation.rs after register allocation.  We do NOT
        // emit them here to avoid duplication.

        std::mem::take(&mut self.instructions)
    }
}

// ===========================================================================
// Helper: convert IrType to CType for ABI classification
// ===========================================================================

/// Map an IR type to a C type for ABI calling convention classification.
fn param_ir_to_ctype(ty: &IrType) -> CType {
    match ty {
        IrType::Void => CType::Void,
        IrType::I1 | IrType::I8 => CType::Char,
        IrType::I16 => CType::Short,
        IrType::I32 => CType::Int,
        IrType::I64 => CType::Long,
        IrType::I128 => CType::LongLong,
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::F80 => CType::LongDouble,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        IrType::Array(_, _) => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        IrType::Struct(_) => CType::Struct {
            name: None,
            fields: Vec::new(),
            packed: false,
            aligned: None,
        },
        IrType::Function(_, _) => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
    }
}
