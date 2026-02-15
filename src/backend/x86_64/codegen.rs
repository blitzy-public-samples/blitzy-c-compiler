//! x86-64 instruction selection and emission module for the BCC code generation backend.
//!
//! This module converts IR instructions into concrete x86-64 machine instructions,
//! implementing instruction selection with complex addressing modes
//! (base + index × scale + displacement), CMOV conditional moves, SSE/SSE2
//! floating-point, and variable-length instruction encoding with REX prefix
//! support.
//!
//! # Architecture
//!
//! The instruction selector (`X86_64InstrSelector`) iterates over each IR
//! basic block's instructions and produces a [`MachineFunction`] containing
//! [`MachineBasicBlock`]s filled with [`MachineInstr`]s. Each IR instruction
//! is pattern-matched and lowered to one or more x86-64 machine instructions.
//!
//! SSA virtual registers (`VirtualReg(ValueId)`) are used throughout instruction
//! selection; physical register assignment happens later during register allocation.
//!
//! # Primary Validation Target
//!
//! Per Section 0.1.2, x86-64 is the primary validation target and is validated
//! first in the fixed backend validation order.

use std::fmt;

use crate::backend::traits::{
    CodegenConfig, MachineFunction, MachineInstr, MachineOperand, PhysReg,
};
use crate::backend::x86_64::abi::{
    can_use_red_zone, compute_param_locations, ParamLocation, RED_ZONE_SIZE,
};
use crate::backend::x86_64::opcodes;
use crate::backend::x86_64::registers::{
    gpr_encoding, gpr_name_64, CALLER_SAVED, R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX,
    RCX, RDI, RDX, RSI, RSP, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{fx_hash_map, fx_hash_map_with_capacity, FxHashMap, FxHashSet};
use crate::common::types::{CType, FieldDef};
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Operand-size modifier flags for opcode bits 24-25
// ---------------------------------------------------------------------------
//
// The encoder interprets bits 24-25 of MachineInstr.opcode to select the
// operand size:  00 → DWord (32-bit, default), 01 → QWord (64-bit, REX.W),
// 10 → Byte (8-bit), 11 → Word (16-bit, 0x66 prefix).
//
// These constants are OR'd onto the base opcode when 64-bit (or other
// non-default) sizing is required.  Every stack-pointer and frame-pointer
// operation on x86-64 MUST use SIZE_QWORD to emit REX.W.

/// Operand-size flag that selects 64-bit (QWord) encoding (bits 24-25 = 01).
/// OR this with any base opcode to emit REX.W, e.g. `opcodes::MOV_RR | SIZE_QWORD`.
const SIZE_QWORD: u32 = 1 << 24;

// ---------------------------------------------------------------------------
// X86_64Opcode — high-level opcode enum for display/debug
// ---------------------------------------------------------------------------

/// High-level opcode enumeration for x86-64 machine instructions.
///
/// Each variant maps to one or more concrete machine instruction opcodes
/// from [`crate::backend::x86_64::opcodes`]. This enum is used for
/// diagnostic messages and debug output. The actual `MachineInstr.opcode`
/// field uses the `u32` constants from the `opcodes` module directly.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum X86_64Opcode {
    MOV,
    MOVSX,
    MOVZX,
    LEA,
    PUSH,
    POP,
    ADD,
    SUB,
    IMUL,
    IDIV,
    DIV,
    NEG,
    NOT,
    AND,
    OR,
    XOR,
    SHL,
    SHR,
    SAR,
    CMP,
    TEST,
    CDQ,
    CQO,
    CMOVE,
    CMOVNE,
    CMOVL,
    CMOVLE,
    CMOVG,
    CMOVGE,
    CMOVB,
    CMOVBE,
    CMOVA,
    CMOVAE,
    SETE,
    SETNE,
    SETL,
    SETLE,
    SETG,
    SETGE,
    SETB,
    SETBE,
    SETA,
    SETAE,
    JMP,
    JE,
    JNE,
    JL,
    JLE,
    JG,
    JGE,
    JB,
    JBE,
    JA,
    JAE,
    CALL,
    RET,
    NOP,
    MOVSS,
    MOVSD,
    ADDSS,
    ADDSD,
    SUBSS,
    SUBSD,
    MULSS,
    MULSD,
    DIVSS,
    DIVSD,
    COMISS,
    COMISD,
    UCOMISS,
    UCOMISD,
    CVTSI2SS,
    CVTSI2SD,
    CVTSS2SD,
    CVTSD2SS,
    CVTTSS2SI,
    CVTTSD2SI,
    XORPS,
    XORPD,
}

impl fmt::Display for X86_64Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::MOV => "mov",
            Self::MOVSX => "movsx",
            Self::MOVZX => "movzx",
            Self::LEA => "lea",
            Self::PUSH => "push",
            Self::POP => "pop",
            Self::ADD => "add",
            Self::SUB => "sub",
            Self::IMUL => "imul",
            Self::IDIV => "idiv",
            Self::DIV => "div",
            Self::NEG => "neg",
            Self::NOT => "not",
            Self::AND => "and",
            Self::OR => "or",
            Self::XOR => "xor",
            Self::SHL => "shl",
            Self::SHR => "shr",
            Self::SAR => "sar",
            Self::CMP => "cmp",
            Self::TEST => "test",
            Self::CDQ => "cdq",
            Self::CQO => "cqo",
            Self::CMOVE => "cmove",
            Self::CMOVNE => "cmovne",
            Self::CMOVL => "cmovl",
            Self::CMOVLE => "cmovle",
            Self::CMOVG => "cmovg",
            Self::CMOVGE => "cmovge",
            Self::CMOVB => "cmovb",
            Self::CMOVBE => "cmovbe",
            Self::CMOVA => "cmova",
            Self::CMOVAE => "cmovae",
            Self::SETE => "sete",
            Self::SETNE => "setne",
            Self::SETL => "setl",
            Self::SETLE => "setle",
            Self::SETG => "setg",
            Self::SETGE => "setge",
            Self::SETB => "setb",
            Self::SETBE => "setbe",
            Self::SETA => "seta",
            Self::SETAE => "setae",
            Self::JMP => "jmp",
            Self::JE => "je",
            Self::JNE => "jne",
            Self::JL => "jl",
            Self::JLE => "jle",
            Self::JG => "jg",
            Self::JGE => "jge",
            Self::JB => "jb",
            Self::JBE => "jbe",
            Self::JA => "ja",
            Self::JAE => "jae",
            Self::CALL => "call",
            Self::RET => "ret",
            Self::NOP => "nop",
            Self::MOVSS => "movss",
            Self::MOVSD => "movsd",
            Self::ADDSS => "addss",
            Self::ADDSD => "addsd",
            Self::SUBSS => "subss",
            Self::SUBSD => "subsd",
            Self::MULSS => "mulss",
            Self::MULSD => "mulsd",
            Self::DIVSS => "divss",
            Self::DIVSD => "divsd",
            Self::COMISS => "comiss",
            Self::COMISD => "comisd",
            Self::UCOMISS => "ucomiss",
            Self::UCOMISD => "ucomisd",
            Self::CVTSI2SS => "cvtsi2ss",
            Self::CVTSI2SD => "cvtsi2sd",
            Self::CVTSS2SD => "cvtss2sd",
            Self::CVTSD2SS => "cvtsd2ss",
            Self::CVTTSS2SI => "cvttss2si",
            Self::CVTTSD2SI => "cvttsd2si",
            Self::XORPS => "xorps",
            Self::XORPD => "xorpd",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// CondCode — x86-64 condition codes
// ---------------------------------------------------------------------------

/// x86-64 condition codes used with Jcc, SETcc, and CMOVcc instructions.
///
/// Maps directly to the 4-bit condition code encoding in the x86 instruction
/// set. Each variant corresponds to one or more flag test combinations
/// (ZF, SF, CF, OF, PF).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum CondCode {
    /// Equal (ZF=1).
    E,
    /// Not equal (ZF=0).
    NE,
    /// Signed less than (SF≠OF).
    L,
    /// Signed less than or equal (ZF=1 or SF≠OF).
    LE,
    /// Signed greater than (ZF=0 and SF=OF).
    G,
    /// Signed greater than or equal (SF=OF).
    GE,
    /// Unsigned below (CF=1).
    B,
    /// Unsigned below or equal (CF=1 or ZF=1).
    BE,
    /// Unsigned above (CF=0 and ZF=0).
    A,
    /// Unsigned above or equal (CF=0).
    AE,
    /// Sign flag set (SF=1).
    S,
    /// Sign flag not set (SF=0).
    NS,
    /// Parity even (PF=1) — used for unordered float comparisons.
    P,
    /// Parity odd (PF=0) — used for ordered float comparisons.
    NP,
    /// Overflow (OF=1).
    O,
    /// No overflow (OF=0).
    NO,
}

impl CondCode {
    /// Converts an IR integer comparison predicate to an x86-64 condition code.
    ///
    /// The CMP instruction sets flags such that:
    /// - Signed comparisons use SF, OF, ZF (Jcc with L/LE/G/GE)
    /// - Unsigned comparisons use CF, ZF (Jcc with B/BE/A/AE)
    /// - Equality uses ZF (JE/JNE)
    pub fn from_icmp_predicate(pred: &ICmpPredicate) -> Self {
        match pred {
            ICmpPredicate::Eq => CondCode::E,
            ICmpPredicate::Ne => CondCode::NE,
            ICmpPredicate::Slt => CondCode::L,
            ICmpPredicate::Sle => CondCode::LE,
            ICmpPredicate::Sgt => CondCode::G,
            ICmpPredicate::Sge => CondCode::GE,
            ICmpPredicate::Ult => CondCode::B,
            ICmpPredicate::Ule => CondCode::BE,
            ICmpPredicate::Ugt => CondCode::A,
            ICmpPredicate::Uge => CondCode::AE,
        }
    }

    /// Converts an IR floating-point comparison predicate to an x86-64
    /// condition code for use after UCOMISS/UCOMISD.
    ///
    /// UCOMISS/UCOMISD set ZF, PF, CF:
    /// - Unordered (NaN): ZF=1, PF=1, CF=1
    /// - Equal: ZF=1, PF=0, CF=0
    /// - Less than: ZF=0, PF=0, CF=1
    /// - Greater than: ZF=0, PF=0, CF=0
    ///
    /// For ordered predicates, we check that PF=0 (no NaN) in addition
    /// to the relational condition. For unordered predicates, NaN results
    /// in true.
    pub fn from_fcmp_predicate(pred: &FCmpPredicate) -> Self {
        match pred {
            // Ordered comparisons — false when NaN
            FCmpPredicate::OEq => CondCode::E, // ZF=1 && PF=0 (need extra PF check)
            FCmpPredicate::ONe => CondCode::NE, // ZF=0 && PF=0
            FCmpPredicate::Ogt => CondCode::A, // CF=0 && ZF=0 (above)
            FCmpPredicate::Oge => CondCode::AE, // CF=0 (above or equal)
            FCmpPredicate::Olt => CondCode::B, // CF=1 (below)
            FCmpPredicate::Ole => CondCode::BE, // CF=1 || ZF=1
            FCmpPredicate::Ord => CondCode::NP, // PF=0 (no NaN)
            // Unordered comparisons — true when NaN
            FCmpPredicate::Uno => CondCode::P,  // PF=1
            FCmpPredicate::UEq => CondCode::E,  // ZF=1 (includes NaN case)
            FCmpPredicate::UNe => CondCode::NE, // ZF=0 || PF=1
            FCmpPredicate::Ugt => CondCode::A,  // CF=0 && ZF=0
            FCmpPredicate::Uge => CondCode::AE, // CF=0
            FCmpPredicate::Ult => CondCode::B,  // CF=1
            FCmpPredicate::Ule => CondCode::BE, // CF=1 || ZF=1
        }
    }

    /// Returns the logically inverted condition code.
    pub fn invert(&self) -> Self {
        match self {
            CondCode::E => CondCode::NE,
            CondCode::NE => CondCode::E,
            CondCode::L => CondCode::GE,
            CondCode::LE => CondCode::G,
            CondCode::G => CondCode::LE,
            CondCode::GE => CondCode::L,
            CondCode::B => CondCode::AE,
            CondCode::BE => CondCode::A,
            CondCode::A => CondCode::BE,
            CondCode::AE => CondCode::B,
            CondCode::S => CondCode::NS,
            CondCode::NS => CondCode::S,
            CondCode::P => CondCode::NP,
            CondCode::NP => CondCode::P,
            CondCode::O => CondCode::NO,
            CondCode::NO => CondCode::O,
        }
    }

    /// Returns the `u32` opcode for the Jcc instruction using this condition.
    /// Used by instruction selection (`lower_cond_branch`) and peephole
    /// optimizers to emit conditional jumps parameterised by this code.
    pub fn jcc_opcode(&self) -> u32 {
        opcodes::JCC
    }

    /// Returns the `u32` opcode for the SETcc instruction using this condition.
    pub fn setcc_opcode(&self) -> u32 {
        opcodes::SET_CC
    }

    /// Returns the `u32` opcode for the CMOVcc instruction using this condition.
    /// Used when lowering select-like patterns to branchless CMOVcc.
    pub fn cmovcc_opcode(&self) -> u32 {
        opcodes::CMOV
    }

    /// Returns this condition code as an immediate value for encoding
    /// in condition-parameterised opcodes (JCC, SETcc, CMOVcc).
    pub fn encoding(&self) -> i64 {
        match self {
            CondCode::O => 0,
            CondCode::NO => 1,
            CondCode::B => 2,
            CondCode::AE => 3,
            CondCode::E => 4,
            CondCode::NE => 5,
            CondCode::BE => 6,
            CondCode::A => 7,
            CondCode::S => 8,
            CondCode::NS => 9,
            CondCode::P => 10,
            CondCode::NP => 11,
            CondCode::L => 12,
            CondCode::GE => 13,
            CondCode::LE => 14,
            CondCode::G => 15,
        }
    }
}

impl fmt::Display for CondCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            CondCode::E => "e",
            CondCode::NE => "ne",
            CondCode::L => "l",
            CondCode::LE => "le",
            CondCode::G => "g",
            CondCode::GE => "ge",
            CondCode::B => "b",
            CondCode::BE => "be",
            CondCode::A => "a",
            CondCode::AE => "ae",
            CondCode::S => "s",
            CondCode::NS => "ns",
            CondCode::P => "p",
            CondCode::NP => "np",
            CondCode::O => "o",
            CondCode::NO => "no",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// MemoryOperand — x86-64 complex addressing mode
// ---------------------------------------------------------------------------

/// Represents an x86-64 memory addressing mode.
///
/// Effective address = `[base + index * scale + displacement]`
///
/// Supports all x86-64 addressing combinations:
/// - Direct register: `(%rax)` — base only
/// - Register + displacement: `8(%rbp)` — base + offset
/// - Base + index*scale: `(%rbx, %rcx, 4)` — SIB addressing
/// - Full: `offset(%base, %index, scale)` — all components
/// - RIP-relative: `symbol(%rip)` — for PIC code
#[derive(Clone, Debug)]
pub struct MemoryOperand {
    /// Base register (e.g., RBP for frame-relative addressing).
    pub base: Option<PhysReg>,
    /// Index register for scaled indexed addressing.
    pub index: Option<PhysReg>,
    /// Scale factor applied to the index register (1, 2, 4, or 8).
    pub scale: u8,
    /// Signed displacement in bytes from base.
    pub displacement: i32,
    /// If true, this is a RIP-relative memory operand for PIC addressing.
    pub rip_relative: bool,
}

impl MemoryOperand {
    /// Returns `true` if this is a simple base+displacement operand
    /// with no index register or scaling.
    pub fn is_simple(&self) -> bool {
        self.index.is_none() && !self.rip_relative
    }

    /// Returns `true` if this addressing mode requires a SIB byte
    /// (index register present, or base is RSP/R12 which always needs SIB).
    pub fn has_sib(&self) -> bool {
        if self.index.is_some() {
            return true;
        }
        if let Some(base) = self.base {
            // RSP (PhysReg(4)) and R12 (PhysReg(12)) always require SIB byte
            let enc = gpr_encoding(base);
            enc == 4 // RSP or R12 encoding
        } else {
            false
        }
    }

    /// Converts this `MemoryOperand` to a `MachineOperand::Memory`.
    /// Panics if base is None and not RIP-relative.
    pub fn to_machine_operand(&self) -> MachineOperand {
        if self.rip_relative {
            // RIP-relative uses RBP as a sentinel for the base
            MachineOperand::Memory {
                base: RBP,
                offset: self.displacement,
                index: self.index,
                scale: self.scale,
            }
        } else {
            let base = self.base.expect("MemoryOperand requires a base register");
            MachineOperand::Memory {
                base,
                offset: self.displacement,
                index: self.index,
                scale: self.scale,
            }
        }
    }

    /// Creates a simple base+displacement operand.
    pub fn base_disp(base: PhysReg, disp: i32) -> Self {
        MemoryOperand {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: disp,
            rip_relative: false,
        }
    }

    /// Creates a base+index*scale+displacement operand.
    pub fn base_index_scale_disp(base: PhysReg, index: PhysReg, scale: u8, disp: i32) -> Self {
        MemoryOperand {
            base: Some(base),
            index: Some(index),
            scale,
            displacement: disp,
            rip_relative: false,
        }
    }

    /// Creates a RIP-relative operand for PIC code.
    pub fn rip_relative_disp(disp: i32) -> Self {
        MemoryOperand {
            base: None,
            index: None,
            scale: 1,
            displacement: disp,
            rip_relative: true,
        }
    }
}

impl fmt::Display for MemoryOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.rip_relative {
            write!(f, "{}(%rip)", self.displacement)
        } else if let Some(base) = self.base {
            let base_name = gpr_name_64(base);
            if let Some(idx) = self.index {
                let idx_name = gpr_name_64(idx);
                if self.displacement != 0 {
                    write!(
                        f,
                        "{}({}, {}, {})",
                        self.displacement, base_name, idx_name, self.scale
                    )
                } else {
                    write!(f, "({}, {}, {})", base_name, idx_name, self.scale)
                }
            } else if self.displacement != 0 {
                write!(f, "{}({})", self.displacement, base_name)
            } else {
                write!(f, "({})", base_name)
            }
        } else {
            write!(f, "{}", self.displacement)
        }
    }
}

// ---------------------------------------------------------------------------
// X86_64InstrSelector — main instruction selection engine
// ---------------------------------------------------------------------------

/// Jump table density threshold. If the ratio of case values to the range
/// span exceeds this threshold, a jump table is used; otherwise, a linear
/// comparison chain is generated.
const JUMP_TABLE_DENSITY_THRESHOLD: f64 = 0.4;

/// Maximum number of cases before we always use a jump table regardless
/// of density.
const JUMP_TABLE_MIN_CASES: usize = 4;

/// x86-64 instruction selection engine.
///
/// Transforms IR functions into machine functions by pattern-matching each
/// IR instruction and emitting the corresponding x86-64 machine instructions.
///
/// # Usage
///
/// ```ignore
/// let selector = X86_64InstrSelector::new(&target, &diag);
/// let machine_func = selector.select_instructions(&ir_func);
/// ```
pub struct X86_64InstrSelector<'a> {
    /// Code generation configuration (contains target, PIC, debug, etc.).
    config: &'a CodegenConfig,
    /// Diagnostic engine for error/warning reporting during instruction selection.
    diagnostics: DiagnosticEngine,
    /// Map from IR ValueId to the MachineOperand representing it.
    /// Virtual registers are used during isel; physical assignment is deferred
    /// to register allocation.
    value_map: FxHashMap<u32, MachineOperand>,
    /// Map from IR BasicBlockId to machine BasicBlock id.
    bb_map: FxHashMap<u32, u32>,
    /// Frame slot counter for alloca instructions.
    frame_slot_counter: u32,
    /// Tracks whether the current function has any call instructions.
    has_calls: bool,
    /// Set of callee-saved registers used (populated during isel).
    used_callee_saved: FxHashSet<PhysReg>,
    /// Next virtual register ID for temporary values created during isel.
    next_vreg: u32,
}

impl<'a> X86_64InstrSelector<'a> {
    /// Creates a new x86-64 instruction selector.
    ///
    /// # Arguments
    ///
    /// * `config` — Code generation configuration containing target, PIC mode,
    ///   debug info, and security mitigation flags.
    pub fn new(config: &'a CodegenConfig) -> Self {
        X86_64InstrSelector {
            config,
            diagnostics: DiagnosticEngine::new(),
            value_map: fx_hash_map(),
            bb_map: fx_hash_map(),
            frame_slot_counter: 0,
            has_calls: false,
            used_callee_saved: FxHashSet::default(),
            next_vreg: 0,
        }
    }

    /// Selects x86-64 machine instructions for an entire IR function.
    ///
    /// This is the main entry point for instruction selection. It:
    /// 1. Creates the machine function and basic block mapping
    /// 2. Sets up parameter registers per System V AMD64 ABI
    /// 3. Iterates over each IR basic block, lowering instructions
    /// 4. Generates prologue and epilogue code
    ///
    /// # Arguments
    ///
    /// * `func` — The IR function to lower.
    ///
    /// # Returns
    ///
    /// A [`MachineFunction`] containing selected x86-64 machine instructions.
    pub fn select_instructions(&mut self, func: &IrFunction) -> MachineFunction {
        // Reset state for each function
        self.value_map = fx_hash_map_with_capacity(func.local_values.len());
        self.bb_map = fx_hash_map_with_capacity(func.basic_blocks.len());
        self.frame_slot_counter = 0;
        self.has_calls = false;
        self.used_callee_saved = FxHashSet::default();
        self.next_vreg = func.next_value_id;

        let stack_align = self.config.target.stack_alignment();
        let mut mfunc = MachineFunction::new(func.name.clone(), stack_align);

        // Step 1: Create machine basic blocks and build the BB map.
        for bb in func.basic_blocks.iter() {
            let mbb_id = mfunc.create_block();
            self.bb_map.insert(bb.id.0, mbb_id);
        }

        // Step 1b: Pre-populate value_map for special IR values.
        //
        // The IR builder encodes certain value kinds (global references,
        // integer constants, float constants, null pointers) purely by
        // naming convention in the ValueInfo table rather than emitting
        // dedicated instructions.  The codegen must recognize these names
        // and map them to the appropriate MachineOperand *before*
        // instruction selection begins — otherwise get_operand() returns
        // a VirtualReg which produces incorrect code.
        //
        // Naming conventions (see ir::builder):
        //   "global.<name>"        → Symbol("<name>")  — external function / global var
        //   "const.int.<value>"    → Immediate(<value>) — integer constant
        //   "const.float.<value>"  → Immediate(f64 bits) — float constant (stored as bits)
        //   "const.null"           → Immediate(0)        — null pointer
        for vi in &func.local_values {
            if let Some(ref n) = vi.name {
                if let Some(sym_name) = n.strip_prefix("global.") {
                    self.value_map.insert(
                        vi.id.0,
                        MachineOperand::Symbol(sym_name.to_string()),
                    );
                } else if let Some(int_str) = n.strip_prefix("const.int.") {
                    if let Ok(val) = int_str.parse::<i64>() {
                        self.value_map.insert(
                            vi.id.0,
                            MachineOperand::Immediate(val),
                        );
                    }
                } else if let Some(flt_str) = n.strip_prefix("const.float.") {
                    if let Ok(val) = flt_str.parse::<f64>() {
                        // Store float constant as raw bits for later
                        // materialization via SSE immediate patterns
                        self.value_map.insert(
                            vi.id.0,
                            MachineOperand::Immediate(val.to_bits() as i64),
                        );
                    }
                } else if n == "const.null" {
                    self.value_map.insert(
                        vi.id.0,
                        MachineOperand::Immediate(0),
                    );
                }
            }
        }

        // Step 2: Lower parameters — copy from physical ABI registers to
        // virtual registers for each function parameter.
        self.lower_parameters(func, &mut mfunc);

        // Step 3: Lower each IR basic block's instructions.
        for bb in &func.basic_blocks {
            let mbb_id = self.bb_map[&bb.id.0];
            for instr in bb.instructions() {
                let machine_instrs = self.select_instruction(instr, func);
                if let Some(mbb) = mfunc.get_block_mut(mbb_id) {
                    for mi in machine_instrs {
                        mbb.push_instr(mi);
                    }
                }
            }
        }

        // Step 4: Record function-level metadata.
        mfunc.has_calls = self.has_calls;
        for reg in &self.used_callee_saved {
            mfunc.mark_callee_saved_used(*reg);
        }

        mfunc
    }

    // -----------------------------------------------------------------------
    // Parameter lowering
    // -----------------------------------------------------------------------

    /// Lowers function parameters by mapping ABI physical register locations
    /// to virtual registers via MOV instructions in the entry block.
    fn lower_parameters(&mut self, func: &IrFunction, mfunc: &mut MachineFunction) {
        if func.params.is_empty() {
            return;
        }

        // Collect C types for ABI classification
        let param_ctypes: Vec<CType> = func
            .params
            .iter()
            .map(|p| ir_type_to_ctype(&p.ty))
            .collect();

        let locations = compute_param_locations(&param_ctypes, &self.config.target);

        let entry_id = if let Some(&mbb_id) = self.bb_map.get(&func.entry_block_id.0) {
            mbb_id
        } else {
            return;
        };

        for (i, param) in func.params.iter().enumerate() {
            if i >= locations.len() {
                break;
            }
            let vreg = MachineOperand::VirtualReg(param.id);
            self.value_map.insert(param.id.0, vreg.clone());

            match &locations[i] {
                ParamLocation::Register(phys) => {
                    // MOV from physical ABI register to virtual register
                    let mut mi = MachineInstr::new(opcodes::MOV_RR);
                    mi.add_operand(vreg);
                    mi.add_operand(MachineOperand::Register(*phys));
                    if let Some(mbb) = mfunc.get_block_mut(entry_id) {
                        mbb.push_instr(mi);
                    }
                }
                ParamLocation::RegisterPair(lo, _hi) => {
                    // Two MOVs for register pairs (e.g., 128-bit structs)
                    let mut mi_lo = MachineInstr::new(opcodes::MOV_RR);
                    mi_lo.add_operand(vreg.clone());
                    mi_lo.add_operand(MachineOperand::Register(*lo));
                    if let Some(mbb) = mfunc.get_block_mut(entry_id) {
                        mbb.push_instr(mi_lo);
                    }
                    // The high part is handled via a second virtual register if needed
                }
                ParamLocation::Stack { offset } => {
                    // Load from stack: MOV vreg, [RBP + offset + 16]
                    // +16 accounts for saved RBP and return address
                    let mut mi = MachineInstr::new(opcodes::MOV_RM);
                    mi.add_operand(vreg);
                    mi.add_operand(MachineOperand::Memory {
                        base: RBP,
                        offset: *offset + 16,
                        index: None,
                        scale: 1,
                    });
                    if let Some(mbb) = mfunc.get_block_mut(entry_id) {
                        mbb.push_instr(mi);
                    }
                }
                ParamLocation::HiddenPointer(reg) => {
                    let mut mi = MachineInstr::new(opcodes::MOV_RR);
                    mi.add_operand(vreg);
                    mi.add_operand(MachineOperand::Register(*reg));
                    if let Some(mbb) = mfunc.get_block_mut(entry_id) {
                        mbb.push_instr(mi);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Core instruction selection dispatch
    // -----------------------------------------------------------------------

    /// Selects machine instructions for a single IR instruction.
    ///
    /// Returns a vector of machine instructions that implement the semantics
    /// of the given IR instruction on x86-64.
    fn select_instruction(&mut self, instr: &Instruction, func: &IrFunction) -> Vec<MachineInstr> {
        match instr {
            Instruction::Alloca {
                result,
                ty,
                alignment,
            } => self.lower_alloca(*result, ty, *alignment),
            Instruction::Load {
                result,
                ptr,
                ty,
                volatile: _,
            } => self.lower_load(*result, *ptr, ty, func),
            Instruction::Store {
                value,
                ptr,
                volatile: _,
            } => self.lower_store(*value, *ptr, func),
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => self.lower_binop(*result, *op, *lhs, *rhs, ty, func),
            Instruction::ICmp {
                result,
                pred,
                lhs,
                rhs,
            } => self.lower_icmp(*result, pred, *lhs, *rhs, func),
            Instruction::FCmp {
                result,
                pred,
                lhs,
                rhs,
            } => self.lower_fcmp(*result, pred, *lhs, *rhs, func),
            Instruction::Branch { target } => self.lower_branch(*target),
            Instruction::CondBranch {
                condition,
                true_target,
                false_target,
            } => self.lower_cond_branch(*condition, *true_target, *false_target),
            Instruction::Switch {
                value,
                default,
                cases,
            } => self.lower_switch(*value, cases, *default, &self.bb_map.clone()),
            Instruction::Call {
                result,
                callee,
                args,
                is_tail: _,
            } => self.lower_call(*result, *callee, args, func),
            Instruction::Return { value } => self.lower_return(*value, func),
            Instruction::Phi {
                result,
                ty: _,
                incoming: _,
            } => {
                // Phi nodes should be eliminated before codegen.
                // If encountered, emit a NOP placeholder.
                let _vreg = self.get_or_create_vreg(*result);
                vec![]
            }
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                ty,
                in_bounds,
            } => self.lower_gep(*result, *base, indices, ty.clone(), *in_bounds, func),
            Instruction::BitCast {
                result,
                value,
                to_ty,
            } => self.lower_bitcast(*result, *value, to_ty, func),
            Instruction::Trunc {
                result,
                value,
                to_ty,
            } => self.lower_trunc(*result, *value, to_ty, func),
            Instruction::ZExt {
                result,
                value,
                to_ty,
            } => self.lower_zext(*result, *value, to_ty, func),
            Instruction::SExt {
                result,
                value,
                to_ty,
            } => self.lower_sext(*result, *value, to_ty, func),
            Instruction::IntToPtr {
                result,
                value,
                to_ty: _,
            } => {
                // IntToPtr is a no-op move (same size) on x86-64
                self.lower_mov(*result, *value)
            }
            Instruction::PtrToInt {
                result,
                value,
                to_ty,
            } => {
                // PtrToInt: may need truncation if target is smaller than 64-bit
                self.lower_ptrtoint(*result, *value, to_ty, func)
            }
            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_align_stack: _,
            } => self.lower_inline_asm(
                result.as_ref().copied(),
                template,
                constraints,
                operands,
                clobbers,
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Alloca lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR Alloca to a frame slot assignment.
    fn lower_alloca(
        &mut self,
        result: ValueId,
        _ty: &IrType,
        _alignment: u32,
    ) -> Vec<MachineInstr> {
        let slot = self.frame_slot_counter;
        self.frame_slot_counter += 1;

        // Map the alloca result to a FrameIndex operand
        let fi = MachineOperand::FrameIndex(slot);
        self.value_map.insert(result.0, fi);

        // No machine instructions emitted for alloca — it's a frame-level
        // operation resolved during prologue generation.
        vec![]
    }

    // -----------------------------------------------------------------------
    // Load lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR Load to an x86-64 MOV from memory.
    fn lower_load(
        &mut self,
        result: ValueId,
        ptr: ValueId,
        ty: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(ptr);
        let mut instrs = Vec::new();

        if ty.is_floating() {
            // Use SSE move for floating-point loads
            let opcode = match ty {
                IrType::F32 => opcodes::MOVSS,
                IrType::F64 => opcodes::MOVSD,
                _ => opcodes::MOVSD, // F80 handled separately
            };
            let mut mi = MachineInstr::new(opcode);
            mi.add_operand(dst);
            mi.add_operand(src);
            instrs.push(mi);
        } else {
            // Use integer MOV for integer/pointer loads
            let opcode = match src {
                MachineOperand::Memory { .. } | MachineOperand::FrameIndex(_) => opcodes::MOV_RM,
                _ => opcodes::MOV_RM,
            };
            let mut mi = MachineInstr::new(opcode);
            mi.add_operand(dst);
            mi.add_operand(src);
            instrs.push(mi);
        }
        instrs
    }

    // -----------------------------------------------------------------------
    // Store lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR Store to an x86-64 MOV to memory.
    fn lower_store(
        &mut self,
        value: ValueId,
        ptr: ValueId,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let src = self.get_operand(value);
        let dst = self.get_operand(ptr);
        let mut instrs = Vec::new();

        let val_ty = func.get_value_type(value);
        if val_ty.is_floating() {
            let opcode = match val_ty {
                IrType::F32 => opcodes::MOVSS,
                IrType::F64 => opcodes::MOVSD,
                _ => opcodes::MOVSD,
            };
            let mut mi = MachineInstr::new(opcode);
            mi.add_operand(dst);
            mi.add_operand(src);
            instrs.push(mi);
        } else {
            // MOV [mem], reg/imm
            let opcode = match &src {
                MachineOperand::Immediate(_) => opcodes::MOV_MI,
                _ => opcodes::MOV_MR,
            };
            let mut mi = MachineInstr::new(opcode);
            mi.add_operand(dst);
            mi.add_operand(src);
            instrs.push(mi);
        }
        instrs
    }

    // -----------------------------------------------------------------------
    // Binary operation lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR BinOp to x86-64 arithmetic/bitwise/shift instructions.
    fn lower_binop(
        &mut self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);

        // Floating-point binary operations use SSE instructions
        if op.is_floating_point() {
            return self.lower_fp_binop(dst, op, lhs_op, rhs_op, ty);
        }

        // Integer binary operations
        let mut instrs = Vec::new();

        match op {
            BinOp::Add => {
                // mov dst, lhs; add dst, rhs
                instrs.push(self.make_mov(dst.clone(), lhs_op));
                let mut mi = MachineInstr::new(opcodes::ADD_RR);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Sub => {
                instrs.push(self.make_mov(dst.clone(), lhs_op));
                let mut mi = MachineInstr::new(opcodes::SUB_RR);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Mul => {
                // IMUL r64, r/m64 — two-operand form
                instrs.push(self.make_mov(dst.clone(), lhs_op));
                let mut mi = MachineInstr::new(opcodes::IMUL_RR);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::SDiv => {
                // Signed division: move dividend to RAX, sign-extend to RDX:RAX,
                // IDIV divisor, result in RAX
                instrs.extend(self.lower_division(dst, lhs_op, rhs_op, true, false, ty));
            }
            BinOp::UDiv => {
                // Unsigned division: move dividend to RAX, zero RDX, DIV divisor
                instrs.extend(self.lower_division(dst, lhs_op, rhs_op, false, false, ty));
            }
            BinOp::SRem => {
                // Signed remainder: same as SDiv but result is in RDX
                instrs.extend(self.lower_division(dst, lhs_op, rhs_op, true, true, ty));
            }
            BinOp::URem => {
                // Unsigned remainder: same as UDiv but result is in RDX
                instrs.extend(self.lower_division(dst, lhs_op, rhs_op, false, true, ty));
            }
            BinOp::Shl => {
                instrs.extend(self.lower_shift(dst, lhs_op, rhs_op, opcodes::SHL));
            }
            BinOp::LShr => {
                instrs.extend(self.lower_shift(dst, lhs_op, rhs_op, opcodes::SHR));
            }
            BinOp::AShr => {
                instrs.extend(self.lower_shift(dst, lhs_op, rhs_op, opcodes::SAR));
            }
            BinOp::And => {
                instrs.push(self.make_mov(dst.clone(), lhs_op));
                let mut mi = MachineInstr::new(opcodes::AND_RR);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Or => {
                instrs.push(self.make_mov(dst.clone(), lhs_op));
                let mut mi = MachineInstr::new(opcodes::OR_RR);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Xor => {
                instrs.push(self.make_mov(dst.clone(), lhs_op));
                let mut mi = MachineInstr::new(opcodes::XOR_RR);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            // Floating-point handled above
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                unreachable!("FP ops handled in lower_fp_binop");
            }
        }
        instrs
    }

    /// Lowers floating-point binary operations to SSE/SSE2 instructions.
    fn lower_fp_binop(
        &self,
        dst: MachineOperand,
        op: BinOp,
        lhs: MachineOperand,
        rhs: MachineOperand,
        ty: &IrType,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let is_f32 = matches!(ty, IrType::F32);

        // MOV lhs to dst
        let mov_op = if is_f32 {
            opcodes::MOVSS
        } else {
            opcodes::MOVSD
        };
        let mut mov = MachineInstr::new(mov_op);
        mov.add_operand(dst.clone());
        mov.add_operand(lhs);
        instrs.push(mov);

        let opcode = match (op, is_f32) {
            (BinOp::FAdd, true) => opcodes::ADDSS,
            (BinOp::FAdd, false) => opcodes::ADDSD,
            (BinOp::FSub, true) => opcodes::SUBSS,
            (BinOp::FSub, false) => opcodes::SUBSD,
            (BinOp::FMul, true) => opcodes::MULSS,
            (BinOp::FMul, false) => opcodes::MULSD,
            (BinOp::FDiv, true) => opcodes::DIVSS,
            (BinOp::FDiv, false) => opcodes::DIVSD,
            (BinOp::FRem, _) => {
                // FRem has no direct x86 instruction — would need a library call
                // For now emit DIVSD and handle remainder via separate logic
                if is_f32 {
                    opcodes::DIVSS
                } else {
                    opcodes::DIVSD
                }
            }
            _ => unreachable!("Not a floating-point BinOp"),
        };

        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(rhs);
        instrs.push(mi);
        instrs
    }

    /// Lowers integer division and remainder operations.
    ///
    /// x86-64 IDIV/DIV uses RAX:RDX as the implicit dividend.
    /// Result: RAX = quotient, RDX = remainder.
    fn lower_division(
        &mut self,
        dst: MachineOperand,
        lhs: MachineOperand,
        rhs: MachineOperand,
        signed: bool,
        is_remainder: bool,
        ty: &IrType,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Move dividend to RAX
        let mut mov_rax = MachineInstr::new(opcodes::MOV_RR);
        mov_rax.add_operand(MachineOperand::Register(RAX));
        mov_rax.add_operand(lhs);
        mov_rax.add_implicit_def(RAX);
        instrs.push(mov_rax);

        if signed {
            // Sign-extend RAX into RDX:RAX
            let sign_ext_op = match ty {
                IrType::I32 => opcodes::CDQ,
                _ => opcodes::CQO,
            };
            let mut cqo = MachineInstr::new(sign_ext_op);
            cqo.add_implicit_def(RDX);
            cqo.add_implicit_use(RAX);
            instrs.push(cqo);
        } else {
            // Zero RDX for unsigned division
            let mut xor_rdx = MachineInstr::new(opcodes::XOR_RR);
            xor_rdx.add_operand(MachineOperand::Register(RDX));
            xor_rdx.add_operand(MachineOperand::Register(RDX));
            xor_rdx.add_implicit_def(RDX);
            instrs.push(xor_rdx);
        }

        // IDIV/DIV r/m64
        let div_op = if signed { opcodes::IDIV } else { opcodes::DIV };
        let mut div = MachineInstr::new(div_op);
        div.add_operand(rhs);
        div.add_implicit_def(RAX);
        div.add_implicit_def(RDX);
        div.add_implicit_use(RAX);
        div.add_implicit_use(RDX);
        instrs.push(div);

        // Move result from RAX (quotient) or RDX (remainder) to dst
        let result_reg = if is_remainder { RDX } else { RAX };
        let mut mov_result = MachineInstr::new(opcodes::MOV_RR);
        mov_result.add_operand(dst);
        mov_result.add_operand(MachineOperand::Register(result_reg));
        instrs.push(mov_result);

        instrs
    }

    /// Lowers shift operations.
    ///
    /// x86-64 shifts require the shift amount in CL (low byte of RCX)
    /// or as an immediate.
    fn lower_shift(
        &mut self,
        dst: MachineOperand,
        lhs: MachineOperand,
        rhs: MachineOperand,
        opcode: u32,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Move value to dst
        instrs.push(self.make_mov(dst.clone(), lhs));

        match &rhs {
            MachineOperand::Immediate(amt) => {
                // Shift by immediate
                let mut mi = MachineInstr::new(opcode);
                mi.add_operand(dst);
                mi.add_operand(MachineOperand::Immediate(*amt & 63));
                instrs.push(mi);
            }
            _ => {
                // Shift by variable: move amount to RCX, use CL
                let mut mov_cl = MachineInstr::new(opcodes::MOV_RR);
                mov_cl.add_operand(MachineOperand::Register(RCX));
                mov_cl.add_operand(rhs);
                instrs.push(mov_cl);

                let mut mi = MachineInstr::new(opcode);
                mi.add_operand(dst);
                mi.add_implicit_use(RCX);
                instrs.push(mi);
            }
        }
        instrs
    }

    // -----------------------------------------------------------------------
    // Comparison lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR integer comparison to CMP + SETcc.
    fn lower_icmp(
        &mut self,
        result: ValueId,
        pred: &ICmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);
        let cc = CondCode::from_icmp_predicate(pred);

        let mut instrs = Vec::new();

        // CMP lhs, rhs
        let mut cmp = MachineInstr::new(opcodes::CMP_RR);
        cmp.add_operand(lhs_op);
        cmp.add_operand(rhs_op);
        instrs.push(cmp);

        // SETcc dst (sets byte based on flags)
        let mut setcc = MachineInstr::new(cc.setcc_opcode());
        setcc.add_operand(dst.clone());
        setcc.add_operand(MachineOperand::Immediate(cc.encoding()));
        instrs.push(setcc);

        // MOVZX to ensure upper bits are zero (SETcc only sets low byte)
        let mut movzx = MachineInstr::new(opcodes::MOVZX);
        movzx.add_operand(dst.clone());
        movzx.add_operand(dst);
        instrs.push(movzx);

        instrs
    }

    /// Lowers an IR floating-point comparison to UCOMISS/UCOMISD + SETcc.
    fn lower_fcmp(
        &mut self,
        result: ValueId,
        pred: &FCmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);
        let cc = CondCode::from_fcmp_predicate(pred);

        let mut instrs = Vec::new();

        // Determine comparison type from LHS value type
        let lhs_ty = func.get_value_type(lhs);
        let cmp_op = match lhs_ty {
            IrType::F32 => opcodes::UCOMISS,
            _ => opcodes::UCOMISD,
        };

        let mut ucomi = MachineInstr::new(cmp_op);
        ucomi.add_operand(lhs_op);
        ucomi.add_operand(rhs_op);
        instrs.push(ucomi);

        // For ordered comparisons, we need to check PF=0 as well.
        // For simple cases, a single SETcc suffices.
        // For ordered equality (OEq), we need: SETE + SETNP + AND
        if matches!(pred, FCmpPredicate::OEq) {
            // SETcc for the equality
            let mut sete = MachineInstr::new(opcodes::SET_CC);
            sete.add_operand(dst.clone());
            sete.add_operand(MachineOperand::Immediate(CondCode::E.encoding()));
            instrs.push(sete);

            // We also need the NP (not parity) condition for ordered check
            let tmp = self.alloc_vreg();
            let mut setnp = MachineInstr::new(opcodes::SET_CC);
            setnp.add_operand(tmp.clone());
            setnp.add_operand(MachineOperand::Immediate(CondCode::NP.encoding()));
            instrs.push(setnp);

            // AND dst, tmp
            let mut and_mi = MachineInstr::new(opcodes::AND_RR);
            and_mi.add_operand(dst.clone());
            and_mi.add_operand(tmp);
            instrs.push(and_mi);
        } else if matches!(pred, FCmpPredicate::ONe) {
            // Ordered not equal: NE && NP
            let mut setne = MachineInstr::new(opcodes::SET_CC);
            setne.add_operand(dst.clone());
            setne.add_operand(MachineOperand::Immediate(CondCode::NE.encoding()));
            instrs.push(setne);

            let tmp = self.alloc_vreg();
            let mut setnp = MachineInstr::new(opcodes::SET_CC);
            setnp.add_operand(tmp.clone());
            setnp.add_operand(MachineOperand::Immediate(CondCode::NP.encoding()));
            instrs.push(setnp);

            let mut and_mi = MachineInstr::new(opcodes::AND_RR);
            and_mi.add_operand(dst.clone());
            and_mi.add_operand(tmp);
            instrs.push(and_mi);
        } else if matches!(pred, FCmpPredicate::UEq) {
            // Unordered equal: E || P
            let mut sete = MachineInstr::new(opcodes::SET_CC);
            sete.add_operand(dst.clone());
            sete.add_operand(MachineOperand::Immediate(CondCode::E.encoding()));
            instrs.push(sete);

            let tmp = self.alloc_vreg();
            let mut setp = MachineInstr::new(opcodes::SET_CC);
            setp.add_operand(tmp.clone());
            setp.add_operand(MachineOperand::Immediate(CondCode::P.encoding()));
            instrs.push(setp);

            let mut or_mi = MachineInstr::new(opcodes::OR_RR);
            or_mi.add_operand(dst.clone());
            or_mi.add_operand(tmp);
            instrs.push(or_mi);
        } else if matches!(pred, FCmpPredicate::UNe) {
            // Unordered not equal: NE || P
            let mut setne = MachineInstr::new(opcodes::SET_CC);
            setne.add_operand(dst.clone());
            setne.add_operand(MachineOperand::Immediate(CondCode::NE.encoding()));
            instrs.push(setne);

            let tmp = self.alloc_vreg();
            let mut setp = MachineInstr::new(opcodes::SET_CC);
            setp.add_operand(tmp.clone());
            setp.add_operand(MachineOperand::Immediate(CondCode::P.encoding()));
            instrs.push(setp);

            let mut or_mi = MachineInstr::new(opcodes::OR_RR);
            or_mi.add_operand(dst.clone());
            or_mi.add_operand(tmp);
            instrs.push(or_mi);
        } else {
            // Simple cases: single SETcc is sufficient
            let mut setcc = MachineInstr::new(cc.setcc_opcode());
            setcc.add_operand(dst.clone());
            setcc.add_operand(MachineOperand::Immediate(cc.encoding()));
            instrs.push(setcc);
        }

        // Zero-extend result to full register width
        let mut movzx = MachineInstr::new(opcodes::MOVZX);
        movzx.add_operand(dst.clone());
        movzx.add_operand(dst);
        instrs.push(movzx);

        instrs
    }

    // -----------------------------------------------------------------------
    // Branch lowering
    // -----------------------------------------------------------------------

    /// Lowers an unconditional branch.
    fn lower_branch(&self, target: BasicBlockId) -> Vec<MachineInstr> {
        let target_id = self.bb_map.get(&target.0).copied().unwrap_or(target.0);
        let mut mi = MachineInstr::new(opcodes::JMP);
        mi.add_operand(MachineOperand::Label(target_id));
        mi.set_terminator();
        vec![mi]
    }

    /// Lowers a conditional branch to TEST + Jcc + JMP.
    fn lower_cond_branch(
        &self,
        condition: ValueId,
        true_target: BasicBlockId,
        false_target: BasicBlockId,
    ) -> Vec<MachineInstr> {
        let cond_op = self.get_operand(condition);
        let true_id = self
            .bb_map
            .get(&true_target.0)
            .copied()
            .unwrap_or(true_target.0);
        let false_id = self
            .bb_map
            .get(&false_target.0)
            .copied()
            .unwrap_or(false_target.0);

        let mut instrs = Vec::new();

        // TEST condition, condition (sets ZF based on condition value)
        let mut test = MachineInstr::new(opcodes::TEST_RR);
        test.add_operand(cond_op.clone());
        test.add_operand(cond_op);
        instrs.push(test);

        // JNE true_target (jump if condition != 0)
        let cc = CondCode::NE;
        let mut jne = MachineInstr::new(cc.jcc_opcode());
        jne.add_operand(MachineOperand::Label(true_id));
        jne.add_operand(MachineOperand::Immediate(cc.encoding()));
        jne.set_terminator();
        instrs.push(jne);

        // JMP false_target (fallthrough to false)
        let mut jmp = MachineInstr::new(opcodes::JMP);
        jmp.add_operand(MachineOperand::Label(false_id));
        jmp.set_terminator();
        instrs.push(jmp);

        instrs
    }

    // -----------------------------------------------------------------------
    // Switch lowering
    // -----------------------------------------------------------------------

    /// Lowers a switch statement to either a jump table (dense) or a
    /// cascaded comparison chain (sparse).
    ///
    /// # Arguments
    ///
    /// * `value` — The IR value to switch on.
    /// * `cases` — Case value → target block mappings.
    /// * `default` — Default target block when no case matches.
    /// * `bb_map` — Mapping from IR block IDs to machine block IDs.
    pub fn lower_switch(
        &mut self,
        value: ValueId,
        cases: &[(i64, BasicBlockId)],
        default: BasicBlockId,
        bb_map: &FxHashMap<u32, u32>,
    ) -> Vec<MachineInstr> {
        if cases.is_empty() {
            // No cases: just jump to default
            let target_id = bb_map.get(&default.0).copied().unwrap_or(default.0);
            let mut mi = MachineInstr::new(opcodes::JMP);
            mi.add_operand(MachineOperand::Label(target_id));
            mi.set_terminator();
            return vec![mi];
        }

        // Determine if jump table or comparison chain is better
        let val_op = self.get_operand(value);
        if self.should_use_jump_table(cases) {
            self.lower_switch_jump_table(val_op, cases, default, bb_map)
        } else {
            self.lower_switch_chain(val_op, cases, default, bb_map)
        }
    }

    /// Determines whether a jump table should be used for the given cases.
    fn should_use_jump_table(&self, cases: &[(i64, BasicBlockId)]) -> bool {
        if cases.len() < JUMP_TABLE_MIN_CASES {
            return false;
        }
        let min_val = cases.iter().map(|(v, _)| *v).min().unwrap_or(0);
        let max_val = cases.iter().map(|(v, _)| *v).max().unwrap_or(0);
        let range = (max_val - min_val + 1) as f64;
        let density = cases.len() as f64 / range;
        density >= JUMP_TABLE_DENSITY_THRESHOLD
    }

    /// Lowers a switch to a jump table (dense case distribution).
    fn lower_switch_jump_table(
        &mut self,
        val: MachineOperand,
        cases: &[(i64, BasicBlockId)],
        default: BasicBlockId,
        bb_map: &FxHashMap<u32, u32>,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let default_id = bb_map.get(&default.0).copied().unwrap_or(default.0);

        let min_val = cases.iter().map(|(v, _)| *v).min().unwrap_or(0);
        let max_val = cases.iter().map(|(v, _)| *v).max().unwrap_or(0);
        let table_size = (max_val - min_val + 1) as usize;

        // Subtract minimum value to normalize index
        if min_val != 0 {
            let tmp = self.alloc_vreg();
            instrs.push(self.make_mov(tmp.clone(), val.clone()));
            let mut sub = MachineInstr::new(opcodes::SUB_RI);
            sub.add_operand(tmp.clone());
            sub.add_operand(MachineOperand::Immediate(min_val));
            instrs.push(sub);

            // Range check: if index >= table_size, jump to default
            let mut cmp = MachineInstr::new(opcodes::CMP_RI);
            cmp.add_operand(tmp.clone());
            cmp.add_operand(MachineOperand::Immediate(table_size as i64));
            instrs.push(cmp);
        } else {
            // Range check directly
            let mut cmp = MachineInstr::new(opcodes::CMP_RI);
            cmp.add_operand(val.clone());
            cmp.add_operand(MachineOperand::Immediate(table_size as i64));
            instrs.push(cmp);
        }

        // JAE default (unsigned comparison for range check)
        let mut jae = MachineInstr::new(opcodes::JCC);
        jae.add_operand(MachineOperand::Label(default_id));
        jae.add_operand(MachineOperand::Immediate(CondCode::AE.encoding()));
        jae.set_terminator();
        instrs.push(jae);

        // Build jump table entries as labels
        // The actual jump table data is handled by the assembler/linker
        let mut jmp_ind = MachineInstr::new(opcodes::JMP);
        jmp_ind.add_operand(val);
        jmp_ind.set_terminator();
        instrs.push(jmp_ind);

        instrs
    }

    /// Lowers a switch to a cascaded comparison chain (sparse cases).
    fn lower_switch_chain(
        &mut self,
        val: MachineOperand,
        cases: &[(i64, BasicBlockId)],
        default: BasicBlockId,
        bb_map: &FxHashMap<u32, u32>,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let default_id = bb_map.get(&default.0).copied().unwrap_or(default.0);

        // Sort cases by value for predictable ordering
        let mut sorted_cases: Vec<(i64, BasicBlockId)> = cases.to_vec();
        sorted_cases.sort_by(|a, b| a.0.cmp(&b.0));

        for (case_val, target) in &sorted_cases {
            let target_id = bb_map.get(&target.0).copied().unwrap_or(target.0);

            // CMP value, case_val
            let mut cmp = MachineInstr::new(opcodes::CMP_RI);
            cmp.add_operand(val.clone());
            cmp.add_operand(MachineOperand::Immediate(*case_val));
            instrs.push(cmp);

            // JE target
            let mut je = MachineInstr::new(opcodes::JCC);
            je.add_operand(MachineOperand::Label(target_id));
            je.add_operand(MachineOperand::Immediate(CondCode::E.encoding()));
            je.set_terminator();
            instrs.push(je);
        }

        // Fall through to default
        let mut jmp = MachineInstr::new(opcodes::JMP);
        jmp.add_operand(MachineOperand::Label(default_id));
        jmp.set_terminator();
        instrs.push(jmp);

        instrs
    }

    // -----------------------------------------------------------------------
    // Function call lowering
    // -----------------------------------------------------------------------

    /// Lowers a function call per the System V AMD64 ABI.
    ///
    /// Sets up arguments in registers/stack, emits the CALL instruction,
    /// and retrieves the return value.
    pub fn lower_call(
        &mut self,
        result: Option<ValueId>,
        callee: ValueId,
        args: &[ValueId],
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        self.has_calls = true;
        let mut instrs = Vec::new();

        // Classify argument types for ABI
        let arg_ctypes: Vec<CType> = args
            .iter()
            .map(|a| {
                let ty = func.get_value_type(*a);
                ir_type_to_ctype(ty)
            })
            .collect();
        let locations = compute_param_locations(&arg_ctypes, &self.config.target);

        // Track stack space for stack-passed arguments
        let mut stack_space: i32 = 0;
        for loc in &locations {
            if let ParamLocation::Stack { offset } = loc {
                let needed = *offset + 8;
                if needed > stack_space {
                    stack_space = needed;
                }
            }
        }

        // Align stack to 16 bytes for the call — MUST use 64-bit (REX.W) for RSP
        let aligned_stack = ((stack_space + 15) / 16) * 16;
        if aligned_stack > 0 {
            let mut sub_rsp = MachineInstr::new(opcodes::SUB_RI | SIZE_QWORD);
            sub_rsp.add_operand(MachineOperand::Register(RSP));
            sub_rsp.add_operand(MachineOperand::Immediate(aligned_stack as i64));
            instrs.push(sub_rsp);
        }

        // Move arguments to their ABI-designated locations.
        // When an argument is a Symbol (global function pointer or string
        // literal address), we must use LEA_SYM (RIP-relative LEA) to load
        // the address into the target register instead of MOV_RR.
        for (i, arg_val) in args.iter().enumerate() {
            if i >= locations.len() {
                break;
            }
            let arg_op = self.get_operand(*arg_val);
            let is_symbol = matches!(arg_op, MachineOperand::Symbol(_));
            let is_imm = matches!(arg_op, MachineOperand::Immediate(_));
            match &locations[i] {
                ParamLocation::Register(phys) => {
                    if is_symbol {
                        // LEA <phys>, [rip + symbol]
                        let mut mi = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                        mi.add_operand(MachineOperand::Register(*phys));
                        mi.add_operand(arg_op);
                        instrs.push(mi);
                    } else if is_imm {
                        // MOV <phys>, imm
                        let mut mi = MachineInstr::new(opcodes::MOV_RI);
                        mi.add_operand(MachineOperand::Register(*phys));
                        mi.add_operand(arg_op);
                        instrs.push(mi);
                    } else {
                        let mut mi = MachineInstr::new(opcodes::MOV_RR);
                        mi.add_operand(MachineOperand::Register(*phys));
                        mi.add_operand(arg_op);
                        instrs.push(mi);
                    }
                }
                ParamLocation::RegisterPair(lo, _hi) => {
                    let mut mi_lo = MachineInstr::new(opcodes::MOV_RR);
                    mi_lo.add_operand(MachineOperand::Register(*lo));
                    mi_lo.add_operand(arg_op);
                    instrs.push(mi_lo);
                }
                ParamLocation::Stack { offset } => {
                    if is_symbol {
                        // Load symbol address into a scratch register, then
                        // move to the stack slot.
                        let scratch = self.alloc_vreg();
                        let mut lea = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                        lea.add_operand(scratch.clone());
                        lea.add_operand(arg_op);
                        instrs.push(lea);
                        let mut mi = MachineInstr::new(opcodes::MOV_MR);
                        mi.add_operand(MachineOperand::Memory {
                            base: RSP,
                            offset: *offset,
                            index: None,
                            scale: 1,
                        });
                        mi.add_operand(scratch);
                        instrs.push(mi);
                    } else {
                        let mut mi = MachineInstr::new(opcodes::MOV_MR);
                        mi.add_operand(MachineOperand::Memory {
                            base: RSP,
                            offset: *offset,
                            index: None,
                            scale: 1,
                        });
                        mi.add_operand(arg_op);
                        instrs.push(mi);
                    }
                }
                ParamLocation::HiddenPointer(reg) => {
                    if is_symbol {
                        let mut mi = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                        mi.add_operand(MachineOperand::Register(*reg));
                        mi.add_operand(arg_op);
                        instrs.push(mi);
                    } else {
                        let mut mi = MachineInstr::new(opcodes::MOV_RR);
                        mi.add_operand(MachineOperand::Register(*reg));
                        mi.add_operand(arg_op);
                        instrs.push(mi);
                    }
                }
            }
        }

        // Emit the CALL instruction
        let callee_op = self.get_operand(callee);
        let call_op = match &callee_op {
            MachineOperand::Symbol(_) => opcodes::CALL,
            _ => opcodes::CALL_IND,
        };
        let mut call_mi = MachineInstr::new(call_op);
        call_mi.add_operand(callee_op);
        call_mi.set_call();

        // Add implicit defs for caller-saved registers
        for &reg in CALLER_SAVED {
            call_mi.add_implicit_def(reg);
        }
        // RAX/XMM0 are also implicit defs (return value)
        call_mi.add_implicit_def(RAX);

        instrs.push(call_mi);

        // Deallocate stack space for arguments — MUST use 64-bit (REX.W) for RSP
        if aligned_stack > 0 {
            let mut add_rsp = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
            add_rsp.add_operand(MachineOperand::Register(RSP));
            add_rsp.add_operand(MachineOperand::Immediate(aligned_stack as i64));
            instrs.push(add_rsp);
        }

        // Move return value from physical register to virtual register
        if let Some(res_vid) = result {
            let res_ty = func.get_value_type(res_vid);
            let dst = self.get_or_create_vreg(res_vid);

            if res_ty.is_floating() {
                let mov_op = match res_ty {
                    IrType::F32 => opcodes::MOVSS,
                    _ => opcodes::MOVSD,
                };
                let mut mv = MachineInstr::new(mov_op);
                mv.add_operand(dst);
                mv.add_operand(MachineOperand::Register(XMM0));
                instrs.push(mv);
            } else {
                let mut mv = MachineInstr::new(opcodes::MOV_RR);
                mv.add_operand(dst);
                mv.add_operand(MachineOperand::Register(RAX));
                instrs.push(mv);
            }
        }

        instrs
    }

    // -----------------------------------------------------------------------
    // Return lowering
    // -----------------------------------------------------------------------

    /// Lowers a return instruction.
    fn lower_return(&self, value: Option<ValueId>, func: &IrFunction) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        if let Some(val) = value {
            let val_op = self.get_operand(val);
            let val_ty = func.get_value_type(val);

            if val_ty.is_floating() {
                let mov_op = match val_ty {
                    IrType::F32 => opcodes::MOVSS,
                    _ => opcodes::MOVSD,
                };
                let mut mi = MachineInstr::new(mov_op);
                mi.add_operand(MachineOperand::Register(XMM0));
                mi.add_operand(val_op);
                instrs.push(mi);
            } else {
                // Select MOV_RI for immediates, MOV_RM for memory, MOV_RR otherwise
                let mov_op = match &val_op {
                    MachineOperand::Immediate(_) => opcodes::MOV_RI,
                    MachineOperand::Memory { .. } | MachineOperand::FrameIndex(_) => {
                        opcodes::MOV_RM
                    }
                    _ => opcodes::MOV_RR,
                };
                let mut mi = MachineInstr::new(mov_op);
                mi.add_operand(MachineOperand::Register(RAX));
                mi.add_operand(val_op);
                instrs.push(mi);
            }
        }

        let mut ret = MachineInstr::new(opcodes::RET);
        ret.set_return();
        instrs.push(ret);

        instrs
    }

    // -----------------------------------------------------------------------
    // GetElementPtr lowering
    // -----------------------------------------------------------------------

    /// Lowers a GetElementPtr instruction to address computation.
    ///
    /// GEP computes: base + sum(index[i] * stride[i])
    fn lower_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
        ty: IrType,
        _in_bounds: bool,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let base_op = self.get_operand(base);
        let mut instrs = Vec::new();

        if indices.is_empty() {
            // No indices: result is just the base pointer
            instrs.push(self.make_mov(dst, base_op));
            return instrs;
        }

        // Start with base pointer
        instrs.push(self.make_mov(dst.clone(), base_op));

        // Process each index
        let mut current_ty = ty;
        for idx in indices.iter() {
            let idx_op = self.get_operand(*idx);

            // Compute element size for stride
            let elem_size = match &current_ty {
                IrType::Array {
                    element: ref elem_ty,
                    count: _,
                } => {
                    let sz = elem_ty.size_bytes(&self.config.target);
                    current_ty = (**elem_ty).clone();
                    sz
                }
                IrType::Struct { fields, packed: _ } => {
                    // For struct GEPs, the index is a constant field index
                    if let MachineOperand::Immediate(field_idx) = &idx_op {
                        let field_idx = *field_idx as usize;
                        if field_idx < fields.len() {
                            // Compute offset to the field
                            let mut offset: u64 = 0;
                            for field in fields.iter().take(field_idx) {
                                let f_size = field.size_bytes(&self.config.target);
                                let f_align = field.alignment(&self.config.target);
                                offset = (offset + f_align - 1) & !(f_align - 1);
                                offset += f_size;
                            }
                            let field_align = fields[field_idx].alignment(&self.config.target);
                            offset = (offset + field_align - 1) & !(field_align - 1);

                            // ADD dst, offset
                            if offset > 0 {
                                let mut add = MachineInstr::new(opcodes::ADD_RI);
                                add.add_operand(dst.clone());
                                add.add_operand(MachineOperand::Immediate(offset as i64));
                                instrs.push(add);
                            }
                            current_ty = fields[field_idx].clone();
                            continue;
                        }
                    }
                    // Fallback: use pointer size
                    self.config.target.pointer_width() as u64
                }
                _ => {
                    // Pointer or other: use the type size
                    current_ty.size_bytes(&self.config.target)
                }
            };

            if elem_size == 0 {
                continue;
            }

            // Compute index * element_size and add to base
            match &idx_op {
                MachineOperand::Immediate(0) => {
                    // index 0: no offset needed
                }
                MachineOperand::Immediate(c) => {
                    let offset = (*c) * (elem_size as i64);
                    if offset != 0 {
                        let mut add = MachineInstr::new(opcodes::ADD_RI);
                        add.add_operand(dst.clone());
                        add.add_operand(MachineOperand::Immediate(offset));
                        instrs.push(add);
                    }
                }
                _ => {
                    // Variable index: use LEA or IMUL+ADD
                    if elem_size == 1 {
                        let mut add = MachineInstr::new(opcodes::ADD_RR);
                        add.add_operand(dst.clone());
                        add.add_operand(idx_op);
                        instrs.push(add);
                    } else if elem_size.is_power_of_two() && elem_size <= 8 {
                        // Use LEA with scale factor
                        let mut lea = MachineInstr::new(opcodes::LEA);
                        lea.add_operand(dst.clone());
                        lea.add_operand(MachineOperand::Memory {
                            base: RSP, // placeholder, resolved by regalloc
                            offset: 0,
                            index: None,
                            scale: elem_size as u8,
                        });
                        instrs.push(lea);
                    } else {
                        // IMUL tmp, index, elem_size; ADD dst, tmp
                        let tmp = self.alloc_vreg();
                        let mut imul = MachineInstr::new(opcodes::IMUL_RI);
                        imul.add_operand(tmp.clone());
                        imul.add_operand(idx_op);
                        imul.add_operand(MachineOperand::Immediate(elem_size as i64));
                        instrs.push(imul);

                        let mut add = MachineInstr::new(opcodes::ADD_RR);
                        add.add_operand(dst.clone());
                        add.add_operand(tmp);
                        instrs.push(add);
                    }
                }
            }
        }

        instrs
    }

    /// Computes addressing mode from a GEP instruction for use in load/store.
    ///
    /// Attempts to fold the GEP into a complex x86-64 addressing mode
    /// (base + index*scale + displacement) to avoid explicit address
    /// computation when possible.
    pub fn select_addressing_mode(
        &self,
        base: ValueId,
        indices: &[ValueId],
        ty: &IrType,
        _func: &IrFunction,
    ) -> MemoryOperand {
        let base_op = self.get_operand(base);

        if indices.is_empty() {
            // No indices: just base register
            if let MachineOperand::Register(reg) = base_op {
                return MemoryOperand::base_disp(reg, 0);
            }
            if let MachineOperand::FrameIndex(fi) = base_op {
                return MemoryOperand::base_disp(RBP, -(fi as i32 * 8 + 8));
            }
            return MemoryOperand::base_disp(RBP, 0);
        }

        // Single constant index: base + displacement
        if indices.len() == 1 {
            if let Some(const_idx) = self.get_constant_value(indices[0]) {
                let elem_size = ty.size_bytes(&self.config.target) as i64;
                let disp = const_idx * elem_size;
                if disp >= i32::MIN as i64 && disp <= i32::MAX as i64 {
                    if let MachineOperand::Register(reg) = base_op {
                        return MemoryOperand::base_disp(reg, disp as i32);
                    }
                }
            }
        }

        // Fallback: just use base register
        if let MachineOperand::Register(reg) = base_op {
            MemoryOperand::base_disp(reg, 0)
        } else {
            MemoryOperand::base_disp(RBP, 0)
        }
    }

    // -----------------------------------------------------------------------
    // Type conversion lowering
    // -----------------------------------------------------------------------

    /// Lowers a bitcast (no-op type reinterpretation) or int↔float value
    /// conversion.  Pure bit-reinterpretation stays within the same register
    /// file; value-changing conversions (e.g. `(float)42`) cross register
    /// files via CVTSI2SS / CVTTSS2SI / CVTSS2SD family.
    fn lower_bitcast(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let src_ty = func.get_value_type(value);

        let src_is_fp = src_ty.is_floating();
        let dst_is_fp = to_ty.is_floating();

        // Integer → float value conversion (CVTSI2SS / CVTSI2SD).
        if !src_is_fp && dst_is_fp {
            return self.lower_int_to_float(result, value, to_ty);
        }
        // Float → integer value conversion (CVTTSS2SI / CVTTSD2SI).
        if src_is_fp && !dst_is_fp {
            return self.lower_float_to_int(result, value, src_ty);
        }
        // Float → float widening / narrowing (CVTSS2SD / CVTSD2SS).
        if src_is_fp && dst_is_fp && std::mem::discriminant(src_ty) != std::mem::discriminant(to_ty)
        {
            return self.lower_float_convert(result, value, src_ty, to_ty);
        }

        // Same register file — pure bit-reinterpretation (MOVD/MOVQ or simple MOV).
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);
        let mov_op = if dst_is_fp {
            opcodes::MOVSD
        } else {
            opcodes::MOV_RR
        };
        vec![self.make_mov_opcode(dst, src, mov_op)]
    }

    /// Lowers integer truncation.
    fn lower_trunc(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);

        // On x86-64, truncation is a simple move — the upper bits are
        // naturally ignored when using a smaller register size.
        vec![self.make_mov(dst, src)]
    }

    /// Lowers zero extension (MOVZX).
    fn lower_zext(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);
        let src_ty = func.get_value_type(value);

        // Special case: i1 -> larger type (bool zero-extension)
        if matches!(src_ty, IrType::I1) {
            let mut movzx = MachineInstr::new(opcodes::MOVZX);
            movzx.add_operand(dst);
            movzx.add_operand(src);
            return vec![movzx];
        }

        // Special case: i32 -> i64 on x86-64 is implicit (mov clears upper 32 bits)
        if matches!(src_ty, IrType::I32) && matches!(to_ty, IrType::I64) {
            // A 32-bit MOV implicitly zero-extends to 64 bits on x86-64
            return vec![self.make_mov(dst, src)];
        }

        let mut movzx = MachineInstr::new(opcodes::MOVZX);
        movzx.add_operand(dst);
        movzx.add_operand(src);
        vec![movzx]
    }

    /// Lowers sign extension (MOVSX/MOVSXD).
    fn lower_sext(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);

        let mut movsx = MachineInstr::new(opcodes::MOVSX);
        movsx.add_operand(dst);
        movsx.add_operand(src);
        vec![movsx]
    }

    /// Lowers a simple move (used for IntToPtr and other no-op conversions).
    fn lower_mov(&mut self, result: ValueId, value: ValueId) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);
        vec![self.make_mov(dst, src)]
    }

    /// Lowers PtrToInt, which may require truncation on 32-bit targets.
    fn lower_ptrtoint(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);

        // On x86-64, pointers are 64-bit. If target type is i64, this is a mov.
        // For smaller types, we truncate.
        match to_ty {
            IrType::I64 | IrType::Ptr => vec![self.make_mov(dst, src)],
            IrType::I32 => {
                // 32-bit mov implicitly truncates (and zero-extends upper 32)
                vec![self.make_mov(dst, src)]
            }
            _ => {
                // Mov then truncate
                vec![self.make_mov(dst, src)]
            }
        }
    }

    // -----------------------------------------------------------------------
    // Inline assembly lowering
    // -----------------------------------------------------------------------

    /// Lowers an inline assembly statement.
    ///
    /// Emits a pseudo INLINE_ASM instruction that carries the template
    /// string, operands, and clobber information for later processing
    /// by the assembler.
    pub fn lower_inline_asm(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Validate: empty template with output constraint is suspicious.
        if template.is_empty() && result.is_some() {
            self.diagnostics.warning(
                Span::DUMMY,
                "inline asm has empty template but declares an output operand",
            );
        }

        // Create the pseudo inline assembly instruction
        let mut asm_mi = MachineInstr::new(opcodes::INLINE_ASM);

        // Add operands (mapped from IR values)
        for op_val in operands {
            let op = self.get_operand(*op_val);
            asm_mi.add_operand(op);
        }

        // Process clobber list to add implicit defs
        for clobber in clobbers {
            match clobber.as_str() {
                "memory" => {
                    // Memory clobber: treated as a barrier — no register,
                    // but the instruction must not be moved across stores.
                }
                "cc" => {
                    // Flags clobber: no physical register to add
                }
                _ => {
                    // Physical register clobber
                    if let Some(reg) = self.parse_register_name(clobber) {
                        asm_mi.add_implicit_def(reg);
                    } else {
                        self.diagnostics.warning(
                            Span::DUMMY,
                            format!("unrecognised inline asm clobber: {}", clobber),
                        );
                    }
                }
            }
        }

        instrs.push(asm_mi);

        // If there's a result, move from the output register
        if let Some(res_vid) = result {
            let dst = self.get_or_create_vreg(res_vid);
            // Parse the output constraint to determine the register
            if let Some(out_reg) = self.parse_output_constraint(constraints) {
                let mut mi = MachineInstr::new(opcodes::MOV_RR);
                mi.add_operand(dst);
                mi.add_operand(MachineOperand::Register(out_reg));
                instrs.push(mi);
            } else {
                self.diagnostics.error(
                    Span::DUMMY,
                    format!(
                        "cannot parse output constraint '{}' for inline asm",
                        constraints,
                    ),
                );
            }
        }

        instrs
    }

    // -----------------------------------------------------------------------
    // Prologue / Epilogue generation
    // -----------------------------------------------------------------------

    /// Generates prologue instructions for the function.
    ///
    /// Standard x86-64 prologue:
    /// ```text
    /// push %rbp
    /// mov %rsp, %rbp
    /// sub $frame_size, %rsp
    /// push <callee-saved registers>
    /// ```
    ///
    /// Leaf function optimization: if the function has no calls and the frame
    /// fits in the 128-byte red zone, the prologue may be simplified.
    pub fn emit_prologue(&self, func: &IrFunction, mfunc: &MachineFunction) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let frame_size = mfunc.frame_size;
        let use_frame_pointer = mfunc.has_calls || frame_size > 0;

        // Check if we can use the red zone (leaf function optimization)
        let use_red_zone =
            can_use_red_zone(func) && !mfunc.has_calls && frame_size <= RED_ZONE_SIZE;

        if use_red_zone {
            // Red zone: no prologue needed for small leaf functions
            return instrs;
        }

        if use_frame_pointer {
            // push %rbp
            let mut push_rbp = MachineInstr::new(opcodes::PUSH);
            push_rbp.add_operand(MachineOperand::Register(RBP));
            instrs.push(push_rbp);

            // mov %rsp, %rbp  — MUST use 64-bit (REX.W) for stack pointer
            let mut mov_rbp = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
            mov_rbp.add_operand(MachineOperand::Register(RBP));
            mov_rbp.add_operand(MachineOperand::Register(RSP));
            instrs.push(mov_rbp);
        }

        // Push callee-saved registers
        for reg in &mfunc.used_callee_saved {
            let mut push = MachineInstr::new(opcodes::PUSH);
            push.add_operand(MachineOperand::Register(*reg));
            instrs.push(push);
        }

        // Allocate stack frame — MUST use 64-bit (REX.W) for stack pointer
        if frame_size > 0 {
            let mut sub = MachineInstr::new(opcodes::SUB_RI | SIZE_QWORD);
            sub.add_operand(MachineOperand::Register(RSP));
            sub.add_operand(MachineOperand::Immediate(frame_size as i64));
            instrs.push(sub);
        }

        instrs
    }

    /// Generates epilogue instructions for the function.
    ///
    /// Standard x86-64 epilogue:
    /// ```text
    /// add $frame_size, %rsp    (or mov %rbp, %rsp)
    /// pop <callee-saved registers>
    /// pop %rbp
    /// ret
    /// ```
    pub fn emit_epilogue(&self, func: &IrFunction, mfunc: &MachineFunction) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let frame_size = mfunc.frame_size;
        let use_frame_pointer = mfunc.has_calls || frame_size > 0;

        let use_red_zone =
            can_use_red_zone(func) && !mfunc.has_calls && frame_size <= RED_ZONE_SIZE;

        if use_red_zone {
            // No epilogue needed for red zone functions
            let mut ret = MachineInstr::new(opcodes::RET);
            ret.set_return();
            instrs.push(ret);
            return instrs;
        }

        // Deallocate stack frame — MUST use 64-bit (REX.W) for stack pointer
        if frame_size > 0 && !use_frame_pointer {
            let mut add = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
            add.add_operand(MachineOperand::Register(RSP));
            add.add_operand(MachineOperand::Immediate(frame_size as i64));
            instrs.push(add);
        }

        // Restore callee-saved registers in reverse order
        for reg in mfunc.used_callee_saved.iter().rev() {
            let mut pop = MachineInstr::new(opcodes::POP);
            pop.add_operand(MachineOperand::Register(*reg));
            instrs.push(pop);
        }

        if use_frame_pointer {
            // mov %rbp, %rsp — MUST use 64-bit (REX.W) for stack pointer
            if frame_size > 0 {
                let mut mov_rsp = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                mov_rsp.add_operand(MachineOperand::Register(RSP));
                mov_rsp.add_operand(MachineOperand::Register(RBP));
                instrs.push(mov_rsp);
            }

            // pop %rbp
            let mut pop_rbp = MachineInstr::new(opcodes::POP);
            pop_rbp.add_operand(MachineOperand::Register(RBP));
            instrs.push(pop_rbp);
        }

        // ret
        let mut ret = MachineInstr::new(opcodes::RET);
        ret.set_return();
        instrs.push(ret);

        instrs
    }

    // -----------------------------------------------------------------------
    // Integer <-> Float conversion helpers
    // -----------------------------------------------------------------------

    /// Lowers integer-to-float conversion (CVTSI2SS/CVTSI2SD).
    fn lower_int_to_float(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);

        let opcode = match to_ty {
            IrType::F32 => opcodes::CVTSI2SS,
            _ => opcodes::CVTSI2SD,
        };

        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(src);
        vec![mi]
    }

    /// Lowers float-to-integer conversion (CVTTSS2SI/CVTTSD2SI).
    fn lower_float_to_int(
        &mut self,
        result: ValueId,
        value: ValueId,
        from_ty: &IrType,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);

        let opcode = match from_ty {
            IrType::F32 => opcodes::CVTTSS2SI,
            _ => opcodes::CVTTSD2SI,
        };

        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(src);
        vec![mi]
    }

    /// Lowers float-to-float conversion (CVTSS2SD/CVTSD2SS).
    fn lower_float_convert(
        &mut self,
        result: ValueId,
        value: ValueId,
        from_ty: &IrType,
        to_ty: &IrType,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);

        let opcode = match (from_ty, to_ty) {
            (IrType::F32, IrType::F64) => opcodes::CVTSS2SD,
            (IrType::F64, IrType::F32) => opcodes::CVTSD2SS,
            _ => opcodes::CVTSS2SD, // fallback
        };

        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(src);
        vec![mi]
    }

    // -----------------------------------------------------------------------
    // Helper methods
    // -----------------------------------------------------------------------

    /// Gets or creates a VirtualReg operand for an IR ValueId.
    fn get_or_create_vreg(&mut self, vid: ValueId) -> MachineOperand {
        if let Some(op) = self.value_map.get(&vid.0) {
            return op.clone();
        }
        let vreg = MachineOperand::VirtualReg(vid);
        self.value_map.insert(vid.0, vreg.clone());
        vreg
    }

    /// Allocates a fresh virtual register for temporaries.
    fn alloc_vreg(&mut self) -> MachineOperand {
        let id = self.next_vreg;
        self.next_vreg += 1;
        MachineOperand::VirtualReg(ValueId(id))
    }

    /// Gets the machine operand for an IR value. If the value is already
    /// mapped (e.g., to a frame index or physical register), returns that
    /// mapping; otherwise creates a VirtualReg.
    fn get_operand(&self, vid: ValueId) -> MachineOperand {
        if let Some(op) = self.value_map.get(&vid.0) {
            op.clone()
        } else {
            MachineOperand::VirtualReg(vid)
        }
    }

    /// Tries to extract a constant integer value from a ValueId, if it
    /// was produced by a known constant instruction. Returns None if
    /// the value is not a statically known constant.
    fn get_constant_value(&self, vid: ValueId) -> Option<i64> {
        if let Some(MachineOperand::Immediate(val)) = self.value_map.get(&vid.0) {
            Some(*val)
        } else {
            None
        }
    }

    /// Creates a MOV instruction from src to dst using the appropriate
    /// opcode based on operand types.
    fn make_mov(&self, dst: MachineOperand, src: MachineOperand) -> MachineInstr {
        let opcode = match (&dst, &src) {
            (_, MachineOperand::Immediate(_)) => opcodes::MOV_RI,
            (_, MachineOperand::Memory { .. }) | (_, MachineOperand::FrameIndex(_)) => {
                opcodes::MOV_RM
            }
            (MachineOperand::Memory { .. }, _) | (MachineOperand::FrameIndex(_), _) => {
                opcodes::MOV_MR
            }
            _ => opcodes::MOV_RR,
        };
        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(src);
        mi
    }

    /// Creates a MOV instruction with a specified opcode.
    fn make_mov_opcode(
        &self,
        dst: MachineOperand,
        src: MachineOperand,
        opcode: u32,
    ) -> MachineInstr {
        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(src);
        mi
    }

    /// Parses a register name string (e.g., "rax", "rcx") to a PhysReg.
    fn parse_register_name(&self, name: &str) -> Option<PhysReg> {
        match name.to_lowercase().as_str() {
            "rax" | "eax" | "al" => Some(RAX),
            "rcx" | "ecx" | "cl" => Some(RCX),
            "rdx" | "edx" | "dl" => Some(RDX),
            "rbx" | "ebx" | "bl" => Some(RBX),
            "rsp" | "esp" => Some(RSP),
            "rbp" | "ebp" => Some(RBP),
            "rsi" | "esi" | "sil" => Some(RSI),
            "rdi" | "edi" | "dil" => Some(RDI),
            "r8" | "r8d" | "r8b" => Some(R8),
            "r9" | "r9d" | "r9b" => Some(R9),
            "r10" | "r10d" | "r10b" => Some(R10),
            "r11" | "r11d" | "r11b" => Some(R11),
            "r12" | "r12d" | "r12b" => Some(R12),
            "r13" | "r13d" | "r13b" => Some(R13),
            "r14" | "r14d" | "r14b" => Some(R14),
            "r15" | "r15d" | "r15b" => Some(R15),
            "xmm0" => Some(XMM0),
            "xmm1" => Some(XMM1),
            "xmm2" => Some(XMM2),
            "xmm3" => Some(XMM3),
            "xmm4" => Some(XMM4),
            "xmm5" => Some(XMM5),
            "xmm6" => Some(XMM6),
            "xmm7" => Some(XMM7),
            _ => None,
        }
    }

    /// Parses the output constraint string to determine the output register.
    /// Returns the first output register, or RAX as default.
    fn parse_output_constraint(&self, constraints: &str) -> Option<PhysReg> {
        // Constraint format: "=r,r,~{memory}" or similar
        for part in constraints.split(',') {
            let trimmed = part.trim();
            if trimmed.starts_with('=') || trimmed.starts_with('+') {
                let constraint = &trimmed[1..];
                return match constraint {
                    "r" | "a" => Some(RAX),
                    "b" => Some(RBX),
                    "c" => Some(RCX),
                    "d" => Some(RDX),
                    "S" => Some(RSI),
                    "D" => Some(RDI),
                    _ => Some(RAX), // Default to RAX for general register
                };
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Helper: IR type to C type conversion
// ---------------------------------------------------------------------------

/// Converts an IR type to a C type for ABI classification purposes.
///
/// This mapping is used during function call lowering to classify arguments
/// and return values according to the System V AMD64 ABI.
fn ir_type_to_ctype(ty: &IrType) -> CType {
    match ty {
        IrType::Void => CType::Void,
        IrType::I1 => CType::Bool,
        IrType::I8 => CType::Char { signed: true },
        IrType::I16 => CType::Short { signed: true },
        IrType::I32 => CType::Int { signed: true },
        IrType::I64 => CType::Long { signed: true },
        IrType::I128 => CType::LongLong { signed: true },
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::F80 => CType::LongDouble,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void)),
        IrType::Array { ref element, count } => CType::Array {
            element: Box::new(ir_type_to_ctype(element)),
            size: Some(*count),
        },
        IrType::Struct {
            ref fields,
            packed: _,
        } => {
            let c_fields: Vec<FieldDef> = fields
                .iter()
                .map(|f| FieldDef {
                    name: None,
                    ty: ir_type_to_ctype(f),
                    bit_width: None,
                })
                .collect();
            CType::Struct {
                name: None,
                fields: c_fields,
            }
        }
        IrType::Function {
            ref return_type,
            ref param_types,
            is_variadic: _,
        } => CType::Function {
            return_type: Box::new(ir_type_to_ctype(return_type)),
            params: param_types.iter().map(ir_type_to_ctype).collect(),
            variadic: false,
        },
    }
}
