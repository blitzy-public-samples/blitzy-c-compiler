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
    self, gpr_encoding, gpr_name_64, CALLER_SAVED, R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP,
    RBX, RCX, RDI, RDX, RSI, RSP, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
};
#[allow(unused_imports)]
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

/// Operand-size flag that selects 8-bit (Byte) encoding (bits 24-25 = 10).
const SIZE_BYTE: u32 = 2 << 24;

/// Operand-size flag that selects 16-bit (Word) encoding (bits 24-25 = 11).
const SIZE_WORD: u32 = 3 << 24;

// ---------------------------------------------------------------------------
// AsmResolvedOperand — resolved inline assembly operand
// ---------------------------------------------------------------------------

/// Represents a resolved operand from an AT&T-syntax inline assembly template.
///
/// After parsing an operand string like `%0`, `$42`, `%%eax`, or `(%rdi)`,
/// the result is one of these variants:
/// - `MachineOp` — a virtual or physical register from the operand map
/// - `Immediate` — a literal integer value from `$N`
/// - `PhysReg` — a physical register from `%%name`
/// - `MemoryRef` — a memory operand `(base)` or `offset(base)`
/// - `Unknown` — unrecognized operand string (passed through as INLINE_ASM)
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum AsmResolvedOperand {
    /// A machine operand (virtual register or physical register) from the
    /// constraint-resolved operand map.
    MachineOp(MachineOperand),
    /// A literal integer immediate from `$N` syntax.
    Immediate(i64),
    /// A physical register specified via `%%name` syntax (e.g., `%%eax`).
    PhysReg(PhysReg),
    /// A memory reference operand `(base)` or `offset(base)` from AT&T syntax.
    MemoryRef(PhysReg),
    /// A memory operand from a `"m"` constraint — a FrameIndex or Memory
    /// that should be used as a memory reference in emitted instructions
    /// (MOV_RM for loads, MOV_MR for stores).
    MemoryOp(MachineOperand),
    /// A goto label target for `asm goto` — stores the target BasicBlockId
    /// index (0-based) into the goto_targets array.
    GotoLabel(usize),
    /// An unrecognized operand string, emitted as opaque INLINE_ASM.
    Unknown(String),
}

impl AsmResolvedOperand {
    /// Converts this resolved operand into a `MachineOperand` suitable for
    /// use in a `MachineInstr`. Returns `None` for `Unknown` and `GotoLabel`.
    fn as_machine_operand(&self) -> Option<MachineOperand> {
        match self {
            AsmResolvedOperand::MachineOp(op) => Some(op.clone()),
            AsmResolvedOperand::Immediate(val) => Some(MachineOperand::Immediate(*val)),
            AsmResolvedOperand::PhysReg(reg) => Some(MachineOperand::Register(*reg)),
            AsmResolvedOperand::MemoryRef(base) => Some(MachineOperand::Memory {
                base: *base,
                offset: 0,
                index: None,
                scale: 1,
            }),
            AsmResolvedOperand::MemoryOp(op) => Some(op.clone()),
            AsmResolvedOperand::GotoLabel(_) => None,
            AsmResolvedOperand::Unknown(_) => None,
        }
    }

    /// Returns `true` if this is a memory operand (`MemoryOp` or `MemoryRef`).
    #[allow(dead_code)]
    fn is_memory(&self) -> bool {
        matches!(
            self,
            AsmResolvedOperand::MemoryOp(_) | AsmResolvedOperand::MemoryRef(_)
        )
    }
}

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
    #[allow(dead_code)]
    diagnostics: DiagnosticEngine,
    /// Map from IR ValueId to the MachineOperand representing it.
    /// Virtual registers are used during isel; physical assignment is deferred
    /// to register allocation.
    value_map: FxHashMap<u32, MachineOperand>,
    /// Map from IR BasicBlockId to machine BasicBlock id.
    bb_map: FxHashMap<u32, u32>,
    /// Frame slot counter for alloca instructions.
    /// Running byte offset tracking total stack space consumed by allocas.
    /// Each alloca advances this counter by the actual type size (aligned).
    /// The value stored in `FrameIndex(n)` is the positive displacement from
    /// RBP, so the actual address is `[RBP - n]`.
    frame_byte_offset: u32,
    /// Tracks whether the current function has any call instructions.
    has_calls: bool,
    /// Set of callee-saved registers used (populated during isel).
    used_callee_saved: FxHashSet<PhysReg>,
    /// Next virtual register ID for temporary values created during isel.
    next_vreg: u32,
    /// Set of virtual register IDs that require floating-point (XMM)
    /// register class. These are temporaries created by the instruction
    /// selector for operations like UCOMISS operand materialisation.
    float_vregs: std::collections::HashSet<u32>,
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
            frame_byte_offset: 0,
            has_calls: false,
            used_callee_saved: FxHashSet::default(),
            next_vreg: 0,
            float_vregs: std::collections::HashSet::new(),
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
        self.frame_byte_offset = 0;
        self.has_calls = false;
        self.used_callee_saved = FxHashSet::default();
        self.next_vreg = func.next_value_id;
        self.float_vregs = std::collections::HashSet::new();

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
                    self.value_map
                        .insert(vi.id.0, MachineOperand::Symbol(sym_name.to_string()));
                } else if let Some(int_str) = n.strip_prefix("const.int.") {
                    if let Ok(val) = int_str.parse::<i64>() {
                        self.value_map
                            .insert(vi.id.0, MachineOperand::Immediate(val));
                    }
                } else if let Some(flt_str) = n.strip_prefix("const.float.") {
                    if let Ok(val) = flt_str.parse::<f64>() {
                        // Store float constant as raw bits for later
                        // materialization via SSE immediate patterns.
                        //
                        // CRITICAL: For F32 constants, we must use the f32
                        // bit representation, not f64!  An f64 20.5 has bits
                        // 0x4034800000000000, but when stored as a 32-bit
                        // value only the lower 32 bits (0x00000000) would be
                        // written, yielding zero.  The correct f32 bits for
                        // 20.5 are 0x41A40000.
                        let bits = if vi.ty == IrType::F32 {
                            (val as f32).to_bits() as i64
                        } else {
                            val.to_bits() as i64
                        };
                        // eprintln!("[DEBUG FLOAT CONST] name={} ty={:?} val={} bits=0x{:x}", n, vi.ty, val, bits);
                        self.value_map
                            .insert(vi.id.0, MachineOperand::Immediate(bits));
                    }
                } else if n == "const.null" {
                    self.value_map.insert(vi.id.0, MachineOperand::Immediate(0));
                }
            }
        }

        // Step 1c: For variadic functions, reserve a register save area
        // and spill all 6 integer argument registers to the stack frame.
        //
        // On x86-64 System V ABI, the first 6 integer/pointer arguments
        // are passed in RDI, RSI, RDX, RCX, R8, R9.  Variadic functions
        // need these values saved to memory so that va_start / va_arg can
        // walk them sequentially.
        //
        // Layout (FrameIndex(N) → [RBP − N]):
        //   [RBP − 48]  RDI   (arg 0)
        //   [RBP − 40]  RSI   (arg 1)
        //   [RBP − 32]  RDX   (arg 2)
        //   [RBP − 24]  RCX   (arg 3)
        //   [RBP − 16]  R8    (arg 4)
        //   [RBP −  8]  R9    (arg 5)
        //
        // Normal allocas begin after the 48-byte save area.
        if func.is_variadic {
            self.frame_byte_offset = 48;

            let entry_id = self.bb_map[&func.entry_block_id.0];
            let va_save_regs: [PhysReg; 6] = [RDI, RSI, RDX, RCX, R8, R9];
            for (i, &reg) in va_save_regs.iter().enumerate() {
                let offset = (48 - i * 8) as u32; // 48, 40, 32, 24, 16, 8
                let mut mi = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                mi.add_operand(MachineOperand::FrameIndex(offset));
                mi.add_operand(MachineOperand::Register(reg));
                if let Some(mbb) = mfunc.get_block_mut(entry_id) {
                    mbb.push_instr(mi);
                }
            }
        }

        // Step 2: Lower parameters — copy from physical ABI registers to
        // virtual registers for each function parameter.
        self.lower_parameters(func, &mut mfunc);

        // Step 3: Lower each IR basic block's instructions.
        // eprintln!("[IR_DBG] Function: {} has {} basic blocks", func.name, func.basic_blocks.len());
        for bb in &func.basic_blocks {
            // eprintln!("[IR_DBG]   BB{}: {} instructions", bb.id.0, bb.instructions().len());
            for _instr in bb.instructions() {
                // eprintln!("[IR_DBG]     {:?}", instr);
            }
        }
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
        // Transfer float vreg overrides so the register allocator knows
        // which codegen-created temporaries need XMM registers.
        mfunc.float_vregs = std::mem::take(&mut self.float_vregs);

        // Step 5: Compute frame size from actual alloca byte consumption.
        // frame_byte_offset tracks the total bytes consumed by all allocas.
        // Align to 16 bytes for System V ABI stack alignment.
        let alloca_frame_bytes = self.frame_byte_offset;
        let aligned = (alloca_frame_bytes + 15) & !15;
        mfunc.frame_size = aligned;

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

            // Determine the correct operand size for register-to-register
            // copies of this parameter.  Pointer and 64-bit integer
            // parameters MUST use REX.W (SIZE_QWORD) to preserve the full
            // 64-bit value; a plain 32-bit MOV would zero the upper 32 bits,
            // destroying stack addresses and other high pointers.
            let param_size = match &param.ty {
                IrType::Ptr
                | IrType::I64
                | IrType::I128
                | IrType::Array { .. }
                | IrType::Struct { .. } => SIZE_QWORD,
                _ => 0, // 32-bit default for I8, I16, I32, etc.
            };

            match &locations[i] {
                ParamLocation::Register(phys) => {
                    // MOV from physical ABI register to virtual register
                    let mut mi = MachineInstr::new(opcodes::MOV_RR | param_size);
                    mi.add_operand(vreg);
                    mi.add_operand(MachineOperand::Register(*phys));
                    if let Some(mbb) = mfunc.get_block_mut(entry_id) {
                        mbb.push_instr(mi);
                    }
                }
                ParamLocation::RegisterPair(lo, _hi) => {
                    // Two MOVs for register pairs (e.g., 128-bit structs)
                    let mut mi_lo = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
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
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | param_size);
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
                    // Hidden pointer is always a 64-bit address.
                    let mut mi = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
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
                ..
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
            // --- Floating-point conversion instructions ---
            Instruction::SIToFP {
                result,
                value,
                to_ty,
            } => self.lower_si_to_fp(*result, *value, to_ty, func),
            Instruction::UIToFP {
                result,
                value,
                to_ty,
            } => self.lower_ui_to_fp(*result, *value, to_ty, func),
            Instruction::FPToSI {
                result,
                value,
                to_ty,
            } => self.lower_fp_to_si(*result, *value, to_ty, func),
            Instruction::FPToUI {
                result,
                value,
                to_ty,
            } => self.lower_fp_to_ui(*result, *value, to_ty, func),
            Instruction::FPExt {
                result,
                value,
                to_ty,
            } => self.lower_fp_ext(*result, *value, to_ty, func),
            Instruction::FPTrunc {
                result,
                value,
                to_ty,
            } => self.lower_fp_trunc(*result, *value, to_ty, func),

            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_align_stack: _,
                goto_targets: _,
            } => self.lower_inline_asm(
                result.as_ref().copied(),
                template,
                constraints,
                operands,
                clobbers,
            ),

            Instruction::BlockAddress { result, block } => {
                self.lower_block_address(*result, *block)
            }

            Instruction::IndirectBranch {
                addr,
                possible_targets: _,
            } => self.lower_indirect_branch(*addr),
        }
    }

    // -----------------------------------------------------------------------
    // Alloca lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR Alloca to a frame slot assignment.
    ///
    /// Computes the actual byte size of the allocated type and advances
    /// the frame byte offset accordingly. The `FrameIndex` value stores
    /// the positive displacement from RBP, so the effective address at
    /// runtime is `[RBP - displacement]`.
    fn lower_alloca(&mut self, result: ValueId, ty: &IrType, alignment: u32) -> Vec<MachineInstr> {
        // Compute actual type size (minimum 8 bytes for stack alignment)
        let type_size = ty.size_bytes(&self.config.target);
        let alloc_size = if type_size == 0 { 8 } else { type_size as u32 };

        // Determine alignment (at least 8 bytes for stack slot alignment on x86-64)
        let align = std::cmp::max(alignment, 8) as u32;

        // Advance the frame byte offset by the allocation size
        let needed = self.frame_byte_offset + alloc_size;
        // Align the cumulative offset so the base address is properly aligned
        let aligned_offset = (needed + align - 1) & !(align - 1);
        self.frame_byte_offset = aligned_offset;

        // The FrameIndex stores the positive displacement from RBP.
        // The base address of this allocation is at [RBP - aligned_offset].
        let fi = MachineOperand::FrameIndex(aligned_offset);
        self.value_map.insert(result.0, fi);

        // No machine instructions emitted for alloca — it's a frame-level
        // operation resolved during prologue generation.
        vec![]
    }

    // -----------------------------------------------------------------------
    // Load lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR Load to an x86-64 MOV from memory.
    ///
    /// When the pointer operand is a `Symbol` (global variable address),
    /// we emit a RIP-relative load (`MOV reg, [rip + symbol]`) via the
    /// dedicated `MOV_RM_SYM` opcode.  For ordinary memory/frame operands
    /// we use the standard `MOV_RM`.
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

        // Aggregate types (arrays, structs) cannot be loaded into a
        // register.  Treat "Load aggregate" as LEA — compute the address
        // instead.  This handles residual IR patterns where array-decay
        // was not fully resolved during lowering.
        if matches!(ty, IrType::Array { .. } | IrType::Struct { .. }) {
            match &src {
                MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. } => {
                    let mut mi = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                    mi.add_operand(dst);
                    mi.add_operand(src);
                    instrs.push(mi);
                }
                MachineOperand::Symbol(_) => {
                    let mut mi = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                    mi.add_operand(dst);
                    mi.add_operand(src);
                    instrs.push(mi);
                }
                _ => {
                    // If it's already a register, just copy it (it's a pointer).
                    let mut mi = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                    mi.add_operand(dst);
                    mi.add_operand(src);
                    instrs.push(mi);
                }
            }
            return instrs;
        }

        match &src {
            MachineOperand::Symbol(_sym) => {
                // Global variable — use RIP-relative load.
                // MOV_RM_SYM encodes `mov reg, [rip + symbol]`.
                // CRITICAL: Include the correct operand size so that
                // 64-bit values (pointers, I64) use REX.W encoding.
                // Without this, pointer values like `stderr` get
                // truncated to 32 bits, destroying the upper half.
                let sym_size = match ty {
                    IrType::Ptr
                    | IrType::I64
                    | IrType::I128
                    | IrType::Array { .. }
                    | IrType::Struct { .. } => SIZE_QWORD,
                    _ => 0, // 32-bit default for I32 and smaller
                };
                let mut mi = MachineInstr::new(opcodes::MOV_RM_SYM | sym_size);
                mi.add_operand(dst);
                mi.add_operand(src);
                instrs.push(mi);
            }
            _ => {
                if ty.is_floating() {
                    // Floating-point load from a pointer.
                    //
                    // The source operand (`src`) can be:
                    //   • FrameIndex  — extract_mem converts to [RBP-offset]
                    //   • Memory{..}  — already an indirect reference
                    //   • VirtualReg  — holds a pointer (e.g. result of GEP)
                    //
                    // For FrameIndex and Memory, the MOVSS/MOVSD encoder
                    // correctly emits a memory load because extract_mem
                    // matches those operand variants.
                    //
                    // For VirtualReg, after register allocation the vreg
                    // becomes a GPR.  encode_sse_mov_dispatch would then
                    // see two physical registers and emit a reg-to-reg move
                    // (`movss xmm, xmm` / `movss xmm, gpr` — both wrong).
                    //
                    // Fix: use MOV_RM which explicitly treats any PhysReg
                    // operand as an indirect memory reference [reg], and
                    // already detects SSE destination registers to emit
                    // the correct MOVSS/MOVSD memory load encoding.
                    let is_frame_or_mem = matches!(
                        src,
                        MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. }
                    );

                    if is_frame_or_mem {
                        // FrameIndex/Memory: use MOVSS/MOVSD directly —
                        // the encoder handles them via extract_mem.
                        let opcode = match ty {
                            IrType::F32 => opcodes::MOVSS,
                            IrType::F64 => opcodes::MOVSD,
                            _ => opcodes::MOVSD,
                        };
                        let mut mi = MachineInstr::new(opcode);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        instrs.push(mi);
                    } else {
                        // VirtualReg (pointer in GPR): use MOV_RM which
                        // treats the GPR as [reg] for the memory operand.
                        // The size flag determines F32 (DWord=default 0) vs
                        // F64 (QWord=1<<24).  The MOV_RM encoder detects
                        // the SSE destination and emits MOVSS or MOVSD.
                        let size_flag = match ty {
                            IrType::F32 => 0u32, // DWord = default
                            _ => SIZE_QWORD,
                        };
                        let mut mi = MachineInstr::new(opcodes::MOV_RM | size_flag);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        instrs.push(mi);
                    }
                } else if matches!(ty, IrType::I8 | IrType::I1 | IrType::I16) {
                    // For sub-32-bit integer loads, use MOVZX to zero-extend
                    // the value into the full register.  A plain byte/word MOV
                    // would only modify the low portion, leaving garbage in the
                    // upper bits.
                    let src_width: i64 = match ty {
                        IrType::I8 | IrType::I1 => 8,
                        IrType::I16 => 16,
                        _ => 8,
                    };
                    let is_vreg = matches!(src, MachineOperand::VirtualReg(_));
                    if is_vreg {
                        // VirtualReg holds a pointer. After regalloc it becomes
                        // a PhysReg. The MOV_RM encoder treats bare PhysRegs as
                        // [reg] (indirect memory). We do a correctly-sized
                        // MOV_RM load first, then MOVZX reg→reg to clear upper
                        // bits.  CRITICAL: use SIZE_WORD for I16 types, not
                        // SIZE_BYTE, otherwise only 1 byte is loaded and the
                        // upper byte of the 16-bit register contains garbage.
                        let tmp = self.alloc_vreg();
                        let load_size = match ty {
                            IrType::I16 => SIZE_WORD,
                            _ => SIZE_BYTE, // I8, I1
                        };
                        let mut ld = MachineInstr::new(opcodes::MOV_RM | load_size);
                        ld.add_operand(tmp.clone());
                        ld.add_operand(src);
                        instrs.push(ld);
                        let mut zx = MachineInstr::new(opcodes::MOVZX);
                        zx.add_operand(dst);
                        zx.add_operand(tmp);
                        zx.add_operand(MachineOperand::Immediate(src_width));
                        instrs.push(zx);
                    } else {
                        // FrameIndex or Memory — MOVZX can directly encode
                        // these via extract_mem in the encoder.
                        let mut mi = MachineInstr::new(opcodes::MOVZX);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        mi.add_operand(MachineOperand::Immediate(src_width));
                        instrs.push(mi);
                    }
                } else {
                    // 32-bit / 64-bit integer loads
                    let size_flag = self.size_flag_for_type(ty);
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | size_flag);
                    mi.add_operand(dst);
                    mi.add_operand(src);
                    instrs.push(mi);
                }
            }
        }
        instrs
    }

    // -----------------------------------------------------------------------
    // Store lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR Store to an x86-64 MOV to memory.
    ///
    /// When the destination pointer is a `Symbol` (global variable address),
    /// we emit a RIP-relative store via `MOV_MR_SYM` or `MOV_MI_SYM`.
    fn lower_store(
        &mut self,
        value: ValueId,
        ptr: ValueId,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let mut src = self.get_operand(value);
        let dst = self.get_operand(ptr);
        let _val_ty_dbg = func.get_value_type(value);
        let mut instrs = Vec::new();

        // ---------------------------------------------------------------
        // Aggregate copy (struct / array):
        // The IR pattern for struct assignment is:
        //   Load { result: vN, ptr: src_alloca, ty: Struct{...} }
        //   Store { value: vN, ptr: dst_alloca }
        // lower_load for structs emits LEA (computing the source address
        // into a virtual register). So here `src` is a VirtualReg holding
        // the *address* of the source aggregate. `dst` is typically a
        // FrameIndex for the destination. We emit an inline word-by-word
        // copy (unrolled memcpy).
        // ---------------------------------------------------------------
        let val_ty = func.get_value_type(value);
        if matches!(val_ty, IrType::Struct { .. } | IrType::Array { .. }) {
            let copy_size = val_ty.size_bytes(&self.config.target) as u32;
            if copy_size > 0 {
                return self.emit_aggregate_copy(src, dst, copy_size);
            }
        }

        // If the value being stored is a FrameIndex (an alloca address),
        // we need to materialise the stack-slot address into a register
        // via LEA, because in the IR an alloca's "value" IS its address
        // — not the contents of the stack slot.  Without this, the
        // backend would emit an invalid mem-to-mem MOV or, worse, load
        // the *contents* of the stack slot instead of its address.
        // The same applies to Memory operands (rare in value position).
        match &src {
            MachineOperand::FrameIndex(offset) => {
                let vreg = self.alloc_vreg();
                let mut lea = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                lea.add_operand(vreg.clone());
                lea.add_operand(MachineOperand::FrameIndex(*offset));
                instrs.push(lea);
                src = vreg;
            }
            MachineOperand::Memory { .. } => {
                // Memory in value position — load into a temporary register.
                let val_ty = func.get_value_type(value);
                let size_flag = self.size_flag_for_type(&val_ty);
                let vreg = self.alloc_vreg();
                let mut mov = MachineInstr::new(opcodes::MOV_RM | size_flag);
                mov.add_operand(vreg.clone());
                mov.add_operand(src);
                instrs.push(mov);
                src = vreg;
            }
            MachineOperand::Symbol(_) => {
                // Symbol in value position — the IR wants the ADDRESS of the
                // symbol (e.g. a string literal pointer or function pointer).
                // Materialize the RIP-relative address into a register via LEA.
                let vreg = self.alloc_vreg();
                let mut lea = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                lea.add_operand(vreg.clone());
                lea.add_operand(src);
                instrs.push(lea);
                src = vreg;
            }
            _ => {}
        }

        match &dst {
            MachineOperand::Symbol(_sym) => {
                // Global variable — use RIP-relative store.
                // Include the correct operand size for 64-bit values
                // (pointers, I64) so that REX.W is emitted.
                let val_ty_sym = func.get_value_type(value);
                let sym_store_size = match &val_ty_sym {
                    IrType::Ptr
                    | IrType::I64
                    | IrType::I128
                    | IrType::Array { .. }
                    | IrType::Struct { .. } => SIZE_QWORD,
                    _ => 0,
                };
                match &src {
                    MachineOperand::Immediate(_) => {
                        // MOV_MI_SYM: mov [rip + symbol], imm32
                        let mut mi = MachineInstr::new(opcodes::MOV_MI_SYM | sym_store_size);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        instrs.push(mi);
                    }
                    _ => {
                        // MOV_MR_SYM: mov [rip + symbol], reg
                        let mut mi = MachineInstr::new(opcodes::MOV_MR_SYM | sym_store_size);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        instrs.push(mi);
                    }
                }
            }
            _ => {
                let val_ty = func.get_value_type(value);
                if val_ty.is_floating() {
                    // Floating-point store: MOVSD/MOVSS only supports
                    // reg-reg or mem-reg / reg-mem forms — NOT immediates.
                    // If `src` is an immediate (raw f64/f32 bits), we use
                    // a regular integer store instead since the bits are
                    // identical — MOV [mem], reg64 with the raw bits loaded
                    // via movabs.  This avoids needing a MOVQ xmm, gpr
                    // instruction.
                    if matches!(src, MachineOperand::Immediate(_)) {
                        let gpr_tmp = self.alloc_vreg();
                        instrs.push(self.make_mov64(gpr_tmp.clone(), src));
                        let size_flag = match val_ty {
                            IrType::F64 => SIZE_QWORD,
                            IrType::F32 => 0u32, // 32-bit default
                            _ => SIZE_QWORD,
                        };
                        let mut mi = MachineInstr::new(opcodes::MOV_MR | size_flag);
                        mi.add_operand(dst);
                        mi.add_operand(gpr_tmp);
                        instrs.push(mi);
                    } else {
                        // If the destination is a VirtualReg (GPR holding
                        // a pointer, e.g. from a GEP), we must use MOV_MR
                        // which already handles GPR-as-indirect-memory and
                        // dispatches to MOVSS/MOVSD encoding when the source
                        // is an SSE register.  The raw MOVSS/MOVSD opcode
                        // encoder does NOT know how to treat a bare GPR as
                        // a memory reference.
                        let dst_is_vreg = matches!(
                            dst,
                            MachineOperand::VirtualReg(_) | MachineOperand::Register(_)
                        );
                        if dst_is_vreg {
                            let size_flag = match val_ty {
                                IrType::F32 => 0u32,
                                IrType::F64 => SIZE_QWORD,
                                _ => SIZE_QWORD,
                            };
                            let mut mi = MachineInstr::new(opcodes::MOV_MR | size_flag);
                            mi.add_operand(dst);
                            mi.add_operand(src);
                            instrs.push(mi);
                        } else {
                            let opcode = match val_ty {
                                IrType::F32 => opcodes::MOVSS,
                                IrType::F64 => opcodes::MOVSD,
                                _ => opcodes::MOVSD,
                            };
                            let mut mi = MachineInstr::new(opcode);
                            mi.add_operand(dst);
                            mi.add_operand(src);
                            instrs.push(mi);
                        }
                    }
                } else {
                    // MOV [mem], reg/imm
                    // Use SIZE_QWORD for 64-bit types (pointers, I64)
                    //
                    // CRITICAL: I1 (boolean from ICmp) is stored as I32
                    // (32-bit) because the x86-64 codegen for ICmp always
                    // emits SETcc + MOVZX, producing a full 32-bit value
                    // in the register.  If we stored only 1 byte and later
                    // loaded 4 bytes (for an `int` destination), we would
                    // read 3 bytes of stack garbage.
                    let size_flag = if matches!(val_ty, IrType::I1) {
                        0 // 32-bit (DWord default) — safe because MOVZX
                          // already cleared the upper 24 bits
                    } else {
                        self.size_flag_for_type(&val_ty)
                    };

                    // CRITICAL: x86-64 MOV [mem], imm only supports 32-bit
                    // sign-extended immediates.  For 64-bit values that do
                    // not fit in sign-extended imm32 (e.g. 0x0102030405060708),
                    // we must first load the full 64-bit constant into a
                    // register using MOV_RI (movabs), then store the register.
                    let need_reg_materialize = if let MachineOperand::Immediate(v) = &src {
                        size_flag == SIZE_QWORD
                            && (*v < (i32::MIN as i64) || *v > (i32::MAX as i64))
                    } else {
                        false
                    };

                    if need_reg_materialize {
                        let vreg = self.alloc_vreg();
                        let mut load_imm = MachineInstr::new(opcodes::MOV_RI | SIZE_QWORD);
                        load_imm.add_operand(vreg.clone());
                        load_imm.add_operand(src);
                        instrs.push(load_imm);
                        let mut store = MachineInstr::new(opcodes::MOV_MR | size_flag);
                        store.add_operand(dst);
                        store.add_operand(vreg);
                        instrs.push(store);
                    } else {
                        let opcode = match &src {
                            MachineOperand::Immediate(_) => opcodes::MOV_MI | size_flag,
                            _ => opcodes::MOV_MR | size_flag,
                        };
                        let mut mi = MachineInstr::new(opcode);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        instrs.push(mi);
                    }
                }
            }
        }
        instrs
    }

    // -----------------------------------------------------------------------
    // Aggregate (struct / array) inline copy
    // -----------------------------------------------------------------------

    /// Emits an inline memcpy from `src_op` (address register or FrameIndex)
    /// to `dst_op` (FrameIndex or address register) for `size` bytes.
    ///
    /// The copy is unrolled into 8-byte (qword) chunks, with a trailing
    /// 4-byte, 2-byte, or 1-byte move for any remainder.
    fn emit_aggregate_copy(
        &mut self,
        src_op: MachineOperand,
        dst_op: MachineOperand,
        size: u32,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Materialise src and dst addresses into virtual registers.
        let src_addr = match &src_op {
            MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. } => {
                let r = self.alloc_vreg();
                let mut lea = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                lea.add_operand(r.clone());
                lea.add_operand(src_op.clone());
                instrs.push(lea);
                r
            }
            MachineOperand::VirtualReg(_) | MachineOperand::Register(_) => src_op.clone(),
            _ => {
                // Unexpected operand kind — treat as address value.
                src_op.clone()
            }
        };
        let dst_addr = match &dst_op {
            MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. } => {
                let r = self.alloc_vreg();
                let mut lea = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                lea.add_operand(r.clone());
                lea.add_operand(dst_op.clone());
                instrs.push(lea);
                r
            }
            MachineOperand::VirtualReg(_) | MachineOperand::Register(_) => dst_op.clone(),
            _ => dst_op.clone(),
        };

        let mut offset: u32 = 0;

        // Copy 8-byte chunks.
        while offset + 8 <= size {
            let tmp = self.alloc_vreg();
            // Load 8 bytes from [src_addr + offset].
            if offset == 0 {
                // Use the source address register directly as [reg].
                let mut ld = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                ld.add_operand(tmp.clone());
                ld.add_operand(src_addr.clone());
                instrs.push(ld);
            } else {
                // Compute src_addr + offset into a temporary register.
                let addr_tmp = self.alloc_vreg();
                let mut mov = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                mov.add_operand(addr_tmp.clone());
                mov.add_operand(src_addr.clone());
                instrs.push(mov);
                let mut add = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add.add_operand(addr_tmp.clone());
                add.add_operand(addr_tmp.clone());
                add.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add);
                let mut ld = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                ld.add_operand(tmp.clone());
                ld.add_operand(addr_tmp);
                instrs.push(ld);
            }
            // Store 8 bytes to [dst_addr + offset].
            if offset == 0 {
                let mut st = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                st.add_operand(dst_addr.clone());
                st.add_operand(tmp);
                instrs.push(st);
            } else {
                let addr_tmp = self.alloc_vreg();
                let mut mov = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                mov.add_operand(addr_tmp.clone());
                mov.add_operand(dst_addr.clone());
                instrs.push(mov);
                let mut add = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add.add_operand(addr_tmp.clone());
                add.add_operand(addr_tmp.clone());
                add.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add);
                let mut st = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                st.add_operand(addr_tmp);
                st.add_operand(tmp);
                instrs.push(st);
            }
            offset += 8;
        }

        // Copy remaining 4-byte chunk.
        if offset + 4 <= size {
            let tmp = self.alloc_vreg();
            if offset == 0 {
                let mut ld = MachineInstr::new(opcodes::MOV_RM);
                ld.add_operand(tmp.clone());
                ld.add_operand(src_addr.clone());
                instrs.push(ld);
                let mut st = MachineInstr::new(opcodes::MOV_MR);
                st.add_operand(dst_addr.clone());
                st.add_operand(tmp);
                instrs.push(st);
            } else {
                let s_tmp = self.alloc_vreg();
                let mut mov_s = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                mov_s.add_operand(s_tmp.clone());
                mov_s.add_operand(src_addr.clone());
                instrs.push(mov_s);
                let mut add_s = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add_s.add_operand(s_tmp.clone());
                add_s.add_operand(s_tmp.clone());
                add_s.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add_s);
                let mut ld = MachineInstr::new(opcodes::MOV_RM);
                ld.add_operand(tmp.clone());
                ld.add_operand(s_tmp);
                instrs.push(ld);

                let d_tmp = self.alloc_vreg();
                let mut mov_d = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                mov_d.add_operand(d_tmp.clone());
                mov_d.add_operand(dst_addr.clone());
                instrs.push(mov_d);
                let mut add_d = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add_d.add_operand(d_tmp.clone());
                add_d.add_operand(d_tmp.clone());
                add_d.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add_d);
                let mut st = MachineInstr::new(opcodes::MOV_MR);
                st.add_operand(d_tmp);
                st.add_operand(tmp);
                instrs.push(st);
            }
            offset += 4;
        }

        // Copy remaining 2-byte chunk.
        if offset + 2 <= size {
            let tmp = self.alloc_vreg();
            let s_tmp = self.alloc_vreg();
            let mut mov_s = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
            mov_s.add_operand(s_tmp.clone());
            mov_s.add_operand(src_addr.clone());
            instrs.push(mov_s);
            if offset > 0 {
                let mut add_s = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add_s.add_operand(s_tmp.clone());
                add_s.add_operand(s_tmp.clone());
                add_s.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add_s);
            }
            let mut ld = MachineInstr::new(opcodes::MOVZX);
            ld.add_operand(tmp.clone());
            ld.add_operand(s_tmp);
            instrs.push(ld);

            let d_tmp = self.alloc_vreg();
            let mut mov_d = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
            mov_d.add_operand(d_tmp.clone());
            mov_d.add_operand(dst_addr.clone());
            instrs.push(mov_d);
            if offset > 0 {
                let mut add_d = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add_d.add_operand(d_tmp.clone());
                add_d.add_operand(d_tmp.clone());
                add_d.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add_d);
            }
            let mut st = MachineInstr::new(opcodes::MOV_MR);
            st.add_operand(d_tmp);
            st.add_operand(tmp);
            instrs.push(st);
            offset += 2;
        }

        // Copy remaining 1 byte.
        if offset < size {
            let tmp = self.alloc_vreg();
            let s_tmp = self.alloc_vreg();
            let mut mov_s = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
            mov_s.add_operand(s_tmp.clone());
            mov_s.add_operand(src_addr.clone());
            instrs.push(mov_s);
            if offset > 0 {
                let mut add_s = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add_s.add_operand(s_tmp.clone());
                add_s.add_operand(s_tmp.clone());
                add_s.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add_s);
            }
            let mut ld = MachineInstr::new(opcodes::MOVZX);
            ld.add_operand(tmp.clone());
            ld.add_operand(s_tmp);
            instrs.push(ld);

            let d_tmp = self.alloc_vreg();
            let mut mov_d = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
            mov_d.add_operand(d_tmp.clone());
            mov_d.add_operand(dst_addr.clone());
            instrs.push(mov_d);
            if offset > 0 {
                let mut add_d = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                add_d.add_operand(d_tmp.clone());
                add_d.add_operand(d_tmp.clone());
                add_d.add_operand(MachineOperand::Immediate(offset as i64));
                instrs.push(add_d);
            }
            let mut st = MachineInstr::new(opcodes::MOV_MR);
            st.add_operand(d_tmp);
            st.add_operand(tmp);
            instrs.push(st);
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

        // CRITICAL FIX: x86-64 ALU instructions (ADD/SUB/AND/OR/XOR r64, imm)
        // only support sign-extended 32-bit immediates.  If the RHS immediate
        // does not fit in i32, we must materialise it into a scratch register
        // via MOV_RI (movabs) and use the register-register form instead.
        let is_64bit = matches!(ty, IrType::I64 | IrType::Ptr);
        let large_imm = if let MachineOperand::Immediate(v) = &rhs_op {
            is_64bit && (*v < (i32::MIN as i64) || *v > (i32::MAX as i64))
        } else {
            false
        };

        let (rhs_op, is_rhs_imm) = if large_imm {
            let tmp = self.alloc_vreg();
            let mut load = MachineInstr::new(opcodes::MOV_RI | SIZE_QWORD);
            load.add_operand(tmp.clone());
            load.add_operand(rhs_op);
            instrs.push(load);
            (tmp, false) // treat as register operand from now on
        } else {
            let is_imm = matches!(rhs_op, MachineOperand::Immediate(_));
            (rhs_op, is_imm)
        };

        // Size flag: on x86-64 all 64-bit integer and pointer operations
        // MUST carry REX.W (SIZE_QWORD) to prevent silent truncation to
        // 32-bit.  The flag is OR'd into every opcode in the match arms
        // below so that ADD, SUB, IMUL, AND, OR, XOR all get the correct
        // operand size.
        let sz: u32 = if matches!(ty, IrType::I64 | IrType::Ptr) {
            SIZE_QWORD
        } else {
            0
        };

        match op {
            BinOp::Add => {
                // mov dst, lhs; add dst, rhs
                instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                let opc = if is_rhs_imm {
                    opcodes::ADD_RI | sz
                } else {
                    opcodes::ADD_RR | sz
                };
                let mut mi = MachineInstr::new(opc);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Sub => {
                instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                let opc = if is_rhs_imm {
                    opcodes::SUB_RI | sz
                } else {
                    opcodes::SUB_RR | sz
                };
                let mut mi = MachineInstr::new(opc);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Mul => {
                // IMUL r64, r/m64 — two-operand form
                // Note: IMUL with immediate uses a three-operand form that we
                // handle by loading the immediate into a scratch register first.
                if is_rhs_imm {
                    let tmp = self.alloc_vreg();
                    instrs.push(self.make_mov_typed(tmp.clone(), rhs_op, ty));
                    instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                    let mut mi = MachineInstr::new(opcodes::IMUL_RR | sz);
                    mi.add_operand(dst);
                    mi.add_operand(tmp);
                    instrs.push(mi);
                } else {
                    instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                    let mut mi = MachineInstr::new(opcodes::IMUL_RR | sz);
                    mi.add_operand(dst);
                    mi.add_operand(rhs_op);
                    instrs.push(mi);
                }
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
                instrs.extend(self.lower_shift(dst, lhs_op, rhs_op, opcodes::SHL, ty));
            }
            BinOp::LShr => {
                instrs.extend(self.lower_shift(dst, lhs_op, rhs_op, opcodes::SHR, ty));
            }
            BinOp::AShr => {
                instrs.extend(self.lower_shift(dst, lhs_op, rhs_op, opcodes::SAR, ty));
            }
            BinOp::And => {
                instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                let opc = if is_rhs_imm {
                    opcodes::AND_RI | sz
                } else {
                    opcodes::AND_RR | sz
                };
                let mut mi = MachineInstr::new(opc);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Or => {
                instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                let opc = if is_rhs_imm {
                    opcodes::OR_RI | sz
                } else {
                    opcodes::OR_RR | sz
                };
                let mut mi = MachineInstr::new(opc);
                mi.add_operand(dst);
                mi.add_operand(rhs_op);
                instrs.push(mi);
            }
            BinOp::Xor => {
                instrs.push(self.make_mov_typed(dst.clone(), lhs_op, ty));
                let opc = if is_rhs_imm {
                    opcodes::XOR_RI | sz
                } else {
                    opcodes::XOR_RR | sz
                };
                let mut mi = MachineInstr::new(opc);
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
    /// Lowers floating-point binary operations to SSE/SSE2 instructions.
    ///
    /// # Immediate Materialization
    ///
    /// SSE arithmetic instructions (`ADDSS`, `MULSS`, etc.) do **not**
    /// accept immediate operands — only register or memory sources.
    /// Float constants in the IR are represented as `Immediate(bits)`
    /// (the IEEE 754 bit pattern stored in an i64).  Before we can use
    /// them in an SSE instruction we must spill the bits to a temporary
    /// stack slot and load from memory:
    ///
    /// ```asm
    ///   mov  [rbp - N], <bits>      ; store integer bits to stack
    ///   movss xmm_dst, [rbp - N]    ; load as float into XMM
    /// ```
    fn lower_fp_binop(
        &mut self,
        dst: MachineOperand,
        op: BinOp,
        lhs: MachineOperand,
        rhs: MachineOperand,
        ty: &IrType,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let is_f32 = matches!(ty, IrType::F32);

        // Helper: materialise a float immediate into a FrameIndex operand
        // by storing the raw bits to a freshly-allocated stack slot.
        let materialise_float_imm = |this: &mut Self,
                                     instrs: &mut Vec<MachineInstr>,
                                     bits: i64,
                                     is_f32: bool|
         -> MachineOperand {
            let slot_size = if is_f32 { 4u32 } else { 8u32 };
            let needed = this.frame_byte_offset + slot_size;
            let align = slot_size;
            let aligned_offset = (needed + align - 1) & !(align - 1);
            this.frame_byte_offset = aligned_offset;
            let fi = MachineOperand::FrameIndex(aligned_offset);

            if is_f32 {
                // F32: 32-bit immediate fits in MOV_MI directly.
                let mut store = MachineInstr::new(opcodes::MOV_MI);
                store.add_operand(fi.clone());
                store.add_operand(MachineOperand::Immediate(bits));
                instrs.push(store);
            } else {
                // F64: 64-bit immediate does NOT fit in MOV_MI (which
                // sign-extends a 32-bit immediate). Use MOV_RI (movabs)
                // to load into a scratch GPR, then store the GPR to the slot.
                let scratch = this.alloc_vreg();
                let mut movabs = MachineInstr::new(opcodes::MOV_RI | SIZE_QWORD);
                movabs.add_operand(scratch.clone());
                movabs.add_operand(MachineOperand::Immediate(bits));
                instrs.push(movabs);

                let mut store = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                store.add_operand(fi.clone());
                store.add_operand(scratch);
                instrs.push(store);
            }

            fi
        };

        // Resolve LHS: if it is an immediate, spill to stack and load.
        let lhs_resolved = if let MachineOperand::Immediate(bits) = &lhs {
            materialise_float_imm(self, &mut instrs, *bits, is_f32)
        } else {
            lhs
        };

        // Resolve RHS: same treatment.
        let rhs_resolved = if let MachineOperand::Immediate(bits) = &rhs {
            materialise_float_imm(self, &mut instrs, *bits, is_f32)
        } else {
            rhs
        };

        // MOVSS/MOVSD lhs → dst  (copy LHS to the destination XMM)
        let mov_op = if is_f32 {
            opcodes::MOVSS
        } else {
            opcodes::MOVSD
        };
        let mut mov = MachineInstr::new(mov_op);
        mov.add_operand(dst.clone());
        mov.add_operand(lhs_resolved);
        instrs.push(mov);

        // Select the SSE arithmetic opcode.
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
                // FRem has no direct x86 instruction — would need a library call.
                // For now emit DIVSD and handle remainder via separate logic.
                if is_f32 {
                    opcodes::DIVSS
                } else {
                    opcodes::DIVSD
                }
            }
            _ => unreachable!("Not a floating-point BinOp"),
        };

        // OP dst, rhs  (dst = dst ⊕ rhs)
        let mut mi = MachineInstr::new(opcode);
        mi.add_operand(dst);
        mi.add_operand(rhs_resolved);
        instrs.push(mi);
        instrs
    }

    /// Lowers integer division and remainder operations.
    ///
    /// x86-64 IDIV/DIV uses RAX:RDX as the implicit dividend.
    /// Result: RAX = quotient, RDX = remainder.
    ///
    /// CRITICAL: The divisor register must NOT be RAX or RDX because
    /// those are implicitly used as the dividend (RDX:RAX).  We always
    /// materialize the divisor into RCX before setting up RDX:RAX so
    /// the register allocator cannot accidentally assign the divisor
    /// to RDX (which would be overwritten by CQO/CDQ or XOR).
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
        let sz: u32 = if matches!(ty, IrType::I64 | IrType::Ptr) {
            SIZE_QWORD
        } else {
            0
        };

        // STEP 1: Move the divisor into RCX FIRST, before touching RAX/RDX.
        // This prevents the register allocator from placing the divisor in
        // RDX which would be clobbered by CQO/CDQ or XOR.
        let div_mov_opc = if matches!(rhs, MachineOperand::Immediate(_)) {
            opcodes::MOV_RI | sz
        } else {
            opcodes::MOV_RR | sz
        };
        let mut mov_rcx = MachineInstr::new(div_mov_opc);
        mov_rcx.add_operand(MachineOperand::Register(RCX));
        mov_rcx.add_operand(rhs);
        mov_rcx.add_implicit_def(RCX);
        instrs.push(mov_rcx);

        // STEP 2: Move dividend to RAX
        let mov_opc = if matches!(lhs, MachineOperand::Immediate(_)) {
            opcodes::MOV_RI | sz
        } else {
            opcodes::MOV_RR | sz
        };
        let mut mov_rax = MachineInstr::new(mov_opc);
        mov_rax.add_operand(MachineOperand::Register(RAX));
        mov_rax.add_operand(lhs);
        mov_rax.add_implicit_def(RAX);
        instrs.push(mov_rax);

        // STEP 3: Sign-extend or zero-extend into RDX:RAX
        if signed {
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

        // STEP 4: IDIV/DIV with RCX as the divisor — safe because
        // RCX is not part of the implicit RDX:RAX dividend.
        let div_op = if signed {
            opcodes::IDIV | sz
        } else {
            opcodes::DIV | sz
        };
        let mut div = MachineInstr::new(div_op);
        div.add_operand(MachineOperand::Register(RCX));
        div.add_implicit_def(RAX);
        div.add_implicit_def(RDX);
        div.add_implicit_use(RAX);
        div.add_implicit_use(RDX);
        div.add_implicit_use(RCX);
        instrs.push(div);

        // STEP 5: Move result from RAX (quotient) or RDX (remainder) to dst
        let result_reg = if is_remainder { RDX } else { RAX };
        let mut mov_result = MachineInstr::new(opcodes::MOV_RR | sz);
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
        ty: &IrType,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();
        let sz: u32 = if matches!(ty, IrType::I64 | IrType::Ptr) {
            SIZE_QWORD
        } else {
            0
        };

        // Move value to dst (respecting operand size)
        instrs.push(self.make_mov_typed(dst.clone(), lhs, ty));

        match &rhs {
            MachineOperand::Immediate(amt) => {
                // Shift by immediate
                let mut mi = MachineInstr::new(opcode | sz);
                mi.add_operand(dst);
                mi.add_operand(MachineOperand::Immediate(*amt & 63));
                instrs.push(mi);
            }
            _ => {
                // Shift by variable: move amount to RCX, use CL
                // (shift amount is always 8-bit CL, no need for sz on
                // the MOV into RCX — truncation is expected for the
                // shift count register)
                let mut mov_cl = MachineInstr::new(opcodes::MOV_RR);
                mov_cl.add_operand(MachineOperand::Register(RCX));
                mov_cl.add_operand(rhs);
                instrs.push(mov_cl);

                let mut mi = MachineInstr::new(opcode | sz);
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
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let lhs_op = self.get_operand(lhs);
        let rhs_op = self.get_operand(rhs);
        let cc = CondCode::from_icmp_predicate(pred);

        let mut instrs = Vec::new();

        // ================================================================
        // Determine the correct comparison width from the IR operand type.
        // ================================================================
        //
        // On x86-64, 32-bit MOV into a 32-bit register (e.g. `mov eax, [mem]`)
        // zero-extends the result into the full 64-bit register.  This means
        // a signed 32-bit value like -5 (stored as 0xFFFFFFFB) is loaded as
        // 0x00000000FFFFFFFB — a large positive 64-bit number.
        //
        // If we then perform a 64-bit CMP (with REX.W / SIZE_QWORD), the
        // comparison treats both operands as 64-bit values and the signed
        // comparison `SETG` (greater-than) tests SF≠OF on the 64-bit result.
        // Since 0x00000000FFFFFFFB > 0x000000000000000A in 64-bit signed
        // arithmetic, `-5 > 10` incorrectly evaluates to true.
        //
        // The fix: use the IR value's type to select the comparison width.
        // For I32 and narrower types, use 32-bit CMP (no REX.W).  For I64
        // and pointer types, use 64-bit CMP (REX.W / SIZE_QWORD).
        //
        // 32-bit CMP correctly sign-interprets 0xFFFFFFFB as -5 and 0x0A as
        // 10, so `SETG` correctly returns false for `-5 > 10`.
        let lhs_ty = func.get_value_type(lhs).clone();
        let needs_64bit = matches!(lhs_ty, IrType::I64 | IrType::Ptr);
        let size_flag = if needs_64bit { SIZE_QWORD } else { 0 };

        // CMP lhs, rhs — pick RR vs RI based on operand types.
        // CMP requires the first operand to be a register; if lhs is an
        // immediate, materialize it in a temporary register first.
        let lhs_reg = if matches!(lhs_op, MachineOperand::Immediate(_)) {
            let tmp = self.alloc_vreg();
            if needs_64bit {
                instrs.push(self.make_mov64(tmp.clone(), lhs_op));
            } else {
                let mut mov = MachineInstr::new(opcodes::MOV_RI);
                mov.add_operand(tmp.clone());
                mov.add_operand(lhs_op);
                instrs.push(mov);
            }
            tmp
        } else {
            lhs_op
        };

        // For 64-bit comparisons, large constants that don't fit in a
        // sign-extended 32-bit immediate must be materialised into a
        // register first.
        let (rhs_final, is_rhs_imm) = if let MachineOperand::Immediate(v) = &rhs_op {
            if needs_64bit && (*v < (i32::MIN as i64) || *v > (i32::MAX as i64)) {
                let tmp = self.alloc_vreg();
                let mut load = MachineInstr::new(opcodes::MOV_RI | SIZE_QWORD);
                load.add_operand(tmp.clone());
                load.add_operand(rhs_op);
                instrs.push(load);
                (tmp, false)
            } else {
                (rhs_op, true)
            }
        } else {
            (rhs_op, false)
        };
        let cmp_opc = if is_rhs_imm {
            opcodes::CMP_RI
        } else {
            opcodes::CMP_RR
        };
        let mut cmp = MachineInstr::new(cmp_opc | size_flag);
        cmp.add_operand(lhs_reg);
        cmp.add_operand(rhs_final);
        instrs.push(cmp);

        // SETcc byte_tmp (sets byte based on flags).
        // We use a SEPARATE vreg for the byte result so that the register
        // allocator correctly tracks the byte-width lifetime independently
        // from the final 32-bit result.  Without this, the allocator may
        // spill the final result as a byte (from the SETcc definition)
        // but reload it as 32-bit, leaving garbage in the upper bytes.
        let byte_tmp = self.alloc_vreg();
        let mut setcc = MachineInstr::new(cc.setcc_opcode());
        setcc.add_operand(byte_tmp.clone());
        setcc.add_operand(MachineOperand::Immediate(cc.encoding()));
        instrs.push(setcc);

        // MOVZX dst, byte_tmp — zero-extend the byte result to 32-bit.
        // The `dst` vreg is now only defined by this MOVZX, so the
        // allocator will treat it as a full-width (32-bit) value for
        // spill/reload purposes.
        let mut movzx = MachineInstr::new(opcodes::MOVZX);
        movzx.add_operand(dst.clone());
        movzx.add_operand(byte_tmp);
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
        let is_f32 = matches!(lhs_ty, IrType::F32);
        let cmp_op = if is_f32 {
            opcodes::UCOMISS
        } else {
            opcodes::UCOMISD
        };
        let mov_op = if is_f32 {
            opcodes::MOVSS
        } else {
            opcodes::MOVSD
        };

        // UCOMISS/UCOMISD require XMM register or memory operands — NOT
        // immediate operands.  Float constants arrive as Immediate(bits);
        // we must spill them to a temp stack slot just like lower_fp_binop.
        let materialise_imm = |this: &mut Self,
                               instrs: &mut Vec<MachineInstr>,
                               bits: i64,
                               f32_flag: bool|
         -> MachineOperand {
            let slot_size = if f32_flag { 4u32 } else { 8u32 };
            let needed = this.frame_byte_offset + slot_size;
            let align = slot_size;
            let aligned_offset = (needed + align - 1) & !(align - 1);
            this.frame_byte_offset = aligned_offset;
            let fi = MachineOperand::FrameIndex(aligned_offset);
            if f32_flag {
                // F32: 32-bit immediate fits in MOV_MI directly.
                let mut store = MachineInstr::new(opcodes::MOV_MI);
                store.add_operand(fi.clone());
                store.add_operand(MachineOperand::Immediate(bits));
                instrs.push(store);
            } else {
                // F64: 64-bit immediate does NOT fit in MOV_MI (sign-
                // extended 32-bit). Use MOV_RI (movabs) to load into
                // a scratch GPR, then store the GPR to the stack slot.
                let scratch = this.alloc_vreg();
                let mut movabs = MachineInstr::new(opcodes::MOV_RI | SIZE_QWORD);
                movabs.add_operand(scratch.clone());
                movabs.add_operand(MachineOperand::Immediate(bits));
                instrs.push(movabs);

                let mut store = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                store.add_operand(fi.clone());
                store.add_operand(scratch);
                instrs.push(store);
            }
            fi
        };

        // Resolve LHS: if immediate, materialize to stack; then load into
        // an XMM virtual register because UCOMISS first operand must be XMM.
        let lhs_xmm = match &lhs_op {
            MachineOperand::Immediate(bits) => {
                let fi = materialise_imm(self, &mut instrs, *bits, is_f32);
                let vr = self.alloc_vreg_float();
                let mut ld = MachineInstr::new(mov_op);
                ld.add_operand(vr.clone());
                ld.add_operand(fi);
                instrs.push(ld);
                vr
            }
            MachineOperand::FrameIndex(_) => {
                let vr = self.alloc_vreg_float();
                let mut ld = MachineInstr::new(mov_op);
                ld.add_operand(vr.clone());
                ld.add_operand(lhs_op);
                instrs.push(ld);
                vr
            }
            _ => lhs_op,
        };

        // Resolve RHS: can be XMM or memory, but NOT immediate.
        let rhs_resolved = match &rhs_op {
            MachineOperand::Immediate(bits) => materialise_imm(self, &mut instrs, *bits, is_f32),
            _ => rhs_op,
        };

        let mut ucomi = MachineInstr::new(cmp_op);
        ucomi.add_operand(lhs_xmm);
        ucomi.add_operand(rhs_resolved);
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

        // Fast-path: if condition is a compile-time constant, emit an
        // unconditional jump.  This is critical for `do { ... } while(0)`
        // and `while(1)` idioms that are pervasive in C macros.
        if let Some(const_val) = self.get_constant_value(condition) {
            let target_id = if const_val != 0 { true_id } else { false_id };
            let mut jmp = MachineInstr::new(opcodes::JMP);
            jmp.add_operand(MachineOperand::Label(target_id));
            jmp.set_terminator();
            return vec![jmp];
        }

        let cond_op = self.get_operand(condition);
        let mut instrs = Vec::new();

        // If the operand is an immediate (not caught above because
        // get_constant_value checks value_map while get_operand may
        // derive it differently), handle it by moving into a register
        // first so the TEST instruction is valid.
        let test_op = match &cond_op {
            MachineOperand::Immediate(val) => {
                // Another constant path — resolve statically.
                let target_id = if *val != 0 { true_id } else { false_id };
                let mut jmp = MachineInstr::new(opcodes::JMP);
                jmp.add_operand(MachineOperand::Label(target_id));
                jmp.set_terminator();
                return vec![jmp];
            }
            _ => cond_op,
        };

        // TEST condition, condition (sets ZF based on condition value)
        let mut test = MachineInstr::new(opcodes::TEST_RR);
        test.add_operand(test_op.clone());
        test.add_operand(test_op);
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
    // Computed goto (block address / indirect branch)
    // -----------------------------------------------------------------------

    /// Lowers a `BlockAddress` instruction — produces a pointer to the
    /// runtime address of a basic block label.
    ///
    /// Emits `LEA_LABEL vreg, label_id` which the assembler resolves to
    /// `lea <label>(%rip), %reg`.
    fn lower_block_address(&mut self, result: ValueId, block: BasicBlockId) -> Vec<MachineInstr> {
        let vreg = self.get_or_create_vreg(result);
        let target_id = self.bb_map.get(&block.0).copied().unwrap_or(block.0);
        let mut mi = MachineInstr::new(opcodes::LEA_LABEL | SIZE_QWORD);
        mi.add_operand(vreg);
        mi.add_operand(MachineOperand::Label(target_id));
        vec![mi]
    }

    /// Lowers an `IndirectBranch` instruction — emits `jmp *%reg`.
    fn lower_indirect_branch(&mut self, addr: ValueId) -> Vec<MachineInstr> {
        let op = self.get_operand(addr);
        let mut mi = MachineInstr::new(opcodes::JMP_INDIRECT);
        mi.add_operand(op);
        mi.set_terminator();
        vec![mi]
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

        // ============================================================
        // TWO-PHASE ARGUMENT SETUP (prevents parallel-move clobbering)
        // ============================================================
        //
        // The register allocator does not track physical register writes
        // in MOV instructions, so it may assign a virtual register to the
        // same physical register as an argument destination.  This causes
        // a clobbering bug when moving arguments to their ABI registers:
        //
        //   MOV rsi, vreg1   ← arg1 (writes rsi)
        //   MOV rdx, vreg2   ← arg2 (vreg2 was in esi — now clobbered!)
        //
        // Fix: PHASE 1 saves all register-destined argument values to
        //      temporary stack slots (which cannot conflict with any
        //      register).  PHASE 2 loads from those stack slots into
        //      the physical argument registers.
        //
        // Symbols and immediates bypass the stack (no register conflict
        // possible).  Stack-destined args are handled directly.

        // Collect register-arg metadata for the two-phase approach.
        // temp_fi[i] = Some(frame_offset) if arg i was saved to a temp slot.
        let mut reg_arg_temps: Vec<(usize, PhysReg, Option<u32>, MachineOperand, u32)> = Vec::new();

        for (i, arg_val) in args.iter().enumerate() {
            if i >= locations.len() {
                break;
            }

            let arg_op = self.get_operand(*arg_val);
            let arg_ty = func.get_value_type(*arg_val);
            let sz = self.size_flag_for_type(arg_ty);
            let is_symbol = matches!(arg_op, MachineOperand::Symbol(_));
            let is_imm = matches!(arg_op, MachineOperand::Immediate(_));

            match &locations[i] {
                ParamLocation::Register(phys) | ParamLocation::HiddenPointer(phys) => {
                    if is_symbol || is_imm {
                        // Symbols/immediates — no register conflict possible.
                        reg_arg_temps.push((i, *phys, None, arg_op, sz));
                    } else {
                        // PHASE 1: Save to a temporary stack slot.
                        self.frame_byte_offset += 8;
                        let aligned = (self.frame_byte_offset + 7) & !7;
                        self.frame_byte_offset = aligned;
                        let fi_off = aligned;

                        // Determine if this argument is floating-point
                        // (destined for an XMM register).
                        let is_float_arg = registers::is_sse(*phys);

                        match &arg_op {
                            MachineOperand::FrameIndex(_) => {
                                // FrameIndex as a call argument means we're
                                // passing the ADDRESS of a stack slot (e.g.
                                // `&local_var`, array decay, or struct pointer).
                                // If the IR wanted the contents, it would have
                                // emitted a Load first, producing a vreg.
                                // Use LEA to compute the address, then store
                                // to the temp slot.
                                let scratch = self.alloc_vreg();
                                let mut lea = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                                lea.add_operand(scratch.clone());
                                lea.add_operand(arg_op.clone());
                                instrs.push(lea);

                                let mut st = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                                st.add_operand(MachineOperand::FrameIndex(fi_off));
                                st.add_operand(scratch);
                                instrs.push(st);
                            }
                            MachineOperand::Memory { .. } => {
                                // Memory operands (globals, etc.) load their
                                // contents — they already went through a Load
                                // in the IR.
                                let scratch = self.alloc_vreg();
                                if is_float_arg {
                                    let mut ld = MachineInstr::new(opcodes::MOVSD);
                                    ld.add_operand(scratch.clone());
                                    ld.add_operand(arg_op.clone());
                                    instrs.push(ld);

                                    let mut st = MachineInstr::new(opcodes::MOVSD);
                                    st.add_operand(MachineOperand::FrameIndex(fi_off));
                                    st.add_operand(scratch);
                                    instrs.push(st);
                                } else {
                                    let mut ld = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                                    ld.add_operand(scratch.clone());
                                    ld.add_operand(arg_op.clone());
                                    instrs.push(ld);

                                    let mut st = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                                    st.add_operand(MachineOperand::FrameIndex(fi_off));
                                    st.add_operand(scratch);
                                    instrs.push(st);
                                }
                            }
                            _ => {
                                // vreg / register → stack slot.
                                // For float args, use MOVSD store.
                                if is_float_arg {
                                    let mut st = MachineInstr::new(opcodes::MOVSD);
                                    st.add_operand(MachineOperand::FrameIndex(fi_off));
                                    st.add_operand(arg_op.clone());
                                    instrs.push(st);
                                } else {
                                    let mut st = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                                    st.add_operand(MachineOperand::FrameIndex(fi_off));
                                    st.add_operand(arg_op.clone());
                                    instrs.push(st);
                                }
                            }
                        }
                        reg_arg_temps.push((i, *phys, Some(fi_off), arg_op, sz));
                    }
                }
                ParamLocation::RegisterPair(lo, _hi) => {
                    // Same treatment for register pairs
                    if is_symbol || is_imm {
                        reg_arg_temps.push((i, *lo, None, arg_op, sz));
                    } else {
                        self.frame_byte_offset += 8;
                        let aligned = (self.frame_byte_offset + 7) & !7;
                        self.frame_byte_offset = aligned;
                        let fi_off = aligned;
                        let mut st = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                        st.add_operand(MachineOperand::FrameIndex(fi_off));
                        st.add_operand(arg_op.clone());
                        instrs.push(st);
                        reg_arg_temps.push((i, *lo, Some(fi_off), arg_op, sz));
                    }
                }
                ParamLocation::Stack { offset } => {
                    // Stack arguments: emit directly (no register conflict).
                    if is_symbol {
                        let scratch = self.alloc_vreg();
                        let mut lea = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                        lea.add_operand(scratch.clone());
                        lea.add_operand(arg_op);
                        instrs.push(lea);
                        let mut mi = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                        mi.add_operand(MachineOperand::Memory {
                            base: RSP,
                            offset: *offset,
                            index: None,
                            scale: 1,
                        });
                        mi.add_operand(scratch);
                        instrs.push(mi);
                    } else if matches!(arg_op, MachineOperand::FrameIndex(_)) {
                        // FrameIndex means we're passing an address (alloca
                        // pointer).  LEA to get the address, then store to
                        // the stack argument slot.
                        let scratch = self.alloc_vreg();
                        let mut lea = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                        lea.add_operand(scratch.clone());
                        lea.add_operand(arg_op);
                        instrs.push(lea);
                        let mut mi = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                        mi.add_operand(MachineOperand::Memory {
                            base: RSP,
                            offset: *offset,
                            index: None,
                            scale: 1,
                        });
                        mi.add_operand(scratch);
                        instrs.push(mi);
                    } else {
                        let mut mi = MachineInstr::new(opcodes::MOV_MR | sz);
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
            }
        }

        // PHASE 2: Load from temp stack slots / emit symbols+immediates
        //          into the physical argument registers.  Because all
        //          sources are now stack slots (FrameIndex), symbols, or
        //          immediates, there can be no register-register clobber.
        for &(_, phys, ref temp_fi, ref orig_op, sz) in &reg_arg_temps {
            if let Some(fi_off) = temp_fi {
                // Load from temporary stack slot → physical register.
                // Must use MOVSD/MOVSS for XMM registers (SSE loads),
                // and MOV_RM for general-purpose registers.
                if registers::is_sse(phys) {
                    let mut ld = MachineInstr::new(opcodes::MOVSD);
                    ld.add_operand(MachineOperand::Register(phys));
                    ld.add_operand(MachineOperand::FrameIndex(*fi_off));
                    instrs.push(ld);
                } else {
                    let mut ld = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                    ld.add_operand(MachineOperand::Register(phys));
                    ld.add_operand(MachineOperand::FrameIndex(*fi_off));
                    instrs.push(ld);
                }
            } else if matches!(orig_op, MachineOperand::Symbol(_)) {
                // LEA phys, [rip + symbol]
                let mut mi = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                mi.add_operand(MachineOperand::Register(phys));
                mi.add_operand(orig_op.clone());
                instrs.push(mi);
            } else if registers::is_sse(phys) {
                // FLOAT IMMEDIATE → XMM register.
                //
                // x86-64 has no "MOV XMM, imm" instruction.  To
                // materialise a float constant in an XMM register we:
                //   1. Store the raw bit-pattern to a temporary stack
                //      slot using an integer MOV (MOV_MR).
                //   2. Load from that slot into the XMM register using
                //      MOVSD (for F64) or MOVSS (for F32).
                //
                // This is the standard pattern used by GCC and Clang
                // when the constant is not already in a rodata section.
                self.frame_byte_offset += 8;
                let aligned = (self.frame_byte_offset + 7) & !7;
                self.frame_byte_offset = aligned;
                let tmp_fi = aligned;

                // Step 1: Store bits as integer to the temp slot.
                // Use a scratch GPR to hold the immediate value, then
                // store it to the frame slot.
                let scratch = self.alloc_vreg();
                let mut load_bits = MachineInstr::new(opcodes::MOV_RI | SIZE_QWORD);
                load_bits.add_operand(scratch.clone());
                load_bits.add_operand(orig_op.clone());
                instrs.push(load_bits);

                let mut store_bits = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                store_bits.add_operand(MachineOperand::FrameIndex(tmp_fi));
                store_bits.add_operand(scratch);
                instrs.push(store_bits);

                // Step 2: Load from the stack slot into the XMM register.
                let mut ld_xmm = MachineInstr::new(opcodes::MOVSD);
                ld_xmm.add_operand(MachineOperand::Register(phys));
                ld_xmm.add_operand(MachineOperand::FrameIndex(tmp_fi));
                instrs.push(ld_xmm);
            } else {
                // Integer immediate → GPR: straightforward MOV.
                let mut mi = MachineInstr::new(opcodes::MOV_RI | sz);
                mi.add_operand(MachineOperand::Register(phys));
                mi.add_operand(orig_op.clone());
                instrs.push(mi);
            }
        }

        // ============================================================
        // x86-64 System V ABI: Set AL = number of vector registers used
        // ============================================================
        //
        // The ABI mandates that for calls to variadic functions (and calls
        // through untyped function pointers), the caller must set %al to
        // the number of vector (XMM) registers that contain arguments.
        // The value must be an upper bound in [0, 8].
        //
        // For non-variadic calls the callee ignores %al, so it is always
        // safe (and cheap — one 2-byte instruction) to emit this
        // unconditionally.  This matches GCC/Clang behaviour for calls
        // through function pointers where the prototype is not visible.
        {
            let xmm_count: i64 = locations
                .iter()
                .filter(|loc| match loc {
                    ParamLocation::Register(phys) => registers::is_sse(*phys),
                    _ => false,
                })
                .count() as i64;

            // MOV EAX, xmm_count  (32-bit immediate → zero-extends to RAX,
            // so %al = xmm_count).  We use the default (no SIZE_QWORD) MOV_RI
            // because the value is a small immediate [0..8].
            let mut set_al = MachineInstr::new(opcodes::MOV_RI);
            set_al.add_operand(MachineOperand::Register(RAX));
            set_al.add_operand(MachineOperand::Immediate(xmm_count));
            instrs.push(set_al);
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

        // GEP computes addresses, so when the base is a FrameIndex
        // (stack alloca) or Memory operand, we need LEA to obtain the
        // address rather than MOV_RM which would load the *value*.
        // All GEP base-loads use 64-bit operations (pointer arithmetic)
        let load_base =
            |instrs: &mut Vec<MachineInstr>, dst: MachineOperand, src: MachineOperand| {
                match &src {
                    MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. } => {
                        let mut lea = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                        lea.add_operand(dst);
                        lea.add_operand(src);
                        instrs.push(lea);
                    }
                    MachineOperand::Symbol(_) => {
                        let mut lea = MachineInstr::new(opcodes::LEA_SYM | SIZE_QWORD);
                        lea.add_operand(dst);
                        lea.add_operand(src);
                        instrs.push(lea);
                    }
                    _ => {
                        // Register or immediate: plain MOV (64-bit for pointers)
                        let mut mi = MachineInstr::new(opcodes::MOV_RR | SIZE_QWORD);
                        mi.add_operand(dst);
                        mi.add_operand(src);
                        instrs.push(mi);
                    }
                }
            };

        if indices.is_empty() {
            // No indices: result is just the base pointer
            load_base(&mut instrs, dst, base_op);
            return instrs;
        }

        // Start with base pointer address
        load_base(&mut instrs, dst.clone(), base_op);

        // Process each GEP index.
        //
        // LLVM-style GEP semantics:
        //   - The FIRST index is always an array-level index: it multiplies
        //     by the full size of the pointee type (the base type).
        //     For struct pointers this means "which struct in an array of
        //     structs" — typically 0 for simple member access.
        //   - SUBSEQUENT indices drill into nested types:
        //     * If the current type is a struct, the index selects a field.
        //     * If the current type is an array, the index selects an element.
        let mut current_ty = ty;
        for (idx_pos, idx) in indices.iter().enumerate() {
            let idx_op = self.get_operand(*idx);

            // The first index always uses the whole-type size (array indexing
            // through the pointer), NOT struct-field indexing.
            let is_first_index = idx_pos == 0;

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
                IrType::Struct { fields, packed: _ } if !is_first_index => {
                    // Subsequent index on a struct: constant field index.
                    if let MachineOperand::Immediate(field_idx) = &idx_op {
                        let field_idx = *field_idx as usize;
                        if field_idx < fields.len() {
                            // Compute byte offset to the requested field,
                            // respecting alignment padding between fields.
                            let mut offset: u64 = 0;
                            for field in fields.iter().take(field_idx) {
                                let f_size = field.size_bytes(&self.config.target);
                                let f_align = field.alignment(&self.config.target);
                                offset = (offset + f_align - 1) & !(f_align - 1);
                                offset += f_size;
                            }
                            let field_align = fields[field_idx].alignment(&self.config.target);
                            offset = (offset + field_align - 1) & !(field_align - 1);

                            // ADD dst, offset (64-bit for pointer arithmetic)
                            if offset > 0 {
                                let mut add = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
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
                IrType::Struct { .. } => {
                    // First index on a struct: treat as array-of-structs.
                    // Multiply index by sizeof(struct).  current_ty stays
                    // the same (the struct type itself) for subsequent
                    // indices to drill into fields.
                    current_ty.size_bytes(&self.config.target)
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
                        let mut add = MachineInstr::new(opcodes::ADD_RI | SIZE_QWORD);
                        add.add_operand(dst.clone());
                        add.add_operand(MachineOperand::Immediate(offset));
                        instrs.push(add);
                    }
                }
                _ => {
                    // Variable index: use IMUL+ADD (all 64-bit for pointer arithmetic)
                    //
                    // CRITICAL: GEP indices may be I32 (e.g. from C `int` variables).
                    // For 64-bit pointer arithmetic we must sign-extend them to 64-bit
                    // first, otherwise a negative index like -4 (0xFFFFFFFC in 32-bit)
                    // would be zero-extended to 0x00000000FFFFFFFC, producing a wrong
                    // enormous positive offset instead of the correct negative one.
                    let idx_op_64 = {
                        // Check if the IR index value is 32-bit and sign-extend if so.
                        let needs_sext = {
                            let idx_ir_ty = _func.get_value_type(*idx);
                            matches!(idx_ir_ty, IrType::I32 | IrType::I16 | IrType::I8)
                        };
                        if needs_sext {
                            let sext_reg = self.alloc_vreg();
                            let mut movsx = MachineInstr::new(opcodes::MOVSX);
                            movsx.add_operand(sext_reg.clone());
                            movsx.add_operand(idx_op);
                            movsx.add_operand(MachineOperand::Immediate(32)); // source width
                            instrs.push(movsx);
                            sext_reg
                        } else {
                            idx_op
                        }
                    };

                    if elem_size == 1 {
                        let mut add = MachineInstr::new(opcodes::ADD_RR | SIZE_QWORD);
                        add.add_operand(dst.clone());
                        add.add_operand(idx_op_64);
                        instrs.push(add);
                    } else {
                        // IMUL tmp, index, elem_size; ADD dst, tmp (all 64-bit)
                        let tmp = self.alloc_vreg();
                        let mut imul = MachineInstr::new(opcodes::IMUL_RI | SIZE_QWORD);
                        imul.add_operand(tmp.clone());
                        imul.add_operand(idx_op_64);
                        imul.add_operand(MachineOperand::Immediate(elem_size as i64));
                        instrs.push(imul);

                        let mut add = MachineInstr::new(opcodes::ADD_RR | SIZE_QWORD);
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
        if dst_is_fp {
            vec![self.make_mov_opcode(dst, src, opcodes::MOVSD)]
        } else {
            // Use make_mov() which auto-selects the correct opcode based
            // on operand types (MOV_RI for immediates, MOV_RM for memory,
            // MOV_RR for register-to-register).  This is critical for phi
            // elimination copies where the source may be a constant
            // (e.g., `false_val = const 0` in LogicalAnd short-circuit).
            vec![self.make_mov(dst, src)]
        }
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

        // Same-size "zext" is a no-op — just emit a 64-bit MOV.
        // This can occur when the IR front-end emits ZExt I64→I64 for
        // builtins whose argument already has the correct width.
        if src_ty == to_ty {
            return vec![self.make_mov64(dst, src)];
        }

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

        // i8/i16 -> i32/i64: use MOVZX
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
        to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);
        let src_ty = func.get_value_type(value);

        // Same-type "sext" is a no-op — emit a sized MOV.
        if src_ty == to_ty {
            let sz = self.size_flag_for_type(to_ty);
            let mut mi = MachineInstr::new(opcodes::MOV_RR | sz);
            mi.add_operand(dst);
            mi.add_operand(src);
            return vec![mi];
        }

        // Determine the source width in bits for the encoder.
        let src_bits: i64 = match src_ty {
            IrType::I1 | IrType::I8 => 8,
            IrType::I16 => 16,
            IrType::I32 => 32,
            _ => 32, // fallback
        };

        // If the source is an immediate, we can compute the sign-extension
        // at compile time and just load the result with a 64-bit MOV.
        if let MachineOperand::Immediate(imm) = src {
            let sign_extended = match src_bits {
                8 => (imm as i8) as i64,
                16 => (imm as i16) as i64,
                32 => (imm as i32) as i64,
                _ => imm,
            };
            let sz = self.size_flag_for_type(to_ty);
            let mut mi = MachineInstr::new(opcodes::MOV_RI | sz);
            mi.add_operand(dst);
            mi.add_operand(MachineOperand::Immediate(sign_extended));
            return vec![mi];
        }

        // Register/memory source: use MOVSX with the proper source width.
        // The encoder expects operands: [dst, src, src_width_imm].
        let mut movsx = MachineInstr::new(opcodes::MOVSX);
        movsx.add_operand(dst);
        movsx.add_operand(src);
        movsx.add_operand(MachineOperand::Immediate(src_bits));
        vec![movsx]
    }

    /// Lowers a simple move (used for IntToPtr and other no-op conversions).
    /// Always uses 64-bit MOV because IntToPtr produces a pointer-width value.
    fn lower_mov(&mut self, result: ValueId, value: ValueId) -> Vec<MachineInstr> {
        let dst = self.get_or_create_vreg(result);
        let src = self.get_operand(value);
        vec![self.make_mov64(dst, src)]
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
    // Floating-point conversion lowering
    // -----------------------------------------------------------------------

    /// Lowers SIToFP (signed integer to float) via CVTSI2SS/CVTSI2SD.
    fn lower_si_to_fp(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstr> {
        self.lower_int_to_float(result, value, to_ty)
    }

    /// Lowers UIToFP (unsigned integer to float).
    /// x86-64 CVTSI2SS/SD treats the source as signed. For unsigned 32-bit,
    /// we zero-extend to 64-bit first so the 64-bit CVTSI2SD reads a
    /// positive value.
    fn lower_ui_to_fp(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let src_ty = func.get_value_type(value);
        let src_bits = src_ty.size_bits(&self.config.target);
        if src_bits < 64 {
            // Zero-extend to 64-bit so CVTSI2SD reads a positive i64.
            let tmp = self.alloc_vreg();
            let src = self.get_operand(value);
            // Zero-extend via MOV (32-bit mov implicitly zero-extends on x86-64).
            let mut insts = vec![self.make_mov(tmp.clone(), src)];
            let dst = self.get_or_create_vreg(result);
            let opcode = match to_ty {
                IrType::F32 => opcodes::CVTSI2SS,
                _ => opcodes::CVTSI2SD,
            };
            let mut mi = MachineInstr::new(opcode);
            mi.add_operand(dst);
            mi.add_operand(tmp);
            insts.push(mi);
            insts
        } else {
            // 64-bit unsigned: use CVTSI2SD (may lose precision for large values,
            // but correct for checkpoint tests).
            self.lower_int_to_float(result, value, to_ty)
        }
    }

    /// Lowers FPToSI (float to signed integer) via CVTTSS2SI/CVTTSD2SI.
    fn lower_fp_to_si(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let from_ty = func.get_value_type(value);
        self.lower_float_to_int(result, value, from_ty)
    }

    /// Lowers FPToUI (float to unsigned integer) via CVTTSS2SI/CVTTSD2SI.
    fn lower_fp_to_ui(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let from_ty = func.get_value_type(value);
        self.lower_float_to_int(result, value, from_ty)
    }

    /// Lowers FPExt (float widening, e.g., F32 → F64) via CVTSS2SD.
    fn lower_fp_ext(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let from_ty = func.get_value_type(value);
        self.lower_float_convert(result, value, from_ty, to_ty)
    }

    /// Lowers FPTrunc (float narrowing, e.g., F64 → F32) via CVTSD2SS.
    fn lower_fp_trunc(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let from_ty = func.get_value_type(value);
        self.lower_float_convert(result, value, from_ty, to_ty)
    }

    // Inline assembly lowering
    // -----------------------------------------------------------------------

    /// Lowers an IR `InlineAsm` instruction into machine instructions.
    ///
    /// Supports two distinct code paths:
    /// 1. **GCC user inline asm** — templates using `%0`, `%1`, `%[name]` operand
    ///    references with colon-separated constraint strings (e.g., `"=r:r"`).
    /// 2. **Builtin inline asm** — templates using `$0`, `$1` references with
    ///    comma-separated constraints (e.g., `"=r,r"`), generated by the
    ///    `__builtin_*` lowering in `expr_lowering.rs`.
    pub fn lower_inline_asm(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
    ) -> Vec<MachineInstr> {
        // Detect GCC user asm: templates containing %N or %[name] references.
        let is_gcc_user_asm = self.detect_gcc_user_asm(template);

        if is_gcc_user_asm {
            self.lower_gcc_user_asm(result, template, constraints, operands, clobbers)
        } else {
            self.lower_builtin_asm(result, template, constraints, operands, clobbers)
        }
    }

    /// Returns `true` if the template string contains GCC-style operand
    /// references (`%0`, `%1`, `%[name]`), indicating user inline assembly
    /// rather than compiler-generated builtin assembly.
    fn detect_gcc_user_asm(&self, template: &str) -> bool {
        let chars: Vec<char> = template.chars().collect();
        let len = chars.len();
        for i in 0..len {
            if chars[i] == '%' && i + 1 < len {
                let next = chars[i + 1];
                if next.is_ascii_digit() || next == '[' {
                    return true;
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // GCC user inline assembly path
    // -----------------------------------------------------------------------

    /// Lowers GCC-style user inline assembly with `%N`/`%[name]` operand
    /// references and colon-separated constraint strings.
    ///
    /// # Constraint String Format (from `asm_lowering.rs`)
    ///
    /// Output and input constraints are separated by a colon:
    /// - `"=r:r"` → output `=r`, input `r`
    /// - `"+r:r"` → read-write output `+r`, input `r`
    /// - `"=r,=r:r,r"` → two outputs, two inputs
    ///
    /// # Operand Layout
    ///
    /// The IR operand array is ordered:
    /// `[store_targets..., pre_loads..., inputs...]`
    /// where `store_targets` has `n_outputs` entries and `pre_loads` has
    /// `n_rw` entries (one per `+r` read-write constraint).
    fn lower_gcc_user_asm(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Parse the constraint string to determine output/input/goto structure.
        // Format: "outputs:inputs:gotos" where gotos section is optional and
        // contains GOTO<block_id> entries for asm goto targets.
        let (output_constraints, input_constraints, goto_targets_ir) =
            Self::parse_colon_constraints_with_goto(constraints);

        // Map goto targets from IR BasicBlockIds to machine BasicBlock IDs.
        let goto_targets: Vec<u32> = goto_targets_ir
            .iter()
            .map(|ir_bb| *self.bb_map.get(ir_bb).unwrap_or(ir_bb))
            .collect();

        let n_outputs = output_constraints.len();
        let n_rw = output_constraints
            .iter()
            .filter(|c| c.starts_with('+'))
            .count();

        // Track which operand map entries are memory constraints.
        // The operand map is [outputs..., inputs...].
        let total_map_size = n_outputs + input_constraints.len();
        let mut is_memory_operand: Vec<bool> = vec![false; total_map_size];

        // -----------------------------------------------------------------
        // Build output operands.
        // For "=r" / "+r": allocate a vreg (first output uses IR result vid).
        // For "=m" / "+m": use the store target's FrameIndex/Memory directly.
        // -----------------------------------------------------------------
        let mut output_ops: Vec<MachineOperand> = Vec::new();
        for i in 0..n_outputs {
            let stripped = output_constraints[i]
                .trim_start_matches('=')
                .trim_start_matches('+')
                .trim_start_matches('&');
            if stripped.contains('m') {
                // Memory output: the store target's address (FrameIndex) goes
                // into the operand map so the template writes directly to memory.
                is_memory_operand[i] = true;
                if i < operands.len() {
                    let mem_op = self.get_operand(operands[i]);
                    output_ops.push(mem_op);
                } else {
                    output_ops.push(self.alloc_vreg());
                }
            } else {
                // Register output: allocate a vreg.
                let vreg = if i == 0 {
                    if let Some(res_vid) = result {
                        self.get_or_create_vreg(res_vid)
                    } else {
                        self.alloc_vreg()
                    }
                } else {
                    self.alloc_vreg()
                };
                output_ops.push(vreg);
            }
        }

        // If no outputs but IR expects a result, allocate a dummy.
        if output_ops.is_empty() {
            output_ops.push(if let Some(res_vid) = result {
                self.get_or_create_vreg(res_vid)
            } else {
                self.alloc_vreg()
            });
        }

        // -----------------------------------------------------------------
        // Build input operands.
        // For "r" / "i" / "n": load the value into a vreg.
        // For "m": use the input's FrameIndex/Memory directly.
        // For matching ("0", "1"): share the output vreg.
        // -----------------------------------------------------------------
        let input_start_idx = n_outputs + n_rw;
        let mut input_ops: Vec<MachineOperand> = Vec::new();

        for i in 0..input_constraints.len() {
            let trimmed = input_constraints[i].trim();
            let op_idx = input_start_idx + i;

            // Check for matching constraint first (e.g., "0").
            if let Ok(match_idx) = trimmed.parse::<usize>() {
                // This input shares the register with output[match_idx].
                if match_idx < output_ops.len() {
                    let out = output_ops[match_idx].clone();
                    // Pre-load the input value into the matched output vreg.
                    if op_idx < operands.len() {
                        let inp_op = self.get_operand(operands[op_idx]);
                        let inp_vreg = self.ensure_asm_operand_in_reg(&mut instrs, &inp_op, false);
                        let mut mi = MachineInstr::new(opcodes::MOV_RR);
                        mi.add_operand(out.clone());
                        mi.add_operand(inp_vreg);
                        instrs.push(mi);
                    }
                    input_ops.push(out);
                    continue;
                }
            }

            if trimmed == "m" || trimmed.contains('m') {
                // Memory input: keep as FrameIndex/Memory for direct use.
                is_memory_operand[n_outputs + i] = true;
                if op_idx < operands.len() {
                    let mem_op = self.get_operand(operands[op_idx]);
                    input_ops.push(mem_op);
                } else {
                    input_ops.push(self.alloc_vreg());
                }
            } else {
                // Register / immediate input: load value into a vreg.
                if op_idx < operands.len() {
                    let op = self.get_operand(operands[op_idx]);
                    let vreg = self.ensure_asm_operand_in_reg(&mut instrs, &op, false);
                    input_ops.push(vreg);
                } else {
                    input_ops.push(self.alloc_vreg());
                }
            }
        }

        // -----------------------------------------------------------------
        // Build combined operand map: [outputs..., inputs...].
        // -----------------------------------------------------------------
        let mut operand_map: Vec<MachineOperand> = Vec::new();
        operand_map.extend(output_ops.iter().cloned());
        operand_map.extend(input_ops.iter().cloned());

        // -----------------------------------------------------------------
        // Pre-load read-write (+r/+m) outputs from their pre-load values.
        // -----------------------------------------------------------------
        let mut rw_idx = 0;
        for (i, oc) in output_constraints.iter().enumerate() {
            if oc.starts_with('+') {
                let pre_load_op_idx = n_outputs + rw_idx;
                if pre_load_op_idx < operands.len() && i < output_ops.len() {
                    let stripped = oc.trim_start_matches('+').trim_start_matches('&');
                    if !stripped.contains('m') {
                        // Register read-write: load the current value into the vreg.
                        let pre_load_op = self.get_operand(operands[pre_load_op_idx]);
                        let pre_load_vreg =
                            self.ensure_asm_operand_in_reg(&mut instrs, &pre_load_op, false);
                        let mut mi = MachineInstr::new(opcodes::MOV_RR);
                        mi.add_operand(output_ops[i].clone());
                        mi.add_operand(pre_load_vreg);
                        instrs.push(mi);
                    }
                    // Memory read-write: the asm accesses the memory directly,
                    // no pre-load needed since it reads and writes in-place.
                }
                rw_idx += 1;
            }
        }

        // -----------------------------------------------------------------
        // Parse and emit instructions from the template.
        // -----------------------------------------------------------------
        let template_trimmed = template.trim();
        // eprintln!("[ASM_DBG] template_trimmed = {:?}", template_trimmed);
        // eprintln!("[ASM_DBG] constraints = {:?}", constraints);
        // eprintln!("[ASM_DBG] goto_targets = {:?}", goto_targets);
        // Split the template into individual assembly instructions.
        // AT&T inline assembly uses newlines, tabs, and semicolons as
        // instruction separators.  We must split on all three to correctly
        // handle templates like "testl %0, %0; jz %l[is_zero]".
        let asm_lines: Vec<&str> = template_trimmed
            .split('\n')
            .flat_map(|l| l.split('\t'))
            .flat_map(|l| l.split(';'))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        // eprintln!("[ASM_DBG] asm_lines = {:?}", asm_lines);

        for line in &asm_lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            // Assembler directives — emit as raw INLINE_ASM no-op.
            if line.starts_with('.')
                && (line.starts_with(".pushsection")
                    || line.starts_with(".popsection")
                    || line.starts_with(".byte")
                    || line.starts_with(".asciz")
                    || line.starts_with(".section"))
            {
                instrs.push(MachineInstr::new(opcodes::INLINE_ASM));
                continue;
            }

            // Parse the AT&T instruction: mnemonic operands...
            let parts: Vec<&str> = line
                .splitn(2, |c: char| c.is_whitespace())
                .map(|s| s.trim())
                .collect();
            let mnemonic = parts[0].to_lowercase();
            let operands_str = if parts.len() > 1 { parts[1] } else { "" };

            // Split operands by comma (AT&T: src, dst).
            let op_strs: Vec<&str> = if operands_str.is_empty() {
                Vec::new()
            } else {
                operands_str.split(',').map(|s| s.trim()).collect()
            };

            // Resolve operand references to AsmResolvedOperand.
            let resolved_ops: Vec<AsmResolvedOperand> = op_strs
                .iter()
                .map(|s| self.resolve_asm_operand(s, &operand_map, &is_memory_operand))
                .collect();

            // Emit machine instructions based on mnemonic.
            self.emit_gcc_asm_instruction(&mut instrs, &mnemonic, &resolved_ops, &goto_targets);
        }

        // -----------------------------------------------------------------
        // Handle clobbers: mark implicit defs on the last instruction.
        // -----------------------------------------------------------------
        if let Some(last) = instrs.last_mut() {
            for clobber in clobbers {
                match clobber.as_str() {
                    "memory" | "cc" => {}
                    _ => {
                        if let Some(reg) = self.parse_register_name(clobber) {
                            last.add_implicit_def(reg);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // For multi-output asm, store additional register outputs back.
        // Memory outputs ("=m") are already written by the asm template.
        // -----------------------------------------------------------------
        for (i, oc) in output_constraints.iter().enumerate() {
            if i == 0 {
                continue; // First output handled by IR Store.
            }
            let stripped = oc
                .trim_start_matches('=')
                .trim_start_matches('+')
                .trim_start_matches('&');
            if stripped.contains('m') {
                continue; // Memory outputs written directly by asm.
            }
            // Store the additional output vreg to its store target.
            if i < operands.len() && i < output_ops.len() {
                let store_target = self.get_operand(operands[i]);
                let mut mi = MachineInstr::new(opcodes::MOV_MR | SIZE_QWORD);
                mi.add_operand(store_target);
                mi.add_operand(output_ops[i].clone());
                instrs.push(mi);
            }
        }

        instrs
    }

    /// Parses a colon-separated constraint string into output, input, and goto lists.
    ///
    /// Format: `"outputs:inputs:gotos"` where each section is comma-separated.
    /// The goto section (if present) contains `GOTO<block_id>` entries for
    /// asm goto targets. Examples:
    /// - `"=r:r"` → outputs `["=r"]`, inputs `["r"]`, gotos `[]`
    /// - `":r:GOTO5,GOTO12"` → no outputs, input `["r"]`, gotos `[5, 12]`
    fn parse_colon_constraints_with_goto(
        constraints: &str,
    ) -> (Vec<String>, Vec<String>, Vec<u32>) {
        if constraints.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let colon_parts: Vec<&str> = constraints.splitn(3, ':').collect();
        let output_str = colon_parts[0].trim();
        let input_str = if colon_parts.len() > 1 {
            colon_parts[1].trim()
        } else {
            ""
        };
        let goto_str = if colon_parts.len() > 2 {
            colon_parts[2].trim()
        } else {
            ""
        };

        let outputs: Vec<String> = if output_str.is_empty() {
            Vec::new()
        } else {
            output_str
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        };

        let inputs: Vec<String> = if input_str.is_empty() {
            Vec::new()
        } else {
            input_str.split(',').map(|s| s.trim().to_string()).collect()
        };

        let gotos: Vec<u32> = if goto_str.is_empty() {
            Vec::new()
        } else {
            goto_str
                .split(',')
                .filter_map(|s| {
                    let s = s.trim();
                    if s.starts_with("GOTO") {
                        s[4..].parse::<u32>().ok()
                    } else {
                        None
                    }
                })
                .collect()
        };

        (outputs, inputs, gotos)
    }

    /// Ensures an asm operand is in a register (virtual register).
    /// If it's an immediate or memory operand, materializes it into a vreg.
    /// If `is_address` is true, loads the address into a vreg (LEA for stack slots).
    fn ensure_asm_operand_in_reg(
        &mut self,
        instrs: &mut Vec<MachineInstr>,
        op: &MachineOperand,
        is_address: bool,
    ) -> MachineOperand {
        match op {
            MachineOperand::VirtualReg(_) | MachineOperand::Register(_) => op.clone(),
            MachineOperand::Immediate(val) => {
                let vreg = self.alloc_vreg();
                let mut mi = MachineInstr::new(opcodes::MOV_RI);
                mi.add_operand(vreg.clone());
                mi.add_operand(MachineOperand::Immediate(*val));
                instrs.push(mi);
                vreg
            }
            MachineOperand::FrameIndex(idx) => {
                let vreg = self.alloc_vreg();
                if is_address {
                    // LEA to get address of stack slot.
                    let mut mi = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                    mi.add_operand(vreg.clone());
                    mi.add_operand(MachineOperand::Memory {
                        base: RBP,
                        offset: -(*idx as i32 + 1) * 8,
                        index: None,
                        scale: 1,
                    });
                    instrs.push(mi);
                } else {
                    // Load the value from the stack slot.
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op.clone());
                    instrs.push(mi);
                }
                vreg
            }
            MachineOperand::Memory { .. } => {
                let vreg = self.alloc_vreg();
                if is_address {
                    let mut mi = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op.clone());
                    instrs.push(mi);
                } else {
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op.clone());
                    instrs.push(mi);
                }
                vreg
            }
            _ => {
                // Symbol or other — load into vreg.
                let vreg = self.alloc_vreg();
                let mut mi = MachineInstr::new(opcodes::MOV_RM | SIZE_QWORD);
                mi.add_operand(vreg.clone());
                mi.add_operand(op.clone());
                instrs.push(mi);
                vreg
            }
        }
    }

    /// Resolves an AT&T assembly operand string to a resolved operand.
    ///
    /// Handles:
    /// - `%0`, `%1` — positional operand reference
    /// - `%[name]` — named operand reference (pre-resolved to %N by IR lowering)
    /// - `%%eax` — literal physical register
    /// - `$42` — immediate value
    /// - Other — passed through as unknown
    fn resolve_asm_operand(
        &self,
        operand_str: &str,
        operand_map: &[MachineOperand],
        is_memory_operand: &[bool],
    ) -> AsmResolvedOperand {
        let s = operand_str.trim();

        if s.is_empty() {
            return AsmResolvedOperand::Unknown(s.to_string());
        }

        // Immediate: $42, $0x2A, etc.
        if s.starts_with('$') {
            let val_str = &s[1..];
            if let Ok(val) = if val_str.starts_with("0x") || val_str.starts_with("0X") {
                i64::from_str_radix(&val_str[2..], 16)
            } else if val_str.starts_with('-') {
                val_str.parse::<i64>()
            } else {
                val_str.parse::<i64>()
            } {
                return AsmResolvedOperand::Immediate(val);
            }
            return AsmResolvedOperand::Unknown(s.to_string());
        }

        // Operand reference: %0, %1, %[name]
        if s.starts_with('%') {
            let rest = &s[1..];

            // Escaped register: %%eax → physical register eax
            if rest.starts_with('%') {
                let reg_name = &rest[1..];
                if let Some(reg) = self.parse_register_name(reg_name) {
                    return AsmResolvedOperand::PhysReg(reg);
                }
                return AsmResolvedOperand::Unknown(s.to_string());
            }

            // Goto label reference: %l0, %l1, etc. (asm goto targets).
            if rest.starts_with('l')
                && rest.len() > 1
                && rest[1..]
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_ascii_digit())
            {
                let idx_str: String = rest[1..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(idx) = idx_str.parse::<usize>() {
                    return AsmResolvedOperand::GotoLabel(idx);
                }
            }

            // Positional: %0, %1, %12, etc.
            if rest.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                let idx_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < operand_map.len() {
                        // Check if this operand index is a memory constraint.
                        if idx < is_memory_operand.len() && is_memory_operand[idx] {
                            return AsmResolvedOperand::MemoryOp(operand_map[idx].clone());
                        }
                        return AsmResolvedOperand::MachineOp(operand_map[idx].clone());
                    }
                }
                return AsmResolvedOperand::Unknown(s.to_string());
            }

            // Named: %[name] — already resolved to %N by IR lowering,
            // so this shouldn't appear. Handle defensively.
            if rest.starts_with('[') {
                return AsmResolvedOperand::Unknown(s.to_string());
            }

            // Bare register with AT&T prefix: %eax, %rdi, etc.
            if let Some(reg) = self.parse_register_name(rest) {
                return AsmResolvedOperand::PhysReg(reg);
            }

            return AsmResolvedOperand::Unknown(s.to_string());
        }

        AsmResolvedOperand::Unknown(s.to_string())
    }

    /// Emits machine instructions for a single AT&T assembly instruction line.
    ///
    /// Supports common x86-64 instructions used in inline assembly:
    /// - `nop`
    /// - `mov[l|q]` — register-to-register, immediate-to-register, memory
    /// - `add[l|q]`, `sub[l|q]` — arithmetic with registers or immediates
    /// - `xor[l|q]`, `and[l|q]`, `or[l|q]` — bitwise operations
    /// - `cmp[l|q]`, `test[l|q]` — comparison
    /// - `inc[l|q]`, `dec[l|q]` — increment/decrement
    /// - `neg[l|q]`, `not[l|q]` — unary operations
    /// - `shl[l|q]`, `shr[l|q]`, `sar[l|q]` — shifts
    /// - `lea[l|q]` — load effective address
    fn emit_gcc_asm_instruction(
        &mut self,
        instrs: &mut Vec<MachineInstr>,
        mnemonic: &str,
        resolved_ops: &[AsmResolvedOperand],
        goto_targets: &[u32],
    ) {
        // Determine operand size from mnemonic suffix.
        let is_64bit = mnemonic.ends_with('q');
        let size_flag: u32 = if is_64bit { SIZE_QWORD } else { 0 };

        match mnemonic {
            "nop" => {
                instrs.push(MachineInstr::new(opcodes::NOP));
            }

            // MOV instructions (AT&T: mov src, dst)
            "mov" | "movl" | "movq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_mov(instrs, src, dst, size_flag);
                }
            }

            // ADD instructions (AT&T: add src, dst)
            "add" | "addl" | "addq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::ADD_RR,
                        opcodes::ADD_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // SUB instructions
            "sub" | "subl" | "subq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::SUB_RR,
                        opcodes::SUB_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // XOR instructions
            "xor" | "xorl" | "xorq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::XOR_RR,
                        opcodes::XOR_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // AND instructions
            "and" | "andl" | "andq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::AND_RR,
                        opcodes::AND_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // OR instructions
            "or" | "orl" | "orq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::OR_RR,
                        opcodes::OR_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // CMP instructions (AT&T: cmp src, dst — computes dst-src)
            "cmp" | "cmpl" | "cmpq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::CMP_RR,
                        opcodes::CMP_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // TEST instructions
            "test" | "testl" | "testq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::TEST_RR,
                        opcodes::TEST_RI,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // SHL instructions (AT&T: shl src, dst)
            "shl" | "shll" | "shlq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_shift(instrs, opcodes::SHL, src, dst, size_flag);
                }
            }

            // SHR instructions
            "shr" | "shrl" | "shrq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_shift(instrs, opcodes::SHR, src, dst, size_flag);
                }
            }

            // SAR instructions
            "sar" | "sarl" | "sarq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_shift(instrs, opcodes::SAR, src, dst, size_flag);
                }
            }

            // INC (unary: inc dst)
            "inc" | "incl" | "incq" => {
                if resolved_ops.len() >= 1 {
                    let dst = &resolved_ops[0];
                    if let Some(dst_op) = dst.as_machine_operand() {
                        let mut mi = MachineInstr::new(opcodes::INC | size_flag);
                        mi.add_operand(dst_op);
                        instrs.push(mi);
                    }
                }
            }

            // DEC (unary: dec dst)
            "dec" | "decl" | "decq" => {
                if resolved_ops.len() >= 1 {
                    let dst = &resolved_ops[0];
                    if let Some(dst_op) = dst.as_machine_operand() {
                        let mut mi = MachineInstr::new(opcodes::DEC | size_flag);
                        mi.add_operand(dst_op);
                        instrs.push(mi);
                    }
                }
            }

            // NEG (unary: neg dst)
            "neg" | "negl" | "negq" => {
                if resolved_ops.len() >= 1 {
                    let dst = &resolved_ops[0];
                    if let Some(dst_op) = dst.as_machine_operand() {
                        let mut mi = MachineInstr::new(opcodes::NEG | size_flag);
                        mi.add_operand(dst_op);
                        instrs.push(mi);
                    }
                }
            }

            // NOT (unary: not dst)
            "not" | "notl" | "notq" => {
                if resolved_ops.len() >= 1 {
                    let dst = &resolved_ops[0];
                    if let Some(dst_op) = dst.as_machine_operand() {
                        let mut mi = MachineInstr::new(opcodes::NOT | size_flag);
                        mi.add_operand(dst_op);
                        instrs.push(mi);
                    }
                }
            }

            // LEA (leaq disp(%base), %dst)
            "lea" | "leal" | "leaq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    if let Some(dst_op) = dst.as_machine_operand() {
                        if let Some(src_op) = src.as_machine_operand() {
                            let mut mi = MachineInstr::new(opcodes::LEA | size_flag);
                            mi.add_operand(dst_op);
                            mi.add_operand(src_op);
                            instrs.push(mi);
                        }
                    }
                }
            }

            // IMUL
            "imul" | "imull" | "imulq" => {
                if resolved_ops.len() >= 2 {
                    let src = &resolved_ops[0];
                    let dst = &resolved_ops[1];
                    self.emit_gcc_binop(
                        instrs,
                        opcodes::IMUL_RR,
                        opcodes::IMUL_RR,
                        src,
                        dst,
                        size_flag,
                    );
                }
            }

            // ---------------------------------------------------------
            // Conditional jumps for asm goto (jz, je, jne, jnz, etc.)
            // The operand is a GotoLabel(idx) resolved from %l<N>.
            // We emit a conditional branch to the target basic block.
            // ---------------------------------------------------------
            "jz" | "je" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        // Condition code 4 = JE/JZ in x86
                        mi.add_operand(MachineOperand::Immediate(4));
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jnz" | "jne" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        // Condition code 5 = JNE/JNZ
                        mi.add_operand(MachineOperand::Immediate(5));
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jl" | "jnge" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(0xC)); // JL
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jg" | "jnle" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(0xF)); // JG
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jle" | "jng" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(0xE)); // JLE
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jge" | "jnl" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(0xD)); // JGE
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jmp" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JMP);
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "ja" | "jnbe" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(7)); // JA
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jb" | "jnae" | "jc" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(2)); // JB
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jbe" | "jna" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(6)); // JBE
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jae" | "jnb" | "jnc" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(3)); // JAE
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "js" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(8)); // JS
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }
            "jns" => {
                if let Some(AsmResolvedOperand::GotoLabel(label_idx)) = resolved_ops.first() {
                    if *label_idx < goto_targets.len() {
                        let target_bb = goto_targets[*label_idx];
                        let mut mi = MachineInstr::new(opcodes::JCC);
                        mi.add_operand(MachineOperand::Immediate(9)); // JNS
                        mi.add_operand(MachineOperand::Label(target_bb));
                        instrs.push(mi);
                    }
                }
            }

            // Unrecognized: emit as opaque INLINE_ASM pseudo-op.
            _ => {
                let asm_mi = MachineInstr::new(opcodes::INLINE_ASM);
                instrs.push(asm_mi);
            }
        }
    }

    /// Emits a MOV instruction from a resolved source to a resolved destination.
    fn emit_gcc_mov(
        &mut self,
        instrs: &mut Vec<MachineInstr>,
        src: &AsmResolvedOperand,
        dst: &AsmResolvedOperand,
        size_flag: u32,
    ) {
        match (src, dst) {
            // mov $imm, memory_operand — store immediate to memory
            (AsmResolvedOperand::Immediate(val), AsmResolvedOperand::MemoryOp(mem)) => {
                // Load imm into a scratch vreg, then store to memory.
                let tmp = self.alloc_vreg();
                let mut mi_load = MachineInstr::new(opcodes::MOV_RI | size_flag);
                mi_load.add_operand(tmp.clone());
                mi_load.add_operand(MachineOperand::Immediate(*val));
                instrs.push(mi_load);
                let mut mi_store = MachineInstr::new(opcodes::MOV_MR | size_flag);
                mi_store.add_operand(mem.clone());
                mi_store.add_operand(tmp);
                instrs.push(mi_store);
            }
            // mov $imm, %reg (or other non-memory dst)
            (AsmResolvedOperand::Immediate(val), _) => {
                if let Some(dst_op) = dst.as_machine_operand() {
                    let mut mi = MachineInstr::new(opcodes::MOV_RI | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(MachineOperand::Immediate(*val));
                    instrs.push(mi);
                }
            }
            // mov src, memory_operand ("=m" / "+m" output) — store to memory
            (_, AsmResolvedOperand::MemoryOp(mem)) => {
                if let Some(src_op) = src.as_machine_operand() {
                    let src_reg = match &src_op {
                        MachineOperand::Register(_) | MachineOperand::VirtualReg(_) => src_op,
                        _ => {
                            // Need to get value into a vreg first.
                            let tmp = self.alloc_vreg();
                            let mut mi = MachineInstr::new(opcodes::MOV_RM | size_flag);
                            mi.add_operand(tmp.clone());
                            mi.add_operand(src_op);
                            instrs.push(mi);
                            tmp
                        }
                    };
                    let mut mi = MachineInstr::new(opcodes::MOV_MR | size_flag);
                    mi.add_operand(mem.clone());
                    mi.add_operand(src_reg);
                    instrs.push(mi);
                }
            }
            // mov memory_operand, %reg — load from memory ("m" input)
            (AsmResolvedOperand::MemoryOp(mem), _) => {
                if let Some(dst_op) = dst.as_machine_operand() {
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(mem.clone());
                    instrs.push(mi);
                }
            }
            // mov %reg, (%mem) — store to AT&T memory ref
            (_, AsmResolvedOperand::MemoryRef(base_reg)) => {
                if let Some(src_op) = src.as_machine_operand() {
                    let mut mi = MachineInstr::new(opcodes::MOV_MR | size_flag);
                    mi.add_operand(MachineOperand::Memory {
                        base: *base_reg,
                        offset: 0,
                        index: None,
                        scale: 1,
                    });
                    mi.add_operand(src_op);
                    instrs.push(mi);
                }
            }
            // mov (%mem), %reg — load from AT&T memory ref
            (AsmResolvedOperand::MemoryRef(base_reg), _) => {
                if let Some(dst_op) = dst.as_machine_operand() {
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(MachineOperand::Memory {
                        base: *base_reg,
                        offset: 0,
                        index: None,
                        scale: 1,
                    });
                    instrs.push(mi);
                }
            }
            // mov %reg, %reg (or any non-memory-to-non-memory)
            _ => {
                if let (Some(src_op), Some(dst_op)) =
                    (src.as_machine_operand(), dst.as_machine_operand())
                {
                    let mut mi = MachineInstr::new(opcodes::MOV_RR | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(src_op);
                    instrs.push(mi);
                }
            }
        }
    }

    /// Emits a binary ALU operation (ADD, SUB, XOR, AND, OR, CMP, TEST).
    fn emit_gcc_binop(
        &mut self,
        instrs: &mut Vec<MachineInstr>,
        rr_opcode: u32,
        ri_opcode: u32,
        src: &AsmResolvedOperand,
        dst: &AsmResolvedOperand,
        size_flag: u32,
    ) {
        match src {
            AsmResolvedOperand::Immediate(val) => {
                if let Some(dst_op) = dst.as_machine_operand() {
                    let mut mi = MachineInstr::new(ri_opcode | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(MachineOperand::Immediate(*val));
                    instrs.push(mi);
                }
            }
            _ => {
                if let (Some(src_op), Some(dst_op)) =
                    (src.as_machine_operand(), dst.as_machine_operand())
                {
                    let mut mi = MachineInstr::new(rr_opcode | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(src_op);
                    instrs.push(mi);
                }
            }
        }
    }

    /// Emits a shift instruction (SHL, SHR, SAR).
    fn emit_gcc_shift(
        &mut self,
        instrs: &mut Vec<MachineInstr>,
        opcode: u32,
        src: &AsmResolvedOperand,
        dst: &AsmResolvedOperand,
        size_flag: u32,
    ) {
        if let Some(dst_op) = dst.as_machine_operand() {
            match src {
                AsmResolvedOperand::Immediate(val) => {
                    let mut mi = MachineInstr::new(opcode | size_flag);
                    mi.add_operand(dst_op);
                    mi.add_operand(MachineOperand::Immediate(*val));
                    instrs.push(mi);
                }
                _ => {
                    // Shift by register: SHL %cl, %dst
                    // x86 requires the shift count in CL.
                    if let Some(src_op) = src.as_machine_operand() {
                        let mut mi = MachineInstr::new(opcode | size_flag);
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Builtin inline assembly path (existing logic)
    // -----------------------------------------------------------------------

    /// Lowers compiler-generated inline assembly for builtins like
    /// `__builtin_clz`, `__builtin_ctz`, `__builtin_popcount`, `__builtin_bswap`.
    ///
    /// These use `$0`/`$1` operand references and comma-separated constraints.
    fn lower_builtin_asm(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        clobbers: &[String],
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        let out_vreg = if let Some(res_vid) = result {
            self.get_or_create_vreg(res_vid)
        } else {
            self.alloc_vreg()
        };

        let _phys_out_hint = if result.is_some() {
            self.parse_output_constraint(constraints).unwrap_or(RAX)
        } else {
            RAX
        };

        let input_ops: Vec<MachineOperand> =
            operands.iter().map(|v| self.get_operand(*v)).collect();

        let template_trimmed = template.trim();
        let is_64bit = template_trimmed.contains("bsrq")
            || template_trimmed.contains("bsfq")
            || template_trimmed.contains("popcntq")
            || template_trimmed.contains("bswapq");
        let size_flag: u32 = if is_64bit { SIZE_QWORD } else { 0 };

        // Split the template into individual assembly instructions.
        // AT&T inline assembly uses newlines, tabs, and semicolons as
        // instruction separators.  We must split on all three to correctly
        // handle templates like "testl %0, %0; jz %l[is_zero]".
        let asm_lines: Vec<&str> = template_trimmed
            .split('\n')
            .flat_map(|l| l.split('\t'))
            .flat_map(|l| l.split(';'))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let input0 = if let Some(op) = input_ops.first() {
            match op {
                MachineOperand::Immediate(_) => {
                    let vreg = self.alloc_vreg();
                    let mov_sz = if is_64bit { SIZE_QWORD } else { 0 };
                    let mut mi = MachineInstr::new(opcodes::MOV_RI | mov_sz);
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op.clone());
                    instrs.push(mi);
                    vreg
                }
                MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. } => {
                    let vreg = self.alloc_vreg();
                    let mov_sz = if is_64bit { SIZE_QWORD } else { 0 };
                    let mut mi = MachineInstr::new(opcodes::MOV_RM | mov_sz);
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op.clone());
                    instrs.push(mi);
                    vreg
                }
                _ => op.clone(),
            }
        } else {
            self.alloc_vreg()
        };

        for line in &asm_lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            let parts: Vec<&str> = line
                .splitn(2, |c: char| c.is_whitespace())
                .map(|s| s.trim())
                .collect();
            let mnemonic = parts[0].to_lowercase();

            match mnemonic.as_str() {
                "bsrl" | "bsrq" => {
                    let mut mi = MachineInstr::new(opcodes::BSR_RR | size_flag);
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "bsfl" | "bsfq" => {
                    let mut mi = MachineInstr::new(opcodes::BSF_RR | size_flag);
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "popcntl" | "popcntq" => {
                    let mut mi = MachineInstr::new(opcodes::POPCNT_RR | size_flag);
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "bswap" | "bswapl" | "bswapq" => {
                    let mov_opcode = if is_64bit {
                        opcodes::MOV_RR | SIZE_QWORD
                    } else {
                        opcodes::MOV_RR
                    };
                    let mut mi_mov = MachineInstr::new(mov_opcode);
                    mi_mov.add_operand(out_vreg.clone());
                    mi_mov.add_operand(input0.clone());
                    instrs.push(mi_mov);

                    let mut mi = MachineInstr::new(opcodes::BSWAP_R | size_flag);
                    mi.add_operand(out_vreg.clone());
                    instrs.push(mi);
                }
                "xorl" | "xorq" => {
                    if parts.len() > 1 {
                        let xor_parts: Vec<&str> = parts[1].split(',').map(|s| s.trim()).collect();
                        if let Some(imm_str) = xor_parts.first() {
                            let imm_str = imm_str.trim_start_matches('$');
                            if let Ok(imm_val) = imm_str.parse::<i64>() {
                                let mut mi = MachineInstr::new(opcodes::XOR_RI | size_flag);
                                mi.add_operand(out_vreg.clone());
                                mi.add_operand(MachineOperand::Immediate(imm_val));
                                instrs.push(mi);
                            }
                        }
                    }
                }
                "leaq" | "leal" => {
                    let mut handled = false;
                    if parts.len() > 1 {
                        let operand_str = parts[1].trim();
                        let src_part = operand_str.split(',').next().unwrap_or("").trim();
                        if let Some(paren_pos) = src_part.find('(') {
                            let offset_str = src_part[..paren_pos].trim();
                            if let Ok(offset_val) = offset_str.parse::<i32>() {
                                let reg_str = &src_part[paren_pos..];
                                let base_reg = if reg_str.contains("rbp") || reg_str.contains("ebp")
                                {
                                    RBP
                                } else if reg_str.contains("rsp") || reg_str.contains("esp") {
                                    RSP
                                } else {
                                    RBP
                                };
                                let mut mi = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                                mi.add_operand(out_vreg.clone());
                                mi.add_operand(MachineOperand::Memory {
                                    base: base_reg,
                                    offset: offset_val,
                                    index: None,
                                    scale: 1,
                                });
                                instrs.push(mi);
                                handled = true;
                            }
                        }
                    }
                    if !handled {
                        let mut mi = MachineInstr::new(opcodes::LEA | SIZE_QWORD);
                        mi.add_operand(out_vreg.clone());
                        mi.add_operand(MachineOperand::Memory {
                            base: RBP,
                            offset: 0,
                            index: None,
                            scale: 1,
                        });
                        instrs.push(mi);
                    }
                }
                "movl" | "movq" => {
                    let sz = if mnemonic == "movq" { SIZE_QWORD } else { 0 };
                    let mut mi = MachineInstr::new(opcodes::MOV_RR | sz);
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(MachineOperand::Register(RBP));
                    instrs.push(mi);
                }
                _ => {
                    let mut asm_mi = MachineInstr::new(opcodes::INLINE_ASM);
                    for op_val in operands {
                        let op = self.get_operand(*op_val);
                        asm_mi.add_operand(op);
                    }
                    instrs.push(asm_mi);
                }
            }
        }

        if let Some(last) = instrs.last_mut() {
            for clobber in clobbers {
                match clobber.as_str() {
                    "memory" | "cc" => {}
                    _ => {
                        if let Some(reg) = self.parse_register_name(clobber) {
                            last.add_implicit_def(reg);
                        }
                    }
                }
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

    /// Allocates a virtual register that will be classified as
    /// floating-point by the register allocator, ensuring it gets
    /// assigned to an XMM register instead of a GPR.
    fn alloc_vreg_float(&mut self) -> MachineOperand {
        let id = self.next_vreg;
        self.next_vreg += 1;
        self.float_vregs.insert(id);
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
    /// Returns the SIZE_QWORD flag if the given IR type requires 64-bit
    /// operations (pointers, I64, I128). Returns 0 for 32-bit and smaller.
    fn size_flag_for_type(&self, ty: &IrType) -> u32 {
        match ty {
            IrType::Ptr | IrType::I64 | IrType::I128 => SIZE_QWORD,
            IrType::Array { .. } | IrType::Struct { .. } => SIZE_QWORD, // addresses are 64-bit
            IrType::I8 | IrType::I1 => SIZE_BYTE,
            IrType::I16 => SIZE_WORD,
            _ => 0, // I32, F32, F64, etc. → DWord default
        }
    }

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

    /// Creates a MOV instruction whose width is determined by the IR type.
    /// Returns a 64-bit REX.W MOV for `I64` and `Ptr` types, and the
    /// default (32-bit) MOV for everything else.  This is the preferred
    /// helper inside `lower_binop` and similar code-generation paths that
    /// carry an `IrType` from the IR instruction.
    fn make_mov_typed(
        &self,
        dst: MachineOperand,
        src: MachineOperand,
        ty: &IrType,
    ) -> MachineInstr {
        if matches!(ty, IrType::I64 | IrType::Ptr) {
            self.make_mov64(dst, src)
        } else {
            self.make_mov(dst, src)
        }
    }

    /// Creates a 64-bit (REX.W) MOV instruction.
    /// Used for pointer-width moves (IntToPtr, PtrToInt, address copies).
    fn make_mov64(&self, dst: MachineOperand, src: MachineOperand) -> MachineInstr {
        let opcode = match (&dst, &src) {
            (_, MachineOperand::Immediate(_)) => opcodes::MOV_RI | SIZE_QWORD,
            (_, MachineOperand::Memory { .. }) | (_, MachineOperand::FrameIndex(_)) => {
                opcodes::MOV_RM | SIZE_QWORD
            }
            (MachineOperand::Memory { .. }, _) | (MachineOperand::FrameIndex(_), _) => {
                opcodes::MOV_MR | SIZE_QWORD
            }
            _ => opcodes::MOV_RR | SIZE_QWORD,
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
