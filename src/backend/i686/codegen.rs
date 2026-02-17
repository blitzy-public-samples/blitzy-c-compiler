//! i686 instruction selection and emission module for the BCC compiler.
//!
//! This module translates IR instructions into i686 (IA-32) machine instructions.
//! It implements the core 32-bit x86 code generation logic with the following
//! architectural constraints:
//!
//! - **8 general-purpose registers** (EAX–EDI) — no R8–R15, no REX prefix
//! - **x87 FPU** for all floating-point operations (stack-based model)
//! - **cdecl calling convention** — all parameters passed on the stack
//! - **Complex addressing modes** — base+index*scale+displacement via SIB
//! - **64-bit integer operations** via register pairs (EAX:EDX)
//!
//! # Instruction Selection Strategy
//!
//! Each IR instruction is pattern-matched and lowered to one or more i686
//! machine instructions. The instruction selector operates on a per-function
//! basis, maintaining a mapping from IR [`ValueId`]s to [`MachineOperand`]s
//! (tracking where each SSA value is materialized).
//!
//! # PIC Support
//!
//! Position-independent code on i686 uses a GOT-base register (EBX) obtained
//! via a `__i686.get_pc_thunk.bx` call. Global accesses go through
//! `symbol@GOT` (indirect) or `symbol@GOTOFF` (local).

use crate::backend::i686::abi::I686Abi;
use crate::backend::i686::registers::{
    self, parse_i686_reg_name, reg_name_32, AL, CALLEE_SAVED, CALLER_SAVED, CL, EAX, EBP, EBX, ECX,
    EDI, EDX, EFLAGS, ESI, ESP, ST0,
};
use crate::backend::traits::{
    CodegenConfig, MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::FxHashMap;
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction, ValueId};
use crate::ir::types::IrType;

// ===========================================================================
// I686Opcode — architecture-specific opcode identifiers
// ===========================================================================

/// i686 machine instruction opcodes.
///
/// Each variant maps to one (or a family of) x86 instructions. The numeric
/// value (obtained via `as u32`) is stored in [`MachineInstr::opcode`] and
/// decoded by the assembler/encoder for binary emission.
///
/// Naming follows Intel syntax mnemonics. Variants are grouped by category:
/// data movement, arithmetic, logic, shifts, multiplication/division,
/// control flow, x87 FPU, and miscellaneous.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum I686Opcode {
    // -- Data movement --
    /// MOV dst, src — general-purpose 32-bit register/memory/immediate move.
    Mov = 0,
    /// MOVSX dst, src — sign-extending move (8→32 or 16→32).
    MovSx = 1,
    /// MOVZX dst, src — zero-extending move (8→32 or 16→32).
    MovZx = 2,
    /// LEA dst, [addr] — load effective address (address computation).
    Lea = 3,
    /// PUSH src — push 32-bit value onto stack (ESP -= 4).
    Push = 4,
    /// POP dst — pop 32-bit value from stack (ESP += 4).
    Pop = 5,
    /// MOV byte [mem], r8 — 8-bit store (uses opcode 0x88 / 0xC6).
    MovByte = 6,
    /// MOV word [mem], r16 — 16-bit store (uses 0x66 prefix + opcode 0x89 / 0xC7).
    MovWord = 7,

    // -- Integer arithmetic --
    /// ADD dst, src — integer addition.
    Add = 10,
    /// ADC dst, src — add with carry (for 64-bit add on 32-bit).
    Adc = 11,
    /// SUB dst, src — integer subtraction.
    Sub = 12,
    /// SBB dst, src — subtract with borrow (for 64-bit sub on 32-bit).
    Sbb = 13,
    /// AND dst, src — bitwise AND.
    And = 14,
    /// OR dst, src — bitwise OR.
    Or = 15,
    /// XOR dst, src — bitwise XOR.
    Xor = 16,
    /// IMUL dst, src (2-operand signed multiply) or IMUL dst, src, imm (3-operand).
    Imul = 17,
    /// IDIV src — signed divide EDX:EAX by src; quotient→EAX, remainder→EDX.
    Idiv = 18,
    /// DIV src — unsigned divide EDX:EAX by src; quotient→EAX, remainder→EDX.
    Div = 19,
    /// MUL src — unsigned multiply EAX * src → EDX:EAX.
    Mul = 20,
    /// NEG dst — two's complement negate.
    Neg = 21,
    /// NOT dst — bitwise complement.
    Not = 22,
    /// INC dst — increment by one.
    Inc = 23,
    /// DEC dst — decrement by one.
    Dec = 24,

    // -- Shifts --
    /// SHL dst, count — logical left shift.
    Shl = 30,
    /// SHR dst, count — logical right shift (zero fill).
    Shr = 31,
    /// SAR dst, count — arithmetic right shift (sign fill).
    Sar = 32,
    /// SHLD dst, src, count — double-precision left shift (for 64-bit shifts).
    Shld = 33,
    /// SHRD dst, src, count — double-precision right shift (for 64-bit shifts).
    Shrd = 34,

    // -- Sign extension --
    /// CDQ — sign-extend EAX into EDX:EAX (before IDIV).
    Cdq = 40,

    // -- Comparison and test --
    /// CMP lhs, rhs — compare (sets EFLAGS: ZF, CF, SF, OF).
    Cmp = 50,
    /// TEST lhs, rhs — bitwise AND, discard result (sets EFLAGS).
    Test = 51,

    // -- Control flow --
    /// JMP target — unconditional jump.
    Jmp = 60,
    /// Jcc target — conditional jump (condition encoded in operand/cc field).
    Jcc = 61,
    /// SETcc dst — set byte to 0/1 based on condition code.
    Setcc = 62,
    /// CMOVcc dst, src — conditional move.
    Cmovcc = 63,
    /// CALL target — direct or indirect function call.
    Call = 64,
    /// RET — return from function.
    Ret = 65,

    // -- x87 FPU --
    /// FLD mem — load float/double from memory onto x87 stack.
    Fld = 70,
    /// FST mem — store x87 ST(0) to memory (without pop).
    Fst = 71,
    /// FSTP mem — store x87 ST(0) to memory and pop.
    Fstp = 72,
    /// FADD src — ST(0) += src.
    Fadd = 73,
    /// FADDP — ST(1) += ST(0), pop.
    Faddp = 74,
    /// FSUB src — ST(0) -= src.
    Fsub = 75,
    /// FSUBP — ST(1) -= ST(0), pop (or variant).
    Fsubp = 76,
    /// FMUL src — ST(0) *= src.
    Fmul = 77,
    /// FMULP — ST(1) *= ST(0), pop.
    Fmulp = 78,
    /// FDIV src — ST(0) /= src.
    Fdiv = 79,
    /// FDIVP — ST(1) /= ST(0), pop.
    Fdivp = 80,
    /// FCHS — negate ST(0).
    Fchs = 81,
    /// FABS — absolute value of ST(0).
    Fabs = 82,
    /// FCOM src — compare ST(0) with src (sets x87 status word).
    Fcom = 83,
    /// FCOMP src — compare ST(0) with src, pop.
    Fcomp = 84,
    /// FCOMPP — compare ST(0) with ST(1), pop both.
    Fcompp = 85,
    /// FUCOMIP ST(0), ST(i) — unordered compare, pop, set EFLAGS.
    Fucomip = 86,
    /// FILD mem — load integer from memory, convert to x87 float, push.
    Fild = 87,
    /// FISTP mem — store ST(0) as integer to memory, pop.
    Fistp = 88,
    /// FXCH ST(i) — exchange ST(0) and ST(i).
    Fxch = 89,

    // -- Register-indirect addressing --
    /// MOV dst, [reg+offset] — load through pointer in register.
    ///
    /// Unlike `Mov` with a `Memory` operand, this form keeps the pointer
    /// address in a `VirtualReg` / `Register` operand that the register
    /// allocator can rewrite. The encoder interprets operand[1] as
    /// `[reg + optional_offset]`.
    ///
    /// Operands: `[Register(dst), Register(addr)]`
    ///           or `[Register(dst), Register(addr), Immediate(offset)]`
    MovLoad = 250,
    /// MOV [reg+offset], src — store through pointer in register.
    ///
    /// Operands: `[Register(addr), Register(src)]`
    ///           or `[Register(addr), Register(src), Immediate(offset)]`
    MovStore = 251,
    /// LEA dst, [mem] — load effective address from Memory operand.
    ///
    /// Used by GEP to compute addresses of stack variables without
    /// loading the value.
    LeaMem = 252,

    // -- Bit manipulation --
    /// BSR dst, src — Bit Scan Reverse (find highest set bit).
    Bsr = 200,
    /// BSF dst, src — Bit Scan Forward (find lowest set bit).
    Bsf = 201,
    /// POPCNT dst, src — Population Count (count set bits).
    Popcnt = 202,
    /// BSWAP reg — Byte Swap (reverse byte order of 32-bit register).
    Bswap = 203,

    // -- Block address / indirect branch (computed goto) --
    /// LEA dst, label — load effective address of a basic-block label.
    LeaLabel = 253,
    /// JMP *reg — indirect jump through a register (computed goto).
    JmpIndirect = 254,

    // -- Misc --
    /// NOP — no operation (for alignment padding).
    Nop = 255,
}

impl I686Opcode {
    /// Converts the opcode to its `u32` encoding for [`MachineInstr::opcode`].
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Attempts to decode a `u32` opcode back to an `I686Opcode`.
    ///
    /// Returns `None` if the value does not correspond to a known opcode.
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::Mov),
            1 => Some(Self::MovSx),
            2 => Some(Self::MovZx),
            3 => Some(Self::Lea),
            4 => Some(Self::Push),
            5 => Some(Self::Pop),
            6 => Some(Self::MovByte),
            7 => Some(Self::MovWord),
            10 => Some(Self::Add),
            11 => Some(Self::Adc),
            12 => Some(Self::Sub),
            13 => Some(Self::Sbb),
            14 => Some(Self::And),
            15 => Some(Self::Or),
            16 => Some(Self::Xor),
            17 => Some(Self::Imul),
            18 => Some(Self::Idiv),
            19 => Some(Self::Div),
            20 => Some(Self::Mul),
            21 => Some(Self::Neg),
            22 => Some(Self::Not),
            23 => Some(Self::Inc),
            24 => Some(Self::Dec),
            30 => Some(Self::Shl),
            31 => Some(Self::Shr),
            32 => Some(Self::Sar),
            33 => Some(Self::Shld),
            34 => Some(Self::Shrd),
            40 => Some(Self::Cdq),
            50 => Some(Self::Cmp),
            51 => Some(Self::Test),
            60 => Some(Self::Jmp),
            61 => Some(Self::Jcc),
            62 => Some(Self::Setcc),
            63 => Some(Self::Cmovcc),
            64 => Some(Self::Call),
            65 => Some(Self::Ret),
            70 => Some(Self::Fld),
            71 => Some(Self::Fst),
            72 => Some(Self::Fstp),
            73 => Some(Self::Fadd),
            74 => Some(Self::Faddp),
            75 => Some(Self::Fsub),
            76 => Some(Self::Fsubp),
            77 => Some(Self::Fmul),
            78 => Some(Self::Fmulp),
            79 => Some(Self::Fdiv),
            80 => Some(Self::Fdivp),
            81 => Some(Self::Fchs),
            82 => Some(Self::Fabs),
            83 => Some(Self::Fcom),
            84 => Some(Self::Fcomp),
            85 => Some(Self::Fcompp),
            86 => Some(Self::Fucomip),
            87 => Some(Self::Fild),
            88 => Some(Self::Fistp),
            89 => Some(Self::Fxch),
            200 => Some(Self::Bsr),
            201 => Some(Self::Bsf),
            202 => Some(Self::Popcnt),
            203 => Some(Self::Bswap),
            250 => Some(Self::MovLoad),
            251 => Some(Self::MovStore),
            252 => Some(Self::LeaMem),
            253 => Some(Self::LeaLabel),
            254 => Some(Self::JmpIndirect),
            255 => Some(Self::Nop),
            _ => None,
        }
    }
}

impl std::fmt::Display for I686Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Mov => "mov",
            Self::MovSx => "movsx",
            Self::MovZx => "movzx",
            Self::Lea => "lea",
            Self::Push => "push",
            Self::Pop => "pop",
            Self::MovByte => "movb",
            Self::MovWord => "movw",
            Self::Add => "add",
            Self::Adc => "adc",
            Self::Sub => "sub",
            Self::Sbb => "sbb",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Imul => "imul",
            Self::Idiv => "idiv",
            Self::Div => "div",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Not => "not",
            Self::Inc => "inc",
            Self::Dec => "dec",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Sar => "sar",
            Self::Shld => "shld",
            Self::Shrd => "shrd",
            Self::Cdq => "cdq",
            Self::Cmp => "cmp",
            Self::Test => "test",
            Self::Jmp => "jmp",
            Self::Jcc => "jcc",
            Self::Setcc => "setcc",
            Self::Cmovcc => "cmovcc",
            Self::Call => "call",
            Self::Ret => "ret",
            Self::Fld => "fld",
            Self::Fst => "fst",
            Self::Fstp => "fstp",
            Self::Fadd => "fadd",
            Self::Faddp => "faddp",
            Self::Fsub => "fsub",
            Self::Fsubp => "fsubp",
            Self::Fmul => "fmul",
            Self::Fmulp => "fmulp",
            Self::Fdiv => "fdiv",
            Self::Fdivp => "fdivp",
            Self::Fchs => "fchs",
            Self::Fabs => "fabs",
            Self::Fcom => "fcom",
            Self::Fcomp => "fcomp",
            Self::Fcompp => "fcompp",
            Self::Fucomip => "fucomip",
            Self::Fild => "fild",
            Self::Fistp => "fistp",
            Self::Fxch => "fxch",
            Self::Bsr => "bsr",
            Self::Bsf => "bsf",
            Self::Popcnt => "popcnt",
            Self::Bswap => "bswap",
            Self::MovLoad => "movload",
            Self::MovStore => "movstore",
            Self::LeaMem => "leamem",
            Self::LeaLabel => "lealabel",
            Self::JmpIndirect => "jmpindirect",
            Self::Nop => "nop",
        };
        f.write_str(name)
    }
}

// ===========================================================================
// ConditionCode — x86 condition codes for Jcc / SETcc / CMOVcc
// ===========================================================================

/// x86 condition codes used with Jcc, SETcc, and CMOVcc instruction families.
///
/// Each variant maps to a specific test on the EFLAGS register bits (CF, ZF,
/// SF, OF, PF). The 4-bit encoding (0x0–0xF) matches the Intel ISA encoding
/// used in the opcode secondary byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConditionCode {
    /// OF=1 — overflow.
    O = 0x0,
    /// OF=0 — no overflow.
    No = 0x1,
    /// CF=1 — below / carry (unsigned less than).
    B = 0x2,
    /// CF=0 — above or equal / no carry (unsigned greater than or equal).
    Ae = 0x3,
    /// ZF=1 — equal / zero.
    E = 0x4,
    /// ZF=0 — not equal / not zero.
    Ne = 0x5,
    /// CF=1 or ZF=1 — below or equal (unsigned less than or equal).
    Be = 0x6,
    /// CF=0 and ZF=0 — above (unsigned greater than).
    A = 0x7,
    /// SF=1 — sign (negative).
    S = 0x8,
    /// SF=0 — no sign (non-negative).
    Ns = 0x9,
    /// PF=1 — parity even.
    P = 0xA,
    /// PF=0 — parity odd.
    Np = 0xB,
    /// SF≠OF — less than (signed).
    L = 0xC,
    /// SF=OF — greater than or equal (signed).
    Ge = 0xD,
    /// ZF=1 or SF≠OF — less than or equal (signed).
    Le = 0xE,
    /// ZF=0 and SF=OF — greater than (signed).
    G = 0xF,
}

impl ConditionCode {
    /// Maps an IR integer comparison predicate to the corresponding x86
    /// condition code.
    ///
    /// This is the primary bridge between the IR comparison semantics and
    /// the x86 conditional instruction encoding.
    pub fn from_icmp_predicate(pred: &ICmpPredicate) -> Self {
        match pred {
            ICmpPredicate::Eq => ConditionCode::E,
            ICmpPredicate::Ne => ConditionCode::Ne,
            ICmpPredicate::Slt => ConditionCode::L,
            ICmpPredicate::Sle => ConditionCode::Le,
            ICmpPredicate::Sgt => ConditionCode::G,
            ICmpPredicate::Sge => ConditionCode::Ge,
            ICmpPredicate::Ult => ConditionCode::B,
            ICmpPredicate::Ule => ConditionCode::Be,
            ICmpPredicate::Ugt => ConditionCode::A,
            ICmpPredicate::Uge => ConditionCode::Ae,
        }
    }

    /// Returns the logically inverted condition code.
    ///
    /// Inversion flips the lowest bit of the encoding, which is the
    /// standard x86 convention (e.g., JE ↔ JNE, JL ↔ JGE).
    #[inline]
    pub fn invert(self) -> Self {
        let raw = self as u8;
        // XOR with 1 flips the condition sense per x86 convention.
        Self::from_u8(raw ^ 1)
    }

    /// Converts a raw `u8` encoding back to a `ConditionCode`.
    ///
    /// # Panics
    /// Panics if `val > 0xF`.
    fn from_u8(val: u8) -> Self {
        match val {
            0x0 => Self::O,
            0x1 => Self::No,
            0x2 => Self::B,
            0x3 => Self::Ae,
            0x4 => Self::E,
            0x5 => Self::Ne,
            0x6 => Self::Be,
            0x7 => Self::A,
            0x8 => Self::S,
            0x9 => Self::Ns,
            0xA => Self::P,
            0xB => Self::Np,
            0xC => Self::L,
            0xD => Self::Ge,
            0xE => Self::Le,
            0xF => Self::G,
            _ => panic!("ConditionCode::from_u8: invalid value {:#x}", val),
        }
    }

    /// Returns the raw 4-bit encoding suitable for instruction encoding.
    #[inline]
    pub fn encoding(self) -> u8 {
        self as u8
    }
}

impl std::fmt::Display for ConditionCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::O => "o",
            Self::No => "no",
            Self::B => "b",
            Self::Ae => "ae",
            Self::E => "e",
            Self::Ne => "ne",
            Self::Be => "be",
            Self::A => "a",
            Self::S => "s",
            Self::Ns => "ns",
            Self::P => "p",
            Self::Np => "np",
            Self::L => "l",
            Self::Ge => "ge",
            Self::Le => "le",
            Self::G => "g",
        };
        f.write_str(s)
    }
}

// ===========================================================================
// I686InstrSel — instruction selection engine
// ===========================================================================

/// The i686 instruction selection engine.
///
/// Translates an entire [`IrFunction`] into a [`MachineFunction`] containing
/// i686-specific [`MachineInstr`]s. The selector maintains state for:
///
/// - **Value map**: IR `ValueId` → `MachineOperand` (where each SSA value lives)
/// - **Block map**: IR `BasicBlockId` → machine block ID (label resolution)
/// - **Frame index counter**: tracks stack slot allocation for spills/locals
/// - **Configuration**: PIC mode, target info, optimization level
///
/// # Usage
///
/// ```ignore
/// let mut sel = I686InstrSel::new(&config, &diag_engine);
/// let machine_func = sel.select_function(&ir_function);
/// ```
pub struct I686InstrSel<'a> {
    /// Code generation configuration (target, PIC, debug flags, etc.).
    config: &'a CodegenConfig,

    /// Diagnostic engine for reporting codegen errors.
    /// Not yet referenced in all code paths; retained for complete
    /// error reporting integration.
    #[allow(dead_code)]
    diag: &'a DiagnosticEngine,

    /// ABI handler for call lowering and stack layout computation.
    /// Retained for full call-lowering expansion (struct returns, HFA, etc.).
    #[allow(dead_code)]
    abi: I686Abi,

    /// Maps IR ValueId → MachineOperand (tracks value materialization).
    value_map: FxHashMap<u32, MachineOperand>,

    /// Maps IR BasicBlockId → machine basic block numeric ID.
    block_map: FxHashMap<u32, u32>,

    /// Next frame index to allocate for local variables / spill slots.
    next_frame_index: u32,

    /// Accumulated frame size for local allocations (in bytes).
    frame_size: u32,

    /// Whether the current function contains any CALL instructions.
    has_calls: bool,

    /// Next virtual register ID for temporary values that don't yet have
    /// a physical register assigned.
    next_vreg: u32,

    /// Tracks the high 32-bit word for 64-bit (I64) values.
    /// Key: ValueId index of the I64 value, Value: MachineOperand for the high word.
    i64_high_map: FxHashMap<u32, MachineOperand>,
}

impl<'a> I686InstrSel<'a> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new i686 instruction selector with the given configuration
    /// and diagnostic engine.
    pub fn new(config: &'a CodegenConfig, diag: &'a DiagnosticEngine) -> Self {
        Self {
            config,
            diag,
            abi: I686Abi::new(),
            value_map: FxHashMap::default(),
            block_map: FxHashMap::default(),
            next_frame_index: 0,
            frame_size: 0,
            has_calls: false,
            next_vreg: 0,
            i64_high_map: FxHashMap::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Top-level: select an entire function
    // -----------------------------------------------------------------------

    /// Translates an entire IR function to a machine function.
    ///
    /// This is the primary entry point for i686 instruction selection. It:
    /// 1. Resets per-function state (value map, block map, frame counters).
    /// 2. Pre-assigns machine block IDs for every IR basic block.
    /// 3. Generates the function prologue (push EBP, mov EBP ESP, sub ESP).
    /// 4. Iterates all IR basic blocks and instructions, calling
    ///    [`select_instruction`] for each.
    /// 5. Returns the completed [`MachineFunction`].
    pub fn select_function(&mut self, func: &IrFunction) -> MachineFunction {
        // Reset per-function state.
        self.value_map.clear();
        self.block_map.clear();
        self.i64_high_map.clear();
        self.next_frame_index = 0;
        self.frame_size = 0;
        self.has_calls = false;
        self.next_vreg = func.local_values.len() as u32;

        let stack_align = self.config.target.stack_alignment();
        let mut mf = MachineFunction::new(func.name.clone(), stack_align);

        // Phase 0b: Pre-populate value_map for special IR values.
        //
        // The IR builder encodes certain value kinds (global references,
        // integer constants, float constants, null pointers) by naming
        // convention in the ValueInfo table rather than emitting dedicated
        // instructions.  The codegen must recognise these names and map
        // them to the appropriate MachineOperand *before* instruction
        // selection begins — otherwise get_operand() returns a VirtualReg
        // which produces incorrect code (e.g. indirect calls instead of
        // direct PLT calls for printf).
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
                        .insert(vi.id.index(), MachineOperand::Symbol(sym_name.to_string()));
                } else if let Some(int_str) = n.strip_prefix("const.int.") {
                    if let Ok(val) = int_str.parse::<i64>() {
                        // On i686, split 64-bit constants into lo/hi halves.
                        let lo = val as i32 as i64; // sign-extend the low 32 bits
                        let hi = (val >> 32) as i32 as i64;
                        self.value_map
                            .insert(vi.id.index(), MachineOperand::Immediate(lo));
                        // Track the high word for I64 types.
                        if matches!(&vi.ty, crate::ir::types::IrType::I64) {
                            self.i64_high_map
                                .insert(vi.id.index(), MachineOperand::Immediate(hi));
                        }
                    }
                } else if let Some(flt_str) = n.strip_prefix("const.float.") {
                    if let Ok(val) = flt_str.parse::<f64>() {
                        // On i686, 32-bit immediates are the maximum MOV width.
                        // F32 constants: convert to 32-bit float bit pattern.
                        // F64 constants: split into lo/hi 32-bit halves (like I64).
                        if matches!(&vi.ty, crate::ir::types::IrType::F32) {
                            let f32_bits = (val as f32).to_bits() as i64;
                            self.value_map
                                .insert(vi.id.index(), MachineOperand::Immediate(f32_bits));
                        } else {
                            // F64: store as lo/hi 32-bit halves.
                            let bits = val.to_bits();
                            let lo = (bits as u32) as i32 as i64;
                            let hi = ((bits >> 32) as u32) as i32 as i64;
                            self.value_map
                                .insert(vi.id.index(), MachineOperand::Immediate(lo));
                            self.i64_high_map
                                .insert(vi.id.index(), MachineOperand::Immediate(hi));
                        }
                    }
                } else if n == "const.null" {
                    self.value_map
                        .insert(vi.id.index(), MachineOperand::Immediate(0));
                }
            }
        }

        // Phase 1: Pre-assign block IDs so forward branches can reference them.
        for (idx, bb) in func.basic_blocks.iter().enumerate() {
            let mbb_id = idx as u32;
            self.block_map.insert(bb.id.index(), mbb_id);
        }

        // Phase 2: Map function parameters to their stack locations.
        // Under cdecl, parameters live at [EBP+8], [EBP+12], etc.
        let mut param_offset: i32 = 8; // first arg is at [EBP+8]
        for param in &func.params {
            let param_size = self.type_size_bytes(&param.ty);
            let aligned_size = align_up(param_size, 4);
            let mem_op = MachineOperand::Memory {
                base: EBP,
                offset: param_offset,
                index: None,
                scale: 1,
            };
            self.value_map.insert(param.id.index(), mem_op);
            param_offset += aligned_size as i32;
        }

        // Phase 3: First pass — allocate frame slots for all Alloca instructions.
        for bb in &func.basic_blocks {
            for inst in bb.instructions() {
                if let Instruction::Alloca {
                    result,
                    ty,
                    alignment,
                } = inst
                {
                    let size = self.type_size_bytes(ty);
                    let align = if *alignment > 0 {
                        *alignment
                    } else {
                        self.type_alignment(ty)
                    };
                    // Align the frame size up
                    self.frame_size = align_up(self.frame_size, align);
                    self.frame_size += size;
                    // Local variables are at negative offsets from EBP.
                    // Use FrameIndex (not Memory) so that:
                    //   • Load/Store via safe_to_memory → encoder handles [EBP+off]
                    //   • ensure_in_register → LEA (gives address, not contents)
                    // This matches the x86-64 backend's approach and correctly
                    // implements array-to-pointer decay and pass-by-reference.
                    let offset = -(self.frame_size as i32);
                    let fi_op = MachineOperand::FrameIndex(offset as u32);
                    // eprintln!("[I686_ALLOCA] value={} type={:?} size={} align={} frame_size={} offset={} fi=0x{:08x}",
                    // result.index(), ty, size, align, self.frame_size, offset, offset as u32);
                    self.value_map.insert(result.index(), fi_op);
                    self.next_frame_index += 1;
                }
            }
        }

        // Align total frame size to stack alignment.
        self.frame_size = align_up(self.frame_size, stack_align);

        // Phase 4: Create the entry block (prologue is NOT emitted here;
        // it will be injected later by ArchCodegen::emit_prologue via
        // compile_function in generation.rs, matching the x86-64 pattern).
        let entry_id = 0u32;
        let entry_block =
            MachineBasicBlock::with_label(entry_id, format!(".L{}_{}", func.name, entry_id));
        mf.add_block(entry_block);

        // Phase 5: Lower each IR basic block's instructions.
        // Debug: dump IR instructions for the first basic block
        if let Some(bb0) = func.basic_blocks.first() {
            for (_ii, _inst) in bb0.instructions().iter().enumerate() {
                // eprintln!("[I686_IR] func={} bb0 inst[{}]: {:?}", func.name, ii, inst);
            }
        }
        for (idx, bb) in func.basic_blocks.iter().enumerate() {
            let mbb_id = idx as u32;
            // For the entry block (idx 0), we already created it above;
            // append instructions to it. For other blocks, create new ones.
            if idx == 0 {
                // Append to the entry block already in mf.
                for inst in bb.instructions() {
                    if inst.is_alloca() {
                        continue; // already handled in Phase 3
                    }
                    let mut tmp_block = MachineBasicBlock::new(mbb_id);
                    self.select_instruction(inst, &mut tmp_block, func);
                    // Move generated instructions into entry block.
                    if let Some(eb) = mf.get_block_mut(entry_id) {
                        for mi in tmp_block.instructions {
                            eb.push_instr(mi);
                        }
                    }
                }
            } else {
                let mut mbb =
                    MachineBasicBlock::with_label(mbb_id, format!(".L{}_{}", func.name, mbb_id));
                for inst in bb.instructions() {
                    if inst.is_alloca() {
                        continue;
                    }
                    self.select_instruction(inst, &mut mbb, func);
                }
                mf.add_block(mbb);
            }
        }

        // Finalize machine function metadata.
        mf.frame_size = self.frame_size;
        mf.has_calls = self.has_calls;
        for &reg in &CALLEE_SAVED {
            mf.mark_callee_saved_used(reg);
        }

        mf
    }

    // -----------------------------------------------------------------------
    // Per-instruction selection
    // -----------------------------------------------------------------------

    /// Translates a single IR instruction into one or more i686 machine
    /// instructions, appending them to the given machine basic block.
    ///
    /// This is the core dispatch method of the instruction selector. Each
    /// IR instruction variant is pattern-matched and lowered to the appropriate
    /// i686 instruction sequence.
    pub fn select_instruction(
        &mut self,
        inst: &Instruction,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        match inst {
            // -- Memory operations --
            Instruction::Alloca { .. } => {
                // Allocas are pre-processed in select_function Phase 3.
                // This branch should not be reached during normal lowering.
            }

            Instruction::Load {
                result, ptr, ty, ..
            } => {
                self.select_load(*result, *ptr, ty, mbb, func);
            }

            Instruction::Store { value, ptr, .. } => {
                self.select_store(*value, *ptr, mbb, func);
            }

            // -- Binary arithmetic / bitwise --
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
            } => {
                self.select_binop(*result, *op, *lhs, *rhs, ty, mbb, func);
            }

            // -- Integer comparison --
            Instruction::ICmp {
                result,
                pred,
                lhs,
                rhs,
            } => {
                self.select_icmp(*result, pred, *lhs, *rhs, mbb, func);
            }

            // -- Floating-point comparison --
            Instruction::FCmp {
                result,
                pred,
                lhs,
                rhs,
            } => {
                self.select_fcmp(*result, pred, *lhs, *rhs, mbb, func);
            }

            // -- Control flow --
            Instruction::Branch { target } => {
                self.select_branch(*target, mbb);
            }

            Instruction::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                self.select_cond_branch(*condition, *true_target, *false_target, mbb, func);
            }

            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                self.select_switch(*value, *default, cases, mbb, func);
            }

            Instruction::Call {
                result,
                callee,
                args,
                is_tail,
                ..
            } => {
                self.select_call(*result, *callee, args, *is_tail, mbb, func);
            }

            Instruction::Return { value } => {
                self.select_return(*value, mbb, func);
            }

            // -- SSA Phi (should be eliminated before codegen, emit copy as fallback) --
            Instruction::Phi {
                result,
                ty: _,
                incoming,
            } => {
                // Phi nodes should have been eliminated by phi_eliminate pass.
                // If we encounter one, treat it as an internal error and emit
                // a nop placeholder. The value from the first incoming edge is used.
                if let Some((first_val, _)) = incoming.first() {
                    let src = self.get_operand(*first_val, func);
                    let dst = self.alloc_vreg(*result);
                    self.emit_mov(dst, src, mbb);
                }
            }

            // -- Pointer arithmetic --
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                ty,
                in_bounds: _,
            } => {
                self.select_gep(*result, *base, indices, ty, mbb, func);
            }

            // -- Type conversions --
            Instruction::BitCast {
                result,
                value,
                to_ty,
            } => {
                self.select_bitcast(*result, *value, to_ty, mbb, func);
            }

            Instruction::Trunc {
                result,
                value,
                to_ty,
            } => {
                self.select_trunc(*result, *value, to_ty, mbb, func);
            }

            Instruction::ZExt {
                result,
                value,
                to_ty,
            } => {
                self.select_zext(*result, *value, to_ty, mbb, func);
            }

            Instruction::SExt {
                result,
                value,
                to_ty,
            } => {
                self.select_sext(*result, *value, to_ty, mbb, func);
            }

            Instruction::IntToPtr {
                result,
                value,
                to_ty: _,
            } => {
                // On i686, integer→pointer is a bitwise reinterpret (same 32-bit width).
                let src = self.get_operand(*value, func);
                let dst = self.alloc_vreg(*result);
                self.emit_mov(dst, src, mbb);
            }

            Instruction::PtrToInt {
                result,
                value,
                to_ty: _,
            } => {
                // On i686, pointer→integer is a bitwise reinterpret (same 32-bit width).
                let src = self.get_operand(*value, func);
                let dst = self.alloc_vreg(*result);
                self.emit_mov(dst, src, mbb);
            }

            // --- Floating-point conversion instructions ---
            Instruction::SIToFP {
                result,
                value,
                to_ty,
            } => {
                self.select_si_to_fp(*result, *value, to_ty, mbb, func);
            }
            Instruction::UIToFP {
                result,
                value,
                to_ty,
            } => {
                self.select_ui_to_fp(*result, *value, to_ty, mbb, func);
            }
            Instruction::FPToSI {
                result,
                value,
                to_ty,
            } => {
                self.select_fp_to_si(*result, *value, to_ty, mbb, func);
            }
            Instruction::FPToUI {
                result,
                value,
                to_ty,
            } => {
                self.select_fp_to_ui(*result, *value, to_ty, mbb, func);
            }
            Instruction::FPExt {
                result,
                value,
                to_ty,
            } => {
                self.select_fp_ext(*result, *value, to_ty, mbb, func);
            }
            Instruction::FPTrunc {
                result,
                value,
                to_ty,
            } => {
                self.select_fp_trunc(*result, *value, to_ty, mbb, func);
            }

            // -- Inline assembly --
            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_align_stack: _,
                goto_targets,
            } => {
                let instrs = self.lower_inline_asm(
                    result.as_ref().copied(),
                    template,
                    constraints,
                    operands,
                    clobbers,
                    goto_targets,
                    func,
                );
                for instr in instrs {
                    mbb.push_instr(instr);
                }
            }

            // -- Computed goto --
            Instruction::BlockAddress { result, block } => {
                // LEA dst, label — load address of a basic-block label.
                let dst = self.alloc_vreg(*result);
                // Map IR block ID → machine block ID via block_map
                let target_id = self.block_map.get(&block.index()).copied().unwrap_or(0);
                let mut mi = MachineInstr::new(I686Opcode::LeaLabel.as_u32());
                mi.add_operand(dst);
                mi.add_operand(MachineOperand::Label(target_id));
                mbb.push_instr(mi);
            }

            Instruction::IndirectBranch {
                addr,
                possible_targets: _,
            } => {
                // JMP *reg — indirect jump through a register.
                let op = self.get_operand(*addr, func);
                let mut mi = MachineInstr::new(I686Opcode::JmpIndirect.as_u32());
                mi.add_operand(op);
                mi.set_terminator();
                mbb.push_instr(mi);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Load selection
    // -----------------------------------------------------------------------

    /// Selects instructions for an IR Load.
    fn select_load(
        &mut self,
        result: ValueId,
        ptr: ValueId,
        ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let addr = self.get_operand(ptr, func);
        let dst = self.alloc_vreg(result);

        // --- F32 GPR CLASSIFICATION FIX ---
        // The register allocator classifies VirtualRegs by IR type.  F32
        // values would be classified as FloatingPoint → assigned x87
        // registers (ST0-ST7).  On i686, x87 register encoding indices
        // 0-7 COLLIDE with GPR encoding indices (EAX-EDI).  Integer
        // instructions like MOV use the opcode to select the GPR file,
        // but if the operand register encodes to the same value as a GPR
        // (e.g., ST5 encodes as 5 = EBP), the CPU interprets it as a
        // GPR access, corrupting the frame pointer.
        //
        // Fix: create a synthetic VirtualReg whose ValueId is not in the
        // IR function.  `classify_value` returns GP for unknown IDs.
        let dst = if matches!(ty, IrType::F32) {
            let raw_id = self.alloc_vreg_raw();
            let gpr_dst = MachineOperand::VirtualReg(ValueId(raw_id));
            self.value_map.insert(result.index(), gpr_dst.clone());
            gpr_dst
        } else {
            dst
        };

        // --- GLOBAL VARIABLE FIX ---
        // When the address is a Symbol (global variable reference), we must
        // first load the symbol's absolute address into a register, then use
        // that register as a base for the indirect load.  If we pass the Symbol
        // through to the encoder directly, encode_mov sees (Register, Symbol)
        // and emits MOV reg, imm32(addr) — loading the ADDRESS, not the VALUE.
        // By converting to a VirtualReg here, we fall into the VirtualReg path
        // which correctly uses MovLoad (indirect load through pointer register).
        if matches!(&addr, MachineOperand::Symbol(_)) {
            let addr_vreg = self.alloc_vreg_raw();
            let addr_op = MachineOperand::VirtualReg(ValueId(addr_vreg));
            self.emit_mov(addr_op.clone(), addr, mbb);
            // Now recurse with the address in a VirtualReg — this hits the
            // VirtualReg path below which correctly uses indirect load.
            self.value_map.insert(ptr.index(), addr_op.clone());
            // Fall through to VirtualReg handling below.
            return self.select_load_vreg_addr(result, addr_op, ty, dst, mbb, func);
        }

        if matches!(ty, IrType::F64 | IrType::F80) {
            // F64: load as two 32-bit halves into GPRs (same layout as I64).
            return self.select_load_f64_to_gprs(result, addr, ty, mbb, func);
        }
        // F32: fall through to the normal 32-bit load logic below (same as int).
        // The F32 bits are stored in a GPR, matching conversion function expectations.

        // If the address is a VirtualReg, we cannot embed it in a Memory
        // operand (register allocator only rewrites top-level operands).
        // Use MovLoad which keeps the address as a top-level Register operand.
        if matches!(&addr, MachineOperand::VirtualReg(_)) {
            let size_bits = self.type_size_bits(ty);
            if size_bits == 8 || size_bits == 16 {
                // For 8/16-bit loads, we still need MovLoad then MovZx from
                // a temporary. Use MovLoad for 32-bit, then mask.
                let tmp = self.alloc_vreg_raw();
                let mut inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
                inst.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                inst.add_operand(addr);
                mbb.push_instr(inst);
                // Mask to the correct size.
                let mask: i64 = if size_bits == 8 { 0xFF } else { 0xFFFF };
                self.emit_mov(dst.clone(), MachineOperand::VirtualReg(ValueId(tmp)), mbb);
                let mut and_inst = MachineInstr::new(I686Opcode::And.as_u32());
                and_inst.add_operand(dst);
                and_inst.add_operand(MachineOperand::Immediate(mask));
                and_inst.add_implicit_def(EFLAGS);
                mbb.push_instr(and_inst);
            } else if size_bits == 64 {
                // 64-bit load through pointer: two 32-bit MovLoads.
                let lo_dst = self.alloc_vreg_raw();
                let mut lo_inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
                lo_inst.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
                lo_inst.add_operand(addr.clone());
                mbb.push_instr(lo_inst);
                // For hi half, add 4 to addr.
                let addr_plus4 = self.alloc_vreg_raw();
                self.emit_mov(MachineOperand::VirtualReg(ValueId(addr_plus4)), addr, mbb);
                let mut add = MachineInstr::new(I686Opcode::Add.as_u32());
                add.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                add.add_operand(MachineOperand::Immediate(4));
                add.add_implicit_def(EFLAGS);
                mbb.push_instr(add);
                let hi_dst = self.alloc_vreg_raw();
                let mut hi_inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
                hi_inst.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
                hi_inst.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                mbb.push_instr(hi_inst);
                self.value_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
                // Track the high 32-bit word for this I64 value.
                self.i64_high_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_dst)));
            } else {
                // 32-bit or pointer load.
                let mut inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
                inst.add_operand(dst);
                inst.add_operand(addr);
                mbb.push_instr(inst);
            }
            return;
        }

        // For FrameIndex, Memory, Symbol, Register — these can be handled
        // directly by the encoder via existing patterns.
        let size_bits = self.type_size_bits(ty);
        match size_bits {
            8 => {
                let mem = self.safe_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::MovZx.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
            16 => {
                let mem = self.safe_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::MovZx.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
            32 => {
                let mem = self.safe_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
            64 => {
                let base_mem = self.safe_to_memory(addr.clone(), mbb);
                let lo_dst = self.alloc_vreg_raw();
                let mut lo_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                lo_inst.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
                lo_inst.add_operand(base_mem.clone());
                mbb.push_instr(lo_inst);
                let hi_mem = self.offset_memory(base_mem, 4);
                let hi_dst = self.alloc_vreg_raw();
                let mut hi_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                hi_inst.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
                hi_inst.add_operand(hi_mem);
                mbb.push_instr(hi_inst);
                self.value_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
                // Track the high 32-bit word for this I64 value.
                self.i64_high_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_dst)));
            }
            _ => {
                let mem = self.safe_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Load via VirtualReg address (factored out for Symbol reuse)
    // -----------------------------------------------------------------------

    /// Performs a load through a pointer held in a VirtualReg.
    /// This is extracted from select_load so that the Symbol path
    /// (which converts a global symbol to a vreg address) can reuse it.
    /// Load an F64 value from memory into two GPR halves (lo + hi).
    fn select_load_f64_to_gprs(
        &mut self,
        result: ValueId,
        addr: MachineOperand,
        _ty: &IrType,
        mbb: &mut MachineBasicBlock,
        _func: &IrFunction,
    ) {
        if matches!(&addr, MachineOperand::VirtualReg(_)) {
            // Load through pointer register.
            let lo_idx = self.alloc_vreg_raw();
            let mut lo_ld = MachineInstr::new(I686Opcode::MovLoad.as_u32());
            lo_ld.add_operand(MachineOperand::VirtualReg(ValueId(lo_idx)));
            lo_ld.add_operand(addr.clone());
            mbb.push_instr(lo_ld);
            // addr+4 for hi half.
            let addr_p4 = self.alloc_vreg_raw();
            self.emit_mov(MachineOperand::VirtualReg(ValueId(addr_p4)), addr, mbb);
            let mut add4 = MachineInstr::new(I686Opcode::Add.as_u32());
            add4.add_operand(MachineOperand::VirtualReg(ValueId(addr_p4)));
            add4.add_operand(MachineOperand::Immediate(4));
            add4.add_implicit_def(EFLAGS);
            mbb.push_instr(add4);
            let hi_idx = self.alloc_vreg_raw();
            let mut hi_ld = MachineInstr::new(I686Opcode::MovLoad.as_u32());
            hi_ld.add_operand(MachineOperand::VirtualReg(ValueId(hi_idx)));
            hi_ld.add_operand(MachineOperand::VirtualReg(ValueId(addr_p4)));
            mbb.push_instr(hi_ld);
            self.value_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_idx)));
            self.i64_high_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_idx)));
        } else {
            // Memory or FrameIndex address.
            let mem = self.safe_to_memory(addr.clone(), mbb);
            let lo_idx = self.alloc_vreg_raw();
            let mut lo_ld = MachineInstr::new(I686Opcode::Mov.as_u32());
            lo_ld.add_operand(MachineOperand::VirtualReg(ValueId(lo_idx)));
            lo_ld.add_operand(mem.clone());
            mbb.push_instr(lo_ld);
            let hi_mem = self.offset_memory(mem, 4);
            let hi_idx = self.alloc_vreg_raw();
            let mut hi_ld = MachineInstr::new(I686Opcode::Mov.as_u32());
            hi_ld.add_operand(MachineOperand::VirtualReg(ValueId(hi_idx)));
            hi_ld.add_operand(hi_mem);
            mbb.push_instr(hi_ld);
            self.value_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_idx)));
            self.i64_high_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_idx)));
        }
    }

    fn select_load_vreg_addr(
        &mut self,
        result: ValueId,
        addr: MachineOperand,
        ty: &IrType,
        dst: MachineOperand,
        mbb: &mut MachineBasicBlock,
        _func: &IrFunction,
    ) {
        // Floating-point through vreg pointer: F32 = 4-byte GPR load (fall through),
        // F64 = two 32-bit halves.
        if matches!(ty, IrType::F64 | IrType::F80) {
            return self.select_load_f64_to_gprs(result, addr, ty, mbb, _func);
        }
        // F32: falls through to the 32-bit load path below.

        let size_bits = self.type_size_bits(ty);
        if size_bits == 8 || size_bits == 16 {
            let tmp = self.alloc_vreg_raw();
            let mut inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
            inst.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
            inst.add_operand(addr);
            mbb.push_instr(inst);
            let mask: i64 = if size_bits == 8 { 0xFF } else { 0xFFFF };
            self.emit_mov(dst.clone(), MachineOperand::VirtualReg(ValueId(tmp)), mbb);
            let mut and_inst = MachineInstr::new(I686Opcode::And.as_u32());
            and_inst.add_operand(dst);
            and_inst.add_operand(MachineOperand::Immediate(mask));
            and_inst.add_implicit_def(EFLAGS);
            mbb.push_instr(and_inst);
        } else if size_bits == 64 {
            let lo_dst = self.alloc_vreg_raw();
            let mut lo_inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
            lo_inst.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
            lo_inst.add_operand(addr.clone());
            mbb.push_instr(lo_inst);
            let addr_plus4 = self.alloc_vreg_raw();
            self.emit_mov(MachineOperand::VirtualReg(ValueId(addr_plus4)), addr, mbb);
            let mut add = MachineInstr::new(I686Opcode::Add.as_u32());
            add.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
            add.add_operand(MachineOperand::Immediate(4));
            add.add_implicit_def(EFLAGS);
            mbb.push_instr(add);
            let hi_dst = self.alloc_vreg_raw();
            let mut hi_inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
            hi_inst.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
            hi_inst.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
            mbb.push_instr(hi_inst);
            self.value_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
            self.i64_high_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_dst)));
        } else {
            let mut inst = MachineInstr::new(I686Opcode::MovLoad.as_u32());
            inst.add_operand(dst);
            inst.add_operand(addr);
            mbb.push_instr(inst);
        }
    }

    // -----------------------------------------------------------------------
    // Store selection
    // -----------------------------------------------------------------------

    /// Returns the correct MOV opcode for the given IR type width:
    ///  - I1 / I8  → MovByte  (8-bit store, opcode 0x88 / 0xC6)
    ///  - I16      → MovWord  (16-bit store, 0x66 prefix + 0x89 / 0xC7)
    ///  - I32 / Ptr / everything else → Mov (32-bit store, 0x89 / 0xC7)
    fn store_opcode_for_type(ty: &crate::ir::types::IrType) -> I686Opcode {
        match ty {
            crate::ir::types::IrType::I1 | crate::ir::types::IrType::I8 => I686Opcode::MovByte,
            crate::ir::types::IrType::I16 => I686Opcode::MovWord,
            _ => I686Opcode::Mov,
        }
    }

    /// Selects instructions for an IR Store.
    fn select_store(
        &mut self,
        value: ValueId,
        ptr: ValueId,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let val_op = self.get_operand(value, func);
        let addr_op = self.get_operand(ptr, func);
        // Store instruction selection — the address operand type governs
        // the memory access pattern used below.

        // --- GLOBAL VARIABLE FIX ---
        // When the address is a Symbol (global variable reference), we must
        // first load the symbol's absolute address into a register, then
        // use indirect store through that register.  The encoder's Mov handler
        // for (Symbol, Register) is missing, so without this conversion the
        // store would be silently dropped.
        if matches!(&addr_op, MachineOperand::Symbol(_)) {
            let addr_vreg = self.alloc_vreg_raw();
            let vr_op = MachineOperand::VirtualReg(ValueId(addr_vreg));
            self.emit_mov(vr_op.clone(), addr_op, mbb);
            // Redirect to VirtualReg address path.
            self.value_map.insert(ptr.index(), vr_op);
            return self.select_store(value, ptr, mbb, func);
        }

        // Determine the width of the value being stored.
        let val_ty = func.get_value_type(value).clone();

        // Check if the value is an x87 FP value (result of FP operation).
        if let MachineOperand::Register(reg) = &val_op {
            if registers::is_fpu(*reg) {
                // FSTP to store from x87 stack to memory.
                let mem = self.safe_to_memory(addr_op, mbb);
                let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
                fstp.add_operand(mem);
                fstp.add_implicit_use(ST0);
                mbb.push_instr(fstp);
                return;
            }
        }

        // Handle float constants stored as Immediate (f64 bit patterns).
        // CRITICAL: F64 constants are pre-split into lo/hi halves during
        // constant pre-population.  The value_map holds only the lo 32
        // bits, and i64_high_map holds the hi 32 bits.  We MUST NOT try
        // to reconstruct f64 from the lo Immediate alone (that gives 0.0
        // for many doubles whose lo half happens to be 0).
        //
        // Strategy:
        // 1. If value has an i64_high_map entry → use GPR pair path (below)
        // 2. Else if F32 Immediate → convert and store as 32-bit
        // 3. Else if F64 Immediate without split → reconstruct (legacy)
        if val_ty.is_floating() {
            // Check for pre-split F64 first — if hi half exists, fall
            // through to the GPR pair path below rather than
            // reconstructing from a truncated immediate.
            let has_hi = self.i64_high_map.contains_key(&value.index());
            if !has_hi {
                if let MachineOperand::Immediate(bits) = &val_op {
                    if matches!(&val_ty, IrType::F32) {
                        // F32: the immediate already holds 32-bit f32 bits
                        let src = self.ensure_in_register(MachineOperand::Immediate(*bits), mbb);
                        if matches!(&addr_op, MachineOperand::VirtualReg(_)) {
                            let mut inst = MachineInstr::new(I686Opcode::MovStore.as_u32());
                            inst.add_operand(addr_op);
                            inst.add_operand(src);
                            mbb.push_instr(inst);
                        } else {
                            let mem = self.safe_to_memory(addr_op, mbb);
                            let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                            inst.add_operand(mem);
                            inst.add_operand(src);
                            mbb.push_instr(inst);
                        }
                        return;
                    }
                    // F64 without split (shouldn't happen normally, but
                    // handle as legacy path).
                    let f64_val = f64::from_bits(*bits as u64);
                    let raw = f64_val.to_bits();
                    let lo = (raw & 0xFFFF_FFFF) as i64;
                    let hi = ((raw >> 32) & 0xFFFF_FFFF) as i64;
                    let lo_src = self.ensure_in_register(MachineOperand::Immediate(lo), mbb);
                    let hi_src = self.ensure_in_register(MachineOperand::Immediate(hi), mbb);
                    if matches!(&addr_op, MachineOperand::VirtualReg(_)) {
                        let mut lo_st = MachineInstr::new(I686Opcode::MovStore.as_u32());
                        lo_st.add_operand(addr_op.clone());
                        lo_st.add_operand(lo_src);
                        mbb.push_instr(lo_st);
                        let addr_plus4 = self.alloc_vreg_raw();
                        self.emit_mov(
                            MachineOperand::VirtualReg(ValueId(addr_plus4)),
                            addr_op,
                            mbb,
                        );
                        let mut add4 = MachineInstr::new(I686Opcode::Add.as_u32());
                        add4.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                        add4.add_operand(MachineOperand::Immediate(4));
                        add4.add_implicit_def(EFLAGS);
                        mbb.push_instr(add4);
                        let mut hi_st = MachineInstr::new(I686Opcode::MovStore.as_u32());
                        hi_st.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                        hi_st.add_operand(hi_src);
                        mbb.push_instr(hi_st);
                    } else {
                        let lo_mem = self.safe_to_memory(addr_op.clone(), mbb);
                        let mut lo_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                        lo_inst.add_operand(lo_mem.clone());
                        lo_inst.add_operand(lo_src);
                        mbb.push_instr(lo_inst);
                        let hi_mem = self.offset_memory(lo_mem, 4);
                        let mut hi_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                        hi_inst.add_operand(hi_mem);
                        hi_inst.add_operand(hi_src);
                        mbb.push_instr(hi_inst);
                    }
                    return;
                }
            }
            // Non-immediate float in GPR(s): store directly to destination.
            if matches!(&val_ty, IrType::F64) {
                // F64 represented as two GPR halves (lo + hi via i64_high_map).
                let lo_reg = self.ensure_in_register(val_op, mbb);
                let hi_op = self
                    .i64_high_map
                    .get(&value.index())
                    .cloned()
                    .unwrap_or(MachineOperand::Immediate(0));
                let hi_reg = self.ensure_in_register(hi_op, mbb);
                if matches!(&addr_op, MachineOperand::VirtualReg(_)) {
                    // Store lo half.
                    let mut lo_st = MachineInstr::new(I686Opcode::MovStore.as_u32());
                    lo_st.add_operand(addr_op.clone());
                    lo_st.add_operand(lo_reg);
                    mbb.push_instr(lo_st);
                    // Store hi half at addr+4.
                    let addr_plus4 = self.alloc_vreg_raw();
                    self.emit_mov(
                        MachineOperand::VirtualReg(ValueId(addr_plus4)),
                        addr_op,
                        mbb,
                    );
                    let mut add4 = MachineInstr::new(I686Opcode::Add.as_u32());
                    add4.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                    add4.add_operand(MachineOperand::Immediate(4));
                    add4.add_implicit_def(EFLAGS);
                    mbb.push_instr(add4);
                    let mut hi_st = MachineInstr::new(I686Opcode::MovStore.as_u32());
                    hi_st.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                    hi_st.add_operand(hi_reg);
                    mbb.push_instr(hi_st);
                } else {
                    let lo_mem = self.safe_to_memory(addr_op.clone(), mbb);
                    let mut lo_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                    lo_inst.add_operand(lo_mem.clone());
                    lo_inst.add_operand(lo_reg);
                    mbb.push_instr(lo_inst);
                    let hi_mem = self.offset_memory(lo_mem, 4);
                    let mut hi_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                    hi_inst.add_operand(hi_mem);
                    hi_inst.add_operand(hi_reg);
                    mbb.push_instr(hi_inst);
                }
                return;
            }
            // F32 in a single GPR: store via simple MOV (no FPU needed).
            let src_reg = self.ensure_in_register(val_op, mbb);
            if matches!(&addr_op, MachineOperand::VirtualReg(_)) {
                let mut inst = MachineInstr::new(I686Opcode::MovStore.as_u32());
                inst.add_operand(addr_op);
                inst.add_operand(src_reg);
                mbb.push_instr(inst);
            } else {
                let mem = self.safe_to_memory(addr_op, mbb);
                let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                inst.add_operand(mem);
                inst.add_operand(src_reg);
                mbb.push_instr(inst);
            }
            return;
        }

        // --- 64-bit store: write both lo and hi halves ---
        if matches!(&val_ty, crate::ir::types::IrType::I64) {
            let hi_op = self.i64_high_map.get(&value.index()).cloned();
            let lo_src = self.ensure_in_register(val_op, mbb);

            if matches!(&addr_op, MachineOperand::VirtualReg(_)) {
                // Store lo half through pointer.
                let mut lo_store = MachineInstr::new(I686Opcode::MovStore.as_u32());
                lo_store.add_operand(addr_op.clone());
                lo_store.add_operand(lo_src);
                mbb.push_instr(lo_store);

                // Compute addr+4 for hi half.
                let addr_plus4 = self.alloc_vreg_raw();
                self.emit_mov(
                    MachineOperand::VirtualReg(ValueId(addr_plus4)),
                    addr_op,
                    mbb,
                );
                let mut add4 = MachineInstr::new(I686Opcode::Add.as_u32());
                add4.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                add4.add_operand(MachineOperand::Immediate(4));
                add4.add_implicit_def(EFLAGS);
                mbb.push_instr(add4);

                let hi_src = if let Some(hi) = hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                let hi_reg = self.ensure_in_register(hi_src, mbb);
                let mut hi_store = MachineInstr::new(I686Opcode::MovStore.as_u32());
                hi_store.add_operand(MachineOperand::VirtualReg(ValueId(addr_plus4)));
                hi_store.add_operand(hi_reg);
                mbb.push_instr(hi_store);
            } else {
                // FrameIndex, Memory, Symbol — use safe_to_memory.
                let lo_mem = self.safe_to_memory(addr_op.clone(), mbb);
                let mut lo_store = MachineInstr::new(I686Opcode::Mov.as_u32());
                lo_store.add_operand(lo_mem.clone());
                lo_store.add_operand(lo_src);
                mbb.push_instr(lo_store);

                // Hi half at offset +4.
                let hi_mem = self.offset_memory(lo_mem, 4);
                let hi_src = if let Some(hi) = hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                let hi_reg = self.ensure_in_register(hi_src, mbb);
                let mut hi_store = MachineInstr::new(I686Opcode::Mov.as_u32());
                hi_store.add_operand(hi_mem);
                hi_store.add_operand(hi_reg);
                mbb.push_instr(hi_store);
            }
            return;
        }

        // Pick the correct opcode based on value width.
        let mov_opc = Self::store_opcode_for_type(&val_ty);

        // If the destination address is a VirtualReg, use the indirect-store
        // pattern: operands [addr_vreg, src_reg].  The encoder sees these as
        // physical registers after allocation and emits MOV [addr], src with
        // the appropriate width prefix.
        if matches!(&addr_op, MachineOperand::VirtualReg(_)) {
            let src = self.ensure_in_register(val_op, mbb);
            if matches!(mov_opc, I686Opcode::Mov) {
                // 32-bit store — use MovStore (special indirect-store opcode).
                let mut inst = MachineInstr::new(I686Opcode::MovStore.as_u32());
                inst.add_operand(addr_op);
                inst.add_operand(src);
                mbb.push_instr(inst);
            } else {
                // Sub-32-bit store (byte or word) through a pointer in a
                // virtual register.  Use the byte/word opcode with the same
                // [addr_vreg, src] operand layout — the encoder for
                // MovByte / MovWord handles this pattern just like MovStore.
                let mut inst = MachineInstr::new(mov_opc.as_u32());
                inst.add_operand(addr_op);
                inst.add_operand(src);
                mbb.push_instr(inst);
            }
            return;
        }

        // For FrameIndex, Memory, Register, Symbol — use normal path.
        let mem = self.safe_to_memory(addr_op, mbb);
        // Ensure value is in a register (can't do mem-to-mem move on x86).
        let src = self.ensure_in_register(val_op, mbb);
        let mut inst = MachineInstr::new(mov_opc.as_u32());
        inst.add_operand(mem);
        inst.add_operand(src);
        mbb.push_instr(inst);
    }

    // -----------------------------------------------------------------------
    // BinOp selection
    // -----------------------------------------------------------------------

    /// Selects instructions for an IR BinOp.
    fn select_binop(
        &mut self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        // Floating-point operations go through the x87 FPU path.
        if op.is_floating_point() {
            self.select_fp_binop(result, op, lhs, rhs, ty, mbb, func);
            return;
        }

        // Check for 64-bit integer operations on 32-bit.
        let bits = self.type_size_bits(ty);
        if bits == 64 {
            self.select_binop_i64(result, op, lhs, rhs, mbb, func);
            return;
        }

        let lhs_op = self.get_operand(lhs, func);
        let rhs_op = self.get_operand(rhs, func);

        match op {
            // Simple commutative/associative ALU ops: ADD, SUB, AND, OR, XOR.
            BinOp::Add | BinOp::Sub | BinOp::And | BinOp::Or | BinOp::Xor => {
                let opcode = match op {
                    BinOp::Add => I686Opcode::Add,
                    BinOp::Sub => I686Opcode::Sub,
                    BinOp::And => I686Opcode::And,
                    BinOp::Or => I686Opcode::Or,
                    BinOp::Xor => I686Opcode::Xor,
                    _ => unreachable!(),
                };
                let dst = self.alloc_vreg(result);
                // MOV dst, lhs
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                self.emit_mov(dst.clone(), lhs_reg, mbb);
                // OP dst, rhs
                let mut inst = MachineInstr::new(opcode.as_u32());
                inst.add_operand(dst);
                inst.add_operand(rhs_op);
                inst.add_implicit_def(EFLAGS);
                mbb.push_instr(inst);
            }

            // Multiplication: IMUL dst, src (2-operand form).
            BinOp::Mul => {
                let dst = self.alloc_vreg(result);
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                self.emit_mov(dst.clone(), lhs_reg, mbb);
                let mut inst = MachineInstr::new(I686Opcode::Imul.as_u32());
                inst.add_operand(dst);
                inst.add_operand(rhs_op);
                inst.add_implicit_def(EFLAGS);
                mbb.push_instr(inst);
            }

            // Signed division: IDIV. Result in EAX (quotient), EDX (remainder).
            BinOp::SDiv => {
                self.select_div(result, lhs_op, rhs_op, true, false, mbb);
            }

            // Unsigned division: DIV. Result in EAX (quotient), EDX (remainder).
            BinOp::UDiv => {
                self.select_div(result, lhs_op, rhs_op, false, false, mbb);
            }

            // Signed remainder: IDIV, result is in EDX.
            BinOp::SRem => {
                self.select_div(result, lhs_op, rhs_op, true, true, mbb);
            }

            // Unsigned remainder: DIV, result is in EDX.
            BinOp::URem => {
                self.select_div(result, lhs_op, rhs_op, false, true, mbb);
            }

            // Shifts: SHL, SHR (LShr), SAR (AShr).
            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                let opcode = match op {
                    BinOp::Shl => I686Opcode::Shl,
                    BinOp::LShr => I686Opcode::Shr,
                    BinOp::AShr => I686Opcode::Sar,
                    _ => unreachable!(),
                };

                // Extract immediate value (if any) before moving operands.
                let rhs_imm = if let MachineOperand::Immediate(imm) = &rhs_op {
                    Some(*imm)
                } else {
                    None
                };

                if let Some(imm) = rhs_imm {
                    // Immediate shift: no register conflict possible.
                    let dst = self.alloc_vreg(result);
                    let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                    self.emit_mov(dst.clone(), lhs_reg, mbb);
                    let mut inst = MachineInstr::new(opcode.as_u32());
                    inst.add_operand(dst);
                    inst.add_operand(MachineOperand::Immediate(imm & 31));
                    inst.add_implicit_def(EFLAGS);
                    mbb.push_instr(inst);
                } else {
                    // Variable shift: use all-physical-register sequence to
                    // avoid register allocator conflicts with the ECX
                    // constraint.
                    //
                    // x86 variable shifts MUST use CL as the shift count
                    // register.  A naïve `MOV ECX, rhs; SHL vreg, CL`
                    // allows the register allocator to assign the
                    // destination vreg to ECX — the MOV then clobbers the
                    // value-to-shift before the SHL executes.
                    //
                    // Solution (mirrors select_div pattern):
                    //   1. Spill shift amount to a temp stack slot
                    //   2. Move lhs value into EAX (physical)
                    //   3. Load shift amount from stack into ECX (physical)
                    //   4. SHL/SHR/SAR EAX, CL  (all physical regs)
                    //   5. Move result from EAX to destination vreg
                    //
                    // Because ALL physical register usage is isolated
                    // (vreg inputs spilled/consumed before; vreg result
                    // created after), no aliasing conflict can occur.

                    // Step 1: Spill shift amount to temp stack slot.
                    let shift_tmp = self.alloc_stack_slot(4);
                    {
                        let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                        let mut spill = MachineInstr::new(I686Opcode::Mov.as_u32());
                        spill.add_operand(MachineOperand::FrameIndex(shift_tmp));
                        spill.add_operand(rhs_reg);
                        mbb.push_instr(spill);
                    }

                    // Step 2: Move lhs value into EAX.
                    {
                        let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                        let mut mov_eax = MachineInstr::new(I686Opcode::Mov.as_u32());
                        mov_eax.add_operand(MachineOperand::Register(EAX));
                        mov_eax.add_operand(lhs_reg);
                        mbb.push_instr(mov_eax);
                    }

                    // Step 3: Load shift amount from stack into ECX.
                    {
                        let mut mov_ecx = MachineInstr::new(I686Opcode::Mov.as_u32());
                        mov_ecx.add_operand(MachineOperand::Register(ECX));
                        mov_ecx.add_operand(MachineOperand::FrameIndex(shift_tmp));
                        mbb.push_instr(mov_ecx);
                    }

                    // Step 4: SHL/SHR/SAR EAX, CL (all physical regs).
                    {
                        let mut inst = MachineInstr::new(opcode.as_u32());
                        inst.add_operand(MachineOperand::Register(EAX));
                        inst.add_operand(MachineOperand::Register(CL));
                        inst.add_implicit_use(EAX);
                        inst.add_implicit_use(ECX);
                        inst.add_implicit_def(EAX);
                        inst.add_implicit_def(EFLAGS);
                        mbb.push_instr(inst);
                    }

                    // Step 5: Move result from EAX to destination vreg.
                    let dst = self.alloc_vreg(result);
                    self.emit_mov(dst, MachineOperand::Register(EAX), mbb);
                }
            }

            // Floating-point ops handled above.
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                unreachable!("FP binops handled in select_fp_binop");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Division helper (shared by SDiv, UDiv, SRem, URem)
    // -----------------------------------------------------------------------

    /// Emits the IDIV/DIV instruction sequence for division and remainder.
    ///
    /// Division on i686 uses the implicit EDX:EAX register pair:
    /// 1. Move dividend into EAX.
    /// 2. Sign-extend (CDQ for signed) or zero EDX.
    /// 3. IDIV/DIV divisor.
    /// 4. Quotient is in EAX, remainder in EDX.
    fn select_div(
        &mut self,
        result: ValueId,
        lhs: MachineOperand,
        rhs: MachineOperand,
        is_signed: bool,
        is_remainder: bool,
        mbb: &mut MachineBasicBlock,
    ) {
        // Division uses an all-physical-register sequence.  We must ensure
        // that loading the dividend and divisor cannot interfere.  The
        // approach: first spill the divisor to the stack (our own temp
        // slot), then load dividend into EAX, CDQ/XOR, then load divisor
        // from the temp slot into ECX, then IDIV.  This guarantees no
        // vreg aliasing issues.

        // Allocate a dedicated temporary frame slot for the divisor.
        let div_tmp_fi = self.alloc_stack_slot(4);
        // Spill divisor to the temp slot.
        {
            let rhs_reg = self.ensure_in_register(rhs, mbb);
            let mut mov = MachineInstr::new(I686Opcode::Mov.as_u32());
            mov.add_operand(MachineOperand::FrameIndex(div_tmp_fi));
            mov.add_operand(rhs_reg);
            mbb.push_instr(mov);
        }

        // Move dividend into EAX (its vreg is consumed here).
        {
            let lhs_reg = self.ensure_in_register(lhs, mbb);
            let mut mov = MachineInstr::new(I686Opcode::Mov.as_u32());
            mov.add_operand(MachineOperand::Register(EAX));
            mov.add_operand(lhs_reg);
            mbb.push_instr(mov);
        }

        // Prepare EDX.
        if is_signed {
            let mut cdq = MachineInstr::new(I686Opcode::Cdq.as_u32());
            cdq.add_implicit_use(EAX);
            cdq.add_implicit_def(EDX);
            mbb.push_instr(cdq);
        } else {
            let mut xor_edx = MachineInstr::new(I686Opcode::Xor.as_u32());
            xor_edx.add_operand(MachineOperand::Register(EDX));
            xor_edx.add_operand(MachineOperand::Register(EDX));
            xor_edx.add_implicit_def(EFLAGS);
            mbb.push_instr(xor_edx);
        }

        // Reload divisor from temp slot into ECX (after EAX/EDX are set).
        {
            let mut mov = MachineInstr::new(I686Opcode::Mov.as_u32());
            mov.add_operand(MachineOperand::Register(ECX));
            mov.add_operand(MachineOperand::FrameIndex(div_tmp_fi));
            mbb.push_instr(mov);
        }

        // Emit IDIV or DIV with ECX as the explicit operand.
        let opcode = if is_signed {
            I686Opcode::Idiv
        } else {
            I686Opcode::Div
        };
        let mut div_inst = MachineInstr::new(opcode.as_u32());
        div_inst.add_operand(MachineOperand::Register(ECX));
        div_inst.add_implicit_use(EAX);
        div_inst.add_implicit_use(EDX);
        div_inst.add_implicit_use(ECX);
        div_inst.add_implicit_def(EAX);
        div_inst.add_implicit_def(EDX);
        div_inst.add_implicit_def(EFLAGS);
        mbb.push_instr(div_inst);

        // Step 5: Move result (quotient or remainder) to the dest vreg.
        let result_reg = if is_remainder {
            MachineOperand::Register(EDX)
        } else {
            MachineOperand::Register(EAX)
        };
        let dst = self.alloc_vreg(result);
        self.emit_mov(dst, result_reg, mbb);
    }

    // -----------------------------------------------------------------------
    // 64-bit integer binop on 32-bit (register pair emulation)
    // -----------------------------------------------------------------------

    /// Handles 64-bit integer binary operations using register pairs.
    fn select_binop_i64(
        &mut self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        // For 64-bit on 32-bit, we need pairs of registers.
        // This is a simplified implementation that covers the core cases.
        let lhs_op = self.get_operand(lhs, func);
        let rhs_op = self.get_operand(rhs, func);
        let dst = self.alloc_vreg(result);

        match op {
            BinOp::Add => {
                // 64-bit add using register pairs:
                //   ADD lo_result, rhs_lo; ADC hi_result, rhs_hi
                let lo_result = self.alloc_vreg_raw();
                let hi_result = self.alloc_vreg_raw();

                let lhs_lo = self.ensure_in_register(lhs_op, mbb);
                let rhs_lo = self.ensure_in_register(rhs_op, mbb);
                let lhs_hi_op = self.i64_high_map.get(&lhs.index()).cloned();
                let rhs_hi_op = self.i64_high_map.get(&rhs.index()).cloned();
                let lo_op = MachineOperand::VirtualReg(ValueId(lo_result));
                let hi_op = MachineOperand::VirtualReg(ValueId(hi_result));

                // MOV lo_result, lhs_lo
                self.emit_mov(lo_op.clone(), lhs_lo, mbb);

                // ADD lo_result, rhs_lo
                let mut add_lo = MachineInstr::new(I686Opcode::Add.as_u32());
                add_lo.add_operand(lo_op);
                add_lo.add_operand(rhs_lo);
                add_lo.add_implicit_def(EFLAGS);
                mbb.push_instr(add_lo);

                // MOV hi_result, lhs_hi
                let lhs_hi = if let Some(hi) = lhs_hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                self.emit_mov(hi_op.clone(), lhs_hi, mbb);

                // ADC hi_result, rhs_hi (add with carry from low addition)
                let rhs_hi = if let Some(hi) = rhs_hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                let mut adc_hi = MachineInstr::new(I686Opcode::Adc.as_u32());
                adc_hi.add_operand(hi_op);
                adc_hi.add_operand(rhs_hi);
                adc_hi.add_implicit_use(EFLAGS);
                adc_hi.add_implicit_def(EFLAGS);
                mbb.push_instr(adc_hi);

                self.value_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(lo_result)),
                );
                self.i64_high_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(hi_result)),
                );
                return;
            }

            BinOp::Sub => {
                // 64-bit sub using register pairs:
                //   SUB lo_result, rhs_lo; SBB hi_result, rhs_hi
                let lo_result = self.alloc_vreg_raw();
                let hi_result = self.alloc_vreg_raw();

                let lhs_lo = self.ensure_in_register(lhs_op, mbb);
                let rhs_lo = self.ensure_in_register(rhs_op, mbb);
                let lhs_hi_op = self.i64_high_map.get(&lhs.index()).cloned();
                let rhs_hi_op = self.i64_high_map.get(&rhs.index()).cloned();
                let lo_op = MachineOperand::VirtualReg(ValueId(lo_result));
                let hi_op = MachineOperand::VirtualReg(ValueId(hi_result));

                // MOV lo_result, lhs_lo
                self.emit_mov(lo_op.clone(), lhs_lo, mbb);

                // SUB lo_result, rhs_lo
                let mut sub_lo = MachineInstr::new(I686Opcode::Sub.as_u32());
                sub_lo.add_operand(lo_op);
                sub_lo.add_operand(rhs_lo);
                sub_lo.add_implicit_def(EFLAGS);
                mbb.push_instr(sub_lo);

                // MOV hi_result, lhs_hi
                let lhs_hi = if let Some(hi) = lhs_hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                self.emit_mov(hi_op.clone(), lhs_hi, mbb);

                // SBB hi_result, rhs_hi (subtract with borrow from low)
                let rhs_hi = if let Some(hi) = rhs_hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                let mut sbb_hi = MachineInstr::new(I686Opcode::Sbb.as_u32());
                sbb_hi.add_operand(hi_op);
                sbb_hi.add_operand(rhs_hi);
                sbb_hi.add_implicit_use(EFLAGS);
                sbb_hi.add_implicit_def(EFLAGS);
                mbb.push_instr(sbb_hi);

                self.value_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(lo_result)),
                );
                self.i64_high_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(hi_result)),
                );
                return;
            }

            BinOp::Mul => {
                // 64-bit multiply: use MUL for the low part, IMUL for cross-products.
                // Simplified: move lhs_lo to EAX, MUL rhs_lo → EDX:EAX
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                let mut mov_eax = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_eax.add_operand(MachineOperand::Register(EAX));
                mov_eax.add_operand(lhs_reg);
                mbb.push_instr(mov_eax);

                let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                let mut mul = MachineInstr::new(I686Opcode::Mul.as_u32());
                mul.add_operand(rhs_reg);
                mul.add_implicit_use(EAX);
                mul.add_implicit_def(EAX);
                mul.add_implicit_def(EDX);
                mul.add_implicit_def(EFLAGS);
                mbb.push_instr(mul);

                // Result lo in EAX, hi in EDX — save both halves.
                // CRITICAL: The first MOV must mark EDX as implicitly used
                // so the register allocator does not assign lo_dst to EDX,
                // which would clobber the hi result before we save it.
                let lo_dst = self.alloc_vreg_raw();
                let mut mov_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
                mov_lo.add_operand(MachineOperand::Register(EAX));
                mov_lo.add_implicit_use(EDX); // keep EDX alive across this MOV
                mbb.push_instr(mov_lo);

                let hi_dst = self.alloc_vreg_raw();
                let mut mov_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
                mov_hi.add_operand(MachineOperand::Register(EDX));
                mbb.push_instr(mov_hi);

                self.value_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
                self.i64_high_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_dst)));
                return;
            }

            BinOp::And | BinOp::Or | BinOp::Xor => {
                // Bitwise ops on 64-bit: apply to both halves independently.
                let opcode = match op {
                    BinOp::And => I686Opcode::And,
                    BinOp::Or => I686Opcode::Or,
                    BinOp::Xor => I686Opcode::Xor,
                    _ => unreachable!(),
                };
                let lhs_lo = self.ensure_in_register(lhs_op, mbb);
                let rhs_lo = self.ensure_in_register(rhs_op, mbb);
                let lhs_hi_op = self.i64_high_map.get(&lhs.index()).cloned();
                let rhs_hi_op = self.i64_high_map.get(&rhs.index()).cloned();
                let lo_result = self.alloc_vreg_raw();
                let hi_result = self.alloc_vreg_raw();
                let lo_op = MachineOperand::VirtualReg(ValueId(lo_result));
                let hi_op = MachineOperand::VirtualReg(ValueId(hi_result));

                // Lo half: MOV lo_result, lhs_lo; OP lo_result, rhs_lo
                self.emit_mov(lo_op.clone(), lhs_lo, mbb);
                let mut op_lo = MachineInstr::new(opcode.as_u32());
                op_lo.add_operand(lo_op);
                op_lo.add_operand(rhs_lo);
                op_lo.add_implicit_def(EFLAGS);
                mbb.push_instr(op_lo);

                // Hi half: MOV hi_result, lhs_hi; OP hi_result, rhs_hi
                let lhs_hi = if let Some(hi) = lhs_hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                self.emit_mov(hi_op.clone(), lhs_hi, mbb);
                let rhs_hi = if let Some(hi) = rhs_hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    MachineOperand::Immediate(0)
                };
                let mut op_hi = MachineInstr::new(opcode.as_u32());
                op_hi.add_operand(hi_op);
                op_hi.add_operand(rhs_hi);
                op_hi.add_implicit_def(EFLAGS);
                mbb.push_instr(op_hi);

                self.value_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(lo_result)),
                );
                self.i64_high_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(hi_result)),
                );
                return;
            }

            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                // Full 64-bit shift on i686 with correct hi/lo handling.
                // Get hi/lo parts of the source operand.
                let lhs_lo = self.ensure_in_register(lhs_op, mbb);
                let lhs_hi_op = self
                    .i64_high_map
                    .get(&lhs.index())
                    .cloned()
                    .unwrap_or(MachineOperand::Immediate(0));
                let lhs_hi = self.ensure_in_register(lhs_hi_op, mbb);

                // Check if the shift amount is a known constant.
                let shift_imm = match &rhs_op {
                    MachineOperand::Immediate(v) => Some(*v as u32),
                    _ => None,
                };

                let lo_result = self.alloc_vreg_raw();
                let hi_result = self.alloc_vreg_raw();

                if let Some(amt) = shift_imm {
                    if amt == 0 {
                        // No shift: result = input
                        self.emit_mov(MachineOperand::VirtualReg(ValueId(lo_result)), lhs_lo, mbb);
                        self.emit_mov(MachineOperand::VirtualReg(ValueId(hi_result)), lhs_hi, mbb);
                    } else if amt >= 64 {
                        // Shift by >= 64: result is 0 (or sign-extended for AShr)
                        self.emit_mov(
                            MachineOperand::VirtualReg(ValueId(lo_result)),
                            MachineOperand::Immediate(0),
                            mbb,
                        );
                        if matches!(op, BinOp::AShr) {
                            // AShr >= 64: fill with sign bit of hi
                            self.emit_mov(
                                MachineOperand::VirtualReg(ValueId(hi_result)),
                                lhs_hi.clone(),
                                mbb,
                            );
                            let mut sar = MachineInstr::new(I686Opcode::Sar.as_u32());
                            sar.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                            sar.add_operand(MachineOperand::Immediate(31));
                            sar.add_implicit_def(EFLAGS);
                            mbb.push_instr(sar);
                            self.emit_mov(
                                MachineOperand::VirtualReg(ValueId(lo_result)),
                                MachineOperand::VirtualReg(ValueId(hi_result)),
                                mbb,
                            );
                        } else {
                            self.emit_mov(
                                MachineOperand::VirtualReg(ValueId(hi_result)),
                                MachineOperand::Immediate(0),
                                mbb,
                            );
                        }
                    } else if amt == 32 {
                        match op {
                            BinOp::Shl => {
                                // hi = lo, lo = 0
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_lo,
                                    mbb,
                                );
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    MachineOperand::Immediate(0),
                                    mbb,
                                );
                            }
                            BinOp::LShr => {
                                // lo = hi, hi = 0
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    MachineOperand::Immediate(0),
                                    mbb,
                                );
                            }
                            BinOp::AShr => {
                                // lo = hi, hi = hi >> 31 (sign extend)
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_hi.clone(),
                                    mbb,
                                );
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                let mut sar = MachineInstr::new(I686Opcode::Sar.as_u32());
                                sar.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                sar.add_operand(MachineOperand::Immediate(31));
                                sar.add_implicit_def(EFLAGS);
                                mbb.push_instr(sar);
                            }
                            _ => unreachable!(),
                        }
                    } else if amt > 32 {
                        let sub_amt = amt - 32;
                        match op {
                            BinOp::Shl => {
                                // hi = lo << (amt-32), lo = 0
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_lo,
                                    mbb,
                                );
                                let mut shl = MachineInstr::new(I686Opcode::Shl.as_u32());
                                shl.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                shl.add_operand(MachineOperand::Immediate(sub_amt as i64));
                                shl.add_implicit_def(EFLAGS);
                                mbb.push_instr(shl);
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    MachineOperand::Immediate(0),
                                    mbb,
                                );
                            }
                            BinOp::LShr => {
                                // lo = hi >> (amt-32), hi = 0
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                let mut shr = MachineInstr::new(I686Opcode::Shr.as_u32());
                                shr.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                shr.add_operand(MachineOperand::Immediate(sub_amt as i64));
                                shr.add_implicit_def(EFLAGS);
                                mbb.push_instr(shr);
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    MachineOperand::Immediate(0),
                                    mbb,
                                );
                            }
                            BinOp::AShr => {
                                // lo = hi >> (amt-32) (arithmetic), hi = hi >> 31
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_hi.clone(),
                                    mbb,
                                );
                                let mut sar = MachineInstr::new(I686Opcode::Sar.as_u32());
                                sar.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                sar.add_operand(MachineOperand::Immediate(sub_amt as i64));
                                sar.add_implicit_def(EFLAGS);
                                mbb.push_instr(sar);
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                let mut sar2 = MachineInstr::new(I686Opcode::Sar.as_u32());
                                sar2.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                sar2.add_operand(MachineOperand::Immediate(31));
                                sar2.add_implicit_def(EFLAGS);
                                mbb.push_instr(sar2);
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        // amt < 32 (and > 0)
                        let comp_amt = 32 - amt;
                        match op {
                            BinOp::Shl => {
                                // hi = (hi << amt) | (lo >> (32-amt))
                                // lo = lo << amt
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                let mut shl_hi = MachineInstr::new(I686Opcode::Shl.as_u32());
                                shl_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                shl_hi.add_operand(MachineOperand::Immediate(amt as i64));
                                shl_hi.add_implicit_def(EFLAGS);
                                mbb.push_instr(shl_hi);
                                let tmp = self.alloc_vreg_raw();
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(tmp)),
                                    lhs_lo.clone(),
                                    mbb,
                                );
                                let mut shr_tmp = MachineInstr::new(I686Opcode::Shr.as_u32());
                                shr_tmp.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                                shr_tmp.add_operand(MachineOperand::Immediate(comp_amt as i64));
                                shr_tmp.add_implicit_def(EFLAGS);
                                mbb.push_instr(shr_tmp);
                                let mut or_hi = MachineInstr::new(I686Opcode::Or.as_u32());
                                or_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                or_hi.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                                or_hi.add_implicit_def(EFLAGS);
                                mbb.push_instr(or_hi);
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_lo,
                                    mbb,
                                );
                                let mut shl_lo = MachineInstr::new(I686Opcode::Shl.as_u32());
                                shl_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                shl_lo.add_operand(MachineOperand::Immediate(amt as i64));
                                shl_lo.add_implicit_def(EFLAGS);
                                mbb.push_instr(shl_lo);
                            }
                            BinOp::LShr => {
                                // lo = (lo >> amt) | (hi << (32-amt))
                                // hi = hi >> amt
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_lo,
                                    mbb,
                                );
                                let mut shr_lo = MachineInstr::new(I686Opcode::Shr.as_u32());
                                shr_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                shr_lo.add_operand(MachineOperand::Immediate(amt as i64));
                                shr_lo.add_implicit_def(EFLAGS);
                                mbb.push_instr(shr_lo);
                                let tmp = self.alloc_vreg_raw();
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(tmp)),
                                    lhs_hi.clone(),
                                    mbb,
                                );
                                let mut shl_tmp = MachineInstr::new(I686Opcode::Shl.as_u32());
                                shl_tmp.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                                shl_tmp.add_operand(MachineOperand::Immediate(comp_amt as i64));
                                shl_tmp.add_implicit_def(EFLAGS);
                                mbb.push_instr(shl_tmp);
                                let mut or_lo = MachineInstr::new(I686Opcode::Or.as_u32());
                                or_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                or_lo.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                                or_lo.add_implicit_def(EFLAGS);
                                mbb.push_instr(or_lo);
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                let mut shr_hi = MachineInstr::new(I686Opcode::Shr.as_u32());
                                shr_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                shr_hi.add_operand(MachineOperand::Immediate(amt as i64));
                                shr_hi.add_implicit_def(EFLAGS);
                                mbb.push_instr(shr_hi);
                            }
                            BinOp::AShr => {
                                // lo = (lo >> amt) | (hi << (32-amt))
                                // hi = hi >> amt (arithmetic)
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(lo_result)),
                                    lhs_lo,
                                    mbb,
                                );
                                let mut shr_lo = MachineInstr::new(I686Opcode::Shr.as_u32());
                                shr_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                shr_lo.add_operand(MachineOperand::Immediate(amt as i64));
                                shr_lo.add_implicit_def(EFLAGS);
                                mbb.push_instr(shr_lo);
                                let tmp = self.alloc_vreg_raw();
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(tmp)),
                                    lhs_hi.clone(),
                                    mbb,
                                );
                                let mut shl_tmp = MachineInstr::new(I686Opcode::Shl.as_u32());
                                shl_tmp.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                                shl_tmp.add_operand(MachineOperand::Immediate(comp_amt as i64));
                                shl_tmp.add_implicit_def(EFLAGS);
                                mbb.push_instr(shl_tmp);
                                let mut or_lo = MachineInstr::new(I686Opcode::Or.as_u32());
                                or_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                                or_lo.add_operand(MachineOperand::VirtualReg(ValueId(tmp)));
                                or_lo.add_implicit_def(EFLAGS);
                                mbb.push_instr(or_lo);
                                self.emit_mov(
                                    MachineOperand::VirtualReg(ValueId(hi_result)),
                                    lhs_hi,
                                    mbb,
                                );
                                let mut sar_hi = MachineInstr::new(I686Opcode::Sar.as_u32());
                                sar_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_result)));
                                sar_hi.add_operand(MachineOperand::Immediate(amt as i64));
                                sar_hi.add_implicit_def(EFLAGS);
                                mbb.push_instr(sar_hi);
                            }
                            _ => unreachable!(),
                        }
                    }
                } else {
                    // Dynamic shift amount — use a simplified approach:
                    // just shift the lo half (matching old behavior for now).
                    let opcode = match op {
                        BinOp::Shl => I686Opcode::Shl,
                        BinOp::LShr => I686Opcode::Shr,
                        BinOp::AShr => I686Opcode::Sar,
                        _ => unreachable!(),
                    };
                    let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                    self.emit_mov(MachineOperand::VirtualReg(ValueId(lo_result)), lhs_lo, mbb);
                    let mut mov_cl = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mov_cl.add_operand(MachineOperand::Register(ECX));
                    mov_cl.add_operand(rhs_reg);
                    mbb.push_instr(mov_cl);
                    let mut shift = MachineInstr::new(opcode.as_u32());
                    shift.add_operand(MachineOperand::VirtualReg(ValueId(lo_result)));
                    shift.add_operand(MachineOperand::Register(CL));
                    shift.add_implicit_def(EFLAGS);
                    shift.add_implicit_use(ECX);
                    mbb.push_instr(shift);
                    self.emit_mov(
                        MachineOperand::VirtualReg(ValueId(hi_result)),
                        MachineOperand::Immediate(0),
                        mbb,
                    );
                }

                self.value_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(lo_result)),
                );
                self.i64_high_map.insert(
                    result.index(),
                    MachineOperand::VirtualReg(ValueId(hi_result)),
                );
                return;
            }

            BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem => {
                // 64-bit division on 32-bit is extremely complex; typically
                // delegated to a runtime helper (__divdi3, __udivdi3, etc.).
                // Emit a CALL to the appropriate runtime function.
                let helper_name = match op {
                    BinOp::SDiv => "__divdi3",
                    BinOp::UDiv => "__udivdi3",
                    BinOp::SRem => "__moddi3",
                    BinOp::URem => "__umoddi3",
                    _ => unreachable!(),
                };
                // Push args (rhs then lhs, right-to-left).
                let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                let mut push_rhs = MachineInstr::new(I686Opcode::Push.as_u32());
                push_rhs.add_operand(rhs_reg);
                push_rhs.add_implicit_use(ESP);
                push_rhs.add_implicit_def(ESP);
                mbb.push_instr(push_rhs);

                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                let mut push_lhs = MachineInstr::new(I686Opcode::Push.as_u32());
                push_lhs.add_operand(lhs_reg);
                push_lhs.add_implicit_use(ESP);
                push_lhs.add_implicit_def(ESP);
                mbb.push_instr(push_lhs);

                let mut call = MachineInstr::new(I686Opcode::Call.as_u32());
                call.add_operand(MachineOperand::Symbol(helper_name.to_string()));
                call.set_call();
                for &r in &CALLER_SAVED {
                    call.add_implicit_def(r);
                }
                call.add_implicit_def(EFLAGS);
                call.add_implicit_use(ESP);
                call.add_implicit_def(ESP);
                mbb.push_instr(call);

                // Clean up stack (2 arguments × 4 bytes = 8 bytes).
                let mut cleanup = MachineInstr::new(I686Opcode::Add.as_u32());
                cleanup.add_operand(MachineOperand::Register(ESP));
                cleanup.add_operand(MachineOperand::Immediate(8));
                cleanup.add_implicit_def(EFLAGS);
                mbb.push_instr(cleanup);

                self.has_calls = true;
                self.value_map
                    .insert(result.index(), MachineOperand::Register(EAX));
                return;
            }

            // FP ops should not reach here.
            _ => {}
        }

        // Default: record result as the lo vreg.
        self.value_map.entry(result.index()).or_insert(dst);
    }

    // -----------------------------------------------------------------------
    // Floating-point binary ops (x87 FPU)
    // -----------------------------------------------------------------------

    /// Handles floating-point binary operations using the x87 FPU stack.
    ///
    /// Pattern for `result = lhs OP rhs`:
    /// 1. FLD lhs (push lhs onto x87 stack → ST(0))
    /// 2. FLD rhs (push rhs → ST(0), lhs moves to ST(1))
    /// 3. FADDP/FSUBP/FMULP/FDIVP (ST(1) OP ST(0), pop → result in ST(0))
    fn select_fp_binop(
        &mut self,
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        _ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let is_f64 = matches!(_ty, IrType::F64 | IrType::F80);

        // Load LHS onto x87 stack.
        self.emit_fld_value(lhs, is_f64, mbb, func);
        // Load RHS onto x87 stack (LHS moves to ST(1)).
        self.emit_fld_value(rhs, is_f64, mbb, func);

        // Select the appropriate FPU pop instruction.
        let opcode = match op {
            BinOp::FAdd => I686Opcode::Faddp,
            BinOp::FSub => I686Opcode::Fsubp,
            BinOp::FMul => I686Opcode::Fmulp,
            BinOp::FDiv => I686Opcode::Fdivp,
            BinOp::FRem => {
                // x87 doesn't have a direct FREM-with-pop; we use FPREM followed
                // by FSTP ST(1). For simplicity, emit FDIVP as placeholder and
                // let a later pass handle the proper FPREM sequence.
                I686Opcode::Fdivp
            }
            _ => unreachable!("Non-FP op in select_fp_binop"),
        };

        let mut fop = MachineInstr::new(opcode.as_u32());
        // FADDP/FSUBP/FMULP/FDIVP default to ST(1),ST(0): add ST(0) to
        // ST(1), pop.  Encode the "1" so the encoder emits DE C1, not DE C0.
        fop.add_operand(MachineOperand::Immediate(1));
        fop.add_implicit_use(ST0);
        fop.add_implicit_def(ST0);
        mbb.push_instr(fop);

        // Result is now in ST(0).  Store to memory and reload into GPR(s)
        // so all float values are consistently in GPRs, not on the x87 stack.
        let is_f64 = matches!(_ty, IrType::F64 | IrType::F80);
        let slot_size: u32 = if is_f64 { 8 } else { 4 };
        let slot = self.frame_size;
        self.frame_size += slot_size;
        let fi = MachineOperand::FrameIndex((-(slot as i32 + slot_size as i32)) as u32);
        let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
        fstp.add_operand(fi.clone());
        if is_f64 {
            fstp.add_operand(MachineOperand::Immediate(64));
        }
        fstp.add_implicit_use(ST0);
        mbb.push_instr(fstp);
        if is_f64 {
            let lo_idx = self.alloc_vreg_raw();
            let mut lo_ld = MachineInstr::new(I686Opcode::Mov.as_u32());
            lo_ld.add_operand(MachineOperand::VirtualReg(ValueId(lo_idx)));
            lo_ld.add_operand(fi.clone());
            mbb.push_instr(lo_ld);
            let hi_fi = self.offset_memory(fi, 4);
            let hi_idx = self.alloc_vreg_raw();
            let mut hi_ld = MachineInstr::new(I686Opcode::Mov.as_u32());
            hi_ld.add_operand(MachineOperand::VirtualReg(ValueId(hi_idx)));
            hi_ld.add_operand(hi_fi);
            mbb.push_instr(hi_ld);
            self.value_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_idx)));
            self.i64_high_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_idx)));
        } else {
            let dst_idx = self.alloc_vreg_raw();
            let mut ld = MachineInstr::new(I686Opcode::Mov.as_u32());
            ld.add_operand(MachineOperand::VirtualReg(ValueId(dst_idx)));
            ld.add_operand(fi);
            mbb.push_instr(ld);
            self.value_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(dst_idx)));
        }
    }

    // -----------------------------------------------------------------------
    // ICmp selection
    // -----------------------------------------------------------------------

    /// Selects instructions for an integer comparison.
    ///
    /// Emits: CMP lhs, rhs → SETCC cc → MOVZX (to get a full 32-bit 0/1).
    fn select_icmp(
        &mut self,
        result: ValueId,
        pred: &ICmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let lhs_op = self.get_operand(lhs, func);
        let rhs_op = self.get_operand(rhs, func);

        // CMP lhs, rhs
        let lhs_reg = self.ensure_in_register(lhs_op, mbb);
        let mut cmp = MachineInstr::new(I686Opcode::Cmp.as_u32());
        cmp.add_operand(lhs_reg);
        cmp.add_operand(rhs_op);
        cmp.add_implicit_def(EFLAGS);
        mbb.push_instr(cmp);

        // SETcc al (set byte based on condition)
        let cc = ConditionCode::from_icmp_predicate(pred);
        let mut setcc = MachineInstr::new(I686Opcode::Setcc.as_u32());
        setcc.add_operand(MachineOperand::Register(AL));
        setcc.add_operand(MachineOperand::Immediate(cc.encoding() as i64));
        setcc.add_implicit_use(EFLAGS);
        mbb.push_instr(setcc);

        // MOVZX eax, al (zero-extend byte result to 32-bit)
        let dst = self.alloc_vreg(result);
        let mut movzx = MachineInstr::new(I686Opcode::MovZx.as_u32());
        movzx.add_operand(dst);
        movzx.add_operand(MachineOperand::Register(AL));
        mbb.push_instr(movzx);
    }

    // -----------------------------------------------------------------------
    // FCmp selection
    // -----------------------------------------------------------------------

    /// Selects instructions for a floating-point comparison.
    ///
    /// Uses FUCOMIP to compare two x87 values and set EFLAGS directly.
    fn select_fcmp(
        &mut self,
        result: ValueId,
        pred: &FCmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        // Determine operand type for correct F32/F64 FLD encoding.
        let is_f64 = self.i64_high_map.contains_key(&lhs.index())
            || self.i64_high_map.contains_key(&rhs.index())
            || matches!(func.get_value_type(lhs), IrType::F64 | IrType::F80);

        // Load both operands onto x87 stack.
        // FUCOMIP compares ST(0) vs ST(1), so we need lhs in ST(0).
        // Load rhs first (goes to ST(0)), then lhs (pushes rhs to ST(1)).
        // Result: ST(0)=lhs, ST(1)=rhs → FUCOMIP compares lhs vs rhs.
        self.emit_fld_value(rhs, is_f64, mbb, func);
        self.emit_fld_value(lhs, is_f64, mbb, func);

        // FUCOMIP ST(0), ST(1) — unordered compare, pop ST(0), set EFLAGS.
        // The explicit operand specifies the ST(i) register to compare
        // against.  ST(0) is always the implicit first operand.
        let mut fucomip = MachineInstr::new(I686Opcode::Fucomip.as_u32());
        fucomip.add_operand(MachineOperand::Register(registers::ST1));
        fucomip.add_implicit_use(ST0);
        fucomip.add_implicit_def(EFLAGS);
        mbb.push_instr(fucomip);

        // FSTP ST(0) — pop the remaining value from x87 stack.
        let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
        fstp.add_operand(MachineOperand::Register(ST0));
        fstp.add_implicit_def(ST0);
        mbb.push_instr(fstp);

        // Map FCmpPredicate to x86 condition code.
        // FUCOMIP sets CF and ZF like an unsigned integer comparison:
        //   ST(0) > ST(1): CF=0, ZF=0  → use A (above)
        //   ST(0) < ST(1): CF=1, ZF=0  → use B (below)
        //   ST(0) = ST(1): CF=0, ZF=1  → use E (equal)
        //   Unordered:     CF=1, ZF=1, PF=1
        let cc = match pred {
            FCmpPredicate::OEq | FCmpPredicate::UEq => ConditionCode::E,
            FCmpPredicate::ONe | FCmpPredicate::UNe => ConditionCode::Ne,
            FCmpPredicate::Ogt | FCmpPredicate::Ugt => ConditionCode::A,
            FCmpPredicate::Oge | FCmpPredicate::Uge => ConditionCode::Ae,
            FCmpPredicate::Olt | FCmpPredicate::Ult => ConditionCode::B,
            FCmpPredicate::Ole | FCmpPredicate::Ule => ConditionCode::Be,
            FCmpPredicate::Ord => ConditionCode::Np, // PF=0 means ordered
            FCmpPredicate::Uno => ConditionCode::P,  // PF=1 means unordered
        };

        // SETcc al
        let mut setcc = MachineInstr::new(I686Opcode::Setcc.as_u32());
        setcc.add_operand(MachineOperand::Register(AL));
        setcc.add_operand(MachineOperand::Immediate(cc.encoding() as i64));
        setcc.add_implicit_use(EFLAGS);
        mbb.push_instr(setcc);

        // MOVZX eax, al
        let dst = self.alloc_vreg(result);
        let mut movzx = MachineInstr::new(I686Opcode::MovZx.as_u32());
        movzx.add_operand(dst);
        movzx.add_operand(MachineOperand::Register(AL));
        mbb.push_instr(movzx);
    }

    // -----------------------------------------------------------------------
    // Branch selection
    // -----------------------------------------------------------------------

    /// Selects instructions for an unconditional branch.
    fn select_branch(&self, target: BasicBlockId, mbb: &mut MachineBasicBlock) {
        let target_id = self.block_map.get(&target.index()).copied().unwrap_or(0);
        let mut jmp = MachineInstr::new(I686Opcode::Jmp.as_u32());
        jmp.add_operand(MachineOperand::Label(target_id));
        jmp.set_terminator();
        mbb.push_instr(jmp);
    }

    // -----------------------------------------------------------------------
    // Conditional branch selection
    // -----------------------------------------------------------------------

    /// Selects instructions for a conditional branch.
    ///
    /// The condition value is expected to be a boolean (I1). We TEST it
    /// against itself and emit JNE (true) / JMP (false).
    fn select_cond_branch(
        &mut self,
        condition: ValueId,
        true_target: BasicBlockId,
        false_target: BasicBlockId,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let cond_op = self.get_operand(condition, func);
        let cond_reg = self.ensure_in_register(cond_op, mbb);

        // TEST cond, cond (sets ZF=1 if cond==0)
        let mut test = MachineInstr::new(I686Opcode::Test.as_u32());
        test.add_operand(cond_reg.clone());
        test.add_operand(cond_reg);
        test.add_implicit_def(EFLAGS);
        mbb.push_instr(test);

        // JNE true_target (jump if condition != 0)
        let true_id = self
            .block_map
            .get(&true_target.index())
            .copied()
            .unwrap_or(0);
        let mut jne = MachineInstr::new(I686Opcode::Jcc.as_u32());
        jne.add_operand(MachineOperand::Label(true_id));
        jne.add_operand(MachineOperand::Immediate(
            ConditionCode::Ne.encoding() as i64
        ));
        jne.add_implicit_use(EFLAGS);
        jne.set_terminator();
        mbb.push_instr(jne);

        // JMP false_target (fallthrough to false path)
        let false_id = self
            .block_map
            .get(&false_target.index())
            .copied()
            .unwrap_or(0);
        let mut jmp = MachineInstr::new(I686Opcode::Jmp.as_u32());
        jmp.add_operand(MachineOperand::Label(false_id));
        jmp.set_terminator();
        mbb.push_instr(jmp);
    }

    // -----------------------------------------------------------------------
    // Switch selection
    // -----------------------------------------------------------------------

    /// Selects instructions for a multi-way switch.
    ///
    /// Emits a cascaded comparison chain: CMP val, case_i → JE target_i,
    /// followed by JMP default.
    fn select_switch(
        &mut self,
        value: ValueId,
        default: BasicBlockId,
        cases: &[(i64, BasicBlockId)],
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let val_op = self.get_operand(value, func);
        let val_reg = self.ensure_in_register(val_op, mbb);

        for (case_val, target) in cases {
            // CMP val, case_val
            let mut cmp = MachineInstr::new(I686Opcode::Cmp.as_u32());
            cmp.add_operand(val_reg.clone());
            cmp.add_operand(MachineOperand::Immediate(*case_val));
            cmp.add_implicit_def(EFLAGS);
            mbb.push_instr(cmp);

            // JE target
            let target_id = self.block_map.get(&target.index()).copied().unwrap_or(0);
            let mut je = MachineInstr::new(I686Opcode::Jcc.as_u32());
            je.add_operand(MachineOperand::Label(target_id));
            je.add_operand(MachineOperand::Immediate(ConditionCode::E.encoding() as i64));
            je.add_implicit_use(EFLAGS);
            je.set_terminator();
            mbb.push_instr(je);
        }

        // JMP default
        let default_id = self.block_map.get(&default.index()).copied().unwrap_or(0);
        let mut jmp = MachineInstr::new(I686Opcode::Jmp.as_u32());
        jmp.add_operand(MachineOperand::Label(default_id));
        jmp.set_terminator();
        mbb.push_instr(jmp);
    }

    // -----------------------------------------------------------------------
    // Call selection (cdecl)
    // -----------------------------------------------------------------------

    /// Selects instructions for a function call under the cdecl convention.
    ///
    /// 1. Push arguments right-to-left onto the stack.
    /// 2. CALL target.
    /// 3. ADD ESP, N*4 (caller cleanup).
    /// 4. Move return value from EAX (or ST(0) for float) to dest vreg.
    fn select_call(
        &mut self,
        result: Option<ValueId>,
        callee: ValueId,
        args: &[ValueId],
        _is_tail: bool,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        self.has_calls = true;

        // Step 1: Push arguments in reverse order (right-to-left for cdecl).
        // Calculate total argument bytes accounting for I64 (8 bytes) and
        // FP double (8 bytes) arguments.
        let mut arg_bytes: u32 = 0;
        for &arg_vid in args.iter() {
            let arg_ty = self.get_value_type(arg_vid, func);
            match &arg_ty {
                crate::ir::types::IrType::I64 => arg_bytes += 8,
                crate::ir::types::IrType::F64 => arg_bytes += 8,
                crate::ir::types::IrType::F32 => arg_bytes += 4,
                _ => arg_bytes += 4,
            }
        }

        for &arg_vid in args.iter().rev() {
            let arg_op = self.get_operand(arg_vid, func);
            let arg_ty = self.get_value_type(arg_vid, func);

            // Check if this is an x87 FP value.
            if let MachineOperand::Register(reg) = &arg_op {
                if registers::is_fpu(*reg) {
                    // Allocate stack space and store FP value.
                    let mut sub_esp = MachineInstr::new(I686Opcode::Sub.as_u32());
                    sub_esp.add_operand(MachineOperand::Register(ESP));
                    sub_esp.add_operand(MachineOperand::Immediate(8)); // double = 8 bytes
                    sub_esp.add_implicit_def(EFLAGS);
                    mbb.push_instr(sub_esp);

                    let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
                    fstp.add_operand(MachineOperand::Memory {
                        base: ESP,
                        offset: 0,
                        index: None,
                        scale: 1,
                    });
                    fstp.add_implicit_use(ST0);
                    mbb.push_instr(fstp);
                    continue;
                }
            }

            // 64-bit value (I64 or F64): push high word first, then low word.
            if matches!(
                &arg_ty,
                crate::ir::types::IrType::I64 | crate::ir::types::IrType::F64
            ) {
                // Push high 32 bits first (higher address in memory).
                let hi_op = self.i64_high_map.get(&arg_vid.index()).cloned();
                let hi_src = if let Some(hi) = hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    // No tracked high word — push zero (common for small constants).
                    let tmp = self.alloc_vreg_raw();
                    self.emit_mov(
                        MachineOperand::VirtualReg(ValueId(tmp)),
                        MachineOperand::Immediate(0),
                        mbb,
                    );
                    MachineOperand::VirtualReg(ValueId(tmp))
                };
                let mut push_hi = MachineInstr::new(I686Opcode::Push.as_u32());
                push_hi.add_operand(hi_src);
                push_hi.add_implicit_use(ESP);
                push_hi.add_implicit_def(ESP);
                mbb.push_instr(push_hi);

                // Push low 32 bits (will be at lower address = first word read).
                let lo_src = self.ensure_in_register(arg_op, mbb);
                let mut push_lo = MachineInstr::new(I686Opcode::Push.as_u32());
                push_lo.add_operand(lo_src);
                push_lo.add_implicit_use(ESP);
                push_lo.add_implicit_def(ESP);
                mbb.push_instr(push_lo);
                continue;
            }

            let arg_reg = self.ensure_in_register(arg_op, mbb);
            let mut push = MachineInstr::new(I686Opcode::Push.as_u32());
            push.add_operand(arg_reg);
            push.add_implicit_use(ESP);
            push.add_implicit_def(ESP);
            mbb.push_instr(push);
        }

        // Step 2: CALL.
        let callee_op = self.get_operand(callee, func);
        let mut call_inst = MachineInstr::new(I686Opcode::Call.as_u32());
        call_inst.add_operand(callee_op);
        call_inst.set_call();
        // Caller-saved registers are clobbered.
        for &r in &CALLER_SAVED {
            call_inst.add_implicit_def(r);
        }
        call_inst.add_implicit_def(EFLAGS);
        call_inst.add_implicit_use(ESP);
        call_inst.add_implicit_def(ESP);
        mbb.push_instr(call_inst);

        // Step 3: Caller cleanup — remove arguments from stack.
        if arg_bytes > 0 {
            let mut cleanup = MachineInstr::new(I686Opcode::Add.as_u32());
            cleanup.add_operand(MachineOperand::Register(ESP));
            cleanup.add_operand(MachineOperand::Immediate(arg_bytes as i64));
            cleanup.add_implicit_def(EFLAGS);
            mbb.push_instr(cleanup);
        }

        // Step 4: Move return value to result vreg.
        if let Some(res_vid) = result {
            // Determine if the return type is floating-point.
            let res_type = self.get_value_type(res_vid, func);
            if res_type.is_floating() {
                // Return value is in ST(0).
                if matches!(&res_type, IrType::F64 | IrType::F80) {
                    // F64: ST(0) is volatile — spill to GPR pair immediately.
                    // SUB ESP, 8 / FSTP QWORD [ESP] / POP lo / POP hi
                    let mut sub = MachineInstr::new(I686Opcode::Sub.as_u32());
                    sub.add_operand(MachineOperand::Register(ESP));
                    sub.add_operand(MachineOperand::Immediate(8));
                    sub.add_implicit_def(EFLAGS);
                    mbb.push_instr(sub);

                    let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
                    fstp.add_operand(MachineOperand::Memory {
                        base: ESP,
                        offset: 0,
                        index: None,
                        scale: 1,
                    });
                    fstp.add_operand(MachineOperand::Immediate(64));
                    fstp.add_implicit_use(ST0);
                    mbb.push_instr(fstp);

                    let lo_vreg = self.alloc_vreg_raw();
                    let mut pop_lo = MachineInstr::new(I686Opcode::Pop.as_u32());
                    pop_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_vreg)));
                    pop_lo.add_implicit_use(ESP);
                    pop_lo.add_implicit_def(ESP);
                    mbb.push_instr(pop_lo);

                    let hi_vreg = self.alloc_vreg_raw();
                    let mut pop_hi = MachineInstr::new(I686Opcode::Pop.as_u32());
                    pop_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_vreg)));
                    pop_hi.add_implicit_use(ESP);
                    pop_hi.add_implicit_def(ESP);
                    mbb.push_instr(pop_hi);

                    self.value_map.insert(
                        res_vid.index(),
                        MachineOperand::VirtualReg(ValueId(lo_vreg)),
                    );
                    self.i64_high_map.insert(
                        res_vid.index(),
                        MachineOperand::VirtualReg(ValueId(hi_vreg)),
                    );
                } else {
                    // F32: ST(0) is volatile — spill to GPR immediately.
                    // Multiple consecutive calls would push prior results
                    // down the x87 stack, corrupting positional ST(0) refs.
                    // SUB ESP, 4 / FSTP DWORD [ESP] / POP vreg
                    let mut sub = MachineInstr::new(I686Opcode::Sub.as_u32());
                    sub.add_operand(MachineOperand::Register(ESP));
                    sub.add_operand(MachineOperand::Immediate(4));
                    sub.add_implicit_def(EFLAGS);
                    mbb.push_instr(sub);

                    let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
                    fstp.add_operand(MachineOperand::Memory {
                        base: ESP,
                        offset: 0,
                        index: None,
                        scale: 1,
                    });
                    fstp.add_implicit_use(ST0);
                    mbb.push_instr(fstp);

                    let vreg = self.alloc_vreg_raw();
                    let mut pop = MachineInstr::new(I686Opcode::Pop.as_u32());
                    pop.add_operand(MachineOperand::VirtualReg(ValueId(vreg)));
                    pop.add_implicit_use(ESP);
                    pop.add_implicit_def(ESP);
                    mbb.push_instr(pop);

                    self.value_map
                        .insert(res_vid.index(), MachineOperand::VirtualReg(ValueId(vreg)));
                }
            } else if matches!(&res_type, crate::ir::types::IrType::I64) {
                // 64-bit integer return value in EDX:EAX.
                // Mark EDX as implicitly used in the first MOV to prevent
                // the register allocator from clobbering it.
                let lo_dst = self.alloc_vreg_raw();
                let mut mov_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
                mov_lo.add_operand(MachineOperand::Register(EAX));
                mov_lo.add_implicit_use(EDX);
                mbb.push_instr(mov_lo);

                let hi_dst = self.alloc_vreg_raw();
                let mut mov_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
                mov_hi.add_operand(MachineOperand::Register(EDX));
                mbb.push_instr(mov_hi);

                self.value_map
                    .insert(res_vid.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
                self.i64_high_map
                    .insert(res_vid.index(), MachineOperand::VirtualReg(ValueId(hi_dst)));
            } else {
                // Integer return value in EAX.
                let dst = self.alloc_vreg(res_vid);
                self.emit_mov(dst, MachineOperand::Register(EAX), mbb);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Return selection
    // -----------------------------------------------------------------------

    /// Selects instructions for a function return.
    ///
    /// Moves the return value to EAX (integer) or leaves it in ST(0) (float),
    /// then emits the epilogue (restore callee-saved, MOV ESP EBP, POP EBP, RET).
    fn select_return(
        &mut self,
        value: Option<ValueId>,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        // Move return value to the appropriate register.
        if let Some(val_vid) = value {
            let val_op = self.get_operand(val_vid, func);
            let val_type = self.get_value_type(val_vid, func);

            if val_type.is_floating() {
                // Ensure the FP value is in ST(0) for the caller to
                // receive via the x87 return convention.
                let is_f64_ret = matches!(&val_type, IrType::F64 | IrType::F80);
                if is_f64_ret {
                    // F64: use emit_fld_value which handles both
                    // GPR pairs and FrameIndex/Memory with QWORD hint.
                    self.emit_fld_value(val_vid, true, mbb, func);
                } else if !matches!(&val_op, MachineOperand::Register(r) if registers::is_fpu(*r)) {
                    self.emit_fld(val_op, mbb);
                }
            } else if matches!(&val_type, crate::ir::types::IrType::I64) {
                // 64-bit return: lo in EAX, hi in EDX.
                let lo_src = self.ensure_in_register(val_op, mbb);
                let mut mov_eax = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_eax.add_operand(MachineOperand::Register(EAX));
                mov_eax.add_operand(lo_src);
                mbb.push_instr(mov_eax);

                let hi_op = self.i64_high_map.get(&val_vid.index()).cloned();
                let hi_src = if let Some(hi) = hi_op {
                    self.ensure_in_register(hi, mbb)
                } else {
                    let tmp = self.alloc_vreg_raw();
                    self.emit_mov(
                        MachineOperand::VirtualReg(ValueId(tmp)),
                        MachineOperand::Immediate(0),
                        mbb,
                    );
                    MachineOperand::VirtualReg(ValueId(tmp))
                };
                let mut mov_edx = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov_edx.add_operand(MachineOperand::Register(EDX));
                mov_edx.add_operand(hi_src);
                mbb.push_instr(mov_edx);
            } else {
                // Move integer result to EAX.
                let src = self.ensure_in_register(val_op, mbb);
                let mut mov = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov.add_operand(MachineOperand::Register(EAX));
                mov.add_operand(src);
                mbb.push_instr(mov);
            }
        }

        // Emit a bare RET marker. The full epilogue (pop callee-saved,
        // mov esp/ebp, pop ebp) is injected by ArchCodegen::emit_epilogue
        // in generation.rs, which replaces instructions with is_return=true.
        let mut ret = MachineInstr::new(I686Opcode::Ret.as_u32());
        ret.set_return();
        mbb.push_instr(ret);
    }

    // -----------------------------------------------------------------------
    // GetElementPtr selection
    // -----------------------------------------------------------------------

    /// Selects instructions for a GEP (pointer arithmetic).
    ///
    /// Computes: base + sum(index_i * stride_i) using LEA or ADD sequences.
    fn select_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
        ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let base_op = self.get_operand(base, func);
        let dst = self.alloc_vreg(result);

        // FrameIndex (alloca): use LEA to compute the stack-slot address.
        // Everything else (Memory for params, VirtualReg, Register, etc.):
        // use ensure_in_register which loads/moves the *value* — that value
        // is already a pointer for GEP to work with.
        match &base_op {
            MachineOperand::FrameIndex(_) => {
                // LeaMem: dst = &stack_slot (load effective address)
                let mut lea = MachineInstr::new(I686Opcode::LeaMem.as_u32());
                lea.add_operand(dst.clone());
                lea.add_operand(base_op);
                mbb.push_instr(lea);
            }
            _ => {
                // Memory (params, globals), VirtualReg, Register, Symbol, Immediate.
                // ensure_in_register loads the value from Memory, or passes
                // through VirtualReg/Register as-is.
                let base_reg = self.ensure_in_register(base_op, mbb);
                self.emit_mov(dst.clone(), base_reg, mbb);
            }
        };

        // Process each GEP index.
        //
        // LLVM-style GEP semantics:
        //   - For Array types, the stride is the ELEMENT size (not the
        //     whole array size), and current_type advances to the element.
        //   - For Struct types (non-first index), the index is a constant
        //     field selector; we compute the byte offset to that field.
        //   - For Struct types (first index), treat as array-of-structs
        //     with stride = sizeof(struct).
        let mut current_type = ty.clone();
        for (idx_pos, &idx_vid) in indices.iter().enumerate() {
            let idx_op = self.get_operand(idx_vid, func);
            let is_first_index = idx_pos == 0;

            // Determine stride and advance current_type based on the type.
            let stride: u64 = match &current_type {
                IrType::Array { element, .. } => {
                    // Array: stride = element size, advance to element type.
                    let sz = self.type_size_bytes(element) as u64;
                    current_type = (**element).clone();
                    sz
                }
                IrType::Struct { fields, packed: _ } if !is_first_index => {
                    // Subsequent index on struct: constant field selector.
                    // Compute the byte offset to the selected field.
                    if let MachineOperand::Immediate(field_idx) = &idx_op {
                        let fi = *field_idx as usize;
                        if fi < fields.len() {
                            let mut offset: u64 = 0;
                            for field in fields.iter().take(fi) {
                                let f_size = self.type_size_bytes(field) as u64;
                                let f_align = field.alignment(&self.config.target) as u64;
                                if f_align > 0 {
                                    offset = (offset + f_align - 1) & !(f_align - 1);
                                }
                                offset += f_size;
                            }
                            let field_align = fields[fi].alignment(&self.config.target) as u64;
                            if field_align > 0 {
                                offset = (offset + field_align - 1) & !(field_align - 1);
                            }
                            if offset > 0 {
                                let mut add = MachineInstr::new(I686Opcode::Add.as_u32());
                                add.add_operand(dst.clone());
                                add.add_operand(MachineOperand::Immediate(offset as i64));
                                add.add_implicit_def(EFLAGS);
                                mbb.push_instr(add);
                            }
                            current_type = fields[fi].clone();
                            continue; // Skip the generic stride handling below.
                        }
                    }
                    // Fallback: use pointer size.
                    4u64
                }
                IrType::Struct { .. } => {
                    // First index on struct: array-of-structs indexing.
                    // Stride = sizeof(struct), current_type stays struct.
                    self.type_size_bytes(&current_type) as u64
                }
                _ => {
                    // Pointer or other: use type size.
                    self.type_size_bytes(&current_type) as u64
                }
            };

            if stride == 0 {
                continue;
            }

            // Apply index * stride to the running address.
            if let MachineOperand::Immediate(imm) = &idx_op {
                let offset = *imm * (stride as i64);
                if offset != 0 {
                    let mut add = MachineInstr::new(I686Opcode::Add.as_u32());
                    add.add_operand(dst.clone());
                    add.add_operand(MachineOperand::Immediate(offset));
                    add.add_implicit_def(EFLAGS);
                    mbb.push_instr(add);
                }
            } else if stride.is_power_of_two() && stride <= 8 {
                // Use LEA with scaled index: [dst + idx * scale]
                let idx_reg = self.ensure_in_register(idx_op, mbb);
                if let MachineOperand::Register(idx_phys) = idx_reg {
                    let mut lea = MachineInstr::new(I686Opcode::Lea.as_u32());
                    lea.add_operand(dst.clone());
                    if let MachineOperand::Register(dst_phys) = &dst {
                        lea.add_operand(MachineOperand::memory_scaled(
                            *dst_phys,
                            0,
                            idx_phys,
                            stride as u8,
                        ));
                    } else {
                        self.emit_gep_mul_add(dst.clone(), idx_reg, stride as u32, mbb);
                    }
                    mbb.push_instr(lea);
                } else {
                    self.emit_gep_mul_add(dst.clone(), idx_reg, stride as u32, mbb);
                }
            } else {
                let idx_reg = self.ensure_in_register(idx_op, mbb);
                self.emit_gep_mul_add(dst.clone(), idx_reg, stride as u32, mbb);
            }
        }
    }

    /// Helper to emit IMUL idx, stride → ADD dst, product for GEP.
    fn emit_gep_mul_add(
        &mut self,
        dst: MachineOperand,
        idx: MachineOperand,
        stride: u32,
        mbb: &mut MachineBasicBlock,
    ) {
        let tmp = self.alloc_vreg_raw();
        let tmp_op = MachineOperand::VirtualReg(ValueId(tmp));

        // IMUL tmp, idx, stride
        let mut imul = MachineInstr::new(I686Opcode::Imul.as_u32());
        imul.add_operand(tmp_op.clone());
        imul.add_operand(idx);
        imul.add_operand(MachineOperand::Immediate(stride as i64));
        imul.add_implicit_def(EFLAGS);
        mbb.push_instr(imul);

        // ADD dst, tmp
        let mut add = MachineInstr::new(I686Opcode::Add.as_u32());
        add.add_operand(dst);
        add.add_operand(tmp_op);
        add.add_implicit_def(EFLAGS);
        mbb.push_instr(add);
    }

    // -----------------------------------------------------------------------
    // Cast / conversion selection
    // -----------------------------------------------------------------------

    /// BitCast: reinterpret without value change (same-width types).
    ///
    /// On i686, 64-bit values (F64, I64) live in a GPR pair
    /// (lo in `value_map`, hi in `i64_high_map`).  Phi-elimination
    /// generates BitCast (copy) instructions for control-flow merges,
    /// so we must copy *both* halves to avoid data corruption.
    ///
    /// There are three possible source representations for a 64-bit value:
    ///   1. GPR pair (has `i64_high_map` entry) — e.g. from a load or call
    ///   2. Memory/FrameIndex (no high map) — e.g. F64 function parameter
    ///   3. Immediate pair (has `i64_high_map` entry) — e.g. F64 constant
    /// Cases 1 and 3 are handled by copying both halves.  Case 2 requires
    /// splitting the memory read into lo+hi GPR pair.
    fn select_bitcast(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let is_64bit = matches!(to_ty, IrType::F64 | IrType::F80 | IrType::I64);

        if is_64bit {
            // Phi-elimination can generate multiple BitCasts targeting the
            // same result (one per predecessor block).  Both the lo and hi
            // virtual registers MUST be consistent across all of them so
            // the register allocator assigns the same physical register.
            // Reuse existing VReg IDs when we've already seen this result.
            let lo_id = match self.value_map.get(&result.index()) {
                Some(MachineOperand::VirtualReg(vid)) => vid.index() as u32,
                _ => self.alloc_vreg_raw(),
            };
            let hi_id = match self.i64_high_map.get(&result.index()) {
                Some(MachineOperand::VirtualReg(vid)) => vid.index() as u32,
                _ => self.alloc_vreg_raw(),
            };
            let lo_dst = MachineOperand::VirtualReg(ValueId(lo_id));
            let hi_dst = MachineOperand::VirtualReg(ValueId(hi_id));

            // Determine source representation and emit 64-bit copy.
            let hi_src_opt = self.i64_high_map.get(&value.index()).cloned();
            let src = self.get_operand(value, func);

            if let Some(hi_src) = hi_src_opt {
                // Case 1: source is a GPR pair — copy both halves.
                self.emit_mov(lo_dst.clone(), src, mbb);
                self.emit_mov(hi_dst.clone(), hi_src, mbb);
            } else if matches!(
                &src,
                MachineOperand::Memory { .. } | MachineOperand::FrameIndex(_)
            ) {
                // Case 2: source is Memory/FrameIndex (e.g. F64 param at
                // [EBP+8]) — split the 8-byte memory read into two halves.
                let mem = self.safe_to_memory(src, mbb);
                let mut lo_ld = MachineInstr::new(I686Opcode::Mov.as_u32());
                lo_ld.add_operand(lo_dst.clone());
                lo_ld.add_operand(mem.clone());
                mbb.push_instr(lo_ld);

                let hi_mem = self.offset_memory(mem, 4);
                let mut hi_ld = MachineInstr::new(I686Opcode::Mov.as_u32());
                hi_ld.add_operand(hi_dst.clone());
                hi_ld.add_operand(hi_mem);
                mbb.push_instr(hi_ld);
            } else {
                // Fallback: source is a single 32-bit value (shouldn't
                // normally happen for F64 but handle gracefully).
                self.emit_mov(lo_dst.clone(), src, mbb);
            }

            self.value_map.insert(result.index(), lo_dst);
            self.i64_high_map.insert(result.index(), hi_dst);
            return;
        }

        // Default: simple 32-bit copy (covers I32, F32, Ptr, etc.).
        let src = self.get_operand(value, func);
        let dst = self.alloc_vreg(result);
        self.emit_mov(dst, src, mbb);
    }

    // -----------------------------------------------------------------------
    // Floating-point conversion (i686)
    // -----------------------------------------------------------------------

    /// Signed integer → float using the x87 FPU.
    ///
    /// Sequence:  store int to mem → FILD → FSTP (as F32 or F64)
    fn select_si_to_fp(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let src_reg = self.ensure_in_register(src, mbb);

        // Spill integer to a 4-byte stack slot.
        let int_slot = self.frame_size;
        self.frame_size += 4;
        let int_fi = MachineOperand::FrameIndex((-(int_slot as i32 + 4)) as u32);
        let mut st_int = MachineInstr::new(I686Opcode::Mov.as_u32());
        st_int.add_operand(int_fi.clone());
        st_int.add_operand(src_reg);
        mbb.push_instr(st_int);

        // FILD: load integer, convert to x87 float.
        let mut fild = MachineInstr::new(I686Opcode::Fild.as_u32());
        fild.add_operand(int_fi);
        fild.add_implicit_def(ST0);
        mbb.push_instr(fild);

        // FSTP to the appropriate float size.
        let is_f64 = matches!(to_ty, IrType::F64 | IrType::F80);
        let dst_slot = self.frame_size;
        let dst_size: u32 = if is_f64 { 8 } else { 4 };
        self.frame_size += dst_size;
        let dst_fi = MachineOperand::FrameIndex((-(dst_slot as i32 + dst_size as i32)) as u32);
        let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
        fstp.add_operand(dst_fi.clone());
        if is_f64 {
            fstp.add_operand(MachineOperand::Immediate(64));
        }
        fstp.add_implicit_use(ST0);
        mbb.push_instr(fstp);

        // Load result back into GPR(s).
        if is_f64 {
            // Load low 32 bits.
            let lo_dst = self.alloc_vreg(result);
            let mut ld_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
            ld_lo.add_operand(lo_dst.clone());
            ld_lo.add_operand(dst_fi.clone());
            mbb.push_instr(ld_lo);
            // Load high 32 bits.
            let hi_idx = self.alloc_vreg_raw();
            let hi_fi = self.offset_memory(dst_fi, 4);
            let mut ld_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
            ld_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_idx)));
            ld_hi.add_operand(hi_fi);
            mbb.push_instr(ld_hi);
            self.i64_high_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_idx)));
        } else {
            let dst = self.alloc_vreg(result);
            let mut ld = MachineInstr::new(I686Opcode::Mov.as_u32());
            ld.add_operand(dst.clone());
            ld.add_operand(dst_fi);
            mbb.push_instr(ld);
        }
    }

    /// Unsigned integer → float.
    fn select_ui_to_fp(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        // For values that fit in 31 bits, FILD works.  For full u32 range,
        // we'd need to handle the sign bit, but for now delegate to signed.
        self.select_si_to_fp(result, value, to_ty, mbb, func);
    }

    /// Float → signed integer using x87 FISTP.
    fn select_fp_to_si(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let src_ty = func.get_value_type(value).clone();
        let is_f64 = matches!(&src_ty, IrType::F64 | IrType::F80);

        if is_f64 {
            // F64 in two GPR halves.  Spill both to memory then FLD.
            let src_reg = self.ensure_in_register(src, mbb);
            let slot = self.frame_size;
            self.frame_size += 8;
            let fi = MachineOperand::FrameIndex((-(slot as i32 + 8)) as u32);
            let mut st_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
            st_lo.add_operand(fi.clone());
            st_lo.add_operand(src_reg);
            mbb.push_instr(st_lo);
            let hi_op = self
                .i64_high_map
                .get(&value.index())
                .cloned()
                .unwrap_or(MachineOperand::Immediate(0));
            let hi_reg = self.ensure_in_register(hi_op, mbb);
            let hi_fi = self.offset_memory(fi.clone(), 4);
            let mut st_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
            st_hi.add_operand(hi_fi);
            st_hi.add_operand(hi_reg);
            mbb.push_instr(st_hi);
            let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
            fld.add_operand(fi);
            fld.add_operand(MachineOperand::Immediate(64));
            fld.add_implicit_def(ST0);
            mbb.push_instr(fld);
        } else {
            // F32 in single GPR.  Spill to memory then FLD.
            let src_reg = self.ensure_in_register(src, mbb);
            let slot = self.frame_size;
            self.frame_size += 4;
            let fi = MachineOperand::FrameIndex((-(slot as i32 + 4)) as u32);
            let mut st = MachineInstr::new(I686Opcode::Mov.as_u32());
            st.add_operand(fi.clone());
            st.add_operand(src_reg);
            mbb.push_instr(st);
            let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
            fld.add_operand(fi);
            fld.add_implicit_def(ST0);
            mbb.push_instr(fld);
        }

        // FISTP: convert x87 top to integer and store.
        let int_slot = self.frame_size;
        self.frame_size += 4;
        let int_fi = MachineOperand::FrameIndex((-(int_slot as i32 + 4)) as u32);
        let mut fistp = MachineInstr::new(I686Opcode::Fistp.as_u32());
        fistp.add_operand(int_fi.clone());
        fistp.add_implicit_use(ST0);
        mbb.push_instr(fistp);

        // Load the integer result.
        let dst = self.alloc_vreg(result);
        let mut ld = MachineInstr::new(I686Opcode::Mov.as_u32());
        ld.add_operand(dst.clone());
        ld.add_operand(int_fi);
        mbb.push_instr(ld);
    }

    /// Float → unsigned integer.
    fn select_fp_to_ui(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        self.select_fp_to_si(result, value, to_ty, mbb, func);
    }

    /// Float widening (F32 → F64) using x87.
    ///
    /// Spill f32 → FLD mem32 → FSTP mem64 → load two 32-bit halves.
    fn select_fp_ext(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let src_reg = self.ensure_in_register(src, mbb);

        // Spill f32 bits to a 4-byte slot.
        let f32_slot = self.frame_size;
        self.frame_size += 4;
        let f32_fi = MachineOperand::FrameIndex((-(f32_slot as i32 + 4)) as u32);
        let mut st_f32 = MachineInstr::new(I686Opcode::Mov.as_u32());
        st_f32.add_operand(f32_fi.clone());
        st_f32.add_operand(src_reg);
        mbb.push_instr(st_f32);

        // FLD from 4-byte slot (loads single-precision onto x87 stack).
        let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
        fld.add_operand(f32_fi);
        // No size hint = 32-bit.
        fld.add_implicit_def(ST0);
        mbb.push_instr(fld);

        // FSTP to 8-byte slot (stores as double-precision).
        let f64_slot = self.frame_size;
        self.frame_size += 8;
        let f64_fi = MachineOperand::FrameIndex((-(f64_slot as i32 + 8)) as u32);
        let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
        fstp.add_operand(f64_fi.clone());
        fstp.add_operand(MachineOperand::Immediate(64)); // 64-bit hint
        fstp.add_implicit_use(ST0);
        mbb.push_instr(fstp);

        // Load both 32-bit halves from the f64 slot into GPRs.
        let lo_dst = self.alloc_vreg(result);
        let mut ld_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
        ld_lo.add_operand(lo_dst.clone());
        ld_lo.add_operand(f64_fi.clone());
        mbb.push_instr(ld_lo);

        let hi_idx = self.alloc_vreg_raw();
        let hi_fi = self.offset_memory(f64_fi, 4);
        let mut ld_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
        ld_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_idx)));
        ld_hi.add_operand(hi_fi);
        mbb.push_instr(ld_hi);

        // Track the high word so the call path pushes both halves for F64.
        self.i64_high_map
            .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_idx)));
    }

    /// Float narrowing (F64 → F32) using x87.
    ///
    /// Store f64 halves → FLD mem64 → FSTP mem32 → load 32-bit result.
    fn select_fp_trunc(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let src_reg = self.ensure_in_register(src, mbb);

        // Store F64 low half.
        let f64_slot = self.frame_size;
        self.frame_size += 8;
        let f64_fi = MachineOperand::FrameIndex((-(f64_slot as i32 + 8)) as u32);
        let mut st_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
        st_lo.add_operand(f64_fi.clone());
        st_lo.add_operand(src_reg);
        mbb.push_instr(st_lo);

        // Store F64 high half.
        let hi_op = self
            .i64_high_map
            .get(&value.index())
            .cloned()
            .unwrap_or(MachineOperand::Immediate(0));
        let hi_reg = self.ensure_in_register(hi_op, mbb);
        let hi_fi = self.offset_memory(f64_fi.clone(), 4);
        let mut st_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
        st_hi.add_operand(hi_fi);
        st_hi.add_operand(hi_reg);
        mbb.push_instr(st_hi);

        // FLD from 8-byte slot (loads double onto x87 stack).
        let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
        fld.add_operand(f64_fi);
        fld.add_operand(MachineOperand::Immediate(64)); // 64-bit
        fld.add_implicit_def(ST0);
        mbb.push_instr(fld);

        // FSTP to 4-byte slot (stores as single precision).
        let f32_slot = self.frame_size;
        self.frame_size += 4;
        let f32_fi = MachineOperand::FrameIndex((-(f32_slot as i32 + 4)) as u32);
        let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
        fstp.add_operand(f32_fi.clone());
        // No size hint = 32-bit.
        fstp.add_implicit_use(ST0);
        mbb.push_instr(fstp);

        // Load 32-bit f32 result back into a GPR.
        let dst = self.alloc_vreg(result);
        let mut ld = MachineInstr::new(I686Opcode::Mov.as_u32());
        ld.add_operand(dst.clone());
        ld.add_operand(f32_fi);
        mbb.push_instr(ld);
    }

    /// Truncation: extract lower bits by using a smaller register / MOV.
    fn select_trunc(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let dst = self.alloc_vreg(result);
        // On i686, truncation to 32-bit or smaller is just a MOV;
        // the upper bits are naturally ignored.
        self.emit_mov(dst, src, mbb);
    }

    /// Zero extension: MOVZX for 8→32 and 16→32.
    fn select_zext(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let dst = self.alloc_vreg(result);

        let src_type = self.get_value_type(value, func);
        let src_bits = self.type_size_bits(&src_type);

        if src_bits < 32 {
            // MOVZX for 8→32 or 16→32.
            let src_reg = self.ensure_in_register(src, mbb);
            let mut movzx = MachineInstr::new(I686Opcode::MovZx.as_u32());
            movzx.add_operand(dst);
            movzx.add_operand(src_reg);
            mbb.push_instr(movzx);
        } else {
            // 32→64: zero-extend into a register pair (hi = 0).
            let src_reg = self.ensure_in_register(src, mbb);
            self.emit_mov(dst, src_reg, mbb);
            // High word is zero for zero-extension.
            self.i64_high_map
                .insert(result.index(), MachineOperand::Immediate(0));
        }
    }

    /// Sign extension: MOVSX for 8→32 and 16→32.
    fn select_sext(
        &mut self,
        result: ValueId,
        value: ValueId,
        to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let dst = self.alloc_vreg(result);

        let src_type = self.get_value_type(value, func);
        let src_bits = self.type_size_bits(&src_type);

        if src_bits < 32 {
            // MOVSX for 8→32 or 16→32.
            let src_reg = self.ensure_in_register(src, mbb);
            let mut movsx = MachineInstr::new(I686Opcode::MovSx.as_u32());
            movsx.add_operand(dst);
            movsx.add_operand(src_reg);
            mbb.push_instr(movsx);
        } else if src_bits == 32 && self.type_size_bits(to_ty) == 64 {
            // 32→64 sign extension: CDQ (sign-extend EAX into EDX:EAX).
            let src_reg = self.ensure_in_register(src, mbb);
            let mut mov_eax = MachineInstr::new(I686Opcode::Mov.as_u32());
            mov_eax.add_operand(MachineOperand::Register(EAX));
            mov_eax.add_operand(src_reg);
            mbb.push_instr(mov_eax);

            let mut cdq = MachineInstr::new(I686Opcode::Cdq.as_u32());
            cdq.add_implicit_use(EAX);
            cdq.add_implicit_def(EDX);
            mbb.push_instr(cdq);

            // Result lo=EAX, hi=EDX — save both halves to vregs.
            // Mark EDX as implicitly used in the first MOV to prevent
            // the register allocator from clobbering it.
            let lo_dst = self.alloc_vreg_raw();
            let mut mov_lo = MachineInstr::new(I686Opcode::Mov.as_u32());
            mov_lo.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
            mov_lo.add_operand(MachineOperand::Register(EAX));
            mov_lo.add_implicit_use(EDX);
            mbb.push_instr(mov_lo);

            let hi_dst = self.alloc_vreg_raw();
            let mut mov_hi = MachineInstr::new(I686Opcode::Mov.as_u32());
            mov_hi.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
            mov_hi.add_operand(MachineOperand::Register(EDX));
            mbb.push_instr(mov_hi);

            self.value_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
            self.i64_high_map
                .insert(result.index(), MachineOperand::VirtualReg(ValueId(hi_dst)));
        } else {
            self.emit_mov(dst, src, mbb);
        }
    }

    // -----------------------------------------------------------------------
    // Prologue / Epilogue generation
    // -----------------------------------------------------------------------

    /// Emits the standard cdecl function prologue.
    ///
    /// ```asm
    /// push ebp
    /// mov  ebp, esp
    /// sub  esp, <frame_size>
    /// push ebx        ; callee-saved (if used)
    /// push esi        ; callee-saved (if used)
    /// push edi        ; callee-saved (if used)
    /// ```
    #[allow(dead_code)]
    fn emit_prologue(&self, mbb: &mut MachineBasicBlock) {
        // PUSH EBP
        let mut push_ebp = MachineInstr::new(I686Opcode::Push.as_u32());
        push_ebp.add_operand(MachineOperand::Register(EBP));
        push_ebp.add_implicit_use(ESP);
        push_ebp.add_implicit_def(ESP);
        mbb.push_instr(push_ebp);

        // MOV EBP, ESP
        let mut mov_ebp = MachineInstr::new(I686Opcode::Mov.as_u32());
        mov_ebp.add_operand(MachineOperand::Register(EBP));
        mov_ebp.add_operand(MachineOperand::Register(ESP));
        mbb.push_instr(mov_ebp);

        // SUB ESP, frame_size (reserve space for locals)
        if self.frame_size > 0 {
            let mut sub_esp = MachineInstr::new(I686Opcode::Sub.as_u32());
            sub_esp.add_operand(MachineOperand::Register(ESP));
            sub_esp.add_operand(MachineOperand::Immediate(self.frame_size as i64));
            sub_esp.add_implicit_def(EFLAGS);
            mbb.push_instr(sub_esp);
        }

        // PUSH callee-saved registers.
        for &reg in &[EBX, ESI, EDI] {
            let mut push = MachineInstr::new(I686Opcode::Push.as_u32());
            push.add_operand(MachineOperand::Register(reg));
            push.add_implicit_use(ESP);
            push.add_implicit_def(ESP);
            mbb.push_instr(push);
        }
    }

    /// Emits the standard cdecl function epilogue.
    ///
    /// ```asm
    /// pop  edi        ; restore callee-saved
    /// pop  esi
    /// pop  ebx
    /// mov  esp, ebp
    /// pop  ebp
    /// ```
    #[allow(dead_code)]
    fn emit_epilogue(&self, mbb: &mut MachineBasicBlock) {
        // POP callee-saved registers (reverse order of prologue).
        for &reg in &[EDI, ESI, EBX] {
            let mut pop = MachineInstr::new(I686Opcode::Pop.as_u32());
            pop.add_operand(MachineOperand::Register(reg));
            pop.add_implicit_use(ESP);
            pop.add_implicit_def(ESP);
            mbb.push_instr(pop);
        }

        // MOV ESP, EBP (restore stack pointer).
        let mut mov_esp = MachineInstr::new(I686Opcode::Mov.as_u32());
        mov_esp.add_operand(MachineOperand::Register(ESP));
        mov_esp.add_operand(MachineOperand::Register(EBP));
        mbb.push_instr(mov_esp);

        // POP EBP.
        let mut pop_ebp = MachineInstr::new(I686Opcode::Pop.as_u32());
        pop_ebp.add_operand(MachineOperand::Register(EBP));
        pop_ebp.add_implicit_use(ESP);
        pop_ebp.add_implicit_def(ESP);
        mbb.push_instr(pop_ebp);
    }

    // -----------------------------------------------------------------------
    // Helper: emit MOV dst, src
    // -----------------------------------------------------------------------

    /// Emits a MOV instruction from `src` to `dst`.
    // ===================================================================
    // Inline Assembly Lowering
    // ===================================================================

    /// Lowers an IR `InlineAsm` instruction into i686 machine instructions.
    ///
    /// Supports two paths:
    /// 1. **Builtin asm** — templates using `$0`, `$1` (from `__builtin_*`
    ///    lowering in `expr_lowering.rs`), with comma-separated constraints.
    /// 2. **GCC user asm** — templates using `%0`, `%1`, `%[name]` with
    ///    colon-separated constraints.
    #[allow(clippy::too_many_arguments)]
    fn lower_inline_asm(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        _clobbers: &[String],
        _goto_targets: &[crate::ir::basic_block::BasicBlockId],
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        // Detect GCC user asm: templates containing %N or %[name].
        let is_gcc_user_asm = {
            let chars: Vec<char> = template.chars().collect();
            let mut found = false;
            for i in 0..chars.len() {
                if chars[i] == '%' && i + 1 < chars.len() {
                    let next = chars[i + 1];
                    if next.is_ascii_digit() || next == '[' {
                        found = true;
                        break;
                    }
                }
            }
            found
        };

        if is_gcc_user_asm {
            self.lower_gcc_user_asm_i686(result, template, constraints, operands, func)
        } else {
            self.lower_builtin_asm_i686(result, template, constraints, operands, func)
        }
    }

    /// Lowers compiler-generated builtin inline assembly (from `__builtin_clz`,
    /// `__builtin_ctz`, `__builtin_popcount`, etc.).
    ///
    /// These use `$0`/`$1` operand references with comma-separated constraints
    /// like `"=r,r"`.
    fn lower_builtin_asm_i686(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        _constraints: &str,
        operands: &[ValueId],
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Allocate output vreg.
        let out_vreg = if let Some(res_vid) = result {
            self.alloc_vreg(res_vid)
        } else {
            MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()))
        };

        // Load input operand into a vreg.
        let input_ops: Vec<MachineOperand> = operands
            .iter()
            .map(|v| self.get_operand(*v, func))
            .collect();

        let input0 = if let Some(op) = input_ops.first() {
            match op {
                MachineOperand::Immediate(_)
                | MachineOperand::FrameIndex(_)
                | MachineOperand::Memory { .. } => {
                    let vreg = MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()));
                    let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op.clone());
                    instrs.push(mi);
                    vreg
                }
                _ => op.clone(),
            }
        } else {
            MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()))
        };

        // Split template into individual instructions.
        let asm_lines: Vec<&str> = template
            .trim()
            .split('\n')
            .flat_map(|l| l.split('\t'))
            .flat_map(|l| l.split(';'))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for line in &asm_lines {
            let parts: Vec<&str> = line
                .splitn(2, |c: char| c.is_whitespace())
                .map(|s| s.trim())
                .collect();
            let mnemonic = parts[0].to_lowercase();

            match mnemonic.as_str() {
                "bsrl" => {
                    let mut mi = MachineInstr::new(I686Opcode::Bsr.as_u32());
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "bsfl" => {
                    let mut mi = MachineInstr::new(I686Opcode::Bsf.as_u32());
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "popcntl" => {
                    let mut mi = MachineInstr::new(I686Opcode::Popcnt.as_u32());
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "bswap" | "bswapl" => {
                    // Move input to output, then bswap in-place.
                    let mut mi_mov = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi_mov.add_operand(out_vreg.clone());
                    mi_mov.add_operand(input0.clone());
                    instrs.push(mi_mov);

                    let mut mi = MachineInstr::new(I686Opcode::Bswap.as_u32());
                    mi.add_operand(out_vreg.clone());
                    instrs.push(mi);
                }
                "xorl" => {
                    // Parse immediate: e.g., "xorl $$31, $0"
                    if parts.len() > 1 {
                        let xor_parts: Vec<&str> = parts[1].split(',').map(|s| s.trim()).collect();
                        if let Some(imm_str) = xor_parts.first() {
                            let imm_str = imm_str.trim_start_matches('$');
                            if let Ok(imm_val) = imm_str.parse::<i64>() {
                                let mut mi = MachineInstr::new(I686Opcode::Xor.as_u32());
                                mi.add_operand(out_vreg.clone());
                                mi.add_operand(MachineOperand::Immediate(imm_val));
                                instrs.push(mi);
                            }
                        }
                    }
                }
                "movl" => {
                    // MOV between $-referenced operands.
                    let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi.add_operand(out_vreg.clone());
                    mi.add_operand(input0.clone());
                    instrs.push(mi);
                }
                "leal" | "lea" => {
                    // LEA dst, [mem] — used by va_start to compute pointer.
                    // Template: "leal disp(%ebp), $0"
                    // Parse the memory operand (first in AT&T), output is $0.
                    if parts.len() > 1 {
                        let operands_str = parts[1];
                        // Split on comma: first is memory src, second is $0 (output)
                        let comma_parts: Vec<&str> =
                            operands_str.split(',').map(|s| s.trim()).collect();
                        let mem_str = comma_parts[0].trim();
                        // Parse "disp(%reg)" memory operand
                        if mem_str.contains('(') && mem_str.contains(')') {
                            if let Some(paren_pos) = mem_str.find('(') {
                                let offset_str = &mem_str[..paren_pos];
                                let reg_str = mem_str[paren_pos + 1..mem_str.len() - 1]
                                    .trim_start_matches('%');
                                let offset = offset_str.parse::<i32>().unwrap_or(0);
                                let base = parse_i686_reg_name(reg_str).unwrap_or(EBP);
                                let mut mi = MachineInstr::new(I686Opcode::LeaMem.as_u32());
                                mi.add_operand(out_vreg.clone());
                                mi.add_operand(MachineOperand::Memory {
                                    base,
                                    offset,
                                    index: None,
                                    scale: 1,
                                });
                                instrs.push(mi);
                            }
                        }
                    }
                }
                _ => {
                    // Unknown mnemonic — emit NOP as fallback.
                    instrs.push(MachineInstr::new(I686Opcode::Nop.as_u32()));
                }
            }
        }

        instrs
    }

    /// Lowers GCC-style user inline assembly with `%N` operand references.
    ///
    /// This handles `asm("..." : "=r"(out) : "r"(in) : "memory")` patterns
    /// from user C code, as used in `test_inline_asm` and kernel code.
    fn lower_gcc_user_asm_i686(
        &mut self,
        result: Option<ValueId>,
        template: &str,
        constraints: &str,
        operands: &[ValueId],
        func: &IrFunction,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::new();

        // Parse constraint string: "outputs:inputs" or "outputs:inputs:gotos"
        let sections: Vec<&str> = constraints.split(':').collect();
        let output_str = sections.first().copied().unwrap_or("");
        let input_str = sections.get(1).copied().unwrap_or("");

        let output_constraints: Vec<&str> = if output_str.is_empty() {
            Vec::new()
        } else {
            output_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect()
        };
        let input_constraints: Vec<&str> = if input_str.is_empty() {
            Vec::new()
        } else {
            input_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect()
        };

        let n_outputs = output_constraints.len();
        let n_rw = output_constraints
            .iter()
            .filter(|c| c.starts_with('+'))
            .count();

        // Build output operands.
        let mut output_ops: Vec<MachineOperand> = Vec::new();
        for i in 0..n_outputs {
            let stripped = output_constraints[i]
                .trim_start_matches('=')
                .trim_start_matches('+')
                .trim_start_matches('&');
            if stripped.contains('m') {
                // Memory output
                if i < operands.len() {
                    output_ops.push(self.get_operand(operands[i], func));
                } else {
                    output_ops.push(MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw())));
                }
            } else {
                // Register output
                let vreg = if i == 0 {
                    if let Some(res_vid) = result {
                        self.alloc_vreg(res_vid)
                    } else {
                        MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()))
                    }
                } else {
                    MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()))
                };
                output_ops.push(vreg);
            }
        }

        if output_ops.is_empty() {
            output_ops.push(if let Some(res_vid) = result {
                self.alloc_vreg(res_vid)
            } else {
                MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()))
            });
        }

        // Build input operands.
        let input_start_idx = n_outputs + n_rw;
        let mut input_ops: Vec<MachineOperand> = Vec::new();
        for i in 0..input_constraints.len() {
            let trimmed = input_constraints[i].trim();
            let op_idx = input_start_idx + i;

            // Matching constraint (e.g., "0")
            if let Ok(match_idx) = trimmed.parse::<usize>() {
                if match_idx < output_ops.len() {
                    let out = output_ops[match_idx].clone();
                    if op_idx < operands.len() {
                        let inp_op = self.get_operand(operands[op_idx], func);
                        let vreg = MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()));
                        let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                        mi.add_operand(vreg.clone());
                        mi.add_operand(inp_op);
                        instrs.push(mi);
                        let mut mi2 = MachineInstr::new(I686Opcode::Mov.as_u32());
                        mi2.add_operand(out.clone());
                        mi2.add_operand(vreg);
                        instrs.push(mi2);
                    }
                    input_ops.push(out);
                    continue;
                }
            }

            if op_idx < operands.len() {
                let op = self.get_operand(operands[op_idx], func);
                if trimmed.contains('m') {
                    input_ops.push(op);
                } else {
                    let vreg = MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()));
                    let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi.add_operand(vreg.clone());
                    mi.add_operand(op);
                    instrs.push(mi);
                    input_ops.push(vreg);
                }
            } else {
                input_ops.push(MachineOperand::Immediate(0));
            }
        }

        // Pre-load read-write outputs ("+r" constraints): load current value
        // into the output vreg before the asm block.
        for i in 0..n_outputs {
            if output_constraints[i].starts_with('+') {
                let rw_load_idx = n_outputs + i;
                if rw_load_idx < operands.len() {
                    let val_op = self.get_operand(operands[rw_load_idx], func);
                    let vreg = MachineOperand::VirtualReg(ValueId(self.alloc_vreg_raw()));
                    let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi.add_operand(vreg.clone());
                    mi.add_operand(val_op);
                    instrs.push(mi);
                    let mut mi2 = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi2.add_operand(output_ops[i].clone());
                    mi2.add_operand(vreg);
                    instrs.push(mi2);
                }
            }
        }

        // Build operand map: [outputs..., inputs...]
        let mut operand_map: Vec<MachineOperand> = Vec::new();
        operand_map.extend(output_ops.iter().cloned());
        operand_map.extend(input_ops.iter().cloned());

        // Parse and emit each assembly instruction from the template.
        let asm_lines: Vec<&str> = template
            .trim()
            .split('\n')
            .flat_map(|l| l.split('\t'))
            .flat_map(|l| l.split(';'))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for line in &asm_lines {
            let expanded = self.expand_gcc_operands(line, &operand_map);
            let parts: Vec<&str> = expanded
                .splitn(2, |c: char| c.is_whitespace())
                .map(|s| s.trim())
                .collect();
            let mnemonic = parts[0].to_lowercase();

            match mnemonic.as_str() {
                "movl" | "mov" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "addl" | "add" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Add.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "subl" | "sub" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Sub.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "xorl" | "xor" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Xor.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "andl" | "and" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::And.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "orl" | "or" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Or.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "bsrl" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Bsr.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "bsfl" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Bsf.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "popcntl" => {
                    if parts.len() > 1 {
                        let (src_op, dst_op) = self.parse_two_att_operands(parts[1], &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Popcnt.as_u32());
                        mi.add_operand(dst_op);
                        mi.add_operand(src_op);
                        instrs.push(mi);
                    }
                }
                "bswap" | "bswapl" => {
                    if parts.len() > 1 {
                        let op = self.parse_single_att_operand(parts[1].trim(), &operand_map);
                        let mut mi = MachineInstr::new(I686Opcode::Bswap.as_u32());
                        mi.add_operand(op);
                        instrs.push(mi);
                    }
                }
                _ => {
                    // Unknown — emit NOP.
                    instrs.push(MachineInstr::new(I686Opcode::Nop.as_u32()));
                }
            }
        }

        // Store outputs to memory for "=m" constraints.
        for i in 0..n_outputs {
            let stripped = output_constraints[i]
                .trim_start_matches('=')
                .trim_start_matches('+')
                .trim_start_matches('&');
            if !stripped.contains('m') && i < operands.len() {
                // For register outputs, store back to the target if operand is
                // a memory location.
                let target = self.get_operand(operands[i], func);
                if matches!(
                    target,
                    MachineOperand::FrameIndex(_) | MachineOperand::Memory { .. }
                ) {
                    let mut mi = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mi.add_operand(target);
                    mi.add_operand(output_ops[i].clone());
                    instrs.push(mi);
                }
            }
        }

        instrs
    }

    /// Expands `%N` / `%[name]` operand references in an asm template to
    /// register names or operand placeholders.
    fn expand_gcc_operands(&self, line: &str, operand_map: &[MachineOperand]) -> String {
        let mut result = String::new();
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '%' && i + 1 < chars.len() {
                let next = chars[i + 1];
                if next == '%' {
                    // Escaped %%
                    result.push('%');
                    i += 2;
                    continue;
                }
                if next.is_ascii_digit() {
                    // %N reference
                    let mut num_str = String::new();
                    let mut j = i + 1;
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        num_str.push(chars[j]);
                        j += 1;
                    }
                    if let Ok(idx) = num_str.parse::<usize>() {
                        if idx < operand_map.len() {
                            result.push_str(&self.operand_to_asm_str(&operand_map[idx]));
                        }
                    }
                    i = j;
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }

    /// Converts a `MachineOperand` to its AT&T-syntax assembly string.
    fn operand_to_asm_str(&self, op: &MachineOperand) -> String {
        match op {
            MachineOperand::Register(r) => {
                format!("%{}", reg_name_32(*r))
            }
            MachineOperand::VirtualReg(v) => {
                format!("%vreg{}", v)
            }
            MachineOperand::Immediate(v) => {
                format!("${}", v)
            }
            MachineOperand::FrameIndex(off) => {
                format!("{}(%ebp)", off)
            }
            MachineOperand::Memory { base, offset, .. } => {
                format!("{}(%{})", offset, reg_name_32(*base))
            }
            _ => "%eax".to_string(),
        }
    }

    /// Parses AT&T-syntax "src, dst" two-operand string into machine operands.
    fn parse_two_att_operands(
        &self,
        operand_str: &str,
        operand_map: &[MachineOperand],
    ) -> (MachineOperand, MachineOperand) {
        let parts: Vec<&str> = operand_str.split(',').map(|s| s.trim()).collect();
        let src = if parts.len() > 0 {
            self.parse_single_att_operand(parts[0], operand_map)
        } else {
            MachineOperand::Immediate(0)
        };
        let dst = if parts.len() > 1 {
            self.parse_single_att_operand(parts[1], operand_map)
        } else {
            MachineOperand::Register(EAX)
        };
        (src, dst)
    }

    /// Parses a single AT&T-syntax operand string.
    fn parse_single_att_operand(&self, s: &str, _operand_map: &[MachineOperand]) -> MachineOperand {
        let s = s.trim();
        // %vreg reference from expanded template
        if s.starts_with("%vreg") {
            if let Ok(v) = s[5..].parse::<u32>() {
                return MachineOperand::VirtualReg(ValueId(v));
            }
        }
        // Physical register reference: %eax, %ecx, etc.
        if s.starts_with('%') {
            let rname = &s[1..];
            if let Some(r) = parse_i686_reg_name(rname) {
                return MachineOperand::Register(r);
            }
        }
        // Immediate: $N
        if s.starts_with('$') {
            if let Ok(v) = s[1..].parse::<i64>() {
                return MachineOperand::Immediate(v);
            }
        }
        // Memory: offset(%reg)
        if s.contains('(') && s.contains(')') {
            if let Some(paren_pos) = s.find('(') {
                let offset_str = &s[..paren_pos];
                let reg_str = &s[paren_pos + 1..s.len() - 1].trim_start_matches('%');
                let offset = offset_str.parse::<i32>().unwrap_or(0);
                let base = parse_i686_reg_name(reg_str).unwrap_or(EBP);
                return MachineOperand::Memory {
                    base,
                    offset,
                    index: None,
                    scale: 1,
                };
            }
        }
        // Fallback: operand map index
        MachineOperand::Register(EAX)
    }

    fn emit_mov(&self, dst: MachineOperand, src: MachineOperand, mbb: &mut MachineBasicBlock) {
        let mut mov = MachineInstr::new(I686Opcode::Mov.as_u32());
        mov.add_operand(dst);
        mov.add_operand(src);
        mbb.push_instr(mov);
    }

    // -----------------------------------------------------------------------
    // Helper: emit FLD (load to x87 stack)
    // -----------------------------------------------------------------------

    /// Emits an FLD to push a value onto the x87 FPU stack.
    ///
    /// If the operand is already in ST(0), this is a no-op. If it's a memory
    /// operand, emits FLD mem. If it's in a GPR, spills to memory first.
    fn emit_fld(&self, op: MachineOperand, mbb: &mut MachineBasicBlock) {
        match &op {
            MachineOperand::Register(reg) if registers::is_fpu(*reg) => {
                // Already on x87 stack — emit FLD ST(i) if not ST(0).
                if *reg != ST0 {
                    let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
                    fld.add_operand(op);
                    fld.add_implicit_def(ST0);
                    mbb.push_instr(fld);
                }
                // If it's ST(0), value is already at the top.
            }
            MachineOperand::Memory { .. } => {
                let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
                fld.add_operand(op);
                fld.add_implicit_def(ST0);
                mbb.push_instr(fld);
            }
            _ => {
                // Value is in a GPR or is an immediate/symbol — need to spill
                // to memory first, then FLD. Use a stack scratch slot.
                // Push to stack, then FLD from [ESP].
                let mut push = MachineInstr::new(I686Opcode::Push.as_u32());
                push.add_operand(op);
                push.add_implicit_use(ESP);
                push.add_implicit_def(ESP);
                mbb.push_instr(push);

                let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
                fld.add_operand(MachineOperand::Memory {
                    base: ESP,
                    offset: 0,
                    index: None,
                    scale: 1,
                });
                fld.add_implicit_def(ST0);
                mbb.push_instr(fld);

                // Restore ESP.
                let mut add_esp = MachineInstr::new(I686Opcode::Add.as_u32());
                add_esp.add_operand(MachineOperand::Register(ESP));
                add_esp.add_operand(MachineOperand::Immediate(4));
                add_esp.add_implicit_def(EFLAGS);
                mbb.push_instr(add_esp);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper: emit FLD for a ValueId with type awareness (F32 vs F64)
    // -----------------------------------------------------------------------

    /// Loads an IR value onto the x87 FPU stack, handling both F32 (single
    /// PUSH + FLD DWORD) and F64 (double PUSH hi/lo + FLD QWORD) correctly.
    ///
    /// On i686, F64 values are stored as GPR pairs (lo in value_map, hi in
    /// i64_high_map).  A single PUSH only moves 4 bytes, which would load
    /// half the double as a 32-bit float — producing garbage.  This function
    /// detects F64 values and pushes both halves before using FLD QWORD.
    fn emit_fld_value(
        &self,
        vid: ValueId,
        is_f64: bool,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let op = self.get_operand(vid, func);

        if is_f64 {
            // Check if there's a high-half in i64_high_map (GPR pair storage).
            if let Some(hi_op) = self.i64_high_map.get(&vid.index()).cloned() {
                // F64 in GPR pair: PUSH hi, PUSH lo, FLD QWORD [ESP], ADD ESP,8
                let mut push_hi = MachineInstr::new(I686Opcode::Push.as_u32());
                push_hi.add_operand(hi_op);
                push_hi.add_implicit_use(ESP);
                push_hi.add_implicit_def(ESP);
                mbb.push_instr(push_hi);

                let mut push_lo = MachineInstr::new(I686Opcode::Push.as_u32());
                push_lo.add_operand(op);
                push_lo.add_implicit_use(ESP);
                push_lo.add_implicit_def(ESP);
                mbb.push_instr(push_lo);

                let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
                fld.add_operand(MachineOperand::Memory {
                    base: ESP,
                    offset: 0,
                    index: None,
                    scale: 1,
                });
                fld.add_operand(MachineOperand::Immediate(64)); // 64-bit hint
                fld.add_implicit_def(ST0);
                mbb.push_instr(fld);

                let mut add_esp = MachineInstr::new(I686Opcode::Add.as_u32());
                add_esp.add_operand(MachineOperand::Register(ESP));
                add_esp.add_operand(MachineOperand::Immediate(8));
                add_esp.add_implicit_def(EFLAGS);
                mbb.push_instr(add_esp);
            } else {
                // F64 from memory (global variable etc.) — FLD QWORD [mem]
                match &op {
                    MachineOperand::Memory { .. } | MachineOperand::FrameIndex(_) => {
                        let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
                        fld.add_operand(op);
                        fld.add_operand(MachineOperand::Immediate(64));
                        fld.add_implicit_def(ST0);
                        mbb.push_instr(fld);
                    }
                    _ => {
                        // Single GPR with no high half — treat as F32 fallback
                        self.emit_fld(op, mbb);
                    }
                }
            }
        } else {
            // F32 — use existing emit_fld for single 32-bit values
            self.emit_fld(op, mbb);
        }
    }

    // -----------------------------------------------------------------------
    // Helper: get operand for a ValueId
    // -----------------------------------------------------------------------

    /// Resolves an IR ValueId to a MachineOperand.
    ///
    /// Looks up the value in the value_map first. If not found, creates a
    /// VirtualReg operand as a placeholder for the register allocator.
    fn get_operand(&self, vid: ValueId, _func: &IrFunction) -> MachineOperand {
        if let Some(op) = self.value_map.get(&vid.index()) {
            op.clone()
        } else {
            MachineOperand::VirtualReg(vid)
        }
    }

    // -----------------------------------------------------------------------
    // Helper: allocate a virtual register for a result
    // -----------------------------------------------------------------------

    /// Allocates a virtual register for an IR result value and records the
    /// mapping in the value_map.
    fn alloc_vreg(&mut self, vid: ValueId) -> MachineOperand {
        let op = MachineOperand::VirtualReg(vid);
        self.value_map.insert(vid.index(), op.clone());
        op
    }

    /// Allocates a fresh anonymous virtual register (not tied to an IR value).
    fn alloc_vreg_raw(&mut self) -> u32 {
        let id = self.next_vreg;
        self.next_vreg += 1;
        id
    }

    /// Allocate a temporary stack slot of the given `size` bytes (4-byte
    /// aligned).  Returns the *negative* EBP offset as a `u32` suitable
    /// for use in `MachineOperand::FrameIndex`.
    fn alloc_stack_slot(&mut self, size: u32) -> u32 {
        self.frame_size = align_up(self.frame_size, 4);
        self.frame_size += size;
        let offset = -(self.frame_size as i32);
        offset as u32
    }

    // -----------------------------------------------------------------------
    // Helper: ensure a value is in a register
    // -----------------------------------------------------------------------

    /// Ensures the operand is in a register. If it's already a register,
    /// returns it as-is. If it's a memory operand, immediate, or symbol,
    /// emits a MOV to a fresh virtual register and returns that.
    ///
    /// **FrameIndex special handling:** A `FrameIndex` represents a stack
    /// slot (alloca).  Using it as a *value* means we want the slot's
    /// ADDRESS (pointer), not its contents.  We therefore emit LEA
    /// instead of MOV (which would dereference the slot).  Load/Store
    /// instructions access the slot's contents through `safe_to_memory`,
    /// which keeps FrameIndex intact for the encoder.
    fn ensure_in_register(
        &mut self,
        op: MachineOperand,
        mbb: &mut MachineBasicBlock,
    ) -> MachineOperand {
        match &op {
            MachineOperand::Register(_) | MachineOperand::VirtualReg(_) => op,
            MachineOperand::FrameIndex(_) => {
                // FrameIndex as a value → compute address with LEA.
                let tmp = self.alloc_vreg_raw();
                let tmp_op = MachineOperand::VirtualReg(ValueId(tmp));
                let mut lea = MachineInstr::new(I686Opcode::LeaMem.as_u32());
                lea.add_operand(tmp_op.clone());
                lea.add_operand(op);
                mbb.push_instr(lea);
                tmp_op
            }
            _ => {
                let tmp = self.alloc_vreg_raw();
                let tmp_op = MachineOperand::VirtualReg(ValueId(tmp));
                self.emit_mov(tmp_op.clone(), op, mbb);
                tmp_op
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper: convert operand to memory addressing mode
    // -----------------------------------------------------------------------

    /// Converts a non-VirtualReg operand to a form suitable for memory access.
    ///
    /// This helper is only called when the operand is known NOT to be a
    /// VirtualReg (those are handled via MovLoad/MovStore). For FrameIndex
    /// operands, they are kept as-is (the encoder handles [EBP+offset]).
    /// For physical registers, they are wrapped in a Memory operand.
    /// For Symbol operands, they are returned as-is (encoder resolves them).
    fn safe_to_memory(
        &mut self,
        op: MachineOperand,
        mbb: &mut MachineBasicBlock,
    ) -> MachineOperand {
        match &op {
            MachineOperand::Memory { .. } => op,
            MachineOperand::FrameIndex(_) => {
                // FrameIndex is directly supported by the encoder as [EBP+n].
                op
            }
            MachineOperand::Register(reg) => MachineOperand::Memory {
                base: *reg,
                offset: 0,
                index: None,
                scale: 1,
            },
            MachineOperand::Symbol(_) => {
                // Symbol is handled directly by the encoder for global access.
                op
            }
            MachineOperand::VirtualReg(_) => {
                // This should not happen — callers should use MovLoad/MovStore.
                // Safety fallback: move to a register and use that.
                let tmp = self.alloc_vreg_raw();
                let tmp_op = MachineOperand::VirtualReg(ValueId(tmp));
                self.emit_mov(tmp_op.clone(), op, mbb);
                // Return the vreg itself — the caller should use MovLoad/MovStore.
                // This path indicates a bug in the caller.
                tmp_op
            }
            _ => {
                // Label, Immediate — move to register first, then use as base.
                let tmp = self.alloc_vreg_raw();
                let tmp_op = MachineOperand::VirtualReg(ValueId(tmp));
                self.emit_mov(tmp_op.clone(), op, mbb);
                tmp_op
            }
        }
    }

    /// Adjusts a Memory operand's offset by the given delta.
    fn offset_memory(&self, mem: MachineOperand, delta: i32) -> MachineOperand {
        match mem {
            MachineOperand::Memory {
                base,
                offset,
                index,
                scale,
            } => MachineOperand::Memory {
                base,
                offset: offset + delta,
                index,
                scale,
            },
            MachineOperand::FrameIndex(idx) => {
                // FrameIndex stores a byte offset from EBP as u32 (really i32).
                // Add the delta to produce the adjusted offset.
                let current_offset = idx as i32;
                MachineOperand::FrameIndex((current_offset + delta) as u32)
            }
            other => other,
        }
    }

    // -----------------------------------------------------------------------
    // Type size / alignment helpers
    // -----------------------------------------------------------------------

    /// Returns the size of an IR type in bytes for the i686 target.
    fn type_size_bytes(&self, ty: &IrType) -> u32 {
        match ty {
            IrType::Void => 0,
            IrType::I1 => 1,
            IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => 8,
            IrType::I128 => 16,
            IrType::F32 => 4,
            IrType::F64 => 8,
            IrType::F80 => 12, // 80-bit extended, 12 bytes on i686 per SysV i386 ABI
            IrType::Ptr => 4,  // 32-bit pointers on i686
            IrType::Array { element, count } => self.type_size_bytes(element) * (*count as u32),
            IrType::Struct { fields, packed } => {
                if *packed {
                    fields.iter().map(|f| self.type_size_bytes(f)).sum()
                } else {
                    let mut size: u32 = 0;
                    let mut max_align: u32 = 1;
                    for field in fields {
                        let field_align = self.type_alignment(field);
                        let field_size = self.type_size_bytes(field);
                        size = align_up(size, field_align);
                        size += field_size;
                        max_align = max_align.max(field_align);
                    }
                    align_up(size, max_align)
                }
            }
            IrType::Function { .. } => 4, // function pointers are 32-bit
        }
    }

    /// Returns the size of an IR type in bits.
    fn type_size_bits(&self, ty: &IrType) -> u32 {
        match ty {
            IrType::I1 => 1,
            IrType::I8 => 8,
            IrType::I16 => 16,
            IrType::I32 => 32,
            IrType::I64 => 64,
            IrType::I128 => 128,
            IrType::F32 => 32,
            IrType::F64 => 64,
            IrType::F80 => 80,
            IrType::Ptr => 32, // 32-bit on i686
            _ => self.type_size_bytes(ty) * 8,
        }
    }

    /// Returns the alignment of an IR type in bytes for the i686 target.
    fn type_alignment(&self, ty: &IrType) -> u32 {
        match ty {
            IrType::Void => 1,
            IrType::I1 => 1,
            IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => 4, // 8-byte values are 4-byte aligned on i686
            IrType::I128 => 4,
            IrType::F32 => 4,
            IrType::F64 => 4, // 4-byte aligned on i686 (not 8 like x86-64)
            IrType::F80 => 4, // 4-byte aligned on i686 per SysV i386 ABI
            IrType::Ptr => 4, // 32-bit pointers, 4-byte aligned
            IrType::Array { element, .. } => self.type_alignment(element),
            IrType::Struct { fields, packed } => {
                if *packed {
                    1
                } else {
                    fields
                        .iter()
                        .map(|f| self.type_alignment(f))
                        .max()
                        .unwrap_or(1)
                }
            }
            IrType::Function { .. } => 4,
        }
    }

    /// Looks up the IR type for a value.
    fn get_value_type(&self, vid: ValueId, func: &IrFunction) -> IrType {
        let idx = vid.index() as usize;
        if idx < func.local_values.len() {
            func.local_values[idx].ty.clone()
        } else {
            // Default to I32 for unknown values (safety fallback).
            IrType::I32
        }
    }
}

// ===========================================================================
// Utility: align_up
// ===========================================================================

/// Rounds `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two.
#[inline]
fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    let mask = alignment - 1;
    (value + mask) & !mask
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_code_from_icmp() {
        assert_eq!(
            ConditionCode::from_icmp_predicate(&ICmpPredicate::Eq),
            ConditionCode::E
        );
        assert_eq!(
            ConditionCode::from_icmp_predicate(&ICmpPredicate::Ne),
            ConditionCode::Ne
        );
        assert_eq!(
            ConditionCode::from_icmp_predicate(&ICmpPredicate::Slt),
            ConditionCode::L
        );
        assert_eq!(
            ConditionCode::from_icmp_predicate(&ICmpPredicate::Ult),
            ConditionCode::B
        );
        assert_eq!(
            ConditionCode::from_icmp_predicate(&ICmpPredicate::Sge),
            ConditionCode::Ge
        );
        assert_eq!(
            ConditionCode::from_icmp_predicate(&ICmpPredicate::Uge),
            ConditionCode::Ae
        );
    }

    #[test]
    fn test_condition_code_invert() {
        assert_eq!(ConditionCode::E.invert(), ConditionCode::Ne);
        assert_eq!(ConditionCode::Ne.invert(), ConditionCode::E);
        assert_eq!(ConditionCode::L.invert(), ConditionCode::Ge);
        assert_eq!(ConditionCode::Ge.invert(), ConditionCode::L);
        assert_eq!(ConditionCode::B.invert(), ConditionCode::Ae);
        assert_eq!(ConditionCode::Ae.invert(), ConditionCode::B);
        assert_eq!(ConditionCode::Le.invert(), ConditionCode::G);
        assert_eq!(ConditionCode::G.invert(), ConditionCode::Le);
    }

    #[test]
    fn test_opcode_round_trip() {
        for opcode in [
            I686Opcode::Mov,
            I686Opcode::Add,
            I686Opcode::Sub,
            I686Opcode::Imul,
            I686Opcode::Idiv,
            I686Opcode::Shl,
            I686Opcode::Cmp,
            I686Opcode::Jmp,
            I686Opcode::Jcc,
            I686Opcode::Setcc,
            I686Opcode::Call,
            I686Opcode::Ret,
            I686Opcode::Fld,
            I686Opcode::Fstp,
            I686Opcode::Faddp,
            I686Opcode::Nop,
        ] {
            let val = opcode.as_u32();
            assert_eq!(I686Opcode::from_u32(val), Some(opcode));
        }
    }

    #[test]
    fn test_opcode_from_u32_invalid() {
        assert_eq!(I686Opcode::from_u32(999), None);
        assert_eq!(I686Opcode::from_u32(100), None);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn test_condition_code_encoding() {
        assert_eq!(ConditionCode::E.encoding(), 0x4);
        assert_eq!(ConditionCode::Ne.encoding(), 0x5);
        assert_eq!(ConditionCode::B.encoding(), 0x2);
        assert_eq!(ConditionCode::Ae.encoding(), 0x3);
        assert_eq!(ConditionCode::L.encoding(), 0xC);
        assert_eq!(ConditionCode::G.encoding(), 0xF);
    }

    #[test]
    fn test_opcode_display() {
        assert_eq!(format!("{}", I686Opcode::Mov), "mov");
        assert_eq!(format!("{}", I686Opcode::Add), "add");
        assert_eq!(format!("{}", I686Opcode::Imul), "imul");
        assert_eq!(format!("{}", I686Opcode::Faddp), "faddp");
        assert_eq!(format!("{}", I686Opcode::Ret), "ret");
    }

    #[test]
    fn test_condition_code_display() {
        assert_eq!(format!("{}", ConditionCode::E), "e");
        assert_eq!(format!("{}", ConditionCode::Ne), "ne");
        assert_eq!(format!("{}", ConditionCode::L), "l");
        assert_eq!(format!("{}", ConditionCode::Ge), "ge");
        assert_eq!(format!("{}", ConditionCode::A), "a");
    }
}
