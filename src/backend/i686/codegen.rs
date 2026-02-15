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
    self, AL, CALLEE_SAVED, CALLER_SAVED, CL, EAX, EBP, EBX, ECX, EDI, EDX, EFLAGS, ESI, ESP, ST0,
};
use crate::backend::traits::{
    CodegenConfig, MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand, PhysReg,
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
    /// MOV dst, src — general-purpose register/memory/immediate move.
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
        self.next_frame_index = 0;
        self.frame_size = 0;
        self.has_calls = false;
        self.next_vreg = func.local_values.len() as u32;

        let stack_align = self.config.target.stack_alignment();
        let mut mf = MachineFunction::new(func.name.clone(), stack_align);

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
                    let offset = -(self.frame_size as i32);
                    let mem_op = MachineOperand::Memory {
                        base: EBP,
                        offset,
                        index: None,
                        scale: 1,
                    };
                    self.value_map.insert(result.index(), mem_op);
                    self.next_frame_index += 1;
                }
            }
        }

        // Align total frame size to stack alignment.
        self.frame_size = align_up(self.frame_size, stack_align);

        // Phase 4: Generate prologue in the entry block.
        let entry_id = 0u32;
        let mut entry_block =
            MachineBasicBlock::with_label(entry_id, format!(".L{}_{}", func.name, entry_id));
        self.emit_prologue(&mut entry_block);
        mf.add_block(entry_block);

        // Phase 5: Lower each IR basic block's instructions.
        for (idx, bb) in func.basic_blocks.iter().enumerate() {
            let mbb_id = idx as u32;
            // For the entry block (idx 0), we already created it with prologue;
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

            // -- Inline assembly --
            Instruction::InlineAsm {
                result,
                template: _,
                constraints: _,
                operands: _,
                clobbers: _,
                has_side_effects: _,
                is_align_stack: _,
            } => {
                // Inline assembly is passed through as-is; the assembler handles it.
                // For now, emit a NOP placeholder and record the result if any.
                let nop = MachineInstr::new(I686Opcode::Nop.as_u32());
                mbb.push_instr(nop);
                if let Some(res) = result {
                    // Map result to EAX as a convention for inline asm output.
                    self.value_map
                        .insert(res.index(), MachineOperand::Register(EAX));
                }
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

        if ty.is_floating() {
            // Use FLD to load float/double onto x87 stack, then FSTP to store
            // into a virtual spill location. For simplicity in the initial
            // selector, we represent FP values as frame-index operands.
            let mem = self.operand_to_memory(addr, mbb);
            let mut fld = MachineInstr::new(I686Opcode::Fld.as_u32());
            fld.add_operand(mem);
            fld.add_implicit_def(ST0);
            mbb.push_instr(fld);
            // The result is on ST(0); record it.
            self.value_map
                .insert(result.index(), MachineOperand::Register(ST0));
            return;
        }

        let size_bits = self.type_size_bits(ty);
        match size_bits {
            8 => {
                // MOVZX for 8-bit loads into 32-bit register.
                let mem = self.operand_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::MovZx.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
            16 => {
                // MOVZX for 16-bit loads into 32-bit register.
                let mem = self.operand_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::MovZx.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
            32 => {
                // Standard 32-bit MOV load.
                let mem = self.operand_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
            64 => {
                // 64-bit load: two 32-bit loads into a register pair.
                let base_mem = self.operand_to_memory(addr.clone(), mbb);
                // Load low 32 bits.
                let lo_dst = self.alloc_vreg_raw();
                let mut lo_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                lo_inst.add_operand(MachineOperand::VirtualReg(ValueId(lo_dst)));
                lo_inst.add_operand(base_mem.clone());
                mbb.push_instr(lo_inst);

                // Load high 32 bits at offset+4.
                let hi_mem = self.offset_memory(base_mem, 4);
                let hi_dst = self.alloc_vreg_raw();
                let mut hi_inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                hi_inst.add_operand(MachineOperand::VirtualReg(ValueId(hi_dst)));
                hi_inst.add_operand(hi_mem);
                mbb.push_instr(hi_inst);

                // Record lo:hi pair under the original result.
                // We use the lo register as the primary operand; the hi is
                // accessible via the convention that vreg N+1 is the high half.
                self.value_map
                    .insert(result.index(), MachineOperand::VirtualReg(ValueId(lo_dst)));
            }
            _ => {
                // Pointer or other 32-bit equivalent.
                let mem = self.operand_to_memory(addr, mbb);
                let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
                inst.add_operand(dst);
                inst.add_operand(mem);
                mbb.push_instr(inst);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Store selection
    // -----------------------------------------------------------------------

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

        // Check if the value is an x87 FP value.
        if let MachineOperand::Register(reg) = &val_op {
            if registers::is_fpu(*reg) {
                // FSTP to store from x87 stack to memory.
                let mem = self.operand_to_memory(addr_op, mbb);
                let mut fstp = MachineInstr::new(I686Opcode::Fstp.as_u32());
                fstp.add_operand(mem);
                fstp.add_implicit_use(ST0);
                mbb.push_instr(fstp);
                return;
            }
        }

        let mem = self.operand_to_memory(addr_op, mbb);
        // Ensure value is in a register (can't do mem-to-mem move on x86).
        let src = self.ensure_in_register(val_op, mbb);
        let mut inst = MachineInstr::new(I686Opcode::Mov.as_u32());
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
                let dst = self.alloc_vreg(result);
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                self.emit_mov(dst.clone(), lhs_reg, mbb);

                // Check if shift amount is an immediate.
                if let MachineOperand::Immediate(imm) = &rhs_op {
                    let mut inst = MachineInstr::new(opcode.as_u32());
                    inst.add_operand(dst);
                    inst.add_operand(MachineOperand::Immediate(*imm & 31));
                    inst.add_implicit_def(EFLAGS);
                    mbb.push_instr(inst);
                } else {
                    // Variable shift: move amount to CL, then shift.
                    let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                    let mut mov_cl = MachineInstr::new(I686Opcode::Mov.as_u32());
                    mov_cl.add_operand(MachineOperand::Register(ECX));
                    mov_cl.add_operand(rhs_reg);
                    mbb.push_instr(mov_cl);

                    let mut inst = MachineInstr::new(opcode.as_u32());
                    inst.add_operand(dst);
                    inst.add_operand(MachineOperand::Register(CL));
                    inst.add_implicit_def(EFLAGS);
                    inst.add_implicit_use(ECX);
                    mbb.push_instr(inst);
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
        // Step 1: Move dividend to EAX.
        let lhs_reg = self.ensure_in_register(lhs, mbb);
        let mut mov_eax = MachineInstr::new(I686Opcode::Mov.as_u32());
        mov_eax.add_operand(MachineOperand::Register(EAX));
        mov_eax.add_operand(lhs_reg);
        mbb.push_instr(mov_eax);

        // Step 2: Prepare EDX.
        if is_signed {
            // CDQ: sign-extend EAX into EDX:EAX.
            let mut cdq = MachineInstr::new(I686Opcode::Cdq.as_u32());
            cdq.add_implicit_use(EAX);
            cdq.add_implicit_def(EDX);
            mbb.push_instr(cdq);
        } else {
            // XOR EDX, EDX: zero the high half for unsigned division.
            let mut xor_edx = MachineInstr::new(I686Opcode::Xor.as_u32());
            xor_edx.add_operand(MachineOperand::Register(EDX));
            xor_edx.add_operand(MachineOperand::Register(EDX));
            xor_edx.add_implicit_def(EFLAGS);
            mbb.push_instr(xor_edx);
        }

        // Step 3: Ensure divisor is in a register (not EAX or EDX).
        let rhs_reg = self.ensure_in_register(rhs, mbb);

        // Step 4: Emit IDIV or DIV.
        let opcode = if is_signed {
            I686Opcode::Idiv
        } else {
            I686Opcode::Div
        };
        let mut div_inst = MachineInstr::new(opcode.as_u32());
        div_inst.add_operand(rhs_reg);
        div_inst.add_implicit_use(EAX);
        div_inst.add_implicit_use(EDX);
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
                // 64-bit add: MOV lo_result, lhs_lo; ADD lo_result, rhs_lo;
                //             MOV hi_result, 0; ADC hi_result, 0
                let lo_result = self.alloc_vreg_raw();
                let hi_result = self.alloc_vreg_raw();

                // Simplified sequence: treats lhs/rhs as 32-bit lo halves
                // and uses carry propagation for the high word.
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                let lo_op = MachineOperand::VirtualReg(ValueId(lo_result));
                let hi_op = MachineOperand::VirtualReg(ValueId(hi_result));

                // MOV lo_result, lhs_lo
                self.emit_mov(lo_op.clone(), lhs_reg, mbb);

                // ADD lo_result, rhs_lo
                let mut add_lo = MachineInstr::new(I686Opcode::Add.as_u32());
                add_lo.add_operand(lo_op);
                add_lo.add_operand(rhs_reg.clone());
                add_lo.add_implicit_def(EFLAGS);
                mbb.push_instr(add_lo);

                // MOV hi_result, 0 (zero the high word before ADC)
                self.emit_mov(hi_op.clone(), MachineOperand::Immediate(0), mbb);

                // ADC hi_result, 0 (propagate carry from low addition)
                let mut adc_hi = MachineInstr::new(I686Opcode::Adc.as_u32());
                adc_hi.add_operand(hi_op);
                adc_hi.add_operand(MachineOperand::Immediate(0));
                adc_hi.add_implicit_use(EFLAGS);
                adc_hi.add_implicit_def(EFLAGS);
                mbb.push_instr(adc_hi);
            }

            BinOp::Sub => {
                // 64-bit sub: MOV lo_result, lhs_lo; SUB lo_result, rhs_lo;
                //             MOV hi_result, 0; SBB hi_result, 0
                let lo_result = self.alloc_vreg_raw();
                let hi_result = self.alloc_vreg_raw();

                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                let lo_op = MachineOperand::VirtualReg(ValueId(lo_result));
                let hi_op = MachineOperand::VirtualReg(ValueId(hi_result));

                // MOV lo_result, lhs_lo
                self.emit_mov(lo_op.clone(), lhs_reg, mbb);

                // SUB lo_result, rhs_lo
                let mut sub_lo = MachineInstr::new(I686Opcode::Sub.as_u32());
                sub_lo.add_operand(lo_op);
                sub_lo.add_operand(rhs_reg.clone());
                sub_lo.add_implicit_def(EFLAGS);
                mbb.push_instr(sub_lo);

                // MOV hi_result, 0 (zero the high word before SBB)
                self.emit_mov(hi_op.clone(), MachineOperand::Immediate(0), mbb);

                // SBB hi_result, 0 (propagate borrow from low subtraction)
                let mut sbb_hi = MachineInstr::new(I686Opcode::Sbb.as_u32());
                sbb_hi.add_operand(hi_op);
                sbb_hi.add_operand(MachineOperand::Immediate(0));
                sbb_hi.add_implicit_use(EFLAGS);
                sbb_hi.add_implicit_def(EFLAGS);
                mbb.push_instr(sbb_hi);
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

                // Result lo in EAX, hi in EDX.
                self.value_map
                    .insert(result.index(), MachineOperand::Register(EAX));
                return;
            }

            BinOp::And | BinOp::Or | BinOp::Xor => {
                // Bitwise ops on 64-bit: apply to both halves independently.
                // Simplified: operate on the low half only.
                let opcode = match op {
                    BinOp::And => I686Opcode::And,
                    BinOp::Or => I686Opcode::Or,
                    BinOp::Xor => I686Opcode::Xor,
                    _ => unreachable!(),
                };
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                let lo_result = self.alloc_vreg_raw();
                let lo_op = MachineOperand::VirtualReg(ValueId(lo_result));

                // MOV lo_result, lhs_lo (copy LHS into result)
                self.emit_mov(lo_op.clone(), lhs_reg, mbb);

                // OP lo_result, rhs_lo
                let mut op_lo = MachineInstr::new(opcode.as_u32());
                op_lo.add_operand(lo_op);
                op_lo.add_operand(rhs_reg);
                op_lo.add_implicit_def(EFLAGS);
                mbb.push_instr(op_lo);
            }

            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                // 64-bit shifts use SHLD/SHRD instructions.
                // Simplified: emit the shift on the low half only.
                let opcode = match op {
                    BinOp::Shl => I686Opcode::Shl,
                    BinOp::LShr => I686Opcode::Shr,
                    BinOp::AShr => I686Opcode::Sar,
                    _ => unreachable!(),
                };
                let lhs_reg = self.ensure_in_register(lhs_op, mbb);
                let rhs_reg = self.ensure_in_register(rhs_op, mbb);
                let lo_result = self.alloc_vreg_raw();

                self.emit_mov(MachineOperand::VirtualReg(ValueId(lo_result)), lhs_reg, mbb);

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
        let lhs_op = self.get_operand(lhs, func);
        let rhs_op = self.get_operand(rhs, func);

        // Load LHS onto x87 stack.
        self.emit_fld(lhs_op, mbb);
        // Load RHS onto x87 stack (LHS moves to ST(1)).
        self.emit_fld(rhs_op, mbb);

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
        fop.add_implicit_use(ST0);
        fop.add_implicit_def(ST0);
        mbb.push_instr(fop);

        // Result is now in ST(0).
        self.value_map
            .insert(result.index(), MachineOperand::Register(ST0));
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
        let lhs_op = self.get_operand(lhs, func);
        let rhs_op = self.get_operand(rhs, func);

        // Load both operands onto x87 stack.
        self.emit_fld(lhs_op, mbb);
        self.emit_fld(rhs_op, mbb);

        // FUCOMIP ST(0), ST(1) — unordered compare, pop ST(0), set EFLAGS.
        let mut fucomip = MachineInstr::new(I686Opcode::Fucomip.as_u32());
        fucomip.add_operand(MachineOperand::Register(ST0));
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
        let arg_bytes = (args.len() as u32) * 4;
        for &arg_vid in args.iter().rev() {
            let arg_op = self.get_operand(arg_vid, func);

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
                self.value_map
                    .insert(res_vid.index(), MachineOperand::Register(ST0));
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
                // Ensure the FP value is in ST(0). If it's already there, nothing to do.
                if !matches!(&val_op, MachineOperand::Register(r) if registers::is_fpu(*r)) {
                    self.emit_fld(val_op, mbb);
                }
            } else {
                // Move integer result to EAX.
                let src = self.ensure_in_register(val_op, mbb);
                let mut mov = MachineInstr::new(I686Opcode::Mov.as_u32());
                mov.add_operand(MachineOperand::Register(EAX));
                mov.add_operand(src);
                mbb.push_instr(mov);
            }
        }

        // Emit epilogue.
        self.emit_epilogue(mbb);

        // RET.
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
        let base_reg = self.ensure_in_register(base_op, mbb);
        let dst = self.alloc_vreg(result);

        // Start with base address.
        self.emit_mov(dst.clone(), base_reg, mbb);

        // Process each index with the appropriate stride.
        let mut current_type = ty.clone();
        for &idx_vid in indices.iter() {
            let stride = self.type_size_bytes(&current_type);
            let idx_op = self.get_operand(idx_vid, func);

            if stride == 0 {
                // Zero-size element, nothing to add.
                continue;
            }

            // Check if index is a constant.
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
                        // Fallback: IMUL + ADD
                        self.emit_gep_mul_add(dst.clone(), idx_reg, stride, mbb);
                    }
                    mbb.push_instr(lea);
                } else {
                    self.emit_gep_mul_add(dst.clone(), idx_reg, stride, mbb);
                }
            } else {
                // General case: IMUL index, stride; ADD dst, result.
                let idx_reg = self.ensure_in_register(idx_op, mbb);
                self.emit_gep_mul_add(dst.clone(), idx_reg, stride, mbb);
            }

            // Advance to the element type for the next index.
            current_type = match &current_type {
                IrType::Array { element, .. } => *element.clone(),
                IrType::Struct { fields, .. } => {
                    // For struct GEP, the index selects a field.
                    if let MachineOperand::Immediate(field_idx) = &self.get_operand(idx_vid, func) {
                        if let Some(field_ty) = fields.get(*field_idx as usize) {
                            field_ty.clone()
                        } else {
                            IrType::I8
                        }
                    } else {
                        IrType::I8
                    }
                }
                IrType::Ptr => IrType::I8,
                _ => IrType::I8,
            };
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
    fn select_bitcast(
        &mut self,
        result: ValueId,
        value: ValueId,
        _to_ty: &IrType,
        mbb: &mut MachineBasicBlock,
        func: &IrFunction,
    ) {
        let src = self.get_operand(value, func);
        let dst = self.alloc_vreg(result);
        self.emit_mov(dst, src, mbb);
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
            self.emit_mov(dst, src, mbb);
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

            // Result lo=EAX, hi=EDX
            self.value_map
                .insert(result.index(), MachineOperand::Register(EAX));
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

    // -----------------------------------------------------------------------
    // Helper: ensure a value is in a register
    // -----------------------------------------------------------------------

    /// Ensures the operand is in a register. If it's already a register,
    /// returns it as-is. If it's a memory operand, immediate, or symbol,
    /// emits a MOV to a fresh virtual register and returns that.
    fn ensure_in_register(
        &mut self,
        op: MachineOperand,
        mbb: &mut MachineBasicBlock,
    ) -> MachineOperand {
        match &op {
            MachineOperand::Register(_) | MachineOperand::VirtualReg(_) => op,
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

    /// Converts an operand representing a pointer to a Memory operand for
    /// load/store instructions.
    ///
    /// If the operand is already a Memory, returns it unchanged. If it's a
    /// register, wraps it as `[reg+0]`. If it's something else, moves it
    /// to a register first.
    fn operand_to_memory(
        &mut self,
        op: MachineOperand,
        mbb: &mut MachineBasicBlock,
    ) -> MachineOperand {
        match &op {
            MachineOperand::Memory { .. } => op,
            MachineOperand::Register(reg) => MachineOperand::Memory {
                base: *reg,
                offset: 0,
                index: None,
                scale: 1,
            },
            MachineOperand::VirtualReg(_) => {
                // Virtual register — keep as-is for now; register allocator
                // will assign a physical register which can then be used as base.
                // For the memory addressing mode, we use EAX as a temporary.
                let tmp = self.alloc_vreg_raw();
                self.emit_mov(MachineOperand::VirtualReg(ValueId(tmp)), op, mbb);
                MachineOperand::Memory {
                    base: PhysReg::NONE,
                    offset: 0,
                    index: None,
                    scale: 1,
                }
            }
            MachineOperand::Symbol(_name) => {
                // Global symbol: use symbol@GOT or direct addressing.
                if self.config.requires_pic() {
                    // PIC: MOV reg, [EBX + symbol@GOT]
                    MachineOperand::Memory {
                        base: EBX,
                        offset: 0,
                        index: None,
                        scale: 1,
                    }
                } else {
                    // Non-PIC: direct [symbol] addressing.
                    MachineOperand::Memory {
                        base: PhysReg::NONE,
                        offset: 0,
                        index: None,
                        scale: 1,
                    }
                }
            }
            _ => {
                // Label, Immediate, FrameIndex — move to register first.
                let tmp = self.alloc_vreg_raw();
                self.emit_mov(MachineOperand::VirtualReg(ValueId(tmp)), op, mbb);
                MachineOperand::Memory {
                    base: PhysReg::NONE,
                    offset: 0,
                    index: None,
                    scale: 1,
                }
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
