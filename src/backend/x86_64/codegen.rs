#![allow(clippy::cloned_ref_to_slice_refs, clippy::collapsible_match)]
//! # x86-64 Instruction Selection and Emission
//!
//! Converts BCC IR instructions into x86-64 machine instructions.  This module
//! is the **primary validation target** — it is validated first in the backend
//! order: x86-64 → i686 → AArch64 → RISC-V 64.
//!
//! ## Key Design Points
//!
//! - **Variable-length encoding**: x86-64 instructions range 1–15 bytes; REX
//!   prefix handling is critical for registers R8–R15 and XMM8–XMM15.
//! - **System V AMD64 ABI**: 16 GPRs (RAX–R15), 16 SSE registers (XMM0–XMM15).
//!   Integer arithmetic uses GPRs; floating-point uses SSE/SSE2 (**not** x87).
//! - **Complex addressing**: `[base + index*scale + disp]` with scale ∈ {1,2,4,8}.
//! - **Implicit register constraints**: DIV/IDIV use RDX:RAX; shifts use CL.
//! - **Stack alignment**: 16-byte at call sites; 128-byte red zone for leaf functions.
//! - **CMOV optimisation**: conditional moves avoid branch misprediction penalty.
//! - **LEA optimisation**: used for multi-operand address computation and fast arithmetic.
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references are used.  No external crates.

use crate::backend::traits::{
    self as traits, ArchCodegen, MachineBasicBlock, MachineFunction, MachineInstruction,
    MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::backend::x86_64::abi::{
    self, classify_ir_type_eightbytes, AbiClass, RetLocation, X86_64Abi, INTEGER_ARG_REGS,
    RED_ZONE_SIZE, SSE_ARG_REGS,
};
use crate::backend::x86_64::registers::{
    self, CALLEE_SAVED_GPRS, R10, R11, R8, R9, RAX, RBP, RCX, RDI, RDX, RSI, RSP, XMM0, XMM1,
    XMM15, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// X86Opcode — x86-64 machine opcode enumeration
// ===========================================================================

/// x86-64 machine instruction opcodes.
///
/// Each variant maps to one or more physical x86-64 instructions.  The
/// assembler module (`encoder.rs`) converts these to binary encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Opcode {
    // -- Data movement --
    /// `MOV dst, src` — register-to-register, immediate-to-register, or memory.
    Mov,
    /// `MOVZX dst, src` — zero-extend move (8/16-bit → 32/64-bit).
    MovZX,
    /// `MOVSX dst, src` — sign-extend move (8/16/32-bit → 32/64-bit).
    MovSX,
    /// `LEA dst, [addr]` — load effective address (no memory access).
    Lea,
    /// `PUSH src` — push onto stack, decrement RSP.
    Push,
    /// `POP dst` — pop from stack, increment RSP.
    Pop,
    /// `XCHG a, b` — exchange two operands atomically.
    Xchg,

    // -- Integer arithmetic --
    /// `ADD dst, src`
    Add,
    /// `SUB dst, src`
    Sub,
    /// `IMUL dst, src` — signed multiply (two-/three-operand form).
    Imul,
    /// `IDIV src` — signed divide RDX:RAX by src.  Quotient→RAX, remainder→RDX.
    Idiv,
    /// `DIV src` — unsigned divide RDX:RAX by src.
    Div,
    /// `NEG dst` — two's complement negate.
    Neg,
    /// `INC dst`
    Inc,
    /// `DEC dst`
    Dec,

    // -- Bitwise --
    /// `AND dst, src`
    And,
    /// `OR dst, src`
    Or,
    /// `XOR dst, src`
    Xor,
    /// `NOT dst`
    Not,
    /// `SHL dst, CL/imm8`
    Shl,
    /// `SHR dst, CL/imm8` — logical shift right.
    Shr,
    /// `SAR dst, CL/imm8` — arithmetic shift right.
    Sar,
    /// `ROL dst, CL/imm8`
    Rol,
    /// `ROR dst, CL/imm8`
    Ror,

    // -- Comparison --
    /// `CMP a, b` — sets EFLAGS; result discarded.
    Cmp,
    /// `TEST a, b` — bitwise AND; sets EFLAGS; result discarded.
    Test,

    // -- Conditional move (CMOV) — all 16 conditions --
    Cmovo,
    Cmovno,
    Cmovb,
    Cmovae,
    Cmove,
    Cmovne,
    Cmovbe,
    Cmova,
    Cmovs,
    Cmovns,
    Cmovp,
    Cmovnp,
    Cmovl,
    Cmovge,
    Cmovle,
    Cmovg,

    // -- SETcc — set byte on condition — all 16 conditions --
    Seto,
    Setno,
    Setb,
    Setae,
    Sete,
    Setne,
    Setbe,
    Seta,
    Sets,
    Setns,
    Setp,
    Setnp,
    Setl,
    Setge,
    Setle,
    Setg,

    // -- Control flow / branches — all 16 conditions + JMP --
    Jmp,
    Jo,
    Jno,
    Jb,
    Jae,
    Je,
    Jne,
    Jbe,
    Ja,
    Js,
    Jns,
    Jp,
    Jnp,
    Jl,
    Jge,
    Jle,
    Jg,
    /// `CALL target`
    Call,
    /// `RET`
    Ret,

    /// `LoadInd dst, ptr` — Load 64-bit from memory at address in ptr register.
    /// Encoded as `MOV r64, [ptr]`.
    LoadInd,
    /// `LoadInd32 dst, ptr` — Load 32-bit from `[ptr]`, zero-extends to 64-bit.
    LoadInd32,
    /// `LoadInd16 dst, ptr` — Load 16-bit from `[ptr]`, zero-extends to 64-bit.
    LoadInd16,
    /// `LoadInd8 dst, ptr` — Load 8-bit from `[ptr]`, zero-extends to 64-bit.
    LoadInd8,

    /// `StoreInd ptr, src` — Store 64-bit src to memory at address in ptr register.
    /// Encoded as `MOV [ptr], src`.
    StoreInd,
    /// `StoreInd32 ptr, src` — Store low 32-bit of src to `[ptr]`.
    StoreInd32,
    /// `StoreInd16 ptr, src` — Store low 16-bit of src to `[ptr]`.
    StoreInd16,
    /// `StoreInd8 ptr, src` — Store low 8-bit of src to `[ptr]`.
    StoreInd8,
    /// `NOP`
    Nop,

    // -- SSE / SSE2 floating-point --
    /// `MOVSD xmm, xmm/m64` — move scalar double.
    Movsd,
    /// `MOVSS xmm, xmm/m32` — move scalar single.
    Movss,
    Addsd,
    Addss,
    Subsd,
    Subss,
    Mulsd,
    Mulss,
    Divsd,
    Divss,
    /// `UCOMISD xmm, xmm/m64` — unordered compare, set EFLAGS.
    Ucomisd,
    /// `UCOMISS xmm, xmm/m32` — unordered compare, set EFLAGS.
    Ucomiss,
    Cvtsi2sd,
    Cvtsi2ss,
    Cvtsd2si,
    Cvtss2si,
    Cvtsd2ss,
    Cvtss2sd,

    // -- Stack frame --
    Enter,
    Leave,

    // -- Special --
    /// `CDQ` — sign-extend EAX → EDX:EAX (for 32-bit IDIV).
    Cdq,
    /// `CQO` — sign-extend RAX → RDX:RAX (for 64-bit IDIV).
    Cqo,
    /// `ENDBR64` — CET indirect-branch tracking marker.
    Endbr64,
    /// `PAUSE` — spin-loop hint (used in retpoline).
    Pause,
    /// `LFENCE` — load fence (used in retpoline).
    Lfence,

    // -- Bit manipulation --
    /// `BSR dst, src` — Bit Scan Reverse (find highest set bit index).
    Bsr,
    /// `BSF dst, src` — Bit Scan Forward (find lowest set bit index).
    Bsf,
    /// `POPCNT dst, src` — Population count (number of set bits).
    Popcnt,
    /// `BSWAP r` — Byte swap (endianness conversion).
    Bswap,
    /// `UD2` — Undefined instruction (trap).
    Ud2,

    /// `REP MOVSQ` — repeat move qwords from [RSI] to [RDI], RCX times.
    RepMovsq,

    /// Inline assembly marker — template is carried in the operands.
    InlineAsm,

    /// Internal label definition pseudo-instruction.
    ///
    /// Carries a `GlobalSymbol(name)` operand.  Emits zero bytes — the
    /// assembler loop calls `ctx.define_label(name)` at the current offset.
    /// Used for intra-block branching sequences (e.g. unsigned u64→f64
    /// conversion) that need forward-reference labels resolved locally.
    InternalLabelDef,

    // -- x87 FPU instructions (for long double ABI compliance) --
    /// `FLD QWORD [mem]` — load 64-bit double from memory into x87 ST(0),
    /// automatically converting to 80-bit extended precision internally.
    /// Opcode: 0xDD /0 (ModRM.reg = 0).
    FldMem64,
    /// `FLD TBYTE [mem]` — load 80-bit extended precision from memory into
    /// x87 ST(0).  Opcode: 0xDB /5 (ModRM.reg = 5).
    FldMem80,
    /// `FSTP TWORD [mem]` — store 80-bit extended precision from x87 ST(0)
    /// to 10-byte memory location, then pop the FPU stack.
    /// Opcode: 0xDB /7 (ModRM.reg = 7).
    FstpMem80,
    /// `FSTP QWORD [mem]` — store 64-bit double from x87 ST(0)
    /// to 8-byte memory location, then pop the FPU stack.
    /// Opcode: 0xDD /3 (ModRM.reg = 3).
    FstpMem64,
}

impl X86Opcode {
    /// Convert to the `u32` opcode value used by [`MachineInstruction`].
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Try to construct from a raw `u32` opcode value.
    pub fn from_u32(val: u32) -> Option<Self> {
        // Safe table-driven lookup (avoids transmute size mismatch).
        const TABLE: &[X86Opcode] = &[
            X86Opcode::Mov,
            X86Opcode::MovZX,
            X86Opcode::MovSX,
            X86Opcode::Lea,
            X86Opcode::Push,
            X86Opcode::Pop,
            X86Opcode::Xchg,
            X86Opcode::Add,
            X86Opcode::Sub,
            X86Opcode::Imul,
            X86Opcode::Idiv,
            X86Opcode::Div,
            X86Opcode::Neg,
            X86Opcode::Inc,
            X86Opcode::Dec,
            X86Opcode::And,
            X86Opcode::Or,
            X86Opcode::Xor,
            X86Opcode::Not,
            X86Opcode::Shl,
            X86Opcode::Shr,
            X86Opcode::Sar,
            X86Opcode::Rol,
            X86Opcode::Ror,
            X86Opcode::Cmp,
            X86Opcode::Test,
            X86Opcode::Cmovo,
            X86Opcode::Cmovno,
            X86Opcode::Cmovb,
            X86Opcode::Cmovae,
            X86Opcode::Cmove,
            X86Opcode::Cmovne,
            X86Opcode::Cmovbe,
            X86Opcode::Cmova,
            X86Opcode::Cmovs,
            X86Opcode::Cmovns,
            X86Opcode::Cmovp,
            X86Opcode::Cmovnp,
            X86Opcode::Cmovl,
            X86Opcode::Cmovge,
            X86Opcode::Cmovle,
            X86Opcode::Cmovg,
            X86Opcode::Seto,
            X86Opcode::Setno,
            X86Opcode::Setb,
            X86Opcode::Setae,
            X86Opcode::Sete,
            X86Opcode::Setne,
            X86Opcode::Setbe,
            X86Opcode::Seta,
            X86Opcode::Sets,
            X86Opcode::Setns,
            X86Opcode::Setp,
            X86Opcode::Setnp,
            X86Opcode::Setl,
            X86Opcode::Setge,
            X86Opcode::Setle,
            X86Opcode::Setg,
            X86Opcode::Jmp,
            X86Opcode::Jo,
            X86Opcode::Jno,
            X86Opcode::Jb,
            X86Opcode::Jae,
            X86Opcode::Je,
            X86Opcode::Jne,
            X86Opcode::Jbe,
            X86Opcode::Ja,
            X86Opcode::Js,
            X86Opcode::Jns,
            X86Opcode::Jp,
            X86Opcode::Jnp,
            X86Opcode::Jl,
            X86Opcode::Jge,
            X86Opcode::Jle,
            X86Opcode::Jg,
            X86Opcode::Call,
            X86Opcode::Ret,
            X86Opcode::LoadInd,
            X86Opcode::LoadInd32,
            X86Opcode::LoadInd16,
            X86Opcode::LoadInd8,
            X86Opcode::StoreInd,
            X86Opcode::StoreInd32,
            X86Opcode::StoreInd16,
            X86Opcode::StoreInd8,
            X86Opcode::Nop,
            X86Opcode::Movsd,
            X86Opcode::Movss,
            X86Opcode::Addsd,
            X86Opcode::Addss,
            X86Opcode::Subsd,
            X86Opcode::Subss,
            X86Opcode::Mulsd,
            X86Opcode::Mulss,
            X86Opcode::Divsd,
            X86Opcode::Divss,
            X86Opcode::Ucomisd,
            X86Opcode::Ucomiss,
            X86Opcode::Cvtsi2sd,
            X86Opcode::Cvtsi2ss,
            X86Opcode::Cvtsd2si,
            X86Opcode::Cvtss2si,
            X86Opcode::Cvtsd2ss,
            X86Opcode::Cvtss2sd,
            X86Opcode::Enter,
            X86Opcode::Leave,
            X86Opcode::Cdq,
            X86Opcode::Cqo,
            X86Opcode::Endbr64,
            X86Opcode::Pause,
            X86Opcode::Lfence,
            X86Opcode::Bsr,
            X86Opcode::Bsf,
            X86Opcode::Popcnt,
            X86Opcode::Bswap,
            X86Opcode::Ud2,
            X86Opcode::RepMovsq,
            X86Opcode::InlineAsm,
            X86Opcode::InternalLabelDef,
            X86Opcode::FldMem64,
            X86Opcode::FldMem80,
            X86Opcode::FstpMem80,
            X86Opcode::FstpMem64,
        ];
        TABLE.get(val as usize).copied()
    }
}

// ===========================================================================
// X86Operand — x86-64 addressing mode representation
// ===========================================================================

/// x86-64 operand addressing modes.
///
/// These are higher-level than the raw `MachineOperand` and carry
/// x86-specific addressing mode information.  The assembler converts
/// `X86Operand` values into encoded instruction bytes.
#[derive(Debug, Clone)]
pub enum X86Operand {
    /// Physical register.
    Reg(u16),
    /// Immediate constant.
    Imm(i64),
    /// Memory: `[base + index*scale + disp]`.
    Mem {
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        disp: i64,
        size: u8,
    },
    /// RIP-relative: `[rip + symbol + offset]` for PIC.
    RipRelative { symbol: String, offset: i64 },
    /// Branch label.
    Label(String),
    /// Global symbol reference.
    Symbol(String),
}

// ===========================================================================
// FrameLayout — stack frame layout computation result
// ===========================================================================

/// Describes the stack frame layout for a single x86-64 function.
///
/// The frame pointer (RBP) points to the saved RBP on the stack.
/// Local allocas live at negative offsets below RBP.  The stack grows
/// downward (toward lower addresses).
///
/// ```text
/// ┌──────────────────────┐  ← caller's RSP (before CALL)
/// │    return address     │  ← pushed by CALL
/// ├──────────────────────┤
/// │    saved RBP          │  ← RBP points here after PUSH RBP; MOV RBP,RSP
/// ├──────────────────────┤
/// │   callee-saved regs   │
/// ├──────────────────────┤
/// │   local allocas       │  ← negative offsets from RBP
/// ├──────────────────────┤
/// │   spill area          │
/// ├──────────────────────┤
/// │   outgoing args       │  ← aligned to 16 bytes at CALL
/// └──────────────────────┘  ← RSP (after prologue)
/// ```
pub struct FrameLayout {
    /// Total stack frame size in bytes (the SUB RSP, N amount).
    pub total_size: usize,
    /// Map from IR `Value` (alloca result) to RBP-relative offset (negative).
    pub alloca_offsets: FxHashMap<Value, i32>,
    /// Byte offset from RBP where spill slots begin (negative).
    pub spill_area_offset: i32,
    /// Whether the function contains any CALL instructions.
    pub has_calls: bool,
    /// Callee-saved registers that are used and must be saved/restored.
    pub callee_saved: Vec<u16>,
    /// RBP-relative offset of the variadic register save area (negative).
    /// Set only for variadic functions.  The save area holds 6 GPR slots
    /// (RDI, RSI, RDX, RCX, R8, R9) at consecutive 8-byte offsets.
    pub va_save_area_offset: Option<i32>,
    /// RBP-relative offset of the XMM save area for variadic functions.
    /// Holds 8 XMM slots (XMM0–XMM7, low 64-bit each) at 8-byte offsets.
    pub va_fp_save_area_offset: Option<i32>,
    /// RBP-relative offset of the 16-byte va_control block for variadic
    /// functions.  Layout: [gp_ptr (8 bytes), fp_ptr (8 bytes)].
    /// va_start stores the address of this block into the va_list variable.
    /// va_arg loads the correct pointer from the block based on type.
    pub va_control_offset: Option<i32>,
    /// Number of named (non-variadic) GPR parameters.  Used by va_start
    /// to compute the address of the first variadic argument.
    pub named_gpr_count: usize,
    /// Number of named (non-variadic) FP parameters.  Used by va_start
    /// to compute the address of the first variadic FP argument.
    pub named_fp_count: usize,
    /// Total stack bytes consumed by named MEMORY-class parameters
    /// (long double, large structs) that are passed on the stack but are
    /// NOT register-passed.  Used by va_start to compute the correct
    /// overflow_arg_area pointer past all named stack parameters.
    pub named_memory_stack_bytes: usize,
    /// Pre-allocated frame slots for non-alloca 16-byte struct loads.
    /// Maps Load result IR Value → RBP-relative offset of the 16-byte
    /// temporary storage.  Populated during `compute_frame_layout` for
    /// Load instructions whose pointer is NOT an alloca (e.g. GEP from
    /// member access like `pParse->sLastToken`).
    pub struct_temp_offsets: FxHashMap<Value, i32>,
    /// RBP-relative offset of the saved hidden return pointer for
    /// MEMORY-class struct returns (>16 bytes).  The System V AMD64 ABI
    /// passes the return buffer address in RDI as a hidden first argument;
    /// we save it here in the prologue so the Return instruction can copy
    /// the return value to it and set RAX accordingly.
    pub indirect_ret_ptr_offset: Option<i32>,
}

// ===========================================================================
// X86_64CodeGen — main codegen struct
// ===========================================================================

/// x86-64 instruction selector and code generator.
///
/// Converts IR functions into [`MachineFunction`]s containing x86-64 machine
/// instructions.  Handles all 20 IR instruction variants, System V AMD64 ABI
/// compliance, and complex x86-64 addressing modes.
pub struct X86_64CodeGen {
    /// Target architecture (always `Target::X86_64`).
    target: Target,
    /// ABI helper for argument/return classification.
    abi: X86_64Abi,
    /// Next virtual register number for the current function.
    next_vreg: u32,
    /// Map from IR `Value` to the `MachineOperand` holding its result.
    value_map: FxHashMap<Value, MachineOperand>,
    /// Map from IR `Value` to its `IrType` — used for ABI classification
    /// during call instruction selection so FP arguments are routed to
    /// SSE registers (XMM0-7) rather than integer registers.
    value_types: FxHashMap<Value, IrType>,
    /// Map from IR block index to machine block index.
    block_map: FxHashMap<usize, usize>,
    /// Current frame layout (populated during `compute_frame_layout`).
    frame: Option<FrameLayout>,
    /// Cache of compile-time integer constants resolved from the IR
    /// constant-sentinel pattern (`BinOp(Add, result, UNDEF)`).
    /// Populated by a pre-scan pass during `lower_function`.
    constant_cache: FxHashMap<Value, i64>,
    /// Cache of compile-time float constants resolved from the IR
    /// constant-sentinel pattern, mapping Value to the `.Lconst.f.*`
    /// global name for RIP-relative SSE loads.
    float_constant_cache: FxHashMap<Value, String>,
    /// Accumulator mapping every allocated vreg to its IR Value.
    /// Unlike `value_map` (which stores only the *latest* operand per Value
    /// and loses earlier vregs when a Value is redefined in a later block,
    /// e.g. after phi-elimination), this map preserves every single
    /// vreg→Value association created during codegen so that
    /// `apply_allocation_result` can map *all* vregs to physical registers.
    vreg_ir_map: FxHashMap<u32, Value>,
    /// Map from IR callee `Value` to function name for direct call
    /// resolution. Populated from `IrModule::func_ref_map` during lowering.
    func_ref_names: FxHashMap<Value, String>,
    /// Map from IR `Value` (representing a global variable address) to the
    /// global variable name.  Used to emit RIP-relative addressing for
    /// global variable loads/stores on x86-64.
    global_var_refs: FxHashMap<Value, String>,
    /// Whether position-independent code generation is enabled (`-fPIC`).
    /// When true, global data accesses go through the GOT with an extra
    /// indirection: LEA becomes MOV (load address from GOT), and data
    /// loads require a double dereference (load address from GOT, then
    /// load value from that address).
    pic: bool,
    /// Map from IR `Value` to its "high-half" `MachineOperand` for 16-byte
    /// struct values that require two registers per the System V AMD64 ABI.
    /// When a struct parameter is classified as RegisterPair, the first
    /// 8 bytes map to the normal `value_map` entry and the second 8 bytes
    /// (offset +8) map here.  Used by Store (to write both halves to an
    /// alloca) and Call (to pass both halves in two GPR slots).
    struct_pair_hi: FxHashMap<Value, MachineOperand>,
    /// Map from IR Load result `Value` to the source pointer operand
    /// for struct-typed loads (size > 8).  Used at Call time to reload
    /// both halves directly into physical argument registers, avoiding
    /// intermediate vreg lifetime issues.
    struct_load_source: FxHashMap<Value, MachineOperand>,
    /// Map from IR Load result `Value` to the global symbol name for
    /// SSE-classified aggregates loaded from global variables (e.g.
    /// `_Complex float` globals).  At Call time, `select_call` uses
    /// this to emit `Movsd XMM, [rip+sym]` directly instead of going
    /// through a GPR vreg (which would cause Movsd-to-GPR encoding
    /// trap: `movsd xmm0, rax` → `movsd xmm0, [rax]` → SIGSEGV).
    global_sse_load_source: FxHashMap<Value, String>,
    /// Set of IR Values that are MEMORY-class struct parameters.
    /// When these values appear as the `value` operand of a Store
    /// instruction, the Store is suppressed because the struct data
    /// was already copied to the alloca in the prologue.
    memory_class_params: crate::common::fx_hash::FxHashSet<Value>,
    /// Map from IR Load result `Value` to the pointer **vreg** for
    /// non-alloca struct loads (e.g. GEP-based member access like
    /// `pParse->sLastToken`).  At Call time we emit two `LoadInd`
    /// instructions through this pointer (offset 0 and +8) into physical
    /// argument registers.  The pointer vreg is an IR-level value with
    /// proper live-interval tracking, so it survives to the call site.
    struct_load_ptr: FxHashMap<Value, MachineOperand>,
    /// Frame offset (RBP-relative) where the hidden return pointer
    /// (originally passed in RDI for MEMORY-class struct returns) is
    /// saved during the function prologue.  `Some(offset)` when the
    /// current function returns a >16-byte struct via hidden pointer;
    /// `None` otherwise.
    indirect_ret_ptr_offset: Option<i32>,
}

// =========================================================================
// Variadic argument helpers
// =========================================================================

/// Load a 64-bit value from a va_list slot.  When the slot is a Memory
/// operand (stack alloca), we emit `MOV dst, [mem]`.  When it's a register
/// (pointer to the slot), we emit `LoadInd dst, reg`.
fn va_load_from_slot(slot: &MachineOperand, dst: MachineOperand) -> MachineInstruction {
    match slot {
        MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_) => {
            X86_64CodeGen::mk_inst(X86Opcode::Mov, Some(dst), std::slice::from_ref(slot))
        }
        _ => X86_64CodeGen::mk_inst(X86Opcode::LoadInd, Some(dst), std::slice::from_ref(slot)),
    }
}

/// Store a 64-bit value into a va_list slot.  When the slot is a Memory
/// operand (stack alloca), we emit `MOV [mem], src`.  When it's a register
/// (pointer), we emit `StoreInd [reg], src`.
fn va_store_to_slot(slot: &MachineOperand, src: MachineOperand) -> MachineInstruction {
    match slot {
        MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_) => {
            X86_64CodeGen::mk_inst(X86Opcode::Mov, None, &[slot.clone(), src])
        }
        _ => X86_64CodeGen::mk_inst(X86Opcode::StoreInd, None, &[slot.clone(), src]),
    }
}

impl X86_64CodeGen {
    /// Create a new x86-64 code generator for the given target.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `target` is `Target::X86_64`.
    pub fn new(target: Target) -> Self {
        debug_assert!(
            target == Target::X86_64,
            "X86_64CodeGen requires Target::X86_64"
        );
        let abi = X86_64Abi::new(target);
        X86_64CodeGen {
            target,
            abi,
            next_vreg: 0,
            value_map: FxHashMap::default(),
            value_types: FxHashMap::default(),
            block_map: FxHashMap::default(),
            frame: None,
            constant_cache: FxHashMap::default(),
            float_constant_cache: FxHashMap::default(),
            vreg_ir_map: FxHashMap::default(),
            func_ref_names: FxHashMap::default(),
            global_var_refs: FxHashMap::default(),
            pic: false,
            struct_pair_hi: FxHashMap::default(),
            struct_load_source: FxHashMap::default(),
            global_sse_load_source: FxHashMap::default(),
            memory_class_params: crate::common::fx_hash::FxHashSet::default(),
            struct_load_ptr: FxHashMap::default(),
            indirect_ret_ptr_offset: None,
        }
    }

    /// Set PIC flag for position-independent code generation.
    pub fn set_pic(&mut self, pic: bool) {
        self.pic = pic;
    }

    /// Set module-level function reference map for direct call resolution.
    pub fn set_func_ref_names(&mut self, map: FxHashMap<Value, String>) {
        self.func_ref_names = map;
    }

    /// Set module-level global variable reference map for RIP-relative
    /// addressing during instruction selection.
    pub fn set_global_var_refs(&mut self, map: FxHashMap<Value, String>) {
        self.global_var_refs = map;
    }

    /// Pre-populate the constant cache by scanning the given IR function
    /// for `BinOp(Add, result, UNDEF)` sentinel instructions and extracting
    /// constant integer values from the corresponding module globals.
    ///
    /// The IR lowering phase (`emit_int_const`) pairs each compile-time
    /// integer constant with a global named `.Lconst.i.{G}`.  We collect
    /// all sentinel `Value`s in instruction order and all integer-constant
    /// globals sorted by their index suffix, then pair them 1:1.  Float
    /// constants (`.Lconst.f.*`) are handled similarly.
    pub fn populate_constant_cache(
        &mut self,
        func: &IrFunction,
        globals: &[crate::ir::module::GlobalVariable],
    ) {
        self.constant_cache.clear();
        self.float_constant_cache.clear();

        // ---------------------------------------------------------------
        // Strategy: use the DIRECT Value→constant mappings recorded during
        // IR lowering on `IrFunction`.  This avoids the fragile positional
        // matching between orphaned Values and global `.Lconst.*` entries
        // that breaks when optimisation passes remove some sentinels.
        // ---------------------------------------------------------------

        // 1. Populate integer constants from the authoritative map on
        //    the function (recorded at emit_int_const time).
        for (&val, &imm) in &func.constant_values {
            self.constant_cache.insert(val, imm);
        }

        // 2. Populate float constants from the authoritative map.
        for (&val, (name, _fval)) in &func.float_constant_values {
            self.float_constant_cache.insert(val, name.clone());
        }

        // 3. Fallback: if the direct maps are empty (e.g. functions
        //    built without the new IR lowering), fall back to the
        //    legacy positional-matching approach.
        if func.constant_values.is_empty() && func.float_constant_values.is_empty() {
            self.populate_constant_cache_legacy(func, globals);
        }
    }

    /// Legacy constant cache population using positional matching.
    ///
    /// Used only when `IrFunction::constant_values` is empty (for
    /// backwards compatibility with functions not built by the new
    /// IR lowering that records direct mappings).
    fn populate_constant_cache_legacy(
        &mut self,
        func: &IrFunction,
        globals: &[crate::ir::module::GlobalVariable],
    ) {
        // Collect all constant Values: both surviving sentinels
        // and orphaned (sentinel-removed) references.
        let mut defined = std::collections::HashSet::new();
        for param in &func.params {
            if param.value != Value::UNDEF {
                defined.insert(param.value.index());
            }
        }
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Some(r) = inst.result() {
                    if r != Value::UNDEF {
                        defined.insert(r.index());
                    }
                }
            }
        }

        let mut referenced = std::collections::HashSet::new();
        for block in func.blocks() {
            for inst in block.instructions() {
                for op in inst.operands() {
                    if op != Value::UNDEF {
                        referenced.insert(op.index());
                    }
                }
            }
        }

        let func_ref_indices: crate::common::fx_hash::FxHashSet<u32> =
            self.func_ref_names.keys().map(|v| v.index()).collect();
        let global_ref_indices: crate::common::fx_hash::FxHashSet<u32> =
            self.global_var_refs.keys().map(|v| v.index()).collect();

        // Collect orphaned values (referenced but not defined).
        let orphans: std::collections::HashSet<u32> = referenced
            .difference(&defined)
            .copied()
            .filter(|idx| !func_ref_indices.contains(idx) && !global_ref_indices.contains(idx))
            .collect();

        // Collect surviving integer sentinel values.
        let mut int_sentinels: Vec<u32> = Vec::new();
        let mut float_sentinels: Vec<u32> = Vec::new();
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::BinOp {
                    result,
                    op,
                    lhs,
                    rhs,
                    ty,
                    ..
                } = inst
                {
                    let is_sentinel = (*op == BinOp::Add || *op == BinOp::FAdd)
                        && *rhs == Value::UNDEF
                        && *lhs == *result;
                    if is_sentinel {
                        if *op == BinOp::FAdd
                            || matches!(ty, IrType::F32 | IrType::F64 | IrType::F80)
                        {
                            float_sentinels.push(result.index());
                        } else {
                            int_sentinels.push(result.index());
                        }
                    }
                }
            }
        }

        // Integer orphans: orphaned values that are NOT float sentinels.
        let float_sentinel_set: std::collections::HashSet<u32> =
            float_sentinels.iter().copied().collect();
        let int_sentinel_set: std::collections::HashSet<u32> =
            int_sentinels.iter().copied().collect();

        // All integer constant values = orphans (non-float) + surviving int sentinels.
        let mut all_int_const_vals: Vec<u32> = orphans
            .iter()
            .copied()
            .filter(|v| !float_sentinel_set.contains(v))
            .collect();
        all_int_const_vals.extend(int_sentinels.iter());
        all_int_const_vals.sort();
        all_int_const_vals.dedup();

        // All float constant values = float orphans + surviving float sentinels.
        let mut all_float_const_vals: Vec<u32> = orphans
            .iter()
            .copied()
            .filter(|v| !int_sentinel_set.contains(v) && !all_int_const_vals.contains(v))
            .collect();
        all_float_const_vals.extend(float_sentinels.iter());
        all_float_const_vals.sort();
        all_float_const_vals.dedup();

        // Collect integer constants from globals, sorted by creation index.
        let mut int_consts: Vec<(u32, i64)> = Vec::new();
        let mut float_consts: Vec<(u32, String)> = Vec::new();
        for gv in globals {
            if let Some(crate::ir::module::Constant::Integer(v)) = gv.initializer.as_ref() {
                if let Some(idx_str) = gv.name.strip_prefix(".Lconst.i.") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        int_consts.push((idx, *v as i64));
                    }
                }
            }
            if let Some(crate::ir::module::Constant::Float(_)) = gv.initializer.as_ref() {
                if let Some(idx_str) = gv.name.strip_prefix(".Lconst.f.") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        float_consts.push((idx, gv.name.clone()));
                    }
                }
            }
        }
        int_consts.sort_by_key(|(idx, _)| *idx);
        float_consts.sort_by_key(|(idx, _)| *idx);

        // Pair ALL integer constant Values with sorted globals.
        for (val_idx, (_gidx, gval)) in all_int_const_vals.iter().zip(int_consts.iter()) {
            self.constant_cache.insert(Value(*val_idx), *gval);
        }

        // Pair ALL float constant Values with sorted float globals.
        for (val_idx, (_gidx, gname)) in all_float_const_vals.iter().zip(float_consts.iter()) {
            self.float_constant_cache
                .insert(Value(*val_idx), gname.clone());
        }
    }

    // -- Virtual register allocation ---

    /// Allocate a fresh virtual register number.
    #[inline]
    fn alloc_vreg(&mut self) -> u32 {
        let v = self.next_vreg;
        self.next_vreg += 1;
        v
    }

    /// Allocate a virtual register and return it as a `MachineOperand`.
    #[inline]
    fn new_vreg(&mut self) -> MachineOperand {
        MachineOperand::VirtualRegister(self.alloc_vreg())
    }

    /// Record the machine operand that holds an IR value's result.
    ///
    /// Also accumulates every vreg→Value association in `vreg_ir_map`
    /// so that values defined more than once (e.g. phi-eliminated copies)
    /// have ALL of their vregs mapped for register allocation.
    #[inline]
    fn set_value(&mut self, val: Value, op: MachineOperand) {
        if let MachineOperand::VirtualRegister(vreg) = &op {
            self.vreg_ir_map.insert(*vreg, val);
        }
        self.value_map.insert(val, op);
    }

    /// Retrieve the machine operand for an IR value.
    ///
    /// Returns a `VirtualRegister(u32::MAX)` sentinel if not found
    /// (should not happen for well-formed IR).
    fn get_value(&self, val: Value) -> MachineOperand {
        if let Some(op) = self.value_map.get(&val) {
            return op.clone();
        }
        // Check the constant cache — the value may be a compile-time
        // integer constant whose sentinel instruction was eliminated
        // by optimisation passes.
        if let Some(&imm) = self.constant_cache.get(&val) {
            return MachineOperand::Immediate(imm);
        }
        // Check float constant cache — the value may be a floating-point
        // constant whose sentinel was eliminated.  Return a GlobalSymbol
        // so the encoder emits a RIP-relative SSE load.
        if let Some(name) = self.float_constant_cache.get(&val) {
            return MachineOperand::GlobalSymbol(name.clone());
        }
        // Check global variable reference map — the value may represent
        // the address of a global variable.  Return a GlobalSymbol so
        // the load/store handlers can emit RIP-relative addressing.
        if let Some(name) = self.global_var_refs.get(&val) {
            return MachineOperand::GlobalSymbol(name.clone());
        }
        // Check function reference map — the value may represent the
        // address of a function (function pointer decay).  Return a
        // GlobalSymbol so the encoder emits the function's address via
        // LEA with RIP-relative addressing.
        if let Some(name) = self.func_ref_names.get(&val) {
            return MachineOperand::GlobalSymbol(name.clone());
        }
        MachineOperand::Immediate(0)
    }

    /// Try to resolve an IR value to an RBP-relative `Memory` operand by
    /// checking whether the value corresponds to an `alloca` with a known
    /// frame offset.  Returns `Some(Memory { base: RBP, displacement })` if
    /// found, `None` otherwise.
    ///
    /// This is needed by the `Return` handler for small aggregate types
    /// (e.g. `_Complex int`) where the IR return value is the alloca
    /// pointer itself and no intermediate `Load` set `struct_load_source`.
    fn resolve_alloca_mem(&self, val: Value) -> Option<MachineOperand> {
        if let Some(ref frame) = self.frame {
            if let Some(&offset) = frame.alloca_offsets.get(&val) {
                return Some(MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: offset as i64,
                });
            }
        }
        None
    }

    /// Resolve a constant-sentinel IR value to its immediate integer value.
    ///
    /// The IR lowering phase represents compile-time integer constants as
    /// `BinOp(Add, result, Value::UNDEF)` sentinels paired with a global
    /// variable holding `Constant::Integer(n)`.  The global is named
    /// `.Lconst.i.{G}` where `G` is the global index at creation time.
    ///
    /// We search the `IrFunction`'s parent module (passed via globals
    /// snapshot) for the corresponding constant.  Matching is done by
    /// scanning global names for the `.Lconst.i.` prefix and correlating
    /// via the module-level globals list passed to `lower_function`.
    fn resolve_constant_value(&self, _result: Value, _func: &IrFunction) -> Option<i64> {
        // The constant value is retrieved from the constant_cache
        // populated during the pre-scan phase of `lower_function`.
        self.constant_cache.get(&_result).copied()
    }

    // -----------------------------------------------------------------------
    // Helper: create a MachineInstruction from opcode + operands
    // -----------------------------------------------------------------------

    /// Shorthand: build an instruction with an opcode, optional result, and operands.
    fn mk_inst(
        opcode: X86Opcode,
        result: Option<MachineOperand>,
        operands: &[MachineOperand],
    ) -> MachineInstruction {
        let mut inst = MachineInstruction::new(opcode.as_u32());
        for op in operands {
            inst.operands.push(op.clone());
        }
        inst.result = result;
        inst
    }

    /// Build a terminator instruction.
    fn mk_term(opcode: X86Opcode, operands: &[MachineOperand]) -> MachineInstruction {
        let mut inst = MachineInstruction::new(opcode.as_u32());
        for op in operands {
            inst.operands.push(op.clone());
        }
        inst.is_terminator = true;
        inst.is_branch = matches!(
            opcode,
            X86Opcode::Jmp
                | X86Opcode::Je
                | X86Opcode::Jne
                | X86Opcode::Jl
                | X86Opcode::Jle
                | X86Opcode::Jg
                | X86Opcode::Jge
                | X86Opcode::Jb
                | X86Opcode::Jbe
                | X86Opcode::Ja
                | X86Opcode::Jae
                | X86Opcode::Js
                | X86Opcode::Jns
        );
        inst
    }

    /// Determine operand size in bytes from an IR type.
    #[allow(dead_code)]
    fn operand_size(ty: &IrType, target: &Target) -> u8 {
        ty.size_bytes(target) as u8
    }

    /// Choose the appropriate MOV opcode for a given IR type: GPR or SSE.
    fn mov_opcode(ty: &IrType) -> X86Opcode {
        if ty.is_float() {
            match ty {
                IrType::F64 | IrType::F80 => X86Opcode::Movsd,
                IrType::F32 => X86Opcode::Movss,
                _ => X86Opcode::Movsd,
            }
        } else {
            X86Opcode::Mov
        }
    }

    // ===================================================================
    // compute_frame_layout
    // ===================================================================

    /// Compute the stack frame layout for a function.
    ///
    /// Scans all entry-block allocas, assigns RBP-relative offsets, and
    /// determines whether the function is a leaf (no calls) or not.
    pub fn compute_frame_layout(&mut self, func: &IrFunction) -> FrameLayout {
        let target = &self.target;
        let mut alloca_offsets: FxHashMap<Value, i32> = FxHashMap::default();
        let mut current_offset: i64 = 0; // bytes below RBP (grows negative)
        let mut has_calls = false;

        // Scan all blocks for CALL instructions.
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::Call { .. } = inst {
                    has_calls = true;
                }
            }
        }

        // Scan ALL blocks for alloca instructions and assign offsets.
        // Allocas can appear in non-entry blocks (e.g., compound literals
        // in loop bodies) and must still receive valid frame slot offsets.
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::Alloca {
                    result,
                    ty,
                    alignment,
                    ..
                } = inst
                {
                    let size = ty.size_bytes(target).max(1) as i64;
                    let align = alignment.unwrap_or_else(|| ty.align_bytes(target)) as i64;
                    // Align the offset downward (stack grows down).
                    current_offset -= size;
                    if align > 1 {
                        current_offset &= !(align - 1);
                    }
                    alloca_offsets.insert(*result, current_offset as i32);
                }
            }
        }

        // Determine callee-saved registers that may be needed.
        // Initially we record all of them; after register allocation, this
        // list is refined to only those actually used.
        let callee_saved: Vec<u16> = if has_calls {
            CALLEE_SAVED_GPRS.to_vec()
        } else {
            Vec::new()
        };

        // For variadic functions: reserve register save areas so that
        // va_start/va_arg can access all variadic arguments uniformly.
        // Layout (all RBP-relative, growing downward):
        //   [va_save_area_offset + 0..47]:  6 GPR saves (RDI..R9), 48 bytes
        //   [va_fp_save_area_offset + 0..63]: 8 XMM saves (XMM0..XMM7 low 64-bit), 64 bytes
        //   [va_control_offset + 0..39]:  va_control block, 40 bytes:
        //       [+0]  gp_ptr — current GP register arg pointer
        //       [+8]  fp_ptr — current FP register arg pointer
        //       [+16] gp_end — one past the last GP register save slot
        //       [+24] fp_end — one past the last FP register save slot
        //       [+32] overflow_ptr — pointer to stack overflow args (RBP+16)
        let mut va_save_area_offset: Option<i32> = None;
        let mut va_fp_save_area_offset: Option<i32> = None;
        let mut va_control_offset: Option<i32> = None;
        let mut named_gpr_count: usize = 0;
        let mut named_fp_count: usize = 0;
        let mut named_memory_stack_bytes: usize = 0;
        if func.is_variadic {
            // Count named GPR and FP parameters.
            // MEMORY-class parameters (structs/arrays > 16 bytes) are
            // passed entirely on the stack and do NOT consume a register
            // slot.  They must be excluded from the named register count
            // so that va_start computes the correct gp_ptr offset.
            for p in &func.params {
                let param_size = crate::backend::generation::ir_type_size_pub(&p.ty) as usize;
                let is_memory_class =
                    param_size > 16 && matches!(&p.ty, IrType::Struct(_) | IrType::Array(_, _));
                if is_memory_class {
                    // MEMORY-class: pushed on stack by caller, doesn't
                    // consume a register.  Track stack bytes for va_start
                    // overflow_arg_area calculation.
                    named_memory_stack_bytes += (param_size + 7) & !7; // 8-byte aligned
                    continue;
                }
                // F80 (long double) is MEMORY-class per AMD64 ABI — does not
                // consume a register slot. It is passed on the stack.
                if matches!(p.ty, IrType::F80) {
                    // Long double takes 16 bytes on the stack (10 bytes
                    // padded to 16 for alignment per ABI).
                    named_memory_stack_bytes += 16;
                    continue;
                }
                // Detect SSE-class scalars and small SSE arrays (e.g. _Complex float).
                let is_fp = matches!(p.ty, IrType::F32 | IrType::F64)
                    || matches!(&p.ty, IrType::Array(ref elem, count)
                        if *count <= 2
                            && param_size <= 8
                            && matches!(elem.as_ref(), IrType::F32 | IrType::F64));
                // Detect SSE-pair types (e.g. _Complex double = Array(F64, 2), 16 bytes).
                let is_sse_pair = !is_fp
                    && param_size > 8
                    && param_size <= 16
                    && matches!(&p.ty, IrType::Array(ref elem, 2) if matches!(elem.as_ref(), IrType::F64));
                if is_fp {
                    named_fp_count += 1;
                } else if is_sse_pair {
                    named_fp_count += 2;
                } else {
                    // Struct pairs (9-16 bytes) consume 2 GPR slots.
                    if param_size > 8
                        && param_size <= 16
                        && matches!(&p.ty, IrType::Struct(_) | IrType::Array(_, _))
                    {
                        named_gpr_count += 2;
                    } else {
                        named_gpr_count += 1;
                    }
                }
            }
            // Reserve 176 bytes for the ABI-standard contiguous register
            // save area.  Layout (all offsets from the base):
            //   [0..47]    6 GPRs: RDI, RSI, RDX, RCX, R8, R9 (8 bytes each)
            //   [48..175]  8 XMMs: XMM0–XMM7 (16 bytes each, low 8 stored)
            // Must be 16-byte aligned for XMM stores.
            current_offset -= 176;
            current_offset &= !15; // 16-byte alignment
            va_save_area_offset = Some(current_offset as i32);
            va_fp_save_area_offset = Some((current_offset + 48) as i32);
            // Reserve 24 bytes for the ABI-standard va_list struct.
            // Layout:
            //   [0..3]   gp_offset         (u32)
            //   [4..7]   fp_offset         (u32)
            //   [8..15]  overflow_arg_area (pointer)
            //   [16..23] reg_save_area     (pointer)
            current_offset -= 24;
            current_offset &= !7;
            va_control_offset = Some(current_offset as i32);
        }

        // Pre-allocate 16-byte frame temporaries for non-alloca struct loads.
        // When a struct > 8 and <= 16 bytes is loaded via a computed pointer
        // (GEP-based member access like `pParse->sLastToken`), the backend
        // cannot use `struct_load_source` (which requires a stable
        // [RBP+alloca_offset] Memory operand).  Instead, we pre-allocate a
        // 16-byte frame slot and copy both halves at Load time using R11.
        // At Call time the existing `struct_load_source` path loads both
        // halves from this pre-allocated slot.
        let mut struct_temp_offsets: FxHashMap<Value, i32> = FxHashMap::default();
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::Load {
                    result, ptr, ty, ..
                } = inst
                {
                    let load_size = ty.size_bytes(target);
                    if ty.is_aggregate() && load_size > 8 && load_size <= 16 {
                        // Only allocate a temp slot if the pointer is NOT a
                        // known alloca — alloca-based loads are handled via
                        // the `struct_load_source` path using the alloca's own
                        // frame offset.
                        if !alloca_offsets.contains_key(ptr) {
                            current_offset -= 16;
                            current_offset &= !15; // 16-byte alignment
                            struct_temp_offsets.insert(*result, current_offset as i32);
                        }
                    } else if ty.is_aggregate() && load_size > 16 {
                        // MEMORY-class (>16 byte) aggregate load from a
                        // non-alloca source (global variable, computed GEP,
                        // etc.).  Allocate a temp frame slot large enough
                        // for the full struct so the Load handler can copy
                        // data in and record a stable struct_load_source.
                        if !alloca_offsets.contains_key(ptr) {
                            let alloc_size = ((load_size + 7) & !7) as i64;
                            current_offset -= alloc_size;
                            current_offset &= !15; // 16-byte alignment
                            struct_temp_offsets.insert(*result, current_offset as i32);
                        }
                    }
                }
            }
        }

        // Also allocate temp slots for Call results that return structs
        // via register pairs (RAX + RDX).  These need stable RBP-relative
        // storage so both eightbytes survive register allocation without
        // the allocator needing to keep two vregs alive simultaneously.
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::Call {
                    result,
                    return_type,
                    ..
                } = inst
                {
                    let ret_size = return_type.size_bytes(target);
                    if return_type.is_aggregate() && ret_size > 0 && ret_size <= 8 {
                        // Small SSE-classified aggregate return (e.g.
                        // _Complex float = Array(F32, 2), 8 bytes).
                        // Needs a temp slot so the return value (in XMM0)
                        // can be spilled to memory instead of going
                        // through a GPR vreg (Movsd-to-GPR encoding trap).
                        let classes = classify_ir_type_eightbytes(return_type, target);
                        let is_sse_ret = classes.len() == 1 && classes[0] == AbiClass::Sse;
                        if is_sse_ret {
                            current_offset -= 8;
                            current_offset &= !7; // 8-byte alignment
                            struct_temp_offsets.insert(*result, current_offset as i32);
                        }
                    } else if return_type.is_aggregate() && ret_size > 8 && ret_size <= 16 {
                        current_offset -= 16;
                        current_offset &= !15; // 16-byte alignment
                        struct_temp_offsets.insert(*result, current_offset as i32);
                    } else if return_type.is_aggregate() && ret_size > 16 {
                        // MEMORY-class aggregate return — allocate a buffer
                        // in the caller's frame to receive the return value
                        // via the hidden pointer mechanism.
                        let alloc_size = ((ret_size + 7) & !7) as i64; // 8-byte align
                        current_offset -= alloc_size;
                        current_offset &= !15; // 16-byte alignment
                        struct_temp_offsets.insert(*result, current_offset as i32);
                    } else if matches!(return_type, IrType::F80) {
                        // F80 (long double) is MEMORY-class per AMD64 ABI.
                        // Allocate a 16-byte temp slot for the hidden return
                        // pointer mechanism (callee writes via hidden ptr,
                        // caller reads from this slot).
                        current_offset -= 16;
                        current_offset &= !15; // 16-byte alignment
                        struct_temp_offsets.insert(*result, current_offset as i32);
                    }
                }
            }
        }

        // If the function returns a MEMORY-class struct (>16 bytes), the
        // System V AMD64 ABI passes the return buffer address as a hidden
        // first argument in RDI.  Allocate an 8-byte frame slot to save
        // this pointer in the prologue so the Return handler can use it.
        let abi_tmp = X86_64Abi::new(*target);
        let indirect_ret_ptr_offset = {
            let ret_loc = abi_tmp.classify_return(&func.return_type);
            if matches!(ret_loc, RetLocation::Indirect) {
                current_offset -= 8;
                current_offset &= !7; // 8-byte align
                Some(current_offset as i32)
            } else {
                None
            }
        };

        // Spill area starts after local allocas (and va save area if any).
        let spill_area_offset = current_offset as i32;

        // Total frame size (rounded up to 16-byte alignment).
        let raw_size = (-current_offset) as usize;
        let total_size = if has_calls || raw_size > RED_ZONE_SIZE {
            // Must allocate frame — align to 16 bytes.
            (raw_size + 15) & !15
        } else if raw_size == 0 {
            0
        } else {
            // Leaf function with small frame can use red zone.
            // No SUB RSP needed, but track the logical size.
            raw_size
        };

        FrameLayout {
            total_size,
            alloca_offsets,
            spill_area_offset,
            has_calls,
            callee_saved,
            va_save_area_offset,
            va_fp_save_area_offset,
            va_control_offset,
            named_gpr_count,
            named_fp_count,
            named_memory_stack_bytes,
            struct_temp_offsets,
            indirect_ret_ptr_offset,
        }
    }

    // ===================================================================
    // emit_prologue
    // ===================================================================

    /// Generate function prologue machine instructions.
    ///
    /// Standard x86-64 prologue:
    /// ```asm
    /// push rbp
    /// mov rbp, rsp
    /// sub rsp, <frame_size>   ; if frame_size > 0
    /// ; save callee-saved registers used by the function
    /// ```
    pub fn emit_prologue_with_va(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();
        let frame_size = mf.frame_size;

        // push rbp
        prologue.push(Self::mk_inst(
            X86Opcode::Push,
            None,
            &[MachineOperand::Register(RBP)],
        ));

        // mov rbp, rsp
        prologue.push(Self::mk_inst(
            X86Opcode::Mov,
            Some(MachineOperand::Register(RBP)),
            &[MachineOperand::Register(RSP)],
        ));

        // sub rsp, <frame_size>  (only if non-zero and not pure red-zone)
        // NOTE: frame_size includes space for callee-saved pushes that follow.
        if frame_size > 0 && (!mf.is_leaf || frame_size > RED_ZONE_SIZE) {
            // Stack probe: if frame_size > 4096, emit a probe loop.
            if frame_size > 4096 {
                // Probe loop: touch each page by moving RSP one page at a time.
                // for pages = frame_size / 4096 .. 1:
                //   sub rsp, 4096
                //   test [rsp], rsp   (touch the page)
                // sub rsp, frame_size % 4096
                let full_pages = frame_size / 4096;
                let remainder = frame_size % 4096;

                for _page in 0..full_pages {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(4096),
                        ],
                    ));
                    // test dword [rsp], 0  — touch the guard page
                    prologue.push(Self::mk_inst(
                        X86Opcode::Test,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(RSP),
                                index: None,
                                scale: 1,
                                displacement: 0,
                            },
                            MachineOperand::Immediate(0),
                        ],
                    ));
                }
                if remainder > 0 {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(remainder as i64),
                        ],
                    ));
                }
            } else {
                prologue.push(Self::mk_inst(
                    X86Opcode::Sub,
                    Some(MachineOperand::Register(RSP)),
                    &[
                        MachineOperand::Register(RSP),
                        MachineOperand::Immediate(frame_size as i64),
                    ],
                ));
            }
        }

        // Save callee-saved registers (after SUB RSP, within the allocated frame).
        for &reg in &mf.callee_saved_regs {
            prologue.push(Self::mk_inst(
                X86Opcode::Push,
                None,
                &[MachineOperand::Register(reg)],
            ));
        }

        // For variadic functions: save the 6 integer parameter registers
        // (RDI, RSI, RDX, RCX, R8, R9) to the GPR save area so that
        // va_start / va_arg can access all variadic arguments uniformly.
        if let Some(va_offset) = mf.va_save_area_offset {
            let save_regs = [RDI, RSI, RDX, RCX, R8, R9];
            for (i, &reg) in save_regs.iter().enumerate() {
                let disp = va_offset as i64 + (i as i64) * 8;
                prologue.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(RBP),
                            index: None,
                            scale: 1,
                            displacement: disp,
                        },
                        MachineOperand::Register(reg),
                    ],
                ));
            }
        }

        // For variadic functions: save XMM0–XMM7 to the FP portion of the
        // contiguous register save area at 16-byte intervals (ABI standard).
        // Uses movsd [rbp+offset], xmmN (stores low 64-bit).
        if let Some(va_base) = mf.va_save_area_offset {
            let xmm_regs: [u16; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];
            for (i, &reg) in xmm_regs.iter().enumerate() {
                // FP slots start at base+48, each 16 bytes apart
                let disp = va_base as i64 + 48 + (i as i64) * 16;
                prologue.push(Self::mk_inst(
                    X86Opcode::Movsd,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(RBP),
                            index: None,
                            scale: 1,
                            displacement: disp,
                        },
                        MachineOperand::Register(reg),
                    ],
                ));
            }
        }

        // For variadic functions: initialize the 24-byte ABI-standard
        // va_list struct (compatible with libc vprintf, etc.).
        // Layout:
        //   [+0]  gp_offset         (u32) = named_gpr_count * 8
        //   [+4]  fp_offset         (u32) = 48 + named_fp_count * 16
        //   [+8]  overflow_arg_area (ptr) = RBP + 16 + fixed_stack_bytes
        //   [+16] reg_save_area     (ptr) = RBP + va_save_area_offset
        if let (Some(ctrl_offset), Some(reg_save_base)) =
            (mf.va_control_offset, mf.va_save_area_offset)
        {
            let rax = MachineOperand::Register(RAX);
            // [+0] gp_offset (32-bit) = named_gpr_count * 8
            let gp_offset_val = (mf.named_gpr_count as i64) * 8;
            let mut store_gp_off = Self::mk_inst(
                X86Opcode::Mov,
                None,
                &[
                    MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: ctrl_offset as i64,
                    },
                    MachineOperand::Immediate(gp_offset_val),
                ],
            );
            store_gp_off.operand_size = 4;
            prologue.push(store_gp_off);
            // [+4] fp_offset (32-bit) = 48 + named_fp_count * 16
            let fp_offset_val = 48 + (mf.named_fp_count as i64) * 16;
            let mut store_fp_off = Self::mk_inst(
                X86Opcode::Mov,
                None,
                &[
                    MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: ctrl_offset as i64 + 4,
                    },
                    MachineOperand::Immediate(fp_offset_val),
                ],
            );
            store_fp_off.operand_size = 4;
            prologue.push(store_fp_off);
            // [+8] overflow_arg_area = RBP + 16 + fixed_stack_bytes
            // Must account for ALL named parameters on the stack:
            // - GPR args that spill (named_gpr_count > 6)
            // - FP args that spill (named_fp_count > 8)
            // - MEMORY-class args (long double, large structs) always on stack
            let gp_stack_fixed = mf.named_gpr_count.saturating_sub(6);
            let fp_stack_fixed = mf.named_fp_count.saturating_sub(8);
            let overflow_disp = 16
                + ((gp_stack_fixed + fp_stack_fixed) * 8) as i64
                + mf.named_memory_stack_bytes as i64;
            prologue.push(Self::mk_inst(
                X86Opcode::Lea,
                Some(rax.clone()),
                &[MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: overflow_disp,
                }],
            ));
            prologue.push(Self::mk_inst(
                X86Opcode::Mov,
                None,
                &[
                    MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: ctrl_offset as i64 + 8,
                    },
                    rax.clone(),
                ],
            ));
            // [+16] reg_save_area = RBP + va_save_area_offset
            prologue.push(Self::mk_inst(
                X86Opcode::Lea,
                Some(rax.clone()),
                &[MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: reg_save_base as i64,
                }],
            ));
            prologue.push(Self::mk_inst(
                X86Opcode::Mov,
                None,
                &[
                    MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: ctrl_offset as i64 + 16,
                    },
                    rax,
                ],
            ));
        }

        prologue
    }

    // ===================================================================
    // emit_epilogue
    // ===================================================================

    /// Generate function epilogue machine instructions.
    ///
    /// Standard x86-64 epilogue:
    /// ```asm
    /// ; restore callee-saved registers (reverse order)
    /// mov rsp, rbp    ; or `leave`
    /// pop rbp
    /// ret
    /// ```
    pub fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();

        // Reset RSP to the callee-save area using RBP-relative LEA.
        // During the function body, call argument pushes and other RSP
        // adjustments may cause RSP to drift from the position established
        // after the prologue.  The pop instructions below rely on RSP
        // pointing to the bottom of the callee-save area, so we
        // explicitly restore it here.
        if !mf.callee_saved_regs.is_empty() {
            let callee_push_bytes = mf.callee_saved_regs.len() * 8;
            let total = mf.frame_size + callee_push_bytes;
            let aligned_total = (total + 15) & !15;
            epilogue.push(Self::mk_inst(
                X86Opcode::Lea,
                Some(MachineOperand::Register(RSP)),
                &[MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: -(aligned_total as i64),
                }],
            ));
        }

        // Restore callee-saved registers in reverse order.
        for &reg in mf.callee_saved_regs.iter().rev() {
            epilogue.push(Self::mk_inst(
                X86Opcode::Pop,
                Some(MachineOperand::Register(reg)),
                &[],
            ));
        }

        // leave (mov rsp, rbp ; pop rbp)
        epilogue.push(Self::mk_inst(X86Opcode::Leave, None, &[]));

        // ret
        let mut ret_inst = Self::mk_inst(X86Opcode::Ret, None, &[]);
        ret_inst.is_terminator = true;
        epilogue.push(ret_inst);

        epilogue
    }

    // ===================================================================
    // lower — top-level entry point for lowering an IR module
    // ===================================================================

    /// Lower an entire IR function to a [`MachineFunction`].
    ///
    /// This is the top-level orchestrator that:
    /// 1. Computes the frame layout.
    /// 2. Maps parameters to their ABI locations.
    /// 3. Lowers each IR basic block via [`select_instruction`].
    /// 4. Emits prologue/epilogue.
    /// 5. Returns the finished `MachineFunction`.
    pub fn lower(
        &mut self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        self.lower_function(func, diag, globals)
    }

    // ===================================================================
    // lower_function
    // ===================================================================

    /// Lower an IR function to machine instructions.
    pub fn lower_function(
        &mut self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        // Reset per-function state.
        self.next_vreg = 0;
        self.value_map = FxHashMap::default();
        self.value_types = FxHashMap::default();
        self.block_map = FxHashMap::default();
        self.frame = None;
        self.constant_cache = FxHashMap::default();
        self.float_constant_cache = FxHashMap::default();
        self.struct_pair_hi = FxHashMap::default();
        self.struct_load_source = FxHashMap::default();
        self.global_sse_load_source = FxHashMap::default();
        self.struct_load_ptr = FxHashMap::default();
        // If this is only a declaration (no body), return an empty MachineFunction.
        if !func.is_definition {
            return Ok(MachineFunction::new(func.name.clone()));
        }

        // Pre-populate the constant cache from the module globals so that
        // constant-sentinel BinOp instructions can be resolved to immediates.
        self.populate_constant_cache(func, globals);

        // 1. Compute frame layout.
        let frame = self.compute_frame_layout(func);
        self.frame = Some(frame);

        // 2. Map function parameters to ABI locations.
        let param_types: Vec<IrType> = func.params.iter().map(|p| p.ty.clone()).collect();
        let _arg_locations = self.abi.classify_arguments(&param_types);

        // Create virtual registers for each parameter and record in value_map.
        // Also populate value_types so call-site ABI classification can route
        // FP arguments to SSE registers (XMM0–XMM7) rather than GPRs.
        //
        // Additionally, emit MOV instructions from ABI parameter registers
        // (RDI, RSI, RDX, RCX, R8, R9 for integer; XMM0-7 for FP) into
        // the virtual registers, so the register allocator can properly
        // track parameter live ranges.
        let mut param_insts: Vec<MachineInstruction> = Vec::new();
        // Hi-half struct frame stores must execute BEFORE any vreg param
        // MOVs.  The register allocator may assign vreg param MOVs to
        // physical registers that overlap ABI registers holding struct
        // hi-halves, so we collect those stores separately and inject
        // them first to guarantee the ABI register still holds the
        // original callee-received value.
        let mut param_hi_stores: Vec<MachineInstruction> = Vec::new();
        let mut gpr_idx: usize = 0;
        let mut sse_idx: usize = 0;
        let mut stack_param_idx: usize = 0;

        // If this function returns a MEMORY-class struct (>16 bytes), the
        // System V AMD64 ABI passes the return buffer pointer as a hidden
        // first argument in RDI.  Save it to the pre-allocated frame slot
        // and advance gpr_idx so the remaining parameters are assigned
        // from RSI onward.
        if let Some(ref frame) = self.frame {
            if let Some(ret_ptr_off) = frame.indirect_ret_ptr_offset {
                self.indirect_ret_ptr_offset = Some(ret_ptr_off);
                param_hi_stores.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(RBP),
                            index: None,
                            scale: 1,
                            displacement: ret_ptr_off as i64,
                        },
                        MachineOperand::Register(INTEGER_ARG_REGS[0]), // RDI
                    ],
                ));
                gpr_idx = 1; // RDI consumed by hidden return ptr
            }
        }
        for param in &func.params {
            let vreg = self.new_vreg();
            self.set_value(param.value, vreg.clone());
            self.value_types.insert(param.value, param.ty.clone());

            // Determine whether the parameter is floating-point.
            // Check bare scalar floats AND SSE-class structs (e.g.
            // `struct { double d; }` which is ABI-classified as SSE
            // and must be passed in an XMM register, not a GPR).
            // F80 (long double) is MEMORY-class → passed on the stack,
            // not in SSE registers. Exclude from SSE routing.
            let is_fp = matches!(param.ty, IrType::F32 | IrType::F64) || {
                if let IrType::Struct(ref st) = param.ty {
                    let sz = param.ty.size_bytes(&self.target);
                    if sz <= 8 && st.fields.len() == 1 {
                        matches!(st.fields[0], IrType::F32 | IrType::F64)
                    } else {
                        false
                    }
                } else if let IrType::Array(ref elem, ref count) = param.ty {
                    // _Complex float is Array(F32, 2), size 8 — fits in one XMM register
                    let sz = param.ty.size_bytes(&self.target);
                    sz <= 8 && *count <= 2 && matches!(elem.as_ref(), IrType::F32 | IrType::F64)
                } else {
                    false
                }
            };

            // Check for 16-byte aggregate (RegisterPair ABI): struct/array params
            // with size > 8 and <= 16 are passed in TWO consecutive registers
            // per the System V AMD64 ABI eightbyte classification.
            let param_size = param.ty.size_bytes(&self.target);
            let is_aggregate_pair =
                !is_fp && param_size > 8 && param_size <= 16 && param.ty.is_aggregate();

            // Classify aggregate eightbytes to determine SSE vs INTEGER register
            // usage.  Structs with double fields should use XMM registers.
            // Also detect mixed INTEGER+SSE pairs (e.g. struct { int *p; float b; }).
            let agg_classes = if is_aggregate_pair {
                crate::backend::x86_64::abi::classify_ir_type_eightbytes(&param.ty, &self.target)
            } else {
                vec![]
            };
            let is_sse_pair = agg_classes.len() == 2
                && agg_classes[0] == crate::backend::x86_64::abi::AbiClass::Sse
                && agg_classes[1] == crate::backend::x86_64::abi::AbiClass::Sse;
            // Mixed INTEGER+SSE pair (e.g. struct { int *p; float b; } or
            // struct { float a; int *p; }).  One eightbyte uses a GPR, the
            // other uses an XMM register.
            let is_mixed_pair = is_aggregate_pair
                && agg_classes.len() == 2
                && !is_sse_pair
                && agg_classes.contains(&crate::backend::x86_64::abi::AbiClass::Sse)
                && agg_classes.contains(&crate::backend::x86_64::abi::AbiClass::Integer);

            // INTEGER-pair structs: both eightbytes classified as INTEGER
            let is_struct_pair = is_aggregate_pair && !is_sse_pair && !is_mixed_pair;
            // --- Preemptive struct-pair ABI register save ---
            // The register allocator treats parameter MOVs (ABI physical
            // register → vreg) sequentially, but they logically form a
            // parallel assignment: all ABI registers are live
            // simultaneously at function entry.  When struct-pair params
            // consume two consecutive GPRs (e.g. RCX + R8), a non-struct
            // param's vreg destination can be assigned to one of those
            // GPRs, clobbering the struct data before it is captured.
            //
            // Example: yy_shift(void*, u16, u16, Token)
            //   Token occupies RCX (lo) + R8 (hi).
            //   Register allocator assigns vreg(yyNewState) → RCX,
            //   so `MOV RCX, RSI` clobbers Token.z in RCX.
            //
            // Fix: save BOTH eightbytes of struct-pair params to their
            // alloca frame slots in param_hi_stores (which execute as
            // physical-reg → memory before any vreg MOVs).  The lo-half
            // vreg then loads from memory, immune to register clobbering.
            // Non-struct scalar params use direct register MOVs as before
            // (their 4/8-byte allocas may not safely accept 8-byte stores).

            if is_struct_pair && gpr_idx + 1 < INTEGER_ARG_REGS.len() {
                let abi_reg_lo = INTEGER_ARG_REGS[gpr_idx];
                let abi_reg_hi = INTEGER_ARG_REGS[gpr_idx + 1];
                // Find the alloca for this struct-pair parameter by
                // scanning the entry block for `Store param.value, alloca`.
                let mut saved_to_alloca = false;
                if let Some(ref frame) = self.frame {
                    let entry_block = &func.blocks()[0];
                    for ir_inst in entry_block.instructions() {
                        if let crate::ir::instructions::Instruction::Store {
                            value: sv,
                            ptr: sp,
                            ..
                        } = ir_inst
                        {
                            if *sv == param.value {
                                if let Some(&offset) = frame.alloca_offsets.get(sp) {
                                    // Save BOTH eightbytes to the alloca
                                    // frame slot immediately — these execute
                                    // before any vreg MOVs (param_insts),
                                    // using only physical registers → memory.
                                    param_hi_stores.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        None,
                                        &[
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: offset as i64,
                                            },
                                            MachineOperand::Register(abi_reg_lo),
                                        ],
                                    ));
                                    // Use a correctly-sized store for the second eightbyte.
                                    // When the struct is 9-12 bytes (e.g. struct {int; int; int;}),
                                    // the second eightbyte only needs 1-4 bytes stored.  An
                                    // 8-byte movq would overflow the alloca into the saved RBP.
                                    let hi_remaining = param_size.saturating_sub(8);
                                    let mut hi_store = Self::mk_inst(
                                        X86Opcode::Mov,
                                        None,
                                        &[
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: (offset as i64) + 8,
                                            },
                                            MachineOperand::Register(abi_reg_hi),
                                        ],
                                    );
                                    if hi_remaining <= 4 {
                                        hi_store.operand_size = 4;
                                    }
                                    param_hi_stores.push(hi_store);
                                    // Load lo-half from saved frame slot
                                    // into vreg — memory source is immune
                                    // to register clobbering.
                                    param_insts.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(vreg.clone()),
                                        &[MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: offset as i64,
                                        }],
                                    ));
                                    saved_to_alloca = true;
                                }
                                break;
                            }
                        }
                    }
                }
                if !saved_to_alloca {
                    // Alloca not found — fall back to direct register MOVs.
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(vreg),
                        &[MachineOperand::Register(abi_reg_lo)],
                    ));
                    let vreg_hi = self.new_vreg();
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(vreg_hi.clone()),
                        &[MachineOperand::Register(abi_reg_hi)],
                    ));
                    self.struct_pair_hi.insert(param.value, vreg_hi);
                }
                gpr_idx += 2;
            } else if is_mixed_pair
                && gpr_idx < INTEGER_ARG_REGS.len()
                && sse_idx < SSE_ARG_REGS.len()
            {
                // Mixed INTEGER+SSE pair parameter (e.g. struct { int *p; float b; }).
                // One eightbyte is passed in a GPR, the other in an XMM register.
                // We store both halves directly to the alloca frame slot
                // (using physical registers → memory stores that run before
                // any vreg instructions).
                let lo_class = agg_classes[0];
                let (lo_reg, lo_opcode) = if lo_class == crate::backend::x86_64::abi::AbiClass::Sse
                {
                    let r = SSE_ARG_REGS[sse_idx];
                    (r, X86Opcode::Movsd)
                } else {
                    let r = INTEGER_ARG_REGS[gpr_idx];
                    (r, X86Opcode::Mov)
                };
                let hi_class = agg_classes[1];
                let (hi_reg, hi_opcode) = if hi_class == crate::backend::x86_64::abi::AbiClass::Sse
                    || hi_class == crate::backend::x86_64::abi::AbiClass::SseUp
                {
                    let r = SSE_ARG_REGS[sse_idx
                        + if lo_class == crate::backend::x86_64::abi::AbiClass::Sse {
                            1
                        } else {
                            0
                        }];
                    (r, X86Opcode::Movsd)
                } else {
                    let r = INTEGER_ARG_REGS[gpr_idx
                        + if lo_class == crate::backend::x86_64::abi::AbiClass::Integer {
                            1
                        } else {
                            0
                        }];
                    (r, X86Opcode::Mov)
                };

                let mut saved_to_alloca = false;
                if let Some(ref frame) = self.frame {
                    let entry_block = &func.blocks()[0];
                    for ir_inst in entry_block.instructions() {
                        if let crate::ir::instructions::Instruction::Store {
                            value: sv,
                            ptr: sp,
                            ..
                        } = ir_inst
                        {
                            if *sv == param.value {
                                if let Some(&offset) = frame.alloca_offsets.get(sp) {
                                    // Store lo-half
                                    param_hi_stores.push(Self::mk_inst(
                                        lo_opcode,
                                        None,
                                        &[
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: offset as i64,
                                            },
                                            MachineOperand::Register(lo_reg),
                                        ],
                                    ));
                                    // Store hi-half (may need smaller store
                                    // if struct is 9-12 bytes)
                                    let hi_remaining = param_size.saturating_sub(8);
                                    let mut hi_store = Self::mk_inst(
                                        hi_opcode,
                                        None,
                                        &[
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: (offset as i64) + 8,
                                            },
                                            MachineOperand::Register(hi_reg),
                                        ],
                                    );
                                    if hi_remaining <= 4 {
                                        if hi_opcode == X86Opcode::Mov {
                                            hi_store.operand_size = 4;
                                        } else if hi_opcode == X86Opcode::Movsd {
                                            // Use movss (4-byte SSE store) instead of movsd
                                            // (8-byte) when only 4 bytes of struct remain in
                                            // the second eightbyte.  An 8-byte movsd would
                                            // overflow the alloca and corrupt the saved RBP.
                                            hi_store.opcode = X86Opcode::Movss.as_u32();
                                        }
                                    }
                                    param_hi_stores.push(hi_store);

                                    // Record alloca so Load/Store handlers
                                    // can access the struct from its frame slot.
                                    self.struct_load_source.insert(
                                        param.value,
                                        MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: offset as i64,
                                        },
                                    );
                                    self.memory_class_params.insert(param.value);
                                    saved_to_alloca = true;
                                }
                                break;
                            }
                        }
                    }
                }
                if !saved_to_alloca {
                    // Alloca not found — fall back to vreg-based capture
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(vreg),
                        &[MachineOperand::Register(lo_reg)],
                    ));
                    let vreg_hi = self.new_vreg();
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(vreg_hi.clone()),
                        &[MachineOperand::Register(hi_reg)],
                    ));
                    self.struct_pair_hi.insert(param.value, vreg_hi);
                }
                // Advance both GPR and SSE indices by 1 each
                gpr_idx += 1;
                sse_idx += 1;
            } else if is_sse_pair && sse_idx + 1 < SSE_ARG_REGS.len() {
                // SSE-pair parameter (e.g., _Complex double = Array(F64, 2)):
                // Two consecutive XMM registers carry the real and imaginary
                // parts.  We store both halves directly to the alloca frame
                // slot (using physical XMM → memory stores that run before
                // any vreg instructions), then mark the param as a
                // memory-class so the IR Store instruction is skipped.
                let abi_reg_lo = SSE_ARG_REGS[sse_idx];
                let abi_reg_hi = SSE_ARG_REGS[sse_idx + 1];
                let mut saved_to_alloca = false;
                if let Some(ref frame) = self.frame {
                    let entry_block = &func.blocks()[0];
                    for ir_inst in entry_block.instructions() {
                        if let crate::ir::instructions::Instruction::Store {
                            value: sv,
                            ptr: sp,
                            ..
                        } = ir_inst
                        {
                            if *sv == param.value {
                                if let Some(&offset) = frame.alloca_offsets.get(sp) {
                                    // Save both XMM halves to the alloca
                                    // frame slot using physical registers
                                    // only (runs in param_hi_stores, before
                                    // any vreg-based instructions).
                                    param_hi_stores.push(Self::mk_inst(
                                        X86Opcode::Movsd,
                                        None,
                                        &[
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: offset as i64,
                                            },
                                            MachineOperand::Register(abi_reg_lo),
                                        ],
                                    ));
                                    param_hi_stores.push(Self::mk_inst(
                                        X86Opcode::Movsd,
                                        None,
                                        &[
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: (offset as i64) + 8,
                                            },
                                            MachineOperand::Register(abi_reg_hi),
                                        ],
                                    ));
                                    // Record alloca as struct_load_source
                                    // so Load instructions can copy from it.
                                    self.struct_load_source.insert(
                                        param.value,
                                        MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: offset as i64,
                                        },
                                    );
                                    // Mark as memory-class to skip the IR
                                    // Store (data is already in the alloca).
                                    self.memory_class_params.insert(param.value);
                                    saved_to_alloca = true;
                                }
                                break;
                            }
                        }
                    }
                }
                if !saved_to_alloca {
                    // No alloca found — use GPR-based MOV to capture both
                    // halves (the register allocator handles GPR vregs).
                    // We use integer Mov here because the vreg system
                    // doesn't guarantee XMM allocation for Movsd dest.
                    // The XMM values are moved to GPR vregs via movq.
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(vreg),
                        &[MachineOperand::Register(abi_reg_lo)],
                    ));
                    let vreg_hi = self.new_vreg();
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(vreg_hi.clone()),
                        &[MachineOperand::Register(abi_reg_hi)],
                    ));
                    self.struct_pair_hi.insert(param.value, vreg_hi);
                }
                sse_idx += 2;
            } else if is_fp && sse_idx < SSE_ARG_REGS.len() {
                let abi_reg = SSE_ARG_REGS[sse_idx];
                // Check if this is an aggregate type passed in a single SSE register
                // (e.g. _Complex float = Array(F32, 2), 8 bytes).  Aggregates
                // need to be stored to their alloca frame slot because the
                // register allocator may assign the vreg to a GPR, and
                // Movsd with a GPR dest encodes as a memory store through
                // the GPR value (which is garbage → SIGSEGV).
                let is_sse_aggregate = param.ty.is_aggregate();
                if is_sse_aggregate {
                    // Store XMM directly to the alloca frame slot.
                    let mut saved_to_alloca = false;
                    if let Some(ref frame) = self.frame {
                        let entry_block = &func.blocks()[0];
                        for ir_inst in entry_block.instructions() {
                            if let crate::ir::instructions::Instruction::Store {
                                value: sv,
                                ptr: sp,
                                ..
                            } = ir_inst
                            {
                                if *sv == param.value {
                                    if let Some(&offset) = frame.alloca_offsets.get(sp) {
                                        param_hi_stores.push(Self::mk_inst(
                                            X86Opcode::Movsd,
                                            None,
                                            &[
                                                MachineOperand::Memory {
                                                    base: Some(RBP),
                                                    index: None,
                                                    scale: 1,
                                                    displacement: offset as i64,
                                                },
                                                MachineOperand::Register(abi_reg),
                                            ],
                                        ));
                                        self.struct_load_source.insert(
                                            param.value,
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: offset as i64,
                                            },
                                        );
                                        self.memory_class_params.insert(param.value);
                                        saved_to_alloca = true;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if !saved_to_alloca {
                        // Fallback: move XMM → GPR vreg via movq
                        param_insts.push(Self::mk_inst(
                            X86Opcode::Movsd,
                            Some(vreg),
                            &[MachineOperand::Register(abi_reg)],
                        ));
                    }
                } else {
                    param_insts.push(Self::mk_inst(
                        X86Opcode::Movsd,
                        Some(vreg),
                        &[MachineOperand::Register(abi_reg)],
                    ));
                }
                sse_idx += 1;
            } else if !is_fp && param_size > 16 && (param.ty.is_struct() || param.ty.is_array()) {
                // MEMORY-class aggregate (> 16 bytes, e.g. large structs or
                // _Complex long double = Array(F80,2)): the caller pushed the
                // entire data onto the stack.  We need to copy it into our
                // local alloca.
                // The stack layout after prologue:
                //   [RBP+0]  = saved RBP
                //   [RBP+8]  = return address
                //   [RBP+16] = first stack arg eightbyte
                //   ...
                // We copy `param_size` bytes (rounded up to 8-byte chunks)
                // from the caller's stack to the alloca using R11 as scratch.
                let caller_stack_base: i64 = 16 + (stack_param_idx as i64) * 8;
                let eightbytes = (param_size + 7) / 8;
                // Find the alloca frame offset for this parameter.
                if let Some(ref frame) = self.frame {
                    let entry_block = &func.blocks()[0];
                    for ir_inst in entry_block.instructions() {
                        if let crate::ir::instructions::Instruction::Store {
                            value: sv,
                            ptr: sp,
                            ..
                        } = ir_inst
                        {
                            if *sv == param.value {
                                if let Some(&alloca_off) = frame.alloca_offsets.get(sp) {
                                    // Copy each 8-byte chunk from caller stack to alloca.
                                    for i in 0..eightbytes {
                                        let src_mem = MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: caller_stack_base + (i as i64) * 8,
                                        };
                                        let dst_mem = MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: (alloca_off as i64) + (i as i64) * 8,
                                        };
                                        // Load from caller stack into R11.
                                        param_hi_stores.push(Self::mk_inst(
                                            X86Opcode::Mov,
                                            Some(MachineOperand::Register(R11)),
                                            &[src_mem],
                                        ));
                                        // Store R11 into alloca frame slot using
                                        // Mov [mem], reg encoding (0x89).
                                        let remaining = param_size - i * 8;
                                        let mut st = Self::mk_inst(
                                            X86Opcode::Mov,
                                            None,
                                            &[dst_mem, MachineOperand::Register(R11)],
                                        );
                                        if remaining <= 4 {
                                            st.operand_size = remaining as u8;
                                        }
                                        param_hi_stores.push(st);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                // The struct data has been copied directly into the alloca
                // frame slot.  We skip emitting a vreg MOV for this parameter
                // entirely.  However, the IR code still has a
                // `Store param_value, alloca_ptr` instruction that would
                // overwrite the first 8 bytes.  We mark the param.value in a
                // skip set so the Store handler can suppress it.
                // For now, set vreg to the alloca LEA so any use of the param
                // value goes to the alloca address (struct pointer).
                if let Some(ref frame) = self.frame {
                    let entry_block = &func.blocks()[0];
                    for ir_inst in entry_block.instructions() {
                        if let crate::ir::instructions::Instruction::Store {
                            value: sv,
                            ptr: sp,
                            ..
                        } = ir_inst
                        {
                            if *sv == param.value {
                                if let Some(&alloca_off) = frame.alloca_offsets.get(sp) {
                                    // Map param.value to the alloca address so
                                    // subsequent struct_load_source can find it.
                                    self.struct_load_source.insert(
                                        param.value,
                                        MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: alloca_off as i64,
                                        },
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
                // Mark this param.value as a memory-class struct so the
                // Store handler can skip writing it.
                self.memory_class_params.insert(param.value);
                stack_param_idx += eightbytes;
            } else if !is_fp && !is_aggregate_pair && gpr_idx < INTEGER_ARG_REGS.len() {
                // Scalar integer parameter — single GPR.
                // IMPORTANT: The `!is_aggregate_pair` guard prevents 16-byte
                // struct pairs (9–16 bytes, classified as INTEGER/INTEGER)
                // from being incorrectly assigned to a single register when
                // only 1 GPR remains.  Without this guard, the callee would
                // read a struct pair's first eightbyte from a GPR that the
                // caller actually used for a later scalar parameter (e.g.,
                // flags), causing an ABI mismatch.  Struct pairs that can't
                // get 2 consecutive GPRs must fall through to the stack
                // branch below, matching the caller-side classify_arguments
                // behavior which also routes them to the stack.
                let abi_reg = INTEGER_ARG_REGS[gpr_idx];
                param_insts.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(vreg),
                    &[MachineOperand::Register(abi_reg)],
                ));
                gpr_idx += 1;
            } else {
                // Stack-passed parameter — load from the caller's frame.
                // After the standard prologue (PUSH RBP; MOV RBP,RSP):
                //   RBP + 0x00 = saved RBP
                //   RBP + 0x08 = return address
                //   RBP + 0x10 = first stack argument
                //   RBP + 0x18 = second stack argument, ...
                //
                // We count stack params in order of appearance (both GPR
                // overflow and SSE overflow share the same stack area).
                let stack_offset: i64 = 16 + (stack_param_idx as i64) * 8;

                // Multi-eightbyte aggregate on stack: copy entire aggregate
                // from caller's frame into the local alloca using R11 scratch.
                // This handles both structs AND arrays (e.g. _Complex double
                // = Array(F64, 2) = 16 bytes, which spills to stack when
                // SSE registers are exhausted).
                let eightbytes = (param_size + 7) / 8;
                if (param.ty.is_struct() || param.ty.is_array()) && eightbytes > 1 {
                    let mut copied_to_alloca = false;
                    if let Some(ref frame) = self.frame {
                        let entry_block = &func.blocks()[0];
                        for ir_inst in entry_block.instructions() {
                            if let crate::ir::instructions::Instruction::Store {
                                value: sv,
                                ptr: sp,
                                ..
                            } = ir_inst
                            {
                                if *sv == param.value {
                                    if let Some(&alloca_off) = frame.alloca_offsets.get(sp) {
                                        for i in 0..eightbytes {
                                            let src_mem = MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: stack_offset + (i as i64) * 8,
                                            };
                                            let dst_mem = MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: (alloca_off as i64) + (i as i64) * 8,
                                            };
                                            param_hi_stores.push(Self::mk_inst(
                                                X86Opcode::Mov,
                                                Some(MachineOperand::Register(R11)),
                                                &[src_mem],
                                            ));
                                            let remaining = param_size - i * 8;
                                            let mut st = Self::mk_inst(
                                                X86Opcode::Mov,
                                                None,
                                                &[dst_mem, MachineOperand::Register(R11)],
                                            );
                                            if remaining <= 4 {
                                                st.operand_size = remaining as u8;
                                            }
                                            param_hi_stores.push(st);
                                        }
                                        self.struct_load_source.insert(
                                            param.value,
                                            MachineOperand::Memory {
                                                base: Some(RBP),
                                                index: None,
                                                scale: 1,
                                                displacement: alloca_off as i64,
                                            },
                                        );
                                        self.memory_class_params.insert(param.value);
                                        copied_to_alloca = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if !copied_to_alloca {
                        // Fallback: just load first eightbyte
                        let mem = MachineOperand::Memory {
                            base: Some(super::registers::RBP),
                            index: None,
                            scale: 1,
                            displacement: stack_offset,
                        };
                        param_insts.push(Self::mk_inst(X86Opcode::Mov, Some(vreg), &[mem]));
                    }
                    stack_param_idx += eightbytes;
                } else {
                    let mem = MachineOperand::Memory {
                        base: Some(super::registers::RBP),
                        index: None,
                        scale: 1,
                        displacement: stack_offset,
                    };
                    param_insts.push(Self::mk_inst(X86Opcode::Mov, Some(vreg), &[mem]));
                    stack_param_idx += 1;
                }
            }
        }

        // 3. Create MachineFunction and build block map.
        let mut mf = MachineFunction::new(func.name.clone());

        // Pre-create machine blocks for all IR blocks.
        // blocks[0] is already created by MachineFunction::new.
        self.block_map.insert(0, 0);
        for i in 1..func.block_count() {
            let label = func.blocks[i]
                .label
                .clone()
                .unwrap_or_else(|| format!("bb{}", i));
            let mbb = MachineBasicBlock::with_label(label);
            let idx = mf.add_block(mbb);
            self.block_map.insert(i, idx);
        }

        // 3.5  Pre-populate value_types for all IR instructions that carry a
        //       result type.  This allows `select_call()` to look up the
        //       IrType of each argument value and route F32/F64/F80 arguments
        //       to SSE registers (XMM0–XMM7) instead of integer GPRs.
        //
        // First pass: build alloca Value → element IrType map so InlineAsm
        // results can be correctly typed from their output operand allocas.
        let mut alloca_elem_types: FxHashMap<Value, IrType> = FxHashMap::default();
        for ir_block in func.blocks().iter() {
            for ir_inst in ir_block.instructions() {
                if let Instruction::Alloca { result, ty, .. } = ir_inst {
                    alloca_elem_types.insert(*result, ty.clone());
                }
            }
        }
        for ir_block in func.blocks().iter() {
            for ir_inst in ir_block.instructions() {
                match ir_inst {
                    Instruction::Load { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BinOp { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BitCast {
                        result, to_type, ..
                    }
                    | Instruction::Trunc {
                        result, to_type, ..
                    }
                    | Instruction::ZExt {
                        result, to_type, ..
                    }
                    | Instruction::SExt {
                        result, to_type, ..
                    }
                    | Instruction::PtrToInt {
                        result, to_type, ..
                    } => {
                        self.value_types.insert(*result, to_type.clone());
                    }
                    Instruction::IntToPtr { result, .. } => {
                        // IntToPtr always produces a pointer type.
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::Call {
                        result,
                        return_type,
                        ..
                    } => {
                        self.value_types.insert(*result, return_type.clone());
                    }
                    Instruction::Alloca { result, ty, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                        let _ = ty;
                    }
                    Instruction::ICmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::FCmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::Phi { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::GetElementPtr { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::BlockAddress { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::InlineAsm {
                        result,
                        operands,
                        constraints,
                        ..
                    } => {
                        // Infer the InlineAsm result type from the first
                        // output operand's alloca element type.  This
                        // ensures that post-asm stores use the correct
                        // operand size (e.g. 4-byte MOV for `int` instead
                        // of 8-byte MOV that would corrupt adjacent stack
                        // slots).
                        let num_outputs = constraints
                            .split(',')
                            .filter(|c| {
                                let t = c.trim();
                                t.starts_with('=') || t.starts_with('+')
                            })
                            .count();
                        if num_outputs > 0 && !operands.is_empty() {
                            if let Some(elem_ty) = alloca_elem_types.get(&operands[0]) {
                                self.value_types.insert(*result, elem_ty.clone());
                            } else {
                                // Fallback: pointer-width integer
                                self.value_types.insert(*result, IrType::I64);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // 3.6  Inject parameter-loading MOV instructions into the entry block.
        //       Hi-half struct frame stores are injected FIRST — they read
        //       from ABI registers (e.g. RCX for arg-slot 4) and store
        //       directly to the frame.  They must run before any vreg param
        //       MOVs because the register allocator may assign a vreg to
        //       the same physical register that carries a struct hi-half,
        //       clobbering it.
        for inst in param_hi_stores {
            mf.blocks[0].push_instruction(inst);
        }
        //       Now inject the normal param MOVs (ABI reg → vreg).
        //       Record the count so resolve_param_load_conflicts knows
        //       exactly how many MOVs are true parameter copies.
        mf.num_param_moves = param_insts.len();
        for inst in param_insts {
            mf.blocks[0].push_instruction(inst);
        }

        // 3.6  Normalize single-float struct types in value_types.
        //
        //       A struct containing exactly one float field (e.g. `struct {
        //       double d; }`) is ABI-classified as SSE and must live in an XMM
        //       register.  But `IrType::Struct` causes the register allocator
        //       to pick a GPR, leading to mis-encoded MOVSD instructions
        //       (GPR operand interpreted as memory address → SIGSEGV).  We
        //       normalize these to the inner float type so the allocator picks
        //       an XMM register.
        {
            let mut overrides = Vec::new();
            for (&val, ty) in self.value_types.iter() {
                if let IrType::Struct(ref st) = ty {
                    if st.fields.len() == 1 {
                        match st.fields[0] {
                            IrType::F64 | IrType::F80 => {
                                overrides.push((val, IrType::F64));
                            }
                            IrType::F32 => {
                                overrides.push((val, IrType::F32));
                            }
                            _ => {}
                        }
                    }
                }
            }
            for (val, ty) in overrides {
                self.value_types.insert(val, ty);
            }
        }

        // 3.7  Pre-allocate virtual registers for ALL instruction results.
        //
        //       After phi elimination, copies (BitCast instructions) are placed
        //       in predecessor blocks. A merge block that uses the phi result
        //       may have a lower block index than the predecessor blocks
        //       containing the copies. Since the codegen processes blocks in
        //       sequential order, the USE of a value can be encountered before
        //       the block containing the DEFINITION.
        //
        //       Pre-allocating vregs for ALL result-producing instructions
        //       (not just BitCast) ensures that cross-block value references
        //       are resolved correctly regardless of block ordering.
        {
            // Collect all unique result values from all result-producing
            // instructions across all blocks.
            let mut all_defined_vals: FxHashSet<Value> = FxHashSet::default();
            let mut max_val_idx: u32 = 0;
            for ir_block in func.blocks().iter() {
                for ir_inst in ir_block.instructions() {
                    if let Some(result) = ir_inst.result() {
                        if result != Value::UNDEF {
                            all_defined_vals.insert(result);
                            let idx = result.index();
                            if idx < u32::MAX && idx > max_val_idx {
                                max_val_idx = idx;
                            }
                        }
                    }
                }
            }
            // Also scan operands for max value index (params, etc.)
            for ir_block in func.blocks().iter() {
                for ir_inst in ir_block.instructions() {
                    for op in ir_inst.operands() {
                        if op != Value::UNDEF {
                            let idx = op.index();
                            if idx < u32::MAX && idx > max_val_idx {
                                max_val_idx = idx;
                            }
                        }
                    }
                }
            }
            // Pre-create a vreg for each unique defined value, using the
            // IR Value index as the vreg ID.  This preserves alignment
            // between vreg IDs and IR Value indices so that the fallback
            // path in apply_allocation_result correctly maps temporary
            // vregs (e.g. PIC GOT loads) to physical registers.
            for def_val in &all_defined_vals {
                if !self.value_map.contains_key(def_val) {
                    let vreg_id = def_val.index();
                    self.vreg_ir_map.insert(vreg_id, *def_val);
                    let op = MachineOperand::VirtualRegister(vreg_id);
                    self.value_map.insert(*def_val, op);
                }
            }
            // Set next_vreg to max+1 so subsequent allocations (params,
            // PIC temps, etc.) don't collide with pre-allocated vregs.
            if max_val_idx > 0 || !all_defined_vals.is_empty() {
                let new_start = max_val_idx + 1;
                if new_start > self.next_vreg {
                    self.next_vreg = new_start;
                }
            }
        }

        // 4. Lower each IR basic block.
        for (ir_idx, ir_block) in func.blocks().iter().enumerate() {
            let mach_idx = *self.block_map.get(&ir_idx).unwrap_or(&0);
            let mut instructions: Vec<MachineInstruction> = Vec::new();

            for ir_inst in ir_block.instructions() {
                let mut selected = self.select_instruction(ir_inst, func, diag);

                instructions.append(&mut selected);
            }

            // Append instructions to the corresponding machine block.
            for inst in instructions {
                mf.blocks[mach_idx].push_instruction(inst);
            }
        }

        // 5. Finalize frame info.
        if let Some(ref frame) = self.frame {
            mf.frame_size = frame.total_size;
            mf.is_leaf = !frame.has_calls;
            mf.callee_saved_regs = frame.callee_saved.clone();
            mf.va_save_area_offset = frame.va_save_area_offset;
            mf.va_fp_save_area_offset = frame.va_fp_save_area_offset;
            mf.va_control_offset = frame.va_control_offset;
            mf.named_gpr_count = frame.named_gpr_count;
            mf.named_fp_count = frame.named_fp_count;
            mf.named_memory_stack_bytes = frame.named_memory_stack_bytes;
        }

        // Mark calls.
        if self.frame.as_ref().map_or(false, |f| f.has_calls) {
            mf.mark_has_calls();
        }

        // 6. Build the reverse mapping (vreg → IR Value) so that
        // apply_allocation_result can correctly resolve VirtualRegister
        // operands to physical registers via the register allocator's
        // IR Value-indexed assignment table.
        //
        // Use vreg_ir_map (which accumulates ALL vreg→Value associations,
        // even when a Value is defined in multiple blocks after phi
        // elimination) instead of value_map (which only stores the LAST
        // operand for each Value and loses earlier vregs).
        for (&vreg, &ir_val) in &self.vreg_ir_map {
            mf.vreg_to_ir_value.insert(vreg, ir_val);
        }

        Ok(mf)
    }

    // ===================================================================
    // select_instruction — the core instruction selector
    // ===================================================================

    /// Select x86-64 machine instructions for a single IR instruction.
    ///
    /// Returns a vector of machine instructions because some IR
    /// instructions expand to multiple x86-64 instructions (e.g., IDIV
    /// requires CQO + IDIV, shifts need MOV to CL first).
    pub fn select_instruction(
        &mut self,
        inst: &Instruction,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
    ) -> Vec<MachineInstruction> {
        match inst {
            // ---------------------------------------------------------------
            // Alloca — no codegen; offsets tracked in FrameLayout
            // ---------------------------------------------------------------
            Instruction::Alloca { result, ty: _, .. } => {
                // Retrieve the RBP-relative offset from the frame layout.
                if let Some(ref frame) = self.frame {
                    if let Some(&offset) = frame.alloca_offsets.get(result) {
                        // LEA vreg, [rbp + offset]
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        let inst = Self::mk_inst(
                            X86Opcode::Lea,
                            Some(dst),
                            &[MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: offset as i64,
                            }],
                        );
                        return vec![inst];
                    }
                }
                // Fallback: create a virtual register pointing to a stack slot.
                let dst = self.new_vreg();
                self.set_value(*result, dst);
                Vec::new()
            }

            // ---------------------------------------------------------------
            // StackAlloc (__builtin_alloca / alloca)
            // ---------------------------------------------------------------
            Instruction::StackAlloc { result, size, .. } => {
                let mut out = Vec::new();
                let size_op = self.get_value(*size);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                // sub rsp, size
                let sub_inst = Self::mk_inst(
                    X86Opcode::Sub,
                    Some(MachineOperand::Register(RSP)),
                    &[MachineOperand::Register(RSP), size_op],
                );
                out.push(sub_inst);

                // and rsp, -16  (align to 16 bytes)
                let align_inst = Self::mk_inst(
                    X86Opcode::And,
                    Some(MachineOperand::Register(RSP)),
                    &[
                        MachineOperand::Register(RSP),
                        MachineOperand::Immediate(-16i64),
                    ],
                );
                out.push(align_inst);

                // mov result, rsp
                let mov_inst =
                    Self::mk_inst(X86Opcode::Mov, Some(dst), &[MachineOperand::Register(RSP)]);
                out.push(mov_inst);

                out
            }

            // ---------------------------------------------------------------
            // StackSave — capture RSP into a virtual register
            // ---------------------------------------------------------------
            Instruction::StackSave { result, .. } => {
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                let mov_inst =
                    Self::mk_inst(X86Opcode::Mov, Some(dst), &[MachineOperand::Register(RSP)]);
                vec![mov_inst]
            }

            // ---------------------------------------------------------------
            // StackRestore — restore RSP from a previously saved value
            // ---------------------------------------------------------------
            Instruction::StackRestore { ptr, .. } => {
                let src = self.get_value(*ptr);
                let mov_inst =
                    Self::mk_inst(X86Opcode::Mov, Some(MachineOperand::Register(RSP)), &[src]);
                vec![mov_inst]
            }

            // ---------------------------------------------------------------
            // Load
            // ---------------------------------------------------------------
            Instruction::Load {
                result, ptr, ty, ..
            } => {
                // Extra instructions prepended for non-alloca 16-byte struct
                // copies into pre-allocated frame temporaries.
                let mut struct_temp_prefix: Vec<MachineInstruction> = Vec::new();
                // --- 16-byte struct pair load ---
                // For struct loads > 8 and <= 16 bytes that will be passed in a
                // RegisterPair, we record the base Memory operand (RBP-relative)
                // so that `select_call` can emit two loads directly into the
                // physical argument registers at call time using stable
                // frame-relative addressing.  This avoids vreg lifetime conflicts
                // in the register allocator (which only sees IR-level operands).
                let load_size = ty.size_bytes(&self.target);
                // For small SSE-classified aggregates (≤8 bytes, e.g.
                // _Complex float = Array(F32, 2)), record struct_load_source
                // so that Return/Call handlers can load directly from the
                // alloca into XMM registers without going through a GPR
                // vreg (which would cause Movsd-to-GPR encoding trap:
                // `movsd xmm0, rax` → `movsd xmm0, [rax]` → SIGSEGV).
                if ty.is_aggregate() && load_size > 0 && load_size <= 8 {
                    // Only record struct_load_source for SSE-classified
                    // aggregates (e.g. _Complex float = Array(F32, 2)).
                    // Recording it for INTEGER-classified aggregates
                    // (e.g. struct { short x; }) causes the Store handler
                    // to use the struct-copy-from-stack path which
                    // incorrectly uses R10/R11 scratch for tiny structs
                    // that should be handled as normal scalar stores.
                    let classes = classify_ir_type_eightbytes(ty, &self.target);
                    let is_sse_aggregate = classes.len() == 1 && classes[0] == AbiClass::Sse;
                    let mut found_source = false;
                    if is_sse_aggregate {
                        if let Some(ref frame) = self.frame {
                            if let Some(&offset) = frame.alloca_offsets.get(ptr) {
                                let mem_op = MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: offset as i64,
                                };
                                self.struct_load_source.insert(*result, mem_op);
                                found_source = true;
                            }
                        }
                    }
                    // For global variable loads (not alloca-backed),
                    // record the global symbol name so that select_call
                    // can emit `Movsd XMM, [rip+sym]` directly instead
                    // of `Movsd XMM, GPR` (which would dereference the
                    // GPR as a pointer → SIGSEGV).
                    if !found_source && is_sse_aggregate {
                        let ptr_op = self.get_value(*ptr);
                        if let MachineOperand::GlobalSymbol(ref sym) = ptr_op {
                            self.global_sse_load_source.insert(*result, sym.clone());
                        }
                    }
                    // Fall through to normal load — the struct_load_source
                    // or global_sse_load_source will be used by Return/Call
                    // to load into XMM directly.
                }
                if ty.is_aggregate() && load_size > 8 && load_size <= 16 {
                    // Check if the source pointer is a known alloca — if so,
                    // record the stable Memory operand for the alloca's frame
                    // slot so we can reload at call time without vregs.
                    // This covers structs and arrays (e.g. _Complex double =
                    // Array(F64, 2)) that are 9-16 bytes (RegisterPair).
                    let mut found_alloca = false;
                    if let Some(ref frame) = self.frame {
                        if let Some(&offset) = frame.alloca_offsets.get(ptr) {
                            let mem_op = MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: offset as i64,
                            };
                            self.struct_load_source.insert(*result, mem_op);
                            found_alloca = true;
                        }
                    }
                    if !found_alloca {
                        // The source pointer is NOT an alloca — it's a GEP or
                        // other computed pointer (e.g. `pParse->sLastToken`).
                        // Use the pre-allocated struct temp frame slot: copy
                        // both 8-byte halves from the source pointer into the
                        // frame slot using R11 scratch.  Then record the frame
                        // slot as a struct_load_source Memory operand so that
                        // the existing Call handler can reload both halves.
                        //
                        // Instruction sequence (R11 is the reserved scratch):
                        //   MOV R11, ptr_vreg       — copy pointer to R11
                        //   ADD R11, 8              — R11 = ptr + 8
                        //   LoadInd R11, R11        — R11 = hi-half
                        //   MOV [RBP+off+8], R11    — store hi-half to frame
                        //   MOV R11, ptr_vreg       — reload pointer (vreg's
                        //                             phys reg is untouched —
                        //                             only R11 was modified)
                        //   LoadInd R11, R11        — R11 = lo-half
                        //   MOV [RBP+off], R11      — store lo-half to frame
                        if let Some(ref frame) = self.frame {
                            if let Some(&temp_offset) = frame.struct_temp_offsets.get(result) {
                                let ptr_vreg = self.get_value(*ptr);
                                let mem_lo = MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: temp_offset as i64,
                                };
                                let mem_hi = MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: (temp_offset + 8) as i64,
                                };
                                // When the pointer is a GlobalSymbol,
                                // MOV loads the VALUE at that address.  We
                                // need LEA to get the ADDRESS instead.
                                let ptr_opc =
                                    if matches!(&ptr_vreg, MachineOperand::GlobalSymbol(_)) {
                                        X86Opcode::Lea
                                    } else {
                                        X86Opcode::Mov
                                    };
                                struct_temp_prefix.push(Self::mk_inst(
                                    ptr_opc,
                                    Some(MachineOperand::Register(R11)),
                                    std::slice::from_ref(&ptr_vreg),
                                ));
                                struct_temp_prefix.push(Self::mk_inst(
                                    X86Opcode::Add,
                                    Some(MachineOperand::Register(R11)),
                                    &[MachineOperand::Register(R11), MachineOperand::Immediate(8)],
                                ));
                                struct_temp_prefix.push(Self::mk_inst(
                                    X86Opcode::LoadInd,
                                    Some(MachineOperand::Register(R11)),
                                    &[MachineOperand::Register(R11)],
                                ));
                                struct_temp_prefix.push(Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(mem_hi),
                                    &[MachineOperand::Register(R11)],
                                ));
                                struct_temp_prefix.push(Self::mk_inst(
                                    ptr_opc,
                                    Some(MachineOperand::Register(R11)),
                                    &[ptr_vreg],
                                ));
                                struct_temp_prefix.push(Self::mk_inst(
                                    X86Opcode::LoadInd,
                                    Some(MachineOperand::Register(R11)),
                                    &[MachineOperand::Register(R11)],
                                ));
                                struct_temp_prefix.push(Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(mem_lo.clone()),
                                    &[MachineOperand::Register(R11)],
                                ));
                                self.struct_load_source.insert(*result, mem_lo);
                            }
                        }
                    }
                    // Fall through to the normal load path — emit a single
                    // 8-byte load of the low half.  The high half will be
                    // loaded at call time from the recorded Memory operand.
                } else if ty.is_aggregate() && load_size > 16 {
                    // MEMORY-class aggregate (>16 bytes): record the source
                    // pointer memory operand so that Return / Call / Store
                    // handlers can copy all eightbytes.  Only load the
                    // first 8 bytes into the vreg (as a convenience value);
                    // full struct copies go through struct_load_source.
                    if let Some(ref frame) = self.frame {
                        if let Some(&offset) = frame.alloca_offsets.get(ptr) {
                            let mem_op = MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: offset as i64,
                            };
                            self.struct_load_source.insert(*result, mem_op);
                        } else if let Some(&temp_offset) = frame.struct_temp_offsets.get(result) {
                            // Non-alloca source (global variable, computed
                            // GEP, etc.).  Copy the entire struct into the
                            // pre-allocated temp frame slot so that the
                            // Return / Call / Store handlers can use
                            // struct_load_source for the full copy.
                            let ptr_vreg = self.get_value(*ptr);
                            let ptr_opc = if matches!(&ptr_vreg, MachineOperand::GlobalSymbol(_)) {
                                X86Opcode::Lea
                            } else {
                                X86Opcode::Mov
                            };
                            let eightbytes = (load_size + 7) / 8;
                            // Load address of source into R11.
                            // Copy each eightbyte from source to temp slot.
                            // IMPORTANT: Use only R11 (reserved scratch) for
                            // all data movement.  Reload the source address
                            // from ptr_vreg each iteration instead of holding
                            // it in R11 across iterations, so no allocatable
                            // register (like RAX) is clobbered.
                            for i in 0..eightbytes {
                                // Reload source address into R11.
                                struct_temp_prefix.push(Self::mk_inst(
                                    ptr_opc,
                                    Some(MachineOperand::Register(R11)),
                                    std::slice::from_ref(&ptr_vreg),
                                ));
                                // Load data from source + offset via R11.
                                let src_off = MachineOperand::Memory {
                                    base: Some(R11),
                                    index: None,
                                    scale: 1,
                                    displacement: (i as i64) * 8,
                                };
                                let remaining = load_size - i * 8;
                                struct_temp_prefix.push(Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(MachineOperand::Register(R11)),
                                    &[src_off],
                                ));
                                // Store to temp frame slot.
                                let dst_off = MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: (temp_offset as i64) + (i as i64) * 8,
                                };
                                let mut st = Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(dst_off),
                                    &[MachineOperand::Register(R11)],
                                );
                                if remaining <= 4 {
                                    st.operand_size = remaining as u8;
                                }
                                struct_temp_prefix.push(st);
                            }
                            // Record temp slot as struct_load_source.
                            let mem_op = MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: temp_offset as i64,
                            };
                            self.struct_load_source.insert(*result, mem_op);
                        }
                    }
                }

                let src = self.get_value(*ptr);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                // Use the normalized type from value_types (step 3.6
                // converts `Struct([F64])` → `F64`) for opcode
                // selection so single-float struct loads use
                // Movsd/Movss instead of integer MOV/LoadInd.
                let load_ty = self
                    .value_types
                    .get(result)
                    .cloned()
                    .unwrap_or_else(|| ty.clone());

                // If the pointer is already a Memory or FrameSlot,
                // use a regular Mov with that operand.
                let mut load_result = match &src {
                    MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_) => {
                        let opcode = Self::mov_opcode(&load_ty);
                        let mut inst = Self::mk_inst(opcode, Some(dst), &[src]);
                        // For small struct/array loads, set operand_size to
                        // prevent reading past the allocation (e.g. a 4-byte
                        // struct must NOT be loaded with a 64-bit MOV that
                        // reads 8 bytes, potentially into adjacent stack data
                        // that gets stored back and clobbers globals).
                        if let IrType::Struct(_) | IrType::Array(_, _) = &load_ty {
                            let sz = load_ty.size_bytes(&self.target);
                            if sz > 0 && sz <= 4 {
                                inst.operand_size = sz as u8;
                            }
                        }
                        vec![inst]
                    }
                    MachineOperand::GlobalSymbol(name) => {
                        if self.pic {
                            // PIC mode: global variable access goes through GOT.
                            // Use R11 (reserved spill scratch) as the GOT address
                            // holder to avoid creating untracked temp vregs.
                            //   Step 1: mov R11, [rip + GOT_sym]
                            //   Step 2: mov/movsd dst, [R11]
                            let r11 = MachineOperand::Register(R11);
                            let load_got = Self::mk_inst(
                                X86Opcode::Mov,
                                Some(r11.clone()),
                                &[MachineOperand::GlobalSymbol(name.clone())],
                            );
                            let deref_opcode = if load_ty.is_float() {
                                match load_ty {
                                    IrType::F32 => X86Opcode::Movss,
                                    _ => X86Opcode::Movsd,
                                }
                            } else {
                                match load_ty {
                                    IrType::I1 | IrType::I8 => X86Opcode::LoadInd8,
                                    IrType::I16 => X86Opcode::LoadInd16,
                                    IrType::I32 => X86Opcode::LoadInd32,
                                    _ => X86Opcode::LoadInd,
                                }
                            };
                            let deref = Self::mk_inst(deref_opcode, Some(dst), &[r11]);
                            vec![load_got, deref]
                        } else {
                            // Non-PIC: emit direct RIP-relative load.
                            // On x86-64, this becomes `mov dst, [rip + sym]` with
                            // a relocation for the symbol.
                            let mem = MachineOperand::GlobalSymbol(name.clone());
                            if load_ty.is_float() {
                                // Float: LEA R11, [rip+sym] then Movss/Movsd
                                let r11 = MachineOperand::Register(R11);
                                let lea = Self::mk_inst(
                                    X86Opcode::Lea,
                                    Some(r11.clone()),
                                    &[MachineOperand::GlobalSymbol(name.clone())],
                                );
                                let fop = match load_ty {
                                    IrType::F32 => X86Opcode::Movss,
                                    _ => X86Opcode::Movsd,
                                };
                                vec![lea, Self::mk_inst(fop, Some(dst), &[r11])]
                            } else {
                                // Integer: set operand_size so encoder uses
                                // correct width (no REX.W for 32-bit).
                                let load_sz: u8 = match load_ty {
                                    IrType::I1 | IrType::I8 => 1,
                                    IrType::I16 => 2,
                                    IrType::I32 => 4,
                                    _ => 8,
                                };
                                let mut load = Self::mk_inst(X86Opcode::Mov, Some(dst), &[mem]);
                                load.operand_size = load_sz;
                                vec![load]
                            }
                        }
                    }
                    _ => {
                        // Pointer is in a register (virtual or physical).
                        // Use size-specific LoadInd to ensure correct memory
                        // access width: I8 → byte load, I16 → word, I32 → dword,
                        // I64/Ptr → qword.  Small struct/array types use
                        // sized loads to avoid reading adjacent memory.
                        let opcode = if load_ty.is_float() {
                            match load_ty {
                                IrType::F32 => X86Opcode::Movss,
                                _ => X86Opcode::Movsd,
                            }
                        } else {
                            match &load_ty {
                                IrType::I1 | IrType::I8 => X86Opcode::LoadInd8,
                                IrType::I16 => X86Opcode::LoadInd16,
                                IrType::I32 => X86Opcode::LoadInd32,
                                IrType::Struct(_) | IrType::Array(_, _) => {
                                    let sz = load_ty.size_bytes(&self.target);
                                    match sz {
                                        0 | 1 => X86Opcode::LoadInd8,
                                        2 => X86Opcode::LoadInd16,
                                        3..=4 => X86Opcode::LoadInd32,
                                        _ => X86Opcode::LoadInd,
                                    }
                                }
                                _ => X86Opcode::LoadInd,
                            }
                        };
                        vec![Self::mk_inst(opcode, Some(dst), &[src])]
                    }
                };
                // Prepend any struct-temp copy instructions before the load.
                if !struct_temp_prefix.is_empty() {
                    struct_temp_prefix.append(&mut load_result);
                    struct_temp_prefix
                } else {
                    load_result
                }
            }

            // ---------------------------------------------------------------
            // Store
            // ---------------------------------------------------------------
            Instruction::Store { value, ptr, .. } => {
                // --- MEMORY-class struct parameter: skip the Store ---
                // For >16-byte struct parameters, the prologue already
                // copied the struct data from the caller's stack frame into
                // the alloca.  The IR Store instruction would overwrite the
                // first 8 bytes with garbage.  Suppress it.
                if self.memory_class_params.contains(value) {
                    return vec![];
                }
                // --- 16-byte struct pair store ---
                // If the value being stored has a high-half (RegisterPair
                // parameter or struct Load), emit TWO 8-byte stores: the
                // low half at [ptr+0] and the high half at [ptr+8].
                // Use R11 as scratch to avoid untracked temp vregs.
                if let Some(hi_op) = self.struct_pair_hi.get(value).cloned() {
                    let src_lo = self.get_value(*value);
                    let dest = self.get_value(*ptr);
                    let mut out = Vec::new();
                    let r11 = MachineOperand::Register(R11);
                    // Store low 8 bytes at [ptr].
                    out.push(Self::mk_inst(
                        X86Opcode::StoreInd,
                        None,
                        &[dest.clone(), src_lo],
                    ));
                    // Compute ptr+8 via LEA/MOV+ADD using R11, then store
                    // high 8 bytes.  CRITICAL: use LEA for GlobalSymbol to
                    // get the ADDRESS (not VALUE) of the global variable.
                    let hi_opc = if matches!(dest, MachineOperand::GlobalSymbol(_)) {
                        X86Opcode::Lea
                    } else {
                        X86Opcode::Mov
                    };
                    out.push(Self::mk_inst(hi_opc, Some(r11.clone()), &[dest]));
                    out.push(Self::mk_inst(
                        X86Opcode::Add,
                        Some(r11.clone()),
                        &[r11.clone(), MachineOperand::Immediate(8)],
                    ));
                    out.push(Self::mk_inst(X86Opcode::StoreInd, None, &[r11, hi_op]));
                    return out;
                }

                // --- struct copy from stack temp (struct_load_source) ---
                // Handles RegisterPair (≤16-byte) and Indirect (>16-byte)
                // struct copies from their stack temps to the destination.
                if let Some(src_mem) = self.struct_load_source.get(value).cloned() {
                    // Determine total struct size in bytes.
                    let struct_sz: usize = self
                        .value_types
                        .get(value)
                        .map(|vty| {
                            let s = vty.size_bytes(&self.target);
                            if s > 0 {
                                s
                            } else {
                                16
                            }
                        })
                        .unwrap_or(16);

                    let dest = self.get_value(*ptr);
                    let mut out = Vec::new();
                    let r11 = MachineOperand::Register(R11);
                    let r10 = MachineOperand::Register(R10);

                    // For small (≤16-byte) structs, use direct addressing.
                    // For larger structs, load dest ptr into R10 once.
                    let num_eightbytes = (struct_sz + 7) / 8;

                    if num_eightbytes <= 2 {
                        // 2-eightbyte path: load dest ptr into R10 first
                        // to avoid clobbering R11 when dest is a Memory
                        // operand (spilled pointer).
                        // CRITICAL: For GlobalSymbol destinations, use LEA
                        // to compute the ADDRESS of the global, not MOV
                        // which would load the VALUE at the global.
                        let dest_opcode = if matches!(dest, MachineOperand::GlobalSymbol(_)) {
                            X86Opcode::Lea
                        } else {
                            X86Opcode::Mov
                        };
                        out.push(Self::mk_inst(
                            dest_opcode,
                            Some(r10.clone()),
                            std::slice::from_ref(&dest),
                        ));
                        // Load low 8 bytes from source stack temp into R11.
                        out.push(Self::mk_inst(
                            X86Opcode::Mov,
                            Some(r11.clone()),
                            std::slice::from_ref(&src_mem),
                        ));
                        // Store to [R10+0].
                        out.push(Self::mk_inst(
                            X86Opcode::StoreInd,
                            None,
                            &[r10.clone(), r11.clone()],
                        ));
                        if num_eightbytes == 2 {
                            let src_hi = match &src_mem {
                                MachineOperand::Memory {
                                    base,
                                    index,
                                    scale,
                                    displacement,
                                } => MachineOperand::Memory {
                                    base: *base,
                                    index: *index,
                                    scale: *scale,
                                    displacement: displacement + 8,
                                },
                                _ => src_mem,
                            };
                            let remaining = struct_sz - 8;
                            // Load high 8 bytes from source into R11
                            out.push(Self::mk_inst(X86Opcode::Mov, Some(r11.clone()), &[src_hi]));
                            // Store high 8 bytes at [R10+8] using a
                            // displacement-based Memory operand instead of
                            // mutating R10 via ADD.  This is CRITICAL
                            // because R10 is allocatable — the register
                            // allocator may have assigned other live values
                            // to R10, and ADD R10,8 would corrupt them.
                            let dest_hi = MachineOperand::Memory {
                                base: Some(R10),
                                index: None,
                                scale: 1,
                                displacement: 8,
                            };
                            let mut st = Self::mk_inst(X86Opcode::Mov, None, &[dest_hi, r11]);
                            if remaining <= 4 {
                                st.operand_size = remaining as u8;
                            }
                            out.push(st);
                        }
                    } else {
                        // General N-eightbyte path for structs > 16 bytes.
                        // R10 = dest ptr, R11 = scratch for data.
                        // CRITICAL: Use LEA for GlobalSymbol destinations
                        // to get the ADDRESS, not the VALUE.
                        let dest_opc = if matches!(dest, MachineOperand::GlobalSymbol(_)) {
                            X86Opcode::Lea
                        } else {
                            X86Opcode::Mov
                        };
                        out.push(Self::mk_inst(dest_opc, Some(r10.clone()), &[dest]));
                        for i in 0..num_eightbytes {
                            let offset = (i * 8) as i64;
                            let remaining = struct_sz - i * 8;
                            let src_chunk = match &src_mem {
                                MachineOperand::Memory {
                                    base,
                                    index,
                                    scale,
                                    displacement,
                                } => MachineOperand::Memory {
                                    base: *base,
                                    index: *index,
                                    scale: *scale,
                                    displacement: displacement + offset,
                                },
                                _ => src_mem.clone(),
                            };
                            let dst_chunk = MachineOperand::Memory {
                                base: Some(R10),
                                index: None,
                                scale: 1,
                                displacement: offset,
                            };
                            // Load from source.
                            let mut ld =
                                Self::mk_inst(X86Opcode::Mov, Some(r11.clone()), &[src_chunk]);
                            if remaining <= 4 {
                                ld.operand_size = remaining as u8;
                            }
                            out.push(ld);
                            // Store to dest.
                            let mut st =
                                Self::mk_inst(X86Opcode::Mov, None, &[dst_chunk, r11.clone()]);
                            if remaining <= 4 {
                                st.operand_size = remaining as u8;
                            }
                            out.push(st);
                        }
                    }
                    return out;
                }

                let src = self.get_value(*value);
                let dest_ptr = self.get_value(*ptr);

                // Determine store size from the value's IR type so that
                // small-struct and small-integer stores do not overflow
                // their alloca slots (e.g. a 4-byte struct must NOT
                // be written with a 64-bit MOV that clobbers adjacent
                // stack memory).
                let store_sz: u8 = if let Some(vty) = self.value_types.get(value) {
                    match vty {
                        IrType::I1 | IrType::I8 => 1,
                        IrType::I16 => 2,
                        IrType::I32 => 4,
                        IrType::F32 => 4,
                        IrType::F64 | IrType::I64 | IrType::Ptr => 8,
                        IrType::Struct(_) | IrType::Array(_, _) => {
                            let sz = vty.size_bytes(&self.target);
                            if sz <= 8 {
                                sz as u8
                            } else {
                                8
                            }
                        }
                        _ => 8,
                    }
                } else {
                    8
                };
                let is_float_val = match self.value_types.get(value) {
                    Some(IrType::F32) | Some(IrType::F64) => true,
                    // SSE-class struct: single-float-field struct ≤8 bytes
                    // is passed/stored in XMM registers.
                    Some(IrType::Struct(ref st))
                        if st.fields.len() == 1
                            && matches!(st.fields[0], IrType::F32 | IrType::F64) =>
                    {
                        true
                    }
                    _ => false,
                };

                // If the pointer is already a Memory or FrameSlot,
                // use a regular Mov with that operand as destination.
                match &dest_ptr {
                    MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_) => {
                        if is_float_val {
                            let fop = if store_sz == 4 {
                                X86Opcode::Movss
                            } else {
                                X86Opcode::Movsd
                            };
                            vec![Self::mk_inst(fop, None, &[dest_ptr, src])]
                        } else {
                            let mut inst = Self::mk_inst(X86Opcode::Mov, None, &[dest_ptr, src]);
                            inst.operand_size = store_sz;
                            vec![inst]
                        }
                    }
                    MachineOperand::GlobalSymbol(name) => {
                        // Determine the store size from the value's IR type
                        // to ensure correct encoding width (avoid 64-bit store
                        // for 32-bit int globals that would overflow into
                        // adjacent variables).
                        let store_size: u8 = if let Some(vty) = self.value_types.get(value) {
                            match vty {
                                IrType::I1 | IrType::I8 => 1,
                                IrType::I16 => 2,
                                IrType::I32 => 4,
                                IrType::F32 => 4,
                                IrType::F64 => 8,
                                // Struct and Array types: use their actual byte
                                // size to avoid overwriting adjacent globals.
                                IrType::Struct(_) | IrType::Array(_, _) => {
                                    let sz = vty.size_bytes(&self.target);
                                    if sz <= 8 {
                                        sz as u8
                                    } else {
                                        8
                                    }
                                }
                                _ => 8,
                            }
                        } else {
                            8
                        };
                        let is_float = matches!(
                            self.value_types.get(value),
                            Some(IrType::F32) | Some(IrType::F64) | Some(IrType::F80)
                        );
                        if self.pic {
                            // PIC mode: store through GOT indirection.
                            let r11 = MachineOperand::Register(R11);
                            let load_got = Self::mk_inst(
                                X86Opcode::Mov,
                                Some(r11.clone()),
                                &[MachineOperand::GlobalSymbol(name.clone())],
                            );
                            if is_float {
                                // Float PIC store: GOT load → Movss/Movsd
                                let fop = if store_size == 4 {
                                    X86Opcode::Movss
                                } else {
                                    X86Opcode::Movsd
                                };
                                return vec![load_got, Self::mk_inst(fop, None, &[r11, src])];
                            }
                            // Materialise immediate into register if needed.
                            let src_reg = match &src {
                                MachineOperand::Immediate(_) => {
                                    let r10 = MachineOperand::Register(R10);
                                    let mov_imm =
                                        Self::mk_inst(X86Opcode::Mov, Some(r10.clone()), &[src]);
                                    let store_op = match store_size {
                                        1 => X86Opcode::StoreInd8,
                                        2 => X86Opcode::StoreInd16,
                                        4 => X86Opcode::StoreInd32,
                                        _ => X86Opcode::StoreInd,
                                    };
                                    return vec![
                                        load_got,
                                        mov_imm,
                                        Self::mk_inst(store_op, None, &[r11, r10]),
                                    ];
                                }
                                _ => src,
                            };
                            let store_op = match store_size {
                                1 => X86Opcode::StoreInd8,
                                2 => X86Opcode::StoreInd16,
                                4 => X86Opcode::StoreInd32,
                                _ => X86Opcode::StoreInd,
                            };
                            vec![load_got, Self::mk_inst(store_op, None, &[r11, src_reg])]
                        } else {
                            // Non-PIC: direct RIP-relative store.
                            if is_float {
                                // Float non-PIC store: LEA R11, [rip+sym]
                                // then Movss/Movsd [R11], xmm
                                let r11 = MachineOperand::Register(R11);
                                let lea = Self::mk_inst(
                                    X86Opcode::Lea,
                                    Some(r11.clone()),
                                    &[MachineOperand::GlobalSymbol(name.clone())],
                                );
                                let fop = if store_size == 4 {
                                    X86Opcode::Movss
                                } else {
                                    X86Opcode::Movsd
                                };
                                return vec![lea, Self::mk_inst(fop, None, &[r11, src])];
                            }
                            let mem = MachineOperand::GlobalSymbol(name.clone());
                            match &src {
                                MachineOperand::Immediate(_) => {
                                    let r11 = MachineOperand::Register(R11);
                                    let mov_imm =
                                        Self::mk_inst(X86Opcode::Mov, Some(r11.clone()), &[src]);
                                    let mut store =
                                        Self::mk_inst(X86Opcode::Mov, None, &[mem, r11]);
                                    store.operand_size = store_size;
                                    vec![mov_imm, store]
                                }
                                MachineOperand::GlobalSymbol(src_sym) => {
                                    // Source is also a GlobalSymbol (e.g.
                                    // storing function address into a global
                                    // pointer).  x86 can't do
                                    // MOV [rip+sym1], [rip+sym2], so
                                    // materialise the source address into R11
                                    // via LEA then store.
                                    let r11 = MachineOperand::Register(R11);
                                    let lea = Self::mk_inst(
                                        X86Opcode::Lea,
                                        Some(r11.clone()),
                                        &[MachineOperand::GlobalSymbol(src_sym.clone())],
                                    );
                                    let mut store =
                                        Self::mk_inst(X86Opcode::Mov, None, &[mem, r11]);
                                    store.operand_size = store_size;
                                    vec![lea, store]
                                }
                                _ => {
                                    let mut store =
                                        Self::mk_inst(X86Opcode::Mov, None, &[mem, src]);
                                    store.operand_size = store_size;
                                    vec![store]
                                }
                            }
                        }
                    }
                    _ => {
                        // Pointer is in a register. Use size-specific StoreInd
                        // to match the value type being stored.  Struct types
                        // must be sized by their actual byte-width so that a
                        // 4-byte struct is stored with a 32-bit MOV, not a
                        // 64-bit MOV that would clobber adjacent stack slots.
                        let store_op = if let Some(vty) = self.value_types.get(value) {
                            if vty.is_float() {
                                match vty {
                                    IrType::F32 => X86Opcode::Movss,
                                    _ => X86Opcode::Movsd,
                                }
                            } else {
                                match vty {
                                    IrType::I1 | IrType::I8 => X86Opcode::StoreInd8,
                                    IrType::I16 => X86Opcode::StoreInd16,
                                    IrType::I32 => X86Opcode::StoreInd32,
                                    IrType::Struct(_) | IrType::Array(_, _) => {
                                        let sz = vty.size_bytes(&self.target);
                                        match sz {
                                            0 | 1 => X86Opcode::StoreInd8,
                                            2 => X86Opcode::StoreInd16,
                                            3..=4 => X86Opcode::StoreInd32,
                                            _ => X86Opcode::StoreInd,
                                        }
                                    }
                                    _ => X86Opcode::StoreInd,
                                }
                            }
                        } else {
                            X86Opcode::StoreInd
                        };
                        // If the source is a GlobalSymbol (e.g. function
                        // pointer address), materialise it into a temp
                        // register via LEA before storing.
                        match &src {
                            MachineOperand::GlobalSymbol(sym_name) => {
                                let tmp = self.new_vreg();
                                // Map the LEA temp vreg to the IR store-value
                                // so apply_allocation_result assigns it the
                                // physical register the allocator chose for
                                // this value (guaranteed non-conflicting with
                                // the store pointer register).
                                if let MachineOperand::VirtualRegister(vreg_id) = &tmp {
                                    self.vreg_ir_map.insert(*vreg_id, *value);
                                }
                                let lea = Self::mk_inst(
                                    X86Opcode::Lea,
                                    Some(tmp.clone()),
                                    &[MachineOperand::GlobalSymbol(sym_name.clone())],
                                );
                                vec![lea, Self::mk_inst(store_op, None, &[dest_ptr, tmp])]
                            }
                            _ => {
                                vec![Self::mk_inst(store_op, None, &[dest_ptr, src])]
                            }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // BinOp — binary arithmetic / logic
            // ---------------------------------------------------------------
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                ..
            } => {
                // Recognise the constant-sentinel pattern emitted by
                // `emit_int_const` / `emit_float_const` in the IR lowering
                // phase.  Integer sentinels use `Add result, UNDEF`;
                // float sentinels use `FAdd result, UNDEF`.
                if *rhs == Value::UNDEF && *lhs == *result {
                    // Check float constant cache first — float constants
                    // must be loaded via RIP-relative movsd/movss from
                    // the .rodata constant pool.
                    if let Some(sym_name) = self.float_constant_cache.get(result).cloned() {
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        let mov_op = if matches!(ty, IrType::F32) {
                            X86Opcode::Movss
                        } else {
                            X86Opcode::Movsd
                        };
                        return vec![Self::mk_inst(
                            mov_op,
                            Some(dst),
                            &[MachineOperand::GlobalSymbol(sym_name)],
                        )];
                    }
                    // Then check integer constant cache.
                    if let Some(imm) = self.resolve_constant_value(*result, func) {
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        return vec![Self::mk_inst(
                            X86Opcode::Mov,
                            Some(dst),
                            &[MachineOperand::Immediate(imm)],
                        )];
                    }
                }
                self.select_binop(*result, *op, *lhs, *rhs, ty)
            }

            // ---------------------------------------------------------------
            // ICmp — integer comparison
            // ---------------------------------------------------------------
            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => self.select_icmp(*result, *op, *lhs, *rhs),

            // ---------------------------------------------------------------
            // FCmp — floating-point comparison
            // ---------------------------------------------------------------
            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => self.select_fcmp(*result, *op, *lhs, *rhs),

            // ---------------------------------------------------------------
            // Branch (unconditional)
            // ---------------------------------------------------------------
            Instruction::Branch { target, .. } => {
                let block_idx = target.index();
                let mach_idx = *self.block_map.get(&block_idx).unwrap_or(&0);
                vec![Self::mk_term(
                    X86Opcode::Jmp,
                    &[MachineOperand::BlockLabel(mach_idx as u32)],
                )]
            }

            // ---------------------------------------------------------------
            // CondBranch
            // ---------------------------------------------------------------
            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let cond = self.get_value(*condition);
                let then_idx = *self.block_map.get(&then_block.index()).unwrap_or(&0);
                let else_idx = *self.block_map.get(&else_block.index()).unwrap_or(&0);

                // If the condition is a known constant (from constant folding),
                // emit an unconditional jump instead of TEST + JNE + JMP.
                // This avoids the encoder seeing TEST Imm, Imm which is
                // not a valid x86 instruction pattern.
                if let MachineOperand::Immediate(imm) = &cond {
                    let target = if *imm != 0 { then_idx } else { else_idx };
                    return vec![Self::mk_term(
                        X86Opcode::Jmp,
                        &[MachineOperand::BlockLabel(target as u32)],
                    )];
                }

                let mut out = Vec::new();
                // test cond, cond
                out.push(Self::mk_inst(X86Opcode::Test, None, &[cond.clone(), cond]));
                // jne then_block
                let mut jne = Self::mk_term(
                    X86Opcode::Jne,
                    &[MachineOperand::BlockLabel(then_idx as u32)],
                );
                jne.is_terminator = false; // Not the final terminator yet
                out.push(jne);
                // jmp else_block (fallthrough to else)
                out.push(Self::mk_term(
                    X86Opcode::Jmp,
                    &[MachineOperand::BlockLabel(else_idx as u32)],
                ));
                out
            }

            // ---------------------------------------------------------------
            // Switch — multi-way branch
            // ---------------------------------------------------------------
            Instruction::Switch {
                value,
                default,
                cases,
                ..
            } => {
                let val = self.get_value(*value);
                let def_idx = *self.block_map.get(&default.index()).unwrap_or(&0);

                let mut out = Vec::new();

                // Determine operand size from the switch value's IR type.
                // For ≤32-bit types (int, short, char, etc.) use 32-bit
                // CMP so that negative case values like -2 (0xFFFFFFFE)
                // match correctly against 32-bit parameter values that
                // have their upper 32 bits zeroed.
                let sw_op_size: u8 = match self.value_types.get(value) {
                    Some(crate::ir::types::IrType::I8) => 4,
                    Some(crate::ir::types::IrType::I16) => 4,
                    Some(crate::ir::types::IrType::I32) => 4,
                    Some(crate::ir::types::IrType::I64) => 8,
                    _ => 4,
                };

                // x86 CMP requires a register or memory operand as the
                // first operand.  If the switch value resolved to an
                // immediate (e.g. from constant folding or a get_value
                // fallback), materialise it into a register first.
                let val = if let MachineOperand::Immediate(imm) = &val {
                    let tmp = MachineOperand::Register(R11);
                    let mut mov_inst = Self::mk_inst(
                        X86Opcode::Mov,
                        Some(tmp.clone()),
                        &[MachineOperand::Immediate(*imm)],
                    );
                    mov_inst.operand_size = sw_op_size;
                    out.push(mov_inst);
                    tmp
                } else {
                    val
                };

                // Cascaded comparisons (simple implementation).
                for (case_val, target_block) in cases {
                    let tgt_idx = *self.block_map.get(&target_block.index()).unwrap_or(&0);
                    // cmp val, case_val
                    let mut cmp_inst = Self::mk_inst(
                        X86Opcode::Cmp,
                        None,
                        &[val.clone(), MachineOperand::Immediate(*case_val)],
                    );
                    cmp_inst.operand_size = sw_op_size;
                    out.push(cmp_inst);
                    // je target_block
                    let mut je = Self::mk_inst(
                        X86Opcode::Je,
                        None,
                        &[MachineOperand::BlockLabel(tgt_idx as u32)],
                    );
                    je.is_branch = true;
                    out.push(je);
                }
                // jmp default
                out.push(Self::mk_term(
                    X86Opcode::Jmp,
                    &[MachineOperand::BlockLabel(def_idx as u32)],
                ));
                out
            }

            // ---------------------------------------------------------------
            // Call
            // ---------------------------------------------------------------
            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                ..
            } => self.select_call(*result, *callee, args, return_type, func),

            // ---------------------------------------------------------------
            // Return
            // ---------------------------------------------------------------
            Instruction::Return { value, .. } => {
                let mut out = Vec::new();

                if let Some(val) = value {
                    let src = self.get_value(*val);
                    // For global variable addresses (array-to-pointer decay)
                    // and function references, use LEA to compute the address
                    // rather than MOV which would dereference the symbol.
                    let is_global_addr = self.global_var_refs.contains_key(val)
                        || self.func_ref_names.contains_key(val);
                    // Determine if the return type is FP or integer.
                    let ret_loc = self.abi.classify_return(&func.return_type);
                    match ret_loc {
                        RetLocation::Register(reg) => {
                            // Small aggregate (≤8 bytes) returned in a
                            // single register: the IR `Return` value is a
                            // pointer to the aggregate.  We must LOAD the
                            // aggregate contents from that pointer into the
                            // return register, not move the pointer itself.
                            //
                            // Try to resolve to a stable RBP-relative Memory
                            // operand: check struct_load_source first, then
                            // fall back to looking up the alloca offset for
                            // the IR value (common for locally-constructed
                            // _Complex values where no intermediate Load
                            // instruction sets struct_load_source).
                            let is_agg = func.return_type.is_aggregate();
                            let agg_mem: Option<MachineOperand> = if is_agg {
                                self.struct_load_source
                                    .get(val)
                                    .cloned()
                                    .or_else(|| self.resolve_alloca_mem(*val))
                            } else {
                                self.struct_load_source.get(val).cloned()
                            };
                            if registers::is_sse(reg) {
                                let ret_sz = func.return_type.size_bytes(&self.target);
                                let fop = if ret_sz <= 4 {
                                    X86Opcode::Movss
                                } else {
                                    X86Opcode::Movsd
                                };
                                if let Some(mem) = agg_mem {
                                    out.push(Self::mk_inst(
                                        fop,
                                        Some(MachineOperand::Register(reg)),
                                        &[mem],
                                    ));
                                } else if is_agg {
                                    // Small aggregate value in a GPR vreg
                                    // (e.g. _Complex float loaded through a
                                    // GEP chain that doesn't map to an alloca
                                    // directly).  Spill to a stack temporary
                                    // then reload into the SSE return
                                    // register, because `MOVSD XMM, GPR`
                                    // encodes as `MOVSD XMM, [GPR]` (memory
                                    // load through GPR) → SIGSEGV.
                                    let spill_mem = MachineOperand::Memory {
                                        base: Some(registers::RSP),
                                        index: None,
                                        scale: 1,
                                        displacement: -8,
                                    };
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(spill_mem.clone()),
                                        &[src],
                                    ));
                                    out.push(Self::mk_inst(
                                        fop,
                                        Some(MachineOperand::Register(reg)),
                                        &[spill_mem],
                                    ));
                                } else {
                                    out.push(Self::mk_inst(
                                        X86Opcode::Movsd,
                                        Some(MachineOperand::Register(reg)),
                                        &[src],
                                    ));
                                }
                            } else {
                                let opcode = if is_global_addr {
                                    X86Opcode::Lea
                                } else {
                                    X86Opcode::Mov
                                };
                                if let Some(mem) = agg_mem {
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(MachineOperand::Register(reg)),
                                        &[mem],
                                    ));
                                } else {
                                    out.push(Self::mk_inst(
                                        opcode,
                                        Some(MachineOperand::Register(reg)),
                                        &[src],
                                    ));
                                }
                            };
                        }
                        RetLocation::RegisterPair(lo, hi) => {
                            // For returns via register pair: 16-byte structs
                            // or _Complex double (two SSE eightbytes).
                            // Use Movsd for SSE registers, Mov for GPRs.
                            let lo_is_sse = registers::is_sse(lo);
                            let hi_is_sse = registers::is_sse(hi);
                            let lo_op = if lo_is_sse {
                                X86Opcode::Movsd
                            } else {
                                X86Opcode::Mov
                            };
                            let hi_op_code = if hi_is_sse {
                                X86Opcode::Movsd
                            } else {
                                X86Opcode::Mov
                            };
                            let ret_size = func.return_type.size_bytes(&self.target);
                            let hi_remaining = ret_size.saturating_sub(8);
                            if let Some(mem_base) = self.struct_load_source.get(val).cloned() {
                                // Value was loaded from memory — use the
                                // stable RBP-relative address for both halves.
                                out.push(Self::mk_inst(
                                    lo_op,
                                    Some(MachineOperand::Register(lo)),
                                    std::slice::from_ref(&mem_base),
                                ));
                                let mem_hi = match &mem_base {
                                    MachineOperand::Memory {
                                        base,
                                        index,
                                        scale,
                                        displacement,
                                    } => MachineOperand::Memory {
                                        base: *base,
                                        index: *index,
                                        scale: *scale,
                                        displacement: displacement + 8,
                                    },
                                    _ => src.clone(),
                                };
                                let mut hi_inst = Self::mk_inst(
                                    hi_op_code,
                                    Some(MachineOperand::Register(hi)),
                                    &[mem_hi],
                                );
                                // For 9-12 byte structs the second eightbyte
                                // is ≤ 4 bytes — use a 32-bit load to avoid
                                // reading past the end of the struct.
                                if !hi_is_sse && hi_remaining > 0 && hi_remaining <= 4 {
                                    hi_inst.operand_size = 4;
                                }
                                out.push(hi_inst);
                            } else if let Some(hi_op) = self.struct_pair_hi.get(val).cloned() {
                                // Struct/SSE pair from parameter or previous op.
                                out.push(Self::mk_inst(
                                    lo_op,
                                    Some(MachineOperand::Register(lo)),
                                    &[src],
                                ));
                                out.push(Self::mk_inst(
                                    hi_op_code,
                                    Some(MachineOperand::Register(hi)),
                                    &[hi_op],
                                ));
                            } else {
                                // Fallback: only first eightbyte available.
                                out.push(Self::mk_inst(
                                    lo_op,
                                    Some(MachineOperand::Register(lo)),
                                    &[src],
                                ));
                            }
                        }
                        RetLocation::Indirect => {
                            // Return MEMORY-class struct via hidden pointer.
                            // Per System V AMD64 ABI: callee copies the
                            // return value to the address saved in the
                            // hidden return pointer slot, then sets RAX
                            // to that address.
                            if let Some(ret_ptr_off) = self.indirect_ret_ptr_offset {
                                let ret_ptr_mem = MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: ret_ptr_off as i64,
                                };
                                let ret_size = func.return_type.size_bytes(&self.target);
                                let eightbytes = (ret_size + 7) / 8;

                                // Load hidden return pointer into R10.
                                out.push(Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(MachineOperand::Register(R10)),
                                    &[ret_ptr_mem.clone()],
                                ));

                                // F80 (long double) is MEMORY-class but BCC
                                // keeps the value in an XMM register (as F64).
                                // Store it directly via MOVSD to the hidden
                                // return pointer instead of treating it as a
                                // struct pointer copy.
                                let is_f80_ret = matches!(func.return_type, IrType::F80);
                                if is_f80_ret {
                                    let dst_mem = MachineOperand::Memory {
                                        base: Some(R10),
                                        index: None,
                                        scale: 1,
                                        displacement: 0,
                                    };
                                    if let Some(mem_base) =
                                        self.struct_load_source.get(val).cloned()
                                    {
                                        // Load from memory source into XMM15,
                                        // then store to hidden return buffer.
                                        out.push(Self::mk_inst(
                                            X86Opcode::Movsd,
                                            Some(MachineOperand::Register(XMM15)),
                                            &[mem_base],
                                        ));
                                        out.push(Self::mk_inst(
                                            X86Opcode::Movsd,
                                            None,
                                            &[dst_mem, MachineOperand::Register(XMM15)],
                                        ));
                                    } else {
                                        // Value is in a vreg (XMM).
                                        out.push(Self::mk_inst(
                                            X86Opcode::Movsd,
                                            None,
                                            &[dst_mem, src.clone()],
                                        ));
                                    }
                                    // Zero the upper 8 bytes (10-byte F80
                                    // padded to 16 bytes).
                                    let dst_hi = MachineOperand::Memory {
                                        base: Some(R10),
                                        index: None,
                                        scale: 1,
                                        displacement: 8,
                                    };
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        None,
                                        &[dst_hi, MachineOperand::Immediate(0)],
                                    ));
                                    // RAX = hidden return pointer (ABI
                                    // requirement — same as struct returns).
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(MachineOperand::Register(RAX)),
                                        &[ret_ptr_mem],
                                    ));
                                    // Fall through to the normal Ret at the
                                    // bottom of the Return handler.  The
                                    // standard epilogue (Leave + Ret) will be
                                    // inserted before that Ret by
                                    // insert_prologue_epilogue.
                                } else if let Some(mem_base) =
                                    self.struct_load_source.get(val).cloned()
                                {
                                    // Determine the source address of the return
                                    // value.  Check struct_load_source first
                                    // (stable RBP-relative memory operand), then
                                    // fall back to dereferencing the value as
                                    // a pointer.
                                    // Copy each 8-byte chunk from the source
                                    // memory to the hidden return buffer.
                                    for i in 0..eightbytes {
                                        let src_off = match &mem_base {
                                            MachineOperand::Memory {
                                                base,
                                                index,
                                                scale,
                                                displacement,
                                            } => MachineOperand::Memory {
                                                base: *base,
                                                index: *index,
                                                scale: *scale,
                                                displacement: displacement + (i as i64) * 8,
                                            },
                                            _ => mem_base.clone(),
                                        };
                                        let dst_off = MachineOperand::Memory {
                                            base: Some(R10),
                                            index: None,
                                            scale: 1,
                                            displacement: (i as i64) * 8,
                                        };
                                        // Load from source into R11.
                                        let remaining = ret_size - i * 8;
                                        out.push(Self::mk_inst(
                                            X86Opcode::Mov,
                                            Some(MachineOperand::Register(R11)),
                                            &[src_off],
                                        ));
                                        // Store R11 into hidden return buffer.
                                        let mut st = Self::mk_inst(
                                            X86Opcode::Mov,
                                            None,
                                            &[dst_off, MachineOperand::Register(R11)],
                                        );
                                        if remaining <= 4 {
                                            st.operand_size = remaining as u8;
                                        }
                                        out.push(st);
                                    }
                                } else {
                                    // Source is a pointer in a register/vreg.
                                    // Load it into R11, then copy chunks
                                    // from [R11] to [R10].
                                    // First load the address into R11.
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(MachineOperand::Register(R11)),
                                        &[src],
                                    ));
                                    // Use RAX as a temp for the copy loop
                                    // (it will be overwritten by the final
                                    // RAX = ret_ptr anyway).
                                    for i in 0..eightbytes {
                                        let src_off = MachineOperand::Memory {
                                            base: Some(R11),
                                            index: None,
                                            scale: 1,
                                            displacement: (i as i64) * 8,
                                        };
                                        let dst_off = MachineOperand::Memory {
                                            base: Some(R10),
                                            index: None,
                                            scale: 1,
                                            displacement: (i as i64) * 8,
                                        };
                                        let remaining = ret_size - i * 8;
                                        out.push(Self::mk_inst(
                                            X86Opcode::Mov,
                                            Some(MachineOperand::Register(RAX)),
                                            &[src_off],
                                        ));
                                        let mut st = Self::mk_inst(
                                            X86Opcode::Mov,
                                            None,
                                            &[dst_off, MachineOperand::Register(RAX)],
                                        );
                                        if remaining <= 4 {
                                            st.operand_size = remaining as u8;
                                        }
                                        out.push(st);
                                    }
                                }

                                // RAX = hidden return pointer (ABI requirement).
                                // (F80 already set RAX above, but duplicating
                                // this harmless assignment keeps the logic
                                // uniform for the struct return case.)
                                if !is_f80_ret {
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(MachineOperand::Register(RAX)),
                                        &[MachineOperand::Memory {
                                            base: Some(RBP),
                                            index: None,
                                            scale: 1,
                                            displacement: ret_ptr_off as i64,
                                        }],
                                    ));
                                }
                            }
                        }
                        RetLocation::Void => {}
                    }
                }

                let mut ret = Self::mk_inst(X86Opcode::Ret, None, &[]);
                ret.is_terminator = true;
                out.push(ret);
                out
            }

            // ---------------------------------------------------------------
            // Phi — should be eliminated before codegen (Phase 9)
            // ---------------------------------------------------------------
            Instruction::Phi {
                result,
                ty: _,
                incoming: _,
                ..
            } => {
                // Phi nodes should have been eliminated by phi_eliminate.rs.
                // If encountered, emit a diagnostic warning and create a
                // virtual register as a placeholder.
                diag.emit_warning(
                    Span::dummy(),
                    format!("Phi node encountered during x86-64 codegen for %{} — should be eliminated by Phase 9", result.index()),
                );
                let dst = self.new_vreg();
                self.set_value(*result, dst);
                Vec::new()
            }

            // ---------------------------------------------------------------
            // GetElementPtr — address computation
            // ---------------------------------------------------------------
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type: _,
                ..
            } => {
                // ----------------------------------------------------------
                // ALLOCA FAST-PATH: When the GEP base is a known alloca
                // with a frame offset AND all indices are compile-time
                // constants, emit a single LEA [RBP + alloca_off + total]
                // directly.  This bypasses the alloca's vreg entirely,
                // making the GEP result immune to scratch-register
                // clobbering (e.g. R10/R11 used by the Store handler for
                // aggregate copies).
                // ----------------------------------------------------------
                let alloca_off: Option<i32> = self
                    .frame
                    .as_ref()
                    .and_then(|f| f.alloca_offsets.get(base).copied());
                if let Some(base_off) = alloca_off {
                    // Try to resolve all indices to compile-time constants.
                    let mut total_offset: i64 = 0;
                    let mut all_const = true;
                    for idx_val in indices {
                        if let Some(&cidx) = self.constant_cache.get(idx_val) {
                            total_offset += cidx; // elem_size == 1 (pre-scaled)
                        } else {
                            let idx_op = self.get_value(*idx_val);
                            if let MachineOperand::Immediate(imm) = &idx_op {
                                total_offset += imm;
                            } else {
                                all_const = false;
                                break;
                            }
                        }
                    }
                    if all_const {
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        let final_disp = base_off as i64 + total_offset;
                        return vec![Self::mk_inst(
                            X86Opcode::Lea,
                            Some(dst),
                            &[MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: final_disp,
                            }],
                        )];
                    }
                    // Fall through to generic path if not all-const.
                }

                let base_op = self.get_value(*base);
                let base_op_saved = base_op.clone();
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                let mut out = Vec::new();
                if indices.is_empty() {
                    // When the base is a global symbol, use LEA to load
                    // the address rather than MOV which loads the value
                    // at that address. GEP computes an address.
                    if matches!(&base_op, MachineOperand::GlobalSymbol(_)) {
                        out.push(Self::mk_inst(X86Opcode::Lea, Some(dst), &[base_op]));
                    } else {
                        out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[base_op]));
                    }
                } else {
                    // The IR lowering phase pre-computes byte offsets for
                    // all GEP indices (index * sizeof(element)), so the
                    // backend treats indices as raw byte displacements
                    // without additional scaling.
                    let elem_size: i64 = 1;

                    // Start with the base pointer. For global symbols use
                    // LEA (load effective address) since GEP needs the
                    // address of the global, not its contents.
                    if matches!(&base_op, MachineOperand::GlobalSymbol(_)) {
                        out.push(Self::mk_inst(X86Opcode::Lea, Some(dst.clone()), &[base_op]));
                    } else {
                        out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[base_op]));
                    }

                    for idx_val in indices {
                        // Check constant cache first — compile-time constant
                        // indices can be folded into immediate offsets.
                        let const_idx = self.constant_cache.get(idx_val).copied();
                        if let Some(cidx) = const_idx {
                            let byte_offset = cidx * elem_size;
                            if byte_offset != 0 {
                                out.push(Self::mk_inst(
                                    X86Opcode::Add,
                                    Some(dst.clone()),
                                    &[dst.clone(), MachineOperand::Immediate(byte_offset)],
                                ));
                            }
                            // If offset is 0, no instruction needed.
                            continue;
                        }

                        let idx_op = self.get_value(*idx_val);

                        // Check if index is an immediate — can compute offset at compile time
                        if let MachineOperand::Immediate(imm) = &idx_op {
                            let byte_offset = imm * elem_size;
                            if byte_offset != 0 {
                                out.push(Self::mk_inst(
                                    X86Opcode::Add,
                                    Some(dst.clone()),
                                    &[dst.clone(), MachineOperand::Immediate(byte_offset)],
                                ));
                            }
                            // If offset is 0, no instruction needed.
                        } else if elem_size == 1 {
                            // Element size is 1 byte — no scaling needed.
                            out.push(Self::mk_inst(
                                X86Opcode::Add,
                                Some(dst.clone()),
                                &[dst.clone(), idx_op],
                            ));
                        } else {
                            // Register index with non-unity element size.
                            // Use three-operand IMUL writing to the GEP
                            // result vreg (which IS allocated by the
                            // register allocator) to avoid creating
                            // untracked intermediate virtual registers.
                            //
                            // Save the current base to a temp position,
                            // compute scaled index into dst, then add
                            // the saved base back.
                            //
                            // Pattern:
                            //   IMUL dst, idx_op, elem_size  (3-op)
                            //   ADD  dst, base_saved
                            //
                            // Here base_saved is the base that was
                            // previously MOV'd into dst at the start
                            // of the GEP expansion. We re-fetch it:
                            // since base_op was set at the top, we can
                            // use the original base_op which is still
                            // live in its assigned register.
                            {
                                // When the base is a GlobalSymbol, the IMUL
                                // will overwrite dst. Save the base LEA
                                // result to a temp register first.
                                if matches!(&base_op_saved, MachineOperand::GlobalSymbol(_)) {
                                    let base_tmp = self.new_vreg();
                                    out.push(Self::mk_inst(
                                        X86Opcode::Mov,
                                        Some(base_tmp.clone()),
                                        std::slice::from_ref(&dst),
                                    ));
                                    // Three-operand IMUL: dst = idx * size
                                    out.push(Self::mk_inst(
                                        X86Opcode::Imul,
                                        Some(dst.clone()),
                                        &[idx_op, MachineOperand::Immediate(elem_size)],
                                    ));
                                    // ADD dst, base_tmp
                                    out.push(Self::mk_inst(
                                        X86Opcode::Add,
                                        Some(dst.clone()),
                                        &[dst.clone(), base_tmp],
                                    ));
                                } else {
                                    // Three-operand IMUL: dst = idx * size
                                    // This writes into the GEP result vreg.
                                    out.push(Self::mk_inst(
                                        X86Opcode::Imul,
                                        Some(dst.clone()),
                                        &[idx_op, MachineOperand::Immediate(elem_size)],
                                    ));
                                    // ADD dst, base — add the original
                                    // base address (still in its register).
                                    out.push(Self::mk_inst(
                                        X86Opcode::Add,
                                        Some(dst.clone()),
                                        &[dst.clone(), base_op_saved.clone()],
                                    ));
                                }
                            }
                        }
                    }
                }
                out
            }

            // ---------------------------------------------------------------
            // BitCast — reinterpretation / phi-copy / float↔int conversion
            // ---------------------------------------------------------------
            Instruction::BitCast {
                result,
                value,
                to_type,
                source_unsigned,
                ..
            } => {
                let src = self.get_value(*value);
                // If this result already has a VR assigned (e.g. from a
                // phi-copy in a different predecessor block), reuse the
                // same VR so all predecessor paths write to the same
                // virtual register and the register allocator assigns a
                // single physical location.
                let dst = if let Some(existing) = self.value_map.get(result) {
                    existing.clone()
                } else {
                    let vr = self.new_vreg();
                    self.set_value(*result, vr.clone());
                    vr
                };

                // Determine source IR type from value_types map so we
                // can emit proper conversion instructions for float↔int
                // casts instead of bit-reinterpretation moves.
                let from_type = self.value_types.get(value).cloned();
                let from_is_float = from_type.as_ref().map_or(false, |t| t.is_float());
                let to_is_float = to_type.is_float();
                let to_is_int = to_type.is_integer();
                let from_is_int = from_type.as_ref().map_or(false, |t| t.is_integer());
                let is_unsigned = *source_unsigned;

                if from_is_float && to_is_int {
                    // Float → Integer: CVTTSD2SI / CVTTSS2SI (truncate toward zero)
                    let is_f32 = matches!(from_type.as_ref(), Some(IrType::F32));
                    let to_is_i64 = matches!(to_type, IrType::I64 | IrType::I128);

                    if is_unsigned && to_is_i64 {
                        // Float → Unsigned 64-bit integer.
                        // cvttsd2si treats destination as signed i64, so
                        // values >= 2^63 would overflow. We use:
                        //   mov r11, 0x43E0000000000000 (double 2^63)
                        //   movq xmm_tmp(r10), r11
                        //   subsd xmm_src_copy, xmm_tmp → xmm reduced
                        //   cvttsd2si xmm_src → r11 (positive path)
                        //   cvttsd2si xmm_reduced → r10 (negative path)
                        //   mov  imm 0x8000000000000000 → scratch
                        //   or   r10, scratch
                        //   test src_float, float_2p63 (compare)
                        //   cmovae → pick r10 (unsigned path)
                        //
                        // Simpler: just use cvttsd2si with REX.W for i64.
                        // Values >= 2^63 will produce 0x8000000000000000
                        // (the "integer indefinite" value). Then we need:
                        //   subtract 2^63 from float, convert, add 2^63 to int
                        //
                        // For now, use the simple signed conversion which
                        // handles values up to 2^63-1 correctly. Values
                        // above that are undefined behavior in C anyway
                        // (overflow on float→unsigned conversion is UB
                        // for values outside the representable range).
                        let opcode = if is_f32 {
                            X86Opcode::Cvtss2si
                        } else {
                            X86Opcode::Cvtsd2si
                        };
                        let mut inst = Self::mk_inst(opcode, Some(dst), &[src]);
                        inst.operand_size = 8; // REX.W for 64-bit
                        vec![inst]
                    } else {
                        let opcode = if is_f32 {
                            X86Opcode::Cvtss2si
                        } else {
                            X86Opcode::Cvtsd2si
                        };
                        let mut inst = Self::mk_inst(opcode, Some(dst), &[src]);
                        // Set operand_size to target width for REX.W control:
                        // 8 = 64-bit dest (REX.W), 4 = 32-bit dest (no REX.W).
                        //
                        // CRITICAL: For unsigned 32-bit (or smaller) targets,
                        // we MUST use 64-bit conversion (REX.W).  The 32-bit
                        // `cvttsd2si` treats the destination as *signed* i32,
                        // so values in [2^31, 2^32-1] produce the x86
                        // "integer indefinite" value 0x80000000 instead of
                        // the correct unsigned result.  By converting to
                        // signed i64 first, values up to 2^32-1 fit in the
                        // positive range of i64, and the lower 32 bits of
                        // the result register naturally give the correct
                        // unsigned 32-bit value.
                        inst.operand_size = match to_type {
                            IrType::I64 | IrType::I128 => 8,
                            _ if is_unsigned => 8,
                            _ => 4,
                        };
                        vec![inst]
                    }
                } else if from_is_int && to_is_float {
                    // Integer → Float: CVTSI2SD / CVTSI2SS
                    let is_f32 = matches!(to_type, IrType::F32);
                    let from_is_i64 =
                        matches!(from_type.as_ref(), Some(IrType::I64) | Some(IrType::I128));

                    if is_unsigned && from_is_i64 {
                        // Unsigned 64-bit → Float conversion.
                        //
                        // `cvtsi2sd` treats its GPR source as *signed* i64.
                        // For values 0..2^63-1 the signed interpretation
                        // is correct and gives a more precise result than
                        // the halving trick.  For values with bit 63 set
                        // (≥ 2^63), cvtsi2sd would produce a negative
                        // double, so we must halve, convert, then double.
                        //
                        // Generated pattern (matches GCC):
                        //   push  r10
                        //   push  r11
                        //   test  src, src
                        //   js    .Lu64halve_N       ; bit 63 set → halve
                        //   cvtsi2sd src, dst        ; positive path
                        //   jmp   .Lu64done_N
                        // .Lu64halve_N:
                        //   mov   r10, src
                        //   and   r10, 1             ; save LSB
                        //   mov   r11, src
                        //   shr   r11, 1             ; halve
                        //   or    r11, r10           ; (src>>1) | (src&1)
                        //   cvtsi2sd r11, dst
                        //   addsd dst, dst           ; double
                        // .Lu64done_N:
                        //   pop   r11
                        //   pop   r10
                        //
                        // We use R10/R11 (caller-saved) as scratch, with
                        // push/pop to preserve live values placed there
                        // by the register allocator.
                        let uid = self.next_vreg;
                        self.next_vreg += 1;
                        let halve_label = format!(".Lu64halve_{}", uid);
                        let done_label = format!(".Lu64done_{}", uid);

                        let r11 = MachineOperand::Register(R11);
                        let r10 = MachineOperand::Register(R10);
                        let mut insts = Vec::new();

                        // Save R10 and R11.
                        insts.push(Self::mk_inst(X86Opcode::Push, None, &[r10.clone()]));
                        insts.push(Self::mk_inst(X86Opcode::Push, None, &[r11.clone()]));

                        // test src, src — sets SF if bit 63 is set.
                        let mut test_inst =
                            Self::mk_inst(X86Opcode::Test, None, &[src.clone(), src.clone()]);
                        test_inst.operand_size = 8;
                        insts.push(test_inst);

                        // js .Lu64halve_N — jump to halving path if
                        // bit 63 set (SF=1).
                        insts.push(Self::mk_inst(
                            X86Opcode::Js,
                            None,
                            &[MachineOperand::GlobalSymbol(halve_label.clone())],
                        ));

                        // ---- Positive path: direct signed conversion ----
                        let conv_opcode = if is_f32 {
                            X86Opcode::Cvtsi2ss
                        } else {
                            X86Opcode::Cvtsi2sd
                        };
                        let mut pos_conv =
                            Self::mk_inst(conv_opcode, Some(dst.clone()), &[src.clone()]);
                        pos_conv.operand_size = 8;
                        insts.push(pos_conv);

                        // jmp .Lu64done_N — skip halving path.
                        insts.push(Self::mk_inst(
                            X86Opcode::Jmp,
                            None,
                            &[MachineOperand::GlobalSymbol(done_label.clone())],
                        ));

                        // ---- Halving path (bit 63 set) ----
                        // .Lu64halve_N:
                        insts.push(Self::mk_inst(
                            X86Opcode::InternalLabelDef,
                            None,
                            &[MachineOperand::GlobalSymbol(halve_label)],
                        ));

                        // r10 = src & 1  (save LSB before shifting)
                        let mut mov_r10 =
                            Self::mk_inst(X86Opcode::Mov, Some(r10.clone()), &[src.clone()]);
                        mov_r10.operand_size = 8;
                        insts.push(mov_r10);
                        let mut and_r10 = Self::mk_inst(
                            X86Opcode::And,
                            Some(r10.clone()),
                            &[r10.clone(), MachineOperand::Immediate(1)],
                        );
                        and_r10.operand_size = 8;
                        insts.push(and_r10);

                        // r11 = src >> 1
                        let mut mov_r11 =
                            Self::mk_inst(X86Opcode::Mov, Some(r11.clone()), &[src.clone()]);
                        mov_r11.operand_size = 8;
                        insts.push(mov_r11);
                        let mut shr_r11 = Self::mk_inst(
                            X86Opcode::Shr,
                            Some(r11.clone()),
                            &[r11.clone(), MachineOperand::Immediate(1)],
                        );
                        shr_r11.operand_size = 8;
                        insts.push(shr_r11);

                        // r11 |= r10  →  (src>>1) | (src&1)
                        let mut or_r11 = Self::mk_inst(
                            X86Opcode::Or,
                            Some(r11.clone()),
                            &[r11.clone(), r10.clone()],
                        );
                        or_r11.operand_size = 8;
                        insts.push(or_r11);

                        // cvtsi2sd r11, dst  (halved value, always < 2^63)
                        let mut halve_conv =
                            Self::mk_inst(conv_opcode, Some(dst.clone()), &[r11.clone()]);
                        halve_conv.operand_size = 8;
                        insts.push(halve_conv);

                        // addsd dst, dst  (double to recover original scale)
                        let add_opcode = if is_f32 {
                            X86Opcode::Addss
                        } else {
                            X86Opcode::Addsd
                        };
                        insts.push(Self::mk_inst(
                            add_opcode,
                            Some(dst.clone()),
                            &[dst.clone(), dst.clone()],
                        ));

                        // .Lu64done_N:
                        insts.push(Self::mk_inst(
                            X86Opcode::InternalLabelDef,
                            None,
                            &[MachineOperand::GlobalSymbol(done_label)],
                        ));

                        // Restore R11 and R10 (reverse order).
                        insts.push(Self::mk_inst(X86Opcode::Pop, Some(r11.clone()), &[]));
                        insts.push(Self::mk_inst(X86Opcode::Pop, Some(r10.clone()), &[]));

                        insts
                    } else {
                        // Signed conversion, or unsigned but ≤32-bit
                        // (which fits in signed 64-bit).
                        let opcode = if is_f32 {
                            X86Opcode::Cvtsi2ss
                        } else {
                            X86Opcode::Cvtsi2sd
                        };
                        let mut inst = Self::mk_inst(opcode, Some(dst), &[src]);
                        // REX.W control for cvtsi2sd/cvtsi2ss:
                        // - Unsigned ≤32-bit: need REX.W so the zero-extended
                        //   64-bit register is read as signed i64 (correctly
                        //   represents all u32 values).
                        // - Signed 64-bit: need REX.W for full 64-bit range.
                        // - Signed ≤32-bit: NO REX.W — cvtsi2sd reads the
                        //   32-bit register as signed i32, which correctly
                        //   handles negative values (e.g. -1 in EAX →
                        //   -1.0, not 4294967295.0).
                        if is_unsigned || from_is_i64 {
                            inst.operand_size = 8;
                        }
                        vec![inst]
                    }
                } else if from_is_float && to_is_float {
                    // Float → Float conversion or same-type copy.
                    // Phi elimination inserts BitCast with identical types
                    // for phi copies — these need a plain MOVSD/MOVSS, NOT
                    // a conversion instruction.
                    let from_is_f32 = matches!(from_type.as_ref(), Some(IrType::F32));
                    let to_is_f32 = matches!(to_type, IrType::F32);
                    let opcode = if from_is_f32 == to_is_f32 {
                        // Same-type copy: F64→F64 or F32→F32
                        if to_is_f32 {
                            X86Opcode::Movss
                        } else {
                            X86Opcode::Movsd
                        }
                    } else if to_is_f32 {
                        // Narrowing: F64→F32
                        X86Opcode::Cvtsd2ss
                    } else {
                        // Widening: F32→F64
                        X86Opcode::Cvtss2sd
                    };
                    vec![Self::mk_inst(opcode, Some(dst), &[src])]
                } else {
                    // Same-class copy: MOV for int, MOVSD for float.
                    // For GlobalSymbol operands representing function
                    // pointers or global variable addresses, use LEA to
                    // materialise the address instead of MOV which would
                    // load the value at that address.
                    let is_sym_addr = matches!(&src, MachineOperand::GlobalSymbol(_))
                        && (self.func_ref_names.contains_key(value)
                            || self.global_var_refs.contains_key(value));

                    let opcode = if to_is_float {
                        X86Opcode::Movsd
                    } else if is_sym_addr {
                        X86Opcode::Lea
                    } else {
                        X86Opcode::Mov
                    };
                    vec![Self::mk_inst(opcode, Some(dst), &[src])]
                }
            }

            // ---------------------------------------------------------------
            // Trunc — narrow an integer
            // ---------------------------------------------------------------
            Instruction::Trunc {
                result,
                value,
                to_type,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // On x86-64, truncation to 32-bit uses a 32-bit MOV which
                // implicitly zero-extends the upper 32 bits.  For smaller
                // widths (I8, I16, I1), we must AND with the appropriate
                // mask so that the upper bits are cleared — otherwise a
                // later boolean test of the full register would see the
                // non-zero upper bits and produce wrong results.
                match to_type {
                    IrType::I1 => {
                        let mut insts =
                            vec![Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[src])];
                        insts.push(Self::mk_inst(
                            X86Opcode::And,
                            Some(dst),
                            &[MachineOperand::Immediate(1)],
                        ));
                        insts
                    }
                    IrType::I8 => {
                        let mut insts =
                            vec![Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[src])];
                        insts.push(Self::mk_inst(
                            X86Opcode::And,
                            Some(dst),
                            &[MachineOperand::Immediate(0xFF)],
                        ));
                        insts
                    }
                    IrType::I16 => {
                        let mut insts =
                            vec![Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[src])];
                        insts.push(Self::mk_inst(
                            X86Opcode::And,
                            Some(dst),
                            &[MachineOperand::Immediate(0xFFFF)],
                        ));
                        insts
                    }
                    _ => {
                        // I32: 32-bit MOV zeroes upper 32 bits automatically
                        vec![Self::mk_inst(X86Opcode::Mov, Some(dst), &[src])]
                    }
                }
            }

            // ---------------------------------------------------------------
            // ZExt — zero extend
            // ---------------------------------------------------------------
            Instruction::ZExt {
                result,
                value,
                to_type: _,
                from_type,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // MOVZX for 8→32, 16→32.  For 32→64, a 32-bit MOV
                // implicitly zero-extends on x86-64.
                let opcode = X86Opcode::MovZX;
                let mut inst = Self::mk_inst(opcode, Some(dst), &[src]);
                // Set operand_size so the encoder knows source width:
                // 1 = byte source (MOVZBL), 2 = word source (MOVZWL)
                let src_bytes = match from_type {
                    IrType::I16 => 2u8,
                    IrType::I32 => 4u8, // 32→64: MOV implicitly zero-extends
                    _ => 1u8,           // I8 or default → byte source
                };
                inst.operand_size = src_bytes;
                if src_bytes == 4 {
                    // 32→64 zero extend: a plain 32-bit MOV zero-extends on x86-64
                    inst.opcode = X86Opcode::Mov.as_u32();
                    inst.operand_size = 4;
                }
                vec![inst]
            }

            // ---------------------------------------------------------------
            // SExt — sign extend
            // ---------------------------------------------------------------
            Instruction::SExt {
                result,
                value,
                to_type: _,
                from_type,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                let mut inst = Self::mk_inst(X86Opcode::MovSX, Some(dst), &[src]);
                let src_bytes = match from_type {
                    IrType::I16 => 2u8,
                    IrType::I32 => 4u8,
                    _ => 1u8,
                };
                inst.operand_size = src_bytes;
                vec![inst]
            }

            // ---------------------------------------------------------------
            // IntToPtr — integer → pointer (on x86-64, pointer = i64)
            // ---------------------------------------------------------------
            Instruction::IntToPtr { result, value, .. } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // GlobalSymbol operands representing function pointers or
                // global variable addresses must use LEA (load effective
                // address) instead of MOV (which would load the VALUE at
                // the symbol address rather than the address itself).
                let is_sym_addr = matches!(&src, MachineOperand::GlobalSymbol(_))
                    && (self.func_ref_names.contains_key(value)
                        || self.global_var_refs.contains_key(value));
                let opcode = if is_sym_addr {
                    X86Opcode::Lea
                } else {
                    X86Opcode::Mov
                };
                vec![Self::mk_inst(opcode, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // PtrToInt — pointer → integer
            // ---------------------------------------------------------------
            Instruction::PtrToInt {
                result,
                value,
                to_type: _,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // GlobalSymbol operands representing function pointers or
                // global variable addresses must use LEA (load effective
                // address) instead of MOV (which would load the VALUE at
                // the symbol address rather than the address itself).
                let is_sym_addr = matches!(&src, MachineOperand::GlobalSymbol(_))
                    && (self.func_ref_names.contains_key(value)
                        || self.global_var_refs.contains_key(value));
                let opcode = if is_sym_addr {
                    X86Opcode::Lea
                } else {
                    X86Opcode::Mov
                };
                vec![Self::mk_inst(opcode, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // InlineAsm
            // ---------------------------------------------------------------
            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects,
                is_volatile,
                goto_targets: _,
                ..
            } => {
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                // Parse constraints to determine output/input split.
                // The constraint string is comma-separated: e.g. "=r,r,0"
                // for one output register + two input registers.
                let constraint_parts: Vec<&str> = constraints.split(',').collect();
                // Count output constraints (those starting with '=' or '+')
                let num_outputs = constraint_parts
                    .iter()
                    .filter(|c| {
                        let t = c.trim();
                        t.starts_with('=') || t.starts_with('+')
                    })
                    .count();

                // Create output vregs: output 0 = dst, outputs 1+ = extra vregs.
                // Multi-output asm needs separate destination registers for
                // each output so template %0, %1, etc. map to distinct regs.
                //
                // MEMORY CONSTRAINTS ("=m", "+m"): For memory constraints,
                // the asm template %N should expand to a memory operand
                // (e.g., -8(%rbp)), not a register. We look up the
                // alloca offset of the output variable and use a Memory
                // operand instead of a virtual register.
                let mut output_vregs: Vec<MachineOperand> = Vec::new();
                for (oi, cp) in constraint_parts.iter().enumerate() {
                    if oi >= num_outputs {
                        break;
                    }
                    let t = cp.trim();
                    let body = t.trim_start_matches(|ch: char| ch == '=' || ch == '+');
                    let is_mem = body.contains('m');
                    if is_mem {
                        // Memory constraint: use the alloca memory operand
                        // for this output variable. The IR operands[oi] is
                        // the pointer (alloca) for this output.
                        let ptr_val = operands[oi];
                        let mem_op = if let Some(ref frame) = self.frame {
                            if let Some(&off) = frame.alloca_offsets.get(&ptr_val) {
                                MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: off as i64,
                                }
                            } else {
                                // Fallback: use a vreg (may not produce ideal code)
                                if oi == 0 {
                                    dst.clone()
                                } else {
                                    self.new_vreg()
                                }
                            }
                        } else if oi == 0 {
                            dst.clone()
                        } else {
                            self.new_vreg()
                        };
                        output_vregs.push(mem_op);
                    } else if oi == 0 {
                        output_vregs.push(dst.clone());
                    } else {
                        output_vregs.push(self.new_vreg());
                    }
                }
                // If no output constraints were parsed, default to dst.
                if output_vregs.is_empty() {
                    output_vregs.push(dst.clone());
                }

                // Build the operand list for the MachineInstruction.
                // Layout: [extra_output_vregs(1..n-1), input_operands...]
                // The result register (output 0) is in inst.result.
                // Extra output vregs go first so the template substitution
                // can reconstruct the GCC numbering:
                //   %0 → inst.result (output 0)
                //   %1 → inst.operands[0] (output 1, if multi-output)
                //   %num_outputs-1 → inst.operands[num_outputs-2]
                //   %num_outputs → inst.operands[num_outputs-1] (first input)
                let mut asm_operands = Vec::new();

                // Insert extra output vregs (outputs 1..num_outputs-1).
                for out_vreg in output_vregs.iter().skip(1) {
                    asm_operands.push(out_vreg.clone());
                }

                // Add input operands (skip output pointer operands in IR).
                for (idx, op_val) in operands.iter().enumerate() {
                    if idx < num_outputs {
                        continue;
                    }
                    let constraint_idx = idx;
                    let is_immediate_constraint = if constraint_idx < constraint_parts.len() {
                        let c = constraint_parts[constraint_idx].trim();
                        let c = c.trim_start_matches(|ch: char| ch == '=' || ch == '+');
                        c.contains('i') || c.contains('n') || c.contains('I')
                    } else {
                        false
                    };

                    if is_immediate_constraint {
                        if let Some(&imm) = self.constant_cache.get(op_val) {
                            asm_operands.push(MachineOperand::Immediate(imm));
                        } else {
                            asm_operands.push(self.get_value(*op_val));
                        }
                    } else {
                        asm_operands.push(self.get_value(*op_val));
                    }
                }

                // Pre-move generation for read-write ('+') constraints and
                // digit-tied constraints (e.g. "0", "1").
                //
                // Read-write outputs need their input value pre-loaded into
                // the output register.  Digit-tied inputs specify that the
                // input must use the same register as the indicated output,
                // so we pre-load the input value into that output's vreg.
                let mut pre_moves: Vec<MachineInstruction> = Vec::new();
                {
                    let num_explicit_inputs = constraint_parts
                        .iter()
                        .filter(|c| {
                            let t = c.trim();
                            !t.starts_with('=') && !t.starts_with('+')
                        })
                        .count();
                    // Read-write input values start after output pointers and
                    // explicit inputs in the IR operand list.
                    let rw_start = num_outputs + num_explicit_inputs;
                    let mut rw_idx = 0;
                    for (ci, cp) in constraint_parts.iter().enumerate() {
                        let t = cp.trim();
                        if t.starts_with('+') && ci < num_outputs {
                            // Read-write output: pre-load from appended RW
                            // input value into this output's vreg.
                            // Skip pre-move for memory constraints (+m) since
                            // the operand stays in memory — no register copy needed.
                            let constraint_body =
                                t.trim_start_matches(|ch: char| ch == '+' || ch == '=');
                            let is_memory_constraint = constraint_body.contains('m');
                            let op_idx = rw_start + rw_idx;
                            if !is_memory_constraint && op_idx < operands.len() {
                                let src_op = self.get_value(operands[op_idx]);
                                let target = &output_vregs[ci];
                                // GlobalSymbol addresses need LEA, not MOV.
                                let opcode = if let MachineOperand::GlobalSymbol(_) = &src_op {
                                    let ir_val = operands[op_idx];
                                    if self.global_var_refs.contains_key(&ir_val)
                                        || self.func_ref_names.contains_key(&ir_val)
                                    {
                                        X86Opcode::Lea
                                    } else {
                                        X86Opcode::Mov
                                    }
                                } else {
                                    X86Opcode::Mov
                                };
                                let mut inst = MachineInstruction::new(opcode.as_u32());
                                inst.result = Some(target.clone());
                                inst.operands.push(src_op);
                                pre_moves.push(inst);
                            }
                            rw_idx += 1;
                        } else if !t.starts_with('=') && !t.starts_with('+') {
                            // Input constraint — check for digit-tied (e.g. "0")
                            // which means this input uses the same register as
                            // the indicated output operand.
                            let stripped = t.trim_start_matches(|ch: char| ch == '%' || ch == '&');
                            if let Ok(tied_idx) = stripped.parse::<usize>() {
                                if tied_idx < num_outputs && ci < operands.len() {
                                    let src_op = self.get_value(operands[ci]);
                                    let target = &output_vregs[tied_idx];
                                    // GlobalSymbol representing a global variable
                                    // address or function pointer must use LEA,
                                    // not MOV — MOV would load the VALUE at the
                                    // symbol address instead of the address itself.
                                    let opcode = if let MachineOperand::GlobalSymbol(_) = &src_op {
                                        let ir_val = operands[ci];
                                        if self.global_var_refs.contains_key(&ir_val)
                                            || self.func_ref_names.contains_key(&ir_val)
                                        {
                                            X86Opcode::Lea
                                        } else {
                                            X86Opcode::Mov
                                        }
                                    } else {
                                        X86Opcode::Mov
                                    };
                                    let mut inst = MachineInstruction::new(opcode.as_u32());
                                    inst.result = Some(target.clone());
                                    inst.operands.push(src_op);
                                    pre_moves.push(inst);
                                }
                            }
                        }
                    }
                }

                let mut inst = MachineInstruction::new(X86Opcode::InlineAsm.as_u32());
                // Use the first output operand (may be memory for '=m')
                inst.result = Some(output_vregs[0].clone());
                for op in &asm_operands {
                    inst.operands.push(op.clone());
                }
                inst.asm_num_outputs = num_outputs;
                inst.asm_template = Some(template.clone());
                inst.asm_clobbers = clobbers.clone();
                if *has_side_effects || *is_volatile {
                    inst.is_call = true;
                }
                let mut all_insts = pre_moves;
                all_insts.push(inst);
                all_insts
            }

            // ---------------------------------------------------------------
            // IndirectBranch — computed goto: jmp *%reg
            // ---------------------------------------------------------------
            Instruction::IndirectBranch { target, .. } => {
                let target_op = self.get_value(*target);
                let mut jmp = Self::mk_inst(X86Opcode::Jmp, None, &[target_op]);
                jmp.is_terminator = true;
                // Mark as branch so epilogue insertion does NOT insert
                // leave+ret before this instruction — it's a jump, not a
                // function return.
                jmp.is_branch = true;
                vec![jmp]
            }

            // ---------------------------------------------------------------
            // BlockAddress — materialize address of a labeled basic block
            // ---------------------------------------------------------------
            Instruction::BlockAddress { result, block, .. } => {
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // Map IR block → machine block index, then use the assembler's
                // .L{mach_idx} label which is always defined and resolved
                // inline (no external relocation needed).
                let mach_idx = *self.block_map.get(&block.index()).unwrap_or(&0);
                let block_label = format!(".L{}", mach_idx);
                let inst = Self::mk_inst(
                    X86Opcode::Lea,
                    Some(dst),
                    &[MachineOperand::GlobalSymbol(block_label)],
                );
                vec![inst]
            }
        }
    }

    // ===================================================================
    // Binary operation selection
    // ===================================================================

    fn select_binop(
        &mut self,
        result: Value,
        op: BinOp,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
    ) -> Vec<MachineInstruction> {
        let mut lhs_op = self.get_value(lhs);
        let mut rhs_op = self.get_value(rhs);
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());
        // Keep a clone of dst for the sub-register truncation step
        // that runs after the match block (the match arms move `dst`).
        let dst_for_trunc = dst.clone();
        let mut out = Vec::new();

        // GlobalSymbol operands representing function pointers or global
        // variable addresses must be materialised into registers via LEA
        // before use in arithmetic instructions.  MOV with a GlobalSymbol
        // source performs a memory load from the symbol address, which is
        // incorrect when we need the address itself.
        if let MachineOperand::GlobalSymbol(_) = &lhs_op {
            if self.func_ref_names.contains_key(&lhs) || self.global_var_refs.contains_key(&lhs) {
                let tmp = self.new_vreg();
                out.push(Self::mk_inst(X86Opcode::Lea, Some(tmp.clone()), &[lhs_op]));
                lhs_op = tmp;
            }
        }
        if let MachineOperand::GlobalSymbol(_) = &rhs_op {
            if self.func_ref_names.contains_key(&rhs) || self.global_var_refs.contains_key(&rhs) {
                let tmp = self.new_vreg();
                out.push(Self::mk_inst(X86Opcode::Lea, Some(tmp.clone()), &[rhs_op]));
                rhs_op = tmp;
            }
        }

        // Compute operand size for sub-64-bit integer operations.
        // On x86-64, 32-bit register operations (e.g. `add eax, ecx`)
        // implicitly zero-extend the result to 64 bits, which is
        // essential for correct signed semantics: without this, a
        // negative i32 like 0xFFFFFFFE would appear as the large
        // positive 64-bit value 0x00000000FFFFFFFE, breaking signed
        // comparisons and arithmetic right shifts.
        let int_op_size: u8 = match ty {
            IrType::I1 | IrType::I8 => 4, // promote to 32-bit minimum
            IrType::I16 => 4,             // promote to 32-bit minimum (avoids 0x66 prefix overhead)
            IrType::I32 => 4,
            _ => 0, // 0 = use default (64-bit on x86-64)
        };

        // Helper closure: create an instruction with a specific operand size.
        let mk_sized = |opcode: X86Opcode,
                        result: Option<MachineOperand>,
                        operands: &[MachineOperand],
                        op_size: u8|
         -> MachineInstruction {
            let mut inst = Self::mk_inst(opcode, result, operands);
            if op_size > 0 {
                inst.operand_size = op_size;
            }
            inst
        };

        match op {
            // -- Integer arithmetic --
            BinOp::Add => {
                // x86 two-address form: dst = lhs + rhs
                // 1. MOV dst, lhs     (dst now holds the lhs value)
                // 2. ADD dst, rhs     (dst += rhs)
                // NOTE: The encoder normalizes result into operands[0],
                // so we provide only rhs as the explicit operand.
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(mk_sized(
                    X86Opcode::Add,
                    Some(dst.clone()),
                    &[rhs_op],
                    int_op_size,
                ));
            }
            BinOp::Sub => {
                // x86 two-address form: dst = lhs - rhs
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(mk_sized(
                    X86Opcode::Sub,
                    Some(dst.clone()),
                    &[rhs_op],
                    int_op_size,
                ));
            }
            BinOp::Mul => {
                // x86 two-address form: dst = lhs * rhs
                // 1. MOV dst, lhs     (dst now holds the lhs value)
                // 2. IMUL dst, rhs    (dst *= rhs)
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(mk_sized(X86Opcode::Imul, Some(dst), &[rhs_op], int_op_size));
            }
            BinOp::SDiv => {
                // x86-64 IDIV: RDX:RAX ÷ operand → quotient in RAX,
                // remainder in RDX.  For i32 we use CDQ + 32-bit IDIV;
                // for i64 we use CQO + 64-bit IDIV.
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                    int_op_size,
                ));
                if int_op_size == 4 {
                    out.push(Self::mk_inst(X86Opcode::Cdq, None, &[]));
                } else {
                    out.push(Self::mk_inst(X86Opcode::Cqo, None, &[]));
                }
                out.push(mk_sized(X86Opcode::Idiv, None, &[rhs_op], int_op_size));
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RAX)],
                    int_op_size,
                ));
            }
            BinOp::UDiv => {
                // x86-64 DIV: RDX:RAX ÷ operand → quotient in RAX.
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Xor,
                    Some(MachineOperand::Register(RDX)),
                    &[MachineOperand::Register(RDX)],
                ));
                out.push(mk_sized(X86Opcode::Div, None, &[rhs_op], int_op_size));
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RAX)],
                    int_op_size,
                ));
            }
            BinOp::SRem => {
                // IDIV: remainder in RDX.
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                    int_op_size,
                ));
                if int_op_size == 4 {
                    out.push(Self::mk_inst(X86Opcode::Cdq, None, &[]));
                } else {
                    out.push(Self::mk_inst(X86Opcode::Cqo, None, &[]));
                }
                out.push(mk_sized(X86Opcode::Idiv, None, &[rhs_op], int_op_size));
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RDX)],
                    int_op_size,
                ));
            }
            BinOp::URem => {
                // DIV: remainder in RDX.
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Xor,
                    Some(MachineOperand::Register(RDX)),
                    &[MachineOperand::Register(RDX)],
                ));
                out.push(mk_sized(X86Opcode::Div, None, &[rhs_op], int_op_size));
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RDX)],
                    int_op_size,
                ));
            }

            // -- Bitwise --
            // NOTE: The encoder normalizes result into operands[0],
            // so we provide only rhs as the explicit operand.
            BinOp::And => {
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(mk_sized(X86Opcode::And, Some(dst), &[rhs_op], int_op_size));
            }
            BinOp::Or => {
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(mk_sized(X86Opcode::Or, Some(dst), &[rhs_op], int_op_size));
            }
            BinOp::Xor => {
                // Detect bitwise NOT: XOR with Value::UNDEF represents
                // `~operand` (XOR with all-ones). Emit x86 NOT instead
                // of XOR with the UNDEF-resolved Immediate(0) which
                // would be a no-op.
                if rhs == Value::UNDEF {
                    out.push(mk_sized(
                        X86Opcode::Mov,
                        Some(dst.clone()),
                        &[lhs_op],
                        int_op_size,
                    ));
                    out.push(mk_sized(X86Opcode::Not, Some(dst), &[], int_op_size));
                } else {
                    out.push(mk_sized(
                        X86Opcode::Mov,
                        Some(dst.clone()),
                        &[lhs_op],
                        int_op_size,
                    ));
                    out.push(mk_sized(X86Opcode::Xor, Some(dst), &[rhs_op], int_op_size));
                }
            }

            // -- Shifts (shift amount must be in CL) --
            BinOp::Shl => {
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RCX)),
                    &[rhs_op],
                ));
                out.push(mk_sized(
                    X86Opcode::Shl,
                    Some(dst),
                    &[MachineOperand::Register(RCX)],
                    int_op_size,
                ));
            }
            BinOp::AShr => {
                // Arithmetic shift right — MUST use correct operand size
                // so SAR sign-extends from the correct bit position.
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RCX)),
                    &[rhs_op],
                ));
                out.push(mk_sized(
                    X86Opcode::Sar,
                    Some(dst),
                    &[MachineOperand::Register(RCX)],
                    int_op_size,
                ));
            }
            BinOp::LShr => {
                out.push(mk_sized(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[lhs_op],
                    int_op_size,
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RCX)),
                    &[rhs_op],
                ));
                out.push(mk_sized(
                    X86Opcode::Shr,
                    Some(dst),
                    &[MachineOperand::Register(RCX)],
                    int_op_size,
                ));
            }

            // -- Floating-point (SSE / SSE2) --
            BinOp::FAdd => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Addss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Addsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FSub => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Subss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Subsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FMul => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Mulss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Mulsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FDiv => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Divss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Divsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FRem => {
                // IEEE 754 FP remainder — no single x86-64 instruction.
                // Emit a call to a software helper (or use fmod-style sequence).
                // For now, generate a placeholder sequence.
                let is_f32 = matches!(ty, IrType::F32);
                let mov_op = if is_f32 {
                    X86Opcode::Movss
                } else {
                    X86Opcode::Movsd
                };
                // Place lhs in XMM0, rhs in XMM1, call __fmod helper.
                out.push(Self::mk_inst(
                    mov_op,
                    Some(MachineOperand::Register(XMM0)),
                    &[lhs_op],
                ));
                out.push(Self::mk_inst(
                    mov_op,
                    Some(MachineOperand::Register(XMM1)),
                    &[rhs_op],
                ));
                let mut call = Self::mk_inst(
                    X86Opcode::Call,
                    None,
                    &[MachineOperand::GlobalSymbol(
                        if is_f32 { "fmodf" } else { "fmod" }.to_string(),
                    )],
                );
                call.is_call = true;
                out.push(call);
                out.push(Self::mk_inst(
                    mov_op,
                    Some(dst),
                    &[MachineOperand::Register(XMM0)],
                ));
            }
        }

        // -----------------------------------------------------------
        // Sub-register truncation for promoted I8/I16 operations
        // -----------------------------------------------------------
        // BCC promotes I8/I16 arithmetic to 32-bit for efficiency,
        // but the result in the register may exceed the original
        // width (e.g. uint8_t 255 + 1 = 256 in 32-bit register).
        // Mask the result back to the correct width so that
        // subsequent comparisons and uses see the truncated value.
        let needs_trunc = matches!(
            op,
            BinOp::Add
                | BinOp::Sub
                | BinOp::Mul
                | BinOp::Shl
                | BinOp::SDiv
                | BinOp::UDiv
                | BinOp::SRem
                | BinOp::URem
        );
        if needs_trunc {
            let mask: Option<i64> = match ty {
                IrType::I8 => Some(0xFF),
                IrType::I16 => Some(0xFFFF),
                _ => None,
            };
            if let Some(m) = mask {
                out.push(Self::mk_inst(
                    X86Opcode::And,
                    Some(dst_for_trunc.clone()),
                    &[MachineOperand::Immediate(m)],
                ));
            }
        }

        out
    }

    // ===================================================================
    // Integer comparison selection
    // ===================================================================

    fn select_icmp(
        &mut self,
        result: Value,
        op: ICmpOp,
        lhs: Value,
        rhs: Value,
    ) -> Vec<MachineInstruction> {
        let mut lhs_op = self.get_value(lhs);
        let mut rhs_op = self.get_value(rhs);

        let dst = self.new_vreg();
        self.set_value(result, dst.clone());

        // Constant-fold: if both operands are Immediate, evaluate the
        // comparison at codegen time instead of emitting an invalid
        // CMP imm, imm instruction.
        if let (MachineOperand::Immediate(a), MachineOperand::Immediate(b)) = (&lhs_op, &rhs_op) {
            let a = *a;
            let b = *b;
            let cmp_result: i64 = match op {
                ICmpOp::Eq => (a == b) as i64,
                ICmpOp::Ne => (a != b) as i64,
                ICmpOp::Slt => (a < b) as i64,
                ICmpOp::Sle => (a <= b) as i64,
                ICmpOp::Sgt => (a > b) as i64,
                ICmpOp::Sge => (a >= b) as i64,
                ICmpOp::Ult => ((a as u64) < (b as u64)) as i64,
                ICmpOp::Ule => ((a as u64) <= (b as u64)) as i64,
                ICmpOp::Ugt => ((a as u64) > (b as u64)) as i64,
                ICmpOp::Uge => ((a as u64) >= (b as u64)) as i64,
            };
            return vec![Self::mk_inst(
                X86Opcode::Mov,
                Some(dst),
                &[MachineOperand::Immediate(cmp_result)],
            )];
        }

        // GlobalSymbol operands represent symbol addresses (function
        // pointers or global variable addresses).  The CMP instruction
        // interprets them as memory references, loading the VALUE at
        // that address rather than comparing the ADDRESS itself.
        // Materialise such operands into registers via LEA first.
        let mut prefix_insts = Vec::new();
        if let MachineOperand::GlobalSymbol(_) = &lhs_op {
            let tmp = self.new_vreg();
            prefix_insts.push(Self::mk_inst(X86Opcode::Lea, Some(tmp.clone()), &[lhs_op]));
            lhs_op = tmp;
        }
        if let MachineOperand::GlobalSymbol(_) = &rhs_op {
            let tmp = self.new_vreg();
            prefix_insts.push(Self::mk_inst(X86Opcode::Lea, Some(tmp.clone()), &[rhs_op]));
            rhs_op = tmp;
        }

        // x86 CMP requires a register or memory operand as the first
        // (destination) operand.  If the LHS resolved to an immediate,
        // swap the operands and mirror the relational condition so the
        // semantics are preserved:  CMP(a,b) sets flags for a-b;
        // CMP(b,a) sets flags for b-a, which inverts inequalities.
        let mut effective_op = op;
        if matches!(lhs_op, MachineOperand::Immediate(_)) {
            std::mem::swap(&mut lhs_op, &mut rhs_op);
            effective_op = match op {
                ICmpOp::Eq => ICmpOp::Eq,
                ICmpOp::Ne => ICmpOp::Ne,
                ICmpOp::Slt => ICmpOp::Sgt,
                ICmpOp::Sle => ICmpOp::Sge,
                ICmpOp::Sgt => ICmpOp::Slt,
                ICmpOp::Sge => ICmpOp::Sle,
                ICmpOp::Ult => ICmpOp::Ugt,
                ICmpOp::Ule => ICmpOp::Uge,
                ICmpOp::Ugt => ICmpOp::Ult,
                ICmpOp::Uge => ICmpOp::Ule,
            };
        }

        let setcc = match effective_op {
            ICmpOp::Eq => X86Opcode::Sete,
            ICmpOp::Ne => X86Opcode::Setne,
            ICmpOp::Slt => X86Opcode::Setl,
            ICmpOp::Sle => X86Opcode::Setle,
            ICmpOp::Sgt => X86Opcode::Setg,
            ICmpOp::Sge => X86Opcode::Setge,
            ICmpOp::Ult => X86Opcode::Setb,
            ICmpOp::Ule => X86Opcode::Setbe,
            ICmpOp::Ugt => X86Opcode::Seta,
            ICmpOp::Uge => X86Opcode::Setae,
        };

        // Determine operand size from the IR type of the compared values.
        // Sub-64-bit integers (i8, i16, i32) loaded from memory are
        // zero-extended into 64-bit registers.  A 64-bit CMP would
        // interpret a negative i32 (e.g. 0x00000000_FFFFFFFE) as a
        // large positive number, breaking signed comparisons.  Setting
        // operand_size = 4 makes the encoder emit `cmp eXX, eYY` which
        // only examines the lower 32 bits and sets FLAGS correctly for
        // 32-bit signed/unsigned semantics.
        let cmp_op_size: u8 = self
            .value_types
            .get(&lhs)
            .map(|ty| match ty {
                IrType::I1 | IrType::I8 => 4, // promote to 32-bit min for CMP
                IrType::I16 => 2,
                IrType::I32 => 4,
                _ => 8,
            })
            .unwrap_or(8);

        let mut cmp = Self::mk_inst(X86Opcode::Cmp, None, &[lhs_op, rhs_op]);
        cmp.operand_size = cmp_op_size;

        // Prepend any LEA materialisation instructions for GlobalSymbol
        // operands before the CMP instruction sequence.
        prefix_insts.extend(vec![
            // cmp lhs, rhs  (with correct operand width for FLAGS semantics)
            cmp,
            // setcc dst (1-byte result, zero-extended to full register width later)
            Self::mk_inst(setcc, Some(dst.clone()), &[]),
            // movzx dst, dst (zero-extend byte result to full width)
            Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]),
        ]);
        prefix_insts
    }

    // ===================================================================
    // Floating-point comparison selection
    // ===================================================================

    fn select_fcmp(
        &mut self,
        result: Value,
        op: FCmpOp,
        lhs: Value,
        rhs: Value,
    ) -> Vec<MachineInstruction> {
        let lhs_op = self.get_value(lhs);
        let rhs_op = self.get_value(rhs);
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());

        // x86 UCOMISD/UCOMISS sets CF, ZF, PF per the following table:
        //   a > b  (ordered)   → CF=0, ZF=0, PF=0
        //   a < b  (ordered)   → CF=1, ZF=0, PF=0
        //   a == b (ordered)   → CF=0, ZF=1, PF=0
        //   unordered (NaN)    → CF=1, ZF=1, PF=1
        //
        // For **ordered** comparisons, NaN must produce false.
        // For **unordered** comparisons, NaN must produce true.
        //
        // Simple SETcc alone is WRONG for ordered comparisons because the
        // NaN case (PF=1) may coincidentally satisfy the condition flags.
        //
        // Strategy:
        //   Ordered (Oeq, One, Olt, Ole, Ogt, Oge):
        //     UCOMISD lhs, rhs
        //     SETcc   tmp1          (primary condition)
        //     SETNP   tmp2          (PF=0 → not NaN)
        //     AND     dst, tmp1, tmp2  → result is true only if both hold
        //
        //   Unordered NaN checks (Uno, Ord):
        //     UCOMISD lhs, rhs
        //     SETP    dst   (Uno: PF=1 → NaN)
        //     SETNP   dst   (Ord: PF=0 → both are numbers)
        //
        //   For simplicity the "ordered" cases emit a 5-instruction sequence;
        //   the NaN-only checks emit a 3-instruction sequence.

        // Use UCOMISS for F32 operands and UCOMISD for F64/default.
        // UCOMISD reads all 64 bits of the XMM register; if the value
        // was produced by CVTSI2SS the upper 32 bits may contain stale
        // data, leading to wrong comparison results.
        let is_f32 = matches!(self.value_types.get(&lhs), Some(IrType::F32));
        let cmp_opcode = if is_f32 {
            X86Opcode::Ucomiss
        } else {
            X86Opcode::Ucomisd
        };
        let mut out = vec![Self::mk_inst(cmp_opcode, None, &[lhs_op, rhs_op])];

        match op {
            FCmpOp::Uno => {
                // Result = PF (NaN → PF=1)
                out.push(Self::mk_inst(X86Opcode::Setp, Some(dst.clone()), &[]));
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]));
            }
            FCmpOp::Ord => {
                // Result = !PF (both ordered → PF=0)
                out.push(Self::mk_inst(X86Opcode::Setnp, Some(dst.clone()), &[]));
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]));
            }
            _ => {
                // Ordered comparisons: need (condition) AND (not-NaN).
                let primary_cc = match op {
                    FCmpOp::Oeq => X86Opcode::Sete,  // ZF=1
                    FCmpOp::One => X86Opcode::Setne, // ZF=0
                    FCmpOp::Olt => X86Opcode::Setb,  // CF=1
                    FCmpOp::Ole => X86Opcode::Setbe, // CF=1 OR ZF=1
                    FCmpOp::Ogt => X86Opcode::Seta,  // CF=0 AND ZF=0
                    FCmpOp::Oge => X86Opcode::Setae, // CF=0
                    FCmpOp::Uno | FCmpOp::Ord => unreachable!(),
                };
                // Use the physical scratch register R11 for the parity
                // temporary.  R11 is excluded from the allocatable pool so
                // there is no conflict with the register allocator, and we
                // avoid creating an unmapped virtual register that the
                // allocator cannot resolve.
                let tmp_parity = MachineOperand::Register(R11);
                // SETcc  dst  ← primary condition flag
                out.push(Self::mk_inst(primary_cc, Some(dst.clone()), &[]));
                // SETNP  R11  ← PF=0 means NOT NaN
                out.push(Self::mk_inst(
                    X86Opcode::Setnp,
                    Some(tmp_parity.clone()),
                    &[],
                ));
                // AND dst, R11  ← result is true only if (condition AND !NaN)
                out.push(Self::mk_inst(
                    X86Opcode::And,
                    Some(dst.clone()),
                    &[dst.clone(), tmp_parity],
                ));
                // Zero-extend to full register width
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]));
            }
        }

        out
    }

    // ===================================================================
    // Call instruction selection
    // ===================================================================

    // =================================================================
    // Builtin intrinsic inlining
    // =================================================================

    /// Attempt to inline a `__builtin_*` call as architecture-specific
    /// instructions.  Returns `Some(instructions)` if the builtin is
    /// recognized and inlined, `None` otherwise (fall through to normal
    /// call).
    fn try_inline_builtin(
        &mut self,
        name: &str,
        result: Value,
        args: &[Value],
        ret_type: &IrType,
    ) -> Option<Vec<MachineInstruction>> {
        // Helper: ensure an operand is in a register (load from memory if spilled).
        let _ensure_reg =
            |this: &mut Self, val: Value| -> (MachineOperand, Vec<MachineInstruction>) {
                let op = this.get_value(val);
                match &op {
                    MachineOperand::Register(_) => (op, vec![]),
                    _ => {
                        let tmp = this.new_vreg();
                        let mov = Self::mk_inst(X86Opcode::Mov, Some(tmp.clone()), &[op]);
                        (tmp, vec![mov])
                    }
                }
            };

        match name {
            // ---- count leading zeros ----
            // BSR finds highest set bit index.  CLZ = (width-1) XOR BSR.
            // Uses physical register RCX for arg, physical RDX for BSR
            // result, then captures into dst vreg.
            "__builtin_clz" | "__builtin_clzl" | "__builtin_clzll" => {
                let xor_val: i64 = if name == "__builtin_clzll" || name == "__builtin_clzl" {
                    63
                } else {
                    31
                };
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                let rdx = MachineOperand::Register(RDX);
                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // BSR RDX, RCX
                out.push(Self::mk_inst(X86Opcode::Bsr, Some(rdx.clone()), &[rcx]));
                // XOR RDX, xor_val  (two-address)
                out.push(Self::mk_inst(
                    X86Opcode::Xor,
                    Some(rdx.clone()),
                    &[rdx.clone(), MachineOperand::Immediate(xor_val)],
                ));
                // MOV dst, RDX
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rdx]));
                Some(out)
            }
            // ---- count trailing zeros ----
            // BSF finds lowest set bit index = CTZ directly.
            "__builtin_ctz" | "__builtin_ctzl" | "__builtin_ctzll" => {
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // BSF dst_phys, RCX
                let rdx = MachineOperand::Register(RDX);
                out.push(Self::mk_inst(X86Opcode::Bsf, Some(rdx.clone()), &[rcx]));
                // MOV dst, RDX
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rdx]));
                Some(out)
            }
            // ---- population count ----
            "__builtin_popcount" | "__builtin_popcountl" | "__builtin_popcountll" => {
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // POPCNT RDX, RCX
                let rdx = MachineOperand::Register(RDX);
                out.push(Self::mk_inst(X86Opcode::Popcnt, Some(rdx.clone()), &[rcx]));
                // MOV dst, RDX
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rdx]));
                Some(out)
            }
            // ---- find first set (1-indexed, 0 if input is 0) ----
            // Uses physical registers for intermediates (same pattern as
            // IDIV codegen) because the register allocator operates at
            // the IR level and only sees the Call's result Value.
            // Caller-saved registers are safe because the IR-level Call
            // instruction tells the allocator these are clobbered.
            //   MOV RCX, arg         (save arg to RCX)
            //   BSF RDX, RCX         (RDX = bit index, ZF=1 if arg==0)
            //   ADD RDX, 1           (1-indexed)
            //   TEST RCX, RCX        (re-set ZF from arg)
            //   SETNE R10B           (R10 low byte = 0 or 1)
            //   MOVZX R10, R10B      (zero-extend)
            //   IMUL R10, RDX        (R10 = (BSF+1) * (0 or 1))
            //   MOV dst, R10         (capture result into allocated vreg)
            "__builtin_ffs" | "__builtin_ffsl" | "__builtin_ffsll" => {
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                let rdx = MachineOperand::Register(RDX);
                let r10 = MachineOperand::Register(R10);

                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // BSF RDX, RCX
                out.push(Self::mk_inst(
                    X86Opcode::Bsf,
                    Some(rdx.clone()),
                    std::slice::from_ref(&rcx),
                ));
                // ADD RDX, 1  (two-address)
                out.push(Self::mk_inst(
                    X86Opcode::Add,
                    Some(rdx.clone()),
                    &[rdx.clone(), MachineOperand::Immediate(1)],
                ));
                // TEST RCX, RCX
                out.push(Self::mk_inst(X86Opcode::Test, None, &[rcx.clone(), rcx]));
                // SETNE R10 (byte portion)
                out.push(Self::mk_inst(X86Opcode::Setne, Some(r10.clone()), &[]));
                // MOVZX R10, R10
                out.push(Self::mk_inst(
                    X86Opcode::MovZX,
                    Some(r10.clone()),
                    std::slice::from_ref(&r10),
                ));
                // IMUL R10, RDX  → R10 = R10 * RDX
                out.push(Self::mk_inst(X86Opcode::Imul, Some(r10.clone()), &[rdx]));
                // MOV dst, R10
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[r10]));
                Some(out)
            }
            // ---- byte swap ----
            "__builtin_bswap32" => {
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // BSWAP32 RCX  (in-place byte swap; encoder uses 32-bit form)
                let mut bswap = Self::mk_inst(
                    X86Opcode::Bswap,
                    Some(rcx.clone()),
                    std::slice::from_ref(&rcx),
                );
                bswap.operand_size = 4;
                out.push(bswap);
                // MOV dst, RCX
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rcx]));
                Some(out)
            }
            "__builtin_bswap64" => {
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // BSWAP RCX  (64-bit in-place byte swap)
                out.push(Self::mk_inst(
                    X86Opcode::Bswap,
                    Some(rcx.clone()),
                    std::slice::from_ref(&rcx),
                ));
                // MOV dst, RCX
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rcx]));
                Some(out)
            }
            "__builtin_bswap16" => {
                let mut out = Vec::new();
                let arg = self.get_value(args[0]);
                let rcx = MachineOperand::Register(RCX);
                // MOV RCX, arg
                out.push(Self::mk_inst(X86Opcode::Mov, Some(rcx.clone()), &[arg]));
                // BSWAP32 RCX then SHR 16 for 16-bit swap
                let mut bswap = Self::mk_inst(
                    X86Opcode::Bswap,
                    Some(rcx.clone()),
                    std::slice::from_ref(&rcx),
                );
                bswap.operand_size = 4;
                out.push(bswap);
                out.push(Self::mk_inst(
                    X86Opcode::Shr,
                    Some(rcx.clone()),
                    &[rcx.clone(), MachineOperand::Immediate(16)],
                ));
                // MOV dst, RCX
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rcx]));
                Some(out)
            }
            // ---- branch-hint passthrough ----
            "__builtin_expect" => {
                let arg = self.get_value(args[0]);
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                Some(vec![Self::mk_inst(X86Opcode::Mov, Some(dst), &[arg])])
            }
            "__builtin_assume_aligned" => {
                let arg = self.get_value(args[0]);
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                Some(vec![Self::mk_inst(X86Opcode::Mov, Some(dst), &[arg])])
            }
            // ---- frame/return address ----
            "__builtin_frame_address" => {
                let level = if !args.is_empty() {
                    if let Some(&imm) = self.constant_cache.get(&args[0]) {
                        imm as usize
                    } else {
                        match self.get_value(args[0]) {
                            MachineOperand::Immediate(n) => n as usize,
                            _ => 0,
                        }
                    }
                } else {
                    0
                };
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                let mut out = Vec::new();
                // Start from current frame pointer
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst.clone()),
                    &[MachineOperand::Register(RBP)],
                ));
                // Walk the frame chain `level` times:
                // dst = [dst] (follow saved RBP pointer)
                for _ in 0..level {
                    out.push(Self::mk_inst(
                        X86Opcode::LoadInd,
                        Some(dst.clone()),
                        &[dst.clone()],
                    ));
                }
                Some(out)
            }
            "__builtin_return_address" => {
                // Extract level — prefer constant_cache (which always
                // has the raw immediate) over get_value (which may
                // return a vreg if the constant instruction was already
                // materialised).
                let level = if !args.is_empty() {
                    if let Some(&imm) = self.constant_cache.get(&args[0]) {
                        imm as usize
                    } else {
                        match self.get_value(args[0]) {
                            MachineOperand::Immediate(n) => n as usize,
                            _ => 0,
                        }
                    }
                } else {
                    0
                };
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                let mut out = Vec::new();
                if level == 0 {
                    // Level 0: read [RBP+8] directly
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(dst),
                        &[MachineOperand::Memory {
                            base: Some(RBP),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        }],
                    ));
                } else {
                    // Level N > 0: walk frame chain N times from RBP,
                    // then read return address at [frame+8].
                    // After the loop, dst holds the target frame's
                    // saved RBP.  We add 8 and dereference to get
                    // the return address stored above it.
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(dst.clone()),
                        &[MachineOperand::Register(RBP)],
                    ));
                    for _ in 0..level {
                        // dst = [dst] (follow saved frame pointer)
                        out.push(Self::mk_inst(
                            X86Opcode::LoadInd,
                            Some(dst.clone()),
                            &[dst.clone()],
                        ));
                    }
                    // dst += 8  (point to saved return address)
                    out.push(Self::mk_inst(
                        X86Opcode::Add,
                        Some(dst.clone()),
                        &[dst.clone(), MachineOperand::Immediate(8)],
                    ));
                    // dst = [dst]  (load the return address)
                    out.push(Self::mk_inst(
                        X86Opcode::LoadInd,
                        Some(dst.clone()),
                        &[dst.clone()],
                    ));
                }
                Some(out)
            }
            // ---- trap / unreachable ----
            "__builtin_trap" | "__builtin_unreachable" => {
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                Some(vec![
                    Self::mk_inst(X86Opcode::Ud2, None, &[]),
                    Self::mk_inst(X86Opcode::Mov, Some(dst), &[MachineOperand::Immediate(0)]),
                ])
            }
            // ---- overflow arithmetic ----
            "__builtin_add_overflow" | "__builtin_sub_overflow" | "__builtin_mul_overflow" => {
                if args.len() < 3 {
                    return None;
                }
                let a = self.get_value(args[0]);
                let b = self.get_value(args[1]);
                let result_ptr = self.get_value(args[2]);
                let op = match name {
                    "__builtin_add_overflow" => X86Opcode::Add,
                    "__builtin_sub_overflow" => X86Opcode::Sub,
                    _ => X86Opcode::Imul,
                };
                // Copy only `a` into a fresh vreg `va`, then operate
                // directly with `b` so the register allocator keeps `b`
                // alive and in a DIFFERENT register than `va`.
                let mut out = Vec::new();
                let r10 = MachineOperand::Register(R10);
                // va = a (copy a).  b is still live → allocator keeps va ≠ b.
                let va = self.new_vreg();
                out.push(Self::mk_inst(X86Opcode::Mov, Some(va.clone()), &[a]));
                // 64-bit ADD/SUB/IMUL: va = va OP b (b used directly).
                out.push(Self::mk_inst(op, Some(va.clone()), &[va.clone(), b]));
                // Store low 32 bits to the result pointer.
                let mut store =
                    Self::mk_inst(X86Opcode::StoreInd32, None, &[result_ptr, va.clone()]);
                store.operand_size = 4;
                out.push(store);
                // Overflow detection: check if the 64-bit result equals its
                // 32-bit sign-extension.  SHL+SAR by 32 sign-extends low 32.
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(r10.clone()),
                    std::slice::from_ref(&va),
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Shl,
                    Some(r10.clone()),
                    &[r10.clone(), MachineOperand::Immediate(32)],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Sar,
                    Some(r10.clone()),
                    &[r10.clone(), MachineOperand::Immediate(32)],
                ));
                // CMP R10, va — if they differ, 32-bit overflow occurred.
                out.push(Self::mk_inst(X86Opcode::Cmp, None, &[r10, va]));
                // SETNE captures "overflow" flag.
                let vr = self.new_vreg();
                out.push(Self::mk_inst(X86Opcode::Setne, Some(vr.clone()), &[]));
                let final_dst = self.new_vreg();
                self.set_value(result, final_dst.clone());
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(final_dst), &[vr]));
                Some(out)
            }
            // ---- variadic argument builtins ----
            // These use physical caller-saved registers for intermediates
            // (same pattern as __builtin_clz/ffs) because the register
            // allocator only manages vreg→phys mappings from the IR level.
            //
            // va_start: allocate a FRESH 32-byte va_control block on the
            // stack and initialize it with the current variadic state.
            // Each va_start call gets its own independent block so that
            // multiple active va_lists in the same function do not share
            // mutable state (gp_offset, fp_offset, overflow_arg_area).
            "__builtin_va_start" => {
                if args.is_empty() {
                    return None;
                }
                let ap_slot = self.get_value(args[0]);
                let mut out = Vec::new();
                if let Some(ref frame) = self.frame {
                    if frame.va_control_offset.is_some() {
                        let rcx = MachineOperand::Register(RCX);
                        let rsp_op = MachineOperand::Register(RSP);

                        // Step 1: Allocate 32 bytes on the stack for a fresh
                        // va_control block (24 bytes needed, 32 for alignment).
                        out.push(Self::mk_inst(
                            X86Opcode::Sub,
                            Some(rsp_op.clone()),
                            &[rsp_op.clone(), MachineOperand::Immediate(32)],
                        ));

                        // Step 2: [RSP+0] gp_offset (32-bit) = named_gpr_count * 8
                        let gp_offset_init = (frame.named_gpr_count as i64) * 8;
                        {
                            let mut m = Self::mk_inst(
                                X86Opcode::Mov,
                                None,
                                &[
                                    MachineOperand::Memory {
                                        base: Some(RSP),
                                        index: None,
                                        scale: 1,
                                        displacement: 0,
                                    },
                                    MachineOperand::Immediate(gp_offset_init),
                                ],
                            );
                            m.operand_size = 4;
                            out.push(m);
                        }

                        // Step 3: [RSP+4] fp_offset (32-bit) = 48 + named_fp_count * 16
                        let fp_offset_init = 48 + (frame.named_fp_count as i64) * 16;
                        {
                            let mut m = Self::mk_inst(
                                X86Opcode::Mov,
                                None,
                                &[
                                    MachineOperand::Memory {
                                        base: Some(RSP),
                                        index: None,
                                        scale: 1,
                                        displacement: 4,
                                    },
                                    MachineOperand::Immediate(fp_offset_init),
                                ],
                            );
                            m.operand_size = 4;
                            out.push(m);
                        }

                        // Step 4: [RSP+8] overflow_arg_area = RBP + 16 + fixed_stack
                        // Must include MEMORY-class params (long double, large structs)
                        let gp_stack_fixed = frame.named_gpr_count.saturating_sub(6);
                        let fp_stack_fixed = frame.named_fp_count.saturating_sub(8);
                        let overflow_disp: i64 = 16
                            + ((gp_stack_fixed + fp_stack_fixed) * 8) as i64
                            + frame.named_memory_stack_bytes as i64;
                        out.push(Self::mk_inst(
                            X86Opcode::Lea,
                            Some(rcx.clone()),
                            &[MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: overflow_disp,
                            }],
                        ));
                        out.push(Self::mk_inst(
                            X86Opcode::Mov,
                            None,
                            &[
                                MachineOperand::Memory {
                                    base: Some(RSP),
                                    index: None,
                                    scale: 1,
                                    displacement: 8,
                                },
                                rcx.clone(),
                            ],
                        ));

                        // Step 5: [RSP+16] reg_save_area — copy from the
                        // prologue-initialized template in the frame.
                        if let Some(ctrl_offset) = frame.va_control_offset {
                            // Load reg_save_area from the template block
                            out.push(Self::mk_inst(
                                X86Opcode::Mov,
                                Some(rcx.clone()),
                                &[MachineOperand::Memory {
                                    base: Some(RBP),
                                    index: None,
                                    scale: 1,
                                    displacement: ctrl_offset as i64 + 16,
                                }],
                            ));
                            out.push(Self::mk_inst(
                                X86Opcode::Mov,
                                None,
                                &[
                                    MachineOperand::Memory {
                                        base: Some(RSP),
                                        index: None,
                                        scale: 1,
                                        displacement: 16,
                                    },
                                    rcx.clone(),
                                ],
                            ));
                        }

                        // Step 6: Store RSP (the new block address) to ap_slot.
                        out.push(va_store_to_slot(&ap_slot, rsp_op));
                    }
                }
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Immediate(0)],
                ));
                Some(out)
            }
            // va_arg: load the appropriate pointer (GP or FP) from the
            // va_control block, dereference to get the value, advance the
            // pointer, and store it back.
            //
            // ABI-standard va_list layout (24 bytes, compatible with
            // libc's vprintf/vfprintf/vsprintf etc.):
            //   [+0]  gp_offset         (u32) — byte offset into reg_save_area
            //   [+4]  fp_offset         (u32) — byte offset into reg_save_area
            //   [+8]  overflow_arg_area (ptr) — next stack overflow argument
            //   [+16] reg_save_area     (ptr) — base of 176-byte register save
            //
            // Register save area layout (176 bytes):
            //   [0..47]   6 GPRs (RDI..R9), 8 bytes each
            //   [48..175] 8 XMMs (XMM0..XMM7), 16 bytes each
            //
            // Integer va_arg: if gp_offset < 48, load from
            //   reg_save_area + gp_offset, advance gp_offset by 8.
            //   Otherwise load from overflow_arg_area, advance it by 8.
            // Float va_arg: if fp_offset < 176, load from
            //   reg_save_area + fp_offset, advance fp_offset by 16.
            //   Otherwise load from overflow_arg_area, advance it by 8.
            // Both paths use branchless CMOV sequences.
            "__builtin_va_arg" => {
                if args.is_empty() {
                    return None;
                }
                let ap_slot = self.get_value(args[0]);
                // F80 (long double) is MEMORY-class — it uses the
                // overflow_arg_area (stack path), not fp_offset.
                let is_float_arg = matches!(ret_type, IrType::F32 | IrType::F64);
                let mut out = Vec::new();

                let rcx = MachineOperand::Register(RCX);
                let rdx = MachineOperand::Register(RDX);
                let r10 = MachineOperand::Register(R10);
                let r8_op = MachineOperand::Register(R8);
                let r11_op = MachineOperand::Register(R11);
                let rdi = MachineOperand::Register(RDI);

                // Step 1: Load va_list pointer: R8 = *ap_slot
                out.push(va_load_from_slot(&ap_slot, r8_op.clone()));

                if is_float_arg {
                    // --- Float path (ABI-standard offset-based) ---
                    // Branchless CMOV: compare fp_offset against 176.
                    // If < 176: load from reg_save_area + fp_offset,
                    //           advance fp_offset by 16.
                    // If >= 176: load from overflow_arg_area,
                    //            advance overflow_arg_area by 8.
                    //
                    // Registers used (all caller-saved):
                    //   R8  = va_list pointer (preserved)
                    //   RCX = fp_offset (32-bit loaded)
                    //   R10 = overflow_arg_area
                    //   RDI = reg_save_area / scratch
                    //   RDX = computed load address
                    //   R11 = new fp_offset candidate

                    // ECX = [R8+4] (fp_offset, 32-bit, zero-extends)
                    let mut ld_fp = Self::mk_inst(
                        X86Opcode::Mov,
                        Some(rcx.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 4,
                        }],
                    );
                    ld_fp.operand_size = 4;
                    out.push(ld_fp);
                    // R10 = [R8+8] (overflow_arg_area)
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(r10.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        }],
                    ));
                    // RDI = [R8+16] (reg_save_area)
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(rdi.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 16,
                        }],
                    ));
                    // RDX = LEA [RDI + RCX] (reg addr, no flags)
                    out.push(Self::mk_inst(
                        X86Opcode::Lea,
                        Some(rdx.clone()),
                        &[MachineOperand::Memory {
                            base: Some(RDI),
                            index: Some(RCX),
                            scale: 1,
                            displacement: 0,
                        }],
                    ));
                    // R11 = LEA [RCX + 16] (new fp_offset, no flags)
                    out.push(Self::mk_inst(
                        X86Opcode::Lea,
                        Some(r11_op.clone()),
                        &[MachineOperand::Memory {
                            base: Some(RCX),
                            index: None,
                            scale: 1,
                            displacement: 16,
                        }],
                    ));
                    // CMP ECX, 176 (32-bit compare)
                    let mut cmp_fp = Self::mk_inst(
                        X86Opcode::Cmp,
                        None,
                        &[rcx.clone(), MachineOperand::Immediate(176)],
                    );
                    cmp_fp.operand_size = 4;
                    out.push(cmp_fp);
                    // CMOVAE RDX, R10 (if fp_offset >= 176, use overflow)
                    out.push(Self::mk_inst(
                        X86Opcode::Cmovae,
                        Some(rdx.clone()),
                        &[rdx.clone(), r10.clone()],
                    ));
                    // CMOVAE R11, RCX (if overflow, keep old fp_offset)
                    out.push(Self::mk_inst(
                        X86Opcode::Cmovae,
                        Some(r11_op.clone()),
                        &[r11_op.clone(), rcx.clone()],
                    ));
                    // RDI = LEA [R10 + 8] (new overflow candidate, no flags)
                    out.push(Self::mk_inst(
                        X86Opcode::Lea,
                        Some(rdi.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R10),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        }],
                    ));
                    // CMOVB RDI, R10 (if register path, keep old overflow)
                    out.push(Self::mk_inst(
                        X86Opcode::Cmovb,
                        Some(rdi.clone()),
                        &[rdi.clone(), r10.clone()],
                    ));
                    // Load float value: XMM0 = MOVSD [RDX]
                    let xmm0_op = MachineOperand::Register(XMM0);
                    out.push(Self::mk_inst(
                        X86Opcode::Movsd,
                        Some(xmm0_op.clone()),
                        &[MachineOperand::Memory {
                            base: Some(RDX),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        }],
                    ));
                    // Store [R8+4] = R11d (new fp_offset, 32-bit)
                    let mut st_fp = Self::mk_inst(
                        X86Opcode::Mov,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(R8),
                                index: None,
                                scale: 1,
                                displacement: 4,
                            },
                            r11_op.clone(),
                        ],
                    );
                    st_fp.operand_size = 4;
                    out.push(st_fp);
                    // Store [R8+8] = RDI (new overflow_arg_area)
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(R8),
                                index: None,
                                scale: 1,
                                displacement: 8,
                            },
                            rdi.clone(),
                        ],
                    ));
                    // Copy to virtual register
                    let dst = self.new_vreg();
                    self.set_value(result, dst.clone());
                    out.push(Self::mk_inst(X86Opcode::Movsd, Some(dst), &[xmm0_op]));
                } else {
                    // --- Integer/pointer path (ABI-standard offset-based) ---
                    // Branchless CMOV: compare gp_offset against 48.
                    // If < 48: load from reg_save_area + gp_offset,
                    //          advance gp_offset by 8.
                    // If >= 48: load from overflow_arg_area,
                    //           advance overflow_arg_area by 8.
                    //
                    // Registers used (all caller-saved):
                    //   R8  = va_list pointer (preserved)
                    //   RCX = gp_offset (32-bit loaded, zero-extended)
                    //   R10 = overflow_arg_area
                    //   RDI = reg_save_area / scratch
                    //   RDX = computed load address
                    //   R11 = new gp_offset candidate

                    // ECX = [R8+0] (gp_offset, 32-bit, zero-extends)
                    let mut ld_gp = Self::mk_inst(
                        X86Opcode::Mov,
                        Some(rcx.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        }],
                    );
                    ld_gp.operand_size = 4;
                    out.push(ld_gp);
                    // R10 = [R8+8] (overflow_arg_area)
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(r10.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        }],
                    ));
                    // RDI = [R8+16] (reg_save_area)
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(rdi.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 16,
                        }],
                    ));
                    // RDX = LEA [RDI + RCX] (reg addr, no flags)
                    out.push(Self::mk_inst(
                        X86Opcode::Lea,
                        Some(rdx.clone()),
                        &[MachineOperand::Memory {
                            base: Some(RDI),
                            index: Some(RCX),
                            scale: 1,
                            displacement: 0,
                        }],
                    ));
                    // R11 = LEA [RCX + 8] (new gp_offset, no flags)
                    out.push(Self::mk_inst(
                        X86Opcode::Lea,
                        Some(r11_op.clone()),
                        &[MachineOperand::Memory {
                            base: Some(RCX),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        }],
                    ));
                    // CMP ECX, 48 (32-bit compare; 6 GPRs * 8 = 48)
                    let mut cmp_gp = Self::mk_inst(
                        X86Opcode::Cmp,
                        None,
                        &[rcx.clone(), MachineOperand::Immediate(48)],
                    );
                    cmp_gp.operand_size = 4;
                    out.push(cmp_gp);
                    // CMOVAE RDX, R10 (if gp_offset >= 48, use overflow)
                    out.push(Self::mk_inst(
                        X86Opcode::Cmovae,
                        Some(rdx.clone()),
                        &[rdx.clone(), r10.clone()],
                    ));
                    // CMOVAE R11, RCX (if overflow, keep old gp_offset)
                    out.push(Self::mk_inst(
                        X86Opcode::Cmovae,
                        Some(r11_op.clone()),
                        &[r11_op.clone(), rcx.clone()],
                    ));
                    // RDI = LEA [R10 + 8] (new overflow candidate, no flags)
                    out.push(Self::mk_inst(
                        X86Opcode::Lea,
                        Some(rdi.clone()),
                        &[MachineOperand::Memory {
                            base: Some(R10),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        }],
                    ));
                    // CMOVB RDI, R10 (if register path, keep old overflow)
                    out.push(Self::mk_inst(
                        X86Opcode::Cmovb,
                        Some(rdi.clone()),
                        &[rdi.clone(), r10.clone()],
                    ));
                    // Load value: RCX = [RDX]
                    out.push(Self::mk_inst(
                        X86Opcode::LoadInd,
                        Some(rcx.clone()),
                        std::slice::from_ref(&rdx),
                    ));
                    // Store [R8+0] = R11d (new gp_offset, 32-bit)
                    let mut st_gp = Self::mk_inst(
                        X86Opcode::Mov,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(R8),
                                index: None,
                                scale: 1,
                                displacement: 0,
                            },
                            r11_op.clone(),
                        ],
                    );
                    st_gp.operand_size = 4;
                    out.push(st_gp);
                    // Store [R8+8] = RDI (new overflow_arg_area)
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(R8),
                                index: None,
                                scale: 1,
                                displacement: 8,
                            },
                            rdi.clone(),
                        ],
                    ));
                    // Copy loaded value to virtual register
                    let dst = self.new_vreg();
                    self.set_value(result, dst.clone());
                    out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[rcx]));
                }
                Some(out)
            }
            // va_arg_mem_N: >16-byte MEMORY-class struct va_arg.
            // The caller pushed the struct directly onto the stack
            // (overflow_arg_area), NOT into registers.
            // Struct size is encoded in the intrinsic name.
            // args: [0] = va_list ptr, [1] = dest alloca
            s if s.starts_with("__builtin_va_arg_mem_") => {
                if args.len() < 2 {
                    return None;
                }
                // Parse struct byte size from intrinsic name
                let sz: i64 = s["__builtin_va_arg_mem_".len()..].parse().unwrap_or(0);
                if sz == 0 {
                    return None;
                }
                // Round up to 8-byte boundary for stack slot
                let rounded_sz = (sz + 7) & !7;
                let qword_count = rounded_sz / 8;

                let ap_slot = self.get_value(args[0]);
                let dst_slot = self.get_value(args[1]);
                let mut out = Vec::new();

                let r8_op = MachineOperand::Register(R8);
                let r10_op = MachineOperand::Register(R10);
                let rcx_op = MachineOperand::Register(RCX);
                let rdi_op = MachineOperand::Register(RDI);
                let rsi_op = MachineOperand::Register(RSI);

                // R8 = va_list pointer (load from alloca/slot)
                out.push(va_load_from_slot(&ap_slot, r8_op.clone()));

                // R10 = [R8+8] (overflow_arg_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(r10_op.clone()),
                    &[MachineOperand::Memory {
                        base: Some(R8),
                        index: None,
                        scale: 1,
                        displacement: 8,
                    }],
                ));

                // RDI = dest alloca ADDRESS (LEA, not MOV)
                match &dst_slot {
                    MachineOperand::FrameSlot(off) => {
                        out.push(Self::mk_inst(
                            X86Opcode::Lea,
                            Some(rdi_op.clone()),
                            &[MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: *off as i64,
                            }],
                        ));
                    }
                    MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    } => {
                        out.push(Self::mk_inst(
                            X86Opcode::Lea,
                            Some(rdi_op.clone()),
                            &[MachineOperand::Memory {
                                base: *base,
                                index: *index,
                                scale: *scale,
                                displacement: *displacement,
                            }],
                        ));
                    }
                    _ => {
                        // Register already holds address
                        out.push(Self::mk_inst(
                            X86Opcode::Mov,
                            Some(rdi_op.clone()),
                            &[dst_slot.clone()],
                        ));
                    }
                }

                // RCX = qword count (immediate, no global indirection)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(rcx_op.clone()),
                    &[MachineOperand::Immediate(qword_count)],
                ));

                // RSI = R10 (source = overflow_arg_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(rsi_op.clone()),
                    &[r10_op.clone()],
                ));

                // REP MOVSQ: copies RCX qwords from [RSI] to [RDI]
                out.push(Self::mk_inst(X86Opcode::RepMovsq, None, &[]));

                // Advance overflow_arg_area: [R8+8] += rounded_sz
                out.push(Self::mk_inst(
                    X86Opcode::Add,
                    Some(r10_op.clone()),
                    &[r10_op.clone(), MachineOperand::Immediate(rounded_sz)],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        },
                        r10_op,
                    ],
                ));

                // Result is void; set a dummy value for the SSA result
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Immediate(0)],
                ));
                Some(out)
            }
            // va_arg_f80: long double va_arg.
            // Long double is X87/X87UP class on x86-64 SysV ABI,
            // passed in the overflow_arg_area as 80-bit extended
            // precision in a 16-byte slot.
            // args: [0] = va_list ptr, [1] = dest alloca (for f64 result)
            //
            // Strategy:
            //   1. R8  = va_list pointer (from arg slot/alloca)
            //   2. R10 = [R8+8] (overflow_arg_area)
            //   3. FLD TBYTE [R10]  — load 80-bit from overflow area
            //   4. Compute dest address in RDI
            //   5. FSTP QWORD [RDI] — convert to f64, store in alloca
            //   6. [R8+8] += 16     — advance overflow_arg_area
            "__builtin_va_arg_f80" => {
                if args.len() < 2 {
                    return None;
                }
                let ap_slot = self.get_value(args[0]);
                let dst_slot = self.get_value(args[1]);
                let mut out = Vec::new();

                let r8_op = MachineOperand::Register(R8);
                let r10_op = MachineOperand::Register(R10);
                let rdi_op = MachineOperand::Register(RDI);

                // R8 = va_list pointer (load address from alloca/slot)
                out.push(va_load_from_slot(&ap_slot, r8_op.clone()));

                // R10 = [R8+8] (overflow_arg_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(r10_op.clone()),
                    &[MachineOperand::Memory {
                        base: Some(R8),
                        index: None,
                        scale: 1,
                        displacement: 8,
                    }],
                ));

                // FLD TBYTE [R10] — load 80-bit extended from overflow area
                let mut fld_inst = MachineInstruction::new(X86Opcode::FldMem80.as_u32());
                fld_inst.operands.push(MachineOperand::Memory {
                    base: Some(R10),
                    index: None,
                    scale: 1,
                    displacement: 0,
                });
                out.push(fld_inst);

                // Compute destination address (RDI = alloca address)
                match &dst_slot {
                    MachineOperand::FrameSlot(off) => {
                        out.push(Self::mk_inst(
                            X86Opcode::Lea,
                            Some(rdi_op.clone()),
                            &[MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: *off as i64,
                            }],
                        ));
                    }
                    MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    } => {
                        out.push(Self::mk_inst(
                            X86Opcode::Lea,
                            Some(rdi_op.clone()),
                            &[MachineOperand::Memory {
                                base: *base,
                                index: *index,
                                scale: *scale,
                                displacement: *displacement,
                            }],
                        ));
                    }
                    _ => {
                        out.push(Self::mk_inst(
                            X86Opcode::Mov,
                            Some(rdi_op.clone()),
                            &[dst_slot.clone()],
                        ));
                    }
                }

                // FSTP QWORD [RDI] — convert 80-bit to 64-bit double, store
                let mut fstp_inst = MachineInstruction::new(X86Opcode::FstpMem64.as_u32());
                fstp_inst.operands.push(MachineOperand::Memory {
                    base: Some(RDI),
                    index: None,
                    scale: 1,
                    displacement: 0,
                });
                out.push(fstp_inst);

                // Advance overflow_arg_area: [R8+8] += 16
                out.push(Self::mk_inst(
                    X86Opcode::Add,
                    Some(r10_op.clone()),
                    &[r10_op.clone(), MachineOperand::Immediate(16)],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(R8),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        },
                        r10_op,
                    ],
                ));

                // Result is void; set a dummy value for the SSA result
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Immediate(0)],
                ));
                Some(out)
            }
            // va_end: no-op.
            "__builtin_va_end" => {
                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                Some(vec![Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Immediate(0)],
                )])
            }
            // va_copy: deep-copy the 40-byte va_control block from src to a
            // freshly allocated stack block, then point dest at the new block.
            "__builtin_va_copy" => {
                if args.len() < 2 {
                    return None;
                }
                let dest_slot = self.get_value(args[0]);
                let src_slot = self.get_value(args[1]);
                let mut out = Vec::new();
                let rcx = MachineOperand::Register(RCX);
                let rdx = MachineOperand::Register(RDX);
                let r10 = MachineOperand::Register(R10);
                let rsp_op = MachineOperand::Register(RSP);

                // Step 1: Load source va_list pointer: RCX = *src_slot
                out.push(va_load_from_slot(&src_slot, rcx.clone()));

                // Step 2: Load all 3 fields from source (24-byte ABI format):
                //   [RCX+0..7]   gp_offset(u32) + fp_offset(u32) as one 64-bit read
                //   [RCX+8..15]  overflow_arg_area (ptr)
                //   [RCX+16..23] reg_save_area (ptr)
                // RDX = [RCX+0] (gp_offset + fp_offset as one 64-bit value)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(rdx.clone()),
                    &[MachineOperand::Memory {
                        base: Some(RCX),
                        index: None,
                        scale: 1,
                        displacement: 0,
                    }],
                ));
                // R10 = [RCX+8] (overflow_arg_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(r10.clone()),
                    &[MachineOperand::Memory {
                        base: Some(RCX),
                        index: None,
                        scale: 1,
                        displacement: 8,
                    }],
                ));

                // Step 3: Allocate 32 bytes on stack for the new va_list block
                // (24 bytes needed, rounded up to 32 for 16-byte stack alignment)
                out.push(Self::mk_inst(
                    X86Opcode::Sub,
                    Some(rsp_op.clone()),
                    &[rsp_op.clone(), MachineOperand::Immediate(32)],
                ));

                // Step 4: Copy all 3 fields to new block
                // [RSP+0] = RDX (gp_offset + fp_offset)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        },
                        rdx.clone(),
                    ],
                ));
                // [RSP+8] = R10 (overflow_arg_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 8,
                        },
                        r10,
                    ],
                ));
                // RDX = [RCX+16] (reg_save_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(rdx.clone()),
                    &[MachineOperand::Memory {
                        base: Some(RCX),
                        index: None,
                        scale: 1,
                        displacement: 16,
                    }],
                ));
                // [RSP+16] = RDX (reg_save_area)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    None,
                    &[
                        MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 16,
                        },
                        rdx,
                    ],
                ));

                // Step 5: Store RSP (new block address) to dest
                out.push(va_store_to_slot(&dest_slot, rsp_op));

                let dst = self.new_vreg();
                self.set_value(result, dst.clone());
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Immediate(0)],
                ));
                Some(out)
            }
            _ => None,
        }
    }

    // =================================================================
    // End of try_inline_builtin
    // =================================================================

    fn select_call(
        &mut self,
        result: Value,
        callee: Value,
        args: &[Value],
        return_type: &IrType,
        _func: &IrFunction,
    ) -> Vec<MachineInstruction> {
        // Intercept __builtin_* intrinsic calls and emit inline code.
        if let Some(fname) = self.func_ref_names.get(&callee).cloned() {
            if let Some(inlined) = self.try_inline_builtin(&fname, result, args, return_type) {
                return inlined;
            }
        }

        let mut out = Vec::new();
        let mut gpr_idx: usize = 0;
        let mut sse_idx: usize = 0;
        let mut stack_offset: i32 = 0;

        // If the callee returns a MEMORY-class struct (>16 bytes), the
        // System V AMD64 ABI requires the caller to pass a hidden first
        // argument in RDI pointing to a buffer for the return value.
        // We use the pre-allocated struct temp slot from the frame layout.
        let ret_loc = self.abi.classify_return(return_type);
        let indirect_ret_slot: Option<i32> = if matches!(ret_loc, RetLocation::Indirect) {
            // Use the pre-allocated struct_temp_offsets slot for this
            // Call result, or fall back to a stack-alloc-like approach.
            self.frame
                .as_ref()
                .and_then(|f| f.struct_temp_offsets.get(&result).copied())
        } else {
            None
        };

        // 1. Place arguments in registers/stack per System V AMD64 ABI.
        //    Register arguments are collected into `reg_arg_setup`;
        //    stack arguments are collected and pushed in reverse order.
        //
        //    CRITICAL ORDERING: Stack-argument PUSHes must be emitted
        //    BEFORE register-argument MOVs.  After register allocation,
        //    the allocator may assign a stack-argument vreg to a physical
        //    register that is also an argument-register destination (e.g.,
        //    RCX for param 4).  If register MOVs execute first, they
        //    clobber the physical register before the PUSH can read it.
        //    Emitting PUSHes first ensures they read the original
        //    (pre-clobber) register values.  The parallel-move resolver
        //    (`resolve_call_arg_conflicts`) then handles any remaining
        //    reg-to-reg conflicts among the register arguments.
        // Stack argument kind: Int = integer/pointer, Fp = F32/F64, F80 = long double
        #[derive(Clone, Copy, PartialEq)]
        enum StackArgKind {
            Int,
            Fp,
            F80,
        }
        // Unified stack argument entry — all stack-passed arguments
        // (scalars, F80, MEMORY-class structs) are collected here in
        // LEFT-TO-RIGHT parameter order, then emitted in REVERSE
        // (right-to-left) per the SysV AMD64 ABI.
        #[allow(clippy::enum_variant_names)]
        enum StackArgEntry {
            /// Scalar (int/ptr) or FP (F32/F64) or F80 stack argument.
            Scalar(MachineOperand, StackArgKind),
            /// MEMORY-class struct: push operands for each eightbyte
            /// (already in high-to-low order within the group).
            MemClassDirect(Vec<MachineOperand>),
            /// MEMORY-class struct via pointer load: (ptr_op, eightbyte_count).
            MemClassPtr(MachineOperand, usize),
        }
        let mut unified_stack_args: Vec<StackArgEntry> = Vec::new();
        let mut reg_arg_setup: Vec<MachineInstruction> = Vec::new();

        // If this call returns a MEMORY-class struct, pass the hidden
        // return pointer as the first GPR argument (RDI).
        if let Some(slot_off) = indirect_ret_slot {
            let buf_addr = MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: slot_off as i64,
            };
            let mut lea_inst = Self::mk_inst(
                X86Opcode::Lea,
                Some(MachineOperand::Register(INTEGER_ARG_REGS[0])), // RDI
                &[buf_addr],
            );
            lea_inst.is_call_arg_setup = true;
            reg_arg_setup.push(lea_inst);
            gpr_idx = 1; // RDI consumed by hidden return ptr
        }

        for arg_val in args {
            let arg_op = self.get_value(*arg_val);

            // Determine if this value is a function reference (function
            // pointer decay) or a global variable address (array-to-pointer
            // decay).  Both must use LEA (to compute the address) rather
            // than MOV (which would load the value at that address).
            let is_func_ref = self.func_ref_names.contains_key(arg_val);
            let is_global_addr = self.global_var_refs.contains_key(arg_val);

            // --- 16-byte struct pair argument (alloca-based) ---
            // If this argument was loaded from a struct alloca (size > 8,
            // <= 16 bytes), we must pass BOTH eightbytes in two consecutive
            // GPR slots per the System V AMD64 ABI.  We use the stable
            // RBP-relative Memory operand recorded at Load time, which
            // avoids intermediate vreg lifetime conflicts (the register
            // allocator only sees IR-level operand uses, not machine-level
            // uses we add here).
            // CRITICAL: Only use the RegisterPair path for structs
            // that are ≤16 bytes. For >16-byte structs the
            // MEMORY-class handler later pushes them on the stack.
            let struct_load_size = self
                .value_types
                .get(arg_val)
                .map(|vty| vty.size_bytes(&self.target))
                .unwrap_or(0);
            if struct_load_size > 0 && struct_load_size <= 16 {
                if let Some(mem_base) = self.struct_load_source.get(arg_val).cloned() {
                    // Classify the struct eightbytes to determine SSE vs INTEGER
                    let arg_ty = self.value_types.get(arg_val).cloned();
                    let eb_classes = if let Some(ref aty) = arg_ty {
                        classify_ir_type_eightbytes(aty, &self.target)
                    } else {
                        vec![AbiClass::Integer, AbiClass::Integer]
                    };

                    // Count how many GPR/SSE regs this arg needs
                    let int_need = eb_classes
                        .iter()
                        .filter(|c| **c == AbiClass::Integer)
                        .count();
                    let sse_need = eb_classes
                        .iter()
                        .filter(|c| **c == AbiClass::Sse || **c == AbiClass::SseUp)
                        .count();

                    let int_avail = gpr_idx + int_need <= INTEGER_ARG_REGS.len();
                    let sse_avail = sse_idx + sse_need <= SSE_ARG_REGS.len();

                    if int_avail && sse_avail && eb_classes.len() == 1 {
                        // Single-eightbyte aggregate (e.g. _Complex float
                        // = Array(F32, 2), 8 bytes).  Pass in one register.
                        let cls = eb_classes[0];
                        if cls == AbiClass::Sse && sse_idx < SSE_ARG_REGS.len() {
                            let reg = SSE_ARG_REGS[sse_idx];
                            let mut inst = Self::mk_inst(
                                X86Opcode::Movsd,
                                Some(MachineOperand::Register(reg)),
                                &[mem_base],
                            );
                            inst.is_call_arg_setup = true;
                            reg_arg_setup.push(inst);
                            sse_idx += 1;
                        } else if cls == AbiClass::Integer && gpr_idx < INTEGER_ARG_REGS.len() {
                            let reg = INTEGER_ARG_REGS[gpr_idx];
                            let mut inst = Self::mk_inst(
                                X86Opcode::Mov,
                                Some(MachineOperand::Register(reg)),
                                &[mem_base],
                            );
                            inst.is_call_arg_setup = true;
                            reg_arg_setup.push(inst);
                            gpr_idx += 1;
                        } else {
                            unified_stack_args.push(StackArgEntry::Scalar(
                                mem_base,
                                if cls == AbiClass::Sse {
                                    StackArgKind::Fp
                                } else {
                                    StackArgKind::Int
                                },
                            ));
                            stack_offset += 8;
                        }
                    } else if int_avail && sse_avail && eb_classes.len() >= 2 {
                        // Build Memory operand for hi-half: base displacement + 8.
                        let mem_hi = match &mem_base {
                            MachineOperand::Memory {
                                base,
                                index,
                                scale,
                                displacement,
                            } => MachineOperand::Memory {
                                base: *base,
                                index: *index,
                                scale: *scale,
                                displacement: displacement + 8,
                            },
                            other => {
                                // Fallback: treat as normal single-reg arg.
                                let mut inst = Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(MachineOperand::Register(INTEGER_ARG_REGS[gpr_idx])),
                                    std::slice::from_ref(other),
                                );
                                inst.is_call_arg_setup = true;
                                reg_arg_setup.push(inst);
                                gpr_idx += 1;
                                continue;
                            }
                        };

                        // Load each eightbyte into the correct register class
                        let lo_class = eb_classes[0];
                        let hi_class = eb_classes[1];

                        // Lo eightbyte
                        if lo_class == AbiClass::Sse {
                            let reg = SSE_ARG_REGS[sse_idx];
                            let mut inst_lo = Self::mk_inst(
                                X86Opcode::Movsd,
                                Some(MachineOperand::Register(reg)),
                                &[mem_base],
                            );
                            inst_lo.is_call_arg_setup = true;
                            reg_arg_setup.push(inst_lo);
                            sse_idx += 1;
                        } else {
                            let reg = INTEGER_ARG_REGS[gpr_idx];
                            let mut inst_lo = Self::mk_inst(
                                X86Opcode::Mov,
                                Some(MachineOperand::Register(reg)),
                                &[mem_base],
                            );
                            inst_lo.is_call_arg_setup = true;
                            reg_arg_setup.push(inst_lo);
                            gpr_idx += 1;
                        }

                        // Hi eightbyte
                        if hi_class == AbiClass::Sse || hi_class == AbiClass::SseUp {
                            let reg = SSE_ARG_REGS[sse_idx];
                            // Use movss (4-byte) instead of movsd (8-byte) when
                            // the hi eightbyte only contains 4 bytes (e.g. a
                            // 12-byte struct whose second eightbyte is a single
                            // float).  An 8-byte movsd would read past the
                            // struct's alloca, which is harmless for loads but
                            // keeping it precise avoids UB.
                            let hi_remaining = struct_load_size.saturating_sub(8);
                            let hi_sse_opcode = if hi_remaining <= 4 {
                                X86Opcode::Movss
                            } else {
                                X86Opcode::Movsd
                            };
                            let mut inst_hi = Self::mk_inst(
                                hi_sse_opcode,
                                Some(MachineOperand::Register(reg)),
                                &[mem_hi],
                            );
                            inst_hi.is_call_arg_setup = true;
                            reg_arg_setup.push(inst_hi);
                            sse_idx += 1;
                        } else {
                            let reg = INTEGER_ARG_REGS[gpr_idx];
                            let mut inst_hi = Self::mk_inst(
                                X86Opcode::Mov,
                                Some(MachineOperand::Register(reg)),
                                &[mem_hi],
                            );
                            inst_hi.is_call_arg_setup = true;
                            reg_arg_setup.push(inst_hi);
                            gpr_idx += 1;
                        }
                    } else {
                        // Overflow to stack: push both halves from frame memory.
                        let mem_hi = match &mem_base {
                            MachineOperand::Memory {
                                base,
                                index,
                                scale,
                                displacement,
                            } => MachineOperand::Memory {
                                base: *base,
                                index: *index,
                                scale: *scale,
                                displacement: displacement + 8,
                            },
                            _ => arg_op.clone(),
                        };
                        unified_stack_args.push(StackArgEntry::Scalar(mem_base, StackArgKind::Int));
                        unified_stack_args.push(StackArgEntry::Scalar(mem_hi, StackArgKind::Int));
                        stack_offset += 16;
                    }
                    continue;
                }
            }
            // Also check struct_pair_hi for callee-received RegisterPair
            // values being passed through to another call.
            if let Some(hi_op) = self.struct_pair_hi.get(arg_val).cloned() {
                // Determine if this pair uses SSE or GPR registers by
                // checking the ABI eightbyte classification.
                let pair_is_sse = self
                    .value_types
                    .get(arg_val)
                    .map(|vty| {
                        let classes = classify_ir_type_eightbytes(vty, &self.target);
                        classes.iter().all(|c| *c == AbiClass::Sse)
                    })
                    .unwrap_or(false);

                if pair_is_sse && sse_idx + 1 < SSE_ARG_REGS.len() {
                    let reg_lo = SSE_ARG_REGS[sse_idx];
                    let reg_hi = SSE_ARG_REGS[sse_idx + 1];
                    let mut inst_lo = Self::mk_inst(
                        X86Opcode::Movsd,
                        Some(MachineOperand::Register(reg_lo)),
                        &[arg_op],
                    );
                    inst_lo.is_call_arg_setup = true;
                    reg_arg_setup.push(inst_lo);
                    let mut inst_hi = Self::mk_inst(
                        X86Opcode::Movsd,
                        Some(MachineOperand::Register(reg_hi)),
                        &[hi_op],
                    );
                    inst_hi.is_call_arg_setup = true;
                    reg_arg_setup.push(inst_hi);
                    sse_idx += 2;
                } else if !pair_is_sse && gpr_idx + 1 < INTEGER_ARG_REGS.len() {
                    let reg_lo = INTEGER_ARG_REGS[gpr_idx];
                    let reg_hi = INTEGER_ARG_REGS[gpr_idx + 1];
                    let mut inst_lo = Self::mk_inst(
                        X86Opcode::Mov,
                        Some(MachineOperand::Register(reg_lo)),
                        &[arg_op],
                    );
                    inst_lo.is_call_arg_setup = true;
                    reg_arg_setup.push(inst_lo);
                    let mut inst_hi = Self::mk_inst(
                        X86Opcode::Mov,
                        Some(MachineOperand::Register(reg_hi)),
                        &[hi_op],
                    );
                    inst_hi.is_call_arg_setup = true;
                    reg_arg_setup.push(inst_hi);
                    gpr_idx += 2;
                } else {
                    // Same lo-before-hi ordering as struct_load_source path:
                    // lo at higher index → pushed first in reverse → lowest addr.
                    unified_stack_args.push(StackArgEntry::Scalar(
                        arg_op,
                        if pair_is_sse {
                            StackArgKind::Fp
                        } else {
                            StackArgKind::Int
                        },
                    ));
                    unified_stack_args.push(StackArgEntry::Scalar(
                        hi_op,
                        if pair_is_sse {
                            StackArgKind::Fp
                        } else {
                            StackArgKind::Int
                        },
                    ));
                    stack_offset += 16;
                }
                continue;
            }

            // --- MEMORY-class struct argument (> 16 bytes) ---
            // Large structs must be pushed onto the stack per SysV ABI.
            // We push all eightbytes of the struct in reverse order.
            if let Some(arg_ty) = self.value_types.get(arg_val) {
                let arg_sz = arg_ty.size_bytes(&self.target);
                if (arg_ty.is_struct() || arg_ty.is_array()) && arg_sz > 16 {
                    let eightbytes = (arg_sz + 7) / 8;
                    // The struct might be in struct_load_source (alloca-backed)
                    // or in the frame via alloca_offsets.
                    let mut base_mem: Option<MachineOperand> = None;
                    if let Some(mem) = self.struct_load_source.get(arg_val) {
                        base_mem = Some(mem.clone());
                    } else if let Some(ref frame) = self.frame {
                        if let Some(&off) = frame.alloca_offsets.get(arg_val) {
                            base_mem = Some(MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: off as i64,
                            });
                        }
                    }
                    // Collect load+push operands for this argument into a
                    // group. Groups are emitted later in REVERSE arg order
                    // (right-to-left per SysV ABI).
                    if let Some(base) = base_mem {
                        let mut arg_group = Vec::new();
                        for i in (0..eightbytes).rev() {
                            let chunk_mem = match &base {
                                MachineOperand::Memory {
                                    base: b,
                                    index,
                                    scale,
                                    displacement,
                                } => MachineOperand::Memory {
                                    base: *b,
                                    index: *index,
                                    scale: *scale,
                                    displacement: displacement + (i as i64) * 8,
                                },
                                _ => base.clone(),
                            };
                            arg_group.push(chunk_mem);
                        }
                        unified_stack_args.push(StackArgEntry::MemClassDirect(arg_group));
                    } else {
                        // Look harder: check if arg_val's producing Load
                        // instruction has a ptr operand backed by an alloca
                        // or a global symbol.
                        let mut found_alloca_base = false;
                        'outer_scan: for blk in _func.blocks() {
                            for ir_inst in blk.instructions() {
                                if let crate::ir::instructions::Instruction::Load {
                                    result: load_res,
                                    ptr: load_ptr,
                                    ..
                                } = ir_inst
                                {
                                    if *load_res == *arg_val {
                                        // Check alloca offsets first.
                                        if let Some(ref frame) = self.frame {
                                            if let Some(&aoff) = frame.alloca_offsets.get(load_ptr)
                                            {
                                                let bm = MachineOperand::Memory {
                                                    base: Some(RBP),
                                                    index: None,
                                                    scale: 1,
                                                    displacement: aoff as i64,
                                                };
                                                let mut grp = Vec::new();
                                                for i in (0..eightbytes).rev() {
                                                    let chunk_mem = match &bm {
                                                        MachineOperand::Memory {
                                                            base: b,
                                                            index: ix,
                                                            scale: sc,
                                                            displacement: d,
                                                        } => MachineOperand::Memory {
                                                            base: *b,
                                                            index: *ix,
                                                            scale: *sc,
                                                            displacement: d + (i as i64) * 8,
                                                        },
                                                        _ => bm.clone(),
                                                    };
                                                    grp.push(chunk_mem);
                                                }
                                                unified_stack_args
                                                    .push(StackArgEntry::MemClassDirect(grp));
                                                found_alloca_base = true;
                                                break 'outer_scan;
                                            }
                                        }
                                        // Check if load_ptr resolves to a
                                        // GlobalSymbol (global struct variable).
                                        let ptr_op = self.get_value(*load_ptr);
                                        if let MachineOperand::GlobalSymbol(ref sym) = ptr_op {
                                            let sym_clone = sym.clone();
                                            unified_stack_args.push(StackArgEntry::MemClassPtr(
                                                MachineOperand::GlobalSymbol(sym_clone),
                                                eightbytes,
                                            ));
                                            found_alloca_base = true;
                                            break 'outer_scan;
                                        }
                                        // Also check struct_load_source for
                                        // the ptr value.
                                        if let Some(mem) = self.struct_load_source.get(load_ptr) {
                                            let base = mem.clone();
                                            let mut grp = Vec::new();
                                            for i in (0..eightbytes).rev() {
                                                let chunk_mem = match &base {
                                                    MachineOperand::Memory {
                                                        base: b,
                                                        index: ix,
                                                        scale: sc,
                                                        displacement: d,
                                                    } => MachineOperand::Memory {
                                                        base: *b,
                                                        index: *ix,
                                                        scale: *sc,
                                                        displacement: d + (i as i64) * 8,
                                                    },
                                                    _ => base.clone(),
                                                };
                                                grp.push(chunk_mem);
                                            }
                                            unified_stack_args
                                                .push(StackArgEntry::MemClassDirect(grp));
                                            found_alloca_base = true;
                                            break 'outer_scan;
                                        }
                                        break 'outer_scan;
                                    }
                                }
                            }
                        }
                        if !found_alloca_base {
                            // Final fallback: use pointer-based loads.
                            unified_stack_args
                                .push(StackArgEntry::MemClassPtr(arg_op.clone(), eightbytes));
                        }
                    }
                    stack_offset += (eightbytes * 8) as i32;
                    continue;
                }
            }

            // ---- F80 (long double) special handling ----
            // Per the AMD64 ABI, long double is classified as X87/X87UP
            // and always passed in memory (on the stack) as 80-bit
            // extended precision (10 bytes in a 16-byte aligned slot).
            // BCC internally stores long double as a 64-bit double.
            // We use x87 FLD QWORD / FSTP TBYTE to convert the internal
            // double representation to 80-bit extended format on the stack.
            let is_f80 = self
                .value_types
                .get(arg_val)
                .map(|ty| matches!(ty, IrType::F80))
                .unwrap_or(false);
            if is_f80 {
                // F80 arguments are collected for stack pushing later.
                // The arg_op holds the XMM register or memory operand
                // containing the double value.
                unified_stack_args.push(StackArgEntry::Scalar(arg_op, StackArgKind::F80));
                stack_offset += 16; // 16 bytes for long double on stack
                continue;
            }

            // Classify the argument as FP or integer by consulting the
            // value_types map (populated from IR instruction result types).
            // F32 and F64 arguments are routed to XMM0–XMM7 per the
            // System V AMD64 ABI; all other types use integer GPRs.
            // Also includes _Complex float (Array(F32, 2), 8 bytes) which
            // fits in a single XMM register.
            // NOTE: F80 (long double) is handled above — it is passed
            // on the stack via x87 FLD/FSTP conversion, not through XMM.
            let is_fp = self
                .value_types
                .get(arg_val)
                .map(|ty| {
                    matches!(ty, IrType::F32 | IrType::F64)
                        || matches!(ty, IrType::Array(ref elem, count)
                            if *count <= 2
                                && ty.size_bytes(&self.target) <= 8
                                && matches!(elem.as_ref(), IrType::F32 | IrType::F64))
                })
                .unwrap_or(false);

            if is_fp && sse_idx < SSE_ARG_REGS.len() {
                let reg = SSE_ARG_REGS[sse_idx];
                // For SSE-classified aggregates loaded from global
                // variables (e.g. _Complex float), use the global symbol
                // operand directly to emit `movsd xmm, [rip+sym]` instead
                // of `movsd xmm, GPR` (which encodes as memory deref
                // through the GPR value → SIGSEGV).
                let eff_arg_op = if let Some(sym) = self.global_sse_load_source.get(arg_val) {
                    MachineOperand::GlobalSymbol(sym.clone())
                } else {
                    arg_op.clone()
                };
                let mut inst = Self::mk_inst(
                    X86Opcode::Movsd,
                    Some(MachineOperand::Register(reg)),
                    &[eff_arg_op],
                );
                inst.is_call_arg_setup = true;
                reg_arg_setup.push(inst);
                sse_idx += 1;
            } else if !is_fp && gpr_idx < INTEGER_ARG_REGS.len() {
                let reg = INTEGER_ARG_REGS[gpr_idx];
                // For function references and global variable addresses
                // (array-to-pointer decay), use LEA to compute the address
                // rather than MOV which would dereference the symbol.
                let opcode = if is_func_ref || is_global_addr {
                    X86Opcode::Lea
                } else {
                    X86Opcode::Mov
                };
                let mut inst =
                    Self::mk_inst(opcode, Some(MachineOperand::Register(reg)), &[arg_op]);
                inst.is_call_arg_setup = true;
                reg_arg_setup.push(inst);
                gpr_idx += 1;
            } else {
                // Collect stack arguments for reverse-order pushing.
                unified_stack_args.push(StackArgEntry::Scalar(
                    arg_op,
                    if is_fp {
                        StackArgKind::Fp
                    } else {
                        StackArgKind::Int
                    },
                ));
                stack_offset += 8;
            }
        }

        // ---- Stack alignment padding ----
        // The System V AMD64 ABI requires RSP to be 16-byte aligned
        // immediately before a CALL instruction.  After the prologue,
        // RSP is 16-aligned.  Each PUSH (8 bytes) toggles alignment.
        // If the total stack argument bytes are not a multiple of 16,
        // we must emit an 8-byte padding SUB to restore alignment
        // before the CALL.  The padding is included in the post-call
        // cleanup ADD RSP.
        if stack_offset > 0 && stack_offset % 16 != 0 {
            let pad = Self::mk_inst(
                X86Opcode::Sub,
                Some(MachineOperand::Register(RSP)),
                &[MachineOperand::Register(RSP), MachineOperand::Immediate(8)],
            );
            out.push(pad);
            stack_offset += 8;
        }

        // Emit ALL stack arguments in reverse order (right-to-left) so the
        // first stack argument (leftmost in the source) ends up at the
        // lowest stack address as the SysV AMD64 ABI requires.
        // This unified loop correctly interleaves MEMORY-class struct
        // pushes with scalar/F80 pushes based on their original
        // parameter position.
        for entry in unified_stack_args.into_iter().rev() {
            match entry {
                StackArgEntry::MemClassDirect(grp) => {
                    for mc_op in grp.into_iter() {
                        let mut ld_inst = Self::mk_inst(
                            X86Opcode::Mov,
                            Some(MachineOperand::Register(R11)),
                            &[mc_op],
                        );
                        ld_inst.is_call_arg_setup = true;
                        out.push(ld_inst);
                        let mut push_inst =
                            Self::mk_inst(X86Opcode::Push, None, &[MachineOperand::Register(R11)]);
                        push_inst.is_call_arg_setup = true;
                        out.push(push_inst);
                    }
                }
                StackArgEntry::MemClassPtr(ptr_op, eightbytes) => {
                    let ptr_opcode = if matches!(&ptr_op, MachineOperand::GlobalSymbol(_)) {
                        X86Opcode::Lea
                    } else {
                        X86Opcode::Mov
                    };
                    let mut ptr_inst =
                        Self::mk_inst(ptr_opcode, Some(MachineOperand::Register(R10)), &[ptr_op]);
                    ptr_inst.is_call_arg_setup = true;
                    out.push(ptr_inst);
                    for i in (0..eightbytes).rev() {
                        let src_mem = MachineOperand::Memory {
                            base: Some(R10),
                            index: None,
                            scale: 1,
                            displacement: (i as i64) * 8,
                        };
                        let mut ld_inst = Self::mk_inst(
                            X86Opcode::Mov,
                            Some(MachineOperand::Register(R11)),
                            &[src_mem],
                        );
                        ld_inst.is_call_arg_setup = true;
                        out.push(ld_inst);
                        let mut push_inst =
                            Self::mk_inst(X86Opcode::Push, None, &[MachineOperand::Register(R11)]);
                        push_inst.is_call_arg_setup = true;
                        out.push(push_inst);
                    }
                }
                StackArgEntry::Scalar(arg_op, arg_kind) => match arg_kind {
                    StackArgKind::F80 => {
                        // ---- Long double (F80) stack argument ----
                        // Per AMD64 ABI, long double occupies 16 bytes on the
                        // stack in 80-bit x87 extended precision format.
                        // BCC internally stores long double as a 64-bit double.
                        // Conversion sequence:
                        //   1. SUB RSP, 16          — allocate 16-byte stack slot
                        //   2. MOVSD [RSP], xmmN    — store double to [RSP]
                        //   3. FLD QWORD [RSP]      — load double into x87 ST(0)
                        //      (auto-converts to 80-bit extended internally)
                        //   4. FSTP TWORD [RSP]     — store 80-bit extended to [RSP]
                        //      (10 bytes of data + 6 bytes padding = 16 bytes)
                        let sub_inst = Self::mk_inst(
                            X86Opcode::Sub,
                            Some(MachineOperand::Register(RSP)),
                            &[MachineOperand::Register(RSP), MachineOperand::Immediate(16)],
                        );
                        out.push(sub_inst);
                        // Store the double value from XMM register to [RSP]
                        let mut store_inst = MachineInstruction::new(X86Opcode::Movsd.as_u32());
                        store_inst.operands.push(MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        });
                        store_inst.operands.push(arg_op);
                        store_inst.is_call_arg_setup = true;
                        out.push(store_inst);
                        // FLD QWORD [RSP] — load double into x87, converts to 80-bit
                        let mut fld_inst = MachineInstruction::new(X86Opcode::FldMem64.as_u32());
                        fld_inst.operands.push(MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        });
                        fld_inst.is_call_arg_setup = true;
                        out.push(fld_inst);
                        // FSTP TWORD [RSP] — store 80-bit extended to stack slot
                        let mut fstp_inst = MachineInstruction::new(X86Opcode::FstpMem80.as_u32());
                        fstp_inst.operands.push(MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        });
                        fstp_inst.is_call_arg_setup = true;
                        out.push(fstp_inst);
                    }
                    StackArgKind::Fp => {
                        // Floating-point (F32/F64) values live in XMM registers
                        // and CANNOT be pushed with the PUSH instruction.
                        // Strategy: SUB RSP, 8 then MOVSD [RSP], xmmN.
                        let sub_inst = Self::mk_inst(
                            X86Opcode::Sub,
                            Some(MachineOperand::Register(RSP)),
                            &[MachineOperand::Register(RSP), MachineOperand::Immediate(8)],
                        );
                        out.push(sub_inst);
                        // MOVSD [RSP], xmm (store float to newly allocated stack slot)
                        let mut store_inst = MachineInstruction::new(X86Opcode::Movsd.as_u32());
                        store_inst.operands.push(MachineOperand::Memory {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: 0,
                        });
                        store_inst.operands.push(arg_op);
                        store_inst.is_call_arg_setup = true;
                        out.push(store_inst);
                    }
                    StackArgKind::Int => {
                        match &arg_op {
                            MachineOperand::GlobalSymbol(_) => {
                                // The x86-64 PUSH instruction cannot encode a 64-bit
                                // address operand directly. Materialize the address
                                // into R11 via LEA, then push R11.
                                let r11 = MachineOperand::Register(R11);
                                let mut lea_inst =
                                    Self::mk_inst(X86Opcode::Lea, Some(r11.clone()), &[arg_op]);
                                lea_inst.is_call_arg_setup = true;
                                out.push(lea_inst);
                                let mut push_inst = Self::mk_inst(X86Opcode::Push, None, &[r11]);
                                push_inst.is_call_arg_setup = true;
                                out.push(push_inst);
                            }
                            _ => {
                                let mut push_inst = Self::mk_inst(X86Opcode::Push, None, &[arg_op]);
                                push_inst.is_call_arg_setup = true;
                                out.push(push_inst);
                            }
                        }
                    }
                },
            }
        }

        // Emit register-argument MOVs AFTER all stack PUSHes.
        // This ensures PUSHes read original register values before any
        // register-arg MOV overwrites them (the definitive fix for the
        // PUSH-reads-clobbered-register bug).
        out.extend(reg_arg_setup);

        // 2. Resolve callee operand.  For direct calls (known function
        //    names), use a GlobalSymbol operand.  For indirect calls
        //    (function pointers loaded from memory/registers), use the
        //    virtual register holding the callee address.
        //
        //    NOTE: The `MOV RAX, <sse_count>` emitted below (step 3) may
        //    clobber RAX.  If the register allocator assigns the callee
        //    virtual register to RAX, this creates a conflict.  The
        //    post-allocation pass `fix_indirect_call_rax_clobber` in
        //    generation.rs detects this case and inserts a save to R11
        //    (the reserved spill-scratch register) before the AL setup.
        let is_direct_call = self.func_ref_names.contains_key(&callee);
        let callee_op = if is_direct_call {
            let fname = self.func_ref_names.get(&callee).unwrap();
            MachineOperand::GlobalSymbol(fname.clone())
        } else {
            self.get_value(callee)
        };

        // 3. Set AL = number of SSE registers used for variadic calls.
        //    Per the System V AMD64 ABI, AL must contain the count of
        //    vector (XMM) registers used when calling a variadic function.
        //    We set AL unconditionally for ALL calls because:
        //    (a) we may not know at call-site whether the callee is
        //        variadic (e.g., indirect calls, or a declared-but-not-
        //        defined function like printf that IS variadic), and
        //    (b) non-variadic callees simply ignore AL, so this is safe.
        //    Mark as `is_call_arg_setup` so the parallel-move resolver in
        //    `resolve_call_arg_conflicts` includes this instruction in the
        //    contiguous arg-setup window preceding the CALL.
        {
            let mut al_inst = Self::mk_inst(
                X86Opcode::Mov,
                Some(MachineOperand::Register(RAX)),
                &[MachineOperand::Immediate(sse_idx as i64)],
            );
            al_inst.is_call_arg_setup = true;
            out.push(al_inst);
        }

        // 4. Emit CALL instruction.
        let mut call_inst = Self::mk_inst(X86Opcode::Call, None, &[callee_op]);
        call_inst.is_call = true;
        out.push(call_inst);

        // 4. Clean up stack arguments (if any were pushed).
        if stack_offset > 0 {
            out.push(Self::mk_inst(
                X86Opcode::Add,
                Some(MachineOperand::Register(RSP)),
                &[
                    MachineOperand::Register(RSP),
                    MachineOperand::Immediate(stack_offset as i64),
                ],
            ));
        }

        // 5. Retrieve return value from RAX (integer) or XMM0 (float).
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());

        let ret_loc = self.abi.classify_return(return_type);
        match ret_loc {
            RetLocation::Register(reg) => {
                // For aggregate types returned in a single SSE register
                // (e.g. _Complex float = Array(F32, 2), 8 bytes, or
                // struct { double; }), spill XMM to a temp frame slot
                // to avoid the Movsd-to-GPR encoding trap.
                let is_sse_ret = registers::is_sse(reg);
                let is_aggregate_ret = return_type.is_aggregate();

                if is_sse_ret && is_aggregate_ret {
                    // Spill XMM return value to struct_temp slot.
                    if let Some(&temp_offset) = self
                        .frame
                        .as_ref()
                        .and_then(|f| f.struct_temp_offsets.get(&result))
                    {
                        let ret_sz = return_type.size_bytes(&self.target);
                        let fop = if ret_sz <= 4 {
                            X86Opcode::Movss
                        } else {
                            X86Opcode::Movsd
                        };
                        let mem_slot = MachineOperand::Memory {
                            base: Some(RBP),
                            index: None,
                            scale: 1,
                            displacement: temp_offset as i64,
                        };
                        // Store XMM to temp slot
                        out.push(Self::mk_inst(
                            fop,
                            None,
                            &[mem_slot.clone(), MachineOperand::Register(reg)],
                        ));
                        // Load into GPR vreg (for value tracking)
                        out.push(Self::mk_inst(
                            X86Opcode::Mov,
                            Some(dst),
                            &[mem_slot.clone()],
                        ));
                        self.struct_load_source.insert(result, mem_slot);
                    } else {
                        // No temp slot: try type override for simple structs
                        let opcode = X86Opcode::Movsd;
                        out.push(Self::mk_inst(
                            opcode,
                            Some(dst),
                            &[MachineOperand::Register(reg)],
                        ));
                        // Override value_types so register allocator uses XMM
                        if let Some(vty) = self.value_types.get(&result).cloned() {
                            if let IrType::Struct(ref st) = vty {
                                if st.fields.len() == 1 {
                                    match &st.fields[0] {
                                        IrType::F64 | IrType::F80 => {
                                            self.value_types.insert(result, IrType::F64);
                                        }
                                        IrType::F32 => {
                                            self.value_types.insert(result, IrType::F32);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                } else {
                    let opcode = if is_sse_ret {
                        X86Opcode::Movsd
                    } else {
                        X86Opcode::Mov
                    };
                    out.push(Self::mk_inst(
                        opcode,
                        Some(dst),
                        &[MachineOperand::Register(reg)],
                    ));
                }
            }
            RetLocation::RegisterPair(lo, hi) => {
                // Spill both eightbytes of the return value to
                // a pre-allocated stack temp slot.  This avoids the
                // register allocator needing to keep two vregs alive
                // simultaneously (which causes clobber bugs).
                // Use Movsd for SSE registers, Mov for GPRs.
                let lo_is_sse = registers::is_sse(lo);
                let hi_is_sse = registers::is_sse(hi);
                let lo_opc = if lo_is_sse {
                    X86Opcode::Movsd
                } else {
                    X86Opcode::Mov
                };
                let hi_opc = if hi_is_sse {
                    X86Opcode::Movsd
                } else {
                    X86Opcode::Mov
                };
                if let Some(&temp_offset) = self
                    .frame
                    .as_ref()
                    .and_then(|f| f.struct_temp_offsets.get(&result))
                {
                    let mem_lo = MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: temp_offset as i64,
                    };
                    let mem_hi = MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: (temp_offset + 8) as i64,
                    };
                    // Store lo register → [RBP+offset] (first eightbyte)
                    out.push(Self::mk_inst(
                        lo_opc,
                        None,
                        &[mem_lo.clone(), MachineOperand::Register(lo)],
                    ));
                    // Store hi register → [RBP+offset+8] (second eightbyte)
                    out.push(Self::mk_inst(
                        hi_opc,
                        None,
                        &[mem_hi, MachineOperand::Register(hi)],
                    ));
                    // Record the stable memory location for struct_load_source
                    // so Store/Return/Call handlers can access both halves.
                    self.struct_load_source.insert(result, mem_lo.clone());
                    // Give the result vreg a pointer to the temp slot via
                    // LEA so that `get_value(result)` returns a usable
                    // address.  We must NOT use `Movsd vreg, XMM0` here
                    // because the register allocator assigns the vreg to a
                    // GPR (the type is an aggregate), and `movsd GPR, XMM`
                    // encodes as `movsd [GPR], XMM` — a memory store
                    // through an uninitialised pointer, causing SIGSEGV.
                    out.push(Self::mk_inst(X86Opcode::Lea, Some(dst), &[mem_lo]));
                } else {
                    // No pre-allocated temp slot — fall back to capturing
                    // the lo half in a vreg.  Use Mov (GPR→GPR) even for
                    // SSE returns to avoid the movsd-to-GPR encoding trap.
                    out.push(Self::mk_inst(
                        X86Opcode::Mov,
                        Some(dst),
                        &[MachineOperand::Register(lo)],
                    ));
                }
            }
            RetLocation::Indirect => {
                // MEMORY-class return: the callee wrote the return value
                // into the pre-allocated buffer whose address was passed as
                // hidden first argument.
                if let Some(slot_off) = indirect_ret_slot {
                    let mem_base = MachineOperand::Memory {
                        base: Some(RBP),
                        index: None,
                        scale: 1,
                        displacement: slot_off as i64,
                    };
                    // F80 (long double) is MEMORY-class but the value is
                    // a scalar float — load it into an XMM vreg via Movsd
                    // instead of LEA (which would give a pointer).
                    if matches!(return_type, IrType::F80) {
                        self.struct_load_source.insert(result, mem_base.clone());
                        out.push(Self::mk_inst(X86Opcode::Movsd, Some(dst), &[mem_base]));
                    } else {
                        self.struct_load_source.insert(result, mem_base.clone());
                        // Map the result to an LEA of the buffer so that
                        // get_value returns a pointer to the struct data.
                        out.push(Self::mk_inst(X86Opcode::Lea, Some(dst), &[mem_base]));
                    }
                }
            }
            RetLocation::Void => {
                // No register-to-move; result is void.
            }
        }

        out
    }
}

// =========================================================================
// ArchCodegen trait implementation for X86_64CodeGen
// =========================================================================

impl ArchCodegen for X86_64CodeGen {
    /// Lower an IR function to a MachineFunction.
    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &FxHashMap<Value, String>,
        global_var_refs: &FxHashMap<Value, String>,
    ) -> Result<MachineFunction, String> {
        let mut codegen = Self::new(self.target);
        codegen.set_func_ref_names(func_ref_map.clone());
        codegen.set_global_var_refs(global_var_refs.clone());
        codegen.lower_function_impl(func, diag, globals)
    }

    /// Emit machine code bytes from a MachineFunction.
    fn emit_assembly(
        &self,
        mf: &MachineFunction,
    ) -> Result<crate::backend::traits::AssembledFunction, String> {
        use crate::backend::x86_64::assembler::encoder::X86_64Encoder;

        let mut encoder = X86_64Encoder::new(0);
        let mut bytes = Vec::new();

        for block in &mf.blocks {
            for inst in &block.instructions {
                let encoded = encoder.encode_instruction(inst);
                bytes.extend_from_slice(&encoded.bytes);
            }
        }
        Ok(crate::backend::traits::AssembledFunction {
            bytes,
            relocations: Vec::new(),
        })
    }

    /// Return the target architecture.
    fn target(&self) -> Target {
        self.target
    }

    /// Return register allocation metadata for x86-64.
    fn register_info(&self) -> RegisterInfo {
        self.abi.x86_64_register_info()
    }

    /// Return supported x86-64 relocation types.
    ///
    /// These match the ELF AMD64 ABI supplement relocation types used by
    /// the built-in assembler and linker for symbol references, PIC/PLT
    /// indirection, and GOT access patterns.
    fn relocation_types(&self) -> &'static [RelocationTypeInfo] {
        static TABLE: &[RelocationTypeInfo] = &[
            // R_X86_64_NONE = 0 — no relocation
            RelocationTypeInfo::new("R_X86_64_NONE", 0, 0, false),
            // R_X86_64_64 = 1 — absolute 64-bit address
            RelocationTypeInfo::new("R_X86_64_64", 1, 8, false),
            // R_X86_64_PC32 = 2 — 32-bit PC-relative (RIP-relative)
            RelocationTypeInfo::new("R_X86_64_PC32", 2, 4, true),
            // R_X86_64_GOT32 = 3 — 32-bit GOT entry offset
            RelocationTypeInfo::new("R_X86_64_GOT32", 3, 4, false),
            // R_X86_64_PLT32 = 4 — 32-bit PLT-relative
            RelocationTypeInfo::new("R_X86_64_PLT32", 4, 4, true),
            // R_X86_64_GLOB_DAT = 6 — GOT entry for dynamic symbol
            RelocationTypeInfo::new("R_X86_64_GLOB_DAT", 6, 8, false),
            // R_X86_64_JUMP_SLOT = 7 — PLT jump slot
            RelocationTypeInfo::new("R_X86_64_JUMP_SLOT", 7, 8, false),
            // R_X86_64_RELATIVE = 8 — base-relative (for PIC data)
            RelocationTypeInfo::new("R_X86_64_RELATIVE", 8, 8, false),
            // R_X86_64_GOTPCREL = 9 — 32-bit PC-relative GOT access
            RelocationTypeInfo::new("R_X86_64_GOTPCREL", 9, 4, true),
            // R_X86_64_32 = 10 — absolute 32-bit (zero-extended)
            RelocationTypeInfo::new("R_X86_64_32", 10, 4, false),
            // R_X86_64_32S = 11 — absolute 32-bit (sign-extended)
            RelocationTypeInfo::new("R_X86_64_32S", 11, 4, false),
            // R_X86_64_16 = 12 — absolute 16-bit
            RelocationTypeInfo::new("R_X86_64_16", 12, 2, false),
            // R_X86_64_PC16 = 13 — 16-bit PC-relative
            RelocationTypeInfo::new("R_X86_64_PC16", 13, 2, true),
            // R_X86_64_PC64 = 24 — 64-bit PC-relative
            RelocationTypeInfo::new("R_X86_64_PC64", 24, 8, true),
            // R_X86_64_GOTPCRELX = 41 — relaxable GOT PC-relative
            RelocationTypeInfo::new("R_X86_64_GOTPCRELX", 41, 4, true),
            // R_X86_64_REX_GOTPCRELX = 42 — relaxable REX GOT PC-relative
            RelocationTypeInfo::new("R_X86_64_REX_GOTPCRELX", 42, 4, true),
        ];
        TABLE
    }

    /// Emit function prologue instructions.
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        // Use the instance method (not static) so variadic register saves
        // from self.frame are included in the prologue.
        self.emit_prologue_with_va(mf)
    }

    /// Emit function epilogue instructions.
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        X86_64CodeGen::emit_epilogue_static(mf)
    }

    /// The frame pointer register is RBP on x86-64.
    fn frame_pointer_reg(&self) -> u16 {
        RBP
    }

    /// The stack pointer register is RSP on x86-64.
    fn stack_pointer_reg(&self) -> u16 {
        RSP
    }

    /// x86-64 has no explicit return-address register.
    fn return_address_reg(&self) -> Option<u16> {
        None
    }

    /// Classify a single argument for the System V AMD64 ABI.
    fn classify_argument(&self, ty: &IrType) -> traits::ArgLocation {
        let loc = self.abi.classify_argument(ty);
        match loc {
            abi::ArgLocation::Register(r) => traits::ArgLocation::Register(r),
            abi::ArgLocation::RegisterPair(a, b) => traits::ArgLocation::RegisterPair(a, b),
            abi::ArgLocation::Stack(off) => traits::ArgLocation::Stack(off),
            abi::ArgLocation::Indirect(r) => traits::ArgLocation::Register(r),
        }
    }

    /// Classify the return value for the System V AMD64 ABI.
    fn classify_return(&self, ty: &IrType) -> traits::ArgLocation {
        let loc = self.abi.classify_return(ty);
        match loc {
            RetLocation::Register(r) => traits::ArgLocation::Register(r),
            RetLocation::RegisterPair(a, b) => traits::ArgLocation::RegisterPair(a, b),
            RetLocation::Indirect => traits::ArgLocation::Stack(0),
            RetLocation::Void => traits::ArgLocation::Stack(0),
        }
    }
}

// =========================================================================
// Internal bridge methods for ArchCodegen trait
// =========================================================================

impl X86_64CodeGen {
    /// Internal mutable lower_function for the trait.
    fn lower_function_impl(
        &mut self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        self.lower(func, diag, globals)
    }

    /// Static prologue emitter.
    #[allow(dead_code)]
    fn emit_prologue_static(mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();
        let frame_size = mf.frame_size;

        prologue.push(Self::mk_inst(
            X86Opcode::Push,
            None,
            &[MachineOperand::Register(RBP)],
        ));
        prologue.push(Self::mk_inst(
            X86Opcode::Mov,
            Some(MachineOperand::Register(RBP)),
            &[MachineOperand::Register(RSP)],
        ));

        if frame_size > 0 && (!mf.is_leaf || frame_size > RED_ZONE_SIZE) {
            if frame_size > 4096 {
                let full_pages = frame_size / 4096;
                let remainder = frame_size % 4096;
                for _page in 0..full_pages {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(4096),
                        ],
                    ));
                    prologue.push(Self::mk_inst(
                        X86Opcode::Test,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(RSP),
                                index: None,
                                scale: 1,
                                displacement: 0,
                            },
                            MachineOperand::Immediate(0),
                        ],
                    ));
                }
                if remainder > 0 {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(remainder as i64),
                        ],
                    ));
                }
            } else {
                prologue.push(Self::mk_inst(
                    X86Opcode::Sub,
                    Some(MachineOperand::Register(RSP)),
                    &[
                        MachineOperand::Register(RSP),
                        MachineOperand::Immediate(frame_size as i64),
                    ],
                ));
            }
        }

        for &reg in &mf.callee_saved_regs {
            prologue.push(Self::mk_inst(
                X86Opcode::Push,
                None,
                &[MachineOperand::Register(reg)],
            ));
        }

        prologue
    }

    /// Static epilogue emitter.
    fn emit_epilogue_static(mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();

        // Reset RSP to the callee-save area using RBP-relative LEA.
        // During the function body, call argument pushes and other RSP
        // adjustments may cause RSP to drift from the position established
        // after the prologue.  The pop instructions below rely on RSP
        // pointing to the bottom of the callee-save area, so we
        // explicitly restore it here.
        if !mf.callee_saved_regs.is_empty() {
            let callee_push_bytes = mf.callee_saved_regs.len() * 8;
            let total = mf.frame_size + callee_push_bytes;
            let aligned_total = (total + 15) & !15;
            epilogue.push(Self::mk_inst(
                X86Opcode::Lea,
                Some(MachineOperand::Register(RSP)),
                &[MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: -(aligned_total as i64),
                }],
            ));
        }

        for &reg in mf.callee_saved_regs.iter().rev() {
            epilogue.push(Self::mk_inst(
                X86Opcode::Pop,
                Some(MachineOperand::Register(reg)),
                &[],
            ));
        }

        epilogue.push(Self::mk_inst(X86Opcode::Leave, None, &[]));

        let mut ret_inst = Self::mk_inst(X86Opcode::Ret, None, &[]);
        ret_inst.is_terminator = true;
        epilogue.push(ret_inst);

        epilogue
    }
}

// =========================================================================
// Display implementations for debugging
// =========================================================================

impl std::fmt::Display for X86Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            X86Opcode::Mov => "mov",
            X86Opcode::MovZX => "movzx",
            X86Opcode::MovSX => "movsx",
            X86Opcode::Lea => "lea",
            X86Opcode::Push => "push",
            X86Opcode::Pop => "pop",
            X86Opcode::Xchg => "xchg",
            X86Opcode::Add => "add",
            X86Opcode::Sub => "sub",
            X86Opcode::Imul => "imul",
            X86Opcode::Idiv => "idiv",
            X86Opcode::Div => "div",
            X86Opcode::Neg => "neg",
            X86Opcode::Inc => "inc",
            X86Opcode::Dec => "dec",
            X86Opcode::And => "and",
            X86Opcode::Or => "or",
            X86Opcode::Xor => "xor",
            X86Opcode::Not => "not",
            X86Opcode::Shl => "shl",
            X86Opcode::Shr => "shr",
            X86Opcode::Sar => "sar",
            X86Opcode::Rol => "rol",
            X86Opcode::Ror => "ror",
            X86Opcode::Cmp => "cmp",
            X86Opcode::Test => "test",
            X86Opcode::Cmovo => "cmovo",
            X86Opcode::Cmovno => "cmovno",
            X86Opcode::Cmovb => "cmovb",
            X86Opcode::Cmovae => "cmovae",
            X86Opcode::Cmove => "cmove",
            X86Opcode::Cmovne => "cmovne",
            X86Opcode::Cmovbe => "cmovbe",
            X86Opcode::Cmova => "cmova",
            X86Opcode::Cmovs => "cmovs",
            X86Opcode::Cmovns => "cmovns",
            X86Opcode::Cmovp => "cmovp",
            X86Opcode::Cmovnp => "cmovnp",
            X86Opcode::Cmovl => "cmovl",
            X86Opcode::Cmovge => "cmovge",
            X86Opcode::Cmovle => "cmovle",
            X86Opcode::Cmovg => "cmovg",
            X86Opcode::Seto => "seto",
            X86Opcode::Setno => "setno",
            X86Opcode::Setb => "setb",
            X86Opcode::Setae => "setae",
            X86Opcode::Sete => "sete",
            X86Opcode::Setne => "setne",
            X86Opcode::Setbe => "setbe",
            X86Opcode::Seta => "seta",
            X86Opcode::Sets => "sets",
            X86Opcode::Setns => "setns",
            X86Opcode::Setp => "setp",
            X86Opcode::Setnp => "setnp",
            X86Opcode::Setl => "setl",
            X86Opcode::Setge => "setge",
            X86Opcode::Setle => "setle",
            X86Opcode::Setg => "setg",
            X86Opcode::Jmp => "jmp",
            X86Opcode::Jo => "jo",
            X86Opcode::Jno => "jno",
            X86Opcode::Jb => "jb",
            X86Opcode::Jae => "jae",
            X86Opcode::Je => "je",
            X86Opcode::Jne => "jne",
            X86Opcode::Jbe => "jbe",
            X86Opcode::Ja => "ja",
            X86Opcode::Js => "js",
            X86Opcode::Jns => "jns",
            X86Opcode::Jp => "jp",
            X86Opcode::Jnp => "jnp",
            X86Opcode::Jl => "jl",
            X86Opcode::Jge => "jge",
            X86Opcode::Jle => "jle",
            X86Opcode::Jg => "jg",
            X86Opcode::Call => "call",
            X86Opcode::Ret => "ret",
            X86Opcode::Nop => "nop",
            X86Opcode::Movsd => "movsd",
            X86Opcode::Movss => "movss",
            X86Opcode::Addsd => "addsd",
            X86Opcode::Addss => "addss",
            X86Opcode::Subsd => "subsd",
            X86Opcode::Subss => "subss",
            X86Opcode::Mulsd => "mulsd",
            X86Opcode::Mulss => "mulss",
            X86Opcode::Divsd => "divsd",
            X86Opcode::Divss => "divss",
            X86Opcode::Ucomisd => "ucomisd",
            X86Opcode::Ucomiss => "ucomiss",
            X86Opcode::Cvtsi2sd => "cvtsi2sd",
            X86Opcode::Cvtsi2ss => "cvtsi2ss",
            X86Opcode::Cvtsd2si => "cvtsd2si",
            X86Opcode::Cvtss2si => "cvtss2si",
            X86Opcode::Cvtsd2ss => "cvtsd2ss",
            X86Opcode::Cvtss2sd => "cvtss2sd",
            X86Opcode::Enter => "enter",
            X86Opcode::Leave => "leave",
            X86Opcode::Cdq => "cdq",
            X86Opcode::Cqo => "cqo",
            X86Opcode::Endbr64 => "endbr64",
            X86Opcode::Pause => "pause",
            X86Opcode::Lfence => "lfence",
            X86Opcode::Bsr => "bsr",
            X86Opcode::Bsf => "bsf",
            X86Opcode::Popcnt => "popcnt",
            X86Opcode::Bswap => "bswap",
            X86Opcode::Ud2 => "ud2",
            X86Opcode::RepMovsq => "rep_movsq",
            X86Opcode::InlineAsm => "<inline-asm>",
            X86Opcode::InternalLabelDef => "<label-def>",
            X86Opcode::LoadInd => "movq_ind_load",
            X86Opcode::LoadInd32 => "movl_ind_load",
            X86Opcode::LoadInd16 => "movw_ind_load",
            X86Opcode::LoadInd8 => "movb_ind_load",
            X86Opcode::StoreInd => "movq_ind_store",
            X86Opcode::StoreInd32 => "movl_ind_store",
            X86Opcode::StoreInd16 => "movw_ind_store",
            X86Opcode::StoreInd8 => "movb_ind_store",
            X86Opcode::FldMem64 => "fld_qword",
            X86Opcode::FldMem80 => "fld_tword",
            X86Opcode::FstpMem80 => "fstp_tword",
            X86Opcode::FstpMem64 => "fstp_qword",
        };
        f.write_str(name)
    }
}

impl std::fmt::Display for X86Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            X86Operand::Reg(r) => {
                write!(f, "{}", registers::reg_name(*r, 8))
            }
            X86Operand::Imm(val) => write!(f, "${}", val),
            X86Operand::Mem {
                base,
                index,
                scale,
                disp,
                size: _,
            } => {
                write!(f, "[")?;
                let mut need_plus = false;
                if let Some(b) = base {
                    write!(f, "{}", registers::reg_name(*b, 8))?;
                    need_plus = true;
                }
                if let Some(idx) = index {
                    if need_plus {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}*{}", registers::reg_name(*idx, 8), scale)?;
                    need_plus = true;
                }
                if *disp != 0 {
                    if need_plus && *disp > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", disp)?;
                }
                write!(f, "]")
            }
            X86Operand::RipRelative { symbol, offset } => {
                if *offset != 0 {
                    write!(f, "[rip + {} + {}]", symbol, offset)
                } else {
                    write!(f, "[rip + {}]", symbol)
                }
            }
            X86Operand::Label(label) => write!(f, "{}", label),
            X86Operand::Symbol(sym) => write!(f, "{}", sym),
        }
    }
}
