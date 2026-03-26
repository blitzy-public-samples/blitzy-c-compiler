//! # ArchCodegen Trait — Architecture Abstraction Layer for BCC
//!
//! This module defines the **core abstraction** that enables multi-architecture
//! support in BCC's backend.  The [`ArchCodegen`] trait is the single dispatch
//! point through which the architecture-agnostic code generation driver
//! ([`crate::backend::generation`]) interacts with architecture-specific
//! backends (x86-64, i686, AArch64, RISC-V 64).
//!
//! ## Exports
//!
//! | Type                  | Kind   | Purpose                                               |
//! |-----------------------|--------|-------------------------------------------------------|
//! | [`ArchCodegen`]       | trait  | Architecture abstraction — instruction selection, ABI  |
//! | [`MachineFunction`]   | struct | Post-selection, pre-regalloc function representation   |
//! | [`MachineBasicBlock`] | struct | Ordered instruction sequence with CFG edges            |
//! | [`MachineInstruction`]| struct | Architecture-agnostic machine instruction container    |
//! | [`MachineOperand`]    | enum   | Operand kinds: registers, immediates, memory, symbols  |
//! | [`RegisterInfo`]      | struct | Architecture register sets for register allocation     |
//! | [`SpillSlot`]         | struct | Stack frame spill slot descriptor                      |
//! | [`RelocationTypeInfo`]| struct | Architecture-specific relocation metadata              |
//! | [`ArgLocation`]       | enum   | ABI argument/return value placement descriptor         |
//!
//! ## Standalone Backend Mode
//!
//! Every architecture backend implements [`ArchCodegen`] to provide:
//! - **Instruction selection** via [`lower_function`](ArchCodegen::lower_function)
//! - **Assembly encoding** via [`emit_assembly`](ArchCodegen::emit_assembly)
//! - **Register information** via [`register_info`](ArchCodegen::register_info)
//! - **ABI classification** via [`classify_argument`](ArchCodegen::classify_argument)
//!   and [`classify_return`](ArchCodegen::classify_return)
//! - **Prologue/epilogue emission** via [`emit_prologue`](ArchCodegen::emit_prologue)
//!   and [`emit_epilogue`](ArchCodegen::emit_epilogue)
//!
//! No external toolchain invocation occurs — BCC includes its own assembler
//! and linker for all four target architectures.
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::ir::function`, `crate::ir::types`,
//! `crate::common::target`, `crate::common::diagnostics`, and the Rust
//! standard library.  No external crates are used.

use std::fmt;

use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::instructions::Value;
use crate::ir::types::IrType;

// ===========================================================================
// MachineOperand — operand kinds for machine instructions
// ===========================================================================

/// An operand of a machine-level instruction.
///
/// `MachineOperand` is deliberately architecture-agnostic — it carries enough
/// information for register allocation, instruction encoding, and relocation
/// emission without embedding architecture-specific constants.
///
/// # Variants
///
/// | Variant          | Usage                                          |
/// |------------------|------------------------------------------------|
/// | `Register`       | Physical register (post-allocation)             |
/// | `VirtualRegister`| Virtual register (pre-allocation SSA value)     |
/// | `Immediate`      | Constant integer operand                        |
/// | `Memory`         | Memory addressing mode (base+index*scale+disp) |
/// | `FrameSlot`      | Stack frame slot reference (resolved later)     |
/// | `GlobalSymbol`   | Reference to a global/external symbol           |
/// | `BlockLabel`     | Target basic block (for branches)               |
#[derive(Debug, Clone, PartialEq)]
pub enum MachineOperand {
    /// A physical (hardware) register identified by its architecture-specific
    /// index.  Register numbering is defined by each architecture's
    /// `registers.rs` module.
    ///
    /// After register allocation, all `VirtualRegister` operands are replaced
    /// with `Register` operands.
    Register(u16),

    /// A virtual register representing an SSA value before register allocation.
    ///
    /// Virtual registers are assigned during instruction selection and mapped
    /// to physical registers (or spill slots) by the register allocator.
    VirtualRegister(u32),

    /// An immediate (constant) integer value embedded in the instruction.
    ///
    /// The encoding width is determined by the instruction format — the
    /// assembler truncates or sign-extends as required.
    Immediate(i64),

    /// A memory operand using the classic base+index×scale+displacement
    /// addressing mode.
    ///
    /// # Fields
    ///
    /// - `base`:  Optional base register (e.g., RBP for stack access).
    /// - `index`: Optional index register (for array-style addressing).
    /// - `scale`: Multiplier for the index (1, 2, 4, or 8 on x86).
    /// - `displacement`: Signed byte offset added to the computed address.
    ///
    /// Not all architectures support all combinations — AArch64 and RISC-V
    /// have more limited addressing modes.  The instruction selector is
    /// responsible for producing only valid combinations.
    Memory {
        /// Base register (e.g., frame pointer, stack pointer, or GPR).
        base: Option<u16>,
        /// Index register for scaled addressing.
        index: Option<u16>,
        /// Scale factor applied to the index register.
        scale: u8,
        /// Signed displacement / offset in bytes.
        displacement: i64,
    },

    /// A reference to a stack frame spill slot, identified by its byte
    /// offset from the frame pointer.
    ///
    /// Frame slots are resolved to concrete `Memory` operands during
    /// prologue/epilogue emission once the frame layout is finalized.
    FrameSlot(i32),

    /// A reference to a named global symbol (function or variable).
    ///
    /// The symbol name is used during relocation processing to bind the
    /// instruction to the correct address at link time.
    GlobalSymbol(String),

    /// A basic block label used as a branch target.
    ///
    /// The `u32` value is the block index within the containing
    /// [`MachineFunction`]'s block list.
    BlockLabel(u32),
}

impl fmt::Display for MachineOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MachineOperand::Register(r) => write!(f, "r{}", r),
            MachineOperand::VirtualRegister(v) => write!(f, "v{}", v),
            MachineOperand::Immediate(imm) => write!(f, "#{}", imm),
            MachineOperand::Memory {
                base,
                index,
                scale,
                displacement,
            } => {
                write!(f, "[")?;
                let mut need_plus = false;
                if let Some(b) = base {
                    write!(f, "r{}", b)?;
                    need_plus = true;
                }
                if let Some(idx) = index {
                    if need_plus {
                        write!(f, "+")?;
                    }
                    write!(f, "r{}*{}", idx, scale)?;
                    need_plus = true;
                }
                if *displacement != 0 || !need_plus {
                    if need_plus && *displacement >= 0 {
                        write!(f, "+")?;
                    }
                    write!(f, "{}", displacement)?;
                }
                write!(f, "]")
            }
            MachineOperand::FrameSlot(offset) => write!(f, "frame[{}]", offset),
            MachineOperand::GlobalSymbol(name) => write!(f, "@{}", name),
            MachineOperand::BlockLabel(id) => write!(f, "bb{}", id),
        }
    }
}

impl MachineOperand {
    /// Returns `true` if this operand is a physical register.
    #[inline]
    pub fn is_register(&self) -> bool {
        matches!(self, MachineOperand::Register(_))
    }

    /// Returns `true` if this operand is a virtual (pre-allocation) register.
    #[inline]
    pub fn is_virtual_register(&self) -> bool {
        matches!(self, MachineOperand::VirtualRegister(_))
    }

    /// Returns `true` if this operand is an immediate value.
    #[inline]
    pub fn is_immediate(&self) -> bool {
        matches!(self, MachineOperand::Immediate(_))
    }

    /// Returns `true` if this operand is a memory reference.
    #[inline]
    pub fn is_memory(&self) -> bool {
        matches!(self, MachineOperand::Memory { .. })
    }

    /// Returns `true` if this operand is a stack frame slot.
    #[inline]
    pub fn is_frame_slot(&self) -> bool {
        matches!(self, MachineOperand::FrameSlot(_))
    }

    /// Returns `true` if this operand references a global symbol.
    #[inline]
    pub fn is_global_symbol(&self) -> bool {
        matches!(self, MachineOperand::GlobalSymbol(_))
    }

    /// Returns `true` if this operand is a basic block label.
    #[inline]
    pub fn is_block_label(&self) -> bool {
        matches!(self, MachineOperand::BlockLabel(_))
    }

    /// If this is a `Register`, return the physical register index.
    #[inline]
    pub fn as_register(&self) -> Option<u16> {
        match self {
            MachineOperand::Register(r) => Some(*r),
            _ => None,
        }
    }

    /// Returns the register number regardless of whether this is a
    /// `Register` or a `VirtualRegister`.  Useful when converting
    /// back to architecture-specific instructions after register
    /// allocation, where surviving VirtualRegisters are a valid
    /// fallback.
    #[inline]
    pub fn as_any_register(&self) -> Option<u16> {
        match self {
            MachineOperand::Register(r) => Some(*r),
            MachineOperand::VirtualRegister(v) => Some(*v as u16),
            _ => None,
        }
    }

    /// If this is a `VirtualRegister`, return the virtual register number.
    #[inline]
    pub fn as_virtual_register(&self) -> Option<u32> {
        match self {
            MachineOperand::VirtualRegister(v) => Some(*v),
            _ => None,
        }
    }

    /// If this is an `Immediate`, return the constant value.
    #[inline]
    pub fn as_immediate(&self) -> Option<i64> {
        match self {
            MachineOperand::Immediate(imm) => Some(*imm),
            _ => None,
        }
    }

    /// If this is a `BlockLabel`, return the block index.
    #[inline]
    pub fn as_block_label(&self) -> Option<u32> {
        match self {
            MachineOperand::BlockLabel(id) => Some(*id),
            _ => None,
        }
    }
}

// ===========================================================================
// MachineInstruction — architecture-agnostic instruction container
// ===========================================================================

/// A single machine-level instruction after instruction selection.
///
/// `MachineInstruction` is the architecture-agnostic container that carries
/// an opcode (architecture-specific), operands ([`MachineOperand`]), an
/// optional result operand, control-flow flags, and the final encoded
/// machine code bytes (filled by the assembler).
///
/// # Lifecycle
///
/// 1. **Instruction selection** creates `MachineInstruction` with opcode,
///    operands, and result — `encoded_bytes` is empty at this stage.
/// 2. **Register allocation** replaces `VirtualRegister` operands with
///    physical `Register` operands.
/// 3. **Assembly encoding** fills `encoded_bytes` with the final machine
///    code for the instruction.
///
/// # Control-Flow Flags
///
/// - `is_terminator`: This instruction ends a basic block (branch, return).
/// - `is_call`: This instruction is a function call (affects register liveness).
/// - `is_branch`: This instruction transfers control to another block.
///
/// These flags are not mutually exclusive — a conditional call could be
/// both `is_call` and `is_branch`.
#[derive(Debug, Clone)]
pub struct MachineInstruction {
    /// Architecture-specific opcode identifying the instruction.
    ///
    /// Opcode values are defined by each architecture's `codegen.rs` module.
    /// The common infrastructure treats this as an opaque `u32`.
    pub opcode: u32,

    /// Input operands consumed by this instruction.
    ///
    /// The order and meaning of operands is opcode-dependent and defined
    /// by each architecture's instruction encoding rules.
    pub operands: Vec<MachineOperand>,

    /// Optional result operand (the register or location written by this
    /// instruction).
    ///
    /// Instructions that produce no result (e.g., stores, branches) set
    /// this to `None`.
    pub result: Option<MachineOperand>,

    /// `true` if this instruction terminates its basic block.
    ///
    /// Terminator instructions include unconditional branches, conditional
    /// branches, returns, switches, and unreachable traps.  A basic block
    /// must have exactly one terminator as its last instruction.
    pub is_terminator: bool,

    /// `true` if this instruction is a function call.
    ///
    /// Call instructions clobber caller-saved registers and potentially
    /// modify memory.  The register allocator uses this flag to determine
    /// which registers must be saved across the call.
    pub is_call: bool,

    /// `true` if this instruction is part of a call argument setup sequence.
    ///
    /// Set by `select_call()` on MOV instructions that place values into
    /// argument registers for an upcoming CALL.  The conflict resolver uses
    /// this flag to distinguish call-arg MOVs from parameter copies that
    /// happen to target the same registers.
    pub is_call_arg_setup: bool,

    /// `true` if this instruction transfers control to another basic block.
    ///
    /// Branch instructions (conditional and unconditional) and switch
    /// dispatches set this flag.
    pub is_branch: bool,

    /// Operand size in bytes for this instruction (0 = use architecture default).
    ///
    /// When non-zero, the assembler encoder uses this to select the correct
    /// encoding width (e.g., 1=8-bit, 2=16-bit, 4=32-bit, 8=64-bit on x86).
    /// When zero, the architecture's default operand size is used (typically
    /// 8 bytes on x86-64, 4 bytes on i686/AArch64/RISC-V).
    pub operand_size: u8,

    /// Final encoded machine code bytes produced by the assembler.
    ///
    /// This vector is empty after instruction selection and register
    /// allocation — it is filled during the assembly encoding phase by
    /// the architecture-specific encoder.
    pub encoded_bytes: Vec<u8>,

    /// Inline assembly template string (AT&T syntax).
    ///
    /// When this instruction represents an inline assembly statement, this
    /// field contains the template string with `%0`, `%1`, etc. placeholders
    /// that must be substituted with the physical registers assigned during
    /// register allocation.  The assembler processes this template after
    /// register allocation to emit the corresponding machine code.
    ///
    /// `None` for all non-inline-assembly instructions.
    pub asm_template: Option<String>,

    /// Inline assembly clobber registers.
    ///
    /// Physical register numbers clobbered by the inline assembly statement.
    /// The register allocator treats these registers as unavailable across
    /// this instruction. Empty for non-inline-asm instructions.
    pub asm_clobbers: Vec<String>,

    /// Number of output operands in the inline assembly statement.
    ///
    /// Used during template substitution to determine which `%N` placeholders
    /// refer to the result register (outputs: `%0`..`%(asm_num_outputs-1)`)
    /// versus input operands stored in `self.operands`.
    /// Template `%N` where N >= asm_num_outputs maps to `self.operands[N - asm_num_outputs]`.
    /// Zero for non-inline-asm instructions.
    pub asm_num_outputs: usize,

    /// Architecture-specific condition code (used by AArch64 CSET, B.cond, etc.).
    ///
    /// This field preserves the condition code through the
    /// `A64Instruction → MachineInstruction → A64Instruction` round-trip
    /// that occurs during register allocation.  The value is the raw 4-bit
    /// AArch64 condition encoding (0=EQ, 1=NE, 2=CS, …, 14=AL).
    /// `None` for instructions that do not use a condition code.
    pub cond_code: Option<u8>,

    /// Architecture-specific shift/rotate encoding (used by AArch64).
    ///
    /// For MOVZ/MOVK/MOVN this is the `hw * 16` halfword shift.
    /// For register-shifted instructions this packs shift-type and amount.
    /// Preserved through the MachineInstruction round-trip.
    pub arch_shift: u8,
}

impl MachineInstruction {
    /// Create a new machine instruction with the given opcode.
    ///
    /// All fields except `opcode` are initialized to empty/default values.
    /// Use the builder methods to add operands, results, and flags.
    pub fn new(opcode: u32) -> Self {
        MachineInstruction {
            opcode,
            operands: Vec::new(),
            result: None,
            is_terminator: false,
            is_call: false,
            is_call_arg_setup: false,
            is_branch: false,
            operand_size: 0,
            encoded_bytes: Vec::new(),
            asm_template: None,
            asm_clobbers: Vec::new(),
            asm_num_outputs: 0,
            cond_code: None,
            arch_shift: 0,
        }
    }

    /// Add an input operand to this instruction.
    ///
    /// Returns `&mut Self` for fluent chaining.
    #[inline]
    pub fn with_operand(mut self, op: MachineOperand) -> Self {
        self.operands.push(op);
        self
    }

    /// Set the result operand for this instruction.
    ///
    /// Returns `&mut Self` for fluent chaining.
    #[inline]
    pub fn with_result(mut self, result: MachineOperand) -> Self {
        self.result = Some(result);
        self
    }

    /// Mark this instruction as a basic block terminator.
    #[inline]
    pub fn set_terminator(mut self) -> Self {
        self.is_terminator = true;
        self
    }

    /// Mark this instruction as a function call.
    #[inline]
    pub fn set_call(mut self) -> Self {
        self.is_call = true;
        self
    }

    /// Mark this instruction as a branch.
    #[inline]
    pub fn set_branch(mut self) -> Self {
        self.is_branch = true;
        self
    }

    /// Returns the number of input operands.
    #[inline]
    pub fn operand_count(&self) -> usize {
        self.operands.len()
    }

    /// Returns `true` if this instruction has encoded bytes (has been assembled).
    #[inline]
    pub fn is_encoded(&self) -> bool {
        !self.encoded_bytes.is_empty()
    }

    /// Returns the encoded size in bytes (0 if not yet assembled).
    #[inline]
    pub fn encoded_size(&self) -> usize {
        self.encoded_bytes.len()
    }

    /// Collect all physical register operands referenced by this instruction
    /// (both inputs and result).
    pub fn referenced_registers(&self) -> Vec<u16> {
        let mut regs = Vec::new();
        for op in &self.operands {
            if let MachineOperand::Register(r) = op {
                regs.push(*r);
            }
            if let MachineOperand::Memory { base, index, .. } = op {
                if let Some(b) = base {
                    regs.push(*b);
                }
                if let Some(i) = index {
                    regs.push(*i);
                }
            }
        }
        if let Some(MachineOperand::Register(r)) = &self.result {
            regs.push(*r);
        }
        regs
    }

    /// Collect all virtual register operands referenced by this instruction.
    pub fn referenced_virtual_registers(&self) -> Vec<u32> {
        let mut vregs = Vec::new();
        for op in &self.operands {
            if let MachineOperand::VirtualRegister(v) = op {
                vregs.push(*v);
            }
        }
        if let Some(MachineOperand::VirtualRegister(v)) = &self.result {
            vregs.push(*v);
        }
        vregs
    }
}

impl fmt::Display for MachineInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref result) = self.result {
            write!(f, "{} = ", result)?;
        }
        write!(f, "op{}", self.opcode)?;
        for (i, op) in self.operands.iter().enumerate() {
            if i == 0 {
                write!(f, " {}", op)?;
            } else {
                write!(f, ", {}", op)?;
            }
        }
        Ok(())
    }
}

// ===========================================================================
// SpillSlot — stack frame spill slot descriptor
// ===========================================================================

/// A spill slot in the function's stack frame.
///
/// When the register allocator runs out of physical registers, it "spills"
/// a virtual register to the stack by allocating a `SpillSlot`.  The
/// prologue/epilogue generator uses spill slot information to compute the
/// total frame size.
///
/// # Fields
///
/// - `offset`: Signed byte offset from the frame pointer (negative on
///   architectures where the stack grows downward).
/// - `size`: Size of the spill slot in bytes (must accommodate the spilled
///   value's type — typically 4 or 8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillSlot {
    /// Byte offset from the frame pointer.
    ///
    /// Negative values indicate positions below the frame pointer (the
    /// common layout on x86-64 and AArch64 where the stack grows downward).
    pub offset: i32,

    /// Size of the spill slot in bytes.
    ///
    /// Matches the width of the spilled value (e.g., 4 for `i32`, 8 for
    /// `i64` or pointer, 16 for `i128` or SIMD values).
    pub size: usize,
}

impl SpillSlot {
    /// Create a new spill slot at the given offset with the given size.
    #[inline]
    pub fn new(offset: i32, size: usize) -> Self {
        SpillSlot { offset, size }
    }
}

impl fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "spill[offset={}, size={}]", self.offset, self.size)
    }
}

// ===========================================================================
// MachineBasicBlock — ordered instruction sequence with CFG edges
// ===========================================================================

/// A basic block in a machine-level function.
///
/// `MachineBasicBlock` mirrors the IR [`BasicBlock`](crate::ir::basic_block::BasicBlock)
/// at the machine instruction level.  Each block has an ordered list of
/// [`MachineInstruction`]s (the last of which must be a terminator), an
/// optional human-readable label, and predecessor/successor edges forming
/// the control flow graph.
///
/// # Block Ordering
///
/// Block ordering within [`MachineFunction::blocks`] is significant:
/// - `blocks[0]` is the function entry point.
/// - Code emission traverses blocks in vector order.
/// - The label is used for branch target resolution during assembly.
#[derive(Debug, Clone)]
pub struct MachineBasicBlock {
    /// Optional human-readable label for this block.
    ///
    /// Used for branch target resolution during assembly encoding and
    /// for debug/display output.  Entry blocks are typically labeled
    /// `"entry"` or the function name.
    pub label: Option<String>,

    /// Ordered list of machine instructions in this block.
    ///
    /// The last instruction must be a terminator (`is_terminator == true`).
    /// Instructions are executed sequentially from first to last.
    pub instructions: Vec<MachineInstruction>,

    /// Indices of predecessor blocks in the containing [`MachineFunction`].
    ///
    /// A predecessor is a block whose terminator can transfer control to
    /// this block.  Used for dataflow analysis and register allocation.
    pub predecessors: Vec<usize>,

    /// Indices of successor blocks in the containing [`MachineFunction`].
    ///
    /// A successor is a block to which this block's terminator can
    /// transfer control.
    pub successors: Vec<usize>,

    /// Whether this block is the target of an indirect branch.
    ///
    /// Set to `true` when the block is reached via computed goto, switch
    /// table indirect jump, `asm goto`, or any other indirect control
    /// transfer.  Used by CET/IBT (x86-64) to insert `endbr64` at the
    /// correct locations.
    pub is_indirect_target: bool,
}

impl MachineBasicBlock {
    /// Create a new empty basic block with an optional label.
    pub fn new(label: Option<String>) -> Self {
        MachineBasicBlock {
            label,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
            is_indirect_target: false,
        }
    }

    /// Create a new empty basic block with a label.
    #[inline]
    pub fn with_label(label: String) -> Self {
        MachineBasicBlock::new(Some(label))
    }

    /// Append an instruction to the end of this block.
    #[inline]
    pub fn push_instruction(&mut self, inst: MachineInstruction) {
        self.instructions.push(inst);
    }

    /// Add a predecessor block index.
    #[inline]
    pub fn add_predecessor(&mut self, pred: usize) {
        if !self.predecessors.contains(&pred) {
            self.predecessors.push(pred);
        }
    }

    /// Add a successor block index.
    #[inline]
    pub fn add_successor(&mut self, succ: usize) {
        if !self.successors.contains(&succ) {
            self.successors.push(succ);
        }
    }

    /// Returns the number of instructions in this block.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` if this block has no instructions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Returns the total encoded size of all instructions in this block.
    pub fn encoded_size(&self) -> usize {
        self.instructions
            .iter()
            .map(|inst| inst.encoded_bytes.len())
            .sum()
    }
}

impl fmt::Display for MachineBasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            writeln!(f, "{}:", label)?;
        }
        for inst in &self.instructions {
            writeln!(f, "  {}", inst)?;
        }
        Ok(())
    }
}

// ===========================================================================
// MachineFunction — post-selection, pre-regalloc function representation
// ===========================================================================

/// A machine-level function after instruction selection.
///
/// `MachineFunction` is produced by [`ArchCodegen::lower_function`] and
/// consumed by the register allocator, prologue/epilogue emitter, and
/// assembly encoder.  It contains:
///
/// - The function's symbol `name` (used in the ELF symbol table)
/// - An ordered list of [`MachineBasicBlock`]s forming the control flow
///   graph — `blocks[0]` is always the entry block
/// - Stack frame metadata: `frame_size`, `spill_slots`, `callee_saved_regs`
/// - The `is_leaf` flag indicating whether this function makes any calls
///   (enables red-zone optimization on x86-64 for leaf functions)
///
/// # Frame Layout
///
/// The `frame_size` field is initially computed by instruction selection
/// (accounting for local variable allocas and alignment), then updated by
/// the register allocator when spill slots are added.  The prologue/
/// epilogue generator uses the final `frame_size` to emit the stack
/// adjustment instructions.
#[derive(Debug, Clone)]
pub struct MachineFunction {
    /// Function name — the symbol name in the output object file.
    pub name: String,

    /// Ordered list of basic blocks forming the function body.
    ///
    /// `blocks[0]` is the entry block.  Block ordering is significant
    /// for code emission — blocks are laid out in vector order.
    pub blocks: Vec<MachineBasicBlock>,

    /// Total stack frame size in bytes.
    ///
    /// Includes space for local variables, spill slots, callee-saved
    /// register save area, and alignment padding.  Updated by the
    /// register allocator when spill slots are allocated.
    pub frame_size: usize,

    /// Spill slots allocated by the register allocator.
    ///
    /// Each slot describes an offset and size within the stack frame
    /// used to store a spilled virtual register value.
    pub spill_slots: Vec<SpillSlot>,

    /// Physical registers that this function uses and that the ABI
    /// requires to be preserved across calls (callee-saved).
    ///
    /// The prologue must save these registers and the epilogue must
    /// restore them.  Populated by the register allocator based on
    /// which callee-saved registers were assigned.
    pub callee_saved_regs: Vec<u16>,

    /// `true` if this function contains no call instructions.
    ///
    /// Leaf functions on x86-64 can use the 128-byte red zone below
    /// RSP without adjusting the stack pointer, improving performance
    /// for small functions.
    pub is_leaf: bool,

    /// Reverse mapping from codegen virtual register numbers to IR Values.
    ///
    /// Populated by the architecture-specific instruction selector when it
    /// maps IR Values to `MachineOperand::VirtualRegister(vreg)` operands.
    /// Used by [`apply_allocation_result`](crate::backend::generation)
    /// to correctly look up physical register assignments from the register
    /// allocator (which operates on IR `Value` indices).
    pub vreg_to_ir_value: FxHashMap<u32, Value>,

    /// Virtual registers that represent SSE (XMM) values rather than GPR values.
    ///
    /// When these vregs are spilled, the spill code must use an XMM scratch
    /// register (e.g., XMM15) and MOVSD instructions instead of a GPR scratch
    /// (R11) and MOV instructions. Populated during instruction selection.
    pub sse_vregs: crate::common::fx_hash::FxHashSet<u32>,

    /// RBP-relative offset of the variadic GPR register save area (negative).
    /// Set only for variadic functions on x86-64.  The save area holds
    /// 6 GPR slots (RDI, RSI, RDX, RCX, R8, R9) at consecutive offsets.
    pub va_save_area_offset: Option<i32>,

    /// RBP-relative offset of the variadic FP register save area (negative).
    /// Holds 8 XMM slots (XMM0–XMM7, low 64-bit each) at 8-byte offsets.
    pub va_fp_save_area_offset: Option<i32>,

    /// RBP-relative offset of the 16-byte va_control block.
    /// Layout: [gp_ptr (8 bytes), fp_ptr (8 bytes)].
    pub va_control_offset: Option<i32>,

    /// Number of named (non-variadic) GPR parameters.  Used by va_start
    /// to compute the address of the first variadic argument in the save area.
    pub named_gpr_count: usize,

    /// Number of named (non-variadic) FP parameters.  Used by va_start
    /// to compute the address of the first variadic FP argument.
    pub named_fp_count: usize,

    /// Total stack bytes consumed by named MEMORY-class parameters
    /// (long double, large structs) that are passed on the stack but are
    /// NOT register-passed.  Used by va_start to compute the correct
    /// overflow_arg_area pointer past all named stack parameters.
    pub named_memory_stack_bytes: usize,

    /// Number of parameter-loading MOV instructions injected at the start
    /// of the entry block during instruction selection.
    ///
    /// Used by [`resolve_param_load_conflicts`] to limit its scan to only
    /// the true parameter copies (ABI register → virtual register) and
    /// avoid incorrectly including register-allocator-inserted copies that
    /// happen to source from ABI registers.
    pub num_param_moves: usize,
}

impl MachineFunction {
    /// Create a new machine function with the given name.
    ///
    /// The function is initialized with one empty entry block (labeled
    /// with the function name), zero frame size, no spill slots, no
    /// callee-saved registers, and `is_leaf` set to `true` (updated
    /// during instruction selection if calls are emitted).
    pub fn new(name: String) -> Self {
        let entry_label = name.clone();
        let entry = MachineBasicBlock::with_label(entry_label);
        MachineFunction {
            name,
            blocks: vec![entry],
            frame_size: 0,
            spill_slots: Vec::new(),
            callee_saved_regs: Vec::new(),
            is_leaf: true,
            vreg_to_ir_value: FxHashMap::default(),
            sse_vregs: crate::common::fx_hash::FxHashSet::default(),
            va_save_area_offset: None,
            va_fp_save_area_offset: None,
            va_control_offset: None,
            named_gpr_count: 0,
            named_fp_count: 0,
            named_memory_stack_bytes: 0,
            num_param_moves: 0,
        }
    }

    /// Append a new basic block to the function and return its index.
    pub fn add_block(&mut self, block: MachineBasicBlock) -> usize {
        let index = self.blocks.len();
        self.blocks.push(block);
        index
    }

    /// Returns an immutable reference to the entry block (`blocks[0]`).
    ///
    /// # Panics
    ///
    /// Panics if the function has no blocks (should never happen for
    /// well-formed functions created via [`new`](MachineFunction::new)).
    #[inline]
    pub fn entry_block(&self) -> &MachineBasicBlock {
        &self.blocks[0]
    }

    /// Returns a mutable reference to the entry block.
    #[inline]
    pub fn entry_block_mut(&mut self) -> &mut MachineBasicBlock {
        &mut self.blocks[0]
    }

    /// Returns the number of basic blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the total number of instructions across all blocks.
    pub fn instruction_count(&self) -> usize {
        self.blocks.iter().map(|b| b.instruction_count()).sum()
    }

    /// Returns the total encoded size of the function in bytes.
    ///
    /// This is the sum of all encoded instruction bytes across all blocks.
    /// Returns 0 if the function has not been assembled yet.
    pub fn encoded_size(&self) -> usize {
        self.blocks.iter().map(|b| b.encoded_size()).sum()
    }

    /// Allocate a new spill slot with the given size and return its offset.
    ///
    /// The offset is computed relative to the current frame size.  The
    /// frame size is increased by `size` (rounded up to alignment).
    pub fn allocate_spill_slot(&mut self, size: usize) -> SpillSlot {
        // Align size to the nearest power of two ≤ 16.
        let align = size.min(16).next_power_of_two();
        let aligned_frame = (self.frame_size + align - 1) & !(align - 1);
        let offset = -(aligned_frame as i32 + size as i32);
        self.frame_size = aligned_frame + size;
        let slot = SpillSlot::new(offset, size);
        self.spill_slots.push(slot);
        slot
    }

    /// Mark this function as non-leaf (it contains at least one call).
    #[inline]
    pub fn mark_has_calls(&mut self) {
        self.is_leaf = false;
    }
}

impl fmt::Display for MachineFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function {}:", self.name)?;
        writeln!(f, "  frame_size: {}", self.frame_size)?;
        writeln!(f, "  is_leaf: {}", self.is_leaf)?;
        writeln!(f, "  callee_saved: {:?}", self.callee_saved_regs)?;
        writeln!(f, "  spill_slots: {}", self.spill_slots.len())?;
        for (i, block) in self.blocks.iter().enumerate() {
            writeln!(f, "  block {}:", i)?;
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

// ===========================================================================
// RegisterInfo — architecture register set descriptor
// ===========================================================================

/// Describes the register file of a target architecture for the register
/// allocator.
///
/// `RegisterInfo` is returned by [`ArchCodegen::register_info`] and
/// consumed by the register allocator to determine which physical registers
/// are available for allocation, which must be preserved across calls
/// (callee-saved), which are clobbered by calls (caller-saved), and
/// which are reserved for special purposes (stack pointer, frame pointer).
///
/// # Register Numbering
///
/// Registers are identified by `u16` indices.  The mapping from index to
/// hardware register name is architecture-specific and defined in each
/// backend's `registers.rs` module.
///
/// # Register Classes
///
/// Two primary classes are distinguished:
/// - **GPR** (General-Purpose Registers): Used for integer values, pointers,
///   and address computations.
/// - **FPR** (Floating-Point Registers): Used for `float`, `double`, and
///   SIMD values.
///
/// The register allocator allocates from the appropriate class based on
/// the type of each virtual register.
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    /// General-purpose registers available for allocation.
    ///
    /// Excludes reserved registers (SP, FP, etc.).  The register allocator
    /// picks from this set for integer/pointer values.
    pub allocatable_gpr: Vec<u16>,

    /// Floating-point/SIMD registers available for allocation.
    ///
    /// Used for `float`, `double`, `long double`, and vector values.
    pub allocatable_fpr: Vec<u16>,

    /// Registers that the callee must preserve across function calls.
    ///
    /// If the register allocator assigns one of these registers, the
    /// function's prologue must save it and the epilogue must restore it.
    pub callee_saved: Vec<u16>,

    /// Registers that the caller must save before a call (clobbered by callee).
    ///
    /// The register allocator inserts spill code around call instructions
    /// for any live values held in caller-saved registers.
    pub caller_saved: Vec<u16>,

    /// Registers reserved for special purposes — never allocated.
    ///
    /// Typically includes the stack pointer, frame pointer, and
    /// architecture-specific reserved registers (e.g., x18 on AArch64
    /// for platform use, x0/zero on RISC-V).
    pub reserved: Vec<u16>,

    /// Registers used for passing integer/pointer function arguments.
    ///
    /// Ordered by ABI convention (e.g., RDI, RSI, RDX, RCX, R8, R9 for
    /// System V AMD64).
    pub argument_gpr: Vec<u16>,

    /// Registers used for passing floating-point function arguments.
    ///
    /// Ordered by ABI convention (e.g., XMM0–XMM7 for System V AMD64).
    pub argument_fpr: Vec<u16>,

    /// Registers used for returning integer/pointer values.
    ///
    /// Typically one or two registers (e.g., RAX, RDX for System V AMD64).
    pub return_gpr: Vec<u16>,

    /// Registers used for returning floating-point values.
    ///
    /// Typically one or two registers (e.g., XMM0, XMM1 for System V AMD64).
    pub return_fpr: Vec<u16>,
}

impl RegisterInfo {
    /// Create an empty `RegisterInfo` with no registers in any category.
    ///
    /// Use the builder methods or direct field assignment to populate.
    pub fn new() -> Self {
        RegisterInfo {
            allocatable_gpr: Vec::new(),
            allocatable_fpr: Vec::new(),
            callee_saved: Vec::new(),
            caller_saved: Vec::new(),
            reserved: Vec::new(),
            argument_gpr: Vec::new(),
            argument_fpr: Vec::new(),
            return_gpr: Vec::new(),
            return_fpr: Vec::new(),
        }
    }

    /// Returns the total number of allocatable registers (GPR + FPR).
    #[inline]
    pub fn total_allocatable(&self) -> usize {
        self.allocatable_gpr.len() + self.allocatable_fpr.len()
    }

    /// Returns `true` if the given register is reserved (should not be
    /// allocated).
    pub fn is_reserved(&self, reg: u16) -> bool {
        self.reserved.contains(&reg)
    }

    /// Returns `true` if the given register is callee-saved.
    pub fn is_callee_saved(&self, reg: u16) -> bool {
        self.callee_saved.contains(&reg)
    }

    /// Returns `true` if the given register is caller-saved.
    pub fn is_caller_saved(&self, reg: u16) -> bool {
        self.caller_saved.contains(&reg)
    }

    /// Returns `true` if the given register is used for integer argument
    /// passing.
    pub fn is_argument_gpr(&self, reg: u16) -> bool {
        self.argument_gpr.contains(&reg)
    }
}

impl Default for RegisterInfo {
    fn default() -> Self {
        RegisterInfo::new()
    }
}

// ===========================================================================
// RelocationTypeInfo — architecture-specific relocation metadata
// ===========================================================================

/// Result of assembling a single [`MachineFunction`] into machine code.
///
/// Contains the encoded instruction bytes and any unresolved relocations
/// that must be applied later (either by resolving intra-module references
/// during text section concatenation, or by writing `.rela.text` entries
/// into the relocatable ELF object for the linker to process).
#[derive(Debug, Clone)]
pub struct AssembledFunction {
    /// Encoded machine code bytes for this function.
    pub bytes: Vec<u8>,
    /// Unresolved relocations emitted by the assembler.
    ///
    /// Each entry has an `offset` relative to the start of *this* function's
    /// bytes, a `symbol` name, a relocation type (encoded as `rel_type_id`),
    /// and an `addend`.  The code generation driver adjusts offsets to be
    /// relative to the combined `.text` section before writing them.
    pub relocations: Vec<FunctionRelocation>,
}

/// A single unresolved relocation from function assembly.
///
/// This is a simplified, architecture-neutral relocation record used to
/// transport relocation information from the assembler through the code
/// generation pipeline.  It is converted to architecture-specific ELF
/// relocation entries when writing the relocatable object file.
#[derive(Debug, Clone)]
pub struct FunctionRelocation {
    /// Byte offset within the function's encoded bytes where the relocation
    /// must be applied.
    pub offset: u64,
    /// Target symbol name (e.g., `"add"`, `"printf"`, `".Lstr0"`).
    pub symbol: String,
    /// ELF relocation type ID (architecture-specific).
    pub rel_type_id: u32,
    /// Addend value for the relocation computation.
    pub addend: i64,
    /// Section name this relocation targets (usually `".text"`).
    pub section: String,
}

/// Metadata describing an architecture-specific ELF relocation type.
///
/// `RelocationTypeInfo` is returned by [`ArchCodegen::relocation_types`]
/// to describe the set of relocation types that an architecture's assembler
/// and linker can produce and consume.  The common linker infrastructure
/// uses this metadata to dispatch relocation application to the correct
/// architecture-specific handler.
///
/// # Fields
///
/// - `name`: Human-readable name (e.g., `"R_X86_64_PC32"`, `"R_AARCH64_CALL26"`).
/// - `type_id`: Numeric ELF relocation type value (from the ELF ABI supplement).
/// - `size`: Width of the relocation field in bytes (1, 2, 4, or 8).
/// - `is_pc_relative`: Whether the relocation computes a PC-relative offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelocationTypeInfo {
    /// Human-readable name of the relocation type.
    ///
    /// Used for diagnostics, debug output, and error messages.  Must match
    /// the canonical ELF ABI supplement name (e.g., `"R_X86_64_PC32"`).
    pub name: &'static str,

    /// Numeric ELF relocation type value.
    ///
    /// This is the value written to the `r_info` field of `Elf64_Rela`
    /// or `Elf32_Rel` entries.  Values are architecture-specific and
    /// drawn from the ELF ABI supplement for each target.
    pub type_id: u32,

    /// Width of the relocation field in bytes.
    ///
    /// Determines how many bytes at the relocation site are patched:
    /// - `1`: 8-bit relocation
    /// - `2`: 16-bit relocation
    /// - `4`: 32-bit relocation (most common)
    /// - `8`: 64-bit relocation
    pub size: u8,

    /// `true` if this relocation computes a PC-relative offset.
    ///
    /// PC-relative relocations subtract the relocation site address from
    /// the target symbol address.  They are essential for PIC code and
    /// RIP-relative addressing on x86-64.
    pub is_pc_relative: bool,
}

impl RelocationTypeInfo {
    /// Create a new relocation type descriptor.
    #[inline]
    pub const fn new(name: &'static str, type_id: u32, size: u8, is_pc_relative: bool) -> Self {
        RelocationTypeInfo {
            name,
            type_id,
            size,
            is_pc_relative,
        }
    }
}

impl fmt::Display for RelocationTypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(id={}, size={}, pc_rel={})",
            self.name, self.type_id, self.size, self.is_pc_relative
        )
    }
}

// ===========================================================================
// ArgLocation — ABI argument/return value placement descriptor
// ===========================================================================

/// Describes where a function argument or return value is placed according
/// to the target architecture's calling convention.
///
/// [`ArchCodegen::classify_argument`] and [`ArchCodegen::classify_return`]
/// use this enum to communicate ABI decisions to the instruction selector
/// and prologue/epilogue generator.
///
/// # Variants
///
/// | Variant        | Meaning                                           |
/// |----------------|---------------------------------------------------|
/// | `Register`     | Value passed in a single physical register         |
/// | `RegisterPair` | Value split across two registers (e.g., 128-bit)  |
/// | `Stack`        | Value passed on the stack at a given offset        |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgLocation {
    /// The value is passed in a single physical register.
    ///
    /// The `u16` is the physical register index from the architecture's
    /// register numbering scheme.
    Register(u16),

    /// The value is split across two physical registers.
    ///
    /// Used for types that don't fit in a single register but can be
    /// decomposed into two halves (e.g., 128-bit integers on x86-64
    /// using RAX+RDX, or large structs split across two GPRs).
    ///
    /// The first register holds the low half and the second holds the
    /// high half.
    RegisterPair(u16, u16),

    /// The value is passed on the stack at the given byte offset from
    /// the stack pointer at the call site.
    ///
    /// The offset is non-negative and measured from the base of the
    /// argument area (which is at or above the current SP depending on
    /// the architecture's calling convention).
    Stack(i32),
}

impl ArgLocation {
    /// Returns `true` if this argument is passed in a register (single
    /// or pair).
    #[inline]
    pub fn is_register(&self) -> bool {
        matches!(
            self,
            ArgLocation::Register(_) | ArgLocation::RegisterPair(_, _)
        )
    }

    /// Returns `true` if this argument is passed on the stack.
    #[inline]
    pub fn is_stack(&self) -> bool {
        matches!(self, ArgLocation::Stack(_))
    }

    /// If this is a single-register location, return the register index.
    #[inline]
    pub fn as_register(&self) -> Option<u16> {
        match self {
            ArgLocation::Register(r) => Some(*r),
            _ => None,
        }
    }
}

impl fmt::Display for ArgLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgLocation::Register(r) => write!(f, "reg(r{})", r),
            ArgLocation::RegisterPair(r1, r2) => write!(f, "regpair(r{}, r{})", r1, r2),
            ArgLocation::Stack(offset) => write!(f, "stack({})", offset),
        }
    }
}

// ===========================================================================
// ArchCodegen — the core architecture abstraction trait
// ===========================================================================

/// The core architecture abstraction trait for BCC's backend.
///
/// `ArchCodegen` is the **single dispatch point** through which the
/// architecture-agnostic code generation driver interacts with
/// architecture-specific backends.  Each of the four target architectures
/// (x86-64, i686, AArch64, RISC-V 64) provides an implementation of this
/// trait.
///
/// # Required Methods
///
/// | Method                | Purpose                                          |
/// |-----------------------|--------------------------------------------------|
/// | `lower_function`      | Instruction selection: IR → machine instructions |
/// | `emit_assembly`       | Encode machine instructions to bytes             |
/// | `target`              | Identify the target architecture                 |
/// | `register_info`       | Provide register set for register allocation     |
/// | `relocation_types`    | Describe supported relocation types              |
/// | `emit_prologue`       | Generate function prologue instructions          |
/// | `emit_epilogue`       | Generate function epilogue instructions          |
/// | `frame_pointer_reg`   | Frame pointer register index                     |
/// | `stack_pointer_reg`   | Stack pointer register index                     |
/// | `return_address_reg`  | Return address register (if any)                 |
/// | `classify_argument`   | ABI: where to place a function argument          |
/// | `classify_return`     | ABI: where to place a return value               |
///
/// # Usage
///
/// The code generation driver ([`crate::backend::generation`]) dispatches
/// to the appropriate implementation based on the `--target` CLI flag:
///
/// ```ignore
/// fn generate_for_arch(
///     arch: &dyn ArchCodegen,
///     func: &IrFunction,
///     diag: &mut DiagnosticEngine,
/// ) -> Result<Vec<u8>, String> {
///     let mf = arch.lower_function(func, diag)?;
///     let bytes = arch.emit_assembly(&mf)?;
///     Ok(bytes)
/// }
/// ```
///
/// # Implementors
///
/// - [`crate::backend::x86_64`] — System V AMD64 ABI, 16 GPRs, SSE2,
///   security mitigations (retpoline, CET, stack probe)
/// - [`crate::backend::i686`] — cdecl / System V i386 ABI, 8 GPRs, x87 FPU
/// - [`crate::backend::aarch64`] — AAPCS64 ABI, 31 GPRs, 32 SIMD/FP regs
/// - [`crate::backend::riscv64`] — LP64D ABI, 32 integer + 32 FP regs
pub trait ArchCodegen {
    /// Perform instruction selection: lower an IR function to machine
    /// instructions.
    ///
    /// This is the primary entry point for converting a semantically
    /// validated, phi-eliminated IR function into architecture-specific
    /// machine instructions.  The produced [`MachineFunction`] contains
    /// virtual registers that will be resolved by the register allocator.
    ///
    /// # Parameters
    ///
    /// - `func`: The IR function to lower.  The instruction selector reads
    ///   `func.name`, `func.params`, `func.return_type`, `func.blocks`,
    ///   `func.calling_convention`, `func.is_variadic`, `func.is_noreturn`,
    ///   `func.entry_block()`, and `func.block_count()`.
    /// - `diag`: The diagnostic engine for reporting errors during
    ///   instruction selection (unsupported constructs, inline asm
    ///   constraint failures, etc.).
    ///
    /// # Returns
    ///
    /// `Ok(MachineFunction)` on success, or `Err(String)` on fatal error.
    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &FxHashMap<Value, String>,
        global_var_refs: &FxHashMap<Value, String>,
    ) -> Result<MachineFunction, String>;

    /// Encode machine instructions to raw bytes (built-in assembler).
    ///
    /// Iterates over every instruction in every block of the machine
    /// function, encoding each to its binary representation and filling
    /// the `encoded_bytes` field.  Also resolves intra-function branch
    /// targets and produces relocation entries for cross-function and
    /// cross-section references.
    ///
    /// # Parameters
    ///
    /// - `mf`: The machine function (post-register-allocation) to assemble.
    ///
    /// # Returns
    ///
    /// `Ok(AssembledFunction)` containing the encoded function body and any
    /// unresolved relocations, or `Err(String)` on encoding failure.
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<AssembledFunction, String>;

    /// Returns the target architecture for this codegen implementation.
    ///
    /// Used by the code generation driver to confirm correct dispatch and
    /// to provide target information to the ELF writer, DWARF emitter,
    /// and linker.
    fn target(&self) -> Target;

    /// Returns the register set information for this architecture.
    ///
    /// The returned [`RegisterInfo`] describes all available registers,
    /// their classification (allocatable, callee-saved, caller-saved,
    /// reserved), and their roles in the calling convention (argument
    /// and return registers).
    ///
    /// Consumed by the register allocator to make allocation decisions.
    fn register_info(&self) -> RegisterInfo;

    /// Returns the architecture-specific relocation types supported by
    /// this backend's assembler and linker.
    ///
    /// The common linker infrastructure uses these descriptors to dispatch
    /// relocation application to the correct architecture-specific handler.
    fn relocation_types(&self) -> &[RelocationTypeInfo];

    /// Generate function prologue instructions.
    ///
    /// The prologue sets up the stack frame:
    /// 1. Save the frame pointer (if used).
    /// 2. Adjust the stack pointer by `frame_size`.
    /// 3. Save callee-saved registers.
    /// 4. On x86-64 with frames > 4096 bytes: emit stack probe loop.
    ///
    /// # Parameters
    ///
    /// - `mf`: The machine function whose frame metadata drives prologue
    ///   generation (`frame_size`, `callee_saved_regs`, `is_leaf`).
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction>;

    /// Generate function epilogue instructions.
    ///
    /// The epilogue tears down the stack frame:
    /// 1. Restore callee-saved registers.
    /// 2. Restore the frame pointer.
    /// 3. Adjust the stack pointer.
    /// 4. Return to the caller.
    ///
    /// # Parameters
    ///
    /// - `mf`: The machine function whose frame metadata drives epilogue
    ///   generation.
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction>;

    /// Returns the physical register index of the frame pointer.
    ///
    /// | Architecture | Register | Index |
    /// |-------------|----------|-------|
    /// | x86-64      | RBP      | arch-defined |
    /// | i686        | EBP      | arch-defined |
    /// | AArch64     | X29 (FP) | arch-defined |
    /// | RISC-V 64   | x8 (s0)  | arch-defined |
    fn frame_pointer_reg(&self) -> u16;

    /// Returns the physical register index of the stack pointer.
    ///
    /// | Architecture | Register | Index |
    /// |-------------|----------|-------|
    /// | x86-64      | RSP      | arch-defined |
    /// | i686        | ESP      | arch-defined |
    /// | AArch64     | SP       | arch-defined |
    /// | RISC-V 64   | x2 (sp)  | arch-defined |
    fn stack_pointer_reg(&self) -> u16;

    /// Returns the physical register holding the return address, if the
    /// architecture has one.
    ///
    /// - x86-64/i686: `None` — return address is on the stack (pushed by
    ///   `CALL`), not in a register.
    /// - AArch64: `Some(LR)` — the link register (X30).
    /// - RISC-V 64: `Some(ra)` — the return address register (x1).
    fn return_address_reg(&self) -> Option<u16>;

    /// Classify where a function argument of the given IR type should be
    /// placed according to the architecture's calling convention.
    ///
    /// The instruction selector calls this method for each argument to
    /// determine whether it goes in a register, register pair, or on the
    /// stack.  The classification depends on the type's size, alignment,
    /// and the number of arguments already classified (consuming available
    /// argument registers).
    ///
    /// # Parameters
    ///
    /// - `ty`: The IR type of the argument.  Implementors may call
    ///   `ty.is_integer()`, `ty.is_float()`, `ty.is_pointer()`,
    ///   `ty.is_struct()`, `ty.is_aggregate()`, and `ty.size_bytes()`
    ///   to make classification decisions.
    ///
    /// # Returns
    ///
    /// An [`ArgLocation`] describing where this argument is placed.
    fn classify_argument(&self, ty: &IrType) -> ArgLocation;

    /// Classify where a function return value of the given IR type should
    /// be placed according to the architecture's calling convention.
    ///
    /// Return values follow architecture-specific rules:
    /// - Small scalars: single register (e.g., RAX on x86-64)
    /// - Large structs: pointer passed in a hidden first argument
    /// - Floating-point: FP register (e.g., XMM0 on x86-64, V0 on AArch64)
    ///
    /// # Parameters
    ///
    /// - `ty`: The IR type of the return value.
    ///
    /// # Returns
    ///
    /// An [`ArgLocation`] describing where the return value is placed.
    fn classify_return(&self, ty: &IrType) -> ArgLocation;

    /// Format a machine instruction as human-readable assembly text.
    ///
    /// The default implementation uses the generic `Display` format
    /// (`op{n} operand1, operand2, ...`). Architecture backends should
    /// override this to produce valid AT&T-syntax (x86-64/i686) or
    /// standard assembly (AArch64/RISC-V 64).
    fn format_instruction(&self, inst: &MachineInstruction) -> String {
        format!("{}", inst)
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- MachineOperand tests --

    #[test]
    fn test_machine_operand_register() {
        let op = MachineOperand::Register(5);
        assert!(op.is_register());
        assert!(!op.is_virtual_register());
        assert!(!op.is_immediate());
        assert!(!op.is_memory());
        assert!(!op.is_frame_slot());
        assert!(!op.is_global_symbol());
        assert!(!op.is_block_label());
        assert_eq!(op.as_register(), Some(5));
        assert_eq!(op.as_virtual_register(), None);
    }

    #[test]
    fn test_machine_operand_virtual_register() {
        let op = MachineOperand::VirtualRegister(42);
        assert!(op.is_virtual_register());
        assert!(!op.is_register());
        assert_eq!(op.as_virtual_register(), Some(42));
        assert_eq!(op.as_register(), None);
    }

    #[test]
    fn test_machine_operand_immediate() {
        let op = MachineOperand::Immediate(-128);
        assert!(op.is_immediate());
        assert_eq!(op.as_immediate(), Some(-128));
    }

    #[test]
    fn test_machine_operand_memory() {
        let op = MachineOperand::Memory {
            base: Some(4),
            index: Some(2),
            scale: 8,
            displacement: -16,
        };
        assert!(op.is_memory());
        assert!(!op.is_register());
    }

    #[test]
    fn test_machine_operand_frame_slot() {
        let op = MachineOperand::FrameSlot(-8);
        assert!(op.is_frame_slot());
    }

    #[test]
    fn test_machine_operand_global_symbol() {
        let op = MachineOperand::GlobalSymbol("printf".to_string());
        assert!(op.is_global_symbol());
    }

    #[test]
    fn test_machine_operand_block_label() {
        let op = MachineOperand::BlockLabel(3);
        assert!(op.is_block_label());
        assert_eq!(op.as_block_label(), Some(3));
    }

    #[test]
    fn test_machine_operand_display() {
        assert_eq!(format!("{}", MachineOperand::Register(0)), "r0");
        assert_eq!(format!("{}", MachineOperand::VirtualRegister(7)), "v7");
        assert_eq!(format!("{}", MachineOperand::Immediate(42)), "#42");
        assert_eq!(format!("{}", MachineOperand::FrameSlot(-8)), "frame[-8]");
        assert_eq!(
            format!("{}", MachineOperand::GlobalSymbol("main".to_string())),
            "@main"
        );
        assert_eq!(format!("{}", MachineOperand::BlockLabel(2)), "bb2");
    }

    // -- MachineInstruction tests --

    #[test]
    fn test_machine_instruction_new() {
        let inst = MachineInstruction::new(0x90);
        assert_eq!(inst.opcode, 0x90);
        assert!(inst.operands.is_empty());
        assert!(inst.result.is_none());
        assert!(!inst.is_terminator);
        assert!(!inst.is_call);
        assert!(!inst.is_branch);
        assert!(inst.encoded_bytes.is_empty());
        assert!(!inst.is_encoded());
        assert_eq!(inst.encoded_size(), 0);
    }

    #[test]
    fn test_machine_instruction_builder() {
        let inst = MachineInstruction::new(1)
            .with_operand(MachineOperand::Register(0))
            .with_operand(MachineOperand::Immediate(10))
            .with_result(MachineOperand::VirtualRegister(3))
            .set_call();

        assert_eq!(inst.operand_count(), 2);
        assert!(inst.result.is_some());
        assert!(inst.is_call);
        assert!(!inst.is_terminator);
        assert!(!inst.is_branch);
    }

    #[test]
    fn test_machine_instruction_referenced_registers() {
        let inst = MachineInstruction::new(1)
            .with_operand(MachineOperand::Register(3))
            .with_operand(MachineOperand::Memory {
                base: Some(5),
                index: Some(7),
                scale: 4,
                displacement: 0,
            })
            .with_result(MachineOperand::Register(1));

        let regs = inst.referenced_registers();
        assert_eq!(regs, vec![3, 5, 7, 1]);
    }

    #[test]
    fn test_machine_instruction_referenced_virtual_registers() {
        let inst = MachineInstruction::new(1)
            .with_operand(MachineOperand::VirtualRegister(10))
            .with_operand(MachineOperand::VirtualRegister(20))
            .with_result(MachineOperand::VirtualRegister(30));

        let vregs = inst.referenced_virtual_registers();
        assert_eq!(vregs, vec![10, 20, 30]);
    }

    // -- SpillSlot tests --

    #[test]
    fn test_spill_slot() {
        let slot = SpillSlot::new(-16, 8);
        assert_eq!(slot.offset, -16);
        assert_eq!(slot.size, 8);
        assert_eq!(format!("{}", slot), "spill[offset=-16, size=8]");
    }

    // -- MachineBasicBlock tests --

    #[test]
    fn test_machine_basic_block_new() {
        let bb = MachineBasicBlock::new(None);
        assert!(bb.label.is_none());
        assert!(bb.instructions.is_empty());
        assert!(bb.predecessors.is_empty());
        assert!(bb.successors.is_empty());
        assert!(bb.is_empty());
        assert_eq!(bb.instruction_count(), 0);
    }

    #[test]
    fn test_machine_basic_block_with_label() {
        let bb = MachineBasicBlock::with_label("entry".to_string());
        assert_eq!(bb.label, Some("entry".to_string()));
    }

    #[test]
    fn test_machine_basic_block_push_instruction() {
        let mut bb = MachineBasicBlock::new(None);
        bb.push_instruction(MachineInstruction::new(1));
        bb.push_instruction(MachineInstruction::new(2));
        assert_eq!(bb.instruction_count(), 2);
        assert!(!bb.is_empty());
    }

    #[test]
    fn test_machine_basic_block_predecessors_successors() {
        let mut bb = MachineBasicBlock::new(None);
        bb.add_predecessor(0);
        bb.add_predecessor(1);
        bb.add_predecessor(0); // duplicate — should not be added
        bb.add_successor(2);
        bb.add_successor(3);
        bb.add_successor(2); // duplicate — should not be added

        assert_eq!(bb.predecessors, vec![0, 1]);
        assert_eq!(bb.successors, vec![2, 3]);
    }

    // -- MachineFunction tests --

    #[test]
    fn test_machine_function_new() {
        let mf = MachineFunction::new("main".to_string());
        assert_eq!(mf.name, "main");
        assert_eq!(mf.block_count(), 1); // entry block
        assert_eq!(mf.frame_size, 0);
        assert!(mf.spill_slots.is_empty());
        assert!(mf.callee_saved_regs.is_empty());
        assert!(mf.is_leaf);
        assert_eq!(mf.instruction_count(), 0);
        assert_eq!(mf.encoded_size(), 0);
    }

    #[test]
    fn test_machine_function_add_block() {
        let mut mf = MachineFunction::new("f".to_string());
        let idx = mf.add_block(MachineBasicBlock::with_label("bb1".to_string()));
        assert_eq!(idx, 1);
        assert_eq!(mf.block_count(), 2);
    }

    #[test]
    fn test_machine_function_mark_has_calls() {
        let mut mf = MachineFunction::new("f".to_string());
        assert!(mf.is_leaf);
        mf.mark_has_calls();
        assert!(!mf.is_leaf);
    }

    #[test]
    fn test_machine_function_allocate_spill_slot() {
        let mut mf = MachineFunction::new("f".to_string());
        let slot1 = mf.allocate_spill_slot(8);
        assert_eq!(slot1.size, 8);
        assert!(slot1.offset < 0); // below frame pointer
        assert_eq!(mf.spill_slots.len(), 1);
        assert!(mf.frame_size >= 8);

        let slot2 = mf.allocate_spill_slot(4);
        assert_eq!(slot2.size, 4);
        assert_eq!(mf.spill_slots.len(), 2);
    }

    // -- RegisterInfo tests --

    #[test]
    fn test_register_info_new() {
        let ri = RegisterInfo::new();
        assert!(ri.allocatable_gpr.is_empty());
        assert!(ri.allocatable_fpr.is_empty());
        assert_eq!(ri.total_allocatable(), 0);
    }

    #[test]
    fn test_register_info_queries() {
        let ri = RegisterInfo {
            allocatable_gpr: vec![0, 1, 2, 3],
            allocatable_fpr: vec![16, 17],
            callee_saved: vec![3, 4, 5],
            caller_saved: vec![0, 1, 2],
            reserved: vec![6, 7],
            argument_gpr: vec![0, 1, 2],
            argument_fpr: vec![16],
            return_gpr: vec![0],
            return_fpr: vec![16],
        };
        assert_eq!(ri.total_allocatable(), 6);
        assert!(ri.is_reserved(6));
        assert!(ri.is_reserved(7));
        assert!(!ri.is_reserved(0));
        assert!(ri.is_callee_saved(3));
        assert!(!ri.is_callee_saved(0));
        assert!(ri.is_caller_saved(0));
        assert!(!ri.is_caller_saved(3));
        assert!(ri.is_argument_gpr(1));
        assert!(!ri.is_argument_gpr(16));
    }

    #[test]
    fn test_register_info_default() {
        let ri = RegisterInfo::default();
        assert_eq!(ri.total_allocatable(), 0);
    }

    // -- RelocationTypeInfo tests --

    #[test]
    fn test_relocation_type_info() {
        let rel = RelocationTypeInfo::new("R_X86_64_PC32", 2, 4, true);
        assert_eq!(rel.name, "R_X86_64_PC32");
        assert_eq!(rel.type_id, 2);
        assert_eq!(rel.size, 4);
        assert!(rel.is_pc_relative);
        let display = format!("{}", rel);
        assert!(display.contains("R_X86_64_PC32"));
        assert!(display.contains("pc_rel=true"));
    }

    #[test]
    fn test_relocation_type_info_absolute() {
        let rel = RelocationTypeInfo::new("R_X86_64_64", 1, 8, false);
        assert!(!rel.is_pc_relative);
        assert_eq!(rel.size, 8);
    }

    // -- ArgLocation tests --

    #[test]
    fn test_arg_location_register() {
        let loc = ArgLocation::Register(0);
        assert!(loc.is_register());
        assert!(!loc.is_stack());
        assert_eq!(loc.as_register(), Some(0));
    }

    #[test]
    fn test_arg_location_register_pair() {
        let loc = ArgLocation::RegisterPair(0, 1);
        assert!(loc.is_register());
        assert!(!loc.is_stack());
        assert_eq!(loc.as_register(), None); // only for single Register
    }

    #[test]
    fn test_arg_location_stack() {
        let loc = ArgLocation::Stack(16);
        assert!(!loc.is_register());
        assert!(loc.is_stack());
        assert_eq!(loc.as_register(), None);
    }

    #[test]
    fn test_arg_location_display() {
        assert_eq!(format!("{}", ArgLocation::Register(5)), "reg(r5)");
        assert_eq!(
            format!("{}", ArgLocation::RegisterPair(0, 1)),
            "regpair(r0, r1)"
        );
        assert_eq!(format!("{}", ArgLocation::Stack(8)), "stack(8)");
    }
}
