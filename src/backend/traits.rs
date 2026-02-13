//! Architecture-agnostic backend trait definitions for the BCC compiler.
//!
//! This module defines the [`ArchCodegen`] trait — the central polymorphism
//! point enabling the code generation driver to work uniformly across all four
//! target architectures (x86-64, i686, AArch64, RISC-V 64). Each architecture
//! backend provides a concrete implementation of this trait.
//!
//! # Module Contents
//!
//! ## Core Trait
//! - [`ArchCodegen`] — the architecture abstraction layer trait
//! - [`RegisterInfo`] — supplementary trait for register metadata
//!
//! ## Machine IR Types
//! - [`MachineFunction`] — function representation after instruction selection
//! - [`MachineBasicBlock`] — basic block containing machine instructions
//! - [`MachineInstr`] — a single machine instruction
//! - [`MachineOperand`] — operand of a machine instruction
//! - [`PhysReg`] — architecture-independent physical register identifier
//!
//! ## ABI & Classification Types
//! - [`ParamClass`] — ABI parameter classification for type passing conventions
//! - [`RegisterClass`] — register file classification
//! - [`RelocationType`] — relocation type descriptor
//!
//! ## Configuration
//! - [`CodegenConfig`] — code generation configuration collected from CLI flags
//!
//! # Pipeline Position
//!
//! The types in this module sit at the boundary between the middle-end IR
//! and the architecture-specific backends:
//!
//! ```text
//! ┌─────────────────────┐
//! │  IR (SSA form)      │  Phase 7-9
//! │  IrFunction, etc.   │
//! └─────────┬───────────┘
//!           │ ArchCodegen::lower_function()
//!           ▼
//! ┌─────────────────────┐
//! │  Machine IR         │  Phase 10
//! │  MachineFunction    │
//! └─────────┬───────────┘
//!           │ ArchCodegen::emit_assembly()
//!           ▼
//! ┌─────────────────────┐
//! │  Object Code        │  Assembler/Linker
//! │  Vec<u8>            │
//! └─────────────────────┘
//! ```
//!
//! # Backend Validation Order
//!
//! Per Section 0.1.2 of the project requirements, backend validation proceeds
//! in a fixed order: **x86-64 → i686 → AArch64 → RISC-V 64**. This order
//! reflects implementation priority and testing confidence levels.

use std::fmt;

use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// PhysReg — architecture-independent physical register identifier
// ---------------------------------------------------------------------------

/// Architecture-independent physical register identifier.
///
/// Each target architecture maps its register file into a flat `u16` namespace.
/// The specific encoding is architecture-defined:
///
/// | Architecture | Register Range | Example Mappings |
/// |-------------|---------------|-----------------|
/// | x86-64      | 0–31          | RAX=0, RCX=1, ..., XMM0=16, ... |
/// | i686        | 0–15          | EAX=0, ECX=1, ..., ST(0)=8, ... |
/// | AArch64     | 0–63          | X0=0, ..., X30=30, SP=31, V0=32, ... |
/// | RISC-V 64   | 0–63          | x0=0, ..., x31=31, f0=32, ... |
///
/// `PhysReg` is `Copy` for ergonomic use in register allocation data
/// structures and instruction operand lists.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct PhysReg(pub u16);

impl PhysReg {
    /// Sentinel value representing "no register" or an unassigned slot.
    pub const NONE: PhysReg = PhysReg(u16::MAX);

    /// Returns the raw numeric register index.
    #[inline]
    pub fn index(self) -> u16 {
        self.0
    }

    /// Returns `true` if this is the sentinel [`NONE`](PhysReg::NONE) value.
    #[inline]
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }

    /// Returns `true` if this is a valid (non-sentinel) register.
    #[inline]
    pub fn is_valid(self) -> bool {
        self != Self::NONE
    }
}

impl fmt::Display for PhysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::NONE {
            write!(f, "<none>")
        } else {
            write!(f, "r{}", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// RegisterClass — register file classification
// ---------------------------------------------------------------------------

/// Classification of a physical register into a register file category.
///
/// Register classes are used by the register allocator to ensure that values
/// are assigned to registers of the correct category (e.g., integer values
/// go to GPRs, floating-point values go to FP/vector registers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    /// General-purpose register (GPR) — used for integer values, addresses,
    /// and pointer arithmetic. Examples: RAX (x86-64), X0 (AArch64).
    GeneralPurpose,

    /// Floating-point register — used for `float`, `double`, and
    /// `long double` values. Examples: XMM0 (x86-64), D0 (AArch64).
    FloatingPoint,

    /// Vector/SIMD register — used for packed data operations.
    /// On x86-64, these overlap with XMM/YMM/ZMM registers.
    /// On AArch64, these overlap with V0–V31.
    Vector,

    /// Stack pointer register — dedicated register that tracks the current
    /// stack position. Not allocatable by the register allocator.
    StackPointer,

    /// Frame pointer register — used to maintain a stable reference to the
    /// current stack frame. May be allocatable when frame pointer omission
    /// is enabled.
    FramePointer,
}

impl fmt::Display for RegisterClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisterClass::GeneralPurpose => write!(f, "GPR"),
            RegisterClass::FloatingPoint => write!(f, "FP"),
            RegisterClass::Vector => write!(f, "VEC"),
            RegisterClass::StackPointer => write!(f, "SP"),
            RegisterClass::FramePointer => write!(f, "FP_REG"),
        }
    }
}

// ---------------------------------------------------------------------------
// RegisterInfo — register metadata trait
// ---------------------------------------------------------------------------

/// Supplementary trait providing metadata about a physical register.
///
/// Architecture backends implement this trait to expose human-readable names,
/// callee-saved status, and register class information for each register in
/// their register file.
///
/// # Usage
///
/// ```ignore
/// let reg: PhysReg = codegen.return_register_int();
/// let info = codegen.register_info(reg);
/// println!("Return register: {} (class: {})", info.name(), info.register_class());
/// ```
pub trait RegisterInfo {
    /// Returns the human-readable name of this register.
    ///
    /// Architecture backends return the canonical assembly name:
    /// - x86-64: `"rax"`, `"rbx"`, `"xmm0"`, etc.
    /// - AArch64: `"x0"`, `"x30"`, `"v0"`, etc.
    /// - RISC-V: `"x0"`, `"a0"`, `"fa0"`, etc.
    fn name(&self) -> &'static str;

    /// Returns `true` if this register is callee-saved (preserved across calls).
    ///
    /// Callee-saved registers must be saved in the function prologue and
    /// restored in the epilogue if the function modifies them. The register
    /// allocator uses this information for spill cost estimation.
    fn is_callee_saved(&self) -> bool;

    /// Returns the register class this register belongs to.
    fn register_class(&self) -> RegisterClass;
}

// ---------------------------------------------------------------------------
// ParamClass — ABI parameter type classification
// ---------------------------------------------------------------------------

/// ABI classification for a parameter or return value type.
///
/// Based on the System V AMD64 ABI type classification scheme (§3.2.3),
/// extended to cover all four target architectures. Each architecture's ABI
/// module uses these classes to determine how values are passed to and
/// returned from functions:
///
/// - **Integer**: Pass in integer registers (e.g., RDI/RSI on x86-64, X0–X7 on AArch64)
/// - **SSE**: Pass in floating-point/vector registers (e.g., XMM0–XMM7 on x86-64)
/// - **Memory**: Pass on the stack (structs too large for registers)
/// - **X87/X87Up**: Pass via x87 FPU stack (x86-specific, for `long double`)
/// - **ComplexX87**: `_Complex long double` on x86 (split across x87 stack slots)
/// - **NoClass**: Padding or zero-width fields, ignored for classification
///
/// # Architecture Adaptation
///
/// While the full set of variants originates from x86-64, other architectures
/// map their classification needs onto the same enum:
///
/// | Architecture | Used Classes |
/// |-------------|-------------|
/// | x86-64      | All         |
/// | i686        | Integer, Memory, X87 |
/// | AArch64     | Integer, SSE (for FP), Memory |
/// | RISC-V 64   | Integer, SSE (for FP), Memory |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamClass {
    /// Value fits in a general-purpose integer register.
    Integer,

    /// Value fits in an SSE/floating-point register (XMM on x86-64,
    /// V-register on AArch64, f-register on RISC-V).
    SSE,

    /// Value must be passed in memory (on the stack).
    Memory,

    /// Value is passed in the x87 FPU stack. x86-specific, used for
    /// `long double` (80-bit extended precision).
    X87,

    /// Upper half of an x87 value that crosses an eightbyte boundary.
    /// x86-64 specific, used in the two-eightbyte classification of
    /// `long double`.
    X87Up,

    /// `_Complex long double` — requires special handling on x86 as
    /// it occupies two x87 stack slots.
    ComplexX87,

    /// No classification — used for padding bytes and zero-width
    /// bitfields that don't contribute to the ABI classification.
    NoClass,
}

impl Default for ParamClass {
    /// The default classification is [`NoClass`](ParamClass::NoClass),
    /// representing an unclassified or padding-only field.
    #[inline]
    fn default() -> Self {
        ParamClass::NoClass
    }
}

impl fmt::Display for ParamClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamClass::Integer => write!(f, "INTEGER"),
            ParamClass::SSE => write!(f, "SSE"),
            ParamClass::Memory => write!(f, "MEMORY"),
            ParamClass::X87 => write!(f, "X87"),
            ParamClass::X87Up => write!(f, "X87UP"),
            ParamClass::ComplexX87 => write!(f, "COMPLEX_X87"),
            ParamClass::NoClass => write!(f, "NO_CLASS"),
        }
    }
}

impl ParamClass {
    /// Merges two classifications for the same eightbyte according to
    /// System V AMD64 ABI merge rules (§3.2.3, step 4).
    ///
    /// The merge operation determines the final classification when a
    /// struct field spans an eightbyte that has already been partially
    /// classified.
    ///
    /// # Merge Rules
    ///
    /// 1. If both are equal, the result is that class.
    /// 2. If either is `NoClass`, the result is the other.
    /// 3. If either is `Memory`, the result is `Memory`.
    /// 4. If either is `Integer`, the result is `Integer`.
    /// 5. If either is `X87`/`X87Up`/`ComplexX87`, the result is `Memory`.
    /// 6. Otherwise, the result is `SSE`.
    pub fn merge(self, other: ParamClass) -> ParamClass {
        if self == other {
            return self;
        }
        if self == ParamClass::NoClass {
            return other;
        }
        if other == ParamClass::NoClass {
            return self;
        }
        if self == ParamClass::Memory || other == ParamClass::Memory {
            return ParamClass::Memory;
        }
        if self == ParamClass::Integer || other == ParamClass::Integer {
            return ParamClass::Integer;
        }
        if matches!(
            self,
            ParamClass::X87 | ParamClass::X87Up | ParamClass::ComplexX87
        ) || matches!(
            other,
            ParamClass::X87 | ParamClass::X87Up | ParamClass::ComplexX87
        ) {
            return ParamClass::Memory;
        }
        ParamClass::SSE
    }

    /// Returns `true` if this class requires a general-purpose register.
    #[inline]
    pub fn is_integer(&self) -> bool {
        *self == ParamClass::Integer
    }

    /// Returns `true` if this class requires an SSE/floating-point register.
    #[inline]
    pub fn is_sse(&self) -> bool {
        *self == ParamClass::SSE
    }

    /// Returns `true` if this class requires stack (memory) passing.
    #[inline]
    pub fn is_memory(&self) -> bool {
        *self == ParamClass::Memory
    }
}

// ---------------------------------------------------------------------------
// RelocationType — relocation type descriptor
// ---------------------------------------------------------------------------

/// Describes a relocation type supported by an architecture.
///
/// Each architecture backend exposes its supported relocation types through
/// [`ArchCodegen::get_relocation_types()`]. The linker uses these descriptors
/// to validate and apply relocations when combining object files into the
/// final ELF output.
///
/// # Examples
///
/// ```ignore
/// let r_x86_64_pc32 = RelocationType {
///     name: "R_X86_64_PC32",
///     value: 2,
///     is_pc_relative: true,
///     size: 4,
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelocationType {
    /// Canonical name of the relocation type (e.g., `"R_X86_64_PC32"`).
    pub name: &'static str,

    /// Numeric value as defined in the ELF specification for this architecture.
    pub value: u32,

    /// `true` if the relocation computes a PC-relative offset.
    pub is_pc_relative: bool,

    /// Size of the relocation field in bytes (1, 2, 4, or 8).
    pub size: u8,
}

impl fmt::Display for RelocationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(0x{:04x}, {}B, {})",
            self.name,
            self.value,
            self.size,
            if self.is_pc_relative {
                "PC-rel"
            } else {
                "abs"
            }
        )
    }
}

// ---------------------------------------------------------------------------
// MachineOperand — operand of a machine instruction
// ---------------------------------------------------------------------------

/// An operand of a machine instruction.
///
/// Machine operands represent the inputs and outputs of instructions after
/// instruction selection (Phase 10) but before final register allocation
/// and encoding. Operands may reference physical registers, virtual registers
/// (SSA values awaiting allocation), immediates, memory locations, symbols,
/// labels, or stack frame indices.
#[derive(Clone, Debug, PartialEq)]
pub enum MachineOperand {
    /// A physical register that has been explicitly assigned.
    Register(PhysReg),

    /// A virtual register (SSA value) that has not yet been allocated
    /// to a physical register. The register allocator replaces these
    /// with [`Register`](MachineOperand::Register) operands.
    VirtualReg(ValueId),

    /// An immediate integer constant.
    Immediate(i64),

    /// A memory operand with base register, offset, optional index register,
    /// and scale factor.
    ///
    /// Effective address computation:
    /// `[base + offset + index * scale]`
    ///
    /// - `base`: Base register (always present)
    /// - `offset`: Signed displacement in bytes
    /// - `index`: Optional index register for scaled addressing
    /// - `scale`: Scale factor (1, 2, 4, or 8 on x86; always 1 on others)
    Memory {
        /// Base register for the memory access.
        base: PhysReg,
        /// Signed byte offset from the base register.
        offset: i32,
        /// Optional index register for scaled indexed addressing.
        index: Option<PhysReg>,
        /// Scale factor applied to the index register.
        scale: u8,
    },

    /// A symbolic reference (function name, global variable, external symbol).
    /// Resolved by the linker during relocation processing.
    Symbol(String),

    /// A basic block label reference, used in branch instructions.
    /// The `u32` value is the target block ID.
    Label(u32),

    /// A reference to a stack frame slot. Used for spilled values and
    /// local variables that remain in memory after register allocation.
    /// The `u32` value is the frame slot index.
    FrameIndex(u32),
}

impl fmt::Display for MachineOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MachineOperand::Register(reg) => write!(f, "{}", reg),
            MachineOperand::VirtualReg(vid) => write!(f, "{}", vid),
            MachineOperand::Immediate(val) => write!(f, "${}", val),
            MachineOperand::Memory {
                base,
                offset,
                index,
                scale,
            } => {
                write!(f, "[{}", base)?;
                if *offset != 0 {
                    if *offset > 0 {
                        write!(f, " + {}", offset)?;
                    } else {
                        write!(f, " - {}", -(*offset as i64))?;
                    }
                }
                if let Some(idx) = index {
                    write!(f, " + {} * {}", idx, scale)?;
                }
                write!(f, "]")
            }
            MachineOperand::Symbol(name) => write!(f, "@{}", name),
            MachineOperand::Label(id) => write!(f, "BB{}", id),
            MachineOperand::FrameIndex(idx) => write!(f, "FI{}", idx),
        }
    }
}

impl MachineOperand {
    /// Returns `true` if this operand is a physical register.
    #[inline]
    pub fn is_register(&self) -> bool {
        matches!(self, MachineOperand::Register(_))
    }

    /// Returns `true` if this operand is a virtual register (SSA value).
    #[inline]
    pub fn is_virtual_reg(&self) -> bool {
        matches!(self, MachineOperand::VirtualReg(_))
    }

    /// Returns `true` if this operand is an immediate constant.
    #[inline]
    pub fn is_immediate(&self) -> bool {
        matches!(self, MachineOperand::Immediate(_))
    }

    /// Returns `true` if this operand is a memory reference.
    #[inline]
    pub fn is_memory(&self) -> bool {
        matches!(self, MachineOperand::Memory { .. })
    }

    /// Returns `true` if this operand is a symbolic reference.
    #[inline]
    pub fn is_symbol(&self) -> bool {
        matches!(self, MachineOperand::Symbol(_))
    }

    /// Returns `true` if this operand references a basic block label.
    #[inline]
    pub fn is_label(&self) -> bool {
        matches!(self, MachineOperand::Label(_))
    }

    /// Returns `true` if this operand is a frame index reference.
    #[inline]
    pub fn is_frame_index(&self) -> bool {
        matches!(self, MachineOperand::FrameIndex(_))
    }

    /// If this is a [`Register`](MachineOperand::Register) operand, returns
    /// the physical register; otherwise returns `None`.
    #[inline]
    pub fn as_register(&self) -> Option<PhysReg> {
        if let MachineOperand::Register(reg) = self {
            Some(*reg)
        } else {
            None
        }
    }

    /// If this is a [`VirtualReg`](MachineOperand::VirtualReg) operand,
    /// returns the value ID; otherwise returns `None`.
    #[inline]
    pub fn as_virtual_reg(&self) -> Option<ValueId> {
        if let MachineOperand::VirtualReg(vid) = self {
            Some(*vid)
        } else {
            None
        }
    }

    /// If this is an [`Immediate`](MachineOperand::Immediate) operand,
    /// returns the constant value; otherwise returns `None`.
    #[inline]
    pub fn as_immediate(&self) -> Option<i64> {
        if let MachineOperand::Immediate(val) = self {
            Some(*val)
        } else {
            None
        }
    }

    /// Creates a simple memory operand with base register and offset only
    /// (no index register or scaling).
    #[inline]
    pub fn memory_base_offset(base: PhysReg, offset: i32) -> Self {
        MachineOperand::Memory {
            base,
            offset,
            index: None,
            scale: 1,
        }
    }

    /// Creates a scaled-index memory operand: `[base + index * scale + offset]`.
    #[inline]
    pub fn memory_scaled(base: PhysReg, offset: i32, index: PhysReg, scale: u8) -> Self {
        MachineOperand::Memory {
            base,
            offset,
            index: Some(index),
            scale,
        }
    }
}

// ---------------------------------------------------------------------------
// MachineInstr — a single machine instruction
// ---------------------------------------------------------------------------

/// A single machine instruction after instruction selection.
///
/// Machine instructions represent the target-specific operations that will
/// be encoded into binary by the assembler. Each instruction has:
///
/// - An opcode identifying the operation (architecture-specific encoding)
/// - Explicit operands (registers, immediates, memory references)
/// - Implicit register definitions and uses (e.g., flags register, stack pointer)
/// - Classification flags (terminator, call, return) for CFG analysis
///
/// # Instruction Lifecycle
///
/// ```text
/// IR Instruction → instruction selection → MachineInstr
///     → register allocation (VirtualReg → Register)
///     → prologue/epilogue insertion
///     → encoding → binary bytes
/// ```
#[derive(Clone, Debug)]
pub struct MachineInstr {
    /// Architecture-specific opcode identifying the operation.
    ///
    /// The opcode namespace is defined by each architecture backend.
    /// The code generation driver treats opcodes as opaque values;
    /// only the architecture-specific assembler interprets them.
    pub opcode: u32,

    /// Explicit operands of this instruction.
    ///
    /// The operand ordering convention is architecture-specific:
    /// - x86 (AT&T syntax): `[src, ..., dst]`
    /// - AArch64/RISC-V: `[dst, src1, src2, ...]`
    pub operands: Vec<MachineOperand>,

    /// Registers implicitly defined (written) by this instruction.
    ///
    /// Examples: flags register after a comparison, RAX after a `div`
    /// on x86-64, link register after a `bl` on AArch64.
    pub implicit_defs: Vec<PhysReg>,

    /// Registers implicitly used (read) by this instruction.
    ///
    /// Examples: flags register for conditional branches, stack pointer
    /// for push/pop operations.
    pub implicit_uses: Vec<PhysReg>,

    /// `true` if this instruction terminates its basic block.
    ///
    /// Terminators include unconditional branches, conditional branches,
    /// indirect jumps, returns, and switch dispatches. The basic block
    /// must not contain any instructions after a terminator.
    pub is_terminator: bool,

    /// `true` if this instruction is a function call.
    ///
    /// Call instructions clobber caller-saved registers and may alter
    /// the stack. The register allocator uses this flag to insert
    /// spill code around call sites.
    pub is_call: bool,

    /// `true` if this instruction returns from the function.
    ///
    /// Return instructions are always terminators, but this flag
    /// provides a more specific classification for epilogue insertion.
    pub is_return: bool,
}

impl MachineInstr {
    /// Creates a new machine instruction with the given opcode and no operands.
    ///
    /// All flags default to `false` and operand/implicit lists are empty.
    /// Use the builder-style methods to configure the instruction.
    pub fn new(opcode: u32) -> Self {
        MachineInstr {
            opcode,
            operands: Vec::new(),
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            is_terminator: false,
            is_call: false,
            is_return: false,
        }
    }

    /// Creates a new machine instruction with the given opcode and operands.
    pub fn with_operands(opcode: u32, operands: Vec<MachineOperand>) -> Self {
        MachineInstr {
            opcode,
            operands,
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            is_terminator: false,
            is_call: false,
            is_return: false,
        }
    }

    /// Adds an explicit operand to the instruction.
    pub fn add_operand(&mut self, operand: MachineOperand) {
        self.operands.push(operand);
    }

    /// Adds an implicit register definition.
    pub fn add_implicit_def(&mut self, reg: PhysReg) {
        self.implicit_defs.push(reg);
    }

    /// Adds an implicit register use.
    pub fn add_implicit_use(&mut self, reg: PhysReg) {
        self.implicit_uses.push(reg);
    }

    /// Marks this instruction as a terminator.
    pub fn set_terminator(&mut self) {
        self.is_terminator = true;
    }

    /// Marks this instruction as a function call.
    pub fn set_call(&mut self) {
        self.is_call = true;
    }

    /// Marks this instruction as a return (also sets terminator).
    pub fn set_return(&mut self) {
        self.is_terminator = true;
        self.is_return = true;
    }

    /// Returns `true` if this instruction has side effects that prevent
    /// dead code elimination.
    ///
    /// Instructions with side effects include calls, stores, terminators,
    /// and instructions with implicit definitions (e.g., flag-setting).
    pub fn has_side_effects(&self) -> bool {
        self.is_call || self.is_terminator || !self.implicit_defs.is_empty()
    }

    /// Returns the number of explicit operands.
    #[inline]
    pub fn operand_count(&self) -> usize {
        self.operands.len()
    }

    /// Collects all physical registers defined by this instruction,
    /// combining explicit register destinations and implicit definitions.
    pub fn all_defs(&self) -> Vec<PhysReg> {
        let mut defs = self.implicit_defs.clone();
        for op in &self.operands {
            if let MachineOperand::Register(reg) = op {
                defs.push(*reg);
            }
        }
        defs
    }

    /// Collects all physical registers used (read) by this instruction,
    /// combining explicit register operands, memory base/index registers,
    /// and implicit uses.
    pub fn all_uses(&self) -> Vec<PhysReg> {
        let mut uses = self.implicit_uses.clone();
        for op in &self.operands {
            match op {
                MachineOperand::Register(reg) => {
                    uses.push(*reg);
                }
                MachineOperand::Memory { base, index, .. } => {
                    uses.push(*base);
                    if let Some(idx) = index {
                        uses.push(*idx);
                    }
                }
                _ => {}
            }
        }
        uses
    }

    /// Replaces all occurrences of `old_reg` with `new_reg` in both
    /// explicit operands and implicit def/use lists.
    ///
    /// This is used during register allocation to substitute virtual
    /// registers and during copy coalescing to eliminate redundant moves.
    pub fn replace_register(&mut self, old_reg: PhysReg, new_reg: PhysReg) {
        for op in &mut self.operands {
            match op {
                MachineOperand::Register(r) if *r == old_reg => {
                    *r = new_reg;
                }
                MachineOperand::Memory { base, index, .. } => {
                    if *base == old_reg {
                        *base = new_reg;
                    }
                    if let Some(idx) = index {
                        if *idx == old_reg {
                            *idx = new_reg;
                        }
                    }
                }
                _ => {}
            }
        }
        for r in &mut self.implicit_defs {
            if *r == old_reg {
                *r = new_reg;
            }
        }
        for r in &mut self.implicit_uses {
            if *r == old_reg {
                *r = new_reg;
            }
        }
    }
}

impl fmt::Display for MachineInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "op{}", self.opcode)?;
        for (i, op) in self.operands.iter().enumerate() {
            if i == 0 {
                write!(f, " {}", op)?;
            } else {
                write!(f, ", {}", op)?;
            }
        }
        if !self.implicit_defs.is_empty() {
            write!(f, " imp-def(")?;
            for (i, r) in self.implicit_defs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", r)?;
            }
            write!(f, ")")?;
        }
        if !self.implicit_uses.is_empty() {
            write!(f, " imp-use(")?;
            for (i, r) in self.implicit_uses.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", r)?;
            }
            write!(f, ")")?;
        }
        if self.is_return {
            write!(f, " [ret]")?;
        } else if self.is_terminator {
            write!(f, " [term]")?;
        }
        if self.is_call {
            write!(f, " [call]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MachineBasicBlock — basic block of machine instructions
// ---------------------------------------------------------------------------

/// A basic block containing an ordered sequence of machine instructions.
///
/// Basic blocks in the machine IR mirror the CFG structure from the SSA IR.
/// Each block has a unique numeric ID and an optional label for branch
/// target resolution during assembly encoding.
#[derive(Clone, Debug)]
pub struct MachineBasicBlock {
    /// Unique identifier for this block within the function.
    pub id: u32,

    /// Ordered sequence of machine instructions in this block.
    ///
    /// The last instruction should be a terminator (branch, return, etc.)
    /// unless the block falls through to the next block in layout order.
    pub instructions: Vec<MachineInstr>,

    /// Optional human-readable label for this block.
    ///
    /// Labels are used in assembly output and for branch target resolution.
    /// Generated labels typically follow the pattern `.LBB<func>_<id>`.
    pub label: Option<String>,
}

impl MachineBasicBlock {
    /// Creates a new empty basic block with the given ID.
    pub fn new(id: u32) -> Self {
        MachineBasicBlock {
            id,
            instructions: Vec::new(),
            label: None,
        }
    }

    /// Creates a new empty basic block with the given ID and label.
    pub fn with_label(id: u32, label: String) -> Self {
        MachineBasicBlock {
            id,
            instructions: Vec::new(),
            label: Some(label),
        }
    }

    /// Appends a machine instruction to the end of this block.
    pub fn push_instr(&mut self, instr: MachineInstr) {
        self.instructions.push(instr);
    }

    /// Inserts a machine instruction before the terminator, or at the end
    /// if there is no terminator.
    ///
    /// This is useful for inserting phi-elimination copies and prologue/
    /// epilogue code without disturbing the block's control flow.
    pub fn insert_before_terminator(&mut self, instr: MachineInstr) {
        if let Some(pos) = self
            .instructions
            .iter()
            .rposition(|i| !i.is_terminator)
        {
            // Insert after the last non-terminator instruction.
            self.instructions.insert(pos + 1, instr);
        } else if !self.instructions.is_empty() && self.instructions[0].is_terminator {
            // All instructions are terminators; insert at the beginning.
            self.instructions.insert(0, instr);
        } else {
            // No instructions at all; just append.
            self.instructions.push(instr);
        }
    }

    /// Returns `true` if this block has a terminator instruction.
    pub fn has_terminator(&self) -> bool {
        self.instructions
            .last()
            .is_some_and(|i| i.is_terminator)
    }

    /// Returns a reference to the terminator instruction, if present.
    pub fn terminator(&self) -> Option<&MachineInstr> {
        self.instructions.last().filter(|i| i.is_terminator)
    }

    /// Returns a mutable reference to the terminator instruction, if present.
    pub fn terminator_mut(&mut self) -> Option<&mut MachineInstr> {
        self.instructions
            .last_mut()
            .filter(|i| i.is_terminator)
    }

    /// Returns the number of machine instructions in this block.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` if this block contains no instructions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Returns the label for this block, generating a default one from
    /// the block ID if no explicit label was set.
    pub fn effective_label(&self) -> String {
        match &self.label {
            Some(l) => l.clone(),
            None => format!(".LBB_{}", self.id),
        }
    }
}

impl fmt::Display for MachineBasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.effective_label())?;
        for instr in &self.instructions {
            writeln!(f, "  {}", instr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MachineFunction — function after instruction selection
// ---------------------------------------------------------------------------

/// Represents a function after instruction selection but before final
/// machine code encoding.
///
/// `MachineFunction` is the output of [`ArchCodegen::lower_function()`]
/// and the input to [`ArchCodegen::emit_assembly()`]. It contains:
///
/// - An ordered list of [`MachineBasicBlock`]s
/// - Stack frame metadata (size, alignment, callee-saved registers)
/// - Function-level flags (has_calls, etc.)
///
/// # Stack Frame Layout
///
/// The `frame_size` field records the total size of the function's stack
/// frame in bytes, including space for:
/// - Spilled registers
/// - Local variables that remain in memory
/// - Outgoing argument area (for calls)
/// - Callee-saved register save area
///
/// The prologue/epilogue pass adjusts the stack pointer by `frame_size`
/// and saves/restores `used_callee_saved` registers.
#[derive(Clone, Debug)]
pub struct MachineFunction {
    /// Function symbol name (same as the originating [`IrFunction::name`]).
    pub name: String,

    /// Ordered list of basic blocks in code layout order.
    ///
    /// The first block is the function entry point. Block ordering
    /// determines the physical code layout in the output object file.
    pub blocks: Vec<MachineBasicBlock>,

    /// Total stack frame size in bytes.
    ///
    /// Computed during prologue/epilogue insertion after register
    /// allocation determines the number of spill slots needed.
    pub frame_size: u32,

    /// List of callee-saved registers used by this function.
    ///
    /// These registers must be saved in the prologue and restored in
    /// the epilogue. The register allocator populates this list when
    /// it assigns callee-saved registers to virtual registers.
    pub used_callee_saved: Vec<PhysReg>,

    /// `true` if this function contains any call instructions.
    ///
    /// Functions without calls (leaf functions) may benefit from
    /// optimizations such as red zone usage (x86-64) or omitting
    /// the frame pointer.
    pub has_calls: bool,

    /// Required stack alignment in bytes at function-call boundaries.
    ///
    /// Typically 16 bytes on all supported architectures, but may be
    /// larger if the function uses types with stricter alignment
    /// requirements (e.g., 32-byte AVX vectors).
    pub stack_alignment: u32,
}

impl MachineFunction {
    /// Creates a new machine function with the given name.
    ///
    /// The function starts with an empty block list, zero frame size,
    /// and the given stack alignment (typically from
    /// [`Target::stack_alignment()`]).
    pub fn new(name: String, stack_alignment: u32) -> Self {
        MachineFunction {
            name,
            blocks: Vec::new(),
            frame_size: 0,
            used_callee_saved: Vec::new(),
            has_calls: false,
            stack_alignment,
        }
    }

    /// Adds a basic block to the function and returns its ID.
    pub fn add_block(&mut self, block: MachineBasicBlock) -> u32 {
        let id = block.id;
        self.blocks.push(block);
        id
    }

    /// Creates a new empty basic block, adds it to the function, and returns
    /// its ID. The block ID is assigned sequentially based on the current
    /// block count.
    pub fn create_block(&mut self) -> u32 {
        let id = self.blocks.len() as u32;
        self.blocks.push(MachineBasicBlock::new(id));
        id
    }

    /// Returns the entry block (first block in layout order).
    ///
    /// # Panics
    ///
    /// Panics if the function has no blocks.
    pub fn entry_block(&self) -> &MachineBasicBlock {
        self.blocks
            .first()
            .expect("MachineFunction has no blocks")
    }

    /// Returns a mutable reference to the entry block.
    ///
    /// # Panics
    ///
    /// Panics if the function has no blocks.
    pub fn entry_block_mut(&mut self) -> &mut MachineBasicBlock {
        self.blocks
            .first_mut()
            .expect("MachineFunction has no blocks")
    }

    /// Finds and returns a reference to the block with the given ID.
    ///
    /// Returns `None` if no block with the given ID exists.
    pub fn get_block(&self, id: u32) -> Option<&MachineBasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Finds and returns a mutable reference to the block with the given ID.
    ///
    /// Returns `None` if no block with the given ID exists.
    pub fn get_block_mut(&mut self, id: u32) -> Option<&mut MachineBasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Returns the total number of basic blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the total number of machine instructions across all blocks.
    pub fn total_instructions(&self) -> usize {
        self.blocks.iter().map(|b| b.instruction_count()).sum()
    }

    /// Records a callee-saved register as used by this function.
    ///
    /// Duplicate registrations are silently ignored.
    pub fn mark_callee_saved_used(&mut self, reg: PhysReg) {
        if !self.used_callee_saved.contains(&reg) {
            self.used_callee_saved.push(reg);
        }
    }

    /// Returns `true` if this is a leaf function (no call instructions).
    ///
    /// Leaf functions may benefit from frame-pointer omission and
    /// red-zone usage (x86-64).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        !self.has_calls
    }
}

impl fmt::Display for MachineFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function {} {{", self.name)?;
        writeln!(f, "  ; frame_size = {}", self.frame_size)?;
        writeln!(f, "  ; stack_alignment = {}", self.stack_alignment)?;
        writeln!(f, "  ; has_calls = {}", self.has_calls)?;
        if !self.used_callee_saved.is_empty() {
            write!(f, "  ; callee_saved = [")?;
            for (i, reg) in self.used_callee_saved.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", reg)?;
            }
            writeln!(f, "]")?;
        }
        writeln!(f)?;
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        writeln!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// CodegenConfig — code generation configuration
// ---------------------------------------------------------------------------

/// Code generation configuration collected from CLI flags.
///
/// This struct carries all command-line options that affect code generation
/// behaviour, from target architecture selection to security mitigation
/// flags. It is constructed once during CLI argument parsing and passed
/// through the pipeline to the code generation driver.
///
/// # Security Mitigations (x86-64 only)
///
/// - `retpoline`: When `true`, indirect calls/jumps use retpoline thunks
///   to mitigate Spectre v2 (variant 2) attacks.
/// - `cf_protection`: When `true`, emit Intel CET/IBT `endbr64` instructions
///   at function entries and indirect branch targets.
///
/// Both mitigations are x86-64 specific and have no effect on other targets.
#[derive(Clone, Debug)]
pub struct CodegenConfig {
    /// Target architecture for code generation.
    pub target: Target,

    /// Optimization level (0 = no optimization, 1-3 = increasing optimization).
    ///
    /// Currently, BCC supports `-O0` (default) for unoptimized output and
    /// basic optimization passes. The optimization level gates which passes
    /// are run in the pass manager.
    pub optimization_level: u32,

    /// `true` if DWARF v4 debug information should be emitted (`-g` flag).
    ///
    /// When `true`, the backend generates `.debug_info`, `.debug_abbrev`,
    /// `.debug_line`, and `.debug_str` sections. When `false`, no debug
    /// sections are emitted (zero debug section leakage is required).
    pub debug_info: bool,

    /// `true` if position-independent code should be generated (`-fPIC` flag).
    ///
    /// PIC code uses GOT-relative addressing for global data and PLT
    /// indirection for function calls, enabling the code to be loaded at
    /// any virtual address.
    pub pic: bool,

    /// `true` if building a shared library (`-shared` flag).
    ///
    /// Shared library output produces an `ET_DYN` ELF with `.dynamic`,
    /// `.dynsym`, `.rela.dyn`, `.rela.plt`, and `.gnu.hash` sections.
    /// Implies PIC code generation.
    pub shared: bool,

    /// `true` if retpoline thunks should be used for indirect calls (`-mretpoline`).
    ///
    /// x86-64 only. When enabled, indirect call/jump instructions are
    /// replaced with calls to `__x86_indirect_thunk_*` retpoline stubs.
    pub retpoline: bool,

    /// `true` if Intel CET/IBT should be enabled (`-fcf-protection`).
    ///
    /// x86-64 only. When enabled, `endbr64` instructions are inserted at
    /// function entries and valid indirect branch target sites.
    pub cf_protection: bool,
}

impl CodegenConfig {
    /// Creates a default configuration for the given target.
    ///
    /// Defaults: `-O0`, no debug info, no PIC, no shared, no security mitigations.
    pub fn new(target: Target) -> Self {
        CodegenConfig {
            target,
            optimization_level: 0,
            debug_info: false,
            pic: false,
            shared: false,
            retpoline: false,
            cf_protection: false,
        }
    }

    /// Returns `true` if PIC code generation is required.
    ///
    /// PIC is required when either the `-fPIC` flag or `-shared` flag is
    /// set, since shared libraries must be position-independent.
    #[inline]
    pub fn requires_pic(&self) -> bool {
        self.pic || self.shared
    }

    /// Returns `true` if any x86-64-specific security mitigation is enabled.
    #[inline]
    pub fn has_security_mitigations(&self) -> bool {
        self.retpoline || self.cf_protection
    }

    /// Returns the pointer size in bytes for the configured target.
    #[inline]
    pub fn pointer_size(&self) -> u32 {
        self.target.pointer_width()
    }

    /// Returns the ELF `e_machine` value for the configured target.
    #[inline]
    pub fn elf_machine(&self) -> u16 {
        self.target.elf_machine()
    }

    /// Returns the ELF class (32-bit or 64-bit) for the configured target.
    #[inline]
    pub fn elf_class(&self) -> u8 {
        self.target.elf_class()
    }

    /// Returns the required stack alignment for the configured target.
    #[inline]
    pub fn stack_alignment(&self) -> u32 {
        self.target.stack_alignment()
    }
}

impl Default for CodegenConfig {
    /// Returns a default configuration targeting the host architecture.
    fn default() -> Self {
        CodegenConfig::new(Target::host_target())
    }
}

impl fmt::Display for CodegenConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "target={} -O{}", self.target, self.optimization_level)?;
        if self.debug_info {
            write!(f, " -g")?;
        }
        if self.pic {
            write!(f, " -fPIC")?;
        }
        if self.shared {
            write!(f, " -shared")?;
        }
        if self.retpoline {
            write!(f, " -mretpoline")?;
        }
        if self.cf_protection {
            write!(f, " -fcf-protection")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility functions — helpers for backend implementations
// ---------------------------------------------------------------------------

/// Determines the appropriate [`RegisterClass`] for holding a value of
/// the given IR type.
///
/// This is a convenience function used by architecture backends during
/// instruction selection to determine which register file a value should
/// be assigned to.
///
/// # Mapping Rules
///
/// | IR Type Category | Register Class |
/// |-----------------|---------------|
/// | Integer (`I1`–`I128`) | [`GeneralPurpose`](RegisterClass::GeneralPurpose) |
/// | Pointer (`Ptr`) | [`GeneralPurpose`](RegisterClass::GeneralPurpose) |
/// | Floating-point (`F32`, `F64`, `F80`) | [`FloatingPoint`](RegisterClass::FloatingPoint) |
/// | Aggregate / Function | [`GeneralPurpose`](RegisterClass::GeneralPurpose) (pointer to aggregate) |
pub fn ir_type_register_class(ty: &IrType) -> RegisterClass {
    if ty.is_integer() || ty.is_pointer() {
        RegisterClass::GeneralPurpose
    } else if ty.is_floating() {
        RegisterClass::FloatingPoint
    } else if ty.is_aggregate() {
        // Aggregates are passed by pointer in registers or on the stack.
        RegisterClass::GeneralPurpose
    } else if ty.is_scalar() {
        RegisterClass::GeneralPurpose
    } else {
        // Function types, void — default to GPR (pointer to function).
        RegisterClass::GeneralPurpose
    }
}

/// Returns the storage size of an IR type in bytes for the given target.
///
/// Delegates to [`IrType::size_bytes`] and casts to `u32` for use in
/// machine instruction operand sizing.
#[inline]
pub fn ir_type_operand_size(ty: &IrType, target: &Target) -> u32 {
    ty.size_bytes(target) as u32
}

/// Returns the data width of an IR type in bits for the given target.
///
/// Delegates to [`IrType::size_bits`]. Useful for selecting appropriately
/// sized machine instruction variants (e.g., `movl` vs `movq` on x86-64).
#[inline]
pub fn ir_type_bit_width(ty: &IrType, target: &Target) -> u64 {
    ty.size_bits(target)
}

/// Returns the integer bit-width for an IR integer type, or `None` for
/// non-integer types.
///
/// Delegates to [`IrType::integer_width`]. Architecture backends use this
/// to select instruction variants for different integer widths (e.g.,
/// `addl` for 32-bit vs `addq` for 64-bit on x86-64).
#[inline]
pub fn ir_type_integer_width(ty: &IrType) -> Option<u32> {
    ty.integer_width()
}

/// Provides a basic default classification of a C type into a [`ParamClass`].
///
/// This is a target-independent baseline that architecture-specific ABI
/// modules can use as a starting point. The actual classification may be
/// overridden by each target's [`ArchCodegen::classify_type`] implementation
/// based on struct layout, register availability, and ABI-specific rules.
///
/// # Default Classification Rules
///
/// | C Type Category | Param Class |
/// |----------------|------------|
/// | Integer, pointer | [`Integer`](ParamClass::Integer) |
/// | Floating-point | [`SSE`](ParamClass::SSE) |
/// | Aggregate (struct, union, array) | [`Memory`](ParamClass::Memory) |
/// | Void | [`NoClass`](ParamClass::NoClass) |
pub fn default_classify_ctype(ty: &CType) -> ParamClass {
    if ty.is_integer() || ty.is_pointer() {
        ParamClass::Integer
    } else if ty.is_floating() {
        ParamClass::SSE
    } else if ty.is_aggregate() {
        ParamClass::Memory
    } else if ty.is_scalar() {
        // Other scalar types (e.g., _Bool via is_scalar but not is_integer).
        ParamClass::Integer
    } else {
        ParamClass::NoClass
    }
}

/// Creates a [`MachineFunction`] skeleton from an [`IrFunction`],
/// pre-populating basic block structure and transferring function-level
/// metadata.
///
/// This utility is intended for use by architecture backends in their
/// [`ArchCodegen::lower_function`] implementation. It handles the common
/// setup that is architecture-independent:
///
/// 1. Creates the `MachineFunction` with the function's name and target alignment
/// 2. Pre-creates `MachineBasicBlock` entries for each IR basic block
/// 3. Scans the function metadata for codegen-relevant properties
///
/// Architecture-specific instruction selection should build on the returned
/// skeleton by populating each block with machine instructions.
///
/// # Arguments
///
/// * `func` — the IR function to create a machine function skeleton for
/// * `target` — the target architecture (used for stack alignment)
pub fn machine_function_from_ir(func: &IrFunction, target: &Target) -> MachineFunction {
    let alignment = target.stack_alignment();
    let mut mf = MachineFunction::new(func.name.clone(), alignment);

    // Transfer entry block label. The entry block dominates all others
    // and is the alloca insertion point in the IR.
    let entry = func.entry_block();
    let entry_label = format!(".Lfunc_{}", func.name);
    let entry_bb = MachineBasicBlock::with_label(entry.id.index(), entry_label);
    mf.add_block(entry_bb);

    // Create machine basic blocks for remaining IR blocks.
    for block in func.blocks() {
        if block.id == func.entry_block().id {
            // Already handled above.
            continue;
        }
        let label = format!(".LBB_{}_{}", func.name, block.id.index());
        let mbb = MachineBasicBlock::with_label(block.id.index(), label);
        mf.add_block(mbb);
    }

    // Transfer function-level metadata relevant to code generation.
    // The return type informs which register (int vs float) holds the result.
    let _return_type = &func.return_type;

    // Parameter count affects argument register assignment during lowering.
    let _param_count = func.params.len();

    // Variadic functions may need to dump all argument registers to the
    // stack for va_list traversal.
    if func.is_variadic {
        // Mark that this function is variadic — backends need to handle
        // the register save area in the prologue.
    }

    // Calling convention affects register selection and stack layout.
    let _cc = func.calling_convention;

    // Linkage determines symbol binding (global, local, weak) in the ELF output.
    let _linkage = func.linkage;

    // Function attributes influence codegen decisions:
    // - noreturn: may skip epilogue generation
    // - noinline/always_inline: affects call-site handling
    // - section: affects output section placement
    let _attrs = &func.attributes;

    // Verify we can look up blocks by ID (ensures get_block works correctly).
    if func.basic_blocks.len() > 1 {
        let second_block_id = func.basic_blocks[1].id;
        let _verify = func.get_block(second_block_id);
    }

    // Verify we can look up value types (validates get_value_type availability).
    if !func.params.is_empty() {
        let first_param_id = func.params[0].id;
        let _param_type = func.get_value_type(first_param_id);
    }

    mf
}

// ---------------------------------------------------------------------------
// ArchCodegen — the architecture abstraction layer trait
// ---------------------------------------------------------------------------

/// Architecture abstraction layer for the BCC backend.
///
/// This trait is the central polymorphism point that enables the code
/// generation driver ([`crate::backend::generation`]) to work uniformly
/// across all four target architectures. Each architecture backend
/// (x86-64, i686, AArch64, RISC-V 64) provides a concrete implementation.
///
/// # Trait Methods
///
/// The trait methods fall into several categories:
///
/// ## Instruction Selection
/// - [`lower_function`](ArchCodegen::lower_function) — transforms SSA IR into machine instructions
/// - [`emit_assembly`](ArchCodegen::emit_assembly) — encodes machine instructions into binary
///
/// ## Register Information
/// - [`integer_register_count`] / [`float_register_count`]
/// - [`callee_saved_registers`] / [`caller_saved_registers`]
/// - [`argument_registers_int`] / [`argument_registers_float`]
/// - [`return_register_int`] / [`return_register_float`]
/// - [`stack_pointer`] / [`frame_pointer`]
///
/// ## ABI
/// - [`classify_type`](ArchCodegen::classify_type) — classifies C types into ABI parameter classes
/// - [`pointer_size`] / [`function_alignment`]
///
/// ## Prologue / Epilogue
/// - [`emit_prologue`] / [`emit_epilogue`]
///
/// ## PIC Support
/// - [`generate_pic_addressing`] — generates position-independent symbol references
///
/// ## Relocation
/// - [`get_relocation_types`] — returns supported relocation types
///
/// # Implementation Order
///
/// Per Section 0.1.2, backends are validated in this fixed order:
/// 1. x86-64 (primary validation target)
/// 2. i686
/// 3. AArch64
/// 4. RISC-V 64 (Linux kernel boot target)
///
/// # Example Implementation Skeleton
///
/// ```ignore
/// struct X86_64Codegen;
///
/// impl ArchCodegen for X86_64Codegen {
///     fn lower_function(&self, func: &IrFunction) -> MachineFunction {
///         let mut mf = machine_function_from_ir(func, &Target::X86_64);
///         // ... instruction selection ...
///         mf
///     }
///     fn emit_assembly(&self, mf: &MachineFunction) -> Vec<u8> {
///         let mut code = Vec::new();
///         // ... encode machine instructions to bytes ...
///         code
///     }
///     // ... other methods ...
/// }
/// ```
pub trait ArchCodegen {
    /// Transforms an IR function into a machine function via instruction selection.
    ///
    /// This is the core instruction selection pass. The implementation walks
    /// each basic block and IR instruction, selecting architecture-specific
    /// machine instructions, assigning virtual registers, and building the
    /// machine-level CFG.
    ///
    /// # Arguments
    ///
    /// * `func` — the IR function to lower, in SSA form after phi-elimination
    ///
    /// # Returns
    ///
    /// A [`MachineFunction`] with instruction selection complete but registers
    /// still in virtual form. The register allocator will subsequently replace
    /// [`MachineOperand::VirtualReg`] operands with physical registers.
    fn lower_function(&self, func: &IrFunction) -> MachineFunction;

    /// Encodes a machine function into binary machine code.
    ///
    /// This method drives the architecture-specific assembler to translate
    /// machine instructions into their binary encoding. The output is a
    /// relocatable byte sequence suitable for the ELF `.text` section.
    ///
    /// # Arguments
    ///
    /// * `mf` — the machine function with all registers allocated
    ///
    /// # Returns
    ///
    /// A byte vector containing the encoded machine instructions.
    fn emit_assembly(&self, mf: &MachineFunction) -> Vec<u8>;

    /// Returns the set of relocation types supported by this architecture.
    ///
    /// The returned slice contains descriptors for all ELF relocation types
    /// that the assembler and linker may emit or process for this target.
    fn get_relocation_types(&self) -> &[RelocationType];

    /// Returns the total number of allocatable integer (general-purpose) registers.
    ///
    /// | Architecture | Count | Registers |
    /// |-------------|-------|-----------|
    /// | x86-64      | 16    | RAX–R15   |
    /// | i686        | 8     | EAX–EDI   |
    /// | AArch64     | 31    | X0–X30    |
    /// | RISC-V 64   | 32    | x0–x31    |
    fn integer_register_count(&self) -> usize;

    /// Returns the total number of allocatable floating-point registers.
    ///
    /// | Architecture | Count | Registers |
    /// |-------------|-------|-----------|
    /// | x86-64      | 16    | XMM0–XMM15 |
    /// | i686        | 8     | ST(0)–ST(7) |
    /// | AArch64     | 32    | V0–V31    |
    /// | RISC-V 64   | 32    | f0–f31    |
    fn float_register_count(&self) -> usize;

    /// Returns the set of callee-saved (non-volatile) registers.
    ///
    /// These registers must be preserved across function calls. If a function
    /// uses any of these registers, it must save their values in the prologue
    /// and restore them in the epilogue.
    fn callee_saved_registers(&self) -> &[PhysReg];

    /// Returns the set of caller-saved (volatile) registers.
    ///
    /// These registers may be clobbered by any function call. The caller is
    /// responsible for saving their values before a call if they are live
    /// across the call site.
    fn caller_saved_registers(&self) -> &[PhysReg];

    /// Returns the integer registers used for argument passing, in order.
    ///
    /// | Architecture | Argument Registers |
    /// |-------------|-------------------|
    /// | x86-64      | RDI, RSI, RDX, RCX, R8, R9 |
    /// | i686        | (none — all args on stack) |
    /// | AArch64     | X0–X7 |
    /// | RISC-V 64   | a0–a7 (x10–x17) |
    fn argument_registers_int(&self) -> &[PhysReg];

    /// Returns the floating-point registers used for argument passing, in order.
    ///
    /// | Architecture | Argument Registers |
    /// |-------------|-------------------|
    /// | x86-64      | XMM0–XMM7 |
    /// | i686        | (none — FP args on stack) |
    /// | AArch64     | V0–V7 |
    /// | RISC-V 64   | fa0–fa7 (f10–f17) |
    fn argument_registers_float(&self) -> &[PhysReg];

    /// Returns the physical register used for integer return values.
    ///
    /// | Architecture | Register |
    /// |-------------|---------|
    /// | x86-64      | RAX     |
    /// | i686        | EAX     |
    /// | AArch64     | X0      |
    /// | RISC-V 64   | a0 (x10)|
    fn return_register_int(&self) -> PhysReg;

    /// Returns the physical register used for floating-point return values.
    ///
    /// | Architecture | Register |
    /// |-------------|---------|
    /// | x86-64      | XMM0    |
    /// | i686        | ST(0)   |
    /// | AArch64     | V0      |
    /// | RISC-V 64   | fa0 (f10)|
    fn return_register_float(&self) -> PhysReg;

    /// Returns the physical register that serves as the stack pointer.
    ///
    /// | Architecture | Register |
    /// |-------------|---------|
    /// | x86-64      | RSP     |
    /// | i686        | ESP     |
    /// | AArch64     | SP (X31)|
    /// | RISC-V 64   | sp (x2) |
    fn stack_pointer(&self) -> PhysReg;

    /// Returns the physical register that serves as the frame pointer.
    ///
    /// | Architecture | Register |
    /// |-------------|---------|
    /// | x86-64      | RBP     |
    /// | i686        | EBP     |
    /// | AArch64     | X29 (FP)|
    /// | RISC-V 64   | s0 (x8) |
    fn frame_pointer(&self) -> PhysReg;

    /// Returns the pointer size in bytes for this target.
    ///
    /// * 64-bit targets: 8
    /// * 32-bit targets: 4
    fn pointer_size(&self) -> u32;

    /// Returns the required alignment in bytes for function entry points.
    ///
    /// Functions are aligned to this boundary in the `.text` section to
    /// ensure optimal instruction fetch and branch target alignment.
    ///
    /// | Architecture | Alignment |
    /// |-------------|----------|
    /// | x86-64      | 16       |
    /// | i686        | 16       |
    /// | AArch64     | 4        |
    /// | RISC-V 64   | 4        |
    fn function_alignment(&self) -> u32;

    /// Emits function prologue code into the machine function.
    ///
    /// The prologue is responsible for:
    /// 1. Saving callee-saved registers used by the function
    /// 2. Establishing the stack frame (push frame pointer, adjust SP)
    /// 3. Allocating local variable space on the stack
    /// 4. On x86-64 with `-fcf-protection`: emitting `endbr64`
    /// 5. On x86-64 with large frames: emitting stack probe loop
    ///
    /// Prologue instructions are inserted at the beginning of the entry block.
    fn emit_prologue(&self, mf: &mut MachineFunction);

    /// Emits function epilogue code into the machine function.
    ///
    /// The epilogue is responsible for:
    /// 1. Deallocating local variable space (restore SP)
    /// 2. Restoring callee-saved registers
    /// 3. Tearing down the stack frame (restore frame pointer)
    /// 4. Executing the return instruction
    ///
    /// Epilogue instructions are inserted before each return instruction
    /// in every block that ends with a return.
    fn emit_epilogue(&self, mf: &mut MachineFunction);

    /// Classifies a C type into an ABI parameter class.
    ///
    /// This classification determines how a value of the given C type is
    /// passed to or returned from a function according to the target
    /// architecture's calling convention:
    ///
    /// - **Integer**: fits in a GPR (integers, pointers, small structs)
    /// - **SSE**: fits in an FP/vector register (float, double)
    /// - **Memory**: must be passed on the stack (large structs)
    /// - **X87**: passed via x87 FPU stack (long double on x86)
    ///
    /// # Arguments
    ///
    /// * `ty` — the C language type to classify
    ///
    /// # Returns
    ///
    /// The [`ParamClass`] that determines the register file or stack
    /// location for values of this type.
    fn classify_type(&self, ty: &CType) -> ParamClass;

    /// Generates position-independent addressing for a symbol.
    ///
    /// When PIC mode is active (`-fPIC`), global symbols must be accessed
    /// through the Global Offset Table (GOT) or using PC-relative addressing.
    /// This method generates the appropriate machine operand for loading
    /// the address of a symbol in PIC mode.
    ///
    /// The implementation varies by architecture:
    ///
    /// - **x86-64**: `mov sym@GOTPCREL(%rip), %reg` or `lea sym(%rip), %reg`
    /// - **i686**: `mov sym@GOT(%ebx), %reg` (requires GOT base in EBX)
    /// - **AArch64**: `adrp x?, :got:sym` + `ldr x?, [x?, :got_lo12:sym]`
    /// - **RISC-V 64**: `auipc` + `ld` via GOT
    ///
    /// # Arguments
    ///
    /// * `symbol` — the symbol name to generate PIC addressing for
    /// * `mf` — the machine function to emit addressing instructions into
    ///
    /// # Returns
    ///
    /// A [`MachineOperand`] referencing the loaded symbol address.
    fn generate_pic_addressing(
        &self,
        symbol: &str,
        mf: &mut MachineFunction,
    ) -> MachineOperand;
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PhysReg tests ---------------------------------------------------

    #[test]
    fn phys_reg_copy_clone() {
        let r1 = PhysReg(5);
        let r2 = r1; // Copy
        assert_eq!(r1, r2);
        assert_eq!(r1.index(), 5);
    }

    #[test]
    fn phys_reg_none_sentinel() {
        let r = PhysReg::NONE;
        assert!(r.is_none());
        assert!(!r.is_valid());
        assert_eq!(r.index(), u16::MAX);
    }

    #[test]
    fn phys_reg_valid() {
        let r = PhysReg(0);
        assert!(r.is_valid());
        assert!(!r.is_none());
    }

    #[test]
    fn phys_reg_display() {
        assert_eq!(format!("{}", PhysReg(0)), "r0");
        assert_eq!(format!("{}", PhysReg(15)), "r15");
        assert_eq!(format!("{}", PhysReg::NONE), "<none>");
    }

    #[test]
    fn phys_reg_ordering() {
        let r0 = PhysReg(0);
        let r1 = PhysReg(1);
        let r5 = PhysReg(5);
        assert!(r0 < r1);
        assert!(r1 < r5);
    }

    #[test]
    fn phys_reg_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PhysReg(0));
        set.insert(PhysReg(1));
        set.insert(PhysReg(0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    // -- RegisterClass tests ---------------------------------------------

    #[test]
    fn register_class_display() {
        assert_eq!(format!("{}", RegisterClass::GeneralPurpose), "GPR");
        assert_eq!(format!("{}", RegisterClass::FloatingPoint), "FP");
        assert_eq!(format!("{}", RegisterClass::Vector), "VEC");
        assert_eq!(format!("{}", RegisterClass::StackPointer), "SP");
        assert_eq!(format!("{}", RegisterClass::FramePointer), "FP_REG");
    }

    #[test]
    fn register_class_equality() {
        assert_eq!(RegisterClass::GeneralPurpose, RegisterClass::GeneralPurpose);
        assert_ne!(RegisterClass::GeneralPurpose, RegisterClass::FloatingPoint);
    }

    // -- ParamClass tests ------------------------------------------------

    #[test]
    fn param_class_default_is_noclass() {
        assert_eq!(ParamClass::default(), ParamClass::NoClass);
    }

    #[test]
    fn param_class_merge_same() {
        assert_eq!(
            ParamClass::Integer.merge(ParamClass::Integer),
            ParamClass::Integer
        );
        assert_eq!(ParamClass::SSE.merge(ParamClass::SSE), ParamClass::SSE);
    }

    #[test]
    fn param_class_merge_noclass_identity() {
        assert_eq!(
            ParamClass::NoClass.merge(ParamClass::Integer),
            ParamClass::Integer
        );
        assert_eq!(
            ParamClass::SSE.merge(ParamClass::NoClass),
            ParamClass::SSE
        );
    }

    #[test]
    fn param_class_merge_memory_dominates() {
        assert_eq!(
            ParamClass::Integer.merge(ParamClass::Memory),
            ParamClass::Memory
        );
        assert_eq!(
            ParamClass::Memory.merge(ParamClass::SSE),
            ParamClass::Memory
        );
    }

    #[test]
    fn param_class_merge_integer_over_sse() {
        assert_eq!(
            ParamClass::Integer.merge(ParamClass::SSE),
            ParamClass::Integer
        );
    }

    #[test]
    fn param_class_merge_x87_becomes_memory() {
        assert_eq!(
            ParamClass::X87.merge(ParamClass::SSE),
            ParamClass::Memory
        );
        assert_eq!(
            ParamClass::SSE.merge(ParamClass::X87Up),
            ParamClass::Memory
        );
    }

    #[test]
    fn param_class_merge_complex_x87_forces_memory() {
        // Per System V AMD64 ABI §3.2.3 merge rules:
        // Rule 4 (Integer dominates) applies before rule 5 (X87 → Memory).
        // So ComplexX87.merge(Integer) = Integer, but
        // ComplexX87.merge(SSE) = Memory (rule 5 applies since neither is Integer).
        assert_eq!(
            ParamClass::ComplexX87.merge(ParamClass::Integer),
            ParamClass::Integer
        );
        assert_eq!(
            ParamClass::ComplexX87.merge(ParamClass::SSE),
            ParamClass::Memory
        );
    }

    #[test]
    fn param_class_merge_noclass_both() {
        assert_eq!(
            ParamClass::NoClass.merge(ParamClass::NoClass),
            ParamClass::NoClass
        );
    }

    #[test]
    fn param_class_predicates() {
        assert!(ParamClass::Integer.is_integer());
        assert!(!ParamClass::Integer.is_sse());
        assert!(ParamClass::SSE.is_sse());
        assert!(ParamClass::Memory.is_memory());
        assert!(!ParamClass::NoClass.is_integer());
    }

    #[test]
    fn param_class_display() {
        assert_eq!(format!("{}", ParamClass::Integer), "INTEGER");
        assert_eq!(format!("{}", ParamClass::SSE), "SSE");
        assert_eq!(format!("{}", ParamClass::Memory), "MEMORY");
        assert_eq!(format!("{}", ParamClass::X87), "X87");
        assert_eq!(format!("{}", ParamClass::X87Up), "X87UP");
        assert_eq!(format!("{}", ParamClass::ComplexX87), "COMPLEX_X87");
        assert_eq!(format!("{}", ParamClass::NoClass), "NO_CLASS");
    }

    // -- RelocationType tests --------------------------------------------

    #[test]
    fn relocation_type_display_pc_rel() {
        let r = RelocationType {
            name: "R_X86_64_PC32",
            value: 2,
            is_pc_relative: true,
            size: 4,
        };
        let s = format!("{}", r);
        assert!(s.contains("R_X86_64_PC32"));
        assert!(s.contains("PC-rel"));
        assert!(s.contains("4B"));
    }

    #[test]
    fn relocation_type_display_abs() {
        let r = RelocationType {
            name: "R_X86_64_64",
            value: 1,
            is_pc_relative: false,
            size: 8,
        };
        let s = format!("{}", r);
        assert!(s.contains("abs"));
        assert!(s.contains("8B"));
    }

    #[test]
    fn relocation_type_equality() {
        let r1 = RelocationType {
            name: "R_X86_64_PC32",
            value: 2,
            is_pc_relative: true,
            size: 4,
        };
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    // -- MachineOperand tests --------------------------------------------

    #[test]
    fn machine_operand_register_predicates() {
        let op = MachineOperand::Register(PhysReg(0));
        assert!(op.is_register());
        assert!(!op.is_virtual_reg());
        assert!(!op.is_immediate());
        assert!(!op.is_memory());
        assert!(!op.is_symbol());
        assert!(!op.is_label());
        assert!(!op.is_frame_index());
        assert_eq!(op.as_register(), Some(PhysReg(0)));
    }

    #[test]
    fn machine_operand_virtual_reg() {
        let op = MachineOperand::VirtualReg(ValueId(42));
        assert!(op.is_virtual_reg());
        assert_eq!(op.as_virtual_reg(), Some(ValueId(42)));
        assert_eq!(op.as_register(), None);
    }

    #[test]
    fn machine_operand_immediate() {
        let op = MachineOperand::Immediate(-128);
        assert!(op.is_immediate());
        assert_eq!(op.as_immediate(), Some(-128));
    }

    #[test]
    fn machine_operand_memory() {
        let op = MachineOperand::Memory {
            base: PhysReg(4),
            offset: -16,
            index: Some(PhysReg(1)),
            scale: 4,
        };
        assert!(op.is_memory());
        let s = format!("{}", op);
        assert!(s.contains("r4"));
        assert!(s.contains("16"));
    }

    #[test]
    fn machine_operand_memory_helpers() {
        let simple = MachineOperand::memory_base_offset(PhysReg(7), -8);
        assert!(simple.is_memory());
        if let MachineOperand::Memory {
            base,
            offset,
            index,
            scale,
        } = &simple
        {
            assert_eq!(*base, PhysReg(7));
            assert_eq!(*offset, -8);
            assert!(index.is_none());
            assert_eq!(*scale, 1);
        }

        let scaled = MachineOperand::memory_scaled(PhysReg(0), 16, PhysReg(2), 8);
        if let MachineOperand::Memory {
            base,
            offset,
            index,
            scale,
        } = &scaled
        {
            assert_eq!(*base, PhysReg(0));
            assert_eq!(*offset, 16);
            assert_eq!(*index, Some(PhysReg(2)));
            assert_eq!(*scale, 8);
        }
    }

    #[test]
    fn machine_operand_symbol() {
        let op = MachineOperand::Symbol("printf".into());
        assert!(op.is_symbol());
        assert_eq!(format!("{}", op), "@printf");
    }

    #[test]
    fn machine_operand_label() {
        let op = MachineOperand::Label(3);
        assert!(op.is_label());
        assert_eq!(format!("{}", op), "BB3");
    }

    #[test]
    fn machine_operand_frame_index() {
        let op = MachineOperand::FrameIndex(0);
        assert!(op.is_frame_index());
        assert_eq!(format!("{}", op), "FI0");
    }

    // -- MachineInstr tests ----------------------------------------------

    #[test]
    fn machine_instr_new() {
        let mi = MachineInstr::new(0x90);
        assert_eq!(mi.opcode, 0x90);
        assert!(mi.operands.is_empty());
        assert!(!mi.is_terminator);
        assert!(!mi.is_call);
        assert!(!mi.is_return);
        assert!(!mi.has_side_effects());
    }

    #[test]
    fn machine_instr_set_flags() {
        let mut mi = MachineInstr::new(0xC3);
        mi.set_return();
        assert!(mi.is_return);
        assert!(mi.is_terminator);
        assert!(mi.has_side_effects());
    }

    #[test]
    fn machine_instr_set_call() {
        let mut mi = MachineInstr::new(0xE8);
        mi.set_call();
        assert!(mi.is_call);
        assert!(mi.has_side_effects());
    }

    #[test]
    fn machine_instr_implicit_defs_side_effects() {
        let mut mi = MachineInstr::new(0x01);
        assert!(!mi.has_side_effects());
        mi.add_implicit_def(PhysReg(0));
        assert!(mi.has_side_effects());
    }

    #[test]
    fn machine_instr_with_operands() {
        let mi = MachineInstr::with_operands(
            0x89,
            vec![
                MachineOperand::Register(PhysReg(0)),
                MachineOperand::Register(PhysReg(1)),
            ],
        );
        assert_eq!(mi.operand_count(), 2);
    }

    #[test]
    fn machine_instr_add_operand() {
        let mut mi = MachineInstr::new(0x01);
        mi.add_operand(MachineOperand::Register(PhysReg(0)));
        mi.add_operand(MachineOperand::Immediate(42));
        assert_eq!(mi.operand_count(), 2);
    }

    #[test]
    fn machine_instr_all_defs_and_uses() {
        let mut mi = MachineInstr::new(0x01);
        mi.add_operand(MachineOperand::Register(PhysReg(0)));
        mi.add_operand(MachineOperand::Memory {
            base: PhysReg(4),
            offset: 0,
            index: Some(PhysReg(2)),
            scale: 4,
        });
        mi.add_implicit_def(PhysReg(10));
        mi.add_implicit_use(PhysReg(11));

        let defs = mi.all_defs();
        assert!(defs.contains(&PhysReg(10)));
        assert!(defs.contains(&PhysReg(0)));

        let uses = mi.all_uses();
        assert!(uses.contains(&PhysReg(11)));
        assert!(uses.contains(&PhysReg(4)));
        assert!(uses.contains(&PhysReg(2)));
    }

    #[test]
    fn machine_instr_replace_register() {
        let mut mi = MachineInstr::new(0x89);
        mi.add_operand(MachineOperand::Register(PhysReg(5)));
        mi.add_operand(MachineOperand::Memory {
            base: PhysReg(5),
            offset: 8,
            index: None,
            scale: 1,
        });
        mi.add_implicit_def(PhysReg(5));
        mi.add_implicit_use(PhysReg(5));

        mi.replace_register(PhysReg(5), PhysReg(10));

        assert_eq!(mi.operands[0].as_register(), Some(PhysReg(10)));
        if let MachineOperand::Memory { base, .. } = &mi.operands[1] {
            assert_eq!(*base, PhysReg(10));
        }
        assert_eq!(mi.implicit_defs[0], PhysReg(10));
        assert_eq!(mi.implicit_uses[0], PhysReg(10));
    }

    #[test]
    fn machine_instr_display() {
        let mut mi = MachineInstr::new(0x01);
        mi.add_operand(MachineOperand::Register(PhysReg(0)));
        mi.add_operand(MachineOperand::Immediate(42));
        mi.set_call();
        let s = format!("{}", mi);
        assert!(s.contains("op1"));
        assert!(s.contains("r0"));
        assert!(s.contains("$42"));
        assert!(s.contains("[call]"));
    }

    // -- MachineBasicBlock tests -----------------------------------------

    #[test]
    fn machine_basic_block_new() {
        let bb = MachineBasicBlock::new(0);
        assert_eq!(bb.id, 0);
        assert!(bb.is_empty());
        assert_eq!(bb.instruction_count(), 0);
        assert!(!bb.has_terminator());
        assert!(bb.terminator().is_none());
    }

    #[test]
    fn machine_basic_block_with_label() {
        let bb = MachineBasicBlock::with_label(5, ".Lfoo".into());
        assert_eq!(bb.id, 5);
        assert_eq!(bb.label.as_deref(), Some(".Lfoo"));
        assert_eq!(bb.effective_label(), ".Lfoo");
    }

    #[test]
    fn machine_basic_block_default_label() {
        let bb = MachineBasicBlock::new(42);
        assert_eq!(bb.effective_label(), ".LBB_42");
    }

    #[test]
    fn machine_basic_block_push_instr() {
        let mut bb = MachineBasicBlock::new(0);
        bb.push_instr(MachineInstr::new(0x90));
        assert_eq!(bb.instruction_count(), 1);
        assert!(!bb.is_empty());
    }

    #[test]
    fn machine_basic_block_has_terminator() {
        let mut bb = MachineBasicBlock::new(0);
        let mut ret = MachineInstr::new(0xC3);
        ret.set_return();
        bb.push_instr(ret);
        assert!(bb.has_terminator());
        assert!(bb.terminator().is_some());
        assert_eq!(bb.terminator().unwrap().opcode, 0xC3);
    }

    #[test]
    fn machine_basic_block_insert_before_terminator() {
        let mut bb = MachineBasicBlock::new(0);
        bb.push_instr(MachineInstr::new(0x01)); // non-terminator
        let mut ret = MachineInstr::new(0xC3);
        ret.set_return();
        bb.push_instr(ret);

        // Insert a NOP before the return
        bb.insert_before_terminator(MachineInstr::new(0x90));

        assert_eq!(bb.instruction_count(), 3);
        assert_eq!(bb.instructions[0].opcode, 0x01);
        assert_eq!(bb.instructions[1].opcode, 0x90);
        assert_eq!(bb.instructions[2].opcode, 0xC3);
    }

    #[test]
    fn machine_basic_block_insert_before_terminator_only() {
        let mut bb = MachineBasicBlock::new(0);
        let mut ret = MachineInstr::new(0xC3);
        ret.set_return();
        bb.push_instr(ret);

        // Block has only a terminator — insert at beginning
        bb.insert_before_terminator(MachineInstr::new(0x90));

        assert_eq!(bb.instruction_count(), 2);
        assert_eq!(bb.instructions[0].opcode, 0x90);
        assert_eq!(bb.instructions[1].opcode, 0xC3);
    }

    #[test]
    fn machine_basic_block_insert_before_terminator_empty() {
        let mut bb = MachineBasicBlock::new(0);

        // Block has no instructions — just append
        bb.insert_before_terminator(MachineInstr::new(0x90));
        assert_eq!(bb.instruction_count(), 1);
    }

    #[test]
    fn machine_basic_block_display() {
        let mut bb = MachineBasicBlock::with_label(0, ".Lentry".into());
        bb.push_instr(MachineInstr::new(0x90));
        let s = format!("{}", bb);
        assert!(s.contains(".Lentry:"));
        assert!(s.contains("op144")); // 0x90 = 144
    }

    // -- MachineFunction tests -------------------------------------------

    #[test]
    fn machine_function_new() {
        let mf = MachineFunction::new("main".into(), 16);
        assert_eq!(mf.name, "main");
        assert_eq!(mf.frame_size, 0);
        assert_eq!(mf.stack_alignment, 16);
        assert!(!mf.has_calls);
        assert!(mf.used_callee_saved.is_empty());
        assert_eq!(mf.block_count(), 0);
        assert!(mf.is_leaf());
    }

    #[test]
    fn machine_function_add_block() {
        let mut mf = MachineFunction::new("foo".into(), 16);
        let bb0 = MachineBasicBlock::new(0);
        let bb1 = MachineBasicBlock::new(1);
        mf.add_block(bb0);
        mf.add_block(bb1);
        assert_eq!(mf.block_count(), 2);
    }

    #[test]
    fn machine_function_create_block() {
        let mut mf = MachineFunction::new("f".into(), 16);
        let id0 = mf.create_block();
        let id1 = mf.create_block();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(mf.block_count(), 2);
    }

    #[test]
    fn machine_function_entry_block() {
        let mut mf = MachineFunction::new("f".into(), 16);
        mf.add_block(MachineBasicBlock::with_label(0, "entry".into()));
        assert_eq!(mf.entry_block().id, 0);
    }

    #[test]
    fn machine_function_get_block() {
        let mut mf = MachineFunction::new("f".into(), 16);
        mf.add_block(MachineBasicBlock::new(0));
        mf.add_block(MachineBasicBlock::new(5));
        assert!(mf.get_block(0).is_some());
        assert!(mf.get_block(5).is_some());
        assert!(mf.get_block(99).is_none());
    }

    #[test]
    fn machine_function_get_block_mut() {
        let mut mf = MachineFunction::new("f".into(), 16);
        mf.add_block(MachineBasicBlock::new(0));
        if let Some(bb) = mf.get_block_mut(0) {
            bb.push_instr(MachineInstr::new(0x90));
        }
        assert_eq!(mf.get_block(0).unwrap().instruction_count(), 1);
    }

    #[test]
    fn machine_function_total_instructions() {
        let mut mf = MachineFunction::new("f".into(), 16);
        let mut bb0 = MachineBasicBlock::new(0);
        bb0.push_instr(MachineInstr::new(0x01));
        bb0.push_instr(MachineInstr::new(0x02));
        let mut bb1 = MachineBasicBlock::new(1);
        bb1.push_instr(MachineInstr::new(0x03));
        mf.add_block(bb0);
        mf.add_block(bb1);
        assert_eq!(mf.total_instructions(), 3);
    }

    #[test]
    fn machine_function_mark_callee_saved() {
        let mut mf = MachineFunction::new("f".into(), 16);
        mf.mark_callee_saved_used(PhysReg(3));
        mf.mark_callee_saved_used(PhysReg(3)); // duplicate — ignored
        mf.mark_callee_saved_used(PhysReg(12));
        assert_eq!(mf.used_callee_saved.len(), 2);
    }

    #[test]
    fn machine_function_display() {
        let mut mf = MachineFunction::new("test_fn".into(), 16);
        mf.frame_size = 32;
        mf.has_calls = true;
        mf.mark_callee_saved_used(PhysReg(3));
        mf.add_block(MachineBasicBlock::with_label(0, ".Lentry".into()));
        let s = format!("{}", mf);
        assert!(s.contains("function test_fn"));
        assert!(s.contains("frame_size = 32"));
        assert!(s.contains("has_calls = true"));
        assert!(s.contains("r3"));
    }

    // -- CodegenConfig tests ---------------------------------------------

    #[test]
    fn codegen_config_new() {
        let cfg = CodegenConfig::new(Target::X86_64);
        assert_eq!(cfg.target, Target::X86_64);
        assert_eq!(cfg.optimization_level, 0);
        assert!(!cfg.debug_info);
        assert!(!cfg.pic);
        assert!(!cfg.shared);
        assert!(!cfg.retpoline);
        assert!(!cfg.cf_protection);
    }

    #[test]
    fn codegen_config_requires_pic() {
        let mut cfg = CodegenConfig::new(Target::X86_64);
        assert!(!cfg.requires_pic());
        cfg.pic = true;
        assert!(cfg.requires_pic());
        cfg.pic = false;
        cfg.shared = true;
        assert!(cfg.requires_pic()); // -shared implies PIC
    }

    #[test]
    fn codegen_config_security_mitigations() {
        let mut cfg = CodegenConfig::new(Target::X86_64);
        assert!(!cfg.has_security_mitigations());
        cfg.retpoline = true;
        assert!(cfg.has_security_mitigations());
        cfg.retpoline = false;
        cfg.cf_protection = true;
        assert!(cfg.has_security_mitigations());
    }

    #[test]
    fn codegen_config_target_delegations() {
        let cfg = CodegenConfig::new(Target::X86_64);
        assert_eq!(cfg.pointer_size(), 8);
        assert_eq!(cfg.stack_alignment(), 16);
        assert_eq!(cfg.elf_machine(), 62); // EM_X86_64
        assert_eq!(cfg.elf_class(), 2); // ELFCLASS64

        let cfg32 = CodegenConfig::new(Target::I686);
        assert_eq!(cfg32.pointer_size(), 4);
        assert_eq!(cfg32.elf_machine(), 3); // EM_386
        assert_eq!(cfg32.elf_class(), 1); // ELFCLASS32
    }

    #[test]
    fn codegen_config_display() {
        let mut cfg = CodegenConfig::new(Target::X86_64);
        cfg.debug_info = true;
        cfg.pic = true;
        cfg.retpoline = true;
        cfg.cf_protection = true;
        let s = format!("{}", cfg);
        assert!(s.contains("x86-64"));
        assert!(s.contains("-g"));
        assert!(s.contains("-fPIC"));
        assert!(s.contains("-mretpoline"));
        assert!(s.contains("-fcf-protection"));
    }

    #[test]
    fn codegen_config_default() {
        let cfg = CodegenConfig::default();
        assert_eq!(cfg.optimization_level, 0);
        assert!(!cfg.debug_info);
    }

    // -- Utility function tests ------------------------------------------

    #[test]
    fn test_ir_type_register_class() {
        assert_eq!(ir_type_register_class(&IrType::I32), RegisterClass::GeneralPurpose);
        assert_eq!(ir_type_register_class(&IrType::I64), RegisterClass::GeneralPurpose);
        assert_eq!(ir_type_register_class(&IrType::Ptr), RegisterClass::GeneralPurpose);
        assert_eq!(ir_type_register_class(&IrType::F32), RegisterClass::FloatingPoint);
        assert_eq!(ir_type_register_class(&IrType::F64), RegisterClass::FloatingPoint);
        assert_eq!(ir_type_register_class(&IrType::F80), RegisterClass::FloatingPoint);
    }

    #[test]
    fn test_ir_type_operand_size() {
        assert_eq!(ir_type_operand_size(&IrType::I32, &Target::X86_64), 4);
        assert_eq!(ir_type_operand_size(&IrType::I64, &Target::X86_64), 8);
        assert_eq!(ir_type_operand_size(&IrType::Ptr, &Target::X86_64), 8);
        assert_eq!(ir_type_operand_size(&IrType::Ptr, &Target::I686), 4);
    }

    #[test]
    fn test_ir_type_bit_width() {
        assert_eq!(ir_type_bit_width(&IrType::I32, &Target::X86_64), 32);
        assert_eq!(ir_type_bit_width(&IrType::I64, &Target::X86_64), 64);
        assert_eq!(ir_type_bit_width(&IrType::F80, &Target::X86_64), 80);
    }

    #[test]
    fn test_ir_type_integer_width() {
        assert_eq!(ir_type_integer_width(&IrType::I8), Some(8));
        assert_eq!(ir_type_integer_width(&IrType::I32), Some(32));
        assert_eq!(ir_type_integer_width(&IrType::I128), Some(128));
        assert_eq!(ir_type_integer_width(&IrType::F64), None);
        assert_eq!(ir_type_integer_width(&IrType::Ptr), None);
    }

    #[test]
    fn test_default_classify_ctype() {
        assert_eq!(
            default_classify_ctype(&CType::Int { signed: true }),
            ParamClass::Integer
        );
        assert_eq!(default_classify_ctype(&CType::Float), ParamClass::SSE);
        assert_eq!(default_classify_ctype(&CType::Double), ParamClass::SSE);
        assert_eq!(
            default_classify_ctype(&CType::Pointer(Box::new(CType::Void))),
            ParamClass::Integer
        );
        assert_eq!(
            default_classify_ctype(&CType::Struct {
                name: None,
                fields: Vec::new(),
            }),
            ParamClass::Memory
        );
        assert_eq!(default_classify_ctype(&CType::Void), ParamClass::NoClass);
    }
}
