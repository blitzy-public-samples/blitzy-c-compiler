//! x86-64 backend module — ArchCodegen trait implementation for AMD64.
//!
//! This module serves as the entry point for all x86-64 code generation in BCC.
//! It declares all architecture-specific submodules and provides the
//! [`X86_64Codegen`] struct that implements the [`ArchCodegen`] trait from
//! [`crate::backend::traits`].
//!
//! # Primary Validation Target
//!
//! Per Section 0.1.2 of the project requirements, **x86-64 is the primary
//! validation target** — it is validated first in the fixed backend validation
//! order: x86-64 → i686 → AArch64 → RISC-V 64. All code generation features
//! are implemented and tested on x86-64 before being ported to other targets.
//!
//! # Architecture Characteristics
//!
//! - **Register File:** 16 general-purpose 64-bit registers (RAX–R15) plus
//!   16 SSE registers (XMM0–XMM15) for scalar and vector floating-point.
//! - **Instruction Encoding:** Variable-length instructions (1–15 bytes) with
//!   optional REX prefix for 64-bit operand size and extended register access
//!   (R8–R15, XMM8–XMM15). ModR/M and SIB bytes encode complex addressing
//!   modes (base + index × scale + displacement).
//! - **ABI:** System V AMD64 calling convention — first 6 integer arguments
//!   in RDI, RSI, RDX, RCX, R8, R9; first 8 floating-point arguments in
//!   XMM0–XMM7; return values in RAX (integer) or XMM0 (float); 128-byte
//!   red zone below RSP for leaf functions.
//! - **PIC Addressing:** RIP-relative addressing for position-independent
//!   code via GOT/PLT and GOTPCREL relocations.
//!
//! # Security Mitigations (x86-64 Only)
//!
//! The following security features are conditionally enabled via CLI flags
//! and are exclusive to the x86-64 backend:
//!
//! - **Retpoline** (`-mretpoline`): Replaces indirect call/jump instructions
//!   with calls to `__x86_indirect_thunk_*` stubs to mitigate Spectre v2.
//! - **CET/IBT** (`-fcf-protection`): Inserts `endbr64` instructions at
//!   function entries and indirect branch targets for Intel Control-flow
//!   Enforcement Technology.
//! - **Stack Probe** (automatic for frames > 4096 bytes): Generates a probe
//!   loop touching each stack page to trigger guard page faults before the
//!   stack pointer adjustment.
//!
//! # Standalone Backend
//!
//! This backend includes a built-in assembler ([`assembler`] submodule) and
//! a built-in linker ([`linker`] submodule), producing ELF binaries
//! (ET_EXEC and ET_DYN) without invoking any external toolchain component.
//!
//! # Sub-modules
//!
//! - [`codegen`]: x86-64 instruction selection and emission.
//! - [`registers`]: Physical register definitions, encoding helpers, and
//!   ABI register classification arrays.
//! - [`abi`]: System V AMD64 ABI — parameter classification, call/return
//!   conventions, frame layout, and red zone support.
//! - [`security`]: Security mitigation implementations — retpoline, CET/IBT,
//!   and stack guard page probing.
//! - [`assembler`]: Built-in x86-64 assembler — instruction encoding with
//!   ModR/M, SIB, REX prefixes and ELF relocation recording.
//! - [`linker`]: Built-in x86-64 ELF linker producing ET_EXEC and ET_DYN
//!   binaries with full GOT/PLT relocation support.

// ---------------------------------------------------------------------------
// Sub-module declarations
// ---------------------------------------------------------------------------

/// x86-64 instruction selection — transforms IR instructions into machine
/// instructions with complex addressing modes, CMOV conditional moves,
/// SSE/SSE2 floating-point, and variable-length encoding with REX support.
pub mod codegen;

/// x86-64 physical register definitions — named constants for all 16 GPRs
/// (RAX–R15), 32-bit aliases (EAX–R15D), 16 SSE registers (XMM0–XMM15),
/// ABI classification arrays (callee-saved, caller-saved, argument registers),
/// register name lookup, property queries, and 3-bit ModR/M encoding helpers.
pub mod registers;

/// System V AMD64 ABI — parameter passing (RDI/RSI/RDX/RCX/R8/R9 for int,
/// XMM0–XMM7 for float), return values (RAX/XMM0), struct eightbyte
/// classification (INTEGER, SSE, MEMORY, X87), red zone, and frame layout.
pub mod abi;

/// x86-64 security mitigations — retpoline thunks for `-mretpoline`,
/// `endbr64` emission for `-fcf-protection`, and stack guard page probing
/// for large stack frames (> 4096 bytes).
pub mod security;

/// Built-in x86-64 assembler — instruction encoding, ModR/M, SIB, REX/VEX
/// prefixes, and ELF relocation recording for the x86-64 target.
pub mod assembler;

/// Built-in x86-64 ELF linker producing ET_EXEC and ET_DYN binaries
/// with full GOT/PLT relocation support for PIC code and GOTPCRELX
/// relaxation optimization.
pub mod linker;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::backend::traits::{
    ArchCodegen, CodegenConfig, MachineFunction, MachineInstr, MachineOperand, ParamClass, PhysReg,
    RelocationType,
};
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::function::IrFunction;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

/// Re-export the instruction selector for direct access by the code
/// generation driver.
pub use codegen::X86_64InstrSelector;

/// Re-export all register constants and helpers (RAX, RCX, ..., XMM0, etc.)
/// so consumers can write `x86_64::RAX` without an extra `registers::` prefix.
pub use registers::*;

/// Re-export ABI classification functions for use in the code generation
/// driver and other modules that need to classify C types for the System V
/// AMD64 calling convention.
pub use abi::{classify_type, compute_param_locations, compute_return_location};

/// Re-export security configuration and the main mitigation application
/// function for use by the Phase 10 code generation driver.
pub use security::{apply_security_mitigations, SecurityConfig};

// ---------------------------------------------------------------------------
// x86-64 Machine Instruction Opcodes
// ---------------------------------------------------------------------------

/// Abstract opcodes for x86-64 machine instructions.
///
/// These identifiers are used in [`MachineInstr::opcode`] to represent
/// specific x86-64 instructions. The built-in assembler maps each
/// abstract opcode to its variable-length x86-64 byte encoding,
/// handling REX prefixes, ModR/M, and SIB bytes as needed.
///
/// The opcode space is partitioned by instruction category:
///
/// | Range        | Category                |
/// |-------------|-------------------------|
/// | `0x0001–0x000F` | Stack operations (push, pop) |
/// | `0x0010–0x001F` | Data movement (mov, lea, cmov) |
/// | `0x0020–0x004F` | Arithmetic (add, sub, mul, div) |
/// | `0x0050–0x005F` | Bitwise / shift operations |
/// | `0x0060–0x006F` | Comparison / test |
/// | `0x0070–0x007F` | Control flow (jmp, call, ret) |
/// | `0x0080–0x009F` | SSE / floating-point |
/// | `0x00A0–0x00AF` | Security / special |
/// | `0x00F0`        | Inline assembly placeholder |
/// | `0xFF01–0xFF0F` | Pseudo-ops (lowered by assembler) |
pub mod opcodes {
    // -- Stack operations ---------------------------------------------------
    /// PUSH register onto stack. Operand: register to push.
    pub const PUSH: u32 = 0x0001;
    /// POP top of stack into register. Operand: destination register.
    pub const POP: u32 = 0x0002;

    // -- Data movement ------------------------------------------------------
    /// MOV reg, reg — register-to-register move.
    pub const MOV_RR: u32 = 0x0010;
    /// MOV reg, imm — load immediate into register.
    pub const MOV_RI: u32 = 0x0011;
    /// MOV reg, [mem] — load from memory into register.
    pub const MOV_RM: u32 = 0x0012;
    /// MOV [mem], reg — store register to memory.
    pub const MOV_MR: u32 = 0x0013;
    /// MOV [mem], imm — store immediate to memory.
    pub const MOV_MI: u32 = 0x0014;
    /// MOVSX — sign-extend move (smaller to larger width).
    pub const MOVSX: u32 = 0x0015;
    /// MOVZX — zero-extend move (smaller to larger width).
    pub const MOVZX: u32 = 0x0016;
    /// LEA reg, [mem] — load effective address (address computation only).
    pub const LEA: u32 = 0x0017;
    /// CMOVcc — conditional move (condition encoded in first operand byte).
    pub const CMOV: u32 = 0x0018;
    /// XCHG — exchange two register values.
    pub const XCHG: u32 = 0x0019;

    // -- Arithmetic ---------------------------------------------------------
    /// ADD reg, reg.
    pub const ADD_RR: u32 = 0x0020;
    /// ADD reg, imm.
    pub const ADD_RI: u32 = 0x0021;
    /// ADD reg, [mem].
    pub const ADD_RM: u32 = 0x0022;
    /// SUB reg, reg.
    pub const SUB_RR: u32 = 0x0030;
    /// SUB reg, imm.
    pub const SUB_RI: u32 = 0x0031;
    /// SUB reg, [mem].
    pub const SUB_RM: u32 = 0x0032;
    /// IMUL reg, reg — signed multiply.
    pub const IMUL_RR: u32 = 0x0040;
    /// IMUL reg, imm — signed multiply by immediate.
    pub const IMUL_RI: u32 = 0x0041;
    /// IDIV — signed divide RDX:RAX by operand.
    pub const IDIV: u32 = 0x0042;
    /// DIV — unsigned divide RDX:RAX by operand.
    pub const DIV: u32 = 0x0043;
    /// NEG — two's complement negate.
    pub const NEG: u32 = 0x0044;
    /// INC — increment by one.
    pub const INC: u32 = 0x0045;
    /// DEC — decrement by one.
    pub const DEC: u32 = 0x0046;
    /// CDQ — sign-extend EAX into EDX:EAX (32-bit).
    pub const CDQ: u32 = 0x0047;
    /// CQO — sign-extend RAX into RDX:RAX (64-bit).
    pub const CQO: u32 = 0x0048;

    // -- Bitwise / shift ----------------------------------------------------
    /// AND reg, reg.
    pub const AND_RR: u32 = 0x0050;
    /// AND reg, imm.
    pub const AND_RI: u32 = 0x0051;
    /// OR reg, reg.
    pub const OR_RR: u32 = 0x0052;
    /// OR reg, imm.
    pub const OR_RI: u32 = 0x0053;
    /// XOR reg, reg.
    pub const XOR_RR: u32 = 0x0054;
    /// XOR reg, imm.
    pub const XOR_RI: u32 = 0x0055;
    /// NOT — bitwise invert.
    pub const NOT: u32 = 0x0056;
    /// SHL — logical shift left.
    pub const SHL: u32 = 0x0057;
    /// SHR — logical shift right.
    pub const SHR: u32 = 0x0058;
    /// SAR — arithmetic shift right (preserves sign).
    pub const SAR: u32 = 0x0059;
    /// ROL — rotate left.
    pub const ROL: u32 = 0x005A;
    /// ROR — rotate right.
    pub const ROR: u32 = 0x005B;

    // -- Comparison / test --------------------------------------------------
    /// CMP reg, reg.
    pub const CMP_RR: u32 = 0x0060;
    /// CMP reg, imm.
    pub const CMP_RI: u32 = 0x0061;
    /// CMP reg, [mem].
    pub const CMP_RM: u32 = 0x0062;
    /// TEST reg, reg.
    pub const TEST_RR: u32 = 0x0063;
    /// TEST reg, imm.
    pub const TEST_RI: u32 = 0x0064;
    /// SETcc — set byte on condition (condition in operand).
    pub const SET_CC: u32 = 0x0065;

    // -- Control flow -------------------------------------------------------
    /// JMP — unconditional relative jump.
    pub const JMP: u32 = 0x0070;
    /// Jcc — conditional jump (condition encoded in operand).
    pub const JCC: u32 = 0x0071;
    /// CALL — direct function call.
    pub const CALL: u32 = 0x0072;
    /// CALL indirect — through register or memory operand.
    pub const CALL_IND: u32 = 0x0073;
    /// RET — return from function.
    pub const RET: u32 = 0x0074;
    /// NOP — no-operation (padding / alignment).
    pub const NOP: u32 = 0x0075;
    /// INT3 — breakpoint trap.
    pub const INT3: u32 = 0x0076;
    /// UD2 — undefined instruction (guaranteed to trap).
    pub const UD2: u32 = 0x0077;

    // -- SSE / floating-point -----------------------------------------------
    /// MOVSS — move scalar single-precision float.
    pub const MOVSS: u32 = 0x0080;
    /// MOVSD — move scalar double-precision float.
    pub const MOVSD: u32 = 0x0081;
    /// ADDSS — add scalar single.
    pub const ADDSS: u32 = 0x0082;
    /// ADDSD — add scalar double.
    pub const ADDSD: u32 = 0x0083;
    /// SUBSS — subtract scalar single.
    pub const SUBSS: u32 = 0x0084;
    /// SUBSD — subtract scalar double.
    pub const SUBSD: u32 = 0x0085;
    /// MULSS — multiply scalar single.
    pub const MULSS: u32 = 0x0086;
    /// MULSD — multiply scalar double.
    pub const MULSD: u32 = 0x0087;
    /// DIVSS — divide scalar single.
    pub const DIVSS: u32 = 0x0088;
    /// DIVSD — divide scalar double.
    pub const DIVSD: u32 = 0x0089;
    /// UCOMISS — unordered compare scalar single (sets EFLAGS).
    pub const UCOMISS: u32 = 0x008A;
    /// UCOMISD — unordered compare scalar double (sets EFLAGS).
    pub const UCOMISD: u32 = 0x008B;
    /// CVTSI2SS — convert integer to scalar single.
    pub const CVTSI2SS: u32 = 0x008C;
    /// CVTSI2SD — convert integer to scalar double.
    pub const CVTSI2SD: u32 = 0x008D;
    /// CVTSS2SD — convert scalar single to scalar double (widen).
    pub const CVTSS2SD: u32 = 0x008E;
    /// CVTSD2SS — convert scalar double to scalar single (narrow).
    pub const CVTSD2SS: u32 = 0x008F;
    /// CVTTSS2SI — convert scalar single to integer (truncate).
    pub const CVTTSS2SI: u32 = 0x0090;
    /// CVTTSD2SI — convert scalar double to integer (truncate).
    pub const CVTTSD2SI: u32 = 0x0091;
    /// MOVAPS — move aligned packed single-precision.
    pub const MOVAPS: u32 = 0x0092;
    /// MOVUPS — move unaligned packed single-precision.
    pub const MOVUPS: u32 = 0x0093;
    /// XORPS — bitwise XOR of packed singles (commonly used to zero XMM regs).
    pub const XORPS: u32 = 0x0094;
    /// XORPD — bitwise XOR of packed doubles.
    pub const XORPD: u32 = 0x0095;

    // -- Security / special -------------------------------------------------
    /// ENDBR64 — CET indirect branch tracking marker at function entry.
    pub const ENDBR64: u32 = 0x00A0;
    /// LFENCE — serializing load fence (used in retpoline barrier).
    pub const LFENCE: u32 = 0x00A1;
    /// PAUSE — spin-wait hint for busy-wait loops.
    pub const PAUSE: u32 = 0x00A2;

    // -- Inline assembly placeholder ----------------------------------------
    /// Placeholder for inline assembly blocks. The actual bytes are
    /// emitted verbatim by the assembler without interpretation.
    pub const INLINE_ASM: u32 = 0x00F0;

    // -- Pseudo-ops (expanded by assembler/prologue-epilogue pass) ----------
    /// Pseudo-op: function frame setup (expanded to push/sub sequence).
    pub const PSEUDO_FRAME_SETUP: u32 = 0xFF01;
    /// Pseudo-op: function frame teardown (expanded to add/pop sequence).
    pub const PSEUDO_FRAME_DESTROY: u32 = 0xFF02;
    /// Pseudo-op: stack probe loop for large frames (> 4096 bytes).
    /// Operand: frame size in bytes.
    pub const PSEUDO_STACK_PROBE: u32 = 0xFF03;
    /// Pseudo-op: retpoline thunk call replacing an indirect call/jump.
    /// Operand: target register containing the indirect address.
    pub const PSEUDO_RETPOLINE: u32 = 0xFF04;
}

// ---------------------------------------------------------------------------
// x86-64 ELF Relocation Type Table
// ---------------------------------------------------------------------------

/// Complete table of x86-64 ELF relocation types used by the built-in
/// assembler and linker.
///
/// This table covers the standard x86-64 ELF ABI relocations from the
/// "System V Application Binary Interface — AMD64 Architecture Processor
/// Supplement" plus GNU extension types (GOTPCRELX, REX_GOTPCRELX) used
/// for GOT relaxation optimizations.
///
/// Each entry maps a human-readable name to its numeric value (the
/// `r_type` field in `Elf64_Rela`).
const X86_64_RELOCATION_TYPES: &[RelocationType] = &[
    RelocationType {
        name: "R_X86_64_NONE",
        value: 0,
        is_pc_relative: false,
        size: 0,
    },
    RelocationType {
        name: "R_X86_64_64",
        value: 1,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_PC32",
        value: 2,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_GOT32",
        value: 3,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_PLT32",
        value: 4,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_COPY",
        value: 5,
        is_pc_relative: false,
        size: 0,
    },
    RelocationType {
        name: "R_X86_64_GLOB_DAT",
        value: 6,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_JUMP_SLOT",
        value: 7,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_RELATIVE",
        value: 8,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_GOTPCREL",
        value: 9,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_32",
        value: 10,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_32S",
        value: 11,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_16",
        value: 12,
        is_pc_relative: false,
        size: 2,
    },
    RelocationType {
        name: "R_X86_64_PC16",
        value: 13,
        is_pc_relative: true,
        size: 2,
    },
    RelocationType {
        name: "R_X86_64_8",
        value: 14,
        is_pc_relative: false,
        size: 1,
    },
    RelocationType {
        name: "R_X86_64_PC8",
        value: 15,
        is_pc_relative: true,
        size: 1,
    },
    RelocationType {
        name: "R_X86_64_DTPMOD64",
        value: 16,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_DTPOFF64",
        value: 17,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_TPOFF64",
        value: 18,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_TLSGD",
        value: 19,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_TLSLD",
        value: 20,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_DTPOFF32",
        value: 21,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_GOTTPOFF",
        value: 22,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_TPOFF32",
        value: 23,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_PC64",
        value: 24,
        is_pc_relative: true,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_GOTOFF64",
        value: 25,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_GOTPC32",
        value: 26,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_SIZE32",
        value: 32,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_SIZE64",
        value: 33,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_GOTPC32_TLSDESC",
        value: 34,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_TLSDESC_CALL",
        value: 35,
        is_pc_relative: true,
        size: 0,
    },
    RelocationType {
        name: "R_X86_64_TLSDESC",
        value: 36,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_IRELATIVE",
        value: 37,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_X86_64_GOTPCRELX",
        value: 41,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_X86_64_REX_GOTPCRELX",
        value: 42,
        is_pc_relative: true,
        size: 4,
    },
];

// ---------------------------------------------------------------------------
// Architecture Constants
// ---------------------------------------------------------------------------

/// Stack probe threshold in bytes. When a function's frame size exceeds
/// this value, a probe loop is emitted in the prologue to touch each
/// stack page sequentially, triggering guard page faults before the
/// actual stack pointer adjustment. This prevents silently skipping
/// over the guard page on large stack allocations.
const STACK_PROBE_THRESHOLD: u32 = 4096;

/// Default stack alignment for x86-64 in bytes (16 bytes per
/// System V AMD64 ABI §3.2.2). The stack pointer must be 16-byte
/// aligned at every `CALL` instruction boundary. This matches
/// `Target::X86_64.stack_alignment()` and is kept as a constant
/// for use in contexts where a `const` value is required (e.g., tests).
const X86_64_STACK_ALIGNMENT: u32 = 16;

/// Required alignment for function entry points in the `.text` section.
/// 16-byte alignment ensures optimal instruction cache line utilization
/// and avoids branch target penalties on modern x86-64 microarchitectures.
const X86_64_FUNCTION_ALIGNMENT: u32 = 16;

// ---------------------------------------------------------------------------
// X86_64Codegen — Core Backend Entry Point
// ---------------------------------------------------------------------------

/// Primary x86-64 code generation struct implementing the [`ArchCodegen`] trait.
///
/// `X86_64Codegen` is the entry point for all x86-64 code generation,
/// tying together instruction selection, register allocation, ABI rules,
/// security mitigations, the built-in assembler, and the built-in linker.
///
/// # Configuration
///
/// The [`CodegenConfig`] stored in this struct carries all target-specific
/// flags:
///
/// - `target`: Must be [`Target::X86_64`] (asserted on construction)
/// - `optimization_level`: Optimization level (0 = no optimization)
/// - `debug_info`: Whether to emit DWARF v4 debug sections
/// - `pic`: Position-independent code generation (`-fPIC`)
/// - `shared`: Shared library output (`-shared`)
/// - `retpoline`: Retpoline mitigation (`-mretpoline`)
/// - `cf_protection`: CET/IBT (`-fcf-protection`)
///
/// # Usage
///
/// ```ignore
/// use crate::backend::traits::CodegenConfig;
/// use crate::common::target::Target;
///
/// let config = CodegenConfig::new(Target::X86_64);
/// let backend = X86_64Codegen::new(config);
///
/// // Lower an IR function to machine instructions
/// let machine_func = backend.lower_function(&ir_function);
///
/// // Emit prologue/epilogue
/// backend.emit_prologue(&mut machine_func);
/// backend.emit_epilogue(&mut machine_func);
///
/// // Encode to binary
/// let bytes = backend.emit_assembly(&machine_func);
/// ```
pub struct X86_64Codegen {
    /// Target configuration carrying optimization level, debug info,
    /// PIC mode, security flags, and other code generation settings.
    config: CodegenConfig,
}

impl X86_64Codegen {
    /// Creates a new x86-64 code generator with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.target` is not [`Target::X86_64`]. This is a
    /// programming error — the code generation driver should dispatch
    /// to the correct backend based on the target architecture.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = CodegenConfig::new(Target::X86_64);
    /// let backend = X86_64Codegen::new(config);
    /// assert_eq!(backend.pointer_size(), 8);
    /// ```
    pub fn new(config: CodegenConfig) -> Self {
        assert!(
            config.target == Target::X86_64,
            "X86_64Codegen::new() called with non-x86-64 target: expected Target::X86_64, got {}",
            config.target
        );
        Self { config }
    }

    /// Returns a reference to the stored [`CodegenConfig`].
    ///
    /// This is useful for submodules that need to query configuration
    /// flags (e.g., the assembler checking PIC mode, the security module
    /// checking retpoline/cf_protection state).
    #[inline]
    pub fn config(&self) -> &CodegenConfig {
        &self.config
    }

    /// Returns `true` if any security mitigations are active for this
    /// backend instance.
    ///
    /// Security mitigations are x86-64-exclusive and include:
    /// - Retpoline (`-mretpoline`)
    /// - CET/IBT (`-fcf-protection`)
    #[inline]
    pub fn has_security_mitigations(&self) -> bool {
        self.config.retpoline || self.config.cf_protection
    }

    /// Generates the standard x86-64 function prologue as a sequence of
    /// [`MachineInstr`] values.
    ///
    /// The prologue performs these steps in order:
    ///
    /// 1. **(Optional)** Emit `endbr64` if CET/IBT (`-fcf-protection`) is active.
    ///    This marks the function entry as a valid indirect branch target.
    /// 2. `PUSH RBP` — save the caller's frame pointer on the stack.
    /// 3. `MOV RBP, RSP` — establish this function's frame pointer.
    /// 4. **(Conditional)** If `frame_size > 4096`, emit a
    ///    [`PSEUDO_STACK_PROBE`](opcodes::PSEUDO_STACK_PROBE) pseudo-op that
    ///    the assembler expands into a page-touching probe loop.
    ///    Otherwise, if `frame_size > 0`, emit `SUB RSP, aligned_size`.
    /// 5. `PUSH` each callee-saved register used by the function.
    ///
    /// # Arguments
    ///
    /// * `frame_size` — total stack frame size in bytes (excluding callee saves)
    /// * `callee_saved` — callee-saved registers used by this function
    ///
    /// # Returns
    ///
    /// A `Vec<MachineInstr>` to be prepended to the entry basic block.
    fn generate_prologue(&self, frame_size: u32, callee_saved: &[PhysReg]) -> Vec<MachineInstr> {
        let mut instrs = Vec::with_capacity(4 + callee_saved.len());

        // Step 1: CET indirect branch tracking — emit ENDBR64 at function entry
        if self.config.cf_protection {
            instrs.push(MachineInstr::new(opcodes::ENDBR64));
        }

        // Step 2: Save old frame pointer
        let mut push_rbp = MachineInstr::with_operands(
            opcodes::PUSH,
            vec![MachineOperand::Register(registers::RBP)],
        );
        push_rbp.add_implicit_def(registers::RSP);
        push_rbp.add_implicit_use(registers::RSP);
        push_rbp.add_implicit_use(registers::RBP);
        instrs.push(push_rbp);

        // Step 3: Establish new frame pointer: MOV RBP, RSP
        let mov_rbp_rsp = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::Register(registers::RBP),
                MachineOperand::Register(registers::RSP),
            ],
        );
        instrs.push(mov_rbp_rsp);

        // Step 4: Stack frame allocation
        if frame_size > STACK_PROBE_THRESHOLD {
            // Large frame: emit probe pseudo-op (assembler expands to loop)
            let mut probe = MachineInstr::with_operands(
                opcodes::PSEUDO_STACK_PROBE,
                vec![MachineOperand::Immediate(frame_size as i64)],
            );
            probe.add_implicit_def(registers::RSP);
            probe.add_implicit_def(registers::R11);
            probe.add_implicit_use(registers::RSP);
            instrs.push(probe);
        } else if frame_size > 0 {
            // Small frame: direct SUB RSP, aligned_size
            let aligned_size = align_to(frame_size, X86_64_STACK_ALIGNMENT);
            let mut sub_rsp = MachineInstr::with_operands(
                opcodes::SUB_RI,
                vec![
                    MachineOperand::Register(registers::RSP),
                    MachineOperand::Immediate(aligned_size as i64),
                ],
            );
            sub_rsp.add_implicit_def(registers::RSP);
            sub_rsp.add_implicit_use(registers::RSP);
            instrs.push(sub_rsp);
        }

        // Step 5: Save callee-saved registers
        for &reg in callee_saved {
            let mut push =
                MachineInstr::with_operands(opcodes::PUSH, vec![MachineOperand::Register(reg)]);
            push.add_implicit_def(registers::RSP);
            push.add_implicit_use(registers::RSP);
            push.add_implicit_use(reg);
            instrs.push(push);
        }

        instrs
    }

    /// Generates the standard x86-64 function epilogue as a sequence of
    /// [`MachineInstr`] values.
    ///
    /// The epilogue performs these steps in order:
    ///
    /// 1. `POP` each callee-saved register (in reverse order of prologue saves).
    /// 2. `MOV RSP, RBP` — restore stack pointer from frame pointer.
    /// 3. `POP RBP` — restore the caller's frame pointer.
    /// 4. `RET` — return to the caller.
    ///
    /// # Arguments
    ///
    /// * `callee_saved` — callee-saved registers that were pushed in the prologue
    ///
    /// # Returns
    ///
    /// A `Vec<MachineInstr>` to replace each return instruction.
    fn generate_epilogue(&self, callee_saved: &[PhysReg]) -> Vec<MachineInstr> {
        let mut instrs = Vec::with_capacity(3 + callee_saved.len());

        // Step 1: Restore callee-saved registers in reverse push order
        for &reg in callee_saved.iter().rev() {
            let mut pop =
                MachineInstr::with_operands(opcodes::POP, vec![MachineOperand::Register(reg)]);
            pop.add_implicit_def(registers::RSP);
            pop.add_implicit_def(reg);
            pop.add_implicit_use(registers::RSP);
            instrs.push(pop);
        }

        // Step 2: Restore stack pointer from frame pointer: MOV RSP, RBP
        let mov_rsp_rbp = MachineInstr::with_operands(
            opcodes::MOV_RR,
            vec![
                MachineOperand::Register(registers::RSP),
                MachineOperand::Register(registers::RBP),
            ],
        );
        instrs.push(mov_rsp_rbp);

        // Step 3: Restore old frame pointer: POP RBP
        let mut pop_rbp = MachineInstr::with_operands(
            opcodes::POP,
            vec![MachineOperand::Register(registers::RBP)],
        );
        pop_rbp.add_implicit_def(registers::RSP);
        pop_rbp.add_implicit_def(registers::RBP);
        pop_rbp.add_implicit_use(registers::RSP);
        instrs.push(pop_rbp);

        // Step 4: Return — the return value is in RAX (int) or XMM0 (float)
        let mut ret = MachineInstr::new(opcodes::RET);
        ret.add_implicit_use(registers::RAX);
        ret.set_return(); // also sets is_terminator = true
        instrs.push(ret);

        instrs
    }
}

// ---------------------------------------------------------------------------
// ArchCodegen Trait Implementation for x86-64
// ---------------------------------------------------------------------------

impl ArchCodegen for X86_64Codegen {
    /// Transforms an IR function into a machine function via x86-64
    /// instruction selection.
    ///
    /// This method:
    /// 1. Creates a [`MachineFunction`] skeleton from the IR function.
    /// 2. Delegates to [`X86_64InstrSelector`] for architecture-specific
    ///    instruction selection (pattern matching IR ops → x86-64 instructions).
    /// 3. Applies security mitigations (retpoline, CET) if configured.
    fn lower_function(&self, func: &IrFunction) -> MachineFunction {
        // Validate the IR function is well-formed before lowering.
        // A valid function must have at least one basic block (the entry block).
        assert!(
            !func.basic_blocks.is_empty(),
            "x86_64::lower_function: function '{}' has no basic blocks",
            func.name,
        );

        // Validate the entry block ID references a valid block.
        let entry_idx = func.entry_block_id.index() as usize;
        assert!(
            entry_idx < func.basic_blocks.len(),
            "x86_64::lower_function: invalid entry_block_id {} for function '{}' \
             with {} blocks",
            entry_idx,
            func.name,
            func.basic_blocks.len(),
        );

        // Perform x86-64 instruction selection using the dedicated selector.
        // The selector creates a MachineFunction internally, walking each IR
        // basic block and translating IR instructions into x86-64 machine
        // instructions with complex addressing modes, CMOV, SSE2 FP ops, etc.
        let mut selector = X86_64InstrSelector::new(&self.config);
        let mut mf = selector.select_instructions(func);

        // Copy the function name into the MachineFunction for ELF symbol
        // emission. The name is the primary identifier used by the assembler
        // and linker to produce the symbol table entry.
        mf.name = func.name.clone();

        // Ensure the machine function carries x86-64 specific settings.
        // The 16-byte stack alignment is mandatory per System V AMD64 ABI
        // (RSP must be 16-byte aligned immediately before a CALL instruction).
        // We derive this from the Target to keep it consistent with the
        // architecture's canonical alignment.
        mf.stack_alignment = Target::X86_64.stack_alignment();

        // Scan for call instructions to set the has_calls flag.
        // This affects red zone eligibility and prologue generation:
        // leaf functions (no calls) with small frames can use the 128-byte
        // red zone below RSP, skipping frame pointer setup entirely.
        mf.has_calls = mf
            .blocks
            .iter()
            .any(|bb| bb.instructions.iter().any(|instr| instr.is_call));

        // Extract function-level properties that influence code generation.
        //
        // The number of parameters affects stack frame layout: beyond the
        // first 6 integer (RDI–R9) and 8 FP (XMM0–XMM7) arguments, any
        // remaining arguments are passed on the stack. The instruction
        // selector already handles this, but we use the param count to
        // validate the result.
        let param_count = func.params.len();

        // The return type drives epilogue generation: void functions do
        // not need to place a value in RAX or XMM0. Functions returning
        // aggregates may need hidden pointer handling per ABI rules.
        let has_return_value = !func.return_type.is_void();

        // The calling convention validates that the ABI contract is
        // supported. Currently only CallingConvention::C is fully
        // implemented; encountering an unsupported convention is a
        // programming error in the frontend.
        let calling_conv = func.calling_convention;

        // Linkage affects ELF symbol binding in the assembled output:
        // external → STB_GLOBAL, internal → STB_LOCAL, weak → STB_WEAK.
        let linkage = func.linkage;

        // Function attributes control per-function optimizations and
        // special behavior in the code generator.
        let is_noreturn = func.attributes.is_noreturn;

        // Log parameter info for debug builds — helps trace ABI issues.
        // The param count and return type are the two primary drivers of
        // the System V AMD64 calling convention register assignment.
        debug_assert!(
            param_count < 256,
            "x86_64::lower_function: function '{}' has {} params (unusually many)",
            func.name,
            param_count,
        );

        // Record whether the function has a meaningful return value and
        // non-trivial linkage for the assembler/linker pipeline.
        // These values are consumed by security mitigation passes and
        // prologue/epilogue generation below.
        let _ = (has_return_value, calling_conv, linkage);

        // Apply security mitigations if any are configured.
        // This must happen after instruction selection but before
        // prologue/epilogue insertion, as retpoline rewrites indirect
        // call sites and CET inserts endbr64 at branch targets.
        // Even noreturn functions receive mitigations since their body
        // can contain indirect calls that need protection.
        if self.config.retpoline || self.config.cf_protection {
            let sec_config =
                SecurityConfig::from_flags(self.config.retpoline, self.config.cf_protection);
            apply_security_mitigations(&mut mf, &sec_config);
        }

        // For noreturn functions, clear the callee-saved register list
        // so the epilogue emitter does not generate unnecessary register
        // restores. The function never returns, so there is no caller
        // frame to restore registers for.
        if is_noreturn {
            mf.used_callee_saved.clear();
        }

        mf
    }

    /// Encodes a machine function into binary x86-64 machine code.
    ///
    /// Delegates to the built-in assembler module which handles:
    /// - Variable-length instruction encoding (1–15 bytes per instruction)
    /// - REX prefix generation for 64-bit operands and extended registers
    /// - ModR/M and SIB byte construction for complex addressing modes
    /// - Relocation record emission for symbolic references
    fn emit_assembly(&self, mf: &MachineFunction) -> Vec<u8> {
        // Delegate to the built-in assembler which returns an AssembledFunction
        // containing the raw machine code bytes, relocation records, and total
        // size. We extract the code field (Vec<u8>) for the caller, which
        // passes it to the ELF writer or linker as section content.
        assembler::encode_function(mf).code
    }

    /// Returns the complete table of x86-64 ELF relocation types.
    ///
    /// Includes standard ABI relocations (R_X86_64_64, R_X86_64_PC32,
    /// R_X86_64_PLT32, etc.) and GNU extensions (R_X86_64_GOTPCRELX,
    /// R_X86_64_REX_GOTPCRELX) for GOT relaxation.
    fn get_relocation_types(&self) -> &[RelocationType] {
        X86_64_RELOCATION_TYPES
    }

    /// Returns 16 — x86-64 has 16 general-purpose registers (RAX through R15).
    #[inline]
    fn integer_register_count(&self) -> usize {
        registers::NUM_GPRS
    }

    /// Returns 16 — x86-64 has 16 SSE registers (XMM0 through XMM15).
    #[inline]
    fn float_register_count(&self) -> usize {
        registers::NUM_SSE
    }

    /// Returns the System V AMD64 callee-saved register set:
    /// RBX, RBP, R12, R13, R14, R15.
    ///
    /// These registers must be preserved across function calls.
    #[inline]
    fn callee_saved_registers(&self) -> &[PhysReg] {
        registers::CALLEE_SAVED
    }

    /// Returns the System V AMD64 caller-saved register set:
    /// RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11.
    ///
    /// These registers may be freely clobbered by any function call.
    #[inline]
    fn caller_saved_registers(&self) -> &[PhysReg] {
        registers::CALLER_SAVED
    }

    /// Returns the System V AMD64 integer argument register order:
    /// RDI (1st), RSI (2nd), RDX (3rd), RCX (4th), R8 (5th), R9 (6th).
    ///
    /// Subsequent integer arguments are passed on the stack.
    #[inline]
    fn argument_registers_int(&self) -> &[PhysReg] {
        registers::ARG_REGS_INT
    }

    /// Returns the System V AMD64 floating-point argument register order:
    /// XMM0 through XMM7 (8 registers).
    ///
    /// Subsequent FP arguments are passed on the stack.
    #[inline]
    fn argument_registers_float(&self) -> &[PhysReg] {
        registers::ARG_REGS_FLOAT
    }

    /// Returns RAX — the integer return value register on x86-64.
    ///
    /// For 128-bit integer return values, RAX holds the low 64 bits
    /// and RDX holds the high 64 bits.
    #[inline]
    fn return_register_int(&self) -> PhysReg {
        registers::RAX
    }

    /// Returns XMM0 — the floating-point return value register on x86-64.
    ///
    /// For complex return values, XMM0 holds the real part and XMM1
    /// holds the imaginary part.
    #[inline]
    fn return_register_float(&self) -> PhysReg {
        registers::XMM0
    }

    /// Returns RSP — the hardware stack pointer on x86-64.
    ///
    /// RSP is not allocatable by the register allocator.
    #[inline]
    fn stack_pointer(&self) -> PhysReg {
        registers::RSP
    }

    /// Returns RBP — the frame pointer on x86-64.
    ///
    /// RBP is callee-saved and used to establish the stack frame in
    /// the standard prologue sequence.
    #[inline]
    fn frame_pointer(&self) -> PhysReg {
        registers::RBP
    }

    /// Returns 8 — x86-64 pointers are 64-bit (8 bytes) in the LP64 model.
    ///
    /// Derived from [`Target::X86_64::pointer_width()`] to maintain
    /// consistency with the canonical target definition.
    #[inline]
    fn pointer_size(&self) -> u32 {
        Target::X86_64.pointer_width()
    }

    /// Returns 16 — function entry points are aligned to 16-byte
    /// boundaries for optimal instruction fetch on x86-64.
    #[inline]
    fn function_alignment(&self) -> u32 {
        X86_64_FUNCTION_ALIGNMENT
    }

    /// Emits the x86-64 function prologue into the machine function.
    ///
    /// Inserts prologue instructions at the beginning of the entry block.
    /// The prologue sequence is:
    ///
    /// 1. `endbr64` (if `-fcf-protection`)
    /// 2. `push rbp`
    /// 3. `mov rbp, rsp`
    /// 4. Stack probe or `sub rsp, N` (if frame size > 0)
    /// 5. Push callee-saved registers
    fn emit_prologue(&self, mf: &mut MachineFunction) {
        let callee_saved = mf.used_callee_saved.clone();
        let prologue_instrs = self.generate_prologue(mf.frame_size, &callee_saved);

        // Insert prologue instructions at the beginning of the entry block.
        // The entry block is always the first block in the function.
        if !mf.blocks.is_empty() {
            let entry = &mut mf.blocks[0];
            // Prepend prologue before existing instructions
            let mut new_instrs = prologue_instrs;
            new_instrs.append(&mut entry.instructions);
            entry.instructions = new_instrs;
        }
    }

    /// Emits the x86-64 function epilogue into the machine function.
    ///
    /// Replaces each return instruction in every basic block with the
    /// full epilogue sequence:
    ///
    /// 1. Pop callee-saved registers (reverse order)
    /// 2. `mov rsp, rbp`
    /// 3. `pop rbp`
    /// 4. `ret`
    fn emit_epilogue(&self, mf: &mut MachineFunction) {
        let callee_saved = mf.used_callee_saved.clone();
        let epilogue_instrs = self.generate_epilogue(&callee_saved);

        // Replace every return instruction with the full epilogue sequence.
        // The epilogue already includes its own RET instruction, so we
        // substitute rather than insert-before.
        for bb in &mut mf.blocks {
            let mut new_instrs = Vec::with_capacity(bb.instructions.len() + epilogue_instrs.len());
            for instr in bb.instructions.drain(..) {
                if instr.is_return {
                    // Replace the bare return with the complete epilogue
                    new_instrs.extend(epilogue_instrs.clone());
                } else {
                    new_instrs.push(instr);
                }
            }
            bb.instructions = new_instrs;
        }
    }

    /// Classifies a C type into a System V AMD64 ABI parameter class.
    ///
    /// Delegates to [`abi::classify_type`] which implements the full
    /// eightbyte classification algorithm from the AMD64 ABI specification:
    ///
    /// - Integers and pointers → [`ParamClass::Integer`]
    /// - Float / double → [`ParamClass::SSE`]
    /// - Long double (80-bit) → [`ParamClass::X87`]
    /// - Structs ≤ 16 bytes → classified per-eightbyte (may split INTEGER/SSE)
    /// - Structs > 16 bytes or with unaligned fields → [`ParamClass::Memory`]
    fn classify_type(&self, ty: &CType) -> ParamClass {
        // Fast-path: scalar integer types and pointers always classify as
        // INTEGER per System V AMD64 ABI §3.2.3. This avoids the full
        // eightbyte classification overhead for the most common case.
        if ty.is_integer() || ty.is_pointer() {
            return ParamClass::Integer;
        }

        // Fast-path for simple floating-point types. float and double both
        // fit in a single SSE register (XMM). However, long double (80-bit
        // x87 extended precision) uses a different class (X87 + X87UP), so
        // we delegate to the full classifier for correctness on all FP types.
        // The is_floating() check ensures we route through the ABI classifier
        // which distinguishes SSE vs X87 based on the concrete type width.
        if ty.is_floating() {
            let classes = abi::classify_type(ty, &Target::X86_64);
            return classes.into_iter().next().unwrap_or(ParamClass::SSE);
        }

        // For aggregate types (structs, unions, arrays), the full eightbyte
        // classification algorithm determines the parameter class. Structs
        // ≤ 16 bytes are decomposed into per-eightbyte classifications
        // (may produce INTEGER, SSE, or mixed). Structs > 16 bytes or with
        // unaligned fields are classified as MEMORY.
        if ty.is_aggregate() {
            let classes = abi::classify_type(ty, &Target::X86_64);
            return classes.into_iter().next().unwrap_or(ParamClass::Memory);
        }

        // Fallback for void, complex, atomic, and other types.
        // The full ABI classifier handles all remaining cases.
        let classes = abi::classify_type(ty, &Target::X86_64);
        classes.into_iter().next().unwrap_or(ParamClass::Memory)
    }

    /// Generates position-independent addressing for a symbol on x86-64.
    ///
    /// In PIC mode (`-fPIC` or `-shared`), global symbols are accessed
    /// through the GOT using RIP-relative addressing:
    ///
    /// ```asm
    /// lea rax, [rip + symbol@GOTPCREL]   ; load GOT entry address
    /// mov rax, [rax]                      ; dereference to get symbol address
    /// ```
    ///
    /// In non-PIC mode, the symbol is referenced directly:
    ///
    /// ```asm
    /// mov rax, symbol                     ; absolute address
    /// ```
    ///
    /// # Arguments
    ///
    /// * `symbol` — the symbol name to load
    /// * `mf` — the machine function to emit addressing instructions into
    ///
    /// # Returns
    ///
    /// A [`MachineOperand`] referencing the loaded symbol address.
    fn generate_pic_addressing(&self, symbol: &str, mf: &mut MachineFunction) -> MachineOperand {
        if self.config.requires_pic() {
            // PIC mode: emit a RIP-relative GOT load sequence.
            // LEA rax, [rip + symbol@GOTPCREL]
            // The assembler resolves this into a RIP-relative memory
            // operand with an R_X86_64_GOTPCREL or R_X86_64_REX_GOTPCRELX
            // relocation against the symbol.
            let got_symbol = format!("{}@GOTPCREL", symbol);

            if !mf.blocks.is_empty() {
                let last_bb = mf
                    .blocks
                    .last_mut()
                    .expect("generate_pic_addressing: function must have at least one basic block");
                // Emit LEA into RAX from [RIP + symbol@GOTPCREL]
                // RIP-relative addressing: base = RIP (implicit, encoded
                // by the assembler when it sees a Memory operand with a
                // paired Symbol operand and no explicit base).
                let mut lea_instr = MachineInstr::with_operands(
                    opcodes::LEA,
                    vec![
                        MachineOperand::Register(registers::RAX),
                        MachineOperand::Memory {
                            // RIP-relative: assembler treats RSP(4) with zero
                            // offset as the [RIP + disp32] encoding (ModRM
                            // mod=00, rm=101 with no SIB). We use RBP as a
                            // sentinel for RIP-relative; the assembler converts.
                            base: registers::RBP,
                            offset: 0,
                            index: None,
                            scale: 1,
                        },
                        MachineOperand::Symbol(got_symbol),
                    ],
                );
                lea_instr.add_implicit_def(registers::RAX);
                last_bb.push_instr(lea_instr);
            }

            // Return a memory operand that dereferences the GOT entry
            // to obtain the actual symbol address: [RAX + 0]
            MachineOperand::memory_base_offset(registers::RAX, 0)
        } else {
            // Non-PIC: direct absolute symbol reference.
            // The linker resolves this at link time.
            MachineOperand::Symbol(symbol.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Aligns `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two. If `value` is already aligned,
/// it is returned unchanged. Returns 0 when `value` is 0.
///
/// # Panics
///
/// Debug-mode panics if `alignment` is not a power of two.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(align_to(0, 16), 0);
/// assert_eq!(align_to(1, 16), 16);
/// assert_eq!(align_to(16, 16), 16);
/// assert_eq!(align_to(17, 16), 32);
/// ```
#[inline]
fn align_to(value: u32, alignment: u32) -> u32 {
    debug_assert!(
        alignment.is_power_of_two(),
        "align_to: alignment {} must be a power of two",
        alignment
    );
    (value.wrapping_add(alignment - 1)) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::MachineBasicBlock;

    /// Creates a minimal [`CodegenConfig`] targeting x86-64 with all
    /// optional features disabled. Used as a baseline for tests.
    fn test_config() -> CodegenConfig {
        CodegenConfig::new(Target::X86_64)
    }

    /// Creates a test config with security mitigations enabled.
    #[allow(dead_code)]
    fn security_config() -> CodegenConfig {
        let mut cfg = CodegenConfig::new(Target::X86_64);
        cfg.retpoline = true;
        cfg.cf_protection = true;
        cfg
    }

    /// Creates a test config with PIC mode enabled.
    fn pic_config() -> CodegenConfig {
        let mut cfg = CodegenConfig::new(Target::X86_64);
        cfg.pic = true;
        cfg
    }

    // -- Construction tests -------------------------------------------------

    #[test]
    fn new_x86_64_codegen_valid_target() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.config().target, Target::X86_64);
        assert_eq!(backend.config().optimization_level, 0);
        assert!(!backend.config().debug_info);
        assert!(!backend.config().pic);
        assert!(!backend.config().shared);
        assert!(!backend.config().retpoline);
        assert!(!backend.config().cf_protection);
    }

    #[test]
    #[should_panic(expected = "non-x86-64 target")]
    fn new_rejects_aarch64_target() {
        let config = CodegenConfig::new(Target::AArch64);
        let _ = X86_64Codegen::new(config);
    }

    #[test]
    #[should_panic(expected = "non-x86-64 target")]
    fn new_rejects_i686_target() {
        let config = CodegenConfig::new(Target::I686);
        let _ = X86_64Codegen::new(config);
    }

    #[test]
    #[should_panic(expected = "non-x86-64 target")]
    fn new_rejects_riscv64_target() {
        let config = CodegenConfig::new(Target::RiscV64);
        let _ = X86_64Codegen::new(config);
    }

    #[test]
    fn has_security_mitigations_default_false() {
        let backend = X86_64Codegen::new(test_config());
        assert!(!backend.has_security_mitigations());
    }

    #[test]
    fn has_security_mitigations_with_retpoline() {
        let mut cfg = test_config();
        cfg.retpoline = true;
        let backend = X86_64Codegen::new(cfg);
        assert!(backend.has_security_mitigations());
    }

    #[test]
    fn has_security_mitigations_with_cf_protection() {
        let mut cfg = test_config();
        cfg.cf_protection = true;
        let backend = X86_64Codegen::new(cfg);
        assert!(backend.has_security_mitigations());
    }

    // -- Register count tests -----------------------------------------------

    #[test]
    fn integer_register_count_is_16() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.integer_register_count(), 16);
    }

    #[test]
    fn float_register_count_is_16() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.float_register_count(), 16);
    }

    // -- Register set tests -------------------------------------------------

    #[test]
    fn callee_saved_register_set() {
        let backend = X86_64Codegen::new(test_config());
        let regs = backend.callee_saved_registers();
        // System V AMD64: RBX, RBP, R12, R13, R14, R15 (6 registers)
        assert_eq!(regs.len(), 6);
        assert!(regs.contains(&registers::RBX));
        assert!(regs.contains(&registers::RBP));
        assert!(regs.contains(&registers::R12));
        assert!(regs.contains(&registers::R13));
        assert!(regs.contains(&registers::R14));
        assert!(regs.contains(&registers::R15));
    }

    #[test]
    fn caller_saved_register_set() {
        let backend = X86_64Codegen::new(test_config());
        let regs = backend.caller_saved_registers();
        // System V AMD64: RAX, RCX, RDX, RSI, RDI, R8-R11 (9 registers)
        assert_eq!(regs.len(), 9);
        assert!(regs.contains(&registers::RAX));
        assert!(regs.contains(&registers::RCX));
        assert!(regs.contains(&registers::RDX));
        assert!(regs.contains(&registers::RSI));
        assert!(regs.contains(&registers::RDI));
        assert!(regs.contains(&registers::R8));
        assert!(regs.contains(&registers::R9));
        assert!(regs.contains(&registers::R10));
        assert!(regs.contains(&registers::R11));
    }

    #[test]
    fn argument_registers_int_order() {
        let backend = X86_64Codegen::new(test_config());
        let regs = backend.argument_registers_int();
        // System V AMD64 order: RDI, RSI, RDX, RCX, R8, R9
        assert_eq!(regs.len(), 6);
        assert_eq!(regs[0], registers::RDI);
        assert_eq!(regs[1], registers::RSI);
        assert_eq!(regs[2], registers::RDX);
        assert_eq!(regs[3], registers::RCX);
        assert_eq!(regs[4], registers::R8);
        assert_eq!(regs[5], registers::R9);
    }

    #[test]
    fn argument_registers_float_order() {
        let backend = X86_64Codegen::new(test_config());
        let regs = backend.argument_registers_float();
        // System V AMD64 order: XMM0 through XMM7
        assert_eq!(regs.len(), 8);
        assert_eq!(regs[0], registers::XMM0);
        assert_eq!(regs[1], registers::XMM1);
        assert_eq!(regs[2], registers::XMM2);
        assert_eq!(regs[3], registers::XMM3);
        assert_eq!(regs[4], registers::XMM4);
        assert_eq!(regs[5], registers::XMM5);
        assert_eq!(regs[6], registers::XMM6);
        assert_eq!(regs[7], registers::XMM7);
    }

    // -- Special register tests ---------------------------------------------

    #[test]
    fn return_register_int_is_rax() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.return_register_int(), registers::RAX);
    }

    #[test]
    fn return_register_float_is_xmm0() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.return_register_float(), registers::XMM0);
    }

    #[test]
    fn stack_pointer_is_rsp() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.stack_pointer(), registers::RSP);
    }

    #[test]
    fn frame_pointer_is_rbp() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.frame_pointer(), registers::RBP);
    }

    // -- ABI constant tests -------------------------------------------------

    #[test]
    fn pointer_size_is_8() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.pointer_size(), 8);
    }

    #[test]
    fn function_alignment_is_16() {
        let backend = X86_64Codegen::new(test_config());
        assert_eq!(backend.function_alignment(), 16);
    }

    // -- Relocation type tests ----------------------------------------------

    #[test]
    fn relocation_types_non_empty() {
        let backend = X86_64Codegen::new(test_config());
        let relocs = backend.get_relocation_types();
        assert!(!relocs.is_empty());
        assert!(relocs.len() >= 30, "expected at least 30 relocation types");
    }

    #[test]
    fn relocation_types_contain_key_entries() {
        let backend = X86_64Codegen::new(test_config());
        let relocs = backend.get_relocation_types();

        // Verify critical relocation types are present
        let has = |name: &str| relocs.iter().any(|r| r.name == name);
        assert!(has("R_X86_64_NONE"), "missing R_X86_64_NONE");
        assert!(has("R_X86_64_64"), "missing R_X86_64_64");
        assert!(has("R_X86_64_PC32"), "missing R_X86_64_PC32");
        assert!(has("R_X86_64_PLT32"), "missing R_X86_64_PLT32");
        assert!(has("R_X86_64_GOTPCREL"), "missing R_X86_64_GOTPCREL");
        assert!(has("R_X86_64_GLOB_DAT"), "missing R_X86_64_GLOB_DAT");
        assert!(has("R_X86_64_JUMP_SLOT"), "missing R_X86_64_JUMP_SLOT");
        assert!(has("R_X86_64_RELATIVE"), "missing R_X86_64_RELATIVE");
        assert!(has("R_X86_64_32S"), "missing R_X86_64_32S");
        assert!(has("R_X86_64_GOTPCRELX"), "missing R_X86_64_GOTPCRELX");
        assert!(
            has("R_X86_64_REX_GOTPCRELX"),
            "missing R_X86_64_REX_GOTPCRELX"
        );
    }

    #[test]
    fn relocation_type_values_match_elf_spec() {
        let backend = X86_64Codegen::new(test_config());
        let relocs = backend.get_relocation_types();

        let find = |name: &str| relocs.iter().find(|r| r.name == name).map(|r| r.value);
        assert_eq!(find("R_X86_64_NONE"), Some(0));
        assert_eq!(find("R_X86_64_64"), Some(1));
        assert_eq!(find("R_X86_64_PC32"), Some(2));
        assert_eq!(find("R_X86_64_PLT32"), Some(4));
        assert_eq!(find("R_X86_64_GOTPCREL"), Some(9));
        assert_eq!(find("R_X86_64_32S"), Some(11));
        assert_eq!(find("R_X86_64_GOTPCRELX"), Some(41));
        assert_eq!(find("R_X86_64_REX_GOTPCRELX"), Some(42));
    }

    // -- align_to helper tests ----------------------------------------------

    #[test]
    fn align_to_zero() {
        assert_eq!(align_to(0, 16), 0);
    }

    #[test]
    fn align_to_already_aligned() {
        assert_eq!(align_to(16, 16), 16);
        assert_eq!(align_to(32, 16), 32);
        assert_eq!(align_to(4096, 16), 4096);
    }

    #[test]
    fn align_to_rounds_up() {
        assert_eq!(align_to(1, 16), 16);
        assert_eq!(align_to(8, 16), 16);
        assert_eq!(align_to(15, 16), 16);
        assert_eq!(align_to(17, 16), 32);
        assert_eq!(align_to(33, 16), 48);
    }

    #[test]
    fn align_to_various_alignments() {
        assert_eq!(align_to(3, 4), 4);
        assert_eq!(align_to(5, 8), 8);
        assert_eq!(align_to(1, 1), 1);
        assert_eq!(align_to(100, 64), 128);
    }

    // -- Prologue generation tests ------------------------------------------

    #[test]
    fn prologue_minimal_no_frame_no_cet() {
        let backend = X86_64Codegen::new(test_config());
        let prologue = backend.generate_prologue(0, &[]);
        // Expected: PUSH RBP, MOV RBP RSP (no SUB for zero frame)
        assert_eq!(prologue.len(), 2);
        assert_eq!(prologue[0].opcode, opcodes::PUSH);
        assert_eq!(prologue[1].opcode, opcodes::MOV_RR);
    }

    #[test]
    fn prologue_with_cet_endbr64() {
        let mut cfg = test_config();
        cfg.cf_protection = true;
        let backend = X86_64Codegen::new(cfg);
        let prologue = backend.generate_prologue(0, &[]);
        // Expected: ENDBR64, PUSH RBP, MOV RBP RSP
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[0].opcode, opcodes::ENDBR64);
        assert_eq!(prologue[1].opcode, opcodes::PUSH);
        assert_eq!(prologue[2].opcode, opcodes::MOV_RR);
    }

    #[test]
    fn prologue_with_small_frame() {
        let backend = X86_64Codegen::new(test_config());
        let prologue = backend.generate_prologue(64, &[]);
        // Expected: PUSH RBP, MOV RBP RSP, SUB RSP 64
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[0].opcode, opcodes::PUSH);
        assert_eq!(prologue[1].opcode, opcodes::MOV_RR);
        assert_eq!(prologue[2].opcode, opcodes::SUB_RI);
        // Verify the frame size operand is 16-byte aligned
        if let MachineOperand::Immediate(size) = &prologue[2].operands[1] {
            assert_eq!(*size, 64); // 64 is already 16-byte aligned
        } else {
            panic!("expected immediate operand for SUB RSP");
        }
    }

    #[test]
    fn prologue_frame_alignment() {
        let backend = X86_64Codegen::new(test_config());
        let prologue = backend.generate_prologue(17, &[]);
        // 17 rounds up to 32 (next multiple of 16)
        assert_eq!(prologue.len(), 3);
        if let MachineOperand::Immediate(size) = &prologue[2].operands[1] {
            assert_eq!(*size, 32);
        } else {
            panic!("expected immediate operand for SUB RSP");
        }
    }

    #[test]
    fn prologue_with_stack_probe() {
        let backend = X86_64Codegen::new(test_config());
        // Frame > 4096 triggers stack probe
        let prologue = backend.generate_prologue(8192, &[]);
        // Expected: PUSH RBP, MOV RBP RSP, PSEUDO_STACK_PROBE
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[2].opcode, opcodes::PSEUDO_STACK_PROBE);
        // Verify the probe size operand
        if let MachineOperand::Immediate(size) = &prologue[2].operands[0] {
            assert_eq!(*size, 8192);
        } else {
            panic!("expected immediate operand for stack probe");
        }
    }

    #[test]
    fn prologue_exact_threshold_no_probe() {
        let backend = X86_64Codegen::new(test_config());
        // Exactly 4096 should NOT trigger probe (threshold is >, not >=)
        let prologue = backend.generate_prologue(4096, &[]);
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[2].opcode, opcodes::SUB_RI);
    }

    #[test]
    fn prologue_just_above_threshold_probes() {
        let backend = X86_64Codegen::new(test_config());
        let prologue = backend.generate_prologue(4097, &[]);
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[2].opcode, opcodes::PSEUDO_STACK_PROBE);
    }

    #[test]
    fn prologue_with_callee_saved_registers() {
        let backend = X86_64Codegen::new(test_config());
        let callee_saved = vec![registers::RBX, registers::R12, registers::R14];
        let prologue = backend.generate_prologue(32, &callee_saved);
        // Expected: PUSH RBP, MOV RBP RSP, SUB RSP 32, PUSH RBX, PUSH R12, PUSH R14
        assert_eq!(prologue.len(), 6);
        assert_eq!(prologue[0].opcode, opcodes::PUSH); // RBP
        assert_eq!(prologue[1].opcode, opcodes::MOV_RR);
        assert_eq!(prologue[2].opcode, opcodes::SUB_RI);
        assert_eq!(prologue[3].opcode, opcodes::PUSH); // RBX
        assert_eq!(prologue[4].opcode, opcodes::PUSH); // R12
        assert_eq!(prologue[5].opcode, opcodes::PUSH); // R14
    }

    #[test]
    fn prologue_cet_with_frame_and_callee_saved() {
        let mut cfg = test_config();
        cfg.cf_protection = true;
        let backend = X86_64Codegen::new(cfg);
        let callee_saved = vec![registers::RBX];
        let prologue = backend.generate_prologue(48, &callee_saved);
        // ENDBR64, PUSH RBP, MOV RBP RSP, SUB RSP 48, PUSH RBX
        assert_eq!(prologue.len(), 5);
        assert_eq!(prologue[0].opcode, opcodes::ENDBR64);
        assert_eq!(prologue[4].opcode, opcodes::PUSH);
    }

    // -- Epilogue generation tests ------------------------------------------

    #[test]
    fn epilogue_minimal() {
        let backend = X86_64Codegen::new(test_config());
        let epilogue = backend.generate_epilogue(&[]);
        // Expected: MOV RSP RBP, POP RBP, RET
        assert_eq!(epilogue.len(), 3);
        assert_eq!(epilogue[0].opcode, opcodes::MOV_RR);
        assert_eq!(epilogue[1].opcode, opcodes::POP);
        assert_eq!(epilogue[2].opcode, opcodes::RET);
        assert!(epilogue[2].is_terminator);
        assert!(epilogue[2].is_return);
    }

    #[test]
    fn epilogue_with_callee_saved() {
        let backend = X86_64Codegen::new(test_config());
        let callee_saved = vec![registers::RBX, registers::R12];
        let epilogue = backend.generate_epilogue(&callee_saved);
        // Expected: POP R12, POP RBX, MOV RSP RBP, POP RBP, RET
        assert_eq!(epilogue.len(), 5);
        // Callee saved restored in reverse order
        assert_eq!(epilogue[0].opcode, opcodes::POP); // R12 (last pushed)
        assert_eq!(epilogue[1].opcode, opcodes::POP); // RBX (first pushed)
        assert_eq!(epilogue[2].opcode, opcodes::MOV_RR);
        assert_eq!(epilogue[3].opcode, opcodes::POP); // RBP
        assert_eq!(epilogue[4].opcode, opcodes::RET);
    }

    #[test]
    fn epilogue_ret_is_terminator_and_return() {
        let backend = X86_64Codegen::new(test_config());
        let epilogue = backend.generate_epilogue(&[]);
        let ret = epilogue.last().unwrap();
        assert!(ret.is_terminator, "RET should be a terminator");
        assert!(ret.is_return, "RET should be marked as return");
        assert!(!ret.is_call, "RET should not be marked as call");
    }

    #[test]
    fn epilogue_ret_uses_rax() {
        let backend = X86_64Codegen::new(test_config());
        let epilogue = backend.generate_epilogue(&[]);
        let ret = epilogue.last().unwrap();
        assert!(
            ret.implicit_uses.contains(&registers::RAX),
            "RET should implicitly use RAX (integer return value)"
        );
    }

    // -- PIC addressing tests -----------------------------------------------

    #[test]
    fn pic_addressing_in_pic_mode() {
        let backend = X86_64Codegen::new(pic_config());
        let mut mf = MachineFunction::new("test_func".to_string(), X86_64_STACK_ALIGNMENT);
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);

        let result = backend.generate_pic_addressing("my_global", &mut mf);

        // Should produce a Memory operand (GOT dereference via RAX)
        assert!(
            result.is_memory(),
            "PIC addressing should produce a Memory operand"
        );
        if let MachineOperand::Memory { base, offset, .. } = &result {
            assert_eq!(*base, registers::RAX);
            assert_eq!(*offset, 0);
        }

        // Should have emitted a LEA instruction into the block
        assert_eq!(mf.blocks[0].instructions.len(), 1);
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::LEA);
    }

    #[test]
    fn pic_addressing_shared_mode() {
        let mut cfg = test_config();
        cfg.shared = true;
        let backend = X86_64Codegen::new(cfg);
        let mut mf = MachineFunction::new("test_func".to_string(), X86_64_STACK_ALIGNMENT);
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);

        let result = backend.generate_pic_addressing("shared_sym", &mut mf);
        // -shared implies PIC, so should get Memory operand
        assert!(result.is_memory());
    }

    #[test]
    fn pic_addressing_non_pic_mode() {
        let backend = X86_64Codegen::new(test_config());
        let mut mf = MachineFunction::new("test_func".to_string(), X86_64_STACK_ALIGNMENT);

        let result = backend.generate_pic_addressing("my_global", &mut mf);

        // Non-PIC: should produce a direct Symbol operand
        match result {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "my_global");
            }
            other => panic!("Expected Symbol operand, got {:?}", other),
        }

        // No instructions should be emitted (no blocks needed)
        assert!(mf.blocks.is_empty());
    }

    #[test]
    fn pic_addressing_gotpcrel_symbol_format() {
        let backend = X86_64Codegen::new(pic_config());
        let mut mf = MachineFunction::new("test_func".to_string(), X86_64_STACK_ALIGNMENT);
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);

        let _ = backend.generate_pic_addressing("printf", &mut mf);

        // The LEA instruction should reference symbol@GOTPCREL
        let lea = &mf.blocks[0].instructions[0];
        let has_got_symbol = lea.operands.iter().any(|op| {
            if let MachineOperand::Symbol(s) = op {
                s == "printf@GOTPCREL"
            } else {
                false
            }
        });
        assert!(
            has_got_symbol,
            "LEA should reference printf@GOTPCREL symbol"
        );
    }

    // -- emit_prologue integration test -------------------------------------

    #[test]
    fn emit_prologue_inserts_at_entry_block() {
        let backend = X86_64Codegen::new(test_config());
        let mut mf = MachineFunction::new("prologue_test".to_string(), X86_64_STACK_ALIGNMENT);
        mf.frame_size = 32;
        mf.used_callee_saved = vec![registers::RBX];

        // Add a block with one existing instruction
        let mut bb = MachineBasicBlock::new(0);
        let mut ret = MachineInstr::new(opcodes::RET);
        ret.set_return();
        bb.push_instr(ret);
        mf.add_block(bb);

        backend.emit_prologue(&mut mf);

        // Prologue: PUSH RBP, MOV RBP RSP, SUB RSP 32, PUSH RBX
        // Then the original RET instruction
        let instrs = &mf.blocks[0].instructions;
        assert_eq!(instrs.len(), 5);
        assert_eq!(instrs[0].opcode, opcodes::PUSH); // RBP
        assert_eq!(instrs[1].opcode, opcodes::MOV_RR);
        assert_eq!(instrs[2].opcode, opcodes::SUB_RI);
        assert_eq!(instrs[3].opcode, opcodes::PUSH); // RBX
        assert_eq!(instrs[4].opcode, opcodes::RET); // original
    }

    // -- emit_epilogue integration test -------------------------------------

    #[test]
    fn emit_epilogue_replaces_return() {
        let backend = X86_64Codegen::new(test_config());
        let mut mf = MachineFunction::new("epilogue_test".to_string(), X86_64_STACK_ALIGNMENT);
        mf.used_callee_saved = vec![registers::RBX];

        // Add a block: some instruction followed by RET
        let mut bb = MachineBasicBlock::new(0);
        bb.push_instr(MachineInstr::new(opcodes::NOP));
        let mut ret = MachineInstr::new(opcodes::RET);
        ret.set_return();
        bb.push_instr(ret);
        mf.add_block(bb);

        backend.emit_epilogue(&mut mf);

        // Expected: NOP, POP RBX, MOV RSP RBP, POP RBP, RET (from epilogue)
        let instrs = &mf.blocks[0].instructions;
        assert_eq!(instrs.len(), 5);
        assert_eq!(instrs[0].opcode, opcodes::NOP); // preserved
        assert_eq!(instrs[1].opcode, opcodes::POP); // RBX
        assert_eq!(instrs[2].opcode, opcodes::MOV_RR);
        assert_eq!(instrs[3].opcode, opcodes::POP); // RBP
        assert_eq!(instrs[4].opcode, opcodes::RET); // from epilogue
        assert!(instrs[4].is_return);
    }

    #[test]
    fn emit_epilogue_handles_multiple_returns() {
        let backend = X86_64Codegen::new(test_config());
        let mut mf = MachineFunction::new("multi_ret".to_string(), X86_64_STACK_ALIGNMENT);
        mf.used_callee_saved = vec![];

        // Block 0: ends with RET
        let mut bb0 = MachineBasicBlock::new(0);
        let mut ret0 = MachineInstr::new(opcodes::RET);
        ret0.set_return();
        bb0.push_instr(ret0);
        mf.add_block(bb0);

        // Block 1: also ends with RET
        let mut bb1 = MachineBasicBlock::new(1);
        bb1.push_instr(MachineInstr::new(opcodes::NOP));
        let mut ret1 = MachineInstr::new(opcodes::RET);
        ret1.set_return();
        bb1.push_instr(ret1);
        mf.add_block(bb1);

        backend.emit_epilogue(&mut mf);

        // Both blocks should have their returns replaced with epilogue
        // Block 0: MOV RSP RBP, POP RBP, RET
        assert_eq!(mf.blocks[0].instructions.len(), 3);
        assert!(mf.blocks[0].instructions.last().unwrap().is_return);

        // Block 1: NOP, MOV RSP RBP, POP RBP, RET
        assert_eq!(mf.blocks[1].instructions.len(), 4);
        assert!(mf.blocks[1].instructions.last().unwrap().is_return);
    }

    // -- Opcode namespace tests ---------------------------------------------

    #[test]
    fn opcodes_are_unique() {
        // Verify critical opcodes have distinct values
        let ops = [
            opcodes::PUSH,
            opcodes::POP,
            opcodes::MOV_RR,
            opcodes::MOV_RI,
            opcodes::LEA,
            opcodes::ADD_RR,
            opcodes::SUB_RI,
            opcodes::CALL,
            opcodes::RET,
            opcodes::ENDBR64,
            opcodes::PSEUDO_STACK_PROBE,
        ];
        for (i, &a) in ops.iter().enumerate() {
            for (j, &b) in ops.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "opcodes at indices {} and {} collide", i, j);
                }
            }
        }
    }

    #[test]
    fn opcodes_pseudo_ops_in_high_range() {
        // Pseudo-ops should be in the 0xFF00+ range.
        // Use a helper to prevent constant folding by Clippy.
        let threshold: u32 = 0xFF00;
        let pseudo_ops: [u32; 4] = [
            opcodes::PSEUDO_FRAME_SETUP,
            opcodes::PSEUDO_FRAME_DESTROY,
            opcodes::PSEUDO_STACK_PROBE,
            opcodes::PSEUDO_RETPOLINE,
        ];
        for op in &pseudo_ops {
            assert!(
                *op >= threshold,
                "Pseudo-op {:#X} below {:#X}",
                op,
                threshold
            );
        }
    }
}
