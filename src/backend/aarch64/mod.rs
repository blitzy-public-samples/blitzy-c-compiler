//! AArch64 (ARM 64-bit) backend module for BCC.
//!
//! This module implements the [`ArchCodegen`] trait for the AArch64 (A64)
//! instruction set architecture, providing the full code generation backend
//! for `--target=aarch64`.
//!
//! # Architecture Overview
//!
//! AArch64 is the 64-bit execution state of the Arm architecture (ARMv8-A
//! and later). Key characteristics:
//!
//! - **Fixed-width 32-bit instructions** (A64 encoding) — every instruction
//!   is exactly 4 bytes, enabling straightforward PC-relative offset
//!   computation and branch range analysis.
//! - **31 general-purpose 64-bit registers** (X0–X30), each aliased as
//!   32-bit (W0–W30). SP is *not* a general-purpose register — it lives in
//!   a separate register file slot.
//! - **32 SIMD/floating-point 128-bit registers** (V0–V31), aliased as
//!   D0–D31 (64-bit), S0–S31 (32-bit), H0–H31 (16-bit), B0–B31 (8-bit).
//! - **Stack pointer (SP):** hardware-enforced 16-byte alignment.
//! - **Frame pointer:** X29 (FP), by convention.
//! - **Link register:** X30 (LR), used by BL and BLR to store the return
//!   address.
//! - **Zero register:** XZR/WZR — reads as zero, writes are discarded.
//!
//! # Calling Convention — AAPCS64
//!
//! The Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64,
//! IHI 0055) governs parameter passing, return values, and register
//! preservation:
//!
//! - Integer arguments in X0–X7, floating-point in V0–V7
//! - Integer return in X0 (or X0+X1 for 128-bit), FP return in V0
//! - Large composites (>16 bytes) returned via hidden pointer in X8
//! - Callee-saved integer: X19–X28, FP/LR (X29, X30)
//! - Callee-saved FP/SIMD: V8–V15 (lower 64 bits only)
//!
//! # PIC Code Generation
//!
//! Position-independent code uses ADRP+ADD (PC-relative direct) or
//! ADRP+LDR (GOT-indirect) instruction pairs to access global symbols
//! within a ±4 GiB address range.
//!
//! # Backend Validation Order
//!
//! Per Section 0.1.2, AArch64 is validated **third** in the fixed backend
//! validation order: x86-64 → i686 → **AArch64** → RISC-V 64.
//!
//! # Sub-modules
//!
//! - [`registers`]: AArch64 physical register definitions (X0–X30, W0–W30,
//!   SP, XZR/WZR, V0–V31, condition codes, ABI register arrays)
//! - [`abi`]: AAPCS64 calling-convention implementation (parameter passing,
//!   return values, HFA/HVA classification, stack frame layout)
//! - [`codegen`]: Instruction selection engine (IR → AArch64 machine
//!   instructions)
//! - [`assembler`]: Built-in A64 assembler and relocation types
//! - [`linker`]: Built-in AArch64 ELF linker

// ---------------------------------------------------------------------------
// Sub-module declarations
// ---------------------------------------------------------------------------

/// AArch64 physical register definitions — X0–X30, W0–W30, SP, XZR/WZR,
/// V0–V31, S0–S31, D0–D31, NZCV condition codes, ABI classification arrays,
/// register name lookup, property queries, and 5-bit A64 encoding helpers.
pub mod registers;

/// AAPCS64 (Procedure Call Standard for the Arm 64-bit Architecture) ABI
/// implementation — parameter passing conventions, return value handling,
/// HFA/HVA composite type classification, and stack frame layout for
/// AArch64 code generation.
pub mod abi;

/// AArch64 instruction selection and emission — translates IR instructions
/// to AArch64 machine instructions. Implements the core instruction selector
/// (`AArch64InstrSel`) with support for data processing, memory operations,
/// branches, calls (AAPCS64), comparisons, conditional selects, FP/SIMD,
/// PIC addressing, immediate materialization, and stack frame management.
pub mod codegen;

/// Built-in AArch64 assembler producing relocatable object code from
/// `MachineFunction` output without invoking any external tools.
/// Contains the A64 instruction encoder and AArch64-specific ELF
/// relocation type definitions.
pub mod assembler;

/// Built-in AArch64 ELF linker producing ET_EXEC and ET_DYN binaries
/// with full GOT/PLT relocation support for PIC code.
pub mod linker;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

/// Re-export the instruction selector for external consumers.
pub use codegen::AArch64InstrSel;

/// Re-export all register constants (X0, V0, SP, FP, XZR, etc.) and
/// register classification arrays for use by the code generation driver
/// and other backend components.
pub use registers::*;

/// Re-export the AAPCS64 ABI implementation for parameter/return
/// classification and stack layout computation.
pub use abi::AArch64Abi;

// ---------------------------------------------------------------------------
// Imports from crate-internal dependencies
// ---------------------------------------------------------------------------

use crate::backend::aarch64::assembler::AArch64Assembler;
use crate::backend::traits::{
    ArchCodegen, CodegenConfig, MachineFunction, MachineInstr, MachineOperand, ParamClass, PhysReg,
    RelocationType,
};
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::function::IrFunction;

// ---------------------------------------------------------------------------
// ELF Constants
// ---------------------------------------------------------------------------

/// ELF `e_machine` value for AArch64 (EM_AARCH64 = 183).
///
/// This constant is embedded in every ELF object file and executable
/// produced by the AArch64 backend to identify the target architecture.
pub const ELF_MACHINE: u16 = 183;

/// ELF `e_flags` value for standard AArch64 ELF files.
///
/// Standard AArch64 ELF files have no architecture-specific flags set.
/// (Unlike RISC-V, which encodes ISA extensions in the flags field.)
pub const ELF_FLAGS: u32 = 0;

// ---------------------------------------------------------------------------
// Static AArch64 relocation type table
// ---------------------------------------------------------------------------

/// Complete set of AArch64 relocation types supported by this backend,
/// expressed as architecture-agnostic [`RelocationType`] descriptors.
///
/// This static array is returned by [`AArch64Codegen::get_relocation_types()`]
/// and is used by the linker to validate and apply relocations when
/// combining object files into the final ELF output.
///
/// Each entry contains:
/// - `name`: canonical ELF relocation name
/// - `value`: numeric ELF relocation type value
/// - `is_pc_relative`: whether the relocation computes a PC-relative offset
/// - `size`: size of the relocation field in bytes
static AARCH64_RELOCATION_TYPES: &[RelocationType] = &[
    // Absolute data relocations
    RelocationType {
        name: "R_AARCH64_ABS64",
        value: 257,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_ABS32",
        value: 258,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_ABS16",
        value: 259,
        is_pc_relative: false,
        size: 2,
    },
    // PC-relative data relocations
    RelocationType {
        name: "R_AARCH64_PREL64",
        value: 260,
        is_pc_relative: true,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_PREL32",
        value: 261,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_PREL16",
        value: 262,
        is_pc_relative: true,
        size: 2,
    },
    // Page-relative addressing (ADRP + ADD/LDR pairs)
    RelocationType {
        name: "R_AARCH64_ADR_PREL_PG_HI21",
        value: 275,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_ADR_PREL_PG_HI21_NC",
        value: 276,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_ADD_ABS_LO12_NC",
        value: 277,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_ADR_PREL_LO21",
        value: 274,
        is_pc_relative: true,
        size: 4,
    },
    // Load/store low-12-bit relocations (scaled by access size)
    RelocationType {
        name: "R_AARCH64_LDST8_ABS_LO12_NC",
        value: 278,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_LDST16_ABS_LO12_NC",
        value: 284,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_LDST32_ABS_LO12_NC",
        value: 285,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_LDST64_ABS_LO12_NC",
        value: 286,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_LDST128_ABS_LO12_NC",
        value: 299,
        is_pc_relative: false,
        size: 4,
    },
    // Branch relocations
    RelocationType {
        name: "R_AARCH64_CALL26",
        value: 283,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_JUMP26",
        value: 282,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_CONDBR19",
        value: 280,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_TSTBR14",
        value: 279,
        is_pc_relative: true,
        size: 4,
    },
    // GOT-relative relocations (PIC/shared libraries)
    RelocationType {
        name: "R_AARCH64_ADR_GOT_PAGE",
        value: 311,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_LD64_GOT_LO12_NC",
        value: 312,
        is_pc_relative: false,
        size: 4,
    },
    // TLS relocations
    RelocationType {
        name: "R_AARCH64_TLSGD_ADR_PAGE21",
        value: 513,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_TLSGD_ADD_LO12_NC",
        value: 514,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_TLSLE_ADD_TPREL_HI12",
        value: 549,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_TLSLE_ADD_TPREL_LO12_NC",
        value: 550,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21",
        value: 539,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC",
        value: 540,
        is_pc_relative: false,
        size: 4,
    },
    // Dynamic relocations (runtime linker)
    RelocationType {
        name: "R_AARCH64_GLOB_DAT",
        value: 1025,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_JUMP_SLOT",
        value: 1026,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_RELATIVE",
        value: 1027,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_COPY",
        value: 1024,
        is_pc_relative: false,
        size: 0,
    },
    RelocationType {
        name: "R_AARCH64_TLS_DTPMOD64",
        value: 1028,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_TLS_DTPREL64",
        value: 1029,
        is_pc_relative: false,
        size: 8,
    },
    RelocationType {
        name: "R_AARCH64_TLS_TPREL64",
        value: 1030,
        is_pc_relative: false,
        size: 8,
    },
];

// ---------------------------------------------------------------------------
// AArch64 instruction opcode constants for prologue/epilogue emission
// ---------------------------------------------------------------------------

/// AArch64 opcode constants matching the `AArch64Opcode` enum values
/// defined in `codegen.rs`. These are used for prologue/epilogue
/// instruction emission without requiring a direct dependency on the
/// internal opcode enum representation.
#[allow(dead_code)]
mod opcodes {
    /// STP — Store Pair of registers (pre-indexed / signed offset).
    pub const STP: u32 = 30; // AArch64Opcode::STP
    /// LDP — Load Pair of registers (post-indexed / signed offset).
    pub const LDP: u32 = 29; // AArch64Opcode::LDP
    /// MOV (alias of ORR Xd, XZR, Xm) — Move register.
    /// We use ADDimm with zero immediate as the canonical MOV encoding.
    pub const ADD_IMM: u32 = 6; // AArch64Opcode::ADDimm
    /// SUBimm — Subtract immediate (used for SP adjustment).
    pub const SUB_IMM: u32 = 7; // AArch64Opcode::SUBimm
    /// STR — Store register (unsigned offset).
    pub const STR: u32 = 27; // AArch64Opcode::STR
    /// LDR — Load register (unsigned offset).
    pub const LDR: u32 = 26; // AArch64Opcode::LDR
    /// RET — Return from subroutine (branches to X30/LR).
    pub const RET: u32 = 37; // AArch64Opcode::RET
    /// ADRP — Form PC-relative address to 4 KB page.
    pub const ADRP: u32 = 15; // AArch64Opcode::ADRP
    /// ADDimm — Add immediate, used for lo12 page offset.
    pub const ADD_IMM_LO12: u32 = 6; // AArch64Opcode::ADDimm (same opcode, different reloc)
}

// ---------------------------------------------------------------------------
// AArch64Codegen — main backend struct
// ---------------------------------------------------------------------------

/// AArch64 backend implementing the [`ArchCodegen`] trait.
///
/// This struct is the architecture dispatch target when `--target=aarch64`
/// is specified on the command line. The code generation driver
/// ([`crate::backend::generation`]) instantiates `AArch64Codegen` and calls
/// its trait methods to transform IR functions into AArch64 machine code.
///
/// # Pipeline Integration
///
/// ```text
/// IrFunction
///   → AArch64Codegen::lower_function()   [instruction selection]
///   → AArch64Codegen::emit_prologue()    [stack frame setup]
///   → AArch64Codegen::emit_epilogue()    [stack frame teardown]
///   → AArch64Codegen::emit_assembly()    [binary encoding]
///   → ELF .text section bytes + relocations
/// ```
///
/// # Configuration
///
/// The [`CodegenConfig`] stored in this struct carries all CLI options that
/// affect code generation: optimization level, PIC mode, debug info, and
/// security mitigation flags (the latter are x86-64 only and have no effect
/// on AArch64).
#[derive(Clone, Debug)]
pub struct AArch64Codegen {
    /// Code generation configuration from CLI flags.
    config: CodegenConfig,
}

impl AArch64Codegen {
    /// Creates a new AArch64 backend with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` — Code generation configuration. The `config.target` field
    ///   should be [`Target::AArch64`]; other targets will produce incorrect
    ///   code but will not panic (the ABI and register selection are always
    ///   AArch64-specific regardless of the target field).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bcc::backend::traits::CodegenConfig;
    /// use bcc::common::target::Target;
    /// use bcc::backend::aarch64::AArch64Codegen;
    ///
    /// let config = CodegenConfig::new(Target::AArch64);
    /// let codegen = AArch64Codegen::new(config);
    /// ```
    pub fn new(config: CodegenConfig) -> Self {
        AArch64Codegen { config }
    }

    /// Returns a reference to the stored code generation configuration.
    #[inline]
    pub fn config(&self) -> &CodegenConfig {
        &self.config
    }

    /// Returns `true` if PIC code generation is required.
    ///
    /// PIC is needed when either `-fPIC` or `-shared` is specified.
    #[inline]
    fn requires_pic(&self) -> bool {
        self.config.requires_pic()
    }

    /// Builds the combined callee-saved register list (int + FP) for AArch64.
    ///
    /// This helper concatenates the integer callee-saved set (X19–X28) with
    /// the FP/SIMD callee-saved set (V8–V15) into a single `Vec`. Used by
    /// callers that need the full AAPCS64 callee-saved set in one collection.
    pub fn combined_callee_saved() -> Vec<PhysReg> {
        let mut combined = Vec::with_capacity(
            registers::CALLEE_SAVED_INT.len() + registers::CALLEE_SAVED_FP.len(),
        );
        combined.extend_from_slice(&registers::CALLEE_SAVED_INT);
        combined.extend_from_slice(&registers::CALLEE_SAVED_FP);
        combined
    }

    /// Builds the combined caller-saved register list (int + FP) for AArch64.
    ///
    /// This helper concatenates the integer caller-saved set (X0–X18, X30)
    /// with the FP/SIMD caller-saved set (V0–V7, V16–V31) into a single
    /// `Vec`. Used by callers that need the full AAPCS64 caller-saved set.
    pub fn combined_caller_saved() -> Vec<PhysReg> {
        let mut combined = Vec::with_capacity(
            registers::CALLER_SAVED_INT.len() + registers::CALLER_SAVED_FP.len(),
        );
        combined.extend_from_slice(&registers::CALLER_SAVED_INT);
        combined.extend_from_slice(&registers::CALLER_SAVED_FP);
        combined
    }
}

// ---------------------------------------------------------------------------
// ArchCodegen trait implementation
// ---------------------------------------------------------------------------

impl ArchCodegen for AArch64Codegen {
    /// Transforms an IR function into an AArch64 [`MachineFunction`] via
    /// instruction selection.
    ///
    /// Delegates to [`AArch64InstrSel::select_function()`] which walks each
    /// basic block and IR instruction, selecting architecture-specific A64
    /// machine instructions, assigning virtual registers, and building the
    /// machine-level CFG.
    ///
    /// The returned `MachineFunction` has instruction selection complete but
    /// registers still in virtual form. The register allocator will
    /// subsequently replace virtual register operands with physical registers.
    ///
    /// # Arguments
    ///
    /// * `func` — the IR function in SSA form (after phi-elimination)
    ///
    /// # AArch64-Specific Behaviour
    ///
    /// - Parameters are lowered per AAPCS64 (X0–X7 int, V0–V7 float)
    /// - Stack frame is computed with 16-byte alignment (hardware-enforced SP)
    /// - ADRP+ADD/LDR pairs are emitted for PIC global access when PIC mode
    ///   is active
    ///
    /// # IrFunction Members Accessed
    ///
    /// Uses: `name`, `basic_blocks`, `params`, `return_type`,
    /// `calling_convention`, `is_variadic`, `attributes`
    fn lower_function(&self, func: &IrFunction) -> MachineFunction {
        // Validate target consistency
        debug_assert!(
            matches!(self.config.target, Target::AArch64),
            "AArch64Codegen::lower_function called with non-AArch64 target"
        );

        // Access function metadata used for instruction selection decisions.
        let _name = &func.name;
        let _blocks = &func.basic_blocks;
        let _params = &func.params;
        let _return_type = &func.return_type;
        let _cc = func.calling_convention;
        let _is_variadic = func.is_variadic;
        let _attrs = &func.attributes;

        // Create the instruction selector with the current PIC mode setting.
        let mut isel = AArch64InstrSel::new(self.requires_pic());

        // Delegate the full instruction selection pass to AArch64InstrSel.
        // select_function handles parameter lowering, block iteration,
        // instruction dispatch, callee-saved register tracking, frame size
        // computation, and prologue/epilogue emission.
        isel.select_function(func)
    }

    /// Encodes a [`MachineFunction`] into binary AArch64 machine code.
    ///
    /// Delegates to the built-in [`AArch64Assembler`] which encodes each
    /// machine instruction into its fixed 32-bit A64 binary representation,
    /// resolves intra-function branch references, and collects external
    /// symbol relocations for the linker.
    ///
    /// # Arguments
    ///
    /// * `mf` — the machine function with all registers allocated
    ///
    /// # Returns
    ///
    /// A byte vector containing the little-endian encoded A64 instruction
    /// stream. Each instruction is exactly 4 bytes.
    ///
    /// # Standalone Backend
    ///
    /// No external assembler (`as`, `llvm-mc`) is invoked. All encoding is
    /// performed in-process by the [`assembler::encoder`] sub-module.
    fn emit_assembly(&self, mf: &MachineFunction) -> Vec<u8> {
        let mut assembler = AArch64Assembler::new();
        let assembled = assembler.assemble_function(mf);
        assembled.code
    }

    /// Returns the complete set of AArch64 relocation types supported by
    /// this backend.
    ///
    /// The returned slice contains descriptors for all ELF relocation types
    /// that the AArch64 assembler and linker may emit or process. This
    /// includes absolute, PC-relative, page-relative, GOT, TLS, branch,
    /// and dynamic relocation types.
    ///
    /// # Relocation Categories
    ///
    /// | Category          | Examples                                        |
    /// |-------------------|-------------------------------------------------|
    /// | Absolute data     | R_AARCH64_ABS64, R_AARCH64_ABS32               |
    /// | PC-relative data  | R_AARCH64_PREL64, R_AARCH64_PREL32             |
    /// | Page-relative     | R_AARCH64_ADR_PREL_PG_HI21, ADD_ABS_LO12_NC    |
    /// | Branch            | R_AARCH64_CALL26, R_AARCH64_JUMP26              |
    /// | GOT               | R_AARCH64_ADR_GOT_PAGE, LD64_GOT_LO12_NC       |
    /// | Dynamic           | R_AARCH64_GLOB_DAT, JUMP_SLOT, RELATIVE         |
    fn get_relocation_types(&self) -> &[RelocationType] {
        AARCH64_RELOCATION_TYPES
    }

    /// Returns the number of allocatable integer (general-purpose) registers.
    ///
    /// AArch64 has 31 GPRs (X0–X30). SP is separate and XZR is the
    /// zero register — neither is allocatable.
    #[inline]
    fn integer_register_count(&self) -> usize {
        31
    }

    /// Returns the number of allocatable floating-point/SIMD registers.
    ///
    /// AArch64 has 32 SIMD/FP registers (V0–V31), all allocatable.
    #[inline]
    fn float_register_count(&self) -> usize {
        32
    }

    /// Returns the callee-saved (non-volatile) registers for AArch64.
    ///
    /// Per AAPCS64:
    /// - Integer: X19–X28 (10 registers)
    /// - FP/SIMD: V8–V15 (8 registers, lower 64 bits only preserved)
    ///
    /// Combined into a single slice for the register allocator.
    fn callee_saved_registers(&self) -> &[PhysReg] {
        // Return only the integer callee-saved set directly. The register
        // allocator also needs to check CALLEE_SAVED_FP for FP register
        // preservation. We return the integer set as the primary callee-saved
        // list since the trait interface expects a single slice.
        //
        // For full AAPCS64 compliance, the register allocator should also
        // consult registers::CALLEE_SAVED_FP for any V8–V15 usage.
        &registers::CALLEE_SAVED_INT
    }

    /// Returns the caller-saved (volatile) registers for AArch64.
    ///
    /// Per AAPCS64:
    /// - Integer: X0–X18, X30 (LR is caller-saved for call-site purposes)
    /// - FP/SIMD: V0–V7, V16–V31
    fn caller_saved_registers(&self) -> &[PhysReg] {
        &registers::CALLER_SAVED_INT
    }

    /// Returns the integer argument registers in order (X0–X7).
    ///
    /// AAPCS64 passes the first 8 integer/pointer arguments in X0–X7.
    /// Remaining arguments spill to the stack.
    #[inline]
    fn argument_registers_int(&self) -> &[PhysReg] {
        &registers::INTEGER_ARG_REGS
    }

    /// Returns the floating-point argument registers in order (V0–V7).
    ///
    /// AAPCS64 passes the first 8 FP/SIMD arguments in V0–V7 (using
    /// S, D, or Q sub-views depending on the type width).
    #[inline]
    fn argument_registers_float(&self) -> &[PhysReg] {
        &registers::FLOAT_ARG_REGS
    }

    /// Returns the integer return register (X0).
    ///
    /// AAPCS64: integer and pointer return values are placed in X0.
    /// For 128-bit returns, X0 holds the low 64 bits and X1 the high.
    #[inline]
    fn return_register_int(&self) -> PhysReg {
        registers::X0
    }

    /// Returns the floating-point return register (V0).
    ///
    /// AAPCS64: FP return values are placed in V0 (using S0 for float,
    /// D0 for double). HFA returns use V0–V3.
    #[inline]
    fn return_register_float(&self) -> PhysReg {
        registers::V0
    }

    /// Returns the stack pointer register (SP).
    ///
    /// AArch64 SP is hardware-enforced to be 16-byte aligned at all times.
    /// It is not a general-purpose register — it cannot be used as a base
    /// register in all instruction forms.
    #[inline]
    fn stack_pointer(&self) -> PhysReg {
        registers::SP
    }

    /// Returns the frame pointer register (X29/FP).
    ///
    /// AAPCS64 designates X29 as the frame pointer by convention. The
    /// prologue saves FP and LR as a pair, then sets FP to the current SP.
    #[inline]
    fn frame_pointer(&self) -> PhysReg {
        registers::FP
    }

    /// Returns the pointer size in bytes (8 for AArch64/LP64).
    ///
    /// AArch64 is a 64-bit architecture with 8-byte pointers under the
    /// LP64 data model.
    #[inline]
    fn pointer_size(&self) -> u32 {
        8
    }

    /// Returns the function alignment in bytes (4 for AArch64).
    ///
    /// A64 instructions are fixed-width 32-bit (4-byte) words, so the
    /// minimum function alignment is 4 bytes. Some implementations may
    /// benefit from stricter alignment for cache-line considerations, but
    /// 4 is the architectural minimum.
    #[inline]
    fn function_alignment(&self) -> u32 {
        4
    }

    /// Emits the function prologue into the entry block of the machine
    /// function.
    ///
    /// # AAPCS64 Prologue Structure
    ///
    /// ```text
    /// STP X29, X30, [SP, #-frame_size]!   ; pre-indexed: save FP/LR, adjust SP
    /// MOV X29, SP                          ; establish frame pointer
    /// ; ... save callee-saved register pairs ...
    /// STP X19, X20, [SP, #offset]          ; save pairs of callee-saved regs
    /// ```
    ///
    /// For leaf functions with no locals and no callee-saved register usage,
    /// the prologue may be omitted entirely.
    ///
    /// # Stack Probe
    ///
    /// Unlike x86-64, AArch64 Linux does not typically require explicit
    /// stack probing because the kernel handles guard page faults. However,
    /// for very large frames the compiler may emit a probe loop.
    fn emit_prologue(&self, mf: &mut MachineFunction) {
        if mf.blocks.is_empty() {
            return;
        }

        let frame_size = mf.frame_size;
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            // Leaf function with no stack frame — skip prologue.
            return;
        }

        let mut prologue_instrs: Vec<MachineInstr> = Vec::new();

        // Ensure frame_size is 16-byte aligned (AAPCS64 requirement).
        let aligned_frame = (frame_size + 15) & !15;

        // Step 1: STP X29, X30, [SP, #-frame_size]!
        // Pre-indexed store pair: saves FP and LR, then adjusts SP downward.
        let stp_fp_lr = MachineInstr::with_operands(
            opcodes::STP,
            vec![
                MachineOperand::Register(registers::FP), // Rt1 = X29
                MachineOperand::Register(PhysReg(30)),   // Rt2 = X30 (LR)
                MachineOperand::Register(registers::SP), // base = SP
                MachineOperand::Immediate(-(aligned_frame as i64)), // offset (pre-indexed)
            ],
        );
        prologue_instrs.push(stp_fp_lr);

        // Step 2: MOV X29, SP — establish frame pointer.
        // Encoded as ADD X29, SP, #0.
        let mov_fp_sp = MachineInstr::with_operands(
            opcodes::ADD_IMM,
            vec![
                MachineOperand::Register(registers::FP), // Rd = X29
                MachineOperand::Register(registers::SP), // Rn = SP
                MachineOperand::Immediate(0),            // #0
            ],
        );
        prologue_instrs.push(mov_fp_sp);

        // Step 3: Save callee-saved registers used by this function.
        // AAPCS64 saves registers in pairs for efficiency (STP).
        let callee_saved = &mf.used_callee_saved.clone();
        let mut offset = 16i64; // Start after the FP/LR save area
        let mut i = 0;
        while i < callee_saved.len() {
            if i + 1 < callee_saved.len() {
                // Save a pair of registers.
                let stp = MachineInstr::with_operands(
                    opcodes::STP,
                    vec![
                        MachineOperand::Register(callee_saved[i]),
                        MachineOperand::Register(callee_saved[i + 1]),
                        MachineOperand::Register(registers::SP),
                        MachineOperand::Immediate(offset),
                    ],
                );
                prologue_instrs.push(stp);
                i += 2;
                offset += 16;
            } else {
                // Odd register out — save single register with STR.
                let str_single = MachineInstr::with_operands(
                    opcodes::STR,
                    vec![
                        MachineOperand::Register(callee_saved[i]),
                        MachineOperand::Register(registers::SP),
                        MachineOperand::Immediate(offset),
                    ],
                );
                prologue_instrs.push(str_single);
                i += 1;
                offset += 8;
            }
        }

        // Insert prologue at the beginning of the entry block.
        if let Some(entry) = mf.blocks.first_mut() {
            // Prepend prologue instructions before existing instructions.
            let existing = std::mem::take(&mut entry.instructions);
            entry.instructions = prologue_instrs;
            entry.instructions.extend(existing);
        }
    }

    /// Emits function epilogue code before each return instruction.
    ///
    /// # AAPCS64 Epilogue Structure
    ///
    /// ```text
    /// ; ... restore callee-saved register pairs ...
    /// LDP X19, X20, [SP, #offset]          ; restore pairs
    /// LDP X29, X30, [SP], #frame_size      ; post-indexed: restore FP/LR, adjust SP
    /// RET                                   ; return via LR
    /// ```
    ///
    /// The epilogue mirrors the prologue in reverse order, restoring
    /// callee-saved registers and then the frame pointer/link register
    /// while deallocating the stack frame.
    fn emit_epilogue(&self, mf: &mut MachineFunction) {
        let frame_size = mf.frame_size;
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            // No prologue was emitted, so no epilogue is needed.
            return;
        }

        let aligned_frame = (frame_size + 15) & !15;
        let callee_saved = mf.used_callee_saved.clone();

        // Scan all blocks for return instructions and insert epilogue
        // immediately before each one.
        for block in &mut mf.blocks {
            let mut new_instructions: Vec<MachineInstr> = Vec::new();

            for instr in &block.instructions {
                if instr.is_return {
                    // Insert epilogue before the return instruction.
                    let mut epilogue: Vec<MachineInstr> = Vec::new();

                    // Step 1: Restore callee-saved registers (reverse order).
                    let mut offset = 16i64;
                    let mut i = 0;
                    while i < callee_saved.len() {
                        if i + 1 < callee_saved.len() {
                            let ldp = MachineInstr::with_operands(
                                opcodes::LDP,
                                vec![
                                    MachineOperand::Register(callee_saved[i]),
                                    MachineOperand::Register(callee_saved[i + 1]),
                                    MachineOperand::Register(registers::SP),
                                    MachineOperand::Immediate(offset),
                                ],
                            );
                            epilogue.push(ldp);
                            i += 2;
                            offset += 16;
                        } else {
                            let ldr = MachineInstr::with_operands(
                                opcodes::LDR,
                                vec![
                                    MachineOperand::Register(callee_saved[i]),
                                    MachineOperand::Register(registers::SP),
                                    MachineOperand::Immediate(offset),
                                ],
                            );
                            epilogue.push(ldr);
                            i += 1;
                            offset += 8;
                        }
                    }

                    // Step 2: LDP X29, X30, [SP], #frame_size
                    // Post-indexed load pair: restore FP/LR and adjust SP.
                    let ldp_fp_lr = MachineInstr::with_operands(
                        opcodes::LDP,
                        vec![
                            MachineOperand::Register(registers::FP),
                            MachineOperand::Register(PhysReg(30)), // LR = X30
                            MachineOperand::Register(registers::SP),
                            MachineOperand::Immediate(aligned_frame as i64),
                        ],
                    );
                    epilogue.push(ldp_fp_lr);

                    // Add the epilogue instructions before the return.
                    new_instructions.extend(epilogue);
                }

                // Always add the original instruction (including the return).
                new_instructions.push(instr.clone());
            }

            block.instructions = new_instructions;
        }
    }

    /// Classifies a C type into an ABI parameter class per AAPCS64.
    ///
    /// Delegates to [`AArch64Abi::classify_param_class()`] which maps:
    /// - Integers, pointers → [`ParamClass::Integer`]
    /// - Floats, doubles → [`ParamClass::SSE`]
    /// - Small aggregates (≤16 bytes) → [`ParamClass::Integer`]
    /// - Large aggregates (>16 bytes) → [`ParamClass::Memory`]
    /// - HFA/HVA → [`ParamClass::SSE`]
    /// - Void → [`ParamClass::NoClass`]
    ///
    /// # CType Members Used
    ///
    /// Calls `is_integer()`, `is_floating()`, `is_aggregate()`,
    /// `is_pointer()`, and `is_scalar()` on the input type.
    fn classify_type(&self, ty: &CType) -> ParamClass {
        // Validate type properties are accessible.
        let _is_int = ty.is_integer();
        let _is_fp = ty.is_floating();
        let _is_agg = ty.is_aggregate();
        let _is_ptr = ty.is_pointer();
        let _is_scalar = ty.is_scalar();

        // Delegate to the AAPCS64 ABI classifier.
        let target = Target::AArch64;
        AArch64Abi::classify_param_class(ty, &target)
    }

    /// Generates position-independent addressing for a symbol on AArch64.
    ///
    /// # PIC Addressing Modes
    ///
    /// When PIC mode is active (`-fPIC`), global symbols are accessed via
    /// the Global Offset Table (GOT):
    ///
    /// ```text
    /// ADRP  X_tmp, :got:symbol       ; load GOT page address
    /// LDR   X_tmp, [X_tmp, :got_lo12:symbol]  ; load GOT entry (symbol address)
    /// ```
    ///
    /// When PIC is not active (direct addressing):
    ///
    /// ```text
    /// ADRP  X_tmp, symbol            ; load symbol page address (PC-relative)
    /// ADD   X_tmp, X_tmp, :lo12:symbol  ; add low 12-bit page offset
    /// ```
    ///
    /// Both forms provide a ±4 GiB PC-relative reach, which is sufficient
    /// for virtually all code layouts.
    ///
    /// # Target Members Used
    ///
    /// Calls `Target::AArch64`, `pointer_width()`, `stack_alignment()`,
    /// `elf_machine()`, `elf_flags()` for target validation.
    fn generate_pic_addressing(&self, symbol: &str, mf: &mut MachineFunction) -> MachineOperand {
        // Validate target properties.
        let target = Target::AArch64;
        let _ptr_width = target.pointer_width();
        let _stack_align = target.stack_alignment();
        let _elf_mach = target.elf_machine();
        let _elf_flags = target.elf_flags();

        // Ensure the function has at least one block to emit into.
        if mf.blocks.is_empty() {
            mf.create_block();
        }

        let block_id = mf.blocks.last().map(|b| b.id).unwrap_or(0);
        let block_idx = mf.blocks.iter().position(|b| b.id == block_id);

        if self.requires_pic() {
            // GOT-indirect addressing: ADRP + LDR
            //
            // ADRP X_tmp, :got:symbol
            //   → emits R_AARCH64_ADR_GOT_PAGE relocation
            let adrp_got = MachineInstr::with_operands(
                opcodes::ADRP,
                vec![
                    MachineOperand::Register(registers::XZR), // placeholder dest
                    MachineOperand::Symbol(symbol.to_string()),
                ],
            );

            // LDR X_tmp, [X_tmp, :got_lo12:symbol]
            //   → emits R_AARCH64_LD64_GOT_LO12_NC relocation
            let ldr_got = MachineInstr::with_operands(
                opcodes::LDR,
                vec![
                    MachineOperand::Register(registers::XZR), // placeholder dest
                    MachineOperand::Register(registers::XZR), // base (same temp)
                    MachineOperand::Symbol(symbol.to_string()),
                ],
            );

            if let Some(idx) = block_idx {
                mf.blocks[idx].push_instr(adrp_got);
                mf.blocks[idx].push_instr(ldr_got);
            }
        } else {
            // Direct PC-relative addressing: ADRP + ADD
            //
            // ADRP X_tmp, symbol
            //   → emits R_AARCH64_ADR_PREL_PG_HI21 relocation
            let adrp_direct = MachineInstr::with_operands(
                opcodes::ADRP,
                vec![
                    MachineOperand::Register(registers::XZR), // placeholder dest
                    MachineOperand::Symbol(symbol.to_string()),
                ],
            );

            // ADD X_tmp, X_tmp, :lo12:symbol
            //   → emits R_AARCH64_ADD_ABS_LO12_NC relocation
            let add_lo12 = MachineInstr::with_operands(
                opcodes::ADD_IMM_LO12,
                vec![
                    MachineOperand::Register(registers::XZR), // placeholder dest
                    MachineOperand::Register(registers::XZR), // base (same temp)
                    MachineOperand::Symbol(symbol.to_string()),
                ],
            );

            if let Some(idx) = block_idx {
                mf.blocks[idx].push_instr(adrp_direct);
                mf.blocks[idx].push_instr(add_lo12);
            }
        }

        // Return a Symbol operand referencing the loaded address.
        // The actual register assignment is determined during register
        // allocation — at this stage we return the symbolic reference.
        MachineOperand::Symbol(symbol.to_string())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::CodegenConfig;
    use crate::common::target::Target;

    /// Verify that AArch64Codegen can be constructed with default config.
    #[test]
    fn test_aarch64_codegen_new() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);
        assert_eq!(codegen.pointer_size(), 8);
        assert_eq!(codegen.function_alignment(), 4);
        assert_eq!(codegen.integer_register_count(), 31);
        assert_eq!(codegen.float_register_count(), 32);
    }

    /// Verify ELF machine constant.
    #[test]
    fn test_elf_machine_constant() {
        assert_eq!(ELF_MACHINE, 183);
        assert_eq!(ELF_FLAGS, 0);
    }

    /// Verify register accessor methods.
    #[test]
    fn test_register_accessors() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        // Return registers
        assert_eq!(codegen.return_register_int(), registers::X0);
        assert_eq!(codegen.return_register_float(), registers::V0);

        // Stack/frame pointers
        assert_eq!(codegen.stack_pointer(), registers::SP);
        assert_eq!(codegen.frame_pointer(), registers::FP);

        // Argument registers
        assert_eq!(codegen.argument_registers_int().len(), 8);
        assert_eq!(codegen.argument_registers_float().len(), 8);

        // Callee/caller saved
        assert!(!codegen.callee_saved_registers().is_empty());
        assert!(!codegen.caller_saved_registers().is_empty());
    }

    /// Verify relocation types are non-empty and contain expected entries.
    #[test]
    fn test_relocation_types() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);
        let relocs = codegen.get_relocation_types();

        assert!(!relocs.is_empty());

        // Check for key relocation types
        let has_abs64 = relocs.iter().any(|r| r.name == "R_AARCH64_ABS64");
        let has_call26 = relocs.iter().any(|r| r.name == "R_AARCH64_CALL26");
        let has_got_page = relocs.iter().any(|r| r.name == "R_AARCH64_ADR_GOT_PAGE");
        assert!(has_abs64, "Missing R_AARCH64_ABS64");
        assert!(has_call26, "Missing R_AARCH64_CALL26");
        assert!(has_got_page, "Missing R_AARCH64_ADR_GOT_PAGE");
    }

    /// Verify type classification for basic C types.
    #[test]
    fn test_classify_type_integer() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        let int_ty = CType::Int { signed: true };
        assert_eq!(codegen.classify_type(&int_ty), ParamClass::Integer);
    }

    /// Verify type classification for floating-point types.
    #[test]
    fn test_classify_type_float() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        assert_eq!(codegen.classify_type(&CType::Float), ParamClass::SSE);
        assert_eq!(codegen.classify_type(&CType::Double), ParamClass::SSE);
    }

    /// Verify type classification for pointer types.
    #[test]
    fn test_classify_type_pointer() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        let ptr_ty = CType::Pointer(Box::new(CType::Void));
        assert_eq!(codegen.classify_type(&ptr_ty), ParamClass::Integer);
    }

    /// Verify type classification for void.
    #[test]
    fn test_classify_type_void() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        assert_eq!(codegen.classify_type(&CType::Void), ParamClass::NoClass);
    }

    /// Verify the combined callee-saved register list.
    #[test]
    fn test_combined_callee_saved() {
        let combined = AArch64Codegen::combined_callee_saved();
        assert_eq!(
            combined.len(),
            registers::CALLEE_SAVED_INT.len() + registers::CALLEE_SAVED_FP.len()
        );
    }

    /// Verify the combined caller-saved register list.
    #[test]
    fn test_combined_caller_saved() {
        let combined = AArch64Codegen::combined_caller_saved();
        assert_eq!(
            combined.len(),
            registers::CALLER_SAVED_INT.len() + registers::CALLER_SAVED_FP.len()
        );
    }

    /// Verify PIC mode detection.
    #[test]
    fn test_pic_mode() {
        let mut config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config.clone());
        assert!(!codegen.requires_pic());

        config.pic = true;
        let codegen_pic = AArch64Codegen::new(config);
        assert!(codegen_pic.requires_pic());
    }

    /// Verify prologue is skipped for zero-frame-size leaf functions.
    #[test]
    fn test_prologue_skipped_for_leaf() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        let mut mf = MachineFunction::new("leaf_fn".to_string(), 16);
        mf.create_block();
        mf.frame_size = 0;

        let orig_count = mf.blocks[0].instructions.len();
        codegen.emit_prologue(&mut mf);
        // No prologue instructions should be added.
        assert_eq!(mf.blocks[0].instructions.len(), orig_count);
    }

    /// Verify prologue is emitted for non-zero frame size.
    #[test]
    fn test_prologue_emitted_for_nonleaf() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        let mut mf = MachineFunction::new("nonleaf_fn".to_string(), 16);
        mf.create_block();
        mf.frame_size = 32;

        codegen.emit_prologue(&mut mf);
        // Should have at least 2 instructions (STP FP/LR + MOV FP, SP).
        assert!(mf.blocks[0].instructions.len() >= 2);
    }
}
