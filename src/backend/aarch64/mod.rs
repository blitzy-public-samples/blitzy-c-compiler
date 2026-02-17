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

    /// Assembles a machine function into raw bytes **and** collects
    /// relocations for external symbol references (e.g. `BL printf`).
    ///
    /// The default trait implementation discards relocations, which is
    /// incorrect for the AArch64 backend where `BL <symbol>` instructions
    /// require `R_AARCH64_CALL26` relocations to be applied by the linker.
    fn emit_assembly_with_relocations(
        &self,
        mf: &MachineFunction,
    ) -> crate::backend::traits::AssemblyOutput {
        let mut assembler = AArch64Assembler::new();
        let assembled = assembler.assemble_function(mf);
        crate::backend::traits::AssemblyOutput {
            code: assembled.code,
            relocations: assembled
                .relocations
                .into_iter()
                .map(|r| crate::backend::traits::AsmRelocation {
                    offset: r.offset as usize,
                    symbol: r.symbol,
                    reloc_type: r.reloc_type.to_elf_value(),
                    addend: r.addend,
                })
                .collect(),
        }
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
        // Return the combined integer + FP callee-saved set.  The register
        // allocator's `float_set()` filters this by register number range
        // to build the FP pool.  Omitting FP registers here leaves the
        // allocator with an empty FP pool and causes operand clobbering.
        //
        // Integer (10): X19–X28
        // FP       (8): V8–V15  (lower 64 bits only per AAPCS64)
        static COMBINED_CALLEE: [PhysReg; 18] = [
            registers::X19,
            registers::X20,
            registers::X21,
            registers::X22,
            registers::X23,
            registers::X24,
            registers::X25,
            registers::X26,
            registers::X27,
            registers::X28,
            registers::V8,
            registers::V9,
            registers::V10,
            registers::V11,
            registers::V12,
            registers::V13,
            registers::V14,
            registers::V15,
        ];
        &COMBINED_CALLEE
    }

    /// Returns the caller-saved (volatile) registers for AArch64.
    ///
    /// Per AAPCS64:
    /// - Integer: X0–X18 (19 registers)
    /// - FP/SIMD: V0–V7, V16–V31 (24 registers)
    fn caller_saved_registers(&self) -> &[PhysReg] {
        // Combined integer + FP caller-saved set for register allocator.
        static COMBINED_CALLER: [PhysReg; 43] = [
            // Integer caller-saved (19)
            registers::X0,
            registers::X1,
            registers::X2,
            registers::X3,
            registers::X4,
            registers::X5,
            registers::X6,
            registers::X7,
            registers::X8,
            registers::X9,
            registers::X10,
            registers::X11,
            registers::X12,
            registers::X13,
            registers::X14,
            registers::X15,
            registers::X16,
            registers::X17,
            registers::X18,
            // FP caller-saved (24)
            registers::V0,
            registers::V1,
            registers::V2,
            registers::V3,
            registers::V4,
            registers::V5,
            registers::V6,
            registers::V7,
            registers::V16,
            registers::V17,
            registers::V18,
            registers::V19,
            registers::V20,
            registers::V21,
            registers::V22,
            registers::V23,
            registers::V24,
            registers::V25,
            registers::V26,
            registers::V27,
            registers::V28,
            registers::V29,
            registers::V30,
            registers::V31,
        ];
        &COMBINED_CALLER
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

    /// Returns the scratch GPR reserved for spill code generation.
    ///
    /// X16 (IP0) is the AArch64 intra-procedure-call register, used by
    /// the linker for PLT stubs and veneers. It is caller-saved and not
    /// used for parameter passing, making it safe to reserve for spill
    /// load/store pseudo-operations. The register allocator excludes this
    /// register from the allocatable pool so it never conflicts with any
    /// live value.
    #[inline]
    fn spill_scratch_gpr(&self) -> PhysReg {
        registers::X16
    }

    /// Returns the scratch FP register reserved for spill code generation.
    ///
    /// V31 is a caller-saved SIMD/FP register not used for parameter
    /// passing or return values. Reserving it for spill code prevents
    /// conflicts with allocated floating-point values.
    #[inline]
    fn spill_scratch_sse(&self) -> PhysReg {
        registers::V31
    }

    /// Returns X30 (LR) — the AArch64 link register that holds the return
    /// address after `BL`/`BLR`. Must be reserved by the register allocator
    /// to prevent clobbering the return address.
    #[inline]
    fn link_register(&self) -> PhysReg {
        registers::X30
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
        // ---------------------------------------------------------------
        // Step 0: Compute callee-saved registers from the *physical*
        // register operands that remain AFTER register allocation.
        // ---------------------------------------------------------------
        let mut used_callee: Vec<PhysReg> = Vec::new();
        for block in &mf.blocks {
            for instr in &block.instructions {
                for op in &instr.operands {
                    if let MachineOperand::Register(reg) = op {
                        if registers::is_callee_saved(*reg) && !used_callee.contains(reg) {
                            used_callee.push(*reg);
                        }
                    }
                }
                for reg in &instr.implicit_defs {
                    if registers::is_callee_saved(*reg) && !used_callee.contains(reg) {
                        used_callee.push(*reg);
                    }
                }
            }
        }
        mf.used_callee_saved = used_callee;

        // ---------------------------------------------------------------
        // Step 1: Compute final frame size.
        //
        // Frame layout (growing upward from FP after prologue):
        //
        //   [FP + 0]               = saved FP (X29)
        //   [FP + 8]               = saved LR (X30)
        //   [FP + 16]              = VA save area (64 bytes if variadic)
        //   [FP + 16 + va]         = callee-saved registers
        //   [FP + local_base]      = local variables (allocas)
        //   [FP + spill_base]      = register allocator spill slots
        //   [FP + frame_size - 1]  = end of frame / previous SP
        //
        // ---------------------------------------------------------------
        use crate::backend::register_allocator::{SPILL_LOAD_OPCODE, SPILL_STORE_OPCODE};

        let fp_lr_size: u32 = 16;
        let va_save_size: u32 = if mf.is_variadic { 64 } else { 0 };
        let callee_save_bytes = (mf.used_callee_saved.len() as u32) * 8;
        let callee_save_aligned = (callee_save_bytes + 15) & !15;
        let local_size = ((-mf.frame_offset_watermark) as u32 + 15) & !15;

        // Capture the spill area size contributed by the register
        // allocator (generate_spill_code sets mf.frame_size to the
        // cumulative spill slot bytes).  We must save this BEFORE we
        // overwrite mf.frame_size with the full frame layout.
        let regalloc_spill_size = mf.frame_size;
        let regalloc_spill_aligned = (regalloc_spill_size + 15) & !15;

        let max_object_align = mf
            .frame_objects
            .iter()
            .map(|fo| fo.1)
            .max()
            .unwrap_or(16)
            .max(16);
        // Include the spill area in the total frame size so that the
        // STP [SP, #-frame_size]! prologue instruction allocates enough
        // space for both locals and spills.
        let total =
            fp_lr_size + va_save_size + callee_save_aligned + local_size + regalloc_spill_aligned;
        let frame_size = (total + max_object_align - 1) & !(max_object_align - 1);
        mf.frame_size = frame_size;

        // Base offset (from FP) where alloca locals start.
        let local_base = fp_lr_size + va_save_size + callee_save_aligned;

        // Base offset (from FP) where spill slots start — immediately
        // after the local variable area.
        let spill_base = local_base + local_size;

        // ---------------------------------------------------------------
        // Step 2: Resolve FrameIndex operands → concrete FP-relative.
        //
        // Two categories of FrameIndex operands exist:
        //   (a) Alloca / frame-object indices — small integers (0, 1, 2…)
        //       that index into mf.frame_objects.  These are emitted by
        //       instruction selection (codegen) and need to be translated
        //       to FP-relative byte offsets via fi_offsets[].
        //   (b) Spill byte-offsets — larger numbers (8, 16, 24…) emitted
        //       by the register allocator on SPILL_LOAD / SPILL_STORE
        //       pseudo-instructions.  These need to be rebased from
        //       alloca-relative (starting at 0) to FP-relative by adding
        //       spill_base.
        //
        // We process both categories in a single pass and use the
        // instruction opcode to distinguish them.
        // ---------------------------------------------------------------
        {
            // Build the alloca frame-object offset table.
            let mut fi_offsets: Vec<i32> = Vec::with_capacity(mf.frame_objects.len());
            for &(_sz, _al, off) in &mf.frame_objects {
                let within_local = local_size as i32 + off;
                fi_offsets.push(local_base as i32 + within_local);
            }

            for block in &mut mf.blocks {
                for instr in &mut block.instructions {
                    // -------------------------------------------------
                    // Category (b): SPILL pseudo-instructions.
                    // Adjust the FrameIndex byte offset to be relative
                    // to FP by adding spill_base.  Leave the operand as
                    // FrameIndex so the encoder's encode_spill_op()
                    // picks it up correctly.
                    // -------------------------------------------------
                    let opc = instr.opcode;
                    let is_spill = opc == SPILL_LOAD_OPCODE
                        || opc == SPILL_STORE_OPCODE
                        || opc == (SPILL_LOAD_OPCODE & 0x00FF_FFFF)
                        || opc == (SPILL_STORE_OPCODE & 0x00FF_FFFF);
                    if is_spill {
                        if let Some(MachineOperand::FrameIndex(off)) = instr.operands.get_mut(1) {
                            // old offset: alloca-relative (8, 16, 24…)
                            // new offset: FP-relative (spill_base + old)
                            *off += spill_base;
                        }
                        continue;
                    }

                    // -------------------------------------------------
                    // Category (a): Normal instructions with FrameIndex
                    // operands (alloca slot indices).
                    // -------------------------------------------------
                    let is_add_fp_fi = instr.operands.len() >= 3
                        && (instr.opcode == codegen::AArch64Opcode::ADDimm.as_u32()
                            || instr.opcode == codegen::AArch64Opcode::ADD.as_u32())
                        && matches!(instr.operands[1], MachineOperand::Register(r) if r == registers::SP || r == registers::FP)
                        && matches!(instr.operands[2], MachineOperand::FrameIndex(_));
                    if is_add_fp_fi {
                        if let MachineOperand::FrameIndex(idx) = instr.operands[2] {
                            let fi = idx as usize;
                            let fp_off = if fi < fi_offsets.len() {
                                fi_offsets[fi]
                            } else {
                                // Fallback for unknown indices — treat
                                // as a raw byte offset in the local
                                // area (shouldn't normally happen).
                                local_base as i32 + idx as i32
                            };
                            instr.operands[1] = MachineOperand::Register(registers::FP);
                            instr.operands[2] = MachineOperand::Immediate(fp_off as i64);
                        }
                    } else {
                        for op in &mut instr.operands {
                            if let MachineOperand::FrameIndex(idx) = op {
                                let fi = *idx as usize;
                                let fp_off = if fi < fi_offsets.len() {
                                    fi_offsets[fi]
                                } else {
                                    local_base as i32 + *idx as i32
                                };
                                *op = MachineOperand::Memory {
                                    base: registers::FP,
                                    offset: fp_off,
                                    index: None,
                                    scale: 1,
                                };
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Step 3: Emit prologue instructions.
        // ---------------------------------------------------------------
        if mf.blocks.is_empty() {
            return;
        }
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            return;
        }

        let mut prologue: Vec<MachineInstr> = Vec::new();

        // The STP pre-index instruction `STP X29, X30, [SP, #-imm]!`
        // uses a 7-bit signed immediate scaled by 8, giving a valid
        // byte range of -512 to +504.  For frames larger than 504
        // bytes we must use a separate SUB to allocate the stack space
        // first, then a regular (signed-offset) STP at offset 0.
        //
        //   Small frame (≤ 504):
        //       stp  x29, x30, [sp, #-frame_size]!
        //       mov  x29, sp
        //
        //   Large frame (> 504):
        //       sub  sp, sp, #frame_size      // may be 2 SUBs for > 4095
        //       stp  x29, x30, [sp]           // signed-offset, imm7 = 0
        //       mov  x29, sp
        //
        const STP_PRE_MAX: u32 = 504; // 63 * 8

        if frame_size <= STP_PRE_MAX {
            // Small frame — single STP pre-index
            prologue.push(MachineInstr::with_operands(
                codegen::AArch64Opcode::StpPre.as_u32(),
                vec![
                    MachineOperand::Register(registers::FP),
                    MachineOperand::Register(registers::LR),
                    MachineOperand::Memory {
                        base: registers::SP,
                        offset: -(frame_size as i32),
                        index: None,
                        scale: 1,
                    },
                ],
            ));
        } else {
            // Large frame — allocate with SUB(s), then STP at [SP, #0].
            //
            // SUBimm uses a 12-bit unsigned immediate (0–4095), with an
            // optional LSL #12 shift for the upper 12 bits.  Split the
            // frame size into high (multiples of 4096) and low parts.
            let hi = frame_size & !0xFFF; // multiple-of-4096 portion
            let lo = frame_size & 0xFFF; // remainder < 4096

            if hi > 0 {
                // SUB SP, SP, #hi_page, LSL #12
                prologue.push(MachineInstr::with_operands(
                    codegen::AArch64Opcode::SUBimm.as_u32(),
                    vec![
                        MachineOperand::Register(registers::SP),
                        MachineOperand::Register(registers::SP),
                        MachineOperand::Immediate((hi >> 12) as i64),
                        MachineOperand::Immediate(1), // shift = true (LSL #12)
                    ],
                ));
            }
            if lo > 0 || hi == 0 {
                // SUB SP, SP, #lo
                prologue.push(MachineInstr::with_operands(
                    codegen::AArch64Opcode::SUBimm.as_u32(),
                    vec![
                        MachineOperand::Register(registers::SP),
                        MachineOperand::Register(registers::SP),
                        MachineOperand::Immediate(lo as i64),
                    ],
                ));
            }

            // STP X29, X30, [SP, #0]  (signed-offset form, imm7 = 0)
            prologue.push(MachineInstr::with_operands(
                codegen::AArch64Opcode::STP.as_u32(),
                vec![
                    MachineOperand::Register(registers::FP),
                    MachineOperand::Register(registers::LR),
                    MachineOperand::Memory {
                        base: registers::SP,
                        offset: 0,
                        index: None,
                        scale: 1,
                    },
                ],
            ));
        }

        // MOV X29, SP  (encoded as ADD X29, SP, #0)
        prologue.push(MachineInstr::with_operands(
            codegen::AArch64Opcode::ADDimm.as_u32(),
            vec![
                MachineOperand::Register(registers::FP),
                MachineOperand::Register(registers::SP),
                MachineOperand::Immediate(0),
            ],
        ));

        // Variadic: spill X0-X7
        if mf.is_variadic {
            for reg_idx in 0u32..8 {
                let arg_reg = registers::INTEGER_ARG_REGS[reg_idx as usize];
                let save_offset = 16 + (reg_idx as i32) * 8;
                prologue.push(MachineInstr::with_operands(
                    codegen::AArch64Opcode::STR.as_u32(),
                    vec![
                        MachineOperand::Register(arg_reg),
                        MachineOperand::Memory {
                            base: registers::SP,
                            offset: save_offset,
                            index: None,
                            scale: 1,
                        },
                    ],
                ));
            }
        }

        // Save callee-saved register pairs
        let callee_regs = &mf.used_callee_saved;
        let callee_start = if mf.is_variadic { 80i32 } else { 16i32 };
        let mut offset = callee_start;
        let mut i = 0;
        while i + 1 < callee_regs.len() {
            prologue.push(MachineInstr::with_operands(
                codegen::AArch64Opcode::STP.as_u32(),
                vec![
                    MachineOperand::Register(callee_regs[i]),
                    MachineOperand::Register(callee_regs[i + 1]),
                    MachineOperand::Memory {
                        base: registers::SP,
                        offset,
                        index: None,
                        scale: 1,
                    },
                ],
            ));
            offset += 16;
            i += 2;
        }
        if i < callee_regs.len() {
            prologue.push(MachineInstr::with_operands(
                codegen::AArch64Opcode::STR.as_u32(),
                vec![
                    MachineOperand::Register(callee_regs[i]),
                    MachineOperand::Memory {
                        base: registers::SP,
                        offset,
                        index: None,
                        scale: 1,
                    },
                ],
            ));
        }

        // Splice prologue at the front of the entry block
        if let Some(entry) = mf.blocks.first_mut() {
            let old_instrs = std::mem::take(&mut entry.instructions);
            entry.instructions = prologue;
            entry.instructions.extend(old_instrs);
        }
    }

    fn emit_epilogue(&self, mf: &mut MachineFunction) {
        let frame_size = mf.frame_size;
        if frame_size == 0 && mf.used_callee_saved.is_empty() {
            return;
        }

        let callee_regs = mf.used_callee_saved.clone();
        let is_variadic = mf.is_variadic;

        for block in &mut mf.blocks {
            let ret_positions: Vec<usize> = block
                .instructions
                .iter()
                .enumerate()
                .filter(|(_, i)| i.opcode == codegen::AArch64Opcode::RET.as_u32())
                .map(|(idx, _)| idx)
                .collect();

            for &pos in ret_positions.iter().rev() {
                let mut epilogue: Vec<MachineInstr> = Vec::new();

                // Restore callee-saved register pairs
                let callee_start = if is_variadic { 80i32 } else { 16i32 };
                let mut offset = callee_start;
                let mut i = 0;
                while i + 1 < callee_regs.len() {
                    epilogue.push(MachineInstr::with_operands(
                        codegen::AArch64Opcode::LDP.as_u32(),
                        vec![
                            MachineOperand::Register(callee_regs[i]),
                            MachineOperand::Register(callee_regs[i + 1]),
                            MachineOperand::Memory {
                                base: registers::SP,
                                offset,
                                index: None,
                                scale: 1,
                            },
                        ],
                    ));
                    offset += 16;
                    i += 2;
                }
                if i < callee_regs.len() {
                    epilogue.push(MachineInstr::with_operands(
                        codegen::AArch64Opcode::LDR.as_u32(),
                        vec![
                            MachineOperand::Register(callee_regs[i]),
                            MachineOperand::Memory {
                                base: registers::SP,
                                offset,
                                index: None,
                                scale: 1,
                            },
                        ],
                    ));
                }

                // Restore FP/LR and deallocate the frame.
                //
                // LDP post-index has the same 7-bit signed immediate
                // limit as STP pre-index (±504 bytes scaled by 8).
                // For large frames, use LDP [SP] + ADD SP, SP, #size.
                const STP_PRE_MAX_EPI: u32 = 504;
                if frame_size <= STP_PRE_MAX_EPI {
                    // Small frame — single LDP post-index
                    epilogue.push(MachineInstr::with_operands(
                        codegen::AArch64Opcode::LdpPost.as_u32(),
                        vec![
                            MachineOperand::Register(registers::FP),
                            MachineOperand::Register(registers::LR),
                            MachineOperand::Memory {
                                base: registers::SP,
                                offset: frame_size as i32,
                                index: None,
                                scale: 1,
                            },
                        ],
                    ));
                } else {
                    // Large frame — LDP at [SP, #0] then ADD SP
                    epilogue.push(MachineInstr::with_operands(
                        codegen::AArch64Opcode::LDP.as_u32(),
                        vec![
                            MachineOperand::Register(registers::FP),
                            MachineOperand::Register(registers::LR),
                            MachineOperand::Memory {
                                base: registers::SP,
                                offset: 0,
                                index: None,
                                scale: 1,
                            },
                        ],
                    ));
                    // ADD SP, SP, #frame_size  (split if > 4095)
                    let hi = frame_size & !0xFFF;
                    let lo = frame_size & 0xFFF;
                    if hi > 0 {
                        epilogue.push(MachineInstr::with_operands(
                            codegen::AArch64Opcode::ADDimm.as_u32(),
                            vec![
                                MachineOperand::Register(registers::SP),
                                MachineOperand::Register(registers::SP),
                                MachineOperand::Immediate((hi >> 12) as i64),
                                MachineOperand::Immediate(1), // LSL #12
                            ],
                        ));
                    }
                    if lo > 0 || hi == 0 {
                        epilogue.push(MachineInstr::with_operands(
                            codegen::AArch64Opcode::ADDimm.as_u32(),
                            vec![
                                MachineOperand::Register(registers::SP),
                                MachineOperand::Register(registers::SP),
                                MachineOperand::Immediate(lo as i64),
                            ],
                        ));
                    }
                }

                // Splice epilogue before the RET
                let tail = block.instructions.split_off(pos);
                block.instructions.extend(epilogue);
                block.instructions.extend(tail);
            }
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

    /// Verify prologue is emitted even for a leaf function with zero
    /// user-level frame size, because the AArch64 prologue always
    /// saves FP/LR (16 bytes) to maintain a valid frame chain.
    #[test]
    fn test_prologue_skipped_for_leaf() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        let mut mf = MachineFunction::new("leaf_fn".to_string(), 16);
        mf.create_block();
        mf.frame_size = 0;

        codegen.emit_prologue(&mut mf);
        // Even with frame_size=0, the FP/LR pair (16 bytes) is always
        // saved, so prologue instructions are generated.
        assert!(
            !mf.blocks[0].instructions.is_empty(),
            "emit_prologue always saves FP/LR on AArch64"
        );
    }

    /// Verify emit_prologue generates code for a non-leaf function.
    ///
    /// For a function with `frame_size > 0`, emit_prologue must produce
    /// at least the STP (save FP/LR) and MOV FP,SP instructions.
    #[test]
    fn test_prologue_emitted_for_nonleaf() {
        let config = CodegenConfig::new(Target::AArch64);
        let codegen = AArch64Codegen::new(config);

        let mut mf = MachineFunction::new("nonleaf_fn".to_string(), 16);
        mf.create_block();
        mf.frame_size = 32;

        codegen.emit_prologue(&mut mf);
        // emit_prologue generates real prologue instructions: at least
        // the STP X29,X30,[SP,#-size]! and MOV X29,SP pair.
        assert!(
            !mf.blocks[0].instructions.is_empty(),
            "emit_prologue should produce prologue instructions for non-leaf"
        );
    }
}
