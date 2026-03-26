//! # AArch64 Backend
//!
//! Complete AArch64 (ARM 64-bit) code generation backend for BCC.
//! Implements the [`ArchCodegen`] trait for the A64 ISA with AAPCS64 ABI.
//!
//! ## Submodules
//! - `codegen` — Instruction selection (IR → AArch64 machine instructions)
//! - `registers` — Register file definitions (X0–X30, SP, V0–V31, NZCV)
//! - `abi` — AAPCS64 calling convention, HFA/HVA handling, stack frame layout
//! - `assembler` — Built-in AArch64 assembler (A64 instruction encoding, relocations)
//! - `linker` — Built-in AArch64 ELF linker (relocation application)
//!
//! ## Architecture Characteristics
//! - Fixed 32-bit instruction width (no variable-length encoding)
//! - Load/store architecture — ALU ops on registers only
//! - 31 general-purpose registers: X0–X30 (64-bit) / W0–W30 (32-bit)
//! - SP (stack pointer) and XZR (zero register) share encoding
//! - 32 SIMD/FP registers: V0–V31 (128-bit), with D/S/H/B views
//! - NZCV condition flags for conditional execution
//! - Little-endian byte order (default)
//! - ELF machine type: EM_AARCH64 (183)
//! - Page size: 4KB (default), 16KB, or 64KB
//!
//! ## Key: ADRP+ADD for PIC addressing, STP/LDP for efficient stack ops.

// ============================================================================
// Submodule Declarations
// ============================================================================

pub mod abi;
pub mod assembler;
pub mod codegen;
pub mod linker;
pub mod registers;

// ============================================================================
// Crate-level Imports
// ============================================================================

use crate::backend::elf_writer_common::{
    ElfSymbol, ElfWriter, Section, ET_REL, SHF_ALLOC, SHF_EXECINSTR, SHF_INFO_LINK, SHF_WRITE,
    SHT_NOBITS, SHT_PROGBITS, SHT_RELA, STB_GLOBAL, STT_FUNC, STV_DEFAULT,
};
use crate::backend::traits::{
    ArchCodegen, ArgLocation, AssembledFunction, FunctionRelocation, MachineFunction,
    MachineInstruction, MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::module::{Constant, IrModule};
use crate::ir::types::IrType;
// CType and MachineType are used for ABI classification in submodules
// and kept available for future extensions.
use crate::common::diagnostics::{DiagnosticEngine, Span};
#[allow(unused_imports)]
use crate::common::types::{CType, MachineType};

// ============================================================================
// Public Re-exports
// ============================================================================

/// Re-export instruction selection types for external access.
pub use self::codegen::{A64Instruction, A64Opcode, AArch64InstructionSelector, CondCode};

/// Re-export register file definitions.
pub use self::registers::{AArch64RegisterInfo, RegClass};

/// Re-export ABI types.
pub use self::abi::{classify_arguments_ir, AArch64Abi, ArgClass, FrameLayout};

// ============================================================================
// AArch64 ELF Constants
// ============================================================================

/// ELF machine type for AArch64.
///
/// Value 183 (0xB7) as defined in the ELF specification for AArch64.
/// Written to the `e_machine` field of the ELF header.
pub const EM_AARCH64: u16 = 183;

/// ELF flags for standard AArch64 (no special flags).
///
/// Unlike RISC-V which encodes ISA extensions in flags, AArch64 ELF
/// files use flags = 0 for the standard configuration.
pub const ELF_FLAGS: u32 = 0;

/// Default base virtual address for AArch64 static executables (ET_EXEC).
///
/// Position-independent executables (PIE) and shared libraries use a
/// base address of 0x0; the dynamic linker handles ASLR placement.
pub const DEFAULT_BASE_ADDRESS: u64 = 0x400000;

/// Default page size for AArch64 (4 KiB).
///
/// AArch64 supports 4 KiB, 16 KiB, and 64 KiB page sizes, but 4 KiB
/// is the standard default for Linux. Used for ADRP page-relative
/// addressing calculations and ELF segment alignment.
pub const PAGE_SIZE: u64 = 4096;

// ============================================================================
// STP/LDP Opcode Constants for Prologue/Epilogue
// ============================================================================

/// Opcode identifier for STP (Store Pair, signed offset) — used in prologue generation.
const OPCODE_STP: u32 = 0xA9_00_00_00;

/// Opcode identifier for STP pre-index — `STP Rt, Rt2, [Rn, #imm]!`
const OPCODE_STP_PRE: u32 = 0xA9_80_00_00;

/// Opcode identifier for LDP (Load Pair, signed offset) — used in epilogue generation.
const OPCODE_LDP: u32 = 0xA9_40_00_00;

/// Opcode identifier for LDP post-index — `LDP Rt, Rt2, [Rn], #imm`
const OPCODE_LDP_POST: u32 = 0xA9_C0_00_00;

/// Opcode identifier for MOV (register) — used for setting frame pointer.
const OPCODE_MOV_REG: u32 = 0xAA_00_00_00;

/// Opcode identifier for RET — return from subroutine.
const OPCODE_RET: u32 = 0xD6_5F_00_00;

/// Opcode identifier for SUB immediate — stack pointer adjustment.
const OPCODE_SUB_IMM: u32 = 0xD1_00_00_00;

/// Opcode identifier for ADD immediate — stack pointer restore.
const OPCODE_ADD_IMM: u32 = 0x91_00_00_00;

/// Opcode identifier for STP (FP pair) — callee-saved FP regs.
const OPCODE_STP_FP: u32 = 0xAD_00_00_00;

/// Opcode identifier for LDP (FP pair) — callee-saved FP reg restore.
const OPCODE_LDP_FP: u32 = 0xAD_40_00_00;

// ============================================================================
// AArch64Codegen — Primary Backend Struct
// ============================================================================

/// AArch64 code generation backend.
///
/// Implements the [`ArchCodegen`] trait, providing AArch64 instruction selection,
/// register allocation interface, prologue/epilogue generation, and machine code
/// emission. This struct is the single entry point for all AArch64-specific code
/// generation, instantiated by `crate::backend::generation` for `Target::AArch64`.
///
/// # Architecture Summary
///
/// - **ISA**: A64 (AArch64) — fixed 32-bit instruction width
/// - **ABI**: AAPCS64 — X0–X7 for integer args, V0–V7 for FP args
/// - **Registers**: 31 GPRs (X0–X30) + SP + ZR, 32 SIMD/FP (V0–V31)
/// - **Frame**: STP/LDP for paired register saves, FP = X29, LR = X30
/// - **PIC**: ADRP+ADD for PC-relative addressing, ADRP+LDR for GOT access
pub struct AArch64Codegen {
    /// Target architecture information — always `Target::AArch64`.
    target: Target,
    /// Register information provider for the AArch64 register file.
    reg_info: AArch64RegisterInfo,
    /// ABI handler for AAPCS64 calling convention classification.
    abi: AArch64Abi,
    /// Position-independent code mode (`-fPIC`).
    pic_mode: bool,
    /// Debug information generation (`-g` flag).
    /// Used by `compile_to_object` to conditionally emit DWARF sections.
    #[allow(dead_code)]
    debug_info: bool,
    /// Relocation type descriptors for AArch64 ELF relocations.
    relocation_types: Vec<RelocationTypeInfo>,
}

// ============================================================================
// AArch64Codegen — Constructor and Helpers
// ============================================================================

impl AArch64Codegen {
    /// Create a new AArch64 code generation backend.
    ///
    /// # Arguments
    ///
    /// * `pic_mode` — If `true`, generate position-independent code suitable
    ///   for shared libraries. Uses GOT-relative addressing for global symbols.
    /// * `debug_info` — If `true`, emit DWARF v4 debug sections.
    pub fn new(pic_mode: bool, debug_info: bool) -> Self {
        Self {
            target: Target::AArch64,
            reg_info: AArch64RegisterInfo::new(),
            abi: AArch64Abi::new(),
            pic_mode,
            debug_info,
            relocation_types: Self::build_relocation_types(),
        }
    }

    /// Build the complete list of AArch64 ELF relocation type descriptors.
    fn build_relocation_types() -> Vec<RelocationTypeInfo> {
        vec![
            RelocationTypeInfo::new("R_AARCH64_NONE", 0, 0, false),
            // Absolute data relocations
            RelocationTypeInfo::new("R_AARCH64_ABS64", 257, 8, false),
            RelocationTypeInfo::new("R_AARCH64_ABS32", 258, 4, false),
            RelocationTypeInfo::new("R_AARCH64_ABS16", 259, 2, false),
            // PC-relative data relocations
            RelocationTypeInfo::new("R_AARCH64_PREL64", 260, 8, true),
            RelocationTypeInfo::new("R_AARCH64_PREL32", 261, 4, true),
            RelocationTypeInfo::new("R_AARCH64_PREL16", 262, 2, true),
            // ADR/ADRP page-relative relocations
            RelocationTypeInfo::new("R_AARCH64_ADR_PREL_LO21", 274, 4, true),
            RelocationTypeInfo::new("R_AARCH64_ADR_PREL_PG_HI21", 275, 4, true),
            // ADD/LDR :lo12: relocations
            RelocationTypeInfo::new("R_AARCH64_ADD_ABS_LO12_NC", 277, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST8_ABS_LO12_NC", 278, 4, false),
            // TBZ/TBNZ relocation
            RelocationTypeInfo::new("R_AARCH64_TSTBR14", 279, 4, true),
            // B.cond relocation
            RelocationTypeInfo::new("R_AARCH64_CONDBR19", 280, 4, true),
            // B/BL relocations
            RelocationTypeInfo::new("R_AARCH64_JUMP26", 282, 4, true),
            RelocationTypeInfo::new("R_AARCH64_CALL26", 283, 4, true),
            // Load/Store :lo12: relocations (scaled)
            RelocationTypeInfo::new("R_AARCH64_LDST16_ABS_LO12_NC", 284, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST32_ABS_LO12_NC", 285, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST64_ABS_LO12_NC", 286, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST128_ABS_LO12_NC", 299, 4, false),
            // GOT-relative relocations (PIC)
            RelocationTypeInfo::new("R_AARCH64_ADR_GOT_PAGE", 311, 4, true),
            RelocationTypeInfo::new("R_AARCH64_LD64_GOT_LO12_NC", 312, 4, false),
            // TLS relocations
            RelocationTypeInfo::new("R_AARCH64_TLSLE_ADD_TPREL_HI12", 549, 4, false),
            RelocationTypeInfo::new("R_AARCH64_TLSLE_ADD_TPREL_LO12_NC", 551, 4, false),
            // Dynamic relocations
            RelocationTypeInfo::new("R_AARCH64_COPY", 1024, 8, false),
            RelocationTypeInfo::new("R_AARCH64_GLOB_DAT", 1025, 8, false),
            RelocationTypeInfo::new("R_AARCH64_JUMP_SLOT", 1026, 8, false),
            RelocationTypeInfo::new("R_AARCH64_RELATIVE", 1027, 8, false),
            // TLS dynamic relocations
            RelocationTypeInfo::new("R_AARCH64_TLS_DTPMOD64", 1028, 8, false),
            RelocationTypeInfo::new("R_AARCH64_TLS_DTPREL64", 1029, 8, false),
            RelocationTypeInfo::new("R_AARCH64_TLS_TPREL64", 1030, 8, false),
            RelocationTypeInfo::new("R_AARCH64_TLSDESC", 1031, 8, false),
        ]
    }

    /// Convert a list of AArch64-specific `A64Instruction`s into
    /// architecture-agnostic `MachineInstruction`s.
    fn convert_instructions(a64_instructions: &[A64Instruction]) -> Vec<MachineInstruction> {
        let mut result = Vec::with_capacity(a64_instructions.len());
        for inst in a64_instructions {
            let mut mi = MachineInstruction::new(inst.opcode as u32);

            // Preserve block label information from NOP pseudo-instructions.
            // The codegen emits NOP with comment "labelname:" to mark basic
            // block boundaries. We store this in asm_template so
            // emit_assembly can call define_label() at the correct offset.
            if inst.opcode == A64Opcode::NOP {
                if let Some(ref comment) = inst.comment {
                    if comment.ends_with(':') {
                        mi.asm_template = Some(format!("LABEL:{}", &comment[..comment.len() - 1]));
                    }
                }
            }

            // Set operand size based on 32-bit vs 64-bit forms.
            mi.operand_size = if inst.is_32bit { 4 } else { 8 };

            // Preserve condition code through the round-trip so that
            // CSET, B.cond, CSEL, etc. retain their condition after
            // register allocation.
            if let Some(cc) = inst.cond {
                mi.cond_code = Some(cc.encoding());
            }

            // Preserve shift/rotate encoding through the round-trip
            // so that MOVZ/MOVK/MOVN halfword shifts and register-
            // shifted ALU operands survive register allocation.
            mi.arch_shift = inst.shift;

            // Map destination register as the result operand.
            // Virtual registers (from IR values) use VirtualRegister so
            // that apply_allocation_result can replace them with physical
            // registers.  Physical registers (SP, FP, LR, argument regs)
            // use Register directly.
            if let Some(rd) = inst.rd {
                if inst.rd_is_virtual() {
                    mi.result = Some(MachineOperand::VirtualRegister(rd));
                } else {
                    mi.result = Some(MachineOperand::Register(rd as u16));
                }
            }

            // Map source registers as input operands.
            if let Some(rn) = inst.rn {
                if inst.rn_is_virtual() {
                    mi.operands.push(MachineOperand::VirtualRegister(rn));
                } else {
                    mi.operands.push(MachineOperand::Register(rn as u16));
                }
            }
            if let Some(rm) = inst.rm {
                if inst.rm_is_virtual() {
                    mi.operands.push(MachineOperand::VirtualRegister(rm));
                } else {
                    mi.operands.push(MachineOperand::Register(rm as u16));
                }
            }
            if let Some(ra) = inst.ra {
                if inst.ra_is_virtual() {
                    mi.operands.push(MachineOperand::VirtualRegister(ra));
                } else {
                    mi.operands.push(MachineOperand::Register(ra as u16));
                }
            }

            // Map immediate values.
            if inst.imm != 0
                || matches!(
                    inst.opcode,
                    A64Opcode::MOVZ | A64Opcode::MOVK | A64Opcode::MOVN
                )
            {
                mi.operands.push(MachineOperand::Immediate(inst.imm));
            }

            // Map symbol references.
            if let Some(ref sym) = inst.symbol {
                mi.operands.push(MachineOperand::GlobalSymbol(sym.clone()));
            }

            // Classify control-flow properties.
            match inst.opcode {
                A64Opcode::B | A64Opcode::BR => {
                    mi.is_terminator = true;
                    mi.is_branch = true;
                }
                A64Opcode::B_cond
                | A64Opcode::CBZ
                | A64Opcode::CBNZ
                | A64Opcode::TBZ
                | A64Opcode::TBNZ => {
                    mi.is_branch = true;
                }
                A64Opcode::BL | A64Opcode::BLR | A64Opcode::CALL => {
                    mi.is_call = true;
                }
                A64Opcode::RET => {
                    mi.is_terminator = true;
                }
                _ => {}
            }

            // Propagate the call-argument-setup flag so the post-register-
            // allocation parallel-move resolver can identify the contiguous
            // arg-setup window before CALL.
            if inst.is_call_arg_setup {
                mi.is_call_arg_setup = true;
            }

            result.push(mi);
        }
        result
    }

    /// Create a `MachineInstruction` for a paired store (STP) operation.
    fn make_stp(rt1: u16, rt2: u16, base: u16, offset: i64, is_fp: bool) -> MachineInstruction {
        let opcode = if is_fp { OPCODE_STP_FP } else { OPCODE_STP };
        let mut mi = MachineInstruction::new(opcode);
        mi.operands.push(MachineOperand::Register(rt1));
        mi.operands.push(MachineOperand::Register(rt2));
        mi.operands.push(MachineOperand::Memory {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: offset,
        });
        mi.operand_size = 8;
        mi
    }

    /// Create a `MachineInstruction` for a paired load (LDP) operation.
    fn make_ldp(rt1: u16, rt2: u16, base: u16, offset: i64, is_fp: bool) -> MachineInstruction {
        let opcode = if is_fp { OPCODE_LDP_FP } else { OPCODE_LDP };
        let mut mi = MachineInstruction::new(opcode);
        mi.result = Some(MachineOperand::Register(rt1));
        mi.operands.push(MachineOperand::Register(rt2));
        mi.operands.push(MachineOperand::Memory {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: offset,
        });
        mi.operand_size = 8;
        mi
    }

    /// Emit an STP to `[base, #offset]`, falling back to an ADD+STP
    /// sequence when the offset exceeds the STP 7-bit signed range.
    ///
    /// The `scratch` register (typically IP0 / X16) is used as the
    /// intermediate base when the offset is too large.
    fn emit_stp_large_offset(
        out: &mut Vec<MachineInstruction>,
        rt1: u16,
        rt2: u16,
        base: u16,
        offset: i64,
        is_fp: bool,
        scratch: u16,
        stp_max: i64,
    ) {
        if offset >= -512 && offset <= stp_max {
            out.push(Self::make_stp(rt1, rt2, base, offset, is_fp));
        } else {
            // ADD scratch, base, #offset   (offset fits in 12-bit unsigned for ≤4095)
            let mut add_mi = MachineInstruction::new(OPCODE_ADD_IMM);
            add_mi.result = Some(MachineOperand::Register(scratch));
            add_mi.operands.push(MachineOperand::Register(base));
            add_mi.operands.push(MachineOperand::Immediate(offset));
            add_mi.operand_size = 8;
            out.push(add_mi);
            // STP rt1, rt2, [scratch, #0]
            out.push(Self::make_stp(rt1, rt2, scratch, 0, is_fp));
        }
    }

    /// Emit an LDP from `[base, #offset]`, falling back to an ADD+LDP
    /// sequence when the offset exceeds the LDP 7-bit signed range.
    fn emit_ldp_large_offset(
        out: &mut Vec<MachineInstruction>,
        rt1: u16,
        rt2: u16,
        base: u16,
        offset: i64,
        is_fp: bool,
        scratch: u16,
        ldp_max: i64,
    ) {
        if offset >= -512 && offset <= ldp_max {
            out.push(Self::make_ldp(rt1, rt2, base, offset, is_fp));
        } else {
            // ADD scratch, base, #offset
            let mut add_mi = MachineInstruction::new(OPCODE_ADD_IMM);
            add_mi.result = Some(MachineOperand::Register(scratch));
            add_mi.operands.push(MachineOperand::Register(base));
            add_mi.operands.push(MachineOperand::Immediate(offset));
            add_mi.operand_size = 8;
            out.push(add_mi);
            // LDP rt1, rt2, [scratch, #0]
            out.push(Self::make_ldp(rt1, rt2, scratch, 0, is_fp));
        }
    }

    /// Create a `MachineInstruction` for a MOV register operation.
    #[allow(dead_code)]
    fn make_mov_reg(rd: u16, rn: u16) -> MachineInstruction {
        let mut mi = MachineInstruction::new(OPCODE_MOV_REG);
        mi.result = Some(MachineOperand::Register(rd));
        mi.operands.push(MachineOperand::Register(rn));
        mi.operand_size = 8;
        mi
    }

    /// Create a `MachineInstruction` for RET.
    fn make_ret() -> MachineInstruction {
        let mut mi = MachineInstruction::new(OPCODE_RET);
        mi.operands
            .push(MachineOperand::Register(registers::LR as u16));
        mi.is_terminator = true;
        mi.operand_size = 8;
        mi
    }

    /// Align a value up to the given alignment.
    #[inline]
    fn align_up(value: usize, align: usize) -> usize {
        if align == 0 {
            return value;
        }
        (value + align - 1) & !(align - 1)
    }

    /// Serialize an IR [`Constant`] to raw little-endian bytes.
    ///
    /// Used during object file generation to produce initialized data
    /// for `.data` and `.rodata` sections.
    #[allow(clippy::only_used_in_recursion)]
    fn serialize_constant(constant: &Constant, target: &Target) -> Vec<u8> {
        match constant {
            Constant::Integer(val) => {
                // Default integer serialization: 8 bytes (i64-width).
                (*val as i64).to_le_bytes().to_vec()
            }
            Constant::Float(val) => val.to_le_bytes().to_vec(),
            Constant::LongDouble(bytes) => bytes.to_vec(),
            Constant::String(bytes) => bytes.clone(),
            Constant::ZeroInit => Vec::new(),
            Constant::Struct(fields) => {
                let mut data = Vec::new();
                for field in fields {
                    data.extend(Self::serialize_constant(field, target));
                }
                data
            }
            Constant::Array(elems) => {
                let mut data = Vec::new();
                for elem in elems {
                    data.extend(Self::serialize_constant(elem, target));
                }
                data
            }
            Constant::GlobalRef(_name) | Constant::GlobalRefOffset(_name, _) => {
                // Emit a placeholder 8-byte address; a relocation will patch it.
                vec![0u8; 8]
            }
            Constant::Null => vec![0u8; 8],
            Constant::Undefined => Vec::new(),
        }
    }

    /// Compile an IR module to an AArch64 ELF relocatable object file (.o).
    ///
    /// Orchestrates the complete code generation pipeline:
    /// 1. Lower each IR function to AArch64 machine instructions.
    /// 2. Assemble machine instructions into binary code.
    /// 3. Build the ELF relocatable object with sections and symbols.
    /// 4. Optionally emit DWARF debug sections.
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module containing functions, globals, and strings.
    /// * `diagnostics` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the complete ELF object bytes on success,
    /// or `Err(())` if code generation fails.
    #[allow(clippy::result_unit_err)]
    pub fn compile_to_object(
        &self,
        module: &IrModule,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<Vec<u8>, ()> {
        let mut asm = assembler::AArch64Assembler::new(self.pic_mode);
        let mut all_functions: Vec<(String, Vec<A64Instruction>)> = Vec::new();

        // Phase 1: Lower each function definition to AArch64 instructions.
        for func in module.functions() {
            if !func.is_definition {
                continue;
            }

            let mut selector = codegen::AArch64InstructionSelector::new(self.target, self.pic_mode);
            let a64_instructions = selector.select_function(func, &self.abi);
            all_functions.push((func.name.clone(), a64_instructions));
        }

        // Phase 2: Assemble all functions into machine code.
        let asm_result = match asm.assemble_module(&all_functions) {
            Ok(result) => result,
            Err(e) => {
                diagnostics.emit_error(Span::dummy(), format!("AArch64 assembly failed: {}", e));
                return Err(());
            }
        };

        // Phase 3: Build ELF relocatable object.
        let mut elf = ElfWriter::new(self.target, ET_REL);

        // Add .text section.
        let text_section = Section {
            name: ".text".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: asm_result.code.clone(),
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 4,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        let text_idx = elf.add_section(text_section);

        // Collect .data bytes from non-BSS, non-const global definitions.
        let mut data_bytes = Vec::new();
        for global in module.globals() {
            if !global.is_definition {
                continue;
            }
            let is_bss = matches!(&global.initializer, Some(Constant::ZeroInit) | None);
            if is_bss || global.is_constant {
                continue;
            }
            if let Some(ref init) = global.initializer {
                let align = global
                    .alignment
                    .unwrap_or(global.ty.align_bytes(&self.target));
                let padding = Self::align_up(data_bytes.len(), align) - data_bytes.len();
                data_bytes.extend(std::iter::repeat(0u8).take(padding));
                data_bytes.extend(Self::serialize_constant(init, &self.target));
            }
        }
        if !data_bytes.is_empty() {
            let data_section = Section {
                name: ".data".to_string(),
                sh_type: SHT_PROGBITS,
                sh_flags: SHF_ALLOC | SHF_WRITE,
                data: data_bytes,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: 0,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            elf.add_section(data_section);
        }

        // Collect .rodata bytes from string pool and constant globals.
        let mut rodata_bytes = Vec::new();
        for string_lit in module.string_pool() {
            let padding = Self::align_up(rodata_bytes.len(), 1) - rodata_bytes.len();
            rodata_bytes.extend(std::iter::repeat(0u8).take(padding));
            rodata_bytes.extend_from_slice(&string_lit.bytes);
        }
        for global in module.globals() {
            if !global.is_definition || !global.is_constant {
                continue;
            }
            let is_bss = matches!(&global.initializer, Some(Constant::ZeroInit) | None);
            if is_bss {
                continue;
            }
            if let Some(ref init) = global.initializer {
                let align = global
                    .alignment
                    .unwrap_or(global.ty.align_bytes(&self.target));
                let padding = Self::align_up(rodata_bytes.len(), align) - rodata_bytes.len();
                rodata_bytes.extend(std::iter::repeat(0u8).take(padding));
                rodata_bytes.extend(Self::serialize_constant(init, &self.target));
            }
        }
        if !rodata_bytes.is_empty() {
            let rodata_section = Section {
                name: ".rodata".to_string(),
                sh_type: SHT_PROGBITS,
                sh_flags: SHF_ALLOC,
                data: rodata_bytes,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: 0,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            elf.add_section(rodata_section);
        }

        // Add .bss section for zero-initialized/undefined globals.
        let mut bss_size: u64 = 0;
        for global in module.globals() {
            if !global.is_definition {
                continue;
            }
            let is_bss = matches!(&global.initializer, Some(Constant::ZeroInit) | None);
            if !is_bss {
                continue;
            }
            let align = global
                .alignment
                .unwrap_or(global.ty.align_bytes(&self.target)) as u64;
            bss_size = (bss_size + align - 1) & !(align - 1);
            bss_size += global.ty.size_bytes(&self.target) as u64;
        }
        if bss_size > 0 {
            let bss_section = Section {
                name: ".bss".to_string(),
                sh_type: SHT_NOBITS,
                sh_flags: SHF_ALLOC | SHF_WRITE,
                data: Vec::new(),
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: 0,
                logical_size: bss_size,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            elf.add_section(bss_section);
        }

        // Add function symbols.
        for (name, _) in &all_functions {
            let offset = asm_result.symbols.get(name.as_str()).copied().unwrap_or(0);
            elf.add_symbol(ElfSymbol {
                name: name.clone(),
                value: offset,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_index: text_idx as u16,
            });
        }

        // Add external function declaration symbols (undefined references).
        // Mark them as STT_FUNC so the dynamic linker creates PLT entries.
        for decl in module.declarations() {
            elf.add_symbol(ElfSymbol {
                name: decl.name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF
            });
        }

        // Add .rela.text section for relocations.
        if !asm_result.relocations.is_empty() {
            let mut rela_data = Vec::new();
            for reloc in &asm_result.relocations {
                // Elf64_Rela: offset(8) + info(8) + addend(8) = 24 bytes
                rela_data.extend_from_slice(&reloc.offset.to_le_bytes());
                let r_info: u64 = reloc.reloc_type as u64;
                rela_data.extend_from_slice(&r_info.to_le_bytes());
                rela_data.extend_from_slice(&reloc.addend.to_le_bytes());
            }
            let rela_section = Section {
                name: ".rela.text".to_string(),
                sh_type: SHT_RELA,
                sh_flags: SHF_INFO_LINK,
                data: rela_data,
                sh_link: 0,
                sh_info: text_idx as u32,
                sh_addralign: 8,
                sh_entsize: 24,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            elf.add_section(rela_section);
        }

        // .note.GNU-stack section — non-executable stack marker.
        elf.add_section(Section {
            name: ".note.GNU-stack".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: 0,
            data: Vec::new(),
            sh_addralign: 1,
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });

        // Phase 4: Write the ELF object.
        let elf_bytes = elf.write();

        if diagnostics.has_errors() {
            return Err(());
        }

        Ok(elf_bytes)
    }
}

// ============================================================================
// ArchCodegen Trait Implementation
// ============================================================================

impl ArchCodegen for AArch64Codegen {
    /// Lower an IR function to AArch64 machine instructions.
    fn lower_function(
        &self,
        func: &IrFunction,
        _diag: &mut DiagnosticEngine,
        _globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
        global_var_refs: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) -> Result<MachineFunction, String> {
        if !func.is_definition {
            return Err(format!(
                "Cannot lower function '{}': not a definition",
                func.name
            ));
        }

        // Build constant cache from globals (same mechanism as x86-64/i686).
        let mut constant_values = crate::common::fx_hash::FxHashMap::default();
        {
            let mut const_vals: Vec<i64> = Vec::new();
            for gv in _globals {
                if gv.name.starts_with(".Lconst.i.") {
                    if let Some(crate::ir::module::Constant::Integer(v)) = &gv.initializer {
                        const_vals.push(*v as i64);
                    }
                }
            }
            let mut ci = 0;
            for block in &func.blocks {
                for inst in block.instructions() {
                    if let crate::ir::instructions::Instruction::BinOp {
                        result, lhs, rhs, ..
                    } = inst
                    {
                        if *lhs == *result
                            && *rhs == crate::ir::instructions::Value::UNDEF
                            && ci < const_vals.len()
                        {
                            // Add VREG_BASE (64) to key — codegen.rs reads
                            // constant_values keyed by vreg number, which is
                            // Value.index() + 64 to avoid collision with
                            // AArch64 physical register numbers 0–63.
                            constant_values.insert(result.index() + 64, const_vals[ci]);
                            ci += 1;
                        }
                    }
                }
            }
        }

        // Merge constants from func.constant_values (authoritative IR
        // constant map produced during lowering / SSA construction).  This
        // is the definitive per-function constant table and MUST override
        // the heuristic values derived from global `.Lconst.i.*` variables
        // above, because the heuristic can assign wrong constants when the
        // global constant list spans multiple functions.
        for (&val, &imm) in &func.constant_values {
            constant_values.insert(val.index() + 64, imm);
        }

        // Build float constant cache from func.float_constant_values.
        // This maps IR Values representing float constants to their
        // rodata symbol names (.Lconst.f.N) for PC-relative loads.
        let mut float_constant_cache = crate::common::fx_hash::FxHashMap::default();
        if std::env::var("BCC_DEBUG_A64").is_ok() {
            eprintln!(
                "=== FLOAT_CONSTANT_VALUES count={} ===",
                func.float_constant_values.len()
            );
            for (val, (name, fval)) in &func.float_constant_values {
                eprintln!("  float_cv: Value({}) -> {} = {}", val.index(), name, fval);
            }
        }
        for (&val, (name, _fval)) in &func.float_constant_values {
            float_constant_cache.insert(val, name.clone());
        }

        // Fallback: match orphaned float sentinel BinOps to .Lconst.f.* globals
        // (same approach as x86-64 for functions where float_constant_values
        // may not be populated by the lowering phase).
        if float_constant_cache.is_empty() {
            // Collect float constant globals sorted by index.
            let mut float_consts: Vec<(u32, String)> = Vec::new();
            for gv in _globals {
                if let Some(idx_str) = gv.name.strip_prefix(".Lconst.f.") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        float_consts.push((idx, gv.name.clone()));
                    }
                }
            }
            float_consts.sort_by_key(|(idx, _)| *idx);

            // Find float sentinel BinOps (BinOp where is_float type and
            // lhs == result, rhs == UNDEF) and map them to float globals.
            let mut float_sentinel_values: Vec<u32> = Vec::new();
            for block in &func.blocks {
                for inst in block.instructions() {
                    if let crate::ir::instructions::Instruction::BinOp {
                        result,
                        lhs,
                        rhs,
                        ty,
                        ..
                    } = inst
                    {
                        if *lhs == *result
                            && *rhs == crate::ir::instructions::Value::UNDEF
                            && ty.is_float()
                        {
                            float_sentinel_values.push(result.index());
                        }
                    }
                }
            }
            // Map float sentinels to float globals in order.
            for (val_idx, (_gidx, gname)) in float_sentinel_values.iter().zip(float_consts.iter()) {
                float_constant_cache
                    .insert(crate::ir::instructions::Value(*val_idx), gname.clone());
            }
        }

        if std::env::var("BCC_DEBUG_A64").is_ok() {
            eprintln!(
                "=== FINAL float_constant_cache count={} ===",
                float_constant_cache.len()
            );
            for (v, n) in &float_constant_cache {
                eprintln!("  fcc: Value({}) -> {}", v.index(), n);
            }
        }

        // Perform instruction selection: IR → A64 instructions.
        let mut selector = codegen::AArch64InstructionSelector::new(self.target, self.pic_mode);
        selector.set_constant_values(constant_values);
        selector.set_func_ref_names(func_ref_map.clone());
        selector.set_global_var_refs(global_var_refs.clone());
        selector.set_float_constant_cache(float_constant_cache);
        let a64_instructions = selector.select_function(func, &self.abi);

        // DEBUG: Print the A64 instructions for diagnosis
        if std::env::var("BCC_DEBUG_A64").is_ok() {
            eprintln!(
                "=== A64 instructions for '{}' ({} total) ===",
                func.name,
                a64_instructions.len()
            );
            for (i, inst) in a64_instructions.iter().enumerate() {
                eprintln!(
                    "  [{:3}] {:?} rd={:?} rn={:?} rm={:?} imm={} sym={:?} comment={:?}",
                    i, inst.opcode, inst.rd, inst.rn, inst.rm, inst.imm, inst.symbol, inst.comment
                );
            }
            eprintln!("=== func_ref_names: {:?} ===", func_ref_map);
            eprintln!("=== global_var_refs: {:?} ===", global_var_refs);
        }

        // Convert to common MachineInstructions.
        let machine_instructions = Self::convert_instructions(&a64_instructions);

        // Build the MachineFunction with a single entry block.
        let mut mf = MachineFunction::new(func.name.clone());

        // Track whether we have calls to set is_leaf.
        let has_calls = machine_instructions.iter().any(|mi| mi.is_call);
        if has_calls {
            mf.mark_has_calls();
        }

        // Populate the entry block.
        let entry = mf.entry_block_mut();
        for inst in machine_instructions {
            entry.push_instruction(inst);
        }

        // Set frame_size to the locals area size so the prologue allocates
        // sufficient stack space.  The register allocator may increase this
        // later to account for spill slots.
        mf.frame_size = selector.get_locals_size();

        // Record parameter-copy MOV count so the post-register-allocation
        // parallel-move resolver knows how many MOVs at the entry-block
        // head are parameter copies (not regular codegen instructions).
        mf.num_param_moves = selector.get_num_param_moves();

        // Propagate variadic function metadata to MachineFunction so the
        // prologue can save X0-X7 and va_start can compute the correct
        // address of the first unnamed argument.
        if let Some(offset) = selector.get_vararg_save_offset() {
            mf.va_save_area_offset = Some(offset as i32);
            mf.named_gpr_count = selector.get_named_gpr_count();
        }

        // Build vreg → IR Value mapping so that apply_allocation_result can
        // translate codegen virtual register numbers back to the register
        // allocator's Value-indexed assignment table.
        //
        // Codegen uses Value.index() + 64 (VREG_BASE) as vreg numbers to
        // avoid collision with AArch64 physical register IDs 0–63.  The
        // register allocator indexes assignments by IR Value directly, so
        // this mapping bridges the two numbering schemes.
        // Map function parameters first — their SSA Values have
        // corresponding virtual registers (param.value.index() + VREG_BASE)
        // produced by emit_arg_setup.  Without this, apply_allocation_result
        // cannot resolve the parameter MOV instructions' VirtualRegister
        // operands to their allocator-assigned physical registers.
        for param in &func.params {
            if param.value != crate::ir::instructions::Value::UNDEF {
                let vreg_id = param.value.index() + 64; // VREG_BASE
                mf.vreg_to_ir_value.insert(vreg_id, param.value);
            }
        }

        // Map instruction results — each IR instruction that produces a
        // result value has a corresponding virtual register.
        for block in &func.blocks {
            for inst in block.instructions() {
                if let Some(result) = inst.result() {
                    if result != crate::ir::instructions::Value::UNDEF {
                        let vreg_id = result.index() + 64; // VREG_BASE
                        mf.vreg_to_ir_value.insert(vreg_id, result);
                    }
                }
            }
        }

        Ok(mf)
    }

    /// Encode a machine function's instructions to raw AArch64 binary bytes.
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<AssembledFunction, String> {
        let mut asm = assembler::AArch64Assembler::new(self.pic_mode);

        // Collect all instructions from all basic blocks and convert
        // MachineInstruction back to A64Instruction for the assembler.
        let mut all_instructions: Vec<A64Instruction> = Vec::new();
        let debug = std::env::var("BCC_DEBUG_A64").is_ok();
        if debug {
            eprintln!(
                "=== emit_assembly for '{}' blocks={} frame_size={} ===",
                mf.name,
                mf.blocks.len(),
                mf.frame_size
            );
        }
        for block in &mf.blocks {
            if debug {
                eprintln!(
                    "  Block '{}' instructions={}",
                    block.label.as_deref().unwrap_or("?"),
                    block.instructions.len()
                );
            }
            for (idx, mi) in block.instructions.iter().enumerate() {
                if debug {
                    eprintln!("    MI[{:3}] op={} result={:?} operands={:?} is_call={} is_term={} size={} asm_tmpl={:?}",
                        idx, mi.opcode, mi.result, mi.operands, mi.is_call, mi.is_terminator, mi.operand_size, mi.asm_template);
                }

                // Check if this MachineInstruction is a block label pseudo-NOP.
                // These have asm_template = Some("LABEL:labelname") set by
                // convert_instructions. Emit a NOP with a label comment so
                // assemble_one() can define the label at the correct offset.
                if let Some(ref tmpl) = mi.asm_template {
                    if let Some(label) = tmpl.strip_prefix("LABEL:") {
                        all_instructions.push(
                            A64Instruction::new(A64Opcode::NOP).with_comment(format!("{}:", label)),
                        );
                        if debug {
                            eprintln!("    -> LABEL NOP for '{}'", label);
                        }
                        continue; // Don't also emit the regular NOP
                    }
                }

                let a64_inst = Self::mi_to_a64(mi);
                if debug {
                    eprintln!(
                        "    -> A64 {:?} rd={:?} rn={:?} rm={:?} imm={} sym={:?}",
                        a64_inst.opcode,
                        a64_inst.rd,
                        a64_inst.rn,
                        a64_inst.rm,
                        a64_inst.imm,
                        a64_inst.symbol
                    );
                }
                all_instructions.push(a64_inst);
            }
        }

        // ── Parallel-move resolution for call argument setup ──────────
        // After register allocation, sequential MOV instructions that set
        // up call arguments (X0-X7) can clobber source registers.
        // Example: if v7 was allocated to X1, the sequence
        //   MOV X1, X25   (arg 1)
        //   MOV X2, X1    (arg 2 — reads WRONG X1)
        // corrupts arg 2.  We detect such conflicts and reorder the moves,
        // using X16 (IP0 scratch) as a temporary for cycle resolution.
        // NOTE: The register allocator already produces correctly-ordered
        // sequential moves for call arguments, including any necessary
        // shuffle moves to preserve source values.  Running a parallel-move
        // resolution on top of this corrupts the ordering because it
        // treats the regalloc shuffles + arg setup as a single parallel
        // set.  We only invoke the fixup when the regalloc has NOT
        // inserted any resolution shuffles (i.e. when only the raw
        // arg-setup MOVs are present and they conflict).
        //
        // Detection: if there are any MOV_reg with dst > max-arg-reg
        // between the arg-setup block and the CALL, regalloc has
        // already resolved.  For safety, we disable the fixup entirely
        // and let regalloc handle ordering.  If regression tests surface
        // a case where regalloc doesn't resolve correctly, we can
        // re-enable a more surgical fixup.
        // Self::fixup_call_arg_moves(&mut all_instructions);

        let result = asm.assemble_function(&all_instructions)?;

        // Convert assembly relocations to FunctionRelocations for the
        // code generation driver to pass to the linker.
        let relocations = result
            .relocations
            .iter()
            .map(|r| FunctionRelocation {
                offset: r.offset,
                symbol: r.symbol.clone(),
                rel_type_id: r.reloc_type,
                addend: r.addend,
                section: ".text".to_string(),
            })
            .collect();

        Ok(AssembledFunction {
            bytes: result.code,
            relocations,
        })
    }

    /// Returns the target architecture: `Target::AArch64`.
    #[inline]
    fn target(&self) -> Target {
        Target::AArch64
    }

    /// Returns the AArch64 register file information.
    fn register_info(&self) -> RegisterInfo {
        self.reg_info.register_info()
    }

    /// Returns AArch64 relocation type descriptors.
    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        &self.relocation_types
    }

    /// Generate AArch64 function prologue instructions.
    ///
    /// ## AArch64 Prologue Pattern
    ///
    /// ```text
    /// stp x29, x30, [sp, #-frame_size]!  // Save FP+LR, pre-decrement SP
    /// mov x29, sp                          // Set frame pointer
    /// stp x19, x20, [sp, #offset]         // Save callee-saved GPR pair
    /// stp d8, d9, [sp, #fp_offset]         // Save callee-saved FP pair
    /// ```
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();
        let sp = registers::SP_REG as u16;
        let fp = registers::FP_REG as u16;
        let lr = registers::LR as u16;

        // Compute total frame size:
        //   16 bytes for FP + LR (stored at [SP, #0])
        //   + 16 bytes per pair of callee-saved registers
        //   + mf.frame_size for local spills/temporaries
        let num_gpr = mf.callee_saved_regs.iter().filter(|&&r| r < 32).count();
        let num_fpr = mf.callee_saved_regs.iter().filter(|&&r| r >= 32).count();
        let gpr_pairs = (num_gpr + 1) / 2; // round up for odd count
        let fpr_pairs = (num_fpr + 1) / 2;
        let callee_save_bytes: usize = (gpr_pairs + fpr_pairs) * 16;
        let frame_size = Self::align_up((16 + callee_save_bytes + mf.frame_size).max(16), 16);

        // Step 1 & 2: Allocate frame and save FP/LR.
        //
        // The STP pre-index form uses a 7-bit signed offset scaled by 8,
        // giving a range of −504 to +504.  For frames ≤ 504 we emit:
        //     STP X29, X30, [SP, #-frame_size]!
        // For larger frames we split the allocation:
        //     SUB SP, SP, #frame_size        (handles up to 4095)
        //     STP X29, X30, [SP]             (offset-form at [SP, #0])
        // Note: frames > 4095 handled via the ADD/SUB negative-immediate
        // flip already implemented in the encoder.
        let stp_pre_limit = 504usize; // max magnitude for 7-bit signed × 8
        if frame_size <= stp_pre_limit {
            // Small frame: use STP pre-index.
            let mut mi = MachineInstruction::new(OPCODE_STP_PRE);
            mi.operands.push(MachineOperand::Register(fp));
            mi.operands.push(MachineOperand::Register(lr));
            mi.operands.push(MachineOperand::Memory {
                base: Some(sp),
                index: None,
                scale: 1,
                displacement: -(frame_size as i64),
            });
            mi.operand_size = 8;
            prologue.push(mi);
        } else {
            // Large frame: SUB SP first, then STP at [SP, #0].
            {
                let mut mi = MachineInstruction::new(OPCODE_SUB_IMM);
                mi.result = Some(MachineOperand::Register(sp));
                mi.operands.push(MachineOperand::Register(sp));
                mi.operands
                    .push(MachineOperand::Immediate(frame_size as i64));
                mi.operand_size = 8;
                prologue.push(mi);
            }
            {
                let mut mi = MachineInstruction::new(OPCODE_STP);
                mi.operands.push(MachineOperand::Register(fp));
                mi.operands.push(MachineOperand::Register(lr));
                mi.operands.push(MachineOperand::Memory {
                    base: Some(sp),
                    index: None,
                    scale: 1,
                    displacement: 0,
                });
                mi.operand_size = 8;
                prologue.push(mi);
            }
        }

        // Establish frame pointer: MOV X29, SP.
        // On AArch64, SP (register 31) cannot be used in ORR-based MOV.
        // MOV Xd, SP is encoded as ADD Xd, SP, #0.
        {
            let mut mi = MachineInstruction::new(OPCODE_ADD_IMM);
            mi.result = Some(MachineOperand::Register(fp));
            mi.operands.push(MachineOperand::Register(sp));
            mi.operands.push(MachineOperand::Immediate(0));
            mi.operand_size = 8;
            prologue.push(mi);
        }

        // For variadic functions: save X0-X7 to the vararg save area at
        // [FP+16..FP+80].  va_start later points to [FP + 16 + named*8].
        if mf.va_save_area_offset.is_some() {
            // Save X0-X7 as 4 STP pairs.
            let arg_regs: [(u16, u16); 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];
            let mut off = 16i64;
            for &(r1, r2) in &arg_regs {
                let mut mi = MachineInstruction::new(OPCODE_STP);
                mi.operands.push(MachineOperand::Register(r1));
                mi.operands.push(MachineOperand::Register(r2));
                mi.operands.push(MachineOperand::Memory {
                    base: Some(fp),
                    index: None,
                    scale: 1,
                    displacement: off,
                });
                mi.operand_size = 8;
                prologue.push(mi);
                off += 16;
            }
        }

        // Step 3: Save callee-saved registers in pairs.
        let mut gpr_saved: Vec<u16> = Vec::new();
        let mut fpr_saved: Vec<u16> = Vec::new();

        for &reg in &mf.callee_saved_regs {
            if reg < 32 {
                gpr_saved.push(reg);
            } else {
                fpr_saved.push(reg);
            }
        }

        // Save GPRs in pairs.
        // Callee-saved regs are placed AFTER the locals area:
        //   [FP + 16 + mf.frame_size]  = first callee-saved pair
        //
        // The STP signed-offset form uses a 7-bit signed immediate
        // scaled by 8, giving a range of [-512, +504].  When the
        // callee-saved offset exceeds this range we use IP0 (X16) —
        // which is reserved and never allocated — to hold the target
        // address.
        let mut offset = 16i64 + mf.frame_size as i64;
        let stp_max: i64 = 504;
        let scratch = registers::IP0 as u16;
        let mut i = 0;
        while i + 1 < gpr_saved.len() {
            Self::emit_stp_large_offset(
                &mut prologue,
                gpr_saved[i],
                gpr_saved[i + 1],
                sp,
                offset,
                false,
                scratch,
                stp_max,
            );
            offset += 16;
            i += 2;
        }
        if i < gpr_saved.len() {
            Self::emit_stp_large_offset(
                &mut prologue,
                gpr_saved[i],
                gpr_saved[i],
                sp,
                offset,
                false,
                scratch,
                stp_max,
            );
            offset += 16;
        }

        // Save FPRs in pairs.
        let mut j = 0;
        while j + 1 < fpr_saved.len() {
            Self::emit_stp_large_offset(
                &mut prologue,
                fpr_saved[j],
                fpr_saved[j + 1],
                sp,
                offset,
                true,
                scratch,
                stp_max,
            );
            offset += 16;
            j += 2;
        }
        if j < fpr_saved.len() {
            Self::emit_stp_large_offset(
                &mut prologue,
                fpr_saved[j],
                fpr_saved[j],
                sp,
                offset,
                true,
                scratch,
                stp_max,
            );
        }

        prologue
    }

    /// Generate AArch64 function epilogue instructions.
    ///
    /// Restores callee-saved registers in reverse order, then returns.
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();
        let sp = registers::SP_REG as u16;
        let fp = registers::FP_REG as u16;
        let lr = registers::LR as u16;

        // Must match the prologue calculation exactly.
        let num_gpr = mf.callee_saved_regs.iter().filter(|&&r| r < 32).count();
        let num_fpr = mf.callee_saved_regs.iter().filter(|&&r| r >= 32).count();
        let gpr_pairs = (num_gpr + 1) / 2;
        let fpr_pairs = (num_fpr + 1) / 2;
        let callee_save_bytes: usize = (gpr_pairs + fpr_pairs) * 16;
        let frame_size = Self::align_up((16 + callee_save_bytes + mf.frame_size).max(16), 16);
        let stp_pre_limit = 504usize;

        // Classify callee-saved registers.
        let mut gpr_saved: Vec<u16> = Vec::new();
        let mut fpr_saved: Vec<u16> = Vec::new();
        for &reg in &mf.callee_saved_regs {
            if reg < 32 {
                gpr_saved.push(reg);
            } else {
                fpr_saved.push(reg);
            }
        }

        // Compute offsets matching prologue layout.
        // Callee-saved regs are placed AFTER the locals area:
        //   [FP + 16 + mf.frame_size]  = first callee-saved pair
        let mut offset = 16i64 + mf.frame_size as i64;
        let mut gpr_offsets: Vec<(usize, i64)> = Vec::new();
        let mut i = 0;
        while i + 1 < gpr_saved.len() {
            gpr_offsets.push((i, offset));
            offset += 16;
            i += 2;
        }
        if i < gpr_saved.len() {
            gpr_offsets.push((i, offset));
            offset += 16;
        }

        let mut fpr_offsets: Vec<(usize, i64)> = Vec::new();
        let mut j = 0;
        while j + 1 < fpr_saved.len() {
            fpr_offsets.push((j, offset));
            offset += 16;
            j += 2;
        }
        if j < fpr_saved.len() {
            fpr_offsets.push((j, offset));
        }

        // Restore callee-saved registers in reverse order.
        // Use the large-offset helper so that offsets exceeding the
        // LDP 7-bit signed range fall back to ADD+LDP via IP0.
        let ldp_max: i64 = 504;
        let scratch = registers::IP0 as u16;

        // Restore FPRs in reverse order.
        for &(idx, off) in fpr_offsets.iter().rev() {
            if idx + 1 < fpr_saved.len() {
                Self::emit_ldp_large_offset(
                    &mut epilogue,
                    fpr_saved[idx],
                    fpr_saved[idx + 1],
                    sp,
                    off,
                    true,
                    scratch,
                    ldp_max,
                );
            } else {
                Self::emit_ldp_large_offset(
                    &mut epilogue,
                    fpr_saved[idx],
                    fpr_saved[idx],
                    sp,
                    off,
                    true,
                    scratch,
                    ldp_max,
                );
            }
        }

        // Restore GPRs in reverse order.
        for &(idx, off) in gpr_offsets.iter().rev() {
            if idx + 1 < gpr_saved.len() {
                Self::emit_ldp_large_offset(
                    &mut epilogue,
                    gpr_saved[idx],
                    gpr_saved[idx + 1],
                    sp,
                    off,
                    false,
                    scratch,
                    ldp_max,
                );
            } else {
                Self::emit_ldp_large_offset(
                    &mut epilogue,
                    gpr_saved[idx],
                    gpr_saved[idx],
                    sp,
                    off,
                    false,
                    scratch,
                    ldp_max,
                );
            }
        }

        // Restore FP/LR and deallocate frame.
        //
        // Mirror the prologue: if the frame fitted in the STP pre-index
        // form (≤ 504 bytes), we can use LDP post-index symmetrically.
        // Otherwise, LDP at [SP, #0] then ADD SP to restore.
        if frame_size <= stp_pre_limit {
            // Small frame: LDP post-index restores and adjusts SP.
            let mut mi = MachineInstruction::new(OPCODE_LDP_POST);
            mi.result = Some(MachineOperand::Register(fp));
            mi.operands.push(MachineOperand::Register(lr));
            mi.operands.push(MachineOperand::Memory {
                base: Some(sp),
                index: None,
                scale: 1,
                displacement: frame_size as i64,
            });
            mi.operand_size = 8;
            epilogue.push(mi);
        } else {
            // Large frame: LDP at [SP, #0] then ADD SP, SP, #frame_size.
            {
                let mut mi = MachineInstruction::new(OPCODE_LDP);
                mi.result = Some(MachineOperand::Register(fp));
                mi.operands.push(MachineOperand::Register(lr));
                mi.operands.push(MachineOperand::Memory {
                    base: Some(sp),
                    index: None,
                    scale: 1,
                    displacement: 0,
                });
                mi.operand_size = 8;
                epilogue.push(mi);
            }
            {
                let mut mi = MachineInstruction::new(OPCODE_ADD_IMM);
                mi.result = Some(MachineOperand::Register(sp));
                mi.operands.push(MachineOperand::Register(sp));
                mi.operands
                    .push(MachineOperand::Immediate(frame_size as i64));
                mi.operand_size = 8;
                epilogue.push(mi);
            }
        }

        // Return via X30 (LR).
        epilogue.push(Self::make_ret());

        epilogue
    }

    /// Returns the physical register index of the frame pointer (X29).
    #[inline]
    fn frame_pointer_reg(&self) -> u16 {
        registers::FP_REG as u16
    }

    /// Returns the physical register index of the stack pointer (SP = 31).
    #[inline]
    fn stack_pointer_reg(&self) -> u16 {
        registers::SP_REG as u16
    }

    /// Returns the physical register index of the return address register (X30/LR).
    #[inline]
    fn return_address_reg(&self) -> Option<u16> {
        Some(registers::LR as u16)
    }

    /// Classify where a function argument should be placed per AAPCS64.
    ///
    /// Delegates to the AAPCS64 batch classifier (`abi::classify_arguments_ir`)
    /// with a single-element slice, matching the pattern used by x86_64. This
    /// ensures correct register advancement when multiple arguments are
    /// classified sequentially via separate calls — each isolated call returns
    /// the first available register (X0 or V0) for that single argument.
    fn classify_argument(&self, ty: &IrType) -> ArgLocation {
        let locs = classify_arguments_ir(std::slice::from_ref(ty), &self.target);
        locs.into_iter().next().unwrap_or(ArgLocation::Stack(0))
    }

    /// Classify where a function return value should be placed per AAPCS64.
    fn classify_return(&self, ty: &IrType) -> ArgLocation {
        match ty {
            IrType::Void => ArgLocation::Stack(0),
            IrType::F32 | IrType::F64 | IrType::F80 => ArgLocation::Register(registers::V0 as u16),
            IrType::Struct(_fields) => {
                let size = ty.size_bytes(&self.target);
                if size <= 8 {
                    ArgLocation::Register(registers::X0 as u16)
                } else if size <= 16 {
                    ArgLocation::RegisterPair(registers::X0 as u16, registers::X1 as u16)
                } else {
                    // Large return: indirect via X8.
                    ArgLocation::Register(registers::X8 as u16)
                }
            }
            _ => ArgLocation::Register(registers::X0 as u16),
        }
    }
}

// ============================================================================
// MachineInstruction ↔ A64Instruction Conversion
// ============================================================================

impl AArch64Codegen {
    /// Convert a `MachineInstruction` back to an `A64Instruction` for
    /// the assembler.
    fn mi_to_a64(mi: &MachineInstruction) -> A64Instruction {
        let mut inst = A64Instruction::new(A64Opcode::NOP);

        // Map the opcode back to an A64Opcode.
        inst.opcode = Self::u32_to_a64_opcode(mi.opcode);

        // --- Special handling for STP/LDP family -------------------------
        // The encoder expects: rd=Rt, rm=Rt2, rn=Rn(base), imm=offset.
        // Our MachineInstruction stores them as:
        //   operands = [Register(rt1), Register(rt2), Memory{base, displacement}]
        // We must map: rd=rt1, rm=rt2, rn=base, imm=displacement.
        match mi.opcode {
            x if x == OPCODE_STP
                || x == OPCODE_STP_PRE
                || x == OPCODE_STP_FP
                || x == OPCODE_LDP_FP =>
            {
                // STP / STP_pre: operands[0]=Rt1, operands[1]=Rt2, operands[2]=Memory
                let rt1 = mi
                    .operands
                    .first()
                    .and_then(|o| o.as_register())
                    .unwrap_or(0) as u32;
                let rt2 = mi
                    .operands
                    .get(1)
                    .and_then(|o| o.as_register())
                    .unwrap_or(0) as u32;
                let (base, disp) = Self::extract_memory_operand(&mi.operands);
                inst.rd = Some(rt1);
                inst.rm = Some(rt2);
                inst.rn = Some(base);
                inst.imm = disp;
                inst.is_32bit = mi.operand_size == 4;
                return inst;
            }
            x if x == OPCODE_LDP || x == OPCODE_LDP_POST => {
                // LDP / LDP_post: result=Rt1, operands[0]=Rt2, operands[1]=Memory
                let rt1 = mi
                    .result
                    .as_ref()
                    .and_then(|o| o.as_register())
                    .unwrap_or(0) as u32;
                let rt2 = mi
                    .operands
                    .first()
                    .and_then(|o| o.as_register())
                    .unwrap_or(0) as u32;
                let (base, disp) = Self::extract_memory_operand(&mi.operands);
                inst.rd = Some(rt1);
                inst.rm = Some(rt2);
                inst.rn = Some(base);
                inst.imm = disp;
                inst.is_32bit = mi.operand_size == 4;
                return inst;
            }
            _ => {}
        }

        // --- Special handling for STR_imm / STRB_imm / STRH_imm spill stores ----
        // Spill stores from generation.rs come with:
        //   result = None
        //   operands = [Memory{base, displacement}, Register(source)]
        // The generic mapping would place the Register operand into `rn`,
        // overwriting the base from Memory.  STR encoding expects:
        //   rd = source register (Rt), rn = base register, imm = offset.
        {
            let opc = mi.opcode;
            if (opc == A64Opcode::STR_imm as u32
                || opc == A64Opcode::STRB_imm as u32
                || opc == A64Opcode::STRH_imm as u32
                || opc == A64Opcode::STR_fp_imm as u32)
                && mi.result.is_none()
                && mi.operands.len() >= 2
            {
                let (base, disp) = Self::extract_memory_operand(&mi.operands);
                inst.rn = Some(base);
                inst.imm = disp;
                // Find the Register operand → this is the source (Rt → rd).
                for op in &mi.operands {
                    if let Some(reg) = op.as_any_register() {
                        inst.rd = Some(reg as u32);
                        break;
                    }
                }
                inst.is_32bit = mi.operand_size == 4;
                return inst;
            }
        }

        // --- Special handling for MOV_reg, NEG_reg, MVN_reg --------------
        // These pseudo-instructions encode using the `rm` field:
        //   MOV Xd, Xm → ORR Xd, XZR, Xm   (source in rm)
        //   NEG Xd, Xm → SUB Xd, XZR, Xm   (source in rm)
        //   MVN Xd, Xm → ORN Xd, XZR, Xm   (source in rm)
        // The MachineInstruction stores: result=Rd, operands[0]=Rm.
        // We must place the source in `rm`, not `rn` (generic default).
        {
            let opc = mi.opcode;
            if opc == A64Opcode::MOV_reg as u32
                || opc == A64Opcode::NEG_reg as u32
                || opc == A64Opcode::MVN_reg as u32
            {
                if let Some(ref result) = mi.result {
                    if let Some(reg) = result.as_any_register() {
                        inst.rd = Some(reg as u32);
                    }
                }
                // First Register operand → rm (source for ORR/SUB/ORN)
                if let Some(src) = mi.operands.first().and_then(|o| o.as_any_register()) {
                    inst.rm = Some(src as u32);
                }
                inst.is_32bit = mi.operand_size == 4;
                return inst;
            }
        }

        // --- Generic operand mapping for non-STP/LDP instructions --------
        // Map result as destination register.
        if let Some(ref result) = mi.result {
            if let Some(reg) = result.as_any_register() {
                inst.rd = Some(reg as u32);
            }
        }

        // Map input operands.
        let mut reg_idx = 0;
        for op in &mi.operands {
            match op {
                MachineOperand::Register(r) => {
                    match reg_idx {
                        0 => inst.rn = Some(*r as u32),
                        1 => inst.rm = Some(*r as u32),
                        2 => inst.ra = Some(*r as u32),
                        _ => {}
                    }
                    reg_idx += 1;
                }
                MachineOperand::VirtualRegister(v) => {
                    // Surviving virtual register — treat identically
                    // to a physical register (may occur if allocation
                    // left some unresolved).
                    let r = *v;
                    match reg_idx {
                        0 => inst.rn = Some(r),
                        1 => inst.rm = Some(r),
                        2 => inst.ra = Some(r),
                        _ => {}
                    }
                    reg_idx += 1;
                }
                MachineOperand::Immediate(imm) => {
                    inst.imm = *imm;
                }
                MachineOperand::GlobalSymbol(ref name) => {
                    inst.symbol = Some(name.clone());
                }
                MachineOperand::Memory {
                    base, displacement, ..
                } => {
                    if let Some(b) = base {
                        if inst.rn.is_none() {
                            inst.rn = Some(*b as u32);
                        }
                    }
                    inst.imm = *displacement;
                }
                _ => {}
            }
        }

        // Set the 32-bit flag from operand_size.
        inst.is_32bit = mi.operand_size == 4;

        // Restore condition code from the MachineInstruction.
        // This is critical for CSET, B_cond, CSEL, CSINC, etc.
        if let Some(cc_enc) = mi.cond_code {
            inst.cond = Some(codegen::CondCode::from_encoding(cc_enc));
        }

        // Restore shift/rotate encoding from the MachineInstruction.
        // Critical for MOVZ/MOVK/MOVN halfword shifts and register-
        // shifted ALU instructions.
        inst.shift = mi.arch_shift;

        inst
    }

    /// Extract base register and displacement from an operand list containing
    /// a `Memory` operand.
    fn extract_memory_operand(operands: &[MachineOperand]) -> (u32, i64) {
        for op in operands {
            if let MachineOperand::Memory {
                base, displacement, ..
            } = op
            {
                return (base.unwrap_or(31) as u32, *displacement);
            }
        }
        (31, 0) // default: SP, offset 0
    }

    // ================================================================
    // Parallel-move resolution for call argument setup
    // ================================================================

    /// Resolve register conflicts in the argument-setup MOV sequence
    /// that precedes every CALL instruction.  After register allocation
    /// a later MOV may read a register that an earlier MOV already
    /// clobbered.  We detect this pattern and reorder (or insert a
    /// temporary via X16/IP0) to eliminate the conflict.
    #[allow(dead_code)]
    fn fixup_call_arg_moves(insts: &mut Vec<A64Instruction>) {
        // AArch64 IP0 — intra-procedure-call scratch, never used by
        // the register allocator and safe to clobber between calls.
        const TEMP_REG: u32 = 16; // X16

        // Phase 1 — collect every CALL site that has a conflicting
        // argument-register move sequence.
        struct CallFixup {
            call_idx: usize,
            mov_indices: Vec<usize>, // indices into `insts` (execution order)
            resolved: Vec<(u32, u32, bool)>, // (dst, src, is_32bit)
        }
        let mut fixups: Vec<CallFixup> = Vec::new();

        for i in 0..insts.len() {
            if insts[i].opcode != A64Opcode::CALL {
                continue;
            }
            // Walk backward to collect MOV_reg with arg-register destinations.
            let mut mov_indices: Vec<usize> = Vec::new();
            let mut j = i;
            while j > 0 {
                j -= 1;
                if insts[j].opcode == A64Opcode::MOV_reg {
                    if let Some(dst) = insts[j].rd {
                        if dst <= 7 {
                            mov_indices.push(j);
                            continue;
                        }
                    }
                }
                // Allow non-MOV instructions that are commonly interleaved
                // with the argument setup (address materialisation, immediates,
                // labels, FP moves, stack stores).
                match insts[j].opcode {
                    A64Opcode::ADRP
                    | A64Opcode::ADD_imm
                    | A64Opcode::MOVZ
                    | A64Opcode::MOVK
                    | A64Opcode::NOP
                    | A64Opcode::FMOV_d
                    | A64Opcode::STR_imm => continue,
                    _ => break,
                }
            }
            if mov_indices.len() < 2 {
                continue;
            }
            mov_indices.reverse(); // execution order (first emitted → first)

            // Build (dst, src, is_32bit) triples.
            // The instruction selector stores the MOV source in `rn`
            // (via `.with_rn(src)`), but after register allocation the
            // physical register may appear in either `rn` or `rm`
            // depending on the ORR encoding used.  Prefer `rn` (the
            // field set by the instruction selector) and fall back to
            // `rm` only if `rn` is absent.
            let moves: Vec<(u32, u32, bool)> = mov_indices
                .iter()
                .map(|&idx| {
                    let dst = insts[idx].rd.unwrap();
                    let src = insts[idx].rn.or(insts[idx].rm).unwrap_or(0);
                    let w = insts[idx].is_32bit;
                    (dst, src, w)
                })
                .collect();

            // Conflict detection: does any move[k].src equal an earlier
            // move[l].dst (l < k)?
            let has_conflict = (1..moves.len()).any(|k| {
                let src_k = moves[k].1;
                (0..k).any(|l| moves[l].0 == src_k)
            });
            if !has_conflict {
                continue;
            }

            let resolved = Self::resolve_parallel_moves(&moves, TEMP_REG);
            fixups.push(CallFixup {
                call_idx: i,
                mov_indices,
                resolved,
            });
        }

        if fixups.is_empty() {
            return;
        }

        // Phase 2 — build a new instruction vector with the original
        // conflicting MOVs replaced by the resolved sequence.
        let skip: std::collections::HashSet<usize> = fixups
            .iter()
            .flat_map(|f| f.mov_indices.iter().copied())
            .collect();

        let mut new_insts = Vec::with_capacity(insts.len() + fixups.len() * 2);
        let mut fix_idx = 0;

        for (i, inst) in insts.iter().enumerate() {
            if skip.contains(&i) {
                continue; // replaced by resolved sequence
            }
            // Insert the resolved moves immediately before the CALL.
            if fix_idx < fixups.len() && i == fixups[fix_idx].call_idx {
                for &(dst, src, w) in &fixups[fix_idx].resolved {
                    let mut mov = A64Instruction::new(A64Opcode::MOV_reg);
                    mov.rd = Some(dst);
                    mov.rm = Some(src);
                    mov.is_32bit = w;
                    new_insts.push(mov);
                }
                fix_idx += 1;
            }
            new_insts.push(inst.clone());
        }

        *insts = new_insts;
    }

    /// Topological parallel-move resolution.
    ///
    /// Given a set of simultaneous register moves `(dst, src)`, produce
    /// a sequential ordering that preserves all source values.  If a
    /// cycle exists (e.g. swap X0↔X1), break it with `temp` (X16/IP0).
    #[allow(dead_code)]
    fn resolve_parallel_moves(moves: &[(u32, u32, bool)], temp: u32) -> Vec<(u32, u32, bool)> {
        // Remove self-moves (dst == src) — they are no-ops.
        let mut pending: Vec<(u32, u32, bool)> =
            moves.iter().copied().filter(|&(d, s, _)| d != s).collect();
        let mut result: Vec<(u32, u32, bool)> = Vec::with_capacity(pending.len() + 1);

        while !pending.is_empty() {
            // Find a move whose destination is NOT a source of any other
            // pending move — it is safe to emit because nothing else
            // still needs to read from `dst`.
            let ready = (0..pending.len()).find(|&i| {
                let d = pending[i].0;
                !pending
                    .iter()
                    .enumerate()
                    .any(|(j, &(_, s, _))| j != i && s == d)
            });

            if let Some(idx) = ready {
                result.push(pending.remove(idx));
            } else {
                // Every remaining move's destination is someone else's
                // source → cycle.  Break it by saving one source to temp.
                let (_d, s, w) = pending[0];
                result.push((temp, s, w)); // MOV temp, src
                pending[0].1 = temp; // rewrite: MOV dst, temp
            }
        }
        result
    }

    fn u32_to_a64_opcode(opcode: u32) -> A64Opcode {
        match opcode {
            x if x == OPCODE_STP => A64Opcode::STP,
            x if x == OPCODE_STP_PRE => A64Opcode::STP_pre,
            x if x == OPCODE_LDP => A64Opcode::LDP,
            x if x == OPCODE_LDP_POST => A64Opcode::LDP_post,
            x if x == OPCODE_MOV_REG => A64Opcode::ORR_reg,
            x if x == OPCODE_RET => A64Opcode::RET,
            x if x == OPCODE_SUB_IMM => A64Opcode::SUB_imm,
            x if x == OPCODE_ADD_IMM => A64Opcode::ADD_imm,
            x if x == OPCODE_STP_FP => A64Opcode::STP_fp,
            x if x == OPCODE_LDP_FP => A64Opcode::LDP_fp,
            other => Self::discriminant_to_opcode(other),
        }
    }

    /// Map a discriminant value back to an A64Opcode via linear scan.
    ///
    /// This list MUST contain every variant of `A64Opcode` so that the
    /// round-trip `A64Instruction → MachineInstruction → A64Instruction`
    /// is lossless. Missing variants silently degrade to NOP.
    fn discriminant_to_opcode(d: u32) -> A64Opcode {
        let opcodes: &[A64Opcode] = &[
            // Data Processing — Immediate
            A64Opcode::ADD_imm,
            A64Opcode::ADDS_imm,
            A64Opcode::SUB_imm,
            A64Opcode::SUBS_imm,
            A64Opcode::MOVZ,
            A64Opcode::MOVK,
            A64Opcode::MOVN,
            A64Opcode::AND_imm,
            A64Opcode::ORR_imm,
            A64Opcode::EOR_imm,
            A64Opcode::ANDS_imm,
            A64Opcode::ADRP,
            A64Opcode::ADR,
            // Data Processing — Register
            A64Opcode::ADD_reg,
            A64Opcode::ADDS_reg,
            A64Opcode::SUB_reg,
            A64Opcode::SUBS_reg,
            A64Opcode::AND_reg,
            A64Opcode::ORR_reg,
            A64Opcode::EOR_reg,
            A64Opcode::ANDS_reg,
            A64Opcode::ORN_reg,
            A64Opcode::BIC_reg,
            A64Opcode::MADD,
            A64Opcode::MSUB,
            A64Opcode::SMULH,
            A64Opcode::UMULH,
            A64Opcode::SDIV,
            A64Opcode::UDIV,
            A64Opcode::LSL_reg,
            A64Opcode::LSR_reg,
            A64Opcode::ASR_reg,
            A64Opcode::ROR_reg,
            A64Opcode::LSL_imm,
            A64Opcode::LSR_imm,
            A64Opcode::ASR_imm,
            // Conditional Select
            A64Opcode::CSEL,
            A64Opcode::CSINC,
            A64Opcode::CSINV,
            A64Opcode::CSNEG,
            A64Opcode::CSET,
            A64Opcode::CSETM,
            // Bit Manipulation
            A64Opcode::CLZ,
            A64Opcode::CLS,
            A64Opcode::RBIT,
            A64Opcode::REV,
            A64Opcode::REV16,
            A64Opcode::REV32,
            A64Opcode::EXTR,
            A64Opcode::BFM,
            A64Opcode::SBFM,
            A64Opcode::UBFM,
            A64Opcode::SXTB,
            A64Opcode::SXTH,
            A64Opcode::SXTW,
            A64Opcode::UXTB,
            A64Opcode::UXTH,
            // Loads
            A64Opcode::LDR_imm,
            A64Opcode::LDRB_imm,
            A64Opcode::LDRH_imm,
            A64Opcode::LDRSB_imm,
            A64Opcode::LDRSH_imm,
            A64Opcode::LDRSW_imm,
            A64Opcode::LDR_reg,
            A64Opcode::LDR_literal,
            A64Opcode::LDP,
            A64Opcode::LDP_post,
            A64Opcode::LDPSW,
            A64Opcode::LDR_pre,
            A64Opcode::LDR_post,
            // Stores
            A64Opcode::STR_imm,
            A64Opcode::STRB_imm,
            A64Opcode::STRH_imm,
            A64Opcode::STR_reg,
            A64Opcode::STP,
            A64Opcode::STP_pre,
            A64Opcode::STR_pre,
            A64Opcode::STR_post,
            // FP/SIMD Loads and Stores
            A64Opcode::LDR_fp_imm,
            A64Opcode::STR_fp_imm,
            A64Opcode::LDP_fp,
            A64Opcode::STP_fp,
            // Branches
            A64Opcode::B,
            A64Opcode::BL,
            A64Opcode::B_cond,
            A64Opcode::BR,
            A64Opcode::BLR,
            A64Opcode::RET,
            A64Opcode::CBZ,
            A64Opcode::CBNZ,
            A64Opcode::TBZ,
            A64Opcode::TBNZ,
            // Comparison aliases
            A64Opcode::CMP_imm,
            A64Opcode::CMP_reg,
            A64Opcode::CMN_imm,
            A64Opcode::CMN_reg,
            A64Opcode::TST_imm,
            A64Opcode::TST_reg,
            A64Opcode::CCMP_imm,
            A64Opcode::CCMP_reg,
            // FP Data Processing
            A64Opcode::FADD_s,
            A64Opcode::FSUB_s,
            A64Opcode::FMUL_s,
            A64Opcode::FDIV_s,
            A64Opcode::FSQRT_s,
            A64Opcode::FADD_d,
            A64Opcode::FSUB_d,
            A64Opcode::FMUL_d,
            A64Opcode::FDIV_d,
            A64Opcode::FSQRT_d,
            A64Opcode::FNEG_s,
            A64Opcode::FNEG_d,
            A64Opcode::FABS_s,
            A64Opcode::FABS_d,
            A64Opcode::FMADD_s,
            A64Opcode::FMSUB_s,
            A64Opcode::FNMADD_s,
            A64Opcode::FNMSUB_s,
            A64Opcode::FMADD_d,
            A64Opcode::FMSUB_d,
            A64Opcode::FNMADD_d,
            A64Opcode::FNMSUB_d,
            A64Opcode::FMIN_s,
            A64Opcode::FMAX_s,
            A64Opcode::FMIN_d,
            A64Opcode::FMAX_d,
            // FP Comparison
            A64Opcode::FCMP_s,
            A64Opcode::FCMP_d,
            A64Opcode::FCMPE_s,
            A64Opcode::FCMPE_d,
            // FP Conversion
            A64Opcode::FCVT_sd,
            A64Opcode::FCVT_ds,
            A64Opcode::FCVTZS_ws,
            A64Opcode::FCVTZS_xs,
            A64Opcode::FCVTZS_wd,
            A64Opcode::FCVTZS_xd,
            A64Opcode::FCVTZU_ws,
            A64Opcode::FCVTZU_xs,
            A64Opcode::FCVTZU_wd,
            A64Opcode::FCVTZU_xd,
            A64Opcode::SCVTF_sw,
            A64Opcode::SCVTF_sx,
            A64Opcode::SCVTF_dw,
            A64Opcode::SCVTF_dx,
            A64Opcode::UCVTF_sw,
            A64Opcode::UCVTF_sx,
            A64Opcode::UCVTF_dw,
            A64Opcode::UCVTF_dx,
            // FP Move
            A64Opcode::FMOV_s,
            A64Opcode::FMOV_d,
            A64Opcode::FMOV_gen_to_fp,
            A64Opcode::FMOV_fp_to_gen,
            // System
            A64Opcode::NOP,
            A64Opcode::DMB,
            A64Opcode::DSB,
            A64Opcode::ISB,
            A64Opcode::SVC,
            A64Opcode::HVC,
            A64Opcode::SMC,
            A64Opcode::MRS,
            A64Opcode::MSR,
            // Pseudo-instructions
            A64Opcode::MOV_reg,
            A64Opcode::MOV_imm,
            A64Opcode::NEG_reg,
            A64Opcode::MVN_reg,
            A64Opcode::LI,
            A64Opcode::LA,
            A64Opcode::CALL,
            A64Opcode::INLINE_ASM,
            A64Opcode::CNT,
            A64Opcode::ADDV,
            A64Opcode::MUL,
            A64Opcode::SMULL,
        ];

        for &op in opcodes {
            if op as u32 == d {
                return op;
            }
        }

        // Defensive fallback for unrecognized opcodes.
        A64Opcode::NOP
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_new() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.target(), Target::AArch64);
        assert!(!codegen.pic_mode);
        assert!(!codegen.debug_info);
    }

    #[test]
    fn test_codegen_new_pic() {
        let codegen = AArch64Codegen::new(true, true);
        assert!(codegen.pic_mode);
        assert!(codegen.debug_info);
        assert_eq!(codegen.target(), Target::AArch64);
    }

    #[test]
    fn test_frame_pointer_reg() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.frame_pointer_reg(), 29);
        assert_eq!(codegen.frame_pointer_reg(), registers::FP_REG as u16);
    }

    #[test]
    fn test_stack_pointer_reg() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.stack_pointer_reg(), 31);
        assert_eq!(codegen.stack_pointer_reg(), registers::SP_REG as u16);
    }

    #[test]
    fn test_return_address_reg() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.return_address_reg(), Some(30));
        assert_eq!(codegen.return_address_reg(), Some(registers::LR as u16));
    }

    #[test]
    fn test_register_info() {
        let codegen = AArch64Codegen::new(false, false);
        let info = codegen.register_info();
        assert_eq!(info.allocatable_gpr.len(), 27); // X16 (IP0), X17 (IP1) reserved as scratch
        assert_eq!(info.allocatable_fpr.len(), 32);
        assert_eq!(info.callee_saved.len(), 18);
        assert_eq!(info.argument_gpr.len(), 8);
        assert_eq!(info.argument_fpr.len(), 8);
        assert_eq!(info.return_gpr.len(), 2);
        assert_eq!(info.return_fpr.len(), 4);
    }

    #[test]
    fn test_relocation_types() {
        let codegen = AArch64Codegen::new(false, false);
        let relocs = codegen.relocation_types();
        assert!(relocs.len() >= 31);

        assert_eq!(relocs[0].name, "R_AARCH64_NONE");
        assert_eq!(relocs[0].type_id, 0);

        let abs64 = relocs.iter().find(|r| r.name == "R_AARCH64_ABS64");
        assert!(abs64.is_some());
        assert_eq!(abs64.unwrap().type_id, 257);
        assert_eq!(abs64.unwrap().size, 8);
        assert!(!abs64.unwrap().is_pc_relative);

        let call26 = relocs.iter().find(|r| r.name == "R_AARCH64_CALL26");
        assert!(call26.is_some());
        assert_eq!(call26.unwrap().type_id, 283);
        assert!(call26.unwrap().is_pc_relative);

        let adrp = relocs
            .iter()
            .find(|r| r.name == "R_AARCH64_ADR_PREL_PG_HI21");
        assert!(adrp.is_some());
        assert_eq!(adrp.unwrap().type_id, 275);

        let jslot = relocs.iter().find(|r| r.name == "R_AARCH64_JUMP_SLOT");
        assert!(jslot.is_some());
        assert_eq!(jslot.unwrap().type_id, 1026);
    }

    #[test]
    fn test_elf_constants() {
        assert_eq!(EM_AARCH64, 183);
        assert_eq!(ELF_FLAGS, 0);
        assert_eq!(DEFAULT_BASE_ADDRESS, 0x400000);
        assert_eq!(PAGE_SIZE, 4096);
    }

    #[test]
    fn test_emit_prologue_basic() {
        let codegen = AArch64Codegen::new(false, false);
        let mf = MachineFunction::new("test_func".to_string());
        let prologue = codegen.emit_prologue(&mf);
        // Minimum: STP (FP, LR) + MOV (FP, SP)
        assert!(
            prologue.len() >= 2,
            "Prologue should have at least 2 instructions"
        );
    }

    #[test]
    fn test_emit_epilogue_basic() {
        let codegen = AArch64Codegen::new(false, false);
        let mf = MachineFunction::new("test_func".to_string());
        let epilogue = codegen.emit_epilogue(&mf);
        // Minimum: LDP (FP, LR) + RET
        assert!(
            epilogue.len() >= 2,
            "Epilogue should have at least 2 instructions"
        );
        let last = epilogue.last().unwrap();
        assert!(
            last.is_terminator,
            "Last epilogue instruction should be a terminator (RET)"
        );
    }

    #[test]
    fn test_classify_argument_integer() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::I64);
        assert!(loc.is_register());
    }

    #[test]
    fn test_classify_argument_float() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::F64);
        assert!(loc.is_register());
    }

    #[test]
    fn test_classify_argument_pointer() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::Ptr);
        assert!(loc.is_register());
    }

    #[test]
    fn test_classify_return_void() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::Void);
        assert!(loc.is_stack());
    }

    #[test]
    fn test_classify_return_integer() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::I32);
        assert!(loc.is_register());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(AArch64Codegen::align_up(0, 16), 0);
        assert_eq!(AArch64Codegen::align_up(1, 16), 16);
        assert_eq!(AArch64Codegen::align_up(15, 16), 16);
        assert_eq!(AArch64Codegen::align_up(16, 16), 16);
        assert_eq!(AArch64Codegen::align_up(17, 16), 32);
        assert_eq!(AArch64Codegen::align_up(0, 4), 0);
        assert_eq!(AArch64Codegen::align_up(3, 4), 4);
    }

    #[test]
    fn test_convert_instructions() {
        let a64_insts = vec![
            A64Instruction::new(A64Opcode::ADD_imm)
                .with_rd(0)
                .with_rn(1)
                .with_imm(42),
            A64Instruction::new(A64Opcode::RET),
        ];

        let mis = AArch64Codegen::convert_instructions(&a64_insts);
        assert_eq!(mis.len(), 2);
        assert!(mis[0].result.is_some());
        assert!(!mis[0].is_terminator);
        assert!(mis[1].is_terminator);
    }

    #[test]
    fn test_serialize_constant_integer() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::Integer(42), &target);
        assert_eq!(bytes.len(), 8);
        assert_eq!(i64::from_le_bytes(bytes[..8].try_into().unwrap()), 42);
    }

    #[test]
    fn test_serialize_constant_float() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::Float(3.14), &target);
        assert_eq!(bytes.len(), 8);
        let val = f64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert!((val - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serialize_constant_string() {
        let target = Target::AArch64;
        let bytes =
            AArch64Codegen::serialize_constant(&Constant::String(b"Hello\0".to_vec()), &target);
        assert_eq!(bytes, b"Hello\0");
    }

    #[test]
    fn test_serialize_constant_zero_init() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::ZeroInit, &target);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_serialize_constant_null() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::Null, &target);
        assert_eq!(bytes.len(), 8);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
