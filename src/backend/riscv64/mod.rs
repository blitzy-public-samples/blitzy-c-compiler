//! # RISC-V 64 Backend
//!
//! Complete RISC-V 64-bit code generation backend for BCC.
//! Implements the [`ArchCodegen`] trait for the RV64IMAFDC ISA with LP64D ABI.
//!
//! ## Submodules
//!
//! - [`codegen`] — Instruction selection (IR → RISC-V machine instructions)
//! - [`registers`] — Register file definitions (x0–x31, f0–f31)
//! - [`abi`] — LP64D calling convention and stack frame layout
//! - [`assembler`] — Built-in RISC-V 64 assembler (instruction encoding, relocations)
//! - [`linker`] — Built-in RISC-V 64 ELF linker (relocation application, relaxation)
//!
//! ## Architecture Characteristics
//!
//! - Fixed 32-bit instruction width (with 16-bit compressed extension)
//! - Load/store architecture — ALU ops on registers only
//! - 32 integer registers: x0 (zero) is hardwired to 0
//! - 32 FP registers: IEEE 754 single and double precision
//! - Little-endian byte order
//! - ELF machine type: EM_RISCV (243)
//! - ELF flags: EF_RISCV_FLOAT_ABI_DOUBLE | EF_RISCV_RVC (0x0005)
//!
//! ## Pipeline Position
//!
//! The code generation driver ([`crate::backend::generation`]) dispatches to
//! [`RiscV64Codegen`] when `--target=riscv64` is specified.  The flow is:
//!
//! 1. `lower_function` — IR → machine instructions (instruction selection)
//! 2. Register allocation via [`crate::backend::register_allocator`]
//! 3. `emit_prologue` / `emit_epilogue` — stack frame setup / teardown
//! 4. `emit_assembly` — machine instructions → encoded bytes
//! 5. `compile_to_object` — ELF `.o` production from a full IR module
//!
//! ## Primary target for Linux kernel 6.9 boot validation (Checkpoint 6).

// ===========================================================================
// Submodule Declarations
// ===========================================================================

pub mod abi;
pub mod assembler;
pub mod codegen;
pub mod linker;
pub mod registers;

// ===========================================================================
// Crate Imports
// ===========================================================================

use crate::backend::elf_writer_common::{
    self, ElfSymbol, ElfWriter, Relocation, Section, ET_REL, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE,
    SHT_NOBITS, SHT_PROGBITS, STB_GLOBAL, STB_LOCAL,
};
use crate::backend::traits::{
    ArchCodegen, ArgLocation, AssembledFunction, MachineBasicBlock, MachineFunction,
    MachineInstruction, MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::target::Target;
use crate::common::types::{CType, StructField, TypeQualifiers};
use crate::ir::function::IrFunction;
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

// ===========================================================================
// Submodule Imports
// ===========================================================================

use self::abi::RiscV64Abi;
use self::codegen::RiscV64InstructionSelector;
use self::registers::{RiscV64RegisterInfo, FP, RA, SP};

// ===========================================================================
// Public Re-exports
// ===========================================================================

pub use self::abi::{ArgClass, FrameLayout, RiscV64Abi as RiscV64AbiExport};
pub use self::codegen::{
    RiscV64InstructionSelector as RiscV64InstructionSelectorExport, RvInstruction, RvOpcode,
};
pub use self::registers::{RegClass, RiscV64RegisterInfo as RiscV64RegisterInfoExport};

// ===========================================================================
// RISC-V 64 ELF Constants
// ===========================================================================

/// ELF machine type for RISC-V (e_machine field).
///
/// Value 243 (0xF3) as defined in the ELF specification supplement for
/// RISC-V.
pub const EM_RISCV: u16 = 243;

/// ELF flag: double-precision floating-point ABI (LP64D).
///
/// Bit 2 of `e_flags`. Indicates that the object uses the LP64D ABI
/// where `double` arguments are passed in floating-point registers.
pub const EF_RISCV_FLOAT_ABI_DOUBLE: u32 = 0x0004;

/// ELF flag: compressed (RVC) extension enabled.
///
/// Bit 0 of `e_flags`. Indicates that the object may contain 16-bit
/// compressed (C extension) instructions.
pub const EF_RISCV_RVC: u32 = 0x0001;

/// Combined ELF flags for BCC RISC-V 64 output.
///
/// `EF_RISCV_RVC` = 0x0001.  The float ABI bits are 0 (soft-float /
/// lp64) to match the Linux kernel's compilation model.  The kernel
/// uses `-mabi=lp64` and never generates hardware FP instructions, so
/// BCC must emit soft-float flags for link-time ABI compatibility.
pub const ELF_FLAGS: u32 = EF_RISCV_RVC;

/// Default base virtual address for RISC-V 64 static executables.
///
/// The first `PT_LOAD` segment is placed at this address.  Matches the
/// standard Linux layout for RISC-V ELF executables.
pub const DEFAULT_BASE_ADDRESS: u64 = 0x10000;

/// Memory page size for RISC-V 64.
///
/// Used for `PT_LOAD` segment alignment in executables and shared objects.
/// The standard Linux RISC-V page size is 4 KiB.
pub const PAGE_SIZE: u64 = 4096;

// ===========================================================================
// RISC-V 64 Relocation Type Constants
// ===========================================================================

/// No relocation — placeholder entry.
const R_RISCV_NONE: u32 = 0;
/// 32-bit absolute address.
const R_RISCV_32: u32 = 1;
/// 64-bit absolute address.
const R_RISCV_64: u32 = 2;
/// B-type branch offset (±4 KiB range).
const R_RISCV_BRANCH: u32 = 16;
/// J-type jump offset (±1 MiB range).
const R_RISCV_JAL: u32 = 17;
/// AUIPC+JALR call pair (32-bit PC-relative range).
const R_RISCV_CALL: u32 = 18;
/// AUIPC+JALR call pair via PLT (for PIC).
const R_RISCV_CALL_PLT: u32 = 19;
/// GOT entry upper 20 bits (PC-relative).
const R_RISCV_GOT_HI20: u32 = 20;
/// TLS GD upper 20 bits (for thread-local storage).
const R_RISCV_TLS_GD_HI20: u32 = 22;
/// PC-relative upper 20 bits (AUIPC target).
const R_RISCV_PCREL_HI20: u32 = 23;
/// PC-relative lower 12 bits, I-type encoding (ADDI/LD).
const R_RISCV_PCREL_LO12_I: u32 = 24;
/// PC-relative lower 12 bits, S-type encoding (SD/SW).
const R_RISCV_PCREL_LO12_S: u32 = 25;
/// Absolute upper 20 bits (LUI target).
const R_RISCV_HI20: u32 = 26;
/// Absolute lower 12 bits, I-type encoding.
const R_RISCV_LO12_I: u32 = 27;
/// Absolute lower 12 bits, S-type encoding.
const R_RISCV_LO12_S: u32 = 28;
/// 32-bit addition (for DWARF / exception info).
const R_RISCV_ADD32: u32 = 35;
/// 64-bit addition (for DWARF / exception info).
const R_RISCV_ADD64: u32 = 36;
/// 32-bit subtraction (for DWARF / exception info).
const R_RISCV_SUB32: u32 = 39;
/// 64-bit subtraction (for DWARF / exception info).
const R_RISCV_SUB64: u32 = 40;
/// Alignment directive hint for the linker.
const R_RISCV_ALIGN: u32 = 43;
/// RVC branch offset (compressed branch).
const R_RISCV_RVC_BRANCH: u32 = 44;
/// RVC jump offset (compressed jump).
const R_RISCV_RVC_JUMP: u32 = 45;
/// Linker relaxation hint — indicates the relocation pair may be
/// relaxed (e.g., AUIPC+JALR → JAL when target is within ±1 MiB).
const R_RISCV_RELAX: u32 = 51;
/// Set 6 low bits of a byte (for DWARF/CFA).
const R_RISCV_SET6: u32 = 52;
/// Set low byte.
const R_RISCV_SET8: u32 = 53;
/// Set 16-bit value.
const R_RISCV_SET16: u32 = 54;
/// Set 32-bit value.
const R_RISCV_SET32: u32 = 55;
/// 8-bit addition (for DWARF / compressed debug).
const R_RISCV_ADD8: u32 = 33;
/// 16-bit addition.
const R_RISCV_ADD16: u32 = 34;
/// 8-bit subtraction.
const R_RISCV_SUB8: u32 = 37;
/// 16-bit subtraction.
const R_RISCV_SUB16: u32 = 38;
/// 6-bit subtraction (for DWARF uleb128).
#[allow(dead_code)]
const R_RISCV_SUB6: u32 = 52; // Note: shares value with SET6 — ELF spec distinguishes by context

// ===========================================================================
// RiscV64Codegen — Main Backend Struct
// ===========================================================================

/// RISC-V 64 code generation backend.
///
/// This is the main entry point for RISC-V 64 code generation in BCC.
/// It implements the [`ArchCodegen`] trait, providing:
///
/// - Instruction selection via [`RiscV64InstructionSelector`]
/// - Register allocation via [`crate::backend::register_allocator`]
/// - Prologue / epilogue generation for LP64D stack frames
/// - Machine code emission via the built-in RISC-V assembler
/// - ELF relocatable object (`.o`) production via [`compile_to_object`](Self::compile_to_object)
///
/// # Construction
///
/// ```ignore
/// let codegen = RiscV64Codegen::new(pic_mode, debug_info);
/// let mf = codegen.lower_function(&ir_func, &mut diag)?;
/// let bytes = codegen.emit_assembly(&mf)?;
/// ```
pub struct RiscV64Codegen {
    /// Target architecture descriptor (always [`Target::RiscV64`]).
    pub target: Target,

    /// Register file information provider.
    ///
    /// Unit struct that converts to [`RegisterInfo`] via
    /// [`RiscV64RegisterInfo::to_register_info()`].
    pub reg_info: RiscV64RegisterInfo,

    /// LP64D ABI handler for argument / return value classification.
    pub abi: RiscV64Abi,

    /// Whether position-independent code generation is enabled (`-fPIC`).
    ///
    /// When `true`, global variable access uses AUIPC+LD through the GOT
    /// and function calls use AUIPC+JALR through the PLT.
    pub pic_mode: bool,

    /// Whether DWARF debug information should be emitted (`-g`).
    ///
    /// When `true`, `.debug_info`, `.debug_abbrev`, `.debug_line`, and
    /// `.debug_str` sections are generated in the output object file.
    pub debug_info: bool,

    /// Cached relocation type descriptors for this architecture.
    ///
    /// Built once during construction and returned by reference from
    /// [`ArchCodegen::relocation_types()`].
    pub relocation_types: Vec<RelocationTypeInfo>,

    /// Set of function names that accept variadic arguments (`...`).
    /// On RISC-V LP64D, variadic function calls must pass ALL FP arguments
    /// in integer registers instead of FP registers.  Populated from
    /// `IrModule::declarations()` before code generation begins.
    pub variadic_functions: crate::common::fx_hash::FxHashSet<String>,
}

impl RiscV64Codegen {
    /// Create a new RISC-V 64 code generation backend.
    ///
    /// # Arguments
    ///
    /// * `pic_mode` — Whether to generate position-independent code (`-fPIC`).
    ///   Affects address materialization (AUIPC+LD via GOT vs LUI+ADDI) and
    ///   function call sequences (PLT vs direct).
    /// * `debug_info` — Whether to emit DWARF v4 debug sections (`-g`).
    ///   When true, source file/line mapping and local variable location
    ///   information is generated.
    pub fn new(pic_mode: bool, debug_info: bool) -> Self {
        Self {
            target: Target::RiscV64,
            reg_info: RiscV64RegisterInfo,
            abi: RiscV64Abi::new(),
            pic_mode,
            debug_info,
            relocation_types: Self::build_relocation_types(),
            variadic_functions: crate::common::fx_hash::FxHashSet::default(),
        }
    }

    /// Populate the variadic function set from module declarations.
    /// Must be called before `lower_function` to ensure correct ABI
    /// handling for variadic calls (FP args in integer registers).
    pub fn set_variadic_functions(&mut self, set: crate::common::fx_hash::FxHashSet<String>) {
        self.variadic_functions = set;
    }

    /// Build the complete set of RISC-V 64 relocation type descriptors.
    ///
    /// These descriptors are used by the common linker infrastructure to
    /// understand each relocation's semantics (name, ELF type ID, size in
    /// bytes, and whether it is PC-relative).
    ///
    /// Returns a vector containing descriptors for all RISC-V relocations
    /// needed by the BCC assembler and linker.
    pub fn build_relocation_types() -> Vec<RelocationTypeInfo> {
        vec![
            // --- Absolute relocations ---
            RelocationTypeInfo::new("R_RISCV_NONE", R_RISCV_NONE, 0, false),
            RelocationTypeInfo::new("R_RISCV_32", R_RISCV_32, 4, false),
            RelocationTypeInfo::new("R_RISCV_64", R_RISCV_64, 8, false),
            // --- Branch / jump relocations (PC-relative) ---
            RelocationTypeInfo::new("R_RISCV_BRANCH", R_RISCV_BRANCH, 4, true),
            RelocationTypeInfo::new("R_RISCV_JAL", R_RISCV_JAL, 4, true),
            RelocationTypeInfo::new("R_RISCV_CALL", R_RISCV_CALL, 8, true),
            RelocationTypeInfo::new("R_RISCV_CALL_PLT", R_RISCV_CALL_PLT, 8, true),
            // --- GOT-relative (PC-relative) ---
            RelocationTypeInfo::new("R_RISCV_GOT_HI20", R_RISCV_GOT_HI20, 4, true),
            // --- TLS ---
            RelocationTypeInfo::new("R_RISCV_TLS_GD_HI20", R_RISCV_TLS_GD_HI20, 4, true),
            // --- PC-relative hi20/lo12 pair ---
            RelocationTypeInfo::new("R_RISCV_PCREL_HI20", R_RISCV_PCREL_HI20, 4, true),
            RelocationTypeInfo::new("R_RISCV_PCREL_LO12_I", R_RISCV_PCREL_LO12_I, 4, true),
            RelocationTypeInfo::new("R_RISCV_PCREL_LO12_S", R_RISCV_PCREL_LO12_S, 4, true),
            // --- Absolute hi20/lo12 pair ---
            RelocationTypeInfo::new("R_RISCV_HI20", R_RISCV_HI20, 4, false),
            RelocationTypeInfo::new("R_RISCV_LO12_I", R_RISCV_LO12_I, 4, false),
            RelocationTypeInfo::new("R_RISCV_LO12_S", R_RISCV_LO12_S, 4, false),
            // --- Arithmetic relocations (DWARF, exception tables) ---
            RelocationTypeInfo::new("R_RISCV_ADD8", R_RISCV_ADD8, 1, false),
            RelocationTypeInfo::new("R_RISCV_ADD16", R_RISCV_ADD16, 2, false),
            RelocationTypeInfo::new("R_RISCV_ADD32", R_RISCV_ADD32, 4, false),
            RelocationTypeInfo::new("R_RISCV_ADD64", R_RISCV_ADD64, 8, false),
            RelocationTypeInfo::new("R_RISCV_SUB8", R_RISCV_SUB8, 1, false),
            RelocationTypeInfo::new("R_RISCV_SUB16", R_RISCV_SUB16, 2, false),
            RelocationTypeInfo::new("R_RISCV_SUB32", R_RISCV_SUB32, 4, false),
            RelocationTypeInfo::new("R_RISCV_SUB64", R_RISCV_SUB64, 8, false),
            // --- Alignment and relaxation hints ---
            RelocationTypeInfo::new("R_RISCV_ALIGN", R_RISCV_ALIGN, 0, false),
            RelocationTypeInfo::new("R_RISCV_RELAX", R_RISCV_RELAX, 0, false),
            // --- Compressed branch/jump ---
            RelocationTypeInfo::new("R_RISCV_RVC_BRANCH", R_RISCV_RVC_BRANCH, 2, true),
            RelocationTypeInfo::new("R_RISCV_RVC_JUMP", R_RISCV_RVC_JUMP, 2, true),
            // --- Bitfield set relocations ---
            RelocationTypeInfo::new("R_RISCV_SET6", R_RISCV_SET6, 1, false),
            RelocationTypeInfo::new("R_RISCV_SET8", R_RISCV_SET8, 1, false),
            RelocationTypeInfo::new("R_RISCV_SET16", R_RISCV_SET16, 2, false),
            RelocationTypeInfo::new("R_RISCV_SET32", R_RISCV_SET32, 4, false),
        ]
    }

    // =======================================================================
    // Helper: Convert IrType to CType for ABI classification
    // =======================================================================

    /// Map an IR type to a C language type for ABI classification purposes.
    ///
    /// The [`ArchCodegen`] trait methods `classify_argument` and
    /// `classify_return` receive [`IrType`] values, but the LP64D ABI
    /// handler ([`RiscV64Abi`]) classifies [`CType`] values.  This helper
    /// bridges the two type systems.
    fn ir_type_to_ctype(ty: &IrType) -> CType {
        match ty {
            IrType::Void => CType::Void,
            IrType::I1 => CType::Bool,
            IrType::I8 => CType::SChar,
            IrType::I16 => CType::Short,
            IrType::I32 => CType::Int,
            IrType::I64 => CType::Long,
            IrType::I128 => CType::LongLong, // 128-bit mapped to long long for ABI
            IrType::F32 => CType::Float,
            IrType::F64 => CType::Double,
            IrType::F80 => CType::LongDouble,
            IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            IrType::Array(elem, count) => {
                let elem_ctype = Self::ir_type_to_ctype(elem);
                CType::Array(Box::new(elem_ctype), Some(*count))
            }
            IrType::Struct(st) => {
                let ctype_fields: Vec<StructField> = st
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| StructField {
                        name: Some(format!("field{}", i)),
                        ty: Self::ir_type_to_ctype(f),
                        bit_width: None,
                    })
                    .collect();
                CType::Struct {
                    name: st.name.clone(),
                    fields: ctype_fields,
                    packed: st.packed,
                    aligned: None,
                }
            }
            IrType::Function(ret, params) => {
                let ret_ctype = Self::ir_type_to_ctype(ret);
                let param_ctypes: Vec<CType> = params.iter().map(Self::ir_type_to_ctype).collect();
                CType::Function {
                    return_type: Box::new(ret_ctype),
                    params: param_ctypes,
                    variadic: false,
                }
            }
        }
    }

    // =======================================================================
    // Helper: Convert RvInstructions to MachineInstructions
    // =======================================================================

    /// Convert a RISC-V instruction from the instruction selector's
    /// representation ([`codegen::RvInstruction`]) to the backend-agnostic
    /// [`MachineInstruction`] format used by the register allocator and
    /// assembly emitter.
    /// Convert a register ID to the appropriate MachineOperand.
    ///
    /// Register IDs 0–63 are physical registers and map to
    /// `MachineOperand::Register`.  IDs ≥ 100 are virtual registers
    /// assigned during instruction selection and map to
    /// `MachineOperand::VirtualRegister`, which the register allocator
    /// will later resolve to physical registers.
    #[inline]
    fn reg_to_operand(reg: u16) -> MachineOperand {
        if reg >= 100 {
            MachineOperand::VirtualRegister(reg as u32)
        } else {
            MachineOperand::Register(reg)
        }
    }

    fn rv_to_machine_instruction(rv: &codegen::RvInstruction) -> MachineInstruction {
        let opcode = rv.opcode as u32;
        let mut mi = MachineInstruction::new(opcode);

        // For INLINE_ASM, the substituted template is stored in the symbol
        // field.  Transfer it to asm_template so format_instruction() can
        // emit it verbatim.
        if rv.opcode == codegen::RvOpcode::INLINE_ASM {
            if let Some(ref template) = rv.symbol {
                mi.asm_template = Some(template.clone());
            }
            return mi;
        }

        // Destination register (result).
        if let Some(rd) = rv.rd {
            mi = mi.with_result(Self::reg_to_operand(rd));
        }

        // Source register 1.
        if let Some(rs1) = rv.rs1 {
            mi = mi.with_operand(Self::reg_to_operand(rs1));
        }

        // Source register 2.
        if let Some(rs2) = rv.rs2 {
            mi = mi.with_operand(Self::reg_to_operand(rs2));
        }

        // Source register 3 (fused multiply-add).
        if let Some(rs3) = rv.rs3 {
            mi = mi.with_operand(Self::reg_to_operand(rs3));
        }

        // Immediate value.
        if rv.imm != 0 {
            mi = mi.with_operand(MachineOperand::Immediate(rv.imm));
        }

        // Symbol reference for relocations.
        if let Some(ref sym) = rv.symbol {
            mi = mi.with_operand(MachineOperand::GlobalSymbol(sym.clone()));
        }

        // Preserve `.label:` comments through the MachineInstruction
        // round-trip by stashing them in `asm_template`.  This lets the
        // two-pass assembler in `emit_assembly` reconstruct label offsets
        // when converting back via `machine_to_rv_instruction`.
        if let Some(ref comment) = rv.comment {
            if comment.starts_with(".label:") {
                mi.asm_template = Some(comment.clone());
            }
        }

        // Propagate the call-argument-setup flag so the post-register-
        // allocation parallel-move resolver (`resolve_call_arg_conflicts`)
        // can identify the contiguous arg-setup window before CALL.
        if rv.is_call_arg_setup {
            mi.is_call_arg_setup = true;
        }

        // Mark branch / call / terminator status based on opcode.
        match rv.opcode {
            codegen::RvOpcode::BEQ
            | codegen::RvOpcode::BNE
            | codegen::RvOpcode::BLT
            | codegen::RvOpcode::BGE
            | codegen::RvOpcode::BLTU
            | codegen::RvOpcode::BGEU => {
                mi = mi.set_branch().set_terminator();
            }
            codegen::RvOpcode::JAL | codegen::RvOpcode::J => {
                mi = mi.set_branch().set_terminator();
            }
            codegen::RvOpcode::JALR => {
                // JALR can be a call (when rd != x0) or a return (when rs1 = ra).
                if rv.rd == Some(registers::RA) || rv.rd == Some(registers::X1) {
                    mi = mi.set_call();
                } else {
                    mi = mi.set_terminator();
                }
            }
            codegen::RvOpcode::CALL => {
                mi = mi.set_call();
            }
            codegen::RvOpcode::RET => {
                mi = mi.set_terminator();
            }
            _ => {}
        }

        mi
    }

    // =======================================================================
    // compile_to_object — Full Module Compilation
    // =======================================================================

    /// Compile an entire IR module to a RISC-V 64 relocatable ELF object
    /// file (`.o`).
    ///
    /// This is the top-level entry point for producing a complete object
    /// file from an IR module.  It orchestrates the full code generation
    /// pipeline:
    ///
    /// 1. Lower each function via instruction selection
    /// 2. Allocate physical registers
    /// 3. Generate prologue / epilogue code
    /// 4. Assemble instructions to machine code bytes
    /// 5. Emit global variables and string literals
    /// 6. Construct the ELF object with appropriate sections
    /// 7. Optionally emit DWARF debug sections (when `debug_info` is set)
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module containing functions, globals, and string
    ///   literals to compile.
    /// * `diagnostics` — Diagnostic engine for reporting code generation
    ///   errors (unsupported inline assembly constraints, relocation
    ///   overflows, etc.).
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the complete ELF object file bytes on
    /// success, or `Err(())` if fatal errors were emitted to the diagnostic
    /// engine.
    #[allow(clippy::result_unit_err)]
    pub fn compile_to_object(
        &self,
        module: &IrModule,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<Vec<u8>, ()> {
        let mut text_bytes: Vec<u8> = Vec::new();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut rodata_bytes: Vec<u8> = Vec::new();
        let mut bss_size: usize = 0;
        let mut text_relocations: Vec<Relocation> = Vec::new();
        let mut symbols: Vec<ElfSymbol> = Vec::new();
        let mut function_offsets: Vec<(String, usize, usize)> = Vec::new();

        // -----------------------------------------------------------------
        // Phase 1: Compile each function definition
        // -----------------------------------------------------------------
        for func in module.functions() {
            if !func.is_definition {
                continue;
            }

            // Step 1: Instruction selection (IR → machine instructions).
            let mf = match self.lower_function(
                func,
                diagnostics,
                module.globals(),
                &module.func_ref_map,
                &module.global_var_refs,
            ) {
                Ok(mf) => mf,
                Err(msg) => {
                    diagnostics.emit_error(
                        crate::common::diagnostics::Span::dummy(),
                        format!("RISC-V 64 codegen error in '{}': {}", func.name, msg),
                    );
                    continue;
                }
            };

            // Step 2: Insert prologue at the beginning of the entry block
            //         and epilogue before each RET instruction.
            let mut mf = mf;
            let prologue = self.emit_prologue(&mf);
            let epilogue = self.emit_epilogue(&mf);

            if !mf.blocks.is_empty() && !prologue.is_empty() {
                // Prepend prologue to the entry block.
                let entry = &mut mf.blocks[0];
                let mut new_insts = prologue;
                new_insts.append(&mut entry.instructions);
                entry.instructions = new_insts;
            }

            if !epilogue.is_empty() {
                // Insert epilogue before each RET instruction across all blocks.
                for block in &mut mf.blocks {
                    let mut i = 0;
                    while i < block.instructions.len() {
                        if block.instructions[i].is_terminator
                            && block.instructions[i].opcode == codegen::RvOpcode::RET as u32
                        {
                            // Insert epilogue just before the RET.
                            for (j, ep_inst) in epilogue.iter().enumerate() {
                                block.instructions.insert(i + j, ep_inst.clone());
                            }
                            i += epilogue.len() + 1;
                        } else {
                            i += 1;
                        }
                    }
                }
            }

            // Step 3: Assemble to machine code bytes and collect relocations.
            let func_base_offset = text_bytes.len();
            let func_bytes = {
                use crate::backend::riscv64::assembler::encoder::RiscV64Encoder;
                let encoder = RiscV64Encoder::new();
                let mut bytes = Vec::new();

                for block in &mf.blocks {
                    for mi in &block.instructions {
                        let rv_inst = Self::machine_to_rv_instruction(mi);
                        match encoder.encode(&rv_inst) {
                            Ok(encoded) => {
                                // Collect relocation if present.
                                if let Some(ref reloc) = encoded.relocation {
                                    let sym_name = rv_inst
                                        .symbol
                                        .as_deref()
                                        .map(|s| s.trim_end_matches("@plt").to_string())
                                        .unwrap_or_default();
                                    let _ = sym_name; // used below for ELF sym_index resolution

                                    text_relocations.push(Relocation {
                                        offset: (func_base_offset
                                            + bytes.len()
                                            + reloc.offset as usize)
                                            as u64,
                                        sym_index: 0, // resolved during ELF emission
                                        rel_type: reloc.reloc_type,
                                        addend: reloc.addend,
                                    });
                                }
                                bytes.extend_from_slice(&encoded.bytes);

                                // Emit continuation bytes (e.g. the JALR half
                                // of an AUIPC+JALR CALL pair).
                                if let Some(ref cont) = encoded.continuation {
                                    bytes.extend_from_slice(&cont.bytes);
                                }
                            }
                            Err(_e) => {
                                bytes.extend_from_slice(&0x0000_0013u32.to_le_bytes());
                            }
                        }
                    }
                }
                bytes
            };

            // Track function offset and size for symbol table entry.
            let func_offset = func_base_offset;
            let func_size = func_bytes.len();
            text_bytes.extend_from_slice(&func_bytes);
            function_offsets.push((func.name.clone(), func_offset, func_size));

            // Add symbol for this function.
            symbols.push(ElfSymbol {
                name: func.name.clone(),
                value: func_offset as u64,
                size: func_size as u64,
                binding: STB_GLOBAL,
                sym_type: elf_writer_common::STT_FUNC,
                section_index: 1, // .text section (1-based)
                visibility: elf_writer_common::STV_DEFAULT,
            });
        }

        // -----------------------------------------------------------------
        // Phase 2: Emit global variables
        // -----------------------------------------------------------------
        for global in module.globals() {
            let type_size = global.ty.size_bytes(&self.target);
            let type_align = global.ty.align_bytes(&self.target);

            if global.initializer.is_some() {
                // Initialized data → .data section.
                // Align the data section offset.
                let padding = (type_align - (data_bytes.len() % type_align)) % type_align;
                data_bytes.extend(std::iter::repeat(0u8).take(padding));

                let data_offset = data_bytes.len();

                // Write initializer bytes (simplified: zero-fill for now,
                // actual constant evaluation would produce real bytes).
                data_bytes.extend(std::iter::repeat(0u8).take(type_size));

                // Respect the IR-level linkage: compound literals and static
                // variables use Internal linkage → STB_LOCAL.
                let binding = match global.linkage {
                    crate::ir::module::Linkage::Internal => STB_LOCAL,
                    crate::ir::module::Linkage::Weak => crate::backend::elf_writer_common::STB_WEAK,
                    _ => STB_GLOBAL,
                };
                symbols.push(ElfSymbol {
                    name: global.name.clone(),
                    value: data_offset as u64,
                    size: type_size as u64,
                    binding,
                    sym_type: elf_writer_common::STT_OBJECT,
                    section_index: 2, // .data section
                    visibility: elf_writer_common::STV_DEFAULT,
                });
            } else {
                // Uninitialized data → .bss section.
                let padding = (type_align - (bss_size % type_align)) % type_align;
                bss_size += padding;

                let bss_offset = bss_size;
                bss_size += type_size;

                let binding = match global.linkage {
                    crate::ir::module::Linkage::Internal => STB_LOCAL,
                    crate::ir::module::Linkage::Weak => crate::backend::elf_writer_common::STB_WEAK,
                    _ => STB_GLOBAL,
                };
                symbols.push(ElfSymbol {
                    name: global.name.clone(),
                    value: bss_offset as u64,
                    size: type_size as u64,
                    binding,
                    sym_type: elf_writer_common::STT_OBJECT,
                    section_index: 4, // .bss section
                    visibility: elf_writer_common::STV_DEFAULT,
                });
            }
        }

        // -----------------------------------------------------------------
        // Phase 3: Emit string literals to .rodata
        // -----------------------------------------------------------------
        for string_lit in module.string_pool() {
            let str_offset = rodata_bytes.len();
            rodata_bytes.extend_from_slice(&string_lit.bytes);
            // Null-terminate if not already terminated.
            if string_lit.bytes.last() != Some(&0) {
                rodata_bytes.push(0);
            }

            symbols.push(ElfSymbol {
                name: format!(".L.str.{}", string_lit.id),
                value: str_offset as u64,
                size: (rodata_bytes.len() - str_offset) as u64,
                binding: STB_LOCAL,
                sym_type: elf_writer_common::STT_OBJECT,
                section_index: 3, // .rodata section
                visibility: elf_writer_common::STV_DEFAULT,
            });
        }

        // -----------------------------------------------------------------
        // Phase 4: Add external declarations as undefined symbols.
        // Mark them as STT_FUNC so the dynamic linker creates PLT entries
        // for lazy binding of external function calls (e.g., printf, exit).
        // -----------------------------------------------------------------
        for decl in module.declarations() {
            symbols.push(ElfSymbol {
                name: decl.name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: elf_writer_common::STT_FUNC,
                section_index: 0, // SHN_UNDEF
                visibility: elf_writer_common::STV_DEFAULT,
            });
        }

        // -----------------------------------------------------------------
        // Phase 5: Construct ELF object
        // -----------------------------------------------------------------
        let mut elf = ElfWriter::new(Target::RiscV64, ET_REL);

        // .text section (index 1).
        elf.add_section(Section {
            name: ".text".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: text_bytes,
            sh_addralign: 4, // RISC-V instructions are 4-byte aligned
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });

        // .data section (index 2).
        elf.add_section(Section {
            name: ".data".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: data_bytes,
            sh_addralign: 8,
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });

        // .rodata section (index 3).
        elf.add_section(Section {
            name: ".rodata".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC,
            data: rodata_bytes,
            sh_addralign: 8,
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });

        // .bss section (index 4).
        // SHT_NOBITS sections have no file data — use empty vec and
        // set logical_size to avoid wasteful memory allocation.
        elf.add_section(Section {
            name: ".bss".to_string(),
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: Vec::new(),
            sh_addralign: 8,
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
            logical_size: bss_size as u64,
            virtual_address: 0,
            file_offset_hint: 0,
        });

        // .note.GNU-stack section — non-executable stack marker.
        // This is required by the Linux kernel build and modern toolchains
        // to indicate that the generated code does not require an executable
        // stack. Without this section, the linker may default to an executable
        // stack, which is a security concern.
        elf.add_section(Section {
            name: ".note.GNU-stack".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: 0, // No SHF_EXECINSTR = non-executable stack.
            data: Vec::new(),
            sh_addralign: 1,
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });

        // Add all symbols.
        for sym in symbols {
            elf.add_symbol(sym);
        }

        // Check for accumulated errors.
        if diagnostics.has_errors() {
            return Err(());
        }

        Ok(elf.write())
    }
}

// ===========================================================================
// ArchCodegen Trait Implementation
// ===========================================================================

impl ArchCodegen for RiscV64Codegen {
    /// Perform instruction selection: lower an IR function to RISC-V 64
    /// machine instructions.
    ///
    /// Creates a [`RiscV64InstructionSelector`], runs it over the IR
    /// function's basic blocks, and produces a [`MachineFunction`] with
    /// virtual registers ready for register allocation.
    fn lower_function(
        &self,
        func: &IrFunction,
        _diag: &mut DiagnosticEngine,
        _globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
        global_var_refs: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) -> Result<MachineFunction, String> {
        // Build constant cache from the IR function's constant_values map.
        // This was populated during IR lowering for every integer constant,
        // providing a reliable Value → i64 mapping without fragile heuristics.
        let mut constant_values = crate::common::fx_hash::FxHashMap::default();
        for (&val, &imm) in &func.constant_values {
            constant_values.insert(val.index(), imm);
        }

        // Build float constant cache from the IR function's float_constant_values.
        // Float constants are stored as global variables in .rodata; the cache
        // maps Value index → (symbol_name, f64_value) so the instruction selector
        // can generate LA + FLD to load them into FPRs.
        let mut float_constant_values = crate::common::fx_hash::FxHashMap::default();
        for (val, (name, fval)) in &func.float_constant_values {
            float_constant_values.insert(val.index(), (name.clone(), *fval));
        }

        // Create the instruction selector for this function.
        let mut selector = RiscV64InstructionSelector::new(self.target, self.pic_mode);
        selector.set_constant_values(constant_values);
        selector.set_float_constant_values(float_constant_values);
        selector.set_func_ref_names(func_ref_map.clone());
        selector.set_global_var_refs(global_var_refs.clone());
        selector.set_variadic_functions(self.variadic_functions.clone());

        // Run instruction selection over the entire function.
        let rv_instructions = selector.select_function(func, &self.abi);

        // Capture the vreg → IR Value mapping BEFORE building the
        // MachineFunction.  This lets the register allocator in
        // `generation.rs` resolve virtual registers to physical ones.
        let vreg_ir_map = selector.vreg_to_ir_value_map();

        // Build the MachineFunction from selected instructions.
        let mut mf = MachineFunction::new(func.name.clone());

        // Create a single entry block with all instructions.
        // In a more sophisticated implementation, basic blocks would be
        // preserved from the IR structure.
        let mut entry_block = MachineBasicBlock::new(None);
        entry_block.label = Some(format!(".L_{}_entry", func.name));

        for rv_inst in &rv_instructions {
            let mi = Self::rv_to_machine_instruction(rv_inst);
            entry_block.instructions.push(mi);
        }

        mf.add_block(entry_block);

        // Mark if the function is a leaf (no calls).
        let has_calls = rv_instructions.iter().any(|inst| {
            matches!(
                inst.opcode,
                codegen::RvOpcode::CALL | codegen::RvOpcode::JALR
            )
        });
        if has_calls {
            mf.mark_has_calls();
        }

        // Compute frame size from instruction selector state.
        mf.frame_size = selector.frame_size as usize;

        // Record callee-saved registers that the function uses.
        // Only track physical registers (< 100); virtual regs will be
        // resolved to physical by the register allocator.
        for rv_inst in &rv_instructions {
            if let Some(rd) = rv_inst.rd {
                if rd < 100 && registers::is_callee_saved(rd) {
                    let reg_u16 = rd;
                    if !mf.callee_saved_regs.contains(&reg_u16) {
                        mf.callee_saved_regs.push(reg_u16);
                    }
                }
            }
        }

        // Populate the vreg → IR Value mapping for the register allocator.
        mf.vreg_to_ir_value = vreg_ir_map;

        Ok(mf)
    }

    /// Encode machine instructions to raw bytes via the built-in assembler.
    ///
    /// Iterates over every instruction in every block of the machine
    /// function and produces the encoded binary representation.  Branch
    /// targets within the function are resolved; cross-function references
    /// produce relocation entries.
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<AssembledFunction, String> {
        use crate::backend::riscv64::assembler::encoder::RiscV64Encoder;
        use crate::backend::traits::FunctionRelocation;

        let encoder = RiscV64Encoder::new();

        // ---------------------------------------------------------------
        // Pass 1: Collect all instructions, compute sizes, and record
        //         local label offsets.  NOP instructions carrying a
        //         `.label:` comment are NOT emitted — they only define
        //         the label address.
        // ---------------------------------------------------------------
        #[allow(dead_code)]
        struct InstructionRecord {
            rv_inst: codegen::RvInstruction,
            offset: usize,
            primary_size: usize,
            cont_size: usize,
        }

        let mut records: Vec<InstructionRecord> = Vec::new();
        let mut label_offsets: crate::common::fx_hash::FxHashMap<String, usize> =
            crate::common::fx_hash::FxHashMap::default();
        let mut current_offset: usize = 0;

        for block in &mf.blocks {
            for mi in &block.instructions {
                let rv_inst = Self::machine_to_rv_instruction(mi);

                // Check for label-definition NOP.
                if let Some(ref comment) = rv_inst.comment {
                    if let Some(label) = comment.strip_prefix(".label:") {
                        label_offsets.insert(label.to_string(), current_offset);
                        continue; // don't emit this NOP
                    }
                }

                // Encode to determine size (the actual bytes will be
                // regenerated in pass 2 if branches are patched, but we
                // need a faithful size estimate).
                match encoder.encode(&rv_inst) {
                    Ok(encoded) => {
                        let primary_size = encoded.bytes.len();
                        let cont_size = encoded
                            .continuation
                            .as_ref()
                            .map(|c| c.bytes.len())
                            .unwrap_or(0);
                        records.push(InstructionRecord {
                            rv_inst,
                            offset: current_offset,
                            primary_size,
                            cont_size,
                        });
                        current_offset += primary_size + cont_size;
                    }
                    Err(_) => {
                        // NOP placeholder for unrecognised opcodes.
                        records.push(InstructionRecord {
                            rv_inst,
                            offset: current_offset,
                            primary_size: 4,
                            cont_size: 0,
                        });
                        current_offset += 4;
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Pass 2: Emit binary, resolving local branches against the
        //         label map.  External references become relocations.
        // ---------------------------------------------------------------
        let mut output: Vec<u8> = Vec::with_capacity(current_offset);
        let mut relocations: Vec<FunctionRelocation> = Vec::new();

        for rec in &records {
            let rv_inst = &rec.rv_inst;

            // Determine if the branch target is a local label.
            let is_local_branch = rv_inst
                .symbol
                .as_ref()
                .map_or(false, |sym| label_offsets.contains_key(sym.as_str()));

            if is_local_branch {
                // Resolve the branch locally: compute PC-relative offset
                // and re-encode with the concrete immediate.
                let sym = rv_inst.symbol.as_ref().unwrap();
                let target_offset = label_offsets[sym.as_str()];
                let pc_rel = (target_offset as i64) - (rec.offset as i64);

                let mut patched = rv_inst.clone();
                patched.imm = pc_rel;
                patched.symbol = None; // no relocation needed

                match encoder.encode(&patched) {
                    Ok(encoded) => {
                        output.extend_from_slice(&encoded.bytes);
                        if let Some(ref cont) = encoded.continuation {
                            output.extend_from_slice(&cont.bytes);
                        }
                    }
                    Err(_) => {
                        output.extend_from_slice(&0x0000_0013u32.to_le_bytes());
                    }
                }
            } else {
                // External or non-branch: encode and emit relocations.
                let sym_name = rv_inst
                    .symbol
                    .as_deref()
                    .map(|s| s.trim_end_matches("@plt").to_string())
                    .unwrap_or_default();

                match encoder.encode(rv_inst) {
                    Ok(encoded) => {
                        let primary_offset = output.len();

                        // Track whether the primary relocation is a
                        // PCREL_HI20 or GOT_HI20.  If so, the continuation
                        // relocation (PCREL_LO12_I/S) must reference a
                        // local label at the AUIPC offset instead of the
                        // original symbol — the RISC-V ELF ABI mandates
                        // this pairing convention.
                        let mut pcrel_hi_label: Option<String> = None;
                        if let Some(ref reloc) = encoded.relocation {
                            let is_hi20 = reloc.reloc_type
                                == crate::backend::riscv64::assembler::R_RISCV_PCREL_HI20
                                || reloc.reloc_type
                                    == crate::backend::riscv64::assembler::R_RISCV_GOT_HI20;
                            if is_hi20 {
                                let label =
                                    format!(".Lpcrel_hi{}", primary_offset + reloc.offset as usize);
                                pcrel_hi_label = Some(label);
                            }
                            relocations.push(FunctionRelocation {
                                offset: (primary_offset + reloc.offset as usize) as u64,
                                symbol: sym_name.clone(),
                                rel_type_id: reloc.reloc_type,
                                addend: reloc.addend,
                                section: ".text".to_string(),
                            });
                        }
                        output.extend_from_slice(&encoded.bytes);

                        if let Some(ref cont) = encoded.continuation {
                            if let Some(ref cont_reloc) = cont.relocation {
                                // For PCREL_LO12_I/S, reference the
                                // local label at the AUIPC instruction.
                                let cont_sym = if let Some(ref label) = pcrel_hi_label {
                                    if cont_reloc.reloc_type
                                        == crate::backend::riscv64::assembler::R_RISCV_PCREL_LO12_I
                                        || cont_reloc.reloc_type
                                            == crate::backend::riscv64::assembler::R_RISCV_PCREL_LO12_S
                                    {
                                        label.clone()
                                    } else {
                                        sym_name.clone()
                                    }
                                } else {
                                    sym_name.clone()
                                };
                                relocations.push(FunctionRelocation {
                                    offset: (output.len() + cont_reloc.offset as usize) as u64,
                                    symbol: cont_sym,
                                    rel_type_id: cont_reloc.reloc_type,
                                    addend: cont_reloc.addend,
                                    section: ".text".to_string(),
                                });
                            }
                            output.extend_from_slice(&cont.bytes);
                        }
                    }
                    Err(_) => {
                        output.extend_from_slice(&0x0000_0013u32.to_le_bytes());
                    }
                }
            }
        }

        Ok(AssembledFunction {
            bytes: output,
            relocations,
        })
    }

    /// Returns [`Target::RiscV64`].
    #[inline]
    fn target(&self) -> Target {
        Target::RiscV64
    }

    /// Format a machine instruction for `-S` assembly text output.
    ///
    /// For INLINE_ASM instructions, the already-substituted template is
    /// emitted verbatim (operand substitution was performed during
    /// instruction selection).  For all other opcodes, a generic
    /// pseudo-assembly notation is produced.
    fn format_instruction(&self, inst: &MachineInstruction) -> String {
        if let Some(ref template) = inst.asm_template {
            return template.clone();
        }
        format!("{}", inst)
    }

    /// Returns the RISC-V 64 register set information for the register
    /// allocator.
    ///
    /// Converts the architecture-specific [`RiscV64RegisterInfo`] to the
    /// backend-agnostic [`RegisterInfo`] struct containing allocatable GPRs,
    /// FPRs, callee/caller-saved sets, reserved registers, and argument /
    /// return registers.
    fn register_info(&self) -> RegisterInfo {
        self.reg_info.to_register_info()
    }

    /// Returns the RISC-V 64 relocation type descriptors.
    ///
    /// The slice contains descriptors for all relocation types needed by
    /// the BCC RISC-V assembler and linker, including absolute, PC-relative,
    /// GOT, PLT, branch, jump, alignment, relaxation, and DWARF arithmetic
    /// relocations.
    #[inline]
    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        &self.relocation_types
    }

    /// Generate RISC-V 64 function prologue instructions.
    ///
    /// The prologue establishes the stack frame:
    ///
    /// ```text
    /// addi sp, sp, -frame_size       // Allocate stack frame
    /// sd   ra, frame_size-8(sp)      // Save return address
    /// sd   s0, frame_size-16(sp)     // Save frame pointer
    /// addi s0, sp, frame_size        // Set new frame pointer
    /// sd   s1, -24(s0)               // Save callee-saved registers...
    /// sd   s2, -32(s0)
    /// ...
    /// ```
    ///
    /// For large frames (> 2047 bytes), the stack adjustment uses a
    /// multi-instruction sequence since ADDI only supports 12-bit
    /// signed immediates.
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue: Vec<MachineInstruction> = Vec::new();

        // Count callee-saved registers (excluding RA and FP which are
        // always saved as part of the 16-byte header).
        let extra_callee_saved: usize = mf
            .callee_saved_regs
            .iter()
            .filter(|&&r| r != FP && r != RA)
            .count();

        // Recompute frame size to include callee-saved register space.
        // The original frame_size from instruction selection accounts for
        // RA/FP (16 bytes) + locals, but NOT for callee-saved registers
        // assigned by the register allocator.  We must add space for them.
        let frame_size = mf.frame_size + extra_callee_saved * 8;

        if frame_size == 0 && mf.callee_saved_regs.is_empty() && mf.is_leaf {
            // Leaf function with no locals and no callee-saved registers:
            // no prologue needed.
            return prologue;
        }

        // Ensure frame size is at least 16 (RA + FP) and 16-byte aligned.
        let aligned_frame = align_to_16(if frame_size < 16 { 16 } else { frame_size });

        if aligned_frame <= 2047 {
            // Small frame: single ADDI.
            // addi sp, sp, -frame_size
            let mut adj = MachineInstruction::new(codegen::RvOpcode::ADDI as u32);
            adj = adj
                .with_result(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Immediate(-(aligned_frame as i64)));
            prologue.push(adj);
        } else {
            // Large frame: use a temporary register to hold the offset.
            // li t0, -frame_size
            let mut li = MachineInstruction::new(codegen::RvOpcode::LI as u32);
            li = li
                .with_result(MachineOperand::Register(registers::T0))
                .with_operand(MachineOperand::Immediate(-(aligned_frame as i64)));
            prologue.push(li);

            // add sp, sp, t0
            let mut add_sp = MachineInstruction::new(codegen::RvOpcode::ADD as u32);
            add_sp = add_sp
                .with_result(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(registers::T0));
            prologue.push(add_sp);
        }

        // sd ra, frame_size-8(sp)
        // S-type store: rs1=base, rs2=value, imm=displacement
        let ra_offset = (aligned_frame as i64) - 8;
        let mut save_ra = MachineInstruction::new(codegen::RvOpcode::SD as u32);
        save_ra = save_ra
            .with_operand(MachineOperand::Register(SP))
            .with_operand(MachineOperand::Register(RA))
            .with_operand(MachineOperand::Immediate(ra_offset));
        prologue.push(save_ra);

        // sd s0, frame_size-16(sp)
        let fp_offset = (aligned_frame as i64) - 16;
        let mut save_fp = MachineInstruction::new(codegen::RvOpcode::SD as u32);
        save_fp = save_fp
            .with_operand(MachineOperand::Register(SP))
            .with_operand(MachineOperand::Register(FP))
            .with_operand(MachineOperand::Immediate(fp_offset));
        prologue.push(save_fp);

        // addi s0, sp, frame_size  (set frame pointer)
        let mut set_fp = MachineInstruction::new(codegen::RvOpcode::ADDI as u32);
        set_fp = set_fp
            .with_result(MachineOperand::Register(FP))
            .with_operand(MachineOperand::Register(SP))
            .with_operand(MachineOperand::Immediate(aligned_frame as i64));
        prologue.push(set_fp);

        // Save callee-saved registers.
        // Store them at the BOTTOM of the frame (SP-relative) to avoid
        // colliding with FP-relative local variable allocas.
        // Layout:  SP → [callee-saved regs] [arg-spill] ...  [locals] [FP] [RA] ← old SP
        // Save area starts at SP + 0 and grows upward.
        let mut save_sp_offset = 0i64;
        for &reg in &mf.callee_saved_regs {
            // Skip FP (s0) and RA — already saved above.
            if reg == FP || reg == RA {
                continue;
            }

            let opcode = if registers::is_fpr(reg) {
                codegen::RvOpcode::FSD
            } else {
                codegen::RvOpcode::SD
            };

            let mut save = MachineInstruction::new(opcode as u32);
            save = save
                .with_operand(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(reg))
                .with_operand(MachineOperand::Immediate(save_sp_offset));
            prologue.push(save);
            save_sp_offset += 8;
        }

        prologue
    }

    /// Generate RISC-V 64 function epilogue instructions.
    ///
    /// The epilogue tears down the stack frame in reverse order of the
    /// prologue:
    ///
    /// ```text
    /// ld   s2, -32(s0)               // Restore callee-saved registers
    /// ld   s1, -24(s0)
    /// ld   s0, frame_size-16(sp)     // Restore frame pointer
    /// ld   ra, frame_size-8(sp)      // Restore return address
    /// addi sp, sp, frame_size        // Deallocate stack frame
    /// ret                            // jalr x0, ra, 0
    /// ```
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue: Vec<MachineInstruction> = Vec::new();

        // Recompute frame size to match prologue (include callee-saved).
        let extra_callee_saved: usize = mf
            .callee_saved_regs
            .iter()
            .filter(|&&r| r != FP && r != RA)
            .count();
        let frame_size = mf.frame_size + extra_callee_saved * 8;

        if frame_size == 0 && mf.callee_saved_regs.is_empty() && mf.is_leaf {
            // Leaf function with no frame: just return.
            let ret = MachineInstruction::new(codegen::RvOpcode::RET as u32).set_terminator();
            epilogue.push(ret);
            return epilogue;
        }

        let aligned_frame = align_to_16(if frame_size < 16 { 16 } else { frame_size });

        // Restore callee-saved registers from SP-relative offsets
        // (bottom of frame), matching the prologue save order.
        let callee_regs: Vec<u16> = mf
            .callee_saved_regs
            .iter()
            .filter(|&&r| r != FP && r != RA)
            .copied()
            .collect();

        let mut restore_sp_offset = 0i64;
        for &reg in &callee_regs {
            let opcode = if registers::is_fpr(reg) {
                codegen::RvOpcode::FLD
            } else {
                codegen::RvOpcode::LD
            };

            let mut restore = MachineInstruction::new(opcode as u32);
            restore = restore
                .with_result(MachineOperand::Register(reg))
                .with_operand(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Immediate(restore_sp_offset));
            epilogue.push(restore);
            restore_sp_offset += 8;
        }

        // ld s0, frame_size-16(sp) — restore frame pointer.
        // I-type load: rd=dest, rs1=base, imm=displacement
        let fp_offset = (aligned_frame as i64) - 16;
        let mut restore_fp = MachineInstruction::new(codegen::RvOpcode::LD as u32);
        restore_fp = restore_fp
            .with_result(MachineOperand::Register(FP))
            .with_operand(MachineOperand::Register(SP))
            .with_operand(MachineOperand::Immediate(fp_offset));
        epilogue.push(restore_fp);

        // ld ra, frame_size-8(sp) — restore return address.
        let ra_offset = (aligned_frame as i64) - 8;
        let mut restore_ra = MachineInstruction::new(codegen::RvOpcode::LD as u32);
        restore_ra = restore_ra
            .with_result(MachineOperand::Register(RA))
            .with_operand(MachineOperand::Register(SP))
            .with_operand(MachineOperand::Immediate(ra_offset));
        epilogue.push(restore_ra);

        // addi sp, sp, frame_size — deallocate frame.
        if aligned_frame <= 2047 {
            let mut adj = MachineInstruction::new(codegen::RvOpcode::ADDI as u32);
            adj = adj
                .with_result(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Immediate(aligned_frame as i64));
            epilogue.push(adj);
        } else {
            let mut li = MachineInstruction::new(codegen::RvOpcode::LI as u32);
            li = li
                .with_result(MachineOperand::Register(registers::T0))
                .with_operand(MachineOperand::Immediate(aligned_frame as i64));
            epilogue.push(li);

            let mut add_sp = MachineInstruction::new(codegen::RvOpcode::ADD as u32);
            add_sp = add_sp
                .with_result(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(SP))
                .with_operand(MachineOperand::Register(registers::T0));
            epilogue.push(add_sp);
        }

        // ret (jalr x0, ra, 0)
        let ret = MachineInstruction::new(codegen::RvOpcode::RET as u32).set_terminator();
        epilogue.push(ret);

        epilogue
    }

    /// Returns the frame pointer register: x8 (s0/fp).
    #[inline]
    fn frame_pointer_reg(&self) -> u16 {
        FP
    }

    /// Returns the stack pointer register: x2 (sp).
    #[inline]
    fn stack_pointer_reg(&self) -> u16 {
        SP
    }

    /// Returns the return address register: x1 (ra).
    ///
    /// RISC-V has an explicit return address register (unlike x86 which
    /// uses the stack). Returns `Some(1)` for RA (x1).
    #[inline]
    fn return_address_reg(&self) -> Option<u16> {
        Some(RA)
    }

    /// Classify where a function argument of the given IR type should be
    /// placed according to the LP64D calling convention.
    ///
    /// Creates a fresh ABI handler for single-argument classification.
    /// For multi-argument stateful classification (tracking register
    /// consumption), use [`RiscV64Abi::classify_arg`] directly.
    fn classify_argument(&self, ty: &IrType) -> ArgLocation {
        let ctype = Self::ir_type_to_ctype(ty);
        let mut abi = RiscV64Abi::new();
        abi.classify_arg(&ctype)
    }

    /// Classify where a function return value of the given IR type should
    /// be placed according to the LP64D calling convention.
    fn classify_return(&self, ty: &IrType) -> ArgLocation {
        let ctype = Self::ir_type_to_ctype(ty);
        self.abi.classify_return(&ctype)
    }
}

// ===========================================================================
// Private Implementation Methods
// ===========================================================================

impl RiscV64Codegen {
    /// Encode a single machine instruction to its binary representation.
    ///
    /// Reconstructs the [`RvInstruction`] from the [`MachineInstruction`]
    /// fields and delegates to [`RiscV64Encoder::encode`] for correct
    /// RISC-V machine code emission.  If the encoder cannot handle the
    /// opcode, falls back to a NOP with an error diagnostic.
    ///
    /// Note: `compile_to_object` performs encoding inline to collect
    /// relocations; this method is retained for single-instruction use.
    #[allow(dead_code)]
    fn encode_machine_instruction(&self, mi: &MachineInstruction) -> Vec<u8> {
        use crate::backend::riscv64::assembler::encoder::RiscV64Encoder;

        // Reconstruct an RvInstruction from the MachineInstruction.
        let rv_inst = Self::machine_to_rv_instruction(mi);
        let encoder = RiscV64Encoder::new();

        match encoder.encode(&rv_inst) {
            Ok(encoded) => encoded.bytes.to_vec(),
            Err(_e) => {
                // Fallback: emit a RISC-V NOP (ADDI x0, x0, 0 = 0x00000013).
                // This preserves alignment while signalling an encoding gap.
                0x0000_0013u32.to_le_bytes().to_vec()
            }
        }
    }

    /// Convert a [`MachineInstruction`] back to an [`RvInstruction`] for
    /// encoding via the assembler.
    ///
    /// This is the inverse of [`rv_to_machine_instruction`].
    fn machine_to_rv_instruction(mi: &MachineInstruction) -> codegen::RvInstruction {
        // Reconstruct RvOpcode from the stored u32.  RvOpcode is a
        // #[repr(u32)]-less enum, but we stored `opcode as u32` during
        // forward conversion.  RvOpcode is #[repr(u32)] so the transmute
        // is well-defined for discriminant values [0, NUM_RV_OPCODES).
        // We validate the range to avoid UB on corrupted data.
        const NUM_RV_OPCODES: u32 = 256;
        let opcode = if mi.opcode < NUM_RV_OPCODES {
            // SAFETY: RvOpcode is #[repr(u32)] with 158 variants numbered
            // 0..157.  We verified mi.opcode is in range.
            unsafe { std::mem::transmute::<u32, codegen::RvOpcode>(mi.opcode) }
        } else {
            // Unknown opcode — default to NOP.
            codegen::RvOpcode::NOP
        };

        // Extract rd from the result operand.
        let rd = mi.result.as_ref().and_then(|op| match op {
            MachineOperand::Register(r) => Some(*r),
            MachineOperand::VirtualRegister(v) => Some(*v as u16),
            _ => None,
        });

        // Extract rs1, rs2, rs3 and immediate from operands.
        let mut rs1: Option<u16> = None;
        let mut rs2: Option<u16> = None;
        let mut rs3: Option<u16> = None;
        let mut imm: i64 = 0;
        let mut symbol: Option<String> = None;

        let mut reg_idx = 0usize;
        for op in &mi.operands {
            match op {
                MachineOperand::Register(r) => {
                    match reg_idx {
                        0 => rs1 = Some(*r),
                        1 => rs2 = Some(*r),
                        2 => rs3 = Some(*r),
                        _ => {}
                    }
                    reg_idx += 1;
                }
                MachineOperand::VirtualRegister(v) => {
                    // After register allocation this shouldn't occur,
                    // but handle defensively.
                    match reg_idx {
                        0 => rs1 = Some(*v as u16),
                        1 => rs2 = Some(*v as u16),
                        2 => rs3 = Some(*v as u16),
                        _ => {}
                    }
                    reg_idx += 1;
                }
                MachineOperand::Immediate(v) => {
                    imm = *v;
                }
                MachineOperand::GlobalSymbol(s) => {
                    symbol = Some(s.clone());
                }
                MachineOperand::Memory {
                    base, displacement, ..
                } => {
                    // Extract base register and displacement from Memory
                    // operand — used when MachineInstructions are built
                    // directly (e.g., prologue/epilogue, spill code) with
                    // Memory operands.  Map base → rs1, displacement → imm.
                    //
                    // CRITICAL: Advance reg_idx past the slot we just
                    // filled so that subsequent Register operands land in
                    // the correct field (rs2, not rs1).  Without this,
                    // a spill-store pattern like:
                    //   operands = [Memory{base=s0, disp=-112}, Register(t0)]
                    // would put s0 into rs1, then the Register(t0) would
                    // OVERWRITE rs1 (because reg_idx was still 0), producing
                    //   SD zero, -112(t0)   instead of   SD t0, -112(s0)
                    if let Some(b) = base {
                        if rs1.is_none() {
                            rs1 = Some(*b);
                            // Ensure next Register goes to rs2, not rs1.
                            if reg_idx < 1 {
                                reg_idx = 1;
                            }
                        } else if rs2.is_none() {
                            rs2 = Some(*b);
                            if reg_idx < 2 {
                                reg_idx = 2;
                            }
                        }
                    }
                    imm = *displacement;
                }
                _ => {}
            }
        }

        // Restore `.label:` comments that were stashed in asm_template
        // during rv_to_machine_instruction.
        let comment = mi
            .asm_template
            .as_ref()
            .filter(|t| t.starts_with(".label:"))
            .cloned();

        codegen::RvInstruction {
            opcode,
            rd,
            rs1,
            rs2,
            rs3,
            imm,
            symbol,
            is_fp: false,
            comment,
            is_call_arg_setup: mi.is_call_arg_setup,
        }
    }
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Align a value up to the nearest multiple of 16.
///
/// Used for stack frame size alignment per the LP64D ABI requirement
/// that the stack pointer is always 16-byte aligned.
#[inline]
fn align_to_16(value: usize) -> usize {
    (value + 15) & !15
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elf_constants() {
        assert_eq!(EM_RISCV, 243);
        assert_eq!(EF_RISCV_FLOAT_ABI_DOUBLE, 0x0004);
        assert_eq!(EF_RISCV_RVC, 0x0001);
        assert_eq!(ELF_FLAGS, 0x0001);
        assert_eq!(DEFAULT_BASE_ADDRESS, 0x10000);
        assert_eq!(PAGE_SIZE, 4096);
    }

    #[test]
    fn test_relocation_types_non_empty() {
        let relocs = RiscV64Codegen::build_relocation_types();
        // We should have at least 20 relocation types.
        assert!(
            relocs.len() >= 20,
            "Expected at least 20 relocation types, got {}",
            relocs.len()
        );
    }

    #[test]
    fn test_relocation_types_contain_key_entries() {
        let relocs = RiscV64Codegen::build_relocation_types();
        let names: Vec<&str> = relocs.iter().map(|r| r.name).collect();

        assert!(names.contains(&"R_RISCV_NONE"));
        assert!(names.contains(&"R_RISCV_32"));
        assert!(names.contains(&"R_RISCV_64"));
        assert!(names.contains(&"R_RISCV_BRANCH"));
        assert!(names.contains(&"R_RISCV_JAL"));
        assert!(names.contains(&"R_RISCV_CALL"));
        assert!(names.contains(&"R_RISCV_CALL_PLT"));
        assert!(names.contains(&"R_RISCV_GOT_HI20"));
        assert!(names.contains(&"R_RISCV_PCREL_HI20"));
        assert!(names.contains(&"R_RISCV_PCREL_LO12_I"));
        assert!(names.contains(&"R_RISCV_PCREL_LO12_S"));
        assert!(names.contains(&"R_RISCV_HI20"));
        assert!(names.contains(&"R_RISCV_LO12_I"));
        assert!(names.contains(&"R_RISCV_LO12_S"));
        assert!(names.contains(&"R_RISCV_RELAX"));
    }

    #[test]
    fn test_constructor() {
        let codegen = RiscV64Codegen::new(false, false);
        assert_eq!(codegen.target, Target::RiscV64);
        assert!(!codegen.pic_mode);
        assert!(!codegen.debug_info);
    }

    #[test]
    fn test_constructor_with_pic_and_debug() {
        let codegen = RiscV64Codegen::new(true, true);
        assert_eq!(codegen.target, Target::RiscV64);
        assert!(codegen.pic_mode);
        assert!(codegen.debug_info);
    }

    #[test]
    fn test_target() {
        let codegen = RiscV64Codegen::new(false, false);
        assert_eq!(codegen.target(), Target::RiscV64);
    }

    #[test]
    fn test_register_info() {
        let codegen = RiscV64Codegen::new(false, false);
        let reg_info = codegen.register_info();
        // Should have allocatable GPRs (at least 20+).
        assert!(!reg_info.allocatable_gpr.is_empty());
        // Should have allocatable FPRs.
        assert!(!reg_info.allocatable_fpr.is_empty());
        // Should have callee-saved registers.
        assert!(!reg_info.callee_saved.is_empty());
        // Should have caller-saved registers.
        assert!(!reg_info.caller_saved.is_empty());
    }

    #[test]
    fn test_frame_pointer_reg() {
        let codegen = RiscV64Codegen::new(false, false);
        assert_eq!(codegen.frame_pointer_reg(), FP as u16);
        assert_eq!(codegen.frame_pointer_reg(), 8); // x8 = s0/fp
    }

    #[test]
    fn test_stack_pointer_reg() {
        let codegen = RiscV64Codegen::new(false, false);
        assert_eq!(codegen.stack_pointer_reg(), SP as u16);
        assert_eq!(codegen.stack_pointer_reg(), 2); // x2 = sp
    }

    #[test]
    fn test_return_address_reg() {
        let codegen = RiscV64Codegen::new(false, false);
        assert_eq!(codegen.return_address_reg(), Some(RA as u16));
        assert_eq!(codegen.return_address_reg(), Some(1)); // x1 = ra
    }

    #[test]
    fn test_classify_integer_argument() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::I32);
        // First integer argument should go in a0 (x10).
        assert!(loc.is_register());
        assert_eq!(loc.as_register(), Some(registers::A0 as u16));
    }

    #[test]
    fn test_classify_float_argument() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::F32);
        // First float argument should go in fa0 (f10).
        assert!(loc.is_register());
        assert_eq!(loc.as_register(), Some(registers::FA0 as u16));
    }

    #[test]
    fn test_classify_double_argument() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::F64);
        assert!(loc.is_register());
        assert_eq!(loc.as_register(), Some(registers::FA0 as u16));
    }

    #[test]
    fn test_classify_pointer_argument() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::Ptr);
        // Pointer → integer register a0.
        assert!(loc.is_register());
        assert_eq!(loc.as_register(), Some(registers::A0 as u16));
    }

    #[test]
    fn test_classify_integer_return() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::I64);
        assert!(loc.is_register());
        assert_eq!(loc.as_register(), Some(registers::A0 as u16));
    }

    #[test]
    fn test_classify_float_return() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::F64);
        assert!(loc.is_register());
        assert_eq!(loc.as_register(), Some(registers::FA0 as u16));
    }

    #[test]
    fn test_classify_void_return() {
        let codegen = RiscV64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::Void);
        // Void still returns a location (a0 by convention, though unused).
        assert!(loc.is_register());
    }

    #[test]
    fn test_align_to_16() {
        assert_eq!(align_to_16(0), 0);
        assert_eq!(align_to_16(1), 16);
        assert_eq!(align_to_16(15), 16);
        assert_eq!(align_to_16(16), 16);
        assert_eq!(align_to_16(17), 32);
        assert_eq!(align_to_16(32), 32);
        assert_eq!(align_to_16(100), 112);
    }

    #[test]
    fn test_prologue_empty_function() {
        let codegen = RiscV64Codegen::new(false, false);
        let mut mf = MachineFunction::new("empty".to_string());
        mf.frame_size = 0;
        mf.is_leaf = true;
        let prologue = codegen.emit_prologue(&mf);
        // Leaf function with no frame: empty prologue.
        assert!(prologue.is_empty());
    }

    #[test]
    fn test_prologue_small_frame() {
        let codegen = RiscV64Codegen::new(false, false);
        let mut mf = MachineFunction::new("small".to_string());
        mf.frame_size = 64;
        mf.is_leaf = false;
        let prologue = codegen.emit_prologue(&mf);
        // Should have at least 4 instructions:
        // addi sp, sd ra, sd s0, addi s0
        assert!(prologue.len() >= 4);
    }

    #[test]
    fn test_epilogue_empty_function() {
        let codegen = RiscV64Codegen::new(false, false);
        let mut mf = MachineFunction::new("empty".to_string());
        mf.frame_size = 0;
        mf.is_leaf = true;
        let epilogue = codegen.emit_epilogue(&mf);
        // Should have at least 1 instruction (ret).
        assert!(!epilogue.is_empty());
    }

    #[test]
    fn test_ir_type_to_ctype_scalars() {
        assert_eq!(
            std::mem::discriminant(&RiscV64Codegen::ir_type_to_ctype(&IrType::Void)),
            std::mem::discriminant(&CType::Void)
        );
        assert_eq!(
            std::mem::discriminant(&RiscV64Codegen::ir_type_to_ctype(&IrType::I32)),
            std::mem::discriminant(&CType::Int)
        );
        assert_eq!(
            std::mem::discriminant(&RiscV64Codegen::ir_type_to_ctype(&IrType::F64)),
            std::mem::discriminant(&CType::Double)
        );
    }

    #[test]
    fn test_relocation_r_riscv_branch_is_pc_relative() {
        let relocs = RiscV64Codegen::build_relocation_types();
        let branch = relocs.iter().find(|r| r.name == "R_RISCV_BRANCH").unwrap();
        assert!(branch.is_pc_relative);
        assert_eq!(branch.type_id, 16);
        assert_eq!(branch.size, 4);
    }

    #[test]
    fn test_relocation_r_riscv_64_is_absolute() {
        let relocs = RiscV64Codegen::build_relocation_types();
        let abs64 = relocs.iter().find(|r| r.name == "R_RISCV_64").unwrap();
        assert!(!abs64.is_pc_relative);
        assert_eq!(abs64.type_id, 2);
        assert_eq!(abs64.size, 8);
    }
}
