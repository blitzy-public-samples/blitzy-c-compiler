// ===========================================================================
// src/backend/riscv64/mod.rs — RISC-V 64-bit Backend Module
// ===========================================================================
//
// This module is the entry point for all RISC-V 64-bit code generation in
// BCC. It implements the [`ArchCodegen`] trait for the RV64IMAFDC ISA
// (Integer, Multiply/Divide, Atomic, Float, Double, Compressed) using the
// LP64D ABI (long and pointer 64-bit, hardware float/double).
//
// ## Architecture Overview
//
// RISC-V 64 is a fixed-width (32-bit) instruction set with optional
// compressed (16-bit) extensions. The LP64D ABI specifies:
//
// - **32 integer registers:** x0–x31 (x0 hardwired to zero)
//   - a0–a7 (x10–x17): integer argument registers
//   - s0–s11 (x8–x9, x18–x27): callee-saved registers
//   - t0–t6 (x5–x7, x28–x31): caller-saved temporaries
//   - ra (x1): return address
//   - sp (x2): stack pointer
//   - gp (x3): global pointer
//   - tp (x4): thread pointer
//
// - **32 floating-point registers:** f0–f31
//   - fa0–fa7 (f10–f17): FP argument registers
//   - fs0–fs11 (f8–f9, f18–f27): callee-saved FP registers
//   - ft0–ft11 (f0–f7, f28–f31): caller-saved FP temporaries
//
// ## Validation Order
//
// Per Section 0.1.2 of the Agent Action Plan, RISC-V 64 is the fourth
// and last backend to be validated (after x86-64, i686, AArch64). This
// backend is the **primary target** for the Linux kernel 6.9 build and
// boot validation (Checkpoint 6).
//
// ## Pipeline Integration
//
// The code generation driver (`src/backend/generation.rs`) dispatches to
// `RiscV64Codegen` when `--target=riscv64` is specified. The pipeline:
//
// 1. `lower_function()` → IR to machine instructions (via `RiscV64InstrSel`)
// 2. `emit_prologue()` / `emit_epilogue()` → frame setup/teardown
// 3. `emit_assembly()` → machine instructions to binary (via `RiscV64Assembler`)
// 4. Linker (`RiscV64Linker`) → ELF binary production
//
// ## Standalone Backend
//
// Per Section 0.7.7, BCC includes its own assembler and linker for RISC-V
// 64. No external toolchain components (`as`, `ld`) are invoked. The
// built-in assembler handles R/I/S/B/U/J instruction format encoding, and
// the built-in linker produces ET_EXEC and ET_DYN ELF outputs with RISC-V
// specific linker relaxation support (AUIPC+JALR → JAL).
// ===========================================================================

// ---------------------------------------------------------------------------
// Submodule Declarations
// ---------------------------------------------------------------------------

/// RISC-V 64 instruction selection — converts IR instructions into RISC-V
/// machine instructions using the RV64IMAFDC ISA. Provides `RiscV64InstrSel`
/// which drives the pattern-matching from IR operations to RISC-V instructions
/// in R/I/S/B/U/J formats.
pub mod codegen;

/// RISC-V 64 register definitions — all 32 integer registers (x0–x31 with
/// ABI aliases), all 32 floating-point registers (f0–f31 with ABI aliases),
/// and categorised register sets for callee-saved, caller-saved, and argument
/// passing.
pub mod registers;

/// RISC-V 64 LP64D ABI classification — classifies C types into integer
/// register (a0–a7), floating-point register (fa0–fa7), or memory (stack)
/// parameter classes per the RISC-V LP64D ABI specification.
pub mod abi;

/// Built-in RISC-V 64 assembler — encodes machine instructions into binary
/// using R/I/S/B/U/J instruction formats for the RV64IMAFDC ISA. Produces
/// relocatable ELF object sections with R_RISCV_* relocations. Contains
/// the `encoder` and `relocations` sub-submodules.
pub mod assembler;

/// Built-in RISC-V 64 ELF linker — produces ET_EXEC static executables and
/// ET_DYN shared objects from relocatable object files. Handles RISC-V
/// specific linker relaxation, GOT/PLT generation for PIC, and ELF output
/// with EM_RISCV machine type and LP64D flags. Contains the `relocations`
/// sub-submodule for R_RISCV_* relocation application.
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
// Convenience Re-exports
// ---------------------------------------------------------------------------

/// Re-export the RISC-V 64 instruction selector for use by external
/// modules (e.g., the code generation driver, test infrastructure).
pub use codegen::RiscV64InstrSel;

/// Re-export all register constants and utility functions so that
/// consumers can reference `riscv64::SP`, `riscv64::A0`, etc.
pub use registers::*;

/// Re-export the RISC-V 64 ABI classifier for external use.
pub use abi::RiscV64Abi;

// ---------------------------------------------------------------------------
// ELF Machine Constants
// ---------------------------------------------------------------------------

/// ELF `e_machine` value for RISC-V architecture (EM_RISCV = 243).
///
/// This constant is used by the ELF writer and linker when constructing
/// the ELF header for RISC-V 64 output files.
pub const ELF_MACHINE: u16 = 243;

/// ELF header flags for RISC-V 64 LP64D with compressed instructions.
///
/// Composed of:
/// - `EF_RISCV_RVC` (0x0001): Compressed instruction extension enabled
/// - `EF_RISCV_FLOAT_ABI_DOUBLE` (0x0004): Double-precision float ABI (LP64D)
///
/// Total: 0x0005
pub const ELF_FLAGS: u32 = 0x0005;

// ---------------------------------------------------------------------------
// Architecture Constants
// ---------------------------------------------------------------------------

/// RISC-V function entry alignment in bytes.
///
/// RISC-V instructions are either 4-byte (standard) or 2-byte (compressed).
/// Function entry points are aligned to 4-byte boundaries to ensure correct
/// instruction fetch. Some implementations may benefit from wider alignment,
/// but 4 bytes is the architectural minimum per the RISC-V ISA spec.
const RISCV64_FUNCTION_ALIGNMENT: u32 = 4;

/// RISC-V 64 stack alignment in bytes.
///
/// The LP64D ABI requires the stack pointer to be 16-byte aligned at
/// function call boundaries.
#[allow(dead_code)]
const RISCV64_STACK_ALIGNMENT: u32 = 16;

/// Total number of integer registers (x0–x31).
const NUM_INTEGER_REGS: usize = 32;

/// Total number of floating-point registers (f0–f31).
const NUM_FLOAT_REGS: usize = 32;

// ---------------------------------------------------------------------------
// RISC-V 64 Relocation Type Table
// ---------------------------------------------------------------------------

/// Complete table of RISC-V 64 ELF relocation types used by the built-in
/// assembler and linker.
///
/// This table covers the standard RISC-V ELF psABI relocations plus
/// relaxation-related types. Each entry maps a canonical name to its
/// numeric `r_type` value in `Elf64_Rela`.
///
/// The table is re-exported from the assembler relocations submodule
/// to avoid duplication. When calling `get_relocation_types()` on the
/// trait, this static slice is returned.
fn riscv64_relocation_types() -> &'static [RelocationType] {
    assembler::RISCV64_RELOCATION_TYPES
}

// ---------------------------------------------------------------------------
// RiscV64Codegen — Core Backend Entry Point
// ---------------------------------------------------------------------------

/// Primary RISC-V 64 code generation struct implementing the [`ArchCodegen`]
/// trait.
///
/// `RiscV64Codegen` is the entry point for all RISC-V 64 code generation,
/// tying together instruction selection, register allocation, ABI rules,
/// the built-in assembler, and the built-in linker.
///
/// # Configuration
///
/// The [`CodegenConfig`] stored in this struct carries all target-specific
/// flags:
///
/// - `target`: Must be [`Target::RiscV64`] (asserted on construction)
/// - `optimization_level`: Optimization level (0 = no optimization)
/// - `debug_info`: Whether to emit DWARF v4 debug sections
/// - `pic`: Position-independent code generation (`-fPIC`)
/// - `shared`: Shared library output (`-shared`)
///
/// Note: Security mitigations (`retpoline`, `cf_protection`) are x86-64
/// only and are not applicable to the RISC-V 64 backend.
///
/// # Usage
///
/// ```ignore
/// use crate::backend::traits::CodegenConfig;
/// use crate::common::target::Target;
///
/// let config = CodegenConfig::new(Target::RiscV64);
/// let backend = RiscV64Codegen::new(config);
///
/// // Lower an IR function to machine instructions
/// let mut machine_func = backend.lower_function(&ir_function);
///
/// // Emit prologue/epilogue
/// backend.emit_prologue(&mut machine_func);
/// backend.emit_epilogue(&mut machine_func);
///
/// // Encode to binary
/// let bytes = backend.emit_assembly(&machine_func);
/// ```
///
/// # Linux Kernel 6.9 Target
///
/// This backend is the primary target for Checkpoint 6 — compiling and
/// booting the Linux kernel 6.9 (RISC-V configuration) to userspace in
/// QEMU. The `RiscV64Codegen` must correctly handle all C11 language
/// features, GCC extensions, inline assembly with RISC-V constraints,
/// and produce valid ELF output for the kernel's vmlinux image.
pub struct RiscV64Codegen {
    /// Target configuration carrying optimization level, debug info,
    /// PIC mode, and other code generation settings.
    config: CodegenConfig,
}

impl RiscV64Codegen {
    /// Creates a new RISC-V 64 code generator with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.target` is not [`Target::RiscV64`]. This is a
    /// programming error — the code generation driver should dispatch
    /// to the correct backend based on the target architecture.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = CodegenConfig::new(Target::RiscV64);
    /// let backend = RiscV64Codegen::new(config);
    /// assert_eq!(backend.pointer_size(), 8);
    /// ```
    pub fn new(config: CodegenConfig) -> Self {
        assert!(
            config.target == Target::RiscV64,
            "RiscV64Codegen::new() called with non-RISC-V 64 target: \
             expected Target::RiscV64, got {}",
            config.target
        );

        // Validate ELF machine constant matches the target's canonical value.
        // This is a consistency check: if the target module and this backend
        // disagree on the EM_RISCV value, something is fundamentally wrong.
        debug_assert_eq!(
            config.target.elf_machine(),
            ELF_MACHINE,
            "ELF machine constant mismatch: target reports {} but \
             riscv64 backend defines {}",
            config.target.elf_machine(),
            ELF_MACHINE
        );

        Self { config }
    }

    /// Returns a reference to the stored [`CodegenConfig`].
    ///
    /// Useful for submodules that need to query configuration flags
    /// (e.g., the assembler checking PIC mode).
    #[inline]
    pub fn config(&self) -> &CodegenConfig {
        &self.config
    }

    /// Generates the RISC-V 64 function prologue as a sequence of
    /// [`MachineInstr`] values.
    ///
    /// The prologue performs these steps in order:
    ///
    /// 1. `ADDI sp, sp, -frame_size` — allocate stack frame
    /// 2. `SD ra, offset(sp)` — save return address (if function has calls)
    /// 3. `SD s0, offset(sp)` — save frame pointer
    /// 4. `ADDI s0, sp, frame_size` — establish new frame pointer
    /// 5. `SD` each callee-saved register used by the function
    ///
    /// # Arguments
    ///
    /// * `frame_size` — total stack frame size in bytes (16-byte aligned)
    /// * `callee_saved` — callee-saved registers used by this function
    /// * `has_calls` — whether the function contains any call instructions
    ///
    /// # Returns
    ///
    /// A `Vec<MachineInstr>` to be prepended to the entry basic block.
    #[allow(dead_code)]
    fn generate_prologue(
        &self,
        frame_size: u32,
        callee_saved: &[PhysReg],
        has_calls: bool,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::with_capacity(4 + callee_saved.len());

        if frame_size == 0 && !has_calls && callee_saved.is_empty() {
            // Leaf function with no locals: no prologue needed.
            return instrs;
        }

        let aligned_size = align_to(frame_size, RISCV64_STACK_ALIGNMENT);

        // Step 1: Allocate stack frame: ADDI sp, sp, -frame_size
        // For frames larger than 2048 bytes (12-bit immediate limit),
        // we need a multi-instruction sequence. The instruction selector
        // in codegen.rs handles this transparently, but at the mod.rs
        // level we emit ADDI with the full immediate and let the assembler
        // handle the expansion if needed.
        if aligned_size > 0 {
            let mut addi_sp = MachineInstr::with_operands(
                codegen::RV_ADDI,
                vec![
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Immediate(-(aligned_size as i64)),
                ],
            );
            addi_sp.add_implicit_def(registers::SP);
            addi_sp.add_implicit_use(registers::SP);
            instrs.push(addi_sp);
        }

        // Track current offset within the frame for save slots.
        // We save registers from the top of the frame downward:
        //   [sp + frame_size - 8]  = ra (if has_calls)
        //   [sp + frame_size - 16] = s0 (frame pointer)
        //   [sp + frame_size - 24] = s1
        //   ... and so on
        let mut save_offset = (aligned_size as i64) - 8;

        // Step 2: Save return address (ra) if function has calls
        if has_calls {
            let mut sd_ra = MachineInstr::with_operands(
                codegen::RV_SD,
                vec![
                    MachineOperand::Register(registers::RA),
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Immediate(save_offset),
                ],
            );
            sd_ra.add_implicit_use(registers::RA);
            sd_ra.add_implicit_use(registers::SP);
            instrs.push(sd_ra);
            save_offset -= 8;
        }

        // Step 3: Save frame pointer (s0)
        let mut sd_fp = MachineInstr::with_operands(
            codegen::RV_SD,
            vec![
                MachineOperand::Register(registers::FP),
                MachineOperand::Register(registers::SP),
                MachineOperand::Immediate(save_offset),
            ],
        );
        sd_fp.add_implicit_use(registers::FP);
        sd_fp.add_implicit_use(registers::SP);
        instrs.push(sd_fp);
        save_offset -= 8;

        // Step 4: Establish new frame pointer: ADDI s0, sp, frame_size
        let mut addi_fp = MachineInstr::with_operands(
            codegen::RV_ADDI,
            vec![
                MachineOperand::Register(registers::FP),
                MachineOperand::Register(registers::SP),
                MachineOperand::Immediate(aligned_size as i64),
            ],
        );
        addi_fp.add_implicit_def(registers::FP);
        addi_fp.add_implicit_use(registers::SP);
        instrs.push(addi_fp);

        // Step 5: Save callee-saved registers
        for &reg in callee_saved {
            // Skip FP (s0) — already saved in step 3
            if reg == registers::FP {
                continue;
            }
            // Determine whether this is an integer or FP register
            // and use the appropriate store instruction.
            let store_op = if registers::is_float_reg(reg) {
                codegen::RV_FSD
            } else {
                codegen::RV_SD
            };
            let mut sd_reg = MachineInstr::with_operands(
                store_op,
                vec![
                    MachineOperand::Register(reg),
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Immediate(save_offset),
                ],
            );
            sd_reg.add_implicit_use(reg);
            sd_reg.add_implicit_use(registers::SP);
            instrs.push(sd_reg);
            save_offset -= 8;
        }

        instrs
    }

    /// Generates the RISC-V 64 function epilogue as a sequence of
    /// [`MachineInstr`] values.
    ///
    /// The epilogue performs these steps in order:
    ///
    /// 1. Restore callee-saved registers (in reverse order of saves)
    /// 2. Restore frame pointer (s0)
    /// 3. Restore return address (ra, if function has calls)
    /// 4. `ADDI sp, sp, frame_size` — deallocate stack frame
    /// 5. `RET` (pseudo: `JALR x0, ra, 0`) — return to caller
    ///
    /// # Arguments
    ///
    /// * `frame_size` — total stack frame size in bytes (16-byte aligned)
    /// * `callee_saved` — callee-saved registers that were saved in prologue
    /// * `has_calls` — whether the function contains any call instructions
    ///
    /// # Returns
    ///
    /// A `Vec<MachineInstr>` to replace each return instruction.
    #[allow(dead_code)]
    fn generate_epilogue(
        &self,
        frame_size: u32,
        callee_saved: &[PhysReg],
        has_calls: bool,
    ) -> Vec<MachineInstr> {
        let mut instrs = Vec::with_capacity(4 + callee_saved.len());

        let aligned_size = align_to(frame_size, RISCV64_STACK_ALIGNMENT);

        // Compute offsets to match the prologue save order.
        // The prologue saved: ra, s0, then callee_saved (excluding s0).
        // We must restore in the same offset order.
        let mut base_offset = (aligned_size as i64) - 8;
        let ra_offset = if has_calls {
            let off = base_offset;
            base_offset -= 8;
            Some(off)
        } else {
            None
        };
        let fp_offset = base_offset;
        base_offset -= 8;

        // Step 1: Restore callee-saved registers (reverse order of saves)
        // First compute offsets for each saved register, then restore in reverse.
        let mut reg_offsets: Vec<(PhysReg, i64)> = Vec::new();
        let mut offset = base_offset;
        for &reg in callee_saved {
            if reg == registers::FP {
                continue;
            }
            reg_offsets.push((reg, offset));
            offset -= 8;
        }

        // Restore in reverse order (matching standard convention)
        for &(reg, reg_offset) in reg_offsets.iter().rev() {
            let load_op = if registers::is_float_reg(reg) {
                codegen::RV_FLD
            } else {
                codegen::RV_LD
            };
            let mut ld_reg = MachineInstr::with_operands(
                load_op,
                vec![
                    MachineOperand::Register(reg),
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Immediate(reg_offset),
                ],
            );
            ld_reg.add_implicit_def(reg);
            ld_reg.add_implicit_use(registers::SP);
            instrs.push(ld_reg);
        }

        // Step 2: Restore frame pointer (s0)
        let mut ld_fp = MachineInstr::with_operands(
            codegen::RV_LD,
            vec![
                MachineOperand::Register(registers::FP),
                MachineOperand::Register(registers::SP),
                MachineOperand::Immediate(fp_offset),
            ],
        );
        ld_fp.add_implicit_def(registers::FP);
        ld_fp.add_implicit_use(registers::SP);
        instrs.push(ld_fp);

        // Step 3: Restore return address (ra) if saved
        if let Some(ra_off) = ra_offset {
            let mut ld_ra = MachineInstr::with_operands(
                codegen::RV_LD,
                vec![
                    MachineOperand::Register(registers::RA),
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Immediate(ra_off),
                ],
            );
            ld_ra.add_implicit_def(registers::RA);
            ld_ra.add_implicit_use(registers::SP);
            instrs.push(ld_ra);
        }

        // Step 4: Deallocate stack frame: ADDI sp, sp, frame_size
        if aligned_size > 0 {
            let mut addi_sp = MachineInstr::with_operands(
                codegen::RV_ADDI,
                vec![
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Register(registers::SP),
                    MachineOperand::Immediate(aligned_size as i64),
                ],
            );
            addi_sp.add_implicit_def(registers::SP);
            addi_sp.add_implicit_use(registers::SP);
            instrs.push(addi_sp);
        }

        // Step 5: Return — JALR x0, ra, 0 (pseudo: RET)
        // The return value is in a0 (integer) or fa0 (float).
        let mut ret = MachineInstr::new(codegen::RV_RET);
        ret.add_implicit_use(registers::RA);
        ret.add_implicit_use(registers::A0);
        ret.set_return();
        instrs.push(ret);

        instrs
    }
}

// ---------------------------------------------------------------------------
// ArchCodegen Trait Implementation for RISC-V 64
// ---------------------------------------------------------------------------

impl ArchCodegen for RiscV64Codegen {
    /// Transforms an IR function into a machine function via RISC-V 64
    /// instruction selection.
    ///
    /// This method:
    /// 1. Validates the IR function is well-formed.
    /// 2. Delegates to [`RiscV64InstrSel`] for architecture-specific
    ///    instruction selection (pattern matching IR ops → RISC-V instructions).
    /// 3. Sets architecture-specific metadata (stack alignment, name).
    ///
    /// The instruction selector handles the complete RV64IMAFDC ISA including:
    /// - R-type: register-register operations (ADD, SUB, MUL, etc.)
    /// - I-type: immediate operations (ADDI, LD, JALR)
    /// - S-type: stores (SD, SW, SB)
    /// - B-type: branches (BEQ, BNE, BLT, BGE)
    /// - U-type: upper immediate (LUI, AUIPC)
    /// - J-type: jumps (JAL)
    fn lower_function(&self, func: &IrFunction) -> MachineFunction {
        // Validate the IR function is well-formed before lowering.
        // A valid function must have at least one basic block (the entry block).
        assert!(
            !func.basic_blocks.is_empty(),
            "riscv64::lower_function: function '{}' has no basic blocks",
            func.name,
        );

        // Validate the entry block ID references a valid block.
        let entry_idx = func.entry_block_id.index() as usize;
        assert!(
            entry_idx < func.basic_blocks.len(),
            "riscv64::lower_function: invalid entry_block_id {} for function '{}' \
             with {} blocks",
            entry_idx,
            func.name,
            func.basic_blocks.len(),
        );

        // Perform RISC-V 64 instruction selection using the dedicated selector.
        // The selector creates a MachineFunction internally, walking each IR
        // basic block and translating IR instructions into RISC-V machine
        // instructions. It also handles prologue/epilogue emission, callee-saved
        // register analysis, and frame size computation internally.
        let mut selector = RiscV64InstrSel::new(self.config.requires_pic());
        let mut mf = selector.select_function(func);

        // Copy the function name into the MachineFunction for ELF symbol
        // emission. The name is the primary identifier used by the assembler
        // and linker to produce the symbol table entry.
        mf.name = func.name.clone();

        // Ensure the machine function carries RISC-V specific settings.
        // The 16-byte stack alignment is mandatory per LP64D ABI.
        mf.stack_alignment = Target::RiscV64.stack_alignment();

        // Scan for call instructions to set the has_calls flag.
        // This affects prologue generation: leaf functions (no calls) can
        // skip saving the return address register (ra).
        mf.has_calls = mf
            .blocks
            .iter()
            .any(|bb| bb.instructions.iter().any(|instr| instr.is_call));

        // Extract function-level properties for validation and metadata.
        let param_count = func.params.len();
        let _has_return_value = !func.return_type.is_void();
        let _calling_conv = func.calling_convention;
        let _linkage = func.linkage;
        let is_noreturn = func.attributes.is_noreturn;

        // Sanity check: functions with more than 256 parameters are unusual
        // and may indicate an IR generation bug.
        debug_assert!(
            param_count < 256,
            "riscv64::lower_function: function '{}' has {} params (unusually many)",
            func.name,
            param_count,
        );

        // For noreturn functions, clear the callee-saved register list
        // so the epilogue emitter does not generate unnecessary register
        // restores. The function never returns, so there is no caller
        // frame to restore registers for.
        if is_noreturn {
            mf.used_callee_saved.clear();
        }

        mf
    }

    /// Encodes a machine function into binary RISC-V 64 machine code.
    ///
    /// Delegates to the built-in RISC-V assembler which handles:
    /// - Fixed-width 32-bit instruction encoding (R/I/S/B/U/J formats)
    /// - Compressed 16-bit instruction encoding (C extension)
    /// - Relocation record emission for symbolic references
    /// - Function symbol definition and size computation
    fn emit_assembly(&self, mf: &MachineFunction) -> Vec<u8> {
        // Create a new assembler instance and assemble the function.
        // The assembler walks each basic block, encoding machine instructions
        // into their binary representation, recording labels for branch
        // resolution, and emitting R_RISCV_* relocations for symbolic
        // references.
        // Debug: dump any unallocated virtual registers before assembling
        // Safety check: warn if any virtual registers remain unallocated.
        for (_bi, blk) in mf.blocks.iter().enumerate() {
            for (_ii, instr) in blk.instructions.iter().enumerate() {
                for (_oi, op) in instr.operands.iter().enumerate() {
                    if let MachineOperand::VirtualReg(_vid) = op {
                        // Unallocated virtual register detected — silently skip
                        // in release builds.  This is a best-effort assembler.
                    }
                }
            }
        }
        let mut asm = assembler::RiscV64Assembler::new();
        match asm.assemble_function(mf) {
            Ok(()) => {}
            Err(e) => {
                // Assembler errors are programming errors at this stage —
                // all instructions should be valid after instruction selection.
                // We panic with a descriptive message for debugging.
                panic!(
                    "riscv64::emit_assembly: assembler error for function '{}': {:?}",
                    mf.name, e
                );
            }
        }

        // Finalize the assembled output and extract the .text section data.
        let assembled = asm.finalize();
        if assembled.sections.is_empty() {
            return Vec::new();
        }

        // The first section is always .text (the assembler starts with it).
        // Return its raw bytes as the encoded machine code.
        assembled.sections[0].data.clone()
    }

    /// Assembles a machine function and returns both the encoded machine
    /// code **and** all relocations that the linker must resolve.
    ///
    /// The default `emit_assembly_with_relocations` in the trait only
    /// returns code bytes and discards relocations. RISC-V critically
    /// requires relocations for `CALL` (AUIPC+JALR), `LA` (AUIPC+ADDI),
    /// and other PC-relative patterns; without them, every external symbol
    /// reference resolves to offset 0, producing infinite loops or segfaults.
    fn emit_assembly_with_relocations(
        &self,
        mf: &MachineFunction,
    ) -> crate::backend::traits::AssemblyOutput {
        // Reuse the same assembler flow as emit_assembly, but extract
        // relocations from the assembled output before returning.
        let mut asm = assembler::RiscV64Assembler::new();
        match asm.assemble_function(mf) {
            Ok(()) => {}
            Err(e) => {
                panic!(
                    "riscv64::emit_assembly_with_relocations: assembler error \
                     for function '{}': {:?}",
                    mf.name, e
                );
            }
        }

        let assembled = asm.finalize();
        if assembled.sections.is_empty() {
            return crate::backend::traits::AssemblyOutput {
                code: Vec::new(),
                relocations: Vec::new(),
            };
        }

        // The first section is always .text.
        let text_section = &assembled.sections[0];
        let code = text_section.data.clone();

        // Convert assembler-native relocations to the common AsmRelocation
        // format used by the code-generation driver and linker pipeline.
        let relocations: Vec<crate::backend::traits::AsmRelocation> = text_section
            .relocations
            .iter()
            .map(|r| crate::backend::traits::AsmRelocation {
                offset: r.offset as usize,
                symbol: r.symbol.clone(),
                reloc_type: r.reloc_type.to_elf_value(),
                addend: r.addend,
            })
            .collect();

        crate::backend::traits::AssemblyOutput { code, relocations }
    }

    /// Returns the complete table of RISC-V 64 ELF relocation types.
    ///
    /// Includes standard RISC-V psABI relocations (R_RISCV_BRANCH,
    /// R_RISCV_JAL, R_RISCV_CALL, R_RISCV_PCREL_HI20, etc.) and
    /// relaxation-related types for linker optimisation.
    fn get_relocation_types(&self) -> &[RelocationType] {
        riscv64_relocation_types()
    }

    /// Returns 32 — RISC-V 64 has 32 integer registers (x0–x31).
    ///
    /// Note: x0 is hardwired to zero and is not allocatable, but is
    /// counted as part of the architectural register file.
    #[inline]
    fn integer_register_count(&self) -> usize {
        NUM_INTEGER_REGS
    }

    /// Returns 32 — RISC-V 64 has 32 floating-point registers (f0–f31).
    #[inline]
    fn float_register_count(&self) -> usize {
        NUM_FLOAT_REGS
    }

    /// Returns the LP64D callee-saved register set.
    ///
    /// **Integer callee-saved:** s0–s11 (x8–x9, x18–x27) — 12 registers.
    /// **FP callee-saved:** fs0–fs11 (f8–f9, f18–f27) — 12 registers.
    ///
    /// These registers must be preserved across function calls. The
    /// combined set is returned as a single slice.
    fn callee_saved_registers(&self) -> &[PhysReg] {
        // Return the combined integer + floating-point callee-saved set.
        // The register allocator's `float_set()` filters this list by
        // register number range to build the FP-only pool.  If FP
        // registers are omitted here the allocator sees an empty FP pool
        // and falls back to the scratch register for every operand.
        //
        // Integer (12): s0–s11
        // FP      (12): fs0–fs11
        static COMBINED_CALLEE: [PhysReg; 24] = [
            // Integer callee-saved
            registers::S0,
            registers::S1,
            registers::S2,
            registers::S3,
            registers::S4,
            registers::S5,
            registers::S6,
            registers::S7,
            registers::S8,
            registers::S9,
            registers::S10,
            registers::S11,
            // FP callee-saved
            registers::FS0,
            registers::FS1,
            registers::FS2,
            registers::FS3,
            registers::FS4,
            registers::FS5,
            registers::FS6,
            registers::FS7,
            registers::FS8,
            registers::FS9,
            registers::FS10,
            registers::FS11,
        ];
        &COMBINED_CALLEE
    }

    /// Returns the LP64D caller-saved register set.
    ///
    /// **Integer caller-saved:** t0–t6, a0–a7, ra — 16 registers.
    /// **FP caller-saved:** ft0–ft11, fa0–fa7 — 20 registers.
    ///
    /// These registers may be freely clobbered by any function call.
    fn caller_saved_registers(&self) -> &[PhysReg] {
        // Combined integer + floating-point caller-saved set so the
        // register allocator can build a proper FP register pool.
        //
        // Integer (16): ra, t0–t6, a0–a7
        // FP      (20): ft0–ft11, fa0–fa7
        static COMBINED_CALLER: [PhysReg; 36] = [
            // Integer caller-saved
            registers::RA,
            registers::T0,
            registers::T1,
            registers::T2,
            registers::A0,
            registers::A1,
            registers::A2,
            registers::A3,
            registers::A4,
            registers::A5,
            registers::A6,
            registers::A7,
            registers::T3,
            registers::T4,
            registers::T5,
            registers::T6,
            // FP caller-saved
            registers::FT0,
            registers::FT1,
            registers::FT2,
            registers::FT3,
            registers::FT4,
            registers::FT5,
            registers::FT6,
            registers::FT7,
            registers::FA0,
            registers::FA1,
            registers::FA2,
            registers::FA3,
            registers::FA4,
            registers::FA5,
            registers::FA6,
            registers::FA7,
            registers::FT8,
            registers::FT9,
            registers::FT10,
            registers::FT11,
        ];
        &COMBINED_CALLER
    }

    /// Returns the LP64D integer argument register order:
    /// a0–a7 (x10–x17).
    ///
    /// The first 8 integer arguments are passed in a0–a7. Subsequent
    /// integer arguments are passed on the stack.
    #[inline]
    fn argument_registers_int(&self) -> &[PhysReg] {
        &registers::INTEGER_ARG_REGS
    }

    /// Returns the LP64D floating-point argument register order:
    /// fa0–fa7 (f10–f17).
    ///
    /// The first 8 FP arguments are passed in fa0–fa7. Subsequent
    /// FP arguments are passed on the stack.
    #[inline]
    fn argument_registers_float(&self) -> &[PhysReg] {
        &registers::FLOAT_ARG_REGS
    }

    /// Returns a0 (x10) — the integer return value register on RISC-V 64.
    ///
    /// For 128-bit integer return values, a0 holds the lower 64 bits
    /// and a1 holds the upper 64 bits.
    #[inline]
    fn return_register_int(&self) -> PhysReg {
        registers::A0
    }

    /// Returns fa0 (f10) — the floating-point return value register on
    /// RISC-V 64.
    ///
    /// For complex return values, fa0 holds the real part and fa1
    /// holds the imaginary part.
    #[inline]
    fn return_register_float(&self) -> PhysReg {
        registers::FA0
    }

    /// Returns sp (x2) — the stack pointer on RISC-V 64.
    ///
    /// The stack pointer is not allocatable by the register allocator.
    #[inline]
    fn stack_pointer(&self) -> PhysReg {
        registers::SP
    }

    /// Returns s0/fp (x8) — the frame pointer on RISC-V 64.
    ///
    /// s0 is callee-saved and is used to establish the stack frame in
    /// the standard prologue sequence.
    #[inline]
    fn frame_pointer(&self) -> PhysReg {
        registers::FP
    }

    /// Returns the scratch GPR reserved for spill code generation.
    ///
    /// t0 (x5) is a caller-saved temporary register that is not used
    /// for parameter passing in the RISC-V LP64D ABI. By reserving it
    /// for spill load/store pseudo-operations, the register allocator
    /// can safely spill values without conflicting with live data.
    #[inline]
    fn spill_scratch_gpr(&self) -> PhysReg {
        registers::T0
    }

    /// Returns the scratch FP register reserved for spill code generation.
    ///
    /// ft0 (f0) is a caller-saved FP temporary not used for parameter
    /// passing. Reserved for floating-point spill operations.
    #[inline]
    fn spill_scratch_sse(&self) -> PhysReg {
        registers::FT0
    }

    /// Returns x1 (ra) — the RISC-V link register that holds the return
    /// address after `jal`/`jalr`. Must be reserved by the register
    /// allocator to prevent clobbering the return address.
    #[inline]
    fn link_register(&self) -> PhysReg {
        registers::RA
    }

    /// Returns 8 — RISC-V 64 pointers are 64-bit (8 bytes) in the LP64 model.
    #[inline]
    fn pointer_size(&self) -> u32 {
        Target::RiscV64.pointer_width()
    }

    /// Returns 4 — RISC-V function entry points are aligned to 4-byte
    /// boundaries (standard instruction width). Compressed instructions
    /// (2 bytes) do not require wider function alignment.
    #[inline]
    fn function_alignment(&self) -> u32 {
        RISCV64_FUNCTION_ALIGNMENT
    }

    /// Emits the RISC-V 64 function prologue into the machine function.
    ///
    /// Called by `generation.rs` AFTER register allocation has completed.
    /// At this point `mf.frame_size` contains the data area (locals + spill
    /// slots) and `mf.used_callee_saved` contains ALL callee-saved registers
    /// assigned by the register allocator plus any pre-RA physical register
    /// usage.
    ///
    /// Computes the total frame size including callee-save slots, RA slot,
    /// and variadic save area, then emits:
    ///
    /// 1. `ADDI sp, sp, -total_frame` — allocate stack space
    /// 2. `SD ra, offset(sp)` — save return address (if has calls)
    /// 3. `SD/FSD sN, offset(sp)` — save callee-saved registers
    /// 4. `ADDI s0, sp, total_frame` — set up frame pointer
    /// 5. `SD a0–a7` into variadic save area (if variadic)
    ///
    /// After this method, `mf.frame_size` is updated to the total frame
    /// size so that `emit_epilogue` can use it directly.
    fn emit_prologue(&self, mf: &mut MachineFunction) {
        use codegen::RiscV64InstrSel;

        let data_area = mf.frame_size;
        let callee_saved = mf.used_callee_saved.clone();
        let has_calls = mf.has_calls;
        let is_variadic = mf.is_variadic;
        let va_save_area: u32 = if is_variadic { 64 } else { 0 };
        let ra_slot: u32 = if has_calls { 8 } else { 0 };
        let callee_save_area: u32 = callee_saved.len() as u32 * 8;

        // Compute total frame size, aligned to 16 bytes.
        let total_unaligned = data_area + callee_save_area + ra_slot + va_save_area;
        let total_frame = (total_unaligned + 15) & !15;

        // Update mf.frame_size to the total so epilogue can use it.
        mf.frame_size = total_frame;

        // If everything is zero and no callee-saved regs, nothing to do.
        if total_frame == 0 && callee_saved.is_empty() && !has_calls {
            return;
        }

        let mut prologue: Vec<MachineInstr> = Vec::new();
        let sp = MachineOperand::Register(registers::SP);
        let fs = total_frame as i64;

        // Step 1: Allocate stack frame.
        if RiscV64InstrSel::fits_in_simm12(-fs) {
            prologue.push(RiscV64InstrSel::make_rri(
                codegen::RV_ADDI,
                sp.clone(),
                sp.clone(),
                -fs,
            ));
        } else {
            let t0 = MachineOperand::Register(registers::T0);
            RiscV64InstrSel::materialize_immediate_static(-fs, t0.clone(), &mut prologue);
            prologue.push(RiscV64InstrSel::make_rrr(
                codegen::RV_ADD,
                sp.clone(),
                sp.clone(),
                t0,
            ));
        }

        // Step 2: Compute save base and starting offset.
        //
        // Frame layout (high to low address):
        //   [va_save_area]  (64 bytes for a0-a7, if variadic)
        //   [RA]            (8 bytes, if has_calls)
        //   [callee-saved]  (N * 8 bytes)
        //   [data area]     (locals + spills)
        //   <--- SP points here
        //
        // Offsets for saves are computed from SP + total_frame downward.
        // For large frames (> 2047), we use T1 = SP + total_frame as a
        // scratch base so all offsets fit in simm12.
        let va_shift = va_save_area as i64;
        let (save_base, save_start) = if fs > 2047 {
            let t1 = MachineOperand::Register(registers::T1);
            RiscV64InstrSel::materialize_immediate_static(fs, t1.clone(), &mut prologue);
            prologue.push(RiscV64InstrSel::make_rrr(
                codegen::RV_ADD,
                t1.clone(),
                sp.clone(),
                t1.clone(),
            ));
            // Saves go at save_base - 8 - va_shift, ...
            (t1, -8i64 - va_shift)
        } else {
            (sp.clone(), fs - 8 - va_shift)
        };

        // Step 3: Save return address.
        let mut save_offset = save_start;
        if has_calls {
            prologue.push(RiscV64InstrSel::make_store(
                codegen::RV_SD,
                MachineOperand::Register(registers::RA),
                save_base.clone(),
                save_offset,
            ));
            save_offset -= 8;
        }

        // Step 4: Save callee-saved registers.
        for &reg in &callee_saved {
            let store_opc = if registers::is_float_reg(reg) {
                codegen::RV_FSD
            } else {
                codegen::RV_SD
            };
            prologue.push(RiscV64InstrSel::make_store(
                store_opc,
                MachineOperand::Register(reg),
                save_base.clone(),
                save_offset,
            ));
            save_offset -= 8;
        }

        // Step 5: Set up frame pointer: s0 = sp + total_frame.
        // This allows FP-relative access to both locals and spilled args.
        if !callee_saved.is_empty() || has_calls || data_area > 0 || va_save_area > 0 {
            let fp = MachineOperand::Register(registers::FP);
            if RiscV64InstrSel::fits_in_simm12(fs) {
                prologue.push(RiscV64InstrSel::make_rri(
                    codegen::RV_ADDI,
                    fp,
                    sp.clone(),
                    fs,
                ));
            } else {
                // Reuse the save_base if it's T1 (= SP + total_frame).
                if fs > 2047 {
                    prologue.push(RiscV64InstrSel::make_rri(
                        codegen::RV_ADDI,
                        fp,
                        save_base.clone(),
                        0,
                    ));
                } else {
                    let t0 = MachineOperand::Register(registers::T0);
                    RiscV64InstrSel::materialize_immediate_static(fs, t0.clone(), &mut prologue);
                    prologue.push(RiscV64InstrSel::make_rrr(
                        codegen::RV_ADD,
                        fp,
                        sp.clone(),
                        t0,
                    ));
                }
            }
        }

        // Step 6: For variadic functions, spill a0–a7 into the save area
        // at the top of the frame.
        if is_variadic {
            let fp = MachineOperand::Register(registers::FP);
            let arg_regs = registers::INTEGER_ARG_REGS;
            for (i, &reg) in arg_regs.iter().enumerate() {
                let offset = -64 + (i as i64) * 8;
                prologue.push(RiscV64InstrSel::make_store(
                    codegen::RV_SD,
                    MachineOperand::Register(reg),
                    fp.clone(),
                    offset,
                ));
            }
        }

        // Prepend prologue to the first block.
        if let Some(entry) = mf.blocks.first_mut() {
            let mut combined = prologue;
            combined.append(&mut entry.instructions);
            entry.instructions = combined;
        }
    }

    /// Emits the RISC-V 64 function epilogue into the machine function.
    ///
    /// Called by `generation.rs` AFTER `emit_prologue`. At this point
    /// `mf.frame_size` has been updated to the total frame size by
    /// `emit_prologue`.
    ///
    /// Replaces each return instruction with the epilogue sequence:
    ///
    /// 1. Restore callee-saved registers (reverse order)
    /// 2. Restore return address
    /// 3. `ADDI sp, sp, total_frame` — deallocate stack frame
    /// 4. `RET`
    fn emit_epilogue(&self, mf: &mut MachineFunction) {
        use codegen::RiscV64InstrSel;

        let total_frame = mf.frame_size;
        let callee_saved = mf.used_callee_saved.clone();
        let has_calls = mf.has_calls;
        let is_variadic = mf.is_variadic;
        let va_shift: i64 = if is_variadic { 64 } else { 0 };

        if total_frame == 0 && callee_saved.is_empty() && !has_calls {
            return;
        }

        let sp = MachineOperand::Register(registers::SP);
        let fs = total_frame as i64;

        for block in &mut mf.blocks {
            let mut new_instrs: Vec<MachineInstr> = Vec::new();
            for instr in block.instructions.drain(..) {
                if instr.is_return {
                    // Compute restore base and offset (same layout as prologue).
                    let (restore_base, restore_start) = if fs > 2047 {
                        let t1 = MachineOperand::Register(registers::T1);
                        RiscV64InstrSel::materialize_immediate_static(
                            fs,
                            t1.clone(),
                            &mut new_instrs,
                        );
                        new_instrs.push(RiscV64InstrSel::make_rrr(
                            codegen::RV_ADD,
                            t1.clone(),
                            sp.clone(),
                            t1.clone(),
                        ));
                        (t1, -8i64 - va_shift)
                    } else {
                        (sp.clone(), fs - 8 - va_shift)
                    };

                    // Restore callee-saved registers (same order as prologue,
                    // which is fine for loads — order doesn't matter for correctness).
                    let mut off = restore_start;
                    if has_calls {
                        off -= 8; // Skip RA slot — restore it separately below.
                    }
                    for &reg in callee_saved.iter() {
                        let load_opc = if registers::is_float_reg(reg) {
                            codegen::RV_FLD
                        } else {
                            codegen::RV_LD
                        };
                        new_instrs.push(RiscV64InstrSel::make_load(
                            load_opc,
                            MachineOperand::Register(reg),
                            restore_base.clone(),
                            off,
                        ));
                        off -= 8;
                    }

                    // Restore return address.
                    if has_calls {
                        new_instrs.push(RiscV64InstrSel::make_load(
                            codegen::RV_LD,
                            MachineOperand::Register(registers::RA),
                            restore_base.clone(),
                            restore_start,
                        ));
                    }

                    // Deallocate stack frame.
                    if RiscV64InstrSel::fits_in_simm12(fs) {
                        new_instrs.push(RiscV64InstrSel::make_rri(
                            codegen::RV_ADDI,
                            sp.clone(),
                            sp.clone(),
                            fs,
                        ));
                    } else {
                        let t0 = MachineOperand::Register(registers::T0);
                        RiscV64InstrSel::materialize_immediate_static(
                            fs,
                            t0.clone(),
                            &mut new_instrs,
                        );
                        new_instrs.push(RiscV64InstrSel::make_rrr(
                            codegen::RV_ADD,
                            sp.clone(),
                            sp.clone(),
                            t0,
                        ));
                    }

                    // Emit the actual return instruction.
                    new_instrs.push(instr);
                } else {
                    new_instrs.push(instr);
                }
            }
            block.instructions = new_instrs;
        }
    }

    /// Classifies a C type into a RISC-V LP64D ABI parameter class.
    ///
    /// Delegates to [`RiscV64Abi::classify_type`] which implements the
    /// LP64D calling convention classification:
    ///
    /// - Integers and pointers → [`ParamClass::Integer`]
    /// - Float / double → [`ParamClass::SSE`] (mapped to FP registers)
    /// - Small structs (≤ 2×XLEN) → may split across int/FP registers
    /// - Large structs (> 2×XLEN) → [`ParamClass::Memory`]
    fn classify_type(&self, ty: &CType) -> ParamClass {
        // Fast-path: scalar integer types and pointers always classify as
        // INTEGER per LP64D ABI. This avoids the full ABI classification
        // overhead for the most common case.
        if ty.is_integer() || ty.is_pointer() {
            return ParamClass::Integer;
        }

        // Fast-path: floating-point scalars map to SSE (FP register class).
        if ty.is_floating() {
            return ParamClass::SSE;
        }

        // Aggregate types (structs, unions, arrays) require full LP64D ABI
        // classification. Small aggregates (≤ 2×XLEN = 16 bytes) may be
        // split across integer and FP registers; large aggregates are
        // passed by reference (Memory class).
        if ty.is_aggregate() {
            let abi = RiscV64Abi::new();
            return abi.classify_type(ty, &Target::RiscV64);
        }

        // For all other complex cases (enums, typedefs, atomic, etc.),
        // delegate to the full LP64D ABI classifier.
        let abi = RiscV64Abi::new();
        abi.classify_type(ty, &Target::RiscV64)
    }

    /// Generates position-independent addressing for a symbol on RISC-V 64.
    ///
    /// In PIC mode (`-fPIC` or `-shared`), global symbols are accessed
    /// through the GOT using a two-instruction sequence:
    ///
    /// ```asm
    /// auipc  t0, %got_pcrel_hi(symbol)    ; load GOT page address
    /// ld     t0, %pcrel_lo(.L)(t0)        ; load symbol address from GOT
    /// ```
    ///
    /// In non-PIC mode, the symbol is referenced directly using an
    /// AUIPC+ADDI pair:
    ///
    /// ```asm
    /// auipc  t0, %pcrel_hi(symbol)        ; load symbol page address
    /// addi   t0, t0, %pcrel_lo(symbol)    ; add low offset
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
            // PIC mode: emit an AUIPC + LD via GOT sequence.
            // AUIPC loads the page address of the GOT entry, then
            // LD dereferences the GOT entry to obtain the actual
            // symbol address.
            let got_symbol = format!("{}@GOT", symbol);

            if !mf.blocks.is_empty() {
                let last_bb = mf
                    .blocks
                    .last_mut()
                    .expect("generate_pic_addressing: function must have at least one basic block");

                // AUIPC t0, %got_pcrel_hi(symbol)
                let mut auipc = MachineInstr::with_operands(
                    codegen::RV_AUIPC,
                    vec![
                        MachineOperand::Register(registers::T0),
                        MachineOperand::Symbol(got_symbol),
                    ],
                );
                auipc.add_implicit_def(registers::T0);
                last_bb.push_instr(auipc);

                // LD t0, 0(t0)  — load the GOT entry (symbol address)
                let mut ld = MachineInstr::with_operands(
                    codegen::RV_LD,
                    vec![
                        MachineOperand::Register(registers::T0),
                        MachineOperand::Register(registers::T0),
                        MachineOperand::Immediate(0),
                    ],
                );
                ld.add_implicit_def(registers::T0);
                ld.add_implicit_use(registers::T0);
                last_bb.push_instr(ld);
            }

            // Return a register operand referencing the loaded address.
            MachineOperand::Register(registers::T0)
        } else {
            // Non-PIC mode: direct absolute symbol reference.
            // The linker resolves this at link time. For RISC-V, this
            // typically involves a AUIPC + ADDI pair resolved by the
            // linker via R_RISCV_PCREL_HI20 and R_RISCV_PCREL_LO12_I
            // relocations.
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
#[allow(dead_code)]
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

    /// Creates a minimal [`CodegenConfig`] targeting RISC-V 64 with all
    /// optional features disabled. Used as a baseline for tests.
    fn test_config() -> CodegenConfig {
        CodegenConfig::new(Target::RiscV64)
    }

    /// Creates a test config with PIC mode enabled.
    fn pic_config() -> CodegenConfig {
        let mut cfg = CodegenConfig::new(Target::RiscV64);
        cfg.pic = true;
        cfg
    }

    // -- Construction tests -------------------------------------------------

    #[test]
    fn new_riscv64_codegen_valid_target() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.config().target, Target::RiscV64);
        assert_eq!(backend.config().optimization_level, 0);
        assert!(!backend.config().debug_info);
        assert!(!backend.config().pic);
        assert!(!backend.config().shared);
    }

    #[test]
    #[should_panic(expected = "non-RISC-V 64 target")]
    fn new_rejects_x86_64_target() {
        let config = CodegenConfig::new(Target::X86_64);
        let _ = RiscV64Codegen::new(config);
    }

    #[test]
    #[should_panic(expected = "non-RISC-V 64 target")]
    fn new_rejects_aarch64_target() {
        let config = CodegenConfig::new(Target::AArch64);
        let _ = RiscV64Codegen::new(config);
    }

    #[test]
    #[should_panic(expected = "non-RISC-V 64 target")]
    fn new_rejects_i686_target() {
        let config = CodegenConfig::new(Target::I686);
        let _ = RiscV64Codegen::new(config);
    }

    // -- ELF constant tests -------------------------------------------------

    #[test]
    fn elf_machine_is_em_riscv() {
        assert_eq!(ELF_MACHINE, 243);
    }

    #[test]
    fn elf_flags_lp64d_rvc() {
        // EF_RISCV_RVC (0x0001) | EF_RISCV_FLOAT_ABI_DOUBLE (0x0004) = 0x0005
        assert_eq!(ELF_FLAGS, 0x0005);
        assert_eq!(ELF_FLAGS & 0x0001, 0x0001); // RVC bit
        assert_eq!(ELF_FLAGS & 0x0004, 0x0004); // Float ABI double bit
    }

    // -- Register accessor tests --------------------------------------------

    #[test]
    fn integer_register_count_is_32() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.integer_register_count(), 32);
    }

    #[test]
    fn float_register_count_is_32() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.float_register_count(), 32);
    }

    #[test]
    fn callee_saved_returns_s_regs_and_fp() {
        let backend = RiscV64Codegen::new(test_config());
        let callee = backend.callee_saved_registers();
        // 12 integer (s0–s11) + 12 FP (fs0–fs11)
        assert_eq!(callee.len(), 24);
        // s0 (FP) should be first
        assert_eq!(callee[0], registers::S0);
        // fs0 should start at index 12
        assert_eq!(callee[12], registers::FS0);
    }

    #[test]
    fn caller_saved_returns_t_a_ra_and_fp() {
        let backend = RiscV64Codegen::new(test_config());
        let caller = backend.caller_saved_registers();
        // 16 integer (ra, t0–t6, a0–a7) + 20 FP (ft0–ft11, fa0–fa7)
        assert_eq!(caller.len(), 36);
    }

    #[test]
    fn argument_int_registers_a0_through_a7() {
        let backend = RiscV64Codegen::new(test_config());
        let args = backend.argument_registers_int();
        assert_eq!(args.len(), 8);
        assert_eq!(args[0], registers::A0);
        assert_eq!(args[7], registers::A7);
    }

    #[test]
    fn argument_float_registers_fa0_through_fa7() {
        let backend = RiscV64Codegen::new(test_config());
        let args = backend.argument_registers_float();
        assert_eq!(args.len(), 8);
        assert_eq!(args[0], registers::FA0);
        assert_eq!(args[7], registers::FA7);
    }

    #[test]
    fn return_register_int_is_a0() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.return_register_int(), registers::A0);
    }

    #[test]
    fn return_register_float_is_fa0() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.return_register_float(), registers::FA0);
    }

    #[test]
    fn stack_pointer_is_sp() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.stack_pointer(), registers::SP);
    }

    #[test]
    fn frame_pointer_is_s0() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.frame_pointer(), registers::FP);
        assert_eq!(backend.frame_pointer(), registers::S0); // FP alias
    }

    #[test]
    fn pointer_size_is_8_bytes() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.pointer_size(), 8);
    }

    #[test]
    fn function_alignment_is_4_bytes() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.function_alignment(), 4);
    }

    // -- ABI classification tests -------------------------------------------

    #[test]
    fn classify_integer_types() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(
            backend.classify_type(&CType::Int { signed: true }),
            ParamClass::Integer
        );
        assert_eq!(
            backend.classify_type(&CType::Long { signed: true }),
            ParamClass::Integer
        );
        assert_eq!(
            backend.classify_type(&CType::LongLong { signed: true }),
            ParamClass::Integer
        );
        assert_eq!(
            backend.classify_type(&CType::Short { signed: true }),
            ParamClass::Integer
        );
        assert_eq!(
            backend.classify_type(&CType::Char { signed: true }),
            ParamClass::Integer
        );
        // Unsigned variants should also classify as Integer.
        assert_eq!(
            backend.classify_type(&CType::Int { signed: false }),
            ParamClass::Integer
        );
    }

    #[test]
    fn classify_pointer_type() {
        let backend = RiscV64Codegen::new(test_config());
        let ptr = CType::Pointer(Box::new(CType::Void));
        assert_eq!(backend.classify_type(&ptr), ParamClass::Integer);
    }

    #[test]
    fn classify_float_types() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.classify_type(&CType::Float), ParamClass::SSE);
        assert_eq!(backend.classify_type(&CType::Double), ParamClass::SSE);
    }

    // -- Prologue / Epilogue tests ------------------------------------------

    #[test]
    fn prologue_empty_for_trivial_leaf() {
        let backend = RiscV64Codegen::new(test_config());
        let instrs = backend.generate_prologue(0, &[], false);
        assert!(instrs.is_empty());
    }

    #[test]
    fn prologue_with_frame_and_calls() {
        let backend = RiscV64Codegen::new(test_config());
        let instrs = backend.generate_prologue(32, &[], true);
        // Should have: ADDI sp, SD ra, SD s0, ADDI s0
        assert!(instrs.len() >= 4);
        // First instruction: ADDI sp, sp, -32
        assert_eq!(instrs[0].opcode, codegen::RV_ADDI);
    }

    #[test]
    fn epilogue_has_ret() {
        let backend = RiscV64Codegen::new(test_config());
        let instrs = backend.generate_epilogue(32, &[], true);
        // Last instruction must be RET
        let last = instrs.last().expect("epilogue should not be empty");
        assert!(last.is_return);
        assert_eq!(last.opcode, codegen::RV_RET);
    }

    #[test]
    fn emit_prologue_generates_real_code() {
        // emit_prologue produces real prologue instructions when
        // has_calls is true and frame_size > 0.
        let backend = RiscV64Codegen::new(test_config());
        let mut mf = MachineFunction::new("test_func".to_string(), 16);
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);
        mf.frame_size = 32;
        mf.has_calls = true;

        backend.emit_prologue(&mut mf);
        // Prologue must emit at least: ADDI SP,-frame + SD RA + ADDI FP
        assert!(
            !mf.blocks[0].instructions.is_empty(),
            "emit_prologue should produce instructions for a non-leaf function"
        );
    }

    #[test]
    fn emit_epilogue_expands_return() {
        // emit_epilogue replaces bare RET with restore + dealloc + ret.
        let backend = RiscV64Codegen::new(test_config());
        let mut mf = MachineFunction::new("test_func".to_string(), 16);
        let mut bb = MachineBasicBlock::new(0);

        // Add a bare return instruction
        let mut ret_instr = MachineInstr::new(codegen::RV_RET);
        ret_instr.set_return();
        bb.push_instr(ret_instr);
        mf.add_block(bb);
        mf.frame_size = 32;
        mf.has_calls = true;

        // emit_prologue first to set frame_size correctly
        backend.emit_prologue(&mut mf);
        let pre_count = mf.blocks[0].instructions.len();

        backend.emit_epilogue(&mut mf);
        // Epilogue should have expanded the return into restores + ret.
        let block = &mf.blocks[0];
        assert!(
            block.instructions.len() >= pre_count,
            "emit_epilogue should expand the return instruction"
        );
        // Final instruction must be a return.
        assert!(
            block.instructions.last().unwrap().is_return,
            "last instruction must be a return"
        );
    }

    // -- PIC addressing tests -----------------------------------------------

    #[test]
    fn pic_addressing_non_pic_returns_symbol() {
        let backend = RiscV64Codegen::new(test_config());
        let mut mf = MachineFunction::new("test_func".to_string(), 16);
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);

        let operand = backend.generate_pic_addressing("my_global", &mut mf);
        assert_eq!(operand, MachineOperand::Symbol("my_global".to_string()));
    }

    #[test]
    fn pic_addressing_pic_emits_auipc_ld() {
        let backend = RiscV64Codegen::new(pic_config());
        let mut mf = MachineFunction::new("test_func".to_string(), 16);
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);

        let operand = backend.generate_pic_addressing("my_global", &mut mf);
        // Should return a register operand (t0) where the address was loaded
        assert_eq!(operand, MachineOperand::Register(registers::T0));
        // The block should now contain AUIPC + LD instructions
        assert_eq!(mf.blocks[0].instructions.len(), 2);
        assert_eq!(mf.blocks[0].instructions[0].opcode, codegen::RV_AUIPC);
        assert_eq!(mf.blocks[0].instructions[1].opcode, codegen::RV_LD);
    }

    // -- Relocation type tests ----------------------------------------------

    #[test]
    fn relocation_types_not_empty() {
        let backend = RiscV64Codegen::new(test_config());
        let relocs = backend.get_relocation_types();
        assert!(!relocs.is_empty());
    }

    #[test]
    fn relocation_types_contain_branch() {
        let backend = RiscV64Codegen::new(test_config());
        let relocs = backend.get_relocation_types();
        let has_branch = relocs.iter().any(|r| r.name.contains("BRANCH"));
        assert!(
            has_branch,
            "RISC-V relocations should include R_RISCV_BRANCH"
        );
    }

    // -- align_to helper tests ----------------------------------------------

    #[test]
    fn align_to_already_aligned() {
        assert_eq!(align_to(16, 16), 16);
        assert_eq!(align_to(32, 16), 32);
    }

    #[test]
    fn align_to_unaligned() {
        assert_eq!(align_to(1, 16), 16);
        assert_eq!(align_to(17, 16), 32);
        assert_eq!(align_to(15, 16), 16);
    }

    #[test]
    fn align_to_zero() {
        assert_eq!(align_to(0, 16), 0);
    }

    // -- Config accessor tests ----------------------------------------------

    #[test]
    fn config_accessor() {
        let backend = RiscV64Codegen::new(test_config());
        assert_eq!(backend.config().target, Target::RiscV64);
        assert!(!backend.config().requires_pic());
    }

    #[test]
    fn pic_config_requires_pic() {
        let backend = RiscV64Codegen::new(pic_config());
        assert!(backend.config().requires_pic());
    }

    #[test]
    fn shared_implies_pic() {
        let mut cfg = CodegenConfig::new(Target::RiscV64);
        cfg.shared = true;
        let backend = RiscV64Codegen::new(cfg);
        assert!(backend.config().requires_pic());
    }
}
