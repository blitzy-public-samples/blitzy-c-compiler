//! i686 (32-bit x86) backend module for the BCC compiler.
//!
//! This module implements the complete i686 target backend, providing the
//! [`ArchCodegen`] trait implementation that ties together instruction selection,
//! register allocation support, ABI conformance, a built-in assembler, and an
//! integrated ELF linker — all for the 32-bit x86 (i386/i686) architecture.
//!
//! # Architecture Overview
//!
//! | Property                    | Value                                          |
//! |-----------------------------|-------------------------------------------------|
//! | Data model                  | ILP32 (int, long, pointer are all 32-bit)      |
//! | General-purpose registers   | 8 (EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP)    |
//! | Floating-point unit         | x87 FPU stack (ST0–ST7)                        |
//! | Instruction encoding        | Variable-length, 1–15 bytes, **no REX prefix** |
//! | Calling convention          | cdecl / System V i386 ABI                      |
//! | Parameter passing           | ALL parameters on stack (pushed right-to-left)  |
//! | Integer return              | EAX (≤ 32-bit), EDX:EAX (64-bit)              |
//! | FP return                   | x87 ST(0) (float, double, long double)         |
//! | PIC addressing              | GOT-relative via EBX (`__i686.get_pc_thunk.*`) |
//! | ELF machine                 | `EM_386` (3)                                   |
//! | ELF class                   | `ELFCLASS32`                                   |
//! | Classic Linux base address  | `0x0804_8000`                                  |
//! | Function alignment          | 16 bytes                                       |
//!
//! # Key Differences from x86-64
//!
//! - Only 8 GPRs (no R8–R15), so the register allocator has much more
//!   register pressure.  Only EAX, ECX, EDX, EBX have addressable 8-bit
//!   sub-registers (AL, CL, DL, BL).
//! - No register-based argument passing — cdecl pushes everything on the stack.
//! - No RIP-relative addressing — PIC code uses EBX as the GOT base register,
//!   established via a `__i686.get_pc_thunk.bx` call sequence.
//! - No REX prefix — encoding is simpler but register set is constrained.
//! - Floating-point uses the x87 FPU stack model, not SSE registers.
//! - All addresses and relocations are 32-bit (R_386_* relocation types).
//!
//! # Sub-modules
//!
//! - [`registers`]: i686 register definitions — 8 GPRs, sub-register aliases,
//!   x87 FPU stack registers (ST0–ST7), EFLAGS, register classification arrays.
//! - [`codegen`]: i686 instruction selection — translates IR instructions to
//!   i686 machine instructions with 32-bit register constraints and x87 FPU.
//! - [`abi`]: cdecl / System V i386 ABI — stack-based parameter passing,
//!   return value classification, stack layout computation.
//! - [`assembler`]: Built-in i686 assembler — 32-bit instruction encoding,
//!   ModR/M+SIB, REL-format relocations.
//! - [`linker`]: Built-in i686 ELF linker — produces ET_EXEC and ET_DYN
//!   binaries with GOT/PLT for PIC code.
//!
//! # Validation Order
//!
//! Per Section 0.1.2 of the Agent Action Plan, the i686 backend is validated
//! **second** after x86-64 in the backend validation sequence:
//! x86-64 → **i686** → AArch64 → RISC-V 64.

// ---------------------------------------------------------------------------
// Sub-module declarations
// ---------------------------------------------------------------------------

/// i686 register definitions — 8 GPRs (EAX–EDI), 16-bit and 8-bit
/// sub-register aliases, x87 FPU stack registers (ST0–ST7), EFLAGS,
/// register classification arrays, and property query functions.
pub mod registers;

/// Built-in i686 assembler — 32-bit x86 instruction encoding without
/// REX prefixes, using REL-format relocations (inline addends) and
/// ModR/M + SIB encoding for all i686 addressing modes.
pub mod assembler;

/// cdecl / System V i386 ABI implementation — stack-based parameter
/// passing, EAX / EDX:EAX integer returns, x87 ST(0) floating-point
/// returns, sret for large struct returns, and 16-byte call-site
/// stack alignment.
pub mod abi;

/// i686 instruction selection and emission — translates IR instructions
/// to i686 machine instructions, handling 32-bit integer operations,
/// x87 FPU floating-point, 64-bit register pair emulation, cdecl
/// calling convention, and PIC addressing modes.
pub mod codegen;

/// Built-in i686 ELF linker producing ET_EXEC and ET_DYN binaries
/// with full GOT/PLT relocation support for PIC code.
pub mod linker;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

/// Re-export the i686 instruction selector for direct use by consumers.
pub use codegen::I686InstrSel;

/// Re-export all register constants (EAX, EBX, ..., ST0, CALLEE_SAVED, etc.)
/// so that consumers of `crate::backend::i686::*` have direct access.
pub use registers::*;

/// Re-export the i686 ABI handler for external consumers.
pub use abi::I686Abi;

// ---------------------------------------------------------------------------
// Imports from the crate
// ---------------------------------------------------------------------------

use crate::backend::traits::{
    ArchCodegen, CodegenConfig, MachineFunction, MachineInstr, MachineOperand, ParamClass, PhysReg,
    RelocationType,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::function::IrFunction;

use self::assembler::I686Assembler;
use self::codegen::I686Opcode;

// ---------------------------------------------------------------------------
// ELF Machine Constants
// ---------------------------------------------------------------------------

/// ELF `e_machine` value for i386 (`EM_386` = 3).
///
/// Written into the ELF header when producing i686 binaries. This matches
/// the `EM_386` constant defined in the ELF specification and the value
/// returned by [`Target::I686.elf_machine()`].
pub const ELF_MACHINE: u16 = 3;

/// ELF `e_flags` value for i386 (no special flags).
///
/// The i386 ELF ABI defines no required processor-specific flags in the
/// ELF header, so this is always zero.
pub const ELF_FLAGS: u32 = 0;

// ---------------------------------------------------------------------------
// Architecture Constants
// ---------------------------------------------------------------------------

/// Stack probe threshold in bytes.  When a function's frame size exceeds
/// this value, a probe loop is emitted in the prologue to touch each
/// stack page sequentially, preventing silent guard page skip-over on
/// large stack allocations.
const STACK_PROBE_THRESHOLD: u32 = 4096;

/// Required alignment for function entry points in the `.text` section.
/// 16-byte alignment ensures optimal instruction cache line utilization
/// and avoids branch target penalties on i686 microarchitectures.
const I686_FUNCTION_ALIGNMENT: u32 = 16;

// ---------------------------------------------------------------------------
// i686 ELF Relocation Type Table
// ---------------------------------------------------------------------------

/// Complete table of i386 ELF relocation types used by the built-in
/// assembler and linker.
///
/// This table covers the standard i386 ELF ABI relocations from the
/// "System V Application Binary Interface — Intel386 Architecture
/// Processor Supplement" plus TLS relocations for `_Thread_local` support.
///
/// Each entry maps a human-readable name to its numeric `r_type` value
/// used in `Elf32_Rel` / `Elf32_Rela` entries.
const I686_RELOCATION_TYPES: &[RelocationType] = &[
    RelocationType {
        name: "R_386_NONE",
        value: 0,
        is_pc_relative: false,
        size: 0,
    },
    RelocationType {
        name: "R_386_32",
        value: 1,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_PC32",
        value: 2,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_386_GOT32",
        value: 3,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_PLT32",
        value: 4,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_386_COPY",
        value: 5,
        is_pc_relative: false,
        size: 0,
    },
    RelocationType {
        name: "R_386_GLOB_DAT",
        value: 6,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_JMP_SLOT",
        value: 7,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_RELATIVE",
        value: 8,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_GOTOFF",
        value: 9,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_GOTPC",
        value: 10,
        is_pc_relative: true,
        size: 4,
    },
    RelocationType {
        name: "R_386_32PLT",
        value: 11,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_TLS_GD_32",
        value: 24,
        is_pc_relative: false,
        size: 4,
    },
    RelocationType {
        name: "R_386_TLS_LE_32",
        value: 34,
        is_pc_relative: false,
        size: 4,
    },
];

// ---------------------------------------------------------------------------
// I686Codegen — Core Backend Entry Point
// ---------------------------------------------------------------------------

/// Primary i686 code generation struct implementing the [`ArchCodegen`] trait.
///
/// `I686Codegen` is the architecture dispatch target when `--target=i686` is
/// specified.  The code generation driver (`src/backend/generation.rs`)
/// instantiates this struct and calls its trait methods to lower IR functions
/// to i686 machine code.
///
/// # Configuration
///
/// The [`CodegenConfig`] stored in this struct carries all target-specific flags:
///
/// - `target`: Must be [`Target::I686`] (asserted on construction)
/// - `optimization_level`: Optimization level (0 = no optimization)
/// - `debug_info`: Whether to emit DWARF v4 debug sections (`-g`)
/// - `pic`: Position-independent code generation (`-fPIC`)
/// - `shared`: Shared library output (`-shared`)
///
/// # Security Mitigations
///
/// Security mitigations (retpoline, CET/IBT) are x86-64-only per the
/// Agent Action Plan (Section 0.6.2).  The i686 backend does not implement
/// these features.  If `config.retpoline` or `config.cf_protection` are set,
/// they are silently ignored.
///
/// # Usage
///
/// ```ignore
/// use crate::backend::traits::CodegenConfig;
/// use crate::common::target::Target;
///
/// let config = CodegenConfig::new(Target::I686);
/// let backend = I686Codegen::new(config);
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
pub struct I686Codegen {
    /// Target configuration carrying optimization level, debug info,
    /// PIC mode, and other code generation settings.
    config: CodegenConfig,
}

impl I686Codegen {
    /// Creates a new i686 code generator with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.target` is not [`Target::I686`].  This is a
    /// programming error — the code generation driver should dispatch
    /// to the correct backend based on the target architecture.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = CodegenConfig::new(Target::I686);
    /// let backend = I686Codegen::new(config);
    /// assert_eq!(backend.pointer_size(), 4);
    /// ```
    pub fn new(config: CodegenConfig) -> Self {
        assert!(
            config.target == Target::I686,
            "I686Codegen::new() called with non-i686 target: expected Target::I686, got {}",
            config.target
        );
        Self { config }
    }

    /// Returns a reference to the stored [`CodegenConfig`].
    ///
    /// Useful for submodules that need to query configuration flags
    /// (e.g., the assembler checking PIC mode, the ABI module querying
    /// target properties).
    #[inline]
    pub fn config(&self) -> &CodegenConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Prologue / Epilogue Generation
    // -----------------------------------------------------------------------

    /// Generates the standard i686 function prologue as a sequence of
    /// [`MachineInstr`] values.
    ///
    /// The i686 prologue performs these steps in order:
    ///
    /// 1. `PUSH EBP` — save the caller's frame pointer.
    /// 2. `MOV EBP, ESP` — establish this function's frame pointer.
    /// 3. **(Conditional)** If `frame_size > STACK_PROBE_THRESHOLD (4096)`,
    ///    emit a stack probe loop that touches each page before the actual
    ///    stack pointer adjustment.  Otherwise, if `frame_size > 0`, emit
    ///    `SUB ESP, aligned_size`.
    /// 4. `PUSH` each callee-saved register used by the function
    ///    (EBX, ESI, EDI — note: EBP is already saved in step 1).
    ///
    /// # Arguments
    ///
    /// * `frame_size` — total stack frame size in bytes (excluding callee saves).
    /// * `callee_saved` — callee-saved registers used by this function.
    ///
    /// # Returns
    ///
    /// A `Vec<MachineInstr>` to be prepended to the entry basic block.
    fn generate_prologue(&self, frame_size: u32, callee_saved: &[PhysReg]) -> Vec<MachineInstr> {
        let mut instrs = Vec::with_capacity(4 + callee_saved.len());

        // Step 1: PUSH EBP — save old frame pointer
        let mut push_ebp = MachineInstr::with_operands(
            I686Opcode::Push as u32,
            vec![MachineOperand::Register(registers::EBP)],
        );
        push_ebp.add_implicit_use(registers::ESP);
        push_ebp.add_implicit_def(registers::ESP);
        instrs.push(push_ebp);

        // Step 2: MOV EBP, ESP — establish new frame pointer
        let mov_ebp_esp = MachineInstr::with_operands(
            I686Opcode::Mov as u32,
            vec![
                MachineOperand::Register(registers::EBP),
                MachineOperand::Register(registers::ESP),
            ],
        );
        instrs.push(mov_ebp_esp);

        // Step 3: Stack allocation — subtract frame_size from ESP
        if frame_size > 0 {
            let aligned_size = align_to(frame_size, self.config.target.stack_alignment());

            if aligned_size > STACK_PROBE_THRESHOLD {
                // For large frames (> 4096 bytes), emit a stack probe loop
                // that touches each page to ensure guard pages are triggered.
                //
                // The probe loop pattern on i686:
                //   mov eax, aligned_size
                // .Lprobe_loop:
                //   sub esp, 4096
                //   test [esp], eax       ; touch the page
                //   sub eax, 4096
                //   jnz .Lprobe_loop
                //   sub esp, (aligned_size % 4096)  ; remaining bytes
                //
                // We emit this as a pseudo-instruction that the assembler
                // expands, similar to the x86-64 PSEUDO_STACK_PROBE pattern.
                //
                // For now, emit the explicit loop instructions:
                let pages = aligned_size / STACK_PROBE_THRESHOLD;
                let remainder = aligned_size % STACK_PROBE_THRESHOLD;

                for _ in 0..pages {
                    // SUB ESP, 4096
                    let mut sub_instr = MachineInstr::with_operands(
                        I686Opcode::Sub as u32,
                        vec![
                            MachineOperand::Register(registers::ESP),
                            MachineOperand::Immediate(STACK_PROBE_THRESHOLD as i64),
                        ],
                    );
                    sub_instr.add_implicit_def(registers::ESP);
                    instrs.push(sub_instr);

                    // TEST DWORD PTR [ESP], ESP  — touch the page
                    let test_instr = MachineInstr::with_operands(
                        I686Opcode::Test as u32,
                        vec![
                            MachineOperand::Memory {
                                base: registers::ESP,
                                offset: 0,
                                index: None,
                                scale: 1,
                            },
                            MachineOperand::Register(registers::ESP),
                        ],
                    );
                    instrs.push(test_instr);
                }

                // Subtract remaining bytes if any
                if remainder > 0 {
                    let mut sub_rem = MachineInstr::with_operands(
                        I686Opcode::Sub as u32,
                        vec![
                            MachineOperand::Register(registers::ESP),
                            MachineOperand::Immediate(remainder as i64),
                        ],
                    );
                    sub_rem.add_implicit_def(registers::ESP);
                    instrs.push(sub_rem);
                }
            } else {
                // Simple SUB ESP, aligned_size for small-to-medium frames
                let mut sub_esp = MachineInstr::with_operands(
                    I686Opcode::Sub as u32,
                    vec![
                        MachineOperand::Register(registers::ESP),
                        MachineOperand::Immediate(aligned_size as i64),
                    ],
                );
                sub_esp.add_implicit_def(registers::ESP);
                instrs.push(sub_esp);
            }
        }

        // Step 4: Save callee-saved registers.
        // On i686 cdecl, callee-saved are EBX, ESI, EDI (EBP is saved
        // in step 1 as part of the frame pointer setup).
        for &reg in callee_saved {
            // Skip EBP — it was saved in step 1.
            if reg == registers::EBP {
                continue;
            }
            let mut push = MachineInstr::with_operands(
                I686Opcode::Push as u32,
                vec![MachineOperand::Register(reg)],
            );
            push.add_implicit_use(registers::ESP);
            push.add_implicit_def(registers::ESP);
            instrs.push(push);
        }

        instrs
    }

    /// Generates the standard i686 function epilogue as a sequence of
    /// [`MachineInstr`] values.
    ///
    /// The epilogue is the mirror image of the prologue:
    ///
    /// 1. `POP` callee-saved registers in reverse order.
    /// 2. `MOV ESP, EBP` — discard the local frame.
    /// 3. `POP EBP` — restore the caller's frame pointer.
    /// 4. `RET` — return to the caller.
    ///
    /// # Arguments
    ///
    /// * `callee_saved` — callee-saved registers that were saved in the prologue.
    ///
    /// # Returns
    ///
    /// A `Vec<MachineInstr>` to replace each `RET` instruction in the function.
    fn generate_epilogue(&self, callee_saved: &[PhysReg]) -> Vec<MachineInstr> {
        let mut instrs = Vec::with_capacity(4 + callee_saved.len());

        // Step 1: Restore callee-saved registers in reverse order.
        // Skip EBP — it is restored by the POP EBP in step 3.
        let saved_without_ebp: Vec<PhysReg> = callee_saved
            .iter()
            .copied()
            .filter(|&r| r != registers::EBP)
            .collect();

        for &reg in saved_without_ebp.iter().rev() {
            let mut pop = MachineInstr::with_operands(
                I686Opcode::Pop as u32,
                vec![MachineOperand::Register(reg)],
            );
            pop.add_implicit_use(registers::ESP);
            pop.add_implicit_def(registers::ESP);
            instrs.push(pop);
        }

        // Step 2: MOV ESP, EBP — discard the local frame
        let mov_esp_ebp = MachineInstr::with_operands(
            I686Opcode::Mov as u32,
            vec![
                MachineOperand::Register(registers::ESP),
                MachineOperand::Register(registers::EBP),
            ],
        );
        instrs.push(mov_esp_ebp);

        // Step 3: POP EBP — restore the caller's frame pointer
        let mut pop_ebp = MachineInstr::with_operands(
            I686Opcode::Pop as u32,
            vec![MachineOperand::Register(registers::EBP)],
        );
        pop_ebp.add_implicit_use(registers::ESP);
        pop_ebp.add_implicit_def(registers::ESP);
        instrs.push(pop_ebp);

        // Step 4: RET — return to the caller
        let mut ret = MachineInstr::new(I686Opcode::Ret as u32);
        ret.set_terminator();
        ret.is_return = true;
        instrs.push(ret);

        instrs
    }
}

// ---------------------------------------------------------------------------
// ArchCodegen Trait Implementation
// ---------------------------------------------------------------------------

impl ArchCodegen for I686Codegen {
    /// Lowers an IR function to i686 machine instructions.
    ///
    /// Delegates to [`I686InstrSel`] which performs pattern-matching
    /// instruction selection, translating each IR instruction into one
    /// or more i686 machine instructions.
    ///
    /// The process:
    /// 1. Validates the IR function is non-empty.
    /// 2. Creates an [`I686InstrSel`] with the current configuration.
    /// 3. Delegates to `I686InstrSel::select_function()` for the full
    ///    instruction selection pass.
    /// 4. Returns the resulting `MachineFunction`.
    ///
    /// # Arguments
    ///
    /// * `func` — the IR function to lower. Must have at least one basic block.
    ///
    /// # Returns
    ///
    /// A [`MachineFunction`] containing i686 machine instructions.
    fn lower_function(&self, func: &IrFunction) -> MachineFunction {
        // Validate: the function must have at least one basic block.
        // An empty function would indicate a bug in the IR lowering phase.
        if func.basic_blocks.is_empty() {
            // Return a minimal machine function for empty IR functions.
            // This can happen for forward-declared functions that have no body.
            let stack_align = self.config.target.stack_alignment();
            let mut mf = MachineFunction::new(func.name.clone(), stack_align);
            mf.frame_size = 0;
            return mf;
        }

        // Create a diagnostic engine for this function's code generation.
        let diag = DiagnosticEngine::new();

        // Delegate to the i686 instruction selector.
        let mut isel = I686InstrSel::new(&self.config, &diag);
        isel.select_function(func)
    }

    /// Encodes a machine function into raw i686 binary bytes.
    ///
    /// Delegates to the built-in i686 assembler which handles:
    /// - ModR/M + SIB encoding for all addressing modes
    /// - 32-bit instruction encoding without REX prefixes
    /// - Relocation recording for unresolved symbol references
    /// - Intra-function branch fixups
    ///
    /// # Arguments
    ///
    /// * `mf` — the machine function to assemble.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the raw machine code bytes.
    fn emit_assembly(&self, mf: &MachineFunction) -> Vec<u8> {
        let mut asm = I686Assembler::with_pic(self.config.pic);
        let result = asm.assemble_function(mf);
        result.code
    }

    /// Returns the complete table of i386 ELF relocation types.
    ///
    /// These relocations are used by both the assembler (to record
    /// unresolved references) and the linker (to apply fixups).
    /// The table covers all standard `R_386_*` relocations from the
    /// i386 ELF ABI supplement.
    fn get_relocation_types(&self) -> &[RelocationType] {
        I686_RELOCATION_TYPES
    }

    /// Returns 8 — i686 has 8 general-purpose 32-bit registers:
    /// EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP.
    ///
    /// Of these, only 6 are typically available for the register
    /// allocator (ESP and EBP are reserved for stack management).
    #[inline]
    fn integer_register_count(&self) -> usize {
        8
    }

    /// Returns 8 — i686 has 8 x87 FPU stack registers: ST(0)–ST(7).
    ///
    /// The x87 FPU operates as a stack-based floating-point unit,
    /// not a flat register file like SSE.  ST(0) is the top of the
    /// stack and is the implicit destination/source for most x87
    /// instructions.
    #[inline]
    fn float_register_count(&self) -> usize {
        8
    }

    /// Returns the i686 callee-saved register set: EBX, ESI, EDI, EBP.
    ///
    /// Under the cdecl / System V i386 ABI, these registers must be
    /// preserved across function calls.  If a function uses any of these,
    /// their values must be saved in the prologue and restored in the
    /// epilogue.
    #[inline]
    fn callee_saved_registers(&self) -> &[PhysReg] {
        &registers::CALLEE_SAVED
    }

    /// Returns the i686 caller-saved (volatile) register set: EAX, ECX, EDX.
    ///
    /// Under cdecl, these registers may be clobbered by any function call.
    /// The caller is responsible for saving their values if they are live
    /// across a call site.
    #[inline]
    fn caller_saved_registers(&self) -> &[PhysReg] {
        &registers::CALLER_SAVED
    }

    /// Returns an empty slice — cdecl passes ALL arguments on the stack.
    ///
    /// Unlike x86-64 System V (which passes the first 6 integer args in
    /// registers), i686 cdecl has no register-based integer argument passing.
    /// All integer arguments are pushed right-to-left onto the stack.
    #[inline]
    fn argument_registers_int(&self) -> &[PhysReg] {
        &registers::INTEGER_ARG_REGS
    }

    /// Returns an empty slice — cdecl passes ALL FP arguments on the stack.
    ///
    /// Unlike x86-64 System V (which passes the first 8 FP args in
    /// XMM0–XMM7), i686 cdecl has no register-based float argument passing.
    /// All floating-point arguments are pushed onto the stack (promoted to
    /// double for the x87 FPU).
    #[inline]
    fn argument_registers_float(&self) -> &[PhysReg] {
        &registers::FLOAT_ARG_REGS
    }

    /// Returns EAX — the integer return register under cdecl.
    ///
    /// Scalar return values ≤ 32 bits (int, long, pointer, enum, bool,
    /// char, short) are returned in EAX.  64-bit values (`long long`) are
    /// returned in the EDX:EAX register pair.
    #[inline]
    fn return_register_int(&self) -> PhysReg {
        registers::EAX
    }

    /// Returns ST0 — the x87 FPU floating-point return register.
    ///
    /// All floating-point return values (float, double, long double) are
    /// returned in ST(0), the top of the x87 floating-point register stack.
    /// This is fundamentally different from x86-64, which returns float/double
    /// in XMM0.
    #[inline]
    fn return_register_float(&self) -> PhysReg {
        registers::ST0
    }

    /// Returns ESP — the stack pointer on i686.
    ///
    /// ESP always points to the top of the stack (lowest used address).
    /// The stack grows downward on i686, like all x86 architectures.
    #[inline]
    fn stack_pointer(&self) -> PhysReg {
        registers::ESP
    }

    /// Returns EBP — the frame pointer on i686.
    ///
    /// EBP is callee-saved and used to establish the stack frame in the
    /// standard prologue sequence (`PUSH EBP; MOV EBP, ESP`).  With the
    /// frame pointer, local variables are accessed at `[EBP - N]` and
    /// function arguments at `[EBP + 8 + offset]`.
    #[inline]
    fn frame_pointer(&self) -> PhysReg {
        registers::EBP
    }

    /// Returns 4 — i686 pointers are 32-bit (4 bytes) in the ILP32 model.
    ///
    /// Derived from [`Target::I686.pointer_width()`] to maintain
    /// consistency with the canonical target definition.
    #[inline]
    fn pointer_size(&self) -> u32 {
        Target::I686.pointer_width()
    }

    /// Returns 16 — function entry points are aligned to 16-byte
    /// boundaries for optimal instruction fetch on i686.
    #[inline]
    fn function_alignment(&self) -> u32 {
        I686_FUNCTION_ALIGNMENT
    }

    /// Emits the i686 function prologue into the machine function.
    ///
    /// Inserts prologue instructions at the beginning of the entry block:
    ///
    /// 1. `PUSH EBP`
    /// 2. `MOV EBP, ESP`
    /// 3. `SUB ESP, framesize` (with stack probing for large frames)
    /// 4. `PUSH` callee-saved registers
    fn emit_prologue(&self, mf: &mut MachineFunction) {
        let callee_saved = mf.used_callee_saved.clone();
        let prologue_instrs = self.generate_prologue(mf.frame_size, &callee_saved);

        // Insert prologue instructions at the beginning of the entry block.
        if !mf.blocks.is_empty() {
            let entry = &mut mf.blocks[0];
            let mut new_instrs = prologue_instrs;
            new_instrs.append(&mut entry.instructions);
            entry.instructions = new_instrs;
        }
    }

    /// Emits the i686 function epilogue into the machine function.
    ///
    /// Replaces each return instruction in every basic block with the
    /// full epilogue sequence:
    ///
    /// 1. `POP` callee-saved registers (reverse order)
    /// 2. `MOV ESP, EBP` — discard the local frame
    /// 3. `POP EBP` — restore caller's frame pointer
    /// 4. `RET`
    fn emit_epilogue(&self, mf: &mut MachineFunction) {
        let callee_saved = mf.used_callee_saved.clone();
        let epilogue_instrs = self.generate_epilogue(&callee_saved);

        // Replace every return instruction with the complete epilogue sequence.
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

    /// Classifies a C type into a cdecl / System V i386 ABI parameter class.
    ///
    /// Under cdecl, the classification is much simpler than x86-64:
    ///
    /// - All integer types and pointers → [`ParamClass::Integer`] (conceptually),
    ///   but are actually passed on the stack, so they receive
    ///   [`ParamClass::Memory`] for the register allocator's benefit.
    /// - Floating-point types → [`ParamClass::X87`] (passed on stack, returned
    ///   in ST(0)).
    /// - Aggregates (structs, unions, arrays) → [`ParamClass::Memory`] (always
    ///   on the stack).
    /// - Void → [`ParamClass::NoClass`].
    ///
    /// Note: The distinction between Integer and Memory matters primarily for
    /// return value handling (EAX vs. sret pointer).  For argument passing,
    /// everything is Memory-class on i686 cdecl.
    fn classify_type(&self, ty: &CType) -> ParamClass {
        let abi = I686Abi::new();

        // Fast path: void type has no class.
        if matches!(ty.canonical(), CType::Void) {
            return ParamClass::NoClass;
        }

        // Use the ABI's return classification to determine the primary class.
        // This gives us the most accurate classification for the return
        // value handling, which is what ArchCodegen::classify_type is
        // primarily used for.
        let ret_class = abi.classify_return(ty, &Target::I686);

        match ret_class {
            abi::ReturnClassification::InRegister { .. } => ParamClass::Integer,
            abi::ReturnClassification::RegisterPair { .. } => ParamClass::Integer,
            abi::ReturnClassification::X87 { .. } => ParamClass::X87,
            abi::ReturnClassification::Indirect => ParamClass::Memory,
            abi::ReturnClassification::Void => ParamClass::NoClass,
        }
    }

    /// Generates position-independent addressing for a symbol on i686.
    ///
    /// i686 PIC addressing is fundamentally different from x86-64:
    /// there is no RIP-relative addressing.  Instead, PIC code uses
    /// the GOT base register (EBX) established via a thunk call:
    ///
    /// ```asm
    /// ; GOT base establishment (done once per function):
    /// call __i686.get_pc_thunk.bx     ; EBX = PC after call
    /// add  ebx, _GLOBAL_OFFSET_TABLE_ ; EBX = GOT base
    ///
    /// ; Symbol access through GOT:
    /// mov  eax, [ebx + symbol@GOT]    ; load GOT entry
    ///
    /// ; Local symbol access via GOTOFF:
    /// lea  eax, [ebx + symbol@GOTOFF] ; address of local symbol
    /// ```
    ///
    /// In non-PIC mode, the symbol is referenced directly with an
    /// absolute address.
    ///
    /// # Arguments
    ///
    /// * `symbol` — the symbol name to load.
    /// * `mf` — the machine function to emit addressing instructions into.
    ///
    /// # Returns
    ///
    /// A [`MachineOperand`] referencing the loaded symbol address.
    fn generate_pic_addressing(&self, symbol: &str, mf: &mut MachineFunction) -> MachineOperand {
        if self.config.requires_pic() {
            // PIC mode on i686: access the symbol through the GOT using
            // EBX as the GOT base register.
            //
            // Emit:  MOV EAX, [EBX + symbol@GOT]
            //
            // This generates a memory operand with EBX (GOT base) as the
            // base register and a symbol reference decorated with @GOT.
            // The assembler records an R_386_GOT32 relocation for the
            // symbol, and the linker resolves it to the GOT entry offset.
            let got_symbol = format!("{}@GOT", symbol);

            if !mf.blocks.is_empty() {
                let last_bb = mf
                    .blocks
                    .last_mut()
                    .expect("generate_pic_addressing: function must have at least one basic block");

                // Emit MOV EAX, [EBX + symbol@GOT]
                // EBX is the GOT base register, established by the
                // __i686.get_pc_thunk.bx call in the function prologue.
                let mut mov_instr = MachineInstr::with_operands(
                    I686Opcode::Mov as u32,
                    vec![
                        MachineOperand::Register(registers::EAX),
                        MachineOperand::Memory {
                            base: registers::EBX,
                            offset: 0,
                            index: None,
                            scale: 1,
                        },
                        MachineOperand::Symbol(got_symbol),
                    ],
                );
                mov_instr.add_implicit_def(registers::EAX);
                mov_instr.add_implicit_use(registers::EBX);
                last_bb.push_instr(mov_instr);
            }

            // Return a register operand — after the MOV above, EAX holds
            // the symbol's actual address (loaded from the GOT entry).
            MachineOperand::Register(registers::EAX)
        } else {
            // Non-PIC mode: direct absolute symbol reference.
            // The linker resolves this at link time using R_386_32.
            MachineOperand::Symbol(symbol.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Aligns `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two.  If `value` is already aligned,
/// it is returned unchanged.  Returns 0 when `value` is 0.
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

    /// Creates a minimal [`CodegenConfig`] targeting i686 with all
    /// optional features disabled.  Used as a baseline for tests.
    fn test_config() -> CodegenConfig {
        CodegenConfig::new(Target::I686)
    }

    /// Creates a test config with PIC mode enabled.
    fn pic_config() -> CodegenConfig {
        let mut cfg = CodegenConfig::new(Target::I686);
        cfg.pic = true;
        cfg
    }

    // -- Construction tests -------------------------------------------------

    #[test]
    fn new_i686_codegen_valid_target() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.config().target, Target::I686);
        assert_eq!(backend.config().optimization_level, 0);
        assert!(!backend.config().debug_info);
        assert!(!backend.config().pic);
        assert!(!backend.config().shared);
    }

    #[test]
    #[should_panic(expected = "non-i686 target")]
    fn new_rejects_x86_64_target() {
        let config = CodegenConfig::new(Target::X86_64);
        let _ = I686Codegen::new(config);
    }

    #[test]
    #[should_panic(expected = "non-i686 target")]
    fn new_rejects_aarch64_target() {
        let config = CodegenConfig::new(Target::AArch64);
        let _ = I686Codegen::new(config);
    }

    #[test]
    #[should_panic(expected = "non-i686 target")]
    fn new_rejects_riscv64_target() {
        let config = CodegenConfig::new(Target::RiscV64);
        let _ = I686Codegen::new(config);
    }

    // -- Register count tests -----------------------------------------------

    #[test]
    fn integer_register_count_is_8() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.integer_register_count(), 8);
    }

    #[test]
    fn float_register_count_is_8() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.float_register_count(), 8);
    }

    // -- Register set tests -------------------------------------------------

    #[test]
    fn callee_saved_register_set() {
        let backend = I686Codegen::new(test_config());
        let regs = backend.callee_saved_registers();
        // cdecl i386: EBX, ESI, EDI, EBP (4 registers)
        assert_eq!(regs.len(), 4);
        assert!(regs.contains(&registers::EBX));
        assert!(regs.contains(&registers::ESI));
        assert!(regs.contains(&registers::EDI));
        assert!(regs.contains(&registers::EBP));
    }

    #[test]
    fn caller_saved_register_set() {
        let backend = I686Codegen::new(test_config());
        let regs = backend.caller_saved_registers();
        // cdecl i386: EAX, ECX, EDX (3 registers)
        assert_eq!(regs.len(), 3);
        assert!(regs.contains(&registers::EAX));
        assert!(regs.contains(&registers::ECX));
        assert!(regs.contains(&registers::EDX));
    }

    #[test]
    fn argument_registers_int_empty() {
        let backend = I686Codegen::new(test_config());
        let regs = backend.argument_registers_int();
        // cdecl: no register-based integer argument passing
        assert!(regs.is_empty());
    }

    #[test]
    fn argument_registers_float_empty() {
        let backend = I686Codegen::new(test_config());
        let regs = backend.argument_registers_float();
        // cdecl: no register-based float argument passing
        assert!(regs.is_empty());
    }

    // -- Special register tests ---------------------------------------------

    #[test]
    fn return_register_int_is_eax() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.return_register_int(), registers::EAX);
    }

    #[test]
    fn return_register_float_is_st0() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.return_register_float(), registers::ST0);
    }

    #[test]
    fn stack_pointer_is_esp() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.stack_pointer(), registers::ESP);
    }

    #[test]
    fn frame_pointer_is_ebp() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.frame_pointer(), registers::EBP);
    }

    // -- ABI constant tests -------------------------------------------------

    #[test]
    fn pointer_size_is_4() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.pointer_size(), 4);
    }

    #[test]
    fn function_alignment_is_16() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.function_alignment(), 16);
    }

    // -- ELF constant tests -------------------------------------------------

    #[test]
    fn elf_machine_is_em_386() {
        assert_eq!(ELF_MACHINE, 3);
    }

    #[test]
    fn elf_flags_is_zero() {
        assert_eq!(ELF_FLAGS, 0);
    }

    // -- Relocation type tests ----------------------------------------------

    #[test]
    fn relocation_types_non_empty() {
        let backend = I686Codegen::new(test_config());
        let relocs = backend.get_relocation_types();
        assert!(!relocs.is_empty());
        assert!(
            relocs.len() >= 12,
            "expected at least 12 relocation types, got {}",
            relocs.len()
        );
    }

    #[test]
    fn relocation_types_contain_key_entries() {
        let backend = I686Codegen::new(test_config());
        let relocs = backend.get_relocation_types();

        // Verify critical relocation types are present
        let has = |name: &str| relocs.iter().any(|r| r.name == name);
        assert!(has("R_386_NONE"), "missing R_386_NONE");
        assert!(has("R_386_32"), "missing R_386_32");
        assert!(has("R_386_PC32"), "missing R_386_PC32");
        assert!(has("R_386_GOT32"), "missing R_386_GOT32");
        assert!(has("R_386_PLT32"), "missing R_386_PLT32");
        assert!(has("R_386_COPY"), "missing R_386_COPY");
        assert!(has("R_386_GLOB_DAT"), "missing R_386_GLOB_DAT");
        assert!(has("R_386_JMP_SLOT"), "missing R_386_JMP_SLOT");
        assert!(has("R_386_RELATIVE"), "missing R_386_RELATIVE");
        assert!(has("R_386_GOTOFF"), "missing R_386_GOTOFF");
        assert!(has("R_386_GOTPC"), "missing R_386_GOTPC");
    }

    #[test]
    fn relocation_type_values_match_elf_spec() {
        let backend = I686Codegen::new(test_config());
        let relocs = backend.get_relocation_types();

        let find = |name: &str| relocs.iter().find(|r| r.name == name).map(|r| r.value);
        assert_eq!(find("R_386_NONE"), Some(0));
        assert_eq!(find("R_386_32"), Some(1));
        assert_eq!(find("R_386_PC32"), Some(2));
        assert_eq!(find("R_386_GOT32"), Some(3));
        assert_eq!(find("R_386_PLT32"), Some(4));
        assert_eq!(find("R_386_COPY"), Some(5));
        assert_eq!(find("R_386_GLOB_DAT"), Some(6));
        assert_eq!(find("R_386_JMP_SLOT"), Some(7));
        assert_eq!(find("R_386_RELATIVE"), Some(8));
        assert_eq!(find("R_386_GOTOFF"), Some(9));
        assert_eq!(find("R_386_GOTPC"), Some(10));
        assert_eq!(find("R_386_32PLT"), Some(11));
    }

    #[test]
    fn relocation_type_pc_relative_flags() {
        let backend = I686Codegen::new(test_config());
        let relocs = backend.get_relocation_types();

        let is_pc_rel = |name: &str| {
            relocs
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.is_pc_relative)
        };
        assert_eq!(is_pc_rel("R_386_32"), Some(false));
        assert_eq!(is_pc_rel("R_386_PC32"), Some(true));
        assert_eq!(is_pc_rel("R_386_GOT32"), Some(false));
        assert_eq!(is_pc_rel("R_386_PLT32"), Some(true));
        assert_eq!(is_pc_rel("R_386_GOTPC"), Some(true));
    }

    // -- classify_type tests ------------------------------------------------

    #[test]
    fn classify_void_is_no_class() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.classify_type(&CType::Void), ParamClass::NoClass);
    }

    #[test]
    fn classify_int_is_integer() {
        let backend = I686Codegen::new(test_config());
        let int_ty = CType::Int { signed: true };
        assert_eq!(backend.classify_type(&int_ty), ParamClass::Integer);
    }

    #[test]
    fn classify_pointer_is_integer() {
        let backend = I686Codegen::new(test_config());
        let ptr_ty = CType::Pointer(Box::new(CType::Void));
        assert_eq!(backend.classify_type(&ptr_ty), ParamClass::Integer);
    }

    #[test]
    fn classify_float_is_x87() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.classify_type(&CType::Float), ParamClass::X87);
    }

    #[test]
    fn classify_double_is_x87() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.classify_type(&CType::Double), ParamClass::X87);
    }

    #[test]
    fn classify_long_double_is_x87() {
        let backend = I686Codegen::new(test_config());
        assert_eq!(backend.classify_type(&CType::LongDouble), ParamClass::X87);
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
        assert_eq!(align_to(15, 16), 16);
        assert_eq!(align_to(17, 16), 32);
        assert_eq!(align_to(4, 4), 4);
        assert_eq!(align_to(3, 4), 4);
        assert_eq!(align_to(5, 4), 8);
    }

    // -- Prologue / epilogue tests ------------------------------------------

    #[test]
    fn prologue_zero_frame_no_callee_saved() {
        let backend = I686Codegen::new(test_config());
        let instrs = backend.generate_prologue(0, &[]);
        // Should have: PUSH EBP, MOV EBP ESP (2 instructions)
        assert_eq!(instrs.len(), 2);
        // First: PUSH EBP
        assert_eq!(instrs[0].opcode, I686Opcode::Push as u32);
        // Second: MOV EBP, ESP
        assert_eq!(instrs[1].opcode, I686Opcode::Mov as u32);
    }

    #[test]
    fn prologue_with_frame_size() {
        let backend = I686Codegen::new(test_config());
        let instrs = backend.generate_prologue(32, &[]);
        // PUSH EBP + MOV EBP ESP + SUB ESP 32 = 3 instructions
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[2].opcode, I686Opcode::Sub as u32);
    }

    #[test]
    fn prologue_with_callee_saved() {
        let backend = I686Codegen::new(test_config());
        let callee = &[registers::EBX, registers::ESI];
        let instrs = backend.generate_prologue(16, callee);
        // PUSH EBP + MOV EBP ESP + SUB ESP 16 + PUSH EBX + PUSH ESI = 5
        assert_eq!(instrs.len(), 5);
    }

    #[test]
    fn prologue_skips_ebp_in_callee_saved() {
        let backend = I686Codegen::new(test_config());
        // If EBP is in callee_saved, it should be skipped (already saved)
        let callee = &[registers::EBP, registers::EBX];
        let instrs = backend.generate_prologue(0, callee);
        // PUSH EBP + MOV EBP ESP + PUSH EBX = 3 (not 4)
        assert_eq!(instrs.len(), 3);
    }

    #[test]
    fn epilogue_basic() {
        let backend = I686Codegen::new(test_config());
        let instrs = backend.generate_epilogue(&[]);
        // MOV ESP EBP + POP EBP + RET = 3 instructions
        assert_eq!(instrs.len(), 3);
        let last = instrs.last().unwrap();
        assert!(last.is_return);
        assert!(last.is_terminator);
    }

    #[test]
    fn epilogue_with_callee_saved() {
        let backend = I686Codegen::new(test_config());
        let callee = &[registers::EBX, registers::ESI];
        let instrs = backend.generate_epilogue(callee);
        // POP ESI + POP EBX + MOV ESP EBP + POP EBP + RET = 5
        assert_eq!(instrs.len(), 5);
        // First two should be POPs in reverse order
        assert_eq!(instrs[0].opcode, I686Opcode::Pop as u32);
        assert_eq!(instrs[1].opcode, I686Opcode::Pop as u32);
    }

    // -- PIC addressing tests -----------------------------------------------

    #[test]
    fn non_pic_generates_symbol_operand() {
        let backend = I686Codegen::new(test_config());
        let mut mf = MachineFunction::new("test".to_string(), 16);
        let bb = MachineBasicBlock::new(0);
        mf.blocks.push(bb);

        let operand = backend.generate_pic_addressing("my_var", &mut mf);
        match operand {
            MachineOperand::Symbol(name) => assert_eq!(name, "my_var"),
            _ => panic!("expected Symbol operand in non-PIC mode"),
        }
    }

    #[test]
    fn pic_generates_got_relative_load() {
        let backend = I686Codegen::new(pic_config());
        let mut mf = MachineFunction::new("test".to_string(), 16);
        let bb = MachineBasicBlock::new(0);
        mf.blocks.push(bb);

        let operand = backend.generate_pic_addressing("my_var", &mut mf);

        // In PIC mode, the result should be a Register operand (EAX)
        // because we load the symbol address into EAX from the GOT.
        match operand {
            MachineOperand::Register(reg) => assert_eq!(reg, registers::EAX),
            _ => panic!("expected Register(EAX) operand in PIC mode"),
        }

        // Verify a MOV instruction was emitted in the last block
        let last_bb = mf.blocks.last().unwrap();
        assert!(!last_bb.instructions.is_empty());
        let mov = &last_bb.instructions[0];
        assert_eq!(mov.opcode, I686Opcode::Mov as u32);
    }

    // -- emit_prologue / emit_epilogue integration --------------------------

    #[test]
    fn emit_prologue_prepends_to_entry_block() {
        let backend = I686Codegen::new(test_config());
        let mut mf = MachineFunction::new("test_fn".to_string(), 16);
        mf.frame_size = 32;
        mf.used_callee_saved = vec![];

        let mut bb = MachineBasicBlock::new(0);
        // Add a dummy instruction
        bb.push_instr(MachineInstr::new(I686Opcode::Nop as u32));
        mf.blocks.push(bb);

        backend.emit_prologue(&mut mf);

        // Prologue: PUSH EBP + MOV EBP ESP + SUB ESP 32 = 3 + 1 NOP = 4
        assert_eq!(mf.blocks[0].instructions.len(), 4);
        // First instruction should be PUSH EBP
        assert_eq!(mf.blocks[0].instructions[0].opcode, I686Opcode::Push as u32);
    }

    #[test]
    fn emit_epilogue_replaces_return_instructions() {
        let backend = I686Codegen::new(test_config());
        let mut mf = MachineFunction::new("test_fn".to_string(), 16);
        mf.used_callee_saved = vec![];

        let mut bb = MachineBasicBlock::new(0);
        let mut ret = MachineInstr::new(I686Opcode::Ret as u32);
        ret.is_return = true;
        bb.push_instr(ret);
        mf.blocks.push(bb);

        backend.emit_epilogue(&mut mf);

        // Epilogue: MOV ESP EBP + POP EBP + RET = 3 instructions
        assert_eq!(mf.blocks[0].instructions.len(), 3);
        // Last instruction should be RET (from the epilogue)
        let last = mf.blocks[0].instructions.last().unwrap();
        assert!(last.is_return);
    }
}
