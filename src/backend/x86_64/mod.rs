//! # x86-64 Backend — System V AMD64 ABI
//!
//! This is the root module for the x86-64 architecture backend.  It serves
//! two critical purposes:
//!
//! 1. **Module declarations**: Declares all submodules (`registers`, `codegen`,
//!    `abi`, `security`, `assembler`, `linker`).
//! 2. **[`ArchCodegen`] trait implementation**: Provides the [`X86_64Backend`]
//!    struct that implements the architecture abstraction trait defined in
//!    [`crate::backend::traits`].
//!
//! ## Architecture Overview
//!
//! - **ABI**: System V AMD64 — 6 integer argument registers (RDI, RSI, RDX,
//!   RCX, R8, R9), 8 SSE argument registers (XMM0–XMM7), 128-byte red zone,
//!   16-byte stack alignment at call sites.
//! - **Register File**: 16 GPRs (RAX–R15), 16 SSE registers (XMM0–XMM15).
//!   SSE/SSE2 is used for floating-point (**not** x87 FPU on x86-64).
//! - **Instruction Encoding**: Variable-length (1–15 bytes), REX prefix
//!   required for registers R8–R15 and XMM8–XMM15.
//! - **Security Mitigations**: Retpoline (`-mretpoline`), CET/IBT
//!   (`-fcf-protection`), and stack guard-page probing are x86-64 **only**.
//!
//! ## Validation Order
//!
//! The x86-64 backend is the **PRIMARY validation target** — it is validated
//! first in the backend order: x86-64 → i686 → AArch64 → RISC-V 64.
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references.  No external crates.

// ===========================================================================
// Submodule declarations — all 6 submodules for the x86-64 backend
// ===========================================================================

pub mod abi;
pub mod assembler;
pub mod codegen;
pub mod linker;
pub mod registers;
pub mod security;

// ===========================================================================
// Re-exports — public API surface for downstream consumers
// ===========================================================================

/// Re-export all 16 GPR register constants (RAX–R15) and 16 SSE register
/// constants (XMM0–XMM15) for convenient access by other modules.
pub use self::registers::{
    R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, XMM0, XMM1,
    XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9,
};

/// Re-export the x86-64 ABI implementation for external usage.
pub use self::abi::X86_64Abi;

/// Re-export the x86-64 instruction selection engine.
pub use self::codegen::X86_64CodeGen;

/// Re-export the security mitigation configuration.
pub use self::security::SecurityConfig;

// ===========================================================================
// Internal imports
// ===========================================================================

use crate::backend::traits::{
    ArchCodegen, ArgLocation as TraitsArgLocation, AssembledFunction, FunctionRelocation,
    MachineFunction, MachineInstruction, MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::types::IrType;

use self::abi::RetLocation;
use self::codegen::X86Opcode;

// ===========================================================================
// Static relocation type descriptors for x86-64 ELF
// ===========================================================================

/// Supported x86-64 ELF relocation types.
///
/// These descriptors are returned by [`X86_64Backend::relocation_types`] and
/// consumed by the common linker infrastructure to dispatch relocation
/// application to the architecture-specific handler.
///
/// | Name                        | ID | Size | PC-Relative | Description                    |
/// |-----------------------------|----|------|-------------|--------------------------------|
/// | `R_X86_64_NONE`             |  0 |   0  | No          | No relocation                  |
/// | `R_X86_64_64`               |  1 |   8  | No          | Direct 64-bit absolute         |
/// | `R_X86_64_PC32`             |  2 |   4  | Yes         | 32-bit PC-relative             |
/// | `R_X86_64_GOT32`            |  3 |   4  | No          | 32-bit GOT entry offset        |
/// | `R_X86_64_PLT32`            |  4 |   4  | Yes         | 32-bit PLT-relative            |
/// | `R_X86_64_GOTPCREL`         |  9 |   4  | Yes         | 32-bit GOT PC-relative         |
/// | `R_X86_64_32`               | 10 |   4  | No          | Direct 32-bit zero-extended    |
/// | `R_X86_64_32S`              | 11 |   4  | No          | Direct 32-bit sign-extended    |
/// | `R_X86_64_GOTPCRELX`        | 41 |   4  | Yes         | Relaxable GOTPCREL             |
/// | `R_X86_64_REX_GOTPCRELX`    | 42 |   4  | Yes         | Relaxable GOTPCREL with REX    |
static X86_64_RELOCATION_TYPES: [RelocationTypeInfo; 10] = [
    RelocationTypeInfo::new("R_X86_64_NONE", 0, 0, false),
    RelocationTypeInfo::new("R_X86_64_64", 1, 8, false),
    RelocationTypeInfo::new("R_X86_64_PC32", 2, 4, true),
    RelocationTypeInfo::new("R_X86_64_GOT32", 3, 4, false),
    RelocationTypeInfo::new("R_X86_64_PLT32", 4, 4, true),
    RelocationTypeInfo::new("R_X86_64_GOTPCREL", 9, 4, true),
    RelocationTypeInfo::new("R_X86_64_32", 10, 4, false),
    RelocationTypeInfo::new("R_X86_64_32S", 11, 4, false),
    RelocationTypeInfo::new("R_X86_64_GOTPCRELX", 41, 4, true),
    RelocationTypeInfo::new("R_X86_64_REX_GOTPCRELX", 42, 4, true),
];

// ===========================================================================
// X86_64Backend — main backend struct
// ===========================================================================

/// The x86-64 architecture backend implementing the [`ArchCodegen`] trait.
///
/// Coordinates instruction selection, register allocation, ABI conformance,
/// security mitigation injection, assembly encoding, and ELF linking for
/// the x86-64 architecture with System V AMD64 calling conventions.
///
/// # Construction
///
/// ```ignore
/// let backend = X86_64Backend::new(
///     Target::X86_64,
///     SecurityConfig::default(),
///     false, // pic
///     false, // shared
///     false, // debug_info
/// );
/// ```
///
/// # Security Mitigations
///
/// The x86-64 backend is the **only** architecture that supports:
/// - **Retpoline** (`-mretpoline`): Indirect call/jump thunk transformation
/// - **CET/IBT** (`-fcf-protection`): `endbr64` insertion at branch targets
/// - **Stack Probe** (always checked): Guard-page probe for frames > 4 KiB
pub struct X86_64Backend {
    /// Target information (pointer width, endianness, predefined macros).
    target: Target,

    /// Security mitigation configuration — retpoline, CET, stack probe settings.
    security_config: SecurityConfig,

    /// Whether PIC code generation is enabled (`-fPIC`).
    pic_enabled: bool,

    /// Whether generating a shared library (`-shared`).
    shared: bool,

    /// Whether debug info generation is enabled (`-g`).
    debug_info: bool,

    /// System V AMD64 ABI implementation for parameter/return classification.
    abi: X86_64Abi,
}

// ===========================================================================
// X86_64Backend — constructor and helper methods
// ===========================================================================

impl X86_64Backend {
    /// Create a new x86-64 backend instance.
    ///
    /// # Arguments
    ///
    /// * `target` — Target configuration (must be `Target::X86_64`).
    /// * `security_config` — Security mitigation settings from CLI flags.
    /// * `pic_enabled` — Whether `-fPIC` was specified.
    /// * `shared` — Whether `-shared` was specified.
    /// * `debug_info` — Whether `-g` was specified.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `target` is `Target::X86_64`.
    pub fn new(
        target: Target,
        security_config: SecurityConfig,
        pic_enabled: bool,
        shared: bool,
        debug_info: bool,
    ) -> Self {
        debug_assert!(
            target == Target::X86_64,
            "X86_64Backend requires Target::X86_64, got {:?}",
            target
        );
        let abi = X86_64Abi::new(target);
        Self {
            target,
            security_config,
            pic_enabled,
            shared,
            debug_info,
            abi,
        }
    }

    /// Get the security configuration.
    #[inline]
    pub fn security_config(&self) -> &SecurityConfig {
        &self.security_config
    }

    /// Whether PIC code generation is enabled.
    #[inline]
    pub fn is_pic(&self) -> bool {
        self.pic_enabled
    }

    /// Whether generating a shared library.
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    /// Whether debug info is enabled.
    #[inline]
    pub fn has_debug_info(&self) -> bool {
        self.debug_info
    }
}

// ===========================================================================
// Helper: convert ABI-specific ArgLocation to traits::ArgLocation
// ===========================================================================

/// Convert an x86-64 ABI argument location to the trait-level `ArgLocation`.
///
/// The x86-64 ABI module has an `Indirect` variant (hidden pointer in a
/// register) that the trait-level enum does not.  We map it to
/// `ArgLocation::Register` since the pointer is physically passed in a
/// register — the caller allocates storage and passes the address.
fn convert_abi_arg_location(loc: self::abi::ArgLocation) -> TraitsArgLocation {
    match loc {
        self::abi::ArgLocation::Register(r) => TraitsArgLocation::Register(r),
        self::abi::ArgLocation::RegisterPair(r1, r2) => TraitsArgLocation::RegisterPair(r1, r2),
        self::abi::ArgLocation::Stack(off) => TraitsArgLocation::Stack(off),
        self::abi::ArgLocation::Indirect(r) => TraitsArgLocation::Register(r),
    }
}

/// Convert an x86-64 ABI return location to the trait-level `ArgLocation`.
///
/// Mapping:
/// - `RetLocation::Register(r)` → `ArgLocation::Register(r)`
/// - `RetLocation::RegisterPair(r1, r2)` → `ArgLocation::RegisterPair(r1, r2)`
/// - `RetLocation::Indirect` → `ArgLocation::Stack(0)` (hidden pointer return)
/// - `RetLocation::Void` → `ArgLocation::Register(RAX)` (unused placeholder)
fn convert_ret_location(loc: RetLocation) -> TraitsArgLocation {
    match loc {
        RetLocation::Register(r) => TraitsArgLocation::Register(r),
        RetLocation::RegisterPair(r1, r2) => TraitsArgLocation::RegisterPair(r1, r2),
        RetLocation::Indirect => TraitsArgLocation::Stack(0),
        RetLocation::Void => TraitsArgLocation::Register(registers::RAX),
    }
}

// ===========================================================================
// ArchCodegen trait implementation
// ===========================================================================

impl ArchCodegen for X86_64Backend {
    /// Lower an IR function to x86-64 machine instructions.
    ///
    /// This is the main entry point for code generation.  It:
    /// 1. Creates an [`X86_64CodeGen`] bound to the target.
    /// 2. Runs x86-64 instruction selection (which computes the frame
    ///    layout, maps parameters to ABI locations, and emits machine
    ///    instructions for every IR instruction).
    /// 3. Returns the [`MachineFunction`] ready for register allocation,
    ///    prologue/epilogue insertion, and assembly encoding.
    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
        global_var_refs: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) -> Result<MachineFunction, String> {
        let mut codegen = X86_64CodeGen::new(self.target);
        codegen.set_pic(self.pic_enabled);
        codegen.set_func_ref_names(func_ref_map.clone());
        codegen.set_global_var_refs(global_var_refs.clone());
        codegen.lower(func, diag, globals)
    }

    /// Emit machine code bytes from a fully register-allocated
    /// [`MachineFunction`].
    ///
    /// Uses the built-in x86-64 assembler (`assembler/`) to encode each
    /// [`MachineInstruction`] into bytes, producing relocatable object code.
    /// Security-related bytes (retpoline thunks, CET note section) are also
    /// handled by the assembler when the corresponding flags are active.
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<AssembledFunction, String> {
        // Apply security mitigations that transform the MachineFunction:
        // - Retpoline: replace indirect call/jmp instructions with thunk calls
        //   so that the assembler can emit CALL rel32 to __x86_indirect_thunk_<reg>.
        let mut mf_secured = mf.clone();
        security::transform_indirect_calls(&mut mf_secured, self.security_config.retpoline);
        let result = assembler::assemble(&mf_secured, &self.security_config, self.pic_enabled);
        let relocations = result
            .relocations
            .iter()
            .map(|r| FunctionRelocation {
                offset: r.offset,
                symbol: r.symbol.clone(),
                rel_type_id: r.rel_type.as_u32(),
                addend: r.addend,
                section: r.section.clone(),
            })
            .collect();
        Ok(AssembledFunction {
            bytes: result.text,
            relocations,
        })
    }

    /// Return the target architecture.
    #[inline]
    fn target(&self) -> Target {
        self.target
    }

    /// Return register allocation info for x86-64.
    ///
    /// Provides:
    /// - 14 allocatable GPRs (all except RSP, RBP)
    /// - 16 allocatable SSE registers
    /// - Callee-saved set: RBX, R12, R13, R14, R15
    /// - Caller-saved set: RAX, RCX, RDX, RSI, RDI, R8–R11
    /// - Argument GPRs: RDI, RSI, RDX, RCX, R8, R9
    /// - Argument SSE: XMM0–XMM7
    /// - Return GPRs: RAX, RDX
    /// - Return SSE: XMM0, XMM1
    fn register_info(&self) -> RegisterInfo {
        RegisterInfo {
            allocatable_gpr: registers::ALLOCATABLE_GPRS.to_vec(),
            allocatable_fpr: registers::ALLOCATABLE_SSE.to_vec(),
            callee_saved: registers::CALLEE_SAVED_GPRS.to_vec(),
            caller_saved: registers::CALLER_SAVED_GPRS.to_vec(),
            reserved: registers::RESERVED_REGS.to_vec(),
            argument_gpr: registers::ARG_GPRS.to_vec(),
            argument_fpr: registers::ARG_SSE.to_vec(),
            return_gpr: registers::RET_GPRS.to_vec(),
            return_fpr: registers::RET_SSE.to_vec(),
        }
    }

    /// Return supported relocation types for x86-64 ELF.
    ///
    /// Returns a static slice of 10 relocation type descriptors covering
    /// the x86-64 ELF ABI supplement relocations used by BCC's assembler
    /// and linker.
    #[inline]
    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        &X86_64_RELOCATION_TYPES
    }

    /// Emit function prologue for x86-64.
    ///
    /// Standard prologue sequence:
    /// 1. `endbr64` (if CET/IBT enabled via `-fcf-protection`)
    /// 2. `push rbp`
    /// 3. `mov rbp, rsp`
    /// 4. Stack probe loop (if frame > 4096 bytes and stack_probe enabled)
    /// 5. `sub rsp, <frame_size>` (16-byte aligned) — only if no probe loop
    ///    already set RSP
    /// 6. Save callee-saved registers used in the function
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();

        // CET: endbr64 at function entry
        if self.security_config.cf_protection {
            prologue.push(security::emit_endbr64());
        }

        // push rbp
        let mut push_rbp = MachineInstruction::new(X86Opcode::Push.as_u32());
        push_rbp.operands.push(MachineOperand::Register(RBP));
        prologue.push(push_rbp);

        // mov rbp, rsp
        let mut mov_rbp_rsp = MachineInstruction::new(X86Opcode::Mov.as_u32());
        mov_rbp_rsp.operands.push(MachineOperand::Register(RSP));
        mov_rbp_rsp.result = Some(MachineOperand::Register(RBP));
        prologue.push(mov_rbp_rsp);

        // Stack frame allocation.
        //
        // The SUB amount must be chosen so that after ALL callee-saved
        // register pushes, RSP is 16-byte aligned (System V AMD64 ABI
        // requirement for CALL instructions).
        //
        // After `push rbp` and `mov rbp, rsp`, RSP is 16-byte aligned.
        // Each subsequent `push` of a callee-saved register decrements
        // RSP by 8.  If the number of pushes is odd, RSP is off by 8.
        // We include the callee push bytes in the alignment calculation
        // and subtract them to get the correct SUB amount.
        let callee_push_bytes = mf.callee_saved_regs.len() * 8;
        let total = mf.frame_size + callee_push_bytes;
        let aligned_total = (total + 15) & !15;
        let aligned_size = aligned_total - callee_push_bytes;

        if aligned_size > 0 {
            if self.security_config.stack_probe && mf.frame_size > security::PAGE_SIZE {
                // Large frame: emit probe loop using pre-encoded raw bytes.
                // We use generate_stack_probe() which produces correctly-encoded
                // machine code with proper TEST [rsp],rsp memory operand and
                // correct JA rel8 back-edge offset within the probe loop.
                // This avoids the encoder limitations with memory operand
                // encoding and BlockLabel resolution for inline probe loops.
                let probe_bytes = security::generate_stack_probe(mf.frame_size);
                let mut probe_inst = MachineInstruction::new(X86Opcode::Nop.as_u32());
                probe_inst.encoded_bytes = probe_bytes;
                prologue.push(probe_inst);
            } else {
                // Small frame or stack_probe disabled: simple SUB RSP, N.
                let mut sub_rsp = MachineInstruction::new(X86Opcode::Sub.as_u32());
                sub_rsp.operands.push(MachineOperand::Register(RSP));
                sub_rsp
                    .operands
                    .push(MachineOperand::Immediate(aligned_size as i64));
                sub_rsp.result = Some(MachineOperand::Register(RSP));
                prologue.push(sub_rsp);
            }
        }

        // Save callee-saved registers that the function uses
        for &reg in &mf.callee_saved_regs {
            let mut push_inst = MachineInstruction::new(X86Opcode::Push.as_u32());
            push_inst.operands.push(MachineOperand::Register(reg));
            prologue.push(push_inst);
        }

        // For variadic functions: save the 6 integer parameter registers
        // (RDI, RSI, RDX, RCX, R8, R9) to the GPR save area.
        if let Some(va_offset) = mf.va_save_area_offset {
            let save_regs: [u16; 6] = [
                registers::RDI,
                registers::RSI,
                registers::RDX,
                registers::RCX,
                registers::R8,
                registers::R9,
            ];
            for (i, &reg) in save_regs.iter().enumerate() {
                let disp = va_offset as i64 + (i as i64) * 8;
                let mut save = MachineInstruction::new(X86Opcode::Mov.as_u32());
                save.operands.push(MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: disp,
                });
                save.operands.push(MachineOperand::Register(reg));
                prologue.push(save);
            }
        }

        // For variadic functions: save XMM0–XMM7 (low 64-bit each) to
        // the FP portion of the contiguous register save area at 16-byte
        // intervals (ABI standard layout).
        if let Some(va_base) = mf.va_save_area_offset {
            let xmm_regs: [u16; 8] = [
                registers::XMM0,
                registers::XMM1,
                registers::XMM2,
                registers::XMM3,
                registers::XMM4,
                registers::XMM5,
                registers::XMM6,
                registers::XMM7,
            ];
            for (i, &reg) in xmm_regs.iter().enumerate() {
                // FP slots start at base+48, each 16 bytes apart
                let disp = va_base as i64 + 48 + (i as i64) * 16;
                let mut save = MachineInstruction::new(X86Opcode::Movsd.as_u32());
                save.operands.push(MachineOperand::Memory {
                    base: Some(RBP),
                    index: None,
                    scale: 1,
                    displacement: disp,
                });
                save.operands.push(MachineOperand::Register(reg));
                prologue.push(save);
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
            // [+0] gp_offset (32-bit) = named_gpr_count * 8
            let gp_offset_val = (mf.named_gpr_count as i64) * 8;
            let mut store_gp_off = MachineInstruction::new(X86Opcode::Mov.as_u32());
            store_gp_off.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: ctrl_offset as i64,
            });
            store_gp_off
                .operands
                .push(MachineOperand::Immediate(gp_offset_val));
            store_gp_off.operand_size = 4;
            prologue.push(store_gp_off);

            // [+4] fp_offset (32-bit) = 48 + named_fp_count * 16
            let fp_offset_val = 48 + (mf.named_fp_count as i64) * 16;
            let mut store_fp_off = MachineInstruction::new(X86Opcode::Mov.as_u32());
            store_fp_off.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: ctrl_offset as i64 + 4,
            });
            store_fp_off
                .operands
                .push(MachineOperand::Immediate(fp_offset_val));
            store_fp_off.operand_size = 4;
            prologue.push(store_fp_off);

            // [+8] overflow_arg_area = RBP + 16 + fixed_stack_bytes
            //       (first variadic stack argument, past any fixed params
            //       that spilled to the stack)
            let gp_stack_fixed = mf.named_gpr_count.saturating_sub(6);
            let fp_stack_fixed = mf.named_fp_count.saturating_sub(8);
            let overflow_disp = 16 + ((gp_stack_fixed + fp_stack_fixed) * 8) as i64;
            let mut lea_ov = MachineInstruction::new(X86Opcode::Lea.as_u32());
            lea_ov.result = Some(MachineOperand::Register(registers::RAX));
            lea_ov.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: overflow_disp,
            });
            prologue.push(lea_ov);
            let mut store_ov = MachineInstruction::new(X86Opcode::Mov.as_u32());
            store_ov.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: ctrl_offset as i64 + 8,
            });
            store_ov
                .operands
                .push(MachineOperand::Register(registers::RAX));
            prologue.push(store_ov);

            // [+16] reg_save_area = RBP + va_save_area_offset
            let mut lea_rs = MachineInstruction::new(X86Opcode::Lea.as_u32());
            lea_rs.result = Some(MachineOperand::Register(registers::RAX));
            lea_rs.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: reg_save_base as i64,
            });
            prologue.push(lea_rs);
            let mut store_rs = MachineInstruction::new(X86Opcode::Mov.as_u32());
            store_rs.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: ctrl_offset as i64 + 16,
            });
            store_rs
                .operands
                .push(MachineOperand::Register(registers::RAX));
            prologue.push(store_rs);
        }

        prologue
    }

    /// Emit function epilogue for x86-64.
    ///
    /// Standard epilogue sequence:
    /// 1. Restore callee-saved registers (reverse order of prologue saves)
    /// 2. `mov rsp, rbp` (restore stack pointer to frame base)
    /// 3. `pop rbp` (restore caller's frame pointer)
    /// 4. `ret` (return to caller)
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();

        // Before restoring callee-saved registers, reset RSP to the position
        // where those registers were pushed during the prologue.
        //
        // The prologue layout is:
        //   push rbp           → RBP = RSP after this + mov rbp,rsp
        //   sub rsp, aligned_size
        //   push <callee-saved regs>
        //
        // So callee-saved regs sit at [RBP - aligned_size - 8], [RBP - aligned_size - 16], etc.
        // RSP after all callee-saved pushes = RBP - aligned_total, where:
        //   aligned_total = (frame_size + callee_push_bytes + 15) & !15
        //
        // During the function body, RSP may drift from this position due to
        // PUSH instructions for call arguments.  The epilogue POP instructions
        // read from the current RSP, so if RSP drifted, they restore wrong values.
        // We must reset RSP before the pops.
        if !mf.callee_saved_regs.is_empty() {
            let callee_push_bytes = mf.callee_saved_regs.len() * 8;
            let total = mf.frame_size + callee_push_bytes;
            let aligned_total = (total + 15) & !15;
            // LEA RSP, [RBP - aligned_total]
            let mut lea_rsp = MachineInstruction::new(X86Opcode::Lea.as_u32());
            lea_rsp.operands.push(MachineOperand::Memory {
                base: Some(RBP),
                index: None,
                scale: 1,
                displacement: -(aligned_total as i64),
            });
            lea_rsp.result = Some(MachineOperand::Register(RSP));
            epilogue.push(lea_rsp);
        }

        // Restore callee-saved registers in reverse order
        for &reg in mf.callee_saved_regs.iter().rev() {
            let mut pop_inst = MachineInstruction::new(X86Opcode::Pop.as_u32());
            pop_inst.operands.push(MachineOperand::Register(reg));
            pop_inst.result = Some(MachineOperand::Register(reg));
            epilogue.push(pop_inst);
        }

        // mov rsp, rbp — restore stack pointer
        let mut mov_rsp_rbp = MachineInstruction::new(X86Opcode::Mov.as_u32());
        mov_rsp_rbp.operands.push(MachineOperand::Register(RBP));
        mov_rsp_rbp.result = Some(MachineOperand::Register(RSP));
        epilogue.push(mov_rsp_rbp);

        // pop rbp — restore caller's frame pointer
        let mut pop_rbp = MachineInstruction::new(X86Opcode::Pop.as_u32());
        pop_rbp.operands.push(MachineOperand::Register(RBP));
        pop_rbp.result = Some(MachineOperand::Register(RBP));
        epilogue.push(pop_rbp);

        // ret — return to caller
        let mut ret_inst = MachineInstruction::new(X86Opcode::Ret.as_u32());
        ret_inst.is_terminator = true;
        epilogue.push(ret_inst);

        epilogue
    }

    /// Return the frame pointer register (RBP for x86-64).
    #[inline]
    fn frame_pointer_reg(&self) -> u16 {
        registers::RBP
    }

    /// Return the stack pointer register (RSP for x86-64).
    #[inline]
    fn stack_pointer_reg(&self) -> u16 {
        registers::RSP
    }

    /// Return the return address register.
    ///
    /// On x86-64, there is no dedicated link register — the return address
    /// is pushed onto the stack by the `CALL` instruction and lives at
    /// `[RBP+8]`.  Returns `None` to indicate stack-based return address.
    #[inline]
    fn return_address_reg(&self) -> Option<u16> {
        // x86-64 has no link register; the return address is on the stack.
        None
    }

    /// Classify where a function argument of the given IR type should be
    /// placed according to the System V AMD64 ABI.
    ///
    /// Delegates to the x86-64 ABI module for eightbyte classification,
    /// then converts the result to the trait-level [`ArgLocation`](TraitsArgLocation).
    fn classify_argument(&self, ty: &IrType) -> TraitsArgLocation {
        let abi_loc = self.abi.classify_argument(ty);
        convert_abi_arg_location(abi_loc)
    }

    /// Classify where a function return value of the given IR type should
    /// be placed according to the System V AMD64 ABI.
    ///
    /// Small scalars return in RAX (integer) or XMM0 (float).  Structs
    /// ≤16 bytes may return in register pairs.  Larger types are returned
    /// via a hidden pointer argument (indirect return).
    fn classify_return(&self, ty: &IrType) -> TraitsArgLocation {
        let ret_loc = self.abi.classify_return(ty);
        convert_ret_location(ret_loc)
    }

    /// Format a machine instruction as valid AT&T-syntax x86-64 assembly.
    ///
    /// Overrides the default `op{n}` format to produce mnemonics recognized
    /// by standard assemblers (e.g., `movq`, `addq`, `imulq`, etc.).
    fn format_instruction(&self, inst: &MachineInstruction) -> String {
        format_x86_instruction(inst)
    }
}

// ===========================================================================
// AT&T-syntax x86-64 assembly formatting for -S output
// ===========================================================================

/// Map a physical register index to its AT&T-syntax 64-bit name.
fn reg_name_64(reg: u16) -> &'static str {
    match reg {
        0 => "%rax",
        1 => "%rcx",
        2 => "%rdx",
        3 => "%rbx",
        4 => "%rsp",
        5 => "%rbp",
        6 => "%rsi",
        7 => "%rdi",
        8 => "%r8",
        9 => "%r9",
        10 => "%r10",
        11 => "%r11",
        12 => "%r12",
        13 => "%r13",
        14 => "%r14",
        15 => "%r15",
        // XMM registers
        16..=31 => {
            const XMM: [&str; 16] = [
                "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8",
                "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15",
            ];
            XMM[(reg - 16) as usize]
        }
        _ => "%???",
    }
}

/// Format a `MachineOperand` in AT&T syntax.
fn format_operand_att(op: &MachineOperand) -> String {
    match op {
        MachineOperand::Register(r) => reg_name_64(*r).to_string(),
        MachineOperand::VirtualRegister(v) => format!("%vr{}", v),
        MachineOperand::Immediate(imm) => format!("${}", imm),
        MachineOperand::Memory {
            base,
            index,
            scale,
            displacement,
        } => {
            let mut s = String::new();
            if *displacement != 0 {
                s.push_str(&format!("{}", displacement));
            }
            s.push('(');
            if let Some(b) = base {
                s.push_str(reg_name_64(*b));
            }
            if let Some(idx) = index {
                s.push(',');
                s.push_str(reg_name_64(*idx));
                if *scale > 1 {
                    s.push_str(&format!(",{}", scale));
                }
            }
            s.push(')');
            s
        }
        MachineOperand::FrameSlot(offset) => {
            format!("{}(%rbp)", offset)
        }
        MachineOperand::GlobalSymbol(name) => name.clone(),
        MachineOperand::BlockLabel(id) => format!(".LBB{}", id),
    }
}

/// Map an X86Opcode to its AT&T-syntax mnemonic string.
fn x86_opcode_mnemonic(opcode: u32) -> &'static str {
    use codegen::X86Opcode;
    match X86Opcode::from_u32(opcode) {
        Some(X86Opcode::Mov) | Some(X86Opcode::LoadInd) | Some(X86Opcode::StoreInd) => "movq",
        Some(X86Opcode::LoadInd32) | Some(X86Opcode::StoreInd32) => "movl",
        Some(X86Opcode::LoadInd16) | Some(X86Opcode::StoreInd16) => "movw",
        Some(X86Opcode::LoadInd8) | Some(X86Opcode::StoreInd8) => "movb",
        Some(X86Opcode::MovZX) => "movzbl",
        Some(X86Opcode::MovSX) => "movsbl",
        Some(X86Opcode::Lea) => "leaq",
        Some(X86Opcode::Push) => "pushq",
        Some(X86Opcode::Pop) => "popq",
        Some(X86Opcode::Xchg) => "xchgq",
        Some(X86Opcode::Add) => "addq",
        Some(X86Opcode::Sub) => "subq",
        Some(X86Opcode::Imul) => "imulq",
        Some(X86Opcode::Idiv) => "idivq",
        Some(X86Opcode::Div) => "divq",
        Some(X86Opcode::Neg) => "negq",
        Some(X86Opcode::Inc) => "incq",
        Some(X86Opcode::Dec) => "decq",
        Some(X86Opcode::And) => "andq",
        Some(X86Opcode::Or) => "orq",
        Some(X86Opcode::Xor) => "xorq",
        Some(X86Opcode::Not) => "notq",
        Some(X86Opcode::Shl) => "shlq",
        Some(X86Opcode::Shr) => "shrq",
        Some(X86Opcode::Sar) => "sarq",
        Some(X86Opcode::Rol) => "rolq",
        Some(X86Opcode::Ror) => "rorq",
        Some(X86Opcode::Cmp) => "cmpq",
        Some(X86Opcode::Test) => "testq",
        Some(X86Opcode::Cmovo) => "cmovo",
        Some(X86Opcode::Cmovno) => "cmovno",
        Some(X86Opcode::Cmovb) => "cmovb",
        Some(X86Opcode::Cmovae) => "cmovae",
        Some(X86Opcode::Cmove) => "cmove",
        Some(X86Opcode::Cmovne) => "cmovne",
        Some(X86Opcode::Cmovbe) => "cmovbe",
        Some(X86Opcode::Cmova) => "cmova",
        Some(X86Opcode::Cmovs) => "cmovs",
        Some(X86Opcode::Cmovns) => "cmovns",
        Some(X86Opcode::Cmovp) => "cmovp",
        Some(X86Opcode::Cmovnp) => "cmovnp",
        Some(X86Opcode::Cmovl) => "cmovl",
        Some(X86Opcode::Cmovge) => "cmovge",
        Some(X86Opcode::Cmovle) => "cmovle",
        Some(X86Opcode::Cmovg) => "cmovg",
        Some(X86Opcode::Seto) => "seto",
        Some(X86Opcode::Setno) => "setno",
        Some(X86Opcode::Setb) => "setb",
        Some(X86Opcode::Setae) => "setae",
        Some(X86Opcode::Sete) => "sete",
        Some(X86Opcode::Setne) => "setne",
        Some(X86Opcode::Setbe) => "setbe",
        Some(X86Opcode::Seta) => "seta",
        Some(X86Opcode::Sets) => "sets",
        Some(X86Opcode::Setns) => "setns",
        Some(X86Opcode::Setp) => "setp",
        Some(X86Opcode::Setnp) => "setnp",
        Some(X86Opcode::Setl) => "setl",
        Some(X86Opcode::Setge) => "setge",
        Some(X86Opcode::Setle) => "setle",
        Some(X86Opcode::Setg) => "setg",
        Some(X86Opcode::Jmp) => "jmp",
        Some(X86Opcode::Jo) => "jo",
        Some(X86Opcode::Jno) => "jno",
        Some(X86Opcode::Jb) => "jb",
        Some(X86Opcode::Jae) => "jae",
        Some(X86Opcode::Je) => "je",
        Some(X86Opcode::Jne) => "jne",
        Some(X86Opcode::Jbe) => "jbe",
        Some(X86Opcode::Ja) => "ja",
        Some(X86Opcode::Js) => "js",
        Some(X86Opcode::Jns) => "jns",
        Some(X86Opcode::Jp) => "jp",
        Some(X86Opcode::Jnp) => "jnp",
        Some(X86Opcode::Jl) => "jl",
        Some(X86Opcode::Jge) => "jge",
        Some(X86Opcode::Jle) => "jle",
        Some(X86Opcode::Jg) => "jg",
        Some(X86Opcode::Call) => "callq",
        Some(X86Opcode::Ret) => "retq",
        Some(X86Opcode::Nop) => "nop",
        Some(X86Opcode::Movsd) => "movsd",
        Some(X86Opcode::Movss) => "movss",
        Some(X86Opcode::Addsd) => "addsd",
        Some(X86Opcode::Addss) => "addss",
        Some(X86Opcode::Subsd) => "subsd",
        Some(X86Opcode::Subss) => "subss",
        Some(X86Opcode::Mulsd) => "mulsd",
        Some(X86Opcode::Mulss) => "mulss",
        Some(X86Opcode::Divsd) => "divsd",
        Some(X86Opcode::Divss) => "divss",
        Some(X86Opcode::Ucomisd) => "ucomisd",
        Some(X86Opcode::Ucomiss) => "ucomiss",
        Some(X86Opcode::Cvtsi2sd) => "cvtsi2sdq",
        Some(X86Opcode::Cvtsi2ss) => "cvtsi2ssq",
        Some(X86Opcode::Cvtsd2si) => "cvtsd2siq",
        Some(X86Opcode::Cvtss2si) => "cvtss2siq",
        Some(X86Opcode::Cvtsd2ss) => "cvtsd2ss",
        Some(X86Opcode::Cvtss2sd) => "cvtss2sd",
        Some(X86Opcode::Enter) => "enter",
        Some(X86Opcode::Leave) => "leaveq",
        Some(X86Opcode::Cdq) => "cdq",
        Some(X86Opcode::Cqo) => "cqo",
        Some(X86Opcode::Endbr64) => "endbr64",
        Some(X86Opcode::Pause) => "pause",
        Some(X86Opcode::Lfence) => "lfence",
        Some(X86Opcode::Bsr) => "bsrq",
        Some(X86Opcode::Bsf) => "bsfq",
        Some(X86Opcode::Popcnt) => "popcntq",
        Some(X86Opcode::Bswap) => "bswapq",
        Some(X86Opcode::Ud2) => "ud2",
        Some(X86Opcode::RepMovsq) => "rep movsq",
        Some(X86Opcode::InlineAsm) => "# inline asm",
        Some(X86Opcode::InternalLabelDef) => "# label",
        Some(X86Opcode::FldMem64) => "fldl",
        Some(X86Opcode::FldMem80) => "fldt",
        Some(X86Opcode::FstpMem80) => "fstpt",
        Some(X86Opcode::FstpMem64) => "fstpl",
        None => "ud2",
    }
}

/// Format a complete machine instruction in AT&T syntax for -S output.
///
/// AT&T syntax uses `mnemonic src, dst` order. `MachineInstruction` stores
/// `result` as the destination and `operands` as sources. We combine them
/// in the correct AT&T order: sources first, destination last.
/// Substitute `%0`, `%1`, … in an inline assembly template with the
/// corresponding operand's representation.  `%%` → `%`.
///
/// For inline asm, `"i"` (immediate) operands are emitted as bare numbers
/// (not `$42`), and register operands use AT&T syntax (`%rax`).
fn substitute_asm_operands(template: &str, operands: &[MachineOperand]) -> String {
    let mut result = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i] == b'%' && i + 1 < len {
            if bytes[i + 1] == b'%' {
                // Escaped percent — emit a single %.
                result.push('%');
                i += 2;
            } else if bytes[i + 1].is_ascii_digit() {
                // Operand reference: %0, %1, %12, etc.
                let start = i + 1;
                let mut end = start;
                while end < len && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                let idx: usize = template[start..end].parse().unwrap_or(0);
                if idx < operands.len() {
                    result.push_str(&format_asm_operand(&operands[idx]));
                } else {
                    // Out-of-range operand — emit literal placeholder.
                    result.push_str(&template[i..end]);
                }
                i = end;
            } else {
                result.push('%');
                i += 1;
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

/// Format a machine operand for inline assembly substitution.
/// Immediates are bare numbers (no `$` prefix), registers use AT&T names.
fn format_asm_operand(op: &MachineOperand) -> String {
    match op {
        MachineOperand::Immediate(imm) => format!("{}", imm),
        _ => format_operand_att(op),
    }
}

fn format_x86_instruction(inst: &MachineInstruction) -> String {
    let mnemonic = x86_opcode_mnemonic(inst.opcode);

    // Inline assembly: emit template with operand substitution.
    // The template is stored in `asm_template` and may contain %0, %1, etc.
    // GCC asm numbering: %0..%(n-1) = output registers, %n.. = input operands.
    // inst.result = output 0 register.
    // inst.operands = [extra_output_1, ..., extra_output_{n-1}, input_0, ...].
    // We build a combined list [output_0, extra_outputs..., inputs...] for
    // correct template substitution matching GCC's numbering scheme.
    if let Some(ref template) = inst.asm_template {
        let mut full_operands = Vec::new();
        // Output 0 = result register.
        if let Some(ref result) = inst.result {
            full_operands.push(result.clone());
        }
        // Remaining operands (extra outputs 1..n-1 followed by inputs).
        full_operands.extend(inst.operands.iter().cloned());
        return substitute_asm_operands(template, &full_operands);
    }
    if mnemonic == "# inline asm" || mnemonic == "<inline-asm>" {
        if let Some(MachineOperand::GlobalSymbol(ref template)) = inst.operands.first() {
            return template.clone();
        }
        return "# inline asm (no template)".to_string();
    }

    // Zero-operand instructions (ret, nop, leave, cqo, cdq, endbr64, pause, lfence).
    let has_result = inst.result.is_some();
    let num_ops = inst.operands.len();
    if !has_result && num_ops == 0 {
        return mnemonic.to_string();
    }

    // Build the full operand list in AT&T order: sources..., destination.
    // In our MachineInstruction:
    //   result = destination register (if present)
    //   operands = source operands
    // AT&T format: mnemonic src, dst
    let mut all_parts: Vec<String> = Vec::new();

    // Source operands first (in their original order).
    for op in &inst.operands {
        all_parts.push(format_operand_att(op));
    }

    // Destination (result) last in AT&T convention.
    if let Some(ref result) = inst.result {
        all_parts.push(format_operand_att(result));
    }

    if all_parts.is_empty() {
        return mnemonic.to_string();
    }

    // For single-operand instructions (push, pop, call, jmp, neg, not, etc.)
    if all_parts.len() == 1 {
        return format!("{} {}", mnemonic, all_parts[0]);
    }

    // For two or more operands: join with comma separator.
    format!("{} {}", mnemonic, all_parts.join(", "))
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::target::Target;

    /// Helper to create a default backend for testing.
    fn test_backend() -> X86_64Backend {
        X86_64Backend::new(
            Target::X86_64,
            SecurityConfig::default(),
            false,
            false,
            false,
        )
    }

    #[test]
    fn test_backend_construction() {
        let backend = test_backend();
        assert_eq!(backend.target(), Target::X86_64);
        assert!(!backend.is_pic());
        assert!(!backend.is_shared());
        assert!(!backend.has_debug_info());
    }

    #[test]
    fn test_backend_with_pic() {
        let backend = X86_64Backend::new(
            Target::X86_64,
            SecurityConfig::default(),
            true,
            false,
            false,
        );
        assert!(backend.is_pic());
        assert!(!backend.is_shared());
    }

    #[test]
    fn test_backend_with_shared() {
        let backend =
            X86_64Backend::new(Target::X86_64, SecurityConfig::default(), true, true, false);
        assert!(backend.is_pic());
        assert!(backend.is_shared());
    }

    #[test]
    fn test_backend_with_debug() {
        let backend = X86_64Backend::new(
            Target::X86_64,
            SecurityConfig::default(),
            false,
            false,
            true,
        );
        assert!(backend.has_debug_info());
    }

    #[test]
    fn test_security_config_accessor() {
        let config = SecurityConfig {
            retpoline: true,
            cf_protection: true,
            stack_probe: true,
        };
        let backend = X86_64Backend::new(Target::X86_64, config, false, false, false);
        assert!(backend.security_config().retpoline);
        assert!(backend.security_config().cf_protection);
        assert!(backend.security_config().stack_probe);
    }

    #[test]
    fn test_register_info() {
        let backend = test_backend();
        let info = backend.register_info();

        // 14 allocatable GPRs (16 - RSP - RBP)
        assert_eq!(info.allocatable_gpr.len(), 13); // R11 reserved as spill scratch
                                                    // 16 allocatable SSE registers
        assert_eq!(info.allocatable_fpr.len(), 15);
        // 5 callee-saved GPRs: RBX, R12, R13, R14, R15
        assert_eq!(info.callee_saved.len(), 5);
        // 8 caller-saved GPRs: RAX, RCX, RDX, RSI, RDI, R8-R10
        // (R11 is reserved as spill scratch)
        assert_eq!(info.caller_saved.len(), 8);
        // 3 reserved: RSP, RBP, R11
        assert_eq!(info.reserved.len(), 3);
        assert!(info.reserved.contains(&registers::RSP));
        assert!(info.reserved.contains(&registers::RBP));
        assert!(info.reserved.contains(&registers::R11));
        // 6 integer argument registers
        assert_eq!(info.argument_gpr.len(), 6);
        assert_eq!(info.argument_gpr[0], registers::RDI);
        // 8 SSE argument registers
        assert_eq!(info.argument_fpr.len(), 8);
        assert_eq!(info.argument_fpr[0], registers::XMM0);
        // 2 integer return registers
        assert_eq!(info.return_gpr.len(), 2);
        assert_eq!(info.return_gpr[0], registers::RAX);
        assert_eq!(info.return_gpr[1], registers::RDX);
        // 2 SSE return registers
        assert_eq!(info.return_fpr.len(), 2);
        assert_eq!(info.return_fpr[0], registers::XMM0);
        assert_eq!(info.return_fpr[1], registers::XMM1);
    }

    #[test]
    fn test_relocation_types() {
        let backend = test_backend();
        let relocs = backend.relocation_types();

        assert_eq!(relocs.len(), 10);
        // Spot-check a few entries
        assert_eq!(relocs[0].name, "R_X86_64_NONE");
        assert_eq!(relocs[0].type_id, 0);
        assert_eq!(relocs[2].name, "R_X86_64_PC32");
        assert_eq!(relocs[2].type_id, 2);
        assert!(relocs[2].is_pc_relative);
        assert_eq!(relocs[4].name, "R_X86_64_PLT32");
        assert_eq!(relocs[4].type_id, 4);
        assert!(relocs[4].is_pc_relative);
    }

    #[test]
    fn test_frame_pointer_reg() {
        let backend = test_backend();
        assert_eq!(backend.frame_pointer_reg(), registers::RBP);
    }

    #[test]
    fn test_stack_pointer_reg() {
        let backend = test_backend();
        assert_eq!(backend.stack_pointer_reg(), registers::RSP);
    }

    #[test]
    fn test_return_address_reg_is_none() {
        let backend = test_backend();
        // x86-64 has no link register — return address is on the stack.
        assert_eq!(backend.return_address_reg(), None);
    }

    #[test]
    fn test_prologue_basic() {
        let backend = test_backend();
        let mf = MachineFunction::new("test_fn".to_string());
        // frame_size=0, no callee-saved regs
        let prologue = backend.emit_prologue(&mf);

        // Expect: push rbp, mov rbp rsp (no sub rsp since frame_size=0)
        assert_eq!(prologue.len(), 2);
        // First instruction: PUSH RBP
        assert_eq!(prologue[0].opcode, X86Opcode::Push.as_u32());
        assert_eq!(
            prologue[0].operands[0],
            MachineOperand::Register(registers::RBP)
        );
        // Second instruction: MOV RBP, RSP
        assert_eq!(prologue[1].opcode, X86Opcode::Mov.as_u32());
    }

    #[test]
    fn test_prologue_with_frame() {
        let backend = test_backend();
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 32;
        let prologue = backend.emit_prologue(&mf);

        // Expect: push rbp, mov rbp rsp, sub rsp 32
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[2].opcode, X86Opcode::Sub.as_u32());
    }

    #[test]
    fn test_prologue_with_cet() {
        let config = SecurityConfig {
            retpoline: false,
            cf_protection: true,
            stack_probe: false,
        };
        let backend = X86_64Backend::new(Target::X86_64, config, false, false, false);
        let mf = MachineFunction::new("test_fn".to_string());
        let prologue = backend.emit_prologue(&mf);

        // Expect: endbr64, push rbp, mov rbp rsp
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[0].opcode, X86Opcode::Endbr64.as_u32());
    }

    #[test]
    fn test_prologue_with_stack_probe() {
        let config = SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe: true,
        };
        let backend = X86_64Backend::new(Target::X86_64, config, false, false, false);
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 8192; // > 4096, triggers probe

        let prologue = backend.emit_prologue(&mf);
        // Expect: push rbp, mov rbp rsp, probe_instruction(encoded_bytes)
        // The probe is emitted as a single MachineInstruction with pre-encoded
        // bytes (28 bytes for the entire probe loop including mov, sub, test,
        // cmp, ja, and final mov rsp,rax).
        assert_eq!(prologue.len(), 3);
        // The probe instruction should have non-empty encoded_bytes.
        assert!(!prologue[2].encoded_bytes.is_empty());
        assert_eq!(prologue[2].encoded_bytes.len(), 28);
    }

    #[test]
    fn test_prologue_with_callee_saved() {
        let backend = test_backend();
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 16;
        mf.callee_saved_regs = vec![registers::RBX, registers::R12];

        let prologue = backend.emit_prologue(&mf);
        // Expect: push rbp, mov rbp rsp, sub rsp 16, push rbx, push r12
        assert_eq!(prologue.len(), 5);
        // Last two should be PUSH for callee-saved
        assert_eq!(prologue[3].opcode, X86Opcode::Push.as_u32());
        assert_eq!(
            prologue[3].operands[0],
            MachineOperand::Register(registers::RBX)
        );
        assert_eq!(prologue[4].opcode, X86Opcode::Push.as_u32());
        assert_eq!(
            prologue[4].operands[0],
            MachineOperand::Register(registers::R12)
        );
    }

    #[test]
    fn test_epilogue_basic() {
        let backend = test_backend();
        let mf = MachineFunction::new("test_fn".to_string());
        let epilogue = backend.emit_epilogue(&mf);

        // Expect: mov rsp rbp, pop rbp, ret
        assert_eq!(epilogue.len(), 3);
        assert_eq!(epilogue[0].opcode, X86Opcode::Mov.as_u32());
        assert_eq!(epilogue[1].opcode, X86Opcode::Pop.as_u32());
        assert_eq!(epilogue[2].opcode, X86Opcode::Ret.as_u32());
        assert!(epilogue[2].is_terminator);
    }

    #[test]
    fn test_epilogue_callee_saved_reverse_order() {
        let backend = test_backend();
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.callee_saved_regs = vec![registers::RBX, registers::R12, registers::R13];

        let epilogue = backend.emit_epilogue(&mf);
        // Expect: lea rsp [rbp-N], pop r13, pop r12, pop rbx, mov rsp rbp, pop rbp, ret
        assert_eq!(epilogue.len(), 7);
        // First instruction is LEA to reset RSP before pops
        assert_eq!(epilogue[0].opcode, X86Opcode::Lea.as_u32());
        // Callee-saved restores in reverse order: R13, R12, RBX
        assert_eq!(
            epilogue[1].operands[0],
            MachineOperand::Register(registers::R13)
        );
        assert_eq!(
            epilogue[2].operands[0],
            MachineOperand::Register(registers::R12)
        );
        assert_eq!(
            epilogue[3].operands[0],
            MachineOperand::Register(registers::RBX)
        );
    }

    #[test]
    fn test_classify_argument_integer() {
        let backend = test_backend();
        let loc = backend.classify_argument(&IrType::I32);
        // First integer argument goes in RDI
        assert!(matches!(loc, TraitsArgLocation::Register(_)));
    }

    #[test]
    fn test_classify_argument_float() {
        let backend = test_backend();
        let loc = backend.classify_argument(&IrType::F64);
        // First FP argument goes in XMM0
        assert!(matches!(loc, TraitsArgLocation::Register(_)));
    }

    #[test]
    fn test_classify_return_integer() {
        let backend = test_backend();
        let loc = backend.classify_return(&IrType::I64);
        // Integer return in RAX
        assert!(matches!(loc, TraitsArgLocation::Register(_)));
    }

    #[test]
    fn test_classify_return_void() {
        let backend = test_backend();
        let loc = backend.classify_return(&IrType::Void);
        // Void return maps to RAX placeholder
        assert!(matches!(loc, TraitsArgLocation::Register(r) if r == registers::RAX));
    }

    #[test]
    fn test_re_exports_accessible() {
        // Verify that re-exported constants are accessible
        assert_eq!(super::RAX, 0);
        assert_eq!(super::RSP, 4);
        assert_eq!(super::RBP, 5);
        assert_eq!(super::XMM0, 16);
        assert_eq!(super::XMM15, 31);
        assert_eq!(super::R15, 15);
    }
}
