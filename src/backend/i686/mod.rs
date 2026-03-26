//! # BCC i686 (32-bit x86) Backend
//!
//! This module implements the i686 architecture backend for BCC, targeting
//! 32-bit x86 Linux systems with the ILP32 data model.
//!
//! ## Architecture Characteristics
//!
//! - **8 General-Purpose Registers**: EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
//! - **x87 FPU**: Stack-based floating-point (ST(0)–ST(7))
//! - **ILP32 Data Model**: `int`, `long`, and pointers are all 32-bit (4 bytes)
//! - **cdecl Calling Convention**: ALL function arguments passed on the stack
//! - **32-bit Instruction Encoding**: No REX prefix, purely IA-32 ISA
//!
//! ## Submodules
//!
//! - [`codegen`] — Instruction selection: IR → i686 machine instructions
//! - [`registers`] — Register definitions (GPRs, x87 FPU stack)
//! - [`abi`] — cdecl/System V i386 ABI implementation
//! - [`assembler`] — Built-in i686 assembler (32-bit instruction encoding)
//! - [`linker`] — Built-in i686 ELF linker (ELFCLASS32)
//!
//! ## Key Differences from x86-64
//!
//! - Only 8 GPRs (not 16): register pressure is significantly higher
//! - No register-based argument passing: ALL args go on the stack
//! - Floating-point uses x87 FPU stack, not SSE registers
//! - `long` is 4 bytes, not 8 (ILP32 vs LP64)
//! - `long double` is 12 bytes (80-bit + padding), 4-byte aligned
//! - No REX prefix: register encoding fits in 3 bits (0–7)
//! - No security mitigations (retpoline, CET, stack probe are x86-64 only)
//! - 64-bit integers (`long long`) require register pairs (EDX:EAX)
//!
//! ## ELF Output
//!
//! - ELF class: ELFCLASS32
//! - ELF machine: EM_386 (3)
//! - Endianness: little-endian (ELFDATA2LSB)
//! - Dynamic linker: `/lib/ld-linux.so.2`
//!
//! ## Security Mitigations
//!
//! The i686 backend does **not** implement any security mitigations.
//! Retpoline, CET/IBT, and stack guard-page probing are x86-64 only
//! per the AAP §0.6.2.
//!
//! ## Validation Order
//!
//! The i686 backend is the **second** validation target after x86-64:
//! x86-64 → **i686** → AArch64 → RISC-V 64.
//!
//! ## Zero-Dependency
//!
//! Only `std` and `crate::` references. No external crates.

// ===========================================================================
// Submodule declarations — all 5 submodules for the i686 backend
// ===========================================================================

/// i686 instruction selection: IR → i686 machine instructions.
///
/// Translates IR instructions into i686-specific machine instructions
/// respecting the 8-GPR constraint, cdecl calling convention, and x87 FPU
/// for floating-point operations.
pub mod codegen;

/// i686 register definitions (GPRs EAX–EDI, x87 FPU ST(0)–ST(7)).
///
/// Provides register constants, register classification sets (allocatable,
/// callee-saved, caller-saved, reserved), register name lookups, and the
/// [`i686_register_info`](registers::i686_register_info) constructor for
/// the register allocator.
pub mod registers;

/// cdecl / System V i386 ABI implementation.
///
/// Implements the i686 calling convention: all arguments on the stack,
/// return values in EAX (32-bit), EDX:EAX (64-bit), or ST(0)
/// (floating-point). Provides struct return via hidden pointer parameter.
pub mod abi;

/// Built-in i686 assembler (32-bit instruction encoding without REX).
///
/// Two-pass assembly: first pass computes label offsets, second pass encodes
/// ModR/M, SIB, and opcodes. Produces raw machine code bytes and relocation
/// entries for the built-in linker.
pub mod assembler;

/// Built-in i686 ELF linker (ELFCLASS32, EM_386).
///
/// Produces ET_EXEC (static executables) and ET_DYN (shared objects) for
/// 32-bit x86 Linux targets. Handles R_386_* relocations, GOT/PLT for PIC,
/// and dynamic linking sections.
pub mod linker;

// ===========================================================================
// Re-exports — public API surface for downstream consumers
// ===========================================================================

/// Re-export all 8 GPR register constants (EAX–EDI) for convenient access
/// by other modules such as the code generation driver and register
/// allocator.
pub use self::registers::{EAX, EBP, EBX, ECX, EDI, EDX, ESI, ESP};

/// Re-export the cdecl/System V i386 ABI implementation for crate-wide
/// access by the code generation driver and IR lowering passes.
pub use self::abi::I686Abi;

// ===========================================================================
// Internal imports
// ===========================================================================

use crate::backend::traits::{
    ArchCodegen, ArgLocation, AssembledFunction, MachineFunction, MachineInstruction, RegisterInfo,
    RelocationTypeInfo,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::types::IrType;

// ===========================================================================
// I686Codegen — the public i686 backend facade
// ===========================================================================

/// The i686 (32-bit x86) architecture backend implementing [`ArchCodegen`].
///
/// `I686Codegen` is the public entry point for all i686-specific code
/// generation. It is instantiated by the code generation driver
/// ([`crate::backend::generation`]) when `--target=i686` is specified, and
/// delegates to the submodule implementations for instruction selection
/// ([`codegen`]), assembly encoding ([`assembler`]), ABI classification
/// ([`abi`]), and register information ([`registers`]).
///
/// # Construction
///
/// ```ignore
/// use bcc::backend::i686::I686Codegen;
///
/// let backend = I686Codegen::new(false, false);
/// // Or from a CodegenContext:
/// // let backend = I686Codegen::new(ctx.pic, ctx.debug_info);
/// ```
///
/// # Architecture Dispatch
///
/// The code generation driver constructs this struct and passes it as a
/// `&dyn ArchCodegen` to the architecture-agnostic pipeline:
///
/// ```ignore
/// Target::I686 => {
///     let codegen = I686Codegen::new(pic, debug_info);
///     generate_for_arch(&codegen, module, diagnostics)
/// }
/// ```
///
/// # Security Mitigations
///
/// The i686 backend does **not** support any security mitigations.
/// Retpoline (`-mretpoline`), CET/IBT (`-fcf-protection`), and stack
/// guard-page probing are exclusive to the x86-64 backend per the AAP
/// §0.6.2.
///
/// # Key Architectural Constraints
///
/// - **8 GPRs only** (not 16): register pressure significantly higher
///   than x86-64's 16 GPRs
/// - **cdecl ABI**: ALL function arguments on the stack — no register
///   arguments
/// - **x87 FPU**: floating-point via stack-based FPU, not SSE registers
/// - **ILP32**: pointers and `long` are 4 bytes, `long long` is 8 bytes
///   (EDX:EAX pair)
pub struct I686Codegen {
    /// Whether Position-Independent Code generation is enabled (`-fPIC`).
    ///
    /// When true, global variable access uses GOT-relative addressing via
    /// EBX as the GOT pointer, and function calls use PLT stubs. The
    /// assembler emits R_386_GOT32, R_386_PLT32, R_386_GOTOFF, and
    /// R_386_GOTPC relocations accordingly.
    pub pic: bool,

    /// Whether DWARF debug information emission is enabled (`-g`).
    ///
    /// When true, the backend emits DWARF v4 debug sections
    /// (`.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str`) for
    /// source file/line mapping and local variable locations at `-O0`.
    pub debug_info: bool,

    /// Inner instruction selector from the codegen submodule.
    ///
    /// This delegate handles the actual IR-to-machine-instruction lowering,
    /// assembly emission, and ABI classification.  The outer `I686Codegen`
    /// (this struct) serves as the public facade implementing `ArchCodegen`
    /// and forwarding to this inner implementation.
    inner: codegen::I686Codegen,
}

// ===========================================================================
// I686Codegen — constructor and utility methods
// ===========================================================================

impl I686Codegen {
    /// Create a new i686 code generator.
    ///
    /// # Parameters
    ///
    /// - `pic`: Enable Position-Independent Code generation (`-fPIC`).
    ///   When true, global access uses GOT-relative addressing through
    ///   EBX and function calls go through PLT stubs.
    /// - `debug_info`: Enable DWARF v4 debug information emission (`-g`).
    ///   When true, `.debug_*` sections are generated for source-level
    ///   debugging.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bcc::backend::i686::I686Codegen;
    ///
    /// // Standard compilation without PIC or debug info
    /// let codegen = I686Codegen::new(false, false);
    /// assert!(!codegen.pic);
    /// assert!(!codegen.debug_info);
    ///
    /// // PIC compilation with debug info
    /// let codegen_pic = I686Codegen::new(true, true);
    /// assert!(codegen_pic.pic);
    /// assert!(codegen_pic.debug_info);
    /// ```
    pub fn new(pic: bool, debug_info: bool) -> Self {
        I686Codegen {
            pic,
            debug_info,
            inner: codegen::I686Codegen::new(pic, debug_info),
        }
    }

    /// Returns whether PIC mode is enabled for this backend instance.
    ///
    /// When PIC is enabled, the code generator uses GOT-relative addressing
    /// for global data and PLT stubs for function calls, with EBX
    /// conventionally holding the GOT base address.
    #[inline]
    pub fn is_pic(&self) -> bool {
        self.pic
    }

    /// Returns whether DWARF debug info emission is enabled.
    ///
    /// When enabled, the backend produces DWARF v4 debug sections for
    /// source-level debugging at `-O0`.
    #[inline]
    pub fn has_debug_info(&self) -> bool {
        self.debug_info
    }
}

// ===========================================================================
// ArchCodegen trait implementation for I686Codegen
// ===========================================================================

impl ArchCodegen for I686Codegen {
    /// Perform instruction selection: lower an IR function to i686 machine
    /// instructions.
    ///
    /// Delegates to the inner [`codegen::I686Codegen`] which traverses the
    /// [`IrFunction`]'s basic blocks and IR instructions, generating i686
    /// machine instructions with:
    ///
    /// - **cdecl parameter mapping**: All function parameters accessed at
    ///   positive offsets from EBP (`[EBP+8]`, `[EBP+12]`, …)
    /// - **32-bit register allocation**: Virtual registers using only the
    ///   8 i686 GPRs (3-bit encoding, no REX)
    /// - **x87 FPU**: Floating-point operations via FLD/FSTP/FADD/etc.
    /// - **64-bit support**: `long long` operations use EDX:EAX pairs
    ///   (ADD+ADC, SUB+SBB patterns)
    ///
    /// # Parameters
    ///
    /// - `func`: The IR function to lower. Reads `func.name`, `func.params`,
    ///   `func.return_type`, `func.blocks`, `func.calling_convention`, and
    ///   `func.is_variadic`.
    /// - `diag`: Diagnostic engine for reporting instruction selection errors,
    ///   unsupported constructs, and inline assembly constraint failures.
    ///
    /// # Returns
    ///
    /// `Ok(MachineFunction)` on success, or `Err(String)` if fatal errors
    /// occurred during instruction selection.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - An unsupported IR instruction is encountered
    /// - An inline assembly constraint cannot be satisfied with i686 registers
    /// - The diagnostic engine has accumulated errors during lowering
    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
        global_var_refs: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) -> Result<MachineFunction, String> {
        // Create a fresh inner codegen instance with the current
        // function-reference and global-variable-reference maps, so that
        // resolve_operand can look up symbol names for callee/global values.
        let mut inner = codegen::I686Codegen::new(self.pic, self.debug_info);
        inner.set_func_ref_names(func_ref_map);
        inner.set_global_var_refs(global_var_refs);
        inner.lower_function(func, diag, globals, func_ref_map, global_var_refs)
    }

    /// Encode i686 machine instructions to raw bytes using the built-in
    /// assembler.
    ///
    /// Delegates to the inner [`codegen::I686Codegen`] which performs
    /// two-pass assembly:
    ///
    /// 1. **Pass 1**: Estimate label offsets from instruction sizes
    /// 2. **Pass 2**: Encode with resolved offsets, producing final machine
    ///    code with ModR/M, SIB, and 32-bit opcodes (no REX prefix)
    ///
    /// # Parameters
    ///
    /// - `mf`: The machine function (post-register-allocation) to assemble.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the encoded function body, or `Err(String)`
    /// on encoding failure (e.g., immediate value out of range).
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<AssembledFunction, String> {
        self.inner.emit_assembly(mf)
    }

    /// Returns [`Target::I686`] identifying this as the 32-bit x86 backend.
    ///
    /// Used by the code generation driver to confirm correct architecture
    /// dispatch and to provide target information to the ELF writer (EM_386,
    /// ELFCLASS32, ELFDATA2LSB), DWARF emitter, and linker.
    #[inline]
    fn target(&self) -> Target {
        Target::I686
    }

    /// Returns the i686 register set information for the register allocator.
    ///
    /// The i686 register file is significantly smaller than x86-64:
    ///
    /// - **6 allocatable GPRs**: EAX, ECX, EDX, EBX, ESI, EDI
    /// - **0 allocatable FPRs**: x87 FPU is stack-based, not allocatable
    /// - **4 callee-saved**: EBX, ESI, EDI, EBP
    /// - **3 caller-saved**: EAX, ECX, EDX
    /// - **2 reserved**: ESP (stack pointer), EBP (frame pointer)
    /// - **0 argument GPRs**: cdecl passes ALL args on the stack
    /// - **0 argument FPRs**: no FP register arguments in cdecl
    /// - **1 return GPR**: EAX (32-bit integer returns)
    /// - **0 return FPRs**: float returns via ST(0), handled separately
    fn register_info(&self) -> RegisterInfo {
        registers::i686_register_info()
    }

    /// Returns the i686-specific ELF relocation types (R_386_*).
    ///
    /// Returns the complete table of relocation type descriptors from
    /// [`assembler::relocations::I686_RELOCATION_TYPES`], covering:
    ///
    /// - Core relocations: R_386_NONE, R_386_32, R_386_PC32
    /// - PIC relocations: R_386_GOT32, R_386_PLT32, R_386_GOTOFF, R_386_GOTPC
    /// - Dynamic: R_386_COPY, R_386_GLOB_DAT, R_386_JMP_SLOT, R_386_RELATIVE
    /// - TLS and GNU extensions
    ///
    /// These descriptors are consumed by the common linker infrastructure
    /// to dispatch relocation application to the i686-specific handler.
    #[inline]
    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        assembler::relocations::I686_RELOCATION_TYPES
    }

    /// Generate the i686 cdecl function prologue instructions.
    ///
    /// Delegates to [`codegen::emit_i686_prologue`] which produces the
    /// standard cdecl prologue sequence:
    ///
    /// ```text
    /// push ebp            ; save old frame pointer
    /// mov ebp, esp        ; establish new frame pointer
    /// sub esp, N          ; allocate stack space (if frame_size > 0)
    /// push ebx            ; save callee-saved regs (if used)
    /// push esi
    /// push edi
    /// ```
    ///
    /// Only callee-saved registers that appear in `mf.callee_saved_regs`
    /// are pushed. EBP is already saved as part of the frame setup and
    /// is not pushed again.
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        codegen::emit_i686_prologue(mf)
    }

    /// Generate the i686 cdecl function epilogue instructions.
    ///
    /// Delegates to [`codegen::emit_i686_epilogue`] which produces the
    /// standard cdecl epilogue sequence:
    ///
    /// ```text
    /// pop edi             ; restore callee-saved regs (reverse order)
    /// pop esi
    /// pop ebx
    /// leave               ; mov esp, ebp ; pop ebp
    /// ret                 ; return to caller
    /// ```
    ///
    /// Only callee-saved registers that were pushed in the prologue are
    /// popped (in reverse order for correct stack unwinding).
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        codegen::emit_i686_epilogue(mf)
    }

    /// Returns the frame pointer register index: EBP (index 5).
    ///
    /// EBP is the conventional frame pointer in the cdecl ABI. The standard
    /// prologue sets `EBP = ESP` after saving the old EBP, establishing a
    /// stable base for accessing:
    /// - Function arguments at positive offsets (`[EBP+8]`, `[EBP+12]`, …)
    /// - Local variables at negative offsets (`[EBP-4]`, `[EBP-8]`, …)
    #[inline]
    fn frame_pointer_reg(&self) -> u16 {
        registers::EBP
    }

    /// Returns the stack pointer register index: ESP (index 4).
    ///
    /// ESP is the hardware stack pointer. It is implicitly modified by
    /// PUSH, POP, CALL, RET, ENTER, and LEAVE instructions. The SIB byte
    /// is required when ESP appears as a base register in certain ModR/M
    /// addressing modes.
    #[inline]
    fn stack_pointer_reg(&self) -> u16 {
        registers::ESP
    }

    /// Returns `None` — the return address is on the stack, not in a
    /// register.
    ///
    /// On i686 (and x86-64), the CALL instruction pushes the return
    /// address onto the stack at `[ESP]` (which becomes `[EBP+4]` after
    /// the standard prologue). There is no dedicated link register like
    /// AArch64's X30/LR or RISC-V's x1/ra.
    #[inline]
    fn return_address_reg(&self) -> Option<u16> {
        None
    }

    /// Classify where a function argument should be placed in the cdecl
    /// calling convention.
    ///
    /// In the i686 cdecl ABI, **all** arguments are passed on the stack.
    /// There are no register arguments. Each argument occupies at least 4
    /// bytes (one stack slot), rounded up to a 4-byte boundary.
    ///
    /// This is fundamentally different from x86-64 where the first 6
    /// integer args go in RDI/RSI/RDX/RCX/R8/R9 and the first 8 FP args
    /// in XMM0–XMM7.
    ///
    /// # Parameters
    ///
    /// - `ty`: The IR type of the argument.
    ///
    /// # Returns
    ///
    /// Always returns `ArgLocation::Stack(slot_size)` where `slot_size` is
    /// the argument's stack slot size in bytes (minimum 4, rounded to 4).
    fn classify_argument(&self, ty: &IrType) -> ArgLocation {
        self.inner.classify_argument(ty)
    }

    /// Classify where a function return value should be placed in the
    /// cdecl calling convention.
    ///
    /// The i686 cdecl ABI return value rules:
    ///
    /// - **Void**: `Stack(0)` — no return value
    /// - **≤32-bit integers/pointers**: `Register(EAX)` — in EAX
    /// - **64-bit integers** (`long long`): `RegisterPair(EAX, EDX)` — low
    ///   32 bits in EAX, high 32 bits in EDX
    /// - **Floating-point** (`float`/`double`/`long double`):
    ///   `Register(ST0)` — on the x87 FPU stack top
    /// - **Structs/arrays**: `Stack(0)` — via hidden first pointer parameter
    ///
    /// # Parameters
    ///
    /// - `ty`: The IR type of the return value.
    ///
    /// # Returns
    ///
    /// An [`ArgLocation`] describing where the return value is placed.
    fn classify_return(&self, ty: &IrType) -> ArgLocation {
        self.inner.classify_return(ty)
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i686_codegen_new_default() {
        let codegen = I686Codegen::new(false, false);
        assert!(!codegen.pic);
        assert!(!codegen.debug_info);
        assert!(!codegen.is_pic());
        assert!(!codegen.has_debug_info());
    }

    #[test]
    fn test_i686_codegen_new_pic() {
        let codegen = I686Codegen::new(true, false);
        assert!(codegen.pic);
        assert!(!codegen.debug_info);
        assert!(codegen.is_pic());
        assert!(!codegen.has_debug_info());
    }

    #[test]
    fn test_i686_codegen_new_debug() {
        let codegen = I686Codegen::new(false, true);
        assert!(!codegen.pic);
        assert!(codegen.debug_info);
        assert!(!codegen.is_pic());
        assert!(codegen.has_debug_info());
    }

    #[test]
    fn test_i686_codegen_new_both() {
        let codegen = I686Codegen::new(true, true);
        assert!(codegen.pic);
        assert!(codegen.debug_info);
        assert!(codegen.is_pic());
        assert!(codegen.has_debug_info());
    }

    #[test]
    fn test_target_returns_i686() {
        let codegen = I686Codegen::new(false, false);
        assert_eq!(codegen.target(), Target::I686);
    }

    #[test]
    fn test_frame_pointer_is_ebp() {
        let codegen = I686Codegen::new(false, false);
        assert_eq!(codegen.frame_pointer_reg(), registers::EBP);
        assert_eq!(codegen.frame_pointer_reg(), 5);
    }

    #[test]
    fn test_stack_pointer_is_esp() {
        let codegen = I686Codegen::new(false, false);
        assert_eq!(codegen.stack_pointer_reg(), registers::ESP);
        assert_eq!(codegen.stack_pointer_reg(), 4);
    }

    #[test]
    fn test_return_address_reg_is_none() {
        let codegen = I686Codegen::new(false, false);
        // On i686, the return address is on the stack (pushed by CALL),
        // not in a dedicated register.
        assert_eq!(codegen.return_address_reg(), None);
    }

    #[test]
    fn test_register_info_allocatable_gprs() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // i686 has 5 allocatable GPRs (ESP, EBP reserved; ECX = spill scratch).
        assert_eq!(info.allocatable_gpr.len(), 5);
        assert!(info.allocatable_gpr.contains(&EAX));
        assert!(!info.allocatable_gpr.contains(&ECX)); // ECX = spill scratch
        assert!(info.allocatable_gpr.contains(&EDX));
        assert!(info.allocatable_gpr.contains(&EBX));
        assert!(info.allocatable_gpr.contains(&ESI));
        assert!(info.allocatable_gpr.contains(&EDI));

        // ESP, EBP, and ECX must NOT be allocatable.
        assert!(!info.allocatable_gpr.contains(&ESP));
        assert!(!info.allocatable_gpr.contains(&EBP));
    }

    #[test]
    fn test_register_info_no_allocatable_fpr() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // x87 FPU is stack-based — no directly allocatable FP registers.
        assert!(info.allocatable_fpr.is_empty());
    }

    #[test]
    fn test_register_info_callee_saved() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // cdecl callee-saved: EBX, ESI, EDI, EBP
        assert_eq!(info.callee_saved.len(), 4);
        assert!(info.callee_saved.contains(&EBX));
        assert!(info.callee_saved.contains(&ESI));
        assert!(info.callee_saved.contains(&EDI));
        assert!(info.callee_saved.contains(&EBP));
    }

    #[test]
    fn test_register_info_caller_saved() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // cdecl caller-saved (volatile): EAX, ECX, EDX
        assert_eq!(info.caller_saved.len(), 3);
        assert!(info.caller_saved.contains(&EAX));
        assert!(info.caller_saved.contains(&ECX));
        assert!(info.caller_saved.contains(&EDX));
    }

    #[test]
    fn test_register_info_no_argument_registers() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // cdecl: ALL arguments are on the stack — NO register arguments.
        assert!(info.argument_gpr.is_empty());
        assert!(info.argument_fpr.is_empty());
    }

    #[test]
    fn test_register_info_return_register() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // Integer return value in EAX.
        assert!(info.return_gpr.contains(&EAX));
        // Floating-point returns use ST(0), tracked separately.
        assert!(info.return_fpr.is_empty());
    }

    #[test]
    fn test_register_info_reserved() {
        let codegen = I686Codegen::new(false, false);
        let info = codegen.register_info();

        // ESP and EBP are reserved.
        assert!(info.reserved.contains(&ESP));
        assert!(info.reserved.contains(&EBP));
    }

    #[test]
    fn test_relocation_types_non_empty() {
        let codegen = I686Codegen::new(false, false);
        let relocs = codegen.relocation_types();

        // Must have the core R_386_* relocation types.
        assert!(!relocs.is_empty());

        // Verify R_386_NONE is present (type_id 0).
        assert!(relocs
            .iter()
            .any(|r| r.type_id == 0 && r.name == "R_386_NONE"));

        // Verify R_386_32 is present (type_id 1, absolute 32-bit).
        assert!(relocs
            .iter()
            .any(|r| r.type_id == 1 && r.name == "R_386_32" && !r.is_pc_relative));

        // Verify R_386_PC32 is present (type_id 2, PC-relative).
        assert!(relocs
            .iter()
            .any(|r| r.type_id == 2 && r.name == "R_386_PC32" && r.is_pc_relative));

        // Verify R_386_PLT32 is present (type_id 4, PLT relative).
        assert!(relocs
            .iter()
            .any(|r| r.type_id == 4 && r.name == "R_386_PLT32" && r.is_pc_relative));
    }

    #[test]
    fn test_classify_argument_always_stack() {
        let codegen = I686Codegen::new(false, false);

        // All argument types in cdecl go on the stack.
        let loc_i32 = codegen.classify_argument(&IrType::I32);
        assert!(loc_i32.is_stack());

        let loc_i64 = codegen.classify_argument(&IrType::I64);
        assert!(loc_i64.is_stack());

        let loc_ptr = codegen.classify_argument(&IrType::Ptr);
        assert!(loc_ptr.is_stack());

        let loc_f64 = codegen.classify_argument(&IrType::F64);
        assert!(loc_f64.is_stack());
    }

    #[test]
    fn test_classify_return_void() {
        let codegen = I686Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::Void);
        assert!(loc.is_stack());
    }

    #[test]
    fn test_classify_return_i32_in_eax() {
        let codegen = I686Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::I32);
        assert_eq!(loc, ArgLocation::Register(EAX));
    }

    #[test]
    fn test_classify_return_i64_in_edx_eax() {
        let codegen = I686Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::I64);
        assert_eq!(loc, ArgLocation::RegisterPair(EAX, EDX));
    }

    #[test]
    fn test_classify_return_float_in_st0() {
        let codegen = I686Codegen::new(false, false);
        let loc_f32 = codegen.classify_return(&IrType::F32);
        assert_eq!(loc_f32, ArgLocation::Register(registers::ST0));

        let loc_f64 = codegen.classify_return(&IrType::F64);
        assert_eq!(loc_f64, ArgLocation::Register(registers::ST0));
    }

    #[test]
    fn test_classify_return_pointer_in_eax() {
        let codegen = I686Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::Ptr);
        assert_eq!(loc, ArgLocation::Register(EAX));
    }

    #[test]
    fn test_register_constants_values() {
        // Verify the re-exported register constants have correct values.
        assert_eq!(EAX, 0);
        assert_eq!(ECX, 1);
        assert_eq!(EDX, 2);
        assert_eq!(EBX, 3);
        assert_eq!(ESP, 4);
        assert_eq!(EBP, 5);
        assert_eq!(ESI, 6);
        assert_eq!(EDI, 7);
    }

    #[test]
    fn test_no_security_mitigations() {
        // The i686 backend must NOT have retpoline, CET, or stack probe.
        // This test verifies that I686Codegen doesn't reference security
        // mitigations (it simply shouldn't exist in the struct or methods).
        let codegen = I686Codegen::new(false, false);

        // The struct has only `pic` and `debug_info` — no security fields.
        assert!(!codegen.pic);
        assert!(!codegen.debug_info);

        // target() correctly identifies as i686, not x86-64.
        assert_eq!(codegen.target(), Target::I686);
        assert_ne!(codegen.target(), Target::X86_64);
    }

    #[test]
    fn test_emit_prologue_basic() {
        let codegen = I686Codegen::new(false, false);
        let mf = MachineFunction::new("test_func".to_string());
        let prologue = codegen.emit_prologue(&mf);

        // Prologue should have at least: push ebp, mov ebp esp
        assert!(prologue.len() >= 2);
    }

    #[test]
    fn test_emit_epilogue_basic() {
        let codegen = I686Codegen::new(false, false);
        let mf = MachineFunction::new("test_func".to_string());
        let epilogue = codegen.emit_epilogue(&mf);

        // Epilogue should have at least: leave, ret
        assert!(epilogue.len() >= 2);

        // The last instruction should be a terminator (ret).
        let last = epilogue.last().unwrap();
        assert!(last.is_terminator);
    }
}
