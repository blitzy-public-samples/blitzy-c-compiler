//! # Phase 10 — Code Generation Driver
//!
//! This module is the central dispatch point for BCC's backend, orchestrating
//! the entire machine-code emission pipeline:
//!
//! 1. Receives phi-eliminated IR from Phase 9
//! 2. Dispatches to the correct architecture backend based on `--target` flag
//! 3. Injects security mitigations (retpoline, CET/IBT, stack probe) for x86-64
//! 4. Orchestrates relocatable object file (`.o`) emission
//! 5. Coordinates the built-in assembler and built-in linker for final ELF
//!    production (ET_EXEC or ET_DYN)
//!
//! ## Standalone Backend Mode
//!
//! BCC does **not** invoke any external toolchain component (no `as`, `ld`,
//! `gcc`, `llvm-mc`, `lld`).  Every step — instruction selection, register
//! allocation, assembly encoding, relocatable object writing, linking — is
//! performed by built-in modules.
//!
//! ## Security Mitigations (x86-64 Only)
//!
//! - **Retpoline** (`-mretpoline`): Indirect call/jump thunks
//! - **CET/IBT** (`-fcf-protection`): `endbr64` insertion
//! - **Stack Probe**: Guard-page probe loop for frames > 4096 bytes
//!
//! These mitigations are injected **only** for the x86-64 target.  Other
//! architectures are unaffected regardless of CLI flags.
//!
//! ## DWARF Debug Information
//!
//! When `-g` is active (and `-O0`), DWARF v4 sections are generated.
//! Without `-g`, **zero** `.debug_*` sections appear in the output.
//!
//! ## Zero-Dependency
//!
//! This module depends only on `std` and `crate::` references.  No external
//! crates are used.

use crate::backend::dwarf::{
    dwarf_address_size, generate_dwarf_sections, should_emit_dwarf, DwarfSections, DWARF_VERSION,
};
use crate::backend::elf_writer_common::{
    Elf32Rel, Elf64Rela, ElfSymbol, ElfWriter, Section, ET_REL, SHF_ALLOC, SHF_EXECINSTR,
    SHF_INFO_LINK, SHF_WRITE, SHT_NOBITS, SHT_PROGBITS, SHT_REL, SHT_RELA, STB_GLOBAL, STB_LOCAL,
    STB_WEAK, STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT, STV_HIDDEN, STV_PROTECTED,
};
use crate::backend::linker_common::relocation::RelocationHandler;
use crate::backend::linker_common::{link, LinkerConfig, LinkerInput, LinkerOutput, OutputType};
use crate::backend::register_allocator::{
    self, allocate_registers, compute_live_intervals, AllocationResult,
};
use crate::backend::traits::{
    ArchCodegen, MachineFunction, MachineInstruction, MachineOperand,
    RegisterInfo as TraitsRegisterInfo,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::ir::function::{IrFunction, Linkage as FunctionLinkage, Visibility};
use crate::ir::module::IrModule;

// ===========================================================================
// CodegenContext — compilation options for the code generation phase
// ===========================================================================

/// Compilation options collected from the CLI flags that control the
/// code generation phase.
///
/// `CodegenContext` is constructed by the CLI driver (`src/main.rs`) and
/// passed through the pipeline to every backend component — instruction
/// selector, register allocator, ELF writer, DWARF emitter, and linker.
///
/// # Security Mitigations
///
/// The `retpoline` and `cf_protection` fields only affect x86-64 targets.
/// When set for other architectures, they are silently ignored by the
/// code generation driver (no error — this matches GCC's behavior of
/// ignoring target-inapplicable flags).
///
/// # Examples
///
/// ```ignore
/// use bcc::backend::generation::CodegenContext;
/// use bcc::common::target::Target;
///
/// let ctx = CodegenContext {
///     target: Target::X86_64,
///     debug_info: true,
///     optimization_level: 0,
///     pic: false,
///     shared: false,
///     retpoline: true,
///     cf_protection: true,
///     output_path: "hello".to_string(),
///     compile_only: false,
///     emit_assembly: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct CodegenContext {
    /// Target architecture (x86-64, i686, AArch64, or RISC-V 64).
    ///
    /// Determines which `ArchCodegen` implementation is instantiated, which
    /// ELF header constants are used, and which ABI conventions apply.
    pub target: Target,

    /// Whether to emit DWARF v4 debug information (`-g` flag).
    ///
    /// When `true` and `optimization_level == 0`, the backend generates
    /// `.debug_info`, `.debug_abbrev`, `.debug_line`, and `.debug_str`
    /// sections.  When `false`, **zero** debug sections are present in the
    /// output binary.
    pub debug_info: bool,

    /// Optimization level (`-O0`, `-O1`, etc.).
    ///
    /// Currently only `-O0` is fully supported.  DWARF debug information
    /// is only emitted at `-O0`.  The value 0 corresponds to no optimization.
    pub optimization_level: u8,

    /// Whether to generate position-independent code (`-fPIC` flag).
    ///
    /// When `true`, global variable access uses GOT-relative addressing
    /// and function calls use PLT stubs.  Required for shared libraries.
    pub pic: bool,

    /// Whether to produce a shared object / shared library (`-shared` flag).
    ///
    /// When `true`, the output is an ET_DYN ELF with `.dynamic`, `.dynsym`,
    /// `.gnu.hash`, GOT, and PLT sections.  Implies PIC code generation.
    pub shared: bool,

    /// Whether to enable retpoline indirect-call thunks (`-mretpoline` flag).
    ///
    /// **x86-64 only.**  Transforms indirect call/jump instructions to use
    /// `__x86_indirect_thunk_*` thunks instead of direct `call *%reg`.
    /// Silently ignored for non-x86-64 targets.
    pub retpoline: bool,

    /// Whether to enable Intel CET / IBT (`-fcf-protection` flag).
    ///
    /// **x86-64 only.**  Inserts `endbr64` at function entries and indirect
    /// branch targets, and generates a `.note.gnu.property` section with
    /// CET flags.  Silently ignored for non-x86-64 targets.
    pub cf_protection: bool,

    /// Output file path (`-o` flag).
    ///
    /// The path where the final ELF binary (or assembly text, or
    /// relocatable object) is written.
    pub output_path: String,

    /// Whether to produce only a relocatable object file (`-c` flag).
    ///
    /// When `true`, the built-in linker is **not** invoked.  The output
    /// is a `.o` file (ET_REL) suitable for later linking.
    pub compile_only: bool,

    /// Whether to emit human-readable assembly text (`-S` flag).
    ///
    /// When `true`, the output is AT&T-syntax assembly (x86-64/i686) or
    /// standard assembly (AArch64/RISC-V 64) instead of binary machine code.
    pub emit_assembly: bool,

    /// Library search paths (`-L` flags).
    pub library_paths: Vec<String>,

    /// Libraries to link (`-l` flags — names without the `lib` prefix).
    pub libraries: Vec<String>,
}

impl CodegenContext {
    /// Returns `true` if DWARF debug sections should be generated.
    ///
    /// Delegates to [`should_emit_dwarf`] — DWARF is emitted only when
    /// both `-g` is active and the optimization level is 0.
    #[inline]
    pub fn should_emit_dwarf(&self) -> bool {
        should_emit_dwarf(self.debug_info, self.optimization_level)
    }

    /// Returns `true` if security mitigations are applicable.
    ///
    /// Security mitigations (retpoline, CET/IBT, stack probe) are only
    /// supported on x86-64.  This helper is used by the code generation
    /// driver to guard mitigation injection.
    #[inline]
    pub fn has_security_mitigations(&self) -> bool {
        self.target == Target::X86_64 && (self.retpoline || self.cf_protection)
    }

    /// Determines the linker output type based on CLI flags.
    ///
    /// - `-c` → `Relocatable` (no linking)
    /// - `-shared` → `SharedLibrary` (ET_DYN)
    /// - otherwise → `Executable` (ET_EXEC)
    pub fn output_type(&self) -> OutputType {
        if self.compile_only {
            OutputType::Relocatable
        } else if self.shared {
            OutputType::SharedLibrary
        } else {
            OutputType::Executable
        }
    }
}

impl Default for CodegenContext {
    fn default() -> Self {
        CodegenContext {
            target: Target::X86_64,
            debug_info: false,
            optimization_level: 0,
            pic: false,
            shared: false,
            retpoline: false,
            cf_protection: false,
            output_path: String::from("a.out"),
            compile_only: false,
            emit_assembly: false,
            library_paths: Vec::new(),
            libraries: Vec::new(),
        }
    }
}

// ===========================================================================
// generate_code — main entry point for Phase 10
// ===========================================================================

/// Main entry point for Phase 10 code generation.
///
/// Receives an IR module (post phi-elimination from Phase 9), dispatches
/// to the correct architecture backend based on the target specified in
/// `ctx`, and produces the final output — either a relocatable object,
/// an ELF executable, a shared library, or assembly text.
///
/// # Pipeline
///
/// 1. **Architecture dispatch**: Instantiate the correct `ArchCodegen`
///    implementation based on `ctx.target`.
/// 2. **Per-function code generation**: For each function definition in the
///    module, perform instruction selection, register allocation, and
///    assembly encoding.
/// 3. **Security mitigation injection** (x86-64 only): Apply retpoline,
///    CET/IBT, and stack probe transformations.
/// 4. **Global data emission**: Emit global variables and string literal
///    pool into `.data` / `.rodata` / `.bss` sections.
/// 5. **DWARF debug info** (conditional on `-g`): Generate `.debug_info`,
///    `.debug_abbrev`, `.debug_line`, `.debug_str` sections.
/// 6. **ELF object writing**: Produce a relocatable `.o` file.
/// 7. **Linking** (unless `-c`): Invoke the built-in linker to produce
///    the final ET_EXEC or ET_DYN ELF binary.
///
/// # Parameters
///
/// - `module`: The IR module to generate code for.
/// - `ctx`: Compilation options (target, flags, output path).
/// - `diagnostics`: Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// On success, returns `Ok(Vec<u8>)` containing the final output bytes
/// (ELF binary, relocatable object, or assembly text).
/// On failure, returns `Err(String)` with a description of the fatal error.
///
/// # Errors
///
/// - Instruction selection failure (unsupported IR constructs)
/// - Register allocation failure (internal consistency error)
/// - Assembly encoding failure (immediate out of range)
/// - DWARF generation errors
/// - Linking errors (undefined symbols, relocation overflow)
pub fn generate_code(
    module: &IrModule,
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
    source_map: &SourceMap,
) -> Result<Vec<u8>, String> {
    // Dispatch to the appropriate architecture backend.
    match ctx.target {
        Target::X86_64 => {
            let security_config = crate::backend::x86_64::SecurityConfig {
                retpoline: ctx.retpoline,
                cf_protection: ctx.cf_protection,
                stack_probe: true, // Always enabled for x86-64
            };
            let codegen = crate::backend::x86_64::X86_64Backend::new(
                Target::X86_64,
                security_config,
                ctx.pic,
                ctx.shared,
                ctx.debug_info,
            );
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
        Target::I686 => {
            let codegen = crate::backend::i686::I686Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
        Target::AArch64 => {
            let codegen = crate::backend::aarch64::AArch64Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
        Target::RiscV64 => {
            let mut codegen = crate::backend::riscv64::RiscV64Codegen::new(ctx.pic, ctx.debug_info);
            // Populate the variadic function set from module declarations.
            // RISC-V LP64D ABI requires variadic FP arguments to be passed
            // in integer registers instead of FP registers.
            let variadic: crate::common::fx_hash::FxHashSet<String> = module
                .declarations()
                .iter()
                .filter(|d| d.is_variadic)
                .map(|d| d.name.clone())
                .collect();
            codegen.set_variadic_functions(variadic);
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
    }
}

// ===========================================================================
// generate_for_arch — architecture-agnostic code generation pipeline
// ===========================================================================

/// Architecture-agnostic code generation pipeline.
///
/// Given an `ArchCodegen` implementation, processes the entire IR module
/// through instruction selection, register allocation, assembly encoding,
/// optional DWARF generation, ELF object writing, and optional linking.
///
/// This function is the core workhorse of Phase 10.  It is called once per
/// compilation with the architecture-specific backend selected by
/// [`generate_code`].
///
/// # Type Parameter
///
/// - `A`: An architecture backend implementing [`ArchCodegen`].
///
/// # Pipeline Steps
///
/// 1. For each function definition in the module:
///    a. `arch.lower_function()` → instruction selection → `MachineFunction`
///    b. `allocate_registers()` → register allocation
///    c. `arch.emit_prologue()` / `arch.emit_epilogue()` → frame setup/teardown
///    d. `arch.emit_assembly()` → encode to raw bytes
/// 2. Emit globals into `.data` / `.rodata` / `.bss` sections.
/// 3. Emit string literal pool into `.rodata`.
/// 4. If `-g`: generate DWARF sections; else: zero debug sections.
/// 5. Write the relocatable object via `ElfWriter`.
/// 6. If not `-c`: invoke the built-in linker.
fn generate_for_arch<A: ArchCodegen>(
    arch: &A,
    module: &IrModule,
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
    source_map: &SourceMap,
) -> Result<Vec<u8>, String> {
    // If `-S` is specified, emit assembly text instead of binary.
    if ctx.emit_assembly {
        return emit_assembly_text(arch, module, ctx, diagnostics);
    }

    // Validate that the architecture backend's target matches the context.
    let backend_target = arch.target();
    if backend_target != ctx.target {
        diagnostics.emit_error(
            Span::dummy(),
            format!(
                "codegen: backend target mismatch: expected {}, got {}",
                ctx.target, backend_target
            ),
        );
        return Err("code generation aborted: target mismatch".to_string());
    }

    // Warn if security mitigations are requested on non-x86-64 targets.
    if (ctx.retpoline || ctx.cf_protection) && ctx.target != Target::X86_64 {
        diagnostics.emit_warning(
            Span::dummy(),
            format!(
                "codegen: security mitigations (retpoline={}, cf_protection={}) are only supported on x86-64; ignoring for {}",
                ctx.retpoline, ctx.cf_protection, ctx.target
            ),
        );
    }

    // Get the register info for this architecture (used for register allocation).
    let reg_info: TraitsRegisterInfo = arch.register_info();

    // Convert to the register allocator's internal RegisterInfo representation.
    let alloc_reg_info: register_allocator::RegisterInfo = reg_info.into();

    // -----------------------------------------------------------------------
    // Phase A: Per-Function Code Generation
    // -----------------------------------------------------------------------
    // For each function definition: instruction selection → register allocation
    // → prologue/epilogue → assembly encoding.

    let mut compiled_functions: Vec<CompiledFunction> = Vec::new();
    let mut text_section_data: Vec<u8> = Vec::new();
    let mut text_section_offsets: Vec<(String, u64, u64)> = Vec::new();
    let mut text_relocations: Vec<crate::backend::traits::FunctionRelocation> = Vec::new();

    let functions: &[IrFunction] = module.functions();
    for func in functions {
        // Skip declaration-only functions (no body to compile).
        if !func.is_definition {
            continue;
        }

        // Step 1: Instruction selection — IR → MachineFunction
        //
        // Use the per-function ref maps (scoped to this function's Value IDs)
        // rather than the module-level maps which contain Values from ALL
        // functions and suffer from cross-function Value-ID collisions.
        let mut mf = match arch.lower_function(
            func,
            diagnostics,
            module.globals(),
            &func.func_ref_map,
            &func.global_var_refs,
        ) {
            Ok(mf) => mf,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!(
                        "codegen: instruction selection failed for '{}': {}",
                        func.name, e
                    ),
                );
                if diagnostics.has_errors() {
                    return Err(format!(
                        "code generation aborted: instruction selection failed for '{}'",
                        func.name
                    ));
                }
                continue;
            }
        };

        // Step 2: Register allocation
        //
        // Compute live intervals from the IR function's blocks() and
        // parameter/return-type metadata, then allocate physical registers.
        let _num_blocks = func.blocks().len();
        let _num_params = func.params.len();
        let _ret_type = &func.return_type;

        let mut intervals = compute_live_intervals(func);
        let alloc_result = allocate_registers(&mut intervals, &alloc_reg_info, &ctx.target);

        // Insert spill/reload code for any virtual registers that couldn't
        // be assigned a physical register.  The spill code is recorded in the
        // allocation result and applied to the MachineFunction below —
        // modifying the IR function is not required.
        insert_spill_code_from_result(&mut mf, &alloc_result);

        // Update MachineFunction with allocation results.
        apply_allocation_result(&mut mf, &alloc_result, &ctx.target);

        // ── Stack alignment fixup ────────────────────────────────────
        // System V AMD64 / i386 ABIs require RSP ≡ 0 (mod 16) before
        // any CALL instruction.  After the prologue:
        //
        //   push rbp             ; RSP goes from 8 mod 16 → 0 mod 16
        //   sub rsp, frame_size  ; RSP stays 0 mod 16 (frame_size is 16-aligned)
        //   push <callee-saved>×N ; each push subtracts 8 bytes
        //
        // For RSP to remain 0 mod 16: (frame_size + 8·N) must be 0 mod 16.
        // If frame_size is 0 mod 16 and N is odd, the pushes leave RSP at
        // 8 mod 16 — violating the ABI.  We pad frame_size by 8 bytes
        // when (frame_size + 8·N) is not a multiple of 16.
        if matches!(ctx.target, Target::X86_64 | Target::I686) {
            let callee_push_bytes = mf.callee_saved_regs.len() * 8;
            let total = mf.frame_size + callee_push_bytes;
            if total % 16 != 0 {
                mf.frame_size += 8;
            }
        }

        // Resolve integer-division register conflicts.
        // After register allocation, the divisor may be assigned to RAX or
        // RDX — which get clobbered by the dividend setup sequence (MOV RAX
        // + CQO/CDQ for signed, XOR RDX + MOV RAX for unsigned).
        if matches!(ctx.target, Target::X86_64) {
            resolve_div_conflicts(&mut mf);
        }

        // Resolve shift-destination / RCX conflicts.
        // After register allocation, the SHL/SHR/SAR destination virtual
        // register may have been assigned to RCX.  Since the shift count
        // is also loaded into RCX (CL), the value-to-shift is clobbered.
        if matches!(ctx.target, Target::X86_64 | Target::I686) {
            resolve_shift_conflicts(&mut mf);
        }

        // Resolve call-argument register move conflicts.
        // After register allocation, argument setup moves like
        //   MOV RSI, <vreg_x>  ; MOV RDX, <vreg_y>
        // may conflict when vreg_y was allocated to RSI (clobbered
        // by the first MOV before the second reads it).  This pass
        // detects and fixes such conflicts using a target-appropriate
        // scratch register (R11 for x86-64, T0 for RISC-V, X16 for
        // AArch64).
        resolve_call_arg_conflicts(&mut mf, &ctx.target);

        // Resolve parameter-loading MOV conflicts at the function entry.
        // After register allocation, parameter MOVs (e.g., MOV RCX, RSI)
        // may clobber ABI registers that are sources for later parameter
        // loads.  This pass applies parallel-move sequentialisation.
        if matches!(
            ctx.target,
            Target::X86_64 | Target::AArch64 | Target::RiscV64
        ) {
            resolve_param_load_conflicts(&mut mf);
        }

        // Step 3: Insert prologue and epilogue instructions.
        let prologue = arch.emit_prologue(&mf);
        let epilogue = arch.emit_epilogue(&mf);
        insert_prologue_epilogue(&mut mf, prologue, epilogue);

        // Step 4: Assembly encoding — machine instructions → raw bytes + relocations
        let asm_result = match arch.emit_assembly(&mf) {
            Ok(asm) => asm,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!(
                        "codegen: assembly encoding failed for '{}': {}",
                        func.name, e
                    ),
                );
                if diagnostics.has_errors() {
                    return Err(format!(
                        "code generation aborted: assembly encoding failed for '{}'",
                        func.name
                    ));
                }
                continue;
            }
        };

        let func_bytes = asm_result.bytes;

        // Track the function's position within the .text section.
        let func_offset = text_section_data.len() as u64;
        let func_size = func_bytes.len() as u64;

        // Collect relocations, adjusting offsets to be relative to the
        // combined .text section rather than this individual function.
        // For RISC-V PCREL local labels (.Lpcrel_hi<N>), also update the
        // label name to use the absolute .text offset so that the symbol
        // table entry created later has the correct value.
        for mut reloc in asm_result.relocations {
            reloc.offset += func_offset;
            // Update PCREL local label references to use absolute offsets
            if reloc.symbol.starts_with(".Lpcrel_hi") {
                if let Some(per_func_str) = reloc.symbol.strip_prefix(".Lpcrel_hi") {
                    if let Ok(per_func_off) = per_func_str.parse::<u64>() {
                        let abs_off = per_func_off + func_offset;
                        reloc.symbol = format!(".Lpcrel_hi{}", abs_off);
                    }
                }
            }
            text_relocations.push(reloc);
        }
        text_section_offsets.push((func.name.clone(), func_offset, func_size));

        // Map IR visibility enum to ELF STV_* constant for symbol table emission.
        let elf_visibility = match func.visibility {
            Visibility::Hidden => STV_HIDDEN,
            Visibility::Protected => STV_PROTECTED,
            Visibility::Default => STV_DEFAULT,
        };

        compiled_functions.push(CompiledFunction {
            name: func.name.clone(),
            bytes: func_bytes.clone(),
            offset: func_offset,
            size: func_size,
            is_global: func.linkage == FunctionLinkage::External
                || func.linkage == FunctionLinkage::Weak,
            elf_binding: match func.linkage {
                FunctionLinkage::Weak => STB_WEAK,
                FunctionLinkage::External => STB_GLOBAL,
                _ => STB_LOCAL,
            },
            visibility: elf_visibility,
        });

        text_section_data.extend_from_slice(&func_bytes);
    }

    // Check for errors after function compilation.
    if diagnostics.has_errors() {
        return Err("code generation aborted due to errors".to_string());
    }

    // -----------------------------------------------------------------------
    // Phase A.4b: Append retpoline thunks to .text (x86-64 only)
    // -----------------------------------------------------------------------
    // When -mretpoline is active, the assembler transforms indirect
    // call/jmp instructions into `CALL __x86_indirect_thunk_<reg>`.
    // The thunk bodies must be physically present in the .text section
    // and registered in the function offset map so that Phase A.5 can
    // resolve the CALL relocations targeting them.
    if ctx.retpoline && ctx.target == Target::X86_64 {
        let thunks = crate::backend::x86_64::assembler::assemble_retpoline_thunks();
        for (thunk_name, thunk_bytes) in &thunks {
            let thunk_offset = text_section_data.len() as u64;
            let thunk_size = thunk_bytes.len() as u64;
            text_section_offsets.push((thunk_name.clone(), thunk_offset, thunk_size));
            compiled_functions.push(CompiledFunction {
                name: thunk_name.clone(),
                bytes: thunk_bytes.clone(),
                offset: thunk_offset,
                size: thunk_size,
                is_global: true,
                elf_binding: STB_GLOBAL,
                visibility: STV_DEFAULT,
            });
            text_section_data.extend_from_slice(thunk_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // Phase A.5: Resolve intra-module text relocations
    // -----------------------------------------------------------------------
    // Build a map from function name → offset in combined .text section.
    // For relocations targeting defined functions, patch the .text bytes
    // directly (PC-relative CALL/JMP).  Remaining relocations (external
    // symbols) are carried forward to the ELF object for the linker.
    //
    // IMPORTANT: The patching format is architecture-dependent.
    // - x86-64 / i686: Write raw 4-byte little-endian PC-relative offset
    //   (R_X86_64_PC32, R_X86_64_PLT32, R_386_PC32 all use 32-bit addend).
    // - AArch64: Encode offset/4 into the instruction's immediate field
    //   while preserving the opcode bits (BL=imm26, B.cond=imm19, etc.).
    // - RISC-V 64: Architecture-specific encoding per relocation type.
    {
        let mut func_offset_map = crate::common::fx_hash::FxHashMap::default();
        for (fname, foff, _fsz) in &text_section_offsets {
            func_offset_map.insert(fname.as_str(), *foff);
        }
        let mut unresolved = Vec::new();
        for reloc in text_relocations.drain(..) {
            // R_386_32 (absolute) relocations must go through the linker,
            // even when targeting a defined function — e.g. computed-goto
            // BlockAddress LEAs encode the block offset as an implicit
            // addend and need the linker to supply the function base.
            let is_absolute_reloc = match ctx.target {
                Target::I686 => reloc.rel_type_id == 1, // R_386_32
                _ => false,
            };
            if !is_absolute_reloc {
                if let Some(&target_offset) = func_offset_map.get(reloc.symbol.as_str()) {
                    // Intra-module relocation — patch text_section_data directly.
                    // Standard ELF PC-relative formula: S + A - P
                    let s = target_offset as i64;
                    let a = reloc.addend;
                    let p = reloc.offset as i64;
                    let value = s + a - p;
                    let fixup_off = reloc.offset as usize;
                    if fixup_off + 4 <= text_section_data.len() {
                        patch_intra_module_reloc(
                            &mut text_section_data,
                            fixup_off,
                            value,
                            reloc.rel_type_id,
                            &ctx.target,
                        );
                    }
                    continue;
                }
            }
            // External symbol or absolute relocation — keep for the linker.
            unresolved.push(reloc);
        }
        text_relocations = unresolved;
    }

    // -----------------------------------------------------------------------
    // Phase B: Global Data Emission
    // -----------------------------------------------------------------------

    let mut data_section: Vec<u8> = Vec::new();
    let mut rodata_section: Vec<u8> = Vec::new();
    let mut bss_size: u64 = 0;
    let mut global_symbols: Vec<GlobalSymbolInfo> = Vec::new();
    // Data-section relocations: (offset_in_section, symbol_name, is_rodata).
    // These are generated for GlobalRef entries in global variable
    // initializers so that the linker can patch symbol addresses.
    // Tuple: (offset_in_section, symbol_name, is_rodata, addend)
    let mut data_relocs: Vec<(u64, String, bool, i64)> = Vec::new();

    for global in module.globals() {
        // Skip non-definition globals (extern declarations with no storage
        // in this translation unit).  They become SHN_UNDEF symbol table
        // entries, NOT BSS allocations.
        if !global.is_definition {
            continue;
        }
        // Record pre-emit offsets to compute relocation positions.
        let data_offset_before = data_section.len() as u64;
        let rodata_offset_before = rodata_section.len() as u64;

        let sym_info = emit_global_variable(
            global,
            &mut data_section,
            &mut rodata_section,
            &mut bss_size,
            ctx,
        );

        // Collect GlobalRef relocations within this global's initializer.
        if let Some(ref init) = global.initializer {
            let mut relocs: Vec<(u64, String, i64)> = Vec::new();
            collect_constant_relocs(init, 0, Some(&global.ty), &mut relocs);
            let is_rodata = matches!(sym_info.section, SectionPlacement::Rodata);
            let section_base = if is_rodata {
                rodata_offset_before
            } else {
                data_offset_before
            };
            for (offset_in_init, sym_name, addend) in relocs {
                data_relocs.push((section_base + offset_in_init, sym_name, is_rodata, addend));
            }
        }

        global_symbols.push(sym_info);
    }

    // -----------------------------------------------------------------------
    // Phase C: String Literal Pool → .rodata
    // -----------------------------------------------------------------------

    let string_pool_offset = rodata_section.len() as u64;
    for string_lit in module.string_pool() {
        let str_offset = rodata_section.len() as u64;
        rodata_section.extend_from_slice(&string_lit.bytes);

        global_symbols.push(GlobalSymbolInfo {
            name: string_lit.label.clone(),
            section: SectionPlacement::Rodata,
            offset: str_offset,
            size: string_lit.bytes.len() as u64,
            is_global: false,
        });
    }
    // Suppress unused-variable warning on string_pool_offset — it's tracked
    // for future use (e.g., relocation generation for string references).
    let _ = string_pool_offset;

    // -----------------------------------------------------------------------
    // Phase D: DWARF Debug Information (conditional on -g)
    // -----------------------------------------------------------------------

    let dwarf_sections = if ctx.should_emit_dwarf() {
        // Use the source_map passed from the frontend, which contains all files
        // registered during preprocessing (file ID 0 = the primary source file).

        // Verify DWARF parameters for this target:
        // - Address size is 4 bytes on 32-bit targets, 8 bytes on 64-bit targets
        // - DWARF version is always v4 for BCC
        let addr_size = dwarf_address_size(&ctx.target);
        debug_assert!(
            (ctx.target.is_64bit() && addr_size == 8) || (!ctx.target.is_64bit() && addr_size == 4),
            "DWARF address size mismatch for target"
        );
        debug_assert_eq!(DWARF_VERSION, 4, "BCC targets DWARF v4");

        let dwarf = generate_dwarf_sections(
            module,
            source_map,
            &ctx.target,
            &text_section_offsets,
            diagnostics,
        );
        Some(dwarf)
    } else {
        // No -g flag: ZERO debug sections in the output.
        None
    };

    // -----------------------------------------------------------------------
    // Phase E: ELF Object Writing
    // -----------------------------------------------------------------------

    let object_bytes = write_relocatable_object(
        &text_section_data,
        &data_section,
        &rodata_section,
        bss_size,
        &compiled_functions,
        &global_symbols,
        &text_relocations,
        &data_relocs,
        &dwarf_sections,
        module,
        ctx,
        diagnostics,
    )?;

    // If -c (compile-only), return the .o bytes directly.
    if ctx.compile_only {
        return Ok(object_bytes);
    }

    // -----------------------------------------------------------------------
    // Phase F: Linking Orchestration
    // -----------------------------------------------------------------------

    link_to_final_output(
        &object_bytes,
        &text_relocations,
        &data_relocs,
        ctx,
        diagnostics,
    )
}

// ===========================================================================
// Internal data structures
// ===========================================================================

/// Information about a compiled function within the .text section.
#[allow(dead_code)]
struct CompiledFunction {
    /// Function symbol name.
    name: String,
    /// Encoded machine code bytes.
    bytes: Vec<u8>,
    /// Byte offset within .text.
    offset: u64,
    /// Size of the function in bytes.
    size: u64,
    /// Whether this function has external/global linkage.
    is_global: bool,
    /// ELF symbol binding (STB_LOCAL, STB_GLOBAL, STB_WEAK).
    /// Tracks weak vs. strong linkage for correct linker resolution.
    elf_binding: u8,
    /// ELF symbol visibility (STV_DEFAULT, STV_HIDDEN, STV_PROTECTED).
    /// Propagated from the IR function's visibility attribute.
    visibility: u8,
}

/// Placement of a global symbol within an ELF section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionPlacement {
    /// `.data` section (initialized, writable).
    Data,
    /// `.rodata` section (initialized, read-only).
    Rodata,
    /// `.bss` section (zero-initialized, writable).
    Bss,
}

/// Information about a global variable or string literal for the symbol table.
struct GlobalSymbolInfo {
    /// Symbol name.
    name: String,
    /// Which section the symbol resides in.
    section: SectionPlacement,
    /// Byte offset within the section.
    offset: u64,
    /// Size of the data in bytes.
    size: u64,
    /// Whether the symbol has external linkage.
    is_global: bool,
}

// ===========================================================================
// insert_spill_code_from_result — record spill metadata on MachineFunction
// ===========================================================================

/// Records spill slot information from the register allocation result
/// directly on the `MachineFunction`.
///
/// Instead of cloning the IR function and modifying it, this function
/// updates the `MachineFunction`'s frame size to account for all spill
/// slots.  Actual spill loads/stores are handled by the architecture's
/// prologue/epilogue emission which reads the frame size.
fn insert_spill_code_from_result(mf: &mut MachineFunction, alloc: &AllocationResult) {
    if alloc.spill_map.is_empty() {
        return;
    }
    // Each spill slot occupies 8 bytes (pointer-width).  Accumulate the
    // total spill area size and update the frame size so prologue/epilogue
    // allocate sufficient stack space.
    let spill_area = alloc.spill_slots.len() * 8;
    mf.frame_size = mf.frame_size.max(alloc.frame_size).max(spill_area);
}

// ===========================================================================
// apply_allocation_result — update MachineFunction with regalloc results
// ===========================================================================

/// Applies register allocation results to a `MachineFunction`.
///
/// Replaces `VirtualRegister` operands with physical `Register` operands
/// based on the allocation mapping.  Updates the function's callee-saved
/// register list with registers that were actually used.
///
/// For **spilled** virtual registers (those assigned a stack slot instead
/// of a physical register), this function inserts machine-level spill
/// loads and stores:
/// - **Before each USE** of a spilled vreg: inserts `MOV scratch, [RBP-offset]`
///   and replaces the operand with the scratch register.
/// - **After each DEF** of a spilled vreg: replaces the result with a scratch
///   register and inserts `MOV [RBP-offset], scratch`.
///
/// The scratch register is chosen dynamically from caller-saved registers
/// that are not otherwise used by the current instruction.
fn apply_allocation_result(mf: &mut MachineFunction, alloc: &AllocationResult, target: &Target) {
    // Record callee-saved registers used by this function.
    mf.callee_saved_regs = alloc.callee_saved_used.iter().map(|pr| pr.0).collect();

    // Update frame size with spill slot requirements.
    mf.frame_size = mf.frame_size.max(alloc.frame_size);

    // Build a set of codegen vreg IDs from vreg_to_ir_value.
    // CRITICAL: This set is used to guard ALL fallback loops below that
    // use val.index() as a vreg number.  Without this guard, val.index()
    // can collide with codegen vreg IDs, causing physical-register vregs
    // to be incorrectly treated as spilled (or vice versa).
    let codegen_vreg_ids: crate::common::fx_hash::FxHashSet<u32> =
        mf.vreg_to_ir_value.keys().copied().collect();

    // Build a complete vreg → physical register lookup table upfront.
    // Primary path: map codegen vregs via vreg_to_ir_value.
    let mut vreg_phys: crate::common::fx_hash::FxHashMap<u32, u16> =
        crate::common::fx_hash::FxHashMap::default();
    for (&vreg, ir_val) in &mf.vreg_to_ir_value {
        if let Some(phys) = alloc.assignments.get(ir_val) {
            vreg_phys.insert(vreg, phys.0);
        }
    }
    // Fallback: for vregs not in vreg_to_ir_value, use val.index().
    // entry().or_insert() ensures codegen vregs already in the map
    // (from the primary loop) are NOT overwritten.
    for (val, phys) in &alloc.assignments {
        let vreg = val.index();
        vreg_phys.entry(vreg).or_insert(phys.0);
    }

    // Build spilled vreg → frame offset mapping.
    //
    // Frame layout conventions differ by architecture:
    //
    // **x86-64 / i686:** FP (RBP) points to the TOP of the frame.  Locals
    //   and spill slots live at *negative* offsets from RBP.
    //
    // **AArch64 / RISC-V 64:** FP points to the BOTTOM of the frame
    //   (FP = SP after the prologue).  The prologue allocates:
    //     total = 16 (FP/LR) + callee_save_bytes + mf.frame_size
    //   **AArch64:** FP/LR are at [FP+0..+15].  Locals occupy
    //   [FP+16 .. FP+16+locals_size).  Callee-saved regs are placed
    //   AFTER locals at [FP+16+locals_size .. +callee_save_bytes).
    //   Spill slots go right after locals (within mf.frame_size), so
    //   they use FP + 16 + existing_locals as their base.
    //
    //   **RISC-V 64:** FP points to old_SP (top of frame).  Locals and
    //   spills are at negative offsets, but the spill base computation
    //   still uses 16 + callee_save_bytes + base_offset as positive.
    let base_offset = mf.frame_size;
    let is_aarch64_or_rv = matches!(target, Target::AArch64 | Target::RiscV64);
    let a64_rv_spill_base: i64 = if is_aarch64_or_rv {
        if matches!(target, Target::AArch64) {
            // AArch64: spills start at FP + 16 + existing locals_size.
            // No callee-saved offset because callee-saved regs are AFTER
            // both locals and spills in the AArch64 frame layout.
            (16 + base_offset) as i64
        } else {
            // RISC-V: keep the original formula.
            let num_gpr = mf
                .callee_saved_regs
                .iter()
                .filter(|&&r| (r as u32) < 32)
                .count();
            let num_fpr = mf
                .callee_saved_regs
                .iter()
                .filter(|&&r| (r as u32) >= 32)
                .count();
            let gpr_pairs = (num_gpr + 1) / 2;
            let fpr_pairs = (num_fpr + 1) / 2;
            let callee_save_bytes = (gpr_pairs + fpr_pairs) * 16;
            (16 + callee_save_bytes + base_offset) as i64
        }
    } else {
        0
    };

    let mut vreg_spill_offset: crate::common::fx_hash::FxHashMap<u32, i64> =
        crate::common::fx_hash::FxHashMap::default();

    // Primary path: map codegen vregs to spill offsets via vreg_to_ir_value.
    for (&vreg, ir_val) in &mf.vreg_to_ir_value {
        if let Some(&slot_idx) = alloc.spill_map.get(ir_val) {
            let offset = if is_aarch64_or_rv {
                // Positive offset from FP for AArch64/RV.
                a64_rv_spill_base + 8 * (slot_idx as i64)
            } else {
                // Negative offset from RBP for x86.
                -((base_offset as i64) + 8 * (slot_idx as i64 + 1))
            };
            vreg_spill_offset.insert(vreg, offset);
        }
    }
    // Fallback path: for IR values whose val.index() is used directly
    // as a vreg in machine instructions (not via vreg_to_ir_value).
    // CRITICAL: Skip entries where val.index() collides with an existing
    // codegen vreg ID — those codegen vregs may refer to completely
    // different IR values and must NOT be marked as spilled.
    for (val, &slot_idx) in &alloc.spill_map {
        let vreg = val.index();
        // Only add if this vreg ID is NOT a known codegen vreg.
        if !codegen_vreg_ids.contains(&vreg) {
            let offset = if is_aarch64_or_rv {
                a64_rv_spill_base + 8 * (slot_idx as i64)
            } else {
                -((base_offset as i64) + 8 * (slot_idx as i64 + 1))
            };
            vreg_spill_offset.entry(vreg).or_insert(offset);
        }
    }

    // Expand frame size to include spill area.
    if !vreg_spill_offset.is_empty() {
        let spill_area = alloc.spill_slots.len() * 8;
        mf.frame_size += spill_area;
    }

    // Architecture-specific constants for spill code.
    //
    // Frame pointer register:
    // - x86-64 / i686: RBP = 5
    // - AArch64: X29 (FP) = 29
    // - RISC-V 64: X8 (s0/fp) = 8
    let frame_ptr_reg: u16 = match target {
        Target::X86_64 | Target::I686 => 5, // RBP
        Target::AArch64 => 29,              // X29 / FP
        Target::RiscV64 => 8,               // X8 / s0 / fp
    };
    // MOV/load opcodes are architecture-dependent:
    // x86-64: X86Opcode::Mov = 0, X86Opcode::Movsd = 86
    // i686:   I686_MOV = 0x100
    // AArch64: LDR_imm = 59 (A64Opcode::LDR_imm enum discriminant)
    // RISC-V 64: LD = 13 (RvOpcode::LD)
    //
    // IMPORTANT: AArch64 opcode values MUST match A64Opcode enum
    // discriminants exactly.  If variants are added to the enum
    // (e.g. MUL, SMULL), all subsequent discriminants shift.
    // Current mapping: LDR_imm=59, STR_imm=72.
    let mov_opcode: u32 = match target {
        Target::I686 => 0x100, // I686_MOV
        Target::AArch64 => 59, // A64Opcode::LDR_imm (64-bit load)
        Target::RiscV64 => 13, // RvOpcode::LD
        _ => 0,                // X86Opcode::Mov
    };
    let movsd_opcode: u32 = match target {
        Target::I686 => 0x100, // i686 uses MOV for all spills (no SSE)
        Target::AArch64 => 59, // AArch64: same LDR_imm for FP spills
        Target::RiscV64 => 13, // RISC-V: LD for FP spills (stored as 64-bit)
        _ => 86,               // X86Opcode::Movsd
    };
    // Store opcodes for spill stores:
    // x86-64 / i686: same opcode as load (MOV is bidirectional via operand layout)
    // AArch64: STR_imm = 72 (A64Opcode::STR_imm enum discriminant)
    // RISC-V 64: SD = 20 (RvOpcode::SD)
    let store_opcode: u32 = match target {
        Target::AArch64 => 72, // A64Opcode::STR_imm
        Target::RiscV64 => 20, // RvOpcode::SD
        _ => mov_opcode,       // x86: same opcode for load/store
    };
    let store_fp_opcode: u32 = match target {
        Target::AArch64 => 72, // AArch64 STR_imm
        Target::RiscV64 => 20, // RISC-V SD
        _ => movsd_opcode,     // x86: MOVSD for float stores
    };
    // Spill scratch register is architecture-dependent:
    // - x86-64: R11 (index 11) — reserved from GPR allocation
    // - i686: ECX (index 1) — caller-saved, used as scratch for spills
    // - AArch64: X16 (IP0, index 16) — scratch register
    // - RISC-V 64: X5 (T0, index 5) — temporary register
    let spill_scratch: u16 = match target {
        Target::I686 => 1,     // ECX
        Target::AArch64 => 16, // X16 / IP0
        Target::RiscV64 => 5,  // X5 / T0
        _ => 11,               // R11 (x86-64)
    };
    let float_spill_scratch: u16 = match target {
        Target::I686 => 1, // ECX (i686 uses x87 stack, no XMM15)
        _ => 31,           // XMM15
    };

    // Secondary integer spill scratch register — used exclusively on
    // load-store architectures (AArch64, RISC-V) where ALU instructions
    // CANNOT use Memory operands.  When two integer operands are both
    // spilled and the primary scratch (X16/X5) is already occupied by
    // the first, the second operand is loaded into this register instead
    // of being left as a Memory operand (which would silently produce
    // wrong code: the generic operand mapper would ignore the Memory
    // base/displacement for ALU ops and the rm field would default to
    // register 0).
    //
    // On x86/i686 this is unused because ALU instructions natively
    // support memory source operands.
    let spill_scratch2: u16 = match target {
        Target::AArch64 => 17, // X17 / IP1
        Target::RiscV64 => 6,  // X6 / T1
        _ => spill_scratch,    // x86: not used, set to same as primary
    };

    // Build a set of codegen vregs that hold float types.
    // CRITICAL: Must use BOTH the vreg_to_ir_value mapping AND direct
    // Value.index() to ensure coverage, since codegen vreg numbering
    // may differ from IR Value indices.
    let mut vreg_is_float: crate::common::fx_hash::FxHashSet<u32> =
        crate::common::fx_hash::FxHashSet::default();
    // Path 1: Map through vreg_to_ir_value (codegen vreg → IR Value → type).
    for (&vreg, ir_val) in &mf.vreg_to_ir_value {
        // Check if this IR value is spilled AND is float.
        if alloc.spill_map.contains_key(ir_val) {
            if let Some(ty) = alloc.value_types.get(ir_val) {
                if ty.is_float() {
                    vreg_is_float.insert(vreg);
                }
            }
        }
        // Check if assigned to an SSE register.
        if let Some(phys) = alloc.assignments.get(ir_val) {
            if phys.0 >= 16 && phys.0 < 32 {
                vreg_is_float.insert(vreg);
            }
        }
    }
    // Path 2: Direct fallback using Value.index() as vreg (for vregs
    // not in vreg_to_ir_value — e.g. the second pass of the mapping).
    // CRITICAL: Skip entries where val.index() collides with a known
    // codegen vreg to prevent marking an integer codegen vreg as float.
    for val in alloc.spill_map.keys() {
        if !codegen_vreg_ids.contains(&val.index()) {
            if let Some(ty) = alloc.value_types.get(val) {
                if ty.is_float() {
                    vreg_is_float.insert(val.index());
                }
            }
        }
    }
    for (val, phys) in &alloc.assignments {
        if !codegen_vreg_ids.contains(&val.index()) && phys.0 >= 16 && phys.0 < 32 {
            vreg_is_float.insert(val.index());
        }
    }

    // Helper: select scratch register and opcode based on float/int class.
    // Helper: select scratch register and LOAD opcode based on float/int class.
    let spill_scratch_for = |is_float: bool| -> (u16, u32) {
        if is_float {
            (float_spill_scratch, movsd_opcode)
        } else {
            (spill_scratch, mov_opcode)
        }
    };
    // Helper: select scratch register and STORE opcode based on float/int class.
    // On x86, load/store use the same opcode (MOV/MOVSD are bidirectional).
    // On RISC-V/AArch64, stores have separate opcodes (SD/STR vs LD/LDR).
    let spill_store_for = |is_float: bool| -> (u16, u32) {
        if is_float {
            (float_spill_scratch, store_fp_opcode)
        } else {
            (spill_scratch, store_opcode)
        }
    };

    // Helper: make a load instruction for spill reload.
    //
    // Architecture-specific layouts:
    // - i686:     result = scratch, operands = [Memory{base=RBP, disp=offset}]
    // - x86-64:   operands = [Register(scratch), Memory{base=RBP, disp=offset}]
    // - AArch64:  result = Register(scratch), operands = [Memory{base=FP, disp=offset}]
    // - RISC-V:   result = Register(scratch), operands = [Memory{base=FP, disp=offset}]
    //             → machine_to_rv_instruction: rd=scratch, rs1=fp, imm=offset → LD scratch, offset(fp)
    let is_i686 = matches!(target, Target::I686);
    let is_risc_or_arm = matches!(target, Target::RiscV64 | Target::AArch64);
    let make_spill_load = |scratch: u16, offset: i64, opcode: u32| -> MachineInstruction {
        let mem = MachineOperand::Memory {
            base: Some(frame_ptr_reg),
            index: None,
            scale: 1,
            displacement: offset,
        };
        if is_i686 || is_risc_or_arm {
            // i686 / RISC-V / AArch64: result = rd, operand = Memory(base+disp)
            let mut inst = MachineInstruction::new(opcode);
            inst.operands.push(mem);
            inst.result = Some(MachineOperand::Register(scratch));
            inst
        } else {
            // x86-64: operands = [Register(scratch), Memory]
            let mut inst = MachineInstruction::new(opcode);
            inst.operands.push(MachineOperand::Register(scratch));
            inst.operands.push(mem);
            inst
        }
    };

    // Helper: make a store instruction for spill save.
    //
    // Architecture-specific layouts:
    // - x86-64/i686: operands = [Memory{base=RBP, disp=offset}, Register(scratch)]
    // - AArch64:     operands = [Memory{base=FP, disp=offset}, Register(scratch)]
    //                → machine_to_a64_instruction: maps to STR scratch, [fp, #offset]
    // - RISC-V:      operands = [Memory{base=FP, disp=offset}, Register(scratch)]
    //                → machine_to_rv_instruction: rs1=fp, rs2=scratch, imm=offset → SD scratch, offset(fp)
    let make_spill_store = |scratch: u16, offset: i64, opcode: u32| -> MachineInstruction {
        let mut inst = MachineInstruction::new(opcode);
        inst.operands.push(MachineOperand::Memory {
            base: Some(frame_ptr_reg),
            index: None,
            scale: 1,
            displacement: offset,
        });
        inst.operands.push(MachineOperand::Register(scratch));
        inst
    };

    // Process each block: replace VRs with physical registers or insert spill code.
    //
    // ## Critical x86 Spill Strategy
    //
    // x86 ALU instructions are **destructive**: `ADD dst, src` computes
    // `dst = dst + src`.  The BCC codegen emits these as:
    //
    //   result = Some(dst_vreg), operands = [src_vreg]
    //
    // The encoder prepends `result` into operands[0], yielding `[dst, src]`.
    // The destination register is therefore BOTH a source (implicit read)
    // and a target (write).
    //
    // When the result vreg is spilled we must:
    //   1. LOAD  R11 ← [RBP + result_offset]   (get the old value — implicit source)
    //   2. Instruction: e.g. ADD R11, <src>      (R11 = old + src)
    //   3. STORE [RBP + result_offset] ← R11    (save new value)
    //
    // Because R11 is occupied by the result, any spilled *explicit operand*
    // must use a **Memory** operand so there is no scratch-register conflict.
    //
    // When the result is NOT spilled (or absent), R11 is free and can be
    // used for the first spilled explicit operand.

    for block in &mut mf.blocks {
        let old_insts = std::mem::take(&mut block.instructions);
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(old_insts.len() + old_insts.len() / 2);

        for mut inst in old_insts {
            // ----------------------------------------------------------
            // Step 1: Determine if the result is spilled.
            // ----------------------------------------------------------
            let result_spill_offset: Option<i64> = inst.result.as_ref().and_then(|r| {
                if let MachineOperand::VirtualRegister(vreg) = r {
                    vreg_spill_offset.get(vreg).copied()
                } else {
                    None
                }
            });

            // Detect if the result vreg is float (needs XMM15 scratch).
            let result_is_float = inst.result.as_ref().map_or(false, |r| {
                if let MachineOperand::VirtualRegister(vreg) = r {
                    vreg_is_float.contains(vreg)
                } else {
                    false
                }
            });

            // ----------------------------------------------------------
            // Step 2: Insert a load for the spilled result BEFORE the
            //         instruction so that the scratch reg holds the old
            //         value. Uses R11 for int, XMM15 for float.
            // ----------------------------------------------------------
            let (result_scratch, result_mov_op) = spill_scratch_for(result_is_float);
            if let Some(offset) = result_spill_offset {
                let mut ld = make_spill_load(result_scratch, offset, result_mov_op);
                // Propagate is_call_arg_setup so the parallel-move
                // resolver includes this instruction in its scope.
                ld.is_call_arg_setup = inst.is_call_arg_setup;
                new_insts.push(ld);
            }

            // ----------------------------------------------------------
            // Step 3: Resolve explicit operands.
            //
            // If the result is spilled (scratch is occupied):
            //   → Spilled operands of the SAME class become Memory.
            //   → Spilled operands of the OTHER class can use their scratch.
            // If the result is NOT spilled:
            //   → The FIRST spilled int operand loads into R11.
            //   → The FIRST spilled float operand loads into XMM15.
            //   → Additional operands of a class become Memory.
            // ----------------------------------------------------------
            // CRITICAL: When the result is spilled, the scratch register is
            // already occupied by Step 2's pre-load of the result value.
            // Mark the appropriate scratch as "used" so that subsequent
            // operand resolution does NOT clobber the result by loading
            // a different value into the same scratch register.
            //
            // Without this, a destructive ALU op like ADD whose result
            // AND operand are both spilled would emit:
            //   LOAD R11, [result_spill]   (Step 2 pre-load)
            //   LOAD R11, [operand_spill]  (CLOBBERS result!)
            //   ADD  R11, R11              (= operand + operand, WRONG!)
            //
            // With this fix, the operand falls through to Memory (x86)
            // or scratch2 (AArch64/RISC-V), producing:
            //   LOAD R11, [result_spill]   (Step 2 pre-load)
            //   ADD  R11, [operand_spill]  (= result + operand, CORRECT!)
            let mut int_scratch_used = result_spill_offset.is_some() && !result_is_float;
            let mut float_scratch_used = result_spill_offset.is_some() && result_is_float;
            // Secondary integer scratch usage tracking — only relevant on
            // load-store architectures (AArch64, RISC-V).
            let mut int_scratch2_used = false;

            // Capture the result vreg (before resolution) so we can detect
            // operands that alias the result — critical for in-place ALU ops
            // (ADD, SUB, etc.) where operand[0] IS the result register.
            let result_vreg: Option<u32> = inst.result.as_ref().and_then(|r| {
                if let MachineOperand::VirtualRegister(v) = r {
                    Some(*v)
                } else {
                    None
                }
            });

            // Propagate is_call_arg_setup to spill load instructions so
            // that the parallel-move resolver sees the full arg-setup
            // sequence even when spill reloads are interspersed.
            let propagate_call_arg_setup = inst.is_call_arg_setup;

            for i in 0..inst.operands.len() {
                if let MachineOperand::VirtualRegister(vreg) = inst.operands[i] {
                    if let Some(&offset) = vreg_spill_offset.get(&vreg) {
                        let op_is_float = vreg_is_float.contains(&vreg);
                        if op_is_float {
                            let same_as_result_f = result_vreg == Some(vreg)
                                && result_spill_offset.is_some()
                                && result_is_float;
                            if same_as_result_f {
                                // Reuse float scratch — already loaded from
                                // the same spill slot in Step 2.
                                let (scratch, _) = spill_scratch_for(true);
                                inst.operands[i] = MachineOperand::Register(scratch);
                            } else if !float_scratch_used {
                                let (scratch, mov_op) = spill_scratch_for(true);
                                let mut ld = make_spill_load(scratch, offset, mov_op);
                                ld.is_call_arg_setup = propagate_call_arg_setup;
                                new_insts.push(ld);
                                inst.operands[i] = MachineOperand::Register(scratch);
                                float_scratch_used = true;
                            } else {
                                // Float scratch occupied — use memory operand.
                                // On x86/i686, SSE ALU instructions support
                                // memory source operands directly.  On AArch64
                                // / RISC-V, FP ALU instructions also cannot use
                                // Memory operands, but two float spills in one
                                // instruction is extremely rare (would require a
                                // dedicated second FP scratch register).
                                inst.operands[i] = MachineOperand::Memory {
                                    base: Some(frame_ptr_reg),
                                    index: None,
                                    scale: 1,
                                    displacement: offset,
                                };
                            }
                        } else {
                            // Check if this operand is the SAME spilled vreg
                            // as the result. This is critical for in-place ALU
                            // ops (ADD, SUB, etc.) where operand[0] IS the
                            // destination: if we used Memory for operand[0]
                            // while result maps to scratch, Step 5's store of
                            // scratch would overwrite the correct in-memory
                            // result with a stale value.
                            let same_as_result = result_vreg == Some(vreg)
                                && result_spill_offset.is_some()
                                && !result_is_float;
                            if same_as_result {
                                // Reuse scratch — already loaded from the same
                                // spill slot in Step 2.  The scratch is already
                                // marked as used (initialized true when result
                                // is spilled), so no state change needed.
                                inst.operands[i] = MachineOperand::Register(spill_scratch);
                            } else if !int_scratch_used {
                                // Scratch is available. Load operand's spill
                                // value into scratch (may overwrite Step 2's
                                // preload of the result — that is safe because
                                // both share the scratch register, and the
                                // instruction will produce the correct value
                                // which Step 5 stores back).
                                let mut ld = make_spill_load(spill_scratch, offset, mov_opcode);
                                ld.is_call_arg_setup = propagate_call_arg_setup;
                                new_insts.push(ld);
                                inst.operands[i] = MachineOperand::Register(spill_scratch);
                                int_scratch_used = true;
                            } else if is_risc_or_arm && !int_scratch2_used {
                                // Load-store architectures: ALU instructions
                                // CANNOT use Memory operands.  Use the secondary
                                // scratch register (X17/IP1 on AArch64, X6/T1 on
                                // RISC-V) to load the second spilled operand.
                                let mut ld = make_spill_load(spill_scratch2, offset, mov_opcode);
                                ld.is_call_arg_setup = propagate_call_arg_setup;
                                new_insts.push(ld);
                                inst.operands[i] = MachineOperand::Register(spill_scratch2);
                                int_scratch2_used = true;
                            } else {
                                // x86/i686: ALU instructions natively support
                                // Memory source operands — emit directly.
                                // (Also fallback if both scratch regs are used,
                                // which is extremely rare for binary ops.)
                                inst.operands[i] = MachineOperand::Memory {
                                    base: Some(frame_ptr_reg),
                                    index: None,
                                    scale: 1,
                                    displacement: offset,
                                };
                            }
                        }
                    } else if let Some(&phys) = vreg_phys.get(&vreg) {
                        inst.operands[i] = MachineOperand::Register(phys);
                    }
                }
            }

            // ----------------------------------------------------------
            // Step 4: Resolve the result operand.
            // ----------------------------------------------------------
            let mut store_after: Option<(i64, bool)> = None;
            if let Some(ref mut result) = inst.result {
                if let MachineOperand::VirtualRegister(vreg) = *result {
                    if let Some(&offset) = vreg_spill_offset.get(&vreg) {
                        let is_float = vreg_is_float.contains(&vreg);
                        let (scratch, _) = spill_scratch_for(is_float);
                        *result = MachineOperand::Register(scratch);
                        store_after = Some((offset, is_float));
                    } else if let Some(&phys) = vreg_phys.get(&vreg) {
                        *result = MachineOperand::Register(phys);
                    }
                }
            }

            // Push the (possibly rewritten) instruction.
            new_insts.push(inst);

            // ----------------------------------------------------------
            // Step 5: Insert a store for the spilled result AFTER the
            //         instruction. Uses store opcode (SD for RISC-V,
            //         STR for AArch64, MOV for x86).
            // ----------------------------------------------------------
            if let Some((offset, is_float)) = store_after {
                let (scratch, st_op) = spill_store_for(is_float);
                new_insts.push(make_spill_store(scratch, offset, st_op));
            }
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// resolve_param_load_conflicts — fix register clobbering in function prologue
// ===========================================================================

/// After register allocation, function-entry parameter-loading MOV sequences
/// like:
///
/// ```text
///   MOV RAX, RDI       ; param 0 → allocated to RAX (OK)
///   MOV RCX, RSI       ; param 1 → allocated to RCX (CLOBBERS param 3 source!)
///   MOV RDX, RDX       ; param 2 → allocated to RDX (no-op)
///   MOV RSI, RCX       ; param 3 → allocated to RSI (BUG: RCX already overwritten!)
/// ```
///
/// may contain conflicts: a destination register of an earlier MOV is the
/// source register of a later MOV.  This is the classical **parallel copy
/// problem** — all parameter loads must logically happen simultaneously.
///
/// This function applies a standard parallel-move sequentialisation algorithm:
///
/// 1. Remove self-moves (`MOV R, R`) — they are no-ops.
/// 2. Topologically order remaining moves: emit moves whose destination is
///    NOT a source for any pending move first.
/// 3. If a cycle is detected (no move can be emitted safely), break it by
///    saving one source to R11 (scratch), substituting R11 as the source,
///    and continuing.
///
/// The resolved sequence of MOVs is then spliced back in place of the
/// original parameter-loading sequence.
fn resolve_param_load_conflicts(mf: &mut MachineFunction) {
    if mf.blocks.is_empty() {
        return;
    }

    const MOV_OPCODE: u32 = 0; // X86Opcode::Mov
    const MOVSD_OPCODE: u32 = 86; // X86Opcode::Movsd
    const SCRATCH_REG: u16 = 11; // R11 (integer scratch)
    const SCRATCH_REG2: u16 = 10; // R10 (secondary integer scratch)
    const FLOAT_SCRATCH_REG: u16 = 31; // XMM15 (float scratch)

    // ABI integer argument registers (System V AMD64).
    let abi_int_regs: crate::common::fx_hash::FxHashSet<u16> = {
        let mut s = crate::common::fx_hash::FxHashSet::default();
        // RDI=7, RSI=6, RDX=2, RCX=1, R8=8, R9=9
        s.insert(7);
        s.insert(6);
        s.insert(2);
        s.insert(1);
        s.insert(8);
        s.insert(9);
        s
    };
    // ABI SSE argument registers.
    let abi_sse_regs: crate::common::fx_hash::FxHashSet<u16> = {
        let mut s = crate::common::fx_hash::FxHashSet::default();
        for i in 16u16..24 {
            // XMM0=16 .. XMM7=23
            s.insert(i);
        }
        s
    };

    let max_param_moves = mf.num_param_moves;
    if max_param_moves <= 1 {
        return;
    }
    // (Debug: set BCC_DEBUG_PARAM=1 to trace parameter move resolution.)

    // ----------------------------------------------------------------
    // Phase 1: Parse the entry block to extract logical parameter moves.
    //
    // After apply_allocation_result, parameter MOVs come in two forms:
    //
    // A) Register-to-register (non-spilled parameter):
    //    [inst]  MOV dst_reg, abi_reg
    //
    // B) Spill-expanded (spilled parameter) — 3 instructions:
    //    [inst]  MOV R11, [spill_slot]     ; pre-load scratch from result slot
    //    [inst]  MOV R11, abi_reg          ; actual parameter copy via scratch
    //    [inst]  MOV [spill_slot], R11     ; write-back scratch to spill slot
    //
    // We parse up to `num_param_moves` logical param moves from the entry
    // block, recognizing both formats.
    // ----------------------------------------------------------------

    /// A logical parameter move extracted from the entry block.
    #[derive(Clone)]
    enum ParamMove {
        /// Non-spilled: single MOV reg, abi_reg
        Register {
            dst: u16,
            src: u16,
            opcode: u32,
            _inst_idx: usize,
        },
        /// Spilled: 3-instruction sequence
        Spill {
            _src: u16,
            inst_start: usize, // index of first instruction (pre-load)
            inst_count: usize, // always 3
        },
    }

    let entry = &mf.blocks[0];
    let insts = &entry.instructions;
    let mut logical_moves: Vec<ParamMove> = Vec::new();
    let mut inst_cursor: usize = 0;
    // Pass-through instructions: struct-pair frame stores and memory
    // loads that appear in the param region but don't participate in
    // the parallel-move register conflict.  They are emitted first
    // (before resolved register moves) in the output.
    let mut pass_through_insts: Vec<MachineInstruction> = Vec::new();

    while logical_moves.len() < max_param_moves && inst_cursor < insts.len() {
        let inst = &insts[inst_cursor];
        if inst.opcode != MOV_OPCODE && inst.opcode != MOVSD_OPCODE {
            break;
        }

        // Check for struct-pair frame store: MOV [mem], abi_reg
        //
        // When a 16-byte struct parameter is saved to a stack alloca,
        // the codegen emits stores like `MOV [RBP-16], RDI` and
        // `MOV [RBP-8], RSI` at the very beginning of the entry block,
        // BEFORE any register-to-register param MOVs.  These read ABI
        // registers and write to memory — they are safe pass-through
        // instructions that must be emitted first and don't participate
        // in the parallel-move register conflict resolution.
        //
        // Without this recognition, the scanner would break on the
        // first frame store (unrecognized format) and never reach the
        // register-to-register param MOVs that follow, leaving their
        // cycles unresolved (e.g. MOV RCX,RDX; MOV RDX,RCX).
        let is_frame_store = match (&inst.result, inst.operands.as_slice()) {
            (None, [MachineOperand::Memory { .. }, MachineOperand::Register(src)]) => {
                abi_int_regs.contains(src) || abi_sse_regs.contains(src)
            }
            _ => false,
        };
        if is_frame_store {
            pass_through_insts.push(inst.clone());
            inst_cursor += 1;
            continue;
        }

        // Check for scratch-register store: MOV [mem], R11/R10
        //
        // When a multi-eightbyte struct parameter is passed on the stack
        // (all 6 integer ABI registers already consumed), the callee
        // copies each eightbyte from the caller's frame to a local alloca
        // using R11 as a scratch register:
        //   MOV R11, [RBP+16]    (load from caller stack)
        //   MOV [RBP-16], R11    (store to local alloca)
        //   MOV R11, [RBP+24]    (load from caller stack)
        //   MOV [RBP-8], R11     (store to local alloca)
        //
        // The load instructions are recognized as is_mem_load (above).
        // The store instructions use R11 (scratch) as the source, which
        // is NOT an ABI register, so is_frame_store does not match.
        // Without this check, the scanner breaks on the store, never
        // reaching the register-to-register param MOVs that follow,
        // leaving their cycles (e.g. RSI↔RCX) unresolved.
        //
        // These stores are safe pass-through instructions: they write to
        // memory (no register conflict) from a scratch register that is
        // not a live incoming parameter value.
        let is_scratch_store = match (&inst.result, inst.operands.as_slice()) {
            (None, [MachineOperand::Memory { .. }, MachineOperand::Register(src)]) => {
                *src == SCRATCH_REG || *src == SCRATCH_REG2
            }
            _ => false,
        };
        if is_scratch_store {
            pass_through_insts.push(inst.clone());
            inst_cursor += 1;
            continue;
        }

        // Check for memory-to-register load: MOV reg, [mem]
        //
        // These include struct-pair lo-half reloads (e.g., MOV RAX, [RBP-16])
        // and stack parameter loads (e.g., MOV R8, [RBP+16]).
        //
        // IMPORTANT: Memory loads CANNOT be treated as simple pass-through
        // instructions.  Their destination register may be an ABI register
        // (e.g., R8, R9) that is also a source for a register-to-register
        // param move.  If the memory load executes first, it overwrites the
        // ABI register's incoming value before the register move can save it.
        //
        // Instead, include memory loads in the parallel-move resolution as
        // moves with a virtual "memory" source (u16::MAX) that does not
        // conflict with any register source.  The topological sort will
        // naturally defer memory loads whose destination register is still
        // needed as a source by a pending register move.
        let is_mem_load = matches!(
            (&inst.result, inst.operands.as_slice()),
            (
                Some(MachineOperand::Register(_)),
                [MachineOperand::Memory { .. }]
            )
        );
        if is_mem_load {
            if let Some(MachineOperand::Register(dst)) = &inst.result {
                // Loads into scratch registers (R11/R10) MAY be part of
                // multi-eightbyte struct copy sequences (load from caller
                // stack → store to local alloca).  However, R10 is also
                // allocatable by the register allocator, so a memory load
                // into R10 might be a legitimate stack parameter load
                // (register allocator assigned R10 to hold the param).
                //
                // Distinction: struct copy sequences always have a matching
                // scratch store immediately following (MOV [mem], R10/R11).
                // Legitimate param loads do NOT have a following store.
                //
                // R11 is fully reserved (never assigned by register alloc),
                // so R11 loads are ALWAYS struct copy sequences.
                // R10 loads require a lookahead check.
                let is_struct_copy_load = if *dst == SCRATCH_REG {
                    // R11 is reserved — always a scratch struct copy load
                    true
                } else if *dst == SCRATCH_REG2 {
                    // R10 is allocatable — check if next instruction is a
                    // matching scratch store (MOV [mem], R10)
                    let next_idx = inst_cursor + 1;
                    if next_idx < insts.len() {
                        let next = &insts[next_idx];
                        matches!(
                            (&next.result, next.operands.as_slice()),
                            (None, [MachineOperand::Memory { .. }, MachineOperand::Register(s)])
                            if *s == SCRATCH_REG2
                        )
                    } else {
                        false
                    }
                } else {
                    false
                };
                if is_struct_copy_load {
                    pass_through_insts.push(inst.clone());
                    inst_cursor += 1;
                    continue;
                }
                logical_moves.push(ParamMove::Register {
                    dst: *dst,
                    src: u16::MAX, // sentinel: memory source, no register conflict
                    opcode: inst.opcode,
                    _inst_idx: inst_cursor,
                });
            }
            inst_cursor += 1;
            continue;
        }

        // Check for spill pattern: pre-load scratch from spill slot.
        // On x86-64: result=None, operands=[Register(R11), Memory{..}]
        // On i686/arm/riscv: result=Some(Register(scratch)), operands=[Memory{..}]
        let is_spill_preload = match (&inst.result, inst.operands.as_slice()) {
            // x86-64 format: result=None, operands=[Register(scratch), Memory]
            (None, [MachineOperand::Register(dst), MachineOperand::Memory { .. }]) => {
                *dst == SCRATCH_REG || *dst == FLOAT_SCRATCH_REG
            }
            // i686/AArch64/RISC-V format: result=Some(Register(scratch)), operands=[Memory]
            (Some(MachineOperand::Register(dst)), [MachineOperand::Memory { .. }]) => {
                *dst == SCRATCH_REG || *dst == FLOAT_SCRATCH_REG
            }
            _ => false,
        };

        if is_spill_preload && inst_cursor + 2 < insts.len() {
            // Verify the next two instructions complete the spill pattern:
            // inst+1: MOV R11, abi_reg (actual move)
            // inst+2: MOV [mem], R11   (write-back)
            let inst1 = &insts[inst_cursor + 1];
            let inst2 = &insts[inst_cursor + 2];

            let abi_src = match (&inst1.result, inst1.operands.first()) {
                (Some(MachineOperand::Register(d)), Some(MachineOperand::Register(s)))
                    if (*d == SCRATCH_REG || *d == FLOAT_SCRATCH_REG)
                        && (abi_int_regs.contains(s) || abi_sse_regs.contains(s)) =>
                {
                    Some(*s)
                }
                _ => None,
            };

            let is_writeback = matches!(
                inst2.operands.as_slice(),
                [MachineOperand::Memory { .. }, MachineOperand::Register(s)]
                    if *s == SCRATCH_REG || *s == FLOAT_SCRATCH_REG
            );

            if let (Some(src), true) = (abi_src, is_writeback) {
                logical_moves.push(ParamMove::Spill {
                    _src: src,
                    inst_start: inst_cursor,
                    inst_count: 3,
                });
                inst_cursor += 3;
                continue;
            }
        }

        // Check for register-to-register param MOV: MOV dst_reg, abi_reg
        match (&inst.result, inst.operands.first()) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)))
                if (abi_int_regs.contains(src) || abi_sse_regs.contains(src))
                    && *dst != SCRATCH_REG =>
            {
                logical_moves.push(ParamMove::Register {
                    dst: *dst,
                    src: *src,
                    opcode: inst.opcode,
                    _inst_idx: inst_cursor,
                });
                inst_cursor += 1;
            }
            _ => break, // unrecognized instruction — stop scanning
        }
    }

    // Total instruction count consumed by the param region.
    let total_param_insts = inst_cursor;

    if logical_moves.len() <= 1 {
        return;
    }

    // ----------------------------------------------------------------
    // Phase 2: Detect conflicts and resolve using parallel-move
    //          sequentialisation.
    //
    // Strategy: spill-destination moves (type B) write only to memory
    //   via R11 — their "destination" cannot conflict with any register
    //   source. However, their ABI source register CAN be clobbered by
    //   a register-destination move (type A) that executes earlier.
    //
    // Solution: emit all spill-destination moves FIRST (they read ABI
    //   registers before anything writes to them), then emit register-
    //   destination moves in parallel-move-resolved order.
    // ----------------------------------------------------------------

    // Separate spill moves from register moves.
    let mut spill_moves: Vec<ParamMove> = Vec::new();
    let mut reg_moves: Vec<(u16, u16, u32)> = Vec::new(); // (dst, src, opcode)
    for lm in &logical_moves {
        match lm {
            ParamMove::Spill { .. } => spill_moves.push(lm.clone()),
            ParamMove::Register {
                dst, src, opcode, ..
            } => reg_moves.push((*dst, *src, *opcode)),
        }
    }

    // Build the resolved instruction sequence.
    let mut resolved: Vec<MachineInstruction> = Vec::with_capacity(total_param_insts + 4);

    // Emit pass-through instructions first — frame stores and memory
    // loads that were identified during scanning.  Frame stores write
    // to memory from ABI registers (must come before any register moves
    // that might clobber those ABI registers).  Memory loads read from
    // memory into non-ABI registers (safe after frame stores).
    for pt in &pass_through_insts {
        resolved.push(pt.clone());
    }

    // Emit all spill moves next — they read ABI registers before any
    // register move can clobber them.
    for sm in &spill_moves {
        if let ParamMove::Spill {
            inst_start,
            inst_count,
            ..
        } = sm
        {
            for inst in insts.iter().skip(*inst_start).take(*inst_count) {
                resolved.push(inst.clone());
            }
        }
    }

    // Apply parallel-move sequentialisation on register moves.
    //
    // Moves with src == u16::MAX are memory-source loads (stack param
    // loads, struct-pair lo reloads).  They have no register source
    // conflict, but their destination register may be needed as a source
    // by another register move, so they participate in the topological
    // ordering.  When emitting them, we reconstruct the original memory
    // load instruction from `insts` rather than a register-to-register
    // MOV.
    //
    // We also build an index map: for each (dst, src=u16::MAX) entry,
    // we store the original instruction index so we can retrieve the
    // memory operand.
    let mut mem_load_insts: crate::common::fx_hash::FxHashMap<u16, MachineInstruction> =
        crate::common::fx_hash::FxHashMap::default();
    for lm in &logical_moves {
        if let ParamMove::Register {
            dst,
            src,
            _inst_idx,
            ..
        } = lm
        {
            if *src == u16::MAX {
                // Memory-source load — save the original instruction.
                mem_load_insts.insert(*dst, insts[*_inst_idx].clone());
            }
        }
    }

    let mut pending = reg_moves.clone();
    // Remove self-moves upfront (never applies to mem loads since src=u16::MAX != dst).
    pending.retain(|&(dst, src, _)| dst != src);

    let mut iteration_limit = pending.len() * pending.len() + 10;
    while !pending.is_empty() && iteration_limit > 0 {
        iteration_limit -= 1;

        // Collect needed source registers.  Memory-source loads (u16::MAX)
        // are excluded — they don't occupy a physical register that could
        // be clobbered by another move's destination.
        let needed_srcs: crate::common::fx_hash::FxHashSet<u16> = pending
            .iter()
            .filter_map(
                |&(_, src, _)| {
                    if src != u16::MAX {
                        Some(src)
                    } else {
                        None
                    }
                },
            )
            .collect();

        let safe_idx = pending
            .iter()
            .position(|&(dst, _, _)| !needed_srcs.contains(&dst));

        if let Some(idx) = safe_idx {
            let (dst, src, opcode) = pending.remove(idx);
            if src == u16::MAX {
                // Memory-source load — emit the original instruction.
                if let Some(orig_inst) = mem_load_insts.get(&dst) {
                    resolved.push(orig_inst.clone());
                }
            } else {
                let mut inst = MachineInstruction::new(opcode);
                inst.result = Some(MachineOperand::Register(dst));
                inst.operands.push(MachineOperand::Register(src));
                resolved.push(inst);
            }
        } else {
            // Cycle detected — break it by saving one source to scratch.
            //
            // Memory-source loads (src=u16::MAX) can never form a cycle
            // (they have no register source), so the cycle must involve
            // register-to-register moves.  Find the first non-mem-load
            // to break.
            let cycle_idx = pending
                .iter()
                .position(|&(_, s, _)| s != u16::MAX)
                .unwrap_or(0);
            let (_, cycle_src, cycle_opcode) = pending[cycle_idx];
            let is_float_cycle = cycle_opcode == MOVSD_OPCODE || (16..32).contains(&cycle_src);
            // Choose a scratch register that is NOT the destination of
            // any pending or original move.  This prevents clobbering a
            // register that was already assigned (e.g., R10 holding x7
            // from a memory load that was resolved earlier).
            let all_dsts: crate::common::fx_hash::FxHashSet<u16> = reg_moves
                .iter()
                .map(|&(dst, _, _)| dst)
                .chain(pending.iter().map(|&(dst, _, _)| dst))
                .collect();
            let (scratch, save_op) = if is_float_cycle {
                (FLOAT_SCRATCH_REG, MOVSD_OPCODE)
            } else {
                // Try R10, R11, then other non-ABI registers.
                let candidates: &[u16] = &[SCRATCH_REG2, SCRATCH_REG, 3, 5]; // R10, R11, RBX, RBP
                let chosen = candidates
                    .iter()
                    .find(|&&r| !all_dsts.contains(&r))
                    .copied()
                    .unwrap_or(SCRATCH_REG2);
                (chosen, MOV_OPCODE)
            };

            let mut save_inst = MachineInstruction::new(save_op);
            save_inst.result = Some(MachineOperand::Register(scratch));
            save_inst.operands.push(MachineOperand::Register(cycle_src));
            resolved.push(save_inst);

            for m in &mut pending {
                if m.1 == cycle_src {
                    m.1 = scratch;
                }
            }
        }
    }

    // Replace the original param region in the entry block.
    let entry = &mut mf.blocks[0];
    let remaining: Vec<MachineInstruction> =
        entry.instructions.drain(total_param_insts..).collect();
    entry.instructions.clear();
    entry.instructions.extend(resolved);
    entry.instructions.extend(remaining);
}

// ===========================================================================
// resolve_div_conflicts — fix divisor-register clobbering in IDIV/DIV
// ===========================================================================

/// After register allocation, integer division sequences may have the divisor
/// assigned to RAX or RDX — registers that are implicitly overwritten by the
/// dividend setup (`MOV RAX, lhs; CQO/CDQ`) or zero-extension (`XOR RDX, RDX`).
///
/// For signed division (SDiv/SRem), the codegen emits:
/// ```text
///   MOV RAX, <lhs>       ← writes RAX  (clobbers divisor if in RAX)
///   CQO / CDQ            ← writes RDX  (clobbers divisor if in RDX)
///   IDIV <divisor>
/// ```
///
/// For unsigned division (UDiv/URem), the codegen emits:
/// ```text
///   XOR RDX, RDX         ← writes RDX  (clobbers LHS/divisor if in RDX)
///   MOV RAX, <lhs>       ← writes RAX  (clobbers divisor if in RAX)
///   DIV <divisor>
/// ```
///
/// This post-allocation pass detects these conflicts and saves the clobbered
/// operand to a scratch register before it is overwritten.
fn resolve_div_conflicts(mf: &mut MachineFunction) {
    const MOV_OP: u32 = 0; // X86Opcode::Mov
    const IDIV_OP: u32 = 10; // X86Opcode::Idiv
    const DIV_OP: u32 = 11; // X86Opcode::Div
    const XOR_OP: u32 = 17; // X86Opcode::Xor
    const CDQ_OP: u32 = 106; // X86Opcode::Cdq
    const CQO_OP: u32 = 107; // X86Opcode::Cqo
    const RAX: u16 = 0;
    const RCX: u16 = 1;
    const RDX: u16 = 2;
    const R11: u16 = 11;

    for block in &mut mf.blocks {
        // We may insert instructions, so iterate by building a new vector.
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(block.instructions.len() + 8);
        let old_insts = std::mem::take(&mut block.instructions);
        let len = old_insts.len();

        let mut i = 0;
        while i < len {
            let opc = old_insts[i].opcode;

            if opc != IDIV_OP && opc != DIV_OP {
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Found IDIV or DIV at position `i`.
            // Divisor is in operands[0].
            let divisor_reg = match old_insts[i].operands.first() {
                Some(MachineOperand::Register(r)) => *r,
                _ => {
                    // Not a register operand (memory, etc.) — no conflict.
                    new_insts.push(old_insts[i].clone());
                    i += 1;
                    continue;
                }
            };

            // Only a problem if divisor is in RAX or RDX.
            if divisor_reg != RAX && divisor_reg != RDX {
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Scan backwards in new_insts to find the "MOV RAX, <lhs>" and
            // optionally CQO/CDQ (for signed) or XOR RDX, RDX (for unsigned).
            // We need to identify the start of the div setup sequence and find
            // the lhs register to avoid scratch conflicts.

            // The division setup sequence (already in new_insts) looks like:
            //   [... MOV RAX, <lhs> ; CQO/CDQ]     for signed  (IDIV)
            //   [... XOR RDX, RDX ; MOV RAX, <lhs>] for unsigned (DIV)
            //
            // We'll look at the last 2-3 instructions in new_insts.

            let ni_len = new_insts.len();

            // Find the MOV RAX, <lhs> instruction (the one that writes RAX).
            let mut mov_rax_pos = None;
            let mut lhs_reg: Option<u16> = None;
            // Search backwards up to 3 instructions.
            let search_start = ni_len.saturating_sub(3);
            for j in (search_start..ni_len).rev() {
                if new_insts[j].opcode == MOV_OP {
                    if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                        if *r == RAX {
                            mov_rax_pos = Some(j);
                            // Get the source register.
                            if let Some(MachineOperand::Register(src)) =
                                new_insts[j].operands.first()
                            {
                                lhs_reg = Some(*src);
                            }
                            break;
                        }
                    }
                }
            }

            // Also check for XOR RDX, RDX or CQO/CDQ in the sequence.
            let mut has_rdx_clobber = false;
            for j in (search_start..ni_len).rev() {
                let op = new_insts[j].opcode;
                if op == CQO_OP || op == CDQ_OP {
                    has_rdx_clobber = true;
                    break;
                }
                if op == XOR_OP {
                    if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                        if *r == RDX {
                            has_rdx_clobber = true;
                            break;
                        }
                    }
                }
            }

            // Determine conflicts:
            // - Divisor in RAX: clobbered by "MOV RAX, lhs"
            // - Divisor in RDX: clobbered by CQO/CDQ or XOR RDX, RDX
            // Also check if LHS might be clobbered (for unsigned: XOR RDX, RDX before MOV RAX):
            //   If lhs was in RDX and unsigned DIV: XOR RDX,RDX clobbers lhs before MOV RAX reads it.

            let div_in_rax = divisor_reg == RAX;
            let div_in_rdx = divisor_reg == RDX && has_rdx_clobber;

            if !div_in_rax && !div_in_rdx {
                // No conflict after all.
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Choose a scratch register: not RAX, not RDX, not lhs_reg.
            let scratch = if lhs_reg != Some(R11) { R11 } else { RCX };

            // Also handle lhs-in-RDX conflict for unsigned division:
            // If opc == DIV_OP and lhs_reg == Some(RDX), the XOR RDX,RDX clobbers lhs.
            let lhs_in_rdx = opc == DIV_OP && lhs_reg == Some(RDX);

            if let Some(insert_pos) = mov_rax_pos {
                // For unsigned DIV with lhs in RDX:
                //   Before: XOR RDX,RDX ; MOV RAX,RDX ; DIV <rhs>
                //   We need to save RDX before the XOR too.
                if lhs_in_rdx {
                    // Find XOR RDX,RDX before insert_pos.
                    let mut xor_pos = None;
                    for j in (search_start..insert_pos).rev() {
                        if new_insts[j].opcode == XOR_OP {
                            if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                                if *r == RDX {
                                    xor_pos = Some(j);
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(xp) = xor_pos {
                        // Pick a second scratch for lhs (different from scratch if
                        // scratch is also needed for divisor).
                        let lhs_scratch = if div_in_rax || div_in_rdx {
                            // Need scratch for divisor, use another for lhs.
                            if scratch != RCX && lhs_reg != Some(RCX) {
                                RCX
                            } else if scratch != R11 && lhs_reg != Some(R11) {
                                R11
                            } else {
                                10 /* R10 */
                            }
                        } else {
                            scratch
                        };
                        // Insert: MOV lhs_scratch, RDX before XOR RDX,RDX
                        let mut save_lhs = MachineInstruction::new(MOV_OP);
                        save_lhs.result = Some(MachineOperand::Register(lhs_scratch));
                        save_lhs.operands.push(MachineOperand::Register(RDX));
                        new_insts.insert(xp, save_lhs);
                        // Update references after insertion.
                        // Find new mov_rax_pos (shifted by 1).
                        let new_mov_rax = insert_pos + 1;
                        // Change MOV RAX, RDX to MOV RAX, lhs_scratch.
                        if let Some(MachineOperand::Register(ref mut src)) =
                            new_insts[new_mov_rax].operands.first_mut()
                        {
                            if *src == RDX {
                                *src = lhs_scratch;
                            }
                        }
                    }
                }

                if div_in_rax || div_in_rdx {
                    // Re-find mov_rax_pos after potential insertions above.
                    let cur_ni_len = new_insts.len();
                    let search_start2 = cur_ni_len.saturating_sub(5);
                    let mut real_insert_pos = None;
                    for j in (search_start2..cur_ni_len).rev() {
                        if new_insts[j].opcode == MOV_OP {
                            if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                                if *r == RAX {
                                    real_insert_pos = Some(j);
                                    break;
                                }
                            }
                        }
                    }
                    // For unsigned DIV with XOR RDX,RDX, if divisor is in RDX,
                    // insert BEFORE the XOR.
                    if div_in_rdx && opc == DIV_OP {
                        let cur_ni_len2 = new_insts.len();
                        let ss = cur_ni_len2.saturating_sub(5);
                        for j in (ss..cur_ni_len2).rev() {
                            if new_insts[j].opcode == XOR_OP {
                                if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                                    if *r == RDX {
                                        real_insert_pos = Some(j);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // For signed IDIV with CQO/CDQ, if divisor is in RDX,
                    // insert BEFORE the MOV RAX (which precedes CQO).
                    if div_in_rdx && opc == IDIV_OP {
                        // real_insert_pos is already the MOV RAX position.
                    }

                    if let Some(rip) = real_insert_pos {
                        // Insert: MOV scratch, divisor_reg BEFORE the clobbering instruction
                        let mut save_div = MachineInstruction::new(MOV_OP);
                        save_div.result = Some(MachineOperand::Register(scratch));
                        save_div
                            .operands
                            .push(MachineOperand::Register(divisor_reg));
                        new_insts.insert(rip, save_div);
                    }

                    // Now update the IDIV/DIV instruction to use scratch instead
                    // of the original divisor register.
                    // The IDIV/DIV hasn't been pushed yet.
                    let mut div_inst = old_insts[i].clone();
                    div_inst.operands[0] = MachineOperand::Register(scratch);
                    new_insts.push(div_inst);
                    i += 1;
                    continue;
                }
            }

            // Fallback: push the original instruction unchanged.
            new_insts.push(old_insts[i].clone());
            i += 1;
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// resolve_shift_conflicts — fix shift-destination / RCX clobbering
// ===========================================================================

/// On x86(-64), variable-count shifts (SHL, SHR, SAR) require the shift
/// amount in CL (the low byte of RCX).  The instruction-selection phase
/// emits:
///
/// ```text
///   MOV  dst, lhs        ; value to shift
///   MOV  RCX, rhs        ; shift amount → CL
///   SHL  dst, CL
/// ```
///
/// **Problem 1 (CASE 1):** If the register allocator assigns `dst = RCX`,
/// the `MOV RCX, rhs` clobbers the lhs.  Spill-code insertion may also
/// interleave instructions between the setup MOVs and the shift, so we
/// cannot assume the three instructions are contiguous.
///
/// **Problem 2 (CASE 2):** If `dst ≠ RCX` but the allocator has placed a
/// live value in RCX, the `MOV RCX, rhs` clobbers it.
///
/// **Robust fix for both cases:**
///
/// *CASE 1 (dst = RCX):* Search backward from the shift for the
/// `MOV RCX, rhs` (shift-amount setup).  Insert `PUSH RCX` before it
/// (saving the lhs that's currently in RCX), then after the `MOV RCX,
/// rhs`, pop the saved lhs into a scratch register, execute the shift
/// on the scratch, and move the result back to RCX:
///
/// ```text
///   PUSH RCX             ; save lhs (currently in RCX)
///   MOV  RCX, rhs        ; load shift amount into CL
///   POP  R11             ; R11 = saved lhs (R11 is reserved spill scratch)
///   SHL  R11, CL         ; shift on scratch
///   MOV  RCX, R11        ; result → RCX
/// ```
///
/// *CASE 2 (dst ≠ RCX):* PUSH RCX before the shift-count setup,
/// execute the shift, then POP RCX to restore the live value.
fn resolve_shift_conflicts(mf: &mut MachineFunction) {
    const MOV_OP: u32 = 0; // X86Opcode::Mov
    const SHL_OP: u32 = 19; // X86Opcode::Shl
    const SHR_OP: u32 = 20; // X86Opcode::Shr
    const SAR_OP: u32 = 21; // X86Opcode::Sar
    const PUSH_OP: u32 = 4; // X86Opcode::Push
    const POP_OP: u32 = 5; // X86Opcode::Pop
    const RCX: u16 = 1;
    // R11 is the dedicated spill scratch register — NEVER allocated by
    // the register allocator, so it is always safe to use as temporary.
    // R10 is in ALLOCATABLE_GPRS and may hold live values — do NOT use.
    const R11: u16 = 11;

    fn is_shift(opcode: u32) -> bool {
        opcode == SHL_OP || opcode == SHR_OP || opcode == SAR_OP
    }

    /// Search backward from position `before` in `insts` for a
    /// `MOV RCX, <src>` instruction.  Returns the index, or `None` if
    /// a shift or block boundary is reached first.
    fn find_mov_rcx_backward(insts: &[MachineInstruction], before: usize) -> Option<usize> {
        for j in (0..before).rev() {
            if insts[j].opcode == MOV_OP
                && matches!(insts[j].result, Some(MachineOperand::Register(r)) if r == RCX)
            {
                return Some(j);
            }
            // Stop searching past other shifts — we don't want to cross
            // into the setup region of a previous shift instruction.
            if is_shift(insts[j].opcode) {
                break;
            }
        }
        None
    }

    for block in &mut mf.blocks {
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(block.instructions.len() + 16);
        let old_insts = std::mem::take(&mut block.instructions);
        let len = old_insts.len();

        let mut i = 0;
        while i < len {
            if !is_shift(old_insts[i].opcode) {
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Check if the shift destination is RCX (register 1).
            let shift_dst_is_rcx =
                matches!(old_insts[i].result, Some(MachineOperand::Register(r)) if r == RCX);

            if shift_dst_is_rcx {
                // ── CASE 1: SHL/SHR/SAR RCX, CL ──
                //
                // The lhs is in RCX (as allocated by regalloc).  The
                // `MOV RCX, rhs` that loads the shift amount will clobber
                // the lhs.  There may be spill reloads between the MOV
                // that placed the lhs in RCX and the MOV that loads rhs.
                //
                // Strategy: find the MOV RCX, <shift_amount> in the
                // already-emitted new_insts.  Insert PUSH RCX before it
                // to save the lhs.  After the MOV RCX (which loads the
                // shift amount), pop the saved lhs into R11 (the
                // dedicated spill scratch — never allocated), shift R11,
                // then MOV RCX ← R11.
                let mov_rcx_idx = find_mov_rcx_backward(&new_insts, new_insts.len());

                if let Some(mcx_idx) = mov_rcx_idx {
                    // Insert PUSH RCX before the MOV RCX, <shift_amount>
                    let mut save_inst = MachineInstruction::new(PUSH_OP);
                    save_inst.operands.push(MachineOperand::Register(RCX));
                    new_insts.insert(mcx_idx, save_inst);
                    // mcx_idx+1 is now the MOV RCX, <shift_amount>

                    // POP R11 — restore saved lhs into the dedicated
                    // scratch register.  R11 is NEVER allocated by the
                    // register allocator, so we cannot clobber any live
                    // value.  (R10 IS allocatable and was causing segfaults
                    // when the allocator had assigned a live pointer to it.)
                    let mut pop_lhs = MachineInstruction::new(POP_OP);
                    pop_lhs.result = Some(MachineOperand::Register(R11));
                    new_insts.push(pop_lhs);

                    // SHL R11, CL — execute the shift on the scratch.
                    //
                    // The shift instruction after regalloc has:
                    //   result = Some(Register(RCX))   ← destination
                    //   operands = [Register(RCX)]     ← shift count (CL)
                    //
                    // The encoder normalises result→operands[0], so it
                    // becomes [dst, count].  We ONLY change the result
                    // (destination) to R11.  The operands[0] must REMAIN
                    // as Register(RCX) because that IS the shift count
                    // (CL register).  Replacing it with R11 would break
                    // the encoder pattern match (it checks *cl == RCX).
                    let mut shift_inst = old_insts[i].clone();
                    shift_inst.result = Some(MachineOperand::Register(R11));
                    // Do NOT modify operands — operands[0] is the CL
                    // shift count and must stay as Register(RCX).
                    new_insts.push(shift_inst);

                    // MOV RCX, R11 — result back to the destination.
                    let mut restore = MachineInstruction::new(MOV_OP);
                    restore.result = Some(MachineOperand::Register(RCX));
                    restore.operands.push(MachineOperand::Register(R11));
                    new_insts.push(restore);
                } else {
                    // Could not find the MOV RCX setup — emit as-is
                    // (this shouldn't happen in practice).
                    new_insts.push(old_insts[i].clone());
                }
            } else {
                // ── CASE 2: SHL/SHR/SAR <dst>, CL  where dst ≠ RCX ──
                //
                // The codegen loaded the shift count into RCX, which may
                // clobber the allocator's live value in RCX.  We bracket
                // the shift-count setup + shift with PUSH/POP RCX.
                let mov_rcx_idx = find_mov_rcx_backward(&new_insts, new_insts.len());

                if let Some(mcx_idx) = mov_rcx_idx {
                    // Insert PUSH RCX before the MOV RCX, <shift_count>
                    let mut save_inst = MachineInstruction::new(PUSH_OP);
                    save_inst.operands.push(MachineOperand::Register(RCX));
                    new_insts.insert(mcx_idx, save_inst);

                    // Push the shift instruction
                    new_insts.push(old_insts[i].clone());

                    // POP RCX — restore the allocator's live value.
                    let mut restore_inst = MachineInstruction::new(POP_OP);
                    restore_inst.result = Some(MachineOperand::Register(RCX));
                    new_insts.push(restore_inst);
                } else {
                    new_insts.push(old_insts[i].clone());
                }
            }

            i += 1;
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// resolve_call_arg_conflicts — fix register clobbering in call setup
// ===========================================================================

/// After register allocation, call-argument setup sequences like:
///
/// ```text
///   MOV RDI, R9       ; arg 0 ← format string
///   MOV RSI, RAX      ; arg 1 ← v1
///   MOV RDX, RSI      ; arg 2 ← v2   (BUG: RSI already overwritten by line 2!)
///   MOV RCX, RDI      ; arg 3 ← v3   (BUG: RDI already overwritten by line 1!)
///   CALL printf
/// ```
///
/// may contain conflicts: a destination register of an earlier MOV is the
/// source register of a later MOV.  This is the classical **parallel copy
/// problem** — all argument setup moves must logically happen simultaneously.
///
/// This function applies a proper parallel-move sequentialisation algorithm:
///
/// 1. Walk backwards from every CALL to find the contiguous
///    `is_call_arg_setup` instruction sequence.
/// 2. Extract register-to-register (dst, src) pairs.  Non-register-source
///    instructions (immediates, LEAs, memory operands) are emitted first
///    since they have no read-after-write conflicts.
/// 3. Remove self-moves (dst == src).
/// 4. Topologically order remaining moves: emit moves whose destination
///    is NOT a source for any pending move first.
/// 5. If a cycle is detected, break it by saving one source to R11
///    (scratch), substituting R11 as the source, and continuing.
fn resolve_call_arg_conflicts(mf: &mut MachineFunction, target: &Target) {
    // Architecture-specific constants for register move opcodes and scratch
    // registers.  Each target uses a different MOV instruction and a
    // different caller-saved scratch register that is NOT part of the ABI
    // argument register set.
    let (mov_opcode, movfp_opcode, scratch_reg, float_scratch_reg): (u32, u32, u16, u16) =
        match target {
            Target::X86_64 => (
                0,  // X86Opcode::Mov
                86, // X86Opcode::Movsd
                11, // R11 (caller-saved, not an arg reg)
                31, // XMM15
            ),
            Target::I686 => (
                0,  // X86Opcode::Mov (same enum)
                86, // X86Opcode::Movsd
                2,  // ECX (caller-saved scratch for i686)
                31, // XMM7
            ),
            Target::AArch64 => (
                161, // A64Opcode::MOV_reg
                149, // A64Opcode::FMOV_d
                16,  // X16 (IP0 — intra-procedure-call scratch)
                48,  // V16 (caller-saved FP scratch)
            ),
            Target::RiscV64 => (
                21,  // RvOpcode::ADDI (MV pseudo = ADDI rd, rs, 0)
                142, // RvOpcode::FSGNJ_D (FMV.D pseudo = FSGNJ.D rd, rs, rs)
                5,   // T0 (x5 — temporary, not an arg reg)
                32,  // FT0 (f0 — temporary, not an FP arg reg)
            ),
        };
    let mov_opcode_val = mov_opcode;
    let movfp_opcode_val = movfp_opcode;

    for block in &mut mf.blocks {
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(block.instructions.len() + 16);
        let insts = std::mem::take(&mut block.instructions);

        for inst in &insts {
            new_insts.push(inst.clone());

            if !inst.is_call {
                continue;
            }

            // The CALL is at the end of new_insts.  Walk backwards to
            // find the contiguous call-argument-setup sequence.
            let call_pos = new_insts.len() - 1;
            let mut seq_start = call_pos;
            while seq_start > 0 {
                let prev = seq_start - 1;
                if !new_insts[prev].is_call_arg_setup {
                    break;
                }
                seq_start = prev;
            }

            if seq_start == call_pos {
                continue; // No argument-setup sequence.
            }

            // ---- Indirect call target protection ----
            // For indirect calls (CALL *reg), the target register may be
            // clobbered by the argument-setup sequence (e.g., MOV RAX,0
            // for AL=0 clobbers RAX if the function pointer is in RAX).
            // We detect this and will insert a save to a safe register
            // at the start of the resolved move sequence.
            //
            // IMPORTANT: We use R10 (not R11) for saving the indirect
            // call target on x86-64.  R11 is the spill-scratch register
            // and spill reloads within the arg-setup window may clobber
            // it.  R10 is caller-saved (safe to clobber at a call site)
            // and is NOT an argument register, NOT the spill scratch,
            // so it will not be overwritten by arg-setup or spill code.
            let indirect_call_save_reg: u16 = match target {
                Target::X86_64 => 10, // R10
                Target::I686 => scratch_reg,
                Target::AArch64 => 17, // X17 (IP1 — second intra-procedure scratch)
                Target::RiscV64 => 6,  // T1 (x6 — temporary, not arg)
            };
            let indirect_save_needed: Option<u16> = {
                let call_inst_ref = &new_insts[call_pos];
                if let Some(MachineOperand::Register(target_reg)) = call_inst_ref.operands.first() {
                    let tr = *target_reg;
                    if tr != indirect_call_save_reg {
                        let clobbered = (seq_start..call_pos).any(|idx| {
                            matches!(&new_insts[idx].result,
                                Some(MachineOperand::Register(dst)) if *dst == tr)
                        });
                        if clobbered {
                            Some(tr)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // If save needed, update CALL to use the safe register.
            if indirect_save_needed.is_some() {
                new_insts[call_pos].operands[0] = MachineOperand::Register(indirect_call_save_reg);
            }

            // Partition instructions into:
            // - `non_reg_moves`: instructions whose source is NOT a plain
            //   register (immediates, LEAs, etc.) — safe to emit in any
            //   order since they don't read from registers that might be
            //   clobbered.
            // - `reg_moves`: register-to-register moves that need
            //   parallel-copy resolution.
            let setup_insts: Vec<MachineInstruction> =
                new_insts.drain(seq_start..call_pos).collect();
            // call_pos shifted; CALL is now at seq_start.

            let mut reg_moves: Vec<(u16, u16, u32)> = Vec::new(); // (dst, src, opcode)
            let mut non_reg_move_insts: Vec<MachineInstruction> = Vec::new();
            // Compound pairs: (Vec<MachineInstruction>, effective_dst).
            // A load-then-move through scratch (e.g., LD t0,-X(s0) then
            // MV a3,t0) MUST stay together — separating them causes the
            // scratch to be clobbered between the load and the move.
            let mut compound_moves: Vec<(Vec<MachineInstruction>, u16)> = Vec::new();
            // Track indices we've already consumed as part of a compound.
            let mut consumed: crate::common::fx_hash::FxHashSet<usize> =
                crate::common::fx_hash::FxHashSet::default();

            for (idx, si) in setup_insts.iter().enumerate() {
                if consumed.contains(&idx) {
                    continue;
                }
                let dst_reg = match &si.result {
                    Some(MachineOperand::Register(r)) => Some(*r),
                    _ => None,
                };
                let src_reg = match si.operands.first() {
                    Some(MachineOperand::Register(r)) => Some(*r),
                    _ => None,
                };

                if let (Some(d), Some(s)) = (dst_reg, src_reg) {
                    // Register-to-register move — needs parallel resolution.
                    reg_moves.push((d, s, si.opcode));
                } else {
                    // Non-register source (immediate, global symbol, LEA,
                    // memory operand, spill load).
                    //
                    // Check if this instruction loads into a scratch
                    // register that a LATER instruction immediately moves
                    // to an arg register.  If so, they form a compound
                    // pair that must be emitted atomically to prevent the
                    // scratch register from being clobbered in between.
                    let has_mem_src = si
                        .operands
                        .iter()
                        .any(|op| matches!(op, MachineOperand::Memory { .. }));
                    if has_mem_src {
                        // Determine the effective destination register of this
                        // spill load. On RISC-V/AArch64/i686, it's in `result`.
                        // On x86-64, MOV instructions encode the destination as
                        // operands[0] and the memory source as operands[1], with
                        // result = None. We must check both.
                        let eff_load_dst = dst_reg.or_else(|| {
                            if si.operands.len() >= 2 {
                                if let MachineOperand::Register(r) = &si.operands[0] {
                                    Some(*r)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        });
                        if let Some(d_scratch) = eff_load_dst {
                            // Search forward through the IMMEDIATELY FOLLOWING
                            // non-consumed setup instructions for a paired move
                            // that reads d_scratch.  This detects spill-reload
                            // compound pairs (LD scratch, [spill]; MV arg, scratch)
                            // which must be emitted atomically.
                            //
                            // IMPORTANT: Only consider the NEAREST non-consumed
                            // instruction as a potential pair.  Searching further
                            // causes incorrect pairing when a struct-pair memory
                            // load (directly to an arg register like RDI) is
                            // matched with a distant reg-to-reg move that happens
                            // to read from the same register.  For example:
                            //   [0] MOV RDI, [RBP-16]   (struct lo → arg reg)
                            //   [1] MOV RSI, [RBP-8]    (struct hi → arg reg)
                            //   [2] MOV RDX, RCX        (scalar arg move)
                            //   [3] MOV RCX, RSI        (scalar arg move)
                            //   [4] MOV R8, RDI         (scalar arg move — reads RDI!)
                            // Without the adjacency restriction, [0] and [4] are
                            // incorrectly paired, consuming [1],[2],[3] as
                            // intermediates.  This collapses all moves into one
                            // compound, bypasses conflict detection, and emits
                            // them in original order — which clobbers RDI/RSI
                            // before the scalar moves at [3]/[4] read them.
                            let mut found_pair = false;
                            // Find the nearest non-consumed instruction after idx.
                            let nearest_j =
                                ((idx + 1)..setup_insts.len()).find(|j| !consumed.contains(j));
                            if let Some(j) = nearest_j {
                                let next_si = &setup_insts[j];
                                let next_dst = match &next_si.result {
                                    Some(MachineOperand::Register(r)) => Some(*r),
                                    _ => None,
                                };
                                // Check if next_si reads from d_scratch:
                                // 1. Register source: operands[0] == Register(d_scratch)
                                // 2. Memory base: operands has Memory{base=d_scratch}
                                let next_reads_scratch =
                                    next_si.operands.iter().any(|op| match op {
                                        MachineOperand::Register(r) => *r == d_scratch,
                                        MachineOperand::Memory { base: Some(r), .. } => {
                                            *r == d_scratch
                                        }
                                        _ => false,
                                    });
                                if next_reads_scratch {
                                    if let Some(eff_dst) = next_dst {
                                        // Only form a compound pair when the
                                        // memory load targets a KNOWN spill
                                        // scratch register (R11 on x86-64,
                                        // X16/X17 on AArch64, X5/X6 on RISC-V,
                                        // ECX on i686).  Spill reloads inserted
                                        // by the register allocator ALWAYS use
                                        // these designated scratch registers.
                                        //
                                        // Memory loads into OTHER registers
                                        // (like argument registers RDX, RSI)
                                        // are codegen-generated direct argument
                                        // placements (e.g., struct-pair loads)
                                        // that participate independently in the
                                        // parallel-move resolution.  Merging
                                        // them into a compound hides write-read
                                        // conflicts: when reg R is both the
                                        // direct argument target (written by the
                                        // mem load) and a live source for an
                                        // independent reg-to-reg move, the
                                        // compound collapses them into one unit,
                                        // bypasses conflict detection, preserves
                                        // the original (wrong) order, and causes
                                        // the reg-to-reg move to read the newly
                                        // loaded value instead of the old one.
                                        let is_spill_scratch = d_scratch == scratch_reg
                                            || d_scratch == float_scratch_reg;
                                        if is_spill_scratch {
                                            let chain = vec![si.clone(), next_si.clone()];
                                            consumed.insert(j);
                                            compound_moves.push((chain, eff_dst));
                                            found_pair = true;
                                        }
                                    }
                                }
                            }
                            if found_pair {
                                continue;
                            }
                        }
                    }
                    non_reg_move_insts.push(si.clone());
                }
            }

            // Check if there are actually any register conflicts.
            // A conflict exists if:
            //   (a) A reg-to-reg move writes to a register that is a
            //       source for another reg-to-reg move, OR
            //   (b) A compound move (memory load → arg reg) writes to a
            //       register that is a source for a reg-to-reg move, OR
            //   (c) A non-reg move (immediate/LEA → reg) writes to a
            //       register that is a source for a reg-to-reg move.
            // Without (b)/(c), the early exit restores original ordering
            // which can clobber a source register before it is read.
            let has_conflict = {
                let dsts: crate::common::fx_hash::FxHashSet<u16> =
                    reg_moves.iter().map(|m| m.0).collect();
                let reg_vs_reg = reg_moves.iter().any(|m| dsts.contains(&m.1));
                let reg_srcs: crate::common::fx_hash::FxHashSet<u16> =
                    reg_moves.iter().map(|m| m.1).collect();
                let compound_vs_reg = compound_moves.iter().any(|cm| reg_srcs.contains(&cm.1));
                let non_reg_vs_reg = non_reg_move_insts.iter().any(|nrm| {
                    // Get destination register from `result` or — on
                    // x86-64 where MOV may encode dst in operands[0]
                    // with result = None — from the first operand if
                    // it is a Register AND another operand is Memory.
                    let dst_r = match &nrm.result {
                        Some(MachineOperand::Register(r)) => Some(*r),
                        _ => {
                            // Fallback: check if operands[0] is Register
                            // and there is a Memory operand elsewhere
                            // (typical x86 MOV dst, [mem] encoding).
                            let op0_reg = nrm.operands.first().and_then(|op| {
                                if let MachineOperand::Register(r) = op {
                                    Some(*r)
                                } else {
                                    None
                                }
                            });
                            let has_mem = nrm
                                .operands
                                .iter()
                                .skip(1)
                                .any(|op| matches!(op, MachineOperand::Memory { .. }));
                            if has_mem {
                                op0_reg
                            } else {
                                None
                            }
                        }
                    };
                    dst_r.map_or(false, |r| reg_srcs.contains(&r))
                });
                // Check if any non-reg-move instruction (e.g., PUSH) READS
                // from a register that is a DESTINATION of a reg-to-reg
                // move.  Without this check, when no other conflicts are
                // detected, the original instruction order is preserved
                // (register-arg MOVs before PUSHes), causing PUSHes to
                // read clobbered register values instead of the original
                // stack-argument values.
                //
                // Example: `sqlite3GenerateConstraintChecks` has 13 params;
                // the allocator assigns stack-arg vregs to RCX/RSI/RDX,
                // which are also destinations for register args 4/2/3.
                // The PUSHes must execute BEFORE those MOVs.
                let non_reg_reads_dst = non_reg_move_insts.iter().any(|nrm| {
                    nrm.operands.iter().any(|op| match op {
                        MachineOperand::Register(r) => dsts.contains(r),
                        _ => false,
                    })
                });
                // Also check compound moves that read from reg-move dests.
                let compound_reads_dst = compound_moves.iter().any(|cm| {
                    cm.0.iter().any(|inst| {
                        inst.operands.iter().any(|op| match op {
                            MachineOperand::Register(r) => dsts.contains(r),
                            _ => false,
                        })
                    })
                });
                reg_vs_reg
                    || compound_vs_reg
                    || non_reg_vs_reg
                    || non_reg_reads_dst
                    || compound_reads_dst
            };

            if !has_conflict && indirect_save_needed.is_none() {
                // No conflicts and no indirect save — put back in original order.
                let call_inst = new_insts.remove(seq_start);
                for si in setup_insts {
                    new_insts.push(si);
                }
                new_insts.push(call_inst);
                continue;
            }
            if let (false, Some(target_reg)) = (has_conflict, indirect_save_needed) {
                // No register-register conflicts, but we need to protect
                // the indirect call target. Insert save before setup.
                let call_inst = new_insts.remove(seq_start);
                let mut save_inst = MachineInstruction::new(mov_opcode_val);
                save_inst.result = Some(MachineOperand::Register(indirect_call_save_reg));
                save_inst
                    .operands
                    .push(MachineOperand::Register(target_reg));
                save_inst.is_call_arg_setup = true;
                new_insts.push(save_inst);
                for si in setup_insts {
                    new_insts.push(si);
                }
                new_insts.push(call_inst);
                continue;
            }

            // ---- Parallel-move sequentialisation for reg-to-reg moves ----
            //
            // Remove self-moves, then topologically order: emit moves whose
            // destination is not a source for any pending move.  If a cycle
            // is found, break it using a scratch register.
            let mut pending = reg_moves;
            pending.retain(|&(d, s, _)| d != s);

            let mut resolved: Vec<MachineInstruction> = Vec::new();

            // If we need to protect an indirect call target register,
            // insert MOV <safe_reg>, target_reg as the VERY FIRST
            // instruction before any arg-setup moves can clobber the
            // target.  Uses indirect_call_save_reg (R10 on x86-64)
            // instead of R11 to avoid conflicts with spill reloads.
            if let Some(target_reg) = indirect_save_needed {
                let mut save_inst = MachineInstruction::new(mov_opcode_val);
                save_inst.result = Some(MachineOperand::Register(indirect_call_save_reg));
                save_inst
                    .operands
                    .push(MachineOperand::Register(target_reg));
                save_inst.is_call_arg_setup = true;
                resolved.push(save_inst);

                // Also update any reg-to-reg moves that read from
                // target_reg to read from R11 instead, since we saved it.
                // (Not strictly necessary since the save happens first,
                // but ensures consistency if target_reg is also used as
                // a source for arg-setup moves.)
            }

            // Emit non-register-source moves first (no conflict risk on
            // reads). BUT: if one of these writes to a register that is a
            // source for a pending reg-to-reg move, we must defer it.
            let pending_srcs: crate::common::fx_hash::FxHashSet<u16> =
                pending.iter().map(|m| m.1).collect();
            let mut deferred_non_reg: Vec<MachineInstruction> = Vec::new();
            // Deferred compound moves: (instructions, effective_dst).
            let mut deferred_compounds: Vec<(Vec<MachineInstruction>, u16)> = Vec::new();
            for nrm in non_reg_move_insts {
                let dst = match &nrm.result {
                    Some(MachineOperand::Register(r)) => Some(*r),
                    _ => None,
                };
                if let Some(d) = dst {
                    if pending_srcs.contains(&d) {
                        // This non-reg move writes to a register that is
                        // a source for a pending reg-to-reg move. Defer it.
                        deferred_non_reg.push(nrm);
                        continue;
                    }
                }
                resolved.push(nrm);
            }
            // Defer compound moves whose effective_dst is a source for a
            // pending reg-to-reg move; otherwise emit them now.
            for cm in compound_moves {
                if pending_srcs.contains(&cm.1) {
                    deferred_compounds.push(cm);
                } else {
                    resolved.extend(cm.0);
                }
            }

            let mut iteration_limit = pending.len() * pending.len() + 10;
            while !pending.is_empty() && iteration_limit > 0 {
                iteration_limit -= 1;

                let needed_srcs: crate::common::fx_hash::FxHashSet<u16> =
                    pending.iter().map(|m| m.1).collect();

                // Find a safe move (dst not needed as source by any pending).
                let safe_idx = pending
                    .iter()
                    .position(|&(d, _, _)| !needed_srcs.contains(&d));

                if let Some(idx) = safe_idx {
                    let (dst, src, opcode) = pending.remove(idx);
                    let mut mi = MachineInstruction::new(opcode);
                    mi.result = Some(MachineOperand::Register(dst));
                    mi.operands.push(MachineOperand::Register(src));
                    // For RISC-V FSGNJ.D: duplicate rs1→rs2 for FMV.D pseudo
                    if opcode == movfp_opcode_val && matches!(target, Target::RiscV64) {
                        mi.operands.push(MachineOperand::Register(src));
                    }
                    mi.is_call_arg_setup = true;
                    resolved.push(mi);

                    // Check if any deferred non-reg moves or compound pairs
                    // can now be emitted.
                    let new_pending_srcs: crate::common::fx_hash::FxHashSet<u16> =
                        pending.iter().map(|m| m.1).collect();
                    let mut still_deferred = Vec::new();
                    for nrm in deferred_non_reg {
                        let d = match &nrm.result {
                            Some(MachineOperand::Register(r)) => Some(*r),
                            _ => None,
                        };
                        if d.map_or(false, |dd| new_pending_srcs.contains(&dd)) {
                            still_deferred.push(nrm);
                        } else {
                            resolved.push(nrm);
                        }
                    }
                    deferred_non_reg = still_deferred;
                    // Also check deferred compounds.
                    let mut still_deferred_c = Vec::new();
                    for cm in deferred_compounds {
                        if new_pending_srcs.contains(&cm.1) {
                            still_deferred_c.push(cm);
                        } else {
                            resolved.extend(cm.0);
                        }
                    }
                    deferred_compounds = still_deferred_c;
                } else {
                    // Cycle detected — break it with a scratch register.
                    let (_, cycle_src, cycle_opcode) = pending[0];
                    // Detect FP register ranges per architecture:
                    // x86-64/i686: XMM regs 16..31, RISC-V/AArch64: FP regs >= 32
                    let is_float = cycle_opcode == movfp_opcode_val
                        || match target {
                            Target::X86_64 | Target::I686 => (16..32).contains(&cycle_src),
                            Target::AArch64 => cycle_src >= 32,
                            Target::RiscV64 => cycle_src >= 32,
                        };
                    let (scratch, save_op) = if is_float {
                        (float_scratch_reg, movfp_opcode_val)
                    } else {
                        (scratch_reg, mov_opcode_val)
                    };

                    let mut save_inst = MachineInstruction::new(save_op);
                    save_inst.result = Some(MachineOperand::Register(scratch));
                    save_inst.operands.push(MachineOperand::Register(cycle_src));
                    // For RISC-V FSGNJ.D and AArch64 FMOV: the FP move
                    // pseudo requires rs2 == rs1 (source duplicated).
                    // Add second register operand for R-type FP moves.
                    if is_float && matches!(target, Target::RiscV64) {
                        save_inst.operands.push(MachineOperand::Register(cycle_src));
                    }
                    save_inst.is_call_arg_setup = true;
                    resolved.push(save_inst);

                    for m in &mut pending {
                        if m.1 == cycle_src {
                            m.1 = scratch;
                        }
                    }
                }
            }

            // Emit any remaining deferred non-reg moves and compounds.
            resolved.extend(deferred_non_reg);
            for cm in deferred_compounds {
                resolved.extend(cm.0);
            }

            // Splice resolved moves back before the CALL instruction.
            let call_inst = new_insts.remove(seq_start);
            new_insts.extend(resolved);
            new_insts.push(call_inst);
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// insert_prologue_epilogue — inject frame setup/teardown instructions
// ===========================================================================

/// Inserts prologue instructions at the beginning of the entry block and
/// epilogue instructions before every return instruction.
fn insert_prologue_epilogue(
    mf: &mut MachineFunction,
    prologue: Vec<MachineInstruction>,
    epilogue: Vec<MachineInstruction>,
) {
    // Insert prologue at the start of the entry block.
    if !prologue.is_empty() && !mf.blocks.is_empty() {
        let entry = &mut mf.blocks[0];
        let mut new_instructions = Vec::with_capacity(prologue.len() + entry.instructions.len());
        new_instructions.extend(prologue);
        new_instructions.append(&mut entry.instructions);
        entry.instructions = new_instructions;
    }

    // Insert epilogue before every return instruction.  Some backends
    // (notably AArch64) place all instructions in a single block, so
    // multiple `ret`s may exist within one block.  We must insert the
    // epilogue before EVERY return — not just the last instruction.
    if !epilogue.is_empty() {
        for block in &mut mf.blocks {
            // Collect indices of all return instructions in this block.
            let ret_indices: Vec<usize> = block
                .instructions
                .iter()
                .enumerate()
                .filter(|(_, mi)| mi.is_terminator && !mi.is_branch)
                .map(|(i, _)| i)
                .collect();

            if ret_indices.is_empty() {
                continue;
            }

            // Rebuild the instruction list, inserting epilogue before each
            // return.  Process in forward order; the growing offset is
            // accounted for by building a fresh vector.
            let epi_len = epilogue.len();
            let mut new_insts =
                Vec::with_capacity(block.instructions.len() + ret_indices.len() * epi_len);
            let mut prev = 0usize;
            for &ri in &ret_indices {
                // Copy instructions up to (but not including) the return.
                new_insts.extend_from_slice(&block.instructions[prev..ri]);
                // Insert the epilogue sequence.
                new_insts.extend(epilogue.clone());
                // Copy the return instruction itself.
                new_insts.push(block.instructions[ri].clone());
                prev = ri + 1;
            }
            // Copy any remaining instructions after the last return.
            if prev < block.instructions.len() {
                new_insts.extend_from_slice(&block.instructions[prev..]);
            }
            block.instructions = new_insts;
        }
    }
}

// ===========================================================================
// emit_global_variable — encode a global variable into the appropriate section
// ===========================================================================

/// Emits a global variable into the appropriate ELF section (.data, .rodata,
/// or .bss) based on its attributes.
fn emit_global_variable(
    global: &crate::ir::module::GlobalVariable,
    data_section: &mut Vec<u8>,
    rodata_section: &mut Vec<u8>,
    bss_size: &mut u64,
    _ctx: &CodegenContext,
) -> GlobalSymbolInfo {
    use crate::ir::module::Constant;

    // Determine placement based on global attributes.
    let (section, offset, size) = match (&global.initializer, global.is_constant) {
        // Constant with initializer → .rodata
        (Some(init), true) => {
            let offset = rodata_section.len() as u64;
            let bytes = constant_to_bytes_typed(init, Some(&global.ty));
            let size = bytes.len() as u64;
            rodata_section.extend_from_slice(&bytes);
            (SectionPlacement::Rodata, offset, size)
        }
        // ZeroInit → .bss
        (Some(Constant::ZeroInit), false) | (None, _) => {
            let size = global_variable_size(global);
            let offset = *bss_size;
            *bss_size += size;
            (SectionPlacement::Bss, offset, size)
        }
        // Initialized non-constant → .data
        (Some(init), false) => {
            let offset = data_section.len() as u64;
            let bytes = constant_to_bytes_typed(init, Some(&global.ty));
            let size = bytes.len() as u64;
            data_section.extend_from_slice(&bytes);
            (SectionPlacement::Data, offset, size)
        }
    };

    let is_global = global.linkage == crate::ir::module::Linkage::External
        || global.linkage == crate::ir::module::Linkage::Weak;

    GlobalSymbolInfo {
        name: global.name.clone(),
        section,
        offset,
        size,
        is_global,
    }
}

/// Converts a constant initializer to raw bytes for section embedding.
///
/// When a type is provided, integers are truncated to the correct byte
/// width. Without a type, integers fall back to 8 bytes (i64 width).
#[allow(dead_code)]
fn constant_to_bytes(constant: &crate::ir::module::Constant) -> Vec<u8> {
    constant_to_bytes_typed(constant, None)
}

/// Type-aware version that uses the IR type to determine integer widths.
fn constant_to_bytes_typed(
    constant: &crate::ir::module::Constant,
    ir_ty: Option<&crate::ir::types::IrType>,
) -> Vec<u8> {
    use crate::ir::module::Constant;
    use crate::ir::types::IrType;

    match constant {
        Constant::Integer(v) => {
            // When the target IR type is a float/double but the constant
            // was evaluated as an integer (e.g. `double C = 2;`), convert
            // the integer value to the appropriate IEEE 754 representation.
            match ir_ty {
                Some(IrType::F32) => {
                    let fv = *v as f32;
                    return fv.to_le_bytes().to_vec();
                }
                Some(IrType::F64) => {
                    let fv = *v as f64;
                    return fv.to_le_bytes().to_vec();
                }
                Some(IrType::F80) => {
                    // Store integer-initialized long double globals as f64
                    // representation so that the codegen's `movsd` (SSE
                    // double load) reads the correct value.  Pad to 16
                    // bytes for sizeof(long double) == 16.
                    let fv = *v as f64;
                    let mut result = fv.to_le_bytes().to_vec();
                    result.resize(16, 0);
                    return result;
                }
                _ => {}
            }
            // Determine byte width from the IR type when available.
            let width = match ir_ty {
                Some(IrType::I1) | Some(IrType::I8) => 1,
                Some(IrType::I16) => 2,
                Some(IrType::I32) => 4,
                Some(IrType::I64) | Some(IrType::Ptr) => 8,
                Some(IrType::I128) => 16,
                _ => 8, // Default to 8 bytes for unknown or unspecified types.
            };
            let full = v.to_le_bytes();
            full[..width].to_vec()
        }
        Constant::Float(v) => match ir_ty {
            Some(IrType::F32) => (*v as f32).to_le_bytes().to_vec(),
            Some(IrType::F80) => {
                // The codegen loads F80 globals with `movsd` (8-byte SSE
                // double load) because all long-double arithmetic is done at
                // f64 precision internally.  Store the f64 bytes in the low
                // 8 bytes so `movsd` reads the correct double value, and pad
                // to 16 bytes to maintain `sizeof(long double) == 16`.
                let mut result = v.to_le_bytes().to_vec();
                result.resize(16, 0);
                result
            }
            // When the target type is an integer (e.g., _Complex char
            // has element type I8), truncate the float to the integer
            // width.  This handles complex-constant initializers where
            // the constant is stored as Float but the destination type
            // is an integer.
            Some(IrType::I1) | Some(IrType::I8) => {
                vec![(*v as i8) as u8]
            }
            Some(IrType::I16) => (*v as i16).to_le_bytes().to_vec(),
            Some(IrType::I32) => (*v as i32).to_le_bytes().to_vec(),
            Some(IrType::I64) => (*v as i64).to_le_bytes().to_vec(),
            Some(IrType::I128) => (*v as i128).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        },
        Constant::LongDouble(bytes) => {
            // 80-bit extended precision — 10 raw bytes.  Convert to f64
            // representation for the low 8 bytes so that the codegen's
            // `movsd` (SSE double load) reads the correct value.  All
            // long-double arithmetic in BCC is performed at f64 precision
            // internally, so this conversion is consistent.
            let raw: [u8; 10] = {
                let mut arr = [0u8; 10];
                for (i, &b) in bytes.iter().enumerate().take(10) {
                    arr[i] = b;
                }
                arr
            };
            let ld = crate::common::long_double::LongDouble::from_bytes(&raw);
            let f64_val = ld.to_f64();
            let mut result = f64_val.to_le_bytes().to_vec();
            result.resize(16, 0);
            result
        }
        Constant::String(bytes) => {
            // When the target IR type is an array, fit the string bytes
            // to the exact array size:
            //  - Pad with zeros if the string is shorter (char[4] = "ab")
            //  - Truncate if the string is longer (char[4] = "1234" where
            //    the literal includes a null terminator that doesn't fit)
            if let Some(ty) = ir_ty {
                let expected_sz = ir_type_size(ty) as usize;
                if expected_sz > 0 && bytes.len() != expected_sz {
                    let mut fitted = bytes[..bytes.len().min(expected_sz)].to_vec();
                    fitted.resize(expected_sz, 0);
                    return fitted;
                }
            }
            bytes.clone()
        }
        Constant::ZeroInit => {
            // When we know the type, emit the correct number of zero bytes.
            if let Some(ty) = ir_ty {
                let sz = ir_type_size(ty) as usize;
                vec![0u8; sz]
            } else {
                Vec::new()
            }
        }
        Constant::Struct(fields) => {
            let mut result = Vec::new();
            let (field_types, packed) = match ir_ty {
                Some(IrType::Struct(st)) => (Some(&st.fields), st.packed),
                _ => (None, false),
            };
            let mut offset: usize = 0;
            for (i, field) in fields.iter().enumerate() {
                let ft = field_types.and_then(|fts| fts.get(i));
                // Insert alignment padding before this field (unless packed).
                if !packed {
                    if let Some(fty) = ft {
                        let fa = ir_type_align(fty) as usize;
                        let rem = offset % fa;
                        if rem != 0 {
                            let pad = fa - rem;
                            result.extend(std::iter::repeat(0u8).take(pad));
                            offset += pad;
                        }
                    }
                }
                let field_bytes = constant_to_bytes_typed(field, ft);
                offset += field_bytes.len();
                result.extend_from_slice(&field_bytes);
            }
            // Add tail padding to reach the full struct ABI size.
            if let Some(sty) = ir_ty {
                let total = ir_type_size(sty) as usize;
                if result.len() < total {
                    result.resize(total, 0);
                }
            }
            result
        }
        Constant::Array(elements) => {
            let mut result = Vec::new();
            let (elem_ty, declared_len) = match ir_ty {
                Some(IrType::Array(et, n)) => (Some(et.as_ref()), Some(*n)),
                _ => (None, None),
            };
            // Compute expected element byte size from the declared element type.
            // When brace-elision occurs in multi-dimensional array initializers,
            // a scalar constant (e.g. Integer(1)) may appear where a sub-array
            // constant is expected.  The serialized bytes of the scalar will be
            // shorter than the element type size, so we must zero-pad each
            // element to the correct ABI size.
            let expected_elem_size = elem_ty.map(|t| ir_type_size(t) as usize);
            for elem in elements {
                let before = result.len();
                result.extend_from_slice(&constant_to_bytes_typed(elem, elem_ty));
                let emitted = result.len() - before;
                if let Some(exp) = expected_elem_size {
                    if emitted < exp {
                        // Zero-pad to the full element size (handles brace
                        // elision where a scalar fills only the first bytes
                        // of a sub-array slot).
                        result.resize(result.len() + (exp - emitted), 0);
                    }
                }
            }
            // If the Constant::Array has fewer elements than the declared
            // array length (e.g. partial initializer), zero-pad the tail.
            if let (Some(decl), Some(esz)) = (declared_len, expected_elem_size) {
                let total_expected = decl * esz;
                if result.len() < total_expected {
                    result.resize(total_expected, 0);
                }
            }
            result
        }
        Constant::GlobalRef(_name) => {
            // Global references are resolved by the linker via relocations.
            // Emit a placeholder (zero bytes) — the linker patches this.
            let ptr_size = match ir_ty {
                Some(IrType::Ptr) => 8,
                _ => 8,
            };
            vec![0u8; ptr_size]
        }
        Constant::GlobalRefOffset(_name, _addend) => {
            // Same as GlobalRef — the actual addend is carried in the
            // relocation entry's r_addend field. The data section holds
            // placeholder zeros (or the addend for REL, but we use RELA).
            let ptr_size = match ir_ty {
                Some(IrType::Ptr) => 8,
                _ => 8,
            };
            vec![0u8; ptr_size]
        }
        Constant::Null => vec![0u8; 8],
        Constant::Undefined => vec![0u8; 8],
    }
}

/// Computes the storage size for a global variable based on its IR type.
/// Recursively collects `GlobalRef` entries within a constant initializer,
/// recording their byte offset within the emitted data so that the caller
/// can generate ELF relocation entries (e.g., `R_X86_64_64`) for the
/// linker to patch at link time.
///
/// `base_offset` is the byte offset of the start of this constant within
/// the containing section (`.data` or `.rodata`).
fn collect_constant_relocs(
    constant: &crate::ir::module::Constant,
    base_offset: u64,
    ir_ty: Option<&crate::ir::types::IrType>,
    relocs: &mut Vec<(u64, String, i64)>,
) {
    use crate::ir::module::Constant;
    use crate::ir::types::IrType;

    match constant {
        Constant::GlobalRef(name) => {
            relocs.push((base_offset, name.clone(), 0));
        }
        Constant::GlobalRefOffset(name, addend) => {
            relocs.push((base_offset, name.clone(), *addend));
        }
        Constant::Struct(fields) => {
            let (field_types, packed) = match ir_ty {
                Some(IrType::Struct(st)) => (Some(&st.fields), st.packed),
                _ => (None, false),
            };
            let mut offset = base_offset;
            for (i, field) in fields.iter().enumerate() {
                let ft = field_types.and_then(|fts| fts.get(i));
                // Account for inter-field alignment padding (mirrors
                // constant_to_bytes_typed).
                if !packed {
                    if let Some(fty) = ft {
                        let fa = ir_type_align(fty);
                        let rem = offset % fa;
                        if rem != 0 {
                            offset += fa - rem;
                        }
                    }
                }
                collect_constant_relocs(field, offset, ft, relocs);
                let field_bytes = constant_to_bytes_typed(field, ft);
                offset += field_bytes.len() as u64;
            }
        }
        Constant::Array(elements) => {
            let elem_ty = match ir_ty {
                Some(IrType::Array(et, _)) => Some(et.as_ref()),
                _ => None,
            };
            let mut offset = base_offset;
            for elem in elements {
                collect_constant_relocs(elem, offset, elem_ty, relocs);
                let elem_bytes = constant_to_bytes_typed(elem, elem_ty);
                offset += elem_bytes.len() as u64;
            }
        }
        // All other constant types (Integer, Float, String, ZeroInit, Null,
        // Undefined, LongDouble) contain no symbol references.
        _ => {}
    }
}

fn global_variable_size(global: &crate::ir::module::GlobalVariable) -> u64 {
    // Use the padding-aware ir_type_size to compute the correct ABI size
    // for structs (including inter-field and tail padding).
    ir_type_size(&global.ty)
}

/// Returns the ABI alignment (in bytes) for an IR type.
///
/// For structs, the alignment is the maximum alignment of any field (or 1
/// for packed structs).  For arrays, the alignment equals the element
/// alignment.
fn ir_type_align(ty: &crate::ir::types::IrType) -> u64 {
    use crate::ir::types::IrType;
    match ty {
        IrType::Void | IrType::I1 | IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 | IrType::F32 => 4,
        IrType::I64 | IrType::F64 | IrType::Ptr => 8,
        IrType::I128 | IrType::F80 => 16,
        IrType::Array(elem, _) => ir_type_align(elem),
        IrType::Struct(st) => {
            if st.packed {
                1
            } else {
                st.fields.iter().map(ir_type_align).max().unwrap_or(1)
            }
        }
        IrType::Function(_, _) => 1,
    }
}

/// Computes the ABI size of an IR type in bytes, accounting for struct
/// field alignment padding and tail padding.
/// Public re-export for use by architecture backends.
pub(crate) fn ir_type_size_pub(ty: &crate::ir::types::IrType) -> u64 {
    ir_type_size(ty)
}

fn ir_type_size(ty: &crate::ir::types::IrType) -> u64 {
    use crate::ir::types::IrType;
    match ty {
        IrType::Void => 0,
        IrType::I1 | IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 | IrType::F32 => 4,
        IrType::I64 | IrType::F64 | IrType::Ptr => 8,
        IrType::I128 | IrType::F80 => 16,
        IrType::Array(elem, count) => ir_type_size(elem) * (*count as u64),
        IrType::Struct(st) => {
            if st.packed {
                // Packed structs: no padding, just sum field sizes.
                return st.fields.iter().map(ir_type_size).sum();
            }
            // Compute size with inter-field alignment padding.
            let mut offset: u64 = 0;
            let mut max_align: u64 = 1;
            for f in &st.fields {
                let fa = ir_type_align(f);
                if fa > max_align {
                    max_align = fa;
                }
                // Align the current offset to this field's requirement.
                let rem = offset % fa;
                if rem != 0 {
                    offset += fa - rem;
                }
                offset += ir_type_size(f);
            }
            // Add tail padding so the struct size is a multiple of its
            // overall alignment (required for arrays of structs).
            let rem = offset % max_align;
            if rem != 0 {
                offset += max_align - rem;
            }
            offset
        }
        IrType::Function(_, _) => 0,
    }
}

// ===========================================================================
// write_relocatable_object — produce a .o ELF file
// ===========================================================================

/// Patch a single intra-module relocation in the combined `.text` section.
///
/// For x86-64 and i686, most PC-relative relocations use a raw 4-byte
/// little-endian addend written directly at the patch site.  For AArch64
/// and RISC-V 64, the offset is encoded into instruction-specific immediate
/// fields while preserving opcode bits.
fn patch_intra_module_reloc(
    text: &mut [u8],
    off: usize,
    value: i64, // S + A - P (byte offset)
    rel_type: u32,
    target: &Target,
) {
    match target {
        Target::X86_64 | Target::I686 => {
            // x86 family: raw 4-byte PC-relative offset
            let bytes = (value as i32).to_le_bytes();
            text[off..off + 4].copy_from_slice(&bytes);
        }
        Target::AArch64 => {
            // Read existing instruction word (preserves opcode)
            let inst = u32::from_le_bytes([text[off], text[off + 1], text[off + 2], text[off + 3]]);
            // AArch64 CALL26 (283), JUMP26 (282): imm26 in bits [25:0], offset/4
            const A64_CALL26: u32 = 283;
            const A64_JUMP26: u32 = 282;
            // AArch64 CONDBR19 (280): imm19 in bits [23:5], offset/4
            const A64_CONDBR19: u32 = 280;
            // AArch64 TSTBR14 (279): imm14 in bits [18:5], offset/4
            const A64_TSTBR14: u32 = 279;
            // AArch64 ADR_PREL_PG_HI21 (275): ADRP page offset
            const A64_ADR_PREL_PG_HI21: u32 = 275;
            const A64_ADR_PREL_PG_HI21_NC: u32 = 276;
            // AArch64 ADD_ABS_LO12_NC (277): ADD lower 12 bits
            const A64_ADD_ABS_LO12_NC: u32 = 277;
            // AArch64 LD64_GOT_LO12_NC (312): LDR lower 12 bits for GOT
            const A64_LDST64_ABS_LO12_NC: u32 = 286;

            let patched = match rel_type {
                A64_CALL26 | A64_JUMP26 => {
                    let imm26 = ((value >> 2) as u32) & 0x03FF_FFFF;
                    (inst & 0xFC00_0000) | imm26
                }
                A64_CONDBR19 => {
                    let imm19 = ((value >> 2) as u32) & 0x7_FFFF;
                    (inst & 0xFF00_001F) | (imm19 << 5)
                }
                A64_TSTBR14 => {
                    let imm14 = ((value >> 2) as u32) & 0x3FFF;
                    (inst & 0xFFF8_001F) | (imm14 << 5)
                }
                A64_ADR_PREL_PG_HI21 | A64_ADR_PREL_PG_HI21_NC => {
                    // ADRP: page-relative offset >> 12, encode immhi:immlo
                    let page_off = value >> 12;
                    let immlo = ((page_off as u32) & 0x3) << 29;
                    let immhi = (((page_off >> 2) as u32) & 0x7_FFFF) << 5;
                    (inst & 0x9F00_001F) | immlo | immhi
                }
                A64_ADD_ABS_LO12_NC => {
                    let lo12 = (value as u32) & 0xFFF;
                    (inst & 0xFFC0_03FF) | (lo12 << 10)
                }
                A64_LDST64_ABS_LO12_NC => {
                    // LDR Xt, [Xn, #lo12] — lo12 scaled by 8 for 64-bit loads
                    let lo12 = ((value as u32) & 0xFFF) >> 3;
                    (inst & 0xFFC0_03FF) | (lo12 << 10)
                }
                _ => {
                    // Unknown relocation type — fall back to raw 4-byte write
                    // (likely wrong, but better than silently dropping).
                    let bytes = (value as i32).to_le_bytes();
                    text[off..off + 4].copy_from_slice(&bytes);
                    return;
                }
            };
            let bytes = patched.to_le_bytes();
            text[off..off + 4].copy_from_slice(&bytes);
        }
        Target::RiscV64 => {
            // RISC-V relocations — most common ones for intra-module calls
            // R_RISCV_JAL (17): 20-bit J-type immediate
            // R_RISCV_BRANCH (16): 12-bit B-type immediate
            // R_RISCV_CALL (18) / R_RISCV_CALL_PLT (19): AUIPC+JALR pair
            const RV_BRANCH: u32 = 16;
            const RV_JAL: u32 = 17;
            const RV_CALL: u32 = 18;
            const RV_CALL_PLT: u32 = 19;

            match rel_type {
                RV_CALL | RV_CALL_PLT => {
                    // AUIPC + JALR pair (8 bytes total)
                    if off + 8 <= text.len() {
                        let hi = ((value as i32 + 0x800) >> 12) & 0xFFFFF;
                        let lo = ((value as i32) & 0xFFF) as u32;
                        let auipc = u32::from_le_bytes([
                            text[off],
                            text[off + 1],
                            text[off + 2],
                            text[off + 3],
                        ]);
                        let jalr = u32::from_le_bytes([
                            text[off + 4],
                            text[off + 5],
                            text[off + 6],
                            text[off + 7],
                        ]);
                        let p_auipc = (auipc & 0xFFF) | ((hi as u32) << 12);
                        let p_jalr = (jalr & 0x000F_FFFF) | (lo << 20);
                        text[off..off + 4].copy_from_slice(&p_auipc.to_le_bytes());
                        text[off + 4..off + 8].copy_from_slice(&p_jalr.to_le_bytes());
                    }
                }
                RV_JAL => {
                    // J-type: imm[20|10:1|11|19:12] in bits [31:12]
                    let inst = u32::from_le_bytes([
                        text[off],
                        text[off + 1],
                        text[off + 2],
                        text[off + 3],
                    ]);
                    let v = value as u32;
                    let enc = ((v & 0x100000) << 11) // imm[20] -> bit 31
                        | ((v & 0x7FE) << 20)        // imm[10:1] -> bits [30:21]
                        | ((v & 0x800) << 9)          // imm[11] -> bit 20
                        | (v & 0xFF000); // imm[19:12] -> bits [19:12]
                    let patched = (inst & 0xFFF) | enc;
                    text[off..off + 4].copy_from_slice(&patched.to_le_bytes());
                }
                RV_BRANCH => {
                    // B-type: imm[12|10:5] in [31:25], imm[4:1|11] in [11:7]
                    let inst = u32::from_le_bytes([
                        text[off],
                        text[off + 1],
                        text[off + 2],
                        text[off + 3],
                    ]);
                    let v = value as u32;
                    let hi = ((v & 0x1000) << 19) | ((v & 0x7E0) << 20);
                    let lo = ((v & 0x1E) << 7) | ((v & 0x800) >> 4);
                    let patched = (inst & 0x01FFF07F) | hi | lo;
                    text[off..off + 4].copy_from_slice(&patched.to_le_bytes());
                }
                _ => {
                    // Fallback: raw 4-byte write
                    let bytes = (value as i32).to_le_bytes();
                    text[off..off + 4].copy_from_slice(&bytes);
                }
            }
        }
    }
}

/// Writes a complete relocatable ELF object file (`.o`) from the compiled
/// code sections, global data, and optional DWARF debug sections.
fn write_relocatable_object(
    text_data: &[u8],
    data_data: &[u8],
    rodata_data: &[u8],
    bss_size: u64,
    functions: &[CompiledFunction],
    globals: &[GlobalSymbolInfo],
    text_relocs: &[crate::backend::traits::FunctionRelocation],
    data_relocs: &[(u64, String, bool, i64)],
    dwarf: &Option<DwarfSections>,
    module: &IrModule,
    ctx: &CodegenContext,
    _diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let target = &ctx.target;
    let mut writer = ElfWriter::new(*target, ET_REL);

    // Log the ELF machine type for this target (e.g., EM_X86_64, EM_AARCH64).
    let _elf_machine = target.elf_machine();
    let _is_64 = target.is_64bit();

    // Track section indices dynamically — index 0 is the NULL section, so
    // user sections start at index 1.
    let mut next_section_index: u16 = 1;
    let mut text_section_index: u16 = 0;
    let mut data_section_index: u16 = 0;
    let mut rodata_section_index: u16 = 0;
    let mut bss_section_index: u16 = 0;

    // --- .text section ---
    if !text_data.is_empty() {
        text_section_index = next_section_index;
        next_section_index += 1;
        let text_section = Section {
            name: ".text".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: text_data.to_vec(),
            sh_addralign: 16,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(text_section);
    }

    // --- .data section ---
    if !data_data.is_empty() {
        data_section_index = next_section_index;
        next_section_index += 1;
        let data_sec = Section {
            name: ".data".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: data_data.to_vec(),
            sh_addralign: 8,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(data_sec);
    }

    // --- .rodata section ---
    if !rodata_data.is_empty() {
        rodata_section_index = next_section_index;
        next_section_index += 1;
        let rodata_sec = Section {
            name: ".rodata".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC,
            data: rodata_data.to_vec(),
            sh_addralign: 8,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(rodata_sec);
    }

    // --- .bss section ---
    if bss_size > 0 {
        bss_section_index = next_section_index;
        next_section_index += 1;
        let bss_sec = Section {
            name: ".bss".to_string(),
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: Vec::new(),
            sh_addralign: 8,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: bss_size,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(bss_sec);
    }
    let _ = next_section_index; // suppress unused warning

    // --- DWARF debug sections (conditional on -g) ---
    if let Some(ref dwarf_data) = dwarf {
        add_dwarf_sections(&mut writer, dwarf_data);
    }

    // --- .note.gnu.property section for CET/IBT (x86-64 only) ---
    // When -fcf-protection is enabled, emit the note section so that the
    // runtime loader can verify CET support in all linked object files.
    if ctx.cf_protection && ctx.target == Target::X86_64 {
        let note_data = crate::backend::x86_64::security::generate_cet_note_section();
        let note_section = Section {
            name: ".note.gnu.property".to_string(),
            sh_type: 7, // SHT_NOTE
            sh_flags: SHF_ALLOC,
            data: note_data,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(note_section);
    }

    // --- Symbol table entries ---

    // Add function symbols with proper visibility and binding from the IR.
    // Weak functions (e.g. __attribute__((weak))) get STB_WEAK binding so
    // the static linker can override them with strong definitions.
    for func in functions {
        let sym = ElfSymbol {
            name: func.name.clone(),
            value: func.offset,
            size: func.size,
            binding: func.elf_binding,
            sym_type: STT_FUNC,
            visibility: func.visibility,
            section_index: text_section_index,
        };
        writer.add_symbol(sym);
    }

    // Add global variable symbols.
    for gsym in globals {
        let binding = if gsym.is_global {
            STB_GLOBAL
        } else {
            STB_LOCAL
        };
        let sym_type = if gsym.name.starts_with(".L") {
            STT_NOTYPE // Local label / string literal
        } else {
            STT_OBJECT
        };
        let section_index = match gsym.section {
            SectionPlacement::Data => data_section_index,
            SectionPlacement::Rodata => rodata_section_index,
            SectionPlacement::Bss => bss_section_index,
        };
        let sym = ElfSymbol {
            name: gsym.name.clone(),
            value: gsym.offset,
            size: gsym.size,
            binding,
            sym_type,
            visibility: STV_DEFAULT,
            section_index,
        };
        writer.add_symbol(sym);
    }

    // Add external function declaration symbols (undefined).
    // Mark them as STT_FUNC so the dynamic linker creates PLT entries
    // for lazy binding of external function calls (e.g. printf, malloc).
    //
    // IMPORTANT: Skip declarations for functions that already have a
    // definition in this translation unit (i.e., already in `functions`).
    // C programs commonly forward-declare static functions before defining
    // them; emitting both a LOCAL defined symbol AND a GLOBAL undefined
    // symbol for the same name confuses the system linker.
    {
        let defined_funcs: crate::common::fx_hash::FxHashSet<&str> =
            functions.iter().map(|f| f.name.as_str()).collect();
        for decl in module.declarations() {
            if defined_funcs.contains(decl.name.as_str()) {
                continue; // Already defined — skip the undefined entry.
            }
            let sym = ElfSymbol {
                name: decl.name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF
            };
            writer.add_symbol(sym);
        }
    }

    // Add external variable declaration symbols (undefined).
    // These are `extern` declarations with no definition in this TU.
    // They must appear as SHN_UNDEF STT_OBJECT entries so the linker
    // resolves them against definitions in other translation units.
    //
    // IMPORTANT: Skip any name that already has a defined symbol entry
    // (i.e., appears in `functions` as a compiled function or in `globals`
    // as a defined global variable).  The IR may spuriously list function
    // names as non-definition globals when they are referenced as function
    // pointers.  Emitting a GLOBAL UNDEF for an already-LOCAL-defined
    // symbol confuses the linker and causes "undefined reference" errors.
    {
        let mut defined_names: crate::common::fx_hash::FxHashSet<String> = functions
            .iter()
            .map(|f| f.name.clone())
            .chain(globals.iter().map(|g| g.name.clone()))
            .collect();
        // Alias names are defined by the alias emission code — do not
        // emit them as undefined extern var declarations.
        for (alias_name, _) in &module.symbol_aliases {
            defined_names.insert(alias_name.clone());
        }
        let mut seen_extern_vars: crate::common::fx_hash::FxHashSet<String> =
            crate::common::fx_hash::FxHashSet::default();
        for global in module.globals() {
            if !global.is_definition
                && !defined_names.contains(&global.name)
                && seen_extern_vars.insert(global.name.clone())
            {
                let binding = match global.linkage {
                    crate::ir::module::Linkage::Weak => STB_WEAK,
                    _ => STB_GLOBAL,
                };
                let sym = ElfSymbol {
                    name: global.name.clone(),
                    value: 0,
                    size: 0,
                    binding,
                    sym_type: STT_NOTYPE,
                    visibility: STV_DEFAULT,
                    section_index: 0, // SHN_UNDEF
                };
                writer.add_symbol(sym);
            }
        }
    }

    // --- Symbol aliases: __attribute__((alias("target"))) ---
    // Emit alias symbols that share the same address as their target.
    // Both function and variable aliases are supported.
    for (alias_name, target_name) in &module.symbol_aliases {
        // Find the target symbol among functions or globals.
        let mut found = false;
        for func in functions {
            if func.name == *target_name {
                let sym = ElfSymbol {
                    name: alias_name.clone(),
                    value: func.offset,
                    size: func.size,
                    binding: STB_GLOBAL,
                    sym_type: STT_FUNC,
                    visibility: STV_DEFAULT,
                    section_index: text_section_index,
                };
                writer.add_symbol(sym);
                found = true;
                break;
            }
        }
        if !found {
            for gsym in globals {
                if gsym.name == *target_name {
                    let section_index = match gsym.section {
                        SectionPlacement::Data => data_section_index,
                        SectionPlacement::Rodata => rodata_section_index,
                        SectionPlacement::Bss => bss_section_index,
                    };
                    let sym = ElfSymbol {
                        name: alias_name.clone(),
                        value: gsym.offset,
                        size: gsym.size,
                        binding: STB_GLOBAL,
                        sym_type: STT_OBJECT,
                        visibility: STV_DEFAULT,
                        section_index,
                    };
                    writer.add_symbol(sym);
                    break;
                }
            }
        }
    }

    // --- .rela.text section (relocations for the .text section) ---
    // Encode the assembler-generated relocations into a proper ELF RELA section
    // so that the linker can patch up cross-section and external symbol references.
    if !text_relocs.is_empty() && text_section_index > 0 {
        let is_64 = ctx.target.is_64bit();

        // For any symbol referenced by a relocation that isn't already in the
        // symbol table (functions, globals, declarations), add it as an
        // undefined (SHN_UNDEF) global symbol — EXCEPT for RISC-V local
        // PCREL labels (.Lpcrel_hi<offset>) which must be LOCAL symbols
        // pointing to the corresponding AUIPC instruction offset in .text.
        //
        // RISC-V PCREL_LO12 relocations reference a local label at the
        // AUIPC instruction. The label name encodes the per-function offset
        // (.Lpcrel_hi{N}), but we need to find the matching PCREL_HI20
        // relocation to determine the absolute .text offset for the symbol.
        //
        // Step 1: Build a map from label name → absolute .text offset by
        //         scanning PCREL_HI20 relocations (type 23 for RISCV).
        let mut pcrel_label_to_abs_offset: FxHashMap<String, u64> = FxHashMap::default();
        {
            // R_RISCV_PCREL_HI20 = 23, R_RISCV_GOT_HI20 = 20
            for reloc in text_relocs {
                if reloc.rel_type_id == 23 || reloc.rel_type_id == 20 {
                    // This HI20 relocation is at absolute offset reloc.offset.
                    // The corresponding local label is ".Lpcrel_hi{reloc.offset}".
                    // BUT: the label name was computed from per-function offsets
                    // before adjustment. We need to check what label name was
                    // generated. The LO12 referencing this HI20 uses a label
                    // with the per-function offset. However, after offset
                    // adjustment (reloc.offset += func_offset), the HI20 has
                    // the absolute offset.
                    //
                    // Strategy: we don't know which label name maps to this
                    // HI20 directly. Instead, record the absolute offset of
                    // every HI20 relocation and match by scanning LO12s.
                    let label = format!(".Lpcrel_hi{}", reloc.offset);
                    pcrel_label_to_abs_offset.insert(label, reloc.offset);
                }
            }
        }

        let mut extra_undef_symbols: Vec<String> = Vec::new();
        let mut pcrel_local_labels: Vec<(String, u64)> = Vec::new();
        {
            let mut known: crate::common::fx_hash::FxHashSet<String> =
                crate::common::fx_hash::FxHashSet::default();
            for func in functions.iter() {
                known.insert(func.name.clone());
            }
            for gsym in globals.iter() {
                known.insert(gsym.name.clone());
            }
            for decl in module.declarations() {
                known.insert(decl.name.clone());
            }
            // Include extern variable declarations (non-definition globals)
            // so that relocations referencing them are NOT duplicated into
            // extra_undef_symbols.  These symbols are already added to the
            // ELF writer in the extern-var-declaration loop above.
            for global in module.globals() {
                if !global.is_definition {
                    known.insert(global.name.clone());
                }
            }
            // Include alias names — they are emitted as defined symbols
            // by the alias emission code above, so must not appear as
            // extra undefined symbols.
            for (alias_name, _) in &module.symbol_aliases {
                known.insert(alias_name.clone());
            }
            for reloc in text_relocs {
                if !known.contains(&reloc.symbol) {
                    known.insert(reloc.symbol.clone());
                    // Check if this is a RISC-V PCREL local label
                    if reloc.symbol.starts_with(".Lpcrel_hi") {
                        if let Some(&abs_offset) = pcrel_label_to_abs_offset.get(&reloc.symbol) {
                            pcrel_local_labels.push((reloc.symbol.clone(), abs_offset));
                        } else {
                            // Label not found in HI20 map — might be a naming
                            // mismatch. Try parsing the offset from the name.
                            if let Some(offset_str) = reloc.symbol.strip_prefix(".Lpcrel_hi") {
                                if let Ok(offset) = offset_str.parse::<u64>() {
                                    pcrel_local_labels.push((reloc.symbol.clone(), offset));
                                } else {
                                    extra_undef_symbols.push(reloc.symbol.clone());
                                }
                            } else {
                                extra_undef_symbols.push(reloc.symbol.clone());
                            }
                        }
                    } else {
                        extra_undef_symbols.push(reloc.symbol.clone());
                    }
                }
            }
        }
        // Add PCREL local labels as STB_LOCAL symbols in .text section
        for (name, offset) in &pcrel_local_labels {
            let sym = ElfSymbol {
                name: name.clone(),
                value: *offset,
                size: 0,
                binding: STB_LOCAL,
                sym_type: STT_NOTYPE,
                visibility: STV_DEFAULT,
                section_index: text_section_index,
            };
            writer.add_symbol(sym);
        }
        // Build a set of symbols that have CALL/PLT-type relocations so we
        // can mark them as STT_FUNC instead of STT_NOTYPE.  Without this,
        // the linker's PLT stub builder (which requires STT_FUNC) may skip
        // these symbols, causing all BL/CALL offsets to be wrong.
        //
        // Call-type relocation IDs per architecture:
        //   AArch64: CALL26=283, Jump26=282
        //   x86-64 : PLT32=4
        //   i686   : PLT32=4
        //   RISC-V : CALL=18, CALL_PLT=19
        let call_reloc_symbols: crate::common::fx_hash::FxHashSet<String> = {
            let mut s = crate::common::fx_hash::FxHashSet::default();
            for reloc in text_relocs {
                let is_call = match ctx.target {
                    crate::common::target::Target::AArch64 => {
                        reloc.rel_type_id == 283 || reloc.rel_type_id == 282
                    }
                    crate::common::target::Target::X86_64 | crate::common::target::Target::I686 => {
                        reloc.rel_type_id == 4
                    }
                    crate::common::target::Target::RiscV64 => {
                        reloc.rel_type_id == 18 || reloc.rel_type_id == 19
                    }
                };
                if is_call {
                    s.insert(reloc.symbol.clone());
                }
            }
            s
        };
        for name in &extra_undef_symbols {
            let sym = ElfSymbol {
                name: name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: if call_reloc_symbols.contains(name.as_str()) {
                    STT_FUNC
                } else {
                    STT_NOTYPE
                },
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF
            };
            writer.add_symbol(sym);
        }

        // Build a name→ELF-symbol-index map that matches the writer's
        // internal ordering: locals first (index 1..), then globals.
        // This mirrors the partitioning in SymbolTable::build_bytes().
        // IMPORTANT: Must use the same declaration filtering as the
        // symbol-adding loop above (skip declarations already defined
        // as functions).
        struct SymEntry {
            name: String,
            binding: u8,
        }
        let rela_defined_funcs: crate::common::fx_hash::FxHashSet<&str> =
            functions.iter().map(|f| f.name.as_str()).collect();
        let mut all_syms: Vec<SymEntry> = Vec::new();
        for func in functions.iter() {
            all_syms.push(SymEntry {
                name: func.name.clone(),
                binding: func.elf_binding,
            });
        }
        for gsym in globals.iter() {
            let b = if gsym.is_global {
                STB_GLOBAL
            } else {
                STB_LOCAL
            };
            all_syms.push(SymEntry {
                name: gsym.name.clone(),
                binding: b,
            });
        }
        for decl in module.declarations() {
            if rela_defined_funcs.contains(decl.name.as_str()) {
                continue; // Filtered — already defined as a function.
            }
            all_syms.push(SymEntry {
                name: decl.name.clone(),
                binding: STB_GLOBAL,
            });
        }
        // Include extern variable declarations (non-definition globals)
        // in the same order they were added to the writer.  This ensures
        // the index map matches the writer's internal symbol ordering.
        // Use a seen-set to avoid duplicating symbols already covered by
        // the `globals` parameter (which contains defined globals only).
        // Also skip any name already present in `functions` or `globals`
        // (the IR may list function-pointer references as non-definition
        // globals, but those already have defined symbol entries).
        {
            let mut defined_names_for_extern: crate::common::fx_hash::FxHashSet<String> = functions
                .iter()
                .map(|f| f.name.clone())
                .chain(globals.iter().map(|g| g.name.clone()))
                .collect();
            for (alias_name, _) in &module.symbol_aliases {
                defined_names_for_extern.insert(alias_name.clone());
            }
            let mut seen_extern_vars: crate::common::fx_hash::FxHashSet<String> =
                crate::common::fx_hash::FxHashSet::default();
            for global in module.globals() {
                if !global.is_definition
                    && !defined_names_for_extern.contains(&global.name)
                    && seen_extern_vars.insert(global.name.clone())
                {
                    let binding = match global.linkage {
                        crate::ir::module::Linkage::Weak => STB_WEAK,
                        _ => STB_GLOBAL,
                    };
                    all_syms.push(SymEntry {
                        name: global.name.clone(),
                        binding,
                    });
                }
            }
        }
        // Symbol aliases — added to writer as defined symbols.
        for (alias_name, _) in &module.symbol_aliases {
            all_syms.push(SymEntry {
                name: alias_name.clone(),
                binding: STB_GLOBAL,
            });
        }
        // RISC-V PCREL local labels are LOCAL symbols
        for (name, _offset) in &pcrel_local_labels {
            all_syms.push(SymEntry {
                name: name.clone(),
                binding: STB_LOCAL,
            });
        }
        for name in &extra_undef_symbols {
            all_syms.push(SymEntry {
                name: name.clone(),
                binding: STB_GLOBAL,
            });
        }

        // Separate into locals and globals, maintaining relative order.
        let locals: Vec<&SymEntry> = all_syms.iter().filter(|s| s.binding == STB_LOCAL).collect();
        let globals_list: Vec<&SymEntry> =
            all_syms.iter().filter(|s| s.binding != STB_LOCAL).collect();

        let mut sym_name_to_idx: FxHashMap<String, u32> = FxHashMap::default();
        // Index 0 = null symbol.  Locals start at index 1.
        for (i, sym) in locals.iter().enumerate() {
            sym_name_to_idx.insert(sym.name.clone(), (i as u32) + 1);
        }
        // Globals start after all locals.
        let global_base = (locals.len() as u32) + 1;
        for (i, sym) in globals_list.iter().enumerate() {
            sym_name_to_idx.insert(sym.name.clone(), global_base + (i as u32));
        }

        // Encode relocation entries into binary RELA format.
        let mut rela_data: Vec<u8> = Vec::new();
        for reloc in text_relocs {
            let sym_idx = sym_name_to_idx.get(&reloc.symbol).copied().unwrap_or(0);

            if is_64 {
                let rela = Elf64Rela::new(reloc.offset, sym_idx, reloc.rel_type_id, reloc.addend);
                rela_data.extend_from_slice(&rela.to_bytes());
            } else {
                // For 32-bit targets, use Elf32Rel format.
                let rel = Elf32Rel::new(reloc.offset as u32, sym_idx, reloc.rel_type_id as u8);
                rela_data.extend_from_slice(&rel.to_bytes());
            }
        }

        if !rela_data.is_empty() {
            // Pre-compute the .symtab section index.  ElfWriter::write()
            // places .symtab right after all user sections at index
            // (total_user + 1).  We must count ALL rela sections that will
            // be added (not just the current one) so the sh_link value is
            // correct for every .rela.* section.
            let rela_count = {
                let mut c = 1u32; // This .rela.text section
                let has_rela_data = data_relocs.iter().any(|(_, _, is_ro, _)| !*is_ro);
                let has_rela_rodata = data_relocs.iter().any(|(_, _, is_ro, _)| *is_ro);
                if has_rela_data {
                    c += 1;
                }
                if has_rela_rodata {
                    c += 1;
                }
                c
            };
            let symtab_section_idx_val = (writer.sections_count() as u32) + rela_count + 1;
            let rela_section = Section {
                name: if is_64 {
                    ".rela.text".to_string()
                } else {
                    ".rel.text".to_string()
                },
                sh_type: if is_64 { SHT_RELA } else { SHT_REL },
                sh_flags: SHF_INFO_LINK,
                data: rela_data,
                sh_addralign: if is_64 { 8 } else { 4 },
                sh_entsize: if is_64 { 24 } else { 8 }, // sizeof(Elf64_Rela) or sizeof(Elf32_Rel)
                sh_link: symtab_section_idx_val,        // Points to .symtab
                sh_info: text_section_index as u32,     // Section being relocated (.text)
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(rela_section);
        }
    }

    // --- .rela.data / .rela.rodata sections (relocations for data sections) ---
    // GlobalRef constants in global variable initializers produce address
    // entries that the linker must patch (e.g., a function pointer stored in
    // a global struct).  Each entry generates an R_X86_64_64 (or
    // R_AARCH64_ABS64, R_RISCV_64, R_386_32) relocation.
    if !data_relocs.is_empty() {
        let is_64 = ctx.target.is_64bit();

        // Determine the absolute-address relocation type for this target.
        let abs_reloc_type: u32 = match ctx.target {
            Target::X86_64 => 1,    // R_X86_64_64
            Target::I686 => 1,      // R_386_32
            Target::AArch64 => 257, // R_AARCH64_ABS64
            Target::RiscV64 => 2,   // R_RISCV_64
        };

        // Ensure all referenced symbols exist in the symbol table. Build a
        // name→index map reusing the same partitioning logic as .rela.text
        // (locals first, then globals, starting from index 1 after the null
        // symbol at index 0).
        //
        // First, collect all existing symbol names.
        let mut known_syms: crate::common::fx_hash::FxHashSet<String> =
            crate::common::fx_hash::FxHashSet::default();
        for func in functions.iter() {
            known_syms.insert(func.name.clone());
        }
        for gsym in globals.iter() {
            known_syms.insert(gsym.name.clone());
        }
        for decl in module.declarations() {
            known_syms.insert(decl.name.clone());
        }
        // Include extern variable declarations (non-definition globals).
        for global in module.globals() {
            if !global.is_definition {
                known_syms.insert(global.name.clone());
            }
        }
        // Include alias names — defined by alias emission code.
        for (alias_name, _) in &module.symbol_aliases {
            known_syms.insert(alias_name.clone());
        }
        // Add any symbols from text relocs that were already added.
        for reloc in text_relocs {
            known_syms.insert(reloc.symbol.clone());
        }

        // Add undefined symbols for any data reloc targets not yet known.
        let mut data_extra_undefs: Vec<String> = Vec::new();
        for (_, sym_name, _, _) in data_relocs {
            if !known_syms.contains(sym_name) {
                known_syms.insert(sym_name.clone());

                data_extra_undefs.push(sym_name.clone());
            }
        }
        for name in &data_extra_undefs {
            let sym = ElfSymbol {
                name: name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_NOTYPE,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF
            };
            writer.add_symbol(sym);
        }

        // Build name→ELF-symbol-index map (same partitioning as .rela.text).
        // IMPORTANT: Must mirror the EXACT same filtering as the main
        // symbol-adding loop (declarations already defined as functions
        // are skipped).
        struct DataSymEntry {
            name: String,
            binding: u8,
        }
        let data_defined_funcs: crate::common::fx_hash::FxHashSet<&str> =
            functions.iter().map(|f| f.name.as_str()).collect();
        let mut all_syms: Vec<DataSymEntry> = Vec::new();
        for func in functions.iter() {
            all_syms.push(DataSymEntry {
                name: func.name.clone(),
                binding: func.elf_binding,
            });
        }
        for gsym in globals.iter() {
            let b = if gsym.is_global {
                STB_GLOBAL
            } else {
                STB_LOCAL
            };
            all_syms.push(DataSymEntry {
                name: gsym.name.clone(),
                binding: b,
            });
        }
        for decl in module.declarations() {
            if data_defined_funcs.contains(decl.name.as_str()) {
                continue; // Filtered — already defined as a function.
            }
            all_syms.push(DataSymEntry {
                name: decl.name.clone(),
                binding: STB_GLOBAL,
            });
        }
        // Include extern variable declarations (non-definition globals).
        // Skip names already defined as functions or global variables.
        {
            let mut data_defined_names: crate::common::fx_hash::FxHashSet<String> = functions
                .iter()
                .map(|f| f.name.clone())
                .chain(globals.iter().map(|g| g.name.clone()))
                .collect();
            for (alias_name, _) in &module.symbol_aliases {
                data_defined_names.insert(alias_name.clone());
            }
            let mut seen_extern_vars: crate::common::fx_hash::FxHashSet<String> =
                crate::common::fx_hash::FxHashSet::default();
            for global in module.globals() {
                if !global.is_definition
                    && !data_defined_names.contains(&global.name)
                    && seen_extern_vars.insert(global.name.clone())
                {
                    let binding = match global.linkage {
                        crate::ir::module::Linkage::Weak => STB_WEAK,
                        _ => STB_GLOBAL,
                    };
                    all_syms.push(DataSymEntry {
                        name: global.name.clone(),
                        binding,
                    });
                }
            }
        }
        // Symbol aliases — added to writer as defined symbols.
        for (alias_name, _) in &module.symbol_aliases {
            all_syms.push(DataSymEntry {
                name: alias_name.clone(),
                binding: STB_GLOBAL,
            });
        }
        // Include extra undef symbols from text relocs (if any were added).
        {
            let mut text_extra: crate::common::fx_hash::FxHashSet<String> =
                crate::common::fx_hash::FxHashSet::default();
            for func in functions.iter() {
                text_extra.insert(func.name.clone());
            }
            for gsym in globals.iter() {
                text_extra.insert(gsym.name.clone());
            }
            for decl in module.declarations() {
                if data_defined_funcs.contains(decl.name.as_str()) {
                    continue; // Filtered — already defined as a function.
                }
                text_extra.insert(decl.name.clone());
            }
            // Include extern variable declarations.
            for global in module.globals() {
                if !global.is_definition {
                    text_extra.insert(global.name.clone());
                }
            }
            // Include alias names.
            for (alias_name, _) in &module.symbol_aliases {
                text_extra.insert(alias_name.clone());
            }
            for reloc in text_relocs {
                if text_extra.insert(reloc.symbol.clone()) {
                    all_syms.push(DataSymEntry {
                        name: reloc.symbol.clone(),
                        binding: STB_GLOBAL,
                    });
                }
            }
        }
        for name in &data_extra_undefs {
            all_syms.push(DataSymEntry {
                name: name.clone(),
                binding: STB_GLOBAL,
            });
        }

        let locals: Vec<&DataSymEntry> =
            all_syms.iter().filter(|s| s.binding == STB_LOCAL).collect();
        let globals_list: Vec<&DataSymEntry> =
            all_syms.iter().filter(|s| s.binding != STB_LOCAL).collect();
        let mut sym_name_to_idx: FxHashMap<String, u32> = FxHashMap::default();
        for (i, sym) in locals.iter().enumerate() {
            sym_name_to_idx.insert(sym.name.clone(), (i as u32) + 1);
        }
        let global_base = (locals.len() as u32) + 1;
        for (i, sym) in globals_list.iter().enumerate() {
            sym_name_to_idx.insert(sym.name.clone(), global_base + (i as u32));
        }

        // Separate data relocs by target section (.data vs .rodata).
        let mut rela_data_bytes: Vec<u8> = Vec::new();
        let mut rela_rodata_bytes: Vec<u8> = Vec::new();

        for (offset, sym_name, is_rodata, addend) in data_relocs {
            let sym_idx = sym_name_to_idx.get(sym_name).copied().unwrap_or(0);
            let rela_buf = if *is_rodata {
                &mut rela_rodata_bytes
            } else {
                &mut rela_data_bytes
            };

            if is_64 {
                let rela = Elf64Rela::new(*offset, sym_idx, abs_reloc_type, *addend);
                rela_buf.extend_from_slice(&rela.to_bytes());
            } else {
                let rel = Elf32Rel::new(*offset as u32, sym_idx, abs_reloc_type as u8);
                rela_buf.extend_from_slice(&rel.to_bytes());
            }
        }

        // Pre-compute the .symtab section index for data/rodata relocs.
        // .symtab is auto-generated at index (total_user_sections + 1).
        // Count remaining rela sections to be added:
        let data_rela_remaining = {
            let mut c = 0u32;
            if !rela_data_bytes.is_empty() && data_section_index > 0 {
                c += 1;
            }
            if !rela_rodata_bytes.is_empty() && rodata_section_index > 0 {
                c += 1;
            }
            c
        };
        let data_symtab_idx = (writer.sections_count() as u32) + data_rela_remaining + 1;

        // Emit .rela.data section.
        if !rela_data_bytes.is_empty() && data_section_index > 0 {
            let rela_section = Section {
                name: if is_64 {
                    ".rela.data".to_string()
                } else {
                    ".rel.data".to_string()
                },
                sh_type: if is_64 { SHT_RELA } else { SHT_REL },
                sh_flags: SHF_INFO_LINK,
                data: rela_data_bytes,
                sh_addralign: if is_64 { 8 } else { 4 },
                sh_entsize: if is_64 { 24 } else { 8 },
                sh_link: data_symtab_idx,
                sh_info: data_section_index as u32,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(rela_section);
        }

        // Emit .rela.rodata section.
        if !rela_rodata_bytes.is_empty() && rodata_section_index > 0 {
            let rela_section = Section {
                name: if is_64 {
                    ".rela.rodata".to_string()
                } else {
                    ".rel.rodata".to_string()
                },
                sh_type: if is_64 { SHT_RELA } else { SHT_REL },
                sh_flags: SHF_INFO_LINK,
                data: rela_rodata_bytes,
                sh_addralign: if is_64 { 8 } else { 4 },
                sh_entsize: if is_64 { 24 } else { 8 },
                sh_link: data_symtab_idx,
                sh_info: rodata_section_index as u32,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(rela_section);
        }
    }

    // Serialize the ELF object.
    let elf_bytes = writer.write();
    Ok(elf_bytes)
}

/// Adds DWARF debug sections to the ELF writer.
///
/// Called only when `-g` is active.  Adds `.debug_info`, `.debug_abbrev`,
/// `.debug_line`, and `.debug_str` as non-allocable (SHF_ALLOC not set)
/// sections.
fn add_dwarf_sections(writer: &mut ElfWriter, dwarf: &DwarfSections) {
    // All DWARF sections are PROGBITS, no flags (not loaded into memory).
    let debug_sections = [
        (".debug_info", &dwarf.debug_info),
        (".debug_abbrev", &dwarf.debug_abbrev),
        (".debug_line", &dwarf.debug_line),
        (".debug_str", &dwarf.debug_str),
    ];

    for (name, data) in &debug_sections {
        if !data.is_empty() {
            let sec = Section {
                name: name.to_string(),
                sh_type: SHT_PROGBITS,
                sh_flags: 0, // Debug sections are NOT loaded (no SHF_ALLOC)
                data: data.to_vec(),
                sh_addralign: 1,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(sec);
        }
    }
}

// ===========================================================================
// link_to_final_output — invoke the built-in linker
// ===========================================================================

/// Invokes the built-in linker to produce the final ELF binary from a
/// relocatable object file.
///
/// The linker is architecture-specific and handles:
/// - Symbol resolution (strong/weak binding)
/// - Section merging and address layout
/// - Relocation processing
/// - Dynamic section generation (for shared libraries)
///
/// No external linker is invoked — this is the standalone backend mode.
fn link_to_final_output(
    object_bytes: &[u8],
    text_relocations: &[crate::backend::traits::FunctionRelocation],
    data_relocations: &[(u64, String, bool, i64)],
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let output_type = ctx.output_type();

    // For relocatable output, just return the object bytes.
    if output_type == OutputType::Relocatable {
        return Ok(object_bytes.to_vec());
    }

    // Configure the linker.
    let mut config = LinkerConfig::new(ctx.target, output_type);
    config.output_path = ctx.output_path.clone();
    config.pic = ctx.pic;
    config.emit_debug = ctx.should_emit_dwarf();

    if ctx.shared {
        config.allow_undefined = true;
    }

    // Implicitly link against libc for executables — equivalent to GCC's
    // default `-lc`.  This is the standard behaviour: unless the user
    // explicitly passes `-nostdlib` or `-nodefaultlibs`, every C program
    // is linked against the C standard library.  The built-in linker
    // produces a dynamically-linked executable with `libc.so.6` as a
    // `DT_NEEDED` entry and `PT_INTERP` pointing to the system dynamic
    // linker.  Undefined symbols from libc (printf, malloc, etc.) are
    // resolved at runtime by `ld-linux`.
    // Add user-specified -l libraries as DT_NEEDED entries.
    config.library_paths = ctx.library_paths.clone();
    config.libraries = ctx.libraries.clone();
    let standard_lib_dirs: &[&str] = &[
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ];
    for lib_name in &ctx.libraries {
        let base_so = format!("lib{}.so", lib_name);
        let mut final_so = base_so.clone();
        // Search library paths + standard dirs to resolve the soname.
        let search_dirs: Vec<&str> = ctx
            .library_paths
            .iter()
            .map(|s| s.as_str())
            .chain(standard_lib_dirs.iter().copied())
            .collect();
        'dir_search: for dir in &search_dirs {
            let candidate = std::path::Path::new(dir).join(&base_so);
            if !candidate.exists() {
                continue;
            }
            // Check if the file is a real ELF.
            if let Ok(bytes) = std::fs::read(&candidate) {
                if bytes.len() >= 4 && &bytes[..4] == b"\x7fELF" {
                    // Real ELF — use its filename as DT_NEEDED soname.
                    // Could also extract DT_SONAME here, but versioned
                    // name is the standard convention.
                    final_so = base_so.clone();
                    break;
                }
                // Not ELF (linker script) — try versioned variants.
                for ver in &["6", "2", "1", "0"] {
                    let versioned = format!("lib{}.so.{}", lib_name, ver);
                    let vpath = std::path::Path::new(dir).join(&versioned);
                    if vpath.exists() {
                        final_so = versioned;
                        break 'dir_search;
                    }
                }
            }
        }
        if !config.needed_libs.contains(&final_so) {
            config.needed_libs.push(final_so);
        }
    }

    if output_type == OutputType::Executable {
        // Add libc if not already present.
        if !config.needed_libs.iter().any(|n| n.starts_with("libc.so")) {
            config.needed_libs.push("libc.so.6".to_string());
        }
        // Add libm for math functions (floor, fabs, pow, sqrt, etc.).
        if !config.needed_libs.iter().any(|n| n.starts_with("libm.so")) {
            config.needed_libs.push("libm.so.6".to_string());
        }
        // Add libgcc_s for compiler runtime helpers (__clzsi2, __divdi3, etc.).
        if !config
            .needed_libs
            .iter()
            .any(|n| n.starts_with("libgcc_s.so"))
        {
            config.needed_libs.push("libgcc_s.so.1".to_string());
        }
        // Allow undefined symbols — they will be resolved at runtime by
        // the dynamic linker via the DT_NEEDED entry.
        config.allow_undefined = true;
    }

    // Parse the relocatable ELF object bytes back into linker internal
    // representation: sections, symbols, and relocations.
    let mut input = parse_elf_object_to_linker_input(0, "input.o", object_bytes);

    // Convert unresolved text relocations (e.g. global variable references)
    // from the code generator into linker-format Relocation entries so the
    // linker can resolve cross-section references (.text → .data/.rodata/.bss).
    for func_reloc in text_relocations {
        input
            .relocations
            .push(crate::backend::linker_common::relocation::Relocation {
                offset: func_reloc.offset,
                symbol_name: func_reloc.symbol.clone(),
                sym_index: 0, // resolved by name, not index
                rel_type: func_reloc.rel_type_id,
                addend: func_reloc.addend,
                object_id: 0,
                section_index: 1, // .text section index in our .o files
                output_section_name: Some(".text".to_string()),
            });
    }

    // Convert data-section relocations (GlobalRef entries in global
    // initializers) into linker-format Relocation entries so the linker
    // patches absolute addresses in .data/.rodata sections.
    {
        // Determine the absolute-address relocation type for this target.
        let abs_reloc_type: u32 = match ctx.target {
            Target::X86_64 => 1,    // R_X86_64_64
            Target::I686 => 1,      // R_386_32
            Target::AArch64 => 257, // R_AARCH64_ABS64
            Target::RiscV64 => 2,   // R_RISCV_64
        };

        // Find the section indices for .data and .rodata in the parsed input.
        let data_sec_idx = input
            .sections
            .iter()
            .position(|s| s.name == ".data")
            .map(|i| (i + 1) as u32) // +1 because section 0 is null
            .unwrap_or(0);
        let rodata_sec_idx = input
            .sections
            .iter()
            .position(|s| s.name == ".rodata")
            .map(|i| (i + 1) as u32)
            .unwrap_or(0);

        for (offset, sym_name, is_rodata, addend) in data_relocations {
            let sec_idx = if *is_rodata {
                rodata_sec_idx
            } else {
                data_sec_idx
            };
            let sec_name = if *is_rodata { ".rodata" } else { ".data" };
            if sec_idx == 0 {
                continue; // Section not found — skip relocation.
            }
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: *offset,
                    symbol_name: sym_name.clone(),
                    sym_index: 0,
                    rel_type: abs_reloc_type,
                    addend: *addend,
                    object_id: 0,
                    section_index: sec_idx,
                    output_section_name: Some(sec_name.to_string()),
                });
        }
    }

    // For executables (not shared libs), inject a CRT `_start` stub that
    // calls `main` and invokes the exit syscall — but only if the user
    // did not already define `_start`.
    if output_type == OutputType::Executable {
        let has_user_start = input
            .symbols
            .iter()
            .any(|s| s.name == "_start" && s.section_index != 0);
        if !has_user_start {
            inject_crt_start(&mut input, ctx.target);
        }
    }

    // Create the architecture-specific relocation handler.
    let handler = create_relocation_handler(ctx.target);

    // Invoke the architecture-specific built-in linker.  The arch-specific
    // linkers handle GOT/PLT stub generation and dynamic linking correctly,
    // which is required for resolving undefined symbols (e.g. libc functions)
    // via the runtime dynamic linker.
    let linker_result: Result<LinkerOutput, String> = match ctx.target {
        Target::X86_64 => {
            crate::backend::x86_64::linker::link_x86_64(config.clone(), vec![input], diagnostics)
        }
        // Other architectures fall back to the common linker.
        _ => link(&config, vec![input], handler.as_ref(), diagnostics),
    };
    match linker_result {
        Ok(output) => {
            // Validate the linker output.
            let LinkerOutput {
                elf_data,
                entry_point: _,
                output_type: linked_type,
            } = output;

            debug_assert_eq!(linked_type, output_type, "linker output type mismatch");

            Ok(elf_data)
        }
        Err(e) => {
            diagnostics.emit_error(Span::dummy(), format!("linker: {}", e));
            Err(format!("linking failed: {}", e))
        }
    }
}

/// Generate a synthetic CRT startup object that defines `_start`.
///
/// The `_start` entry point:
/// 1. Clears the frame pointer (ABI requirement)
/// 2. Calls `main`
/// 3. Passes the return value to the `exit` syscall
///
/// This is the minimal CRT startup needed to produce a working executable
/// without linking against the system CRT objects (crt0.o, crti.o, etc.).
/// Inject a CRT `_start` stub into the linker input, appending it to
/// the existing .text section with the call to `main` pre-resolved.
///
/// The stub is placed immediately after the user's code. Because both `main`
/// and `_start` are in the same .text section, the PC-relative call
/// displacement can be computed at injection time without a relocation.
fn inject_crt_start(input: &mut LinkerInput, target: Target) {
    use crate::backend::linker_common::symbol_resolver::{
        InputSymbol, STB_GLOBAL, STT_FUNC, STV_DEFAULT,
    };

    // Find the .text section in the existing input.
    let text_idx = input.sections.iter().position(|s| s.name == ".text");

    let text_idx = match text_idx {
        Some(idx) => idx,
        None => return, // No .text — nothing to inject into.
    };

    // Align _start to 16 bytes within .text.
    let current_len = input.sections[text_idx].data.len();
    let aligned = (current_len + 15) & !15;
    // Pad with NOP (0x90 for x86, 0x00 is fine for non-x86 since it won't execute).
    let nop_byte = match target {
        Target::X86_64 | Target::I686 => 0x90u8,
        _ => 0x00u8,
    };
    input.sections[text_idx].data.resize(aligned, nop_byte);

    let start_offset = input.sections[text_idx].data.len();
    // Find the actual offset of `main` in .text by looking up its symbol.
    // The symbol's section_index is the ELF section header index (1-based,
    // with 0 = SHN_UNDEF), so we filter for any defined (non-zero) section.
    let main_offset = input
        .symbols
        .iter()
        .find(|s| s.name == "main" && s.section_index != 0)
        .map(|s| s.value)
        .unwrap_or(0);

    match target {
        Target::X86_64 => {
            // _start stub (24 bytes):
            //   xor %ebp, %ebp          ; 31 ed           (2 bytes)  offset 0
            //   pop %rdi                ; 5f              (1 byte)   offset 2  argc
            //   mov %rsp, %rsi          ; 48 89 e6        (3 bytes)  offset 3  argv
            //   and $-16, %rsp          ; 48 83 e4 f0     (4 bytes)  offset 6  align
            //   call main               ; e8 <disp32>     (5 bytes)  offset 10
            //   mov %eax, %edi           ; 89 c7           (2 bytes)  offset 15
            //   call exit               ; e8 <00 00 00 00> (5 bytes)  offset 17
            //   ud2                      ; 0f 0b           (2 bytes)  offset 22
            //
            // Linux kernel puts argc at (%rsp) and argv[] starting at
            // 8(%rsp) on process entry.  We pop argc into %rdi (first
            // arg) and set %rsi to the remaining stack (argv).  The
            // stack is 16-byte aligned before the CALL per the SysV ABI.
            //
            // Using `call exit` (libc) instead of raw SYS_exit so that
            // atexit handlers run and stdio buffers are flushed.
            let call_main_site = start_offset + 10;
            let call_main_next = call_main_site + 5;
            let disp = (main_offset as i64) - (call_main_next as i64);
            let disp_bytes = (disp as i32).to_le_bytes();
            let stub = vec![
                0x31,
                0xed, // xor %ebp, %ebp
                0x5f, // pop %rdi  (argc)
                0x48,
                0x89,
                0xe6, // mov %rsp, %rsi  (argv)
                0x48,
                0x83,
                0xe4,
                0xf0, // and $-16, %rsp
                0xe8,
                disp_bytes[0],
                disp_bytes[1], // call main
                disp_bytes[2],
                disp_bytes[3],
                0x89,
                0xc7, // mov %eax, %edi
                0xe8,
                0x00,
                0x00,
                0x00,
                0x00, // call exit (PLT32)
                0x0f,
                0x0b, // ud2
            ];
            // Add R_X86_64_PLT32 relocation for the `exit` call.
            // The `e8` opcode is at stub byte 17; the 4-byte displacement
            // field starts at stub byte 18 (start_offset + 18).
            let exit_reloc_offset = (start_offset + 18) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 4, // R_X86_64_PLT32
                    addend: -4,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::I686 => {
            // _start stub for i686 (24 bytes):
            //   xor %ebp, %ebp          ; 31 ed           (2 bytes) offset 0
            //   pop %ecx                ; 59              (1 byte)  offset 2  argc
            //   mov %esp, %edx          ; 89 e2           (2 bytes) offset 3  argv
            //   and $-16, %esp          ; 83 e4 f0        (3 bytes) offset 5  align
            //   push %edx               ; 52              (1 byte)  offset 8  argv
            //   push %ecx               ; 51              (1 byte)  offset 9  argc
            //   call main               ; e8 <disp32>     (5 bytes) offset 10
            //   push %eax               ; 50              (1 byte)  offset 15
            //   call exit               ; e8 <00 00 00 00> (5 bytes) offset 16
            //   ud2                      ; 0f 0b           (2 bytes) offset 21
            //   nop; nop                ; 90 90           (2 bytes) offset 23
            //
            // i686 cdecl: arguments pushed right-to-left on stack.
            // Push argv then argc so main sees (argc, argv).
            let call_site = start_offset + 10;
            let call_next = call_site + 5;
            let disp = (main_offset as i64) - (call_next as i64);
            let disp_bytes = (disp as i32).to_le_bytes();
            let stub = vec![
                0x31,
                0xed, // xor %ebp, %ebp
                0x59, // pop %ecx  (argc)
                0x89,
                0xe2, // mov %esp, %edx  (argv)
                0x83,
                0xe4,
                0xf0, // and $-16, %esp
                0x52, // push %edx (argv — second arg, pushed first)
                0x51, // push %ecx (argc — first arg, pushed second)
                0xe8,
                disp_bytes[0],
                disp_bytes[1], // call main
                disp_bytes[2],
                disp_bytes[3],
                0x50, // push %eax (exit status for cdecl)
                0xe8,
                0x00,
                0x00,
                0x00,
                0x00, // call exit (PLT32 relocation)
                0x0f,
                0x0b, // ud2
                0x90,
                0x90, // nop (alignment pad)
            ];
            // Add R_386_PLT32 relocation for the `exit` call.
            // The `e8` opcode is at stub byte 16; the 4-byte displacement
            // field starts at stub byte 17 (start_offset + 17).
            let exit_reloc_offset = (start_offset + 17) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 4, // R_386_PLT32
                    addend: -4,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::AArch64 => {
            // _start stub for AArch64 (28 bytes / 7 instructions):
            //   ldr x0, [sp]           ; argc            (4 bytes) offset 0
            //   add x1, sp, #8         ; argv            (4 bytes) offset 4
            //   nop                    ; pad (AArch64 kernel guarantees 16B-aligned SP)
            //   bl main                ; call main       (4 bytes) offset 12
            //   bl exit                ; call exit       (4 bytes) offset 16
            //   brk #1                 ; unreachable     (4 bytes) offset 20
            //   nop                    ; pad             (4 bytes) offset 24
            let ldr_x0_sp = 0xF940_03E0u32; // ldr x0, [sp]
            let add_x1_sp_8 = 0x9100_23E1u32; // add x1, sp, #8
            let and_sp_align = 0xD503_201Fu32; // nop — alignment not needed
            let bl_offset = start_offset + 12;
            let offset_bytes = ((main_offset as i64) - (bl_offset as i64)) / 4;
            let imm26 = (offset_bytes as u32) & 0x03FF_FFFF;
            let bl_main = 0x9400_0000u32 | imm26;
            let bl_exit_placeholder = 0x9400_0000u32; // BL with disp 0 (reloc fills)
            let brk1 = 0xD420_0020u32; // brk #1
            let nop = 0xD503_201Fu32;
            let mut stub = Vec::new();
            stub.extend_from_slice(&ldr_x0_sp.to_le_bytes());
            stub.extend_from_slice(&add_x1_sp_8.to_le_bytes());
            stub.extend_from_slice(&and_sp_align.to_le_bytes());
            stub.extend_from_slice(&bl_main.to_le_bytes());
            stub.extend_from_slice(&bl_exit_placeholder.to_le_bytes());
            stub.extend_from_slice(&brk1.to_le_bytes());
            stub.extend_from_slice(&nop.to_le_bytes());
            // Add R_AARCH64_CALL26 relocation for the `exit` call.
            let exit_reloc_offset = (start_offset + 16) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 283, // R_AARCH64_CALL26
                    addend: 0,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::RiscV64 => {
            // _start stub for RISC-V 64 (32 bytes / 8 instructions):
            //   ld a0, 0(sp)            ; argc                        (4 bytes) offset 0
            //   addi a1, sp, 8          ; argv                        (4 bytes) offset 4
            //   andi sp, sp, -16        ; align stack to 16 bytes     (4 bytes) offset 8
            //   jal ra, main            ; call main                   (4 bytes) offset 12
            //   auipc ra, 0             ; exit call via PLT           (4 bytes) offset 16
            //   jalr ra, ra, 0          ; (CALL_PLT pair)             (4 bytes) offset 20
            //   ebreak                  ; unreachable                 (4 bytes) offset 24
            //   nop                     ; alignment pad               (4 bytes) offset 28
            // ld a0, 0(sp): funct3=011 (doubleword), rd=a0(x10), rs1=sp(x2)
            // Encoding: imm[11:0]=0, rs1=2, funct3=011, rd=10, opcode=0000011
            let ld_a0_sp = 0x0001_3503u32; // ld a0, 0(sp)
                                           // addi a1, sp, 8: rd=a1(x11), rs1=sp(x2), imm=8
            let addi_a1_sp_8 = 0x0081_0593u32; // addi a1, sp, 8
                                               // andi sp, sp, -16: rd=sp(x2), rs1=sp(x2), imm=-16
            let andi_sp = 0xFF01_7113u32; // andi sp, sp, -16
            let jal_offset = start_offset + 12;
            let offset = (main_offset as i64) - (jal_offset as i64);
            let imm = offset as i32;
            let imm20 = ((imm >> 20) & 1) as u32;
            let imm10_1 = ((imm >> 1) & 0x3FF) as u32;
            let imm11 = ((imm >> 11) & 1) as u32;
            let imm19_12 = ((imm >> 12) & 0xFF) as u32;
            let jal_main = (imm20 << 31)
                | (imm10_1 << 21)
                | (imm11 << 20)
                | (imm19_12 << 12)
                | (1 << 7)
                | 0x6F;
            // AUIPC ra, 0 — upper 20 bits filled by linker relocation
            let auipc_exit = 0x0000_0097u32; // auipc ra, 0
                                             // JALR ra, ra, 0 — lower 12 bits filled by linker relocation
            let jalr_exit = 0x0000_80E7_u32; // jalr ra, 0(ra)
            let ebreak = 0x0010_0073u32;
            let nop = 0x0000_0013u32; // addi x0, x0, 0
            let mut stub = Vec::new();
            stub.extend_from_slice(&ld_a0_sp.to_le_bytes());
            stub.extend_from_slice(&addi_a1_sp_8.to_le_bytes());
            stub.extend_from_slice(&andi_sp.to_le_bytes());
            stub.extend_from_slice(&jal_main.to_le_bytes());
            stub.extend_from_slice(&auipc_exit.to_le_bytes());
            stub.extend_from_slice(&jalr_exit.to_le_bytes());
            stub.extend_from_slice(&ebreak.to_le_bytes());
            stub.extend_from_slice(&nop.to_le_bytes());
            // Add R_RISCV_CALL_PLT relocation for `exit` at the auipc+jalr pair.
            let exit_reloc_offset = (start_offset + 16) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 19, // R_RISCV_CALL_PLT
                    addend: 0,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
    }

    let stub_size = input.sections[text_idx].data.len() - start_offset;

    // Update section size.
    input.sections[text_idx].size = input.sections[text_idx].data.len() as u64;

    // Add the `_start` symbol at the stub's offset.
    let section_original_index = input.sections[text_idx].original_index;
    input.symbols.push(InputSymbol {
        name: "_start".to_string(),
        value: start_offset as u64,
        size: stub_size as u64,
        binding: STB_GLOBAL,
        sym_type: STT_FUNC,
        visibility: STV_DEFAULT,
        section_index: section_original_index as u16,
        object_file_id: input.object_id,
    });

    // For all architectures, the _start stub calls `exit` through PLT
    // (libc's exit) so that stdio buffers are flushed and atexit handlers
    // run. Add `exit` as an undefined external symbol so the linker creates
    // a PLT entry and the dynamic linker resolves it at runtime.
    {
        let has_exit = input.symbols.iter().any(|s| s.name == "exit");
        if !has_exit {
            input.symbols.push(InputSymbol {
                name: "exit".to_string(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF — external
                object_file_id: input.object_id,
            });
        }
    }
}

/// Create an architecture-specific relocation handler based on the target.
///
/// Each architecture provides its own `RelocationHandler` implementation
/// in its `linker/relocations.rs` module.  This function dispatches to the
/// correct handler for the given compilation target.
pub fn create_relocation_handler(target: Target) -> Box<dyn RelocationHandler> {
    match target {
        Target::X86_64 => Box::new(crate::backend::x86_64::linker::X86_64RelocationHandler::new()),
        Target::I686 => Box::new(crate::backend::i686::linker::I686RelocationHandler::new()),
        Target::AArch64 => {
            Box::new(crate::backend::aarch64::linker::relocations::AArch64RelocationHandler::new())
        }
        Target::RiscV64 => {
            Box::new(crate::backend::riscv64::linker::relocations::RiscV64RelocationHandler::new())
        }
    }
}

/// Parse a relocatable ELF object file (ET_REL) byte buffer into a
/// [`LinkerInput`] suitable for the built-in linker.
///
/// This performs a lightweight ELF header / section header scan:
/// - Identifies `.text`, `.data`, `.rodata`, `.bss` sections and copies
///   their bytes into [`InputSection`]s
/// - Extracts `.symtab` entries into [`InputSymbol`]s
/// - Extracts `.rela.*` relocations (if any)
///
/// The implementation handles the common ELF structures produced by the
/// BCC ELF writer (`elf_writer_common.rs`).  External object files with
/// unusual section layouts may not parse perfectly, but BCC only links
/// its own objects in standalone mode.
pub fn parse_elf_object_to_linker_input(
    object_id: u32,
    filename: &str,
    data: &[u8],
) -> LinkerInput {
    parse_elf_object_to_linker_input_impl(object_id, filename, data, false)
}

/// Parse an ELF relocatable object with optional LOCAL symbol uniquification.
/// When `uniquify_locals` is true, non-section LOCAL symbols get an `__obj<id>`
/// suffix to prevent name collisions across different object files during
/// multi-object linking.
pub fn parse_elf_object_to_linker_input_uniquified(
    object_id: u32,
    filename: &str,
    data: &[u8],
) -> LinkerInput {
    parse_elf_object_to_linker_input_impl(object_id, filename, data, true)
}

fn parse_elf_object_to_linker_input_impl(
    object_id: u32,
    filename: &str,
    data: &[u8],
    uniquify_locals: bool,
) -> LinkerInput {
    use crate::backend::linker_common::section_merger::InputSection;
    use crate::backend::linker_common::symbol_resolver::{
        InputSymbol, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT,
    };

    let mut input = LinkerInput::new(object_id, filename.to_string());

    // Minimum ELF header size check (64-byte ELF64 header).
    if data.len() < 64 {
        // Too small to be a valid ELF — return an input with the raw data
        // as a single .text section so the linker can at least try.
        input.sections.push(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: data.to_vec(),
            size: data.len() as u64,
            alignment: 16,
            object_id,
            original_index: 1,
            group_signature: None,
            relocations: Vec::new(),
        });
        return input;
    }

    // Detect ELF class from e_ident[EI_CLASS].
    let is_64bit = data[4] == 2;
    // Detect endianness from e_ident[EI_DATA].
    let is_le = data[5] == 1;

    // Helper closures for reading ELF integers.
    let read_u16 = |off: usize| -> u16 {
        if is_le {
            u16::from_le_bytes([data[off], data[off + 1]])
        } else {
            u16::from_be_bytes([data[off], data[off + 1]])
        }
    };
    let read_u32 = |off: usize| -> u32 {
        if is_le {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        }
    };
    let read_u64 = |off: usize| -> u64 {
        if is_le {
            u64::from_le_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
                data[off + 4],
                data[off + 5],
                data[off + 6],
                data[off + 7],
            ])
        } else {
            u64::from_be_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
                data[off + 4],
                data[off + 5],
                data[off + 6],
                data[off + 7],
            ])
        }
    };

    // Parse ELF header fields.
    let (e_shoff, e_shentsize, e_shnum, e_shstrndx) = if is_64bit {
        (
            read_u64(40) as usize,
            read_u16(58) as usize,
            read_u16(60) as usize,
            read_u16(62) as usize,
        )
    } else {
        (
            read_u32(32) as usize,
            read_u16(46) as usize,
            read_u16(48) as usize,
            read_u16(50) as usize,
        )
    };

    if e_shoff == 0 || e_shnum == 0 || e_shoff + e_shnum * e_shentsize > data.len() {
        // Malformed section headers — fall back to raw .text section.
        input.sections.push(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: data.to_vec(),
            size: data.len() as u64,
            alignment: 16,
            object_id,
            original_index: 1,
            group_signature: None,
            relocations: Vec::new(),
        });
        return input;
    }

    // Read section header entries into a temporary vector.
    #[allow(dead_code)]
    struct ShdrInfo {
        sh_name: u32,
        sh_type: u32,
        sh_flags: u64,
        sh_offset: usize,
        sh_size: usize,
        sh_link: u32,
        sh_info: u32,
        sh_addralign: u64,
        sh_entsize: u64,
    }

    let mut shdrs: Vec<ShdrInfo> = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let base = e_shoff + i * e_shentsize;
        if base + e_shentsize > data.len() {
            break;
        }
        if is_64bit {
            shdrs.push(ShdrInfo {
                sh_name: read_u32(base),
                sh_type: read_u32(base + 4),
                sh_flags: read_u64(base + 8),
                sh_offset: read_u64(base + 24) as usize,
                sh_size: read_u64(base + 32) as usize,
                sh_link: read_u32(base + 40),
                sh_info: read_u32(base + 44),
                sh_addralign: read_u64(base + 48),
                sh_entsize: read_u64(base + 56),
            });
        } else {
            shdrs.push(ShdrInfo {
                sh_name: read_u32(base),
                sh_type: read_u32(base + 4),
                sh_flags: read_u32(base + 8) as u64,
                sh_offset: read_u32(base + 16) as usize,
                sh_size: read_u32(base + 20) as usize,
                sh_link: read_u32(base + 24),
                sh_info: read_u32(base + 28),
                sh_addralign: read_u32(base + 32) as u64,
                sh_entsize: read_u32(base + 36) as u64,
            });
        }
    }

    // Build section name lookup from .shstrtab.
    let shstrtab_data: &[u8] = if e_shstrndx < shdrs.len() {
        let sh = &shdrs[e_shstrndx];
        let end = sh.sh_offset.saturating_add(sh.sh_size).min(data.len());
        &data[sh.sh_offset..end]
    } else {
        &[]
    };

    let section_name = |name_off: u32| -> String {
        let off = name_off as usize;
        if off >= shstrtab_data.len() {
            return String::new();
        }
        let end = shstrtab_data[off..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(shstrtab_data.len() - off);
        String::from_utf8_lossy(&shstrtab_data[off..off + end]).into_owned()
    };

    // Locate the symbol table (.symtab) and its associated string table.
    let mut symtab_idx: Option<usize> = None;
    for (i, sh) in shdrs.iter().enumerate() {
        if sh.sh_type == 2 {
            // SHT_SYMTAB
            symtab_idx = Some(i);
            break;
        }
    }

    // Extract symbols from .symtab.
    if let Some(si) = symtab_idx {
        let sym_sh = &shdrs[si];
        let strtab_sh = if (sym_sh.sh_link as usize) < shdrs.len() {
            &shdrs[sym_sh.sh_link as usize]
        } else {
            // No string table — skip symbol extraction.
            &shdrs[0]
        };

        let strtab_data: &[u8] = {
            let end = strtab_sh
                .sh_offset
                .saturating_add(strtab_sh.sh_size)
                .min(data.len());
            &data[strtab_sh.sh_offset..end]
        };

        let sym_name_from = |name_off: u32| -> String {
            let off = name_off as usize;
            if off >= strtab_data.len() {
                return String::new();
            }
            let end = strtab_data[off..]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(strtab_data.len() - off);
            String::from_utf8_lossy(&strtab_data[off..off + end]).into_owned()
        };

        let entry_size = if sym_sh.sh_entsize > 0 {
            sym_sh.sh_entsize as usize
        } else if is_64bit {
            24
        } else {
            16
        };

        let num_syms = if entry_size > 0 {
            sym_sh.sh_size / entry_size
        } else {
            0
        };

        for j in 0..num_syms {
            let sym_off = sym_sh.sh_offset + j * entry_size;
            if sym_off + entry_size > data.len() {
                break;
            }

            let (st_name, st_info, st_other, st_shndx, st_value, st_size) = if is_64bit {
                (
                    read_u32(sym_off),
                    data[sym_off + 4],
                    data[sym_off + 5],
                    read_u16(sym_off + 6),
                    read_u64(sym_off + 8),
                    read_u64(sym_off + 16),
                )
            } else {
                (
                    read_u32(sym_off),
                    data[sym_off + 12],
                    data[sym_off + 13],
                    read_u16(sym_off + 14),
                    read_u32(sym_off + 4) as u64,
                    read_u32(sym_off + 8) as u64,
                )
            };

            let binding = st_info >> 4;
            let sym_type = st_info & 0xf;
            let mut name = sym_name_from(st_name);

            // Skip the null symbol entry.
            if j == 0 && name.is_empty() && st_value == 0 && st_shndx == 0 {
                continue;
            }

            // Handle ELF section symbols (STT_SECTION = 3).
            // Section symbols typically have an empty name but reference a
            // section via st_shndx.  GCC-produced objects use these as
            // relocation targets (e.g. R_X86_64_PC32 to .rodata for string
            // literals).  We assign a unique name derived from the section
            // name and object_id so the linker's symbol resolver can track
            // per-object section addresses after section merging.
            if sym_type == 3 && (st_shndx as usize) < shdrs.len() && st_shndx != 0 {
                let sec_hdr = &shdrs[st_shndx as usize];
                let sec_nm = section_name(sec_hdr.sh_name);
                if !sec_nm.is_empty() {
                    name = format!("__secsym_{}_{}", sec_nm, object_id);
                }
            }

            // Uniquify non-section LOCAL symbols by appending the object ID.
            // Different .o files may have identically-named local symbols
            // (e.g. `.str.0`, `.L.str.0`, static helper functions) that
            // must not collide in the linker's symbol address map.
            // Only applied when linking multiple object files together.
            if uniquify_locals
                && binding == 0
                && !name.is_empty()
                && sym_type != 3
                && !name.starts_with("__secsym_")
            {
                name = format!("{}__obj{}", name, object_id);
            }

            input.symbols.push(InputSymbol {
                name,
                value: st_value,
                size: st_size,
                binding: match binding {
                    0 => STB_LOCAL,
                    1 => STB_GLOBAL,
                    2 => STB_WEAK,
                    _ => STB_GLOBAL,
                },
                sym_type: match sym_type {
                    0 => STT_NOTYPE,
                    1 => STT_OBJECT,
                    2 => STT_FUNC,
                    3 => STT_NOTYPE, // STT_SECTION → treated as NOTYPE
                    _ => STT_NOTYPE,
                },
                visibility: st_other & 0x3,
                section_index: st_shndx,
                object_file_id: object_id,
            });
        }
    }

    // Extract loadable sections (.text, .data, .rodata, .bss, .debug_*).
    for (i, sh) in shdrs.iter().enumerate() {
        if i == 0 {
            continue; // skip null section
        }
        let name = section_name(sh.sh_name);
        // Skip non-loadable sections that the linker doesn't need directly.
        // Include PROGBITS, NOBITS, and RELA sections.
        let is_progbits = sh.sh_type == SHT_PROGBITS;
        let is_nobits = sh.sh_type == SHT_NOBITS;
        let is_rela = sh.sh_type == 4; // SHT_RELA
        if !is_progbits && !is_nobits && !is_rela {
            continue;
        }
        if is_rela {
            // Relocation sections are processed separately below.
            continue;
        }
        // Skip sections that are not needed for linking executables:
        // .eh_frame, .note.*, .comment, .GCC.command.line, etc.
        if name.starts_with(".eh_frame")
            || name.starts_with(".note")
            || name == ".comment"
            || name.starts_with(".GCC")
            || name.starts_with(".gnu.hash")
        {
            continue;
        }

        let sec_data = if is_nobits {
            Vec::new()
        } else {
            let end = sh.sh_offset.saturating_add(sh.sh_size).min(data.len());
            data[sh.sh_offset..end].to_vec()
        };

        input.sections.push(InputSection {
            name,
            section_type: sh.sh_type,
            flags: sh.sh_flags,
            data: sec_data,
            size: sh.sh_size as u64,
            alignment: sh.sh_addralign.max(1),
            object_id,
            original_index: i as u32,
            group_signature: None,
            relocations: Vec::new(),
        });
    }

    // -----------------------------------------------------------------------
    // Parse .rela.* / .rel.* relocation sections.
    // -----------------------------------------------------------------------
    // We build an ELF symbol-index → name mapping from the raw symbol table
    // so that we can populate the top-level `input.relocations` (which is
    // what the common linker actually processes).  We also record the
    // section name that each relocation targets, since the linker needs
    // `output_section_name` to compute patch addresses within merged output
    // sections.
    {
        // Build ELF symbol-index → name + section-index mapping.
        // Index 0 in ELF is the null symbol; we preserve that slot so
        // relocation sym_index values can index directly.
        let mut elf_sym_names: Vec<String> = Vec::new();
        let mut elf_sym_sec_idx: Vec<u16> = Vec::new();
        // Track binding and type for each ELF symbol so relocation
        // processing can uniquify local symbol references consistently.
        let mut elf_sym_bindings: Vec<u8> = Vec::new();
        let mut elf_sym_types: Vec<u8> = Vec::new();

        if let Some(si) = symtab_idx {
            let sym_sh = &shdrs[si];
            let strtab_sh = if (sym_sh.sh_link as usize) < shdrs.len() {
                &shdrs[sym_sh.sh_link as usize]
            } else {
                &shdrs[0]
            };
            let strtab_data_local: &[u8] = {
                let end = strtab_sh
                    .sh_offset
                    .saturating_add(strtab_sh.sh_size)
                    .min(data.len());
                &data[strtab_sh.sh_offset..end]
            };
            let sym_name_local = |st_name: u32| -> String {
                let off = st_name as usize;
                if off >= strtab_data_local.len() {
                    return String::new();
                }
                let end = strtab_data_local[off..]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(strtab_data_local.len() - off);
                String::from_utf8_lossy(&strtab_data_local[off..off + end]).into_owned()
            };

            let sym_entry_size = if sym_sh.sh_entsize > 0 {
                sym_sh.sh_entsize as usize
            } else if is_64bit {
                24
            } else {
                16
            };
            let num_syms_local = if sym_entry_size > 0 {
                sym_sh.sh_size / sym_entry_size
            } else {
                0
            };
            for j in 0..num_syms_local {
                let sym_off = sym_sh.sh_offset + j * sym_entry_size;
                if sym_off + sym_entry_size > data.len() {
                    elf_sym_names.push(String::new());
                    elf_sym_sec_idx.push(0);
                    elf_sym_bindings.push(1); // default global
                    elf_sym_types.push(0);
                    continue;
                }
                let (st_name_val, st_shndx_val, st_info_val) = if is_64bit {
                    (read_u32(sym_off), read_u16(sym_off + 6), data[sym_off + 4])
                } else {
                    (
                        read_u32(sym_off),
                        read_u16(sym_off + 14),
                        data[sym_off + 12],
                    )
                };
                elf_sym_names.push(sym_name_local(st_name_val));
                elf_sym_sec_idx.push(st_shndx_val);
                elf_sym_bindings.push(st_info_val >> 4);
                elf_sym_types.push(st_info_val & 0xf);
            }
        }

        // Build a map from section original_index to position in
        // input.sections so we can look up the section name.
        let mut sec_idx_map: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, sec) in input.sections.iter().enumerate() {
            sec_idx_map.insert(sec.original_index, pos);
        }

        // Map from section original_index to section name (covers all
        // loaded sections including .text, .data, .rodata, .bss).
        let sec_name_map: std::collections::HashMap<u32, String> = input
            .sections
            .iter()
            .map(|s| (s.original_index, s.name.clone()))
            .collect();

        for sh in shdrs.iter() {
            // SHT_RELA = 4, SHT_REL = 9
            if sh.sh_type != 4 && sh.sh_type != 9 {
                continue;
            }
            // sh_info points to the section these relocations apply to.
            let target_sec_idx = sh.sh_info;
            // Skip relocations targeting sections we didn't load
            // (e.g. .eh_frame, .note.*, .comment).
            if !sec_idx_map.contains_key(&target_sec_idx) {
                continue;
            }
            let _target_sec_name = sec_name_map
                .get(&target_sec_idx)
                .cloned()
                .unwrap_or_default();
            // Determine the input.sections position (for section_index).
            let _sec_input_pos = sec_idx_map.get(&target_sec_idx).copied();

            let is_rela = sh.sh_type == 4;
            let entry_size = if sh.sh_entsize > 0 {
                sh.sh_entsize as usize
            } else if is_64bit {
                if is_rela {
                    24
                } else {
                    16
                }
            } else if is_rela {
                12
            } else {
                8
            };

            if entry_size == 0 {
                continue;
            }
            let num_entries = sh.sh_size / entry_size;

            for j in 0..num_entries {
                let rel_off = sh.sh_offset + j * entry_size;
                if rel_off + entry_size > data.len() {
                    break;
                }

                let (r_offset, r_info, r_addend) = if is_64bit {
                    let offset = read_u64(rel_off);
                    let info = read_u64(rel_off + 8);
                    let addend = if is_rela {
                        i64::from_le_bytes([
                            data[rel_off + 16],
                            data[rel_off + 17],
                            data[rel_off + 18],
                            data[rel_off + 19],
                            data[rel_off + 20],
                            data[rel_off + 21],
                            data[rel_off + 22],
                            data[rel_off + 23],
                        ])
                    } else {
                        0i64
                    };
                    (offset, info, addend)
                } else {
                    let offset = read_u32(rel_off) as u64;
                    let info = read_u32(rel_off + 4) as u64;
                    let addend = if is_rela {
                        i32::from_le_bytes([
                            data[rel_off + 8],
                            data[rel_off + 9],
                            data[rel_off + 10],
                            data[rel_off + 11],
                        ]) as i64
                    } else {
                        0i64
                    };
                    (offset, info, addend)
                };

                // Extract sym_index and rel_type from r_info.
                let (sym_index, rel_type) = if is_64bit {
                    ((r_info >> 32) as u32, (r_info & 0xffffffff) as u32)
                } else {
                    ((r_info >> 8) as u32, (r_info & 0xff) as u32)
                };

                // Resolve symbol name from the ELF symbol table mapping.
                let symbol_name = if (sym_index as usize) < elf_sym_names.len() {
                    elf_sym_names[sym_index as usize].clone()
                } else {
                    String::new()
                };

                // For relocations referencing section symbols (name is
                // empty, STT_SECTION type), use the per-object unique
                // section symbol name so the linker can resolve them to the
                // correct fragment address within merged output sections.
                let symbol_name = if symbol_name.is_empty() {
                    if (sym_index as usize) < elf_sym_sec_idx.len() {
                        let sym_sec = elf_sym_sec_idx[sym_index as usize];
                        match sec_name_map.get(&(sym_sec as u32)) {
                            Some(sec_nm) => {
                                format!("__secsym_{}_{}", sec_nm, object_id)
                            }
                            None => String::new(),
                        }
                    } else {
                        String::new()
                    }
                } else {
                    symbol_name
                };

                // Uniquify non-section LOCAL symbol references by
                // appending the object ID — must match the uniquification
                // applied in the input.symbols loop above so that
                // relocations resolve to the correct per-object symbol.
                // Only applied during multi-object linking.
                let symbol_name = if uniquify_locals {
                    let sym_binding_val = elf_sym_bindings
                        .get(sym_index as usize)
                        .copied()
                        .unwrap_or(1);
                    let sym_type_val = elf_sym_types.get(sym_index as usize).copied().unwrap_or(0);
                    if sym_binding_val == 0
                        && !symbol_name.is_empty()
                        && sym_type_val != 3
                        && !symbol_name.starts_with("__secsym_")
                    {
                        format!("{}__obj{}", symbol_name, object_id)
                    } else {
                        symbol_name
                    }
                } else {
                    symbol_name
                };

                // Use the original ELF section header index as
                // section_index so it matches the fragment map key
                // (InputSection::original_index).  Do NOT pre-set
                // output_section_name — let annotate_relocations both
                // set the name AND adjust the offset by the fragment's
                // offset_in_output.  Pre-setting the name would cause
                // annotate_relocations to skip offset adjustment,
                // leading to relocations applied at wrong positions
                // in the merged output section.
                let section_index = target_sec_idx;

                // Push into the top-level relocations that the common
                // linker's Phase 5 collects.
                input
                    .relocations
                    .push(crate::backend::linker_common::relocation::Relocation {
                        offset: r_offset,
                        symbol_name,
                        sym_index,
                        rel_type,
                        addend: r_addend,
                        object_id,
                        section_index,
                        output_section_name: None,
                    });
            }
        }
    }

    // If no sections were extracted, provide the raw bytes as a .text
    // section to ensure the linker has something to work with.
    if input.sections.is_empty() {
        input.sections.push(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: data.to_vec(),
            size: data.len() as u64,
            alignment: 16,
            object_id,
            original_index: 1,
            group_signature: None,
            relocations: Vec::new(),
        });
    }

    // If no symbols were extracted, add a default _start symbol at offset 0
    // so the linker can resolve the entry point.
    if input.symbols.is_empty() {
        input.symbols.push(InputSymbol {
            name: "_start".to_string(),
            value: 0,
            size: 0,
            binding: STB_GLOBAL,
            sym_type: STT_FUNC,
            visibility: STV_DEFAULT,
            section_index: 1, // .text
            object_file_id: object_id,
        });
    }

    input
}

// ===========================================================================
// emit_assembly_text — produce human-readable assembly output (-S)
// ===========================================================================

/// Emits human-readable assembly text when the `-S` flag is specified.
///
/// Produces AT&T syntax for x86-64/i686 and standard syntax for
/// AArch64/RISC-V 64.
fn emit_assembly_text<A: ArchCodegen>(
    arch: &A,
    module: &IrModule,
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let mut output = String::new();

    // File header with target information.
    output.push_str(&format!(
        "\t.file\t\"{}\"\n",
        module.name.replace('"', "\\\"")
    ));
    output.push_str(&format!("\t# Target: {}\n", ctx.target));
    output.push('\n');

    // Emit top-level inline assembly blocks verbatim.
    for asm_block in &module.inline_asm_blocks {
        output.push_str(&asm_block.assembly);
        output.push('\n');
    }

    // .text section
    output.push_str("\t.text\n");

    for func in module.functions() {
        if !func.is_definition {
            continue;
        }

        // Emit function label.
        if func.linkage == FunctionLinkage::External {
            output.push_str(&format!("\t.globl\t{}\n", func.name));
        }
        output.push_str(&format!("\t.type\t{}, @function\n", func.name));
        output.push_str(&format!("{}:\n", func.name));

        // Perform instruction selection for assembly text output.
        // Use per-function ref maps to avoid cross-function Value-ID collisions.
        let mut mf = match arch.lower_function(
            func,
            diagnostics,
            module.globals(),
            &func.func_ref_map,
            &func.global_var_refs,
        ) {
            Ok(mf) => mf,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!(
                        "assembly output: instruction selection failed for '{}': {}",
                        func.name, e
                    ),
                );
                continue;
            }
        };

        // Register allocation — resolve virtual registers to physical regs.
        {
            let reg_info: TraitsRegisterInfo = arch.register_info();
            let alloc_reg_info: register_allocator::RegisterInfo = reg_info.into();
            let mut intervals = compute_live_intervals(func);
            let alloc_result = allocate_registers(&mut intervals, &alloc_reg_info, &ctx.target);
            insert_spill_code_from_result(&mut mf, &alloc_result);
            apply_allocation_result(&mut mf, &alloc_result, &ctx.target);

            // Stack alignment fixup (same as the object-code path).
            if matches!(ctx.target, Target::X86_64 | Target::I686) {
                let callee_push_bytes = mf.callee_saved_regs.len() * 8;
                let total = mf.frame_size + callee_push_bytes;
                if total % 16 != 0 {
                    mf.frame_size += 8;
                }
            }

            // Insert prologue/epilogue.
            let prologue = arch.emit_prologue(&mf);
            let epilogue = arch.emit_epilogue(&mf);
            insert_prologue_epilogue(&mut mf, prologue, epilogue);
        }

        // Emit machine instructions as text.
        for (i, block) in mf.blocks.iter().enumerate() {
            if i > 0 {
                if let Some(ref label) = block.label {
                    output.push_str(&format!("{}:\n", label));
                } else {
                    output.push_str(&format!(".LBB_{}_{}:\n", func.name, i));
                }
            }
            for inst in &block.instructions {
                output.push_str(&format!("\t{}\n", arch.format_instruction(inst)));
            }
        }

        output.push_str(&format!("\t.size\t{}, .-{}\n", func.name, func.name));
        output.push('\n');
    }

    // .data section
    if !module.globals().is_empty() {
        output.push_str("\t.data\n");
        for global in module.globals() {
            if !global.is_constant && global.initializer.is_some() {
                if global.linkage == crate::ir::module::Linkage::External {
                    output.push_str(&format!("\t.globl\t{}\n", global.name));
                }
                output.push_str(&format!("{}:\n", global.name));
                if let Some(ref init) = global.initializer {
                    emit_constant_as_asm(init, &mut output);
                }
            }
        }
    }

    // .rodata section for constants and string literals.
    let has_rodata =
        module.globals().iter().any(|g| g.is_constant) || !module.string_pool().is_empty();
    if has_rodata {
        output.push_str("\n\t.section\t.rodata\n");
        for global in module.globals() {
            if global.is_constant {
                if global.linkage == crate::ir::module::Linkage::External {
                    output.push_str(&format!("\t.globl\t{}\n", global.name));
                }
                output.push_str(&format!("{}:\n", global.name));
                if let Some(ref init) = global.initializer {
                    emit_constant_as_asm(init, &mut output);
                }
            }
        }
        for string_lit in module.string_pool() {
            output.push_str(&format!("{}:\n", string_lit.label));
            output.push_str("\t.ascii\t\"");
            for &b in &string_lit.bytes {
                if b == b'\\' {
                    output.push_str("\\\\");
                } else if b == b'"' {
                    output.push_str("\\\"");
                } else if b == b'\n' {
                    output.push_str("\\n");
                } else if b == b'\t' {
                    output.push_str("\\t");
                } else if b == 0 {
                    output.push_str("\\0");
                } else if b.is_ascii_graphic() || b == b' ' {
                    output.push(b as char);
                } else {
                    output.push_str(&format!("\\{:03o}", b));
                }
            }
            output.push_str("\"\n");
        }
    }

    // .bss section
    let has_bss = module.globals().iter().any(|g| {
        g.initializer.is_none()
            || matches!(&g.initializer, Some(crate::ir::module::Constant::ZeroInit))
    });
    if has_bss {
        output.push_str("\n\t.bss\n");
        for global in module.globals() {
            let is_bss = global.initializer.is_none()
                || matches!(
                    &global.initializer,
                    Some(crate::ir::module::Constant::ZeroInit)
                );
            if is_bss {
                if global.linkage == crate::ir::module::Linkage::External {
                    output.push_str(&format!("\t.globl\t{}\n", global.name));
                }
                let size = global_variable_size(global);
                output.push_str(&format!("\t.comm\t{},{}\n", global.name, size));
            }
        }
    }

    // External declarations.
    for decl in module.declarations() {
        output.push_str(&format!("\t.extern\t{}\n", decl.name));
    }

    Ok(output.into_bytes())
}

/// Emits a constant initializer as assembly directives.
fn emit_constant_as_asm(constant: &crate::ir::module::Constant, output: &mut String) {
    use crate::ir::module::Constant;

    match constant {
        Constant::Integer(v) => {
            if *v >= i32::MIN as i128 && *v <= u32::MAX as i128 {
                output.push_str(&format!("\t.long\t{}\n", *v as i64));
            } else {
                output.push_str(&format!("\t.quad\t{}\n", *v as i64));
            }
        }
        Constant::Float(v) => {
            // Emit as raw bytes to preserve exact bit pattern.
            let bytes = v.to_le_bytes();
            output.push_str("\t.byte\t");
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("0x{:02x}", b));
            }
            output.push('\n');
        }
        Constant::LongDouble(bytes) => {
            output.push_str("\t.byte\t");
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("0x{:02x}", b));
            }
            output.push('\n');
            // Pad to 16 bytes.
            output.push_str("\t.zero\t6\n");
        }
        Constant::String(bytes) => {
            output.push_str("\t.ascii\t\"");
            for &b in bytes {
                if b.is_ascii_graphic() || b == b' ' {
                    output.push(b as char);
                } else {
                    output.push_str(&format!("\\{:03o}", b));
                }
            }
            output.push_str("\"\n");
        }
        Constant::ZeroInit => {
            output.push_str("\t.zero\t8\n");
        }
        Constant::Struct(fields) => {
            for field in fields {
                emit_constant_as_asm(field, output);
            }
        }
        Constant::Array(elements) => {
            for elem in elements {
                emit_constant_as_asm(elem, output);
            }
        }
        Constant::GlobalRef(name) => {
            output.push_str(&format!("\t.quad\t{}\n", name));
        }
        Constant::GlobalRefOffset(name, addend) => {
            if *addend >= 0 {
                output.push_str(&format!("\t.quad\t{}+{}\n", name, addend));
            } else {
                output.push_str(&format!("\t.quad\t{}{}\n", name, addend));
            }
        }
        Constant::Null => {
            output.push_str("\t.quad\t0\n");
        }
        Constant::Undefined => {
            output.push_str("\t.zero\t8\n");
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_context_default() {
        let ctx = CodegenContext::default();
        assert_eq!(ctx.target, Target::X86_64);
        assert!(!ctx.debug_info);
        assert_eq!(ctx.optimization_level, 0);
        assert!(!ctx.pic);
        assert!(!ctx.shared);
        assert!(!ctx.retpoline);
        assert!(!ctx.cf_protection);
        assert_eq!(ctx.output_path, "a.out");
        assert!(!ctx.compile_only);
        assert!(!ctx.emit_assembly);
    }

    #[test]
    fn test_should_emit_dwarf() {
        let mut ctx = CodegenContext::default();
        assert!(!ctx.should_emit_dwarf()); // no -g

        ctx.debug_info = true;
        ctx.optimization_level = 0;
        assert!(ctx.should_emit_dwarf()); // -g -O0

        ctx.optimization_level = 1;
        assert!(!ctx.should_emit_dwarf()); // -g -O1 → no DWARF
    }

    #[test]
    fn test_security_mitigations() {
        let mut ctx = CodegenContext::default();
        ctx.target = Target::X86_64;
        ctx.retpoline = true;
        assert!(ctx.has_security_mitigations());

        ctx.target = Target::AArch64;
        assert!(!ctx.has_security_mitigations()); // Not x86-64
    }

    #[test]
    fn test_output_type() {
        let mut ctx = CodegenContext::default();
        assert_eq!(ctx.output_type(), OutputType::Executable);

        ctx.compile_only = true;
        assert_eq!(ctx.output_type(), OutputType::Relocatable);

        ctx.compile_only = false;
        ctx.shared = true;
        assert_eq!(ctx.output_type(), OutputType::SharedLibrary);
    }

    #[test]
    fn test_constant_to_bytes_integer() {
        use crate::ir::module::Constant;
        let bytes = constant_to_bytes(&Constant::Integer(42));
        assert!(!bytes.is_empty());
        // Should be little-endian encoding of 42.
        assert_eq!(bytes[0], 42);
    }

    #[test]
    fn test_constant_to_bytes_string() {
        use crate::ir::module::Constant;
        let data = b"Hello\0".to_vec();
        let bytes = constant_to_bytes(&Constant::String(data.clone()));
        assert_eq!(bytes, data);
    }

    #[test]
    fn test_constant_to_bytes_zero_init() {
        use crate::ir::module::Constant;
        let bytes = constant_to_bytes(&Constant::ZeroInit);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_ir_type_sizes() {
        use crate::ir::types::IrType;
        assert_eq!(ir_type_size(&IrType::I8), 1);
        assert_eq!(ir_type_size(&IrType::I16), 2);
        assert_eq!(ir_type_size(&IrType::I32), 4);
        assert_eq!(ir_type_size(&IrType::I64), 8);
        assert_eq!(ir_type_size(&IrType::Ptr), 8);
        assert_eq!(ir_type_size(&IrType::F32), 4);
        assert_eq!(ir_type_size(&IrType::F64), 8);
    }
}
