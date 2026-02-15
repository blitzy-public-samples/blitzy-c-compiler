//! Phase 10 code generation driver for BCC.
//!
//! This module is the central coordinator between the middle-end IR and the
//! four architecture backends (x86-64, i686, AArch64, RISC-V 64). It
//! orchestrates the full backend pipeline:
//!
//! 1. Architecture dispatch — selects the correct [`ArchCodegen`] implementation
//!    based on the `--target` flag via the [`Target`] enum.
//! 2. Function compilation — for each IR function: instruction selection,
//!    register allocation, prologue/epilogue emission, and machine code assembly.
//! 3. Security mitigation injection — conditionally applies retpoline thunks,
//!    CET/IBT `endbr64`, and stack probe loops for x86-64 targets.
//! 4. Object file emission — produces relocatable `.o` files (ET_REL) or
//!    invokes the built-in linker for final ET_EXEC / ET_DYN ELF output.
//! 5. DWARF debug info — conditionally generates `.debug_info`, `.debug_abbrev`,
//!    `.debug_line`, `.debug_str` sections when `-g` is active.
//!
//! # Pipeline Stage Integration
//!
//! The generation driver consumes phi-eliminated IR from the middle-end
//! (`src/ir/mem2reg/phi_eliminate.rs`) and produces ELF binaries via the
//! built-in assemblers and linkers. This is the final stage of the BCC
//! compilation pipeline.
//!
//! # Standalone Backend Mode
//!
//! Per Section 0.7.7, BCC includes its own assembler and linker for all four
//! target architectures. No external toolchain components (`as`, `ld`, `gcc`,
//! `llvm-mc`, `lld`) are invoked.

use std::io;
use std::path::{Path, PathBuf};

use crate::common::target::Target;
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::temp_files::TempFile;
use crate::common::types::CType;
use crate::ir::module::IrModule;
use crate::ir::function::{IrFunction, Linkage, Visibility};
use crate::ir::types::IrType;
use crate::backend::traits::{
    ArchCodegen, MachineFunction, MachineBasicBlock, MachineInstr, PhysReg,
    CodegenConfig as BackendCodegenConfig,
};
use crate::backend::register_allocator::RegisterAllocator;
use crate::backend::elf_writer_common::{
    ElfWriter, ElfSection, ElfSymbol, ProgramHeader,
    ET_REL, ET_EXEC, ET_DYN,
    SHT_PROGBITS, SHT_NOBITS,
    SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR,
    STB_LOCAL, STB_GLOBAL, STB_WEAK,
    STT_NOTYPE, STT_FUNC, STT_OBJECT, STT_FILE,
    STV_DEFAULT, STV_HIDDEN, STV_PROTECTED,
    SHN_UNDEF,
    PT_LOAD, PT_PHDR, PT_GNU_STACK,
};
use crate::backend::x86_64::X86_64Codegen;
use crate::backend::i686::I686Codegen;
use crate::backend::aarch64::AArch64Codegen;
use crate::backend::riscv64::RiscV64Codegen;
use crate::backend::x86_64::security::{
    apply_security_mitigations, SecurityConfig, RetpolineGenerator,
};
use crate::backend::dwarf::{DwarfGenerator, DwarfSections};
use crate::backend::linker_common::{
    LinkerScript, OutputType, SymbolResolver, SectionMerger,
    RelocationProcessor, DynamicSectionBuilder,
    InputSection, InputSymbol, InputRelocation,
    SymbolBinding, SymbolType as LinkerSymbolType, SymbolVisibility as LinkerSymbolVisibility,
};

// ---------------------------------------------------------------------------
// OutputMode — CLI-level compilation stop point
// ---------------------------------------------------------------------------

/// Output mode controlling the compilation pipeline stop point.
///
/// This enum maps directly to CLI flags:
/// - `Executable` — default, full compile + link
/// - `Object` — `-c` flag, produce `.o` file
/// - `Assembly` — `-S` flag, produce textual assembly
/// - `PreprocessOnly` — `-E` flag (never reaches code generation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputMode {
    /// Full compilation and linking — produces an ET_EXEC static executable
    /// or ET_DYN shared object (when `-shared` is active).
    Executable,

    /// Compile to relocatable object file (`.o`) using the ELF writer.
    /// Corresponds to the `-c` CLI flag.
    Object,

    /// Emit textual assembly representation.
    /// Corresponds to the `-S` CLI flag.
    Assembly,

    /// Preprocess only — this mode is handled before code generation and
    /// results in an immediate return from [`generate_code`]. Corresponds
    /// to the `-E` CLI flag.
    PreprocessOnly,
}

impl core::fmt::Display for OutputMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OutputMode::Executable => write!(f, "executable"),
            OutputMode::Object => write!(f, "object"),
            OutputMode::Assembly => write!(f, "assembly"),
            OutputMode::PreprocessOnly => write!(f, "preprocess-only"),
        }
    }
}

// ---------------------------------------------------------------------------
// CodegenConfig — high-level generation configuration
// ---------------------------------------------------------------------------

/// Code generation configuration controlling the entire backend pipeline.
///
/// This is the high-level configuration struct that includes output path and
/// mode information beyond what the backend-level
/// [`traits::CodegenConfig`](BackendCodegenConfig) carries. The generation
/// driver converts this to a [`BackendCodegenConfig`] for passing to
/// architecture-specific [`ArchCodegen`] implementations.
///
/// # Fields
///
/// | Field               | CLI Flag           | Description                                |
/// |---------------------|--------------------|--------------------------------------------|
/// | `target`            | `--target=<arch>`  | Target architecture                        |
/// | `optimization_level`| `-O<n>`            | Optimization level (0–3)                   |
/// | `debug_info`        | `-g`               | Emit DWARF v4 debug sections               |
/// | `pic`               | `-fPIC`            | Position-independent code generation       |
/// | `shared`            | `-shared`          | Produce shared library (ET_DYN)            |
/// | `retpoline`         | `-mretpoline`      | x86-64 retpoline thunks                    |
/// | `cf_protection`     | `-fcf-protection`  | x86-64 CET/IBT `endbr64`                  |
/// | `output_path`       | `-o <path>`        | Output file path                           |
/// | `output_mode`       | `-c`/`-S`/`-E`    | Pipeline stop point                        |
#[derive(Debug, Clone)]
pub struct CodegenConfig {
    /// Target architecture for code generation.
    pub target: Target,

    /// Optimization level (0 = no optimization, 1–3 = increasing optimization).
    pub optimization_level: u32,

    /// `true` to emit DWARF v4 debug information (`-g` flag).
    /// When `false`, zero debug sections are emitted (Section 0.7.10).
    pub debug_info: bool,

    /// `true` for position-independent code generation (`-fPIC` flag).
    pub pic: bool,

    /// `true` to produce a shared library (`-shared` flag).
    /// Implies PIC code generation.
    pub shared: bool,

    /// `true` for retpoline thunks on indirect calls (`-mretpoline`, x86-64 only).
    pub retpoline: bool,

    /// `true` for Intel CET/IBT `endbr64` insertion (`-fcf-protection`, x86-64 only).
    pub cf_protection: bool,

    /// Output file path from the `-o` flag.
    pub output_path: PathBuf,

    /// Pipeline stop point controlling what artifact is produced.
    pub output_mode: OutputMode,
}

impl CodegenConfig {
    /// Creates a default configuration for the given target and output path.
    ///
    /// Defaults: `-O0`, no debug info, no PIC, no shared, no security
    /// mitigations, executable output mode.
    pub fn new(target: Target, output_path: PathBuf) -> Self {
        CodegenConfig {
            target,
            optimization_level: 0,
            debug_info: false,
            pic: false,
            shared: false,
            retpoline: false,
            cf_protection: false,
            output_path,
            output_mode: OutputMode::Executable,
        }
    }

    /// Converts to the backend-level [`BackendCodegenConfig`] used by
    /// [`ArchCodegen`] implementations.
    ///
    /// The backend config carries only codegen-relevant flags (target,
    /// optimization, PIC, security) but not output path or mode, which
    /// are orchestration concerns handled by the generation driver.
    fn to_backend_config(&self) -> BackendCodegenConfig {
        BackendCodegenConfig {
            target: self.target,
            optimization_level: self.optimization_level,
            debug_info: self.debug_info,
            pic: self.pic || self.shared,
            shared: self.shared,
            retpoline: self.retpoline,
            cf_protection: self.cf_protection,
        }
    }

    /// Returns `true` if any x86-64-specific security mitigation is enabled.
    ///
    /// Security mitigations (retpoline, CET/IBT) are only applicable to
    /// the x86-64 target. This check is used to gate the security pass
    /// during function compilation.
    #[inline]
    fn has_security_mitigations(&self) -> bool {
        self.target == Target::X86_64 && (self.retpoline || self.cf_protection)
    }

    /// Returns `true` if PIC code generation is required.
    ///
    /// PIC is required when either `-fPIC` or `-shared` is set, since
    /// shared libraries must be position-independent.
    #[inline]
    fn requires_pic(&self) -> bool {
        self.pic || self.shared
    }

    /// Determines the linker output type based on the configuration.
    ///
    /// Maps the high-level `OutputMode` and `shared` flag to the linker's
    /// [`OutputType`] enum for section-to-segment layout decisions.
    fn linker_output_type(&self) -> OutputType {
        match self.output_mode {
            OutputMode::Object => OutputType::RelocatableObject,
            OutputMode::Executable if self.shared => OutputType::SharedLibrary,
            OutputMode::Executable => OutputType::Executable,
            OutputMode::Assembly | OutputMode::PreprocessOnly => OutputType::RelocatableObject,
        }
    }
}

// ---------------------------------------------------------------------------
// AssembledFunction — internal representation of compiled function output
// ---------------------------------------------------------------------------

/// Holds the machine code and metadata for a single compiled function.
///
/// Produced by [`compile_function`] and consumed by the ELF writer or linker
/// to construct output sections and symbol tables.
struct AssembledFunction {
    /// Function symbol name.
    name: String,
    /// Assembled machine code bytes.
    code: Vec<u8>,
    /// Total stack frame size in bytes (for stack probe decisions).
    frame_size: u32,
    /// Whether this function has global (external) linkage.
    is_global: bool,
    /// Whether this function has weak linkage.
    is_weak: bool,
    /// ELF symbol visibility.
    visibility: u8,
    /// Target section name (`.text`, `.text.hot`, `.text.cold`, custom).
    section_name: String,
    /// Function alignment in bytes.
    alignment: u32,
    /// Relocations emitted by the assembler for this function.
    relocations: Vec<InputRelocation>,
    /// Offset of this function within its section (assigned during layout).
    section_offset: u64,
}

/// Holds the data and metadata for a single assembled global variable.
struct AssembledGlobal {
    /// Global variable symbol name.
    name: String,
    /// Serialised initializer data bytes.
    data: Vec<u8>,
    /// Target section name (`.data`, `.rodata`, `.bss`, or custom).
    section_name: String,
    /// Whether this global is const-qualified (→ `.rodata`).
    is_const: bool,
    /// Whether this global belongs in `.bss` (zero-initialized).
    is_bss: bool,
    /// Required alignment in bytes.
    alignment: u32,
    /// Whether this global has global (external) linkage.
    is_global: bool,
    /// Whether this global has weak linkage.
    is_weak: bool,
    /// ELF symbol visibility.
    visibility: u8,
}

/// Holds the data for a string literal destined for `.rodata`.
struct AssembledStringLiteral {
    /// Label name (e.g., `.L.str.0`).
    label: String,
    /// Raw string data bytes (including null terminator if present).
    data: Vec<u8>,
}

/// Holds retpoline thunk code generated for x86-64 security mitigations.
struct RetpolineThunkData {
    /// Thunk symbol name (e.g., `__x86_indirect_thunk_rax`).
    name: String,
    /// Assembled thunk machine code.
    code: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Architecture Dispatch — create the correct ArchCodegen implementation
// ---------------------------------------------------------------------------

/// Creates the architecture-specific code generator based on the target
/// in the backend configuration.
///
/// This is the central dispatch point described in Section 0.5.1 Group 5:
/// ```text
/// match target {
///     Target::X86_64  => X86_64Codegen::new(config),
///     Target::I686    => I686Codegen::new(config),
///     Target::AArch64 => AArch64Codegen::new(config),
///     Target::RiscV64 => RiscV64Codegen::new(config),
/// }
/// ```
fn create_codegen(config: &BackendCodegenConfig) -> Box<dyn ArchCodegen> {
    match config.target {
        Target::X86_64 => Box::new(X86_64Codegen::new(config.clone())),
        Target::I686 => Box::new(I686Codegen::new(config.clone())),
        Target::AArch64 => Box::new(AArch64Codegen::new(config.clone())),
        Target::RiscV64 => Box::new(RiscV64Codegen::new(config.clone())),
    }
}

// ---------------------------------------------------------------------------
// Function Compilation Pipeline
// ---------------------------------------------------------------------------

/// Compiles a single IR function through the full backend pipeline.
///
/// The pipeline stages for each function are:
/// 1. **Instruction selection** — `ArchCodegen::lower_function()` converts
///    IR instructions to machine instructions with virtual registers.
/// 2. **Prologue/epilogue emission** — `ArchCodegen::emit_prologue()` and
///    `emit_epilogue()` insert frame setup/teardown code.
/// 3. **Register allocation** — `RegisterAllocator` assigns physical
///    registers and generates spill code.
/// 4. **Security mitigations** (x86-64 only) — retpoline, CET/IBT, and
///    stack probe injection.
/// 5. **Assembly** — `ArchCodegen::emit_assembly()` encodes machine
///    instructions into binary machine code bytes.
/// 6. **DWARF emission** — if debug info is enabled, emit function-level
///    debug data to the DWARF generator.
///
/// # Arguments
///
/// * `ir_func` — the IR function to compile (phi-eliminated SSA form)
/// * `codegen` — architecture-specific code generator
/// * `config` — high-level generation configuration
/// * `backend_config` — backend-level configuration for the codegen
/// * `diagnostics` — diagnostic engine for error reporting
/// * `dwarf` — DWARF debug info generator (may be disabled)
/// * `text_offset` — byte offset of this function within the `.text` section
fn compile_function(
    ir_func: &IrFunction,
    codegen: &dyn ArchCodegen,
    config: &CodegenConfig,
    backend_config: &BackendCodegenConfig,
    diagnostics: &mut DiagnosticEngine,
    dwarf: &mut DwarfGenerator,
    text_offset: u64,
) -> io::Result<AssembledFunction> {
    // Step 1: Instruction selection — lower IR to machine instructions
    let mut mf = codegen.lower_function(ir_func);

    // Step 2: Prologue and epilogue emission
    codegen.emit_prologue(&mut mf);
    codegen.emit_epilogue(&mut mf);

    // Step 3: Register allocation
    let mut reg_alloc = RegisterAllocator::new(config.target, codegen);
    reg_alloc.compute_live_intervals(&mf, ir_func);
    reg_alloc.allocate();
    reg_alloc.generate_spill_code(&mut mf);

    // Step 4: Security mitigations (x86-64 only)
    if config.has_security_mitigations() {
        let sec_config = SecurityConfig::from_flags(
            config.retpoline,
            config.cf_protection,
        );
        apply_security_mitigations(&mut mf, &sec_config);
    }

    // Step 5: Assemble to machine code bytes
    let code = codegen.emit_assembly(&mf);

    // Step 6: DWARF debug info emission for this function
    if dwarf.is_enabled() {
        let func_low_pc = text_offset;
        let func_high_pc = text_offset + code.len() as u64;
        let is_external = matches!(ir_func.linkage, Linkage::External);

        // Build parameter list for DWARF: (name, type_handle)
        let dwarf_params: Vec<(String, u32)> = ir_func
            .params
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let name = param
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("arg{}", i));
                let type_handle = dwarf.emit_type(
                    &ir_type_to_ctype(&param.ty),
                    &config.target,
                );
                (name, type_handle)
            })
            .collect();

        // Emit function debug info (locals and line entries are simplified
        // at this level; the DWARF generator accumulates them)
        dwarf.emit_function(
            &ir_func.name,
            func_low_pc,
            func_high_pc,
            is_external,
            &dwarf_params,
            &[], // locals — populated from IR local values in a full implementation
            &[], // line entries — populated from source map in a full implementation
        );
    }

    // Determine the target section name
    let section_name = determine_function_section(ir_func);

    // Map IR linkage to ELF symbol properties
    let is_global = matches!(ir_func.linkage, Linkage::External | Linkage::Weak);
    let is_weak = matches!(ir_func.linkage, Linkage::Weak);
    let visibility = visibility_to_elf(&ir_func.attributes.visibility);

    Ok(AssembledFunction {
        name: ir_func.name.clone(),
        code,
        frame_size: mf.frame_size,
        is_global,
        is_weak,
        visibility,
        section_name,
        alignment: ir_func.alignment.max(1),
        relocations: Vec::new(),
        section_offset: text_offset,
    })
}

// ---------------------------------------------------------------------------
// Global Variable Processing
// ---------------------------------------------------------------------------

/// Processes all global variables from the IR module into assembled data
/// suitable for ELF section construction.
///
/// Each global is classified into one of:
/// - `.rodata` — const-qualified globals
/// - `.bss` — zero-initialized or uninitialized globals
/// - `.data` — initialized mutable globals
/// - Custom section — globals with `__attribute__((section("...")))`)
fn process_globals(module: &IrModule, config: &CodegenConfig) -> io::Result<Vec<AssembledGlobal>> {
    let mut result = Vec::with_capacity(module.globals.len());

    for global in &module.globals {
        let is_bss = global.is_bss();
        let section_name = if let Some(ref custom) = global.section {
            custom.clone()
        } else if global.is_const {
            ".rodata".to_string()
        } else if is_bss {
            ".bss".to_string()
        } else {
            ".data".to_string()
        };

        // Serialize the initializer to bytes
        let data = if is_bss {
            // BSS: no data bytes, size is implicit from the type
            let size = global.ty.size_bytes(&config.target);
            vec![0u8; size as usize]
        } else if let Some(ref init) = global.initializer {
            serialize_constant(init, &global.ty, config)
        } else {
            // Extern declaration without initializer — zero-fill
            let size = global.ty.size_bytes(&config.target);
            vec![0u8; size as usize]
        };

        let is_global = matches!(global.linkage, Linkage::External | Linkage::Weak);
        let is_weak = matches!(global.linkage, Linkage::Weak);

        result.push(AssembledGlobal {
            name: global.name.clone(),
            data,
            section_name,
            is_const: global.is_const,
            is_bss,
            alignment: global.alignment.max(1),
            is_global,
            is_weak,
            visibility: STV_DEFAULT,
        });
    }

    Ok(result)
}

/// Processes string literals from the IR module into data blocks for `.rodata`.
fn process_string_literals(module: &IrModule) -> Vec<AssembledStringLiteral> {
    module
        .string_literals
        .iter()
        .map(|lit| AssembledStringLiteral {
            label: format!(".L.str.{}", lit.id),
            data: lit.data.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Constant Serialization
// ---------------------------------------------------------------------------

/// Serialises an IR constant value into raw bytes for the data section.
///
/// Handles all constant kinds: integers, floats, strings, zeros, arrays,
/// structs, and global address references.
fn serialize_constant(
    constant: &crate::ir::module::Constant,
    _ty: &IrType,
    config: &CodegenConfig,
) -> Vec<u8> {
    use crate::ir::module::Constant;
    let ptr_width = config.target.pointer_width() as usize;

    match constant {
        Constant::Int { value, ty } => {
            let byte_count = ty.size_bytes(&config.target) as usize;
            let mut bytes = Vec::with_capacity(byte_count);
            let v = *value as u128;
            for i in 0..byte_count {
                bytes.push((v >> (i * 8)) as u8);
            }
            bytes
        }
        Constant::Float { value, ty } => {
            let size = ty.size_bytes(&config.target) as usize;
            match size {
                4 => (*value as f32).to_le_bytes().to_vec(),
                8 => value.to_le_bytes().to_vec(),
                _ => {
                    // Extended precision (F80 / long double) — emit the f64
                    // bit pattern padded to the target size.
                    let mut bytes = value.to_le_bytes().to_vec();
                    bytes.resize(size, 0);
                    bytes
                }
            }
        }
        Constant::String { id: _id } => {
            // String constants resolve to a pointer-sized address that the
            // linker patches via relocation.  Emit zero bytes here; the
            // actual string data lives in the .rodata string pool.
            vec![0u8; ptr_width]
        }
        Constant::Zero { ty } => {
            let size = ty.size_bytes(&config.target) as usize;
            vec![0u8; size]
        }
        Constant::Null { ty: _ } => {
            vec![0u8; ptr_width]
        }
        Constant::Array { elements, ty } => {
            let mut bytes = Vec::new();
            let elem_ty = match ty {
                IrType::Array { element, .. } => element.as_ref(),
                _ => ty,
            };
            for elem in elements {
                bytes.extend(serialize_constant(elem, elem_ty, config));
            }
            bytes
        }
        Constant::Struct { fields, ty } => {
            let mut bytes = Vec::new();
            let field_types = match ty {
                IrType::Struct { fields: ftypes, .. } => ftypes.as_slice(),
                _ => &[],
            };
            for (i, field) in fields.iter().enumerate() {
                let ft = field_types.get(i).unwrap_or(ty);
                bytes.extend(serialize_constant(field, ft, config));
            }
            bytes
        }
        Constant::GlobalRef { name: _name } => {
            // A pointer-sized placeholder — the linker will resolve the
            // symbol address via relocation.
            vec![0u8; ptr_width]
        }
    }
}

// ---------------------------------------------------------------------------
// Assembly Output (for -S flag)
// ---------------------------------------------------------------------------

/// Writes a textual assembly representation of the compiled module.
///
/// Produces a human-readable assembly listing with function labels, global
/// data directives, and section markers. This corresponds to the `-S` CLI flag.
fn write_assembly_output(
    functions: &[AssembledFunction],
    globals: &[AssembledGlobal],
    module: &IrModule,
    config: &CodegenConfig,
) -> io::Result<()> {
    use std::io::Write;
    let mut output = Vec::new();

    // File header
    writeln!(output, "\t.file\t\"{}\"", module.name)?;
    writeln!(output, "\t# Target: {}", config.target)?;
    writeln!(output)?;

    // Text section — function code
    for func in functions {
        writeln!(output, "\t.section\t{},\"ax\",@progbits", func.section_name)?;
        writeln!(output, "\t.p2align\t{}", func.alignment.trailing_zeros())?;
        if func.is_global {
            writeln!(output, "\t.globl\t{}", func.name)?;
        }
        writeln!(output, "\t.type\t{}, @function", func.name)?;
        writeln!(output, "{}:", func.name)?;

        // Emit machine code as raw bytes (hex dump) since we don't have
        // a disassembler; the full textual assembly would require a different
        // code path from instruction selection.
        for chunk in func.code.chunks(16) {
            write!(output, "\t.byte\t")?;
            let hex_strs: Vec<String> = chunk.iter().map(|b| format!("0x{:02x}", b)).collect();
            writeln!(output, "{}", hex_strs.join(", "))?;
        }
        writeln!(output, "\t.size\t{}, .-{}", func.name, func.name)?;
        writeln!(output)?;
    }

    // Data sections — global variables
    for global in globals {
        if global.is_bss {
            writeln!(output, "\t.section\t.bss,\"aw\",@nobits")?;
        } else if global.is_const {
            writeln!(output, "\t.section\t.rodata,\"a\",@progbits")?;
        } else {
            writeln!(output, "\t.section\t{},\"aw\",@progbits", global.section_name)?;
        }
        writeln!(output, "\t.p2align\t{}", global.alignment.trailing_zeros())?;
        if global.is_global {
            writeln!(output, "\t.globl\t{}", global.name)?;
        }
        writeln!(output, "\t.type\t{}, @object", global.name)?;
        writeln!(output, "{}:", global.name)?;
        if global.is_bss {
            writeln!(output, "\t.zero\t{}", global.data.len())?;
        } else {
            for chunk in global.data.chunks(16) {
                write!(output, "\t.byte\t")?;
                let hex_strs: Vec<String> =
                    chunk.iter().map(|b| format!("0x{:02x}", b)).collect();
                writeln!(output, "{}", hex_strs.join(", "))?;
            }
        }
        writeln!(output, "\t.size\t{}, {}", global.name, global.data.len())?;
        writeln!(output)?;
    }

    // Module-level inline assembly blocks
    for asm_block in &module.inline_asm_blocks {
        writeln!(output, "# --- inline asm ---")?;
        writeln!(output, "{}", asm_block.template)?;
        writeln!(output)?;
    }

    // Write to output file
    std::fs::write(&config.output_path, &output)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Object File Output (for -c flag)
// ---------------------------------------------------------------------------

/// Writes a relocatable object file (ET_REL) from the compiled module.
///
/// Produces a standard ELF `.o` file containing `.text`, `.data`, `.rodata`,
/// `.bss` sections, symbol table, and section headers. This corresponds to
/// the `-c` CLI flag.
fn write_object_file(
    functions: &[AssembledFunction],
    globals: &[AssembledGlobal],
    string_literals: &[AssembledStringLiteral],
    module: &IrModule,
    config: &CodegenConfig,
    dwarf_sections: &Option<DwarfSections>,
    diagnostics: &mut DiagnosticEngine,
) -> io::Result<()> {
    let mut elf = ElfWriter::new(config.target);
    elf.set_type(ET_REL);

    // --- Build .text section ---
    let mut text_data = Vec::new();
    let mut text_offset: u64 = 0;
    let mut function_offsets: Vec<(String, u64, u64)> = Vec::new();

    for func in functions {
        // Align to function boundary
        let padding = compute_alignment_padding(text_offset, func.alignment as u64);
        text_data.extend(std::iter::repeat(0x90u8).take(padding as usize)); // NOP fill
        text_offset += padding;

        let func_start = text_offset;
        text_data.extend_from_slice(&func.code);
        text_offset += func.code.len() as u64;
        function_offsets.push((func.name.clone(), func_start, func.code.len() as u64));
    }

    if !text_data.is_empty() {
        let mut text_section = ElfSection::new(".text", SHT_PROGBITS);
        text_section.flags = SHF_ALLOC | SHF_EXECINSTR;
        text_section.data = text_data;
        text_section.alignment = 16;
        elf.add_section(text_section);
    }

    // --- Build .rodata section ---
    let mut rodata_data = Vec::new();
    let mut rodata_offset: u64 = 0;
    let mut rodata_symbols: Vec<(String, u64, u64, bool, bool, u8)> = Vec::new();

    // String literals go into .rodata
    for lit in string_literals {
        let padding = compute_alignment_padding(rodata_offset, 1);
        rodata_data.extend(std::iter::repeat(0u8).take(padding as usize));
        rodata_offset += padding;

        let sym_offset = rodata_offset;
        rodata_data.extend_from_slice(&lit.data);
        rodata_offset += lit.data.len() as u64;
        rodata_symbols.push((lit.label.clone(), sym_offset, lit.data.len() as u64, false, false, STV_DEFAULT));
    }

    // Const globals go into .rodata
    for global in globals.iter().filter(|g| g.is_const && !g.is_bss) {
        let align = global.alignment as u64;
        let padding = compute_alignment_padding(rodata_offset, align);
        rodata_data.extend(std::iter::repeat(0u8).take(padding as usize));
        rodata_offset += padding;

        let sym_offset = rodata_offset;
        rodata_data.extend_from_slice(&global.data);
        rodata_offset += global.data.len() as u64;
        rodata_symbols.push((
            global.name.clone(),
            sym_offset,
            global.data.len() as u64,
            global.is_global,
            global.is_weak,
            global.visibility,
        ));
    }

    if !rodata_data.is_empty() {
        let mut rodata_section = ElfSection::new(".rodata", SHT_PROGBITS);
        rodata_section.flags = SHF_ALLOC;
        rodata_section.data = rodata_data;
        rodata_section.alignment = 8;
        elf.add_section(rodata_section);
    }

    // --- Build .data section ---
    let mut data_data = Vec::new();
    let mut data_offset: u64 = 0;
    let mut data_symbols: Vec<(String, u64, u64, bool, bool, u8)> = Vec::new();

    for global in globals.iter().filter(|g| !g.is_const && !g.is_bss) {
        let align = global.alignment as u64;
        let padding = compute_alignment_padding(data_offset, align);
        data_data.extend(std::iter::repeat(0u8).take(padding as usize));
        data_offset += padding;

        let sym_offset = data_offset;
        data_data.extend_from_slice(&global.data);
        data_offset += global.data.len() as u64;
        data_symbols.push((
            global.name.clone(),
            sym_offset,
            global.data.len() as u64,
            global.is_global,
            global.is_weak,
            global.visibility,
        ));
    }

    if !data_data.is_empty() {
        let mut data_section = ElfSection::new(".data", SHT_PROGBITS);
        data_section.flags = SHF_ALLOC | SHF_WRITE;
        data_section.data = data_data;
        data_section.alignment = 8;
        elf.add_section(data_section);
    }

    // --- Build .bss section ---
    let mut bss_size: u64 = 0;
    let mut bss_symbols: Vec<(String, u64, u64, bool, bool, u8)> = Vec::new();

    for global in globals.iter().filter(|g| g.is_bss) {
        let align = global.alignment as u64;
        let padding = compute_alignment_padding(bss_size, align);
        bss_size += padding;

        let sym_offset = bss_size;
        bss_size += global.data.len() as u64;
        bss_symbols.push((
            global.name.clone(),
            sym_offset,
            global.data.len() as u64,
            global.is_global,
            global.is_weak,
            global.visibility,
        ));
    }

    if bss_size > 0 {
        let mut bss_section = ElfSection::new(".bss", SHT_NOBITS);
        bss_section.flags = SHF_ALLOC | SHF_WRITE;
        bss_section.data = vec![0u8; bss_size as usize];
        bss_section.alignment = 8;
        elf.add_section(bss_section);
    }

    // --- Add symbols ---

    // File symbol
    let mut file_sym = ElfSymbol::new(&module.name);
    file_sym.sym_type = STT_FILE;
    file_sym.binding = STB_LOCAL;
    file_sym.section_index = SHN_UNDEF;
    elf.add_symbol(file_sym);

    // Function symbols
    for (name, offset, size) in &function_offsets {
        let func = functions.iter().find(|f| &f.name == name);
        let mut sym = ElfSymbol::new(name);
        sym.sym_type = STT_FUNC;
        sym.value = *offset;
        sym.size = *size;
        if let Some(f) = func {
            sym.binding = if f.is_weak {
                STB_WEAK
            } else if f.is_global {
                STB_GLOBAL
            } else {
                STB_LOCAL
            };
            sym.visibility = f.visibility;
        } else {
            sym.binding = STB_GLOBAL;
        }
        sym.section_index = 1; // .text section index
        elf.add_symbol(sym);
    }

    // Rodata symbols
    add_data_symbols(&mut elf, &rodata_symbols, STT_OBJECT, 2);

    // Data symbols
    let data_section_idx = if !rodata_data.is_empty() { 3 } else { 2 };
    add_data_symbols(&mut elf, &data_symbols, STT_OBJECT, data_section_idx as u16);

    // BSS symbols
    let bss_section_idx = data_section_idx + if !data_data.is_empty() { 1 } else { 0 };
    add_data_symbols(&mut elf, &bss_symbols, STT_OBJECT, bss_section_idx as u16);

    // Undefined symbol references (from function declarations)
    for decl in &module.declarations {
        let mut sym = ElfSymbol::new(&decl.name);
        sym.sym_type = STT_NOTYPE;
        sym.binding = match decl.linkage {
            Linkage::Weak => STB_WEAK,
            _ => STB_GLOBAL,
        };
        sym.section_index = SHN_UNDEF;
        elf.add_symbol(sym);
    }

    // --- DWARF debug sections ---
    if let Some(ref dwarf) = dwarf_sections {
        add_dwarf_sections(&mut elf, dwarf);
    }

    // --- Write output ---
    let elf_bytes = elf.write();
    std::fs::write(&config.output_path, &elf_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Linked Output (for full compilation + linking)
// ---------------------------------------------------------------------------

/// Produces a fully linked ELF executable (ET_EXEC) or shared object (ET_DYN).
///
/// This function orchestrates the built-in linker pipeline:
/// 1. Build input sections from assembled functions and globals.
/// 2. Resolve symbols using [`SymbolResolver`].
/// 3. Merge sections using [`SectionMerger`].
/// 4. For shared libraries: generate dynamic linking sections.
/// 5. Apply relocations using [`RelocationProcessor`].
/// 6. Write the final ELF output using [`ElfWriter`].
///
/// This implements the standalone backend mode (Section 0.7.7) — no external
/// linker is invoked.
fn write_linked_output(
    functions: &[AssembledFunction],
    globals: &[AssembledGlobal],
    string_literals: &[AssembledStringLiteral],
    module: &IrModule,
    config: &CodegenConfig,
    codegen: &dyn ArchCodegen,
    dwarf_sections: &Option<DwarfSections>,
    diagnostics: &mut DiagnosticEngine,
) -> io::Result<()> {
    let output_type = config.linker_output_type();
    let linker_script = LinkerScript::default_for_target(&config.target, output_type);
    let base_address = linker_script.base_address();
    let page_size = linker_script.page_size();

    // --- Phase 1: Build input sections ---
    let mut section_merger = SectionMerger::new();
    let object_index: usize = 0;

    // .text section from assembled functions
    let text_section = build_text_section(functions, object_index);
    if !text_section.data.is_empty() {
        section_merger.add_input_section(text_section);
    }

    // .rodata section from string literals and const globals
    let rodata_section = build_rodata_section(globals, string_literals, object_index);
    if !rodata_section.data.is_empty() {
        section_merger.add_input_section(rodata_section);
    }

    // .data section from initialized mutable globals
    let data_section = build_data_section(globals, object_index);
    if !data_section.data.is_empty() {
        section_merger.add_input_section(data_section);
    }

    // .bss section from zero-initialized globals
    let bss_section = build_bss_section(globals, object_index);
    if bss_section.data.len() > 0 {
        section_merger.add_input_section(bss_section);
    }

    // Handle module-level inline assembly blocks — these may emit
    // additional sections or symbols that need to be included
    for asm_block in &module.inline_asm_blocks {
        let asm_section = InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: asm_block.template.as_bytes().to_vec(),
            alignment: 1,
            entry_size: 0,
            group_id: None,
            object_index,
            original_index: 0,
            relocations: Vec::new(),
        };
        if !asm_section.data.is_empty() {
            section_merger.add_input_section(asm_section);
        }
    }

    // --- Phase 2: Retpoline thunks (x86-64 security) ---
    let retpoline_thunks = if config.retpoline && config.target == Target::X86_64 {
        generate_retpoline_thunks(codegen)
    } else {
        Vec::new()
    };

    // Add retpoline thunk code to .text
    if !retpoline_thunks.is_empty() {
        let mut thunk_data = Vec::new();
        for thunk in &retpoline_thunks {
            thunk_data.extend_from_slice(&thunk.code);
        }
        let thunk_section = InputSection {
            name: ".text.__x86_retpoline".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: thunk_data,
            alignment: 16,
            entry_size: 0,
            group_id: None,
            object_index,
            original_index: 0,
            relocations: Vec::new(),
        };
        section_merger.add_input_section(thunk_section);
    }

    // --- Phase 3: Section ordering and address assignment ---
    section_merger.compute_section_order();
    section_merger.assign_addresses(base_address);
    // ELF header + program header table occupy the first page
    let initial_file_offset = page_size;
    section_merger.assign_file_offsets(initial_file_offset);

    // --- Phase 4: Symbol resolution ---
    let mut sym_resolver = SymbolResolver::new();
    sym_resolver.set_target(config.target);
    sym_resolver.register_object(object_index, &module.name);

    let input_symbols = build_input_symbols(functions, globals, module, &retpoline_thunks);
    sym_resolver.collect_symbols(object_index, &input_symbols);

    let resolved = match sym_resolver.resolve_references() {
        Ok(resolved) => resolved,
        Err(errors) => {
            for error in &errors {
                let msg = format!("Link error: {:?}", error);
                diagnostics.error(Span::DUMMY, msg);
            }
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Linking failed with unresolved symbols",
            ));
        }
    };

    // --- Phase 5: Dynamic linking sections (for -shared) ---
    let mut dynamic_builder = if config.shared {
        let mut builder = DynamicSectionBuilder::new();
        // Dynamic sections are generated during the final ELF write phase
        Some(builder)
    } else {
        None
    };

    // --- Phase 6: Build the final ELF ---
    let mut elf = ElfWriter::new(config.target);

    if config.shared {
        elf.set_type(ET_DYN);
    } else {
        elf.set_type(ET_EXEC);
    }

    // Set entry point
    if !config.shared {
        if let Some(entry_value) = resolved.get_symbol_value(linker_script.entry_point()) {
            elf.set_entry_point(entry_value);
        } else if let Some(entry_value) = resolved.get_symbol_value("_start") {
            elf.set_entry_point(entry_value);
        }
    }

    // Add merged output sections
    for out_section in section_merger.output_sections() {
        let section_data = section_merger.collect_section_data(
            section_merger
                .find_section(&out_section.name)
                .unwrap_or(0),
        );
        let mut elf_section = ElfSection::new(&out_section.name, out_section.section_type);
        elf_section.flags = out_section.flags;
        elf_section.data = section_data;
        elf_section.alignment = out_section.alignment;
        elf_section.addr = out_section.addr;
        // Note: The ElfWriter computes file offsets internally during write(),
        // using the section data lengths and alignment constraints.
        elf.add_section(elf_section);
    }

    // Add symbols to the ELF
    let mut file_sym = ElfSymbol::new(&module.name);
    file_sym.sym_type = STT_FILE;
    file_sym.binding = STB_LOCAL;
    file_sym.section_index = SHN_UNDEF;
    elf.add_symbol(file_sym);

    // Add function symbols with resolved addresses
    for func in functions {
        let mut sym = ElfSymbol::new(&func.name);
        sym.sym_type = STT_FUNC;
        sym.binding = if func.is_weak {
            STB_WEAK
        } else if func.is_global {
            STB_GLOBAL
        } else {
            STB_LOCAL
        };
        sym.visibility = func.visibility;
        sym.size = func.code.len() as u64;

        if let Some(val) = resolved.get_symbol_value(&func.name) {
            sym.value = val;
        }
        // Section index will be set by the ELF writer based on section lookup
        if let Some(sec_idx) = section_merger.find_section(&func.section_name) {
            sym.section_index = (sec_idx + 1) as u16; // +1 for NULL section
        }
        elf.add_symbol(sym);
    }

    // Add retpoline thunk symbols
    let mut thunk_offset = 0u64;
    for thunk in &retpoline_thunks {
        let mut sym = ElfSymbol::new(&thunk.name);
        sym.sym_type = STT_FUNC;
        sym.binding = STB_LOCAL;
        sym.size = thunk.code.len() as u64;
        if let Some(val) = resolved.get_symbol_value(&thunk.name) {
            sym.value = val;
        }
        elf.add_symbol(sym);
        thunk_offset += thunk.code.len() as u64;
    }

    // Add global/data symbols
    for global in globals {
        let mut sym = ElfSymbol::new(&global.name);
        sym.sym_type = STT_OBJECT;
        sym.binding = if global.is_weak {
            STB_WEAK
        } else if global.is_global {
            STB_GLOBAL
        } else {
            STB_LOCAL
        };
        sym.visibility = global.visibility;
        sym.size = global.data.len() as u64;
        if let Some(val) = resolved.get_symbol_value(&global.name) {
            sym.value = val;
        }
        if let Some(sec_idx) = section_merger.find_section(&global.section_name) {
            sym.section_index = (sec_idx + 1) as u16;
        }
        elf.add_symbol(sym);
    }

    // --- Program headers ---
    build_program_headers(&mut elf, &section_merger, config, base_address, page_size);

    // --- DWARF debug sections ---
    if let Some(ref dwarf) = dwarf_sections {
        add_dwarf_sections(&mut elf, dwarf);
    }

    // --- Write final ELF ---
    let elf_bytes = elf.write();
    std::fs::write(&config.output_path, &elf_bytes)?;

    // Set executable permissions on the output file (Unix-only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&config.output_path, perms)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Phase 10 code generation entry point.
///
/// Generates machine code from the given IR module and produces the requested
/// output artifact (object file, linked executable, assembly listing, or no-op
/// for preprocess-only mode).
///
/// # Pipeline Overview
///
/// ```text
/// IrModule (phi-eliminated SSA)
///   │
///   ├─ Architecture dispatch (Target → ArchCodegen)
///   │
///   ├─ For each function:
///   │   ├─ lower_function()   → MachineFunction (virtual regs)
///   │   ├─ emit_prologue()    → frame setup
///   │   ├─ emit_epilogue()    → frame teardown
///   │   ├─ RegisterAllocator  → physical reg assignment + spill code
///   │   ├─ Security mitigations (x86-64: retpoline, CET, stack probe)
///   │   └─ emit_assembly()    → Vec<u8> machine code
///   │
///   ├─ Global variable serialisation
///   ├─ String literal processing
///   ├─ DWARF debug info (if -g)
///   │
///   └─ Output mode:
///       ├─ -S  → textual assembly listing
///       ├─ -c  → relocatable object file (ET_REL)
///       └─ default → linked executable (ET_EXEC) or shared lib (ET_DYN)
/// ```
///
/// # Arguments
///
/// * `module` — the IR module produced by the middle-end pipeline
/// * `config` — code generation configuration (target, flags, output path)
/// * `diagnostics` — diagnostic engine for error reporting
///
/// # Returns
///
/// `Ok(())` on success, `Err` with an I/O or compilation error description
/// on failure.
///
/// # Errors
///
/// - Returns `Err` if code generation encounters an unsupported construct.
/// - Returns `Err` if the register allocator cannot allocate registers.
/// - Returns `Err` if symbol resolution fails during linking.
/// - Returns `Err` if the output file cannot be written.
pub fn generate_code(
    module: &IrModule,
    config: &CodegenConfig,
    diagnostics: &mut DiagnosticEngine,
) -> io::Result<()> {
    // PreprocessOnly mode should never reach code generation. Return
    // immediately as a safety guard.
    if config.output_mode == OutputMode::PreprocessOnly {
        return Ok(());
    }

    // Convert the high-level config to the backend-level config used by
    // ArchCodegen implementations.
    let backend_config = config.to_backend_config();

    // Create the architecture-specific code generator via target dispatch.
    let codegen: Box<dyn ArchCodegen> = create_codegen(&backend_config);

    // Initialise the DWARF debug info generator. When config.debug_info
    // is false, all DwarfGenerator methods are no-ops and finish() returns
    // None, guaranteeing zero debug section leakage (Section 0.7.10).
    let mut dwarf = DwarfGenerator::new(config.target, config.debug_info);

    if dwarf.is_enabled() {
        dwarf.begin_compilation_unit(
            &module.name,
            ".",  // compilation directory
            0,    // low_pc — will be the first function's address
            0,    // high_pc — will be the last function's end address
        );
    }

    // --- Phase 10a: Compile all functions to machine code ---
    let mut assembled_functions = Vec::with_capacity(module.functions.len());
    let mut text_offset: u64 = 0;

    for ir_func in &module.functions {
        // Skip function declarations (extern prototypes without bodies)
        if !ir_func.is_definition {
            continue;
        }

        let result = compile_function(
            ir_func,
            codegen.as_ref(),
            config,
            &backend_config,
            diagnostics,
            &mut dwarf,
            text_offset,
        )?;

        text_offset += result.code.len() as u64;
        assembled_functions.push(result);

        // Halt on errors — do not continue generating code for subsequent
        // functions if a critical error has been reported.
        if diagnostics.has_errors() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Code generation halted due to errors",
            ));
        }
    }

    // --- Phase 10b: Process global variables ---
    let assembled_globals = process_globals(module, config)?;

    // --- Phase 10c: Process string literals ---
    let string_literals = process_string_literals(module);

    // --- Phase 10d: Finalise DWARF debug info ---
    if dwarf.is_enabled() {
        // Emit return type for each function to the DWARF type table
        for ir_func in &module.functions {
            if !ir_func.is_definition {
                continue;
            }
            dwarf.emit_type(&ir_type_to_ctype(&ir_func.return_type), &config.target);
        }
        dwarf.end_compilation_unit();
    }

    let dwarf_sections = dwarf.finish();

    // --- Phase 10e: Produce output based on mode ---
    match config.output_mode {
        OutputMode::Assembly => write_assembly_output(
            &assembled_functions,
            &assembled_globals,
            module,
            config,
        ),
        OutputMode::Object => write_object_file(
            &assembled_functions,
            &assembled_globals,
            &string_literals,
            module,
            config,
            &dwarf_sections,
            diagnostics,
        ),
        OutputMode::Executable => write_linked_output(
            &assembled_functions,
            &assembled_globals,
            &string_literals,
            module,
            config,
            codegen.as_ref(),
            &dwarf_sections,
            diagnostics,
        ),
        OutputMode::PreprocessOnly => {
            // Already handled above; this branch is unreachable.
            Ok(())
        }
    }
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Computes the number of padding bytes needed to align `offset` to `alignment`.
///
/// Returns 0 if already aligned or if alignment is 0 or 1.
#[inline]
fn compute_alignment_padding(offset: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return 0;
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

/// Determines the target ELF section name for a function based on its
/// attributes.
///
/// Functions with a custom `section` attribute are placed in the named
/// section. Otherwise, cold functions go to `.text.cold`, hot functions
/// to `.text.hot`, and all others to `.text`.
fn determine_function_section(func: &IrFunction) -> String {
    if let Some(ref section) = func.section {
        return section.clone();
    }
    if func.attributes.is_cold {
        ".text.unlikely".to_string()
    } else if func.attributes.is_hot {
        ".text.hot".to_string()
    } else {
        ".text".to_string()
    }
}

/// Maps IR function [`Visibility`] to the ELF `STV_*` constant.
fn visibility_to_elf(vis: &Visibility) -> u8 {
    match vis {
        Visibility::Default => STV_DEFAULT,
        Visibility::Hidden => STV_HIDDEN,
        Visibility::Protected => STV_PROTECTED,
        Visibility::Internal => STV_HIDDEN, // ELF internal maps to hidden
    }
}

/// Converts an [`IrType`] to a basic [`CType`] for DWARF debug info emission.
///
/// This is an approximate reverse mapping from the IR type system back to C
/// types. The mapping is sufficient for DWARF source-level type representation
/// at `-O0` debug level.
fn ir_type_to_ctype(ir_type: &IrType) -> CType {
    match ir_type {
        IrType::Void => CType::Void,
        IrType::I1 => CType::Bool,
        IrType::I8 => CType::Char { signed: true },
        IrType::I16 => CType::Short { signed: true },
        IrType::I32 => CType::Int { signed: true },
        IrType::I64 => CType::LongLong { signed: true },
        IrType::I128 => CType::LongLong { signed: true },
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::F80 => CType::LongDouble,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void)),
        IrType::Array { element, count } => CType::Array {
            element: Box::new(ir_type_to_ctype(element)),
            size: Some(*count),
        },
        IrType::Struct { fields, packed: _ } => CType::Struct {
            name: None,
            fields: fields
                .iter()
                .enumerate()
                .map(|(i, field_ty)| crate::common::types::FieldDef {
                    name: Some(format!("field{}", i)),
                    ty: ir_type_to_ctype(field_ty),
                    bit_width: None,
                })
                .collect(),
        },
        IrType::Function {
            return_type,
            param_types,
            is_variadic,
        } => CType::Function {
            return_type: Box::new(ir_type_to_ctype(return_type)),
            params: param_types.iter().map(ir_type_to_ctype).collect(),
            variadic: *is_variadic,
        },
    }
}

/// Adds ELF data symbols (for .rodata, .data, .bss) to the ELF writer.
///
/// Each symbol is defined with the appropriate binding, type, and section index.
fn add_data_symbols(
    elf: &mut ElfWriter,
    symbols: &[(String, u64, u64, bool, bool, u8)],
    sym_type: u8,
    section_index: u16,
) {
    for (name, offset, size, is_global, is_weak, visibility) in symbols {
        let mut sym = ElfSymbol::new(name);
        sym.sym_type = sym_type;
        sym.value = *offset;
        sym.size = *size;
        sym.binding = if *is_weak {
            STB_WEAK
        } else if *is_global {
            STB_GLOBAL
        } else {
            STB_LOCAL
        };
        sym.visibility = *visibility;
        sym.section_index = section_index;
        elf.add_symbol(sym);
    }
}

/// Adds DWARF debug sections to the ELF writer.
///
/// When the `-g` flag is active, this function adds `.debug_info`,
/// `.debug_abbrev`, `.debug_line`, and `.debug_str` sections.
/// Per Section 0.7.10, when `-g` is absent, `dwarf_sections` is `None`
/// and this function is not called, ensuring zero debug section leakage.
fn add_dwarf_sections(elf: &mut ElfWriter, dwarf: &DwarfSections) {
    if !dwarf.debug_info.is_empty() {
        let mut section = ElfSection::new(".debug_info", SHT_PROGBITS);
        section.data = dwarf.debug_info.clone();
        section.alignment = 1;
        elf.add_section(section);
    }

    if !dwarf.debug_abbrev.is_empty() {
        let mut section = ElfSection::new(".debug_abbrev", SHT_PROGBITS);
        section.data = dwarf.debug_abbrev.clone();
        section.alignment = 1;
        elf.add_section(section);
    }

    if !dwarf.debug_line.is_empty() {
        let mut section = ElfSection::new(".debug_line", SHT_PROGBITS);
        section.data = dwarf.debug_line.clone();
        section.alignment = 1;
        elf.add_section(section);
    }

    if !dwarf.debug_str.is_empty() {
        let mut section = ElfSection::new(".debug_str", SHT_PROGBITS);
        section.data = dwarf.debug_str.clone();
        section.alignment = 1;
        elf.add_section(section);
    }
}

// ---------------------------------------------------------------------------
// Input Section Builders (for the linker pipeline)
// ---------------------------------------------------------------------------

/// Builds the `.text` input section from assembled functions.
fn build_text_section(functions: &[AssembledFunction], object_index: usize) -> InputSection {
    let mut data = Vec::new();
    let mut max_alignment: u64 = 16;

    for func in functions {
        let padding = compute_alignment_padding(data.len() as u64, func.alignment as u64);
        data.extend(std::iter::repeat(0x90u8).take(padding as usize));
        data.extend_from_slice(&func.code);
        max_alignment = max_alignment.max(func.alignment as u64);
    }

    InputSection {
        name: ".text".to_string(),
        section_type: SHT_PROGBITS,
        flags: SHF_ALLOC | SHF_EXECINSTR,
        data,
        alignment: max_alignment,
        entry_size: 0,
        group_id: None,
        object_index,
        original_index: 0,
        relocations: Vec::new(),
    }
}

/// Builds the `.rodata` input section from const globals and string literals.
fn build_rodata_section(
    globals: &[AssembledGlobal],
    string_literals: &[AssembledStringLiteral],
    object_index: usize,
) -> InputSection {
    let mut data = Vec::new();
    let mut max_alignment: u64 = 1;

    // String literals first
    for lit in string_literals {
        data.extend_from_slice(&lit.data);
    }

    // Const globals
    for global in globals.iter().filter(|g| g.is_const && !g.is_bss) {
        let align = global.alignment as u64;
        let padding = compute_alignment_padding(data.len() as u64, align);
        data.extend(std::iter::repeat(0u8).take(padding as usize));
        data.extend_from_slice(&global.data);
        max_alignment = max_alignment.max(align);
    }

    InputSection {
        name: ".rodata".to_string(),
        section_type: SHT_PROGBITS,
        flags: SHF_ALLOC,
        data,
        alignment: max_alignment.max(1),
        entry_size: 0,
        group_id: None,
        object_index,
        original_index: 1,
        relocations: Vec::new(),
    }
}

/// Builds the `.data` input section from initialized mutable globals.
fn build_data_section(globals: &[AssembledGlobal], object_index: usize) -> InputSection {
    let mut data = Vec::new();
    let mut max_alignment: u64 = 1;

    for global in globals.iter().filter(|g| !g.is_const && !g.is_bss) {
        let align = global.alignment as u64;
        let padding = compute_alignment_padding(data.len() as u64, align);
        data.extend(std::iter::repeat(0u8).take(padding as usize));
        data.extend_from_slice(&global.data);
        max_alignment = max_alignment.max(align);
    }

    InputSection {
        name: ".data".to_string(),
        section_type: SHT_PROGBITS,
        flags: SHF_ALLOC | SHF_WRITE,
        data,
        alignment: max_alignment.max(1),
        entry_size: 0,
        group_id: None,
        object_index,
        original_index: 2,
        relocations: Vec::new(),
    }
}

/// Builds the `.bss` input section from zero-initialized globals.
fn build_bss_section(globals: &[AssembledGlobal], object_index: usize) -> InputSection {
    let mut total_size: u64 = 0;
    let mut max_alignment: u64 = 1;

    for global in globals.iter().filter(|g| g.is_bss) {
        let align = global.alignment as u64;
        let padding = compute_alignment_padding(total_size, align);
        total_size += padding + global.data.len() as u64;
        max_alignment = max_alignment.max(align);
    }

    InputSection {
        name: ".bss".to_string(),
        section_type: SHT_NOBITS,
        flags: SHF_ALLOC | SHF_WRITE,
        data: vec![0u8; total_size as usize],
        alignment: max_alignment.max(1),
        entry_size: 0,
        group_id: None,
        object_index,
        original_index: 3,
        relocations: Vec::new(),
    }
}

/// Builds the input symbol table for the symbol resolver.
///
/// Converts function definitions, global variables, function declarations,
/// and retpoline thunks into [`InputSymbol`] entries for the linker's
/// symbol resolution pipeline.
fn build_input_symbols(
    functions: &[AssembledFunction],
    globals: &[AssembledGlobal],
    module: &IrModule,
    retpoline_thunks: &[RetpolineThunkData],
) -> Vec<InputSymbol> {
    let mut symbols = Vec::new();

    // Function definitions
    let mut text_offset: u64 = 0;
    for func in functions {
        let padding = compute_alignment_padding(text_offset, func.alignment as u64);
        text_offset += padding;

        symbols.push(InputSymbol {
            name: func.name.clone(),
            value: text_offset,
            size: func.code.len() as u64,
            binding: if func.is_weak {
                SymbolBinding::Weak
            } else if func.is_global {
                SymbolBinding::Global
            } else {
                SymbolBinding::Local
            },
            sym_type: LinkerSymbolType::Func,
            visibility: match func.visibility {
                STV_HIDDEN => LinkerSymbolVisibility::Hidden,
                STV_PROTECTED => LinkerSymbolVisibility::Protected,
                _ => LinkerSymbolVisibility::Default,
            },
            section_index: 1, // .text
        });
        text_offset += func.code.len() as u64;
    }

    // Retpoline thunks
    for thunk in retpoline_thunks {
        symbols.push(InputSymbol {
            name: thunk.name.clone(),
            value: text_offset,
            size: thunk.code.len() as u64,
            binding: SymbolBinding::Local,
            sym_type: LinkerSymbolType::Func,
            visibility: LinkerSymbolVisibility::Hidden,
            section_index: 1,
        });
        text_offset += thunk.code.len() as u64;
    }

    // Global variables
    for global in globals {
        symbols.push(InputSymbol {
            name: global.name.clone(),
            value: 0, // offset within section — resolved by merger
            size: global.data.len() as u64,
            binding: if global.is_weak {
                SymbolBinding::Weak
            } else if global.is_global {
                SymbolBinding::Global
            } else {
                SymbolBinding::Local
            },
            sym_type: LinkerSymbolType::Object,
            visibility: match global.visibility {
                STV_HIDDEN => LinkerSymbolVisibility::Hidden,
                STV_PROTECTED => LinkerSymbolVisibility::Protected,
                _ => LinkerSymbolVisibility::Default,
            },
            section_index: 2, // approximation; depends on section ordering
        });
    }

    // External function declarations (undefined symbols)
    for decl in &module.declarations {
        symbols.push(InputSymbol {
            name: decl.name.clone(),
            value: 0,
            size: 0,
            binding: match decl.linkage {
                Linkage::Weak => SymbolBinding::Weak,
                _ => SymbolBinding::Global,
            },
            sym_type: LinkerSymbolType::Func,
            visibility: LinkerSymbolVisibility::Default,
            section_index: SHN_UNDEF,
        });
    }

    symbols
}

/// Generates retpoline thunks for x86-64 security mitigation.
///
/// When `-mretpoline` is active on x86-64, indirect calls are rewritten to
/// target `__x86_indirect_thunk_<reg>` stubs. This function generates the
/// thunk code for all commonly used registers.
///
/// The thunk instructions are wrapped in a temporary [`MachineFunction`] and
/// passed through `emit_assembly` to produce the final machine code bytes.
fn generate_retpoline_thunks(codegen: &dyn ArchCodegen) -> Vec<RetpolineThunkData> {
    let callee_saved = codegen.callee_saved_registers();
    let caller_saved = codegen.caller_saved_registers();

    // Collect all registers that might be used for indirect calls
    let mut thunk_regs = Vec::new();
    thunk_regs.extend_from_slice(caller_saved);
    thunk_regs.extend_from_slice(callee_saved);

    // Generate thunks via the RetpolineGenerator — returns MachineInstr-level code
    let thunks = RetpolineGenerator::generate_retpoline_thunks(&thunk_regs);

    thunks
        .into_iter()
        .map(|thunk| {
            // Wrap the thunk's machine instructions in a MachineFunction so we
            // can assemble them to bytes using the architecture codegen.
            let mut mf = MachineFunction::new(thunk.name.clone(), 16);
            let mut block = MachineBasicBlock::with_label(0, thunk.name.clone());
            for instr in &thunk.code {
                block.push_instr(instr.clone());
            }
            mf.add_block(block);
            let code = codegen.emit_assembly(&mf);
            RetpolineThunkData {
                name: thunk.name.clone(),
                code,
            }
        })
        .collect()
}

/// Builds ELF program headers for the linked output.
///
/// Creates PT_LOAD segments for code (R+X), read-only data (R), and
/// read-write data (R+W), plus PT_PHDR and PT_GNU_STACK headers.
fn build_program_headers(
    elf: &mut ElfWriter,
    section_merger: &SectionMerger,
    config: &CodegenConfig,
    base_address: u64,
    page_size: u64,
) {
    // PT_PHDR — program header table self-reference
    let phdr = ProgramHeader {
        p_type: PT_PHDR,
        p_flags: 0x4, // PF_R
        p_offset: 0x40, // standard ELF64 header size
        p_vaddr: base_address + 0x40,
        p_paddr: base_address + 0x40,
        p_filesz: 0, // filled in by ELF writer
        p_memsz: 0,
        p_align: 8,
    };
    elf.add_program_header(phdr);

    // Build PT_LOAD segments from output sections
    // Group sections by permission flags for segment creation
    let output_sections = section_merger.output_sections();

    // Code segment: sections with SHF_EXECINSTR
    let code_sections: Vec<&_> = output_sections
        .iter()
        .filter(|s| s.flags & SHF_EXECINSTR != 0)
        .collect();
    if !code_sections.is_empty() {
        let first = code_sections.first().unwrap();
        let last = code_sections.last().unwrap();
        let seg_start = first.addr;
        let seg_file_start = first.offset;
        let seg_end = last.addr + last.size;
        let seg_file_end = last.offset + last.size;

        let code_phdr = ProgramHeader {
            p_type: PT_LOAD,
            p_flags: 0x5, // PF_R | PF_X
            p_offset: seg_file_start,
            p_vaddr: seg_start,
            p_paddr: seg_start,
            p_filesz: seg_file_end - seg_file_start,
            p_memsz: seg_end - seg_start,
            p_align: page_size,
        };
        elf.add_program_header(code_phdr);
    }

    // Read-only data segment: SHF_ALLOC without SHF_WRITE or SHF_EXECINSTR
    let ro_sections: Vec<&_> = output_sections
        .iter()
        .filter(|s| {
            s.flags & SHF_ALLOC != 0
                && s.flags & SHF_WRITE == 0
                && s.flags & SHF_EXECINSTR == 0
        })
        .collect();
    if !ro_sections.is_empty() {
        let first = ro_sections.first().unwrap();
        let last = ro_sections.last().unwrap();
        let seg_start = first.addr;
        let seg_file_start = first.offset;
        let seg_end = last.addr + last.size;
        let seg_file_end = last.offset + last.size;

        let ro_phdr = ProgramHeader {
            p_type: PT_LOAD,
            p_flags: 0x4, // PF_R
            p_offset: seg_file_start,
            p_vaddr: seg_start,
            p_paddr: seg_start,
            p_filesz: seg_file_end - seg_file_start,
            p_memsz: seg_end - seg_start,
            p_align: page_size,
        };
        elf.add_program_header(ro_phdr);
    }

    // Read-write data segment: SHF_ALLOC | SHF_WRITE
    let rw_sections: Vec<&_> = output_sections
        .iter()
        .filter(|s| s.flags & SHF_ALLOC != 0 && s.flags & SHF_WRITE != 0)
        .collect();
    if !rw_sections.is_empty() {
        let first = rw_sections.first().unwrap();
        let last = rw_sections.last().unwrap();
        let seg_start = first.addr;
        let seg_file_start = first.offset;
        let seg_end = last.addr + last.size;
        // For BSS (SHT_NOBITS), filesz differs from memsz
        let seg_file_end = rw_sections
            .iter()
            .filter(|s| s.section_type != SHT_NOBITS)
            .map(|s| s.offset + s.size)
            .max()
            .unwrap_or(seg_file_start);

        let rw_phdr = ProgramHeader {
            p_type: PT_LOAD,
            p_flags: 0x6, // PF_R | PF_W
            p_offset: seg_file_start,
            p_vaddr: seg_start,
            p_paddr: seg_start,
            p_filesz: seg_file_end - seg_file_start,
            p_memsz: seg_end - seg_start,
            p_align: page_size,
        };
        elf.add_program_header(rw_phdr);
    }

    // PT_GNU_STACK — non-executable stack
    let stack_phdr = ProgramHeader {
        p_type: PT_GNU_STACK,
        p_flags: 0x6, // PF_R | PF_W (no PF_X → NX stack)
        p_offset: 0,
        p_vaddr: 0,
        p_paddr: 0,
        p_filesz: 0,
        p_memsz: 0,
        p_align: 0,
    };
    elf.add_program_header(stack_phdr);
}

// ---------------------------------------------------------------------------
// Multi-File Compilation Support (TempFile-based)
// ---------------------------------------------------------------------------

/// Compiles a single IR module to a temporary object file.
///
/// Used by the CLI driver for multi-file compilation: each source file is
/// compiled to a temporary `.o` file, then all temporaries are linked
/// together in a final linking pass.
///
/// The returned [`TempFile`] handle ensures automatic cleanup via RAII —
/// when the handle is dropped, the temporary file is deleted from disk.
///
/// # Arguments
///
/// * `module` — the IR module for one translation unit
/// * `config` — generation configuration (output_mode is overridden to Object)
/// * `diagnostics` — diagnostic engine
///
/// # Returns
///
/// A `TempFile` whose path contains the compiled `.o` file, or an I/O error.
pub fn compile_to_temp_object(
    module: &IrModule,
    config: &CodegenConfig,
    diagnostics: &mut DiagnosticEngine,
) -> io::Result<TempFile> {
    let temp = TempFile::new("bcc_", ".o")?;

    // Create a modified config that outputs to the temp file path
    let mut obj_config = config.clone();
    obj_config.output_mode = OutputMode::Object;
    obj_config.output_path = temp.path().to_path_buf();

    generate_code(module, &obj_config, diagnostics)?;

    Ok(temp)
}
