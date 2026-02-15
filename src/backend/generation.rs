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
use std::path::PathBuf;

use crate::backend::aarch64::AArch64Codegen;
use crate::backend::dwarf::{DwarfGenerator, DwarfSections};
use crate::backend::elf_writer_common::{
    ElfSection, ElfSymbol, ElfWriter, ProgramHeader, ET_DYN, ET_EXEC, ET_REL, PF_R, PF_W,
    PT_DYNAMIC, PT_GNU_STACK, PT_INTERP, PT_LOAD, PT_PHDR, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE,
    SHN_UNDEF, SHT_DYNAMIC, SHT_DYNSYM, SHT_NOBITS, SHT_PROGBITS, SHT_STRTAB, STB_GLOBAL,
    STB_LOCAL, STB_WEAK, STT_FILE, STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT, STV_HIDDEN,
    STV_PROTECTED,
};
use crate::backend::i686::I686Codegen;
use crate::backend::linker_common::{
    InputRelocation, InputSection, InputSymbol, LinkError, LinkerScript,
    OutputType, SectionMerger, SymbolBinding, SymbolResolver,
    SymbolType as LinkerSymbolType, SymbolVisibility as LinkerSymbolVisibility,
};
use crate::backend::register_allocator::RegisterAllocator;
use crate::backend::riscv64::RiscV64Codegen;
use crate::backend::traits::{
    ArchCodegen, CodegenConfig as BackendCodegenConfig, MachineBasicBlock, MachineFunction,
};
use crate::backend::x86_64::security::{
    apply_security_mitigations, RetpolineGenerator, SecurityConfig,
};
use crate::backend::x86_64::X86_64Codegen;
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::target::Target;
use crate::common::temp_files::TempFile;
use crate::common::types::CType;
use crate::ir::function::{IrFunction, Linkage, Visibility};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

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
    pub(crate) fn to_backend_config(&self) -> BackendCodegenConfig {
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
    pub(crate) fn has_security_mitigations(&self) -> bool {
        self.target == Target::X86_64 && (self.retpoline || self.cf_protection)
    }

    /// Returns `true` if PIC code generation is required.
    ///
    /// PIC is required when either `-fPIC` or `-shared` is set, since
    /// shared libraries must be position-independent.
    #[inline]
    pub fn requires_pic(&self) -> bool {
        self.pic || self.shared
    }

    /// Determines the linker output type based on the configuration.
    ///
    /// Maps the high-level `OutputMode` and `shared` flag to the linker's
    /// [`OutputType`] enum for section-to-segment layout decisions.
    pub(crate) fn linker_output_type(&self) -> OutputType {
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
#[allow(dead_code)]
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
    /// Relocations emitted by the assembler for this function (linker format).
    relocations: Vec<InputRelocation>,
    /// Raw assembler relocations preserving symbol names for dynamic linking.
    asm_relocations: Vec<crate::backend::traits::AsmRelocation>,
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
    _backend_config: &BackendCodegenConfig,
    _diagnostics: &mut DiagnosticEngine,
    dwarf: &mut DwarfGenerator,
    text_offset: u64,
) -> io::Result<AssembledFunction> {
    // DEBUG: dump IR function details before lowering
    eprintln!("[DEBUG compile_function] func='{}' is_def={} blocks={} params={}",
        ir_func.name, ir_func.is_definition,
        ir_func.basic_blocks.len(), ir_func.params.len());
    for (bi, bb) in ir_func.basic_blocks.iter().enumerate() {
        let icount = bb.instructions().len();
        eprintln!("[DEBUG]   block[{}] id={:?} instructions={}", bi, bb.id, icount);
        for (ii, inst) in bb.instructions().iter().enumerate() {
            eprintln!("[DEBUG]     instr[{}]: {:?}", ii, inst);
        }
    }

    // Step 1: Instruction selection — lower IR to machine instructions
    let mut mf = codegen.lower_function(ir_func);

    // DEBUG: dump MachineFunction after instruction selection
    eprintln!("[DEBUG compile_function] after isel: func='{}' blocks={}", 
        mf.name, mf.blocks.len());
    for (bi, mbb) in mf.blocks.iter().enumerate() {
        eprintln!("[DEBUG]   mbb[{}] instructions={}", bi, mbb.instructions.len());
        for (ii, mi) in mbb.instructions.iter().enumerate() {
            eprintln!("[DEBUG]     mi[{}]: opcode={} operands={}", ii, mi.opcode, mi.operands.len());
        }
    }

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
        let sec_config = SecurityConfig::from_flags(config.retpoline, config.cf_protection);
        apply_security_mitigations(&mut mf, &sec_config);
    }

    // Step 5: Assemble to machine code bytes *with* relocations
    let asm_output = codegen.emit_assembly_with_relocations(&mf);
    let code = asm_output.code;
    let asm_relocs = asm_output.relocations;

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
                let name = param.name.clone().unwrap_or_else(|| format!("arg{}", i));
                let type_handle = dwarf.emit_type(&ir_type_to_ctype(&param.ty), &config.target);
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

    // Convert assembler relocations to linker InputRelocations.
    // Each relocation records the symbol name in the addend field encoding
    // and uses a symbol_index that will be resolved later during linking.
    // For now, we store a hash of the symbol name in symbol_index and keep
    // the symbol name available via a side channel (the function's own
    // relocation tracking).
    let func_relocations: Vec<InputRelocation> = asm_relocs
        .iter()
        .map(|r| InputRelocation {
            offset: r.offset as u64,
            reloc_type: r.reloc_type,
            symbol_index: 0, // resolved during linking
            addend: r.addend,
            section_index: 0,
        })
        .collect();

    Ok(AssembledFunction {
        name: ir_func.name.clone(),
        code,
        frame_size: mf.frame_size,
        is_global,
        is_weak,
        visibility,
        section_name,
        alignment: ir_func.alignment.max(1),
        relocations: func_relocations,
        section_offset: text_offset,
        asm_relocations: asm_relocs,
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
            writeln!(
                output,
                "\t.section\t{},\"aw\",@progbits",
                global.section_name
            )?;
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
                let hex_strs: Vec<String> = chunk.iter().map(|b| format!("0x{:02x}", b)).collect();
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
    _diagnostics: &mut DiagnosticEngine,
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
        rodata_symbols.push((
            lit.label.clone(),
            sym_offset,
            lit.data.len() as u64,
            false,
            false,
            STV_DEFAULT,
        ));
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

    let has_rodata = !rodata_data.is_empty();
    if has_rodata {
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

    let has_data = !data_data.is_empty();
    if has_data {
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
    let data_section_idx = if has_rodata { 3 } else { 2 };
    add_data_symbols(&mut elf, &data_symbols, STT_OBJECT, data_section_idx as u16);

    // BSS symbols
    let bss_section_idx = data_section_idx + if has_data { 1 } else { 0 };
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
    if !bss_section.data.is_empty() {
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

    // Collect the set of extern function declaration names for dynamic import
    // detection. If symbol resolution fails, we check whether all unresolved
    // symbols are extern function declarations — if so, they become dynamic
    // imports from libc and we produce a dynamically-linked executable.
    let extern_decl_names: std::collections::HashSet<String> = module
        .declarations
        .iter()
        .map(|d| d.name.clone())
        .collect();

    let (resolved, dynamic_imports) = match sym_resolver.resolve_references() {
        Ok(resolved) => (resolved, Vec::new()),
        Err(errors) => {
            // Separate truly unresolved symbols from extern declarations
            // that should be resolved by the dynamic linker at runtime.
            let mut truly_unresolved = Vec::new();
            let mut dyn_imports: Vec<String> = Vec::new();

            for error in &errors {
                match error {
                    LinkError::UndefinedSymbol {
                        name,
                        referenced_by: _,
                    } => {
                        if extern_decl_names.contains(name) {
                            // This is an extern function declaration — resolve
                            // via dynamic linking (libc.so.6) at runtime.
                            dyn_imports.push(name.clone());
                        } else {
                            truly_unresolved.push(error.clone());
                        }
                    }
                    other => truly_unresolved.push(other.clone()),
                }
            }

            if !truly_unresolved.is_empty() {
                for error in &truly_unresolved {
                    let msg = format!("Link error: {:?}", error);
                    diagnostics.error(Span::DUMMY, msg);
                }
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Linking failed with unresolved symbols",
                ));
            }

            // All unresolved symbols are extern declarations — treat as
            // dynamic imports. Re-run resolution with those symbols removed
            // from the input so it succeeds.
            let mut sym_resolver2 = SymbolResolver::new();
            sym_resolver2.set_target(config.target);
            sym_resolver2.register_object(object_index, &module.name);

            let filtered_symbols: Vec<InputSymbol> = input_symbols
                .into_iter()
                .filter(|s| !dyn_imports.contains(&s.name))
                .collect();
            sym_resolver2.collect_symbols(object_index, &filtered_symbols);

            match sym_resolver2.resolve_references() {
                Ok(resolved) => (resolved, dyn_imports),
                Err(errors2) => {
                    for error in &errors2 {
                        let msg = format!("Link error: {:?}", error);
                        diagnostics.error(Span::DUMMY, msg);
                    }
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Linking failed with unresolved symbols",
                    ));
                }
            }
        }
    };

    // Determine if we need dynamic linking infrastructure.
    let needs_dynamic = config.shared || !dynamic_imports.is_empty();

    if needs_dynamic && !config.shared {
        // ================================================================
        // Dynamic executable path — construct the ELF binary directly for
        // full control over the file layout. This is needed because the
        // ElfWriter computes its own file offsets which would conflict with
        // the program header offsets we need to set precisely.
        // ================================================================
        return write_dynamic_executable(
            functions,
            globals,
            string_literals,
            &dynamic_imports,
            &resolved,
            module,
            config,
            &section_merger,
            &retpoline_thunks,
            dwarf_sections,
        );
    }

    // --- Phase 6: Build the final ELF (static or shared-object path) ---
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
        let section_data = section_merger
            .collect_section_data(section_merger.find_section(&out_section.name).unwrap_or(0));
        let mut elf_section = ElfSection::new(&out_section.name, out_section.section_type);
        elf_section.flags = out_section.flags;
        elf_section.data = section_data;
        elf_section.alignment = out_section.alignment;
        elf_section.addr = out_section.addr;
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
        if let Some(sec_idx) = section_merger.find_section(&func.section_name) {
            sym.section_index = (sec_idx + 1) as u16;
        }
        elf.add_symbol(sym);
    }

    // Add retpoline thunk symbols
    for thunk in &retpoline_thunks {
        let mut sym = ElfSymbol::new(&thunk.name);
        sym.sym_type = STT_FUNC;
        sym.binding = STB_LOCAL;
        sym.size = thunk.code.len() as u64;
        if let Some(val) = resolved.get_symbol_value(&thunk.name) {
            sym.value = val;
        }
        elf.add_symbol(sym);
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
            ".", // compilation directory
            0,   // low_pc — will be the first function's address
            0,   // high_pc — will be the last function's end address
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
        OutputMode::Assembly => {
            write_assembly_output(&assembled_functions, &assembled_globals, module, config)
        }
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

    // External function declarations (undefined symbols).
    // Only emit declarations that are actually referenced (called or address-taken)
    // in the generated code. Declarations merely present from header parsing
    // (e.g., all of stdio.h) but never used should NOT appear as undefined
    // symbols — otherwise the linker fails on every libc prototype.
    //
    // We collect the set of referenced global names by scanning all function
    // bodies for values whose name matches the `global.<name>` pattern
    // produced by `IrBuilder::build_global_ref`.
    let mut referenced_globals: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for func in &module.functions {
        for vi in &func.local_values {
            if let Some(ref name) = vi.name {
                if let Some(stripped) = name.strip_prefix("global.") {
                    referenced_globals.insert(stripped.to_string());
                }
            }
        }
    }

    for decl in &module.declarations {
        // Skip declarations that are never actually referenced in the code.
        if !referenced_globals.contains(&decl.name) {
            continue;
        }
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

// ===========================================================================
// Dynamic Executable Builder
// ===========================================================================

/// Produces a dynamically-linked ELF executable (ET_EXEC) with .interp,
/// .dynsym, .dynstr, .gnu.hash, .rela.plt, .plt, .got.plt, and .dynamic
/// sections. This function constructs the ELF binary directly as a byte
/// vector for complete control over file layout, which is required because
/// program header file offsets must exactly match section positions.
///
/// The layout model uses `vaddr = base_address + file_offset` so that
/// a single PT_LOAD mapping covers the entire read-execute segment, and
/// a second PT_LOAD covers the read-write segment (.got.plt, .dynamic, .data).
///
/// A synthetic `_start` stub is prepended to .text that calls `main` and
/// then invokes the `exit` syscall with main's return value, ensuring
/// clean process termination without requiring crt1.o.
#[allow(clippy::too_many_arguments)]
fn write_dynamic_executable(
    functions: &[AssembledFunction],
    globals: &[AssembledGlobal],
    string_literals: &[AssembledStringLiteral],
    dynamic_imports: &[String],
    _resolved: &crate::backend::linker_common::ResolvedSymbols,
    _module: &crate::ir::module::IrModule,
    config: &CodegenConfig,
    _section_merger: &SectionMerger,
    retpoline_thunks: &[RetpolineThunkData],
    _dwarf_sections: &Option<DwarfSections>,
) -> io::Result<()> {
    use crate::backend::linker_common::dynamic::{
        DynamicLayout, DynamicRelocation, DynamicSectionBuilder, DynamicSymbolTable,
        GotBuilder, GotEntry, PltBuilder, PltEntry, build_rela_plt, interp_string,
    };

    let target = &config.target;
    let is_64bit = match target {
        Target::X86_64 | Target::AArch64 | Target::RiscV64 => true,
        Target::I686 => false,
    };
    let ptr_size: u64 = if is_64bit { 8 } else { 4 };
    let base_address: u64 = if is_64bit { 0x400000 } else { 0x08048000 };
    let ehdr_size: usize = if is_64bit { 64 } else { 52 };
    let phdr_entry_size: usize = if is_64bit { 56 } else { 32 };

    // We need 5 program headers:
    // PT_PHDR, PT_INTERP, PT_LOAD (rx), PT_LOAD (rw), PT_DYNAMIC
    let num_phdrs: usize = 5;
    let phdr_table_size = num_phdrs * phdr_entry_size;
    let phdrs_end = ehdr_size + phdr_table_size;

    // ---- Collect section data from the compilation ----
    // Gather .rodata from string literals and const globals.
    // Build a name-to-offset map so that relocations referencing
    // string literal symbols can be correctly patched.
    let mut rodata_data = Vec::new();
    let mut rodata_sym_offsets: Vec<(String, u64)> = Vec::new();
    for lit in string_literals {
        let off = rodata_data.len() as u64;
        // Store multiple name variants so relocation matching is robust.
        // The assembler may emit ".str.0" while the literal is labelled ".L.str.0".
        rodata_sym_offsets.push((lit.label.clone(), off));
        // Also store a stripped variant without leading ".L" so that both
        // ".L.str.0" and ".str.0" resolve to the same offset.
        if let Some(stripped) = lit.label.strip_prefix(".L") {
            rodata_sym_offsets.push((stripped.to_string(), off));
        }
        rodata_data.extend_from_slice(&lit.data);
    }
    for global in globals.iter().filter(|g| g.is_const && !g.is_bss) {
        let align = global.alignment as u64;
        let pad = dyn_align_pad(rodata_data.len() as u64, align);
        rodata_data.extend(std::iter::repeat(0u8).take(pad));
        rodata_data.extend_from_slice(&global.data);
    }

    // Gather .data from initialized mutable globals
    let mut data_data = Vec::new();
    for global in globals.iter().filter(|g| !g.is_const && !g.is_bss) {
        let align = global.alignment as u64;
        let pad = dyn_align_pad(data_data.len() as u64, align);
        data_data.extend(std::iter::repeat(0u8).take(pad));
        data_data.extend_from_slice(&global.data);
    }

    // .bss size from zero-initialized globals
    let mut bss_size: u64 = 0;
    for global in globals.iter().filter(|g| g.is_bss) {
        let align = global.alignment as u64;
        bss_size = dyn_align_up(bss_size, align);
        bss_size += global.data.len() as u64;
    }

    // ---- Build .interp section ----
    let interp_str = interp_string(target);
    let mut interp_data = interp_str.as_bytes().to_vec();
    interp_data.push(0); // NUL terminator

    // ---- Build .dynsym and .dynstr ----
    // Ensure `exit` is always in the dynamic import set so that the _start
    // stub can call libc's exit() to properly flush stdio buffers.
    let mut all_dynamic_imports: Vec<String> = dynamic_imports.to_vec();
    if !all_dynamic_imports.iter().any(|s| s == "exit") {
        all_dynamic_imports.push("exit".to_string());
    }
    let dynamic_imports = &all_dynamic_imports;

    let mut dynsym_table = DynamicSymbolTable::new();
    if !is_64bit {
        dynsym_table.set_32bit(true);
    }
    for imp in dynamic_imports {
        // Add each dynamic import as an undefined global function symbol
        let sym_entry = crate::backend::linker_common::SymbolEntry {
            name: imp.clone(),
            value: 0,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: LinkerSymbolType::Func,
            visibility: LinkerSymbolVisibility::Default,
            section_index: 0, // SHN_UNDEF
            is_defined: false,
            defining_object: 0,
        };
        dynsym_table.add_symbol(&sym_entry);
    }
    // Intern the needed library name into the same .dynstr table so that
    // the DT_NEEDED entry offset matches the actual .dynstr contents.
    let libc_needed_offset = dynsym_table.intern_needed_string("libc.so.6");
    let dynsym_data = dynsym_table.build_dynsym();
    let dynstr_data = dynsym_table.build_dynstr();
    let gnu_hash_data = dynsym_table.build_gnu_hash();

    // ---- Pre-compute sizes for layout ----
    let rela_entry_size: usize = if is_64bit { 24 } else { 12 };
    let rela_plt_size = dynamic_imports.len() * rela_entry_size;

    // PLT sizes: PLT[0] (resolver) + N per-function stubs
    let plt0_sz: usize = match target {
        Target::X86_64 | Target::I686 => 16,
        Target::AArch64 | Target::RiscV64 => 32,
    };
    let plt_entry_sz: usize = 16;
    let plt_total_size = plt0_sz + dynamic_imports.len() * plt_entry_sz;

    // GOT.PLT: 3 reserved slots + N function slots
    let got_plt_size = (3 + dynamic_imports.len()) as u64 * ptr_size;

    // ---- Compute file layout ----
    // Everything in the RX segment is laid out contiguously after program
    // headers. Virtual address = base_address + file_offset.
    let mut offset = phdrs_end;

    // .interp
    let interp_off = offset;
    offset += interp_data.len();

    // .gnu.hash (align to 8)
    offset = dyn_align_up_usize(offset, 8);
    let gnu_hash_off = offset;
    offset += gnu_hash_data.len();

    // .dynsym (align to 8)
    offset = dyn_align_up_usize(offset, 8);
    let dynsym_off = offset;
    offset += dynsym_data.len();

    // .dynstr
    let dynstr_off = offset;
    offset += dynstr_data.len();

    // .rela.plt (align to 8)
    offset = dyn_align_up_usize(offset, 8);
    let rela_plt_off = offset;
    offset += rela_plt_size;

    // .plt (align to 16)
    offset = dyn_align_up_usize(offset, 16);
    let plt_off = offset;
    offset += plt_total_size;

    // .text (align to 16) — we'll prepend a _start stub
    offset = dyn_align_up_usize(offset, 16);
    let text_off = offset;

    // Build _start stub + collect function code
    // The _start stub calls main and then invokes exit syscall.
    // We don't know main's offset yet, so we'll patch it after.
    let start_stub = build_start_stub(target);
    let _start_stub_len = start_stub.len();

    // Collect user function code with alignment, tracking each function's
    // offset within the combined .text
    let mut text_data = start_stub;
    let mut func_offsets: Vec<(usize, usize)> = Vec::new(); // (offset_in_text, code_len)
    for func in functions {
        let align = func.alignment as u64;
        let pad = dyn_align_pad(text_data.len() as u64, align);
        text_data.extend(std::iter::repeat(0x90u8).take(pad));
        let fn_off = text_data.len();
        text_data.extend_from_slice(&func.code);
        func_offsets.push((fn_off, func.code.len()));
    }

    // Retpoline thunks (if any)
    if !retpoline_thunks.is_empty() {
        let pad = dyn_align_pad(text_data.len() as u64, 16);
        text_data.extend(std::iter::repeat(0x90u8).take(pad));
        for thunk in retpoline_thunks {
            text_data.extend_from_slice(&thunk.code);
        }
    }

    let text_size = text_data.len();
    offset += text_size;

    // .rodata (align to 8)
    offset = dyn_align_up_usize(offset, 8);
    let rodata_off = offset;
    offset += rodata_data.len();

    // End of RX segment — pad to page boundary for RW segment
    let page_size: usize = 0x1000;
    offset = dyn_align_up_usize(offset, page_size);

    // RW segment starts here
    let _rw_segment_off = offset;

    // .got.plt (align to 8)
    let got_plt_off = offset;
    offset += got_plt_size as usize;

    // .dynamic (align to 8)
    offset = dyn_align_up_usize(offset, 8);
    let dynamic_off = offset;
    // We'll compute .dynamic size after building it, but we need the address
    // now for GOT.PLT[0]. Use a placeholder and adjust.

    // .data
    // We need to know .dynamic size first. Estimate conservatively:
    // ~20 DT entries × 16 bytes = 320 bytes (64-bit), then we'll finalize.
    let dynamic_estimated_size = 20 * (if is_64bit { 16 } else { 8 });
    offset += dynamic_estimated_size;
    offset = dyn_align_up_usize(offset, 8);
    let data_off = offset;
    offset += data_data.len();

    // .bss is NOBITS — no file space, but extends memsz
    let bss_off = offset; // conceptual offset
    let _ = bss_off;

    // Total file size (before section/string tables)
    let _loadable_end = offset;

    // ---- Compute virtual addresses ----
    let interp_vaddr = base_address + interp_off as u64;
    let gnu_hash_vaddr = base_address + gnu_hash_off as u64;
    let dynsym_vaddr = base_address + dynsym_off as u64;
    let dynstr_vaddr = base_address + dynstr_off as u64;
    let rela_plt_vaddr = base_address + rela_plt_off as u64;
    let plt_vaddr = base_address + plt_off as u64;
    let text_vaddr = base_address + text_off as u64;
    let _rodata_vaddr = base_address + rodata_off as u64;
    let got_plt_vaddr = base_address + got_plt_off as u64;
    let dynamic_vaddr = base_address + dynamic_off as u64;
    let _data_vaddr = base_address + data_off as u64;

    // ---- Build PLT and GOT.PLT with known addresses ----
    let mut plt_builder = PltBuilder::new(plt_vaddr, got_plt_vaddr, *target);
    let mut plt_addr_map: Vec<(String, u64)> = Vec::new();
    for (i, imp) in dynamic_imports.iter().enumerate() {
        let got_entry_offset = (3 + i) as u64 * ptr_size;
        let entry = PltEntry {
            symbol_name: imp.clone(),
            got_offset: got_entry_offset,
            plt_index: i as u32,
        };
        let addr = plt_builder.add_entry(entry);
        plt_addr_map.push((imp.clone(), addr));
    }
    let plt_data = plt_builder.build_plt(target);

    // GOT.PLT: slot[0]=.dynamic addr, slot[1]=0 (link_map), slot[2]=0 (resolver),
    // slot[3+]=initial value pointing to PLT push instruction (lazy binding).
    let mut got_builder = GotBuilder::new(0, got_plt_vaddr, dynamic_vaddr, target);
    for (i, imp) in dynamic_imports.iter().enumerate() {
        // Initial value = address of PLT[N]'s push instruction (offset +6 in the stub)
        let plt_push_addr = plt_vaddr
            + plt0_sz as u64
            + (i as u64) * plt_entry_sz as u64
            + 6;
        got_builder.add_plt_entry(GotEntry {
            symbol_name: imp.clone(),
            offset: (3 + i) as u64 * ptr_size,
            initial_value: plt_push_addr,
        });
    }
    let got_plt_data = got_builder.build_got_plt();

    // ---- Build .rela.plt ----
    let mut rela_plt_relocs: Vec<DynamicRelocation> = Vec::new();
    for (i, _imp) in dynamic_imports.iter().enumerate() {
        let sym_index = (i + 1) as u32; // dynsym index (0 is null)
        let got_entry_vaddr = got_plt_vaddr + (3 + i as u64) * ptr_size;
        let reloc_type = match target {
            Target::X86_64 => 7,     // R_X86_64_JUMP_SLOT
            Target::I686 => 7,       // R_386_JMP_SLOT
            Target::AArch64 => 1026, // R_AARCH64_JUMP_SLOT
            Target::RiscV64 => 5,    // R_RISCV_JUMP_SLOT
        };
        rela_plt_relocs.push(DynamicRelocation {
            offset: got_entry_vaddr,
            reloc_type,
            symbol_index: sym_index,
            addend: 0,
        });
    }
    let rela_plt_data = build_rela_plt(&rela_plt_relocs);

    // ---- Build .dynamic section ----
    let layout = DynamicLayout {
        dynamic_addr: dynamic_vaddr,
        dynsym_addr: dynsym_vaddr,
        dynstr_addr: dynstr_vaddr,
        gnu_hash_addr: gnu_hash_vaddr,
        got_addr: 0,
        got_plt_addr: got_plt_vaddr,
        plt_addr: plt_vaddr,
        rela_dyn_addr: 0,
        rela_plt_addr: rela_plt_vaddr,
        interp_addr: interp_vaddr,
    };

    let mut dyn_builder = DynamicSectionBuilder::new();
    dyn_builder.add_needed_precomputed(libc_needed_offset as u64);
    dyn_builder.set_dynstr_size(dynstr_data.len() as u64);
    dyn_builder.set_rela_plt_size(rela_plt_data.len() as u64);
    if !is_64bit {
        dyn_builder.set_32bit(true);
    }
    let dynamic_data = dyn_builder.build(&layout);
    let dynamic_actual_size = dynamic_data.len();

    // Recompute .data offset now that we know the exact .dynamic size
    let data_off_final = dyn_align_up_usize(dynamic_off + dynamic_actual_size, 8);
    let data_vaddr_final = base_address + data_off_final as u64;
    let _ = data_vaddr_final;
    let loadable_end_final = data_off_final + data_data.len();

    // DEBUG: dump assembled functions info
    for (_i, func) in functions.iter().enumerate() {
        eprintln!("[DYNELF DEBUG] Function '{}': {} bytes of code, {} relocations",
            func.name, func.code.len(), func.asm_relocations.len());
        for reloc in &func.asm_relocations {
            eprintln!("[DYNELF DEBUG]   reloc: symbol='{}' offset={} addend={}",
                reloc.symbol, reloc.offset, reloc.addend);
        }
        // Dump first 32 bytes of code as hex
        let display_len = func.code.len().min(64);
        let hex: Vec<String> = func.code[..display_len].iter().map(|b| format!("{:02x}", b)).collect();
        eprintln!("[DYNELF DEBUG]   code: {}", hex.join(" "));
    }
    eprintln!("[DYNELF DEBUG] dynamic_imports: {:?}", dynamic_imports);
    eprintln!("[DYNELF DEBUG] string_literals: {} entries, {} total bytes",
        string_literals.len(), string_literals.iter().map(|s| s.data.len()).sum::<usize>());
    for lit in string_literals {
        eprintln!("[DYNELF DEBUG]   string_lit: name='{}' len={} data={:?}",
            lit.label, lit.data.len(), String::from_utf8_lossy(&lit.data));
    }

    // ---- Patch .text: _start stub's call to main ----
    // Find main's offset within our .text
    let mut main_offset_in_text: Option<usize> = None;
    for (i, func) in functions.iter().enumerate() {
        if func.name == "main" {
            main_offset_in_text = Some(func_offsets[i].0);
            break;
        }
    }
    if let Some(main_off) = main_offset_in_text {
        patch_start_stub_call(&mut text_data, main_off, target);
    }

    // ---- Patch .text: _start stub's call to exit@plt ----
    // The exit call displacement is at byte 17 in the x86-64 _start stub.
    if let Some((_, exit_plt_addr)) =
        plt_addr_map.iter().find(|(name, _)| name == "exit")
    {
        let exit_call_disp_off: usize = match target {
            Target::X86_64 => 17, // see build_start_stub: e8 at offset 16, disp at 17
            Target::I686 => 12,   // similar layout
            _ => 0,               // AArch64/RiscV use different instruction format
        };
        if exit_call_disp_off > 0 && exit_call_disp_off + 4 <= text_data.len() {
            let p = text_vaddr + exit_call_disp_off as u64 + 4; // RIP after call
            let disp = (*exit_plt_addr as i64) - (p as i64);
            let bytes = (disp as i32).to_le_bytes();
            text_data[exit_call_disp_off..exit_call_disp_off + 4]
                .copy_from_slice(&bytes);
        }
    }

    // ---- Patch .text: external calls to use PLT ----
    for (i, func) in functions.iter().enumerate() {
        let fn_off_in_text = func_offsets[i].0;
        for reloc in &func.asm_relocations {
            if let Some((_, plt_entry_addr)) =
                plt_addr_map.iter().find(|(name, _)| name == &reloc.symbol)
            {
                // This relocation references a dynamic import — patch to PLT
                let text_byte_offset = fn_off_in_text + reloc.offset;
                if text_byte_offset + 4 <= text_data.len() {
                    // R_X86_64_PLT32 / R_X86_64_PC32: S + A - P
                    let p = text_vaddr + text_byte_offset as u64;
                    let s = *plt_entry_addr;
                    let value = (s as i64) + reloc.addend - (p as i64);
                    let bytes = (value as i32).to_le_bytes();
                    text_data[text_byte_offset..text_byte_offset + 4]
                        .copy_from_slice(&bytes);
                }
            }
        }
    }

    // ---- Patch .text: internal call relocations (non-dynamic) ----
    // Also patch any local function calls that the assembler left as
    // relocations (e.g., call from one function to another).
    for (i, func) in functions.iter().enumerate() {
        let fn_off_in_text = func_offsets[i].0;
        for reloc in &func.asm_relocations {
            // Skip dynamic imports (already patched above)
            if plt_addr_map.iter().any(|(name, _)| name == &reloc.symbol) {
                continue;
            }
            // Find the target function in our text section
            let mut target_off_in_text: Option<usize> = None;
            for (j, other_func) in functions.iter().enumerate() {
                if other_func.name == reloc.symbol {
                    target_off_in_text = Some(func_offsets[j].0);
                    break;
                }
            }
            if let Some(target_off) = target_off_in_text {
                let text_byte_offset = fn_off_in_text + reloc.offset;
                if text_byte_offset + 4 <= text_data.len() {
                    let p = text_vaddr + text_byte_offset as u64;
                    let s = text_vaddr + target_off as u64;
                    let value = (s as i64) + reloc.addend - (p as i64);
                    let bytes = (value as i32).to_le_bytes();
                    text_data[text_byte_offset..text_byte_offset + 4]
                        .copy_from_slice(&bytes);
                }
            }
        }
    }

    // ---- Also patch relocations to .rodata (string literals) ----
    // Use the `rodata_sym_offsets` map built earlier so that each symbol
    // resolves to the correct byte offset within the .rodata section.
    let rodata_vaddr = base_address + rodata_off as u64;
    for (i, func) in functions.iter().enumerate() {
        let fn_off_in_text = func_offsets[i].0;
        for reloc in &func.asm_relocations {
            // Skip dynamic imports — already handled above.
            if plt_addr_map.iter().any(|(name, _)| name == &reloc.symbol) {
                continue;
            }
            // Skip local function-to-function calls — already handled above.
            if functions.iter().any(|f| f.name == reloc.symbol) {
                continue;
            }
            // Look up the symbol in our rodata name map.
            let rodata_match = rodata_sym_offsets
                .iter()
                .find(|(name, _)| name == &reloc.symbol);
            if let Some((_, sym_off_in_rodata)) = rodata_match {
                let text_byte_offset = fn_off_in_text + reloc.offset;
                if text_byte_offset + 4 <= text_data.len() {
                    let p = text_vaddr + text_byte_offset as u64;
                    let s = rodata_vaddr + sym_off_in_rodata;
                    let value = (s as i64) + reloc.addend - (p as i64);
                    let bytes = (value as i32).to_le_bytes();
                    text_data[text_byte_offset..text_byte_offset + 4]
                        .copy_from_slice(&bytes);
                }
            }
        }
    }

    // ---- Build the raw ELF binary ----
    let entry_vaddr = text_vaddr; // _start is at the beginning of .text
    let rx_segment_end = dyn_align_up_usize(rodata_off + rodata_data.len(), page_size);
    let rw_segment_start = rx_segment_end;
    let rw_segment_filesz = loadable_end_final - rw_segment_start;
    let rw_segment_memsz = rw_segment_filesz + bss_size as usize;

    // Construct the output buffer
    let mut out = Vec::with_capacity(loadable_end_final + 4096);

    // ---- ELF Header ----
    write_elf_header_raw(
        &mut out,
        is_64bit,
        ET_EXEC,
        target.elf_machine(),
        entry_vaddr,
        ehdr_size as u64,       // e_phoff
        0u64,                   // e_shoff (patched later)
        phdr_entry_size as u16,
        num_phdrs as u16,
        0u16,                   // e_shentsize (patched later)
        0u16,                   // e_shnum (patched later)
        0u16,                   // e_shstrndx (patched later)
    );

    // ---- Program Headers ----
    // PT_PHDR
    write_phdr_raw(&mut out, is_64bit, PT_PHDR, PF_R,
        ehdr_size as u64,
        base_address + ehdr_size as u64,
        phdr_table_size as u64,
        phdr_table_size as u64,
        8,
    );

    // PT_INTERP
    write_phdr_raw(&mut out, is_64bit, PT_INTERP, PF_R,
        interp_off as u64,
        interp_vaddr,
        interp_data.len() as u64,
        interp_data.len() as u64,
        1,
    );

    // PT_LOAD (RX segment): from file start to end of .rodata
    write_phdr_raw(&mut out, is_64bit, PT_LOAD, PF_R | 0x1, // PF_R | PF_X
        0,
        base_address,
        rx_segment_end as u64,
        rx_segment_end as u64,
        page_size as u64,
    );

    // PT_LOAD (RW segment): .got.plt, .dynamic, .data, .bss
    write_phdr_raw(&mut out, is_64bit, PT_LOAD, PF_R | PF_W,
        rw_segment_start as u64,
        base_address + rw_segment_start as u64,
        rw_segment_filesz as u64,
        rw_segment_memsz as u64,
        page_size as u64,
    );

    // PT_DYNAMIC
    write_phdr_raw(&mut out, is_64bit, PT_DYNAMIC, PF_R | PF_W,
        dynamic_off as u64,
        dynamic_vaddr,
        dynamic_actual_size as u64,
        dynamic_actual_size as u64,
        8,
    );

    // ---- Section Data ----
    // Pad and write each section at its computed file offset

    // .interp
    pad_to_offset(&mut out, interp_off);
    out.extend_from_slice(&interp_data);

    // .gnu.hash
    pad_to_offset(&mut out, gnu_hash_off);
    out.extend_from_slice(&gnu_hash_data);

    // .dynsym
    pad_to_offset(&mut out, dynsym_off);
    out.extend_from_slice(&dynsym_data);

    // .dynstr
    pad_to_offset(&mut out, dynstr_off);
    out.extend_from_slice(&dynstr_data);

    // .rela.plt
    pad_to_offset(&mut out, rela_plt_off);
    out.extend_from_slice(&rela_plt_data);

    // .plt
    pad_to_offset(&mut out, plt_off);
    out.extend_from_slice(&plt_data);

    // .text
    pad_to_offset(&mut out, text_off);
    out.extend_from_slice(&text_data);

    // .rodata
    pad_to_offset(&mut out, rodata_off);
    out.extend_from_slice(&rodata_data);

    // RW segment sections
    // .got.plt
    pad_to_offset(&mut out, got_plt_off);
    out.extend_from_slice(&got_plt_data);

    // .dynamic
    pad_to_offset(&mut out, dynamic_off);
    out.extend_from_slice(&dynamic_data);

    // .data
    pad_to_offset(&mut out, data_off_final);
    out.extend_from_slice(&data_data);

    // ---- Section Header Table ----
    // Build a minimal section header table for debugging tools.
    // Sections: NULL, .interp, .gnu.hash, .dynsym, .dynstr, .rela.plt,
    //           .plt, .text, .rodata, .got.plt, .dynamic, .data, .bss,
    //           .shstrtab
    let shdr_entry_size: usize = if is_64bit { 64 } else { 40 };

    // Build .shstrtab
    let section_names = [
        "", ".interp", ".gnu.hash", ".dynsym", ".dynstr", ".rela.plt",
        ".plt", ".text", ".rodata", ".got.plt", ".dynamic", ".data",
        ".bss", ".shstrtab",
    ];
    let mut shstrtab = Vec::new();
    let mut name_offsets = Vec::new();
    for name in &section_names {
        let off = shstrtab.len();
        name_offsets.push(off as u32);
        shstrtab.extend_from_slice(name.as_bytes());
        shstrtab.push(0);
    }

    let num_sections = section_names.len(); // 14 (including NULL)
    let shstrtab_idx = num_sections - 1;    // last section

    // Align to 8 for section header table
    let shstrtab_off = dyn_align_up_usize(out.len(), 8);
    pad_to_offset(&mut out, shstrtab_off);
    out.extend_from_slice(&shstrtab);

    let shdr_off = dyn_align_up_usize(out.len(), 8);
    pad_to_offset(&mut out, shdr_off);

    // Write section headers
    // 0: NULL
    write_shdr_raw(&mut out, is_64bit, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // 1: .interp (SHT_PROGBITS, SHF_ALLOC)
    write_shdr_raw(&mut out, is_64bit, name_offsets[1], SHT_PROGBITS,
        SHF_ALLOC as u64, interp_vaddr, interp_off as u64,
        interp_data.len() as u64, 0, 0, 1, 0);
    // 2: .gnu.hash (SHT_GNU_HASH=0x6ffffff6, SHF_ALLOC)
    write_shdr_raw(&mut out, is_64bit, name_offsets[2], 0x6ffffff6u32,
        SHF_ALLOC as u64, gnu_hash_vaddr, gnu_hash_off as u64,
        gnu_hash_data.len() as u64, 3, 0, 8, 0); // sh_link=.dynsym(3)
    // 3: .dynsym (SHT_DYNSYM, SHF_ALLOC)
    let dynsym_entsize = if is_64bit { 24u64 } else { 16u64 };
    write_shdr_raw(&mut out, is_64bit, name_offsets[3], SHT_DYNSYM,
        SHF_ALLOC as u64, dynsym_vaddr, dynsym_off as u64,
        dynsym_data.len() as u64, 4, 1, 8, dynsym_entsize); // sh_link=.dynstr(4), sh_info=1 (first non-local)
    // 4: .dynstr (SHT_STRTAB, SHF_ALLOC)
    write_shdr_raw(&mut out, is_64bit, name_offsets[4], SHT_STRTAB,
        SHF_ALLOC as u64, dynstr_vaddr, dynstr_off as u64,
        dynstr_data.len() as u64, 0, 0, 1, 0);
    // 5: .rela.plt (SHT_RELA=4, SHF_ALLOC|SHF_INFO_LINK)
    let rela_entsize = if is_64bit { 24u64 } else { 12u64 };
    write_shdr_raw(&mut out, is_64bit, name_offsets[5], 4u32, // SHT_RELA
        (SHF_ALLOC | 0x40) as u64, // SHF_ALLOC | SHF_INFO_LINK
        rela_plt_vaddr, rela_plt_off as u64,
        rela_plt_data.len() as u64, 3, 9, 8, rela_entsize); // sh_link=.dynsym(3), sh_info=.got.plt(9)
    // 6: .plt (SHT_PROGBITS, SHF_ALLOC|SHF_EXECINSTR)
    write_shdr_raw(&mut out, is_64bit, name_offsets[6], SHT_PROGBITS,
        (SHF_ALLOC | SHF_EXECINSTR) as u64, plt_vaddr, plt_off as u64,
        plt_data.len() as u64, 0, 0, 16, plt_entry_sz as u64);
    // 7: .text (SHT_PROGBITS, SHF_ALLOC|SHF_EXECINSTR)
    write_shdr_raw(&mut out, is_64bit, name_offsets[7], SHT_PROGBITS,
        (SHF_ALLOC | SHF_EXECINSTR) as u64, text_vaddr, text_off as u64,
        text_size as u64, 0, 0, 16, 0);
    // 8: .rodata (SHT_PROGBITS, SHF_ALLOC)
    write_shdr_raw(&mut out, is_64bit, name_offsets[8], SHT_PROGBITS,
        SHF_ALLOC as u64, rodata_vaddr, rodata_off as u64,
        rodata_data.len() as u64, 0, 0, 8, 0);
    // 9: .got.plt (SHT_PROGBITS, SHF_ALLOC|SHF_WRITE)
    write_shdr_raw(&mut out, is_64bit, name_offsets[9], SHT_PROGBITS,
        (SHF_ALLOC | SHF_WRITE) as u64, got_plt_vaddr, got_plt_off as u64,
        got_plt_data.len() as u64, 0, 0, 8, ptr_size);
    // 10: .dynamic (SHT_DYNAMIC, SHF_ALLOC|SHF_WRITE)
    let dyn_entsize = if is_64bit { 16u64 } else { 8u64 };
    write_shdr_raw(&mut out, is_64bit, name_offsets[10], SHT_DYNAMIC,
        (SHF_ALLOC | SHF_WRITE) as u64, dynamic_vaddr, dynamic_off as u64,
        dynamic_data.len() as u64, 4, 0, 8, dyn_entsize); // sh_link=.dynstr(4)
    // 11: .data (SHT_PROGBITS, SHF_ALLOC|SHF_WRITE)
    write_shdr_raw(&mut out, is_64bit, name_offsets[11], SHT_PROGBITS,
        (SHF_ALLOC | SHF_WRITE) as u64, data_vaddr_final, data_off_final as u64,
        data_data.len() as u64, 0, 0, 8, 0);
    // 12: .bss (SHT_NOBITS, SHF_ALLOC|SHF_WRITE)
    let bss_vaddr = data_vaddr_final + data_data.len() as u64;
    write_shdr_raw(&mut out, is_64bit, name_offsets[12], SHT_NOBITS,
        (SHF_ALLOC | SHF_WRITE) as u64, bss_vaddr, loadable_end_final as u64,
        bss_size, 0, 0, 8, 0);
    // 13: .shstrtab (SHT_STRTAB)
    write_shdr_raw(&mut out, is_64bit, name_offsets[13], SHT_STRTAB,
        0, 0, shstrtab_off as u64,
        shstrtab.len() as u64, 0, 0, 1, 0);

    // ---- Patch ELF header with section header info ----
    // e_shoff
    if is_64bit {
        out[40..48].copy_from_slice(&(shdr_off as u64).to_le_bytes());
        out[58..60].copy_from_slice(&(shdr_entry_size as u16).to_le_bytes()); // e_shentsize
        out[60..62].copy_from_slice(&(num_sections as u16).to_le_bytes());    // e_shnum
        out[62..64].copy_from_slice(&(shstrtab_idx as u16).to_le_bytes());    // e_shstrndx
    } else {
        out[32..36].copy_from_slice(&(shdr_off as u32).to_le_bytes());
        out[46..48].copy_from_slice(&(shdr_entry_size as u16).to_le_bytes());
        out[48..50].copy_from_slice(&(num_sections as u16).to_le_bytes());
        out[50..52].copy_from_slice(&(shstrtab_idx as u16).to_le_bytes());
    }

    // ---- Write to file ----
    std::fs::write(&config.output_path, &out)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&config.output_path, perms)?;
    }

    Ok(())
}

/// Builds a minimal `_start` assembly stub for the given target that:
/// 1. Sets up the stack frame per ABI
/// 2. Calls `main` (PC-relative, patched later)
/// 3. Passes `main`'s return value as the argument to libc `exit()`
/// 4. Calls `exit@plt` to properly flush stdio buffers before terminating
///
/// Returns the stub bytes. Both call displacements are left as 0x00000000
/// and must be patched after function/PLT layout is finalized.
///
/// For x86-64, the layout is:
///   offset  0: xor %ebp, %ebp          (2 bytes)
///   offset  2: mov %rsp, %rdi          (3 bytes)
///   offset  5: and $-16, %rsp          (4 bytes)
///   offset  9: call main               (5 bytes, disp at 10)
///   offset 14: mov %eax, %edi          (2 bytes)
///   offset 16: call exit               (5 bytes, disp at 17)
///   offset 21: hlt                     (1 byte, unreachable guard)
///   Total: 22 bytes, padded to 32 for alignment.
fn build_start_stub(target: &Target) -> Vec<u8> {
    match target {
        Target::X86_64 => {
            let mut stub = vec![
                0x31, 0xed,             // xor %ebp, %ebp
                0x48, 0x89, 0xe7,       // mov %rsp, %rdi
                0x48, 0x83, 0xe4, 0xf0, // and $-16, %rsp
                0xe8, 0x00, 0x00, 0x00, 0x00, // call main (placeholder, disp at byte 10)
                0x89, 0xc7,             // mov %eax, %edi
                0xe8, 0x00, 0x00, 0x00, 0x00, // call exit (placeholder, disp at byte 17)
                0xf4,                   // hlt (unreachable)
            ];
            // Pad to 32 bytes (16-byte aligned) for function alignment.
            while stub.len() < 32 {
                stub.push(0x90); // nop
            }
            stub
        }
        Target::I686 => {
            // _start:
            //   xor  %ebp, %ebp          ; 31 ed
            //   and  $-16, %esp           ; 83 e4 f0
            //   call main                 ; e8 XX XX XX XX (placeholder)
            //   mov  %eax, %ebx           ; 89 c3
            //   mov  $1, %eax             ; b8 01 00 00 00 (__NR_exit)
            //   int  $0x80                ; cd 80
            vec![
                0x31, 0xed,             // xor %ebp, %ebp
                0x83, 0xe4, 0xf0,       // and $-16, %esp
                0xe8, 0x00, 0x00, 0x00, 0x00, // call main (placeholder)
                0x89, 0xc3,             // mov %eax, %ebx
                0xb8, 0x01, 0x00, 0x00, 0x00, // mov $1, %eax
                0xcd, 0x80,             // int $0x80
            ]
        }
        Target::AArch64 => {
            // _start:
            //   bl main                  ; 94 00 00 00 (placeholder)
            //   mov x8, #93              ; exit_group syscall
            //   svc #0
            let mut stub = Vec::new();
            stub.extend_from_slice(&0x94000000u32.to_le_bytes()); // bl main (placeholder)
            stub.extend_from_slice(&0xd2800ba8u32.to_le_bytes()); // mov x8, #93
            stub.extend_from_slice(&0xd4000001u32.to_le_bytes()); // svc #0
            stub
        }
        Target::RiscV64 => {
            // _start:
            //   jal ra, main             ; placeholder
            //   li a7, 93                ; exit_group syscall number
            //   ecall
            let mut stub = Vec::new();
            stub.extend_from_slice(&0x000000efu32.to_le_bytes()); // jal ra, 0 (placeholder)
            stub.extend_from_slice(&0x05d00893u32.to_le_bytes()); // li a7, 93
            stub.extend_from_slice(&0x00000073u32.to_le_bytes()); // ecall
            stub
        }
    }
}

/// Patches the `_start` stub's call instruction to target `main` at the
/// given offset within the .text section.
fn patch_start_stub_call(text_data: &mut [u8], main_offset: usize, target: &Target) {
    match target {
        Target::X86_64 => {
            // call instruction is at offset 9 in the stub (0xe8 byte)
            // displacement field is at offset 10, 4 bytes
            // call displacement = target - (call_instruction_end)
            //                   = main_offset - (10 + 4)
            //                   = main_offset - 14
            let call_end = 14usize; // offset 10 + 4 bytes of displacement
            let disp = (main_offset as i64) - (call_end as i64);
            text_data[10..14].copy_from_slice(&(disp as i32).to_le_bytes());
        }
        Target::I686 => {
            // call instruction is at offset 5 in the stub (0xe8 byte)
            // displacement field is at offset 6, 4 bytes
            let call_end = 10usize; // 6 + 4
            let disp = (main_offset as i64) - (call_end as i64);
            text_data[6..10].copy_from_slice(&(disp as i32).to_le_bytes());
        }
        Target::AArch64 => {
            // bl instruction is at offset 0, 4 bytes
            // imm26 field encodes (target - pc) / 4
            let disp = (main_offset as i64) / 4;
            let insn = 0x94000000u32 | ((disp as u32) & 0x03FF_FFFF);
            text_data[0..4].copy_from_slice(&insn.to_le_bytes());
        }
        Target::RiscV64 => {
            // jal ra instruction is at offset 0, 4 bytes
            // J-type encoding
            let disp = main_offset as i32;
            let imm20 = ((disp >> 20) & 1) as u32;
            let imm10_1 = ((disp >> 1) & 0x3FF) as u32;
            let imm11 = ((disp >> 11) & 1) as u32;
            let imm19_12 = ((disp >> 12) & 0xFF) as u32;
            let insn = (imm20 << 31)
                | (imm10_1 << 21)
                | (imm11 << 20)
                | (imm19_12 << 12)
                | (1 << 7)   // rd = ra (x1)
                | 0x6f;      // JAL opcode
            text_data[0..4].copy_from_slice(&insn.to_le_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// Raw ELF binary construction helpers
// ---------------------------------------------------------------------------

/// Pad the output buffer to a target file offset with zero bytes.
fn pad_to_offset(out: &mut Vec<u8>, target: usize) {
    if out.len() < target {
        out.resize(target, 0);
    }
}

/// Align a `usize` value upward to the given alignment.
fn dyn_align_up_usize(value: usize, align: usize) -> usize {
    if align <= 1 { return value; }
    (value + align - 1) & !(align - 1)
}

/// Align a `u64` value upward to the given alignment.
fn dyn_align_up(value: u64, align: u64) -> u64 {
    if align <= 1 { return value; }
    (value + align - 1) & !(align - 1)
}

/// Compute padding bytes needed to align `current` to `align`.
fn dyn_align_pad(current: u64, align: u64) -> usize {
    if align <= 1 { return 0; }
    let aligned = dyn_align_up(current, align);
    (aligned - current) as usize
}

/// Writes an ELF header directly to the output buffer.
#[allow(clippy::too_many_arguments)]
fn write_elf_header_raw(
    out: &mut Vec<u8>,
    is_64bit: bool,
    elf_type: u16,
    machine: u16,
    entry: u64,
    phoff: u64,
    shoff: u64,
    phentsize: u16,
    phnum: u16,
    shentsize: u16,
    shnum: u16,
    shstrndx: u16,
) {
    // ELF magic
    out.extend_from_slice(&[0x7f, b'E', b'L', b'F']);

    if is_64bit {
        out.push(2); // EI_CLASS = ELFCLASS64
    } else {
        out.push(1); // EI_CLASS = ELFCLASS32
    }
    out.push(1); // EI_DATA = ELFDATA2LSB (little-endian)
    out.push(1); // EI_VERSION = EV_CURRENT
    out.push(0); // EI_OSABI = ELFOSABI_NONE
    out.extend_from_slice(&[0; 8]); // EI_ABIVERSION + padding

    if is_64bit {
        // 64-bit ELF header (total 64 bytes)
        out.extend_from_slice(&elf_type.to_le_bytes());    // e_type
        out.extend_from_slice(&machine.to_le_bytes());     // e_machine
        out.extend_from_slice(&1u32.to_le_bytes());        // e_version
        out.extend_from_slice(&entry.to_le_bytes());       // e_entry
        out.extend_from_slice(&phoff.to_le_bytes());       // e_phoff
        out.extend_from_slice(&shoff.to_le_bytes());       // e_shoff
        out.extend_from_slice(&0u32.to_le_bytes());        // e_flags
        out.extend_from_slice(&64u16.to_le_bytes());       // e_ehsize
        out.extend_from_slice(&phentsize.to_le_bytes());   // e_phentsize
        out.extend_from_slice(&phnum.to_le_bytes());       // e_phnum
        out.extend_from_slice(&shentsize.to_le_bytes());   // e_shentsize
        out.extend_from_slice(&shnum.to_le_bytes());       // e_shnum
        out.extend_from_slice(&shstrndx.to_le_bytes());    // e_shstrndx
    } else {
        // 32-bit ELF header (total 52 bytes)
        out.extend_from_slice(&elf_type.to_le_bytes());
        out.extend_from_slice(&machine.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(entry as u32).to_le_bytes());
        out.extend_from_slice(&(phoff as u32).to_le_bytes());
        out.extend_from_slice(&(shoff as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&52u16.to_le_bytes());
        out.extend_from_slice(&phentsize.to_le_bytes());
        out.extend_from_slice(&phnum.to_le_bytes());
        out.extend_from_slice(&shentsize.to_le_bytes());
        out.extend_from_slice(&shnum.to_le_bytes());
        out.extend_from_slice(&shstrndx.to_le_bytes());
    }
}

/// Writes a single program header to the output buffer.
#[allow(clippy::too_many_arguments)]
fn write_phdr_raw(
    out: &mut Vec<u8>,
    is_64bit: bool,
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
) {
    let p_paddr = p_vaddr;
    if is_64bit {
        out.extend_from_slice(&p_type.to_le_bytes());
        out.extend_from_slice(&p_flags.to_le_bytes());
        out.extend_from_slice(&p_offset.to_le_bytes());
        out.extend_from_slice(&p_vaddr.to_le_bytes());
        out.extend_from_slice(&p_paddr.to_le_bytes());
        out.extend_from_slice(&p_filesz.to_le_bytes());
        out.extend_from_slice(&p_memsz.to_le_bytes());
        out.extend_from_slice(&p_align.to_le_bytes());
    } else {
        out.extend_from_slice(&p_type.to_le_bytes());
        out.extend_from_slice(&(p_offset as u32).to_le_bytes());
        out.extend_from_slice(&(p_vaddr as u32).to_le_bytes());
        out.extend_from_slice(&(p_paddr as u32).to_le_bytes());
        out.extend_from_slice(&(p_filesz as u32).to_le_bytes());
        out.extend_from_slice(&(p_memsz as u32).to_le_bytes());
        out.extend_from_slice(&p_flags.to_le_bytes());
        out.extend_from_slice(&(p_align as u32).to_le_bytes());
    }
}

/// Writes a single section header to the output buffer.
#[allow(clippy::too_many_arguments)]
fn write_shdr_raw(
    out: &mut Vec<u8>,
    is_64bit: bool,
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
) {
    if is_64bit {
        out.extend_from_slice(&sh_name.to_le_bytes());
        out.extend_from_slice(&sh_type.to_le_bytes());
        out.extend_from_slice(&sh_flags.to_le_bytes());
        out.extend_from_slice(&sh_addr.to_le_bytes());
        out.extend_from_slice(&sh_offset.to_le_bytes());
        out.extend_from_slice(&sh_size.to_le_bytes());
        out.extend_from_slice(&sh_link.to_le_bytes());
        out.extend_from_slice(&sh_info.to_le_bytes());
        out.extend_from_slice(&sh_addralign.to_le_bytes());
        out.extend_from_slice(&sh_entsize.to_le_bytes());
    } else {
        out.extend_from_slice(&sh_name.to_le_bytes());
        out.extend_from_slice(&sh_type.to_le_bytes());
        out.extend_from_slice(&(sh_flags as u32).to_le_bytes());
        out.extend_from_slice(&(sh_addr as u32).to_le_bytes());
        out.extend_from_slice(&(sh_offset as u32).to_le_bytes());
        out.extend_from_slice(&(sh_size as u32).to_le_bytes());
        out.extend_from_slice(&sh_link.to_le_bytes());
        out.extend_from_slice(&sh_info.to_le_bytes());
        out.extend_from_slice(&(sh_addralign as u32).to_le_bytes());
        out.extend_from_slice(&(sh_entsize as u32).to_le_bytes());
    }
}

/// Builds ELF program headers for the linked output.
///
/// Creates PT_LOAD segments for code (R+X), read-only data (R), and
/// read-write data (R+W), plus PT_PHDR and PT_GNU_STACK headers.
fn build_program_headers(
    elf: &mut ElfWriter,
    section_merger: &SectionMerger,
    _config: &CodegenConfig,
    base_address: u64,
    page_size: u64,
) {
    // PT_PHDR — program header table self-reference
    let phdr = ProgramHeader {
        p_type: PT_PHDR,
        p_flags: 0x4,   // PF_R
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
            s.flags & SHF_ALLOC != 0 && s.flags & SHF_WRITE == 0 && s.flags & SHF_EXECINSTR == 0
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
