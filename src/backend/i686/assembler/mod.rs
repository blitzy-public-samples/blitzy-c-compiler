//! Built-in i686 (32-bit x86) assembler module.
//!
//! This module implements the integrated assembler for the i686 (IA-32) target
//! architecture, encoding 32-bit x86 machine instructions without REX prefixes
//! and recording ELF relocations for unresolved symbols. Unlike x86-64, i686
//! uses the classic ModR/M and SIB encoding without REX byte extensions, and
//! relocations use the REL format (inline addends) rather than RELA.
//!
//! The assembler operates entirely in-process — no external `as` or `llvm-mc`
//! is ever invoked, enforcing the standalone backend mandate (Section 0.7.7).
//!
//! Output is relocatable object code consumed by the i686 linker to produce
//! final ELF executables (`ET_EXEC`) or shared objects (`ET_DYN`).
//!
//! # Sub-modules
//!
//! - [`encoder`]: i686 instruction encoder — ModR/M, SIB, prefix byte
//!   generation and opcode dispatch for all i686 instructions including
//!   integer ALU, shifts, data movement, control flow, x87 FPU, and
//!   conditional operations. Produces raw machine code bytes from
//!   `MachineInstr` input. No REX prefixes are emitted.
//! - [`relocations`]: i686-specific ELF relocation type definitions —
//!   `R_386_*` constants, metadata queries (size, PC-relative, GOT/PLT
//!   requirements), REL format helpers, PIC-aware relocation selection,
//!   and relocation application functions for all i686 relocation types.
//!
//! # Key Differences from x86-64
//!
//! - No REX prefix — only 8 GPRs (EAX–EDI) directly encodable
//! - REL relocations (8-byte entries: offset + info) instead of RELA
//!   (24-byte entries: offset + info + addend) — addend is inline in the
//!   instruction stream
//! - PIC code uses `__i686.get_pc_thunk.bx` + `R_386_GOTPC` to establish
//!   the GOT base in EBX, rather than RIP-relative addressing
//! - All addresses and relocations are 32-bit (no 64-bit relocations)
//!
//! # Usage
//!
//! ```ignore
//! use crate::backend::i686::assembler::I686Assembler;
//!
//! let mut asm = I686Assembler::new();
//! let result = asm.assemble_function(&machine_function);
//! // result.code contains raw machine code bytes
//! // result.relocations contains unresolved symbol references
//! ```

// ---------------------------------------------------------------------------
// Sub-module declarations
// ---------------------------------------------------------------------------

/// i686 instruction encoder — handles ModR/M, SIB, prefix byte generation
/// and opcode dispatch for all i686 machine instructions.
pub mod encoder;

/// i686 ELF relocation type definitions used by both the assembler (to record
/// relocations during instruction encoding) and the linker (to apply relocations
/// when producing final ELF executables and shared objects).
pub mod relocations;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

/// Re-export the core encoding function for direct use by consumers.
pub use encoder::encode_instruction;

/// Re-export the relocation type enum for consumers that need to inspect
/// or create relocation entries.
pub use relocations::I686RelocType;

// ---------------------------------------------------------------------------
// Imports from the crate
// ---------------------------------------------------------------------------

use crate::backend::i686::registers;
use crate::backend::traits::{MachineFunction, MachineInstr, MachineOperand, PhysReg};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{fx_hash_map, FxHashMap};

use self::encoder::{EncodedInstr, EncoderContext};

// ---------------------------------------------------------------------------
// SectionKind — identifies the ELF output section
// ---------------------------------------------------------------------------

/// Identifies which ELF section a symbol or data block belongs to.
///
/// Used in [`AsmSymbol`] to classify where each assembled entity resides
/// and in the [`AssembledModule`] output to organize section data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionKind {
    /// Executable code section (`.text`).
    Text,
    /// Initialized read-write data section (`.data`).
    Data,
    /// Initialized read-only data section (`.rodata`).
    Rodata,
    /// Uninitialized data section (`.bss`) — occupies no file space.
    Bss,
}

impl SectionKind {
    /// Returns the conventional ELF section name string.
    pub fn name(&self) -> &'static str {
        match self {
            SectionKind::Text => ".text",
            SectionKind::Data => ".data",
            SectionKind::Rodata => ".rodata",
            SectionKind::Bss => ".bss",
        }
    }

    /// Returns `true` if this section is loadable (occupies file space).
    pub fn is_loadable(&self) -> bool {
        !matches!(self, SectionKind::Bss)
    }

    /// Returns `true` if this section contains executable code.
    pub fn is_executable(&self) -> bool {
        matches!(self, SectionKind::Text)
    }

    /// Returns `true` if this section is writable at runtime.
    pub fn is_writable(&self) -> bool {
        matches!(self, SectionKind::Data | SectionKind::Bss)
    }
}

// ---------------------------------------------------------------------------
// FixupKind — internal displacement field width
// ---------------------------------------------------------------------------

/// The width of a relative fixup displacement field.
///
/// When encoding a branch or jump instruction whose target label has not yet
/// been seen (forward reference), the encoder emits a placeholder displacement.
/// The fixup kind determines the width of that placeholder and how it is
/// patched once the target label position is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixupKind {
    /// 8-bit signed relative displacement (short branches, `Jcc rel8`).
    Rel8,
    /// 32-bit signed relative displacement (near branches/calls, `Jcc rel32`,
    /// `JMP rel32`, `CALL rel32`).
    Rel32,
}

impl FixupKind {
    /// Number of displacement bytes consumed by this fixup kind.
    #[inline]
    fn size(&self) -> u32 {
        match self {
            FixupKind::Rel8 => 1,
            FixupKind::Rel32 => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// PendingFixup — internal forward-reference tracking
// ---------------------------------------------------------------------------

/// A pending fixup for an intra-function branch or jump whose target label
/// has not yet been resolved to a byte offset.
///
/// After all basic blocks in a function have been encoded, the assembler
/// iterates over pending fixups and patches the displacement fields with the
/// computed relative offsets.
#[derive(Debug, Clone)]
struct PendingFixup {
    /// Byte offset within the code buffer where the displacement field
    /// begins. This is the position that will be patched.
    code_offset: u32,
    /// Name of the target label, derived from the basic block's
    /// [`MachineBasicBlock::effective_label()`].
    target_label: String,
    /// Width of the displacement field to patch.
    kind: FixupKind,
}

// ---------------------------------------------------------------------------
// RelocationEntry
// ---------------------------------------------------------------------------

/// An ELF relocation entry produced during assembly, representing an
/// unresolved reference to an external (or section-relative) symbol that
/// the linker must patch when producing the final ELF output.
///
/// For i686, the native ELF format is REL (8-byte entries with inline
/// addends), but this struct carries the addend explicitly for ease of
/// processing during linker consumption.
///
/// # Fields
///
/// - `offset`: Position of the relocation site within the containing section
/// - `reloc_type`: Architecture-specific relocation kind (`R_386_*`)
/// - `symbol`: Name of the referenced symbol
/// - `addend`: Value to add during relocation application
#[derive(Debug, Clone)]
pub struct RelocationEntry {
    /// Byte offset of the relocation site, relative to the start of the
    /// containing section.
    pub offset: u32,
    /// The i686-specific relocation type (e.g. `R_386_PC32`, `R_386_32`).
    pub reloc_type: I686RelocType,
    /// Name of the referenced symbol.
    pub symbol: String,
    /// Addend value to add during relocation application. For REL-format
    /// relocations this addend is also stored inline at the relocation
    /// site in the instruction stream.
    pub addend: i32,
}

impl RelocationEntry {
    /// Compute the ELF `r_info` value for this relocation entry.
    ///
    /// For i686 ELF (Elf32_Rel), `r_info` encodes the symbol table index
    /// in the upper 24 bits and the relocation type in the lower 8 bits:
    ///
    /// ```text
    /// r_info = (sym_index << 8) | reloc_type
    /// ```
    ///
    /// # Arguments
    ///
    /// * `sym_index` — The symbol's index in the ELF `.symtab` section.
    ///   Must be resolved by the linker or ELF writer before calling this.
    pub fn to_elf_info(&self, sym_index: u32) -> u32 {
        (sym_index << 8) | (self.reloc_type.to_elf_value() as u32)
    }

    /// Returns `true` if this relocation is PC-relative.
    ///
    /// Delegates to [`I686RelocType::is_pc_relative()`].
    #[inline]
    pub fn is_pc_relative(&self) -> bool {
        self.reloc_type.is_pc_relative()
    }

    /// Returns the human-readable name of this relocation's type.
    ///
    /// Delegates to [`I686RelocType::name()`].
    #[inline]
    pub fn type_name(&self) -> &'static str {
        self.reloc_type.name()
    }
}

// ---------------------------------------------------------------------------
// AsmSymbol
// ---------------------------------------------------------------------------

/// A symbol definition emitted by the assembler, representing a function,
/// global variable, or section label with its resolved byte offset.
///
/// Symbols are collected during module assembly and emitted into the ELF
/// `.symtab` section by the linker or ELF writer.
#[derive(Debug, Clone)]
pub struct AsmSymbol {
    /// Symbol name (e.g. `"main"`, `"_start"`, `".L0"`).
    pub name: String,
    /// Byte offset within the containing section.
    pub offset: u32,
    /// Which section this symbol belongs to.
    pub section: SectionKind,
    /// Whether the symbol has global (external) visibility in the ELF
    /// symbol table (`STB_GLOBAL`).
    pub is_global: bool,
    /// Whether the symbol has weak binding (`STB_WEAK`), allowing it to
    /// be overridden by a strong definition from another object.
    pub is_weak: bool,
    /// Size of the entity this symbol represents (in bytes). For functions,
    /// this is the total code size; for objects, it is the data size.
    pub size: u32,
}

// ---------------------------------------------------------------------------
// AssembledFunction
// ---------------------------------------------------------------------------

/// The assembled output of a single function, containing machine code bytes
/// and any relocations that reference external symbols.
///
/// Internal branch/jump fixups have already been resolved — the code bytes
/// contain correct relative displacements for all intra-function branches.
/// Only relocations for external (inter-object) symbol references remain.
#[derive(Debug, Clone)]
pub struct AssembledFunction {
    /// Function name (matches the originating `MachineFunction::name`).
    pub name: String,
    /// Raw i686 machine code bytes for this function.
    pub code: Vec<u8>,
    /// Relocations for external symbol references within this function.
    /// Offsets are relative to the start of this function's code.
    pub relocations: Vec<RelocationEntry>,
    /// Total size of the function code in bytes.
    pub size: u32,
}

// ---------------------------------------------------------------------------
// AssembledModule
// ---------------------------------------------------------------------------

/// The assembled output of an entire compilation unit (module), containing
/// all ELF sections ready for object file emission or linker consumption.
///
/// The assembler produces this structure after assembling all functions and
/// global data declarations in a module. It contains the complete content
/// for `.text`, `.data`, `.rodata`, and `.bss` sections, along with the
/// symbol table and relocation entries.
#[derive(Debug, Clone)]
pub struct AssembledModule {
    /// Concatenated machine code for all functions (`.text` section).
    pub text: Vec<u8>,
    /// Initialized read-write data (`.data` section).
    pub data: Vec<u8>,
    /// Initialized read-only data (`.rodata` section).
    pub rodata: Vec<u8>,
    /// Total size of uninitialized data (`.bss` section) in bytes.
    pub bss_size: u32,
    /// All symbol definitions (functions, globals, section labels).
    pub symbols: Vec<AsmSymbol>,
    /// All relocations across all sections, with offsets relative to the
    /// start of their respective sections.
    pub relocations: Vec<RelocationEntry>,
}

// ---------------------------------------------------------------------------
// I686Assembler — the main assembler driver
// ---------------------------------------------------------------------------

/// Built-in i686 assembler that encodes [`MachineFunction`] output from
/// instruction selection into 32-bit x86 machine code.
///
/// The assembler performs the following pipeline for each function:
///
/// 1. Pre-scans basic blocks to build a block-ID → label-name mapping
/// 2. Iterates blocks in layout order, recording label byte offsets
/// 3. Delegates instruction encoding to [`encoder::encode_instruction`]
/// 4. Collects external-symbol relocations (e.g. `R_386_PC32` for calls)
/// 5. Records internal branch fixups for forward references
/// 6. Resolves all intra-function fixups (patching displacement fields)
/// 7. Returns assembled code bytes and unresolved relocations
///
/// For full module assembly, the assembler concatenates function code into
/// a `.text` section, collects `.data`/`.rodata`/`.bss` content, and
/// produces an [`AssembledModule`] ready for ELF object emission.
///
/// No external `as` binary is invoked — this is the standalone backend
/// as mandated by Section 0.7.7.
pub struct I686Assembler {
    /// Accumulated encoded instruction bytes for the `.text` section.
    code: Vec<u8>,
    /// Accumulated initialized data bytes for the `.data` section.
    data: Vec<u8>,
    /// Accumulated read-only data bytes for the `.rodata` section.
    rodata: Vec<u8>,
    /// Accumulated BSS segment size in bytes.
    bss_size: u32,
    /// Collected relocation entries for external symbol references.
    relocations: Vec<RelocationEntry>,
    /// Local and global symbol definitions.
    symbols: Vec<AsmSymbol>,
    /// Current write position in the code buffer (next instruction offset).
    current_offset: u32,
    /// Resolved label name → byte offset mapping. Populated as basic blocks
    /// are encoded; consumed during fixup resolution.
    label_offsets: FxHashMap<String, u32>,
    /// Unresolved intra-function branch/jump targets awaiting fixup after
    /// all blocks in the current function have been encoded.
    pending_fixups: Vec<PendingFixup>,
    /// Whether to generate PIC (Position-Independent Code) relocations.
    /// When true, symbol references use GOT/PLT-relative relocation types
    /// (`R_386_GOT32`, `R_386_GOTOFF`, `R_386_GOTPC`, `R_386_PLT32`)
    /// instead of absolute references. Enables `__i686.get_pc_thunk.*`
    /// patterns for obtaining the GOT base address.
    pic_mode: bool,
    /// Diagnostic engine for reporting assembler errors and warnings.
    ///
    /// Errors are emitted for unresolvable label references (indicates a
    /// code-generator bug) and for relocation overflow conditions.
    /// Warnings are emitted for rel8 displacement overflow (truncation).
    /// After assembly, the caller can query `has_errors()` to determine
    /// whether the produced machine code is valid.
    diagnostics: DiagnosticEngine,
}

impl Default for I686Assembler {
    /// Provides a default i686 assembler with empty buffers and non-PIC
    /// configuration, equivalent to [`I686Assembler::new()`].
    fn default() -> Self {
        Self::new()
    }
}

impl I686Assembler {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new i686 assembler with empty buffers and default (non-PIC)
    /// configuration.
    ///
    /// No target configuration beyond i686 defaults (32-bit, little-endian)
    /// is required — the i686 ISA is fully determined.
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096),
            data: Vec::new(),
            rodata: Vec::new(),
            bss_size: 0,
            relocations: Vec::new(),
            symbols: Vec::new(),
            current_offset: 0,
            label_offsets: fx_hash_map(),
            pending_fixups: Vec::new(),
            pic_mode: false,
            diagnostics: DiagnosticEngine::new(),
        }
    }

    /// Create a new i686 assembler configured for PIC code generation.
    ///
    /// When `pic` is `true`, the assembler emits GOT-relative relocations
    /// (`R_386_GOT32`, `R_386_GOTOFF`, `R_386_GOTPC`) and PLT call
    /// relocations (`R_386_PLT32`) instead of absolute references. This is
    /// required when producing shared objects (`.so`) or position-independent
    /// executables (PIE).
    pub fn with_pic(pic: bool) -> Self {
        let mut asm = Self::new();
        asm.pic_mode = pic;
        asm
    }

    // -----------------------------------------------------------------------
    // Function assembly
    // -----------------------------------------------------------------------

    /// Assemble a single function from its machine-level representation.
    ///
    /// This method processes a [`MachineFunction`] produced by the i686
    /// instruction selector and register allocator, encoding each machine
    /// instruction into raw i686 binary bytes.
    ///
    /// # Process
    ///
    /// 1. Pre-scan basic blocks to build a block-ID → label-name mapping
    /// 2. Iterate blocks in layout order, recording each block's entry
    ///    byte offset as the label position
    /// 3. For each instruction, delegate to [`encoder::encode_instruction`]
    /// 4. Collect external-symbol relocations from the encoder
    /// 5. Detect internal label references and record fixups
    /// 6. After all blocks, resolve fixups by patching displacement fields
    /// 7. Extract function code and relocations into [`AssembledFunction`]
    ///
    /// # Returns
    ///
    /// An [`AssembledFunction`] containing the raw machine code, any
    /// unresolved external-symbol relocations (offsets relative to the
    /// function start), and the total code size.
    pub fn assemble_function(&mut self, mf: &MachineFunction) -> AssembledFunction {
        // Save the code-buffer offset at function entry so we can extract
        // only this function's bytes afterwards.
        let func_start = self.current_offset;

        // Clear per-function state from any previous assembly pass.
        self.label_offsets.clear();
        self.pending_fixups.clear();

        // Pre-build a block-ID → label-name mapping so that we can
        // translate MachineOperand::Label(id) references during encoding.
        // If the block has an explicit label we use it directly; otherwise
        // `effective_label()` synthesises `.LBB_<id>`.
        let mut block_id_to_label: FxHashMap<u32, String> = fx_hash_map();
        for block in &mf.blocks {
            // Prefer the explicit label when present (preserves user-defined
            // or code-gen-defined names); fall back to the auto-generated
            // label based on block ID.
            let lbl = match &block.label {
                Some(l) => l.clone(),
                None => block.effective_label(),
            };
            block_id_to_label.insert(block.id, lbl);
        }

        // --- Main encoding loop -------------------------------------------
        for block in &mf.blocks {
            // Record the byte offset for this block's label. This is the
            // address that intra-function branches will resolve to.
            let label_name = block.effective_label();

            // Detect duplicate label definitions within the same function.
            // This should not happen in well-formed IR but would indicate a
            // code-generator bug if it does — warn so the problem is visible.
            if self.label_offsets.contains_key(&label_name) {
                self.diagnostics.warning(
                    Span::DUMMY,
                    format!(
                        "assembler: duplicate label '{}' in function '{}' \
                         at code offset 0x{:x}",
                        label_name, mf.name, self.current_offset
                    ),
                );
            }
            self.label_offsets.insert(label_name, self.current_offset);

            for instr in &block.instructions {
                // Determine if this instruction targets an internal label
                // (e.g. a Jcc or Jmp with a Label operand). If so, we will
                // need to create a fixup after encoding.
                let label_target = Self::find_label_target(instr, &block_id_to_label);

                // Prepare encoder context at the current code offset.
                let mut ctx = EncoderContext {
                    offset: self.current_offset,
                    relocations: Vec::new(),
                    pic_mode: self.pic_mode,
                };

                // Encode the instruction into raw machine-code bytes.
                let encoded: EncodedInstr = encode_instruction(instr, &mut ctx);
                let instr_len = encoded.bytes.len() as u32;

                // Guard: the encoder should always produce at least one byte
                // for non-pseudo instructions. Log opcode details for
                // diagnosis if encoding produces zero bytes.
                if instr_len == 0 && !instr.operands.is_empty() {
                    self.diagnostics.warning(
                        Span::DUMMY,
                        format!(
                            "assembler: encoder produced 0 bytes for opcode {} \
                             with {} operand(s) at code offset 0x{:x}",
                            instr.opcode,
                            instr.operands.len(),
                            self.current_offset,
                        ),
                    );
                }

                // Append encoded bytes to the code buffer.
                self.code.extend_from_slice(&encoded.bytes);

                // Collect external-symbol relocations emitted by the encoder.
                // These come from two sources: the returned EncodedInstr and
                // the mutable EncoderContext (encoder may use either path).
                for reloc in encoded.relocations {
                    self.relocations.push(reloc);
                }
                for reloc in ctx.relocations {
                    self.relocations.push(reloc);
                }

                // If the instruction references an internal label (branch
                // or jump), record a fixup for later resolution. The
                // displacement field is at the tail of the encoded bytes:
                //   - rel32: last 4 bytes (Jcc rel32, JMP rel32, CALL rel32)
                //   - rel8:  last 1 byte  (Jcc rel8, JMP rel8)
                if let Some(target_label) = label_target {
                    self.record_label_fixup(instr_len, target_label);
                }

                self.current_offset += instr_len;
            }
        }

        // Resolve all intra-function branch/jump fixups now that every
        // label's byte offset is known.
        self.resolve_fixups();

        // Extract the function's code bytes from the accumulated buffer.
        let func_end = self.current_offset;
        let func_size = func_end - func_start;
        let func_code = self.code[func_start as usize..func_end as usize].to_vec();

        // Collect relocations that fall within this function's code range
        // and adjust their offsets to be function-relative.
        let func_relocs: Vec<RelocationEntry> = self
            .relocations
            .iter()
            .filter(|r| r.offset >= func_start && r.offset < func_end)
            .map(|r| RelocationEntry {
                offset: r.offset - func_start,
                reloc_type: r.reloc_type,
                symbol: r.symbol.clone(),
                addend: r.addend,
            })
            .collect();

        AssembledFunction {
            name: mf.name.clone(),
            code: func_code,
            relocations: func_relocs,
            size: func_size,
        }
    }

    // -----------------------------------------------------------------------
    // Module assembly
    // -----------------------------------------------------------------------

    /// Assemble an entire compilation module (a collection of functions) and
    /// produce an [`AssembledModule`] containing all ELF section data.
    ///
    /// This method:
    /// 1. Resets all internal state for a clean module assembly
    /// 2. Assembles each function in sequence, accumulating code bytes
    /// 3. Records a global symbol for each function at its code offset
    /// 4. Returns the complete module with `.text`, `.data`, `.rodata`,
    ///    `.bss`, symbol table, and relocation table
    ///
    /// # Arguments
    ///
    /// * `module` — Slice of [`MachineFunction`]s representing all functions
    ///   in the compilation unit, in the desired layout order.
    ///
    /// # Returns
    ///
    /// An [`AssembledModule`] ready for ELF object file emission or direct
    /// consumption by the i686 linker.
    pub fn assemble_module(&mut self, module: &[MachineFunction]) -> AssembledModule {
        // Reset all state for a fresh module assembly.
        self.reset();

        let mut assembled_functions: Vec<AssembledFunction> = Vec::with_capacity(module.len());

        for mf in module {
            // Record the function's start offset before assembly.
            let func_start = self.current_offset;

            let assembled = self.assemble_function(mf);

            // Record a global symbol for this function in the `.text` section.
            self.symbols.push(AsmSymbol {
                name: assembled.name.clone(),
                offset: func_start,
                section: SectionKind::Text,
                is_global: true,
                is_weak: false,
                size: assembled.size,
            });

            assembled_functions.push(assembled);
        }

        // Build the final output. The code buffer already contains the
        // complete `.text` section (all functions concatenated in order).
        // Relocations are section-relative since the code buffer starts
        // at offset 0.
        AssembledModule {
            text: self.code.clone(),
            data: self.data.clone(),
            rodata: self.rodata.clone(),
            bss_size: self.bss_size,
            symbols: self.symbols.clone(),
            relocations: self.relocations.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Data section helpers
    // -----------------------------------------------------------------------

    /// Append initialized data to the `.data` section and return the byte
    /// offset at which it was placed.
    pub fn add_data(&mut self, bytes: &[u8]) -> u32 {
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(bytes);
        offset
    }

    /// Append read-only data to the `.rodata` section and return the byte
    /// offset at which it was placed.
    pub fn add_rodata(&mut self, bytes: &[u8]) -> u32 {
        let offset = self.rodata.len() as u32;
        self.rodata.extend_from_slice(bytes);
        offset
    }

    /// Reserve `size` bytes in the `.bss` section, aligned to the given
    /// alignment boundary, and return the start offset of the reservation.
    ///
    /// The `.bss` section is zero-initialized at load time and occupies no
    /// file space in the ELF object.
    pub fn add_bss(&mut self, size: u32, align: u32) -> u32 {
        // Ensure alignment is at least 1.
        let align = if align == 0 { 1 } else { align };
        let aligned = (self.bss_size + align - 1) & !(align - 1);
        self.bss_size = aligned + size;
        aligned
    }

    /// Add an explicit symbol definition to the symbol table.
    pub fn add_symbol(&mut self, sym: AsmSymbol) {
        self.symbols.push(sym);
    }

    /// Add an explicit relocation entry to the relocation table.
    pub fn add_relocation(&mut self, entry: RelocationEntry) {
        self.relocations.push(entry);
    }

    /// Align the `.data` section to the given power-of-two byte boundary,
    /// padding with zero bytes.
    pub fn align_data(&mut self, align: u32) {
        let align = if align == 0 { 1 } else { align };
        let current = self.data.len() as u32;
        let aligned = (current + align - 1) & !(align - 1);
        let padding = (aligned - current) as usize;
        self.data.extend(core::iter::repeat(0u8).take(padding));
    }

    /// Align the `.rodata` section to the given power-of-two byte boundary,
    /// padding with zero bytes.
    pub fn align_rodata(&mut self, align: u32) {
        let align = if align == 0 { 1 } else { align };
        let current = self.rodata.len() as u32;
        let aligned = (current + align - 1) & !(align - 1);
        let padding = (aligned - current) as usize;
        self.rodata.extend(core::iter::repeat(0u8).take(padding));
    }

    /// Align the `.text` (code) section to the given power-of-two byte
    /// boundary, padding with NOP instructions (`0x90`).
    ///
    /// NOP-padding is used instead of zero-padding because the `.text`
    /// section is executable and may be reached by fall-through.
    pub fn align_code(&mut self, align: u32) {
        let align = if align == 0 { 1 } else { align };
        let current = self.current_offset;
        let aligned = (current + align - 1) & !(align - 1);
        let padding = (aligned - current) as usize;
        // Pad with single-byte NOP (0x90). Multi-byte NOPs could be used
        // for better performance but single-byte is simplest and correct.
        for _ in 0..padding {
            self.code.push(0x90);
        }
        self.current_offset = aligned;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return the current code buffer size (also the next instruction
    /// offset within the `.text` section).
    #[inline]
    pub fn code_offset(&self) -> u32 {
        self.current_offset
    }

    /// Return whether PIC mode is active.
    #[inline]
    pub fn is_pic(&self) -> bool {
        self.pic_mode
    }

    /// Set or clear PIC (Position-Independent Code) mode.
    pub fn set_pic(&mut self, pic: bool) {
        self.pic_mode = pic;
    }

    /// Return a reference to the accumulated code bytes.
    pub fn code_bytes(&self) -> &[u8] {
        &self.code
    }

    /// Return a reference to the collected relocations.
    pub fn relocation_entries(&self) -> &[RelocationEntry] {
        &self.relocations
    }

    /// Return a reference to the collected symbols.
    pub fn symbol_entries(&self) -> &[AsmSymbol] {
        &self.symbols
    }

    /// Return `true` if any errors were reported during assembly.
    ///
    /// This delegates to [`DiagnosticEngine::has_errors()`] and should be
    /// checked after [`assemble_function`](Self::assemble_function) or
    /// [`assemble_module`](Self::assemble_module) to determine whether the
    /// produced machine code is valid. Common errors include unresolved
    /// label references (code-generator bug) and relocation overflow.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    /// Return a reference to the assembler's diagnostic engine.
    ///
    /// Callers can use this to inspect collected warnings and errors, or
    /// to render diagnostics to stderr.
    pub fn diagnostics(&self) -> &DiagnosticEngine {
        &self.diagnostics
    }

    /// Return a mutable reference to the assembler's diagnostic engine.
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticEngine {
        &mut self.diagnostics
    }

    // -----------------------------------------------------------------------
    // Internal fixup resolution
    // -----------------------------------------------------------------------

    /// Resolve all pending intra-function fixups by patching relative
    /// displacement fields in the code buffer.
    ///
    /// # Displacement Formulas
    ///
    /// - **`Rel32`:** `displacement = target_offset − (fixup_offset + 4)`
    /// - **`Rel8`:**  `displacement = target_offset − (fixup_offset + 1)`
    ///
    /// Both are written as signed little-endian values at the fixup site.
    ///
    /// # Error Handling
    ///
    /// - Unresolved labels (programming error in the code generator) are
    ///   silently skipped with zeroed displacement, which will surface as
    ///   incorrect branch targets during testing.
    /// - `Rel8` overflow (displacement exceeds ±127 bytes) is truncated;
    ///   a production assembler would relax to `Rel32` but the encoder
    ///   should already emit `Rel32` for forward references of unknown
    ///   distance.
    fn resolve_fixups(&mut self) {
        // Drain pending fixups to avoid borrow-checker conflicts, since we
        // need mutable access to self.code while iterating fixups.
        let fixups: Vec<PendingFixup> = self.pending_fixups.drain(..).collect();

        for fixup in fixups {
            let target_offset = match self.label_offsets.get(&fixup.target_label) {
                Some(&offset) => offset,
                None => {
                    // Label not found — this indicates a code generator bug
                    // (the target block is missing). Report the error and
                    // leave displacement as zero, which will manifest as a
                    // jump-to-self — easy to diagnose in testing.
                    self.diagnostics.error(
                        Span::DUMMY,
                        format!(
                            "assembler: unresolved label '{}' at code offset 0x{:x}",
                            fixup.target_label, fixup.code_offset
                        ),
                    );
                    continue;
                }
            };

            // Compute the relative displacement. The instruction pointer
            // at the time of execution has already advanced past the
            // displacement field, so we subtract the end of the field.
            let fixup_end = fixup.code_offset as i64 + fixup.kind.size() as i64;
            let displacement = target_offset as i64 - fixup_end;

            match fixup.kind {
                FixupKind::Rel32 => {
                    let disp32 = displacement as i32;
                    let bytes = disp32.to_le_bytes();
                    let off = fixup.code_offset as usize;
                    if off + 4 <= self.code.len() {
                        self.code[off..off + 4].copy_from_slice(&bytes);
                    }
                }
                FixupKind::Rel8 => {
                    // Check for rel8 overflow: displacement must fit in
                    // a signed 8-bit value [-128, +127].
                    if !(-128..=127).contains(&displacement) {
                        self.diagnostics.warning(
                            Span::DUMMY,
                            format!(
                                "assembler: rel8 displacement overflow ({}) for \
                                 label '{}' at code offset 0x{:x}; \
                                 encoder should have used rel32",
                                displacement, fixup.target_label, fixup.code_offset
                            ),
                        );
                    }
                    let off = fixup.code_offset as usize;
                    if off < self.code.len() {
                        self.code[off] = displacement as i8 as u8;
                    }
                }
            }
        }
    }

    /// Record a label fixup for the instruction that was just encoded.
    ///
    /// Determines the fixup kind (rel8 vs rel32) and the displacement
    /// field position based on the instruction length, then adds a
    /// [`PendingFixup`] to the pending list.
    ///
    /// # Arguments
    ///
    /// * `instr_len` — Total encoded length of the instruction in bytes
    /// * `target_label` — Name of the target block's label
    fn record_label_fixup(&mut self, instr_len: u32, target_label: String) {
        if instr_len == 0 {
            return;
        }

        // Heuristic for fixup kind: instructions ≥ 2 bytes with a short
        // branch pattern (opcode + rel8) use Rel8 if the instruction is
        // exactly 2 bytes (e.g. `EB cb` for JMP rel8, `70+cc cb` for
        // short Jcc). Otherwise, use Rel32 (the common case for near
        // branches and calls).
        let (kind, disp_offset) = if instr_len == 2 {
            // Short branch: opcode byte + 1-byte displacement
            (FixupKind::Rel8, self.current_offset + instr_len - 1)
        } else if instr_len >= 5 {
            // Near branch/call: opcode(s) + 4-byte displacement
            // Covers: E9 cd (JMP rel32), E8 cd (CALL rel32),
            //         0F 80+cc cd (Jcc rel32)
            (FixupKind::Rel32, self.current_offset + instr_len - 4)
        } else if instr_len >= 2 {
            // Other short-ish encodings — treat as rel32 if we have at
            // least 4 bytes of displacement, otherwise rel8.
            if instr_len >= 4 {
                (FixupKind::Rel32, self.current_offset + instr_len - 4)
            } else {
                // 3-byte instruction with label — likely a rel8 with
                // prefix or similar edge case.
                (FixupKind::Rel8, self.current_offset + instr_len - 1)
            }
        } else {
            // 1-byte instruction cannot contain a displacement field;
            // this should not occur for branch instructions.
            return;
        };

        self.pending_fixups.push(PendingFixup {
            code_offset: disp_offset,
            target_label,
            kind,
        });
    }

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /// Scan an instruction's operands for a [`MachineOperand::Label`] and
    /// return the corresponding label name string from the block-ID map.
    ///
    /// Only the first label operand is returned — branch/jump instructions
    /// typically have exactly one target label.
    fn find_label_target(
        instr: &MachineInstr,
        block_id_to_label: &FxHashMap<u32, String>,
    ) -> Option<String> {
        for operand in &instr.operands {
            if let MachineOperand::Label(id) = operand {
                return block_id_to_label.get(id).cloned();
            }
        }
        None
    }

    /// Reset all assembler state for a fresh module assembly.
    ///
    /// Called at the start of [`assemble_module`](Self::assemble_module) to
    /// ensure no state leaks between successive module assembly invocations.
    fn reset(&mut self) {
        self.code.clear();
        self.data.clear();
        self.rodata.clear();
        self.bss_size = 0;
        self.relocations.clear();
        self.symbols.clear();
        self.current_offset = 0;
        self.label_offsets.clear();
        self.pending_fixups.clear();
        // Re-create diagnostics to clear any accumulated errors/warnings
        // from a previous assembly pass.
        self.diagnostics = DiagnosticEngine::new();
    }
}

// ---------------------------------------------------------------------------
// Operand formatting for diagnostics
// ---------------------------------------------------------------------------

/// Format a [`MachineOperand`] as a human-readable string for diagnostic
/// messages and debug output.
///
/// Uses the operand-kind query methods (`is_register`, `is_immediate`,
/// `is_memory`) and i686 register utilities to produce assembly-like
/// syntax for each operand variant.
///
/// # Examples
///
/// ```ignore
/// let reg_op = MachineOperand::Register(PhysReg(0)); // EAX
/// assert_eq!(format_operand_debug(&reg_op), "%eax");
///
/// let imm_op = MachineOperand::Immediate(42);
/// assert_eq!(format_operand_debug(&imm_op), "$42");
/// ```
pub fn format_operand_debug(operand: &MachineOperand) -> String {
    if operand.is_register() {
        // Extract the physical register from the operand.
        if let MachineOperand::Register(reg) = operand {
            let idx = reg.index();
            // Validate that this is a known i686 register and format
            // with the AT&T-syntax percent prefix.
            if registers::is_gpr(*reg) {
                let enc = registers::encoding(*reg);
                format!("%{} (enc={})", registers::reg_name(*reg), enc)
            } else if (24..32).contains(&idx) {
                // x87 FPU register
                format!("%{}", registers::reg_name(*reg))
            } else if idx < 24 {
                // Sub-register (16-bit or 8-bit)
                let has_byte = if idx < 8 {
                    registers::has_byte_subreg(*reg)
                } else {
                    false
                };
                format!(
                    "%{} (sub-reg, byte_accessible={})",
                    registers::reg_name(*reg),
                    has_byte,
                )
            } else {
                format!("PhysReg({})", idx)
            }
        } else {
            "reg?".to_string()
        }
    } else if operand.is_immediate() {
        if let MachineOperand::Immediate(val) = operand {
            format!("${}", val)
        } else {
            "imm?".to_string()
        }
    } else if operand.is_memory() {
        if let MachineOperand::Memory {
            base,
            offset,
            index,
            scale,
        } = operand
        {
            let base_str = format!("%{}", registers::reg_name(*base));
            let index_str = match index {
                Some(r) => format!("%{}", registers::reg_name(*r)),
                None => "none".to_string(),
            };
            format!("{}({}+{}*{})", offset, base_str, index_str, scale)
        } else {
            "mem?".to_string()
        }
    } else if operand.is_label() {
        if let MachineOperand::Label(id) = operand {
            format!("label({})", id)
        } else {
            "label?".to_string()
        }
    } else {
        format!("{:?}", operand)
    }
}

/// Format a complete [`MachineInstr`] as a diagnostic string showing the
/// opcode and all operands in AT&T-style syntax.
///
/// This is primarily used for diagnostic output when reporting assembler
/// errors or warnings that need to include the failing instruction.
pub fn format_instr_debug(instr: &MachineInstr) -> String {
    let mut parts = Vec::new();
    parts.push(format!("op{}", instr.opcode));
    for operand in &instr.operands {
        parts.push(format_operand_debug(operand));
    }
    parts.join(" ")
}

/// Validate that a [`PhysReg`] used in a register operand is a valid i686
/// register (index in the expected range).
///
/// Returns `true` if the register is a known i686 physical register. This
/// check catches code-generator bugs where an x86-64-only register (R8–R15)
/// or an out-of-range register index might be accidentally used.
///
/// Uses [`PhysReg::index()`] and [`registers::is_gpr()`] along with the
/// i686-specific register constants ([`registers::EAX`] through
/// [`registers::EDI`]) for validation.
pub fn validate_i686_register(reg: PhysReg) -> bool {
    let idx = reg.index();
    // Valid i686 register ranges:
    //   0–7:   32-bit GPRs (EAX–EDI)
    //   8–15:  16-bit sub-registers (AX–DI)
    //   16–23: 8-bit sub-registers (AL–BH)
    //   24–31: x87 FPU (ST0–ST7)
    //   32:    EFLAGS
    idx <= 32
}

/// Validate all register operands in a [`MachineInstr`] are valid i686
/// physical registers.
///
/// Returns the first invalid register found, or `None` if all registers
/// are valid. This catches code-generator bugs where x86-64-only registers
/// (R8–R15, XMM8–XMM15) might appear in i686 instruction output.
pub fn validate_instr_registers(instr: &MachineInstr) -> Option<PhysReg> {
    for operand in &instr.operands {
        match operand {
            MachineOperand::Register(reg) => {
                if !validate_i686_register(*reg) {
                    return Some(*reg);
                }
            }
            MachineOperand::Memory { base, index, .. } => {
                if !validate_i686_register(*base) {
                    return Some(*base);
                }
                if let Some(i) = index {
                    if !validate_i686_register(*i) {
                        return Some(*i);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::{MachineInstr, MachineOperand, PhysReg};
    use crate::common::diagnostics::Span;
    use crate::common::fx_hash::fx_hash_map;

    // -------------------------------------------------------------------
    // SectionKind tests
    // -------------------------------------------------------------------

    #[test]
    fn test_section_kind_names() {
        assert_eq!(SectionKind::Text.name(), ".text");
        assert_eq!(SectionKind::Data.name(), ".data");
        assert_eq!(SectionKind::Rodata.name(), ".rodata");
        assert_eq!(SectionKind::Bss.name(), ".bss");
    }

    #[test]
    fn test_section_kind_properties() {
        assert!(SectionKind::Text.is_executable());
        assert!(!SectionKind::Data.is_executable());
        assert!(!SectionKind::Rodata.is_executable());
        assert!(!SectionKind::Bss.is_executable());

        assert!(!SectionKind::Text.is_writable());
        assert!(SectionKind::Data.is_writable());
        assert!(!SectionKind::Rodata.is_writable());
        assert!(SectionKind::Bss.is_writable());

        assert!(SectionKind::Text.is_loadable());
        assert!(SectionKind::Data.is_loadable());
        assert!(SectionKind::Rodata.is_loadable());
        assert!(!SectionKind::Bss.is_loadable());
    }

    // -------------------------------------------------------------------
    // FixupKind tests
    // -------------------------------------------------------------------

    #[test]
    fn test_fixup_kind_sizes() {
        assert_eq!(FixupKind::Rel8.size(), 1);
        assert_eq!(FixupKind::Rel32.size(), 4);
    }

    // -------------------------------------------------------------------
    // I686Assembler construction tests
    // -------------------------------------------------------------------

    #[test]
    fn test_assembler_new() {
        let asm = I686Assembler::new();
        assert_eq!(asm.current_offset, 0);
        assert!(asm.code.is_empty());
        assert!(asm.data.is_empty());
        assert!(asm.rodata.is_empty());
        assert_eq!(asm.bss_size, 0);
        assert!(asm.relocations.is_empty());
        assert!(asm.symbols.is_empty());
        assert!(!asm.pic_mode);
        assert!(!asm.has_errors());
    }

    #[test]
    fn test_assembler_with_pic() {
        let asm = I686Assembler::with_pic(true);
        assert!(asm.is_pic());

        let asm2 = I686Assembler::with_pic(false);
        assert!(!asm2.is_pic());
    }

    #[test]
    fn test_set_pic() {
        let mut asm = I686Assembler::new();
        assert!(!asm.is_pic());
        asm.set_pic(true);
        assert!(asm.is_pic());
        asm.set_pic(false);
        assert!(!asm.is_pic());
    }

    // -------------------------------------------------------------------
    // Data section helper tests
    // -------------------------------------------------------------------

    #[test]
    fn test_add_data() {
        let mut asm = I686Assembler::new();
        let off1 = asm.add_data(&[1, 2, 3, 4]);
        assert_eq!(off1, 0);
        let off2 = asm.add_data(&[5, 6]);
        assert_eq!(off2, 4);
        assert_eq!(asm.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_add_data_empty() {
        let mut asm = I686Assembler::new();
        let off = asm.add_data(&[]);
        assert_eq!(off, 0);
        assert!(asm.data.is_empty());
    }

    #[test]
    fn test_add_rodata() {
        let mut asm = I686Assembler::new();
        let off = asm.add_rodata(b"Hello\0");
        assert_eq!(off, 0);
        assert_eq!(&asm.rodata, b"Hello\0");
        let off2 = asm.add_rodata(b"World\0");
        assert_eq!(off2, 6);
    }

    #[test]
    fn test_add_bss_basic() {
        let mut asm = I686Assembler::new();
        let off1 = asm.add_bss(16, 4);
        assert_eq!(off1, 0);
        assert_eq!(asm.bss_size, 16);

        let off2 = asm.add_bss(8, 16);
        assert_eq!(off2, 16); // 16 aligned to 16 = 16
        assert_eq!(asm.bss_size, 24); // 16 + 8
    }

    #[test]
    fn test_add_bss_alignment_padding() {
        let mut asm = I686Assembler::new();
        asm.bss_size = 5;
        let off = asm.add_bss(4, 4);
        // 5 aligned up to 4 = 8
        assert_eq!(off, 8);
        assert_eq!(asm.bss_size, 12); // 8 + 4
    }

    #[test]
    fn test_add_bss_zero_align() {
        let mut asm = I686Assembler::new();
        asm.bss_size = 7;
        let off = asm.add_bss(3, 0);
        // align=0 is treated as align=1, so offset=7
        assert_eq!(off, 7);
        assert_eq!(asm.bss_size, 10);
    }

    // -------------------------------------------------------------------
    // Alignment tests
    // -------------------------------------------------------------------

    #[test]
    fn test_align_data() {
        let mut asm = I686Assembler::new();
        asm.add_data(&[1, 2, 3]); // 3 bytes
        asm.align_data(4);
        assert_eq!(asm.data.len(), 4); // padded to 4
        assert_eq!(asm.data[3], 0); // zero padding
    }

    #[test]
    fn test_align_data_already_aligned() {
        let mut asm = I686Assembler::new();
        asm.add_data(&[1, 2, 3, 4]); // 4 bytes, already aligned to 4
        asm.align_data(4);
        assert_eq!(asm.data.len(), 4); // no change
    }

    #[test]
    fn test_align_rodata() {
        let mut asm = I686Assembler::new();
        asm.add_rodata(&[1]); // 1 byte
        asm.align_rodata(8);
        assert_eq!(asm.rodata.len(), 8); // padded to 8
        for i in 1..8 {
            assert_eq!(asm.rodata[i], 0); // zero padding
        }
    }

    #[test]
    fn test_align_code_nop_padding() {
        let mut asm = I686Assembler::new();
        asm.code.extend_from_slice(&[0xCC, 0xCC, 0xCC]); // 3 bytes
        asm.current_offset = 3;
        asm.align_code(4);
        assert_eq!(asm.current_offset, 4);
        assert_eq!(asm.code.len(), 4);
        assert_eq!(asm.code[3], 0x90); // NOP padding
    }

    #[test]
    fn test_align_code_already_aligned() {
        let mut asm = I686Assembler::new();
        asm.code.extend_from_slice(&[0xCC; 16]); // 16 bytes
        asm.current_offset = 16;
        asm.align_code(16);
        assert_eq!(asm.current_offset, 16); // no change
        assert_eq!(asm.code.len(), 16);
    }

    // -------------------------------------------------------------------
    // Symbol and relocation tests
    // -------------------------------------------------------------------

    #[test]
    fn test_add_symbol() {
        let mut asm = I686Assembler::new();
        asm.add_symbol(AsmSymbol {
            name: "main".to_string(),
            offset: 0,
            section: SectionKind::Text,
            is_global: true,
            is_weak: false,
            size: 64,
        });
        assert_eq!(asm.symbols.len(), 1);
        assert_eq!(asm.symbols[0].name, "main");
        assert!(asm.symbols[0].is_global);
    }

    #[test]
    fn test_add_relocation() {
        let mut asm = I686Assembler::new();
        asm.add_relocation(RelocationEntry {
            offset: 0x10,
            reloc_type: I686RelocType::R386Pc32,
            symbol: "printf".to_string(),
            addend: -4,
        });
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].offset, 0x10);
        assert!(asm.relocations[0].reloc_type.is_pc_relative());
        assert_eq!(asm.relocations[0].symbol, "printf");
        assert_eq!(asm.relocations[0].addend, -4);
    }

    // -------------------------------------------------------------------
    // Relocation and symbol struct tests
    // -------------------------------------------------------------------

    #[test]
    fn test_relocation_entry_fields() {
        let entry = RelocationEntry {
            offset: 0x42,
            reloc_type: I686RelocType::R386_32,
            symbol: "global_var".to_string(),
            addend: 0,
        };
        assert_eq!(entry.offset, 0x42);
        assert_eq!(entry.symbol, "global_var");
        assert_eq!(entry.addend, 0);
        assert_eq!(entry.reloc_type.name(), "R_386_32");
    }

    #[test]
    fn test_asm_symbol_fields() {
        let sym = AsmSymbol {
            name: "_start".to_string(),
            offset: 0,
            section: SectionKind::Text,
            is_global: true,
            is_weak: false,
            size: 128,
        };
        assert_eq!(sym.name, "_start");
        assert_eq!(sym.offset, 0);
        assert_eq!(sym.section, SectionKind::Text);
        assert!(sym.is_global);
        assert!(!sym.is_weak);
        assert_eq!(sym.size, 128);
    }

    #[test]
    fn test_asm_symbol_weak() {
        let sym = AsmSymbol {
            name: "weak_fn".to_string(),
            offset: 256,
            section: SectionKind::Text,
            is_global: true,
            is_weak: true,
            size: 32,
        };
        assert!(sym.is_weak);
        assert!(sym.is_global);
    }

    // -------------------------------------------------------------------
    // Reset and accessor tests
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut asm = I686Assembler::new();
        asm.add_data(&[1, 2, 3]);
        asm.add_rodata(&[4, 5]);
        asm.add_bss(16, 4);
        asm.add_symbol(AsmSymbol {
            name: "test".to_string(),
            offset: 0,
            section: SectionKind::Data,
            is_global: false,
            is_weak: false,
            size: 3,
        });
        asm.add_relocation(RelocationEntry {
            offset: 0,
            reloc_type: I686RelocType::R386_32,
            symbol: "x".to_string(),
            addend: 0,
        });
        asm.code.push(0x90);
        asm.current_offset = 1;
        // Inject an error to verify diagnostics are also cleared.
        asm.diagnostics.error(Span::DUMMY, "test error");
        assert!(asm.has_errors());

        asm.reset();

        assert!(asm.code.is_empty());
        assert!(asm.data.is_empty());
        assert!(asm.rodata.is_empty());
        assert_eq!(asm.bss_size, 0);
        assert!(asm.relocations.is_empty());
        assert!(asm.symbols.is_empty());
        assert_eq!(asm.current_offset, 0);
        // Diagnostics should also be cleared after reset.
        assert!(!asm.has_errors());
    }

    #[test]
    fn test_code_offset_accessor() {
        let mut asm = I686Assembler::new();
        assert_eq!(asm.code_offset(), 0);
        asm.current_offset = 42;
        assert_eq!(asm.code_offset(), 42);
    }

    #[test]
    fn test_code_bytes_accessor() {
        let mut asm = I686Assembler::new();
        assert!(asm.code_bytes().is_empty());
        asm.code.extend_from_slice(&[0x55, 0x89, 0xE5]);
        assert_eq!(asm.code_bytes(), &[0x55, 0x89, 0xE5]);
    }

    // -------------------------------------------------------------------
    // Fixup resolution tests
    // -------------------------------------------------------------------

    #[test]
    fn test_resolve_fixup_rel32_forward() {
        let mut asm = I686Assembler::new();
        // Simulate a JMP rel32 (E9 00 00 00 00) at offset 0, targeting label
        // at offset 10. Displacement field at bytes 1..5.
        asm.code = vec![0xE9, 0x00, 0x00, 0x00, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90];
        asm.current_offset = 10;
        asm.label_offsets.insert("target".to_string(), 10);
        asm.pending_fixups.push(PendingFixup {
            code_offset: 1,
            target_label: "target".to_string(),
            kind: FixupKind::Rel32,
        });

        asm.resolve_fixups();

        // displacement = target(10) - (fixup_offset(1) + 4) = 10 - 5 = 5
        let disp = i32::from_le_bytes([asm.code[1], asm.code[2], asm.code[3], asm.code[4]]);
        assert_eq!(disp, 5);
    }

    #[test]
    fn test_resolve_fixup_rel32_backward() {
        let mut asm = I686Assembler::new();
        // Simulate a backward branch: code at offset 10, targeting offset 0.
        asm.code = vec![
            0x90, 0x90, 0x90, 0x90, 0x90, // padding (offsets 0-4)
            0x90, 0x90, 0x90, 0x90, 0x90, // padding (offsets 5-9)
            0xE9, 0x00, 0x00, 0x00, 0x00, // JMP at offset 10, disp at 11
        ];
        asm.current_offset = 15;
        asm.label_offsets.insert("loop_top".to_string(), 0);
        asm.pending_fixups.push(PendingFixup {
            code_offset: 11,
            target_label: "loop_top".to_string(),
            kind: FixupKind::Rel32,
        });

        asm.resolve_fixups();

        // displacement = target(0) - (fixup_offset(11) + 4) = 0 - 15 = -15
        let disp = i32::from_le_bytes([asm.code[11], asm.code[12], asm.code[13], asm.code[14]]);
        assert_eq!(disp, -15);
    }

    #[test]
    fn test_resolve_fixup_rel8() {
        let mut asm = I686Assembler::new();
        // Short JMP (EB 00) at offset 0, targeting offset 4.
        asm.code = vec![0xEB, 0x00, 0x90, 0x90];
        asm.current_offset = 4;
        asm.label_offsets.insert("short_target".to_string(), 4);
        asm.pending_fixups.push(PendingFixup {
            code_offset: 1,
            target_label: "short_target".to_string(),
            kind: FixupKind::Rel8,
        });

        asm.resolve_fixups();

        // displacement = target(4) - (fixup_offset(1) + 1) = 4 - 2 = 2
        assert_eq!(asm.code[1] as i8, 2);
    }

    #[test]
    fn test_resolve_fixup_unresolved_label() {
        let mut asm = I686Assembler::new();
        asm.code = vec![0xE9, 0x00, 0x00, 0x00, 0x00];
        asm.current_offset = 5;
        // No label_offsets entry for "missing_label"
        asm.pending_fixups.push(PendingFixup {
            code_offset: 1,
            target_label: "missing_label".to_string(),
            kind: FixupKind::Rel32,
        });

        assert!(!asm.has_errors());
        asm.resolve_fixups();

        // Displacement should remain zeroed (unresolved).
        let disp = i32::from_le_bytes([asm.code[1], asm.code[2], asm.code[3], asm.code[4]]);
        assert_eq!(disp, 0);
        // An error should have been reported for the unresolved label.
        assert!(asm.has_errors());
    }

    #[test]
    fn test_resolve_fixup_rel8_overflow_warning() {
        let mut asm = I686Assembler::new();
        // Create code buffer with enough distance to overflow rel8.
        // Place a short branch at offset 0, with target at offset 200.
        let mut code = vec![0xEB, 0x00]; // EB cb (short JMP)
        code.extend(vec![0x90; 198]); // 198 NOP padding
        asm.code = code;
        asm.current_offset = 200;
        asm.label_offsets.insert("far_target".to_string(), 200);
        asm.pending_fixups.push(PendingFixup {
            code_offset: 1,
            target_label: "far_target".to_string(),
            kind: FixupKind::Rel8,
        });

        asm.resolve_fixups();

        // displacement = 200 - (1 + 1) = 198, which overflows rel8 range
        // A warning should have been emitted but not an error.
        assert!(!asm.has_errors());
        assert!(asm.diagnostics().warning_count() > 0);
    }

    // -------------------------------------------------------------------
    // find_label_target tests
    // -------------------------------------------------------------------

    #[test]
    fn test_find_label_target_present() {
        let mut map: FxHashMap<u32, String> = fx_hash_map();
        map.insert(5, ".LBB_5".to_string());

        let instr = MachineInstr {
            opcode: 0,
            operands: vec![MachineOperand::Label(5)],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: true,
            is_call: false,
            is_return: false,
        };

        let result = I686Assembler::find_label_target(&instr, &map);
        assert_eq!(result, Some(".LBB_5".to_string()));
    }

    #[test]
    fn test_find_label_target_absent() {
        let map: FxHashMap<u32, String> = fx_hash_map();

        let instr = MachineInstr {
            opcode: 0,
            operands: vec![MachineOperand::Immediate(42)],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };

        let result = I686Assembler::find_label_target(&instr, &map);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_label_target_label_not_in_map() {
        let map: FxHashMap<u32, String> = fx_hash_map();

        let instr = MachineInstr {
            opcode: 0,
            operands: vec![MachineOperand::Label(99)],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: true,
            is_call: false,
            is_return: false,
        };

        let result = I686Assembler::find_label_target(&instr, &map);
        assert_eq!(result, None);
    }

    // -------------------------------------------------------------------
    // RelocationEntry method tests
    // -------------------------------------------------------------------

    #[test]
    fn test_relocation_entry_to_elf_info() {
        let entry = RelocationEntry {
            offset: 0x10,
            reloc_type: I686RelocType::R386Pc32,
            symbol: "puts".to_string(),
            addend: -4,
        };
        // R_386_PC32 has ELF value 2.
        // r_info = (sym_index << 8) | reloc_type
        let info = entry.to_elf_info(5);
        assert_eq!(info, (5 << 8) | 2);
    }

    #[test]
    fn test_relocation_entry_to_elf_info_zero_index() {
        let entry = RelocationEntry {
            offset: 0,
            reloc_type: I686RelocType::R386_32,
            symbol: "x".to_string(),
            addend: 0,
        };
        // R_386_32 has ELF value 1.
        let info = entry.to_elf_info(0);
        assert_eq!(info, 1);
    }

    #[test]
    fn test_relocation_entry_is_pc_relative() {
        let pc_rel = RelocationEntry {
            offset: 0,
            reloc_type: I686RelocType::R386Pc32,
            symbol: "f".to_string(),
            addend: 0,
        };
        assert!(pc_rel.is_pc_relative());

        let abs = RelocationEntry {
            offset: 0,
            reloc_type: I686RelocType::R386_32,
            symbol: "g".to_string(),
            addend: 0,
        };
        assert!(!abs.is_pc_relative());
    }

    #[test]
    fn test_relocation_entry_type_name() {
        let entry = RelocationEntry {
            offset: 0,
            reloc_type: I686RelocType::R386Plt32,
            symbol: "printf".to_string(),
            addend: 0,
        };
        assert_eq!(entry.type_name(), "R_386_PLT32");
    }

    // -------------------------------------------------------------------
    // Validation helper tests
    // -------------------------------------------------------------------

    #[test]
    fn test_validate_i686_register_valid_gprs() {
        // 32-bit GPRs (indices 0–7) should all be valid.
        for idx in 0..=7u16 {
            assert!(
                validate_i686_register(PhysReg(idx)),
                "GPR index {} should be valid",
                idx
            );
        }
    }

    #[test]
    fn test_validate_i686_register_valid_sub_regs() {
        // 16-bit sub-registers (8–15) and 8-bit (16–23) should be valid.
        for idx in 8..=23u16 {
            assert!(
                validate_i686_register(PhysReg(idx)),
                "Sub-register index {} should be valid",
                idx
            );
        }
    }

    #[test]
    fn test_validate_i686_register_valid_fpu() {
        // x87 FPU registers (24–31) should be valid.
        for idx in 24..=31u16 {
            assert!(
                validate_i686_register(PhysReg(idx)),
                "FPU register index {} should be valid",
                idx
            );
        }
    }

    #[test]
    fn test_validate_i686_register_eflags() {
        // EFLAGS at index 32 should be valid.
        assert!(validate_i686_register(PhysReg(32)));
    }

    #[test]
    fn test_validate_i686_register_out_of_range() {
        // Indices beyond 32 should be invalid (e.g. x86-64 R8–R15).
        assert!(!validate_i686_register(PhysReg(33)));
        assert!(!validate_i686_register(PhysReg(64)));
        assert!(!validate_i686_register(PhysReg(255)));
    }

    #[test]
    fn test_validate_instr_registers_all_valid() {
        let instr = MachineInstr {
            opcode: 1,
            operands: vec![
                MachineOperand::Register(PhysReg(0)), // EAX
                MachineOperand::Register(PhysReg(3)), // EBX
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        assert_eq!(validate_instr_registers(&instr), None);
    }

    #[test]
    fn test_validate_instr_registers_invalid_register() {
        let instr = MachineInstr {
            opcode: 1,
            operands: vec![
                MachineOperand::Register(PhysReg(0)),  // EAX — valid
                MachineOperand::Register(PhysReg(64)), // Out of range — invalid
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let result = validate_instr_registers(&instr);
        assert_eq!(result, Some(PhysReg(64)));
    }

    #[test]
    fn test_validate_instr_registers_invalid_memory_base() {
        let instr = MachineInstr {
            opcode: 2,
            operands: vec![MachineOperand::Memory {
                base: PhysReg(100), // Out of range
                offset: 0,
                index: None,
                scale: 1,
            }],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let result = validate_instr_registers(&instr);
        assert_eq!(result, Some(PhysReg(100)));
    }

    #[test]
    fn test_validate_instr_registers_invalid_memory_index() {
        let instr = MachineInstr {
            opcode: 2,
            operands: vec![MachineOperand::Memory {
                base: PhysReg(0), // EAX — valid
                offset: 8,
                index: Some(PhysReg(50)), // Out of range
                scale: 4,
            }],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let result = validate_instr_registers(&instr);
        assert_eq!(result, Some(PhysReg(50)));
    }

    #[test]
    fn test_validate_instr_registers_imm_and_label_ignored() {
        let instr = MachineInstr {
            opcode: 0,
            operands: vec![MachineOperand::Immediate(42), MachineOperand::Label(7)],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        // Non-register operands should be skipped; result is None.
        assert_eq!(validate_instr_registers(&instr), None);
    }

    // -------------------------------------------------------------------
    // format_operand_debug / format_instr_debug tests
    // -------------------------------------------------------------------

    #[test]
    fn test_format_operand_debug_register() {
        // PhysReg(0) = EAX (a GPR), should produce AT&T-style output.
        let operand = MachineOperand::Register(PhysReg(0));
        let s = format_operand_debug(&operand);
        // Should contain the register name and encoding info.
        assert!(s.contains("enc="), "Expected encoding info, got: {}", s);
    }

    #[test]
    fn test_format_operand_debug_immediate() {
        let operand = MachineOperand::Immediate(42);
        let s = format_operand_debug(&operand);
        assert_eq!(s, "$42");
    }

    #[test]
    fn test_format_operand_debug_negative_immediate() {
        let operand = MachineOperand::Immediate(-1);
        let s = format_operand_debug(&operand);
        assert_eq!(s, "$-1");
    }

    #[test]
    fn test_format_operand_debug_label() {
        let operand = MachineOperand::Label(3);
        let s = format_operand_debug(&operand);
        assert_eq!(s, "label(3)");
    }

    #[test]
    fn test_format_operand_debug_memory() {
        let operand = MachineOperand::Memory {
            base: PhysReg(5), // EBP
            offset: -8,
            index: None,
            scale: 1,
        };
        let s = format_operand_debug(&operand);
        // Should contain the displacement and base register info.
        assert!(s.contains("-8"), "Expected displacement, got: {}", s);
    }

    #[test]
    fn test_format_instr_debug_with_operands() {
        let instr = MachineInstr {
            opcode: 42,
            operands: vec![
                MachineOperand::Register(PhysReg(0)),
                MachineOperand::Immediate(10),
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let s = format_instr_debug(&instr);
        assert!(s.starts_with("op42"), "Expected opcode prefix, got: {}", s);
        assert!(s.contains("$10"), "Expected immediate operand, got: {}", s);
    }

    #[test]
    fn test_format_instr_debug_no_operands() {
        let instr = MachineInstr {
            opcode: 99,
            operands: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let s = format_instr_debug(&instr);
        assert_eq!(s, "op99");
    }
}
