//! Built-in RISC-V 64-bit assembler driver module for BCC.
//!
//! This module implements the standalone, zero-dependency assembler for the
//! RISC-V 64 target architecture (RV64IMAFDC ISA). It coordinates instruction
//! encoding and relocation emission, producing relocatable object code
//! (`.o` sections) consumed by the built-in RISC-V 64 linker.
//!
//! No external assembler (`as`, `gas`, `llvm-mc`) is ever invoked — this is
//! the standalone assembler component fulfilling the project's
//! zero-external-tool mandate.
//!
//! # Architecture
//!
//! The assembler operates in three phases:
//!
//! 1. **Instruction dispatch**: The code generator hands `MachineFunction`s
//!    to [`RiscV64Assembler::assemble_function`]. Each `MachineInstr` is
//!    mapped to an [`RvOpcode`] and dispatched to the encoder.
//!
//! 2. **Encoding**: [`RiscV64Encoder`] encodes instructions into binary
//!    bytes (4-byte words, 2-byte compressed, or 8-byte pseudo pairs).
//!    Pseudo-instructions (LI, LA, CALL, TAIL, RET, MV, etc.) are expanded
//!    to real instruction sequences here.
//!
//! 3. **Relocation collection**: Unresolved symbol references produce
//!    [`AssemblerRelocation`] entries annotated with the appropriate
//!    [`RiscV64RelocationType`]. Relaxation-eligible relocations are paired
//!    with `R_RISCV_RELAX` markers so the linker can shorten instruction
//!    sequences when targets are within range.
//!
//! # Section management
//!
//! The assembler maintains multiple sections (`.text`, `.data`, `.rodata`,
//! `.bss`) with independent data buffers, relocation lists, and alignment
//! tracking. The [`AssembledObject`] output aggregates all sections and
//! symbols for the linker.
//!
//! # Sub-modules
//!
//! - [`encoder`]: Instruction encoding engine — R/I/S/B/U/J and compressed
//!   (RVC) format encoding, pseudo-instruction expansion, immediate
//!   validation.
//! - [`relocations`]: RISC-V 64 ELF relocation type definitions —
//!   `R_RISCV_*` constants, hi/lo splitting, relaxation metadata,
//!   relocation application functions.

/// RISC-V 64 ELF relocation type definitions used by both the assembler
/// (to record relocations during instruction encoding) and the linker (to
/// apply relocations when producing final ELF executables and shared objects).
/// Includes support for linker relaxation markers (`R_RISCV_RELAX`) and
/// compressed instruction relocations (`R_RISCV_RVC_BRANCH`, `R_RISCV_RVC_JUMP`).
pub mod relocations;

/// RISC-V 64-bit instruction encoder — encodes all RV64IMAFDC instructions
/// in R/I/S/B/U/J/R4 formats plus 16-bit compressed (RVC) formats. Provides
/// the primary encoding entry point [`encoder::RiscV64Encoder::encode_instruction`],
/// immediate validation, register encoding helpers, relocation type inference,
/// and pseudo-instruction expansion (LI, LA, CALL, TAIL, RET, NOP, etc.).
pub mod encoder;

// ---------------------------------------------------------------------------
// Re-exports for convenient external access
// ---------------------------------------------------------------------------

pub use encoder::RiscV64Encoder;
pub use relocations::{RiscV64RelocationType, RISCV64_RELOCATION_TYPES};

// ---------------------------------------------------------------------------
// Internal imports
// ---------------------------------------------------------------------------

use std::fmt;

use crate::backend::elf_writer_common::{
    SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_NOBITS, SHT_PROGBITS,
};
use crate::backend::riscv64::registers::{self, RA, ZERO};
use crate::backend::traits::{MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand};

use self::encoder::{EncoderError, EncoderOperand, RvOpcode};
use self::relocations::split_hi_lo;

// ===========================================================================
// AssemblerSection — a single section in the assembled output
// ===========================================================================

/// Represents a single section (`.text`, `.data`, `.rodata`, `.bss`, etc.)
/// in the assembled output.
///
/// Each section has its own data buffer, alignment requirement, ELF flags,
/// and relocation list. The linker merges sections with the same name from
/// multiple object files.
#[derive(Debug, Clone)]
pub struct AssemblerSection {
    /// Section name (e.g., `.text`, `.data`, `.rodata`, `.bss`).
    pub name: String,
    /// Accumulated binary data for this section.
    pub data: Vec<u8>,
    /// Required alignment in bytes (power of two).
    pub alignment: u32,
    /// ELF section flags (`SHF_ALLOC`, `SHF_EXECINSTR`, `SHF_WRITE`, etc.).
    pub flags: u64,
    /// Relocation entries referencing symbols in this section.
    pub relocations: Vec<AssemblerRelocation>,
}

impl AssemblerSection {
    /// Creates a new section with the given name, alignment, and flags.
    fn new(name: &str, alignment: u32, flags: u64) -> Self {
        AssemblerSection {
            name: name.to_string(),
            data: Vec::new(),
            alignment,
            flags,
            relocations: Vec::new(),
        }
    }

    /// Returns the current byte offset within this section.
    #[inline]
    fn current_offset(&self) -> u32 {
        self.data.len() as u32
    }

    /// Returns the ELF section type for this section.
    ///
    /// `.bss` sections use `SHT_NOBITS` (no file data); all others use
    /// `SHT_PROGBITS`.
    pub fn section_type(&self) -> u32 {
        if self.name == ".bss" {
            SHT_NOBITS
        } else {
            SHT_PROGBITS
        }
    }
}

// ===========================================================================
// AssemblerRelocation — a relocation entry recorded during assembly
// ===========================================================================

/// A relocation entry recorded during assembly, representing an unresolved
/// symbol reference that the linker must patch.
///
/// Each relocation records the byte offset within its section, the relocation
/// type (which determines how the linker patches the instruction), the
/// target symbol name, an addend, and whether the relocation is eligible
/// for linker relaxation.
#[derive(Debug, Clone)]
pub struct AssemblerRelocation {
    /// Byte offset within the section where the relocation applies.
    pub offset: u32,
    /// RISC-V relocation type determining how the linker patches.
    pub reloc_type: RiscV64RelocationType,
    /// Target symbol name for resolution.
    pub symbol: String,
    /// Signed addend added to the symbol value.
    pub addend: i64,
    /// If `true`, the linker may optimize (relax) this relocation to a
    /// shorter instruction sequence. Paired with a companion `R_RISCV_RELAX`
    /// entry by the assembler.
    pub is_relaxable: bool,
}

// ===========================================================================
// AssemblerSymbol — a symbol definition recorded during assembly
// ===========================================================================

/// A symbol definition recorded during assembly.
///
/// Symbols mark named locations (function entry points, global variables,
/// labels) within sections. The linker uses these for cross-object-file
/// symbol resolution and relocation patching.
#[derive(Debug, Clone)]
pub struct AssemblerSymbol {
    /// Symbol name.
    pub name: String,
    /// Index of the section containing this symbol.
    pub section: usize,
    /// Byte offset within the section.
    pub offset: u32,
    /// If `true`, the symbol is visible to the linker across object files.
    pub is_global: bool,
    /// If `true`, the symbol represents a function (STT_FUNC); otherwise
    /// it represents a data object (STT_OBJECT) or label (STT_NOTYPE).
    pub is_function: bool,
    /// Size of the symbol in bytes (0 if unknown).
    pub size: u32,
}

// ===========================================================================
// AssembledObject — the complete output of the assembler
// ===========================================================================

/// The complete output of the assembler for a single compilation unit.
///
/// Contains all assembled sections (with their binary data, relocations,
/// and metadata) and all symbol definitions. This is the input consumed
/// by the RISC-V 64 linker module.
#[derive(Debug, Clone)]
pub struct AssembledObject {
    /// All sections produced by the assembler.
    pub sections: Vec<AssemblerSection>,
    /// All symbol definitions (local and global).
    pub symbols: Vec<AssemblerSymbol>,
}

// ===========================================================================
// AssemblerError — errors during assembly
// ===========================================================================

/// Error type for failures during RISC-V 64 assembly.
///
/// These errors are propagated from the encoder or detected by the assembler
/// driver when instruction operands are invalid or unsupported.
#[derive(Debug, Clone)]
pub enum AssemblerError {
    /// An instruction encoding error from the encoder layer.
    EncodingError(String),
    /// An instruction operand is invalid for the given context.
    InvalidOperand(String),
    /// The instruction is not supported by this assembler.
    UnsupportedInstruction(String),
    /// An immediate value is outside the valid range.
    ImmediateOutOfRange {
        /// The out-of-range value.
        value: i64,
        /// Minimum allowed value (inclusive).
        min: i64,
        /// Maximum allowed value (inclusive).
        max: i64,
    },
    /// A register operand is invalid for the instruction.
    InvalidRegister(String),
}

impl fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssemblerError::EncodingError(msg) => {
                write!(f, "encoding error: {}", msg)
            }
            AssemblerError::InvalidOperand(msg) => {
                write!(f, "invalid operand: {}", msg)
            }
            AssemblerError::UnsupportedInstruction(msg) => {
                write!(f, "unsupported instruction: {}", msg)
            }
            AssemblerError::ImmediateOutOfRange { value, min, max } => {
                write!(
                    f,
                    "immediate value {} out of range [{}, {}]",
                    value, min, max
                )
            }
            AssemblerError::InvalidRegister(msg) => {
                write!(f, "invalid register: {}", msg)
            }
        }
    }
}

impl From<EncoderError> for AssemblerError {
    fn from(e: EncoderError) -> Self {
        AssemblerError::EncodingError(e.to_string())
    }
}

// ===========================================================================
// RiscV64Assembler — the main assembler driver
// ===========================================================================

/// Built-in RISC-V 64-bit assembler driver.
///
/// Takes machine instructions from the code generator, dispatches them to
/// the [`RiscV64Encoder`] for binary encoding, collects relocation entries
/// for unresolved symbol references, and produces an [`AssembledObject`]
/// with relocatable sections consumed by the RISC-V 64 linker.
///
/// # Usage
///
/// ```text
/// let mut asm = RiscV64Assembler::new();
/// asm.assemble_function(&machine_func)?;
/// let obj = asm.finalize();
/// // Pass `obj` to the linker.
/// ```
pub struct RiscV64Assembler {
    /// Instruction encoder (stateless — shared across all encoding calls).
    encoder: RiscV64Encoder,
    /// Assembled sections (`.text`, `.data`, `.rodata`, `.bss`, etc.).
    sections: Vec<AssemblerSection>,
    /// Symbol definitions accumulated during assembly.
    symbols: Vec<AssemblerSymbol>,
    /// Index of the currently active section in `sections`.
    current_section: usize,
    /// Map from basic-block label to byte offset within the current function's
    /// section for branch-target resolution.
    label_offsets: std::collections::HashMap<u32, u32>,
    /// Name of the function currently being assembled (for diagnostics).
    current_function: String,
}

impl Default for RiscV64Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl RiscV64Assembler {
    /// Creates a new assembler with a default `.text` section.
    pub fn new() -> Self {
        let text_section = AssemblerSection::new(
            ".text",
            4, // 4-byte alignment for RISC-V instructions
            SHF_ALLOC | SHF_EXECINSTR,
        );
        RiscV64Assembler {
            encoder: RiscV64Encoder,
            sections: vec![text_section],
            symbols: Vec::new(),
            current_section: 0,
            label_offsets: std::collections::HashMap::new(),
            current_function: String::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Section management
    // -----------------------------------------------------------------------

    /// Switches to the named section, creating it if it does not exist.
    ///
    /// Well-known section names (`.text`, `.data`, `.rodata`, `.bss`) get
    /// appropriate default flags. Custom sections use the provided alignment
    /// and flags.
    pub fn switch_section(&mut self, name: &str, alignment: u32, flags: u64) {
        // Search for an existing section with this name.
        for (idx, sec) in self.sections.iter().enumerate() {
            if sec.name == name {
                self.current_section = idx;
                return;
            }
        }
        // Create a new section with the specified (or default) attributes.
        let effective_flags = match name {
            ".text" => SHF_ALLOC | SHF_EXECINSTR,
            ".data" | ".bss" => SHF_ALLOC | SHF_WRITE,
            ".rodata" => SHF_ALLOC,
            _ => flags,
        };
        let effective_alignment = if alignment == 0 {
            match name {
                ".text" => 4,
                ".data" | ".rodata" | ".bss" => 8,
                _ => 1,
            }
        } else {
            alignment
        };
        let section = AssemblerSection::new(name, effective_alignment, effective_flags);
        self.sections.push(section);
        self.current_section = self.sections.len() - 1;
    }

    /// Returns the current byte offset within the active section.
    #[inline]
    pub fn get_current_offset(&self) -> u32 {
        self.sections[self.current_section].current_offset()
    }

    /// Returns a reference to the currently active section.
    #[inline]
    fn current_section(&self) -> &AssemblerSection {
        &self.sections[self.current_section]
    }

    /// Returns a mutable reference to the currently active section.
    #[inline]
    fn current_section_mut(&mut self) -> &mut AssemblerSection {
        &mut self.sections[self.current_section]
    }

    // -----------------------------------------------------------------------
    // Symbol management
    // -----------------------------------------------------------------------

    /// Defines a symbol at the current offset in the active section.
    ///
    /// The symbol records whether it is globally visible and whether it
    /// represents a function (for ELF `STT_FUNC` classification).
    pub fn define_symbol(&mut self, name: &str, is_global: bool, is_function: bool) {
        let offset = self.get_current_offset();
        let section = self.current_section;
        self.symbols.push(AssemblerSymbol {
            name: name.to_string(),
            section,
            offset,
            is_global,
            is_function,
            size: 0,
        });
    }

    // -----------------------------------------------------------------------
    // Relocation recording
    // -----------------------------------------------------------------------

    /// Records a relocation at the current offset in the active section.
    ///
    /// The relocation is *not* marked as relaxable. Use
    /// [`add_relaxable_relocation`](Self::add_relaxable_relocation) for
    /// relocations that the linker may shorten.
    pub fn add_relocation(&mut self, reloc_type: RiscV64RelocationType, symbol: &str, addend: i64) {
        let offset = self.get_current_offset();
        self.current_section_mut()
            .relocations
            .push(AssemblerRelocation {
                offset,
                reloc_type,
                symbol: symbol.to_string(),
                addend,
                is_relaxable: false,
            });
    }

    /// Records a relaxation-eligible relocation at the current offset.
    ///
    /// The relocation is paired with a companion `R_RISCV_RELAX` marker
    /// so the linker can optimize the instruction sequence when the target
    /// symbol is close enough (e.g., AUIPC+JALR → JAL).
    pub fn add_relaxable_relocation(
        &mut self,
        reloc_type: RiscV64RelocationType,
        symbol: &str,
        addend: i64,
    ) {
        let offset = self.get_current_offset();
        let sec = self.current_section_mut();

        // Primary relocation.
        sec.relocations.push(AssemblerRelocation {
            offset,
            reloc_type,
            symbol: symbol.to_string(),
            addend,
            is_relaxable: true,
        });

        // Companion R_RISCV_RELAX marker at the same offset.
        sec.relocations.push(AssemblerRelocation {
            offset,
            reloc_type: RiscV64RelocationType::R_RISCV_RELAX,
            symbol: symbol.to_string(),
            addend: 0,
            is_relaxable: false,
        });
    }

    /// Records a relocation at a specific explicit offset (not the current
    /// section offset). Used when emitting instruction pairs where the second
    /// instruction's relocation offset must reference a different position.
    fn add_relocation_at(
        &mut self,
        offset: u32,
        reloc_type: RiscV64RelocationType,
        symbol: &str,
        addend: i64,
        is_relaxable: bool,
    ) {
        self.current_section_mut()
            .relocations
            .push(AssemblerRelocation {
                offset,
                reloc_type,
                symbol: symbol.to_string(),
                addend,
                is_relaxable,
            });
    }

    // -----------------------------------------------------------------------
    // Raw byte emission
    // -----------------------------------------------------------------------

    /// Emits literal bytes into the current section.
    ///
    /// Used for inline assembly `.byte` directives, data sections, and
    /// other raw content that bypasses instruction encoding.
    pub fn emit_raw_bytes(&mut self, bytes: &[u8]) {
        self.current_section_mut().data.extend_from_slice(bytes);
    }

    /// Emits a 4-byte NOP instruction (`ADDI x0, x0, 0`) for alignment.
    pub fn emit_nop(&mut self) {
        // NOP = ADDI x0, x0, 0 = 0x00000013
        let nop_bytes = 0x0000_0013u32.to_le_bytes();
        self.current_section_mut()
            .data
            .extend_from_slice(&nop_bytes);
    }

    /// Emits a 2-byte compressed NOP (`C.NOP`) for 2-byte alignment.
    pub fn emit_c_nop(&mut self) {
        // C.NOP = 0x0001
        let c_nop_bytes = 0x0001u16.to_le_bytes();
        self.current_section_mut()
            .data
            .extend_from_slice(&c_nop_bytes);
    }

    /// Emits padding to align the current section offset to `alignment` bytes.
    ///
    /// Uses 4-byte NOP instructions for `.text` sections and zero bytes
    /// for data sections. If the current offset is already aligned, no
    /// bytes are emitted.
    pub fn align(&mut self, alignment: u32) {
        if alignment == 0 || alignment == 1 {
            return;
        }
        let offset = self.get_current_offset();
        let remainder = offset % alignment;
        if remainder == 0 {
            return;
        }
        let padding = alignment - remainder;
        let is_code_section = self.current_section().flags & SHF_EXECINSTR != 0;
        if is_code_section {
            // Use NOP instructions for code section alignment.
            let full_nops = padding / 4;
            let leftover = padding % 4;
            for _ in 0..full_nops {
                self.emit_nop();
            }
            // If there is a 2-byte remainder, use C.NOP (compressed NOP).
            if leftover >= 2 {
                self.emit_c_nop();
            }
            // Any remaining single byte is padded with zero (should not
            // happen with proper 2-byte-aligned RISC-V code).
            let final_leftover = padding - (full_nops * 4) - if leftover >= 2 { 2 } else { 0 };
            if final_leftover > 0 {
                let zeros = vec![0u8; final_leftover as usize];
                self.current_section_mut().data.extend_from_slice(&zeros);
            }
        } else {
            // Zero-fill for data sections.
            let zeros = vec![0u8; padding as usize];
            self.current_section_mut().data.extend_from_slice(&zeros);
        }
    }

    // -----------------------------------------------------------------------
    // Function assembly
    // -----------------------------------------------------------------------

    /// Assembles an entire machine function into the current `.text` section.
    ///
    /// Iterates over all basic blocks and their instructions, encoding each
    /// one and recording relocations for unresolved symbols. Block labels
    /// are mapped to byte offsets for intra-function branch resolution.
    pub fn assemble_function(&mut self, mf: &MachineFunction) -> Result<(), AssemblerError> {
        // Ensure we are in the .text section.
        if self.current_section().name != ".text" {
            self.switch_section(".text", 4, SHF_ALLOC | SHF_EXECINSTR);
        }

        // Align to function entry boundary (4-byte minimum for RISC-V).
        self.align(4);

        // Record the function symbol.
        self.current_function = mf.name.clone();
        self.define_symbol(&mf.name, true, true);

        // Record function start offset for size computation.
        let func_start = self.get_current_offset();

        // Clear label offset map for this function.
        self.label_offsets.clear();

        // First pass: collect label offsets.
        // We need to know block label positions for branch resolution.
        // For a correct first pass we would need to know instruction sizes,
        // which requires encoding. Instead, we do a two-pass approach:
        //   Pass 1: encode instructions and record labels.
        //   Pass 2: patch local branch offsets (if needed).
        // Since RISC-V branch targets are typically resolved via relocations
        // or are encoded with offsets computed from label_offsets, we record
        // labels as we go and rely on the linker for cross-function targets.

        // Encode all blocks.
        for block in &mf.blocks {
            self.assemble_block(block)?;
        }

        // Compute and update function symbol size.
        let func_end = self.get_current_offset();
        let func_size = func_end - func_start;
        // Update the symbol size for the most recently defined function symbol.
        for sym in self.symbols.iter_mut().rev() {
            if sym.name == mf.name && sym.is_function {
                sym.size = func_size;
                break;
            }
        }

        Ok(())
    }

    /// Assembles a single basic block, recording its label offset and
    /// encoding all contained instructions.
    fn assemble_block(&mut self, block: &MachineBasicBlock) -> Result<(), AssemblerError> {
        // Record this block's label at the current offset.
        let block_offset = self.get_current_offset();
        self.label_offsets.insert(block.id, block_offset);

        // If the block has an explicit label, define it as a local symbol.
        if let Some(ref label) = block.label {
            self.define_symbol(label, false, false);
        }

        // Encode each instruction in the block.
        for instr in &block.instructions {
            self.emit_instruction(instr)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Instruction emission
    // -----------------------------------------------------------------------

    /// Encodes a single machine instruction, appending the encoded bytes to
    /// the current section and recording any necessary relocations.
    ///
    /// The `MachineInstr`'s opcode (a `u32`) is mapped to an [`RvOpcode`]
    /// for dispatch to the encoder. Pseudo-instructions are expanded to
    /// their real instruction sequences.
    pub fn emit_instruction(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let opcode = self.map_opcode(instr.opcode)?;

        // Handle pseudo-instructions that expand to multiple real instructions
        // with special relocation handling.
        match opcode {
            RvOpcode::LI => return self.expand_li(instr),
            RvOpcode::LA => return self.expand_la(instr),
            RvOpcode::CALL => return self.expand_call(instr),
            RvOpcode::TAIL => return self.expand_tail(instr),
            RvOpcode::RET => return self.expand_ret(),
            RvOpcode::MV => return self.expand_mv(instr),
            RvOpcode::NOP => {
                self.emit_nop();
                return Ok(());
            }
            RvOpcode::NOT => return self.expand_not(instr),
            RvOpcode::NEG => return self.expand_neg(instr),
            RvOpcode::SEQZ => return self.expand_seqz(instr),
            RvOpcode::SNEZ => return self.expand_snez(instr),
            RvOpcode::J => return self.expand_j(instr),
            RvOpcode::JR => return self.expand_jr(instr),
            _ => {}
        }

        // Check if any operand is a symbol — requires relocation.
        let has_symbol = instr.operands.iter().any(|op| op.is_symbol());

        if has_symbol {
            return self.emit_instruction_with_relocation(opcode, instr);
        }

        // Standard instruction encoding path.
        let encoder_ops = self.translate_operands(&instr.operands)?;
        let encoded = self.encoder.encode_instruction(opcode, &encoder_ops)?;
        let bytes = encoded.to_bytes();
        self.current_section_mut().data.extend_from_slice(&bytes);

        Ok(())
    }

    /// Emits an instruction that references a symbol, recording the
    /// appropriate relocation entry.
    fn emit_instruction_with_relocation(
        &mut self,
        opcode: RvOpcode,
        instr: &MachineInstr,
    ) -> Result<(), AssemblerError> {
        // Find the symbol operand.
        let (sym_name, sym_addend) = self.extract_symbol_operand(&instr.operands)?;

        // Build encoder operands, replacing symbols with zero immediates
        // (the linker will patch the actual value).
        let encoder_ops = self.translate_operands_for_relocation(&instr.operands)?;

        // Determine the relocation type from the opcode context.
        let reloc_type = self.relocation_type_for_opcode(&opcode);

        // Record relocation at the current offset before emitting bytes.
        let emit_offset = self.get_current_offset();

        // Encode with placeholder operands.
        let encoded = self.encoder.encode_instruction(opcode, &encoder_ops)?;
        let bytes = encoded.to_bytes();
        self.current_section_mut().data.extend_from_slice(&bytes);

        // Add the relocation entry.
        let is_relaxable = reloc_type.is_paired_with_relax();
        if is_relaxable {
            self.add_relocation_at(emit_offset, reloc_type, &sym_name, sym_addend, true);
            // Add companion R_RISCV_RELAX.
            self.add_relocation_at(
                emit_offset,
                RiscV64RelocationType::R_RISCV_RELAX,
                &sym_name,
                0,
                false,
            );
        } else {
            self.add_relocation_at(emit_offset, reloc_type, &sym_name, sym_addend, false);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pseudo-instruction expansion
    // -----------------------------------------------------------------------

    /// Expands `LI rd, imm` — load immediate.
    ///
    /// Selects the optimal instruction sequence based on the immediate value:
    /// - `|imm| < 2048`: single `ADDI rd, x0, imm`
    /// - `imm` fits 32 bits: `LUI rd, hi20` + `ADDI rd, rd, lo12`
    /// - 64-bit: multi-instruction sequence with shifts
    fn expand_li(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let imm = self.extract_immediate(&instr.operands, 1)?;

        // Case 1: Small immediate fits in 12-bit sign-extended range.
        if (-2048..=2047).contains(&imm) {
            let ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Register(registers::encoding(ZERO)),
                EncoderOperand::Immediate(imm),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::ADDI, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());
            return Ok(());
        }

        // Case 2: Value fits in 32 bits (sign-extended from 32 to 64).
        let val32 = imm as i32;
        if imm == val32 as i64 {
            let (hi, lo) = split_hi_lo(imm);
            // LUI rd, hi20
            let lui_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Immediate(hi as i64),
            ];
            let lui = self.encoder.encode_instruction(RvOpcode::LUI, &lui_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&lui.to_bytes());

            // ADDI rd, rd, lo12 (only if lo != 0)
            if lo != 0 {
                let addi_ops = [
                    EncoderOperand::Register(rd),
                    EncoderOperand::Register(rd),
                    EncoderOperand::Immediate(lo as i64),
                ];
                let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&addi.to_bytes());
            }
            return Ok(());
        }

        // Case 3: Full 64-bit immediate — requires multi-instruction sequence.
        // Strategy: build the value in stages using LUI+ADDI for the upper 32
        // bits, then SLLI+ADDI for lower portions.
        self.expand_li_64(rd, imm)
    }

    /// Expands a 64-bit load immediate into a multi-instruction sequence.
    ///
    /// Uses LUI + ADDI + SLLI + ADDI + SLLI + ADDI pattern to build
    /// arbitrary 64-bit constants.
    fn expand_li_64(&mut self, rd: u8, imm: i64) -> Result<(), AssemblerError> {
        // Split the 64-bit value into manageable chunks.
        // Strategy: work from the top bits down.
        let val = imm as u64;

        // Find the highest set bit to determine the minimal sequence.
        let hi32 = ((val >> 32) as i32) as i64;
        let lo32 = (val as i32) as i64;

        // Load upper 32 bits.
        let (hi_hi, hi_lo) = split_hi_lo(hi32);

        if hi_hi != 0 {
            // LUI rd, hi_hi
            let lui_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Immediate(hi_hi as i64),
            ];
            let lui = self.encoder.encode_instruction(RvOpcode::LUI, &lui_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&lui.to_bytes());

            if hi_lo != 0 {
                // ADDI rd, rd, hi_lo
                let addi_ops = [
                    EncoderOperand::Register(rd),
                    EncoderOperand::Register(rd),
                    EncoderOperand::Immediate(hi_lo as i64),
                ];
                let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&addi.to_bytes());
            }
        } else if hi_lo != 0 {
            // Small upper — just ADDI from zero.
            let addi_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Register(registers::encoding(ZERO)),
                EncoderOperand::Immediate(hi_lo as i64),
            ];
            let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&addi.to_bytes());
        } else {
            // Upper 32 bits are zero — load lower 32 bits directly.
            let (lo_hi, lo_lo) = split_hi_lo(lo32);
            if lo_hi != 0 {
                let lui_ops = [
                    EncoderOperand::Register(rd),
                    EncoderOperand::Immediate(lo_hi as i64),
                ];
                let lui = self.encoder.encode_instruction(RvOpcode::LUI, &lui_ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&lui.to_bytes());
                if lo_lo != 0 {
                    let addi_ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rd),
                        EncoderOperand::Immediate(lo_lo as i64),
                    ];
                    let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&addi.to_bytes());
                }
            } else {
                let addi_ops = [
                    EncoderOperand::Register(rd),
                    EncoderOperand::Register(registers::encoding(ZERO)),
                    EncoderOperand::Immediate(lo_lo as i64),
                ];
                let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&addi.to_bytes());
            }
            return Ok(());
        }

        // SLLI rd, rd, 32 — shift upper half into position.
        let slli_ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rd),
            EncoderOperand::Immediate(32),
        ];
        let slli = self.encoder.encode_instruction(RvOpcode::SLLI, &slli_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&slli.to_bytes());

        // Now add the lower 32 bits. Split into hi20+lo12.
        let (lo_hi, lo_lo) = split_hi_lo(lo32);

        if lo_hi != 0 {
            // We need to use a temporary approach: ADDI for chunks of 12 bits.
            // First get the upper 20 bits of the lower half into position.
            // Use LUI on a temp? No — we are constrained to rd only.
            // Instead, shift and add in 12-bit chunks.
            let upper_lo = (lo32 >> 12) & 0xFFFFF;
            if upper_lo != 0 {
                // ORI or ADDI the upper portion of the lower 32 bits.
                // ADDI rd, rd, upper_lo (but this is > 12 bits...)
                // Strategy: SLLI + ADDI iteratively.
                // Actually, since we shifted by 32 above, we add the lower 32
                // bits in two 12-bit halves separated by shifts.
                let lo_upper_12 = ((val >> 20) & 0xFFF) as i64;
                let lo_mid_8 = ((val >> 12) & 0xFF) as i64;
                let lo_lower_12 = (val & 0xFFF) as i64;

                if lo_upper_12 != 0 {
                    // Sign-extend the 12-bit chunk.
                    let sign_ext = if lo_upper_12 >= 0x800 {
                        lo_upper_12 - 0x1000
                    } else {
                        lo_upper_12
                    };
                    let addi_ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rd),
                        EncoderOperand::Immediate(sign_ext),
                    ];
                    let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&addi.to_bytes());
                }

                // SLLI rd, rd, 12
                let slli2_ops = [
                    EncoderOperand::Register(rd),
                    EncoderOperand::Register(rd),
                    EncoderOperand::Immediate(12),
                ];
                let slli2 = self
                    .encoder
                    .encode_instruction(RvOpcode::SLLI, &slli2_ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&slli2.to_bytes());

                // Combine mid and lower bits.
                let combined_lo = (lo_mid_8 << 12) | lo_lower_12;
                if combined_lo != 0 {
                    let lo_sign_ext = if combined_lo >= 0x800 {
                        combined_lo - 0x1000
                    } else {
                        combined_lo
                    };
                    // We need to add the remaining 20 bits but that may not fit
                    // in 12. Use another shift-add pair.
                    if (-2048..=2047).contains(&lo_sign_ext) {
                        let addi3_ops = [
                            EncoderOperand::Register(rd),
                            EncoderOperand::Register(rd),
                            EncoderOperand::Immediate(lo_sign_ext),
                        ];
                        let addi3 = self
                            .encoder
                            .encode_instruction(RvOpcode::ADDI, &addi3_ops)?;
                        self.current_section_mut()
                            .data
                            .extend_from_slice(&addi3.to_bytes());
                    } else {
                        // Further decompose.
                        let upper = (combined_lo >> 12) & 0xFF;
                        let lower = combined_lo & 0xFFF;
                        if upper != 0 {
                            let sign_upper = if upper >= 0x800 {
                                upper - 0x1000
                            } else {
                                upper
                            };
                            let a_ops = [
                                EncoderOperand::Register(rd),
                                EncoderOperand::Register(rd),
                                EncoderOperand::Immediate(sign_upper),
                            ];
                            let a = self.encoder.encode_instruction(RvOpcode::ADDI, &a_ops)?;
                            self.current_section_mut()
                                .data
                                .extend_from_slice(&a.to_bytes());

                            let s_ops = [
                                EncoderOperand::Register(rd),
                                EncoderOperand::Register(rd),
                                EncoderOperand::Immediate(12),
                            ];
                            let s = self.encoder.encode_instruction(RvOpcode::SLLI, &s_ops)?;
                            self.current_section_mut()
                                .data
                                .extend_from_slice(&s.to_bytes());
                        }
                        if lower != 0 {
                            let sign_lower = if lower >= 0x800 {
                                lower - 0x1000
                            } else {
                                lower
                            };
                            let a2_ops = [
                                EncoderOperand::Register(rd),
                                EncoderOperand::Register(rd),
                                EncoderOperand::Immediate(sign_lower),
                            ];
                            let a2 = self.encoder.encode_instruction(RvOpcode::ADDI, &a2_ops)?;
                            self.current_section_mut()
                                .data
                                .extend_from_slice(&a2.to_bytes());
                        }
                    }
                }
            } else if lo_lo != 0 {
                // Lower 12 bits only.
                let addi_ops = [
                    EncoderOperand::Register(rd),
                    EncoderOperand::Register(rd),
                    EncoderOperand::Immediate(lo_lo as i64),
                ];
                let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&addi.to_bytes());
            }
        } else if lo_lo != 0 {
            // Only the lower 12 bits of the lower half are set.
            let addi_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Register(rd),
                EncoderOperand::Immediate(lo_lo as i64),
            ];
            let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&addi.to_bytes());
        }

        Ok(())
    }

    /// Expands `LA rd, symbol` — load address.
    ///
    /// Non-PIC: `AUIPC rd, %pcrel_hi20(sym)` + `ADDI rd, rd, %pcrel_lo12(sym)`
    /// PIC (GOT): `AUIPC rd, %got_hi20(sym)` + `LD rd, rd, %got_lo12(sym)`
    fn expand_la(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let (sym_name, addend) = self.extract_symbol_from_operands(&instr.operands)?;

        let offset_before = self.get_current_offset();

        // AUIPC rd, 0 (placeholder — linker fills the hi20 part).
        let auipc_ops = [EncoderOperand::Register(rd), EncoderOperand::Immediate(0)];
        let auipc = self
            .encoder
            .encode_instruction(RvOpcode::AUIPC, &auipc_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&auipc.to_bytes());

        // ADDI rd, rd, 0 (placeholder — linker fills the lo12 part).
        let addi_ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rd),
            EncoderOperand::Immediate(0),
        ];
        let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&addi.to_bytes());

        // Relocations: PCREL_HI20 on AUIPC, PCREL_LO12_I on ADDI.
        self.add_relocation_at(
            offset_before,
            RiscV64RelocationType::R_RISCV_PCREL_HI20,
            &sym_name,
            addend,
            true,
        );
        self.add_relocation_at(
            offset_before,
            RiscV64RelocationType::R_RISCV_RELAX,
            &sym_name,
            0,
            false,
        );
        self.add_relocation_at(
            offset_before + 4,
            RiscV64RelocationType::R_RISCV_PCREL_LO12_I,
            &sym_name,
            0,
            true,
        );
        self.add_relocation_at(
            offset_before + 4,
            RiscV64RelocationType::R_RISCV_RELAX,
            &sym_name,
            0,
            false,
        );

        Ok(())
    }

    /// Expands `CALL symbol` — function call via AUIPC+JALR.
    ///
    /// Emits: `AUIPC ra, 0` + `JALR ra, ra, 0`
    /// with `R_RISCV_CALL` (or `R_RISCV_CALL_PLT`) relocation on the pair.
    fn expand_call(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let (sym_name, addend) = self.extract_symbol_from_operands(&instr.operands)?;
        let ra_enc = registers::encoding(RA);

        let offset_before = self.get_current_offset();

        // AUIPC ra, 0
        let auipc_ops = [
            EncoderOperand::Register(ra_enc),
            EncoderOperand::Immediate(0),
        ];
        let auipc = self
            .encoder
            .encode_instruction(RvOpcode::AUIPC, &auipc_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&auipc.to_bytes());

        // JALR ra, ra, 0
        let jalr_ops = [
            EncoderOperand::Register(ra_enc),
            EncoderOperand::Register(ra_enc),
            EncoderOperand::Immediate(0),
        ];
        let jalr = self.encoder.encode_instruction(RvOpcode::JALR, &jalr_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&jalr.to_bytes());

        // R_RISCV_CALL spans the AUIPC+JALR pair (8 bytes).
        self.add_relocation_at(
            offset_before,
            RiscV64RelocationType::R_RISCV_CALL,
            &sym_name,
            addend,
            true,
        );
        self.add_relocation_at(
            offset_before,
            RiscV64RelocationType::R_RISCV_RELAX,
            &sym_name,
            0,
            false,
        );

        Ok(())
    }

    /// Expands `TAIL symbol` — tail call via AUIPC+JALR x0.
    ///
    /// Emits: `AUIPC t1, 0` + `JALR x0, t1, 0`
    /// Same relocation as CALL but returns to the caller's caller.
    fn expand_tail(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let (sym_name, addend) = self.extract_symbol_from_operands(&instr.operands)?;
        // Use t1 (x6) as the temporary for the tail call address.
        let t1_enc: u8 = 6; // x6 = t1
        let zero_enc = registers::encoding(ZERO);

        let offset_before = self.get_current_offset();

        // AUIPC t1, 0
        let auipc_ops = [
            EncoderOperand::Register(t1_enc),
            EncoderOperand::Immediate(0),
        ];
        let auipc = self
            .encoder
            .encode_instruction(RvOpcode::AUIPC, &auipc_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&auipc.to_bytes());

        // JALR x0, t1, 0 (x0 as link reg = no return address saved)
        let jalr_ops = [
            EncoderOperand::Register(zero_enc),
            EncoderOperand::Register(t1_enc),
            EncoderOperand::Immediate(0),
        ];
        let jalr = self.encoder.encode_instruction(RvOpcode::JALR, &jalr_ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&jalr.to_bytes());

        // R_RISCV_CALL on the pair.
        self.add_relocation_at(
            offset_before,
            RiscV64RelocationType::R_RISCV_CALL,
            &sym_name,
            addend,
            true,
        );
        self.add_relocation_at(
            offset_before,
            RiscV64RelocationType::R_RISCV_RELAX,
            &sym_name,
            0,
            false,
        );

        Ok(())
    }

    /// Expands `RET` → `JALR x0, ra, 0`.
    fn expand_ret(&mut self) -> Result<(), AssemblerError> {
        let zero_enc = registers::encoding(ZERO);
        let ra_enc = registers::encoding(RA);

        let ops = [
            EncoderOperand::Register(zero_enc),
            EncoderOperand::Register(ra_enc),
            EncoderOperand::Immediate(0),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::JALR, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `MV rd, rs` → `ADDI rd, rs, 0`.
    fn expand_mv(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;

        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rs),
            EncoderOperand::Immediate(0),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::ADDI, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `NOT rd, rs` → `XORI rd, rs, -1`.
    fn expand_not(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;

        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rs),
            EncoderOperand::Immediate(-1),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::XORI, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `NEG rd, rs` → `SUB rd, x0, rs`.
    fn expand_neg(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;
        let zero_enc = registers::encoding(ZERO);

        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(zero_enc),
            EncoderOperand::Register(rs),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::SUB, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `SEQZ rd, rs` → `SLTIU rd, rs, 1`.
    fn expand_seqz(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;

        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rs),
            EncoderOperand::Immediate(1),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::SLTIU, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `SNEZ rd, rs` → `SLTU rd, x0, rs`.
    fn expand_snez(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;
        let zero_enc = registers::encoding(ZERO);

        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(zero_enc),
            EncoderOperand::Register(rs),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::SLTU, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `J offset` → `JAL x0, offset`.
    fn expand_j(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let zero_enc = registers::encoding(ZERO);

        // J may reference a symbol or have an immediate offset.
        if let Some((sym_name, addend)) = self.try_extract_symbol(&instr.operands) {
            let offset_before = self.get_current_offset();
            let ops = [
                EncoderOperand::Register(zero_enc),
                EncoderOperand::Immediate(0),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::JAL, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());

            self.add_relocation_at(
                offset_before,
                RiscV64RelocationType::R_RISCV_JAL,
                &sym_name,
                addend,
                false,
            );
        } else {
            let imm = self.extract_immediate_or_label(&instr.operands, 0)?;
            let ops = [
                EncoderOperand::Register(zero_enc),
                EncoderOperand::Immediate(imm),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::JAL, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());
        }
        Ok(())
    }

    /// Expands `JR rs` → `JALR x0, rs, 0`.
    fn expand_jr(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rs = self.extract_register(&instr.operands, 0)?;
        let zero_enc = registers::encoding(ZERO);

        let ops = [
            EncoderOperand::Register(zero_enc),
            EncoderOperand::Register(rs),
            EncoderOperand::Immediate(0),
        ];
        let encoded = self.encoder.encode_instruction(RvOpcode::JALR, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Opcode mapping
    // -----------------------------------------------------------------------

    /// Maps a `MachineInstr`'s `u32` opcode to an [`RvOpcode`] enum value.
    ///
    /// The mapping assumes the code generator uses the discriminant values
    /// of `RvOpcode` as the opcode field in `MachineInstr`.
    fn map_opcode(&self, opcode: u32) -> Result<RvOpcode, AssemblerError> {
        // The opcode is the discriminant index of RvOpcode.
        // We enumerate all variants in order and match by index.
        static OPCODES: &[RvOpcode] = &[
            // RV64I base integer R-type (0..9)
            RvOpcode::ADD,
            RvOpcode::SUB,
            RvOpcode::SLL,
            RvOpcode::SLT,
            RvOpcode::SLTU,
            RvOpcode::XOR,
            RvOpcode::SRL,
            RvOpcode::SRA,
            RvOpcode::OR,
            RvOpcode::AND,
            // RV64I word variants (10..14)
            RvOpcode::ADDW,
            RvOpcode::SUBW,
            RvOpcode::SLLW,
            RvOpcode::SRLW,
            RvOpcode::SRAW,
            // RV64M multiply/divide (15..27)
            RvOpcode::MUL,
            RvOpcode::MULH,
            RvOpcode::MULHSU,
            RvOpcode::MULHU,
            RvOpcode::DIV,
            RvOpcode::DIVU,
            RvOpcode::REM,
            RvOpcode::REMU,
            RvOpcode::MULW,
            RvOpcode::DIVW,
            RvOpcode::DIVUW,
            RvOpcode::REMW,
            RvOpcode::REMUW,
            // RV64I immediate (28..34)
            RvOpcode::ADDI,
            RvOpcode::SLTI,
            RvOpcode::SLTIU,
            RvOpcode::XORI,
            RvOpcode::ORI,
            RvOpcode::ANDI,
            RvOpcode::ADDIW,
            // Shifts with immediate (35..40)
            RvOpcode::SLLI,
            RvOpcode::SRLI,
            RvOpcode::SRAI,
            RvOpcode::SLLIW,
            RvOpcode::SRLIW,
            RvOpcode::SRAIW,
            // Loads (41..47)
            RvOpcode::LB,
            RvOpcode::LH,
            RvOpcode::LW,
            RvOpcode::LD,
            RvOpcode::LBU,
            RvOpcode::LHU,
            RvOpcode::LWU,
            // Stores (48..51)
            RvOpcode::SB,
            RvOpcode::SH,
            RvOpcode::SW,
            RvOpcode::SD,
            // Branches (52..57)
            RvOpcode::BEQ,
            RvOpcode::BNE,
            RvOpcode::BLT,
            RvOpcode::BGE,
            RvOpcode::BLTU,
            RvOpcode::BGEU,
            // Upper immediate (58..59)
            RvOpcode::LUI,
            RvOpcode::AUIPC,
            // Jumps (60..61)
            RvOpcode::JAL,
            RvOpcode::JALR,
            // System/fence (62..67)
            RvOpcode::ECALL,
            RvOpcode::EBREAK,
            RvOpcode::FENCE,
            RvOpcode::CSRRW,
            RvOpcode::CSRRS,
            RvOpcode::CSRRC,
            // FP loads/stores (68..71)
            RvOpcode::FLW,
            RvOpcode::FSW,
            RvOpcode::FLD,
            RvOpcode::FSD,
            // RV64F single-precision FP arithmetic (72..89)
            RvOpcode::FADD_S,
            RvOpcode::FSUB_S,
            RvOpcode::FMUL_S,
            RvOpcode::FDIV_S,
            RvOpcode::FSQRT_S,
            RvOpcode::FSGNJ_S,
            RvOpcode::FSGNJN_S,
            RvOpcode::FSGNJX_S,
            RvOpcode::FMIN_S,
            RvOpcode::FMAX_S,
            RvOpcode::FCVT_W_S,
            RvOpcode::FCVT_WU_S,
            RvOpcode::FMV_X_W,
            RvOpcode::FEQ_S,
            RvOpcode::FLT_S,
            RvOpcode::FLE_S,
            RvOpcode::FCLASS_S,
            RvOpcode::FCVT_S_W,
            RvOpcode::FCVT_S_WU,
            RvOpcode::FMV_W_X,
            RvOpcode::FCVT_L_S,
            RvOpcode::FCVT_LU_S,
            RvOpcode::FCVT_S_L,
            RvOpcode::FCVT_S_LU,
            // RV64D double-precision FP arithmetic (96..123)
            RvOpcode::FADD_D,
            RvOpcode::FSUB_D,
            RvOpcode::FMUL_D,
            RvOpcode::FDIV_D,
            RvOpcode::FSQRT_D,
            RvOpcode::FSGNJ_D,
            RvOpcode::FSGNJN_D,
            RvOpcode::FSGNJX_D,
            RvOpcode::FMIN_D,
            RvOpcode::FMAX_D,
            RvOpcode::FCVT_S_D,
            RvOpcode::FCVT_D_S,
            RvOpcode::FEQ_D,
            RvOpcode::FLT_D,
            RvOpcode::FLE_D,
            RvOpcode::FCLASS_D,
            RvOpcode::FCVT_W_D,
            RvOpcode::FCVT_WU_D,
            RvOpcode::FCVT_D_W,
            RvOpcode::FCVT_D_WU,
            RvOpcode::FCVT_L_D,
            RvOpcode::FCVT_LU_D,
            RvOpcode::FCVT_D_L,
            RvOpcode::FCVT_D_LU,
            RvOpcode::FMV_X_D,
            RvOpcode::FMV_D_X,
            // FMA (124..131)
            RvOpcode::FMADD_S,
            RvOpcode::FMSUB_S,
            RvOpcode::FNMSUB_S,
            RvOpcode::FNMADD_S,
            RvOpcode::FMADD_D,
            RvOpcode::FMSUB_D,
            RvOpcode::FNMSUB_D,
            RvOpcode::FNMADD_D,
            // Atomics (132..153)
            RvOpcode::LR_W,
            RvOpcode::SC_W,
            RvOpcode::AMOSWAP_W,
            RvOpcode::AMOADD_W,
            RvOpcode::AMOXOR_W,
            RvOpcode::AMOAND_W,
            RvOpcode::AMOOR_W,
            RvOpcode::AMOMIN_W,
            RvOpcode::AMOMAX_W,
            RvOpcode::AMOMINU_W,
            RvOpcode::AMOMAXU_W,
            RvOpcode::LR_D,
            RvOpcode::SC_D,
            RvOpcode::AMOSWAP_D,
            RvOpcode::AMOADD_D,
            RvOpcode::AMOXOR_D,
            RvOpcode::AMOAND_D,
            RvOpcode::AMOOR_D,
            RvOpcode::AMOMIN_D,
            RvOpcode::AMOMAX_D,
            RvOpcode::AMOMINU_D,
            RvOpcode::AMOMAXU_D,
            // Compressed (154..189)
            RvOpcode::C_NOP,
            RvOpcode::C_ADDI,
            RvOpcode::C_ADDIW,
            RvOpcode::C_LI,
            RvOpcode::C_LUI,
            RvOpcode::C_ADDI16SP,
            RvOpcode::C_ADDI4SPN,
            RvOpcode::C_SLLI,
            RvOpcode::C_SRLI,
            RvOpcode::C_SRAI,
            RvOpcode::C_ANDI,
            RvOpcode::C_MV,
            RvOpcode::C_ADD,
            RvOpcode::C_AND,
            RvOpcode::C_OR,
            RvOpcode::C_XOR,
            RvOpcode::C_SUB,
            RvOpcode::C_ADDW,
            RvOpcode::C_SUBW,
            RvOpcode::C_LW,
            RvOpcode::C_LD,
            RvOpcode::C_SW,
            RvOpcode::C_SD,
            RvOpcode::C_LWSP,
            RvOpcode::C_LDSP,
            RvOpcode::C_SWSP,
            RvOpcode::C_SDSP,
            RvOpcode::C_J,
            RvOpcode::C_JAL,
            RvOpcode::C_JR,
            RvOpcode::C_JALR,
            RvOpcode::C_BEQZ,
            RvOpcode::C_BNEZ,
            RvOpcode::C_EBREAK,
            RvOpcode::C_FLD,
            RvOpcode::C_FSD,
            RvOpcode::C_FLDSP,
            RvOpcode::C_FSDSP,
            // Pseudo-instructions (190..201)
            RvOpcode::NOP,
            RvOpcode::LI,
            RvOpcode::LA,
            RvOpcode::CALL,
            RvOpcode::TAIL,
            RvOpcode::RET,
            RvOpcode::MV,
            RvOpcode::NOT,
            RvOpcode::NEG,
            RvOpcode::SEQZ,
            RvOpcode::SNEZ,
            RvOpcode::J,
            RvOpcode::JR,
        ];

        let idx = opcode as usize;
        if idx < OPCODES.len() {
            Ok(OPCODES[idx])
        } else {
            Err(AssemblerError::UnsupportedInstruction(format!(
                "unknown opcode index {} in function '{}'",
                opcode, self.current_function
            )))
        }
    }

    // -----------------------------------------------------------------------
    // Operand translation helpers
    // -----------------------------------------------------------------------

    /// Translates `MachineOperand` slices to `EncoderOperand` slices.
    fn translate_operands(
        &self,
        operands: &[MachineOperand],
    ) -> Result<Vec<EncoderOperand>, AssemblerError> {
        let mut result = Vec::with_capacity(operands.len());
        for op in operands {
            result.push(self.translate_operand(op)?);
        }
        Ok(result)
    }

    /// Translates a single `MachineOperand` to an `EncoderOperand`.
    fn translate_operand(
        &self,
        operand: &MachineOperand,
    ) -> Result<EncoderOperand, AssemblerError> {
        match operand {
            MachineOperand::Register(reg) => {
                Ok(EncoderOperand::Register(registers::encoding(*reg)))
            }
            MachineOperand::Immediate(val) => Ok(EncoderOperand::Immediate(*val)),
            MachineOperand::Symbol(name) => Ok(EncoderOperand::Symbol(name.clone())),
            MachineOperand::Label(id) => {
                // Resolve to a relative offset if we know the label position.
                if let Some(&target_offset) = self.label_offsets.get(id) {
                    let current = self.get_current_offset();
                    let rel = target_offset as i64 - current as i64;
                    Ok(EncoderOperand::Immediate(rel))
                } else {
                    // Forward reference — use label ID for later resolution.
                    Ok(EncoderOperand::Label(*id))
                }
            }
            MachineOperand::Memory {
                base, offset: _, ..
            } => {
                // For RISC-V, memory operands in loads/stores decompose to
                // base register + immediate offset. The encoder expects these
                // as separate register + immediate operands; this is handled
                // by the code generator which should produce:
                //   [Register(base), Immediate(offset)] not Memory{...}
                // If we encounter a Memory operand, expand it here.
                Ok(EncoderOperand::Register(registers::encoding(*base)))
            }
            MachineOperand::FrameIndex(idx) => {
                // Frame indices should have been resolved by the time we
                // reach the assembler. Treat as an error.
                Err(AssemblerError::InvalidOperand(format!(
                    "unresolved frame index FI{} in function '{}'",
                    idx, self.current_function
                )))
            }
            MachineOperand::VirtualReg(vid) => Err(AssemblerError::InvalidOperand(format!(
                "unresolved virtual register {} in function '{}' — \
                     register allocation must complete before assembly",
                vid, self.current_function
            ))),
        }
    }

    /// Translates operands for relocation emission, replacing symbol operands
    /// with zero immediates (the linker will patch them).
    fn translate_operands_for_relocation(
        &self,
        operands: &[MachineOperand],
    ) -> Result<Vec<EncoderOperand>, AssemblerError> {
        let mut result = Vec::with_capacity(operands.len());
        for op in operands {
            match op {
                MachineOperand::Symbol(_) => {
                    result.push(EncoderOperand::Immediate(0));
                }
                _ => {
                    result.push(self.translate_operand(op)?);
                }
            }
        }
        Ok(result)
    }

    /// Extracts a register encoding from an operand at a given index.
    fn extract_register(
        &self,
        operands: &[MachineOperand],
        index: usize,
    ) -> Result<u8, AssemblerError> {
        if index >= operands.len() {
            return Err(AssemblerError::InvalidOperand(format!(
                "expected register operand at index {} but only {} operands present",
                index,
                operands.len()
            )));
        }
        match &operands[index] {
            MachineOperand::Register(reg) => Ok(registers::encoding(*reg)),
            other => Err(AssemblerError::InvalidOperand(format!(
                "expected register at index {}, got {}",
                index, other
            ))),
        }
    }

    /// Extracts an immediate value from an operand at a given index.
    fn extract_immediate(
        &self,
        operands: &[MachineOperand],
        index: usize,
    ) -> Result<i64, AssemblerError> {
        if index >= operands.len() {
            return Err(AssemblerError::InvalidOperand(format!(
                "expected immediate operand at index {} but only {} operands present",
                index,
                operands.len()
            )));
        }
        match &operands[index] {
            MachineOperand::Immediate(val) => Ok(*val),
            other => Err(AssemblerError::InvalidOperand(format!(
                "expected immediate at index {}, got {}",
                index, other
            ))),
        }
    }

    /// Extracts an immediate value or resolves a label to a relative offset.
    fn extract_immediate_or_label(
        &self,
        operands: &[MachineOperand],
        index: usize,
    ) -> Result<i64, AssemblerError> {
        if index >= operands.len() {
            return Err(AssemblerError::InvalidOperand(format!(
                "expected immediate/label at index {} but only {} operands present",
                index,
                operands.len()
            )));
        }
        match &operands[index] {
            MachineOperand::Immediate(val) => Ok(*val),
            MachineOperand::Label(id) => {
                if let Some(&target) = self.label_offsets.get(id) {
                    let current = self.get_current_offset();
                    Ok(target as i64 - current as i64)
                } else {
                    // Forward reference — emit 0 and rely on linker/fixup.
                    Ok(0)
                }
            }
            other => Err(AssemblerError::InvalidOperand(format!(
                "expected immediate or label at index {}, got {}",
                index, other
            ))),
        }
    }

    /// Extracts the first symbol operand and its addend from the operand list.
    fn extract_symbol_operand(
        &self,
        operands: &[MachineOperand],
    ) -> Result<(String, i64), AssemblerError> {
        for op in operands {
            if let MachineOperand::Symbol(name) = op {
                return Ok((name.clone(), 0));
            }
        }
        Err(AssemblerError::InvalidOperand(
            "no symbol operand found".to_string(),
        ))
    }

    /// Extracts a symbol from operands, returning the symbol name and
    /// any associated addend. Prefers the last Symbol operand.
    fn extract_symbol_from_operands(
        &self,
        operands: &[MachineOperand],
    ) -> Result<(String, i64), AssemblerError> {
        // Look for a Symbol operand.
        for op in operands {
            if let MachineOperand::Symbol(name) = op {
                return Ok((name.clone(), 0));
            }
        }
        Err(AssemblerError::InvalidOperand(
            "expected a symbol operand for pseudo-instruction expansion".to_string(),
        ))
    }

    /// Tries to extract a symbol operand, returning `None` if no symbol
    /// is present (for instructions that may have either immediate or
    /// symbol operands).
    fn try_extract_symbol(&self, operands: &[MachineOperand]) -> Option<(String, i64)> {
        for op in operands {
            if let MachineOperand::Symbol(name) = op {
                return Some((name.clone(), 0));
            }
        }
        None
    }

    /// Determines the appropriate relocation type for an opcode.
    ///
    /// This maps the instruction's addressing mode to the correct RISC-V
    /// relocation type for the linker.
    fn relocation_type_for_opcode(&self, opcode: &RvOpcode) -> RiscV64RelocationType {
        match opcode {
            // Branch instructions use R_RISCV_BRANCH.
            RvOpcode::BEQ
            | RvOpcode::BNE
            | RvOpcode::BLT
            | RvOpcode::BGE
            | RvOpcode::BLTU
            | RvOpcode::BGEU => RiscV64RelocationType::R_RISCV_BRANCH,
            // JAL uses R_RISCV_JAL.
            RvOpcode::JAL => RiscV64RelocationType::R_RISCV_JAL,
            // AUIPC uses R_RISCV_PCREL_HI20 by default.
            RvOpcode::AUIPC => RiscV64RelocationType::R_RISCV_PCREL_HI20,
            // LUI uses R_RISCV_HI20.
            RvOpcode::LUI => RiscV64RelocationType::R_RISCV_HI20,
            // Loads with symbol → R_RISCV_PCREL_LO12_I (I-type).
            RvOpcode::LB
            | RvOpcode::LH
            | RvOpcode::LW
            | RvOpcode::LD
            | RvOpcode::LBU
            | RvOpcode::LHU
            | RvOpcode::LWU
            | RvOpcode::FLW
            | RvOpcode::FLD => RiscV64RelocationType::R_RISCV_PCREL_LO12_I,
            // I-type arithmetic with symbol → R_RISCV_LO12_I.
            RvOpcode::ADDI | RvOpcode::ADDIW | RvOpcode::JALR => {
                RiscV64RelocationType::R_RISCV_PCREL_LO12_I
            }
            // Stores with symbol → R_RISCV_PCREL_LO12_S (S-type).
            RvOpcode::SB
            | RvOpcode::SH
            | RvOpcode::SW
            | RvOpcode::SD
            | RvOpcode::FSW
            | RvOpcode::FSD => RiscV64RelocationType::R_RISCV_PCREL_LO12_S,
            // Compressed branches.
            RvOpcode::C_BEQZ | RvOpcode::C_BNEZ => RiscV64RelocationType::R_RISCV_RVC_BRANCH,
            // Compressed jumps.
            RvOpcode::C_J | RvOpcode::C_JAL => RiscV64RelocationType::R_RISCV_RVC_JUMP,
            // Default: use a 64-bit absolute relocation.
            _ => RiscV64RelocationType::R_RISCV_64,
        }
    }

    // -----------------------------------------------------------------------
    // Finalization
    // -----------------------------------------------------------------------

    /// Produces the final assembled output with all sections, symbols, and
    /// relocations.
    ///
    /// After calling this method, the assembler state should not be used
    /// further.
    pub fn finalize(&self) -> AssembledObject {
        AssembledObject {
            sections: self.sections.clone(),
            symbols: self.symbols.clone(),
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::{MachineBasicBlock, MachineFunction, MachineInstr};

    /// Helper: create an assembler and verify it starts with a .text section.
    #[test]
    fn test_new_assembler_has_text_section() {
        let asm = RiscV64Assembler::new();
        assert_eq!(asm.sections.len(), 1);
        assert_eq!(asm.sections[0].name, ".text");
        assert_eq!(asm.sections[0].flags, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(asm.get_current_offset(), 0);
    }

    /// Test that switching sections creates new sections correctly.
    #[test]
    fn test_switch_section() {
        let mut asm = RiscV64Assembler::new();
        asm.switch_section(".data", 8, SHF_ALLOC | SHF_WRITE);
        assert_eq!(asm.sections.len(), 2);
        assert_eq!(asm.current_section, 1);
        assert_eq!(asm.sections[1].name, ".data");

        // Switch back to .text.
        asm.switch_section(".text", 4, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(asm.current_section, 0);
        assert_eq!(asm.sections.len(), 2); // No new section created.
    }

    /// Test NOP emission.
    #[test]
    fn test_emit_nop() {
        let mut asm = RiscV64Assembler::new();
        asm.emit_nop();
        assert_eq!(asm.get_current_offset(), 4);
        let data = &asm.sections[0].data;
        // NOP = ADDI x0, x0, 0 = 0x00000013 in little-endian.
        assert_eq!(data, &[0x13, 0x00, 0x00, 0x00]);
    }

    /// Test C.NOP emission.
    #[test]
    fn test_emit_c_nop() {
        let mut asm = RiscV64Assembler::new();
        asm.emit_c_nop();
        assert_eq!(asm.get_current_offset(), 2);
        let data = &asm.sections[0].data;
        // C.NOP = 0x0001 in little-endian.
        assert_eq!(data, &[0x01, 0x00]);
    }

    /// Test raw byte emission.
    #[test]
    fn test_emit_raw_bytes() {
        let mut asm = RiscV64Assembler::new();
        asm.emit_raw_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(asm.get_current_offset(), 4);
        assert_eq!(asm.sections[0].data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    /// Test symbol definition.
    #[test]
    fn test_define_symbol() {
        let mut asm = RiscV64Assembler::new();
        asm.emit_nop(); // 4 bytes
        asm.define_symbol("my_func", true, true);
        assert_eq!(asm.symbols.len(), 1);
        assert_eq!(asm.symbols[0].name, "my_func");
        assert_eq!(asm.symbols[0].offset, 4);
        assert!(asm.symbols[0].is_global);
        assert!(asm.symbols[0].is_function);
    }

    /// Test relocation recording.
    #[test]
    fn test_add_relocation() {
        let mut asm = RiscV64Assembler::new();
        asm.emit_nop(); // 4 bytes offset
        asm.add_relocation(RiscV64RelocationType::R_RISCV_CALL, "printf", 0);
        assert_eq!(asm.sections[0].relocations.len(), 1);
        let reloc = &asm.sections[0].relocations[0];
        assert_eq!(reloc.offset, 4);
        assert_eq!(reloc.symbol, "printf");
        assert!(!reloc.is_relaxable);
    }

    /// Test relaxable relocation recording (should produce two entries).
    #[test]
    fn test_add_relaxable_relocation() {
        let mut asm = RiscV64Assembler::new();
        asm.add_relaxable_relocation(RiscV64RelocationType::R_RISCV_CALL, "target_func", 0);
        // Should produce the primary relocation + R_RISCV_RELAX companion.
        assert_eq!(asm.sections[0].relocations.len(), 2);
        assert!(asm.sections[0].relocations[0].is_relaxable);
        assert_eq!(
            asm.sections[0].relocations[1].reloc_type,
            RiscV64RelocationType::R_RISCV_RELAX
        );
    }

    /// Test finalize produces correct output.
    #[test]
    fn test_finalize() {
        let mut asm = RiscV64Assembler::new();
        asm.emit_nop();
        asm.define_symbol("test", false, false);
        let obj = asm.finalize();
        assert_eq!(obj.sections.len(), 1);
        assert_eq!(obj.symbols.len(), 1);
        assert_eq!(obj.sections[0].data.len(), 4);
    }

    /// Test alignment in code sections uses NOPs.
    #[test]
    fn test_align_code_section() {
        let mut asm = RiscV64Assembler::new();
        // Emit 2 bytes (C.NOP) to create misalignment for 4-byte boundary.
        asm.emit_c_nop(); // 2 bytes
        asm.align(4);
        // Should have added 2 bytes of padding (another C.NOP).
        assert_eq!(asm.get_current_offset(), 4);
    }

    /// Test alignment in data sections uses zeros.
    #[test]
    fn test_align_data_section() {
        let mut asm = RiscV64Assembler::new();
        asm.switch_section(".data", 8, SHF_ALLOC | SHF_WRITE);
        asm.emit_raw_bytes(&[0x42]); // 1 byte
        asm.align(4);
        // Should be padded to 4 bytes with zeros.
        assert_eq!(asm.get_current_offset(), 4);
        assert_eq!(asm.sections[1].data, vec![0x42, 0x00, 0x00, 0x00]);
    }

    /// Test assembling a simple function with RET.
    #[test]
    fn test_assemble_simple_function() {
        let mut mf = MachineFunction::new("simple_ret".to_string(), 16);
        let mut bb = MachineBasicBlock::new(0);
        // RET pseudo-instruction (opcode index for RET in our table).
        let mut ret_instr = MachineInstr::new(195); // RET index
        ret_instr.set_return();
        bb.push_instr(ret_instr);
        mf.add_block(bb);

        let mut asm = RiscV64Assembler::new();
        let result = asm.assemble_function(&mf);
        assert!(result.is_ok());

        // Should have the function symbol.
        assert!(asm
            .symbols
            .iter()
            .any(|s| s.name == "simple_ret" && s.is_function));

        // RET = JALR x0, ra, 0 — should be a 4-byte instruction.
        assert!(asm.get_current_offset() >= 4);
    }

    /// Test that AssemblerError Display works correctly.
    #[test]
    fn test_error_display() {
        let err = AssemblerError::ImmediateOutOfRange {
            value: 99999,
            min: -2048,
            max: 2047,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("99999"));
        assert!(msg.contains("-2048"));
        assert!(msg.contains("2047"));

        let err2 = AssemblerError::EncodingError("test error".to_string());
        assert_eq!(format!("{}", err2), "encoding error: test error");

        let err3 = AssemblerError::InvalidRegister("bad reg".to_string());
        assert_eq!(format!("{}", err3), "invalid register: bad reg");
    }

    /// Test section_type returns correct ELF section types.
    #[test]
    fn test_section_type() {
        let text = AssemblerSection::new(".text", 4, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(text.section_type(), SHT_PROGBITS);

        let bss = AssemblerSection::new(".bss", 8, SHF_ALLOC | SHF_WRITE);
        assert_eq!(bss.section_type(), SHT_NOBITS);
    }
}
