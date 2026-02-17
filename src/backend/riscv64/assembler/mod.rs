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

/// Maps a RISC-V integer register encoding (0–31) to its ABI name
/// for inline assembly operand substitution.
fn asm_encoding_to_name(enc: u8) -> &'static str {
    match enc {
        0 => "zero",
        1 => "ra",
        2 => "sp",
        3 => "gp",
        4 => "tp",
        5 => "t0",
        6 => "t1",
        7 => "t2",
        8 => "s0",
        9 => "s1",
        10 => "a0",
        11 => "a1",
        12 => "a2",
        13 => "a3",
        14 => "a4",
        15 => "a5",
        16 => "a6",
        17 => "a7",
        18 => "s2",
        19 => "s3",
        20 => "s4",
        21 => "s5",
        22 => "s6",
        23 => "s7",
        24 => "s8",
        25 => "s9",
        26 => "s10",
        27 => "s11",
        28 => "t3",
        29 => "t4",
        30 => "t5",
        31 => "t6",
        _ => "zero",
    }
}

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
/// A pending branch fixup for a forward label reference.
///
/// When a branch instruction targets a label that hasn't been assembled yet,
/// the assembler emits the instruction with a zero offset and records this
/// fixup. After all blocks in the function are assembled, fixups are resolved
/// by patching the branch offset fields with the correct relative displacement.
struct BranchFixup {
    /// Byte offset in the section data where the branch instruction starts.
    instr_offset: u32,
    /// The target basic-block label ID.
    target_label: u32,
    /// Whether this is a B-type (conditional) or J-type (unconditional) branch.
    is_j_type: bool,
}

/// Fixup for label-address loads (AUIPC+ADDI pairs used by computed gotos).
/// After all blocks are assembled, the AUIPC immediate is patched to encode
/// the PC-relative offset's upper 20 bits and the ADDI encodes the lower 12.
struct LabelAddressFixup {
    /// Byte offset in the section data where the AUIPC instruction starts.
    auipc_offset: u32,
    /// The target basic-block label ID whose address we want to load.
    target_label: u32,
}

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
    /// Pending branch fixups for forward label references in the current function.
    branch_fixups: Vec<BranchFixup>,
    /// Pending AUIPC+ADDI fixups for label-address loads (computed gotos).
    label_address_fixups: Vec<LabelAddressFixup>,
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
            branch_fixups: Vec::new(),
            label_address_fixups: Vec::new(),
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

        // Clear label offset map and fixup list for this function.
        self.label_offsets.clear();
        self.branch_fixups.clear();

        // Single-pass encoding: instructions are encoded and labels are
        // recorded as we go. Forward branch references are emitted with
        // zero offsets and tracked in `branch_fixups`. After all blocks
        // are assembled, fixups are resolved by patching branch offset
        // fields with the correct relative displacements.

        // Encode all blocks.
        for block in &mf.blocks {
            self.assemble_block(block)?;
        }

        // Resolve forward branch fixups now that all labels are known.
        self.resolve_branch_fixups()?;

        // Resolve label-address fixups (AUIPC+ADDI for computed gotos).
        self.resolve_label_address_fixups()?;

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
        // ---- Handle SPILL pseudo-ops generated by the register allocator ----
        // These must be intercepted before map_opcode since they use sentinel
        // opcode values (0xFFFF_FFFE, 0xFFFF_FFFF) that have no RvOpcode mapping.
        {
            use crate::backend::register_allocator::{SPILL_LOAD_OPCODE, SPILL_STORE_OPCODE};
            match instr.opcode {
                SPILL_LOAD_OPCODE => return self.expand_spill_load(instr),
                SPILL_STORE_OPCODE => return self.expand_spill_store(instr),
                _ => {}
            }
        }

        // ---- Handle inline assembly pseudo-op ----
        // Inline ASM instructions carry the template as a Symbol operand.
        // We parse the template and emit the corresponding RISC-V machine
        // instructions directly.  Unsupported / unrecognised lines are
        // emitted as NOPs to keep the binary valid.
        {
            use crate::backend::riscv64::codegen::RV_INLINE_ASM;
            if instr.opcode == RV_INLINE_ASM {
                return self.expand_inline_asm(instr);
            }
        }

        // ---- Handle LA_LABEL pseudo-op (load address of local label). ----
        // Used for computed gotos: loads the address of a basic block label
        // into a GPR via AUIPC+ADDI with a forward-fixup.
        {
            use crate::backend::riscv64::codegen::RV_LA_LABEL;
            if instr.opcode == RV_LA_LABEL {
                return self.expand_la_label(instr);
            }
        }

        // ---- Handle FP pseudo-ops before map_opcode (they have no
        //      corresponding RvOpcode variant and expand to fsgnj / fsgnjn). ----
        {
            use crate::backend::riscv64::codegen::{RV_FMOV_D, RV_FMOV_S, RV_FNEG_D, RV_FNEG_S};
            match instr.opcode {
                RV_FMOV_S => return self.expand_fmov(instr, RvOpcode::FSGNJ_S),
                RV_FMOV_D => return self.expand_fmov(instr, RvOpcode::FSGNJ_D),
                RV_FNEG_S => return self.expand_fneg(instr, RvOpcode::FSGNJN_S),
                RV_FNEG_D => return self.expand_fneg(instr, RvOpcode::FSGNJN_D),
                _ => {}
            }
        }

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

        // ----- Forward-label detection for branch / jump instructions -----
        //
        // B-type (BEQ, BNE, BLT, BGE, BLTU, BGEU) operands: [rs1, rs2, Label]
        // J-type (JAL) operands: [rd, Label]
        //
        // If the Label is a forward reference (not yet in `label_offsets`),
        // we emit the instruction with offset = 0 and record a `BranchFixup`
        // so that `resolve_branch_fixups` can patch the real offset later.

        let is_b_type = matches!(
            opcode,
            RvOpcode::BEQ
                | RvOpcode::BNE
                | RvOpcode::BLT
                | RvOpcode::BGE
                | RvOpcode::BLTU
                | RvOpcode::BGEU
        );
        let is_j_type = matches!(opcode, RvOpcode::JAL);

        if is_b_type || is_j_type {
            if let Some(fwd_label) = self.find_forward_label(&instr.operands) {
                let instr_offset = self.get_current_offset();
                self.branch_fixups.push(BranchFixup {
                    instr_offset,
                    target_label: fwd_label,
                    is_j_type,
                });

                // Build encoder operands with the label replaced by Immediate(0).
                let encoder_ops = self.translate_operands_replacing_labels(&instr.operands)?;
                let encoded = self
                    .encoder
                    .encode_instruction(opcode, &encoder_ops)
                    .map_err(|e| {
                        AssemblerError::EncodingError(format!(
                            "forward-label fixup encoding: {:?}",
                            e
                        ))
                    })?;
                let bytes = encoded.to_bytes();
                self.current_section_mut().data.extend_from_slice(&bytes);
                return Ok(());
            }
        }

        // Standard instruction encoding path (no forward labels).
        let encoder_ops = self.translate_operands(&instr.operands)?;
        let encoded = self.encoder.encode_instruction(opcode, &encoder_ops).map_err(|e| {
            eprintln!(
                "[RV64_ASM_DBG] encode error in fn='{}': opcode={:?}, operands={:?}, enc_ops={:?}, err={:?}",
                self.current_function, opcode, instr.operands, encoder_ops, e
            );
            AssemblerError::EncodingError(e.to_string())
        })?;
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
    // -----------------------------------------------------------------------
    // Spill pseudo-op expansion
    // -----------------------------------------------------------------------

    /// Expands a SPILL_LOAD pseudo-instruction into a real load from a
    /// stack slot.
    ///
    /// Operands: `[Register(scratch), FrameIndex(offset)]`
    ///
    /// Emits: `LD scratch, offset(SP)` for GPR spills
    ///        `FLD scratch, offset(SP)` for FP spills
    fn expand_spill_load(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        use crate::backend::traits::MachineOperand;
        let scratch = self.extract_register(&instr.operands, 0)?;
        let offset: i64 = match instr.operands.get(1) {
            Some(MachineOperand::FrameIndex(off)) => *off as i64,
            Some(MachineOperand::Immediate(off)) => *off as i64,
            _ => 0,
        };
        let sp = registers::encoding(registers::SP);

        // Determine if FP register (PhysReg index >= 32)
        let is_fp = instr.operands.first().map_or(false, |op| {
            if let MachineOperand::Register(r) = op {
                r.0 >= 32
            } else {
                false
            }
        });

        if is_fp {
            // FLD rd, offset(sp) - I-type with opcode 0x07, funct3=011
            let ops = [
                EncoderOperand::Register(scratch),
                EncoderOperand::Register(sp),
                EncoderOperand::Immediate(offset),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::FLD, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());
        } else {
            // LD rd, offset(sp) - I-type
            let ops = [
                EncoderOperand::Register(scratch),
                EncoderOperand::Register(sp),
                EncoderOperand::Immediate(offset),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::LD, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());
        }
        Ok(())
    }

    /// Expands a SPILL_STORE pseudo-instruction into a real store to a
    /// stack slot.
    ///
    /// Operands: `[Register(scratch), FrameIndex(offset)]`
    ///
    /// Emits: `SD scratch, offset(SP)` for GPR spills
    ///        `FSD scratch, offset(SP)` for FP spills
    fn expand_spill_store(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        use crate::backend::traits::MachineOperand;
        let scratch = self.extract_register(&instr.operands, 0)?;
        let offset: i64 = match instr.operands.get(1) {
            Some(MachineOperand::FrameIndex(off)) => *off as i64,
            Some(MachineOperand::Immediate(off)) => *off as i64,
            _ => 0,
        };
        let sp = registers::encoding(registers::SP);

        // Determine if FP register (PhysReg index >= 32)
        let is_fp = instr.operands.first().map_or(false, |op| {
            if let MachineOperand::Register(r) = op {
                r.0 >= 32
            } else {
                false
            }
        });

        if is_fp {
            // FSD rs2, offset(sp) - S-type with opcode 0x27, funct3=011
            let ops = [
                EncoderOperand::Register(scratch),
                EncoderOperand::Register(sp),
                EncoderOperand::Immediate(offset),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::FSD, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());
        } else {
            // SD rs2, offset(sp) - S-type
            let ops = [
                EncoderOperand::Register(scratch),
                EncoderOperand::Register(sp),
                EncoderOperand::Immediate(offset),
            ];
            let encoded = self.encoder.encode_instruction(RvOpcode::SD, &ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&encoded.to_bytes());
        }
        Ok(())
    }

    /// Expands an inline-assembly pseudo-instruction.
    ///
    /// The template string is extracted from the instruction's Symbol operand.
    /// Each line of the template is individually parsed: recognised RISC-V
    /// mnemonics are encoded normally; unrecognised lines are emitted as NOPs
    /// so that the binary remains valid and the surrounding code's offsets
    /// are not disturbed.
    fn expand_inline_asm(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        use crate::backend::traits::MachineOperand;

        // Collect operand registers:
        // The instruction operands are arranged as:
        //   [result_vreg/reg?, input_vreg/reg*, Symbol(template), Symbol("constraints:...")]
        // We build a mapping from $N → physical register encoding.
        let mut operand_regs: Vec<u8> = Vec::new();
        let mut template_str: Option<&str> = None;

        for op in &instr.operands {
            match op {
                MachineOperand::Register(r) => {
                    operand_regs.push(registers::encoding(*r));
                }
                MachineOperand::VirtualReg(_) => {
                    // Virtual register — should have been allocated to a
                    // physical register by the register allocator.  The
                    // allocator replaces VirtualReg with Register operands,
                    // but if we reach here before allocation, emit a NOP
                    // as a safe fallback.
                    // After register allocation, all VirtualRegs should be
                    // resolved.  For safety, push encoding 0 (x0/zero).
                    operand_regs.push(0);
                }
                MachineOperand::Symbol(s) => {
                    if !s.starts_with("constraints:") && template_str.is_none() {
                        template_str = Some(s.as_str());
                    }
                }
                _ => {}
            }
        }

        let template = template_str.unwrap_or("");

        // Each semicolon or newline-separated element is one asm line.
        for raw_line in template.split(|c| c == ';' || c == '\n') {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            // Skip assembler directives (.pushsection, .popsection, etc.)
            if line.starts_with('.') {
                continue;
            }

            // Substitute $N operand references with actual register names,
            // then parse and encode the resulting instruction.
            let expanded = self.substitute_asm_operands(line, &operand_regs);
            if !self.try_encode_asm_line(&expanded)? {
                // Unrecognised instruction — emit NOP as safe fallback.
                self.emit_nop();
            }
        }
        Ok(())
    }

    /// Substitutes `$N` operand references in an inline assembly template
    /// with the ABI register name corresponding to the N-th operand.
    fn substitute_asm_operands(&self, template: &str, operand_regs: &[u8]) -> String {
        let mut result = String::with_capacity(template.len() + 16);
        let mut chars = template.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '$' {
                // Parse the operand index (one or more digits).
                let mut digits = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() {
                        digits.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Ok(idx) = digits.parse::<usize>() {
                    if idx < operand_regs.len() {
                        // Map encoding to ABI name.
                        let enc = operand_regs[idx];
                        result.push_str(asm_encoding_to_name(enc));
                    } else {
                        // Out-of-range operand — keep literal.
                        result.push('$');
                        result.push_str(&digits);
                    }
                } else {
                    result.push('$');
                    result.push_str(&digits);
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    /// Attempts to parse and encode a single inline assembly line.
    /// Returns `true` if the instruction was successfully encoded,
    /// `false` if the mnemonic was not recognised.
    fn try_encode_asm_line(&mut self, line: &str) -> Result<bool, AssemblerError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(true); // empty line
        }

        let mnemonic = parts[0].to_lowercase();
        // Parse operand tokens (comma-separated after mnemonic).
        let operand_str = if parts.len() > 1 {
            parts[1..].join(" ")
        } else {
            String::new()
        };
        let operand_tokens: Vec<&str> = if operand_str.is_empty() {
            Vec::new()
        } else {
            operand_str.split(',').map(|s| s.trim()).collect()
        };

        match mnemonic.as_str() {
            "addi" | "addiw" => {
                // addi rd, rs, imm
                if operand_tokens.len() >= 3 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs = self.parse_asm_reg(operand_tokens[1]);
                    let imm = self.parse_asm_imm(operand_tokens[2]);
                    let opc = if mnemonic == "addiw" {
                        RvOpcode::ADDIW
                    } else {
                        RvOpcode::ADDI
                    };
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs),
                        EncoderOperand::Immediate(imm),
                    ];
                    let encoded = self.encoder.encode_instruction(opc, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "add" | "addw" => {
                if operand_tokens.len() >= 3 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs1 = self.parse_asm_reg(operand_tokens[1]);
                    let rs2 = self.parse_asm_reg(operand_tokens[2]);
                    let opc = if mnemonic == "addw" {
                        RvOpcode::ADDW
                    } else {
                        RvOpcode::ADD
                    };
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs1),
                        EncoderOperand::Register(rs2),
                    ];
                    let encoded = self.encoder.encode_instruction(opc, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "sub" | "subw" => {
                if operand_tokens.len() >= 3 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs1 = self.parse_asm_reg(operand_tokens[1]);
                    let rs2 = self.parse_asm_reg(operand_tokens[2]);
                    let opc = if mnemonic == "subw" {
                        RvOpcode::SUBW
                    } else {
                        RvOpcode::SUB
                    };
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs1),
                        EncoderOperand::Register(rs2),
                    ];
                    let encoded = self.encoder.encode_instruction(opc, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "mv" => {
                // mv rd, rs -> addi rd, rs, 0
                if operand_tokens.len() >= 2 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs = self.parse_asm_reg(operand_tokens[1]);
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs),
                        EncoderOperand::Immediate(0),
                    ];
                    let encoded = self.encoder.encode_instruction(RvOpcode::ADDI, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "li" => {
                // li rd, imm
                if operand_tokens.len() >= 2 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let imm = self.parse_asm_imm(operand_tokens[1]);
                    if (-2048..=2047).contains(&imm) {
                        let ops = [
                            EncoderOperand::Register(rd),
                            EncoderOperand::Register(registers::encoding(registers::ZERO)),
                            EncoderOperand::Immediate(imm),
                        ];
                        let encoded = self.encoder.encode_instruction(RvOpcode::ADDI, &ops)?;
                        self.current_section_mut()
                            .data
                            .extend_from_slice(&encoded.to_bytes());
                    } else {
                        // Multi-instruction sequence.
                        let (hi, lo) = split_hi_lo(imm);
                        let lui_ops = [
                            EncoderOperand::Register(rd),
                            EncoderOperand::Immediate(hi as i64),
                        ];
                        let lui = self.encoder.encode_instruction(RvOpcode::LUI, &lui_ops)?;
                        self.current_section_mut()
                            .data
                            .extend_from_slice(&lui.to_bytes());
                        if lo != 0 {
                            let addi_ops = [
                                EncoderOperand::Register(rd),
                                EncoderOperand::Register(rd),
                                EncoderOperand::Immediate(lo as i64),
                            ];
                            let addi =
                                self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
                            self.current_section_mut()
                                .data
                                .extend_from_slice(&addi.to_bytes());
                        }
                    }
                    return Ok(true);
                }
            }
            "ld" | "lw" | "lh" | "lb" | "lbu" | "lhu" | "lwu" => {
                // ld rd, offset(rs)
                if operand_tokens.len() >= 2 {
                    if let Some((off, base)) = self.parse_asm_mem_operand(operand_tokens[1]) {
                        let rd = self.parse_asm_reg(operand_tokens[0]);
                        let opc = match mnemonic.as_str() {
                            "ld" => RvOpcode::LD,
                            "lw" => RvOpcode::LW,
                            "lh" => RvOpcode::LH,
                            "lb" => RvOpcode::LB,
                            "lbu" => RvOpcode::LBU,
                            "lhu" => RvOpcode::LHU,
                            "lwu" => RvOpcode::LWU,
                            _ => unreachable!(),
                        };
                        let ops = [
                            EncoderOperand::Register(rd),
                            EncoderOperand::Register(base),
                            EncoderOperand::Immediate(off),
                        ];
                        let encoded = self.encoder.encode_instruction(opc, &ops)?;
                        self.current_section_mut()
                            .data
                            .extend_from_slice(&encoded.to_bytes());
                        return Ok(true);
                    }
                }
            }
            "sd" | "sw" | "sh" | "sb" => {
                // sd rs2, offset(rs1)
                if operand_tokens.len() >= 2 {
                    if let Some((off, base)) = self.parse_asm_mem_operand(operand_tokens[1]) {
                        let rs2 = self.parse_asm_reg(operand_tokens[0]);
                        let opc = match mnemonic.as_str() {
                            "sd" => RvOpcode::SD,
                            "sw" => RvOpcode::SW,
                            "sh" => RvOpcode::SH,
                            "sb" => RvOpcode::SB,
                            _ => unreachable!(),
                        };
                        let ops = [
                            EncoderOperand::Register(rs2),
                            EncoderOperand::Register(base),
                            EncoderOperand::Immediate(off),
                        ];
                        let encoded = self.encoder.encode_instruction(opc, &ops)?;
                        self.current_section_mut()
                            .data
                            .extend_from_slice(&encoded.to_bytes());
                        return Ok(true);
                    }
                }
            }
            "slli" | "srli" | "srai" | "slliw" | "srliw" | "sraiw" => {
                if operand_tokens.len() >= 3 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs = self.parse_asm_reg(operand_tokens[1]);
                    let imm = self.parse_asm_imm(operand_tokens[2]);
                    let opc = match mnemonic.as_str() {
                        "slli" => RvOpcode::SLLI,
                        "srli" => RvOpcode::SRLI,
                        "srai" => RvOpcode::SRAI,
                        "slliw" => RvOpcode::SLLIW,
                        "srliw" => RvOpcode::SRLIW,
                        "sraiw" => RvOpcode::SRAIW,
                        _ => unreachable!(),
                    };
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs),
                        EncoderOperand::Immediate(imm),
                    ];
                    let encoded = self.encoder.encode_instruction(opc, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "and" | "or" | "xor" | "sll" | "srl" | "sra" => {
                if operand_tokens.len() >= 3 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs1 = self.parse_asm_reg(operand_tokens[1]);
                    let rs2 = self.parse_asm_reg(operand_tokens[2]);
                    let opc = match mnemonic.as_str() {
                        "and" => RvOpcode::AND,
                        "or" => RvOpcode::OR,
                        "xor" => RvOpcode::XOR,
                        "sll" => RvOpcode::SLL,
                        "srl" => RvOpcode::SRL,
                        "sra" => RvOpcode::SRA,
                        _ => unreachable!(),
                    };
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs1),
                        EncoderOperand::Register(rs2),
                    ];
                    let encoded = self.encoder.encode_instruction(opc, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "andi" | "ori" | "xori" => {
                if operand_tokens.len() >= 3 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs = self.parse_asm_reg(operand_tokens[1]);
                    let imm = self.parse_asm_imm(operand_tokens[2]);
                    let opc = match mnemonic.as_str() {
                        "andi" => RvOpcode::ANDI,
                        "ori" => RvOpcode::ORI,
                        "xori" => RvOpcode::XORI,
                        _ => unreachable!(),
                    };
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs),
                        EncoderOperand::Immediate(imm),
                    ];
                    let encoded = self.encoder.encode_instruction(opc, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            "nop" => {
                self.emit_nop();
                return Ok(true);
            }
            "ret" => {
                // ret = jalr x0, ra, 0
                let ops = [
                    EncoderOperand::Register(0), // x0
                    EncoderOperand::Register(1), // ra
                    EncoderOperand::Immediate(0),
                ];
                let encoded = self.encoder.encode_instruction(RvOpcode::JALR, &ops)?;
                self.current_section_mut()
                    .data
                    .extend_from_slice(&encoded.to_bytes());
                return Ok(true);
            }
            "sext.w" => {
                // sext.w rd, rs → addiw rd, rs, 0
                if operand_tokens.len() >= 2 {
                    let rd = self.parse_asm_reg(operand_tokens[0]);
                    let rs = self.parse_asm_reg(operand_tokens[1]);
                    let ops = [
                        EncoderOperand::Register(rd),
                        EncoderOperand::Register(rs),
                        EncoderOperand::Immediate(0),
                    ];
                    let encoded = self.encoder.encode_instruction(RvOpcode::ADDIW, &ops)?;
                    self.current_section_mut()
                        .data
                        .extend_from_slice(&encoded.to_bytes());
                    return Ok(true);
                }
            }
            _ => {}
        }

        // Unrecognised mnemonic.
        Ok(false)
    }

    /// Parses a register name from inline assembly text.
    /// Returns the register encoding (0–31).
    fn parse_asm_reg(&self, token: &str) -> u8 {
        let t = token.trim().trim_end_matches(',');
        match t {
            "zero" | "x0" => 0,
            "ra" | "x1" => 1,
            "sp" | "x2" => 2,
            "gp" | "x3" => 3,
            "tp" | "x4" => 4,
            "t0" | "x5" => 5,
            "t1" | "x6" => 6,
            "t2" | "x7" => 7,
            "s0" | "fp" | "x8" => 8,
            "s1" | "x9" => 9,
            "a0" | "x10" => 10,
            "a1" | "x11" => 11,
            "a2" | "x12" => 12,
            "a3" | "x13" => 13,
            "a4" | "x14" => 14,
            "a5" | "x15" => 15,
            "a6" | "x16" => 16,
            "a7" | "x17" => 17,
            "s2" | "x18" => 18,
            "s3" | "x19" => 19,
            "s4" | "x20" => 20,
            "s5" | "x21" => 21,
            "s6" | "x22" => 22,
            "s7" | "x23" => 23,
            "s8" | "x24" => 24,
            "s9" | "x25" => 25,
            "s10" | "x26" => 26,
            "s11" | "x27" => 27,
            "t3" | "x28" => 28,
            "t4" | "x29" => 29,
            "t5" | "x30" => 30,
            "t6" | "x31" => 31,
            _ => {
                // Try xN format
                if let Some(n) = t.strip_prefix('x') {
                    n.parse::<u8>().unwrap_or(0)
                } else {
                    0 // fallback to zero register
                }
            }
        }
    }

    /// Parses an immediate value from inline assembly text.
    fn parse_asm_imm(&self, token: &str) -> i64 {
        let t = token.trim().trim_end_matches(',');
        if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            i64::from_str_radix(hex, 16).unwrap_or(0)
        } else if let Some(neg_hex) = t.strip_prefix("-0x").or_else(|| t.strip_prefix("-0X")) {
            -(i64::from_str_radix(neg_hex, 16).unwrap_or(0))
        } else {
            t.parse::<i64>().unwrap_or(0)
        }
    }

    /// Parses a memory operand of the form `offset(reg)`.
    /// Returns `(offset, reg_encoding)`.
    fn parse_asm_mem_operand(&self, token: &str) -> Option<(i64, u8)> {
        let t = token.trim();
        if let Some(paren_pos) = t.find('(') {
            let offset_str = &t[..paren_pos];
            let reg_str = t[paren_pos + 1..].trim_end_matches(')');
            let offset = self.parse_asm_imm(offset_str);
            let reg = self.parse_asm_reg(reg_str);
            Some((offset, reg))
        } else {
            None
        }
    }

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

    /// Expands `FMOV.S rd, rs` → `FSGNJ.S rd, rs, rs`
    /// or      `FMOV.D rd, rs` → `FSGNJ.D rd, rs, rs`.
    ///
    /// RISC-V does not have a dedicated FP-move instruction; the canonical
    /// encoding uses sign-injection with both sources the same register.
    fn expand_fmov(
        &mut self,
        instr: &MachineInstr,
        fsgnj_op: RvOpcode,
    ) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;
        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rs),
            EncoderOperand::Register(rs),
        ];
        let encoded = self.encoder.encode_instruction(fsgnj_op, &ops)?;
        self.current_section_mut()
            .data
            .extend_from_slice(&encoded.to_bytes());
        Ok(())
    }

    /// Expands `FNEG.S rd, rs` → `FSGNJN.S rd, rs, rs`
    /// or      `FNEG.D rd, rs` → `FSGNJN.D rd, rs, rs`.
    ///
    /// Negation is achieved by injecting the negated sign of the source.
    fn expand_fneg(
        &mut self,
        instr: &MachineInstr,
        fsgnjn_op: RvOpcode,
    ) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;
        let rs = self.extract_register(&instr.operands, 1)?;
        let ops = [
            EncoderOperand::Register(rd),
            EncoderOperand::Register(rs),
            EncoderOperand::Register(rs),
        ];
        let encoded = self.encoder.encode_instruction(fsgnjn_op, &ops)?;
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
            // Check for forward label reference and record fixup if needed.
            let is_forward_label = matches!(
                instr.operands.first(),
                Some(MachineOperand::Label(id)) if !self.label_offsets.contains_key(id)
            );
            let forward_label_id = if is_forward_label {
                if let Some(MachineOperand::Label(id)) = instr.operands.first() {
                    Some(*id)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(label_id) = forward_label_id {
                let instr_offset = self.get_current_offset();
                self.branch_fixups.push(BranchFixup {
                    instr_offset,
                    target_label: label_id,
                    is_j_type: true,
                });
            }

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
        // Direct match on the codegen opcode constants defined in
        // `src/backend/riscv64/codegen.rs`.  Using a match instead of
        // an index-based array avoids fragile ordering dependencies.
        use crate::backend::riscv64::codegen::*;
        match opcode {
            // ---- R-type integer arithmetic ----
            RV_ADD => Ok(RvOpcode::ADD),
            RV_SUB => Ok(RvOpcode::SUB),
            RV_AND => Ok(RvOpcode::AND),
            RV_OR => Ok(RvOpcode::OR),
            RV_XOR => Ok(RvOpcode::XOR),
            RV_SLL => Ok(RvOpcode::SLL),
            RV_SRL => Ok(RvOpcode::SRL),
            RV_SRA => Ok(RvOpcode::SRA),
            RV_SLT => Ok(RvOpcode::SLT),
            RV_SLTU => Ok(RvOpcode::SLTU),
            // ---- R-type word (32-bit on RV64) ----
            RV_ADDW => Ok(RvOpcode::ADDW),
            RV_SUBW => Ok(RvOpcode::SUBW),
            RV_SLLW => Ok(RvOpcode::SLLW),
            RV_SRLW => Ok(RvOpcode::SRLW),
            RV_SRAW => Ok(RvOpcode::SRAW),
            // ---- M-extension multiply/divide 64-bit ----
            RV_MUL => Ok(RvOpcode::MUL),
            RV_MULH => Ok(RvOpcode::MULH),
            RV_MULHU => Ok(RvOpcode::MULHU),
            RV_MULHSU => Ok(RvOpcode::MULHSU),
            RV_DIV => Ok(RvOpcode::DIV),
            RV_DIVU => Ok(RvOpcode::DIVU),
            RV_REM => Ok(RvOpcode::REM),
            RV_REMU => Ok(RvOpcode::REMU),
            // ---- M-extension word ops ----
            RV_MULW => Ok(RvOpcode::MULW),
            RV_DIVW => Ok(RvOpcode::DIVW),
            RV_DIVUW => Ok(RvOpcode::DIVUW),
            RV_REMW => Ok(RvOpcode::REMW),
            RV_REMUW => Ok(RvOpcode::REMUW),
            // ---- I-type immediate arithmetic ----
            RV_ADDI => Ok(RvOpcode::ADDI),
            RV_ANDI => Ok(RvOpcode::ANDI),
            RV_ORI => Ok(RvOpcode::ORI),
            RV_XORI => Ok(RvOpcode::XORI),
            RV_SLTI => Ok(RvOpcode::SLTI),
            RV_SLTIU => Ok(RvOpcode::SLTIU),
            RV_ADDIW => Ok(RvOpcode::ADDIW),
            // ---- I-type immediate shifts ----
            RV_SLLI => Ok(RvOpcode::SLLI),
            RV_SRLI => Ok(RvOpcode::SRLI),
            RV_SRAI => Ok(RvOpcode::SRAI),
            // ---- Loads ----
            RV_LB => Ok(RvOpcode::LB),
            RV_LBU => Ok(RvOpcode::LBU),
            RV_LH => Ok(RvOpcode::LH),
            RV_LHU => Ok(RvOpcode::LHU),
            RV_LW => Ok(RvOpcode::LW),
            RV_LWU => Ok(RvOpcode::LWU),
            RV_LD => Ok(RvOpcode::LD),
            // ---- Stores ----
            RV_SB => Ok(RvOpcode::SB),
            RV_SH => Ok(RvOpcode::SH),
            RV_SW => Ok(RvOpcode::SW),
            RV_SD => Ok(RvOpcode::SD),
            // ---- Branches ----
            RV_BEQ => Ok(RvOpcode::BEQ),
            RV_BNE => Ok(RvOpcode::BNE),
            RV_BLT => Ok(RvOpcode::BLT),
            RV_BGE => Ok(RvOpcode::BGE),
            RV_BLTU => Ok(RvOpcode::BLTU),
            RV_BGEU => Ok(RvOpcode::BGEU),
            // ---- Upper immediate ----
            RV_LUI => Ok(RvOpcode::LUI),
            RV_AUIPC => Ok(RvOpcode::AUIPC),
            // ---- Jumps ----
            RV_JAL => Ok(RvOpcode::JAL),
            RV_JALR => Ok(RvOpcode::JALR),
            // ---- F-extension single-precision FP ----
            RV_FLW => Ok(RvOpcode::FLW),
            RV_FSW => Ok(RvOpcode::FSW),
            RV_FADD_S => Ok(RvOpcode::FADD_S),
            RV_FSUB_S => Ok(RvOpcode::FSUB_S),
            RV_FMUL_S => Ok(RvOpcode::FMUL_S),
            RV_FDIV_S => Ok(RvOpcode::FDIV_S),
            RV_FSQRT_S => Ok(RvOpcode::FSQRT_S),
            RV_FMIN_S => Ok(RvOpcode::FMIN_S),
            RV_FMAX_S => Ok(RvOpcode::FMAX_S),
            RV_FEQ_S => Ok(RvOpcode::FEQ_S),
            RV_FLT_S => Ok(RvOpcode::FLT_S),
            RV_FLE_S => Ok(RvOpcode::FLE_S),
            RV_FCLASS_S => Ok(RvOpcode::FCLASS_S),
            RV_FCVT_W_S => Ok(RvOpcode::FCVT_W_S),
            RV_FCVT_WU_S => Ok(RvOpcode::FCVT_WU_S),
            RV_FCVT_L_S => Ok(RvOpcode::FCVT_L_S),
            RV_FCVT_LU_S => Ok(RvOpcode::FCVT_LU_S),
            RV_FCVT_S_W => Ok(RvOpcode::FCVT_S_W),
            RV_FCVT_S_WU => Ok(RvOpcode::FCVT_S_WU),
            RV_FCVT_S_L => Ok(RvOpcode::FCVT_S_L),
            RV_FCVT_S_LU => Ok(RvOpcode::FCVT_S_LU),
            RV_FMV_X_W => Ok(RvOpcode::FMV_X_W),
            RV_FMV_W_X => Ok(RvOpcode::FMV_W_X),
            // ---- D-extension double-precision FP ----
            RV_FLD => Ok(RvOpcode::FLD),
            RV_FSD => Ok(RvOpcode::FSD),
            RV_FADD_D => Ok(RvOpcode::FADD_D),
            RV_FSUB_D => Ok(RvOpcode::FSUB_D),
            RV_FMUL_D => Ok(RvOpcode::FMUL_D),
            RV_FDIV_D => Ok(RvOpcode::FDIV_D),
            RV_FSQRT_D => Ok(RvOpcode::FSQRT_D),
            RV_FMIN_D => Ok(RvOpcode::FMIN_D),
            RV_FMAX_D => Ok(RvOpcode::FMAX_D),
            RV_FEQ_D => Ok(RvOpcode::FEQ_D),
            RV_FLT_D => Ok(RvOpcode::FLT_D),
            RV_FLE_D => Ok(RvOpcode::FLE_D),
            RV_FCLASS_D => Ok(RvOpcode::FCLASS_D),
            RV_FCVT_W_D => Ok(RvOpcode::FCVT_W_D),
            RV_FCVT_WU_D => Ok(RvOpcode::FCVT_WU_D),
            RV_FCVT_L_D => Ok(RvOpcode::FCVT_L_D),
            RV_FCVT_LU_D => Ok(RvOpcode::FCVT_LU_D),
            RV_FCVT_D_W => Ok(RvOpcode::FCVT_D_W),
            RV_FCVT_D_WU => Ok(RvOpcode::FCVT_D_WU),
            RV_FCVT_D_L => Ok(RvOpcode::FCVT_D_L),
            RV_FCVT_D_LU => Ok(RvOpcode::FCVT_D_LU),
            RV_FCVT_S_D => Ok(RvOpcode::FCVT_S_D),
            RV_FCVT_D_S => Ok(RvOpcode::FCVT_D_S),
            RV_FMV_X_D => Ok(RvOpcode::FMV_X_D),
            RV_FMV_D_X => Ok(RvOpcode::FMV_D_X),
            // ---- Pseudo-instructions ----
            RV_NOP => Ok(RvOpcode::NOP),
            RV_MV => Ok(RvOpcode::MV),
            RV_NEG => Ok(RvOpcode::NEG),
            RV_LI => Ok(RvOpcode::LI),
            RV_CALL => Ok(RvOpcode::CALL),
            RV_TAIL => Ok(RvOpcode::TAIL),
            RV_RET => Ok(RvOpcode::RET),
            RV_LA => Ok(RvOpcode::LA),
            RV_JR => Ok(RvOpcode::JR),
            // RV_FMOV_S, RV_FMOV_D, RV_FNEG_S, RV_FNEG_D are handled
            // before map_opcode is called (see emit_instruction).
            RV_INLINE_ASM => {
                // Inline ASM is handled earlier in emit_instruction; should
                // never reach map_opcode.
                Err(AssemblerError::UnsupportedInstruction(
                    "inline asm should not reach map_opcode".into(),
                ))
            }
            // ---- Spill pseudo-ops and PHI sentinel ----
            // SPILL_LOAD (0xFFFF_FFFF) and SPILL_STORE (0xFFFF_FFFE) are
            // intercepted in emit_instruction before map_opcode is called.
            // If they reach here, something went wrong.
            0xFFFF_FFFE | 0xFFFF_FFFF => Err(AssemblerError::UnsupportedInstruction(
                "spill/PHI pseudo-instruction reached map_opcode (should be handled earlier)"
                    .into(),
            )),
            _ => Err(AssemblerError::UnsupportedInstruction(format!(
                "unknown opcode {} in function '{}'",
                opcode, self.current_function
            ))),
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

    // -----------------------------------------------------------------------
    // Forward-label fixup helpers
    // -----------------------------------------------------------------------

    /// Returns the label ID of the first forward-reference `Label` operand,
    /// or `None` if all labels are already resolved (backward references).
    fn find_forward_label(&self, operands: &[MachineOperand]) -> Option<u32> {
        for op in operands {
            if let MachineOperand::Label(id) = op {
                if !self.label_offsets.contains_key(id) {
                    return Some(*id);
                }
            }
        }
        None
    }

    /// Like [`translate_operands`], but replaces any `Label` operand with
    /// `Immediate(0)` when the label is a forward reference (not yet in
    /// `label_offsets`). Backward-reference labels are resolved to their
    /// relative offset as usual.
    fn translate_operands_replacing_labels(
        &self,
        operands: &[MachineOperand],
    ) -> Result<Vec<EncoderOperand>, AssemblerError> {
        let mut result = Vec::with_capacity(operands.len());
        for op in operands {
            match op {
                MachineOperand::Label(id) => {
                    if let Some(&target_offset) = self.label_offsets.get(id) {
                        let current = self.get_current_offset();
                        let rel = target_offset as i64 - current as i64;
                        result.push(EncoderOperand::Immediate(rel));
                    } else {
                        // Forward reference — placeholder zero.
                        result.push(EncoderOperand::Immediate(0));
                    }
                }
                _ => {
                    result.push(self.translate_operand(op)?);
                }
            }
        }
        Ok(result)
    }

    /// Resolves all pending branch fixups for the current function.
    ///
    /// After all basic blocks have been assembled and all label offsets are
    /// known, this method patches each placeholder branch offset with the
    /// correct relative displacement.
    fn resolve_branch_fixups(&mut self) -> Result<(), AssemblerError> {
        for fixup in &self.branch_fixups {
            let target_offset = match self.label_offsets.get(&fixup.target_label) {
                Some(&off) => off,
                None => {
                    return Err(AssemblerError::EncodingError(format!(
                        "unresolved forward label {} in function '{}'",
                        fixup.target_label, self.current_function
                    )));
                }
            };

            let offset = target_offset as i64 - fixup.instr_offset as i64;
            let pos = fixup.instr_offset as usize;
            let data = &mut self.sections[self.current_section].data;

            if pos + 4 > data.len() {
                return Err(AssemblerError::EncodingError(format!(
                    "fixup at offset {} exceeds section size {}",
                    pos,
                    data.len()
                )));
            }

            // Read the existing instruction word (little-endian).
            let mut word =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);

            let imm = offset as i32;

            if fixup.is_j_type {
                // J-type (JAL) immediate encoding:
                //   [31]    = imm[20]
                //   [30:21] = imm[10:1]
                //   [20]    = imm[11]
                //   [19:12] = imm[19:12]
                //   [11:7]  = rd        (preserved)
                //   [6:0]   = opcode    (preserved)
                //
                // Clear bits [31:12], preserve [11:0] (rd + opcode).
                word &= 0x0000_0FFF;

                let bit20 = ((imm >> 20) & 1) as u32;
                let bits10_1 = ((imm >> 1) & 0x3FF) as u32;
                let bit11 = ((imm >> 11) & 1) as u32;
                let bits19_12 = ((imm >> 12) & 0xFF) as u32;

                word |= (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12);
            } else {
                // B-type (BEQ, BNE, …) immediate encoding:
                //   [31]    = imm[12]
                //   [30:25] = imm[10:5]
                //   [24:20] = rs2       (preserved)
                //   [19:15] = rs1       (preserved)
                //   [14:12] = funct3    (preserved)
                //   [11:8]  = imm[4:1]
                //   [7]     = imm[11]
                //   [6:0]   = opcode    (preserved)
                //
                // Clear bits [31:25] and [11:7], preserve the rest.
                word &= 0x01FF_F07F;

                let bit12 = ((imm >> 12) & 1) as u32;
                let bits10_5 = ((imm >> 5) & 0x3F) as u32;
                let bits4_1 = ((imm >> 1) & 0xF) as u32;
                let bit11 = ((imm >> 11) & 1) as u32;

                word |= (bit12 << 31) | (bits10_5 << 25) | (bits4_1 << 8) | (bit11 << 7);
            }

            let new_bytes = word.to_le_bytes();
            data[pos..pos + 4].copy_from_slice(&new_bytes);
        }
        Ok(())
    }

    /// Resolves label-address fixups (AUIPC+ADDI pairs for computed gotos).
    ///
    /// For each fixup, computes the PC-relative offset from the AUIPC
    /// instruction to the target label, then encodes the upper 20 bits
    /// in the AUIPC immediate and the lower 12 bits in the ADDI immediate.
    fn resolve_label_address_fixups(&mut self) -> Result<(), AssemblerError> {
        let fixups = std::mem::take(&mut self.label_address_fixups);
        for fixup in &fixups {
            let target_offset = match self.label_offsets.get(&fixup.target_label) {
                Some(&off) => off,
                None => {
                    return Err(AssemblerError::EncodingError(format!(
                        "unresolved label-address target {} in function '{}'",
                        fixup.target_label, self.current_function
                    )));
                }
            };

            // PC-relative offset from AUIPC instruction to target label.
            let offset = target_offset as i64 - fixup.auipc_offset as i64;
            let offset_i32 = offset as i32;

            // Split into hi20 and lo12 with sign adjustment.
            // If lo12 is negative (bit 11 set), we need to add 1 to hi20
            // because ADDI sign-extends and effectively subtracts from the
            // AUIPC base.
            let lo12 = ((offset_i32 << 20) >> 20) as i32; // sign-extend low 12
            let mut hi20 = (offset_i32 - lo12) >> 12;
            if lo12 < 0 {
                // Already handled by the subtraction above, but let's be safe
            }
            let _ = hi20; // suppress unused warning
            hi20 = (offset_i32.wrapping_add(0x800) >> 12) & 0xFFFFF;

            let pos_auipc = fixup.auipc_offset as usize;
            let pos_addi = pos_auipc + 4;
            let data = &mut self.sections[self.current_section].data;

            if pos_addi + 4 > data.len() {
                return Err(AssemblerError::EncodingError(format!(
                    "label-address fixup at offset {} exceeds section size {}",
                    pos_auipc,
                    data.len()
                )));
            }

            // Patch AUIPC: imm[31:12] = hi20, preserve rd[11:7] and opcode[6:0]
            let mut auipc_word = u32::from_le_bytes([
                data[pos_auipc],
                data[pos_auipc + 1],
                data[pos_auipc + 2],
                data[pos_auipc + 3],
            ]);
            auipc_word &= 0x0000_0FFF; // preserve rd + opcode (bits 11:0)
            auipc_word |= (hi20 as u32) << 12;
            data[pos_auipc..pos_auipc + 4].copy_from_slice(&auipc_word.to_le_bytes());

            // Patch ADDI: imm[31:20] = lo12, preserve rs1[19:15], funct3[14:12],
            //             rd[11:7], opcode[6:0]
            let lo12_u = (offset_i32 & 0xFFF) as u32;
            let mut addi_word = u32::from_le_bytes([
                data[pos_addi],
                data[pos_addi + 1],
                data[pos_addi + 2],
                data[pos_addi + 3],
            ]);
            addi_word &= 0x000F_FFFF; // preserve bits 19:0
            addi_word |= lo12_u << 20;
            data[pos_addi..pos_addi + 4].copy_from_slice(&addi_word.to_le_bytes());
        }
        Ok(())
    }

    /// Expands `RV_LA_LABEL rd, Label(bb_id)` into AUIPC+ADDI with a
    /// forward fixup for computed gotos.
    fn expand_la_label(&mut self, instr: &MachineInstr) -> Result<(), AssemblerError> {
        let rd = self.extract_register(&instr.operands, 0)?;

        // Extract the label ID from operand 1.
        let label_id = match &instr.operands.get(1) {
            Some(MachineOperand::Label(id)) => *id,
            Some(MachineOperand::Immediate(id)) => *id as u32,
            other => {
                return Err(AssemblerError::InvalidOperand(format!(
                    "LA_LABEL: expected Label operand at index 1, got {:?}",
                    other
                )));
            }
        };

        let auipc_offset = self.get_current_offset();

        // Check if the label is already known (backward reference).
        if let Some(&target_off) = self.label_offsets.get(&label_id) {
            // Already resolved — compute offset and encode directly.
            let offset = target_off as i64 - auipc_offset as i64;
            let offset_i32 = offset as i32;
            let hi20 = ((offset_i32.wrapping_add(0x800)) >> 12) & 0xFFFFF;
            let lo12 = offset_i32 & 0xFFF;

            let auipc_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Immediate(hi20 as i64),
            ];
            let auipc = self
                .encoder
                .encode_instruction(RvOpcode::AUIPC, &auipc_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&auipc.to_bytes());

            let addi_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Register(rd),
                EncoderOperand::Immediate(lo12 as i64),
            ];
            let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&addi.to_bytes());
        } else {
            // Forward reference — emit placeholder AUIPC+ADDI and record fixup.
            let auipc_ops = [EncoderOperand::Register(rd), EncoderOperand::Immediate(0)];
            let auipc = self
                .encoder
                .encode_instruction(RvOpcode::AUIPC, &auipc_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&auipc.to_bytes());

            let addi_ops = [
                EncoderOperand::Register(rd),
                EncoderOperand::Register(rd),
                EncoderOperand::Immediate(0),
            ];
            let addi = self.encoder.encode_instruction(RvOpcode::ADDI, &addi_ops)?;
            self.current_section_mut()
                .data
                .extend_from_slice(&addi.to_bytes());

            self.label_address_fixups.push(LabelAddressFixup {
                auipc_offset,
                target_label: label_id,
            });
        }

        Ok(())
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
    use crate::backend::riscv64::codegen::RV_RET;
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
        // RET pseudo-instruction.
        let mut ret_instr = MachineInstr::new(RV_RET);
        ret_instr.set_return();
        bb.push_instr(ret_instr);
        mf.add_block(bb);

        let mut asm = RiscV64Assembler::new();
        let result = asm.assemble_function(&mf);
        assert!(
            result.is_ok(),
            "assemble_function failed: {:?}",
            result.err()
        );

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
