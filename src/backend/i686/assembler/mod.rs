//! # BCC i686 Built-in Assembler
//!
//! This module implements the built-in i686 (32-bit x86) assembler for BCC.
//! It encodes i686 machine instructions directly into raw bytes without
//! invoking any external assembler (`as`, `nasm`, `llvm-mc`).
//!
//! ## Architecture
//!
//! - **Two-pass assembly**: Pass 1 encodes all instructions with preliminary
//!   label offsets to determine exact instruction sizes and compute final
//!   label positions.  Pass 2 re-encodes with the correct label offsets to
//!   produce final machine code with accurate branch displacements.
//! - **Relocation collection**: External or global symbol references that
//!   cannot be resolved within the function produce `R_386_*` relocation
//!   entries for the built-in linker.
//! - **32-bit encoding**: No REX prefix, no RIP-relative addressing — purely
//!   IA-32 ISA with register indices 0–7 in ModR/M bytes.
//!
//! ## Submodules
//!
//! - [`encoder`] — Instruction-level encoding: ModR/M, SIB, opcodes, prefixes
//! - [`relocations`] — i686 ELF relocation type definitions (`R_386_*`)
//!
//! ## Zero-Dependency
//!
//! Only `crate::` and `std::` imports.  No external crates.

pub mod encoder;
pub mod relocations;

use crate::backend::elf_writer_common::Relocation;
use crate::backend::i686::codegen;
use crate::backend::i686::registers;
use crate::backend::traits::{MachineFunction, MachineInstruction, MachineOperand};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// AssemblyOutput — Public Output Structure
// ===========================================================================

/// Output from the i686 assembler: encoded machine code and relocation entries.
///
/// Produced by [`assemble`] after a successful two-pass assembly of a
/// [`MachineFunction`].  Contains:
///
/// - **`code`**: Raw machine code bytes, ready for inclusion in an ELF
///   `.text` section.  All internal branch displacements are resolved.
/// - **`relocations`**: ELF relocation entries (`R_386_*` types) for
///   external symbol references that require linker resolution.
///   [`Relocation::sym_index`] values are provisional sequential indices
///   assigned during assembly; the caller must reconcile these with the
///   final ELF symbol table.
/// - **`label_offsets`**: Maps label names to byte offsets within `code`.
///   Includes block-index labels (`.L0`, `.L1`, …) and the function
///   entry label (the function name).
#[derive(Debug)]
pub struct AssemblyOutput {
    /// Raw machine code bytes for the entire function.
    pub code: Vec<u8>,
    /// Relocation entries for symbols that need linker resolution.
    pub relocations: Vec<Relocation>,
    /// Map from label name to byte offset within the code buffer.
    pub label_offsets: FxHashMap<String, usize>,
}

// ===========================================================================
// BranchFixup — Deferred Branch Resolution
// ===========================================================================

/// A branch or jump displacement that requires post-encoding fixup.
///
/// The two-pass assembly scheme resolves most branches via the `label_offsets`
/// map passed to the encoder.  `BranchFixup` captures any remaining fixups
/// that need patching after all code has been emitted — for example, branches
/// to external labels not present in `label_offsets`.
struct BranchFixup {
    /// Byte offset in the code buffer where the displacement is written.
    code_offset: usize,
    /// Target label name (e.g., `.L3` or an external symbol).
    target_label: String,
    /// `true` for PC-relative (JMP, Jcc), `false` for absolute.
    is_relative: bool,
    /// Size of the displacement field in bytes (1 for short, 4 for near).
    disp_size: u8,
}

// ===========================================================================
// I686Assembler — Internal Assembler State
// ===========================================================================

/// Internal assembler state machine for the i686 two-pass assembly process.
///
/// Created per-function by [`assemble`] and consumed to produce an
/// [`AssemblyOutput`].  Tracks the code buffer, relocation entries,
/// label-to-offset mappings, and provisional symbol indices across both
/// assembly passes.
struct I686Assembler {
    /// Accumulated machine code bytes (populated during pass 2).
    code: Vec<u8>,
    /// Current byte offset in the code buffer.
    current_offset: usize,
    /// Map: label name → byte offset.  Populated with preliminary values
    /// before pass 1 and updated to exact values during pass 1 iteration.
    label_offsets: FxHashMap<String, usize>,
    /// Collected relocation entries for the linker.
    relocations: Vec<Relocation>,
    /// Deferred branch fixups — resolved after pass 2 encoding.
    fixups: Vec<BranchFixup>,
    /// Map: external symbol name → provisional symbol table index.
    /// Each unique external symbol gets a sequential index starting from 1.
    symbol_index_map: FxHashMap<String, u32>,
    /// Next available provisional symbol index (0 is reserved for the
    /// ELF undefined/null symbol).
    next_symbol_index: u32,
}

// ===========================================================================
// Public Entry Point
// ===========================================================================

/// Assemble an i686 [`MachineFunction`] into raw bytes and relocation entries.
///
/// Uses a two-pass approach:
/// - **Pass 1**: Encode all instructions with preliminary label offsets to
///   compute exact instruction sizes and determine final label positions.
/// - **Pass 2**: Re-encode with correct label offsets to produce final
///   machine code with accurate branch displacements.
///
/// # Parameters
///
/// - `mf`: The machine function to assemble.  Must contain valid i686
///   instructions (opcodes from [`codegen`]) with physical registers
///   (indices 0–7) — no virtual registers should remain after register
///   allocation.
///
/// # Returns
///
/// - `Ok(AssemblyOutput)` on success.
/// - `Err(String)` if any instruction fails to encode (e.g., invalid
///   operand combinations, unknown opcodes, register index > 7).
///
/// # Example
///
/// ```ignore
/// use bcc::backend::i686::assembler::assemble;
/// use bcc::backend::traits::MachineFunction;
///
/// let mf = MachineFunction::new("my_func".to_string());
/// // ... populate blocks and instructions ...
/// let output = assemble(&mf).expect("assembly failed");
/// assert!(output.label_offsets.contains_key("my_func"));
/// ```
pub fn assemble(mf: &MachineFunction) -> Result<AssemblyOutput, String> {
    let mut assembler = I686Assembler::new();
    assembler.assemble_function(mf)
}

// ===========================================================================
// I686Assembler Implementation
// ===========================================================================

impl I686Assembler {
    /// Create a new assembler instance with empty state.
    fn new() -> Self {
        I686Assembler {
            code: Vec::with_capacity(4096),
            current_offset: 0,
            label_offsets: FxHashMap::default(),
            relocations: Vec::new(),
            fixups: Vec::new(),
            symbol_index_map: FxHashMap::default(),
            next_symbol_index: 1,
        }
    }

    /// Run the complete two-pass assembly pipeline for a function.
    ///
    /// 1. Pre-populate all labels with offset 0 (so forward references
    ///    produce valid — though incorrect — displacements in pass 1).
    /// 2. Pass 1: encode for exact sizes and compute final label offsets.
    /// 3. Pass 2: encode with correct label offsets; collect bytes and
    ///    relocation entries.
    /// 4. Resolve any deferred branch fixups.
    fn assemble_function(&mut self, mf: &MachineFunction) -> Result<AssemblyOutput, String> {
        // Validate the function targets the i686 architecture.
        self.validate_function(mf)?;

        // Pre-populate all labels at offset 0 so forward references in
        // pass 1 find a valid entry (producing correct-size encodings
        // despite wrong displacement values).
        self.pre_populate_labels(mf);

        // Pass 1: encode all instructions (for exact size computation).
        // Discard the encoded bytes — only accumulate label offsets.
        self.pass1_compute_sizes(mf)?;

        // Record the total function size for code buffer pre-allocation.
        let total_size = self.current_offset;

        // Reset state for pass 2.
        self.code.reserve(total_size);
        self.current_offset = 0;

        // Pass 2: final encoding with correct label offsets.
        self.pass2_encode(mf)?;

        // Resolve any deferred branch fixups.
        self.resolve_fixups()?;

        Ok(AssemblyOutput {
            code: std::mem::take(&mut self.code),
            relocations: std::mem::take(&mut self.relocations),
            label_offsets: std::mem::take(&mut self.label_offsets),
        })
    }

    // -----------------------------------------------------------------------
    // Pre-population and Validation
    // -----------------------------------------------------------------------

    /// Pre-populate `label_offsets` with all labels at offset 0.
    ///
    /// This ensures that the encoder's label lookup in pass 1 never fails
    /// for intra-function branches, even when the target block has not yet
    /// been visited.  The displacement value will be wrong, but the
    /// instruction SIZE is correct (since i686 always uses 32-bit near
    /// displacements for direct branches).
    fn pre_populate_labels(&mut self, mf: &MachineFunction) {
        // Function entry label.
        self.label_offsets.insert(mf.name.clone(), 0);

        // Block-index labels and string labels for every block.
        for (block_idx, block) in mf.blocks.iter().enumerate() {
            let block_key = format!(".L{}", block_idx);
            self.label_offsets.insert(block_key, 0);

            if let Some(ref label) = block.label {
                self.label_offsets.insert(label.clone(), 0);
            }
        }
    }

    /// Perform pre-assembly validation of the machine function.
    ///
    /// Checks that register operands are within the i686 GPR range (0–7)
    /// or are valid FPU registers (100–107), and logs warnings for potential
    /// issues like unreachable blocks or missing terminators.
    fn validate_function(&self, mf: &MachineFunction) -> Result<(), String> {
        // Verify this is an i686 function (architecture sanity check).
        // The i686 target has 4-byte pointers and page-aligned frames.
        let _target = Target::I686;

        // Validate register operands in each instruction.
        for (block_idx, block) in mf.blocks.iter().enumerate() {
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                // Validate all input operands.
                for op in &instr.operands {
                    self.validate_operand(op, &mf.name, block_idx, instr_idx)?;
                }
                // Validate result operand if present.
                if let Some(ref result) = instr.result {
                    self.validate_operand(result, &mf.name, block_idx, instr_idx)?;
                }
            }
        }

        // Validate callee-saved registers are within GPR range.
        for &reg in &mf.callee_saved_regs {
            if !registers::is_gpr(reg) && !registers::is_fpu_reg(reg) {
                return Err(format!(
                    "i686 assembler: callee-saved register {} is not a valid GPR or FPU reg \
                     in function '{}'",
                    reg, mf.name
                ));
            }
        }

        Ok(())
    }

    /// Validate a single machine operand for i686 constraints.
    ///
    /// Ensures register indices are within the i686 range (0–7 for GPRs,
    /// 100–107 for x87 FPU), and that memory operands use valid base/index
    /// registers.
    fn validate_operand(
        &self,
        op: &MachineOperand,
        func_name: &str,
        block_idx: usize,
        instr_idx: usize,
    ) -> Result<(), String> {
        match op {
            MachineOperand::Register(r) => {
                if !registers::is_gpr(*r) && !registers::is_fpu_reg(*r) {
                    return Err(format!(
                        "i686 assembler: register {} ({}) out of range in \
                         function '{}' block {} instruction {} — \
                         i686 has 8 GPRs (0–7) and 8 FPU regs (100–107), no REX prefix",
                        r,
                        registers::reg_name_32(*r),
                        func_name,
                        block_idx,
                        instr_idx,
                    ));
                }
            }
            MachineOperand::Memory { base, index, .. } => {
                if let Some(b) = base {
                    if !registers::is_gpr(*b) {
                        return Err(format!(
                            "i686 assembler: memory base register {} out of GPR range \
                             in function '{}' block {} instruction {}",
                            b, func_name, block_idx, instr_idx,
                        ));
                    }
                }
                if let Some(i) = index {
                    // ESP (4) cannot be used as an index register on i686.
                    if !registers::is_gpr(*i) || *i == registers::ESP {
                        return Err(format!(
                            "i686 assembler: memory index register {} invalid \
                             (ESP cannot be index) in function '{}' block {} instruction {}",
                            i, func_name, block_idx, instr_idx,
                        ));
                    }
                }
            }
            MachineOperand::FrameSlot(_)
            | MachineOperand::Immediate(_)
            | MachineOperand::GlobalSymbol(_)
            | MachineOperand::BlockLabel(_)
            | MachineOperand::VirtualRegister(_) => {}
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pass 1 — Size Computation
    // -----------------------------------------------------------------------

    /// Pass 1: encode all instructions to determine exact sizes and compute
    /// final label offsets.
    ///
    /// Iterates through every block and instruction, calling the encoder to
    /// produce encoded bytes (which are discarded — only their length is
    /// retained).  As each block is visited, its label offset is updated
    /// from the preliminary value (0) to the actual byte position.
    ///
    /// After this pass, `self.label_offsets` contains the correct byte offset
    /// for every label, and `self.current_offset` is the total function size.
    fn pass1_compute_sizes(&mut self, mf: &MachineFunction) -> Result<(), String> {
        self.current_offset = 0;

        for (block_idx, block) in mf.blocks.iter().enumerate() {
            // Update this block's labels to the actual offset.
            let block_key = format!(".L{}", block_idx);
            self.label_offsets.insert(block_key, self.current_offset);

            if let Some(ref label) = block.label {
                self.label_offsets
                    .insert(label.clone(), self.current_offset);
            }

            // Encode each instruction to get its exact size.
            for instr in &block.instructions {
                let encoded =
                    encoder::encode_instruction(instr, &self.label_offsets, self.current_offset)
                        .map_err(|e| {
                            format!(
                                "i686 assembler pass 1: encoding error at offset {} in '{}': {}",
                                self.current_offset, mf.name, e
                            )
                        })?;

                // Advance by the actual encoded size (not worst-case).
                self.current_offset += encoded.bytes.len();
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pass 2 — Final Encoding
    // -----------------------------------------------------------------------

    /// Pass 2: encode all instructions with correct label offsets and
    /// collect the final machine code bytes and relocation entries.
    ///
    /// All labels in `self.label_offsets` are now at their final byte
    /// positions (computed in pass 1), so branch displacements are accurate.
    /// External symbol references produce [`Relocation`] entries with
    /// `R_386_*` types.
    fn pass2_encode(&mut self, mf: &MachineFunction) -> Result<(), String> {
        self.current_offset = 0;

        if std::env::var("BCC_DEBUG_ASM").is_ok() {
            eprintln!("[i686-asm] pass2 label_offsets:");
            let mut labels: Vec<_> = self.label_offsets.iter().collect();
            labels.sort_by_key(|(_, v)| **v);
            for (k, v) in &labels {
                eprintln!("  {} = 0x{:x}", k, v);
            }
        }

        for (block_idx, block) in mf.blocks.iter().enumerate() {
            if std::env::var("BCC_DEBUG_ASM").is_ok() {
                eprintln!(
                    "[i686-asm] pass2 block {} (label={:?}) at offset 0x{:x}",
                    block_idx, block.label, self.current_offset
                );
            }
            for instr in &block.instructions {
                let instr_start = self.current_offset;

                // Encode the instruction using the encoder.
                let encoded =
                    encoder::encode_instruction(instr, &self.label_offsets, self.current_offset)
                        .map_err(|e| {
                            format!(
                                "i686 assembler pass 2: encoding error at offset {} in '{}': {}",
                                self.current_offset, mf.name, e
                            )
                        })?;

                // Process any relocation from the encoder.
                if let Some(instr_reloc) = encoded.relocation {
                    let global_offset =
                        instr_start as u64 + instr_reloc.offset_in_instruction as u64;
                    let sym_idx = self.get_or_assign_symbol_index(&instr_reloc.symbol);

                    self.relocations.push(Relocation {
                        offset: global_offset,
                        sym_index: sym_idx,
                        rel_type: instr_reloc.rel_type,
                        addend: instr_reloc.addend,
                    });
                }

                // Check for branch/call instructions that may need fixup
                // for targets not resolved by the encoder (safety net).
                self.check_branch_for_fixup(instr, instr_start, encoded.bytes.len());

                // Append the encoded bytes to the code buffer.
                let encoded_len = encoded.bytes.len();
                self.code.extend_from_slice(&encoded.bytes);
                self.current_offset += encoded_len;
            }
        }
        Ok(())
    }

    /// Check if a branch/call instruction needs a deferred fixup.
    ///
    /// This is a safety-net mechanism: the encoder resolves most branches
    /// using the `label_offsets` map.  If an instruction targets a label
    /// that the encoder could not resolve (e.g., an external function), and
    /// no relocation was emitted by the encoder, this function records a
    /// [`BranchFixup`] for post-encoding resolution.
    fn check_branch_for_fixup(
        &mut self,
        instr: &MachineInstruction,
        instr_start: usize,
        instr_len: usize,
    ) {
        // Only examine branch and call instructions.
        if !instr.is_branch && !instr.is_call {
            return;
        }

        // Identify the branch target operand.
        let is_jmp = instr.opcode == codegen::I686_JMP;
        let is_jcc = instr.opcode == codegen::I686_JCC;
        let is_call = instr.opcode == codegen::I686_CALL;

        if !is_jmp && !is_jcc && !is_call {
            return;
        }

        for op in &instr.operands {
            match op {
                MachineOperand::GlobalSymbol(ref name) => {
                    // External symbol — if no relocation was already emitted
                    // by the encoder, record a fixup.
                    if !self.label_offsets.contains_key(name) {
                        // Determine the displacement field offset within
                        // the instruction:
                        // - JMP E9 rel32 → displacement at byte 1
                        // - JCC 0F 8x rel32 → displacement at byte 2
                        // - CALL E8 rel32 → displacement at byte 1
                        let disp_offset_in_instr = if is_jcc { 2 } else { 1 };
                        let code_offset = instr_start + disp_offset_in_instr;

                        if code_offset + 4 <= instr_start + instr_len {
                            self.fixups.push(BranchFixup {
                                code_offset,
                                target_label: name.clone(),
                                is_relative: true,
                                disp_size: 4,
                            });
                        }
                    }
                }
                MachineOperand::BlockLabel(id) => {
                    let target_label = format!(".L{}", id);
                    if !self.label_offsets.contains_key(&target_label) {
                        let disp_offset_in_instr = if is_jcc { 2 } else { 1 };
                        let code_offset = instr_start + disp_offset_in_instr;

                        if code_offset + 4 <= instr_start + instr_len {
                            self.fixups.push(BranchFixup {
                                code_offset,
                                target_label,
                                is_relative: true,
                                disp_size: 4,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // -----------------------------------------------------------------------
    // Branch Fixup Resolution
    // -----------------------------------------------------------------------

    /// Resolve deferred branch fixups by patching displacement bytes in
    /// the code buffer.
    ///
    /// For each pending [`BranchFixup`]:
    /// - If the target label is in `label_offsets`, compute and patch the
    ///   displacement.
    /// - If the target label is external (not in `label_offsets`), create
    ///   a [`Relocation`] entry using the appropriate `R_386_*` type.
    fn resolve_fixups(&mut self) -> Result<(), String> {
        // Take ownership of the fixups vector to avoid borrow-checker
        // conflicts (iterating &self.fixups while mutating self.code
        // and self.relocations is not allowed).
        let fixups = std::mem::take(&mut self.fixups);

        for fixup in &fixups {
            if let Some(&target_offset) = self.label_offsets.get(&fixup.target_label) {
                // Local label found — patch the displacement in the code.
                if fixup.is_relative {
                    // PC-relative: disp = target - (fixup_addr + disp_size)
                    let pc_after = fixup.code_offset as i64 + fixup.disp_size as i64;
                    let displacement = target_offset as i64 - pc_after;

                    if fixup.disp_size == 4 {
                        let disp_bytes = (displacement as i32).to_le_bytes();
                        self.patch_code(fixup.code_offset, &disp_bytes)?;
                    } else if fixup.disp_size == 1 {
                        if !(-128..=127).contains(&displacement) {
                            return Err(format!(
                                "i686 assembler: short branch displacement {} \
                                 out of range [-128, 127] for label '{}'",
                                displacement, fixup.target_label
                            ));
                        }
                        self.patch_code(fixup.code_offset, &[displacement as i8 as u8])?;
                    }
                } else {
                    // Absolute: write the target address directly.
                    if fixup.disp_size == 4 {
                        let addr_bytes = (target_offset as u32).to_le_bytes();
                        self.patch_code(fixup.code_offset, &addr_bytes)?;
                    }
                }
            } else {
                // External label — emit a relocation entry.
                let sym_idx = self.get_or_assign_symbol_index(&fixup.target_label);

                let rel_type = if fixup.is_relative {
                    // PC-relative calls typically use R_386_PC32; the linker
                    // may upgrade to R_386_PLT32 for PIC shared libraries.
                    if relocations::is_pc_relative(relocations::R_386_PC32) {
                        relocations::R_386_PC32
                    } else {
                        relocations::R_386_32
                    }
                } else {
                    relocations::R_386_32
                };

                // Addend accounts for the displacement field's position
                // relative to the next instruction.
                let addend = if fixup.is_relative {
                    -(fixup.disp_size as i64)
                } else {
                    0
                };

                self.relocations.push(Relocation {
                    offset: fixup.code_offset as u64,
                    sym_index: sym_idx,
                    rel_type,
                    addend,
                });
            }
        }
        Ok(())
    }

    /// Patch bytes into the code buffer at the given offset.
    fn patch_code(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        if offset + bytes.len() > self.code.len() {
            return Err(format!(
                "i686 assembler: patch offset {}+{} exceeds code length {}",
                offset,
                bytes.len(),
                self.code.len()
            ));
        }
        self.code[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Symbol Index Management
    // -----------------------------------------------------------------------

    /// Get or assign a provisional ELF symbol table index for a symbol.
    ///
    /// Each unique symbol name receives a sequential index starting from 1
    /// (index 0 is reserved for the undefined/null ELF symbol entry).
    /// These provisional indices are written into [`Relocation::sym_index`]
    /// and must be reconciled with the final ELF symbol table by the caller
    /// (typically the linker or object-file writer).
    fn get_or_assign_symbol_index(&mut self, name: &str) -> u32 {
        if let Some(&idx) = self.symbol_index_map.get(name) {
            idx
        } else {
            let idx = self.next_symbol_index;
            self.symbol_index_map.insert(name.to_string(), idx);
            self.next_symbol_index += 1;
            idx
        }
    }

    // -----------------------------------------------------------------------
    // Relocation Classification Helpers
    // -----------------------------------------------------------------------

    /// Determine the appropriate `R_386_*` relocation type for an external
    /// symbol reference based on the instruction context.
    ///
    /// - **Direct CALL to a function**: `R_386_PC32` (or `R_386_PLT32` in
    ///   PIC mode for shared-library calls through the PLT).
    /// - **Absolute data reference**: `R_386_32` for loading a symbol's
    ///   address.
    /// - **GOT-relative access**: `R_386_GOT32` for PIC data access.
    /// - **GOT base computation**: `R_386_GOTPC` for `__x86.get_pc_thunk.*`.
    /// - **GOT-base-relative offset**: `R_386_GOTOFF` for offsets from EBX
    ///   in PIC mode.
    #[allow(dead_code)]
    fn classify_relocation_type(
        &self,
        instr: &MachineInstruction,
        symbol: &str,
        _is_pic: bool,
    ) -> u32 {
        // Calls use PC-relative relocations.
        if instr.is_call || instr.opcode == codegen::I686_CALL {
            return relocations::R_386_PC32;
        }

        // Jump instructions use PC-relative relocations.
        if instr.is_branch || instr.opcode == codegen::I686_JMP || instr.opcode == codegen::I686_JCC
        {
            return relocations::R_386_PC32;
        }

        // Check for GOT-related symbols (PIC mode patterns).
        if symbol.starts_with("_GLOBAL_OFFSET_TABLE_") {
            return relocations::R_386_GOTPC;
        }
        if symbol.ends_with("@GOT") {
            return relocations::R_386_GOT32;
        }
        if symbol.ends_with("@GOTOFF") {
            return relocations::R_386_GOTOFF;
        }
        if symbol.ends_with("@PLT") {
            return relocations::R_386_PLT32;
        }

        // Default: absolute 32-bit relocation.
        relocations::R_386_32
    }

    // -----------------------------------------------------------------------
    // Alignment and NOP Padding
    // -----------------------------------------------------------------------

    /// Emit NOP padding to align the current offset to the given boundary.
    ///
    /// Uses Intel-recommended multi-byte NOP sequences for minimal
    /// instruction-count padding on i686.
    ///
    /// # Parameters
    ///
    /// - `alignment`: Target alignment in bytes (must be a power of 2).
    ///   If zero or the current offset is already aligned, no padding
    ///   is emitted.
    #[allow(dead_code)]
    pub fn emit_alignment(&mut self, alignment: usize) {
        if alignment <= 1 {
            return;
        }
        let remainder = self.current_offset % alignment;
        if remainder == 0 {
            return;
        }
        let padding = alignment - remainder;
        self.emit_nop_sequence(padding);
    }

    /// Emit a NOP sequence of exactly `count` bytes.
    ///
    /// Uses optimal multi-byte NOP encodings (Intel-recommended for i686)
    /// to minimize the number of fetched instructions:
    ///
    /// | Bytes | Encoding                             |
    /// |-------|--------------------------------------|
    /// | 1     | `90`                                 |
    /// | 2     | `66 90`                              |
    /// | 3     | `0F 1F 00`                           |
    /// | 4     | `0F 1F 40 00`                        |
    /// | 5     | `0F 1F 44 00 00`                     |
    /// | 6     | `66 0F 1F 44 00 00`                  |
    /// | 7     | `0F 1F 80 00 00 00 00`               |
    #[allow(dead_code)]
    pub fn emit_nop_sequence(&mut self, mut count: usize) {
        static NOP_1: [u8; 1] = [0x90];
        static NOP_2: [u8; 2] = [0x66, 0x90];
        static NOP_3: [u8; 3] = [0x0F, 0x1F, 0x00];
        static NOP_4: [u8; 4] = [0x0F, 0x1F, 0x40, 0x00];
        static NOP_5: [u8; 5] = [0x0F, 0x1F, 0x44, 0x00, 0x00];
        static NOP_6: [u8; 6] = [0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00];
        static NOP_7: [u8; 7] = [0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00];

        while count > 0 {
            let nop: &[u8] = match count {
                1 => &NOP_1,
                2 => &NOP_2,
                3 => &NOP_3,
                4 => &NOP_4,
                5 => &NOP_5,
                6 => &NOP_6,
                _ => &NOP_7,
            };
            self.code.extend_from_slice(nop);
            self.current_offset += nop.len();
            count -= nop.len();
        }
    }

    // -----------------------------------------------------------------------
    // Data Emission Helpers
    // -----------------------------------------------------------------------

    /// Append raw bytes to the code buffer.
    ///
    /// Used for inline assembly data, `.byte` directives, and raw data
    /// embedding.
    #[allow(dead_code)]
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
        self.current_offset += bytes.len();
    }

    /// Emit a 32-bit little-endian value to the code buffer.
    #[allow(dead_code)]
    pub fn emit_u32_le(&mut self, value: u32) {
        self.code.extend_from_slice(&value.to_le_bytes());
        self.current_offset += 4;
    }

    /// Emit a 16-bit little-endian value to the code buffer.
    #[allow(dead_code)]
    pub fn emit_u16_le(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
        self.current_offset += 2;
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::{
        MachineBasicBlock, MachineFunction, MachineInstruction, MachineOperand,
    };

    /// Assembling an empty function produces an empty code buffer.
    #[test]
    fn test_assemble_empty_function() {
        let mf = MachineFunction::new("empty_fn".to_string());
        let output = assemble(&mf).unwrap();
        assert!(output.code.is_empty());
        assert!(output.relocations.is_empty());
        assert!(output.label_offsets.contains_key("empty_fn"));
        assert!(output.label_offsets.contains_key(".L0"));
        assert_eq!(*output.label_offsets.get("empty_fn").unwrap(), 0);
    }

    /// A single RET instruction assembles to valid machine code.
    #[test]
    fn test_assemble_ret_only() {
        let mut mf = MachineFunction::new("ret_fn".to_string());
        let ret = MachineInstruction::new(codegen::I686_RET).set_terminator();
        mf.blocks[0].push_instruction(ret);

        let output = assemble(&mf).unwrap();
        assert!(!output.code.is_empty());
        // RET is 0xC3 (1 byte) or 0xC2 imm16 (3 bytes).
        assert!(output.code.len() <= 3);
        assert!(output.label_offsets.contains_key("ret_fn"));
    }

    /// NOP instruction encodes as 0x90.
    #[test]
    fn test_assemble_nop() {
        let mut mf = MachineFunction::new("nop_fn".to_string());
        let nop = MachineInstruction::new(codegen::I686_NOP);
        let ret = MachineInstruction::new(codegen::I686_RET).set_terminator();
        mf.blocks[0].push_instruction(nop);
        mf.blocks[0].push_instruction(ret);

        let output = assemble(&mf).unwrap();
        assert!(output.code.len() >= 2);
        assert_eq!(output.code[0], 0x90);
    }

    /// Multi-byte NOP sequences use correct encodings.
    #[test]
    fn test_nop_sequence_encodings() {
        let mut asm = I686Assembler::new();

        // 1-byte NOP
        asm.emit_nop_sequence(1);
        assert_eq!(asm.code, vec![0x90]);
        assert_eq!(asm.current_offset, 1);

        // 7-byte NOP
        asm.code.clear();
        asm.current_offset = 0;
        asm.emit_nop_sequence(7);
        assert_eq!(asm.code.len(), 7);
        assert_eq!(asm.code[0], 0x0F);
        assert_eq!(asm.code[1], 0x1F);
        assert_eq!(asm.code[2], 0x80);

        // 10-byte NOP = 7 + 3
        asm.code.clear();
        asm.current_offset = 0;
        asm.emit_nop_sequence(10);
        assert_eq!(asm.code.len(), 10);
    }

    /// Alignment padding pads to the correct boundary.
    #[test]
    fn test_emit_alignment() {
        let mut asm = I686Assembler::new();
        asm.current_offset = 5;
        asm.emit_alignment(16);
        assert_eq!(asm.current_offset, 16);
        assert_eq!(asm.code.len(), 11); // 16 - 5 = 11 bytes of NOPs

        // Already aligned — no padding.
        let prev_len = asm.code.len();
        asm.emit_alignment(16);
        assert_eq!(asm.code.len(), prev_len);
    }

    /// Data emission helpers produce correct little-endian bytes.
    #[test]
    fn test_data_emission_helpers() {
        let mut asm = I686Assembler::new();

        asm.emit_u32_le(0xDEAD_BEEF);
        assert_eq!(&asm.code[0..4], &[0xEF, 0xBE, 0xAD, 0xDE]);
        assert_eq!(asm.current_offset, 4);

        asm.emit_u16_le(0x1234);
        assert_eq!(&asm.code[4..6], &[0x34, 0x12]);
        assert_eq!(asm.current_offset, 6);

        asm.emit_bytes(&[0xAA, 0xBB, 0xCC]);
        assert_eq!(&asm.code[6..9], &[0xAA, 0xBB, 0xCC]);
        assert_eq!(asm.current_offset, 9);
    }

    /// Symbol index assignment is sequential and deduplicating.
    #[test]
    fn test_symbol_index_assignment() {
        let mut asm = I686Assembler::new();
        let idx1 = asm.get_or_assign_symbol_index("printf");
        let idx2 = asm.get_or_assign_symbol_index("malloc");
        let idx3 = asm.get_or_assign_symbol_index("printf");

        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(idx3, 1); // Same symbol → same index.
    }

    /// Multi-block function assigns correct label offsets.
    #[test]
    fn test_multi_block_label_offsets() {
        let mut mf = MachineFunction::new("multi_fn".to_string());

        // Block 0: NOP + JMP → block 1
        let nop = MachineInstruction::new(codegen::I686_NOP);
        mf.blocks[0].push_instruction(nop);
        let jmp = MachineInstruction::new(codegen::I686_JMP)
            .with_operand(MachineOperand::BlockLabel(1))
            .set_terminator()
            .set_branch();
        mf.blocks[0].push_instruction(jmp);

        // Block 1: RET
        let mut block1 = MachineBasicBlock::with_label(".Lreturn".to_string());
        let ret = MachineInstruction::new(codegen::I686_RET).set_terminator();
        block1.push_instruction(ret);
        mf.add_block(block1);

        let output = assemble(&mf).unwrap();
        assert!(output.label_offsets.contains_key(".L0"));
        assert!(output.label_offsets.contains_key(".L1"));
        assert!(output.label_offsets.contains_key(".Lreturn"));

        // .L0 starts at 0
        assert_eq!(*output.label_offsets.get(".L0").unwrap(), 0);
        // .L1 starts after block 0's instructions (NOP=1 + JMP=5 = 6)
        let l1_offset = *output.label_offsets.get(".L1").unwrap();
        assert!(l1_offset > 0, ".L1 offset should be > 0, got {}", l1_offset);
    }

    /// Function name is recorded in label_offsets at offset 0.
    #[test]
    fn test_function_name_label() {
        let mf = MachineFunction::new("my_function".to_string());
        let output = assemble(&mf).unwrap();
        assert!(output.label_offsets.contains_key("my_function"));
        assert_eq!(*output.label_offsets.get("my_function").unwrap(), 0);
    }

    /// Validate that the assembler rejects registers > 7 for GPR.
    #[test]
    fn test_validate_register_range() {
        let mut mf = MachineFunction::new("bad_reg".to_string());
        // Register 8 is invalid on i686 (would need REX on x86-64).
        let bad_instr =
            MachineInstruction::new(codegen::I686_NOP).with_operand(MachineOperand::Register(8));
        mf.blocks[0].push_instruction(bad_instr);

        let result = assemble(&mf);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("out of range"), "Error: {}", err);
    }

    /// Relocation classification returns correct types.
    #[test]
    fn test_classify_relocation_type() {
        let asm = I686Assembler::new();

        // CALL instruction → R_386_PC32
        let call_instr = MachineInstruction::new(codegen::I686_CALL).set_call();
        assert_eq!(
            asm.classify_relocation_type(&call_instr, "printf", false),
            relocations::R_386_PC32
        );

        // GOTPC symbol
        let mov_instr = MachineInstruction::new(codegen::I686_NOP);
        assert_eq!(
            asm.classify_relocation_type(&mov_instr, "_GLOBAL_OFFSET_TABLE_", true),
            relocations::R_386_GOTPC
        );

        // Default → R_386_32
        assert_eq!(
            asm.classify_relocation_type(&mov_instr, "some_var", false),
            relocations::R_386_32
        );
    }

    /// Frame size and callee-saved registers are accessible during validation.
    #[test]
    fn test_function_metadata_access() {
        let mut mf = MachineFunction::new("meta_fn".to_string());
        mf.frame_size = 64;
        mf.callee_saved_regs = vec![registers::EBX, registers::ESI, registers::EDI];

        // Should succeed: all callee-saved regs are valid GPRs.
        let output = assemble(&mf);
        assert!(output.is_ok());
    }

    /// EBP as frame base register is valid.
    #[test]
    fn test_frame_pointer_register() {
        assert!(registers::is_gpr(registers::EBP));
        assert!(registers::is_gpr(registers::ESP));
        assert_eq!(registers::reg_name_32(registers::EAX), "eax");
        assert_eq!(registers::reg_name_32(registers::EBP), "ebp");
    }
}
