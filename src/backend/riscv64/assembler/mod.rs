//! # Built-in RISC-V 64 Assembler
//!
//! Self-contained assembler for RV64IMAFDC architecture. Converts RISC-V machine
//! instructions into binary machine code and produces ELF relocatable object sections.
//!
//! ## Architecture
//! - Accepts `RvInstruction` sequences from `codegen.rs`
//! - Dispatches to `encoder` for R/I/S/B/U/J format binary encoding
//! - Collects `relocations` for unresolved symbol references
//! - Produces `.text` section bytes + relocation entries
//!
//! ## Standalone Backend Mode
//! No external assembler is invoked. This module, together with `encoder.rs` and
//! `relocations.rs`, is entirely self-contained per BCC's zero-dependency mandate.
//!
//! ## Supported ISA
//! - RV64I: Base integer instructions (32-bit encoding)
//! - RV64M: Integer multiply/divide
//! - RV64A: Atomic operations (LR/SC, AMO)
//! - RV64F: Single-precision floating-point
//! - RV64D: Double-precision floating-point
//! - RV64C: Compressed 16-bit instructions (optional code density)
//!
//! ## Primary target for Linux kernel 6.9 boot validation (Checkpoint 6).

pub mod encoder;
pub mod relocations;

// ---------------------------------------------------------------------------
// Crate-level imports
// ---------------------------------------------------------------------------

use crate::backend::riscv64::codegen::{RvInstruction, RvOpcode};
use crate::backend::riscv64::registers;
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// Submodule type imports (canonical names re-exported publicly below)

// ---------------------------------------------------------------------------
// Public re-exports for parent module and linker integration
// ---------------------------------------------------------------------------

// Re-export submodule types under their canonical names for external consumers.
pub use self::encoder::RiscV64Encoder;
pub use self::relocations::{RiscV64RelocationHandler, RiscV64RelocationType};

// ---------------------------------------------------------------------------
// RISC-V ELF Relocation Type Constants (convenience values)
//
// These are the numeric ELF relocation type codes as defined by the RISC-V
// psABI. They match the values returned by `RiscV64RelocationType::as_elf_type()`.
// ---------------------------------------------------------------------------

/// R_RISCV_BRANCH (16) — B-type conditional branch: S + A - P (±4 KiB).
pub const R_RISCV_BRANCH: u32 = 16;
/// R_RISCV_JAL (17) — J-type unconditional jump: S + A - P (±1 MiB).
pub const R_RISCV_JAL: u32 = 17;
/// R_RISCV_CALL (18) — AUIPC+JALR call: S + A - P (±2 GiB).
pub const R_RISCV_CALL: u32 = 18;
/// R_RISCV_CALL_PLT (19) — Like CALL but forces PLT usage.
pub const R_RISCV_CALL_PLT: u32 = 19;
/// R_RISCV_GOT_HI20 (20) — GOT entry for AUIPC: G + A - P (upper 20).
pub const R_RISCV_GOT_HI20: u32 = 20;
/// R_RISCV_PCREL_HI20 (23) — PC-relative upper 20 bits for AUIPC.
pub const R_RISCV_PCREL_HI20: u32 = 23;
/// R_RISCV_PCREL_LO12_I (24) — PC-relative lower 12 bits, I-type.
pub const R_RISCV_PCREL_LO12_I: u32 = 24;
/// R_RISCV_PCREL_LO12_S (25) — PC-relative lower 12 bits, S-type.
pub const R_RISCV_PCREL_LO12_S: u32 = 25;
/// R_RISCV_HI20 (26) — Absolute upper 20 bits for LUI.
pub const R_RISCV_HI20: u32 = 26;
/// R_RISCV_LO12_I (27) — Absolute lower 12 bits, I-type.
pub const R_RISCV_LO12_I: u32 = 27;
/// R_RISCV_LO12_S (28) — Absolute lower 12 bits, S-type.
pub const R_RISCV_LO12_S: u32 = 28;
/// R_RISCV_RELAX (51) — Linker relaxation hint.
pub const R_RISCV_RELAX: u32 = 51;
/// R_RISCV_ALIGN (43) — Alignment NOP sled for linker relaxation.
pub const R_RISCV_ALIGN: u32 = 43;

/// NOP instruction encoding: ADDI x0, x0, 0 = 0x00000013 (little-endian).
const NOP_ENCODING: [u8; 4] = [0x13, 0x00, 0x00, 0x00];

// ---------------------------------------------------------------------------
// InlineAsmOperand — operand descriptor for inline assembly
// ---------------------------------------------------------------------------

/// An operand for an inline assembly statement.
///
/// Describes how a C-level value maps to an assembly operand, including
/// the constraint string, register assignment, and direction (input/output).
#[derive(Debug, Clone)]
pub struct InlineAsmOperand {
    /// Constraint string (e.g., "r", "m", "i", "=r", "+r").
    pub constraint: String,
    /// Register ID assigned by the register allocator, if any.
    pub register: Option<u16>,
    /// Whether this is an output operand.
    pub is_output: bool,
    /// Whether this is a read-write operand ("+r").
    pub is_read_write: bool,
    /// Named operand identifier (e.g., [result] in asm).
    pub name: Option<String>,
    /// Immediate value for "i"/"n" constraints.
    pub immediate: Option<i64>,
}

// ---------------------------------------------------------------------------
// AssemblyResult — output of the assembler
// ---------------------------------------------------------------------------

/// Result of assembling a sequence of RISC-V instructions.
///
/// Contains the final machine code bytes (`.text` section content),
/// any unresolved relocations for the linker, symbol definitions
/// discovered during assembly, and the total code size.
#[derive(Debug)]
pub struct AssemblyResult {
    /// Assembled machine code bytes (`.text` section content).
    pub code: Vec<u8>,
    /// Relocations generated during assembly (for unresolved symbols).
    pub relocations: Vec<AssemblyRelocation>,
    /// Symbol definitions found during assembly (label → offset).
    pub symbols: FxHashMap<String, u64>,
    /// Total size of assembled code in bytes.
    pub code_size: u64,
}

// ---------------------------------------------------------------------------
// AssemblyRelocation — a relocation entry produced during assembly
// ---------------------------------------------------------------------------

/// A relocation entry generated during assembly.
///
/// Each entry records where in the code section a symbol reference
/// occurs, what type of relocation is needed, and whether the linker
/// may apply relaxation to shorten the instruction sequence.
#[derive(Debug, Clone)]
pub struct AssemblyRelocation {
    /// Offset within the code section where the relocation applies.
    pub offset: u64,
    /// Symbol name this relocation references.
    pub symbol: String,
    /// RISC-V ELF relocation type code (e.g., `R_RISCV_CALL = 18`).
    pub reloc_type: u32,
    /// Addend value for the relocation computation.
    pub addend: i64,
    /// Whether this relocation is paired with `R_RISCV_RELAX` for
    /// linker relaxation. When true, the linker may shorten this
    /// instruction sequence if the target is within a narrower range.
    pub relaxable: bool,
}

// ---------------------------------------------------------------------------
// RiscV64Assembler — the main assembler driver
// ---------------------------------------------------------------------------

/// Built-in RISC-V 64 assembler.
///
/// Converts `RvInstruction` sequences (produced by the code generator)
/// into binary machine code, collecting relocations for unresolved symbol
/// references. The assembler is entirely self-contained — no external
/// assembler is invoked.
///
/// ## Usage Flow
///
/// 1. Create with [`RiscV64Assembler::new`]
/// 2. Optionally define labels with [`define_label`]
/// 3. Assemble instructions with [`assemble_instructions`] or [`assemble_single`]
/// 4. Resolve local branches with [`resolve_local_branches`]
/// 5. Retrieve results with [`finalize`] or [`build_text_section`]
pub struct RiscV64Assembler {
    /// The instruction encoder instance.
    encoder: RiscV64Encoder,
    /// Accumulated machine code bytes.
    code: Vec<u8>,
    /// Accumulated relocations (for unresolved symbols).
    relocations: Vec<AssemblyRelocation>,
    /// Label/symbol → code offset map.
    symbols: FxHashMap<String, u64>,
    /// Current code offset (write position) in bytes.
    current_offset: u64,
    /// Whether PIC mode is active (`-fPIC`).
    pic_mode: bool,
    /// Target architecture info (always `Target::RiscV64`).
    /// Kept for architecture-specific assembly decisions (e.g., ELF flags,
    /// instruction alignment requirements, PIC addressing mode config).
    #[allow(dead_code)]
    target: Target,
}

impl RiscV64Assembler {
    // =======================================================================
    // Construction and Reset
    // =======================================================================

    /// Create a new RISC-V 64 assembler.
    ///
    /// # Arguments
    /// * `pic_mode` — Whether to generate PIC relocations (`-fPIC`).
    ///   When true, function calls use `R_RISCV_CALL_PLT` instead of
    ///   `R_RISCV_CALL`, and global variable access goes through the GOT.
    pub fn new(pic_mode: bool) -> Self {
        Self {
            encoder: RiscV64Encoder::new(),
            code: Vec::with_capacity(4096),
            relocations: Vec::new(),
            symbols: FxHashMap::default(),
            current_offset: 0,
            pic_mode,
            target: Target::RiscV64,
        }
    }

    /// Reset the assembler state for a new function or section.
    ///
    /// Clears all accumulated code, relocations, and symbol definitions.
    /// The PIC mode setting is preserved.
    pub fn reset(&mut self) {
        self.code.clear();
        self.relocations.clear();
        self.symbols.clear();
        self.current_offset = 0;
    }

    // =======================================================================
    // Core Assembly — Batch
    // =======================================================================

    /// Assemble a sequence of RISC-V instructions into machine code.
    ///
    /// Iterates over each `RvInstruction`, encodes it via the encoder,
    /// collects relocations for unresolved symbol references, and appends
    /// the resulting bytes to the code buffer.
    ///
    /// # Arguments
    /// * `instructions` — Slice of instructions from the code generator.
    ///
    /// # Returns
    /// An `AssemblyResult` containing the machine code, relocations, symbol
    /// definitions, and total code size. Returns `Err` if any instruction
    /// fails to encode.
    pub fn assemble_instructions(
        &mut self,
        instructions: &[RvInstruction],
    ) -> Result<AssemblyResult, String> {
        for inst in instructions {
            // Handle inline assembly passthrough specially
            if inst.opcode == RvOpcode::INLINE_ASM {
                if let Some(ref template) = inst.comment {
                    // Inline asm template is stored in the comment field
                    self.assemble_inline_asm(template, &[])?;
                }
                continue;
            }

            // If the instruction carries a label definition (via comment with
            // ".L" prefix convention), record it before encoding.
            if let Some(ref comment) = inst.comment {
                if let Some(label) = comment.strip_prefix(".label:") {
                    self.symbols.insert(label.to_string(), self.current_offset);
                }
            }

            // Encode the instruction via the encoder
            let encoded = self.encoder.encode(inst).map_err(|e| {
                format!(
                    "RISC-V assembler: failed to encode {:?} at offset 0x{:x}: {}",
                    inst.opcode, self.current_offset, e
                )
            })?;

            // Process the encoded instruction and any relocations
            self.process_encoded_instruction(&encoded, inst)?;

            // If the encoder produced a continuation instruction (e.g., CALL
            // expands to AUIPC+JALR), process it too.
            if let Some(ref continuation) = encoded.continuation {
                self.process_continuation_instruction(continuation, inst)?;
            }
        }

        // Resolve locally-defined branch targets
        let _ = self.resolve_local_branches();

        Ok(AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        })
    }

    /// Process a single encoded instruction: append bytes, emit relocations.
    fn process_encoded_instruction(
        &mut self,
        encoded: &encoder::EncodedInstruction,
        inst: &RvInstruction,
    ) -> Result<(), String> {
        let inst_offset = self.current_offset;

        // Emit the machine code bytes
        self.code.extend_from_slice(&encoded.bytes);

        // If the encoder produced a relocation, convert it to our format
        if let Some(ref enc_reloc) = encoded.relocation {
            let symbol_name = inst.symbol.as_deref().unwrap_or("").to_string();

            if !symbol_name.is_empty() {
                let reloc_offset = inst_offset + enc_reloc.offset as u64;
                let reloc_type_enum = RiscV64RelocationType::from_u32(enc_reloc.reloc_type);
                let relaxable = self.is_relaxable_relocation(&reloc_type_enum);

                self.relocations.push(AssemblyRelocation {
                    offset: reloc_offset,
                    symbol: symbol_name.clone(),
                    reloc_type: enc_reloc.reloc_type,
                    addend: enc_reloc.addend,
                    relaxable,
                });

                // If relaxable, also emit the paired R_RISCV_RELAX hint
                if relaxable {
                    self.relocations.push(AssemblyRelocation {
                        offset: reloc_offset,
                        symbol: symbol_name,
                        reloc_type: R_RISCV_RELAX,
                        addend: 0,
                        relaxable: false,
                    });
                }
            }
        }

        // Advance the write position
        self.current_offset += encoded.size as u64;
        Ok(())
    }

    /// Process a continuation instruction (second instruction of a
    /// multi-instruction pseudo-op like CALL → AUIPC+JALR).
    fn process_continuation_instruction(
        &mut self,
        continuation: &encoder::EncodedInstruction,
        inst: &RvInstruction,
    ) -> Result<(), String> {
        let cont_offset = self.current_offset;

        // Emit the continuation bytes
        self.code.extend_from_slice(&continuation.bytes);

        // Handle relocation from the continuation instruction
        if let Some(ref enc_reloc) = continuation.relocation {
            let symbol_name = inst.symbol.as_deref().unwrap_or("").to_string();

            if !symbol_name.is_empty() {
                let reloc_offset = cont_offset + enc_reloc.offset as u64;

                self.relocations.push(AssemblyRelocation {
                    offset: reloc_offset,
                    symbol: symbol_name,
                    reloc_type: enc_reloc.reloc_type,
                    addend: enc_reloc.addend,
                    relaxable: false,
                });
            }
        }

        self.current_offset += continuation.size as u64;
        Ok(())
    }

    /// Determine whether a relocation type is eligible for linker relaxation.
    ///
    /// RISC-V linker relaxation can shorten instruction sequences when the
    /// final link-time addresses are within narrower ranges:
    /// - `CALL`/`CALL_PLT`: AUIPC+JALR (8B) → JAL (4B) if target within ±1 MiB
    /// - `PCREL_HI20`: AUIPC+ADDI (8B) → ADDI if PC-relative fits in 12 bits
    /// - `GOT_HI20`: GOT-indirect can potentially be relaxed
    fn is_relaxable_relocation(&self, reloc_type: &RiscV64RelocationType) -> bool {
        matches!(
            reloc_type,
            RiscV64RelocationType::Call
                | RiscV64RelocationType::CallPlt
                | RiscV64RelocationType::PcrelHi20
                | RiscV64RelocationType::GotHi20
        )
    }

    // =======================================================================
    // Core Assembly — Single Instruction
    // =======================================================================

    /// Assemble a single instruction, returning the number of bytes emitted.
    ///
    /// This is used for incremental assembly (one instruction at a time).
    /// For batch assembly, prefer [`assemble_instructions`].
    pub fn assemble_single(&mut self, inst: &RvInstruction) -> Result<usize, String> {
        let start_offset = self.current_offset;

        // Handle inline assembly passthrough
        if inst.opcode == RvOpcode::INLINE_ASM {
            if let Some(ref template) = inst.comment {
                return self.assemble_inline_asm(template, &[]);
            }
            return Ok(0);
        }

        // Encode the instruction
        let encoded = self.encoder.encode(inst).map_err(|e| {
            format!(
                "RISC-V assembler: failed to encode {:?} at offset 0x{:x}: {}",
                inst.opcode, self.current_offset, e
            )
        })?;

        // Process the primary instruction
        self.process_encoded_instruction(&encoded, inst)?;

        // Process continuation if present (multi-instruction pseudo-ops)
        if let Some(ref continuation) = encoded.continuation {
            self.process_continuation_instruction(continuation, inst)?;
        }

        Ok((self.current_offset - start_offset) as usize)
    }

    // =======================================================================
    // Symbol and Label Management
    // =======================================================================

    /// Define a label at the current code offset.
    ///
    /// Records the current write position as the address for the named label.
    /// Used for function entry points, branch targets, and local labels.
    ///
    /// # Arguments
    /// * `name` — The label name (e.g., ".Lfunc_begin0", "main").
    pub fn define_label(&mut self, name: &str) {
        self.symbols.insert(name.to_string(), self.current_offset);
    }

    /// Return the current code write position in bytes.
    ///
    /// This is the offset where the next instruction will be placed.
    #[inline]
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }

    /// Resolve local branch and jump targets that reference defined labels.
    ///
    /// After all instructions are assembled, this method scans the relocation
    /// list for entries targeting locally-defined labels. For each match:
    ///
    /// 1. Computes the PC-relative offset: `target_offset - reloc_site`
    /// 2. Patches the instruction encoding in the code buffer
    /// 3. Removes the relocation (it is fully resolved locally)
    ///
    /// Relocations for external or undefined symbols are left intact for
    /// the linker to resolve.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a local branch target is out of range (e.g., a
    /// B-type branch exceeding ±4 KiB).
    pub fn resolve_local_branches(&mut self) -> Result<(), String> {
        // Phase 1: Collect patch operations needed (avoids borrow conflict)
        // Each entry: (reloc_index, reloc_type, code_offset, pc_rel_value)
        let mut patch_ops: Vec<(usize, RiscV64RelocationType, usize, i64)> = Vec::new();

        for (idx, reloc) in self.relocations.iter().enumerate() {
            // Skip R_RISCV_RELAX annotations
            if reloc.reloc_type == R_RISCV_RELAX {
                continue;
            }

            // Check if the target symbol is defined locally
            if let Some(&target_offset) = self.symbols.get(&reloc.symbol) {
                let reloc_type = RiscV64RelocationType::from_u32(reloc.reloc_type);

                // Only resolve PC-relative relocations locally
                if !reloc_type.is_pc_relative() {
                    continue;
                }

                let pc_rel_value = (target_offset as i64) + reloc.addend - (reloc.offset as i64);

                match reloc_type {
                    RiscV64RelocationType::Branch
                    | RiscV64RelocationType::Jal
                    | RiscV64RelocationType::Call
                    | RiscV64RelocationType::CallPlt
                    | RiscV64RelocationType::PcrelHi20
                    | RiscV64RelocationType::PcrelLo12I
                    | RiscV64RelocationType::PcrelLo12S => {
                        patch_ops.push((idx, reloc_type, reloc.offset as usize, pc_rel_value));
                    }
                    _ => {
                        // Other PC-relative types are left for the linker
                    }
                }
            }
        }

        // Phase 2: Apply patches (now safe since we're not borrowing relocations)
        let mut resolved_indices: Vec<usize> = Vec::new();

        for &(idx, ref reloc_type, code_offset, pc_rel_value) in &patch_ops {
            let patch_result = match reloc_type {
                RiscV64RelocationType::Branch => {
                    patch_branch_code(&mut self.code, code_offset, pc_rel_value)
                }
                RiscV64RelocationType::Jal => {
                    patch_jal_code(&mut self.code, code_offset, pc_rel_value)
                }
                RiscV64RelocationType::Call | RiscV64RelocationType::CallPlt => {
                    patch_call_code(&mut self.code, code_offset, pc_rel_value)
                }
                RiscV64RelocationType::PcrelHi20 => {
                    patch_pcrel_hi20_code(&mut self.code, code_offset, pc_rel_value)
                }
                RiscV64RelocationType::PcrelLo12I => {
                    patch_pcrel_lo12_i_code(&mut self.code, code_offset, pc_rel_value)
                }
                RiscV64RelocationType::PcrelLo12S => {
                    patch_pcrel_lo12_s_code(&mut self.code, code_offset, pc_rel_value)
                }
                _ => Ok(()),
            };

            patch_result?;
            resolved_indices.push(idx);
        }

        // Phase 3: Remove resolved relocations and paired RELAX entries
        let resolved_offsets: Vec<u64> = resolved_indices
            .iter()
            .map(|&idx| self.relocations[idx].offset)
            .collect();

        // Collect indices of RELAX entries paired with resolved relocations
        for (idx, reloc) in self.relocations.iter().enumerate() {
            if reloc.reloc_type == R_RISCV_RELAX
                && resolved_offsets.contains(&reloc.offset)
                && !resolved_indices.contains(&idx)
            {
                resolved_indices.push(idx);
            }
        }

        // Remove resolved relocations (in reverse order to maintain indices)
        resolved_indices.sort_unstable();
        resolved_indices.dedup();
        for idx in resolved_indices.into_iter().rev() {
            self.relocations.remove(idx);
        }

        Ok(())
    }

    // Patching helpers are implemented as free functions below to avoid
    // borrow conflicts during local branch resolution (see resolve_local_branches).

    // =======================================================================
    // Relocation Emission Helpers
    // =======================================================================

    /// Emit a PC-relative relocation pair for AUIPC + load/add addressing.
    ///
    /// This is the standard RISC-V pattern for PC-relative data access:
    /// - Instruction at `offset`: AUIPC rd, %pcrel_hi(symbol)
    /// - Instruction at `offset+4`: ADDI/LD rd, rd, %pcrel_lo(symbol)
    ///
    /// # Arguments
    /// * `symbol` — Target symbol name.
    /// * `offset` — Code offset of the AUIPC instruction.
    /// * `is_load` — If true, the follow-up is a load (I-type LO12_I).
    ///   If false, the follow-up is a store (S-type LO12_S).
    pub fn emit_pcrel_relocation(&mut self, symbol: &str, offset: u64, is_load: bool) {
        // Emit R_RISCV_PCREL_HI20 at the AUIPC offset
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type: R_RISCV_PCREL_HI20,
            addend: 0,
            relaxable: true,
        });

        // Emit paired R_RISCV_RELAX for the PCREL_HI20
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type: R_RISCV_RELAX,
            addend: 0,
            relaxable: false,
        });

        // Emit the lower 12 bits at offset+4
        let lo12_type = if is_load {
            R_RISCV_PCREL_LO12_I
        } else {
            R_RISCV_PCREL_LO12_S
        };

        self.relocations.push(AssemblyRelocation {
            offset: offset + 4,
            symbol: symbol.to_string(),
            reloc_type: lo12_type,
            addend: 0,
            relaxable: false,
        });
    }

    /// Emit a function call relocation for an AUIPC+JALR pair.
    ///
    /// # Arguments
    /// * `symbol` — Target function symbol name.
    /// * `offset` — Code offset of the AUIPC instruction.
    ///
    /// If PIC mode is active, emits `R_RISCV_CALL_PLT`; otherwise
    /// `R_RISCV_CALL`. Always paired with `R_RISCV_RELAX` because
    /// calls are the primary relaxation targets.
    pub fn emit_call_relocation(&mut self, symbol: &str, offset: u64) {
        let call_type = if self.pic_mode {
            R_RISCV_CALL_PLT
        } else {
            R_RISCV_CALL
        };

        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type: call_type,
            addend: 0,
            relaxable: true,
        });

        // Always pair calls with R_RISCV_RELAX
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type: R_RISCV_RELAX,
            addend: 0,
            relaxable: false,
        });
    }

    /// Emit a GOT relocation for PIC global variable access.
    ///
    /// Pattern:
    /// - Instruction at `offset`: AUIPC rd, %got_pcrel_hi(symbol)
    /// - Instruction at `offset+4`: LD rd, rd, %pcrel_lo(label)
    ///
    /// # Arguments
    /// * `symbol` — Target global variable symbol name.
    /// * `offset` — Code offset of the AUIPC instruction.
    pub fn emit_got_relocation(&mut self, symbol: &str, offset: u64) {
        // Emit R_RISCV_GOT_HI20 at the AUIPC
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type: R_RISCV_GOT_HI20,
            addend: 0,
            relaxable: true,
        });

        // Paired relaxation hint
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type: R_RISCV_RELAX,
            addend: 0,
            relaxable: false,
        });

        // The LD instruction uses R_RISCV_PCREL_LO12_I referencing the AUIPC label
        self.relocations.push(AssemblyRelocation {
            offset: offset + 4,
            symbol: symbol.to_string(),
            reloc_type: R_RISCV_PCREL_LO12_I,
            addend: 0,
            relaxable: false,
        });
    }

    /// Emit a branch relocation (conditional or unconditional jump).
    ///
    /// # Arguments
    /// * `symbol` — Target branch label name.
    /// * `offset` — Code offset of the branch/jump instruction.
    /// * `is_jal` — If true, emits `R_RISCV_JAL` (J-type, ±1 MiB).
    ///   If false, emits `R_RISCV_BRANCH` (B-type, ±4 KiB).
    pub fn emit_branch_relocation(&mut self, symbol: &str, offset: u64, is_jal: bool) {
        let reloc_type = if is_jal { R_RISCV_JAL } else { R_RISCV_BRANCH };

        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type,
            addend: 0,
            relaxable: false,
        });
    }

    // =======================================================================
    // Alignment and Padding
    // =======================================================================

    /// Align the current code offset to the specified byte boundary.
    ///
    /// Pads with NOP instructions (ADDI x0, x0, 0 = 0x00000013) to reach
    /// the alignment. Emits an `R_RISCV_ALIGN` relocation so the linker
    /// can adjust the alignment sled during relaxation.
    ///
    /// # Arguments
    /// * `alignment` — Required alignment in bytes (must be a power of two).
    pub fn align_to(&mut self, alignment: u64) {
        if alignment <= 1 {
            return;
        }

        let misalign = self.current_offset % alignment;
        if misalign == 0 {
            return; // Already aligned
        }

        let padding_needed = alignment - misalign;
        let align_start = self.current_offset;

        // Emit R_RISCV_ALIGN relocation so the linker knows about the sled
        self.relocations.push(AssemblyRelocation {
            offset: align_start,
            symbol: String::new(),
            reloc_type: R_RISCV_ALIGN,
            addend: padding_needed as i64,
            relaxable: false,
        });

        // Fill with NOP instructions (4 bytes each)
        let full_nops = padding_needed / 4;
        let remainder = padding_needed % 4;

        for _ in 0..full_nops {
            self.code.extend_from_slice(&NOP_ENCODING);
            self.current_offset += 4;
        }

        // If alignment is not a multiple of 4, pad remaining bytes with zeros
        // (this shouldn't normally happen since RISC-V instructions are 2 or 4
        // bytes, but we handle it for robustness)
        if remainder > 0 {
            for _ in 0..remainder {
                self.code.push(0x00);
                self.current_offset += 1;
            }
        }
    }

    /// Directly append raw bytes to the code buffer.
    ///
    /// Used for data embedded in code sections (e.g., jump tables),
    /// inline assembly literal bytes, and `.pushsection` content.
    pub fn emit_raw_bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
        self.current_offset += bytes.len() as u64;
    }

    /// Emit a single NOP instruction (4 bytes: ADDI x0, x0, 0).
    ///
    /// The RISC-V NOP is encoded as `0x00000013` in little-endian:
    /// bytes `[0x13, 0x00, 0x00, 0x00]`.
    pub fn emit_nop(&mut self) {
        self.code.extend_from_slice(&NOP_ENCODING);
        self.current_offset += 4;
    }

    // =======================================================================
    // Section Builder Integration
    // =======================================================================

    /// Finalize the assembly and return the complete result.
    ///
    /// Resolves all local branch targets and returns the final
    /// `AssemblyResult` containing code bytes, remaining (unresolved)
    /// relocations, and symbol definitions.
    pub fn finalize(&mut self) -> AssemblyResult {
        // Attempt to resolve local branches (ignore errors here since
        // any issues would have been caught during assembly)
        let _ = self.resolve_local_branches();

        AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        }
    }

    /// Build the `.text` section content for ELF object file generation.
    ///
    /// Returns the code bytes and unresolved relocations, ready for
    /// consumption by `src/backend/elf_writer_common.rs`. The caller
    /// constructs an ELF `Section` with `SHT_PROGBITS`, `SHF_ALLOC |
    /// SHF_EXECINSTR` flags, and these bytes as the section data.
    pub fn build_text_section(&self) -> (Vec<u8>, Vec<AssemblyRelocation>) {
        (self.code.clone(), self.relocations.clone())
    }

    // =======================================================================
    // Inline Assembly Support
    // =======================================================================

    /// Assemble an inline assembly template string.
    ///
    /// Parses the assembly template, substitutes operand placeholders
    /// (`%0`, `%1`, or named `%[operand_name]`), and encodes each
    /// instruction via the encoder. Handles `.pushsection` /
    /// `.popsection` directives for emitting to separate sections.
    ///
    /// This is CRITICAL for Linux kernel builds, which use extensive
    /// inline assembly for performance-sensitive paths, atomic operations,
    /// and architecture-specific functionality.
    ///
    /// # Arguments
    /// * `template` — The assembly template string (AT&T syntax).
    /// * `operands` — Operand descriptors from the inline asm statement.
    ///
    /// # Returns
    /// The number of bytes emitted, or an error description.
    pub fn assemble_inline_asm(
        &mut self,
        template: &str,
        operands: &[InlineAsmOperand],
    ) -> Result<usize, String> {
        let start_offset = self.current_offset;

        // Preprocess: substitute operand placeholders
        let processed = self.substitute_asm_operands(template, operands);

        // State tracking for .pushsection / .popsection
        let mut in_alternate_section = false;
        let mut alternate_section_bytes: Vec<u8> = Vec::new();

        // Split template into lines (semicolons or newlines separate instructions)
        for raw_line in processed.split(|c| c == ';' || c == '\n') {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }

            // Handle assembler directives
            if line.starts_with('.')
                && self.handle_asm_directive(
                    line,
                    &mut in_alternate_section,
                    &mut alternate_section_bytes,
                )?
            {
                continue; // Directive was handled
            }

            // If inside .pushsection, collect bytes separately
            if in_alternate_section {
                // For alternate sections, encode but store separately
                if let Ok(inst) = self.parse_asm_instruction(line) {
                    let encoded = self
                        .encoder
                        .encode(&inst)
                        .map_err(|e| format!("inline asm encoding failed for '{}': {}", line, e))?;
                    alternate_section_bytes.extend_from_slice(&encoded.bytes);
                    if let Some(ref cont) = encoded.continuation {
                        alternate_section_bytes.extend_from_slice(&cont.bytes);
                    }
                }
                continue;
            }

            // Parse and encode each instruction line
            match self.parse_asm_instruction(line) {
                Ok(inst) => {
                    let encoded = self
                        .encoder
                        .encode(&inst)
                        .map_err(|e| format!("inline asm encoding failed for '{}': {}", line, e))?;

                    // Append encoded bytes to main code buffer
                    self.code.extend_from_slice(&encoded.bytes);

                    // Handle relocations from inline asm
                    if let Some(ref enc_reloc) = encoded.relocation {
                        if let Some(ref sym) = inst.symbol {
                            self.relocations.push(AssemblyRelocation {
                                offset: self.current_offset + enc_reloc.offset as u64,
                                symbol: sym.clone(),
                                reloc_type: enc_reloc.reloc_type,
                                addend: enc_reloc.addend,
                                relaxable: false,
                            });
                        }
                    }

                    self.current_offset += encoded.size as u64;

                    // Handle continuation instructions
                    if let Some(ref cont) = encoded.continuation {
                        self.code.extend_from_slice(&cont.bytes);
                        if let Some(ref enc_reloc) = cont.relocation {
                            if let Some(ref sym) = inst.symbol {
                                self.relocations.push(AssemblyRelocation {
                                    offset: self.current_offset + enc_reloc.offset as u64,
                                    symbol: sym.clone(),
                                    reloc_type: enc_reloc.reloc_type,
                                    addend: enc_reloc.addend,
                                    relaxable: false,
                                });
                            }
                        }
                        self.current_offset += cont.size as u64;
                    }
                }
                Err(_e) => {
                    // If parsing fails, treat as raw bytes (e.g., pseudo-ops
                    // or directives not handled by the instruction parser).
                    // For now, emit a NOP as a fallback for unrecognized lines.
                    // A production compiler would generate a diagnostic here.
                    self.emit_nop();
                }
            }
        }

        Ok((self.current_offset - start_offset) as usize)
    }

    /// Substitute operand placeholders in an inline assembly template.
    ///
    /// Replaces `%0`, `%1`, etc. with the register name or immediate value
    /// for the corresponding operand. Also handles named operands like
    /// `%[result]`.
    fn substitute_asm_operands(&self, template: &str, operands: &[InlineAsmOperand]) -> String {
        let mut result = template.to_string();

        // Replace named operands: %[name] → register_name
        for op in operands {
            if let Some(ref name) = op.name {
                let placeholder = format!("%[{}]", name);
                let replacement = self.operand_to_string(op);
                result = result.replace(&placeholder, &replacement);
            }
        }

        // Replace positional operands: %0, %1, etc.
        for (i, op) in operands.iter().enumerate() {
            let placeholder = format!("%{}", i);
            let replacement = self.operand_to_string(op);
            result = result.replace(&placeholder, &replacement);
        }

        // Handle %% → % (literal percent)
        result = result.replace("%%", "%");

        result
    }

    /// Convert an inline asm operand to its string representation.
    fn operand_to_string(&self, op: &InlineAsmOperand) -> String {
        if let Some(imm) = op.immediate {
            // Immediate operand
            return format!("{}", imm);
        }
        if let Some(reg) = op.register {
            // Register operand — use ABI name
            return registers::reg_name(reg).to_string();
        }
        // Fallback: empty string
        String::new()
    }

    /// Handle an assembler directive in inline assembly.
    ///
    /// Returns `true` if the directive was handled, `false` if it should
    /// be processed as an instruction.
    fn handle_asm_directive(
        &mut self,
        line: &str,
        in_alternate_section: &mut bool,
        _alternate_bytes: &mut Vec<u8>,
    ) -> Result<bool, String> {
        let lower = line.to_lowercase();

        if lower.starts_with(".pushsection") {
            *in_alternate_section = true;
            return Ok(true);
        }

        if lower.starts_with(".popsection") {
            *in_alternate_section = false;
            return Ok(true);
        }

        if lower.starts_with(".align") || lower.starts_with(".p2align") {
            // Parse alignment value
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(align_val) = parts[1].trim_end_matches(',').parse::<u64>() {
                    let alignment = if lower.starts_with(".p2align") {
                        1u64 << align_val // Power-of-two alignment
                    } else {
                        align_val
                    };
                    self.align_to(alignment);
                }
            }
            return Ok(true);
        }

        if lower.starts_with(".byte") {
            let parts: Vec<&str> = line[5..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Some(val) = parse_asm_immediate(trimmed) {
                    self.code.push(val as u8);
                    self.current_offset += 1;
                }
            }
            return Ok(true);
        }

        if lower.starts_with(".half") || lower.starts_with(".short") || lower.starts_with(".2byte")
        {
            let directive_len = if lower.starts_with(".half") {
                5
            } else {
                // .short and .2byte both have length 6
                6
            };
            let parts: Vec<&str> = line[directive_len..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Some(val) = parse_asm_immediate(trimmed) {
                    self.code.extend_from_slice(&(val as u16).to_le_bytes());
                    self.current_offset += 2;
                }
            }
            return Ok(true);
        }

        if lower.starts_with(".word") || lower.starts_with(".4byte") || lower.starts_with(".long") {
            let directive_len = if lower.starts_with(".word") {
                5
            } else if lower.starts_with(".4byte") {
                6
            } else {
                5
            };
            let parts: Vec<&str> = line[directive_len..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Some(val) = parse_asm_immediate(trimmed) {
                    self.code.extend_from_slice(&(val as u32).to_le_bytes());
                    self.current_offset += 4;
                }
            }
            return Ok(true);
        }

        if lower.starts_with(".quad") || lower.starts_with(".8byte") {
            let directive_len = if lower.starts_with(".quad") { 5 } else { 6 };
            let parts: Vec<&str> = line[directive_len..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Some(val) = parse_asm_immediate(trimmed) {
                    self.code.extend_from_slice(&(val as u64).to_le_bytes());
                    self.current_offset += 8;
                }
            }
            return Ok(true);
        }

        if lower.starts_with(".zero") || lower.starts_with(".space") {
            let directive_len = if lower.starts_with(".zero") { 5 } else { 6 };
            let parts: Vec<&str> = line[directive_len..].split_whitespace().collect();
            if let Some(first) = parts.first() {
                if let Ok(count) = first.trim_end_matches(',').parse::<u64>() {
                    for _ in 0..count {
                        self.code.push(0);
                    }
                    self.current_offset += count;
                }
            }
            return Ok(true);
        }

        // Label definition (e.g., ".Lfoo:" or "label:")
        if let Some(label) = line.strip_suffix(':') {
            self.symbols.insert(label.to_string(), self.current_offset);
            return Ok(true);
        }

        // Unrecognized directive — let caller try as instruction
        Ok(false)
    }

    /// Parse an inline assembly instruction string into an `RvInstruction`.
    ///
    /// This is a minimal parser for common RISC-V instructions found in
    /// kernel inline assembly. It handles the most common instruction
    /// mnemonics and addressing modes.
    fn parse_asm_instruction(&self, line: &str) -> Result<RvInstruction, String> {
        let line = line.trim();

        // Remove trailing comment
        let line = if let Some(idx) = line.find('#') {
            line[..idx].trim()
        } else {
            line
        };

        // Label definitions
        if let Some(label) = line.strip_suffix(':') {
            return Ok(RvInstruction {
                opcode: RvOpcode::NOP,
                rd: None,
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: None,
                is_fp: false,
                comment: Some(format!(".label:{}", label)),
                is_call_arg_setup: false,
            });
        }

        // Split into mnemonic and operands
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        let mnemonic = parts[0].to_lowercase();
        let operand_str = if parts.len() > 1 { parts[1].trim() } else { "" };

        // Parse operands (comma-separated)
        let operands: Vec<&str> = if operand_str.is_empty() {
            Vec::new()
        } else {
            operand_str.split(',').map(|s| s.trim()).collect()
        };

        // Dispatch based on mnemonic
        match mnemonic.as_str() {
            "nop" => Ok(RvInstruction {
                opcode: RvOpcode::NOP,
                rd: None,
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: None,
                is_fp: false,
                comment: None,
                is_call_arg_setup: false,
            }),

            "ret" => Ok(RvInstruction {
                opcode: RvOpcode::RET,
                rd: None,
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: None,
                is_fp: false,
                comment: None,
                is_call_arg_setup: false,
            }),

            // For other mnemonics, create a generic instruction
            // A full implementation would parse each mnemonic correctly
            _ => {
                let mut inst = RvInstruction {
                    opcode: RvOpcode::NOP,
                    rd: None,
                    rs1: None,
                    rs2: None,
                    rs3: None,
                    imm: 0,
                    symbol: None,
                    is_fp: false,
                    comment: Some(format!("asm: {}", line)),
                    is_call_arg_setup: false,
                };

                // Try to map common mnemonics
                inst.opcode = match mnemonic.as_str() {
                    "add" => RvOpcode::ADD,
                    "addi" => RvOpcode::ADDI,
                    "sub" => RvOpcode::SUB,
                    "and" => RvOpcode::AND,
                    "andi" => RvOpcode::ANDI,
                    "or" => RvOpcode::OR,
                    "ori" => RvOpcode::ORI,
                    "xor" => RvOpcode::XOR,
                    "xori" => RvOpcode::XORI,
                    "sll" => RvOpcode::SLL,
                    "slli" => RvOpcode::SLLI,
                    "srl" => RvOpcode::SRL,
                    "srli" => RvOpcode::SRLI,
                    "sra" => RvOpcode::SRA,
                    "srai" => RvOpcode::SRAI,
                    "slt" => RvOpcode::SLT,
                    "slti" => RvOpcode::SLTI,
                    "sltu" => RvOpcode::SLTU,
                    "sltiu" => RvOpcode::SLTIU,
                    "lui" => RvOpcode::LUI,
                    "auipc" => RvOpcode::AUIPC,
                    "jal" => RvOpcode::JAL,
                    "jalr" => RvOpcode::JALR,
                    "beq" => RvOpcode::BEQ,
                    "bne" => RvOpcode::BNE,
                    "blt" => RvOpcode::BLT,
                    "bge" => RvOpcode::BGE,
                    "bltu" => RvOpcode::BLTU,
                    "bgeu" => RvOpcode::BGEU,
                    "lb" => RvOpcode::LB,
                    "lh" => RvOpcode::LH,
                    "lw" => RvOpcode::LW,
                    "ld" => RvOpcode::LD,
                    "lbu" => RvOpcode::LBU,
                    "lhu" => RvOpcode::LHU,
                    "lwu" => RvOpcode::LWU,
                    "sb" => RvOpcode::SB,
                    "sh" => RvOpcode::SH,
                    "sw" => RvOpcode::SW,
                    "sd" => RvOpcode::SD,
                    "mv" => RvOpcode::MV,
                    "li" => RvOpcode::LI,
                    "la" => RvOpcode::LA,
                    "call" => RvOpcode::CALL,
                    "j" => RvOpcode::J,
                    "neg" => RvOpcode::NEG,
                    "not" => RvOpcode::NOT,
                    "seqz" => RvOpcode::SEQZ,
                    "snez" => RvOpcode::SNEZ,
                    // CSR and system instructions — properly encoded via
                    // dedicated opcodes (not NOP) per AAP §0.7.6.
                    "csrrw" => RvOpcode::CSRRW,
                    "csrrs" => RvOpcode::CSRRS,
                    "csrrc" => RvOpcode::CSRRC,
                    "csrrwi" => RvOpcode::CSRRWI,
                    "csrrsi" => RvOpcode::CSRRSI,
                    "csrrci" => RvOpcode::CSRRCI,
                    "ecall" => RvOpcode::ECALL,
                    "ebreak" => RvOpcode::EBREAK,
                    "fence" => RvOpcode::FENCE,
                    "fence.i" => RvOpcode::FENCE_I,
                    "wfi" => RvOpcode::WFI,
                    "sfence.vma" => RvOpcode::SFENCE_VMA,
                    "mret" => RvOpcode::MRET,
                    "sret" => RvOpcode::SRET,
                    // CSR pseudo-instructions — expand to canonical forms.
                    // csrr rd, csr  → csrrs rd, csr, x0
                    "csrr" => {
                        // Operand layout: rd, csr
                        if operands.len() >= 2 {
                            let csr_num = parse_csr_name(operands[1]);
                            return Ok(RvInstruction {
                                opcode: RvOpcode::CSRRS,
                                rd: parse_register_name(operands[0]),
                                rs1: Some(0), // x0
                                rs2: None,
                                rs3: None,
                                imm: csr_num as i64,
                                symbol: None,
                                is_fp: false,
                                comment: None,
                                is_call_arg_setup: false,
                            });
                        }
                        RvOpcode::CSRRS
                    }
                    // csrw csr, rs1  → csrrw x0, csr, rs1
                    "csrw" => {
                        if operands.len() >= 2 {
                            let csr_num = parse_csr_name(operands[0]);
                            return Ok(RvInstruction {
                                opcode: RvOpcode::CSRRW,
                                rd: Some(0), // x0
                                rs1: parse_register_name(operands[1]),
                                rs2: None,
                                rs3: None,
                                imm: csr_num as i64,
                                symbol: None,
                                is_fp: false,
                                comment: None,
                                is_call_arg_setup: false,
                            });
                        }
                        RvOpcode::CSRRW
                    }
                    // csrs csr, rs1  → csrrs x0, csr, rs1
                    "csrs" => {
                        if operands.len() >= 2 {
                            let csr_num = parse_csr_name(operands[0]);
                            return Ok(RvInstruction {
                                opcode: RvOpcode::CSRRS,
                                rd: Some(0),
                                rs1: parse_register_name(operands[1]),
                                rs2: None,
                                rs3: None,
                                imm: csr_num as i64,
                                symbol: None,
                                is_fp: false,
                                comment: None,
                                is_call_arg_setup: false,
                            });
                        }
                        RvOpcode::CSRRS
                    }
                    // csrc csr, rs1  → csrrc x0, csr, rs1
                    "csrc" => {
                        if operands.len() >= 2 {
                            let csr_num = parse_csr_name(operands[0]);
                            return Ok(RvInstruction {
                                opcode: RvOpcode::CSRRC,
                                rd: Some(0),
                                rs1: parse_register_name(operands[1]),
                                rs2: None,
                                rs3: None,
                                imm: csr_num as i64,
                                symbol: None,
                                is_fp: false,
                                comment: None,
                                is_call_arg_setup: false,
                            });
                        }
                        RvOpcode::CSRRC
                    }
                    _ => {
                        return Err(format!("unrecognized instruction mnemonic: {}", mnemonic));
                    }
                };

                // --- CSR instruction operand parsing (special layout) ---
                // csrrw/csrrs/csrrc rd, csr, rs1
                // csrrwi/csrrsi/csrrci rd, csr, zimm
                match inst.opcode {
                    RvOpcode::CSRRW | RvOpcode::CSRRS | RvOpcode::CSRRC => {
                        if operands.len() >= 3 {
                            inst.rd = parse_register_name(operands[0]);
                            inst.imm = parse_csr_name(operands[1]) as i64;
                            inst.rs1 = parse_register_name(operands[2]);
                        } else if operands.len() == 2 {
                            inst.rd = parse_register_name(operands[0]);
                            inst.imm = parse_csr_name(operands[1]) as i64;
                            inst.rs1 = Some(0); // x0
                        }
                        return Ok(inst);
                    }
                    RvOpcode::CSRRWI | RvOpcode::CSRRSI | RvOpcode::CSRRCI => {
                        if operands.len() >= 3 {
                            inst.rd = parse_register_name(operands[0]);
                            inst.imm = parse_csr_name(operands[1]) as i64;
                            // zimm is a 5-bit unsigned immediate stored in the rs1
                            // field; encode as a register number directly.
                            let zimm = parse_asm_immediate(operands[2]).unwrap_or(0) as u16;
                            inst.rs1 = Some(zimm & 0x1F);
                        } else if operands.len() == 2 {
                            inst.rd = parse_register_name(operands[0]);
                            inst.imm = parse_csr_name(operands[1]) as i64;
                            inst.rs1 = Some(0);
                        }
                        return Ok(inst);
                    }
                    RvOpcode::SFENCE_VMA => {
                        // sfence.vma rs1, rs2
                        inst.rs1 = if !operands.is_empty() {
                            parse_register_name(operands[0])
                        } else {
                            Some(0)
                        };
                        inst.rs2 = if operands.len() > 1 {
                            parse_register_name(operands[1])
                        } else {
                            Some(0)
                        };
                        return Ok(inst);
                    }
                    RvOpcode::ECALL
                    | RvOpcode::EBREAK
                    | RvOpcode::FENCE
                    | RvOpcode::FENCE_I
                    | RvOpcode::WFI
                    | RvOpcode::MRET
                    | RvOpcode::SRET => {
                        // No operands needed — encoding is fixed.
                        return Ok(inst);
                    }
                    _ => {} // fall through to generic operand parsing
                }

                // --- Generic operand parsing for non-CSR/system instructions ---
                if !operands.is_empty() {
                    inst.rd = parse_register_name(operands[0]);
                }
                if operands.len() > 1 {
                    // Check for memory operand pattern: offset(reg)
                    if let Some((off, reg)) = parse_memory_operand(operands[1]) {
                        inst.rs1 = Some(reg);
                        inst.imm = off;
                    } else {
                        inst.rs1 = parse_register_name(operands[1]);
                    }
                }
                if operands.len() > 2 {
                    // Third operand could be register or immediate
                    if let Some(reg) = parse_register_name(operands[2]) {
                        inst.rs2 = Some(reg);
                    } else if let Some(imm) = parse_asm_immediate(operands[2]) {
                        inst.imm = imm;
                    } else {
                        // Could be a symbol reference
                        inst.symbol = Some(operands[2].to_string());
                    }
                }

                Ok(inst)
            }
        }
    }
}

// ===========================================================================
// Instruction word manipulation helpers (local to this module)
// ===========================================================================

/// Read a 32-bit little-endian value from a byte slice.
#[inline]
fn read_u32_le(code: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        code[offset],
        code[offset + 1],
        code[offset + 2],
        code[offset + 3],
    ])
}

/// Write a 32-bit little-endian value to a byte slice.
#[inline]
fn write_u32_le(code: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    code[offset..offset + 4].copy_from_slice(&bytes);
}

/// Insert a B-type immediate into an instruction word.
///
/// B-type immediate encoding (scrambled):
/// - bit 31 ← imm[12]
/// - bits 30:25 ← imm[10:5]
/// - bits 11:8 ← imm[4:1]
/// - bit 7 ← imm[11]
fn insert_b_imm(insn: u32, imm: i32) -> u32 {
    let imm_u = imm as u32;
    let bit12 = (imm_u >> 12) & 1;
    let bit11 = (imm_u >> 11) & 1;
    let bits10_5 = (imm_u >> 5) & 0x3F;
    let bits4_1 = (imm_u >> 1) & 0xF;
    let cleared = insn & 0x01FFF07F;
    cleared | (bit12 << 31) | (bits10_5 << 25) | (bits4_1 << 8) | (bit11 << 7)
}

/// Insert a J-type immediate into an instruction word.
///
/// J-type immediate encoding (scrambled):
/// - bit 31 ← imm[20]
/// - bits 30:21 ← imm[10:1]
/// - bit 20 ← imm[11]
/// - bits 19:12 ← imm[19:12]
fn insert_j_imm(insn: u32, imm: i32) -> u32 {
    let imm_u = imm as u32;
    let bit20 = (imm_u >> 20) & 1;
    let bits10_1 = (imm_u >> 1) & 0x3FF;
    let bit11 = (imm_u >> 11) & 1;
    let bits19_12 = (imm_u >> 12) & 0xFF;
    let cleared = insn & 0x00000FFF;
    cleared | (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12)
}

/// Insert a U-type upper 20-bit immediate (for LUI/AUIPC).
fn insert_u_imm(insn: u32, imm: i32) -> u32 {
    let cleared = insn & 0x00000FFF;
    cleared | ((imm as u32) & 0xFFFFF000)
}

/// Insert an I-type 12-bit immediate.
fn insert_i_imm(insn: u32, imm: i32) -> u32 {
    let cleared = insn & 0x000FFFFF;
    cleared | (((imm as u32) & 0xFFF) << 20)
}

/// Insert an S-type 12-bit split immediate.
fn insert_s_imm(insn: u32, imm: i32) -> u32 {
    let imm_u = (imm as u32) & 0xFFF;
    let upper = (imm_u >> 5) & 0x7F;
    let lower = imm_u & 0x1F;
    let cleared = insn & 0x01FFF07F;
    cleared | (upper << 25) | (lower << 7)
}

/// Compute upper 20-bit and lower 12-bit split for a value.
///
/// When the lower 12 bits are negative (bit 11 set), the upper 20 bits
/// are incremented by 1 to compensate for ADDI sign extension.
fn compute_hi_lo(value: i64) -> (i32, i32) {
    let lo = ((value & 0xFFF) as i32) << 20 >> 20;
    let hi = ((value.wrapping_add(0x800)) >> 12) as i32;
    (hi, lo)
}

// ===========================================================================
// Free-standing instruction patching functions (for local branch resolution)
//
// These are free functions that operate on the code buffer directly to avoid
// borrow conflicts when iterating over relocations while patching code.
// ===========================================================================

/// Patch a B-type branch instruction with a resolved PC-relative offset.
fn patch_branch_code(code: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    if !(-4096..4096).contains(&value) {
        return Err(format!(
            "B-type branch at offset 0x{:x}: target offset {} out of ±4KiB range",
            offset, value
        ));
    }
    if (value & 1) != 0 {
        return Err(format!(
            "B-type branch at offset 0x{:x}: target offset {} is not 2-byte aligned",
            offset, value
        ));
    }
    if offset + 4 > code.len() {
        return Err(format!(
            "B-type branch at offset 0x{:x}: out of code buffer bounds",
            offset
        ));
    }
    let insn = read_u32_le(code, offset);
    let patched = insert_b_imm(insn, value as i32);
    write_u32_le(code, offset, patched);
    Ok(())
}

/// Patch a J-type jump instruction with a resolved PC-relative offset.
fn patch_jal_code(code: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    if !(-(1i64 << 20)..(1i64 << 20)).contains(&value) {
        return Err(format!(
            "J-type jump at offset 0x{:x}: target offset {} out of ±1MiB range",
            offset, value
        ));
    }
    if (value & 1) != 0 {
        return Err(format!(
            "J-type jump at offset 0x{:x}: target offset {} is not 2-byte aligned",
            offset, value
        ));
    }
    if offset + 4 > code.len() {
        return Err(format!(
            "J-type jump at offset 0x{:x}: out of code buffer bounds",
            offset
        ));
    }
    let insn = read_u32_le(code, offset);
    let patched = insert_j_imm(insn, value as i32);
    write_u32_le(code, offset, patched);
    Ok(())
}

/// Patch an AUIPC+JALR call pair with a resolved PC-relative offset.
fn patch_call_code(code: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    if !(-(1i64 << 31)..(1i64 << 31)).contains(&value) {
        return Err(format!(
            "CALL at offset 0x{:x}: target offset {} out of ±2GiB range",
            offset, value
        ));
    }
    if offset + 8 > code.len() {
        return Err(format!(
            "CALL at offset 0x{:x}: need 8 bytes for AUIPC+JALR pair",
            offset
        ));
    }
    let (hi, lo) = compute_hi_lo(value);
    let auipc = read_u32_le(code, offset);
    let auipc_patched = insert_u_imm(auipc, hi << 12);
    write_u32_le(code, offset, auipc_patched);
    let jalr = read_u32_le(code, offset + 4);
    let jalr_patched = insert_i_imm(jalr, lo);
    write_u32_le(code, offset + 4, jalr_patched);
    Ok(())
}

/// Patch an AUIPC instruction with a PC-relative upper 20 bits.
fn patch_pcrel_hi20_code(code: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    if offset + 4 > code.len() {
        return Err(format!(
            "PCREL_HI20 at offset 0x{:x}: out of code buffer bounds",
            offset
        ));
    }
    let (hi, _lo) = compute_hi_lo(value);
    let insn = read_u32_le(code, offset);
    let patched = insert_u_imm(insn, hi << 12);
    write_u32_le(code, offset, patched);
    Ok(())
}

/// Patch an I-type instruction with a PC-relative lower 12 bits.
fn patch_pcrel_lo12_i_code(code: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    if offset + 4 > code.len() {
        return Err(format!(
            "PCREL_LO12_I at offset 0x{:x}: out of code buffer bounds",
            offset
        ));
    }
    let lo = (value & 0xFFF) as i32;
    let insn = read_u32_le(code, offset);
    let patched = insert_i_imm(insn, lo);
    write_u32_le(code, offset, patched);
    Ok(())
}

/// Patch an S-type instruction with a PC-relative lower 12 bits.
fn patch_pcrel_lo12_s_code(code: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    if offset + 4 > code.len() {
        return Err(format!(
            "PCREL_LO12_S at offset 0x{:x}: out of code buffer bounds",
            offset
        ));
    }
    let lo = (value & 0xFFF) as i32;
    let insn = read_u32_le(code, offset);
    let patched = insert_s_imm(insn, lo);
    write_u32_le(code, offset, patched);
    Ok(())
}

// ===========================================================================
// Inline assembly parsing helpers
// ===========================================================================

/// Parse a RISC-V register name (ABI or numeric) to a register ID.
///
/// Returns `None` if the string is not a valid register name.
fn parse_register_name(name: &str) -> Option<u16> {
    let name = name.trim();
    match name {
        // ABI integer register names
        "zero" => Some(registers::ZERO),
        "ra" => Some(registers::RA),
        "sp" => Some(registers::SP),
        "gp" => Some(registers::GP),
        "tp" => Some(registers::TP),
        "fp" | "s0" => Some(registers::FP),
        "s1" => Some(registers::S1),
        "s2" => Some(registers::S2),
        "s3" => Some(registers::S3),
        "s4" => Some(registers::S4),
        "s5" => Some(registers::S5),
        "s6" => Some(registers::S6),
        "s7" => Some(registers::S7),
        "s8" => Some(registers::S8),
        "s9" => Some(registers::S9),
        "s10" => Some(registers::S10),
        "s11" => Some(registers::S11),
        "a0" => Some(registers::A0),
        "a1" => Some(registers::A1),
        "a2" => Some(registers::A2),
        "a3" => Some(registers::A3),
        "a4" => Some(registers::A4),
        "a5" => Some(registers::A5),
        "a6" => Some(registers::A6),
        "a7" => Some(registers::A7),
        "t0" => Some(registers::T0),
        "t1" => Some(registers::T1),
        "t2" => Some(registers::T2),
        "t3" => Some(registers::T3),
        "t4" => Some(registers::T4),
        "t5" => Some(registers::T5),
        "t6" => Some(registers::T6),
        _ => {
            // Try numeric format: x0–x31
            if let Some(stripped) = name.strip_prefix('x') {
                if let Ok(n) = stripped.parse::<u16>() {
                    if n < 32 {
                        return Some(n);
                    }
                }
            }
            // Try FP format: f0–f31 or ft0–ft11, fs0–fs11, fa0–fa7
            if let Some(stripped) = name.strip_prefix('f') {
                if let Ok(n) = stripped.parse::<u16>() {
                    if n < 32 {
                        return Some(n + 32); // FP register IDs are 32-63
                    }
                }
            }
            // FP ABI names
            match name {
                "ft0" => Some(registers::FT0),
                "ft1" => Some(registers::FT1),
                "ft2" => Some(registers::FT2),
                "ft3" => Some(registers::FT3),
                "ft4" => Some(registers::FT4),
                "ft5" => Some(registers::FT5),
                "ft6" => Some(registers::FT6),
                "ft7" => Some(registers::FT7),
                "fs0" => Some(registers::FS0),
                "fs1" => Some(registers::FS1),
                "fa0" => Some(registers::FA0),
                "fa1" => Some(registers::FA1),
                "fa2" => Some(registers::FA2),
                "fa3" => Some(registers::FA3),
                "fa4" => Some(registers::FA4),
                "fa5" => Some(registers::FA5),
                "fa6" => Some(registers::FA6),
                "fa7" => Some(registers::FA7),
                "fs2" => Some(registers::FS2),
                "fs3" => Some(registers::FS3),
                "fs4" => Some(registers::FS4),
                "fs5" => Some(registers::FS5),
                "fs6" => Some(registers::FS6),
                "fs7" => Some(registers::FS7),
                "fs8" => Some(registers::FS8),
                "fs9" => Some(registers::FS9),
                "fs10" => Some(registers::FS10),
                "fs11" => Some(registers::FS11),
                "ft8" => Some(registers::FT8),
                "ft9" => Some(registers::FT9),
                "ft10" => Some(registers::FT10),
                "ft11" => Some(registers::FT11),
                _ => None,
            }
        }
    }
}

/// Parse a memory operand of the form `offset(register)`.
///
/// Returns `(offset, register_id)` or `None` if not a memory operand.
fn parse_memory_operand(s: &str) -> Option<(i64, u16)> {
    let s = s.trim();
    if let Some(paren_start) = s.find('(') {
        if s.ends_with(')') {
            let offset_str = &s[..paren_start];
            let reg_str = &s[paren_start + 1..s.len() - 1];

            let offset = if offset_str.is_empty() {
                0i64
            } else {
                parse_asm_immediate(offset_str)?
            };

            let reg = parse_register_name(reg_str)?;
            return Some((offset, reg));
        }
    }
    None
}

/// Parse an assembly immediate value (decimal, hex, octal, or binary).
fn parse_asm_immediate(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Check for negative sign
    let (negative, s) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else {
        (false, s)
    };

    let value = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()?
    } else if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        i64::from_str_radix(bin, 2).ok()?
    } else if s.starts_with('0') && s.len() > 1 && s.chars().all(|c| c.is_ascii_digit()) {
        i64::from_str_radix(s, 8).ok()?
    } else {
        s.parse::<i64>().ok()?
    };

    Some(if negative { -value } else { value })
}

/// Parse a CSR name or numeric CSR address into a 12-bit CSR number.
///
/// Handles both symbolic names used by the Linux kernel (e.g., `sstatus`,
/// `mtvec`, `satp`) and numeric CSR addresses (e.g., `0x100`, `768`).
/// Returns 0 for unrecognized names to avoid blocking compilation — the
/// kernel uses many CSR aliases and the set may grow.
fn parse_csr_name(name: &str) -> u32 {
    let name = name.trim();

    // Try numeric parse first (decimal, hex).
    if let Some(hex) = name.strip_prefix("0x").or_else(|| name.strip_prefix("0X")) {
        if let Ok(v) = u32::from_str_radix(hex, 16) {
            return v & 0xFFF;
        }
    }
    if let Ok(v) = name.parse::<u32>() {
        return v & 0xFFF;
    }

    // Symbolic names — covers the CSRs commonly used by the Linux kernel.
    // Machine-level CSRs (M-mode, 0x3xx)
    match name {
        // User trap handling
        "ustatus" => 0x000,
        "uie" => 0x004,
        "utvec" => 0x005,
        // User trap registers
        "uscratch" => 0x040,
        "uepc" => 0x041,
        "ucause" => 0x042,
        "utval" => 0x043,
        "uip" => 0x044,
        // Supervisor trap setup
        "sstatus" => 0x100,
        "sedeleg" => 0x102,
        "sideleg" => 0x103,
        "sie" => 0x104,
        "stvec" => 0x105,
        "scounteren" => 0x106,
        // Supervisor trap handling
        "sscratch" => 0x140,
        "sepc" => 0x141,
        "scause" => 0x142,
        "stval" => 0x143,
        "sip" => 0x144,
        // Supervisor protection and translation
        "satp" => 0x180,
        // Machine information registers
        "mvendorid" => 0xF11,
        "marchid" => 0xF12,
        "mimpid" => 0xF13,
        "mhartid" => 0xF14,
        // Machine trap setup
        "mstatus" => 0x300,
        "misa" => 0x301,
        "medeleg" => 0x302,
        "mideleg" => 0x303,
        "mie" => 0x304,
        "mtvec" => 0x305,
        "mcounteren" => 0x306,
        // Machine trap handling
        "mscratch" => 0x340,
        "mepc" => 0x341,
        "mcause" => 0x342,
        "mtval" => 0x343,
        "mip" => 0x344,
        // Machine counter / timers
        "mcycle" => 0xB00,
        "minstret" => 0xB02,
        "cycle" => 0xC00,
        "time" => 0xC01,
        "instret" => 0xC02,
        "cycleh" => 0xC80,
        "timeh" => 0xC81,
        "instreth" => 0xC82,
        // Floating-point CSRs
        "fflags" => 0x001,
        "frm" => 0x002,
        "fcsr" => 0x003,
        // Supervisor address translation and protection (RISC-V H extension)
        "hstatus" => 0x600,
        "hedeleg" => 0x602,
        "hideleg" => 0x603,
        "hie" => 0x604,
        "htimedelta" => 0x605,
        "hcounteren" => 0x606,
        "hgeie" => 0x607,
        "htval" => 0x643,
        "hip" => 0x644,
        "hvip" => 0x645,
        "htinst" => 0x64A,
        "hgeip" => 0xE12,
        "henvcfg" => 0x60A,
        "henvcfgh" => 0x61A,
        "hgatp" => 0x680,
        // Supervisor environment configuration
        "senvcfg" => 0x10A,
        // Machine environment configuration
        "menvcfg" => 0x30A,
        "menvcfgh" => 0x31A,
        // Performance monitoring
        "scountovf" => 0xDA0,
        _ => 0, // default to 0 for unrecognized names
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembler_new() {
        let asm = RiscV64Assembler::new(false);
        assert_eq!(asm.current_offset(), 0);
        assert!(asm.code.is_empty());
        assert!(!asm.pic_mode);
        assert_eq!(asm.target, Target::RiscV64);
    }

    #[test]
    fn test_assembler_new_pic() {
        let asm = RiscV64Assembler::new(true);
        assert!(asm.pic_mode);
    }

    #[test]
    fn test_define_label() {
        let mut asm = RiscV64Assembler::new(false);
        asm.define_label("test_label");
        assert_eq!(asm.symbols.get("test_label"), Some(&0));

        asm.emit_nop();
        asm.define_label("after_nop");
        assert_eq!(asm.symbols.get("after_nop"), Some(&4));
    }

    #[test]
    fn test_emit_nop() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_nop();
        assert_eq!(asm.code, vec![0x13, 0x00, 0x00, 0x00]);
        assert_eq!(asm.current_offset(), 4);
    }

    #[test]
    fn test_emit_raw_bytes() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_raw_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(asm.code, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(asm.current_offset(), 4);
    }

    #[test]
    fn test_align_to() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_raw_bytes(&[0x01]); // 1 byte
        asm.align_to(4);
        // Should pad to 4 bytes total: 1 original + 3 padding
        assert_eq!(asm.current_offset(), 4);
    }

    #[test]
    fn test_align_already_aligned() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_nop(); // 4 bytes, already aligned to 4
        let prev_offset = asm.current_offset();
        asm.align_to(4);
        assert_eq!(asm.current_offset(), prev_offset);
    }

    #[test]
    fn test_reset() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_nop();
        asm.define_label("test");
        asm.reset();
        assert_eq!(asm.current_offset(), 0);
        assert!(asm.code.is_empty());
        assert!(asm.symbols.is_empty());
        assert!(asm.relocations.is_empty());
    }

    #[test]
    fn test_nop_encoding_matches_spec() {
        // ADDI x0, x0, 0 = 0x00000013 in little-endian
        assert_eq!(NOP_ENCODING, [0x13, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_assemble_nop_instruction() {
        let mut asm = RiscV64Assembler::new(false);
        let nop = RvInstruction {
            opcode: RvOpcode::NOP,
            rd: None,
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        };
        let result = asm.assemble_single(&nop);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4);
        // NOP = ADDI x0, x0, 0 = 0x00000013
        assert_eq!(asm.code, vec![0x13, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_build_text_section() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_nop();
        asm.emit_nop();
        let (code, relocs) = asm.build_text_section();
        assert_eq!(code.len(), 8);
        assert!(relocs.is_empty());
    }

    #[test]
    fn test_finalize() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_nop();
        asm.define_label("test_sym");
        asm.emit_nop();
        let result = asm.finalize();
        assert_eq!(result.code_size, 8);
        assert_eq!(result.symbols.get("test_sym"), Some(&4));
    }

    #[test]
    fn test_call_relocation_non_pic() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_call_relocation("target_func", 0);
        assert_eq!(asm.relocations.len(), 2);
        assert_eq!(asm.relocations[0].reloc_type, R_RISCV_CALL);
        assert!(asm.relocations[0].relaxable);
        assert_eq!(asm.relocations[1].reloc_type, R_RISCV_RELAX);
    }

    #[test]
    fn test_call_relocation_pic() {
        let mut asm = RiscV64Assembler::new(true);
        asm.emit_call_relocation("target_func", 0);
        assert_eq!(asm.relocations[0].reloc_type, R_RISCV_CALL_PLT);
    }

    #[test]
    fn test_pcrel_relocation() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_pcrel_relocation("my_global", 0, true);
        assert_eq!(asm.relocations.len(), 3);
        assert_eq!(asm.relocations[0].reloc_type, R_RISCV_PCREL_HI20);
        assert_eq!(asm.relocations[1].reloc_type, R_RISCV_RELAX);
        assert_eq!(asm.relocations[2].reloc_type, R_RISCV_PCREL_LO12_I);
        assert_eq!(asm.relocations[2].offset, 4);
    }

    #[test]
    fn test_got_relocation() {
        let mut asm = RiscV64Assembler::new(true);
        asm.emit_got_relocation("external_sym", 0);
        assert_eq!(asm.relocations.len(), 3);
        assert_eq!(asm.relocations[0].reloc_type, R_RISCV_GOT_HI20);
        assert_eq!(asm.relocations[1].reloc_type, R_RISCV_RELAX);
        assert_eq!(asm.relocations[2].reloc_type, R_RISCV_PCREL_LO12_I);
    }

    #[test]
    fn test_branch_relocation() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_branch_relocation("loop_start", 0, false);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].reloc_type, R_RISCV_BRANCH);
    }

    #[test]
    fn test_jal_relocation() {
        let mut asm = RiscV64Assembler::new(false);
        asm.emit_branch_relocation("far_target", 0, true);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].reloc_type, R_RISCV_JAL);
    }

    #[test]
    fn test_parse_register_name() {
        assert_eq!(parse_register_name("zero"), Some(0));
        assert_eq!(parse_register_name("ra"), Some(1));
        assert_eq!(parse_register_name("sp"), Some(2));
        assert_eq!(parse_register_name("fp"), Some(8));
        assert_eq!(parse_register_name("s0"), Some(8));
        assert_eq!(parse_register_name("a0"), Some(10));
        assert_eq!(parse_register_name("t0"), Some(5));
        assert_eq!(parse_register_name("x0"), Some(0));
        assert_eq!(parse_register_name("x31"), Some(31));
        assert_eq!(parse_register_name("invalid"), None);
    }

    #[test]
    fn test_parse_asm_immediate() {
        assert_eq!(parse_asm_immediate("42"), Some(42));
        assert_eq!(parse_asm_immediate("-1"), Some(-1));
        assert_eq!(parse_asm_immediate("0x1F"), Some(31));
        assert_eq!(parse_asm_immediate("0b1010"), Some(10));
        assert_eq!(parse_asm_immediate(""), None);
    }

    #[test]
    fn test_parse_memory_operand() {
        assert_eq!(parse_memory_operand("0(sp)"), Some((0, 2)));
        assert_eq!(parse_memory_operand("8(a0)"), Some((8, 10)));
        assert_eq!(parse_memory_operand("-16(fp)"), Some((-16, 8)));
        assert_eq!(parse_memory_operand("(sp)"), Some((0, 2)));
        assert_eq!(parse_memory_operand("not_memory"), None);
    }

    #[test]
    fn test_compute_hi_lo() {
        // Simple positive value
        let (hi, lo) = compute_hi_lo(0x12345);
        assert_eq!((hi as i64) << 12 | ((lo as i64) & 0xFFF), 0x12345);

        // Value requiring sign-extension compensation
        let (hi, lo) = compute_hi_lo(0x1800);
        let reconstructed = ((hi as i64) << 12).wrapping_add(lo as i64);
        assert_eq!(reconstructed, 0x1800);
    }

    #[test]
    fn test_insert_b_imm() {
        // Test with offset = 0
        let insn = 0x00000063u32; // Base BEQ x0, x0
        let patched = insert_b_imm(insn, 0);
        assert_eq!(patched, 0x00000063);

        // Test with offset = 4
        let patched = insert_b_imm(insn, 4);
        // bits4_1 = 0b0010, encoded in bits[11:8]
        assert_ne!(patched, 0x00000063);
    }

    #[test]
    fn test_relaxable_relocation_classification() {
        let asm = RiscV64Assembler::new(false);
        assert!(asm.is_relaxable_relocation(&RiscV64RelocationType::Call));
        assert!(asm.is_relaxable_relocation(&RiscV64RelocationType::CallPlt));
        assert!(asm.is_relaxable_relocation(&RiscV64RelocationType::PcrelHi20));
        assert!(asm.is_relaxable_relocation(&RiscV64RelocationType::GotHi20));
        assert!(!asm.is_relaxable_relocation(&RiscV64RelocationType::Branch));
        assert!(!asm.is_relaxable_relocation(&RiscV64RelocationType::Jal));
    }

    #[test]
    fn test_relocation_constants() {
        // Verify our constants match RiscV64RelocationType::as_elf_type()
        assert_eq!(R_RISCV_BRANCH, RiscV64RelocationType::Branch.as_elf_type());
        assert_eq!(R_RISCV_JAL, RiscV64RelocationType::Jal.as_elf_type());
        assert_eq!(R_RISCV_CALL, RiscV64RelocationType::Call.as_elf_type());
        assert_eq!(
            R_RISCV_CALL_PLT,
            RiscV64RelocationType::CallPlt.as_elf_type()
        );
        assert_eq!(
            R_RISCV_PCREL_HI20,
            RiscV64RelocationType::PcrelHi20.as_elf_type()
        );
        assert_eq!(
            R_RISCV_PCREL_LO12_I,
            RiscV64RelocationType::PcrelLo12I.as_elf_type()
        );
        assert_eq!(
            R_RISCV_PCREL_LO12_S,
            RiscV64RelocationType::PcrelLo12S.as_elf_type()
        );
        assert_eq!(
            R_RISCV_GOT_HI20,
            RiscV64RelocationType::GotHi20.as_elf_type()
        );
        assert_eq!(R_RISCV_HI20, RiscV64RelocationType::Hi20.as_elf_type());
        assert_eq!(R_RISCV_LO12_I, RiscV64RelocationType::Lo12I.as_elf_type());
        assert_eq!(R_RISCV_LO12_S, RiscV64RelocationType::Lo12S.as_elf_type());
        assert_eq!(R_RISCV_RELAX, RiscV64RelocationType::Relax.as_elf_type());
        assert_eq!(R_RISCV_ALIGN, RiscV64RelocationType::Align.as_elf_type());
    }
}
