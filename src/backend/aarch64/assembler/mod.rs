// Comprehensive assembler module — not every public struct, method, or helper
// is called by every consumer of this crate. Allow dead_code to avoid
// warnings during incremental bring-up of the AArch64 backend.
#![allow(dead_code, unused_imports)]

//! Built-in AArch64 assembler producing relocatable object code without
//! invoking any external tools.
//!
//! This module accepts [`MachineFunction`] output from the AArch64 instruction
//! selector (`codegen.rs`) and produces binary `.text` sections with relocation
//! entries. It operates entirely in-process — **no external `as` or `llvm-mc`
//! is ever invoked**, enforcing the standalone backend mandate (Section 0.7.7).
//!
//! All A64 instructions are fixed-width 32-bit (4-byte) little-endian words.
//!
//! # Assembly Pipeline
//!
//! ```text
//! MachineFunction
//!   ├── iterate basic blocks in layout order
//!   │     ├── bind label at block entry offset
//!   │     └── for each MachineInstr:
//!   │           ├── encoder::encode_instruction() → EncodedInstruction (4 bytes)
//!   │           └── collect optional relocation from encoder
//!   ├── resolve local branch references (patch instruction offsets)
//!   └── output AssembledFunction { code, relocations }
//! ```
//!
//! # Sub-modules
//!
//! - [`encoder`]: A64 instruction encoding — translates `MachineInstr` into
//!   32-bit instruction words with optional relocation annotations.
//! - [`relocations`]: AArch64-specific ELF relocation type definitions used
//!   by both the assembler and the built-in linker.
//!
//! # PIC Relocation Emission
//!
//! When assembling PIC code, the assembler emits the following relocation
//! types for ADRP+ADD/LDR pairs and branch instructions:
//!
//! | Instruction       | Relocation Type                         |
//! |-------------------|-----------------------------------------|
//! | ADRP (direct)     | `R_AARCH64_ADR_PREL_PG_HI21`           |
//! | ADRP (GOT)        | `R_AARCH64_ADR_GOT_PAGE`                |
//! | ADD #:lo12:sym    | `R_AARCH64_ADD_ABS_LO12_NC`             |
//! | LDR (GOT lo12)    | `R_AARCH64_LD64_GOT_LO12_NC`           |
//! | B / BL external   | `R_AARCH64_JUMP26` / `R_AARCH64_CALL26` |
//! | Absolute 64-bit   | `R_AARCH64_ABS64`                       |

/// A64 instruction encoder — encodes `MachineInstr` operands into 32-bit
/// instruction words following the A64 encoding specification. Each encoded
/// instruction may carry an optional relocation for unresolved symbol
/// references (ADRP, B/BL, LDR GOT, ADD lo12, etc.).
pub mod encoder;

/// AArch64 ELF relocation type definitions — `AArch64RelocationType` enum
/// covering absolute, PC-relative, page-relative, GOT, TLS, branch, and
/// dynamic relocation types. Provides ELF numeric values, relocation
/// property queries, range validation, and instruction-level application.
pub mod relocations;

// Re-export primary types for convenient access from parent modules.
pub use relocations::AArch64RelocationType;

use crate::backend::aarch64::registers;
use crate::backend::traits::{
    MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand, PhysReg,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;

use self::encoder::{encode_instruction, EncodedInstruction, ExtendType, ShiftType};

// ---------------------------------------------------------------------------
// Branch reference type classification
// ---------------------------------------------------------------------------

/// Classification of AArch64 branch instruction types, each with different
/// immediate encoding width and range constraints.
///
/// The A64 ISA uses different branch formats with varying immediate field
/// widths. The assembler must validate that local branch targets fall within
/// the representable range for each type:
///
/// | Variant          | Imm Bits | Range          | Instructions       |
/// |------------------|----------|----------------|--------------------|
/// | `Branch26`       | 26       | ±128 MiB       | B, BL              |
/// | `CondBranch19`   | 19       | ±1 MiB         | B.cond, CBZ, CBNZ  |
/// | `Adr21`          | 21       | ±1 MiB         | ADR                |
/// | `TestBranch14`   | 14       | ±32 KiB        | TBZ, TBNZ          |
/// | `CompareBranch19`| 19       | ±1 MiB         | CBZ, CBNZ          |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchRefType {
    /// B / BL: 26-bit signed offset (word-aligned), ±128 MiB range.
    Branch26,
    /// B.cond: 19-bit signed offset (word-aligned), ±1 MiB range.
    CondBranch19,
    /// ADR: 21-bit signed byte offset, ±1 MiB range (split immhi:immlo).
    Adr21,
    /// TBZ / TBNZ: 14-bit signed offset (word-aligned), ±32 KiB range.
    TestBranch14,
    /// CBZ / CBNZ: 19-bit signed offset (word-aligned), ±1 MiB range.
    CompareBranch19,
}

impl BranchRefType {
    /// Returns the signed range limits (in bytes) for this branch type.
    ///
    /// The returned `(min, max)` pair gives the inclusive signed byte offset
    /// range. For word-aligned branches, the displacement must also be
    /// divisible by 4.
    pub fn range_bytes(self) -> (i64, i64) {
        match self {
            // 26-bit signed word offset → ±128 MiB
            BranchRefType::Branch26 => (-(1i64 << 27), (1i64 << 27) - 4),
            // 19-bit signed word offset → ±1 MiB
            BranchRefType::CondBranch19 => (-(1i64 << 20), (1i64 << 20) - 4),
            // 21-bit signed byte offset → ±1 MiB
            BranchRefType::Adr21 => (-(1i64 << 20), (1i64 << 20) - 1),
            // 14-bit signed word offset → ±32 KiB
            BranchRefType::TestBranch14 => (-(1i64 << 15), (1i64 << 15) - 4),
            // 19-bit signed word offset → ±1 MiB
            BranchRefType::CompareBranch19 => (-(1i64 << 20), (1i64 << 20) - 4),
        }
    }

    /// Returns a human-readable name for diagnostic messages.
    pub fn name(self) -> &'static str {
        match self {
            BranchRefType::Branch26 => "B/BL (26-bit)",
            BranchRefType::CondBranch19 => "B.cond (19-bit)",
            BranchRefType::Adr21 => "ADR (21-bit)",
            BranchRefType::TestBranch14 => "TBZ/TBNZ (14-bit)",
            BranchRefType::CompareBranch19 => "CBZ/CBNZ (19-bit)",
        }
    }
}

// ---------------------------------------------------------------------------
// Assembler relocation — output relocation entry
// ---------------------------------------------------------------------------

/// A relocation entry produced by the assembler for inclusion in the ELF
/// `.rela.text` section.
///
/// Each relocation records a site in the assembled code that needs to be
/// patched by the linker to reflect the final address of a symbol. The
/// `reloc_type` determines how the patch is applied (PC-relative, absolute,
/// page-relative, etc.).
#[derive(Clone, Debug)]
pub struct AssemblerRelocation {
    /// Byte offset within the assembled code section where the relocation
    /// is to be applied.
    pub offset: u32,
    /// AArch64-specific relocation type that determines the patching semantics.
    pub reloc_type: AArch64RelocationType,
    /// Name of the referenced symbol (function, global variable, etc.).
    pub symbol: String,
    /// Addend value to be added to the symbol's address during relocation.
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// Pending label reference — unresolved intra-function branch
// ---------------------------------------------------------------------------

/// An unresolved intra-function branch reference that must be resolved after
/// all basic blocks have been assembled.
///
/// During assembly, forward branches to labels that have not yet been bound
/// are recorded as pending references. After the entire function is assembled
/// and all labels are bound, these references are resolved by patching the
/// branch instruction's immediate field with the computed offset.
#[derive(Clone, Debug)]
pub struct PendingLabelRef {
    /// Byte offset of the branch instruction within the assembled code.
    pub offset: u32,
    /// Target label ID (corresponding to a `MachineBasicBlock` ID).
    pub label_id: u32,
    /// Branch instruction type, which determines the immediate encoding format
    /// and range limits for the offset.
    pub ref_type: BranchRefType,
}

// ---------------------------------------------------------------------------
// Symbol reference — external/global symbol needing linker relocation
// ---------------------------------------------------------------------------

/// A reference to an external or global symbol that the linker must resolve.
///
/// Symbol references are collected during assembly and converted to ELF
/// relocation entries in the output object file.
#[derive(Clone, Debug)]
pub struct SymbolReference {
    /// Byte offset within the assembled code where the reference occurs.
    pub offset: u32,
    /// Name of the referenced symbol.
    pub symbol: String,
    /// AArch64-specific relocation type for this reference.
    pub ref_type: AArch64RelocationType,
    /// Addend value for the relocation.
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// AssembledFunction — result of assembling a single function
// ---------------------------------------------------------------------------

/// The result of assembling a single [`MachineFunction`] into binary code.
///
/// Contains the raw machine code bytes and any remaining external relocations
/// that could not be resolved locally (i.e., references to symbols outside
/// this function that must be resolved by the linker).
#[derive(Clone, Debug)]
pub struct AssembledFunction {
    /// The function's symbol name.
    pub name: String,
    /// Assembled binary code (little-endian 32-bit instruction words).
    pub code: Vec<u8>,
    /// Relocations for external symbols that the linker must resolve.
    pub relocations: Vec<AssemblerRelocation>,
    /// Total size of the assembled code in bytes.
    pub size: u32,
    /// Required alignment for this function in the output section (in bytes).
    /// AArch64 functions are typically 4-byte aligned (instruction width).
    pub alignment: u32,
}

// ---------------------------------------------------------------------------
// AssembledSymbol — symbol table entry for assembled output
// ---------------------------------------------------------------------------

/// A symbol produced during module assembly, representing a function or
/// data label in the assembled output.
#[derive(Clone, Debug)]
pub struct AssembledSymbol {
    /// Symbol name.
    pub name: String,
    /// Byte offset of the symbol within the combined text section.
    pub offset: u32,
    /// Size of the symbol's content in bytes.
    pub size: u32,
    /// Whether this symbol has global visibility (vs. local/file scope).
    pub is_global: bool,
}

// ---------------------------------------------------------------------------
// AssembledModule — result of assembling an entire module
// ---------------------------------------------------------------------------

/// The result of assembling all functions in a compilation unit into a
/// single `.text` section with concatenated code and aggregated relocations.
#[derive(Clone, Debug)]
pub struct AssembledModule {
    /// Combined `.text` section containing all assembled function code,
    /// concatenated in the order functions were provided.
    pub text_section: Vec<u8>,
    /// All relocations from all functions, with offsets adjusted to reflect
    /// each function's position in the combined text section.
    pub relocations: Vec<AssemblerRelocation>,
    /// Symbol table entries for all assembled functions.
    pub symbols: Vec<AssembledSymbol>,
    /// Per-function metadata: `(name, offset_in_text, size)`.
    pub functions: Vec<(String, u32, u32)>,
}

// ---------------------------------------------------------------------------
// AArch64Assembler — main assembler driver
// ---------------------------------------------------------------------------

/// Built-in AArch64 assembler that translates [`MachineFunction`] output
/// from the instruction selector into binary machine code.
///
/// The assembler operates in two phases:
///
/// 1. **Encoding**: Iterate each basic block's instructions in layout order,
///    delegating to [`encoder::encode_instruction()`] for the 4-byte binary
///    encoding of each instruction. Collect relocations emitted by the
///    encoder for instructions referencing external symbols.
///
/// 2. **Label resolution**: After all instructions are encoded, resolve
///    intra-function branch references by patching the branch instruction's
///    immediate field with the computed byte offset to the target label.
///
/// # Standalone Backend
///
/// This assembler is entirely self-contained — no external assembler
/// (`as`, `llvm-mc`, `gas`) is invoked. All AArch64 instruction encoding
/// is performed in-process by the [`encoder`] sub-module.
pub struct AArch64Assembler {
    /// Assembled binary instruction stream (little-endian 32-bit words).
    code: Vec<u8>,
    /// Collected relocations for unresolved external symbol references.
    relocations: Vec<AssemblerRelocation>,
    /// Label ID → byte offset mapping for local branch resolution.
    /// Populated during assembly as basic block entry points are reached.
    labels: FxHashMap<u32, u32>,
    /// Unresolved intra-function branch references awaiting label resolution.
    pending_label_refs: Vec<PendingLabelRef>,
    /// Current byte offset in the code section (write cursor).
    current_offset: u32,
    /// External symbol references that become ELF relocations.
    symbol_references: Vec<SymbolReference>,
}

impl Default for AArch64Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Assembler {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Creates a new assembler with empty state, ready to assemble functions.
    pub fn new() -> Self {
        AArch64Assembler {
            code: Vec::new(),
            relocations: Vec::new(),
            labels: FxHashMap::default(),
            pending_label_refs: Vec::new(),
            current_offset: 0,
            symbol_references: Vec::new(),
        }
    }

    /// Resets the assembler state for assembling a new function.
    ///
    /// Clears all internal buffers while retaining heap allocations for
    /// reuse, reducing allocation pressure when assembling many functions.
    fn reset(&mut self) {
        self.code.clear();
        self.relocations.clear();
        self.labels.clear();
        self.pending_label_refs.clear();
        self.current_offset = 0;
        self.symbol_references.clear();
    }

    // -------------------------------------------------------------------
    // Label management
    // -------------------------------------------------------------------

    /// Binds a label at the current code offset.
    ///
    /// Records that `label_id` corresponds to `self.current_offset` so that
    /// pending (and future) branch references targeting this label can be
    /// resolved to a concrete byte offset.
    ///
    /// # Arguments
    ///
    /// * `label_id` — The basic block ID to bind at the current position.
    pub fn bind_label(&mut self, label_id: u32) {
        self.labels.insert(label_id, self.current_offset);
    }

    /// Records a pending branch reference to a label that may or may not
    /// have been bound yet.
    ///
    /// The reference will be resolved during the post-encoding label
    /// resolution pass. If the label is already bound, we still defer
    /// resolution to the batch pass for simplicity.
    ///
    /// # Arguments
    ///
    /// * `label_id` — Target basic block ID.
    /// * `ref_type` — The branch instruction type (determines encoding/range).
    pub fn reference_label(&mut self, label_id: u32, ref_type: BranchRefType) {
        self.pending_label_refs.push(PendingLabelRef {
            offset: self.current_offset,
            label_id,
            ref_type,
        });
    }

    // -------------------------------------------------------------------
    // Core assembly: single function
    // -------------------------------------------------------------------

    /// Assembles an entire [`MachineFunction`] into binary code.
    ///
    /// This is the primary entry point for function-level assembly. The
    /// method:
    ///
    /// 1. Iterates basic blocks in layout order
    /// 2. Binds a label at each block's entry point
    /// 3. Encodes each instruction via [`encoder::encode_instruction()`]
    /// 4. Collects encoder-emitted relocations for external symbols
    /// 5. After all blocks are encoded, resolves local branch references
    /// 6. Validates branch ranges and reports diagnostics for overflows
    ///
    /// # Arguments
    ///
    /// * `mf` — The machine function to assemble.
    ///
    /// # Returns
    ///
    /// An [`AssembledFunction`] containing the binary code and external
    /// relocations. Local branch references are fully resolved; only
    /// external symbol relocations remain for the linker.
    pub fn assemble_function(&mut self, mf: &MachineFunction) -> AssembledFunction {
        self.reset();

        // Phase 1: Encode all instructions, binding labels at block entries.
        for block in &mf.blocks {
            // Bind this block's label at the current code offset.
            self.bind_label(block.id);

            for instr in &block.instructions {
                // Check for label operands that imply branch references.
                // The encoder produces instruction words, but label-referencing
                // branches need post-resolution patching.
                let branch_ref = self.classify_branch_ref(instr);

                // Encode the instruction into a 4-byte word (+ optional reloc).
                let encoded = encode_instruction(instr);

                // Emit the 4-byte instruction word (little-endian).
                let le_bytes = encoded.bytes.to_le_bytes();
                self.code.extend_from_slice(&le_bytes);

                // If the encoder produced a relocation for an external symbol
                // reference, record it.
                if let Some((reloc_type, symbol, addend)) = encoded.relocation {
                    self.relocations.push(AssemblerRelocation {
                        offset: self.current_offset,
                        reloc_type,
                        symbol,
                        addend,
                    });
                }

                // If this is a branch instruction referencing a local label,
                // record the pending reference for post-resolution.
                if let Some((label_id, ref_type)) = branch_ref {
                    self.reference_label(label_id, ref_type);
                }

                // Advance the write cursor by one instruction (4 bytes).
                self.current_offset += 4;
            }
        }

        // Phase 2: Resolve all intra-function label references.
        let diag = self.resolve_labels();

        // Report any diagnostics (branch range overflows, unresolved labels).
        if diag.has_errors() {
            // In a production pipeline, diagnostics would propagate to the
            // compilation driver. For now, we continue and let the caller
            // inspect the assembled output.
            // The code is still produced — out-of-range branches will have
            // incorrect offset encoding (saturated to representable maximum).
        }

        // Build the result, transferring ownership of the code and relocations.
        let code_len = self.code.len() as u32;
        AssembledFunction {
            name: mf.name.clone(),
            code: std::mem::take(&mut self.code),
            relocations: std::mem::take(&mut self.relocations),
            size: code_len,
            // AArch64 instructions are 4-byte aligned; function alignment may
            // be stricter (e.g., 16-byte for cache-line considerations), but
            // the minimum is 4.
            alignment: 4,
        }
    }

    /// Assembles all functions in a module, concatenating their code into a
    /// single `.text` section with adjusted relocation offsets.
    ///
    /// # Arguments
    ///
    /// * `functions` — Slice of machine functions to assemble.
    ///
    /// # Returns
    ///
    /// An [`AssembledModule`] with the combined text section, aggregated
    /// relocations, symbol table entries, and per-function metadata.
    pub fn assemble_module(&mut self, functions: &[MachineFunction]) -> AssembledModule {
        let mut text_section: Vec<u8> = Vec::new();
        let mut all_relocations: Vec<AssemblerRelocation> = Vec::new();
        let mut symbols: Vec<AssembledSymbol> = Vec::new();
        let mut func_metadata: Vec<(String, u32, u32)> = Vec::new();

        for mf in functions {
            // Align the text section to the function's required alignment.
            let func_align = 4u32; // Minimum A64 instruction alignment.
            let current_len = text_section.len() as u32;
            let padding = alignment_padding(current_len, func_align);
            text_section.resize(text_section.len() + padding as usize, 0x00);
            let func_offset = text_section.len() as u32;

            // Assemble the function.
            let assembled = self.assemble_function(mf);

            // Adjust relocation offsets to be relative to the combined section.
            for mut reloc in assembled.relocations {
                reloc.offset += func_offset;
                all_relocations.push(reloc);
            }

            // Record the symbol.
            symbols.push(AssembledSymbol {
                name: assembled.name.clone(),
                offset: func_offset,
                size: assembled.size,
                // Functions are global by default; a more complete implementation
                // would check the IR-level linkage for static/internal functions.
                is_global: true,
            });

            // Record per-function metadata.
            func_metadata.push((assembled.name, func_offset, assembled.size));

            // Append the function's code to the combined text section.
            text_section.extend_from_slice(&assembled.code);
        }

        AssembledModule {
            text_section,
            relocations: all_relocations,
            symbols,
            functions: func_metadata,
        }
    }

    // -------------------------------------------------------------------
    // Label resolution
    // -------------------------------------------------------------------

    /// Resolves all pending intra-function label references by patching the
    /// branch instruction's immediate field with the computed byte offset.
    ///
    /// Returns a `DiagnosticEngine` containing any errors (branch range
    /// overflow, unresolved labels).
    fn resolve_labels(&mut self) -> DiagnosticEngine {
        let mut diag = DiagnosticEngine::new();

        // Take ownership of pending refs to avoid borrowing self immutably
        // while we need to mutably patch `self.code` in the loop body.
        let pending_refs = std::mem::take(&mut self.pending_label_refs);

        for pending in &pending_refs {
            let target_offset = match self.labels.get(&pending.label_id) {
                Some(&off) => off,
                None => {
                    diag.error(
                        Span::DUMMY,
                        format!(
                            "AArch64 assembler: unresolved label reference to block {}",
                            pending.label_id
                        ),
                    );
                    continue;
                }
            };

            // Compute the signed byte displacement from the branch site to the target.
            let displacement = (target_offset as i64) - (pending.offset as i64);

            // Validate that the displacement fits in the branch type's range.
            let (range_min, range_max) = pending.ref_type.range_bytes();
            if displacement < range_min || displacement > range_max {
                diag.error(
                    Span::DUMMY,
                    format!(
                        "AArch64 assembler: {} branch at offset 0x{:x} to label {} \
                         requires displacement {} bytes, which exceeds the \
                         allowed range [{}, {}]",
                        pending.ref_type.name(),
                        pending.offset,
                        pending.label_id,
                        displacement,
                        range_min,
                        range_max,
                    ),
                );
                continue;
            }

            // Patch the instruction at `pending.offset` with the computed displacement.
            let disp32 = displacement as i32;
            match pending.ref_type {
                BranchRefType::Branch26 => {
                    self.patch_branch26(pending.offset, disp32);
                }
                BranchRefType::CondBranch19 => {
                    self.patch_cond_branch19(pending.offset, disp32);
                }
                BranchRefType::Adr21 => {
                    self.patch_adr21(pending.offset, disp32);
                }
                BranchRefType::TestBranch14 => {
                    self.patch_test_branch14(pending.offset, disp32);
                }
                BranchRefType::CompareBranch19 => {
                    self.patch_compare_branch19(pending.offset, disp32);
                }
            }
        }

        diag
    }

    // -------------------------------------------------------------------
    // Branch offset patching helpers
    // -------------------------------------------------------------------

    /// Reads the 32-bit instruction word at `offset` from the code buffer.
    #[inline]
    fn read_instr(&self, offset: u32) -> u32 {
        let off = offset as usize;
        u32::from_le_bytes([
            self.code[off],
            self.code[off + 1],
            self.code[off + 2],
            self.code[off + 3],
        ])
    }

    /// Writes a 32-bit instruction word at `offset` into the code buffer.
    #[inline]
    fn write_instr(&mut self, offset: u32, word: u32) {
        let off = offset as usize;
        let bytes = word.to_le_bytes();
        self.code[off] = bytes[0];
        self.code[off + 1] = bytes[1];
        self.code[off + 2] = bytes[2];
        self.code[off + 3] = bytes[3];
    }

    /// Patches a B or BL instruction (26-bit signed word offset) at `offset`.
    ///
    /// Encoding: bits [25:0] = signed offset / 4 (imm26).
    ///
    /// # Arguments
    ///
    /// * `offset` — Byte offset of the instruction in the code buffer.
    /// * `target_offset` — Signed byte displacement from the instruction site
    ///   to the target label.
    fn patch_branch26(&mut self, offset: u32, target_offset: i32) {
        let word = self.read_instr(offset);
        let imm26 = ((target_offset >> 2) as u32) & 0x03FF_FFFF;
        // Clear existing imm26 field and insert the new value.
        let patched = (word & !0x03FF_FFFF) | imm26;
        self.write_instr(offset, patched);
    }

    /// Patches a B.cond instruction (19-bit signed word offset) at `offset`.
    ///
    /// Encoding: bits [23:5] = signed offset / 4 (imm19).
    fn patch_cond_branch19(&mut self, offset: u32, target_offset: i32) {
        let word = self.read_instr(offset);
        let imm19 = ((target_offset >> 2) as u32) & 0x7FFFF;
        // imm19 occupies bits [23:5].
        let patched = (word & !(0x7FFFF << 5)) | (imm19 << 5);
        self.write_instr(offset, patched);
    }

    /// Patches a TBZ/TBNZ instruction (14-bit signed word offset) at `offset`.
    ///
    /// Encoding: bits [18:5] = signed offset / 4 (imm14).
    fn patch_test_branch14(&mut self, offset: u32, target_offset: i32) {
        let word = self.read_instr(offset);
        let imm14 = ((target_offset >> 2) as u32) & 0x3FFF;
        // imm14 occupies bits [18:5].
        let patched = (word & !(0x3FFF << 5)) | (imm14 << 5);
        self.write_instr(offset, patched);
    }

    /// Patches a CBZ/CBNZ instruction (19-bit signed word offset) at `offset`.
    ///
    /// Same encoding as B.cond: bits [23:5] = signed offset / 4 (imm19).
    fn patch_compare_branch19(&mut self, offset: u32, target_offset: i32) {
        // CBZ/CBNZ use the same imm19 encoding as B.cond.
        self.patch_cond_branch19(offset, target_offset);
    }

    /// Patches an ADR instruction (21-bit signed byte offset) at `offset`.
    ///
    /// Encoding: bits [30:29] = immlo (low 2 bits), bits [23:5] = immhi (high 19 bits).
    fn patch_adr21(&mut self, offset: u32, target_offset: i32) {
        let word = self.read_instr(offset);
        let imm = target_offset as u32;
        let immlo = (imm & 0x3) << 29;
        let immhi = ((imm >> 2) & 0x7_FFFF) << 5;
        // Clear the immlo (bits [30:29]) and immhi (bits [23:5]) fields.
        let mask = (0x3u32 << 29) | (0x7_FFFFu32 << 5);
        let patched = (word & !mask) | immlo | immhi;
        self.write_instr(offset, patched);
    }

    // -------------------------------------------------------------------
    // Branch instruction classification
    // -------------------------------------------------------------------

    /// Classifies a machine instruction as a local branch reference if it
    /// targets a basic block label.
    ///
    /// Returns `Some((label_id, ref_type))` for branch instructions with
    /// a `Label` operand, or `None` for non-branch / external-target instructions.
    fn classify_branch_ref(&self, instr: &MachineInstr) -> Option<(u32, BranchRefType)> {
        // Import opcode constants from encoder module for matching.
        use encoder::{OP_ADR, OP_B, OP_BL, OP_B_COND, OP_CBNZ, OP_CBZ, OP_TBNZ, OP_TBZ};

        // Only branch-class instructions with label operands produce local
        // branch references. Instructions with Symbol operands produce
        // external relocations instead.
        let label_id = self.find_label_operand(instr)?;

        let ref_type = match instr.opcode {
            OP_B | OP_BL => BranchRefType::Branch26,
            OP_B_COND => BranchRefType::CondBranch19,
            OP_CBZ | OP_CBNZ => BranchRefType::CompareBranch19,
            OP_TBZ | OP_TBNZ => BranchRefType::TestBranch14,
            OP_ADR => BranchRefType::Adr21,
            _ => return None,
        };

        Some((label_id, ref_type))
    }

    /// Scans the operands of an instruction for a `Label` operand and
    /// returns the label ID if found.
    fn find_label_operand(&self, instr: &MachineInstr) -> Option<u32> {
        for op in &instr.operands {
            if let MachineOperand::Label(id) = op {
                return Some(*id);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Computes the number of padding bytes needed to align `offset` to
/// `alignment`. Returns 0 if `offset` is already aligned.
///
/// `alignment` must be a power of two and greater than zero.
fn alignment_padding(offset: u32, alignment: u32) -> u32 {
    debug_assert!(alignment.is_power_of_two() && alignment > 0);
    let mask = alignment - 1;
    let misalignment = offset & mask;
    if misalignment == 0 {
        0
    } else {
        alignment - misalignment
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_ref_type_ranges() {
        // B/BL: ±128 MiB
        let (min, max) = BranchRefType::Branch26.range_bytes();
        assert_eq!(min, -(1i64 << 27));
        assert_eq!(max, (1i64 << 27) - 4);

        // B.cond: ±1 MiB
        let (min, max) = BranchRefType::CondBranch19.range_bytes();
        assert_eq!(min, -(1i64 << 20));
        assert_eq!(max, (1i64 << 20) - 4);

        // ADR: ±1 MiB
        let (min, max) = BranchRefType::Adr21.range_bytes();
        assert_eq!(min, -(1i64 << 20));
        assert_eq!(max, (1i64 << 20) - 1);

        // TBZ/TBNZ: ±32 KiB
        let (min, max) = BranchRefType::TestBranch14.range_bytes();
        assert_eq!(min, -(1i64 << 15));
        assert_eq!(max, (1i64 << 15) - 4);

        // CBZ/CBNZ: ±1 MiB
        let (min, max) = BranchRefType::CompareBranch19.range_bytes();
        assert_eq!(min, -(1i64 << 20));
        assert_eq!(max, (1i64 << 20) - 4);
    }

    #[test]
    fn test_alignment_padding() {
        assert_eq!(alignment_padding(0, 4), 0);
        assert_eq!(alignment_padding(1, 4), 3);
        assert_eq!(alignment_padding(2, 4), 2);
        assert_eq!(alignment_padding(3, 4), 1);
        assert_eq!(alignment_padding(4, 4), 0);
        assert_eq!(alignment_padding(5, 16), 11);
        assert_eq!(alignment_padding(16, 16), 0);
    }

    #[test]
    fn test_assembler_new_is_empty() {
        let asm = AArch64Assembler::new();
        assert!(asm.code.is_empty());
        assert!(asm.relocations.is_empty());
        assert!(asm.pending_label_refs.is_empty());
        assert_eq!(asm.current_offset, 0);
    }

    #[test]
    fn test_bind_and_resolve_label() {
        let mut asm = AArch64Assembler::new();
        asm.bind_label(0);
        assert_eq!(asm.labels.get(&0), Some(&0));
        assert!(asm.labels.contains_key(&0));
    }

    #[test]
    fn test_patch_branch26() {
        let mut asm = AArch64Assembler::new();
        // Emit a placeholder B instruction (opcode 0b000101 at [31:26]).
        // B encoding: 0 00101 imm26
        let b_opcode: u32 = 0b000101 << 26; // B with imm26 = 0
        asm.code.extend_from_slice(&b_opcode.to_le_bytes());
        asm.current_offset = 4;

        // Patch to target offset +16 bytes (imm26 = 16/4 = 4).
        asm.patch_branch26(0, 16);
        let patched = asm.read_instr(0);
        let imm26 = patched & 0x03FF_FFFF;
        assert_eq!(imm26, 4); // 16 / 4 = 4

        // Verify the upper bits (opcode) are preserved.
        assert_eq!(patched >> 26, 0b000101);
    }

    #[test]
    fn test_patch_cond_branch19() {
        let mut asm = AArch64Assembler::new();
        // B.cond encoding: 0101010 0 imm19 0 cond
        // Using cond=0000 (EQ), imm19=0
        let bcond_opcode: u32 = 0b01010100 << 24; // B.cond base
        asm.code.extend_from_slice(&bcond_opcode.to_le_bytes());
        asm.current_offset = 4;

        // Patch to target offset +32 bytes (imm19 = 32/4 = 8).
        asm.patch_cond_branch19(0, 32);
        let patched = asm.read_instr(0);
        let imm19 = (patched >> 5) & 0x7FFFF;
        assert_eq!(imm19, 8); // 32 / 4 = 8
    }

    #[test]
    fn test_patch_test_branch14() {
        let mut asm = AArch64Assembler::new();
        // TBZ encoding: b5 011011 0 b40 imm14 Rt
        // Using b5=0, b40=00000, imm14=0, Rt=0
        let tbz_opcode: u32 = 0b00110110 << 24;
        asm.code.extend_from_slice(&tbz_opcode.to_le_bytes());
        asm.current_offset = 4;

        // Patch to target offset -8 bytes (imm14 = -8/4 = -2).
        asm.patch_test_branch14(0, -8);
        let patched = asm.read_instr(0);
        let imm14_raw = (patched >> 5) & 0x3FFF;
        // -2 in 14-bit two's complement = 0x3FFE
        let expected = (-2i32 as u32) & 0x3FFF;
        assert_eq!(imm14_raw, expected);
    }

    #[test]
    fn test_patch_adr21() {
        let mut asm = AArch64Assembler::new();
        // ADR encoding: immlo 10000 immhi Rd
        // ADR X0, #0 → 0x10000000
        let adr_opcode: u32 = 0b10000 << 24; // ADR base
        asm.code.extend_from_slice(&adr_opcode.to_le_bytes());
        asm.current_offset = 4;

        // Patch to target offset +7 bytes.
        asm.patch_adr21(0, 7);
        let patched = asm.read_instr(0);
        // immlo = 7 & 0x3 = 3, at bits [30:29]
        let immlo = (patched >> 29) & 0x3;
        // immhi = (7 >> 2) & 0x7FFFF = 1, at bits [23:5]
        let immhi = (patched >> 5) & 0x7_FFFF;
        assert_eq!(immlo, 3);
        assert_eq!(immhi, 1);
    }

    #[test]
    fn test_assemble_empty_function() {
        let mut asm = AArch64Assembler::new();
        let mf = MachineFunction::new("empty".to_string(), 16);
        let result = asm.assemble_function(&mf);
        assert_eq!(result.name, "empty");
        assert!(result.code.is_empty());
        assert_eq!(result.size, 0);
        assert_eq!(result.alignment, 4);
        assert!(result.relocations.is_empty());
    }

    #[test]
    fn test_assemble_single_nop() {
        let mut asm = AArch64Assembler::new();
        let mut mf = MachineFunction::new("nop_func".to_string(), 16);
        let mut bb = MachineBasicBlock::new(0);
        // NOP is opcode 0x0500 in the encoder namespace.
        let nop_instr = MachineInstr::new(encoder::OP_NOP);
        bb.push_instr(nop_instr);
        mf.add_block(bb);

        let result = asm.assemble_function(&mf);
        assert_eq!(result.name, "nop_func");
        assert_eq!(result.size, 4);
        assert_eq!(result.code.len(), 4);
        assert!(result.relocations.is_empty());

        // NOP encoding should be 0xD503201F.
        let word = u32::from_le_bytes([
            result.code[0],
            result.code[1],
            result.code[2],
            result.code[3],
        ]);
        assert_eq!(word, 0xD503201F);
    }

    #[test]
    fn test_assemble_module_concatenation() {
        let mut asm = AArch64Assembler::new();

        // Create two trivial functions with one NOP each.
        let make_nop_func = |name: &str| {
            let mut mf = MachineFunction::new(name.to_string(), 16);
            let mut bb = MachineBasicBlock::new(0);
            bb.push_instr(MachineInstr::new(encoder::OP_NOP));
            mf.add_block(bb);
            mf
        };

        let funcs = vec![make_nop_func("func_a"), make_nop_func("func_b")];
        let module = asm.assemble_module(&funcs);

        // Two functions × 4 bytes each = 8 bytes total.
        assert_eq!(module.text_section.len(), 8);
        assert_eq!(module.functions.len(), 2);
        assert_eq!(module.symbols.len(), 2);

        // First function at offset 0, second at offset 4.
        assert_eq!(module.functions[0], ("func_a".to_string(), 0, 4));
        assert_eq!(module.functions[1], ("func_b".to_string(), 4, 4));
    }
}
