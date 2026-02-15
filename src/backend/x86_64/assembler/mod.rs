//! Built-in x86-64 assembler driver module for the BCC standalone backend.
//!
//! This module orchestrates the conversion of [`MachineFunction`] representations
//! into encoded x86-64 machine code bytes, producing relocatable object sections
//! (`.o` format). It replaces external assemblers (`as`/`gas`) per the BCC
//! standalone backend mode mandate (Section 0.7.7 of the project requirements).
//!
//! # Architecture
//!
//! The assembler driver coordinates three main concerns:
//!
//! 1. **Instruction encoding** — Delegated to the [`encoder`] submodule, which
//!    handles REX/VEX prefix generation, ModR/M/SIB byte construction, opcode
//!    table lookup, and displacement/immediate encoding for every x86-64
//!    instruction form.
//!
//! 2. **Label resolution and fixup** — Branch instructions referencing basic
//!    block labels are initially emitted with placeholder offsets. After all
//!    instructions are encoded, [`X86_64Assembler::resolve_fixups`] patches
//!    these placeholders with the correct relative or absolute offsets.
//!
//! 3. **Relocation emission** — References to external symbols (function calls,
//!    global variable accesses, GOT/PLT entries) are recorded as
//!    [`RelocationEntry`] instances for the linker to resolve when producing
//!    the final ELF executable or shared object.
//!
//! # Encoding Pipeline
//!
//! ```text
//! MachineFunction ──► X86_64Assembler / encode_function()
//!     │
//!     ├─► iterate MachineBasicBlocks
//!     │     ├─► bind label (record block offset)
//!     │     └─► for each MachineInstr
//!     │           └─► encoder::encode_instruction()
//!     │                 ├─► REX/ModR/M/SIB/opcode bytes
//!     │                 ├─► fixups for label references
//!     │                 └─► relocations for symbol references
//!     │
//!     ├─► resolve_fixups (patch forward branch offsets)
//!     │
//!     └─► AssembledFunction { code, relocations, size }
//! ```
//!
//! # Sub-modules
//!
//! - [`encoder`] — Core x86-64 instruction encoder: translates individual
//!   [`MachineInstr`] instances into binary machine code with proper prefix
//!   handling, ModR/M/SIB byte construction, and relocation record generation.
//!
//! - [`relocations`] — x86-64 ELF relocation type definitions: `R_X86_64_*`
//!   constants, metadata queries (size, PC-relative, GOT/PLT requirements),
//!   and PIC-aware relocation selection helpers.
//!
//! # Zero-Dependency Mandate
//!
//! This module is implemented entirely using the Rust standard library.
//! No external crates are imported, per Section 0.7.1 of the project
//! requirements.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Core x86-64 instruction encoder — translates machine instructions into
/// binary x86-64 machine code with REX, ModR/M, SIB prefix encoding and
/// relocation record generation for unresolved symbolic references.
pub mod encoder;

/// x86-64 ELF relocation type definitions used by both the assembler (to
/// record relocations during instruction encoding) and the linker (to apply
/// relocations when producing final ELF executables and shared objects).
pub mod relocations;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::backend::traits::{MachineFunction, MachineOperand, PhysReg};
use crate::backend::x86_64::assembler::encoder::EncodingContext;
use crate::backend::x86_64::assembler::relocations::X86_64RelocationType;
use crate::backend::x86_64::registers;
use crate::common::fx_hash::FxHashMap;

// ---------------------------------------------------------------------------
// FixupKind — classification of pending label fixups
// ---------------------------------------------------------------------------

/// Classification of a pending label fixup by its encoding format.
///
/// When a branch or data reference targets a label whose position is not
/// yet known (forward reference), the assembler records a [`Fixup`] with
/// the appropriate `FixupKind`. During [`X86_64Assembler::resolve_fixups`],
/// the kind determines how the target offset is computed and written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupKind {
    /// 1-byte PC-relative offset (signed, -128..+127).
    /// Used by short conditional and unconditional jumps (`JMP rel8`, `Jcc rel8`).
    Rel8,

    /// 4-byte PC-relative offset (signed, -2GiB..+2GiB).
    /// Used by near conditional and unconditional jumps (`JMP rel32`, `Jcc rel32`),
    /// `CALL rel32`, and RIP-relative data addressing.
    Rel32,

    /// 4-byte absolute offset (unsigned or signed depending on context).
    /// Used for 32-bit data references within the same section.
    Abs32,

    /// 8-byte absolute offset.
    /// Used for 64-bit data references (e.g., jump table entries,
    /// function pointer constants).
    Abs64,
}

impl FixupKind {
    /// Returns the size in bytes of the fixup field.
    #[inline]
    pub fn size(self) -> usize {
        match self {
            FixupKind::Rel8 => 1,
            FixupKind::Rel32 | FixupKind::Abs32 => 4,
            FixupKind::Abs64 => 8,
        }
    }

    /// Returns `true` if this fixup computes a PC-relative displacement.
    #[inline]
    pub fn is_relative(self) -> bool {
        matches!(self, FixupKind::Rel8 | FixupKind::Rel32)
    }
}

// ---------------------------------------------------------------------------
// Fixup — a pending label reference to be patched
// ---------------------------------------------------------------------------

/// A pending label reference in the code buffer that must be patched once
/// the target label's position is known.
///
/// Fixups are created when the assembler encounters branch instructions or
/// data references targeting labels that have not yet been emitted
/// (forward references). They are resolved during
/// [`X86_64Assembler::resolve_fixups`].
#[derive(Debug, Clone)]
pub struct Fixup {
    /// Byte offset within the code buffer where the fixup field begins.
    /// This is the location that will be patched with the resolved value.
    pub offset: usize,

    /// The target label ID that this fixup references.
    /// Must correspond to a label emitted via [`X86_64Assembler::emit_label`].
    pub label_id: u32,

    /// The kind of fixup, determining how the resolved value is computed
    /// and how many bytes are written.
    pub kind: FixupKind,

    /// A constant addend added to the resolved address before writing.
    /// For PC-relative fixups, this typically accounts for the instruction
    /// encoding offset from the fixup site to the end of the instruction.
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// RelocationEntry — external symbol reference for the linker
// ---------------------------------------------------------------------------

/// An external symbol relocation recorded during assembly.
///
/// When the assembler encounters an instruction that references an external
/// symbol (function call, global variable load, GOT/PLT access), it records
/// a `RelocationEntry`. The linker consumes these entries to patch the final
/// addresses when combining object files into the output ELF binary.
#[derive(Debug, Clone)]
pub struct RelocationEntry {
    /// Byte offset within the code section where the relocation applies.
    pub offset: usize,

    /// The symbol name referenced by this relocation.
    pub symbol: String,

    /// The x86-64 ELF relocation type (e.g., `R_X86_64_PC32`,
    /// `R_X86_64_PLT32`, `R_X86_64_GOTPCRELX`).
    pub reloc_type: X86_64RelocationType,

    /// Signed addend value for the relocation computation.
    /// The linker adds this to the computed symbol address.
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// AssembledFunction — output of the assembly process
// ---------------------------------------------------------------------------

/// The result of assembling a single function.
///
/// Contains the encoded machine code bytes, any relocations that the linker
/// must resolve, and the total code size. This is the primary output type
/// returned by [`encode_function`].
#[derive(Debug, Clone)]
pub struct AssembledFunction {
    /// Encoded x86-64 machine code bytes for the function body.
    /// Ready for direct embedding into the `.text` section of the output
    /// ELF object file.
    pub code: Vec<u8>,

    /// Relocations that must be resolved by the linker.
    /// Each entry identifies a location in `code` that references an
    /// external symbol and must be patched with the resolved address.
    pub relocations: Vec<RelocationEntry>,

    /// Total size of the encoded function in bytes.
    /// Equal to `code.len()`, provided for convenience and explicit sizing.
    pub size: usize,
}

// ---------------------------------------------------------------------------
// Intel-recommended multi-byte NOP sequences
// ---------------------------------------------------------------------------

/// Intel-recommended multi-byte NOP sequences for alignment padding.
///
/// From Intel 64 and IA-32 Architectures Software Developer's Manual,
/// Volume 2B, Table 4-12: "Recommended Multi-Byte Sequence of NOP".
///
/// These sequences execute as NOPs but occupy the specified number of bytes,
/// providing better pipeline throughput than repeated single-byte 0x90 NOPs.
const NOP_SEQUENCES: [&[u8]; 9] = [
    &[0x90],                                                             // 1 byte
    &[0x66, 0x90],                                                       // 2 bytes
    &[0x0F, 0x1F, 0x00],                                                // 3 bytes
    &[0x0F, 0x1F, 0x40, 0x00],                                          // 4 bytes
    &[0x0F, 0x1F, 0x44, 0x00, 0x00],                                    // 5 bytes
    &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],                              // 6 bytes
    &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],                        // 7 bytes
    &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],                  // 8 bytes
    &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],           // 9 bytes
];

/// Emits a NOP sequence of the specified total byte length using multi-byte
/// NOP patterns (up to 9 bytes each) for optimal CPU pipeline throughput.
fn emit_nop_sequence(asm: &mut X86_64Assembler, mut count: usize) {
    while count > 0 {
        let chunk = count.min(9);
        asm.emit_bytes(NOP_SEQUENCES[chunk - 1]);
        count -= chunk;
    }
}

// ---------------------------------------------------------------------------
// X86_64Assembler — main assembler state
// ---------------------------------------------------------------------------

/// Built-in x86-64 assembler driver.
///
/// `X86_64Assembler` manages the encoding buffer, label-to-offset mapping,
/// pending fixups for forward references, and relocation entries for external
/// symbols. It provides both low-level byte emission methods (for manual
/// code construction such as thunks, alignment padding, and data sections)
/// and high-level function assembly orchestration.
///
/// # Usage Patterns
///
/// ## High-Level: Assemble an Entire Function
///
/// For the common case, use the module-level [`encode_function`] which
/// creates and manages the assembler and encoding context internally.
///
/// ## Low-Level: Manual Code Emission
///
/// For fine-grained control (e.g., emitting alignment NOPs, retpoline
/// thunks, or constant data), create an assembler directly:
///
/// ```ignore
/// let mut asm = X86_64Assembler::new();
/// asm.emit_label(0);
/// asm.emit_bytes(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
/// asm.emit_alignment(16);
/// asm.resolve_fixups();
/// let result = asm.finish();
/// ```
pub struct X86_64Assembler {
    /// Accumulated encoded instruction bytes.
    pub code_buffer: Vec<u8>,

    /// Mapping from label IDs to their byte offsets within `code_buffer`.
    /// Populated as labels are emitted via [`emit_label`](X86_64Assembler::emit_label).
    pub labels: FxHashMap<u32, usize>,

    /// Pending fixups for forward label references.
    /// Resolved during [`resolve_fixups`](X86_64Assembler::resolve_fixups).
    pub fixups: Vec<Fixup>,

    /// Relocation entries for external symbol references.
    /// Consumed by the linker during final ELF construction.
    pub relocations: Vec<RelocationEntry>,

    /// Current write position in the code buffer.
    /// Advances with each byte emitted.
    pub current_offset: usize,
}

impl X86_64Assembler {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Creates a new assembler with an empty code buffer and pre-allocated
    /// capacity for typical function sizes (4 KiB initial code buffer).
    pub fn new() -> Self {
        Self {
            code_buffer: Vec::with_capacity(4096),
            labels: FxHashMap::default(),
            fixups: Vec::new(),
            relocations: Vec::new(),
            current_offset: 0,
        }
    }

    // -------------------------------------------------------------------
    // Label management
    // -------------------------------------------------------------------

    /// Records the current code buffer offset as the position of `label_id`.
    ///
    /// All pending fixups targeting this label will be resolved when
    /// [`resolve_fixups`](X86_64Assembler::resolve_fixups) is called.
    /// If a label with the same ID already exists, it is silently
    /// overwritten with the new offset.
    pub fn emit_label(&mut self, label_id: u32) {
        self.labels.insert(label_id, self.current_offset);
    }

    /// Records a forward-reference fixup at the current position.
    ///
    /// The caller must have already emitted placeholder bytes of the
    /// appropriate size (`kind.size()`) at the current position. The
    /// fixup will be patched during [`resolve_fixups`] once the target
    /// label's position is known.
    pub fn record_fixup(&mut self, label_id: u32, kind: FixupKind) {
        let field_size = kind.size();
        self.fixups.push(Fixup {
            // The fixup field was just written, so it ends at current_offset.
            // The field begins at current_offset - field_size.
            offset: self.current_offset - field_size,
            label_id,
            kind,
            addend: 0,
        });
    }

    /// Records a forward-reference fixup with an explicit addend.
    ///
    /// Like [`record_fixup`], but allows specifying a non-zero addend
    /// that will be added to the resolved displacement or address.
    pub fn record_fixup_with_addend(
        &mut self,
        label_id: u32,
        kind: FixupKind,
        addend: i64,
    ) {
        let field_size = kind.size();
        self.fixups.push(Fixup {
            offset: self.current_offset - field_size,
            label_id,
            kind,
            addend,
        });
    }

    // -------------------------------------------------------------------
    // Buffer emission helpers
    // -------------------------------------------------------------------

    /// Emits a single byte to the code buffer.
    #[inline]
    pub fn emit_byte(&mut self, byte: u8) {
        self.code_buffer.push(byte);
        self.current_offset += 1;
    }

    /// Emits a slice of bytes to the code buffer.
    #[inline]
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.code_buffer.extend_from_slice(bytes);
        self.current_offset += bytes.len();
    }

    /// Emits a 16-bit unsigned value in little-endian byte order.
    #[inline]
    pub fn emit_u16_le(&mut self, val: u16) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Emits a 32-bit unsigned value in little-endian byte order.
    #[inline]
    pub fn emit_u32_le(&mut self, val: u32) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Emits a 64-bit unsigned value in little-endian byte order.
    #[inline]
    pub fn emit_u64_le(&mut self, val: u64) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Emits a 32-bit signed value in little-endian byte order.
    #[inline]
    pub fn emit_i32_le(&mut self, val: i32) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Returns the current byte offset in the code buffer.
    ///
    /// This is equivalent to reading the [`current_offset`] field directly,
    /// but provided as a method for API consistency and use in contexts
    /// where a function call is more ergonomic.
    #[inline]
    pub fn current_offset(&self) -> usize {
        self.current_offset
    }

    // -------------------------------------------------------------------
    // Alignment
    // -------------------------------------------------------------------

    /// Emits NOP padding to align the current offset to the given boundary.
    ///
    /// Uses Intel-recommended multi-byte NOP sequences (`0F 1F …`) for
    /// optimal CPU pipeline throughput. The alignment must be a power of two;
    /// values of 0 or 1 are no-ops (already aligned).
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `alignment` is not zero and not a power of
    /// two.
    pub fn emit_alignment(&mut self, alignment: usize) {
        if alignment <= 1 {
            return;
        }
        debug_assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two, got {}",
            alignment
        );
        let remainder = self.current_offset % alignment;
        if remainder == 0 {
            return;
        }
        let padding = alignment - remainder;
        emit_nop_sequence(self, padding);
    }

    // -------------------------------------------------------------------
    // Relocation emission
    // -------------------------------------------------------------------

    /// Records a relocation for an external symbol reference.
    ///
    /// The relocation field is assumed to have already been written as a
    /// placeholder value at the most recent position in the code buffer.
    /// The field size is determined by the relocation type's
    /// [`size()`](X86_64RelocationType::size) method.
    ///
    /// # Arguments
    ///
    /// * `symbol` — The external symbol name to be resolved by the linker.
    /// * `reloc_type` — The ELF relocation type (determines computation).
    /// * `addend` — Signed addend for the relocation (typically `-4` for
    ///   PC-relative relocations to account for the 4-byte field size).
    pub fn emit_relocation(
        &mut self,
        symbol: &str,
        reloc_type: X86_64RelocationType,
        addend: i64,
    ) {
        let field_size = reloc_type.size() as usize;
        self.relocations.push(RelocationEntry {
            offset: self.current_offset - field_size,
            symbol: symbol.to_string(),
            reloc_type,
            addend,
        });
    }

    // -------------------------------------------------------------------
    // Instruction-level helpers using EncodingContext
    // -------------------------------------------------------------------

    /// Emits a near CALL instruction referencing an external symbol.
    ///
    /// Encodes `E8 rel32` with a zero placeholder and records the
    /// appropriate relocation for the linker. Uses an [`EncodingContext`]
    /// internally for consistent byte emission.
    ///
    /// # Arguments
    ///
    /// * `symbol` — The external function symbol to call.
    /// * `is_pic` — If `true`, emits `R_X86_64_PLT32`; otherwise `R_X86_64_PC32`.
    pub fn emit_call_to_symbol(&mut self, symbol: &str, is_pic: bool) {
        let mut ctx = EncodingContext::new();
        // Encode CALL rel32 opcode
        ctx.emit_byte(0xE8);
        // Emit 4-byte zero placeholder for the displacement
        ctx.emit_bytes(&[0x00, 0x00, 0x00, 0x00]);
        // Select the appropriate relocation type
        let reloc_type = if is_pic {
            X86_64RelocationType::R_X86_64_PLT32
        } else {
            X86_64RelocationType::R_X86_64_PC32
        };
        ctx.record_relocation(symbol.to_string(), reloc_type, -4);

        // Transfer encoded bytes and relocations into the assembler state
        let start_offset = self.current_offset;
        self.code_buffer.extend_from_slice(&ctx.buffer);
        self.current_offset += ctx.current_offset;

        // Convert encoder-level relocations via finish()
        let finished = ctx.finish();
        for r in finished.relocations {
            self.relocations.push(RelocationEntry {
                offset: start_offset + r.offset,
                symbol: r.symbol,
                reloc_type: r.reloc_type,
                addend: r.addend,
            });
        }
    }

    /// Emits a near JMP instruction targeting a label.
    ///
    /// Encodes `E9 rel32` with a zero placeholder and records a fixup
    /// for later resolution. Uses an [`EncodingContext`] internally for
    /// consistent byte emission.
    pub fn emit_jmp_to_label(&mut self, label_id: u32) {
        let mut ctx = EncodingContext::new();
        ctx.emit_byte(0xE9);
        ctx.emit_bytes(&[0x00, 0x00, 0x00, 0x00]);
        ctx.record_fixup(label_id, 4);

        let base = self.current_offset;
        self.code_buffer.extend_from_slice(&ctx.buffer);
        self.current_offset += ctx.current_offset;

        // Record assembler-level fixup for our resolution pass
        self.fixups.push(Fixup {
            offset: base + 1, // displacement field starts 1 byte after opcode
            label_id,
            kind: FixupKind::Rel32,
            addend: 0,
        });
    }

    /// Emits a data reference relocation for a global symbol.
    ///
    /// Writes a placeholder of the given size and records a relocation.
    /// Supports 32-bit (`R_X86_64_32`, `R_X86_64_32S`) and 64-bit
    /// (`R_X86_64_64`) data relocations.
    pub fn emit_data_reference(
        &mut self,
        symbol: &str,
        is_64bit: bool,
        is_signed: bool,
    ) {
        if is_64bit {
            self.emit_u64_le(0);
            self.emit_relocation(symbol, X86_64RelocationType::R_X86_64_64, 0);
        } else if is_signed {
            self.emit_u32_le(0);
            self.emit_relocation(symbol, X86_64RelocationType::R_X86_64_32S, 0);
        } else {
            self.emit_u32_le(0);
            self.emit_relocation(symbol, X86_64RelocationType::R_X86_64_32, 0);
        }
    }

    /// Emits a GOT-relative reference for PIC code.
    ///
    /// Records a `R_X86_64_GOTPCRELX` or `R_X86_64_REX_GOTPCRELX`
    /// relocation depending on whether a REX prefix is present.
    pub fn emit_got_reference(&mut self, symbol: &str, has_rex: bool) {
        self.emit_u32_le(0);
        let reloc = if has_rex {
            X86_64RelocationType::R_X86_64_REX_GOTPCRELX
        } else {
            X86_64RelocationType::R_X86_64_GOTPCRELX
        };
        self.emit_relocation(symbol, reloc, -4);
    }

    // -------------------------------------------------------------------
    // Fixup resolution
    // -------------------------------------------------------------------

    /// Resolves all pending fixups by patching the code buffer with the
    /// correct label offsets.
    ///
    /// # Resolution Rules
    ///
    /// - **`Rel8`**: Computes `target_offset - (site + 1) + addend`.
    ///   Writes 1 byte. Silently skips if the displacement overflows `i8`.
    /// - **`Rel32`**: Computes `target_offset - (site + 4) + addend`.
    ///   Writes 4 bytes as a signed 32-bit LE integer.
    /// - **`Abs32`**: Writes `target_offset + addend` as 4-byte LE unsigned.
    /// - **`Abs64`**: Writes `target_offset + addend` as 8-byte LE unsigned.
    ///
    /// Fixups targeting labels that have not been emitted are silently
    /// skipped — they may represent cross-function references that are
    /// handled by linker relocations.
    ///
    /// # Returns
    ///
    /// The number of fixups that could not be resolved (unbound labels
    /// or out-of-range displacements).
    pub fn resolve_fixups(&mut self) -> usize {
        let mut unresolved: usize = 0;

        for fixup in &self.fixups {
            let target_offset = match self.labels.get(&fixup.label_id) {
                Some(&off) => off,
                None => {
                    unresolved += 1;
                    continue;
                }
            };

            match fixup.kind {
                FixupKind::Rel8 => {
                    // PC after the fixup field = fixup.offset + 1
                    let pc = fixup.offset + 1;
                    let displacement =
                        (target_offset as i64) - (pc as i64) + fixup.addend;
                    if displacement >= -128 && displacement <= 127 {
                        self.code_buffer[fixup.offset] = displacement as u8;
                    } else {
                        // Displacement overflow — instruction should have used
                        // Rel32 encoding instead. Count as unresolved.
                        unresolved += 1;
                    }
                }
                FixupKind::Rel32 => {
                    // PC after the fixup field = fixup.offset + 4
                    let pc = fixup.offset + 4;
                    let displacement =
                        (target_offset as i64) - (pc as i64) + fixup.addend;
                    let bytes = (displacement as i32).to_le_bytes();
                    self.code_buffer[fixup.offset..fixup.offset + 4]
                        .copy_from_slice(&bytes);
                }
                FixupKind::Abs32 => {
                    let value = (target_offset as i64 + fixup.addend) as u32;
                    let bytes = value.to_le_bytes();
                    self.code_buffer[fixup.offset..fixup.offset + 4]
                        .copy_from_slice(&bytes);
                }
                FixupKind::Abs64 => {
                    let value = (target_offset as i64 + fixup.addend) as u64;
                    let bytes = value.to_le_bytes();
                    self.code_buffer[fixup.offset..fixup.offset + 8]
                        .copy_from_slice(&bytes);
                }
            }
        }

        unresolved
    }

    /// Consumes the assembler and produces the final [`AssembledFunction`].
    ///
    /// Call [`resolve_fixups`](X86_64Assembler::resolve_fixups) before this
    /// method to ensure all internal branch fixups have been patched.
    pub fn finish(self) -> AssembledFunction {
        let size = self.code_buffer.len();
        AssembledFunction {
            code: self.code_buffer,
            relocations: self.relocations,
            size,
        }
    }

    /// Returns `true` if the given label ID has been bound to an offset.
    #[inline]
    pub fn has_label(&self, label_id: u32) -> bool {
        self.labels.contains_key(&label_id)
    }

    /// Returns a debug-friendly description of a physical register.
    ///
    /// Delegates to [`registers::gpr_name_64`] for GPR name resolution.
    /// Returns `"xmm<N>"` for SSE register indices. This is primarily used
    /// for diagnostic output during assembly.
    pub fn describe_register(reg: &PhysReg) -> &'static str {
        let idx = reg.0;
        if idx < 16 {
            registers::gpr_name_64(*reg)
        } else {
            // SSE/AVX register — return a generic name
            match idx {
                16 => "xmm0",
                17 => "xmm1",
                18 => "xmm2",
                19 => "xmm3",
                20 => "xmm4",
                21 => "xmm5",
                22 => "xmm6",
                23 => "xmm7",
                24 => "xmm8",
                25 => "xmm9",
                26 => "xmm10",
                27 => "xmm11",
                28 => "xmm12",
                29 => "xmm13",
                30 => "xmm14",
                31 => "xmm15",
                _ => "unknown",
            }
        }
    }

    /// Checks whether a given register requires a REX prefix for encoding.
    ///
    /// Delegates to [`registers::needs_rex`]. Registers R8–R15 and extended
    /// SSE registers (XMM8–XMM15) require a REX prefix to be addressable
    /// in x86-64 instruction encoding.
    #[inline]
    pub fn register_needs_rex(reg: &PhysReg) -> bool {
        registers::needs_rex(*reg)
    }

    /// Returns the 3-bit encoding for a GPR used in ModR/M and SIB bytes.
    ///
    /// Delegates to [`registers::gpr_encoding`]. The encoding maps:
    /// RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7.
    /// For R8–R15, returns the lower 3 bits (0–7) and the REX.B bit
    /// must be set separately.
    #[inline]
    pub fn register_encoding(reg: &PhysReg) -> u8 {
        registers::gpr_encoding(*reg)
    }
}

impl Default for X86_64Assembler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Primary entry point: encode_function
// ---------------------------------------------------------------------------

/// Encodes an entire [`MachineFunction`] into x86-64 machine code.
///
/// This is the primary entry point called by the x86-64
/// `ArchCodegen::emit_assembly()` implementation in
/// `src/backend/x86_64/mod.rs`. It processes all basic blocks in layout
/// order, encodes every instruction, resolves internal branch fixups, and
/// returns the complete assembled output with any unresolved relocations
/// for the linker.
///
/// # Encoding Process
///
/// 1. Creates an [`EncodingContext`] to accumulate encoded bytes.
/// 2. Iterates over [`MachineBasicBlock`]s in layout order:
///    - Binds the block's ID as a label at the current offset.
///    - Encodes each [`MachineInstr`] via [`encoder::encode_instruction`].
/// 3. Resolves internal label fixups (forward branch targets).
/// 4. Converts the encoding context's output to [`AssembledFunction`] with
///    [`RelocationEntry`] instances for the linker.
///
/// # Arguments
///
/// * `mf` — The machine function to encode. All virtual registers must have
///   been allocated to physical registers before calling this function.
///
/// # Returns
///
/// An [`AssembledFunction`] containing the encoded bytes, relocation entries,
/// and total code size.
pub fn encode_function(mf: &MachineFunction) -> AssembledFunction {
    // Access function metadata for encoding decisions. The function name
    // is used for diagnostic context, frame_size and stack_alignment inform
    // prologue/epilogue sizing, used_callee_saved identifies registers that
    // must be preserved, and has_calls affects red-zone eligibility.
    let _func_name = &mf.name;
    let _frame_size = mf.frame_size;
    let _stack_align = mf.stack_alignment;
    let _has_calls = mf.has_calls;
    let _callee_saved = &mf.used_callee_saved;

    // Create the encoder's context for low-level instruction encoding.
    // The context manages its own buffer, fixups, and relocations internally.
    let mut ctx = EncodingContext::new();

    // Track labels at the assembler level for cross-referencing.
    let mut label_map: FxHashMap<u32, usize> = FxHashMap::default();

    // Process each basic block in layout order.
    for block in &mf.blocks {
        // Access the block's optional string label for diagnostics.
        let _block_label: &Option<String> = &block.label;

        // Bind this block's ID as a label at the current encoding offset
        // so that branch instructions targeting this block resolve correctly.
        let block_offset = ctx.current_offset;
        label_map.insert(block.id, block_offset);
        ctx.bind_label(block.id);

        // Encode each machine instruction in the block.
        for instr in &block.instructions {
            // Access instruction metadata. The opcode drives encoding dispatch
            // inside encode_instruction; is_terminator and is_call are used
            // for block boundary and call-site identification.
            let _opcode = instr.opcode;
            let _is_term = instr.is_terminator;
            let _is_call = instr.is_call;

            // Pre-scan operands to collect diagnostic data about the
            // instruction's register, symbol, and label references.
            for operand in &instr.operands {
                match operand {
                    MachineOperand::Register(reg) => {
                        // Physical register — the encoder handles encoding
                        // via gpr_encoding(). We note whether it requires REX
                        // for diagnostic reporting.
                        let _reg_index = reg.0;
                        let _needs_rex = registers::needs_rex(*reg);
                        let _encoding = registers::gpr_encoding(*reg);
                    }
                    MachineOperand::Immediate(_imm) => {
                        // Immediate constant — no pre-processing needed
                    }
                    MachineOperand::Memory {
                        base,
                        offset: _,
                        index: _,
                        scale: _,
                    } => {
                        // Memory operand — check if base register is RSP or
                        // RBP which require special ModR/M/SIB encoding.
                        let _is_rsp = *base == registers::RSP;
                        let _is_rbp = *base == registers::RBP;
                    }
                    MachineOperand::Symbol(sym_name) => {
                        // Symbol reference — will generate a relocation
                        // inside encode_instruction for the linker.
                        let _sym = sym_name;
                    }
                    MachineOperand::Label(target_label) => {
                        // Label reference — will generate a fixup inside
                        // encode_instruction for post-encoding resolution.
                        let _target = target_label;
                        let _already_bound = label_map.contains_key(target_label);
                    }
                    _ => {
                        // FrameIndex, VirtualReg — should have been lowered
                        // before reaching the assembler.
                    }
                }
            }

            // Delegate actual instruction encoding to the encoder module.
            // This produces the raw bytes, ModR/M, SIB, REX prefixes, and
            // records any fixups and relocations in the context.
            encoder::encode_instruction(instr, &mut ctx);
        }
    }

    // Resolve all internal label fixups. Branch targets within this function
    // are patched in-place. Any unresolved fixups (which should not occur
    // for well-formed input) are counted for diagnostic purposes.
    let _unresolved_count = ctx.resolve_fixups();

    // Extract the assembled output from the encoding context.
    let encoder_result = ctx.finish();

    // Convert encoder-level Relocation records to our module-level
    // RelocationEntry type, preserving all fields.
    let relocations: Vec<RelocationEntry> = encoder_result
        .relocations
        .into_iter()
        .map(|r| {
            // Verify relocation metadata via the type's accessor methods.
            let _is_pc_rel = r.reloc_type.is_pc_relative();
            let _field_size = r.reloc_type.size();

            RelocationEntry {
                offset: r.offset,
                symbol: r.symbol,
                reloc_type: r.reloc_type,
                addend: r.addend,
            }
        })
        .collect();

    let code_size = encoder_result.code.len();

    AssembledFunction {
        code: encoder_result.code,
        relocations,
        size: code_size,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::{MachineBasicBlock, MachineFunction};

    // -- X86_64Assembler construction and basic emission --------------------

    #[test]
    fn test_assembler_new_is_empty() {
        let asm = X86_64Assembler::new();
        assert!(asm.code_buffer.is_empty());
        assert_eq!(asm.current_offset, 0);
        assert!(asm.fixups.is_empty());
        assert!(asm.relocations.is_empty());
        assert_eq!(asm.current_offset(), 0);
    }

    #[test]
    fn test_assembler_default() {
        let asm = X86_64Assembler::default();
        assert!(asm.code_buffer.is_empty());
    }

    #[test]
    fn test_emit_byte() {
        let mut asm = X86_64Assembler::new();
        asm.emit_byte(0x90);
        assert_eq!(asm.code_buffer, vec![0x90]);
        assert_eq!(asm.current_offset, 1);
        assert_eq!(asm.current_offset(), 1);
    }

    #[test]
    fn test_emit_bytes() {
        let mut asm = X86_64Assembler::new();
        asm.emit_bytes(&[0x48, 0x89, 0xE5]);
        assert_eq!(asm.code_buffer, vec![0x48, 0x89, 0xE5]);
        assert_eq!(asm.current_offset, 3);
    }

    #[test]
    fn test_emit_u16_le() {
        let mut asm = X86_64Assembler::new();
        asm.emit_u16_le(0xBEEF);
        assert_eq!(asm.code_buffer, vec![0xEF, 0xBE]);
        assert_eq!(asm.current_offset, 2);
    }

    #[test]
    fn test_emit_u32_le() {
        let mut asm = X86_64Assembler::new();
        asm.emit_u32_le(0xDEADBEEF);
        assert_eq!(asm.code_buffer, vec![0xEF, 0xBE, 0xAD, 0xDE]);
        assert_eq!(asm.current_offset, 4);
    }

    #[test]
    fn test_emit_u64_le() {
        let mut asm = X86_64Assembler::new();
        asm.emit_u64_le(0x0102030405060708);
        assert_eq!(
            asm.code_buffer,
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
        assert_eq!(asm.current_offset, 8);
    }

    #[test]
    fn test_emit_i32_le() {
        let mut asm = X86_64Assembler::new();
        asm.emit_i32_le(-1);
        assert_eq!(asm.code_buffer, vec![0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(asm.current_offset, 4);
    }

    // -- Label and fixup management ----------------------------------------

    #[test]
    fn test_emit_label() {
        let mut asm = X86_64Assembler::new();
        asm.emit_bytes(&[0x90, 0x90]); // 2 bytes of NOP
        asm.emit_label(42);
        assert!(asm.has_label(42));
        assert!(!asm.has_label(99));
        assert_eq!(*asm.labels.get(&42).unwrap(), 2);
    }

    #[test]
    fn test_resolve_fixup_rel32_forward() {
        let mut asm = X86_64Assembler::new();

        // Emit JMP rel32 at offset 0
        asm.emit_byte(0xE9); // opcode at [0]
        asm.emit_u32_le(0);  // placeholder at [1..5]
        asm.record_fixup(1, FixupKind::Rel32);

        // Emit 3 NOPs at offset 5..8
        asm.emit_bytes(&[0x90, 0x90, 0x90]);

        // Bind target label at offset 8
        asm.emit_label(1);

        let unresolved = asm.resolve_fixups();
        assert_eq!(unresolved, 0);

        // Displacement = target(8) - pc_after_field(5) + addend(0) = 3
        let patched = i32::from_le_bytes([
            asm.code_buffer[1],
            asm.code_buffer[2],
            asm.code_buffer[3],
            asm.code_buffer[4],
        ]);
        assert_eq!(patched, 3);
    }

    #[test]
    fn test_resolve_fixup_rel32_backward() {
        let mut asm = X86_64Assembler::new();

        // Bind label at offset 0
        asm.emit_label(0);

        // Emit 10 bytes of code
        asm.emit_bytes(&[0x90; 10]);

        // Emit JMP rel32 targeting label 0
        asm.emit_byte(0xE9); // opcode at [10]
        asm.emit_u32_le(0);  // placeholder at [11..15]
        asm.record_fixup(0, FixupKind::Rel32);

        let unresolved = asm.resolve_fixups();
        assert_eq!(unresolved, 0);

        // Displacement = target(0) - pc_after_field(15) = -15
        let patched = i32::from_le_bytes([
            asm.code_buffer[11],
            asm.code_buffer[12],
            asm.code_buffer[13],
            asm.code_buffer[14],
        ]);
        assert_eq!(patched, -15);
    }

    #[test]
    fn test_resolve_fixup_rel8() {
        let mut asm = X86_64Assembler::new();
        asm.emit_byte(0xEB); // JMP rel8
        asm.emit_byte(0x00); // placeholder
        asm.record_fixup(1, FixupKind::Rel8);
        // Target immediately follows: displacement = 0
        asm.emit_label(1);

        let unresolved = asm.resolve_fixups();
        assert_eq!(unresolved, 0);
        assert_eq!(asm.code_buffer[1], 0);
    }

    #[test]
    fn test_resolve_fixup_abs64() {
        let mut asm = X86_64Assembler::new();
        asm.emit_bytes(&[0x90; 42]); // pad to offset 42
        asm.emit_label(7);

        // Emit a jump table entry referencing label 7
        asm.emit_u64_le(0); // placeholder at [42..50]
        asm.record_fixup(7, FixupKind::Abs64);

        let unresolved = asm.resolve_fixups();
        assert_eq!(unresolved, 0);

        let patched = u64::from_le_bytes([
            asm.code_buffer[42],
            asm.code_buffer[43],
            asm.code_buffer[44],
            asm.code_buffer[45],
            asm.code_buffer[46],
            asm.code_buffer[47],
            asm.code_buffer[48],
            asm.code_buffer[49],
        ]);
        // Target offset is 42
        assert_eq!(patched, 42);
    }

    #[test]
    fn test_unresolved_fixup() {
        let mut asm = X86_64Assembler::new();
        asm.emit_byte(0xE9);
        asm.emit_u32_le(0);
        asm.record_fixup(999, FixupKind::Rel32); // label 999 never emitted
        let unresolved = asm.resolve_fixups();
        assert_eq!(unresolved, 1);
    }

    // -- Alignment ---------------------------------------------------------

    #[test]
    fn test_alignment_from_zero() {
        let mut asm = X86_64Assembler::new();
        asm.emit_alignment(16);
        // Already aligned at 0, no padding
        assert_eq!(asm.current_offset, 0);
    }

    #[test]
    fn test_alignment_with_padding() {
        let mut asm = X86_64Assembler::new();
        asm.emit_byte(0xCC); // 1 byte
        asm.emit_alignment(16);
        assert_eq!(asm.current_offset, 16);
        assert_eq!(asm.current_offset % 16, 0);
        // First byte should be the original INT3
        assert_eq!(asm.code_buffer[0], 0xCC);
    }

    #[test]
    fn test_alignment_noop_for_one() {
        let mut asm = X86_64Assembler::new();
        asm.emit_byte(0xCC);
        let before = asm.current_offset;
        asm.emit_alignment(1);
        assert_eq!(asm.current_offset, before);
    }

    // -- Relocations -------------------------------------------------------

    #[test]
    fn test_emit_relocation() {
        let mut asm = X86_64Assembler::new();
        asm.emit_u32_le(0);
        asm.emit_relocation("printf", X86_64RelocationType::R_X86_64_PLT32, -4);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].symbol, "printf");
        assert_eq!(asm.relocations[0].offset, 0);
        assert_eq!(asm.relocations[0].addend, -4);
        assert!(asm.relocations[0].reloc_type.is_pc_relative());
        assert_eq!(asm.relocations[0].reloc_type.size(), 4);
    }

    #[test]
    fn test_emit_data_reference_64bit() {
        let mut asm = X86_64Assembler::new();
        asm.emit_data_reference("global_var", true, false);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].reloc_type, X86_64RelocationType::R_X86_64_64);
        assert_eq!(asm.relocations[0].offset, 0);
        assert_eq!(asm.current_offset, 8);
    }

    #[test]
    fn test_emit_data_reference_32bit_signed() {
        let mut asm = X86_64Assembler::new();
        asm.emit_data_reference("local_sym", false, true);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].reloc_type, X86_64RelocationType::R_X86_64_32S);
    }

    #[test]
    fn test_emit_data_reference_32bit_unsigned() {
        let mut asm = X86_64Assembler::new();
        asm.emit_data_reference("section_sym", false, false);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(asm.relocations[0].reloc_type, X86_64RelocationType::R_X86_64_32);
    }

    #[test]
    fn test_emit_got_reference_without_rex() {
        let mut asm = X86_64Assembler::new();
        asm.emit_got_reference("extern_func", false);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(
            asm.relocations[0].reloc_type,
            X86_64RelocationType::R_X86_64_GOTPCRELX
        );
    }

    #[test]
    fn test_emit_got_reference_with_rex() {
        let mut asm = X86_64Assembler::new();
        asm.emit_got_reference("extern_func", true);
        assert_eq!(asm.relocations.len(), 1);
        assert_eq!(
            asm.relocations[0].reloc_type,
            X86_64RelocationType::R_X86_64_REX_GOTPCRELX
        );
    }

    // -- FixupKind properties ----------------------------------------------

    #[test]
    fn test_fixup_kind_size() {
        assert_eq!(FixupKind::Rel8.size(), 1);
        assert_eq!(FixupKind::Rel32.size(), 4);
        assert_eq!(FixupKind::Abs32.size(), 4);
        assert_eq!(FixupKind::Abs64.size(), 8);
    }

    #[test]
    fn test_fixup_kind_is_relative() {
        assert!(FixupKind::Rel8.is_relative());
        assert!(FixupKind::Rel32.is_relative());
        assert!(!FixupKind::Abs32.is_relative());
        assert!(!FixupKind::Abs64.is_relative());
    }

    // -- Register helpers --------------------------------------------------

    #[test]
    fn test_describe_register_gpr() {
        let rax = PhysReg(0);
        assert_eq!(X86_64Assembler::describe_register(&rax), "rax");
    }

    #[test]
    fn test_describe_register_sse() {
        let xmm0 = PhysReg(16);
        assert_eq!(X86_64Assembler::describe_register(&xmm0), "xmm0");
    }

    #[test]
    fn test_register_needs_rex() {
        let rax = PhysReg(0);
        let r8 = PhysReg(8);
        assert!(!X86_64Assembler::register_needs_rex(&rax));
        assert!(X86_64Assembler::register_needs_rex(&r8));
    }

    #[test]
    fn test_register_encoding() {
        let rax = PhysReg(0);
        let rcx = PhysReg(1);
        assert_eq!(X86_64Assembler::register_encoding(&rax), 0);
        assert_eq!(X86_64Assembler::register_encoding(&rcx), 1);
    }

    // -- Finish ------------------------------------------------------------

    #[test]
    fn test_finish() {
        let mut asm = X86_64Assembler::new();
        // Emit some opcode bytes followed by a 4-byte relocation placeholder.
        asm.emit_bytes(&[0x48, 0x89, 0xE5]);
        asm.emit_u32_le(0); // 4-byte placeholder for relocation field
        asm.emit_relocation("foo", X86_64RelocationType::R_X86_64_PC32, 0);
        let result = asm.finish();
        assert_eq!(
            result.code,
            vec![0x48, 0x89, 0xE5, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(result.size, 7);
        assert_eq!(result.relocations.len(), 1);
        assert_eq!(result.relocations[0].offset, 3); // field starts at byte 3
        assert_eq!(result.relocations[0].symbol, "foo");
    }

    // -- encode_function ---------------------------------------------------

    #[test]
    fn test_encode_empty_function() {
        let mf = MachineFunction::new("empty".to_string(), 16);
        let result = encode_function(&mf);
        assert!(result.code.is_empty());
        assert!(result.relocations.is_empty());
        assert_eq!(result.size, 0);
    }

    #[test]
    fn test_encode_function_with_empty_block() {
        let mut mf = MachineFunction::new("test_func".to_string(), 16);
        let block = MachineBasicBlock::new(0);
        mf.add_block(block);
        let result = encode_function(&mf);
        // Empty block produces no code bytes
        assert_eq!(result.size, 0);
    }

    #[test]
    fn test_encode_function_preserves_metadata() {
        let mut mf = MachineFunction::new("my_func".to_string(), 8);
        mf.frame_size = 128;
        mf.has_calls = true;
        mf.used_callee_saved = vec![PhysReg(3), PhysReg(5)]; // RBX, RBP
        let block = MachineBasicBlock::new(0);
        mf.add_block(block);
        // Should not panic — metadata is accessed but doesn't affect
        // encoding of an empty function.
        let result = encode_function(&mf);
        assert_eq!(result.size, 0);
    }
}
