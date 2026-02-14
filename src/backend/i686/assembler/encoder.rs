//! i686 instruction encoder — 32-bit x86 machine code encoding without REX prefixes.
//!
//! This module implements the core instruction encoding logic for the i686
//! (IA-32) target architecture. It translates [`MachineInstr`] structures
//! (produced by the instruction selector and register allocator) into raw
//! binary machine code bytes and relocation entries.
//!
//! # Key Differences from x86-64 Encoding
//!
//! - **No REX prefix:** Only 8 GPRs (EAX–EDI) are directly encodable with
//!   3-bit register fields. There is no REX.B/REX.R extension bit.
//! - **32-bit operands by default:** The default operand size is 32 bits;
//!   16-bit operands require an operand-size prefix (0x66).
//! - **32-bit addressing:** All addresses and displacements are 32-bit.
//!   No RIP-relative addressing — PIC uses GOT-relative via EBX.
//!
//! # Encoding Overview
//!
//! Each instruction is encoded as:
//!
//! ```text
//! [prefixes] [opcode (1-3 bytes)] [ModR/M] [SIB] [displacement] [immediate]
//! ```
//!
//! The encoder handles ModR/M byte construction, SIB byte generation for
//! complex addressing modes, displacement field sizing (imm8 vs imm32),
//! and relocation emission for unresolved symbol references.
//!
//! # Standalone Backend Mandate
//!
//! This encoder is part of the standalone backend — no external `as` or
//! `llvm-mc` is invoked. All encoding is performed in-process.

use crate::backend::i686::registers;
use crate::backend::traits::{MachineInstr, MachineOperand, PhysReg};

use super::relocations::I686RelocType;
use super::RelocationEntry;

// ---------------------------------------------------------------------------
// Opcode Constants
// ---------------------------------------------------------------------------
// These numeric opcode identifiers correspond to the I686Opcode enum values
// defined in crate::backend::i686::codegen. They are duplicated here to
// allow the encoder to be compiled independently of the codegen module
// (which may not yet exist during incremental compilation). When codegen.rs
// is available, these values MUST match the discriminant values of I686Opcode.

/// i686 machine instruction opcode identifiers.
///
/// These constants are used to dispatch on `MachineInstr.opcode` values
/// during instruction encoding. They map 1:1 to the `I686Opcode` enum
/// variants defined in the codegen module.
pub mod opcodes {
    // Data movement
    pub const MOV: u32 = 0;
    pub const MOV_SX: u32 = 1;
    pub const MOV_ZX: u32 = 2;
    pub const LEA: u32 = 3;
    pub const PUSH: u32 = 4;
    pub const POP: u32 = 5;

    // Integer ALU
    pub const ADD: u32 = 10;
    pub const ADC: u32 = 11;
    pub const SUB: u32 = 12;
    pub const SBB: u32 = 13;
    pub const AND: u32 = 14;
    pub const OR: u32 = 15;
    pub const XOR: u32 = 16;
    pub const IMUL: u32 = 17;
    pub const IDIV: u32 = 18;
    pub const DIV: u32 = 19;
    pub const MUL: u32 = 20;
    pub const NEG: u32 = 21;
    pub const NOT: u32 = 22;
    pub const INC: u32 = 23;
    pub const DEC: u32 = 24;

    // Shifts
    pub const SHL: u32 = 30;
    pub const SHR: u32 = 31;
    pub const SAR: u32 = 32;
    pub const SHLD: u32 = 33;
    pub const SHRD: u32 = 34;

    // Misc ALU
    pub const CDQ: u32 = 40;
    pub const CMP: u32 = 41;
    pub const TEST: u32 = 42;

    // Control flow
    pub const JMP: u32 = 50;
    pub const JCC: u32 = 51;
    pub const SETCC: u32 = 52;
    pub const CMOVCC: u32 = 53;
    pub const CALL: u32 = 54;
    pub const RET: u32 = 55;

    // x87 FPU
    pub const FLD: u32 = 60;
    pub const FST: u32 = 61;
    pub const FSTP: u32 = 62;
    pub const FADD: u32 = 63;
    pub const FADDP: u32 = 64;
    pub const FSUB: u32 = 65;
    pub const FSUBP: u32 = 66;
    pub const FMUL: u32 = 67;
    pub const FMULP: u32 = 68;
    pub const FDIV: u32 = 69;
    pub const FDIVP: u32 = 70;
    pub const FCHS: u32 = 71;
    pub const FABS: u32 = 72;
    pub const FCOM: u32 = 73;
    pub const FCOMP: u32 = 74;
    pub const FCOMPP: u32 = 75;
    pub const FUCOMIP: u32 = 76;
    pub const FILD: u32 = 77;
    pub const FISTP: u32 = 78;
    pub const FXCH: u32 = 79;

    // No-op
    pub const NOP: u32 = 90;
}

// ---------------------------------------------------------------------------
// Condition code encoding for Jcc / SETcc / CMOVcc
// ---------------------------------------------------------------------------

/// Maps a condition-code index (0–15) to the x86 condition nibble.
///
/// The condition code is used as the low nibble in:
/// - Jcc short: `0x70 + cc`
/// - Jcc near:  `0x0F 0x80 + cc`
/// - SETcc:     `0x0F 0x90 + cc`
/// - CMOVcc:    `0x0F 0x40 + cc`
///
/// Standard x86 condition code encoding:
///   0=O, 1=NO, 2=B/C/NAE, 3=NB/NC/AE, 4=E/Z, 5=NE/NZ,
///   6=BE/NA, 7=A/NBE, 8=S, 9=NS, A=P/PE, B=NP/PO,
///   C=L/NGE, D=GE/NL, E=LE/NG, F=G/NLE
#[inline]
fn cc_nibble(cc_index: u8) -> u8 {
    cc_index & 0x0F
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Encoder state tracked across the instruction stream within a single
/// function assembly pass.
///
/// The assembler creates an `EncoderContext` before encoding each instruction,
/// setting `offset` to the current code buffer position. After encoding, any
/// relocations emitted by the encoder are collected from this context.
#[derive(Debug)]
pub struct EncoderContext {
    /// Current byte offset within the code buffer (start of the instruction
    /// being encoded). Used to compute relocation site offsets.
    pub offset: u32,
    /// Relocations emitted during encoding of the current instruction.
    /// These are collected by the assembler after each `encode_instruction`
    /// call and merged into the module relocation table.
    pub relocations: Vec<RelocationEntry>,
    /// Whether PIC (Position-Independent Code) mode is active.
    /// When true, symbol references emit GOT/PLT-relative relocations
    /// instead of absolute relocations.
    pub pic_mode: bool,
}

/// The result of encoding a single machine instruction.
///
/// Contains the raw machine code bytes and any relocation entries that
/// were generated during encoding (e.g. for symbol references in
/// immediate or displacement fields).
#[derive(Debug, Clone)]
pub struct EncodedInstr {
    /// Raw i686 machine code bytes for this instruction.
    pub bytes: Vec<u8>,
    /// Relocation entries generated during encoding of this instruction.
    /// Offsets are relative to the start of the containing code buffer
    /// (as tracked by `EncoderContext::offset`).
    pub relocations: Vec<RelocationEntry>,
}

/// A memory operand for i686 addressing mode encoding.
///
/// Represents the full range of i686 memory addressing modes:
/// - `[base]` — register indirect
/// - `[base + disp]` — base + displacement
/// - `[base + index * scale]` — base + scaled index
/// - `[base + index * scale + disp]` — full SIB addressing
/// - `[disp32]` — absolute addressing (base=None)
///
/// When `symbol` is `Some`, a relocation is emitted for the displacement
/// field referencing the named symbol.
#[derive(Debug, Clone)]
pub struct MemOperand {
    /// Base register (None for absolute addressing).
    pub base: Option<PhysReg>,
    /// Index register (None if no scaled index).
    pub index: Option<PhysReg>,
    /// Scale factor: 1, 2, 4, or 8.
    pub scale: u8,
    /// Signed displacement.
    pub disp: i32,
    /// Optional symbol name for relocation.
    pub symbol: Option<String>,
}

// ---------------------------------------------------------------------------
// ModR/M and SIB byte encoding
// ---------------------------------------------------------------------------

/// Construct a ModR/M byte from its three fields.
///
/// ```text
/// ModR/M byte layout:
///   bits [7:6] = mod  (addressing mode)
///   bits [5:3] = reg  (register or opcode extension)
///   bits [2:0] = rm   (register or memory operand)
/// ```
///
/// # Arguments
///
/// * `mod_val` — 2-bit addressing mode (0=indirect, 1=disp8, 2=disp32, 3=register)
/// * `reg` — 3-bit register field or opcode extension
/// * `rm` — 3-bit register/memory field
#[inline]
pub fn encode_modrm(mod_val: u8, reg: u8, rm: u8) -> u8 {
    ((mod_val & 0x03) << 6) | ((reg & 0x07) << 3) | (rm & 0x07)
}

/// Construct a SIB (Scale-Index-Base) byte.
///
/// ```text
/// SIB byte layout:
///   bits [7:6] = scale (00=×1, 01=×2, 10=×4, 11=×8)
///   bits [5:3] = index (register number; 4=no index)
///   bits [2:0] = base  (register number; 5=disp32 when mod=00)
/// ```
///
/// # Notes
///
/// - ESP (4) cannot be used as an index register.
/// - When `base` is 5 (EBP encoding) and `mod` is 00, the base is
///   replaced by a 32-bit displacement (disp32-only addressing).
#[inline]
pub fn encode_sib(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x03) << 6) | ((index & 0x07) << 3) | (base & 0x07)
}

/// Extract the 3-bit register encoding value from a [`PhysReg`].
///
/// Maps the i686 GPR indices to their 3-bit encoding:
/// - EAX=0, ECX=1, EDX=2, EBX=3, ESP=4, EBP=5, ESI=6, EDI=7
///
/// For i686, only 3 bits are needed (no REX.B extension unlike x86-64).
/// Delegates to [`registers::encoding()`] for the actual mapping.
#[inline]
pub fn reg_encoding(reg: PhysReg) -> u8 {
    registers::encoding(reg)
}

// ---------------------------------------------------------------------------
// Scale factor encoding
// ---------------------------------------------------------------------------

/// Convert a scale factor (1, 2, 4, 8) to its 2-bit SIB encoding.
///
/// Returns 0 for scale=1, 1 for scale=2, 2 for scale=4, 3 for scale=8.
/// Invalid values default to 0 (scale=1).
#[inline]
fn scale_to_ss(scale: u8) -> u8 {
    match scale {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => 0, // Default to ×1 for invalid values
    }
}

// ---------------------------------------------------------------------------
// Memory operand encoding helpers
// ---------------------------------------------------------------------------

/// Convert a `MachineOperand::Memory` into a `MemOperand` for encoding.
///
/// The `MachineOperand::Memory` variant from `traits.rs` uses a mandatory
/// `PhysReg` base and optional index. This helper wraps it into the
/// encoder's `MemOperand` representation.
fn machine_mem_to_mem_operand(
    base: &PhysReg,
    offset: i32,
    index: &Option<PhysReg>,
    scale: u8,
) -> MemOperand {
    MemOperand {
        base: Some(*base),
        index: *index,
        scale,
        disp: offset,
        symbol: None,
    }
}

/// Encode a memory operand into ModR/M (and optionally SIB + displacement)
/// bytes, appending them to the output buffer.
///
/// # Arguments
///
/// * `buf` — Output byte buffer to append ModR/M, SIB, and displacement to
/// * `reg_field` — The 3-bit value for the ModR/M `reg` field (register
///   operand or opcode extension /digit)
/// * `mem` — The memory operand to encode
/// * `ctx` — Encoder context for relocation emission
///
/// # Addressing Mode Encoding Rules
///
/// - ESP as base always requires a SIB byte (rm=4 signals SIB follows)
/// - EBP as base with no displacement uses mod=01 + disp8(0) to avoid
///   the [disp32] ambiguity (rm=5 mod=00 means absolute addressing)
/// - Displacement is sized as disp8 (±127) or disp32 as appropriate
fn encode_memory(
    buf: &mut Vec<u8>,
    reg_field: u8,
    mem: &MemOperand,
    ctx: &mut EncoderContext,
) {
    let has_base = mem.base.is_some();
    let has_index = mem.index.is_some();

    if !has_base && !has_index {
        // [disp32] — absolute addressing. ModR/M: mod=00, rm=5
        buf.push(encode_modrm(0b00, reg_field, 0b101));
        buf.extend_from_slice(&mem.disp.to_le_bytes());
        // If there's a symbol, emit a relocation for this displacement.
        if let Some(ref sym) = mem.symbol {
            let reloc_type = if ctx.pic_mode {
                I686RelocType::R386Gotoff
            } else {
                I686RelocType::R386_32
            };
            ctx.relocations.push(RelocationEntry {
                offset: ctx.offset + (buf.len() as u32 - 4),
                reloc_type,
                symbol: sym.clone(),
                addend: mem.disp,
            });
        }
        return;
    }

    let base_reg = mem.base.unwrap_or(PhysReg(5)); // 5 = EBP encoding
    let base_enc = reg_encoding(base_reg);
    let needs_sib = has_index || base_enc == 4; // ESP base always needs SIB

    // Determine the mod field and displacement size.
    let (mod_val, disp_bytes): (u8, usize) = if !has_base {
        // No base: [index*scale + disp32] — encode via SIB with base=5 mod=00
        (0b00, 4)
    } else if mem.disp == 0 && base_enc != 5 {
        // [base] or [base + index*scale] — no displacement (mod=00)
        // Note: EBP (enc=5) in mod=00 means [disp32], so we must use
        // mod=01 + disp8(0) for [EBP] with zero displacement.
        (0b00, 0)
    } else if mem.disp >= -128 && mem.disp <= 127 && mem.symbol.is_none() {
        // [base + disp8] — 8-bit signed displacement (mod=01)
        (0b01, 1)
    } else {
        // [base + disp32] — 32-bit displacement (mod=10)
        (0b10, 4)
    };

    if needs_sib {
        // Emit ModR/M with rm=4 (SIB follows)
        buf.push(encode_modrm(mod_val, reg_field, 0b100));

        let index_enc = match mem.index {
            Some(idx) => {
                let enc = reg_encoding(idx);
                // ESP (4) cannot be used as index — use 4 as "no index"
                if enc == 4 { 4 } else { enc }
            }
            None => 4, // No index
        };

        let sib_base = if has_base {
            base_enc
        } else {
            5 // disp32-only base when mod=00
        };

        let ss = scale_to_ss(mem.scale);
        buf.push(encode_sib(ss, index_enc, sib_base));
    } else {
        // Simple ModR/M without SIB
        buf.push(encode_modrm(mod_val, reg_field, base_enc));
    }

    // Emit displacement bytes.
    match disp_bytes {
        1 => buf.push(mem.disp as i8 as u8),
        4 => {
            let disp_offset = ctx.offset + buf.len() as u32;
            buf.extend_from_slice(&mem.disp.to_le_bytes());
            // Emit relocation for symbol reference in displacement.
            if let Some(ref sym) = mem.symbol {
                let reloc_type = if ctx.pic_mode {
                    I686RelocType::R386Gotoff
                } else {
                    I686RelocType::R386_32
                };
                ctx.relocations.push(RelocationEntry {
                    offset: disp_offset,
                    reloc_type,
                    symbol: sym.clone(),
                    addend: mem.disp,
                });
            }
        }
        _ => {} // No displacement
    }
}

// ---------------------------------------------------------------------------
// Instruction-specific encoding helpers
// ---------------------------------------------------------------------------

/// Encode a register-to-register ALU operation (e.g. ADD EAX, ECX).
///
/// Format: `[opcode] ModR/M(mod=11, reg=src, rm=dst)`
fn encode_reg_reg(buf: &mut Vec<u8>, opcode: &[u8], dst: PhysReg, src: PhysReg) {
    buf.extend_from_slice(opcode);
    buf.push(encode_modrm(0b11, reg_encoding(src), reg_encoding(dst)));
}

/// Encode a register-immediate ALU operation with opcode extension.
///
/// Format: `[opcode] ModR/M(mod=11, reg=ext, rm=dst) [imm8|imm32]`
///
/// If the immediate fits in a signed byte and `short_opcode` is provided,
/// uses the compact imm8 encoding; otherwise uses the full imm32 form.
fn encode_reg_imm(
    buf: &mut Vec<u8>,
    opcode_imm32: u8,
    opcode_imm8: u8,
    ext: u8,
    dst: PhysReg,
    imm: i32,
) {
    let dst_enc = reg_encoding(dst);
    if imm >= -128 && imm <= 127 {
        // Short form: opcode_imm8 ModR/M imm8
        buf.push(opcode_imm8);
        buf.push(encode_modrm(0b11, ext, dst_enc));
        buf.push(imm as i8 as u8);
    } else {
        // Full form: opcode_imm32 ModR/M imm32
        buf.push(opcode_imm32);
        buf.push(encode_modrm(0b11, ext, dst_enc));
        buf.extend_from_slice(&imm.to_le_bytes());
    }
}

/// Encode a memory-to-register or register-to-memory operation.
///
/// Format: `[opcode] ModR/M [SIB] [disp]`
fn encode_reg_mem(
    buf: &mut Vec<u8>,
    opcode: &[u8],
    reg: PhysReg,
    mem: &MemOperand,
    ctx: &mut EncoderContext,
) {
    buf.extend_from_slice(opcode);
    encode_memory(buf, reg_encoding(reg), mem, ctx);
}

/// Encode a unary memory operation with opcode extension.
///
/// Format: `[opcode] ModR/M(reg=ext) [SIB] [disp]`
fn encode_mem_ext(
    buf: &mut Vec<u8>,
    opcode: &[u8],
    ext: u8,
    mem: &MemOperand,
    ctx: &mut EncoderContext,
) {
    buf.extend_from_slice(opcode);
    encode_memory(buf, ext, mem, ctx);
}

// ---------------------------------------------------------------------------
// Symbol reference helpers
// ---------------------------------------------------------------------------

/// Emit a 32-bit symbol relocation for a CALL or JMP to a named symbol.
///
/// Writes a placeholder displacement (0x00000000) and records the
/// appropriate relocation (R_386_PC32 for direct calls, R_386_PLT32
/// for PIC mode calls).
fn emit_symbol_call(
    buf: &mut Vec<u8>,
    symbol: &str,
    ctx: &mut EncoderContext,
    is_call: bool,
) {
    // CALL rel32 or JMP rel32 opcode
    if is_call {
        buf.push(0xE8);
    } else {
        buf.push(0xE9);
    }

    let reloc_offset = ctx.offset + buf.len() as u32;

    // Placeholder displacement
    buf.extend_from_slice(&0i32.to_le_bytes());

    let reloc_type = if ctx.pic_mode {
        I686RelocType::R386Plt32
    } else {
        I686RelocType::R386Pc32
    };

    ctx.relocations.push(RelocationEntry {
        offset: reloc_offset,
        reloc_type,
        symbol: symbol.to_string(),
        addend: -4, // PC-relative adjustment
    });
}

/// Emit a 32-bit absolute symbol reference in an immediate field.
///
/// Used for MOV reg, symbol_address patterns. Writes a placeholder
/// value and records an R_386_32 (or R_386_GOT32 in PIC mode) relocation.
fn emit_symbol_imm32(
    buf: &mut Vec<u8>,
    symbol: &str,
    ctx: &mut EncoderContext,
) {
    let reloc_offset = ctx.offset + buf.len() as u32;
    buf.extend_from_slice(&0i32.to_le_bytes());

    let reloc_type = if ctx.pic_mode {
        I686RelocType::R386Got32
    } else {
        I686RelocType::R386_32
    };

    ctx.relocations.push(RelocationEntry {
        offset: reloc_offset,
        reloc_type,
        symbol: symbol.to_string(),
        addend: 0,
    });
}

// ---------------------------------------------------------------------------
// Main encoding function
// ---------------------------------------------------------------------------

/// Encode a single i686 machine instruction into raw binary bytes.
///
/// This is the primary entry point for the encoder. It accepts a
/// [`MachineInstr`] and an [`EncoderContext`], and returns an
/// [`EncodedInstr`] containing the encoded bytes and any relocations.
///
/// # Arguments
///
/// * `instr` — The machine instruction to encode. The `opcode` field
///   identifies the instruction type (must match the opcode constants
///   defined in the `opcodes` module or the `I686Opcode` enum).
/// * `ctx` — Mutable encoder context tracking the current code offset
///   and collecting relocations.
///
/// # Returns
///
/// An [`EncodedInstr`] with the raw machine code bytes and any relocation
/// entries. The assembler appends these bytes to its code buffer and
/// collects the relocations.
///
/// # Encoding Strategy
///
/// The function dispatches on the instruction opcode to select the
/// appropriate encoding format. For each instruction class:
///
/// 1. Extract operands from `instr.operands`
/// 2. Determine the encoding form (reg-reg, reg-imm, reg-mem, etc.)
/// 3. Emit prefix bytes if needed (0x66 for 16-bit, 0x0F for two-byte opcodes)
/// 4. Emit opcode byte(s)
/// 5. Emit ModR/M byte (and SIB, displacement as needed)
/// 6. Emit immediate operand bytes
/// 7. Record relocations for symbol references
pub fn encode_instruction(instr: &MachineInstr, ctx: &mut EncoderContext) -> EncodedInstr {
    let mut bytes: Vec<u8> = Vec::with_capacity(16);
    let mut relocs: Vec<RelocationEntry> = Vec::new();

    // Save relocations length so we can split encoder-generated relocs
    // from context-accumulated relocs at the end.
    let ctx_relocs_before = ctx.relocations.len();

    match instr.opcode {
        // ---------------------------------------------------------------
        // NOP (0x90)
        // ---------------------------------------------------------------
        opcodes::NOP => {
            bytes.push(0x90);
        }

        // ---------------------------------------------------------------
        // MOV — data movement
        // ---------------------------------------------------------------
        opcodes::MOV => {
            encode_mov(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // MOVSX — sign extension (8→32, 16→32)
        // ---------------------------------------------------------------
        opcodes::MOV_SX => {
            encode_movsx_movzx(instr, &mut bytes, ctx, true);
        }

        // ---------------------------------------------------------------
        // MOVZX — zero extension (8→32, 16→32)
        // ---------------------------------------------------------------
        opcodes::MOV_ZX => {
            encode_movsx_movzx(instr, &mut bytes, ctx, false);
        }

        // ---------------------------------------------------------------
        // LEA — load effective address
        // ---------------------------------------------------------------
        opcodes::LEA => {
            encode_lea(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // PUSH
        // ---------------------------------------------------------------
        opcodes::PUSH => {
            encode_push(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // POP
        // ---------------------------------------------------------------
        opcodes::POP => {
            encode_pop(instr, &mut bytes);
        }

        // ---------------------------------------------------------------
        // ADD, ADC, SUB, SBB, AND, OR, XOR, CMP
        // ---------------------------------------------------------------
        opcodes::ADD => encode_alu(instr, &mut bytes, ctx, 0x01, 0x03, 0x81, 0x83, 0),
        opcodes::ADC => encode_alu(instr, &mut bytes, ctx, 0x11, 0x13, 0x81, 0x83, 2),
        opcodes::SUB => encode_alu(instr, &mut bytes, ctx, 0x29, 0x2B, 0x81, 0x83, 5),
        opcodes::SBB => encode_alu(instr, &mut bytes, ctx, 0x19, 0x1B, 0x81, 0x83, 3),
        opcodes::AND => encode_alu(instr, &mut bytes, ctx, 0x21, 0x23, 0x81, 0x83, 4),
        opcodes::OR  => encode_alu(instr, &mut bytes, ctx, 0x09, 0x0B, 0x81, 0x83, 1),
        opcodes::XOR => encode_alu(instr, &mut bytes, ctx, 0x31, 0x33, 0x81, 0x83, 6),
        opcodes::CMP => encode_alu(instr, &mut bytes, ctx, 0x39, 0x3B, 0x81, 0x83, 7),

        // ---------------------------------------------------------------
        // TEST (special: no reg←mem form with same opcode pattern)
        // ---------------------------------------------------------------
        opcodes::TEST => {
            encode_test(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // IMUL (2-operand: 0F AF, 3-operand: 69/6B)
        // ---------------------------------------------------------------
        opcodes::IMUL => {
            encode_imul(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // IDIV (F7 /7), DIV (F7 /6), MUL (F7 /4)
        // ---------------------------------------------------------------
        opcodes::IDIV => encode_unary_rm(instr, &mut bytes, ctx, 0xF7, 7),
        opcodes::DIV  => encode_unary_rm(instr, &mut bytes, ctx, 0xF7, 6),
        opcodes::MUL  => encode_unary_rm(instr, &mut bytes, ctx, 0xF7, 4),
        opcodes::NEG  => encode_unary_rm(instr, &mut bytes, ctx, 0xF7, 3),
        opcodes::NOT  => encode_unary_rm(instr, &mut bytes, ctx, 0xF7, 2),

        // ---------------------------------------------------------------
        // INC (FF /0), DEC (FF /1)
        // ---------------------------------------------------------------
        opcodes::INC => encode_inc_dec(instr, &mut bytes, ctx, true),
        opcodes::DEC => encode_inc_dec(instr, &mut bytes, ctx, false),

        // ---------------------------------------------------------------
        // SHL, SHR, SAR
        // ---------------------------------------------------------------
        opcodes::SHL => encode_shift(instr, &mut bytes, ctx, 4),
        opcodes::SHR => encode_shift(instr, &mut bytes, ctx, 5),
        opcodes::SAR => encode_shift(instr, &mut bytes, ctx, 7),

        // ---------------------------------------------------------------
        // SHLD, SHRD (double-precision shift)
        // ---------------------------------------------------------------
        opcodes::SHLD => encode_double_shift(instr, &mut bytes, 0xA4, 0xA5),
        opcodes::SHRD => encode_double_shift(instr, &mut bytes, 0xAC, 0xAD),

        // ---------------------------------------------------------------
        // CDQ (99) — sign extend EAX into EDX:EAX
        // ---------------------------------------------------------------
        opcodes::CDQ => {
            bytes.push(0x99);
        }

        // ---------------------------------------------------------------
        // JMP — unconditional jump
        // ---------------------------------------------------------------
        opcodes::JMP => {
            encode_jmp(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // Jcc — conditional jump
        // ---------------------------------------------------------------
        opcodes::JCC => {
            encode_jcc(instr, &mut bytes);
        }

        // ---------------------------------------------------------------
        // SETcc — set byte based on condition
        // ---------------------------------------------------------------
        opcodes::SETCC => {
            encode_setcc(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // CMOVcc — conditional move
        // ---------------------------------------------------------------
        opcodes::CMOVCC => {
            encode_cmovcc(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // CALL — function call
        // ---------------------------------------------------------------
        opcodes::CALL => {
            encode_call(instr, &mut bytes, ctx);
        }

        // ---------------------------------------------------------------
        // RET — return
        // ---------------------------------------------------------------
        opcodes::RET => {
            if instr.operands.is_empty() {
                bytes.push(0xC3); // RET
            } else if let Some(MachineOperand::Immediate(n)) = instr.operands.first() {
                bytes.push(0xC2); // RET imm16
                bytes.extend_from_slice(&(*n as u16).to_le_bytes());
            } else {
                bytes.push(0xC3);
            }
        }

        // ---------------------------------------------------------------
        // x87 FPU instructions
        // ---------------------------------------------------------------
        opcodes::FLD    => encode_fld(instr, &mut bytes, ctx),
        opcodes::FST    => encode_fst_fstp(instr, &mut bytes, ctx, false),
        opcodes::FSTP   => encode_fst_fstp(instr, &mut bytes, ctx, true),
        opcodes::FADD   => encode_fpu_arith(instr, &mut bytes, ctx, 0xD8, 0xDC, 0xC0),
        opcodes::FADDP  => { bytes.push(0xDE); encode_fpu_stack_op(&mut bytes, instr, 0xC0); }
        opcodes::FSUB   => encode_fpu_arith(instr, &mut bytes, ctx, 0xD8, 0xDC, 0xE0),
        opcodes::FSUBP  => { bytes.push(0xDE); encode_fpu_stack_op(&mut bytes, instr, 0xE8); }
        opcodes::FMUL   => encode_fpu_arith(instr, &mut bytes, ctx, 0xD8, 0xDC, 0xC8),
        opcodes::FMULP  => { bytes.push(0xDE); encode_fpu_stack_op(&mut bytes, instr, 0xC8); }
        opcodes::FDIV   => encode_fpu_arith(instr, &mut bytes, ctx, 0xD8, 0xDC, 0xF0),
        opcodes::FDIVP  => { bytes.push(0xDE); encode_fpu_stack_op(&mut bytes, instr, 0xF8); }
        opcodes::FCHS   => { bytes.push(0xD9); bytes.push(0xE0); }
        opcodes::FABS   => { bytes.push(0xD9); bytes.push(0xE1); }
        opcodes::FUCOMIP => {
            bytes.push(0xDF);
            let st_i = extract_fpu_index(instr, 0);
            bytes.push(0xE8 + st_i);
        }
        opcodes::FILD   => encode_fild(instr, &mut bytes, ctx),
        opcodes::FISTP  => encode_fistp(instr, &mut bytes, ctx),
        opcodes::FXCH   => {
            bytes.push(0xD9);
            let st_i = extract_fpu_index(instr, 0);
            bytes.push(0xC8 + st_i);
        }
        opcodes::FCOM => {
            bytes.push(0xD8);
            let st_i = extract_fpu_index(instr, 0);
            bytes.push(0xD0 + st_i);
        }
        opcodes::FCOMP => {
            bytes.push(0xD8);
            let st_i = extract_fpu_index(instr, 0);
            bytes.push(0xD8 + st_i);
        }
        opcodes::FCOMPP => {
            bytes.push(0xDE);
            bytes.push(0xD9);
        }

        // ---------------------------------------------------------------
        // Unknown opcode — emit nothing (assembler will warn)
        // ---------------------------------------------------------------
        _ => {
            // Unknown opcode: return empty bytes. The assembler driver
            // checks for zero-length encoding and emits a warning.
        }
    }

    // Collect any relocations emitted into ctx during this instruction.
    let new_ctx_relocs: Vec<RelocationEntry> =
        ctx.relocations.drain(ctx_relocs_before..).collect();
    relocs.extend(new_ctx_relocs);

    EncodedInstr {
        bytes,
        relocations: relocs,
    }
}

// ---------------------------------------------------------------------------
// Per-instruction encoding implementations
// ---------------------------------------------------------------------------

/// Encode a MOV instruction in its various forms.
///
/// Handles:
/// - MOV reg, reg (89 /r)
/// - MOV reg, imm32 (B8+rd id)
/// - MOV reg, [mem] (8B /r)
/// - MOV [mem], reg (89 /r)
/// - MOV [mem], imm32 (C7 /0 id)
/// - MOV reg, symbol (B8+rd with relocation)
fn encode_mov(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    match (&ops[0], &ops[1]) {
        // MOV reg, reg
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            buf.push(0x89);
            buf.push(encode_modrm(0b11, reg_encoding(*src), reg_encoding(*dst)));
        }
        // MOV reg, imm32
        (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
            buf.push(0xB8 + reg_encoding(*dst));
            buf.extend_from_slice(&(*imm as i32).to_le_bytes());
        }
        // MOV reg, symbol
        (MachineOperand::Register(dst), MachineOperand::Symbol(sym)) => {
            buf.push(0xB8 + reg_encoding(*dst));
            emit_symbol_imm32(buf, sym, ctx);
        }
        // MOV reg, [mem]
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset, index, scale }) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_reg_mem(buf, &[0x8B], *dst, &mem, ctx);
        }
        // MOV [mem], reg
        (MachineOperand::Memory { base, offset, index, scale }, MachineOperand::Register(src)) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_reg_mem(buf, &[0x89], *src, &mem, ctx);
        }
        // MOV [mem], imm32
        (MachineOperand::Memory { base, offset, index, scale }, MachineOperand::Immediate(imm)) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_mem_ext(buf, &[0xC7], 0, &mem, ctx);
            buf.extend_from_slice(&(*imm as i32).to_le_bytes());
        }
        _ => {} // Unsupported operand combination
    }
}

/// Encode MOVSX or MOVZX (sign/zero extension).
///
/// - MOVSX 8→32:  0F BE /r
/// - MOVSX 16→32: 0F BF /r
/// - MOVZX 8→32:  0F B6 /r
/// - MOVZX 16→32: 0F B7 /r
///
/// The source width is inferred from the third operand (immediate hint)
/// or defaults to 8-bit if only two operands are present.
fn encode_movsx_movzx(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    is_sign_extend: bool,
) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    // Determine source width from optional third operand (width hint).
    let src_width: u8 = if ops.len() > 2 {
        if let MachineOperand::Immediate(w) = &ops[2] {
            *w as u8
        } else {
            8
        }
    } else {
        8 // Default to byte-width source
    };

    let opcode_byte = match (is_sign_extend, src_width) {
        (true, 8)  => 0xBE, // MOVSX r32, r/m8
        (true, _)  => 0xBF, // MOVSX r32, r/m16
        (false, 8) => 0xB6, // MOVZX r32, r/m8
        (false, _) => 0xB7, // MOVZX r32, r/m16
    };

    buf.push(0x0F);
    buf.push(opcode_byte);

    match (&ops[0], &ops[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            buf.push(encode_modrm(0b11, reg_encoding(*dst), reg_encoding(*src)));
        }
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset, index, scale }) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_memory(buf, reg_encoding(*dst), &mem, ctx);
        }
        _ => {}
    }
}

/// Encode LEA (Load Effective Address): 8D /r
fn encode_lea(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    if let (MachineOperand::Register(dst), MachineOperand::Memory { base, offset, index, scale }) =
        (&ops[0], &ops[1])
    {
        let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
        encode_reg_mem(buf, &[0x8D], *dst, &mem, ctx);
    }
}

/// Encode PUSH instruction.
///
/// - PUSH reg: 50+rd
/// - PUSH imm8: 6A ib
/// - PUSH imm32: 68 id
/// - PUSH [mem]: FF /6
fn encode_push(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    match &ops[0] {
        MachineOperand::Register(reg) => {
            buf.push(0x50 + reg_encoding(*reg));
        }
        MachineOperand::Immediate(imm) => {
            let val = *imm as i32;
            if val >= -128 && val <= 127 {
                buf.push(0x6A);
                buf.push(val as i8 as u8);
            } else {
                buf.push(0x68);
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_mem_ext(buf, &[0xFF], 6, &mem, ctx);
        }
        _ => {}
    }
}

/// Encode POP instruction.
///
/// - POP reg: 58+rd
fn encode_pop(instr: &MachineInstr, buf: &mut Vec<u8>) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    if let MachineOperand::Register(reg) = &ops[0] {
        buf.push(0x58 + reg_encoding(*reg));
    }
}

/// Encode a standard two-operand ALU instruction (ADD, SUB, AND, OR, XOR, CMP).
///
/// # Parameters
///
/// * `op_rm_r` — Opcode for reg→r/m (e.g. 0x01 for ADD r/m32, r32)
/// * `op_r_rm` — Opcode for r/m→reg (e.g. 0x03 for ADD r32, r/m32)
/// * `op_imm32` — Opcode for r/m32, imm32 (e.g. 0x81)
/// * `op_imm8` — Opcode for r/m32, imm8 (e.g. 0x83)
/// * `ext` — ModR/M extension digit (/0 for ADD, /5 for SUB, etc.)
fn encode_alu(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    op_rm_r: u8,
    op_r_rm: u8,
    op_imm32: u8,
    op_imm8: u8,
    ext: u8,
) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    match (&ops[0], &ops[1]) {
        // reg, reg
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            encode_reg_reg(buf, &[op_rm_r], *dst, *src);
        }
        // reg, imm
        (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
            encode_reg_imm(buf, op_imm32, op_imm8, ext, *dst, *imm as i32);
        }
        // reg, [mem]
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset, index, scale }) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_reg_mem(buf, &[op_r_rm], *dst, &mem, ctx);
        }
        // [mem], reg
        (MachineOperand::Memory { base, offset, index, scale }, MachineOperand::Register(src)) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_reg_mem(buf, &[op_rm_r], *src, &mem, ctx);
        }
        // [mem], imm
        (MachineOperand::Memory { base, offset, index, scale }, MachineOperand::Immediate(imm)) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            let val = *imm as i32;
            if val >= -128 && val <= 127 {
                encode_mem_ext(buf, &[op_imm8], ext, &mem, ctx);
                buf.push(val as i8 as u8);
            } else {
                encode_mem_ext(buf, &[op_imm32], ext, &mem, ctx);
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        _ => {}
    }
}

/// Encode TEST instruction.
///
/// - TEST r/m32, r32: 85 /r
/// - TEST r/m32, imm32: F7 /0
fn encode_test(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    match (&ops[0], &ops[1]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            encode_reg_reg(buf, &[0x85], *dst, *src);
        }
        (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
            buf.push(0xF7);
            buf.push(encode_modrm(0b11, 0, reg_encoding(*dst)));
            buf.extend_from_slice(&(*imm as i32).to_le_bytes());
        }
        (MachineOperand::Memory { base, offset, index, scale }, MachineOperand::Register(src)) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_reg_mem(buf, &[0x85], *src, &mem, ctx);
        }
        _ => {}
    }
}

/// Encode 2-operand IMUL (0F AF /r) or 3-operand IMUL (69/6B).
fn encode_imul(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;

    match ops.len() {
        // 1 operand: implicit EAX * r/m32 → EDX:EAX (F7 /5)
        1 => {
            encode_unary_rm(instr, buf, ctx, 0xF7, 5);
        }
        // 2 operands: IMUL r32, r/m32 (0F AF /r)
        2 => {
            match (&ops[0], &ops[1]) {
                (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
                    buf.push(0x0F);
                    buf.push(0xAF);
                    buf.push(encode_modrm(0b11, reg_encoding(*dst), reg_encoding(*src)));
                }
                (MachineOperand::Register(dst), MachineOperand::Memory { base, offset, index, scale }) => {
                    let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
                    buf.push(0x0F);
                    buf.push(0xAF);
                    encode_memory(buf, reg_encoding(*dst), &mem, ctx);
                }
                _ => {}
            }
        }
        // 3 operands: IMUL r32, r/m32, imm (69/6B)
        _ => {
            if let (
                MachineOperand::Register(dst),
                MachineOperand::Register(src),
                MachineOperand::Immediate(imm),
            ) = (&ops[0], &ops[1], &ops[2])
            {
                let val = *imm as i32;
                if val >= -128 && val <= 127 {
                    buf.push(0x6B);
                    buf.push(encode_modrm(0b11, reg_encoding(*dst), reg_encoding(*src)));
                    buf.push(val as i8 as u8);
                } else {
                    buf.push(0x69);
                    buf.push(encode_modrm(0b11, reg_encoding(*dst), reg_encoding(*src)));
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
    }
}

/// Encode a unary r/m32 instruction with opcode extension (NEG, NOT, IDIV, etc.).
///
/// Format: `opcode ModR/M(mod=11, reg=ext, rm=operand)` for register,
///         `opcode ModR/M(reg=ext) [SIB] [disp]` for memory.
fn encode_unary_rm(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    opcode: u8,
    ext: u8,
) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    match &ops[0] {
        MachineOperand::Register(reg) => {
            buf.push(opcode);
            buf.push(encode_modrm(0b11, ext, reg_encoding(*reg)));
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_mem_ext(buf, &[opcode], ext, &mem, ctx);
        }
        _ => {}
    }
}

/// Encode INC/DEC (short form: 40+rd / 48+rd, or long form: FF /0 or FF /1).
fn encode_inc_dec(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    is_inc: bool,
) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    let ext = if is_inc { 0 } else { 1 };

    match &ops[0] {
        MachineOperand::Register(reg) => {
            // Use short form: 40+rd for INC, 48+rd for DEC
            let base = if is_inc { 0x40 } else { 0x48 };
            buf.push(base + reg_encoding(*reg));
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_mem_ext(buf, &[0xFF], ext, &mem, ctx);
        }
        _ => {}
    }
}

/// Encode shift instructions (SHL, SHR, SAR).
///
/// - Shift by CL: D3 /ext
/// - Shift by imm8: C1 /ext ib
/// - Shift by 1 (special): D1 /ext
fn encode_shift(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    ext: u8,
) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    match (&ops[0], &ops[1]) {
        // reg, imm
        (MachineOperand::Register(dst), MachineOperand::Immediate(count)) => {
            let c = *count as u8;
            if c == 1 {
                buf.push(0xD1);
                buf.push(encode_modrm(0b11, ext, reg_encoding(*dst)));
            } else {
                buf.push(0xC1);
                buf.push(encode_modrm(0b11, ext, reg_encoding(*dst)));
                buf.push(c);
            }
        }
        // reg, CL (register operand — CL is implicit for variable shifts)
        (MachineOperand::Register(dst), MachineOperand::Register(_cl)) => {
            buf.push(0xD3);
            buf.push(encode_modrm(0b11, ext, reg_encoding(*dst)));
        }
        // [mem], imm
        (MachineOperand::Memory { base, offset, index, scale }, MachineOperand::Immediate(count)) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            let c = *count as u8;
            if c == 1 {
                encode_mem_ext(buf, &[0xD1], ext, &mem, ctx);
            } else {
                encode_mem_ext(buf, &[0xC1], ext, &mem, ctx);
                buf.push(c);
            }
        }
        _ => {}
    }
}

/// Encode double-precision shift (SHLD/SHRD).
///
/// - SHLD r/m32, r32, imm8: 0F A4 /r ib
/// - SHLD r/m32, r32, CL:   0F A5 /r
/// - SHRD r/m32, r32, imm8: 0F AC /r ib
/// - SHRD r/m32, r32, CL:   0F AD /r
fn encode_double_shift(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    imm_opcode: u8,
    cl_opcode: u8,
) {
    let ops = &instr.operands;
    if ops.len() < 3 {
        return;
    }

    if let (
        MachineOperand::Register(dst),
        MachineOperand::Register(src),
        third,
    ) = (&ops[0], &ops[1], &ops[2])
    {
        match third {
            MachineOperand::Immediate(count) => {
                buf.push(0x0F);
                buf.push(imm_opcode);
                buf.push(encode_modrm(0b11, reg_encoding(*src), reg_encoding(*dst)));
                buf.push(*count as u8);
            }
            MachineOperand::Register(_) => {
                // CL variant
                buf.push(0x0F);
                buf.push(cl_opcode);
                buf.push(encode_modrm(0b11, reg_encoding(*src), reg_encoding(*dst)));
            }
            _ => {}
        }
    }
}

/// Encode JMP (unconditional jump).
///
/// - JMP rel32: E9 cd
/// - JMP rel8:  EB cb (only used when target is known and short)
/// - JMP r/m32: FF /4 (indirect jump)
/// - JMP symbol: E9 + relocation
fn encode_jmp(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    match &ops[0] {
        MachineOperand::Label(_) => {
            // Emit JMP rel32 with placeholder displacement.
            // The assembler's fixup resolution will patch this.
            buf.push(0xE9);
            buf.extend_from_slice(&0i32.to_le_bytes());
        }
        MachineOperand::Symbol(sym) => {
            emit_symbol_call(buf, sym, ctx, false);
        }
        MachineOperand::Register(reg) => {
            // JMP r/m32 (indirect): FF /4
            buf.push(0xFF);
            buf.push(encode_modrm(0b11, 4, reg_encoding(*reg)));
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            // JMP [mem] (indirect): FF /4
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_mem_ext(buf, &[0xFF], 4, &mem, ctx);
        }
        _ => {}
    }
}

/// Encode Jcc (conditional jump).
///
/// The condition code is extracted from the first operand (immediate value
/// representing the condition code index). The second operand is the label.
///
/// - Jcc rel32: 0F 80+cc cd (6 bytes)
fn encode_jcc(instr: &MachineInstr, buf: &mut Vec<u8>) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    // First operand: condition code (immediate)
    let cc = if let MachineOperand::Immediate(cc_val) = &ops[0] {
        cc_nibble(*cc_val as u8)
    } else {
        0x04 // Default to JE if malformed
    };

    // Second operand: label (filled by fixup) or symbol
    // Emit Jcc rel32: 0F 80+cc cd
    buf.push(0x0F);
    buf.push(0x80 + cc);
    buf.extend_from_slice(&0i32.to_le_bytes()); // Placeholder
}

/// Encode SETcc (set byte based on condition).
///
/// - SETcc r/m8: 0F 90+cc /0
///
/// First operand: condition code (immediate). Second: destination register.
fn encode_setcc(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.len() < 2 {
        return;
    }

    let cc = if let MachineOperand::Immediate(cc_val) = &ops[0] {
        cc_nibble(*cc_val as u8)
    } else {
        0x04
    };

    buf.push(0x0F);
    buf.push(0x90 + cc);

    match &ops[1] {
        MachineOperand::Register(dst) => {
            buf.push(encode_modrm(0b11, 0, reg_encoding(*dst)));
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_memory(buf, 0, &mem, ctx);
        }
        _ => {}
    }
}

/// Encode CMOVcc (conditional move).
///
/// - CMOVcc r32, r/m32: 0F 40+cc /r
///
/// First operand: condition code. Second: destination. Third: source.
fn encode_cmovcc(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.len() < 3 {
        return;
    }

    let cc = if let MachineOperand::Immediate(cc_val) = &ops[0] {
        cc_nibble(*cc_val as u8)
    } else {
        0x04
    };

    buf.push(0x0F);
    buf.push(0x40 + cc);

    match (&ops[1], &ops[2]) {
        (MachineOperand::Register(dst), MachineOperand::Register(src)) => {
            buf.push(encode_modrm(0b11, reg_encoding(*dst), reg_encoding(*src)));
        }
        (MachineOperand::Register(dst), MachineOperand::Memory { base, offset, index, scale }) => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_memory(buf, reg_encoding(*dst), &mem, ctx);
        }
        _ => {}
    }
}

/// Encode CALL instruction.
///
/// - CALL rel32: E8 cd (direct call to symbol)
/// - CALL r/m32: FF /2 (indirect call)
fn encode_call(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    match &ops[0] {
        MachineOperand::Symbol(sym) => {
            emit_symbol_call(buf, sym, ctx, true);
        }
        MachineOperand::Label(_) => {
            // Internal call — emit CALL rel32 with placeholder
            buf.push(0xE8);
            buf.extend_from_slice(&0i32.to_le_bytes());
        }
        MachineOperand::Register(reg) => {
            // CALL r/m32 (indirect): FF /2
            buf.push(0xFF);
            buf.push(encode_modrm(0b11, 2, reg_encoding(*reg)));
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            // CALL [mem] (indirect): FF /2
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            encode_mem_ext(buf, &[0xFF], 2, &mem, ctx);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// x87 FPU encoding helpers
// ---------------------------------------------------------------------------

/// Extract the x87 FPU stack register index (0–7) from an operand.
///
/// FPU registers in the register file have indices 24–31 (ST0–ST7).
/// This function extracts the stack position (0–7) for FPU instruction
/// encoding.
fn extract_fpu_index(instr: &MachineInstr, operand_idx: usize) -> u8 {
    if let Some(MachineOperand::Register(reg)) = instr.operands.get(operand_idx) {
        let idx = reg.index();
        // FPU registers are at indices 24–31; extract the ST(i) position
        if idx >= 24 && idx < 32 {
            (idx - 24) as u8
        } else {
            0 // Default to ST(0)
        }
    } else if let Some(MachineOperand::Immediate(val)) = instr.operands.get(operand_idx) {
        (*val as u8) & 0x07
    } else {
        0
    }
}

/// Encode an x87 FPU stack operation (e.g. FADDP, FMULP).
///
/// These instructions use the pattern: escape_byte (already emitted)
/// followed by base_modrm + ST(i) index.
fn encode_fpu_stack_op(buf: &mut Vec<u8>, instr: &MachineInstr, base_modrm: u8) {
    let st_i = extract_fpu_index(instr, 0);
    buf.push(base_modrm + st_i);
}

/// Encode FLD instruction.
///
/// - FLD mem32: D9 /0
/// - FLD mem64: DD /0
/// - FLD ST(i): D9 C0+i
fn encode_fld(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    match &ops[0] {
        MachineOperand::Register(reg) => {
            // FLD ST(i)
            let st_i = reg.index();
            if st_i >= 24 && st_i < 32 {
                buf.push(0xD9);
                buf.push(0xC0 + (st_i - 24) as u8);
            }
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            // Check for 64-bit via optional second operand hint
            let is_64 = ops.len() > 1
                && matches!(&ops[1], MachineOperand::Immediate(64));
            if is_64 {
                encode_mem_ext(buf, &[0xDD], 0, &mem, ctx); // FLD mem64
            } else {
                encode_mem_ext(buf, &[0xD9], 0, &mem, ctx); // FLD mem32
            }
        }
        _ => {}
    }
}

/// Encode FST/FSTP instruction.
///
/// - FST mem32:  D9 /2, FSTP mem32:  D9 /3
/// - FST mem64:  DD /2, FSTP mem64:  DD /3
/// - FSTP ST(i): DD D8+i
fn encode_fst_fstp(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    is_pop: bool,
) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    let ext = if is_pop { 3 } else { 2 };

    match &ops[0] {
        MachineOperand::Register(reg) if is_pop => {
            // FSTP ST(i): DD D8+i
            let st_i = reg.index();
            if st_i >= 24 && st_i < 32 {
                buf.push(0xDD);
                buf.push(0xD8 + (st_i - 24) as u8);
            }
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            let is_64 = ops.len() > 1
                && matches!(&ops[1], MachineOperand::Immediate(64));
            if is_64 {
                encode_mem_ext(buf, &[0xDD], ext, &mem, ctx);
            } else {
                encode_mem_ext(buf, &[0xD9], ext, &mem, ctx);
            }
        }
        _ => {}
    }
}

/// Encode an x87 FPU arithmetic instruction (FADD, FSUB, FMUL, FDIV).
///
/// - mem32: escape32 /0 (0xD8 for single, 0xDC for double)
/// - ST(0), ST(i): escape32 (base_modrm + i)
fn encode_fpu_arith(
    instr: &MachineInstr,
    buf: &mut Vec<u8>,
    ctx: &mut EncoderContext,
    escape32: u8,
    escape64: u8,
    reg_modrm_base: u8,
) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    match &ops[0] {
        MachineOperand::Register(reg) => {
            // FADD ST(0), ST(i) or similar
            let st_i = if reg.index() >= 24 { (reg.index() - 24) as u8 } else { 0 };
            buf.push(escape32);
            buf.push(reg_modrm_base + st_i);
        }
        MachineOperand::Memory { base, offset, index, scale } => {
            let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
            let is_64 = ops.len() > 1
                && matches!(&ops[1], MachineOperand::Immediate(64));
            if is_64 {
                encode_mem_ext(buf, &[escape64], 0, &mem, ctx);
            } else {
                encode_mem_ext(buf, &[escape32], 0, &mem, ctx);
            }
        }
        _ => {}
    }
}

/// Encode FILD (push integer from memory onto x87 stack).
///
/// - FILD mem32: DB /0
/// - FILD mem64: DF /5
fn encode_fild(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    if let MachineOperand::Memory { base, offset, index, scale } = &ops[0] {
        let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
        let is_64 = ops.len() > 1
            && matches!(&ops[1], MachineOperand::Immediate(64));
        if is_64 {
            encode_mem_ext(buf, &[0xDF], 5, &mem, ctx);
        } else {
            encode_mem_ext(buf, &[0xDB], 0, &mem, ctx);
        }
    }
}

/// Encode FISTP (store x87 top to memory as integer, with pop).
///
/// - FISTP mem32: DB /3
/// - FISTP mem64: DF /7
fn encode_fistp(instr: &MachineInstr, buf: &mut Vec<u8>, ctx: &mut EncoderContext) {
    let ops = &instr.operands;
    if ops.is_empty() {
        return;
    }

    if let MachineOperand::Memory { base, offset, index, scale } = &ops[0] {
        let mem = machine_mem_to_mem_operand(base, *offset, index, *scale);
        let is_64 = ops.len() > 1
            && matches!(&ops[1], MachineOperand::Immediate(64));
        if is_64 {
            encode_mem_ext(buf, &[0xDF], 7, &mem, ctx);
        } else {
            encode_mem_ext(buf, &[0xDB], 3, &mem, ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_modrm_register_direct() {
        // mod=11, reg=0 (EAX), rm=1 (ECX)
        let byte = encode_modrm(0b11, 0, 1);
        assert_eq!(byte, 0xC1);
    }

    #[test]
    fn test_encode_modrm_indirect() {
        // mod=00, reg=3 (EBX), rm=6 (ESI)
        let byte = encode_modrm(0b00, 3, 6);
        assert_eq!(byte, 0x1E);
    }

    #[test]
    fn test_encode_sib() {
        // scale=2 (×4), index=1 (ECX), base=0 (EAX)
        let byte = encode_sib(2, 1, 0);
        assert_eq!(byte, 0x88); // (2<<6)|(1<<3)|0 = 128+8+0 = 0x88
    }

    #[test]
    fn test_scale_to_ss() {
        assert_eq!(scale_to_ss(1), 0);
        assert_eq!(scale_to_ss(2), 1);
        assert_eq!(scale_to_ss(4), 2);
        assert_eq!(scale_to_ss(8), 3);
        assert_eq!(scale_to_ss(3), 0); // Invalid, defaults to ×1
    }

    #[test]
    fn test_encode_nop() {
        let instr = MachineInstr {
            opcode: opcodes::NOP,
            operands: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x90]);
        assert!(result.relocations.is_empty());
    }

    #[test]
    fn test_encode_ret() {
        let instr = MachineInstr {
            opcode: opcodes::RET,
            operands: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: true,
            is_call: false,
            is_return: true,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0xC3]);
    }

    #[test]
    fn test_encode_cdq() {
        let instr = MachineInstr {
            opcode: opcodes::CDQ,
            operands: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x99]);
    }

    #[test]
    fn test_encode_push_reg() {
        // PUSH EAX (50+0 = 0x50)
        let instr = MachineInstr {
            opcode: opcodes::PUSH,
            operands: vec![MachineOperand::Register(PhysReg(0))],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x50]);
    }

    #[test]
    fn test_encode_pop_reg() {
        // POP ECX (58+1 = 0x59)
        let instr = MachineInstr {
            opcode: opcodes::POP,
            operands: vec![MachineOperand::Register(PhysReg(1))],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x59]);
    }

    #[test]
    fn test_encode_mov_reg_reg() {
        // MOV ECX, EAX → 89 C1 (89 /r: mod=11, reg=EAX(0), rm=ECX(1))
        let instr = MachineInstr {
            opcode: opcodes::MOV,
            operands: vec![
                MachineOperand::Register(PhysReg(1)), // dst: ECX
                MachineOperand::Register(PhysReg(0)), // src: EAX
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x89, 0xC1]);
    }

    #[test]
    fn test_encode_mov_reg_imm32() {
        // MOV EAX, 0x12345678 → B8 78 56 34 12
        let instr = MachineInstr {
            opcode: opcodes::MOV,
            operands: vec![
                MachineOperand::Register(PhysReg(0)), // dst: EAX
                MachineOperand::Immediate(0x12345678),
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0xB8, 0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_encode_jmp_label() {
        // JMP label → E9 00 00 00 00 (placeholder)
        let instr = MachineInstr {
            opcode: opcodes::JMP,
            operands: vec![MachineOperand::Label(0)],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: true,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes.len(), 5);
        assert_eq!(result.bytes[0], 0xE9);
    }

    #[test]
    fn test_encode_call_symbol_non_pic() {
        // CALL printf → E8 + relocation
        let instr = MachineInstr {
            opcode: opcodes::CALL,
            operands: vec![MachineOperand::Symbol("printf".to_string())],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: true,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes.len(), 5);
        assert_eq!(result.bytes[0], 0xE8);
        assert_eq!(result.relocations.len(), 1);
        assert_eq!(result.relocations[0].symbol, "printf");
        assert!(matches!(result.relocations[0].reloc_type, I686RelocType::R386Pc32));
    }

    #[test]
    fn test_encode_call_symbol_pic() {
        // CALL printf@PLT → E8 + R_386_PLT32 relocation
        let instr = MachineInstr {
            opcode: opcodes::CALL,
            operands: vec![MachineOperand::Symbol("printf".to_string())],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: true,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: true,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes.len(), 5);
        assert_eq!(result.relocations.len(), 1);
        assert!(matches!(result.relocations[0].reloc_type, I686RelocType::R386Plt32));
    }

    #[test]
    fn test_encode_add_reg_imm_short() {
        // ADD ECX, 5 → 83 C1 05 (imm8 form: 83 /0 ib)
        let instr = MachineInstr {
            opcode: opcodes::ADD,
            operands: vec![
                MachineOperand::Register(PhysReg(1)), // ECX
                MachineOperand::Immediate(5),
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x83, 0xC1, 0x05]);
    }

    #[test]
    fn test_encode_inc_reg() {
        // INC EAX → 40 (short form 40+rd)
        let instr = MachineInstr {
            opcode: opcodes::INC,
            operands: vec![MachineOperand::Register(PhysReg(0))],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0x40]);
    }

    #[test]
    fn test_encode_fchs() {
        let instr = MachineInstr {
            opcode: opcodes::FCHS,
            operands: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            is_terminator: false,
            is_call: false,
            is_return: false,
        };
        let mut ctx = EncoderContext {
            offset: 0,
            relocations: Vec::new(),
            pic_mode: false,
        };
        let result = encode_instruction(&instr, &mut ctx);
        assert_eq!(result.bytes, vec![0xD9, 0xE0]);
    }
}
