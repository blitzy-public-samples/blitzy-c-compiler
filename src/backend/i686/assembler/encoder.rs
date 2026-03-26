//! # i686 (32-bit x86) Instruction Encoder
//!
//! Core encoding engine for BCC's built-in i686 assembler.  Converts
//! [`MachineInstruction`]s into raw machine code bytes with proper ModR/M,
//! SIB, displacement, and immediate encoding.
//!
//! ## Encoding Coverage
//!
//! - **Data movement**: MOV, MOVZX, MOVSX, LEA, PUSH, POP, XCHG
//! - **Arithmetic**: ADD, SUB, ADC, SBB, IMUL, IDIV, MUL, DIV, NEG, INC, DEC, CDQ
//! - **Bitwise**: AND, OR, XOR, NOT, SHL, SHR, SAR
//! - **Comparison / Test**: CMP, TEST, SETcc, CMOVcc
//! - **Control flow**: JMP, Jcc, CALL, RET
//! - **x87 FPU**: FLD, FSTP, FADD, FSUB, FMUL, FDIV, FCHS, FCOMP, FILD, FISTP, FXCH, FUCOMIP
//! - **Stack frame**: ENTER, LEAVE
//! - **Miscellaneous**: NOP, INT3, UD2
//!
//! ## Key Differences from x86-64
//!
//! - **No REX prefix** — register indices 0–7 fit in 3 bits
//! - **No RIP-relative addressing** — `mod=00, r/m=101` means `[disp32]`
//! - **INC/DEC short forms** — `0x40+rd` / `0x48+rd` are available
//! - **x87 FPU active** — D8h–DFh opcodes for floating-point
//! - **32-bit immediates/displacements only** — no 8-byte immediates
//!
//! ## Zero-Dependency
//!
//! Only `crate::` and `std::` imports.  No external crates.

use crate::backend::i686::assembler::relocations;
use crate::backend::i686::codegen;
use crate::backend::i686::registers;
use crate::backend::traits::{MachineInstruction, MachineOperand};
use crate::common::fx_hash::FxHashMap;

// ===========================================================================
// Exported Types
// ===========================================================================

/// Result of encoding a single i686 instruction into machine code bytes.
///
/// Contains the raw bytes and an optional relocation entry when the
/// instruction references an unresolved symbol.
pub struct EncodedInstruction {
    /// Encoded machine code bytes in emission order: optional prefix(es),
    /// opcode byte(s), ModR/M, SIB, displacement, immediate.
    /// Multi-byte values are little-endian.
    pub bytes: Vec<u8>,

    /// Optional relocation entry for unresolved external symbol references.
    /// `None` for fully-resolved instructions (register ops, known labels).
    pub relocation: Option<InstructionRelocation>,
}

/// A relocation entry emitted during instruction encoding.
///
/// Records a byte range within the encoded instruction that must be
/// patched by the linker to the correct symbol address.
pub struct InstructionRelocation {
    /// Byte offset within [`EncodedInstruction::bytes`] where the
    /// relocatable 32-bit value begins.
    pub offset_in_instruction: usize,

    /// Name of the referenced symbol.
    pub symbol: String,

    /// ELF relocation type — one of the `R_386_*` constants from
    /// [`crate::backend::i686::assembler::relocations`].
    pub rel_type: u32,

    /// Addend for the relocation computation.  For `R_386_PC32` this is
    /// typically `−4` to account for the displacement being relative to
    /// the byte *after* the immediate field.
    pub addend: i64,
}

// ===========================================================================
// Internal Constants
// ===========================================================================

/// ESP register index — triggers mandatory SIB encoding.
const ESP_IDX: u8 = registers::ESP as u8;
/// EBP register index — special-case for `[disp32]` vs `[EBP]`.
const EBP_IDX: u8 = registers::EBP as u8;
/// EAX register index — enables short-form ALU encodings.
const EAX_IDX: u8 = registers::EAX as u8;

// ===========================================================================
// Low-Level Encoding Helpers
// ===========================================================================

/// Construct a ModR/M byte: `[mod(2)][reg(3)][r/m(3)]`.
#[inline]
fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0x03) << 6) | ((reg & 0x07) << 3) | (rm & 0x07)
}

/// Construct a SIB byte: `[scale(2)][index(3)][base(3)]`.
#[inline]
fn sib_byte(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x03) << 6) | ((index & 0x07) << 3) | (base & 0x07)
}

/// True if `val` fits in a signed 8-bit immediate (−128..127).
#[inline]
fn fits_in_i8(val: i64) -> bool {
    (-128..=127).contains(&val)
}

/// Convert a scale factor (1/2/4/8) to the 2-bit SIB scale field.
#[inline]
fn scale_to_sib(scale: u8) -> u8 {
    match scale {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => 0,
    }
}

/// Encode an 8-bit signed immediate.
#[inline]
fn emit_i8(val: i64) -> u8 {
    (val as i8) as u8
}

/// Encode a 16-bit value as 2 little-endian bytes.
#[inline]
fn emit_i16(val: i64) -> [u8; 2] {
    (val as u16).to_le_bytes()
}

/// Encode a 32-bit value as 4 little-endian bytes.
#[inline]
fn emit_i32(val: i64) -> [u8; 4] {
    (val as i32 as u32).to_le_bytes()
}

/// Extract a GPR index (0–7) from a register operand.
fn gpr(op: &MachineOperand) -> Result<u8, String> {
    match op {
        MachineOperand::Register(r) => {
            let idx = *r;
            if idx > 7 && !registers::is_fpu_reg(idx) {
                return Err(format!(
                    "i686: register index {} exceeds 7 (no REX prefix available)",
                    idx
                ));
            }
            Ok(idx as u8)
        }
        _ => Err(format!("Expected register, got {}", op)),
    }
}

/// Extract an immediate value.
fn imm(op: &MachineOperand) -> Result<i64, String> {
    match op {
        MachineOperand::Immediate(v) => Ok(*v),
        _ => Err(format!("Expected immediate, got {}", op)),
    }
}

/// True for Memory or FrameSlot operands.
#[inline]
fn is_mem(op: &MachineOperand) -> bool {
    matches!(
        op,
        MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_)
    )
}

/// Create an [`EncodedInstruction`] with no relocation.
#[inline]
fn enc(bytes: Vec<u8>) -> EncodedInstruction {
    EncodedInstruction {
        bytes,
        relocation: None,
    }
}

/// Create an [`EncodedInstruction`] with a relocation.
#[inline]
fn enc_reloc(bytes: Vec<u8>, reloc: InstructionRelocation) -> EncodedInstruction {
    EncodedInstruction {
        bytes,
        relocation: Some(reloc),
    }
}

// ===========================================================================
// Memory Operand Encoding
// ===========================================================================

/// Encode a memory addressing mode into ModR/M + optional SIB + displacement.
///
/// `reg_field` goes into the ModR/M `reg` bits (register index or `/digit`
/// opcode extension).
///
/// # i686 Addressing-Mode Summary
///
/// | Mode                        | mod  | r/m  | Extra                  |
/// |-----------------------------|------|------|------------------------|
/// | `[reg]`                     | 00   | reg  | SIB if reg=ESP         |
/// | `[reg + disp8]`             | 01   | reg  | disp8 (+ SIB if ESP)   |
/// | `[reg + disp32]`            | 10   | reg  | disp32 (+ SIB if ESP)  |
/// | `[disp32]` (absolute)       | 00   | 101  | disp32                 |
/// | `[base + idx×sc]`           | 00   | 100  | SIB                    |
/// | `[base + idx×sc + disp8]`   | 01   | 100  | SIB + disp8            |
/// | `[base + idx×sc + disp32]`  | 10   | 100  | SIB + disp32           |
/// | `[idx×sc + disp32]`         | 00   | 100  | SIB(base=101) + disp32 |
fn encode_mem_raw(
    reg_field: u8,
    base: Option<u16>,
    index: Option<u16>,
    scale: u8,
    disp: i64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    let has_base = base.is_some();
    let has_index = index.is_some();

    // Case 1: displacement-only [disp32]
    if !has_base && !has_index {
        out.push(modrm(0b00, reg_field, 0b101));
        out.extend_from_slice(&emit_i32(disp));
        return out;
    }

    // Determine if SIB byte is needed
    let need_sib = has_index || (has_base && (base.unwrap() as u8) == ESP_IDX);

    if need_sib {
        let bv = base.map(|r| r as u8).unwrap_or(0b101);
        let iv = index.map(|r| r as u8).unwrap_or(0b100); // 100 = no index
        let sc = scale_to_sib(if has_index { scale } else { 0 });

        if !has_base {
            // [index×scale + disp32]
            out.push(modrm(0b00, reg_field, 0b100));
            out.push(sib_byte(sc, iv, 0b101));
            out.extend_from_slice(&emit_i32(disp));
        } else if disp == 0 && bv != EBP_IDX {
            out.push(modrm(0b00, reg_field, 0b100));
            out.push(sib_byte(sc, iv, bv));
        } else if fits_in_i8(disp) {
            out.push(modrm(0b01, reg_field, 0b100));
            out.push(sib_byte(sc, iv, bv));
            out.push(emit_i8(disp));
        } else {
            out.push(modrm(0b10, reg_field, 0b100));
            out.push(sib_byte(sc, iv, bv));
            out.extend_from_slice(&emit_i32(disp));
        }
        return out;
    }

    // Base only, no index, base ≠ ESP
    let br = base.unwrap() as u8;
    if br == EBP_IDX && disp == 0 {
        // [EBP] → mod=01, disp8=0  (mod=00 r/m=101 means [disp32])
        out.push(modrm(0b01, reg_field, br));
        out.push(0x00);
    } else if disp == 0 {
        out.push(modrm(0b00, reg_field, br));
    } else if fits_in_i8(disp) {
        out.push(modrm(0b01, reg_field, br));
        out.push(emit_i8(disp));
    } else {
        out.push(modrm(0b10, reg_field, br));
        out.extend_from_slice(&emit_i32(disp));
    }
    out
}

/// Encode a memory operand from a [`MachineOperand`].
///
/// For `GlobalSymbol` operands, encodes as `[disp32]` with a placeholder
/// displacement (the caller must produce a relocation to fill the actual
/// address at link time).
fn encode_mem(reg_field: u8, op: &MachineOperand) -> Result<Vec<u8>, String> {
    match op {
        MachineOperand::Memory {
            base,
            index,
            scale,
            displacement,
        } => Ok(encode_mem_raw(
            reg_field,
            *base,
            *index,
            *scale,
            *displacement,
        )),
        MachineOperand::FrameSlot(off) => Ok(encode_mem_raw(
            reg_field,
            Some(registers::EBP),
            None,
            1,
            *off as i64,
        )),
        MachineOperand::GlobalSymbol(_) => {
            // [disp32] addressing: mod=00, rm=101 (no base register)
            // The 4-byte displacement will be patched by a relocation.
            Ok(encode_mem_raw(reg_field, None, None, 1, 0))
        }
        _ => Err(format!("Expected memory operand, got {}", op)),
    }
}

// ===========================================================================
// Condition Code Mapping
// ===========================================================================

/// Map a [`CondCode`](codegen::CondCode) to the x86 4-bit `tttn` byte.
///
/// Used by `Jcc` (`0x0F 0x80+cc`), `SETcc` (`0x0F 0x90+cc`), and
/// `CMOVcc` (`0x0F 0x40+cc`) encodings.
pub fn condition_code_to_byte(cc: &codegen::CondCode) -> u8 {
    match cc {
        codegen::CondCode::Equal => 0x04,
        codegen::CondCode::NotEqual => 0x05,
        codegen::CondCode::Less => 0x0C,
        codegen::CondCode::LessEqual => 0x0E,
        codegen::CondCode::Greater => 0x0F,
        codegen::CondCode::GreaterEqual => 0x0D,
        codegen::CondCode::Below => 0x02,
        codegen::CondCode::BelowEqual => 0x06,
        codegen::CondCode::Above => 0x07,
        codegen::CondCode::AboveEqual => 0x03,
    }
}

/// Extract a condition code byte (0x00..0x0F) from an Immediate operand.
fn extract_cc(instr: &MachineInstruction) -> Result<u8, String> {
    for op in &instr.operands {
        if let MachineOperand::Immediate(v) = op {
            let b = *v as u8;
            if b <= 0x0F {
                return Ok(b);
            }
        }
    }
    Err("No condition code (Immediate 0x00..0x0F) in operands".into())
}

/// Find the branch/call target operand (BlockLabel or GlobalSymbol).
fn find_target(instr: &MachineInstruction) -> Option<&MachineOperand> {
    instr.operands.iter().find(|op| {
        matches!(
            op,
            MachineOperand::BlockLabel(_) | MachineOperand::GlobalSymbol(_)
        )
    })
}

/// Convert a block label ID to its label-offset key string.
#[inline]
fn label_key(id: u32) -> String {
    format!(".L{}", id)
}

// ===========================================================================
// Main Dispatch — encode_instruction
// ===========================================================================

/// Encode a single [`MachineInstruction`] into raw machine code bytes.
///
/// # Parameters
/// - `instr`:          The instruction to encode.
/// - `label_offsets`:  Label name → byte offset map (from pass 1).
/// - `current_offset`: Byte position of this instruction in the output.
pub fn encode_instruction(
    instr: &MachineInstruction,
    label_offsets: &FxHashMap<String, usize>,
    current_offset: usize,
) -> Result<EncodedInstruction, String> {
    match instr.opcode {
        // Data movement
        codegen::I686_MOV => encode_mov(instr),
        codegen::I686_MOVZX => encode_movzx(instr),
        codegen::I686_MOVSX => encode_movsx(instr),
        codegen::I686_LEA => encode_lea(instr),
        codegen::I686_PUSH => encode_push(instr),
        codegen::I686_POP => encode_pop(instr),
        codegen::I686_XCHG => encode_xchg(instr),
        // ALU (shared pattern)
        codegen::I686_ADD => encode_alu(instr, 0),
        codegen::I686_OR => encode_alu(instr, 1),
        codegen::I686_ADC => encode_alu(instr, 2),
        codegen::I686_SBB => encode_alu(instr, 3),
        codegen::I686_AND => encode_alu(instr, 4),
        codegen::I686_SUB => encode_alu(instr, 5),
        codegen::I686_XOR => encode_alu(instr, 6),
        codegen::I686_CMP => encode_alu(instr, 7),
        // Multiply / Divide / Unary
        codegen::I686_IMUL => encode_imul(instr),
        codegen::I686_MUL => encode_unary_f7(instr, 4),
        codegen::I686_DIV => encode_unary_f7(instr, 6),
        codegen::I686_IDIV => encode_unary_f7(instr, 7),
        codegen::I686_NEG => encode_unary_f7(instr, 3),
        codegen::I686_NOT => encode_unary_f7(instr, 2),
        codegen::I686_INC => encode_inc(instr),
        codegen::I686_DEC => encode_dec(instr),
        codegen::I686_CDQ => Ok(enc(vec![0x99])),
        // Shifts
        codegen::I686_SHL => encode_shift(instr, 4),
        codegen::I686_SHR => encode_shift(instr, 5),
        codegen::I686_SAR => encode_shift(instr, 7),
        // Test
        codegen::I686_TEST => encode_test(instr),
        // Conditional set / move
        codegen::I686_SETCC => encode_setcc(instr),
        codegen::I686_CMOVCC => encode_cmovcc(instr),
        // Control flow
        codegen::I686_JMP => encode_jmp(instr, label_offsets, current_offset),
        codegen::I686_JCC => encode_jcc(instr, label_offsets, current_offset),
        codegen::I686_CALL => encode_call(instr, label_offsets, current_offset),
        codegen::I686_RET => encode_ret(instr),
        // x87 FPU
        codegen::I686_FLD => encode_fld(instr),
        codegen::I686_FSTP => encode_fstp(instr),
        codegen::I686_FADD => encode_fpu_arith(instr, 0xC0, 0, 0),
        codegen::I686_FSUB => encode_fpu_arith(instr, 0xE0, 4, 4),
        codegen::I686_FMUL => encode_fpu_arith(instr, 0xC8, 1, 1),
        codegen::I686_FDIV => encode_fpu_arith(instr, 0xF0, 6, 6),
        codegen::I686_FCHS => Ok(enc(vec![0xD9, 0xE0])),
        codegen::I686_FCOMP => encode_fcomp(instr),
        codegen::I686_FILD => encode_fild(instr),
        codegen::I686_FISTP => encode_fistp(instr),
        codegen::I686_FXCH => encode_fxch(instr),
        codegen::I686_FUCOMIP => encode_fucomip(instr),
        // Stack frame
        codegen::I686_ENTER => encode_enter(instr),
        codegen::I686_LEAVE => Ok(enc(vec![0xC9])),
        // Miscellaneous
        codegen::I686_NOP => Ok(enc(vec![0x90])),
        codegen::I686_INT3 => Ok(enc(vec![0xCC])),
        codegen::I686_UD2 => Ok(enc(vec![0x0F, 0x0B])),
        // Global variable load/store with absolute addressing
        codegen::I686_MOV_LOAD_GLOBAL => encode_mov_load_global(instr),
        codegen::I686_MOV_STORE_GLOBAL => encode_mov_store_global(instr),
        // Register-indirect load/store (pointer in a register)
        codegen::I686_MOV_LOAD_INDIRECT => encode_mov_load_indirect(instr),
        codegen::I686_MOV_STORE_INDIRECT => encode_mov_store_indirect(instr),
        // Byte/word-sized register-indirect load (MOVZX from memory)
        codegen::I686_MOVZX_LOAD_INDIRECT_BYTE => encode_movzx_load_indirect(instr, false),
        codegen::I686_MOVZX_LOAD_INDIRECT_WORD => encode_movzx_load_indirect(instr, true),
        // Byte/word-sized register-indirect store
        codegen::I686_MOV_STORE_INDIRECT_BYTE => encode_mov_store_indirect_byte(instr),
        codegen::I686_MOV_STORE_INDIRECT_WORD => encode_mov_store_indirect_word(instr),
        // BSWAP — byte-swap a 32-bit register (0F C8+rd)
        codegen::I686_BSWAP => encode_bswap(instr),
        _ => Err(format!("i686 encoder: unknown opcode 0x{:X}", instr.opcode)),
    }
}

// ===========================================================================
// Data Movement Encoders
// ===========================================================================

/// Encode `MOV` — all forms.
fn encode_mov(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let mut bytes = Vec::new();
    let mut reloc: Option<InstructionRelocation> = None;

    if let Some(ref res) = instr.result {
        let d = gpr(res)?;
        if instr.operands.is_empty() {
            return Err("MOV: missing source operand".into());
        }
        match &instr.operands[0] {
            MachineOperand::Register(s) => {
                // MOV r32, r32 → 0x8B ModR/M(11,dst,src)
                bytes.push(0x8B);
                bytes.push(modrm(0b11, d, *s as u8));
            }
            MachineOperand::Immediate(v) => {
                // MOV r32, imm32 → 0xB8+rd imm32
                bytes.push(0xB8 + d);
                bytes.extend_from_slice(&emit_i32(*v));
            }
            op if is_mem(op) => {
                // MOV r32, [mem] → 0x8B ModR/M
                bytes.push(0x8B);
                bytes.extend(encode_mem(d, op)?);
            }
            MachineOperand::GlobalSymbol(name) => {
                // MOV r32, &symbol → 0xB8+rd + R_386_32
                // Load the address of a symbol into a register.
                bytes.push(0xB8 + d);
                let off = bytes.len();
                bytes.extend_from_slice(&[0; 4]);
                reloc = Some(InstructionRelocation {
                    offset_in_instruction: off,
                    symbol: name.clone(),
                    rel_type: relocations::R_386_32,
                    addend: 0,
                });
            }
            other => return Err(format!("MOV: unsupported source {}", other)),
        }
    } else {
        // Store form: operands = [dst_mem, src]
        if instr.operands.len() < 2 {
            return Err("MOV store: need [dst_mem, src]".into());
        }
        let raw_dst = &instr.operands[0];
        let src = &instr.operands[1];
        // If the destination is a bare Register, treat it as an indirect
        // store through that register: [reg] with zero displacement.
        // This happens when a GEP result (pointer in a register) is used
        // as the Store destination after register allocation.
        let reg_as_mem;
        let dst = if let MachineOperand::Register(r) = raw_dst {
            reg_as_mem = MachineOperand::Memory {
                base: Some(*r),
                index: None,
                scale: 1,
                displacement: 0,
            };
            &reg_as_mem
        } else if is_mem(raw_dst) {
            raw_dst
        } else {
            return Err(format!("MOV store: dst must be memory, got {}", raw_dst));
        };
        // For GlobalSymbol destinations, emit relocation for the address
        let dst_is_global = matches!(dst, MachineOperand::GlobalSymbol(_));
        match src {
            MachineOperand::Register(s) => {
                // MOV [mem], r32 → 0x89 ModR/M
                bytes.push(0x89);
                bytes.extend(encode_mem(*s as u8, dst)?);
                if let MachineOperand::GlobalSymbol(name) = dst {
                    // Patch disp32 at offset 2 (opcode + ModR/M=05 + disp32)
                    reloc = Some(InstructionRelocation {
                        offset_in_instruction: 2,
                        symbol: name.clone(),
                        rel_type: relocations::R_386_32,
                        addend: 0,
                    });
                }
            }
            MachineOperand::Immediate(v) => {
                // MOV [mem], imm32 → 0xC7 /0 ModR/M imm32
                bytes.push(0xC7);
                bytes.extend(encode_mem(0, dst)?);
                if dst_is_global {
                    if let MachineOperand::GlobalSymbol(name) = dst {
                        reloc = Some(InstructionRelocation {
                            offset_in_instruction: 2,
                            symbol: name.clone(),
                            rel_type: relocations::R_386_32,
                            addend: 0,
                        });
                    }
                }
                bytes.extend_from_slice(&emit_i32(*v));
            }
            MachineOperand::GlobalSymbol(name) => {
                bytes.push(0xC7);
                bytes.extend(encode_mem(0, dst)?);
                if let MachineOperand::GlobalSymbol(dst_name) = dst {
                    reloc = Some(InstructionRelocation {
                        offset_in_instruction: 2,
                        symbol: dst_name.clone(),
                        rel_type: relocations::R_386_32,
                        addend: 0,
                    });
                }
                let off = bytes.len();
                bytes.extend_from_slice(&[0; 4]);
                // Second relocation for the source global symbol is needed
                // but we only have one slot — use the destination relocation
                // since it's more critical for correctness.
                if !dst_is_global {
                    reloc = Some(InstructionRelocation {
                        offset_in_instruction: off,
                        symbol: name.clone(),
                        rel_type: relocations::R_386_32,
                        addend: 0,
                    });
                }
            }
            other => return Err(format!("MOV store: unsupported src {}", other)),
        }
    }
    Ok(EncodedInstruction {
        bytes,
        relocation: reloc,
    })
}

/// Encode `MOV r32, [global]` — load from absolute address of a global symbol.
/// `result = Register(dst)`, `operands = [GlobalSymbol(name)]`.
/// Produces: `0x8B ModR/M(00, dst, 101) disp32=0` + R_386_32 relocation.
fn encode_mov_load_global(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let d = gpr(instr.result.as_ref().ok_or("MOV_LOAD_GLOBAL: no result")?)?;
    if instr.operands.is_empty() {
        return Err("MOV_LOAD_GLOBAL: no operand".into());
    }
    let name = match &instr.operands[0] {
        MachineOperand::GlobalSymbol(n) => n.clone(),
        other => {
            return Err(format!(
                "MOV_LOAD_GLOBAL: expected GlobalSymbol, got {}",
                other
            ))
        }
    };
    let mut bytes = Vec::new();
    // MOV r32, [disp32] → 0x8B ModR/M(00, reg, 101=disp32)
    bytes.push(0x8B);
    // mod=00, reg=d, rm=101 (absolute disp32 on i686)
    bytes.push(modrm(0b00, d, 0b101));
    let off = bytes.len();
    bytes.extend_from_slice(&[0; 4]); // placeholder for disp32
    Ok(EncodedInstruction {
        bytes,
        relocation: Some(InstructionRelocation {
            offset_in_instruction: off,
            symbol: name,
            rel_type: relocations::R_386_32,
            addend: 0,
        }),
    })
}

/// Encode `MOV [global], r32` — store to absolute address of a global symbol.
/// `operands = [GlobalSymbol(name), Register(src)]`.
/// Produces: `0x89 ModR/M(00, src, 101) disp32=0` + R_386_32 relocation.
fn encode_mov_store_global(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.len() < 2 {
        return Err("MOV_STORE_GLOBAL: need [GlobalSymbol, Register]".into());
    }
    let name = match &instr.operands[0] {
        MachineOperand::GlobalSymbol(n) => n.clone(),
        other => {
            return Err(format!(
                "MOV_STORE_GLOBAL: expected GlobalSymbol as dst, got {}",
                other
            ))
        }
    };
    let s = match &instr.operands[1] {
        MachineOperand::Register(r) => *r as u8,
        other => {
            return Err(format!(
                "MOV_STORE_GLOBAL: expected Register as src, got {}",
                other
            ))
        }
    };
    let mut bytes = Vec::new();
    // MOV [disp32], r32 → 0x89 ModR/M(00, src, 101)
    bytes.push(0x89);
    bytes.push(modrm(0b00, s, 0b101));
    let off = bytes.len();
    bytes.extend_from_slice(&[0; 4]); // placeholder for disp32
    Ok(EncodedInstruction {
        bytes,
        relocation: Some(InstructionRelocation {
            offset_in_instruction: off,
            symbol: name,
            rel_type: relocations::R_386_32,
            addend: 0,
        }),
    })
}

/// Encode `MOV_LOAD_INDIRECT` — load value through a pointer held in a
/// register: `MOV r32, [reg]`.
///
/// Layout: `result = dst_register, operands = [src_pointer_register]`.
/// After register allocation, `operands[0]` is a physical `Register`
/// holding the memory address to dereference.
fn encode_mov_load_indirect(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let d = gpr(instr
        .result
        .as_ref()
        .ok_or("MOV_LOAD_INDIRECT: no result")?)?;
    if instr.operands.is_empty() {
        return Err("MOV_LOAD_INDIRECT: need source pointer register".into());
    }
    let base_reg = match &instr.operands[0] {
        MachineOperand::Register(r) => *r as u8,
        op if is_mem(op) => {
            // Already a memory operand (e.g. spill slot) — just encode normally.
            let mut b = vec![0x8B];
            b.extend(encode_mem(d, op)?);
            return Ok(enc(b));
        }
        other => {
            return Err(format!(
                "MOV_LOAD_INDIRECT: expected Register or Memory, got {}",
                other
            ))
        }
    };
    // Encode MOV r32, [reg] using indirect addressing.
    // Special cases for x86 addressing:
    //   - [EBP] (r5) needs ModR/M mode 01 with disp8=0 (mode 00 + r/m=101 = disp32)
    //   - [ESP] (r4) needs a SIB byte (r/m=100 encodes SIB, not [ESP])
    let mut b = vec![0x8B]; // MOV r32, r/m32
    if base_reg == 5 {
        // [EBP] → ModR/M(01, dst, 101) + disp8(0)
        b.push(modrm(0b01, d, 5));
        b.push(0x00);
    } else if base_reg == 4 {
        // [ESP] → ModR/M(00, dst, 100) + SIB(00, 100, 100)
        b.push(modrm(0b00, d, 4));
        b.push(sib_byte(0, 4, 4)); // scale=1, index=none(ESP), base=ESP
    } else {
        // [reg] → ModR/M(00, dst, base)
        b.push(modrm(0b00, d, base_reg));
    }
    Ok(enc(b))
}

/// Encode `MOVZX_LOAD_INDIRECT_BYTE` / `MOVZX_LOAD_INDIRECT_WORD` — load a
/// byte or word through a register pointer with zero extension to 32 bits.
///
/// Byte variant:  `MOVZX r32, BYTE PTR [reg]`  (0x0F 0xB6 + ModR/M)
/// Word variant:  `MOVZX r32, WORD PTR [reg]`  (0x0F 0xB7 + ModR/M)
///
/// This avoids the i686 8-bit register alias problem: a 32-bit
/// `MOV_LOAD_INDIRECT` followed by a register-to-register `MOVZX` fails
/// when the source register is ESI, EDI, EBP, or ESP because those
/// registers have no 8-bit aliases on i686 (register codes 4–7 in 8-bit
/// mode map to AH, CH, DH, BH — the HIGH bytes, not low bytes).
///
/// Layout: `result = dst, operands = [src_pointer_register]`.
fn encode_movzx_load_indirect(
    instr: &MachineInstruction,
    word: bool,
) -> Result<EncodedInstruction, String> {
    let d = gpr(instr
        .result
        .as_ref()
        .ok_or("MOVZX_LOAD_INDIRECT: no result")?)?;
    if instr.operands.is_empty() {
        return Err("MOVZX_LOAD_INDIRECT: need source pointer register".into());
    }
    let base_reg = match &instr.operands[0] {
        MachineOperand::Register(r) => *r as u8,
        op if is_mem(op) => {
            let op2: u8 = if word { 0xB7 } else { 0xB6 };
            let mut b = vec![0x0F, op2];
            b.extend(encode_mem(d, op)?);
            return Ok(enc(b));
        }
        other => {
            return Err(format!(
                "MOVZX_LOAD_INDIRECT: expected Register or Memory, got {}",
                other
            ))
        }
    };
    let op2: u8 = if word { 0xB7 } else { 0xB6 };
    let mut b = vec![0x0F, op2];
    if base_reg == 5 {
        // [EBP] → ModR/M(01, dst, 101) + disp8(0)
        b.push(modrm(0b01, d, 5));
        b.push(0x00);
    } else if base_reg == 4 {
        // [ESP] → ModR/M(00, dst, 100) + SIB(00, 100, 100)
        b.push(modrm(0b00, d, 4));
        b.push(sib_byte(0, 4, 4));
    } else {
        // [reg] → ModR/M(00, dst, base)
        b.push(modrm(0b00, d, base_reg));
    }
    Ok(enc(b))
}

/// Encode `MOV_STORE_INDIRECT_BYTE` — store a byte through a register
/// pointer: `MOV BYTE PTR [reg], r8` (0x88 + ModR/M).
///
/// Layout: `operands = [dst_pointer_register, src_value]`.
fn encode_mov_store_indirect_byte(
    instr: &MachineInstruction,
) -> Result<EncodedInstruction, String> {
    if instr.operands.len() < 2 {
        return Err("MOV_STORE_INDIRECT_BYTE: need [ptr, src]".into());
    }
    let base_reg = match &instr.operands[0] {
        MachineOperand::Register(r) => *r as u8,
        op if is_mem(op) => {
            let s = match &instr.operands[1] {
                MachineOperand::Register(r) => *r as u8,
                MachineOperand::Immediate(v) => {
                    // MOV BYTE PTR [mem], imm8 → 0xC6 /0 + mem + imm8
                    let mut b = vec![0xC6];
                    b.extend(encode_mem(0, op)?);
                    b.push(*v as u8);
                    return Ok(enc(b));
                }
                other => return Err(format!("MOV_STORE_INDIRECT_BYTE: bad src {}", other)),
            };
            // 0x88 /r: MOV r/m8, r8
            let mut b = vec![0x88];
            b.extend(encode_mem(s, op)?);
            return Ok(enc(b));
        }
        other => {
            return Err(format!(
                "MOV_STORE_INDIRECT_BYTE: expected Register, got {}",
                other
            ))
        }
    };
    let src_reg = match &instr.operands[1] {
        MachineOperand::Register(r) => *r as u8,
        MachineOperand::Immediate(v) => {
            // MOV BYTE PTR [reg], imm8 → 0xC6 + ModR/M + imm8
            let mut b = vec![0xC6];
            if base_reg == 5 {
                b.push(modrm(0b01, 0, 5));
                b.push(0x00);
            } else if base_reg == 4 {
                b.push(modrm(0b00, 0, 4));
                b.push(sib_byte(0, 4, 4));
            } else {
                b.push(modrm(0b00, 0, base_reg));
            }
            b.push(*v as u8);
            return Ok(enc(b));
        }
        other => return Err(format!("MOV_STORE_INDIRECT_BYTE: bad src {}", other)),
    };
    // 0x88 /r: MOV BYTE PTR [base_reg], r8
    let mut b = vec![0x88];
    if base_reg == 5 {
        b.push(modrm(0b01, src_reg, 5));
        b.push(0x00);
    } else if base_reg == 4 {
        b.push(modrm(0b00, src_reg, 4));
        b.push(sib_byte(0, 4, 4));
    } else {
        b.push(modrm(0b00, src_reg, base_reg));
    }
    Ok(enc(b))
}

/// Encode `MOV_STORE_INDIRECT_WORD` — store a word through a register
/// pointer: `MOV WORD PTR [reg], r16` (0x66 0x89 + ModR/M).
///
/// Layout: `operands = [dst_pointer_register, src_value]`.
fn encode_mov_store_indirect_word(
    instr: &MachineInstruction,
) -> Result<EncodedInstruction, String> {
    if instr.operands.len() < 2 {
        return Err("MOV_STORE_INDIRECT_WORD: need [ptr, src]".into());
    }
    let base_reg = match &instr.operands[0] {
        MachineOperand::Register(r) => *r as u8,
        op if is_mem(op) => {
            let s = match &instr.operands[1] {
                MachineOperand::Register(r) => *r as u8,
                MachineOperand::Immediate(v) => {
                    // 0x66 0xC7 /0 + mem + imm16
                    let mut b = vec![0x66, 0xC7];
                    b.extend(encode_mem(0, op)?);
                    b.extend_from_slice(&(*v as u16).to_le_bytes());
                    return Ok(enc(b));
                }
                other => return Err(format!("MOV_STORE_INDIRECT_WORD: bad src {}", other)),
            };
            let mut b = vec![0x66, 0x89];
            b.extend(encode_mem(s, op)?);
            return Ok(enc(b));
        }
        other => {
            return Err(format!(
                "MOV_STORE_INDIRECT_WORD: expected Register, got {}",
                other
            ))
        }
    };
    let src_reg = match &instr.operands[1] {
        MachineOperand::Register(r) => *r as u8,
        MachineOperand::Immediate(v) => {
            let mut b = vec![0x66, 0xC7];
            if base_reg == 5 {
                b.push(modrm(0b01, 0, 5));
                b.push(0x00);
            } else if base_reg == 4 {
                b.push(modrm(0b00, 0, 4));
                b.push(sib_byte(0, 4, 4));
            } else {
                b.push(modrm(0b00, 0, base_reg));
            }
            b.extend_from_slice(&(*v as u16).to_le_bytes());
            return Ok(enc(b));
        }
        other => return Err(format!("MOV_STORE_INDIRECT_WORD: bad src {}", other)),
    };
    let mut b = vec![0x66, 0x89];
    if base_reg == 5 {
        b.push(modrm(0b01, src_reg, 5));
        b.push(0x00);
    } else if base_reg == 4 {
        b.push(modrm(0b00, src_reg, 4));
        b.push(sib_byte(0, 4, 4));
    } else {
        b.push(modrm(0b00, src_reg, base_reg));
    }
    Ok(enc(b))
}

/// Encode `MOV_STORE_INDIRECT` — store a value through a pointer held in a
/// register: `MOV [reg], src`.
///
/// Layout: `operands = [dst_pointer_register, src_value]`.
/// After register allocation, `operands[0]` is a physical `Register`
/// holding the memory address to write to.
fn encode_mov_store_indirect(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.len() < 2 {
        return Err("MOV_STORE_INDIRECT: need [ptr_register, src_value]".into());
    }
    let base_reg = match &instr.operands[0] {
        MachineOperand::Register(r) => *r as u8,
        op if is_mem(op) => {
            // Already a memory operand — encode as normal MOV store.
            let s = match &instr.operands[1] {
                MachineOperand::Register(r) => *r as u8,
                MachineOperand::Immediate(v) => {
                    // MOV [mem], imm32 → 0xC7 /0 + mem + imm32
                    let mut b = vec![0xC7];
                    b.extend(encode_mem(0, op)?);
                    b.extend_from_slice(&emit_i32(*v));
                    return Ok(enc(b));
                }
                other => {
                    return Err(format!(
                        "MOV_STORE_INDIRECT: expected Register/Imm as src, got {}",
                        other
                    ))
                }
            };
            let mut b = vec![0x89];
            b.extend(encode_mem(s, op)?);
            return Ok(enc(b));
        }
        other => {
            return Err(format!(
                "MOV_STORE_INDIRECT: expected Register or Memory as dst ptr, got {}",
                other
            ))
        }
    };
    match &instr.operands[1] {
        MachineOperand::Register(s) => {
            let src = *s as u8;
            // MOV [reg], r32 → 0x89 ModR/M(mode, src, base)
            let mut b = vec![0x89];
            if base_reg == 5 {
                // [EBP] → mode 01 + disp8(0)
                b.push(modrm(0b01, src, 5));
                b.push(0x00);
            } else if base_reg == 4 {
                // [ESP] → mode 00, r/m=100 + SIB
                b.push(modrm(0b00, src, 4));
                b.push(sib_byte(0, 4, 4));
            } else {
                b.push(modrm(0b00, src, base_reg));
            }
            Ok(enc(b))
        }
        MachineOperand::Immediate(v) => {
            // MOV [reg], imm32 → 0xC7 /0 + indirect addressing + imm32
            let mut b = vec![0xC7];
            if base_reg == 5 {
                b.push(modrm(0b01, 0, 5));
                b.push(0x00);
            } else if base_reg == 4 {
                b.push(modrm(0b00, 0, 4));
                b.push(sib_byte(0, 4, 4));
            } else {
                b.push(modrm(0b00, 0, base_reg));
            }
            b.extend_from_slice(&emit_i32(*v));
            Ok(enc(b))
        }
        other => Err(format!(
            "MOV_STORE_INDIRECT: unsupported src operand {}",
            other
        )),
    }
}

/// Encode `BSWAP r32` — 0F C8+rd.
fn encode_bswap(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    // The result register IS the operand (in-place swap).
    let rd = gpr(instr.result.as_ref().ok_or("BSWAP: no result")?)?;
    Ok(enc(vec![0x0F, 0xC8 + rd]))
}

/// Encode `MOVZX r32, r/m8` or `MOVZX r32, r/m16`.
fn encode_movzx(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let d = gpr(instr.result.as_ref().ok_or("MOVZX: no result")?)?;
    if instr.operands.is_empty() {
        return Err("MOVZX: no source".into());
    }
    let w: u8 = if instr.operands.len() > 1 {
        imm(&instr.operands[1]).unwrap_or(8) as u8
    } else {
        8
    };
    // On i686, the register-to-register form of `MOVZX r32, r8` only
    // works correctly for registers 0–3 (AL, CL, DL, BL). Register
    // codes 4–7 in 8-bit mode map to AH, CH, DH, BH (the HIGH bytes)
    // rather than the low bytes of ESP–EDI, because those 8-bit aliases
    // do not exist on i686.
    //
    // When the source register is >= 4 and we need an 8-bit zero-extend,
    // fall back to: MOV dst, src; AND dst, 0xFF (or 0xFFFF for 16-bit).
    if w != 16 {
        if let MachineOperand::Register(s) = &instr.operands[0] {
            let sr = *s as u8;
            if sr >= 4 {
                // MOV dst, src (32-bit copy)
                let mut b = vec![0x8B, modrm(0b11, d, sr)];
                // AND dst, 0xFF
                if d == 0 {
                    // Shorter form: AND EAX, imm32 (0x25)
                    b.extend_from_slice(&[0x25]);
                    b.extend_from_slice(&0xFFu32.to_le_bytes());
                } else {
                    // AND r32, imm32 (0x81 /4)
                    b.extend_from_slice(&[0x81, modrm(0b11, 4, d)]);
                    b.extend_from_slice(&0xFFu32.to_le_bytes());
                }
                return Ok(enc(b));
            }
        }
    }
    let op2 = if w == 16 { 0xB7u8 } else { 0xB6u8 };
    let mut b = vec![0x0F, op2];
    match &instr.operands[0] {
        MachineOperand::Register(s) => b.push(modrm(0b11, d, *s as u8)),
        op if is_mem(op) => b.extend(encode_mem(d, op)?),
        o => return Err(format!("MOVZX: bad source {}", o)),
    }
    Ok(enc(b))
}

/// Encode `MOVSX r32, r/m8` or `MOVSX r32, r/m16`.
fn encode_movsx(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let d = gpr(instr.result.as_ref().ok_or("MOVSX: no result")?)?;
    if instr.operands.is_empty() {
        return Err("MOVSX: no source".into());
    }
    let w: u8 = if instr.operands.len() > 1 {
        imm(&instr.operands[1]).unwrap_or(8) as u8
    } else {
        8
    };
    // Same i686 8-bit register alias issue as encode_movzx: register
    // codes 4–7 in 8-bit mode map to AH/CH/DH/BH not the low bytes
    // of ESP–EDI.  Fall back to MOV + shift-based sign extension.
    if w != 16 {
        if let MachineOperand::Register(s) = &instr.operands[0] {
            let sr = *s as u8;
            if sr >= 4 {
                // MOV dst, src; SHL dst, 24; SAR dst, 24
                let mut b = vec![0x8B, modrm(0b11, d, sr)];
                // SHL dst, 24 (0xC1 /4 ib)
                b.extend_from_slice(&[0xC1, modrm(0b11, 4, d), 24]);
                // SAR dst, 24 (0xC1 /7 ib)
                b.extend_from_slice(&[0xC1, modrm(0b11, 7, d), 24]);
                return Ok(enc(b));
            }
        }
    }
    let op2 = if w == 16 { 0xBFu8 } else { 0xBEu8 };
    let mut b = vec![0x0F, op2];
    match &instr.operands[0] {
        MachineOperand::Register(s) => b.push(modrm(0b11, d, *s as u8)),
        op if is_mem(op) => b.extend(encode_mem(d, op)?),
        o => return Err(format!("MOVSX: bad source {}", o)),
    }
    Ok(enc(b))
}

/// Encode `LEA r32, [mem]`.
fn encode_lea(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let d = gpr(instr.result.as_ref().ok_or("LEA: no result")?)?;
    if instr.operands.is_empty() {
        return Err("LEA: no memory operand".into());
    }
    let op = &instr.operands[0];
    if let MachineOperand::GlobalSymbol(name) = op {
        // LEA r32, [symbol] → 0x8D ModRM(00,d,101) disp32 + R_386_32
        let mut b = vec![0x8D];
        b.extend(encode_mem_raw(d, None, None, 1, 0));
        let off = b.len() - 4; // disp32 is the last 4 bytes
        return Ok(enc_reloc(
            b,
            InstructionRelocation {
                offset_in_instruction: off,
                symbol: name.clone(),
                rel_type: relocations::R_386_32,
                addend: 0,
            },
        ));
    }
    let mut b = vec![0x8D];
    b.extend(encode_mem(d, op)?);
    Ok(enc(b))
}

/// Encode `PUSH` (register, immediate, memory, or global symbol).
fn encode_push(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("PUSH: no operand".into());
    }
    let mut b = Vec::new();
    match &instr.operands[0] {
        MachineOperand::Register(r) => b.push(0x50 + *r as u8),
        MachineOperand::Immediate(v) => {
            if fits_in_i8(*v) {
                b.push(0x6A);
                b.push(emit_i8(*v));
            } else {
                b.push(0x68);
                b.extend_from_slice(&emit_i32(*v));
            }
        }
        op if is_mem(op) => {
            b.push(0xFF);
            b.extend(encode_mem(6, op)?);
        }
        MachineOperand::GlobalSymbol(name) => {
            b.push(0x68);
            let off = b.len();
            b.extend_from_slice(&[0; 4]);
            return Ok(enc_reloc(
                b,
                InstructionRelocation {
                    offset_in_instruction: off,
                    symbol: name.clone(),
                    rel_type: relocations::R_386_32,
                    addend: 0,
                },
            ));
        }
        o => return Err(format!("PUSH: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `POP` (register or memory).
fn encode_pop(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let op = instr
        .result
        .as_ref()
        .or_else(|| instr.operands.first())
        .ok_or("POP: no destination")?;
    let mut b = Vec::new();
    match op {
        MachineOperand::Register(r) => b.push(0x58 + *r as u8),
        m if is_mem(m) => {
            b.push(0x8F);
            b.extend(encode_mem(0, m)?);
        }
        o => return Err(format!("POP: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `XCHG` (reg-reg or reg-mem).
fn encode_xchg(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.len() < 2 {
        return Err("XCHG: need 2 operands".into());
    }
    let mut b = Vec::new();
    match (&instr.operands[0], &instr.operands[1]) {
        (MachineOperand::Register(a), MachineOperand::Register(bb)) => {
            let (a, bb) = (*a as u8, *bb as u8);
            if a == EAX_IDX {
                b.push(0x90 + bb);
            } else if bb == EAX_IDX {
                b.push(0x90 + a);
            } else {
                b.push(0x87);
                b.push(modrm(0b11, a, bb));
            }
        }
        (MachineOperand::Register(r), m) if is_mem(m) => {
            b.push(0x87);
            b.extend(encode_mem(*r as u8, m)?);
        }
        (m, MachineOperand::Register(r)) if is_mem(m) => {
            b.push(0x87);
            b.extend(encode_mem(*r as u8, m)?);
        }
        _ => return Err("XCHG: unsupported operand combination".into()),
    }
    Ok(enc(b))
}

// ===========================================================================
// ALU Instruction Encoders (shared pattern)
// ===========================================================================

/// Encode an ALU-group instruction (ADD, OR, ADC, SBB, AND, SUB, XOR, CMP).
///
/// All 8 ALU ops share the same encoding pattern, differing only in the
/// opcode extension digit `ext` (0..7):
///
/// | ext | Instruction | mr     | rm     | ax,imm | /ext  |
/// |-----|-------------|--------|--------|--------|-------|
/// |  0  | ADD         | 0x01   | 0x03   | 0x05   | /0    |
/// |  1  | OR          | 0x09   | 0x0B   | 0x0D   | /1    |
/// |  2  | ADC         | 0x11   | 0x13   | 0x15   | /2    |
/// |  3  | SBB         | 0x19   | 0x1B   | 0x1D   | /3    |
/// |  4  | AND         | 0x21   | 0x23   | 0x25   | /4    |
/// |  5  | SUB         | 0x29   | 0x2B   | 0x2D   | /5    |
/// |  6  | XOR         | 0x31   | 0x33   | 0x35   | /6    |
/// |  7  | CMP         | 0x39   | 0x3B   | 0x3D   | /7    |
fn encode_alu(instr: &MachineInstruction, ext: u8) -> Result<EncodedInstruction, String> {
    let mr = ext * 8 + 0x01; // r/m32, r32   (reg field = source)
    let rm = ext * 8 + 0x03; // r32,   r/m32 (reg field = dest)
    let axi = ext * 8 + 0x05; // EAX,   imm32

    if instr.operands.len() < 2 {
        return Err(format!("ALU ext={}: need 2 operands", ext));
    }
    let dst = &instr.operands[0];
    let src = &instr.operands[1];
    let mut b = Vec::new();

    match (dst, src) {
        // reg, reg
        (MachineOperand::Register(d), MachineOperand::Register(s)) => {
            b.push(mr);
            b.push(modrm(0b11, *s as u8, *d as u8));
        }
        // reg, imm
        (MachineOperand::Register(d), MachineOperand::Immediate(v)) => {
            let d = *d as u8;
            let v = *v;
            if d == EAX_IDX && !fits_in_i8(v) {
                b.push(axi);
                b.extend_from_slice(&emit_i32(v));
            } else if fits_in_i8(v) {
                b.push(0x83);
                b.push(modrm(0b11, ext, d));
                b.push(emit_i8(v));
            } else {
                b.push(0x81);
                b.push(modrm(0b11, ext, d));
                b.extend_from_slice(&emit_i32(v));
            }
        }
        // reg, [mem]
        (MachineOperand::Register(d), m) if is_mem(m) => {
            b.push(rm);
            b.extend(encode_mem(*d as u8, m)?);
        }
        // [mem], reg
        (m, MachineOperand::Register(s)) if is_mem(m) => {
            b.push(mr);
            b.extend(encode_mem(*s as u8, m)?);
        }
        // [mem], imm
        (m, MachineOperand::Immediate(v)) if is_mem(m) => {
            let v = *v;
            if fits_in_i8(v) {
                b.push(0x83);
                b.extend(encode_mem(ext, m)?);
                b.push(emit_i8(v));
            } else {
                b.push(0x81);
                b.extend(encode_mem(ext, m)?);
                b.extend_from_slice(&emit_i32(v));
            }
        }
        _ => {
            return Err(format!(
                "ALU ext={}: unsupported operands {} , {}",
                ext, dst, src
            ))
        }
    }
    Ok(enc(b))
}

// ===========================================================================
// Multiply / Divide / Unary Encoders
// ===========================================================================

/// Encode `IMUL` — 1, 2, or 3 operand forms.
fn encode_imul(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let ops = &instr.operands;
    let mut b = Vec::new();

    if ops.is_empty() {
        return Err("IMUL: no operands".into());
    }

    // Check if the last operand is an immediate → 3-operand form
    let last_is_imm = matches!(ops.last(), Some(MachineOperand::Immediate(_)));

    if ops.len() == 1 && !last_is_imm {
        // One-operand: IMUL r/m32 → F7 /5
        b.push(0xF7);
        match &ops[0] {
            MachineOperand::Register(r) => b.push(modrm(0b11, 5, *r as u8)),
            m if is_mem(m) => b.extend(encode_mem(5, m)?),
            o => return Err(format!("IMUL: bad operand {}", o)),
        }
    } else if last_is_imm {
        // Three-operand (or two-operand with imm): IMUL r32, r/m32, imm
        let dst = if let Some(ref r) = instr.result {
            gpr(r)?
        } else {
            gpr(&ops[0])?
        };
        let src_idx = if ops.len() >= 3 { 1 } else { 0 };
        let imm_idx = ops.len() - 1;
        let v = imm(&ops[imm_idx])?;
        let src = if src_idx < imm_idx {
            &ops[src_idx]
        } else {
            &ops[0]
        };

        if fits_in_i8(v) {
            b.push(0x6B);
        } else {
            b.push(0x69);
        }
        match src {
            MachineOperand::Register(s) => b.push(modrm(0b11, dst, *s as u8)),
            m if is_mem(m) => b.extend(encode_mem(dst, m)?),
            _ => {
                // src == dst (IMUL reg, imm shorthand)
                b.push(modrm(0b11, dst, dst));
            }
        }
        if fits_in_i8(v) {
            b.push(emit_i8(v));
        } else {
            b.extend_from_slice(&emit_i32(v));
        }
    } else {
        // Two-operand: IMUL r32, r/m32 → 0F AF
        // dst = r32 (destination and first multiplicand), src = r/m32
        let dst = if let Some(ref r) = instr.result {
            gpr(r)?
        } else {
            gpr(&ops[0])?
        };
        // When `result` is set and there are ≥2 operands, ops[0] is a
        // copy of the destination register (lhs); the actual source
        // (rhs multiplicand) is ops[1].
        let src_idx = if ops.len() >= 2 { 1 } else { 0 };
        let src = &ops[src_idx];

        b.push(0x0F);
        b.push(0xAF);
        match src {
            MachineOperand::Register(s) => b.push(modrm(0b11, dst, *s as u8)),
            m if is_mem(m) => b.extend(encode_mem(dst, m)?),
            o => return Err(format!("IMUL 2-op: bad source {}", o)),
        }
    }
    Ok(enc(b))
}

/// Encode a unary `0xF7 /ext` instruction (MUL, DIV, IDIV, NEG, NOT).
fn encode_unary_f7(instr: &MachineInstruction, ext: u8) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err(format!("Unary F7/{}: no operand", ext));
    }
    let mut b = vec![0xF7u8];
    match &instr.operands[0] {
        MachineOperand::Register(r) => b.push(modrm(0b11, ext, *r as u8)),
        m if is_mem(m) => b.extend(encode_mem(ext, m)?),
        o => return Err(format!("Unary F7/{}: bad operand {}", ext, o)),
    }
    Ok(enc(b))
}

/// Encode `INC` — short form `0x40+rd` for registers.
fn encode_inc(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let op = instr
        .result
        .as_ref()
        .or_else(|| instr.operands.first())
        .ok_or("INC: no operand")?;
    let mut b = Vec::new();
    match op {
        MachineOperand::Register(r) => b.push(0x40 + *r as u8),
        m if is_mem(m) => {
            b.push(0xFF);
            b.extend(encode_mem(0, m)?);
        }
        o => return Err(format!("INC: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `DEC` — short form `0x48+rd` for registers.
fn encode_dec(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let op = instr
        .result
        .as_ref()
        .or_else(|| instr.operands.first())
        .ok_or("DEC: no operand")?;
    let mut b = Vec::new();
    match op {
        MachineOperand::Register(r) => b.push(0x48 + *r as u8),
        m if is_mem(m) => {
            b.push(0xFF);
            b.extend(encode_mem(1, m)?);
        }
        o => return Err(format!("DEC: bad operand {}", o)),
    }
    Ok(enc(b))
}

// ===========================================================================
// Shift Encoders
// ===========================================================================

/// Encode `SHL`/`SHR`/`SAR` — shift by 1, CL, or imm8.
///
/// `ext` is the opcode extension: SHL=4, SHR=5, SAR=7.
fn encode_shift(instr: &MachineInstruction, ext: u8) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err(format!("Shift /{}: no operand", ext));
    }
    let mut b = Vec::new();

    // operands[0] = destination reg/mem, operands[1] = shift amount (optional)
    let dst = &instr.operands[0];

    // Determine shift source from operands[1] (if present)
    let shift_src = instr.operands.get(1);

    match shift_src {
        Some(MachineOperand::Immediate(1)) => {
            // SHx r/m32, 1 → 0xD1 /ext
            b.push(0xD1);
        }
        Some(MachineOperand::Immediate(v)) => {
            // SHx r/m32, imm8 → 0xC1 /ext imm8
            b.push(0xC1);
            match dst {
                MachineOperand::Register(r) => b.push(modrm(0b11, ext, *r as u8)),
                m if is_mem(m) => b.extend(encode_mem(ext, m)?),
                o => return Err(format!("Shift: bad dst {}", o)),
            }
            b.push(emit_i8(*v));
            return Ok(enc(b));
        }
        Some(MachineOperand::Register(r)) if *r == registers::ECX => {
            // SHx r/m32, CL → 0xD3 /ext
            b.push(0xD3);
        }
        None => {
            // Default: shift by 1
            b.push(0xD1);
        }
        _ => {
            // Treat as CL shift
            b.push(0xD3);
        }
    }

    match dst {
        MachineOperand::Register(r) => b.push(modrm(0b11, ext, *r as u8)),
        m if is_mem(m) => b.extend(encode_mem(ext, m)?),
        o => return Err(format!("Shift: bad dst {}", o)),
    }
    Ok(enc(b))
}

// ===========================================================================
// Comparison / Test Encoders
// ===========================================================================

/// Encode `TEST` — bitwise AND that only sets flags.
fn encode_test(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.len() < 2 {
        return Err("TEST: need 2 operands".into());
    }
    let dst = &instr.operands[0];
    let src = &instr.operands[1];
    let mut b = Vec::new();

    match (dst, src) {
        (MachineOperand::Register(d), MachineOperand::Register(s)) => {
            // TEST r/m32, r32 → 0x85 ModR/M
            b.push(0x85);
            b.push(modrm(0b11, *s as u8, *d as u8));
        }
        (MachineOperand::Register(d), MachineOperand::Immediate(v)) => {
            let d = *d as u8;
            if d == EAX_IDX {
                // TEST EAX, imm32 → 0xA9 imm32
                b.push(0xA9);
                b.extend_from_slice(&emit_i32(*v));
            } else {
                // TEST r/m32, imm32 → 0xF7 /0 ModR/M imm32
                b.push(0xF7);
                b.push(modrm(0b11, 0, d));
                b.extend_from_slice(&emit_i32(*v));
            }
        }
        (m, MachineOperand::Register(s)) if is_mem(m) => {
            b.push(0x85);
            b.extend(encode_mem(*s as u8, m)?);
        }
        (m, MachineOperand::Immediate(v)) if is_mem(m) => {
            b.push(0xF7);
            b.extend(encode_mem(0, m)?);
            b.extend_from_slice(&emit_i32(*v));
        }
        _ => return Err(format!("TEST: unsupported operands {}, {}", dst, src)),
    }
    Ok(enc(b))
}

// ===========================================================================
// Conditional Set / Conditional Move
// ===========================================================================

/// Encode `SETcc r/m8` → `0x0F 0x90+cc ModR/M`.
///
/// On i686, the `SETcc r8` form uses the legacy 8-bit register encoding
/// where register codes 4–7 map to AH/CH/DH/BH (high bytes), NOT the
/// low bytes of ESP–EDI.  When the destination register is >= 4, this
/// function falls back to an equivalent sequence that avoids 8-bit
/// register aliasing:
///
/// ```text
/// MOV reg, 0        ; clear to 0 (preserves flags)
/// Jcc_inverse +1    ; skip INC if condition NOT met
/// INC reg           ; set to 1
/// ```
///
/// This produces the same 0-or-1 result without touching byte registers.
fn encode_setcc(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let cc = extract_cc(instr)?;
    let dst = instr
        .result
        .as_ref()
        .or_else(|| instr.operands.first())
        .ok_or("SETCC: no destination")?;
    match dst {
        MachineOperand::Register(r) => {
            let rd = *r as u8;
            if rd >= 4 {
                // Registers ESP–EDI have no low-byte alias on i686.
                // We cannot use `SETcc r8` because register codes 4–7
                // in 8-bit mode map to AH/CH/DH/BH, not the low bytes.
                //
                // Emit the equivalent flag-safe sequence:
                //     MOV reg, 0        ; does NOT modify flags
                //     Jcc_inverse +1    ; skip INC if condition NOT met
                //     INC reg           ; set to 1
                let mut b = Vec::new();
                // MOV reg, imm32(0) — B8+rd 00 00 00 00 (5 bytes, flag-safe)
                b.push(0xB8 + rd);
                b.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                // Jcc_inverse over INC (short jump: 0x70+inv_cc disp8)
                // INC r32 is 1 byte (0x40+rd), so disp8 = 1
                let inv_cc = cc ^ 1; // flip low bit: E↔NE, L↔GE, etc.
                b.push(0x70 + inv_cc);
                b.push(0x01); // skip 1 byte (the INC)
                              // INC reg (0x40+rd)
                b.push(0x40 + rd);
                return Ok(enc(b));
            }
            // Registers 0–3 (EAX–EBX) have safe low-byte aliases.
            let mut b = vec![0x0Fu8, 0x90 + cc];
            b.push(modrm(0b11, 0, rd));
            Ok(enc(b))
        }
        m if is_mem(m) => {
            let mut b = vec![0x0Fu8, 0x90 + cc];
            b.extend(encode_mem(0, m)?);
            Ok(enc(b))
        }
        o => Err(format!("SETCC: bad dst {}", o)),
    }
}

/// Encode `CMOVcc r32, r/m32` → `0x0F 0x40+cc ModR/M`.
fn encode_cmovcc(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let cc = extract_cc(instr)?;
    let dst = if let Some(ref r) = instr.result {
        gpr(r)?
    } else {
        gpr(&instr.operands[0])?
    };
    let src_idx = if instr.result.is_some() { 0 } else { 1 };
    let src = instr
        .operands
        .get(src_idx)
        .ok_or("CMOVCC: no source operand")?;

    let mut b = vec![0x0Fu8, 0x40 + cc];
    match src {
        MachineOperand::Register(s) => b.push(modrm(0b11, dst, *s as u8)),
        m if is_mem(m) => b.extend(encode_mem(dst, m)?),
        o => return Err(format!("CMOVCC: bad source {}", o)),
    }
    Ok(enc(b))
}

// ===========================================================================
// Control Flow Encoders
// ===========================================================================

/// Encode `JMP` — unconditional near/short jump or indirect.
fn encode_jmp(
    instr: &MachineInstruction,
    label_offsets: &FxHashMap<String, usize>,
    current_offset: usize,
) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("JMP: no operand".into());
    }
    let target = &instr.operands[0];
    let mut b = Vec::new();

    match target {
        MachineOperand::BlockLabel(id) => {
            let key = label_key(*id);
            if let Some(&off) = label_offsets.get(&key) {
                // Known target — always use near JMP (5 bytes: E9 rel32)
                // to ensure size stability between pass 1 and pass 2.
                let rel = (off as i64) - (current_offset as i64) - 5;
                b.push(0xE9);
                b.extend_from_slice(&emit_i32(rel));
            } else {
                // Unknown target — emit near jump with zero displacement (to be patched).
                b.push(0xE9);
                b.extend_from_slice(&emit_i32(0));
            }
        }
        MachineOperand::GlobalSymbol(sym) => {
            b.push(0xE9);
            let reloc_off = b.len();
            b.extend_from_slice(&emit_i32(0));
            return Ok(enc_reloc(
                b,
                InstructionRelocation {
                    offset_in_instruction: reloc_off,
                    symbol: sym.clone(),
                    rel_type: relocations::R_386_PC32,
                    addend: -4,
                },
            ));
        }
        MachineOperand::Register(r) => {
            // JMP r/m32 → FF /4
            b.push(0xFF);
            b.push(modrm(0b11, 4, *r as u8));
        }
        m if is_mem(m) => {
            b.push(0xFF);
            b.extend(encode_mem(4, m)?);
        }
        o => return Err(format!("JMP: unsupported operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `Jcc` — conditional near/short jump.
fn encode_jcc(
    instr: &MachineInstruction,
    label_offsets: &FxHashMap<String, usize>,
    current_offset: usize,
) -> Result<EncodedInstruction, String> {
    let cc = extract_cc(instr)?;
    let target = find_target(instr).ok_or_else(|| "JCC: no branch target operand".to_string())?;
    let mut b = Vec::new();

    match target {
        MachineOperand::BlockLabel(id) => {
            let key = label_key(*id);
            if let Some(&off) = label_offsets.get(&key) {
                // Known: Jcc near = 6 bytes (0F 8x xx xx xx xx)
                let rel = (off as i64) - (current_offset as i64) - 6;
                b.push(0x0F);
                b.push(0x80 + cc);
                b.extend_from_slice(&emit_i32(rel));
            } else {
                // Unknown — placeholder for later fixup
                b.push(0x0F);
                b.push(0x80 + cc);
                b.extend_from_slice(&emit_i32(0));
            }
        }
        MachineOperand::GlobalSymbol(sym) => {
            b.push(0x0F);
            b.push(0x80 + cc);
            let reloc_off = b.len();
            b.extend_from_slice(&emit_i32(0));
            return Ok(enc_reloc(
                b,
                InstructionRelocation {
                    offset_in_instruction: reloc_off,
                    symbol: sym.clone(),
                    rel_type: relocations::R_386_PC32,
                    addend: -4,
                },
            ));
        }
        o => return Err(format!("JCC: unsupported target {}", o)),
    }
    Ok(enc(b))
}

/// Encode `CALL` — direct (rel32) or indirect (r/m32).
fn encode_call(
    instr: &MachineInstruction,
    label_offsets: &FxHashMap<String, usize>,
    current_offset: usize,
) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("CALL: no operand".into());
    }
    let target = &instr.operands[0];
    let mut b = Vec::new();

    match target {
        MachineOperand::GlobalSymbol(sym) => {
            // CALL rel32 → E8 rel32  with R_386_PC32 or R_386_PLT32 relocation
            b.push(0xE8);
            let reloc_off = b.len();
            b.extend_from_slice(&emit_i32(0));
            return Ok(enc_reloc(
                b,
                InstructionRelocation {
                    offset_in_instruction: reloc_off,
                    symbol: sym.clone(),
                    rel_type: relocations::R_386_PLT32,
                    addend: -4,
                },
            ));
        }
        MachineOperand::BlockLabel(id) => {
            let key = label_key(*id);
            if let Some(&off) = label_offsets.get(&key) {
                let rel = (off as i64) - (current_offset as i64) - 5;
                b.push(0xE8);
                b.extend_from_slice(&emit_i32(rel));
            } else {
                b.push(0xE8);
                b.extend_from_slice(&emit_i32(0));
            }
        }
        MachineOperand::Register(r) => {
            // CALL r/m32 → FF /2
            b.push(0xFF);
            b.push(modrm(0b11, 2, *r as u8));
        }
        m if is_mem(m) => {
            b.push(0xFF);
            b.extend(encode_mem(2, m)?);
        }
        o => return Err(format!("CALL: unsupported operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `RET` — near return, optionally popping imm16 bytes.
fn encode_ret(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let mut b = Vec::new();
    if let Some(MachineOperand::Immediate(v)) = instr.operands.first() {
        // RET imm16 → C2 imm16
        b.push(0xC2);
        b.extend_from_slice(&emit_i16(*v));
    } else {
        // RET → C3
        b.push(0xC3);
    }
    Ok(enc(b))
}

// CDQ (0x99) and FCHS (D9 E0) are encoded inline in the dispatcher
// because they take no operands and are just one or two bytes.

// ===========================================================================
// x87 FPU Instruction Encoders (D8h-DFh)
// ===========================================================================

/// Helper: extract the FPU stack index (0..7) from an operand.
fn fpu_idx(op: &MachineOperand) -> Result<u8, String> {
    match op {
        MachineOperand::Register(r) => {
            if registers::is_fpu_reg(*r) {
                registers::fpu_stack_index(*r).ok_or_else(|| format!("Bad FPU reg {}", r))
            } else {
                Err(format!("Expected FPU register, got GPR {}", r))
            }
        }
        _ => Err(format!("Expected FPU register, got {}", op)),
    }
}

/// Helper: determine FPU memory width from an optional third operand (immediate 4, 8, or 10).
/// Returns (opcode_byte, extension) for the given default extension.
fn fpu_mem_width(instr: &MachineInstruction, default_ext: u8) -> (u8, u8) {
    // If there's a width hint as a second or third immediate, use it.
    // Convention: operands may include an Immediate(4|8|10) for width.
    let width = instr
        .operands
        .iter()
        .find_map(|op| match op {
            MachineOperand::Immediate(4) => Some(4i64),
            MachineOperand::Immediate(8) => Some(8),
            MachineOperand::Immediate(10) => Some(10),
            _ => None,
        })
        .unwrap_or(8); // default to 64-bit double

    match width {
        4 => (0xD9, default_ext), // 32-bit float
        10 | 16 => {
            // 80-bit extended precision: DB /5 for FLD, DB /7 for FSTP
            if default_ext == 0 {
                (0xDB, 5)
            } else {
                (0xDB, 7)
            }
        }
        _ => (0xDD, default_ext), // 64-bit double (default)
    }
}

/// Encode `FLD` — load floating-point value onto FPU stack.
fn encode_fld(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FLD: no operand".into());
    }
    let src = &instr.operands[0];
    let mut b = Vec::new();

    match src {
        MachineOperand::Register(r) if registers::is_fpu_reg(*r) => {
            let i =
                registers::fpu_stack_index(*r).ok_or_else(|| format!("FLD: bad FPU reg {}", r))?;
            // FLD ST(i) → D9 C0+i
            b.push(0xD9);
            b.push(0xC0 + i);
        }
        m if is_mem(m) => {
            let (opc, ext) = fpu_mem_width(instr, 0);
            b.push(opc);
            b.extend(encode_mem(ext, m)?);
        }
        o => return Err(format!("FLD: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `FSTP` — store FPU stack top and pop.
fn encode_fstp(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FSTP: no operand".into());
    }
    let dst = &instr.operands[0];
    let mut b = Vec::new();

    match dst {
        MachineOperand::Register(r) if registers::is_fpu_reg(*r) => {
            let i =
                registers::fpu_stack_index(*r).ok_or_else(|| format!("FSTP: bad FPU reg {}", r))?;
            // FSTP ST(i) → DD D8+i
            b.push(0xDD);
            b.push(0xD8 + i);
        }
        m if is_mem(m) => {
            let (opc, ext) = fpu_mem_width(instr, 3);
            b.push(opc);
            b.extend(encode_mem(ext, m)?);
        }
        o => return Err(format!("FSTP: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode a standard FPU arithmetic instruction (FADD, FSUB, FMUL, FDIV).
///
/// * `reg_base` — the base byte added to `i` for register-register form
///   (e.g., C0 for FADD, E0 for FSUB, C8 for FMUL, F0 for FDIV)
/// * `ext32` — opcode extension digit for D8 (32-bit) memory form
/// * `ext64` — opcode extension digit for DC (64-bit) memory form
fn encode_fpu_arith(
    instr: &MachineInstruction,
    reg_base: u8,
    ext32: u8,
    ext64: u8,
) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FPU arith: no operand".into());
    }
    let src = &instr.operands[0];
    let mut b = Vec::new();

    match src {
        MachineOperand::Register(r) if registers::is_fpu_reg(*r) => {
            let i = registers::fpu_stack_index(*r)
                .ok_or_else(|| format!("FPU arith: bad FPU reg {}", r))?;
            // FADD ST, ST(i) → D8 <reg_base>+i
            b.push(0xD8);
            b.push(reg_base + i);
        }
        m if is_mem(m) => {
            // Determine width: 4-byte → D8 /ext32, 8-byte → DC /ext64
            let width = instr.operands.iter().find_map(|op| match op {
                MachineOperand::Immediate(4) => Some(4i64),
                _ => None,
            });
            if width == Some(4) {
                b.push(0xD8);
                b.extend(encode_mem(ext32, m)?);
            } else {
                b.push(0xDC);
                b.extend(encode_mem(ext64, m)?);
            }
        }
        o => return Err(format!("FPU arith: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `FCOMP` — compare ST(0) with operand and pop.
fn encode_fcomp(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FCOMP: no operand".into());
    }
    let src = &instr.operands[0];
    let mut b = Vec::new();

    match src {
        MachineOperand::Register(r) if registers::is_fpu_reg(*r) => {
            let i = registers::fpu_stack_index(*r)
                .ok_or_else(|| format!("FCOMP: bad FPU reg {}", r))?;
            // FCOMP ST(i) → D8 D8+i
            b.push(0xD8);
            b.push(0xD8 + i);
        }
        m if is_mem(m) => {
            let width = instr.operands.iter().find_map(|op| match op {
                MachineOperand::Immediate(4) => Some(4i64),
                _ => None,
            });
            if width == Some(4) {
                b.push(0xD8);
                b.extend(encode_mem(3, m)?);
            } else {
                b.push(0xDC);
                b.extend(encode_mem(3, m)?);
            }
        }
        o => return Err(format!("FCOMP: bad operand {}", o)),
    }
    Ok(enc(b))
}

/// Encode `FILD` — load integer from memory onto FPU stack.
fn encode_fild(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FILD: no operand".into());
    }
    let src = &instr.operands[0];
    if !is_mem(src) {
        return Err(format!("FILD: expected memory operand, got {}", src));
    }
    let mut b = Vec::new();

    let width = instr
        .operands
        .iter()
        .find_map(|op| match op {
            MachineOperand::Immediate(2) => Some(2i64),
            MachineOperand::Immediate(4) => Some(4),
            MachineOperand::Immediate(8) => Some(8),
            _ => None,
        })
        .unwrap_or(4);

    match width {
        2 => {
            // FILD m16int → DF /0
            b.push(0xDF);
            b.extend(encode_mem(0, src)?);
        }
        8 => {
            // FILD m64int → DF /5
            b.push(0xDF);
            b.extend(encode_mem(5, src)?);
        }
        _ => {
            // FILD m32int → DB /0  (default)
            b.push(0xDB);
            b.extend(encode_mem(0, src)?);
        }
    }
    Ok(enc(b))
}

/// Encode `FISTP` — store FPU stack top as integer and pop.
fn encode_fistp(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FISTP: no operand".into());
    }
    let dst = &instr.operands[0];
    if !is_mem(dst) {
        return Err(format!("FISTP: expected memory operand, got {}", dst));
    }
    let mut b = Vec::new();

    let width = instr
        .operands
        .iter()
        .find_map(|op| match op {
            MachineOperand::Immediate(2) => Some(2i64),
            MachineOperand::Immediate(4) => Some(4),
            MachineOperand::Immediate(8) => Some(8),
            _ => None,
        })
        .unwrap_or(4);

    match width {
        2 => {
            // FISTP m16int → DF /3
            b.push(0xDF);
            b.extend(encode_mem(3, dst)?);
        }
        8 => {
            // FISTP m64int → DF /7
            b.push(0xDF);
            b.extend(encode_mem(7, dst)?);
        }
        _ => {
            // FISTP m32int → DB /3 (default)
            b.push(0xDB);
            b.extend(encode_mem(3, dst)?);
        }
    }
    Ok(enc(b))
}

/// Encode `FXCH ST(i)` → D9 C8+i.
fn encode_fxch(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let i = if let Some(op) = instr.operands.first() {
        fpu_idx(op)?
    } else {
        1 // FXCH with no operand defaults to ST(1)
    };
    Ok(enc(vec![0xD9, 0xC8 + i]))
}

/// Encode `FUCOMIP ST, ST(i)` → DF E8+i.
fn encode_fucomip(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    if instr.operands.is_empty() {
        return Err("FUCOMIP: no operand".into());
    }
    // The source ST(i) may be the first or second operand
    // (first might be ST(0) implicitly)
    let op = instr.operands.last().unwrap();
    let i = fpu_idx(op)?;
    Ok(enc(vec![0xDF, 0xE8 + i]))
}

// ===========================================================================
// Stack Frame Instructions
// ===========================================================================

/// Encode `ENTER imm16, 0` → C8 imm16 00.
fn encode_enter(instr: &MachineInstruction) -> Result<EncodedInstruction, String> {
    let size = if let Some(MachineOperand::Immediate(v)) = instr.operands.first() {
        *v as u16
    } else {
        0u16
    };
    let nesting = if let Some(MachineOperand::Immediate(v)) = instr.operands.get(1) {
        *v as u8
    } else {
        0u8
    };
    let mut b = vec![0xC8u8];
    b.extend_from_slice(&size.to_le_bytes());
    b.push(nesting);
    Ok(enc(b))
}

// ===========================================================================
// Instruction Size Computation (Pass 1)
// ===========================================================================

/// Compute the **maximum** encoded size of a `MachineInstruction` in bytes.
///
/// Used by the two-pass assembler's first pass to lay out label offsets.
/// For branch instructions, we always assume near (32-bit displacement) forms
/// to guarantee the computed offsets are conservative upper bounds. A subsequent
/// relaxation pass may shrink short branches.
pub fn compute_instruction_size(instr: &MachineInstruction) -> usize {
    use crate::backend::i686::codegen::*;

    // Helper: maximum memory operand encoding size.
    // ModR/M(1) + SIB(1) + disp32(4) = 6 bytes worst case.
    let mem_max = 6usize;

    match instr.opcode {
        // ---- Data Movement ----
        I686_MOV => {
            // Worst case: C7 /0 + mem(6) + imm32(4) = 11
            // Or: 0x8B + mem(6) = 7
            11
        }
        I686_MOVZX | I686_MOVSX => {
            // 0x0F + opcode + ModR/M(+SIB+disp) = 2 + 6 = 8
            2 + mem_max
        }
        I686_LEA => {
            // 0x8D + mem(6) = 7
            1 + mem_max
        }
        I686_PUSH => {
            // Worst case: 68 imm32 = 5
            // Or FF /6 + mem(6) = 7
            7
        }
        I686_POP => {
            // 8F /0 + mem(6) = 7
            7
        }
        I686_XCHG => {
            // 87 + ModR/M(+SIB+disp) = 1+6 = 7
            7
        }

        // ---- ALU ----
        I686_ADD | I686_SUB | I686_ADC | I686_SBB | I686_AND | I686_OR | I686_XOR | I686_CMP => {
            // Worst case: 81 /ext + mem(6) + imm32(4) = 11
            11
        }
        I686_IMUL => {
            // Worst case: 69 + ModR/M + SIB + disp32 + imm32 = 1+6+4 = 11
            // Or 0F AF + mem(6) = 8
            11
        }
        I686_MUL | I686_DIV | I686_IDIV | I686_NEG | I686_NOT => {
            // F7 /ext + mem(6) = 7
            1 + mem_max
        }
        I686_INC | I686_DEC => {
            // Register: 1 byte (40+rd or 48+rd)
            // Memory: FF /ext + mem(6) = 7
            7
        }
        I686_CDQ => 1,

        // ---- Shifts ----
        I686_SHL | I686_SHR | I686_SAR => {
            // C1 /ext + mem(6) + imm8(1) = 8
            8
        }

        // ---- Test / Compare ----
        I686_TEST => {
            // F7 /0 + mem(6) + imm32(4) = 11
            11
        }

        // ---- Conditional ----
        I686_SETCC => {
            // 0F 9x + ModR/M(+SIB+disp) = 2+6 = 8
            8
        }
        I686_CMOVCC => {
            // 0F 4x + ModR/M(+SIB+disp) = 2+6 = 8
            8
        }

        // ---- Control Flow ----
        I686_JMP => {
            // Near JMP: E9 + rel32 = 5
            // Indirect: FF /4 + mem(6) = 7
            7
        }
        I686_JCC => {
            // Near Jcc: 0F 8x + rel32 = 6
            6
        }
        I686_CALL => {
            // Direct: E8 + rel32 = 5
            // Indirect: FF /2 + mem(6) = 7
            7
        }
        I686_RET => {
            // RET imm16 = 3, RET = 1
            3
        }

        // ---- FPU ----
        I686_FLD | I686_FSTP => {
            // Register form: 2 bytes (Dx xx)
            // Memory form: 1 + mem(6) = 7
            7
        }
        I686_FADD | I686_FSUB | I686_FMUL | I686_FDIV => {
            // Register: 2 bytes
            // Memory: 1 + mem(6) = 7
            7
        }
        I686_FCHS => 2,
        I686_FCOMP => 7,
        I686_FILD | I686_FISTP => {
            // 1 + mem(6) = 7
            7
        }
        I686_FXCH => 2,
        I686_FUCOMIP => 2,

        // ---- Stack Frame ----
        I686_ENTER => 4, // C8 imm16 imm8
        I686_LEAVE => 1, // C9

        // ---- Misc ----
        I686_NOP => 1,   // 90
        I686_INT3 => 1,  // CC
        I686_UD2 => 2,   // 0F 0B
        I686_BSWAP => 2, // 0F C8+rd

        // Unknown opcode — pessimistic estimate
        _ => 15,
    }
}
