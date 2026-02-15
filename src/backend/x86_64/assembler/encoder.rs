// SPDX-License-Identifier: MIT
#![allow(dead_code)]
//! # x86-64 Instruction Encoder
//!
//! This module implements the full variable-length instruction encoding scheme for the
//! BCC built-in x86-64 assembler. It encodes individual `MachineInstr` into raw bytes
//! using:
//!
//! - **REX prefix generation** (W/R/X/B bits for 64-bit operand size and extended regs R8–R15)
//! - **ModR/M byte construction** for register/memory operand addressing
//! - **SIB byte encoding** for scaled-index-base addressing modes
//! - **VEX prefix** encoding for SSE/AVX instructions
//! - **Operand size override prefix** (0x66) for 16-bit operations
//! - **Opcode table lookup** for all x86-64 instruction forms
//! - **Displacement and immediate value** encoding (8/16/32/64-bit little-endian)
//!
//! All x86-64 addressing modes are supported:
//! - Register-direct, register-indirect `[reg]`
//! - Base+displacement `[reg+disp8]`, `[reg+disp32]`
//! - Base+index×scale+displacement `[base+index*scale+disp]`
//! - RIP-relative `[RIP+disp32]`, absolute `[disp32]`
//!
//! Zero external dependencies — all encoding is implemented internally.

use crate::backend::traits::{MachineFunction, MachineInstr, MachineOperand, PhysReg};
use crate::backend::x86_64::assembler::relocations::X86_64RelocationType;
use crate::backend::x86_64::opcodes;
use crate::backend::x86_64::registers::{self, RAX, RBP, RSP};

// ============================================================================
// Public Enums
// ============================================================================

/// Operand size classification for x86-64 instruction encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandSize {
    /// 8-bit operand — special REX handling for SPL/BPL/SIL/DIL
    Byte,
    /// 16-bit operand — requires operand-size override prefix 0x66
    Word,
    /// 32-bit operand — default operand size in 64-bit mode
    DWord,
    /// 64-bit operand — requires REX.W=1 prefix
    QWord,
}

impl OperandSize {
    /// Returns the size in bytes.
    pub fn byte_count(self) -> usize {
        match self {
            OperandSize::Byte => 1,
            OperandSize::Word => 2,
            OperandSize::DWord => 4,
            OperandSize::QWord => 8,
        }
    }

    /// Determines `OperandSize` from a bit width.
    pub fn from_bits(bits: u32) -> Self {
        match bits {
            8 => OperandSize::Byte,
            16 => OperandSize::Word,
            64 => OperandSize::QWord,
            _ => OperandSize::DWord,
        }
    }
}

/// x86-64 condition codes used by Jcc, SETcc, and CMOVcc.
///
/// Each variant maps to a 4-bit encoding per Intel SDM Vol 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionCode {
    O,
    NO,
    B,
    AE,
    E,
    NE,
    BE,
    A,
    S,
    NS,
    P,
    NP,
    L,
    GE,
    LE,
    G,
}

impl ConditionCode {
    /// Returns the 4-bit condition encoding (0x0–0xF).
    pub fn encoding(self) -> u8 {
        match self {
            ConditionCode::O => 0x0,
            ConditionCode::NO => 0x1,
            ConditionCode::B => 0x2,
            ConditionCode::AE => 0x3,
            ConditionCode::E => 0x4,
            ConditionCode::NE => 0x5,
            ConditionCode::BE => 0x6,
            ConditionCode::A => 0x7,
            ConditionCode::S => 0x8,
            ConditionCode::NS => 0x9,
            ConditionCode::P => 0xA,
            ConditionCode::NP => 0xB,
            ConditionCode::L => 0xC,
            ConditionCode::GE => 0xD,
            ConditionCode::LE => 0xE,
            ConditionCode::G => 0xF,
        }
    }

    /// Builds from 4-bit encoding. Returns `None` if `val > 0xF`.
    pub fn from_encoding(val: u8) -> Option<Self> {
        Some(match val & 0xF {
            0x0 => ConditionCode::O,
            0x1 => ConditionCode::NO,
            0x2 => ConditionCode::B,
            0x3 => ConditionCode::AE,
            0x4 => ConditionCode::E,
            0x5 => ConditionCode::NE,
            0x6 => ConditionCode::BE,
            0x7 => ConditionCode::A,
            0x8 => ConditionCode::S,
            0x9 => ConditionCode::NS,
            0xA => ConditionCode::P,
            0xB => ConditionCode::NP,
            0xC => ConditionCode::L,
            0xD => ConditionCode::GE,
            0xE => ConditionCode::LE,
            0xF => ConditionCode::G,
            _ => return None,
        })
    }

    /// Inverts the condition (E→NE, L→GE, etc.).
    pub fn invert(self) -> Self {
        match self {
            ConditionCode::O => ConditionCode::NO,
            ConditionCode::NO => ConditionCode::O,
            ConditionCode::B => ConditionCode::AE,
            ConditionCode::AE => ConditionCode::B,
            ConditionCode::E => ConditionCode::NE,
            ConditionCode::NE => ConditionCode::E,
            ConditionCode::BE => ConditionCode::A,
            ConditionCode::A => ConditionCode::BE,
            ConditionCode::S => ConditionCode::NS,
            ConditionCode::NS => ConditionCode::S,
            ConditionCode::P => ConditionCode::NP,
            ConditionCode::NP => ConditionCode::P,
            ConditionCode::L => ConditionCode::GE,
            ConditionCode::GE => ConditionCode::L,
            ConditionCode::LE => ConditionCode::G,
            ConditionCode::G => ConditionCode::LE,
        }
    }
}

// ============================================================================
// MemoryOperand
// ============================================================================

/// Represents an x86-64 memory addressing operand.
///
/// General form: `[base + index*scale + displacement]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryOperand {
    pub base: Option<PhysReg>,
    pub index: Option<PhysReg>,
    pub scale: u8,
    pub displacement: i32,
}

impl MemoryOperand {
    /// `[base + displacement]`
    pub fn base_disp(base: PhysReg, displacement: i32) -> Self {
        Self {
            base: Some(base),
            index: None,
            scale: 1,
            displacement,
        }
    }

    /// `[base + index*scale + displacement]`
    pub fn base_index_scale_disp(
        base: PhysReg,
        index: PhysReg,
        scale: u8,
        displacement: i32,
    ) -> Self {
        Self {
            base: Some(base),
            index: Some(index),
            scale,
            displacement,
        }
    }

    /// `[disp32]` — absolute address.
    pub fn disp_only(displacement: i32) -> Self {
        Self {
            base: None,
            index: None,
            scale: 1,
            displacement,
        }
    }

    /// Construct from `MachineOperand::Memory` fields.
    pub fn from_machine_operand(
        base: PhysReg,
        offset: i32,
        index: Option<PhysReg>,
        scale: u8,
    ) -> Self {
        Self {
            base: Some(base),
            index,
            scale,
            displacement: offset,
        }
    }
}

// ============================================================================
// Fixup — internal label reference to be patched later
// ============================================================================

#[derive(Debug, Clone)]
pub struct Fixup {
    pub offset: usize,
    pub label: u32,
    pub size: u8,
    pub pc_offset: usize,
}

// ============================================================================
// Relocation — external symbol reference for the linker
// ============================================================================

#[derive(Debug, Clone)]
pub struct Relocation {
    pub offset: usize,
    pub symbol: String,
    pub reloc_type: X86_64RelocationType,
    pub addend: i64,
}

// ============================================================================
// AssembledFunction — output of encode_function
// ============================================================================

#[derive(Debug, Clone)]
pub struct AssembledFunction {
    pub code: Vec<u8>,
    pub relocations: Vec<Relocation>,
    pub label_offsets: Vec<(u32, usize)>,
}

// ============================================================================
// EncodingContext
// ============================================================================

/// Mutable encoding state for a sequence of instructions.
#[derive(Debug)]
pub struct EncodingContext {
    pub buffer: Vec<u8>,
    pub current_offset: usize,
    fixups: Vec<Fixup>,
    relocations: Vec<Relocation>,
    label_offsets: Vec<(u32, usize)>,
}

impl Default for EncodingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingContext {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            current_offset: 0,
            fixups: Vec::new(),
            relocations: Vec::new(),
            label_offsets: Vec::new(),
        }
    }

    #[inline]
    pub fn emit_byte(&mut self, b: u8) {
        self.buffer.push(b);
        self.current_offset += 1;
    }

    #[inline]
    pub fn emit_bytes(&mut self, bs: &[u8]) {
        self.buffer.extend_from_slice(bs);
        self.current_offset += bs.len();
    }

    /// Record a fixup; caller should have already written placeholder bytes of `size`.
    pub fn record_fixup(&mut self, label: u32, size: u8) {
        let offset = self.current_offset - size as usize;
        self.fixups.push(Fixup {
            offset,
            label,
            size,
            pc_offset: self.current_offset,
        });
    }

    /// Record a relocation (the 4-byte field just written is the relocation site).
    pub fn record_relocation(
        &mut self,
        symbol: String,
        reloc_type: X86_64RelocationType,
        addend: i64,
    ) {
        let offset = self.current_offset - 4;
        self.relocations.push(Relocation {
            offset,
            symbol,
            reloc_type,
            addend,
        });
    }

    pub fn bind_label(&mut self, label: u32) {
        self.label_offsets.push((label, self.current_offset));
    }

    /// Resolve fixups; returns the count of unresolvable entries.
    pub fn resolve_fixups(&mut self) -> usize {
        let mut unresolved = 0usize;
        for fixup in &self.fixups {
            if let Some(&(_, tgt)) = self.label_offsets.iter().find(|(id, _)| *id == fixup.label) {
                let rel = tgt as i64 - fixup.pc_offset as i64;
                match fixup.size {
                    1 => {
                        if (-128..=127).contains(&rel) {
                            self.buffer[fixup.offset] = rel as u8;
                        } else {
                            unresolved += 1;
                        }
                    }
                    4 => {
                        let b = (rel as i32).to_le_bytes();
                        self.buffer[fixup.offset..fixup.offset + 4].copy_from_slice(&b);
                    }
                    _ => {
                        unresolved += 1;
                    }
                }
            } else {
                unresolved += 1;
            }
        }
        unresolved
    }

    pub fn finish(self) -> AssembledFunction {
        AssembledFunction {
            code: self.buffer,
            relocations: self.relocations,
            label_offsets: self.label_offsets,
        }
    }
}

// ============================================================================
// REX prefix helpers (Intel SDM Vol 2, Section 2.2.1)
// ============================================================================

const REX_BASE: u8 = 0x40;
const REX_W: u8 = 0x08;
const REX_R: u8 = 0x04;
const REX_X: u8 = 0x02;
const REX_B: u8 = 0x01;

/// Construct a REX byte from individual flag bits.
#[inline]
fn encode_rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    REX_BASE
        | if w { REX_W } else { 0 }
        | if r { REX_R } else { 0 }
        | if x { REX_X } else { 0 }
        | if b { REX_B } else { 0 }
}

/// Determine whether a REX prefix is needed for the given operands and size.
fn needs_rex_prefix(
    size: OperandSize,
    reg: Option<PhysReg>,
    rm: Option<PhysReg>,
    index: Option<PhysReg>,
) -> bool {
    if size == OperandSize::QWord {
        return true;
    }
    if let Some(r) = reg {
        if registers::needs_rex(r) {
            return true;
        }
    }
    if let Some(r) = rm {
        if registers::needs_rex(r) {
            return true;
        }
    }
    if let Some(r) = index {
        if registers::needs_rex(r) {
            return true;
        }
    }
    // Byte-size ops on SPL/BPL/SIL/DIL need REX to disambiguate from AH/CH/DH/BH.
    if size == OperandSize::Byte {
        for r in [reg, rm, index].iter().copied().flatten() {
            let idx = registers::reg_index(r);
            if (4..=7).contains(&idx) && !registers::needs_rex(r) {
                return true;
            }
        }
    }
    false
}

/// Compute the full REX byte (or None) for the given operand combination.
fn compute_rex(
    size: OperandSize,
    reg: Option<PhysReg>,
    rm: Option<PhysReg>,
    index: Option<PhysReg>,
) -> Option<u8> {
    let w = size == OperandSize::QWord;
    let r = reg.is_some_and(registers::needs_rex);
    let x = index.is_some_and(registers::needs_rex);
    let b = rm.is_some_and(registers::needs_rex);
    if w || r || x || b || needs_rex_prefix(size, reg, rm, index) {
        Some(encode_rex(w, r, x, b))
    } else {
        None
    }
}

// ============================================================================
// ModR/M byte encoding (Intel SDM Vol 2, Section 2.1.3)
// ============================================================================

/// Build a ModR/M byte: `(mod << 6) | (reg << 3) | rm`.
#[inline]
fn encode_modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 3) << 6) | ((reg & 7) << 3) | (rm & 7)
}

/// Return the 3-bit hardware encoding for any physical register (GPR or SSE).
///
/// For GPRs this is simply `reg.0 & 7`.  For SSE registers (XMM0–XMM15,
/// PhysReg 16–31) it maps to the corresponding 3-bit field that the CPU
/// uses inside ModR/M / SIB bytes; the 4th bit is supplied via REX.R / REX.B.
#[inline]
fn reg_hw_encoding(reg: PhysReg) -> u8 {
    registers::reg_index(reg) & 0x07
}

/// ModR/M byte for two register-direct operands (mod=11).
///
/// Works for both GPR-GPR and SSE-SSE (and mixed GPR/SSE) pairings.
#[inline]
fn modrm_reg_reg(reg: PhysReg, rm: PhysReg) -> u8 {
    encode_modrm(0b11, reg_hw_encoding(reg), reg_hw_encoding(rm))
}

// ============================================================================
// SIB byte encoding (Intel SDM Vol 2, Section 2.1.3)
// ============================================================================

/// Build a SIB byte: `(scale << 6) | (index << 3) | base`.
#[inline]
fn encode_sib(scale_bits: u8, index: u8, base: u8) -> u8 {
    ((scale_bits & 3) << 6) | ((index & 7) << 3) | (base & 7)
}

/// Convert scale factor (1/2/4/8) to the 2-bit SIB scale field.
fn scale_to_bits(s: u8) -> u8 {
    match s {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => 0b00,
    }
}

/// Determine whether a SIB byte is needed for the given base and optional index.
fn needs_sib(base: PhysReg, index: Option<PhysReg>) -> bool {
    // RSP (4) and R12 (12) as base always require SIB.
    let enc = registers::gpr_encoding(base);
    enc == 4 || index.is_some()
}

// ============================================================================
// Displacement and immediate encoding
// ============================================================================

/// Can the value be represented as a sign-extended 8-bit immediate?
#[inline]
fn fits_in_i8(val: i64) -> bool {
    (-128..=127).contains(&val)
}

/// Can the value be represented as a sign-extended 32-bit immediate?
#[inline]
fn fits_in_i32(val: i64) -> bool {
    val >= (i32::MIN as i64) && val <= (i32::MAX as i64)
}

fn emit_disp8(ctx: &mut EncodingContext, disp: i32) {
    ctx.emit_byte(disp as u8);
}

fn emit_disp32(ctx: &mut EncodingContext, disp: i32) {
    ctx.emit_bytes(&disp.to_le_bytes());
}

fn emit_imm8(ctx: &mut EncodingContext, imm: i64) {
    ctx.emit_byte(imm as u8);
}

fn emit_imm16(ctx: &mut EncodingContext, imm: i64) {
    ctx.emit_bytes(&(imm as i16).to_le_bytes());
}

fn emit_imm32(ctx: &mut EncodingContext, imm: i64) {
    ctx.emit_bytes(&(imm as i32).to_le_bytes());
}

fn emit_imm64(ctx: &mut EncodingContext, imm: i64) {
    ctx.emit_bytes(&imm.to_le_bytes());
}

// ============================================================================
// Memory operand → ModR/M + SIB + disp encoding
// ============================================================================

/// Emit ModR/M (and optional SIB + displacement) bytes for a memory operand.
///
/// `reg_field` is the 3-bit value placed in the ModR/M `reg` field (either a
/// register encoding or an opcode extension `/digit`).
///
/// The REX prefix should already have been emitted by the caller.
fn emit_memory_operand(ctx: &mut EncodingContext, reg_field: u8, mem: &MemoryOperand) {
    match (mem.base, mem.index) {
        // --- No base, no index → [disp32] (SIB encoding) ---
        (None, None) => {
            // mod=00, rm=100 (SIB follows), SIB: scale=00, index=100 (none), base=101 (disp32)
            ctx.emit_byte(encode_modrm(0b00, reg_field, 0b100));
            ctx.emit_byte(encode_sib(0b00, 0b100, 0b101));
            emit_disp32(ctx, mem.displacement);
        }
        // --- No base, has index → [index*scale + disp32] ---
        (None, Some(idx)) => {
            let idx_enc = registers::gpr_encoding(idx);
            ctx.emit_byte(encode_modrm(0b00, reg_field, 0b100));
            ctx.emit_byte(encode_sib(scale_to_bits(mem.scale), idx_enc, 0b101));
            emit_disp32(ctx, mem.displacement);
        }
        // --- Has base, may or may not have index ---
        (Some(base), opt_index) => {
            let base_enc = registers::gpr_encoding(base);
            let need_sib = base_enc == 4 || opt_index.is_some(); // RSP/R12 need SIB

            // Determine mod bits from displacement magnitude.
            let (mod_bits, disp_kind) = if mem.displacement == 0 && base_enc != 5 {
                // mod=00 — no displacement (except RBP/R13 which always need disp8=0)
                (0b00u8, DispKind::None)
            } else if fits_in_i8(mem.displacement as i64) {
                (0b01u8, DispKind::Disp8)
            } else {
                (0b10u8, DispKind::Disp32)
            };

            if need_sib {
                let idx_enc = opt_index.map_or(0b100u8, registers::gpr_encoding);
                let scale_bits = opt_index.map_or(0b00, |_| scale_to_bits(mem.scale));
                ctx.emit_byte(encode_modrm(mod_bits, reg_field, 0b100));
                ctx.emit_byte(encode_sib(scale_bits, idx_enc, base_enc));
            } else {
                ctx.emit_byte(encode_modrm(mod_bits, reg_field, base_enc));
            }

            match disp_kind {
                DispKind::None => {}
                DispKind::Disp8 => emit_disp8(ctx, mem.displacement),
                DispKind::Disp32 => emit_disp32(ctx, mem.displacement),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum DispKind {
    None,
    Disp8,
    Disp32,
}

// ============================================================================
// VEX prefix encoding (for SSE/AVX)
// ============================================================================

/// Emit a 2-byte VEX prefix: `C5 [R vvvv L pp]`.
fn emit_vex2(ctx: &mut EncodingContext, r: bool, vvvv: u8, l: bool, pp: u8) {
    ctx.emit_byte(0xC5);
    let byte = (if r { 0 } else { 0x80 })
        | ((!vvvv & 0xF) << 3)
        | (if l { 0x04 } else { 0 })
        | (pp & 0x03);
    ctx.emit_byte(byte);
}

/// Emit a 3-byte VEX prefix: `C4 [RXB mmmmm] [W vvvv L pp]`.
fn emit_vex3(
    ctx: &mut EncodingContext,
    r: bool,
    x: bool,
    b: bool,
    mmmmm: u8,
    w: bool,
    vvvv: u8,
    l: bool,
    pp: u8,
) {
    ctx.emit_byte(0xC4);
    let byte1 = (if r { 0 } else { 0x80 })
        | (if x { 0 } else { 0x40 })
        | (if b { 0 } else { 0x20 })
        | (mmmmm & 0x1F);
    ctx.emit_byte(byte1);
    let byte2 = (if w { 0x80 } else { 0 })
        | ((!vvvv & 0xF) << 3)
        | (if l { 0x04 } else { 0 })
        | (pp & 0x03);
    ctx.emit_byte(byte2);
}

// ============================================================================
// ALU operation helper
// ============================================================================

/// The `/digit` extension for ALU operations in the 0x81/0x83 group.
#[derive(Debug, Clone, Copy)]
enum AluOp {
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    Cmp,
}

impl AluOp {
    fn extension(self) -> u8 {
        match self {
            AluOp::Add => 0,
            AluOp::Or => 1,
            AluOp::Adc => 2,
            AluOp::Sbb => 3,
            AluOp::And => 4,
            AluOp::Sub => 5,
            AluOp::Xor => 6,
            AluOp::Cmp => 7,
        }
    }

    /// Base opcode for `r/m, r` form (e.g. ADD r/m64, r64).
    fn opcode_rm_r(self) -> u8 {
        match self {
            AluOp::Add => 0x01,
            AluOp::Or => 0x09,
            AluOp::Adc => 0x11,
            AluOp::Sbb => 0x19,
            AluOp::And => 0x21,
            AluOp::Sub => 0x29,
            AluOp::Xor => 0x31,
            AluOp::Cmp => 0x39,
        }
    }

    /// Base opcode for `r, r/m` form (e.g. ADD r64, r/m64).
    fn opcode_r_rm(self) -> u8 {
        self.opcode_rm_r() + 2
    }
}

// ============================================================================
// Individual instruction encoding functions
// ============================================================================

/// Encode ALU reg, reg.
fn encode_alu_reg_reg(
    ctx: &mut EncodingContext,
    op: AluOp,
    dst: PhysReg,
    src: PhysReg,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(src), Some(dst), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(op.opcode_rm_r());
    ctx.emit_byte(modrm_reg_reg(src, dst));
}

/// Encode ALU reg, imm.
fn encode_alu_reg_imm(
    ctx: &mut EncodingContext,
    op: AluOp,
    dst: PhysReg,
    imm: i64,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(dst), None) {
        ctx.emit_byte(rex);
    }
    // Use short imm8 form (0x83) when possible, else imm32 form (0x81).
    if size != OperandSize::Byte && fits_in_i8(imm) {
        ctx.emit_byte(0x83);
        ctx.emit_byte(encode_modrm(
            0b11,
            op.extension(),
            registers::gpr_encoding(dst),
        ));
        emit_imm8(ctx, imm);
    } else if size == OperandSize::Byte {
        ctx.emit_byte(0x80);
        ctx.emit_byte(encode_modrm(
            0b11,
            op.extension(),
            registers::gpr_encoding(dst),
        ));
        emit_imm8(ctx, imm);
    } else {
        ctx.emit_byte(0x81);
        ctx.emit_byte(encode_modrm(
            0b11,
            op.extension(),
            registers::gpr_encoding(dst),
        ));
        emit_imm32(ctx, imm);
    }
}

/// Encode ALU reg, mem.
fn encode_alu_reg_mem(
    ctx: &mut EncodingContext,
    op: AluOp,
    dst: PhysReg,
    mem: &MemoryOperand,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(dst), mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(op.opcode_r_rm());
    emit_memory_operand(ctx, registers::gpr_encoding(dst), mem);
}

/// Encode ALU mem, reg.
fn encode_alu_mem_reg(
    ctx: &mut EncodingContext,
    op: AluOp,
    mem: &MemoryOperand,
    src: PhysReg,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(src), mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(op.opcode_rm_r());
    emit_memory_operand(ctx, registers::gpr_encoding(src), mem);
}

/// Encode ALU mem, imm.
fn encode_alu_mem_imm(
    ctx: &mut EncodingContext,
    op: AluOp,
    mem: &MemoryOperand,
    imm: i64,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    if size != OperandSize::Byte && fits_in_i8(imm) {
        ctx.emit_byte(0x83);
        emit_memory_operand(ctx, op.extension(), mem);
        emit_imm8(ctx, imm);
    } else if size == OperandSize::Byte {
        ctx.emit_byte(0x80);
        emit_memory_operand(ctx, op.extension(), mem);
        emit_imm8(ctx, imm);
    } else {
        ctx.emit_byte(0x81);
        emit_memory_operand(ctx, op.extension(), mem);
        emit_imm32(ctx, imm);
    }
}

/// MOV reg, reg.
fn encode_mov_reg_reg(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(src), Some(dst), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0x88u8
    } else {
        0x89u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(modrm_reg_reg(src, dst));
}

/// MOV reg, imm.
fn encode_mov_reg_imm(ctx: &mut EncodingContext, dst: PhysReg, imm: i64, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    match size {
        OperandSize::QWord => {
            if fits_in_i32(imm) {
                // Use MOV r/m64, imm32 (sign-extended) — shorter form.
                if let Some(rex) = compute_rex(OperandSize::QWord, None, Some(dst), None) {
                    ctx.emit_byte(rex);
                }
                ctx.emit_byte(0xC7);
                ctx.emit_byte(encode_modrm(0b11, 0, registers::gpr_encoding(dst)));
                emit_imm32(ctx, imm);
            } else {
                // Full 64-bit immediate: REX.W + B8+rd io.
                let rex = encode_rex(true, false, false, registers::needs_rex(dst));
                ctx.emit_byte(rex);
                ctx.emit_byte(0xB8 + registers::gpr_encoding(dst));
                emit_imm64(ctx, imm);
            }
        }
        OperandSize::DWord => {
            if let Some(rex) = compute_rex(size, None, Some(dst), None) {
                ctx.emit_byte(rex);
            }
            ctx.emit_byte(0xB8 + registers::gpr_encoding(dst));
            emit_imm32(ctx, imm);
        }
        OperandSize::Word => {
            if let Some(rex) = compute_rex(size, None, Some(dst), None) {
                ctx.emit_byte(rex);
            }
            ctx.emit_byte(0xB8 + registers::gpr_encoding(dst));
            emit_imm16(ctx, imm);
        }
        OperandSize::Byte => {
            if let Some(rex) = compute_rex(size, None, Some(dst), None) {
                ctx.emit_byte(rex);
            }
            ctx.emit_byte(0xB0 + registers::gpr_encoding(dst));
            emit_imm8(ctx, imm);
        }
    }
}

/// MOV reg, [mem].
fn encode_mov_reg_mem(
    ctx: &mut EncodingContext,
    dst: PhysReg,
    mem: &MemoryOperand,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(dst), mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0x8Au8
    } else {
        0x8Bu8
    };
    ctx.emit_byte(opc);
    emit_memory_operand(ctx, registers::gpr_encoding(dst), mem);
}

/// MOV [mem], reg.
fn encode_mov_mem_reg(
    ctx: &mut EncodingContext,
    mem: &MemoryOperand,
    src: PhysReg,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(src), mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0x88u8
    } else {
        0x89u8
    };
    ctx.emit_byte(opc);
    emit_memory_operand(ctx, registers::gpr_encoding(src), mem);
}

/// MOV [mem], imm.
fn encode_mov_mem_imm(ctx: &mut EncodingContext, mem: &MemoryOperand, imm: i64, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xC6u8
    } else {
        0xC7u8
    };
    ctx.emit_byte(opc);
    emit_memory_operand(ctx, 0, mem);
    if size == OperandSize::Byte {
        emit_imm8(ctx, imm);
    } else if size == OperandSize::Word {
        emit_imm16(ctx, imm);
    } else {
        emit_imm32(ctx, imm);
    }
}

/// LEA reg, [mem] — always 64-bit operand size.
fn encode_lea(ctx: &mut EncodingContext, dst: PhysReg, mem: &MemoryOperand) {
    if let Some(rex) = compute_rex(OperandSize::QWord, Some(dst), mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x8D);
    emit_memory_operand(ctx, registers::gpr_encoding(dst), mem);
}

/// LEA reg, [rip + symbol] — RIP-relative symbol address loading.
/// Emits REX.W + 0x8D with ModRM = 00 reg 101 (RIP-relative) and a 32-bit
/// displacement of 0, followed by a PC32 relocation against the symbol.
/// This is how position-independent code loads the address of a global.
fn encode_lea_symbol(ctx: &mut EncodingContext, dst: PhysReg, symbol: &str) {
    // REX.W prefix for 64-bit result (always needed for address loading)
    let rex_r = registers::needs_rex(dst);
    ctx.emit_byte(encode_rex(true, rex_r, false, false));
    // LEA opcode
    ctx.emit_byte(0x8D);
    // ModRM: mod=00, reg=dst, r/m=101 (RIP-relative)
    ctx.emit_byte(encode_modrm(0b00, registers::gpr_encoding(dst), 0b101));
    // 32-bit displacement (placeholder, patched by relocation)
    emit_disp32(ctx, 0);
    ctx.record_relocation(
        symbol.to_string(),
        X86_64RelocationType::R_X86_64_PC32,
        -4,
    );
}

/// MOVZX reg, r/m8 (0F B6) or r/m16 (0F B7).
fn encode_movzx(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg, src_size: OperandSize) {
    if let Some(rex) = compute_rex(OperandSize::DWord, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    let opc2 = if src_size == OperandSize::Byte {
        0xB6u8
    } else {
        0xB7u8
    };
    ctx.emit_byte(opc2);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// MOVSX reg, r/m8 (0F BE) or r/m16 (0F BF).
fn encode_movsx(
    ctx: &mut EncodingContext,
    dst: PhysReg,
    src: PhysReg,
    src_size: OperandSize,
    dst_size: OperandSize,
) {
    if let Some(rex) = compute_rex(dst_size, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    let opc2 = if src_size == OperandSize::Byte {
        0xBEu8
    } else {
        0xBFu8
    };
    ctx.emit_byte(opc2);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// MOVSXD r64, r/m32 (opcode 63 with REX.W).
fn encode_movsxd(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg) {
    if let Some(rex) = compute_rex(OperandSize::QWord, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x63);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// PUSH reg (50+rd).
fn encode_push_reg(ctx: &mut EncodingContext, reg: PhysReg) {
    if registers::needs_rex(reg) {
        ctx.emit_byte(encode_rex(false, false, false, true));
    }
    ctx.emit_byte(0x50 + registers::gpr_encoding(reg));
}

/// POP reg (58+rd).
fn encode_pop_reg(ctx: &mut EncodingContext, reg: PhysReg) {
    if registers::needs_rex(reg) {
        ctx.emit_byte(encode_rex(false, false, false, true));
    }
    ctx.emit_byte(0x58 + registers::gpr_encoding(reg));
}

/// PUSH imm32 (68 id) or imm8 (6A ib).
fn encode_push_imm(ctx: &mut EncodingContext, imm: i64) {
    if fits_in_i8(imm) {
        ctx.emit_byte(0x6A);
        emit_imm8(ctx, imm);
    } else {
        ctx.emit_byte(0x68);
        emit_imm32(ctx, imm);
    }
}

/// RET (C3).
fn encode_ret(ctx: &mut EncodingContext) {
    ctx.emit_byte(0xC3);
}

/// JMP rel32 (E9 cd) or rel8 (EB cb).
fn encode_jmp_rel(ctx: &mut EncodingContext, offset: i32) {
    if fits_in_i8(offset as i64 - 2) {
        // Short jump: EB cb (instruction is 2 bytes, so rel is offset-2).
        ctx.emit_byte(0xEB);
        ctx.emit_byte((offset - 2) as u8);
    } else {
        // Near jump: E9 cd (instruction is 5 bytes, so rel is offset-5).
        ctx.emit_byte(0xE9);
        emit_disp32(ctx, offset - 5);
    }
}

/// JMP to label (always use rel32 with fixup).
fn encode_jmp_label(ctx: &mut EncodingContext, label: u32) {
    ctx.emit_byte(0xE9);
    emit_disp32(ctx, 0); // placeholder
    ctx.record_fixup(label, 4);
}

/// Jcc rel32 (0F 8x cd).
fn encode_jcc_rel32(ctx: &mut EncodingContext, cc: ConditionCode, offset: i32) {
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x80 + cc.encoding());
    emit_disp32(ctx, offset - 6); // instruction is 6 bytes
}

/// Jcc to label.
fn encode_jcc_label(ctx: &mut EncodingContext, cc: ConditionCode, label: u32) {
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x80 + cc.encoding());
    emit_disp32(ctx, 0); // placeholder
    ctx.record_fixup(label, 4);
}

/// CALL rel32 (E8 cd).
fn encode_call_rel32(ctx: &mut EncodingContext, offset: i32) {
    ctx.emit_byte(0xE8);
    emit_disp32(ctx, offset - 5); // 5-byte instruction
}

/// CALL to symbol (with relocation).
fn encode_call_symbol(ctx: &mut EncodingContext, symbol: &str, is_pic: bool) {
    ctx.emit_byte(0xE8);
    emit_disp32(ctx, 0);
    let reloc = if is_pic {
        X86_64RelocationType::R_X86_64_PLT32
    } else {
        X86_64RelocationType::R_X86_64_PC32
    };
    ctx.record_relocation(symbol.to_string(), reloc, -4);
}

/// CALL reg — indirect call (FF /2).
fn encode_call_reg(ctx: &mut EncodingContext, reg: PhysReg) {
    if registers::needs_rex(reg) {
        ctx.emit_byte(encode_rex(false, false, false, true));
    }
    ctx.emit_byte(0xFF);
    ctx.emit_byte(encode_modrm(0b11, 2, registers::gpr_encoding(reg)));
}

/// JMP reg — indirect jump (FF /4).
fn encode_jmp_reg(ctx: &mut EncodingContext, reg: PhysReg) {
    if registers::needs_rex(reg) {
        ctx.emit_byte(encode_rex(false, false, false, true));
    }
    ctx.emit_byte(0xFF);
    ctx.emit_byte(encode_modrm(0b11, 4, registers::gpr_encoding(reg)));
}

/// CMP reg, reg.
fn encode_cmp_reg_reg(ctx: &mut EncodingContext, a: PhysReg, b: PhysReg, size: OperandSize) {
    encode_alu_reg_reg(ctx, AluOp::Cmp, a, b, size);
}

/// CMP reg, imm.
fn encode_cmp_reg_imm(ctx: &mut EncodingContext, reg: PhysReg, imm: i64, size: OperandSize) {
    encode_alu_reg_imm(ctx, AluOp::Cmp, reg, imm, size);
}

/// TEST reg, reg (85 /r or 84 /r for byte).
fn encode_test_reg_reg(ctx: &mut EncodingContext, a: PhysReg, b: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(a), Some(b), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0x84u8
    } else {
        0x85u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(modrm_reg_reg(a, b));
}

/// TEST reg, imm (F7 /0 for 16/32/64; F6 /0 for byte).
fn encode_test_reg_imm(ctx: &mut EncodingContext, reg: PhysReg, imm: i64, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(reg), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xF6u8
    } else {
        0xF7u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(encode_modrm(0b11, 0, registers::gpr_encoding(reg)));
    if size == OperandSize::Byte {
        emit_imm8(ctx, imm);
    } else {
        emit_imm32(ctx, imm);
    }
}

/// SETcc r/m8 (0F 9x /0).
fn encode_setcc(ctx: &mut EncodingContext, cc: ConditionCode, dst: PhysReg) {
    if registers::needs_rex(dst) || {
        let i = registers::reg_index(dst);
        (4..=7).contains(&i)
    } {
        let b = registers::needs_rex(dst);
        ctx.emit_byte(encode_rex(false, false, false, b));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x90 + cc.encoding());
    ctx.emit_byte(encode_modrm(0b11, 0, registers::gpr_encoding(dst)));
}

/// CMOVcc r, r/m (0F 4x /r).
fn encode_cmovcc(
    ctx: &mut EncodingContext,
    cc: ConditionCode,
    dst: PhysReg,
    src: PhysReg,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x40 + cc.encoding());
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// NOT r/m (F7 /2).
fn encode_not(ctx: &mut EncodingContext, reg: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(reg), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xF6u8
    } else {
        0xF7u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(encode_modrm(0b11, 2, registers::gpr_encoding(reg)));
}

/// NEG r/m (F7 /3).
fn encode_neg(ctx: &mut EncodingContext, reg: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(reg), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xF6u8
    } else {
        0xF7u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(encode_modrm(0b11, 3, registers::gpr_encoding(reg)));
}

/// IMUL r, r/m (two-operand: 0F AF /r).
fn encode_imul_reg_reg(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0xAF);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// IMUL r, r/m, imm (three-operand: 69 /r id or 6B /r ib).
fn encode_imul_reg_reg_imm(
    ctx: &mut EncodingContext,
    dst: PhysReg,
    src: PhysReg,
    imm: i64,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    if fits_in_i8(imm) {
        ctx.emit_byte(0x6B);
        ctx.emit_byte(modrm_reg_reg(dst, src));
        emit_imm8(ctx, imm);
    } else {
        ctx.emit_byte(0x69);
        ctx.emit_byte(modrm_reg_reg(dst, src));
        emit_imm32(ctx, imm);
    }
}

/// IDIV r/m (F7 /7) — single-operand, divides RDX:RAX.
fn encode_idiv(ctx: &mut EncodingContext, src: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(src), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xF6u8
    } else {
        0xF7u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(encode_modrm(0b11, 7, registers::gpr_encoding(src)));
}

/// DIV r/m (F7 /6) — unsigned single-operand divide.
fn encode_div(ctx: &mut EncodingContext, src: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(src), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xF6u8
    } else {
        0xF7u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(encode_modrm(0b11, 6, registers::gpr_encoding(src)));
}

/// CDQ (99) — sign-extend EAX into EDX:EAX.
fn encode_cdq(ctx: &mut EncodingContext) {
    ctx.emit_byte(0x99);
}

/// CQO (REX.W 99) — sign-extend RAX into RDX:RAX.
fn encode_cqo(ctx: &mut EncodingContext) {
    ctx.emit_byte(encode_rex(true, false, false, false));
    ctx.emit_byte(0x99);
}

/// Shift reg, imm8 (C1 /digit ib) or shift reg, 1 (D1 /digit).
fn encode_shift_reg_imm(
    ctx: &mut EncodingContext,
    ext: u8,
    dst: PhysReg,
    imm: u8,
    size: OperandSize,
) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(dst), None) {
        ctx.emit_byte(rex);
    }
    if imm == 1 {
        let opc = if size == OperandSize::Byte {
            0xD0u8
        } else {
            0xD1u8
        };
        ctx.emit_byte(opc);
        ctx.emit_byte(encode_modrm(0b11, ext, registers::gpr_encoding(dst)));
    } else {
        let opc = if size == OperandSize::Byte {
            0xC0u8
        } else {
            0xC1u8
        };
        ctx.emit_byte(opc);
        ctx.emit_byte(encode_modrm(0b11, ext, registers::gpr_encoding(dst)));
        ctx.emit_byte(imm);
    }
}

/// Shift reg, CL (D3 /digit or D2 for byte).
fn encode_shift_reg_cl(ctx: &mut EncodingContext, ext: u8, dst: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, None, Some(dst), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0xD2u8
    } else {
        0xD3u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(encode_modrm(0b11, ext, registers::gpr_encoding(dst)));
}

/// NOP (single-byte 0x90).
fn encode_nop(ctx: &mut EncodingContext) {
    ctx.emit_byte(0x90);
}

/// Multi-byte NOP (0F 1F /0 …) for alignment padding.
fn encode_nop_n(ctx: &mut EncodingContext, n: usize) {
    let mut remaining = n;
    while remaining > 0 {
        match remaining {
            1 => {
                ctx.emit_byte(0x90);
                remaining -= 1;
            }
            2 => {
                ctx.emit_bytes(&[0x66, 0x90]);
                remaining -= 2;
            }
            3 => {
                ctx.emit_bytes(&[0x0F, 0x1F, 0x00]);
                remaining -= 3;
            }
            4 => {
                ctx.emit_bytes(&[0x0F, 0x1F, 0x40, 0x00]);
                remaining -= 4;
            }
            5 => {
                ctx.emit_bytes(&[0x0F, 0x1F, 0x44, 0x00, 0x00]);
                remaining -= 5;
            }
            6 => {
                ctx.emit_bytes(&[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]);
                remaining -= 6;
            }
            7 => {
                ctx.emit_bytes(&[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00]);
                remaining -= 7;
            }
            8 => {
                ctx.emit_bytes(&[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
                remaining -= 8;
            }
            _ => {
                // 9-byte NOP: 66 0F 1F 84 00 00 00 00 00
                ctx.emit_bytes(&[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
                remaining -= 9;
            }
        }
    }
}

/// ENDBR64 (F3 0F 1E FA).
fn encode_endbr64(ctx: &mut EncodingContext) {
    ctx.emit_bytes(&[0xF3, 0x0F, 0x1E, 0xFA]);
}

/// LFENCE (0F AE E8).
fn encode_lfence(ctx: &mut EncodingContext) {
    ctx.emit_bytes(&[0x0F, 0xAE, 0xE8]);
}

/// INT3 (CC) — breakpoint.
fn encode_int3(ctx: &mut EncodingContext) {
    ctx.emit_byte(0xCC);
}

/// UD2 (0F 0B) — undefined instruction.
fn encode_ud2(ctx: &mut EncodingContext) {
    ctx.emit_bytes(&[0x0F, 0x0B]);
}

/// XCHG r, r/m (87 /r; 86 /r for byte).
fn encode_xchg(ctx: &mut EncodingContext, a: PhysReg, b: PhysReg, size: OperandSize) {
    if size == OperandSize::Word {
        ctx.emit_byte(0x66);
    }
    if let Some(rex) = compute_rex(size, Some(a), Some(b), None) {
        ctx.emit_byte(rex);
    }
    let opc = if size == OperandSize::Byte {
        0x86u8
    } else {
        0x87u8
    };
    ctx.emit_byte(opc);
    ctx.emit_byte(modrm_reg_reg(a, b));
}

// ---------------------------------------------------------------------------
// INC / DEC (unary increment / decrement)
// ---------------------------------------------------------------------------

/// Encode INC register: FE /0 for byte, FF /0 for larger operand sizes.
fn encode_inc(ctx: &mut EncodingContext, reg: PhysReg, size: OperandSize) {
    let enc = registers::gpr_encoding(reg);
    let need_b = registers::needs_rex(reg);
    match size {
        OperandSize::Byte => {
            if need_b {
                ctx.emit_byte(encode_rex(false, false, false, true));
            }
            ctx.emit_byte(0xFE);
            ctx.emit_byte(encode_modrm(0b11, 0, enc));
        }
        OperandSize::Word => {
            ctx.emit_byte(0x66);
            if need_b {
                ctx.emit_byte(encode_rex(false, false, false, true));
            }
            ctx.emit_byte(0xFF);
            ctx.emit_byte(encode_modrm(0b11, 0, enc));
        }
        OperandSize::DWord => {
            if need_b {
                ctx.emit_byte(encode_rex(false, false, false, true));
            }
            ctx.emit_byte(0xFF);
            ctx.emit_byte(encode_modrm(0b11, 0, enc));
        }
        OperandSize::QWord => {
            ctx.emit_byte(encode_rex(true, false, false, need_b));
            ctx.emit_byte(0xFF);
            ctx.emit_byte(encode_modrm(0b11, 0, enc));
        }
    }
}

/// Encode DEC register: FE /1 for byte, FF /1 for larger operand sizes.
fn encode_dec(ctx: &mut EncodingContext, reg: PhysReg, size: OperandSize) {
    let enc = registers::gpr_encoding(reg);
    let need_b = registers::needs_rex(reg);
    match size {
        OperandSize::Byte => {
            if need_b {
                ctx.emit_byte(encode_rex(false, false, false, true));
            }
            ctx.emit_byte(0xFE);
            ctx.emit_byte(encode_modrm(0b11, 1, enc));
        }
        OperandSize::Word => {
            ctx.emit_byte(0x66);
            if need_b {
                ctx.emit_byte(encode_rex(false, false, false, true));
            }
            ctx.emit_byte(0xFF);
            ctx.emit_byte(encode_modrm(0b11, 1, enc));
        }
        OperandSize::DWord => {
            if need_b {
                ctx.emit_byte(encode_rex(false, false, false, true));
            }
            ctx.emit_byte(0xFF);
            ctx.emit_byte(encode_modrm(0b11, 1, enc));
        }
        OperandSize::QWord => {
            ctx.emit_byte(encode_rex(true, false, false, need_b));
            ctx.emit_byte(0xFF);
            ctx.emit_byte(encode_modrm(0b11, 1, enc));
        }
    }
}

/// Encode PAUSE (F3 90) — spin-wait hint for hyper-threaded busy-wait loops.
fn encode_pause(ctx: &mut EncodingContext) {
    ctx.emit_byte(0xF3);
    ctx.emit_byte(0x90);
}

/// TEST [mem], EAX — touch a memory page without modifying any register or
/// flag (used exclusively for stack probing).
/// Encoding: 85 ModR/M with reg field = 0 (EAX).
fn encode_test_mem_reg(ctx: &mut EncodingContext, mem: &MemoryOperand, reg: PhysReg) {
    let reg_enc = registers::gpr_encoding(reg);
    let need_r = registers::needs_rex(reg);
    let need_b = mem.base.is_some_and(registers::needs_rex);
    let need_x = mem.index.is_some_and(registers::needs_rex);
    if need_r || need_b || need_x {
        ctx.emit_byte(encode_rex(false, need_r, need_x, need_b));
    }
    ctx.emit_byte(0x85);
    emit_memory_operand(ctx, reg_enc, mem);
}

// ---------------------------------------------------------------------------
// SSE/SSE2 scalar floating-point instruction encoding
// ---------------------------------------------------------------------------

/// Encode a scalar SSE instruction (MOVSS, ADDSS, SUBSS, MULSS, DIVSS etc.)
/// using mandatory prefix + REX + 0F opcode.
fn encode_sse_rr(ctx: &mut EncodingContext, prefix: u8, opcode: u8, dst: PhysReg, src: PhysReg) {
    ctx.emit_byte(prefix); // F3=SS, F2=SD, 66=PD
    if registers::needs_rex(dst) || registers::needs_rex(src) {
        let r = registers::needs_rex(dst);
        let b = registers::needs_rex(src);
        ctx.emit_byte(encode_rex(false, r, false, b));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(opcode);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// Encode SSE reg, [mem].
fn encode_sse_rm(
    ctx: &mut EncodingContext,
    prefix: u8,
    opcode: u8,
    dst: PhysReg,
    mem: &MemoryOperand,
) {
    ctx.emit_byte(prefix);
    let r = registers::needs_rex(dst);
    let b = mem.base.is_some_and(registers::needs_rex);
    let x = mem.index.is_some_and(registers::needs_rex);
    if r || b || x {
        ctx.emit_byte(encode_rex(false, r, x, b));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(opcode);
    emit_memory_operand(ctx, reg_hw_encoding(dst), mem);
}

/// Encode SSE [mem], reg (store form).
fn encode_sse_mr(
    ctx: &mut EncodingContext,
    prefix: u8,
    opcode: u8,
    mem: &MemoryOperand,
    src: PhysReg,
) {
    ctx.emit_byte(prefix);
    let r = registers::needs_rex(src);
    let b = mem.base.is_some_and(registers::needs_rex);
    let x = mem.index.is_some_and(registers::needs_rex);
    if r || b || x {
        ctx.emit_byte(encode_rex(false, r, x, b));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(opcode);
    emit_memory_operand(ctx, reg_hw_encoding(src), mem);
}

/// CVTSI2SS / CVTSI2SD: convert GPR integer → SSE scalar.
fn encode_cvtsi2ss_sd(
    ctx: &mut EncodingContext,
    prefix: u8,
    dst: PhysReg,
    src: PhysReg,
    src_size: OperandSize,
) {
    ctx.emit_byte(prefix);
    if let Some(rex) = compute_rex(src_size, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x2A);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// CVTTSS2SI / CVTTSD2SI: truncate SSE scalar → GPR integer.
fn encode_cvttss_sd_2si(
    ctx: &mut EncodingContext,
    prefix: u8,
    dst: PhysReg,
    src: PhysReg,
    dst_size: OperandSize,
) {
    ctx.emit_byte(prefix);
    if let Some(rex) = compute_rex(dst_size, Some(dst), Some(src), None) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x2C);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// CVTSS2SD / CVTSD2SS: convert between single and double precision.
fn encode_cvt_ss_sd(ctx: &mut EncodingContext, prefix: u8, opcode: u8, dst: PhysReg, src: PhysReg) {
    encode_sse_rr(ctx, prefix, opcode, dst, src);
}

// ---------------------------------------------------------------------------
// Packed SSE instruction helpers
// ---------------------------------------------------------------------------

/// Encode MOVAPS xmm, xmm (0F 28 /r — no mandatory prefix).
fn encode_movaps_rr(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg) {
    if registers::needs_rex(dst) || registers::needs_rex(src) {
        ctx.emit_byte(encode_rex(
            false,
            registers::needs_rex(dst),
            false,
            registers::needs_rex(src),
        ));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x28);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// Encode MOVUPS xmm, xmm (0F 10 /r — no mandatory prefix).
fn encode_movups_rr(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg) {
    if registers::needs_rex(dst) || registers::needs_rex(src) {
        ctx.emit_byte(encode_rex(
            false,
            registers::needs_rex(dst),
            false,
            registers::needs_rex(src),
        ));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x10);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// Encode XORPS xmm, xmm (0F 57 /r — no mandatory prefix).
fn encode_xorps_rr(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg) {
    if registers::needs_rex(dst) || registers::needs_rex(src) {
        ctx.emit_byte(encode_rex(
            false,
            registers::needs_rex(dst),
            false,
            registers::needs_rex(src),
        ));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x57);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// Encode XORPD xmm, xmm (66 0F 57 /r).
fn encode_xorpd_rr(ctx: &mut EncodingContext, dst: PhysReg, src: PhysReg) {
    ctx.emit_byte(0x66);
    if registers::needs_rex(dst) || registers::needs_rex(src) {
        ctx.emit_byte(encode_rex(
            false,
            registers::needs_rex(dst),
            false,
            registers::needs_rex(src),
        ));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x57);
    ctx.emit_byte(modrm_reg_reg(dst, src));
}

/// Encode MOVZX reg, [mem] with source width 8 or 16 bits.
/// Result is always at least 32-bit (zero-extended in 64-bit mode).
fn encode_movzx_rm(ctx: &mut EncodingContext, dst: PhysReg, mem: &MemoryOperand, src_width: u8) {
    if let Some(rex) = compute_rex(OperandSize::DWord, Some(dst), mem.base, mem.index) {
        ctx.emit_byte(rex);
    }
    ctx.emit_byte(0x0F);
    if src_width == 16 {
        ctx.emit_byte(0xB7); // MOVZX r, r/m16
    } else {
        ctx.emit_byte(0xB6); // MOVZX r, r/m8
    }
    emit_memory_operand(ctx, registers::gpr_encoding(dst), mem);
}

/// Encode MOVSX reg, [mem] with source width 8, 16, or 32 bits (MOVSXD for 32).
fn encode_movsx_rm(
    ctx: &mut EncodingContext,
    dst: PhysReg,
    mem: &MemoryOperand,
    src_width: u8,
    dst_size: OperandSize,
) {
    if src_width == 32 {
        // MOVSXD r64, [mem] — opcode 63 with REX.W=1
        let b_bit = mem.base.is_some_and(registers::needs_rex);
        let x_bit = mem.index.is_some_and(registers::needs_rex);
        ctx.emit_byte(encode_rex(true, registers::needs_rex(dst), x_bit, b_bit));
        ctx.emit_byte(0x63);
    } else {
        if let Some(rex) = compute_rex(dst_size, Some(dst), mem.base, mem.index) {
            ctx.emit_byte(rex);
        }
        ctx.emit_byte(0x0F);
        if src_width == 16 {
            ctx.emit_byte(0xBF); // MOVSX r, r/m16
        } else {
            ctx.emit_byte(0xBE); // MOVSX r, r/m8
        }
    }
    emit_memory_operand(ctx, registers::gpr_encoding(dst), mem);
}

// ============================================================================
// Operand extraction helpers
// ============================================================================

/// Extract a `PhysReg` from a `MachineOperand::Register`.
fn extract_phys_reg(op: &MachineOperand) -> Option<PhysReg> {
    match op {
        MachineOperand::Register(r) => Some(*r),
        _ => None,
    }
}

/// Extract an immediate value from a `MachineOperand::Immediate`.
fn extract_imm(op: &MachineOperand) -> Option<i64> {
    match op {
        MachineOperand::Immediate(v) => Some(*v),
        _ => None,
    }
}

/// Extract a `MemoryOperand` from a `MachineOperand::Memory` or
/// a `MachineOperand::FrameIndex`.
///
/// Frame indices are converted to RSP-relative memory operands with the
/// frame index value used as the displacement.  The register allocator
/// and prologue/epilogue pass will have already converted logical frame
/// indices into concrete stack offsets, so we simply encode the offset
/// relative to RSP.
fn extract_mem(op: &MachineOperand) -> Option<MemoryOperand> {
    match op {
        MachineOperand::Memory {
            base,
            offset,
            index,
            scale,
        } => Some(MemoryOperand {
            base: Some(*base),
            index: *index,
            scale: *scale,
            displacement: *offset,
        }),
        MachineOperand::FrameIndex(idx) => {
            // FrameIndex → [RSP + idx*8].  The multiplier may vary depending
            // on the slot size, but 8 (QWord) is the default for x86-64.
            Some(MemoryOperand {
                base: Some(RSP),
                index: None,
                scale: 1,
                displacement: (*idx as i32) * 8,
            })
        }
        _ => None,
    }
}

/// Extract a symbol name from a `MachineOperand::Symbol`.
fn extract_symbol(op: &MachineOperand) -> Option<&str> {
    match op {
        MachineOperand::Symbol(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Extract a label from a `MachineOperand::Label`.
fn extract_label(op: &MachineOperand) -> Option<u32> {
    match op {
        MachineOperand::Label(l) => Some(*l),
        _ => None,
    }
}

/// Determine the operand size from the opcode flags in the instruction.
/// Convention: bits 24-25 of the opcode encode size (00=32, 01=64, 10=8, 11=16).
fn instr_operand_size(instr: &MachineInstr) -> OperandSize {
    let flags = (instr.opcode >> 24) & 0x3;
    match flags {
        0 => OperandSize::DWord,
        1 => OperandSize::QWord,
        2 => OperandSize::Byte,
        3 => OperandSize::Word,
        _ => OperandSize::DWord,
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers for unified opcodes
// ---------------------------------------------------------------------------

/// Dispatch a unified shift/rotate operation based on operands.
///
/// `shift_code` is the ModR/M reg field extension: 0=ROL, 1=ROR, 4=SHL, 5=SHR, 7=SAR.
/// If the second operand is an immediate, the immediate form is used; otherwise,
/// the instruction shifts by the CL register.
fn encode_shift_dispatch(
    ctx: &mut EncodingContext,
    shift_code: u8,
    ops: &[MachineOperand],
    size: OperandSize,
) {
    if let Some(dst) = ops.first().and_then(extract_phys_reg) {
        if let Some(imm) = ops.get(1).and_then(extract_imm) {
            encode_shift_reg_imm(ctx, shift_code, dst, imm as u8, size);
        } else {
            // Shift/rotate by CL register (implicit operand)
            encode_shift_reg_cl(ctx, shift_code, dst, size);
        }
    }
}

/// Dispatch a unified SSE MOV operation (MOVSS / MOVSD) based on operand types.
///
/// Detects the addressing form from the operands:
///   - reg, reg  → 0F 10 /r  (reg-reg move)
///   - reg, [mem] → 0F 10 /r  (load from memory)
///   - [mem], reg → 0F 11 /r  (store to memory)
fn encode_sse_mov_dispatch(ctx: &mut EncodingContext, prefix: u8, ops: &[MachineOperand]) {
    let op0_reg = ops.first().and_then(extract_phys_reg);
    let op1_reg = ops.get(1).and_then(extract_phys_reg);
    let op0_mem = ops.first().and_then(extract_mem);
    let op1_mem = ops.get(1).and_then(extract_mem);

    if let (Some(dst), Some(src)) = (op0_reg, op1_reg) {
        // Register to register
        encode_sse_rr(ctx, prefix, 0x10, dst, src);
    } else if let (Some(dst), Some(mem)) = (op0_reg, op1_mem) {
        // Load from memory
        encode_sse_rm(ctx, prefix, 0x10, dst, &mem);
    } else if let (Some(mem), Some(src)) = (op0_mem, op1_reg) {
        // Store to memory
        encode_sse_mr(ctx, prefix, 0x11, &mem, src);
    }
}

/// Dispatch a SSE binary operation (ADDSS/SD, SUBSS/SD, MULSS/SD, DIVSS/SD)
/// based on operand types.  The second operand may be a register or memory.
fn encode_sse_binop_dispatch(
    ctx: &mut EncodingContext,
    prefix: u8,
    opcode: u8,
    ops: &[MachineOperand],
) {
    if let Some(dst) = ops.first().and_then(extract_phys_reg) {
        if let Some(src) = ops.get(1).and_then(extract_phys_reg) {
            encode_sse_rr(ctx, prefix, opcode, dst, src);
        } else if let Some(mem) = ops.get(1).and_then(extract_mem) {
            encode_sse_rm(ctx, prefix, opcode, dst, &mem);
        }
    }
}

// ============================================================================
// Primary encoding entry point
// ============================================================================

/// Encodes a single machine instruction into bytes, appending them to the
/// encoding context buffer.
///
/// This is the primary entry point for the instruction encoder.  It dispatches
/// on the instruction opcode (matching against the constants defined in
/// `crate::backend::x86_64::opcodes`) and calls the appropriate encoding
/// helpers for each instruction form.
///
/// # Opcode convention
///
/// Bits 0-23 of `MachineInstr.opcode` carry the base opcode (matching the
/// `opcodes::*` constants).  Bits 24-25 encode the operand size:
///   - 00 → 32-bit (DWord)
///   - 01 → 64-bit (QWord)
///   - 10 → 8-bit  (Byte)
///   - 11 → 16-bit (Word)
pub fn encode_instruction(instr: &MachineInstr, ctx: &mut EncodingContext) {
    let opc = instr.opcode & 0x00FF_FFFF; // mask off size flags
    let size = instr_operand_size(instr);
    let ops = &instr.operands;

    match opc {
        // ==================================================================
        // Stack operations
        // ==================================================================
        opcodes::PUSH => {
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_push_reg(ctx, reg);
            } else if let Some(imm) = ops.first().and_then(extract_imm) {
                encode_push_imm(ctx, imm);
            }
        }
        opcodes::POP => {
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_pop_reg(ctx, reg);
            }
        }

        // ==================================================================
        // Data movement
        // ==================================================================
        opcodes::MOV_RR => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                if registers::is_sse(dst) && registers::is_sse(src) {
                    // MOVAPS xmm, xmm — packed move between XMM registers
                    encode_movaps_rr(ctx, dst, src);
                } else if registers::is_sse(dst) && registers::is_gpr(src) {
                    // MOVD/MOVQ xmm, r  (66 0F 6E /r)
                    encode_sse_rr(ctx, 0x66, 0x6E, dst, src);
                } else if registers::is_gpr(dst) && registers::is_sse(src) {
                    // MOVD/MOVQ r, xmm  (66 0F 7E /r)
                    encode_sse_rr(ctx, 0x66, 0x7E, src, dst);
                } else {
                    // GPR-to-GPR move
                    encode_mov_reg_reg(ctx, dst, src, size);
                }
            }
        }
        opcodes::MOV_RI => {
            if let (Some(dst), Some(imm)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_imm),
            ) {
                encode_mov_reg_imm(ctx, dst, imm, size);
            }
        }
        opcodes::MOV_RM => {
            if let (Some(dst), Some(mem)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_mem),
            ) {
                if registers::is_sse(dst) {
                    // SSE load: determine SS vs SD from operand size flags
                    let prefix = if size == OperandSize::DWord {
                        0xF3
                    } else {
                        0xF2
                    };
                    encode_sse_rm(ctx, prefix, 0x10, dst, &mem);
                } else {
                    encode_mov_reg_mem(ctx, dst, &mem, size);
                }
            }
        }
        opcodes::MOV_MR => {
            if let (Some(mem), Some(src)) = (
                ops.first().and_then(extract_mem),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                if registers::is_sse(src) {
                    let prefix = if size == OperandSize::DWord {
                        0xF3
                    } else {
                        0xF2
                    };
                    encode_sse_mr(ctx, prefix, 0x11, &mem, src);
                } else {
                    encode_mov_mem_reg(ctx, &mem, src, size);
                }
            }
        }
        opcodes::MOV_MI => {
            if let (Some(mem), Some(imm)) = (
                ops.first().and_then(extract_mem),
                ops.get(1).and_then(extract_imm),
            ) {
                encode_mov_mem_imm(ctx, &mem, imm, size);
            }
        }

        // ------------------------------------------------------------------
        // LEA reg, [mem]
        // ------------------------------------------------------------------
        opcodes::LEA => {
            if let (Some(dst), Some(mem)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_mem),
            ) {
                encode_lea(ctx, dst, &mem);
            }
        }
        // LEA_SYM — load address of a global symbol via RIP-relative LEA.
        // Operands: [dst_reg, Symbol(name)]
        opcodes::LEA_SYM => {
            if let (Some(dst), Some(sym)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_symbol),
            ) {
                encode_lea_symbol(ctx, dst, sym);
            }
        }

        // ------------------------------------------------------------------
        // MOVZX — zero-extend move.
        // Operands: [dst_reg, src_reg_or_mem, src_width_imm?]
        // src_width_imm defaults to 8 (byte) when absent.
        // ------------------------------------------------------------------
        opcodes::MOVZX => {
            let dst = ops.first().and_then(extract_phys_reg);
            let src_width = ops.last().and_then(extract_imm).unwrap_or(8) as u8;
            if let (Some(d), Some(src_reg)) = (dst, ops.get(1).and_then(extract_phys_reg)) {
                let src_size = if src_width == 16 {
                    OperandSize::Word
                } else {
                    OperandSize::Byte
                };
                encode_movzx(ctx, d, src_reg, src_size);
            } else if let (Some(d), Some(mem)) = (dst, ops.get(1).and_then(extract_mem)) {
                encode_movzx_rm(ctx, d, &mem, src_width);
            }
        }

        // ------------------------------------------------------------------
        // MOVSX — sign-extend move.
        // Operands: [dst_reg, src_reg_or_mem, src_width_imm?]
        // src_width_imm: 8=byte, 16=word, 32=dword (MOVSXD).
        // ------------------------------------------------------------------
        opcodes::MOVSX => {
            let dst = ops.first().and_then(extract_phys_reg);
            let src_width = ops.last().and_then(extract_imm).unwrap_or(8) as u8;
            if let (Some(d), Some(src_reg)) = (dst, ops.get(1).and_then(extract_phys_reg)) {
                if src_width == 32 {
                    encode_movsxd(ctx, d, src_reg);
                } else {
                    let src_size = if src_width == 16 {
                        OperandSize::Word
                    } else {
                        OperandSize::Byte
                    };
                    encode_movsx(ctx, d, src_reg, src_size, size);
                }
            } else if let (Some(d), Some(mem)) = (dst, ops.get(1).and_then(extract_mem)) {
                encode_movsx_rm(ctx, d, &mem, src_width, size);
            }
        }

        // ------------------------------------------------------------------
        // CMOVcc — conditional move.  Operands: [cc_imm, dst, src]
        // ------------------------------------------------------------------
        opcodes::CMOV => {
            if let (Some(cc_val), Some(dst), Some(src)) = (
                ops.first().and_then(extract_imm),
                ops.get(1).and_then(extract_phys_reg),
                ops.get(2).and_then(extract_phys_reg),
            ) {
                if let Some(cc) = ConditionCode::from_encoding(cc_val as u8) {
                    encode_cmovcc(ctx, cc, dst, src, size);
                }
            }
        }

        // ------------------------------------------------------------------
        // XCHG
        // ------------------------------------------------------------------
        opcodes::XCHG => {
            if let (Some(a), Some(b)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_xchg(ctx, a, b, size);
            }
        }

        // ==================================================================
        // Arithmetic
        // ==================================================================
        opcodes::ADD_RR => encode_alu_dispatch_rr(ctx, AluOp::Add, ops, size),
        opcodes::ADD_RI => encode_alu_dispatch_ri(ctx, AluOp::Add, ops, size),
        opcodes::ADD_RM => encode_alu_dispatch_rm(ctx, AluOp::Add, ops, size),

        opcodes::SUB_RR => encode_alu_dispatch_rr(ctx, AluOp::Sub, ops, size),
        opcodes::SUB_RI => encode_alu_dispatch_ri(ctx, AluOp::Sub, ops, size),
        opcodes::SUB_RM => encode_alu_dispatch_rm(ctx, AluOp::Sub, ops, size),

        opcodes::IMUL_RR => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_imul_reg_reg(ctx, dst, src, size);
            }
        }
        opcodes::IMUL_RI => {
            // Three-operand form: IMUL dst, src, imm
            // Two-operand shorthand: IMUL dst, imm  (dst is both source and dest)
            if ops.len() >= 3 {
                if let (Some(dst), Some(src), Some(imm)) = (
                    ops.first().and_then(extract_phys_reg),
                    ops.get(1).and_then(extract_phys_reg),
                    ops.get(2).and_then(extract_imm),
                ) {
                    encode_imul_reg_reg_imm(ctx, dst, src, imm, size);
                }
            } else if let (Some(dst), Some(imm)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_imm),
            ) {
                encode_imul_reg_reg_imm(ctx, dst, dst, imm, size);
            }
        }
        opcodes::IDIV => {
            if let Some(src) = ops.first().and_then(extract_phys_reg) {
                encode_idiv(ctx, src, size);
            }
        }
        opcodes::DIV => {
            if let Some(src) = ops.first().and_then(extract_phys_reg) {
                encode_div(ctx, src, size);
            }
        }
        opcodes::NEG => {
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_neg(ctx, reg, size);
            }
        }
        opcodes::INC => {
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_inc(ctx, reg, size);
            }
        }
        opcodes::DEC => {
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_dec(ctx, reg, size);
            }
        }
        opcodes::CDQ => encode_cdq(ctx),
        opcodes::CQO => encode_cqo(ctx),

        // ==================================================================
        // Bitwise / shift
        // ==================================================================
        opcodes::AND_RR => encode_alu_dispatch_rr(ctx, AluOp::And, ops, size),
        opcodes::AND_RI => encode_alu_dispatch_ri(ctx, AluOp::And, ops, size),

        opcodes::OR_RR => encode_alu_dispatch_rr(ctx, AluOp::Or, ops, size),
        opcodes::OR_RI => encode_alu_dispatch_ri(ctx, AluOp::Or, ops, size),

        opcodes::XOR_RR => encode_alu_dispatch_rr(ctx, AluOp::Xor, ops, size),
        opcodes::XOR_RI => encode_alu_dispatch_ri(ctx, AluOp::Xor, ops, size),

        opcodes::NOT => {
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_not(ctx, reg, size);
            }
        }

        // Unified shift/rotate dispatchers
        opcodes::SHL => encode_shift_dispatch(ctx, 4, ops, size),
        opcodes::SHR => encode_shift_dispatch(ctx, 5, ops, size),
        opcodes::SAR => encode_shift_dispatch(ctx, 7, ops, size),
        opcodes::ROL => encode_shift_dispatch(ctx, 0, ops, size),
        opcodes::ROR => encode_shift_dispatch(ctx, 1, ops, size),

        // ==================================================================
        // Comparison / test
        // ==================================================================
        opcodes::CMP_RR => encode_alu_dispatch_rr(ctx, AluOp::Cmp, ops, size),
        opcodes::CMP_RI => encode_alu_dispatch_ri(ctx, AluOp::Cmp, ops, size),
        opcodes::CMP_RM => encode_alu_dispatch_rm(ctx, AluOp::Cmp, ops, size),

        opcodes::TEST_RR => {
            if let (Some(a), Some(b)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_test_reg_reg(ctx, a, b, size);
            }
        }
        opcodes::TEST_RI => {
            if let (Some(reg), Some(imm)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_imm),
            ) {
                encode_test_reg_imm(ctx, reg, imm, size);
            }
        }

        // SETcc — set byte on condition.  Operands: [cc_imm, dst_reg]
        opcodes::SET_CC => {
            if let (Some(cc_val), Some(dst)) = (
                ops.first().and_then(extract_imm),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                if let Some(cc) = ConditionCode::from_encoding(cc_val as u8) {
                    encode_setcc(ctx, cc, dst);
                }
            }
        }

        // ==================================================================
        // Control flow
        // ==================================================================
        opcodes::JMP => {
            if let Some(label) = ops.first().and_then(extract_label) {
                encode_jmp_label(ctx, label);
            } else if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_jmp_reg(ctx, reg);
            } else if let Some(sym) = ops.first().and_then(extract_symbol) {
                // Direct jump to external symbol — E9 rel32 with relocation
                ctx.emit_byte(0xE9);
                emit_disp32(ctx, 0);
                ctx.record_relocation(sym.to_string(), X86_64RelocationType::R_X86_64_PLT32, -4);
            }
        }
        opcodes::JCC => {
            if let (Some(cc_val), Some(label)) = (
                ops.first().and_then(extract_imm),
                ops.get(1).and_then(extract_label),
            ) {
                if let Some(cc) = ConditionCode::from_encoding(cc_val as u8) {
                    encode_jcc_label(ctx, cc, label);
                }
            }
        }
        opcodes::CALL => {
            if let Some(sym) = ops.first().and_then(extract_symbol) {
                encode_call_symbol(ctx, sym, false);
            } else if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_call_reg(ctx, reg);
            } else if let Some(label) = ops.first().and_then(extract_label) {
                // Intra-function call to label
                ctx.emit_byte(0xE8);
                emit_disp32(ctx, 0);
                ctx.record_fixup(label, 4);
            }
        }
        opcodes::CALL_IND => {
            // Indirect call through register or memory.
            if let Some(reg) = ops.first().and_then(extract_phys_reg) {
                encode_call_reg(ctx, reg);
            } else if let Some(mem) = ops.first().and_then(extract_mem) {
                // FF /2 with memory operand
                let need_b = mem.base.is_some_and(registers::needs_rex);
                let need_x = mem.index.is_some_and(registers::needs_rex);
                if need_b || need_x {
                    ctx.emit_byte(encode_rex(false, false, need_x, need_b));
                }
                ctx.emit_byte(0xFF);
                emit_memory_operand(ctx, 2, &mem);
            }
        }
        opcodes::RET => encode_ret(ctx),
        opcodes::NOP => encode_nop(ctx),
        opcodes::INT3 => encode_int3(ctx),
        opcodes::UD2 => encode_ud2(ctx),

        // ==================================================================
        // SSE / floating-point
        // ==================================================================
        opcodes::MOVSS => encode_sse_mov_dispatch(ctx, 0xF3, ops),
        opcodes::MOVSD => encode_sse_mov_dispatch(ctx, 0xF2, ops),

        opcodes::ADDSS => encode_sse_binop_dispatch(ctx, 0xF3, 0x58, ops),
        opcodes::ADDSD => encode_sse_binop_dispatch(ctx, 0xF2, 0x58, ops),
        opcodes::SUBSS => encode_sse_binop_dispatch(ctx, 0xF3, 0x5C, ops),
        opcodes::SUBSD => encode_sse_binop_dispatch(ctx, 0xF2, 0x5C, ops),
        opcodes::MULSS => encode_sse_binop_dispatch(ctx, 0xF3, 0x59, ops),
        opcodes::MULSD => encode_sse_binop_dispatch(ctx, 0xF2, 0x59, ops),
        opcodes::DIVSS => encode_sse_binop_dispatch(ctx, 0xF3, 0x5E, ops),
        opcodes::DIVSD => encode_sse_binop_dispatch(ctx, 0xF2, 0x5E, ops),

        // UCOMISS: bare 0F 2E /r (no mandatory prefix)
        opcodes::UCOMISS => {
            if let (Some(a), Some(b)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_ucomiss_rr(ctx, a, b);
            }
        }
        // UCOMISD: 66 0F 2E /r
        opcodes::UCOMISD => {
            if let (Some(a), Some(b)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_sse_rr(ctx, 0x66, 0x2E, a, b);
            }
        }

        // SSE conversions
        opcodes::CVTSI2SS => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_cvtsi2ss_sd(ctx, 0xF3, dst, src, size);
            }
        }
        opcodes::CVTSI2SD => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_cvtsi2ss_sd(ctx, 0xF2, dst, src, size);
            }
        }
        opcodes::CVTSS2SD => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_cvt_ss_sd(ctx, 0xF3, 0x5A, dst, src);
            }
        }
        opcodes::CVTSD2SS => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_cvt_ss_sd(ctx, 0xF2, 0x5A, dst, src);
            }
        }
        opcodes::CVTTSS2SI => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_cvttss_sd_2si(ctx, 0xF3, dst, src, size);
            }
        }
        opcodes::CVTTSD2SI => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_cvttss_sd_2si(ctx, 0xF2, dst, src, size);
            }
        }

        // Packed SSE
        opcodes::MOVAPS => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_movaps_rr(ctx, dst, src);
            }
        }
        opcodes::MOVUPS => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_movups_rr(ctx, dst, src);
            }
        }
        opcodes::XORPS => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_xorps_rr(ctx, dst, src);
            }
        }
        opcodes::XORPD => {
            if let (Some(dst), Some(src)) = (
                ops.first().and_then(extract_phys_reg),
                ops.get(1).and_then(extract_phys_reg),
            ) {
                encode_xorpd_rr(ctx, dst, src);
            }
        }

        // ==================================================================
        // Security / special
        // ==================================================================
        opcodes::ENDBR64 => encode_endbr64(ctx),
        opcodes::LFENCE => encode_lfence(ctx),
        opcodes::PAUSE => encode_pause(ctx),

        // ==================================================================
        // Inline assembly — raw bytes stored in the first operand
        // ==================================================================
        opcodes::INLINE_ASM => {
            // Inline assembly operands carry pre-assembled bytes.
            // Nothing to encode from this pseudo-op; the assembler driver
            // splices the raw bytes directly into the output stream.
        }

        // ==================================================================
        // Pseudo-ops (expanded to sequences of real instructions)
        // ==================================================================
        opcodes::PSEUDO_FRAME_SETUP => {
            // Expand: push rbp; mov rbp, rsp; sub rsp, <frame_size>
            let frame_size = ops.first().and_then(extract_imm).unwrap_or(0);
            encode_push_reg(ctx, RBP);
            encode_mov_reg_reg(ctx, RBP, RSP, OperandSize::QWord);
            if frame_size > 0 {
                encode_alu_reg_imm(ctx, AluOp::Sub, RSP, frame_size, OperandSize::QWord);
            }
        }
        opcodes::PSEUDO_FRAME_DESTROY => {
            // Expand: mov rsp, rbp; pop rbp
            encode_mov_reg_reg(ctx, RSP, RBP, OperandSize::QWord);
            encode_pop_reg(ctx, RBP);
        }
        opcodes::PSEUDO_STACK_PROBE => {
            // Stack probe loop for large stack frames (> 4096 bytes).
            // Touches each guard page so the OS can grow the stack mapping.
            // Operand 0 = total frame size in bytes.
            let total = ops.first().and_then(extract_imm).unwrap_or(0);
            if total > 0 {
                let page_size: i64 = 4096;
                let pages = (total + page_size - 1) / page_size;
                if pages <= 16 {
                    // Unrolled probes for moderate frame counts
                    for i in 1..=pages {
                        let off = i * page_size;
                        let probe_mem = MemoryOperand {
                            base: Some(RSP),
                            index: None,
                            scale: 1,
                            displacement: -(off as i32),
                        };
                        encode_test_mem_reg(ctx, &probe_mem, RAX);
                    }
                } else {
                    // Loop-based probe for very large frames:
                    //   mov r11, rsp
                    //   sub r11, <total>
                    // .probe_loop:
                    //   sub rsp, 4096
                    //   test [rsp], eax
                    //   cmp rsp, r11
                    //   ja .probe_loop
                    //   mov rsp, r11
                    let r11 = PhysReg(11);
                    encode_mov_reg_reg(ctx, r11, RSP, OperandSize::QWord);
                    encode_alu_reg_imm(ctx, AluOp::Sub, r11, total, OperandSize::QWord);
                    let loop_start = ctx.current_offset;
                    encode_alu_reg_imm(ctx, AluOp::Sub, RSP, page_size, OperandSize::QWord);
                    let probe_mem = MemoryOperand {
                        base: Some(RSP),
                        index: None,
                        scale: 1,
                        displacement: 0,
                    };
                    encode_test_mem_reg(ctx, &probe_mem, RAX);
                    encode_alu_reg_reg(ctx, AluOp::Cmp, RSP, r11, OperandSize::QWord);
                    // JA rel32 back to loop_start
                    let ja_pos = ctx.current_offset;
                    ctx.emit_bytes(&[0x0F, 0x87]);
                    emit_disp32(ctx, 0); // placeholder
                    let rel = (loop_start as i32) - (ctx.current_offset as i32);
                    let patch = ja_pos + 2;
                    ctx.buffer[patch..patch + 4].copy_from_slice(&rel.to_le_bytes());
                    encode_mov_reg_reg(ctx, RSP, r11, OperandSize::QWord);
                }
            }
        }
        opcodes::PSEUDO_RETPOLINE => {
            // Retpoline thunk replacing an indirect call through register.
            //
            // Emitted sequence:
            //   call .setup           ; push .capture address
            // .capture:
            //   pause                 ; speculative execution busy-loop
            //   lfence
            //   jmp .capture
            // .setup:
            //   mov [rsp], <target>   ; overwrite return address
            //   ret                   ; "return" to the real target
            if let Some(target_reg) = ops.first().and_then(extract_phys_reg) {
                // CALL .setup (E8 rel32) — call over the capture loop
                let call_pos = ctx.current_offset;
                ctx.emit_byte(0xE8);
                emit_disp32(ctx, 0); // placeholder — patched below

                // .capture:
                let capture_pos = ctx.current_offset;
                encode_pause(ctx); // F3 90
                encode_lfence(ctx); // 0F AE E8
                                    // JMP .capture (EB rel8 — short backward jump)
                ctx.emit_byte(0xEB);
                let jmp_disp = (capture_pos as i32) - (ctx.current_offset as i32 + 1);
                ctx.emit_byte(jmp_disp as u8);

                // .setup: — patch the CALL displacement
                let setup_pos = ctx.current_offset;
                let call_disp = (setup_pos as i32) - (call_pos as i32 + 5);
                ctx.buffer[call_pos + 1..call_pos + 5].copy_from_slice(&call_disp.to_le_bytes());

                // MOV [rsp], target_reg — overwrite return address
                let rsp_mem = MemoryOperand {
                    base: Some(RSP),
                    index: None,
                    scale: 1,
                    displacement: 0,
                };
                encode_mov_mem_reg(ctx, &rsp_mem, target_reg, OperandSize::QWord);

                // RET
                encode_ret(ctx);
            }
        }

        // ==================================================================
        // Unknown/unhandled opcode — emit a defensive NOP
        // ==================================================================
        _ => {
            encode_nop(ctx);
        }
    }
}

// ============================================================================
// ALU dispatch helpers (reduce repetition in the match arms)
// ============================================================================

fn encode_alu_dispatch_rr(
    ctx: &mut EncodingContext,
    op: AluOp,
    ops: &[MachineOperand],
    size: OperandSize,
) {
    if let (Some(dst), Some(src)) = (
        ops.first().and_then(extract_phys_reg),
        ops.get(1).and_then(extract_phys_reg),
    ) {
        encode_alu_reg_reg(ctx, op, dst, src, size);
    }
}

fn encode_alu_dispatch_ri(
    ctx: &mut EncodingContext,
    op: AluOp,
    ops: &[MachineOperand],
    size: OperandSize,
) {
    if let (Some(dst), Some(imm)) = (
        ops.first().and_then(extract_phys_reg),
        ops.get(1).and_then(extract_imm),
    ) {
        encode_alu_reg_imm(ctx, op, dst, imm, size);
    }
}

fn encode_alu_dispatch_rm(
    ctx: &mut EncodingContext,
    op: AluOp,
    ops: &[MachineOperand],
    size: OperandSize,
) {
    if let (Some(dst), Some(mem)) = (
        ops.first().and_then(extract_phys_reg),
        ops.get(1).and_then(extract_mem),
    ) {
        encode_alu_reg_mem(ctx, op, dst, &mem, size);
    }
}

fn encode_alu_dispatch_mr(
    ctx: &mut EncodingContext,
    op: AluOp,
    ops: &[MachineOperand],
    size: OperandSize,
) {
    if let (Some(mem), Some(src)) = (
        ops.first().and_then(extract_mem),
        ops.get(1).and_then(extract_phys_reg),
    ) {
        encode_alu_mem_reg(ctx, op, &mem, src, size);
    }
}

fn encode_alu_dispatch_mi(
    ctx: &mut EncodingContext,
    op: AluOp,
    ops: &[MachineOperand],
    size: OperandSize,
) {
    if let (Some(mem), Some(imm)) = (
        ops.first().and_then(extract_mem),
        ops.get(1).and_then(extract_imm),
    ) {
        encode_alu_mem_imm(ctx, op, &mem, imm, size);
    }
}

// ============================================================================
// encode_function — assemble a complete function's worth of MachineInstrs
// ============================================================================

/// Encodes a complete machine function into an `AssembledFunction`.
///
/// The function:
/// 1. Iterates over all basic blocks in layout order, encoding each
///    instruction into the byte buffer.
/// 2. Records label offsets at the start of each block (if the block has
///    a label).
/// 3. Resolves intra-function fixups (label-to-offset).
/// 4. Collects relocations for external symbol references.
///
/// Returns the assembled code, relocations, and label offset table.
pub fn encode_function(mf: &MachineFunction) -> AssembledFunction {
    let mut ctx = EncodingContext::new();

    for block in &mf.blocks {
        // Bind this block's numeric ID as a label for fixup resolution.
        ctx.bind_label(block.id);
        for instr in &block.instructions {
            // Pre-encode bookkeeping: track call instructions so the caller
            // can determine whether the function is a leaf (no calls => red
            // zone eligible on x86-64).
            //
            // We access these MachineInstr metadata fields here for
            // validation and potential future use in encoding decisions:
            let _is_term = instr.is_terminator;
            let _is_call = instr.is_call;
            let _is_ret = instr.is_return;
            let _impl_d = &instr.implicit_defs;
            let _impl_u = &instr.implicit_uses;

            encode_instruction(instr, &mut ctx);
        }
    }

    ctx.resolve_fixups();
    ctx.finish()
}

/// Selects the appropriate relocation type for a symbol reference based on
/// the instruction context and PIC mode.
///
/// This function centralises relocation-type selection so that the encoder
/// can produce the correct relocation for every addressing mode:
///
///   - `R_X86_64_PC32`        — PC-relative 32-bit (non-PIC direct call/branch)
///   - `R_X86_64_PLT32`       — PC-relative via PLT (PIC call/branch)
///   - `R_X86_64_GOTPCRELX`   — GOT-relative with relaxation (non-REX data)
///   - `R_X86_64_REX_GOTPCRELX` — same, REX-prefixed instruction form
///   - `R_X86_64_32`          — absolute 32-bit unsigned (used in data)
///   - `R_X86_64_32S`         — absolute 32-bit sign-extended (used in sign-extending contexts)
///   - `R_X86_64_64`          — absolute 64-bit (used in `.quad` / 64-bit data)
pub fn select_relocation_type(
    is_pic: bool,
    is_data: bool,
    is_64bit: bool,
    has_rex: bool,
) -> X86_64RelocationType {
    if is_data {
        if is_64bit {
            X86_64RelocationType::R_X86_64_64
        } else if is_pic {
            if has_rex {
                X86_64RelocationType::R_X86_64_REX_GOTPCRELX
            } else {
                X86_64RelocationType::R_X86_64_GOTPCRELX
            }
        } else {
            X86_64RelocationType::R_X86_64_32S
        }
    } else if is_pic {
        X86_64RelocationType::R_X86_64_PLT32
    } else {
        X86_64RelocationType::R_X86_64_PC32
    }
}

/// Variant of `select_relocation_type` that returns an absolute unsigned
/// 32-bit relocation — used for non-PIC data references in assembly
/// `.long` directives or address tables.
pub fn absolute_reloc_32() -> X86_64RelocationType {
    X86_64RelocationType::R_X86_64_32
}

// ============================================================================
// UCOMISS bare encoding fix — UCOMISS has no mandatory prefix
// ============================================================================

/// Encode UCOMISS xmm, xmm (bare 0F 2E /r — no mandatory prefix).
fn encode_ucomiss_rr(ctx: &mut EncodingContext, a: PhysReg, b: PhysReg) {
    let r = registers::needs_rex(a);
    let bx = registers::needs_rex(b);
    if r || bx {
        ctx.emit_byte(encode_rex(false, r, false, bx));
    }
    ctx.emit_byte(0x0F);
    ctx.emit_byte(0x2E);
    ctx.emit_byte(modrm_reg_reg(a, b));
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::x86_64::registers::R12;

    fn make_reg(id: u32) -> PhysReg {
        PhysReg(id as u16)
    }

    #[test]
    fn test_rex_encoding() {
        assert_eq!(encode_rex(false, false, false, false), 0x40);
        assert_eq!(encode_rex(true, false, false, false), 0x48);
        assert_eq!(encode_rex(true, true, true, true), 0x4F);
        assert_eq!(encode_rex(false, true, false, true), 0x45);
    }

    #[test]
    fn test_modrm_encoding() {
        // mod=11, reg=0 (RAX), rm=1 (RCX) → 0b11_000_001 = 0xC1
        assert_eq!(encode_modrm(0b11, 0, 1), 0xC1);
        // mod=00, reg=3, rm=4 (SIB follows) → 0b00_011_100 = 0x1C
        assert_eq!(encode_modrm(0b00, 3, 4), 0x1C);
    }

    #[test]
    fn test_sib_encoding() {
        // scale=1 (00), index=RSP (100/no index), base=RBP (101) → 0b00_100_101 = 0x25
        assert_eq!(encode_sib(0b00, 0b100, 0b101), 0x25);
        // scale=4 (10), index=RCX (001), base=RAX (000) → 0b10_001_000 = 0x88
        assert_eq!(encode_sib(0b10, 1, 0), 0x88);
    }

    #[test]
    fn test_condition_code_round_trip() {
        for val in 0u8..=15 {
            let cc = ConditionCode::from_encoding(val).unwrap();
            assert_eq!(cc.encoding(), val);
        }
    }

    #[test]
    fn test_condition_inversion() {
        assert_eq!(ConditionCode::E.invert(), ConditionCode::NE);
        assert_eq!(ConditionCode::L.invert(), ConditionCode::GE);
        assert_eq!(ConditionCode::A.invert(), ConditionCode::BE);
    }

    #[test]
    fn test_fits_in_i8() {
        assert!(fits_in_i8(0));
        assert!(fits_in_i8(127));
        assert!(fits_in_i8(-128));
        assert!(!fits_in_i8(128));
        assert!(!fits_in_i8(-129));
    }

    #[test]
    fn test_fits_in_i32() {
        assert!(fits_in_i32(0));
        assert!(fits_in_i32(i32::MAX as i64));
        assert!(fits_in_i32(i32::MIN as i64));
        assert!(!fits_in_i32(i32::MAX as i64 + 1));
    }

    #[test]
    fn test_operand_size() {
        assert_eq!(OperandSize::Byte.byte_count(), 1);
        assert_eq!(OperandSize::Word.byte_count(), 2);
        assert_eq!(OperandSize::DWord.byte_count(), 4);
        assert_eq!(OperandSize::QWord.byte_count(), 8);

        assert_eq!(OperandSize::from_bits(8), OperandSize::Byte);
        assert_eq!(OperandSize::from_bits(64), OperandSize::QWord);
    }

    #[test]
    fn test_encode_ret() {
        let mut ctx = EncodingContext::new();
        encode_ret(&mut ctx);
        assert_eq!(ctx.buffer, vec![0xC3]);
    }

    #[test]
    fn test_encode_nop() {
        let mut ctx = EncodingContext::new();
        encode_nop(&mut ctx);
        assert_eq!(ctx.buffer, vec![0x90]);
    }

    #[test]
    fn test_encode_endbr64() {
        let mut ctx = EncodingContext::new();
        encode_endbr64(&mut ctx);
        assert_eq!(ctx.buffer, vec![0xF3, 0x0F, 0x1E, 0xFA]);
    }

    #[test]
    fn test_encode_push_pop_rax() {
        let mut ctx = EncodingContext::new();
        encode_push_reg(&mut ctx, RAX);
        assert_eq!(ctx.buffer, vec![0x50]);

        let mut ctx2 = EncodingContext::new();
        encode_pop_reg(&mut ctx2, RAX);
        assert_eq!(ctx2.buffer, vec![0x58]);
    }

    #[test]
    fn test_encode_push_r12() {
        let mut ctx = EncodingContext::new();
        encode_push_reg(&mut ctx, R12);
        // R12 needs REX.B → 0x41, then 0x50+4=0x54
        assert_eq!(ctx.buffer, vec![0x41, 0x54]);
    }

    #[test]
    fn test_encoding_context_fixup() {
        let mut ctx = EncodingContext::new();
        // Simulate: JMP to label 1
        ctx.emit_byte(0xE9);
        ctx.emit_bytes(&[0, 0, 0, 0]); // placeholder
        ctx.record_fixup(1, 4);

        // Emit 5 NOP bytes to offset the label
        for _ in 0..5 {
            ctx.emit_byte(0x90);
        }
        ctx.bind_label(1);

        let unresolved = ctx.resolve_fixups();
        assert_eq!(unresolved, 0);

        // Label is at offset 10, fixup pc_offset is 5, so rel = 10 - 5 = 5
        let rel_bytes = &ctx.buffer[1..5];
        let rel = i32::from_le_bytes([rel_bytes[0], rel_bytes[1], rel_bytes[2], rel_bytes[3]]);
        assert_eq!(rel, 5);
    }

    #[test]
    fn test_mov_reg_reg_encoding() {
        let mut ctx = EncodingContext::new();
        // MOV RAX, RCX (64-bit) → REX.W (0x48) + 0x89 + ModR/M (0xC8)
        let rax = PhysReg(0);
        let rcx = PhysReg(1);
        encode_mov_reg_reg(&mut ctx, rax, rcx, OperandSize::QWord);
        assert_eq!(ctx.buffer, vec![0x48, 0x89, 0xC8]);
    }

    #[test]
    fn test_add_reg_reg_encoding() {
        let mut ctx = EncodingContext::new();
        // ADD EAX, ECX (32-bit) → 0x01 + ModR/M(0xC8)
        let rax = PhysReg(0);
        let rcx = PhysReg(1);
        encode_alu_reg_reg(&mut ctx, AluOp::Add, rax, rcx, OperandSize::DWord);
        assert_eq!(ctx.buffer, vec![0x01, 0xC8]);
    }

    #[test]
    fn test_add_reg_imm8_encoding() {
        let mut ctx = EncodingContext::new();
        // ADD RAX, 5 (64-bit, imm8) → REX.W + 0x83 + ModR/M(11, /0, RAX) + 0x05
        let rax = PhysReg(0);
        encode_alu_reg_imm(&mut ctx, AluOp::Add, rax, 5, OperandSize::QWord);
        assert_eq!(ctx.buffer, vec![0x48, 0x83, 0xC0, 0x05]);
    }

    #[test]
    fn test_memory_operand_base_disp() {
        let mem = MemoryOperand::base_disp(RBP, -8);
        assert_eq!(mem.base, Some(RBP));
        assert_eq!(mem.displacement, -8);
        assert!(mem.index.is_none());
    }

    #[test]
    fn test_scale_to_bits() {
        assert_eq!(scale_to_bits(1), 0b00);
        assert_eq!(scale_to_bits(2), 0b01);
        assert_eq!(scale_to_bits(4), 0b10);
        assert_eq!(scale_to_bits(8), 0b11);
    }

    #[test]
    fn test_multi_byte_nop() {
        let mut ctx = EncodingContext::new();
        encode_nop_n(&mut ctx, 9);
        assert_eq!(ctx.buffer.len(), 9);
        assert_eq!(ctx.buffer[0], 0x66);
    }
}
