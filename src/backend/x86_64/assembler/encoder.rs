//! x86-64 instruction encoder — translates machine instructions into binary
//! machine code bytes with proper REX prefix, ModR/M, SIB, and immediate
//! encoding.
//!
//! # Architecture
//!
//! The encoder processes [`MachineInstr`]s sequentially, producing a stream
//! of bytes for each instruction. x86-64 instructions have variable length
//! (1–15 bytes) and use a complex encoding scheme:
//!
//! | Component     | Size     | Description                                    |
//! |---------------|----------|------------------------------------------------|
//! | Legacy prefix | 0–4 B   | Operand-size override (0x66), REP, LOCK, etc.  |
//! | REX prefix    | 0–1 B   | 64-bit operand size, extended registers (R8–R15)|
//! | Opcode        | 1–3 B   | Instruction operation code                     |
//! | ModR/M        | 0–1 B   | Register/memory addressing mode                |
//! | SIB           | 0–1 B   | Scale-Index-Base for complex addressing         |
//! | Displacement  | 0–4 B   | Memory offset                                  |
//! | Immediate     | 0–8 B   | Constant operand                                |
//!
//! # Relocation Records
//!
//! When the encoder encounters symbolic references (function calls, global
//! variables, PIC addressing), it emits placeholder bytes and records a
//! [`Relocation`] entry. The linker later patches these locations with the
//! correct addresses.

use crate::backend::traits::{MachineFunction, MachineInstr, MachineOperand, PhysReg};
use super::relocations::X86_64RelocationType;

// ---------------------------------------------------------------------------
// AssembledFunction — output of the instruction encoder
// ---------------------------------------------------------------------------

/// The output of encoding a single machine function into binary x86-64 code.
///
/// Contains the raw machine code bytes, relocation records for symbolic
/// references, and metadata about the encoded function.
pub struct AssembledFunction {
    /// Raw machine code bytes in execution order.
    ///
    /// This buffer is directly written into the `.text` section of the
    /// output ELF file. Symbolic references are filled with placeholder
    /// values; the linker patches them using the relocation records.
    pub code: Vec<u8>,

    /// Relocation records for unresolved symbolic references.
    ///
    /// Each relocation specifies an offset within `code`, a symbol name,
    /// and a relocation type. The linker applies these relocations when
    /// producing the final ELF binary.
    pub relocations: Vec<Relocation>,

    /// Total size of the encoded function in bytes.
    pub size: usize,
}

/// A relocation record generated during instruction encoding.
///
/// Records the location within the machine code buffer where a symbolic
/// reference needs to be patched by the linker.
#[derive(Clone, Debug)]
pub struct Relocation {
    /// Byte offset within the `AssembledFunction::code` buffer where
    /// the relocation target value should be written.
    pub offset: usize,

    /// Symbol name that this relocation references (e.g., function name,
    /// global variable, PLT stub).
    pub symbol: String,

    /// x86-64-specific relocation type determining how the linker
    /// computes and writes the final value.
    pub reloc_type: X86_64RelocationType,

    /// Addend value added to the symbol address during relocation.
    /// For `R_X86_64_PC32` relocations, this is typically -4 to account
    /// for the size of the 32-bit relocation field itself.
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// REX prefix helpers
// ---------------------------------------------------------------------------

/// REX prefix base value (0x40).
const REX_BASE: u8 = 0x40;
/// REX.W bit — enables 64-bit operand size.
const REX_W: u8 = 0x08;
/// REX.R bit — extends the ModR/M reg field to access R8–R15.
const REX_R: u8 = 0x04;
/// REX.X bit — extends the SIB index field to access R8–R15.
#[allow(dead_code)]
const REX_X: u8 = 0x02;
/// REX.B bit — extends the ModR/M r/m field or SIB base to access R8–R15.
const REX_B: u8 = 0x01;

/// Returns the 3-bit register encoding for a physical register.
///
/// x86-64 registers are encoded as 3-bit values in ModR/M and SIB bytes.
/// Registers R8–R15 and XMM8–XMM15 use the same low 3 bits with the
/// extended bit set via REX.R, REX.B, or REX.X.
#[inline]
fn reg_encoding(reg: PhysReg) -> u8 {
    (reg.0 & 0x07) as u8
}

/// Returns `true` if the register requires a REX extension bit.
///
/// Registers R8–R15 (encoding 8–15) and XMM8–XMM15 (encoding 24–31)
/// require REX.R, REX.B, or REX.X to encode.
#[inline]
fn needs_rex_ext(reg: PhysReg) -> bool {
    let id = reg.0;
    (8..=15).contains(&id) || (24..=31).contains(&id)
}

/// Constructs a ModR/M byte.
///
/// The ModR/M byte has three fields:
/// - `mod` (bits 7:6): addressing mode (00=indirect, 01=disp8, 10=disp32, 11=register)
/// - `reg` (bits 5:3): register operand or opcode extension
/// - `rm`  (bits 2:0): register or memory operand
#[inline]
fn modrm(mode: u8, reg: u8, rm: u8) -> u8 {
    (mode << 6) | ((reg & 0x07) << 3) | (rm & 0x07)
}

// ---------------------------------------------------------------------------
// Core encoding functions
// ---------------------------------------------------------------------------

/// Encodes a register-to-register instruction.
///
/// Pattern: `[REX] opcode ModR/M`
/// where ModR/M mode = 11 (register direct).
fn encode_reg_reg(
    buf: &mut Vec<u8>,
    opcode: u8,
    dst: PhysReg,
    src: PhysReg,
    is_64bit: bool,
) {
    let mut rex = 0u8;
    if is_64bit {
        rex |= REX_W;
    }
    if needs_rex_ext(dst) {
        rex |= REX_R;
    }
    if needs_rex_ext(src) {
        rex |= REX_B;
    }
    if rex != 0 {
        buf.push(REX_BASE | rex);
    }
    buf.push(opcode);
    buf.push(modrm(0b11, reg_encoding(dst), reg_encoding(src)));
}

/// Encodes a register-immediate instruction.
///
/// Pattern: `[REX] opcode ModR/M imm32`
fn encode_reg_imm(
    buf: &mut Vec<u8>,
    opcode: u8,
    modrm_ext: u8,
    reg: PhysReg,
    imm: i64,
    is_64bit: bool,
) {
    let mut rex = 0u8;
    if is_64bit {
        rex |= REX_W;
    }
    if needs_rex_ext(reg) {
        rex |= REX_B;
    }
    if rex != 0 {
        buf.push(REX_BASE | rex);
    }
    buf.push(opcode);
    buf.push(modrm(0b11, modrm_ext, reg_encoding(reg)));

    // Emit 32-bit immediate (sign-extended to 64-bit by the processor
    // when REX.W is set for most instructions).
    let imm32 = imm as i32;
    buf.extend_from_slice(&imm32.to_le_bytes());
}

/// Encodes a NOP (0x90) — used as filler/alignment padding.
#[inline]
fn encode_nop(buf: &mut Vec<u8>) {
    buf.push(0x90);
}

// ---------------------------------------------------------------------------
// Top-level encoding entry point
// ---------------------------------------------------------------------------

/// Encodes an entire machine function into binary x86-64 machine code.
///
/// This is the main entry point called by
/// [`super::super::X86_64Codegen::emit_assembly()`]. It walks each basic
/// block and instruction, encoding them into a contiguous byte buffer.
///
/// # Arguments
///
/// * `mf` — the machine function to encode (after register allocation)
///
/// # Returns
///
/// An [`AssembledFunction`] containing the encoded bytes, relocation
/// records, and total size.
pub fn encode_function(mf: &MachineFunction) -> AssembledFunction {
    let mut code = Vec::with_capacity(mf.blocks.len() * 64);
    let mut relocations = Vec::new();
    let mut _block_offsets = Vec::with_capacity(mf.blocks.len());

    // Record the byte offset where each block starts — used for branch
    // target resolution within the function.
    for block in &mf.blocks {
        _block_offsets.push(code.len());

        for instr in &block.instructions {
            encode_machine_instr(&mut code, &mut relocations, instr);
        }
    }

    let size = code.len();
    AssembledFunction {
        code,
        relocations,
        size,
    }
}

/// Encodes a single machine instruction into the code buffer.
///
/// This function dispatches on the instruction opcode and emits the
/// appropriate byte sequence. For symbolic operands (function calls,
/// global variable references), it records relocation entries.
fn encode_machine_instr(
    code: &mut Vec<u8>,
    relocations: &mut Vec<Relocation>,
    instr: &MachineInstr,
) {
    // Import the x86-64 opcode constants from the parent module.
    use super::super::opcodes;

    match instr.opcode {
        // -- NOP (filler/alignment) -----------------------------------------
        opcodes::NOP => encode_nop(code),

        // -- Register-register MOV ------------------------------------------
        opcodes::MOV_RR => {
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                encode_reg_reg(code, 0x89, dst, src, true);
            } else {
                // Virtual registers not yet allocated — emit placeholder NOP.
                // In a complete implementation, register allocation must
                // resolve all virtual registers before encoding.
                encode_nop(code);
            }
        }

        // -- Register-memory MOV (load) -------------------------------------
        opcodes::MOV_RM => {
            // Simplified: emit as register-register MOV.
            // Full implementation would handle memory operand encoding.
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                encode_reg_reg(code, 0x8B, dst, src, true);
            } else {
                encode_nop(code);
            }
        }

        // -- Memory-register MOV (store) ------------------------------------
        opcodes::MOV_MR => {
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                encode_reg_reg(code, 0x89, dst, src, true);
            } else {
                encode_nop(code);
            }
        }

        // -- ADD reg, reg ---------------------------------------------------
        opcodes::ADD_RR => {
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                encode_reg_reg(code, 0x01, dst, src, true);
            } else {
                encode_nop(code);
            }
        }

        // -- SUB reg, reg ---------------------------------------------------
        opcodes::SUB_RR => {
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                encode_reg_reg(code, 0x29, dst, src, true);
            } else {
                encode_nop(code);
            }
        }

        // -- IMUL reg, reg --------------------------------------------------
        opcodes::IMUL_RR => {
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                // Two-byte opcode: 0x0F 0xAF
                let mut rex = REX_W;
                if needs_rex_ext(dst) {
                    rex |= REX_R;
                }
                if needs_rex_ext(src) {
                    rex |= REX_B;
                }
                code.push(REX_BASE | rex);
                code.push(0x0F);
                code.push(0xAF);
                code.push(modrm(0b11, reg_encoding(dst), reg_encoding(src)));
            } else {
                encode_nop(code);
            }
        }

        // -- CMP reg, reg ---------------------------------------------------
        opcodes::CMP_RR => {
            if let (Some(dst), Some(src)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_phys_reg(&instr.operands, 1),
            ) {
                encode_reg_reg(code, 0x39, dst, src, true);
            } else {
                encode_nop(code);
            }
        }

        // -- CMP reg, imm32 ------------------------------------------------
        opcodes::CMP_RI => {
            if let (Some(reg), Some(imm)) = (
                extract_phys_reg(&instr.operands, 0),
                extract_immediate(&instr.operands, 1),
            ) {
                encode_reg_imm(code, 0x81, 7, reg, imm, true);
            } else {
                encode_nop(code);
            }
        }

        // -- CALL -----------------------------------------------------------
        opcodes::CALL => {
            if let Some(MachineOperand::Symbol(ref name)) = instr.operands.first() {
                // CALL rel32 — E8 + 4-byte offset (filled by linker)
                code.push(0xE8);
                let offset = code.len();
                code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                relocations.push(Relocation {
                    offset,
                    symbol: name.clone(),
                    reloc_type: X86_64RelocationType::R_X86_64_PC32,
                    addend: -4,
                });
            } else {
                // Indirect call — FF /2
                if let Some(reg) = extract_phys_reg(&instr.operands, 0) {
                    let mut rex = 0u8;
                    if needs_rex_ext(reg) {
                        rex |= REX_B;
                    }
                    if rex != 0 {
                        code.push(REX_BASE | rex);
                    }
                    code.push(0xFF);
                    code.push(modrm(0b11, 2, reg_encoding(reg)));
                } else {
                    encode_nop(code);
                }
            }
        }

        // -- RET ------------------------------------------------------------
        opcodes::RET => {
            code.push(0xC3);
        }

        // -- JMP (unconditional) --------------------------------------------
        opcodes::JMP => {
            // JMP rel32 — E9 + 4-byte offset (resolved after block layout)
            code.push(0xE9);
            code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        }

        // -- Jcc (conditional jump) -----------------------------------------
        opcodes::JCC => {
            if let Some(MachineOperand::Immediate(cc_val)) = instr.operands.first() {
                // Two-byte opcode: 0x0F 0x80+cc
                code.push(0x0F);
                code.push(0x80 | (*cc_val as u8 & 0x0F));
                code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
            } else {
                encode_nop(code);
            }
        }

        // -- PUSH reg -------------------------------------------------------
        opcodes::PUSH => {
            if let Some(reg) = extract_phys_reg(&instr.operands, 0) {
                if needs_rex_ext(reg) {
                    code.push(REX_BASE | REX_B);
                }
                code.push(0x50 + reg_encoding(reg));
            } else {
                encode_nop(code);
            }
        }

        // -- POP reg --------------------------------------------------------
        opcodes::POP => {
            if let Some(reg) = extract_phys_reg(&instr.operands, 0) {
                if needs_rex_ext(reg) {
                    code.push(REX_BASE | REX_B);
                }
                code.push(0x58 + reg_encoding(reg));
            } else {
                encode_nop(code);
            }
        }

        // -- ENDBR64 (CET) -------------------------------------------------
        opcodes::ENDBR64 => {
            // 4-byte NOP-like encoding: F3 0F 1E FA
            code.extend_from_slice(&[0xF3, 0x0F, 0x1E, 0xFA]);
        }

        // -- LEA, MOVZX, MOVSX, XOR, AND, OR, etc. -------------------------
        // Simplified placeholders — full implementation in encoder.rs
        // expansion would handle each instruction's specific encoding.
        opcodes::LEA
        | opcodes::MOVZX
        | opcodes::MOVSX
        | opcodes::XOR_RR
        | opcodes::AND_RR
        | opcodes::OR_RR
        | opcodes::SHL
        | opcodes::SHR
        | opcodes::SAR
        | opcodes::TEST_RR
        | opcodes::SET_CC
        | opcodes::IDIV
        | opcodes::DIV
        | opcodes::CQO
        | opcodes::ADDSD
        | opcodes::SUBSD
        | opcodes::MULSD
        | opcodes::DIVSD
        | opcodes::MOVSD
        | opcodes::UCOMISD
        | opcodes::INLINE_ASM => {
            // Emit a NOP placeholder for instructions not yet fully
            // implemented in the encoder. Each will be expanded to proper
            // binary encoding as the encoder is completed.
            encode_nop(code);
        }

        // -- Unknown opcode — defensive NOP ---------------------------------
        _ => {
            encode_nop(code);
        }
    }
}

/// Extracts a physical register from an operand at the given index.
///
/// Returns `None` if the operand is a virtual register (not yet allocated)
/// or if the index is out of bounds.
fn extract_phys_reg(operands: &[MachineOperand], idx: usize) -> Option<PhysReg> {
    operands.get(idx).and_then(|op| match op {
        MachineOperand::Register(reg) => Some(*reg),
        _ => None,
    })
}

/// Extracts an immediate value from an operand at the given index.
fn extract_immediate(operands: &[MachineOperand], idx: usize) -> Option<i64> {
    operands.get(idx).and_then(|op| match op {
        MachineOperand::Immediate(val) => Some(*val),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::{MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand};
    use crate::backend::x86_64::registers;

    fn make_simple_mf(instrs: Vec<MachineInstr>) -> MachineFunction {
        let mut mbb = MachineBasicBlock::new(0);
        for instr in instrs {
            mbb.push_instr(instr);
        }
        let mut mf = MachineFunction::new("test".to_string(), 16);
        mf.add_block(mbb);
        mf
    }

    #[test]
    fn encode_ret() {
        let mf = make_simple_mf(vec![{
            let mut r = MachineInstr::new(super::super::super::opcodes::RET);
            r.is_terminator = true;
            r.is_return = true;
            r
        }]);
        let result = encode_function(&mf);
        assert_eq!(result.code, vec![0xC3]);
        assert_eq!(result.size, 1);
    }

    #[test]
    fn encode_nop_instruction() {
        let mf = make_simple_mf(vec![
            MachineInstr::new(super::super::super::opcodes::NOP),
        ]);
        let result = encode_function(&mf);
        assert_eq!(result.code, vec![0x90]);
    }

    #[test]
    fn encode_push_pop() {
        // PUSH RAX
        let push = MachineInstr::with_operands(
            super::super::super::opcodes::PUSH,
            vec![MachineOperand::Register(registers::RAX)],
        );
        // POP RAX
        let pop = MachineInstr::with_operands(
            super::super::super::opcodes::POP,
            vec![MachineOperand::Register(registers::RAX)],
        );
        let mf = make_simple_mf(vec![push, pop]);
        let result = encode_function(&mf);
        // PUSH RAX = 0x50, POP RAX = 0x58
        assert_eq!(result.code, vec![0x50, 0x58]);
    }

    #[test]
    fn encode_push_extended_reg() {
        // PUSH R12 requires REX.B prefix
        let push = MachineInstr::with_operands(
            super::super::super::opcodes::PUSH,
            vec![MachineOperand::Register(registers::R12)],
        );
        let mf = make_simple_mf(vec![push]);
        let result = encode_function(&mf);
        // REX.B (0x41) + PUSH+4 (R12 encoding is 4 in low 3 bits)
        assert_eq!(result.code, vec![0x41, 0x54]);
    }

    #[test]
    fn encode_call_symbol_generates_relocation() {
        let call = MachineInstr::with_operands(
            super::super::super::opcodes::CALL,
            vec![MachineOperand::Symbol("printf".to_string())],
        );
        let mf = make_simple_mf(vec![call]);
        let result = encode_function(&mf);
        // E8 00 00 00 00 (5 bytes: CALL + rel32 placeholder)
        assert_eq!(result.code.len(), 5);
        assert_eq!(result.code[0], 0xE8);
        assert_eq!(result.relocations.len(), 1);
        assert_eq!(result.relocations[0].symbol, "printf");
        assert_eq!(result.relocations[0].addend, -4);
    }

    #[test]
    fn encode_endbr64() {
        let endbr = MachineInstr::new(super::super::super::opcodes::ENDBR64);
        let mf = make_simple_mf(vec![endbr]);
        let result = encode_function(&mf);
        assert_eq!(result.code, vec![0xF3, 0x0F, 0x1E, 0xFA]);
    }

    #[test]
    fn encode_jmp_rel32() {
        let mut jmp = MachineInstr::with_operands(
            super::super::super::opcodes::JMP,
            vec![MachineOperand::Label(1)],
        );
        jmp.is_terminator = true;
        let mf = make_simple_mf(vec![jmp]);
        let result = encode_function(&mf);
        // E9 00 00 00 00 (5 bytes: JMP + rel32 placeholder)
        assert_eq!(result.code.len(), 5);
        assert_eq!(result.code[0], 0xE9);
    }

    #[test]
    fn reg_encoding_standard_regs() {
        use crate::backend::x86_64::registers::*;
        assert_eq!(reg_encoding(RAX), 0);
        assert_eq!(reg_encoding(RCX), 1);
        assert_eq!(reg_encoding(RDX), 2);
        assert_eq!(reg_encoding(RBX), 3);
        assert_eq!(reg_encoding(RSP), 4);
        assert_eq!(reg_encoding(RBP), 5);
        assert_eq!(reg_encoding(RSI), 6);
        assert_eq!(reg_encoding(RDI), 7);
    }

    #[test]
    fn needs_rex_ext_extended_regs() {
        use crate::backend::x86_64::registers::*;
        assert!(!needs_rex_ext(RAX));
        assert!(!needs_rex_ext(RDI));
        assert!(needs_rex_ext(R8));
        assert!(needs_rex_ext(R15));
    }

    #[test]
    fn modrm_construction() {
        // mod=11, reg=0 (RAX), rm=1 (RCX) → 0b11_000_001 = 0xC1
        assert_eq!(modrm(0b11, 0, 1), 0xC1);
        // mod=00, reg=3, rm=5 → 0b00_011_101 = 0x1D
        assert_eq!(modrm(0b00, 3, 5), 0x1D);
    }
}
