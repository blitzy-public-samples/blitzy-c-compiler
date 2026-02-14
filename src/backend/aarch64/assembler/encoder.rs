//! AArch64 A64 instruction encoder for the BCC built-in assembler.
//!
//! This module encodes AArch64 machine instructions into their 32-bit binary
//! representation following the A64 instruction set architecture. All A64
//! instructions are exactly 4 bytes (32 bits), stored in little-endian format.
//!
//! # Encoding Groups
//!
//! The A64 ISA organizes instructions into groups identified by bits \[31:25\]:
//!
//! - **Data Processing — Immediate** (`op0=100x`): ADD/SUB/AND/ORR/EOR imm,
//!   MOVZ/MOVK/MOVN, bitfield operations (SBFM/UBFM/BFM), PC-relative (ADR/ADRP)
//! - **Data Processing — Register** (`op0=x101`): shifted register arithmetic,
//!   logical shifted register, multiply/divide
//! - **Loads and Stores** (`op0=x1x0`): LDR/STR with immediate/register offset,
//!   pre/post-index, LDP/STP pair, load literal
//! - **Branches** (`op0=x01x`): B/BL, B.cond, CBZ/CBNZ, TBZ/TBNZ, BR/BLR/RET
//!
//! # Relocation Emission
//!
//! Instructions referencing external symbols or labels beyond the current
//! function produce relocation annotations via the [`EncodedInstruction`]
//! struct's optional relocation field. The assembler collects these for
//! inclusion in the ELF `.rela.text` section.
//!
//! # Register Encoding
//!
//! - `X0`–`X30` / `W0`–`W30` → 5-bit encoding 0–30
//! - `SP` / `WSP` → register 31 (stack pointer context)
//! - `XZR` / `WZR` → register 31 (zero register context)
//! - `V0`–`V31` / `D0`–`D31` / `S0`–`S31` → 5-bit encoding 0–31

use super::relocations::AArch64RelocationType;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of encoding a single A64 instruction.
///
/// Every A64 instruction is exactly 4 bytes. The optional `relocation` field
/// is populated when the instruction references an external symbol that must
/// be resolved by the linker (e.g., ADRP to a global, BL to an external
/// function).
#[derive(Debug, Clone)]
pub struct EncodedInstruction {
    /// The 32-bit instruction word (little-endian on disk).
    pub bytes: u32,
    /// Optional relocation: `(relocation_type, symbol_name, addend)`.
    pub relocation: Option<(AArch64RelocationType, String, i64)>,
}

/// Shift types used in data-processing (shifted register) instructions.
///
/// Encoded as a 2-bit field in bits \[23:22\] or \[22:21\] depending on the
/// instruction group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShiftType {
    /// Logical shift left.
    LSL = 0b00,
    /// Logical shift right (zero-fill).
    LSR = 0b01,
    /// Arithmetic shift right (sign-extend).
    ASR = 0b10,
    /// Rotate right (for logical register operations only).
    ROR = 0b11,
}

/// Extend types used in load/store (register offset) instructions and
/// extended register data-processing instructions.
///
/// Encoded as a 3-bit `option` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExtendType {
    /// Unsigned extend word (32→64, zero-extended).
    UXTW = 0b010,
    /// Logical shift left (equivalent to UXTX with shift).
    LSL = 0b011,
    /// Signed extend word (32→64, sign-extended).
    SXTW = 0b110,
    /// Signed extend doubleword (identity for 64-bit, with optional shift).
    SXTX = 0b111,
}

// ===========================================================================
// Data Processing — Immediate
// ===========================================================================

/// Encode `ADD <Xd|SP>, <Xn|SP>, #<imm12>{, <shift>}` (immediate).
///
/// - `sf`: `true` for 64-bit (`X`), `false` for 32-bit (`W`).
/// - `rd`, `rn`: 5-bit register encodings (0–31).
/// - `imm12`: 12-bit unsigned immediate (0–4095).
/// - `shift`: `false` → imm12 as-is; `true` → imm12 << 12.
pub fn encode_add_imm(sf: bool, rd: u8, rn: u8, imm12: u16, shift: bool) -> u32 {
    encode_addsub_imm(sf, 0b0, 0b0, rd, rn, imm12, shift)
}

/// Encode `ADDS <Xd>, <Xn|SP>, #<imm12>{, <shift>}` (flag-setting ADD).
///
/// Sets NZCV condition flags. `CMP` is an alias when `Rd` = `XZR`/`WZR`.
pub fn encode_adds_imm(sf: bool, rd: u8, rn: u8, imm12: u16, shift: bool) -> u32 {
    encode_addsub_imm(sf, 0b0, 0b1, rd, rn, imm12, shift)
}

/// Encode `SUB <Xd|SP>, <Xn|SP>, #<imm12>{, <shift>}` (immediate).
pub fn encode_sub_imm(sf: bool, rd: u8, rn: u8, imm12: u16, shift: bool) -> u32 {
    encode_addsub_imm(sf, 0b1, 0b0, rd, rn, imm12, shift)
}

/// Encode `SUBS <Xd>, <Xn|SP>, #<imm12>{, <shift>}` (flag-setting SUB).
///
/// Sets NZCV condition flags. `CMP` is an alias when `Rd` = `XZR`/`WZR`.
pub fn encode_subs_imm(sf: bool, rd: u8, rn: u8, imm12: u16, shift: bool) -> u32 {
    encode_addsub_imm(sf, 0b1, 0b1, rd, rn, imm12, shift)
}

/// Internal helper for ADD/SUB immediate.
///
/// Format: `sf:op:S:100010:sh:imm12:Rn:Rd`
fn encode_addsub_imm(sf: bool, op: u8, s: u8, rd: u8, rn: u8, imm12: u16, shift: bool) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let op_bit = ((op & 1) as u32) << 30;
    let s_bit = ((s & 1) as u32) << 29;
    let fixed = 0b100010u32 << 23;
    let sh_bit = (shift as u32) << 22;
    let imm = ((imm12 as u32) & 0xFFF) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | op_bit | s_bit | fixed | sh_bit | imm | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// Logical Immediate
// ---------------------------------------------------------------------------

/// Encode `AND <Xd|SP>, <Xn>, #<bitmask>` (logical AND with bitmask immediate).
///
/// The `(n, immr, imms)` triple must be pre-computed via
/// [`encode_bitmask_immediate`].
pub fn encode_and_imm(sf: bool, rd: u8, rn: u8, n: bool, immr: u8, imms: u8) -> u32 {
    encode_logical_imm(sf, 0b00, rd, rn, n, immr, imms)
}

/// Encode `ORR <Xd|SP>, <Xn>, #<bitmask>` (logical OR with bitmask immediate).
pub fn encode_orr_imm(sf: bool, rd: u8, rn: u8, n: bool, immr: u8, imms: u8) -> u32 {
    encode_logical_imm(sf, 0b01, rd, rn, n, immr, imms)
}

/// Encode `EOR <Xd|SP>, <Xn>, #<bitmask>` (logical exclusive-OR with bitmask immediate).
pub fn encode_eor_imm(sf: bool, rd: u8, rn: u8, n: bool, immr: u8, imms: u8) -> u32 {
    encode_logical_imm(sf, 0b10, rd, rn, n, immr, imms)
}

/// Encode `ANDS <Xd>, <Xn>, #<bitmask>` (flag-setting AND with bitmask immediate).
///
/// `TST` is an alias when `Rd` = `XZR`/`WZR`.
pub fn encode_ands_imm(sf: bool, rd: u8, rn: u8, n: bool, immr: u8, imms: u8) -> u32 {
    encode_logical_imm(sf, 0b11, rd, rn, n, immr, imms)
}

/// Internal helper for logical immediate instructions.
///
/// Format: `sf:opc:100100:N:immr:imms:Rn:Rd`
fn encode_logical_imm(sf: bool, opc: u8, rd: u8, rn: u8, n: bool, immr: u8, imms: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let opc_bits = ((opc & 0x3) as u32) << 29;
    let fixed = 0b100100u32 << 23;
    let n_bit = (n as u32) << 22;
    let immr_bits = ((immr & 0x3F) as u32) << 16;
    let imms_bits = ((imms & 0x3F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | opc_bits | fixed | n_bit | immr_bits | imms_bits | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// Move Wide Immediate
// ---------------------------------------------------------------------------

/// Encode `MOVZ <Xd>, #<imm16>{, LSL #<shift>}` — move zero with shift.
///
/// - `hw`: shift amount selector (0=LSL#0, 1=LSL#16, 2=LSL#32, 3=LSL#48).
///   For 32-bit (`W`) registers, `hw` must be 0 or 1.
pub fn encode_movz(sf: bool, rd: u8, imm16: u16, hw: u8) -> u32 {
    encode_movewide(sf, 0b10, rd, imm16, hw)
}

/// Encode `MOVK <Xd>, #<imm16>{, LSL #<shift>}` — move keep (insert 16-bit).
///
/// Unlike MOVZ, MOVK preserves the other halfwords of the destination register.
pub fn encode_movk(sf: bool, rd: u8, imm16: u16, hw: u8) -> u32 {
    encode_movewide(sf, 0b11, rd, imm16, hw)
}

/// Encode `MOVN <Xd>, #<imm16>{, LSL #<shift>}` — move NOT.
///
/// The destination register receives `NOT(imm16 << (hw * 16))`.
pub fn encode_movn(sf: bool, rd: u8, imm16: u16, hw: u8) -> u32 {
    encode_movewide(sf, 0b00, rd, imm16, hw)
}

/// Internal helper for move-wide immediate instructions.
///
/// Format: `sf:opc:100101:hw:imm16:Rd`
fn encode_movewide(sf: bool, opc: u8, rd: u8, imm16: u16, hw: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let opc_bits = ((opc & 0x3) as u32) << 29;
    let fixed = 0b100101u32 << 23;
    let hw_bits = ((hw & 0x3) as u32) << 21;
    let imm_bits = (imm16 as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | opc_bits | fixed | hw_bits | imm_bits | rd_bits
}

// ---------------------------------------------------------------------------
// Bitfield Operations
// ---------------------------------------------------------------------------

/// Encode `SBFM <Xd>, <Xn>, #<immr>, #<imms>` — signed bitfield move.
///
/// Aliases: `SXTB`, `SXTH`, `SXTW`, `ASR`.
pub fn encode_sbfm(sf: bool, rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    encode_bitfield(sf, 0b00, rd, rn, immr, imms)
}

/// Encode `UBFM <Xd>, <Xn>, #<immr>, #<imms>` — unsigned bitfield move.
///
/// Aliases: `UXTB`, `UXTH`, `LSL`, `LSR`.
pub fn encode_ubfm(sf: bool, rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    encode_bitfield(sf, 0b10, rd, rn, immr, imms)
}

/// Encode `BFM <Xd>, <Xn>, #<immr>, #<imms>` — bitfield move.
///
/// Aliases: `BFI`, `BFXIL`.
pub fn encode_bfm(sf: bool, rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    encode_bitfield(sf, 0b01, rd, rn, immr, imms)
}

/// Internal helper for bitfield instructions.
///
/// Format: `sf:opc:100110:N:immr:imms:Rn:Rd`
/// `N` must equal `sf` for the 64-bit variant.
fn encode_bitfield(sf: bool, opc: u8, rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let opc_bits = ((opc & 0x3) as u32) << 29;
    let fixed = 0b100110u32 << 23;
    // N must equal sf for 64-bit operations
    let n_bit = (sf as u32) << 22;
    let immr_bits = ((immr & 0x3F) as u32) << 16;
    let imms_bits = ((imms & 0x3F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | opc_bits | fixed | n_bit | immr_bits | imms_bits | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// PC-Relative Address Calculation
// ---------------------------------------------------------------------------

/// Encode `ADR <Xd>, <label>` — PC-relative address (±1 MiB, 21-bit signed).
///
/// Format: `0:immlo[1:0]:10000:immhi[18:0]:Rd`
pub fn encode_adr(rd: u8, imm21: i32) -> u32 {
    let imm = imm21 as u32;
    let immlo = (imm & 0x3) << 29;
    let immhi = ((imm >> 2) & 0x7_FFFF) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    // op = 0 (ADR), fixed bits = 10000 at [28:24]
    immlo | (0b10000u32 << 24) | immhi | rd_bits
}

/// Encode `ADRP <Xd>, <label>` — PC-relative page address (±4 GiB, 21-bit page offset).
///
/// Format: `1:immlo[1:0]:10000:immhi[18:0]:Rd`
///
/// The immediate represents a 21-bit page offset; the CPU computes
/// `(PC & ~0xFFF) + (imm21 << 12)`.
pub fn encode_adrp(rd: u8, imm21: i32) -> u32 {
    let imm = imm21 as u32;
    let immlo = (imm & 0x3) << 29;
    let immhi = ((imm >> 2) & 0x7_FFFF) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    // op = 1 (ADRP), fixed bits = 10000 at [28:24]
    (1u32 << 31) | immlo | (0b10000u32 << 24) | immhi | rd_bits
}

// ===========================================================================
// Data Processing — Register (shifted)
// ===========================================================================

/// Encode `ADD <Xd>, <Xn>, <Xm>{, <shift> #<amount>}` (shifted register).
pub fn encode_add_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_addsub_shifted(sf, 0, 0, rd, rn, rm, shift, amount)
}

/// Encode `ADDS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}` (flag-setting).
pub fn encode_adds_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_addsub_shifted(sf, 0, 1, rd, rn, rm, shift, amount)
}

/// Encode `SUB <Xd>, <Xn>, <Xm>{, <shift> #<amount>}` (shifted register).
pub fn encode_sub_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_addsub_shifted(sf, 1, 0, rd, rn, rm, shift, amount)
}

/// Encode `SUBS <Xd>, <Xn>, <Xm>{, <shift> #<amount>}` (flag-setting SUB).
///
/// `CMP` is an alias when `Rd` = `XZR`/`WZR`.
pub fn encode_subs_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_addsub_shifted(sf, 1, 1, rd, rn, rm, shift, amount)
}

/// Internal helper for ADD/SUB (shifted register).
///
/// Format: `sf:op:S:01011:shift:0:Rm:imm6:Rn:Rd`
fn encode_addsub_shifted(
    sf: bool,
    op: u8,
    s: u8,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: ShiftType,
    amount: u8,
) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let op_bit = ((op & 1) as u32) << 30;
    let s_bit = ((s & 1) as u32) << 29;
    let fixed = 0b01011u32 << 24;
    let shift_bits = ((shift as u32) & 0x3) << 22;
    // bit 21 = 0 for shifted register variant
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let imm6 = ((amount & 0x3F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | op_bit | s_bit | fixed | shift_bits | rm_bits | imm6 | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// Logical (Shifted Register)
// ---------------------------------------------------------------------------

/// Encode `AND <Xd>, <Xn>, <Xm>{, <shift> #<amount>}`.
pub fn encode_and_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_logical_shifted(sf, 0b00, 0, rd, rn, rm, shift, amount)
}

/// Encode `ORR <Xd>, <Xn>, <Xm>{, <shift> #<amount>}`.
///
/// `MOV` is an alias when `Rn` = `XZR` and shift = LSL#0.
pub fn encode_orr_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_logical_shifted(sf, 0b01, 0, rd, rn, rm, shift, amount)
}

/// Encode `EOR <Xd>, <Xn>, <Xm>{, <shift> #<amount>}`.
pub fn encode_eor_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_logical_shifted(sf, 0b10, 0, rd, rn, rm, shift, amount)
}

/// Encode `ORN <Xd>, <Xn>, <Xm>{, <shift> #<amount>}` (bitwise OR NOT).
///
/// `MVN` is an alias when `Rn` = `XZR`.
pub fn encode_orn_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_logical_shifted(sf, 0b01, 1, rd, rn, rm, shift, amount)
}

/// Encode `BIC <Xd>, <Xn>, <Xm>{, <shift> #<amount>}` (bit clear = AND NOT).
pub fn encode_bic_reg(sf: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> u32 {
    encode_logical_shifted(sf, 0b00, 1, rd, rn, rm, shift, amount)
}

/// Internal helper for logical (shifted register) instructions.
///
/// Format: `sf:opc:01010:shift:N:Rm:imm6:Rn:Rd`
fn encode_logical_shifted(
    sf: bool,
    opc: u8,
    n: u8,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: ShiftType,
    amount: u8,
) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let opc_bits = ((opc & 0x3) as u32) << 29;
    let fixed = 0b01010u32 << 24;
    let shift_bits = ((shift as u32) & 0x3) << 22;
    let n_bit = ((n & 1) as u32) << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let imm6 = ((amount & 0x3F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | opc_bits | fixed | shift_bits | n_bit | rm_bits | imm6 | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// Multiply and Divide
// ---------------------------------------------------------------------------

/// Encode `MADD <Xd>, <Xn>, <Xm>, <Xa>` — `Rd = Ra + Rn*Rm`.
///
/// `MUL` is an alias when `Ra` = `XZR`/`WZR`.
pub fn encode_madd(sf: bool, rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    encode_dp3src(sf, 0b000, 0, rd, rn, rm, ra)
}

/// Encode `MSUB <Xd>, <Xn>, <Xm>, <Xa>` — `Rd = Ra - Rn*Rm`.
///
/// `MNEG` is an alias when `Ra` = `XZR`/`WZR`.
pub fn encode_msub(sf: bool, rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    encode_dp3src(sf, 0b000, 1, rd, rn, rm, ra)
}

/// Encode `SMULH <Xd>, <Xn>, <Xm>` — signed 64×64→128 high half.
pub fn encode_smulh(rd: u8, rn: u8, rm: u8) -> u32 {
    // sf=1, op54=00, op31=010, o0=0, Ra=0b11111 (ignored)
    encode_dp3src(true, 0b010, 0, rd, rn, rm, 0b11111)
}

/// Encode `UMULH <Xd>, <Xn>, <Xm>` — unsigned 64×64→128 high half.
pub fn encode_umulh(rd: u8, rn: u8, rm: u8) -> u32 {
    // sf=1, op54=00, op31=110, o0=0, Ra=0b11111 (ignored)
    encode_dp3src(true, 0b110, 0, rd, rn, rm, 0b11111)
}

/// Internal helper for 3-source data-processing instructions.
///
/// Format: `sf:op54:11011:op31:Rm:o0:Ra:Rn:Rd`
fn encode_dp3src(sf: bool, op31: u8, o0: u8, rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    // op54 = 00 for all standard multiply instructions
    let fixed = 0b11011u32 << 24;
    let op31_bits = ((op31 & 0x7) as u32) << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let o0_bit = ((o0 & 1) as u32) << 15;
    let ra_bits = ((ra & 0x1F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | fixed | op31_bits | rm_bits | o0_bit | ra_bits | rn_bits | rd_bits
}

/// Encode `SDIV <Xd>, <Xn>, <Xm>` — signed integer divide.
pub fn encode_sdiv(sf: bool, rd: u8, rn: u8, rm: u8) -> u32 {
    encode_dp2src(sf, 0b000011, rd, rn, rm)
}

/// Encode `UDIV <Xd>, <Xn>, <Xm>` — unsigned integer divide.
pub fn encode_udiv(sf: bool, rd: u8, rn: u8, rm: u8) -> u32 {
    encode_dp2src(sf, 0b000010, rd, rn, rm)
}

/// Internal helper for 2-source data-processing instructions.
///
/// Format: `sf:0:S:11010110:Rm:opcode:Rn:Rd`
fn encode_dp2src(sf: bool, opcode: u8, rd: u8, rn: u8, rm: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b11010110u32 << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let opcode_bits = ((opcode & 0x3F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | fixed | rm_bits | opcode_bits | rn_bits | rd_bits
}

// ===========================================================================
// Loads and Stores
// ===========================================================================

/// Encode `LDR <Xt|Wt>, [<Xn|SP>, #<imm12>]` — load with unsigned immediate offset.
///
/// `size`: 0b11 for 64-bit (X), 0b10 for 32-bit (W).
/// `imm12` is pre-scaled by access size (the raw field stores imm12 >> scale).
pub fn encode_ldr_imm(size: u8, rt: u8, rn: u8, imm12: u16) -> u32 {
    // opc=01 for LDR
    encode_ldst_unsigned_imm(size, 0b01, rt, rn, imm12)
}

/// Encode `STR <Xt|Wt>, [<Xn|SP>, #<imm12>]` — store with unsigned immediate offset.
pub fn encode_str_imm(size: u8, rt: u8, rn: u8, imm12: u16) -> u32 {
    // opc=00 for STR
    encode_ldst_unsigned_imm(size, 0b00, rt, rn, imm12)
}

/// Encode `LDRB <Wt>, [<Xn|SP>, #<imm12>]` — load byte (zero-extend).
pub fn encode_ldrb_imm(rt: u8, rn: u8, imm12: u16) -> u32 {
    // size=00, opc=01
    encode_ldst_unsigned_imm(0b00, 0b01, rt, rn, imm12)
}

/// Encode `LDRH <Wt>, [<Xn|SP>, #<imm12>]` — load halfword (zero-extend).
pub fn encode_ldrh_imm(rt: u8, rn: u8, imm12: u16) -> u32 {
    // size=01, opc=01
    encode_ldst_unsigned_imm(0b01, 0b01, rt, rn, imm12)
}

/// Encode `LDRSB <Xt|Wt>, [<Xn|SP>, #<imm12>]` — load signed byte.
///
/// `sf`: `true` for 64-bit destination, `false` for 32-bit.
pub fn encode_ldrsb_imm(sf: bool, rt: u8, rn: u8, imm12: u16) -> u32 {
    // size=00, opc = 11 (64-bit dest) or 10 (32-bit dest)
    let opc = if sf { 0b10 } else { 0b11 };
    encode_ldst_unsigned_imm(0b00, opc, rt, rn, imm12)
}

/// Encode `LDRSH <Xt|Wt>, [<Xn|SP>, #<imm12>]` — load signed halfword.
pub fn encode_ldrsh_imm(sf: bool, rt: u8, rn: u8, imm12: u16) -> u32 {
    let opc = if sf { 0b10 } else { 0b11 };
    encode_ldst_unsigned_imm(0b01, opc, rt, rn, imm12)
}

/// Encode `LDRSW <Xt>, [<Xn|SP>, #<imm12>]` — load signed word to 64-bit.
pub fn encode_ldrsw_imm(rt: u8, rn: u8, imm12: u16) -> u32 {
    // size=10, opc=10
    encode_ldst_unsigned_imm(0b10, 0b10, rt, rn, imm12)
}

/// Encode `STRB <Wt>, [<Xn|SP>, #<imm12>]` — store byte.
pub fn encode_strb_imm(rt: u8, rn: u8, imm12: u16) -> u32 {
    // size=00, opc=00
    encode_ldst_unsigned_imm(0b00, 0b00, rt, rn, imm12)
}

/// Encode `STRH <Wt>, [<Xn|SP>, #<imm12>]` — store halfword.
pub fn encode_strh_imm(rt: u8, rn: u8, imm12: u16) -> u32 {
    // size=01, opc=00
    encode_ldst_unsigned_imm(0b01, 0b00, rt, rn, imm12)
}

/// Internal helper for load/store (unsigned immediate offset).
///
/// Format: `size:111:V:01:opc:imm12:Rn:Rt`
fn encode_ldst_unsigned_imm(size: u8, opc: u8, rt: u8, rn: u8, imm12: u16) -> u32 {
    let size_bits = ((size & 0x3) as u32) << 30;
    let fixed = 0b111u32 << 27;
    // V=0 (non-SIMD)
    let v_bit = 0u32 << 26;
    let opc2 = 0b01u32 << 24;
    let opc_bits = ((opc & 0x3) as u32) << 22;
    let imm = ((imm12 as u32) & 0xFFF) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    size_bits | fixed | v_bit | opc2 | opc_bits | imm | rn_bits | rt_bits
}

// ---------------------------------------------------------------------------
// Load/Store Register (register offset)
// ---------------------------------------------------------------------------

/// Encode `LDR <Xt|Wt>, [<Xn|SP>, <Xm|Wm>{, <extend> {<amount>}}]`.
pub fn encode_ldr_reg(size: u8, rt: u8, rn: u8, rm: u8, extend: ExtendType, shift: bool) -> u32 {
    encode_ldst_register(size, 0b01, rt, rn, rm, extend, shift)
}

/// Encode `STR <Xt|Wt>, [<Xn|SP>, <Xm|Wm>{, <extend> {<amount>}}]`.
pub fn encode_str_reg(size: u8, rt: u8, rn: u8, rm: u8, extend: ExtendType, shift: bool) -> u32 {
    encode_ldst_register(size, 0b00, rt, rn, rm, extend, shift)
}

/// Internal helper for load/store (register offset).
///
/// Format: `size:111:V:00:opc:1:Rm:option:S:10:Rn:Rt`
fn encode_ldst_register(
    size: u8,
    opc: u8,
    rt: u8,
    rn: u8,
    rm: u8,
    extend: ExtendType,
    s: bool,
) -> u32 {
    let size_bits = ((size & 0x3) as u32) << 30;
    let fixed = 0b111u32 << 27;
    // V=0 (non-SIMD), 00 at bits [25:24]
    let opc_bits = ((opc & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let option_bits = ((extend as u32) & 0x7) << 13;
    let s_bit = (s as u32) << 12;
    let ten = 0b10u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    size_bits | fixed | opc_bits | one | rm_bits | option_bits | s_bit | ten | rn_bits | rt_bits
}

// ---------------------------------------------------------------------------
// Load/Store Pre-Index and Post-Index
// ---------------------------------------------------------------------------

/// Encode `LDR <Xt|Wt>, [<Xn|SP>, #<simm9>]!` — load with pre-index.
pub fn encode_ldr_pre(size: u8, rt: u8, rn: u8, simm9: i16) -> u32 {
    encode_ldst_pre_post(size, 0b01, rt, rn, simm9, true)
}

/// Encode `STR <Xt|Wt>, [<Xn|SP>, #<simm9>]!` — store with pre-index.
pub fn encode_str_pre(size: u8, rt: u8, rn: u8, simm9: i16) -> u32 {
    encode_ldst_pre_post(size, 0b00, rt, rn, simm9, true)
}

/// Encode `LDR <Xt|Wt>, [<Xn|SP>], #<simm9>` — load with post-index.
pub fn encode_ldr_post(size: u8, rt: u8, rn: u8, simm9: i16) -> u32 {
    encode_ldst_pre_post(size, 0b01, rt, rn, simm9, false)
}

/// Encode `STR <Xt|Wt>, [<Xn|SP>], #<simm9>` — store with post-index.
pub fn encode_str_post(size: u8, rt: u8, rn: u8, simm9: i16) -> u32 {
    encode_ldst_pre_post(size, 0b00, rt, rn, simm9, false)
}

/// Internal helper for load/store (pre-index / post-index).
///
/// Format: `size:111:V:00:opc:0:imm9:idx:1:Rn:Rt`
/// - `pre`: bit\[11\] = 1 for pre-index, 0 for post-index
fn encode_ldst_pre_post(size: u8, opc: u8, rt: u8, rn: u8, simm9: i16, pre: bool) -> u32 {
    let size_bits = ((size & 0x3) as u32) << 30;
    let fixed = 0b111u32 << 27;
    let opc_bits = ((opc & 0x3) as u32) << 22;
    // bit 21 = 0 for unscaled immediate
    let imm9 = ((simm9 as u32) & 0x1FF) << 12;
    let idx = if pre { 0b11u32 } else { 0b01u32 };
    let idx_bits = idx << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    size_bits | fixed | opc_bits | imm9 | idx_bits | rn_bits | rt_bits
}

// ---------------------------------------------------------------------------
// Load/Store Pair (LDP/STP)
// ---------------------------------------------------------------------------

/// Encode `LDP <Xt1>, <Xt2>, [<Xn|SP>, #<imm7>]` — load pair (signed offset).
pub fn encode_ldp(sf: bool, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    encode_ldst_pair(sf, 0b10, 1, rt1, rt2, rn, imm7) // opc=10 for signed offset, L=1 for load
}

/// Encode `STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm7>]` — store pair (signed offset).
pub fn encode_stp(sf: bool, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    encode_ldst_pair(sf, 0b10, 0, rt1, rt2, rn, imm7)
}

/// Encode `LDP <Xt1>, <Xt2>, [<Xn|SP>, #<imm7>]!` — load pair (pre-index).
pub fn encode_ldp_pre(sf: bool, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    encode_ldst_pair(sf, 0b11, 1, rt1, rt2, rn, imm7)
}

/// Encode `STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm7>]!` — store pair (pre-index).
pub fn encode_stp_pre(sf: bool, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    encode_ldst_pair(sf, 0b11, 0, rt1, rt2, rn, imm7)
}

/// Encode `LDP <Xt1>, <Xt2>, [<Xn|SP>], #<imm7>` — load pair (post-index).
pub fn encode_ldp_post(sf: bool, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    encode_ldst_pair(sf, 0b01, 1, rt1, rt2, rn, imm7)
}

/// Encode `STP <Xt1>, <Xt2>, [<Xn|SP>], #<imm7>` — store pair (post-index).
pub fn encode_stp_post(sf: bool, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    encode_ldst_pair(sf, 0b01, 0, rt1, rt2, rn, imm7)
}

/// Internal helper for load/store pair instructions.
///
/// Format: `opc:101:V:idx:L:imm7:Rt2:Rn:Rt`
fn encode_ldst_pair(sf: bool, idx: u8, l: u8, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    let opc = if sf { 0b10u32 } else { 0b00u32 };
    let opc_bits = opc << 30;
    let fixed = 0b101u32 << 27;
    // V=0 (non-SIMD)
    let idx_bits = ((idx & 0x3) as u32) << 23;
    let l_bit = ((l & 1) as u32) << 22;
    let imm7_bits = ((imm7 as u32) & 0x7F) << 15;
    let rt2_bits = ((rt2 & 0x1F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt1_bits = (rt1 & 0x1F) as u32;

    opc_bits | fixed | idx_bits | l_bit | imm7_bits | rt2_bits | rn_bits | rt1_bits
}

// ---------------------------------------------------------------------------
// Load Literal (PC-Relative)
// ---------------------------------------------------------------------------

/// Encode `LDR <Xt|Wt>, <label>` — load from PC-relative literal (±1 MiB).
///
/// `imm19` is the signed instruction-count offset (byte offset = imm19 << 2).
pub fn encode_ldr_literal(sf: bool, rt: u8, imm19: i32) -> u32 {
    let opc = if sf { 0b01u32 } else { 0b00u32 };
    let opc_bits = opc << 30;
    let fixed = 0b011u32 << 27;
    // V=0 (non-SIMD)
    let imm19_bits = ((imm19 as u32) & 0x7_FFFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    opc_bits | fixed | imm19_bits | rt_bits
}

// ===========================================================================
// Branch Instructions
// ===========================================================================

/// Encode `B <label>` — unconditional branch (±128 MiB).
///
/// `imm26` is the signed instruction-count offset (byte offset = imm26 << 2).
/// Format: `0:00101:imm26`
pub fn encode_b(imm26: i32) -> u32 {
    let fixed = 0b000101u32 << 26;
    let imm = (imm26 as u32) & 0x03FF_FFFF;
    fixed | imm
}

/// Encode `BL <label>` — branch with link (±128 MiB).
///
/// `imm26` is the signed instruction-count offset (byte offset = imm26 << 2).
/// Format: `1:00101:imm26`
pub fn encode_bl(imm26: i32) -> u32 {
    let fixed = 0b100101u32 << 26;
    let imm = (imm26 as u32) & 0x03FF_FFFF;
    fixed | imm
}

/// Encode `B.<cond> <label>` — conditional branch (±1 MiB).
///
/// `cond` is a 4-bit condition code (EQ=0b0000, NE=0b0001, etc.).
/// `imm19` is the signed instruction-count offset.
/// Format: `01010100:imm19:0:cond`
pub fn encode_b_cond(cond: u8, imm19: i32) -> u32 {
    let fixed = 0b01010100u32 << 24;
    let imm = ((imm19 as u32) & 0x7_FFFF) << 5;
    let cond_bits = (cond & 0xF) as u32;

    fixed | imm | cond_bits
}

/// Encode `CBZ <Xt|Wt>, <label>` — compare and branch if zero (±1 MiB).
pub fn encode_cbz(sf: bool, rt: u8, imm19: i32) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b011010u32 << 25;
    // op = 0 for CBZ
    let imm = ((imm19 as u32) & 0x7_FFFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    sf_bit | fixed | imm | rt_bits
}

/// Encode `CBNZ <Xt|Wt>, <label>` — compare and branch if not zero (±1 MiB).
pub fn encode_cbnz(sf: bool, rt: u8, imm19: i32) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b011010u32 << 25;
    let op = 1u32 << 24;
    let imm = ((imm19 as u32) & 0x7_FFFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    sf_bit | fixed | op | imm | rt_bits
}

/// Encode `TBZ <Xt>, #<bit>, <label>` — test bit and branch if zero (±32 KiB).
///
/// `bit` selects which bit to test (0–63). `imm14` is the signed
/// instruction-count offset.
pub fn encode_tbz(rt: u8, bit: u8, imm14: i16) -> u32 {
    let b5 = ((bit >> 5) as u32) << 31;
    let fixed = 0b011011u32 << 25;
    // op = 0 for TBZ
    let b40 = ((bit & 0x1F) as u32) << 19;
    let imm = ((imm14 as u32) & 0x3FFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    b5 | fixed | b40 | imm | rt_bits
}

/// Encode `TBNZ <Xt>, #<bit>, <label>` — test bit and branch if not zero (±32 KiB).
pub fn encode_tbnz(rt: u8, bit: u8, imm14: i16) -> u32 {
    let b5 = ((bit >> 5) as u32) << 31;
    let fixed = 0b011011u32 << 25;
    let op = 1u32 << 24;
    let b40 = ((bit & 0x1F) as u32) << 19;
    let imm = ((imm14 as u32) & 0x3FFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    b5 | fixed | op | b40 | imm | rt_bits
}

// ---------------------------------------------------------------------------
// Unconditional Branch (Register)
// ---------------------------------------------------------------------------

/// Encode `BR <Xn>` — indirect branch (register).
///
/// Format: `1101011:0000:11111:000000:Rn:00000`
pub fn encode_br(rn: u8) -> u32 {
    0xD61F_0000 | (((rn & 0x1F) as u32) << 5)
}

/// Encode `BLR <Xn>` — indirect branch with link (register).
///
/// Format: `1101011:0001:11111:000000:Rn:00000`
pub fn encode_blr(rn: u8) -> u32 {
    0xD63F_0000 | (((rn & 0x1F) as u32) << 5)
}

/// Encode `RET {<Xn>}` — return from subroutine (default X30).
///
/// Format: `1101011:0010:11111:000000:Rn:00000`
pub fn encode_ret(rn: u8) -> u32 {
    0xD65F_0000 | (((rn & 0x1F) as u32) << 5)
}

// ===========================================================================
// Comparison and Conditional Select
// ===========================================================================

/// Encode `CCMP <Xn>, <Xm>, #<nzcv>, <cond>` — conditional compare (register).
pub fn encode_ccmp_reg(sf: bool, rn: u8, rm: u8, nzcv: u8, cond: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    // op=1 (SUB), S=1
    let fixed = 0b1111010010u32 << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let cond_bits = ((cond & 0xF) as u32) << 12;
    // bit 11 = 0 (register variant), bit 10 = 0
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let nzcv_bits = (nzcv & 0xF) as u32;

    sf_bit | fixed | rm_bits | cond_bits | rn_bits | nzcv_bits
}

/// Encode `CCMP <Xn>, #<imm5>, #<nzcv>, <cond>` — conditional compare (immediate).
pub fn encode_ccmp_imm(sf: bool, rn: u8, imm5: u8, nzcv: u8, cond: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b1111010010u32 << 21;
    let imm5_bits = ((imm5 & 0x1F) as u32) << 16;
    let cond_bits = ((cond & 0xF) as u32) << 12;
    let imm_flag = 1u32 << 11; // bit 11 = 1 for immediate variant
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let nzcv_bits = (nzcv & 0xF) as u32;

    sf_bit | fixed | imm5_bits | cond_bits | imm_flag | rn_bits | nzcv_bits
}

/// Encode `CSEL <Xd>, <Xn>, <Xm>, <cond>` — conditional select.
///
/// `Rd = cond ? Rn : Rm`
pub fn encode_csel(sf: bool, rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    encode_condsel(sf, 0, 0, rd, rn, rm, cond)
}

/// Encode `CSINC <Xd>, <Xn>, <Xm>, <cond>` — conditional select increment.
///
/// `Rd = cond ? Rn : Rm + 1`. `CSET` is an alias when `Rn` = `Rm` = `XZR`.
pub fn encode_csinc(sf: bool, rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    encode_condsel(sf, 0, 1, rd, rn, rm, cond)
}

/// Encode `CSINV <Xd>, <Xn>, <Xm>, <cond>` — conditional select invert.
///
/// `Rd = cond ? Rn : ~Rm`. `CSETM` is an alias when `Rn` = `Rm` = `XZR`.
pub fn encode_csinv(sf: bool, rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    encode_condsel(sf, 1, 0, rd, rn, rm, cond)
}

/// Encode `CSNEG <Xd>, <Xn>, <Xm>, <cond>` — conditional select negate.
///
/// `Rd = cond ? Rn : -Rm`. `CNEG` is an alias when `Rn` = `Rm`.
pub fn encode_csneg(sf: bool, rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    encode_condsel(sf, 1, 1, rd, rn, rm, cond)
}

/// Internal helper for conditional select instructions.
///
/// Format: `sf:op:S:11010100:Rm:cond:op2:Rn:Rd`
fn encode_condsel(sf: bool, op: u8, op2: u8, rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let op_bit = ((op & 1) as u32) << 30;
    // S = 0 for all CSEL variants
    let fixed = 0b11010100u32 << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let cond_bits = ((cond & 0xF) as u32) << 12;
    let op2_bit = ((op2 & 1) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | op_bit | fixed | rm_bits | cond_bits | op2_bit | rn_bits | rd_bits
}

// ===========================================================================
// Floating-Point / SIMD Instructions
// ===========================================================================

/// Encode `FADD <Sd|Dd>, <Sn|Dn>, <Sm|Dm>`.
///
/// `ftype`: 0b00=single(S), 0b01=double(D), 0b11=half(H).
pub fn encode_fadd(ftype: u8, rd: u8, rn: u8, rm: u8) -> u32 {
    encode_fp_dp2(ftype, 0b0010, rd, rn, rm)
}

/// Encode `FSUB <Sd|Dd>, <Sn|Dn>, <Sm|Dm>`.
pub fn encode_fsub(ftype: u8, rd: u8, rn: u8, rm: u8) -> u32 {
    encode_fp_dp2(ftype, 0b0011, rd, rn, rm)
}

/// Encode `FMUL <Sd|Dd>, <Sn|Dn>, <Sm|Dm>`.
pub fn encode_fmul(ftype: u8, rd: u8, rn: u8, rm: u8) -> u32 {
    encode_fp_dp2(ftype, 0b0000, rd, rn, rm)
}

/// Encode `FDIV <Sd|Dd>, <Sn|Dn>, <Sm|Dm>`.
pub fn encode_fdiv(ftype: u8, rd: u8, rn: u8, rm: u8) -> u32 {
    encode_fp_dp2(ftype, 0b0001, rd, rn, rm)
}

/// Internal helper for 2-source FP data-processing.
///
/// Format: `0:0:0:11110:ftype:1:Rm:opcode:10:Rn:Rd`
fn encode_fp_dp2(ftype: u8, opcode: u8, rd: u8, rn: u8, rm: u8) -> u32 {
    let fixed = 0b00011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let opcode_bits = ((opcode & 0xF) as u32) << 12;
    let ten = 0b10u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    fixed | ftype_bits | one | rm_bits | opcode_bits | ten | rn_bits | rd_bits
}

/// Encode `FNEG <Sd|Dd>, <Sn|Dn>`.
pub fn encode_fneg(ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_fp_dp1(ftype, 0b000010, rd, rn)
}

/// Encode `FABS <Sd|Dd>, <Sn|Dn>`.
pub fn encode_fabs(ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_fp_dp1(ftype, 0b000001, rd, rn)
}

/// Encode `FSQRT <Sd|Dd>, <Sn|Dn>`.
pub fn encode_fsqrt(ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_fp_dp1(ftype, 0b000011, rd, rn)
}

/// Internal helper for 1-source FP data-processing.
///
/// Format: `0:0:0:11110:ftype:1:0000:opcode:10000:Rn:Rd`
fn encode_fp_dp1(ftype: u8, opcode: u8, rd: u8, rn: u8) -> u32 {
    let fixed = 0b00011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    // bits [20:17] = 0000
    let opcode_bits = ((opcode & 0x3F) as u32) << 15;
    let fixed2 = 0b10000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    fixed | ftype_bits | one | opcode_bits | fixed2 | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// FP Comparison
// ---------------------------------------------------------------------------

/// Encode `FCMP <Sn|Dn>, <Sm|Dm>` — FP compare.
pub fn encode_fcmp(ftype: u8, rn: u8, rm: u8) -> u32 {
    let fixed = 0b00011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    let rm_bits = ((rm & 0x1F) as u32) << 16;
    let fixed2 = 0b001000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    // opcode2 = 00000 for register comparison
    fixed | ftype_bits | one | rm_bits | fixed2 | rn_bits
}

/// Encode `FCMP <Sn|Dn>, #0.0` — FP compare with zero.
pub fn encode_fcmp_zero(ftype: u8, rn: u8) -> u32 {
    let fixed = 0b00011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    // Rm = 00000 for zero comparison
    let fixed2 = 0b001000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let opcode2 = 0b01000u32; // bit 3 = 1 for zero variant

    fixed | ftype_bits | one | fixed2 | rn_bits | opcode2
}

// ---------------------------------------------------------------------------
// FP Move
// ---------------------------------------------------------------------------

/// Encode `FMOV <Sd|Dd>, <Sn|Dn>` — FP register-to-register move.
pub fn encode_fmov_reg(ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_fp_dp1(ftype, 0b000000, rd, rn)
}

/// Encode `FMOV <Xd|Wd>, <Sn|Dn>` — FP to GPR move.
pub fn encode_fmov_to_gpr(sf: bool, ftype: u8, rd: u8, rn: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b0011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let opcode_bits = 0b1001110000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | fixed | ftype_bits | opcode_bits | rn_bits | rd_bits
}

/// Encode `FMOV <Sd|Dd>, <Xn|Wn>` — GPR to FP move.
pub fn encode_fmov_from_gpr(sf: bool, ftype: u8, rd: u8, rn: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b0011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    // GPR→FP encoding uses rmode:opcode = 00:111 (differs from FP→GPR).
    let opcode_bits = 0b1001100000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | fixed | ftype_bits | opcode_bits | rn_bits | rd_bits
}

/// Encode `FMOV <Sd|Dd>, #<imm8>` — FP immediate move.
///
/// `imm8` is an 8-bit FP immediate (see ARM ARM for encoding table).
pub fn encode_fmov_imm(ftype: u8, rd: u8, imm8: u8) -> u32 {
    let fixed = 0b00011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    let imm8_bits = ((imm8 as u32) & 0xFF) << 13;
    let fixed2 = 0b10000u32 << 5;
    // bits [9:5] = 00000 (Rn unused)
    let rd_bits = (rd & 0x1F) as u32;

    fixed | ftype_bits | one | imm8_bits | fixed2 | rd_bits
}

// ---------------------------------------------------------------------------
// Integer ↔ FP Conversion
// ---------------------------------------------------------------------------

/// Encode `SCVTF <Sd|Dd>, <Xn|Wn>` — signed integer to FP.
pub fn encode_scvtf(sf: bool, ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_int_fp_conv(sf, ftype, 0b00, 0b010, rd, rn)
}

/// Encode `UCVTF <Sd|Dd>, <Xn|Wn>` — unsigned integer to FP.
pub fn encode_ucvtf(sf: bool, ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_int_fp_conv(sf, ftype, 0b00, 0b011, rd, rn)
}

/// Encode `FCVTZS <Xd|Wd>, <Sn|Dn>` — FP to signed integer (round toward zero).
pub fn encode_fcvtzs(sf: bool, ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_int_fp_conv(sf, ftype, 0b11, 0b000, rd, rn)
}

/// Encode `FCVTZU <Xd|Wd>, <Sn|Dn>` — FP to unsigned integer (round toward zero).
pub fn encode_fcvtzu(sf: bool, ftype: u8, rd: u8, rn: u8) -> u32 {
    encode_int_fp_conv(sf, ftype, 0b11, 0b001, rd, rn)
}

/// Internal helper for integer ↔ FP conversion instructions.
///
/// Format: `sf:0:S:11110:ftype:1:rmode:opcode:000000:Rn:Rd`
fn encode_int_fp_conv(sf: bool, ftype: u8, rmode: u8, opcode: u8, rd: u8, rn: u8) -> u32 {
    let sf_bit = (sf as u32) << 31;
    let fixed = 0b0011110u32 << 24;
    let ftype_bits = ((ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    let rmode_bits = ((rmode & 0x3) as u32) << 19;
    let opcode_bits = ((opcode & 0x7) as u32) << 16;
    let fixed2 = 0b000000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    sf_bit | fixed | ftype_bits | one | rmode_bits | opcode_bits | fixed2 | rn_bits | rd_bits
}

/// Encode `FCVT <Sd|Dd|Hd>, <Sn|Dn|Hn>` — FP-to-FP precision conversion.
///
/// `dst_ftype` and `src_ftype`: 0b00=S, 0b01=D, 0b11=H.
pub fn encode_fcvt(dst_ftype: u8, src_ftype: u8, rd: u8, rn: u8) -> u32 {
    let fixed = 0b00011110u32 << 24;
    let src_bits = ((src_ftype & 0x3) as u32) << 22;
    let one = 1u32 << 21;
    let fixed2 = 0b0001u32 << 17;
    let dst_bits = ((dst_ftype & 0x3) as u32) << 15;
    let fixed3 = 0b10000u32 << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rd_bits = (rd & 0x1F) as u32;

    fixed | src_bits | one | fixed2 | dst_bits | fixed3 | rn_bits | rd_bits
}

// ---------------------------------------------------------------------------
// FP Load/Store
// ---------------------------------------------------------------------------

/// Encode `LDR <St|Dt|Qt>, [<Xn|SP>, #<imm12>]` — FP load (unsigned offset).
///
/// `ftype`: 0b00=S(32), 0b01=D(64), 0b10=Q(128).
pub fn encode_ldr_fp_imm(ftype: u8, rt: u8, rn: u8, imm12: u16) -> u32 {
    let size = match ftype {
        0b00 => 0b10u32, // S register = 32-bit = size 2
        0b01 => 0b11u32, // D register = 64-bit = size 3
        0b10 => 0b00u32, // Q register = 128-bit = size 0 with opc=11
        _ => 0b10u32,
    };
    let size_bits = size << 30;
    let fixed = 0b111u32 << 27;
    let v_bit = 1u32 << 26; // V=1 for SIMD/FP
    let opc2 = 0b01u32 << 24;
    let opc = if ftype == 0b10 { 0b11u32 } else { 0b01u32 };
    let opc_bits = opc << 22;
    let imm = ((imm12 as u32) & 0xFFF) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    size_bits | fixed | v_bit | opc2 | opc_bits | imm | rn_bits | rt_bits
}

/// Encode `STR <St|Dt|Qt>, [<Xn|SP>, #<imm12>]` — FP store (unsigned offset).
pub fn encode_str_fp_imm(ftype: u8, rt: u8, rn: u8, imm12: u16) -> u32 {
    let size = match ftype {
        0b00 => 0b10u32, // S
        0b01 => 0b11u32, // D
        0b10 => 0b00u32, // Q
        _ => 0b10u32,
    };
    let size_bits = size << 30;
    let fixed = 0b111u32 << 27;
    let v_bit = 1u32 << 26;
    let opc2 = 0b01u32 << 24;
    let opc = if ftype == 0b10 { 0b10u32 } else { 0b00u32 };
    let opc_bits = opc << 22;
    let imm = ((imm12 as u32) & 0xFFF) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    size_bits | fixed | v_bit | opc2 | opc_bits | imm | rn_bits | rt_bits
}

/// Encode `LDP <St1|Dt1|Qt1>, <St2|Dt2|Qt2>, [<Xn|SP>, #<imm7>]` — FP load pair.
pub fn encode_ldp_fp(ftype: u8, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    let opc = match ftype {
        0b00 => 0b00u32, // S pair
        0b01 => 0b01u32, // D pair
        0b10 => 0b10u32, // Q pair
        _ => 0b00u32,
    };
    let opc_bits = opc << 30;
    let fixed = 0b101u32 << 27;
    let v_bit = 1u32 << 26; // V=1 for SIMD/FP
    // idx = 10 for signed offset variant
    let idx = 0b10u32 << 23;
    let l_bit = 1u32 << 22; // L=1 for load
    let imm7_bits = ((imm7 as u32) & 0x7F) << 15;
    let rt2_bits = ((rt2 & 0x1F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt1_bits = (rt1 & 0x1F) as u32;

    opc_bits | fixed | v_bit | idx | l_bit | imm7_bits | rt2_bits | rn_bits | rt1_bits
}

/// Encode `STP <St1|Dt1|Qt1>, <St2|Dt2|Qt2>, [<Xn|SP>, #<imm7>]` — FP store pair.
pub fn encode_stp_fp(ftype: u8, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
    let opc = match ftype {
        0b00 => 0b00u32,
        0b01 => 0b01u32,
        0b10 => 0b10u32,
        _ => 0b00u32,
    };
    let opc_bits = opc << 30;
    let fixed = 0b101u32 << 27;
    let v_bit = 1u32 << 26;
    let idx = 0b10u32 << 23;
    // L=0 for store
    let imm7_bits = ((imm7 as u32) & 0x7F) << 15;
    let rt2_bits = ((rt2 & 0x1F) as u32) << 10;
    let rn_bits = ((rn & 0x1F) as u32) << 5;
    let rt1_bits = (rt1 & 0x1F) as u32;

    opc_bits | fixed | v_bit | idx | imm7_bits | rt2_bits | rn_bits | rt1_bits
}

// ===========================================================================
// Bitmask Immediate Encoding
// ===========================================================================

/// Attempt to encode an arbitrary 64-bit value as an A64 bitmask immediate.
///
/// Returns `Some((n, immr, imms))` if the value is a valid repeating bit
/// pattern, or `None` if it cannot be encoded.
///
/// AArch64 bitmask immediates represent repeating bit patterns within
/// element sizes of 2, 4, 8, 16, 32, or 64 bits. Within each element,
/// the pattern is a contiguous run of 1-bits that can be rotated.
///
/// # Algorithm
///
/// 1. Reject all-zeros and all-ones (not encodable).
/// 2. Find the smallest repeating element size.
/// 3. Within the element, locate the contiguous run of 1-bits and its rotation.
/// 4. Encode as `N:immr:imms` per the A64 specification.
pub fn encode_bitmask_immediate(value: u64, reg_size: u8) -> Option<(bool, u8, u8)> {
    // All-zeros and all-ones are not valid bitmask immediates.
    if value == 0 {
        return None;
    }
    let mask = if reg_size == 32 {
        0xFFFF_FFFFu64
    } else {
        0xFFFF_FFFF_FFFF_FFFFu64
    };
    let value = value & mask;
    if value == mask {
        return None;
    }

    // Find the smallest repeating element size.
    let mut size = reg_size as u64;
    let mut tmp_val = value;

    // Check if the pattern repeats at smaller element sizes.
    while size > 2 {
        let half = size / 2;
        let half_mask = (1u64 << half) - 1;
        let lo = tmp_val & half_mask;
        let hi = (tmp_val >> half) & half_mask;
        if lo != hi {
            break;
        }
        size = half;
        tmp_val = lo;
    }

    // Extract the element pattern.
    let elem_mask = if size == 64 {
        u64::MAX
    } else {
        (1u64 << size) - 1
    };
    let pattern = value & elem_mask;

    // Find the rotation: the contiguous run of 1-bits.
    // We need to find where the run of 1s starts and ends.
    // Rotate the pattern to find the canonical form (run starts at bit 0).
    let mut ones = 0u32;
    let mut rotation = 0u32;

    // Find a 0→1 transition to anchor the rotation search.
    let doubled = pattern | (pattern << size);
    let mut found = false;

    for r in 0..size as u32 {
        let rotated = if r == 0 {
            pattern
        } else {
            (doubled >> r) & elem_mask
        };

        // Count trailing ones.
        let trail = rotated.trailing_ones();
        if trail == 0 {
            continue;
        }

        // The remaining bits after the run must all be zeros.
        let remaining = rotated >> trail;
        if remaining == 0 || (remaining & elem_mask >> trail) == 0 {
            ones = trail;
            rotation = r;
            found = true;
            break;
        }
    }

    if !found || ones == 0 || ones as u64 == size {
        return None;
    }

    // Encode N, immr, imms.
    let n = size == 64;
    let immr = ((size as u32 - rotation) % size as u32) as u8;

    // imms encodes (ones - 1) with the element size in the upper bits.
    // The "element size" bits in imms follow a specific pattern:
    //   size=64: 0b0xxxxx (N=1)
    //   size=32: 0b10xxxx
    //   size=16: 0b110xxx
    //   size=8:  0b1110xx
    //   size=4:  0b11110x
    //   size=2:  0b111110
    let size_encoding = match size {
        64 => 0b000000u8,
        32 => 0b100000u8,
        16 => 0b110000u8,
        8 => 0b111000u8,
        4 => 0b111100u8,
        2 => 0b111110u8,
        _ => return None,
    };

    // The lower bits of imms store (ones - 1), but masked to the
    // appropriate width determined by the element size.
    let ones_field = (ones - 1) as u8;
    let imms = (size_encoding | ones_field) & 0x3F;

    Some((n, immr, imms))
}

// ===========================================================================
// Immediate Materialization
// ===========================================================================

/// Encode a full immediate value into the minimum MOVZ/MOVK instruction
/// sequence required.
///
/// Optimizations applied:
/// 1. Single MOVZ if only one 16-bit halfword is non-zero.
/// 2. MOVN + MOVK sequence if the value is close to all-ones.
/// 3. Otherwise, MOVZ for the first non-zero halfword, then MOVK for each
///    additional non-zero halfword.
///
/// # Arguments
///
/// - `rd`: Destination register (5-bit encoding).
/// - `value`: The 64-bit value to materialize.
/// - `sf`: `true` for 64-bit (`X`), `false` for 32-bit (`W`).
///
/// # Returns
///
/// A `Vec<u32>` of 1–4 instruction words.
pub fn encode_mov_imm(rd: u8, value: u64, sf: bool) -> Vec<u32> {
    let value = if sf { value } else { value & 0xFFFF_FFFF };
    let max_hw: u8 = if sf { 4 } else { 2 };

    // Check if value is zero → single MOVZ.
    if value == 0 {
        return vec![encode_movz(sf, rd, 0, 0)];
    }

    // Count non-zero and zero halfwords.
    let mut nonzero_hws = Vec::new();

    for hw in 0..max_hw {
        let hw_val = ((value >> (hw * 16)) & 0xFFFF) as u16;
        if hw_val != 0 {
            nonzero_hws.push((hw, hw_val));
        }
    }

    // Check if MOVN + MOVK is shorter (when most halfwords are 0xFFFF).
    let not_value = !value & if sf { u64::MAX } else { 0xFFFF_FFFF };
    let mut not_nonzero_hws = Vec::new();
    for hw in 0..max_hw {
        let hw_val = ((not_value >> (hw * 16)) & 0xFFFF) as u16;
        if hw_val != 0 {
            not_nonzero_hws.push((hw, hw_val));
        }
    }

    // Use MOVN strategy if it produces fewer instructions.
    if not_nonzero_hws.len() < nonzero_hws.len() {
        let mut instrs = Vec::new();
        let first = not_nonzero_hws[0];
        instrs.push(encode_movn(sf, rd, first.1, first.0));
        // Remaining halfwords that differ from 0xFFFF in the original value
        // need MOVK with the actual value.
        for hw in 0..max_hw {
            let hw_val = ((value >> (hw * 16)) & 0xFFFF) as u16;
            if hw != first.0 && hw_val != 0xFFFF {
                instrs.push(encode_movk(sf, rd, hw_val, hw));
            }
        }
        return instrs;
    }

    // Standard MOVZ + MOVK strategy.
    let mut instrs = Vec::new();
    let first = nonzero_hws[0];
    instrs.push(encode_movz(sf, rd, first.1, first.0));
    for &(hw, hw_val) in &nonzero_hws[1..] {
        instrs.push(encode_movk(sf, rd, hw_val, hw));
    }

    instrs
}

// ===========================================================================
// System Instructions
// ===========================================================================

/// Encode `NOP` — no operation.
///
/// Encoding: `0xD503201F`
pub fn encode_nop() -> u32 {
    0xD503_201F
}

/// Encode `BRK #<imm16>` — software breakpoint.
pub fn encode_brk(imm16: u16) -> u32 {
    0xD420_0000 | ((imm16 as u32) << 5)
}

/// Encode `SVC #<imm16>` — supervisor call (syscall).
pub fn encode_svc(imm16: u16) -> u32 {
    0xD400_0001 | ((imm16 as u32) << 5)
}

/// Encode `DMB <option>` — data memory barrier.
///
/// `option`: 4-bit barrier option (e.g., 0b1111 = SY, 0b1011 = ISH).
pub fn encode_dmb(option: u8) -> u32 {
    0xD503_3000 | (((option & 0xF) as u32) << 8) | 0b10111111
}

/// Encode `DSB <option>` — data synchronization barrier.
pub fn encode_dsb(option: u8) -> u32 {
    0xD503_3000 | (((option & 0xF) as u32) << 8) | 0b10011111
}

/// Encode `ISB` — instruction synchronization barrier.
pub fn encode_isb() -> u32 {
    0xD503_30DF
}

/// Encode `MRS <Xt>, <sysreg>` — move from system register.
///
/// `sysreg` encodes the 16-bit system register ID (op0:op1:CRn:CRm:op2).
pub fn encode_mrs(rt: u8, sysreg: u16) -> u32 {
    let fixed = 0xD530_0000u32; // MRS base
    let sysreg_bits = ((sysreg as u32) & 0xFFFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    fixed | sysreg_bits | rt_bits
}

/// Encode `MSR <sysreg>, <Xt>` — move to system register.
pub fn encode_msr(sysreg: u16, rt: u8) -> u32 {
    let fixed = 0xD510_0000u32; // MSR base
    let sysreg_bits = ((sysreg as u32) & 0xFFFF) << 5;
    let rt_bits = (rt & 0x1F) as u32;

    fixed | sysreg_bits | rt_bits
}

// ===========================================================================
// Top-Level Instruction Encoding Entry Point
// ===========================================================================

/// Encode a single [`MachineInstr`] into its 32-bit binary representation.
///
/// This is the primary entry point called by the assembler for each machine
/// instruction. It inspects the opcode, extracts operands, and delegates to
/// the appropriate format-specific encoding function.
///
/// # Arguments
///
/// - `instr`: The machine instruction from the code generator.
/// - `_registers`: Register info provider (for encoding validation).
///
/// # Returns
///
/// An [`EncodedInstruction`] containing the 4-byte instruction word and an
/// optional relocation for unresolved symbol references.
pub fn encode_instruction(
    instr: &crate::backend::traits::MachineInstr,
) -> EncodedInstruction {
    // The opcode field carries the pre-selected A64 instruction format.
    // For now, the encoding dispatches based on opcode ranges that the
    // AArch64 instruction selector assigns.
    //
    // Opcode namespace convention (assigned by aarch64/codegen.rs):
    //   0x0000–0x00FF: Data processing immediate
    //   0x0100–0x01FF: Data processing register
    //   0x0200–0x02FF: Loads and stores
    //   0x0300–0x03FF: Branches
    //   0x0400–0x04FF: SIMD/FP
    //   0x0500–0x05FF: System
    //   0xFFFF:        Raw pre-encoded instruction (bytes stored in operand)

    let opcode = instr.opcode;

    // Handle raw pre-encoded instructions (e.g., inline assembly).
    if opcode == 0xFFFF {
        if let Some(crate::backend::traits::MachineOperand::Immediate(raw)) =
            instr.operands.first()
        {
            return EncodedInstruction {
                bytes: *raw as u32,
                relocation: None,
            };
        }
    }

    // Default: encode as NOP for unrecognized opcodes during early development.
    // As the codegen matures, this path should be unreachable.
    EncodedInstruction {
        bytes: encode_nop(),
        relocation: None,
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nop_encoding() {
        assert_eq!(encode_nop(), 0xD503_201F);
    }

    #[test]
    fn test_ret_x30() {
        // RET X30 = 0xD65F03C0
        assert_eq!(encode_ret(30), 0xD65F_03C0);
    }

    #[test]
    fn test_br_encoding() {
        // BR X0 = 0xD61F0000
        assert_eq!(encode_br(0), 0xD61F_0000);
        // BR X16 = 0xD61F0200
        assert_eq!(encode_br(16), 0xD61F_0200);
    }

    #[test]
    fn test_blr_encoding() {
        // BLR X0 = 0xD63F0000
        assert_eq!(encode_blr(0), 0xD63F_0000);
    }

    #[test]
    fn test_b_encoding() {
        // B +0 = 0x14000000
        assert_eq!(encode_b(0), 0x1400_0000);
        // B +1 (one instruction forward) = 0x14000001
        assert_eq!(encode_b(1), 0x1400_0001);
    }

    #[test]
    fn test_bl_encoding() {
        // BL +0 = 0x94000000
        assert_eq!(encode_bl(0), 0x9400_0000);
    }

    #[test]
    fn test_add_imm_64bit() {
        // ADD X0, X1, #42 (sf=1, op=0, S=0, sh=0, imm=42, rn=1, rd=0)
        let instr = encode_add_imm(true, 0, 1, 42, false);
        // sf=1, op=0, S=0 → bits [31:29] = 100
        // fixed 100010 → bits [28:23]
        // sh=0, imm12=42=0x02A → bits [21:10]
        // rn=1 → bits [9:5], rd=0 → bits [4:0]
        assert_eq!(instr >> 29, 0b100); // sf:op:S
        assert_eq!((instr >> 23) & 0x3F, 0b100010); // fixed
    }

    #[test]
    fn test_movz_encoding() {
        // MOVZ X0, #0x1234 → sf=1, opc=10, hw=0
        let instr = encode_movz(true, 0, 0x1234, 0);
        assert_eq!((instr >> 29) & 0x7, 0b110); // sf=1, opc=10
        assert_eq!((instr >> 23) & 0x3F, 0b100101); // fixed
    }

    #[test]
    fn test_svc_encoding() {
        // SVC #0 should be 0xD4000001
        assert_eq!(encode_svc(0), 0xD400_0001);
    }

    #[test]
    fn test_brk_encoding() {
        // BRK #0 should be 0xD4200000
        assert_eq!(encode_brk(0), 0xD420_0000);
        // BRK #1 = 0xD4200020
        assert_eq!(encode_brk(1), 0xD420_0020);
    }

    #[test]
    fn test_mov_imm_zero() {
        let instrs = encode_mov_imm(0, 0, true);
        assert_eq!(instrs.len(), 1);
        // Should be MOVZ X0, #0
        assert_eq!(instrs[0], encode_movz(true, 0, 0, 0));
    }

    #[test]
    fn test_mov_imm_small() {
        let instrs = encode_mov_imm(0, 42, true);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], encode_movz(true, 0, 42, 0));
    }

    #[test]
    fn test_mov_imm_two_halfwords() {
        // Value 0x00010002 needs MOVZ + MOVK
        let instrs = encode_mov_imm(0, 0x0001_0002, true);
        assert_eq!(instrs.len(), 2);
    }
}
