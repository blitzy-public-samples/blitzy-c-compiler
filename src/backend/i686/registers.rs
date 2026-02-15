//! i686 (IA-32) register definitions for the BCC compiler backend.
//!
//! This module provides named constants for all physical registers in the
//! Intel 32-bit (i686) architecture, organized as follows:
//!
//! - **32-bit GPRs** (EAX–EDI): 8 general-purpose registers
//! - **16-bit sub-registers** (AX–DI): lower 16 bits of each GPR
//! - **8-bit sub-registers** (AL/AH–BL/BH): byte halves of the first 4 GPRs
//! - **x87 FPU stack registers** (ST0–ST7): floating-point stack
//! - **EFLAGS**: implicit condition flags register
//!
//! # PhysReg Encoding Layout
//!
//! The flat `PhysReg(u16)` namespace for i686 is partitioned as:
//!
//! | Range  | Registers         | Count |
//! |--------|-------------------|-------|
//! | 0–7    | EAX–EDI (32-bit)  | 8     |
//! | 8–15   | AX–DI (16-bit)    | 8     |
//! | 16–23  | AL,CL,DL,BL,AH…BH| 8     |
//! | 24–31  | ST(0)–ST(7) (x87) | 8     |
//! | 32     | EFLAGS            | 1     |
//!
//! # Key Architectural Constraints (vs. x86-64)
//!
//! - Only 8 GPRs are available (no R8–R15, no REX prefix)
//! - Only EAX, ECX, EDX, EBX have 8-bit sub-register access (AL/AH, etc.)
//! - The cdecl calling convention passes ALL arguments on the stack —
//!   `INTEGER_ARG_REGS` and `FLOAT_ARG_REGS` are empty arrays
//! - EBX is the PIC GOT base register when `-fPIC` is active
//! - Floating-point return values go in ST(0)
//!
//! # Register Encoding for ModR/M Byte
//!
//! The 3-bit encoding value (0–7) maps directly to the register numbering
//! used in ModR/M, SIB, and opcode reg-field slots:
//!
//! | Encoding | 32-bit | 16-bit | 8-bit lo | 8-bit hi |
//! |----------|--------|--------|----------|----------|
//! | 0        | EAX    | AX     | AL       | AH       |
//! | 1        | ECX    | CX     | CL       | CH       |
//! | 2        | EDX    | DX     | DL       | DH       |
//! | 3        | EBX    | BX     | BL       | BH       |
//! | 4        | ESP    | SP     | (SPL*)   | —        |
//! | 5        | EBP    | BP     | (BPL*)   | —        |
//! | 6        | ESI    | SI     | (SIL*)   | —        |
//! | 7        | EDI    | DI     | (DIL*)   | —        |
//!
//! *SPL/BPL/SIL/DIL require REX prefix (x86-64 only), NOT available on i686.

use crate::backend::traits::PhysReg;

// ===========================================================================
// 32-bit General-Purpose Registers (GPRs)
// ===========================================================================

/// EAX — accumulator register. Used as implicit operand for MUL/DIV/IMUL/IDIV,
/// carries the integer return value (cdecl), and has the shortest encoding for
/// many instructions (e.g., `ADD EAX, imm32` is 5 bytes vs. 6 for other GPRs).
/// Caller-saved.
pub const EAX: PhysReg = PhysReg(0);

/// ECX — counter register. Used as implicit shift/rotate count (`SHL reg, CL`),
/// loop counter for `LOOP`/`REP` prefix instructions. Caller-saved.
pub const ECX: PhysReg = PhysReg(1);

/// EDX — data register. Forms the high 32 bits of 64-bit results in MUL/IMUL
/// and the high 32 bits of the dividend in DIV/IDIV (`EDX:EAX`). Also used for
/// I/O port addressing (`IN`/`OUT`). Caller-saved.
pub const EDX: PhysReg = PhysReg(2);

/// EBX — base register. Callee-saved. Serves as the PIC GOT base register
/// when generating position-independent code (`-fPIC`). Must be preserved
/// across function calls per the cdecl/System V i386 ABI.
pub const EBX: PhysReg = PhysReg(3);

/// ESP — stack pointer. Points to the top of the current stack frame.
/// NOT allocatable by the register allocator — it is always reserved for
/// stack management. Hardware-enforced alignment requirements apply.
pub const ESP: PhysReg = PhysReg(4);

/// EBP — frame pointer. Callee-saved. When frame pointer usage is enabled
/// (the default), EBP points to the base of the current stack frame and is
/// NOT allocatable. When frame pointer omission is active (`-fomit-frame-pointer`),
/// EBP becomes available for general-purpose register allocation.
pub const EBP: PhysReg = PhysReg(5);

/// ESI — source index register. Callee-saved. Used as implicit source operand
/// for string instructions (`MOVSB`, `CMPSB`, `LODSB`, etc.).
pub const ESI: PhysReg = PhysReg(6);

/// EDI — destination index register. Callee-saved. Used as implicit destination
/// operand for string instructions (`MOVSB`, `STOSB`, `SCASB`, etc.).
pub const EDI: PhysReg = PhysReg(7);

// ===========================================================================
// 16-bit Sub-Register Aliases
// ===========================================================================
// These represent the lower 16 bits of the corresponding 32-bit GPR.
// Writing to a 16-bit register does NOT zero-extend the upper 16 bits
// (unlike x86-64's 32-bit write → 64-bit zero extension).

/// AX — lower 16 bits of EAX.
pub const AX: PhysReg = PhysReg(8);

/// CX — lower 16 bits of ECX.
pub const CX: PhysReg = PhysReg(9);

/// DX — lower 16 bits of EDX.
pub const DX: PhysReg = PhysReg(10);

/// BX — lower 16 bits of EBX.
pub const BX: PhysReg = PhysReg(11);

/// SP — lower 16 bits of ESP.
pub const SP: PhysReg = PhysReg(12);

/// BP — lower 16 bits of EBP.
pub const BP: PhysReg = PhysReg(13);

/// SI — lower 16 bits of ESI.
pub const SI: PhysReg = PhysReg(14);

/// DI — lower 16 bits of EDI.
pub const DI: PhysReg = PhysReg(15);

// ===========================================================================
// 8-bit Sub-Register Aliases
// ===========================================================================
// On i686, only the first four GPRs (EAX, ECX, EDX, EBX) have addressable
// 8-bit sub-registers. ESI, EDI, EBP, ESP do NOT have byte-addressable
// sub-registers in 32-bit mode (SPL/BPL/SIL/DIL require REX, x86-64 only).

/// AL — low byte (bits 7:0) of EAX.
pub const AL: PhysReg = PhysReg(16);

/// CL — low byte (bits 7:0) of ECX. Used as implicit shift count operand.
pub const CL: PhysReg = PhysReg(17);

/// DL — low byte (bits 7:0) of EDX.
pub const DL: PhysReg = PhysReg(18);

/// BL — low byte (bits 7:0) of EBX.
pub const BL: PhysReg = PhysReg(19);

/// AH — high byte (bits 15:8) of AX (and EAX).
pub const AH: PhysReg = PhysReg(20);

/// CH — high byte (bits 15:8) of CX (and ECX).
pub const CH: PhysReg = PhysReg(21);

/// DH — high byte (bits 15:8) of DX (and EDX).
pub const DH: PhysReg = PhysReg(22);

/// BH — high byte (bits 15:8) of BX (and EBX).
pub const BH: PhysReg = PhysReg(23);

// ===========================================================================
// x87 FPU Stack Registers
// ===========================================================================
// The x87 FPU operates as a stack of eight 80-bit extended-precision
// registers. ST(0) is always the stack top. Most x87 instructions
// implicitly operate on ST(0). Floating-point return values in cdecl
// are placed in ST(0). Arguments are passed on the memory stack, NOT
// in FPU registers.

/// ST(0) — x87 FPU stack top. Floating-point return value register in cdecl.
pub const ST0: PhysReg = PhysReg(24);

/// ST(1) — second element on the x87 FPU stack.
pub const ST1: PhysReg = PhysReg(25);

/// ST(2) — third element on the x87 FPU stack.
pub const ST2: PhysReg = PhysReg(26);

/// ST(3) — fourth element on the x87 FPU stack.
pub const ST3: PhysReg = PhysReg(27);

/// ST(4) — fifth element on the x87 FPU stack.
pub const ST4: PhysReg = PhysReg(28);

/// ST(5) — sixth element on the x87 FPU stack.
pub const ST5: PhysReg = PhysReg(29);

/// ST(6) — seventh element on the x87 FPU stack.
pub const ST6: PhysReg = PhysReg(30);

/// ST(7) — eighth (bottom) element on the x87 FPU stack.
pub const ST7: PhysReg = PhysReg(31);

// ===========================================================================
// EFLAGS — Condition Flags Register (implicit)
// ===========================================================================

/// EFLAGS — processor status/condition flags register.
///
/// Contains the result flags (CF, ZF, SF, OF, PF, AF) set by arithmetic and
/// comparison instructions (CMP, TEST, SUB, ADD, etc.) and consumed by
/// conditional instructions (Jcc, SETcc, CMOVcc). Not directly allocatable
/// by the register allocator — it is implicitly read/written by many
/// instructions.
pub const EFLAGS: PhysReg = PhysReg(32);

// ===========================================================================
// Register Classification Arrays
// ===========================================================================

/// Total number of distinct physical register slots (including sub-registers
/// and EFLAGS).
pub const NUM_PHYS_REGS: usize = 33;

/// Callee-saved registers: must be preserved across function calls.
/// Per the System V i386 ABI (cdecl), EBX, ESI, EDI, and EBP are callee-saved.
pub const CALLEE_SAVED: [PhysReg; 4] = [EBX, ESI, EDI, EBP];

/// Caller-saved (volatile) registers: may be clobbered by function calls.
/// The caller must save these before a call if their values are needed after.
pub const CALLER_SAVED: [PhysReg; 3] = [EAX, ECX, EDX];

/// Allocatable integer registers (with frame pointer in use).
/// Excludes ESP (always stack pointer) and EBP (frame pointer when active).
/// These are the GPRs available for the register allocator to assign virtual
/// registers to physical registers.
pub const ALLOCATABLE_INT: [PhysReg; 6] = [EAX, ECX, EDX, EBX, ESI, EDI];

/// Allocatable integer registers when frame pointer is omitted.
/// When `-fomit-frame-pointer` is active, EBP becomes available for general
/// allocation, giving us one extra register.
pub const ALLOCATABLE_INT_NO_FP: [PhysReg; 7] = [EAX, ECX, EDX, EBX, ESI, EDI, EBP];

/// Integer argument registers for the cdecl calling convention.
/// cdecl passes ALL arguments on the stack — this array is intentionally empty.
/// This contrasts with x86-64 System V which passes the first 6 integer args
/// in registers (RDI, RSI, RDX, RCX, R8, R9).
pub const INTEGER_ARG_REGS: [PhysReg; 0] = [];

/// Floating-point argument registers for the cdecl calling convention.
/// cdecl passes ALL floating-point arguments on the stack — this array is
/// intentionally empty. FP return values are placed in ST(0).
pub const FLOAT_ARG_REGS: [PhysReg; 0] = [];

/// Registers that have addressable 8-bit sub-registers on i686.
/// Only EAX, ECX, EDX, and EBX have byte-addressable halves (AL/AH, CL/CH,
/// DL/DH, BL/BH). ESI, EDI, EBP, and ESP do NOT have 8-bit sub-registers
/// in 32-bit mode (SIL/DIL/BPL/SPL require the REX prefix, x86-64 only).
/// This is critical for the register allocator when it needs to emit 8-bit
/// operations (e.g., `SETCC`, byte-width loads/stores).
pub const BYTE_ADDRESSABLE: [PhysReg; 4] = [EAX, ECX, EDX, EBX];

// ===========================================================================
// Register Name Lookup Functions
// ===========================================================================

/// Lookup table for 32-bit GPR names, indexed by ModR/M encoding (0–7).
const GPR32_NAMES: [&str; 8] = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"];

/// Lookup table for 16-bit sub-register names, indexed by ModR/M encoding (0–7).
const GPR16_NAMES: [&str; 8] = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"];

/// Lookup table for 8-bit low-byte sub-register names (encoding 0–3 only).
const GPR8LO_NAMES: [&str; 4] = ["al", "cl", "dl", "bl"];

/// Lookup table for 8-bit high-byte sub-register names (encoding 0–3 only).
const GPR8HI_NAMES: [&str; 4] = ["ah", "ch", "dh", "bh"];

/// Lookup table for x87 FPU stack register names (ST(0)–ST(7)).
const FPU_NAMES: [&str; 8] = [
    "st(0)", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)",
];

/// Returns the 32-bit register name for a GPR.
///
/// For sub-registers (16-bit or 8-bit), the parent register's 32-bit name
/// is returned.
///
/// # Arguments
/// * `reg` — A PhysReg in the 32-bit GPR range (0–7), or any sub-register
///   whose parent is a 32-bit GPR.
///
/// # Panics
/// Panics if `reg` is not a valid 32-bit GPR or a sub-register whose parent
/// is a 32-bit GPR.
///
/// # Examples
/// ```ignore
/// assert_eq!(reg_name_32(EAX), "eax");
/// assert_eq!(reg_name_32(EDI), "edi");
/// ```
pub fn reg_name_32(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if idx < 8 {
        GPR32_NAMES[idx]
    } else if idx < 16 {
        // 16-bit sub-register → return parent 32-bit name
        GPR32_NAMES[idx - 8]
    } else if idx < 20 {
        // 8-bit low sub-register → return parent 32-bit name
        GPR32_NAMES[idx - 16]
    } else if idx < 24 {
        // 8-bit high sub-register → return parent 32-bit name
        GPR32_NAMES[idx - 20]
    } else {
        panic!(
            "reg_name_32: PhysReg({}) is not a GPR or GPR sub-register",
            reg.0
        )
    }
}

/// Returns the 16-bit register name for a GPR or 16-bit sub-register.
///
/// # Arguments
/// * `reg` — A PhysReg in the 32-bit GPR range (0–7) or 16-bit range (8–15).
///
/// # Panics
/// Panics if `reg` is outside the GPR/16-bit sub-register range.
///
/// # Examples
/// ```ignore
/// assert_eq!(reg_name_16(AX), "ax");
/// assert_eq!(reg_name_16(EAX), "ax");  // 32-bit maps to 16-bit name
/// ```
pub fn reg_name_16(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if idx < 8 {
        // 32-bit GPR → return 16-bit equivalent name
        GPR16_NAMES[idx]
    } else if idx < 16 {
        GPR16_NAMES[idx - 8]
    } else {
        panic!(
            "reg_name_16: PhysReg({}) is not a GPR or 16-bit sub-register",
            reg.0
        )
    }
}

/// Returns the low-byte (8-bit) register name for a GPR.
///
/// Only EAX, ECX, EDX, EBX (encodings 0–3) have low-byte sub-registers
/// on i686. Attempting to get the low-byte name of ESI/EDI/EBP/ESP panics.
///
/// # Arguments
/// * `reg` — A PhysReg representing one of the first four GPRs (0–3),
///   their 16-bit aliases (8–11), or their 8-bit-low aliases (16–19).
///
/// # Panics
/// Panics if `reg` does not have a low-byte sub-register.
///
/// # Examples
/// ```ignore
/// assert_eq!(reg_name_8lo(AL), "al");
/// assert_eq!(reg_name_8lo(EAX), "al");  // 32-bit maps to low byte
/// ```
pub fn reg_name_8lo(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if idx < 4 {
        // 32-bit GPR (EAX-EBX) → return low-byte name
        GPR8LO_NAMES[idx]
    } else if (8..12).contains(&idx) {
        // 16-bit sub-register (AX-BX) → return low-byte name
        GPR8LO_NAMES[idx - 8]
    } else if (16..20).contains(&idx) {
        // Already a low-byte sub-register (AL-BL)
        GPR8LO_NAMES[idx - 16]
    } else {
        panic!(
            "reg_name_8lo: PhysReg({}) does not have a low-byte sub-register on i686",
            reg.0
        )
    }
}

/// Returns the high-byte (8-bit) register name for a GPR.
///
/// Only EAX, ECX, EDX, EBX (encodings 0–3) have high-byte sub-registers
/// (AH, CH, DH, BH) on i686.
///
/// # Arguments
/// * `reg` — A PhysReg representing one of the first four GPRs (0–3),
///   their 16-bit aliases (8–11), or their 8-bit-high aliases (20–23).
///
/// # Panics
/// Panics if `reg` does not have a high-byte sub-register.
///
/// # Examples
/// ```ignore
/// assert_eq!(reg_name_8hi(AH), "ah");
/// assert_eq!(reg_name_8hi(EAX), "ah");  // 32-bit maps to high byte
/// ```
pub fn reg_name_8hi(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if idx < 4 {
        // 32-bit GPR (EAX-EBX) → return high-byte name
        GPR8HI_NAMES[idx]
    } else if (8..12).contains(&idx) {
        // 16-bit sub-register (AX-BX) → return high-byte name
        GPR8HI_NAMES[idx - 8]
    } else if (20..24).contains(&idx) {
        // Already a high-byte sub-register (AH-BH)
        GPR8HI_NAMES[idx - 20]
    } else {
        panic!(
            "reg_name_8hi: PhysReg({}) does not have a high-byte sub-register on i686",
            reg.0
        )
    }
}

/// Returns the x87 FPU stack register name.
///
/// # Arguments
/// * `reg` — A PhysReg in the FPU range (24–31), corresponding to ST(0)–ST(7).
///
/// # Panics
/// Panics if `reg` is not an FPU register.
///
/// # Examples
/// ```ignore
/// assert_eq!(fpu_reg_name(ST0), "st(0)");
/// assert_eq!(fpu_reg_name(ST7), "st(7)");
/// ```
pub fn fpu_reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if (24..32).contains(&idx) {
        FPU_NAMES[idx - 24]
    } else {
        panic!(
            "fpu_reg_name: PhysReg({}) is not an x87 FPU register",
            reg.0
        )
    }
}

/// Returns the canonical assembly name for any i686 physical register.
///
/// Dispatches to the appropriate name function based on the PhysReg range:
/// - 0–7: 32-bit GPR names (eax, ecx, edx, ebx, esp, ebp, esi, edi)
/// - 8–15: 16-bit sub-register names (ax, cx, dx, bx, sp, bp, si, di)
/// - 16–19: 8-bit low names (al, cl, dl, bl)
/// - 20–23: 8-bit high names (ah, ch, dh, bh)
/// - 24–31: x87 FPU names (st(0)–st(7))
/// - 32: "eflags"
///
/// # Panics
/// Panics if `reg` is outside all known register ranges.
///
/// # Examples
/// ```ignore
/// assert_eq!(reg_name(EAX), "eax");
/// assert_eq!(reg_name(AL), "al");
/// assert_eq!(reg_name(ST0), "st(0)");
/// assert_eq!(reg_name(EFLAGS), "eflags");
/// ```
pub fn reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    match idx {
        0..=7 => GPR32_NAMES[idx],
        8..=15 => GPR16_NAMES[idx - 8],
        16..=19 => GPR8LO_NAMES[idx - 16],
        20..=23 => GPR8HI_NAMES[idx - 20],
        24..=31 => FPU_NAMES[idx - 24],
        32 => "eflags",
        _ => panic!("reg_name: unknown PhysReg({})", reg.0),
    }
}

// ===========================================================================
// Register Property Query Functions
// ===========================================================================

/// Returns `true` if the given register is a 32-bit general-purpose register.
///
/// GPRs are PhysReg(0) through PhysReg(7): EAX, ECX, EDX, EBX, ESP, EBP,
/// ESI, EDI.
///
/// # Examples
/// ```ignore
/// assert!(is_gpr(EAX));
/// assert!(is_gpr(ESP));
/// assert!(!is_gpr(AL));   // 8-bit sub-register, not a 32-bit GPR
/// assert!(!is_gpr(ST0));  // FPU register
/// ```
#[inline]
pub fn is_gpr(reg: PhysReg) -> bool {
    reg.0 < 8
}

/// Returns `true` if the given register is an x87 FPU stack register.
///
/// FPU registers are PhysReg(24) through PhysReg(31): ST(0)–ST(7).
///
/// # Examples
/// ```ignore
/// assert!(is_fpu(ST0));
/// assert!(is_fpu(ST7));
/// assert!(!is_fpu(EAX));
/// ```
#[inline]
pub fn is_fpu(reg: PhysReg) -> bool {
    reg.0 >= 24 && reg.0 < 32
}

/// Returns `true` if the given register is callee-saved per the cdecl ABI.
///
/// Callee-saved registers are: EBX, ESI, EDI, EBP. These must be preserved
/// by the called function if it uses them.
///
/// # Examples
/// ```ignore
/// assert!(is_callee_saved(EBX));
/// assert!(is_callee_saved(EBP));
/// assert!(!is_callee_saved(EAX));
/// assert!(!is_callee_saved(ECX));
/// ```
#[inline]
pub fn is_callee_saved(reg: PhysReg) -> bool {
    matches!(reg, r if r == EBX || r == ESI || r == EDI || r == EBP)
}

/// Returns `true` if the given register can be used by the register allocator.
///
/// ESP is never allocatable (it is the dedicated stack pointer). All other
/// 32-bit GPRs are potentially allocatable (EBP's allocatability depends on
/// frame pointer usage, but this function returns `true` for EBP — the
/// caller should use `ALLOCATABLE_INT` vs `ALLOCATABLE_INT_NO_FP` to respect
/// the frame pointer setting).
///
/// x87 FPU registers, EFLAGS, and sub-registers are not directly allocatable
/// in the integer register allocator.
///
/// # Examples
/// ```ignore
/// assert!(is_allocatable(EAX));
/// assert!(is_allocatable(EBP));  // may be allocatable if FP omitted
/// assert!(!is_allocatable(ESP)); // never allocatable
/// ```
#[inline]
pub fn is_allocatable(reg: PhysReg) -> bool {
    // Only 32-bit GPRs are allocatable for integer values, ESP is excluded
    reg.0 < 8 && reg != ESP
}

/// Returns `true` if the given 32-bit GPR has addressable 8-bit sub-registers.
///
/// On i686, only EAX, ECX, EDX, and EBX have 8-bit sub-register access
/// (AL/AH, CL/CH, DL/DH, BL/BH). ESI, EDI, EBP, and ESP do NOT have
/// byte-accessible sub-registers in 32-bit mode.
///
/// This is important for the register allocator when allocating registers
/// for 8-bit operations (e.g., `SETCC r/m8`, byte-width MOV).
///
/// # Examples
/// ```ignore
/// assert!(has_byte_subreg(EAX));
/// assert!(has_byte_subreg(EBX));
/// assert!(!has_byte_subreg(ESI));
/// assert!(!has_byte_subreg(ESP));
/// ```
#[inline]
pub fn has_byte_subreg(reg: PhysReg) -> bool {
    reg.0 < 4
}

/// Returns the 3-bit register encoding used in ModR/M, SIB, and opcode reg
/// fields for instruction encoding.
///
/// The encoding maps directly to the standard x86 numbering:
///
/// | GPR | Encoding |
/// |-----|----------|
/// | EAX | 0        |
/// | ECX | 1        |
/// | EDX | 2        |
/// | EBX | 3        |
/// | ESP | 4        |
/// | EBP | 5        |
/// | ESI | 6        |
/// | EDI | 7        |
///
/// For sub-registers, the instruction-context encoding is returned:
/// - 16-bit sub-registers use the same encoding as the parent GPR
/// - 8-bit low sub-registers (AL, CL, DL, BL) encode as 0–3
/// - 8-bit high sub-registers (AH, CH, DH, BH) encode as 4–7
///
/// For x87 FPU registers, the stack index (0–7) is returned.
///
/// # Panics
/// Panics if `reg` is EFLAGS or an unknown register.
///
/// # Examples
/// ```ignore
/// assert_eq!(encoding(EAX), 0);
/// assert_eq!(encoding(EDI), 7);
/// assert_eq!(encoding(AL), 0);
/// assert_eq!(encoding(AH), 4);
/// assert_eq!(encoding(ST0), 0);
/// ```
pub fn encoding(reg: PhysReg) -> u8 {
    let idx = reg.0;
    match idx {
        // 32-bit GPRs: direct encoding 0–7
        0..=7 => idx as u8,
        // 16-bit sub-registers: same encoding as parent GPR
        8..=15 => (idx - 8) as u8,
        // 8-bit low sub-registers (AL=0, CL=1, DL=2, BL=3)
        16..=19 => (idx - 16) as u8,
        // 8-bit high sub-registers (AH=4, CH=5, DH=6, BH=7)
        // In the ModR/M encoding for 8-bit operands:
        //   AH=4, CH=5, DH=6, BH=7
        20..=23 => (idx - 20 + 4) as u8,
        // x87 FPU stack registers: ST(i) encodes as i
        24..=31 => (idx - 24) as u8,
        _ => panic!("encoding: PhysReg({}) has no ModR/M encoding", idx),
    }
}

/// Returns the 32-bit parent register for any sub-register.
///
/// For 32-bit GPRs, returns the register itself. For 16-bit and 8-bit
/// sub-registers, returns the corresponding 32-bit GPR. For x87 FPU
/// registers and EFLAGS, returns the register itself (no parent hierarchy).
///
/// # Examples
/// ```ignore
/// assert_eq!(parent_reg(AL), EAX);
/// assert_eq!(parent_reg(AH), EAX);
/// assert_eq!(parent_reg(AX), EAX);
/// assert_eq!(parent_reg(EAX), EAX);
/// assert_eq!(parent_reg(ST0), ST0);
/// assert_eq!(parent_reg(EFLAGS), EFLAGS);
/// ```
pub fn parent_reg(reg: PhysReg) -> PhysReg {
    let idx = reg.0;
    match idx {
        // 32-bit GPRs are their own parent
        0..=7 => reg,
        // 16-bit sub-registers → 32-bit parent
        8..=15 => PhysReg(idx - 8),
        // 8-bit low sub-registers → 32-bit parent
        16..=19 => PhysReg(idx - 16),
        // 8-bit high sub-registers → 32-bit parent
        20..=23 => PhysReg(idx - 20),
        // x87 FPU registers are their own parent (no hierarchy)
        24..=31 => reg,
        // EFLAGS is its own parent
        32 => reg,
        _ => panic!("parent_reg: unknown PhysReg({})", idx),
    }
}

// ===========================================================================
// Additional Utility Functions
// ===========================================================================

/// Returns the 8-bit low sub-register PhysReg for a given 32-bit GPR.
///
/// Only valid for EAX, ECX, EDX, EBX (the first four GPRs that have
/// byte-addressable sub-registers on i686).
///
/// # Panics
/// Debug-panics if the register doesn't have a low-byte sub-register.
///
/// # Examples
/// ```ignore
/// assert_eq!(low_byte_reg(EAX), AL);
/// assert_eq!(low_byte_reg(ECX), CL);
/// ```
#[inline]
pub fn low_byte_reg(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 < 4,
        "low_byte_reg: PhysReg({}) has no low-byte sub-register on i686",
        reg.0
    );
    PhysReg(reg.0 + 16)
}

/// Returns the 8-bit high sub-register PhysReg for a given 32-bit GPR.
///
/// Only valid for EAX, ECX, EDX, EBX (the first four GPRs that have
/// byte-addressable sub-registers on i686).
///
/// # Panics
/// Debug-panics if the register doesn't have a high-byte sub-register.
///
/// # Examples
/// ```ignore
/// assert_eq!(high_byte_reg(EAX), AH);
/// assert_eq!(high_byte_reg(ECX), CH);
/// ```
#[inline]
pub fn high_byte_reg(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 < 4,
        "high_byte_reg: PhysReg({}) has no high-byte sub-register on i686",
        reg.0
    );
    PhysReg(reg.0 + 20)
}

/// Returns the 16-bit sub-register PhysReg for a given 32-bit GPR.
///
/// Valid for all 8 GPRs (EAX–EDI).
///
/// # Panics
/// Debug-panics if the register is not a 32-bit GPR.
///
/// # Examples
/// ```ignore
/// assert_eq!(word_reg(EAX), AX);
/// assert_eq!(word_reg(ESP), SP);
/// ```
#[inline]
pub fn word_reg(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 < 8,
        "word_reg: PhysReg({}) is not a 32-bit GPR",
        reg.0
    );
    PhysReg(reg.0 + 8)
}

/// Returns the `RegisterClass` for the given physical register.
///
/// Classification mapping:
/// - ESP → `StackPointer`
/// - EBP → `FramePointer`
/// - Other GPRs and their sub-registers → `GeneralPurpose`
/// - ST(0)–ST(7) → `FloatingPoint`
/// - EFLAGS → `GeneralPurpose` (sentinel classification)
///
/// The register allocator should reference the classification arrays
/// (`ALLOCATABLE_INT`, `CALLEE_SAVED`, etc.) rather than this function
/// for allocation decisions.
pub fn register_class(reg: PhysReg) -> crate::backend::traits::RegisterClass {
    use crate::backend::traits::RegisterClass;

    let parent = parent_reg(reg);
    match parent.0 {
        4 => RegisterClass::StackPointer,
        5 => RegisterClass::FramePointer,
        0..=3 | 6..=7 => RegisterClass::GeneralPurpose,
        24..=31 => RegisterClass::FloatingPoint,
        _ => RegisterClass::GeneralPurpose,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Constant value tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpr_constants() {
        assert_eq!(EAX.0, 0);
        assert_eq!(ECX.0, 1);
        assert_eq!(EDX.0, 2);
        assert_eq!(EBX.0, 3);
        assert_eq!(ESP.0, 4);
        assert_eq!(EBP.0, 5);
        assert_eq!(ESI.0, 6);
        assert_eq!(EDI.0, 7);
    }

    #[test]
    fn test_16bit_sub_registers() {
        assert_eq!(AX.0, 8);
        assert_eq!(CX.0, 9);
        assert_eq!(DX.0, 10);
        assert_eq!(BX.0, 11);
        assert_eq!(SP.0, 12);
        assert_eq!(BP.0, 13);
        assert_eq!(SI.0, 14);
        assert_eq!(DI.0, 15);
    }

    #[test]
    fn test_8bit_sub_registers() {
        assert_eq!(AL.0, 16);
        assert_eq!(CL.0, 17);
        assert_eq!(DL.0, 18);
        assert_eq!(BL.0, 19);
        assert_eq!(AH.0, 20);
        assert_eq!(CH.0, 21);
        assert_eq!(DH.0, 22);
        assert_eq!(BH.0, 23);
    }

    #[test]
    fn test_fpu_registers() {
        assert_eq!(ST0.0, 24);
        assert_eq!(ST1.0, 25);
        assert_eq!(ST2.0, 26);
        assert_eq!(ST3.0, 27);
        assert_eq!(ST4.0, 28);
        assert_eq!(ST5.0, 29);
        assert_eq!(ST6.0, 30);
        assert_eq!(ST7.0, 31);
    }

    #[test]
    fn test_eflags() {
        assert_eq!(EFLAGS.0, 32);
    }

    // -----------------------------------------------------------------------
    // Classification array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_callee_saved() {
        assert_eq!(CALLEE_SAVED.len(), 4);
        assert!(CALLEE_SAVED.contains(&EBX));
        assert!(CALLEE_SAVED.contains(&ESI));
        assert!(CALLEE_SAVED.contains(&EDI));
        assert!(CALLEE_SAVED.contains(&EBP));
        // Ensure caller-saved regs are NOT in callee-saved
        assert!(!CALLEE_SAVED.contains(&EAX));
        assert!(!CALLEE_SAVED.contains(&ECX));
        assert!(!CALLEE_SAVED.contains(&EDX));
    }

    #[test]
    fn test_caller_saved() {
        assert_eq!(CALLER_SAVED.len(), 3);
        assert!(CALLER_SAVED.contains(&EAX));
        assert!(CALLER_SAVED.contains(&ECX));
        assert!(CALLER_SAVED.contains(&EDX));
    }

    #[test]
    fn test_allocatable_int() {
        assert_eq!(ALLOCATABLE_INT.len(), 6);
        assert!(ALLOCATABLE_INT.contains(&EAX));
        assert!(ALLOCATABLE_INT.contains(&ECX));
        assert!(ALLOCATABLE_INT.contains(&EDX));
        assert!(ALLOCATABLE_INT.contains(&EBX));
        assert!(ALLOCATABLE_INT.contains(&ESI));
        assert!(ALLOCATABLE_INT.contains(&EDI));
        assert!(!ALLOCATABLE_INT.contains(&ESP));
        assert!(!ALLOCATABLE_INT.contains(&EBP));
    }

    #[test]
    fn test_allocatable_int_no_fp() {
        assert_eq!(ALLOCATABLE_INT_NO_FP.len(), 7);
        assert!(ALLOCATABLE_INT_NO_FP.contains(&EBP));
        assert!(!ALLOCATABLE_INT_NO_FP.contains(&ESP));
    }

    #[test]
    fn test_arg_regs_empty() {
        // cdecl passes everything on the stack
        assert_eq!(INTEGER_ARG_REGS.len(), 0);
        assert_eq!(FLOAT_ARG_REGS.len(), 0);
    }

    #[test]
    fn test_byte_addressable() {
        assert_eq!(BYTE_ADDRESSABLE.len(), 4);
        assert!(BYTE_ADDRESSABLE.contains(&EAX));
        assert!(BYTE_ADDRESSABLE.contains(&ECX));
        assert!(BYTE_ADDRESSABLE.contains(&EDX));
        assert!(BYTE_ADDRESSABLE.contains(&EBX));
        assert!(!BYTE_ADDRESSABLE.contains(&ESI));
        assert!(!BYTE_ADDRESSABLE.contains(&EDI));
        assert!(!BYTE_ADDRESSABLE.contains(&EBP));
        assert!(!BYTE_ADDRESSABLE.contains(&ESP));
    }

    // -----------------------------------------------------------------------
    // Register name tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reg_name_32_all() {
        assert_eq!(reg_name_32(EAX), "eax");
        assert_eq!(reg_name_32(ECX), "ecx");
        assert_eq!(reg_name_32(EDX), "edx");
        assert_eq!(reg_name_32(EBX), "ebx");
        assert_eq!(reg_name_32(ESP), "esp");
        assert_eq!(reg_name_32(EBP), "ebp");
        assert_eq!(reg_name_32(ESI), "esi");
        assert_eq!(reg_name_32(EDI), "edi");
    }

    #[test]
    fn test_reg_name_32_from_sub_regs() {
        // 16-bit sub-registers map to parent 32-bit name
        assert_eq!(reg_name_32(AX), "eax");
        assert_eq!(reg_name_32(DI), "edi");
        // 8-bit low sub-registers map to parent 32-bit name
        assert_eq!(reg_name_32(AL), "eax");
        assert_eq!(reg_name_32(BL), "ebx");
        // 8-bit high sub-registers map to parent 32-bit name
        assert_eq!(reg_name_32(AH), "eax");
        assert_eq!(reg_name_32(BH), "ebx");
    }

    #[test]
    fn test_reg_name_16_all() {
        assert_eq!(reg_name_16(AX), "ax");
        assert_eq!(reg_name_16(CX), "cx");
        assert_eq!(reg_name_16(DX), "dx");
        assert_eq!(reg_name_16(BX), "bx");
        assert_eq!(reg_name_16(SP), "sp");
        assert_eq!(reg_name_16(BP), "bp");
        assert_eq!(reg_name_16(SI), "si");
        assert_eq!(reg_name_16(DI), "di");
    }

    #[test]
    fn test_reg_name_16_from_32bit() {
        // 32-bit GPR maps to its 16-bit name
        assert_eq!(reg_name_16(EAX), "ax");
        assert_eq!(reg_name_16(EDI), "di");
    }

    #[test]
    fn test_reg_name_8lo_all() {
        assert_eq!(reg_name_8lo(AL), "al");
        assert_eq!(reg_name_8lo(CL), "cl");
        assert_eq!(reg_name_8lo(DL), "dl");
        assert_eq!(reg_name_8lo(BL), "bl");
    }

    #[test]
    fn test_reg_name_8lo_from_parents() {
        // 32-bit → 8-bit-low mapping
        assert_eq!(reg_name_8lo(EAX), "al");
        assert_eq!(reg_name_8lo(EBX), "bl");
        // 16-bit → 8-bit-low mapping
        assert_eq!(reg_name_8lo(AX), "al");
        assert_eq!(reg_name_8lo(BX), "bl");
    }

    #[test]
    fn test_reg_name_8hi_all() {
        assert_eq!(reg_name_8hi(AH), "ah");
        assert_eq!(reg_name_8hi(CH), "ch");
        assert_eq!(reg_name_8hi(DH), "dh");
        assert_eq!(reg_name_8hi(BH), "bh");
    }

    #[test]
    fn test_reg_name_8hi_from_parents() {
        // 32-bit → 8-bit-high mapping
        assert_eq!(reg_name_8hi(EAX), "ah");
        assert_eq!(reg_name_8hi(EBX), "bh");
        // 16-bit → 8-bit-high mapping
        assert_eq!(reg_name_8hi(AX), "ah");
        assert_eq!(reg_name_8hi(BX), "bh");
    }

    #[test]
    fn test_fpu_reg_name_all() {
        assert_eq!(fpu_reg_name(ST0), "st(0)");
        assert_eq!(fpu_reg_name(ST1), "st(1)");
        assert_eq!(fpu_reg_name(ST2), "st(2)");
        assert_eq!(fpu_reg_name(ST3), "st(3)");
        assert_eq!(fpu_reg_name(ST4), "st(4)");
        assert_eq!(fpu_reg_name(ST5), "st(5)");
        assert_eq!(fpu_reg_name(ST6), "st(6)");
        assert_eq!(fpu_reg_name(ST7), "st(7)");
    }

    #[test]
    fn test_reg_name_dispatch_all() {
        // 32-bit
        assert_eq!(reg_name(EAX), "eax");
        assert_eq!(reg_name(EDI), "edi");
        // 16-bit
        assert_eq!(reg_name(AX), "ax");
        assert_eq!(reg_name(DI), "di");
        // 8-bit low
        assert_eq!(reg_name(AL), "al");
        assert_eq!(reg_name(BL), "bl");
        // 8-bit high
        assert_eq!(reg_name(AH), "ah");
        assert_eq!(reg_name(BH), "bh");
        // FPU
        assert_eq!(reg_name(ST0), "st(0)");
        assert_eq!(reg_name(ST7), "st(7)");
        // EFLAGS
        assert_eq!(reg_name(EFLAGS), "eflags");
    }

    // -----------------------------------------------------------------------
    // Property query tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_gpr() {
        for i in 0..8u16 {
            assert!(is_gpr(PhysReg(i)), "PhysReg({}) should be GPR", i);
        }
        for i in 8..33u16 {
            assert!(!is_gpr(PhysReg(i)), "PhysReg({}) should not be GPR", i);
        }
    }

    #[test]
    fn test_is_fpu() {
        for i in 24..32u16 {
            assert!(is_fpu(PhysReg(i)), "PhysReg({}) should be FPU", i);
        }
        for i in 0..24u16 {
            assert!(!is_fpu(PhysReg(i)), "PhysReg({}) should not be FPU", i);
        }
        assert!(!is_fpu(EFLAGS));
    }

    #[test]
    fn test_is_callee_saved() {
        assert!(is_callee_saved(EBX));
        assert!(is_callee_saved(ESI));
        assert!(is_callee_saved(EDI));
        assert!(is_callee_saved(EBP));
        assert!(!is_callee_saved(EAX));
        assert!(!is_callee_saved(ECX));
        assert!(!is_callee_saved(EDX));
        assert!(!is_callee_saved(ESP));
    }

    #[test]
    fn test_is_allocatable() {
        assert!(is_allocatable(EAX));
        assert!(is_allocatable(ECX));
        assert!(is_allocatable(EDX));
        assert!(is_allocatable(EBX));
        assert!(is_allocatable(EBP));
        assert!(is_allocatable(ESI));
        assert!(is_allocatable(EDI));
        assert!(!is_allocatable(ESP));
        // Sub-registers and FPU are not allocatable in integer allocator
        assert!(!is_allocatable(AX));
        assert!(!is_allocatable(AL));
        assert!(!is_allocatable(ST0));
        assert!(!is_allocatable(EFLAGS));
    }

    #[test]
    fn test_has_byte_subreg() {
        assert!(has_byte_subreg(EAX));
        assert!(has_byte_subreg(ECX));
        assert!(has_byte_subreg(EDX));
        assert!(has_byte_subreg(EBX));
        assert!(!has_byte_subreg(ESP));
        assert!(!has_byte_subreg(EBP));
        assert!(!has_byte_subreg(ESI));
        assert!(!has_byte_subreg(EDI));
    }

    // -----------------------------------------------------------------------
    // Encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoding_32bit() {
        assert_eq!(encoding(EAX), 0);
        assert_eq!(encoding(ECX), 1);
        assert_eq!(encoding(EDX), 2);
        assert_eq!(encoding(EBX), 3);
        assert_eq!(encoding(ESP), 4);
        assert_eq!(encoding(EBP), 5);
        assert_eq!(encoding(ESI), 6);
        assert_eq!(encoding(EDI), 7);
    }

    #[test]
    fn test_encoding_16bit() {
        assert_eq!(encoding(AX), 0);
        assert_eq!(encoding(CX), 1);
        assert_eq!(encoding(DX), 2);
        assert_eq!(encoding(BX), 3);
        assert_eq!(encoding(SP), 4);
        assert_eq!(encoding(BP), 5);
        assert_eq!(encoding(SI), 6);
        assert_eq!(encoding(DI), 7);
    }

    #[test]
    fn test_encoding_8bit_low() {
        assert_eq!(encoding(AL), 0);
        assert_eq!(encoding(CL), 1);
        assert_eq!(encoding(DL), 2);
        assert_eq!(encoding(BL), 3);
    }

    #[test]
    fn test_encoding_8bit_high() {
        // In 8-bit context: AH=4, CH=5, DH=6, BH=7
        assert_eq!(encoding(AH), 4);
        assert_eq!(encoding(CH), 5);
        assert_eq!(encoding(DH), 6);
        assert_eq!(encoding(BH), 7);
    }

    #[test]
    fn test_encoding_fpu() {
        assert_eq!(encoding(ST0), 0);
        assert_eq!(encoding(ST1), 1);
        assert_eq!(encoding(ST2), 2);
        assert_eq!(encoding(ST3), 3);
        assert_eq!(encoding(ST4), 4);
        assert_eq!(encoding(ST5), 5);
        assert_eq!(encoding(ST6), 6);
        assert_eq!(encoding(ST7), 7);
    }

    // -----------------------------------------------------------------------
    // Parent register tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parent_reg_identity() {
        // 32-bit GPRs are their own parent
        for i in 0..8u16 {
            let reg = PhysReg(i);
            assert_eq!(parent_reg(reg), reg);
        }
    }

    #[test]
    fn test_parent_reg_16bit() {
        assert_eq!(parent_reg(AX), EAX);
        assert_eq!(parent_reg(CX), ECX);
        assert_eq!(parent_reg(DX), EDX);
        assert_eq!(parent_reg(BX), EBX);
        assert_eq!(parent_reg(SP), ESP);
        assert_eq!(parent_reg(BP), EBP);
        assert_eq!(parent_reg(SI), ESI);
        assert_eq!(parent_reg(DI), EDI);
    }

    #[test]
    fn test_parent_reg_8bit_low() {
        assert_eq!(parent_reg(AL), EAX);
        assert_eq!(parent_reg(CL), ECX);
        assert_eq!(parent_reg(DL), EDX);
        assert_eq!(parent_reg(BL), EBX);
    }

    #[test]
    fn test_parent_reg_8bit_high() {
        assert_eq!(parent_reg(AH), EAX);
        assert_eq!(parent_reg(CH), ECX);
        assert_eq!(parent_reg(DH), EDX);
        assert_eq!(parent_reg(BH), EBX);
    }

    #[test]
    fn test_parent_reg_fpu_identity() {
        for i in 24..32u16 {
            let reg = PhysReg(i);
            assert_eq!(parent_reg(reg), reg);
        }
    }

    #[test]
    fn test_parent_reg_eflags_identity() {
        assert_eq!(parent_reg(EFLAGS), EFLAGS);
    }

    // -----------------------------------------------------------------------
    // Utility function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_low_byte_reg() {
        assert_eq!(low_byte_reg(EAX), AL);
        assert_eq!(low_byte_reg(ECX), CL);
        assert_eq!(low_byte_reg(EDX), DL);
        assert_eq!(low_byte_reg(EBX), BL);
    }

    #[test]
    fn test_high_byte_reg() {
        assert_eq!(high_byte_reg(EAX), AH);
        assert_eq!(high_byte_reg(ECX), CH);
        assert_eq!(high_byte_reg(EDX), DH);
        assert_eq!(high_byte_reg(EBX), BH);
    }

    #[test]
    fn test_word_reg() {
        assert_eq!(word_reg(EAX), AX);
        assert_eq!(word_reg(ECX), CX);
        assert_eq!(word_reg(EDX), DX);
        assert_eq!(word_reg(EBX), BX);
        assert_eq!(word_reg(ESP), SP);
        assert_eq!(word_reg(EBP), BP);
        assert_eq!(word_reg(ESI), SI);
        assert_eq!(word_reg(EDI), DI);
    }

    #[test]
    fn test_register_class() {
        use crate::backend::traits::RegisterClass;

        assert_eq!(register_class(EAX), RegisterClass::GeneralPurpose);
        assert_eq!(register_class(ECX), RegisterClass::GeneralPurpose);
        assert_eq!(register_class(ESI), RegisterClass::GeneralPurpose);
        assert_eq!(register_class(ESP), RegisterClass::StackPointer);
        assert_eq!(register_class(EBP), RegisterClass::FramePointer);
        assert_eq!(register_class(ST0), RegisterClass::FloatingPoint);
        assert_eq!(register_class(ST7), RegisterClass::FloatingPoint);
        // Sub-registers inherit parent's class
        assert_eq!(register_class(AL), RegisterClass::GeneralPurpose);
        assert_eq!(register_class(AX), RegisterClass::GeneralPurpose);
        assert_eq!(register_class(SP), RegisterClass::StackPointer);
        assert_eq!(register_class(BP), RegisterClass::FramePointer);
    }

    // -----------------------------------------------------------------------
    // Edge case / consistency tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_callee_saved_in_allocatable() {
        // All callee-saved regs should appear in one of the allocatable sets
        for &reg in &CALLEE_SAVED {
            assert!(
                ALLOCATABLE_INT.contains(&reg) || ALLOCATABLE_INT_NO_FP.contains(&reg),
                "{:?} is callee-saved but not in any allocatable set",
                reg
            );
        }
    }

    #[test]
    fn test_all_caller_saved_in_allocatable() {
        // All caller-saved regs should appear in allocatable sets
        for &reg in &CALLER_SAVED {
            assert!(
                ALLOCATABLE_INT.contains(&reg),
                "{:?} is caller-saved but not allocatable",
                reg
            );
        }
    }

    #[test]
    fn test_esp_not_in_any_allocatable_set() {
        assert!(!ALLOCATABLE_INT.contains(&ESP));
        assert!(!ALLOCATABLE_INT_NO_FP.contains(&ESP));
    }

    #[test]
    fn test_num_phys_regs() {
        assert_eq!(NUM_PHYS_REGS, 33);
    }

    #[test]
    fn test_encoding_values_in_range() {
        // All 32-bit GPR encodings must be in 0..8
        for i in 0..8u16 {
            let enc = encoding(PhysReg(i));
            assert!(
                enc < 8,
                "encoding for PhysReg({}) is {} (must be < 8)",
                i,
                enc
            );
        }
        // All FPU encodings must be in 0..8
        for i in 24..32u16 {
            let enc = encoding(PhysReg(i));
            assert!(
                enc < 8,
                "encoding for PhysReg({}) is {} (must be < 8)",
                i,
                enc
            );
        }
    }
}
