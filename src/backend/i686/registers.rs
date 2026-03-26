//! # i686 (32-bit x86) Register Definitions
//!
//! This module defines **all** register constants, register classification sets,
//! register name lookup functions, and register property queries for the i686
//! (32-bit x86) architecture.
//!
//! ## Architecture Characteristics
//!
//! The i686 has fundamentally different register constraints compared to x86-64:
//!
//! - **8 General-Purpose Registers** (GPRs): EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
//!   - Encoded in 3 bits (0–7) in the ModR/M byte — **no REX prefix exists**
//!   - ESP (stack pointer) and EBP (frame pointer) are **reserved** — never allocatable
//!   - Only 6 GPRs are available for the register allocator
//!
//! - **x87 FPU Stack** (ST(0)–ST(7)):
//!   - Stack-based floating-point, **not** directly addressable like SSE/AVX registers
//!   - ST(0) is always the top-of-stack; arithmetic operates on ST(0)/ST(1)
//!   - Floating-point function return values are in ST(0) (cdecl ABI)
//!
//! - **Sub-Register Variants**:
//!   - 32-bit: EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
//!   - 16-bit: AX, CX, DX, BX, SP, BP, SI, DI (with `0x66` operand-size prefix)
//!   - 8-bit low: AL, CL, DL, BL (low byte of EAX–EBX)
//!   - 8-bit high: AH, CH, DH, BH (high byte of AX–BX)
//!   - **No** SPL/BPL/SIL/DIL — those require REX, which i686 does not have
//!
//! - **cdecl Calling Convention**:
//!   - ALL function arguments are passed on the stack (no register arguments)
//!   - Integer return value in EAX (32-bit) or EDX:EAX pair (64-bit `long long`)
//!   - Floating-point return in ST(0)
//!
//! ## Register Numbering
//!
//! | Index | 32-bit | 16-bit | 8-bit low | 8-bit high | Role                     |
//! |-------|--------|--------|-----------|------------|--------------------------|
//! | 0     | EAX    | AX     | AL        | AH (idx 4)| Accumulator, return value|
//! | 1     | ECX    | CX     | CL        | CH (idx 5)| Counter, shift amounts   |
//! | 2     | EDX    | DX     | DL        | DH (idx 6)| Data, EDX:EAX for 64-bit |
//! | 3     | EBX    | BX     | BL        | BH (idx 7)| Base, GOT ptr in PIC     |
//! | 4     | ESP    | SP     | —         | —          | Stack pointer (RESERVED) |
//! | 5     | EBP    | BP     | —         | —          | Frame pointer (RESERVED) |
//! | 6     | ESI    | SI     | —         | —          | Source index             |
//! | 7     | EDI    | DI     | —         | —          | Destination index        |
//!
//! FPU registers use indices 100–107 to avoid collision with GPR encoding.
//!
//! ## Zero-Dependency
//!
//! This module depends only on [`crate::backend::traits::RegisterInfo`] and the
//! Rust standard library.  No external crates are used.

use crate::backend::traits::RegisterInfo;

// ===========================================================================
// 32-bit General-Purpose Registers (GPRs) — ModR/M encoding (3-bit)
// ===========================================================================

/// EAX — accumulator register, integer return value (cdecl ABI).
///
/// ModR/M encoding: `0b000` (index 0).  Used for the result of MUL/DIV
/// instructions, return value storage, and general-purpose computation.
pub const EAX: u16 = 0;

/// ECX — counter register, shift count source (CL).
///
/// ModR/M encoding: `0b001` (index 1).  The CL sub-register is the implicit
/// operand for variable-count shift and rotate instructions (SHL, SHR, SAR,
/// ROL, ROR, RCL, RCR).
pub const ECX: u16 = 1;

/// EDX — data register, high word of 64-bit results.
///
/// ModR/M encoding: `0b010` (index 2).  Forms the EDX:EAX pair for 64-bit
/// multiplication results (MUL/IMUL), division dividends (DIV/IDIV), and
/// `long long` return values in cdecl.
pub const EDX: u16 = 2;

/// EBX — base register, callee-saved, GOT pointer in PIC mode.
///
/// ModR/M encoding: `0b011` (index 3).  In Position-Independent Code (PIC),
/// EBX conventionally holds the GOT base address after a
/// `call __x86.get_pc_thunk.bx` sequence.  Must be preserved across calls.
pub const EBX: u16 = 3;

/// ESP — stack pointer register (**RESERVED** — never allocatable).
///
/// ModR/M encoding: `0b100` (index 4).  Always points to the top of the
/// current stack frame.  Implicitly modified by PUSH, POP, CALL, RET,
/// ENTER, and LEAVE instructions.  The SIB byte is required when ESP
/// appears in certain ModR/M addressing modes.
pub const ESP: u16 = 4;

/// EBP — frame pointer register (**RESERVED** — callee-saved).
///
/// ModR/M encoding: `0b101` (index 5).  Used as the frame base pointer
/// in the standard cdecl prologue (`push ebp; mov ebp, esp`).  Stack
/// arguments are accessed at positive offsets from EBP (`[EBP+8]` = first
/// argument), and local variables at negative offsets (`[EBP-4]`).
pub const EBP: u16 = 5;

/// ESI — source index register, callee-saved.
///
/// ModR/M encoding: `0b110` (index 6).  Historically used as the source
/// pointer for string instructions (MOVS, CMPS, LODS).  Available for
/// general allocation but must be preserved by the callee.
pub const ESI: u16 = 6;

/// EDI — destination index register, callee-saved.
///
/// ModR/M encoding: `0b111` (index 7).  Historically used as the
/// destination pointer for string instructions (MOVS, STOS, SCAS).
/// Available for general allocation but must be preserved by the callee.
pub const EDI: u16 = 7;

// ===========================================================================
// 16-bit Sub-Register Constants
// ===========================================================================

/// AX — 16-bit sub-register of EAX (use with `0x66` operand-size prefix).
pub const AX: u16 = 0;

/// CX — 16-bit sub-register of ECX.
pub const CX: u16 = 1;

/// DX — 16-bit sub-register of EDX.
pub const DX: u16 = 2;

/// BX — 16-bit sub-register of EBX.
pub const BX: u16 = 3;

/// SP — 16-bit sub-register of ESP.
pub const SP: u16 = 4;

/// BP — 16-bit sub-register of EBP.
pub const BP: u16 = 5;

/// SI — 16-bit sub-register of ESI.
pub const SI: u16 = 6;

/// DI — 16-bit sub-register of EDI.
pub const DI: u16 = 7;

// ===========================================================================
// 8-bit Sub-Register Constants — Low Byte
// ===========================================================================

/// AL — low byte of EAX (bits 7:0).
///
/// Used for byte-sized operations, SETcc results, and as the implicit
/// source for single-byte MUL/DIV instructions.
pub const AL: u16 = 0;

/// CL — low byte of ECX (bits 7:0).
///
/// Implicit shift-count operand for variable-count shifts (SHL, SHR, SAR).
pub const CL: u16 = 1;

/// DL — low byte of EDX (bits 7:0).
pub const DL: u16 = 2;

/// BL — low byte of EBX (bits 7:0).
pub const BL: u16 = 3;

// ===========================================================================
// 8-bit Sub-Register Constants — High Byte
// ===========================================================================
//
// On i686, ModR/M register indices 4–7 in 8-bit operand mode encode the
// HIGH bytes of AX–BX (AH, CH, DH, BH), NOT SPL/BPL/SIL/DIL (which
// require a REX prefix and exist only on x86-64).

/// AH — high byte of AX (bits 15:8 of EAX).
pub const AH: u16 = 4;

/// CH — high byte of CX (bits 15:8 of ECX).
pub const CH: u16 = 5;

/// DH — high byte of DX (bits 15:8 of EDX).
pub const DH: u16 = 6;

/// BH — high byte of BX (bits 15:8 of EBX).
pub const BH: u16 = 7;

// ===========================================================================
// x87 FPU Stack Registers
// ===========================================================================
//
// The x87 FPU uses a stack model: ST(0) is the top of the floating-point
// register stack.  Arithmetic instructions operate on ST(0) and optionally
// ST(i).  These are NOT directly addressable like GPRs — they are accessed
// through FPU-specific opcodes (FLD, FSTP, FADD, etc.).
//
// We use indices 100–107 to clearly distinguish FPU registers from GPR
// indices (0–7).  The actual FPU register field in instruction encoding
// uses values 0–7 embedded in the opcode byte.

/// ST(0) — top of the x87 FPU stack.
///
/// The primary accumulator for floating-point operations.  In the cdecl
/// ABI, `float`, `double`, and `long double` return values are placed in
/// ST(0).
pub const ST0: u16 = 100;

/// ST(1) — second element on the x87 FPU stack.
pub const ST1: u16 = 101;

/// ST(2) — third element on the x87 FPU stack.
pub const ST2: u16 = 102;

/// ST(3) — fourth element on the x87 FPU stack.
pub const ST3: u16 = 103;

/// ST(4) — fifth element on the x87 FPU stack.
pub const ST4: u16 = 104;

/// ST(5) — sixth element on the x87 FPU stack.
pub const ST5: u16 = 105;

/// ST(6) — seventh element on the x87 FPU stack.
pub const ST6: u16 = 106;

/// ST(7) — eighth (bottom) element on the x87 FPU stack.
pub const ST7: u16 = 107;

// ===========================================================================
// Register Classification Sets
// ===========================================================================

/// Registers available for the register allocator (excludes ESP, EBP, ECX).
///
/// Only 5 of the 8 GPRs are allocatable — ESP is the stack pointer,
/// EBP is the frame pointer, and ECX is **reserved** as the dedicated
/// spill scratch register (analogous to R11 on x86-64).  Excluding ECX
/// from allocation ensures that spill code (which loads/stores via ECX)
/// never clobbers a live value.
pub const ALLOCATABLE_GPRS: &[u16] = &[EAX, EDX, EBX, ESI, EDI];

/// Callee-saved registers: the callee must preserve these across calls.
///
/// The cdecl ABI requires that EBX, ESI, EDI, and EBP are preserved by
/// the called function.  If the register allocator assigns any of these
/// registers, the prologue must save them and the epilogue must restore them.
pub const CALLEE_SAVED: &[u16] = &[EBX, ESI, EDI, EBP];

/// Caller-saved (volatile) registers: may be clobbered by any call.
///
/// The caller must save EAX, ECX, and EDX before a call if it needs their
/// values afterwards.  These are the "scratch" registers in cdecl.
pub const CALLER_SAVED: &[u16] = &[EAX, ECX, EDX];

/// Reserved registers — **never** allocated by the register allocator.
///
/// ESP is the hardware stack pointer; EBP is the frame pointer used for
/// stack argument access (`[EBP+8]`, `[EBP+12]`, etc.) and local variable
/// access (`[EBP-4]`, `[EBP-8]`, etc.).  ECX is reserved as the dedicated
/// spill scratch register (analogous to R11 on x86-64).
pub const RESERVED: &[u16] = &[ESP, EBP, ECX];

/// Return value registers for 32-bit integer/pointer returns.
///
/// In the cdecl ABI, scalar integer and pointer return values (≤32 bits)
/// are placed in EAX.
pub const RETURN_GPRS: &[u16] = &[EAX];

/// Extended return value registers for 64-bit integer returns (`long long`).
///
/// In the cdecl ABI, 64-bit integer return values use the EDX:EAX register
/// pair: the low 32 bits in EAX and the high 32 bits in EDX.
pub const RETURN_GPRS_64BIT: &[u16] = &[EAX, EDX];

/// Total number of general-purpose registers in the i686 architecture.
pub const NUM_GPRS: usize = 8;

/// Number of GPRs available for allocation (excludes ESP, EBP, ECX).
pub const NUM_ALLOCATABLE_GPRS: usize = 5;

// ===========================================================================
// Register Name Lookup Functions
// ===========================================================================

/// Returns the AT&T-syntax name of a 32-bit GPR.
///
/// # Arguments
///
/// * `reg` — GPR index (0–7 corresponding to EAX–EDI).
///
/// # Returns
///
/// The lowercase register name (e.g., `"eax"`, `"ecx"`), or `"unknown"`
/// if `reg` is outside the valid GPR range.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(reg_name_32(EAX), "eax");
/// assert_eq!(reg_name_32(ESP), "esp");
/// assert_eq!(reg_name_32(99), "unknown");
/// ```
#[inline]
pub fn reg_name_32(reg: u16) -> &'static str {
    match reg {
        EAX => "eax",
        ECX => "ecx",
        EDX => "edx",
        EBX => "ebx",
        ESP => "esp",
        EBP => "ebp",
        ESI => "esi",
        EDI => "edi",
        _ => "unknown",
    }
}

/// Returns the AT&T-syntax name of a 16-bit sub-register.
///
/// # Arguments
///
/// * `reg` — Register index (0–7 corresponding to AX–DI).
///
/// # Returns
///
/// The lowercase register name (e.g., `"ax"`, `"cx"`), or `"unknown"`
/// if `reg` is outside the valid range.
#[inline]
pub fn reg_name_16(reg: u16) -> &'static str {
    match reg {
        0 => "ax",
        1 => "cx",
        2 => "dx",
        3 => "bx",
        4 => "sp",
        5 => "bp",
        6 => "si",
        7 => "di",
        _ => "unknown",
    }
}

/// Returns the AT&T-syntax name of an 8-bit register.
///
/// On i686, indices 0–3 are the low bytes (AL, CL, DL, BL) and indices
/// 4–7 are the high bytes (AH, CH, DH, BH).  There are **no** SPL, BPL,
/// SIL, or DIL registers on i686 — those require REX (x86-64 only).
///
/// # Arguments
///
/// * `reg` — 8-bit register index (0–7).
///
/// # Returns
///
/// The lowercase register name, or `"unknown"` if outside the valid range.
#[inline]
pub fn reg_name_8(reg: u16) -> &'static str {
    match reg {
        AL => "al",
        CL => "cl",
        DL => "dl",
        BL => "bl",
        AH => "ah",
        CH => "ch",
        DH => "dh",
        BH => "bh",
        _ => "unknown",
    }
}

/// Returns the AT&T-syntax name of an x87 FPU stack register.
///
/// # Arguments
///
/// * `reg` — FPU register constant (100–107 corresponding to ST(0)–ST(7)).
///
/// # Returns
///
/// The register name in `st(N)` format, or `"unknown"` if `reg` is not
/// a valid FPU register constant.
#[inline]
pub fn fpu_reg_name(reg: u16) -> &'static str {
    match reg {
        ST0 => "st(0)",
        ST1 => "st(1)",
        ST2 => "st(2)",
        ST3 => "st(3)",
        ST4 => "st(4)",
        ST5 => "st(5)",
        ST6 => "st(6)",
        ST7 => "st(7)",
        _ => "unknown",
    }
}

// ===========================================================================
// Register Property Queries
// ===========================================================================

/// Returns `true` if the given register is callee-saved (EBX, ESI, EDI, EBP).
///
/// Callee-saved registers must be preserved by the called function.  If used,
/// the function's prologue must save them and the epilogue must restore them.
#[inline]
pub fn is_callee_saved(reg: u16) -> bool {
    CALLEE_SAVED.contains(&reg)
}

/// Returns `true` if the given register is caller-saved (EAX, ECX, EDX).
///
/// Caller-saved registers may be freely clobbered by any called function.
/// The caller is responsible for saving values before a call if needed.
#[inline]
pub fn is_caller_saved(reg: u16) -> bool {
    CALLER_SAVED.contains(&reg)
}

/// Returns `true` if the given register is reserved (ESP, EBP).
///
/// Reserved registers are **never** assigned by the register allocator.
/// They serve dedicated architectural purposes (stack pointer, frame pointer).
#[inline]
pub fn is_reserved(reg: u16) -> bool {
    RESERVED.contains(&reg)
}

/// Returns `true` if the given register is an x87 FPU stack register.
///
/// FPU registers use indices 100–107 (ST(0)–ST(7)).
#[inline]
pub fn is_fpu_reg(reg: u16) -> bool {
    (ST0..=ST7).contains(&reg)
}

/// Returns `true` if the given register is a general-purpose register.
///
/// GPRs use indices 0–7 (EAX–EDI).
#[inline]
pub fn is_gpr(reg: u16) -> bool {
    reg <= EDI
}

/// Returns the 8-bit low sub-register index for a 32-bit GPR, if one exists.
///
/// On i686, only EAX–EBX (indices 0–3) have addressable 8-bit low byte
/// sub-registers (AL, CL, DL, BL).  ESP, EBP, ESI, and EDI do **not**
/// have 8-bit sub-registers on i686 — SPL, BPL, SIL, and DIL require
/// a REX prefix and exist only on x86-64.
///
/// # Arguments
///
/// * `reg` — 32-bit GPR index (0–7).
///
/// # Returns
///
/// * `Some(low_reg)` for EAX→AL, ECX→CL, EDX→DL, EBX→BL.
/// * `None` for ESP, EBP, ESI, EDI, or any non-GPR index.
#[inline]
pub fn low_byte_reg(reg: u16) -> Option<u16> {
    match reg {
        EAX => Some(AL),
        ECX => Some(CL),
        EDX => Some(DL),
        EBX => Some(BL),
        _ => None, // ESI, EDI, ESP, EBP have no 8-bit sub-registers on i686
    }
}

/// Extracts the FPU stack index (0–7) from an x87 FPU register constant.
///
/// The returned index is the value used in the FPU instruction opcode to
/// address the specific stack position (e.g., `FSTP ST(3)` encodes index 3).
///
/// # Arguments
///
/// * `reg` — Register constant (should be one of ST0–ST7, i.e., 100–107).
///
/// # Returns
///
/// * `Some(index)` where index ∈ 0..=7 if `reg` is a valid FPU register.
/// * `None` if `reg` is not an FPU register constant.
#[inline]
pub fn fpu_stack_index(reg: u16) -> Option<u8> {
    if is_fpu_reg(reg) {
        Some((reg - ST0) as u8)
    } else {
        None
    }
}

// ===========================================================================
// RegisterInfo Construction
// ===========================================================================

/// Constructs the [`RegisterInfo`] descriptor for the i686 architecture.
///
/// This function is called by the i686 `ArchCodegen` implementation's
/// `register_info()` method.  It describes the register file for the
/// register allocator:
///
/// - **6 allocatable GPRs**: EAX, ECX, EDX, EBX, ESI, EDI
/// - **0 allocatable FPRs**: x87 FPU is stack-based, not directly allocatable
/// - **4 callee-saved**: EBX, ESI, EDI, EBP
/// - **3 caller-saved**: EAX, ECX, EDX
/// - **2 reserved**: ESP (stack pointer), EBP (frame pointer)
/// - **0 argument GPRs**: cdecl passes ALL arguments on the stack
/// - **0 argument FPRs**: cdecl has no FP register arguments
/// - **1 return GPR**: EAX (32-bit integer returns)
/// - **0 return FPRs**: float/double returns use ST(0), handled separately
///
/// # cdecl ABI Key Point
///
/// Unlike x86-64 (which passes up to 6 integer args in registers), the cdecl
/// convention passes **all** function arguments on the stack.  This means
/// `argument_gpr` and `argument_fpr` are both empty vectors.
pub fn i686_register_info() -> RegisterInfo {
    RegisterInfo {
        allocatable_gpr: ALLOCATABLE_GPRS.to_vec(),
        allocatable_fpr: Vec::new(), // x87 FPU is stack-based, not directly allocatable
        callee_saved: CALLEE_SAVED.to_vec(),
        caller_saved: CALLER_SAVED.to_vec(),
        reserved: RESERVED.to_vec(),
        argument_gpr: Vec::new(), // cdecl: NO register arguments — all args on stack
        argument_fpr: Vec::new(), // cdecl: NO FP register arguments
        return_gpr: RETURN_GPRS.to_vec(),
        return_fpr: Vec::new(), // float returns in ST(0), handled separately by ABI module
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpr_indices() {
        // Verify all 8 GPRs have the correct ModR/M encoding values (0-7)
        assert_eq!(EAX, 0);
        assert_eq!(ECX, 1);
        assert_eq!(EDX, 2);
        assert_eq!(EBX, 3);
        assert_eq!(ESP, 4);
        assert_eq!(EBP, 5);
        assert_eq!(ESI, 6);
        assert_eq!(EDI, 7);
    }

    #[test]
    fn test_no_gpr_exceeds_3bit_encoding() {
        // On i686, all GPR indices must fit in 3 bits (0-7).
        // There are NO R8-R15 registers (those require REX prefix).
        for &reg in ALLOCATABLE_GPRS {
            assert!(reg <= 7, "GPR index {} exceeds 3-bit limit", reg);
        }
        for &reg in CALLEE_SAVED {
            assert!(
                reg <= 7,
                "Callee-saved register index {} exceeds 3-bit limit",
                reg
            );
        }
        for &reg in CALLER_SAVED {
            assert!(
                reg <= 7,
                "Caller-saved register index {} exceeds 3-bit limit",
                reg
            );
        }
        for &reg in RESERVED {
            assert!(
                reg <= 7,
                "Reserved register index {} exceeds 3-bit limit",
                reg
            );
        }
    }

    #[test]
    fn test_16bit_registers_same_encoding() {
        // 16-bit registers use the same indices as 32-bit, just different prefix
        assert_eq!(AX, EAX);
        assert_eq!(CX, ECX);
        assert_eq!(DX, EDX);
        assert_eq!(BX, EBX);
        assert_eq!(SP, ESP);
        assert_eq!(BP, EBP);
        assert_eq!(SI, ESI);
        assert_eq!(DI, EDI);
    }

    #[test]
    fn test_8bit_low_registers() {
        assert_eq!(AL, 0);
        assert_eq!(CL, 1);
        assert_eq!(DL, 2);
        assert_eq!(BL, 3);
    }

    #[test]
    fn test_8bit_high_registers() {
        // On i686, 8-bit indices 4-7 are AH/CH/DH/BH (NOT SPL/BPL/SIL/DIL)
        assert_eq!(AH, 4);
        assert_eq!(CH, 5);
        assert_eq!(DH, 6);
        assert_eq!(BH, 7);
    }

    #[test]
    fn test_fpu_registers() {
        // FPU registers must have distinct indices from GPRs
        assert_eq!(ST0, 100);
        assert_eq!(ST1, 101);
        assert_eq!(ST2, 102);
        assert_eq!(ST3, 103);
        assert_eq!(ST4, 104);
        assert_eq!(ST5, 105);
        assert_eq!(ST6, 106);
        assert_eq!(ST7, 107);

        // Verify no overlap with GPR indices
        for i in 0u16..=7 {
            assert!(!is_fpu_reg(i));
        }
        for i in 100u16..=107 {
            assert!(is_fpu_reg(i));
            assert!(!is_gpr(i));
        }
    }

    #[test]
    fn test_allocatable_gprs_excludes_reserved() {
        // ESP, EBP, and ECX (spill scratch) must NOT be in the allocatable set
        assert!(!ALLOCATABLE_GPRS.contains(&ESP));
        assert!(!ALLOCATABLE_GPRS.contains(&EBP));
        assert!(!ALLOCATABLE_GPRS.contains(&ECX));
        assert_eq!(ALLOCATABLE_GPRS.len(), NUM_ALLOCATABLE_GPRS);
        assert_eq!(NUM_ALLOCATABLE_GPRS, 5);
    }

    #[test]
    fn test_callee_saved_set() {
        assert!(CALLEE_SAVED.contains(&EBX));
        assert!(CALLEE_SAVED.contains(&ESI));
        assert!(CALLEE_SAVED.contains(&EDI));
        assert!(CALLEE_SAVED.contains(&EBP));
        assert_eq!(CALLEE_SAVED.len(), 4);
    }

    #[test]
    fn test_caller_saved_set() {
        assert!(CALLER_SAVED.contains(&EAX));
        assert!(CALLER_SAVED.contains(&ECX));
        assert!(CALLER_SAVED.contains(&EDX));
        assert_eq!(CALLER_SAVED.len(), 3);
    }

    #[test]
    fn test_reserved_set() {
        assert!(RESERVED.contains(&ESP));
        assert!(RESERVED.contains(&EBP));
        assert!(RESERVED.contains(&ECX)); // ECX reserved as spill scratch
        assert_eq!(RESERVED.len(), 3);
    }

    #[test]
    fn test_return_gprs() {
        assert_eq!(RETURN_GPRS, &[EAX]);
        assert_eq!(RETURN_GPRS_64BIT, &[EAX, EDX]);
    }

    #[test]
    fn test_num_gprs() {
        assert_eq!(NUM_GPRS, 8);
        assert_eq!(NUM_ALLOCATABLE_GPRS, 5);
        // 8 total - 3 reserved (ESP, EBP, ECX) = 5 allocatable
        assert_eq!(NUM_GPRS - RESERVED.len(), NUM_ALLOCATABLE_GPRS);
    }

    #[test]
    fn test_reg_name_32() {
        assert_eq!(reg_name_32(EAX), "eax");
        assert_eq!(reg_name_32(ECX), "ecx");
        assert_eq!(reg_name_32(EDX), "edx");
        assert_eq!(reg_name_32(EBX), "ebx");
        assert_eq!(reg_name_32(ESP), "esp");
        assert_eq!(reg_name_32(EBP), "ebp");
        assert_eq!(reg_name_32(ESI), "esi");
        assert_eq!(reg_name_32(EDI), "edi");
        assert_eq!(reg_name_32(99), "unknown");
    }

    #[test]
    fn test_reg_name_16() {
        assert_eq!(reg_name_16(AX), "ax");
        assert_eq!(reg_name_16(CX), "cx");
        assert_eq!(reg_name_16(DX), "dx");
        assert_eq!(reg_name_16(BX), "bx");
        assert_eq!(reg_name_16(SP), "sp");
        assert_eq!(reg_name_16(BP), "bp");
        assert_eq!(reg_name_16(SI), "si");
        assert_eq!(reg_name_16(DI), "di");
        assert_eq!(reg_name_16(99), "unknown");
    }

    #[test]
    fn test_reg_name_8() {
        assert_eq!(reg_name_8(AL), "al");
        assert_eq!(reg_name_8(CL), "cl");
        assert_eq!(reg_name_8(DL), "dl");
        assert_eq!(reg_name_8(BL), "bl");
        assert_eq!(reg_name_8(AH), "ah");
        assert_eq!(reg_name_8(CH), "ch");
        assert_eq!(reg_name_8(DH), "dh");
        assert_eq!(reg_name_8(BH), "bh");
        assert_eq!(reg_name_8(99), "unknown");
    }

    #[test]
    fn test_fpu_reg_name() {
        assert_eq!(fpu_reg_name(ST0), "st(0)");
        assert_eq!(fpu_reg_name(ST1), "st(1)");
        assert_eq!(fpu_reg_name(ST2), "st(2)");
        assert_eq!(fpu_reg_name(ST3), "st(3)");
        assert_eq!(fpu_reg_name(ST4), "st(4)");
        assert_eq!(fpu_reg_name(ST5), "st(5)");
        assert_eq!(fpu_reg_name(ST6), "st(6)");
        assert_eq!(fpu_reg_name(ST7), "st(7)");
        assert_eq!(fpu_reg_name(0), "unknown");
        assert_eq!(fpu_reg_name(200), "unknown");
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
    fn test_is_caller_saved() {
        assert!(is_caller_saved(EAX));
        assert!(is_caller_saved(ECX));
        assert!(is_caller_saved(EDX));
        assert!(!is_caller_saved(EBX));
        assert!(!is_caller_saved(ESI));
        assert!(!is_caller_saved(EDI));
        assert!(!is_caller_saved(ESP));
        assert!(!is_caller_saved(EBP));
    }

    #[test]
    fn test_is_reserved() {
        assert!(is_reserved(ESP));
        assert!(is_reserved(EBP));
        assert!(is_reserved(ECX)); // ECX reserved as spill scratch
        assert!(!is_reserved(EAX));
        assert!(!is_reserved(EDX));
        assert!(!is_reserved(EBX));
        assert!(!is_reserved(ESI));
        assert!(!is_reserved(EDI));
    }

    #[test]
    fn test_is_fpu_reg() {
        for i in 0u16..=99 {
            assert!(!is_fpu_reg(i), "index {} should not be FPU", i);
        }
        for i in 100u16..=107 {
            assert!(is_fpu_reg(i), "index {} should be FPU", i);
        }
        assert!(!is_fpu_reg(108));
        assert!(!is_fpu_reg(u16::MAX));
    }

    #[test]
    fn test_is_gpr() {
        for i in 0u16..=7 {
            assert!(is_gpr(i), "index {} should be GPR", i);
        }
        assert!(!is_gpr(8));
        assert!(!is_gpr(100));
        assert!(!is_gpr(u16::MAX));
    }

    #[test]
    fn test_low_byte_reg() {
        // EAX-EBX have addressable 8-bit low sub-registers
        assert_eq!(low_byte_reg(EAX), Some(AL));
        assert_eq!(low_byte_reg(ECX), Some(CL));
        assert_eq!(low_byte_reg(EDX), Some(DL));
        assert_eq!(low_byte_reg(EBX), Some(BL));

        // ESP, EBP, ESI, EDI have NO 8-bit sub-registers on i686
        assert_eq!(low_byte_reg(ESP), None);
        assert_eq!(low_byte_reg(EBP), None);
        assert_eq!(low_byte_reg(ESI), None);
        assert_eq!(low_byte_reg(EDI), None);

        // Out-of-range indices
        assert_eq!(low_byte_reg(8), None);
        assert_eq!(low_byte_reg(100), None);
    }

    #[test]
    fn test_fpu_stack_index() {
        assert_eq!(fpu_stack_index(ST0), Some(0));
        assert_eq!(fpu_stack_index(ST1), Some(1));
        assert_eq!(fpu_stack_index(ST2), Some(2));
        assert_eq!(fpu_stack_index(ST3), Some(3));
        assert_eq!(fpu_stack_index(ST4), Some(4));
        assert_eq!(fpu_stack_index(ST5), Some(5));
        assert_eq!(fpu_stack_index(ST6), Some(6));
        assert_eq!(fpu_stack_index(ST7), Some(7));

        // Non-FPU registers return None
        assert_eq!(fpu_stack_index(EAX), None);
        assert_eq!(fpu_stack_index(ESP), None);
        assert_eq!(fpu_stack_index(0), None);
        assert_eq!(fpu_stack_index(99), None);
        assert_eq!(fpu_stack_index(108), None);
    }

    #[test]
    fn test_i686_register_info() {
        let info = i686_register_info();

        // Verify allocatable GPRs (5 registers: ESP, EBP reserved; ECX = spill scratch)
        assert_eq!(info.allocatable_gpr.len(), 5);
        assert!(info.allocatable_gpr.contains(&EAX));
        assert!(!info.allocatable_gpr.contains(&ECX)); // ECX = spill scratch
        assert!(info.allocatable_gpr.contains(&EDX));
        assert!(info.allocatable_gpr.contains(&EBX));
        assert!(info.allocatable_gpr.contains(&ESI));
        assert!(info.allocatable_gpr.contains(&EDI));
        assert!(!info.allocatable_gpr.contains(&ESP));
        assert!(!info.allocatable_gpr.contains(&EBP));

        // x87 FPU is stack-based, not directly allocatable
        assert!(info.allocatable_fpr.is_empty());

        // Callee-saved: EBX, ESI, EDI, EBP
        assert_eq!(info.callee_saved.len(), 4);
        assert!(info.callee_saved.contains(&EBX));
        assert!(info.callee_saved.contains(&ESI));
        assert!(info.callee_saved.contains(&EDI));
        assert!(info.callee_saved.contains(&EBP));

        // Caller-saved: EAX, ECX, EDX
        assert_eq!(info.caller_saved.len(), 3);
        assert!(info.caller_saved.contains(&EAX));
        assert!(info.caller_saved.contains(&ECX));
        assert!(info.caller_saved.contains(&EDX));

        // Reserved: ESP, EBP, ECX (spill scratch)
        assert_eq!(info.reserved.len(), 3);
        assert!(info.reserved.contains(&ESP));
        assert!(info.reserved.contains(&EBP));
        assert!(info.reserved.contains(&ECX));

        // cdecl: NO register arguments — all arguments on the stack
        assert!(info.argument_gpr.is_empty());
        assert!(info.argument_fpr.is_empty());

        // Return value: EAX for 32-bit integers
        assert_eq!(info.return_gpr.len(), 1);
        assert!(info.return_gpr.contains(&EAX));

        // FP returns in ST(0), handled separately
        assert!(info.return_fpr.is_empty());
    }

    #[test]
    fn test_register_sets_disjoint() {
        // Verify caller-saved and callee-saved sets are disjoint
        for &reg in CALLER_SAVED {
            assert!(
                !CALLEE_SAVED.contains(&reg),
                "Register {} is in both caller-saved and callee-saved sets",
                reg_name_32(reg)
            );
        }

        // Verify reserved registers are not in allocatable set
        for &reg in RESERVED {
            assert!(
                !ALLOCATABLE_GPRS.contains(&reg),
                "Reserved register {} is in allocatable set",
                reg_name_32(reg)
            );
        }
    }

    #[test]
    fn test_all_gprs_classified() {
        // Every GPR should be either caller-saved, callee-saved, or reserved
        // (with possible overlap for callee-saved + reserved for EBP)
        for i in 0u16..=7 {
            let in_caller = CALLER_SAVED.contains(&i);
            let in_callee = CALLEE_SAVED.contains(&i);
            let in_reserved = RESERVED.contains(&i);
            assert!(
                in_caller || in_callee || in_reserved,
                "GPR {} ({}) is not classified in any set",
                i,
                reg_name_32(i)
            );
        }
    }
}
