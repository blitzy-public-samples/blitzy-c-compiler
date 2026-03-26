//! # x86-64 Register Definitions
//!
//! Defines all x86-64 register constants, register sets (callee-saved, caller-saved,
//! argument passing), register name/encoding mappings, and REX prefix helpers.
//!
//! ## Register Encoding
//!
//! Register IDs use `u16` for compatibility with `MachineOperand::Register(u16)` in
//! `traits.rs`. GPR constants 0–15 match the x86-64 ModR/M register field encoding
//! directly. SSE constants 16–31 are offset by 16 so the register allocator can
//! distinguish register classes.
//!
//! ## Architecture Reference
//!
//! - 16 General-Purpose Registers (GPRs): RAX–R15 (64-bit)
//!   - Sub-registers: EAX–R15D (32-bit), AX–R15W (16-bit), AL–R15B (8-bit low)
//!   - Legacy 8-bit high: AH, CH, DH, BH (only for RAX–RBX)
//! - 16 SSE Registers: XMM0–XMM15 (128-bit, used for scalar float/double via SSE2)
//! - Registers R8–R15 and XMM8–XMM15 require a REX prefix bit in instruction encoding

// =============================================================================
// GPR Constants (64-bit) — encoding matches x86-64 ModR/M register field
// =============================================================================

/// RAX — accumulator, return value register, implicit operand for MUL/DIV/CDQE.
pub const RAX: u16 = 0;
/// RCX — counter register, 4th integer argument (System V AMD64), shift count.
pub const RCX: u16 = 1;
/// RDX — data register, 3rd integer argument, high half of 128-bit DIV result.
pub const RDX: u16 = 2;
/// RBX — base register, callee-saved.
pub const RBX: u16 = 3;
/// RSP — stack pointer, reserved (never allocatable).
pub const RSP: u16 = 4;
/// RBP — frame pointer, reserved (callee-saved, handled separately).
pub const RBP: u16 = 5;
/// RSI — source index, 2nd integer argument (System V AMD64).
pub const RSI: u16 = 6;
/// RDI — destination index, 1st integer argument (System V AMD64).
pub const RDI: u16 = 7;
/// R8 — 5th integer argument. Requires REX prefix.
pub const R8: u16 = 8;
/// R9 — 6th integer argument. Requires REX prefix.
pub const R9: u16 = 9;
/// R10 — caller-saved, static chain pointer. Requires REX prefix.
pub const R10: u16 = 10;
/// R11 — caller-saved, scratch. Requires REX prefix.
pub const R11: u16 = 11;
/// R12 — callee-saved. Requires REX prefix.
pub const R12: u16 = 12;
/// R13 — callee-saved. Requires REX prefix.
pub const R13: u16 = 13;
/// R14 — callee-saved. Requires REX prefix.
pub const R14: u16 = 14;
/// R15 — callee-saved. Requires REX prefix.
pub const R15: u16 = 15;

// =============================================================================
// SSE Register Constants — offset by 16 to separate from GPR namespace
// =============================================================================

/// XMM0 — 1st FP argument, 1st FP return value (System V AMD64).
pub const XMM0: u16 = 16;
/// XMM1 — 2nd FP argument, 2nd FP return value.
pub const XMM1: u16 = 17;
/// XMM2 — 3rd FP argument.
pub const XMM2: u16 = 18;
/// XMM3 — 4th FP argument.
pub const XMM3: u16 = 19;
/// XMM4 — 5th FP argument.
pub const XMM4: u16 = 20;
/// XMM5 — 6th FP argument.
pub const XMM5: u16 = 21;
/// XMM6 — 7th FP argument.
pub const XMM6: u16 = 22;
/// XMM7 — 8th FP argument.
pub const XMM7: u16 = 23;
/// XMM8 — caller-saved. Requires REX prefix.
pub const XMM8: u16 = 24;
/// XMM9 — caller-saved. Requires REX prefix.
pub const XMM9: u16 = 25;
/// XMM10 — caller-saved. Requires REX prefix.
pub const XMM10: u16 = 26;
/// XMM11 — caller-saved. Requires REX prefix.
pub const XMM11: u16 = 27;
/// XMM12 — caller-saved. Requires REX prefix.
pub const XMM12: u16 = 28;
/// XMM13 — caller-saved. Requires REX prefix.
pub const XMM13: u16 = 29;
/// XMM14 — caller-saved. Requires REX prefix.
pub const XMM14: u16 = 30;
/// XMM15 — caller-saved. Requires REX prefix.
pub const XMM15: u16 = 31;

// =============================================================================
// Register Count Constants
// =============================================================================

/// Total number of general-purpose registers (RAX–R15).
pub const NUM_GPRS: usize = 16;
/// Total number of SSE registers (XMM0–XMM15).
pub const NUM_SSE: usize = 16;
/// Total number of physical registers (GPRs + SSE).
pub const TOTAL_REGS: usize = NUM_GPRS + NUM_SSE; // 32

// =============================================================================
// Register Sets (Arrays) — used by register allocator and ABI modules
// =============================================================================

/// All allocatable GPRs (excludes RSP and RBP which are reserved).
///
/// Caller-saved registers are listed first as they are preferred for allocation
/// (using them avoids the cost of save/restore in the prologue/epilogue).
/// Allocatable general-purpose registers.
///
/// R11 is **reserved** as a dedicated scratch register for spill
/// load/store code insertion.  The register allocator never assigns
/// R11 to a live interval, so spill code can safely use it to
/// move values between physical registers and stack spill slots
/// without clobbering any allocated value.
pub const ALLOCATABLE_GPRS: [u16; 13] = [
    RAX, RCX, RDX, RSI, RDI, R8, R9, R10, // caller-saved (preferred)
    RBX, R12, R13, R14, R15, // callee-saved
];

/// All allocatable SSE registers (XMM0–XMM14 are allocatable, all are
/// caller-saved).  XMM15 is reserved as the float spill scratch register
/// (analogous to R11 for integer spills).
pub const ALLOCATABLE_SSE: [u16; 15] = [
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
];

/// Callee-saved GPRs — the function prologue must save these if used, and the
/// epilogue must restore them. RBP is callee-saved per the ABI but is handled
/// separately as the frame pointer.
pub const CALLEE_SAVED_GPRS: [u16; 5] = [RBX, R12, R13, R14, R15];

/// Caller-saved GPRs — the caller must save these across function calls if their
/// values are needed after the call. The callee may freely clobber them.
/// Caller-saved GPRs (excluding R11, which is reserved as spill scratch).
pub const CALLER_SAVED_GPRS: [u16; 8] = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10];

/// Reserved registers — never allocated by the register allocator.
/// RSP is the stack pointer; RBP is the frame pointer.
/// Reserved registers: RSP (stack pointer), RBP (frame pointer), and R11
/// (dedicated spill scratch — see `apply_allocation_result` in
/// `generation.rs`).
pub const RESERVED_REGS: [u16; 3] = [RSP, RBP, R11];

/// Integer argument passing registers in the System V AMD64 ABI order.
/// The first 6 integer/pointer arguments are passed in these registers.
pub const ARG_GPRS: [u16; 6] = [RDI, RSI, RDX, RCX, R8, R9];

/// Floating-point argument passing registers in the System V AMD64 ABI order.
/// The first 8 float/double arguments are passed in these registers.
pub const ARG_SSE: [u16; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];

/// Integer return value registers. RAX holds the primary return value;
/// RDX holds the high 64 bits for 128-bit returns or struct returns
/// that fit in two INTEGER eightbytes.
pub const RET_GPRS: [u16; 2] = [RAX, RDX];

/// Floating-point return value registers. XMM0 holds the primary FP return;
/// XMM1 holds the second component for struct returns with two SSE eightbytes.
pub const RET_SSE: [u16; 2] = [XMM0, XMM1];

// =============================================================================
// Sub-Register Encoding Constants
// =============================================================================

/// Operand-size override prefix byte (0x66) for 16-bit operations.
///
/// When present, this prefix overrides the default operand size from 32-bit to
/// 16-bit in 64-bit mode (or from 16-bit to 32-bit in 16-bit mode).
pub const OPERAND_SIZE_PREFIX: u8 = 0x66;

// =============================================================================
// Register Classification Functions
// =============================================================================

/// Returns `true` if `reg` is a general-purpose register (RAX–R15, IDs 0–15).
#[inline]
pub fn is_gpr(reg: u16) -> bool {
    reg < 16
}

/// Returns `true` if `reg` is an SSE register (XMM0–XMM15, IDs 16–31).
#[inline]
pub fn is_sse(reg: u16) -> bool {
    (16..32).contains(&reg)
}

/// Returns the 4-bit hardware encoding (0–15) for a register.
///
/// For GPRs, this is the ModR/M `reg` or `r/m` field value.
/// For SSE registers, this is the XMM register number (subtract 16 from the ID).
/// The lower 3 bits go into the ModR/M field; bit 3 goes into the REX.R/REX.B/REX.X
/// extension bit.
#[inline]
pub fn hw_encoding(reg: u16) -> u8 {
    (reg % 16) as u8
}

/// Returns `true` if this register requires a REX prefix for encoding.
///
/// Registers R8–R15 (GPR IDs 8–15) and XMM8–XMM15 (SSE IDs 24–31) have hardware
/// encodings 8–15, which require the REX.B, REX.R, or REX.X extension bit.
#[inline]
pub fn needs_rex(reg: u16) -> bool {
    hw_encoding(reg) >= 8
}

/// Returns `true` when `reg` requires a mandatory REX prefix for byte-sized
/// operand access on x86-64.
///
/// Registers RSP, RBP, RSI, and RDI (hardware encodings 4–7) map to the
/// legacy high-byte registers %ah, %ch, %dh, %bh in byte context when no
/// REX prefix is present. A REX prefix (even the bare 0x40) redirects the
/// encoding to the low-byte forms %spl, %bpl, %sil, %dil. Without this
/// prefix, byte operations on these registers silently access the wrong
/// physical storage.
#[inline]
pub fn needs_rex_for_byte_reg(reg: u16) -> bool {
    let enc = hw_encoding(reg);
    // hw encodings 4-7 are RSP/RBP/RSI/RDI — they need REX for byte access
    // hw encodings 8+ already need REX (R8-R15) — handled by needs_rex()
    (4..8).contains(&enc)
}

/// Returns `true` if `reg` is callee-saved per the System V AMD64 ABI.
///
/// Callee-saved: RBX, RBP, R12, R13, R14, R15.
/// If a function uses any of these, its prologue must save and epilogue must restore them.
#[inline]
pub fn is_callee_saved(reg: u16) -> bool {
    matches!(reg, RBX | RBP | R12 | R13 | R14 | R15)
}

/// Returns `true` if `reg` is caller-saved per the System V AMD64 ABI.
///
/// Caller-saved: RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11.
/// These registers may be clobbered by any function call. The caller must preserve
/// their values across calls if they are needed afterwards.
#[inline]
pub fn is_caller_saved(reg: u16) -> bool {
    matches!(reg, RAX | RCX | RDX | RSI | RDI | R8 | R9 | R10 | R11)
}

// =============================================================================
// Register Name Mapping Functions
// =============================================================================

/// Returns the 64-bit register name (AT&T/Intel syntax, lowercase).
///
/// # Panics
/// Panics if `reg` is not a valid GPR (ID 0–15).
pub fn reg_name_64(reg: u16) -> &'static str {
    match reg {
        RAX => "rax",
        RCX => "rcx",
        RDX => "rdx",
        RBX => "rbx",
        RSP => "rsp",
        RBP => "rbp",
        RSI => "rsi",
        RDI => "rdi",
        R8 => "r8",
        R9 => "r9",
        R10 => "r10",
        R11 => "r11",
        R12 => "r12",
        R13 => "r13",
        R14 => "r14",
        R15 => "r15",
        _ => panic!("Not a GPR: {}", reg),
    }
}

/// Returns the 32-bit sub-register name.
///
/// Writing to a 32-bit sub-register implicitly zero-extends the result to 64 bits
/// on x86-64 (e.g., `mov eax, 1` clears the upper 32 bits of RAX).
///
/// # Panics
/// Panics if `reg` is not a valid GPR (ID 0–15).
pub fn reg_name_32(reg: u16) -> &'static str {
    match reg {
        RAX => "eax",
        RCX => "ecx",
        RDX => "edx",
        RBX => "ebx",
        RSP => "esp",
        RBP => "ebp",
        RSI => "esi",
        RDI => "edi",
        R8 => "r8d",
        R9 => "r9d",
        R10 => "r10d",
        R11 => "r11d",
        R12 => "r12d",
        R13 => "r13d",
        R14 => "r14d",
        R15 => "r15d",
        _ => panic!("Not a GPR: {}", reg),
    }
}

/// Returns the 16-bit sub-register name.
///
/// # Panics
/// Panics if `reg` is not a valid GPR (ID 0–15).
pub fn reg_name_16(reg: u16) -> &'static str {
    match reg {
        RAX => "ax",
        RCX => "cx",
        RDX => "dx",
        RBX => "bx",
        RSP => "sp",
        RBP => "bp",
        RSI => "si",
        RDI => "di",
        R8 => "r8w",
        R9 => "r9w",
        R10 => "r10w",
        R11 => "r11w",
        R12 => "r12w",
        R13 => "r13w",
        R14 => "r14w",
        R15 => "r15w",
        _ => panic!("Not a GPR: {}", reg),
    }
}

/// Returns the 8-bit low sub-register name.
///
/// On x86-64, all 16 GPRs have an 8-bit low sub-register (AL, CL, …, R15B).
/// Accessing the 8-bit low sub-register of RSP/RBP/RSI/RDI (SPL, BPL, SIL, DIL)
/// requires a REX prefix even though the register number is < 8, because without
/// REX the encodings 4–7 in 8-bit context refer to AH, CH, DH, BH instead.
///
/// # Panics
/// Panics if `reg` is not a valid GPR (ID 0–15).
pub fn reg_name_8(reg: u16) -> &'static str {
    match reg {
        RAX => "al",
        RCX => "cl",
        RDX => "dl",
        RBX => "bl",
        RSP => "spl",
        RBP => "bpl",
        RSI => "sil",
        RDI => "dil",
        R8 => "r8b",
        R9 => "r9b",
        R10 => "r10b",
        R11 => "r11b",
        R12 => "r12b",
        R13 => "r13b",
        R14 => "r14b",
        R15 => "r15b",
        _ => panic!("Not a GPR: {}", reg),
    }
}

/// Returns the SSE register name (e.g., "xmm0", "xmm15").
///
/// # Panics
/// Panics if `reg` is not a valid SSE register (ID 16–31).
pub fn sse_reg_name(reg: u16) -> &'static str {
    match reg {
        XMM0 => "xmm0",
        XMM1 => "xmm1",
        XMM2 => "xmm2",
        XMM3 => "xmm3",
        XMM4 => "xmm4",
        XMM5 => "xmm5",
        XMM6 => "xmm6",
        XMM7 => "xmm7",
        XMM8 => "xmm8",
        XMM9 => "xmm9",
        XMM10 => "xmm10",
        XMM11 => "xmm11",
        XMM12 => "xmm12",
        XMM13 => "xmm13",
        XMM14 => "xmm14",
        XMM15 => "xmm15",
        _ => panic!("Not an SSE register: {}", reg),
    }
}

/// Returns the register name appropriate for the given operand `size` in bytes.
///
/// For SSE registers, the size parameter is ignored (always returns "xmmN").
/// For GPRs, dispatches to the sub-register name based on size:
/// - 8 bytes → 64-bit name (rax, r8, …)
/// - 4 bytes → 32-bit name (eax, r8d, …)
/// - 2 bytes → 16-bit name (ax, r8w, …)
/// - 1 byte  → 8-bit low name (al, r8b, …)
/// - other   → defaults to 64-bit name
///
/// # Panics
/// Panics if `reg` is neither a valid GPR nor a valid SSE register.
pub fn reg_name(reg: u16, size: u8) -> &'static str {
    if is_sse(reg) {
        sse_reg_name(reg)
    } else {
        match size {
            8 => reg_name_64(reg),
            4 => reg_name_32(reg),
            2 => reg_name_16(reg),
            1 => reg_name_8(reg),
            _ => reg_name_64(reg),
        }
    }
}

// =============================================================================
// REX Prefix Construction
// =============================================================================

/// Constructs a REX prefix byte from individual flag bits.
///
/// The REX prefix (0x40–0x4F) is required in 64-bit mode when:
/// - Accessing 64-bit operand sizes (REX.W = 1)
/// - Accessing extended registers R8–R15 or XMM8–XMM15 (REX.R, REX.X, REX.B)
/// - Accessing the new 8-bit registers SPL, BPL, SIL, DIL
///
/// # Arguments
/// - `w` — REX.W: set for 64-bit operand size (bit 3)
/// - `r` — REX.R: extension of the ModR/M `reg` field (bit 2)
/// - `x` — REX.X: extension of the SIB `index` field (bit 1)
/// - `b` — REX.B: extension of the ModR/M `r/m` field or SIB `base` field (bit 0)
///
/// # Returns
/// The REX prefix byte (0x40–0x4F).
#[inline]
pub fn rex_byte(w: bool, r: bool, x: bool, b: bool) -> u8 {
    let mut rex: u8 = 0x40; // REX base
    if w {
        rex |= 0x08;
    } // bit 3: 64-bit operand size
    if r {
        rex |= 0x04;
    } // bit 2: ModR/M reg extension
    if x {
        rex |= 0x02;
    } // bit 1: SIB index extension
    if b {
        rex |= 0x01;
    } // bit 0: ModR/M r/m or SIB base extension
    rex
}

/// Returns `true` if the instruction needs a REX.W prefix for the given operand
/// size in bytes. REX.W is needed for 64-bit operand sizes on x86-64.
#[inline]
pub fn needs_rex_w(size: u8) -> bool {
    size == 8
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // GPR encoding verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_gpr_encodings() {
        assert_eq!(RAX, 0);
        assert_eq!(RCX, 1);
        assert_eq!(RDX, 2);
        assert_eq!(RBX, 3);
        assert_eq!(RSP, 4);
        assert_eq!(RBP, 5);
        assert_eq!(RSI, 6);
        assert_eq!(RDI, 7);
        assert_eq!(R8, 8);
        assert_eq!(R9, 9);
        assert_eq!(R10, 10);
        assert_eq!(R11, 11);
        assert_eq!(R12, 12);
        assert_eq!(R13, 13);
        assert_eq!(R14, 14);
        assert_eq!(R15, 15);
    }

    // -------------------------------------------------------------------------
    // SSE encoding verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_sse_encodings() {
        assert_eq!(XMM0, 16);
        assert_eq!(XMM1, 17);
        assert_eq!(XMM2, 18);
        assert_eq!(XMM3, 19);
        assert_eq!(XMM4, 20);
        assert_eq!(XMM5, 21);
        assert_eq!(XMM6, 22);
        assert_eq!(XMM7, 23);
        assert_eq!(XMM8, 24);
        assert_eq!(XMM9, 25);
        assert_eq!(XMM10, 26);
        assert_eq!(XMM11, 27);
        assert_eq!(XMM12, 28);
        assert_eq!(XMM13, 29);
        assert_eq!(XMM14, 30);
        assert_eq!(XMM15, 31);
    }

    #[test]
    fn test_no_gpr_sse_collision() {
        // GPR IDs 0–15 and SSE IDs 16–31 must not overlap
        for gpr in 0u16..16 {
            for sse in 16u16..32 {
                assert_ne!(gpr, sse);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Count constants
    // -------------------------------------------------------------------------

    #[test]
    fn test_register_counts() {
        assert_eq!(NUM_GPRS, 16);
        assert_eq!(NUM_SSE, 16);
        assert_eq!(TOTAL_REGS, 32);
    }

    // -------------------------------------------------------------------------
    // Register classification
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_gpr() {
        for id in 0u16..16 {
            assert!(is_gpr(id), "is_gpr({}) should be true", id);
        }
        for id in 16u16..64 {
            assert!(!is_gpr(id), "is_gpr({}) should be false", id);
        }
    }

    #[test]
    fn test_is_sse() {
        for id in 0u16..16 {
            assert!(!is_sse(id), "is_sse({}) should be false", id);
        }
        for id in 16u16..32 {
            assert!(is_sse(id), "is_sse({}) should be true", id);
        }
        for id in 32u16..64 {
            assert!(!is_sse(id), "is_sse({}) should be false", id);
        }
    }

    #[test]
    fn test_hw_encoding() {
        // GPRs: hardware encoding = register ID itself
        for id in 0u16..16 {
            assert_eq!(hw_encoding(id), id as u8);
        }
        // SSE: hardware encoding = register ID minus 16
        for id in 16u16..32 {
            assert_eq!(hw_encoding(id), (id - 16) as u8);
        }
    }

    #[test]
    fn test_needs_rex() {
        // GPRs 0–7 do not need REX
        for id in 0u16..8 {
            assert!(!needs_rex(id), "needs_rex({}) should be false", id);
        }
        // GPRs 8–15 need REX
        for id in 8u16..16 {
            assert!(needs_rex(id), "needs_rex({}) should be true", id);
        }
        // SSE 16–23 (XMM0–7) do not need REX
        for id in 16u16..24 {
            assert!(!needs_rex(id), "needs_rex({}) should be false", id);
        }
        // SSE 24–31 (XMM8–15) need REX
        for id in 24u16..32 {
            assert!(needs_rex(id), "needs_rex({}) should be true", id);
        }
    }

    // -------------------------------------------------------------------------
    // Callee/caller saved classification
    // -------------------------------------------------------------------------

    #[test]
    fn test_callee_saved() {
        // Callee-saved: RBX, RBP, R12, R13, R14, R15
        assert!(is_callee_saved(RBX));
        assert!(is_callee_saved(RBP));
        assert!(is_callee_saved(R12));
        assert!(is_callee_saved(R13));
        assert!(is_callee_saved(R14));
        assert!(is_callee_saved(R15));
        // Non callee-saved
        assert!(!is_callee_saved(RAX));
        assert!(!is_callee_saved(RCX));
        assert!(!is_callee_saved(RDX));
        assert!(!is_callee_saved(RSI));
        assert!(!is_callee_saved(RDI));
        assert!(!is_callee_saved(RSP)); // reserved, not callee-saved in the normal sense
        assert!(!is_callee_saved(R8));
        assert!(!is_callee_saved(R9));
        assert!(!is_callee_saved(R10));
        assert!(!is_callee_saved(R11));
    }

    #[test]
    fn test_caller_saved() {
        // Caller-saved: RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11
        assert!(is_caller_saved(RAX));
        assert!(is_caller_saved(RCX));
        assert!(is_caller_saved(RDX));
        assert!(is_caller_saved(RSI));
        assert!(is_caller_saved(RDI));
        assert!(is_caller_saved(R8));
        assert!(is_caller_saved(R9));
        assert!(is_caller_saved(R10));
        assert!(is_caller_saved(R11));
        // Not caller-saved
        assert!(!is_caller_saved(RBX));
        assert!(!is_caller_saved(RBP));
        assert!(!is_caller_saved(RSP));
        assert!(!is_caller_saved(R12));
        assert!(!is_caller_saved(R13));
        assert!(!is_caller_saved(R14));
        assert!(!is_caller_saved(R15));
    }

    // -------------------------------------------------------------------------
    // Register set arrays
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocatable_gprs_count() {
        assert_eq!(ALLOCATABLE_GPRS.len(), 13); // R11 reserved for spill scratch
                                                // Must not contain RSP or RBP
        assert!(!ALLOCATABLE_GPRS.contains(&RSP));
        assert!(!ALLOCATABLE_GPRS.contains(&RBP));
        // Must contain all other GPRs exactly once
        for &r in &ALLOCATABLE_GPRS {
            assert!(is_gpr(r), "Allocatable GPR {} is not a GPR", r);
        }
    }

    #[test]
    fn test_allocatable_sse_count() {
        assert_eq!(ALLOCATABLE_SSE.len(), 15); // XMM15 reserved as float spill scratch
        for &r in &ALLOCATABLE_SSE {
            assert!(is_sse(r), "Allocatable SSE {} is not an SSE register", r);
        }
    }

    #[test]
    fn test_callee_saved_gprs_set() {
        assert_eq!(CALLEE_SAVED_GPRS.len(), 5);
        assert_eq!(CALLEE_SAVED_GPRS, [RBX, R12, R13, R14, R15]);
    }

    #[test]
    fn test_caller_saved_gprs_set() {
        assert_eq!(CALLER_SAVED_GPRS.len(), 8);
        assert_eq!(CALLER_SAVED_GPRS, [RAX, RCX, RDX, RSI, RDI, R8, R9, R10]);
    }

    #[test]
    fn test_reserved_regs() {
        assert_eq!(RESERVED_REGS.len(), 3);
        assert_eq!(RESERVED_REGS, [RSP, RBP, R11]);
    }

    #[test]
    fn test_arg_gprs_order() {
        // System V AMD64: RDI, RSI, RDX, RCX, R8, R9
        assert_eq!(ARG_GPRS, [RDI, RSI, RDX, RCX, R8, R9]);
    }

    #[test]
    fn test_arg_sse_order() {
        assert_eq!(ARG_SSE, [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]);
    }

    #[test]
    fn test_ret_gprs() {
        assert_eq!(RET_GPRS, [RAX, RDX]);
    }

    #[test]
    fn test_ret_sse() {
        assert_eq!(RET_SSE, [XMM0, XMM1]);
    }

    // -------------------------------------------------------------------------
    // Register name mapping
    // -------------------------------------------------------------------------

    #[test]
    fn test_reg_name_64() {
        assert_eq!(reg_name_64(RAX), "rax");
        assert_eq!(reg_name_64(RCX), "rcx");
        assert_eq!(reg_name_64(RDX), "rdx");
        assert_eq!(reg_name_64(RBX), "rbx");
        assert_eq!(reg_name_64(RSP), "rsp");
        assert_eq!(reg_name_64(RBP), "rbp");
        assert_eq!(reg_name_64(RSI), "rsi");
        assert_eq!(reg_name_64(RDI), "rdi");
        assert_eq!(reg_name_64(R8), "r8");
        assert_eq!(reg_name_64(R9), "r9");
        assert_eq!(reg_name_64(R10), "r10");
        assert_eq!(reg_name_64(R11), "r11");
        assert_eq!(reg_name_64(R12), "r12");
        assert_eq!(reg_name_64(R13), "r13");
        assert_eq!(reg_name_64(R14), "r14");
        assert_eq!(reg_name_64(R15), "r15");
    }

    #[test]
    fn test_reg_name_32() {
        assert_eq!(reg_name_32(RAX), "eax");
        assert_eq!(reg_name_32(RCX), "ecx");
        assert_eq!(reg_name_32(RDX), "edx");
        assert_eq!(reg_name_32(RBX), "ebx");
        assert_eq!(reg_name_32(RSP), "esp");
        assert_eq!(reg_name_32(RBP), "ebp");
        assert_eq!(reg_name_32(RSI), "esi");
        assert_eq!(reg_name_32(RDI), "edi");
        assert_eq!(reg_name_32(R8), "r8d");
        assert_eq!(reg_name_32(R9), "r9d");
        assert_eq!(reg_name_32(R10), "r10d");
        assert_eq!(reg_name_32(R11), "r11d");
        assert_eq!(reg_name_32(R12), "r12d");
        assert_eq!(reg_name_32(R13), "r13d");
        assert_eq!(reg_name_32(R14), "r14d");
        assert_eq!(reg_name_32(R15), "r15d");
    }

    #[test]
    fn test_reg_name_16() {
        assert_eq!(reg_name_16(RAX), "ax");
        assert_eq!(reg_name_16(RCX), "cx");
        assert_eq!(reg_name_16(RDX), "dx");
        assert_eq!(reg_name_16(RBX), "bx");
        assert_eq!(reg_name_16(RSP), "sp");
        assert_eq!(reg_name_16(RBP), "bp");
        assert_eq!(reg_name_16(RSI), "si");
        assert_eq!(reg_name_16(RDI), "di");
        assert_eq!(reg_name_16(R8), "r8w");
        assert_eq!(reg_name_16(R9), "r9w");
        assert_eq!(reg_name_16(R10), "r10w");
        assert_eq!(reg_name_16(R11), "r11w");
        assert_eq!(reg_name_16(R12), "r12w");
        assert_eq!(reg_name_16(R13), "r13w");
        assert_eq!(reg_name_16(R14), "r14w");
        assert_eq!(reg_name_16(R15), "r15w");
    }

    #[test]
    fn test_reg_name_8() {
        assert_eq!(reg_name_8(RAX), "al");
        assert_eq!(reg_name_8(RCX), "cl");
        assert_eq!(reg_name_8(RDX), "dl");
        assert_eq!(reg_name_8(RBX), "bl");
        assert_eq!(reg_name_8(RSP), "spl");
        assert_eq!(reg_name_8(RBP), "bpl");
        assert_eq!(reg_name_8(RSI), "sil");
        assert_eq!(reg_name_8(RDI), "dil");
        assert_eq!(reg_name_8(R8), "r8b");
        assert_eq!(reg_name_8(R9), "r9b");
        assert_eq!(reg_name_8(R10), "r10b");
        assert_eq!(reg_name_8(R11), "r11b");
        assert_eq!(reg_name_8(R12), "r12b");
        assert_eq!(reg_name_8(R13), "r13b");
        assert_eq!(reg_name_8(R14), "r14b");
        assert_eq!(reg_name_8(R15), "r15b");
    }

    #[test]
    fn test_sse_reg_names() {
        assert_eq!(sse_reg_name(XMM0), "xmm0");
        assert_eq!(sse_reg_name(XMM1), "xmm1");
        assert_eq!(sse_reg_name(XMM2), "xmm2");
        assert_eq!(sse_reg_name(XMM3), "xmm3");
        assert_eq!(sse_reg_name(XMM4), "xmm4");
        assert_eq!(sse_reg_name(XMM5), "xmm5");
        assert_eq!(sse_reg_name(XMM6), "xmm6");
        assert_eq!(sse_reg_name(XMM7), "xmm7");
        assert_eq!(sse_reg_name(XMM8), "xmm8");
        assert_eq!(sse_reg_name(XMM9), "xmm9");
        assert_eq!(sse_reg_name(XMM10), "xmm10");
        assert_eq!(sse_reg_name(XMM11), "xmm11");
        assert_eq!(sse_reg_name(XMM12), "xmm12");
        assert_eq!(sse_reg_name(XMM13), "xmm13");
        assert_eq!(sse_reg_name(XMM14), "xmm14");
        assert_eq!(sse_reg_name(XMM15), "xmm15");
    }

    #[test]
    fn test_reg_name_dispatch() {
        // GPR sizes dispatch correctly
        assert_eq!(reg_name(RAX, 8), "rax");
        assert_eq!(reg_name(RAX, 4), "eax");
        assert_eq!(reg_name(RAX, 2), "ax");
        assert_eq!(reg_name(RAX, 1), "al");
        assert_eq!(reg_name(R8, 8), "r8");
        assert_eq!(reg_name(R8, 4), "r8d");
        assert_eq!(reg_name(R8, 2), "r8w");
        assert_eq!(reg_name(R8, 1), "r8b");
        // Default to 64-bit for unknown sizes
        assert_eq!(reg_name(RAX, 16), "rax");
        assert_eq!(reg_name(RAX, 0), "rax");
        // SSE ignores size parameter
        assert_eq!(reg_name(XMM0, 8), "xmm0");
        assert_eq!(reg_name(XMM0, 4), "xmm0");
        assert_eq!(reg_name(XMM0, 16), "xmm0");
        assert_eq!(reg_name(XMM15, 8), "xmm15");
    }

    // -------------------------------------------------------------------------
    // REX prefix construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_rex_byte_base() {
        // All flags false → 0x40 (bare REX, no extensions)
        assert_eq!(rex_byte(false, false, false, false), 0x40);
    }

    #[test]
    fn test_rex_byte_w() {
        // REX.W only → 0x48 (64-bit operand size)
        assert_eq!(rex_byte(true, false, false, false), 0x48);
    }

    #[test]
    fn test_rex_byte_r() {
        // REX.R only → 0x44 (extend ModR/M reg field)
        assert_eq!(rex_byte(false, true, false, false), 0x44);
    }

    #[test]
    fn test_rex_byte_x() {
        // REX.X only → 0x42 (extend SIB index field)
        assert_eq!(rex_byte(false, false, true, false), 0x42);
    }

    #[test]
    fn test_rex_byte_b() {
        // REX.B only → 0x41 (extend ModR/M r/m or SIB base)
        assert_eq!(rex_byte(false, false, false, true), 0x41);
    }

    #[test]
    fn test_rex_byte_all() {
        // All flags set → 0x4F
        assert_eq!(rex_byte(true, true, true, true), 0x4F);
    }

    #[test]
    fn test_rex_byte_wr() {
        // REX.W + REX.R → 0x4C
        assert_eq!(rex_byte(true, true, false, false), 0x4C);
    }

    #[test]
    fn test_rex_byte_wb() {
        // REX.W + REX.B → 0x49 (e.g., 64-bit mov to R8)
        assert_eq!(rex_byte(true, false, false, true), 0x49);
    }

    // -------------------------------------------------------------------------
    // REX.W sizing
    // -------------------------------------------------------------------------

    #[test]
    fn test_needs_rex_w() {
        assert!(needs_rex_w(8));
        assert!(!needs_rex_w(4));
        assert!(!needs_rex_w(2));
        assert!(!needs_rex_w(1));
        assert!(!needs_rex_w(0));
        assert!(!needs_rex_w(16));
    }

    // -------------------------------------------------------------------------
    // Operand size prefix
    // -------------------------------------------------------------------------

    #[test]
    fn test_operand_size_prefix() {
        assert_eq!(OPERAND_SIZE_PREFIX, 0x66);
    }

    // -------------------------------------------------------------------------
    // Cross-validation: sets are consistent
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocatable_plus_reserved_equals_all_gprs() {
        let mut all: Vec<u16> = ALLOCATABLE_GPRS.to_vec();
        all.extend_from_slice(&RESERVED_REGS);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), NUM_GPRS);
        for id in 0u16..16 {
            assert!(
                all.contains(&id),
                "Missing GPR {} from union of allocatable + reserved",
                id
            );
        }
    }

    #[test]
    fn test_caller_callee_coverage() {
        // Caller-saved (8) + callee-saved (5) + reserved (3) = 16 GPRs
        // Note: RBP is callee-saved by ABI but in RESERVED_REGS (handled separately)
        // R11 is in RESERVED_REGS as spill scratch.
        let mut all: Vec<u16> = CALLER_SAVED_GPRS.to_vec();
        all.extend_from_slice(&CALLEE_SAVED_GPRS);
        all.extend_from_slice(&RESERVED_REGS);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), NUM_GPRS);
    }
}
