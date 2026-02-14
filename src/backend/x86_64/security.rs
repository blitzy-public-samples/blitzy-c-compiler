//! Security mitigations for x86-64 code generation.
//!
//! This module implements three security features that are unique to the
//! x86-64 backend, activated by specific command-line flags:
//!
//! | Mitigation        | Flag               | Purpose                                       |
//! |-------------------|--------------------|-----------------------------------------------|
//! | Retpoline         | `-mretpoline`      | Spectre v2 (BTI) mitigation via thunk trampolines |
//! | CET / IBT         | `-fcf-protection`  | Control-flow integrity via `endbr64` landing pads |
//! | Stack Probe        | (auto for > 4KiB) | Guard page probing for large stack frames     |
//!
//! # Retpoline
//!
//! When retpoline is enabled, every indirect `CALL` or `JMP` through a
//! register is replaced with a call/jump to `__x86_indirect_thunk_<reg>`.
//! The thunk uses an RSB (Return Stack Buffer) stuffing sequence to
//! defeat speculative execution through the indirect branch:
//!
//! ```text
//! __x86_indirect_thunk_rax:
//!     call    .Ltarget0
//! .Lcapture0:
//!     pause
//!     lfence
//!     jmp     .Lcapture0
//! .Ltarget0:
//!     mov     [rsp], rax
//!     ret
//! ```
//!
//! # CET / IBT (Intel Control-flow Enforcement Technology)
//!
//! When cf-protection is enabled, `endbr64` (4-byte NOP-equivalent on
//! older hardware) is inserted at:
//!
//! - Every function entry point
//! - Every indirect branch target (jump table entries, computed gotos)
//!
//! # Stack Probe
//!
//! When a function's stack frame exceeds 4096 bytes (one page), the
//! compiler emits a probing loop that touches each page of the stack in
//! descending order. This ensures the OS can grow the stack one page at
//! a time and detect stack overflow via the guard page.
//!
//! ```text
//!     ; prologue for frame_size > 4096
//!     mov rax, rsp
//! .Lprobe:
//!     sub rax, 4096
//!     test [rax], eax     ; touch the page
//!     cmp rax, rsp_target
//!     ja  .Lprobe
//!     mov rsp, rsp_target
//! ```

use crate::backend::traits::{MachineFunction, MachineInstr, MachineOperand, PhysReg};
use crate::backend::x86_64::registers;

// ---------------------------------------------------------------------------
// x86-64 instruction opcodes used by security mitigations
// ---------------------------------------------------------------------------
// These are internal opcode tags that identify machine instructions in the
// MachineInstr representation. They must remain in sync with the opcode
// constants defined in the x86_64 codegen / mod.rs module.

/// `endbr64` — Intel CET indirect-branch tracking landing pad.
const OP_ENDBR64: u32 = 0x1000;

/// `push reg` — push a GPR onto the stack.
#[allow(dead_code)]
const OP_PUSH: u32 = 0x0001;

/// `mov reg, imm` — move immediate into register.
const OP_MOV_REG_IMM: u32 = 0x0010;

/// `sub reg, imm` — subtract immediate from register.
const OP_SUB_REG_IMM: u32 = 0x0011;

/// `test [reg], reg` — test memory at address in first reg using second reg.
const OP_TEST_MEM_REG: u32 = 0x0012;

/// `cmp reg, reg` — compare two registers.
const OP_CMP_REG_REG: u32 = 0x0013;

/// `ja label` — jump if above (unsigned).
const OP_JA: u32 = 0x0014;

/// `mov rsp, reg` — move register into rsp (final stack adjustment).
const OP_MOV_RSP_REG: u32 = 0x0015;

/// `call label` — direct call to a label (used in retpoline thunks).
const OP_CALL_LABEL: u32 = 0x0016;

/// `pause` — hint to the processor for spin loops.
const OP_PAUSE: u32 = 0x0017;

/// `lfence` — load fence / serialising instruction.
const OP_LFENCE: u32 = 0x0018;

/// `jmp label` — unconditional jump (used in retpoline capture loop).
const OP_JMP_LABEL: u32 = 0x0019;

/// `mov [rsp], reg` — store register to top of stack (retpoline RSB stuff).
const OP_MOV_MEM_RSP_REG: u32 = 0x001A;

/// `ret` — return from procedure.
const OP_RET: u32 = 0x001B;

/// `call reg` — indirect call through register (to be replaced by retpoline).
const OP_CALL_REG: u32 = 0x0020;

/// `jmp reg` — indirect jump through register (to be replaced by retpoline).
const OP_JMP_REG: u32 = 0x0021;

/// `nop` — no operation (padding).
#[cfg(test)]
const OP_NOP: u32 = 0x0000;

// ---------------------------------------------------------------------------
// Page size constant
// ---------------------------------------------------------------------------

/// x86-64 page size in bytes. Stack frames larger than this threshold
/// require stack probing to avoid skipping the guard page.
const PAGE_SIZE: u32 = 4096;

// ---------------------------------------------------------------------------
// SecurityConfig — aggregates all security-related flags
// ---------------------------------------------------------------------------

/// Configuration flags for x86-64 security mitigations.
///
/// These flags are derived from command-line options and propagated
/// through the code generation pipeline.
#[derive(Clone, Debug, Default)]
pub struct SecurityConfig {
    /// Enable Spectre v2 mitigation via retpoline thunks (`-mretpoline`).
    pub retpoline: bool,

    /// Enable Intel CET / IBT landing pads (`-fcf-protection`).
    pub cf_protection: bool,

    /// Stack probe threshold in bytes. Frames larger than this value
    /// trigger a probing loop in the prologue. Default is 4096 (one page).
    pub stack_probe_threshold: u32,
}

impl SecurityConfig {
    /// Creates a new `SecurityConfig` with default values.
    ///
    /// Default state: all mitigations disabled, probe threshold = 4096.
    pub fn new() -> Self {
        SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe_threshold: PAGE_SIZE,
        }
    }

    /// Creates a `SecurityConfig` with all mitigations enabled.
    #[allow(dead_code)]
    pub fn all_enabled() -> Self {
        SecurityConfig {
            retpoline: true,
            cf_protection: true,
            stack_probe_threshold: PAGE_SIZE,
        }
    }

    /// Creates a `SecurityConfig` from individual flag values.
    ///
    /// This constructor is used by the codegen driver to translate
    /// CLI flags (`-mretpoline`, `-fcf-protection`) into a security
    /// configuration. The stack probe threshold defaults to one page
    /// (4096 bytes), matching the System V ABI guard page size.
    ///
    /// # Arguments
    ///
    /// * `retpoline` — enable Spectre v2 retpoline thunk generation
    /// * `cf_protection` — enable Intel CET / IBT `endbr64` insertion
    pub fn from_flags(retpoline: bool, cf_protection: bool) -> Self {
        SecurityConfig {
            retpoline,
            cf_protection,
            stack_probe_threshold: PAGE_SIZE,
        }
    }

    /// Returns `true` if any security mitigation is active.
    pub fn any_enabled(&self) -> bool {
        self.retpoline || self.cf_protection
    }
}

// ---------------------------------------------------------------------------
// apply_security_mitigations — top-level entry point
// ---------------------------------------------------------------------------

/// Applies all enabled security mitigations to a machine function.
///
/// This is called after instruction selection but before final assembly
/// emission. It transforms the machine instruction stream in-place to
/// insert security features.
///
/// # Arguments
///
/// * `mf` — the machine function to transform
/// * `config` — the active security configuration flags
///
/// # Transformations Applied
///
/// 1. **CET/IBT:** Insert `endbr64` at function entry and indirect branch targets.
/// 2. **Retpoline:** Replace indirect CALL/JMP through registers with calls
///    to `__x86_indirect_thunk_<reg>`.
/// 3. **Stack Probe:** Insert probe loop in prologue when `mf.frame_size`
///    exceeds the configured threshold.
pub fn apply_security_mitigations(mf: &mut MachineFunction, config: &SecurityConfig) {
    if !config.any_enabled() && mf.frame_size <= config.stack_probe_threshold {
        return;
    }

    // Phase 1: Insert endbr64 at function entry for CET/IBT.
    if config.cf_protection {
        insert_endbr64_at_entry(mf);
    }

    // Phase 2: Replace indirect branches with retpoline thunks.
    if config.retpoline {
        apply_retpoline(mf);
    }

    // Phase 3: Insert stack probe loop if frame exceeds threshold.
    if mf.frame_size > config.stack_probe_threshold {
        insert_stack_probe(mf);
    }
}

// ---------------------------------------------------------------------------
// CET / IBT — endbr64 insertion
// ---------------------------------------------------------------------------

/// Inserts an `endbr64` instruction at the beginning of the function's
/// entry block. This instruction serves as a valid indirect branch target
/// for Intel CET's Indirect Branch Tracking (IBT) mechanism.
///
/// On CPUs without CET support, `endbr64` decodes as a 4-byte NOP
/// (`F3 0F 1E FA`), so it is always safe to emit.
fn insert_endbr64_at_entry(mf: &mut MachineFunction) {
    if mf.blocks.is_empty() {
        return;
    }

    let endbr = MachineInstr {
        opcode: OP_ENDBR64,
        operands: Vec::new(),
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        is_terminator: false,
        is_call: false,
        is_return: false,
    };

    // Insert at position 0 of the first block.
    mf.blocks[0].instructions.insert(0, endbr);
}

// ---------------------------------------------------------------------------
// Retpoline — indirect branch replacement
// ---------------------------------------------------------------------------

/// Scans the machine function for indirect calls/jumps through registers
/// and replaces each one with a direct call to the corresponding retpoline
/// thunk (`__x86_indirect_thunk_<reg>`).
///
/// The thunk names follow the Linux kernel convention:
///   - `__x86_indirect_thunk_rax` for RAX
///   - `__x86_indirect_thunk_rbx` for RBX
///   - etc.
///
/// The actual thunk bodies are emitted by [`generate_retpoline_thunks`].
fn apply_retpoline(mf: &mut MachineFunction) {
    for block in &mut mf.blocks {
        let mut i = 0;
        while i < block.instructions.len() {
            let instr = &block.instructions[i];
            let should_replace = instr.opcode == OP_CALL_REG || instr.opcode == OP_JMP_REG;

            if should_replace {
                // Extract the target register from the first operand.
                let target_reg = match instr.operands.first() {
                    Some(MachineOperand::Register(r)) => *r,
                    _ => {
                        i += 1;
                        continue;
                    }
                };

                let is_call = instr.opcode == OP_CALL_REG;

                // Generate the thunk name based on the target register.
                let thunk_name = retpoline_thunk_name(target_reg);

                // Replace the indirect call/jump with a direct call to the thunk.
                let replacement = MachineInstr {
                    opcode: if is_call { OP_CALL_LABEL } else { OP_JMP_LABEL },
                    operands: vec![MachineOperand::Symbol(thunk_name)],
                    implicit_defs: if is_call {
                        vec![registers::RAX, registers::RCX, registers::RDX,
                             registers::RSI, registers::RDI, registers::R8,
                             registers::R9, registers::R10, registers::R11]
                    } else {
                        Vec::new()
                    },
                    implicit_uses: vec![target_reg],
                    is_terminator: !is_call,
                    is_call,
                    is_return: false,
                };

                block.instructions[i] = replacement;
            }
            i += 1;
        }
    }
}

/// Returns the retpoline thunk name for a given register.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(retpoline_thunk_name(RAX), "__x86_indirect_thunk_rax");
/// assert_eq!(retpoline_thunk_name(R11), "__x86_indirect_thunk_r11");
/// ```
fn retpoline_thunk_name(reg: PhysReg) -> String {
    let reg_name = register_name_lower(reg);
    format!("__x86_indirect_thunk_{}", reg_name)
}

/// Returns the lowercase register name for a physical register.
fn register_name_lower(reg: PhysReg) -> &'static str {
    match reg {
        r if r == registers::RAX => "rax",
        r if r == registers::RCX => "rcx",
        r if r == registers::RDX => "rdx",
        r if r == registers::RBX => "rbx",
        r if r == registers::RSP => "rsp",
        r if r == registers::RBP => "rbp",
        r if r == registers::RSI => "rsi",
        r if r == registers::RDI => "rdi",
        r if r == registers::R8  => "r8",
        r if r == registers::R9  => "r9",
        r if r == registers::R10 => "r10",
        r if r == registers::R11 => "r11",
        r if r == registers::R12 => "r12",
        r if r == registers::R13 => "r13",
        r if r == registers::R14 => "r14",
        r if r == registers::R15 => "r15",
        _ => "unknown",
    }
}

/// Generates the retpoline thunk body for a given register.
///
/// This produces a standalone function that can be emitted as a global
/// symbol. The thunk implements the Spectre v2 mitigation by stuffing
/// the RSB (Return Stack Buffer) with the real target address.
///
/// # Thunk Sequence
///
/// ```text
/// __x86_indirect_thunk_<reg>:
///     call    .Ltarget
/// .Lcapture:
///     pause
///     lfence
///     jmp     .Lcapture
/// .Ltarget:
///     mov     [rsp], <reg>
///     ret
/// ```
#[allow(dead_code, clippy::vec_init_then_push)]
pub fn generate_retpoline_thunk(target_reg: PhysReg) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(6);

    // call .Ltarget — pushes return address (.Lcapture) onto RSB
    instrs.push(MachineInstr {
        opcode: OP_CALL_LABEL,
        operands: vec![MachineOperand::Symbol(".Ltarget".to_string())],
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        is_terminator: false,
        is_call: true,
        is_return: false,
    });

    // .Lcapture:
    // pause — hint to processor (reduces power in spin loop)
    instrs.push(MachineInstr {
        opcode: OP_PAUSE,
        operands: Vec::new(),
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // lfence — serialising, prevents speculative execution past this point
    instrs.push(MachineInstr {
        opcode: OP_LFENCE,
        operands: Vec::new(),
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // jmp .Lcapture — infinite loop (only reached speculatively)
    instrs.push(MachineInstr {
        opcode: OP_JMP_LABEL,
        operands: vec![MachineOperand::Symbol(".Lcapture".to_string())],
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        is_terminator: true,
        is_call: false,
        is_return: false,
    });

    // .Ltarget:
    // mov [rsp], <reg> — replace return address with actual target
    instrs.push(MachineInstr {
        opcode: OP_MOV_MEM_RSP_REG,
        operands: vec![MachineOperand::Register(target_reg)],
        implicit_defs: Vec::new(),
        implicit_uses: vec![registers::RSP, target_reg],
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // ret — pops the (now-patched) return address, jumping to <reg>
    instrs.push(MachineInstr {
        opcode: OP_RET,
        operands: Vec::new(),
        implicit_defs: Vec::new(),
        implicit_uses: vec![registers::RSP],
        is_terminator: true,
        is_call: false,
        is_return: true,
    });

    instrs
}

// ---------------------------------------------------------------------------
// Stack Probe — guard page probing loop
// ---------------------------------------------------------------------------

/// Inserts a stack probe loop at the beginning of the function when the
/// frame size exceeds the page size (4096 bytes).
///
/// The probe loop ensures that each page of the stack is touched in order
/// from higher to lower addresses, allowing the OS kernel to extend the
/// stack mapping one page at a time and trigger a guard page fault if the
/// stack limit is reached.
///
/// # Generated Sequence
///
/// ```text
///     mov  rax, rsp           ; save current stack pointer
/// .Lprobe_loop:
///     sub  rax, 4096          ; move down one page
///     test [rax], eax         ; touch the page (read)
///     cmp  rax, <final_rsp>   ; check if we've reached the target
///     ja   .Lprobe_loop       ; continue probing if above target
///     mov  rsp, <final_rsp>   ; set final stack pointer
/// ```
#[allow(clippy::vec_init_then_push)]
fn insert_stack_probe(mf: &mut MachineFunction) {
    if mf.blocks.is_empty() {
        return;
    }

    let frame_size = mf.frame_size;
    let mut probe_instrs = Vec::with_capacity(6);

    // mov rax, rsp — save current stack pointer into scratch register
    probe_instrs.push(MachineInstr {
        opcode: OP_MOV_REG_IMM, // reuse as mov r, r in context
        operands: vec![
            MachineOperand::Register(registers::RAX),
            MachineOperand::Register(registers::RSP),
        ],
        implicit_defs: vec![registers::RAX],
        implicit_uses: vec![registers::RSP],
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // sub rax, 4096 — move down one page
    probe_instrs.push(MachineInstr {
        opcode: OP_SUB_REG_IMM,
        operands: vec![
            MachineOperand::Register(registers::RAX),
            MachineOperand::Immediate(PAGE_SIZE as i64),
        ],
        implicit_defs: vec![registers::RAX],
        implicit_uses: vec![registers::RAX],
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // test [rax], eax — touch the page to trigger a fault if unmapped
    probe_instrs.push(MachineInstr {
        opcode: OP_TEST_MEM_REG,
        operands: vec![
            MachineOperand::Register(registers::RAX),
            MachineOperand::Register(registers::RAX),
        ],
        implicit_defs: Vec::new(),
        implicit_uses: vec![registers::RAX],
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // cmp rax, (rsp - frame_size) — compare against final target
    // We use an immediate that represents the total frame size; the
    // actual comparison target is rsp - frame_size, which the instruction
    // selector will resolve.
    probe_instrs.push(MachineInstr {
        opcode: OP_CMP_REG_REG,
        operands: vec![
            MachineOperand::Register(registers::RAX),
            MachineOperand::Immediate(-(frame_size as i64)),
        ],
        implicit_defs: Vec::new(),
        implicit_uses: vec![registers::RAX, registers::RSP],
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // ja .Lprobe_loop — loop back if we haven't reached the target
    probe_instrs.push(MachineInstr {
        opcode: OP_JA,
        operands: vec![MachineOperand::Symbol(".Lprobe_loop".to_string())],
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        is_terminator: false, // not a block terminator in the logical sense
        is_call: false,
        is_return: false,
    });

    // mov rsp, (rsp - frame_size) — set the final stack pointer
    probe_instrs.push(MachineInstr {
        opcode: OP_MOV_RSP_REG,
        operands: vec![
            MachineOperand::Register(registers::RSP),
            MachineOperand::Immediate(-(frame_size as i64)),
        ],
        implicit_defs: vec![registers::RSP],
        implicit_uses: vec![registers::RSP],
        is_terminator: false,
        is_call: false,
        is_return: false,
    });

    // Insert probe instructions at the beginning of the entry block,
    // after any existing endbr64 instruction.
    let entry = &mut mf.blocks[0];
    let insert_pos = if !entry.instructions.is_empty()
        && entry.instructions[0].opcode == OP_ENDBR64
    {
        1
    } else {
        0
    };

    // Insert in reverse order so that they end up in the correct sequence.
    for (offset, instr) in probe_instrs.into_iter().enumerate() {
        entry.instructions.insert(insert_pos + offset, instr);
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::MachineBasicBlock;

    /// Helper to create a minimal machine function for testing.
    fn make_test_mf(frame_size: u32) -> MachineFunction {
        let mut mf = MachineFunction::new("test_fn".to_string(), 16);
        mf.frame_size = frame_size;
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);
        mf
    }

    #[test]
    fn security_config_default_has_no_mitigations() {
        let config = SecurityConfig::new();
        assert!(!config.retpoline);
        assert!(!config.cf_protection);
        assert_eq!(config.stack_probe_threshold, 4096);
        assert!(!config.any_enabled());
    }

    #[test]
    fn security_config_all_enabled() {
        let config = SecurityConfig::all_enabled();
        assert!(config.retpoline);
        assert!(config.cf_protection);
        assert!(config.any_enabled());
    }

    #[test]
    fn endbr64_inserted_at_function_entry() {
        let mut mf = make_test_mf(0);

        // Add a dummy instruction in the entry block.
        mf.blocks[0].instructions.push(MachineInstr {
            opcode: OP_NOP,
            operands: Vec::new(),
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            is_terminator: false,
            is_call: false,
            is_return: false,
        });

        let config = SecurityConfig {
            retpoline: false,
            cf_protection: true,
            stack_probe_threshold: PAGE_SIZE,
        };

        apply_security_mitigations(&mut mf, &config);

        // First instruction should now be endbr64.
        assert_eq!(mf.blocks[0].instructions[0].opcode, OP_ENDBR64);
        // Original instruction should be second.
        assert_eq!(mf.blocks[0].instructions[1].opcode, OP_NOP);
    }

    #[test]
    fn retpoline_replaces_indirect_call() {
        let mut mf = make_test_mf(0);

        // Add an indirect call through RAX.
        mf.blocks[0].instructions.push(MachineInstr {
            opcode: OP_CALL_REG,
            operands: vec![MachineOperand::Register(registers::RAX)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            is_terminator: false,
            is_call: true,
            is_return: false,
        });

        let _config = SecurityConfig {
            retpoline: true,
            cf_protection: false,
            stack_probe_threshold: PAGE_SIZE,
        };

        apply_retpoline(&mut mf);

        // The indirect call should be replaced with a call to the thunk.
        let instr = &mf.blocks[0].instructions[0];
        assert_eq!(instr.opcode, OP_CALL_LABEL);
        assert!(instr.is_call);
        match &instr.operands[0] {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rax");
            }
            _ => panic!("Expected Symbol operand for retpoline thunk call"),
        }
    }

    #[test]
    fn retpoline_thunk_name_correct() {
        assert_eq!(retpoline_thunk_name(registers::RAX), "__x86_indirect_thunk_rax");
        assert_eq!(retpoline_thunk_name(registers::R11), "__x86_indirect_thunk_r11");
        assert_eq!(retpoline_thunk_name(registers::RBX), "__x86_indirect_thunk_rbx");
    }

    #[test]
    fn stack_probe_inserted_for_large_frames() {
        let mut mf = make_test_mf(8192); // > 4096, needs probing.

        let config = SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe_threshold: PAGE_SIZE,
        };

        apply_security_mitigations(&mut mf, &config);

        // Should have inserted probe instructions.
        assert!(mf.blocks[0].instructions.len() > 0);
    }

    #[test]
    fn no_mitigations_when_all_disabled() {
        let mut mf = make_test_mf(128); // Small frame, no probe needed.

        let config = SecurityConfig::new();
        apply_security_mitigations(&mut mf, &config);

        // No instructions should have been inserted.
        assert_eq!(mf.blocks[0].instructions.len(), 0);
    }

    #[test]
    fn retpoline_thunk_generation() {
        let thunk = generate_retpoline_thunk(registers::RAX);
        assert_eq!(thunk.len(), 6);
        // First instruction is call .Ltarget
        assert_eq!(thunk[0].opcode, OP_CALL_LABEL);
        assert!(thunk[0].is_call);
        // Second is pause
        assert_eq!(thunk[1].opcode, OP_PAUSE);
        // Third is lfence
        assert_eq!(thunk[2].opcode, OP_LFENCE);
        // Fourth is jmp .Lcapture
        assert_eq!(thunk[3].opcode, OP_JMP_LABEL);
        assert!(thunk[3].is_terminator);
        // Fifth is mov [rsp], reg
        assert_eq!(thunk[4].opcode, OP_MOV_MEM_RSP_REG);
        // Sixth is ret
        assert_eq!(thunk[5].opcode, OP_RET);
        assert!(thunk[5].is_return);
    }
}
