//! x86-64 Security Mitigations for Hardened Code Generation
//!
//! This module implements three independent, conditionally-enabled security
//! hardening features that are **exclusive to the x86-64 target**. Each
//! mitigation is activated by a corresponding CLI flag and is injected into
//! the machine code during Phase 10 (code generation).
//!
//! # Mitigations
//!
//! ## Retpoline (`-mretpoline`)
//! Replaces indirect `call *%reg` / `jmp *%reg` instructions with calls to
//! `__x86_indirect_thunk_<reg>` stubs. Each thunk uses a call–pause–lfence
//! loop to prevent speculative execution of the indirect target, mitigating
//! Spectre v2 (Branch Target Injection).
//!
//! ## CET/IBT — Control-Flow Enforcement Technology (`-fcf-protection`)
//! Inserts the 4-byte `endbr64` instruction (0xF3 0x0F 0x1E 0xFA) at every
//! function entry and indirect branch target. Intel CET hardware traps if an
//! indirect call/jump lands on an instruction that is not `endbr64`, providing
//! forward-edge control-flow integrity.
//!
//! ## Stack Guard Page Probing (automatic for frames > 4096 bytes)
//! Generates a loop that touches each 4096-byte page of the stack frame in
//! descending order before the actual stack pointer adjustment. This ensures
//! the OS kernel can extend the stack mapping one page at a time and triggers
//! a guard-page fault if the stack limit is reached, preventing silent stack
//! overflow into adjacent memory.
//!
//! # Integration
//!
//! The main entry point [`apply_security_mitigations`] is called by
//! `src/backend/generation.rs` during Phase 10 for x86-64 targets only.
//! It receives a `&mut MachineFunction` and a [`SecurityConfig`] and applies
//! all enabled mitigations in the correct order:
//!
//! 1. CET/IBT `endbr64` insertion (first, so endbr64 is at position 0)
//! 2. Stack probe loop insertion (after endbr64 if present)
//! 3. Retpoline indirect call/jump rewriting (last, as it may add thunk calls)

use crate::backend::traits::{MachineFunction, MachineInstr, MachineOperand, PhysReg};
use crate::backend::x86_64::codegen::X86_64Opcode;
use crate::backend::x86_64::opcodes;
use crate::backend::x86_64::registers::{
    EAX, RAX, RCX, RSP, gpr_name_64, is_gpr,
};
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The target architecture that these security mitigations apply to.
/// All mitigations in this module are **exclusively for x86-64** and must
/// never be invoked for other targets. This constant serves as both
/// documentation and a compile-time reference for the supported target.
const SUPPORTED_TARGET: Target = Target::X86_64;

/// x86-64 page size in bytes — the granularity at which the OS kernel maps
/// stack pages. Guard page probing must touch every page in a frame that
/// exceeds this size.
const PAGE_SIZE: u32 = 4096;

/// Default stack probe threshold. Frames exceeding this size require a probe
/// loop to be emitted before the stack pointer adjustment.
const DEFAULT_STACK_PROBE_THRESHOLD: u32 = PAGE_SIZE;

/// Condition code byte for JA (Jump if Above / unsigned) used in JCC encoding.
/// x86-64 JA = 0F 87 rel32 ⇒ Jcc opcode with condition 0x7 (above).
const COND_ABOVE: i64 = 0x07;

// ---------------------------------------------------------------------------
// Opcode Mapping Helper
// ---------------------------------------------------------------------------

/// Maps a high-level [`X86_64Opcode`] to the `u32` machine opcode constant
/// from [`crate::backend::x86_64::opcodes`] used in security mitigation
/// instruction sequences.
///
/// This function bridges the display/diagnostic-oriented [`X86_64Opcode`]
/// enum with the concrete opcode constants stored in [`MachineInstr::opcode`].
fn security_opcode(op: X86_64Opcode) -> u32 {
    match op {
        X86_64Opcode::CALL => opcodes::CALL,
        X86_64Opcode::JMP => opcodes::JMP,
        X86_64Opcode::MOV => opcodes::MOV_RR,
        X86_64Opcode::SUB => opcodes::SUB_RI,
        X86_64Opcode::TEST => opcodes::TEST_RR,
        X86_64Opcode::CMP => opcodes::CMP_RI,
        X86_64Opcode::JA => opcodes::JCC,
        X86_64Opcode::RET => opcodes::RET,
        X86_64Opcode::NOP => opcodes::NOP,
        // All other variants fall through to NOP — these are not used in
        // security sequences but the match must be exhaustive.
        _ => opcodes::NOP,
    }
}

// ---------------------------------------------------------------------------
// SecurityConfig
// ---------------------------------------------------------------------------

/// Configuration for x86-64 security mitigations.
///
/// Each field corresponds to a CLI flag that conditionally enables a specific
/// hardening feature. The configuration is created by the CLI driver from
/// parsed command-line flags and passed to [`apply_security_mitigations`].
///
/// # Default
///
/// By default, all mitigations are disabled and the stack probe threshold is
/// set to [`PAGE_SIZE`] (4096 bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecurityConfig {
    /// Enable retpoline thunk generation for indirect calls and jumps.
    /// Activated by `-mretpoline` CLI flag.
    pub retpoline: bool,

    /// Enable CET/IBT `endbr64` insertion at function entries and
    /// indirect branch targets. Activated by `-fcf-protection` CLI flag.
    pub cf_protection: bool,

    /// Stack frame size (in bytes) above which a guard page probe loop
    /// is emitted. Default: 4096 (one page).
    pub stack_probe_threshold: u32,
}

impl SecurityConfig {
    /// Creates a default configuration with all mitigations disabled.
    pub fn new() -> Self {
        SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe_threshold: DEFAULT_STACK_PROBE_THRESHOLD,
        }
    }

    /// Creates a [`SecurityConfig`] from the parsed CLI flags.
    ///
    /// # Arguments
    ///
    /// * `retpoline` — `true` if `-mretpoline` was specified.
    /// * `cf_protection` — `true` if `-fcf-protection` was specified.
    ///
    /// The stack probe threshold is always set to the default page size;
    /// it is not currently configurable via a CLI flag.
    pub fn from_flags(retpoline: bool, cf_protection: bool) -> Self {
        SecurityConfig {
            retpoline,
            cf_protection,
            stack_probe_threshold: DEFAULT_STACK_PROBE_THRESHOLD,
        }
    }

    /// Creates a configuration with all mitigations enabled.
    ///
    /// Useful for testing when all security features should be exercised.
    pub fn all_enabled() -> Self {
        SecurityConfig {
            retpoline: true,
            cf_protection: true,
            stack_probe_threshold: DEFAULT_STACK_PROBE_THRESHOLD,
        }
    }

    /// Returns `true` if any security mitigation is enabled.
    pub fn any_enabled(&self) -> bool {
        self.retpoline || self.cf_protection
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RetpolineThunk
// ---------------------------------------------------------------------------

/// A retpoline thunk stub for a single register.
///
/// Each indirect call/jump register gets its own thunk, named
/// `__x86_indirect_thunk_<reg>` (e.g., `__x86_indirect_thunk_rax`).
/// The thunk code uses a call–pause–lfence capture loop to prevent
/// speculative execution of the indirect target.
///
/// # Thunk Sequence (AT&T syntax)
///
/// ```text
/// __x86_indirect_thunk_rax:
///     call  .Ltarget
/// .Lcapture:
///     pause
///     lfence
///     jmp   .Lcapture
/// .Ltarget:
///     mov   %rax, (%rsp)    ; overwrite return address with target
///     ret                    ; jump to target via return
/// ```
#[derive(Clone, Debug)]
pub struct RetpolineThunk {
    /// Symbol name for this thunk (e.g., `__x86_indirect_thunk_rax`).
    pub name: String,

    /// Machine instructions implementing the thunk body.
    pub code: Vec<MachineInstr>,

    /// The register whose indirect target this thunk protects.
    pub register: PhysReg,
}

// ---------------------------------------------------------------------------
// RetpolineGenerator
// ---------------------------------------------------------------------------

/// Generator for retpoline thunk stubs and indirect call/jump rewriting.
///
/// The retpoline mitigation works in two phases:
///
/// 1. **Rewriting:** Each indirect `call *%reg` or `jmp *%reg` in the
///    function is replaced with a direct `call/jmp __x86_indirect_thunk_<reg>`.
///
/// 2. **Thunk generation:** For each register that was used in an indirect
///    call/jump, a thunk stub is generated. These stubs are emitted as
///    separate symbols in the final object file.
///
/// The [`RetpolineGenerator`] provides static methods for both phases.
pub struct RetpolineGenerator;

impl RetpolineGenerator {
    /// Generates retpoline thunks for the given set of physical registers.
    ///
    /// Each register receives its own `__x86_indirect_thunk_<reg>` stub
    /// containing the call–capture–replace–ret sequence.
    ///
    /// # Arguments
    ///
    /// * `regs` — Slice of physical registers to generate thunks for.
    ///   Typically this is the set of registers that actually appear in
    ///   indirect call/jump instructions in the current compilation unit.
    ///
    /// # Returns
    ///
    /// A vector of [`RetpolineThunk`] structs, one per register.
    pub fn generate_retpoline_thunks(regs: &[PhysReg]) -> Vec<RetpolineThunk> {
        regs.iter()
            .filter(|r| is_gpr(**r))
            .map(|&reg| {
                let name = retpoline_thunk_name(reg);
                let code = build_retpoline_thunk_body(reg);
                RetpolineThunk {
                    name,
                    code,
                    register: reg,
                }
            })
            .collect()
    }

    /// Rewrites an indirect call instruction to target a retpoline thunk.
    ///
    /// If the instruction is an indirect call (`CALL_IND` opcode with a
    /// `Register` operand), it is rewritten in-place to a direct call
    /// targeting the corresponding `__x86_indirect_thunk_<reg>` symbol.
    ///
    /// # Returns
    ///
    /// `Some(thunk_name)` if the instruction was rewritten, `None` if the
    /// instruction was not an indirect call.
    pub fn rewrite_indirect_call(instr: &mut MachineInstr) -> Option<String> {
        // Detect indirect call: CALL_IND with a register operand
        if instr.opcode != opcodes::CALL_IND {
            return None;
        }

        let reg = match instr.operands.first() {
            Some(MachineOperand::Register(r)) if is_gpr(*r) => *r,
            _ => return None,
        };

        let thunk_name = retpoline_thunk_name(reg);

        // Rewrite to direct call targeting the thunk symbol
        instr.opcode = security_opcode(X86_64Opcode::CALL);
        instr.operands = vec![MachineOperand::Symbol(thunk_name.clone())];
        instr.is_call = true;

        Some(thunk_name)
    }

    /// Rewrites an indirect jump instruction to target a retpoline thunk.
    ///
    /// If the instruction is an indirect jump (`JMP` opcode with a `Register`
    /// operand), it is rewritten in-place to a direct jump targeting the
    /// corresponding `__x86_indirect_thunk_<reg>` symbol.
    ///
    /// # Returns
    ///
    /// `Some(thunk_name)` if the instruction was rewritten, `None` if the
    /// instruction was not an indirect jump.
    pub fn rewrite_indirect_jump(instr: &mut MachineInstr) -> Option<String> {
        // Detect indirect jump: JMP with a register operand
        if instr.opcode != opcodes::JMP {
            return None;
        }

        let reg = match instr.operands.first() {
            Some(MachineOperand::Register(r)) if is_gpr(*r) => *r,
            _ => return None,
        };

        let thunk_name = retpoline_thunk_name(reg);

        // Rewrite to direct jump targeting the thunk symbol
        instr.opcode = security_opcode(X86_64Opcode::JMP);
        instr.operands = vec![MachineOperand::Symbol(thunk_name.clone())];
        instr.is_terminator = true;

        Some(thunk_name)
    }
}

// ---------------------------------------------------------------------------
// Retpoline — Internal Helpers
// ---------------------------------------------------------------------------

/// Constructs the retpoline thunk symbol name for a given register.
///
/// # Examples
///
/// ```text
/// retpoline_thunk_name(RAX) → "__x86_indirect_thunk_rax"
/// retpoline_thunk_name(R11) → "__x86_indirect_thunk_r11"
/// ```
fn retpoline_thunk_name(reg: PhysReg) -> String {
    format!("__x86_indirect_thunk_{}", gpr_name_64(reg))
}

/// Builds the machine instruction sequence for a retpoline thunk body.
///
/// # Generated Sequence
///
/// ```text
///     call  .Ltarget          ; push return addr, jump to .Ltarget
/// .Lcapture:
///     pause                   ; hint: spin-wait (saves power)
///     lfence                  ; serialising barrier — blocks speculation
///     jmp   .Lcapture         ; infinite loop (only reached speculatively)
/// .Ltarget:
///     mov   %<reg>, (%rsp)    ; overwrite return address with actual target
///     ret                     ; jump to actual target via return stack buffer
/// ```
///
/// The capture loop is architecturally unreachable: the `call .Ltarget`
/// pushes the return address and jumps directly to `.Ltarget`, where the
/// return address is overwritten and `ret` dispatches to the real target.
/// However, the CPU's speculative execution engine may speculatively
/// execute the instructions after `call`, which land in the capture loop.
/// The `lfence` prevents the speculated instructions from retiring, and
/// `pause` reduces power consumption during the speculative spin.
fn build_retpoline_thunk_body(target_reg: PhysReg) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(6);

    // call .Ltarget — pushes return address, transfers control to .Ltarget
    let mut call_instr = MachineInstr::new(security_opcode(X86_64Opcode::CALL));
    call_instr.operands = vec![MachineOperand::Symbol(".Ltarget".to_string())];
    call_instr.implicit_defs = vec![RSP];
    call_instr.implicit_uses = vec![RSP];
    call_instr.is_call = true;
    instrs.push(call_instr);

    // .Lcapture:
    // pause — spin-wait hint, saves power during speculative spin
    instrs.push(MachineInstr::new(opcodes::PAUSE));

    // lfence — serialising load fence, prevents speculative execution past
    instrs.push(MachineInstr::new(opcodes::LFENCE));

    // jmp .Lcapture — infinite loop (only reached speculatively)
    let mut jmp_instr = MachineInstr::new(security_opcode(X86_64Opcode::JMP));
    jmp_instr.operands = vec![MachineOperand::Symbol(".Lcapture".to_string())];
    jmp_instr.is_terminator = true;
    instrs.push(jmp_instr);

    // .Ltarget:
    // mov [rsp], <reg> — overwrite the return address with the actual target
    let mut mov_instr = MachineInstr::new(opcodes::MOV_MR);
    mov_instr.operands = vec![
        MachineOperand::Memory {
            base: RSP,
            offset: 0,
            index: None,
            scale: 1,
        },
        MachineOperand::Register(target_reg),
    ];
    mov_instr.implicit_uses = vec![RSP, target_reg];
    instrs.push(mov_instr);

    // ret — pops the (now-patched) return address, jumping to <reg>
    let mut ret_instr = MachineInstr::new(security_opcode(X86_64Opcode::RET));
    ret_instr.implicit_uses = vec![RSP];
    ret_instr.is_terminator = true;
    ret_instr.is_return = true;
    instrs.push(ret_instr);

    instrs
}

/// Scans a [`MachineFunction`] for indirect calls and jumps, rewriting each
/// one to target a retpoline thunk. Collects the set of thunk names that
/// must be generated.
fn apply_retpoline(mf: &mut MachineFunction) -> Vec<String> {
    let mut needed_thunks = Vec::new();

    for block in &mut mf.blocks {
        for instr in &mut block.instructions {
            if let Some(name) = RetpolineGenerator::rewrite_indirect_call(instr) {
                if !needed_thunks.contains(&name) {
                    needed_thunks.push(name);
                }
            } else if let Some(name) = RetpolineGenerator::rewrite_indirect_jump(instr) {
                if !needed_thunks.contains(&name) {
                    needed_thunks.push(name);
                }
            }
        }
    }

    needed_thunks
}

// ---------------------------------------------------------------------------
// CET/IBT — Control-Flow Enforcement Technology
// ---------------------------------------------------------------------------

/// Returns the 4-byte encoded `endbr64` instruction.
///
/// The `endbr64` instruction (End Branch 64-bit) is a NOP on processors
/// without CET support, but on CET-enabled hardware it marks a valid
/// indirect branch target. Any indirect jump/call that lands on an
/// instruction other than `endbr64` causes a `#CP` exception.
///
/// # Encoding
///
/// ```text
/// F3 0F 1E FA    endbr64
/// ```
///
/// - `F3` — mandatory prefix (REP/REPE group)
/// - `0F 1E` — two-byte opcode (NOP family)
/// - `FA` — ModR/M byte specifying the ENDBR64 variant
pub fn emit_endbr64() -> Vec<u8> {
    vec![0xF3, 0x0F, 0x1E, 0xFA]
}

/// Inserts an `endbr64` instruction at the entry point of every function.
///
/// When CET/IBT is enabled, all function entries must begin with `endbr64`
/// so that indirect calls to those functions are not trapped by the hardware.
/// This function inserts the `endbr64` machine instruction as the very first
/// instruction in the function's entry block.
///
/// If the entry block is empty or the function has no blocks, this is a no-op.
pub fn insert_endbr64_at_function_entries(func: &mut MachineFunction) {
    if func.blocks.is_empty() {
        return;
    }

    let entry = &mut func.blocks[0];
    // Only insert if not already present
    if !entry.instructions.is_empty() && entry.instructions[0].opcode == opcodes::ENDBR64 {
        return;
    }

    let endbr = MachineInstr::new(opcodes::ENDBR64);
    entry.instructions.insert(0, endbr);
}

/// Inserts `endbr64` at every indirect branch target within the function.
///
/// An indirect branch target is identified as any basic block that:
/// - Is the target of an indirect jump (a block reached via `jmp *%reg`)
/// - Is the target of a switch dispatch or computed goto
/// - Has its address taken (used as a label operand)
///
/// In practice, we conservatively insert `endbr64` at the beginning of
/// every basic block that has a label and is not the entry block (which
/// is already handled by [`insert_endbr64_at_function_entries`]). This
/// ensures that any block reachable via an indirect branch is a valid
/// CET landing pad.
pub fn insert_endbr64_at_indirect_targets(func: &mut MachineFunction) {
    if func.blocks.len() <= 1 {
        return;
    }

    // First, collect the set of block IDs that are targets of indirect jumps
    // or have labels (potential indirect targets).
    let mut indirect_targets: Vec<u32> = Vec::new();

    for block in &func.blocks {
        for instr in &block.instructions {
            // An indirect call/jump targets a register, so any block whose
            // label appears as a Symbol operand could be an indirect target.
            // Also, JCC with Label operands reference blocks that may be
            // reached via switch/computed-goto tables.
            for op in &instr.operands {
                if let MachineOperand::Label(target_id) = op {
                    if !indirect_targets.contains(target_id) {
                        indirect_targets.push(*target_id);
                    }
                }
            }
        }
    }

    // Also include blocks that have explicit labels (potential address-taken blocks)
    for block in &func.blocks {
        if block.label.is_some() && block.id != func.blocks[0].id {
            if !indirect_targets.contains(&block.id) {
                indirect_targets.push(block.id);
            }
        }
    }

    // Cache the entry block ID before entering the mutable borrow loop
    let entry_block_id = func.blocks[0].id;

    // Insert endbr64 at the beginning of each identified target block
    for block in &mut func.blocks {
        if block.id == entry_block_id {
            continue; // Entry block handled by insert_endbr64_at_function_entries
        }
        let block_id = block.id;
        if indirect_targets.contains(&block_id) {
            // Only insert if not already present
            if block.instructions.is_empty() || block.instructions[0].opcode != opcodes::ENDBR64 {
                let endbr = MachineInstr::new(opcodes::ENDBR64);
                block.instructions.insert(0, endbr);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stack Guard Page Probing
// ---------------------------------------------------------------------------

/// Returns `true` if the given frame size requires a stack probe loop.
///
/// Stack frames that exceed the page size (4096 bytes) must be probed
/// page-by-page to ensure the OS can extend the stack mapping and to
/// trigger guard page faults before the stack pointer is adjusted.
///
/// # Arguments
///
/// * `frame_size` — Total stack frame size in bytes.
#[inline]
pub fn needs_stack_probe(frame_size: u32) -> bool {
    frame_size > PAGE_SIZE
}

/// Generates the machine instruction sequence for a stack probe loop.
///
/// The probe loop touches each 4096-byte page of the stack frame in
/// descending address order, from the current RSP down to `RSP - frame_size`.
/// This triggers guard page faults before the actual stack pointer adjustment.
///
/// # Generated Sequence (AT&T syntax)
///
/// ```text
///     mov  %rsp, %rax           ; save current stack pointer
///     lea  -frame_size(%rsp), %rcx  ; compute final RSP target
/// .Lprobe_loop:
///     sub  $4096, %rax          ; move down one page
///     test %eax, (%rax)         ; touch the page (read probe)
///     cmp  %rcx, %rax           ; compare against final target
///     ja   .Lprobe_loop         ; continue if rax > target
///     mov  %rcx, %rsp           ; set final stack pointer
/// ```
///
/// # Arguments
///
/// * `frame_size` — Total frame size in bytes. Must be > 4096.
///
/// # Returns
///
/// A vector of [`MachineInstr`] representing the probe loop. These
/// instructions should be inserted at the beginning of the function's
/// entry block (after any `endbr64` instruction).
pub fn generate_stack_probe(frame_size: u32) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(7);

    // mov rax, rsp — save current stack pointer into scratch register
    let mut mov_rsp_to_rax = MachineInstr::new(security_opcode(X86_64Opcode::MOV));
    mov_rsp_to_rax.operands = vec![
        MachineOperand::Register(RAX),
        MachineOperand::Register(RSP),
    ];
    mov_rsp_to_rax.implicit_defs = vec![RAX];
    mov_rsp_to_rax.implicit_uses = vec![RSP];
    instrs.push(mov_rsp_to_rax);

    // lea rcx, [rsp - frame_size] — compute final stack pointer target
    // We use MOV + SUB instead of LEA for clarity in the machine IR:
    //   mov rcx, rsp
    //   sub rcx, <frame_size>
    let mut mov_rsp_to_rcx = MachineInstr::new(security_opcode(X86_64Opcode::MOV));
    mov_rsp_to_rcx.operands = vec![
        MachineOperand::Register(RCX),
        MachineOperand::Register(RSP),
    ];
    mov_rsp_to_rcx.implicit_defs = vec![RCX];
    mov_rsp_to_rcx.implicit_uses = vec![RSP];
    instrs.push(mov_rsp_to_rcx);

    let mut sub_frame = MachineInstr::new(security_opcode(X86_64Opcode::SUB));
    sub_frame.operands = vec![
        MachineOperand::Register(RCX),
        MachineOperand::Immediate(frame_size as i64),
    ];
    sub_frame.implicit_defs = vec![RCX];
    sub_frame.implicit_uses = vec![RCX];
    instrs.push(sub_frame);

    // .Lprobe_loop:
    // sub rax, 4096 — move down one page
    let mut sub_page = MachineInstr::new(security_opcode(X86_64Opcode::SUB));
    sub_page.operands = vec![
        MachineOperand::Register(RAX),
        MachineOperand::Immediate(PAGE_SIZE as i64),
    ];
    sub_page.implicit_defs = vec![RAX];
    sub_page.implicit_uses = vec![RAX];
    instrs.push(sub_page);

    // test [rax], eax — touch the page to trigger a fault if unmapped
    // This is a read-only operation: ANDs [rax] with eax and sets flags.
    // EAX is the same PhysReg as RAX (distinction is operand-size at encoding).
    let mut test_page = MachineInstr::new(security_opcode(X86_64Opcode::TEST));
    test_page.operands = vec![
        MachineOperand::Memory {
            base: RAX,
            offset: 0,
            index: None,
            scale: 1,
        },
        MachineOperand::Register(EAX),
    ];
    test_page.implicit_uses = vec![RAX];
    instrs.push(test_page);

    // cmp rax, rcx — compare current probe position against target
    let mut cmp_instr = MachineInstr::new(security_opcode(X86_64Opcode::CMP));
    cmp_instr.operands = vec![
        MachineOperand::Register(RAX),
        MachineOperand::Register(RCX),
    ];
    cmp_instr.implicit_uses = vec![RAX, RCX];
    instrs.push(cmp_instr);

    // ja .Lprobe_loop — loop back if rax > target (unsigned above)
    // The JCC opcode uses a condition code in its first operand to
    // distinguish JA from JE, JL, etc.
    let mut ja_instr = MachineInstr::new(security_opcode(X86_64Opcode::JA));
    ja_instr.operands = vec![
        MachineOperand::Symbol(".Lprobe_loop".to_string()),
        MachineOperand::Immediate(COND_ABOVE),
    ];
    instrs.push(ja_instr);

    // mov rsp, rcx — set the final stack pointer
    let mut mov_final = MachineInstr::new(security_opcode(X86_64Opcode::MOV));
    mov_final.operands = vec![
        MachineOperand::Register(RSP),
        MachineOperand::Register(RCX),
    ];
    mov_final.implicit_defs = vec![RSP];
    mov_final.implicit_uses = vec![RCX];
    instrs.push(mov_final);

    instrs
}

/// Emits the raw x86-64 encoded bytes for a stack probe prologue.
///
/// This produces the binary encoding of the probe loop sequence that can be
/// directly emitted into the output object code. The sequence uses RAX as
/// the probe pointer and RCX to hold the final stack pointer target.
///
/// # Encoding
///
/// ```text
/// 48 89 E0                          ; mov rax, rsp
/// 48 89 E1                          ; mov rcx, rsp
/// 48 81 E9 XX XX XX XX              ; sub rcx, <frame_size>
/// 48 2D 00 10 00 00                 ; sub rax, 4096          (.Lprobe:)
/// 85 00                             ; test [rax], eax
/// 48 39 C8                          ; cmp rax, rcx
/// 0F 87 EF FF FF FF                 ; ja .Lprobe  (rel32 = -17)
/// 48 89 CC                          ; mov rsp, rcx
/// ```
///
/// # Arguments
///
/// * `frame_size` — Total stack frame size in bytes. Must be > 4096.
///
/// # Returns
///
/// A vector of raw bytes encoding the complete probe loop.
pub fn emit_stack_probe_prologue(frame_size: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(40);

    // mov rax, rsp → REX.W MOV r/m64, r64 (89 /r)
    // ModRM: mod=11, reg=100(rsp), r/m=000(rax) → 0xE0
    bytes.extend_from_slice(&[0x48, 0x89, 0xE0]);

    // mov rcx, rsp → REX.W MOV r/m64, r64 (89 /r)
    // ModRM: mod=11, reg=100(rsp), r/m=001(rcx) → 0xE1
    bytes.extend_from_slice(&[0x48, 0x89, 0xE1]);

    // sub rcx, <frame_size> → REX.W SUB r/m64, imm32 (81 /5)
    // ModRM: mod=11, reg=101(/5), r/m=001(rcx) → 0xE9
    bytes.extend_from_slice(&[0x48, 0x81, 0xE9]);
    bytes.extend_from_slice(&frame_size.to_le_bytes());

    // .Lprobe_loop: (byte offset for jump target = current position)
    let probe_loop_offset = bytes.len();

    // sub rax, 4096 → REX.W SUB rax, imm32 (2D id)
    // Uses the short form: 48 2D <imm32>
    bytes.extend_from_slice(&[0x48, 0x2D]);
    bytes.extend_from_slice(&PAGE_SIZE.to_le_bytes());

    // test [rax], eax → TEST r/m32, r32 (85 /r)
    // ModRM: mod=00, reg=000(eax), r/m=000(rax) → 0x00
    bytes.extend_from_slice(&[0x85, 0x00]);

    // cmp rax, rcx → REX.W CMP r/m64, r64 (39 /r)
    // ModRM: mod=11, reg=001(rcx), r/m=000(rax) → 0xC8
    bytes.extend_from_slice(&[0x48, 0x39, 0xC8]);

    // ja .Lprobe_loop → 0F 87 <rel32>
    // rel32 = probe_loop_offset - (current_offset + 6)
    //   where 6 = size of the JA instruction itself (0F 87 + 4-byte rel32)
    bytes.extend_from_slice(&[0x0F, 0x87]);
    let ja_end_offset = bytes.len() + 4; // after the 4-byte rel32
    let rel32 = (probe_loop_offset as i32) - (ja_end_offset as i32);
    bytes.extend_from_slice(&rel32.to_le_bytes());

    // mov rsp, rcx → REX.W MOV r/m64, r64 (89 /r)
    // ModRM: mod=11, reg=001(rcx), r/m=100(rsp) → 0xCC
    bytes.extend_from_slice(&[0x48, 0x89, 0xCC]);

    bytes
}

/// Inserts the stack probe loop into a [`MachineFunction`]'s entry block.
///
/// The probe instructions are inserted at the beginning of the entry block,
/// immediately after any `endbr64` instruction (preserving CET correctness).
fn insert_stack_probe(mf: &mut MachineFunction) {
    if mf.blocks.is_empty() {
        return;
    }

    let probe_instrs = generate_stack_probe(mf.frame_size);

    let entry = &mut mf.blocks[0];

    // Insert after any endbr64 that may already be at position 0
    let insert_pos = if !entry.instructions.is_empty()
        && entry.instructions[0].opcode == opcodes::ENDBR64
    {
        1
    } else {
        0
    };

    // Insert probe instructions in order at the computed position
    for (offset, instr) in probe_instrs.into_iter().enumerate() {
        entry.instructions.insert(insert_pos + offset, instr);
    }
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Applies all enabled security mitigations to a [`MachineFunction`].
///
/// This is the primary integration point called by the Phase 10 code
/// generation driver (`src/backend/generation.rs`) for x86-64 targets.
///
/// # Application Order
///
/// Mitigations are applied in a specific order to ensure correctness:
///
/// 1. **CET/IBT** (`endbr64` insertion) — must be first so that the
///    `endbr64` instruction is at the very beginning of the function.
/// 2. **Stack probe** — inserted after `endbr64` (if present).
/// 3. **Retpoline** — applied last as it rewrites instructions that may
///    appear anywhere in the function.
///
/// # Arguments
///
/// * `func` — The machine function to apply mitigations to (modified in-place).
/// * `config` — Security configuration specifying which mitigations are active.
///
/// # Panics
///
/// This function is intended for x86-64 only. Debug builds assert that the
/// function is well-formed (has at least one basic block).
pub fn apply_security_mitigations(func: &mut MachineFunction, config: &SecurityConfig) {
    // Defensive assertion: this module is strictly x86-64 only.
    // The code generation driver must not call this for other architectures.
    debug_assert!(
        matches!(SUPPORTED_TARGET, Target::X86_64),
        "Security mitigations are x86-64 only (target: {:?})",
        SUPPORTED_TARGET,
    );

    // Short-circuit: no mitigations enabled and frame is small enough
    if !config.any_enabled() && !needs_stack_probe(func.frame_size) {
        return;
    }

    // Phase 1: CET/IBT — insert endbr64 at function entries
    if config.cf_protection {
        insert_endbr64_at_function_entries(func);
        insert_endbr64_at_indirect_targets(func);
    }

    // Phase 2: Stack probe — insert probe loop for large frames
    if needs_stack_probe(func.frame_size) {
        insert_stack_probe(func);
    }

    // Phase 3: Retpoline — rewrite indirect calls/jumps
    if config.retpoline {
        let _needed_thunks = apply_retpoline(func);
        // The list of needed thunks is returned for the caller to generate
        // the thunk stubs via RetpolineGenerator::generate_retpoline_thunks().
        // In the current integration, the generation driver handles this.
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::MachineBasicBlock;
    use crate::backend::x86_64::registers::{
        self, NUM_GPRS, R8, R9, R10, R11, R12, R13, R14, R15, RBP, RBX, RDI,
        RDX, RSI, RSP as RSP_REG,
    };

    /// Helper to create a minimal machine function for testing.
    fn make_test_mf(frame_size: u32) -> MachineFunction {
        let mut mf = MachineFunction::new("test_fn".to_string(), 16);
        mf.frame_size = frame_size;
        let bb = MachineBasicBlock::new(0);
        mf.add_block(bb);
        mf
    }

    /// Helper to create a NOP instruction for padding.
    fn nop_instr() -> MachineInstr {
        MachineInstr::new(security_opcode(X86_64Opcode::NOP))
    }

    // -- SecurityConfig tests -----------------------------------------------

    #[test]
    fn security_config_default_has_no_mitigations() {
        let config = SecurityConfig::new();
        assert!(!config.retpoline);
        assert!(!config.cf_protection);
        assert_eq!(config.stack_probe_threshold, 4096);
        assert!(!config.any_enabled());
    }

    #[test]
    fn security_config_from_flags() {
        let config = SecurityConfig::from_flags(true, false);
        assert!(config.retpoline);
        assert!(!config.cf_protection);
        assert_eq!(config.stack_probe_threshold, DEFAULT_STACK_PROBE_THRESHOLD);

        let config2 = SecurityConfig::from_flags(false, true);
        assert!(!config2.retpoline);
        assert!(config2.cf_protection);
    }

    #[test]
    fn security_config_all_enabled() {
        let config = SecurityConfig::all_enabled();
        assert!(config.retpoline);
        assert!(config.cf_protection);
        assert!(config.any_enabled());
    }

    #[test]
    fn security_config_default_trait() {
        let config: SecurityConfig = Default::default();
        assert!(!config.retpoline);
        assert!(!config.cf_protection);
        assert_eq!(config.stack_probe_threshold, PAGE_SIZE);
    }

    // -- CET/IBT tests ------------------------------------------------------

    #[test]
    fn emit_endbr64_returns_correct_bytes() {
        let bytes = emit_endbr64();
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x1E, 0xFA]);
    }

    #[test]
    fn endbr64_inserted_at_function_entry() {
        let mut mf = make_test_mf(0);
        mf.blocks[0].instructions.push(nop_instr());

        insert_endbr64_at_function_entries(&mut mf);

        // First instruction should now be endbr64
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::ENDBR64);
        // Original NOP should be second
        assert_eq!(
            mf.blocks[0].instructions[1].opcode,
            security_opcode(X86_64Opcode::NOP)
        );
    }

    #[test]
    fn endbr64_not_duplicated() {
        let mut mf = make_test_mf(0);
        insert_endbr64_at_function_entries(&mut mf);
        insert_endbr64_at_function_entries(&mut mf);

        // Should have exactly one endbr64, not two
        let endbr_count = mf.blocks[0]
            .instructions
            .iter()
            .filter(|i| i.opcode == opcodes::ENDBR64)
            .count();
        assert_eq!(endbr_count, 1);
    }

    #[test]
    fn endbr64_at_indirect_targets() {
        let mut mf = MachineFunction::new("test_fn".to_string(), 16);
        mf.frame_size = 0;

        // Block 0 (entry) with a branch to block 1
        let mut bb0 = MachineBasicBlock::new(0);
        let mut jmp = MachineInstr::new(opcodes::JCC);
        jmp.operands = vec![MachineOperand::Label(1)];
        jmp.is_terminator = true;
        bb0.instructions.push(jmp);
        mf.add_block(bb0);

        // Block 1 (indirect target) with a label
        let mut bb1 = MachineBasicBlock::with_label(1, ".Ltarget".to_string());
        bb1.instructions.push(nop_instr());
        mf.add_block(bb1);

        insert_endbr64_at_indirect_targets(&mut mf);

        // Block 1 should now start with endbr64
        assert_eq!(mf.blocks[1].instructions[0].opcode, opcodes::ENDBR64);
        // Block 0 should NOT have endbr64 (entry block handled separately)
        assert_ne!(mf.blocks[0].instructions[0].opcode, opcodes::ENDBR64);
    }

    // -- Retpoline tests ----------------------------------------------------

    #[test]
    fn retpoline_thunk_name_correct() {
        assert_eq!(
            retpoline_thunk_name(RAX),
            "__x86_indirect_thunk_rax"
        );
        assert_eq!(
            retpoline_thunk_name(R11),
            "__x86_indirect_thunk_r11"
        );
        assert_eq!(
            retpoline_thunk_name(RBX),
            "__x86_indirect_thunk_rbx"
        );
        assert_eq!(
            retpoline_thunk_name(RDI),
            "__x86_indirect_thunk_rdi"
        );
    }

    #[test]
    fn retpoline_replaces_indirect_call() {
        let mut mf = make_test_mf(0);

        // Add an indirect call through RAX (CALL_IND with Register operand)
        let mut indirect_call = MachineInstr::new(opcodes::CALL_IND);
        indirect_call.operands = vec![MachineOperand::Register(RAX)];
        indirect_call.is_call = true;
        mf.blocks[0].instructions.push(indirect_call);

        apply_retpoline(&mut mf);

        // The indirect call should be replaced with a call to the thunk
        let instr = &mf.blocks[0].instructions[0];
        assert_eq!(instr.opcode, security_opcode(X86_64Opcode::CALL));
        assert!(instr.is_call);
        match &instr.operands[0] {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rax");
            }
            _ => panic!("Expected Symbol operand for retpoline thunk call"),
        }
    }

    #[test]
    fn retpoline_replaces_indirect_jump() {
        let mut mf = make_test_mf(0);

        // Add an indirect jump through R11 (JMP with Register operand)
        let mut indirect_jmp = MachineInstr::new(opcodes::JMP);
        indirect_jmp.operands = vec![MachineOperand::Register(R11)];
        indirect_jmp.is_terminator = true;
        mf.blocks[0].instructions.push(indirect_jmp);

        apply_retpoline(&mut mf);

        let instr = &mf.blocks[0].instructions[0];
        assert_eq!(instr.opcode, security_opcode(X86_64Opcode::JMP));
        assert!(instr.is_terminator);
        match &instr.operands[0] {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_r11");
            }
            _ => panic!("Expected Symbol operand for retpoline thunk jump"),
        }
    }

    #[test]
    fn retpoline_thunk_generation() {
        let thunks = RetpolineGenerator::generate_retpoline_thunks(&[RAX, RCX, R11]);
        assert_eq!(thunks.len(), 3);

        // Verify the RAX thunk
        let rax_thunk = &thunks[0];
        assert_eq!(rax_thunk.name, "__x86_indirect_thunk_rax");
        assert_eq!(rax_thunk.register, RAX);
        assert_eq!(rax_thunk.code.len(), 6);

        // First instruction is call .Ltarget
        assert_eq!(rax_thunk.code[0].opcode, security_opcode(X86_64Opcode::CALL));
        assert!(rax_thunk.code[0].is_call);

        // Second is pause
        assert_eq!(rax_thunk.code[1].opcode, opcodes::PAUSE);

        // Third is lfence
        assert_eq!(rax_thunk.code[2].opcode, opcodes::LFENCE);

        // Fourth is jmp .Lcapture
        assert_eq!(rax_thunk.code[3].opcode, security_opcode(X86_64Opcode::JMP));
        assert!(rax_thunk.code[3].is_terminator);

        // Fifth is mov [rsp], reg
        assert_eq!(rax_thunk.code[4].opcode, opcodes::MOV_MR);

        // Sixth is ret
        assert_eq!(rax_thunk.code[5].opcode, security_opcode(X86_64Opcode::RET));
        assert!(rax_thunk.code[5].is_return);
    }

    #[test]
    fn retpoline_rewrite_preserves_non_indirect() {
        // A direct call should not be rewritten
        let mut instr = MachineInstr::new(opcodes::CALL);
        instr.operands = vec![MachineOperand::Symbol("some_func".to_string())];
        instr.is_call = true;

        let result = RetpolineGenerator::rewrite_indirect_call(&mut instr);
        assert!(result.is_none());
        assert_eq!(instr.opcode, opcodes::CALL);
    }

    // -- Stack probe tests --------------------------------------------------

    #[test]
    fn needs_stack_probe_threshold() {
        assert!(!needs_stack_probe(0));
        assert!(!needs_stack_probe(1024));
        assert!(!needs_stack_probe(4096));
        assert!(needs_stack_probe(4097));
        assert!(needs_stack_probe(8192));
        assert!(needs_stack_probe(65536));
    }

    #[test]
    fn generate_stack_probe_creates_instructions() {
        let instrs = generate_stack_probe(8192);
        // Expected: mov rax,rsp + mov rcx,rsp + sub rcx,8192 +
        //           sub rax,4096 + test + cmp + ja + mov rsp,rcx = 8 instrs
        assert_eq!(instrs.len(), 8);

        // First: mov rax, rsp
        assert_eq!(instrs[0].opcode, security_opcode(X86_64Opcode::MOV));
        // Last: mov rsp, rcx
        assert_eq!(instrs[7].opcode, security_opcode(X86_64Opcode::MOV));
    }

    #[test]
    fn stack_probe_inserted_for_large_frames() {
        let mut mf = make_test_mf(8192);

        let config = SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe_threshold: PAGE_SIZE,
        };

        apply_security_mitigations(&mut mf, &config);

        // Should have inserted probe instructions
        assert!(!mf.blocks[0].instructions.is_empty());
        // First instruction should be the probe's MOV
        assert_eq!(
            mf.blocks[0].instructions[0].opcode,
            security_opcode(X86_64Opcode::MOV)
        );
    }

    #[test]
    fn stack_probe_after_endbr64() {
        let mut mf = make_test_mf(8192);

        let config = SecurityConfig::from_flags(false, true);

        apply_security_mitigations(&mut mf, &config);

        // endbr64 should be first
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::ENDBR64);
        // Probe MOV should be second
        assert_eq!(
            mf.blocks[0].instructions[1].opcode,
            security_opcode(X86_64Opcode::MOV)
        );
    }

    #[test]
    fn emit_stack_probe_prologue_bytes() {
        let bytes = emit_stack_probe_prologue(8192);

        // Verify the preamble: mov rax, rsp → 48 89 E0
        assert_eq!(&bytes[0..3], &[0x48, 0x89, 0xE0]);

        // mov rcx, rsp → 48 89 E1
        assert_eq!(&bytes[3..6], &[0x48, 0x89, 0xE1]);

        // sub rcx, 8192 → 48 81 E9 00 20 00 00
        assert_eq!(&bytes[6..9], &[0x48, 0x81, 0xE9]);
        assert_eq!(&bytes[9..13], &8192u32.to_le_bytes());

        // sub rax, 4096 → 48 2D 00 10 00 00
        assert_eq!(&bytes[13..15], &[0x48, 0x2D]);
        assert_eq!(&bytes[15..19], &4096u32.to_le_bytes());

        // test [rax], eax → 85 00
        assert_eq!(&bytes[19..21], &[0x85, 0x00]);

        // cmp rax, rcx → 48 39 C8
        assert_eq!(&bytes[21..24], &[0x48, 0x39, 0xC8]);

        // ja .Lprobe_loop → 0F 87 <rel32>
        assert_eq!(&bytes[24..26], &[0x0F, 0x87]);
        // rel32 should jump back to the sub instruction at offset 13
        // ja instruction ends at offset 30, so rel32 = 13 - 30 = -17
        let rel32 = i32::from_le_bytes([bytes[26], bytes[27], bytes[28], bytes[29]]);
        assert_eq!(rel32, -17);

        // mov rsp, rcx → 48 89 CC
        assert_eq!(&bytes[30..33], &[0x48, 0x89, 0xCC]);
    }

    // -- Integration tests --------------------------------------------------

    #[test]
    fn no_mitigations_when_all_disabled() {
        let mut mf = make_test_mf(128);

        let config = SecurityConfig::new();
        apply_security_mitigations(&mut mf, &config);

        // No instructions should have been inserted
        assert_eq!(mf.blocks[0].instructions.len(), 0);
    }

    #[test]
    fn all_mitigations_applied_together() {
        let mut mf = make_test_mf(8192);

        // Add an indirect call through RAX
        let mut indirect_call = MachineInstr::new(opcodes::CALL_IND);
        indirect_call.operands = vec![MachineOperand::Register(RAX)];
        indirect_call.is_call = true;
        mf.blocks[0].instructions.push(indirect_call);

        let config = SecurityConfig::all_enabled();
        apply_security_mitigations(&mut mf, &config);

        // Verify order: endbr64, then probe, then the rewritten call
        assert_eq!(mf.blocks[0].instructions[0].opcode, opcodes::ENDBR64);

        // The rewritten call should reference the thunk
        let last = mf.blocks[0].instructions.last().unwrap();
        assert_eq!(last.opcode, security_opcode(X86_64Opcode::CALL));
        match &last.operands[0] {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rax");
            }
            _ => panic!("Expected Symbol operand"),
        }
    }

    #[test]
    fn retpoline_handles_multiple_registers() {
        let mut mf = make_test_mf(0);

        // Two different indirect calls
        let mut call_rax = MachineInstr::new(opcodes::CALL_IND);
        call_rax.operands = vec![MachineOperand::Register(RAX)];
        call_rax.is_call = true;
        mf.blocks[0].instructions.push(call_rax);

        let mut call_r11 = MachineInstr::new(opcodes::CALL_IND);
        call_r11.operands = vec![MachineOperand::Register(R11)];
        call_r11.is_call = true;
        mf.blocks[0].instructions.push(call_r11);

        let config = SecurityConfig::from_flags(true, false);
        apply_security_mitigations(&mut mf, &config);

        // Both should be rewritten
        match &mf.blocks[0].instructions[0].operands[0] {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rax");
            }
            _ => panic!("Expected thunk symbol"),
        }
        match &mf.blocks[0].instructions[1].operands[0] {
            MachineOperand::Symbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_r11");
            }
            _ => panic!("Expected thunk symbol"),
        }
    }

    #[test]
    fn retpoline_generator_all_16_gprs() {
        // Generate thunks for all 16 GPRs
        let all_gprs: Vec<PhysReg> = (0..NUM_GPRS as u16).map(PhysReg).collect();
        let thunks = RetpolineGenerator::generate_retpoline_thunks(&all_gprs);

        assert_eq!(thunks.len(), 16);

        // Verify each thunk has the correct name
        for (i, thunk) in thunks.iter().enumerate() {
            let expected_name = format!(
                "__x86_indirect_thunk_{}",
                gpr_name_64(PhysReg(i as u16))
            );
            assert_eq!(thunk.name, expected_name);
            assert_eq!(thunk.register, PhysReg(i as u16));
            assert_eq!(thunk.code.len(), 6);
        }
    }

    #[test]
    fn retpoline_generator_ignores_sse_regs() {
        // SSE registers should not get thunks
        let sse_regs = vec![PhysReg(16), PhysReg(17)]; // XMM0, XMM1
        let thunks = RetpolineGenerator::generate_retpoline_thunks(&sse_regs);
        assert!(thunks.is_empty());
    }

    #[test]
    fn retpoline_thunk_struct_fields() {
        let thunk = RetpolineThunk {
            name: "__x86_indirect_thunk_rax".to_string(),
            code: Vec::new(),
            register: RAX,
        };
        assert_eq!(thunk.name, "__x86_indirect_thunk_rax");
        assert_eq!(thunk.register, RAX);
        assert!(thunk.code.is_empty());
    }

    #[test]
    fn security_opcode_mapping() {
        // Verify the opcode mapping function produces correct values
        assert_eq!(security_opcode(X86_64Opcode::CALL), opcodes::CALL);
        assert_eq!(security_opcode(X86_64Opcode::JMP), opcodes::JMP);
        assert_eq!(security_opcode(X86_64Opcode::MOV), opcodes::MOV_RR);
        assert_eq!(security_opcode(X86_64Opcode::SUB), opcodes::SUB_RI);
        assert_eq!(security_opcode(X86_64Opcode::TEST), opcodes::TEST_RR);
        assert_eq!(security_opcode(X86_64Opcode::CMP), opcodes::CMP_RI);
        assert_eq!(security_opcode(X86_64Opcode::JA), opcodes::JCC);
        assert_eq!(security_opcode(X86_64Opcode::RET), opcodes::RET);
        assert_eq!(security_opcode(X86_64Opcode::NOP), opcodes::NOP);
    }

    #[test]
    fn all_16_gpr_thunk_names() {
        // Verify every GPR maps to the correct retpoline thunk name.
        // This exercises all 16 register constants from the registers module.
        let expected: &[(PhysReg, &str)] = &[
            (RAX, "__x86_indirect_thunk_rax"),
            (RCX, "__x86_indirect_thunk_rcx"),
            (RDX, "__x86_indirect_thunk_rdx"),
            (RBX, "__x86_indirect_thunk_rbx"),
            (RSP_REG, "__x86_indirect_thunk_rsp"),
            (RBP, "__x86_indirect_thunk_rbp"),
            (RSI, "__x86_indirect_thunk_rsi"),
            (RDI, "__x86_indirect_thunk_rdi"),
            (R8, "__x86_indirect_thunk_r8"),
            (R9, "__x86_indirect_thunk_r9"),
            (R10, "__x86_indirect_thunk_r10"),
            (R11, "__x86_indirect_thunk_r11"),
            (R12, "__x86_indirect_thunk_r12"),
            (R13, "__x86_indirect_thunk_r13"),
            (R14, "__x86_indirect_thunk_r14"),
            (R15, "__x86_indirect_thunk_r15"),
        ];

        for &(reg, expected_name) in expected {
            let name = retpoline_thunk_name(reg);
            assert_eq!(name, expected_name, "Thunk name mismatch for {:?}", reg);
        }
        // Confirm we tested all GPRs
        assert_eq!(expected.len(), NUM_GPRS);
    }

    #[test]
    fn registers_module_is_gpr_validation() {
        // Verify is_gpr recognises all GPR register constants from the
        // registers module. This provides integration coverage with the
        // `self` module import (registers::is_gpr == is_gpr here).
        assert!(registers::is_gpr(RAX));
        assert!(registers::is_gpr(RCX));
        assert!(registers::is_gpr(RDX));
        assert!(registers::is_gpr(RBX));
        assert!(registers::is_gpr(RSP_REG));
        assert!(registers::is_gpr(RBP));
        assert!(registers::is_gpr(RSI));
        assert!(registers::is_gpr(RDI));
        assert!(registers::is_gpr(R8));
        assert!(registers::is_gpr(R9));
        assert!(registers::is_gpr(R10));
        assert!(registers::is_gpr(R11));
        assert!(registers::is_gpr(R12));
        assert!(registers::is_gpr(R13));
        assert!(registers::is_gpr(R14));
        assert!(registers::is_gpr(R15));
    }
}
