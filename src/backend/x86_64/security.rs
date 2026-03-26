//! # x86-64 Security Mitigations
//!
//! Implements security-hardened code generation for x86-64:
//!
//! - **Retpoline** (`-mretpoline`): Transforms indirect calls/jumps into thunk
//!   calls that prevent speculative execution of the indirect target (Spectre v2
//!   mitigation).
//! - **CET/IBT** (`-fcf-protection`): Inserts `endbr64` instructions at function
//!   entries and indirect branch targets for Intel Control-flow Enforcement
//!   Technology — Indirect Branch Tracking.
//! - **Stack guard page probing**: For stack frames exceeding 4,096 bytes, emits
//!   a loop that touches each page before the large allocation to prevent
//!   stack-clash attacks that skip over the guard page.
//!
//! ## x86-64 Only
//!
//! These mitigations are **exclusively** for the x86-64 target.  They MUST NOT
//! be applied to i686, AArch64, or RISC-V 64.
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references.  No external crates.

use crate::backend::traits::{MachineFunction, MachineInstruction, MachineOperand};
use crate::backend::x86_64::codegen::X86Opcode;
use crate::backend::x86_64::registers::{
    hw_encoding, reg_name_64, R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX,
    RSI, RSP,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Machine-code encoding of the `endbr64` instruction (CET indirect-branch
/// tracking marker).  This 4-byte sequence is a NOP on non-CET hardware,
/// ensuring full backward compatibility.
pub const ENDBR64_BYTES: [u8; 4] = [0xF3, 0x0F, 0x1E, 0xFA];

/// Virtual-memory page size in bytes.  Stack frames exceeding this threshold
/// require a guard-page probe loop during the prologue to prevent stack-clash
/// attacks.
pub const PAGE_SIZE: usize = 4096;

/// All 16 general-purpose registers for which retpoline thunks are generated.
/// While RSP and RBP are unlikely indirect-call targets in practice, thunks
/// are emitted for all 16 GPRs for completeness.
const RETPOLINE_GPRS: [u16; 16] = [
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
];

// ===========================================================================
// SecurityConfig — CLI flag mapping
// ===========================================================================

/// Security mitigation configuration derived from CLI flags.
///
/// Each field corresponds to a compiler flag:
///
/// | Field           | CLI Flag          | Behaviour                        |
/// |-----------------|-------------------|----------------------------------|
/// | `retpoline`     | `-mretpoline`     | Transform indirect calls         |
/// | `cf_protection` | `-fcf-protection` | Insert `endbr64` instructions    |
/// | `stack_probe`   | (always active)   | Probe pages when frame > 4 KiB  |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityConfig {
    /// Enable retpoline thunk generation and indirect-call transformation.
    pub retpoline: bool,
    /// Enable CET/IBT `endbr64` insertion at function entries and indirect
    /// branch targets.
    pub cf_protection: bool,
    /// Enable stack guard-page probing for large frames.  This is always
    /// checked during prologue generation for x86-64 — the flag allows
    /// explicit opt-out in unusual scenarios.
    pub stack_probe: bool,
}

impl SecurityConfig {
    /// Create a configuration with all mitigations disabled.
    #[inline]
    pub fn none() -> Self {
        SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe: false,
        }
    }

    /// Create a configuration with all mitigations enabled.
    #[inline]
    pub fn all() -> Self {
        SecurityConfig {
            retpoline: true,
            cf_protection: true,
            stack_probe: true,
        }
    }

    /// Returns `true` if any mitigation is active.
    #[inline]
    pub fn any_active(&self) -> bool {
        self.retpoline || self.cf_protection || self.stack_probe
    }
}

impl Default for SecurityConfig {
    /// Default: stack probing enabled (always for x86-64), others disabled.
    fn default() -> Self {
        SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe: true,
        }
    }
}

// ===========================================================================
// RetpolineThunk — pre-encoded retpoline thunk data
// ===========================================================================

/// A pre-encoded retpoline thunk for a specific register.
///
/// The retpoline thunk prevents speculative execution of indirect branch
/// targets (Spectre v2 / CVE-2017-5715 mitigation).  Each thunk:
///
/// 1. Pushes a return address onto the stack via `call`.
/// 2. Enters an infinite `pause; lfence; jmp` speculation trap.
/// 3. At the call target, overwrites the return address with the intended
///    indirect target register value, then `ret`s to it.
///
/// ## Machine Code Layout
///
/// ```text
/// __x86_indirect_thunk_<reg>:
///     call .Lretpoline_call_target       ; E8 07 00 00 00
/// .Lretpoline_capture:
///     pause                              ; F3 90
///     lfence                             ; 0F AE E8
///     jmp .Lretpoline_capture            ; EB F9
/// .Lretpoline_call_target:
///     mov [rsp], <reg>                   ; <REX> 89 <ModR/M> 24
///     ret                                ; C3
/// ```
#[derive(Debug, Clone)]
pub struct RetpolineThunk {
    /// Symbol name, e.g. `__x86_indirect_thunk_rax`.
    pub name: String,
    /// Physical register index (matches `MachineOperand::Register` encoding).
    pub register: u16,
    /// Pre-encoded machine code bytes for this thunk.
    pub encoded_bytes: Vec<u8>,
}

// ===========================================================================
// Retpoline — thunk name generation
// ===========================================================================

/// Construct the retpoline thunk symbol name for a given register name.
///
/// # Examples
///
/// ```
/// # use bcc::backend::x86_64::security::retpoline_thunk_name;
/// assert_eq!(retpoline_thunk_name("rax"), "__x86_indirect_thunk_rax");
/// assert_eq!(retpoline_thunk_name("r11"), "__x86_indirect_thunk_r11");
/// ```
#[inline]
pub fn retpoline_thunk_name(reg: &str) -> String {
    format!("__x86_indirect_thunk_{}", reg)
}

// ===========================================================================
// Retpoline — thunk machine-code encoding
// ===========================================================================

/// Encode the `mov [rsp], <reg>` instruction for the given physical register.
///
/// Returns the 4-byte encoding: `<REX> 89 <ModR/M> 24`.
///
/// - For registers 0–7 (RAX–RDI): REX = 0x48 (REX.W only).
/// - For registers 8–15 (R8–R15): REX = 0x4C (REX.W + REX.R).
fn encode_mov_to_rsp_slot(reg: u16) -> [u8; 4] {
    let hw = hw_encoding(reg);
    // REX.W = 0x48; REX.R = 0x04 when hw >= 8 (extended register).
    let rex = 0x48u8 | if hw >= 8 { 0x04 } else { 0x00 };
    // ModR/M: mod=00, reg=<lower 3 bits>, r/m=100 (SIB follows).
    let modrm = ((hw & 0x07) << 3) | 0x04;
    // SIB: base=RSP(100), index=none(100), scale=0.
    let sib = 0x24u8;
    [rex, 0x89, modrm, sib]
}

/// Encode a complete retpoline thunk for the given physical register.
///
/// Returns the 17-byte machine code sequence:
///
/// | Offset | Bytes              | Instruction                            |
/// |--------|--------------------|----------------------------------------|
/// | 0      | `E8 07 00 00 00`   | `call .Lretpoline_call_target` (+12)   |
/// | 5      | `F3 90`            | `pause`                                |
/// | 7      | `0F AE E8`         | `lfence`                               |
/// | 10     | `EB F9`            | `jmp .Lretpoline_capture` (back to 5)  |
/// | 12     | `<mov [rsp], reg>` | 4-byte register-specific encoding      |
/// | 16     | `C3`               | `ret`                                  |
fn encode_retpoline_thunk(reg: u16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(17);

    // call .Lretpoline_call_target (offset 12, relative from IP after call = +7)
    bytes.push(0xE8);
    bytes.extend_from_slice(&7i32.to_le_bytes());

    // .Lretpoline_capture:
    // pause
    bytes.push(0xF3);
    bytes.push(0x90);

    // lfence
    bytes.push(0x0F);
    bytes.push(0xAE);
    bytes.push(0xE8);

    // jmp .Lretpoline_capture (offset 5, IP after jmp = 12, rel = 5-12 = -7)
    bytes.push(0xEB);
    bytes.push((-7i8) as u8);

    // .Lretpoline_call_target:
    // mov [rsp], <reg>
    let mov_bytes = encode_mov_to_rsp_slot(reg);
    bytes.extend_from_slice(&mov_bytes);

    // ret
    bytes.push(0xC3);

    debug_assert_eq!(bytes.len(), 17);
    bytes
}

/// Generate retpoline thunks for all 16 x86-64 general-purpose registers.
///
/// Returns a vector of [`RetpolineThunk`] instances, one per GPR (RAX through
/// R15).  Each contains the pre-encoded machine code for the retpoline
/// speculation barrier pattern.
///
/// The thunks are intended to be emitted as separate symbols in the ELF
/// output so that transformed indirect calls can reference them.
pub fn generate_retpoline_thunks() -> Vec<RetpolineThunk> {
    RETPOLINE_GPRS
        .iter()
        .map(|&reg| {
            let name = retpoline_thunk_name(reg_name_64(reg));
            let encoded_bytes = encode_retpoline_thunk(reg);
            RetpolineThunk {
                name,
                register: reg,
                encoded_bytes,
            }
        })
        .collect()
}

// ===========================================================================
// Retpoline — indirect call/jump transformation
// ===========================================================================

/// Transform indirect calls and jumps in `func` to use retpoline thunks.
///
/// When `enabled` is `true`, every instruction that performs an indirect call
/// (`call *%reg`) or indirect jump (`jmp *%reg`) through a register is
/// replaced with a direct call/jump to the corresponding
/// `__x86_indirect_thunk_<reg>` symbol.
///
/// When `enabled` is `false`, this function is a no-op.
///
/// # Retpoline Replacement
///
/// ```text
/// BEFORE:  call *%rax          (indirect call through RAX)
/// AFTER:   call __x86_indirect_thunk_rax  (direct call to thunk)
/// ```
pub fn transform_indirect_calls(func: &mut MachineFunction, enabled: bool) {
    if !enabled {
        return;
    }

    for block in &mut func.blocks {
        let mut i = 0;
        while i < block.instructions.len() {
            let inst = &block.instructions[i];

            // Detect indirect call/jump: opcode is Call or Jmp and the first
            // operand is a Register (physical register holding the target).
            let is_indirect_call = inst.is_call
                && !inst.operands.is_empty()
                && matches!(inst.operands[0], MachineOperand::Register(_));

            let is_indirect_jmp = inst.is_branch
                && X86Opcode::from_u32(inst.opcode) == Some(X86Opcode::Jmp)
                && !inst.operands.is_empty()
                && matches!(inst.operands[0], MachineOperand::Register(_));

            if is_indirect_call || is_indirect_jmp {
                if let MachineOperand::Register(reg) = inst.operands[0] {
                    // Build the thunk symbol name from the register.
                    let thunk_name = retpoline_thunk_name(reg_name_64(reg));

                    // Determine the replacement opcode (Call for indirect calls,
                    // Jmp for indirect jumps — both become direct references).
                    let opcode = if is_indirect_call {
                        X86Opcode::Call
                    } else {
                        X86Opcode::Jmp
                    };

                    let mut replacement = MachineInstruction::new(opcode.as_u32());
                    replacement
                        .operands
                        .push(MachineOperand::GlobalSymbol(thunk_name));
                    replacement.is_call = is_indirect_call;
                    replacement.is_branch = is_indirect_jmp;
                    replacement.is_terminator = inst.is_terminator;

                    block.instructions[i] = replacement;
                }
            }

            i += 1;
        }
    }
}

// ===========================================================================
// CET / IBT — endbr64 instruction emission
// ===========================================================================

/// Create a single `endbr64` [`MachineInstruction`].
///
/// The returned instruction carries the `Endbr64` opcode and the 4-byte
/// encoding `[F3 0F 1E FA]` in its `encoded_bytes` field.
pub fn emit_endbr64() -> MachineInstruction {
    let mut inst = MachineInstruction::new(X86Opcode::Endbr64.as_u32());
    inst.encoded_bytes = ENDBR64_BYTES.to_vec();
    inst
}

/// Insert `endbr64` instructions into `func` at CET-required locations.
///
/// When `enabled` is `true`, `endbr64` is inserted at:
///
/// 1. **Function entry**: The very first instruction of `blocks[0]`.
/// 2. **Indirect branch targets**: Every block that has more than one
///    predecessor (potential indirect branch target) or is referenced by
///    an indirect jump.
///
/// `endbr64` is a 4-byte NOP on non-CET hardware, so inserting it has
/// negligible performance impact on older processors.
///
/// When `enabled` is `false`, this function is a no-op.
pub fn insert_endbr64(func: &mut MachineFunction, enabled: bool) {
    if !enabled {
        return;
    }

    // Phase 1: Insert at function entry (block 0, position 0).
    if !func.blocks.is_empty() {
        let entry = &func.blocks[0];
        // Avoid duplicate insertion if the first instruction is already endbr64.
        let already_has = !entry.instructions.is_empty()
            && X86Opcode::from_u32(entry.instructions[0].opcode) == Some(X86Opcode::Endbr64);

        if !already_has {
            let endbr = emit_endbr64();
            func.blocks[0].instructions.insert(0, endbr);
        }
    }

    // Phase 2: Insert at indirect branch targets.
    //
    // A block is considered an indirect branch target if it has the
    // `is_indirect_target` flag set.  This flag should be set by the
    // code generator when processing computed gotos, switch tables,
    // `asm goto` targets, or any other indirect control transfer.
    //
    // As a conservative fallback, blocks with multiple predecessors
    // also receive endbr64 — this may over-insert at normal merge
    // points but guarantees no indirect target is left unprotected.
    // The codegen driver (generation.rs) should refine this by setting
    // `is_indirect_target` precisely for blocks reachable via indirect
    // branches so the multi-predecessor heuristic can be removed.
    let block_count = func.blocks.len();
    for idx in 1..block_count {
        let needs_endbr =
            func.blocks[idx].is_indirect_target || func.blocks[idx].predecessors.len() > 1;

        if needs_endbr {
            let already_has = !func.blocks[idx].instructions.is_empty()
                && X86Opcode::from_u32(func.blocks[idx].instructions[0].opcode)
                    == Some(X86Opcode::Endbr64);

            if !already_has {
                let endbr = emit_endbr64();
                func.blocks[idx].instructions.insert(0, endbr);
            }
        }
    }
}

/// Generate the `.note.gnu.property` section content for CET/IBT.
///
/// The returned bytes encode an ELF note with `GNU_PROPERTY_X86_FEATURE_1_AND`
/// signalling that the object requires indirect branch tracking (IBT) and
/// shadow stack (SHSTK) support.
///
/// ## ELF Note Layout (32 bytes)
///
/// | Offset | Size | Field     | Value                              |
/// |--------|------|-----------|------------------------------------|
/// | 0      | 4    | namesz   | 4 (`"GNU\0"`)                      |
/// | 4      | 4    | descsz   | 16                                 |
/// | 8      | 4    | type     | 5 (`NT_GNU_PROPERTY_TYPE_0`)       |
/// | 12     | 4    | name     | `"GNU\0"`                          |
/// | 16     | 4    | pr_type  | 0xC000_0002 (`GNU_PROPERTY_X86_FEATURE_1_AND`) |
/// | 20     | 4    | pr_datasz| 4                                  |
/// | 24     | 4    | pr_data  | 0x0000_0003 (IBT + SHSTK)         |
/// | 28     | 4    | padding  | 0                                  |
pub fn generate_cet_note_section() -> Vec<u8> {
    let mut note = Vec::with_capacity(32);

    // Name size: 4 bytes for "GNU\0".
    note.extend_from_slice(&4u32.to_le_bytes());
    // Description size: 16 bytes (property type + datasz + data + padding).
    note.extend_from_slice(&16u32.to_le_bytes());
    // Note type: NT_GNU_PROPERTY_TYPE_0 = 5.
    note.extend_from_slice(&5u32.to_le_bytes());
    // Name: "GNU\0".
    note.extend_from_slice(b"GNU\0");

    // Property entry: GNU_PROPERTY_X86_FEATURE_1_AND.
    // pr_type = 0xC000_0002.
    note.extend_from_slice(&0xC000_0002u32.to_le_bytes());
    // pr_datasz = 4.
    note.extend_from_slice(&4u32.to_le_bytes());
    // pr_data = 0x3 (GNU_PROPERTY_X86_FEATURE_1_IBT | GNU_PROPERTY_X86_FEATURE_1_SHSTK).
    note.extend_from_slice(&0x0000_0003u32.to_le_bytes());
    // Padding to 8-byte alignment.
    note.extend_from_slice(&0u32.to_le_bytes());

    debug_assert_eq!(note.len(), 32);
    note
}

// ===========================================================================
// Stack Guard Page Probing — raw byte emission
// ===========================================================================

/// Returns `true` if the given frame size requires a stack probe loop.
///
/// Frames exceeding [`PAGE_SIZE`] (4,096 bytes) risk skipping over the
/// guard page, so a probe loop must touch every intervening page before
/// the large allocation.
#[inline]
pub fn needs_stack_probe(frame_size: usize) -> bool {
    frame_size > PAGE_SIZE
}

/// Generate raw machine-code bytes for a stack guard-page probe loop.
///
/// If `frame_size <= PAGE_SIZE`, returns an empty vector (no probe needed).
///
/// For frames larger than one page, the generated loop:
///
/// 1. Saves the current stack pointer in RAX.
/// 2. Computes the target stack pointer (`RSP - frame_size`) in RAX.
/// 3. Iteratively subtracts one page from RSP and touches the memory at
///    `[RSP]` to trigger a guard-page fault if necessary.
/// 4. Sets RSP to the final target value.
///
/// ## Machine Code Layout (28 bytes for frame_size > 4096)
///
/// ```text
/// mov  rax, rsp                    ; 48 89 E0
/// sub  rax, <frame_size>           ; 48 2D <i32>
/// .Lprobe_loop:
/// sub  rsp, 4096                   ; 48 81 EC 00 10 00 00
/// test QWORD PTR [rsp], rsp       ; 48 85 24 24
/// cmp  rsp, rax                    ; 48 39 C4
/// ja   .Lprobe_loop                ; 77 F0
/// mov  rsp, rax                    ; 48 89 C4
/// ```
/// # Integration Requirement
///
/// This function is a standalone probe generator that returns raw bytes.
/// The codegen driver (`generation.rs`) or the x86-64 `emit_prologue()`
/// must call `generate_stack_probe(frame_size)` during prologue emission
/// for any function whose stack frame exceeds 4096 bytes, and prepend the
/// returned bytes before the main `sub rsp, <frame_size>` instruction.
///
/// Typical integration point:
///
/// ```ignore
/// let probe_bytes = generate_stack_probe(mf.frame_size);
/// if !probe_bytes.is_empty() {
///     // Insert probe at the start of the prologue code
///     prologue_code.extend_from_slice(&probe_bytes);
/// }
/// ```
pub fn generate_stack_probe(frame_size: usize) -> Vec<u8> {
    if frame_size <= PAGE_SIZE {
        return Vec::new();
    }

    let mut bytes = Vec::with_capacity(28);

    // mov rax, rsp  (save current stack pointer)
    // Encoding: REX.W(48) MOV r/m64,r64(89) ModR/M(E0 = 11_100_000: mod=11, reg=RSP, r/m=RAX)
    bytes.push(0x48);
    bytes.push(0x89);
    bytes.push(0xE0);

    // sub rax, <frame_size>  (compute target stack pointer)
    // Encoding: REX.W(48) SUB rAX,imm32(2D) <i32>
    bytes.push(0x48);
    bytes.push(0x2D);
    bytes.extend_from_slice(&(frame_size as u32).to_le_bytes());

    // .Lprobe_loop (offset 9):
    // sub rsp, 4096
    // Encoding: REX.W(48) 81 /5(EC) imm32(00 10 00 00)
    bytes.push(0x48);
    bytes.push(0x81);
    bytes.push(0xEC);
    bytes.extend_from_slice(&PAGE_SIZE.to_le_bytes()[..4]);

    // test QWORD PTR [rsp], rsp  (touch the page — read access triggers guard fault)
    // Encoding: REX.W(48) TEST r/m64,r64(85) ModR/M(24 = 00_100_100) SIB(24 = RSP base)
    bytes.push(0x48);
    bytes.push(0x85);
    bytes.push(0x24);
    bytes.push(0x24);

    // cmp rsp, rax  (have we probed past the target?)
    // Encoding: REX.W(48) CMP r/m64,r64(39) ModR/M(C4 = 11_000_100: mod=11, reg=RAX, r/m=RSP)
    bytes.push(0x48);
    bytes.push(0x39);
    bytes.push(0xC4);

    // ja .Lprobe_loop  (if RSP > RAX, continue probing)
    // Encoding: 77 rel8   (rel8 = offset_9 - (offset_23 + 2) = 9 - 25 = -16 = 0xF0)
    bytes.push(0x77);
    bytes.push(0xF0u8);

    // mov rsp, rax  (set final stack pointer)
    // Encoding: REX.W(48) MOV r/m64,r64(89) ModR/M(C4 = 11_000_100)
    bytes.push(0x48);
    bytes.push(0x89);
    bytes.push(0xC4);

    debug_assert_eq!(bytes.len(), 28);
    bytes
}

// ===========================================================================
// Stack Guard Page Probing — MachineInstruction emission
// ===========================================================================

/// Create [`MachineInstruction`]s for a stack probe sequence.
///
/// Returns an empty vector if `frame_size <= PAGE_SIZE`.  Otherwise returns
/// the probe loop as a sequence of machine instructions that the assembler
/// can encode.
pub fn emit_stack_probe(frame_size: usize) -> Vec<MachineInstruction> {
    if frame_size <= PAGE_SIZE {
        return Vec::new();
    }

    let mut instrs = Vec::new();

    // mov rax, rsp
    let mut mov_rax_rsp = MachineInstruction::new(X86Opcode::Mov.as_u32());
    mov_rax_rsp.operands.push(MachineOperand::Register(RSP));
    mov_rax_rsp.result = Some(MachineOperand::Register(RAX));
    instrs.push(mov_rax_rsp);

    // sub rax, frame_size
    let mut sub_rax = MachineInstruction::new(X86Opcode::Sub.as_u32());
    sub_rax.operands.push(MachineOperand::Register(RAX));
    sub_rax
        .operands
        .push(MachineOperand::Immediate(frame_size as i64));
    sub_rax.result = Some(MachineOperand::Register(RAX));
    instrs.push(sub_rax);

    // .Lprobe_loop:
    // sub rsp, 4096
    let mut sub_rsp = MachineInstruction::new(X86Opcode::Sub.as_u32());
    sub_rsp.operands.push(MachineOperand::Register(RSP));
    sub_rsp
        .operands
        .push(MachineOperand::Immediate(PAGE_SIZE as i64));
    sub_rsp.result = Some(MachineOperand::Register(RSP));
    instrs.push(sub_rsp);

    // test [rsp], rsp  (touch the page)
    let mut test_rsp = MachineInstruction::new(X86Opcode::Test.as_u32());
    test_rsp.operands.push(MachineOperand::Memory {
        base: Some(RSP),
        index: None,
        scale: 1,
        displacement: 0,
    });
    test_rsp.operands.push(MachineOperand::Register(RSP));
    instrs.push(test_rsp);

    // cmp rsp, rax
    let mut cmp_instr = MachineInstruction::new(X86Opcode::Cmp.as_u32());
    cmp_instr.operands.push(MachineOperand::Register(RSP));
    cmp_instr.operands.push(MachineOperand::Register(RAX));
    instrs.push(cmp_instr);

    // ja .Lprobe_loop  (encoded as Ja with a back-edge block label)
    // The actual branch target is resolved during assembly; we represent it
    // as a relative block label pointing back to the sub_rsp instruction.
    let mut ja_instr = MachineInstruction::new(X86Opcode::Ja.as_u32());
    ja_instr.is_branch = true;
    // Operand: self-loop back-reference.  The assembler resolves the label
    // to the correct relative offset.  We use BlockLabel(0) as a sentinel
    // indicating the probe loop header within this inline sequence.
    ja_instr.operands.push(MachineOperand::BlockLabel(0));
    instrs.push(ja_instr);

    // mov rsp, rax  (set final stack pointer)
    let mut mov_rsp_rax = MachineInstruction::new(X86Opcode::Mov.as_u32());
    mov_rsp_rax.operands.push(MachineOperand::Register(RAX));
    mov_rsp_rax.result = Some(MachineOperand::Register(RSP));
    instrs.push(mov_rsp_rax);

    instrs
}

/// Generate a complete function prologue with stack probe.
///
/// The returned instruction sequence:
///
/// 1. `push rbp`
/// 2. `mov rbp, rsp`
/// 3. If `frame_size > PAGE_SIZE`: stack probe loop (touches each page)
/// 4. `sub rsp, frame_size` (only when no probe — the probe loop already
///    sets RSP to the target)
///
/// This integrates with x86-64 codegen prologue generation.
pub fn emit_prologue_with_probe(frame_size: usize) -> Vec<MachineInstruction> {
    let mut instrs = Vec::new();

    // push rbp
    let mut push_rbp = MachineInstruction::new(X86Opcode::Push.as_u32());
    push_rbp.operands.push(MachineOperand::Register(RBP));
    instrs.push(push_rbp);

    // mov rbp, rsp
    let mut mov_rbp_rsp = MachineInstruction::new(X86Opcode::Mov.as_u32());
    mov_rbp_rsp.operands.push(MachineOperand::Register(RSP));
    mov_rbp_rsp.result = Some(MachineOperand::Register(RBP));
    instrs.push(mov_rbp_rsp);

    if frame_size == 0 {
        // No stack allocation needed.
        return instrs;
    }

    if needs_stack_probe(frame_size) {
        // Emit the probe loop — this sets RSP to the final target.
        let probe_instrs = emit_stack_probe(frame_size);
        instrs.extend(probe_instrs);
    } else {
        // Simple allocation: sub rsp, frame_size.
        let mut sub_rsp = MachineInstruction::new(X86Opcode::Sub.as_u32());
        sub_rsp.operands.push(MachineOperand::Register(RSP));
        sub_rsp
            .operands
            .push(MachineOperand::Immediate(frame_size as i64));
        sub_rsp.result = Some(MachineOperand::Register(RSP));
        instrs.push(sub_rsp);
    }

    instrs
}

// ===========================================================================
// Orchestrator — apply all security mitigations
// ===========================================================================

/// Apply all enabled security mitigations to a machine function.
///
/// This is the single entry point called from `src/backend/generation.rs`
/// after instruction selection and register allocation.  It dispatches to
/// each mitigation pass based on `config`:
///
/// 1. **Retpoline**: transforms indirect calls/jumps to thunk references.
/// 2. **CET/IBT**: inserts `endbr64` at function entries and indirect
///    branch targets.
/// 3. **Stack probing**: handled during prologue generation (not applied
///    here — the frame size is available to `emit_prologue_with_probe`).
///
/// Stack probing is intentionally omitted from this orchestrator because
/// it must be integrated into prologue generation, which occurs earlier
/// in the pipeline.  The `config.stack_probe` flag is checked by the
/// prologue emitter directly via [`needs_stack_probe`].
pub fn apply_security_mitigations(func: &mut MachineFunction, config: &SecurityConfig) {
    // Retpoline: transform indirect calls/jumps to thunk calls.
    transform_indirect_calls(func, config.retpoline);

    // CET/IBT: insert endbr64 at entry points and indirect branch targets.
    insert_endbr64(func, config.cf_protection);

    // Stack probing is handled during prologue emission — the frame_size is
    // checked in emit_prologue_with_probe which is called by the codegen.
    // We record that probing was requested for diagnostic purposes.
    if config.stack_probe && needs_stack_probe(func.frame_size) {
        // The actual probe is emitted by the prologue generator.  We can
        // verify here that the function's frame_size exceeds the threshold.
        // No modification to func.blocks is needed here — the prologue
        // insertion is done by the architecture codegen module.
    }
}

// ===========================================================================
// Validation Helpers — Checkpoint 5 test support
// ===========================================================================

/// Check if the given machine code contains a retpoline thunk pattern.
///
/// Scans `bytes` for the characteristic retpoline signature:
/// `E8 07 00 00 00` (call +7) followed by `F3 90` (pause) and
/// `0F AE E8` (lfence).
///
/// Used by Checkpoint 5 validation tests to confirm retpoline thunks are
/// present in the generated output.
pub fn has_retpoline_thunk(bytes: &[u8]) -> bool {
    if bytes.len() < 17 {
        return false;
    }

    // Retpoline signature: call +7, pause, lfence, jmp -7.
    let call_pattern: [u8; 5] = [0xE8, 0x07, 0x00, 0x00, 0x00];
    let pause_pattern: [u8; 2] = [0xF3, 0x90];
    let lfence_pattern: [u8; 3] = [0x0F, 0xAE, 0xE8];

    // Search for the call pattern anywhere in the byte stream.
    for window_start in 0..=(bytes.len().saturating_sub(17)) {
        let window = &bytes[window_start..];
        if window.len() < 17 {
            break;
        }
        if window[0..5] == call_pattern
            && window[5..7] == pause_pattern
            && window[7..10] == lfence_pattern
        {
            return true;
        }
    }

    false
}

/// Check if the given machine code starts with an `endbr64` instruction.
///
/// Returns `true` if the first 4 bytes are `[F3 0F 1E FA]`.
///
/// Used by Checkpoint 5 validation tests to confirm CET/IBT `endbr64`
/// markers are present at function entries.
pub fn has_endbr64(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0..4] == ENDBR64_BYTES
}

/// Check if the given machine code contains a stack probe loop pattern.
///
/// Scans for the characteristic probe signature: `sub rsp, 4096`
/// (`48 81 EC 00 10 00 00`) followed by a memory touch and backward jump.
///
/// Used by Checkpoint 5 validation tests to confirm stack probing is
/// present for functions with large stack frames.
pub fn has_stack_probe(bytes: &[u8]) -> bool {
    // Signature: sub rsp, 4096 = 48 81 EC 00 10 00 00 (7 bytes).
    let sub_rsp_page: [u8; 7] = [0x48, 0x81, 0xEC, 0x00, 0x10, 0x00, 0x00];

    // Also look for the probe loop setup: mov rax, rsp = 48 89 E0.
    let mov_rax_rsp: [u8; 3] = [0x48, 0x89, 0xE0];

    let has_sub = bytes.windows(7).any(|w| w == sub_rsp_page);

    let has_mov = bytes.windows(3).any(|w| w == mov_rax_rsp);

    // Both the setup (mov rax, rsp) and the page-stepping (sub rsp, 4096)
    // must be present for a valid probe loop.
    has_sub && has_mov
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::MachineBasicBlock;
    use crate::common::target::Target;

    // -- SecurityConfig ---------------------------------------------------

    #[test]
    fn security_config_none_has_all_false() {
        let cfg = SecurityConfig::none();
        assert!(!cfg.retpoline);
        assert!(!cfg.cf_protection);
        assert!(!cfg.stack_probe);
        assert!(!cfg.any_active());
    }

    #[test]
    fn security_config_all_has_all_true() {
        let cfg = SecurityConfig::all();
        assert!(cfg.retpoline);
        assert!(cfg.cf_protection);
        assert!(cfg.stack_probe);
        assert!(cfg.any_active());
    }

    #[test]
    fn security_config_default_has_stack_probe() {
        let cfg = SecurityConfig::default();
        assert!(!cfg.retpoline);
        assert!(!cfg.cf_protection);
        assert!(cfg.stack_probe);
        assert!(cfg.any_active());
    }

    // -- Retpoline thunk name generation ----------------------------------

    #[test]
    fn retpoline_thunk_name_formats_correctly() {
        assert_eq!(retpoline_thunk_name("rax"), "__x86_indirect_thunk_rax");
        assert_eq!(retpoline_thunk_name("r11"), "__x86_indirect_thunk_r11");
        assert_eq!(retpoline_thunk_name("rbx"), "__x86_indirect_thunk_rbx");
    }

    // -- Retpoline thunk generation --------------------------------------

    #[test]
    fn generate_retpoline_thunks_produces_16_thunks() {
        let thunks = generate_retpoline_thunks();
        assert_eq!(thunks.len(), 16);
    }

    #[test]
    fn retpoline_thunks_have_correct_names() {
        let thunks = generate_retpoline_thunks();
        let names: Vec<&str> = thunks.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"__x86_indirect_thunk_rax"));
        assert!(names.contains(&"__x86_indirect_thunk_rcx"));
        assert!(names.contains(&"__x86_indirect_thunk_rdx"));
        assert!(names.contains(&"__x86_indirect_thunk_rbx"));
        assert!(names.contains(&"__x86_indirect_thunk_rsp"));
        assert!(names.contains(&"__x86_indirect_thunk_rbp"));
        assert!(names.contains(&"__x86_indirect_thunk_rsi"));
        assert!(names.contains(&"__x86_indirect_thunk_rdi"));
        assert!(names.contains(&"__x86_indirect_thunk_r8"));
        assert!(names.contains(&"__x86_indirect_thunk_r9"));
        assert!(names.contains(&"__x86_indirect_thunk_r10"));
        assert!(names.contains(&"__x86_indirect_thunk_r11"));
        assert!(names.contains(&"__x86_indirect_thunk_r12"));
        assert!(names.contains(&"__x86_indirect_thunk_r13"));
        assert!(names.contains(&"__x86_indirect_thunk_r14"));
        assert!(names.contains(&"__x86_indirect_thunk_r15"));
    }

    #[test]
    fn retpoline_thunk_bytes_are_17_bytes_each() {
        let thunks = generate_retpoline_thunks();
        for thunk in &thunks {
            assert_eq!(
                thunk.encoded_bytes.len(),
                17,
                "Thunk {} has {} bytes (expected 17)",
                thunk.name,
                thunk.encoded_bytes.len()
            );
        }
    }

    #[test]
    fn retpoline_thunk_rax_encoding_correct() {
        let thunks = generate_retpoline_thunks();
        let rax_thunk = thunks.iter().find(|t| t.register == RAX).unwrap();

        // call +7
        assert_eq!(
            rax_thunk.encoded_bytes[0..5],
            [0xE8, 0x07, 0x00, 0x00, 0x00]
        );
        // pause
        assert_eq!(rax_thunk.encoded_bytes[5..7], [0xF3, 0x90]);
        // lfence
        assert_eq!(rax_thunk.encoded_bytes[7..10], [0x0F, 0xAE, 0xE8]);
        // jmp -7
        assert_eq!(rax_thunk.encoded_bytes[10..12], [0xEB, 0xF9]);
        // mov [rsp], rax = 48 89 04 24
        assert_eq!(rax_thunk.encoded_bytes[12..16], [0x48, 0x89, 0x04, 0x24]);
        // ret
        assert_eq!(rax_thunk.encoded_bytes[16], 0xC3);
    }

    #[test]
    fn retpoline_thunk_r11_encoding_uses_rex_r() {
        let thunks = generate_retpoline_thunks();
        let r11_thunk = thunks.iter().find(|t| t.register == R11).unwrap();

        // mov [rsp], r11: REX = 0x4C (REX.W + REX.R), 89, ModR/M, SIB
        // hw_encoding(R11) = 11. Lower 3 bits = 3. REX.R set.
        // ModR/M = (3 << 3) | 4 = 0x1C.
        assert_eq!(r11_thunk.encoded_bytes[12..16], [0x4C, 0x89, 0x1C, 0x24]);
    }

    #[test]
    fn retpoline_thunk_has_valid_pattern() {
        let thunks = generate_retpoline_thunks();
        for thunk in &thunks {
            assert!(
                has_retpoline_thunk(&thunk.encoded_bytes),
                "Thunk {} not detected by has_retpoline_thunk",
                thunk.name
            );
        }
    }

    // -- Transform indirect calls ----------------------------------------

    #[test]
    fn transform_indirect_calls_disabled_is_noop() {
        let mut func = MachineFunction::new("test_func".to_string());
        let mut inst = MachineInstruction::new(X86Opcode::Call.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        inst.is_call = true;
        func.blocks[0].instructions.push(inst);

        transform_indirect_calls(&mut func, false);

        // Should remain as register call.
        assert!(matches!(
            func.blocks[0].instructions[0].operands[0],
            MachineOperand::Register(RAX)
        ));
    }

    #[test]
    fn transform_indirect_calls_replaces_call_reg_with_thunk() {
        let mut func = MachineFunction::new("test_func".to_string());
        let mut inst = MachineInstruction::new(X86Opcode::Call.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        inst.is_call = true;
        func.blocks[0].instructions.push(inst);

        transform_indirect_calls(&mut func, true);

        // Should now reference the thunk symbol.
        match &func.blocks[0].instructions[0].operands[0] {
            MachineOperand::GlobalSymbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rax");
            }
            other => panic!("Expected GlobalSymbol, got {:?}", other),
        }
    }

    #[test]
    fn transform_indirect_jmp_replaces_with_thunk() {
        let mut func = MachineFunction::new("test_func".to_string());
        let mut inst = MachineInstruction::new(X86Opcode::Jmp.as_u32());
        inst.operands.push(MachineOperand::Register(R11));
        inst.is_branch = true;
        inst.is_terminator = true;
        func.blocks[0].instructions.push(inst);

        transform_indirect_calls(&mut func, true);

        match &func.blocks[0].instructions[0].operands[0] {
            MachineOperand::GlobalSymbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_r11");
            }
            other => panic!("Expected GlobalSymbol, got {:?}", other),
        }
        assert!(func.blocks[0].instructions[0].is_terminator);
    }

    #[test]
    fn transform_does_not_touch_direct_calls() {
        let mut func = MachineFunction::new("test_func".to_string());
        let mut inst = MachineInstruction::new(X86Opcode::Call.as_u32());
        inst.operands
            .push(MachineOperand::GlobalSymbol("printf".to_string()));
        inst.is_call = true;
        func.blocks[0].instructions.push(inst);

        transform_indirect_calls(&mut func, true);

        // Direct call to symbol should remain unchanged.
        match &func.blocks[0].instructions[0].operands[0] {
            MachineOperand::GlobalSymbol(name) => {
                assert_eq!(name, "printf");
            }
            other => panic!("Expected GlobalSymbol(printf), got {:?}", other),
        }
    }

    // -- CET / IBT -------------------------------------------------------

    #[test]
    fn emit_endbr64_creates_correct_instruction() {
        let inst = emit_endbr64();
        assert_eq!(X86Opcode::from_u32(inst.opcode), Some(X86Opcode::Endbr64));
        assert_eq!(inst.encoded_bytes, ENDBR64_BYTES);
    }

    #[test]
    fn insert_endbr64_disabled_is_noop() {
        let mut func = MachineFunction::new("test_func".to_string());
        let nop = MachineInstruction::new(X86Opcode::Nop.as_u32());
        func.blocks[0].instructions.push(nop);

        insert_endbr64(&mut func, false);

        assert_eq!(func.blocks[0].instructions.len(), 1);
        assert_eq!(
            X86Opcode::from_u32(func.blocks[0].instructions[0].opcode),
            Some(X86Opcode::Nop)
        );
    }

    #[test]
    fn insert_endbr64_adds_at_function_entry() {
        let mut func = MachineFunction::new("test_func".to_string());
        let nop = MachineInstruction::new(X86Opcode::Nop.as_u32());
        func.blocks[0].instructions.push(nop);

        insert_endbr64(&mut func, true);

        assert_eq!(func.blocks[0].instructions.len(), 2);
        assert_eq!(
            X86Opcode::from_u32(func.blocks[0].instructions[0].opcode),
            Some(X86Opcode::Endbr64)
        );
    }

    #[test]
    fn insert_endbr64_no_duplicate() {
        let mut func = MachineFunction::new("test_func".to_string());
        let endbr = emit_endbr64();
        func.blocks[0].instructions.push(endbr);

        insert_endbr64(&mut func, true);

        // Should NOT add another endbr64.
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn insert_endbr64_at_indirect_targets() {
        let mut func = MachineFunction::new("test_func".to_string());

        // Entry block with a nop.
        let nop = MachineInstruction::new(X86Opcode::Nop.as_u32());
        func.blocks[0].instructions.push(nop);

        // Second block with two predecessors (indirect branch target).
        let mut bb1 = MachineBasicBlock::with_label("bb1".to_string());
        bb1.predecessors.push(0);
        bb1.predecessors.push(2);
        let ret = MachineInstruction::new(X86Opcode::Ret.as_u32());
        bb1.instructions.push(ret);
        func.blocks.push(bb1);

        // Third block with one predecessor (not an indirect target).
        let mut bb2 = MachineBasicBlock::with_label("bb2".to_string());
        bb2.predecessors.push(0);
        let ret2 = MachineInstruction::new(X86Opcode::Ret.as_u32());
        bb2.instructions.push(ret2);
        func.blocks.push(bb2);

        insert_endbr64(&mut func, true);

        // Entry block: endbr64 inserted at position 0.
        assert_eq!(
            X86Opcode::from_u32(func.blocks[0].instructions[0].opcode),
            Some(X86Opcode::Endbr64)
        );

        // bb1 (2 predecessors): endbr64 inserted.
        assert_eq!(
            X86Opcode::from_u32(func.blocks[1].instructions[0].opcode),
            Some(X86Opcode::Endbr64)
        );

        // bb2 (1 predecessor): NO endbr64.
        assert_ne!(
            X86Opcode::from_u32(func.blocks[2].instructions[0].opcode),
            Some(X86Opcode::Endbr64)
        );
    }

    // -- CET note section -------------------------------------------------

    #[test]
    fn cet_note_section_is_32_bytes() {
        let note = generate_cet_note_section();
        assert_eq!(note.len(), 32);
    }

    #[test]
    fn cet_note_section_has_correct_header() {
        let note = generate_cet_note_section();
        // namesz = 4
        assert_eq!(&note[0..4], &4u32.to_le_bytes());
        // descsz = 16
        assert_eq!(&note[4..8], &16u32.to_le_bytes());
        // type = NT_GNU_PROPERTY_TYPE_0 = 5
        assert_eq!(&note[8..12], &5u32.to_le_bytes());
        // name = "GNU\0"
        assert_eq!(&note[12..16], b"GNU\0");
    }

    #[test]
    fn cet_note_section_has_correct_properties() {
        let note = generate_cet_note_section();
        // pr_type = GNU_PROPERTY_X86_FEATURE_1_AND = 0xC000_0002
        assert_eq!(&note[16..20], &0xC000_0002u32.to_le_bytes());
        // pr_datasz = 4
        assert_eq!(&note[20..24], &4u32.to_le_bytes());
        // pr_data = IBT + SHSTK = 0x3
        assert_eq!(&note[24..28], &3u32.to_le_bytes());
        // padding = 0
        assert_eq!(&note[28..32], &0u32.to_le_bytes());
    }

    // -- Stack probe -----------------------------------------------------

    #[test]
    fn needs_stack_probe_small_frame() {
        assert!(!needs_stack_probe(0));
        assert!(!needs_stack_probe(128));
        assert!(!needs_stack_probe(4096));
    }

    #[test]
    fn needs_stack_probe_large_frame() {
        assert!(needs_stack_probe(4097));
        assert!(needs_stack_probe(8192));
        assert!(needs_stack_probe(1_000_000));
    }

    #[test]
    fn generate_stack_probe_empty_for_small_frame() {
        assert!(generate_stack_probe(0).is_empty());
        assert!(generate_stack_probe(4096).is_empty());
    }

    #[test]
    fn generate_stack_probe_28_bytes_for_large_frame() {
        let probe = generate_stack_probe(8192);
        assert_eq!(probe.len(), 28);
    }

    #[test]
    fn generate_stack_probe_starts_with_mov_rax_rsp() {
        let probe = generate_stack_probe(8192);
        // mov rax, rsp: 48 89 E0
        assert_eq!(&probe[0..3], &[0x48, 0x89, 0xE0]);
    }

    #[test]
    fn generate_stack_probe_has_sub_rax_framesize() {
        let probe = generate_stack_probe(8192);
        // sub rax, 8192: 48 2D 00 20 00 00
        assert_eq!(&probe[3..5], &[0x48, 0x2D]);
        let frame_bytes = u32::from_le_bytes([probe[5], probe[6], probe[7], probe[8]]);
        assert_eq!(frame_bytes, 8192);
    }

    #[test]
    fn generate_stack_probe_has_sub_rsp_page() {
        let probe = generate_stack_probe(8192);
        // sub rsp, 4096: 48 81 EC 00 10 00 00
        assert_eq!(&probe[9..16], &[0x48, 0x81, 0xEC, 0x00, 0x10, 0x00, 0x00]);
    }

    #[test]
    fn generate_stack_probe_has_test_touch() {
        let probe = generate_stack_probe(8192);
        // test [rsp], rsp: 48 85 24 24
        assert_eq!(&probe[16..20], &[0x48, 0x85, 0x24, 0x24]);
    }

    #[test]
    fn generate_stack_probe_has_backward_jump() {
        let probe = generate_stack_probe(8192);
        // ja rel8: 77 F0 (jump back 16 bytes)
        assert_eq!(&probe[23..25], &[0x77, 0xF0]);
    }

    #[test]
    fn generate_stack_probe_ends_with_mov_rsp_rax() {
        let probe = generate_stack_probe(8192);
        // mov rsp, rax: 48 89 C4
        assert_eq!(&probe[25..28], &[0x48, 0x89, 0xC4]);
    }

    #[test]
    fn has_stack_probe_detects_probe_pattern() {
        let probe = generate_stack_probe(8192);
        assert!(has_stack_probe(&probe));
    }

    #[test]
    fn has_stack_probe_rejects_non_probe() {
        assert!(!has_stack_probe(&[0x90, 0x90, 0x90])); // NOP sled
        assert!(!has_stack_probe(&[]));
    }

    // -- emit_stack_probe (MachineInstruction) ----------------------------

    #[test]
    fn emit_stack_probe_empty_for_small_frame() {
        assert!(emit_stack_probe(0).is_empty());
        assert!(emit_stack_probe(4096).is_empty());
    }

    #[test]
    fn emit_stack_probe_produces_7_instructions_for_large_frame() {
        let instrs = emit_stack_probe(8192);
        assert_eq!(instrs.len(), 7);
    }

    // -- emit_prologue_with_probe ----------------------------------------

    #[test]
    fn prologue_no_frame_has_push_mov_only() {
        let instrs = emit_prologue_with_probe(0);
        assert_eq!(instrs.len(), 2);
        assert_eq!(X86Opcode::from_u32(instrs[0].opcode), Some(X86Opcode::Push));
        assert_eq!(X86Opcode::from_u32(instrs[1].opcode), Some(X86Opcode::Mov));
    }

    #[test]
    fn prologue_small_frame_has_sub() {
        let instrs = emit_prologue_with_probe(256);
        // push rbp + mov rbp,rsp + sub rsp,256
        assert_eq!(instrs.len(), 3);
        assert_eq!(X86Opcode::from_u32(instrs[2].opcode), Some(X86Opcode::Sub));
    }

    #[test]
    fn prologue_large_frame_has_probe() {
        let instrs = emit_prologue_with_probe(8192);
        // push rbp + mov rbp,rsp + 7 probe instructions = 9
        assert_eq!(instrs.len(), 9);
    }

    // -- has_endbr64 validation helper -----------------------------------

    #[test]
    fn has_endbr64_positive() {
        let bytes = [0xF3, 0x0F, 0x1E, 0xFA, 0x90];
        assert!(has_endbr64(&bytes));
    }

    #[test]
    fn has_endbr64_negative() {
        let bytes = [0x90, 0xF3, 0x0F, 0x1E, 0xFA];
        assert!(!has_endbr64(&bytes));
    }

    #[test]
    fn has_endbr64_too_short() {
        assert!(!has_endbr64(&[0xF3, 0x0F, 0x1E]));
        assert!(!has_endbr64(&[]));
    }

    // -- apply_security_mitigations --------------------------------------

    #[test]
    fn apply_mitigations_retpoline_transforms_calls() {
        let mut func = MachineFunction::new("test".to_string());
        let mut call = MachineInstruction::new(X86Opcode::Call.as_u32());
        call.operands.push(MachineOperand::Register(RCX));
        call.is_call = true;
        func.blocks[0].instructions.push(call);

        let config = SecurityConfig {
            retpoline: true,
            cf_protection: false,
            stack_probe: false,
        };
        apply_security_mitigations(&mut func, &config);

        match &func.blocks[0].instructions[0].operands[0] {
            MachineOperand::GlobalSymbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rcx");
            }
            other => panic!("Expected GlobalSymbol, got {:?}", other),
        }
    }

    #[test]
    fn apply_mitigations_cet_adds_endbr64() {
        let mut func = MachineFunction::new("test".to_string());
        let nop = MachineInstruction::new(X86Opcode::Nop.as_u32());
        func.blocks[0].instructions.push(nop);

        let config = SecurityConfig {
            retpoline: false,
            cf_protection: true,
            stack_probe: false,
        };
        apply_security_mitigations(&mut func, &config);

        assert_eq!(
            X86Opcode::from_u32(func.blocks[0].instructions[0].opcode),
            Some(X86Opcode::Endbr64)
        );
    }

    #[test]
    fn apply_mitigations_both_retpoline_and_cet() {
        let mut func = MachineFunction::new("test".to_string());
        let mut call = MachineInstruction::new(X86Opcode::Call.as_u32());
        call.operands.push(MachineOperand::Register(RDI));
        call.is_call = true;
        func.blocks[0].instructions.push(call);

        let config = SecurityConfig {
            retpoline: true,
            cf_protection: true,
            stack_probe: false,
        };
        apply_security_mitigations(&mut func, &config);

        // First instruction should be endbr64 (CET).
        assert_eq!(
            X86Opcode::from_u32(func.blocks[0].instructions[0].opcode),
            Some(X86Opcode::Endbr64)
        );

        // Second instruction should be the transformed call to thunk.
        match &func.blocks[0].instructions[1].operands[0] {
            MachineOperand::GlobalSymbol(name) => {
                assert_eq!(name, "__x86_indirect_thunk_rdi");
            }
            other => panic!("Expected GlobalSymbol, got {:?}", other),
        }
    }

    // -- Target page_size consistency ------------------------------------

    #[test]
    fn page_size_matches_target() {
        assert_eq!(PAGE_SIZE, Target::X86_64.page_size());
    }

    // -- ENDBR64 constant -----------------------------------------------

    #[test]
    fn endbr64_bytes_correct() {
        assert_eq!(ENDBR64_BYTES, [0xF3, 0x0F, 0x1E, 0xFA]);
    }
}
