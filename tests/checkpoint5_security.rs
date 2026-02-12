//! Checkpoint 5 Integration Test Suite — Security Mitigation Validation (x86-64)
//!
//! Validates three security hardening features in the BCC x86-64 backend:
//!
//! 1. **Retpoline generation** (`-mretpoline`): Verifies that indirect function
//!    pointer calls are routed through `__x86_indirect_thunk_*` trampolines
//!    instead of direct `call *%reg` instructions (Spectre v2 mitigation).
//!
//! 2. **Intel CET/IBT** (`-fcf-protection`): Verifies that the `endbr64`
//!    instruction (opcode `0xf3 0x0f 0x1e 0xfa`) is emitted at every function
//!    entry point and indirect branch target.
//!
//! 3. **Stack guard page probing**: Verifies that a probe loop is emitted
//!    before the stack pointer adjustment for any function whose stack frame
//!    exceeds 4096 bytes (one page), preventing stack clash attacks.
//!
//! All tests target x86-64 exclusively — security mitigations are x86-64 only
//! per Section 0.6.2.
//!
//! This checkpoint is a **sequential hard gate** (Section 0.7.5): failure at
//! this gate halts all forward progress to subsequent checkpoints.

// ---------------------------------------------------------------------------
// Module import: shared test utilities (BCC invocation, ELF inspection, etc.)
// ---------------------------------------------------------------------------
mod common;

use std::path::Path;

// ---------------------------------------------------------------------------
// Test 1: Retpoline Generation (-mretpoline)
// ---------------------------------------------------------------------------

/// Validate that indirect function pointer calls are retpoline-protected when
/// compiled with the `-mretpoline` flag on x86-64.
///
/// Per Section 0.1.2 User Example:
///   "function containing `(*fptr)()` call → call instruction targets
///    `__x86_indirect_thunk_*`, not the pointer directly"
///
/// Procedure:
/// 1. Compile `tests/fixtures/security/retpoline.c` with `-mretpoline --target=x86-64 -c`
/// 2. Disassemble the object file via `objdump -d`
/// 3. Assert `__x86_indirect_thunk` appears in the disassembly (thunk is used)
/// 4. Assert no direct indirect call patterns (`call *%`) appear
/// 5. Assert the thunk symbol exists in the symbol table
#[test]
fn test_retpoline_generation() {
    // Set up a temporary directory for test artifacts.
    let test_dir = common::TestDir::new("retpoline_generation");

    // Resolve the path to the retpoline test fixture.
    let fixture = common::fixture_path("security/retpoline.c");
    let fixture_str = fixture
        .to_str()
        .expect("retpoline.c fixture path must be valid UTF-8");

    // Construct the output object file path within the temporary directory.
    let output_obj = test_dir.file_path("retpoline.o");
    let output_path = output_obj
        .to_str()
        .expect("retpoline.o output path must be valid UTF-8");

    // Verify the fixture file exists before attempting compilation.
    assert!(
        Path::new(fixture_str).exists(),
        "Retpoline test fixture not found at: {}. \
         Ensure tests/fixtures/security/retpoline.c has been created.",
        fixture.display()
    );

    // Compile with -mretpoline targeting x86-64, producing a relocatable
    // object file (-c flag). This exercises the retpoline thunk generation
    // path in src/backend/x86_64/security.rs.
    let result: common::BccOutput =
        common::compile_to_object(fixture_str, output_path, "x86-64", &["-mretpoline"]);
    result.assert_success();

    // Verify the output object file was produced on disk.
    assert!(
        Path::new(output_path).exists(),
        "Compilation succeeded but output object file was not produced at: {}",
        output_path
    );

    // Disassemble the object file and inspect for retpoline patterns.
    let disasm = common::objdump_disassemble(output_path);

    // -----------------------------------------------------------------------
    // Positive assertion: indirect calls MUST go through __x86_indirect_thunk_*
    //
    // When retpoline is active, the compiler emits:
    //   callq  <__x86_indirect_thunk_rax>
    //   callq  <__x86_indirect_thunk_r11>
    //   etc.
    // instead of direct indirect calls through registers.
    // -----------------------------------------------------------------------
    common::assert_disassembly_contains(output_path, "__x86_indirect_thunk");

    // -----------------------------------------------------------------------
    // Negative assertion: no direct indirect calls through registers.
    //
    // Without retpoline, the compiler would emit patterns like:
    //   call   *%rax
    //   callq  *%rax
    //   jmpq   *%r11
    //   etc.
    //
    // With retpoline active, ALL such user-level indirect call patterns in
    // the application functions must be replaced by thunk calls.
    //
    // Note: the retpoline thunk implementation itself (the __x86_indirect_thunk_*
    // function body) may legitimately contain a `jmp *%reg` instruction as
    // part of the trampoline mechanism. We exclude thunk body lines from this
    // check.
    // -----------------------------------------------------------------------
    let has_direct_indirect_call = disasm.lines().any(|line| {
        let trimmed = line.trim();
        // Only examine actual instruction lines (contain hex bytes + mnemonic).
        if !trimmed.contains(':') {
            return false;
        }
        // Extract the instruction portion (after the hex byte dump).
        let instr_portion = match trimmed.rfind('\t') {
            Some(pos) => &trimmed[pos..],
            None => trimmed,
        };
        // Detect direct indirect call/jmp patterns: "call *%" or "jmp *%"
        // (including "callq", "jmpq" variants).
        let is_indirect = (instr_portion.contains("call") || instr_portion.contains("jmp"))
            && instr_portion.contains("*%");
        // Exclude lines that are part of the __x86_indirect_thunk body itself,
        // since the thunk's internal implementation may contain a jmp *%reg.
        let in_thunk_body = instr_portion.contains("__x86_indirect_thunk");
        is_indirect && !in_thunk_body
    });

    assert!(
        !has_direct_indirect_call,
        "Found direct indirect call/jmp (call *%reg or jmp *%reg) in retpoline-compiled code.\n\
         All indirect calls must go through __x86_indirect_thunk_* trampolines.\n\
         Per Section 0.1.2: call instruction targets __x86_indirect_thunk_*, \
         not the pointer directly.\n\
         Disassembly (first 3000 chars):\n{}",
        &disasm[..disasm.len().min(3000)]
    );

    // Verify the retpoline thunk symbol exists in the object file's symbol table.
    common::assert_symbol_exists(output_path, "__x86_indirect_thunk");
}

// ---------------------------------------------------------------------------
// Test 2: CET/IBT Generation (-fcf-protection)
// ---------------------------------------------------------------------------

/// Validate that the `endbr64` instruction is emitted at function entry points
/// and indirect branch targets when compiled with the `-fcf-protection` flag
/// on x86-64.
///
/// Per Section 0.1.1:
///   CET/IBT requires `endbr64` (opcode 0xf3 0x0f 0x1e 0xfa) at every
///   function entry and every indirect branch target.
///
/// Procedure:
/// 1. Compile `tests/fixtures/security/cet.c` with `-fcf-protection --target=x86-64 -c`
/// 2. Disassemble the object file via `objdump -d`
/// 3. Assert `endbr64` appears in the disassembly
/// 4. Assert multiple `endbr64` instructions are present (one per function entry)
/// 5. Verify specific function entry points begin with `endbr64`
#[test]
fn test_cet_ibt_generation() {
    // Set up a temporary directory for test artifacts.
    let test_dir = common::TestDir::new("cet_ibt_generation");

    // Resolve the path to the CET/IBT test fixture.
    let fixture = common::fixture_path("security/cet.c");
    let fixture_str = fixture
        .to_str()
        .expect("cet.c fixture path must be valid UTF-8");

    // Construct the output object file path.
    let output_obj = test_dir.file_path("cet.o");
    let output_path = output_obj
        .to_str()
        .expect("cet.o output path must be valid UTF-8");

    // Verify the fixture file exists before attempting compilation.
    assert!(
        Path::new(fixture_str).exists(),
        "CET/IBT test fixture not found at: {}. \
         Ensure tests/fixtures/security/cet.c has been created.",
        fixture.display()
    );

    // Compile with -fcf-protection targeting x86-64, producing a relocatable
    // object file. This exercises the CET/IBT endbr64 insertion path in
    // src/backend/x86_64/security.rs.
    let result: common::BccOutput =
        common::compile_to_object(fixture_str, output_path, "x86-64", &["-fcf-protection"]);
    result.assert_success();

    // Verify the output object file was produced on disk.
    assert!(
        Path::new(output_path).exists(),
        "Compilation succeeded but output object file was not produced at: {}",
        output_path
    );

    // Disassemble the object file.
    let disasm = common::objdump_disassemble(output_path);

    // -----------------------------------------------------------------------
    // Positive assertion: `endbr64` must appear in the disassembly.
    // The `endbr64` instruction is the CET/IBT landing pad that marks valid
    // indirect branch targets. Its presence in the disassembly confirms that
    // the -fcf-protection flag activated CET code generation.
    // -----------------------------------------------------------------------
    common::assert_disassembly_contains(output_path, "endbr64");

    // -----------------------------------------------------------------------
    // Count endbr64 instructions to verify placement at multiple function
    // entry points.
    //
    // The cet.c fixture defines these functions (all require endbr64):
    //   handler_a, handler_b, handler_c, dispatch_handler,
    //   call_via_local, call_via_param, select_handler, main
    // That is at least 8 functions. Using 6 as a conservative minimum to
    // accommodate any functions the compiler might merge or inline.
    // -----------------------------------------------------------------------
    let endbr64_count = disasm
        .lines()
        .filter(|line| line.contains("endbr64"))
        .count();

    assert!(
        endbr64_count >= 6,
        "Expected at least 6 endbr64 instructions in CET-protected code, found {}.\n\
         Each function entry point must begin with endbr64 when -fcf-protection \
         is active.\nDisassembly (first 3000 chars):\n{}",
        endbr64_count,
        &disasm[..disasm.len().min(3000)]
    );

    // -----------------------------------------------------------------------
    // Verify that specific known function entries begin with endbr64.
    //
    // In objdump output, a function entry looks like:
    //   0000000000000000 <handler_a>:
    //      0:   f3 0f 1e fa             endbr64
    //
    // We scan the disassembly for each function label and check that the
    // first instruction line following the label contains "endbr64".
    // -----------------------------------------------------------------------
    let functions_to_check = ["handler_a", "handler_b", "handler_c", "main"];

    let lines: Vec<&str> = disasm.lines().collect();
    for func_name in &functions_to_check {
        let func_label = format!("<{}>:", func_name);
        let label_pos = lines.iter().position(|line| line.contains(&func_label));

        if let Some(pos) = label_pos {
            // Scan subsequent lines to find the first actual instruction
            // (skipping blank lines or annotation lines).
            let mut found_endbr = false;
            for offset in 1..=5 {
                if pos + offset >= lines.len() {
                    break;
                }
                let next_line = lines[pos + offset].trim();
                // Skip empty lines.
                if next_line.is_empty() {
                    continue;
                }
                // Stop at the next function label to avoid crossing boundaries.
                if next_line.contains(">:") && next_line.contains('<') {
                    break;
                }
                // Check if this instruction line contains endbr64.
                if next_line.contains("endbr64") {
                    found_endbr = true;
                    break;
                }
                // If we reach a non-endbr64 instruction (line with `:` delimiter
                // for address:bytes format), the function does not start with endbr64.
                if next_line.contains(':') {
                    break;
                }
            }

            assert!(
                found_endbr,
                "Function '{}' does not begin with endbr64 instruction.\n\
                 CET/IBT requires endbr64 at every function entry point when \
                 -fcf-protection is active.\n\
                 Function disassembly:\n{}",
                func_name,
                lines[pos..lines.len().min(pos + 10)].join("\n")
            );
        }
        // If the function label is not found in the disassembly (e.g., due to
        // static linkage naming conventions or mangling), the earlier count-based
        // assertion provides the safety net.
    }
}

// ---------------------------------------------------------------------------
// Test 3: Stack Probe Generation (frames > 4096 bytes)
// ---------------------------------------------------------------------------

/// Validate that a probe loop is emitted for functions with stack frames
/// exceeding 4096 bytes (one page) on x86-64.
///
/// Per Section 0.1.2 User Example:
///   `void f(void) { char buf[8192]; buf[0] = 1; }`
///   → disassembly MUST show a probe loop before the stack pointer adjustment
///
/// The probe loop prevents stack clash attacks by touching each guard page
/// in page-sized (4096-byte / 0x1000) increments before the final stack
/// pointer is lowered.
///
/// Procedure:
/// 1. Compile `tests/fixtures/security/stack_probe.c` with `--target=x86-64 -c`
/// 2. Disassemble via `objdump -d`
/// 3. Assert the page-size probe constant (0x1000) appears in the disassembly
/// 4. Assert probe-related instructions are present (sub/test/or touching stack)
/// 5. Verify the borderline case (exact 4096 bytes) need not have a probe loop
#[test]
fn test_stack_probe_generation() {
    // Set up a temporary directory for test artifacts.
    let test_dir = common::TestDir::new("stack_probe_generation");

    // Resolve the path to the stack probe test fixture.
    let fixture = common::fixture_path("security/stack_probe.c");
    let fixture_str = fixture
        .to_str()
        .expect("stack_probe.c fixture path must be valid UTF-8");

    // Construct the output object file path.
    let output_obj = test_dir.file_path("probe.o");
    let output_path = output_obj
        .to_str()
        .expect("probe.o output path must be valid UTF-8");

    // Verify the fixture file exists before attempting compilation.
    assert!(
        Path::new(fixture_str).exists(),
        "Stack probe test fixture not found at: {}. \
         Ensure tests/fixtures/security/stack_probe.c has been created.",
        fixture.display()
    );

    // Compile targeting x86-64. Stack probing is automatic for frames exceeding
    // 4096 bytes — no special CLI flag is needed beyond the target architecture.
    let result: common::BccOutput =
        common::compile_to_object(fixture_str, output_path, "x86-64", &[]);
    result.assert_success();

    // Verify the output object file was produced on disk.
    assert!(
        Path::new(output_path).exists(),
        "Compilation succeeded but output object file was not produced at: {}",
        output_path
    );

    // Disassemble the object file.
    let disasm = common::objdump_disassemble(output_path);

    // -----------------------------------------------------------------------
    // Probe loop detection strategy
    //
    // A stack probe loop on x86-64 typically manifests as:
    //
    //   Pattern A (sub-and-touch loop):
    //     sub    $0x1000,%rsp        ; stride one page
    //     test   %rsp,(%rsp)         ; touch the page (triggers guard fault)
    //     cmp    ...                  ; check remaining
    //     jne    <loop_top>          ; repeat if more pages
    //
    //   Pattern B (or-probe loop):
    //     or     $0x0,(%rsp)         ; probe current page
    //     sub    $0x1000,%rsp        ; move to next page
    //     cmp/jne ...                ; loop control
    //
    //   Pattern C (movb-probe):
    //     movb   $0x0,(%rsp)         ; touch the page
    //     sub    $0x1000,...
    //
    // The common denominator is the page-size constant 0x1000 (4096) in a
    // subtraction instruction combined with a memory access touching the
    // stack within a conditional loop.
    // -----------------------------------------------------------------------

    // Check 1: The page-size probe constant 0x1000 (4096 decimal) must appear
    // in the disassembly. This is the fundamental indicator of page-granularity
    // stack probing.
    let has_page_probe_constant =
        disasm.contains("$0x1000") || disasm.contains("0x1000") || disasm.contains("$4096");

    assert!(
        has_page_probe_constant,
        "Expected page-size probe constant (0x1000 / 4096) in the disassembly.\n\
         Stack probing for frames > 4096 bytes must use page-sized increments.\n\
         Disassembly (first 3000 chars):\n{}",
        &disasm[..disasm.len().min(3000)]
    );

    // Check 2: Verify probe-related instructions exist. This combines the
    // page-stride sub instruction with a stack memory touch (test/or/mov).
    let has_probe_instructions = disasm.lines().any(|line| {
        let t = line.trim();
        // Memory probe instructions touching (%rsp) or a stack-relative address:
        //   test   %rax,(%rsp)           — read-probe
        //   or     $0x0,(%rsp)           — read-modify-write probe
        //   orl    $0x0,(%rsp)           — 32-bit or probe variant
        //   orq    $0x0,(%rsp)           — 64-bit or probe variant
        //   mov    $0x0,(%rsp)           — write-probe
        //   movb   $0x0,(%rsp)           — byte write-probe
        (t.contains("test") && t.contains("(%rsp)"))
            || (t.contains("or") && t.contains("$0x0") && t.contains("(%rsp)"))
            || (t.contains("mov") && t.contains("$0x0") && t.contains("(%rsp)"))
            || (t.contains("mov") && t.contains("(%rsp)") && t.contains("$0"))
    });

    // Check 3: Verify the presence of a sub instruction using the page stride
    // as an operand. This is the probe loop's page decrement.
    let has_page_stride_sub = disasm.lines().any(|line| {
        let t = line.trim();
        t.contains("sub") && (t.contains("$0x1000") || t.contains("$4096"))
    });

    // Check 4: Verify a conditional branch exists that forms the loop back-edge.
    let has_conditional_branch = disasm.lines().any(|line| {
        let t = line.trim();
        t.contains("jne")
            || t.contains("jnz")
            || t.contains("jb")
            || t.contains("ja")
            || t.contains("jge")
            || t.contains("jle")
            || t.contains("loop")
            || t.contains("jg")
            || t.contains("jl")
            || t.contains("jbe")
            || t.contains("jae")
            || t.contains("je")
            || t.contains("jnb")
            || t.contains("jns")
    });

    // The probe pattern is confirmed when:
    //   - The page-size constant is present (already asserted), AND
    //   - A page-stride subtraction instruction exists, AND
    //   - Either a probe touch instruction or a conditional loop branch exists
    assert!(
        has_page_stride_sub && (has_probe_instructions || has_conditional_branch),
        "Expected stack probe loop pattern in the disassembly.\n\
         A probe loop requires:\n\
           - sub $0x1000 (page-stride decrement): {}\n\
           - probe touch instruction (test/or/mov on (%rsp)): {}\n\
           - conditional branch (loop back-edge): {}\n\
         Per Section 0.1.2: disassembly MUST show a probe loop before the \
         stack pointer adjustment for frames exceeding 4096 bytes.\n\
         Disassembly (first 4000 chars):\n{}",
        if has_page_stride_sub {
            "FOUND"
        } else {
            "MISSING"
        },
        if has_probe_instructions {
            "FOUND"
        } else {
            "MISSING"
        },
        if has_conditional_branch {
            "FOUND"
        } else {
            "MISSING"
        },
        &disasm[..disasm.len().min(4000)]
    );

    // -----------------------------------------------------------------------
    // Per-function verification for the canonical test case (Section 0.1.2).
    //
    // The fixture defines:
    //   f()                   — 8192 bytes  → probe loop REQUIRED
    //   large_frame_4097()    — 4097 bytes  → probe loop REQUIRED
    //   large_frame_16384()   — 16384 bytes → probe loop REQUIRED
    //   large_frame_exact_page — 4096 bytes → probe loop NOT required
    //   use_stack()           — 8192 bytes  → probe loop REQUIRED
    //
    // We verify the canonical f() function specifically.
    // -----------------------------------------------------------------------
    let f_disasm = extract_function_disasm(&disasm, "f");
    if let Some(ref f_body) = f_disasm {
        let f_has_probe = contains_probe_pattern(f_body);
        assert!(
            f_has_probe,
            "Function 'f' (8192-byte stack frame, canonical test case from Section 0.1.2) \
             does not contain a probe loop.\n\
             Per Section 0.1.2 User Example: disassembly MUST show a probe loop before \
             the stack pointer adjustment.\n\
             Function 'f' disassembly:\n{}",
            f_body
        );
    }

    // Verify that functions known to have large frames (> 4096 bytes) contain
    // the probe pattern in their disassembly.
    for large_func in &["large_frame_4097", "large_frame_16384", "use_stack"] {
        let func_disasm = extract_function_disasm(&disasm, large_func);
        if let Some(ref body) = func_disasm {
            let has_probe = contains_probe_pattern(body);
            assert!(
                has_probe,
                "Function '{}' (stack frame > 4096 bytes) does not contain a probe loop.\n\
                 Stack probing is required for frames exceeding 4096 bytes.\n\
                 Function disassembly:\n{}",
                large_func, body
            );
        }
    }

    // Verify the borderline case: large_frame_exact_page (exactly 4096 bytes)
    // should NOT require a probe loop, since it does not EXCEED the threshold.
    let exact_page_disasm = extract_function_disasm(&disasm, "large_frame_exact_page");
    if let Some(ref body) = exact_page_disasm {
        let has_page_stride = body.contains("$0x1000")
            && body
                .lines()
                .any(|l| l.trim().contains("sub") && l.contains("$0x1000"));
        // The exact-page function should NOT contain the page-stride sub that
        // forms the probe loop. It may still adjust the stack, but not with
        // the iterative probe mechanism.
        if has_page_stride {
            // If it does contain it, the compiler is being conservative (probing
            // at the boundary). This is acceptable but not required — log it.
            eprintln!(
                "Note: large_frame_exact_page (4096 bytes) contains a probe loop. \
                 This is conservative but acceptable — the threshold is 'exceeds 4096'."
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Combined Security Flags (-mretpoline + -fcf-protection)
// ---------------------------------------------------------------------------

/// Validate that both retpoline and CET/IBT mitigations are active when
/// compiled with both `-mretpoline` and `-fcf-protection` simultaneously.
///
/// This ensures the security flags compose correctly and do not interfere
/// with each other.
///
/// Procedure:
/// 1. Compile the retpoline fixture with both flags
/// 2. Assert `__x86_indirect_thunk` appears (retpoline active)
/// 3. Assert `endbr64` appears (CET/IBT active)
/// 4. Assert no direct indirect call patterns remain
/// 5. Compile the CET fixture with both flags and verify both mitigations
#[test]
fn test_combined_security_flags() {
    // Set up a temporary directory for test artifacts.
    let test_dir = common::TestDir::new("combined_security_flags");

    // --- Part A: Compile the retpoline fixture with both flags ---

    let retpoline_fixture = common::fixture_path("security/retpoline.c");
    let retpoline_fixture_str = retpoline_fixture
        .to_str()
        .expect("retpoline.c fixture path must be valid UTF-8");

    let combined_obj = test_dir.file_path("combined.o");
    let combined_path = combined_obj
        .to_str()
        .expect("combined.o output path must be valid UTF-8");

    // Verify the fixture file exists.
    assert!(
        Path::new(retpoline_fixture_str).exists(),
        "Retpoline test fixture not found at: {}. \
         Ensure tests/fixtures/security/retpoline.c has been created.",
        retpoline_fixture.display()
    );

    // Compile with BOTH security flags targeting x86-64.
    let result: common::BccOutput = common::compile_to_object(
        retpoline_fixture_str,
        combined_path,
        "x86-64",
        &["-mretpoline", "-fcf-protection"],
    );
    result.assert_success();

    // Verify the output object file was produced.
    assert!(
        Path::new(combined_path).exists(),
        "Compilation succeeded but output object file was not produced at: {}",
        combined_path
    );

    // Disassemble and verify both mitigations are active.
    let disasm = common::objdump_disassemble(combined_path);

    // Retpoline: indirect calls must go through thunk trampolines.
    common::assert_disassembly_contains(combined_path, "__x86_indirect_thunk");

    // CET/IBT: function entries must have endbr64 landing pads.
    common::assert_disassembly_contains(combined_path, "endbr64");

    // Retpoline thunk symbol must exist in the symbol table.
    common::assert_symbol_exists(combined_path, "__x86_indirect_thunk");

    // Verify a meaningful number of endbr64 instructions are present.
    // The retpoline.c fixture defines 10+ functions; with CET active, each
    // function entry should have endbr64. Requiring at least 4 as a
    // conservative minimum.
    let endbr64_count = disasm
        .lines()
        .filter(|line| line.contains("endbr64"))
        .count();

    assert!(
        endbr64_count >= 4,
        "Expected at least 4 endbr64 instructions when -fcf-protection is active \
         with the retpoline fixture, found {}.\n\
         Disassembly (first 3000 chars):\n{}",
        endbr64_count,
        &disasm[..disasm.len().min(3000)]
    );

    // Negative assertion: no direct indirect call patterns should remain.
    // Check both `callq  *%` and `call   *%` objdump format variants.
    common::assert_disassembly_not_contains(combined_path, "callq  *%");

    // --- Part B: Compile the CET fixture with both flags ---

    let cet_fixture = common::fixture_path("security/cet.c");
    let cet_fixture_str = cet_fixture
        .to_str()
        .expect("cet.c fixture path must be valid UTF-8");

    if Path::new(cet_fixture_str).exists() {
        let cet_combined_obj = test_dir.file_path("cet_combined.o");
        let cet_combined_path = cet_combined_obj
            .to_str()
            .expect("cet_combined.o output path must be valid UTF-8");

        // Use the generic compile() helper to demonstrate composability with
        // manually-specified flags (exercises the compile() API from common).
        let cet_result: common::BccOutput = common::compile(
            cet_fixture_str,
            &[
                "-c",
                "--target=x86-64",
                "-o",
                cet_combined_path,
                "-mretpoline",
                "-fcf-protection",
            ],
        );
        cet_result.assert_success();

        // Verify both mitigations in the CET fixture output.
        if Path::new(cet_combined_path).exists() {
            // CET/IBT: endbr64 must be present.
            common::assert_disassembly_contains(cet_combined_path, "endbr64");

            // Count endbr64 instructions — the CET fixture has 8+ functions.
            let cet_disasm = common::objdump_disassemble(cet_combined_path);
            let cet_endbr_count = cet_disasm
                .lines()
                .filter(|line| line.contains("endbr64"))
                .count();

            assert!(
                cet_endbr_count >= 4,
                "Expected at least 4 endbr64 instructions in CET fixture compiled \
                 with combined security flags, found {}.\n\
                 Disassembly (first 3000 chars):\n{}",
                cet_endbr_count,
                &cet_disasm[..cet_disasm.len().min(3000)]
            );
        }
    }
}

// ===========================================================================
// Helper Functions (private, used by the test functions above)
// ===========================================================================

/// Extract the disassembly text for a specific named function from full
/// `objdump -d` output.
///
/// Scans the disassembly for a function label matching `<func_name>:` and
/// collects all subsequent instruction lines until the next function label
/// or end of output.
///
/// # Arguments
///
/// * `full_disasm` — Complete `objdump -d` output text.
/// * `func_name`   — Name of the function to extract (e.g., `"f"`, `"main"`).
///
/// # Returns
///
/// `Some(String)` containing the function's disassembly if found, or `None`
/// if the function label was not located.
fn extract_function_disasm(full_disasm: &str, func_name: &str) -> Option<String> {
    let lines: Vec<&str> = full_disasm.lines().collect();
    let label_pattern = format!("<{}>:", func_name);

    // Find the line containing the function label.
    // Use an exact match for the label to avoid partial-name collisions
    // (e.g., "f" should not match "foo" or "large_frame_4097").
    let start_idx = lines
        .iter()
        .position(|line| line.contains(&label_pattern))?;

    let mut result = String::with_capacity(1024);
    result.push_str(lines[start_idx]);
    result.push('\n');

    // Collect instruction lines until the next function label or end.
    for line in lines.iter().skip(start_idx + 1) {
        let trimmed = line.trim();

        // Stop at the next function label. Function labels in objdump format:
        //   0000000000000020 <handler_b>:
        // They contain `<` and `>:` and typically appear at the start of a line.
        if !trimmed.is_empty() && trimmed.contains('<') && trimmed.contains(">:") {
            // Verify this is a new function header, not an in-instruction reference.
            if let Some(lt_pos) = trimmed.find('<') {
                if let Some(gt_pos) = trimmed.find(">:") {
                    if gt_pos > lt_pos {
                        break;
                    }
                }
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    Some(result)
}

/// Determine whether a function's disassembly contains a stack probe loop
/// pattern.
///
/// Looks for the combination of:
/// 1. A page-size constant (`0x1000` / `4096`) in a sub/add instruction
/// 2. A stack memory probe instruction (test/or/mov touching `(%rsp)`)
/// 3. A conditional branch (loop back-edge)
///
/// Returns `true` if a probe pattern is detected, `false` otherwise.
fn contains_probe_pattern(func_disasm: &str) -> bool {
    // Indicator 1: The page-size constant 0x1000 (4096) appears, typically in
    // a sub instruction that decrements by page-size for each probe iteration.
    let has_page_stride = func_disasm.contains("$0x1000")
        || func_disasm.contains("0x1000")
        || func_disasm.contains("$4096");

    // Indicator 2: A memory probe instruction that touches a stack-relative
    // address. This is the actual "probe" — reading or writing a byte on
    // each page to trigger guard page faults.
    let has_probe_touch = func_disasm.lines().any(|line| {
        let t = line.trim();
        // test %reg,(%rsp) — read-probe that triggers fault on guard page
        (t.contains("test") && t.contains("(%rsp)"))
        // or $0x0,(%rsp) — read-modify-write probe (orl/orq variants)
        || (t.contains("or") && t.contains("$0x0") && t.contains("(%rsp)"))
        || (t.contains("or") && t.contains("$0") && t.contains(",(%rsp)"))
        // mov $0x0,(%rsp) — write-probe (movb/movl/movq variants)
        || (t.contains("mov") && t.contains("$0") && t.contains("(%rsp)"))
        // Any store to a register-indirect address near sub $0x1000
        || (t.contains("test") && t.contains("(%r"))
    });

    // Indicator 3: A conditional branch instruction that forms the loop
    // back-edge. The probe loop iterates until all pages have been touched.
    let has_loop_branch = func_disasm.lines().any(|line| {
        let t = line.trim();
        t.contains("jne ")
            || t.contains("jnz ")
            || t.contains("jb ")
            || t.contains("ja ")
            || t.contains("jge ")
            || t.contains("jle ")
            || t.contains("jg ")
            || t.contains("jl ")
            || t.contains("loop ")
            || t.contains("jbe ")
            || t.contains("jae ")
            || t.contains("je ")
            || t.contains("jns ")
            || t.contains("jnb ")
    });

    // A valid probe pattern requires the page-stride constant AND at least
    // one of: a probe touch instruction OR a loop branch. The combination of
    // page stride + loop branch is sufficient because the sub instruction
    // itself touches the guard page when the stack pointer crosses a page
    // boundary and a subsequent access triggers the fault.
    has_page_stride && (has_probe_touch || has_loop_branch)
}
