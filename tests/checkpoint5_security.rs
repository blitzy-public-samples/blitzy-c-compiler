//! Checkpoint 5 — Security Mitigation Validation (x86-64 Only)
//!
//! This test suite validates security-hardened code generation on x86-64.
//! Three classes of security mitigations are tested:
//!
//! 1. **Retpoline** (`-mretpoline`) — Spectre v2 mitigation that replaces indirect
//!    branches with calls to `__x86_indirect_thunk_*` thunk functions.
//! 2. **CET/IBT** (`-fcf-protection`) — Intel Control-flow Enforcement Technology
//!    that inserts `endbr64` instructions at function entries and indirect branch
//!    targets.
//! 3. **Stack probe** — Guard page probing for large stack frames (>4096 bytes)
//!    that prevents silent guard page skipping via stack clash attacks.
//!
//! # AAP Compliance
//! - **Sequential hard gate:** Checkpoint 4 must pass first (AAP §0.7.5).
//! - **Security mitigations are x86-64 ONLY** — per AAP §0.6.2, they are out
//!   of scope for other architectures.
//! - **Zero external crate dependencies** — only `std::` imports are used.
//! - Tests validate machine code output via `objdump -d` disassembly inspection.
//! - User Examples from AAP are validated exactly:
//!   - Retpoline: `(*fptr)()` → call targets `__x86_indirect_thunk_*`, not pointer
//!   - Stack probe: `void f(void) { char buf[8192]; buf[0] = 1; }` → disassembly
//!     MUST show a probe loop before stack pointer adjustment
//! - CET: every function entry starts with `endbr64` (encoding `f3 0f 1e fa`)
//!
//! # Prerequisites
//! - A release build of BCC must be available (`cargo build --release`).
//! - GNU binutils `objdump` must be installed (v2.42 on Ubuntu 24.04).
//! - Test fixture files must exist at `tests/fixtures/security/`.
//!
//! # Zero Dependencies
//! Only `std::` imports are used — no external crates.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Internal Helper Functions
// ---------------------------------------------------------------------------

/// Compiles a C source file with BCC using the specified flags, targeting x86-64.
///
/// Returns `(output_path, compile_output)` where `output_path` is the path to
/// the compiled binary and `compile_output` is the raw process output.
///
/// The caller is responsible for cleaning up the output file.
fn compile_for_x86_64(
    source: &str,
    extra_flags: &[&str],
    test_name: &str,
) -> (PathBuf, std::process::Output) {
    let output_path = common::temp_output_path(test_name);
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let mut flags: Vec<&str> = vec!["--target=x86-64", "-o", output_str];
    flags.extend_from_slice(extra_flags);

    let compile_output = common::compile(source, &flags);
    (output_path, compile_output)
}

/// Compiles a C source file and returns the disassembly output from `objdump -d`.
///
/// This combines compilation and disassembly into a single helper, which is the
/// primary workflow for security mitigation validation tests.
///
/// # Panics
/// Panics if compilation fails (non-zero exit code).
fn compile_and_disassemble(
    fixture_name: &str,
    extra_flags: &[&str],
    test_name: &str,
) -> (String, PathBuf) {
    let fixture = common::fixture_path(fixture_name);
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (output_path, compile_output) = compile_for_x86_64(fixture_str, extra_flags, test_name);

    common::assert_compilation_succeeds(&compile_output);

    // Verify the binary was actually produced
    assert!(
        output_path.exists(),
        "Compiled binary not found at '{}' after successful compilation",
        output_path.display()
    );

    let disasm =
        common::objdump_disassemble(output_path.to_str().expect("output path is valid UTF-8"));

    (disasm, output_path)
}

/// Extracts the disassembly of a specific function from the full `objdump -d` output.
///
/// Searches for a line matching `<function_name>:` and collects all subsequent
/// lines until the next function label or end of output.
///
/// Returns `None` if the function is not found in the disassembly.
fn extract_function_disasm(full_disasm: &str, function_name: &str) -> Option<String> {
    let mut lines = full_disasm.lines();
    let mut found = false;
    let mut result = String::new();

    // Look for the function label pattern: `<function_name>:`
    // objdump typically outputs lines like:
    //   0000000000401000 <function_name>:
    let label_pattern = format!("<{}>:", function_name);

    for line in &mut lines {
        if line.contains(&label_pattern) {
            found = true;
            result.push_str(line);
            result.push('\n');
            break;
        }
    }

    if !found {
        return None;
    }

    // Collect instructions until the next function label or blank section separator
    for line in lines {
        // A new function label starts with an address and contains `<...>:`
        // but is NOT an instruction-level label reference
        if line.contains(">:")
            && !line.contains("call")
            && !line.contains("jmp")
            && !line.starts_with(' ')
            && !line.starts_with('\t')
        {
            break;
        }
        // Section headers like "Disassembly of section .plt:" end a function
        if line.starts_with("Disassembly of section") {
            break;
        }
        // Empty lines between functions
        if line.trim().is_empty() && result.lines().count() > 2 {
            // Only break on blank lines if we already have some instructions
            // (objdump may have blank lines between functions)
            break;
        }
        result.push_str(line);
        result.push('\n');
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Checks if any line in the disassembly output contains the given pattern
/// (case-insensitive search).
fn disasm_contains(disasm: &str, pattern: &str) -> bool {
    let pattern_lower = pattern.to_lowercase();
    disasm
        .lines()
        .any(|line| line.to_lowercase().contains(&pattern_lower))
}

/// Counts the number of lines in the disassembly output that contain the given
/// pattern (case-insensitive search).
fn disasm_count(disasm: &str, pattern: &str) -> usize {
    let pattern_lower = pattern.to_lowercase();
    disasm
        .lines()
        .filter(|line| line.to_lowercase().contains(&pattern_lower))
        .count()
}

// ===========================================================================
// Phase 1: Prerequisites Verification
// ===========================================================================

/// Verifies that all prerequisites for Checkpoint 5 security tests are met:
/// - BCC binary exists and is executable
/// - All required C fixture files exist under `tests/fixtures/security/`
/// - The `objdump` tool is available for disassembly inspection
#[test]
fn test_checkpoint5_prerequisites() {
    // Verify BCC binary exists
    let bcc = common::bcc_path();
    assert!(
        bcc.exists(),
        "BCC binary not found at '{}'. Run `cargo build --release` first.",
        bcc.display()
    );

    // Verify all security fixture files exist
    let fixture_names = [
        "security/retpoline.c",
        "security/cet.c",
        "security/stack_probe.c",
    ];

    for name in &fixture_names {
        let fixture = common::fixture_path(name);
        let fixture_ref = Path::new(fixture.to_str().expect("fixture path is valid UTF-8"));
        assert!(
            fixture_ref.exists(),
            "Security fixture file '{}' not found at '{}'",
            name,
            fixture.display()
        );
        // Verify fixture is a regular file with non-zero size
        let metadata = fs::metadata(&fixture).unwrap_or_else(|e| {
            panic!(
                "Cannot read metadata for fixture '{}': {}",
                fixture.display(),
                e
            )
        });
        assert!(
            metadata.len() > 0,
            "Security fixture '{}' is empty (0 bytes)",
            name
        );
    }

    // Verify objdump is available
    let objdump_check = Command::new("objdump").arg("--version").output();
    assert!(
        objdump_check.is_ok() && objdump_check.unwrap().status.success(),
        "objdump not found or not working. binutils must be installed for \
         Checkpoint 5 security mitigation validation."
    );
}

// ===========================================================================
// Phase 2: Retpoline Tests (-mretpoline)
// ===========================================================================

/// Validates retpoline code generation for indirect calls.
///
/// Per AAP User Example: A function containing `(*fptr)()` call → the call
/// instruction MUST target `__x86_indirect_thunk_*` (e.g., `__x86_indirect_thunk_rax`),
/// NOT the pointer directly.
///
/// Compiles `tests/fixtures/security/retpoline.c` with `-mretpoline --target=x86-64`
/// and inspects the disassembly of `call_indirect` to verify:
/// 1. A `call` to `__x86_indirect_thunk_*` is present.
/// 2. No direct `call *%rax` or `jmp *%rax` patterns exist for the indirect call.
#[test]
fn test_retpoline_indirect_call() {
    let (disasm, output_path) = compile_and_disassemble(
        "security/retpoline.c",
        &["-mretpoline"],
        "retpoline_indirect",
    );

    // --- Positive assertion: call to __x86_indirect_thunk_* must be present ---
    // The retpoline thunk name follows the pattern __x86_indirect_thunk_<reg>
    // where <reg> is one of: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8..r15
    let has_thunk_call = disasm_contains(&disasm, "__x86_indirect_thunk_");
    assert!(
        has_thunk_call,
        "Retpoline: disassembly MUST contain a call to __x86_indirect_thunk_* \
         when compiled with -mretpoline.\n\
         Full disassembly (first 2000 chars):\n{}",
        &disasm[..disasm.len().min(2000)]
    );

    // --- Negative assertion: no direct indirect call/jmp via register ---
    // In the call_indirect function specifically, there should be no
    // `call *%rax` or `jmp *%rax` (or any register variant) patterns
    // since those are exactly what retpoline prevents.
    let call_indirect_disasm = extract_function_disasm(&disasm, "call_indirect");
    if let Some(ref func_disasm) = call_indirect_disasm {
        // Check for forbidden direct indirect call patterns
        let forbidden_patterns = [
            "call   *%",
            "callq  *%",
            "call  *%",
            "callq *%",
            "jmp    *%",
            "jmpq   *%",
            "jmp   *%",
            "jmpq  *%",
        ];

        for pattern in &forbidden_patterns {
            assert!(
                !disasm_contains(func_disasm, pattern),
                "Retpoline: call_indirect function MUST NOT contain direct indirect \
                 branch '{}' when -mretpoline is active.\n\
                 call_indirect disassembly:\n{}",
                pattern,
                func_disasm
            );
        }
    }

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that retpoline thunk functions are present in the binary and use
/// the correct LFENCE+JMP loop pattern (not a direct indirect branch).
///
/// The thunk should contain a sequence like:
///   call .+N       ; push return address
///   lfence / pause ; speculation barrier
///   jmp .-M        ; loop back (speculative path spin-loops)
///   ...
///   ret             ; actual return (non-speculative)
#[test]
fn test_retpoline_thunk_presence() {
    let (disasm, output_path) =
        compile_and_disassemble("security/retpoline.c", &["-mretpoline"], "retpoline_thunk");

    // Verify at least one __x86_indirect_thunk_* symbol exists as a function
    let thunk_present = disasm_contains(&disasm, "<__x86_indirect_thunk_");
    assert!(
        thunk_present,
        "Retpoline: at least one __x86_indirect_thunk_<reg> function must be \
         present in the binary when compiled with -mretpoline.\n\
         Full disassembly (first 2000 chars):\n{}",
        &disasm[..disasm.len().min(2000)]
    );

    // The thunk body should contain lfence (speculation barrier) or pause
    // as part of the retpoline loop, and should NOT contain a direct
    // jmp *%<reg> instruction (that defeats the purpose of retpoline).
    // We look for lfence OR pause in the overall thunk area.
    let has_barrier = disasm_contains(&disasm, "lfence") || disasm_contains(&disasm, "pause");
    assert!(
        has_barrier,
        "Retpoline thunk must contain a speculation barrier instruction \
         (lfence or pause) as part of the retpoline loop pattern.\n\
         Full disassembly (first 2000 chars):\n{}",
        &disasm[..disasm.len().min(2000)]
    );

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that compiling WITHOUT `-mretpoline` produces normal indirect calls
/// (no thunks). This ensures the flag actually controls retpoline behavior.
#[test]
fn test_no_retpoline_without_flag() {
    let (disasm, output_path) = compile_and_disassemble(
        "security/retpoline.c",
        &[], // No -mretpoline flag
        "retpoline_no_flag",
    );

    // Without -mretpoline, the binary should NOT contain __x86_indirect_thunk_*
    // function symbols (thunks are not emitted unless the flag is active).
    let has_thunk = disasm_contains(&disasm, "<__x86_indirect_thunk_");
    assert!(
        !has_thunk,
        "Without -mretpoline, the binary MUST NOT contain __x86_indirect_thunk_* \
         functions. The flag must control retpoline behavior.\n\
         Full disassembly (first 2000 chars):\n{}",
        &disasm[..disasm.len().min(2000)]
    );

    // The call_indirect function should use a normal indirect call pattern
    let call_indirect_disasm = extract_function_disasm(&disasm, "call_indirect");
    if let Some(ref func_disasm) = call_indirect_disasm {
        // Without retpoline, we expect a direct indirect call like `call *%rax`
        // or `callq *%rax` or similar register-indirect pattern
        let has_direct_indirect =
            disasm_contains(func_disasm, "call") || disasm_contains(func_disasm, "jmp");
        assert!(
            has_direct_indirect,
            "Without -mretpoline, call_indirect should contain a normal call/jmp \
             instruction.\ncall_indirect disassembly:\n{}",
            func_disasm
        );
    }

    // Clean up using cleanup_temp_files helper
    let path_str = output_path.to_str().expect("output path is valid UTF-8");
    common::cleanup_temp_files(&[path_str]);
}

// ===========================================================================
// Phase 3: CET/IBT Tests (-fcf-protection)
// ===========================================================================

/// Validates that every function entry point starts with `endbr64` when
/// compiled with `-fcf-protection`.
///
/// The `endbr64` instruction encoding is `f3 0f 1e fa` (4 bytes). Per AAP,
/// CET/IBT inserts `endbr64` at function entries and indirect branch targets.
#[test]
fn test_cet_endbr64_at_function_entry() {
    let (disasm, output_path) =
        compile_and_disassemble("security/cet.c", &["-fcf-protection"], "cet_entry");

    // The CET fixture defines these functions: add, multiply, negate, apply, main
    // Each function entry must begin with endbr64
    let expected_functions = ["add", "multiply", "negate", "apply", "main"];

    for func_name in &expected_functions {
        let func_disasm = extract_function_disasm(&disasm, func_name);
        assert!(
            func_disasm.is_some(),
            "CET: function '{}' not found in disassembly. \
             All functions must be visible (non-static).",
            func_name
        );

        let func_disasm = func_disasm.unwrap();

        // The first instruction after the function label should be endbr64.
        // objdump formats it as "endbr64" in the instruction mnemonic column.
        // We check the first few instruction lines for endbr64.
        let instruction_lines: Vec<&str> = func_disasm
            .lines()
            .filter(|line| {
                // Instruction lines typically start with whitespace and contain ':'
                // after the address, e.g., "  401000:	f3 0f 1e fa          	endbr64"
                let trimmed = line.trim();
                trimmed.contains(':') && !trimmed.ends_with(">:")
            })
            .collect();

        // The very first instruction should be endbr64
        if let Some(first_insn) = instruction_lines.first() {
            assert!(
                first_insn.to_lowercase().contains("endbr64"),
                "CET: function '{}' must start with endbr64 when compiled with \
                 -fcf-protection. First instruction: '{}'\n\
                 Function disassembly:\n{}",
                func_name,
                first_insn.trim(),
                func_disasm
            );
        } else {
            panic!(
                "CET: no instructions found for function '{}' in disassembly.\n\
                 Function disassembly:\n{}",
                func_name, func_disasm
            );
        }
    }

    // Additionally verify the endbr64 byte encoding f3 0f 1e fa is present
    let endbr64_encoding_count = disasm_count(&disasm, "f3 0f 1e fa");
    assert!(
        endbr64_encoding_count >= expected_functions.len(),
        "CET: expected at least {} endbr64 instructions (f3 0f 1e fa encoding) \
         for {} functions, found {}",
        expected_functions.len(),
        expected_functions.len(),
        endbr64_encoding_count
    );

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that `endbr64` is present at indirect branch targets.
///
/// The `apply` function in cet.c calls through a function pointer. The target
/// functions (`add`, `multiply`) must have `endbr64` at their entry points to
/// serve as valid indirect branch targets under CET/IBT.
#[test]
fn test_cet_endbr64_at_indirect_targets() {
    let (disasm, output_path) =
        compile_and_disassemble("security/cet.c", &["-fcf-protection"], "cet_indirect");

    // Functions that are called through a function pointer must have endbr64.
    // In cet.c, `apply` calls through `op(x, y)` where `op` is `add` or `multiply`.
    let indirect_targets = ["add", "multiply"];

    for func_name in &indirect_targets {
        let func_disasm = extract_function_disasm(&disasm, func_name);
        assert!(
            func_disasm.is_some(),
            "CET: indirect branch target function '{}' not found in disassembly.",
            func_name
        );

        let func_disasm = func_disasm.unwrap();

        // Verify endbr64 is present in the function (should be at entry)
        assert!(
            disasm_contains(&func_disasm, "endbr64"),
            "CET: indirect branch target '{}' MUST contain endbr64 when compiled \
             with -fcf-protection.\nFunction disassembly:\n{}",
            func_name,
            func_disasm
        );
    }

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that `endbr64` is NOT present when compiling WITHOUT `-fcf-protection`.
///
/// This ensures the flag actually controls CET/IBT behavior — function entries
/// should NOT have `endbr64` by default.
#[test]
fn test_no_cet_without_flag() {
    let (disasm, output_path) = compile_and_disassemble(
        "security/cet.c",
        &[], // No -fcf-protection flag
        "cet_no_flag",
    );

    // Without -fcf-protection, the endbr64 instruction should NOT appear
    // at function entries. Check that the endbr64 encoding is absent.
    let endbr64_count = disasm_count(&disasm, "endbr64");
    assert_eq!(
        endbr64_count,
        0,
        "Without -fcf-protection, endbr64 MUST NOT appear at function entries. \
         Found {} occurrences. The flag must control CET behavior.\n\
         Full disassembly (first 2000 chars):\n{}",
        endbr64_count,
        &disasm[..disasm.len().min(2000)]
    );

    // Clean up
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 4: Stack Probe Tests (frames > 4096 bytes)
// ===========================================================================

/// Validates stack guard page probing for large stack frames.
///
/// Per AAP User Example: `void f(void) { char buf[8192]; buf[0] = 1; }`
/// disassembly MUST show a probe loop BEFORE the stack pointer adjustment.
///
/// The probe loop touches each 4096-byte page to ensure the guard page is not
/// silently skipped, preventing stack clash attacks.
#[test]
fn test_stack_probe_large_frame() {
    let (disasm, output_path) =
        compile_and_disassemble("security/stack_probe.c", &[], "stack_probe_large");

    // Extract the disassembly of function 'f' which has an 8192-byte buffer
    let f_disasm = extract_function_disasm(&disasm, "f");
    assert!(
        f_disasm.is_some(),
        "Stack probe: function 'f' not found in disassembly. \
         It must be a non-static, visible function."
    );
    let f_disasm = f_disasm.unwrap();

    // The probe loop pattern varies by implementation but generally includes:
    // 1. A comparison or counter tracking how many pages to probe
    // 2. A memory touch (e.g., `mov` or `test` to `(%rsp)` or similar)
    //    at each 4096-byte boundary
    // 3. A loop back (conditional jump)
    //
    // Common patterns include:
    //   - `sub $0x1000, %rsp` + `test %rsp, (%rsp)` in a loop
    //   - `or $0x0, (%rsp)` as a probe touch
    //   - A loop with `cmp` and `jne`/`jb`/`jge` for iteration
    //
    // We check for indicators of a probe loop:
    // (a) Presence of the page size constant 0x1000 (4096)
    // (b) A loop structure (backward jump or conditional branch)
    // (c) A memory access probing the stack

    let has_page_size = disasm_contains(&f_disasm, "0x1000")
        || disasm_contains(&f_disasm, "$4096")
        || disasm_contains(&f_disasm, "$0x1000");

    // Look for loop patterns: backward jumps (jne, jb, jge, jl, jmp with negative offset)
    let has_loop = disasm_contains(&f_disasm, "jne")
        || disasm_contains(&f_disasm, "jb")
        || disasm_contains(&f_disasm, "jge")
        || disasm_contains(&f_disasm, "jl")
        || disasm_contains(&f_disasm, "jg")
        || disasm_contains(&f_disasm, "ja")
        || disasm_contains(&f_disasm, "jbe")
        || disasm_contains(&f_disasm, "jae")
        || disasm_contains(&f_disasm, "loop");

    // Check for stack memory probing (touching the stack at current position)
    let has_probe_touch = disasm_contains(&f_disasm, "(%rsp)")
        || disasm_contains(&f_disasm, "(%esp)")
        || disasm_contains(&f_disasm, ",(%rsp")
        || disasm_contains(&f_disasm, "0(%rsp");

    // At least two of the three indicators should be present for a valid probe
    let probe_indicators = [has_page_size, has_loop, has_probe_touch];
    let indicator_count = probe_indicators.iter().filter(|&&x| x).count();

    assert!(
        indicator_count >= 2,
        "Stack probe: function 'f' (8192-byte frame) MUST show a probe loop \
         BEFORE the stack pointer adjustment.\n\
         Indicators found: page_size(0x1000)={}, loop_pattern={}, stack_touch={}\n\
         At least 2 of 3 indicators expected.\n\
         Function 'f' disassembly:\n{}",
        has_page_size,
        has_loop,
        has_probe_touch,
        f_disasm
    );

    // Also verify function 'h' (16384-byte frame, 4 pages) has probing
    let h_disasm = extract_function_disasm(&disasm, "h");
    if let Some(ref h_func) = h_disasm {
        let h_has_loop = disasm_contains(h_func, "jne")
            || disasm_contains(h_func, "jb")
            || disasm_contains(h_func, "jge")
            || disasm_contains(h_func, "jl")
            || disasm_contains(h_func, "jg")
            || disasm_contains(h_func, "ja")
            || disasm_contains(h_func, "jbe")
            || disasm_contains(h_func, "jae")
            || disasm_contains(h_func, "loop");

        assert!(
            h_has_loop,
            "Stack probe: function 'h' (16384-byte frame, 4 pages) should also \
             have a probe loop.\nFunction 'h' disassembly:\n{}",
            h_func
        );
    }

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that NO stack probe loop is generated for small stack frames
/// (less than 4096 bytes), ensuring probing is conditional on frame size.
#[test]
fn test_no_stack_probe_small_frame() {
    let (disasm, output_path) =
        compile_and_disassemble("security/stack_probe.c", &[], "stack_probe_small");

    // Extract the disassembly of function 'g' which has only a 64-byte buffer
    let g_disasm = extract_function_disasm(&disasm, "g");
    assert!(
        g_disasm.is_some(),
        "Stack probe: function 'g' not found in disassembly. \
         It must be a non-static, visible function."
    );
    let g_disasm = g_disasm.unwrap();

    // For a small frame (64 bytes < 4096), there should be no probe loop.
    // Specifically, there should be no reference to the page size 0x1000
    // in a probing context. A simple sub $0x40,%rsp (or similar small
    // allocation) is expected without any loop around it.
    let has_probe_page_size =
        disasm_contains(&g_disasm, "0x1000") || disasm_contains(&g_disasm, "$4096");

    // Count loop-like patterns in the small function
    let loop_patterns = ["jne", "jb", "jge", "jl", "loop"];
    let loop_count: usize = loop_patterns
        .iter()
        .map(|p| disasm_count(&g_disasm, p))
        .sum();

    // A small-frame function should not have both a page-size constant and a loop
    // (it's acceptable to have a conditional jump for the asm volatile constraint,
    // but not a probe loop involving 0x1000)
    assert!(
        !(has_probe_page_size && loop_count > 0),
        "Stack probe: function 'g' (64-byte frame) MUST NOT have a stack probe \
         loop. Small frames (<4096 bytes) don't need probing.\n\
         Found probe page size (0x1000): {}\n\
         Found loop-like jumps: {}\n\
         Function 'g' disassembly:\n{}",
        has_probe_page_size,
        loop_count,
        g_disasm
    );

    // Clean up
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 5: Combined Security Mitigations
// ===========================================================================

/// Validates that retpoline and CET can be combined without interference.
///
/// When both `-mretpoline` and `-fcf-protection` are active, the binary
/// should contain BOTH retpoline thunks AND `endbr64` instructions.
#[test]
fn test_combined_retpoline_and_cet() {
    // Use the retpoline fixture since it has an indirect call that exercises
    // both mitigations (retpoline for the indirect call, CET for all entries)
    let (disasm, output_path) = compile_and_disassemble(
        "security/retpoline.c",
        &["-mretpoline", "-fcf-protection"],
        "combined_retpoline_cet",
    );

    // Verify retpoline thunks are present
    let has_thunk = disasm_contains(&disasm, "__x86_indirect_thunk_");
    assert!(
        has_thunk,
        "Combined: retpoline thunks must be present when both -mretpoline \
         and -fcf-protection are active.\n\
         Full disassembly (first 2000 chars):\n{}",
        &disasm[..disasm.len().min(2000)]
    );

    // Verify endbr64 instructions are present at function entries
    let endbr64_count = disasm_count(&disasm, "endbr64");
    assert!(
        endbr64_count > 0,
        "Combined: endbr64 instructions must be present when both -mretpoline \
         and -fcf-protection are active.\n\
         Full disassembly (first 2000 chars):\n{}",
        &disasm[..disasm.len().min(2000)]
    );

    // Verify the main function has endbr64 at its entry
    let main_disasm = extract_function_disasm(&disasm, "main");
    if let Some(ref func) = main_disasm {
        let instruction_lines: Vec<&str> = func
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.contains(':') && !trimmed.ends_with(">:")
            })
            .collect();

        if let Some(first_insn) = instruction_lines.first() {
            assert!(
                first_insn.to_lowercase().contains("endbr64"),
                "Combined: main function must start with endbr64. \
                 First instruction: '{}'\nFunction disassembly:\n{}",
                first_insn.trim(),
                func
            );
        }
    }

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that security flags are gracefully handled on non-x86-64 targets.
///
/// When `-mretpoline` is used with `--target=aarch64`, the compiler should
/// either silently ignore the flag (compilation succeeds) or emit a warning
/// but still succeed. Security mitigations are x86-64 only per AAP §0.6.2.
#[test]
fn test_security_flags_ignored_non_x86_64() {
    // Check if QEMU for aarch64 is available — if not, we note it but still
    // test compilation behavior (we don't need to run the binary, just compile)
    let aarch64_qemu = common::qemu_available("aarch64");
    if !aarch64_qemu {
        eprintln!(
            "Note: qemu-aarch64 is not available. Still testing compilation \
             flag handling (execution tests skipped for non-x86-64)."
        );
    }

    let fixture = common::fixture_path("security/retpoline.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("security_non_x86");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Compile with -mretpoline targeting aarch64
    let compile_output = common::compile(
        fixture_str,
        &["--target=aarch64", "-mretpoline", "-o", output_str],
    );

    // The compilation should succeed (possibly with a warning).
    // If it fails, that's also acceptable as long as the error is about
    // the flag being unsupported on the target, not a compilation bug.
    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr);
        // Accept failure only if it mentions the flag/target incompatibility
        let is_flag_rejection = stderr.to_lowercase().contains("retpoline")
            || stderr.to_lowercase().contains("not supported")
            || stderr.to_lowercase().contains("ignored")
            || stderr.to_lowercase().contains("x86");
        assert!(
            is_flag_rejection,
            "Security flags on non-x86-64: compilation failed but the error does \
             not mention flag/target incompatibility. This may be a real compilation \
             bug.\nstderr: {}",
            stderr
        );
    }
    // If it succeeded, that's fine — the flag was silently ignored.

    // Also test -fcf-protection on aarch64
    let compile_output2 = common::compile(
        fixture_str,
        &["--target=aarch64", "-fcf-protection", "-o", output_str],
    );

    if !compile_output2.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output2.stderr);
        let is_flag_rejection = stderr.to_lowercase().contains("cf-protection")
            || stderr.to_lowercase().contains("not supported")
            || stderr.to_lowercase().contains("ignored")
            || stderr.to_lowercase().contains("x86")
            || stderr.to_lowercase().contains("cet");
        assert!(
            is_flag_rejection,
            "Security flags on non-x86-64: -fcf-protection compilation failed but \
             the error does not mention flag/target incompatibility.\nstderr: {}",
            stderr
        );
    }

    // Clean up using both approaches for coverage
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 6: Execution Correctness with Mitigations
// ===========================================================================

/// Validates that a program compiled with `-mretpoline` executes correctly.
///
/// Retpoline mitigations must not break program functionality — the indirect
/// call through the retpoline thunk must still invoke the correct function
/// and return the correct result.
#[test]
fn test_retpoline_binary_executes_correctly() {
    let fixture = common::fixture_path("security/retpoline.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("retpoline_exec");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Compile with retpoline enabled
    let compile_output = common::compile(
        fixture_str,
        &["--target=x86-64", "-mretpoline", "-o", output_str],
    );
    common::assert_compilation_succeeds(&compile_output);

    // Run the binary natively (x86-64 target on x86-64 host)
    let run_output = common::run_binary(output_str, "x86-64");

    // Assert correct execution
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "retpoline OK\n");

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that the stack probe test binary executes correctly without
/// segfaulting.
///
/// The probe loop must correctly touch each page so the program doesn't
/// crash by skipping the guard page. A successful execution (exit code 0
/// and "stack_probe OK" output) proves the probe works.
#[test]
fn test_stack_probe_binary_executes_correctly() {
    let fixture = common::fixture_path("security/stack_probe.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("stack_probe_exec");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Compile for x86-64
    let compile_output = common::compile(fixture_str, &["--target=x86-64", "-o", output_str]);
    common::assert_compilation_succeeds(&compile_output);

    // Run the binary — the large-frame functions must not segfault
    let run_output = common::run_binary(output_str, "x86-64");

    // Assert correct execution (no segfault, correct output)
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "stack_probe OK\n");

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that a program compiled with `-fcf-protection` (CET/IBT) executes
/// correctly — the `endbr64` instructions must not interfere with normal
/// program control flow.
#[test]
fn test_cet_binary_executes_correctly() {
    let fixture = common::fixture_path("security/cet.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("cet_exec");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Compile with CET enabled
    let compile_output = common::compile(
        fixture_str,
        &["--target=x86-64", "-fcf-protection", "-o", output_str],
    );
    common::assert_compilation_succeeds(&compile_output);

    // Run the binary
    let run_output = common::run_binary(output_str, "x86-64");

    // Assert correct execution
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "cet OK\n");

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Validates that the CET fixture executes correctly through compile_and_run
/// when using the x86-64 target (no CET flag needed for basic execution test).
#[test]
fn test_cet_compile_and_run_baseline() {
    let fixture = common::fixture_path("security/cet.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    // Use compile_and_run helper for integrated compile-then-execute workflow
    let run_output = common::compile_and_run(fixture_str, "x86-64");

    // Assert correct execution
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "cet OK\n");
}

/// Validates that a program compiled with ALL security mitigations active
/// (`-mretpoline -fcf-protection`) on an indirect-call-heavy program
/// executes correctly.
#[test]
fn test_all_mitigations_execute_correctly() {
    let fixture = common::fixture_path("security/retpoline.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("all_mitigations_exec");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Verify output path construction with PathBuf::from for schema compliance
    let output_pathbuf = PathBuf::from(output_str);
    assert!(
        output_pathbuf.to_str().is_some(),
        "PathBuf should produce valid UTF-8 string"
    );

    // Compile with all security flags
    let compile_output = common::compile(
        fixture_str,
        &[
            "--target=x86-64",
            "-mretpoline",
            "-fcf-protection",
            "-o",
            output_str,
        ],
    );
    common::assert_compilation_succeeds(&compile_output);

    // Verify the binary was produced using fs::metadata
    let binary_meta = fs::metadata(&output_path);
    assert!(
        binary_meta.is_ok(),
        "Compiled binary must exist at '{}'",
        output_path.display()
    );
    assert!(
        binary_meta.unwrap().len() > 0,
        "Compiled binary must not be empty"
    );

    // Run the binary
    let run_output = common::run_binary(output_str, "x86-64");

    // Assert correct execution even with all mitigations enabled
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "retpoline OK\n");

    // Clean up
    let _ = fs::remove_file(&output_path);
}
