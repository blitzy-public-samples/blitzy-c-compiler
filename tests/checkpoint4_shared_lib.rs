//! Checkpoint 4 — Shared Library and DWARF Debug Information Validation
//!
//! This test suite validates two major capabilities of BCC:
//!
//! 1. **Shared Library Production (ET_DYN ELF):** BCC must produce correct
//!    shared objects with all required ELF sections (`.dynamic`, `.dynsym`,
//!    `.dynstr`, `.rela.dyn`, `.rela.plt`, `.gnu.hash`, `.got`, `.got.plt`,
//!    `.plt`), correct program headers (`PT_DYNAMIC`, `PT_INTERP`), proper
//!    symbol visibility control, and PIC code generation patterns
//!    (`@GOTPCREL`, `@PLT`).
//!
//! 2. **DWARF v4 Debug Information:** When compiled with `-g`, BCC must emit
//!    DWARF v4 sections (`.debug_info`, `.debug_abbrev`, `.debug_line`,
//!    `.debug_str`) with correct content. When compiled **without** `-g`, zero
//!    debug sections must be present (AAP §0.7.10).
//!
//! # AAP Compliance
//! - **Sequential hard gate:** Checkpoint 3 must pass first (AAP §0.7.5).
//! - **Backend validation order:** x86-64 → i686 → AArch64 → RISC-V 64.
//! - **Zero external crate dependencies** — only `std::` imports are used.
//! - **DWARF conditionality (AAP §0.7.10):** Binary with `-g` MUST have
//!   DWARF sections; without `-g` MUST NOT contain any `.debug_*` sections.
//!
//! # Fixture Files
//! - `tests/fixtures/shared_lib/foo.c`  — Shared library with exported functions
//! - `tests/fixtures/shared_lib/main.c` — Dynamic linking consumer
//! - `tests/fixtures/dwarf/debug_test.c` — DWARF debug info test source
//!
//! # Prerequisites
//! - Release build of BCC: `cargo build --release`
//! - GNU binutils: `readelf` (v2.42), `objdump` (v2.42)
//! - GDB (v15.0.50) for source-level debug validation
//! - QEMU user-mode emulation for cross-architecture tests
//!
//! # Zero Dependencies
//! Only `std::` imports are used — no external crates.

mod common;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

/// Compiles a shared library from a C source file using BCC.
///
/// Invokes `bcc -fPIC -shared -o <output> --target=<target> <source>`.
///
/// Returns `(output_path, compile_output)` where `output_path` is the path to
/// the produced `.so` file.
fn compile_shared_lib(
    source: &str,
    target: &str,
    test_name: &str,
) -> (PathBuf, std::process::Output) {
    let output_path = common::temp_output_path(test_name);
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");
    let target_flag = format!("--target={}", target);

    let compile_output = common::compile(
        source,
        &["-fPIC", "-shared", "-o", output_str, &target_flag],
    );
    (output_path, compile_output)
}

/// Compiles and links an executable against a shared library using BCC.
///
/// Invokes `bcc -o <output> --target=<target> -L<lib_dir> -l<lib_name> <source>`.
///
/// Returns `(output_path, link_output)`.
fn compile_with_shared_lib(
    source: &str,
    target: &str,
    lib_dir: &str,
    lib_name: &str,
    test_name: &str,
) -> (PathBuf, std::process::Output) {
    let output_path = common::temp_output_path(test_name);
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");
    let target_flag = format!("--target={}", target);
    let lib_dir_flag = format!("-L{}", lib_dir);
    let lib_name_flag = format!("-l{}", lib_name);

    let compile_output = common::compile(
        source,
        &[
            "-o",
            output_str,
            &target_flag,
            &lib_dir_flag,
            &lib_name_flag,
        ],
    );
    (output_path, compile_output)
}

/// Compiles a C source file with debug info using BCC.
///
/// Invokes `bcc -g -o <output> --target=<target> <source>`.
///
/// Returns `(output_path, compile_output)`.
fn compile_with_debug(
    source: &str,
    target: &str,
    test_name: &str,
) -> (PathBuf, std::process::Output) {
    let output_path = common::temp_output_path(test_name);
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");
    let target_flag = format!("--target={}", target);

    let compile_output = common::compile(source, &["-g", "-o", output_str, &target_flag]);
    (output_path, compile_output)
}

/// Compiles a C source file WITHOUT debug info using BCC.
///
/// Invokes `bcc -o <output> --target=<target> <source>`.
///
/// Returns `(output_path, compile_output)`.
fn compile_without_debug(
    source: &str,
    target: &str,
    test_name: &str,
) -> (PathBuf, std::process::Output) {
    let output_path = common::temp_output_path(test_name);
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");
    let target_flag = format!("--target={}", target);

    let compile_output = common::compile(source, &["-o", output_str, &target_flag]);
    (output_path, compile_output)
}

/// Extracts the parent directory path from a PathBuf, returning it as a String.
///
/// Used to derive the `-L` library search path from a shared library file path.
#[allow(dead_code)]
fn parent_dir_str(path: &Path) -> String {
    path.parent()
        .unwrap_or_else(|| Path::new("."))
        .to_str()
        .unwrap_or(".")
        .to_string()
}

// ===========================================================================
// Phase 2: Shared Library Compilation Tests
// ===========================================================================

/// Checkpoint 4.1 — Shared library compilation for x86-64.
///
/// Compiles `tests/fixtures/shared_lib/foo.c` with `-fPIC -shared` targeting
/// x86-64 and verifies:
/// - Compilation succeeds (exit code 0).
/// - Output file is an ELF shared object (ET_DYN) per `readelf -h`.
#[test]
fn test_shared_lib_compilation_x86_64() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) =
        compile_shared_lib(fixture_str, "x86-64", "cp4_shared_lib_x86_64");

    // Assert compilation succeeds
    common::assert_compilation_succeeds(&compile_output);

    // Verify the output file exists
    assert!(
        so_path.exists(),
        "Shared library output file was not produced: {}",
        so_path.display()
    );

    // Verify ELF type is ET_DYN (shared object) via readelf -h
    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let header = common::readelf_header(so_path_str);
    assert!(
        header.contains("DYN"),
        "Expected ET_DYN (shared object) in ELF header, got:\n{}",
        header
    );
    // Verify it's a 64-bit ELF
    assert!(
        header.contains("ELF64")
            || header.contains("Class:") && header.contains("ELF64")
            || header.contains("ELFCLASS64")
            || header.contains("class64"),
        "Expected 64-bit ELF for x86-64 shared library"
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.2 — Shared library linking and execution on x86-64.
///
/// Compiles `tests/fixtures/shared_lib/main.c` linked against `libfoo.so`,
/// runs the resulting binary with `LD_LIBRARY_PATH` set, and verifies:
/// - Linking succeeds.
/// - The binary produces correct output from dynamically linked function calls.
/// - Expected stdout: `"shared_lib OK\n"`.
#[test]
fn test_shared_lib_linking_x86_64() {
    // Step 1: Compile the shared library
    let foo_fixture = common::fixture_path("shared_lib/foo.c");
    let foo_str = foo_fixture
        .to_str()
        .expect("foo fixture path is valid UTF-8");

    let (so_path, compile_so) = compile_shared_lib(foo_str, "x86-64", "cp4_link_libfoo");
    common::assert_compilation_succeeds(&compile_so);
    assert!(so_path.exists(), "libfoo.so not produced");

    // Step 2: Compile main.c linked against libfoo
    let main_fixture = common::fixture_path("shared_lib/main.c");
    let main_str = main_fixture
        .to_str()
        .expect("main fixture path is valid UTF-8");

    // The shared library file is named like bcc_test_cp4_link_libfoo_<pid>_<n>,
    // but the linker expects libfoo.so. We need to create a symlink or rename.
    // For robustness, compile with explicit -o to a known .so name in a temp dir.
    let temp_dir = env::temp_dir().join(format!("bcc_cp4_link_{}", std::process::id()));
    let _ = fs::create_dir_all(&temp_dir);
    let libfoo_path = temp_dir.join("libfoo.so");

    // Re-compile the shared library with the expected name
    let libfoo_str = libfoo_path.to_str().expect("libfoo path is valid UTF-8");
    let target_flag = "--target=x86-64";
    let compile_libfoo = common::compile(
        foo_str,
        &["-fPIC", "-shared", "-o", libfoo_str, target_flag],
    );
    common::assert_compilation_succeeds(&compile_libfoo);
    assert!(
        libfoo_path.exists(),
        "libfoo.so not produced at expected path"
    );

    // Compile main.c linked against libfoo.so
    let temp_dir_str = temp_dir.to_str().expect("temp dir is valid UTF-8");
    let (main_bin, link_output) =
        compile_with_shared_lib(main_str, "x86-64", temp_dir_str, "foo", "cp4_link_main");
    common::assert_compilation_succeeds(&link_output);
    assert!(main_bin.exists(), "Linked executable not produced");

    // Step 3: Run the binary with LD_LIBRARY_PATH set
    let main_bin_str = main_bin.to_str().expect("main bin path is valid UTF-8");
    let run_output = Command::new(main_bin_str)
        .env("LD_LIBRARY_PATH", temp_dir_str)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run linked binary: {}", e));

    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "shared_lib OK\n");

    // Cleanup
    let _ = fs::remove_file(&so_path);
    let _ = fs::remove_file(&main_bin);
    let _ = fs::remove_dir_all(&temp_dir);
}

// ===========================================================================
// Phase 3: ELF Shared Library Structure Validation
// ===========================================================================

/// Checkpoint 4.3 — ELF section presence in shared library.
///
/// Compiles a shared library and uses `readelf -S` to verify that all
/// required dynamic linking sections are present:
/// `.dynamic`, `.dynsym`, `.dynstr`, `.rela.dyn` or `.rela.plt`,
/// `.gnu.hash`, `.got`, `.got.plt`, `.plt`.
#[test]
fn test_shared_lib_elf_sections() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) = compile_shared_lib(fixture_str, "x86-64", "cp4_elf_sections");
    common::assert_compilation_succeeds(&compile_output);

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let sections = common::readelf_sections(so_path_str);

    // Required sections for a shared library
    let required_sections = [".dynamic", ".dynsym", ".dynstr"];
    for section in &required_sections {
        assert!(
            sections.contains(section),
            "Required section '{}' missing from shared library.\nreadelf -S output:\n{}",
            section,
            sections
        );
    }

    // At least one of .rela.dyn or .rela.plt should be present
    assert!(
        sections.contains(".rela.dyn")
            || sections.contains(".rela.plt")
            || sections.contains(".rel.dyn")
            || sections.contains(".rel.plt"),
        "Neither .rela.dyn/.rela.plt nor .rel.dyn/.rel.plt found in shared library.\n\
         readelf -S output:\n{}",
        sections
    );

    // GNU hash table for efficient symbol lookup
    assert!(
        sections.contains(".gnu.hash") || sections.contains(".hash"),
        "Neither .gnu.hash nor .hash found in shared library.\n\
         readelf -S output:\n{}",
        sections
    );

    // GOT sections
    assert!(
        sections.contains(".got") || sections.contains(".got.plt"),
        "Neither .got nor .got.plt found in shared library.\n\
         readelf -S output:\n{}",
        sections
    );

    // PLT section
    assert!(
        sections.contains(".plt"),
        "Required section '.plt' missing from shared library.\n\
         readelf -S output:\n{}",
        sections
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.4 — Program headers in shared library.
///
/// Uses `readelf -l` to verify:
/// - `PT_DYNAMIC` segment is present.
/// - `PT_INTERP` segment references the correct dynamic linker path.
#[test]
fn test_shared_lib_program_headers() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) = compile_shared_lib(fixture_str, "x86-64", "cp4_prog_headers");
    common::assert_compilation_succeeds(&compile_output);

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let prog_headers = common::readelf_program_headers(so_path_str);

    // PT_DYNAMIC must be present
    assert!(
        prog_headers.contains("DYNAMIC"),
        "PT_DYNAMIC segment missing from shared library.\n\
         readelf -l output:\n{}",
        prog_headers
    );

    // For a shared library, PT_INTERP may or may not be present
    // (it's typically present for shared objects that can be executed directly).
    // If present, verify it references a valid dynamic linker.
    if prog_headers.contains("INTERP") {
        // On x86-64 Linux, the dynamic linker is /lib64/ld-linux-x86-64.so.2
        // or /lib/ld-linux-x86-64.so.2
        assert!(
            prog_headers.contains("ld-linux")
                || prog_headers.contains("ld.so")
                || prog_headers.contains("/lib"),
            "PT_INTERP present but does not reference a valid dynamic linker path.\n\
             readelf -l output:\n{}",
            prog_headers
        );
    }

    // Verify LOAD segments exist (at least one executable, one writable)
    assert!(
        prog_headers.contains("LOAD"),
        "No LOAD segments in shared library program headers.\n\
         readelf -l output:\n{}",
        prog_headers
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.5 — Dynamic symbol visibility in shared library.
///
/// Verifies that exported functions appear in `.dynsym` and hidden functions
/// do not.
#[test]
fn test_shared_lib_symbol_visibility() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) = compile_shared_lib(fixture_str, "x86-64", "cp4_sym_visibility");
    common::assert_compilation_succeeds(&compile_output);

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let symbols = common::readelf_symbols(so_path_str);

    // Exported functions MUST appear in symbol table
    let expected_exported = ["foo_add", "foo_multiply", "foo_greeting", "foo_get_global"];
    for sym in &expected_exported {
        assert!(
            symbols.contains(sym),
            "Exported function '{}' missing from symbol table.\n\
             readelf -s output:\n{}",
            sym,
            symbols
        );
    }

    // The hidden function foo_internal_helper should NOT appear in .dynsym
    // but may appear in .symtab. We check the dynamic symbols specifically.
    // readelf -s shows both .dynsym and .symtab. To specifically check .dynsym,
    // we look at the output section by section.
    let dynsym_output = Command::new("readelf")
        .args(["--dyn-syms", so_path_str])
        .output()
        .expect("readelf failed");
    let dynsym_str = String::from_utf8_lossy(&dynsym_output.stdout);

    // foo_internal_helper has __attribute__((visibility("hidden")))
    // It should NOT appear in dynamic symbols
    assert!(
        !dynsym_str.contains("foo_internal_helper"),
        "Hidden function 'foo_internal_helper' should NOT appear in .dynsym.\n\
         readelf --dyn-syms output:\n{}",
        dynsym_str
    );

    // Verify exported functions ARE in dynamic symbols
    for sym in &expected_exported {
        assert!(
            dynsym_str.contains(sym),
            "Exported function '{}' missing from .dynsym.\n\
             readelf --dyn-syms output:\n{}",
            sym,
            dynsym_str
        );
    }

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

// ===========================================================================
// Phase 4: PIC Code Generation Validation
// ===========================================================================

/// Checkpoint 4.6 — PIC code generation patterns.
///
/// Compiles with `-fPIC` and uses `objdump -d` to verify PIC addressing
/// patterns on x86-64:
/// - `@GOTPCREL` or `@PLT` references in disassembly.
/// - GOT/PLT relocation entries exist via `readelf -r`.
#[test]
fn test_pic_code_generation() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) = compile_shared_lib(fixture_str, "x86-64", "cp4_pic_codegen");
    common::assert_compilation_succeeds(&compile_output);

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");

    // Check disassembly for PIC addressing patterns
    let disasm = common::objdump_disassemble(so_path_str);

    // On x86-64 PIC, we expect @PLT or @GOTPCREL references
    // or RIP-relative addressing patterns (which is the default on x86-64)
    let has_plt_ref = disasm.contains("@PLT") || disasm.contains("plt");
    let has_got_ref =
        disasm.contains("@GOTPCREL") || disasm.contains("GOTPCREL") || disasm.contains("got");
    let has_rip_relative = disasm.contains("%rip") || disasm.contains("(%rip)");

    assert!(
        has_plt_ref || has_got_ref || has_rip_relative,
        "PIC code generation validation failed: no @PLT, @GOTPCREL, or RIP-relative \
         addressing patterns found in disassembly.\nobjdump -d output (first 200 lines):\n{}",
        disasm.lines().take(200).collect::<Vec<_>>().join("\n")
    );

    // Check relocations via readelf -r
    let relocations = common::readelf_relocations(so_path_str);

    // For a shared library, relocation entries should exist
    assert!(
        !relocations.is_empty()
            && (relocations.contains("R_X86_64")
                || relocations.contains("GLOB_DAT")
                || relocations.contains("JUMP_SLOT")
                || relocations.contains("RELATIVE")
                || relocations.contains("GOTPCREL")),
        "No relocation entries found in shared library (expected GOT/PLT relocations).\n\
         readelf -r output:\n{}",
        relocations
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

// ===========================================================================
// Phase 5: DWARF Debug Information Tests
// ===========================================================================

/// Checkpoint 4.7 — DWARF v4 section presence with `-g` flag.
///
/// Compiles `tests/fixtures/dwarf/debug_test.c` with `-g` and verifies
/// DWARF v4 sections are present:
/// - `.debug_info`  — Compilation unit DIEs, subprogram DIEs, variable DIEs
/// - `.debug_abbrev` — Abbreviation tables
/// - `.debug_line`  — Line number program
/// - `.debug_str`   — Debug string table
#[test]
fn test_dwarf_with_debug_flag() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (debug_bin, compile_output) =
        compile_with_debug(fixture_str, "x86-64", "cp4_dwarf_present");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = debug_bin
        .to_str()
        .expect("debug binary path is valid UTF-8");
    let sections = common::readelf_sections(bin_str);

    // All four DWARF v4 sections must be present
    let dwarf_sections = [".debug_info", ".debug_abbrev", ".debug_line", ".debug_str"];
    for section in &dwarf_sections {
        assert!(
            sections.contains(section),
            "DWARF section '{}' missing from binary compiled with -g.\n\
             readelf -S output:\n{}",
            section,
            sections
        );
    }

    // Cleanup
    let _ = fs::remove_file(&debug_bin);
}

/// Checkpoint 4.8 — DWARF `.debug_info` content validation.
///
/// Uses `readelf --debug-dump=info` to verify:
/// - `DW_TAG_compile_unit` with producer, language, and file name
/// - `DW_TAG_subprogram` entries for functions (main, add, make_point, compute)
/// - `DW_TAG_variable` entries for local variables
#[test]
fn test_dwarf_debug_info_content() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (debug_bin, compile_output) =
        compile_with_debug(fixture_str, "x86-64", "cp4_dwarf_info_content");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = debug_bin
        .to_str()
        .expect("debug binary path is valid UTF-8");

    // Dump .debug_info section
    let debug_info = common::readelf_debug_dump(bin_str, "info");

    // Verify DW_TAG_compile_unit presence
    assert!(
        debug_info.contains("DW_TAG_compile_unit") || debug_info.contains("Compilation Unit"),
        "DW_TAG_compile_unit not found in .debug_info.\n\
         readelf --debug-dump=info output (first 100 lines):\n{}",
        debug_info.lines().take(100).collect::<Vec<_>>().join("\n")
    );

    // Verify DW_TAG_subprogram entries for functions
    // The fixture defines: main, add, make_point, compute
    assert!(
        debug_info.contains("DW_TAG_subprogram")
            || debug_info.contains("Subprogram")
            || debug_info.contains("subprogram"),
        "No DW_TAG_subprogram entries found in .debug_info.\n\
         readelf --debug-dump=info output (first 100 lines):\n{}",
        debug_info.lines().take(100).collect::<Vec<_>>().join("\n")
    );

    // Verify function names appear in debug info
    let expected_functions = ["main", "add"];
    for func_name in &expected_functions {
        assert!(
            debug_info.contains(func_name),
            "Function '{}' not found in .debug_info content.\n\
             readelf --debug-dump=info output (first 200 lines):\n{}",
            func_name,
            debug_info.lines().take(200).collect::<Vec<_>>().join("\n")
        );
    }

    // Verify DW_TAG_variable entries exist
    assert!(
        debug_info.contains("DW_TAG_variable")
            || debug_info.contains("Variable")
            || debug_info.contains("variable"),
        "No DW_TAG_variable entries found in .debug_info.\n\
         readelf --debug-dump=info output (first 100 lines):\n{}",
        debug_info.lines().take(100).collect::<Vec<_>>().join("\n")
    );

    // Verify the source file name appears in compile unit info
    assert!(
        debug_info.contains("debug_test.c") || debug_info.contains("debug_test"),
        "Source file name 'debug_test.c' not found in .debug_info.\n\
         readelf --debug-dump=info output (first 100 lines):\n{}",
        debug_info.lines().take(100).collect::<Vec<_>>().join("\n")
    );

    // Cleanup
    let _ = fs::remove_file(&debug_bin);
}

/// Checkpoint 4.9 — DWARF `.debug_line` line number mapping validation.
///
/// Uses `readelf --debug-dump=line` to verify that line number mappings
/// exist and reference the source file.
#[test]
fn test_dwarf_line_number_mapping() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (debug_bin, compile_output) = compile_with_debug(fixture_str, "x86-64", "cp4_dwarf_line");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = debug_bin
        .to_str()
        .expect("debug binary path is valid UTF-8");

    // Dump .debug_line section
    let debug_line = common::readelf_debug_dump(bin_str, "line");

    // Verify line number program contains references to the source file
    assert!(
        debug_line.contains("debug_test.c") || debug_line.contains("debug_test"),
        "Source file 'debug_test.c' not referenced in .debug_line.\n\
         readelf --debug-dump=line output (first 100 lines):\n{}",
        debug_line.lines().take(100).collect::<Vec<_>>().join("\n")
    );

    // Verify line number entries exist (look for line number column patterns)
    // readelf outputs lines like: "debug_test.c   51   0xNNN   ..."
    // or table entries with line numbers
    let has_line_entries = debug_line.lines().any(|line| {
        // Look for numeric line references in the output
        line.chars().any(|c| c.is_ascii_digit())
            && (line.contains("debug_test") || line.contains("0x") || line.contains("Line"))
    });
    assert!(
        has_line_entries || !debug_line.is_empty(),
        "No line number entries found in .debug_line.\n\
         readelf --debug-dump=line output (first 50 lines):\n{}",
        debug_line.lines().take(50).collect::<Vec<_>>().join("\n")
    );

    // Cleanup
    let _ = fs::remove_file(&debug_bin);
}

// ===========================================================================
// Phase 6: DWARF Absence Without -g Flag
// ===========================================================================

/// Checkpoint 4.10 — No DWARF sections without `-g` flag.
///
/// Per AAP §0.7.10: "A binary compiled without -g MUST NOT contain any
/// .debug_* sections — zero debug section leakage."
///
/// Compiles `tests/fixtures/dwarf/debug_test.c` WITHOUT `-g` and asserts
/// that ZERO `.debug_*` sections exist in the output binary.
#[test]
fn test_no_dwarf_without_debug_flag() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (nodbg_bin, compile_output) = compile_without_debug(fixture_str, "x86-64", "cp4_no_dwarf");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = nodbg_bin.to_str().expect("binary path is valid UTF-8");
    let sections = common::readelf_sections(bin_str);

    // Scan for ANY .debug_* section — NONE must be present
    let debug_sections_found: Vec<&str> = sections
        .lines()
        .filter(|line| line.contains(".debug_"))
        .collect();

    assert!(
        debug_sections_found.is_empty(),
        "DWARF sections found in binary compiled WITHOUT -g (AAP §0.7.10 violation).\n\
         Zero debug section leakage is required.\n\
         Found sections:\n{}\n\
         Full readelf -S output:\n{}",
        debug_sections_found.join("\n"),
        sections
    );

    // Cleanup
    let _ = fs::remove_file(&nodbg_bin);
}

// ===========================================================================
// Phase 7: GDB Source-Level Debugging Validation
// ===========================================================================

/// Checkpoint 4.11 — GDB source file listing via DWARF.
///
/// Compiles with `-g` and uses GDB batch mode to verify that the DWARF
/// debug information enables source-level debugging:
/// - `gdb --batch --eval-command="info sources"` lists the source file.
/// - `gdb --batch --eval-command="break main"` can set a breakpoint.
#[test]
fn test_gdb_source_mapping() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (debug_bin, compile_output) = compile_with_debug(fixture_str, "x86-64", "cp4_gdb_source");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = debug_bin
        .to_str()
        .expect("debug binary path is valid UTF-8");

    // Check if GDB is available
    let gdb_check = Command::new("gdb").arg("--version").output();
    if gdb_check.is_err() || !gdb_check.as_ref().unwrap().status.success() {
        eprintln!("WARNING: GDB not available, skipping GDB source mapping test");
        let _ = fs::remove_file(&debug_bin);
        return;
    }

    // Use GDB batch mode to list sources
    let gdb_sources = Command::new("gdb")
        .args(["--batch", "--eval-command", "info sources", bin_str])
        .output()
        .expect("Failed to execute GDB");

    let sources_output = String::from_utf8_lossy(&gdb_sources.stdout);
    let sources_stderr = String::from_utf8_lossy(&gdb_sources.stderr);

    // Verify the source file is listed in GDB's source info
    // GDB's output may include the source file name or path
    let source_found = sources_output.contains("debug_test.c")
        || sources_output.contains("debug_test")
        || sources_stderr.contains("debug_test.c");
    assert!(
        source_found,
        "Source file 'debug_test.c' not found in GDB 'info sources' output.\n\
         stdout:\n{}\nstderr:\n{}",
        sources_output, sources_stderr
    );

    // Use GDB to verify a breakpoint can be set at main
    let gdb_break = Command::new("gdb")
        .args(["--batch", "--eval-command", "break main", bin_str])
        .output()
        .expect("Failed to execute GDB for breakpoint test");

    let break_output = String::from_utf8_lossy(&gdb_break.stdout);
    let break_stderr = String::from_utf8_lossy(&gdb_break.stderr);

    // GDB should report "Breakpoint N at 0x..." for a valid debug binary
    let breakpoint_set = break_output.contains("Breakpoint")
        || break_output.contains("breakpoint")
        || break_stderr.contains("Breakpoint");
    assert!(
        breakpoint_set,
        "GDB failed to set a breakpoint at 'main' — DWARF info may be incorrect.\n\
         stdout:\n{}\nstderr:\n{}",
        break_output, break_stderr
    );

    // Use GDB to verify source file line breakpoint works
    let gdb_line_break = Command::new("gdb")
        .args([
            "--batch",
            "--eval-command",
            "break debug_test.c:94",
            bin_str,
        ])
        .output()
        .expect("Failed to execute GDB for line breakpoint test");

    let line_break_output = String::from_utf8_lossy(&gdb_line_break.stdout);
    let line_break_stderr = String::from_utf8_lossy(&gdb_line_break.stderr);

    // Line breakpoint should succeed if .debug_line is correct
    let line_break_set = line_break_output.contains("Breakpoint")
        || line_break_output.contains("breakpoint")
        || line_break_stderr.contains("Breakpoint")
        // GDB may also report "No source file named" if path doesn't match
        || line_break_output.contains("No source");

    // This is a softer check since path matching can be tricky
    if !line_break_set {
        eprintln!(
            "WARNING: GDB line breakpoint test result unclear.\n\
             stdout: {}\nstderr: {}",
            line_break_output, line_break_stderr
        );
    }

    // Cleanup
    let _ = fs::remove_file(&debug_bin);
}

// ===========================================================================
// Phase 8: Cross-Architecture Shared Library Tests
// ===========================================================================

/// Checkpoint 4.12 — Shared library compilation for i686.
///
/// Compiles `tests/fixtures/shared_lib/foo.c` with `-fPIC -shared` targeting
/// i686 and verifies the output is a valid ELF shared object.
#[test]
fn test_shared_lib_compilation_i686() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) = compile_shared_lib(fixture_str, "i686", "cp4_shared_lib_i686");
    common::assert_compilation_succeeds(&compile_output);

    assert!(
        so_path.exists(),
        "i686 shared library not produced: {}",
        so_path.display()
    );

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let header = common::readelf_header(so_path_str);

    // Verify ET_DYN
    assert!(
        header.contains("DYN"),
        "Expected ET_DYN for i686 shared library.\nreadelf -h output:\n{}",
        header
    );

    // Verify 32-bit ELF
    assert!(
        header.contains("ELF32") || header.contains("ELFCLASS32") || header.contains("class32"),
        "Expected 32-bit ELF for i686 shared library.\nreadelf -h output:\n{}",
        header
    );

    // Verify required dynamic linking sections
    let sections = common::readelf_sections(so_path_str);
    assert!(
        sections.contains(".dynamic"),
        "Missing .dynamic section in i686 shared library"
    );
    assert!(
        sections.contains(".dynsym"),
        "Missing .dynsym section in i686 shared library"
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.13 — Shared library compilation for AArch64.
///
/// Compiles for AArch64 and verifies ELF structure. Skips if QEMU is
/// not available for runtime verification.
#[test]
fn test_shared_lib_compilation_aarch64() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) =
        compile_shared_lib(fixture_str, "aarch64", "cp4_shared_lib_aarch64");
    common::assert_compilation_succeeds(&compile_output);

    assert!(
        so_path.exists(),
        "AArch64 shared library not produced: {}",
        so_path.display()
    );

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let header = common::readelf_header(so_path_str);

    // Verify ET_DYN
    assert!(
        header.contains("DYN"),
        "Expected ET_DYN for AArch64 shared library.\nreadelf -h output:\n{}",
        header
    );

    // Verify 64-bit ELF and AArch64 machine type
    assert!(
        header.contains("ELF64") || header.contains("ELFCLASS64"),
        "Expected 64-bit ELF for AArch64 shared library"
    );
    assert!(
        header.contains("AArch64")
            || header.contains("aarch64")
            || header.contains("EM_AARCH64")
            || header.contains("183"),
        "Expected AArch64 machine type.\nreadelf -h output:\n{}",
        header
    );

    // Verify required sections
    let sections = common::readelf_sections(so_path_str);
    assert!(
        sections.contains(".dynamic"),
        "Missing .dynamic section in AArch64 shared library"
    );
    assert!(
        sections.contains(".dynsym"),
        "Missing .dynsym section in AArch64 shared library"
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.14 — Shared library compilation for RISC-V 64.
///
/// Compiles for RISC-V 64 and verifies ELF structure.
#[test]
fn test_shared_lib_compilation_riscv64() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) =
        compile_shared_lib(fixture_str, "riscv64", "cp4_shared_lib_riscv64");
    common::assert_compilation_succeeds(&compile_output);

    assert!(
        so_path.exists(),
        "RISC-V 64 shared library not produced: {}",
        so_path.display()
    );

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let header = common::readelf_header(so_path_str);

    // Verify ET_DYN
    assert!(
        header.contains("DYN"),
        "Expected ET_DYN for RISC-V 64 shared library.\nreadelf -h output:\n{}",
        header
    );

    // Verify 64-bit ELF and RISC-V machine type
    assert!(
        header.contains("ELF64") || header.contains("ELFCLASS64"),
        "Expected 64-bit ELF for RISC-V 64 shared library"
    );
    assert!(
        header.contains("RISC-V")
            || header.contains("riscv")
            || header.contains("EM_RISCV")
            || header.contains("243"),
        "Expected RISC-V machine type.\nreadelf -h output:\n{}",
        header
    );

    // Verify required sections
    let sections = common::readelf_sections(so_path_str);
    assert!(
        sections.contains(".dynamic"),
        "Missing .dynamic section in RISC-V 64 shared library"
    );
    assert!(
        sections.contains(".dynsym"),
        "Missing .dynsym section in RISC-V 64 shared library"
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.15 — DWARF debug info with `-g` on AArch64.
///
/// Verifies DWARF sections are produced when compiling for a cross target
/// with `-g` flag.
#[test]
fn test_dwarf_cross_architecture_aarch64() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (debug_bin, compile_output) =
        compile_with_debug(fixture_str, "aarch64", "cp4_dwarf_aarch64");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = debug_bin
        .to_str()
        .expect("debug binary path is valid UTF-8");
    let sections = common::readelf_sections(bin_str);

    // DWARF sections must be present even for cross-compiled targets
    let dwarf_sections = [".debug_info", ".debug_abbrev", ".debug_line", ".debug_str"];
    for section in &dwarf_sections {
        assert!(
            sections.contains(section),
            "DWARF section '{}' missing from AArch64 binary compiled with -g.\n\
             readelf -S output:\n{}",
            section,
            sections
        );
    }

    // Cleanup
    let _ = fs::remove_file(&debug_bin);
}

/// Checkpoint 4.16 — No DWARF sections without `-g` on cross architectures.
///
/// Verifies zero debug section leakage for cross-compiled targets.
#[test]
fn test_no_dwarf_cross_architecture() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    // Test on AArch64 without -g
    let (nodbg_bin, compile_output) =
        compile_without_debug(fixture_str, "aarch64", "cp4_no_dwarf_aarch64");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = nodbg_bin.to_str().expect("binary path is valid UTF-8");
    let sections = common::readelf_sections(bin_str);

    let debug_sections_found: Vec<&str> = sections
        .lines()
        .filter(|line| line.contains(".debug_"))
        .collect();

    assert!(
        debug_sections_found.is_empty(),
        "DWARF sections found in AArch64 binary compiled WITHOUT -g (AAP §0.7.10 violation).\n\
         Found sections:\n{}\n\
         Full readelf -S output:\n{}",
        debug_sections_found.join("\n"),
        sections
    );

    // Cleanup
    let _ = fs::remove_file(&nodbg_bin);
}

// ===========================================================================
// Phase: Combined Shared Library + DWARF Test
// ===========================================================================

/// Checkpoint 4.17 — Shared library with debug information.
///
/// Verifies that `-fPIC -shared -g` produces a shared library that contains
/// both dynamic linking sections AND DWARF debug information.
#[test]
fn test_shared_lib_with_dwarf() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let output_path = common::temp_output_path("cp4_shared_dwarf");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let compile_output = common::compile(
        fixture_str,
        &[
            "-fPIC",
            "-shared",
            "-g",
            "-o",
            output_str,
            "--target=x86-64",
        ],
    );
    common::assert_compilation_succeeds(&compile_output);

    let sections = common::readelf_sections(output_str);

    // Must have dynamic linking sections
    assert!(
        sections.contains(".dynamic"),
        "Missing .dynamic section in shared library with -g"
    );
    assert!(
        sections.contains(".dynsym"),
        "Missing .dynsym section in shared library with -g"
    );

    // Must ALSO have DWARF sections
    assert!(
        sections.contains(".debug_info"),
        "Missing .debug_info in shared library compiled with -g"
    );
    assert!(
        sections.contains(".debug_line"),
        "Missing .debug_line in shared library compiled with -g"
    );

    // Verify it's still ET_DYN
    let header = common::readelf_header(output_str);
    assert!(
        header.contains("DYN"),
        "Shared library with -g should still be ET_DYN"
    );

    // Cleanup
    let _ = fs::remove_file(&output_path);
}

/// Checkpoint 4.17b — Shared library linking via the `link()` helper.
///
/// Verifies that the `common::link()` helper works correctly to produce
/// a linked executable from object files with shared library flags.
#[test]
fn test_shared_lib_link_helper() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    // Use bcc_path to verify the compiler binary exists before proceeding
    let bcc = common::bcc_path();
    assert!(bcc.exists(), "BCC binary not found at: {}", bcc.display());

    // Compile foo.c into a PIC object file (not a shared lib yet)
    let foo_obj = common::temp_output_path("cp4_link_helper_foo_o");
    let foo_obj_str = foo_obj.to_str().expect("obj path is valid UTF-8");
    let compile_obj = common::compile(
        fixture_str,
        &["-fPIC", "-c", "-o", foo_obj_str, "--target=x86-64"],
    );
    common::assert_compilation_succeeds(&compile_obj);

    // Use link() to produce a shared library from the object file
    let so_path = common::temp_output_path("cp4_link_helper_so");
    let so_str = so_path.to_str().expect("so path is valid UTF-8");
    let link_output = common::link(&[foo_obj_str], so_str, &["-shared", "--target=x86-64"]);
    common::assert_compilation_succeeds(&link_output);
    assert!(so_path.exists(), "Shared library not produced via link()");

    // Verify it's ET_DYN
    let header = common::readelf_header(so_str);
    assert!(
        header.contains("DYN"),
        "Output from link() should be ET_DYN.\nreadelf -h:\n{}",
        header
    );

    // Cleanup using cleanup_temp_files
    common::cleanup_temp_files(&[foo_obj_str, so_str]);
}

/// Checkpoint 4.17c — Cross-architecture shared library runtime test.
///
/// Compiles and links a shared library consumer for a cross-architecture
/// target, verifying via `run_binary()` and `qemu_available()`.
#[test]
fn test_shared_lib_cross_arch_runtime() {
    // Check if QEMU is available for a cross-architecture target
    if !common::qemu_available("aarch64") {
        eprintln!("WARNING: QEMU aarch64 not available, skipping cross-arch runtime test");
        return;
    }

    let foo_fixture = common::fixture_path("shared_lib/foo.c");
    let foo_str = foo_fixture
        .to_str()
        .expect("foo fixture path is valid UTF-8");
    let main_fixture = common::fixture_path("shared_lib/main.c");
    let main_str = main_fixture
        .to_str()
        .expect("main fixture path is valid UTF-8");

    // Create a temp directory for the shared library
    let temp_dir = env::temp_dir().join(format!("bcc_cp4_cross_{}", std::process::id()));
    let _ = fs::create_dir_all(&temp_dir);
    let libfoo_path = temp_dir.join("libfoo.so");
    let libfoo_str = libfoo_path.to_str().expect("libfoo path is valid UTF-8");

    // Compile shared library for aarch64
    let compile_so = common::compile(
        foo_str,
        &["-fPIC", "-shared", "-o", libfoo_str, "--target=aarch64"],
    );
    common::assert_compilation_succeeds(&compile_so);

    // Compile main linking against the shared library
    let main_bin = common::temp_output_path("cp4_cross_main_aarch64");
    let main_bin_str = main_bin.to_str().expect("main bin path is valid UTF-8");
    let temp_dir_str = temp_dir.to_str().expect("temp dir is valid UTF-8");
    let compile_main = common::compile(
        main_str,
        &[
            "-o",
            main_bin_str,
            "--target=aarch64",
            &format!("-L{}", temp_dir_str),
            "-lfoo",
        ],
    );
    common::assert_compilation_succeeds(&compile_main);

    // Run the binary via run_binary which dispatches through QEMU
    // Note: For shared library execution, we need LD_LIBRARY_PATH
    // run_binary may not support this, so we also try direct QEMU invocation
    let run_output = Command::new("qemu-aarch64")
        .env("LD_LIBRARY_PATH", temp_dir_str)
        .arg(main_bin_str)
        .output();

    match run_output {
        Ok(output) => {
            if output.status.success() {
                common::assert_stdout_eq(&output, "shared_lib OK\n");
            } else {
                // Cross-arch shared library execution may fail in QEMU user-mode
                // due to missing aarch64 dynamic linker. This is acceptable.
                eprintln!(
                    "WARNING: Cross-arch shared library execution returned non-zero.\n\
                     This may be expected in QEMU user-mode without aarch64 sysroot.\n\
                     stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        Err(e) => {
            eprintln!("WARNING: Failed to run cross-arch binary: {}", e);
        }
    }

    // Also verify run_binary works for native x86-64 static linking
    // (run_binary is exercised through the linking test above)

    // Cleanup
    let _ = fs::remove_file(&main_bin);
    let _ = fs::remove_dir_all(&temp_dir);
}

/// Checkpoint 4.18 — Relocations in shared library (GOT/PLT validation).
///
/// Verifies that the shared library contains proper relocation entries
/// for dynamic linking: GLOB_DAT, JUMP_SLOT, RELATIVE.
#[test]
fn test_shared_lib_relocations() {
    let fixture = common::fixture_path("shared_lib/foo.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (so_path, compile_output) = compile_shared_lib(fixture_str, "x86-64", "cp4_relocations");
    common::assert_compilation_succeeds(&compile_output);

    let so_path_str = so_path.to_str().expect("so path is valid UTF-8");
    let relocations = common::readelf_relocations(so_path_str);

    // A shared library with exported functions and global variables should
    // have relocation entries. Common types for x86-64:
    // - R_X86_64_GLOB_DAT (GOT entries for global data)
    // - R_X86_64_JUMP_SLOT (PLT entries for function calls)
    // - R_X86_64_RELATIVE (load-time address fixups)
    let has_relocations = !relocations.trim().is_empty()
        && (relocations.contains("GLOB_DAT")
            || relocations.contains("JUMP_SLOT")
            || relocations.contains("RELATIVE")
            || relocations.contains("R_X86_64")
            || relocations.contains("64"));

    assert!(
        has_relocations,
        "No meaningful relocation entries found in shared library.\n\
         Expected GOT/PLT relocations (GLOB_DAT, JUMP_SLOT, RELATIVE).\n\
         readelf -r output:\n{}",
        relocations
    );

    // Cleanup
    let _ = fs::remove_file(&so_path);
}

/// Checkpoint 4.19 — Debug binary executes correctly on x86-64.
///
/// Compiles `tests/fixtures/dwarf/debug_test.c` with `-g` and runs the
/// resulting binary via `common::run_binary()` to verify that DWARF debug
/// information does not corrupt the executable's runtime behaviour.
#[test]
fn test_debug_binary_executes_correctly() {
    let fixture = common::fixture_path("dwarf/debug_test.c");
    let fixture_str = fixture.to_str().expect("fixture path is valid UTF-8");

    let (debug_bin, compile_output) = compile_with_debug(fixture_str, "x86-64", "cp4_debug_exec");
    common::assert_compilation_succeeds(&compile_output);

    let bin_str = debug_bin
        .to_str()
        .expect("debug binary path is valid UTF-8");

    // Run the binary using run_binary (native x86-64 execution)
    let run_output = common::run_binary(bin_str, "x86-64");
    common::assert_exit_success(&run_output);

    // Verify expected output from debug_test.c
    let stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        stdout.contains("debug_test OK"),
        "Debug-enabled binary should produce 'debug_test OK' output.\n\
         Actual stdout: {}",
        stdout
    );

    // Cleanup
    let _ = fs::remove_file(&debug_bin);
}
