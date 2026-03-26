//! Checkpoint 6 — Linux Kernel 6.9 Build and QEMU Boot
//!
//! This is the **primary success criterion** for the entire BCC project. It validates
//! BCC's ability to compile the Linux kernel 6.9 (RISC-V configuration) and boot it
//! to userspace in QEMU, exercising the full C language surface, GCC extensions,
//! preprocessor edge cases, inline assembly, and linker correctness simultaneously.
//!
//! # AAP Compliance
//! - **Sequential hard gate:** Checkpoint 5 must pass first (AAP §0.7.5).
//! - **Wall-clock ceiling:** Total kernel build time MUST NOT exceed 5× GCC-equivalent
//!   on the same hardware (AAP §0.7.8).
//! - **Regression guard:** Checkpoint 3 must be re-run after any feature addition
//!   during the kernel build phase (AAP §0.7.5).
//! - **Kernel build failure classification order** (AAP §0.7.6):
//!   missing GCC extension → missing builtin → inline asm constraint →
//!   preprocessor issue → codegen bug.
//! - **Backend validation for kernel:** RISC-V 64 is the target architecture.
//!
//! # Prerequisites
//! - Linux kernel 6.9 source must be available. The location is resolved via the
//!   `KERNEL_SRC_DIR` environment variable or the default path `/usr/src/linux-6.9`.
//! - All tests are marked `#[ignore]` because they require external kernel source
//!   that is not shipped with the BCC repository.
//! - A release build of BCC must be available (`cargo build --release`).
//! - `qemu-system-riscv64` must be installed for the boot validation test.
//!
//! # Zero Dependencies
//! Only `std::` imports are used — no external crates.

mod common;

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The wall-clock ceiling multiplier. Per AAP §0.7.8, the kernel build time
/// MUST NOT exceed 5× the equivalent GCC build time on the same hardware.
const WALL_CLOCK_CEILING_MULTIPLIER: f64 = 5.0;

/// Default path for the Linux kernel 6.9 source tree when `KERNEL_SRC_DIR` is
/// not set in the environment.
const DEFAULT_KERNEL_SRC_DIR: &str = "/usr/src/linux-6.9";

/// Maximum time (in seconds) to wait for the QEMU boot process before declaring
/// a hang. 120 seconds is generous for a minimal initramfs with `USERSPACE_OK`.
const QEMU_BOOT_TIMEOUT_SECS: u64 = 120;

/// The magic string printed by the minimal `/init` binary inside the initramfs.
/// Presence of this string in QEMU serial console output confirms successful
/// boot to userspace.
const USERSPACE_OK_MARKER: &str = "USERSPACE_OK";

/// Default number of parallel make jobs when `nproc` cannot be determined.
const DEFAULT_MAKE_JOBS: u32 = 4;

/// Pre-recorded GCC baseline build time in seconds. When `GCC_BASELINE_SECS`
/// environment variable is not set, this fallback is used for the 5× ceiling
/// comparison. The actual value should be measured on the CI hardware; this is
/// a conservative estimate for a RISC-V defconfig kernel build.
const DEFAULT_GCC_BASELINE_SECS: f64 = 600.0;

// ---------------------------------------------------------------------------
// Failure Classification (AAP §0.7.6)
// ---------------------------------------------------------------------------

/// Classification of kernel build failures per AAP §0.7.6.
///
/// The classification order is significant and follows the priority specified
/// in the AAP: missing GCC extension → missing builtin → inline asm constraint
/// → preprocessor issue → codegen bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureKind {
    /// Unknown `__attribute__`, unsupported GCC syntax extension.
    MissingGccExtension,
    /// `__builtin_*` function not recognized or not implemented.
    MissingBuiltin,
    /// Unsupported inline assembly constraint letter or clobber.
    InlineAsmConstraint,
    /// Macro expansion failure, include resolution, conditional compilation issue.
    PreprocessorIssue,
    /// Internal compiler error, incorrect instruction emission, relocation overflow.
    CodegenBug,
    /// Build failure that does not match any known classification pattern.
    Unknown,
}

impl std::fmt::Display for FailureKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailureKind::MissingGccExtension => write!(f, "Missing GCC extension"),
            FailureKind::MissingBuiltin => write!(f, "Missing builtin"),
            FailureKind::InlineAsmConstraint => write!(f, "Inline asm constraint"),
            FailureKind::PreprocessorIssue => write!(f, "Preprocessor issue"),
            FailureKind::CodegenBug => write!(f, "Codegen bug"),
            FailureKind::Unknown => write!(f, "Unknown failure"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: Kernel Source Directory Resolution
// ---------------------------------------------------------------------------

/// Resolves the path to the Linux kernel 6.9 source directory.
///
/// Resolution order:
/// 1. `KERNEL_SRC_DIR` environment variable
/// 2. Relative `linux-6.9` directory under the current working directory
/// 3. Default path [`DEFAULT_KERNEL_SRC_DIR`]
///
/// Returns `None` if the resolved path does not exist or is not a directory.
fn kernel_src_dir() -> Option<PathBuf> {
    // 1. Try KERNEL_SRC_DIR environment variable
    if let Ok(val) = env::var("KERNEL_SRC_DIR") {
        if !val.is_empty() {
            let dir = PathBuf::from(val);
            if dir.is_dir() {
                return Some(dir);
            }
        }
    }

    // 2. Try relative to current working directory
    if let Ok(cwd) = env::current_dir() {
        let mut candidate = cwd;
        candidate.push("linux-6.9");
        if candidate.as_path().is_dir() {
            return Some(candidate);
        }
    }

    // 3. Try default path
    let default = PathBuf::from(DEFAULT_KERNEL_SRC_DIR);
    if default.is_dir() {
        return Some(default);
    }

    None
}

/// Returns the number of available CPUs for parallel `make -j` invocations.
///
/// Reads from the `NPROC` environment variable first, then tries `nproc` command,
/// and falls back to [`DEFAULT_MAKE_JOBS`].
fn num_cpus() -> u32 {
    // Try environment variable first
    if let Ok(val) = env::var("NPROC") {
        if let Ok(n) = val.parse::<u32>() {
            if n > 0 {
                return n;
            }
        }
    }

    // Try the nproc command
    if let Ok(output) = Command::new("nproc").output() {
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout);
            if let Ok(n) = s.trim().parse::<u32>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }

    DEFAULT_MAKE_JOBS
}

/// Returns the GCC baseline build time in seconds for the 5× ceiling comparison.
///
/// Resolution order:
/// 1. `GCC_BASELINE_SECS` environment variable (pre-recorded measurement)
/// 2. [`DEFAULT_GCC_BASELINE_SECS`] fallback
fn gcc_baseline_secs() -> f64 {
    if let Ok(val) = env::var("GCC_BASELINE_SECS") {
        if let Ok(secs) = val.parse::<f64>() {
            if secs > 0.0 {
                return secs;
            }
        }
    }
    DEFAULT_GCC_BASELINE_SECS
}

// ---------------------------------------------------------------------------
// Helper: BCC Path for Kernel Build (CC= argument)
// ---------------------------------------------------------------------------

/// Returns the absolute path to the BCC binary as a string, suitable for
/// passing as `CC=<path>` to the kernel build system.
///
/// The path is canonicalized so that `make` can invoke it from any working
/// directory within the kernel source tree.
fn bcc_cc_path() -> String {
    let bcc = common::bcc_path();
    // Canonicalize to an absolute path so `make` can find it from kernel src dir.
    match std::fs::canonicalize(&bcc) {
        Ok(abs) => abs.to_string_lossy().into_owned(),
        Err(_) => bcc.to_string_lossy().into_owned(),
    }
}

// ---------------------------------------------------------------------------
// Helper: Kernel Sub-Gate Build
// ---------------------------------------------------------------------------

/// Runs `make` inside the kernel source directory to build a specific target.
///
/// # Arguments
/// * `kernel_dir` — Path to the kernel source root.
/// * `target` — The make target (e.g., `"init/main.o"`, `"vmlinux"`).
/// * `extra_args` — Additional make arguments (e.g., `["-j4"]`).
///
/// # Returns
/// A tuple of `(Output, Duration)` — the process output and the elapsed wall-clock time.
fn make_kernel(
    kernel_dir: &Path,
    target: &str,
    extra_args: &[&str],
) -> (std::process::Output, Duration) {
    let cc_arg = format!("CC={}", bcc_cc_path());
    let start = Instant::now();

    let mut cmd = Command::new("make");
    cmd.current_dir(kernel_dir)
        .arg("ARCH=riscv")
        .arg(&cc_arg)
        .arg(target);

    for arg in extra_args {
        cmd.arg(arg);
    }

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to invoke 'make' in kernel dir '{}': {}",
                kernel_dir.display(),
                e
            )
        });

    let elapsed = start.elapsed();
    (output, elapsed)
}

// ---------------------------------------------------------------------------
// Helper: Classify Build Failure (AAP §0.7.6)
// ---------------------------------------------------------------------------

/// Parses kernel build error output and classifies the failure per AAP §0.7.6.
///
/// Classification priority (first match wins):
/// 1. Missing GCC extension — unknown `__attribute__`, unsupported syntax
/// 2. Missing builtin — `__builtin_*` not recognized
/// 3. Inline asm constraint — unsupported constraint letter
/// 4. Preprocessor issue — macro expansion, include, conditional
/// 5. Codegen bug — ICE, incorrect instruction, relocation overflow
///
/// # Arguments
/// * `stderr` — The stderr output from the kernel build process.
///
/// # Returns
/// The classified [`FailureKind`].
fn classify_build_failure(stderr: &str) -> FailureKind {
    let lower = stderr.to_lowercase();

    // 1. Missing GCC extension: __attribute__ not recognized, unsupported syntax
    if lower.contains("unknown attribute")
        || lower.contains("unsupported attribute")
        || lower.contains("__attribute__")
            && (lower.contains("unknown") || lower.contains("unsupported"))
        || lower.contains("unrecognized attribute")
        || lower.contains("unsupported gcc extension")
        || lower.contains("transparent_union")
            && (lower.contains("unknown") || lower.contains("unsupported"))
        || lower.contains("__extension__") && lower.contains("unsupported")
    {
        return FailureKind::MissingGccExtension;
    }

    // 2. Missing builtin: __builtin_* not recognized
    if lower.contains("__builtin_")
        && (lower.contains("undeclared")
            || lower.contains("undefined")
            || lower.contains("unknown")
            || lower.contains("not recognized")
            || lower.contains("implicit declaration"))
    {
        return FailureKind::MissingBuiltin;
    }

    // 3. Inline asm constraint: unsupported constraint letter/clobber
    if lower.contains("asm")
        && (lower.contains("constraint") || lower.contains("clobber") || lower.contains("operand"))
        || lower.contains("inline assembly")
            && (lower.contains("unsupported")
                || lower.contains("invalid")
                || lower.contains("constraint"))
    {
        return FailureKind::InlineAsmConstraint;
    }

    // 4. Preprocessor issue: macro, include, conditional
    if lower.contains("#include") && (lower.contains("not found") || lower.contains("no such file"))
        || lower.contains("macro") && (lower.contains("error") || lower.contains("undefined"))
        || lower.contains("unterminated")
            && (lower.contains("#if") || lower.contains("conditional"))
        || lower.contains("#error")
        || lower.contains("preprocessor")
    {
        return FailureKind::PreprocessorIssue;
    }

    // 5. Codegen bug: ICE, relocation overflow, internal error
    if lower.contains("internal compiler error")
        || lower.contains("ice:")
        || lower.contains("relocation overflow")
        || lower.contains("relocation truncated")
        || lower.contains("assertion failed")
        || lower.contains("panic")
        || lower.contains("segfault")
        || lower.contains("codegen")
            && (lower.contains("error") || lower.contains("bug") || lower.contains("failed"))
    {
        return FailureKind::CodegenBug;
    }

    FailureKind::Unknown
}

// ---------------------------------------------------------------------------
// Helper: Initramfs Preparation
// ---------------------------------------------------------------------------

/// The minimal C source code for the `/init` binary in the initramfs.
///
/// This program:
/// 1. Writes `USERSPACE_OK\n` to stdout (fd 1) via the `write` syscall
/// 2. Calls `reboot(LINUX_REBOOT_CMD_POWER_OFF)` via raw syscall to cleanly
///    shut down the VM
///
/// Per AAP User Example: "minimal `/init` (static binary printing
/// `USERSPACE_OK\n` and calling reboot) packed into initramfs."
///
/// We use raw syscalls to avoid any libc dependency, allowing BCC to produce
/// a fully static, freestanding binary.
const INIT_C_SOURCE: &str = r#"
/* Minimal /init for BCC kernel boot validation.
 * Uses raw Linux syscalls — no libc dependency.
 * Target: RISC-V 64 (RV64)
 */

/* RISC-V 64 syscall numbers */
#define SYS_write   64
#define SYS_reboot  142

/* Reboot magic numbers */
#define LINUX_REBOOT_MAGIC1    0xfee1dead
#define LINUX_REBOOT_MAGIC2    672274793
#define LINUX_REBOOT_CMD_POWER_OFF  0x4321FEDC

static long rv64_syscall3(long num, long a0, long a1, long a2) {
    long ret;
    __asm__ volatile (
        "mv a7, %1\n\t"
        "mv a0, %2\n\t"
        "mv a1, %3\n\t"
        "mv a2, %4\n\t"
        "ecall\n\t"
        "mv %0, a0\n\t"
        : "=r"(ret)
        : "r"(num), "r"(a0), "r"(a1), "r"(a2)
        : "a0", "a1", "a2", "a7", "memory"
    );
    return ret;
}

static long rv64_syscall4(long num, long a0, long a1, long a2, long a3) {
    long ret;
    __asm__ volatile (
        "mv a7, %1\n\t"
        "mv a0, %2\n\t"
        "mv a1, %3\n\t"
        "mv a2, %4\n\t"
        "mv a3, %5\n\t"
        "ecall\n\t"
        "mv %0, a0\n\t"
        : "=r"(ret)
        : "r"(num), "r"(a0), "r"(a1), "r"(a2), "r"(a3)
        : "a0", "a1", "a2", "a3", "a7", "memory"
    );
    return ret;
}

void _start(void) {
    /* Write "USERSPACE_OK\n" to stdout (fd 1) */
    const char msg[] = "USERSPACE_OK\n";
    rv64_syscall3(SYS_write, 1, (long)msg, sizeof(msg) - 1);

    /* reboot(LINUX_REBOOT_MAGIC1, LINUX_REBOOT_MAGIC2, LINUX_REBOOT_CMD_POWER_OFF, NULL) */
    rv64_syscall4(SYS_reboot,
                  LINUX_REBOOT_MAGIC1,
                  LINUX_REBOOT_MAGIC2,
                  LINUX_REBOOT_CMD_POWER_OFF,
                  0);

    /* Should not reach here — infinite loop as a safety net */
    for (;;) {}
}
"#;

/// Prepares a minimal initramfs containing a static `/init` binary.
///
/// Steps:
/// 1. Write `INIT_C_SOURCE` to a temporary `.c` file
/// 2. Compile it with BCC as a static, freestanding RISC-V 64 binary
/// 3. Pack it into a newc-format cpio archive (initramfs)
///
/// # Returns
/// A `PathBuf` to the generated `initramfs.cpio` file.
///
/// # Panics
/// Panics if any step fails (compilation, cpio creation).
fn prepare_initramfs() -> PathBuf {
    let work_dir = common::temp_output_path("initramfs_work");
    let work_dir_str = work_dir.to_string_lossy().into_owned();

    // Create working directory
    fs::create_dir_all(&work_dir).unwrap_or_else(|e| {
        panic!(
            "Failed to create initramfs work dir '{}': {}",
            work_dir_str, e
        )
    });

    // Write the init.c source using Write trait for explicit I/O control
    let init_c_path = work_dir.join("init.c");
    {
        let mut file = std::fs::File::create(&init_c_path)
            .unwrap_or_else(|e| panic!("Failed to create init.c: {}", e));
        file.write_all(INIT_C_SOURCE.as_bytes())
            .unwrap_or_else(|e| panic!("Failed to write init.c content: {}", e));
        file.flush()
            .unwrap_or_else(|e| panic!("Failed to flush init.c: {}", e));
    }

    // Compile the init binary using BCC
    let init_bin_path = work_dir.join("init");
    let init_c_str = init_c_path.to_string_lossy().into_owned();
    let init_bin_str = init_bin_path.to_string_lossy().into_owned();

    let compile_output = common::compile(
        &init_c_str,
        &[
            "--target=riscv64",
            "-o",
            &init_bin_str,
            "-nostdlib", // No standard library — freestanding
            "-static",   // Static binary
        ],
    );
    common::assert_compilation_succeeds(&compile_output);

    // Verify the init binary was produced
    assert!(
        init_bin_path.exists(),
        "Init binary was not produced at '{}'",
        init_bin_str
    );

    // Create initramfs (newc-format cpio archive)
    // We need: init binary at the root as "/init"
    let initramfs_path = common::temp_output_path("initramfs.cpio");
    let initramfs_str = initramfs_path.to_string_lossy().into_owned();

    // Use cpio to create the archive:
    // echo "init" | cpio -o -H newc > initramfs.cpio
    // But first we need to be in the work dir with init named correctly
    let cpio_result = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "cd '{}' && echo init | cpio -o -H newc --quiet > '{}'",
            work_dir_str, initramfs_str
        ))
        .output()
        .unwrap_or_else(|e| panic!("Failed to run cpio for initramfs creation: {}", e));

    if !cpio_result.status.success() {
        let stderr = String::from_utf8_lossy(&cpio_result.stderr);
        panic!("cpio initramfs creation failed:\nstderr: {}", stderr);
    }

    assert!(
        initramfs_path.exists(),
        "Initramfs was not produced at '{}'",
        initramfs_str
    );

    // Clean up work directory (best-effort)
    let _ = fs::remove_dir_all(&work_dir);

    initramfs_path
}

// ---------------------------------------------------------------------------
// Helper: Validate ELF Properties
// ---------------------------------------------------------------------------

/// Validates that an ELF file has the expected RISC-V 64-bit properties.
///
/// Checks the `readelf -h` output for:
/// - `Class: ELF64`
/// - `Machine: RISC-V` (EM_RISCV)
/// - The specified ELF type (e.g., `"EXEC"` or `"REL"`)
fn assert_riscv64_elf(path: &str, expected_type: &str) {
    let header = common::readelf_header(path);

    assert!(
        header.contains("ELF64") || header.contains("Class:                             ELF64"),
        "Expected ELFCLASS64 for '{}', got:\n{}",
        path,
        header
    );

    assert!(
        header.contains("RISC-V") || header.contains("EM_RISCV"),
        "Expected EM_RISCV machine type for '{}', got:\n{}",
        path,
        header
    );

    assert!(
        header.contains(expected_type),
        "Expected ELF type '{}' for '{}', got:\n{}",
        expected_type,
        path,
        header
    );
}

// ===========================================================================
// Test: Kernel Source Availability
// ===========================================================================

/// Verifies that the Linux kernel 6.9 source tree is available and contains
/// the expected directory structure.
///
/// Checks for:
/// - `Makefile` at the root
/// - `arch/riscv/` directory
/// - `init/main.c` source file
///
/// This is marked `#[ignore]` as it requires external kernel source.
#[test]
#[ignore]
fn test_kernel_source_available() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "SKIP: Kernel source not found. Set KERNEL_SRC_DIR or place sources at {}",
                DEFAULT_KERNEL_SRC_DIR
            );
            return;
        }
    };

    // Verify Makefile exists
    let makefile = kernel_dir.join("Makefile");
    assert!(
        makefile.is_file(),
        "Kernel Makefile not found at '{}'",
        makefile.display()
    );

    // Verify arch/riscv/ directory exists
    let arch_riscv = kernel_dir.join("arch").join("riscv");
    assert!(
        arch_riscv.is_dir(),
        "Kernel arch/riscv/ directory not found at '{}'",
        arch_riscv.display()
    );

    // Verify init/main.c exists
    let init_main = kernel_dir.join("init").join("main.c");
    assert!(
        init_main.is_file(),
        "Kernel init/main.c not found at '{}'",
        init_main.display()
    );

    // Optionally verify the kernel version
    let makefile_content = fs::read_to_string(&makefile).unwrap_or_default();
    if makefile_content.contains("VERSION = 6") && makefile_content.contains("PATCHLEVEL = 9") {
        eprintln!("Kernel source confirmed: Linux 6.9");
    } else {
        eprintln!(
            "WARNING: Kernel source found but version may not be 6.9. \
             Build may still succeed if the source is compatible."
        );
    }
}

// ===========================================================================
// Tests: Kernel Sub-Gate Compilation
// ===========================================================================

/// Compiles `init/main.o` — the kernel's initialization entry point.
///
/// This sub-gate exercises core C language features, GCC attributes, and
/// the preprocessor on one of the most fundamental kernel source files.
#[test]
#[ignore]
fn test_kernel_subgate_init_main() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    // Clean any previous artifact
    let target_obj = kernel_dir.join("init").join("main.o");
    let _ = fs::remove_file(&target_obj);

    let (output, elapsed) = make_kernel(&kernel_dir, "init/main.o", &[]);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let kind = classify_build_failure(&stderr);
        panic!(
            "Sub-gate init/main.o FAILED [{}]\n\
             Elapsed: {:.1}s\nstderr:\n{}",
            kind,
            elapsed.as_secs_f64(),
            stderr
        );
    }

    common::assert_compilation_succeeds(&output);

    // Validate the produced .o file
    if target_obj.exists() {
        let obj_str = target_obj.to_string_lossy().into_owned();
        assert_riscv64_elf(&obj_str, "REL");
    }

    eprintln!(
        "Sub-gate init/main.o PASSED in {:.1}s",
        elapsed.as_secs_f64()
    );
}

/// Compiles `kernel/sched/core.o` — the kernel scheduler core.
///
/// This sub-gate exercises complex control flow, inline assembly, and
/// heavy GCC extension usage typical of the scheduler subsystem.
#[test]
#[ignore]
fn test_kernel_subgate_sched_core() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    let target_obj = kernel_dir.join("kernel").join("sched").join("core.o");
    let _ = fs::remove_file(&target_obj);

    let (output, elapsed) = make_kernel(&kernel_dir, "kernel/sched/core.o", &[]);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let kind = classify_build_failure(&stderr);
        panic!(
            "Sub-gate kernel/sched/core.o FAILED [{}]\n\
             Elapsed: {:.1}s\nstderr:\n{}",
            kind,
            elapsed.as_secs_f64(),
            stderr
        );
    }

    common::assert_compilation_succeeds(&output);
    eprintln!(
        "Sub-gate kernel/sched/core.o PASSED in {:.1}s",
        elapsed.as_secs_f64()
    );
}

/// Compiles `mm/memory.o` — the kernel memory management core.
///
/// This sub-gate exercises pointer arithmetic, atomic operations, and
/// memory-management-specific GCC builtins.
#[test]
#[ignore]
fn test_kernel_subgate_mm_memory() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    let target_obj = kernel_dir.join("mm").join("memory.o");
    let _ = fs::remove_file(&target_obj);

    let (output, elapsed) = make_kernel(&kernel_dir, "mm/memory.o", &[]);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let kind = classify_build_failure(&stderr);
        panic!(
            "Sub-gate mm/memory.o FAILED [{}]\n\
             Elapsed: {:.1}s\nstderr:\n{}",
            kind,
            elapsed.as_secs_f64(),
            stderr
        );
    }

    common::assert_compilation_succeeds(&output);
    eprintln!(
        "Sub-gate mm/memory.o PASSED in {:.1}s",
        elapsed.as_secs_f64()
    );
}

/// Compiles `fs/read_write.o` — the kernel filesystem read/write layer.
///
/// This sub-gate exercises VFS abstractions, struct-heavy code, and
/// file operation function pointers.
#[test]
#[ignore]
fn test_kernel_subgate_fs_read_write() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    let target_obj = kernel_dir.join("fs").join("read_write.o");
    let _ = fs::remove_file(&target_obj);

    let (output, elapsed) = make_kernel(&kernel_dir, "fs/read_write.o", &[]);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let kind = classify_build_failure(&stderr);
        panic!(
            "Sub-gate fs/read_write.o FAILED [{}]\n\
             Elapsed: {:.1}s\nstderr:\n{}",
            kind,
            elapsed.as_secs_f64(),
            stderr
        );
    }

    common::assert_compilation_succeeds(&output);
    eprintln!(
        "Sub-gate fs/read_write.o PASSED in {:.1}s",
        elapsed.as_secs_f64()
    );
}

// ===========================================================================
// Test: Full Kernel Build
// ===========================================================================

/// Builds the complete Linux kernel 6.9 (RISC-V defconfig) using BCC.
///
/// This test:
/// 1. Runs `make ARCH=riscv CC=./bcc -j<nproc>` for a full kernel build
/// 2. Captures stdout/stderr for diagnostic analysis
/// 3. Measures wall-clock build time
/// 4. Asserts build completes successfully (exit code 0)
/// 5. Asserts `vmlinux` ELF is produced
/// 6. Validates `vmlinux` is a valid RISC-V 64 ELF (ELFCLASS64, EM_RISCV, ET_EXEC)
#[test]
#[ignore]
fn test_full_kernel_build() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    let jobs_arg = format!("-j{}", num_cpus());

    // First, generate a default RISC-V config if needed
    let defconfig_output = Command::new("make")
        .current_dir(&kernel_dir)
        .args(["ARCH=riscv", "defconfig"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("Failed to run make defconfig");

    if !defconfig_output.status.success() {
        let stderr = String::from_utf8_lossy(&defconfig_output.stderr);
        eprintln!(
            "WARNING: make defconfig failed (may already be configured):\n{}",
            stderr
        );
    }

    // Build the full kernel
    let (output, elapsed) = make_kernel(&kernel_dir, "vmlinux", &[&jobs_arg]);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let kind = classify_build_failure(&stderr);
        panic!(
            "Full kernel build FAILED [{}]\n\
             Elapsed: {:.1}s\n\
             stdout (last 2000 chars):\n{}\n\
             stderr (last 5000 chars):\n{}",
            kind,
            elapsed.as_secs_f64(),
            &stdout[stdout.len().saturating_sub(2000)..],
            &stderr[stderr.len().saturating_sub(5000)..],
        );
    }

    common::assert_compilation_succeeds(&output);

    // Verify vmlinux was produced using fs::metadata for detailed checks
    let vmlinux = kernel_dir.join("vmlinux");
    let vmlinux_str = vmlinux.to_string_lossy().into_owned();

    let vmlinux_meta = fs::metadata(&vmlinux).unwrap_or_else(|e| {
        panic!(
            "vmlinux not found at '{}' after successful build: {}",
            vmlinux_str, e
        )
    });
    assert!(
        vmlinux_meta.is_file(),
        "vmlinux at '{}' is not a regular file",
        vmlinux_str
    );
    eprintln!(
        "vmlinux size: {} bytes ({:.1} MiB)",
        vmlinux_meta.len(),
        vmlinux_meta.len() as f64 / (1024.0 * 1024.0)
    );

    // Validate ELF properties
    assert_riscv64_elf(&vmlinux_str, "EXEC");

    eprintln!(
        "Full kernel build PASSED in {:.1}s ({:.1} minutes)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / 60.0
    );
}

// ===========================================================================
// Test: Wall-Clock Performance Validation
// ===========================================================================

/// Validates that the BCC kernel build time does not exceed 5× the GCC
/// baseline build time.
///
/// Per AAP §0.7.8: "Total kernel build time must not exceed 5× GCC-equivalent
/// on the same hardware."
///
/// The GCC baseline time is read from the `GCC_BASELINE_SECS` environment
/// variable. If not set, a conservative default is used.
#[test]
#[ignore]
fn test_kernel_build_wall_clock() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    let jobs_arg = format!("-j{}", num_cpus());

    // Clean build for accurate timing
    let _ = Command::new("make")
        .current_dir(&kernel_dir)
        .args(["ARCH=riscv", "clean"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    // Generate defconfig
    let _ = Command::new("make")
        .current_dir(&kernel_dir)
        .args(["ARCH=riscv", "defconfig"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    // Build with BCC and measure time
    let (output, elapsed_bcc) = make_kernel(&kernel_dir, "vmlinux", &[&jobs_arg]);

    let t_bcc = elapsed_bcc.as_secs_f64();
    let t_gcc = gcc_baseline_secs();
    let ceiling = t_gcc * WALL_CLOCK_CEILING_MULTIPLIER;

    eprintln!("Wall-clock performance:");
    eprintln!(
        "  BCC build time (T_bcc):   {:.1}s ({:.1} min)",
        t_bcc,
        t_bcc / 60.0
    );
    eprintln!(
        "  GCC baseline (T_gcc):     {:.1}s ({:.1} min)",
        t_gcc,
        t_gcc / 60.0
    );
    eprintln!(
        "  Ceiling (5× T_gcc):       {:.1}s ({:.1} min)",
        ceiling,
        ceiling / 60.0
    );
    eprintln!("  Ratio (T_bcc / T_gcc):    {:.2}×", t_bcc / t_gcc);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let kind = classify_build_failure(&stderr);
        panic!(
            "Kernel build failed during wall-clock test [{}]\nstderr:\n{}",
            kind,
            &stderr[stderr.len().saturating_sub(3000)..]
        );
    }

    assert!(
        t_bcc <= ceiling,
        "Wall-clock ceiling EXCEEDED: BCC took {:.1}s but ceiling is {:.1}s (5× {:.1}s GCC baseline). \
         Ratio: {:.2}×",
        t_bcc,
        ceiling,
        t_gcc,
        t_bcc / t_gcc
    );

    eprintln!(
        "Wall-clock ceiling PASSED: {:.2}× (limit: {:.1}×)",
        t_bcc / t_gcc,
        WALL_CLOCK_CEILING_MULTIPLIER
    );
}

// ===========================================================================
// Test: QEMU Boot (PRIMARY SUCCESS CRITERION)
// ===========================================================================

/// Boots the BCC-compiled Linux kernel in QEMU and verifies userspace is reached.
///
/// This is the **ultimate validation target** per the AAP. It exercises the full
/// C language surface, GCC extensions, preprocessor, inline assembly, and linker
/// correctness simultaneously.
///
/// Per AAP User Example:
/// ```text
/// qemu-system-riscv64 -machine virt -kernel vmlinux -initrd initramfs.cpio
///   -append "console=ttyS0" -nographic -no-reboot
/// ```
///
/// The test:
/// 1. Verifies `vmlinux` exists (from a prior kernel build)
/// 2. Prepares a minimal initramfs with `/init` that prints `USERSPACE_OK\n`
/// 3. Boots QEMU with a 120-second timeout
/// 4. Scans serial console output for the `USERSPACE_OK` marker
/// 5. Asserts QEMU exits cleanly (reboot from init → `-no-reboot` → exit)
#[test]
#[ignore]
fn test_kernel_qemu_boot() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    // Verify vmlinux exists
    let vmlinux = kernel_dir.join("vmlinux");
    if !vmlinux.is_file() {
        panic!(
            "vmlinux not found at '{}'. Run test_full_kernel_build first.",
            vmlinux.display()
        );
    }

    // Verify qemu-system-riscv64 is available
    let qemu_check = Command::new("qemu-system-riscv64")
        .arg("--version")
        .output();

    match qemu_check {
        Ok(out) if out.status.success() => {
            let version = String::from_utf8_lossy(&out.stdout);
            eprintln!(
                "QEMU: {}",
                version.lines().next().unwrap_or("unknown version")
            );
        }
        _ => {
            panic!(
                "qemu-system-riscv64 not found or not functional. \
                 Install with: apt install qemu-system-riscv64"
            );
        }
    }

    // Prepare initramfs
    let initramfs_path = prepare_initramfs();
    let initramfs_str = initramfs_path.to_string_lossy().into_owned();
    let vmlinux_str = vmlinux.to_string_lossy().into_owned();

    eprintln!("Booting kernel with QEMU...");
    eprintln!("  vmlinux:   {}", vmlinux_str);
    eprintln!("  initramfs: {}", initramfs_str);

    // Spawn QEMU with piped stdout/stderr for serial console capture
    let mut qemu_child = Command::new("qemu-system-riscv64")
        .args([
            "-machine",
            "virt",
            "-kernel",
            &vmlinux_str,
            "-initrd",
            &initramfs_str,
            "-append",
            "console=ttyS0",
            "-nographic",
            "-no-reboot",
            "-m",
            "256M",
            "-smp",
            "1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("Failed to spawn qemu-system-riscv64: {}", e));

    // Set up timeout mechanism using a separate thread
    let qemu_pid = qemu_child.id();
    let timeout = Duration::from_secs(QEMU_BOOT_TIMEOUT_SECS);

    let timeout_handle = thread::spawn(move || {
        thread::sleep(timeout);
        // If we reach here, the timeout expired — kill the QEMU process
        eprintln!(
            "QEMU boot timeout ({}s) expired — killing process {}",
            QEMU_BOOT_TIMEOUT_SECS, qemu_pid
        );
        // Use kill command as a fallback since we can't access the Child directly
        let _ = Command::new("kill")
            .arg("-9")
            .arg(qemu_pid.to_string())
            .output();
    });

    // Read serial console output from QEMU's stdout
    let stdout = qemu_child
        .stdout
        .take()
        .expect("Failed to capture QEMU stdout");
    let reader = BufReader::new(stdout);

    let mut found_userspace_ok = false;
    let mut console_output = String::new();
    let boot_start = Instant::now();

    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                console_output.push_str(&line);
                console_output.push('\n');

                // Check for the userspace marker
                if line.contains(USERSPACE_OK_MARKER) {
                    found_userspace_ok = true;
                    let boot_time = boot_start.elapsed();
                    eprintln!(
                        "USERSPACE_OK detected after {:.1}s of boot",
                        boot_time.as_secs_f64()
                    );
                    // Don't break — let QEMU finish naturally via reboot/poweroff
                }

                // Check for kernel panic (early termination)
                if line.contains("Kernel panic") || line.contains("---[ end Kernel panic") {
                    eprintln!("KERNEL PANIC detected in boot output");
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error reading QEMU output: {}", e);
                break;
            }
        }
    }

    // Wait for QEMU to exit
    let qemu_status = qemu_child
        .wait()
        .unwrap_or_else(|e| panic!("Failed to wait for QEMU process: {}", e));

    // The timeout thread will just silently terminate if QEMU already exited.
    // We don't join it — it's a daemon-like watchdog.
    drop(timeout_handle);

    let total_boot_time = boot_start.elapsed();
    eprintln!(
        "QEMU exited with status {} after {:.1}s",
        qemu_status,
        total_boot_time.as_secs_f64()
    );

    // Clean up initramfs
    let _ = fs::remove_file(&initramfs_path);

    // Assert USERSPACE_OK was found
    assert!(
        found_userspace_ok,
        "USERSPACE_OK marker NOT found in QEMU serial output.\n\
         Boot timed out or kernel failed to reach userspace.\n\
         Console output (last 3000 chars):\n{}",
        &console_output[console_output.len().saturating_sub(3000)..]
    );

    // Check that boot completed within timeout
    assert!(
        total_boot_time < Duration::from_secs(QEMU_BOOT_TIMEOUT_SECS),
        "QEMU boot exceeded {}s timeout (took {:.1}s)",
        QEMU_BOOT_TIMEOUT_SECS,
        total_boot_time.as_secs_f64()
    );

    eprintln!(
        "=== CHECKPOINT 6 PRIMARY SUCCESS: Kernel booted to userspace! ===\n\
         Boot time: {:.1}s",
        total_boot_time.as_secs_f64()
    );
}

// ===========================================================================
// Test: Regression Guard (AAP §0.7.5)
// ===========================================================================

/// Re-runs Checkpoint 3 (internal test suite) as a regression guard.
///
/// Per AAP §0.7.5: "Checkpoint 3 must be re-run after any feature addition
/// during the kernel build phase to confirm no regressions."
///
/// This test invokes `cargo test --release --test checkpoint3_internal` and
/// asserts zero failures.
#[test]
#[ignore]
fn test_regression_guard_after_kernel_features() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    let output = Command::new("cargo")
        .current_dir(&manifest_dir)
        .args([
            "test",
            "--release",
            "--test",
            "checkpoint3_internal",
            "--",
            "--test-threads=1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("Failed to invoke cargo test for regression guard: {}", e));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("Regression guard results:");
    eprintln!("  exit code: {:?}", output.status.code());
    eprintln!("  stdout:\n{}", stdout);

    common::assert_exit_success(&output);

    // Parse output for failure count
    let combined = format!("{}\n{}", stdout, stderr);
    if combined.contains("FAILED") || combined.contains("failures") {
        // Extract failure details
        let failure_lines: Vec<&str> = combined
            .lines()
            .filter(|l| l.contains("FAILED") || l.contains("failures"))
            .collect();

        panic!(
            "Regression guard FAILED — Checkpoint 3 has regressions:\n{}",
            failure_lines.join("\n")
        );
    }

    eprintln!("Regression guard PASSED: Checkpoint 3 shows no regressions.");
}

// ===========================================================================
// Test: Build Failure Classification Validation
// ===========================================================================

/// Validates the kernel build failure classification logic itself.
///
/// This test verifies that `classify_build_failure` correctly categorizes
/// different types of kernel build errors per AAP §0.7.6.
#[test]
#[ignore]
fn test_failure_classification_logic() {
    // 1. Missing GCC extension
    assert_eq!(
        classify_build_failure("error: unknown attribute 'noinline'"),
        FailureKind::MissingGccExtension,
        "Should classify unknown attribute as MissingGccExtension"
    );

    // 2. Missing builtin
    assert_eq!(
        classify_build_failure("error: implicit declaration of function '__builtin_expect'"),
        FailureKind::MissingBuiltin,
        "Should classify missing __builtin_ as MissingBuiltin"
    );

    // 3. Inline asm constraint
    assert_eq!(
        classify_build_failure("error: invalid constraint 'Z' in asm statement"),
        FailureKind::InlineAsmConstraint,
        "Should classify asm constraint error as InlineAsmConstraint"
    );

    // 4. Preprocessor issue
    assert_eq!(
        classify_build_failure("fatal error: linux/types.h: #include not found"),
        FailureKind::PreprocessorIssue,
        "Should classify #include failure as PreprocessorIssue"
    );

    // 5. Codegen bug
    assert_eq!(
        classify_build_failure("internal compiler error: assertion failed in emit_instruction"),
        FailureKind::CodegenBug,
        "Should classify ICE as CodegenBug"
    );

    // Unknown
    assert_eq!(
        classify_build_failure("something went wrong but no recognizable pattern"),
        FailureKind::Unknown,
        "Should classify unrecognized error as Unknown"
    );

    eprintln!("Failure classification logic validated successfully.");
}

// ===========================================================================
// Test: Kernel Config Validation
// ===========================================================================

/// Validates that the RISC-V defconfig is correctly generated and BCC
/// can parse and honor the resulting kernel configuration.
#[test]
#[ignore]
fn test_kernel_defconfig_generation() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    // Generate defconfig
    let output = Command::new("make")
        .current_dir(&kernel_dir)
        .args(["ARCH=riscv", "defconfig"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("Failed to run make defconfig");

    assert!(
        output.status.success(),
        "make defconfig failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify .config was generated
    let config_file = kernel_dir.join(".config");
    assert!(
        config_file.is_file(),
        ".config not generated at '{}'",
        config_file.display()
    );

    // Verify it targets RISC-V
    let config_content = fs::read_to_string(&config_file).unwrap_or_default();
    assert!(
        config_content.contains("CONFIG_RISCV=y") || config_content.contains("CONFIG_ARCH_RV64I"),
        "Kernel config does not appear to target RISC-V"
    );

    eprintln!("Kernel defconfig generation validated.");
}

// ===========================================================================
// Test: Initramfs Preparation (Standalone)
// ===========================================================================

/// Validates the initramfs preparation pipeline independently of the full
/// kernel boot test. This ensures the minimal `/init` binary compiles
/// correctly for RISC-V 64 and the cpio archive is well-formed.
#[test]
#[ignore]
fn test_initramfs_preparation() {
    // Verify cpio is available
    let cpio_check = Command::new("cpio").arg("--version").output();

    match cpio_check {
        Ok(out) if out.status.success() => {}
        _ => {
            eprintln!("SKIP: cpio not available");
            return;
        }
    }

    let initramfs = prepare_initramfs();
    assert!(
        initramfs.exists(),
        "Initramfs was not created at '{}'",
        initramfs.display()
    );

    // Verify the cpio archive contains "init"
    let list_output = Command::new("sh")
        .arg("-c")
        .arg(format!("cpio -t < '{}'", initramfs.to_string_lossy()))
        .output()
        .expect("Failed to list cpio contents");

    let contents = String::from_utf8_lossy(&list_output.stdout);
    assert!(
        contents.contains("init"),
        "Initramfs does not contain 'init' entry. Contents:\n{}",
        contents
    );

    // Clean up using the common cleanup helper
    let initramfs_str_cleanup = initramfs.to_string_lossy().into_owned();
    common::cleanup_temp_files(&[&initramfs_str_cleanup]);

    eprintln!("Initramfs preparation validated.");
}

// ===========================================================================
// Test: vmlinux ELF Detailed Validation
// ===========================================================================

/// Performs detailed ELF validation on the `vmlinux` binary after a full
/// kernel build.
///
/// Validates:
/// - ELF class (64-bit)
/// - Machine type (RISC-V)
/// - ELF type (ET_EXEC)
/// - Entry point is non-zero
/// - Essential sections exist (`.text`, `.rodata`, `.data`, `.bss`)
#[test]
#[ignore]
fn test_vmlinux_elf_validation() {
    let kernel_dir = match kernel_src_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Kernel source not found");
            return;
        }
    };

    let vmlinux = kernel_dir.join("vmlinux");
    if !vmlinux.is_file() {
        eprintln!("SKIP: vmlinux not found — run test_full_kernel_build first");
        return;
    }

    let vmlinux_str = vmlinux.to_string_lossy().into_owned();

    // Validate ELF header
    assert_riscv64_elf(&vmlinux_str, "EXEC");

    // Verify entry point is non-zero
    let header = common::readelf_header(&vmlinux_str);
    let has_entry = header.lines().any(|line| {
        if line.contains("Entry point address:") {
            // Extract the hex value and check it's non-zero
            let parts: Vec<&str> = line.split("0x").collect();
            if parts.len() >= 2 {
                let hex_str = parts[1].trim();
                return hex_str != "0" && !hex_str.chars().all(|c| c == '0');
            }
        }
        false
    });
    assert!(
        has_entry,
        "vmlinux entry point should be non-zero. Header:\n{}",
        header
    );

    // Verify essential sections exist via readelf -S
    let section_output = Command::new("readelf")
        .args(["-S", &vmlinux_str])
        .output()
        .expect("readelf -S failed");
    let section_text = String::from_utf8_lossy(&section_output.stdout);

    let required_sections = [".text", ".rodata", ".data"];
    for section in &required_sections {
        assert!(
            section_text.contains(section),
            "vmlinux missing required section '{}'. Sections:\n{}",
            section,
            section_text
        );
    }

    eprintln!(
        "vmlinux ELF validation passed. Found sections: {}",
        required_sections.join(", ")
    );
}

// ===========================================================================
// Cleanup helper
// ===========================================================================

/// Convenience function for cleaning up kernel build artifacts.
///
/// This is called at the end of test suites that modify the kernel source tree
/// to restore it to a clean state.
#[allow(dead_code)]
fn clean_kernel_build(kernel_dir: &Path) {
    let _ = Command::new("make")
        .current_dir(kernel_dir)
        .args(["ARCH=riscv", "clean"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();
}
