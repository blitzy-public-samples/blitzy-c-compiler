//! Checkpoint 6 integration test suite — Linux kernel 6.9 (RISC-V configuration)
//! build and QEMU boot orchestration.
//!
//! This is the **primary success validation gate** for the BCC compiler. It
//! compiles the full Linux kernel with BCC (`make ARCH=riscv CC=./bcc`),
//! validates the resulting `vmlinux` ELF, boots it in `qemu-system-riscv64`
//! with a minimal initramfs containing a static `/init` that prints
//! `USERSPACE_OK`, and asserts successful userspace entry.
//!
//! # Test Structure
//!
//! | Test | Description |
//! |------|-------------|
//! | `test_kernel_init_main_o` | Compile `init/main.o` — sub-gate |
//! | `test_kernel_sched_core_o` | Compile `kernel/sched/core.o` — sub-gate |
//! | `test_kernel_mm_memory_o` | Compile `mm/memory.o` — sub-gate |
//! | `test_kernel_fs_read_write_o` | Compile `fs/read_write.o` — sub-gate |
//! | `test_full_kernel_build` | Full kernel build + 5× GCC ceiling |
//! | `test_kernel_qemu_boot` | QEMU boot to userspace validation |
//!
//! # Environment Variables
//!
//! - `KERNEL_SRC_DIR` — Absolute path to a Linux 6.9 kernel source tree (**required**)
//! - `GCC_BENCHMARK_SECS` — GCC build time in seconds for 5× ceiling comparison
//!   (optional; default 600 s)
//! - `KERNEL_BUILD_DIR` — Pre-existing out-of-tree build directory containing
//!   `vmlinux` (optional; used by the QEMU boot test to skip a redundant build)
//!
//! # Sequential Gate Enforcement (Section 0.7.5)
//!
//! This checkpoint is a **hard gate** — failure halts all forward progress.
//! Tests are ordered so that sub-gate compilation units are validated before
//! the full build, and the full build is validated before the QEMU boot.
//!
//! All tests are marked `#[ignore]` because they require a kernel source tree
//! and QEMU system emulator that are not available in ordinary CI runs.
//!
//! # Running
//!
//! ```sh
//! KERNEL_SRC_DIR=/path/to/linux-6.9 cargo test --test checkpoint6_kernel -- --ignored
//! ```

mod common;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Environment variable name for the Linux kernel source tree path.
const KERNEL_SRC_ENV: &str = "KERNEL_SRC_DIR";

/// Environment variable name for the GCC benchmark build time (seconds).
const GCC_BENCHMARK_ENV: &str = "GCC_BENCHMARK_SECS";

/// Environment variable name for a pre-existing kernel build directory.
const KERNEL_BUILD_DIR_ENV: &str = "KERNEL_BUILD_DIR";

/// Maximum time (seconds) allowed for the QEMU boot before timeout.
const QEMU_BOOT_TIMEOUT_SECS: u64 = 120;

/// Wall-clock multiplier applied to the GCC benchmark (Section 0.7.8).
const WALL_CLOCK_MULTIPLIER: u64 = 5;

/// Marker string emitted by the minimal `/init` to signal userspace entry.
const USERSPACE_OK_MARKER: &str = "USERSPACE_OK";

/// Default GCC benchmark time (seconds) when `GCC_BENCHMARK_SECS` is unset.
const DEFAULT_GCC_BENCHMARK_SECS: u64 = 600;

/// RISC-V ELF machine string as reported by `readelf -h`.
const RISCV_MACHINE: &str = "RISC-V";

/// ELF-64 class string as reported by `readelf -h`.
const ELF64_CLASS: &str = "ELF64";

/// Kernel architecture passed to `make ARCH=...`.
const KERNEL_ARCH: &str = "riscv";

// ---------------------------------------------------------------------------
// Helper: Kernel source directory resolution
// ---------------------------------------------------------------------------

/// Resolve the kernel source directory from the `KERNEL_SRC_DIR` environment
/// variable.  Returns `None` when the variable is not set.
fn kernel_source_dir() -> Option<PathBuf> {
    // Use var_os to handle non-UTF-8 paths gracefully.
    env::var_os(KERNEL_SRC_ENV).map(PathBuf::from)
}

/// Require the kernel source directory to be set, to exist, and to look like
/// a real kernel tree.
///
/// # Panics
///
/// Panics with a human-readable message when:
/// - `KERNEL_SRC_DIR` is not set
/// - The path does not exist or is not a directory
/// - The directory lacks a top-level `Kconfig` file
fn require_kernel_source() -> PathBuf {
    let dir = match kernel_source_dir() {
        Some(d) => d,
        None => panic!(
            "Environment variable {} is not set. \
             Set it to the absolute path of a Linux 6.9 kernel source tree \
             to run Checkpoint 6 kernel tests.",
            KERNEL_SRC_ENV
        ),
    };

    assert!(
        dir.exists(),
        "{} path '{}' does not exist.",
        KERNEL_SRC_ENV,
        dir.display()
    );
    assert!(
        dir.is_dir(),
        "{} path '{}' is not a directory.",
        KERNEL_SRC_ENV,
        dir.display()
    );

    // Sanity check: a kernel tree must have a root-level Kconfig.
    let kconfig = dir.join("Kconfig");
    assert!(
        kconfig.exists(),
        "{} path '{}' does not contain a top-level 'Kconfig' — \
         it does not appear to be a valid Linux kernel source tree.",
        KERNEL_SRC_ENV,
        dir.display()
    );

    dir
}

// ---------------------------------------------------------------------------
// Helper: BCC binary resolution
// ---------------------------------------------------------------------------

/// Return the **absolute** path to the BCC binary as a `String`, suitable for
/// passing as `CC=…` to the kernel build system.
fn bcc_cc_path() -> String {
    let path = common::bcc_binary_path();
    match path.canonicalize() {
        Ok(abs) => abs.to_string_lossy().into_owned(),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// Create a thin wrapper shell script that invokes BCC with
/// `--target=riscv64`.
///
/// The Linux kernel build system calls `$(CC)` directly.  Because BCC uses an
/// explicit `--target=<arch>` flag (rather than a cross-compiler prefix), we
/// wrap BCC in a script that always prepends the correct target flag.
///
/// Returns the absolute path to the wrapper script.
fn create_bcc_wrapper(work_dir: &Path) -> PathBuf {
    let wrapper_path = work_dir.join("bcc-riscv64");
    let bcc_abs = bcc_cc_path();
    let script = format!(
        "#!/bin/sh\nexec \"{}\" --target=riscv64 \"$@\"\n",
        bcc_abs
    );
    fs::write(&wrapper_path, &script).unwrap_or_else(|e| {
        panic!(
            "Failed to write BCC wrapper at '{}': {}",
            wrapper_path.display(),
            e
        )
    });

    // Make the wrapper executable (Unix only).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&wrapper_path, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to chmod wrapper script '{}': {}",
                    wrapper_path.display(),
                    e
                )
            });
    }

    wrapper_path
}

/// Retrieve the GCC benchmark time for the 5× wall-clock ceiling check.
///
/// Reads `GCC_BENCHMARK_SECS` from the environment; falls back to
/// `DEFAULT_GCC_BENCHMARK_SECS` when not set or invalid.
fn gcc_benchmark_secs() -> u64 {
    match env::var(GCC_BENCHMARK_ENV) {
        Ok(val) => val.parse::<u64>().unwrap_or_else(|_| {
            eprintln!(
                "Warning: {} value '{}' is not a valid integer; \
                 using default {}s.",
                GCC_BENCHMARK_ENV, val, DEFAULT_GCC_BENCHMARK_SECS
            );
            DEFAULT_GCC_BENCHMARK_SECS
        }),
        Err(_) => {
            eprintln!(
                "Note: {} not set; using default benchmark of {}s \
                 (5× ceiling = {}s).",
                GCC_BENCHMARK_ENV,
                DEFAULT_GCC_BENCHMARK_SECS,
                DEFAULT_GCC_BENCHMARK_SECS * WALL_CLOCK_MULTIPLIER
            );
            DEFAULT_GCC_BENCHMARK_SECS
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: Kernel build preparation
// ---------------------------------------------------------------------------

/// Prepare the kernel build directory with a RISC-V defconfig and generated
/// headers.
///
/// Runs:
/// 1. `make ARCH=riscv O=<build_dir> defconfig`
/// 2. `make ARCH=riscv O=<build_dir> HOSTCC=gcc prepare`
///
/// Uses the **host** GCC for `HOSTCC` because BCC targets RISC-V, not the
/// build host.
fn prepare_kernel_build(kernel_src: &Path, build_dir: &Path) {
    fs::create_dir_all(build_dir).unwrap_or_else(|e| {
        panic!(
            "Failed to create kernel build directory '{}': {}",
            build_dir.display(),
            e
        )
    });

    // Step 1: defconfig.
    let defconfig = Command::new("make")
        .current_dir(kernel_src)
        .arg(format!("O={}", build_dir.display()))
        .arg(format!("ARCH={}", KERNEL_ARCH))
        .arg("defconfig")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run 'make defconfig': {}", e));

    assert!(
        defconfig.status.success(),
        "'make ARCH=riscv defconfig' failed.\nstderr: {}",
        String::from_utf8_lossy(&defconfig.stderr)
    );

    // Verify .config was generated.
    let config_path = build_dir.join(".config");
    assert!(
        config_path.is_file(),
        ".config not generated in '{}'.",
        build_dir.display()
    );

    // Read and log a snippet of the config for diagnostics.
    if let Ok(config_text) = fs::read_to_string(&config_path) {
        let snippet: String = config_text.lines().take(10).collect::<Vec<_>>().join("\n");
        eprintln!("Kernel .config (first 10 lines):\n{}", snippet);
    }

    // Step 2: Generate auto-generated headers (prepare).
    let prepare = Command::new("make")
        .current_dir(kernel_src)
        .arg(format!("O={}", build_dir.display()))
        .arg(format!("ARCH={}", KERNEL_ARCH))
        .arg("HOSTCC=gcc")
        .arg("prepare")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run 'make prepare': {}", e));

    if !prepare.status.success() {
        eprintln!(
            "Warning: 'make prepare' exited with non-zero status.\nstderr: {}",
            String::from_utf8_lossy(&prepare.stderr)
        );
    }
}

// ---------------------------------------------------------------------------
// Kernel build failure classification (Section 0.7.6)
// ---------------------------------------------------------------------------

/// Classification categories for kernel build failures.
///
/// Priority order per Section 0.7.6 protocol:
/// 1. Missing GCC extension
/// 2. Missing builtin
/// 3. Inline asm constraint
/// 4. Preprocessor issue
/// 5. Code generation bug
/// 6. Unknown / unclassified
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildFailureCategory {
    /// A GCC language extension or `__attribute__` is not supported.
    MissingGccExtension,
    /// A `__builtin_*` function is not recognised.
    MissingBuiltin,
    /// An inline assembly constraint, operand, or template is invalid.
    InlineAsmConstraint,
    /// A preprocessor directive or macro expansion failed.
    PreprocessorIssue,
    /// An internal code-generation error occurred.
    CodegenBug,
    /// The failure could not be classified into any of the above categories.
    Unknown,
}

impl std::fmt::Display for BuildFailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingGccExtension => write!(f, "Missing GCC extension"),
            Self::MissingBuiltin => write!(f, "Missing builtin"),
            Self::InlineAsmConstraint => write!(f, "Inline asm constraint issue"),
            Self::PreprocessorIssue => write!(f, "Preprocessor issue"),
            Self::CodegenBug => write!(f, "Code generation bug"),
            Self::Unknown => write!(f, "Unknown / unclassified failure"),
        }
    }
}

/// Classify a kernel build failure based on the compiler's stderr output.
///
/// Implements the priority-ordered protocol from Section 0.7.6:
///
/// missing GCC extension → missing builtin → inline asm constraint →
/// preprocessor issue → codegen bug → unknown
fn classify_build_failure(stderr: &str) -> BuildFailureCategory {
    let lower = stderr.to_lowercase();

    // 1. Missing GCC extension — attribute / language extension not recognised.
    let gcc_extension_indicators: &[&str] = &[
        "unsupported attribute",
        "unknown attribute",
        "unsupported extension",
        "unrecognized attribute",
        "statement expression",
        "typeof",
        "computed goto",
        "case range",
        "transparent union",
        "zero-length array",
        "local label",
        "__extension__",
    ];
    for pattern in gcc_extension_indicators {
        if lower.contains(pattern) {
            return BuildFailureCategory::MissingGccExtension;
        }
    }

    // 2. Missing builtin — __builtin_* function not recognised.
    let builtin_indicators: &[&str] = &[
        "__builtin_",
        "unknown builtin",
        "unsupported builtin",
        "unrecognized builtin",
    ];
    for pattern in builtin_indicators {
        if lower.contains(pattern) {
            return BuildFailureCategory::MissingBuiltin;
        }
    }

    // 3. Inline assembly constraint issue.
    let asm_indicators: &[&str] = &[
        "asm constraint",
        "inline asm",
        "asm operand",
        "asm template",
        "invalid constraint",
        "unsupported constraint",
        "clobber",
        "asm goto",
    ];
    for pattern in asm_indicators {
        if lower.contains(pattern) {
            return BuildFailureCategory::InlineAsmConstraint;
        }
    }

    // 4. Preprocessor issue.
    let preprocessor_indicators: &[&str] = &[
        "#include",
        "#define",
        "#if ",
        "#ifdef",
        "#ifndef",
        "macro",
        "preprocessor",
        "unterminated",
        "undefined macro",
        "#error",
    ];
    for pattern in preprocessor_indicators {
        if lower.contains(pattern) {
            return BuildFailureCategory::PreprocessorIssue;
        }
    }

    // 5. Code generation bug (catch-all for internal errors).
    let codegen_indicators: &[&str] = &[
        "codegen",
        "code generation",
        "register alloc",
        "relocation",
        "internal error",
        "assertion failed",
        "ice:",
        "overflow",
    ];
    for pattern in codegen_indicators {
        if lower.contains(pattern) {
            return BuildFailureCategory::CodegenBug;
        }
    }

    BuildFailureCategory::Unknown
}

/// Format a human-readable diagnostic report for a kernel build failure.
fn format_failure_report(target: &str, output: &common::BccOutput) -> String {
    let category = classify_build_failure(&output.stderr);
    let stderr_snippet = &output.stderr[..output.stderr.len().min(4096)];
    let stdout_snippet = &output.stdout[..output.stdout.len().min(2048)];
    format!(
        "=== Kernel Build Failure Report ===\n\
         Target:         {}\n\
         Classification: {} (Section 0.7.6)\n\
         Exit code:      {:?}\n\
         ------- stderr (first 4096 bytes) -------\n\
         {}\n\
         ------- stdout (first 2048 bytes) -------\n\
         {}",
        target, category, output.exit_code(), stderr_snippet, stdout_snippet,
    )
}

// ---------------------------------------------------------------------------
// Helper: Compile and validate a single kernel object
// ---------------------------------------------------------------------------

/// Compile a single kernel compilation unit using the kernel build system.
///
/// Invokes: `make ARCH=riscv CC=<wrapper> HOSTCC=gcc O=<build_dir> <target>`
fn compile_kernel_object_via_make(
    kernel_src: &Path,
    build_dir: &Path,
    wrapper_path: &Path,
    target_obj: &str,
) -> common::BccOutput {
    let result = Command::new("make")
        .current_dir(kernel_src)
        .arg(format!("O={}", build_dir.display()))
        .arg(format!("ARCH={}", KERNEL_ARCH))
        .arg(format!("CC={}", wrapper_path.display()))
        .arg("HOSTCC=gcc")
        .arg(target_obj)
        .arg("-j1")
        .env("LC_ALL", "C")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| {
            panic!("Failed to execute 'make {}': {}", target_obj, e)
        });

    common::BccOutput {
        status: result.status,
        stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&result.stderr).into_owned(),
    }
}

/// Validate that a compiled kernel object is a valid RISC-V 64-bit
/// relocatable ELF with at least a `.text` section.
fn validate_kernel_object(obj_path: &str) {
    let path = Path::new(obj_path);
    assert!(
        path.is_file(),
        "Kernel object '{}' does not exist after compilation.",
        obj_path
    );

    // Verify non-zero file size.
    let meta = fs::metadata(obj_path).unwrap_or_else(|e| {
        panic!("Cannot stat '{}': {}", obj_path, e)
    });
    assert!(
        meta.len() > 0,
        "Kernel object '{}' is empty (0 bytes).",
        obj_path
    );

    // Log the ELF header for diagnostic context.
    let header = common::readelf_header(obj_path);
    eprintln!("ELF header for {}:\n{}", obj_path, header);

    // Relocatable, RISC-V, 64-bit.
    common::assert_elf_type(obj_path, "REL");
    common::assert_elf_machine(obj_path, RISCV_MACHINE);
    common::assert_elf_class(obj_path, ELF64_CLASS);

    // Must have a .text section.
    common::assert_section_exists(obj_path, ".text");

    // Log section list.
    let sections = common::readelf_sections(obj_path);
    eprintln!("Sections for {}:\n{}", obj_path, sections);
}

// ---------------------------------------------------------------------------
// Helper: Sub-gate test driver
// ---------------------------------------------------------------------------

/// Execute a sub-gate test: prepare the kernel build, compile a single
/// object file with BCC, classify any failure, and validate the result.
fn run_subgate_test(target_obj: &str) {
    let kernel_src = require_kernel_source();
    let test_dir = common::TestDir::new(
        &format!("kernel_subgate_{}", target_obj.replace('/', "_")),
    );
    let build_dir = test_dir.file_path("build");

    // Prepare kernel build (defconfig + auto-generated headers).
    prepare_kernel_build(&kernel_src, &build_dir);

    // Create BCC wrapper script targeting riscv64.
    let wrapper = create_bcc_wrapper(test_dir.path());

    // Compile the target object.
    let result = compile_kernel_object_via_make(
        &kernel_src,
        &build_dir,
        &wrapper,
        target_obj,
    );

    // On failure: classify per Section 0.7.6 and panic with diagnostics.
    if !result.success() {
        let report = format_failure_report(target_obj, &result);
        panic!(
            "Sub-gate compilation FAILED for '{}'.\n{}",
            target_obj, report
        );
    }

    // Validate the produced object file.
    let mut obj_path = PathBuf::from(&build_dir);
    obj_path.push(target_obj);
    validate_kernel_object(obj_path.to_str().unwrap());

    eprintln!(
        "Sub-gate PASSED: '{}' is a valid RISC-V 64 relocatable ELF.",
        target_obj
    );
}

// ---------------------------------------------------------------------------
// Minimal /init source for QEMU boot validation
// ---------------------------------------------------------------------------

/// C source for the minimal `/init` program.
///
/// Runs as PID 1 inside the QEMU initramfs.  Writes `"USERSPACE_OK\n"` to
/// the serial console (fd 1) via the `write` syscall, then powers off the
/// machine via the `reboot` syscall.
///
/// Uses raw RISC-V 64 Linux syscalls to avoid any C library dependency,
/// so the binary can be compiled with BCC using `-nostdlib`.
const INIT_SOURCE: &str = r#"
/* Minimal /init for BCC kernel boot validation (RISC-V 64).
 *
 * Uses raw syscalls via inline assembly — no C library dependency.
 * RISC-V syscall convention: a7 = syscall number, a0-a5 = args, ecall.
 */

void _start(void)
{
    static const char msg[] = "USERSPACE_OK\n";
    long ret;

    /* write(1, msg, 13)  —  __NR_write = 64 */
    {
        long fd    = 1;
        long buf   = (long)msg;
        long count = 13;
        long nr    = 64;
        __asm__ __volatile__ (
            "mv a0, %1\n\t"
            "mv a1, %2\n\t"
            "mv a2, %3\n\t"
            "mv a7, %4\n\t"
            "ecall\n\t"
            "mv %0, a0"
            : "=r"(ret)
            : "r"(fd), "r"(buf), "r"(count), "r"(nr)
            : "a0", "a1", "a2", "a7", "memory"
        );
    }

    /* reboot(MAGIC1, MAGIC2, POWER_OFF, NULL)  —  __NR_reboot = 142 */
    {
        long magic1 = (long)0xfee1deadUL;
        long magic2 = (long)0x28121969UL;
        long cmd    = (long)0x4321fedcUL;
        long nr     = 142;
        __asm__ __volatile__ (
            "mv a0, %0\n\t"
            "mv a1, %1\n\t"
            "mv a2, %2\n\t"
            "mv a3, zero\n\t"
            "mv a7, %3\n\t"
            "ecall"
            :
            : "r"(magic1), "r"(magic2), "r"(cmd), "r"(nr)
            : "a0", "a1", "a2", "a3", "a7", "memory"
        );
    }

    /* exit(0)  —  __NR_exit = 93 (fallback) */
    {
        long code = 0;
        long nr   = 93;
        __asm__ __volatile__ (
            "mv a0, %0\n\t"
            "mv a7, %1\n\t"
            "ecall"
            :
            : "r"(code), "r"(nr)
            : "a0", "a7"
        );
    }

    /* Unreachable — loop forever as a final safety net. */
    for (;;) { }
}
"#;

/// Build the minimal `/init` binary and package it into a cpio initramfs
/// archive.
///
/// Steps:
/// 1. Write `init.c` source to `work_dir`.
/// 2. Compile `init.c` → `init.o` via `compile_to_object`.
/// 3. Link `init.o` → `init` static binary via `compile`.
/// 4. Lay out the initramfs directory tree.
/// 5. Create `initramfs.cpio` via `find | cpio`.
///
/// Returns the path to the `initramfs.cpio` file.
fn build_initramfs(work_dir: &Path) -> PathBuf {
    // Step 1: Write init.c.
    let init_c = work_dir.join("init.c");
    fs::write(&init_c, INIT_SOURCE).unwrap_or_else(|e| {
        panic!("Failed to write '{}': {}", init_c.display(), e)
    });

    // Step 2: Compile init.c → init.o using the shared test utility.
    let init_o = work_dir.join("init.o");
    let obj_result = common::compile_to_object(
        init_c.to_str().unwrap(),
        init_o.to_str().unwrap(),
        "riscv64",
        &[],
    );
    assert!(
        obj_result.success(),
        "Failed to compile init.c to object.\nstderr: {}",
        obj_result.stderr
    );

    // Step 3: Link init.o → static init binary.
    let init_bin = work_dir.join("init");
    let link_result = common::compile(
        init_o.to_str().unwrap(),
        &[
            "--target=riscv64",
            "-nostdlib",
            "-o",
            init_bin.to_str().unwrap(),
        ],
    );
    assert!(
        link_result.success(),
        "Failed to link init binary.\nstderr: {}",
        link_result.stderr
    );
    assert!(
        init_bin.is_file(),
        "Init binary was not produced at '{}'.",
        init_bin.display()
    );

    // Step 4: Create initramfs directory tree.
    let initramfs_dir = work_dir.join("initramfs");
    fs::create_dir_all(&initramfs_dir).unwrap_or_else(|e| {
        panic!("Failed to create initramfs directory: {}", e)
    });

    // Copy the init binary into the initramfs root.
    let initramfs_init = initramfs_dir.join("init");
    let init_bytes = fs::read(&init_bin).unwrap_or_else(|e| {
        panic!("Failed to read init binary: {}", e)
    });
    fs::write(&initramfs_init, &init_bytes).unwrap_or_else(|e| {
        panic!("Failed to write initramfs/init: {}", e)
    });

    // Make /init executable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            &initramfs_init,
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap_or_else(|e| {
            panic!("Failed to set /init permissions: {}", e)
        });
    }

    // Step 5: Create cpio archive.
    let cpio_path = work_dir.join("initramfs.cpio");
    let cpio_cmd = format!(
        "cd '{}' && find . | cpio --quiet -o -H newc > '{}'",
        initramfs_dir.display(),
        cpio_path.display()
    );
    let cpio_out = Command::new("sh")
        .args(["-c", &cpio_cmd])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute cpio: {}", e));

    assert!(
        cpio_out.status.success(),
        "cpio initramfs creation failed.\nstderr: {}",
        String::from_utf8_lossy(&cpio_out.stderr)
    );
    assert!(
        cpio_path.is_file(),
        "initramfs.cpio not created at '{}'.",
        cpio_path.display()
    );

    cpio_path
}

// ---------------------------------------------------------------------------
// Helper: QEMU boot with timeout
// ---------------------------------------------------------------------------

/// Boot the compiled kernel in `qemu-system-riscv64` and capture serial
/// console output.
///
/// A **watchdog thread** enforces the timeout: if the kernel fails to boot
/// within `timeout_secs`, the QEMU process is killed.
///
/// Returns `(serial_output, timed_out)`.
fn qemu_boot_kernel(
    vmlinux: &Path,
    initramfs_cpio: &Path,
    timeout_secs: u64,
) -> (String, bool) {
    let start = Instant::now();

    let child = Command::new("qemu-system-riscv64")
        .args(["-machine", "virt", "-nographic", "-m", "512M", "-bios", "none"])
        .arg("-kernel")
        .arg(vmlinux.as_os_str())
        .arg("-initrd")
        .arg(initramfs_cpio.as_os_str())
        .arg("-append")
        .arg("console=ttyS0 panic=-1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to spawn qemu-system-riscv64: {}. \
                 Ensure it is installed (apt install qemu-system-misc).",
                e
            )
        });

    let child_id = child.id();
    let timeout = Duration::from_secs(timeout_secs);

    // Watchdog: kill QEMU on timeout.
    let watchdog = thread::spawn(move || {
        thread::sleep(timeout);
        // Use the `kill` command so we don't need the libc crate.
        let _ = Command::new("kill")
            .arg("-9")
            .arg(child_id.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output();
    });

    // Wait for QEMU to exit (normally or killed by watchdog).
    let output = child.wait_with_output().unwrap_or_else(|e| {
        panic!("Failed to wait on qemu-system-riscv64: {}", e)
    });

    let elapsed = start.elapsed();
    let timed_out = elapsed.as_secs() >= timeout_secs;

    // Combine stdout and stderr — QEMU may emit serial data on either.
    let stdout_text = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr_text = String::from_utf8_lossy(&output.stderr).into_owned();
    let combined = format!("{}{}", stdout_text, stderr_text);

    eprintln!(
        "QEMU boot finished in {:.1}s (limit: {}s, timed_out: {}).",
        elapsed.as_secs_f64(),
        timeout_secs,
        timed_out
    );

    // The watchdog thread will terminate on its own after the sleep expires;
    // dropping the handle is safe.
    drop(watchdog);

    (combined, timed_out)
}

// ---------------------------------------------------------------------------
// Helper: Locate or build vmlinux for the QEMU boot test
// ---------------------------------------------------------------------------

/// Locate a pre-existing `vmlinux` or build the kernel afresh.
///
/// Search order:
/// 1. `$KERNEL_BUILD_DIR/vmlinux`
/// 2. `<kernel_src>/vmlinux` (in-tree build)
/// 3. Build the kernel into `<work_dir>/kernel_build/`.
fn find_or_build_vmlinux(kernel_src: &Path, work_dir: &Path) -> PathBuf {
    // Candidate 1: KERNEL_BUILD_DIR environment variable.
    if let Ok(dir) = env::var(KERNEL_BUILD_DIR_ENV) {
        let mut candidate = PathBuf::from(dir);
        candidate.push("vmlinux");
        if candidate.as_path().is_file() {
            eprintln!(
                "Using pre-built vmlinux from {}: '{}'",
                KERNEL_BUILD_DIR_ENV,
                candidate.display()
            );
            return candidate;
        }
    }

    // Candidate 2: In-tree vmlinux.
    let in_tree = kernel_src.join("vmlinux");
    if in_tree.is_file() {
        eprintln!("Using in-tree vmlinux at '{}'.", in_tree.display());
        return in_tree;
    }

    // Candidate 3: Build the kernel ourselves.
    eprintln!(
        "No pre-built vmlinux found.  Building kernel for QEMU boot test …"
    );

    let build_dir = work_dir.join("kernel_build");
    // Clean any stale build artifacts from a prior run.
    if build_dir.exists() {
        let _ = fs::remove_dir_all(&build_dir);
    }

    prepare_kernel_build(kernel_src, &build_dir);
    let wrapper = create_bcc_wrapper(work_dir);

    // Determine available parallelism via /proc/cpuinfo (MSRV-compatible).
    let nproc = fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.matches("processor\t:").count())
        .unwrap_or(0)
        .max(1);

    let build_out = Command::new("make")
        .current_dir(kernel_src)
        .arg(format!("O={}", build_dir.display()))
        .arg(format!("ARCH={}", KERNEL_ARCH))
        .arg(format!("CC={}", wrapper.display()))
        .arg("HOSTCC=gcc")
        .arg(format!("-j{}", nproc))
        .env("LC_ALL", "C")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| {
            panic!("Failed to build kernel for QEMU boot test: {}", e)
        });

    assert!(
        build_out.status.success(),
        "Kernel build for QEMU boot test FAILED.\nstderr: {}",
        String::from_utf8_lossy(&build_out.stderr)
    );

    let vmlinux = build_dir.join("vmlinux");
    assert!(
        vmlinux.is_file(),
        "vmlinux not produced at '{}' after kernel build.",
        vmlinux.display()
    );

    vmlinux
}

// ===========================================================================
// Checkpoint 6 Tests
// ===========================================================================
//
// All tests carry `#[ignore]` because they require:
//   - A Linux 6.9 kernel source tree (KERNEL_SRC_DIR)
//   - qemu-system-riscv64
//   - The BCC binary built (`cargo build --release`)
//
// Run: KERNEL_SRC_DIR=/path/to/linux-6.9 cargo test --test checkpoint6_kernel -- --ignored

// ---------------------------------------------------------------------------
// Sub-gate: init/main.o
// ---------------------------------------------------------------------------

/// Checkpoint 6 sub-gate: compile `init/main.o` from the Linux kernel tree
/// with BCC and verify it is a valid RISC-V 64 relocatable ELF.
///
/// `init/main.o` is the kernel's primary entry point and exercises basic C
/// language features, preprocessor includes, and inline assembly.
#[test]
#[ignore]
fn test_kernel_init_main_o() {
    run_subgate_test("init/main.o");
}

// ---------------------------------------------------------------------------
// Sub-gate: kernel/sched/core.o
// ---------------------------------------------------------------------------

/// Checkpoint 6 sub-gate: compile `kernel/sched/core.o`.
///
/// The scheduler core exercises complex GCC extensions (statement expressions,
/// `typeof`, computed gotos for the preemption fast-path), heavy macro usage,
/// and inline assembly for context switching.
#[test]
#[ignore]
fn test_kernel_sched_core_o() {
    run_subgate_test("kernel/sched/core.o");
}

// ---------------------------------------------------------------------------
// Sub-gate: mm/memory.o
// ---------------------------------------------------------------------------

/// Checkpoint 6 sub-gate: compile `mm/memory.o`.
///
/// The memory management subsystem exercises pointer arithmetic, bitfield
/// manipulation, atomic operations, and architecture-specific inline assembly
/// for page table management.
#[test]
#[ignore]
fn test_kernel_mm_memory_o() {
    run_subgate_test("mm/memory.o");
}

// ---------------------------------------------------------------------------
// Sub-gate: fs/read_write.o
// ---------------------------------------------------------------------------

/// Checkpoint 6 sub-gate: compile `fs/read_write.o`.
///
/// The VFS read / write path exercises function pointers, designated
/// initializers, GCC attributes (`section`, `used`, `aligned`), and
/// conditional compilation via preprocessor macros.
#[test]
#[ignore]
fn test_kernel_fs_read_write_o() {
    run_subgate_test("fs/read_write.o");
}

// ---------------------------------------------------------------------------
// Full kernel build
// ---------------------------------------------------------------------------

/// Checkpoint 6 primary test: build the complete Linux kernel 6.9 for RISC-V
/// with BCC.
///
/// Validates:
/// 1. The build completes successfully.
/// 2. `vmlinux` is a valid RISC-V 64 ELF executable (`ET_EXEC`).
/// 3. Build time ≤ 5× GCC benchmark (Section 0.7.8).
/// 4. Expected sections (`.text`, `.rodata`, `.data`, `.bss`) are present.
/// 5. Expected kernel symbols (`start_kernel`) are present.
///
/// On failure the error is classified per Section 0.7.6 protocol.
#[test]
#[ignore]
fn test_full_kernel_build() {
    let kernel_src = require_kernel_source();
    let test_dir = common::TestDir::new("kernel_full_build");
    let build_dir = test_dir.file_path("build");

    // Prepare (defconfig + generated headers).
    prepare_kernel_build(&kernel_src, &build_dir);

    // Create BCC wrapper for RISC-V.
    let wrapper = create_bcc_wrapper(test_dir.path());

    // Determine parallelism level.
    // Determine available parallelism for the kernel build.
    // Read from /proc/cpuinfo (Linux) as a MSRV-compatible approach.
    let nproc = fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.matches("processor\t:").count())
        .unwrap_or(0)
        .max(1);

    // Time the full build for the 5× GCC ceiling check.
    let (build_result, build_duration) = common::timed_execution(|| {
        let out = Command::new("make")
            .current_dir(&kernel_src)
            .arg(format!("O={}", build_dir.display()))
            .arg(format!("ARCH={}", KERNEL_ARCH))
            .arg(format!("CC={}", wrapper.display()))
            .arg("HOSTCC=gcc")
            .arg(format!("-j{}", nproc))
            .env("LC_ALL", "C")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap_or_else(|e| {
                panic!("Failed to execute 'make' for full kernel build: {}", e)
            });

        common::BccOutput {
            status: out.status,
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    });

    eprintln!(
        "Full kernel build finished in {:.1}s.",
        build_duration.as_secs_f64()
    );

    // ---- Assert success ----
    if !build_result.success() {
        let report = format_failure_report("full kernel build", &build_result);
        panic!("Full kernel build FAILED.\n{}", report);
    }

    // ---- Validate vmlinux ----
    let vmlinux_path = build_dir.join("vmlinux");
    assert!(
        vmlinux_path.is_file(),
        "vmlinux not found at '{}' after build.",
        vmlinux_path.display()
    );

    let vmlinux_meta = fs::metadata(&vmlinux_path).unwrap_or_else(|e| {
        panic!("Cannot stat vmlinux: {}", e)
    });
    assert!(
        vmlinux_meta.len() > 0,
        "vmlinux at '{}' is empty.",
        vmlinux_path.display()
    );

    let vmlinux_str = vmlinux_path.to_str().unwrap();

    // Log ELF header.
    let header = common::readelf_header(vmlinux_str);
    eprintln!("vmlinux ELF header:\n{}", header);

    // ELF type = EXEC, machine = RISC-V, class = ELF64.
    common::assert_elf_type(vmlinux_str, "EXEC");
    common::assert_elf_machine(vmlinux_str, RISCV_MACHINE);
    common::assert_elf_class(vmlinux_str, ELF64_CLASS);

    // Expected sections.
    common::assert_section_exists(vmlinux_str, ".text");
    common::assert_section_exists(vmlinux_str, ".rodata");
    common::assert_section_exists(vmlinux_str, ".data");
    common::assert_section_exists(vmlinux_str, ".bss");

    // Log section table.
    let sections = common::readelf_sections(vmlinux_str);
    eprintln!("vmlinux sections:\n{}", sections);

    // Check key kernel symbols.
    let symbols = common::readelf_symbols(vmlinux_str);
    assert!(
        symbols.contains("start_kernel"),
        "vmlinux symbol table does not contain 'start_kernel'.\n\
         Symbols (first 4096 chars):\n{}",
        &symbols[..symbols.len().min(4096)]
    );

    // ---- 5× GCC wall-clock ceiling (Section 0.7.8) ----
    let gcc_secs = gcc_benchmark_secs();
    let ceiling_secs = gcc_secs.saturating_mul(WALL_CLOCK_MULTIPLIER);
    let actual_secs = build_duration.as_secs();

    eprintln!(
        "Build time: {}s | GCC benchmark: {}s | 5× ceiling: {}s",
        actual_secs, gcc_secs, ceiling_secs
    );

    assert!(
        actual_secs <= ceiling_secs,
        "Kernel build took {}s, exceeding the 5× GCC ceiling of {}s \
         (GCC benchmark: {}s).  Per Section 0.7.8, compiler performance \
         is insufficient.",
        actual_secs,
        ceiling_secs,
        gcc_secs
    );

    eprintln!("Full kernel build PASSED.");
}

// ---------------------------------------------------------------------------
// QEMU boot to userspace
// ---------------------------------------------------------------------------

/// Checkpoint 6 capstone test: boot the BCC-compiled Linux kernel in QEMU
/// and verify userspace entry.
///
/// 1. Builds a minimal `/init` binary with BCC for RISC-V 64.
/// 2. Packages `/init` into a cpio initramfs.
/// 3. Boots `vmlinux` in `qemu-system-riscv64` with serial console capture.
/// 4. Asserts that `"USERSPACE_OK"` appears in the serial output.
///
/// The QEMU process is killed after `QEMU_BOOT_TIMEOUT_SECS` if the kernel
/// hangs.
#[test]
#[ignore]
fn test_kernel_qemu_boot() {
    let kernel_src = require_kernel_source();
    let test_dir = common::TestDir::new("kernel_qemu_boot");

    // Locate (or build) vmlinux.
    let vmlinux = find_or_build_vmlinux(&kernel_src, test_dir.path());

    // Build the minimal initramfs with /init.
    let initramfs_work = test_dir.file_path("initramfs_work");
    fs::create_dir_all(&initramfs_work).unwrap_or_else(|e| {
        panic!("Failed to create initramfs work directory: {}", e)
    });
    let initramfs_cpio = build_initramfs(&initramfs_work);

    // Boot the kernel.
    eprintln!(
        "Booting kernel in qemu-system-riscv64 (timeout: {}s) …",
        QEMU_BOOT_TIMEOUT_SECS
    );
    let (serial_output, timed_out) = qemu_boot_kernel(
        vmlinux.as_path(),
        initramfs_cpio.as_path(),
        QEMU_BOOT_TIMEOUT_SECS,
    );

    // ---- Assert no timeout ----
    assert!(
        !timed_out,
        "QEMU boot timed out after {}s — kernel did not reach userspace.\n\
         Serial output (last 4096 chars):\n{}",
        QEMU_BOOT_TIMEOUT_SECS,
        &serial_output[serial_output.len().saturating_sub(4096)..]
    );

    // ---- Assert USERSPACE_OK marker ----
    assert!(
        serial_output.contains(USERSPACE_OK_MARKER),
        "Serial output does not contain '{}' — kernel failed to reach \
         userspace.\nFull serial output ({} bytes, first 8192 shown):\n{}",
        USERSPACE_OK_MARKER,
        serial_output.len(),
        &serial_output[..serial_output.len().min(8192)]
    );

    eprintln!(
        "SUCCESS: Kernel booted to userspace and printed '{}'.",
        USERSPACE_OK_MARKER
    );
}
