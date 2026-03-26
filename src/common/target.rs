//! Target architecture definitions for BCC.
//!
//! Defines the [`Target`] enumeration and per-target constants for all four
//! supported architectures: x86-64, i686, AArch64, and RISC-V 64.
//!
//! This module is **fully standalone** within `src/common/` — it has no
//! dependencies on any other BCC module. It provides:
//!
//! - Pointer widths, alignment, and data-model identifiers (LP64 vs ILP32)
//! - [`Endianness`] classification (all four targets are little-endian)
//! - ELF header constants (`e_machine`, `EI_CLASS`, `EI_DATA`, `e_flags`)
//! - Architecture-specific predefined preprocessor macros
//! - ABI-related constants (max alignment, page size, dynamic linker path)
//!
//! [`Target`] information flows from the CLI driver through every pipeline
//! stage — preprocessor, lexer, parser, semantic analysis, IR lowering,
//! code generation, assembler, and linker — to enable architecture-dependent
//! behavior at each level.

use std::fmt;

// ---------------------------------------------------------------------------
// Endianness
// ---------------------------------------------------------------------------

/// Byte ordering of a target architecture.
///
/// All four architectures currently supported by BCC are little-endian.
/// The `Big` variant is included for forward-compatibility should a
/// big-endian target be added in the future.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// Least-significant byte first (x86-64, i686, AArch64, RISC-V 64).
    Little,
    /// Most-significant byte first (currently unused).
    Big,
}

impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Endianness::Little => write!(f, "little-endian"),
            Endianness::Big => write!(f, "big-endian"),
        }
    }
}

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------

/// Supported compilation target architectures.
///
/// Each variant corresponds to a Linux ELF target for which BCC can produce
/// native executables (ET_EXEC) and shared objects (ET_DYN) using its
/// built-in assembler and linker.
///
/// # Data Models
///
/// | Variant   | Data Model | Pointer Width |
/// |-----------|-----------|---------------|
/// | `X86_64`  | LP64      | 8 bytes       |
/// | `I686`    | ILP32     | 4 bytes       |
/// | `AArch64` | LP64      | 8 bytes       |
/// | `RiscV64` | LP64      | 8 bytes       |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// x86-64 (AMD64) — System V AMD64 ABI, LP64 data model.
    X86_64,
    /// i686 (IA-32) — cdecl / System V i386 ABI, ILP32 data model.
    I686,
    /// AArch64 (ARM 64-bit) — AAPCS64 ABI, LP64 data model.
    AArch64,
    /// RISC-V 64 (RV64IMAFDC) — LP64D ABI.
    RiscV64,
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::X86_64 => write!(f, "x86-64"),
            Target::I686 => write!(f, "i686"),
            Target::AArch64 => write!(f, "aarch64"),
            Target::RiscV64 => write!(f, "riscv64"),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

impl Target {
    /// Parse a target architecture from a CLI string.
    ///
    /// Accepts the canonical target names as well as common aliases and
    /// GNU-style triple prefixes so that `--target=x86-64`,
    /// `--target=x86_64`, and `--target=x86-64-linux-gnu` all resolve to
    /// [`Target::X86_64`].
    ///
    /// Returns `None` for unrecognised target strings.
    ///
    /// # Examples
    ///
    /// ```
    /// use bcc::common::target::Target;
    ///
    /// assert_eq!(Target::from_str("x86-64"), Some(Target::X86_64));
    /// assert_eq!(Target::from_str("aarch64-linux-gnu"), Some(Target::AArch64));
    /// assert_eq!(Target::from_str("unknown"), None);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Target> {
        // Normalise: lower-case the input so matching is case-insensitive.
        let lower = s.to_ascii_lowercase();
        let lower = lower.as_str();

        match lower {
            // x86-64 variants
            "x86-64" | "x86_64" | "amd64" | "x86-64-linux-gnu" | "x86_64-linux-gnu" => {
                Some(Target::X86_64)
            }
            // i686 / i386 variants
            "i686" | "i386" | "i486" | "i586" | "i686-linux-gnu" | "i386-linux-gnu" => {
                Some(Target::I686)
            }
            // AArch64 variants
            "aarch64" | "arm64" | "aarch64-linux-gnu" | "arm64-linux-gnu" => Some(Target::AArch64),
            // RISC-V 64 variants
            "riscv64" | "riscv64gc" | "riscv64-linux-gnu" | "riscv64gc-linux-gnu" => {
                Some(Target::RiscV64)
            }
            // Unrecognised target string.
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Data-Model Constants
// ---------------------------------------------------------------------------

impl Target {
    /// Width of a pointer (and `size_t`, `ptrdiff_t`, `intptr_t`) in bytes.
    ///
    /// - ILP32 (i686): 4 bytes
    /// - LP64  (x86-64, AArch64, RISC-V 64): 8 bytes
    #[inline]
    pub fn pointer_width(&self) -> usize {
        match self {
            Target::I686 => 4,
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => 8,
        }
    }

    /// Natural alignment of a pointer in bytes.
    ///
    /// Matches [`pointer_width`](Target::pointer_width) on all supported
    /// targets.
    #[inline]
    pub fn pointer_align(&self) -> usize {
        self.pointer_width()
    }

    /// Returns `true` if the target has a 64-bit pointer width.
    #[inline]
    pub fn is_64bit(&self) -> bool {
        self.pointer_width() == 8
    }

    /// Byte order of the target.
    ///
    /// All four supported targets are little-endian.
    #[inline]
    pub fn endianness(&self) -> Endianness {
        // All currently supported architectures are little-endian.
        // Should a big-endian target be added (e.g. s390x), this match
        // would dispatch per variant.
        match self {
            Target::X86_64 | Target::I686 | Target::AArch64 | Target::RiscV64 => Endianness::Little,
        }
    }

    /// Size of `long` (and `unsigned long`) in bytes.
    ///
    /// - ILP32 (i686): 4 bytes
    /// - LP64  (x86-64, AArch64, RISC-V 64): 8 bytes
    #[inline]
    pub fn long_size(&self) -> usize {
        match self {
            Target::I686 => 4,
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => 8,
        }
    }

    /// Size of `long double` in bytes, **including** padding.
    ///
    /// - i686:  12 bytes (80-bit x87 extended precision, padded to 12)
    /// - x86-64: 16 bytes (80-bit extended with 16-byte alignment padding)
    /// - AArch64: 16 bytes (IEEE 754 128-bit quad precision)
    /// - RISC-V 64: 16 bytes (IEEE 754 128-bit quad precision)
    #[inline]
    pub fn long_double_size(&self) -> usize {
        match self {
            Target::I686 => 12,
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => 16,
        }
    }

    /// Required alignment of `long double` in bytes.
    ///
    /// - i686:   4 bytes (System V i386 ABI)
    /// - Others: 16 bytes
    #[inline]
    pub fn long_double_align(&self) -> usize {
        match self {
            Target::I686 => 4,
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => 16,
        }
    }
}

// ---------------------------------------------------------------------------
// ELF Constants
// ---------------------------------------------------------------------------

impl Target {
    /// ELF `e_machine` header value for this target.
    ///
    /// Values are drawn from the ELF specification and `/usr/include/elf.h`:
    ///
    /// | Target   | Value | Constant      |
    /// |----------|-------|---------------|
    /// | x86-64   | 62    | `EM_X86_64`   |
    /// | i686     | 3     | `EM_386`       |
    /// | AArch64  | 183   | `EM_AARCH64`  |
    /// | RISC-V   | 243   | `EM_RISCV`    |
    #[inline]
    pub fn elf_machine(&self) -> u16 {
        match self {
            Target::X86_64 => 62,   // EM_X86_64
            Target::I686 => 3,      // EM_386
            Target::AArch64 => 183, // EM_AARCH64
            Target::RiscV64 => 243, // EM_RISCV
        }
    }

    /// ELF `EI_CLASS` byte — 32-bit or 64-bit object.
    ///
    /// - i686: `1` (ELFCLASS32)
    /// - Others: `2` (ELFCLASS64)
    #[inline]
    pub fn elf_class(&self) -> u8 {
        match self {
            Target::I686 => 1,                                       // ELFCLASS32
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => 2, // ELFCLASS64
        }
    }

    /// ELF `EI_DATA` byte — byte order.
    ///
    /// All supported targets use `1` (ELFDATA2LSB, little-endian).
    #[inline]
    pub fn elf_data(&self) -> u8 {
        // ELFDATA2LSB = 1 (little-endian) for every supported target.
        1
    }

    /// Architecture-specific ELF `e_flags` header value.
    ///
    /// - RISC-V 64: `0x0005` — `EF_RISCV_FLOAT_ABI_DOUBLE` (0x0004) |
    ///   `EF_RISCV_RVC` (0x0001)
    /// - All others: `0`
    #[inline]
    pub fn elf_flags(&self) -> u32 {
        match self {
            // EF_RISCV_RVC (0x1): objects may contain compressed (C extension)
            // instructions.  Float ABI bits are 0 → soft-float (lp64), which
            // is the ABI used by the Linux kernel (`-mabi=lp64`).  Using
            // double-float (0x4) here would prevent linking with kernel
            // objects that use the soft-float ABI.
            Target::RiscV64 => 0x0001, // EF_RISCV_RVC only; soft-float ABI
            Target::X86_64 | Target::I686 | Target::AArch64 => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Predefined Preprocessor Macros
// ---------------------------------------------------------------------------

impl Target {
    /// Returns the set of predefined preprocessor macros for this target.
    ///
    /// The returned vector contains `(name, replacement)` tuples that should
    /// be injected into the preprocessor environment before any user source
    /// is processed.
    ///
    /// Common macros (present for all targets):
    /// - `__STDC__`, `__STDC_VERSION__`, `__STDC_HOSTED__`
    /// - `__linux__`, `__unix__`, `__ELF__`
    ///
    /// Architecture-specific macros are appended according to the target.
    pub fn predefined_macros(&self) -> Vec<(&'static str, &'static str)> {
        // Common macros shared across all supported targets.
        let mut macros: Vec<(&'static str, &'static str)> = vec![
            ("__STDC__", "1"),
            ("__STDC_VERSION__", "201112L"), // C11
            ("__STDC_HOSTED__", "1"),
            ("__linux__", "1"),
            ("__linux", "1"),
            ("linux", "1"),
            ("__unix__", "1"),
            ("__unix", "1"),
            ("unix", "1"),
            ("__ELF__", "1"),
        ];

        // Architecture-specific macros.
        match self {
            Target::X86_64 => {
                macros.extend_from_slice(&[
                    ("__x86_64__", "1"),
                    ("__x86_64", "1"),
                    ("__amd64__", "1"),
                    ("__amd64", "1"),
                    ("__LP64__", "1"),
                    ("_LP64", "1"),
                    ("__SSE__", "1"),
                    ("__SSE2__", "1"),
                ]);
            }
            Target::I686 => {
                macros.extend_from_slice(&[
                    ("__i386__", "1"),
                    ("__i386", "1"),
                    ("i386", "1"),
                    ("__i686__", "1"),
                    ("__ILP32__", "1"),
                ]);
            }
            Target::AArch64 => {
                macros.extend_from_slice(&[
                    ("__aarch64__", "1"),
                    ("__ARM_64BIT_STATE", "1"),
                    ("__ARM_ARCH", "8"),
                    ("__ARM_ARCH_ISA_A64", "1"),
                    ("__LP64__", "1"),
                    ("_LP64", "1"),
                ]);
            }
            Target::RiscV64 => {
                macros.extend_from_slice(&[
                    ("__riscv", "1"),
                    ("__riscv_xlen", "64"),
                    ("__riscv_flen", "64"),
                    ("__riscv_float_abi_double", "1"),
                    ("__riscv_mul", "1"),
                    ("__riscv_div", "1"),
                    ("__riscv_atomic", "1"),
                    ("__riscv_compressed", "1"),
                    // Code model — default to medany (position-independent,
                    // ±2 GiB PC-relative addressing) which is the standard
                    // model for kernel code.
                    ("__riscv_cmodel_medany", "1"),
                    ("__LP64__", "1"),
                    ("_LP64", "1"),
                ]);
            }
        }

        macros
    }
}

// ---------------------------------------------------------------------------
// ABI-Related Constants
// ---------------------------------------------------------------------------

impl Target {
    /// Maximum natural alignment for any fundamental type on this target.
    ///
    /// All four supported targets use 16-byte maximum alignment (for SSE,
    /// NEON, or vector register types).  This value also governs the
    /// alignment of `max_align_t`.
    #[inline]
    pub fn max_align(&self) -> usize {
        16
    }

    /// Virtual-memory page size in bytes.
    ///
    /// Used for stack-probe loop calculations (frames exceeding one page
    /// must touch each intervening page to avoid skipping the guard page).
    /// All four targets default to 4 KiB pages.
    #[inline]
    pub fn page_size(&self) -> usize {
        4096
    }

    /// Filesystem path to the dynamic linker (ELF interpreter) for this
    /// target, used to populate the `PT_INTERP` program header when
    /// producing dynamically-linked executables.
    #[inline]
    pub fn dynamic_linker(&self) -> &'static str {
        match self {
            Target::X86_64 => "/lib64/ld-linux-x86-64.so.2",
            Target::I686 => "/lib/ld-linux.so.2",
            Target::AArch64 => "/lib/ld-linux-aarch64.so.1",
            Target::RiscV64 => "/lib/ld-linux-riscv64-lp64d.so.1",
        }
    }

    /// Default entry-point symbol name for static executables.
    ///
    /// The built-in linker resolves this symbol as the ELF `e_entry`
    /// address when no explicit entry point is specified.
    #[inline]
    pub fn default_entry_point(&self) -> &'static str {
        "_start"
    }

    /// Returns the ordered list of system include paths appropriate for
    /// this target architecture.
    ///
    /// When cross-compiling (e.g. `--target=i686` on an x86-64 host), the
    /// architecture-specific multiarch directory (e.g.
    /// `/usr/include/i386-linux-gnu`) must be used instead of the host's
    /// directory (e.g. `/usr/include/x86_64-linux-gnu`), because glibc's
    /// `gnu/stubs.h` conditionally includes architecture-specific stub
    /// headers (e.g. `stubs-32.h` vs `stubs-64.h`) that are only present
    /// under the correct multiarch directory.
    ///
    /// The order is:
    /// 1. Architecture-specific multiarch path
    /// 2. Generic `/usr/include`
    pub fn system_include_paths(&self) -> Vec<&'static str> {
        // Primary: target-specific include path.
        // Fallback: x86_64-linux-gnu path (the host arch) so that when
        // cross-compiling on an x86-64 host without cross-compilation
        // headers installed, the host's `bits/` headers are still found.
        // This matches GCC's behavior of falling back to host headers.
        //
        // GCC internal headers (float.h, stddef.h, stdarg.h, etc.) live
        // in /usr/lib/gcc[-cross]/<triple>/<version>/include.  We include
        // these so that system headers that chain via `#include_next`
        // (e.g., glibc's <limits.h> → GCC's <limits.h>) resolve correctly.
        match self {
            Target::X86_64 => vec![
                "/usr/lib/gcc/x86_64-linux-gnu/13/include",
                "/usr/lib/gcc/x86_64-linux-gnu/14/include",
                "/usr/include/x86_64-linux-gnu",
                "/usr/include",
            ],
            Target::I686 => vec![
                "/usr/lib/gcc/x86_64-linux-gnu/13/include",
                "/usr/lib/gcc/x86_64-linux-gnu/14/include",
                "/usr/include/i386-linux-gnu",
                "/usr/include/x86_64-linux-gnu",
                "/usr/include",
            ],
            Target::AArch64 => vec![
                "/usr/lib/gcc-cross/aarch64-linux-gnu/13/include",
                "/usr/lib/gcc-cross/aarch64-linux-gnu/14/include",
                "/usr/lib/gcc/x86_64-linux-gnu/13/include",
                "/usr/include/aarch64-linux-gnu",
                "/usr/include/x86_64-linux-gnu",
                "/usr/include",
            ],
            Target::RiscV64 => vec![
                "/usr/lib/gcc-cross/riscv64-linux-gnu/13/include",
                "/usr/lib/gcc-cross/riscv64-linux-gnu/14/include",
                "/usr/lib/gcc/x86_64-linux-gnu/13/include",
                "/usr/include/riscv64-linux-gnu",
                "/usr/include/x86_64-linux-gnu",
                "/usr/include",
            ],
        }
    }

    /// Returns the ordered list of system library paths appropriate for
    /// this target architecture.
    ///
    /// Used by the built-in linker to locate shared libraries (e.g.
    /// `libc.so.6`) when resolving `-l` flags.
    pub fn system_library_paths(&self) -> Vec<&'static str> {
        match self {
            Target::X86_64 => vec![
                "/usr/lib/x86_64-linux-gnu",
                "/lib/x86_64-linux-gnu",
                "/usr/lib64",
                "/lib64",
            ],
            Target::I686 => vec![
                "/usr/lib/i386-linux-gnu",
                "/lib/i386-linux-gnu",
                "/usr/lib32",
                "/lib32",
            ],
            Target::AArch64 => vec![
                "/usr/lib/aarch64-linux-gnu",
                "/lib/aarch64-linux-gnu",
                "/usr/lib",
                "/lib",
            ],
            Target::RiscV64 => vec![
                "/usr/lib/riscv64-linux-gnu",
                "/lib/riscv64-linux-gnu",
                "/usr/lib",
                "/lib",
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Target::from_str ---------------------------------------------------

    #[test]
    fn from_str_x86_64_variants() {
        assert_eq!(Target::from_str("x86-64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("x86_64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("amd64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("x86-64-linux-gnu"), Some(Target::X86_64));
        assert_eq!(Target::from_str("x86_64-linux-gnu"), Some(Target::X86_64));
    }

    #[test]
    fn from_str_i686_variants() {
        assert_eq!(Target::from_str("i686"), Some(Target::I686));
        assert_eq!(Target::from_str("i386"), Some(Target::I686));
        assert_eq!(Target::from_str("i486"), Some(Target::I686));
        assert_eq!(Target::from_str("i586"), Some(Target::I686));
        assert_eq!(Target::from_str("i686-linux-gnu"), Some(Target::I686));
        assert_eq!(Target::from_str("i386-linux-gnu"), Some(Target::I686));
    }

    #[test]
    fn from_str_aarch64_variants() {
        assert_eq!(Target::from_str("aarch64"), Some(Target::AArch64));
        assert_eq!(Target::from_str("arm64"), Some(Target::AArch64));
        assert_eq!(Target::from_str("aarch64-linux-gnu"), Some(Target::AArch64));
        assert_eq!(Target::from_str("arm64-linux-gnu"), Some(Target::AArch64));
    }

    #[test]
    fn from_str_riscv64_variants() {
        assert_eq!(Target::from_str("riscv64"), Some(Target::RiscV64));
        assert_eq!(Target::from_str("riscv64gc"), Some(Target::RiscV64));
        assert_eq!(Target::from_str("riscv64-linux-gnu"), Some(Target::RiscV64));
        assert_eq!(
            Target::from_str("riscv64gc-linux-gnu"),
            Some(Target::RiscV64)
        );
    }

    #[test]
    fn from_str_case_insensitive() {
        assert_eq!(Target::from_str("X86-64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("AARCH64"), Some(Target::AArch64));
        assert_eq!(Target::from_str("RISCV64"), Some(Target::RiscV64));
        assert_eq!(Target::from_str("I686"), Some(Target::I686));
    }

    #[test]
    fn from_str_unknown() {
        assert_eq!(Target::from_str("unknown"), None);
        assert_eq!(Target::from_str(""), None);
        assert_eq!(Target::from_str("mips64"), None);
        assert_eq!(Target::from_str("sparc"), None);
    }

    // -- Display ------------------------------------------------------------

    #[test]
    fn display_target() {
        assert_eq!(format!("{}", Target::X86_64), "x86-64");
        assert_eq!(format!("{}", Target::I686), "i686");
        assert_eq!(format!("{}", Target::AArch64), "aarch64");
        assert_eq!(format!("{}", Target::RiscV64), "riscv64");
    }

    #[test]
    fn display_endianness() {
        assert_eq!(format!("{}", Endianness::Little), "little-endian");
        assert_eq!(format!("{}", Endianness::Big), "big-endian");
    }

    // -- Pointer width & alignment ------------------------------------------

    #[test]
    fn pointer_width() {
        assert_eq!(Target::X86_64.pointer_width(), 8);
        assert_eq!(Target::I686.pointer_width(), 4);
        assert_eq!(Target::AArch64.pointer_width(), 8);
        assert_eq!(Target::RiscV64.pointer_width(), 8);
    }

    #[test]
    fn pointer_align_matches_width() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(target.pointer_align(), target.pointer_width());
        }
    }

    #[test]
    fn is_64bit() {
        assert!(Target::X86_64.is_64bit());
        assert!(!Target::I686.is_64bit());
        assert!(Target::AArch64.is_64bit());
        assert!(Target::RiscV64.is_64bit());
    }

    // -- Endianness ---------------------------------------------------------

    #[test]
    fn all_targets_little_endian() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(target.endianness(), Endianness::Little);
        }
    }

    // -- long / long double -------------------------------------------------

    #[test]
    fn long_size() {
        assert_eq!(Target::I686.long_size(), 4);
        assert_eq!(Target::X86_64.long_size(), 8);
        assert_eq!(Target::AArch64.long_size(), 8);
        assert_eq!(Target::RiscV64.long_size(), 8);
    }

    #[test]
    fn long_double_size() {
        assert_eq!(Target::I686.long_double_size(), 12);
        assert_eq!(Target::X86_64.long_double_size(), 16);
        assert_eq!(Target::AArch64.long_double_size(), 16);
        assert_eq!(Target::RiscV64.long_double_size(), 16);
    }

    #[test]
    fn long_double_align() {
        assert_eq!(Target::I686.long_double_align(), 4);
        assert_eq!(Target::X86_64.long_double_align(), 16);
        assert_eq!(Target::AArch64.long_double_align(), 16);
        assert_eq!(Target::RiscV64.long_double_align(), 16);
    }

    // -- ELF constants ------------------------------------------------------

    #[test]
    fn elf_machine() {
        assert_eq!(Target::X86_64.elf_machine(), 62); // EM_X86_64
        assert_eq!(Target::I686.elf_machine(), 3); // EM_386
        assert_eq!(Target::AArch64.elf_machine(), 183); // EM_AARCH64
        assert_eq!(Target::RiscV64.elf_machine(), 243); // EM_RISCV
    }

    #[test]
    fn elf_class() {
        assert_eq!(Target::I686.elf_class(), 1); // ELFCLASS32
        assert_eq!(Target::X86_64.elf_class(), 2); // ELFCLASS64
        assert_eq!(Target::AArch64.elf_class(), 2); // ELFCLASS64
        assert_eq!(Target::RiscV64.elf_class(), 2); // ELFCLASS64
    }

    #[test]
    fn elf_data_all_little_endian() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(target.elf_data(), 1); // ELFDATA2LSB
        }
    }

    #[test]
    fn elf_flags() {
        assert_eq!(Target::X86_64.elf_flags(), 0);
        assert_eq!(Target::I686.elf_flags(), 0);
        assert_eq!(Target::AArch64.elf_flags(), 0);
        assert_eq!(Target::RiscV64.elf_flags(), 0x0001);
    }

    // -- Predefined macros --------------------------------------------------

    #[test]
    fn predefined_macros_common() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let macros = target.predefined_macros();
            let names: Vec<&str> = macros.iter().map(|(n, _)| *n).collect();
            assert!(names.contains(&"__STDC__"), "missing __STDC__");
            assert!(
                names.contains(&"__STDC_VERSION__"),
                "missing __STDC_VERSION__"
            );
            assert!(
                names.contains(&"__STDC_HOSTED__"),
                "missing __STDC_HOSTED__"
            );
            assert!(names.contains(&"__linux__"), "missing __linux__");
            assert!(names.contains(&"__unix__"), "missing __unix__");
            assert!(names.contains(&"__ELF__"), "missing __ELF__");

            // Verify __STDC_VERSION__ value is C11.
            let version = macros
                .iter()
                .find(|(n, _)| *n == "__STDC_VERSION__")
                .unwrap()
                .1;
            assert_eq!(version, "201112L");
        }
    }

    #[test]
    fn predefined_macros_x86_64() {
        let macros = Target::X86_64.predefined_macros();
        let names: Vec<&str> = macros.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"__x86_64__"));
        assert!(names.contains(&"__x86_64"));
        assert!(names.contains(&"__amd64__"));
        assert!(names.contains(&"__amd64"));
        assert!(names.contains(&"__LP64__"));
        assert!(names.contains(&"_LP64"));
    }

    #[test]
    fn predefined_macros_i686() {
        let macros = Target::I686.predefined_macros();
        let names: Vec<&str> = macros.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"__i386__"));
        assert!(names.contains(&"__i386"));
        assert!(names.contains(&"__i686__"));
        assert!(names.contains(&"i386"));
        assert!(names.contains(&"__ILP32__"));
    }

    #[test]
    fn predefined_macros_aarch64() {
        let macros = Target::AArch64.predefined_macros();
        let names: Vec<&str> = macros.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"__aarch64__"));
        assert!(names.contains(&"__ARM_64BIT_STATE"));
        assert!(names.contains(&"__ARM_ARCH"));
        assert!(names.contains(&"__LP64__"));
        assert!(names.contains(&"_LP64"));
    }

    #[test]
    fn predefined_macros_riscv64() {
        let macros = Target::RiscV64.predefined_macros();
        let names: Vec<&str> = macros.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"__riscv"));
        assert!(names.contains(&"__LP64__"));
        assert!(names.contains(&"_LP64"));

        // Verify xlen / flen values.
        let xlen = macros.iter().find(|(n, _)| *n == "__riscv_xlen").unwrap().1;
        assert_eq!(xlen, "64");
        let flen = macros.iter().find(|(n, _)| *n == "__riscv_flen").unwrap().1;
        assert_eq!(flen, "64");
    }

    // -- ABI constants ------------------------------------------------------

    #[test]
    fn max_align_all_targets() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(target.max_align(), 16);
        }
    }

    #[test]
    fn page_size_all_targets() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(target.page_size(), 4096);
        }
    }

    #[test]
    fn dynamic_linker() {
        assert_eq!(
            Target::X86_64.dynamic_linker(),
            "/lib64/ld-linux-x86-64.so.2"
        );
        assert_eq!(Target::I686.dynamic_linker(), "/lib/ld-linux.so.2");
        assert_eq!(
            Target::AArch64.dynamic_linker(),
            "/lib/ld-linux-aarch64.so.1"
        );
        assert_eq!(
            Target::RiscV64.dynamic_linker(),
            "/lib/ld-linux-riscv64-lp64d.so.1"
        );
    }

    #[test]
    fn default_entry_point() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(target.default_entry_point(), "_start");
        }
    }

    // -- Trait derivations --------------------------------------------------

    #[test]
    fn target_clone_copy_eq_hash() {
        let a = Target::X86_64;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);

        // Hash — ensure different targets produce different hashes (probabilistic).
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Target::X86_64);
        set.insert(Target::I686);
        set.insert(Target::AArch64);
        set.insert(Target::RiscV64);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn endianness_clone_copy_eq() {
        let a = Endianness::Little;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(Endianness::Little, Endianness::Big);
    }
}
