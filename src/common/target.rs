//! Target architecture definitions and constants for the BCC compiler.
//!
//! This module defines the four supported target architectures (x86-64, i686,
//! AArch64, RISC-V 64) along with their architecture-specific properties
//! including pointer width, endianness, predefined preprocessor macro sets,
//! data model (LP64 vs ILP32), stack alignment, ELF machine constants,
//! and dynamic linker paths.
//!
//! Target information flows from CLI argument parsing through every pipeline
//! stage: preprocessor (architecture defines), parser (sizeof resolution),
//! semantic analysis (struct layout), and code generation (architecture dispatch).

use std::fmt;

// ---------------------------------------------------------------------------
// ELF Header Constants
// ---------------------------------------------------------------------------

/// ELF `e_machine` value: Intel 80386 (i686 / IA-32).
const EM_386: u16 = 3;

/// ELF `e_machine` value: AMD x86-64 architecture.
const EM_X86_64: u16 = 62;

/// ELF `e_machine` value: ARM AARCH64 (64-bit ARM).
const EM_AARCH64: u16 = 183;

/// ELF `e_machine` value: RISC-V.
const EM_RISCV: u16 = 243;

/// ELF class for 32-bit object files (`ELFCLASS32`).
const ELFCLASS32: u8 = 1;

/// ELF class for 64-bit object files (`ELFCLASS64`).
const ELFCLASS64: u8 = 2;

/// RISC-V ELF flag indicating the compressed instruction extension (RVC / C).
const EF_RISCV_RVC: u32 = 0x0001;

/// RISC-V ELF flag indicating the double-precision floating-point ABI.
const EF_RISCV_FLOAT_ABI_DOUBLE: u32 = 0x0004;

// ---------------------------------------------------------------------------
// DataModel
// ---------------------------------------------------------------------------

/// Data model describing the sizes of fundamental C integer types.
///
/// The data model determines the relationship between `int`, `long`, and
/// pointer sizes, which directly affects struct layout, ABI conventions,
/// and code generation across every stage of the compiler pipeline.
///
/// # Variants
///
/// - [`LP64`](DataModel::LP64) — `long` and pointer are 64-bit; `int` is 32-bit.
///   Used on x86-64, AArch64, and RISC-V 64 Linux targets.
/// - [`ILP32`](DataModel::ILP32) — `int`, `long`, and pointer are all 32-bit.
///   Used on i686 Linux targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataModel {
    /// LP64: `long` = 64 bits, `int` = 32 bits, pointer = 64 bits.
    LP64,
    /// ILP32: `int` = 32 bits, `long` = 32 bits, pointer = 32 bits.
    ILP32,
}

impl fmt::Display for DataModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataModel::LP64 => write!(f, "LP64"),
            DataModel::ILP32 => write!(f, "ILP32"),
        }
    }
}

// ---------------------------------------------------------------------------
// Endianness
// ---------------------------------------------------------------------------

/// Byte order of the target architecture.
///
/// All four currently supported BCC targets (x86-64, i686, AArch64, RISC-V 64)
/// use little-endian byte order on Linux.  The [`Big`](Endianness::Big) variant
/// is included for completeness and to support potential future architectures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// Least-significant byte stored at the lowest memory address.
    Little,
    /// Most-significant byte stored at the lowest memory address.
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

/// Supported target architectures for the BCC compiler.
///
/// Each variant represents a distinct instruction-set architecture (ISA) with
/// its own ABI, register file, instruction encoding, and ELF conventions.
/// Target information is queried by every compilation pipeline stage to ensure
/// architecture-correct behaviour.
///
/// # Derived Traits
///
/// `Clone`, `Copy`, `Debug`, `PartialEq`, `Eq`, `Hash` — allowing efficient
/// storage in collections and use as match discriminants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// AMD64 / x86-64: 64-bit extension of x86.
    /// System V AMD64 ABI, variable-length instruction encoding,
    /// 16 GPRs (RAX–R15), 16 SSE registers (XMM0–XMM15).
    X86_64,

    /// Intel 80686 / IA-32: 32-bit x86.
    /// cdecl / System V i386 ABI, 8 GPRs (EAX–EDI), x87 FPU stack.
    I686,

    /// ARM 64-bit Architecture (ARMv8-A).
    /// AAPCS64 ABI, fixed 32-bit instruction width,
    /// 31 GPRs (X0–X30), 32 SIMD/FP registers (V0–V31).
    AArch64,

    /// RISC-V 64-bit with RV64IMAFDC ISA extensions.
    /// LP64D ABI, 32 integer registers (x0–x31),
    /// 32 floating-point registers (f0–f31).
    RiscV64,
}

impl Target {
    // -------------------------------------------------------------------
    // Size / alignment queries
    // -------------------------------------------------------------------

    /// Returns the pointer width in bytes for this target.
    ///
    /// * 64-bit targets (x86-64, AArch64, RISC-V 64): **8** bytes
    /// * 32-bit targets (i686): **4** bytes
    #[inline]
    pub fn pointer_width(&self) -> u32 {
        match self {
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => 8,
            Target::I686 => 4,
        }
    }

    /// Returns the natural alignment of a pointer in bytes.
    ///
    /// Pointer alignment matches pointer width on every supported architecture,
    /// guaranteeing naturally-aligned pointer loads and stores.
    #[inline]
    pub fn pointer_align(&self) -> u32 {
        self.pointer_width()
    }

    /// Returns the data model for this target.
    ///
    /// * [`LP64`](DataModel::LP64): x86-64, AArch64, RISC-V 64
    /// * [`ILP32`](DataModel::ILP32): i686
    #[inline]
    pub fn data_model(&self) -> DataModel {
        match self {
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => DataModel::LP64,
            Target::I686 => DataModel::ILP32,
        }
    }

    /// Returns the byte order for this target.
    ///
    /// All four supported targets use **little-endian** byte order on Linux.
    #[inline]
    pub fn endianness(&self) -> Endianness {
        // x86/x86-64 — inherently little-endian.
        // AArch64    — defaults to little-endian on Linux (bi-endian hardware).
        // RISC-V     — little-endian by convention on Linux.
        Endianness::Little
    }

    /// Returns the required stack alignment in bytes at function-call boundaries.
    ///
    /// All supported targets mandate **16-byte** stack alignment as specified by
    /// their respective System V ABI documents:
    ///
    /// * x86-64  — System V AMD64 ABI §3.2.2
    /// * i686    — System V i386 ABI (modern revision, GCC ≥ 4.5 default)
    /// * AArch64 — AAPCS64 §5.2.2.1
    /// * RISC-V  — RISC-V Calling Convention §1
    #[inline]
    pub fn stack_alignment(&self) -> u32 {
        // All four architectures share the same 16-byte call-site alignment.
        16
    }

    /// Returns the size of the C `long` type in bytes.
    ///
    /// * LP64 targets: **8** bytes (64-bit `long`)
    /// * ILP32 targets: **4** bytes (32-bit `long`)
    #[inline]
    pub fn long_size(&self) -> u32 {
        match self.data_model() {
            DataModel::LP64 => 8,
            DataModel::ILP32 => 4,
        }
    }

    /// Returns the storage size of the C `long double` type in bytes.
    ///
    /// The size varies by architecture:
    ///
    /// | Target     | Size | Representation |
    /// |------------|------|----------------|
    /// | x86-64     | 16   | 80-bit x87 extended, padded to 16 for alignment |
    /// | i686       | 12   | 80-bit x87 extended, padded to 12 |
    /// | AArch64    | 8    | Mapped to IEEE 754 double precision |
    /// | RISC-V 64  | 8    | Mapped to IEEE 754 double precision |
    #[inline]
    pub fn long_double_size(&self) -> u32 {
        match self {
            Target::X86_64 => 16,
            Target::I686 => 12,
            Target::AArch64 | Target::RiscV64 => 8,
        }
    }

    /// Returns the maximum width in bytes for lock-free atomic operations.
    ///
    /// This determines the largest type for which `_Atomic` operations can be
    /// performed without a library call (e.g. `libatomic`):
    ///
    /// | Target     | Width | Mechanism |
    /// |------------|-------|-----------|
    /// | x86-64     | 16    | `CMPXCHG16B` |
    /// | i686       | 8     | `CMPXCHG8B`  |
    /// | AArch64    | 16    | `LDXP`/`STXP` pair |
    /// | RISC-V 64  | 8     | RV64A atomic instructions |
    #[inline]
    pub fn max_atomic_width(&self) -> u32 {
        match self {
            Target::X86_64 | Target::AArch64 => 16,
            Target::I686 | Target::RiscV64 => 8,
        }
    }

    // -------------------------------------------------------------------
    // Predefined preprocessor macros
    // -------------------------------------------------------------------

    /// Returns the set of predefined preprocessor macros for this target.
    ///
    /// Each element is a `(name, value)` pair injected into the preprocessor
    /// macro table before processing any user source code.  The set includes:
    ///
    /// 1. **C11 standard** macros (`__STDC__`, `__STDC_VERSION__`, etc.)
    /// 2. **Platform** macros (`__linux__`, `__ELF__`, `__unix__`, etc.)
    /// 3. **Architecture-specific** macros (e.g. `__x86_64__`, `__aarch64__`)
    /// 4. **Compiler identification** (`__BCC__`)
    /// 5. **`__SIZEOF_*__`** type-size macros and byte-order macros
    pub fn predefined_macros(&self) -> Vec<(&'static str, &'static str)> {
        // Pre-allocate enough capacity for all macros to avoid repeated growth.
        let mut macros = Vec::with_capacity(40);

        // ---- C standard compliance ----
        macros.push(("__STDC__", "1"));
        macros.push(("__STDC_VERSION__", "201112L"));
        macros.push(("__STDC_HOSTED__", "1"));

        // ---- Linux / ELF platform ----
        macros.push(("__linux__", "1"));
        macros.push(("__linux", "1"));
        macros.push(("linux", "1"));
        macros.push(("__gnu_linux__", "1"));
        macros.push(("__ELF__", "1"));
        macros.push(("__unix__", "1"));
        macros.push(("__unix", "1"));
        macros.push(("unix", "1"));

        // ---- Architecture-specific ----
        match self {
            Target::X86_64 => {
                macros.push(("__x86_64__", "1"));
                macros.push(("__x86_64", "1"));
                macros.push(("__amd64__", "1"));
                macros.push(("__amd64", "1"));
                macros.push(("__LP64__", "1"));
                macros.push(("_LP64", "1"));
            }
            Target::I686 => {
                macros.push(("__i386__", "1"));
                macros.push(("__i386", "1"));
                macros.push(("i386", "1"));
                macros.push(("__i686__", "1"));
                macros.push(("__i686", "1"));
                macros.push(("__ILP32__", "1"));
                macros.push(("_ILP32", "1"));
            }
            Target::AArch64 => {
                macros.push(("__aarch64__", "1"));
                macros.push(("__LP64__", "1"));
                macros.push(("_LP64", "1"));
                macros.push(("__ARM_64BIT_STATE", "1"));
                macros.push(("__ARM_ARCH", "8"));
                macros.push(("__ARM_ARCH_ISA_A64", "1"));
            }
            Target::RiscV64 => {
                macros.push(("__riscv", "1"));
                macros.push(("__riscv_xlen", "64"));
                macros.push(("__riscv_flen", "64"));
                macros.push(("__riscv_float_abi_double", "1"));
                macros.push(("__riscv_mul", "1"));
                macros.push(("__riscv_muldiv", "1"));
                macros.push(("__riscv_div", "1"));
                macros.push(("__riscv_atomic", "1"));
                macros.push(("__riscv_compressed", "1"));
                macros.push(("__LP64__", "1"));
                macros.push(("_LP64", "1"));
            }
        }

        // ---- Compiler identification ----
        macros.push(("__BCC__", "1"));

        // ---- GCC version compatibility (required by Linux kernel headers) ----
        macros.push(("__GNUC__", "12"));
        macros.push(("__GNUC_MINOR__", "0"));
        macros.push(("__GNUC_PATCHLEVEL__", "0"));

        // ---- Type size macros (target-dependent) ----
        match self.data_model() {
            DataModel::LP64 => {
                macros.push(("__SIZEOF_LONG__", "8"));
                macros.push(("__SIZEOF_POINTER__", "8"));
                macros.push(("__SIZEOF_SIZE_T__", "8"));
                macros.push(("__SIZEOF_PTRDIFF_T__", "8"));
            }
            DataModel::ILP32 => {
                macros.push(("__SIZEOF_LONG__", "4"));
                macros.push(("__SIZEOF_POINTER__", "4"));
                macros.push(("__SIZEOF_SIZE_T__", "4"));
                macros.push(("__SIZEOF_PTRDIFF_T__", "4"));
            }
        }
        macros.push(("__SIZEOF_INT__", "4"));
        macros.push(("__SIZEOF_SHORT__", "2"));
        macros.push(("__SIZEOF_FLOAT__", "4"));
        macros.push(("__SIZEOF_DOUBLE__", "8"));
        macros.push(("__SIZEOF_LONG_LONG__", "8"));
        macros.push(("__CHAR_BIT__", "8"));

        // ---- Byte order ----
        macros.push(("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__"));
        macros.push(("__ORDER_LITTLE_ENDIAN__", "1234"));
        macros.push(("__ORDER_BIG_ENDIAN__", "4321"));
        macros.push(("__ORDER_PDP_ENDIAN__", "3412"));

        macros
    }

    // -------------------------------------------------------------------
    // Target string parsing
    // -------------------------------------------------------------------

    /// Parses a target architecture from a CLI `--target` flag value.
    ///
    /// Matching is **case-insensitive**.  Accepted aliases:
    ///
    /// | Target     | Accepted strings |
    /// |------------|-----------------|
    /// | x86-64     | `x86-64`, `x86_64`, `amd64` |
    /// | i686       | `i686`, `i386`, `i586`, `x86` |
    /// | AArch64    | `aarch64`, `arm64` |
    /// | RISC-V 64  | `riscv64`, `riscv64gc` |
    ///
    /// Returns `None` if the string does not match any known target.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Target> {
        match s.to_ascii_lowercase().as_str() {
            "x86-64" | "x86_64" | "amd64" => Some(Target::X86_64),
            "i686" | "i386" | "i586" | "x86" => Some(Target::I686),
            "aarch64" | "arm64" => Some(Target::AArch64),
            "riscv64" | "riscv64gc" => Some(Target::RiscV64),
            _ => None,
        }
    }

    /// Detects the host machine's target architecture at **compile time**.
    ///
    /// Uses Rust's `cfg!(target_arch = "...")` to determine the architecture
    /// the BCC compiler itself was compiled for.  Falls back to
    /// [`Target::X86_64`] when the host is not one of the four supported
    /// targets.
    pub fn host_target() -> Target {
        if cfg!(target_arch = "x86_64") {
            Target::X86_64
        } else if cfg!(target_arch = "x86") {
            Target::I686
        } else if cfg!(target_arch = "aarch64") {
            Target::AArch64
        } else if cfg!(target_arch = "riscv64") {
            Target::RiscV64
        } else {
            // Safe fallback — target only affects code generation output,
            // not the compiler's own execution.
            Target::X86_64
        }
    }

    // -------------------------------------------------------------------
    // ELF constants
    // -------------------------------------------------------------------

    /// Returns the ELF `e_machine` header value for this target.
    ///
    /// | Target     | Value | Constant |
    /// |------------|-------|----------|
    /// | x86-64     | 62    | `EM_X86_64` |
    /// | i686       | 3     | `EM_386`    |
    /// | AArch64    | 183   | `EM_AARCH64`|
    /// | RISC-V 64  | 243   | `EM_RISCV`  |
    #[inline]
    pub fn elf_machine(&self) -> u16 {
        match self {
            Target::X86_64 => EM_X86_64,
            Target::I686 => EM_386,
            Target::AArch64 => EM_AARCH64,
            Target::RiscV64 => EM_RISCV,
        }
    }

    /// Returns the ELF class (32-bit or 64-bit) for this target.
    ///
    /// * `ELFCLASS32` (1) — i686
    /// * `ELFCLASS64` (2) — x86-64, AArch64, RISC-V 64
    #[inline]
    pub fn elf_class(&self) -> u8 {
        match self {
            Target::I686 => ELFCLASS32,
            Target::X86_64 | Target::AArch64 | Target::RiscV64 => ELFCLASS64,
        }
    }

    /// Returns architecture-specific ELF flags for the `e_flags` header field.
    ///
    /// Most architectures use zero flags.  For RISC-V the flags encode the ISA
    /// extension set and floating-point ABI:
    ///
    /// * `EF_RISCV_RVC` (0x0001) — compressed instruction extension (C)
    /// * `EF_RISCV_FLOAT_ABI_DOUBLE` (0x0004) — double-precision float ABI
    #[inline]
    pub fn elf_flags(&self) -> u32 {
        match self {
            Target::X86_64 | Target::I686 | Target::AArch64 => 0,
            // RV64IMAFDC with LP64D ABI:
            //   C extension present → EF_RISCV_RVC
            //   double-float ABI    → EF_RISCV_FLOAT_ABI_DOUBLE
            Target::RiscV64 => EF_RISCV_RVC | EF_RISCV_FLOAT_ABI_DOUBLE,
        }
    }

    // -------------------------------------------------------------------
    // Dynamic linker
    // -------------------------------------------------------------------

    /// Returns the filesystem path to the Linux dynamic linker (ELF interpreter)
    /// for this target.
    ///
    /// This path is embedded in the `PT_INTERP` program header of dynamically
    /// linked ELF executables and position-independent executables (PIE).
    ///
    /// | Target     | Path |
    /// |------------|------|
    /// | x86-64     | `/lib64/ld-linux-x86-64.so.2` |
    /// | i686       | `/lib/ld-linux.so.2` |
    /// | AArch64    | `/lib/ld-linux-aarch64.so.1` |
    /// | RISC-V 64  | `/lib/ld-linux-riscv64-lp64d.so.1` |
    #[inline]
    pub fn dynamic_linker_path(&self) -> &'static str {
        match self {
            Target::X86_64 => "/lib64/ld-linux-x86-64.so.2",
            Target::I686 => "/lib/ld-linux.so.2",
            Target::AArch64 => "/lib/ld-linux-aarch64.so.1",
            Target::RiscV64 => "/lib/ld-linux-riscv64-lp64d.so.1",
        }
    }
}

// ---------------------------------------------------------------------------
// Display — human-readable names for diagnostic messages
// ---------------------------------------------------------------------------

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
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- pointer_width / pointer_align -----------------------------------

    #[test]
    fn pointer_width_64bit_targets() {
        assert_eq!(Target::X86_64.pointer_width(), 8);
        assert_eq!(Target::AArch64.pointer_width(), 8);
        assert_eq!(Target::RiscV64.pointer_width(), 8);
    }

    #[test]
    fn pointer_width_32bit_target() {
        assert_eq!(Target::I686.pointer_width(), 4);
    }

    #[test]
    fn pointer_align_matches_width() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(
                t.pointer_align(),
                t.pointer_width(),
                "pointer_align != pointer_width for {}",
                t
            );
        }
    }

    // -- data_model ------------------------------------------------------

    #[test]
    fn data_model_lp64() {
        assert_eq!(Target::X86_64.data_model(), DataModel::LP64);
        assert_eq!(Target::AArch64.data_model(), DataModel::LP64);
        assert_eq!(Target::RiscV64.data_model(), DataModel::LP64);
    }

    #[test]
    fn data_model_ilp32() {
        assert_eq!(Target::I686.data_model(), DataModel::ILP32);
    }

    // -- endianness ------------------------------------------------------

    #[test]
    fn all_targets_little_endian() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(
                t.endianness(),
                Endianness::Little,
                "{} should be little-endian",
                t
            );
        }
    }

    // -- stack_alignment -------------------------------------------------

    #[test]
    fn stack_alignment_16_for_all() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(
                t.stack_alignment(),
                16,
                "{} stack alignment should be 16",
                t
            );
        }
    }

    // -- long_size -------------------------------------------------------

    #[test]
    fn long_size_by_data_model() {
        assert_eq!(Target::X86_64.long_size(), 8);
        assert_eq!(Target::AArch64.long_size(), 8);
        assert_eq!(Target::RiscV64.long_size(), 8);
        assert_eq!(Target::I686.long_size(), 4);
    }

    // -- long_double_size ------------------------------------------------

    #[test]
    fn long_double_size_x86_64() {
        assert_eq!(Target::X86_64.long_double_size(), 16);
    }

    #[test]
    fn long_double_size_i686() {
        assert_eq!(Target::I686.long_double_size(), 12);
    }

    #[test]
    fn long_double_size_aarch64_and_riscv() {
        assert_eq!(Target::AArch64.long_double_size(), 8);
        assert_eq!(Target::RiscV64.long_double_size(), 8);
    }

    // -- max_atomic_width ------------------------------------------------

    #[test]
    fn max_atomic_width_values() {
        assert_eq!(Target::X86_64.max_atomic_width(), 16);
        assert_eq!(Target::I686.max_atomic_width(), 8);
        assert_eq!(Target::AArch64.max_atomic_width(), 16);
        assert_eq!(Target::RiscV64.max_atomic_width(), 8);
    }

    // -- predefined_macros -----------------------------------------------

    #[test]
    fn predefined_macros_common_stdc() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let macros = t.predefined_macros();
            assert!(
                macros.contains(&("__STDC__", "1")),
                "{}: missing __STDC__",
                t
            );
            assert!(
                macros.contains(&("__STDC_VERSION__", "201112L")),
                "{}: missing __STDC_VERSION__",
                t
            );
            assert!(
                macros.contains(&("__STDC_HOSTED__", "1")),
                "{}: missing __STDC_HOSTED__",
                t
            );
        }
    }

    #[test]
    fn predefined_macros_common_platform() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let macros = t.predefined_macros();
            assert!(
                macros.contains(&("__linux__", "1")),
                "{}: missing __linux__",
                t
            );
            assert!(
                macros.contains(&("__gnu_linux__", "1")),
                "{}: missing __gnu_linux__",
                t
            );
            assert!(macros.contains(&("__ELF__", "1")), "{}: missing __ELF__", t);
            assert!(
                macros.contains(&("__unix__", "1")),
                "{}: missing __unix__",
                t
            );
        }
    }

    #[test]
    fn predefined_macros_x86_64_arch() {
        let macros = Target::X86_64.predefined_macros();
        assert!(macros.contains(&("__x86_64__", "1")));
        assert!(macros.contains(&("__amd64__", "1")));
        assert!(macros.contains(&("__LP64__", "1")));
    }

    #[test]
    fn predefined_macros_i686_arch() {
        let macros = Target::I686.predefined_macros();
        assert!(macros.contains(&("__i386__", "1")));
        assert!(macros.contains(&("__i686__", "1")));
        assert!(macros.contains(&("__ILP32__", "1")));
    }

    #[test]
    fn predefined_macros_aarch64_arch() {
        let macros = Target::AArch64.predefined_macros();
        assert!(macros.contains(&("__aarch64__", "1")));
        assert!(macros.contains(&("__LP64__", "1")));
        assert!(macros.contains(&("__ARM_64BIT_STATE", "1")));
    }

    #[test]
    fn predefined_macros_riscv64_arch() {
        let macros = Target::RiscV64.predefined_macros();
        assert!(macros.contains(&("__riscv", "1")));
        assert!(macros.contains(&("__riscv_xlen", "64")));
        assert!(macros.contains(&("__LP64__", "1")));
    }

    #[test]
    fn predefined_macros_sizeof_pointer() {
        // LP64 targets should report 8-byte pointers.
        let m64 = Target::X86_64.predefined_macros();
        assert!(m64.contains(&("__SIZEOF_POINTER__", "8")));

        // ILP32 targets should report 4-byte pointers.
        let m32 = Target::I686.predefined_macros();
        assert!(m32.contains(&("__SIZEOF_POINTER__", "4")));
    }

    #[test]
    fn predefined_macros_compiler_id() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let macros = t.predefined_macros();
            assert!(macros.contains(&("__BCC__", "1")), "{}: missing __BCC__", t);
        }
    }

    // -- from_str --------------------------------------------------------

    #[test]
    fn from_str_x86_64_aliases() {
        assert_eq!(Target::from_str("x86-64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("x86_64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("amd64"), Some(Target::X86_64));
        assert_eq!(Target::from_str("X86_64"), Some(Target::X86_64)); // case-insensitive
        assert_eq!(Target::from_str("AMD64"), Some(Target::X86_64));
    }

    #[test]
    fn from_str_i686_aliases() {
        assert_eq!(Target::from_str("i686"), Some(Target::I686));
        assert_eq!(Target::from_str("i386"), Some(Target::I686));
        assert_eq!(Target::from_str("i586"), Some(Target::I686));
        assert_eq!(Target::from_str("x86"), Some(Target::I686));
    }

    #[test]
    fn from_str_aarch64_aliases() {
        assert_eq!(Target::from_str("aarch64"), Some(Target::AArch64));
        assert_eq!(Target::from_str("arm64"), Some(Target::AArch64));
        assert_eq!(Target::from_str("AARCH64"), Some(Target::AArch64));
    }

    #[test]
    fn from_str_riscv64_aliases() {
        assert_eq!(Target::from_str("riscv64"), Some(Target::RiscV64));
        assert_eq!(Target::from_str("riscv64gc"), Some(Target::RiscV64));
    }

    #[test]
    fn from_str_invalid() {
        assert_eq!(Target::from_str("mips"), None);
        assert_eq!(Target::from_str(""), None);
        assert_eq!(Target::from_str("powerpc64"), None);
        assert_eq!(Target::from_str("sparc"), None);
    }

    // -- host_target -----------------------------------------------------

    #[test]
    fn host_target_is_valid() {
        let host = Target::host_target();
        // Must be one of the four supported targets (or the x86-64 fallback).
        assert!(
            host == Target::X86_64
                || host == Target::I686
                || host == Target::AArch64
                || host == Target::RiscV64,
            "host_target() returned unexpected value: {:?}",
            host,
        );
    }

    // -- elf_machine -----------------------------------------------------

    #[test]
    fn elf_machine_values() {
        assert_eq!(Target::X86_64.elf_machine(), 62);
        assert_eq!(Target::I686.elf_machine(), 3);
        assert_eq!(Target::AArch64.elf_machine(), 183);
        assert_eq!(Target::RiscV64.elf_machine(), 243);
    }

    // -- elf_class -------------------------------------------------------

    #[test]
    fn elf_class_64bit() {
        assert_eq!(Target::X86_64.elf_class(), ELFCLASS64);
        assert_eq!(Target::AArch64.elf_class(), ELFCLASS64);
        assert_eq!(Target::RiscV64.elf_class(), ELFCLASS64);
    }

    #[test]
    fn elf_class_32bit() {
        assert_eq!(Target::I686.elf_class(), ELFCLASS32);
    }

    // -- elf_flags -------------------------------------------------------

    #[test]
    fn elf_flags_zero_for_non_riscv() {
        assert_eq!(Target::X86_64.elf_flags(), 0);
        assert_eq!(Target::I686.elf_flags(), 0);
        assert_eq!(Target::AArch64.elf_flags(), 0);
    }

    #[test]
    fn elf_flags_riscv64() {
        let flags = Target::RiscV64.elf_flags();
        // EF_RISCV_RVC (0x0001) | EF_RISCV_FLOAT_ABI_DOUBLE (0x0004)
        assert_eq!(flags, 0x0005);
        assert_ne!(flags & EF_RISCV_RVC, 0, "RVC flag not set");
        assert_ne!(
            flags & EF_RISCV_FLOAT_ABI_DOUBLE,
            0,
            "FLOAT_ABI_DOUBLE flag not set"
        );
    }

    // -- dynamic_linker_path ---------------------------------------------

    #[test]
    fn dynamic_linker_paths() {
        assert_eq!(
            Target::X86_64.dynamic_linker_path(),
            "/lib64/ld-linux-x86-64.so.2"
        );
        assert_eq!(Target::I686.dynamic_linker_path(), "/lib/ld-linux.so.2");
        assert_eq!(
            Target::AArch64.dynamic_linker_path(),
            "/lib/ld-linux-aarch64.so.1"
        );
        assert_eq!(
            Target::RiscV64.dynamic_linker_path(),
            "/lib/ld-linux-riscv64-lp64d.so.1"
        );
    }

    // -- Display implementations -----------------------------------------

    #[test]
    fn display_target() {
        assert_eq!(format!("{}", Target::X86_64), "x86-64");
        assert_eq!(format!("{}", Target::I686), "i686");
        assert_eq!(format!("{}", Target::AArch64), "aarch64");
        assert_eq!(format!("{}", Target::RiscV64), "riscv64");
    }

    #[test]
    fn display_data_model() {
        assert_eq!(format!("{}", DataModel::LP64), "LP64");
        assert_eq!(format!("{}", DataModel::ILP32), "ILP32");
    }

    #[test]
    fn display_endianness() {
        assert_eq!(format!("{}", Endianness::Little), "little-endian");
        assert_eq!(format!("{}", Endianness::Big), "big-endian");
    }

    // -- Derive-trait smoke tests ----------------------------------------

    #[test]
    fn target_copy_clone_eq_hash() {
        let a = Target::X86_64;
        let b = a; // Copy
        let c = b; // Copy again — validates both Copy and Clone (Clone is auto-derived for Copy types)
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(Target::X86_64, Target::I686);

        // Hash — verify different targets produce different hashes
        // (not strictly required by Hash contract, but expected).
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Target::X86_64);
        set.insert(Target::I686);
        set.insert(Target::AArch64);
        set.insert(Target::RiscV64);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn data_model_copy_clone_eq_hash() {
        let a = DataModel::LP64;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(DataModel::LP64, DataModel::ILP32);
    }

    #[test]
    fn endianness_copy_clone_eq_hash() {
        let a = Endianness::Little;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(Endianness::Little, Endianness::Big);
    }

    // -- Round-trip: Display → from_str ----------------------------------

    #[test]
    fn display_roundtrip() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let s = format!("{}", t);
            let parsed = Target::from_str(&s);
            assert_eq!(
                parsed,
                Some(*t),
                "from_str(Display({})) should round-trip",
                t
            );
        }
    }
}
