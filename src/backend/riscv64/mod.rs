//! RISC-V 64-bit backend module.
//!
//! Implements the `ArchCodegen` trait for the RISC-V 64 (RV64IMAFDC) target
//! architecture, providing instruction selection, register allocation
//! configuration, and ABI-conformant code generation following the LP64D ABI.
//!
//! # Sub-modules
//!
//! - [`linker`]: Built-in RISC-V 64 linker — relocation application with
//!   linker relaxation support, ELF linking for ET_EXEC and ET_DYN output.
//!
//! # Architecture Characteristics
//!
//! - 32 integer registers (x0–x31) + 32 floating-point registers (f0–f31)
//! - Fixed 32-bit base instruction width with 16-bit compressed extensions (RVC)
//! - RISC-V LP64D calling convention: a0–a7 for integer arguments,
//!   fa0–fa7 for floating-point arguments
//! - LUI/AUIPC for large immediate materialization
//! - RV64IMAFDC ISA (Integer, Multiply, Atomic, Float, Double, Compressed)

/// Built-in RISC-V 64 linker — relocation application with relaxation support,
/// producing ELF executables and shared objects for the RISC-V 64 target.
pub mod linker;
