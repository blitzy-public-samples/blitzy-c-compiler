//! # AArch64 Built-in ELF Linker
//!
//! Built-in ELF linker for the AArch64 architecture, producing ET_EXEC (static
//! executables) and ET_DYN (shared objects). This is part of BCC's standalone
//! backend — NO external linker is invoked.
//!
//! ## Entry Points
//! - [`AArch64Linker::link_executable()`] — Produce a static ELF executable (ET_EXEC)
//! - [`AArch64Linker::link_shared_library()`] — Produce a shared object (ET_DYN)
//! - [`link_aarch64()`] — Convenience function dispatching based on [`OutputType`]
//!
//! ## Architecture Details
//! - ELF machine type: EM_AARCH64 (183)
//! - ELF class: ELFCLASS64 (always 64-bit)
//! - Data encoding: ELFDATA2LSB (little-endian)
//! - Default base address: 0x400000 for ET_EXEC, 0x0 for ET_DYN
//! - Page size: 4096 bytes (4 KiB)
//!
//! ## PLT/GOT for PIC and Shared Libraries
//! AArch64 PLT stubs use the ADRP+LDR+BR instruction sequence:
//! ```text
//! PLT0 (32 bytes):
//!   stp x16, x30, [sp, #-16]!   // save IP0 and LR
//!   adrp x16, GOT+16            // load page of GOT[2]
//!   ldr  x17, [x16, :lo12:GOT+16] // load _dl_runtime_resolve
//!   add  x16, x16, :lo12:GOT+16   // compute full GOT address
//!   br   x17                     // jump to resolver
//!   nop                          // padding (align to 32 bytes)
//!   nop
//!   nop
//!
//! PLTn (16 bytes each):
//!   adrp x16, GOT[n]            // load page of GOT entry
//!   ldr  x17, [x16, :lo12:GOT[n]] // load function address from GOT
//!   add  x16, x16, :lo12:GOT[n]   // compute full GOT address (for lazy resolution)
//!   br   x17                     // jump to function (or back to PLT0 for lazy)
//! ```
//!
//! ## Relocation Handling
//! Architecture-specific relocation application is delegated to the sibling
//! [`relocations`] module, which implements the `RelocationHandler` trait from
//! `crate::backend::linker_common::relocation`.
//!
//! ## Dynamic Linking Sections
//! When producing shared libraries (-shared), generates:
//! - `.dynamic` — Dynamic section with DT_* entries
//! - `.dynsym` / `.dynstr` — Dynamic symbol table and string table
//! - `.gnu.hash` — GNU hash table for fast symbol lookup
//! - `.rela.dyn` — Dynamic relocations (R_AARCH64_RELATIVE, R_AARCH64_GLOB_DAT)
//! - `.rela.plt` — PLT relocations (R_AARCH64_JUMP_SLOT)
//! - `.got` / `.got.plt` — Global Offset Table
//! - `.plt` — Procedure Linkage Table stubs

// ============================================================================
// Submodule declaration
// ============================================================================

/// AArch64-specific relocation handler implementing the `RelocationHandler`
/// trait for all AArch64 ELF relocation types.
pub mod relocations;

// ============================================================================
// Re-exports
// ============================================================================

/// Re-export the AArch64 relocation handler for convenient access.
pub use self::relocations::AArch64RelocationHandler;

// ============================================================================
// Imports — linker_common shared infrastructure
// ============================================================================

use crate::backend::linker_common::section_merger::OutputSection;
use crate::backend::linker_common::symbol_resolver::ResolvedSymbol;
use crate::backend::linker_common::{LinkerConfig, LinkerInput, LinkerOutput, OutputType};

// ============================================================================
// Imports — common infrastructure
// ============================================================================

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ============================================================================
// AArch64-Specific ELF Constants
// ============================================================================

/// ELF machine type for AArch64.
pub const EM_AARCH64: u16 = 183;

/// ELF class: always 64-bit for AArch64.
pub const ELFCLASS64: u8 = 2;

/// ELF data encoding: little-endian.
pub const ELFDATA2LSB: u8 = 1;

/// Default base address for AArch64 static executables.
///
/// Shared libraries use a base of 0x0 (position-independent).
pub const DEFAULT_BASE_ADDRESS: u64 = 0x400000;

/// Page size for AArch64 (4 KiB).
///
/// Used for ADRP page-relative addressing and segment alignment.
pub const PAGE_SIZE: u64 = 4096;

/// PLT header (PLT0) size in bytes.
///
/// PLT0 contains 8 AArch64 instructions (8 × 4 = 32 bytes):
/// `stp`, `adrp`, `ldr`, `add`, `br`, `nop`, `nop`, `nop`.
pub const PLT0_SIZE: usize = 32;

/// PLT entry (PLTn) size in bytes.
///
/// Each PLTn stub contains 4 AArch64 instructions (4 × 4 = 16 bytes):
/// `adrp`, `ldr`, `add`, `br`.
pub const PLTN_SIZE: usize = 16;

/// GOT entry size in bytes (64-bit pointer).
pub const GOT_ENTRY_SIZE: usize = 8;

/// Number of reserved GOT.PLT entries.
///
/// The first three GOT.PLT entries are reserved:
/// - GOT.PLT[0]: Address of `.dynamic` section (filled by dynamic linker)
/// - GOT.PLT[1]: `link_map` pointer (filled by dynamic linker)
/// - GOT.PLT[2]: `_dl_runtime_resolve` (filled by dynamic linker)
pub const GOT_PLT_RESERVED_ENTRIES: usize = 3;

/// Dynamic linker path for AArch64 Linux.
pub const DYNAMIC_LINKER_PATH: &str = "/lib/ld-linux-aarch64.so.1";

// ============================================================================
// AArch64Linker — Primary Linker Driver
// ============================================================================

/// AArch64 ELF linker driver.
///
/// Orchestrates the complete linking process for AArch64 targets:
/// 1. Symbol resolution (via `linker_common::symbol_resolver`)
/// 2. Section merging (via `linker_common::section_merger`)
/// 3. Address layout (via `linker_common::linker_script`)
/// 4. Relocation processing (via `self::relocations` + `linker_common::relocation`)
/// 5. Dynamic section generation (via `linker_common::dynamic`) — if shared library
/// 6. ELF output writing (via `elf_writer_common`)
///
/// # Examples
///
/// ```ignore
/// use bcc::backend::aarch64::linker::AArch64Linker;
/// use bcc::backend::linker_common::{LinkerConfig, OutputType};
/// use bcc::common::target::Target;
/// use bcc::common::diagnostics::DiagnosticEngine;
///
/// let config = LinkerConfig::new(Target::AArch64, OutputType::Executable);
/// let mut linker = AArch64Linker::new(config);
/// let mut diag = DiagnosticEngine::new();
/// let result = linker.link_executable(vec![], &mut diag);
/// ```
pub struct AArch64Linker {
    /// Linker configuration (output type, paths, flags).
    config: LinkerConfig,
    /// AArch64-specific relocation handler implementing `RelocationHandler`.
    reloc_handler: AArch64RelocationHandler,
    /// Accumulated diagnostic messages for AArch64-specific issues.
    diagnostics: Vec<String>,
}

impl AArch64Linker {
    /// Create a new AArch64 linker with the given configuration.
    ///
    /// The relocation handler is initialized based on the PIC flag in the
    /// configuration — `with_pic(true)` for shared libraries and PIC
    /// executables, `with_pic(false)` otherwise.
    ///
    /// # Arguments
    ///
    /// * `config` — Linker configuration specifying target, output type,
    ///   paths, library search directories, and linking flags.
    pub fn new(config: LinkerConfig) -> Self {
        // Determine if PIC mode is active (shared libraries always require PIC).
        let pic_mode = config.pic || config.output_type == OutputType::SharedLibrary;
        let reloc_handler = if pic_mode {
            AArch64RelocationHandler::with_pic(true)
        } else {
            AArch64RelocationHandler::new()
        };

        AArch64Linker {
            config,
            reloc_handler,
            diagnostics: Vec::new(),
        }
    }

    /// Link AArch64 object files into a static ELF executable (ET_EXEC).
    ///
    /// Performs the complete linking pipeline:
    /// 1. **Symbol Resolution** — Collects and resolves all symbols from input
    ///    objects. Undefined non-weak symbols are fatal errors.
    /// 2. **Section Merging** — Merges input sections into output sections
    ///    (`.text`, `.rodata`, `.data`, `.bss`) with proper alignment.
    /// 3. **Address Layout** — Computes virtual addresses starting from
    ///    `DEFAULT_BASE_ADDRESS` (0x400000) with 4 KiB page alignment.
    /// 4. **Relocation Processing** — Resolves and applies all AArch64
    ///    relocations using the `AArch64RelocationHandler`.
    /// 5. **Linker Symbols** — Defines `_start`, `__bss_start`, `_edata`,
    ///    `_end`, `__executable_start`.
    /// 6. **ELF Construction** — Builds the final ELF binary with:
    ///    - `e_machine` = `EM_AARCH64` (183), `e_type` = `ET_EXEC` (2)
    ///    - Program headers: `PT_PHDR`, `PT_LOAD` (R+X, R, R+W), `PT_GNU_STACK`
    ///    - Symbol table (`.symtab`) and string table (`.strtab`)
    ///
    /// # Arguments
    ///
    /// * `inputs` — Parsed input object files (`.o`) to link.
    /// * `diagnostics` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// `Ok(LinkerOutput)` containing the complete ELF binary on success,
    /// or `Err(String)` with a descriptive error message on failure.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Undefined symbols are present (and not weak)
    /// - Relocation overflow occurs (e.g., `R_AARCH64_CALL26` exceeds ±128 MiB)
    /// - The `_start` entry point symbol is undefined
    /// - Section merging encounters incompatible attributes
    pub fn link_executable(
        &mut self,
        inputs: Vec<LinkerInput>,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<LinkerOutput, String> {
        // Validate target architecture.
        self.validate_target(diagnostics)?;

        // Ensure the config is set up for executable linking.
        let mut exec_config = self.config.clone();
        exec_config.output_type = OutputType::Executable;
        // Preserve allow_undefined from the caller — generation.rs sets it
        // to true for dynamically linked executables that reference libc.

        // Delegate to the shared linker infrastructure which handles:
        // - Symbol resolution with strong/weak binding
        // - Section merging with COMDAT deduplication
        // - Address layout via DefaultLinkerScript (base = 0x400000)
        // - Relocation processing via our AArch64RelocationHandler
        // - Dynamic section generation (skipped for executables)
        // - ELF output writing with proper program headers
        let result = crate::backend::linker_common::link(
            &exec_config,
            inputs,
            &self.reloc_handler,
            diagnostics,
        );

        // Capture any AArch64-specific diagnostics.
        if diagnostics.has_errors() {
            self.record_diagnostic(format!(
                "AArch64 executable linking completed with {} error(s)",
                diagnostics.error_count()
            ));
        }

        result
    }

    /// Link AArch64 object files into a shared library (ET_DYN).
    ///
    /// Performs the complete shared library linking pipeline:
    /// 1. **Symbol Resolution** — Collects and resolves symbols. Undefined
    ///    symbols are permitted (resolved at runtime by the dynamic linker).
    /// 2. **Section Merging** — Same as executable.
    /// 3. **GOT/PLT Generation** — Scans relocations for GOT needs
    ///    (`R_AARCH64_ADR_GOT_PAGE`, `R_AARCH64_LD64_GOT_LO12_NC`) and
    ///    PLT needs (function calls to external symbols). Generates:
    ///    - GOT entries (8 bytes each)
    ///    - GOT.PLT entries (3 reserved + one per PLT stub)
    ///    - PLT stubs (PLT0 = 32 bytes, PLTn = 16 bytes each, ADRP+LDR+BR)
    /// 4. **Dynamic Sections** — Builds `.dynsym`, `.dynstr`, `.gnu.hash`,
    ///    `.rela.dyn`, `.rela.plt`, `.dynamic`, `.interp`, `.got`, `.got.plt`,
    ///    `.plt`.
    /// 5. **Address Layout** — Base address = 0 (PIC).
    /// 6. **Relocation Processing** — Applies static relocations, emits
    ///    dynamic relocations (`R_AARCH64_RELATIVE`, `R_AARCH64_GLOB_DAT`,
    ///    `R_AARCH64_JUMP_SLOT`).
    /// 7. **ELF Construction** — Builds ELF with `e_type` = `ET_DYN` (3) and
    ///    program headers: `PT_PHDR`, `PT_INTERP`, `PT_LOAD`, `PT_DYNAMIC`,
    ///    `PT_GNU_STACK`, `PT_GNU_RELRO`.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Parsed input object files (`.o`) to link.
    /// * `diagnostics` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// `Ok(LinkerOutput)` containing the complete shared library ELF on
    /// success, or `Err(String)` with a descriptive error message.
    pub fn link_shared_library(
        &mut self,
        inputs: Vec<LinkerInput>,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<LinkerOutput, String> {
        // Validate target architecture.
        self.validate_target(diagnostics)?;

        // Ensure the config is set up for shared library linking.
        let mut shared_config = self.config.clone();
        shared_config.output_type = OutputType::SharedLibrary;
        shared_config.allow_undefined = true;
        shared_config.pic = true;

        // Delegate to the shared linker infrastructure which handles:
        // - Symbol resolution with undefined symbols allowed
        // - Section merging
        // - Address layout via DefaultLinkerScript (base = 0x0 for PIC)
        // - Relocation processing with PIC-aware handler
        // - Dynamic section generation (builds .dynamic, .dynsym, .got, .plt, etc.)
        // - ELF output writing with PT_INTERP, PT_DYNAMIC, PT_GNU_RELRO
        let result = crate::backend::linker_common::link(
            &shared_config,
            inputs,
            &self.reloc_handler,
            diagnostics,
        );

        // Capture any AArch64-specific diagnostics.
        if diagnostics.has_errors() {
            self.record_diagnostic(format!(
                "AArch64 shared library linking completed with {} error(s)",
                diagnostics.error_count()
            ));
        }

        result
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Validate that the target architecture is AArch64.
    ///
    /// Returns an error if the linker configuration targets a different
    /// architecture, preventing accidental misuse.
    fn validate_target(&self, diagnostics: &mut DiagnosticEngine) -> Result<(), String> {
        if self.config.target != Target::AArch64 {
            let msg = format!(
                "AArch64 linker invoked for wrong target: expected AArch64, got {:?}",
                self.config.target
            );
            diagnostics.emit(Diagnostic::error(
                crate::common::diagnostics::Span::dummy(),
                msg.clone(),
            ));
            return Err(msg);
        }
        Ok(())
    }

    /// Record an AArch64-specific diagnostic message.
    fn record_diagnostic(&mut self, message: String) {
        self.diagnostics.push(message);
    }

    /// Return the number of accumulated AArch64-specific diagnostic messages.
    #[allow(dead_code)]
    fn diagnostic_count(&self) -> usize {
        self.diagnostics.len()
    }

    /// Return a reference to the underlying relocation handler.
    #[allow(dead_code)]
    fn relocation_handler(&self) -> &AArch64RelocationHandler {
        &self.reloc_handler
    }

    /// Return a reference to the linker configuration.
    #[allow(dead_code)]
    fn config(&self) -> &LinkerConfig {
        &self.config
    }
}

// ============================================================================
// PLT Stub Generation Helpers
// ============================================================================

/// Compute the ADRP page delta between a target address and a source address.
///
/// ADRP uses 4 KiB (12-bit) page granularity. The returned value is the
/// signed page number difference: `(target >> 12) - (source >> 12)`.
///
/// # Arguments
///
/// * `target` — Target address (e.g., GOT entry address).
/// * `source` — Source address (e.g., PLT instruction address).
///
/// # Returns
///
/// Signed page delta suitable for encoding into an ADRP immediate field.
#[inline]
fn compute_adrp_page_delta(target: u64, source: u64) -> i64 {
    let target_page = (target >> 12) as i64;
    let source_page = (source >> 12) as i64;
    target_page - source_page
}

/// Extract the low 12 bits of an address (page offset).
///
/// Used for the `:lo12:` relocation modifier in ADRP+ADD/LDR pairs.
#[inline]
fn page_offset(addr: u64) -> u32 {
    (addr & 0xFFF) as u32
}

/// Compute the 4 KiB-aligned page address (clear low 12 bits).
#[inline]
#[allow(dead_code)]
fn page_align(addr: u64) -> u64 {
    addr & !0xFFF
}

/// Generate the PLT0 header stub for AArch64 (32 bytes).
///
/// PLT0 is the resolver entry point called by PLTn stubs when a function
/// has not yet been lazily resolved. It saves registers and jumps to the
/// dynamic linker's `_dl_runtime_resolve` function.
///
/// # Layout (8 instructions × 4 bytes = 32 bytes)
///
/// ```text
/// stp x16, x30, [sp, #-16]!     // Save IP0 and LR
/// adrp x16, PAGE(GOT+16)        // Page of GOT[2] (_dl_runtime_resolve)
/// ldr  x17, [x16, #LO12(GOT+16)] // Load resolver address from GOT[2]
/// add  x16, x16, #LO12(GOT+8)   // Compute full address of GOT[1] (link_map)
/// br   x17                       // Jump to resolver
/// nop                            // Padding to 32 bytes
/// nop
/// nop
/// ```
///
/// # Arguments
///
/// * `got_plt_addr` — Virtual address of the `.got.plt` section start.
/// * `plt_addr` — Virtual address of the `.plt` section start (PLT0 address).
///
/// # Returns
///
/// Exactly 32 bytes of machine code for the PLT0 stub.
pub fn generate_plt0_stub(got_plt_addr: u64, plt_addr: u64) -> Vec<u8> {
    let mut code = Vec::with_capacity(PLT0_SIZE);

    // Instruction 0: stp x16, x30, [sp, #-16]!
    // Save IP0 (x16) and LR (x30) to the stack.
    code.extend_from_slice(&0xA9BF_7BF0u32.to_le_bytes());

    // Instruction 1: adrp x16, PAGE(GOT.PLT[2])
    // GOT.PLT[2] = got_plt_addr + 16 (third reserved entry).
    let got2_addr = got_plt_addr + 16;
    // ADRP is PC-relative from this instruction's address (plt_addr + 4).
    let page_delta = compute_adrp_page_delta(got2_addr, plt_addr + 4);
    code.extend_from_slice(&encode_adrp(16, page_delta as i32).to_le_bytes());

    // Instruction 2: ldr x17, [x16, #LO12(GOT.PLT[2])]
    // Load the _dl_runtime_resolve address from GOT[2].
    let got2_lo12 = page_offset(got2_addr);
    code.extend_from_slice(&encode_ldr64_uimm(17, 16, got2_lo12 / 8).to_le_bytes());

    // Instruction 3: add x16, x16, #LO12(GOT.PLT[1])
    // Compute full address of GOT[1] (link_map) for the resolver.
    let got1_lo12 = page_offset(got_plt_addr + 8);
    code.extend_from_slice(&encode_add_imm64(16, 16, got1_lo12).to_le_bytes());

    // Instruction 4: br x17
    // Jump to _dl_runtime_resolve.
    code.extend_from_slice(&0xD61F_0220u32.to_le_bytes());

    // Instructions 5-7: nop × 3 (padding to 32 bytes).
    for _ in 0..3 {
        code.extend_from_slice(&0xD503_201Fu32.to_le_bytes());
    }

    debug_assert_eq!(
        code.len(),
        PLT0_SIZE,
        "PLT0 stub must be exactly {} bytes",
        PLT0_SIZE
    );
    code
}

/// Generate a PLTn entry stub for AArch64 (16 bytes).
///
/// Each PLTn stub loads the target function's address from its GOT entry
/// and branches to it. On first call (lazy binding), the GOT entry points
/// back to PLT0, which invokes the dynamic linker to resolve the symbol.
///
/// # Layout (4 instructions × 4 bytes = 16 bytes)
///
/// ```text
/// adrp x16, PAGE(GOT[n])            // Page of this function's GOT entry
/// ldr  x17, [x16, #LO12(GOT[n])]   // Load function address from GOT
/// add  x16, x16, #LO12(GOT[n])     // Compute full GOT address (for DT_PLTGOT)
/// br   x17                          // Jump to function or back to resolver
/// ```
///
/// # Arguments
///
/// * `got_entry_addr` — Virtual address of this function's GOT entry.
/// * `plt_entry_addr` — Virtual address of this PLT entry.
///
/// # Returns
///
/// Exactly 16 bytes of machine code for the PLTn stub.
pub fn generate_pltn_stub(got_entry_addr: u64, plt_entry_addr: u64) -> Vec<u8> {
    let mut code = Vec::with_capacity(PLTN_SIZE);

    // Instruction 0: adrp x16, PAGE(GOT[n])
    let page_delta = compute_adrp_page_delta(got_entry_addr, plt_entry_addr);
    code.extend_from_slice(&encode_adrp(16, page_delta as i32).to_le_bytes());

    // Instruction 1: ldr x17, [x16, #LO12(GOT[n])]
    let lo12 = page_offset(got_entry_addr);
    code.extend_from_slice(&encode_ldr64_uimm(17, 16, lo12 / 8).to_le_bytes());

    // Instruction 2: add x16, x16, #LO12(GOT[n])
    code.extend_from_slice(&encode_add_imm64(16, 16, lo12).to_le_bytes());

    // Instruction 3: br x17
    code.extend_from_slice(&0xD61F_0220u32.to_le_bytes());

    debug_assert_eq!(
        code.len(),
        PLTN_SIZE,
        "PLTn stub must be exactly {} bytes",
        PLTN_SIZE
    );
    code
}

// ============================================================================
// AArch64 Instruction Encoding Helpers
// ============================================================================

/// Encode an AArch64 ADRP instruction.
///
/// `ADRP Xd, label` — Forms a PC-relative address to a 4 KiB page.
/// The immediate is a signed 21-bit page offset (±4 GiB range).
///
/// Encoding (32 bits):
/// ```text
/// [31]     = 1       (op = 1 for ADRP)
/// [30:29]  = immlo   (low 2 bits of page offset)
/// [28:24]  = 10000   (opcode)
/// [23:5]   = immhi   (high 19 bits of page offset)
/// [4:0]    = Rd      (destination register)
/// ```
#[inline]
fn encode_adrp(rd: u32, imm21: i32) -> u32 {
    let bits = imm21 as u32 & 0x1F_FFFF;
    let immlo = bits & 0x3;
    let immhi = (bits >> 2) & 0x7_FFFF;
    (1u32 << 31) | (immlo << 29) | (0b10000 << 24) | (immhi << 5) | (rd & 0x1F)
}

/// Encode an AArch64 LDR (64-bit, unsigned immediate offset) instruction.
///
/// `LDR Xt, [Xn, #pimm]` — Load a 64-bit value from `[Xn + pimm*8]`.
///
/// Encoding: `0xF9400000 | (pimm12 << 10) | (Rn << 5) | Rt`
/// where `pimm12 = offset / 8` (scaled by 8 for 64-bit loads).
#[inline]
fn encode_ldr64_uimm(rt: u32, rn: u32, pimm: u32) -> u32 {
    0xF940_0000u32 | ((pimm & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)
}

/// Encode an AArch64 ADD (64-bit, immediate) instruction.
///
/// `ADD Xd, Xn, #imm12` — Add a 12-bit immediate to Xn.
///
/// Encoding: `0x91000000 | (imm12 << 10) | (Rn << 5) | Rd`
#[inline]
fn encode_add_imm64(rd: u32, rn: u32, imm12: u32) -> u32 {
    0x9100_0000u32 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Build concatenated section data from an output section's fragments.
///
/// Iterates over all fragments in the output section, writing each fragment's
/// data at its assigned offset within the output buffer. Gaps between
/// fragments (due to alignment) are zero-filled.
///
/// # Arguments
///
/// * `section` — The merged output section containing fragments.
///
/// # Returns
///
/// A byte vector containing the complete section data with proper alignment.
#[allow(dead_code)]
fn build_output_section_data(section: &OutputSection) -> Vec<u8> {
    if section.total_size == 0 {
        return Vec::new();
    }

    let mut data = vec![0u8; section.total_size as usize];

    for fragment in &section.fragments {
        let start = fragment.offset_in_output as usize;
        let end = start + fragment.size as usize;
        if end <= data.len() && fragment.data.len() >= fragment.size as usize {
            data[start..end].copy_from_slice(&fragment.data[..fragment.size as usize]);
        } else if !fragment.data.is_empty() {
            // Partial copy for fragments where data is shorter than size
            // (e.g., BSS-like fragments with partial initialization).
            let copy_len = fragment.data.len().min(data.len().saturating_sub(start));
            if copy_len > 0 {
                data[start..start + copy_len].copy_from_slice(&fragment.data[..copy_len]);
            }
        }
    }

    data
}

/// Resolve the entry point symbol address from a resolved symbol map.
///
/// Looks up the `_start` symbol (or a custom entry point name) in the
/// resolved symbol table and returns its final virtual address.
///
/// # Arguments
///
/// * `entry_name` — Name of the entry point symbol (typically `"_start"`).
/// * `symbols` — Map of symbol names to their resolved addresses.
///
/// # Returns
///
/// `Ok(address)` if the symbol is found, or `Err(message)` if undefined.
#[allow(dead_code)]
fn resolve_entry_point_address(
    entry_name: &str,
    symbols: &FxHashMap<String, ResolvedSymbol>,
) -> Result<u64, String> {
    match symbols.get(entry_name) {
        Some(sym) if sym.is_defined => Ok(sym.final_address),
        Some(_) => Err(format!(
            "entry point symbol '{}' is defined but has no address (undefined reference)",
            entry_name
        )),
        None => Err(format!(
            "undefined reference to entry point symbol '{}'; \
             cannot produce AArch64 executable without an entry point",
            entry_name
        )),
    }
}

/// Classify a symbol as exported, imported, or local for dynamic linking.
///
/// Used during shared library linking to populate `.dynsym`:
/// - **Exported**: Defined, global/weak, and visible (default/protected).
/// - **Imported**: Undefined, global/weak — resolved at load time.
/// - **Local**: Not exported to the dynamic symbol table.
///
/// # Arguments
///
/// * `sym` — A resolved symbol from the symbol table.
///
/// # Returns
///
/// `Some(true)` if the symbol should be exported, `Some(false)` if it should
/// be imported, `None` if it is local and not relevant to dynamic linking.
#[allow(dead_code)]
fn classify_dynamic_symbol(sym: &ResolvedSymbol) -> Option<bool> {
    let is_global_or_weak = sym.binding
        == crate::backend::linker_common::symbol_resolver::STB_GLOBAL
        || sym.binding == crate::backend::linker_common::symbol_resolver::STB_WEAK;

    if !is_global_or_weak || sym.name.is_empty() {
        return None; // Local symbol — not in .dynsym.
    }

    if sym.is_defined {
        Some(true) // Exported symbol.
    } else {
        Some(false) // Imported symbol.
    }
}

/// Validate that an AArch64 relocation value fits within the instruction's
/// immediate field range.
///
/// # Arguments
///
/// * `reloc_name` — Human-readable relocation type name for error messages.
/// * `value` — The computed relocation value to check.
/// * `bit_width` — Number of bits available in the immediate field.
/// * `signed` — Whether the immediate field is signed.
///
/// # Returns
///
/// `Ok(())` if the value fits, `Err(message)` with a descriptive overflow error.
#[allow(dead_code)]
fn check_relocation_range(
    reloc_name: &str,
    value: i64,
    bit_width: u32,
    signed: bool,
) -> Result<(), String> {
    if signed {
        let max = (1i64 << (bit_width - 1)) - 1;
        let min = -(1i64 << (bit_width - 1));
        if value < min || value > max {
            return Err(format!(
                "relocation {} out of range: value 0x{:x} exceeds ±{} (signed {}-bit)",
                reloc_name, value as u64, max, bit_width
            ));
        }
    } else {
        let max = (1u64 << bit_width) - 1;
        if (value as u64) > max {
            return Err(format!(
                "relocation {} out of range: value 0x{:x} exceeds {} (unsigned {}-bit)",
                reloc_name, value as u64, max, bit_width
            ));
        }
    }
    Ok(())
}

/// Format an AArch64-specific linker error message for diagnostic reporting.
///
/// Produces consistent, informative error messages that include the
/// relocation type, symbol name, and computed value for debugging.
#[allow(dead_code)]
fn format_linker_error(error_type: &str, symbol: &str, detail: &str) -> String {
    if symbol.is_empty() {
        format!("AArch64 linker: {}: {}", error_type, detail)
    } else {
        format!(
            "AArch64 linker: {} for '{}': {}",
            error_type, symbol, detail
        )
    }
}

// ============================================================================
// Public Convenience API
// ============================================================================

/// Link AArch64 object files into a final ELF binary.
///
/// This is the primary public API for AArch64 linking. It creates an
/// `AArch64Linker` instance and dispatches to the appropriate linking
/// method based on the output type specified in the configuration.
///
/// # Arguments
///
/// * `config` — Linker configuration (target, output type, paths, flags).
/// * `inputs` — List of parsed input object files to link.
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// `Ok(LinkerOutput)` containing the complete ELF binary on success,
/// or `Err(String)` with a descriptive error message on failure.
///
/// # Examples
///
/// ```ignore
/// use bcc::backend::aarch64::linker::link_aarch64;
/// use bcc::backend::linker_common::{LinkerConfig, OutputType};
/// use bcc::common::target::Target;
/// use bcc::common::diagnostics::DiagnosticEngine;
///
/// let config = LinkerConfig::new(Target::AArch64, OutputType::Executable);
/// let mut diag = DiagnosticEngine::new();
/// let result = link_aarch64(config, vec![], &mut diag);
/// ```
pub fn link_aarch64(
    config: LinkerConfig,
    inputs: Vec<LinkerInput>,
    diagnostics: &mut DiagnosticEngine,
) -> Result<LinkerOutput, String> {
    let mut linker = AArch64Linker::new(config.clone());
    match config.output_type {
        OutputType::Executable => linker.link_executable(inputs, diagnostics),
        OutputType::SharedLibrary => linker.link_shared_library(inputs, diagnostics),
        OutputType::Relocatable => {
            // Relocatable passthrough: delegate directly to linker_common::link()
            // which handles ET_REL output without performing full linking.
            let reloc_handler = AArch64RelocationHandler::new();
            crate::backend::linker_common::link(&config, inputs, &reloc_handler, diagnostics)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(EM_AARCH64, 183);
        assert_eq!(ELFCLASS64, 2);
        assert_eq!(ELFDATA2LSB, 1);
        assert_eq!(DEFAULT_BASE_ADDRESS, 0x400000);
        assert_eq!(PAGE_SIZE, 4096);
        assert_eq!(PLT0_SIZE, 32);
        assert_eq!(PLTN_SIZE, 16);
        assert_eq!(GOT_ENTRY_SIZE, 8);
        assert_eq!(GOT_PLT_RESERVED_ENTRIES, 3);
        assert_eq!(DYNAMIC_LINKER_PATH, "/lib/ld-linux-aarch64.so.1");
    }

    #[test]
    fn test_constants_match_target() {
        let target = Target::AArch64;
        assert_eq!(target.elf_machine(), EM_AARCH64);
        assert_eq!(target.page_size(), PAGE_SIZE as usize);
        assert!(target.is_64bit());
        assert_eq!(target.dynamic_linker(), DYNAMIC_LINKER_PATH);
    }

    #[test]
    fn test_compute_adrp_page_delta() {
        // Same page → delta = 0
        assert_eq!(compute_adrp_page_delta(0x1000, 0x1004), 0);
        // Next page → delta = 1
        assert_eq!(compute_adrp_page_delta(0x2000, 0x1004), 1);
        // Previous page → delta = -1
        assert_eq!(compute_adrp_page_delta(0x1000, 0x2004), -1);
        // Large positive delta
        assert_eq!(compute_adrp_page_delta(0x400000, 0x1000), 0x3FF);
        // Zero addresses
        assert_eq!(compute_adrp_page_delta(0, 0), 0);
    }

    #[test]
    fn test_page_offset() {
        assert_eq!(page_offset(0x1234), 0x234);
        assert_eq!(page_offset(0x1000), 0);
        assert_eq!(page_offset(0xFFF), 0xFFF);
        assert_eq!(page_offset(0), 0);
    }

    #[test]
    fn test_page_align() {
        assert_eq!(page_align(0x1234), 0x1000);
        assert_eq!(page_align(0x1000), 0x1000);
        assert_eq!(page_align(0xFFF), 0);
        assert_eq!(page_align(0), 0);
    }

    #[test]
    fn test_plt0_stub_size() {
        let stub = generate_plt0_stub(0x600000, 0x400000);
        assert_eq!(stub.len(), PLT0_SIZE);
    }

    #[test]
    fn test_pltn_stub_size() {
        let stub = generate_pltn_stub(0x600100, 0x400020);
        assert_eq!(stub.len(), PLTN_SIZE);
    }

    #[test]
    fn test_plt0_first_instruction_is_stp() {
        let stub = generate_plt0_stub(0x600000, 0x400000);
        let first_insn = u32::from_le_bytes([stub[0], stub[1], stub[2], stub[3]]);
        // stp x16, x30, [sp, #-16]!
        assert_eq!(first_insn, 0xA9BF_7BF0);
    }

    #[test]
    fn test_plt0_last_instructions_are_nop() {
        let stub = generate_plt0_stub(0x600000, 0x400000);
        // Last 3 instructions (12 bytes) should be NOPs.
        for i in 0..3 {
            let offset = 20 + i * 4;
            let insn = u32::from_le_bytes([
                stub[offset],
                stub[offset + 1],
                stub[offset + 2],
                stub[offset + 3],
            ]);
            assert_eq!(
                insn, 0xD503_201F,
                "Instruction at offset {} should be NOP",
                offset
            );
        }
    }

    #[test]
    fn test_pltn_last_instruction_is_br_x17() {
        let stub = generate_pltn_stub(0x600100, 0x400020);
        let last_insn = u32::from_le_bytes([stub[12], stub[13], stub[14], stub[15]]);
        // br x17
        assert_eq!(last_insn, 0xD61F_0220);
    }

    #[test]
    fn test_encode_adrp_zero_delta() {
        let insn = encode_adrp(16, 0);
        // op=1, immlo=0, opcode=10000, immhi=0, Rd=16
        assert_eq!(insn & 0x1F, 16); // Rd
        assert_eq!((insn >> 24) & 0x1F, 0b10000); // opcode
        assert_eq!(insn >> 31, 1); // op bit
    }

    #[test]
    fn test_encode_ldr64_uimm() {
        let insn = encode_ldr64_uimm(17, 16, 2);
        assert_eq!(insn & 0x1F, 17); // Rt = x17
        assert_eq!((insn >> 5) & 0x1F, 16); // Rn = x16
        assert_eq!((insn >> 10) & 0xFFF, 2); // pimm = 2
        assert_eq!(insn & 0xFFC0_0000, 0xF940_0000); // base encoding
    }

    #[test]
    fn test_encode_add_imm64() {
        let insn = encode_add_imm64(16, 16, 0x10);
        assert_eq!(insn & 0x1F, 16); // Rd = x16
        assert_eq!((insn >> 5) & 0x1F, 16); // Rn = x16
        assert_eq!((insn >> 10) & 0xFFF, 0x10); // imm12 = 0x10
        assert_eq!(insn & 0xFF00_0000, 0x9100_0000); // base encoding
    }

    #[test]
    fn test_linker_new() {
        let config = LinkerConfig::new(Target::AArch64, OutputType::Executable);
        let linker = AArch64Linker::new(config);
        assert_eq!(linker.diagnostics.len(), 0);
    }

    #[test]
    fn test_linker_new_shared() {
        let config = LinkerConfig::new(Target::AArch64, OutputType::SharedLibrary);
        let linker = AArch64Linker::new(config);
        assert_eq!(linker.diagnostics.len(), 0);
    }

    #[test]
    fn test_build_output_section_data_empty() {
        let section = OutputSection {
            name: ".text".to_string(),
            section_type: 1,
            flags: 6,
            alignment: 4,
            fragments: vec![],
            total_size: 0,
            virtual_address: 0,
            file_offset: 0,
        };
        let data = build_output_section_data(&section);
        assert!(data.is_empty());
    }

    #[test]
    fn test_resolve_entry_point_address_found() {
        let mut symbols = FxHashMap::default();
        symbols.insert(
            "_start".to_string(),
            ResolvedSymbol {
                name: "_start".to_string(),
                final_address: 0x400100,
                size: 0,
                binding: 1,
                sym_type: 2,
                visibility: 0,
                section_name: ".text".to_string(),
                is_defined: true,
                from_object: 0,
                export_dynamic: false,
            },
        );
        let result = resolve_entry_point_address("_start", &symbols);
        assert_eq!(result, Ok(0x400100));
    }

    #[test]
    fn test_resolve_entry_point_address_not_found() {
        let symbols = FxHashMap::default();
        let result = resolve_entry_point_address("_start", &symbols);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("undefined reference"));
    }

    #[test]
    fn test_format_linker_error_with_symbol() {
        let msg = format_linker_error(
            "relocation overflow",
            "my_func",
            "value 0x10000000 exceeds ±128 MiB",
        );
        assert!(msg.contains("my_func"));
        assert!(msg.contains("relocation overflow"));
    }

    #[test]
    fn test_format_linker_error_without_symbol() {
        let msg = format_linker_error("internal error", "", "unexpected state");
        assert!(msg.contains("internal error"));
        assert!(!msg.contains("for ''"));
    }

    #[test]
    fn test_link_aarch64_empty_inputs() {
        let config = LinkerConfig::new(Target::AArch64, OutputType::Relocatable);
        let mut diag = DiagnosticEngine::new();
        // Relocatable with no inputs should succeed (empty output).
        let result = link_aarch64(config, vec![], &mut diag);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.output_type, OutputType::Relocatable);
    }

    #[test]
    fn test_wrong_target_error() {
        let config = LinkerConfig::new(Target::X86_64, OutputType::Executable);
        let mut linker = AArch64Linker::new(config);
        let mut diag = DiagnosticEngine::new();
        let result = linker.link_executable(vec![], &mut diag);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("wrong target"));
    }

    #[test]
    fn test_check_relocation_range_signed() {
        // 26-bit signed: ±33554431
        assert!(check_relocation_range("R_AARCH64_CALL26", 0, 26, true).is_ok());
        assert!(check_relocation_range("R_AARCH64_CALL26", 33554431, 26, true).is_ok());
        assert!(check_relocation_range("R_AARCH64_CALL26", -33554432, 26, true).is_ok());
        assert!(check_relocation_range("R_AARCH64_CALL26", 33554432, 26, true).is_err());
    }

    #[test]
    fn test_check_relocation_range_unsigned() {
        // 12-bit unsigned: 0..4095
        assert!(check_relocation_range("ADD_ABS_LO12", 0, 12, false).is_ok());
        assert!(check_relocation_range("ADD_ABS_LO12", 4095, 12, false).is_ok());
        assert!(check_relocation_range("ADD_ABS_LO12", 4096, 12, false).is_err());
    }
}
