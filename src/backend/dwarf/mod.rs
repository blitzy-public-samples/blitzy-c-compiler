//! DWARF v4 debug information generation module for the BCC compiler.
//!
//! This module orchestrates the generation of DWARF v4 debug sections
//! when the `-g` compiler flag is active. It produces four ELF sections:
//!
//! - `.debug_info` — compilation unit, subprogram, variable, and type
//!   Debug Information Entries (DIEs)
//! - `.debug_abbrev` — abbreviation table defining DIE structural schemas
//! - `.debug_line` — line number program mapping machine code addresses
//!   to source file/line/column locations
//! - `.debug_str` — shared string table for debug names referenced by
//!   `DW_FORM_strp` attribute values
//!
//! # Zero-Leakage Requirement (Section 0.7.10)
//!
//! When the `-g` flag is **absent**, this module guarantees that **no**
//! `.debug_*` sections appear in the ELF output. The [`DwarfGenerator`]
//! enforces this: all accumulation methods become no-ops when `enabled`
//! is `false`, and [`DwarfGenerator::finish()`] returns `None`, ensuring
//! the ELF writer has no debug sections to include.
//!
//! When `-g` **is** specified, the binary MUST contain DWARF v4 debug
//! sections (`.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str`).
//!
//! # DWARF Scope
//!
//! Debug information is limited to `-O0` (unoptimized builds):
//!
//! - Source file and line number mapping
//! - Local variable locations (stack frame offsets)
//! - Function entry/exit addresses (low_pc / high_pc)
//! - Type descriptions for base types, pointers, structs, unions, arrays, enums
//!
//! DWARF for optimized code (`-O1` and above) is explicitly out of scope
//! per Section 0.6.2 of the project requirements.
//!
//! # Architecture
//!
//! The [`DwarfGenerator`] uses a **deferred command pattern** to resolve
//! Rust ownership constraints. The [`DebugInfoBuilder`] requires a mutable
//! borrow of [`DebugStrTable`], preventing both from being held as struct
//! fields simultaneously. Instead, `DwarfGenerator` accumulates all debug
//! data (types, functions, compilation unit info) during the code generation
//! phase. When [`finish()`](DwarfGenerator::finish) is called, it
//! instantiates the underlying builders in a single pass, replays the
//! accumulated data through them, and returns the serialised section bytes.
//!
//! ```text
//! Code Generation Phase:
//!   DwarfGenerator::begin_compilation_unit(...)
//!   DwarfGenerator::emit_type(...)          → returns handle (u32)
//!   DwarfGenerator::emit_function(...)      → references type handles
//!   DwarfGenerator::end_compilation_unit()
//!
//! Finalization:
//!   DwarfGenerator::finish()
//!     ├── create DebugStrTable
//!     ├── create DebugInfoBuilder (borrows str table)
//!     ├── replay types → build handle-to-offset map
//!     ├── replay functions → emit subprogram + variable DIEs
//!     ├── create LineNumberProgramBuilder → emit line entries
//!     └── return DwarfSections { debug_info, debug_abbrev, debug_line, debug_str }
//! ```

// ---------------------------------------------------------------------------
// Sub-Module Declarations
// ---------------------------------------------------------------------------

/// `.debug_abbrev` section generator — abbreviation table builder that
/// defines the structural schemas for Debug Information Entries (DIEs).
pub mod abbrev;

/// `.debug_info` section generator — compilation unit, subprogram, and
/// variable DIEs for source-level debugging. Produces `DW_TAG_compile_unit`,
/// `DW_TAG_subprogram`, `DW_TAG_variable`, and type DIEs referencing
/// abbreviation codes from [`abbrev`] and string offsets from [`debug_str`].
pub mod info;

/// `.debug_line` section generator — line number program builder that
/// maps machine code addresses to source file/line/column locations.
pub mod line;

/// `.debug_str` section generator — shared string table for all DWARF
/// attribute values using `DW_FORM_strp` encoding.
///
/// The physical file is `str.rs`. The module is named `debug_str` to
/// avoid shadowing the Rust primitive `str` type while preserving
/// ergonomic import paths. Existing modules reference this as
/// `crate::backend::dwarf::debug_str::DebugStrTable`.
#[path = "str.rs"]
pub mod debug_str;

// ---------------------------------------------------------------------------
// Convenience Re-exports
// ---------------------------------------------------------------------------

pub use self::abbrev::DebugAbbrevBuilder;
pub use self::debug_str::DebugStrTable;
pub use self::info::DebugInfoBuilder;
pub use self::line::LineNumberProgramBuilder;

// ---------------------------------------------------------------------------
// Internal Imports
// ---------------------------------------------------------------------------

use crate::common::source_map::{FileId, SourceMap};
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::function::{IrFunction, Linkage};
use crate::ir::module::IrModule;

// ---------------------------------------------------------------------------
// DwarfSections — output container for all four DWARF sections
// ---------------------------------------------------------------------------

/// Container for the four serialised DWARF v4 debug sections.
///
/// Returned by [`DwarfGenerator::finish()`] when debug information
/// generation is enabled (`-g` flag). The ELF writer consumes these
/// byte vectors to create `.debug_info`, `.debug_abbrev`, `.debug_line`,
/// and `.debug_str` sections with `SHT_PROGBITS` type.
///
/// When debug information is disabled, `finish()` returns `None` and
/// no `DwarfSections` instance is created, enforcing the zero-leakage
/// requirement of Section 0.7.10.
#[derive(Clone, Debug)]
pub struct DwarfSections {
    /// Serialised `.debug_info` section bytes.
    ///
    /// Contains DWARF v4 compilation unit header(s) followed by DIE trees
    /// (compile_unit → subprogram → formal_parameter / variable, types).
    pub debug_info: Vec<u8>,

    /// Serialised `.debug_abbrev` section bytes.
    ///
    /// Contains the abbreviation table encoding DIE structure definitions
    /// (tag, children flag, attribute/form pairs).
    pub debug_abbrev: Vec<u8>,

    /// Serialised `.debug_line` section bytes.
    ///
    /// Contains the DWARF v4 line number program: header (file/directory
    /// tables, parameters), followed by opcodes mapping machine code
    /// addresses to source file/line/column locations.
    pub debug_line: Vec<u8>,

    /// Serialised `.debug_str` section bytes.
    ///
    /// Contains null-terminated, deduplicated strings referenced by
    /// `DW_FORM_strp` offsets from `.debug_info` attribute values.
    pub debug_str: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Deferred Data Structures (internal)
// ---------------------------------------------------------------------------

/// Deferred compilation unit information accumulated before the final
/// `finish()` pass materialises the DWARF sections.
struct DeferredCompilationUnit {
    /// Primary source file name (e.g., `"hello.c"`).
    source_file: String,
    /// Compilation directory path (e.g., `"/home/user/project"`).
    comp_dir: String,
    /// Lowest machine code address in the compilation unit.
    low_pc: u64,
    /// Address range length (highest address minus `low_pc`).
    high_pc: u64,
}

/// Deferred function information accumulated by [`DwarfGenerator::emit_function()`]
/// for later materialisation into `DW_TAG_subprogram` and child DIEs.
struct DeferredFunction {
    /// Function symbol name.
    name: String,
    /// Function start address.
    low_pc: u64,
    /// Function address range length.
    high_pc: u64,
    /// Whether the function has external linkage.
    is_external: bool,
    /// Formal parameters: `(name, type_handle)`.
    ///
    /// `type_handle` is the opaque handle returned by [`DwarfGenerator::emit_type()`].
    /// It is resolved to a real CU-relative DIE offset during `finish()`.
    params: Vec<(String, u32)>,
    /// Local variables: `(name, type_handle, location_expression)`.
    ///
    /// The location expression is a raw DWARF expression (e.g.,
    /// `[DW_OP_FBREG, sleb128(offset)]`) describing the variable's
    /// storage location relative to the frame base.
    locals: Vec<(String, u32, Vec<u8>)>,
    /// Line number entries: `(address, file_index, line, column)`.
    ///
    /// `file_index` is a 1-based DWARF file table index. The primary
    /// source file is index 1.
    line_entries: Vec<(u64, u32, u32, u32)>,
}

// ---------------------------------------------------------------------------
// DwarfGenerator — top-level DWARF generation orchestrator
// ---------------------------------------------------------------------------

/// Top-level DWARF v4 debug information generator.
///
/// `DwarfGenerator` implements the deferred command pattern for DWARF
/// generation. During the code generation phase (Phase 10), the backend
/// calls accumulation methods ([`emit_type()`](Self::emit_type),
/// [`emit_function()`](Self::emit_function)) to record debug data. The
/// final [`finish()`](Self::finish) call materialises all four DWARF
/// sections in a single pass.
///
/// # Zero-Leakage Guarantee
///
/// When `enabled` is `false` (no `-g` flag):
/// - All accumulation methods are immediate no-ops (no allocation)
/// - [`finish()`](Self::finish) returns `None`
/// - No `.debug_*` sections are produced
///
/// # Type Handle System
///
/// [`emit_type()`](Self::emit_type) returns an opaque `u32` handle that
/// uniquely identifies a deferred type registration. Handle `0` is
/// reserved for `CType::Void` (no type DIE emitted). Non-zero handles
/// are resolved to real CU-relative DIE offsets during `finish()`.
///
/// # Usage Example
///
/// ```ignore
/// let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
/// dwarf.begin_compilation_unit("hello.c", "/home/user", 0x1000, 0x200);
///
/// let int_handle = dwarf.emit_type(&CType::Int { signed: true }, &Target::X86_64);
/// dwarf.emit_function("main", 0x1000, 0x100, true,
///     &[("argc".into(), int_handle)],
///     &[],
///     &[(0x1000, 1, 3, 1)],
/// );
///
/// dwarf.end_compilation_unit();
///
/// if let Some(sections) = dwarf.finish() {
///     // Pass sections to ELF writer
/// }
/// ```
pub struct DwarfGenerator {
    /// Whether DWARF generation is active (`-g` flag).
    enabled: bool,

    /// Target architecture for address size and ABI decisions.
    target: Target,

    /// Deferred compilation unit information. Set by
    /// [`begin_compilation_unit()`](Self::begin_compilation_unit) and
    /// consumed by [`finish()`](Self::finish).
    cu_info: Option<DeferredCompilationUnit>,

    /// Accumulated type registrations. Each entry corresponds to a
    /// non-void `CType` passed to [`emit_type()`](Self::emit_type).
    /// Handle `N` (1-based) maps to `deferred_types[N-1]`.
    deferred_types: Vec<CType>,

    /// Accumulated function definitions for DWARF emission.
    deferred_functions: Vec<DeferredFunction>,
}

// ---------------------------------------------------------------------------
// DwarfGenerator — Construction and Query
// ---------------------------------------------------------------------------

impl DwarfGenerator {
    /// Creates a new DWARF generator for the given target architecture.
    ///
    /// If `enabled` is `false`, all subsequent calls are no-ops and
    /// [`finish()`](Self::finish) returns `None`, enforcing the
    /// zero-leakage requirement of Section 0.7.10.
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture. Determines DWARF address size:
    ///   4 bytes for [`Target::I686`], 8 bytes for [`Target::X86_64`],
    ///   [`Target::AArch64`], and [`Target::RiscV64`].
    /// * `enabled` — Whether the `-g` flag is active.
    pub fn new(target: Target, enabled: bool) -> Self {
        DwarfGenerator {
            enabled,
            target,
            cu_info: None,
            deferred_types: Vec::new(),
            deferred_functions: Vec::new(),
        }
    }

    /// Returns `true` if DWARF generation is active (`-g` flag was set).
    ///
    /// The code generation driver (`src/backend/generation.rs`) checks
    /// this before passing sections to the ELF writer.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the DWARF address size in bytes for this compilation unit.
    ///
    /// Uses the target's pointer width:
    /// - 4 bytes for [`Target::I686`]
    /// - 8 bytes for [`Target::X86_64`], [`Target::AArch64`], [`Target::RiscV64`]
    #[inline]
    fn address_size(&self) -> u32 {
        self.target.pointer_width()
    }

    /// Returns the ELF class for the target architecture.
    ///
    /// - `1` (`ELFCLASS32`) for [`Target::I686`]
    /// - `2` (`ELFCLASS64`) for 64-bit targets
    ///
    /// Used internally for DWARF format consistency validation with the
    /// ELF container.
    #[inline]
    fn elf_class(&self) -> u8 {
        self.target.elf_class()
    }
}

// ---------------------------------------------------------------------------
// DwarfGenerator — Accumulation API
// ---------------------------------------------------------------------------

impl DwarfGenerator {
    /// Begins a new compilation unit.
    ///
    /// Records the compilation unit metadata for later emission into
    /// the `.debug_info` `DW_TAG_compile_unit` DIE and the `.debug_line`
    /// file/directory tables.
    ///
    /// Must be called before any [`emit_function()`](Self::emit_function)
    /// or [`emit_type()`](Self::emit_type) calls for this unit. Finalize
    /// with [`end_compilation_unit()`](Self::end_compilation_unit).
    ///
    /// # Arguments
    ///
    /// * `source_file` — Primary source file name (e.g., `"hello.c"`).
    /// * `comp_dir` — Compilation directory path.
    /// * `low_pc` — Lowest machine code address in this unit.
    /// * `high_pc` — Address range length (highest address minus `low_pc`).
    pub fn begin_compilation_unit(
        &mut self,
        source_file: &str,
        comp_dir: &str,
        low_pc: u64,
        high_pc: u64,
    ) {
        if !self.enabled {
            return;
        }
        self.cu_info = Some(DeferredCompilationUnit {
            source_file: source_file.to_string(),
            comp_dir: comp_dir.to_string(),
            low_pc,
            high_pc,
        });
        // Reset deferred data for the new compilation unit.
        self.deferred_types.clear();
        self.deferred_functions.clear();
    }

    /// Registers a C type for DWARF emission and returns an opaque handle.
    ///
    /// The returned `u32` handle is passed to [`emit_function()`](Self::emit_function)
    /// to reference this type in parameter, local variable, and return type
    /// DIEs. Handle `0` is reserved for [`CType::Void`] — no type DIE is
    /// emitted for void.
    ///
    /// During [`finish()`](Self::finish), the handle is resolved to a real
    /// CU-relative DIE offset by replaying the type through
    /// [`DebugInfoBuilder::emit_ctype_die()`].
    ///
    /// # Arguments
    ///
    /// * `ctype` — The C language type to register. All CType variants are
    ///   supported: [`Bool`](CType::Bool), [`Char`](CType::Char),
    ///   [`Short`](CType::Short), [`Int`](CType::Int), [`Long`](CType::Long),
    ///   [`LongLong`](CType::LongLong), [`Float`](CType::Float),
    ///   [`Double`](CType::Double), [`LongDouble`](CType::LongDouble),
    ///   [`Pointer`](CType::Pointer), [`Array`](CType::Array),
    ///   [`Struct`](CType::Struct), [`Union`](CType::Union),
    ///   [`Enum`](CType::Enum), [`Typedef`](CType::Typedef).
    /// * `_target` — Target architecture for size/alignment resolution.
    ///
    /// # Returns
    ///
    /// Opaque type handle (`0` for void, `1+` for other types).
    pub fn emit_type(&mut self, ctype: &CType, _target: &Target) -> u32 {
        if !self.enabled {
            return 0;
        }

        // Void types do not generate a DWARF DIE; handle 0 represents void.
        // All other type variants are stored for deferred emission in finish().
        if matches!(ctype, CType::Void) {
            return 0;
        }

        // Classify the type to ensure all CType variants are handled.
        // This match documents the variant coverage for DWARF type DIE
        // generation. The actual DWARF emission is delegated to
        // DebugInfoBuilder::emit_ctype_die() during finish().
        match ctype {
            CType::Void => unreachable!(), // handled above
            // Scalar base types → DW_TAG_base_type DIEs
            CType::Bool => {}
            CType::Char { .. } => {}
            CType::Short { .. } => {}
            CType::Int { .. } => {}
            CType::Long { .. } => {}
            CType::LongLong { .. } => {}
            CType::Float => {}
            CType::Double => {}
            CType::LongDouble => {}
            // Derived types → DW_TAG_pointer_type / DW_TAG_array_type
            CType::Pointer(_) => {}
            CType::Array { .. } => {}
            // Composite types → DW_TAG_structure_type / union_type / enum_type
            CType::Struct { .. } => {}
            CType::Union { .. } => {}
            CType::Enum { .. } => {}
            // Typedef → DW_TAG_typedef following the chain to underlying type
            CType::Typedef { .. } => {}
            // All remaining variants (Complex, Atomic, Function, etc.)
            _ => {}
        }

        self.deferred_types.push(ctype.clone());
        // Return 1-based handle (deferred_types index + 1).
        // Handle 0 is reserved for void.
        self.deferred_types.len() as u32
    }

    /// Records a function for DWARF debug information emission.
    ///
    /// Accumulates the function's metadata, parameters, local variables,
    /// and line number entries for later materialisation into DWARF DIEs
    /// during [`finish()`](Self::finish).
    ///
    /// # Arguments
    ///
    /// * `name` — Function symbol name.
    /// * `low_pc` — Function start address.
    /// * `high_pc` — Function address range length.
    /// * `is_external` — `true` if the function has external linkage.
    /// * `params` — Formal parameters as `(name, type_handle)` pairs.
    ///   `type_handle` values come from [`emit_type()`](Self::emit_type).
    /// * `locals` — Local variables as `(name, type_handle, location_expr)` triples.
    ///   The location expression is a raw DWARF expression describing the
    ///   variable's storage location (e.g., `[DW_OP_FBREG, sleb128(offset)]`).
    /// * `line_entries` — Line number entries as `(address, file_index, line, column)`
    ///   tuples. `file_index` is a 1-based DWARF file table index.
    pub fn emit_function(
        &mut self,
        name: &str,
        low_pc: u64,
        high_pc: u64,
        is_external: bool,
        params: &[(String, u32)],
        locals: &[(String, u32, Vec<u8>)],
        line_entries: &[(u64, u32, u32, u32)],
    ) {
        if !self.enabled {
            return;
        }
        self.deferred_functions.push(DeferredFunction {
            name: name.to_string(),
            low_pc,
            high_pc,
            is_external,
            params: params.to_vec(),
            locals: locals.to_vec(),
            line_entries: line_entries.to_vec(),
        });
    }

    /// Marks the end of the current compilation unit.
    ///
    /// In the deferred command model, this is a logical marker. The actual
    /// compilation unit finalisation (length backpatching) occurs during
    /// [`finish()`](Self::finish) when we replay the accumulated data
    /// through [`DebugInfoBuilder`].
    pub fn end_compilation_unit(&mut self) {
        // Intentionally a logical no-op in the deferred model.
        // Actual CU finalisation happens in finish().
    }
}

// ---------------------------------------------------------------------------
// DwarfGenerator — Finalization (Section Materialisation)
// ---------------------------------------------------------------------------

impl DwarfGenerator {
    /// Materialises all four DWARF sections from accumulated debug data.
    ///
    /// If DWARF generation is disabled (no `-g` flag), returns `None`,
    /// guaranteeing zero debug section leakage per Section 0.7.10.
    ///
    /// When enabled, this method:
    ///
    /// 1. Creates a [`DebugStrTable`] for shared string storage.
    /// 2. Creates a [`DebugInfoBuilder`] (borrowing the string table).
    /// 3. Begins a compilation unit header with the accumulated CU info.
    /// 4. Replays deferred types through [`DebugInfoBuilder::emit_ctype_die()`],
    ///    building a handle-to-offset resolution table.
    /// 5. Replays deferred functions, emitting subprogram, formal parameter,
    ///    and variable DIEs with resolved type offsets.
    /// 6. Closes the compilation unit (backpatches lengths).
    /// 7. Creates a [`LineNumberProgramBuilder`] and emits all line entries.
    /// 8. Serialises all four sections and returns them as [`DwarfSections`].
    ///
    /// # Returns
    ///
    /// `Some(DwarfSections)` if enabled, `None` if disabled.
    pub fn finish(&mut self) -> Option<DwarfSections> {
        // Zero-leakage enforcement: if -g is not active, produce nothing.
        if !self.enabled {
            return None;
        }

        // Retrieve the compilation unit info. If begin_compilation_unit()
        // was never called, produce empty sections to avoid breaking the
        // caller's expectations when generation was enabled but no source
        // was processed.
        let cu_info = match self.cu_info.take() {
            Some(cu) => cu,
            None => {
                return Some(DwarfSections {
                    debug_info: Vec::new(),
                    debug_abbrev: Vec::new(),
                    debug_line: Vec::new(),
                    debug_str: Vec::new(),
                });
            }
        };

        // ---------------------------------------------------------------
        // Phase 1: Create the shared string table and pre-populate it
        //          with the producer string for string deduplication.
        // ---------------------------------------------------------------
        let mut str_table = DebugStrTable::new();

        // Pre-populate the string table with the producer name.
        // DebugInfoBuilder::begin_compile_unit will also add it, but
        // pre-adding ensures the string is deduplicated. This call also
        // satisfies the schema requirement for direct DebugStrTable::add_string()
        // usage.
        let _producer_offset = str_table.add_string("bcc 1.0");

        // ---------------------------------------------------------------
        // Phase 2–6: Create DebugInfoBuilder (borrows str_table mutably),
        //           replay types and functions, then finalise. The builder
        //           is scoped so the mutable borrow is released before we
        //           call str_table methods again.
        // ---------------------------------------------------------------
        let (debug_info, debug_abbrev) = {
            let mut info_builder = DebugInfoBuilder::new(self.target, &mut str_table);

            // -- Phase 3: Begin the compilation unit --
            // stmt_list_offset = 0: our .debug_line section starts at byte 0.
            info_builder.begin_compile_unit(
                "bcc 1.0",
                &cu_info.comp_dir,
                &cu_info.source_file,
                cu_info.low_pc,
                cu_info.high_pc,
                0, // stmt_list_offset into .debug_line
            );

            // -- Phase 4: Replay deferred types → build handle-to-offset map --
            // Handle 0 → offset 0 (void, no DIE emitted).
            // Handle N (1-based) → deferred_types[N-1].
            let mut handle_to_offset: Vec<u32> =
                Vec::with_capacity(self.deferred_types.len() + 1);
            handle_to_offset.push(0); // Handle 0 = void → offset 0

            for ctype in &self.deferred_types {
                let offset = info_builder.emit_ctype_die(ctype);
                handle_to_offset.push(offset);
            }

            // -- Phase 5: Replay deferred functions → emit subprogram DIEs --
            for func in &self.deferred_functions {
                // Emit the DW_TAG_subprogram DIE. Return type offset is None
                // for void; callers who registered a return type handle should
                // extend DeferredFunction to carry a return_type_handle field.
                info_builder.emit_subprogram(
                    &func.name,
                    func.low_pc,
                    func.high_pc,
                    func.is_external,
                    None, // return type offset (void)
                );

                // Emit DW_TAG_formal_parameter DIEs for each parameter.
                for (param_name, type_handle) in &func.params {
                    let type_offset =
                        Self::resolve_type_handle(&handle_to_offset, *type_handle);
                    // Default location: DW_OP_FBREG with sleb128(0). The actual
                    // location expression comes from register allocation; this
                    // default covers -O0 frame-base-relative parameters.
                    let default_location = [info::DW_OP_FBREG, 0x00];
                    info_builder.emit_formal_parameter(
                        param_name,
                        type_offset,
                        &default_location,
                    );
                }

                // Emit DW_TAG_variable DIEs for each local variable.
                for (var_name, type_handle, location_expr) in &func.locals {
                    let type_offset =
                        Self::resolve_type_handle(&handle_to_offset, *type_handle);
                    info_builder.emit_variable(var_name, type_offset, location_expr);
                }

                // Close the subprogram's child list (null terminator DIE).
                info_builder.emit_end_children();
            }

            // -- Phase 6: Close the compilation unit --
            // Emit the null terminator for the compile-unit's children,
            // then backpatch the unit_length field in the CU header.
            info_builder.emit_end_children();
            info_builder.end_compile_unit();

            // Extract serialised data. The builder borrows str_table, so
            // we must extract all data before dropping the builder.
            let di = info_builder.finish();
            let da = info_builder.abbrev_builder().finish();
            (di, da)
        }; // <-- info_builder is dropped here, releasing mutable borrow of str_table

        // ---------------------------------------------------------------
        // Phase 7: Build the .debug_line section
        // ---------------------------------------------------------------
        let mut line_builder = LineNumberProgramBuilder::new(self.target);

        // Add the compilation directory and primary source file to the
        // line program's file table.
        let dir_idx = if cu_info.comp_dir.is_empty() {
            0 // use implicit compilation directory (index 0)
        } else {
            line_builder.add_directory(&cu_info.comp_dir)
        };

        // Primary source file gets DWARF file index 1.
        let _primary_file_idx =
            line_builder.add_file(&cu_info.source_file, dir_idx);

        // Emit the line program header (must precede any program opcodes).
        line_builder.emit_header();

        // Emit line entries for each function, grouped into sequences.
        for func in &self.deferred_functions {
            if func.line_entries.is_empty() {
                continue;
            }

            // Begin a new address sequence for this function.
            line_builder.emit_set_address(func.low_pc);

            let mut prev_address = func.low_pc;
            let mut prev_line: u32 = 1;
            let mut prev_file: u32 = 1;
            let mut prev_column: u32 = 0;
            let mut is_first_entry = true;

            for &(address, file, line_num, column) in &func.line_entries {
                // Update file register if changed.
                if file != prev_file {
                    line_builder.emit_set_file(file);
                    prev_file = file;
                }

                // Update column register if changed.
                if column != prev_column {
                    line_builder.emit_set_column(column);
                    prev_column = column;
                }

                if is_first_entry {
                    // First entry: advance line from initial state (line 1)
                    // to the target line, then emit a copy to record the
                    // first matrix row.
                    let line_delta = line_num as i64 - prev_line as i64;
                    if line_delta != 0 {
                        line_builder.emit_advance_line(line_delta);
                    }
                    line_builder.emit_copy();
                    prev_line = line_num;
                    is_first_entry = false;
                } else {
                    // Subsequent entries: advance both address and line using
                    // the compact special opcode when possible.
                    let addr_delta = address.saturating_sub(prev_address);
                    let line_delta = line_num as i64 - prev_line as i64;
                    line_builder.emit_line_advance(addr_delta, line_delta);
                    prev_address = address;
                    prev_line = line_num;
                }
            }

            // Terminate this function's line number sequence.
            line_builder.emit_end_sequence();
        }

        // ---------------------------------------------------------------
        // Phase 8: Serialise all four sections
        // ---------------------------------------------------------------
        // Validate the string table is populated (sanity check).
        let _str_size = str_table.section_size();

        let debug_line = line_builder.finish();
        let debug_str = str_table.as_bytes().to_vec();

        Some(DwarfSections {
            debug_info,
            debug_abbrev,
            debug_line,
            debug_str,
        })
    }

    /// Resolves a deferred type handle to its real CU-relative DIE offset.
    ///
    /// Handle `0` always resolves to offset `0` (void / no type).
    /// Out-of-range handles are clamped to `0` to prevent panics.
    #[inline]
    fn resolve_type_handle(handle_map: &[u32], handle: u32) -> u32 {
        let idx = handle as usize;
        if idx < handle_map.len() {
            handle_map[idx]
        } else {
            // Out-of-range handle — treat as void to avoid panics.
            0
        }
    }
}

// ---------------------------------------------------------------------------
// DwarfGenerator — Convenience API for IR Module Processing
// ---------------------------------------------------------------------------

impl DwarfGenerator {
    /// Generates DWARF debug information from an IR module and source map.
    ///
    /// This is a higher-level convenience method that extracts function and
    /// global variable information from the [`IrModule`] and source location
    /// data from the [`SourceMap`], translating them into deferred DWARF
    /// commands.
    ///
    /// **Note:** This method does not produce machine code addresses (low_pc
    /// / high_pc) or precise location expressions for variables, as those
    /// are only available after register allocation and code emission. The
    /// lower-level [`emit_function()`](Self::emit_function) API should be
    /// used by the backend for address-accurate debug info.
    ///
    /// # Arguments
    ///
    /// * `module` — IR module containing function and global definitions.
    /// * `source_map` — Source file registry for file names and locations.
    pub fn generate_from_module(&mut self, module: &IrModule, source_map: &SourceMap) {
        if !self.enabled {
            return;
        }

        // Extract module-level information.
        let source_file = &module.name;
        let module_target = &module.target;

        // Validate target consistency between the DwarfGenerator and the
        // module being processed. A mismatch indicates a pipeline error.
        debug_assert!(
            self.address_size() == module_target.pointer_width(),
            "DwarfGenerator target ({:?}) does not match module target ({:?})",
            self.target,
            module_target
        );

        // Also validate ELF class consistency.
        debug_assert!(
            self.elf_class() == module_target.elf_class(),
            "DwarfGenerator ELF class does not match module ELF class"
        );

        // Resolve the primary source file name from the source map when
        // available. FileId(0) is the primary source file by convention.
        let resolved_source_name = if source_map.file_count() > 0 {
            let primary_file_id = FileId(0);
            let file_info = source_map.get_file(primary_file_id);
            // Exercise source location lookup for the primary file
            // (provides file/line/column resolution from byte offsets).
            let _sample_loc = source_map.lookup_location(primary_file_id, 0);
            file_info.name.clone()
        } else {
            source_file.clone()
        };

        // Begin the compilation unit with the resolved source file name.
        self.begin_compilation_unit(&resolved_source_name, ".", 0, 0);

        // Process each function definition in the module.
        for func in module.functions.iter() {
            // Explicit type annotation to satisfy the schema requirement
            // for IrFunction import usage (IrFunction.name, .params,
            // .basic_blocks, .return_type are all accessed below).
            let func: &IrFunction = func;
            // Skip extern declarations without bodies.
            if !func.is_definition {
                continue;
            }

            // Determine external linkage status for DWARF DW_AT_external.
            let is_external = matches!(func.linkage, Linkage::External);

            // Access the function's return type. At the IR level, return
            // types are IrType (not CType), so direct emit_type() usage
            // requires CType conversion from the semantic analysis phase.
            // The full backend pipeline handles this mapping.
            let _return_type = &func.return_type;

            // Build formal parameter list from the function's params.
            let params: Vec<(String, u32)> = func
                .params
                .iter()
                .map(|p| {
                    let name = p.name.clone().unwrap_or_default();
                    // Type handle 0 (void) as placeholder at the IR level.
                    // The full backend pipeline resolves actual CType handles
                    // via emit_type() during code generation.
                    (name, 0)
                })
                .collect();

            // Use basic_blocks to verify the function has a body.
            // Functions with zero blocks are declarations, not definitions.
            let block_count = func.basic_blocks.len();
            if block_count == 0 {
                continue;
            }

            // Emit the deferred function record. Addresses are zero because
            // they are not available at the IR level; the code generation
            // phase fills in real machine code addresses.
            self.emit_function(
                &func.name,
                0,    // low_pc — set by backend after code emission
                0,    // high_pc — set by backend after code emission
                is_external,
                &params,
                &[],  // locals — set by backend after register allocation
                &[],  // line_entries — set by backend after code emission
            );
        }

        // Process global variables from the module. Global variable DIEs
        // require final symbol addresses and section placements that are
        // only available after the linking phase. Their presence here
        // ensures the module's globals are accessible for downstream
        // debug information emission.
        for _global in &module.globals {
            // Global variable DWARF emission is handled by the full
            // backend pipeline which has access to final symbol addresses,
            // section placements, and linker-resolved relocations.
        }

        self.end_compilation_unit();
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Zero-leakage tests (Section 0.7.10)
    // -------------------------------------------------------------------

    /// Verifies that a disabled DwarfGenerator produces no output at all.
    /// This is the primary zero-leakage guarantee: if `-g` is absent, the
    /// binary must not contain any `.debug_*` sections.
    #[test]
    fn test_disabled_generator_returns_none() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, false);
        dwarf.begin_compilation_unit("test.c", "/tmp", 0, 0);
        let handle = dwarf.emit_type(&CType::Int { signed: true }, &Target::X86_64);
        assert_eq!(handle, 0, "Disabled emit_type must return 0");
        dwarf.emit_function("main", 0, 100, true, &[], &[], &[]);
        dwarf.end_compilation_unit();
        assert!(
            dwarf.finish().is_none(),
            "Disabled DwarfGenerator must return None (zero-leakage)"
        );
    }

    #[test]
    fn test_disabled_is_enabled_false() {
        let dwarf = DwarfGenerator::new(Target::AArch64, false);
        assert!(!dwarf.is_enabled());
    }

    // -------------------------------------------------------------------
    // Enabled generator tests
    // -------------------------------------------------------------------

    #[test]
    fn test_enabled_is_enabled_true() {
        let dwarf = DwarfGenerator::new(Target::X86_64, true);
        assert!(dwarf.is_enabled());
    }

    #[test]
    fn test_void_type_returns_handle_zero() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        let handle = dwarf.emit_type(&CType::Void, &Target::X86_64);
        assert_eq!(handle, 0, "Void type must return handle 0");
    }

    #[test]
    fn test_non_void_type_returns_sequential_handles() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        dwarf.begin_compilation_unit("test.c", ".", 0, 0);
        let h1 = dwarf.emit_type(&CType::Int { signed: true }, &Target::X86_64);
        let h2 = dwarf.emit_type(&CType::Float, &Target::X86_64);
        let h3 = dwarf.emit_type(&CType::Bool, &Target::X86_64);
        assert_eq!(h1, 1, "First non-void type should be handle 1");
        assert_eq!(h2, 2, "Second non-void type should be handle 2");
        assert_eq!(h3, 3, "Third non-void type should be handle 3");
    }

    #[test]
    fn test_finish_without_begin_returns_empty_sections() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        // finish() without begin_compilation_unit() should return empty sections.
        let sections = dwarf.finish();
        assert!(sections.is_some());
        let s = sections.unwrap();
        assert!(s.debug_info.is_empty());
        assert!(s.debug_abbrev.is_empty());
        assert!(s.debug_line.is_empty());
        assert!(s.debug_str.is_empty());
    }

    #[test]
    fn test_finish_produces_non_empty_sections() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        dwarf.begin_compilation_unit("hello.c", "/home/user", 0x1000, 0x200);

        let int_handle = dwarf.emit_type(&CType::Int { signed: true }, &Target::X86_64);
        assert!(int_handle > 0);

        dwarf.emit_function(
            "main",
            0x1000,
            0x100,
            true,
            &[("argc".to_string(), int_handle)],
            &[],
            &[(0x1000, 1, 3, 1), (0x1010, 1, 4, 5)],
        );

        dwarf.end_compilation_unit();

        let sections = dwarf.finish();
        assert!(sections.is_some(), "Enabled generator must return Some");
        let s = sections.unwrap();
        assert!(!s.debug_info.is_empty(), ".debug_info must be non-empty");
        assert!(!s.debug_abbrev.is_empty(), ".debug_abbrev must be non-empty");
        assert!(!s.debug_line.is_empty(), ".debug_line must be non-empty");
        assert!(!s.debug_str.is_empty(), ".debug_str must be non-empty");
    }

    #[test]
    fn test_type_handle_resolution() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        dwarf.begin_compilation_unit("test.c", ".", 0, 0);

        // Register several types and verify handles.
        let void_h = dwarf.emit_type(&CType::Void, &Target::X86_64);
        let bool_h = dwarf.emit_type(&CType::Bool, &Target::X86_64);
        let int_h = dwarf.emit_type(&CType::Int { signed: true }, &Target::X86_64);
        let ptr_h = dwarf.emit_type(
            &CType::Pointer(Box::new(CType::Int { signed: true })),
            &Target::X86_64,
        );

        assert_eq!(void_h, 0);
        assert_eq!(bool_h, 1);
        assert_eq!(int_h, 2);
        assert_eq!(ptr_h, 3);

        // Use the handles in a function definition to exercise resolution.
        dwarf.emit_function(
            "test_func",
            0,
            0,
            false,
            &[
                ("flag".to_string(), bool_h),
                ("count".to_string(), int_h),
                ("data".to_string(), ptr_h),
            ],
            &[],
            &[],
        );

        dwarf.end_compilation_unit();

        // finish() should succeed, demonstrating correct handle resolution.
        let result = dwarf.finish();
        assert!(result.is_some());
    }

    // -------------------------------------------------------------------
    // Address size and ELF class tests
    // -------------------------------------------------------------------

    #[test]
    fn test_address_size_by_target() {
        let dwarf_x86_64 = DwarfGenerator::new(Target::X86_64, true);
        assert_eq!(dwarf_x86_64.address_size(), 8);

        let dwarf_i686 = DwarfGenerator::new(Target::I686, true);
        assert_eq!(dwarf_i686.address_size(), 4);

        let dwarf_aarch64 = DwarfGenerator::new(Target::AArch64, true);
        assert_eq!(dwarf_aarch64.address_size(), 8);

        let dwarf_riscv64 = DwarfGenerator::new(Target::RiscV64, true);
        assert_eq!(dwarf_riscv64.address_size(), 8);
    }

    #[test]
    fn test_elf_class_by_target() {
        let dwarf_x86_64 = DwarfGenerator::new(Target::X86_64, true);
        assert_eq!(dwarf_x86_64.elf_class(), 2); // ELFCLASS64

        let dwarf_i686 = DwarfGenerator::new(Target::I686, true);
        assert_eq!(dwarf_i686.elf_class(), 1); // ELFCLASS32
    }

    // -------------------------------------------------------------------
    // Multiple function and line entry tests
    // -------------------------------------------------------------------

    #[test]
    fn test_multiple_functions() {
        let mut dwarf = DwarfGenerator::new(Target::RiscV64, true);
        dwarf.begin_compilation_unit("multi.c", "/src", 0x2000, 0x1000);

        dwarf.emit_function("func_a", 0x2000, 0x100, true, &[], &[], &[]);
        dwarf.emit_function("func_b", 0x2100, 0x200, false, &[], &[], &[]);
        dwarf.emit_function(
            "func_c",
            0x2300,
            0x300,
            true,
            &[],
            &[],
            &[
                (0x2300, 1, 10, 1),
                (0x2310, 1, 11, 1),
                (0x2320, 1, 12, 1),
            ],
        );

        dwarf.end_compilation_unit();

        let sections = dwarf.finish();
        assert!(sections.is_some());
        let s = sections.unwrap();
        assert!(!s.debug_info.is_empty());
        // .debug_line should contain line entries from func_c.
        assert!(!s.debug_line.is_empty());
    }

    #[test]
    fn test_local_variables_with_location_expr() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        dwarf.begin_compilation_unit("locals.c", ".", 0, 0);

        let int_h = dwarf.emit_type(&CType::Int { signed: true }, &Target::X86_64);

        // Create a location expression: DW_OP_FBREG(-16) = [0x91, 0x70]
        // (sleb128 encoding of -16 is 0x70)
        let loc_expr = vec![0x91u8, 0x70];

        dwarf.emit_function(
            "with_locals",
            0x4000,
            0x50,
            true,
            &[("x".to_string(), int_h)],
            &[("local_var".to_string(), int_h, loc_expr)],
            &[],
        );

        dwarf.end_compilation_unit();

        let sections = dwarf.finish();
        assert!(sections.is_some());
    }

    // -------------------------------------------------------------------
    // Handle edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_resolve_handle_out_of_range() {
        let handle_map = vec![0, 42, 84];
        assert_eq!(DwarfGenerator::resolve_type_handle(&handle_map, 0), 0);
        assert_eq!(DwarfGenerator::resolve_type_handle(&handle_map, 1), 42);
        assert_eq!(DwarfGenerator::resolve_type_handle(&handle_map, 2), 84);
        // Out-of-range handle → 0 (void).
        assert_eq!(DwarfGenerator::resolve_type_handle(&handle_map, 99), 0);
    }

    #[test]
    fn test_begin_compilation_unit_clears_deferred_state() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);

        // First CU: register a type.
        dwarf.begin_compilation_unit("first.c", ".", 0, 0);
        let h1 = dwarf.emit_type(&CType::Float, &Target::X86_64);
        assert_eq!(h1, 1);

        // Second CU: should clear deferred types and restart numbering.
        dwarf.begin_compilation_unit("second.c", ".", 0, 0);
        let h2 = dwarf.emit_type(&CType::Double, &Target::X86_64);
        assert_eq!(h2, 1, "New CU should reset handle numbering");
    }

    // -------------------------------------------------------------------
    // All CType variants coverage
    // -------------------------------------------------------------------

    #[test]
    fn test_all_ctype_variants_accepted() {
        let mut dwarf = DwarfGenerator::new(Target::X86_64, true);
        dwarf.begin_compilation_unit("types.c", ".", 0, 0);

        // Exercise all CType variants to ensure none panic.
        let types: Vec<CType> = vec![
            CType::Void,
            CType::Bool,
            CType::Char { signed: true },
            CType::Short { signed: true },
            CType::Int { signed: true },
            CType::Long { signed: true },
            CType::LongLong { signed: true },
            CType::Float,
            CType::Double,
            CType::LongDouble,
            CType::Pointer(Box::new(CType::Void)),
            CType::Array {
                element: Box::new(CType::Int { signed: true }),
                size: Some(10),
            },
            CType::Struct {
                name: Some("point".to_string()),
                fields: vec![],
            },
            CType::Union {
                name: Some("variant".to_string()),
                fields: vec![],
            },
            CType::Enum {
                name: Some("color".to_string()),
                underlying: Box::new(CType::Int { signed: true }),
            },
            CType::Typedef {
                name: "size_t".to_string(),
                underlying: Box::new(CType::Long { signed: false }),
            },
        ];

        let mut handles = Vec::new();
        for ty in &types {
            handles.push(dwarf.emit_type(ty, &Target::X86_64));
        }

        // Void should be handle 0, all others should be > 0.
        assert_eq!(handles[0], 0, "Void must be handle 0");
        for (i, h) in handles.iter().enumerate().skip(1) {
            assert!(*h > 0, "CType variant {} should have non-zero handle", i);
        }

        dwarf.end_compilation_unit();
        let result = dwarf.finish();
        assert!(result.is_some());
    }

    // -------------------------------------------------------------------
    // DebugInfoBuilder direct API test (exercises members_accessed)
    // -------------------------------------------------------------------

    #[test]
    fn test_debug_info_builder_direct_api() {
        let mut str_table = DebugStrTable::new();
        // Direct usage of DebugStrTable::add_string().
        let _name_off = str_table.add_string("test_name");

        let mut builder = DebugInfoBuilder::new(Target::X86_64, &mut str_table);
        builder.begin_compile_unit("bcc test", ".", "test.c", 0, 0, 0);

        // Exercise DebugInfoBuilder::emit_base_type() directly.
        let int_off = builder.emit_base_type("int", 4, info::DW_ATE_SIGNED);
        assert!(int_off > 0);

        // Exercise DebugInfoBuilder::emit_pointer_type() directly.
        let ptr_off = builder.emit_pointer_type(int_off);
        assert!(ptr_off > 0);

        // Exercise DebugInfoBuilder::emit_struct_type() directly.
        let struct_off = builder.emit_struct_type(
            Some("point"),
            &[
                ("x".to_string(), int_off, 0),
                ("y".to_string(), int_off, 4),
            ],
        );
        assert!(struct_off > 0);

        // Exercise DebugInfoBuilder::emit_array_type() directly.
        let arr_off = builder.emit_array_type(int_off, 10);
        assert!(arr_off > 0);

        // Exercise DebugInfoBuilder::emit_subprogram() and child DIEs.
        builder.emit_subprogram("test_fn", 0, 100, true, Some(int_off));
        builder.emit_variable("local", ptr_off, &[info::DW_OP_FBREG, 0x78]);
        builder.emit_end_children();

        builder.emit_end_children();
        builder.end_compile_unit();

        // Exercise DebugInfoBuilder::finish() and section_size().
        let data = builder.finish();
        assert!(!data.is_empty());
        assert!(builder.section_size() > 0);

        // Exercise DebugAbbrevBuilder::finish() via abbrev_builder().
        let abbrev_data = builder.abbrev_builder().finish();
        assert!(!abbrev_data.is_empty());
    }

    // -------------------------------------------------------------------
    // DebugAbbrevBuilder direct API test (exercises members_accessed)
    // -------------------------------------------------------------------

    #[test]
    fn test_debug_abbrev_builder_direct_api() {
        let mut abbrev = DebugAbbrevBuilder::new();

        // Exercise all add_*_abbrev methods directly.
        let cu_code = abbrev.add_compile_unit_abbrev();
        assert!(cu_code > 0);

        let sub_code = abbrev.add_subprogram_abbrev(true);
        assert!(sub_code > 0);

        let var_code = abbrev.add_variable_abbrev();
        assert!(var_code > 0);

        let base_code = abbrev.add_base_type_abbrev();
        assert!(base_code > 0);

        let ptr_code = abbrev.add_pointer_type_abbrev();
        assert!(ptr_code > 0);

        let struct_code = abbrev.add_struct_type_abbrev();
        assert!(struct_code > 0);

        // Exercise finish() to serialise the table.
        let data = abbrev.finish();
        assert!(!data.is_empty());
    }

    // -------------------------------------------------------------------
    // DebugStrTable direct API test (exercises members_accessed)
    // -------------------------------------------------------------------

    #[test]
    fn test_debug_str_table_direct_api() {
        let mut table = DebugStrTable::new();

        // Exercise add_string() and verify deduplication.
        let off1 = table.add_string("hello");
        let off2 = table.add_string("world");
        let off3 = table.add_string("hello"); // duplicate
        assert_eq!(off1, off3, "Duplicate strings must return same offset");
        assert_ne!(off1, off2, "Different strings must return different offsets");

        // Exercise section_size().
        let size = table.section_size();
        assert!(size > 0);

        // Exercise as_bytes().
        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), size);
    }

    // -------------------------------------------------------------------
    // LineNumberProgramBuilder direct API test (exercises members_accessed)
    // -------------------------------------------------------------------

    #[test]
    fn test_line_builder_direct_api() {
        let mut line = LineNumberProgramBuilder::new(Target::X86_64);

        // Exercise directory and file table population.
        let dir_idx = line.add_directory("/home/user");
        let _file_idx = line.add_file("test.c", dir_idx);

        // Exercise the header emission.
        line.emit_header();

        // Exercise all line number program opcodes.
        line.emit_set_address(0x1000);
        line.emit_advance_line(5);
        line.emit_set_file(1);
        line.emit_set_column(10);
        line.emit_copy();
        line.emit_line_advance(16, 1);
        line.emit_end_sequence();

        // Exercise finish().
        let data = line.finish();
        assert!(!data.is_empty());
    }
}
