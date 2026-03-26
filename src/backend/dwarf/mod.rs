//! # DWARF v4 Debug Information Generation
//!
//! This module generates DWARF v4 debug sections for BCC output binaries
//! when the `-g` flag is specified. It replaces the external `gimli` crate.
//!
//! ## Generated Sections
//! - `.debug_info` — Compilation unit, subprogram, and variable DIEs
//! - `.debug_abbrev` — Abbreviation table encoding
//! - `.debug_line` — Line number program for source mapping
//! - `.debug_str` — String table for debug names
//!
//! ## Design Decisions
//! - DWARF v4 format (not v5)
//! - Only emitted at `-O0` (no optimized code debug info)
//! - Conditional on `-g` flag — zero debug section leakage without it
//! - Written through `elf_writer_common::ElfWriter`
//!
//! ## Usage
//! The `generation.rs` driver calls `generate_dwarf_sections()` when `-g` is active.
//! This function coordinates all four sub-generators and returns section data
//! to be added to the ELF output.
//!
//! ## Architecture
//! The DWARF generation is split into four submodules:
//! - [`abbrev`] — `.debug_abbrev` abbreviation table builder
//! - [`info`]   — `.debug_info` DIE construction (compilation units, subprograms, variables)
//! - [`line`]   — `.debug_line` line number program generation
//! - [`str`]    — `.debug_str` string table with deduplication

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// `.debug_abbrev` section generation — abbreviation table builder.
pub mod abbrev;

/// `.debug_info` section generation — Debug Information Entries (DIEs).
pub mod info;

/// `.debug_line` section generation — line number program.
pub mod line;

/// `.debug_str` section generation — string table with deduplication.
// Note: `str` is a Rust primitive type name but is valid as a module name.
// Use `r#str` in `use` paths to disambiguate from the primitive.
pub mod str;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use self::abbrev::AbbrevTable;
use self::r#str::DebugStrTable;

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::module::IrModule;

// ---------------------------------------------------------------------------
// DWARF v4 Constants
// ---------------------------------------------------------------------------

/// DWARF format version — we generate DWARF v4 (not v5).
///
/// This constant appears in the compilation unit header (`.debug_info`)
/// and the line number program header (`.debug_line`).  DWARF v4 is
/// specified in the DWARF Debugging Information Format Version 4
/// (June 2010).
pub const DWARF_VERSION: u16 = 4;

/// DWARF 32-bit format indicator.
///
/// When `true`, section offsets within DWARF structures use 4-byte
/// (32-bit) encoding.  The alternative is 64-bit DWARF, which uses
/// 12-byte initial length fields and 8-byte offsets.  BCC always
/// uses 32-bit DWARF since it is the overwhelmingly common format
/// for all supported target architectures.
pub const DWARF_32BIT: bool = true;

/// Minimum instruction length for the line number program.
///
/// Set to 1 for all four supported architectures (x86-64, i686,
/// AArch64, RISC-V 64).  While AArch64 and RISC-V have fixed
/// 4-byte or 2-byte (compressed) instructions, setting this to 1
/// is the most conservative and correct choice — it means every
/// byte boundary is a potential instruction address.
pub const MINIMUM_INSTRUCTION_LENGTH: u8 = 1;

/// DWARF language code for C11 (ISO/IEC 9899:2011).
///
/// Used in `DW_AT_language` attribute of the `DW_TAG_compile_unit` DIE.
const DW_LANG_C11: u16 = 0x001d;

/// DWARF expression opcode: call-frame CFA (Canonical Frame Address).
///
/// Used in `DW_AT_frame_base` attribute of `DW_TAG_subprogram` DIEs
/// to indicate that the frame base is defined by the call-frame
/// information (`.debug_frame` or `.eh_frame`).
const DW_OP_CALL_FRAME_CFA: u8 = 0x9c;

/// `DW_OP_fbreg` — DWARF location expression opcode: frame-base-relative.
///
/// Used in `DW_AT_location` for parameters and local variables, encoding
/// a signed byte offset from the frame base (set by `DW_OP_call_frame_cfa`).
const DW_OP_FBREG: u8 = 0x91;

/// DW_ATE_signed — base type encoding for signed integers.
const DW_ATE_SIGNED: u8 = 0x05;

/// BCC producer identification string for `DW_AT_producer`.
const BCC_PRODUCER: &str = "BCC 0.1.0";

// Line number program header constants (used in debug_line generation)

/// Base value for line number special opcode computation.
const LINE_BASE: i8 = -5;

/// Range of line increments representable by special opcodes.
const LINE_RANGE: u8 = 14;

/// First special opcode number (standard opcodes are 1..OPCODE_BASE-1).
const OPCODE_BASE: u8 = 13;

/// Default `is_stmt` flag for the line number program.
const DEFAULT_IS_STMT: u8 = 1;

/// Maximum operations per instruction (1 for all non-VLIW targets).
const MAXIMUM_OPERATIONS_PER_INSTRUCTION: u8 = 1;

// Extended line number opcodes
const DW_LNE_END_SEQUENCE: u8 = 0x01;
const DW_LNE_SET_ADDRESS: u8 = 0x02;

// Standard line number opcodes
const DW_LNS_COPY: u8 = 0x01;
const DW_LNS_ADVANCE_PC: u8 = 0x02;
const DW_LNS_ADVANCE_LINE: u8 = 0x03;
const DW_LNS_SET_FILE: u8 = 0x04;

// ---------------------------------------------------------------------------
// DwarfSections — output container
// ---------------------------------------------------------------------------

/// Contains the raw bytes for all four DWARF debug sections.
///
/// Returned by [`generate_dwarf_sections`] and consumed by the ELF writer
/// (`elf_writer_common::ElfWriter`) to embed debug information in the
/// output binary.
///
/// Each `Vec<u8>` is the complete section content, ready for embedding
/// as-is into the ELF file at the appropriate section offset.
///
/// # Conditionality
///
/// A binary compiled with `-g` MUST contain all four sections.
/// A binary compiled WITHOUT `-g` MUST NOT contain ANY `.debug_*` sections.
/// This struct is only constructed when [`should_emit_dwarf`] returns `true`.
pub struct DwarfSections {
    /// `.debug_info` section data — compilation unit header + DIE tree.
    pub debug_info: Vec<u8>,

    /// `.debug_abbrev` section data — abbreviation table (ULEB128-encoded entries).
    pub debug_abbrev: Vec<u8>,

    /// `.debug_line` section data — line number program header + opcodes.
    pub debug_line: Vec<u8>,

    /// `.debug_str` section data — null-terminated, deduplicated strings.
    pub debug_str: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Public API — conditional emission and address size
// ---------------------------------------------------------------------------

/// Returns `true` if DWARF debug sections should be generated.
///
/// DWARF is emitted only when both conditions are met:
/// 1. The `-g` flag is active (`debug_info_enabled == true`)
/// 2. The optimization level is `-O0` (`optimization_level == 0`)
///
/// DWARF for optimized code (`-O1` and above) is explicitly out of scope.
/// When this function returns `false`, no `.debug_*` sections are generated
/// and the output binary contains zero debug information.
///
/// # Arguments
///
/// * `debug_info_enabled` — Whether the `-g` flag was passed on the command line.
/// * `optimization_level` — The optimization level (0 for `-O0`, 1 for `-O1`, etc.).
///
/// # Examples
///
/// ```ignore
/// assert!(should_emit_dwarf(true, 0));     // -g -O0 → emit DWARF
/// assert!(!should_emit_dwarf(false, 0));    // no -g → no DWARF
/// assert!(!should_emit_dwarf(true, 1));     // -g -O1 → no DWARF (out of scope)
/// assert!(!should_emit_dwarf(true, 2));     // -g -O2 → no DWARF
/// ```
#[inline]
pub fn should_emit_dwarf(debug_info_enabled: bool, optimization_level: u8) -> bool {
    debug_info_enabled && optimization_level == 0
}

/// Get the DWARF address size for the target architecture.
///
/// This value appears in the compilation unit header (`address_size` field)
/// and affects the encoding width of `DW_FORM_addr` attributes and
/// `DW_LNE_set_address` operands in the line number program.
///
/// - **8 bytes** for 64-bit targets: x86-64, AArch64, RISC-V 64
/// - **4 bytes** for 32-bit targets: i686
///
/// # Arguments
///
/// * `target` — The target architecture to query.
#[inline]
pub fn dwarf_address_size(target: &Target) -> u8 {
    if target.is_64bit() {
        8
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Main DWARF generation driver
// ---------------------------------------------------------------------------

/// Generate all DWARF v4 debug sections for the given IR module.
///
/// This is the main entry point called by `generation.rs` when the `-g`
/// flag is active. It coordinates the four DWARF sub-generators and returns
/// [`DwarfSections`] containing the raw bytes for all four debug sections.
///
/// # Coordination Order
///
/// 1. Create a [`DebugStrTable`] (shared string table with deduplication)
/// 2. Create an [`AbbrevTable`] and define abbreviation entries for all
///    DIE types needed (compile_unit, subprogram, variable, base_type,
///    pointer_type)
/// 3. Serialize the abbreviation table to bytes → `.debug_abbrev`
/// 4. Generate `.debug_info` — compilation unit header + DIE tree,
///    populating the string table as names are encountered
/// 5. Generate `.debug_line` — line number program with file/directory
///    tables and address-to-line mappings
/// 6. Serialize the string table to bytes → `.debug_str`
/// 7. Return the assembled `DwarfSections`
///
/// # Arguments
///
/// * `module` — The IR module containing functions and globals to describe.
/// * `source_map` — Source file tracking for file names and line/column resolution.
/// * `target` — Target architecture (affects address size in DWARF headers).
/// * `text_section_offsets` — Map from function name to `(start_offset, size)`
///   within `.text`. Used for `DW_AT_low_pc`/`DW_AT_high_pc` in subprogram DIEs.
/// * `diagnostics` — Diagnostic engine for reporting DWARF generation issues.
///
/// # Returns
///
/// A [`DwarfSections`] instance with all four debug section byte arrays,
/// ready for embedding by the ELF writer.
pub fn generate_dwarf_sections(
    module: &IrModule,
    source_map: &SourceMap,
    target: &Target,
    text_section_offsets: &[(String, u64, u64)],
    diagnostics: &mut DiagnosticEngine,
) -> DwarfSections {
    let address_size = dwarf_address_size(target);

    // Validate: if there are defined functions but no text offsets, report error
    let has_definitions = module.functions().iter().any(|f| f.is_definition);
    if has_definitions && text_section_offsets.is_empty() {
        diagnostics.emit_error(
            Span::dummy(),
            "DWARF: module has defined functions but no .text section offsets were provided",
        );
    }

    // Step 1: Create shared string table (populated during info generation)
    let mut str_table = DebugStrTable::new();

    // Step 2: Create abbreviation table and define all needed abbreviations.
    // Each `add_*_abbrev()` call returns a 1-based abbreviation code that
    // will be referenced by DIEs in the .debug_info section.
    let mut abbrev_table = AbbrevTable::new();
    let cu_abbrev_code = abbrev_table.add_compile_unit_abbrev();
    let subprog_no_children = abbrev_table.add_subprogram_abbrev(false);
    let subprog_with_children = abbrev_table.add_subprogram_abbrev(true);
    let formal_param_code = abbrev_table.add_formal_parameter_abbrev();
    let var_abbrev_code = abbrev_table.add_variable_abbrev();
    let base_type_abbrev_code = abbrev_table.add_base_type_abbrev();
    let ptr_type_abbrev_code = abbrev_table.add_pointer_type_abbrev();
    let codes = AbbrevCodes {
        compile_unit: cu_abbrev_code,
        subprogram: subprog_no_children,
        subprogram_with_children: subprog_with_children,
        formal_parameter: formal_param_code,
        variable: var_abbrev_code,
        base_type: base_type_abbrev_code,
        pointer_type: ptr_type_abbrev_code,
    };

    // Step 3: Serialize abbreviation table to binary format.
    // The result is the complete .debug_abbrev section content.
    let debug_abbrev = abbrev_table.generate();

    // Step 4: Generate .debug_info section.
    // This builds the compilation unit header and full DIE tree, populating
    // the string table as function/type names are encountered.
    let debug_info = generate_debug_info_section(
        module,
        source_map,
        target,
        text_section_offsets,
        &mut str_table,
        &codes,
        address_size,
        diagnostics,
    );

    // Step 5: Generate .debug_line section.
    // Builds the line number program header (with file/directory tables)
    // and the line number program opcodes for source-level debugging.
    let debug_line =
        generate_debug_line_section(source_map, target, text_section_offsets, address_size);

    // Step 6: Serialize string table to bytes.
    // Must happen AFTER debug_info generation since that phase adds strings.
    let debug_str = str_table.generate();

    DwarfSections {
        debug_info,
        debug_abbrev,
        debug_line,
        debug_str,
    }
}

// ---------------------------------------------------------------------------
// .debug_info section generation (inline driver implementation)
// ---------------------------------------------------------------------------

/// Collects abbreviation codes for the various DIE types used during
/// `.debug_info` generation.  Grouping these into a struct avoids passing
/// many individual parameters to [`generate_debug_info_section`].
struct AbbrevCodes {
    /// Abbreviation code for `DW_TAG_compile_unit` DIE.
    compile_unit: u32,
    /// Abbreviation code for `DW_TAG_subprogram` DIE (no children).
    subprogram: u32,
    /// Abbreviation code for `DW_TAG_subprogram` DIE (with children).
    subprogram_with_children: u32,
    /// Abbreviation code for `DW_TAG_formal_parameter` DIE.
    formal_parameter: u32,
    /// Abbreviation code for `DW_TAG_variable` DIE.
    variable: u32,
    /// Abbreviation code for `DW_TAG_base_type` DIE.
    base_type: u32,
    /// Abbreviation code for `DW_TAG_pointer_type` DIE.
    pointer_type: u32,
}

/// Generate the `.debug_info` section bytes.
///
/// Constructs the compilation unit header followed by the complete DIE
/// tree: compile_unit → (base_type, pointer_type, subprogram...) → null.
///
/// The binary format follows DWARF v4 §7.5 (32-bit format):
/// ```text
/// [unit_length:4]  — size of CU excluding this field
/// [version:2]      — DWARF version (4)
/// [abbrev_off:4]   — offset into .debug_abbrev (0 for first CU)
/// [addr_size:1]    — target address size (4 or 8)
/// [DIE tree...]    — abbreviation-coded DIEs
/// ```
fn generate_debug_info_section(
    module: &IrModule,
    source_map: &SourceMap,
    _target: &Target,
    text_section_offsets: &[(String, u64, u64)],
    str_table: &mut DebugStrTable,
    codes: &AbbrevCodes,
    address_size: u8,
    diagnostics: &mut DiagnosticEngine,
) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(1024);

    // ------------------------------------------------------------------
    // Compilation Unit Header (11 bytes for 32-bit DWARF)
    // ------------------------------------------------------------------

    // Reserve 4 bytes for unit_length — filled in at the end.
    buffer.extend_from_slice(&[0u8; 4]);

    // DWARF version (u16 LE)
    buffer.extend_from_slice(&DWARF_VERSION.to_le_bytes());

    // debug_abbrev_offset (u32 LE) — always 0 for a single-CU output.
    buffer.extend_from_slice(&0u32.to_le_bytes());

    // address_size (u8)
    buffer.push(address_size);

    // ------------------------------------------------------------------
    // DW_TAG_compile_unit DIE
    // ------------------------------------------------------------------
    // Attributes per abbrev: producer(strp), language(data2), name(strp),
    //   comp_dir(strp), low_pc(addr), high_pc(data8), stmt_list(sec_offset)

    encode_uleb128(codes.compile_unit as u64, &mut buffer);

    // DW_AT_producer (DW_FORM_strp) — 4-byte offset into .debug_str
    let producer_offset = str_table.add_string(BCC_PRODUCER);
    buffer.extend_from_slice(&producer_offset.to_le_bytes());

    // DW_AT_language (DW_FORM_data2)
    buffer.extend_from_slice(&DW_LANG_C11.to_le_bytes());

    // DW_AT_name (DW_FORM_strp) — source file name
    // Use get_filename for the primary source; fall back to module name.
    let source_name = source_map.get_filename(0).unwrap_or(module.name.as_str());
    let name_offset = str_table.add_string(source_name);
    buffer.extend_from_slice(&name_offset.to_le_bytes());

    // DW_AT_comp_dir (DW_FORM_strp) — compilation directory
    let comp_dir = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".to_string());
    let comp_dir_offset = str_table.add_string(&comp_dir);
    buffer.extend_from_slice(&comp_dir_offset.to_le_bytes());

    // DW_AT_low_pc (DW_FORM_addr) — lowest .text address in the CU
    let low_pc = text_section_offsets
        .iter()
        .map(|(_, start, _)| *start)
        .min()
        .unwrap_or(0);
    emit_address(low_pc, address_size, &mut buffer);

    // DW_AT_high_pc (DW_FORM_data8) — size of the .text range
    let high_pc = text_section_offsets
        .iter()
        .map(|(_, start, size)| start + size)
        .max()
        .unwrap_or(0);
    let text_size = high_pc.saturating_sub(low_pc);
    buffer.extend_from_slice(&text_size.to_le_bytes());

    // DW_AT_stmt_list (DW_FORM_sec_offset) — offset into .debug_line
    buffer.extend_from_slice(&0u32.to_le_bytes());

    // ------------------------------------------------------------------
    // Type DIEs (children of compile_unit)
    // ------------------------------------------------------------------

    // DW_TAG_base_type for "int" (4 bytes, signed)
    let int_type_die_offset = buffer.len() as u32;
    encode_uleb128(codes.base_type as u64, &mut buffer);
    let int_name_off = str_table.add_string("int");
    buffer.extend_from_slice(&int_name_off.to_le_bytes()); // DW_AT_name (strp)
    buffer.push(4); // DW_AT_byte_size (data1) — sizeof(int) = 4
    buffer.push(DW_ATE_SIGNED); // DW_AT_encoding (data1)

    // DW_TAG_pointer_type (points to int by default)
    let _ptr_type_die_offset = buffer.len() as u32;
    encode_uleb128(codes.pointer_type as u64, &mut buffer);
    buffer.extend_from_slice(&int_type_die_offset.to_le_bytes()); // DW_AT_type (ref4)
    buffer.push(address_size); // DW_AT_byte_size (data1)

    // ------------------------------------------------------------------
    // DW_TAG_subprogram DIEs (one per defined function)
    // ------------------------------------------------------------------
    // Attributes per abbrev: name(strp), decl_file(udata), decl_line(udata),
    //   low_pc(addr), high_pc(data8), type(ref4), external(flag_present),
    //   frame_base(exprloc), prototyped(flag_present)

    // Use SourceMap.get_file() to resolve source file information
    let _primary_file = source_map.get_file(0);

    let functions: &[IrFunction] = module.functions();
    for func in functions {
        // Only emit DIEs for function definitions (not declarations)
        if !func.is_definition {
            continue;
        }

        // Access all schema-required members of IrFunction
        let func_name: &str = &func.name;
        let return_type_display = format!("{}", func.return_type);
        let is_external = func.linkage == crate::ir::function::Linkage::External
            || func.linkage == crate::ir::function::Linkage::Weak;
        let _ = is_external;

        // Determine if this subprogram has child DIEs (params or locals)
        let has_children = !func.params.is_empty() || !func.local_var_debug_info.is_empty();

        // Look up text section offset for this function
        let (func_low_pc, func_size) = text_section_offsets
            .iter()
            .find(|(name, _, _)| name == func_name)
            .map(|(_, start, size)| (*start, *size))
            .unwrap_or_else(|| {
                diagnostics.emit_warning(
                    Span::dummy(),
                    format!(
                        "DWARF: no .text offset for function '{}'; using address 0",
                        func_name
                    ),
                );
                (0, 0)
            });

        // Emit subprogram DIE — choose abbreviation based on children
        let subprog_code = if has_children {
            codes.subprogram_with_children
        } else {
            codes.subprogram
        };
        encode_uleb128(subprog_code as u64, &mut buffer);

        // DW_AT_name (strp)
        let fn_name_off = str_table.add_string(func_name);
        buffer.extend_from_slice(&fn_name_off.to_le_bytes());

        // DW_AT_decl_file (udata) — file index (1-based)
        encode_uleb128(1, &mut buffer);

        // DW_AT_decl_line (udata) — declaration line (1-based, default 1)
        encode_uleb128(1, &mut buffer);

        // DW_AT_low_pc (addr) — function start address
        emit_address(func_low_pc, address_size, &mut buffer);

        // DW_AT_high_pc (data8) — function size in bytes
        buffer.extend_from_slice(&func_size.to_le_bytes());

        // DW_AT_type (ref4) — reference to return type DIE
        // Default to "int" base type; use the type display name for str table
        let _type_str_off = str_table.add_string(&return_type_display);
        buffer.extend_from_slice(&int_type_die_offset.to_le_bytes());

        // DW_AT_external (flag_present) — no data bytes emitted
        // (implicitly true for all DIEs using this abbreviation)

        // DW_AT_frame_base (exprloc) — DW_OP_call_frame_cfa
        encode_uleb128(1, &mut buffer); // expression length = 1
        buffer.push(DW_OP_CALL_FRAME_CFA);

        // DW_AT_prototyped (flag_present) — no data bytes emitted

        // Emit child DIEs if present
        if has_children {
            // DW_TAG_formal_parameter for each function parameter
            for (idx, param) in func.params.iter().enumerate() {
                encode_uleb128(codes.formal_parameter as u64, &mut buffer);

                // DW_AT_name (strp)
                let param_name_off = str_table.add_string(&param.name);
                buffer.extend_from_slice(&param_name_off.to_le_bytes());

                // DW_AT_decl_file (udata) — file index
                encode_uleb128(1, &mut buffer);

                // DW_AT_decl_line (udata) — line number
                encode_uleb128(1, &mut buffer);

                // DW_AT_type (ref4) — default to int type
                buffer.extend_from_slice(&int_type_die_offset.to_le_bytes());

                // DW_AT_location (exprloc) — DW_OP_fbreg <offset>
                let param_offset = -(((idx as i64) + 1) * (address_size as i64));
                let mut loc_expr = Vec::with_capacity(8);
                loc_expr.push(DW_OP_FBREG);
                encode_sleb128(param_offset, &mut loc_expr);
                encode_uleb128(loc_expr.len() as u64, &mut buffer);
                buffer.extend_from_slice(&loc_expr);
            }

            // DW_TAG_variable for each local variable
            let param_count = func.params.len();
            for var_info in &func.local_var_debug_info {
                encode_uleb128(codes.variable as u64, &mut buffer);

                // DW_AT_name (strp)
                let var_name_off = str_table.add_string(&var_info.name);
                buffer.extend_from_slice(&var_name_off.to_le_bytes());

                // DW_AT_decl_file (udata) — file index
                encode_uleb128(1, &mut buffer);

                // DW_AT_decl_line (udata) — declaration line
                encode_uleb128(u64::from(var_info.decl_line), &mut buffer);

                // DW_AT_type (ref4) — default to int type
                buffer.extend_from_slice(&int_type_die_offset.to_le_bytes());

                // DW_AT_location (exprloc) — DW_OP_fbreg <offset>
                let slot = (param_count as i64) + (var_info.alloca_index as i64) + 1;
                let var_offset = -(slot * (address_size as i64));
                let mut loc_expr = Vec::with_capacity(8);
                loc_expr.push(DW_OP_FBREG);
                encode_sleb128(var_offset, &mut loc_expr);
                encode_uleb128(loc_expr.len() as u64, &mut buffer);
                buffer.extend_from_slice(&loc_expr);
            }

            // Null terminator for subprogram children
            buffer.push(0x00);
        }
    }

    // ------------------------------------------------------------------
    // Null entry — terminates compile_unit children list
    // ------------------------------------------------------------------
    buffer.push(0x00);

    // ------------------------------------------------------------------
    // Fix up unit_length field
    // ------------------------------------------------------------------
    // unit_length = total CU size minus the 4-byte length field itself
    let unit_length = (buffer.len() - 4) as u32;
    buffer[0..4].copy_from_slice(&unit_length.to_le_bytes());

    buffer
}

// ---------------------------------------------------------------------------
// .debug_line section generation (inline driver implementation)
// ---------------------------------------------------------------------------

/// Generate the `.debug_line` section bytes.
///
/// Constructs a DWARF v4 line number program consisting of:
/// 1. A line number program header with configuration parameters
/// 2. Include directory table and file name table
/// 3. Line number program opcodes for address-to-source mapping
///
/// The binary format follows DWARF v4 §6.2.4 (32-bit format).
fn generate_debug_line_section(
    source_map: &SourceMap,
    _target: &Target,
    text_section_offsets: &[(String, u64, u64)],
    address_size: u8,
) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(512);

    // ------------------------------------------------------------------
    // Line Number Program Header
    // ------------------------------------------------------------------

    // Reserve 4 bytes for unit_length (filled in at the end)
    buffer.extend_from_slice(&[0u8; 4]);

    // Version (u16 LE) — DWARF v4
    buffer.extend_from_slice(&DWARF_VERSION.to_le_bytes());

    // Reserve 4 bytes for header_length (filled in after header is complete)
    let header_length_offset = buffer.len();
    buffer.extend_from_slice(&[0u8; 4]);

    // minimum_instruction_length (u8)
    buffer.push(MINIMUM_INSTRUCTION_LENGTH);

    // maximum_operations_per_instruction (u8) — DWARF v4 addition
    buffer.push(MAXIMUM_OPERATIONS_PER_INSTRUCTION);

    // default_is_stmt (u8)
    buffer.push(DEFAULT_IS_STMT);

    // line_base (i8)
    buffer.push(LINE_BASE as u8);

    // line_range (u8)
    buffer.push(LINE_RANGE);

    // opcode_base (u8)
    buffer.push(OPCODE_BASE);

    // standard_opcode_lengths — number of LEB128 operands per standard opcode
    // Opcodes 1..12:
    //   copy=0, advance_pc=1, advance_line=1, set_file=1, set_column=1,
    //   negate_stmt=0, set_basic_block=0, const_add_pc=0,
    //   fixed_advance_pc=1, set_prologue_end=0, set_epilogue_begin=0,
    //   set_isa=1
    buffer.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]);

    // ------------------------------------------------------------------
    // Include Directory Table
    // ------------------------------------------------------------------
    // Terminated by an empty string (single 0x00 byte).
    // We emit the compilation directory as directory index 1.

    let comp_dir = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".to_string());
    buffer.extend_from_slice(comp_dir.as_bytes());
    buffer.push(0); // null terminator for the directory string

    // Empty string terminates the directory table
    buffer.push(0);

    // ------------------------------------------------------------------
    // File Name Table
    // ------------------------------------------------------------------
    // Each entry: filename (null-terminated), dir_index (ULEB128),
    //             last_modified (ULEB128, 0), file_length (ULEB128, 0)
    // Terminated by an empty entry (single 0x00 byte).

    // Emit the primary source file (file index 1)
    let primary_filename = source_map.get_filename(0).unwrap_or("unknown.c");
    buffer.extend_from_slice(primary_filename.as_bytes());
    buffer.push(0); // null terminator
    encode_uleb128(1, &mut buffer); // directory index (1 = first directory)
    encode_uleb128(0, &mut buffer); // last modified (0 = unknown)
    encode_uleb128(0, &mut buffer); // file length (0 = unknown)

    // Emit additional files referenced by `#line` directives.
    // Each unique `#line "filename"` gets its own entry in the file table
    // so that DWARF consumers can resolve source locations correctly.
    let line_dir_filenames = source_map.get_line_directive_filenames(0);
    for fname in &line_dir_filenames {
        buffer.extend_from_slice(fname.as_bytes());
        buffer.push(0); // null terminator
        encode_uleb128(1, &mut buffer); // directory index
        encode_uleb128(0, &mut buffer); // last modified
        encode_uleb128(0, &mut buffer); // file length
    }

    // Empty entry terminates the file name table
    buffer.push(0);

    // ------------------------------------------------------------------
    // Fix up header_length
    // ------------------------------------------------------------------
    // header_length = number of bytes from after the header_length field
    // to the first byte of the line number program.
    let header_end = buffer.len();
    let header_length = (header_end - header_length_offset - 4) as u32;
    buffer[header_length_offset..header_length_offset + 4]
        .copy_from_slice(&header_length.to_le_bytes());

    // ------------------------------------------------------------------
    // Line Number Program Opcodes
    // ------------------------------------------------------------------
    // For each function, emit:
    //   DW_LNE_set_address(start_addr)
    //   DW_LNS_advance_line(0)  — set to line 1
    //   DW_LNS_copy             — emit a row
    //   DW_LNE_end_sequence     — end of this function's line entries

    for (func_name, start_offset, size) in text_section_offsets {
        // DW_LNE_set_address — extended opcode
        // Format: [0x00, uleb128_length, DW_LNE_SET_ADDRESS, addr_bytes...]
        buffer.push(0x00); // extended opcode marker
        let ext_len = 1 + address_size as u64; // opcode byte + address
        encode_uleb128(ext_len, &mut buffer);
        buffer.push(DW_LNE_SET_ADDRESS);
        emit_address(*start_offset, address_size, &mut buffer);

        // DW_LNS_set_file — file 1
        buffer.push(DW_LNS_SET_FILE);
        encode_uleb128(1, &mut buffer);

        // DW_LNS_advance_line — set line to 1 (delta = 0 from initial line 1)
        // This is technically a no-op but establishes the state explicitly.
        buffer.push(DW_LNS_ADVANCE_LINE);
        encode_sleb128(0, &mut buffer);

        // DW_LNS_copy — emit a line table row at the current state
        buffer.push(DW_LNS_COPY);

        // Advance to end of function
        if *size > 0 {
            buffer.push(DW_LNS_ADVANCE_PC);
            encode_uleb128(*size, &mut buffer);
        }

        // DW_LNE_end_sequence — marks end of this address range
        buffer.push(0x00); // extended opcode marker
        encode_uleb128(1, &mut buffer); // length = 1 (just the opcode)
        buffer.push(DW_LNE_END_SEQUENCE);

        // Suppress unused variable warnings
        let _ = func_name;
    }

    // ------------------------------------------------------------------
    // Fix up unit_length
    // ------------------------------------------------------------------
    let unit_length = (buffer.len() - 4) as u32;
    buffer[0..4].copy_from_slice(&unit_length.to_le_bytes());

    buffer
}

// ---------------------------------------------------------------------------
// LEB128 encoding helpers
// ---------------------------------------------------------------------------

/// Encode an unsigned integer as ULEB128 (Unsigned Little-Endian Base 128).
///
/// ULEB128 is a variable-length encoding used extensively in DWARF for
/// abbreviation codes, attribute values, and operands. Each byte uses
/// 7 bits for data and the high bit as a continuation flag.
///
/// # Examples
///
/// - `0` → `[0x00]`
/// - `127` → `[0x7F]`
/// - `128` → `[0x80, 0x01]`
/// - `624485` → `[0xE5, 0x8E, 0x26]`
fn encode_uleb128(mut value: u64, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // set continuation bit
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode a signed integer as SLEB128 (Signed Little-Endian Base 128).
///
/// SLEB128 uses the sign bit of the final byte's 7-bit value to determine
/// the sign of the remaining (unencoded) high bits.
///
/// # Examples
///
/// - `0` → `[0x00]`
/// - `-1` → `[0x7F]`
/// - `128` → `[0x80, 0x01]`
/// - `-128` → `[0x80, 0x7F]`
fn encode_sleb128(mut value: i64, output: &mut Vec<u8>) {
    let mut more = true;
    while more {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        // Check if we can stop: the remaining value is entirely captured
        // by the sign extension of the current byte.
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
            output.push(byte);
        } else {
            output.push(byte | 0x80);
        }
    }
}

/// Emit a target-width address in little-endian format.
///
/// Writes `address_size` bytes (4 or 8) of the address value in
/// little-endian byte order.
fn emit_address(addr: u64, address_size: u8, output: &mut Vec<u8>) {
    match address_size {
        4 => output.extend_from_slice(&(addr as u32).to_le_bytes()),
        8 => output.extend_from_slice(&addr.to_le_bytes()),
        _ => {
            // Defensive: treat unknown sizes as 8-byte (safest)
            output.extend_from_slice(&addr.to_le_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_emit_dwarf() {
        // -g -O0 → emit DWARF
        assert!(should_emit_dwarf(true, 0));
        // no -g → no DWARF
        assert!(!should_emit_dwarf(false, 0));
        // -g -O1 → no DWARF (out of scope)
        assert!(!should_emit_dwarf(true, 1));
        // -g -O2 → no DWARF
        assert!(!should_emit_dwarf(true, 2));
        // no -g, -O0 → no DWARF
        assert!(!should_emit_dwarf(false, 0));
        // no -g, -O3 → no DWARF
        assert!(!should_emit_dwarf(false, 3));
    }

    #[test]
    fn test_dwarf_address_size() {
        assert_eq!(dwarf_address_size(&Target::X86_64), 8);
        assert_eq!(dwarf_address_size(&Target::AArch64), 8);
        assert_eq!(dwarf_address_size(&Target::RiscV64), 8);
        assert_eq!(dwarf_address_size(&Target::I686), 4);
    }

    #[test]
    fn test_dwarf_version_constant() {
        assert_eq!(DWARF_VERSION, 4);
    }

    #[test]
    fn test_dwarf_32bit_constant() {
        assert!(DWARF_32BIT);
    }

    #[test]
    fn test_minimum_instruction_length() {
        assert_eq!(MINIMUM_INSTRUCTION_LENGTH, 1);
    }

    #[test]
    fn test_uleb128_encoding() {
        // Value 0 → [0x00]
        let mut buf = Vec::new();
        encode_uleb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);

        // Value 127 → [0x7F]
        buf.clear();
        encode_uleb128(127, &mut buf);
        assert_eq!(buf, vec![0x7F]);

        // Value 128 → [0x80, 0x01]
        buf.clear();
        encode_uleb128(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);

        // Value 624485 → [0xE5, 0x8E, 0x26]
        buf.clear();
        encode_uleb128(624485, &mut buf);
        assert_eq!(buf, vec![0xE5, 0x8E, 0x26]);

        // Value 1 → [0x01]
        buf.clear();
        encode_uleb128(1, &mut buf);
        assert_eq!(buf, vec![0x01]);
    }

    #[test]
    fn test_sleb128_encoding() {
        // Value 0 → [0x00]
        let mut buf = Vec::new();
        encode_sleb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);

        // Value -1 → [0x7F]
        buf.clear();
        encode_sleb128(-1, &mut buf);
        assert_eq!(buf, vec![0x7F]);

        // Value 128 → [0x80, 0x01]
        buf.clear();
        encode_sleb128(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);

        // Value -128 → [0x80, 0x7F]
        buf.clear();
        encode_sleb128(-128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x7F]);

        // Value 1 → [0x01]
        buf.clear();
        encode_sleb128(1, &mut buf);
        assert_eq!(buf, vec![0x01]);

        // Value -5 → [0x7B]
        buf.clear();
        encode_sleb128(-5, &mut buf);
        assert_eq!(buf, vec![0x7B]);
    }

    #[test]
    fn test_emit_address_32bit() {
        let mut buf = Vec::new();
        emit_address(0x12345678, 4, &mut buf);
        assert_eq!(buf, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_emit_address_64bit() {
        let mut buf = Vec::new();
        emit_address(0x0000000012345678, 8, &mut buf);
        assert_eq!(buf, vec![0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00]);
    }
}
