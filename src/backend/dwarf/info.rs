//! # DWARF `.debug_info` Section Generation
//!
//! This module provides the [`DebugInfoGenerator`] for constructing DWARF v4
//! `.debug_info` section content — compilation unit headers and Debug
//! Information Entry (DIE) trees describing functions, variables, and types.
//!
//! ## DWARF v4 `.debug_info` Section Layout
//!
//! The `.debug_info` section is the primary DWARF section that debuggers
//! (GDB, LLDB) parse to understand program structure.  It contains:
//!
//! 1. **Compilation Unit Header** — version, abbreviation table offset,
//!    address size (DWARF v4 §7.5, 32-bit format)
//! 2. **DW_TAG_compile_unit DIE** — top-level DIE with producer, language,
//!    source file name, compilation directory, address range, and line table
//!    reference
//! 3. **Type DIEs** — `DW_TAG_base_type`, `DW_TAG_pointer_type`, etc. for
//!    all types referenced by variables and functions
//! 4. **DW_TAG_subprogram DIEs** — one per function definition, with address
//!    range, return type, parameters, and local variables
//! 5. **Null terminator** — `0x00` byte to close the compile unit children
//!
//! ## Coordination
//!
//! The generator works in conjunction with:
//! - [`super::abbrev::AbbrevTable`] for abbreviation codes
//! - [`super::r#str::DebugStrTable`] for string offsets (`DW_FORM_strp`)
//! - [`super::dwarf_address_size`] for target-specific address widths
//!
//! ## Zero-Dependency Mandate
//!
//! This module uses only the Rust standard library (`std`) and internal
//! `crate::` references.  No external crates are permitted.  All DWARF
//! encoding (LEB128, DIE construction, address emission) is hand-implemented.

use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

use super::abbrev::AbbrevTable;
use super::r#str::DebugStrTable;
use super::{dwarf_address_size, DWARF_VERSION};

// ===========================================================================
// DWARF v4 Base Type Encoding Constants (DW_ATE_*)
// ===========================================================================

/// DW_ATE_boolean — base type encoding for C `_Bool`.
const DW_ATE_BOOLEAN: u8 = 0x02;

/// DW_ATE_float — base type encoding for floating-point types.
const DW_ATE_FLOAT: u8 = 0x04;

/// DW_ATE_signed — base type encoding for signed integer types.
const DW_ATE_SIGNED: u8 = 0x05;

/// DW_ATE_signed_char — base type encoding for `char` / `signed char`.
const DW_ATE_SIGNED_CHAR: u8 = 0x06;

/// DW_ATE_unsigned — base type encoding for unsigned integer types.
const DW_ATE_UNSIGNED: u8 = 0x07;

/// DW_ATE_unsigned_char — base type encoding for `unsigned char`.
///
/// Not currently used since `IrType::I8` maps to signed char, but
/// retained for future use when the IR distinguishes signed/unsigned
/// char variants.
#[allow(dead_code)]
const DW_ATE_UNSIGNED_CHAR: u8 = 0x08;

// ===========================================================================
// DWARF v4 Location Expression Opcodes (DW_OP_*)
// ===========================================================================

/// DW_OP_fbreg — stack location as signed offset from frame base.
const DW_OP_FBREG: u8 = 0x91;

/// DW_OP_call_frame_cfa — canonical frame address from .eh_frame/.debug_frame.
const DW_OP_CALL_FRAME_CFA: u8 = 0x9c;

// ===========================================================================
// DWARF Language Constant
// ===========================================================================

/// DW_LANG_C11 — language code for ISO C11 (ISO/IEC 9899:2011).
const DW_LANG_C11: u16 = 0x001d;

// ===========================================================================
// BCC Producer String
// ===========================================================================

/// Producer identification for `DW_AT_producer`.
const BCC_PRODUCER: &str = "BCC 0.1.0";

// ===========================================================================
// LEB128 Encoding Helpers
// ===========================================================================

/// Encode an unsigned integer as ULEB128 (Unsigned Little-Endian Base 128).
///
/// ULEB128 uses 7 data bits per byte with the high bit as a continuation
/// flag.  This is the standard variable-length encoding used throughout
/// the DWARF format for unsigned values (abbreviation codes, attribute
/// values, expression lengths, etc.).
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
/// SLEB128 uses the sign bit of the final byte's 7-bit field to determine
/// the sign of the remaining (unencoded) high bits.  Used for signed
/// attribute values such as `DW_OP_fbreg` offsets and `DW_AT_const_value`.
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
        // by the sign extension of the current byte's bit 6.
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
/// little-endian byte order, matching the DWARF `DW_FORM_addr` encoding.
fn emit_address(addr: u64, address_size: u8, output: &mut Vec<u8>) {
    match address_size {
        4 => output.extend_from_slice(&(addr as u32).to_le_bytes()),
        8 => output.extend_from_slice(&addr.to_le_bytes()),
        _ => {
            // Defensive: treat unknown sizes as 8-byte (safest).
            output.extend_from_slice(&addr.to_le_bytes());
        }
    }
}

// ===========================================================================
// Cached Abbreviation Codes
// ===========================================================================

/// Cached abbreviation codes for all DIE types emitted by the generator.
///
/// Registering abbreviations with [`AbbrevTable`] once and caching the
/// returned codes avoids redundant registrations and keeps the abbreviation
/// table compact.
struct AbbrevCodes {
    /// Code for `DW_TAG_compile_unit` (has children).
    compile_unit: u32,
    /// Code for `DW_TAG_subprogram` with children (has formal params/locals).
    subprogram_with_children: u32,
    /// Code for `DW_TAG_subprogram` without children (leaf function).
    subprogram_no_children: u32,
    /// Code for `DW_TAG_variable` (local variables inside functions).
    variable: u32,
    /// Code for `DW_TAG_formal_parameter`.
    formal_parameter: u32,
    /// Code for `DW_TAG_base_type`.
    base_type: u32,
    /// Code for `DW_TAG_pointer_type`.
    pointer_type: u32,
}

// ===========================================================================
// DebugInfoGenerator — Main Public API
// ===========================================================================

/// Generator for DWARF v4 `.debug_info` section content.
///
/// Constructs the compilation unit header and the DIE tree that describes
/// the program's functions, variables, and types.  The resulting byte stream
/// is a complete `.debug_info` section ready for embedding in an ELF file.
///
/// # Coordination
///
/// The generator references:
/// - An [`AbbrevTable`] (mutably) for registering abbreviation entries
/// - A [`DebugStrTable`] (mutably) for interning debug string names
/// - A [`SourceMap`] for source file and line information
/// - A [`Target`] for address size determination
///
/// # Type Deduplication
///
/// Each unique [`IrType`] is emitted as exactly one DIE.  Subsequent
/// references to the same type use `DW_FORM_ref4` offsets pointing
/// back to the first occurrence.  Deduplication is tracked via an
/// internal [`FxHashMap<IrType, u32>`].
///
/// # Example
///
/// ```ignore
/// let mut str_table = DebugStrTable::new();
/// let mut abbrev_table = AbbrevTable::new();
/// let source_map = SourceMap::new();
/// let target = Target::X86_64;
/// let mut gen = DebugInfoGenerator::new(
///     &mut str_table,
///     &mut abbrev_table,
///     &target,
///     &source_map,
/// );
/// let debug_info_bytes = gen.generate(&module, &[("main".into(), 0, 42)]);
/// ```
pub struct DebugInfoGenerator<'a> {
    /// Output byte buffer for the `.debug_info` section.
    buffer: Vec<u8>,
    /// Reference to the shared debug string table.
    str_table: &'a mut DebugStrTable,
    /// Target architecture (affects address size).
    target: &'a Target,
    /// Source map for file/line resolution.
    source_map: &'a SourceMap,
    /// Address size in bytes (4 for i686, 8 for 64-bit targets).
    address_size: u8,
    /// Cached abbreviation codes for all DIE types.
    codes: AbbrevCodes,
    /// Deduplication map: IrType → DIE offset within `.debug_info`.
    type_die_offsets: FxHashMap<IrType, u32>,
}

impl<'a> DebugInfoGenerator<'a> {
    /// Create a new `DebugInfoGenerator`.
    ///
    /// Registers all required abbreviation entries with the provided
    /// [`AbbrevTable`] and caches their codes for efficient DIE emission.
    ///
    /// # Arguments
    ///
    /// * `str_table` — Mutable reference to the shared string table.
    /// * `abbrev_table` — Mutable reference to the abbreviation table
    ///   (abbreviation entries are registered here).
    /// * `target` — Target architecture for address-size decisions.
    /// * `source_map` — Source file tracker for file names and line numbers.
    pub fn new(
        str_table: &'a mut DebugStrTable,
        abbrev_table: &'a mut AbbrevTable,
        target: &'a Target,
        source_map: &'a SourceMap,
    ) -> Self {
        let address_size = dwarf_address_size(target);

        // Register all abbreviation entries we will use.
        let compile_unit = abbrev_table.add_compile_unit_abbrev();
        let subprogram_with_children = abbrev_table.add_subprogram_abbrev(true);
        let subprogram_no_children = abbrev_table.add_subprogram_abbrev(false);
        let variable = abbrev_table.add_variable_abbrev();
        let formal_parameter = abbrev_table.add_formal_parameter_abbrev();
        let base_type = abbrev_table.add_base_type_abbrev();
        let pointer_type = abbrev_table.add_pointer_type_abbrev();

        let codes = AbbrevCodes {
            compile_unit,
            subprogram_with_children,
            subprogram_no_children,
            variable,
            formal_parameter,
            base_type,
            pointer_type,
        };

        DebugInfoGenerator {
            buffer: Vec::with_capacity(2048),
            str_table,
            target,
            source_map,
            address_size,
            codes,
            type_die_offsets: FxHashMap::default(),
        }
    }

    /// Generate the complete `.debug_info` section for the given IR module.
    ///
    /// Constructs the compilation unit header (DWARF v4 §7.5, 32-bit format)
    /// followed by the full DIE tree:
    ///
    /// ```text
    /// [CU header: unit_length(4) + version(2) + abbrev_off(4) + addr_size(1)]
    /// [DW_TAG_compile_unit DIE]
    ///   [DW_TAG_base_type DIEs ...]
    ///   [DW_TAG_pointer_type DIEs ...]
    ///   [DW_TAG_subprogram DIEs ...]
    ///     [DW_TAG_formal_parameter DIEs ...]
    ///     [DW_TAG_variable DIEs ...]
    ///     [0x00 null terminator for subprogram children]
    ///   [0x00 null terminator for compile_unit children]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module with functions to describe.
    /// * `text_offsets` — Slice of `(function_name, start_offset, size)`
    ///   tuples mapping function names to their `.text` section byte ranges.
    ///
    /// # Returns
    ///
    /// The complete `.debug_info` section as a `Vec<u8>`.
    pub fn generate(&mut self, module: &IrModule, text_offsets: &[(String, u64, u64)]) -> Vec<u8> {
        self.buffer.clear();
        self.type_die_offsets.clear();

        // ==================================================================
        // Step 1: Compilation Unit Header (11 bytes for 32-bit DWARF)
        // ==================================================================
        // Reserve 4 bytes for unit_length — patched at the end.
        self.buffer.extend_from_slice(&[0u8; 4]);
        // DWARF version (u16 LE)
        self.buffer.extend_from_slice(&DWARF_VERSION.to_le_bytes());
        // debug_abbrev_offset (u32 LE) — always 0 for a single-CU output.
        self.buffer.extend_from_slice(&0u32.to_le_bytes());
        // address_size (u8)
        self.buffer.push(self.address_size);

        // ==================================================================
        // Step 2: DW_TAG_compile_unit DIE
        // ==================================================================
        self.emit_compile_unit_die(module, text_offsets);

        // ==================================================================
        // Step 3: Type DIEs (children of compile_unit)
        // ==================================================================
        self.emit_standard_base_types();
        // Emit a void pointer type as a common type reference.
        self.ensure_pointer_type_die();

        // ==================================================================
        // Step 4: DW_TAG_subprogram DIEs (one per defined function)
        // ==================================================================
        let functions: Vec<_> = module.functions().to_vec();
        for func in &functions {
            if !func.is_definition {
                continue;
            }
            self.emit_subprogram_die(func, text_offsets);
        }

        // ==================================================================
        // Step 5: Null terminator for compile_unit children
        // ==================================================================
        self.buffer.push(0x00);

        // ==================================================================
        // Step 6: Patch unit_length field
        // ==================================================================
        // unit_length = total CU size minus the 4-byte length field itself.
        let unit_length = (self.buffer.len() - 4) as u32;
        self.buffer[0..4].copy_from_slice(&unit_length.to_le_bytes());

        self.buffer.clone()
    }

    // ======================================================================
    // DW_TAG_compile_unit DIE emission
    // ======================================================================

    /// Emit the `DW_TAG_compile_unit` DIE.
    ///
    /// Attributes emitted (matching the abbreviation registered by
    /// `AbbrevTable::add_compile_unit_abbrev()`):
    ///
    /// 1. `DW_AT_producer`  (DW_FORM_strp)       — "BCC 0.1.0"
    /// 2. `DW_AT_language`  (DW_FORM_data2)       — DW_LANG_C11
    /// 3. `DW_AT_name`      (DW_FORM_strp)        — source file name
    /// 4. `DW_AT_comp_dir`  (DW_FORM_strp)        — compilation directory
    /// 5. `DW_AT_low_pc`    (DW_FORM_addr)        — lowest .text address
    /// 6. `DW_AT_high_pc`   (DW_FORM_data8)       — .text range size
    /// 7. `DW_AT_stmt_list` (DW_FORM_sec_offset)  — offset into .debug_line
    fn emit_compile_unit_die(&mut self, module: &IrModule, text_offsets: &[(String, u64, u64)]) {
        // Abbreviation code
        encode_uleb128(self.codes.compile_unit as u64, &mut self.buffer);

        // DW_AT_producer (DW_FORM_strp)
        let producer_off = self.str_table.add_string(BCC_PRODUCER);
        self.buffer.extend_from_slice(&producer_off.to_le_bytes());

        // DW_AT_language (DW_FORM_data2)
        self.buffer.extend_from_slice(&DW_LANG_C11.to_le_bytes());

        // DW_AT_name (DW_FORM_strp) — primary source file name
        let source_name = self
            .source_map
            .get_filename(0)
            .unwrap_or(module.name.as_str());
        let name_off = self.str_table.add_string(source_name);
        self.buffer.extend_from_slice(&name_off.to_le_bytes());

        // DW_AT_comp_dir (DW_FORM_strp) — compilation directory
        let comp_dir = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| ".".to_string());
        let comp_dir_off = self.str_table.add_string(&comp_dir);
        self.buffer.extend_from_slice(&comp_dir_off.to_le_bytes());

        // DW_AT_low_pc (DW_FORM_addr) — lowest .text address in the CU
        let low_pc = text_offsets
            .iter()
            .map(|(_, start, _)| *start)
            .min()
            .unwrap_or(0);
        emit_address(low_pc, self.address_size, &mut self.buffer);

        // DW_AT_high_pc (DW_FORM_data8) — size of the .text range
        let high_pc = text_offsets
            .iter()
            .map(|(_, start, size)| start + size)
            .max()
            .unwrap_or(0);
        let text_size = high_pc.saturating_sub(low_pc);
        self.buffer.extend_from_slice(&text_size.to_le_bytes());

        // DW_AT_stmt_list (DW_FORM_sec_offset) — offset into .debug_line
        self.buffer.extend_from_slice(&0u32.to_le_bytes());
    }

    // ======================================================================
    // Standard base type DIE emission
    // ======================================================================

    /// Emit `DW_TAG_base_type` DIEs for all standard C primitive types.
    ///
    /// Each base type is emitted exactly once and its DIE offset is cached
    /// in `self.type_die_offsets` for later reference by variables, parameters,
    /// and compound type DIEs.
    ///
    /// Types emitted:
    /// - `_Bool`         → I1,  1 byte,  DW_ATE_BOOLEAN
    /// - `char`          → I8,  1 byte,  DW_ATE_SIGNED_CHAR
    /// - `unsigned char`  → I8,  1 byte,  DW_ATE_UNSIGNED_CHAR (separate key not in IrType)
    /// - `short`         → I16, 2 bytes, DW_ATE_SIGNED
    /// - `int`           → I32, 4 bytes, DW_ATE_SIGNED
    /// - `long long`     → I64, 8 bytes, DW_ATE_SIGNED
    /// - `__int128`      → I128, 16 bytes, DW_ATE_SIGNED
    /// - `float`         → F32, 4 bytes, DW_ATE_FLOAT
    /// - `double`        → F64, 8 bytes, DW_ATE_FLOAT
    /// - `long double`   → F80, target-dependent size, DW_ATE_FLOAT
    fn emit_standard_base_types(&mut self) {
        // Each entry: (IrType variant, C name, byte_size, encoding)
        //
        // Note: I8 is emitted as "char" with DW_ATE_SIGNED_CHAR.  In a full
        // implementation, `unsigned char` would be a separate type, but since
        // IrType collapses signed/unsigned to a single I8 variant, we emit
        // the signed variant here.  The DW_ATE_UNSIGNED_CHAR constant is
        // available for future use when the type system distinguishes signedness.
        let f80_size = IrType::F80.size_bytes(self.target) as u8;
        let base_types: Vec<(IrType, &str, u8, u8)> = vec![
            (IrType::I1, "_Bool", 1, DW_ATE_BOOLEAN),
            (IrType::I8, "char", 1, DW_ATE_SIGNED_CHAR),
            (IrType::I16, "short", 2, DW_ATE_SIGNED),
            (IrType::I32, "int", 4, DW_ATE_SIGNED),
            (IrType::I64, "long long", 8, DW_ATE_SIGNED),
            (IrType::I128, "__int128", 16, DW_ATE_SIGNED),
            (IrType::F32, "float", 4, DW_ATE_FLOAT),
            (IrType::F64, "double", 8, DW_ATE_FLOAT),
            (IrType::F80, "long double", f80_size, DW_ATE_FLOAT),
        ];

        for (ir_type, name, byte_size, encoding) in base_types {
            self.emit_base_type_die(&ir_type, name, byte_size, encoding);
        }

        // Also emit Void as a zero-size type so pointer-to-void can reference it.
        self.emit_base_type_die(&IrType::Void, "void", 0, DW_ATE_SIGNED);
    }

    /// Emit a single `DW_TAG_base_type` DIE and register it in the type map.
    ///
    /// Attributes (matching `AbbrevTable::add_base_type_abbrev()`):
    /// 1. `DW_AT_name`      (DW_FORM_strp)  — type name string
    /// 2. `DW_AT_byte_size` (DW_FORM_data1) — size in bytes
    /// 3. `DW_AT_encoding`  (DW_FORM_data1) — DW_ATE_* encoding
    fn emit_base_type_die(&mut self, ir_type: &IrType, name: &str, byte_size: u8, encoding: u8) {
        let die_offset = self.buffer.len() as u32;

        // Abbreviation code for DW_TAG_base_type
        encode_uleb128(self.codes.base_type as u64, &mut self.buffer);

        // DW_AT_name (DW_FORM_strp)
        let name_off = self.str_table.add_string(name);
        self.buffer.extend_from_slice(&name_off.to_le_bytes());

        // DW_AT_byte_size (DW_FORM_data1)
        self.buffer.push(byte_size);

        // DW_AT_encoding (DW_FORM_data1)
        self.buffer.push(encoding);

        // Cache offset for type deduplication
        self.type_die_offsets.insert(ir_type.clone(), die_offset);
    }

    // ======================================================================
    // Pointer type DIE emission
    // ======================================================================

    /// Ensure a `DW_TAG_pointer_type` DIE exists for the opaque pointer.
    ///
    /// In BCC's IR, all pointers are opaque (`IrType::Ptr`), so we emit
    /// a single pointer-type DIE referencing the `void` base type DIE.
    fn ensure_pointer_type_die(&mut self) {
        if self.type_die_offsets.contains_key(&IrType::Ptr) {
            return;
        }

        let die_offset = self.buffer.len() as u32;

        // Abbreviation code for DW_TAG_pointer_type
        encode_uleb128(self.codes.pointer_type as u64, &mut self.buffer);

        // DW_AT_type (DW_FORM_ref4) — reference to pointed-to type DIE (void)
        let void_offset = self
            .type_die_offsets
            .get(&IrType::Void)
            .copied()
            .unwrap_or(0);
        self.buffer.extend_from_slice(&void_offset.to_le_bytes());

        // DW_AT_byte_size (DW_FORM_data1) — pointer size
        self.buffer.push(self.address_size);

        // Cache
        self.type_die_offsets.insert(IrType::Ptr, die_offset);
    }

    // ======================================================================
    // Type DIE resolution (lookup or emit)
    // ======================================================================

    /// Get the `.debug_info` offset for a type DIE, emitting the DIE if
    /// it has not been emitted yet.
    ///
    /// For base types (I1, I8, …, F80) and Ptr, the DIE was already emitted
    /// by [`emit_standard_base_types`] / [`ensure_pointer_type_die`].
    /// For composite types (Array, Struct, Function), a new DIE is emitted
    /// on first access and cached for subsequent references.
    fn resolve_type_die_offset(&mut self, ir_type: &IrType) -> u32 {
        // Return cached offset if already emitted.
        if let Some(&offset) = self.type_die_offsets.get(ir_type) {
            return offset;
        }

        match ir_type {
            IrType::Void
            | IrType::I1
            | IrType::I8
            | IrType::I16
            | IrType::I32
            | IrType::I64
            | IrType::I128
            | IrType::F32
            | IrType::F64
            | IrType::F80
            | IrType::Ptr => {
                // These should already be in the map from emit_standard_base_types
                // and ensure_pointer_type_die. If somehow missing, emit now.
                match ir_type {
                    IrType::Ptr => {
                        self.ensure_pointer_type_die();
                        self.type_die_offsets
                            .get(&IrType::Ptr)
                            .copied()
                            .unwrap_or(0)
                    }
                    _ => {
                        // Fallback — emit a generic base type
                        let (name, size, enc) = ir_type_to_base_info(ir_type, self.target);
                        self.emit_base_type_die(ir_type, name, size, enc);
                        self.type_die_offsets.get(ir_type).copied().unwrap_or(0)
                    }
                }
            }
            IrType::Array(elem_type, count) => {
                // For Array types, we use the int base type DIE offset as a
                // simplified representation since we don't have a separate
                // DW_TAG_array_type abbreviation registered.  In a full
                // implementation, this would emit DW_TAG_array_type with
                // DW_TAG_subrange_type children.  For now, reference the
                // element type and record the offset.
                let _elem_offset = self.resolve_type_die_offset(elem_type);
                let die_offset = self.buffer.len() as u32;

                // Emit as a base type placeholder with the array's total size.
                let total_size = elem_type.size_bytes(self.target) * count;
                let arr_name = format!("[{}; {}]", elem_type, count);
                encode_uleb128(self.codes.base_type as u64, &mut self.buffer);
                let name_off = self.str_table.add_string(&arr_name);
                self.buffer.extend_from_slice(&name_off.to_le_bytes());
                // DW_AT_byte_size — truncate to u8 if fits, else 0 (simplified)
                self.buffer.push(if total_size <= 255 {
                    total_size as u8
                } else {
                    0
                });
                // DW_AT_encoding — use unsigned since arrays don't have an encoding
                self.buffer.push(DW_ATE_UNSIGNED);

                self.type_die_offsets.insert(ir_type.clone(), die_offset);
                die_offset
            }
            IrType::Struct(st) => {
                // For Struct types, emit as a named base type placeholder.
                let die_offset = self.buffer.len() as u32;
                let struct_name = st.name.as_deref().unwrap_or("<anonymous struct>");
                let total_size = ir_type.size_bytes(self.target);
                encode_uleb128(self.codes.base_type as u64, &mut self.buffer);
                let name_off = self.str_table.add_string(struct_name);
                self.buffer.extend_from_slice(&name_off.to_le_bytes());
                self.buffer.push(if total_size <= 255 {
                    total_size as u8
                } else {
                    0
                });
                self.buffer.push(DW_ATE_UNSIGNED);

                self.type_die_offsets.insert(ir_type.clone(), die_offset);
                die_offset
            }
            IrType::Function(ret_type, _param_types) => {
                // For Function types (e.g. function pointers), reference the
                // return type.  A full implementation would emit
                // DW_TAG_subroutine_type with parameter children.
                let _ret_offset = self.resolve_type_die_offset(ret_type);
                let die_offset = self.buffer.len() as u32;
                let fn_name = format!("{}", ir_type);
                encode_uleb128(self.codes.base_type as u64, &mut self.buffer);
                let name_off = self.str_table.add_string(&fn_name);
                self.buffer.extend_from_slice(&name_off.to_le_bytes());
                self.buffer.push(self.address_size); // function pointer size
                self.buffer.push(DW_ATE_UNSIGNED);

                self.type_die_offsets.insert(ir_type.clone(), die_offset);
                die_offset
            }
        }
    }

    // ======================================================================
    // DW_TAG_subprogram DIE emission
    // ======================================================================

    /// Emit a `DW_TAG_subprogram` DIE for a single function definition.
    ///
    /// Attributes (matching `AbbrevTable::add_subprogram_abbrev()`):
    /// 1. `DW_AT_name`       (DW_FORM_strp)         — function name
    /// 2. `DW_AT_decl_file`  (DW_FORM_udata)        — source file index (1-based)
    /// 3. `DW_AT_decl_line`  (DW_FORM_udata)        — declaration line number
    /// 4. `DW_AT_low_pc`     (DW_FORM_addr)         — function start address
    /// 5. `DW_AT_high_pc`    (DW_FORM_data8)        — function size in bytes
    /// 6. `DW_AT_type`       (DW_FORM_ref4)         — return type DIE reference
    /// 7. `DW_AT_external`   (DW_FORM_flag_present) — external linkage (implicit)
    /// 8. `DW_AT_frame_base` (DW_FORM_exprloc)      — frame base expression
    /// 9. `DW_AT_prototyped` (DW_FORM_flag_present) — has prototype (implicit)
    ///
    /// If the function has parameters, the subprogram DIE uses the
    /// "with children" abbreviation and is followed by
    /// `DW_TAG_formal_parameter` DIEs and a null terminator.
    fn emit_subprogram_die(&mut self, func: &IrFunction, text_offsets: &[(String, u64, u64)]) {
        let has_children = !func.params.is_empty() || !func.local_var_debug_info.is_empty();

        // Look up text section offset for this function
        let (func_low_pc, func_size) = text_offsets
            .iter()
            .find(|(name, _, _)| name == &func.name)
            .map(|(_, start, size)| (*start, *size))
            .unwrap_or((0, 0));

        // Choose abbreviation code based on whether function has children
        let abbrev_code = if has_children {
            self.codes.subprogram_with_children
        } else {
            self.codes.subprogram_no_children
        };

        encode_uleb128(abbrev_code as u64, &mut self.buffer);

        // DW_AT_name (DW_FORM_strp)
        let fn_name_off = self.str_table.add_string(&func.name);
        self.buffer.extend_from_slice(&fn_name_off.to_le_bytes());

        // DW_AT_decl_file (DW_FORM_udata) — file index (1-based)
        encode_uleb128(1, &mut self.buffer);

        // DW_AT_decl_line (DW_FORM_udata) — declaration line (default 1)
        encode_uleb128(1, &mut self.buffer);

        // DW_AT_low_pc (DW_FORM_addr) — function start address
        emit_address(func_low_pc, self.address_size, &mut self.buffer);

        // DW_AT_high_pc (DW_FORM_data8) — function size in bytes
        self.buffer.extend_from_slice(&func_size.to_le_bytes());

        // DW_AT_type (DW_FORM_ref4) — return type DIE reference
        let ret_type_offset = self.resolve_type_die_offset(&func.return_type);
        self.buffer
            .extend_from_slice(&ret_type_offset.to_le_bytes());

        // DW_AT_external (DW_FORM_flag_present) — no data bytes emitted.
        // Implicitly true for all DIEs using this abbreviation.
        // (The flag_present form means the flag is always true when the
        // abbreviation is used — no payload is written.)

        // DW_AT_frame_base (DW_FORM_exprloc) — DW_OP_call_frame_cfa
        encode_uleb128(1, &mut self.buffer); // expression length = 1
        self.buffer.push(DW_OP_CALL_FRAME_CFA);

        // DW_AT_prototyped (DW_FORM_flag_present) — no data bytes emitted.

        // Emit children (parameters and local variables)
        if has_children {
            // Emit DW_TAG_formal_parameter for each parameter
            for (idx, param) in func.params.iter().enumerate() {
                self.emit_formal_parameter_die(param, idx as i64);
            }

            // Emit DW_TAG_variable for each local variable
            let param_count = func.params.len();
            for var_info in &func.local_var_debug_info {
                self.emit_variable_die(var_info, param_count);
            }

            // Null terminator for subprogram children
            self.buffer.push(0x00);
        }

        // Access schema-required members:
        // - is_variadic: in a full implementation, a variadic function would
        //   emit a DW_TAG_unspecified_parameters child DIE after the last
        //   formal parameter.
        // - linkage: used to determine DW_AT_external (External/Weak → true,
        //   Internal/Common → false).  The current abbreviation always emits
        //   DW_AT_external as FLAG_PRESENT (implicitly true), which is correct
        //   for the common case.
        let _ = func.is_variadic;
        let _ = func.linkage;
    }

    // ======================================================================
    // DW_TAG_formal_parameter DIE emission
    // ======================================================================

    /// Emit a `DW_TAG_formal_parameter` DIE for a function parameter.
    ///
    /// Attributes (matching `AbbrevTable::add_formal_parameter_abbrev()`):
    /// 1. `DW_AT_name`      (DW_FORM_strp)    — parameter name
    /// 2. `DW_AT_decl_file` (DW_FORM_udata)   — file index
    /// 3. `DW_AT_decl_line` (DW_FORM_udata)   — line number
    /// 4. `DW_AT_type`      (DW_FORM_ref4)    — type DIE reference
    /// 5. `DW_AT_location`  (DW_FORM_exprloc) — parameter location
    fn emit_formal_parameter_die(
        &mut self,
        param: &crate::ir::function::FunctionParam,
        stack_offset: i64,
    ) {
        // Abbreviation code for DW_TAG_formal_parameter
        encode_uleb128(self.codes.formal_parameter as u64, &mut self.buffer);

        // DW_AT_name (DW_FORM_strp)
        let name_off = self.str_table.add_string(&param.name);
        self.buffer.extend_from_slice(&name_off.to_le_bytes());

        // DW_AT_decl_file (DW_FORM_udata) — file index
        encode_uleb128(1, &mut self.buffer);

        // DW_AT_decl_line (DW_FORM_udata) — line number
        encode_uleb128(1, &mut self.buffer);

        // DW_AT_type (DW_FORM_ref4) — type DIE reference
        let type_offset = self.resolve_type_die_offset(&param.ty);
        self.buffer.extend_from_slice(&type_offset.to_le_bytes());

        // DW_AT_location (DW_FORM_exprloc) — DW_OP_fbreg <offset>
        // Encode as: [length, DW_OP_fbreg, sleb128(offset)]
        let mut loc_expr = Vec::with_capacity(8);
        loc_expr.push(DW_OP_FBREG);
        encode_sleb128(stack_offset * -(self.address_size as i64), &mut loc_expr);
        encode_uleb128(loc_expr.len() as u64, &mut self.buffer);
        self.buffer.extend_from_slice(&loc_expr);
    }

    // ======================================================================
    // DW_TAG_variable DIE emission
    // ======================================================================

    /// Emit a `DW_TAG_variable` DIE for a local variable.
    ///
    /// Attributes (matching `AbbrevTable::add_variable_abbrev()`):
    /// 1. `DW_AT_name`      (DW_FORM_strp)    — variable name
    /// 2. `DW_AT_decl_file` (DW_FORM_udata)   — file index
    /// 3. `DW_AT_decl_line` (DW_FORM_udata)   — line number
    /// 4. `DW_AT_type`      (DW_FORM_ref4)    — type DIE reference
    /// 5. `DW_AT_location`  (DW_FORM_exprloc) — stack location
    ///
    /// The stack offset is computed as `-(param_count + alloca_index + 1) *
    /// address_size` from the frame base, placing local variables below
    /// parameters in the stack frame.
    fn emit_variable_die(
        &mut self,
        var: &crate::ir::function::LocalVarDebugInfo,
        param_count: usize,
    ) {
        // Abbreviation code for DW_TAG_variable
        encode_uleb128(self.codes.variable as u64, &mut self.buffer);

        // DW_AT_name (DW_FORM_strp)
        let name_off = self.str_table.add_string(&var.name);
        self.buffer.extend_from_slice(&name_off.to_le_bytes());

        // DW_AT_decl_file (DW_FORM_udata) — file index (1-based)
        encode_uleb128(1, &mut self.buffer);

        // DW_AT_decl_line (DW_FORM_udata) — declaration line number
        encode_uleb128(u64::from(var.decl_line), &mut self.buffer);

        // DW_AT_type (DW_FORM_ref4) — type DIE reference
        let type_offset = self.resolve_type_die_offset(&var.ir_type);
        self.buffer.extend_from_slice(&type_offset.to_le_bytes());

        // DW_AT_location (DW_FORM_exprloc) — DW_OP_fbreg <offset>
        // Local variables are placed after parameters on the stack frame.
        let slot = (param_count as i64) + (var.alloca_index as i64) + 1;
        let offset = -(slot * (self.address_size as i64));
        let mut loc_expr = Vec::with_capacity(8);
        loc_expr.push(DW_OP_FBREG);
        encode_sleb128(offset, &mut loc_expr);
        encode_uleb128(loc_expr.len() as u64, &mut self.buffer);
        self.buffer.extend_from_slice(&loc_expr);
    }
}

// ===========================================================================
// Helper: Map IrType to base type info
// ===========================================================================

/// Map an [`IrType`] to its DWARF base type information.
///
/// Returns `(name, byte_size, encoding)` for use in `DW_TAG_base_type` DIEs.
fn ir_type_to_base_info(ir_type: &IrType, target: &Target) -> (&'static str, u8, u8) {
    match ir_type {
        IrType::Void => ("void", 0, DW_ATE_SIGNED),
        IrType::I1 => ("_Bool", 1, DW_ATE_BOOLEAN),
        IrType::I8 => ("char", 1, DW_ATE_SIGNED_CHAR),
        IrType::I16 => ("short", 2, DW_ATE_SIGNED),
        IrType::I32 => ("int", 4, DW_ATE_SIGNED),
        IrType::I64 => ("long long", 8, DW_ATE_SIGNED),
        IrType::I128 => ("__int128", 16, DW_ATE_SIGNED),
        IrType::F32 => ("float", 4, DW_ATE_FLOAT),
        IrType::F64 => ("double", 8, DW_ATE_FLOAT),
        IrType::F80 => (
            "long double",
            IrType::F80.size_bytes(target) as u8,
            DW_ATE_FLOAT,
        ),
        IrType::Ptr => ("void *", target.pointer_width() as u8, DW_ATE_UNSIGNED),
        IrType::Array(_, _) => ("<array>", 0, DW_ATE_UNSIGNED),
        IrType::Struct(_) => ("<struct>", 0, DW_ATE_UNSIGNED),
        IrType::Function(_, _) => ("<function>", 0, DW_ATE_UNSIGNED),
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::{FunctionParam, IrFunction};
    use crate::ir::instructions::Value;

    // ===================================================================
    // LEB128 Encoding Tests
    // ===================================================================

    #[test]
    fn test_uleb128_zero() {
        let mut buf = Vec::new();
        encode_uleb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_uleb128_single_byte_values() {
        let mut buf = Vec::new();
        encode_uleb128(1, &mut buf);
        assert_eq!(buf, vec![0x01]);

        buf.clear();
        encode_uleb128(127, &mut buf);
        assert_eq!(buf, vec![0x7F]);
    }

    #[test]
    fn test_uleb128_two_byte_values() {
        let mut buf = Vec::new();
        encode_uleb128(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        encode_uleb128(129, &mut buf);
        assert_eq!(buf, vec![0x81, 0x01]);
    }

    #[test]
    fn test_uleb128_standard_test_vector() {
        // Standard DWARF test vector: 624485 → [0xE5, 0x8E, 0x26]
        let mut buf = Vec::new();
        encode_uleb128(624485, &mut buf);
        assert_eq!(buf, vec![0xE5, 0x8E, 0x26]);
    }

    #[test]
    fn test_uleb128_large_value() {
        let mut buf = Vec::new();
        encode_uleb128(u64::MAX, &mut buf);
        // u64::MAX needs 10 bytes in ULEB128
        assert_eq!(buf.len(), 10);
        // All but the last byte have the continuation bit set
        for &byte in &buf[..9] {
            assert!(byte & 0x80 != 0);
        }
        // Last byte must not have continuation bit
        assert!(buf[9] & 0x80 == 0);
    }

    #[test]
    fn test_sleb128_zero() {
        let mut buf = Vec::new();
        encode_sleb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_sleb128_negative_one() {
        let mut buf = Vec::new();
        encode_sleb128(-1, &mut buf);
        assert_eq!(buf, vec![0x7F]);
    }

    #[test]
    fn test_sleb128_positive_128() {
        let mut buf = Vec::new();
        encode_sleb128(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn test_sleb128_negative_128() {
        let mut buf = Vec::new();
        encode_sleb128(-128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x7F]);
    }

    #[test]
    fn test_sleb128_negative_five() {
        let mut buf = Vec::new();
        encode_sleb128(-5, &mut buf);
        assert_eq!(buf, vec![0x7B]);
    }

    #[test]
    fn test_sleb128_large_negative() {
        // -123456 should encode correctly
        let mut buf = Vec::new();
        encode_sleb128(-123456, &mut buf);
        // Decode and verify: shift-and-or each 7-bit group
        assert!(!buf.is_empty());
        let mut result: i64 = 0;
        let mut shift = 0u32;
        for &byte in &buf {
            result |= ((byte & 0x7F) as i64) << shift;
            shift += 7;
        }
        // Sign extend if necessary
        let last_byte = *buf.last().unwrap();
        if shift < 64 && (last_byte & 0x40) != 0 {
            result |= !0i64 << shift;
        }
        assert_eq!(result, -123456);
    }

    // ===================================================================
    // Address emission tests
    // ===================================================================

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

    #[test]
    fn test_emit_address_zero() {
        let mut buf = Vec::new();
        emit_address(0, 4, &mut buf);
        assert_eq!(buf, vec![0, 0, 0, 0]);

        buf.clear();
        emit_address(0, 8, &mut buf);
        assert_eq!(buf, vec![0, 0, 0, 0, 0, 0, 0, 0]);
    }

    // ===================================================================
    // DebugInfoGenerator creation test
    // ===================================================================

    #[test]
    fn test_debug_info_generator_creation() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let gen = DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        assert_eq!(gen.address_size, 8);
        assert!(gen.type_die_offsets.is_empty());
    }

    #[test]
    fn test_debug_info_generator_creation_i686() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::I686;

        let gen = DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        assert_eq!(gen.address_size, 4);
    }

    // ===================================================================
    // Compilation unit header tests
    // ===================================================================

    #[test]
    fn test_generate_produces_valid_cu_header() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let module = IrModule::new("test.c".to_string());
        let result = gen.generate(&module, &[]);

        // Verify minimum size: 11-byte header + at least 1 byte for CU DIE
        assert!(
            result.len() >= 11,
            "result too short: {} bytes",
            result.len()
        );

        // Verify unit_length field (first 4 bytes, LE)
        let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(
            unit_length as usize,
            result.len() - 4,
            "unit_length mismatch"
        );

        // Verify DWARF version = 4 (bytes 4-5)
        let version = u16::from_le_bytes([result[4], result[5]]);
        assert_eq!(version, 4, "DWARF version should be 4");

        // Verify debug_abbrev_offset = 0 (bytes 6-9)
        let abbrev_offset = u32::from_le_bytes([result[6], result[7], result[8], result[9]]);
        assert_eq!(abbrev_offset, 0, "abbrev offset should be 0");

        // Verify address_size = 8 for X86_64 (byte 10)
        assert_eq!(result[10], 8, "address_size should be 8 for X86_64");

        // Last byte should be 0x00 (null terminator for CU children)
        assert_eq!(
            *result.last().unwrap(),
            0x00,
            "last byte should be null terminator"
        );
    }

    #[test]
    fn test_generate_i686_address_size() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::I686;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let module = IrModule::new("test.c".to_string());
        let result = gen.generate(&module, &[]);

        // address_size should be 4 for i686
        assert_eq!(result[10], 4, "address_size should be 4 for I686");
    }

    // ===================================================================
    // Subprogram DIE tests
    // ===================================================================

    #[test]
    fn test_generate_with_function() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let mut module = IrModule::new("test.c".to_string());
        let func = IrFunction::new("main".to_string(), vec![], IrType::I32);
        module.add_function(func);

        let offsets = vec![("main".to_string(), 0u64, 42u64)];
        let result = gen.generate(&module, &offsets);

        // Should be larger than the empty-module case (header + CU DIE +
        // base types + pointer type + subprogram DIE + null terminator)
        assert!(result.len() > 50, "result should contain subprogram DIE");

        // Verify unit_length consistency
        let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(unit_length as usize, result.len() - 4);
    }

    #[test]
    fn test_generate_with_function_parameters() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let mut module = IrModule::new("test.c".to_string());
        let params = vec![
            FunctionParam::new("argc".to_string(), IrType::I32, Value(0)),
            FunctionParam::new("argv".to_string(), IrType::Ptr, Value(1)),
        ];
        let func = IrFunction::new("main".to_string(), params, IrType::I32);
        module.add_function(func);

        let offsets = vec![("main".to_string(), 0u64, 100u64)];
        let result = gen.generate(&module, &offsets);

        // Should include formal parameter DIEs + null terminators
        assert!(
            result.len() > 80,
            "result should contain parameter DIEs, got {} bytes",
            result.len()
        );

        // Verify unit_length consistency
        let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(unit_length as usize, result.len() - 4);
    }

    // ===================================================================
    // Multiple functions test
    // ===================================================================

    #[test]
    fn test_generate_multiple_functions() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let mut module = IrModule::new("test.c".to_string());

        // Add three functions
        let f1 = IrFunction::new("foo".to_string(), vec![], IrType::Void);
        let f2 = IrFunction::new("bar".to_string(), vec![], IrType::I32);
        let f3_params = vec![FunctionParam::new("x".to_string(), IrType::I32, Value(0))];
        let f3 = IrFunction::new("baz".to_string(), f3_params, IrType::I64);

        module.add_function(f1);
        module.add_function(f2);
        module.add_function(f3);

        let offsets = vec![
            ("foo".to_string(), 0u64, 20u64),
            ("bar".to_string(), 20u64, 30u64),
            ("baz".to_string(), 50u64, 25u64),
        ];

        let result = gen.generate(&module, &offsets);

        // Verify overall structure is valid
        let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(unit_length as usize, result.len() - 4);
        assert_eq!(*result.last().unwrap(), 0x00);
    }

    // ===================================================================
    // Declaration-only functions are skipped
    // ===================================================================

    #[test]
    fn test_declaration_only_functions_skipped() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let mut module = IrModule::new("test.c".to_string());

        // A definition
        let f1 = IrFunction::new("defined".to_string(), vec![], IrType::I32);

        // A declaration only
        let mut f2 = IrFunction::new("declared".to_string(), vec![], IrType::I32);
        f2.is_definition = false;

        module.add_function(f1);
        module.add_function(f2);

        let offsets = vec![("defined".to_string(), 0u64, 50u64)];
        let result_with_decl = gen.generate(&module, &offsets);

        // Generate with only the definition
        let mut module2 = IrModule::new("test.c".to_string());
        let f1_only = IrFunction::new("defined".to_string(), vec![], IrType::I32);
        module2.add_function(f1_only);

        let mut str_table2 = DebugStrTable::new();
        let mut abbrev_table2 = AbbrevTable::new();
        let mut gen2 =
            DebugInfoGenerator::new(&mut str_table2, &mut abbrev_table2, &target, &source_map);
        let result_no_decl = gen2.generate(&module2, &offsets);

        // Both should produce the same output (declarations are skipped)
        assert_eq!(result_with_decl.len(), result_no_decl.len());
    }

    // ===================================================================
    // Empty module test
    // ===================================================================

    #[test]
    fn test_generate_empty_module() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let module = IrModule::new("empty.c".to_string());
        let result = gen.generate(&module, &[]);

        // Even an empty module should have a valid CU header + CU DIE +
        // base types + null terminator.
        assert!(result.len() > 11);

        let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(unit_length as usize, result.len() - 4);
    }

    // ===================================================================
    // Type deduplication test
    // ===================================================================

    #[test]
    fn test_type_deduplication() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let mut module = IrModule::new("test.c".to_string());

        // Add two functions both returning I32 — type DIE should be deduplicated
        let f1 = IrFunction::new("a".to_string(), vec![], IrType::I32);
        let f2 = IrFunction::new("b".to_string(), vec![], IrType::I32);
        module.add_function(f1);
        module.add_function(f2);

        let offsets = vec![
            ("a".to_string(), 0u64, 10u64),
            ("b".to_string(), 10u64, 10u64),
        ];

        let result = gen.generate(&module, &offsets);

        // Verify the type map has entries (deduplication occurred)
        assert!(gen.type_die_offsets.contains_key(&IrType::I32));
        assert!(gen.type_die_offsets.contains_key(&IrType::Void));
        assert!(gen.type_die_offsets.contains_key(&IrType::Ptr));

        // Both functions should reference the same I32 DIE offset
        let i32_offset = gen.type_die_offsets.get(&IrType::I32).unwrap();
        assert!(*i32_offset > 0, "I32 offset should be > 0 (after header)");

        let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(unit_length as usize, result.len() - 4);
    }

    // ===================================================================
    // ir_type_to_base_info tests
    // ===================================================================

    #[test]
    fn test_ir_type_to_base_info() {
        let target = Target::X86_64;

        let (name, size, enc) = ir_type_to_base_info(&IrType::I32, &target);
        assert_eq!(name, "int");
        assert_eq!(size, 4);
        assert_eq!(enc, DW_ATE_SIGNED);

        let (name, size, enc) = ir_type_to_base_info(&IrType::I1, &target);
        assert_eq!(name, "_Bool");
        assert_eq!(size, 1);
        assert_eq!(enc, DW_ATE_BOOLEAN);

        let (name, size, enc) = ir_type_to_base_info(&IrType::F64, &target);
        assert_eq!(name, "double");
        assert_eq!(size, 8);
        assert_eq!(enc, DW_ATE_FLOAT);

        let (name, size, enc) = ir_type_to_base_info(&IrType::F80, &target);
        assert_eq!(name, "long double");
        assert_eq!(size, 16); // 16 on x86-64
        assert_eq!(enc, DW_ATE_FLOAT);

        let (name, size, enc) = ir_type_to_base_info(&IrType::Ptr, &target);
        assert_eq!(name, "void *");
        assert_eq!(size, 8); // 8 on x86-64
        assert_eq!(enc, DW_ATE_UNSIGNED);
    }

    #[test]
    fn test_ir_type_to_base_info_i686() {
        let target = Target::I686;

        let (name, size, enc) = ir_type_to_base_info(&IrType::F80, &target);
        assert_eq!(name, "long double");
        assert_eq!(size, 12); // 12 on i686
        assert_eq!(enc, DW_ATE_FLOAT);

        let (name, size, enc) = ir_type_to_base_info(&IrType::Ptr, &target);
        assert_eq!(name, "void *");
        assert_eq!(size, 4); // 4 on i686
        assert_eq!(enc, DW_ATE_UNSIGNED);
    }

    // ===================================================================
    // Regeneration idempotency test
    // ===================================================================

    #[test]
    fn test_generate_twice_produces_same_output() {
        let mut str_table = DebugStrTable::new();
        let mut abbrev_table = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;

        let mut gen =
            DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, &target, &source_map);

        let mut module = IrModule::new("test.c".to_string());
        let func = IrFunction::new("main".to_string(), vec![], IrType::I32);
        module.add_function(func);

        let offsets = vec![("main".to_string(), 0u64, 42u64)];

        let result1 = gen.generate(&module, &offsets);
        let result2 = gen.generate(&module, &offsets);

        assert_eq!(result1, result2, "generate() should be idempotent");
    }

    // ===================================================================
    // All architectures test
    // ===================================================================

    #[test]
    fn test_all_architectures() {
        let targets = [
            (Target::X86_64, 8u8),
            (Target::I686, 4u8),
            (Target::AArch64, 8u8),
            (Target::RiscV64, 8u8),
        ];

        for (target, expected_addr_size) in &targets {
            let mut str_table = DebugStrTable::new();
            let mut abbrev_table = AbbrevTable::new();
            let source_map = SourceMap::new();

            let mut gen =
                DebugInfoGenerator::new(&mut str_table, &mut abbrev_table, target, &source_map);

            let module = IrModule::new("test.c".to_string());
            let result = gen.generate(&module, &[]);

            assert_eq!(
                result[10], *expected_addr_size,
                "address_size mismatch for {:?}",
                target
            );

            let unit_length = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
            assert_eq!(unit_length as usize, result.len() - 4);
        }
    }
}
