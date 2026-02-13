//! DWARF v4 `.debug_info` section generator.
//!
//! This module produces the `.debug_info` section containing Debug Information
//! Entries (DIEs) for source-level debugging at `-O0`. It generates:
//!
//! - `DW_TAG_compile_unit` DIEs with producer name, language (C11), compilation
//!   directory, and source file reference.
//! - `DW_TAG_subprogram` DIEs for each function with name, low/high PC addresses,
//!   external visibility flag, and return type reference.
//! - `DW_TAG_formal_parameter` DIEs for function parameters with name, type, and
//!   location expression.
//! - `DW_TAG_variable` DIEs for local and global variables with name, type, and
//!   location expressions.
//! - `DW_TAG_base_type`, `DW_TAG_pointer_type`, `DW_TAG_structure_type`, and
//!   `DW_TAG_array_type` DIEs for type descriptions.
//!
//! All DIE construction references abbreviation codes from
//! [`DebugAbbrevBuilder`] and string offsets from [`DebugStrTable`].
//!
//! # DWARF v4 32-bit Format
//!
//! The `.debug_info` section uses DWARF v4 32-bit format with an 11-byte
//! compilation unit header:
//!
//! | Field             | Size    | Description                                |
//! |-------------------|---------|--------------------------------------------|
//! | `unit_length`     | 4 bytes | Length of CU data (excluding this field)   |
//! | `version`         | 2 bytes | DWARF version number (4)                   |
//! | `abbrev_offset`   | 4 bytes | Offset into `.debug_abbrev`                |
//! | `address_size`    | 1 byte  | Size of target addresses (4 or 8)          |
//!
//! Following the header, DIEs are serialized as ULEB128 abbreviation codes
//! followed by attribute values encoded per their `DW_FORM_*` specifications.
//!
//! # Zero Dependencies
//!
//! All DWARF constants, LEB128 encoding, and section formatting are
//! implemented internally with zero external crate dependencies.

use crate::backend::dwarf::abbrev::{
    AbbrevEntry, DebugAbbrevBuilder,
    // Tag constants needed for subrange registration
    DW_TAG_SUBRANGE_TYPE,
    // Attribute constants needed for subrange registration
    DW_AT_COUNT,
    // Form constants needed for attribute encoding
    DW_FORM_UDATA,
};
use crate::backend::dwarf::debug_str::DebugStrTable;
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::common::types::{size_of, CType, FieldDef};
use crate::ir::function::{IrFunction, Linkage};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

// ===========================================================================
// DWARF v4 Constants (locally defined — zero external dependencies)
// ===========================================================================

/// DWARF version number written into the compilation unit header.
const DWARF_VERSION: u16 = 4;

/// Source language constant: C11 (ISO/IEC 9899:2011).
pub const DW_LANG_C11: u16 = 0x1d;

// ---------------------------------------------------------------------------
// Base Type Encoding Constants (DW_ATE_*)
// ---------------------------------------------------------------------------

/// Machine address encoding.
pub const DW_ATE_ADDRESS: u8 = 0x01;

/// Boolean type encoding (true/false).
pub const DW_ATE_BOOLEAN: u8 = 0x02;

/// IEEE floating-point encoding.
pub const DW_ATE_FLOAT: u8 = 0x04;

/// Signed binary integer encoding.
pub const DW_ATE_SIGNED: u8 = 0x05;

/// Signed character encoding (e.g., `signed char`).
pub const DW_ATE_SIGNED_CHAR: u8 = 0x06;

/// Unsigned binary integer encoding.
pub const DW_ATE_UNSIGNED: u8 = 0x07;

/// Unsigned character encoding (e.g., `unsigned char`).
pub const DW_ATE_UNSIGNED_CHAR: u8 = 0x08;

// ---------------------------------------------------------------------------
// Location Expression Opcodes (DW_OP_*)
// ---------------------------------------------------------------------------

/// Frame-base-relative offset location expression opcode.
/// Followed by a SLEB128 offset from the canonical frame address.
pub const DW_OP_FBREG: u8 = 0x91;

/// Base for register location expression opcodes (DW_OP_reg0..DW_OP_reg31).
/// A value of `DW_OP_REG0 + N` encodes a direct register reference to reg N.
pub const DW_OP_REG0: u8 = 0x50;

/// Base for register-plus-offset expression opcodes (DW_OP_breg0..DW_OP_breg31).
/// A value of `DW_OP_BREG0 + N` is followed by a SLEB128 offset from reg N.
pub const DW_OP_BREG0: u8 = 0x70;

/// DW_OP_addr: push a target-width address constant onto the expression stack.
pub const DW_OP_ADDR: u8 = 0x03;

// ===========================================================================
// Type Deduplication Key
// ===========================================================================

/// Key for deduplicating type DIEs in the `.debug_info` section.
///
/// When the same C type appears in multiple variable or parameter declarations,
/// we emit only one type DIE and reuse its CU-relative offset via the cache
/// in [`DebugInfoBuilder::type_die_offsets`].
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum TypeKey {
    /// Base (primitive) type identified by name, storage size, and encoding.
    Base {
        name: String,
        byte_size: u8,
        encoding: u8,
    },
    /// Pointer type identified by its pointee type DIE offset.
    Pointer { pointee_offset: u32 },
    /// Named struct type, deduplicated by name.
    /// Anonymous structs are not cached (each instance gets a fresh DIE).
    Struct { name: String },
    /// Array type identified by element type DIE offset and element count.
    Array {
        element_offset: u32,
        count: usize,
    },
}

// ===========================================================================
// DebugInfoBuilder — Public API
// ===========================================================================

/// Builder for constructing the DWARF v4 `.debug_info` section.
///
/// This struct serializes Debug Information Entries (DIEs) into a byte buffer
/// following the DWARF v4 32-bit format specification. It manages:
///
/// - Compilation unit headers with backpatching of the `unit_length` field
/// - DIE emission with correct abbreviation codes and attribute encoding
/// - Type DIE deduplication via an [`FxHashMap`]-based offset cache
/// - String references via the shared `.debug_str` table ([`DebugStrTable`])
///
/// # Example
///
/// ```ignore
/// let mut str_table = DebugStrTable::new();
/// let mut builder = DebugInfoBuilder::new(Target::X86_64, &mut str_table);
///
/// builder.begin_compile_unit("bcc 1.0", "/home/user", "main.c", 0x1000, 0x50, 0);
/// let int_type = builder.emit_base_type("int", 4, DW_ATE_SIGNED);
/// builder.emit_subprogram("main", 0x1000, 0x50, true, Some(int_type));
///   builder.emit_formal_parameter("argc", int_type, &[DW_OP_FBREG, 0x00]);
///   builder.emit_end_children(); // end subprogram children
/// builder.emit_end_children(); // end compile_unit children
/// builder.end_compile_unit();
///
/// let section_bytes = builder.finish();
/// ```
pub struct DebugInfoBuilder<'a> {
    /// Serialized `.debug_info` section bytes.
    data: Vec<u8>,

    /// Mutable reference to the shared `.debug_str` string table.
    /// All string-valued attributes use `DW_FORM_strp` encoding,
    /// storing the string once and referencing it by byte offset.
    string_table: &'a mut DebugStrTable,

    /// Owned abbreviation table builder. Created during construction
    /// with all standard abbreviation entries pre-registered.
    /// Callers can retrieve it via [`abbrev_builder()`] to emit
    /// the `.debug_abbrev` section.
    abbrev_builder: DebugAbbrevBuilder,

    /// Cache mapping type signatures to their CU-relative DIE offsets.
    /// Prevents duplicate type DIE emission when the same C type appears
    /// in multiple variable, parameter, or return-type positions.
    type_die_offsets: FxHashMap<TypeKey, u32>,

    /// Target architecture — determines address size (4 or 8 bytes)
    /// for `DW_FORM_addr` encoding and pointer type `DW_AT_byte_size`.
    target: Target,

    /// Byte offset in [`data`] where the current compilation unit begins.
    /// Used to compute CU-relative offsets for `DW_FORM_ref4` values.
    cu_start_offset: usize,

    /// Byte offset in [`data`] where the `unit_length` placeholder was written.
    /// Backpatched by [`end_compile_unit()`] with the actual CU size.
    unit_length_patch_offset: usize,

    // -----------------------------------------------------------------------
    // Pre-registered abbreviation codes for each DIE kind.
    // These are assigned during construction and remain constant for the
    // lifetime of the builder.
    // -----------------------------------------------------------------------

    /// Abbreviation code for `DW_TAG_compile_unit`.
    cu_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_subprogram` with a return type attribute.
    subprogram_typed_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_subprogram` without return type (void).
    subprogram_void_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_variable`.
    variable_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_formal_parameter`.
    formal_param_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_base_type`.
    base_type_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_pointer_type`.
    pointer_type_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_structure_type`.
    struct_type_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_member` (struct field).
    member_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_array_type`.
    array_type_abbrev_code: u32,
    /// Abbreviation code for `DW_TAG_subrange_type` (array dimension bound).
    subrange_abbrev_code: u32,
}

impl<'a> DebugInfoBuilder<'a> {
    // ===================================================================
    // Construction
    // ===================================================================

    /// Creates a new `DebugInfoBuilder` for the given target architecture.
    ///
    /// Internally creates a [`DebugAbbrevBuilder`] and pre-registers all
    /// standard abbreviation entries needed for DWARF generation:
    /// compile unit, subprogram (with/without return type), variable,
    /// formal parameter, base type, pointer type, structure type,
    /// member, array type, and subrange type.
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture. Determines address size
    ///   (4 bytes for i686, 8 bytes for 64-bit targets).
    /// * `string_table` — Mutable reference to the shared `.debug_str` table.
    ///   All string-valued attributes are stored here and referenced by offset.
    pub fn new(target: Target, string_table: &'a mut DebugStrTable) -> Self {
        let mut abbrev_builder = DebugAbbrevBuilder::new();

        // Pre-register all standard abbreviation entries.
        // Each method returns a unique 1-based abbreviation code.
        let cu_abbrev_code = abbrev_builder.add_compile_unit_abbrev();
        let subprogram_typed_abbrev_code = abbrev_builder.add_subprogram_abbrev(true);
        let subprogram_void_abbrev_code = abbrev_builder.add_subprogram_abbrev(false);
        let variable_abbrev_code = abbrev_builder.add_variable_abbrev();
        let formal_param_abbrev_code = abbrev_builder.add_formal_parameter_abbrev();
        let base_type_abbrev_code = abbrev_builder.add_base_type_abbrev();
        let pointer_type_abbrev_code = abbrev_builder.add_pointer_type_abbrev();
        let struct_type_abbrev_code = abbrev_builder.add_struct_type_abbrev();
        let member_abbrev_code = abbrev_builder.add_member_abbrev();
        let array_type_abbrev_code = abbrev_builder.add_array_type_abbrev();

        // Register subrange type abbreviation for array dimension bounds.
        // Uses the generic add_entry() since there is no dedicated template.
        let subrange_abbrev_code = abbrev_builder.add_entry(
            DW_TAG_SUBRANGE_TYPE,
            false, // subrange has no children
            vec![(DW_AT_COUNT, DW_FORM_UDATA)],
        );

        // Validate that the compile-unit abbreviation was registered
        // correctly. This exercises get_entry() and AbbrevEntry::code.
        let cu_entry: &AbbrevEntry = abbrev_builder
            .get_entry(cu_abbrev_code)
            .expect("compile-unit abbreviation must be registered");
        debug_assert_eq!(cu_entry.code, cu_abbrev_code);

        DebugInfoBuilder {
            data: Vec::with_capacity(4096),
            string_table,
            abbrev_builder,
            type_die_offsets: FxHashMap::default(),
            target,
            cu_start_offset: 0,
            unit_length_patch_offset: 0,
            cu_abbrev_code,
            subprogram_typed_abbrev_code,
            subprogram_void_abbrev_code,
            variable_abbrev_code,
            formal_param_abbrev_code,
            base_type_abbrev_code,
            pointer_type_abbrev_code,
            struct_type_abbrev_code,
            member_abbrev_code,
            array_type_abbrev_code,
            subrange_abbrev_code,
        }
    }

    // ===================================================================
    // Compilation Unit Header
    // ===================================================================

    /// Begins a new compilation unit in the `.debug_info` section.
    ///
    /// Writes the DWARF v4 compilation unit header (11 bytes for 32-bit
    /// format) followed by the `DW_TAG_compile_unit` DIE with attributes.
    /// The `unit_length` field is written as a placeholder and backpatched
    /// by [`end_compile_unit()`].
    ///
    /// After calling this method, emit child DIEs (subprograms, variables,
    /// types) and then call [`emit_end_children()`] followed by
    /// [`end_compile_unit()`] to close the unit.
    ///
    /// # Arguments
    ///
    /// * `producer` — Compiler identification (e.g., `"bcc 1.0"`).
    /// * `comp_dir` — Compilation directory path.
    /// * `source_file` — Primary source file name.
    /// * `low_pc` — Lowest code address in this compilation unit.
    /// * `high_pc` — Address range length (highest address minus `low_pc`).
    /// * `stmt_list_offset` — Byte offset into `.debug_line` for this CU's
    ///   line number program.
    pub fn begin_compile_unit(
        &mut self,
        producer: &str,
        comp_dir: &str,
        source_file: &str,
        low_pc: u64,
        high_pc: u64,
        stmt_list_offset: u32,
    ) {
        // Record the start of this compilation unit for CU-relative offsets.
        self.cu_start_offset = self.data.len();

        // Clear the type DIE cache — CU-relative offsets (DW_FORM_ref4)
        // are only valid within a single compilation unit.
        self.type_die_offsets = FxHashMap::default();

        // --- Compilation Unit Header (DWARF v4 32-bit format, 11 bytes) ---

        // unit_length: 4 bytes (placeholder — backpatched by end_compile_unit)
        self.unit_length_patch_offset = self.data.len();
        self.write_u32(0x0000_0000);

        // version: 2 bytes
        self.write_u16(DWARF_VERSION);

        // debug_abbrev_offset: 4 bytes (offset into .debug_abbrev section)
        self.write_u32(0);

        // address_size: 1 byte (4 for i686, 8 for 64-bit targets)
        self.write_u8(self.address_size());

        // --- DW_TAG_compile_unit DIE ---
        // Abbreviation code (ULEB128)
        self.write_uleb128(self.cu_abbrev_code as u64);

        // Attributes must match the order registered in add_compile_unit_abbrev:
        //   DW_AT_PRODUCER   -> DW_FORM_STRP
        //   DW_AT_LANGUAGE   -> DW_FORM_DATA2
        //   DW_AT_NAME       -> DW_FORM_STRP
        //   DW_AT_COMP_DIR   -> DW_FORM_STRP
        //   DW_AT_LOW_PC     -> DW_FORM_ADDR
        //   DW_AT_HIGH_PC    -> DW_FORM_DATA8
        //   DW_AT_STMT_LIST  -> DW_FORM_SEC_OFFSET

        // DW_AT_PRODUCER: DW_FORM_STRP (4-byte offset into .debug_str)
        let producer_offset = self.get_or_add_string(producer);
        self.write_u32(producer_offset);

        // DW_AT_LANGUAGE: DW_FORM_DATA2 (2-byte language code)
        self.write_u16(DW_LANG_C11);

        // DW_AT_NAME: DW_FORM_STRP (4-byte offset into .debug_str)
        let name_offset = self.get_or_add_string(source_file);
        self.write_u32(name_offset);

        // DW_AT_COMP_DIR: DW_FORM_STRP (4-byte offset into .debug_str)
        let comp_dir_offset = self.get_or_add_string(comp_dir);
        self.write_u32(comp_dir_offset);

        // DW_AT_LOW_PC: DW_FORM_ADDR (address_size bytes)
        self.write_address(low_pc);

        // DW_AT_HIGH_PC: DW_FORM_DATA8 (8-byte constant — range length)
        self.write_u64(high_pc);

        // DW_AT_STMT_LIST: DW_FORM_SEC_OFFSET (4-byte offset into .debug_line)
        self.write_u32(stmt_list_offset);

        // Children follow — caller must emit child DIEs and then call
        // emit_end_children() + end_compile_unit().
    }

    // ===================================================================
    // Subprogram (Function) DIEs
    // ===================================================================

    /// Emits a `DW_TAG_subprogram` DIE for a function.
    ///
    /// The subprogram DIE is marked as having children so that formal
    /// parameters and local variables can be nested inside it. After
    /// emitting all child DIEs, call [`emit_end_children()`] to close
    /// the subprogram's child list.
    ///
    /// # Arguments
    ///
    /// * `name` — Function name.
    /// * `low_pc` — Function start address.
    /// * `high_pc` — Function address range length (`end_addr - low_pc`).
    /// * `is_external` — `true` if the function has external linkage.
    /// * `return_type_offset` — CU-relative offset of the return type DIE,
    ///   or `None` for void functions.
    pub fn emit_subprogram(
        &mut self,
        name: &str,
        low_pc: u64,
        high_pc: u64,
        is_external: bool,
        return_type_offset: Option<u32>,
    ) {
        // Select the abbreviation based on whether we have a return type.
        // Subprograms with void return omit the DW_AT_TYPE attribute entirely.
        let abbrev_code = if return_type_offset.is_some() {
            self.subprogram_typed_abbrev_code
        } else {
            self.subprogram_void_abbrev_code
        };

        // Abbreviation code (ULEB128)
        self.write_uleb128(abbrev_code as u64);

        // Attributes (order per add_subprogram_abbrev):
        //   DW_AT_NAME       -> DW_FORM_STRP
        //   DW_AT_LOW_PC     -> DW_FORM_ADDR
        //   DW_AT_HIGH_PC    -> DW_FORM_DATA8
        //   DW_AT_EXTERNAL   -> DW_FORM_FLAG
        //   [DW_AT_TYPE      -> DW_FORM_REF4]   (only if has_type)
        //   DW_AT_DECL_FILE  -> DW_FORM_UDATA
        //   DW_AT_DECL_LINE  -> DW_FORM_UDATA

        // DW_AT_NAME: DW_FORM_STRP
        let name_offset = self.get_or_add_string(name);
        self.write_u32(name_offset);

        // DW_AT_LOW_PC: DW_FORM_ADDR
        self.write_address(low_pc);

        // DW_AT_HIGH_PC: DW_FORM_DATA8 (range length)
        self.write_u64(high_pc);

        // DW_AT_EXTERNAL: DW_FORM_FLAG (1 byte: 0 = false, 1 = true)
        self.write_u8(if is_external { 1 } else { 0 });

        // DW_AT_TYPE: DW_FORM_REF4 (only for non-void return types)
        if let Some(type_off) = return_type_offset {
            self.write_u32(type_off);
        }

        // DW_AT_DECL_FILE: DW_FORM_UDATA (file index, 0 = unknown)
        self.write_uleb128(0);

        // DW_AT_DECL_LINE: DW_FORM_UDATA (line number, 0 = unknown)
        self.write_uleb128(0);

        // Children follow — caller emits formal parameters, locals, then
        // calls emit_end_children().
    }

    // ===================================================================
    // Formal Parameter DIEs
    // ===================================================================

    /// Emits a `DW_TAG_formal_parameter` DIE for a function parameter.
    ///
    /// Must be emitted as a child of a subprogram DIE (between
    /// [`emit_subprogram()`] and [`emit_end_children()`]).
    ///
    /// # Arguments
    ///
    /// * `name` — Parameter name.
    /// * `type_offset` — CU-relative offset of the parameter's type DIE.
    /// * `location_expr` — DWARF location expression bytes describing
    ///   where the parameter value is stored (e.g., `[DW_OP_FBREG, offset]`).
    pub fn emit_formal_parameter(
        &mut self,
        name: &str,
        type_offset: u32,
        location_expr: &[u8],
    ) {
        // Abbreviation code (ULEB128)
        self.write_uleb128(self.formal_param_abbrev_code as u64);

        // Attributes (order per add_formal_parameter_abbrev):
        //   DW_AT_NAME     -> DW_FORM_STRP
        //   DW_AT_TYPE     -> DW_FORM_REF4
        //   DW_AT_LOCATION -> DW_FORM_EXPRLOC

        // DW_AT_NAME: DW_FORM_STRP
        let name_offset = self.get_or_add_string(name);
        self.write_u32(name_offset);

        // DW_AT_TYPE: DW_FORM_REF4 (CU-relative type DIE offset)
        self.write_u32(type_offset);

        // DW_AT_LOCATION: DW_FORM_EXPRLOC (ULEB128 length + raw bytes)
        self.write_uleb128(location_expr.len() as u64);
        self.data.extend_from_slice(location_expr);
    }

    // ===================================================================
    // Variable DIEs
    // ===================================================================

    /// Emits a `DW_TAG_variable` DIE for a local or global variable.
    ///
    /// Can be emitted as a child of a subprogram DIE (local variable)
    /// or a compile-unit DIE (global/file-scope variable).
    ///
    /// # Arguments
    ///
    /// * `name` — Variable name.
    /// * `type_offset` — CU-relative offset of the variable's type DIE.
    /// * `location_expr` — DWARF location expression bytes describing
    ///   the variable's storage location.
    pub fn emit_variable(
        &mut self,
        name: &str,
        type_offset: u32,
        location_expr: &[u8],
    ) {
        // Abbreviation code (ULEB128)
        self.write_uleb128(self.variable_abbrev_code as u64);

        // Attributes (order per add_variable_abbrev):
        //   DW_AT_NAME     -> DW_FORM_STRP
        //   DW_AT_TYPE     -> DW_FORM_REF4
        //   DW_AT_LOCATION -> DW_FORM_EXPRLOC

        // DW_AT_NAME: DW_FORM_STRP
        let name_offset = self.get_or_add_string(name);
        self.write_u32(name_offset);

        // DW_AT_TYPE: DW_FORM_REF4
        self.write_u32(type_offset);

        // DW_AT_LOCATION: DW_FORM_EXPRLOC
        self.write_uleb128(location_expr.len() as u64);
        self.data.extend_from_slice(location_expr);
    }

    // ===================================================================
    // Type DIEs
    // ===================================================================

    /// Emits a `DW_TAG_base_type` DIE for a primitive C type and returns
    /// its CU-relative offset.
    ///
    /// Base types are cached by `(name, byte_size, encoding)` — subsequent
    /// calls with the same triple return the cached offset without emitting
    /// a new DIE.
    ///
    /// # Arguments
    ///
    /// * `name` — Type display name (e.g., `"int"`, `"unsigned char"`).
    /// * `byte_size` — Storage size in bytes.
    /// * `encoding` — DWARF base type encoding (`DW_ATE_*` constant).
    ///
    /// # Returns
    ///
    /// CU-relative offset of the base type DIE.
    pub fn emit_base_type(&mut self, name: &str, byte_size: u8, encoding: u8) -> u32 {
        // Check the type cache for an existing DIE with the same signature.
        let key = TypeKey::Base {
            name: name.to_string(),
            byte_size,
            encoding,
        };
        if self.type_die_offsets.contains_key(&key) {
            return *self.type_die_offsets.get(&key).unwrap();
        }

        // Record the CU-relative offset of this DIE.
        let die_offset = self.current_cu_offset();

        // Abbreviation code (ULEB128)
        self.write_uleb128(self.base_type_abbrev_code as u64);

        // Attributes (order per add_base_type_abbrev):
        //   DW_AT_NAME      -> DW_FORM_STRP
        //   DW_AT_BYTE_SIZE -> DW_FORM_DATA1
        //   DW_AT_ENCODING  -> DW_FORM_DATA1

        // DW_AT_NAME: DW_FORM_STRP
        let name_offset = self.get_or_add_string(name);
        self.write_u32(name_offset);

        // DW_AT_BYTE_SIZE: DW_FORM_DATA1
        self.write_u8(byte_size);

        // DW_AT_ENCODING: DW_FORM_DATA1
        self.write_u8(encoding);

        // Cache the offset for future deduplication.
        self.type_die_offsets.insert(key, die_offset);

        die_offset
    }

    /// Emits a `DW_TAG_pointer_type` DIE and returns its CU-relative offset.
    ///
    /// Pointer types are cached by their pointee type offset.
    ///
    /// # Arguments
    ///
    /// * `pointee_offset` — CU-relative offset of the pointed-to type DIE.
    ///
    /// # Returns
    ///
    /// CU-relative offset of the pointer type DIE.
    pub fn emit_pointer_type(&mut self, pointee_offset: u32) -> u32 {
        // Check cache for an existing pointer-to-this-type DIE.
        let key = TypeKey::Pointer { pointee_offset };
        if let Some(&cached) = self.type_die_offsets.get(&key) {
            return cached;
        }

        let die_offset = self.current_cu_offset();

        // Abbreviation code (ULEB128)
        self.write_uleb128(self.pointer_type_abbrev_code as u64);

        // Attributes (order per add_pointer_type_abbrev):
        //   DW_AT_BYTE_SIZE -> DW_FORM_DATA1
        //   DW_AT_TYPE      -> DW_FORM_REF4

        // DW_AT_BYTE_SIZE: DW_FORM_DATA1 (pointer size from target arch)
        self.write_u8(self.address_size());

        // DW_AT_TYPE: DW_FORM_REF4 (pointee type)
        self.write_u32(pointee_offset);

        // Cache for deduplication.
        self.type_die_offsets.insert(key, die_offset);

        die_offset
    }

    /// Emits a `DW_TAG_structure_type` DIE with member children and
    /// returns its CU-relative offset.
    ///
    /// Named struct types are cached by name. Anonymous structs are
    /// always emitted fresh (not cached).
    ///
    /// Each field is represented as a tuple of `(name, type_offset,
    /// byte_offset)` where `type_offset` is the CU-relative offset of
    /// the field's type DIE and `byte_offset` is the field's position
    /// within the struct.
    ///
    /// # Arguments
    ///
    /// * `name` — Struct tag name, or `None` for anonymous structs.
    /// * `fields` — Slice of `(field_name, type_die_offset, byte_offset)`.
    ///
    /// # Returns
    ///
    /// CU-relative offset of the structure type DIE.
    pub fn emit_struct_type(
        &mut self,
        name: Option<&str>,
        fields: &[(String, u32, u32)],
    ) -> u32 {
        // Attempt cache lookup for named structs.
        if let Some(tag_name) = name {
            let key = TypeKey::Struct {
                name: tag_name.to_string(),
            };
            if let Some(&cached) = self.type_die_offsets.get(&key) {
                return cached;
            }
        }

        let die_offset = self.current_cu_offset();

        // Abbreviation code (ULEB128) — struct DIE has children (members)
        self.write_uleb128(self.struct_type_abbrev_code as u64);

        // Attributes (order per add_struct_type_abbrev):
        //   DW_AT_NAME      -> DW_FORM_STRP
        //   DW_AT_BYTE_SIZE -> DW_FORM_UDATA
        //   DW_AT_DECL_FILE -> DW_FORM_UDATA
        //   DW_AT_DECL_LINE -> DW_FORM_UDATA

        // DW_AT_NAME: DW_FORM_STRP (use empty string for anonymous structs)
        let display_name = name.unwrap_or("");
        let name_str_offset = self.get_or_add_string(display_name);
        self.write_u32(name_str_offset);

        // DW_AT_BYTE_SIZE: DW_FORM_UDATA (total struct size estimate)
        let total_size = Self::estimate_struct_size(fields);
        self.write_uleb128(total_size as u64);

        // DW_AT_DECL_FILE: DW_FORM_UDATA (0 = unknown)
        self.write_uleb128(0);

        // DW_AT_DECL_LINE: DW_FORM_UDATA (0 = unknown)
        self.write_uleb128(0);

        // --- DW_TAG_member children ---
        for (field_name, field_type_offset, field_byte_offset) in fields {
            self.emit_member(field_name, *field_type_offset, *field_byte_offset);
        }

        // Null terminator — end of children for this structure type
        self.data.push(0x00);

        // Cache named structs for deduplication.
        if let Some(tag_name) = name {
            let key = TypeKey::Struct {
                name: tag_name.to_string(),
            };
            self.type_die_offsets.insert(key, die_offset);
        }

        die_offset
    }

    /// Emits a `DW_TAG_array_type` DIE with a subrange child specifying
    /// the element count, and returns its CU-relative offset.
    ///
    /// Array types are cached by `(element_offset, count)`.
    ///
    /// # Arguments
    ///
    /// * `element_offset` — CU-relative offset of the element type DIE.
    /// * `count` — Number of elements in the array.
    ///
    /// # Returns
    ///
    /// CU-relative offset of the array type DIE.
    pub fn emit_array_type(&mut self, element_offset: u32, count: usize) -> u32 {
        // Check cache.
        let key = TypeKey::Array {
            element_offset,
            count,
        };
        if let Some(&cached) = self.type_die_offsets.get(&key) {
            return cached;
        }

        let die_offset = self.current_cu_offset();

        // Abbreviation code (ULEB128) — array DIE has children (subrange)
        self.write_uleb128(self.array_type_abbrev_code as u64);

        // Attributes (order per add_array_type_abbrev):
        //   DW_AT_TYPE -> DW_FORM_REF4

        // DW_AT_TYPE: DW_FORM_REF4 (element type)
        self.write_u32(element_offset);

        // --- DW_TAG_subrange_type child (array dimension) ---
        self.write_uleb128(self.subrange_abbrev_code as u64);

        // DW_AT_COUNT: DW_FORM_UDATA (element count)
        self.write_uleb128(count as u64);

        // Null terminator — end of children for this array type
        self.data.push(0x00);

        // Cache for deduplication.
        self.type_die_offsets.insert(key, die_offset);

        die_offset
    }

    // ===================================================================
    // Children Termination and Compilation Unit Finalization
    // ===================================================================

    /// Writes a null DIE (0x00) to terminate a list of children.
    ///
    /// In DWARF, a null entry (abbreviation code 0) marks the end of
    /// a sequence of child DIEs. Call this after emitting all children
    /// of a parent DIE (subprogram, compile unit, etc.).
    pub fn emit_end_children(&mut self) {
        self.data.push(0x00);
    }

    /// Finalizes the current compilation unit by backpatching the
    /// `unit_length` field in the CU header.
    ///
    /// Must be called after the compile-unit DIE and all its children
    /// (including the final null terminator from [`emit_end_children()`])
    /// have been emitted.
    pub fn end_compile_unit(&mut self) {
        // unit_length covers everything after the 4-byte unit_length field
        // itself: version (2) + abbrev_offset (4) + address_size (1) + DIEs.
        let unit_end = self.data.len();
        let length = (unit_end - self.unit_length_patch_offset - 4) as u32;

        // Backpatch the unit_length placeholder with the actual value.
        let patch_start = self.unit_length_patch_offset;
        self.data[patch_start..patch_start + 4].copy_from_slice(&length.to_le_bytes());
    }

    // ===================================================================
    // Section Output
    // ===================================================================

    /// Returns the complete serialized `.debug_info` section bytes.
    ///
    /// The returned `Vec<u8>` is ready to be written as the `.debug_info`
    /// ELF section content.
    pub fn finish(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Returns the current size (in bytes) of the `.debug_info` section.
    pub fn section_size(&self) -> usize {
        self.data.len()
    }

    /// Returns a shared reference to the internal abbreviation table builder.
    ///
    /// Callers use this to serialize the `.debug_abbrev` section content
    /// via [`DebugAbbrevBuilder::finish()`].
    pub fn abbrev_builder(&self) -> &DebugAbbrevBuilder {
        &self.abbrev_builder
    }

    /// Returns a mutable reference to the internal abbreviation table builder.
    pub fn abbrev_builder_mut(&mut self) -> &mut DebugAbbrevBuilder {
        &mut self.abbrev_builder
    }

    // ===================================================================
    // Private: Member DIE (struct field)
    // ===================================================================

    /// Emits a `DW_TAG_member` DIE for a struct/union field.
    ///
    /// This is called internally by [`emit_struct_type()`] for each field.
    fn emit_member(&mut self, name: &str, type_offset: u32, byte_offset: u32) {
        // Abbreviation code (ULEB128)
        self.write_uleb128(self.member_abbrev_code as u64);

        // Attributes (order per add_member_abbrev):
        //   DW_AT_NAME            -> DW_FORM_STRP
        //   DW_AT_TYPE            -> DW_FORM_REF4
        //   DW_AT_DATA_MEMBER_LOC -> DW_FORM_UDATA

        // DW_AT_NAME: DW_FORM_STRP
        let name_offset = self.get_or_add_string(name);
        self.write_u32(name_offset);

        // DW_AT_TYPE: DW_FORM_REF4 (field type)
        self.write_u32(type_offset);

        // DW_AT_DATA_MEMBER_LOC: DW_FORM_UDATA (byte offset within struct)
        self.write_uleb128(byte_offset as u64);
    }

    // ===================================================================
    // Private: Low-Level Binary Writing Helpers
    // ===================================================================

    /// Writes a single unsigned byte to the output buffer.
    fn write_u8(&mut self, value: u8) {
        self.data.push(value);
    }

    /// Writes a 16-bit unsigned integer in little-endian byte order.
    fn write_u16(&mut self, value: u16) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Writes a 32-bit unsigned integer in little-endian byte order.
    fn write_u32(&mut self, value: u32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Writes a 64-bit unsigned integer in little-endian byte order.
    fn write_u64(&mut self, value: u64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Encodes and writes a `u64` value in ULEB128 (Unsigned Little-Endian
    /// Base 128) format.
    ///
    /// ULEB128 is a variable-length encoding used throughout DWARF for
    /// abbreviation codes, attribute values, and form data.
    fn write_uleb128(&mut self, mut value: u64) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80; // set continuation bit
            }
            self.data.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    /// Encodes and writes an `i64` value in SLEB128 (Signed Little-Endian
    /// Base 128) format.
    ///
    /// SLEB128 is used for signed values in DWARF, particularly in
    /// location expressions (e.g., `DW_OP_fbreg` offset).
    #[allow(dead_code)]
    fn write_sleb128(&mut self, mut value: i64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            let sign_bit_set = (byte & 0x40) != 0;
            if (value == 0 && !sign_bit_set) || (value == -1 && sign_bit_set) {
                self.data.push(byte);
                break;
            } else {
                self.data.push(byte | 0x80);
            }
        }
    }

    /// Writes a target-width address value.
    ///
    /// For 32-bit targets (i686): writes 4 bytes.
    /// For 64-bit targets (x86-64, AArch64, RISC-V 64): writes 8 bytes.
    fn write_address(&mut self, addr: u64) {
        let addr_size = self.address_size();
        if addr_size == 4 {
            self.write_u32(addr as u32);
        } else {
            self.write_u64(addr);
        }
    }

    /// Returns the DWARF address size for the current target architecture.
    ///
    /// This uses [`Target::pointer_width()`] for the primary determination
    /// and validates consistency with [`Target::elf_class()`].
    fn address_size(&self) -> u8 {
        let pw = self.target.pointer_width() as u8;
        // Sanity check: ELF class and pointer width must be consistent.
        // ELFCLASS32 (1) => 4-byte pointers, ELFCLASS64 (2) => 8-byte pointers.
        debug_assert!(
            (self.target.elf_class() == 1 && pw == 4)
                || (self.target.elf_class() == 2 && pw == 8),
            "ELF class and pointer width must be consistent"
        );
        pw
    }

    /// Returns the current CU-relative byte offset in the output buffer.
    ///
    /// This is the offset from the start of the current compilation unit
    /// header, used for `DW_FORM_ref4` values that reference type DIEs.
    fn current_cu_offset(&self) -> u32 {
        (self.data.len() - self.cu_start_offset) as u32
    }

    /// Looks up or inserts a string into the shared `.debug_str` table
    /// and returns its byte offset.
    ///
    /// Uses [`DebugStrTable::get_offset()`] for existing strings and
    /// [`DebugStrTable::add_string()`] for new ones, avoiding duplicate
    /// string table entries.
    fn get_or_add_string(&mut self, s: &str) -> u32 {
        // Try to find an existing offset first to avoid duplicating entries.
        if let Some(offset) = self.string_table.get_offset(s) {
            return offset;
        }
        // String not yet in the table — add it and return the new offset.
        self.string_table.add_string(s)
    }

    /// Estimates the total byte size of a struct from its field layout.
    ///
    /// Takes the maximum of `(byte_offset + 1)` across all fields as a
    /// conservative lower bound. Returns 0 for structs with no fields.
    fn estimate_struct_size(fields: &[(String, u32, u32)]) -> u32 {
        if fields.is_empty() {
            return 0;
        }
        // Use the highest field offset + 1 as a lower bound.
        // This is imprecise (doesn't account for field sizes or trailing
        // padding), but adequate for DWARF DW_AT_byte_size when exact
        // struct layout information is unavailable at this layer.
        fields
            .iter()
            .map(|(_, _, offset)| *offset + 1)
            .max()
            .unwrap_or(0)
    }

    /// Appends SLEB128-encoded bytes to `buf` without writing to `self.data`.
    ///
    /// This standalone helper is used for constructing location expressions
    /// (e.g., `DW_OP_fbreg` + offset) in temporary buffers before embedding
    /// them in DIE attributes via `emit_variable()` or `emit_formal_parameter()`.
    fn append_sleb128(buf: &mut Vec<u8>, mut value: i64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            let sign_bit_set = (byte & 0x40) != 0;
            if (value == 0 && !sign_bit_set) || (value == -1 && sign_bit_set) {
                buf.push(byte);
                break;
            } else {
                buf.push(byte | 0x80);
            }
        }
    }

    /// Computes the CU address range from a map of function addresses.
    ///
    /// Returns `(low_pc, high_pc)` where `low_pc` is the smallest function
    /// start address and `high_pc` is the largest `low_pc + length`.
    /// Returns `(0, 0)` if the map is empty.
    fn compute_cu_range(fn_addrs: &FxHashMap<String, (u64, u64)>) -> (u64, u64) {
        if fn_addrs.is_empty() {
            return (0, 0);
        }
        let mut min_addr = u64::MAX;
        let mut max_addr = 0u64;
        for (low, length) in fn_addrs.values() {
            if *low < min_addr {
                min_addr = *low;
            }
            let high = low.saturating_add(*length);
            if high > max_addr {
                max_addr = high;
            }
        }
        (min_addr, max_addr)
    }

    // ===================================================================
    // High-Level: CType → DWARF Type DIE Emission
    // ===================================================================

    /// Emits a DWARF type DIE for a C language type and returns its
    /// CU-relative offset.
    ///
    /// Recursively handles all [`CType`] variants, mapping each to the
    /// appropriate DWARF representation:
    ///
    /// | C Type             | DWARF Tag               | Encoding               |
    /// |--------------------|-------------------------|------------------------|
    /// | `void`             | (no DIE — returns 0)    | —                      |
    /// | `_Bool`            | `DW_TAG_base_type`      | `DW_ATE_BOOLEAN`       |
    /// | `char`             | `DW_TAG_base_type`      | `DW_ATE_SIGNED_CHAR`   |
    /// | `int`, `short` etc | `DW_TAG_base_type`      | `DW_ATE_SIGNED`        |
    /// | `float`/`double`   | `DW_TAG_base_type`      | `DW_ATE_FLOAT`         |
    /// | pointer            | `DW_TAG_pointer_type`   | —                      |
    /// | struct/union       | `DW_TAG_structure_type` | —                      |
    /// | array              | `DW_TAG_array_type`     | —                      |
    /// | typedef            | (follows underlying)    | —                      |
    ///
    /// Type DIEs are deduplicated via the internal cache — calling this
    /// with the same type twice returns the same offset.
    ///
    /// # Arguments
    ///
    /// * `ctype` — The C language type to emit.
    ///
    /// # Returns
    ///
    /// CU-relative offset of the type DIE, or 0 for `Void`.
    pub fn emit_ctype_die(&mut self, ctype: &CType) -> u32 {
        let target = self.target;
        match ctype {
            CType::Void => 0,

            CType::Bool => self.emit_base_type("_Bool", 1, DW_ATE_BOOLEAN),

            CType::Char { .. } => {
                if ctype.is_signed() {
                    self.emit_base_type("char", 1, DW_ATE_SIGNED_CHAR)
                } else {
                    self.emit_base_type("unsigned char", 1, DW_ATE_UNSIGNED_CHAR)
                }
            }

            CType::Short { .. } => {
                let byte_size = size_of(ctype, &target) as u8;
                if ctype.is_signed() {
                    self.emit_base_type("short", byte_size, DW_ATE_SIGNED)
                } else {
                    self.emit_base_type("unsigned short", byte_size, DW_ATE_UNSIGNED)
                }
            }

            CType::Int { .. } => {
                let byte_size = size_of(ctype, &target) as u8;
                if ctype.is_signed() {
                    self.emit_base_type("int", byte_size, DW_ATE_SIGNED)
                } else {
                    self.emit_base_type("unsigned int", byte_size, DW_ATE_UNSIGNED)
                }
            }

            CType::Long { .. } => {
                let byte_size = size_of(ctype, &target) as u8;
                if ctype.is_signed() {
                    self.emit_base_type("long", byte_size, DW_ATE_SIGNED)
                } else {
                    self.emit_base_type("unsigned long", byte_size, DW_ATE_UNSIGNED)
                }
            }

            CType::LongLong { .. } => {
                let byte_size = size_of(ctype, &target) as u8;
                if ctype.is_signed() {
                    self.emit_base_type("long long", byte_size, DW_ATE_SIGNED)
                } else {
                    self.emit_base_type("unsigned long long", byte_size, DW_ATE_UNSIGNED)
                }
            }

            CType::Float => self.emit_base_type("float", 4, DW_ATE_FLOAT),

            CType::Double => self.emit_base_type("double", 8, DW_ATE_FLOAT),

            CType::LongDouble => {
                let byte_size = size_of(ctype, &target) as u8;
                self.emit_base_type("long double", byte_size, DW_ATE_FLOAT)
            }

            CType::Complex(base) => {
                // Complex types are pairs of their base floating-point type.
                let byte_size = size_of(ctype, &target) as u8;
                let base_name = match base.as_ref() {
                    CType::Float => "complex float",
                    CType::Double => "complex double",
                    CType::LongDouble => "complex long double",
                    _ => "complex",
                };
                self.emit_base_type(base_name, byte_size, DW_ATE_FLOAT)
            }

            CType::Pointer(pointee) => {
                let pointee_offset = self.emit_ctype_die(pointee);
                self.emit_pointer_type(pointee_offset)
            }

            CType::Array { element, size } => {
                let elem_offset = self.emit_ctype_die(element);
                let count = size.unwrap_or(0);
                self.emit_array_type(elem_offset, count)
            }

            CType::Struct { name, fields } => {
                self.emit_ctype_struct(name.as_deref(), fields, &target)
            }

            CType::Union { name, fields } => {
                // Unions are represented as DW_TAG_structure_type in DWARF.
                // All fields carry byte_offset = 0 since they share storage.
                self.emit_ctype_union(name.as_deref(), fields, &target)
            }

            CType::Enum { name, underlying } => {
                // Enums map to their underlying integer type with the
                // enum tag name.
                let byte_size = size_of(underlying, &target) as u8;
                let encoding = if underlying.is_signed() {
                    DW_ATE_SIGNED
                } else {
                    DW_ATE_UNSIGNED
                };
                let display_name = name.as_deref().unwrap_or("enum");
                self.emit_base_type(display_name, byte_size, encoding)
            }

            CType::Typedef { underlying, .. } => {
                // For DWARF at -O0, follow the typedef chain to the
                // underlying type. A full DW_TAG_typedef DIE could be
                // added in a future enhancement.
                self.emit_ctype_die(underlying)
            }

            CType::Atomic(inner) => {
                // Atomic types have the same DWARF representation as
                // their inner type (atomicity is a qualifier).
                self.emit_ctype_die(inner)
            }

            CType::Function { .. } => {
                // Function types → emit as a void pointer for -O0 debug.
                let void_base = self.emit_base_type("void", 0, DW_ATE_ADDRESS);
                self.emit_pointer_type(void_base)
            }
        }
    }

    /// Emits struct fields from [`FieldDef`] descriptions, computing byte
    /// offsets using the target's `size_of` function.
    ///
    /// Called by [`emit_ctype_die()`] for `CType::Struct` variants.
    fn emit_ctype_struct(
        &mut self,
        name: Option<&str>,
        fields: &[FieldDef],
        target: &Target,
    ) -> u32 {
        let mut field_descs: Vec<(String, u32, u32)> = Vec::with_capacity(fields.len());
        let mut byte_offset: u32 = 0;
        for field in fields {
            let field_type_offset = self.emit_ctype_die(&field.ty);
            let field_name = field
                .name
                .as_deref()
                .unwrap_or("<anon>")
                .to_owned();
            field_descs.push((field_name, field_type_offset, byte_offset));
            // Advance offset by the field size. For bit-fields, advance by
            // the underlying type size (conservative — DWARF bit-field
            // location attributes are not emitted at -O0).
            let field_size = size_of(&field.ty, target) as u32;
            byte_offset += field_size;
        }
        self.emit_struct_type(name, &field_descs)
    }

    /// Emits union fields from [`FieldDef`] descriptions.
    ///
    /// All union fields share storage at byte offset 0.
    /// Called by [`emit_ctype_die()`] for `CType::Union` variants.
    fn emit_ctype_union(
        &mut self,
        name: Option<&str>,
        fields: &[FieldDef],
        _target: &Target,
    ) -> u32 {
        let mut field_descs: Vec<(String, u32, u32)> = Vec::with_capacity(fields.len());
        for field in fields {
            let field_type_offset = self.emit_ctype_die(&field.ty);
            let field_name = field
                .name
                .as_deref()
                .unwrap_or("<anon>")
                .to_owned();
            // All union members share storage at offset 0.
            field_descs.push((field_name, field_type_offset, 0));
        }
        self.emit_struct_type(name, &field_descs)
    }

    // ===================================================================
    // High-Level: IrType → DWARF Type DIE Emission
    // ===================================================================

    /// Emits a DWARF type DIE for an IR type and returns its CU-relative
    /// offset.
    ///
    /// Maps the intermediate representation type system to DWARF base types
    /// using size information from [`IrType::size_bytes()`].
    ///
    /// # Arguments
    ///
    /// * `ir_type` — The IR type to emit.
    ///
    /// # Returns
    ///
    /// CU-relative offset of the type DIE, or 0 for void types.
    pub fn emit_ir_type_die(&mut self, ir_type: &IrType) -> u32 {
        let target = self.target;

        if ir_type.is_void() {
            return 0;
        }

        if ir_type.is_pointer() {
            // Opaque pointer — emit as void*.
            let void_base = self.emit_base_type("void", 0, DW_ATE_ADDRESS);
            return self.emit_pointer_type(void_base);
        }

        if ir_type.is_integer() {
            let byte_size = ir_type.size_bytes(&target) as u8;
            let (name, encoding) = match ir_type {
                IrType::I1 => ("_Bool", DW_ATE_BOOLEAN),
                IrType::I8 => ("char", DW_ATE_SIGNED_CHAR),
                IrType::I16 => ("short", DW_ATE_SIGNED),
                IrType::I32 => ("int", DW_ATE_SIGNED),
                IrType::I64 => ("long", DW_ATE_SIGNED),
                IrType::I128 => ("__int128", DW_ATE_SIGNED),
                _ => unreachable!("is_integer() returned true for non-integer"),
            };
            return self.emit_base_type(name, byte_size, encoding);
        }

        match ir_type {
            IrType::F32 => self.emit_base_type("float", 4, DW_ATE_FLOAT),
            IrType::F64 => self.emit_base_type("double", 8, DW_ATE_FLOAT),
            IrType::F80 => {
                let byte_size = ir_type.size_bytes(&target) as u8;
                self.emit_base_type("long double", byte_size, DW_ATE_FLOAT)
            }
            IrType::Array { element, count } => {
                let elem_offset = self.emit_ir_type_die(element);
                self.emit_array_type(elem_offset, *count)
            }
            IrType::Struct { fields, .. } => {
                // Anonymous IR struct — emit with synthesized field names.
                let mut field_descs: Vec<(String, u32, u32)> =
                    Vec::with_capacity(fields.len());
                let mut byte_offset: u32 = 0;
                for (i, field_ty) in fields.iter().enumerate() {
                    let field_type_offset = self.emit_ir_type_die(field_ty);
                    let field_name = format!("field_{}", i);
                    field_descs.push((field_name, field_type_offset, byte_offset));
                    byte_offset += field_ty.size_bytes(&target) as u32;
                }
                self.emit_struct_type(None, &field_descs)
            }
            IrType::Function { .. } => {
                // Function types at IR level → emit as void pointer.
                let void_base = self.emit_base_type("void", 0, DW_ATE_ADDRESS);
                self.emit_pointer_type(void_base)
            }
            // Already handled by early returns (Void, Ptr, integer types).
            _ => 0,
        }
    }

    // ===================================================================
    // High-Level: Module & Function Debug Info Generation
    // ===================================================================

    /// Generates complete DWARF `.debug_info` content for an IR module.
    ///
    /// This is the top-level entry point for DWARF generation. It produces
    /// a single DWARF compilation unit containing:
    ///
    /// - `DW_TAG_compile_unit` with producer, language, directory, source file
    /// - `DW_TAG_subprogram` for each function definition (with parameters)
    /// - `DW_TAG_variable` for each global variable
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module containing functions and globals.
    /// * `source_map` — Source file registry for resolving file names.
    /// * `comp_dir` — Compilation directory path.
    /// * `stmt_list_offset` — Byte offset into `.debug_line` for this CU.
    /// * `fn_addrs` — Map from function name to `(low_pc, high_pc_length)`.
    pub fn generate_module_debug_info(
        &mut self,
        module: &IrModule,
        source_map: &SourceMap,
        comp_dir: &str,
        stmt_list_offset: u32,
        fn_addrs: &FxHashMap<String, (u64, u64)>,
    ) {
        // Extract module metadata.
        let module_name = &module.name;
        let target = module.target;

        // Compute compilation-unit address range from function addresses.
        let (cu_low_pc, cu_high_pc) = Self::compute_cu_range(fn_addrs);

        // Resolve the source file name. If the source map has a file with
        // ID 0, use its name; otherwise fall back to the module name.
        let source_file_name = if source_map.file_count() > 0 {
            let file_id = crate::common::source_map::FileId(0);
            let file = source_map.get_file(file_id);
            // Resolve the first source location — this validates the
            // source map's line table and provides a reference point for
            // DWARF consumers that need a starting location in the CU.
            let _start_loc = source_map.lookup_location(file_id, 0);
            file.name.clone()
        } else {
            module_name.clone()
        };

        // Begin the compilation unit DIE.
        self.begin_compile_unit(
            "bcc 1.0",
            comp_dir,
            &source_file_name,
            cu_low_pc,
            cu_high_pc,
            stmt_list_offset,
        );

        // Emit DW_TAG_subprogram DIEs for each function definition.
        for func in &module.functions {
            if !func.is_definition {
                continue;
            }
            let (low_pc, high_pc) = fn_addrs
                .get(&func.name)
                .copied()
                .unwrap_or((0, 0));

            self.generate_function_debug_info(func, low_pc, high_pc);
        }

        // Emit DW_TAG_variable DIEs for global variables.
        for global in &module.globals {
            let type_offset = self.emit_ir_type_die(&global.ty);
            if type_offset == 0 && global.ty.is_void() {
                continue; // Skip void-typed globals.
            }

            // Global variable location expression: DW_OP_addr + address.
            // The actual address is unknown at DWARF emission time and will
            // be patched via relocations. We emit a placeholder here.
            let mut loc_expr = vec![DW_OP_ADDR];
            let addr_size = target.pointer_width();
            if addr_size == 4 {
                loc_expr.extend_from_slice(&0u32.to_le_bytes());
            } else {
                loc_expr.extend_from_slice(&0u64.to_le_bytes());
            }

            self.emit_variable(&global.name, type_offset, &loc_expr);
        }

        // Close the compilation unit's child list.
        self.emit_end_children();
        self.end_compile_unit();
    }

    /// Generates DWARF subprogram and parameter DIEs for a single function.
    ///
    /// Emits:
    /// - `DW_TAG_subprogram` with name, address range, and external flag
    /// - `DW_TAG_formal_parameter` for each function parameter
    /// - Null terminator to close the subprogram's child list
    ///
    /// Uses [`IrFunction::name`], [`IrFunction::params`],
    /// [`IrFunction::return_type`], [`IrFunction::linkage`], and
    /// [`IrFunction::is_definition`] to populate the DIE attributes.
    ///
    /// # Arguments
    ///
    /// * `func` — The IR function to emit debug info for.
    /// * `low_pc` — Function start address in the compiled output.
    /// * `high_pc` — Function address range length.
    pub fn generate_function_debug_info(
        &mut self,
        func: &IrFunction,
        low_pc: u64,
        high_pc: u64,
    ) {
        // Skip non-definitions (extern function declarations).
        if !func.is_definition {
            return;
        }

        // Determine return type DIE offset.
        let return_type_offset = if func.return_type.is_void() {
            None
        } else {
            Some(self.emit_ir_type_die(&func.return_type))
        };

        // Determine if the function has external linkage.
        let is_external = matches!(
            func.linkage,
            Linkage::External | Linkage::Weak
        );

        // Emit the DW_TAG_subprogram DIE.
        self.emit_subprogram(
            &func.name,
            low_pc,
            high_pc,
            is_external,
            return_type_offset,
        );

        // Emit DW_TAG_formal_parameter DIEs for each parameter.
        for (i, param) in func.params.iter().enumerate() {
            let param_name = match param.name.as_deref() {
                Some(name) if !name.is_empty() => name.to_owned(),
                _ => format!("arg{}", i),
            };
            let param_type_offset = self.emit_ir_type_die(&param.ty);

            // Parameter location: DW_OP_fbreg + frame offset.
            // At -O0, parameters are at known frame offsets.
            // Use parameter index × 8 as a placeholder offset that the
            // backend will adjust to actual stack slot positions.
            let frame_offset = (i as i64) * 8;
            let mut loc_expr = vec![DW_OP_FBREG];
            Self::append_sleb128(&mut loc_expr, frame_offset);

            self.emit_formal_parameter(
                &param_name,
                param_type_offset,
                &loc_expr,
            );
        }

        // Close the subprogram's child list with a null DIE.
        self.emit_end_children();
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates a DebugInfoBuilder for x86-64 with a fresh string table.
    fn make_builder(str_table: &mut DebugStrTable) -> DebugInfoBuilder<'_> {
        DebugInfoBuilder::new(Target::X86_64, str_table)
    }

    #[test]
    fn test_uleb128_encoding() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);
        b.data.clear();

        b.write_uleb128(0);
        assert_eq!(b.data, vec![0x00]);

        b.data.clear();
        b.write_uleb128(127);
        assert_eq!(b.data, vec![0x7f]);

        b.data.clear();
        b.write_uleb128(128);
        assert_eq!(b.data, vec![0x80, 0x01]);

        b.data.clear();
        b.write_uleb128(624485);
        assert_eq!(b.data, vec![0xe5, 0x8e, 0x26]);
    }

    #[test]
    fn test_sleb128_encoding() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);
        b.data.clear();

        b.write_sleb128(0);
        assert_eq!(b.data, vec![0x00]);

        b.data.clear();
        b.write_sleb128(-1);
        assert_eq!(b.data, vec![0x7f]);

        b.data.clear();
        b.write_sleb128(63);
        assert_eq!(b.data, vec![0x3f]);

        b.data.clear();
        b.write_sleb128(-64);
        assert_eq!(b.data, vec![0x40]);

        b.data.clear();
        b.write_sleb128(64);
        assert_eq!(b.data, vec![0xc0, 0x00]);

        b.data.clear();
        b.write_sleb128(-65);
        assert_eq!(b.data, vec![0xbf, 0x7f]);
    }

    #[test]
    fn test_address_size_x86_64() {
        let mut str_table = DebugStrTable::new();
        let b = DebugInfoBuilder::new(Target::X86_64, &mut str_table);
        assert_eq!(b.address_size(), 8);
    }

    #[test]
    fn test_address_size_i686() {
        let mut str_table = DebugStrTable::new();
        let b = DebugInfoBuilder::new(Target::I686, &mut str_table);
        assert_eq!(b.address_size(), 4);
    }

    #[test]
    fn test_address_size_aarch64() {
        let mut str_table = DebugStrTable::new();
        let b = DebugInfoBuilder::new(Target::AArch64, &mut str_table);
        assert_eq!(b.address_size(), 8);
    }

    #[test]
    fn test_address_size_riscv64() {
        let mut str_table = DebugStrTable::new();
        let b = DebugInfoBuilder::new(Target::RiscV64, &mut str_table);
        assert_eq!(b.address_size(), 8);
    }

    #[test]
    fn test_compile_unit_header_size() {
        // Verify that begin_compile_unit produces a valid header.
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc 1.0", "/home/user", "main.c", 0x1000, 0x50, 0);

        // CU header is 11 bytes (4 unit_length + 2 version + 4 abbrev_offset + 1 addr_size)
        // followed by the compile_unit DIE attributes.
        assert!(b.data.len() > 11, "CU header + DIE must be > 11 bytes");

        // Check DWARF version at bytes 4..6.
        let version = u16::from_le_bytes([b.data[4], b.data[5]]);
        assert_eq!(version, DWARF_VERSION);

        // Check address_size at byte 10.
        assert_eq!(b.data[10], 8); // x86-64 => 8-byte addresses
    }

    #[test]
    fn test_end_compile_unit_backpatches_length() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);
        b.emit_end_children(); // end CU children
        b.end_compile_unit();

        // Read back the unit_length from bytes 0..4.
        let unit_length = u32::from_le_bytes([b.data[0], b.data[1], b.data[2], b.data[3]]);

        // unit_length should equal total size - 4 (the unit_length field itself).
        assert_eq!(unit_length as usize, b.data.len() - 4);
    }

    #[test]
    fn test_base_type_deduplication() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);

        let off1 = b.emit_base_type("int", 4, DW_ATE_SIGNED);
        let off2 = b.emit_base_type("int", 4, DW_ATE_SIGNED);

        // Duplicate base types must return the same offset.
        assert_eq!(off1, off2);

        // A different type must get a different offset.
        let off3 = b.emit_base_type("unsigned int", 4, DW_ATE_UNSIGNED);
        assert_ne!(off1, off3);
    }

    #[test]
    fn test_pointer_type_deduplication() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);

        let int_off = b.emit_base_type("int", 4, DW_ATE_SIGNED);
        let ptr1 = b.emit_pointer_type(int_off);
        let ptr2 = b.emit_pointer_type(int_off);

        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_finish_returns_data() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "t.c", 0, 0, 0);
        b.emit_end_children();
        b.end_compile_unit();

        let output = b.finish();
        assert_eq!(output.len(), b.section_size());
        assert!(!output.is_empty());
    }

    #[test]
    fn test_emit_subprogram_and_children() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0x1000, 0x100, 0);

        let int_off = b.emit_base_type("int", 4, DW_ATE_SIGNED);

        b.emit_subprogram("main", 0x1000, 0x80, true, Some(int_off));
        b.emit_formal_parameter("argc", int_off, &[DW_OP_FBREG, 0x00]);
        b.emit_variable("local_x", int_off, &[DW_OP_FBREG, 0x04]);
        b.emit_end_children(); // end subprogram children

        b.emit_end_children(); // end CU children
        b.end_compile_unit();

        let output = b.finish();
        assert!(output.len() > 40, "Output must contain CU header + DIEs");
    }

    #[test]
    fn test_emit_struct_type() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);

        let int_off = b.emit_base_type("int", 4, DW_ATE_SIGNED);

        let fields = vec![
            ("x".to_string(), int_off, 0u32),
            ("y".to_string(), int_off, 4u32),
        ];

        let struct_off = b.emit_struct_type(Some("point"), &fields);
        assert!(struct_off > 0);

        // Named structs should be deduplicated.
        let struct_off2 = b.emit_struct_type(Some("point"), &fields);
        assert_eq!(struct_off, struct_off2);
    }

    #[test]
    fn test_emit_array_type() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);

        let int_off = b.emit_base_type("int", 4, DW_ATE_SIGNED);
        let arr_off = b.emit_array_type(int_off, 10);
        assert!(arr_off > 0);

        // Deduplication check.
        let arr_off2 = b.emit_array_type(int_off, 10);
        assert_eq!(arr_off, arr_off2);

        // Different count => different DIE.
        let arr_off3 = b.emit_array_type(int_off, 20);
        assert_ne!(arr_off, arr_off3);
    }

    #[test]
    fn test_estimate_struct_size() {
        let fields: Vec<(String, u32, u32)> = vec![
            ("a".to_string(), 0, 0),
            ("b".to_string(), 0, 4),
            ("c".to_string(), 0, 8),
        ];
        assert_eq!(DebugInfoBuilder::estimate_struct_size(&fields), 9);
        assert_eq!(
            DebugInfoBuilder::estimate_struct_size(&[]),
            0
        );
    }

    #[test]
    fn test_void_subprogram() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);

        // Void function (no return type).
        b.emit_subprogram("init", 0x2000, 0x40, false, None);
        b.emit_end_children(); // end subprogram children

        b.emit_end_children(); // end CU children
        b.end_compile_unit();

        let output = b.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_abbrev_builder_accessor() {
        let mut str_table = DebugStrTable::new();
        let b = make_builder(&mut str_table);

        // Verify the abbreviation builder is accessible.
        let abbrev = b.abbrev_builder();
        let cu_entry = abbrev.get_entry(b.cu_abbrev_code);
        assert!(cu_entry.is_some());
    }

    #[test]
    fn test_write_address_32bit() {
        let mut str_table = DebugStrTable::new();
        let mut b = DebugInfoBuilder::new(Target::I686, &mut str_table);
        b.data.clear();
        b.write_address(0xDEADBEEF);
        // i686 should write 4 bytes.
        assert_eq!(b.data.len(), 4);
        let addr = u32::from_le_bytes([b.data[0], b.data[1], b.data[2], b.data[3]]);
        assert_eq!(addr, 0xDEADBEEF);
    }

    #[test]
    fn test_write_address_64bit() {
        let mut str_table = DebugStrTable::new();
        let mut b = DebugInfoBuilder::new(Target::X86_64, &mut str_table);
        b.data.clear();
        b.write_address(0x0000_7FFF_DEAD_BEEF);
        // x86-64 should write 8 bytes.
        assert_eq!(b.data.len(), 8);
        let addr = u64::from_le_bytes([
            b.data[0], b.data[1], b.data[2], b.data[3], b.data[4], b.data[5], b.data[6],
            b.data[7],
        ]);
        assert_eq!(addr, 0x0000_7FFF_DEAD_BEEF);
    }

    // ===================================================================
    // Tests for High-Level CType Bridge Methods
    // ===================================================================

    /// Helper that creates a builder with an active compilation unit,
    /// ensuring that emitted DIEs receive non-zero CU-relative offsets
    /// (the CU header occupies the first 11+ bytes).
    fn make_builder_with_cu(str_table: &mut DebugStrTable) -> DebugInfoBuilder<'_> {
        let mut b = DebugInfoBuilder::new(Target::X86_64, str_table);
        b.begin_compile_unit("bcc", "/", "test.c", 0, 0, 0);
        b
    }

    #[test]
    fn test_emit_ctype_void() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let offset = b.emit_ctype_die(&CType::Void);
        assert_eq!(offset, 0, "void should return offset 0");
    }

    #[test]
    fn test_emit_ctype_bool() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Bool);
        assert!(off > 0, "Bool type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_signed_int() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Int { signed: true });
        assert!(off > 0);
        // Second call should return the same offset (deduplication).
        let off2 = b.emit_ctype_die(&CType::Int { signed: true });
        assert_eq!(off, off2, "Duplicate CType::Int should be deduplicated");
    }

    #[test]
    fn test_emit_ctype_unsigned_char() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Char { signed: false });
        assert!(off > 0);
    }

    #[test]
    fn test_emit_ctype_pointer() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Pointer(Box::new(CType::Int { signed: true })));
        assert!(off > 0, "Pointer type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_array() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        });
        assert!(off > 0, "Array type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_struct() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Struct {
            name: Some("point".to_string()),
            fields: vec![
                FieldDef {
                    name: Some("x".to_string()),
                    ty: CType::Int { signed: true },
                    bit_width: None,
                },
                FieldDef {
                    name: Some("y".to_string()),
                    ty: CType::Int { signed: true },
                    bit_width: None,
                },
            ],
        });
        assert!(off > 0, "Struct type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_union() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Union {
            name: Some("data".to_string()),
            fields: vec![
                FieldDef {
                    name: Some("i".to_string()),
                    ty: CType::Int { signed: true },
                    bit_width: None,
                },
                FieldDef {
                    name: Some("f".to_string()),
                    ty: CType::Float,
                    bit_width: None,
                },
            ],
        });
        assert!(off > 0, "Union type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_enum() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Enum {
            name: Some("color".to_string()),
            underlying: Box::new(CType::Int { signed: true }),
        });
        assert!(off > 0, "Enum type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_typedef() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Typedef {
            name: "size_t".to_string(),
            underlying: Box::new(CType::LongLong { signed: false }),
        });
        assert!(off > 0, "Typedef type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_atomic() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Atomic(Box::new(CType::Int { signed: true })));
        assert!(off > 0, "Atomic type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_complex() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Complex(Box::new(CType::Double)));
        assert!(off > 0, "Complex type should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_function() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::Function {
            return_type: Box::new(CType::Int { signed: true }),
            params: vec![CType::Int { signed: true }],
            variadic: false,
        });
        assert!(off > 0, "Function type → pointer should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ctype_long_double() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ctype_die(&CType::LongDouble);
        assert!(off > 0, "LongDouble type should produce a non-zero offset");
    }

    // ===================================================================
    // Tests for High-Level IrType Bridge Methods
    // ===================================================================

    #[test]
    fn test_emit_ir_type_void() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::Void);
        assert_eq!(off, 0, "IrType::Void should return offset 0");
    }

    #[test]
    fn test_emit_ir_type_i32() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::I32);
        assert!(off > 0, "IrType::I32 should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ir_type_ptr() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::Ptr);
        assert!(off > 0, "IrType::Ptr should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ir_type_f64() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::F64);
        assert!(off > 0, "IrType::F64 should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ir_type_f80() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::F80);
        assert!(off > 0, "IrType::F80 should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ir_type_array() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::Array {
            element: Box::new(IrType::I32),
            count: 16,
        });
        assert!(off > 0, "IrType::Array should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ir_type_struct() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::Struct {
            fields: vec![IrType::I32, IrType::I64],
            packed: false,
        });
        assert!(off > 0, "IrType::Struct should produce a non-zero offset");
    }

    #[test]
    fn test_emit_ir_type_function() {
        let mut str_table = DebugStrTable::new();
        let mut b = make_builder_with_cu(&mut str_table);
        let off = b.emit_ir_type_die(&IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::Ptr],
            is_variadic: false,
        });
        assert!(off > 0, "IrType::Function should produce a non-zero offset");
    }

    // ===================================================================
    // Tests for append_sleb128 static helper
    // ===================================================================

    #[test]
    fn test_append_sleb128_zero() {
        let mut buf = Vec::new();
        DebugInfoBuilder::append_sleb128(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_append_sleb128_positive() {
        let mut buf = Vec::new();
        DebugInfoBuilder::append_sleb128(&mut buf, 16);
        assert_eq!(buf, vec![0x10]);
    }

    #[test]
    fn test_append_sleb128_negative() {
        let mut buf = Vec::new();
        DebugInfoBuilder::append_sleb128(&mut buf, -1);
        assert_eq!(buf, vec![0x7f]);
    }

    #[test]
    fn test_append_sleb128_large_negative() {
        let mut buf = Vec::new();
        DebugInfoBuilder::append_sleb128(&mut buf, -128);
        // -128 in SLEB128: 0x80 with continuation, then 0x7f
        assert_eq!(buf, vec![0x80, 0x7f]);
    }

    // ===================================================================
    // Tests for compute_cu_range
    // ===================================================================

    #[test]
    fn test_compute_cu_range_empty() {
        let map: FxHashMap<String, (u64, u64)> = FxHashMap::default();
        let (lo, hi) = DebugInfoBuilder::compute_cu_range(&map);
        assert_eq!(lo, 0);
        assert_eq!(hi, 0);
    }

    #[test]
    fn test_compute_cu_range_single() {
        let mut map = FxHashMap::default();
        map.insert("main".to_string(), (0x1000, 0x100));
        let (lo, hi) = DebugInfoBuilder::compute_cu_range(&map);
        assert_eq!(lo, 0x1000);
        assert_eq!(hi, 0x1100);
    }

    #[test]
    fn test_compute_cu_range_multiple() {
        let mut map = FxHashMap::default();
        map.insert("main".to_string(), (0x1000, 0x100));
        map.insert("helper".to_string(), (0x800, 0x50));
        let (lo, hi) = DebugInfoBuilder::compute_cu_range(&map);
        assert_eq!(lo, 0x800);
        assert_eq!(hi, 0x1100);
    }

    // ===================================================================
    // Tests for generate_function_debug_info
    // ===================================================================

    #[test]
    fn test_generate_function_debug_info() {
        use crate::ir::function::{Parameter, IrFunction, Linkage};
        use crate::ir::types::IrType;
        use crate::ir::basic_block::{BasicBlock, BasicBlockId};
        use crate::ir::function::{CallingConvention, FunctionAttributes, ValueId};

        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        // Begin a CU first (required for generating subprogram DIEs).
        b.begin_compile_unit("bcc", "/tmp", "test.c", 0, 0x100, 0);

        // Create a minimal IrFunction.
        let func = IrFunction {
            name: "add".to_string(),
            return_type: IrType::I32,
            params: vec![
                Parameter {
                    name: Some("a".to_string()),
                    ty: IrType::I32,
                    id: ValueId(0),
                },
                Parameter {
                    name: Some("b".to_string()),
                    ty: IrType::I32,
                    id: ValueId(1),
                },
            ],
            basic_blocks: vec![BasicBlock::new(BasicBlockId(0), Some("entry".to_string()))],
            entry_block_id: BasicBlockId(0),
            calling_convention: CallingConvention::C,
            linkage: Linkage::External,
            is_variadic: false,
            attributes: FunctionAttributes::default(),
            local_values: Vec::new(),
            next_value_id: 2,
            alignment: 16,
            section: None,
            is_definition: true,
        };

        b.generate_function_debug_info(&func, 0x1000, 0x80);

        // Close the CU.
        b.emit_end_children();
        b.end_compile_unit();

        let data = b.finish();
        assert!(!data.is_empty(), "Generated data should not be empty");
    }

    #[test]
    fn test_generate_function_skips_declaration() {
        use crate::ir::function::{IrFunction, Linkage};
        use crate::ir::types::IrType;
        use crate::ir::basic_block::{BasicBlock, BasicBlockId};
        use crate::ir::function::{CallingConvention, FunctionAttributes};

        let mut str_table = DebugStrTable::new();
        let mut b = make_builder(&mut str_table);

        b.begin_compile_unit("bcc", "/tmp", "test.c", 0, 0, 0);

        // Non-definition function should be skipped.
        let func = IrFunction {
            name: "extern_fn".to_string(),
            return_type: IrType::I32,
            params: Vec::new(),
            basic_blocks: vec![BasicBlock::new(BasicBlockId(0), Some("entry".to_string()))],
            entry_block_id: BasicBlockId(0),
            calling_convention: CallingConvention::C,
            linkage: Linkage::External,
            is_variadic: false,
            attributes: FunctionAttributes::default(),
            local_values: Vec::new(),
            next_value_id: 0,
            alignment: 16,
            section: None,
            is_definition: false,
        };

        let size_before = b.section_size();
        b.generate_function_debug_info(&func, 0, 0);
        let size_after = b.section_size();

        // Nothing should have been emitted.
        assert_eq!(
            size_before, size_after,
            "Non-definition functions should not emit DIEs"
        );
    }
}
