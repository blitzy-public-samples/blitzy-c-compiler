//! DWARF v4 `.debug_abbrev` section generator.
//!
//! This module produces the abbreviation table that describes the structure of
//! Debug Information Entries (DIEs) in the `.debug_info` section. Each abbreviation
//! entry defines:
//!
//! - A tag (e.g., `DW_TAG_COMPILE_UNIT`, `DW_TAG_SUBPROGRAM`)
//! - Whether the DIE has child entries (`DW_CHILDREN_YES` / `DW_CHILDREN_NO`)
//! - A list of attribute specifications as `(DW_AT_*, DW_FORM_*)` pairs
//!
//! The abbreviation table is referenced by `.debug_info` entries via 1-based
//! ULEB128-encoded abbreviation codes. Without this table, debuggers cannot
//! interpret `.debug_info` section contents.
//!
//! This module is completely self-contained with zero external dependencies.
//! All DWARF constants are defined locally and all encoding is hand-implemented.

// ===========================================================================
// DWARF v4 Children Flag Constants
// ===========================================================================

/// Indicates that a DIE has child entries following it before the null terminator.
pub const DW_CHILDREN_YES: u8 = 0x01;

/// Indicates that a DIE has no child entries; the next sibling follows immediately.
pub const DW_CHILDREN_NO: u8 = 0x00;

// ===========================================================================
// DWARF v4 Tag Constants (DW_TAG_*)
//
// Each tag identifies the kind of entity described by a Debug Information Entry.
// Tags are encoded as ULEB128 values in the abbreviation table.
// ===========================================================================

/// Array type descriptor (`DW_TAG_array_type`).
pub const DW_TAG_ARRAY_TYPE: u16 = 0x01;

/// Enumeration type descriptor (`DW_TAG_enumeration_type`).
pub const DW_TAG_ENUMERATION_TYPE: u16 = 0x04;

/// Formal parameter of a subprogram (`DW_TAG_formal_parameter`).
pub const DW_TAG_FORMAL_PARAMETER: u16 = 0x05;

/// Member of a structure, union, or class (`DW_TAG_member`).
pub const DW_TAG_MEMBER: u16 = 0x0d;

/// Pointer type descriptor (`DW_TAG_pointer_type`).
pub const DW_TAG_POINTER_TYPE: u16 = 0x0f;

/// Compilation unit root DIE (`DW_TAG_compile_unit`).
pub const DW_TAG_COMPILE_UNIT: u16 = 0x11;

/// Structure (struct) type descriptor (`DW_TAG_structure_type`).
pub const DW_TAG_STRUCTURE_TYPE: u16 = 0x13;

/// Subroutine (function pointer) type descriptor (`DW_TAG_subroutine_type`).
pub const DW_TAG_SUBROUTINE_TYPE: u16 = 0x15;

/// Typedef alias (`DW_TAG_typedef`).
pub const DW_TAG_TYPEDEF: u16 = 0x16;

/// Union type descriptor (`DW_TAG_union_type`).
pub const DW_TAG_UNION_TYPE: u16 = 0x17;

/// Subrange type for array dimension bounds (`DW_TAG_subrange_type`).
pub const DW_TAG_SUBRANGE_TYPE: u16 = 0x21;

/// Base (primitive) type descriptor (`DW_TAG_base_type`).
pub const DW_TAG_BASE_TYPE: u16 = 0x24;

/// Const-qualified type wrapper (`DW_TAG_const_type`).
pub const DW_TAG_CONST_TYPE: u16 = 0x26;

/// Enumerator (named enum constant) (`DW_TAG_enumerator`).
pub const DW_TAG_ENUMERATOR: u16 = 0x28;

/// Subprogram (function definition or declaration) (`DW_TAG_subprogram`).
pub const DW_TAG_SUBPROGRAM: u16 = 0x2e;

/// Variable declaration or definition (`DW_TAG_variable`).
pub const DW_TAG_VARIABLE: u16 = 0x34;

/// Volatile-qualified type wrapper (`DW_TAG_volatile_type`).
pub const DW_TAG_VOLATILE_TYPE: u16 = 0x35;

// ===========================================================================
// DWARF v4 Attribute Constants (DW_AT_*)
//
// Attributes describe properties of entities. Each attribute is paired with
// a form that determines how its value is encoded in `.debug_info`.
// ===========================================================================

/// Location description for a variable or parameter (`DW_AT_location`).
pub const DW_AT_LOCATION: u16 = 0x02;

/// Name of the entity (`DW_AT_name`).
pub const DW_AT_NAME: u16 = 0x03;

/// Size of the entity in bytes (`DW_AT_byte_size`).
pub const DW_AT_BYTE_SIZE: u16 = 0x0b;

/// Offset into `.debug_line` for line number information (`DW_AT_stmt_list`).
pub const DW_AT_STMT_LIST: u16 = 0x10;

/// Lowest machine address of the entity (`DW_AT_low_pc`).
pub const DW_AT_LOW_PC: u16 = 0x11;

/// Highest machine address or address range length (`DW_AT_high_pc`).
pub const DW_AT_HIGH_PC: u16 = 0x12;

/// Source language identifier (`DW_AT_language`).
pub const DW_AT_LANGUAGE: u16 = 0x13;

/// Compilation directory path (`DW_AT_comp_dir`).
pub const DW_AT_COMP_DIR: u16 = 0x1b;

/// Constant value of an enumerator or const variable (`DW_AT_const_value`).
pub const DW_AT_CONST_VALUE: u16 = 0x1c;

/// Producer identification string (compiler name/version) (`DW_AT_producer`).
pub const DW_AT_PRODUCER: u16 = 0x25;

/// Upper bound of an array subrange (`DW_AT_upper_bound`).
pub const DW_AT_UPPER_BOUND: u16 = 0x2f;

/// Element count of an array subrange (`DW_AT_count`).
pub const DW_AT_COUNT: u16 = 0x37;

/// Byte offset of a member within its containing type (`DW_AT_data_member_location`).
pub const DW_AT_DATA_MEMBER_LOCATION: u16 = 0x38;

/// Source file index where the entity is declared (`DW_AT_decl_file`).
pub const DW_AT_DECL_FILE: u16 = 0x3a;

/// Source line number where the entity is declared (`DW_AT_decl_line`).
pub const DW_AT_DECL_LINE: u16 = 0x3b;

/// Base type encoding identifier (`DW_AT_encoding`).
pub const DW_AT_ENCODING: u16 = 0x3e;

/// External linkage visibility flag (`DW_AT_external`).
pub const DW_AT_EXTERNAL: u16 = 0x3f;

/// Reference to the type of the entity (`DW_AT_type`).
pub const DW_AT_TYPE: u16 = 0x49;

// ===========================================================================
// DWARF v4 Form Constants (DW_FORM_*)
//
// Forms describe how attribute values are encoded in the `.debug_info` section.
// Each form determines the byte layout and interpretation of the value.
// ===========================================================================

/// Target-address-sized value; width depends on compilation unit address size.
pub const DW_FORM_ADDR: u16 = 0x01;

/// 2-byte unsigned integer constant.
pub const DW_FORM_DATA2: u16 = 0x05;

/// 4-byte unsigned integer constant.
pub const DW_FORM_DATA4: u16 = 0x06;

/// 8-byte unsigned integer constant.
pub const DW_FORM_DATA8: u16 = 0x07;

/// 1-byte unsigned integer constant.
pub const DW_FORM_DATA1: u16 = 0x0b;

/// 1-byte boolean flag (0 = false, non-zero = true).
pub const DW_FORM_FLAG: u16 = 0x0c;

/// Signed LEB128 encoded integer constant.
pub const DW_FORM_SDATA: u16 = 0x0d;

/// 4-byte offset into the `.debug_str` section for string values.
pub const DW_FORM_STRP: u16 = 0x0e;

/// Unsigned LEB128 encoded integer constant.
pub const DW_FORM_UDATA: u16 = 0x0f;

/// 4-byte offset reference within `.debug_info` (compilation-unit-relative).
pub const DW_FORM_REF4: u16 = 0x13;

/// 4-byte offset for cross-section references (`.debug_line`, etc.).
pub const DW_FORM_SEC_OFFSET: u16 = 0x17;

/// Variable-length location expression block (ULEB128 length prefix + bytes).
pub const DW_FORM_EXPRLOC: u16 = 0x18;

/// Implicit boolean flag: presence in abbreviation means true, no data in DIE.
pub const DW_FORM_FLAG_PRESENT: u16 = 0x19;

// ===========================================================================
// Data Structures
// ===========================================================================

/// A single abbreviation entry in the DWARF abbreviation table.
///
/// Each entry describes the schema of a class of Debug Information Entries (DIEs):
/// which tag identifies the entity kind, whether it has child DIEs nested within
/// it, and which attribute specifications (name + encoding form pairs) it carries.
///
/// The `.debug_info` section references entries by their `code` field, which is
/// a 1-based sequential index assigned during registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbbrevEntry {
    /// Abbreviation code (1-based, assigned sequentially by `DebugAbbrevBuilder`).
    /// DIEs in `.debug_info` reference this code via ULEB128 encoding to declare
    /// which abbreviation schema they follow.
    pub code: u32,

    /// DWARF tag (`DW_TAG_*`) identifying the kind of entity this DIE describes.
    pub tag: u16,

    /// Whether DIEs using this abbreviation have child entries.
    /// `true` corresponds to `DW_CHILDREN_YES` (0x01), `false` to `DW_CHILDREN_NO` (0x00).
    pub has_children: bool,

    /// Ordered list of attribute specifications as `(DW_AT_*, DW_FORM_*)` pairs.
    /// Each pair declares one attribute name and the form (encoding) used for its
    /// value in the `.debug_info` section. The order here must match the order
    /// in which values appear in each DIE that uses this abbreviation.
    pub attributes: Vec<(u16, u16)>,
}

/// Builder for constructing the DWARF `.debug_abbrev` section.
///
/// The abbreviation table defines the schema for all DIEs that appear in the
/// `.debug_info` section. Each registered entry gets a unique 1-based code that
/// is referenced by DIEs to declare their structure.
///
/// # Usage
///
/// ```ignore
/// let mut builder = DebugAbbrevBuilder::new();
/// let cu_code = builder.add_compile_unit_abbrev();
/// let sub_code = builder.add_subprogram_abbrev(true);
/// let bytes = builder.finish();
/// // `bytes` contains the serialized .debug_abbrev section
/// ```
///
/// The builder provides both pre-defined templates for common DIE types and
/// a generic `add_entry()` method for custom abbreviation schemas.
pub struct DebugAbbrevBuilder {
    /// All registered abbreviation entries, in order of registration.
    entries: Vec<AbbrevEntry>,

    /// Next abbreviation code to assign. Starts at 1 and increments with each
    /// registered entry. Code 0 is reserved as the null entry terminator.
    next_code: u32,
}

// ===========================================================================
// ULEB128 Encoding Helpers
// ===========================================================================

/// Encode an unsigned 64-bit value as ULEB128 and append the bytes to `data`.
///
/// ULEB128 (Unsigned Little-Endian Base 128) encoding stores values in groups
/// of 7 bits per byte, with the high bit indicating continuation. This is the
/// standard encoding used throughout DWARF for variable-length integers.
fn write_uleb128(data: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // set continuation bit
        }
        data.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Compute the number of bytes required to encode a value as ULEB128.
///
/// Used by `section_size()` to calculate the section size without allocating
/// the actual byte buffer.
fn uleb128_size(mut value: u64) -> usize {
    let mut size = 0usize;
    loop {
        size += 1;
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    size
}

// ===========================================================================
// DebugAbbrevBuilder Implementation
// ===========================================================================

impl DebugAbbrevBuilder {
    /// Create a new, empty abbreviation table builder.
    ///
    /// The first entry registered will receive code 1. Code 0 is reserved by
    /// the DWARF specification as the null entry terminator.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_code: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Generic Entry Registration
    // -----------------------------------------------------------------------

    /// Register a new abbreviation entry with the given tag, children flag,
    /// and attribute specification list.
    ///
    /// Returns the assigned abbreviation code (1-based, incrementing). Each call
    /// to this method assigns a new unique code regardless of whether a
    /// structurally identical entry already exists. Callers that wish to reuse
    /// codes for duplicate schemas should track them externally.
    ///
    /// # Arguments
    ///
    /// * `tag` — DWARF tag constant (`DW_TAG_*`) for the DIE kind.
    /// * `has_children` — Whether DIEs using this abbreviation contain child entries.
    /// * `attributes` — Ordered list of `(DW_AT_*, DW_FORM_*)` pairs defining
    ///   the attributes and their encoding forms.
    ///
    /// # Returns
    ///
    /// The unique abbreviation code assigned to this entry.
    pub fn add_entry(
        &mut self,
        tag: u16,
        has_children: bool,
        attributes: Vec<(u16, u16)>,
    ) -> u32 {
        let code = self.next_code;
        self.entries.push(AbbrevEntry {
            code,
            tag,
            has_children,
            attributes,
        });
        self.next_code += 1;
        code
    }

    // -----------------------------------------------------------------------
    // Pre-defined Abbreviation Templates
    // -----------------------------------------------------------------------

    /// Register a standard compilation unit abbreviation.
    ///
    /// This is the root DIE for each compilation unit in `.debug_info`, containing
    /// producer identification, source language, file name, compilation directory,
    /// address range, and a reference to the `.debug_line` section.
    ///
    /// Tag: `DW_TAG_COMPILE_UNIT` with children (subprograms, variables, types).
    ///
    /// Attributes (in order):
    /// - `DW_AT_PRODUCER`  → `DW_FORM_STRP`       (compiler identification string)
    /// - `DW_AT_LANGUAGE`  → `DW_FORM_DATA2`       (source language code)
    /// - `DW_AT_NAME`      → `DW_FORM_STRP`        (source file name)
    /// - `DW_AT_COMP_DIR`  → `DW_FORM_STRP`        (compilation directory)
    /// - `DW_AT_LOW_PC`    → `DW_FORM_ADDR`         (start address)
    /// - `DW_AT_HIGH_PC`   → `DW_FORM_DATA8`        (address range length)
    /// - `DW_AT_STMT_LIST` → `DW_FORM_SEC_OFFSET`   (offset into .debug_line)
    pub fn add_compile_unit_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_COMPILE_UNIT,
            true, // has children: subprograms, variables, types
            vec![
                (DW_AT_PRODUCER, DW_FORM_STRP),
                (DW_AT_LANGUAGE, DW_FORM_DATA2),
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_COMP_DIR, DW_FORM_STRP),
                (DW_AT_LOW_PC, DW_FORM_ADDR),
                (DW_AT_HIGH_PC, DW_FORM_DATA8),
                (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
            ],
        )
    }

    /// Register a standard subprogram (function) abbreviation.
    ///
    /// Subprograms have children (formal parameters, local variables) and include
    /// name, address range, external visibility, and optionally a return type reference.
    ///
    /// # Arguments
    ///
    /// * `has_type` — If `true`, includes `DW_AT_TYPE` for non-void return types.
    ///   If `false`, omits the type attribute (used for `void` functions).
    ///
    /// Tag: `DW_TAG_SUBPROGRAM` with children.
    ///
    /// Attributes (in order):
    /// - `DW_AT_NAME`     → `DW_FORM_STRP`   (function name)
    /// - `DW_AT_LOW_PC`   → `DW_FORM_ADDR`    (start address)
    /// - `DW_AT_HIGH_PC`  → `DW_FORM_DATA8`   (address range length)
    /// - `DW_AT_EXTERNAL` → `DW_FORM_FLAG`    (external linkage boolean)
    /// - (optional) `DW_AT_TYPE` → `DW_FORM_REF4` (return type reference)
    /// - `DW_AT_DECL_FILE` → `DW_FORM_UDATA`  (source file index)
    /// - `DW_AT_DECL_LINE` → `DW_FORM_UDATA`  (source line number)
    pub fn add_subprogram_abbrev(&mut self, has_type: bool) -> u32 {
        let mut attrs = vec![
            (DW_AT_NAME, DW_FORM_STRP),
            (DW_AT_LOW_PC, DW_FORM_ADDR),
            (DW_AT_HIGH_PC, DW_FORM_DATA8),
            (DW_AT_EXTERNAL, DW_FORM_FLAG),
        ];
        if has_type {
            attrs.push((DW_AT_TYPE, DW_FORM_REF4));
        }
        attrs.push((DW_AT_DECL_FILE, DW_FORM_UDATA));
        attrs.push((DW_AT_DECL_LINE, DW_FORM_UDATA));
        self.add_entry(DW_TAG_SUBPROGRAM, true, attrs)
    }

    /// Register a standard variable abbreviation.
    ///
    /// Used for local variables and file-scope variables. Variables carry a name,
    /// a type reference, and a location expression describing where the variable
    /// resides at runtime (e.g., frame-relative offset via `DW_OP_fbreg`).
    ///
    /// Tag: `DW_TAG_VARIABLE` with no children.
    ///
    /// Attributes (in order):
    /// - `DW_AT_NAME`     → `DW_FORM_STRP`    (variable name)
    /// - `DW_AT_TYPE`     → `DW_FORM_REF4`    (type reference)
    /// - `DW_AT_LOCATION` → `DW_FORM_EXPRLOC` (location expression)
    pub fn add_variable_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_VARIABLE,
            false, // no children
            vec![
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_TYPE, DW_FORM_REF4),
                (DW_AT_LOCATION, DW_FORM_EXPRLOC),
            ],
        )
    }

    /// Register a standard formal parameter abbreviation.
    ///
    /// Used for function parameters. Each parameter carries a name, type reference,
    /// and location expression for its runtime position (register or stack slot).
    ///
    /// Tag: `DW_TAG_FORMAL_PARAMETER` with no children.
    ///
    /// Attributes (in order):
    /// - `DW_AT_NAME`     → `DW_FORM_STRP`    (parameter name)
    /// - `DW_AT_TYPE`     → `DW_FORM_REF4`    (type reference)
    /// - `DW_AT_LOCATION` → `DW_FORM_EXPRLOC` (location expression)
    pub fn add_formal_parameter_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_FORMAL_PARAMETER,
            false, // no children
            vec![
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_TYPE, DW_FORM_REF4),
                (DW_AT_LOCATION, DW_FORM_EXPRLOC),
            ],
        )
    }

    /// Register a standard base (primitive) type abbreviation.
    ///
    /// Base types describe fundamental C types (int, char, float, etc.) with
    /// a name, byte size, and encoding classification.
    ///
    /// Tag: `DW_TAG_BASE_TYPE` with no children.
    ///
    /// Attributes (in order):
    /// - `DW_AT_NAME`     → `DW_FORM_STRP`  (type name, e.g., "int")
    /// - `DW_AT_BYTE_SIZE` → `DW_FORM_DATA1` (size in bytes)
    /// - `DW_AT_ENCODING` → `DW_FORM_DATA1`  (DW_ATE_* encoding class)
    pub fn add_base_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_BASE_TYPE,
            false, // no children
            vec![
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_BYTE_SIZE, DW_FORM_DATA1),
                (DW_AT_ENCODING, DW_FORM_DATA1),
            ],
        )
    }

    /// Register a standard pointer type abbreviation.
    ///
    /// Pointer types reference the pointee type and specify the pointer's byte size
    /// (4 for 32-bit, 8 for 64-bit targets).
    ///
    /// Tag: `DW_TAG_POINTER_TYPE` with no children.
    ///
    /// Attributes (in order):
    /// - `DW_AT_BYTE_SIZE` → `DW_FORM_DATA1` (pointer size in bytes)
    /// - `DW_AT_TYPE`     → `DW_FORM_REF4`   (pointee type reference)
    pub fn add_pointer_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_POINTER_TYPE,
            false, // no children
            vec![
                (DW_AT_BYTE_SIZE, DW_FORM_DATA1),
                (DW_AT_TYPE, DW_FORM_REF4),
            ],
        )
    }

    /// Register a standard structure type abbreviation.
    ///
    /// Structure types have children (member DIEs) and carry a name, byte size,
    /// and source declaration location.
    ///
    /// Tag: `DW_TAG_STRUCTURE_TYPE` with children (members).
    ///
    /// Attributes (in order):
    /// - `DW_AT_NAME`      → `DW_FORM_STRP`  (struct name)
    /// - `DW_AT_BYTE_SIZE` → `DW_FORM_UDATA` (total size in bytes)
    /// - `DW_AT_DECL_FILE` → `DW_FORM_UDATA` (source file index)
    /// - `DW_AT_DECL_LINE` → `DW_FORM_UDATA` (source line number)
    pub fn add_struct_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_STRUCTURE_TYPE,
            true, // has children: member DIEs
            vec![
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_BYTE_SIZE, DW_FORM_UDATA),
                (DW_AT_DECL_FILE, DW_FORM_UDATA),
                (DW_AT_DECL_LINE, DW_FORM_UDATA),
            ],
        )
    }

    /// Register a standard struct/union member abbreviation.
    ///
    /// Members carry a name, type reference, and byte offset within the containing
    /// structure or union.
    ///
    /// Tag: `DW_TAG_MEMBER` with no children.
    ///
    /// Attributes (in order):
    /// - `DW_AT_NAME`                 → `DW_FORM_STRP`  (member name)
    /// - `DW_AT_TYPE`                 → `DW_FORM_REF4`  (member type reference)
    /// - `DW_AT_DATA_MEMBER_LOCATION` → `DW_FORM_UDATA` (byte offset in parent)
    pub fn add_member_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_MEMBER,
            false, // no children
            vec![
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_TYPE, DW_FORM_REF4),
                (DW_AT_DATA_MEMBER_LOCATION, DW_FORM_UDATA),
            ],
        )
    }

    /// Register a standard array type abbreviation.
    ///
    /// Array types reference their element type and have children (subrange DIEs
    /// that describe dimension bounds). The caller should separately register
    /// a subrange abbreviation via `add_entry()` with `DW_TAG_SUBRANGE_TYPE`.
    ///
    /// Tag: `DW_TAG_ARRAY_TYPE` with children (subrange DIEs).
    ///
    /// Attributes (in order):
    /// - `DW_AT_TYPE` → `DW_FORM_REF4` (element type reference)
    pub fn add_array_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_ARRAY_TYPE,
            true, // has children: subrange DIEs for dimension bounds
            vec![(DW_AT_TYPE, DW_FORM_REF4)],
        )
    }

    // -----------------------------------------------------------------------
    // Lookup and Introspection
    // -----------------------------------------------------------------------

    /// Look up an abbreviation entry by its assigned code.
    ///
    /// Returns `None` if no entry with the given code has been registered.
    /// Codes are 1-based; code 0 is the reserved null terminator.
    pub fn get_entry(&self, code: u32) -> Option<&AbbrevEntry> {
        if code == 0 || code > self.entries.len() as u32 {
            return None;
        }
        // Codes are 1-based and assigned sequentially, so code N is at index N-1.
        Some(&self.entries[(code - 1) as usize])
    }

    /// Return the number of abbreviation entries registered in this table.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    // -----------------------------------------------------------------------
    // Section Serialization
    // -----------------------------------------------------------------------

    /// Serialize the complete `.debug_abbrev` section to raw bytes.
    ///
    /// The output format follows the DWARF v4 specification §7.5.3:
    ///
    /// For each registered entry:
    /// 1. Abbreviation code (ULEB128)
    /// 2. Tag (ULEB128)
    /// 3. Children flag (1 byte: `DW_CHILDREN_YES` or `DW_CHILDREN_NO`)
    /// 4. For each attribute specification:
    ///    a. Attribute name (ULEB128, `DW_AT_*`)
    ///    b. Attribute form (ULEB128, `DW_FORM_*`)
    /// 5. Attribute list terminator: two zero bytes `(0, 0)`
    ///
    /// After all entries, a single zero byte terminates the abbreviation table
    /// (the null abbreviation entry with code 0).
    pub fn finish(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.section_size());

        for entry in &self.entries {
            // Write abbreviation code (1-based ULEB128)
            write_uleb128(&mut data, entry.code as u64);

            // Write tag (ULEB128)
            write_uleb128(&mut data, entry.tag as u64);

            // Write children flag (1 byte)
            data.push(if entry.has_children {
                DW_CHILDREN_YES
            } else {
                DW_CHILDREN_NO
            });

            // Write attribute specifications
            for &(attr_name, attr_form) in &entry.attributes {
                write_uleb128(&mut data, attr_name as u64);
                write_uleb128(&mut data, attr_form as u64);
            }

            // Terminate attribute list with (0, 0) sentinel pair
            data.push(0x00);
            data.push(0x00);
        }

        // Terminate the abbreviation table with null entry (code 0)
        data.push(0x00);

        data
    }

    /// Compute the byte size of the serialized `.debug_abbrev` section without
    /// performing the actual serialization.
    ///
    /// This is useful for pre-allocating buffers or calculating section offsets
    /// in the ELF file before the abbreviation table has been serialized.
    pub fn section_size(&self) -> usize {
        let mut size = 0usize;

        for entry in &self.entries {
            // Abbreviation code (ULEB128)
            size += uleb128_size(entry.code as u64);
            // Tag (ULEB128)
            size += uleb128_size(entry.tag as u64);
            // Children flag (1 byte)
            size += 1;
            // Attribute specifications
            for &(attr_name, attr_form) in &entry.attributes {
                size += uleb128_size(attr_name as u64);
                size += uleb128_size(attr_form as u64);
            }
            // Attribute list terminator (0, 0) — 2 bytes
            size += 2;
        }

        // Null abbreviation entry terminator — 1 byte
        size += 1;

        size
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_builder_is_empty() {
        let builder = DebugAbbrevBuilder::new();
        assert_eq!(builder.entry_count(), 0);
        assert!(builder.get_entry(0).is_none());
        assert!(builder.get_entry(1).is_none());
    }

    #[test]
    fn test_add_entry_assigns_sequential_codes() {
        let mut builder = DebugAbbrevBuilder::new();
        let code1 = builder.add_entry(DW_TAG_COMPILE_UNIT, true, vec![]);
        let code2 = builder.add_entry(DW_TAG_SUBPROGRAM, true, vec![]);
        let code3 = builder.add_entry(DW_TAG_VARIABLE, false, vec![]);

        assert_eq!(code1, 1);
        assert_eq!(code2, 2);
        assert_eq!(code3, 3);
        assert_eq!(builder.entry_count(), 3);
    }

    #[test]
    fn test_get_entry_by_code() {
        let mut builder = DebugAbbrevBuilder::new();
        builder.add_entry(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_NAME, DW_FORM_STRP)],
        );
        builder.add_entry(DW_TAG_VARIABLE, false, vec![]);

        let entry1 = builder.get_entry(1).unwrap();
        assert_eq!(entry1.code, 1);
        assert_eq!(entry1.tag, DW_TAG_COMPILE_UNIT);
        assert!(entry1.has_children);
        assert_eq!(entry1.attributes.len(), 1);
        assert_eq!(entry1.attributes[0], (DW_AT_NAME, DW_FORM_STRP));

        let entry2 = builder.get_entry(2).unwrap();
        assert_eq!(entry2.code, 2);
        assert_eq!(entry2.tag, DW_TAG_VARIABLE);
        assert!(!entry2.has_children);
        assert!(entry2.attributes.is_empty());

        assert!(builder.get_entry(0).is_none());
        assert!(builder.get_entry(3).is_none());
    }

    #[test]
    fn test_compile_unit_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_compile_unit_abbrev();
        assert_eq!(code, 1);

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_COMPILE_UNIT);
        assert!(entry.has_children);
        assert_eq!(entry.attributes.len(), 7);

        // Verify exact attribute order and forms per specification
        assert_eq!(entry.attributes[0], (DW_AT_PRODUCER, DW_FORM_STRP));
        assert_eq!(entry.attributes[1], (DW_AT_LANGUAGE, DW_FORM_DATA2));
        assert_eq!(entry.attributes[2], (DW_AT_NAME, DW_FORM_STRP));
        assert_eq!(entry.attributes[3], (DW_AT_COMP_DIR, DW_FORM_STRP));
        assert_eq!(entry.attributes[4], (DW_AT_LOW_PC, DW_FORM_ADDR));
        assert_eq!(entry.attributes[5], (DW_AT_HIGH_PC, DW_FORM_DATA8));
        assert_eq!(entry.attributes[6], (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET));
    }

    #[test]
    fn test_subprogram_abbrev_with_type() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_subprogram_abbrev(true);

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_SUBPROGRAM);
        assert!(entry.has_children);
        // Should contain DW_AT_TYPE
        assert!(
            entry
                .attributes
                .iter()
                .any(|&(at, _)| at == DW_AT_TYPE)
        );
        assert_eq!(entry.attributes.len(), 7);
    }

    #[test]
    fn test_subprogram_abbrev_without_type() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_subprogram_abbrev(false);

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_SUBPROGRAM);
        assert!(entry.has_children);
        // Should NOT contain DW_AT_TYPE
        assert!(
            !entry
                .attributes
                .iter()
                .any(|&(at, _)| at == DW_AT_TYPE)
        );
        assert_eq!(entry.attributes.len(), 6);
    }

    #[test]
    fn test_variable_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_variable_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_VARIABLE);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 3);
        assert_eq!(entry.attributes[0], (DW_AT_NAME, DW_FORM_STRP));
        assert_eq!(entry.attributes[1], (DW_AT_TYPE, DW_FORM_REF4));
        assert_eq!(entry.attributes[2], (DW_AT_LOCATION, DW_FORM_EXPRLOC));
    }

    #[test]
    fn test_formal_parameter_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_formal_parameter_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_FORMAL_PARAMETER);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 3);
        assert_eq!(entry.attributes[0], (DW_AT_NAME, DW_FORM_STRP));
        assert_eq!(entry.attributes[1], (DW_AT_TYPE, DW_FORM_REF4));
        assert_eq!(entry.attributes[2], (DW_AT_LOCATION, DW_FORM_EXPRLOC));
    }

    #[test]
    fn test_base_type_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_base_type_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_BASE_TYPE);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 3);
        assert_eq!(entry.attributes[0], (DW_AT_NAME, DW_FORM_STRP));
        assert_eq!(entry.attributes[1], (DW_AT_BYTE_SIZE, DW_FORM_DATA1));
        assert_eq!(entry.attributes[2], (DW_AT_ENCODING, DW_FORM_DATA1));
    }

    #[test]
    fn test_pointer_type_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_pointer_type_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_POINTER_TYPE);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 2);
        assert_eq!(entry.attributes[0], (DW_AT_BYTE_SIZE, DW_FORM_DATA1));
        assert_eq!(entry.attributes[1], (DW_AT_TYPE, DW_FORM_REF4));
    }

    #[test]
    fn test_struct_type_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_struct_type_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_STRUCTURE_TYPE);
        assert!(entry.has_children);
        assert_eq!(entry.attributes.len(), 4);
        assert_eq!(entry.attributes[0], (DW_AT_NAME, DW_FORM_STRP));
        assert_eq!(entry.attributes[1], (DW_AT_BYTE_SIZE, DW_FORM_UDATA));
        assert_eq!(entry.attributes[2], (DW_AT_DECL_FILE, DW_FORM_UDATA));
        assert_eq!(entry.attributes[3], (DW_AT_DECL_LINE, DW_FORM_UDATA));
    }

    #[test]
    fn test_member_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_member_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_MEMBER);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 3);
        assert_eq!(entry.attributes[0], (DW_AT_NAME, DW_FORM_STRP));
        assert_eq!(entry.attributes[1], (DW_AT_TYPE, DW_FORM_REF4));
        assert_eq!(
            entry.attributes[2],
            (DW_AT_DATA_MEMBER_LOCATION, DW_FORM_UDATA)
        );
    }

    #[test]
    fn test_array_type_abbrev_template() {
        let mut builder = DebugAbbrevBuilder::new();
        let code = builder.add_array_type_abbrev();

        let entry = builder.get_entry(code).unwrap();
        assert_eq!(entry.tag, DW_TAG_ARRAY_TYPE);
        assert!(entry.has_children);
        assert_eq!(entry.attributes.len(), 1);
        assert_eq!(entry.attributes[0], (DW_AT_TYPE, DW_FORM_REF4));
    }

    #[test]
    fn test_uleb128_encoding() {
        // Value 0 → single byte 0x00
        let mut data = Vec::new();
        write_uleb128(&mut data, 0);
        assert_eq!(data, vec![0x00]);

        // Value 1 → single byte 0x01
        let mut data = Vec::new();
        write_uleb128(&mut data, 1);
        assert_eq!(data, vec![0x01]);

        // Value 127 (0x7f) → single byte 0x7F
        let mut data = Vec::new();
        write_uleb128(&mut data, 127);
        assert_eq!(data, vec![0x7F]);

        // Value 128 (0x80) → two bytes: 0x80, 0x01
        let mut data = Vec::new();
        write_uleb128(&mut data, 128);
        assert_eq!(data, vec![0x80, 0x01]);

        // Value 624485 → three bytes per DWARF spec example
        let mut data = Vec::new();
        write_uleb128(&mut data, 624485);
        assert_eq!(data, vec![0xE5, 0x8E, 0x26]);

        // Value 16384 → three bytes: 0x80, 0x80, 0x01
        let mut data = Vec::new();
        write_uleb128(&mut data, 16384);
        assert_eq!(data, vec![0x80, 0x80, 0x01]);
    }

    #[test]
    fn test_uleb128_size_calculation() {
        assert_eq!(uleb128_size(0), 1);
        assert_eq!(uleb128_size(1), 1);
        assert_eq!(uleb128_size(127), 1);
        assert_eq!(uleb128_size(128), 2);
        assert_eq!(uleb128_size(16383), 2);
        assert_eq!(uleb128_size(16384), 3);
        assert_eq!(uleb128_size(624485), 3);
    }

    #[test]
    fn test_empty_table_serialization() {
        let builder = DebugAbbrevBuilder::new();
        let data = builder.finish();
        // Empty table should only contain the null terminator byte
        assert_eq!(data, vec![0x00]);
        assert_eq!(builder.section_size(), 1);
    }

    #[test]
    fn test_section_size_matches_finish_length() {
        let mut builder = DebugAbbrevBuilder::new();
        builder.add_compile_unit_abbrev();
        builder.add_subprogram_abbrev(true);
        builder.add_subprogram_abbrev(false);
        builder.add_variable_abbrev();
        builder.add_formal_parameter_abbrev();
        builder.add_base_type_abbrev();
        builder.add_pointer_type_abbrev();
        builder.add_struct_type_abbrev();
        builder.add_member_abbrev();
        builder.add_array_type_abbrev();

        let data = builder.finish();
        assert_eq!(data.len(), builder.section_size());
    }

    #[test]
    fn test_finish_format_single_entry() {
        let mut builder = DebugAbbrevBuilder::new();
        builder.add_entry(
            DW_TAG_BASE_TYPE,       // tag = 0x24
            false,                  // no children
            vec![(DW_AT_NAME, DW_FORM_STRP)], // one attribute pair
        );

        let data = builder.finish();

        // Expected format:
        // [0] code=1 (ULEB128: 0x01)
        // [1] tag=0x24 (ULEB128: 0x24)
        // [2] has_children=0 (DW_CHILDREN_NO)
        // [3] DW_AT_NAME=0x03 (ULEB128: 0x03)
        // [4] DW_FORM_STRP=0x0e (ULEB128: 0x0e)
        // [5] attribute terminator 0x00
        // [6] attribute terminator 0x00
        // [7] null abbreviation entry 0x00
        assert_eq!(
            data,
            vec![
                0x01, // code = 1
                0x24, // tag = DW_TAG_BASE_TYPE
                0x00, // DW_CHILDREN_NO
                0x03, // DW_AT_NAME
                0x0e, // DW_FORM_STRP
                0x00, // attr terminator (name)
                0x00, // attr terminator (form)
                0x00, // null abbreviation entry
            ]
        );
    }

    #[test]
    fn test_finish_format_multiple_entries() {
        let mut builder = DebugAbbrevBuilder::new();

        // Entry 1: compile unit with one attribute
        builder.add_entry(
            DW_TAG_COMPILE_UNIT,
            true,
            vec![(DW_AT_PRODUCER, DW_FORM_STRP)],
        );

        // Entry 2: variable with no attributes
        builder.add_entry(DW_TAG_VARIABLE, false, vec![]);

        let data = builder.finish();

        // Entry 1: code=1, tag=0x11, children=1, (0x25, 0x0e), (0, 0)
        // Entry 2: code=2, tag=0x34, children=0, (0, 0)
        // Null terminator: 0
        assert_eq!(
            data,
            vec![
                // Entry 1
                0x01, // code = 1
                0x11, // tag = DW_TAG_COMPILE_UNIT
                0x01, // DW_CHILDREN_YES
                0x25, // DW_AT_PRODUCER
                0x0e, // DW_FORM_STRP
                0x00, // attr terminator
                0x00, // attr terminator
                // Entry 2
                0x02, // code = 2
                0x34, // tag = DW_TAG_VARIABLE
                0x00, // DW_CHILDREN_NO
                0x00, // attr terminator
                0x00, // attr terminator
                // Null terminator
                0x00,
            ]
        );
    }

    #[test]
    fn test_finish_with_multi_byte_uleb128_tag() {
        let mut builder = DebugAbbrevBuilder::new();
        // Use a custom tag value > 127 to test multi-byte ULEB128 encoding of tags.
        // DW_TAG_volatile_type = 0x35 fits in one byte, so use a hypothetical large tag.
        // Actually, let's test with a real scenario: after 127+ entries, the code
        // field will need multi-byte ULEB128. Instead, test with attribute value > 127.
        // DW_AT_TYPE = 0x49 = 73 (fits in 1 byte), but no standard attribute > 127.
        // However, the ULEB128 encoder should handle any value. Test via code > 127.
        let mut builder = DebugAbbrevBuilder::new();
        // Add 127 dummy entries to push next_code to 128
        for _ in 0..127 {
            builder.add_entry(DW_TAG_BASE_TYPE, false, vec![]);
        }
        // Entry 128 should have code encoded as 2-byte ULEB128
        let code128 = builder.add_entry(DW_TAG_VARIABLE, false, vec![]);
        assert_eq!(code128, 128);

        let entry = builder.get_entry(128).unwrap();
        assert_eq!(entry.code, 128);
        assert_eq!(entry.tag, DW_TAG_VARIABLE);

        // Verify section_size still matches
        let data = builder.finish();
        assert_eq!(data.len(), builder.section_size());
    }

    #[test]
    fn test_dwarf_constant_values() {
        // Verify all constant values match the DWARF v4 specification exactly
        assert_eq!(DW_CHILDREN_YES, 0x01);
        assert_eq!(DW_CHILDREN_NO, 0x00);

        // Tags
        assert_eq!(DW_TAG_ARRAY_TYPE, 0x01);
        assert_eq!(DW_TAG_ENUMERATION_TYPE, 0x04);
        assert_eq!(DW_TAG_FORMAL_PARAMETER, 0x05);
        assert_eq!(DW_TAG_MEMBER, 0x0d);
        assert_eq!(DW_TAG_POINTER_TYPE, 0x0f);
        assert_eq!(DW_TAG_COMPILE_UNIT, 0x11);
        assert_eq!(DW_TAG_STRUCTURE_TYPE, 0x13);
        assert_eq!(DW_TAG_SUBROUTINE_TYPE, 0x15);
        assert_eq!(DW_TAG_TYPEDEF, 0x16);
        assert_eq!(DW_TAG_UNION_TYPE, 0x17);
        assert_eq!(DW_TAG_SUBRANGE_TYPE, 0x21);
        assert_eq!(DW_TAG_BASE_TYPE, 0x24);
        assert_eq!(DW_TAG_CONST_TYPE, 0x26);
        assert_eq!(DW_TAG_ENUMERATOR, 0x28);
        assert_eq!(DW_TAG_SUBPROGRAM, 0x2e);
        assert_eq!(DW_TAG_VARIABLE, 0x34);
        assert_eq!(DW_TAG_VOLATILE_TYPE, 0x35);

        // Attributes
        assert_eq!(DW_AT_LOCATION, 0x02);
        assert_eq!(DW_AT_NAME, 0x03);
        assert_eq!(DW_AT_BYTE_SIZE, 0x0b);
        assert_eq!(DW_AT_STMT_LIST, 0x10);
        assert_eq!(DW_AT_LOW_PC, 0x11);
        assert_eq!(DW_AT_HIGH_PC, 0x12);
        assert_eq!(DW_AT_LANGUAGE, 0x13);
        assert_eq!(DW_AT_COMP_DIR, 0x1b);
        assert_eq!(DW_AT_CONST_VALUE, 0x1c);
        assert_eq!(DW_AT_PRODUCER, 0x25);
        assert_eq!(DW_AT_UPPER_BOUND, 0x2f);
        assert_eq!(DW_AT_COUNT, 0x37);
        assert_eq!(DW_AT_DATA_MEMBER_LOCATION, 0x38);
        assert_eq!(DW_AT_DECL_FILE, 0x3a);
        assert_eq!(DW_AT_DECL_LINE, 0x3b);
        assert_eq!(DW_AT_ENCODING, 0x3e);
        assert_eq!(DW_AT_EXTERNAL, 0x3f);
        assert_eq!(DW_AT_TYPE, 0x49);

        // Forms
        assert_eq!(DW_FORM_ADDR, 0x01);
        assert_eq!(DW_FORM_DATA2, 0x05);
        assert_eq!(DW_FORM_DATA4, 0x06);
        assert_eq!(DW_FORM_DATA8, 0x07);
        assert_eq!(DW_FORM_DATA1, 0x0b);
        assert_eq!(DW_FORM_FLAG, 0x0c);
        assert_eq!(DW_FORM_SDATA, 0x0d);
        assert_eq!(DW_FORM_STRP, 0x0e);
        assert_eq!(DW_FORM_UDATA, 0x0f);
        assert_eq!(DW_FORM_REF4, 0x13);
        assert_eq!(DW_FORM_SEC_OFFSET, 0x17);
        assert_eq!(DW_FORM_EXPRLOC, 0x18);
        assert_eq!(DW_FORM_FLAG_PRESENT, 0x19);
    }

    #[test]
    fn test_full_typical_table() {
        // Build a realistic abbreviation table matching typical compiler output
        let mut builder = DebugAbbrevBuilder::new();

        let cu_code = builder.add_compile_unit_abbrev();
        let sub_typed = builder.add_subprogram_abbrev(true);
        let sub_void = builder.add_subprogram_abbrev(false);
        let var_code = builder.add_variable_abbrev();
        let param_code = builder.add_formal_parameter_abbrev();
        let base_code = builder.add_base_type_abbrev();
        let ptr_code = builder.add_pointer_type_abbrev();
        let struct_code = builder.add_struct_type_abbrev();
        let member_code = builder.add_member_abbrev();
        let array_code = builder.add_array_type_abbrev();

        // Codes should be sequential 1..=10
        assert_eq!(cu_code, 1);
        assert_eq!(sub_typed, 2);
        assert_eq!(sub_void, 3);
        assert_eq!(var_code, 4);
        assert_eq!(param_code, 5);
        assert_eq!(base_code, 6);
        assert_eq!(ptr_code, 7);
        assert_eq!(struct_code, 8);
        assert_eq!(member_code, 9);
        assert_eq!(array_code, 10);

        assert_eq!(builder.entry_count(), 10);

        // Serialize and verify it's well-formed (non-zero length, ends with 0x00)
        let data = builder.finish();
        assert!(data.len() > 1);
        assert_eq!(*data.last().unwrap(), 0x00);
        assert_eq!(data.len(), builder.section_size());
    }
}
