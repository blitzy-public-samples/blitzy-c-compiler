//! # `.debug_abbrev` Section Generation
//!
//! Generates the DWARF v4 abbreviation table for BCC. The abbreviation table defines
//! templates for Debug Information Entries (DIEs) in `.debug_info`.
//!
//! Each abbreviation specifies:
//! - A unique code (1-based, referenced by DIEs in `.debug_info`)
//! - A DWARF tag (`DW_TAG_*`)
//! - Whether the DIE has children
//! - An ordered list of (attribute, form) pairs
//!
//! The `.debug_info` section references abbreviation codes instead of repeating full
//! tag/attribute specifications for every DIE, achieving significant size compression.
//!
//! The table is terminated by a null abbreviation code (0 byte).
//! Each entry's attribute list is terminated by a (0, 0) pair.
//!
//! ## Binary Format
//! ```text
//! [ULEB128 code] [ULEB128 tag] [u8 children] {[ULEB128 attr] [ULEB128 form]}* [0] [0]
//! ...
//! [0]  // null terminator
//! ```
//!
//! ## References
//! - DWARF v4, Section 7.5.3 (Abbreviation Tables)
//! - DWARF v4, Table 7.1 (Tag encodings)
//! - DWARF v4, Table 7.5 (Attribute encodings)
//! - DWARF v4, Table 7.6 (Attribute form encodings)

// ============================================================================
// DWARF v4 Tag Constants (Table 7.1)
// ============================================================================

/// Tag for a compilation unit DIE — the top-level DIE in `.debug_info`.
pub const DW_TAG_COMPILE_UNIT: u16 = 0x11;

/// Tag for a subprogram (function) DIE.
pub const DW_TAG_SUBPROGRAM: u16 = 0x2e;

/// Tag for a variable DIE (local or global).
pub const DW_TAG_VARIABLE: u16 = 0x34;

/// Tag for a formal parameter DIE (function parameter).
pub const DW_TAG_FORMAL_PARAMETER: u16 = 0x05;

/// Tag for a base (primitive) type DIE (int, char, float, etc.).
pub const DW_TAG_BASE_TYPE: u16 = 0x24;

/// Tag for a pointer type DIE.
pub const DW_TAG_POINTER_TYPE: u16 = 0x0f;

/// Tag for a typedef DIE.
pub const DW_TAG_TYPEDEF: u16 = 0x16;

/// Tag for a structure (struct) type DIE.
pub const DW_TAG_STRUCTURE_TYPE: u16 = 0x13;

/// Tag for a union type DIE.
pub const DW_TAG_UNION_TYPE: u16 = 0x17;

/// Tag for an enumeration type DIE.
pub const DW_TAG_ENUMERATION_TYPE: u16 = 0x04;

/// Tag for an enumerator (enum constant) DIE.
pub const DW_TAG_ENUMERATOR: u16 = 0x28;

/// Tag for a struct/union member DIE.
pub const DW_TAG_MEMBER: u16 = 0x0d;

/// Tag for an array type DIE.
pub const DW_TAG_ARRAY_TYPE: u16 = 0x01;

/// Tag for an array subrange (dimension bounds) DIE.
pub const DW_TAG_SUBRANGE_TYPE: u16 = 0x21;

/// Tag for a subroutine (function pointer) type DIE.
pub const DW_TAG_SUBROUTINE_TYPE: u16 = 0x15;

/// Tag for a const-qualified type DIE.
pub const DW_TAG_CONST_TYPE: u16 = 0x26;

/// Tag for a volatile-qualified type DIE.
pub const DW_TAG_VOLATILE_TYPE: u16 = 0x35;

/// Tag for a restrict-qualified type DIE.
pub const DW_TAG_RESTRICT_TYPE: u16 = 0x37;

/// Tag for unspecified parameters (variadic `...` in function signatures).
pub const DW_TAG_UNSPECIFIED_PARAMETERS: u16 = 0x18;

/// Tag for a lexical block (scope within a function body).
pub const DW_TAG_LEXICAL_BLOCK: u16 = 0x0b;

// ============================================================================
// DWARF v4 Attribute Constants (Table 7.5)
// ============================================================================

/// Attribute: name of the entity.
pub const DW_AT_NAME: u16 = 0x03;

/// Attribute: offset into `.debug_line` section (statement list).
pub const DW_AT_STMT_LIST: u16 = 0x10;

/// Attribute: lowest machine code address in the entity's range.
pub const DW_AT_LOW_PC: u16 = 0x11;

/// Attribute: highest machine code address (or size) in the entity's range.
pub const DW_AT_HIGH_PC: u16 = 0x12;

/// Attribute: source language identifier (e.g., DW_LANG_C11).
pub const DW_AT_LANGUAGE: u16 = 0x13;

/// Attribute: compilation directory path.
pub const DW_AT_COMP_DIR: u16 = 0x1b;

/// Attribute: compiler producer identification string.
pub const DW_AT_PRODUCER: u16 = 0x25;

/// Attribute: type reference for variables, parameters, return types.
pub const DW_AT_TYPE: u16 = 0x49;

/// Attribute: location description for variables (stack offset, register, etc.).
pub const DW_AT_LOCATION: u16 = 0x02;

/// Attribute: whether the entity has external (global) linkage.
pub const DW_AT_EXTERNAL: u16 = 0x3f;

/// Attribute: source file index (1-based, references `.debug_line` file table).
pub const DW_AT_DECL_FILE: u16 = 0x3a;

/// Attribute: source line number of the declaration.
pub const DW_AT_DECL_LINE: u16 = 0x3b;

/// Attribute: size in bytes of the described type or entity.
pub const DW_AT_BYTE_SIZE: u16 = 0x0b;

/// Attribute: base type encoding (signed, unsigned, float, etc.).
pub const DW_AT_ENCODING: u16 = 0x3e;

/// Attribute: whether the subprogram has a prototype.
pub const DW_AT_PROTOTYPED: u16 = 0x27;

/// Attribute: frame base location expression for the subprogram.
pub const DW_AT_FRAME_BASE: u16 = 0x40;

/// Attribute: upper bound of an array subrange.
pub const DW_AT_UPPER_BOUND: u16 = 0x2f;

/// Attribute: byte offset of a member within a struct/union.
pub const DW_AT_DATA_MEMBER_LOCATION: u16 = 0x38;

/// Attribute: constant value for an enumerator or compile-time constant.
pub const DW_AT_CONST_VALUE: u16 = 0x1c;

/// Attribute: whether the entity is a declaration (not a definition).
pub const DW_AT_DECLARATION: u16 = 0x3c;

/// Attribute: visibility of the entity (public, protected, private).
pub const DW_AT_VISIBILITY: u16 = 0x17;

// ============================================================================
// DWARF v4 Attribute Form Constants (Table 7.6)
// ============================================================================

/// Form: target-architecture-sized address (4 or 8 bytes).
pub const DW_FORM_ADDR: u16 = 0x01;

/// Form: 1-byte unsigned constant.
pub const DW_FORM_DATA1: u16 = 0x0b;

/// Form: 2-byte unsigned constant.
pub const DW_FORM_DATA2: u16 = 0x05;

/// Form: 4-byte unsigned constant.
pub const DW_FORM_DATA4: u16 = 0x06;

/// Form: 8-byte unsigned constant.
pub const DW_FORM_DATA8: u16 = 0x07;

/// Form: signed LEB128 encoded value.
pub const DW_FORM_SDATA: u16 = 0x0d;

/// Form: unsigned LEB128 encoded value.
pub const DW_FORM_UDATA: u16 = 0x0f;

/// Form: 4-byte offset into `.debug_str` section (32-bit DWARF).
pub const DW_FORM_STRP: u16 = 0x0e;

/// Form: 1-byte boolean flag.
pub const DW_FORM_FLAG: u16 = 0x0c;

/// Form: implicit boolean flag (true), no data emitted (DWARF v4 addition).
pub const DW_FORM_FLAG_PRESENT: u16 = 0x19;

/// Form: 4-byte offset within the same `.debug_info` compilation unit.
pub const DW_FORM_REF4: u16 = 0x13;

/// Form: 4-byte section offset (32-bit DWARF) — cross-section reference.
pub const DW_FORM_SEC_OFFSET: u16 = 0x17;

/// Form: ULEB128 length followed by a byte block — location expression (DWARF v4).
pub const DW_FORM_EXPRLOC: u16 = 0x18;

/// Form: null-terminated inline string (not referenced via `.debug_str`).
pub const DW_FORM_STRING: u16 = 0x08;

// ============================================================================
// DWARF v4 Children Flag Constants
// ============================================================================

/// The DIE does NOT have child DIEs.
pub const DW_CHILDREN_NO: u8 = 0x00;

/// The DIE has child DIEs (followed by a null entry terminator).
pub const DW_CHILDREN_YES: u8 = 0x01;

// ============================================================================
// Data Structures
// ============================================================================

/// An (attribute, form) pair in an abbreviation entry.
///
/// Each abbreviation entry contains an ordered list of these pairs, specifying
/// which DWARF attributes the DIE carries and how each attribute's value is
/// encoded in `.debug_info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbbrevAttribute {
    /// DWARF attribute identifier (`DW_AT_*` constant).
    pub attribute: u16,
    /// DWARF attribute form — encoding format (`DW_FORM_*` constant).
    pub form: u16,
}

/// A complete abbreviation entry in the abbreviation table.
///
/// Each entry defines a template for a class of DIEs in `.debug_info`.
/// DIEs reference entries by their unique abbreviation code.
#[derive(Debug, Clone)]
pub struct AbbrevEntry {
    /// Abbreviation code (1-based, unique within the table).
    /// DIEs in `.debug_info` reference this code to identify their template.
    pub code: u32,
    /// DWARF tag identifying the kind of DIE (`DW_TAG_*` constant).
    pub tag: u16,
    /// Whether DIEs using this abbreviation have child DIEs.
    /// If true, the DIE's children are terminated by a null entry (code 0).
    pub has_children: bool,
    /// Ordered list of (attribute, form) specifications.
    /// Defines which attributes the DIE carries and their encoding formats.
    pub attributes: Vec<AbbrevAttribute>,
}

// ============================================================================
// AbbrevTable Builder
// ============================================================================

/// Builder for the `.debug_abbrev` section.
///
/// Collects abbreviation entries and serializes them into the DWARF v4 binary
/// format. Abbreviation codes are assigned sequentially starting from 1.
///
/// # Usage
/// ```ignore
/// let mut table = AbbrevTable::new();
/// let cu_code = table.add_compile_unit_abbrev();
/// let func_code = table.add_subprogram_abbrev(true);
/// let bytes = table.generate();
/// ```
pub struct AbbrevTable {
    /// Ordered list of abbreviation entries.
    entries: Vec<AbbrevEntry>,
    /// Next abbreviation code to assign (1-based, monotonically increasing).
    next_code: u32,
}

impl AbbrevTable {
    /// Create a new, empty abbreviation table.
    ///
    /// The first entry added will receive abbreviation code 1.
    pub fn new() -> Self {
        AbbrevTable {
            entries: Vec::new(),
            next_code: 1,
        }
    }

    /// Add a new abbreviation entry with the given tag, children flag, and
    /// attribute list. Returns the assigned abbreviation code.
    ///
    /// The code is assigned sequentially (1, 2, 3, ...) and is used by DIEs
    /// in `.debug_info` to reference this abbreviation template.
    pub fn add_entry(
        &mut self,
        tag: u16,
        has_children: bool,
        attributes: Vec<AbbrevAttribute>,
    ) -> u32 {
        let code = self.next_code;
        self.next_code += 1;
        self.entries.push(AbbrevEntry {
            code,
            tag,
            has_children,
            attributes,
        });
        code
    }

    // ========================================================================
    // Predefined abbreviation builders for common DIE types
    // ========================================================================

    /// Add abbreviation for `DW_TAG_compile_unit`.
    ///
    /// This is the top-level DIE in every `.debug_info` compilation unit.
    /// It always has children (subprograms, types, variables).
    ///
    /// Attributes:
    /// - `DW_AT_producer` (`DW_FORM_STRP`) — compiler identification string
    /// - `DW_AT_language` (`DW_FORM_DATA2`) — source language (C11)
    /// - `DW_AT_name` (`DW_FORM_STRP`) — source file name
    /// - `DW_AT_comp_dir` (`DW_FORM_STRP`) — compilation directory
    /// - `DW_AT_low_pc` (`DW_FORM_ADDR`) — lowest code address
    /// - `DW_AT_high_pc` (`DW_FORM_DATA8`) — size of code range
    /// - `DW_AT_stmt_list` (`DW_FORM_SEC_OFFSET`) — offset into `.debug_line`
    pub fn add_compile_unit_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_COMPILE_UNIT,
            true, // compile units always have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_PRODUCER,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_LANGUAGE,
                    form: DW_FORM_DATA2,
                },
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_COMP_DIR,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_LOW_PC,
                    form: DW_FORM_ADDR,
                },
                AbbrevAttribute {
                    attribute: DW_AT_HIGH_PC,
                    form: DW_FORM_DATA8,
                },
                AbbrevAttribute {
                    attribute: DW_AT_STMT_LIST,
                    form: DW_FORM_SEC_OFFSET,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_subprogram` (function definition).
    ///
    /// The `has_children` flag controls whether the subprogram DIE can contain
    /// child DIEs (formal parameters, local variables, lexical blocks).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — function name
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    /// - `DW_AT_low_pc` (`DW_FORM_ADDR`) — function start address
    /// - `DW_AT_high_pc` (`DW_FORM_DATA8`) — function size
    /// - `DW_AT_type` (`DW_FORM_REF4`) — return type DIE reference
    /// - `DW_AT_external` (`DW_FORM_FLAG_PRESENT`) — external linkage flag
    /// - `DW_AT_frame_base` (`DW_FORM_EXPRLOC`) — frame base location
    /// - `DW_AT_prototyped` (`DW_FORM_FLAG_PRESENT`) — has prototype flag
    pub fn add_subprogram_abbrev(&mut self, has_children: bool) -> u32 {
        self.add_entry(
            DW_TAG_SUBPROGRAM,
            has_children,
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_LOW_PC,
                    form: DW_FORM_ADDR,
                },
                AbbrevAttribute {
                    attribute: DW_AT_HIGH_PC,
                    form: DW_FORM_DATA8,
                },
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_EXTERNAL,
                    form: DW_FORM_FLAG_PRESENT,
                },
                AbbrevAttribute {
                    attribute: DW_AT_FRAME_BASE,
                    form: DW_FORM_EXPRLOC,
                },
                AbbrevAttribute {
                    attribute: DW_AT_PROTOTYPED,
                    form: DW_FORM_FLAG_PRESENT,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_variable` (local or global variable).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — variable name
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    /// - `DW_AT_type` (`DW_FORM_REF4`) — type DIE reference
    /// - `DW_AT_location` (`DW_FORM_EXPRLOC`) — location description
    pub fn add_variable_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_VARIABLE,
            false, // variables do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_LOCATION,
                    form: DW_FORM_EXPRLOC,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_formal_parameter` (function parameter).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — parameter name
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    /// - `DW_AT_type` (`DW_FORM_REF4`) — type DIE reference
    /// - `DW_AT_location` (`DW_FORM_EXPRLOC`) — location description
    pub fn add_formal_parameter_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_FORMAL_PARAMETER,
            false, // parameters do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_LOCATION,
                    form: DW_FORM_EXPRLOC,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_base_type` (primitive type like int, char).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — type name (e.g., "int", "char")
    /// - `DW_AT_byte_size` (`DW_FORM_DATA1`) — size in bytes
    /// - `DW_AT_encoding` (`DW_FORM_DATA1`) — DW_ATE_* encoding constant
    pub fn add_base_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_BASE_TYPE,
            false, // base types do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_BYTE_SIZE,
                    form: DW_FORM_DATA1,
                },
                AbbrevAttribute {
                    attribute: DW_AT_ENCODING,
                    form: DW_FORM_DATA1,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_pointer_type`.
    ///
    /// Attributes:
    /// - `DW_AT_type` (`DW_FORM_REF4`) — pointed-to type DIE reference
    /// - `DW_AT_byte_size` (`DW_FORM_DATA1`) — pointer size (4 or 8 bytes)
    pub fn add_pointer_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_POINTER_TYPE,
            false, // pointer types do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_BYTE_SIZE,
                    form: DW_FORM_DATA1,
                },
            ],
        )
    }

    /// Add abbreviation for composite types (`DW_TAG_structure_type` or
    /// `DW_TAG_union_type`).
    ///
    /// The `tag` parameter should be `DW_TAG_STRUCTURE_TYPE` or `DW_TAG_UNION_TYPE`.
    /// The `has_children` flag is typically true (members are children).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — struct/union name
    /// - `DW_AT_byte_size` (`DW_FORM_UDATA`) — total size in bytes
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    pub fn add_composite_type_abbrev(&mut self, tag: u16, has_children: bool) -> u32 {
        self.add_entry(
            tag,
            has_children,
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_BYTE_SIZE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_member` (struct/union member).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — member name
    /// - `DW_AT_type` (`DW_FORM_REF4`) — member type DIE reference
    /// - `DW_AT_data_member_location` (`DW_FORM_UDATA`) — byte offset within
    ///   the containing struct/union
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    pub fn add_member_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_MEMBER,
            false, // members do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DATA_MEMBER_LOCATION,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_array_type`.
    ///
    /// Array types have children (subrange types defining dimensions).
    ///
    /// Attributes:
    /// - `DW_AT_type` (`DW_FORM_REF4`) — element type DIE reference
    pub fn add_array_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_ARRAY_TYPE,
            true, // arrays have subrange children for dimensions
            vec![AbbrevAttribute {
                attribute: DW_AT_TYPE,
                form: DW_FORM_REF4,
            }],
        )
    }

    /// Add abbreviation for `DW_TAG_subrange_type` (array dimension bounds).
    ///
    /// Attributes:
    /// - `DW_AT_type` (`DW_FORM_REF4`) — index type DIE reference
    /// - `DW_AT_upper_bound` (`DW_FORM_UDATA`) — upper bound (count - 1)
    pub fn add_subrange_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_SUBRANGE_TYPE,
            false, // subranges do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_UPPER_BOUND,
                    form: DW_FORM_UDATA,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_typedef`.
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — typedef name
    /// - `DW_AT_type` (`DW_FORM_REF4`) — underlying type DIE reference
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    pub fn add_typedef_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_TYPEDEF,
            false, // typedefs do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
            ],
        )
    }

    /// Add abbreviation for a type modifier (`DW_TAG_const_type`,
    /// `DW_TAG_volatile_type`, or `DW_TAG_restrict_type`).
    ///
    /// These are "wrapper" type DIEs that modify the underlying type.
    ///
    /// Attributes:
    /// - `DW_AT_type` (`DW_FORM_REF4`) — modified type DIE reference
    pub fn add_type_modifier_abbrev(&mut self, tag: u16) -> u32 {
        self.add_entry(
            tag,
            false, // type modifiers do not have children
            vec![AbbrevAttribute {
                attribute: DW_AT_TYPE,
                form: DW_FORM_REF4,
            }],
        )
    }

    /// Add abbreviation for `DW_TAG_subroutine_type` (function pointer type).
    ///
    /// The `has_children` flag is true when the function type has parameters
    /// (represented as `DW_TAG_formal_parameter` and optionally
    /// `DW_TAG_unspecified_parameters` children).
    ///
    /// Attributes:
    /// - `DW_AT_type` (`DW_FORM_REF4`) — return type DIE reference
    /// - `DW_AT_prototyped` (`DW_FORM_FLAG_PRESENT`) — has prototype flag
    pub fn add_subroutine_type_abbrev(&mut self, has_children: bool) -> u32 {
        self.add_entry(
            DW_TAG_SUBROUTINE_TYPE,
            has_children,
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_PROTOTYPED,
                    form: DW_FORM_FLAG_PRESENT,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_enumeration_type`.
    ///
    /// Enumeration types have children (enumerator constant DIEs).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — enum type name
    /// - `DW_AT_byte_size` (`DW_FORM_UDATA`) — size in bytes
    /// - `DW_AT_type` (`DW_FORM_REF4`) — underlying integer type
    /// - `DW_AT_decl_file` (`DW_FORM_UDATA`) — source file index
    /// - `DW_AT_decl_line` (`DW_FORM_UDATA`) — declaration line number
    pub fn add_enum_type_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_ENUMERATION_TYPE,
            true, // enumerations have enumerator children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_BYTE_SIZE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_TYPE,
                    form: DW_FORM_REF4,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_FILE,
                    form: DW_FORM_UDATA,
                },
                AbbrevAttribute {
                    attribute: DW_AT_DECL_LINE,
                    form: DW_FORM_UDATA,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_enumerator` (enum constant value).
    ///
    /// Attributes:
    /// - `DW_AT_name` (`DW_FORM_STRP`) — enumerator name
    /// - `DW_AT_const_value` (`DW_FORM_SDATA`) — constant value (signed LEB128)
    pub fn add_enumerator_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_ENUMERATOR,
            false, // enumerators do not have children
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_CONST_VALUE,
                    form: DW_FORM_SDATA,
                },
            ],
        )
    }

    /// Add abbreviation for `DW_TAG_unspecified_parameters` (variadic `...`).
    ///
    /// This DIE has no attributes — it simply marks that the function accepts
    /// additional unspecified parameters beyond those listed.
    pub fn add_unspecified_params_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_UNSPECIFIED_PARAMETERS,
            false,  // no children
            vec![], // no attributes
        )
    }

    /// Add abbreviation for `DW_TAG_lexical_block` (inner scope).
    ///
    /// Lexical blocks represent scopes within a function body (e.g., compound
    /// statements that introduce a new scope). They have children (variables,
    /// nested blocks).
    ///
    /// Attributes:
    /// - `DW_AT_low_pc` (`DW_FORM_ADDR`) — block start address
    /// - `DW_AT_high_pc` (`DW_FORM_DATA8`) — block size
    pub fn add_lexical_block_abbrev(&mut self) -> u32 {
        self.add_entry(
            DW_TAG_LEXICAL_BLOCK,
            true, // lexical blocks have children (variables, nested blocks)
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_LOW_PC,
                    form: DW_FORM_ADDR,
                },
                AbbrevAttribute {
                    attribute: DW_AT_HIGH_PC,
                    form: DW_FORM_DATA8,
                },
            ],
        )
    }

    // ========================================================================
    // Binary Serialization
    // ========================================================================

    /// Serialize the entire abbreviation table to the `.debug_abbrev` binary format.
    ///
    /// The output follows the DWARF v4 abbreviation table encoding:
    /// - Each entry: `[ULEB128 code] [ULEB128 tag] [u8 children] {[ULEB128 attr] [ULEB128 form]}* [0] [0]`
    /// - Table terminator: `[0]` (null abbreviation code)
    ///
    /// # Returns
    /// A `Vec<u8>` containing the complete `.debug_abbrev` section data,
    /// ready for embedding in an ELF file.
    pub fn generate(&self) -> Vec<u8> {
        // Pre-allocate a reasonable buffer size to minimize reallocations.
        // Each entry is roughly: 1 + 2 + 1 + (attributes * 4) + 2 bytes.
        let estimated_size = self
            .entries
            .iter()
            .fold(0usize, |acc, e| acc + 4 + e.attributes.len() * 4 + 2)
            + 1;
        let mut buffer = Vec::with_capacity(estimated_size);

        for entry in &self.entries {
            // 1. Abbreviation code (ULEB128)
            encode_uleb128(entry.code as u64, &mut buffer);

            // 2. Tag (ULEB128)
            encode_uleb128(entry.tag as u64, &mut buffer);

            // 3. Children flag (1 byte)
            buffer.push(if entry.has_children {
                DW_CHILDREN_YES
            } else {
                DW_CHILDREN_NO
            });

            // 4. Attribute specifications: (attribute ULEB128, form ULEB128) pairs
            for attr in &entry.attributes {
                encode_uleb128(attr.attribute as u64, &mut buffer);
                encode_uleb128(attr.form as u64, &mut buffer);
            }

            // 5. Terminate attribute list with (0, 0)
            buffer.push(0);
            buffer.push(0);
        }

        // 6. Terminate the abbreviation table with a 0 byte (null abbreviation code)
        buffer.push(0);

        buffer
    }
}

/// Implement `Default` for `AbbrevTable` to match Rust conventions.
impl Default for AbbrevTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LEB128 Encoding
// ============================================================================

/// Encode an unsigned integer as ULEB128 (Unsigned Little-Endian Base 128).
///
/// ULEB128 encoding uses 7 bits per byte, with the high bit indicating whether
/// more bytes follow (1 = more, 0 = last byte). This is the standard
/// variable-length encoding used throughout the DWARF format for unsigned values.
///
/// # Arguments
/// * `value` — The unsigned value to encode
/// * `output` — The byte buffer to append encoded bytes to
///
/// # Examples
/// - `encode_uleb128(0, &mut buf)` → `[0x00]`
/// - `encode_uleb128(127, &mut buf)` → `[0x7F]`
/// - `encode_uleb128(128, &mut buf)` → `[0x80, 0x01]`
/// - `encode_uleb128(624485, &mut buf)` → `[0xE5, 0x8E, 0x26]`
fn encode_uleb128(mut value: u64, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // set high bit to indicate more bytes follow
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uleb128_zero() {
        let mut buf = Vec::new();
        encode_uleb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_uleb128_single_byte() {
        let mut buf = Vec::new();
        encode_uleb128(1, &mut buf);
        assert_eq!(buf, vec![0x01]);

        buf.clear();
        encode_uleb128(127, &mut buf);
        assert_eq!(buf, vec![0x7F]);
    }

    #[test]
    fn test_uleb128_two_bytes() {
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
        // Verify all but the last byte have high bit set
        for &byte in &buf[..9] {
            assert!(byte & 0x80 != 0, "continuation bit must be set");
        }
        // Last byte must not have high bit set
        assert!(
            buf[9] & 0x80 == 0,
            "last byte must not have continuation bit"
        );
    }

    #[test]
    fn test_empty_table_generates_single_null_byte() {
        let table = AbbrevTable::new();
        let bytes = table.generate();
        // An empty table should just be a null terminator
        assert_eq!(bytes, vec![0x00]);
    }

    #[test]
    fn test_codes_are_one_based_and_sequential() {
        let mut table = AbbrevTable::new();
        let code1 = table.add_base_type_abbrev();
        let code2 = table.add_pointer_type_abbrev();
        let code3 = table.add_variable_abbrev();
        assert_eq!(code1, 1);
        assert_eq!(code2, 2);
        assert_eq!(code3, 3);
    }

    #[test]
    fn test_generate_single_entry() {
        let mut table = AbbrevTable::new();
        table.add_entry(
            DW_TAG_BASE_TYPE,
            false,
            vec![
                AbbrevAttribute {
                    attribute: DW_AT_NAME,
                    form: DW_FORM_STRP,
                },
                AbbrevAttribute {
                    attribute: DW_AT_BYTE_SIZE,
                    form: DW_FORM_DATA1,
                },
            ],
        );
        let bytes = table.generate();

        let mut expected = Vec::new();
        // Code 1 (ULEB128)
        encode_uleb128(1, &mut expected);
        // Tag DW_TAG_BASE_TYPE (0x24) (ULEB128)
        encode_uleb128(DW_TAG_BASE_TYPE as u64, &mut expected);
        // Children: no
        expected.push(DW_CHILDREN_NO);
        // Attr: DW_AT_NAME (0x03), DW_FORM_STRP (0x0e)
        encode_uleb128(DW_AT_NAME as u64, &mut expected);
        encode_uleb128(DW_FORM_STRP as u64, &mut expected);
        // Attr: DW_AT_BYTE_SIZE (0x0b), DW_FORM_DATA1 (0x0b)
        encode_uleb128(DW_AT_BYTE_SIZE as u64, &mut expected);
        encode_uleb128(DW_FORM_DATA1 as u64, &mut expected);
        // Attribute list terminator (0, 0)
        expected.push(0);
        expected.push(0);
        // Table terminator
        expected.push(0);

        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_compile_unit_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_compile_unit_abbrev();
        assert_eq!(code, 1);

        // Verify the entry properties
        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_COMPILE_UNIT);
        assert!(entry.has_children);
        assert_eq!(entry.attributes.len(), 7);
        // Verify first attribute is producer
        assert_eq!(entry.attributes[0].attribute, DW_AT_PRODUCER);
        assert_eq!(entry.attributes[0].form, DW_FORM_STRP);
        // Verify last attribute is stmt_list with sec_offset form
        assert_eq!(entry.attributes[6].attribute, DW_AT_STMT_LIST);
        assert_eq!(entry.attributes[6].form, DW_FORM_SEC_OFFSET);
    }

    #[test]
    fn test_subprogram_abbrev_with_children() {
        let mut table = AbbrevTable::new();
        let code = table.add_subprogram_abbrev(true);
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_SUBPROGRAM);
        assert!(entry.has_children);
        assert_eq!(entry.attributes.len(), 9);
        // Verify external uses FLAG_PRESENT (DWARF v4 — no data emitted)
        let external_attr = entry
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_EXTERNAL)
            .expect("DW_AT_EXTERNAL should be present");
        assert_eq!(external_attr.form, DW_FORM_FLAG_PRESENT);
        // Verify frame_base uses EXPRLOC (DWARF v4)
        let frame_attr = entry
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_FRAME_BASE)
            .expect("DW_AT_FRAME_BASE should be present");
        assert_eq!(frame_attr.form, DW_FORM_EXPRLOC);
    }

    #[test]
    fn test_subprogram_abbrev_without_children() {
        let mut table = AbbrevTable::new();
        let code = table.add_subprogram_abbrev(false);
        assert_eq!(code, 1);
        let entry = &table.entries[0];
        assert!(!entry.has_children);
    }

    #[test]
    fn test_variable_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_variable_abbrev();
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_VARIABLE);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 5);
        // Verify location uses EXPRLOC (DWARF v4)
        let loc_attr = entry
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_LOCATION)
            .expect("DW_AT_LOCATION should be present");
        assert_eq!(loc_attr.form, DW_FORM_EXPRLOC);
    }

    #[test]
    fn test_formal_parameter_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_formal_parameter_abbrev();
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_FORMAL_PARAMETER);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 5);
    }

    #[test]
    fn test_composite_type_abbrev() {
        let mut table = AbbrevTable::new();
        let struct_code = table.add_composite_type_abbrev(DW_TAG_STRUCTURE_TYPE, true);
        let union_code = table.add_composite_type_abbrev(DW_TAG_UNION_TYPE, false);
        assert_eq!(struct_code, 1);
        assert_eq!(union_code, 2);

        assert_eq!(table.entries[0].tag, DW_TAG_STRUCTURE_TYPE);
        assert!(table.entries[0].has_children);
        assert_eq!(table.entries[1].tag, DW_TAG_UNION_TYPE);
        assert!(!table.entries[1].has_children);
    }

    #[test]
    fn test_member_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_member_abbrev();
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_MEMBER);
        assert!(!entry.has_children);
        // Verify data_member_location is present with UDATA form
        let loc_attr = entry
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_DATA_MEMBER_LOCATION)
            .expect("DW_AT_DATA_MEMBER_LOCATION should be present");
        assert_eq!(loc_attr.form, DW_FORM_UDATA);
    }

    #[test]
    fn test_array_and_subrange_abbrevs() {
        let mut table = AbbrevTable::new();
        let arr_code = table.add_array_type_abbrev();
        let sub_code = table.add_subrange_type_abbrev();
        assert_eq!(arr_code, 1);
        assert_eq!(sub_code, 2);

        // Array has children (subranges)
        assert!(table.entries[0].has_children);
        // Subrange does not
        assert!(!table.entries[1].has_children);
        // Subrange has upper_bound
        let ub_attr = table.entries[1]
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_UPPER_BOUND)
            .expect("DW_AT_UPPER_BOUND should be present");
        assert_eq!(ub_attr.form, DW_FORM_UDATA);
    }

    #[test]
    fn test_typedef_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_typedef_abbrev();
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_TYPEDEF);
        assert!(!entry.has_children);
        assert_eq!(entry.attributes.len(), 4);
    }

    #[test]
    fn test_type_modifier_abbrevs() {
        let mut table = AbbrevTable::new();
        let c = table.add_type_modifier_abbrev(DW_TAG_CONST_TYPE);
        let v = table.add_type_modifier_abbrev(DW_TAG_VOLATILE_TYPE);
        let r = table.add_type_modifier_abbrev(DW_TAG_RESTRICT_TYPE);
        assert_eq!(c, 1);
        assert_eq!(v, 2);
        assert_eq!(r, 3);

        for entry in &table.entries {
            assert!(!entry.has_children);
            assert_eq!(entry.attributes.len(), 1);
            assert_eq!(entry.attributes[0].attribute, DW_AT_TYPE);
            assert_eq!(entry.attributes[0].form, DW_FORM_REF4);
        }
    }

    #[test]
    fn test_subroutine_type_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_subroutine_type_abbrev(true);
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_SUBROUTINE_TYPE);
        assert!(entry.has_children);
        // Verify prototyped uses FLAG_PRESENT
        let proto_attr = entry
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_PROTOTYPED)
            .expect("DW_AT_PROTOTYPED should be present");
        assert_eq!(proto_attr.form, DW_FORM_FLAG_PRESENT);
    }

    #[test]
    fn test_enum_type_and_enumerator_abbrevs() {
        let mut table = AbbrevTable::new();
        let enum_code = table.add_enum_type_abbrev();
        let etor_code = table.add_enumerator_abbrev();
        assert_eq!(enum_code, 1);
        assert_eq!(etor_code, 2);

        // Enum type has children (enumerators)
        assert!(table.entries[0].has_children);
        assert!(!table.entries[1].has_children);

        // Enumerator uses SDATA for const_value
        let cv_attr = table.entries[1]
            .attributes
            .iter()
            .find(|a| a.attribute == DW_AT_CONST_VALUE)
            .expect("DW_AT_CONST_VALUE should be present");
        assert_eq!(cv_attr.form, DW_FORM_SDATA);
    }

    #[test]
    fn test_unspecified_params_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_unspecified_params_abbrev();
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_UNSPECIFIED_PARAMETERS);
        assert!(!entry.has_children);
        assert!(entry.attributes.is_empty());
    }

    #[test]
    fn test_lexical_block_abbrev() {
        let mut table = AbbrevTable::new();
        let code = table.add_lexical_block_abbrev();
        assert_eq!(code, 1);

        let entry = &table.entries[0];
        assert_eq!(entry.tag, DW_TAG_LEXICAL_BLOCK);
        assert!(entry.has_children);
        assert_eq!(entry.attributes.len(), 2);
    }

    #[test]
    fn test_multiple_entries_sequential_codes() {
        let mut table = AbbrevTable::new();
        let c1 = table.add_compile_unit_abbrev();
        let c2 = table.add_subprogram_abbrev(true);
        let c3 = table.add_variable_abbrev();
        let c4 = table.add_formal_parameter_abbrev();
        let c5 = table.add_base_type_abbrev();
        let c6 = table.add_pointer_type_abbrev();
        assert_eq!(c1, 1);
        assert_eq!(c2, 2);
        assert_eq!(c3, 3);
        assert_eq!(c4, 4);
        assert_eq!(c5, 5);
        assert_eq!(c6, 6);
    }

    #[test]
    fn test_generate_terminates_with_null() {
        let mut table = AbbrevTable::new();
        table.add_base_type_abbrev();
        let bytes = table.generate();
        // Last byte must be the table-level null terminator
        assert_eq!(*bytes.last().unwrap(), 0x00);
    }

    #[test]
    fn test_generate_entry_terminates_attributes_with_null_pair() {
        let mut table = AbbrevTable::new();
        // Add entry with no attributes (unspecified params)
        table.add_unspecified_params_abbrev();
        let bytes = table.generate();
        // Should be: [code=1] [tag=0x18] [children=0] [0] [0] [0]
        // The (0, 0) after children=0 terminates the empty attribute list
        // The final 0 terminates the table
        assert!(bytes.len() >= 5);
        // Attribute list terminates with two zero bytes before table terminator
        let table_terminator = bytes[bytes.len() - 1];
        let attr_term_2 = bytes[bytes.len() - 2];
        let attr_term_1 = bytes[bytes.len() - 3];
        assert_eq!(table_terminator, 0x00);
        assert_eq!(attr_term_1, 0x00);
        assert_eq!(attr_term_2, 0x00);
    }

    #[test]
    fn test_default_trait() {
        let table = AbbrevTable::default();
        let bytes = table.generate();
        assert_eq!(bytes, vec![0x00]);
    }

    #[test]
    fn test_dwarf_tag_constants() {
        // Verify key constant values against DWARF v4 specification
        assert_eq!(DW_TAG_ARRAY_TYPE, 0x01);
        assert_eq!(DW_TAG_ENUMERATION_TYPE, 0x04);
        assert_eq!(DW_TAG_FORMAL_PARAMETER, 0x05);
        assert_eq!(DW_TAG_LEXICAL_BLOCK, 0x0b);
        assert_eq!(DW_TAG_MEMBER, 0x0d);
        assert_eq!(DW_TAG_POINTER_TYPE, 0x0f);
        assert_eq!(DW_TAG_COMPILE_UNIT, 0x11);
        assert_eq!(DW_TAG_STRUCTURE_TYPE, 0x13);
        assert_eq!(DW_TAG_SUBROUTINE_TYPE, 0x15);
        assert_eq!(DW_TAG_TYPEDEF, 0x16);
        assert_eq!(DW_TAG_UNION_TYPE, 0x17);
        assert_eq!(DW_TAG_UNSPECIFIED_PARAMETERS, 0x18);
        assert_eq!(DW_TAG_SUBRANGE_TYPE, 0x21);
        assert_eq!(DW_TAG_BASE_TYPE, 0x24);
        assert_eq!(DW_TAG_CONST_TYPE, 0x26);
        assert_eq!(DW_TAG_ENUMERATOR, 0x28);
        assert_eq!(DW_TAG_SUBPROGRAM, 0x2e);
        assert_eq!(DW_TAG_VARIABLE, 0x34);
        assert_eq!(DW_TAG_VOLATILE_TYPE, 0x35);
        assert_eq!(DW_TAG_RESTRICT_TYPE, 0x37);
    }

    #[test]
    fn test_dwarf_at_constants() {
        // Verify key attribute constant values
        assert_eq!(DW_AT_LOCATION, 0x02);
        assert_eq!(DW_AT_NAME, 0x03);
        assert_eq!(DW_AT_BYTE_SIZE, 0x0b);
        assert_eq!(DW_AT_STMT_LIST, 0x10);
        assert_eq!(DW_AT_LOW_PC, 0x11);
        assert_eq!(DW_AT_HIGH_PC, 0x12);
        assert_eq!(DW_AT_LANGUAGE, 0x13);
        assert_eq!(DW_AT_VISIBILITY, 0x17);
        assert_eq!(DW_AT_COMP_DIR, 0x1b);
        assert_eq!(DW_AT_CONST_VALUE, 0x1c);
        assert_eq!(DW_AT_PRODUCER, 0x25);
        assert_eq!(DW_AT_PROTOTYPED, 0x27);
        assert_eq!(DW_AT_UPPER_BOUND, 0x2f);
        assert_eq!(DW_AT_DATA_MEMBER_LOCATION, 0x38);
        assert_eq!(DW_AT_DECL_FILE, 0x3a);
        assert_eq!(DW_AT_DECL_LINE, 0x3b);
        assert_eq!(DW_AT_DECLARATION, 0x3c);
        assert_eq!(DW_AT_ENCODING, 0x3e);
        assert_eq!(DW_AT_EXTERNAL, 0x3f);
        assert_eq!(DW_AT_FRAME_BASE, 0x40);
        assert_eq!(DW_AT_TYPE, 0x49);
    }

    #[test]
    fn test_dwarf_form_constants() {
        // Verify key form constant values
        assert_eq!(DW_FORM_ADDR, 0x01);
        assert_eq!(DW_FORM_DATA2, 0x05);
        assert_eq!(DW_FORM_DATA4, 0x06);
        assert_eq!(DW_FORM_DATA8, 0x07);
        assert_eq!(DW_FORM_STRING, 0x08);
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
    fn test_children_flag_constants() {
        assert_eq!(DW_CHILDREN_NO, 0x00);
        assert_eq!(DW_CHILDREN_YES, 0x01);
    }

    #[test]
    fn test_full_table_round_trip() {
        // Build a representative abbreviation table and verify generation
        let mut table = AbbrevTable::new();
        let cu = table.add_compile_unit_abbrev();
        let sp = table.add_subprogram_abbrev(true);
        let var = table.add_variable_abbrev();
        let fp = table.add_formal_parameter_abbrev();
        let bt = table.add_base_type_abbrev();
        let pt = table.add_pointer_type_abbrev();

        assert_eq!(cu, 1);
        assert_eq!(sp, 2);
        assert_eq!(var, 3);
        assert_eq!(fp, 4);
        assert_eq!(bt, 5);
        assert_eq!(pt, 6);

        let bytes = table.generate();
        // Verify it starts with code 1 and ends with null terminator
        assert_eq!(bytes[0], 1); // code 1 fits in single ULEB128 byte
        assert_eq!(*bytes.last().unwrap(), 0x00); // null terminator
                                                  // Verify total size is reasonable (all small values, single-byte ULEB128)
        assert!(bytes.len() > 10);
    }
}
