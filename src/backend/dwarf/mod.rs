//! DWARF debug information generation module.
//!
//! This module implements DWARF v4 debug section generation for the BCC compiler.
//! When the `-g` flag is specified, the compiler emits `.debug_info`, `.debug_abbrev`,
//! `.debug_line`, and `.debug_str` sections into the output ELF binary. When `-g` is
//! absent, no debug sections are emitted (zero debug section leakage).
//!
//! # Sub-modules
//!
//! - [`abbrev`]: `.debug_abbrev` — abbreviation table construction defining the
//!   schemas (tag, children flag, attribute specifications) used by DIEs in
//!   `.debug_info`.
//! - [`line`]: `.debug_line` — line number program builder that maps machine
//!   code addresses to source file/line locations for source-level debugging.
//! - [`debug_str`]: `.debug_str` — shared string table for debug information
//!   names referenced by `.debug_info` DIEs.
//!
//! # DWARF v4 Scope
//!
//! Debug information is limited to `-O0` (unoptimized builds):
//! - Source file and line number mapping
//! - Local variable locations
//! - Function entry/exit addresses
//! - Type descriptions for base types, pointers, structs, unions, arrays, enums

/// `.debug_abbrev` section generator — abbreviation table builder that defines
/// the structural schemas for Debug Information Entries (DIEs).
pub mod abbrev;

/// `.debug_line` section generator — line number program builder that maps
/// machine code addresses to source file locations, enabling source-level
/// debugging in GDB and other DWARF-aware debuggers.
pub mod line;

/// `.debug_info` section generator — compilation unit, subprogram, and
/// variable Debug Information Entries (DIEs) for source-level debugging.
/// Produces DW_TAG_compile_unit, DW_TAG_subprogram, DW_TAG_variable, and
/// type DIEs referencing abbreviation codes from [`abbrev`] and string
/// offsets from [`debug_str`].
pub mod info;

/// `.debug_str` section generator — string table construction for debug
/// information names, providing `DW_FORM_strp` offset management so that
/// string-valued attributes in `.debug_info` DIEs reference this shared
/// string pool rather than embedding strings inline.
#[path = "str.rs"]
pub mod debug_str;
