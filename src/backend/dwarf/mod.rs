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
