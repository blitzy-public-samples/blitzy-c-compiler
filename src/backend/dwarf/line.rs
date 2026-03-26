//! # `.debug_line` Section Generation
//!
//! Generates the DWARF v4 line number program that maps machine code
//! addresses to source file, line, and column positions.
//!
//! The section contains:
//! - A header with configuration parameters and file/directory tables
//! - A line number program (sequence of opcodes) that drives a state machine
//!
//! Debuggers (GDB) use this section for:
//! - Setting breakpoints at source lines
//! - Stepping through source code
//! - Displaying current source location during debugging
//!
//! ## Special Opcodes
//! The most common and compact encoding. A single byte encodes both
//! address advance and line advance simultaneously.
//!
//! ## Standard Opcodes
//! Used when special opcodes can't represent the deltas (e.g., large
//! address jumps, negative line numbers, file changes).
//!
//! ## Extended Opcodes
//! Used for absolute address setting (DW_LNE_set_address) and
//! sequence termination (DW_LNE_end_sequence).
//!
//! ## Zero-Dependency
//! This module depends only on the Rust standard library and internal
//! BCC modules. It replaces the external `gimli` crate with a fully
//! hand-implemented DWARF v4 line program generator.

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use super::{dwarf_address_size, DWARF_VERSION};
use crate::common::source_map::SourceMap;
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// DWARF v4 Standard Opcodes (Table 6.4)
// ---------------------------------------------------------------------------

/// Append current state as a row to the line table; reset special flags.
const DW_LNS_COPY: u8 = 0x01;
/// Advance the address register by ULEB128 × `minimum_instruction_length`.
const DW_LNS_ADVANCE_PC: u8 = 0x02;
/// Advance the line register by a SLEB128 value (can be negative).
const DW_LNS_ADVANCE_LINE: u8 = 0x03;
/// Set the file register to the ULEB128 operand value.
const DW_LNS_SET_FILE: u8 = 0x04;
/// Set the column register to the ULEB128 operand value.
const DW_LNS_SET_COLUMN: u8 = 0x05;
/// Toggle the `is_stmt` flag.
const DW_LNS_NEGATE_STMT: u8 = 0x06;
/// Set the `basic_block` flag.
#[allow(dead_code)]
const DW_LNS_SET_BASIC_BLOCK: u8 = 0x07;
/// Advance address by the constant `(255 - OPCODE_BASE) / LINE_RANGE`
/// times `minimum_instruction_length`.
const DW_LNS_CONST_ADD_PC: u8 = 0x08;
/// Advance the address by a fixed u16 operand (non-scaled).
#[allow(dead_code)]
const DW_LNS_FIXED_ADVANCE_PC: u8 = 0x09;
/// Mark the prologue end.
const DW_LNS_SET_PROLOGUE_END: u8 = 0x0a;
/// Mark the epilogue begin.
#[allow(dead_code)]
const DW_LNS_SET_EPILOGUE_BEGIN: u8 = 0x0b;
/// Set the instruction set architecture value.
#[allow(dead_code)]
const DW_LNS_SET_ISA: u8 = 0x0c;

// ---------------------------------------------------------------------------
// DWARF v4 Extended Opcodes
// ---------------------------------------------------------------------------

/// End of a contiguous address range; appends a row and resets state.
const DW_LNE_END_SEQUENCE: u8 = 0x01;
/// Set the address register to an absolute address.
const DW_LNE_SET_ADDRESS: u8 = 0x02;
/// Define a new source file (rarely used; files are in the header table).
#[allow(dead_code)]
const DW_LNE_DEFINE_FILE: u8 = 0x03;
/// Set the discriminator value (DWARF v4 addition).
#[allow(dead_code)]
const DW_LNE_SET_DISCRIMINATOR: u8 = 0x04;

// ---------------------------------------------------------------------------
// Header Configuration Constants
// ---------------------------------------------------------------------------

/// First special opcode number. Standard opcodes are numbered 1..12,
/// so special opcodes begin at 13.
const OPCODE_BASE: u8 = 13;

/// Minimum line advance representable by a special opcode.
/// With `LINE_BASE = -5` and `LINE_RANGE = 14`, special opcodes can
/// represent line advances from -5 to +8 (inclusive).
const LINE_BASE: i8 = -5;

/// Number of distinct line advance values representable by special opcodes.
const LINE_RANGE: u8 = 14;

/// Minimum instruction length in bytes. Set to 1 for all supported
/// architectures (x86-64, i686, AArch64, RISC-V 64).
const MINIMUM_INSTRUCTION_LENGTH: u8 = 1;

/// Maximum operations per instruction. Set to 1 for all supported
/// (non-VLIW) architectures. This field was added in DWARF v4.
const MAXIMUM_OPERATIONS_PER_INSTRUCTION: u8 = 1;

/// Default value of the `is_stmt` flag. When 1, each instruction is
/// considered a recommended breakpoint location by default.
const DEFAULT_IS_STMT: u8 = 1;

/// Number of LEB128 operands for each standard opcode (opcodes 1..12).
const STANDARD_OPCODE_LENGTHS: [u8; 12] = [
    0, // DW_LNS_copy
    1, // DW_LNS_advance_pc
    1, // DW_LNS_advance_line
    1, // DW_LNS_set_file
    1, // DW_LNS_set_column
    0, // DW_LNS_negate_stmt
    0, // DW_LNS_set_basic_block
    0, // DW_LNS_const_add_pc
    1, // DW_LNS_fixed_advance_pc
    0, // DW_LNS_set_prologue_end
    0, // DW_LNS_set_epilogue_begin
    1, // DW_LNS_set_isa
];

// ---------------------------------------------------------------------------
// Line Number State Machine (DWARF v4 §6.2.2)
// ---------------------------------------------------------------------------

/// Line number program state machine registers.
///
/// The line program opcodes manipulate these registers to build the line
/// table. Each "row" appended to the table is a snapshot of these registers.
struct LineStateMachine {
    /// Current machine code address (byte offset within `.text`).
    address: u64,
    /// Current file index (1-based, from the file name table).
    file: u32,
    /// Current source line number (1-based).
    line: u32,
    /// Current source column (0 = not specified).
    column: u32,
    /// Whether the current address is a recommended breakpoint position.
    is_stmt: bool,
    /// Whether the current address is the beginning of a basic block.
    #[allow(dead_code)]
    basic_block: bool,
    /// Whether the current address marks the end of a sequence.
    end_sequence: bool,
    /// Whether the current address is immediately after a function prologue.
    #[allow(dead_code)]
    prologue_end: bool,
    /// Whether the current address is immediately before a function epilogue.
    #[allow(dead_code)]
    epilogue_begin: bool,
    /// Instruction set architecture identifier (0 = default).
    #[allow(dead_code)]
    isa: u32,
    /// Block discriminator (DWARF v4 addition).
    #[allow(dead_code)]
    discriminator: u32,
}

impl LineStateMachine {
    /// Create a new state machine with the DWARF-specified initial values.
    fn new() -> Self {
        Self {
            address: 0,
            file: 1,
            line: 1,
            column: 0,
            is_stmt: DEFAULT_IS_STMT != 0,
            basic_block: false,
            end_sequence: false,
            prologue_end: false,
            epilogue_begin: false,
            isa: 0,
            discriminator: 0,
        }
    }

    /// Reset the state machine to initial values after `DW_LNE_end_sequence`.
    fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// LineEntry — input data for line mapping
// ---------------------------------------------------------------------------

/// A mapping from a machine code address to a source location.
///
/// This is the input to the line program generator — one entry per
/// address-to-source-line mapping point.
///
/// # Field Conventions
///
/// - `file` is 1-based, matching the DWARF file name table indices.
/// - `line` is 1-based (source line number).
/// - `column` is 0 if unspecified, otherwise 1-based.
/// - `end_sequence` marks the end of a contiguous address range (function).
#[derive(Debug, Clone)]
pub struct LineEntry {
    /// Machine code address (byte offset within `.text` section).
    pub address: u64,
    /// File index (1-based, matching the DWARF file name table).
    pub file: u32,
    /// Source line number (1-based).
    pub line: u32,
    /// Source column number (0 = unspecified, otherwise 1-based).
    pub column: u32,
    /// Whether this address is a recommended breakpoint (statement boundary).
    pub is_stmt: bool,
    /// Whether this entry marks the end of a contiguous address range.
    pub end_sequence: bool,
}

// ---------------------------------------------------------------------------
// FileTableEntry — internal helper for directory/file table construction
// ---------------------------------------------------------------------------

/// Internal representation of a file entry for the DWARF file name table.
struct FileTableEntry {
    /// Base filename (without directory path).
    name: String,
    /// Index into the directory table (0 = compilation directory).
    dir_index: u32,
}

// ---------------------------------------------------------------------------
// DebugLineGenerator
// ---------------------------------------------------------------------------

/// Generator for the DWARF v4 `.debug_line` section.
///
/// Constructs the complete line number program including:
/// - Program header with configuration parameters
/// - Include directory table and file name table
/// - Line number program opcodes (standard, extended, and special)
pub struct DebugLineGenerator<'a> {
    /// Output byte buffer for the `.debug_line` section.
    buffer: Vec<u8>,
    /// Target architecture (affects address size).
    target: &'a Target,
    /// Source map for file and directory information.
    source_map: &'a SourceMap,
    /// Address size in bytes (4 for i686, 8 for 64-bit targets).
    address_size: u8,
}

impl<'a> DebugLineGenerator<'a> {
    /// Create a new `DebugLineGenerator`.
    ///
    /// Uses [`dwarf_address_size`] to determine the address width (4 or 8
    /// bytes) based on the target architecture.
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture for address-size decisions.
    /// * `source_map` — Source file tracker providing file names and directories.
    pub fn new(target: &'a Target, source_map: &'a SourceMap) -> Self {
        let address_size = dwarf_address_size(target);
        Self {
            buffer: Vec::with_capacity(1024),
            target,
            source_map,
            address_size,
        }
    }

    /// Generate the complete `.debug_line` section bytes.
    ///
    /// Constructs the line number program header (with directory and file
    /// tables built from the [`SourceMap`]), then emits the line number
    /// program opcodes that encode the address-to-source mappings.
    ///
    /// # Arguments
    ///
    /// * `line_entries` — Slice of address-to-source mappings. Entries with
    ///   `end_sequence = true` mark the end of a contiguous address range.
    ///
    /// # Returns
    ///
    /// The complete `.debug_line` section as a byte vector.
    pub fn generate(&mut self, line_entries: &[LineEntry]) -> Vec<u8> {
        self.buffer.clear();

        // Collect file and directory information from the source map.
        let (directories, file_entries) = self.collect_file_info();

        // ---------------------------------------------------------------
        // Unit Length (placeholder — filled in at the end)
        // ---------------------------------------------------------------
        self.buffer.extend_from_slice(&[0u8; 4]);

        // ---------------------------------------------------------------
        // Version (u16 LE) — DWARF v4
        // ---------------------------------------------------------------
        self.buffer.extend_from_slice(&DWARF_VERSION.to_le_bytes());

        // ---------------------------------------------------------------
        // Header Length (placeholder — filled in after header is complete)
        // ---------------------------------------------------------------
        let header_length_offset = self.buffer.len();
        self.buffer.extend_from_slice(&[0u8; 4]);

        // ---------------------------------------------------------------
        // Header Parameters
        // ---------------------------------------------------------------
        self.emit_header_parameters();

        // ---------------------------------------------------------------
        // Include Directory Table
        // ---------------------------------------------------------------
        self.emit_directory_table(&directories);

        // ---------------------------------------------------------------
        // File Name Table
        // ---------------------------------------------------------------
        self.emit_file_table(&file_entries);

        // ---------------------------------------------------------------
        // Fix up header_length
        // ---------------------------------------------------------------
        let header_end = self.buffer.len();
        let header_length = (header_end - header_length_offset - 4) as u32;
        self.buffer[header_length_offset..header_length_offset + 4]
            .copy_from_slice(&header_length.to_le_bytes());

        // ---------------------------------------------------------------
        // Line Number Program Opcodes
        // ---------------------------------------------------------------
        self.emit_line_program(line_entries);

        // ---------------------------------------------------------------
        // Fix up unit_length
        // ---------------------------------------------------------------
        let unit_length = (self.buffer.len() - 4) as u32;
        self.buffer[0..4].copy_from_slice(&unit_length.to_le_bytes());

        // Suppress unused lint for target field which is used indirectly
        // through dwarf_address_size() in new().
        let _ = &self.target;

        std::mem::take(&mut self.buffer)
    }

    // -------------------------------------------------------------------
    // collect_file_info
    // -------------------------------------------------------------------

    /// Collect file and directory information from the source map.
    ///
    /// Iterates all files tracked by the [`SourceMap`] using both
    /// `get_file()` and `get_filename()`, extracting directory paths and
    /// base filenames for the DWARF directory and file tables.
    fn collect_file_info(&self) -> (Vec<String>, Vec<FileTableEntry>) {
        let mut directories: Vec<String> = Vec::new();
        let mut file_entries: Vec<FileTableEntry> = Vec::new();

        let mut file_id = 0u32;
        while let Some(_source_file) = self.source_map.get_file(file_id) {
            // Use get_filename() to retrieve the canonical filename string.
            let full_path = self.source_map.get_filename(file_id).unwrap_or("unknown.c");

            // Split into directory and base filename.
            let (dir, name) = split_path(full_path);

            // Directory index 0 = compilation directory (no explicit path).
            // Indices 1+ reference entries in the directory table.
            let dir_index = if dir.is_empty() {
                0
            } else {
                match directories.iter().position(|d| d == &dir) {
                    Some(pos) => (pos as u32) + 1,
                    None => {
                        directories.push(dir);
                        directories.len() as u32
                    }
                }
            };

            file_entries.push(FileTableEntry {
                name: name.to_string(),
                dir_index,
            });

            file_id += 1;
        }

        // Fallback: ensure the file table is never completely empty.
        if file_entries.is_empty() {
            file_entries.push(FileTableEntry {
                name: "unknown.c".to_string(),
                dir_index: 0,
            });
        }

        (directories, file_entries)
    }

    // -------------------------------------------------------------------
    // emit_header_parameters
    // -------------------------------------------------------------------

    /// Emit the fixed header parameter bytes after the header_length field.
    fn emit_header_parameters(&mut self) {
        self.buffer.push(MINIMUM_INSTRUCTION_LENGTH);
        self.buffer.push(MAXIMUM_OPERATIONS_PER_INSTRUCTION);
        self.buffer.push(DEFAULT_IS_STMT);
        self.buffer.push(LINE_BASE as u8);
        self.buffer.push(LINE_RANGE);
        self.buffer.push(OPCODE_BASE);
        self.buffer.extend_from_slice(&STANDARD_OPCODE_LENGTHS);
    }

    // -------------------------------------------------------------------
    // emit_directory_table
    // -------------------------------------------------------------------

    /// Emit the include directory table.
    ///
    /// Each directory is a null-terminated string. The table is terminated
    /// by an empty string (single `0x00` byte). Directory indices are 1-based.
    fn emit_directory_table(&mut self, directories: &[String]) {
        for dir in directories {
            self.buffer.extend_from_slice(dir.as_bytes());
            self.buffer.push(0); // null terminator
        }
        // Terminate with empty string.
        self.buffer.push(0);
    }

    // -------------------------------------------------------------------
    // emit_file_table
    // -------------------------------------------------------------------

    /// Emit the file name table.
    ///
    /// Each entry: file name (null-terminated), directory index (ULEB128),
    /// last-modified (ULEB128, 0), file length (ULEB128, 0).
    /// Terminated by an empty entry (single `0x00` byte).
    fn emit_file_table(&mut self, file_entries: &[FileTableEntry]) {
        for entry in file_entries {
            self.buffer.extend_from_slice(entry.name.as_bytes());
            self.buffer.push(0);
            encode_uleb128(entry.dir_index as u64, &mut self.buffer);
            encode_uleb128(0, &mut self.buffer); // last modified
            encode_uleb128(0, &mut self.buffer); // file length
        }
        self.buffer.push(0); // terminate file table
    }

    // -------------------------------------------------------------------
    // emit_line_program
    // -------------------------------------------------------------------

    /// Emit the line number program opcodes for all line entries.
    ///
    /// Processes entries in address order, grouping them into sequences
    /// delimited by `end_sequence` markers. Each sequence begins with
    /// `DW_LNE_set_address` and ends with `DW_LNE_end_sequence`.
    fn emit_line_program(&mut self, entries: &[LineEntry]) {
        if entries.is_empty() {
            return;
        }

        // Sort entries by address for correct delta computation.
        let mut sorted: Vec<&LineEntry> = entries.iter().collect();
        sorted.sort_by_key(|e| e.address);

        let mut state = LineStateMachine::new();
        let mut need_set_address = true;

        for entry in &sorted {
            // Start of a new sequence: emit DW_LNE_set_address.
            if need_set_address {
                self.emit_set_address(entry.address);
                state.address = entry.address;
                need_set_address = false;
                // Mark prologue end at the first entry of each sequence.
                self.buffer.push(DW_LNS_SET_PROLOGUE_END);
            }

            // Handle file change.
            if entry.file != state.file {
                self.buffer.push(DW_LNS_SET_FILE);
                encode_uleb128(entry.file as u64, &mut self.buffer);
                state.file = entry.file;
            }

            // Handle column change.
            if entry.column != state.column {
                self.buffer.push(DW_LNS_SET_COLUMN);
                encode_uleb128(entry.column as u64, &mut self.buffer);
                state.column = entry.column;
            }

            // Handle is_stmt toggle.
            if entry.is_stmt != state.is_stmt {
                self.buffer.push(DW_LNS_NEGATE_STMT);
                state.is_stmt = entry.is_stmt;
            }

            // Handle end_sequence entries.
            if entry.end_sequence {
                if entry.address > state.address {
                    let addr_advance = entry.address - state.address;
                    self.buffer.push(DW_LNS_ADVANCE_PC);
                    encode_uleb128(addr_advance, &mut self.buffer);
                    state.address = entry.address;
                }
                self.emit_end_sequence();
                state.reset();
                need_set_address = true;
                continue;
            }

            // Compute deltas.
            let address_advance = entry.address.saturating_sub(state.address);
            let line_advance = (entry.line as i64) - (state.line as i64);

            // Try special opcode (most compact).
            if let Some(opcode) = try_special_opcode(address_advance, line_advance) {
                self.buffer.push(opcode);
                state.address = entry.address;
                state.line = entry.line;
            } else {
                // Fall back to standard opcodes.
                if address_advance > 0 {
                    let const_add_pc_advance = ((255 - OPCODE_BASE as u64) / LINE_RANGE as u64)
                        * MINIMUM_INSTRUCTION_LENGTH as u64;
                    if address_advance >= const_add_pc_advance && const_add_pc_advance > 0 {
                        self.buffer.push(DW_LNS_CONST_ADD_PC);
                        let remaining = address_advance - const_add_pc_advance;
                        state.address += const_add_pc_advance;
                        if remaining > 0 {
                            if let Some(opcode) = try_special_opcode(remaining, line_advance) {
                                self.buffer.push(opcode);
                                state.address = entry.address;
                                state.line = entry.line;
                                continue;
                            }
                            self.buffer.push(DW_LNS_ADVANCE_PC);
                            encode_uleb128(remaining, &mut self.buffer);
                            state.address = entry.address;
                        }
                    } else {
                        self.buffer.push(DW_LNS_ADVANCE_PC);
                        encode_uleb128(address_advance, &mut self.buffer);
                        state.address = entry.address;
                    }
                }

                if line_advance != 0 {
                    self.buffer.push(DW_LNS_ADVANCE_LINE);
                    encode_sleb128(line_advance, &mut self.buffer);
                    state.line = entry.line;
                }

                self.buffer.push(DW_LNS_COPY);
            }
        }

        // Close the final sequence if not already ended.
        if !state.end_sequence && !need_set_address {
            self.emit_end_sequence();
        }
    }

    // -------------------------------------------------------------------
    // emit_set_address
    // -------------------------------------------------------------------

    /// Emit `DW_LNE_set_address` extended opcode.
    ///
    /// Format: `[0x00, uleb128_length, DW_LNE_SET_ADDRESS, addr_bytes...]`
    fn emit_set_address(&mut self, address: u64) {
        self.buffer.push(0x00); // extended opcode marker
        let ext_len = 1u64 + self.address_size as u64;
        encode_uleb128(ext_len, &mut self.buffer);
        self.buffer.push(DW_LNE_SET_ADDRESS);
        emit_address(address, self.address_size, &mut self.buffer);
    }

    // -------------------------------------------------------------------
    // emit_end_sequence
    // -------------------------------------------------------------------

    /// Emit `DW_LNE_end_sequence` extended opcode.
    ///
    /// Appends a row with `end_sequence = true`, then resets the state machine.
    fn emit_end_sequence(&mut self) {
        self.buffer.push(0x00); // extended opcode marker
        encode_uleb128(1, &mut self.buffer); // length = 1
        self.buffer.push(DW_LNE_END_SEQUENCE);
    }
}

// ---------------------------------------------------------------------------
// Special Opcode Computation
// ---------------------------------------------------------------------------

/// Try to encode an (address_advance, line_advance) pair as a special opcode.
///
/// Special opcodes are single bytes (`OPCODE_BASE..=255`) that encode both
/// an address advance and a line advance simultaneously.
///
/// Formula: `opcode = (line_advance - LINE_BASE) + (LINE_RANGE × op_advance) + OPCODE_BASE`
///
/// Returns `Some(opcode)` if representable, `None` otherwise.
fn try_special_opcode(address_advance: u64, line_advance: i64) -> Option<u8> {
    let line_delta = line_advance - (LINE_BASE as i64);
    if line_delta < 0 || line_delta >= LINE_RANGE as i64 {
        return None;
    }
    let op_advance = address_advance; // MINIMUM_INSTRUCTION_LENGTH = 1
    let opcode = (line_delta as u64) + (LINE_RANGE as u64 * op_advance) + (OPCODE_BASE as u64);
    if opcode <= 255 {
        Some(opcode as u8)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Path Splitting Helper
// ---------------------------------------------------------------------------

/// Split a file path into directory and base filename components.
///
/// Uses the last `/` as the separator. If no `/` is found, the directory
/// is empty (meaning the file is in the compilation directory).
fn split_path(path: &str) -> (String, &str) {
    if let Some(pos) = path.rfind('/') {
        (path[..pos].to_string(), &path[pos + 1..])
    } else {
        (String::new(), path)
    }
}

// ---------------------------------------------------------------------------
// LEB128 Encoding Helpers
// ---------------------------------------------------------------------------

/// Encode an unsigned integer as ULEB128.
fn encode_uleb128(mut value: u64, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode a signed integer as SLEB128.
fn encode_sleb128(mut value: i64, output: &mut Vec<u8>) {
    let mut more = true;
    while more {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
            output.push(byte);
        } else {
            output.push(byte | 0x80);
        }
    }
}

// ---------------------------------------------------------------------------
// Address Emission Helper
// ---------------------------------------------------------------------------

/// Emit a target-width address in little-endian format.
fn emit_address(addr: u64, address_size: u8, output: &mut Vec<u8>) {
    match address_size {
        4 => output.extend_from_slice(&(addr as u32).to_le_bytes()),
        8 => output.extend_from_slice(&addr.to_le_bytes()),
        _ => output.extend_from_slice(&addr.to_le_bytes()),
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_entry_creation() {
        let entry = LineEntry {
            address: 0x1000,
            file: 1,
            line: 42,
            column: 5,
            is_stmt: true,
            end_sequence: false,
        };
        assert_eq!(entry.address, 0x1000);
        assert_eq!(entry.file, 1);
        assert_eq!(entry.line, 42);
        assert_eq!(entry.column, 5);
        assert!(entry.is_stmt);
        assert!(!entry.end_sequence);
    }

    #[test]
    fn test_line_entry_clone() {
        let entry = LineEntry {
            address: 0x2000,
            file: 2,
            line: 10,
            column: 0,
            is_stmt: false,
            end_sequence: true,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.address, entry.address);
        assert_eq!(cloned.file, entry.file);
        assert_eq!(cloned.end_sequence, entry.end_sequence);
    }

    #[test]
    fn test_state_machine_initial() {
        let sm = LineStateMachine::new();
        assert_eq!(sm.address, 0);
        assert_eq!(sm.file, 1);
        assert_eq!(sm.line, 1);
        assert_eq!(sm.column, 0);
        assert!(sm.is_stmt);
        assert!(!sm.basic_block);
        assert!(!sm.end_sequence);
        assert!(!sm.prologue_end);
        assert!(!sm.epilogue_begin);
        assert_eq!(sm.isa, 0);
        assert_eq!(sm.discriminator, 0);
    }

    #[test]
    fn test_state_machine_reset() {
        let mut sm = LineStateMachine::new();
        sm.address = 0x1234;
        sm.file = 3;
        sm.line = 100;
        sm.column = 20;
        sm.is_stmt = false;
        sm.end_sequence = true;
        sm.reset();
        assert_eq!(sm.address, 0);
        assert_eq!(sm.file, 1);
        assert_eq!(sm.line, 1);
        assert!(sm.is_stmt);
        assert!(!sm.end_sequence);
    }

    #[test]
    fn test_special_opcode_zero_deltas() {
        // opcode = (0 - (-5)) + (14 * 0) + 13 = 18
        assert_eq!(try_special_opcode(0, 0), Some(18));
    }

    #[test]
    fn test_special_opcode_small_advance() {
        // opcode = (1 - (-5)) + (14 * 1) + 13 = 33
        assert_eq!(try_special_opcode(1, 1), Some(33));
    }

    #[test]
    fn test_special_opcode_negative_line() {
        // opcode = (-5 - (-5)) + (14 * 0) + 13 = 13
        assert_eq!(try_special_opcode(0, -5), Some(13));
    }

    #[test]
    fn test_special_opcode_max_line() {
        // opcode = (8 - (-5)) + (14 * 0) + 13 = 26
        assert_eq!(try_special_opcode(0, 8), Some(26));
    }

    #[test]
    fn test_special_opcode_line_out_of_range() {
        assert!(try_special_opcode(0, 9).is_none());
        assert!(try_special_opcode(0, -6).is_none());
    }

    #[test]
    fn test_special_opcode_large_address() {
        // opcode = 5 + (14 * 20) + 13 = 298 > 255
        assert!(try_special_opcode(20, 0).is_none());
    }

    #[test]
    fn test_special_opcode_max_representable() {
        // addr=16: opcode = 5 + 14*16 + 13 = 242
        assert_eq!(try_special_opcode(16, 0), Some(242));
        // addr=17: opcode = 5 + 14*17 + 13 = 256 > 255
        assert!(try_special_opcode(17, 0).is_none());
    }

    #[test]
    fn test_uleb128_zero() {
        let mut buf = Vec::new();
        encode_uleb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_uleb128_single_byte() {
        let mut buf = Vec::new();
        encode_uleb128(127, &mut buf);
        assert_eq!(buf, vec![0x7F]);
    }

    #[test]
    fn test_uleb128_two_bytes() {
        let mut buf = Vec::new();
        encode_uleb128(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn test_uleb128_large() {
        let mut buf = Vec::new();
        encode_uleb128(624485, &mut buf);
        assert_eq!(buf, vec![0xE5, 0x8E, 0x26]);
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
    fn test_split_path_with_directory() {
        let (dir, name) = split_path("src/foo/bar.c");
        assert_eq!(dir, "src/foo");
        assert_eq!(name, "bar.c");
    }

    #[test]
    fn test_split_path_no_directory() {
        let (dir, name) = split_path("hello.c");
        assert_eq!(dir, "");
        assert_eq!(name, "hello.c");
    }

    #[test]
    fn test_split_path_absolute() {
        let (dir, name) = split_path("/absolute/path/main.c");
        assert_eq!(dir, "/absolute/path");
        assert_eq!(name, "main.c");
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

    #[test]
    fn test_generator_creation() {
        let source_map = SourceMap::new();
        let target = Target::X86_64;
        let gen = DebugLineGenerator::new(&target, &source_map);
        assert_eq!(gen.address_size, 8);
    }

    #[test]
    fn test_generator_creation_i686() {
        let source_map = SourceMap::new();
        let target = Target::I686;
        let gen = DebugLineGenerator::new(&target, &source_map);
        assert_eq!(gen.address_size, 4);
    }

    #[test]
    fn test_generate_empty_entries() {
        let source_map = SourceMap::new();
        let target = Target::X86_64;
        let mut gen = DebugLineGenerator::new(&target, &source_map);
        let bytes = gen.generate(&[]);

        assert!(bytes.len() > 10);
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 4);
    }

    #[test]
    fn test_generate_with_source_file() {
        let mut source_map = SourceMap::new();
        source_map.add_file("test.c".to_string(), "int main() {}\n".to_string());
        let target = Target::X86_64;
        let mut gen = DebugLineGenerator::new(&target, &source_map);

        let entries = vec![
            LineEntry {
                address: 0,
                file: 1,
                line: 1,
                column: 0,
                is_stmt: true,
                end_sequence: false,
            },
            LineEntry {
                address: 16,
                file: 1,
                line: 1,
                column: 0,
                is_stmt: true,
                end_sequence: true,
            },
        ];
        let bytes = gen.generate(&entries);

        assert!(bytes.len() > 20);
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 4);
    }

    #[test]
    fn test_generate_multiple_sequences() {
        let mut source_map = SourceMap::new();
        source_map.add_file("multi.c".to_string(), "void f(){} void g(){}\n".to_string());
        let target = Target::AArch64;
        let mut gen = DebugLineGenerator::new(&target, &source_map);

        let entries = vec![
            LineEntry {
                address: 0,
                file: 1,
                line: 1,
                column: 0,
                is_stmt: true,
                end_sequence: false,
            },
            LineEntry {
                address: 20,
                file: 1,
                line: 1,
                column: 0,
                is_stmt: true,
                end_sequence: true,
            },
            LineEntry {
                address: 24,
                file: 1,
                line: 2,
                column: 0,
                is_stmt: true,
                end_sequence: false,
            },
            LineEntry {
                address: 40,
                file: 1,
                line: 2,
                column: 0,
                is_stmt: true,
                end_sequence: true,
            },
        ];
        let bytes = gen.generate(&entries);

        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_generate_with_directory_files() {
        let mut source_map = SourceMap::new();
        source_map.add_file("src/main.c".to_string(), "int main(){}\n".to_string());
        source_map.add_file("src/util.c".to_string(), "void util(){}\n".to_string());
        source_map.add_file("include/header.h".to_string(), "/* h */\n".to_string());
        let target = Target::RiscV64;
        let mut gen = DebugLineGenerator::new(&target, &source_map);

        let entries = vec![
            LineEntry {
                address: 0,
                file: 1,
                line: 1,
                column: 0,
                is_stmt: true,
                end_sequence: false,
            },
            LineEntry {
                address: 32,
                file: 1,
                line: 1,
                column: 0,
                is_stmt: true,
                end_sequence: true,
            },
        ];
        let bytes = gen.generate(&entries);

        let main_c = b"main.c";
        assert!(bytes.windows(main_c.len()).any(|w| w == main_c));
    }

    #[test]
    fn test_header_opcode_base() {
        assert_eq!(OPCODE_BASE, 13);
        assert_eq!(STANDARD_OPCODE_LENGTHS.len(), 12);
    }

    #[test]
    fn test_special_opcode_formula_consistency() {
        for addr in 0..=16u64 {
            for line in -5..=8i64 {
                if let Some(opcode) = try_special_opcode(addr, line) {
                    assert!(opcode >= OPCODE_BASE);
                    let adjusted = (opcode - OPCODE_BASE) as u64;
                    let decoded_line = (adjusted % LINE_RANGE as u64) as i64 + LINE_BASE as i64;
                    let decoded_addr = adjusted / LINE_RANGE as u64;
                    assert_eq!(decoded_addr, addr);
                    assert_eq!(decoded_line, line);
                }
            }
        }
    }

    #[test]
    fn test_constants_consistency() {
        assert_eq!(LINE_BASE, -5);
        assert_eq!(LINE_RANGE, 14);
        assert_eq!(MINIMUM_INSTRUCTION_LENGTH, 1);
        assert_eq!(MAXIMUM_OPERATIONS_PER_INSTRUCTION, 1);
        assert_eq!(DEFAULT_IS_STMT, 1);
    }
}
