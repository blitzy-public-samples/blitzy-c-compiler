// src/backend/dwarf/line.rs
//
// DWARF v4 `.debug_line` section generator.
//
// Produces the line number program that maps machine code addresses to source
// file locations, enabling source-level debugging in GDB and other debuggers.
//
// The generator constructs:
// - Line number program header (opcode_base, standard opcode lengths, directory
//   table, file table)
// - Standard opcodes (DW_LNS_advance_pc, DW_LNS_advance_line, DW_LNS_set_file,
//   DW_LNS_copy, DW_LNS_set_column, DW_LNS_negate_stmt, DW_LNS_const_add_pc)
// - Extended opcodes (DW_LNE_set_address, DW_LNE_end_sequence)
// - Special opcodes for compressed line-to-address mapping
//
// Implements the DWARF v4 line number state machine register model:
// address, file, line, column, is_stmt, basic_block, end_sequence,
// prologue_end, epilogue_begin, isa, discriminator.
//
// Scoped to `-O0` only — no optimized debug information is produced.
//
// Integration with the SourceMap module allows automatic population of file
// and directory tables from the compiler's source file registry, and
// resolving byte offsets to line/column for line program entries.

use crate::common::source_map::{FileId, SourceMap};
use crate::common::target::Target;

// ===========================================================================
// DWARF v4 Line Number Program Constants
// ===========================================================================

// --- Standard opcodes (DWARF v4, Section 6.2.5.2) ---

/// `DW_LNS_copy` — Append a row to the matrix using current register values,
/// then reset `basic_block`, `prologue_end`, `epilogue_begin`, `discriminator`.
const DW_LNS_COPY: u8 = 1;

/// `DW_LNS_advance_pc` — Takes one ULEB128 operand (operation advance) and
/// adds `minimum_instruction_length * operation_advance` to `address`.
const DW_LNS_ADVANCE_PC: u8 = 2;

/// `DW_LNS_advance_line` — Takes one SLEB128 operand and adds it to `line`.
const DW_LNS_ADVANCE_LINE: u8 = 3;

/// `DW_LNS_set_file` — Takes one ULEB128 operand and sets `file`.
const DW_LNS_SET_FILE: u8 = 4;

/// `DW_LNS_set_column` — Takes one ULEB128 operand and sets `column`.
const DW_LNS_SET_COLUMN: u8 = 5;

/// `DW_LNS_negate_stmt` — Toggles the `is_stmt` register.
const DW_LNS_NEGATE_STMT: u8 = 6;

/// `DW_LNS_set_basic_block` — Sets `basic_block` to true.
#[allow(dead_code)]
const DW_LNS_SET_BASIC_BLOCK: u8 = 7;

/// `DW_LNS_const_add_pc` — Advances `address` by the increment corresponding
/// to special opcode 255 (the maximum special opcode address advance).
#[allow(dead_code)]
const DW_LNS_CONST_ADD_PC: u8 = 8;

/// `DW_LNS_fixed_advance_pc` — Takes one u16 operand, advances `address`
/// by exactly that amount (no minimum_instruction_length scaling).
#[allow(dead_code)]
const DW_LNS_FIXED_ADVANCE_PC: u8 = 9;

/// `DW_LNS_set_prologue_end` — Sets `prologue_end` to true.
#[allow(dead_code)]
const DW_LNS_SET_PROLOGUE_END: u8 = 10;

/// `DW_LNS_set_epilogue_begin` — Sets `epilogue_begin` to true.
#[allow(dead_code)]
const DW_LNS_SET_EPILOGUE_BEGIN: u8 = 11;

/// `DW_LNS_set_isa` — Takes one ULEB128 operand and sets `isa`.
#[allow(dead_code)]
const DW_LNS_SET_ISA: u8 = 12;

// --- Extended opcodes (DWARF v4, Section 6.2.5.3) ---

/// `DW_LNE_end_sequence` — Sets `end_sequence` to true, appends a row,
/// then resets all registers to initial state.
const DW_LNE_END_SEQUENCE: u8 = 1;

/// `DW_LNE_set_address` — Sets `address` to a relocatable target address.
const DW_LNE_SET_ADDRESS: u8 = 2;

/// `DW_LNE_define_file` — Defines a new file entry inline within the program.
#[allow(dead_code)]
const DW_LNE_DEFINE_FILE: u8 = 3;

/// `DW_LNE_set_discriminator` — Sets the `discriminator` register.
#[allow(dead_code)]
const DW_LNE_SET_DISCRIMINATOR: u8 = 4;

// --- Default line number program parameters ---

/// Opcode base: first special opcode value. Standard opcodes are numbered
/// 1 through `OPCODE_BASE - 1`. Special opcodes range from `OPCODE_BASE`
/// through 255.
const OPCODE_BASE: u8 = 13;

/// Smallest line number increment representable by a special opcode.
/// Special opcodes can encode line deltas in the range
/// `[LINE_BASE, LINE_BASE + LINE_RANGE - 1]` = `[-5, 8]`.
const LINE_BASE: i8 = -5;

/// Number of distinct line increments representable by a special opcode.
/// Together with `LINE_BASE`, defines the encodable range of line deltas.
const LINE_RANGE: u8 = 14;

/// Default value for the `is_stmt` register. 1 = every instruction that
/// begins a source statement is a recommended breakpoint location.
const DEFAULT_IS_STMT: u8 = 1;

/// Maximum operations per instruction. Set to 1 for all BCC targets
/// (none are VLIW architectures).
const MAX_OPS_PER_INSN: u8 = 1;

/// Number of operands for each standard opcode (opcodes 1 through 12).
///
/// This array is written directly into the `.debug_line` header. Each entry
/// tells a DWARF consumer how many ULEB128/SLEB128 operands follow the
/// opcode byte, enabling forward-compatible parsing of unknown standard
/// opcodes.
///
/// Index 0 → opcode 1 (`DW_LNS_copy`), index 11 → opcode 12 (`DW_LNS_set_isa`).
const STANDARD_OPCODE_LENGTHS: [u8; 12] = [
    0, // DW_LNS_copy (1):              0 operands
    1, // DW_LNS_advance_pc (2):        1 ULEB128 operand
    1, // DW_LNS_advance_line (3):      1 SLEB128 operand
    1, // DW_LNS_set_file (4):          1 ULEB128 operand
    1, // DW_LNS_set_column (5):        1 ULEB128 operand
    0, // DW_LNS_negate_stmt (6):       0 operands
    0, // DW_LNS_set_basic_block (7):   0 operands
    0, // DW_LNS_const_add_pc (8):      0 operands
    1, // DW_LNS_fixed_advance_pc (9):  1 uhalf operand
    0, // DW_LNS_set_prologue_end (10): 0 operands
    0, // DW_LNS_set_epilogue_begin(11):0 operands
    1, // DW_LNS_set_isa (12):          1 ULEB128 operand
];

/// DWARF version number written into the `.debug_line` section header.
const DWARF_VERSION: u16 = 4;

// ===========================================================================
// FileEntry
// ===========================================================================

/// Represents a file entry in the DWARF `.debug_line` file table.
///
/// Each entry records metadata about a source file referenced by the line
/// number program. In DWARF v4, file indices are 1-based; index 0 is not
/// used for file entries.
///
/// Fields are serialized into the `.debug_line` header as:
/// - null-terminated file name string
/// - ULEB128 directory index
/// - ULEB128 modification time (0 = unknown)
/// - ULEB128 file size (0 = unknown)
#[derive(Clone, Debug)]
pub struct FileEntry {
    /// File name (may include directory components if `dir_index` is 0).
    pub name: String,
    /// Index into the directory table. 0 = compilation directory (implicit).
    pub dir_index: u32,
    /// Last modification timestamp. 0 if unknown.
    pub mod_time: u64,
    /// File size in bytes. 0 if unknown.
    pub file_size: u64,
}

// ===========================================================================
// LineState — Line number state machine registers (internal)
// ===========================================================================

/// DWARF line number state machine registers.
///
/// The state machine conceptually maintains these registers while executing
/// opcodes in the line number program. A "row" is appended to the line number
/// matrix when `DW_LNS_copy` or a special opcode is executed, or when
/// `DW_LNE_end_sequence` terminates a sequence.
///
/// All register names and semantics match DWARF v4 Section 6.2.2.
///
/// Some registers (`isa`, `end_sequence`, `prologue_end`, `epilogue_begin`,
/// `discriminator`) are maintained per the DWARF spec but may not be explicitly
/// read in all code paths; they exist to track state machine semantics.
#[allow(dead_code)]
struct LineState {
    /// Current program counter address (absolute byte address).
    address: u64,
    /// Current source file (1-based index into the file table).
    file: u32,
    /// Current source line number (1-based).
    line: u32,
    /// Current source column number (0 = column info unavailable).
    column: u32,
    /// Whether the current instruction is a recommended breakpoint position.
    is_stmt: bool,
    /// Whether the current instruction begins a basic block.
    basic_block: bool,
    /// Set when the address is one past the last instruction in a sequence.
    end_sequence: bool,
    /// Whether the current address is immediately after a function prologue.
    prologue_end: bool,
    /// Whether the current address is immediately before a function epilogue.
    epilogue_begin: bool,
    /// Instruction set architecture selector (for multi-ISA targets).
    isa: u32,
    /// Discriminator for distinguishing multiple blocks on the same source line.
    discriminator: u32,
}

impl LineState {
    /// Creates a new `LineState` with DWARF v4 initial register values.
    ///
    /// Initial state per DWARF v4 Section 6.2.2:
    /// - `address` = 0
    /// - `file` = 1
    /// - `line` = 1
    /// - `column` = 0
    /// - `is_stmt` = `default_is_stmt` (true)
    /// - All flags = false, isa = 0, discriminator = 0
    fn new() -> Self {
        LineState {
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

    /// Resets the registers that are cleared after appending a matrix row
    /// (via `DW_LNS_copy` or a special opcode).
    ///
    /// Per DWARF v4 Section 6.2.5.1: after copying, `basic_block`,
    /// `prologue_end`, `epilogue_begin`, and `discriminator` are reset.
    fn reset_after_copy(&mut self) {
        self.basic_block = false;
        self.prologue_end = false;
        self.epilogue_begin = false;
        self.discriminator = 0;
    }

    /// Resets all registers to their initial state.
    ///
    /// Called after `DW_LNE_end_sequence` terminates a sequence.
    fn reset(&mut self) {
        *self = LineState::new();
    }
}

// ===========================================================================
// LineNumberProgramBuilder
// ===========================================================================

/// Builder for the DWARF v4 `.debug_line` section.
///
/// Constructs the complete `.debug_line` section including the line number
/// program header (with directory and file tables) and the line number program
/// opcodes. The builder serializes everything into a byte vector that can be
/// directly emitted into the ELF `.debug_line` section.
///
/// # Usage
///
/// ```text
/// let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
/// let dir_idx = builder.add_directory("/path/to/src");
/// let file_idx = builder.add_file("main.c", dir_idx);
/// builder.emit_header();
///
/// // Emit line number program for a function
/// builder.emit_set_address(0x401000);
/// builder.emit_set_file(file_idx);
/// builder.emit_line_advance(0, 0);       // First line at start address
/// builder.emit_line_advance(10, 1);      // +10 bytes, +1 line
/// builder.emit_end_sequence();
///
/// let section_bytes = builder.finish();
/// ```
///
/// # Architecture Dependence
///
/// The target architecture affects two header parameters:
/// - `minimum_instruction_length`: 1 for x86-64/i686 (variable-length), 4
///   for AArch64/RISC-V 64 (fixed 32-bit instructions)
/// - `address_size`: 4 for i686, 8 for 64-bit targets — controls the
///   address field width in `DW_LNE_set_address`
pub struct LineNumberProgramBuilder {
    /// Serialized `.debug_line` section bytes (header + program opcodes).
    data: Vec<u8>,
    /// Directory table entries. Index 0 in DWARF is the compilation directory
    /// (implicit); entries here are indexed 1-based in the DWARF table.
    directories: Vec<String>,
    /// File table entries (each references a directory by index).
    files: Vec<FileEntry>,
    /// Target architecture — determines minimum_instruction_length and address_size.
    target: Target,
    /// Byte offset of the `unit_length` field in `data` (for backpatching in `finish()`).
    unit_length_offset: usize,
    /// Byte offset of the `header_length` field in `data` (backpatched in `emit_header()`).
    header_length_offset: usize,
    /// Current line number state machine registers (tracks implicit state for
    /// delta computations and validation).
    state: LineState,
}

impl LineNumberProgramBuilder {
    /// Creates a new, empty line number program builder for the given target.
    ///
    /// The builder starts with no directories, no files, and an empty data
    /// buffer. Call `add_directory()` / `add_file()` to populate the tables,
    /// then `emit_header()` to write the header, followed by program opcodes,
    /// and finally `finish()` to obtain the complete section bytes.
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture. Used to determine
    ///   `minimum_instruction_length` (1 for x86-64/i686, 4 for AArch64/RISC-V 64)
    ///   and `address_size` (4 for i686, 8 for 64-bit targets).
    pub fn new(target: Target) -> Self {
        LineNumberProgramBuilder {
            data: Vec::with_capacity(512),
            directories: Vec::new(),
            files: Vec::new(),
            target,
            unit_length_offset: 0,
            header_length_offset: 0,
            state: LineState::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Directory and File Table Construction
    // -----------------------------------------------------------------------

    /// Adds a directory to the include directory table.
    ///
    /// In DWARF v4, directory index 0 is implicitly the compilation directory
    /// and is NOT stored in the table. Directories added here are assigned
    /// 1-based indices starting from 1.
    ///
    /// # Arguments
    ///
    /// * `dir` — Directory path string (UTF-8).
    ///
    /// # Returns
    ///
    /// The 1-based directory index for use in subsequent `add_file()` calls.
    pub fn add_directory(&mut self, dir: &str) -> u32 {
        // Check if this directory already exists to avoid duplicates
        for (i, existing) in self.directories.iter().enumerate() {
            if existing == dir {
                return (i + 1) as u32;
            }
        }
        self.directories.push(dir.to_string());
        self.directories.len() as u32
    }

    /// Adds a file entry to the file name table.
    ///
    /// In DWARF v4, file indices are 1-based. The returned index can be
    /// used with `emit_set_file()` to reference this file in the line
    /// number program.
    ///
    /// # Arguments
    ///
    /// * `name` — File name (without directory prefix if `dir_index > 0`).
    /// * `dir_index` — Index into the directory table. 0 = compilation
    ///   directory; otherwise the value returned from `add_directory()`.
    ///
    /// # Returns
    ///
    /// The 1-based file index for use in `emit_set_file()` calls.
    pub fn add_file(&mut self, name: &str, dir_index: u32) -> u32 {
        self.files.push(FileEntry {
            name: name.to_string(),
            dir_index,
            mod_time: 0,
            file_size: 0,
        });
        self.files.len() as u32
    }

    // -----------------------------------------------------------------------
    // SourceMap Integration
    // -----------------------------------------------------------------------

    /// Populates the directory and file tables from a `SourceMap`.
    ///
    /// For each provided `FileId`, resolves the file's name via
    /// `SourceMap::get_file()`, splits it into directory and base name
    /// components, and registers them in the builder's tables. Returns a
    /// mapping from each input `FileId` position to its DWARF 1-based file
    /// index, suitable for passing to `emit_set_file()`.
    ///
    /// # Arguments
    ///
    /// * `source_map` — The compiler's source file registry.
    /// * `file_ids` — Slice of `FileId`s for all source files used in this
    ///   compilation unit.
    ///
    /// # Returns
    ///
    /// A `Vec<u32>` where element `i` is the DWARF file index corresponding
    /// to `file_ids[i]`.
    pub fn add_files_from_source_map(
        &mut self,
        source_map: &SourceMap,
        file_ids: &[FileId],
    ) -> Vec<u32> {
        let mut dwarf_indices = Vec::with_capacity(file_ids.len());
        for &fid in file_ids {
            let source_file = source_map.get_file(fid);
            let (dir_path, base_name) = split_path(&source_file.name);
            let dir_index = if dir_path.is_empty() {
                0 // compilation directory
            } else {
                self.add_directory(dir_path)
            };
            let file_idx = self.add_file(base_name, dir_index);
            dwarf_indices.push(file_idx);
        }
        dwarf_indices
    }

    /// Emits a line number entry by resolving a source location from the
    /// `SourceMap`.
    ///
    /// Given a byte offset within a source file and the corresponding machine
    /// code address, uses `SourceMap::lookup_location()` to resolve the byte
    /// offset to a line/column, then emits the appropriate line program
    /// opcodes (set_file, set_column, and a special/standard opcode for
    /// the combined address+line advance).
    ///
    /// # Arguments
    ///
    /// * `source_map` — The compiler's source file registry.
    /// * `file_id` — The source file containing the code.
    /// * `byte_offset` — Byte offset within the source file content.
    /// * `machine_address` — Absolute machine code address.
    /// * `dwarf_file_index` — The DWARF 1-based file index (from
    ///   `add_files_from_source_map()` or `add_file()`).
    pub fn emit_location_from_source_map(
        &mut self,
        source_map: &SourceMap,
        file_id: FileId,
        byte_offset: u32,
        machine_address: u64,
        dwarf_file_index: u32,
    ) {
        let loc = source_map.lookup_location(file_id, byte_offset);

        // Update file register if changed
        if self.state.file != dwarf_file_index {
            self.emit_set_file(dwarf_file_index);
        }

        // Update column register if changed
        if self.state.column != loc.column {
            self.emit_set_column(loc.column);
        }

        // Compute deltas from current state
        let address_delta = machine_address.saturating_sub(self.state.address);
        let line_delta = loc.line as i64 - self.state.line as i64;

        // Emit the combined address + line advance (or fallback)
        self.emit_line_advance(address_delta, line_delta);
    }

    // -----------------------------------------------------------------------
    // Line Number Program Header
    // -----------------------------------------------------------------------

    /// Emits the complete `.debug_line` section header.
    ///
    /// Writes the DWARF v4 line number program header including:
    /// - `unit_length` (4 bytes, placeholder — backpatched by `finish()`)
    /// - `version` (2 bytes = 4)
    /// - `header_length` (4 bytes — backpatched immediately after file table)
    /// - Program parameters (`minimum_instruction_length`, `line_base`, etc.)
    /// - Standard opcode lengths array (12 entries)
    /// - Include directory table (null-terminated strings, terminated by null byte)
    /// - File name table (entries, terminated by null byte)
    ///
    /// **Must be called after** all directories and files have been added via
    /// `add_directory()` / `add_file()`, and **before** any program opcodes
    /// are emitted.
    pub fn emit_header(&mut self) {
        // -- unit_length (32-bit DWARF format) --
        // Record position for later backpatching in finish().
        self.unit_length_offset = self.data.len();
        write_u32_le(&mut self.data, 0); // placeholder

        // -- version --
        write_u16_le(&mut self.data, DWARF_VERSION);

        // -- header_length --
        // Record position for backpatching at the end of this method.
        self.header_length_offset = self.data.len();
        write_u32_le(&mut self.data, 0); // placeholder

        // -- Fixed header fields --
        self.data.push(self.minimum_instruction_length());
        self.data.push(MAX_OPS_PER_INSN);
        self.data.push(DEFAULT_IS_STMT);
        self.data.push(LINE_BASE as u8);
        self.data.push(LINE_RANGE);
        self.data.push(OPCODE_BASE);

        // -- standard_opcode_lengths (12 entries for opcodes 1–12) --
        for &len in &STANDARD_OPCODE_LENGTHS {
            self.data.push(len);
        }

        // -- Include directory table --
        // Each entry: null-terminated string. Table ends with empty string (single 0 byte).
        for dir in &self.directories {
            self.data.extend_from_slice(dir.as_bytes());
            self.data.push(0); // null terminator for this directory string
        }
        self.data.push(0); // end-of-directory-table marker

        // -- File name table --
        // Each entry: null-terminated name, ULEB128 dir_index, ULEB128 mod_time,
        //             ULEB128 file_size. Table ends with empty name (single 0 byte).
        for file in &self.files {
            self.data.extend_from_slice(file.name.as_bytes());
            self.data.push(0); // null terminator for file name
            write_uleb128(&mut self.data, file.dir_index as u64);
            write_uleb128(&mut self.data, file.mod_time);
            write_uleb128(&mut self.data, file.file_size);
        }
        self.data.push(0); // end-of-file-table marker

        // -- Backpatch header_length --
        // header_length = distance from the byte after the header_length field
        // to the first program opcode (the current position).
        let header_end = self.data.len();
        let header_length = header_end - (self.header_length_offset + 4);
        patch_u32_le(
            &mut self.data,
            self.header_length_offset,
            header_length as u32,
        );
    }

    // -----------------------------------------------------------------------
    // Target Helpers
    // -----------------------------------------------------------------------

    /// Returns the minimum instruction length for this target architecture.
    ///
    /// - x86-64 / i686: 1 byte (variable-length instruction encoding)
    /// - AArch64 / RISC-V 64: 4 bytes (fixed 32-bit instruction width)
    fn minimum_instruction_length(&self) -> u8 {
        match self.target {
            Target::X86_64 | Target::I686 => 1,
            Target::AArch64 | Target::RiscV64 => 4,
        }
    }

    /// Returns the address size in bytes for this target architecture.
    ///
    /// - i686: 4 bytes (32-bit addresses)
    /// - x86-64 / AArch64 / RISC-V 64: 8 bytes (64-bit addresses)
    fn address_size(&self) -> u8 {
        self.target.pointer_width() as u8
    }

    // -----------------------------------------------------------------------
    // Extended Opcode Emission
    // -----------------------------------------------------------------------

    /// Emits a `DW_LNE_set_address` extended opcode.
    ///
    /// Sets the address register of the line number state machine to an
    /// absolute machine code address. Typically emitted at the start of
    /// each function's contribution to the line number program.
    ///
    /// The address is encoded in target-native width (4 bytes for i686,
    /// 8 bytes for 64-bit targets).
    ///
    /// # Arguments
    ///
    /// * `address` — Absolute machine code address (byte address).
    pub fn emit_set_address(&mut self, address: u64) {
        let addr_size = self.address_size();
        // Extended opcode format: 0x00, ULEB128(length), DW_LNE_SET_ADDRESS, address
        // length = 1 (opcode byte) + addr_size (address bytes)
        self.data.push(0); // extended opcode marker
        write_uleb128(&mut self.data, 1 + addr_size as u64);
        self.data.push(DW_LNE_SET_ADDRESS);
        if addr_size == 4 {
            write_u32_le(&mut self.data, address as u32);
        } else {
            write_u64_le(&mut self.data, address);
        }
        self.state.address = address;
    }

    /// Emits a `DW_LNE_end_sequence` extended opcode.
    ///
    /// Marks the end of a sequence of target machine instructions. This
    /// appends a row to the matrix with `end_sequence = true`, then resets
    /// all state machine registers to their initial values.
    ///
    /// Every sequence started with `emit_set_address()` must be terminated
    /// with `emit_end_sequence()`.
    pub fn emit_end_sequence(&mut self) {
        // Extended opcode format: 0x00, ULEB128(length), DW_LNE_END_SEQUENCE
        self.data.push(0); // extended opcode marker
        write_uleb128(&mut self.data, 1); // length = 1 (just the opcode byte)
        self.data.push(DW_LNE_END_SEQUENCE);
        self.state.end_sequence = true;
        // Reset to initial state after end_sequence
        self.state.reset();
    }

    // -----------------------------------------------------------------------
    // Standard Opcode Emission
    // -----------------------------------------------------------------------

    /// Emits a `DW_LNS_advance_pc` standard opcode.
    ///
    /// Advances the address register by `delta` bytes. The DWARF operand
    /// is the "operation advance" (= `delta / minimum_instruction_length`),
    /// encoded as ULEB128.
    ///
    /// For x86 targets (`minimum_instruction_length = 1`), the operation
    /// advance equals the byte delta directly. For AArch64/RISC-V
    /// (`minimum_instruction_length = 4`), the operation advance is the
    /// number of 4-byte instructions.
    ///
    /// # Arguments
    ///
    /// * `delta` — Address advance in bytes.
    pub fn emit_advance_pc(&mut self, delta: u64) {
        let min_len = self.minimum_instruction_length() as u64;
        // Compute operation advance: number of minimum-instruction-length units
        let op_advance = if min_len > 0 { delta / min_len } else { delta };
        self.data.push(DW_LNS_ADVANCE_PC);
        write_uleb128(&mut self.data, op_advance);
        // Update state: address advances by op_advance * min_len (which may
        // differ from delta if delta is not a multiple of min_len, but we
        // follow the DWARF spec which truncates)
        self.state.address = self.state.address.wrapping_add(op_advance * min_len);
    }

    /// Emits a `DW_LNS_advance_line` standard opcode.
    ///
    /// Advances the line register by the signed `delta` value, encoded
    /// as SLEB128.
    ///
    /// # Arguments
    ///
    /// * `delta` — Line number advance (positive or negative).
    pub fn emit_advance_line(&mut self, delta: i64) {
        self.data.push(DW_LNS_ADVANCE_LINE);
        write_sleb128(&mut self.data, delta);
        self.state.line = (self.state.line as i64 + delta) as u32;
    }

    /// Emits a `DW_LNS_set_file` standard opcode.
    ///
    /// Sets the file register to the specified 1-based file index.
    ///
    /// # Arguments
    ///
    /// * `file_index` — 1-based file index from the file table.
    pub fn emit_set_file(&mut self, file_index: u32) {
        self.data.push(DW_LNS_SET_FILE);
        write_uleb128(&mut self.data, file_index as u64);
        self.state.file = file_index;
    }

    /// Emits a `DW_LNS_set_column` standard opcode.
    ///
    /// Sets the column register.
    ///
    /// # Arguments
    ///
    /// * `column` — Column number. 0 indicates column information is unavailable.
    pub fn emit_set_column(&mut self, column: u32) {
        self.data.push(DW_LNS_SET_COLUMN);
        write_uleb128(&mut self.data, column as u64);
        self.state.column = column;
    }

    /// Emits a `DW_LNS_copy` standard opcode.
    ///
    /// Appends a row to the line number matrix using the current register
    /// values, then resets `basic_block`, `prologue_end`, `epilogue_begin`,
    /// and `discriminator`.
    pub fn emit_copy(&mut self) {
        self.data.push(DW_LNS_COPY);
        self.state.reset_after_copy();
    }

    /// Emits a `DW_LNS_negate_stmt` standard opcode.
    ///
    /// Toggles the `is_stmt` register: if it was `true` it becomes `false`,
    /// and vice versa.
    pub fn emit_negate_stmt(&mut self) {
        self.data.push(DW_LNS_NEGATE_STMT);
        self.state.is_stmt = !self.state.is_stmt;
    }

    // -----------------------------------------------------------------------
    // Special Opcode Emission (Compressed Line/Address Deltas)
    // -----------------------------------------------------------------------

    /// Emits the most compact encoding for a simultaneous address and line advance.
    ///
    /// Attempts to use a DWARF special opcode if the deltas are small enough.
    /// A single special opcode byte simultaneously advances both address and
    /// line, then appends a matrix row — this is the most compact encoding
    /// and the primary mechanism for efficient `.debug_line` encoding.
    ///
    /// If the deltas cannot be represented as a single special opcode, the
    /// method falls back to emitting separate `DW_LNS_advance_pc` +
    /// `DW_LNS_advance_line` + `DW_LNS_copy`.
    ///
    /// # Arguments
    ///
    /// * `address_delta` — Address advance in bytes.
    /// * `line_delta` — Line number advance (signed, may be negative).
    pub fn emit_line_advance(&mut self, address_delta: u64, line_delta: i64) {
        let min_len = self.minimum_instruction_length() as u64;
        let op_advance = if min_len > 0 {
            address_delta / min_len
        } else {
            address_delta
        };

        // Try to encode as a single special opcode
        if let Some(opcode) = compute_special_opcode(line_delta, op_advance) {
            self.data.push(opcode);
            // Update state: both address and line advance
            self.state.address = self.state.address.wrapping_add(op_advance * min_len);
            self.state.line = (self.state.line as i64 + line_delta) as u32;
            // Special opcodes implicitly copy and reset flags
            self.state.reset_after_copy();
        } else {
            // Fall back to separate opcodes
            if address_delta > 0 {
                self.emit_advance_pc(address_delta);
            }
            if line_delta != 0 {
                self.emit_advance_line(line_delta);
            }
            self.emit_copy();
        }
    }

    // -----------------------------------------------------------------------
    // Section Finalization
    // -----------------------------------------------------------------------

    /// Finalizes the `.debug_line` section and returns the complete section bytes.
    ///
    /// Backpatches the `unit_length` field in the header to reflect the
    /// actual total section size. After calling this method, the builder
    /// should not be used further (though it is not consumed, to allow
    /// callers flexibility).
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the complete, self-consistent `.debug_line`
    /// section content ready to be written into an ELF file.
    pub fn finish(&mut self) -> Vec<u8> {
        // Backpatch unit_length: total size minus the 4 bytes of the
        // unit_length field itself.
        let total = self.data.len();
        let unit_length = total - (self.unit_length_offset + 4);
        patch_u32_le(
            &mut self.data,
            self.unit_length_offset,
            unit_length as u32,
        );
        self.data.clone()
    }

    /// Returns the byte offset of this compilation unit's contribution
    /// within the `.debug_line` section.
    ///
    /// This value is used for the `DW_AT_stmt_list` attribute in `.debug_info`
    /// to reference the start of the line number program for a compilation unit.
    ///
    /// # Returns
    ///
    /// Byte offset into the `.debug_line` section (typically 0 for the
    /// first/only compilation unit).
    pub fn section_offset(&self) -> u32 {
        self.unit_length_offset as u32
    }
}

// ===========================================================================
// Special Opcode Computation
// ===========================================================================

/// Computes a DWARF special opcode for simultaneous address and line advancement.
///
/// Special opcodes are single-byte opcodes in the range `[OPCODE_BASE, 255]`
/// that simultaneously advance both the address and line registers. This
/// provides the most compact encoding for the common case of sequential
/// address/line changes.
///
/// # Formula
///
/// ```text
/// adjusted_opcode = (line_delta - LINE_BASE) + (LINE_RANGE * op_advance)
/// opcode = adjusted_opcode + OPCODE_BASE
/// ```
///
/// # Constraints
///
/// - `line_delta` must be in `[LINE_BASE, LINE_BASE + LINE_RANGE - 1]` = `[-5, 8]`
/// - The resulting `opcode` must be in `[OPCODE_BASE, 255]`
///
/// # Arguments
///
/// * `line_delta` — Signed line number change.
/// * `op_advance` — Operation advance (= address delta / minimum_instruction_length).
///
/// # Returns
///
/// `Some(opcode)` if representable as a special opcode, `None` otherwise.
fn compute_special_opcode(line_delta: i64, op_advance: u64) -> Option<u8> {
    // Check that line_delta is within the representable range
    let line_inc = line_delta - LINE_BASE as i64;
    if line_inc < 0 || line_inc >= LINE_RANGE as i64 {
        return None;
    }

    // Guard against overflow in the multiplication
    if op_advance > (u64::MAX - line_inc as u64) / LINE_RANGE as u64 {
        return None;
    }

    let adjusted = line_inc as u64 + LINE_RANGE as u64 * op_advance;
    let opcode = adjusted + OPCODE_BASE as u64;

    if opcode > 255 {
        None
    } else {
        Some(opcode as u8)
    }
}

// ===========================================================================
// LEB128 Encoding
// ===========================================================================

/// Encodes an unsigned 64-bit integer in ULEB128 (Unsigned Little-Endian Base 128)
/// format and appends the bytes to `data`.
///
/// ULEB128 uses a variable number of bytes where each byte contributes 7 bits
/// of value data. The high bit of each byte is a continuation flag: 1 means
/// more bytes follow, 0 means this is the final byte.
///
/// # Examples
///
/// - 0 → `[0x00]` (1 byte)
/// - 127 → `[0x7F]` (1 byte)
/// - 128 → `[0x80, 0x01]` (2 bytes)
/// - 624485 → `[0xE5, 0x8E, 0x26]` (3 bytes)
fn write_uleb128(data: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // continuation bit
        }
        data.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encodes a signed 64-bit integer in SLEB128 (Signed Little-Endian Base 128)
/// format and appends the bytes to `data`.
///
/// SLEB128 uses two's complement encoding. Each byte contributes 7 bits of
/// value data plus a continuation bit. The sign bit of the final byte's 7-bit
/// group extends the value.
///
/// # Examples
///
/// - 0 → `[0x00]`
/// - 2 → `[0x02]`
/// - -1 → `[0x7F]`
/// - -123456 → `[0xC0, 0xBB, 0x78]`
fn write_sleb128(data: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7F) as u8;
        // Arithmetic right shift preserves sign
        value >>= 7;
        // Termination condition:
        //   - If value is 0 and bit 6 of byte is clear (positive), done.
        //   - If value is -1 and bit 6 of byte is set (negative), done.
        let done = (value == 0 && (byte & 0x40) == 0)
            || (value == -1 && (byte & 0x40) != 0);
        if done {
            data.push(byte);
            break;
        } else {
            data.push(byte | 0x80);
        }
    }
}

// ===========================================================================
// Binary Writing Helpers
// ===========================================================================

/// Appends a little-endian 16-bit unsigned integer to the byte vector.
#[inline]
fn write_u16_le(data: &mut Vec<u8>, value: u16) {
    data.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian 32-bit unsigned integer to the byte vector.
#[inline]
fn write_u32_le(data: &mut Vec<u8>, value: u32) {
    data.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian 64-bit unsigned integer to the byte vector.
#[inline]
fn write_u64_le(data: &mut Vec<u8>, value: u64) {
    data.extend_from_slice(&value.to_le_bytes());
}

/// Patches a 4-byte region in `data` at the given offset with a little-endian
/// 32-bit value. Used for backpatching `unit_length` and `header_length` fields.
///
/// # Panics
///
/// Panics if `offset + 4 > data.len()`.
#[inline]
fn patch_u32_le(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset..offset + 4].copy_from_slice(&bytes);
}

// ===========================================================================
// Path Splitting Helper
// ===========================================================================

/// Splits a file path into a directory portion and a base name.
///
/// Uses the last `/` (or `\` on mixed-separator input) as the split point.
/// If no separator is found, returns an empty directory and the full path
/// as the base name.
///
/// # Arguments
///
/// * `path` — File path string.
///
/// # Returns
///
/// Tuple of `(directory, base_name)` string slices.
fn split_path(path: &str) -> (&str, &str) {
    // Find the last path separator (forward slash or backslash)
    if let Some(pos) = path.rfind('/') {
        (&path[..pos], &path[pos + 1..])
    } else if let Some(pos) = path.rfind('\\') {
        (&path[..pos], &path[pos + 1..])
    } else {
        ("", path)
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ULEB128 encoding --------------------------------------------------

    #[test]
    fn test_uleb128_zero() {
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_uleb128_small() {
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 127);
        assert_eq!(buf, vec![0x7F]);
    }

    #[test]
    fn test_uleb128_two_bytes() {
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn test_uleb128_large() {
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 624485);
        assert_eq!(buf, vec![0xE5, 0x8E, 0x26]);
    }

    #[test]
    fn test_uleb128_one() {
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 1);
        assert_eq!(buf, vec![0x01]);
    }

    #[test]
    fn test_uleb128_max_one_byte() {
        // 127 is the maximum value encodable in a single ULEB128 byte
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 0x7F);
        assert_eq!(buf, vec![0x7F]);
    }

    // -- SLEB128 encoding --------------------------------------------------

    #[test]
    fn test_sleb128_zero() {
        let mut buf = Vec::new();
        write_sleb128(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_sleb128_positive_small() {
        let mut buf = Vec::new();
        write_sleb128(&mut buf, 2);
        assert_eq!(buf, vec![0x02]);
    }

    #[test]
    fn test_sleb128_negative_one() {
        let mut buf = Vec::new();
        write_sleb128(&mut buf, -1);
        assert_eq!(buf, vec![0x7F]);
    }

    #[test]
    fn test_sleb128_negative_large() {
        let mut buf = Vec::new();
        write_sleb128(&mut buf, -123456);
        assert_eq!(buf, vec![0xC0, 0xBB, 0x78]);
    }

    #[test]
    fn test_sleb128_positive_63() {
        // 63 (0x3F) fits in one byte with sign bit clear
        let mut buf = Vec::new();
        write_sleb128(&mut buf, 63);
        assert_eq!(buf, vec![0x3F]);
    }

    #[test]
    fn test_sleb128_positive_64() {
        // 64 (0x40) needs two bytes because bit 6 is set (looks negative in 7-bit)
        let mut buf = Vec::new();
        write_sleb128(&mut buf, 64);
        assert_eq!(buf, vec![0xC0, 0x00]);
    }

    #[test]
    fn test_sleb128_negative_64() {
        // -64 (signed 7-bit: 0x40) fits in one byte
        let mut buf = Vec::new();
        write_sleb128(&mut buf, -64);
        assert_eq!(buf, vec![0x40]);
    }

    #[test]
    fn test_sleb128_negative_65() {
        // -65 needs two bytes
        let mut buf = Vec::new();
        write_sleb128(&mut buf, -65);
        assert_eq!(buf, vec![0xBF, 0x7F]);
    }

    // -- Special opcode computation ----------------------------------------

    #[test]
    fn test_special_opcode_line_plus_one_no_addr() {
        // line_delta=1, op_advance=0
        // line_inc = 1 - (-5) = 6
        // adjusted = 6 + 14*0 = 6
        // opcode = 6 + 13 = 19
        assert_eq!(compute_special_opcode(1, 0), Some(19));
    }

    #[test]
    fn test_special_opcode_line_plus_one_addr_one() {
        // line_delta=1, op_advance=1
        // line_inc = 6, adjusted = 6 + 14 = 20, opcode = 33
        assert_eq!(compute_special_opcode(1, 1), Some(33));
    }

    #[test]
    fn test_special_opcode_min_line() {
        // line_delta=-5 (minimum), op_advance=0
        // line_inc = 0, adjusted = 0, opcode = 13 (= OPCODE_BASE, smallest special)
        assert_eq!(compute_special_opcode(-5, 0), Some(13));
    }

    #[test]
    fn test_special_opcode_max_line() {
        // line_delta=8 (LINE_BASE + LINE_RANGE - 1 = -5+14-1 = 8), op_advance=0
        // line_inc = 13, adjusted = 13, opcode = 26
        assert_eq!(compute_special_opcode(8, 0), Some(26));
    }

    #[test]
    fn test_special_opcode_line_out_of_range_positive() {
        // line_delta=9 → line_inc = 14 ≥ LINE_RANGE, not encodable
        assert_eq!(compute_special_opcode(9, 0), None);
    }

    #[test]
    fn test_special_opcode_line_out_of_range_negative() {
        // line_delta=-6 → line_inc = -1 < 0, not encodable
        assert_eq!(compute_special_opcode(-6, 0), None);
    }

    #[test]
    fn test_special_opcode_addr_too_large() {
        // line_delta=1, op_advance=18
        // adjusted = 6 + 14*18 = 258, opcode = 258+13 = 271 > 255
        assert_eq!(compute_special_opcode(1, 18), None);
    }

    #[test]
    fn test_special_opcode_max_special_255() {
        // Find inputs that produce opcode = 255:
        // 255 = adjusted + 13 → adjusted = 242
        // line_delta=-5 (line_inc=0): op_advance = 242/14 = 17, remainder = 4
        // Not exact. Let's try: 242 = 0 + 14*17 = 238. nope.
        // 242 = line_inc + 14*op_advance
        // line_inc=4 (line_delta=-1), op_advance=17: 4 + 14*17 = 4+238=242. opcode=255
        assert_eq!(compute_special_opcode(-1, 17), Some(255));
    }

    #[test]
    fn test_special_opcode_barely_fits() {
        // line_delta=-5, op_advance=17
        // line_inc=0, adjusted = 14*17 = 238, opcode = 251 ≤ 255
        assert_eq!(compute_special_opcode(-5, 17), Some(251));
    }

    #[test]
    fn test_special_opcode_zero_zero() {
        // line_delta=0, op_advance=0
        // line_inc = 5, adjusted = 5, opcode = 18
        assert_eq!(compute_special_opcode(0, 0), Some(18));
    }

    // -- Binary writing helpers --------------------------------------------

    #[test]
    fn test_write_u16_le() {
        let mut buf = Vec::new();
        write_u16_le(&mut buf, 0x0102);
        assert_eq!(buf, vec![0x02, 0x01]);
    }

    #[test]
    fn test_write_u32_le() {
        let mut buf = Vec::new();
        write_u32_le(&mut buf, 0x01020304);
        assert_eq!(buf, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_write_u64_le() {
        let mut buf = Vec::new();
        write_u64_le(&mut buf, 0x0102030405060708);
        assert_eq!(buf, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_patch_u32_le() {
        let mut buf = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xAA];
        patch_u32_le(&mut buf, 0, 0x12345678);
        assert_eq!(buf, vec![0x78, 0x56, 0x34, 0x12, 0xAA]);
    }

    #[test]
    fn test_patch_u32_le_mid_buffer() {
        let mut buf = vec![0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xBB];
        patch_u32_le(&mut buf, 2, 0xDEADBEEF);
        assert_eq!(buf, vec![0x00, 0x00, 0xEF, 0xBE, 0xAD, 0xDE, 0xBB]);
    }

    // -- Path splitting ----------------------------------------------------

    #[test]
    fn test_split_path_unix() {
        assert_eq!(split_path("/home/user/src/main.c"), ("/home/user/src", "main.c"));
    }

    #[test]
    fn test_split_path_no_dir() {
        assert_eq!(split_path("main.c"), ("", "main.c"));
    }

    #[test]
    fn test_split_path_trailing_slash() {
        assert_eq!(split_path("/dir/"), ("/dir", ""));
    }

    // -- LineNumberProgramBuilder basic API ---------------------------------

    #[test]
    fn test_builder_new() {
        let builder = LineNumberProgramBuilder::new(Target::X86_64);
        assert!(builder.data.is_empty());
        assert!(builder.directories.is_empty());
        assert!(builder.files.is_empty());
        assert_eq!(builder.section_offset(), 0);
    }

    #[test]
    fn test_add_directory_returns_1_based() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        let idx1 = builder.add_directory("/usr/include");
        let idx2 = builder.add_directory("/home/user/src");
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    #[test]
    fn test_add_directory_deduplicates() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        let idx1 = builder.add_directory("/usr/include");
        let idx2 = builder.add_directory("/usr/include");
        assert_eq!(idx1, idx2);
        assert_eq!(builder.directories.len(), 1);
    }

    #[test]
    fn test_add_file_returns_1_based() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        let dir = builder.add_directory("/src");
        let f1 = builder.add_file("main.c", dir);
        let f2 = builder.add_file("util.c", dir);
        assert_eq!(f1, 1);
        assert_eq!(f2, 2);
    }

    // -- minimum_instruction_length and address_size -----------------------

    #[test]
    fn test_min_instr_len_x86() {
        let builder = LineNumberProgramBuilder::new(Target::X86_64);
        assert_eq!(builder.minimum_instruction_length(), 1);
    }

    #[test]
    fn test_min_instr_len_i686() {
        let builder = LineNumberProgramBuilder::new(Target::I686);
        assert_eq!(builder.minimum_instruction_length(), 1);
    }

    #[test]
    fn test_min_instr_len_aarch64() {
        let builder = LineNumberProgramBuilder::new(Target::AArch64);
        assert_eq!(builder.minimum_instruction_length(), 4);
    }

    #[test]
    fn test_min_instr_len_riscv() {
        let builder = LineNumberProgramBuilder::new(Target::RiscV64);
        assert_eq!(builder.minimum_instruction_length(), 4);
    }

    #[test]
    fn test_address_size_x86_64() {
        let builder = LineNumberProgramBuilder::new(Target::X86_64);
        assert_eq!(builder.address_size(), 8);
    }

    #[test]
    fn test_address_size_i686() {
        let builder = LineNumberProgramBuilder::new(Target::I686);
        assert_eq!(builder.address_size(), 4);
    }

    #[test]
    fn test_address_size_aarch64() {
        let builder = LineNumberProgramBuilder::new(Target::AArch64);
        assert_eq!(builder.address_size(), 8);
    }

    #[test]
    fn test_address_size_riscv64() {
        let builder = LineNumberProgramBuilder::new(Target::RiscV64);
        assert_eq!(builder.address_size(), 8);
    }

    // -- Header emission and field verification ----------------------------

    #[test]
    fn test_emit_header_version() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.add_file("test.c", 0);
        builder.emit_header();
        // version is at offset 4..5 (after unit_length)
        let version = u16::from_le_bytes([builder.data[4], builder.data[5]]);
        assert_eq!(version, 4);
    }

    #[test]
    fn test_emit_header_fixed_fields_x86() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.add_file("hello.c", 0);
        builder.emit_header();
        let d = &builder.data;
        // Fixed fields start at offset 10 (after unit_length[4] + version[2] + header_length[4])
        assert_eq!(d[10], 1, "minimum_instruction_length");
        assert_eq!(d[11], 1, "maximum_operations_per_instruction");
        assert_eq!(d[12], 1, "default_is_stmt");
        assert_eq!(d[13] as i8, -5, "line_base");
        assert_eq!(d[14], 14, "line_range");
        assert_eq!(d[15], 13, "opcode_base");
    }

    #[test]
    fn test_emit_header_fixed_fields_aarch64() {
        let mut builder = LineNumberProgramBuilder::new(Target::AArch64);
        builder.add_file("test.c", 0);
        builder.emit_header();
        assert_eq!(builder.data[10], 4, "minimum_instruction_length for AArch64");
    }

    #[test]
    fn test_emit_header_opcode_lengths_array() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        // Opcode lengths are at offsets 16..27 (inclusive)
        let expected: [u8; 12] = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];
        for (i, &expected_len) in expected.iter().enumerate() {
            assert_eq!(builder.data[16 + i], expected_len,
                "opcode {} length mismatch", i + 1);
        }
    }

    // -- State machine updates ---------------------------------------------

    #[test]
    fn test_emit_set_address_updates_state() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.emit_set_address(0x401000);
        assert_eq!(builder.state.address, 0x401000);
    }

    #[test]
    fn test_emit_advance_pc_updates_state() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.state.address = 0x1000;
        builder.emit_advance_pc(16);
        assert_eq!(builder.state.address, 0x1010);
    }

    #[test]
    fn test_emit_advance_pc_aarch64() {
        // AArch64: min_instr_len=4, so delta=8 → op_advance=2
        let mut builder = LineNumberProgramBuilder::new(Target::AArch64);
        builder.emit_header();
        builder.state.address = 0x1000;
        builder.emit_advance_pc(8);
        assert_eq!(builder.state.address, 0x1008);
    }

    #[test]
    fn test_emit_advance_line_positive() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.state.line = 10;
        builder.emit_advance_line(5);
        assert_eq!(builder.state.line, 15);
    }

    #[test]
    fn test_emit_advance_line_negative() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.state.line = 10;
        builder.emit_advance_line(-3);
        assert_eq!(builder.state.line, 7);
    }

    #[test]
    fn test_emit_set_file_updates_state() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_set_file(3);
        assert_eq!(builder.state.file, 3);
    }

    #[test]
    fn test_emit_set_column_updates_state() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_set_column(42);
        assert_eq!(builder.state.column, 42);
    }

    #[test]
    fn test_emit_copy_resets_flags() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.state.basic_block = true;
        builder.state.prologue_end = true;
        builder.state.epilogue_begin = true;
        builder.state.discriminator = 5;
        builder.emit_copy();
        assert!(!builder.state.basic_block);
        assert!(!builder.state.prologue_end);
        assert!(!builder.state.epilogue_begin);
        assert_eq!(builder.state.discriminator, 0);
        // Other registers should be unchanged
        assert_eq!(builder.state.file, 1);
        assert_eq!(builder.state.line, 1);
    }

    #[test]
    fn test_emit_negate_stmt_toggles() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        assert!(builder.state.is_stmt); // default is true
        builder.emit_negate_stmt();
        assert!(!builder.state.is_stmt);
        builder.emit_negate_stmt();
        assert!(builder.state.is_stmt);
    }

    #[test]
    fn test_emit_end_sequence_resets_all() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.emit_set_address(0x2000);
        builder.emit_set_file(2);
        builder.emit_advance_line(50);
        builder.emit_set_column(10);
        builder.emit_end_sequence();
        // All registers should be back to initial state
        assert_eq!(builder.state.address, 0);
        assert_eq!(builder.state.file, 1);
        assert_eq!(builder.state.line, 1);
        assert_eq!(builder.state.column, 0);
        assert!(builder.state.is_stmt);
        assert!(!builder.state.basic_block);
        assert!(!builder.state.end_sequence);
    }

    // -- Special opcode / emit_line_advance --------------------------------

    #[test]
    fn test_emit_line_advance_special_opcode() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.emit_set_address(0x1000);
        let len_before = builder.data.len();
        // line_delta=1, address_delta=5
        // For x86 (min_len=1): op_advance=5
        // line_inc=1-(-5)=6, adjusted=6+14*5=76, opcode=76+13=89
        builder.emit_line_advance(5, 1);
        // Should be exactly 1 byte (special opcode)
        assert_eq!(builder.data.len() - len_before, 1);
        assert_eq!(builder.data[len_before], 89);
        assert_eq!(builder.state.address, 0x1005);
        assert_eq!(builder.state.line, 2); // was 1, advanced by 1
    }

    #[test]
    fn test_emit_line_advance_fallback() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.emit_set_address(0x1000);
        let len_before = builder.data.len();
        // line_delta=100 → far outside special opcode range
        builder.emit_line_advance(10, 100);
        // Fallback: advance_pc + advance_line + copy → more than 1 byte
        assert!(builder.data.len() - len_before > 1);
        assert_eq!(builder.state.address, 0x100A);
        assert_eq!(builder.state.line, 101); // was 1, advanced by 100
    }

    #[test]
    fn test_emit_line_advance_zero_delta() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.emit_set_address(0x1000);
        let len_before = builder.data.len();
        // line_delta=0, address_delta=0 → special opcode 18
        builder.emit_line_advance(0, 0);
        assert_eq!(builder.data.len() - len_before, 1);
        assert_eq!(builder.data[len_before], 18);
    }

    #[test]
    fn test_emit_line_advance_negative_line() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.emit_header();
        builder.state.line = 10;
        builder.state.address = 0x1000;
        // line_delta=-3, address_delta=2
        // line_inc = -3 - (-5) = 2, adjusted = 2 + 14*2 = 30, opcode = 43
        builder.emit_line_advance(2, -3);
        assert_eq!(builder.state.line, 7);
        assert_eq!(builder.state.address, 0x1002);
    }

    // -- Full end-to-end section construction -------------------------------

    #[test]
    fn test_full_section_x86_64() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        let dir_idx = builder.add_directory("/home/user/src");
        let file_idx = builder.add_file("main.c", dir_idx);
        builder.emit_header();

        builder.emit_set_address(0x401000);
        builder.emit_set_file(file_idx);
        builder.emit_line_advance(0, 0);       // line 1 at start
        builder.emit_line_advance(5, 1);       // +5 bytes, +1 line
        builder.emit_line_advance(3, 1);       // +3 bytes, +1 line
        builder.emit_end_sequence();

        let bytes = builder.finish();
        assert!(!bytes.is_empty());

        // Verify unit_length matches
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);

        // Verify version
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 4);
    }

    #[test]
    fn test_full_section_i686() {
        let mut builder = LineNumberProgramBuilder::new(Target::I686);
        builder.add_file("test.c", 0);
        builder.emit_header();

        builder.emit_set_address(0x08048000);
        builder.emit_line_advance(4, 1);
        builder.emit_end_sequence();

        let bytes = builder.finish();

        // i686 uses 4-byte addresses in DW_LNE_set_address
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_full_section_aarch64() {
        let mut builder = LineNumberProgramBuilder::new(Target::AArch64);
        builder.add_file("test.c", 0);
        builder.emit_header();

        builder.emit_set_address(0x400000);
        // AArch64: min_instr_len=4, so address deltas should be multiples of 4
        builder.emit_line_advance(4, 1);
        builder.emit_line_advance(8, 2);
        builder.emit_end_sequence();

        let bytes = builder.finish();
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_full_section_riscv64() {
        let mut builder = LineNumberProgramBuilder::new(Target::RiscV64);
        builder.add_file("test.c", 0);
        builder.emit_header();

        builder.emit_set_address(0x80000000);
        builder.emit_line_advance(4, 1);
        builder.emit_end_sequence();

        let bytes = builder.finish();
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_multiple_sequences() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        builder.add_file("multi.c", 0);
        builder.emit_header();

        // First sequence (function 1)
        builder.emit_set_address(0x1000);
        builder.emit_line_advance(5, 1);
        builder.emit_end_sequence();

        // Second sequence (function 2)
        builder.emit_set_address(0x2000);
        builder.emit_line_advance(10, 3);
        builder.emit_end_sequence();

        let bytes = builder.finish();
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_header_length_consistency() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        let dir = builder.add_directory("/long/directory/path/for/testing");
        builder.add_file("file1.c", dir);
        builder.add_file("file2.c", dir);
        builder.add_file("file3_with_a_very_long_name.c", 0);
        builder.emit_header();

        // header_length is at offset 6..9
        let header_length = u32::from_le_bytes([
            builder.data[6], builder.data[7],
            builder.data[8], builder.data[9],
        ]);
        // The header content starts at offset 10 (after header_length field)
        // and ends at the first program opcode.
        // header_length should equal: data.len() - 10 (when no opcodes yet)
        assert_eq!(header_length as usize, builder.data.len() - 10);
    }

    #[test]
    fn test_empty_directory_and_file_tables() {
        let mut builder = LineNumberProgramBuilder::new(Target::X86_64);
        // No directories or files added
        builder.emit_header();
        builder.emit_set_address(0x1000);
        builder.emit_end_sequence();
        let bytes = builder.finish();
        // Should still produce valid output
        let unit_length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(unit_length as usize, bytes.len() - 4);
    }

    #[test]
    fn test_section_offset_is_zero() {
        let builder = LineNumberProgramBuilder::new(Target::X86_64);
        assert_eq!(builder.section_offset(), 0);
    }

    // -- FileEntry struct --------------------------------------------------

    #[test]
    fn test_file_entry_clone() {
        let entry = FileEntry {
            name: "test.c".to_string(),
            dir_index: 1,
            mod_time: 12345,
            file_size: 9876,
        };
        let clone = entry.clone();
        assert_eq!(clone.name, "test.c");
        assert_eq!(clone.dir_index, 1);
        assert_eq!(clone.mod_time, 12345);
        assert_eq!(clone.file_size, 9876);
    }
}
