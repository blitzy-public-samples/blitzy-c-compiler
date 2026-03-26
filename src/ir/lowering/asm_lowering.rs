//! # Inline Assembly Lowering
//!
//! Lowers AST inline assembly statements (`asm`/`__asm__`) into IR `InlineAsm` instructions.
//!
//! Handles:
//! - Template string parsing and operand substitution placeholder tracking
//! - Output/input operand constraint validation per target architecture
//! - Operand binding: connecting C-level variables to IR values for the asm
//! - Clobber set collection and propagation to register allocator
//! - `asm goto` target block wiring (connecting asm jump targets to IR basic blocks)
//! - `asm volatile` side-effect annotation
//!
//! ## Architecture-Specific Constraint Codes
//!
//! | Constraint | x86-64/i686     | AArch64        | RISC-V 64      |
//! |-----------|-----------------|----------------|----------------|
//! | `r`       | General GPR     | General GPR    | General GPR    |
//! | `m`       | Memory operand  | Memory operand | Memory operand |
//! | `i`/`n`   | Immediate       | Immediate      | Immediate      |
//! | `a`       | RAX/EAX         | N/A            | N/A            |
//! | `b`       | RBX/EBX         | N/A            | N/A            |
//! | `c`       | RCX/ECX         | N/A            | N/A            |
//! | `d`       | RDX/EDX         | N/A            | N/A            |
//! | `S`       | RSI/ESI         | N/A            | N/A            |
//! | `D`       | RDI/EDI         | N/A            | N/A            |
//! | `A`       | RDX:RAX pair    | N/A            | N/A            |
//! | `f`       | N/A             | N/A            | FP register    |
//! | `w`       | N/A             | SIMD register  | N/A            |
//! | `x`       | N/A             | Any GPR        | N/A            |
//!
//! ## Dependencies
//! - `crate::ir::*` — IR types, instructions, builder, function, basic_block
//! - `crate::frontend::parser::ast` — AST inline asm statement nodes
//! - `crate::common::target` — Architecture-specific constraint validation
//! - `crate::common::diagnostics` — Error reporting for invalid constraints
//! - `crate::common::fx_hash` — Performance-optimized hash maps

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::common::types::CType;
use crate::frontend::parser::ast;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// AsmConstraint — Parsed inline asm operand constraint
// ===========================================================================

/// Parsed representation of a single inline asm operand constraint.
///
/// GCC inline asm constraints encode how an operand is passed between the
/// C code and the assembly template. Constraints specify whether the operand
/// is input, output, or read-write, and what kind of register or memory
/// location the operand occupies.
///
/// # Modifier Prefixes
///
/// - `=` — output only: the operand is written by the asm, not read
/// - `+` — read-write: the operand is both read and written
/// - `&` — early clobber: the output is written before all inputs are consumed
///
/// # Examples
///
/// - `"=r"` → output, register, not read-write, not early-clobber
/// - `"+m"` → read-write, memory
/// - `"=&r"` → output, register, early-clobber
/// - `"i"` → input, immediate integer
/// - `"r"` → input, register
#[derive(Debug, Clone)]
pub struct AsmConstraint {
    /// Original constraint string (e.g., `"=r"`, `"+m"`, `"i"`).
    pub raw: String,
    /// Is this an output constraint? (`=` or `+` prefix present)
    pub is_output: bool,
    /// Is this a read-write constraint? (`+` prefix)
    pub is_read_write: bool,
    /// Is this an early-clobber constraint? (`&` modifier)
    pub is_early_clobber: bool,
    /// The constraint letter(s) after stripping modifiers (e.g., `"r"`, `"m"`, `"i"`).
    pub constraint_code: String,
}

// ===========================================================================
// AsmLoweringContext — Context for lowering a single asm statement
// ===========================================================================

/// Context for lowering a single inline assembly statement into IR.
///
/// Aggregates the mutable and immutable references needed throughout the
/// lowering of one `asm`/`__asm__` statement. Created by the statement
/// lowering driver (`stmt_lowering.rs`) and passed to the asm lowering
/// functions.
///
/// # Alloca-First Pattern
///
/// All local variables referenced by asm operands should already exist as
/// `alloca` instructions in the function's entry block, created by the
/// lowering driver. This module loads from and stores to those allocas as
/// needed for operand binding.
pub struct AsmLoweringContext<'a> {
    /// IR builder for creating load, store, and inline asm instructions.
    pub builder: &'a mut IrBuilder,
    /// Current function being lowered — provides access to basic blocks
    /// for asm goto target wiring.
    pub function: &'a mut IrFunction,
    /// Target architecture — used for constraint and clobber validation.
    pub target: &'a Target,
    /// Diagnostic engine for reporting errors (invalid constraints,
    /// unknown clobbers, undefined goto labels).
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Map from variable names to their alloca `Value` handles. Populated
    /// by the lowering driver from function-scope declarations.
    pub local_vars: &'a FxHashMap<String, Value>,
}

// ===========================================================================
// Constraint Parsing
// ===========================================================================

/// Parse an operand constraint string into its structured components.
///
/// Handles constraint modifier prefixes (`=`, `+`, `&`) followed by one
/// or more constraint code letters (`r`, `m`, `i`, `n`, `g`, `X`, and
/// architecture-specific codes).
///
/// # Examples
///
/// ```text
/// "=r"  → AsmConstraint { is_output: true,  is_read_write: false, constraint_code: "r" }
/// "+m"  → AsmConstraint { is_output: true,  is_read_write: true,  constraint_code: "m" }
/// "=&r" → AsmConstraint { is_output: true,  is_early_clobber: true, constraint_code: "r" }
/// "i"   → AsmConstraint { is_output: false, constraint_code: "i" }
/// "r"   → AsmConstraint { is_output: false, constraint_code: "r" }
/// ```
///
/// # Errors
///
/// Returns `Err` with a descriptive message if the constraint string is
/// empty or contains no constraint code after stripping modifiers.
fn parse_constraint(constraint_str: &str) -> Result<AsmConstraint, String> {
    if constraint_str.is_empty() {
        return Err("empty constraint string".to_string());
    }

    let raw = constraint_str.to_string();
    let mut is_output = false;
    let mut is_read_write = false;
    let mut is_early_clobber = false;

    let mut chars = constraint_str.chars().peekable();

    // Parse modifier prefixes: =, +, &
    // '=' and '+' must appear first; '&' may follow.
    match chars.peek() {
        Some('=') => {
            is_output = true;
            chars.next();
        }
        Some('+') => {
            is_output = true;
            is_read_write = true;
            chars.next();
        }
        _ => {}
    }

    // Check for early-clobber modifier '&' (may follow '=' or '+')
    if chars.peek() == Some(&'&') {
        is_early_clobber = true;
        chars.next();
    }

    // The remaining characters form the constraint code(s).
    let constraint_code: String = chars.collect();

    if constraint_code.is_empty() {
        return Err(format!(
            "constraint '{}' has no constraint code after modifiers",
            constraint_str
        ));
    }

    Ok(AsmConstraint {
        raw,
        is_output,
        is_read_write,
        is_early_clobber,
        constraint_code,
    })
}

// ===========================================================================
// Architecture-Specific Constraint Validation
// ===========================================================================

/// Validate that a parsed constraint is valid for the given target
/// architecture and operand type.
///
/// Each architecture supports a different set of constraint codes:
/// - **Universal**: `r` (register), `m` (memory), `i`/`n` (immediate),
///   `g` (general), `X` (any), `0`-`9` (tied to another operand)
/// - **x86-64/i686**: `a` (eax/rax), `b` (ebx/rbx), `c` (ecx/rcx),
///   `d` (edx/rdx), `S` (esi/rsi), `D` (edi/rdi), `A` (edx:eax / rdx:rax)
/// - **AArch64**: `w` (SIMD/FP register), `x` (any GPR)
/// - **RISC-V 64**: `f` (floating-point register)
///
/// For immediate constraints (`i`, `n`), the operand type should be an
/// integer or pointer constant. For register constraints, integer/pointer/
/// float types are generally acceptable.
///
/// # Returns
///
/// `true` if the constraint is valid; `false` otherwise (with a diagnostic
/// error emitted).
fn validate_constraint(
    constraint: &AsmConstraint,
    target: &Target,
    operand_type: Option<&CType>,
    diagnostics: &mut DiagnosticEngine,
    span: Span,
) -> bool {
    // Universal constraint codes valid on all architectures.
    let universal_codes = ['r', 'm', 'i', 'n', 'g', 'X', 'p'];

    for ch in constraint.constraint_code.chars() {
        // Digit constraints (operand-tied): '0'-'9'
        if ch.is_ascii_digit() {
            continue;
        }

        // Check universal codes
        if universal_codes.contains(&ch) {
            // For immediate constraints, verify operand type compatibility
            if (ch == 'i' || ch == 'n') && !constraint.is_output {
                if let Some(ty) = operand_type {
                    if !is_immediate_compatible_type(ty) {
                        diagnostics.emit(Diagnostic::warning(
                            span,
                            format!(
                                "immediate constraint '{}' used with non-integer/pointer type",
                                ch
                            ),
                        ));
                    }
                }
            }
            continue;
        }

        // Architecture-specific validation
        let valid = match target {
            Target::X86_64 | Target::I686 => {
                matches!(
                    ch,
                    'a' | 'b'
                        | 'c'
                        | 'd'
                        | 'S'
                        | 'D'
                        | 'A'
                        | 'q'
                        | 'Q'
                        | 'R'
                        | 'l'
                        | 'f'
                        | 't'
                        | 'u'
                        | 'y'
                        | 'x'
                        | 'Y'
                        | 'e'
                        | 'Z'
                )
            }
            Target::AArch64 => {
                matches!(ch, 'w' | 'x' | 'z' | 'Q' | 'U')
            }
            Target::RiscV64 => {
                matches!(ch, 'f' | 'J' | 'K' | 'I' | 'A' | 'S' | 'v' | 'c')
            }
        };

        if !valid {
            diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "invalid asm constraint code '{}' for target '{}'",
                    ch, target
                ),
            ));
            return false;
        }
    }

    true
}

/// Check whether a C type is compatible with immediate constraints (`i`, `n`).
///
/// Immediate constraints require a compile-time constant value. This is
/// typically an integer, enum, or address constant (pointer to global).
fn is_immediate_compatible_type(ty: &CType) -> bool {
    matches!(
        ty,
        CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong
            | CType::Pointer(_, _)
            | CType::Enum { .. }
    )
}

// ===========================================================================
// Main Lowering Entry Point
// ===========================================================================

/// Lower an AST inline assembly statement into an IR `InlineAsm` instruction.
///
/// This is called from `stmt_lowering.rs` when an `asm` / `__asm__`
/// statement is encountered. The caller is responsible for:
///
/// 1. **Lowering output operand expressions** to alloca pointer `Value`s
///    (since the output expressions are lvalues whose address is needed
///    for post-asm stores).
/// 2. **Lowering input operand expressions** to rvalue `Value`s (the
///    current value of the input, loaded from its alloca or computed).
/// 3. **Resolving `Symbol` handles** from `asm_stmt.goto_labels` and
///    `AsmOperand.symbolic_name` to `String` names (since this module
///    does not depend on the string interner).
///
/// # Parameters
///
/// - `asm_stmt`: The AST representation of the inline assembly statement.
/// - `builder`: IR builder for creating load, store, and InlineAsm instructions.
/// - `function`: Current IR function (for asm goto block access).
/// - `target`: Target architecture for constraint/clobber validation.
/// - `diagnostics`: Diagnostic engine for error reporting.
/// - `local_vars`: Map from variable names to alloca `Value`s.
/// - `label_blocks`: Map from C label names to `BlockId`s (for asm goto).
/// - `output_ptrs`: Pre-lowered alloca pointers for output operands (one per output).
/// - `input_vals`: Pre-lowered rvalues for input operands (one per input).
/// - `goto_label_names`: Pre-resolved goto label name strings.
/// - `named_operand_strs`: Resolved symbolic names for ALL operands
///   (outputs first, then inputs). `None` for positional-only operands.
/// - `span`: Source location for the entire asm statement.
///
/// # Returns
///
/// `Some(Value)` — the SSA result of the InlineAsm instruction (used for
/// single-output asm), or `None` if errors prevented instruction creation.
#[allow(clippy::too_many_arguments)]
pub fn lower_inline_asm(
    asm_stmt: &ast::AsmStatement,
    builder: &mut IrBuilder,
    function: &mut IrFunction,
    target: &Target,
    diagnostics: &mut DiagnosticEngine,
    _local_vars: &FxHashMap<String, Value>,
    label_blocks: &FxHashMap<String, BlockId>,
    output_ptrs: &[Value],
    input_vals: &[Value],
    goto_label_names: &[String],
    named_operand_strs: &[Option<String>],
    span: Span,
) -> Option<Value> {
    // ---- 1. Parse the template string from raw bytes --------------------
    let template_raw = String::from_utf8_lossy(&asm_stmt.template).to_string();

    // ---- 2. Parse and validate output constraints -----------------------
    let mut output_constraints: Vec<AsmConstraint> = Vec::with_capacity(asm_stmt.outputs.len());
    let mut all_valid = true;

    for (idx, output) in asm_stmt.outputs.iter().enumerate() {
        let constraint_str = String::from_utf8_lossy(&output.constraint).to_string();
        match parse_constraint(&constraint_str) {
            Ok(c) => {
                // Force output flag — output operands MUST have = or +
                if !c.is_output {
                    diagnostics.emit(Diagnostic::error(
                        output.span,
                        format!(
                            "output operand {} constraint '{}' missing '=' or '+' modifier",
                            idx, constraint_str
                        ),
                    ));
                    all_valid = false;
                }
                // Validate against target architecture
                if !validate_constraint(&c, target, None, diagnostics, output.span) {
                    all_valid = false;
                }
                output_constraints.push(c);
            }
            Err(msg) => {
                diagnostics.emit(Diagnostic::error(
                    output.span,
                    format!("invalid output constraint for operand {}: {}", idx, msg),
                ));
                all_valid = false;
            }
        }
    }

    // ---- 3. Parse and validate input constraints ------------------------
    let mut input_constraints: Vec<AsmConstraint> = Vec::with_capacity(asm_stmt.inputs.len());

    for (idx, input) in asm_stmt.inputs.iter().enumerate() {
        let constraint_str = String::from_utf8_lossy(&input.constraint).to_string();
        match parse_constraint(&constraint_str) {
            Ok(c) => {
                // Input constraints must NOT have = or + modifiers
                if c.is_output {
                    diagnostics.emit(Diagnostic::error(
                        input.span,
                        format!(
                            "input operand {} constraint '{}' has output modifier '{}' — \
                             use an output operand instead",
                            idx,
                            constraint_str,
                            if c.is_read_write { "+" } else { "=" }
                        ),
                    ));
                    all_valid = false;
                }
                if !validate_constraint(&c, target, None, diagnostics, input.span) {
                    all_valid = false;
                }
                input_constraints.push(c);
            }
            Err(msg) => {
                diagnostics.emit(Diagnostic::error(
                    input.span,
                    format!("invalid input constraint for operand {}: {}", idx, msg),
                ));
                all_valid = false;
            }
        }
    }

    // ---- 4. Validate operand count consistency -------------------------
    if output_ptrs.len() != asm_stmt.outputs.len() {
        diagnostics.emit(Diagnostic::error(
            span,
            format!(
                "asm lowering: expected {} output values but got {}",
                asm_stmt.outputs.len(),
                output_ptrs.len()
            ),
        ));
        all_valid = false;
    }
    if input_vals.len() != asm_stmt.inputs.len() {
        diagnostics.emit(Diagnostic::error(
            span,
            format!(
                "asm lowering: expected {} input values but got {}",
                asm_stmt.inputs.len(),
                input_vals.len()
            ),
        ));
        all_valid = false;
    }

    // ---- 5. Validate and collect clobber list ---------------------------
    let clobbers = process_clobbers(
        &asm_stmt
            .clobbers
            .iter()
            .map(|c| String::from_utf8_lossy(&c.register).to_string())
            .collect::<Vec<_>>(),
        target,
        diagnostics,
        span,
    );

    // ---- 6. Wire asm goto targets to IR basic blocks --------------------
    let goto_targets = if asm_stmt.is_goto {
        wire_asm_goto_targets(goto_label_names, label_blocks, diagnostics, span)
    } else {
        Vec::new()
    };

    // If any constraint or clobber validation failed, bail out before
    // creating instructions to avoid emitting invalid IR.
    if diagnostics.has_errors() && !all_valid {
        return None;
    }

    // ---- 7. Build named operand map and resolve template ----------------
    let named_map = build_named_operand_map_from_strs(named_operand_strs, asm_stmt.outputs.len());
    let processed_template = analyze_template(&template_raw, &named_map);

    // ---- 8. Handle read-write ('+') outputs — load current value --------
    // Read-write outputs need their current value loaded as an implicit
    // input. We collect these additional input values here.
    let mut all_operand_values: Vec<Value> = Vec::new();
    let mut read_write_input_values: Vec<Value> = Vec::new();

    // Build a map from alloca Value → element IrType so that
    // read-write loads use the correct operand size (e.g. I32 for
    // `int` instead of I64 which would read garbage past the alloca).
    let mut alloca_elem_types: FxHashMap<Value, IrType> = FxHashMap::default();
    for blk in function.blocks.iter() {
        for inst in blk.instructions() {
            if let Instruction::Alloca { result, ty, .. } = inst {
                alloca_elem_types.insert(*result, ty.clone());
            }
        }
    }

    // Process output operands: they become operands in the InlineAsm instruction
    for (idx, constraint) in output_constraints.iter().enumerate() {
        if idx < output_ptrs.len() {
            let ptr = output_ptrs[idx];
            if constraint.is_read_write {
                // Read-write: load current value as an implicit input.
                // Use the alloca's element type for the load so we read
                // exactly the right number of bytes (not pointer-width).
                let ir_type = alloca_elem_types
                    .get(&ptr)
                    .cloned()
                    .unwrap_or_else(|| infer_ir_type_for_constraint(constraint, target));
                let (loaded_val, load_inst) = builder.build_load(ptr, ir_type, span);
                // Insert the load into the current block
                if let Some(block_id) = builder.get_insert_block() {
                    let block_idx = block_id.index();
                    if block_idx < function.blocks.len() {
                        function.blocks[block_idx].push_instruction(load_inst);
                    }
                }
                read_write_input_values.push(loaded_val);
            }
            all_operand_values.push(ptr);
        }
    }

    // Process input operands: add their values to the operand list
    for (idx, _constraint) in input_constraints.iter().enumerate() {
        if idx < input_vals.len() {
            all_operand_values.push(input_vals[idx]);
        }
    }

    // Append read-write implicit input values
    all_operand_values.extend_from_slice(&read_write_input_values);

    // ---- 9. Build the concatenated constraint string --------------------
    let constraints_string = build_constraints_string(&output_constraints, &input_constraints);

    // ---- 10. Determine side-effect flags --------------------------------
    let has_memory_clobber = clobbers.iter().any(|c| c == "memory");
    let has_side_effects = asm_stmt.is_volatile || has_memory_clobber || asm_stmt.is_goto;
    let is_volatile = asm_stmt.is_volatile;

    // ---- 11. Create the InlineAsm instruction ---------------------------
    let (result_val, mut asm_inst) = builder.build_inline_asm(
        processed_template,
        constraints_string,
        all_operand_values,
        clobbers,
        has_side_effects,
        span,
    );

    // Patch the instruction with goto targets and volatile flag if the
    // builder's default doesn't match what we need.
    patch_inline_asm_instruction(&mut asm_inst, &goto_targets, is_volatile);

    // Insert the InlineAsm instruction into the current block
    if let Some(block_id) = builder.get_insert_block() {
        let block_idx = block_id.index();
        if block_idx < function.blocks.len() {
            function.blocks[block_idx].push_instruction(asm_inst);
        }
    }

    // ---- 12. Create post-asm stores for output operands ------------------
    // For each output, store the asm result back to the output's alloca.
    // With a single-result InlineAsm, all outputs share the same result value.
    // The backend is responsible for extracting individual output values
    // from the constraint specification.
    // For memory constraints ("=m", "+m"), the value is already in memory
    // — the asm operates on the memory location directly, so no post-store
    // from the asm result register is needed.
    for (idx, constraint) in output_constraints.iter().enumerate() {
        if idx < output_ptrs.len() && constraint.is_output {
            let is_memory_constraint = constraint.constraint_code.contains('m');
            if !is_memory_constraint {
                let store_inst = builder.build_store(result_val, output_ptrs[idx], span);
                if let Some(block_id) = builder.get_insert_block() {
                    let block_idx = block_id.index();
                    if block_idx < function.blocks.len() {
                        function.blocks[block_idx].push_instruction(store_inst);
                    }
                }
            }
        }
    }

    // ---- 13. Add asm goto target blocks as successors for CFG -----------
    if !goto_targets.is_empty() {
        if let Some(block_id) = builder.get_insert_block() {
            let current_block_idx = block_id.index();
            if current_block_idx < function.blocks.len() {
                for target_block in &goto_targets {
                    let target_idx = target_block.index();
                    // Add successor edge from current block to goto target
                    if !function.blocks[current_block_idx]
                        .successors
                        .contains(&target_idx)
                    {
                        function.blocks[current_block_idx]
                            .successors
                            .push(target_idx);
                    }
                    // Add predecessor edge from goto target back to current block
                    if target_idx < function.blocks.len()
                        && !function.blocks[target_idx]
                            .predecessors
                            .contains(&current_block_idx)
                    {
                        function.blocks[target_idx]
                            .predecessors
                            .push(current_block_idx);
                    }
                }
            }
        }
    }

    Some(result_val)
}

// ===========================================================================
// Clobber List Processing
// ===========================================================================

/// Process and validate the clobber list from the asm statement.
///
/// Validates that each clobber name is either a special keyword (`"memory"`,
/// `"cc"`) or a valid register name for the target architecture. Invalid
/// clobber names are diagnosed but not removed — the list is returned with
/// all entries to avoid losing information.
///
/// # Special Clobbers
///
/// - `"memory"` — acts as a memory barrier; the compiler must assume all
///   memory is modified by the asm. The register allocator does not need
///   to reserve any register, but memory loads/stores may not be moved
///   across the asm statement.
/// - `"cc"` — the condition codes / flags register is clobbered. On x86
///   this is EFLAGS; on ARM it is NZCV.
fn process_clobbers(
    clobbers: &[String],
    target: &Target,
    diagnostics: &mut DiagnosticEngine,
    span: Span,
) -> Vec<String> {
    let mut validated_clobbers: Vec<String> = Vec::with_capacity(clobbers.len());

    for clobber in clobbers {
        let name = clobber.trim().to_string();
        if name.is_empty() {
            continue;
        }

        // Special clobbers accepted on all architectures
        if name == "memory" || name == "cc" {
            validated_clobbers.push(name);
            continue;
        }

        // Architecture-specific register name validation
        if validate_clobber_register(&name, target) {
            validated_clobbers.push(name);
        } else {
            diagnostics.emit(Diagnostic::warning(
                span,
                format!(
                    "unknown clobber register '{}' for target '{}'",
                    name, target
                ),
            ));
            // Still include it — the assembler may recognize it
            validated_clobbers.push(name);
        }
    }

    validated_clobbers
}

/// Validate that a register name is valid for the given target architecture.
///
/// This function checks against the canonical register names used by GCC
/// inline asm for each supported architecture. Both full names (e.g., `"rax"`)
/// and GCC shorthand (e.g., `"eax"`, `"ax"`, `"al"`) are accepted for x86.
fn validate_clobber_register(name: &str, target: &Target) -> bool {
    match target {
        Target::X86_64 => is_valid_x86_64_register(name),
        Target::I686 => is_valid_i686_register(name),
        Target::AArch64 => is_valid_aarch64_register(name),
        Target::RiscV64 => is_valid_riscv64_register(name),
    }
}

/// Check if a register name is valid for x86-64.
fn is_valid_x86_64_register(name: &str) -> bool {
    // 64-bit GPRs
    let gpr64 = [
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    // 32-bit GPR aliases
    let gpr32 = [
        "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "r8d", "r9d", "r10d", "r11d",
        "r12d", "r13d", "r14d", "r15d",
    ];
    // 16-bit GPR aliases
    let gpr16 = [
        "ax", "bx", "cx", "dx", "si", "di", "bp", "sp", "r8w", "r9w", "r10w", "r11w", "r12w",
        "r13w", "r14w", "r15w",
    ];
    // 8-bit GPR aliases
    let gpr8 = [
        "al", "bl", "cl", "dl", "sil", "dil", "bpl", "spl", "ah", "bh", "ch", "dh", "r8b", "r9b",
        "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
    ];
    // SSE/AVX registers
    let sse: Vec<String> = (0..16).map(|i| format!("xmm{}", i)).collect();
    let avx: Vec<String> = (0..16).map(|i| format!("ymm{}", i)).collect();

    gpr64.contains(&name)
        || gpr32.contains(&name)
        || gpr16.contains(&name)
        || gpr8.contains(&name)
        || sse.iter().any(|r| r == name)
        || avx.iter().any(|r| r == name)
        || name == "st"
        || name.starts_with("st(")
        || name == "flags"
        || name == "fpsr"
        || name == "fpcr"
        || name == "dirflag"
}

/// Check if a register name is valid for i686.
fn is_valid_i686_register(name: &str) -> bool {
    let gpr32 = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"];
    let gpr16 = ["ax", "bx", "cx", "dx", "si", "di", "bp", "sp"];
    let gpr8 = ["al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"];
    let sse: Vec<String> = (0..8).map(|i| format!("xmm{}", i)).collect();

    gpr32.contains(&name)
        || gpr16.contains(&name)
        || gpr8.contains(&name)
        || sse.iter().any(|r| r == name)
        || name == "st"
        || name.starts_with("st(")
        || name == "flags"
        || name == "fpsr"
        || name == "fpcr"
        || name == "dirflag"
}

/// Check if a register name is valid for AArch64.
fn is_valid_aarch64_register(name: &str) -> bool {
    // GPRs: x0-x30, w0-w30, sp, xzr, wzr
    if let Some(rest) = name.strip_prefix('x') {
        if let Ok(n) = rest.parse::<u32>() {
            return n <= 30;
        }
    }
    if let Some(rest) = name.strip_prefix('w') {
        if let Ok(n) = rest.parse::<u32>() {
            return n <= 30;
        }
    }
    // SIMD/FP: v0-v31, d0-d31, s0-s31, h0-h31, b0-b31, q0-q31
    for prefix in &['v', 'd', 's', 'h', 'b', 'q'] {
        if let Some(rest) = name.strip_prefix(*prefix) {
            if let Ok(n) = rest.parse::<u32>() {
                if n <= 31 {
                    return true;
                }
            }
        }
    }
    // Special names
    matches!(name, "sp" | "xzr" | "wzr" | "nzcv" | "fpcr" | "fpsr" | "lr")
}

/// Check if a register name is valid for RISC-V 64.
fn is_valid_riscv64_register(name: &str) -> bool {
    // Integer registers: x0-x31
    if let Some(rest) = name.strip_prefix('x') {
        if let Ok(n) = rest.parse::<u32>() {
            return n <= 31;
        }
    }
    // FP registers: f0-f31
    if let Some(rest) = name.strip_prefix('f') {
        if let Ok(n) = rest.parse::<u32>() {
            return n <= 31;
        }
    }
    // ABI names for integer registers
    let abi_int = [
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s0", "s1", "s2",
        "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "a0", "a1", "a2", "a3", "a4", "a5",
        "a6", "a7",
    ];
    // ABI names for FP registers
    let abi_fp = [
        "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",
        "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
        "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
    ];

    abi_int.contains(&name) || abi_fp.contains(&name)
}

// ===========================================================================
// Asm Goto Target Wiring
// ===========================================================================

/// Wire asm goto jump targets to IR basic blocks.
///
/// For each label name specified in the asm goto statement, looks up the
/// corresponding `BlockId` in the `label_blocks` map. If a label is not
/// found, a diagnostic error is emitted (the label may not have been
/// declared in the current function scope).
///
/// # CFG Implications
///
/// The returned `BlockId` list becomes the `goto_targets` field of the
/// `InlineAsm` instruction. The caller must also add these targets as
/// successors of the current basic block to maintain CFG correctness.
fn wire_asm_goto_targets(
    goto_labels: &[String],
    label_blocks: &FxHashMap<String, BlockId>,
    diagnostics: &mut DiagnosticEngine,
    span: Span,
) -> Vec<BlockId> {
    let mut targets: Vec<BlockId> = Vec::with_capacity(goto_labels.len());

    for label_name in goto_labels {
        match label_blocks.get(label_name) {
            Some(&block_id) => {
                targets.push(block_id);
            }
            None => {
                diagnostics.emit(Diagnostic::error(
                    span,
                    format!(
                        "asm goto label '{}' is not defined in the current function",
                        label_name
                    ),
                ));
            }
        }
    }

    targets
}

// ===========================================================================
// Template String Processing
// ===========================================================================

/// Analyze the asm template string and resolve named operand references.
///
/// GCC asm templates use the following substitution syntax:
/// - `%0`, `%1`, ... — positional operand references (outputs first, then inputs)
/// - `%[name]` — named operand reference (resolved to positional index)
/// - `%%` — literal percent sign
/// - `%h0`, `%b1`, `%w2`, ... — operand modifiers (pass through to assembler)
///
/// Named operand references (`%[name]`) are resolved to their positional
/// index using the `named_operands` map, producing `%N` in the output.
///
/// `.pushsection`/`.popsection` directives within the template are
/// preserved as-is — they are processed by the built-in assembler during
/// code generation, not during IR lowering.
fn analyze_template(template: &str, named_operands: &FxHashMap<String, usize>) -> String {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '%' {
            match chars.peek() {
                // '%%' → literal '%'
                Some('%') => {
                    chars.next();
                    result.push('%');
                    result.push('%');
                }
                // '%[name]' → named operand reference
                Some('[') => {
                    chars.next(); // consume '['
                    let mut name = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == ']' {
                            chars.next(); // consume ']'
                            break;
                        }
                        name.push(c);
                        chars.next();
                    }
                    // Resolve named operand to positional index
                    if let Some(&idx) = named_operands.get(&name) {
                        result.push('%');
                        result.push_str(&idx.to_string());
                    } else {
                        // Unknown named operand — preserve original for diagnostics
                        result.push('%');
                        result.push('[');
                        result.push_str(&name);
                        result.push(']');
                    }
                }
                // Operand modifier followed by digit: %h0, %b1, %w2, etc.
                // Or just a positional: %0, %1, etc.
                Some(c) if c.is_ascii_alphanumeric() => {
                    result.push('%');
                    // Pass through the modifier/digit sequence
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_alphanumeric() {
                            result.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                // Other: just pass through
                _ => {
                    result.push('%');
                }
            }
        } else {
            result.push(ch);
        }
    }

    result
}

// ===========================================================================
// Named Operand Resolution
// ===========================================================================

/// Build a mapping from named operand labels to positional indices using
/// pre-resolved string names.
///
/// Outputs are numbered first (0, 1, 2, ...), then inputs continue the
/// sequence. This map allows `%[name]` in the template to resolve to the
/// correct positional `%N`.
///
/// This function accepts pre-resolved string names from the caller (since
/// the AST stores symbolic names as `Symbol` handles that require an
/// `Interner` to resolve, which is not in this module's dependencies).
///
/// # Parameters
///
/// - `named_strs`: Pre-resolved symbolic names for all operands (outputs
///   first, then inputs). `None` for positional-only operands.
/// - `_output_count`: Number of output operands (reserved for future use
///   in offset calculation).
fn build_named_operand_map_from_strs(
    named_strs: &[Option<String>],
    _output_count: usize,
) -> FxHashMap<String, usize> {
    let mut map = FxHashMap::default();

    for (pos, name_opt) in named_strs.iter().enumerate() {
        if let Some(name) = name_opt {
            if !name.is_empty() {
                map.insert(name.clone(), pos);
            }
        }
    }

    map
}

/// Build a named operand mapping directly from AST operands.
///
/// This is a convenience function for callers that can provide a symbol
/// resolver function. Outputs are indexed 0..N, inputs are indexed N..N+M.
///
/// The `resolve_name` closure should return the string name for an operand's
/// `symbolic_name` field, or `None` if the operand has no named label.
pub fn build_named_operand_map(
    outputs: &[ast::AsmOperand],
    inputs: &[ast::AsmOperand],
    resolve_name: &dyn Fn(&ast::AsmOperand) -> Option<String>,
) -> FxHashMap<String, usize> {
    let mut map = FxHashMap::default();
    let mut index: usize = 0;

    for operand in outputs {
        if let Some(name) = resolve_name(operand) {
            if !name.is_empty() {
                map.insert(name, index);
            }
        }
        index += 1;
    }

    for operand in inputs {
        if let Some(name) = resolve_name(operand) {
            if !name.is_empty() {
                map.insert(name, index);
            }
        }
        index += 1;
    }

    map
}

// ===========================================================================
// Constraint String Construction
// ===========================================================================

/// Build the concatenated constraint string for the InlineAsm instruction.
///
/// The constraint string encodes all output and input constraints separated
/// by commas, matching the format expected by the code generation backend.
///
/// # Format
///
/// Output constraints come first, then input constraints, comma-separated:
/// `"=r,=m,r,i"` for two outputs (register and memory) and two inputs
/// (register and immediate).
fn build_constraints_string(outputs: &[AsmConstraint], inputs: &[AsmConstraint]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(outputs.len() + inputs.len());

    for c in outputs {
        parts.push(c.raw.clone());
    }
    for c in inputs {
        parts.push(c.raw.clone());
    }

    parts.join(",")
}

// ===========================================================================
// InlineAsm Instruction Patching
// ===========================================================================

/// Patch an InlineAsm instruction with goto targets and volatile flag.
///
/// The `IrBuilder::build_inline_asm` method creates the instruction with
/// empty `goto_targets` and derives `is_volatile` from `has_side_effects`.
/// This function patches those fields to their correct values for asm goto
/// and explicit volatile annotations.
fn patch_inline_asm_instruction(
    inst: &mut Instruction,
    goto_targets: &[BlockId],
    is_volatile: bool,
) {
    if let Instruction::InlineAsm {
        goto_targets: ref mut targets,
        is_volatile: ref mut vol,
        ..
    } = inst
    {
        *targets = goto_targets.to_vec();
        *vol = is_volatile;
    }
}

// ===========================================================================
// IR Type Inference for Constraints
// ===========================================================================

/// Infer an appropriate IR type for an operand based on its constraint.
///
/// This is used when creating load/store instructions for read-write
/// operand binding, where we need to know the data type being transferred.
///
/// The inference is based on the constraint code and target architecture:
///
/// - Register constraints (`r`, `a`-`d`, `S`, `D`): pointer-width integer
/// - Memory constraints (`m`): pointer (address)
/// - Immediate constraints (`i`, `n`): 32-bit integer (common case)
/// - Float constraints (`f`, `t`, `u`): 64-bit float
fn infer_ir_type_for_constraint(constraint: &AsmConstraint, target: &Target) -> IrType {
    let code = &constraint.constraint_code;

    if code.is_empty() {
        return if target.is_64bit() {
            IrType::I64
        } else {
            IrType::I32
        };
    }

    let first_char = code.chars().next().unwrap_or('r');

    match first_char {
        // Register: use pointer-width integer
        'r' | 'a' | 'b' | 'c' | 'd' | 'S' | 'D' | 'A' | 'q' | 'Q' | 'x' | 'w' | 'l' => {
            if target.is_64bit() {
                IrType::I64
            } else {
                IrType::I32
            }
        }
        // Memory: pointer type
        'm' | 'p' => IrType::Ptr,
        // Immediate: 32-bit integer (most common for inline asm immediates)
        'i' | 'n' | 'I' | 'J' | 'K' => IrType::I32,
        // General: pointer-width integer
        'g' | 'X' => {
            if target.is_64bit() {
                IrType::I64
            } else {
                IrType::I32
            }
        }
        // Floating-point register
        'f' | 't' | 'u' | 'y' | 'Y' => IrType::F64,
        // Digit (tied to another operand): default to pointer-width
        c if c.is_ascii_digit() => {
            if target.is_64bit() {
                IrType::I64
            } else {
                IrType::I32
            }
        }
        // Default: pointer-width integer
        _ => {
            if target.is_64bit() {
                IrType::I64
            } else {
                IrType::I32
            }
        }
    }
}

// ===========================================================================
// Section Directive Detection
// ===========================================================================

/// Check if the asm template contains `.pushsection`/`.popsection` directives.
///
/// The Linux kernel extensively uses these directives in inline assembly to
/// place exception tables, alternative instructions, and other metadata in
/// special ELF sections. These directives are passed through to the built-in
/// assembler as-is — no special IR lowering is needed.
///
/// # Returns
///
/// `true` if the template contains any section-switching directive.
pub fn has_section_directives(template: &str) -> bool {
    template.contains(".pushsection")
        || template.contains(".popsection")
        || template.contains(".section")
        || template.contains(".previous")
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Constraint parsing tests -----------------------------------------

    #[test]
    fn test_parse_output_register_constraint() {
        let c = parse_constraint("=r").unwrap();
        assert!(c.is_output);
        assert!(!c.is_read_write);
        assert!(!c.is_early_clobber);
        assert_eq!(c.constraint_code, "r");
    }

    #[test]
    fn test_parse_read_write_memory_constraint() {
        let c = parse_constraint("+m").unwrap();
        assert!(c.is_output);
        assert!(c.is_read_write);
        assert!(!c.is_early_clobber);
        assert_eq!(c.constraint_code, "m");
    }

    #[test]
    fn test_parse_early_clobber_constraint() {
        let c = parse_constraint("=&r").unwrap();
        assert!(c.is_output);
        assert!(!c.is_read_write);
        assert!(c.is_early_clobber);
        assert_eq!(c.constraint_code, "r");
    }

    #[test]
    fn test_parse_input_register_constraint() {
        let c = parse_constraint("r").unwrap();
        assert!(!c.is_output);
        assert!(!c.is_read_write);
        assert!(!c.is_early_clobber);
        assert_eq!(c.constraint_code, "r");
    }

    #[test]
    fn test_parse_immediate_constraint() {
        let c = parse_constraint("i").unwrap();
        assert!(!c.is_output);
        assert_eq!(c.constraint_code, "i");
    }

    #[test]
    fn test_parse_tied_operand_constraint() {
        let c = parse_constraint("0").unwrap();
        assert!(!c.is_output);
        assert_eq!(c.constraint_code, "0");
    }

    #[test]
    fn test_parse_empty_constraint_error() {
        assert!(parse_constraint("").is_err());
    }

    #[test]
    fn test_parse_modifier_only_error() {
        assert!(parse_constraint("=").is_err());
        assert!(parse_constraint("+").is_err());
    }

    #[test]
    fn test_parse_read_write_register() {
        let c = parse_constraint("+r").unwrap();
        assert!(c.is_output);
        assert!(c.is_read_write);
        assert_eq!(c.constraint_code, "r");
    }

    #[test]
    fn test_parse_general_constraint() {
        let c = parse_constraint("g").unwrap();
        assert!(!c.is_output);
        assert_eq!(c.constraint_code, "g");
    }

    #[test]
    fn test_parse_any_constraint() {
        let c = parse_constraint("X").unwrap();
        assert!(!c.is_output);
        assert_eq!(c.constraint_code, "X");
    }

    // -- Architecture constraint validation tests -------------------------

    #[test]
    fn test_validate_x86_specific_constraint() {
        let c = parse_constraint("a").unwrap();
        let mut diag = DiagnosticEngine::new();
        assert!(validate_constraint(
            &c,
            &Target::X86_64,
            None,
            &mut diag,
            Span::dummy()
        ));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_validate_x86_invalid_on_aarch64() {
        let c = parse_constraint("a").unwrap();
        let mut diag = DiagnosticEngine::new();
        assert!(!validate_constraint(
            &c,
            &Target::AArch64,
            None,
            &mut diag,
            Span::dummy()
        ));
        assert!(diag.has_errors());
    }

    #[test]
    fn test_validate_aarch64_simd_constraint() {
        let c = parse_constraint("w").unwrap();
        let mut diag = DiagnosticEngine::new();
        assert!(validate_constraint(
            &c,
            &Target::AArch64,
            None,
            &mut diag,
            Span::dummy()
        ));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_validate_riscv_fp_constraint() {
        let c = parse_constraint("f").unwrap();
        let mut diag = DiagnosticEngine::new();
        assert!(validate_constraint(
            &c,
            &Target::RiscV64,
            None,
            &mut diag,
            Span::dummy()
        ));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_validate_universal_constraint_all_targets() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let c = parse_constraint("r").unwrap();
            let mut diag = DiagnosticEngine::new();
            assert!(validate_constraint(
                &c,
                target,
                None,
                &mut diag,
                Span::dummy()
            ));
            assert!(!diag.has_errors());
        }
    }

    #[test]
    fn test_validate_memory_constraint_all_targets() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let c = parse_constraint("m").unwrap();
            let mut diag = DiagnosticEngine::new();
            assert!(validate_constraint(
                &c,
                target,
                None,
                &mut diag,
                Span::dummy()
            ));
            assert!(!diag.has_errors());
        }
    }

    // -- Clobber validation tests -----------------------------------------

    #[test]
    fn test_clobber_memory_and_cc() {
        let clobbers = vec!["memory".to_string(), "cc".to_string()];
        let mut diag = DiagnosticEngine::new();
        let result = process_clobbers(&clobbers, &Target::X86_64, &mut diag, Span::dummy());
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"memory".to_string()));
        assert!(result.contains(&"cc".to_string()));
    }

    #[test]
    fn test_clobber_x86_register() {
        let clobbers = vec!["rax".to_string(), "rbx".to_string()];
        let mut diag = DiagnosticEngine::new();
        let result = process_clobbers(&clobbers, &Target::X86_64, &mut diag, Span::dummy());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_clobber_aarch64_register() {
        let clobbers = vec!["x0".to_string(), "x30".to_string(), "v0".to_string()];
        let mut diag = DiagnosticEngine::new();
        let result = process_clobbers(&clobbers, &Target::AArch64, &mut diag, Span::dummy());
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_clobber_riscv_register() {
        let clobbers = vec!["a0".to_string(), "t0".to_string(), "ra".to_string()];
        let mut diag = DiagnosticEngine::new();
        let result = process_clobbers(&clobbers, &Target::RiscV64, &mut diag, Span::dummy());
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_clobber_empty_string_skipped() {
        let clobbers = vec!["".to_string(), "memory".to_string()];
        let mut diag = DiagnosticEngine::new();
        let result = process_clobbers(&clobbers, &Target::X86_64, &mut diag, Span::dummy());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "memory");
    }

    // -- Template analysis tests ------------------------------------------

    #[test]
    fn test_template_positional_operands() {
        let named = FxHashMap::default();
        let result = analyze_template("mov %0, %1", &named);
        assert_eq!(result, "mov %0, %1");
    }

    #[test]
    fn test_template_literal_percent() {
        let named = FxHashMap::default();
        let result = analyze_template("movl $100, %%eax", &named);
        assert_eq!(result, "movl $100, %%eax");
    }

    #[test]
    fn test_template_named_operand_resolution() {
        let mut named = FxHashMap::default();
        named.insert("val".to_string(), 0usize);
        named.insert("result".to_string(), 1usize);
        let result = analyze_template("mov %[val], %[result]", &named);
        assert_eq!(result, "mov %0, %1");
    }

    #[test]
    fn test_template_unknown_named_operand() {
        let named = FxHashMap::default();
        let result = analyze_template("mov %[unknown], %0", &named);
        assert_eq!(result, "mov %[unknown], %0");
    }

    #[test]
    fn test_template_operand_modifier() {
        let named = FxHashMap::default();
        let result = analyze_template("movb %h0, %b1", &named);
        assert_eq!(result, "movb %h0, %b1");
    }

    #[test]
    fn test_template_pushsection_preserved() {
        let named = FxHashMap::default();
        let template = "mov %0, %1\n.pushsection .fixup\nnop\n.popsection";
        let result = analyze_template(template, &named);
        assert!(result.contains(".pushsection .fixup"));
        assert!(result.contains(".popsection"));
    }

    // -- Named operand map tests ------------------------------------------

    #[test]
    fn test_named_operand_map_from_strs() {
        let names = vec![Some("out".to_string()), None, Some("in1".to_string())];
        let map = build_named_operand_map_from_strs(&names, 2);
        assert_eq!(map.get("out"), Some(&0));
        assert_eq!(map.get("in1"), Some(&2));
        assert_eq!(map.get("nonexistent"), None);
    }

    #[test]
    fn test_named_operand_map_empty() {
        let names: Vec<Option<String>> = vec![];
        let map = build_named_operand_map_from_strs(&names, 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_named_operand_map_all_none() {
        let names = vec![None, None, None];
        let map = build_named_operand_map_from_strs(&names, 2);
        assert!(map.is_empty());
    }

    // -- Goto target wiring tests -----------------------------------------

    #[test]
    fn test_wire_goto_targets_found() {
        let mut label_blocks = FxHashMap::default();
        label_blocks.insert("error".to_string(), BlockId(5));
        label_blocks.insert("retry".to_string(), BlockId(8));

        let labels = vec!["error".to_string(), "retry".to_string()];
        let mut diag = DiagnosticEngine::new();
        let targets = wire_asm_goto_targets(&labels, &label_blocks, &mut diag, Span::dummy());

        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0], BlockId(5));
        assert_eq!(targets[1], BlockId(8));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_wire_goto_targets_missing_label() {
        let label_blocks = FxHashMap::default();
        let labels = vec!["nonexistent".to_string()];
        let mut diag = DiagnosticEngine::new();
        let targets = wire_asm_goto_targets(&labels, &label_blocks, &mut diag, Span::dummy());

        assert!(targets.is_empty());
        assert!(diag.has_errors());
    }

    #[test]
    fn test_wire_goto_targets_empty_labels() {
        let label_blocks = FxHashMap::default();
        let labels: Vec<String> = vec![];
        let mut diag = DiagnosticEngine::new();
        let targets = wire_asm_goto_targets(&labels, &label_blocks, &mut diag, Span::dummy());

        assert!(targets.is_empty());
        assert!(!diag.has_errors());
    }

    // -- Section directive detection tests --------------------------------

    #[test]
    fn test_has_pushsection() {
        assert!(has_section_directives(".pushsection .fixup\n.popsection"));
        assert!(has_section_directives("nop\n.section .text\n"));
        assert!(has_section_directives("mov %0, %1\n.previous"));
    }

    #[test]
    fn test_no_section_directives() {
        assert!(!has_section_directives("mov %0, %1\nnop"));
        assert!(!has_section_directives(""));
    }

    // -- Constraints string building tests --------------------------------

    #[test]
    fn test_build_constraints_string() {
        let outputs = vec![AsmConstraint {
            raw: "=r".to_string(),
            is_output: true,
            is_read_write: false,
            is_early_clobber: false,
            constraint_code: "r".to_string(),
        }];
        let inputs = vec![
            AsmConstraint {
                raw: "r".to_string(),
                is_output: false,
                is_read_write: false,
                is_early_clobber: false,
                constraint_code: "r".to_string(),
            },
            AsmConstraint {
                raw: "i".to_string(),
                is_output: false,
                is_read_write: false,
                is_early_clobber: false,
                constraint_code: "i".to_string(),
            },
        ];
        let result = build_constraints_string(&outputs, &inputs);
        assert_eq!(result, "=r,r,i");
    }

    #[test]
    fn test_build_constraints_string_empty() {
        let outputs: Vec<AsmConstraint> = vec![];
        let inputs: Vec<AsmConstraint> = vec![];
        let result = build_constraints_string(&outputs, &inputs);
        assert_eq!(result, "");
    }

    // -- IR type inference tests ------------------------------------------

    #[test]
    fn test_infer_type_register_64bit() {
        let c = AsmConstraint {
            raw: "=r".to_string(),
            is_output: true,
            is_read_write: false,
            is_early_clobber: false,
            constraint_code: "r".to_string(),
        };
        assert_eq!(
            infer_ir_type_for_constraint(&c, &Target::X86_64),
            IrType::I64
        );
        assert_eq!(infer_ir_type_for_constraint(&c, &Target::I686), IrType::I32);
    }

    #[test]
    fn test_infer_type_memory() {
        let c = AsmConstraint {
            raw: "m".to_string(),
            is_output: false,
            is_read_write: false,
            is_early_clobber: false,
            constraint_code: "m".to_string(),
        };
        assert_eq!(
            infer_ir_type_for_constraint(&c, &Target::X86_64),
            IrType::Ptr
        );
    }

    #[test]
    fn test_infer_type_immediate() {
        let c = AsmConstraint {
            raw: "i".to_string(),
            is_output: false,
            is_read_write: false,
            is_early_clobber: false,
            constraint_code: "i".to_string(),
        };
        assert_eq!(
            infer_ir_type_for_constraint(&c, &Target::X86_64),
            IrType::I32
        );
    }

    #[test]
    fn test_infer_type_float_register() {
        let c = AsmConstraint {
            raw: "f".to_string(),
            is_output: false,
            is_read_write: false,
            is_early_clobber: false,
            constraint_code: "f".to_string(),
        };
        assert_eq!(
            infer_ir_type_for_constraint(&c, &Target::RiscV64),
            IrType::F64
        );
    }

    // -- Register validation tests ----------------------------------------

    #[test]
    fn test_x86_64_register_validation() {
        assert!(is_valid_x86_64_register("rax"));
        assert!(is_valid_x86_64_register("r15"));
        assert!(is_valid_x86_64_register("eax"));
        assert!(is_valid_x86_64_register("al"));
        assert!(is_valid_x86_64_register("xmm0"));
        assert!(is_valid_x86_64_register("xmm15"));
        assert!(!is_valid_x86_64_register("nonexistent"));
    }

    #[test]
    fn test_i686_register_validation() {
        assert!(is_valid_i686_register("eax"));
        assert!(is_valid_i686_register("al"));
        assert!(is_valid_i686_register("xmm0"));
        assert!(!is_valid_i686_register("rax")); // 64-bit not valid on i686
    }

    #[test]
    fn test_aarch64_register_validation() {
        assert!(is_valid_aarch64_register("x0"));
        assert!(is_valid_aarch64_register("x30"));
        assert!(is_valid_aarch64_register("w0"));
        assert!(is_valid_aarch64_register("v0"));
        assert!(is_valid_aarch64_register("sp"));
        assert!(is_valid_aarch64_register("lr"));
        assert!(!is_valid_aarch64_register("x31")); // only x0-x30
        assert!(!is_valid_aarch64_register("rax"));
    }

    #[test]
    fn test_riscv64_register_validation() {
        assert!(is_valid_riscv64_register("x0"));
        assert!(is_valid_riscv64_register("x31"));
        assert!(is_valid_riscv64_register("a0"));
        assert!(is_valid_riscv64_register("t0"));
        assert!(is_valid_riscv64_register("ra"));
        assert!(is_valid_riscv64_register("sp"));
        assert!(is_valid_riscv64_register("fa0"));
        assert!(!is_valid_riscv64_register("rax"));
    }
}
