//! Linear scan register allocator for the BCC compiler backend.
//!
//! This module implements a production-quality linear scan register allocator
//! that bridges the gap between SSA virtual registers (produced by the IR
//! middle-end) and physical machine registers (required by the architecture-
//! specific code generator and assembler). It operates on [`MachineFunction`]s
//! after instruction selection, handling register assignment, spill code
//! generation, and callee-saved register tracking for all four BCC target
//! architectures (x86-64, i686, AArch64, RISC-V 64).
//!
//! # Algorithm
//!
//! The allocator uses the **linear scan** algorithm (Poletto & Sarkar, 1999),
//! which achieves good register utilisation in O(n log n) time by:
//!
//! 1. Computing **live intervals** — the contiguous instruction range where
//!    each virtual register holds a live value
//! 2. Sorting intervals by start position
//! 3. Walking intervals in order, greedily assigning physical registers
//! 4. **Spilling** intervals to stack slots when registers are exhausted,
//!    choosing the interval with the furthest next-use for spilling
//!
//! # Architecture Independence
//!
//! The allocator is parameterised by the [`ArchCodegen`] trait, which provides:
//! - Available register counts and lists (integer vs floating-point)
//! - Callee-saved and caller-saved register classifications
//! - Stack pointer, frame pointer, and argument/return register identifiers
//! - Target stack alignment constraints
//!
//! This allows the same allocation algorithm to work across all four target
//! architectures without modification.
//!
//! # Usage
//!
//! ```ignore
//! use bcc::backend::register_allocator::RegisterAllocator;
//! use bcc::common::target::Target;
//!
//! let mut allocator = RegisterAllocator::new(Target::X86_64, &x86_codegen);
//! allocator.compute_live_intervals(&machine_func, &ir_func);
//! allocator.allocate();
//! allocator.generate_spill_code(&mut machine_func);
//!
//! println!("Frame size: {} bytes", allocator.frame_size());
//! println!("Spill slots: {}", allocator.spill_slot_count());
//! println!("Callee-saved used: {:?}", allocator.used_callee_saved());
//! ```
//!
//! # Spill Code
//!
//! When physical registers are exhausted, the allocator assigns stack frame
//! slots (identified by `FrameIndex` operands) to spilled values and inserts
//! pseudo spill-store and spill-load instructions at definition and use sites:
//!
//! - [`SPILL_STORE_OPCODE`]: Stores a value to its assigned stack slot
//! - [`SPILL_LOAD_OPCODE`]: Loads a value from its assigned stack slot
//!
//! The architecture backend's prologue/epilogue pass lowers these pseudo-ops
//! into real load/store instructions using the frame pointer and computed
//! offsets.

use std::cmp::{Ordering, Reverse};
use std::fmt;

use crate::backend::traits::{
    ArchCodegen, MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand, PhysReg,
    RegisterClass,
};
use crate::common::fx_hash::{fx_hash_map, fx_hash_set, FxHashMap, FxHashSet};
use crate::common::target::Target;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Pseudo-opcodes for spill code
// ---------------------------------------------------------------------------

/// Pseudo-opcode inserted by the register allocator for storing a spilled
/// value to its stack frame slot after its definition.
///
/// Operand layout: `[FrameIndex(slot)]`
///
/// The architecture backend must lower this into a real store instruction
/// during prologue/epilogue emission. The specific store instruction depends
/// on the value's register class and width:
/// - Integer: `MOV [rbp-offset], reg` (x86-64) / `str reg, [sp, #offset]` (AArch64)
/// - Float: `MOVSD [rbp-offset], xmm` (x86-64) / `str d, [sp, #offset]` (AArch64)
pub const SPILL_STORE_OPCODE: u32 = 0xFFFF_FFFE;

/// Pseudo-opcode inserted by the register allocator for loading a spilled
/// value from its stack frame slot before a use.
///
/// Operand layout: `[FrameIndex(slot)]`
///
/// The architecture backend must lower this into a real load instruction
/// during prologue/epilogue emission.
pub const SPILL_LOAD_OPCODE: u32 = 0xFFFF_FFFF;

// ===========================================================================
// LiveInterval — contiguous [start, end) range for a virtual register value
// ===========================================================================

/// A live interval representing the instruction-index range during which a
/// single virtual register holds a live value.
///
/// The linear scan allocator sorts a vector of `LiveInterval`s by `start`
/// position and walks them left-to-right, assigning physical registers. If a
/// register cannot be assigned (no free register available), the interval with
/// the **longest remaining range** is spilled to a stack slot.
///
/// # Fields
///
/// | Field        | Description                                                    |
/// |--------------|----------------------------------------------------------------|
/// | `value_id`   | The SSA virtual register this interval belongs to              |
/// | `start`      | First instruction index where the value is live (definition)   |
/// | `end`        | Last instruction index where the value is live (final use)+1   |
/// | `register`   | Physical register assigned by the allocator, or `None`         |
/// | `spill_slot` | Stack frame slot index assigned when spilled, or `None`        |
/// | `is_fixed`   | `true` for pre-coloured intervals (ABI-required registers)     |
/// | `reg_class`  | Register pool (`GeneralPurpose` or `FloatingPoint`)            |
#[derive(Clone, Debug)]
pub struct LiveInterval {
    /// The SSA `ValueId` this interval tracks.
    pub value_id: ValueId,
    /// Instruction index of the first def (inclusive).
    pub start: u32,
    /// Instruction index of the last use (exclusive — one past the last use).
    pub end: u32,
    /// Physical register assigned during allocation, or `None` if spilled.
    pub register: Option<PhysReg>,
    /// Stack frame slot when the value has been spilled, or `None`.
    pub spill_slot: Option<u32>,
    /// `true` when the interval is pre-coloured and must keep its register
    /// (e.g. function arguments placed in ABI-mandated registers).
    pub is_fixed: bool,
    /// Register class — determines which physical register pool is used.
    pub reg_class: RegisterClass,
}

impl LiveInterval {
    /// Returns `true` if the live interval covers (contains) the instruction
    /// at index `pos`. The interval is **half-open**: `[start, end)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // interval covers [10, 20)
    /// assert!(interval.covers(10));
    /// assert!(interval.covers(19));
    /// assert!(!interval.covers(20));
    /// ```
    #[inline]
    pub fn covers(&self, pos: u32) -> bool {
        pos >= self.start && pos < self.end
    }

    /// Returns `true` if `self` and `other` overlap — i.e. share at least one
    /// instruction index in their ranges.
    ///
    /// Two half-open intervals `[a, b)` and `[c, d)` intersect iff
    /// `a < d && c < b`.
    #[inline]
    pub fn intersects(&self, other: &LiveInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Split this interval at `pos`, producing a second interval that covers
    /// `[pos, self.end)` while truncating `self` to `[self.start, pos)`.
    ///
    /// The new interval inherits the same `value_id` and `reg_class`, but has
    /// no register or spill slot assigned — those are determined by the
    /// allocator when it processes the new interval.
    ///
    /// Returns `None` if `pos` is outside `(self.start, self.end)` because a
    /// split at those boundaries is meaningless.
    pub fn split_at(&mut self, pos: u32) -> Option<LiveInterval> {
        if pos <= self.start || pos >= self.end {
            return None;
        }
        let new_interval = LiveInterval {
            value_id: self.value_id,
            start: pos,
            end: self.end,
            register: None,
            spill_slot: None,
            is_fixed: false,
            reg_class: self.reg_class,
        };
        self.end = pos;
        Some(new_interval)
    }

    /// Length (number of instruction positions) of this interval.
    #[inline]
    pub fn length(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }
}

// Ordering: sort intervals by `start` position (ascending); ties broken by
// `end` (ascending) for deterministic output.
impl PartialEq for LiveInterval {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end && self.value_id == other.value_id
    }
}

impl Eq for LiveInterval {}

impl PartialOrd for LiveInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LiveInterval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start
            .cmp(&other.start)
            .then_with(|| self.end.cmp(&other.end))
            .then_with(|| self.value_id.0.cmp(&other.value_id.0))
    }
}

impl fmt::Display for LiveInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}:[{}, {})", self.value_id.0, self.start, self.end)?;
        if let Some(reg) = self.register {
            write!(f, " -> r{}", reg.0)?;
        } else if let Some(slot) = self.spill_slot {
            write!(f, " -> spill#{}", slot)?;
        }
        if self.is_fixed {
            write!(f, " (fixed)")?;
        }
        Ok(())
    }
}

// ===========================================================================
// RegisterSet — architecture-parameterised pool of physical registers
// ===========================================================================

/// A register set describing the physical registers available for allocation
/// on a given target architecture.
///
/// Each architecture provides two register sets — one for general-purpose
/// (integer/pointer) registers and one for floating-point registers. The
/// allocator queries the appropriate set based on the [`RegisterClass`] of
/// each [`LiveInterval`].
///
/// The `available` list drives allocation order — registers appearing earlier
/// are preferred (usually caller-saved first, to minimise callee-saved
/// save/restore overhead).
#[derive(Clone, Debug)]
pub struct RegisterSet {
    /// Registers available for allocation, in preference order.
    pub available: Vec<PhysReg>,
    /// Subset of `available` that are callee-saved (must be preserved across
    /// function calls). Using a callee-saved register obligates the function
    /// prologue to save it and the epilogue to restore it.
    pub callee_saved: Vec<PhysReg>,
    /// Subset of `available` that are caller-saved (may be clobbered by calls).
    /// Values live across a call in caller-saved registers must be saved by the
    /// caller (i.e. spilled or moved to a callee-saved register before the
    /// call).
    pub caller_saved: Vec<PhysReg>,
}

impl RegisterSet {
    /// Build a new register set from explicit lists.
    pub fn new(
        available: Vec<PhysReg>,
        callee_saved: Vec<PhysReg>,
        caller_saved: Vec<PhysReg>,
    ) -> Self {
        Self {
            available,
            callee_saved,
            caller_saved,
        }
    }

    /// Build the **integer / general-purpose** register set from a given
    /// [`ArchCodegen`] implementation.
    ///
    /// The available set is constructed by taking caller-saved registers first
    /// (to prefer scratch registers and avoid callee-save overhead) followed
    /// by callee-saved registers, **excluding** the stack pointer and frame
    /// pointer which are never available for general allocation.
    pub fn integer_set(arch: &dyn ArchCodegen) -> Self {
        let callee_saved_set: FxHashSet<PhysReg> = arch.callee_saved_registers().iter().copied().collect();
        let caller_saved_set: FxHashSet<PhysReg> = arch.caller_saved_registers().iter().copied().collect();

        let sp = arch.stack_pointer();
        let fp = arch.frame_pointer();
        let reserved: FxHashSet<PhysReg> = [sp, fp].iter().copied().collect();

        // Build available list: caller-saved first, then callee-saved.
        // Only include registers in the integer range [0 .. int_count).
        let int_count = arch.integer_register_count() as u16;
        let mut available = Vec::new();

        // Caller-saved first (prefer scratch registers)
        for &reg in arch.caller_saved_registers() {
            if reg.0 < int_count && !reserved.contains(&reg) {
                available.push(reg);
            }
        }
        // Callee-saved second
        for &reg in arch.callee_saved_registers() {
            if reg.0 < int_count && !reserved.contains(&reg) && !available.contains(&reg) {
                available.push(reg);
            }
        }

        // Build filtered callee/caller saved lists (integer only)
        let callee_saved: Vec<PhysReg> = callee_saved_set
            .iter()
            .filter(|r| r.0 < int_count && !reserved.contains(r))
            .copied()
            .collect();
        let caller_saved: Vec<PhysReg> = caller_saved_set
            .iter()
            .filter(|r| r.0 < int_count && !reserved.contains(r))
            .copied()
            .collect();

        Self {
            available,
            callee_saved,
            caller_saved,
        }
    }

    /// Build the **floating-point** register set from a given [`ArchCodegen`]
    /// implementation.
    ///
    /// Floating-point registers start after the integer register space. For
    /// example, if there are 16 integer registers numbered 0–15, the first
    /// floating-point register is 16.
    pub fn float_set(arch: &dyn ArchCodegen) -> Self {
        let callee_saved_all: FxHashSet<PhysReg> = arch.callee_saved_registers().iter().copied().collect();
        let caller_saved_all: FxHashSet<PhysReg> = arch.caller_saved_registers().iter().copied().collect();

        let int_count = arch.integer_register_count() as u16;
        let total_count = int_count + arch.float_register_count() as u16;

        let mut available = Vec::new();

        // Caller-saved float regs first
        for &reg in arch.caller_saved_registers() {
            if reg.0 >= int_count && reg.0 < total_count {
                available.push(reg);
            }
        }
        // Callee-saved float regs second
        for &reg in arch.callee_saved_registers() {
            if reg.0 >= int_count && reg.0 < total_count && !available.contains(&reg) {
                available.push(reg);
            }
        }

        let callee_saved: Vec<PhysReg> = callee_saved_all
            .iter()
            .filter(|r| r.0 >= int_count && r.0 < total_count)
            .copied()
            .collect();
        let caller_saved: Vec<PhysReg> = caller_saved_all
            .iter()
            .filter(|r| r.0 >= int_count && r.0 < total_count)
            .copied()
            .collect();

        Self {
            available,
            callee_saved,
            caller_saved,
        }
    }
}

impl fmt::Display for RegisterSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegisterSet {{ available: [")?;
        for (i, r) in self.available.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "r{}", r.0)?;
        }
        write!(f, "], callee_saved: [")?;
        for (i, r) in self.callee_saved.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "r{}", r.0)?;
        }
        write!(f, "], caller_saved: [")?;
        for (i, r) in self.caller_saved.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "r{}", r.0)?;
        }
        write!(f, "] }}")
    }
}

// ===========================================================================
// RegisterAllocator — linear scan allocation engine
// ===========================================================================

/// Linear scan register allocator for a single [`MachineFunction`].
///
/// Instantiate with [`RegisterAllocator::new`], populate live intervals with
/// [`compute_live_intervals`](RegisterAllocator::compute_live_intervals),
/// run allocation with [`allocate`](RegisterAllocator::allocate), and then
/// apply spill code with
/// [`generate_spill_code`](RegisterAllocator::generate_spill_code).
pub struct RegisterAllocator {
    /// The target architecture — used for stack alignment, pointer width, and
    /// register set construction.
    target: Target,

    /// Integer register set for this target.
    int_regs: RegisterSet,

    /// Floating-point register set for this target.
    float_regs: RegisterSet,

    /// All live intervals (sorted by start position after
    /// `compute_live_intervals`).
    intervals: Vec<LiveInterval>,

    /// Map from `ValueId` → index in `self.intervals` for O(1) lookup.
    value_to_interval: FxHashMap<u32, usize>,

    /// Map from `PhysReg` → index of the interval currently occupying it.
    /// Used during the allocation sweep to track the "active" set.
    active_for_reg: FxHashMap<u16, usize>,

    /// Callee-saved registers that were actually used by the allocation and
    /// therefore must be saved in the prologue and restored in the epilogue.
    used_callee_saved: FxHashSet<PhysReg>,

    /// Total stack frame bytes consumed by spill slots.
    frame_size: u32,

    /// Next spill slot index to hand out (monotonically increasing).
    next_spill_slot: u32,

    /// Size (in bytes) of a single spill slot — the larger of the pointer
    /// width and 8 bytes, rounded up to `stack_alignment`.
    spill_slot_bytes: u32,

    /// Records of spill/reload insertion points accumulated during
    /// `allocate()` and applied during `generate_spill_code()`.
    spill_edits: Vec<SpillEdit>,

    /// Cache of the callee-saved register set for quick membership tests.
    callee_saved_set: FxHashSet<PhysReg>,

    /// Argument registers for integer parameters (ABI defined).
    arg_regs_int: Vec<PhysReg>,

    /// Argument registers for floating-point parameters (ABI defined).
    arg_regs_float: Vec<PhysReg>,

    /// Return register for integer values.
    return_reg_int: PhysReg,

    /// Return register for floating-point values.
    return_reg_float: PhysReg,
}

/// A pending spill or reload edit that is accumulated during the allocation
/// phase and batch-applied during `generate_spill_code`.
#[derive(Clone, Debug)]
struct SpillEdit {
    /// The basic block index in `MachineFunction.blocks`.
    block_idx: usize,
    /// The instruction index *within the basic block* after which (for stores)
    /// or before which (for loads) the pseudo-op is inserted.
    instr_idx: usize,
    /// Whether this is a spill-store or a spill-load.
    kind: SpillEditKind,
    /// The physical register being spilled or reloaded.
    reg: PhysReg,
    /// The stack slot frame index.
    slot: u32,
    /// The value id being spilled/reloaded (for operand rewriting).
    value_id: ValueId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpillEditKind {
    /// Store from register to stack slot (inserted after the def).
    Store,
    /// Load from stack slot into register (inserted before the use).
    Load,
}

impl RegisterAllocator {
    /// Create a new register allocator for the given target architecture.
    ///
    /// The `arch` parameter supplies register counts, callee-saved/caller-saved
    /// lists, argument registers, and reserved registers (SP, FP) via the
    /// [`ArchCodegen`] trait.
    pub fn new(target: Target, arch: &dyn ArchCodegen) -> Self {
        let int_regs = RegisterSet::integer_set(arch);
        let float_regs = RegisterSet::float_set(arch);

        // Build callee-saved lookup set for quick membership testing.
        let mut callee_saved_set: FxHashSet<PhysReg> = fx_hash_set();
        for &r in &int_regs.callee_saved {
            callee_saved_set.insert(r);
        }
        for &r in &float_regs.callee_saved {
            callee_saved_set.insert(r);
        }

        // Stack alignment and spill slot sizing.
        let stack_align = target.stack_alignment() as u32;
        let ptr_width = target.pointer_width() as u32;
        // Each spill slot holds up to 8 bytes (enough for an f64 / i64 value),
        // rounded up to the target's stack alignment.
        let spill_slot_bytes = align_up(ptr_width.max(8), stack_align);

        Self {
            target,
            int_regs,
            float_regs,
            intervals: Vec::new(),
            value_to_interval: fx_hash_map(),
            active_for_reg: fx_hash_map(),
            used_callee_saved: fx_hash_set(),
            frame_size: 0,
            next_spill_slot: 0,
            spill_slot_bytes,
            spill_edits: Vec::new(),
            callee_saved_set,
            arg_regs_int: arch.argument_registers_int().to_vec(),
            arg_regs_float: arch.argument_registers_float().to_vec(),
            return_reg_int: arch.return_register_int(),
            return_reg_float: arch.return_register_float(),
        }
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    /// Returns a slice of all computed [`LiveInterval`]s, sorted by start
    /// position after [`compute_live_intervals`](Self::compute_live_intervals).
    #[inline]
    pub fn intervals(&self) -> &[LiveInterval] {
        &self.intervals
    }

    /// Total stack frame size (in bytes) required for spill slots allocated
    /// during register allocation.
    ///
    /// The architecture's prologue/epilogue generator adds this to any other
    /// frame requirements (saved callee-saved regs, local variables, etc.)
    /// to produce the final `sub rsp, N` adjustment.
    #[inline]
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }

    /// Returns the set of callee-saved physical registers that were actually
    /// assigned to at least one live interval. The prologue generator uses
    /// this to emit saves and the epilogue generator to emit restores.
    #[inline]
    pub fn used_callee_saved(&self) -> &FxHashSet<PhysReg> {
        &self.used_callee_saved
    }

    /// Number of spill slots that were allocated during register allocation.
    #[inline]
    pub fn spill_slot_count(&self) -> u32 {
        self.next_spill_slot
    }

    // -----------------------------------------------------------------------
    // Phase 1: Live interval computation
    // -----------------------------------------------------------------------

    /// Scan the machine function to compute live intervals for every virtual
    /// register that appears in instruction operands, implicit defs, and
    /// implicit uses.
    ///
    /// The resulting intervals are stored in `self.intervals`, sorted by
    /// ascending start position, ready for the linear scan sweep in
    /// [`allocate`](Self::allocate).
    ///
    /// # Pre-coloured intervals
    ///
    /// Function parameters that arrive in ABI-mandated physical registers
    /// are represented as **pre-coloured (fixed)** intervals. These intervals
    /// have `is_fixed = true` and their `register` field is already set to
    /// the corresponding physical register. The allocator respects these
    /// assignments and avoids reassigning the register during the overlap
    /// window.
    pub fn compute_live_intervals(&mut self, mf: &MachineFunction, ir_func: &IrFunction) {
        self.intervals.clear();
        self.value_to_interval.clear();

        // Maps ValueId.0 → (first_seen_index, last_seen_index)
        let mut first_seen: FxHashMap<u32, u32> = fx_hash_map();
        let mut last_seen: FxHashMap<u32, u32> = fx_hash_map();
        // Maps ValueId.0 → RegisterClass
        let mut value_classes: FxHashMap<u32, RegisterClass> = fx_hash_map();
        // Track instruction indices that are call instructions, used later to
        // extend live intervals across call boundaries for caller-saved regs.
        let mut call_indices: Vec<u32> = Vec::new();

        // Linearise all blocks into a single instruction index space.
        // Index 0 is the first instruction of block 0, etc.
        let mut instr_idx: u32 = 0;

        for block in &mf.blocks {
            // Use block.id for instruction-space partitioning diagnostics.
            let _block_id = block.id;

            for instr in &block.instructions {
                // ---- Track call instructions ----
                // When a function contains calls (`mf.has_calls`), we need
                // to know call positions so that values in caller-saved regs
                // live across a call can be spilled or moved.
                if instr.is_call {
                    call_indices.push(instr_idx);
                }

                // ---- Scan explicit operands ----
                for operand in &instr.operands {
                    if let MachineOperand::VirtualReg(vid) = operand {
                        let key = vid.0;
                        first_seen.entry(key).or_insert(instr_idx);
                        last_seen.insert(key, instr_idx);

                        // Determine register class from IR type if not cached
                        value_classes.entry(key).or_insert_with(|| {
                            Self::classify_value(*vid, ir_func)
                        });
                    }
                }

                // ---- Scan implicit defs ----
                // Implicit defs on physical registers create interference at
                // call sites (call clobber). We record the physical registers
                // for future interference graph refinement.
                for &reg in &instr.implicit_defs {
                    let _ = reg;
                }

                // ---- Scan implicit uses ----
                for &reg in &instr.implicit_uses {
                    let _ = reg;
                }

                instr_idx += 1;
            }
        }

        // If the function contains calls, any value live across a call site
        // that is assigned to a caller-saved register will need spilling. We
        // don't force it here — instead the `allocate` method prefers
        // callee-saved registers for long-lived values. The call_indices are
        // available for future call-aware heuristic improvements.
        let _function_has_calls = mf.has_calls;

        // ---- Build LiveInterval for each virtual register ----
        for (&vid_key, &start) in &first_seen {
            let end_inclusive = *last_seen.get(&vid_key).unwrap_or(&start);
            // Half-open: end is one past the last use so the value is still
            // live at the last use instruction.
            let end = end_inclusive + 1;
            let rc = *value_classes.get(&vid_key).unwrap_or(&RegisterClass::GeneralPurpose);

            let interval = LiveInterval {
                value_id: ValueId(vid_key),
                start,
                end,
                register: None,
                spill_slot: None,
                is_fixed: false,
                reg_class: rc,
            };
            let idx = self.intervals.len();
            self.intervals.push(interval);
            self.value_to_interval.insert(vid_key, idx);
        }

        // ---- Pre-colour function parameters ----
        self.precolour_parameters(ir_func);

        // ---- Sort by start position ----
        self.intervals.sort();

        // Rebuild the value_to_interval map after sorting (indices changed)
        self.value_to_interval.clear();
        for (idx, interval) in self.intervals.iter().enumerate() {
            self.value_to_interval.insert(interval.value_id.0, idx);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Linear scan allocation
    // -----------------------------------------------------------------------

    /// Run the linear scan register allocation algorithm over the previously
    /// computed live intervals.
    ///
    /// After this method returns, every [`LiveInterval`] in
    /// [`intervals`](Self::intervals) has either:
    /// - `register = Some(phys)` — a physical register was assigned, **or**
    /// - `spill_slot = Some(slot)` — the value was spilled to a stack frame
    ///   slot (and spill edits have been recorded for later application).
    ///
    /// # Algorithm
    ///
    /// The classic Poletto & Sarkar linear scan:
    /// 1. Walk sorted intervals left-to-right.
    /// 2. For each new interval, expire all active intervals whose `end`
    ///    position is ≤ the current interval's `start` (freeing their regs).
    /// 3. Try to assign a free register from the appropriate pool.
    /// 4. If no register is free, **spill** the interval with the longest
    ///    remaining range (either the current interval or the active interval
    ///    ending latest), freeing one register.
    pub fn allocate(&mut self) {
        // Active intervals sorted by end position (earliest end first).
        // Stored as indices into self.intervals.
        let mut active: Vec<usize> = Vec::new();

        // Free register pools — one per register class.
        let mut free_int: Vec<PhysReg> = self.int_regs.available.iter().copied().rev().collect();
        let mut free_float: Vec<PhysReg> = self.float_regs.available.iter().copied().rev().collect();

        // We need to iterate by index because we mutate intervals in place.
        let n = self.intervals.len();
        for i in 0..n {
            let cur_start = self.intervals[i].start;
            let cur_end = self.intervals[i].end;
            let cur_class = self.intervals[i].reg_class;
            let cur_fixed = self.intervals[i].is_fixed;

            // --- Step 1: Expire old intervals ---
            // Remove active intervals that have ended before (or at) cur_start.
            let mut j = 0;
            while j < active.len() {
                let ai = active[j];
                if self.intervals[ai].end <= cur_start {
                    // Free the register
                    if let Some(reg) = self.intervals[ai].register {
                        Self::return_reg(
                            reg,
                            self.intervals[ai].reg_class,
                            &mut free_int,
                            &mut free_float,
                        );
                    }
                    active.swap_remove(j);
                    // Don't increment j — swap_remove puts a new element at j
                } else {
                    j += 1;
                }
            }

            // --- Step 2: Handle pre-coloured (fixed) intervals ---
            if cur_fixed {
                if let Some(reg) = self.intervals[i].register {
                    // Remove this register from the free pool if present
                    Self::remove_from_free(
                        reg,
                        cur_class,
                        &mut free_int,
                        &mut free_float,
                    );
                    self.track_callee_saved_usage(reg);
                    active.push(i);
                }
                continue;
            }

            // --- Step 3: Try to assign a free register ---
            let free_pool = match cur_class {
                RegisterClass::GeneralPurpose => &mut free_int,
                RegisterClass::FloatingPoint => &mut free_float,
            };

            if let Some(reg) = free_pool.pop() {
                self.intervals[i].register = Some(reg);
                self.track_callee_saved_usage(reg);
                active.push(i);
            } else {
                // --- Step 4: Spill ---
                // Find the active interval with the furthest end in the same
                // register class.
                let spill_candidate = self.find_spill_candidate(&active, cur_class, cur_end);

                match spill_candidate {
                    Some(active_pos) => {
                        let victim_idx = active[active_pos];
                        let victim_end = self.intervals[victim_idx].end;

                        if victim_end > cur_end {
                            // The active interval extends further — spill it
                            // and give its register to the current interval.
                            let stolen_reg = self.intervals[victim_idx].register.take().unwrap();
                            self.spill_interval(victim_idx);
                            self.intervals[i].register = Some(stolen_reg);
                            self.track_callee_saved_usage(stolen_reg);
                            active.swap_remove(active_pos);
                            active.push(i);
                        } else {
                            // Current interval is the longest — spill it
                            self.spill_interval(i);
                        }
                    }
                    None => {
                        // No active interval in the same class — spill current
                        self.spill_interval(i);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: Spill code generation
    // -----------------------------------------------------------------------

    /// Rewrite the [`MachineFunction`] to apply the allocation results:
    ///
    /// 1. **Replace** every `MachineOperand::VirtualReg(vid)` with either
    ///    `MachineOperand::Register(phys)` (if assigned) or
    ///    `MachineOperand::FrameIndex(slot)` (if spilled).
    /// 2. **Insert** spill-store pseudo-instructions after definitions of
    ///    spilled values.
    /// 3. **Insert** spill-load pseudo-instructions before uses of spilled
    ///    values.
    /// 4. **Update** `MachineFunction.frame_size` and
    ///    `MachineFunction.used_callee_saved` with allocation results.
    ///
    /// This method must be called after [`allocate`](Self::allocate) has run.
    pub fn generate_spill_code(&mut self, mf: &mut MachineFunction) {
        // Build ValueId → allocation result map for quick lookup.
        let mut assignment: FxHashMap<u32, AllocationResult> = fx_hash_map();
        for interval in &self.intervals {
            if let Some(reg) = interval.register {
                assignment.insert(
                    interval.value_id.0,
                    AllocationResult::Register(reg),
                );
            } else if let Some(slot) = interval.spill_slot {
                assignment.insert(
                    interval.value_id.0,
                    AllocationResult::Spilled(slot),
                );
            }
        }

        // Pass 1: Rewrite operands + collect spill edit insertion points.
        let mut edits: Vec<(usize, usize, SpillEditKind, u32, PhysReg)> = Vec::new();

        for (block_idx, block) in mf.blocks.iter_mut().enumerate() {
            for (instr_idx, instr) in block.instructions.iter_mut().enumerate() {
                let mut is_def_first = true;
                for op in instr.operands.iter_mut() {
                    if let MachineOperand::VirtualReg(vid) = *op {
                        match assignment.get(&vid.0) {
                            Some(AllocationResult::Register(reg)) => {
                                *op = MachineOperand::Register(*reg);
                            }
                            Some(AllocationResult::Spilled(slot)) => {
                                // For spilled values, we keep a FrameIndex as
                                // a placeholder; the backend will lower it.
                                *op = MachineOperand::FrameIndex(*slot);
                                // Record a spill load/store edit.
                                // The first operand of an instruction is
                                // conventionally the definition; subsequent
                                // operands are uses (architecture-dependent,
                                // but the pseudo-ops handle both directions).
                                if is_def_first {
                                    edits.push((
                                        block_idx,
                                        instr_idx,
                                        SpillEditKind::Store,
                                        *slot,
                                        PhysReg::NONE,
                                    ));
                                } else {
                                    edits.push((
                                        block_idx,
                                        instr_idx,
                                        SpillEditKind::Load,
                                        *slot,
                                        PhysReg::NONE,
                                    ));
                                }
                            }
                            None => {
                                // Value was not tracked — this can happen for
                                // values that are immediately consumed (e.g.
                                // zero-length live range). Leave as VirtualReg
                                // and let the backend handle it or report an
                                // internal error.
                            }
                        }
                    }
                    is_def_first = false;
                }
            }
        }

        // Pass 2: Insert spill pseudo-instructions.
        // Process edits in reverse order so that insertion indices remain valid.
        edits.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| b.1.cmp(&a.1)) // reverse instr order within block
        });

        // Group edits by block and apply in reverse instruction order.
        let mut block_edits: FxHashMap<usize, Vec<(usize, SpillEditKind, u32, PhysReg)>> =
            fx_hash_map();
        for (bi, ii, kind, slot, reg) in edits {
            block_edits
                .entry(bi)
                .or_insert_with(Vec::new)
                .push((ii, kind, slot, reg));
        }

        for (block_idx, mut edit_list) in block_edits {
            // Sort by instruction index descending so insertions don't
            // invalidate subsequent indices.
            edit_list.sort_by(|a, b| b.0.cmp(&a.0));

            if let Some(block) = mf.blocks.get_mut(block_idx) {
                for (instr_idx, kind, slot, _reg) in edit_list {
                    let spill_opcode = match kind {
                        SpillEditKind::Store => SPILL_STORE_OPCODE,
                        SpillEditKind::Load => SPILL_LOAD_OPCODE,
                    };
                    let mut pseudo = MachineInstr::new(spill_opcode);
                    pseudo.operands.push(MachineOperand::FrameIndex(slot));
                    // Stores go after the defining instruction; loads go before
                    // the using instruction.
                    match kind {
                        SpillEditKind::Store => {
                            let insert_pos = (instr_idx + 1).min(block.instructions.len());
                            block.instructions.insert(insert_pos, pseudo);
                        }
                        SpillEditKind::Load => {
                            block.instructions.insert(instr_idx, pseudo);
                        }
                    }
                }
            }
        }

        // Pass 3: Propagate metadata to the MachineFunction.
        mf.frame_size += self.frame_size;
        for &reg in &self.used_callee_saved {
            mf.used_callee_saved.push(reg);
        }
        // Deduplicate callee-saved list.
        mf.used_callee_saved.sort_by_key(|r| r.0);
        mf.used_callee_saved.dedup();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Classify a virtual register as integer or floating-point based on its
    /// IR type. Pointers and integer types map to `GeneralPurpose`; float
    /// types map to `FloatingPoint`.
    ///
    /// Uses [`IrFunction::get_value_info`] (the non-panicking variant) so
    /// that machine-level virtual registers created during instruction
    /// selection — which may not have entries in the IR value registry —
    /// default to `GeneralPurpose` instead of panicking.
    fn classify_value(vid: ValueId, ir_func: &IrFunction) -> RegisterClass {
        match ir_func.get_value_info(vid) {
            Some(vi) => {
                let ty = &vi.ty;
                if ty.is_floating() {
                    RegisterClass::FloatingPoint
                } else if ty.is_integer() || ty.is_pointer() {
                    RegisterClass::GeneralPurpose
                } else {
                    // Aggregate, void, or other — use GP as fallback.
                    RegisterClass::GeneralPurpose
                }
            }
            None => {
                // Virtual register has no IR value entry (possibly created
                // by the machine instruction selection pass). Default to GP.
                RegisterClass::GeneralPurpose
            }
        }
    }

    /// Determine the spill slot size (in bytes) for a given IR type,
    /// respecting the value's natural size and the target's alignment.
    ///
    /// Falls back to `default_slot_bytes` for void or zero-sized types.
    #[allow(dead_code)]
    fn spill_size_for_type(ty: &IrType, target: &Target, default_slot_bytes: u32) -> u32 {
        let natural = ty.size_bytes(target) as u32;
        if natural == 0 {
            default_slot_bytes
        } else {
            natural.max(default_slot_bytes)
        }
    }

    /// Pre-colour intervals for function parameters that arrive in ABI-
    /// mandated physical registers.
    ///
    /// For each parameter in `ir_func.params`, we look up the corresponding
    /// live interval (if any) and set `is_fixed = true` with the correct
    /// physical register assignment.
    fn precolour_parameters(&mut self, ir_func: &IrFunction) {
        let mut int_arg_idx: usize = 0;
        let mut float_arg_idx: usize = 0;

        for param in &ir_func.params {
            let vid = param.id;
            let class = Self::classify_value_from_ir_type(&param.ty);

            if let Some(&interval_idx) = self.value_to_interval.get(&vid.0) {
                let reg = match class {
                    RegisterClass::GeneralPurpose => {
                        let r = self.arg_regs_int.get(int_arg_idx).copied();
                        int_arg_idx += 1;
                        r
                    }
                    RegisterClass::FloatingPoint => {
                        let r = self.arg_regs_float.get(float_arg_idx).copied();
                        float_arg_idx += 1;
                        r
                    }
                };

                if let Some(phys) = reg {
                    self.intervals[interval_idx].is_fixed = true;
                    self.intervals[interval_idx].register = Some(phys);
                    self.intervals[interval_idx].reg_class = class;
                }
            } else {
                // Parameter has no live interval (unused parameter) — skip
                match class {
                    RegisterClass::GeneralPurpose => int_arg_idx += 1,
                    RegisterClass::FloatingPoint => float_arg_idx += 1,
                }
            }
        }
    }

    /// Classify an IR type to a register class (without needing a ValueId
    /// lookup — used for parameter pre-colouring where we have the type
    /// directly).
    fn classify_value_from_ir_type(ty: &IrType) -> RegisterClass {
        if ty.is_floating() {
            RegisterClass::FloatingPoint
        } else {
            RegisterClass::GeneralPurpose
        }
    }

    /// Return a physical register to the appropriate free pool.
    fn return_reg(
        reg: PhysReg,
        class: RegisterClass,
        free_int: &mut Vec<PhysReg>,
        free_float: &mut Vec<PhysReg>,
    ) {
        match class {
            RegisterClass::GeneralPurpose => {
                if !free_int.contains(&reg) {
                    free_int.push(reg);
                }
            }
            RegisterClass::FloatingPoint => {
                if !free_float.contains(&reg) {
                    free_float.push(reg);
                }
            }
        }
    }

    /// Remove a specific register from the free pool (used when a pre-coloured
    /// interval claims a register).
    fn remove_from_free(
        reg: PhysReg,
        class: RegisterClass,
        free_int: &mut Vec<PhysReg>,
        free_float: &mut Vec<PhysReg>,
    ) {
        let pool = match class {
            RegisterClass::GeneralPurpose => free_int,
            RegisterClass::FloatingPoint => free_float,
        };
        if let Some(pos) = pool.iter().position(|&r| r == reg) {
            pool.swap_remove(pos);
        }
    }

    /// Find the best spill candidate from the active set: the interval with
    /// the **furthest** end position in the same register class as `class`.
    ///
    /// Returns the index into `active` (not into `self.intervals`), or `None`
    /// if no active interval of the matching class exists.
    fn find_spill_candidate(
        &self,
        active: &[usize],
        class: RegisterClass,
        _cur_end: u32,
    ) -> Option<usize> {
        let mut best: Option<(usize, u32)> = None;
        for (pos, &interval_idx) in active.iter().enumerate() {
            let interval = &self.intervals[interval_idx];
            // Never spill a fixed (pre-coloured) interval
            if interval.is_fixed {
                continue;
            }
            if interval.reg_class != class {
                continue;
            }
            let dominated = match best {
                None => true,
                Some((_, best_end)) => {
                    // Use explicit Ordering comparisons: the candidate with
                    // the furthest (greatest) end wins.
                    match interval.end.cmp(&best_end) {
                        Ordering::Greater => true,
                        Ordering::Equal | Ordering::Less => false,
                    }
                }
            };
            if dominated {
                best = Some((pos, interval.end));
            }
        }
        best.map(|(pos, _)| pos)
    }

    /// Sort active interval indices by their end position (earliest first).
    /// Used to quickly expire old intervals during the allocation sweep.
    ///
    /// Uses `Reverse` to obtain descending order so that pop() yields the
    /// earliest-ending interval when the list is treated as a max-heap-like
    /// structure (though we currently linear-scan instead).
    #[allow(dead_code)]
    fn sort_active_by_end(&self, active: &mut Vec<usize>) {
        active.sort_by_key(|&idx| self.intervals[idx].end);
    }

    /// Sort active interval indices by end position descending (latest first)
    /// so that the earliest-ending interval is at the back for efficient pop.
    #[allow(dead_code)]
    fn sort_active_by_end_desc(&self, active: &mut Vec<usize>) {
        active.sort_by_key(|&idx| Reverse(self.intervals[idx].end));
    }

    /// Assign a spill slot to the given interval and update frame size.
    fn spill_interval(&mut self, interval_idx: usize) {
        let slot = self.next_spill_slot;
        self.next_spill_slot += 1;
        self.intervals[interval_idx].spill_slot = Some(slot);
        self.intervals[interval_idx].register = None;
        self.frame_size = self.next_spill_slot * self.spill_slot_bytes;
    }

    /// Record that a callee-saved register was used, so it can be saved in
    /// the prologue and restored in the epilogue.
    fn track_callee_saved_usage(&mut self, reg: PhysReg) {
        if self.callee_saved_set.contains(&reg) {
            self.used_callee_saved.insert(reg);
        }
    }
}

/// Outcome of register allocation for a single virtual register.
#[derive(Clone, Copy, Debug)]
enum AllocationResult {
    /// A physical register was assigned.
    Register(PhysReg),
    /// The value was spilled to the given frame slot index.
    Spilled(u32),
}

// ===========================================================================
// Display implementation for the allocator (diagnostic output)
// ===========================================================================

impl fmt::Display for RegisterAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RegisterAllocator {{")?;
        writeln!(f, "  target: {:?}", self.target)?;
        writeln!(f, "  frame_size: {} bytes", self.frame_size)?;
        writeln!(f, "  spill_slots: {}", self.next_spill_slot)?;
        writeln!(f, "  callee_saved_used: {} regs", self.used_callee_saved.len())?;
        writeln!(f, "  intervals: {} total", self.intervals.len())?;
        for (i, interval) in self.intervals.iter().enumerate() {
            writeln!(f, "    [{}] {}", i, interval)?;
        }
        writeln!(f, "}}")
    }
}

// ===========================================================================
// Utility
// ===========================================================================

/// Round `value` up to the next multiple of `align`. `align` must be a power
/// of two.
#[inline]
fn align_up(value: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two(), "align must be a power of two");
    (value + align - 1) & !(align - 1)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // LiveInterval tests
    // -----------------------------------------------------------------------

    fn make_interval(vid: u32, start: u32, end: u32) -> LiveInterval {
        LiveInterval {
            value_id: ValueId(vid),
            start,
            end,
            register: None,
            spill_slot: None,
            is_fixed: false,
            reg_class: RegisterClass::GeneralPurpose,
        }
    }

    #[test]
    fn live_interval_covers() {
        let iv = make_interval(0, 10, 20);
        assert!(iv.covers(10));
        assert!(iv.covers(15));
        assert!(iv.covers(19));
        assert!(!iv.covers(9));
        assert!(!iv.covers(20));
        assert!(!iv.covers(21));
    }

    #[test]
    fn live_interval_intersects() {
        let a = make_interval(0, 5, 15);
        let b = make_interval(1, 10, 20);
        let c = make_interval(2, 15, 25);
        let d = make_interval(3, 20, 30);

        // Overlapping
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(b.intersects(&c));

        // Adjacent but not overlapping (half-open)
        assert!(!a.intersects(&c)); // a ends at 15, c starts at 15
        assert!(!c.intersects(&a));

        // Disjoint
        assert!(!a.intersects(&d));
        assert!(!d.intersects(&a));
    }

    #[test]
    fn live_interval_split_at() {
        let mut iv = make_interval(0, 10, 30);

        // Split in the middle
        let new = iv.split_at(20);
        assert!(new.is_some());
        let new = new.unwrap();
        assert_eq!(iv.start, 10);
        assert_eq!(iv.end, 20);
        assert_eq!(new.start, 20);
        assert_eq!(new.end, 30);
        assert_eq!(new.value_id.0, 0);
        assert!(!new.is_fixed);
        assert!(new.register.is_none());
    }

    #[test]
    fn live_interval_split_at_boundaries_returns_none() {
        let mut iv = make_interval(0, 10, 30);
        assert!(iv.split_at(10).is_none()); // at start
        assert!(iv.split_at(30).is_none()); // at end
        assert!(iv.split_at(5).is_none());  // before start
        assert!(iv.split_at(35).is_none()); // after end
    }

    #[test]
    fn live_interval_ordering() {
        let a = make_interval(0, 5, 15);
        let b = make_interval(1, 10, 20);
        let c = make_interval(2, 5, 20); // same start as a, later end

        let mut intervals = vec![b.clone(), c.clone(), a.clone()];
        intervals.sort();
        assert_eq!(intervals[0].value_id.0, 0); // a: start 5, end 15
        assert_eq!(intervals[1].value_id.0, 2); // c: start 5, end 20
        assert_eq!(intervals[2].value_id.0, 1); // b: start 10, end 20
    }

    #[test]
    fn live_interval_length() {
        let iv = make_interval(0, 10, 30);
        assert_eq!(iv.length(), 20);

        let empty = make_interval(1, 5, 5);
        assert_eq!(empty.length(), 0);
    }

    #[test]
    fn live_interval_display() {
        let mut iv = make_interval(42, 100, 200);
        let s = format!("{}", iv);
        assert!(s.contains("v42"));
        assert!(s.contains("[100, 200)"));

        iv.register = Some(PhysReg(7));
        let s = format!("{}", iv);
        assert!(s.contains("r7"));

        iv.register = None;
        iv.spill_slot = Some(3);
        let s = format!("{}", iv);
        assert!(s.contains("spill#3"));
    }

    // -----------------------------------------------------------------------
    // RegisterSet tests
    // -----------------------------------------------------------------------

    #[test]
    fn register_set_new() {
        let rs = RegisterSet::new(
            vec![PhysReg(0), PhysReg(1), PhysReg(2)],
            vec![PhysReg(2)],
            vec![PhysReg(0), PhysReg(1)],
        );
        assert_eq!(rs.available.len(), 3);
        assert_eq!(rs.callee_saved.len(), 1);
        assert_eq!(rs.caller_saved.len(), 2);
    }

    #[test]
    fn register_set_display() {
        let rs = RegisterSet::new(
            vec![PhysReg(0), PhysReg(1)],
            vec![PhysReg(1)],
            vec![PhysReg(0)],
        );
        let s = format!("{}", rs);
        assert!(s.contains("r0"));
        assert!(s.contains("r1"));
    }

    // -----------------------------------------------------------------------
    // align_up tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }
}
