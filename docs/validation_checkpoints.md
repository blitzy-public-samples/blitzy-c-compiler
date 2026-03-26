# BCC Validation Checkpoints

BCC (Blitzy's C Compiler) employs a **sequential checkpoint validation strategy** to ensure correctness, completeness, and production-readiness at every stage of compiler development. Seven checkpoints are defined, each exercising progressively broader surface area of the compilation pipeline — from a trivial "Hello, World!" program through full Linux kernel 6.9 compilation and boot.

**Key principles:**

- **Checkpoints 1–6 are strictly sequential hard gates.** Failure at any checkpoint halts all forward progress. A checkpoint must be diagnosed, resolved, and re-passed before the next checkpoint may be attempted.
- **Checkpoint 7 (stretch targets) is optional.** It may execute in parallel after Checkpoint 6 passes and serves as a milestone indicator, not a hard gate.
- **Backend validation order is fixed:** x86-64 → i686 → AArch64 → RISC-V 64. Each checkpoint that exercises architecture-specific behavior must be validated in this order.

---

## Checkpoint Protocol

### Sequential Hard Gate Enforcement

The following rules govern checkpoint execution and are non-negotiable:

1. **Strict Numerical Order:** Checkpoints MUST be executed in strict numerical order: 1, 2, 3, 4, 5, 6. No checkpoint may be attempted until all preceding checkpoints pass.
2. **Halt on Failure:** A failure at ANY checkpoint halts all forward progress. No subsequent checkpoint may be attempted.
3. **Diagnose, Resolve, Re-Pass:** Checkpoint failures MUST be diagnosed, the root cause resolved, and the checkpoint re-passed before proceeding to the next gate.
4. **Regression Definition:** Any test that passed before the current change and fails after is a **regression**. Regression resolution is mandatory before proceeding — no exceptions.
5. **Checkpoint 3 Regression Gate:** Checkpoint 3 (internal test suite) MUST be re-run after any feature addition during the kernel build phase (Checkpoint 6 iteration) to confirm no regressions were introduced. This applies to every GCC extension, builtin, or codegen fix added during kernel compilation work.
6. **Checkpoint 7 Is a Milestone:** Checkpoint 7 (stretch targets) is not a hard gate. It may be skipped or executed in parallel after Checkpoint 6 passes.

### Wall-Clock Performance Ceiling

- **Linux kernel 6.9** full build time MUST NOT exceed **5× GCC-equivalent** on the same source with equivalent configuration on the same hardware.
- Each **stretch target** (SQLite, Redis, PostgreSQL, FFmpeg) build MUST NOT exceed **5× equivalent GCC build time**.
- If the ceiling is exceeded, codegen performance is deemed insufficient for practical use and optimization is required before the checkpoint can pass.

### Backend Validation Order

All checkpoints that exercise architecture-specific behavior MUST validate backends in the following fixed order:

| Order | Architecture | Role |
|-------|-------------|------|
| 1 | **x86-64** | Primary development and testing platform |
| 2 | **i686** | 32-bit x86 compatibility validation |
| 3 | **AArch64** | ARM 64-bit cross-compilation validation |
| 4 | **RISC-V 64** | Kernel boot target architecture |

**Rationale:** x86-64 is validated first because it is the native host architecture and the most exercised during development. RISC-V 64 is validated last because it is the Linux kernel boot target — issues discovered in earlier architectures are likely to surface and be fixed before reaching the kernel build checkpoint.

---

## Checkpoint 1: Hello World

### Objective

Compile and execute a trivial C program on **all four target architectures**. This checkpoint validates the entire end-to-end compilation pipeline — from preprocessing through code generation, assembler, and linker — producing a working ELF executable.

### Test Source

**File:** `tests/fixtures/hello.c`

```c
#include <stdio.h>
int main(void) { printf("Hello, World!\n"); return 0; }
```

### Execution

For each target architecture (x86-64, i686, AArch64, RISC-V 64):

```bash
# Compile
./bcc --target=<arch> -o hello hello.c

# Execute (native for x86-64, qemu-user for cross-arch)
./hello                        # x86-64 (native)
qemu-i386 ./hello              # i686
qemu-aarch64 ./hello           # AArch64
qemu-riscv64 ./hello           # RISC-V 64
```

### Pass Criteria

For **each** of the four architectures, ALL of the following must hold:

- `./bcc --target=<arch> -o hello hello.c` exits with code **0** (no compilation errors)
- Executing the `hello` binary (native or via `qemu-user` for cross-architectures) produces exactly:
  ```
  Hello, World!
  ```
  on stdout and exits with code **0**
- The output binary is valid ELF: `readelf -h hello` shows:
  - Correct `e_machine` value for the target architecture (`EM_X86_64`, `EM_386`, `EM_AARCH64`, `EM_RISCV`)
  - ELF type `ET_EXEC` (static executable)

### Fail Criteria

- Any compilation error (non-zero exit from `bcc`)
- Non-zero exit code from the compiled binary
- Incorrect stdout output (missing newline, wrong text, extra output)
- Invalid or corrupt ELF header
- Wrong `e_machine` value for the target architecture

### Test File

`tests/checkpoint1_hello_world.rs`

---

## Checkpoint 2: Language and Preprocessor Correctness

### Objective

Validate C11 language features, GCC extensions, and preprocessor edge cases. This checkpoint exercises the preprocessor's paint-marker recursion protection, PUA encoding for non-UTF-8 byte round-tripping, statement expressions, typeof, designated initializers, inline assembly, computed gotos, GCC builtins, and C11 features (`_Static_assert`, `_Generic`).

### Test Sources

All test fixtures in `tests/fixtures/`:

| Fixture File | Feature Tested | Key Validation |
|-------------|----------------|----------------|
| `pua_roundtrip.c` | PUA encoding fidelity | `\x80\xFF` in string literal → exact bytes `80 ff` in `.rodata` |
| `recursive_macro.c` | Paint-marker recursion protection | `#define A A` with `int x = A;` terminates in <5 seconds |
| `stmt_expr.c` | GCC statement expressions | `({ int x = 1; x + 2; })` evaluates to `3` |
| `typeof_test.c` | `typeof` / `__typeof__` | Yields correct type for arbitrary expressions |
| `designated_init.c` | Designated initializers | Out-of-order and nested field/array designators |
| `inline_asm_basic.c` | Basic inline assembly | Register constraints, clobber lists |
| `inline_asm_constraints.c` | Complex inline assembly | Named operands, multiple constraints, `asm goto` |
| `computed_goto.c` | Computed goto (`goto *ptr`) | Dispatch table with label addresses |
| `zero_length_array.c` | Zero-length arrays | Flexible array member at end of struct |
| `builtins.c` | GCC builtins | `__builtin_expect`, `__builtin_clz`, `__builtin_bswap*`, etc. |
| `static_assert.c` | `_Static_assert` | Compile-time assertion checks |
| `generic.c` | `_Generic` selection | Type-based expression selection |

### Pass Criteria

- **All** test source files compile without errors on all four architectures
- **All** runtime tests produce expected output when executed
- **PUA round-trip** produces **byte-exact** output: compiling `pua_roundtrip.c` and inspecting `.rodata` with `objdump -s` confirms exact bytes `80 ff` are present at the expected location
- **Recursive macro** test completes within a **5-second timeout** — the paint-marker system must suppress re-expansion of self-referential macros without entering an infinite loop
- **Statement expression** evaluation yields the correct value (`3`)
- **Designated initializer** output matches expected field values after out-of-order initialization
- **Inline assembly** constraints are respected and register allocation does not clobber operands
- **Computed goto** dispatch table jumps to correct labels
- **Builtins** produce correct results for all tested `__builtin_*` functions
- **`_Static_assert`** passes for true conditions and produces a diagnostic for false conditions
- **`_Generic`** selects the correct branch based on the controlling expression's type

### Fail Criteria

- Any compilation error on any architecture
- Runtime output mismatch for any fixture
- Byte-inexact output for PUA round-trip (even a single byte difference is a failure)
- Timeout (>5 seconds) on recursive macro test
- Incorrect type deduction for `typeof`
- Silent miscompilation of any GCC extension (hard failure — the compiler must never silently miscompile)

### Test File

`tests/checkpoint2_language.rs`

---

## Checkpoint 3: Internal Test Suite

### Objective

Verify that **all** internal unit tests and integration tests pass at 100% — zero failures, zero panics. This checkpoint is the comprehensive internal quality gate.

### Execution

```bash
cargo test --release
```

### Pass Criteria

- **Zero** test failures across the entire test suite
- **Zero** panics or unexpected aborts
- All tests complete within reasonable time bounds (no hangs or infinite loops)

### Fail Criteria

- Any single test failure
- Any panic or unexpected abort
- Any test timeout or hang

### Regression Gate

This checkpoint has a special **regression gate** role throughout the development lifecycle:

- **After any feature addition** during the kernel build phase (Checkpoint 6 iteration), Checkpoint 3 MUST be re-run to confirm no regressions
- **Regression definition:** Any test that passed before the current change and fails after is a regression
- **Resolution is mandatory:** Regressions must be fixed before proceeding with further kernel build work
- **Scope:** This includes additions of GCC extensions, new builtins, inline assembly constraint support, preprocessor fixes, and codegen bug fixes discovered during kernel compilation

### Re-Run Protocol

```
1. Make changes to fix kernel build issue
2. Run: cargo test --release
3. If ALL tests pass → proceed with kernel build
4. If ANY test fails that previously passed → STOP, fix regression, go to step 2
```

### Test File

`tests/checkpoint3_internal.rs`

---

## Checkpoint 4: Shared Library and DWARF Debug Information

### Objective

Validate two critical capabilities:

1. **PIC code generation and shared library linking** — produce valid `ET_DYN` ELF shared objects with correct GOT/PLT infrastructure
2. **DWARF v4 debug information** — emit debug sections when `-g` is specified and ensure zero debug leakage without `-g`

### Test Sources

- `tests/fixtures/shared_lib/foo.c` — Exported functions for the shared library
- `tests/fixtures/shared_lib/main.c` — Consumer that dynamically links against the shared library
- `tests/fixtures/dwarf/debug_test.c` — Source with functions and local variables for DWARF validation

### Shared Library — Pass Criteria

1. **Compilation to shared object:**
   ```bash
   ./bcc -fPIC -shared -o libfoo.so foo.c
   ```
   - Exit code **0**
   - Output file `libfoo.so` is a valid ELF with type `ET_DYN`

2. **ELF structure validation:**
   ```bash
   readelf -d libfoo.so
   ```
   - Shows `.dynamic` section with required dynamic tags
   - Shows `.dynsym` dynamic symbol table
   - Shows `.rela.dyn` and/or `.rela.plt` relocation sections

   ```bash
   readelf -S libfoo.so
   ```
   - Shows `.got` (Global Offset Table) section
   - Shows `.plt` (Procedure Linkage Table) section

3. **Linking and execution:**
   ```bash
   ./bcc -o main main.c -L. -lfoo
   LD_LIBRARY_PATH=. ./main
   ```
   - Compilation succeeds (exit code **0**)
   - Execution produces correct output
   - Dynamic linker resolves symbols from `libfoo.so` at runtime

### DWARF Debug Information — Pass Criteria

1. **Debug compilation:**
   ```bash
   ./bcc -g -O0 -o debug_test debug_test.c
   ```
   - Exit code **0**

2. **DWARF section presence:**
   ```bash
   readelf --debug-dump=info debug_test
   ```
   - `.debug_info` section is present and contains:
     - `DW_TAG_compile_unit` — with producer string, language (C), source file name
     - `DW_TAG_subprogram` — for each function, with name, low PC, high PC
     - `DW_TAG_variable` — for local variables, with name, type, and location

   ```bash
   readelf --debug-dump=line debug_test
   ```
   - `.debug_line` section is present with:
     - File table referencing the source file
     - Line number program mapping addresses to source lines

3. **Zero debug leakage:**
   ```bash
   ./bcc -O0 -o no_debug debug_test.c   # Note: NO -g flag
   readelf -S no_debug | grep debug
   ```
   - **Must produce no output** — the binary compiled without `-g` MUST NOT contain any `.debug_info`, `.debug_abbrev`, `.debug_line`, or `.debug_str` sections

### Fail Criteria

- Missing or malformed ELF sections in the shared library (`.dynamic`, `.dynsym`, `.got`, `.plt`)
- Dynamic linking failure at runtime
- Missing DWARF sections when `-g` is specified
- Invalid DWARF encoding (malformed DIEs, incorrect abbreviation references)
- **Debug leakage:** Any `.debug_*` section present in a binary compiled without `-g`

### Test File

`tests/checkpoint4_shared_lib.rs`

---

## Checkpoint 5: Security Mitigations (x86-64)

### Objective

Validate three security hardening features on the **x86-64** architecture:

1. **Retpoline** — Indirect call/jump mitigation against Spectre v2
2. **CET/IBT** — Intel Control-flow Enforcement Technology with Indirect Branch Tracking
3. **Stack Probe** — Guard page probing for large stack frames (>4096 bytes)

> **Note:** Security mitigations are validated on x86-64 only. They are explicitly out of scope for i686, AArch64, and RISC-V 64.

### Test Sources

- `tests/fixtures/security/retpoline.c` — Contains a function pointer indirect call (`(*fptr)()`)
- `tests/fixtures/security/cet.c` — Contains indirect branch targets for CET/IBT validation
- `tests/fixtures/security/stack_probe.c` — Contains `void f(void) { char buf[8192]; buf[0] = 1; }`

### Retpoline — Pass Criteria

```bash
./bcc -mretpoline -o retpoline retpoline.c
objdump -d retpoline
```

- The disassembly shows call instructions targeting `__x86_indirect_thunk_rax` (or other `__x86_indirect_thunk_*` variants), **NOT** the function pointer directly
- **No unguarded indirect calls** remain in the disassembly — every indirect call/jump through a register must route through a retpoline thunk
- The retpoline thunk itself is present in the binary and implements the speculative execution barrier pattern (LFENCE-based or equivalent)

### CET/IBT — Pass Criteria

```bash
./bcc -fcf-protection -o cet cet.c
objdump -d cet
```

- `endbr64` instruction appears at:
  - **Every function entry point** — the first instruction of each function is `endbr64`
  - **Indirect branch targets** — any location that may be reached via an indirect call or jump

### Stack Probe — Pass Criteria

```bash
./bcc -o stack_probe stack_probe.c
objdump -d stack_probe
```

- The disassembly of function `f` shows a **probe loop BEFORE the stack pointer adjustment**
- The probe loop touches each page (4096-byte intervals) between the current stack pointer and the target frame size
- For `char buf[8192]`: at least two page-touching probes must be visible (8192 / 4096 = 2 pages)
- The stack pointer is NOT adjusted by the full frame size in a single `sub rsp, N` instruction without prior probing

### Fail Criteria

- **Retpoline:** Any unguarded indirect call in the disassembly; missing `__x86_indirect_thunk_*` symbols
- **CET/IBT:** Missing `endbr64` at function entries or indirect branch targets
- **Stack Probe:** Stack pointer adjusted by more than 4096 bytes in a single instruction without a preceding probe loop; missing page-touching stores

### Test File

`tests/checkpoint5_security.rs`

---

## Checkpoint 6: Linux Kernel 6.9 Build and QEMU Boot

### Objective

The ultimate validation target — compile the **Linux kernel 6.9** with RISC-V configuration using BCC as the C compiler, and boot the resulting kernel image to userspace in QEMU. This exercises the full C language surface, GCC extensions, preprocessor edge cases, inline assembly, and linker correctness simultaneously.

### Build Process

```bash
# Configure the kernel for RISC-V with default configuration
make ARCH=riscv CC=./bcc defconfig

# Build the kernel using BCC
make ARCH=riscv CC=./bcc -j$(nproc)
```

The build must complete **without compilation errors** and produce a valid `vmlinux` ELF binary.

### Sub-Gates (Iterative Compilation)

The kernel build is approached iteratively through sub-gates of increasing complexity. Each sub-gate must succeed before proceeding:

| Order | Target Object | Subsystem | Key Challenges |
|-------|--------------|-----------|----------------|
| 1 | `init/main.o` | Kernel initialization | Basic C features, GCC attributes, builtins |
| 2 | `kernel/sched/core.o` | Scheduler core | Complex macros, inline assembly, per-CPU variables |
| 3 | `mm/memory.o` | Memory management | Pointer arithmetic, bitfield operations, memory barriers |
| 4 | `fs/read_write.o` | Filesystem I/O | Function pointers, struct operations, VFS abstractions |
| 5 | Full `vmlinux` link | Complete kernel | All objects linked, all symbols resolved, ELF output valid |

### Boot Validation

After a successful kernel build, validate that the kernel boots to userspace:

1. **Create minimal `/init`:**
   - A statically-linked RISC-V binary that prints `USERSPACE_OK\n` to stdout and calls `reboot()`
   - This binary is compiled separately (may use GCC or a prior validated BCC build)

2. **Pack initramfs:**
   ```bash
   echo init | cpio -o -H newc > initramfs.cpio
   ```

3. **Launch QEMU:**
   ```bash
   qemu-system-riscv64 \
     -machine virt \
     -kernel vmlinux \
     -initrd initramfs.cpio \
     -append "console=ttyS0" \
     -nographic
   ```

4. **Verify boot:**
   - QEMU stdout contains the string `USERSPACE_OK`
   - QEMU exits cleanly after `reboot()` is called (or a timeout mechanism terminates it)

### Wall-Clock Performance Ceiling

- Total kernel build time MUST NOT exceed **5× GCC-equivalent** on the same source with equivalent configuration on the same hardware
- Measurement: `time make ARCH=riscv CC=./bcc -j$(nproc)` vs `time make ARCH=riscv CC=riscv64-linux-gnu-gcc -j$(nproc)`
- If the ceiling is exceeded, codegen performance optimization is required

### Failure Classification Protocol

When a kernel build failure is encountered, classify the failure root cause in this order:

| Priority | Classification | Example | Action |
|----------|---------------|---------|--------|
| 1 | Missing GCC extension | Unknown `__attribute__((cleanup(...)))` | Implement the extension |
| 2 | Missing builtin | `__builtin_expect_with_probability` not recognized | Implement the builtin |
| 3 | Inline asm constraint | Unsupported constraint letter or modifier | Add constraint support |
| 4 | Preprocessor issue | Macro expansion divergence, `#pragma` handling | Fix preprocessor behavior |
| 5 | Codegen bug | Incorrect instruction selection, register allocation failure | Debug and fix codegen |

**After every fix:**

1. Re-run **Checkpoint 3** (regression gate) — `cargo test --release`
2. Verify no regressions were introduced
3. Retry the kernel build from the point of failure

### Test File

`tests/checkpoint6_kernel.rs`

---

## Checkpoint 7: Stretch Targets (Optional)

### Objective

Compile and validate additional real-world C projects to demonstrate BCC's breadth of compatibility beyond the Linux kernel. This checkpoint is a **milestone target**, not a hard gate — failure here does not block the project.

### Target Projects

| Project | Category | Key Challenges |
|---------|----------|----------------|
| **SQLite** | Embedded database | Single-file amalgamation, extensive preprocessor use, portable C |
| **Redis** | In-memory data store | Event loop, networking, memory allocation |
| **PostgreSQL** | Relational database | Complex build system, platform-specific code, large codebase |
| **FFmpeg** | Multimedia framework | SIMD intrinsics, inline assembly, complex type usage |

### Pass Criteria

For each stretch target project:

1. The project compiles without errors using BCC as the C compiler
2. Basic functionality tests pass for each compiled project:
   - **SQLite:** Create database, insert/query/delete rows
   - **Redis:** Start server, SET/GET key-value pairs
   - **PostgreSQL:** initdb, createdb, simple SQL queries
   - **FFmpeg:** Encode/decode a sample media file
3. Build time does not exceed **5× GCC-equivalent** on same hardware

### Status

- **Optional milestone** — may execute in parallel after Checkpoint 6 passes
- Not a hard gate — failure does not block kernel boot validation
- Results are tracked for coverage reporting but do not affect the project's primary success criteria

### Test File

`tests/checkpoint7_stretch.rs`

---

## Backend Validation Order

### Fixed Order

All multi-architecture validation MUST follow this fixed order:

```
x86-64  →  i686  →  AArch64  →  RISC-V 64
```

### Per-Architecture Details

| # | Architecture | `--target` Flag | `e_machine` | QEMU Binary | ABI |
|---|-------------|----------------|-------------|-------------|-----|
| 1 | x86-64 | `--target=x86-64` | `EM_X86_64` | (native) | System V AMD64 |
| 2 | i686 | `--target=i686` | `EM_386` | `qemu-i386` | cdecl / System V i386 |
| 3 | AArch64 | `--target=aarch64` | `EM_AARCH64` | `qemu-aarch64` | AAPCS64 |
| 4 | RISC-V 64 | `--target=riscv64` | `EM_RISCV` | `qemu-riscv64` | LP64D |

### Rationale

1. **x86-64 first:** This is the native host architecture for the development environment. Bugs are easiest to diagnose here — native execution (no QEMU), native debugging (GDB without remote protocol), and direct `objdump`/`readelf` inspection. Most compiler development and testing naturally happens on x86-64.

2. **i686 second:** Shares the x86 instruction set family with x86-64 but exercises 32-bit code paths — 8 GPRs instead of 16, no REX prefix, stack-based parameter passing. Validates that the 32-bit code generation works correctly before moving to entirely different ISAs.

3. **AArch64 third:** First non-x86 architecture. Fixed-width 32-bit instruction encoding, register-rich ISA (31 GPRs), different ABI (AAPCS64). Validates that the backend abstraction (`ArchCodegen` trait) correctly separates architecture-specific concerns.

4. **RISC-V 64 last:** The kernel boot target architecture. By the time RISC-V 64 is validated, issues common to cross-compilation (ELF structure, relocation handling, ABI correctness) have already been resolved on AArch64. RISC-V 64 validation culminates in Checkpoint 6 (kernel build and boot).

---

## Checkpoint Summary Matrix

| Checkpoint | Gate Type | Scope | Key Validation | Test File |
|-----------|-----------|-------|----------------|-----------|
| 1 | Hard gate | All 4 architectures | Hello World compilation and execution | `tests/checkpoint1_hello_world.rs` |
| 2 | Hard gate | All 4 architectures | C11 features, GCC extensions, preprocessor | `tests/checkpoint2_language.rs` |
| 3 | Hard gate | Internal | 100% unit/integration test pass rate | `tests/checkpoint3_internal.rs` |
| 4 | Hard gate | All 4 architectures | Shared libraries (ET_DYN) + DWARF v4 | `tests/checkpoint4_shared_lib.rs` |
| 5 | Hard gate | x86-64 only | Retpoline, CET/IBT, stack probe | `tests/checkpoint5_security.rs` |
| 6 | Hard gate | RISC-V 64 | Linux kernel 6.9 build + QEMU boot | `tests/checkpoint6_kernel.rs` |
| 7 | Milestone | All 4 architectures | SQLite, Redis, PostgreSQL, FFmpeg | `tests/checkpoint7_stretch.rs` |

---

## Quick Reference: Checkpoint Execution Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                    CHECKPOINT EXECUTION FLOW                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    PASS    ┌──────────────┐    PASS          │
│  │ Checkpoint 1 │──────────→│ Checkpoint 2 │──────────→ ...   │
│  │ Hello World  │           │ Language/PP  │                   │
│  └──────┬───────┘           └──────┬───────┘                   │
│         │ FAIL                     │ FAIL                      │
│         ▼                          ▼                           │
│    ┌─────────┐               ┌─────────┐                      │
│    │ DIAGNOSE│               │ DIAGNOSE│                      │
│    │  & FIX  │               │  & FIX  │                      │
│    └────┬────┘               └────┬────┘                      │
│         │                         │                            │
│         └──── RE-RUN ◄────────────┘                            │
│                                                                │
│  ... ──→ Checkpoint 3 ──→ Checkpoint 4 ──→ Checkpoint 5       │
│              │                                                 │
│              │  (Regression gate: re-run after                 │
│              │   any change during Checkpoint 6)               │
│              ▼                                                 │
│         Checkpoint 6 ──PASS──→ Checkpoint 7 (optional)        │
│         Kernel Boot            Stretch Targets                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Order of Operations

1. Pass Checkpoint 1 (Hello World) on all architectures in order: x86-64 → i686 → AArch64 → RISC-V 64
2. Pass Checkpoint 2 (Language/Preprocessor) on all architectures
3. Pass Checkpoint 3 (Internal Test Suite) — `cargo test --release` at 100%
4. Pass Checkpoint 4 (Shared Library + DWARF) on all architectures
5. Pass Checkpoint 5 (Security Mitigations) on x86-64
6. Pass Checkpoint 6 (Kernel Build + Boot) on RISC-V 64
   - After each fix during kernel build iteration: re-run Checkpoint 3
7. (Optional) Attempt Checkpoint 7 (Stretch Targets)
