# BCC Validation Checkpoints

## Overview

BCC employs **seven sequential validation checkpoints** that serve as hard gates governing the
development lifecycle. Each checkpoint validates a specific capability tier of the compiler and
must be fully passed before work on the next tier may begin.

**Checkpoints 1–6 are strictly sequential hard gates.** A failure at any checkpoint immediately
halts all forward progress. The failing checkpoint must be diagnosed, resolved, and re-passed
before any subsequent checkpoint may be attempted.

**Checkpoint 7 is an optional milestone.** It may execute in parallel once Checkpoint 6 has been
fully passed.

### Checkpoint Summary

| Checkpoint | Name | Type | Scope |
|:----------:|------|:----:|-------|
| 1 | Hello World | Hard Gate | All 4 architectures |
| 2 | Language & Preprocessor Correctness | Hard Gate | C11 + GCC extensions |
| 3 | Internal Test Suite | Hard Gate | Full unit + integration tests |
| 4 | Shared Library & DWARF | Hard Gate | ELF dynamic linking + debug info |
| 5 | Security Mitigations | Hard Gate | x86-64 only |
| 6 | Linux Kernel Build & Boot | Hard Gate | RISC-V kernel + QEMU |
| 7 | Stretch Targets | Optional | SQLite, Redis, PostgreSQL, FFmpeg |

### Backend Validation Order

Within every checkpoint that exercises multiple architectures, the **fixed backend validation
order** is:

```
x86-64  →  i686  →  AArch64  →  RISC-V 64
```

x86-64 is always validated first because it executes natively on the development host without
emulation. Subsequent architectures are validated using QEMU user-mode emulation (i686, AArch64,
RISC-V 64) or QEMU system emulation (Checkpoint 6 kernel boot).

---

## Execution Rules

### Sequential Gate Protocol

1. Begin at Checkpoint 1. No checkpoint may be skipped.
2. Execute all tests for the current checkpoint.
3. If **any** test fails, **STOP**. Diagnose and fix the failure.
4. Re-run the **entire** current checkpoint after every fix.
5. Only when the checkpoint achieves a **100% pass rate**, advance to the next checkpoint.
6. If a fix for a later checkpoint introduces a regression in an earlier checkpoint, the earlier
   checkpoint must be re-passed before continuing.

### Wall-Clock Performance Ceiling

All compilation benchmarks are subject to a **5× GCC-equivalent wall-clock ceiling**. If BCC
takes longer than five times the duration GCC requires to compile the same source with equivalent
configuration on the same hardware, the build is considered a performance failure. This ceiling
applies to:

- Linux kernel 6.9 full build (Checkpoint 6)
- Each stretch target build (Checkpoint 7)

### Resource Constraints During Validation

- Worker threads run with a **64 MiB stack** (`std::thread::Builder::new().stack_size(64 * 1024 * 1024)`)
- A **512-depth recursion limit** is enforced in the parser and macro expander
- No external toolchain components (`as`, `ld`, `gcc`, `llvm-mc`, `lld`) may be invoked — the
  standalone backend handles all assembly and linking

---

## Checkpoint 1 — Hello World (Hard Gate)

### Objective

Compile and execute a minimal Hello World program on all four target architectures, verifying
end-to-end pipeline correctness from source input to ELF executable output.

### Test Source

**File:** `tests/fixtures/hello.c`

```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

### Execution Protocol

For each architecture, in the fixed backend validation order:

#### x86-64 (Native Execution)

```bash
./bcc --target=x86-64 -o hello tests/fixtures/hello.c
./hello
```

#### i686 (QEMU User-Mode)

```bash
./bcc --target=i686 -o hello_i686 tests/fixtures/hello.c
qemu-i386 ./hello_i686
```

#### AArch64 (QEMU User-Mode)

```bash
./bcc --target=aarch64 -o hello_aarch64 tests/fixtures/hello.c
qemu-aarch64 ./hello_aarch64
```

#### RISC-V 64 (QEMU User-Mode)

```bash
./bcc --target=riscv64 -o hello_riscv64 tests/fixtures/hello.c
qemu-riscv64 ./hello_riscv64
```

### Pass Criteria

| Criterion | Requirement |
|-----------|-------------|
| stdout | Exactly `Hello, World!\n` (14 bytes including newline) |
| Exit code | 0 |
| Architectures | All four must pass |
| ELF validity | Output is a valid ELF binary (`readelf -h` succeeds) |
| No external tools | BCC must not invoke `as`, `ld`, or any external assembler/linker |

### Failure Response

If any architecture fails:

1. Identify the failing pipeline stage (preprocessing, lexing, parsing, sema, IR, codegen,
   assembly, linking, or runtime)
2. Fix the issue for the failing architecture
3. Re-run **all four architectures** (not just the fixed one)
4. Do not advance to Checkpoint 2 until all four pass

### Test File

- **Integration test:** `tests/checkpoint1_hello_world.rs`
- **Fixture:** `tests/fixtures/hello.c`

---

## Checkpoint 2 — Language and Preprocessor Correctness (Hard Gate)

### Objective

Validate that BCC correctly implements C11 language features, GCC extensions, and preprocessor
behavior including edge cases critical to real-world codebases (especially the Linux kernel).

### Test Matrix

Each test is independently pass/fail. All tests must pass for the checkpoint to succeed.

#### 2.1 PUA Round-Trip (Non-UTF-8 Byte Fidelity)

**File:** `tests/fixtures/pua_roundtrip.c`

**Purpose:** Verify that non-UTF-8 bytes (0x80–0xFF) in string literals survive the entire
compilation pipeline with byte-exact fidelity via PUA encoding (U+E080–U+E0FF).

**Validation:**
```bash
./bcc -o pua_test tests/fixtures/pua_roundtrip.c
objdump -s -j .rodata pua_test | grep -q "80 ff"
```

**Pass criterion:** The `.rodata` section contains the exact bytes `0x80 0xFF` at the expected
offset. No byte is lost, reordered, or substituted during compilation.

#### 2.2 Recursive Macro Termination

**File:** `tests/fixtures/recursive_macro.c`

**Purpose:** Verify that the paint-marker recursion protection correctly handles self-referential
macros without entering an infinite expansion loop.

**Test source:**
```c
#define A A
int x = A;
```

**Pass criterion:** Compilation completes in **less than 5 seconds**. The token `A` in the
initializer is treated as an ordinary identifier (not re-expanded) after being painted during
its own macro expansion. No hang, no stack overflow.

#### 2.3 Statement Expressions

**File:** `tests/fixtures/stmt_expr.c`

**Purpose:** Validate GCC statement expression syntax `({ ... })`, ensuring the value of the
last expression in the block is used as the result.

**Pass criterion:** Compiles without error, executes correctly, produces expected output values.

#### 2.4 typeof / __typeof__

**File:** `tests/fixtures/typeof_test.c`

**Purpose:** Verify that `typeof(expr)` and `__typeof__(expr)` correctly infer types at compile
time, including compound types and expressions with side effects (which must not be evaluated).

**Pass criterion:** Type inference matches expected types; no spurious errors or incorrect
type resolution.

#### 2.5 Designated Initializers

**File:** `tests/fixtures/designated_init.c`

**Purpose:** Validate designated initializer support including out-of-order field designation,
nested designation (`.field.subfield`), array index designation (`[N] = value`), and brace
elision. Unspecified members must be zero-initialized.

**Pass criterion:** All initialized values match expected values at runtime. Zero-initialization
of unspecified members is verifiable.

#### 2.6 Computed Gotos

**File:** `tests/fixtures/computed_goto.c`

**Purpose:** Validate GCC computed goto extension (`void *labels[] = {&&L1, ...}; goto *labels[i];`),
a critical pattern used in the Linux kernel's interpreter dispatch loops.

**Pass criterion:** Indirect jump targets are resolved correctly; control flow visits the
expected sequence of labels.

#### 2.7 Zero-Length Arrays

**File:** `tests/fixtures/zero_length_array.c`

**Purpose:** Verify that zero-length arrays (`int arr[0]`) at the end of structures are
accepted and do not contribute to `sizeof` the containing struct (GCC flexible array extension).

**Pass criterion:** `sizeof(struct)` excludes the zero-length array; access beyond the struct
boundary functions correctly when memory is appropriately allocated.

#### 2.8 GCC Builtins

**File:** `tests/fixtures/builtins.c`

**Purpose:** Validate the compile-time and runtime behavior of all supported GCC builtins,
including but not limited to:

- `__builtin_expect` — branch prediction hint (transparent at -O0)
- `__builtin_unreachable` — unreachable code marker
- `__builtin_constant_p` — compile-time constant check
- `__builtin_offsetof` — struct member offset
- `__builtin_types_compatible_p` — type compatibility check
- `__builtin_choose_expr` — compile-time conditional selection
- `__builtin_clz`, `__builtin_ctz`, `__builtin_popcount` — bit manipulation
- `__builtin_bswap16`, `__builtin_bswap32`, `__builtin_bswap64` — byte swap
- `__builtin_ffs` — find first set bit
- `__builtin_va_start`, `__builtin_va_end`, `__builtin_va_arg`, `__builtin_va_copy` — variadic args
- `__builtin_frame_address`, `__builtin_return_address` — stack introspection
- `__builtin_trap` — abort trap
- `__builtin_assume_aligned` — alignment hint
- `__builtin_add_overflow`, `__builtin_sub_overflow`, `__builtin_mul_overflow` — checked arithmetic

**Pass criterion:** All builtins produce correct compile-time or runtime results.

#### 2.9 _Static_assert

**File:** `tests/fixtures/static_assert.c`

**Purpose:** Verify C11 `_Static_assert(constant-expression, string-literal)` — compilation
succeeds when the expression is true and produces a diagnostic error when false.

**Pass criterion:** Valid assertions compile; invalid assertions produce a clear error message
containing the string literal.

#### 2.10 _Generic

**File:** `tests/fixtures/generic.c`

**Purpose:** Validate C11 `_Generic` type-based selection, ensuring the correct association is
chosen based on the controlling expression's type.

**Pass criterion:** The correct branch is selected for each controlling type.

#### 2.11 Inline Assembly (Basic)

**File:** `tests/fixtures/inline_asm_basic.c`

**Purpose:** Validate basic inline assembly with AT&T syntax, output and input operands,
and clobber lists.

**Pass criterion:** Inline assembly executes and produces correct results.

#### 2.12 Inline Assembly (Constraints)

**File:** `tests/fixtures/inline_asm_constraints.c`

**Purpose:** Validate the full constraint system for inline assembly: register constraints
(`"=r"`, `"+r"`, `"r"`), memory constraints (`"=m"`, `"m"`), immediate constraints (`"i"`, `"n"`),
named operands (`[name] "=r" (var)`), `"memory"` and `"cc"` clobbers, and `asm volatile`
semantics.

**Pass criterion:** All constraint combinations produce correct register allocation and memory
operand behavior.

### Pass Criteria (Aggregate)

| Criterion | Requirement |
|-----------|-------------|
| Individual tests | All 12 test categories must pass |
| Architecture | Primary validation on x86-64; cross-architecture where applicable |
| Time limit | Recursive macro test completes in < 5 seconds |
| Byte fidelity | PUA round-trip preserves exact bytes in `.rodata` |

### Test Files

- **Integration test:** `tests/checkpoint2_language.rs`
- **Fixtures:** `tests/fixtures/pua_roundtrip.c`, `tests/fixtures/recursive_macro.c`,
  `tests/fixtures/stmt_expr.c`, `tests/fixtures/typeof_test.c`,
  `tests/fixtures/designated_init.c`, `tests/fixtures/computed_goto.c`,
  `tests/fixtures/zero_length_array.c`, `tests/fixtures/builtins.c`,
  `tests/fixtures/static_assert.c`, `tests/fixtures/generic.c`,
  `tests/fixtures/inline_asm_basic.c`, `tests/fixtures/inline_asm_constraints.c`

---

## Checkpoint 3 — Internal Test Suite (Hard Gate)

### Objective

Achieve a **100% pass rate** on the complete internal unit test and integration test suite,
confirming that all compiler components function correctly in isolation and in combination.

### Execution

```bash
cargo test --release
```

This command runs:

- **Unit tests** embedded in source modules (`#[cfg(test)]` modules within `src/**/*.rs`)
- **Integration tests** in the `tests/` directory (`tests/checkpoint3_internal.rs` and all
  related test files)

### Pass Criteria

| Criterion | Requirement |
|-----------|-------------|
| Test results | **Zero failures** — every test must pass |
| Test count | All registered tests execute (no skipped or ignored tests without justification) |
| Execution | `cargo test --release` exits with code 0 |

### Regression Gate Rule

This checkpoint has a **special re-run requirement**:

> **Checkpoint 3 MUST be re-run after every feature addition made during the kernel build phase
> (Checkpoint 6 sub-gate iteration).** Any test that previously passed and now fails constitutes
> a **regression** and must be resolved before continuing the kernel build work.

This ensures that features added to support kernel compilation do not break previously validated
compiler behavior.

### Test File

- **Integration test:** `tests/checkpoint3_internal.rs`

---

## Checkpoint 4 — Shared Library and DWARF (Hard Gate)

### Objective

Validate two capabilities:

1. **Shared library generation** — BCC produces well-formed ELF shared objects (`ET_DYN`) with
   all required dynamic linking sections
2. **DWARF debug information** — BCC emits correct DWARF v4 debug sections when `-g` is
   specified and emits **zero** debug sections when `-g` is absent

### Test 4.1 — Shared Library ELF Structure

**Files:** `tests/fixtures/shared_lib/foo.c`, `tests/fixtures/shared_lib/main.c`

**Build process:**
```bash
# Compile shared library
./bcc -fPIC -shared -o libfoo.so tests/fixtures/shared_lib/foo.c

# Compile consumer
./bcc -o main tests/fixtures/shared_lib/main.c -L. -lfoo

# Execute (with library path)
LD_LIBRARY_PATH=. ./main
```

**Required ELF sections in `libfoo.so`:**

| Section | Type | Purpose |
|---------|------|---------|
| `.dynamic` | `SHT_DYNAMIC` | Dynamic linking entries |
| `.dynsym` | `SHT_DYNSYM` | Dynamic symbol table |
| `.dynstr` | `SHT_STRTAB` | Dynamic string table |
| `.rela.dyn` | `SHT_RELA` | Dynamic relocations (GOT entries) |
| `.rela.plt` | `SHT_RELA` | PLT relocations (lazy binding) |
| `.gnu.hash` | `SHT_GNU_HASH` | GNU hash table for fast symbol lookup |
| `.got` | `SHT_PROGBITS` | Global Offset Table |
| `.got.plt` | `SHT_PROGBITS` | GOT entries for PLT |
| `.plt` | `SHT_PROGBITS` | Procedure Linkage Table |

**Validation commands:**
```bash
readelf -d libfoo.so   # Verify .dynamic entries
readelf -s libfoo.so   # Verify .dynsym symbols
readelf -r libfoo.so   # Verify relocations
readelf -S libfoo.so   # Verify all section headers present
```

**Pass criteria:** All listed sections are present and well-formed. The `PT_DYNAMIC` program
header points to `.dynamic`. Exported functions appear in `.dynsym` with `STB_GLOBAL` binding.

### Test 4.2 — DWARF Debug Information (With -g)

**File:** `tests/fixtures/dwarf/debug_test.c`

**Build process:**
```bash
./bcc -g -o debug_test tests/fixtures/dwarf/debug_test.c
```

**Required DWARF sections:**

| Section | Purpose |
|---------|---------|
| `.debug_info` | Compilation unit DIEs, subprogram DIEs, variable DIEs |
| `.debug_abbrev` | Abbreviation table |
| `.debug_line` | Line number program (file/line mapping) |
| `.debug_str` | String table for debug names |

**Validation commands:**
```bash
readelf --debug-dump=info debug_test      # Verify .debug_info
readelf --debug-dump=abbrev debug_test    # Verify .debug_abbrev
readelf --debug-dump=line debug_test      # Verify .debug_line
readelf -S debug_test | grep .debug_str   # Verify .debug_str exists
```

**Pass criteria:**

- All four `.debug_*` sections are present
- `.debug_info` contains a `DW_TAG_compile_unit` DIE with producer and file information
- `.debug_info` contains `DW_TAG_subprogram` DIEs for each function
- `.debug_line` produces a valid line number program that maps to the original source file
  and line numbers
- GDB can resolve source file and line number: `gdb -batch -ex "info line main" debug_test`
  shows correct file:line

### Test 4.3 — No Debug Leakage (Without -g)

**Build process:**
```bash
./bcc -o no_debug tests/fixtures/dwarf/debug_test.c
```

**Validation:**
```bash
readelf -S no_debug | grep -c ".debug_"
```

**Pass criterion:** The output is `0` — **zero** `.debug_*` sections exist in the binary.
This is a strict no-leakage requirement.

### Pass Criteria (Aggregate)

| Criterion | Requirement |
|-----------|-------------|
| Shared library | All dynamic linking sections present and well-formed |
| Dynamic execution | `LD_LIBRARY_PATH=. ./main` produces correct output |
| DWARF with -g | All four `.debug_*` sections present, GDB resolves source/line |
| No -g leakage | Zero `.debug_*` sections without the `-g` flag |

### Test Files

- **Integration test:** `tests/checkpoint4_shared_lib.rs`
- **Fixtures:** `tests/fixtures/shared_lib/foo.c`, `tests/fixtures/shared_lib/main.c`,
  `tests/fixtures/dwarf/debug_test.c`

---

## Checkpoint 5 — Security Mitigations (Hard Gate, x86-64 Only)

### Objective

Validate that BCC generates the three required security mitigations on x86-64 when the
corresponding CLI flags are active:

1. **Retpoline** — Indirect branch protection against Spectre v2
2. **CET/IBT** — Intel Control-flow Enforcement Technology / Indirect Branch Tracking
3. **Stack probe** — Guard page probing for large stack frames

These mitigations apply **only to x86-64**. Other architectures are not tested in this checkpoint.

### Test 5.1 — Retpoline (`-mretpoline`)

**File:** `tests/fixtures/security/retpoline.c`

**Test source (representative):**
```c
void call_indirect(void (*fptr)(void)) {
    (*fptr)();
}
```

**Build and validate:**
```bash
./bcc --target=x86-64 -mretpoline -c -o retpoline.o tests/fixtures/security/retpoline.c
objdump -d retpoline.o
```

**Pass criterion:** The disassembly of `call_indirect` shows a `call` instruction targeting
`__x86_indirect_thunk_rax` (or another `__x86_indirect_thunk_*` variant depending on register
allocation). The function pointer is **not** called directly via `call *%rax` or `jmp *%rax`.

### Test 5.2 — CET/IBT (`-fcf-protection`)

**File:** `tests/fixtures/security/cet.c`

**Build and validate:**
```bash
./bcc --target=x86-64 -fcf-protection -c -o cet.o tests/fixtures/security/cet.c
objdump -d cet.o
```

**Pass criterion:** Every function entry point begins with the `endbr64` instruction
(`0xF3 0x0F 0x1E 0xFA`). This marks the function as a valid indirect branch target under
Intel CET.

### Test 5.3 — Stack Probe (Large Stack Frames)

**File:** `tests/fixtures/security/stack_probe.c`

**Test source (representative):**
```c
void f(void) {
    char buf[8192];
    buf[0] = 1;
}
```

**Build and validate:**
```bash
./bcc --target=x86-64 -c -o stack_probe.o tests/fixtures/security/stack_probe.c
objdump -d stack_probe.o
```

**Pass criterion:** The disassembly of `f` contains a **probe loop** that touches each 4,096-byte
page before the final stack pointer adjustment. The stack pointer is not simply decremented by
8,192 bytes in a single instruction. The probe loop ensures the guard page is hit if the stack
allocation crosses a page boundary.

### Pass Criteria (Aggregate)

| Criterion | Requirement |
|-----------|-------------|
| Retpoline | `call __x86_indirect_thunk_*` present in disassembly |
| CET/IBT | `endbr64` at every function entry |
| Stack probe | Probe loop present for frames > 4,096 bytes |
| Architecture | x86-64 only |

### Test Files

- **Integration test:** `tests/checkpoint5_security.rs`
- **Fixtures:** `tests/fixtures/security/retpoline.c`, `tests/fixtures/security/cet.c`,
  `tests/fixtures/security/stack_probe.c`

---

## Checkpoint 6 — Linux Kernel Build and Boot (Hard Gate — Primary Success)

### Objective

Compile the **Linux kernel 6.9** for the RISC-V architecture using BCC as the C compiler and
successfully boot the resulting `vmlinux` image to userspace in QEMU.

This is the **primary success criterion** for the BCC project. It exercises the full C language
surface, GCC extensions, preprocessor edge cases, inline assembly, and linker correctness
simultaneously.

### Build Process

#### Step 1 — Kernel Configuration

```bash
# Obtain Linux 6.9 source
tar xf linux-6.9.tar.xz
cd linux-6.9

# Generate a minimal RISC-V configuration
make ARCH=riscv defconfig
```

#### Step 2 — Kernel Compilation

```bash
make ARCH=riscv CC=./bcc -j$(nproc)
```

This produces `vmlinux` — a statically-linked ELF executable for RISC-V 64.

#### Sub-Gates

The kernel build is validated incrementally through four compilation sub-gates. Each sub-gate
must produce a valid relocatable object file before the full build is attempted:

| Sub-Gate | File | Significance |
|----------|------|-------------|
| 6a | `init/main.o` | Kernel initialization, startup paths |
| 6b | `kernel/sched/core.o` | Scheduler core, heavy macro and inline asm use |
| 6c | `mm/memory.o` | Memory management, complex pointer arithmetic |
| 6d | `fs/read_write.o` | Filesystem read/write, syscall interface |

Each sub-gate is validated with:
```bash
readelf -h <file>.o   # Verify valid ELF relocatable object
readelf -s <file>.o   # Verify symbol table integrity
```

#### Step 3 — Initramfs Preparation

Create a minimal `/init` program as a static RISC-V 64 binary:

```c
// init.c — minimal init process
#include <unistd.h>
#include <sys/reboot.h>

int main(void) {
    write(1, "USERSPACE_OK\n", 13);
    reboot(0x01234567);  // RB_AUTOBOOT
    return 0;
}
```

```bash
./bcc --target=riscv64 -static -o init init.c
echo init | cpio -o --format=newc | gzip > initramfs.cpio.gz
```

#### Step 4 — QEMU Boot

```bash
qemu-system-riscv64 \
    -machine virt \
    -kernel vmlinux \
    -initrd initramfs.cpio.gz \
    -append "console=ttyS0 rdinit=/init" \
    -nographic \
    -no-reboot \
    -m 256M \
    2>&1 | tee boot.log
```

### Pass Criteria

| Criterion | Requirement |
|-----------|-------------|
| Kernel compilation | `make ARCH=riscv CC=./bcc` completes without error |
| vmlinux validity | `readelf -h vmlinux` shows valid ELF64 RISC-V executable |
| QEMU boot | Boot log contains `USERSPACE_OK` |
| Clean exit | QEMU exits without hang or kernel panic |
| Wall-clock ceiling | Total build time ≤ **5× GCC-equivalent** on same hardware |
| Sub-gates | All four sub-gate objects (`init/main.o`, `kernel/sched/core.o`, `mm/memory.o`, `fs/read_write.o`) are valid |

### Kernel Build Failure Classification

When the kernel build encounters a compilation error, failures are classified in the following
priority order (see [Failure Classification Protocol](#failure-classification-protocol) below):

```
missing GCC extension → missing builtin → inline asm constraint
    → preprocessor issue → codegen bug
```

Each failure category has a distinct resolution path. Features added during this phase must
trigger a **Checkpoint 3 re-run** to verify no regressions.

### Test File

- **Integration test:** `tests/checkpoint6_kernel.rs`

---

## Checkpoint 7 — Stretch Targets (Optional Milestone)

### Objective

Compile four large, real-world open-source C projects using BCC, validating that the compiler
handles diverse C codebases beyond the Linux kernel.

### Execution Rules

- Checkpoint 7 may **only** begin after Checkpoint 6 has fully passed
- Checkpoint 7 may execute **in parallel** across its four sub-targets
- Checkpoint 7 failure does **not** block the project — it is an optional quality milestone

### Targets

#### 7.1 — SQLite

```bash
./bcc -o sqlite3 shell.c sqlite3.c -lpthread -ldl
./sqlite3 :memory: "SELECT 1+1;"
```

**Pass criterion:** Output is `2`. SQLite's amalgamation (200K+ lines) compiles and the
query engine functions correctly.

#### 7.2 — Redis

```bash
make CC=./bcc
./src/redis-server --port 6380 &
./src/redis-cli -p 6380 PING
```

**Pass criterion:** Response is `PONG`. Redis server starts and responds to commands.

#### 7.3 — PostgreSQL

```bash
./configure CC=./bcc
make
```

**Pass criterion:** Build completes without error. `initdb` and `pg_ctl start` succeed.

#### 7.4 — FFmpeg

```bash
./configure --cc=./bcc --disable-x86asm
make
./ffmpeg -version
```

**Pass criterion:** Build completes; `ffmpeg -version` prints version information.

### Pass Criteria (Aggregate)

| Criterion | Requirement |
|-----------|-------------|
| Compilation | Each target compiles without error |
| Execution | Basic functionality test passes for each target |
| Wall-clock ceiling | Each build ≤ **5× GCC-equivalent** time |
| Prerequisite | Checkpoint 6 must have already passed |

### Test File

- **Integration test:** `tests/checkpoint7_stretch.rs`

---

## Regression Policy

### Definition

> A **regression** is any test that **passed before a change** and **fails after the change**.

### Resolution Requirements

1. **Mandatory resolution.** Regressions must be fixed before any forward progress on the
   current or subsequent checkpoints.
2. **Re-run scope.** After fixing a regression, re-run the **entire checkpoint** that contained
   the regression, plus all earlier checkpoints that might be affected.
3. **Checkpoint 3 re-run mandate.** Checkpoint 3 (Internal Test Suite) **MUST** be re-run after
   every feature addition made during the kernel build phase (Checkpoint 6 sub-gate iteration).
   This is a non-negotiable requirement to prevent feature additions from silently breaking
   previously validated behavior.

### Regression Prevention Workflow

```
1. Identify needed feature/fix for current checkpoint
2. Implement the change
3. Re-run Checkpoint 3 (cargo test --release)
   ├── All tests pass → Continue with current checkpoint
   └── Any test fails → STOP, fix regression, go to step 3
4. Re-run current checkpoint
   ├── All tests pass → Advance (or continue sub-gates)
   └── Any test fails → STOP, fix issue, go to step 3
```

---

## Failure Classification Protocol

When a compilation failure occurs (particularly during Checkpoint 6 — Kernel Build), failures
are diagnosed using the following priority-ordered classification:

### Classification Hierarchy

```
Priority 1: Missing GCC Extension
    ↓ (if not applicable)
Priority 2: Missing Builtin
    ↓ (if not applicable)
Priority 3: Inline Assembly Constraint Issue
    ↓ (if not applicable)
Priority 4: Preprocessor Issue
    ↓ (if not applicable)
Priority 5: Code Generation Bug
```

### Classification Details

| Priority | Category | Symptoms | Resolution Path |
|:--------:|----------|----------|----------------|
| 1 | **Missing GCC Extension** | Unknown attribute, unrecognized syntax (e.g., `__attribute__((xyz))`, case range, statement expression) | Implement the extension in `src/frontend/parser/gcc_extensions.rs` and `src/frontend/sema/attribute_handler.rs`; update `docs/gcc_extensions.md` |
| 2 | **Missing Builtin** | `implicit declaration of function '__builtin_*'`, incorrect builtin evaluation | Implement in `src/frontend/sema/builtin_eval.rs` (compile-time) or IR lowering (runtime); update `docs/gcc_extensions.md` |
| 3 | **Inline ASM Constraint** | `unsupported constraint`, incorrect operand binding, clobber handling error | Fix in `src/frontend/parser/inline_asm.rs` (parsing) or `src/ir/lowering/asm_lowering.rs` (lowering) |
| 4 | **Preprocessor Issue** | Incorrect macro expansion, missing predefined macro, `#if` evaluation error, include resolution failure | Fix in `src/frontend/preprocessor/` modules |
| 5 | **Code Generation Bug** | Incorrect machine code, wrong relocation, ABI violation, miscompiled operation | Fix in `src/backend/*/codegen.rs`, `src/backend/*/abi.rs`, or `src/backend/*/assembler/` |

### Extension Discovery Tracking

All GCC extensions discovered during the kernel build phase that are **not** in the initial
extension list (documented in `docs/gcc_extensions.md`) are tracked as amendments. Each
amendment records:

- Extension name and syntax
- Source file where it was first encountered
- Classification category
- Implementation status
- Implementing source file(s)

### Critical Rule

> The compiler **MUST NOT** silently miscompile unknown extensions. Every unrecognized construct
> must either be correctly implemented or produce a **clear diagnostic error** identifying the
> unsupported feature. Silent miscompilation is a critical defect.

---

## Quick Reference

### Checkpoint Progression Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Checkpoint 1 │────▶│ Checkpoint 2 │────▶│ Checkpoint 3 │
│ Hello World  │ OK  │  Language &  │ OK  │   Internal   │
│  (4 archs)   │     │ Preprocessor │     │  Test Suite  │
└──────────────┘     └──────────────┘     └──────────────┘
       │ FAIL              │ FAIL              │ FAIL
       ▼                   ▼                   ▼
    ┌──────┐           ┌──────┐           ┌──────┐
    │ HALT │           │ HALT │           │ HALT │
    └──────┘           └──────┘           └──────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Checkpoint 4 │────▶│ Checkpoint 5 │────▶│ Checkpoint 6 │
│ Shared Lib & │ OK  │   Security   │ OK  │ Kernel Build │
│    DWARF     │     │  (x86-64)    │     │  & Boot      │
└──────────────┘     └──────────────┘     └──────────────┘
       │ FAIL              │ FAIL              │ FAIL
       ▼                   ▼                   ▼
    ┌──────┐           ┌──────┐           ┌──────┐
    │ HALT │           │ HALT │           │ HALT │
    └──────┘           └──────┘           └──────┘

                                          │ OK (Checkpoint 6 passed)
                                          ▼
                                   ┌──────────────┐
                                   │ Checkpoint 7 │
                                   │   Stretch    │  (Optional)
                                   │   Targets    │
                                   └──────────────┘
```

### Command Quick Reference

| Checkpoint | Primary Command |
|:----------:|----------------|
| 1 | `./bcc -o hello hello.c && ./hello` |
| 2 | `cargo test --release checkpoint2` |
| 3 | `cargo test --release` |
| 4 | `./bcc -fPIC -shared -o libfoo.so foo.c && readelf -d libfoo.so` |
| 5 | `./bcc -mretpoline -c retpoline.c && objdump -d retpoline.o` |
| 6 | `make ARCH=riscv CC=./bcc && qemu-system-riscv64 -kernel vmlinux ...` |
| 7 | `make CC=./bcc` (per stretch target) |

### Backend Validation Order

```
x86-64 (native) → i686 (qemu-i386) → AArch64 (qemu-aarch64) → RISC-V 64 (qemu-riscv64)
```

### Key Constraints

| Constraint | Value |
|-----------|-------|
| Worker thread stack | 64 MiB |
| Recursion depth limit | 512 |
| External tools allowed | None (standalone backend) |
| Wall-clock ceiling | 5× GCC-equivalent |
| Debug info scope | `-O0` only (DWARF v4) |
| Platform | Linux-only, ELF-only |
