# blitzy-c-compiler

**BCC (Blitzy's C Compiler)** — A complete, self-contained, zero-external-dependency C compilation
toolchain implemented in Rust (2021 Edition) that cross-compiles C source code into native Linux
ELF executables and shared objects for four target architectures.

[![BCC CI](https://github.com/blitzy-public-samples/blitzy-c-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/blitzy-public-samples/blitzy-c-compiler/actions/workflows/ci.yml)
[![Checkpoints](https://github.com/blitzy-public-samples/blitzy-c-compiler/actions/workflows/checkpoints.yml/badge.svg)](https://github.com/blitzy-public-samples/blitzy-c-compiler/actions/workflows/checkpoints.yml)
![Rust Edition](https://img.shields.io/badge/Rust-2021_Edition-orange)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Linux_ELF-blue)
![Architectures](https://img.shields.io/badge/Targets-x86--64_%7C_i686_%7C_AArch64_%7C_RISC--V_64-purple)

## Overview

BCC is a from-scratch C11 compiler with comprehensive GCC extension support, designed to compile
real-world C codebases — including the Linux kernel 6.9 — without depending on any external Rust
crates or invoking any external toolchain components. Every capability — preprocessing, lexing,
parsing, semantic analysis, IR construction, optimization, code generation, assembly, and linking —
is implemented internally using only the Rust standard library.

### Target Architectures

| Architecture | ABI | Data Model | ELF Machine |
|-------------|-----|------------|-------------|
| **x86-64** | System V AMD64 | LP64 | `EM_X86_64` |
| **i686** | cdecl / System V i386 | ILP32 | `EM_386` |
| **AArch64** | AAPCS64 | LP64 | `EM_AARCH64` |
| **RISC-V 64** | LP64D | LP64 | `EM_RISCV` |

### Zero-Dependency Mandate

The `[dependencies]` section in `Cargo.toml` is **empty by design**. All functionality — FxHash,
PUA/UTF-8 encoding, software long-double arithmetic, ELF writing, DWARF emission, assemblers, and
linkers — is hand-implemented in Rust. No external crates of any kind are used (this extends to
`[dev-dependencies]` and `[build-dependencies]` as well).

---

## Building

### Prerequisites

- **Rust 1.93.0+** (stable) — install via [rustup](https://rustup.rs/)
- **Cargo** (bundled with the Rust toolchain)
- **Linux** host system (BCC is a Linux-only toolchain)

### Build Commands

```bash
# Source the Rust toolchain environment (if needed)
source "$HOME/.cargo/env"

# Debug build
cargo build

# Release build (recommended — optimized for compiler performance)
cargo build --release
```

The release profile is configured with maximum optimizations for the compiler binary itself:

- `opt-level = 3` — maximum optimization
- `lto = "thin"` — link-time optimization
- `codegen-units = 1` — single codegen unit for best optimization

The resulting binary is located at `target/release/bcc`.

### Worker Thread Stack

BCC spawns a worker thread with a **64 MiB stack** (`67,108,864` bytes) to handle deeply nested
kernel macro expansions and recursive parser invocations. This is configured via the
`RUST_MIN_STACK` environment variable in `.cargo/config.toml` and enforced at runtime through
`std::thread::Builder::new().stack_size(64 * 1024 * 1024)`. A **512-depth recursion limit** is
also enforced in the parser and macro expander to prevent stack overflow.

---

## Usage

### CLI Interface

```
./bcc [flags] <input.c> [-o output]
```

BCC accepts GCC-compatible flags for drop-in compatibility (e.g., `make CC=./bcc`).

### Supported Flags

| Flag | Description |
|------|-------------|
| `--target={x86-64\|i686\|aarch64\|riscv64}` | Select target architecture (default: host) |
| `-o <file>` | Specify output file path |
| `-c` | Compile only — produce a relocatable object file (`.o`) |
| `-S` | Emit assembly output |
| `-E` | Preprocess only — output preprocessed source to stdout |
| `-g` | Emit DWARF v4 debug information (`.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str`) |
| `-O0` | No optimization (default) |
| `-fPIC` | Generate position-independent code for shared libraries |
| `-shared` | Produce a shared library (`ET_DYN` ELF) instead of an executable |
| `-mretpoline` | Enable retpoline indirect branch mitigation (x86-64 only) |
| `-fcf-protection` | Enable Intel CET / IBT control-flow enforcement (x86-64 only) |
| `-I<dir>` | Add `<dir>` to the include search path |
| `-D<macro>[=value]` | Define a preprocessor macro |
| `-L<dir>` | Add `<dir>` to the library search path |
| `-l<lib>` | Link against library `<lib>` |

### Current Status

BCC is under active development. The compiler pipeline (preprocessing, lexing, parsing, semantic
analysis, IR construction, optimization, code generation, and object file assembly) is functional
for all four target architectures. The following capabilities are **fully operational**:

- **Compile to object file** (`-c`): Produces valid ELF relocatable (`.o`) files with correct
  sections, symbols, and relocations
- **Preprocess** (`-E`): Full macro expansion, `#include` resolution, conditional compilation
- **Emit assembly** (`-S`): Produces target-architecture assembly output (AT&T syntax for x86)
- **DWARF debug info** (`-g`): Emits DWARF v4 debug sections at `-O0`
- **Security mitigations** (x86-64): Retpoline, CET/IBT, stack probes all functional
- **PIC code generation** (`-fPIC`): Position-independent code with GOT-relative addressing
- **Cross-architecture compilation** (`--target=`): Produces correct ELF object files for all 4
  architectures (when compiling files that do not depend on system headers)

**Known limitation — system library linking:** The built-in linker does not yet search or link
against system libraries (e.g., libc). The `-l` and `-L` flags are accepted but library file
resolution is not yet implemented. Programs that reference libc symbols (`printf`, `puts`,
`malloc`, etc.) will fail at the link stage with undefined symbol errors. Freestanding programs
(those that do not depend on libc) can be linked into ELF executables successfully.

**Known limitation — cross-architecture system headers:** Cross-compilation with `--target=`
requires the target architecture's system headers (sysroot) to be installed on the host. For
example, compiling for AArch64 requires `libc6-dev-arm64-cross` (or equivalent) to be installed.
Files that do not use system `#include` directives (e.g., `#include <stdio.h>`) can be compiled
for any target without additional packages.

### Examples

**Compile to object file (all architectures):**

```bash
# Compile a C file to a relocatable object file
./bcc -c -o main.o main.c

# Cross-compile to object file for AArch64 (no system headers needed for header-free files)
./bcc --target=aarch64 -c -o main.o main.c
```

**Preprocess only:**

```bash
# Output preprocessed source to stdout
./bcc -E main.c
```

**Compile with debug information:**

```bash
# Produces .o with DWARF v4 debug sections
./bcc -g -c -o myprogram.o myprogram.c
```

**Compile with security mitigations (x86-64):**

```bash
# Object file with retpoline + CET/IBT protections
./bcc -mretpoline -fcf-protection -c -o secure.o secure.c
```

**Produce a PIC object file for shared library use:**

```bash
./bcc -fPIC -c -o foo.o foo.c
```

**Hello World — full compilation target (requires libc linking support):**

```bash
# This is the target workflow once system library linking is implemented:
cat > hello.c << 'EOF'
#include <stdio.h>
int main(void) {
    printf("Hello, World!\n");
    return 0;
}
EOF

# Currently, compile-to-object works:
./bcc -c -o hello.o hello.c

# Full linking (./bcc -o hello hello.c && ./hello) requires system library
# resolution which is under development.
```

---

## Architecture

BCC implements a complete 10+ phase compilation pipeline, from source text to native ELF binaries.

### Compilation Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Phase 1-2   │    │  Phase 3    │    │  Phase 4    │    │  Phase 5    │
│Preprocessing│───▶│   Lexer     │───▶│   Parser    │───▶│  Semantic   │
│             │    │             │    │             │    │  Analysis   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│ Phase 10    │    │  Phase 9    │    │  Phase 8    │    │  Phase 6    │
│   Code      │◀───│    Phi      │◀───│Optimization │◀───│ IR Lowering │
│ Generation  │    │ Elimination │    │   Passes    │    │  (alloca)   │
└──────┬──────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
       │                                                        │
       │                                                 ┌──────▼──────┐
       │                                                 │  Phase 7    │
       │                                                 │   mem2reg   │
       │                                                 │(SSA promote)│
       │                                                 └─────────────┘
       │
┌──────▼──────┐    ┌─────────────┐
│  Built-in   │───▶│  Built-in   │───▶  ELF Binary (.o / executable / .so)
│  Assembler  │    │   Linker    │
└─────────────┘    └─────────────┘
```

### Pipeline Phases

| Phase | Stage | Description |
|-------|-------|-------------|
| 1–2 | **Preprocessing** | Trigraph replacement, line splicing, `#include` resolution, `#define`/`#if` directive processing, macro expansion with paint-marker recursion protection, PUA encoding for non-UTF-8 byte round-tripping |
| 3 | **Lexical Analysis** | Tokenization with PUA-aware UTF-8 scanning, keyword recognition (C11 + GCC extensions), numeric/string/character literal parsing |
| 4 | **Parsing** | Recursive-descent C11 parser with full GCC extension support (statement expressions, `typeof`, computed gotos, case ranges, inline assembly, `__attribute__`) |
| 5 | **Semantic Analysis** | Type checking, implicit conversions, scope management, symbol table construction, constant evaluation, GCC builtin evaluation, attribute validation, designated initializer analysis |
| 6 | **IR Lowering** | AST-to-IR translation using the "alloca-first" pattern — all local variables are initially placed in memory via `alloca` instructions |
| 7 | **SSA Construction** | mem2reg pass promotes eligible allocas to SSA virtual registers using Lengauer-Tarjan dominator tree computation and iterated dominance frontier phi-node insertion |
| 8 | **Optimization** | Constant folding and propagation, dead code elimination, control-flow graph simplification (managed by a pass scheduler iterating to fixpoint) |
| 9 | **Phi Elimination** | Converts SSA phi nodes to parallel copy operations at predecessor block ends, producing a register-allocator-friendly IR |
| 10 | **Code Generation** | Architecture-specific instruction selection, register allocation (linear scan), security mitigation injection (retpoline, CET, stack probe for x86-64) |
| — | **Assembly** | Built-in assembler encodes architecture-specific machine instructions with full relocation support |
| — | **Linking** | Built-in linker performs symbol resolution, section merging, relocation patching, and produces final ELF output (`ET_EXEC` or `ET_DYN`) with optional GOT/PLT/dynamic sections |

### Module Structure

```
src/
├── main.rs                  # CLI entry point, driver orchestration, worker thread spawn
├── lib.rs                   # Library root, public module declarations
├── common/                  # Infrastructure layer
│   ├── fx_hash.rs           #   FxHasher for performant symbol table hashing
│   ├── encoding.rs          #   PUA/UTF-8 encoding for non-UTF-8 byte round-tripping
│   ├── long_double.rs       #   Software long-double (80-bit) arithmetic
│   ├── temp_files.rs        #   RAII temporary file management
│   ├── types.rs             #   Dual type system (C language types + machine types)
│   ├── type_builder.rs      #   Builder API for constructing complex types
│   ├── diagnostics.rs       #   Multi-error diagnostic reporting engine
│   ├── source_map.rs        #   Source file tracking, line/column mapping
│   ├── string_interner.rs   #   String interning with FxHash
│   └── target.rs            #   Target triple definitions, arch-specific constants
├── frontend/                # Frontend pipeline
│   ├── preprocessor/        #   Phase 1-2: directives, macros, paint markers
│   ├── lexer/               #   Phase 3: tokenization, PUA-aware scanning
│   ├── parser/              #   Phase 4: C11 + GCC extension parsing, AST
│   └── sema/                #   Phase 5: type checking, scopes, builtins
├── ir/                      # Middle-end
│   ├── instructions.rs      #   IR instruction definitions
│   ├── basic_block.rs       #   Basic block representation
│   ├── function.rs          #   IR function representation
│   ├── module.rs            #   IR module (globals, functions, strings)
│   ├── types.rs             #   IR type system
│   ├── builder.rs           #   IR builder API
│   ├── lowering/            #   Phase 6: AST-to-IR, alloca insertion
│   └── mem2reg/             #   Phase 7 + 9: SSA construction, phi elimination
├── passes/                  # Optimization passes (Phase 8)
│   ├── pass_manager.rs      #   Pass scheduling framework
│   ├── constant_folding.rs  #   Constant folding and propagation
│   ├── dead_code_elimination.rs  # Dead code elimination
│   └── simplify_cfg.rs      #   CFG simplification
└── backend/                 # Code generation and output
    ├── traits.rs            #   ArchCodegen trait (architecture abstraction)
    ├── generation.rs        #   Phase 10: dispatch + security mitigation injection
    ├── register_allocator.rs #  Linear scan register allocation
    ├── elf_writer_common.rs #   Common ELF writing infrastructure
    ├── dwarf/               #   DWARF v4 debug info (.debug_info, _abbrev, _line, _str)
    ├── linker_common/       #   Shared linker infrastructure, dynamic linking
    ├── x86_64/              #   x86-64 codegen, assembler, linker
    ├── i686/                #   i686 codegen, assembler, linker
    ├── aarch64/             #   AArch64 codegen, assembler, linker
    └── riscv64/             #   RISC-V 64 codegen, assembler, linker
```

Each architecture backend (`x86_64/`, `i686/`, `aarch64/`, `riscv64/`) contains:

- `mod.rs` — `ArchCodegen` trait implementation
- `codegen.rs` — Architecture-specific instruction selection
- `registers.rs` — Register definitions and classification
- `abi.rs` — Calling convention and parameter passing rules
- `assembler/` — Built-in assembler (instruction encoding, relocations)
- `linker/` — Built-in ELF linker (relocation application)

---

## Features

### C11 Compliance with GCC Extensions

- **C11 (ISO/IEC 9899:2011)** baseline language support including `_Static_assert`, `_Alignof`/
  `_Alignas`, `_Noreturn`, `_Generic`, `_Atomic`, `_Thread_local`, anonymous structs/unions,
  unicode character literals, and `_Complex`
- **21+ GCC attributes:** `aligned`, `packed`, `section`, `used`, `unused`, `weak`, `constructor`,
  `destructor`, `visibility`, `deprecated`, `noreturn`, `noinline`, `always_inline`, `cold`, `hot`,
  `format`, `format_arg`, `malloc`, `pure`, `const`, `warn_unused_result`, `fallthrough`
- **GCC language extensions:** statement expressions (`({ ... })`), `typeof`/`__typeof__`,
  zero-length arrays, designated initializers (out-of-order, nested), computed gotos (`goto *ptr`),
  case ranges (`case 1 ... 5:`), conditional expression omission (`x ?: y`), `__extension__`,
  transparent unions, local labels (`__label__`)
- **~30 GCC builtins:** `__builtin_expect`, `__builtin_unreachable`, `__builtin_constant_p`,
  `__builtin_offsetof`, `__builtin_types_compatible_p`, `__builtin_choose_expr`,
  `__builtin_clz`/`ctz`/`popcount`, `__builtin_bswap*`, `__builtin_ffs`, `__builtin_va_*`,
  `__builtin_frame_address`/`return_address`, `__builtin_trap`, `__builtin_assume_aligned`,
  overflow arithmetic builtins, and more

### Inline Assembly

- Full AT&T syntax support with output/input operands and clobber lists
- Constraint support: `"r"`, `"m"`, `"i"`, `"n"`, `"=r"`, `"=m"`, `"+r"`, `"memory"`, `"cc"`
- Named operands (`[name]`), `asm volatile`, `asm goto` with jump labels
- `.pushsection`/`.popsection` directive handling

### Standalone Backend (No External Toolchain)

BCC includes its own **built-in assembler** and **built-in linker** for all four target
architectures. The compiler never invokes external tools (`as`, `ld`, `gcc`, `llvm-mc`, `lld`).
All machine code encoding, relocation processing, ELF section construction, and final binary
production are handled internally.

### DWARF v4 Debug Information

When compiled with `-g`, BCC emits DWARF v4 debug sections:

- `.debug_info` — Compilation unit, subprogram, and variable DIEs
- `.debug_abbrev` — Abbreviation table
- `.debug_line` — Line number program for source file/line mapping
- `.debug_str` — Debug string table

Debug information is supported at `-O0` only. Binaries compiled without `-g` contain **zero**
debug sections.

### PIC and Shared Library Support

- Full `-fPIC` position-independent code generation across all four architectures
- `-shared` produces `ET_DYN` ELF shared objects
- GOT (Global Offset Table) and PLT (Procedure Linkage Table) relocation emission
- `.dynamic`, `.dynsym`, `.rela.dyn`, `.rela.plt`, `.gnu.hash` section generation
- Symbol visibility control (`default`, `hidden`, `protected`)

### Security Mitigations (x86-64)

- **Retpoline** (`-mretpoline`): indirect call/jump instructions are redirected through
  `__x86_indirect_thunk_*` stubs to mitigate Spectre v2
- **CET / IBT** (`-fcf-protection`): `endbr64` instructions are inserted at function entries
  and indirect branch targets for Intel Control-flow Enforcement Technology
- **Stack Probe**: functions with stack frames exceeding 4,096 bytes emit a probe loop before
  the stack pointer adjustment to touch each guard page

### Software Long-Double Arithmetic

IEEE 754 extended-precision (80-bit) arithmetic is implemented entirely in software
(`src/common/long_double.rs`) — no external math libraries are required, honoring the
zero-dependency mandate.

### PUA Encoding for Non-UTF-8 Fidelity

Non-UTF-8 bytes (0x80–0xFF) in C source files are mapped to Unicode Private Use Area code points
(U+E080–U+E0FF) during preprocessing and decoded back to the exact original bytes during code
generation. This ensures **byte-exact fidelity** through the entire compilation pipeline — critical
for Linux kernel source files containing binary data in string literals and inline assembly.

---

## Validation Checkpoints

BCC follows a strictly sequential checkpoint validation protocol. Checkpoints 1–6 are **hard
gates** — failure at any checkpoint halts all forward progress. Checkpoint 7 is optional.

| # | Checkpoint | Criteria |
|---|-----------|----------|
| 1 | **Hello World** | Compile and run "Hello, World!" on all four architectures (x86-64, i686, AArch64, RISC-V 64) |
| 2 | **Language Correctness** | PUA round-trip fidelity, recursive macro termination, statement expressions, `typeof`, designated initializers, inline assembly, computed gotos, `_Static_assert`, `_Generic`, GCC builtins |
| 3 | **Internal Test Suite** | 100% pass rate on the full unit and integration test suite; must be re-run after any feature addition to confirm no regressions |
| 4 | **Shared Library + DWARF** | Validate ELF shared library structure (GOT/PLT, `.dynamic`, `.dynsym`), verify DWARF debug sections with GDB source/line resolution |
| 5 | **Security Mitigations** | Retpoline thunk verification (indirect calls via `__x86_indirect_thunk_*`), CET `endbr64` at function entries, stack probe loop for large frames (x86-64 only) |
| 6 | **Linux Kernel Boot** | Compile Linux kernel 6.9 (RISC-V configuration) with `make ARCH=riscv CC=./bcc` and boot to userspace in QEMU, verifying `USERSPACE_OK` output from a minimal `/init` |
| 7 | **Stretch Targets** *(optional)* | Successfully compile SQLite, Redis, PostgreSQL, and FFmpeg — each within 5× GCC-equivalent build time |

### Backend Validation Order

Architecture backends are validated in a fixed order: **x86-64** → **i686** → **AArch64** →
**RISC-V 64**.

---

## Testing

### Running Tests

```bash
# Run the full test suite (release mode recommended)
cargo test --release

# Run with verbose output
cargo test --release -- --nocapture

# Run a specific checkpoint test
cargo test --release checkpoint1_hello_world
cargo test --release checkpoint2_language

# Clippy lint check
cargo clippy -- -W clippy::all

# Format check
cargo fmt --check
```

### Test Directory Structure

```
tests/
├── common/
│   └── mod.rs                        # Shared test utilities (BCC invocation, ELF inspection)
├── checkpoint1_hello_world.rs        # Hello World — all four architectures
├── checkpoint2_language.rs           # Language and preprocessor correctness
├── checkpoint3_internal.rs           # Full internal test suite
├── checkpoint4_shared_lib.rs         # Shared library + DWARF validation
├── checkpoint5_security.rs           # Security mitigations (x86-64)
├── checkpoint6_kernel.rs             # Linux kernel build + QEMU boot
├── checkpoint7_stretch.rs            # Stretch targets (optional)
└── fixtures/
    ├── hello.c                       # Hello World source
    ├── pua_roundtrip.c               # Non-UTF-8 byte round-trip test
    ├── recursive_macro.c             # #define A A recursion test
    ├── stmt_expr.c                   # Statement expression test
    ├── typeof_test.c                 # typeof test
    ├── designated_init.c             # Designated initializer test
    ├── inline_asm_basic.c            # Basic inline assembly test
    ├── inline_asm_constraints.c      # Inline assembly constraints test
    ├── computed_goto.c               # Computed goto dispatch test
    ├── zero_length_array.c           # Zero-length array test
    ├── builtins.c                    # GCC builtins coverage test
    ├── static_assert.c              # _Static_assert test
    ├── generic.c                     # _Generic selection test
    ├── shared_lib/
    │   ├── foo.c                     # Shared library exported functions
    │   └── main.c                    # Dynamic linking consumer
    ├── security/
    │   ├── retpoline.c               # Retpoline function pointer test
    │   ├── cet.c                     # CET/IBT indirect branch test
    │   └── stack_probe.c             # Large stack frame probe test
    └── dwarf/
        └── debug_test.c              # DWARF debug information test
```

### Validation Tools

The following host tools are used during test validation (not compile-time dependencies):

- **binutils** (`readelf`, `objdump`) — ELF structure and disassembly inspection
- **QEMU user-mode** (`qemu-aarch64`, `qemu-riscv64`) — cross-architecture binary execution
- **QEMU system** (`qemu-system-riscv64`) — full system emulation for kernel boot
- **GDB** — DWARF debug information verification

---

## Documentation

Detailed technical documentation is maintained in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | System architecture — pipeline stages, module interactions, data flow diagrams |
| [`docs/gcc_extensions.md`](docs/gcc_extensions.md) | GCC extension manifest — supported attributes, builtins, language extensions with implementation status |
| [`docs/validation_checkpoints.md`](docs/validation_checkpoints.md) | Validation protocol — checkpoint definitions, pass/fail criteria, execution order |
| [`docs/abi_reference.md`](docs/abi_reference.md) | ABI documentation — calling conventions for all four target architectures |
| [`docs/elf_format.md`](docs/elf_format.md) | ELF output format — section layout, program headers, dynamic linking structures |

---

## Resource Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Worker thread stack | 64 MiB (`67,108,864` bytes) | Deeply nested kernel macro expansions |
| Recursion depth limit | 512 | Parser and macro expander stack overflow prevention |
| Wall-clock ceiling | 5× GCC-equivalent | Linux kernel build time must not exceed 5× GCC on same hardware |
| Output format | ELF only (`ET_EXEC`, `ET_DYN`) | Linux-only target platform |
| Dependencies | Zero | No external Rust crates — standard library only |
