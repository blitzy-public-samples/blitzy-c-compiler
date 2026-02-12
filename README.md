# BCC — Blitzy's C Compiler

A complete, self-contained, zero-external-dependency C compilation toolchain implemented in Rust (2021 Edition) that cross-compiles C source code into native Linux ELF executables and shared objects for four target architectures: **x86-64**, **i686**, **AArch64**, and **RISC-V 64**.

BCC implements the full compilation pipeline — from preprocessing through code generation — including its own built-in assembler and built-in linker for every supported architecture. No external toolchain components (`as`, `ld`, `gcc`, `llvm-mc`) are invoked at any point during compilation.

---

## Table of Contents

- [Features](#features)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Compilation Pipeline Architecture](#compilation-pipeline-architecture)
- [Module Structure](#module-structure)
- [Supported Architectures](#supported-architectures)
- [Validation Checkpoints](#validation-checkpoints)
- [Design Constraints](#design-constraints)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Full C11 Compiler Pipeline

BCC implements a complete C11 (ISO/IEC 9899:2011) compilation pipeline spanning 10+ phases:

- **Preprocessing (Phases 1–2):** Trigraph replacement, line splicing, `#include` resolution, `#define`/`#undef` macro expansion (object-like and function-like with variadic support), conditional compilation (`#if`/`#ifdef`/`#elif`/`#else`/`#endif`), `#pragma`, `#error`, `#warning`, and `#line` directives. PUA encoding (U+E080–U+E0FF) ensures byte-exact round-tripping of non-UTF-8 source bytes. Paint-marker recursion protection prevents infinite expansion of self-referential macros.
- **Lexical Analysis (Phase 3):** Full C11 tokenization with PUA-aware UTF-8 scanning, numeric literal parsing (decimal, hex, octal, binary, floating-point, hex float), string/character literal parsing (escape sequences, wide/unicode prefixes `L`, `u8`, `u`, `U`), and GCC extension keyword recognition.
- **Parsing (Phase 4):** Recursive-descent C11 parser with extensive GCC extension support including statement expressions, `typeof`/`__typeof__`, computed gotos, case ranges, zero-length arrays, designated initializers, `__attribute__((...))`, and full inline assembly (AT&T syntax with constraints, clobbers, named operands, `asm goto`).
- **Semantic Analysis (Phase 5):** Type checking with implicit conversions, integer promotion, usual arithmetic conversions, scope management (block/function/file/global), symbol table with linkage resolution, compile-time constant evaluation, GCC builtin evaluation, designated initializer analysis, and attribute semantic validation.
- **IR Lowering (Phase 6):** AST-to-IR translation using the alloca-then-promote pattern — all local variables are initially placed as `alloca` instructions in the entry block.
- **SSA Construction (Phase 7):** Mem2reg pass promotes eligible allocas to SSA virtual registers using Lengauer-Tarjan dominator tree computation and iterated dominance frontier phi-node placement.
- **Optimization (Phase 8):** Constant folding and propagation, dead code elimination, and control-flow graph simplification, run to fixpoint.
- **Phi Elimination (Phase 9):** Converts SSA phi nodes to parallel copies at predecessor block terminators for register allocation.
- **Code Generation (Phase 10):** Architecture-dispatching instruction selection, register allocation (linear scan), and machine code emission with security mitigation injection for x86-64.
- **Assembly & Linking:** Built-in assembler and linker per architecture produce relocatable objects, static ELF executables (`ET_EXEC`), and shared objects (`ET_DYN`) without invoking any external tool.

### Zero-Dependency Mandate

The `[dependencies]` section of `Cargo.toml` is empty. Every capability is hand-implemented in Rust using only the standard library:

| Typical External Crate | BCC Internal Module | Purpose |
|------------------------|---------------------|---------|
| `fxhash` / `ahash` | `src/common/fx_hash.rs` | FxHasher for symbol tables and lookup maps |
| `encoding_rs` | `src/common/encoding.rs` | PUA/UTF-8 encoding for non-UTF-8 byte round-tripping |
| `num` / `rug` | `src/common/long_double.rs` | Software 80-bit extended-precision arithmetic |
| `tempfile` | `src/common/temp_files.rs` | RAII temporary file management |
| `object` / `elf` | `src/backend/elf_writer_common.rs` | ELF binary format writing |
| `gimli` | `src/backend/dwarf/` | DWARF v4 debug information generation |
| `clap` / `structopt` | `src/main.rs` | Hand-rolled CLI argument parsing |
| `codespan-reporting` | `src/common/diagnostics.rs` | Diagnostic formatting and error reporting |

### Self-Contained Assembler and Linker

Each target architecture includes a fully integrated assembler and linker:

- **Assemblers** encode architecture-specific machine instructions (ModR/M, SIB, REX for x86-64; fixed-width A64 for AArch64; R/I/S/B/U/J formats for RISC-V) and emit relocatable object code.
- **Linkers** perform symbol resolution (strong/weak binding), section merging, relocation application, and produce final ELF binaries with correct program headers, section headers, and string/symbol tables.

### Multi-Architecture Code Generation

Native machine code emission for four Linux architectures, each with architecture-specific ABI conformance, calling conventions, and instruction selection:

- **x86-64** — System V AMD64 ABI, 16 GPRs + 16 SSE registers, variable-length encoding
- **i686** — cdecl/System V i386 ABI, 8 GPRs, stack-based parameter passing
- **AArch64** — AAPCS64 ABI, 31 GPRs + 32 SIMD/FP registers, fixed 32-bit instruction width
- **RISC-V 64** — LP64D ABI, 32 integer + 32 FP registers, RV64IMAFDC ISA

### GCC Extension Coverage

Comprehensive support for GCC attributes, language extensions, builtins, and inline assembly as required by real-world C codebases including the Linux kernel:

- **21+ Attributes:** `aligned`, `packed`, `section`, `used`, `unused`, `weak`, `constructor`, `destructor`, `visibility`, `deprecated`, `noreturn`, `noinline`, `always_inline`, `cold`, `hot`, `format`, `format_arg`, `malloc`, `pure`, `const`, `warn_unused_result`, `fallthrough`
- **Language Extensions:** Statement expressions `({ ... })`, `typeof`/`__typeof__`, zero-length arrays, computed gotos (`goto *ptr`), case ranges (`1 ... 5`), conditional operand omission (`x ?: y`), `__extension__`, transparent unions, local labels (`__label__`)
- **~30 Builtins:** `__builtin_expect`, `__builtin_unreachable`, `__builtin_constant_p`, `__builtin_offsetof`, `__builtin_types_compatible_p`, `__builtin_choose_expr`, `__builtin_clz`/`ctz`/`popcount`, `__builtin_bswap*`, `__builtin_ffs`, `__builtin_va_*`, `__builtin_frame_address`, `__builtin_return_address`, `__builtin_trap`, `__builtin_assume_aligned`, overflow arithmetic builtins
- **Inline Assembly:** AT&T syntax, output/input operands with full constraint support (`"=r"`, `"=m"`, `"+r"`, `"r"`, `"i"`, `"n"`), clobber lists (`"memory"`, `"cc"`), named operands (`[name]`), `asm volatile`, `asm goto` with jump labels, `.pushsection`/`.popsection`

### Security-Hardened Code Generation (x86-64)

- **Retpoline** (`-mretpoline`): Indirect calls/jumps route through `__x86_indirect_thunk_*` trampolines to mitigate Spectre v2
- **Intel CET/IBT** (`-fcf-protection`): `endbr64` landing pads inserted at function entries and indirect branch targets
- **Stack Guard Page Probing:** Frames exceeding 4,096 bytes emit a probe loop before the stack pointer adjustment to ensure guard pages are touched

### PIC and Shared Library Support

Full `-fPIC` code generation and `-shared` linking across all four architectures:

- GOT/PLT relocation emission for position-independent code
- `.dynamic`, `.dynsym`, `.dynstr`, `.rela.dyn`, `.rela.plt`, `.gnu.hash` section generation
- `PT_DYNAMIC` and `PT_INTERP` program header emission
- Symbol visibility control (`default`, `hidden`, `protected`)

### DWARF v4 Debug Information

When compiled with `-g`, BCC emits DWARF v4 debug sections at `-O0`:

- `.debug_info` — Compilation unit, subprogram, and variable DIEs
- `.debug_abbrev` — Abbreviation table encoding
- `.debug_line` — Line number program with file/directory tables
- `.debug_str` — Debug string table

Binaries compiled without `-g` contain zero `.debug_*` sections.

---

## Build Instructions

### Prerequisites

- **Rust 1.93.0+** (stable) — install via [rustup](https://rustup.rs/)
- **Linux host** (Ubuntu 24.04 recommended)
- No external Rust crates or C libraries are required

### Building

```bash
# Clone the repository
git clone <repository-url>
cd blitzy-c-compiler

# Build the release binary
cargo build --release
```

The `bcc` binary is produced at `target/release/bcc`.

### Release Profile

The release build is configured for maximum performance:

| Setting | Value | Purpose |
|---------|-------|---------|
| `opt-level` | `3` | Maximum optimization for compiler binary speed |
| `lto` | `"thin"` | Thin link-time optimization for cross-module inlining |
| `codegen-units` | `1` | Single codegen unit for best optimization opportunities |

### Running Tests

```bash
# Run the full test suite
cargo test --release

# Run lints
cargo clippy

# Check formatting
cargo fmt --check
```

---

## Usage

### Basic Invocation

```
./bcc [flags] <input.c> [-o output]
```

### Examples

```bash
# Compile and run a Hello World program
./bcc -o hello hello.c && ./hello

# Compile to object file only
./bcc -c -o main.o main.c

# Preprocess only (output to stdout)
./bcc -E input.c

# Generate assembly output
./bcc -S -o output.s input.c

# Cross-compile for AArch64
./bcc --target=aarch64 -o hello_arm hello.c

# Compile with debug information
./bcc -g -O0 -o debug_binary program.c

# Build a shared library with PIC
./bcc -fPIC -shared -o libfoo.so foo.c

# Compile with security mitigations (x86-64)
./bcc -mretpoline -fcf-protection -o hardened program.c

# Use the Linux kernel build system
make ARCH=riscv CC=./bcc
```

### CLI Flags Reference

| Flag | Description |
|------|-------------|
| `-o <file>` | Write output to `<file>` |
| `-c` | Compile and assemble, but do not link (produce `.o` object file) |
| `-S` | Compile only, produce assembly output |
| `-E` | Preprocess only, output to stdout |
| `-g` | Emit DWARF v4 debug information (at `-O0` only) |
| `-O0` | No optimization (default) |
| `--target={x86-64\|i686\|aarch64\|riscv64}` | Select target architecture |
| `-fPIC` | Generate position-independent code |
| `-shared` | Produce a shared object (`ET_DYN`) instead of an executable |
| `-mretpoline` | Enable retpoline indirect branch mitigation (x86-64 only) |
| `-fcf-protection` | Enable Intel CET/IBT control-flow protection (x86-64 only) |
| `-I<dir>` | Add `<dir>` to the include search path |
| `-D<macro>[=<value>]` | Define a preprocessor macro |
| `-L<dir>` | Add `<dir>` to the library search path |
| `-l<lib>` | Link against library `lib<lib>` |

---

## Compilation Pipeline Architecture

BCC processes C source code through a multi-phase pipeline, transforming source text into native ELF binaries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BCC Compilation Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌───────┐   ┌────────┐   ┌──────────┐   ┌────────────┐   │
│  │ Preproc. │──▶│ Lexer │──▶│ Parser │──▶│ Semantic │──▶│ IR Lowering│   │
│  │ Phase 1-2│   │Phase 3│   │Phase 4 │   │ Analysis │   │  Phase 6   │   │
│  │          │   │       │   │        │   │ Phase 5  │   │ (alloca)   │   │
│  └──────────┘   └───────┘   └────────┘   └──────────┘   └─────┬──────┘   │
│                                                                 │          │
│                                                                 ▼          │
│  ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌────────────────────┐      │
│  │   ELF    │◀──│ Linker  │◀──│Assembler │◀──│   Code Generation  │      │
│  │  Output  │   │         │   │          │   │     Phase 10       │      │
│  └──────────┘   └─────────┘   └──────────┘   └────────┬───────────┘      │
│                                                         │                  │
│       ┌──────────────┐   ┌───────────┐   ┌─────────────┘                  │
│       │     Phi      │──▶│Optimization│──▶│ SSA / mem2reg │               │
│       │ Elimination  │   │  Passes   │   │   Phase 7     │               │
│       │  Phase 9     │   │  Phase 8  │   │  (promote)    │               │
│       └──────────────┘   └───────────┘   └───────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Summary

| Phase | Component | Input | Output |
|-------|-----------|-------|--------|
| 1–2 | Preprocessor | C source files | Macro-expanded token stream |
| 3 | Lexer | Expanded character stream | Token stream |
| 4 | Parser | Token stream | Abstract Syntax Tree (AST) |
| 5 | Semantic Analyzer | AST | Type-annotated AST |
| 6 | IR Lowering | Annotated AST | IR with alloca instructions |
| 7 | Mem2Reg (SSA) | IR with allocas | SSA-form IR with phi nodes |
| 8 | Optimization Passes | SSA IR | Optimized SSA IR |
| 9 | Phi Elimination | SSA IR | Register-assignment IR |
| 10 | Code Generation | IR | Machine code + relocations |
| — | Assembler | Machine code | Relocatable object (`.o`) |
| — | Linker | Object files | ELF executable or shared object |

---

## Module Structure

```
src/
├── main.rs                      # CLI entry point, driver, 64 MiB worker thread
├── lib.rs                       # Library root, public module declarations
│
├── common/                      # Infrastructure layer
│   ├── mod.rs                   # Module declarations
│   ├── fx_hash.rs               # FxHasher — fast hash for symbol tables
│   ├── encoding.rs              # PUA/UTF-8 encoding for non-UTF-8 round-tripping
│   ├── long_double.rs           # Software 80-bit extended-precision math
│   ├── temp_files.rs            # RAII temporary file management
│   ├── types.rs                 # Dual type system (C types + machine types)
│   ├── type_builder.rs          # Type construction builder API
│   ├── diagnostics.rs           # Multi-error diagnostic reporting engine
│   ├── source_map.rs            # Source file tracking, line/column mapping
│   ├── string_interner.rs       # String interning with FxHash
│   └── target.rs                # Target triple definitions and arch constants
│
├── frontend/                    # Frontend pipeline
│   ├── mod.rs                   # Module declarations
│   ├── preprocessor/            # Phases 1–2
│   │   ├── mod.rs               # Preprocessor driver
│   │   ├── directives.rs        # #include, #define, #if, #pragma, etc.
│   │   ├── macro_expander.rs    # Macro expansion with paint-marker protection
│   │   ├── paint_marker.rs      # Token-level paint markers for recursion suppression
│   │   ├── include_handler.rs   # #include file resolution and guard detection
│   │   ├── token_paster.rs      # ## concatenation and # stringification
│   │   ├── expression.rs        # #if constant expression evaluation
│   │   └── predefined.rs        # __FILE__, __LINE__, arch-specific defines
│   ├── lexer/                   # Phase 3
│   │   ├── mod.rs               # Lexer driver
│   │   ├── token.rs             # Token type definitions
│   │   ├── scanner.rs           # PUA-aware character scanner
│   │   ├── number_literal.rs    # Numeric literal parsing
│   │   └── string_literal.rs    # String/character literal parsing
│   ├── parser/                  # Phase 4
│   │   ├── mod.rs               # Recursive-descent parser driver
│   │   ├── ast.rs               # AST node definitions
│   │   ├── declarations.rs      # Declaration parsing
│   │   ├── expressions.rs       # Expression parsing (precedence climbing)
│   │   ├── statements.rs        # Statement parsing
│   │   ├── types.rs             # Type specifier/qualifier parsing
│   │   ├── gcc_extensions.rs    # GCC extension dispatch
│   │   ├── attributes.rs        # __attribute__((...)) parsing
│   │   └── inline_asm.rs        # Inline assembly parsing
│   └── sema/                    # Phase 5
│       ├── mod.rs               # Semantic analysis driver
│       ├── type_checker.rs      # Type checking and conversions
│       ├── scope.rs             # Lexical scope management
│       ├── symbol_table.rs      # Symbol table with linkage tracking
│       ├── constant_eval.rs     # Compile-time constant evaluation
│       ├── builtin_eval.rs      # GCC builtin evaluation
│       ├── initializer.rs       # Designated initializer analysis
│       └── attribute_handler.rs # Attribute semantic validation
│
├── ir/                          # Middle-end IR
│   ├── mod.rs                   # Module declarations
│   ├── instructions.rs          # IR instruction definitions
│   ├── basic_block.rs           # Basic block representation
│   ├── function.rs              # IR function representation
│   ├── module.rs                # IR module (globals, functions, strings)
│   ├── types.rs                 # IR type system
│   ├── builder.rs               # IR builder API
│   ├── lowering/                # Phase 6
│   │   ├── mod.rs               # AST-to-IR lowering driver
│   │   ├── expr_lowering.rs     # Expression lowering
│   │   ├── stmt_lowering.rs     # Statement lowering (CFG construction)
│   │   ├── decl_lowering.rs     # Declaration lowering
│   │   └── asm_lowering.rs      # Inline assembly lowering
│   └── mem2reg/                 # Phases 7 + 9
│       ├── mod.rs               # SSA construction driver
│       ├── dominator_tree.rs    # Lengauer-Tarjan dominator tree
│       ├── dominance_frontier.rs# Dominance frontier computation
│       ├── ssa_builder.rs       # SSA renaming pass
│       └── phi_eliminate.rs     # Phi-node elimination (Phase 9)
│
├── passes/                      # Phase 8 — Optimization
│   ├── mod.rs                   # Module declarations
│   ├── pass_manager.rs          # Pass scheduling and execution
│   ├── constant_folding.rs      # Constant folding and propagation
│   ├── dead_code_elimination.rs # Dead code elimination
│   └── simplify_cfg.rs          # CFG simplification
│
└── backend/                     # Code generation and output
    ├── mod.rs                   # Module declarations, arch dispatch
    ├── traits.rs                # ArchCodegen trait definition
    ├── generation.rs            # Phase 10 driver, security injection
    ├── register_allocator.rs    # Linear scan register allocator
    ├── elf_writer_common.rs     # Common ELF writing infrastructure
    ├── linker_common/           # Shared linker infrastructure
    │   ├── mod.rs               # Linker common module
    │   ├── symbol_resolver.rs   # Symbol resolution (strong/weak)
    │   ├── section_merger.rs    # Section aggregation and layout
    │   ├── relocation.rs        # Architecture-agnostic relocation framework
    │   ├── dynamic.rs           # Dynamic linking (.dynamic, .dynsym, etc.)
    │   └── linker_script.rs     # Default section-to-segment mapping
    ├── dwarf/                   # DWARF v4 debug information
    │   ├── mod.rs               # DWARF generation driver
    │   ├── info.rs              # .debug_info section
    │   ├── line.rs              # .debug_line section
    │   ├── abbrev.rs            # .debug_abbrev section
    │   └── str.rs               # .debug_str section
    ├── x86_64/                  # x86-64 backend
    │   ├── mod.rs               # ArchCodegen implementation
    │   ├── codegen.rs           # Instruction selection
    │   ├── registers.rs         # Register definitions
    │   ├── abi.rs               # System V AMD64 ABI
    │   ├── security.rs          # Retpoline, CET/IBT, stack probe
    │   ├── assembler/           # Built-in assembler
    │   │   ├── mod.rs
    │   │   ├── encoder.rs
    │   │   └── relocations.rs
    │   └── linker/              # Built-in linker
    │       ├── mod.rs
    │       └── relocations.rs
    ├── i686/                    # i686 backend
    │   ├── mod.rs               # ArchCodegen implementation
    │   ├── codegen.rs           # Instruction selection
    │   ├── registers.rs         # Register definitions
    │   ├── abi.rs               # cdecl/System V i386 ABI
    │   ├── assembler/
    │   │   ├── mod.rs
    │   │   ├── encoder.rs
    │   │   └── relocations.rs
    │   └── linker/
    │       ├── mod.rs
    │       └── relocations.rs
    ├── aarch64/                 # AArch64 backend
    │   ├── mod.rs               # ArchCodegen implementation
    │   ├── codegen.rs           # Instruction selection
    │   ├── registers.rs         # Register definitions
    │   ├── abi.rs               # AAPCS64 ABI
    │   ├── assembler/
    │   │   ├── mod.rs
    │   │   ├── encoder.rs
    │   │   └── relocations.rs
    │   └── linker/
    │       ├── mod.rs
    │       └── relocations.rs
    └── riscv64/                 # RISC-V 64 backend
        ├── mod.rs               # ArchCodegen implementation
        ├── codegen.rs           # Instruction selection
        ├── registers.rs         # Register definitions
        ├── abi.rs               # LP64D ABI
        ├── assembler/
        │   ├── mod.rs
        │   ├── encoder.rs
        │   └── relocations.rs
        └── linker/
            ├── mod.rs
            └── relocations.rs
```

---

## Supported Architectures

| Architecture | ABI | Pointer Width | Endianness | Data Model | Register File |
|-------------|-----|---------------|------------|------------|---------------|
| **x86-64** | System V AMD64 | 64-bit | Little | LP64 | 16 GPRs (RAX–R15) + 16 SSE (XMM0–XMM15) |
| **i686** | cdecl / System V i386 | 32-bit | Little | ILP32 | 8 GPRs (EAX–EDI) + x87 FPU stack |
| **AArch64** | AAPCS64 | 64-bit | Little | LP64 | 31 GPRs (X0–X30) + 32 SIMD/FP (V0–V31) |
| **RISC-V 64** | LP64D | 64-bit | Little | LP64 | 32 integer (x0–x31) + 32 FP (f0–f31) |

### Architecture-Specific Details

**x86-64 (System V AMD64)**
- Integer args: RDI, RSI, RDX, RCX, R8, R9
- FP args: XMM0–XMM7
- Return: RAX (integer), XMM0 (float)
- Callee-saved: RBX, RBP, R12–R15
- Struct classification: INTEGER, SSE, MEMORY
- Security mitigations: retpoline, CET/IBT, stack probing

**i686 (cdecl)**
- All arguments passed on the stack (right-to-left push order)
- Return: EAX (integer), x87 ST(0) (float)
- Callee-saved: EBX, ESI, EDI, EBP

**AArch64 (AAPCS64)**
- Integer args: X0–X7
- FP args: V0–V7 (D0–D7 for double, S0–S7 for float)
- Return: X0 (integer), V0 (float)
- HFA/HVA (Homogeneous Float/Vector Aggregate) handling for struct passing
- Callee-saved: X19–X28, X29 (FP), X30 (LR)

**RISC-V 64 (LP64D)**
- Integer args: a0–a7 (x10–x17)
- FP args: fa0–fa7 (f10–f17)
- Return: a0 (integer), fa0 (float)
- ISA: RV64IMAFDC (Integer, Multiply, Atomic, Float, Double, Compressed)
- LUI/AUIPC for large immediate materialization

---

## Validation Checkpoints

BCC uses a sequential checkpoint validation protocol. Checkpoints 1–6 are **strict hard gates** — failure at any gate halts forward progress. All must pass in order.

| Checkpoint | Name | Validation Criteria |
|-----------|------|---------------------|
| **1** | Hello World | Compile and execute `hello.c` on all four architectures. `./hello` prints `Hello, World!\n` with exit code 0. |
| **2** | Language Correctness | PUA byte round-tripping, recursive macro termination (`#define A A` completes in <5s), statement expressions, `typeof`, designated initializers, computed gotos, inline assembly, `_Static_assert`, `_Generic`. |
| **3** | Internal Test Suite | 100% pass rate on the full `cargo test --release` suite. Must be re-run after any feature addition to confirm zero regressions. |
| **4** | Shared Library & DWARF | Produce a working shared object (`-fPIC -shared`) with correct `.dynamic`/`.dynsym`/`.rela.dyn`/`.rela.plt`/`.gnu.hash` sections validated via `readelf`. DWARF v4 debug info validated via GDB source file/line resolution. |
| **5** | Security Mitigations | x86-64 retpoline: indirect calls target `__x86_indirect_thunk_*`. CET/IBT: `endbr64` at function entries. Stack probe: 8 KiB frame shows probe loop in disassembly. |
| **6** | Linux Kernel Boot | Compile Linux kernel 6.9 (RISC-V configuration) with `make ARCH=riscv CC=./bcc`, boot in QEMU with minimal `/init` printing `USERSPACE_OK\n` and rebooting. Build time ≤ 5× GCC equivalent. |
| **7** | Stretch Targets | *(Optional)* Compile SQLite, Redis, PostgreSQL, and FFmpeg. Not a hard gate. |

### Regression Policy

Any test that passed before a change and fails after constitutes a regression. Resolution is mandatory before proceeding to the next checkpoint. Checkpoint 3 must be re-executed after every feature addition during the kernel build phase.

---

## Design Constraints

### Zero-Dependency Mandate

No external Rust crates appear in `Cargo.toml`. This extends to `[dependencies]`, `[dev-dependencies]`, and `[build-dependencies]`. All hash functions, encoding utilities, math operations, ELF writing, DWARF emission, assemblers, and linkers are implemented internally using only the Rust standard library (`std`).

### Alloca-Then-Promote SSA Architecture

The IR lowering phase (Phase 6) initially emits all local variables as `alloca` instructions in the function entry block. The mem2reg pass (Phase 7) then promotes eligible allocas (scalar, non-address-taken) to SSA virtual registers using dominance frontier computation. This mirrors the LLVM approach to SSA construction and is a mandated architectural pattern.

### Resource Constraints

- **64 MiB Worker Thread Stack:** Worker threads are spawned via `std::thread::Builder::new().stack_size(64 * 1024 * 1024)` to handle deeply nested kernel macro expansions and recursive AST structures.
- **512-Depth Recursion Limit:** Enforced in the parser and macro expander to prevent stack overflow on deeply nested constructs.

### Standalone Backend Mode

BCC includes its own assembler and linker for all four target architectures. No external toolchain component (`as`, `ld`, `gcc`, `llvm-mc`, `lld`) is invoked at any point during compilation. PIC relocation handling (GOT, PLT) is entirely internal.

### Linux-Only ELF Output

Output format is exclusively ELF — `ET_EXEC` (static executables) and `ET_DYN` (shared objects). No support for macOS Mach-O, Windows PE/COFF, or any non-ELF binary format. The host platform is strictly Linux.

### Performance Ceiling

Linux kernel 6.9 full build time must not exceed 5× the time taken by GCC on the same source with equivalent configuration on the same hardware.

### PUA Encoding Fidelity

Non-UTF-8 bytes (0x80–0xFF) in C source files survive the entire pipeline with byte-exact fidelity via Private Use Area code point mapping (U+E080–U+E0FF). This is critical for the Linux kernel, which contains binary data in string literals and inline assembly operands.

---

## Contributing

Contributions to BCC are welcome. When contributing, please observe the following guidelines:

1. **Zero-Dependency Rule:** Do not add external crates to `Cargo.toml`. All functionality must be implemented using only the Rust standard library.
2. **Code Style:** Run `cargo fmt` before submitting. Code must pass `cargo clippy` with no warnings.
3. **Testing:** All changes must include appropriate tests. Run `cargo test --release` and ensure 100% pass rate. Never introduce regressions to existing checkpoints.
4. **Checkpoint Integrity:** If your change affects code generation or linking, re-validate all applicable checkpoints in sequential order.
5. **GCC Extension Additions:** New GCC extensions discovered during kernel builds should be documented in `docs/gcc_extensions.md` with implementation status.
6. **Architecture Parity:** Features implemented for one architecture should be implemented for all four unless the feature is architecture-specific (e.g., retpoline for x86-64 only).
7. **Commit Messages:** Use clear, descriptive commit messages referencing the relevant pipeline phase or checkpoint.

### Development Workflow

```bash
# Build
cargo build --release

# Test
cargo test --release

# Lint
cargo clippy

# Format
cargo fmt

# Run a specific checkpoint test
cargo test --release checkpoint1_hello_world
```

---

## License

Copyright (c) 2025 Blitzy. All rights reserved.
