# BCC ELF Output Format Reference

## Table of Contents

- [1. Overview](#1-overview)
- [2. ELF Header Configuration](#2-elf-header-configuration)
- [3. Section Layout](#3-section-layout)
- [4. Program Headers](#4-program-headers)
- [5. Symbol Tables](#5-symbol-tables)
- [6. Dynamic Linking Structures](#6-dynamic-linking-structures)
- [7. Relocation Types Per Architecture](#7-relocation-types-per-architecture)
- [8. ET_EXEC vs ET_DYN Differences](#8-et_exec-vs-et_dyn-differences)
- [9. DWARF Debug Sections](#9-dwarf-debug-sections)
- [10. Entry Point Convention](#10-entry-point-convention)

---

## 1. Overview

BCC produces **Linux ELF binaries exclusively**. The two output modes are:

| ELF Type | Value | Description | CLI Flags |
|----------|-------|-------------|-----------|
| **ET_EXEC** | `2` | Static executable | (default when linking) |
| **ET_DYN** | `3` | Shared object / position-independent executable | `-shared` |

BCC targets four architectures, all producing ELF output:

| Architecture | ELF Class | Data Encoding | Machine ID |
|-------------|-----------|---------------|------------|
| x86-64 | 64-bit | Little-endian | `EM_X86_64` (62) |
| i686 | 32-bit | Little-endian | `EM_386` (3) |
| AArch64 | 64-bit | Little-endian | `EM_AARCH64` (183) |
| RISC-V 64 | 64-bit | Little-endian | `EM_RISCV` (243) |

The BCC built-in linker and ELF writer (`src/backend/elf_writer_common.rs`) handle all binary format generation without invoking any external tools (`as`, `ld`, `lld`). This is the **standalone backend** architecture: the compiler assembles machine code internally and links relocatable objects into final executables or shared objects entirely within the BCC process.

No non-ELF formats (Mach-O, PE/COFF) are supported. The target platform is strictly Linux.

### Implementation Files

| Component | Source File |
|-----------|------------|
| Common ELF writing | `src/backend/elf_writer_common.rs` |
| Section merging | `src/backend/linker_common/section_merger.rs` |
| Symbol resolution | `src/backend/linker_common/symbol_resolver.rs` |
| Relocation processing | `src/backend/linker_common/relocation.rs` |
| Dynamic linking | `src/backend/linker_common/dynamic.rs` |
| Linker script / segment mapping | `src/backend/linker_common/linker_script.rs` |
| DWARF generation | `src/backend/dwarf/` |
| Architecture-specific assemblers | `src/backend/{x86_64,i686,aarch64,riscv64}/assembler/` |
| Architecture-specific linkers | `src/backend/{x86_64,i686,aarch64,riscv64}/linker/` |

---

## 2. ELF Header Configuration

Every ELF file produced by BCC begins with a standard ELF header. The header is either 52 bytes (32-bit / `ELFCLASS32`) or 64 bytes (64-bit / `ELFCLASS64`), depending on the target architecture.

### 2.1 ELF Identification (`e_ident`)

The first 16 bytes of the ELF header form the identification array (`e_ident[EI_NIDENT]`):

| Offset | Field | Size | Value | Description |
|--------|-------|------|-------|-------------|
| 0–3 | `EI_MAG0`–`EI_MAG3` | 4 bytes | `0x7f 0x45 0x4c 0x46` | ELF magic number (`\x7fELF`) |
| 4 | `EI_CLASS` | 1 byte | Architecture-dependent | File class (32-bit or 64-bit) |
| 5 | `EI_DATA` | 1 byte | `1` (ELFDATA2LSB) | Data encoding — always little-endian for all BCC targets |
| 6 | `EI_VERSION` | 1 byte | `1` (EV_CURRENT) | ELF version |
| 7 | `EI_OSABI` | 1 byte | `0` (ELFOSABI_NONE) | OS/ABI identification — System V / no specific OS |
| 8 | `EI_ABIVERSION` | 1 byte | `0` | ABI version |
| 9–15 | `EI_PAD` | 7 bytes | `0x00` | Padding bytes — reserved, set to zero |

### 2.2 Per-Architecture ELF Header Fields

The following fields vary by target architecture:

#### x86-64

| Field | Value | Notes |
|-------|-------|-------|
| `EI_CLASS` | `2` (ELFCLASS64) | 64-bit ELF |
| `EI_DATA` | `1` (ELFDATA2LSB) | Little-endian |
| `e_machine` | `62` (EM_X86_64) | AMD x86-64 architecture |
| `e_flags` | `0x00000000` | No architecture-specific flags |
| `e_ehsize` | `64` | 64-byte ELF header |
| `e_phentsize` | `56` | Program header entry size (Elf64_Phdr) |
| `e_shentsize` | `64` | Section header entry size (Elf64_Shdr) |

#### i686

| Field | Value | Notes |
|-------|-------|-------|
| `EI_CLASS` | `1` (ELFCLASS32) | 32-bit ELF |
| `EI_DATA` | `1` (ELFDATA2LSB) | Little-endian |
| `e_machine` | `3` (EM_386) | Intel 80386 |
| `e_flags` | `0x00000000` | No architecture-specific flags |
| `e_ehsize` | `52` | 52-byte ELF header |
| `e_phentsize` | `32` | Program header entry size (Elf32_Phdr) |
| `e_shentsize` | `40` | Section header entry size (Elf32_Shdr) |

#### AArch64

| Field | Value | Notes |
|-------|-------|-------|
| `EI_CLASS` | `2` (ELFCLASS64) | 64-bit ELF |
| `EI_DATA` | `1` (ELFDATA2LSB) | Little-endian |
| `e_machine` | `183` (EM_AARCH64) | ARM AARCH64 |
| `e_flags` | `0x00000000` | No architecture-specific flags |
| `e_ehsize` | `64` | 64-byte ELF header |
| `e_phentsize` | `56` | Program header entry size (Elf64_Phdr) |
| `e_shentsize` | `64` | Section header entry size (Elf64_Shdr) |

#### RISC-V 64

| Field | Value | Notes |
|-------|-------|-------|
| `EI_CLASS` | `2` (ELFCLASS64) | 64-bit ELF |
| `EI_DATA` | `1` (ELFDATA2LSB) | Little-endian |
| `e_machine` | `243` (EM_RISCV) | RISC-V |
| `e_flags` | `0x00000005` | `EF_RISCV_RVC` (0x1) + `EF_RISCV_FLOAT_ABI_DOUBLE` (0x4) for RV64IMAFDC / LP64D |
| `e_ehsize` | `64` | 64-byte ELF header |
| `e_phentsize` | `56` | Program header entry size (Elf64_Phdr) |
| `e_shentsize` | `64` | Section header entry size (Elf64_Shdr) |

### 2.3 Common Header Fields

| Field | ET_EXEC Value | ET_DYN Value | Description |
|-------|---------------|--------------|-------------|
| `e_type` | `2` (ET_EXEC) | `3` (ET_DYN) | Object file type |
| `e_version` | `1` (EV_CURRENT) | `1` (EV_CURRENT) | Object file version |
| `e_entry` | Virtual address of `_start` | `0` (or entry if specified) | Entry point address |
| `e_phoff` | Offset to program header table | Offset to program header table | Program header table offset |
| `e_shoff` | Offset to section header table | Offset to section header table | Section header table offset |
| `e_phnum` | Number of program headers | Number of program headers | Program header count |
| `e_shnum` | Number of section headers | Number of section headers | Section header count |
| `e_shstrndx` | Index of `.shstrtab` section | Index of `.shstrtab` section | Section name string table index |

### 2.4 Implementation

The ELF header is constructed by `src/backend/elf_writer_common.rs`, which:

1. Selects the correct `EI_CLASS`, `EI_DATA`, and `e_machine` based on the target from `src/common/target.rs`
2. Sets `e_type` based on whether `-shared` was specified (ET_DYN) or not (ET_EXEC)
3. Computes `e_entry` from the `_start` symbol address for ET_EXEC, or zero for ET_DYN
4. Fills `e_phoff`, `e_shoff`, `e_phnum`, `e_shnum`, `e_shstrndx` after all sections and segments are laid out

---

## 3. Section Layout

BCC produces ELF files with sections organized according to a fixed ordering convention. The linker (`src/backend/linker_common/section_merger.rs`) merges input sections from relocatable objects and arranges them into the output file.

### 3.1 Standard Sections

#### `.text` — Executable Code

| Property | Value |
|----------|-------|
| Type | `SHT_PROGBITS` (1) |
| Flags | `SHF_ALLOC \| SHF_EXECINSTR` (0x6) |
| Alignment | Architecture-dependent (typically 16 bytes for x86, 4 bytes for ARM/RISC-V) |
| Segment | `PT_LOAD` with `PF_R \| PF_X` |

Contains compiled machine code for all functions. When `-fPIC` is active, code uses PC-relative addressing with GOT/PLT indirection for external symbol references.

#### `.rodata` — Read-Only Data

| Property | Value |
|----------|-------|
| Type | `SHT_PROGBITS` (1) |
| Flags | `SHF_ALLOC` (0x2) |
| Alignment | Varies by content (string literals: 1, larger constants: natural alignment) |
| Segment | `PT_LOAD` with `PF_R` |

Contains string literals, floating-point constants, jump tables, and other read-only data. Non-UTF-8 bytes in string literals are preserved with byte-exact fidelity via the PUA encoding round-trip (encoded during preprocessing, decoded during code generation).

#### `.data` — Initialized Writable Data

| Property | Value |
|----------|-------|
| Type | `SHT_PROGBITS` (1) |
| Flags | `SHF_ALLOC \| SHF_WRITE` (0x3) |
| Alignment | Natural alignment of largest element |
| Segment | `PT_LOAD` with `PF_R \| PF_W` |

Contains initialized global and static variables with non-zero initial values.

#### `.bss` — Zero-Initialized Data

| Property | Value |
|----------|-------|
| Type | `SHT_NOBITS` (8) |
| Flags | `SHF_ALLOC \| SHF_WRITE` (0x3) |
| Alignment | Natural alignment of largest element |
| Segment | `PT_LOAD` with `PF_R \| PF_W` (same segment as `.data`) |

Contains zero-initialized global and static variables. `SHT_NOBITS` means this section occupies no space in the file — the operating system zero-fills the memory at load time. The `.bss` section is placed after `.data` within the same writable PT_LOAD segment.

#### `.symtab` — Symbol Table

| Property | Value |
|----------|-------|
| Type | `SHT_SYMTAB` (2) |
| Flags | `0` (not loaded into memory) |
| Entry size | 16 bytes (Elf32_Sym) or 24 bytes (Elf64_Sym) |
| Link | Index of associated `.strtab` section |
| Info | Index of first non-local symbol (one past last `STB_LOCAL`) |

The complete symbol table containing all symbols: local, global, and weak. See [Section 5](#5-symbol-tables) for symbol table details. Not loaded into process memory at runtime; used by debuggers and analysis tools.

#### `.strtab` — String Table

| Property | Value |
|----------|-------|
| Type | `SHT_STRTAB` (3) |
| Flags | `0` (not loaded into memory) |

Contains null-terminated strings referenced by `.symtab` entries. The first byte is always `0x00` (the null string). Symbol names are stored sequentially, each terminated by a null byte.

#### `.shstrtab` — Section Header String Table

| Property | Value |
|----------|-------|
| Type | `SHT_STRTAB` (3) |
| Flags | `0` (not loaded into memory) |

Contains section names (`.text`, `.data`, `.rodata`, etc.) referenced by section header `sh_name` fields. The ELF header's `e_shstrndx` points to this section's index.

#### `.note` — Note Sections

| Property | Value |
|----------|-------|
| Type | `SHT_NOTE` (7) |
| Flags | `SHF_ALLOC` (0x2) |

Contains structured note entries. BCC emits a `.note.GNU-stack` section to signal non-executable stack preference to the runtime loader.

### 3.2 Section Ordering

BCC arranges output sections in the following canonical order:

```
ELF Header
Program Header Table

[Loaded Sections — in PT_LOAD segment order]
  .interp            (if ET_DYN)
  .note.GNU-stack
  .gnu.hash          (if ET_DYN)
  .dynsym            (if ET_DYN)
  .dynstr            (if ET_DYN)
  .rela.dyn          (if ET_DYN)
  .rela.plt          (if ET_DYN)
  .plt               (if ET_DYN)
  .text
  .rodata
  .dynamic           (if ET_DYN)
  .got               (if PIC)
  .got.plt           (if ET_DYN)
  .data
  .bss

[Non-Loaded Sections]
  .symtab
  .strtab
  .shstrtab
  .debug_info        (if -g)
  .debug_abbrev      (if -g)
  .debug_line        (if -g)
  .debug_str         (if -g)

Section Header Table
```

### 3.3 Alignment Rules

- Sections within a `PT_LOAD` segment are page-aligned at segment boundaries (typically 4096 bytes, or 65536 bytes on AArch64 with 64K pages)
- Within a segment, sections are packed with their natural alignment, padded as needed
- The `.bss` section always follows `.data` without file padding (since `.bss` is `SHT_NOBITS`)
- The section header table is aligned to the architecture's natural word size (4 or 8 bytes)

### 3.4 Custom Sections

The `__attribute__((section("name")))` GCC extension allows placement of functions and variables in named ELF sections. BCC honors these by creating output sections with the specified names, applying default flags based on the symbol type:

- Functions → `SHF_ALLOC | SHF_EXECINSTR`
- Writable data → `SHF_ALLOC | SHF_WRITE`
- Read-only data → `SHF_ALLOC`

The Linux kernel extensively uses custom sections (`.init.text`, `.init.data`, `.exit.text`, `.modinfo`, etc.), so this support is critical for Checkpoint 6 (kernel build).

---

## 4. Program Headers

Program headers describe runtime segments. They are present in executables (ET_EXEC) and shared objects (ET_DYN) but not in relocatable objects (ET_REL / `.o` files). The program header table immediately follows the ELF header.

### 4.1 PT_PHDR — Program Header Table

| Field | Value |
|-------|-------|
| `p_type` | `6` (PT_PHDR) |
| `p_flags` | `PF_R` (0x4) |
| `p_offset` | Offset of the program header table in the file |
| `p_vaddr` | Virtual address of the program header table |
| `p_filesz` | `e_phnum * e_phentsize` |
| `p_memsz` | Same as `p_filesz` |
| `p_align` | Architecture word size (4 or 8) |

Self-referencing entry describing the program header table itself. Required for the dynamic linker to locate the program header table in memory.

### 4.2 PT_INTERP — Dynamic Linker Path

| Field | Value |
|-------|-------|
| `p_type` | `3` (PT_INTERP) |
| `p_flags` | `PF_R` (0x4) |
| `p_offset` | Offset of `.interp` section |
| `p_filesz` | Length of interpreter path string (including null terminator) |
| `p_memsz` | Same as `p_filesz` |

Present only in dynamically linked executables and shared objects that require a dynamic linker. Contains a null-terminated path to the ELF interpreter (dynamic linker).

**Per-Architecture Dynamic Linker Paths:**

| Architecture | Interpreter Path |
|-------------|-----------------|
| x86-64 | `/lib64/ld-linux-x86-64.so.2` |
| i686 | `/lib/ld-linux.so.2` |
| AArch64 | `/lib/ld-linux-aarch64.so.1` |
| RISC-V 64 | `/lib/ld-linux-riscv64-lp64d.so.1` |

### 4.3 PT_LOAD — Loadable Segments

PT_LOAD segments describe contiguous regions of the file that are mapped into process memory. BCC emits the following PT_LOAD segments:

#### Executable Segment (Code)

| Field | Value |
|-------|-------|
| `p_type` | `1` (PT_LOAD) |
| `p_flags` | `PF_R \| PF_X` (0x5) |
| `p_align` | Page size (typically `0x1000` = 4096) |
| Sections | `.text`, `.plt` (if present) |

Contains all executable code. The `.plt` stubs (Procedure Linkage Table) are placed in this segment when dynamic linking is active.

#### Read-Only Data Segment

| Field | Value |
|-------|-------|
| `p_type` | `1` (PT_LOAD) |
| `p_flags` | `PF_R` (0x4) |
| `p_align` | Page size |
| Sections | `.rodata`, `.gnu.hash`, `.dynsym`, `.dynstr`, `.rela.dyn`, `.rela.plt` (if present) |

Contains read-only data. Dynamic linking metadata sections (`.gnu.hash`, `.dynsym`, `.dynstr`, relocation tables) are placed here because they are read-only at runtime.

#### Writable Data Segment

| Field | Value |
|-------|-------|
| `p_type` | `1` (PT_LOAD) |
| `p_flags` | `PF_R \| PF_W` (0x6) |
| `p_align` | Page size |
| Sections | `.data`, `.bss`, `.got`, `.got.plt`, `.dynamic` (if present) |

Contains all writable data. The GOT (Global Offset Table) is writable because the dynamic linker patches it at load time. The `.dynamic` section is placed here for the same reason. The `.bss` section extends the memory size (`p_memsz`) beyond the file size (`p_filesz`) — the difference is zero-filled by the OS.

### 4.4 PT_DYNAMIC — Dynamic Section

| Field | Value |
|-------|-------|
| `p_type` | `2` (PT_DYNAMIC) |
| `p_flags` | `PF_R \| PF_W` (0x6) |
| `p_offset` | Offset of `.dynamic` section |
| `p_filesz` | Size of `.dynamic` section |
| `p_memsz` | Same as `p_filesz` |

Present only in ET_DYN shared objects and dynamically linked ET_EXEC executables. Points to the `.dynamic` section containing the dynamic linking information table. See [Section 6](#6-dynamic-linking-structures) for the `.dynamic` section contents.

### 4.5 PT_GNU_STACK — Stack Executability

| Field | Value |
|-------|-------|
| `p_type` | `0x6474e551` (PT_GNU_STACK) |
| `p_flags` | `PF_R \| PF_W` (0x6) — non-executable stack |
| `p_offset` | `0` |
| `p_filesz` | `0` |
| `p_memsz` | `0` |

Signals to the Linux kernel and dynamic linker that the program does not require an executable stack. BCC always emits this segment with `PF_R | PF_W` (no `PF_X`), enforcing W^X (Write XOR Execute) security policy.

### 4.6 Segment Layout Summary

**ET_EXEC (static executable):**

```
PT_PHDR     → program header table (PF_R)
PT_LOAD     → .text (PF_R|PF_X)
PT_LOAD     → .rodata (PF_R)
PT_LOAD     → .data, .bss (PF_R|PF_W)
PT_GNU_STACK → (PF_R|PF_W, non-executable)
```

**ET_DYN (shared object):**

```
PT_PHDR      → program header table (PF_R)
PT_INTERP    → .interp (PF_R) [path to dynamic linker]
PT_LOAD      → .text, .plt (PF_R|PF_X)
PT_LOAD      → .rodata, .gnu.hash, .dynsym, .dynstr, .rela.dyn, .rela.plt (PF_R)
PT_LOAD      → .data, .bss, .got, .got.plt, .dynamic (PF_R|PF_W)
PT_DYNAMIC   → .dynamic (PF_R|PF_W)
PT_GNU_STACK → (PF_R|PF_W, non-executable)
```

### 4.7 Implementation

Program headers are generated by `src/backend/linker_common/linker_script.rs`, which:

1. Assigns virtual addresses to all output sections
2. Groups sections into PT_LOAD segments by permission flags
3. Computes `p_offset`, `p_vaddr`, `p_paddr`, `p_filesz`, `p_memsz` for each segment
4. Ensures page-aligned segment boundaries
5. Adds PT_PHDR, PT_INTERP, PT_DYNAMIC, and PT_GNU_STACK as needed

---

## 5. Symbol Tables

BCC produces two types of symbol tables: the static symbol table (`.symtab`) for debuggers and tools, and the dynamic symbol table (`.dynsym`) for the runtime dynamic linker.

### 5.1 Symbol Table Entry Format

**Elf64_Sym (24 bytes):**

| Offset | Field | Size | Description |
|--------|-------|------|-------------|
| 0 | `st_name` | 4 bytes | Offset into string table (`.strtab` or `.dynstr`) |
| 4 | `st_info` | 1 byte | Symbol type (lower 4 bits) and binding (upper 4 bits) |
| 5 | `st_other` | 1 byte | Symbol visibility (lower 2 bits) |
| 6 | `st_shndx` | 2 bytes | Section index where symbol is defined |
| 8 | `st_value` | 8 bytes | Symbol value (address for defined symbols) |
| 16 | `st_size` | 8 bytes | Symbol size in bytes |

**Elf32_Sym (16 bytes):**

| Offset | Field | Size | Description |
|--------|-------|------|-------------|
| 0 | `st_name` | 4 bytes | Offset into string table |
| 4 | `st_value` | 4 bytes | Symbol value |
| 8 | `st_size` | 4 bytes | Symbol size |
| 12 | `st_info` | 1 byte | Type and binding |
| 13 | `st_other` | 1 byte | Visibility |
| 14 | `st_shndx` | 2 bytes | Section index |

### 5.2 Symbol Binding (`STB_*`)

| Binding | Value | Description |
|---------|-------|-------------|
| `STB_LOCAL` | 0 | Symbol not visible outside the object file. Static functions and variables. |
| `STB_GLOBAL` | 1 | Symbol visible to all object files. Extern functions and global variables. |
| `STB_WEAK` | 2 | Like `STB_GLOBAL`, but may be overridden by a `STB_GLOBAL` definition. Set by `__attribute__((weak))`. |

In `.symtab`, all `STB_LOCAL` symbols precede all `STB_GLOBAL` and `STB_WEAK` symbols. The section header's `sh_info` field records the index of the first non-local symbol.

### 5.3 Symbol Types (`STT_*`)

| Type | Value | Description |
|------|-------|-------------|
| `STT_NOTYPE` | 0 | Unspecified type |
| `STT_OBJECT` | 1 | Data object (global/static variable) |
| `STT_FUNC` | 2 | Function entry point |
| `STT_SECTION` | 3 | Section symbol (one per section, used in relocations) |
| `STT_FILE` | 4 | Source file name (always `STB_LOCAL`, `st_shndx = SHN_ABS`) |

### 5.4 Symbol Visibility (`STV_*`)

| Visibility | Value | Description |
|------------|-------|-------------|
| `STV_DEFAULT` | 0 | Default visibility — symbol is visible according to its binding |
| `STV_HIDDEN` | 2 | Symbol is not visible to other shared objects. Set by `__attribute__((visibility("hidden")))` |
| `STV_PROTECTED` | 3 | Symbol is visible but cannot be preempted. Set by `__attribute__((visibility("protected")))` |

Symbol visibility is controlled by the `visibility` GCC attribute and affects dynamic linking behavior:

- `STV_DEFAULT` symbols in shared objects are exported in `.dynsym` and can be interposed
- `STV_HIDDEN` symbols are stripped from `.dynsym` — invisible to other shared objects
- `STV_PROTECTED` symbols are in `.dynsym` but the dynamic linker will not allow interposition

### 5.5 Static Symbol Table (`.symtab`)

The `.symtab` section contains every symbol in the linked output:

- `STT_FILE` entries for each input source file
- `STT_SECTION` entries for each output section
- `STB_LOCAL` function and object symbols (static linkage)
- `STB_GLOBAL` and `STB_WEAK` function and object symbols (external linkage)

The associated string table is `.strtab`. Not loaded at runtime.

### 5.6 Dynamic Symbol Table (`.dynsym`)

Present only in ET_DYN shared objects. Contains only symbols needed for dynamic linking:

- Exported symbols (functions and variables with `STV_DEFAULT` or `STV_PROTECTED` visibility)
- Imported symbols (undefined references that must be resolved at load time)

The associated string table is `.dynstr`. Loaded at runtime and used by the dynamic linker for symbol resolution.

### 5.7 Implementation

Symbol tables are built by `src/backend/linker_common/symbol_resolver.rs`, which performs two-pass resolution:

1. **Collection pass:** Gather all symbol definitions and references from input objects
2. **Resolution pass:** Match references to definitions using strong/weak binding rules:
   - A `STB_GLOBAL` definition satisfies a `STB_GLOBAL` reference
   - A `STB_WEAK` definition is overridden if a `STB_GLOBAL` definition of the same name exists
   - Undefined `STB_GLOBAL` symbols at link time produce an error (for ET_EXEC) or remain as dynamic imports (for ET_DYN)
   - Undefined `STB_WEAK` symbols are silently resolved to zero

---

## 6. Dynamic Linking Structures

When BCC produces a shared object (`-shared` flag, ET_DYN), it generates a full set of dynamic linking sections. These sections enable the Linux dynamic linker (`ld-linux-*.so`) to resolve symbols, apply relocations, and bind function calls at load time.

**Implementation file:** `src/backend/linker_common/dynamic.rs`

### 6.1 `.dynamic` Section

The `.dynamic` section contains an array of `Elf64_Dyn` (or `Elf32_Dyn`) entries, each consisting of a tag and a value. The table is terminated by a `DT_NULL` entry.

| Tag | Value | Description |
|-----|-------|-------------|
| `DT_NEEDED` | 1 | Name of a required shared library (offset into `.dynstr`). One entry per `-l` library. |
| `DT_SONAME` | 14 | Shared object name (offset into `.dynstr`). Set when producing a shared library. |
| `DT_SYMTAB` | 6 | Address of `.dynsym` section |
| `DT_STRTAB` | 5 | Address of `.dynstr` section |
| `DT_STRSZ` | 10 | Size of `.dynstr` in bytes |
| `DT_SYMENT` | 11 | Size of one `.dynsym` entry (24 bytes for Elf64, 16 bytes for Elf32) |
| `DT_HASH` | 4 | Address of `.gnu.hash` section (BCC uses GNU hash, not SYSV hash) |
| `DT_GNU_HASH` | 0x6ffffef5 | Address of `.gnu.hash` section |
| `DT_RELA` | 7 | Address of `.rela.dyn` section |
| `DT_RELASZ` | 8 | Total size of `.rela.dyn` in bytes |
| `DT_RELAENT` | 9 | Size of one relocation entry (24 bytes for Elf64_Rela) |
| `DT_JMPREL` | 23 | Address of `.rela.plt` section |
| `DT_PLTRELSZ` | 2 | Total size of `.rela.plt` in bytes |
| `DT_PLTREL` | 20 | Type of PLT relocations (`DT_RELA` = 7) |
| `DT_PLTGOT` | 3 | Address of `.got.plt` section |
| `DT_INIT` | 12 | Address of initialization function (if `__attribute__((constructor))` is used) |
| `DT_FINI` | 13 | Address of finalization function (if `__attribute__((destructor))` is used) |
| `DT_INIT_ARRAY` | 25 | Address of `.init_array` section (constructor function pointers) |
| `DT_INIT_ARRAYSZ` | 27 | Size of `.init_array` in bytes |
| `DT_FINI_ARRAY` | 26 | Address of `.fini_array` section (destructor function pointers) |
| `DT_FINI_ARRAYSZ` | 28 | Size of `.fini_array` in bytes |
| `DT_FLAGS` | 30 | Flags (e.g., `DF_SYMBOLIC`, `DF_TEXTREL`, `DF_BIND_NOW`) |
| `DT_FLAGS_1` | 0x6ffffffb | Extended flags (e.g., `DF_1_NOW`, `DF_1_PIE`) |
| `DT_NULL` | 0 | Marks end of dynamic table |

### 6.2 `.dynsym` — Dynamic Symbol Table

Dynamic symbol table containing only symbols relevant to dynamic linking. Structure is identical to `.symtab` (see [Section 5.1](#51-symbol-table-entry-format)) but contains a subset:

- Exported function and object symbols
- Imported (undefined) symbols referenced from code
- The first entry (index 0) is always the undefined null symbol

### 6.3 `.dynstr` — Dynamic String Table

String table for symbol names referenced by `.dynsym` and other dynamic sections (`.dynamic` entries like `DT_NEEDED`, `DT_SONAME`). Format is identical to `.strtab`: null-terminated strings with a leading null byte.

### 6.4 `.rela.dyn` — Dynamic Relocations

Contains relocations that must be applied by the dynamic linker at load time for non-PLT references (e.g., GOT entries for global variables, absolute address references in data sections).

**Entry format (Elf64_Rela, 24 bytes):**

| Offset | Field | Size | Description |
|--------|-------|------|-------------|
| 0 | `r_offset` | 8 bytes | Address where the relocation applies (typically a GOT slot) |
| 8 | `r_info` | 8 bytes | Symbol index (upper 32 bits) + relocation type (lower 32 bits) |
| 16 | `r_addend` | 8 bytes | Constant addend for the relocation computation |

Common dynamic relocation types:

| Architecture | Type | Description |
|-------------|------|-------------|
| x86-64 | `R_X86_64_GLOB_DAT` | GOT entry for a data symbol |
| x86-64 | `R_X86_64_RELATIVE` | Base-relative adjustment for PIC |
| x86-64 | `R_X86_64_64` | Absolute 64-bit address |
| i686 | `R_386_GLOB_DAT` | GOT entry for a data symbol |
| i686 | `R_386_RELATIVE` | Base-relative adjustment |
| AArch64 | `R_AARCH64_GLOB_DAT` | GOT entry for a data symbol |
| AArch64 | `R_AARCH64_RELATIVE` | Base-relative adjustment |
| RISC-V 64 | `R_RISCV_64` | Absolute 64-bit address |
| RISC-V 64 | `R_RISCV_RELATIVE` | Base-relative adjustment |

### 6.5 `.rela.plt` — PLT Relocations

Contains relocations for PLT (Procedure Linkage Table) entries, supporting lazy symbol binding. Each entry corresponds to one PLT stub and one `.got.plt` slot.

Common PLT relocation types:

| Architecture | Type | Description |
|-------------|------|-------------|
| x86-64 | `R_X86_64_JUMP_SLOT` | PLT GOT entry for a function symbol |
| i686 | `R_386_JMP_SLOT` | PLT GOT entry for a function symbol |
| AArch64 | `R_AARCH64_JUMP_SLOT` | PLT GOT entry for a function symbol |
| RISC-V 64 | `R_RISCV_JUMP_SLOT` | PLT GOT entry for a function symbol |

### 6.6 `.gnu.hash` — GNU Hash Table

The `.gnu.hash` section provides fast symbol lookup in `.dynsym` using a Bloom filter and hash buckets. It replaces the older `DT_HASH` (SYSV hash) with better performance characteristics.

**Layout:**

```
┌─────────────────────┐
│ nbuckets (4 bytes)   │  Number of hash buckets
│ symndx (4 bytes)     │  Index of first hashed symbol in .dynsym
│ maskwords (4 bytes)  │  Number of Bloom filter words
│ shift2 (4 bytes)     │  Bloom filter shift count
├─────────────────────┤
│ Bloom filter         │  maskwords × (4 or 8) bytes
├─────────────────────┤
│ Hash buckets         │  nbuckets × 4 bytes
├─────────────────────┤
│ Hash values          │  One 32-bit hash per hashed symbol
└─────────────────────┘
```

The hash function is the standard GNU hash: for each byte `c` in the symbol name, `h = h * 33 + c`. The Bloom filter provides a fast rejection test before bucket lookup.

### 6.7 `.got` — Global Offset Table

The GOT contains addresses of global data symbols. In PIC code, data references go through the GOT so that the dynamic linker can patch the correct absolute addresses at load time.

| Property | Value |
|----------|-------|
| Type | `SHT_PROGBITS` |
| Flags | `SHF_ALLOC \| SHF_WRITE` |
| Entry size | Architecture word size (4 or 8 bytes) |

Each GOT entry initially contains zero or a base-relative offset; the dynamic linker fills in the actual address at load time via `R_*_GLOB_DAT` relocations.

### 6.8 `.got.plt` — GOT for PLT

A specialized portion of the GOT dedicated to PLT entries. The first three entries are reserved:

| Index | Content |
|-------|---------|
| 0 | Address of `.dynamic` section |
| 1 | Pointer to link_map structure (filled by dynamic linker) |
| 2 | Address of `_dl_runtime_resolve` (filled by dynamic linker) |
| 3+ | Function addresses — initially point to PLT stub's fallback code |

On the first call through a PLT stub, the `.got.plt` entry points back into the PLT, which pushes the relocation index and jumps to `_dl_runtime_resolve`. After resolution, the `.got.plt` entry is patched with the resolved function address, so subsequent calls go directly to the target (lazy binding).

### 6.9 `.plt` — Procedure Linkage Table

The PLT provides indirect function call stubs for external function references. Each PLT entry is a short code sequence that loads the target address from the corresponding `.got.plt` slot and jumps to it.

**PLT stub structure varies by architecture:**

#### x86-64 PLT Entry (16 bytes)

```asm
jmp    *got_plt_offset(%rip)   # Jump through GOT.PLT entry
push   $relocation_index        # Push relocation index for resolver
jmp    plt[0]                   # Jump to PLT header (resolver trampoline)
```

#### i686 PLT Entry (16 bytes)

```asm
jmp    *got_plt_address         # Jump through GOT.PLT entry
push   $relocation_index        # Push relocation index
jmp    plt[0]                   # Jump to PLT header
```

#### AArch64 PLT Entry (16 bytes)

```asm
adrp   x16, got_plt_page        # Load page of GOT.PLT entry
ldr    x17, [x16, got_plt_off]  # Load GOT.PLT entry
add    x16, x16, got_plt_off    # Compute GOT.PLT address for resolver
br     x17                      # Branch to target
```

#### RISC-V 64 PLT Entry (16 bytes)

```asm
auipc  t3, got_plt_hi20         # Load upper bits of GOT.PLT address
ld     t3, got_plt_lo12(t3)     # Load GOT.PLT entry
jalr   t1, t3                   # Jump to target, link register for resolver
nop                              # Padding
```

### 6.10 Section Presence Summary

| Section | ET_EXEC (static) | ET_DYN (shared) |
|---------|-------------------|-----------------|
| `.dynamic` | No | **Yes** |
| `.dynsym` | No | **Yes** |
| `.dynstr` | No | **Yes** |
| `.rela.dyn` | No | **Yes** (if dynamic relocations needed) |
| `.rela.plt` | No | **Yes** (if PLT entries exist) |
| `.gnu.hash` | No | **Yes** |
| `.got` | No (or minimal for PIC ET_EXEC) | **Yes** |
| `.got.plt` | No | **Yes** |
| `.plt` | No | **Yes** |
| `.interp` | No (statically linked) | **Yes** |

---

## 7. Relocation Types Per Architecture

Relocations instruct the linker on how to patch code and data references during linking. BCC uses `Rela`-style relocations (with explicit addend) on all 64-bit architectures and `Rel`-style (implicit addend) on i686.

**Implementation files:**
- Assembler-emitted relocations: `src/backend/{arch}/assembler/relocations.rs`
- Linker relocation application: `src/backend/{arch}/linker/relocations.rs`
- Common relocation framework: `src/backend/linker_common/relocation.rs`

### 7.1 x86-64 Relocations

| Type | Value | Calculation | Description |
|------|-------|-------------|-------------|
| `R_X86_64_NONE` | 0 | — | No relocation |
| `R_X86_64_64` | 1 | `S + A` | Absolute 64-bit address |
| `R_X86_64_PC32` | 2 | `S + A - P` | 32-bit PC-relative offset |
| `R_X86_64_GOT32` | 3 | `G + A` | 32-bit GOT entry offset |
| `R_X86_64_PLT32` | 4 | `L + A - P` | 32-bit PLT entry PC-relative |
| `R_X86_64_GLOB_DAT` | 6 | `S` | GOT entry for data symbol (dynamic) |
| `R_X86_64_JUMP_SLOT` | 7 | `S` | PLT GOT entry (dynamic, lazy binding) |
| `R_X86_64_RELATIVE` | 8 | `B + A` | Base-relative (dynamic) |
| `R_X86_64_GOTPCREL` | 9 | `G + GOT + A - P` | 32-bit PC-relative GOT offset |
| `R_X86_64_32` | 10 | `S + A` | Absolute 32-bit (truncated, zero-extended) |
| `R_X86_64_32S` | 11 | `S + A` | Absolute 32-bit (truncated, sign-extended) |
| `R_X86_64_16` | 12 | `S + A` | Absolute 16-bit |
| `R_X86_64_PC16` | 13 | `S + A - P` | 16-bit PC-relative |
| `R_X86_64_8` | 14 | `S + A` | Absolute 8-bit |
| `R_X86_64_PC8` | 15 | `S + A - P` | 8-bit PC-relative |
| `R_X86_64_GOTPCRELX` | 41 | `G + GOT + A - P` | Relaxable GOT-PC-relative (for `mov` optimization) |
| `R_X86_64_REX_GOTPCRELX` | 42 | `G + GOT + A - P` | Relaxable GOT-PC-relative with REX prefix |

**Legend:** `S` = symbol value, `A` = addend, `P` = relocation location address, `G` = GOT entry offset, `B` = base address, `L` = PLT entry address, `GOT` = GOT base address.

**Relaxation:** `R_X86_64_GOTPCRELX` and `R_X86_64_REX_GOTPCRELX` allow the linker to optimize GOT-indirect references to direct references when the symbol is defined locally. The `mov` instruction loading from GOT can be converted to a `lea` computing the address directly.

### 7.2 i686 Relocations

| Type | Value | Calculation | Description |
|------|-------|-------------|-------------|
| `R_386_NONE` | 0 | — | No relocation |
| `R_386_32` | 1 | `S + A` | Absolute 32-bit address |
| `R_386_PC32` | 2 | `S + A - P` | 32-bit PC-relative offset |
| `R_386_GOT32` | 3 | `G + A` | 32-bit GOT entry offset from GOT base |
| `R_386_PLT32` | 4 | `L + A - P` | 32-bit PLT entry PC-relative |
| `R_386_COPY` | 5 | — | Copy data from shared object (dynamic) |
| `R_386_GLOB_DAT` | 6 | `S` | GOT entry for data symbol (dynamic) |
| `R_386_JMP_SLOT` | 7 | `S` | PLT GOT entry (dynamic, lazy binding) |
| `R_386_RELATIVE` | 8 | `B + A` | Base-relative (dynamic) |
| `R_386_GOTOFF` | 9 | `S + A - GOT` | Offset from GOT base to symbol |
| `R_386_GOTPC` | 10 | `GOT + A - P` | PC-relative offset to GOT base |
| `R_386_32PLT` | 11 | `L + A` | Absolute 32-bit PLT address |

Note: i686 uses `Rel`-style relocations (no explicit addend field; the addend is encoded in the instruction or data at the relocation site).

### 7.3 AArch64 Relocations

| Type | Value | Calculation | Description |
|------|-------|-------------|-------------|
| `R_AARCH64_NONE` | 0 | — | No relocation |
| `R_AARCH64_ABS64` | 257 | `S + A` | Absolute 64-bit address |
| `R_AARCH64_ABS32` | 258 | `S + A` | Absolute 32-bit address (truncated) |
| `R_AARCH64_ABS16` | 259 | `S + A` | Absolute 16-bit address (truncated) |
| `R_AARCH64_PREL64` | 260 | `S + A - P` | 64-bit PC-relative |
| `R_AARCH64_PREL32` | 261 | `S + A - P` | 32-bit PC-relative |
| `R_AARCH64_PREL16` | 262 | `S + A - P` | 16-bit PC-relative |
| `R_AARCH64_ADR_PREL_PG_HI21` | 275 | `Page(S+A) - Page(P)` | ADRP: 21-bit page-relative (4K pages, bits [32:12]) |
| `R_AARCH64_ADD_ABS_LO12_NC` | 277 | `(S + A) & 0xFFF` | ADD: 12-bit page offset (no overflow check) |
| `R_AARCH64_LDST8_ABS_LO12_NC` | 278 | `(S + A) & 0xFFF` | LD/ST byte: 12-bit page offset |
| `R_AARCH64_LDST16_ABS_LO12_NC` | 284 | `(S + A) & 0xFFF` | LD/ST halfword: 12-bit page offset (scaled) |
| `R_AARCH64_LDST32_ABS_LO12_NC` | 285 | `(S + A) & 0xFFF` | LD/ST word: 12-bit page offset (scaled) |
| `R_AARCH64_LDST64_ABS_LO12_NC` | 286 | `(S + A) & 0xFFF` | LD/ST doubleword: 12-bit page offset (scaled) |
| `R_AARCH64_LDST128_ABS_LO12_NC` | 299 | `(S + A) & 0xFFF` | LD/ST quadword: 12-bit page offset (scaled) |
| `R_AARCH64_CALL26` | 283 | `S + A - P` | BL: 26-bit PC-relative function call (±128 MiB range) |
| `R_AARCH64_JUMP26` | 282 | `S + A - P` | B: 26-bit PC-relative branch (±128 MiB range) |
| `R_AARCH64_ADR_GOT_PAGE` | 311 | `Page(G(S)) - Page(P)` | ADRP to GOT entry page |
| `R_AARCH64_LD64_GOT_LO12_NC` | 312 | `G(S) & 0xFFF` | LDR from GOT entry (12-bit page offset) |
| `R_AARCH64_GLOB_DAT` | 1025 | `S + A` | GOT entry (dynamic) |
| `R_AARCH64_JUMP_SLOT` | 1026 | `S + A` | PLT GOT entry (dynamic) |
| `R_AARCH64_RELATIVE` | 1027 | `Delta(S) + A` | Base-relative (dynamic) |

**AArch64 Addressing Pattern:** AArch64 uses a two-instruction sequence for most address computations:
1. `ADRP Xd, symbol` — loads the page address (4K-aligned) of the symbol into `Xd` using `R_AARCH64_ADR_PREL_PG_HI21`
2. `ADD Xd, Xd, :lo12:symbol` — adds the 12-bit page offset using `R_AARCH64_ADD_ABS_LO12_NC`

For GOT-indirect references (PIC), the pattern becomes `ADRP`+`LDR` using `R_AARCH64_ADR_GOT_PAGE` and `R_AARCH64_LD64_GOT_LO12_NC`.

### 7.4 RISC-V 64 Relocations

| Type | Value | Calculation | Description |
|------|-------|-------------|-------------|
| `R_RISCV_NONE` | 0 | — | No relocation |
| `R_RISCV_32` | 1 | `S + A` | Absolute 32-bit |
| `R_RISCV_64` | 2 | `S + A` | Absolute 64-bit |
| `R_RISCV_RELATIVE` | 3 | `B + A` | Base-relative (dynamic) |
| `R_RISCV_COPY` | 4 | — | Copy from shared object (dynamic) |
| `R_RISCV_JUMP_SLOT` | 5 | `S` | PLT GOT entry (dynamic) |
| `R_RISCV_BRANCH` | 16 | `S + A - P` | B-type: 12-bit conditional branch (±4 KiB) |
| `R_RISCV_JAL` | 17 | `S + A - P` | J-type: 20-bit unconditional jump (±1 MiB) |
| `R_RISCV_CALL` | 18 | `S + A - P` | AUIPC+JALR pair: 32-bit PC-relative call (±2 GiB) |
| `R_RISCV_CALL_PLT` | 19 | `S + A - P` | AUIPC+JALR pair via PLT |
| `R_RISCV_GOT_HI20` | 20 | `G(S) + GOT + A - P` | AUIPC for GOT entry (upper 20 bits) |
| `R_RISCV_PCREL_HI20` | 23 | `S + A - P` | AUIPC: upper 20 bits of PC-relative offset |
| `R_RISCV_PCREL_LO12_I` | 24 | `S - P_hi` | I-type: lower 12 bits (paired with HI20 at `P_hi`) |
| `R_RISCV_PCREL_LO12_S` | 25 | `S - P_hi` | S-type: lower 12 bits (paired with HI20 at `P_hi`) |
| `R_RISCV_HI20` | 26 | `S + A` | LUI: upper 20 bits of absolute address |
| `R_RISCV_LO12_I` | 27 | `S + A` | I-type: lower 12 bits of absolute address |
| `R_RISCV_LO12_S` | 28 | `S + A` | S-type: lower 12 bits of absolute address |
| `R_RISCV_ADD32` | 35 | `V + S + A` | 32-bit add (for relaxation) |
| `R_RISCV_SUB32` | 39 | `V - S - A` | 32-bit subtract (for relaxation) |
| `R_RISCV_ALIGN` | 36 | — | Alignment directive (for relaxation) |
| `R_RISCV_RELAX` | 51 | — | Linker relaxation marker |
| `R_RISCV_SET6` | 53 | — | Set 6-bit value |
| `R_RISCV_SUB6` | 54 | — | Subtract 6-bit value |

**RISC-V Addressing Pattern:** RISC-V uses a two-instruction sequence for 32-bit PC-relative addresses:
1. `AUIPC rd, symbol` — loads `PC + (upper 20 bits << 12)` into `rd` using `R_RISCV_PCREL_HI20`
2. `ADDI rd, rd, symbol` — adds the lower 12 bits using `R_RISCV_PCREL_LO12_I`

The `R_RISCV_PCREL_LO12_I` relocation references the **address of the AUIPC instruction** (not the symbol), creating a paired-relocation dependency.

**RISC-V Linker Relaxation:** RISC-V supports linker relaxation, which allows the linker to shorten instruction sequences when the target is close enough. The `R_RISCV_RELAX` relocation marks instructions eligible for relaxation:

- `AUIPC`+`JALR` (32-bit call) can be relaxed to `JAL` (20-bit jump) if within ±1 MiB
- `AUIPC`+`ADDI` (32-bit address load) can be relaxed to a single instruction if within GP-relative range
- `LUI`+`ADDI` (absolute address) can be relaxed similarly

When relaxation shortens code, all subsequent addresses shift, requiring the `R_RISCV_ALIGN`, `R_RISCV_ADD*`, and `R_RISCV_SUB*` relocations to maintain alignment and computed offsets.

---

## 8. ET_EXEC vs ET_DYN Differences

BCC produces two ELF types. The differences between them affect virtually every aspect of the output file.

### 8.1 Comparison Table

| Aspect | ET_EXEC (Static Executable) | ET_DYN (Shared Object) |
|--------|----------------------------|------------------------|
| **`e_type`** | `2` | `3` |
| **Load addresses** | Fixed virtual addresses | Position-independent (base address varies) |
| **Code generation** | Absolute or PC-relative addressing | PIC: all external references via GOT/PLT |
| **CLI flags** | Default (no `-shared`, no `-fPIC`) | `-shared` and/or `-fPIC` |
| **Entry point** | `_start` symbol (required) | None required (0 if not specified) |
| **PT_INTERP** | Not present (statically linked) | Present (dynamic linker path) |
| **PT_DYNAMIC** | Not present | Present (references `.dynamic`) |
| **`.dynamic`** | Not present | Present (dynamic linking table) |
| **`.dynsym`** | Not present | Present (exported/imported symbols) |
| **`.dynstr`** | Not present | Present (dynamic string table) |
| **`.gnu.hash`** | Not present | Present (fast symbol lookup) |
| **`.got`** | Not present (or minimal) | Present (global data indirection) |
| **`.got.plt`** | Not present | Present (lazy function binding) |
| **`.plt`** | Not present | Present (function call stubs) |
| **`.rela.dyn`** | Not present | Present (dynamic data relocations) |
| **`.rela.plt`** | Not present | Present (PLT relocations) |
| **`.interp`** | Not present | Present (interpreter path string) |
| **Relocations** | Fully resolved at link time | Some deferred to load time |
| **Symbol visibility** | All symbols in `.symtab` | Exported symbols also in `.dynsym` |
| **Undefined symbols** | Error at link time | Allowed (resolved by dynamic linker) |
| **Weak symbols** | Resolved to 0 if undefined | Resolved at load time or to 0 |

### 8.2 ET_EXEC Details

Static executables produced by BCC:

- **All symbol references are fully resolved** at link time. No undefined symbols remain in the final binary.
- **The `_start` symbol must be defined.** This is the entry point where the Linux kernel transfers control after `execve()`. Typically provided by the C runtime startup code (e.g., `crt1.o`), or defined directly in the source.
- **Fixed virtual addresses.** The linker assigns absolute virtual addresses to all segments. The default base address is architecture-dependent:

| Architecture | Default Text Base Address |
|-------------|--------------------------|
| x86-64 | `0x400000` |
| i686 | `0x08048000` |
| AArch64 | `0x400000` |
| RISC-V 64 | `0x10000` (or `0x80000000` for kernel images) |

- **Minimal overhead.** No GOT, PLT, dynamic sections, or runtime relocation processing.

### 8.3 ET_DYN Details

Shared objects produced by BCC with `-shared`:

- **Position-independent code is required.** All code must use PC-relative or GOT/PLT-indirect addressing. The `-fPIC` flag enables this in the code generator.
- **Symbol exports are controlled by visibility.** Only symbols with `STV_DEFAULT` or `STV_PROTECTED` visibility appear in `.dynsym`. Symbols marked `__attribute__((visibility("hidden")))` are local to the shared object.
- **Lazy binding is the default.** Function calls through PLT stubs are resolved on first invocation by the dynamic linker. The `.got.plt` entries are initially set to point back into the PLT stub's resolver trampoline.
- **GOT entries for global data** are filled at load time by the dynamic linker processing `.rela.dyn` relocations.
- **Constructor and destructor functions** (`__attribute__((constructor))` / `__attribute__((destructor))`) are recorded via `DT_INIT_ARRAY` / `DT_FINI_ARRAY` entries in `.dynamic`, and executed by the dynamic linker during `dlopen()` / `dlclose()` or program startup/exit.

### 8.4 PIC Code Generation Patterns

When `-fPIC` is active, the code generator uses indirect addressing for all external references:

**Data access (global variable):**

| Architecture | Non-PIC | PIC |
|-------------|---------|-----|
| x86-64 | `mov rax, [symbol]` (absolute) | `mov rax, [rip + symbol@GOTPCREL]` (GOT-relative) |
| i686 | `mov eax, [symbol]` (absolute) | `call __x86.get_pc_thunk.bx; mov eax, [ebx + symbol@GOT]` |
| AArch64 | `adrp x0, symbol; ldr x0, [x0, :lo12:symbol]` | `adrp x0, :got:symbol; ldr x0, [x0, :got_lo12:symbol]; ldr x0, [x0]` |
| RISC-V 64 | `lui rd, %hi(symbol); ld rd, %lo(symbol)(rd)` | `auipc rd, %got_pcrel_hi(symbol); ld rd, %pcrel_lo(label)(rd)` |

**Function call (external function):**

| Architecture | Non-PIC | PIC |
|-------------|---------|-----|
| x86-64 | `call symbol` (direct) | `call symbol@PLT` (through PLT) |
| i686 | `call symbol` (direct) | `call symbol@PLT` (through PLT) |
| AArch64 | `bl symbol` (direct) | `bl symbol` (via PLT stub, linker inserts) |
| RISC-V 64 | `call symbol` (AUIPC+JALR) | `call symbol@plt` (via PLT stub) |

---

## 9. DWARF Debug Sections

When the `-g` flag is specified, BCC emits DWARF v4 debug information sections. When `-g` is **not** specified, **zero** `.debug_*` sections are present in the output — strict no-leakage policy.

**Implementation files:** `src/backend/dwarf/` (`mod.rs`, `info.rs`, `abbrev.rs`, `line.rs`, `str.rs`)

### 9.1 Debug Section Overview

| Section | Type | Purpose |
|---------|------|---------|
| `.debug_info` | `SHT_PROGBITS` | Debugging Information Entries (DIEs) describing compilation units, functions, variables, and types |
| `.debug_abbrev` | `SHT_PROGBITS` | Abbreviation table defining the structure of DIEs in `.debug_info` |
| `.debug_line` | `SHT_PROGBITS` | Line number program mapping machine code addresses to source file/line numbers |
| `.debug_str` | `SHT_PROGBITS` | String table for debug information (referenced via `DW_FORM_strp` offsets) |

All debug sections have `SHF_ALLOC` flag cleared (flags = 0) — they are not loaded into process memory. They occupy space only in the ELF file on disk.

### 9.2 `.debug_info` Structure

The `.debug_info` section contains a tree of Debugging Information Entries (DIEs):

```
Compilation Unit Header:
  unit_length     (4 bytes)  — Length of the compilation unit (excluding this field)
  version         (2 bytes)  — DWARF version (4)
  debug_abbrev_offset (4 bytes)  — Offset into .debug_abbrev
  address_size    (1 byte)   — Size of an address (4 or 8 bytes)

DIE Tree:
  DW_TAG_compile_unit
    DW_AT_producer    — "BCC (Blitzy's C Compiler)"
    DW_AT_language    — DW_LANG_C11 (0x001d)
    DW_AT_name        — Source file name
    DW_AT_comp_dir    — Compilation directory
    DW_AT_low_pc      — Lowest code address in this unit
    DW_AT_high_pc     — Highest code address (or length)
    DW_AT_stmt_list   — Offset into .debug_line

    DW_TAG_subprogram (one per function)
      DW_AT_name      — Function name
      DW_AT_low_pc    — Function start address
      DW_AT_high_pc   — Function end address (or length)
      DW_AT_type      — Return type reference

      DW_TAG_variable (one per local variable)
        DW_AT_name    — Variable name
        DW_AT_type    — Type reference
        DW_AT_location — Location description (DW_OP_fbreg+offset, DW_OP_reg*)
```

BCC emits DWARF only at `-O0` — variable locations reflect their alloca-assigned stack positions before any optimization.

### 9.3 `.debug_abbrev` Structure

The abbreviation table defines templates for DIEs in `.debug_info`:

```
Abbreviation Entry:
  abbreviation_code  (ULEB128)  — Unique code (1, 2, 3, ...)
  tag                (ULEB128)  — DW_TAG_* value
  children           (1 byte)   — DW_CHILDREN_yes (1) or DW_CHILDREN_no (0)
  attribute_specs:
    attribute_name   (ULEB128)  — DW_AT_* value
    attribute_form   (ULEB128)  — DW_FORM_* value (encoding of the attribute)
  terminator         (2 bytes)  — 0, 0

Table Terminator:
  0                  (1 byte)   — End of abbreviation table
```

### 9.4 `.debug_line` Structure

The line number program maps code addresses to source locations:

```
Line Number Program Header:
  unit_length                (4 bytes)
  version                    (2 bytes)  — 4 (DWARF v4)
  header_length              (4 bytes)
  minimum_instruction_length (1 byte)
  maximum_operations_per_insn (1 byte)  — 1 (no VLIW)
  default_is_stmt            (1 byte)   — 1
  line_base                  (1 byte)   — -5 (signed)
  line_range                 (1 byte)   — 14
  opcode_base                (1 byte)   — 13
  standard_opcode_lengths    (12 bytes) — Operand counts for opcodes 1-12
  include_directories        — Null-terminated list of directory paths
  file_names                 — Entries: name, directory index, mtime, size

Line Number Program (bytecode):
  DW_LNS_advance_pc    (1)  — Advance address
  DW_LNS_advance_line  (2)  — Advance line number
  DW_LNS_set_file      (3)  — Set current file
  DW_LNS_set_column    (4)  — Set current column
  DW_LNE_set_address   (ext) — Set absolute address
  DW_LNE_end_sequence  (ext) — End of instruction sequence
  Special opcodes            — Compact (address, line) delta encoding
```

### 9.5 `.debug_str` Structure

A simple table of null-terminated strings. DIE attributes using `DW_FORM_strp` contain 4-byte offsets into this section. String deduplication is performed so that identical strings (e.g., repeated type names) share a single entry.

### 9.6 Debug Section Conditionality

| Condition | Debug Sections Present |
|-----------|----------------------|
| Compiled with `-g` | `.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str` |
| Compiled without `-g` | **None** — zero `.debug_*` sections in the output |

This is a strict rule. The ELF writer checks the `-g` flag and conditionally includes the DWARF emitter in the output pipeline. Debug sections are never accidentally included.

---

## 10. Entry Point Convention

### 10.1 ET_EXEC — Static Executables

The ELF header's `e_entry` field is set to the virtual address of the `_start` symbol. This is the address where the Linux kernel transfers control after loading the program via `execve()`.

- The `_start` symbol **must** be defined in the linked objects. It is typically provided by the C runtime startup code (`crt1.o`) or defined directly in assembly/C source for freestanding programs.
- If `_start` is not found, the BCC linker emits an error for ET_EXEC output.
- The `-e <symbol>` flag (if supported) can override the entry point symbol name.

### 10.2 ET_DYN — Shared Objects

For shared objects produced with `-shared`:

- `e_entry` is set to `0` — shared objects do not have a default entry point.
- The dynamic linker invokes constructor functions (via `DT_INIT_ARRAY`) when the shared object is loaded, but this is not an entry point in the traditional sense.
- If the shared object is also an executable (a "PIE" — position-independent executable), then `e_entry` is set to the `_start` symbol address. BCC distinguishes PIE from shared library by the presence of `_start` in the symbol table.

### 10.3 Kernel Entry Point

For Linux kernel images (the primary validation target), the entry point is the architecture-specific kernel entry function. For RISC-V 64:

- Entry symbol: `_start` (defined in `arch/riscv/kernel/head.S`)
- The ELF `e_entry` is set to this symbol's virtual address
- The bootloader (or QEMU direct kernel boot) transfers control to this address

---

## Appendix A: Numeric Constants Reference

### ELF Header Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `ELFMAG` | `\x7fELF` | Magic number |
| `ELFCLASS32` | 1 | 32-bit ELF |
| `ELFCLASS64` | 2 | 64-bit ELF |
| `ELFDATA2LSB` | 1 | Little-endian |
| `ELFDATA2MSB` | 2 | Big-endian (not used by BCC) |
| `EV_CURRENT` | 1 | Current ELF version |
| `ELFOSABI_NONE` | 0 | No specific OS/ABI |
| `ET_EXEC` | 2 | Executable file |
| `ET_DYN` | 3 | Shared object |
| `ET_REL` | 1 | Relocatable file (`.o`) |
| `EM_386` | 3 | Intel 80386 |
| `EM_X86_64` | 62 | AMD x86-64 |
| `EM_AARCH64` | 183 | ARM AARCH64 |
| `EM_RISCV` | 243 | RISC-V |

### Section Header Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SHT_NULL` | 0 | Inactive section header |
| `SHT_PROGBITS` | 1 | Program data |
| `SHT_SYMTAB` | 2 | Symbol table |
| `SHT_STRTAB` | 3 | String table |
| `SHT_RELA` | 4 | Relocation entries with addends |
| `SHT_HASH` | 5 | Symbol hash table |
| `SHT_DYNAMIC` | 6 | Dynamic linking information |
| `SHT_NOTE` | 7 | Notes |
| `SHT_NOBITS` | 8 | Program space with no data (`.bss`) |
| `SHT_REL` | 9 | Relocation entries without addends |
| `SHT_DYNSYM` | 11 | Dynamic linker symbol table |
| `SHT_GNU_HASH` | 0x6ffffff6 | GNU hash table |
| `SHF_WRITE` | 0x1 | Writable |
| `SHF_ALLOC` | 0x2 | Occupies memory during execution |
| `SHF_EXECINSTR` | 0x4 | Executable |

### Program Header Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PT_NULL` | 0 | Unused entry |
| `PT_LOAD` | 1 | Loadable segment |
| `PT_DYNAMIC` | 2 | Dynamic linking information |
| `PT_INTERP` | 3 | Program interpreter |
| `PT_NOTE` | 4 | Auxiliary information |
| `PT_PHDR` | 6 | Program header table |
| `PT_GNU_STACK` | 0x6474e551 | Stack executability |
| `PF_X` | 0x1 | Execute permission |
| `PF_W` | 0x2 | Write permission |
| `PF_R` | 0x4 | Read permission |

### Symbol Table Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `STB_LOCAL` | 0 | Local symbol |
| `STB_GLOBAL` | 1 | Global symbol |
| `STB_WEAK` | 2 | Weak symbol |
| `STT_NOTYPE` | 0 | No type |
| `STT_OBJECT` | 1 | Data object |
| `STT_FUNC` | 2 | Function |
| `STT_SECTION` | 3 | Section symbol |
| `STT_FILE` | 4 | File symbol |
| `STV_DEFAULT` | 0 | Default visibility |
| `STV_HIDDEN` | 2 | Hidden visibility |
| `STV_PROTECTED` | 3 | Protected visibility |
| `SHN_UNDEF` | 0 | Undefined section |
| `SHN_ABS` | 0xfff1 | Absolute symbol |
| `SHN_COMMON` | 0xfff2 | Common symbol |

### Dynamic Section Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DT_NULL` | 0 | End of dynamic table |
| `DT_NEEDED` | 1 | Name of needed library |
| `DT_PLTRELSZ` | 2 | Size of PLT relocations |
| `DT_PLTGOT` | 3 | Address of GOT.PLT |
| `DT_HASH` | 4 | Address of symbol hash table |
| `DT_STRTAB` | 5 | Address of string table |
| `DT_SYMTAB` | 6 | Address of symbol table |
| `DT_RELA` | 7 | Address of Rela relocations |
| `DT_RELASZ` | 8 | Total size of Rela relocations |
| `DT_RELAENT` | 9 | Size of one Rela entry |
| `DT_STRSZ` | 10 | Size of string table |
| `DT_SYMENT` | 11 | Size of one symbol table entry |
| `DT_INIT` | 12 | Address of init function |
| `DT_FINI` | 13 | Address of fini function |
| `DT_SONAME` | 14 | Name of shared object |
| `DT_JMPREL` | 23 | Address of PLT relocations |
| `DT_INIT_ARRAY` | 25 | Address of init function array |
| `DT_FINI_ARRAY` | 26 | Address of fini function array |
| `DT_INIT_ARRAYSZ` | 27 | Size of init function array |
| `DT_FINI_ARRAYSZ` | 28 | Size of fini function array |
| `DT_FLAGS` | 30 | Flags |
| `DT_GNU_HASH` | 0x6ffffef5 | Address of GNU hash table |
| `DT_FLAGS_1` | 0x6ffffffb | State flags |

### RISC-V ELF Flags

| Constant | Value | Description |
|----------|-------|-------------|
| `EF_RISCV_RVC` | 0x0001 | Compressed (C) extension |
| `EF_RISCV_FLOAT_ABI_SOFT` | 0x0000 | Soft-float ABI |
| `EF_RISCV_FLOAT_ABI_SINGLE` | 0x0002 | Single-float ABI |
| `EF_RISCV_FLOAT_ABI_DOUBLE` | 0x0004 | Double-float ABI (LP64D) |
| `EF_RISCV_FLOAT_ABI_QUAD` | 0x0006 | Quad-float ABI |
