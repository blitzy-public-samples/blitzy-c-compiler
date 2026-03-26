# ELF Output Format Reference

BCC (Blitzy's C Compiler) produces **Linux ELF (Executable and Linkable Format) binaries exclusively**. The two supported ELF output types are:

- **ET_EXEC** — Static executables (produced by default when linking)
- **ET_DYN** — Shared objects and position-independent executables (produced with `-shared` or `-fPIC` linking)

BCC includes its own **built-in ELF writer** (`src/backend/elf_writer_common.rs`) and **built-in linker** (`src/backend/linker_common/`). No external linker (`ld`, `lld`, `gold`) is ever invoked. This is a core architectural constraint of the project: the entire ELF generation pipeline — from relocatable object files through final linked output — is hand-implemented in Rust using only the standard library (`std`), in accordance with BCC's **zero-dependency mandate**.

This document serves as the authoritative reference for the ELF output produced by BCC across all four supported target architectures: **x86-64**, **i686**, **AArch64**, and **RISC-V 64**.

---

## ELF Header

Every ELF file produced by BCC begins with a standard ELF header. The header identifies the file class, data encoding, target machine, and entry point. BCC populates the header fields based on the selected target architecture.

### Header Field Summary

| Field | x86-64 | i686 | AArch64 | RISC-V 64 |
|:------|:-------|:-----|:--------|:----------|
| `e_ident[EI_MAG0..EI_MAG3]` | `\x7fELF` | `\x7fELF` | `\x7fELF` | `\x7fELF` |
| `e_ident[EI_CLASS]` | ELFCLASS64 (2) | ELFCLASS32 (1) | ELFCLASS64 (2) | ELFCLASS64 (2) |
| `e_ident[EI_DATA]` | ELFDATA2LSB (1) | ELFDATA2LSB (1) | ELFDATA2LSB (1) | ELFDATA2LSB (1) |
| `e_ident[EI_VERSION]` | EV_CURRENT (1) | EV_CURRENT (1) | EV_CURRENT (1) | EV_CURRENT (1) |
| `e_ident[EI_OSABI]` | ELFOSABI_NONE (0) | ELFOSABI_NONE (0) | ELFOSABI_NONE (0) | ELFOSABI_NONE (0) |
| `e_type` | ET_EXEC (2) or ET_DYN (3) | ET_EXEC (2) or ET_DYN (3) | ET_EXEC (2) or ET_DYN (3) | ET_EXEC (2) or ET_DYN (3) |
| `e_machine` | EM_X86_64 (0x3E) | EM_386 (0x03) | EM_AARCH64 (0xB7) | EM_RISCV (0xF3) |
| `e_version` | EV_CURRENT (1) | EV_CURRENT (1) | EV_CURRENT (1) | EV_CURRENT (1) |
| `e_entry` | `_start` address | `_start` address | `_start` address | `_start` address |
| `e_ehsize` | 64 bytes | 52 bytes | 64 bytes | 64 bytes |
| `e_phentsize` | 56 bytes | 32 bytes | 56 bytes | 56 bytes |
| `e_shentsize` | 64 bytes | 40 bytes | 64 bytes | 64 bytes |

### Header Field Details

- **`EI_CLASS`**: Determines the file's word size. All 64-bit targets (x86-64, AArch64, RISC-V 64) use ELFCLASS64. The i686 target uses ELFCLASS32.
- **`EI_DATA`**: All four supported architectures are little-endian (ELFDATA2LSB).
- **`EI_OSABI`**: Set to ELFOSABI_NONE (0), which is the standard value for Linux ELF binaries. The Linux kernel accepts this value for all architectures.
- **`e_type`**: ET_EXEC (2) for statically linked executables; ET_DYN (3) for shared objects (`-shared`) and position-independent executables.
- **`e_machine`**: Architecture identifier — each target has a unique value as specified in the ELF standard.
- **`e_entry`**: The virtual address of the program entry point. For executables, this is the address of the `_start` symbol. For shared objects that do not define an entry point, this field is set to 0.
- **`e_flags`**: Architecture-specific flags. For RISC-V 64, this encodes the ISA extension set (e.g., RV64IMAFDC) and the floating-point ABI (LP64D). For other architectures, this is typically 0.

### RISC-V ELF Flags

For RISC-V 64 targets, the `e_flags` field encodes the following:

| Flag | Value | Meaning |
|:-----|:------|:--------|
| `EF_RISCV_RVC` | 0x0001 | Compressed (C extension) instructions present |
| `EF_RISCV_FLOAT_ABI_DOUBLE` | 0x0004 | Double-precision float ABI (LP64D) |
| `EF_RISCV_RVE` | 0x0008 | RV32E base ISA (not used for RV64) |

BCC sets `e_flags = EF_RISCV_FLOAT_ABI_DOUBLE` (0x0004) for standard RV64IMAFDC targets, or `e_flags = EF_RISCV_RVC | EF_RISCV_FLOAT_ABI_DOUBLE` (0x0005) when compressed instructions are emitted.

---

## Section Layout

BCC's linker produces ELF files with a well-defined set of sections. The sections present depend on the output type (ET_EXEC vs. ET_DYN), the presence of debug information (`-g`), and whether the output is a relocatable object file (`-c`).

### Code and Data Sections

These sections are always present in linked output:

| Section | Type | Flags | Description |
|:--------|:-----|:------|:------------|
| `.text` | SHT_PROGBITS | SHF_ALLOC \| SHF_EXECINSTR | Executable machine code for all compiled functions. This is the primary code section. |
| `.rodata` | SHT_PROGBITS | SHF_ALLOC | Read-only data including string literals, floating-point constants, jump tables, and other constant data. |
| `.data` | SHT_PROGBITS | SHF_ALLOC \| SHF_WRITE | Initialized global and static variables with non-zero initial values. |
| `.bss` | SHT_NOBITS | SHF_ALLOC \| SHF_WRITE | Uninitialized (zero-filled) global and static variables. This section occupies no file space — the loader zero-fills the memory at load time. |

### Symbol and String Tables

| Section | Type | Flags | Description |
|:--------|:-----|:------|:------------|
| `.symtab` | SHT_SYMTAB | (none) | Full symbol table containing all local and global symbols — function names, variable names, section symbols, and file symbols. Not loaded at runtime. |
| `.strtab` | SHT_STRTAB | (none) | String table for `.symtab`. Contains null-terminated symbol name strings referenced by index from symbol table entries. |
| `.shstrtab` | SHT_STRTAB | (none) | Section header string table. Contains null-terminated section name strings referenced by the `sh_name` field of each section header. |

### Constructor and Destructor Sections

| Section | Type | Flags | Description |
|:--------|:-----|:------|:------------|
| `.init_array` | SHT_INIT_ARRAY | SHF_ALLOC \| SHF_WRITE | Array of function pointers to constructors. Functions marked with `__attribute__((constructor))` have their addresses placed here. The runtime startup code iterates this array and calls each function before `main()`. |
| `.fini_array` | SHT_FINI_ARRAY | SHF_ALLOC \| SHF_WRITE | Array of function pointers to destructors. Functions marked with `__attribute__((destructor))` have their addresses placed here. The runtime shutdown code iterates this array and calls each function after `main()` returns. |

### Special Sections

| Section | Type | Flags | Description |
|:--------|:-----|:------|:------------|
| `.note.GNU-stack` | SHT_NOTE | (none) | GNU stack executability marker. BCC emits this section with no SHF_EXECINSTR flag to indicate a non-executable stack, which the Linux kernel respects when setting up the process memory map. |
| `.comment` | SHT_PROGBITS | SHF_MERGE \| SHF_STRINGS | Compiler identification string. BCC writes `BCC 0.1.0` (or the current version) to identify the producing toolchain. This section is not loaded at runtime. |

### Debug Sections (Conditional on `-g`)

When the `-g` flag is specified, BCC emits DWARF v4 debug information in the following sections. These sections are generated by the DWARF emitter (`src/backend/dwarf/`).

| Section | Type | Flags | Description |
|:--------|:-----|:------|:------------|
| `.debug_info` | SHT_PROGBITS | (none) | DWARF `.debug_info` — contains the Debugging Information Entries (DIEs) describing compilation units (`DW_TAG_compile_unit`), subprograms/functions (`DW_TAG_subprogram`), and variables (`DW_TAG_variable`). Uses DWARF version 4 format. |
| `.debug_abbrev` | SHT_PROGBITS | (none) | DWARF `.debug_abbrev` — abbreviation table defining the structure of DIEs in `.debug_info`. Each abbreviation specifies a tag, whether the DIE has children, and the list of attribute name/form pairs. |
| `.debug_line` | SHT_PROGBITS | (none) | DWARF `.debug_line` — line number program that maps machine code addresses to source file locations (file, line, column). Contains the file table, directory table, and a sequence of opcodes encoding address/line deltas. |
| `.debug_str` | SHT_PROGBITS | SHF_MERGE \| SHF_STRINGS | DWARF `.debug_str` — string table for debug information. Stores null-terminated strings (file names, function names, variable names, type names) referenced via `DW_FORM_strp` from `.debug_info`. |

**Critical constraint:** When `-g` is **not** specified, none of these `.debug_*` sections are present in the output. BCC enforces **zero debug section leakage** — a binary compiled without `-g` must contain no debug information whatsoever.

### Custom Sections

Functions and variables annotated with `__attribute__((section("name")))` are placed in the named section. BCC creates the section on demand with appropriate flags:

- Code sections (function targets): SHF_ALLOC | SHF_EXECINSTR
- Data sections (variable targets): SHF_ALLOC | SHF_WRITE (or SHF_ALLOC for const-qualified)

### Section Ordering

BCC's linker orders sections within the output ELF file following a standard convention that groups sections by their load-time permissions:

1. **ELF Header** — always at file offset 0
2. **Program Header Table** — immediately after the ELF header
3. **Read + Execute sections** — `.text` (and any custom executable sections)
4. **Read-only sections** — `.rodata`, `.note.*`, `.comment`
5. **Read + Write sections** — `.data`, `.init_array`, `.fini_array`, dynamic linking sections
6. **BSS section** — `.bss` (no file content, immediately follows `.data`)
7. **Non-loadable sections** — `.symtab`, `.strtab`, `.debug_*`
8. **Section Header Table** — at the end of the file
9. **Section Header String Table** — `.shstrtab` (referenced by the ELF header's `e_shstrndx`)

Sections within the same permission group are ordered by alignment requirements (largest alignment first) to minimize padding. The linker inserts padding between groups that require different page permissions to ensure proper page-aligned segment boundaries.

---

## Program Headers

Program headers (also called segments) tell the operating system's loader how to map the ELF file into memory. BCC generates different program headers depending on whether the output is a static executable (ET_EXEC) or a shared object / PIE (ET_DYN).

### Program Headers for Static Executables (ET_EXEC)

| Segment | Type | Flags | Contents | Description |
|:--------|:-----|:------|:---------|:------------|
| PHDR | PT_PHDR | PF_R | Program header table | Self-referencing segment that describes the program header table itself. Allows the dynamic linker to locate the program headers in memory. |
| LOAD (code) | PT_LOAD | PF_R \| PF_X | `.text` | Loadable segment containing executable code. Mapped as read + execute. |
| LOAD (rodata) | PT_LOAD | PF_R | `.rodata` | Loadable segment containing read-only data. Mapped as read-only. |
| LOAD (data) | PT_LOAD | PF_R \| PF_W | `.data`, `.bss` | Loadable segment containing read-write data. The `.bss` portion is zero-filled by the loader beyond the file-backed `.data` content. |
| GNU_STACK | PT_GNU_STACK | PF_R \| PF_W | (empty) | Declares stack permissions. BCC always emits this with PF_R \| PF_W (no PF_X), enforcing a **non-executable stack** for security. |

### Program Headers for Shared Objects and PIE (ET_DYN)

Shared objects (produced with `-shared`) and position-independent executables include additional segments for dynamic linking:

| Segment | Type | Flags | Contents | Description |
|:--------|:-----|:------|:---------|:------------|
| PHDR | PT_PHDR | PF_R | Program header table | Self-referencing segment. |
| INTERP | PT_INTERP | PF_R | `.interp` | Path to the dynamic linker (interpreter). This segment is present in PIE executables and dynamically linked executables, but **not** in pure shared objects (`.so` files that are always loaded by another program). |
| LOAD (code) | PT_LOAD | PF_R \| PF_X | `.text`, `.plt` | Executable code segment. Includes the PLT (Procedure Linkage Table) stubs. |
| LOAD (rodata) | PT_LOAD | PF_R | `.rodata`, `.gnu.hash`, `.dynsym`, `.dynstr` | Read-only data segment. Includes dynamic symbol information. |
| LOAD (data) | PT_LOAD | PF_R \| PF_W | `.data`, `.bss`, `.got`, `.got.plt`, `.dynamic` | Read-write data segment. Includes the GOT (Global Offset Table) and dynamic section. |
| DYNAMIC | PT_DYNAMIC | PF_R \| PF_W | `.dynamic` | Points to the `.dynamic` section. The dynamic linker uses this to find all other dynamic linking structures. This segment is **mandatory** for ET_DYN. |
| GNU_STACK | PT_GNU_STACK | PF_R \| PF_W | (empty) | Non-executable stack declaration. |

### Architecture-Specific Dynamic Linker Paths

When a PT_INTERP segment is present, it contains the null-terminated path to the architecture-specific dynamic linker (ELF interpreter) on Linux:

| Architecture | Dynamic Linker Path |
|:-------------|:--------------------|
| x86-64 | `/lib64/ld-linux-x86-64.so.2` |
| i686 | `/lib/ld-linux.so.2` |
| AArch64 | `/lib/ld-linux-aarch64.so.1` |
| RISC-V 64 | `/lib/ld-linux-riscv64-lp64d.so.1` |

These paths are the standard Linux dynamic linker locations for each architecture. BCC hardcodes these paths in the linker based on the selected `--target` architecture.

### Segment Alignment

All PT_LOAD segments are aligned to the system page size:

| Architecture | Page Size | Segment Alignment |
|:-------------|:----------|:------------------|
| x86-64 | 4096 bytes (4 KiB) | 0x1000 |
| i686 | 4096 bytes (4 KiB) | 0x1000 |
| AArch64 | 65536 bytes (64 KiB) | 0x10000 |
| RISC-V 64 | 4096 bytes (4 KiB) | 0x1000 |

AArch64 uses a 64 KiB alignment to support systems with 64 KiB page sizes, ensuring the binary runs correctly on all AArch64 Linux configurations.

---

## Dynamic Linking Structures

When BCC produces a shared object (`-shared`) or a dynamically linked / position-independent executable, several additional sections are generated to support runtime dynamic linking. These structures are produced by `src/backend/linker_common/dynamic.rs` and related modules.

### `.dynamic` Section

The `.dynamic` section contains an array of `ElfN_Dyn` entries, each consisting of a tag (identifying the entry type) and a value (an address or integer). This section is the root structure for all dynamic linking information.

**Key `.dynamic` entries produced by BCC:**

| Tag | Name | Value Type | Description |
|:----|:-----|:-----------|:------------|
| 1 | DT_NEEDED | String offset | Name of a required shared library (one entry per `-l` dependency). |
| 5 | DT_STRTAB | Address | Address of the `.dynstr` string table. |
| 6 | DT_SYMTAB | Address | Address of the `.dynsym` symbol table. |
| 7 | DT_RELA | Address | Address of the `.rela.dyn` relocation table (64-bit targets). |
| 8 | DT_RELASZ | Integer | Total size in bytes of the `.rela.dyn` section. |
| 9 | DT_RELAENT | Integer | Size of a single `ElfN_Rela` entry (24 bytes for 64-bit, 12 bytes for 32-bit). |
| 10 | DT_STRSZ | Integer | Size in bytes of the `.dynstr` string table. |
| 11 | DT_SYMENT | Integer | Size of a single `.dynsym` entry (24 bytes for 64-bit, 16 bytes for 32-bit). |
| 14 | DT_SONAME | String offset | Shared object name (when `-soname` is specified). |
| 17 | DT_REL | Address | Address of the `.rel.dyn` relocation table (32-bit targets, uses Rel instead of Rela). |
| 18 | DT_RELSZ | Integer | Total size of `.rel.dyn`. |
| 19 | DT_RELENT | Integer | Size of a single `ElfN_Rel` entry. |
| 20 | DT_PLTREL | Integer | Type of PLT relocations (DT_RELA = 7 for 64-bit, DT_REL = 17 for i686). |
| 23 | DT_JMPREL | Address | Address of the `.rela.plt` (or `.rel.plt`) section for PLT relocations. |
| 2 | DT_PLTRELSZ | Integer | Total size of PLT relocation entries. |
| 3 | DT_PLTGOT | Address | Address of the `.got.plt` section (processor-specific, holds PLT GOT entries). |
| 25 | DT_INIT_ARRAY | Address | Address of `.init_array` (constructor function pointer array). |
| 26 | DT_FINI_ARRAY | Address | Address of `.fini_array` (destructor function pointer array). |
| 27 | DT_INIT_ARRAYSZ | Integer | Size of `.init_array` in bytes. |
| 28 | DT_FINI_ARRAYSZ | Integer | Size of `.fini_array` in bytes. |
| 0x6ffffef5 | DT_GNU_HASH | Address | Address of the `.gnu.hash` section (GNU-style hash table). |
| 0 | DT_NULL | 0 | Sentinel entry marking the end of the `.dynamic` array. |

### `.dynsym` Section

The `.dynsym` section is the **dynamic symbol table** — it contains entries for all symbols that must be visible to the dynamic linker at runtime. This is a subset of the full `.symtab`.

**Symbol table entry structure (`ElfN_Sym`):**

| Field | Size (64-bit) | Description |
|:------|:--------------|:------------|
| `st_name` | 4 bytes | Offset into `.dynstr` for the symbol name. |
| `st_info` | 1 byte | Symbol type (function, object, etc.) and binding (local, global, weak). |
| `st_other` | 1 byte | Symbol visibility (default, hidden, protected). |
| `st_shndx` | 2 bytes | Section index where the symbol is defined (or SHN_UNDEF for imports). |
| `st_value` | 8 bytes | Symbol value (virtual address for defined symbols, 0 for undefined). |
| `st_size` | 8 bytes | Size of the symbol (e.g., function code size, variable size). |

**Symbol visibility control** via `__attribute__((visibility(...)))`:

| Visibility | `st_other` Value | Behavior |
|:-----------|:-----------------|:---------|
| `default` | STV_DEFAULT (0) | Symbol is visible to other shared objects. Can be preempted (overridden) by a symbol with the same name in another shared object. |
| `hidden` | STV_HIDDEN (2) | Symbol is **not** exported from the shared object. Not visible to the dynamic linker — resolves only within the defining shared object. |
| `protected` | STV_PROTECTED (3) | Symbol is visible to other shared objects but **cannot be preempted**. References from within the defining shared object always resolve to the local definition. |

### `.dynstr` Section

The `.dynstr` section is the string table for `.dynsym`. It stores null-terminated strings for all dynamic symbol names and library names referenced by DT_NEEDED entries. The first byte is always a null byte (the empty string), allowing index 0 to represent "no name."

### `.gnu.hash` Section

BCC generates a **GNU-style hash table** (`.gnu.hash`) for fast dynamic symbol lookup, preferred over the older SysV `.hash` format. The GNU hash table provides superior average-case lookup performance through a combination of:

1. **Bloom filter** — A bit-array filter that quickly rejects symbols that are definitely not present, avoiding unnecessary hash chain traversal for the common case of symbol-not-found.
2. **Hash buckets** — Standard hash table buckets mapping hash values to symbol chains.
3. **Hash values array** — Precomputed hash values for each symbol, enabling single-comparison chain traversal.

The result is **O(1) average-case** symbol resolution for successful lookups and fast rejection for failed lookups.

**`.gnu.hash` structure layout:**

| Component | Description |
|:----------|:------------|
| Header | `nbuckets`, `symoffset`, `bloom_size`, `bloom_shift` |
| Bloom filter | Array of `bloom_size` machine-word-sized entries |
| Buckets | Array of `nbuckets` 32-bit indices into the symbol table |
| Hash values | Array of 32-bit hash values (one per exported symbol) |

### `.rela.dyn` and `.rela.plt` Sections

These sections contain relocation entries that the dynamic linker processes at load time (`.rela.dyn`) or lazily at first call (`.rela.plt`).

**`.rela.dyn`** — Relocations for data references:
- GOT entries that need to be filled with absolute addresses
- Absolute data relocations in writable sections
- Copy relocations for external data symbols

**`.rela.plt`** — Relocations for PLT (function call) entries:
- One entry per dynamically linked function
- Processed lazily by default (filled on first call via the PLT resolver stub)

**Relocation entry structure (`ElfN_Rela`):**

| Field | Size (64-bit) | Description |
|:------|:--------------|:------------|
| `r_offset` | 8 bytes | Virtual address where the relocation is applied (typically a GOT slot). |
| `r_info` | 8 bytes | Encodes both the symbol index (upper 32 bits) and relocation type (lower 32 bits). |
| `r_addend` | 8 bytes | Constant addend used in the relocation calculation. |

For i686 (32-bit), the `.rel.dyn` and `.rel.plt` sections use the `ElfN_Rel` format (without an explicit addend — the addend is read from the relocation target location).

### `.got` and `.got.plt` Sections

The **Global Offset Table (GOT)** is a writable data section containing addresses that are resolved at load time or lazily at runtime. It is the cornerstone of position-independent code (PIC).

**`.got`** — General GOT entries:
- Contains resolved addresses for global data symbols accessed from PIC code
- Populated by the dynamic linker when processing `.rela.dyn` relocations
- Accessed via PC-relative addressing (e.g., `GOTPCREL` on x86-64, `ADRP`+`LDR` on AArch64)

**`.got.plt`** — PLT-specific GOT entries:
- The first three entries are reserved:
  - `GOT[0]` — Address of the `.dynamic` section
  - `GOT[1]` — Address of the link map structure (filled by the dynamic linker)
  - `GOT[2]` — Address of the `_dl_runtime_resolve` function (filled by the dynamic linker)
- Subsequent entries correspond to PLT stubs (one per dynamically linked function)
- Initially point back to the PLT stub (for lazy resolution); overwritten with the actual function address after first call

### `.plt` Section

The **Procedure Linkage Table (PLT)** contains executable code stubs that implement lazy symbol resolution for dynamically linked function calls.

**PLT layout:**

| Entry | Contents | Description |
|:------|:---------|:------------|
| PLT[0] | Resolver stub | Pushes the link map address, then jumps to `_dl_runtime_resolve` via `GOT[2]`. This stub is called by other PLT entries on their first invocation. |
| PLT[1..N] | Per-function stub | Loads the target address from the corresponding `.got.plt` entry and jumps to it. On the first call, this jumps back to PLT[0] (the resolver); after resolution, it jumps directly to the resolved function. |

**Architecture-specific PLT stub formats:**

- **x86-64:** `jmp *GOT[n](%rip)` / `push $index` / `jmp PLT[0]`
- **i686:** `jmp *GOT[n]` / `push $index` / `jmp PLT[0]`
- **AArch64:** `adrp x16, GOT[n]` / `ldr x17, [x16, #lo12]` / `br x17`
- **RISC-V 64:** `auipc t3, GOT[n]` / `ld t3, t3, lo12` / `jalr t1, t3`

### Implementation Source Map

| Component | Source File |
|:----------|:-----------|
| Dynamic section generation | `src/backend/linker_common/dynamic.rs` |
| Symbol resolution engine | `src/backend/linker_common/symbol_resolver.rs` |
| Section merging and layout | `src/backend/linker_common/section_merger.rs` |
| Relocation processing framework | `src/backend/linker_common/relocation.rs` |
| Linker script (section-to-segment mapping) | `src/backend/linker_common/linker_script.rs` |
| Common ELF writing infrastructure | `src/backend/elf_writer_common.rs` |
| x86-64 relocation application | `src/backend/x86_64/linker/relocations.rs` |
| i686 relocation application | `src/backend/i686/linker/relocations.rs` |
| AArch64 relocation application | `src/backend/aarch64/linker/relocations.rs` |
| RISC-V 64 relocation application | `src/backend/riscv64/linker/relocations.rs` |

---

## Relocation Types

Relocations are the mechanism by which the assembler and linker resolve symbolic references to concrete addresses. BCC's assembler emits relocations in `.rela.text` and `.rela.data` sections of object files. The linker processes these relocations when combining object files into the final executable or shared object.

In the relocation calculation formulas below:
- **S** = Value (address) of the symbol being referenced
- **A** = Addend (constant offset embedded in the relocation entry)
- **P** = Address of the place (location) being relocated
- **G** = Offset of the symbol's GOT entry from the GOT base
- **GOT** = Address of the Global Offset Table base
- **L** = Address of the PLT entry for the symbol

### x86-64 Relocations

Source files: `src/backend/x86_64/assembler/relocations.rs`, `src/backend/x86_64/linker/relocations.rs`

| Relocation | Value | Calculation | Field Size | Description |
|:-----------|:------|:------------|:-----------|:------------|
| `R_X86_64_NONE` | 0 | — | — | No relocation (placeholder). |
| `R_X86_64_64` | 1 | S + A | 64 bits | Absolute 64-bit address. Used for data pointers and absolute references. |
| `R_X86_64_PC32` | 2 | S + A - P | 32 bits | PC-relative 32-bit signed offset. Used for direct function calls (`call`), conditional branches, and RIP-relative data access. |
| `R_X86_64_GOT32` | 3 | G + A | 32 bits | 32-bit offset from the GOT base to the symbol's GOT entry. |
| `R_X86_64_PLT32` | 4 | L + A - P | 32 bits | PC-relative 32-bit offset to the PLT entry. Used for function calls that may resolve to a different shared object. The linker may relax this to `R_X86_64_PC32` for local symbols. |
| `R_X86_64_GLOB_DAT` | 6 | S | 64 bits | GOT entry fill — writes the absolute symbol address into a GOT slot. Used in `.rela.dyn` for data symbol resolution. |
| `R_X86_64_JUMP_SLOT` | 7 | S | 64 bits | PLT GOT entry fill — writes the absolute symbol address into a `.got.plt` slot. Used in `.rela.plt` for lazy function resolution. |
| `R_X86_64_RELATIVE` | 8 | B + A | 64 bits | Base-relative relocation for PIE/shared objects. B is the load base address. Used for internal pointer relocations in position-independent code. |
| `R_X86_64_GOTPCREL` | 9 | G + GOT + A - P | 32 bits | PC-relative 32-bit offset to the symbol's GOT entry. Used for PIC data access via the GOT. |
| `R_X86_64_32` | 10 | S + A | 32 bits | Absolute 32-bit address (zero-extended to 64 bits). Used in the small code model for absolute addressing. Linker verifies the result fits in 32 unsigned bits. |
| `R_X86_64_32S` | 11 | S + A | 32 bits | Absolute 32-bit address (sign-extended to 64 bits). Used in the small code model. Linker verifies the result fits in 32 signed bits. |
| `R_X86_64_16` | 12 | S + A | 16 bits | Absolute 16-bit address. |
| `R_X86_64_PC16` | 13 | S + A - P | 16 bits | PC-relative 16-bit offset. |
| `R_X86_64_8` | 14 | S + A | 8 bits | Absolute 8-bit address. |
| `R_X86_64_PC8` | 15 | S + A - P | 8 bits | PC-relative 8-bit offset. |
| `R_X86_64_GOTPCRELX` | 41 | G + GOT + A - P | 32 bits | Relaxable variant of `R_X86_64_GOTPCREL`. The linker may transform `mov` instructions using this relocation to `lea` when the symbol is locally defined (GOT optimization). |
| `R_X86_64_REX_GOTPCRELX` | 42 | G + GOT + A - P | 32 bits | Same as `R_X86_64_GOTPCRELX` but for instructions with a REX prefix. The linker applies the same GOT-to-direct optimization when possible. |

### i686 Relocations

Source files: `src/backend/i686/assembler/relocations.rs`, `src/backend/i686/linker/relocations.rs`

| Relocation | Value | Calculation | Description |
|:-----------|:------|:------------|:------------|
| `R_386_NONE` | 0 | — | No relocation. |
| `R_386_32` | 1 | S + A | Absolute 32-bit address. Direct address reference for function pointers and data. |
| `R_386_PC32` | 2 | S + A - P | PC-relative 32-bit. Used for direct function calls and branches. |
| `R_386_GOT32` | 3 | G + A | 32-bit offset from GOT base to the symbol's GOT entry. |
| `R_386_PLT32` | 4 | L + A - P | PC-relative 32-bit offset to PLT entry. |
| `R_386_COPY` | 5 | — | Copy relocation — instructs the dynamic linker to copy the symbol's data from the shared object into the executable's BSS. |
| `R_386_GLOB_DAT` | 6 | S | GOT entry fill — absolute symbol address written to GOT slot. |
| `R_386_JMP_SLOT` | 7 | S | PLT GOT slot fill for lazy binding. |
| `R_386_RELATIVE` | 8 | B + A | Base-relative relocation for PIE/shared objects. |
| `R_386_GOTOFF` | 9 | S + A - GOT | GOT-relative offset. Computes the distance from the GOT base to the symbol. Used for accessing static data in PIC code. |
| `R_386_GOTPC` | 10 | GOT + A - P | PC-relative offset to the GOT base. Used to compute the GOT base address at runtime. |

### AArch64 Relocations

Source files: `src/backend/aarch64/assembler/relocations.rs`, `src/backend/aarch64/linker/relocations.rs`

| Relocation | Value | Calculation | Description |
|:-----------|:------|:------------|:------------|
| `R_AARCH64_NONE` | 0 | — | No relocation. |
| `R_AARCH64_ABS64` | 257 | S + A | Absolute 64-bit address. Used for data pointers in writable sections. |
| `R_AARCH64_ABS32` | 258 | S + A | Absolute 32-bit address. |
| `R_AARCH64_ABS16` | 259 | S + A | Absolute 16-bit address. |
| `R_AARCH64_PREL64` | 260 | S + A - P | PC-relative 64-bit offset. |
| `R_AARCH64_PREL32` | 261 | S + A - P | PC-relative 32-bit offset. |
| `R_AARCH64_CALL26` | 283 | S + A - P | PC-relative 26-bit offset for `BL` (branch-and-link) instruction. The 26-bit immediate is shifted left by 2 (4-byte alignment), giving a ±128 MiB range. |
| `R_AARCH64_JUMP26` | 282 | S + A - P | PC-relative 26-bit offset for `B` (unconditional branch) instruction. Same encoding and range as `R_AARCH64_CALL26`. |
| `R_AARCH64_ADR_PREL_PG_HI21` | 275 | Page(S + A) - Page(P) | Page-relative 21-bit offset for `ADRP` instruction. Computes the page (4 KiB aligned) address difference. The 21-bit immediate is shifted left by 12, giving a ±4 GiB range. |
| `R_AARCH64_ADD_ABS_LO12_NC` | 277 | (S + A) & 0xFFF | Low 12-bit page offset for `ADD` instruction. Paired with `ADRP` to form a full address. "NC" means no overflow check. |
| `R_AARCH64_LDST8_ABS_LO12_NC` | 278 | (S + A) & 0xFFF | Low 12-bit page offset for byte-sized load/store instructions. |
| `R_AARCH64_LDST16_ABS_LO12_NC` | 284 | (S + A) & 0xFFF | Low 12-bit page offset for 16-bit load/store (shifted right by 1). |
| `R_AARCH64_LDST32_ABS_LO12_NC` | 285 | (S + A) & 0xFFF | Low 12-bit page offset for 32-bit load/store (shifted right by 2). |
| `R_AARCH64_LDST64_ABS_LO12_NC` | 286 | (S + A) & 0xFFF | Low 12-bit page offset for 64-bit load/store (shifted right by 3). |
| `R_AARCH64_LDST128_ABS_LO12_NC` | 299 | (S + A) & 0xFFF | Low 12-bit page offset for 128-bit load/store (shifted right by 4). |
| `R_AARCH64_ADR_GOT_PAGE` | 311 | Page(G + GOT) - Page(P) | GOT page-relative 21-bit for `ADRP`. Used to load the page containing the target GOT entry. |
| `R_AARCH64_LD64_GOT_LO12_NC` | 312 | (G + GOT) & 0xFF8 | Low 12-bit GOT entry offset for `LDR` (64-bit load). Paired with `ADRP`+`R_AARCH64_ADR_GOT_PAGE` to access GOT entries in PIC code. |
| `R_AARCH64_GLOB_DAT` | 1025 | S | GOT entry fill with absolute address. |
| `R_AARCH64_JUMP_SLOT` | 1026 | S | PLT GOT slot fill for lazy binding. |
| `R_AARCH64_RELATIVE` | 1027 | B + A | Base-relative relocation for PIE/shared objects. |

**AArch64 addressing pattern:** Most code references use a two-instruction `ADRP`+`ADD` or `ADRP`+`LDR` pair. The `ADRP` instruction loads the page-aligned base address using `R_AARCH64_ADR_PREL_PG_HI21`, and the second instruction adds or loads the low 12-bit offset using the corresponding `LO12` relocation.

### RISC-V 64 Relocations

Source files: `src/backend/riscv64/assembler/relocations.rs`, `src/backend/riscv64/linker/relocations.rs`

| Relocation | Value | Calculation | Description |
|:-----------|:------|:------------|:------------|
| `R_RISCV_NONE` | 0 | — | No relocation. |
| `R_RISCV_32` | 1 | S + A | Absolute 32-bit address. |
| `R_RISCV_64` | 2 | S + A | Absolute 64-bit address. |
| `R_RISCV_BRANCH` | 16 | S + A - P | PC-relative offset for B-type (conditional branch) instructions. The immediate is encoded in a split format across instruction bits. Range: ±4 KiB. |
| `R_RISCV_JAL` | 17 | S + A - P | PC-relative offset for J-type (`JAL`) instruction. The 20-bit immediate (shifted left by 1) gives a ±1 MiB range. |
| `R_RISCV_CALL` | 18 | S + A - P | PC-relative offset for a call sequence: `AUIPC` (loads high 20 bits) + `JALR` (adds low 12 bits). The linker patches both instructions. Combined range: ±2 GiB. |
| `R_RISCV_CALL_PLT` | 19 | S + A - P | Same as `R_RISCV_CALL` but may resolve through the PLT for cross-object calls. |
| `R_RISCV_PCREL_HI20` | 23 | S + A - P | PC-relative high 20-bit offset for `AUIPC` instruction. The companion low-12 instruction references this relocation via `R_RISCV_PCREL_LO12_I` or `R_RISCV_PCREL_LO12_S`. |
| `R_RISCV_PCREL_LO12_I` | 24 | (see HI20) | Low 12-bit PC-relative offset for I-type instructions (loads, `ADDI`). Must reference a label at the position of the corresponding `R_RISCV_PCREL_HI20` relocation. |
| `R_RISCV_PCREL_LO12_S` | 25 | (see HI20) | Low 12-bit PC-relative offset for S-type instructions (stores). Same referencing semantics as `R_RISCV_PCREL_LO12_I`. |
| `R_RISCV_HI20` | 26 | S + A | Absolute high 20-bit for `LUI` instruction. |
| `R_RISCV_LO12_I` | 27 | S + A | Absolute low 12-bit for I-type instructions. |
| `R_RISCV_LO12_S` | 28 | S + A | Absolute low 12-bit for S-type instructions. |
| `R_RISCV_GOT_HI20` | 190 | G + GOT + A - P | GOT-relative high 20-bit for `AUIPC`. Used with a subsequent `LDR` to access a GOT entry. |
| `R_RISCV_COPY` | 4 | — | Copy relocation for dynamic linking. |
| `R_RISCV_GLOB_DAT` | 5 | S | GOT entry fill. |
| `R_RISCV_JUMP_SLOT` | 6 | S | PLT GOT slot fill. |
| `R_RISCV_RELATIVE` | 3 | B + A | Base-relative relocation for PIE/shared objects. |
| `R_RISCV_RELAX` | 51 | — | **Relaxation hint.** Signals to the linker that the preceding relocation's instruction sequence may be shortened if the target is within a smaller range. |

### RISC-V Linker Relaxation

RISC-V is unique among BCC's supported architectures in its support for **linker relaxation**. The assembler emits conservative (long-range) instruction sequences and annotates them with `R_RISCV_RELAX` hints. During linking, BCC's RISC-V linker may transform:

| Original Sequence | Relaxed Form | Condition |
|:------------------|:-------------|:----------|
| `AUIPC` + `JALR` (call) | `JAL` (single instruction) | Target within ±1 MiB of the call site |
| `AUIPC` + `ADDI` (address load) | `C.ADDI` or similar (compressed) | Target within range and C extension enabled |
| `AUIPC` + `LD` (GOT access) | Direct `ADDI` | Symbol is local and within range |
| `LUI` + `ADDI` (absolute load) | `C.LI` or `C.LUI` | Value fits in compressed immediate |

Relaxation is performed iteratively — each relaxation may bring other targets into shorter range, requiring multiple passes until a fixpoint is reached. All offsets and relocation targets are updated after each relaxation pass.

---

## Default Linker Script

BCC's built-in linker (`src/backend/linker_common/linker_script.rs`) applies a default section-to-segment mapping when producing linked output. This mapping controls how input sections from object files are combined and placed into output segments.

### Section-to-Segment Mapping

The default mapping organizes sections by permission (read, write, execute) to minimize the number of segments while maximizing memory protection:

| Output Segment | Type | Permissions | Input Sections |
|:---------------|:-----|:------------|:---------------|
| Program Headers | PT_PHDR | PF_R | (self-referencing) |
| Interpreter | PT_INTERP | PF_R | `.interp` (if dynamic linking) |
| Code | PT_LOAD | PF_R \| PF_X | `.text`, `.plt`, custom executable sections |
| Read-only Data | PT_LOAD | PF_R | `.rodata`, `.gnu.hash`, `.dynsym`, `.dynstr`, `.rela.dyn`, `.rela.plt`, `.note.*` |
| Read-write Data | PT_LOAD | PF_R \| PF_W | `.data`, `.init_array`, `.fini_array`, `.dynamic`, `.got`, `.got.plt`, `.bss` |
| Dynamic | PT_DYNAMIC | PF_R \| PF_W | `.dynamic` (overlaps with read-write data segment) |
| Stack | PT_GNU_STACK | PF_R \| PF_W | (empty — declares non-executable stack) |

### Entry Point

- For ET_EXEC: The entry point (`e_entry`) is set to the address of the `_start` symbol. If `_start` is not defined, the linker reports an error.
- For ET_DYN (shared objects): The entry point is set to 0 unless the shared object defines a `_start` symbol.
- The entry point can be overridden with the `-e <symbol>` linker flag.

### Segment Alignment and Padding

- **Page alignment**: All PT_LOAD segments are aligned to the target page size (4 KiB for x86-64/i686/RISC-V 64, 64 KiB for AArch64). This ensures each segment starts on a page boundary, allowing the kernel to set per-page memory protection.
- **Permission transitions**: When consecutive sections have different permissions (e.g., `.text` [R+X] followed by `.rodata` [R]), the linker inserts padding to ensure the transition occurs at a page boundary. This gap in the file may be zero-filled.
- **Section alignment**: Within a segment, sections are placed according to their `sh_addralign` values. The linker inserts the minimum padding necessary to satisfy alignment requirements.

### COMDAT Group Handling

BCC's linker supports **COMDAT section groups** (SHT_GROUP with GRP_COMDAT flag). When multiple object files define the same COMDAT group (e.g., inline function definitions included from a header in multiple translation units), the linker keeps only one copy and discards duplicates. This deduplication ensures:

- Inline functions compiled in multiple translation units are not multiply-defined
- Template-like patterns (when used via macros) do not cause linker errors
- The kept group's sections are included in the final output; discarded groups' sections and symbols are removed

### Base Addresses

The default virtual base addresses for ET_EXEC output are:

| Architecture | Default Base Address |
|:-------------|:---------------------|
| x86-64 | `0x400000` |
| i686 | `0x08048000` |
| AArch64 | `0x400000` |
| RISC-V 64 | `0x10000` |

For ET_DYN output, the base address is 0 (position-independent — the actual load address is determined by the kernel's ASLR).

---

## Relocatable Object Files (.o)

When BCC is invoked with the `-c` flag, it produces **relocatable object files** (ET_REL) instead of linked executables. These `.o` files serve as input to BCC's built-in linker during the final link step.

### Object File Structure

| Component | Description |
|:----------|:------------|
| ELF Header | `e_type = ET_REL` (1). All other header fields match the target architecture as described in the ELF Header section. `e_entry = 0` (no entry point for object files). |
| `.text` | Machine code for all functions defined in the translation unit. |
| `.rodata` | Read-only data (string literals, constants) from the translation unit. |
| `.data` | Initialized global and static variables. |
| `.bss` | Uninitialized global and static variables (SHT_NOBITS). |
| `.rela.text` | RELA-format relocations for the `.text` section. Each entry describes a symbolic reference in the machine code that needs to be resolved at link time. |
| `.rela.data` | RELA-format relocations for the `.data` section. Applies to initialized data containing pointer values that reference external symbols. |
| `.rela.rodata` | RELA-format relocations for `.rodata` (if it contains relocatable references, e.g., address constants in jump tables). |
| `.symtab` | Symbol table containing all local and global symbols. Local symbols (static functions, file-scope variables) have `STB_LOCAL` binding. Externally visible symbols have `STB_GLOBAL` binding. Weak symbols have `STB_WEAK` binding. |
| `.strtab` | String table for `.symtab`. |
| `.shstrtab` | Section header string table. |
| `.note.GNU-stack` | Stack executability marker (empty, no SHF_EXECINSTR). |
| `.comment` | Compiler identification. |

### Object File Conventions

- **RELA format**: BCC always uses RELA relocations (with explicit addends) rather than REL relocations, even for i686 targets. This simplifies the linker implementation — the addend is always in the relocation entry, never in the section content.
- **Section indices**: The `.symtab` section's `sh_link` field points to `.strtab`. Each `.rela.*` section's `sh_link` points to `.symtab`, and `sh_info` points to the section being relocated.
- **Symbol ordering**: In `.symtab`, all `STB_LOCAL` symbols precede all `STB_GLOBAL` and `STB_WEAK` symbols. The `sh_info` field of the `.symtab` section header gives the index of the first non-local symbol.
- **Section symbols**: One `STT_SECTION` symbol is emitted for each defined section, enabling section-relative relocations.
- **Undefined symbols**: References to functions or variables defined in other translation units appear as `SHN_UNDEF` symbols in `.symtab`. The linker resolves these during the link step.

### Debug Sections in Object Files

When `-g` is specified, object files additionally contain:

| Section | Description |
|:--------|:------------|
| `.debug_info` | DWARF v4 debug information for the translation unit. |
| `.debug_abbrev` | Abbreviation table for `.debug_info`. |
| `.debug_line` | Line number program for source mapping. |
| `.debug_str` | Debug string table. |
| `.rela.debug_info` | Relocations within `.debug_info` (e.g., references to `.text` addresses for function low/high PC). |
| `.rela.debug_line` | Relocations within `.debug_line` (e.g., references to `.text` for the statement program's base address). |

These debug sections and their relocations are carried through to the final linked output by the linker, which applies the debug relocations to produce correct address references in the final binary.

---

## Appendix: ELF Constants Reference

This appendix lists the numeric values of commonly referenced ELF constants used throughout BCC's ELF writer and linker.

### ELF Type Constants

| Constant | Value | Description |
|:---------|:------|:------------|
| `ET_NONE` | 0 | No file type |
| `ET_REL` | 1 | Relocatable object file |
| `ET_EXEC` | 2 | Executable file |
| `ET_DYN` | 3 | Shared object / PIE |

### Section Header Types

| Constant | Value | Description |
|:---------|:------|:------------|
| `SHT_NULL` | 0 | Inactive section header |
| `SHT_PROGBITS` | 1 | Program data |
| `SHT_SYMTAB` | 2 | Symbol table |
| `SHT_STRTAB` | 3 | String table |
| `SHT_RELA` | 4 | Relocation entries with addends |
| `SHT_HASH` | 5 | Symbol hash table (SysV) |
| `SHT_DYNAMIC` | 6 | Dynamic linking information |
| `SHT_NOTE` | 7 | Notes |
| `SHT_NOBITS` | 8 | Section occupies no file space (BSS) |
| `SHT_REL` | 9 | Relocation entries without addends |
| `SHT_INIT_ARRAY` | 14 | Array of constructors |
| `SHT_FINI_ARRAY` | 15 | Array of destructors |
| `SHT_GROUP` | 17 | Section group (COMDAT) |
| `SHT_GNU_HASH` | 0x6ffffff6 | GNU hash table |

### Section Header Flags

| Constant | Value | Description |
|:---------|:------|:------------|
| `SHF_WRITE` | 0x1 | Section is writable |
| `SHF_ALLOC` | 0x2 | Section occupies memory during execution |
| `SHF_EXECINSTR` | 0x4 | Section contains executable instructions |
| `SHF_MERGE` | 0x10 | Section may be merged (e.g., string deduplication) |
| `SHF_STRINGS` | 0x20 | Section contains null-terminated strings |
| `SHF_GROUP` | 0x200 | Section is a member of a group |
| `SHF_TLS` | 0x400 | Section holds thread-local data |

### Program Header Types

| Constant | Value | Description |
|:---------|:------|:------------|
| `PT_NULL` | 0 | Unused entry |
| `PT_LOAD` | 1 | Loadable segment |
| `PT_DYNAMIC` | 2 | Dynamic linking information |
| `PT_INTERP` | 3 | Path to interpreter |
| `PT_NOTE` | 4 | Auxiliary information |
| `PT_PHDR` | 6 | Program header table |
| `PT_TLS` | 7 | Thread-local storage template |
| `PT_GNU_STACK` | 0x6474e551 | Stack permissions |
| `PT_GNU_RELRO` | 0x6474e552 | Read-only after relocation |

### Program Header Flags

| Constant | Value | Description |
|:---------|:------|:------------|
| `PF_X` | 0x1 | Segment is executable |
| `PF_W` | 0x2 | Segment is writable |
| `PF_R` | 0x4 | Segment is readable |

### Symbol Binding

| Constant | Value | Description |
|:---------|:------|:------------|
| `STB_LOCAL` | 0 | Local symbol — not visible outside the object file |
| `STB_GLOBAL` | 1 | Global symbol — visible to all object files |
| `STB_WEAK` | 2 | Weak symbol — like global but may be overridden |

### Symbol Types

| Constant | Value | Description |
|:---------|:------|:------------|
| `STT_NOTYPE` | 0 | Unspecified type |
| `STT_OBJECT` | 1 | Data object (variable) |
| `STT_FUNC` | 2 | Function |
| `STT_SECTION` | 3 | Section symbol |
| `STT_FILE` | 4 | File name symbol |

### Symbol Visibility

| Constant | Value | Description |
|:---------|:------|:------------|
| `STV_DEFAULT` | 0 | Default visibility (exported) |
| `STV_INTERNAL` | 1 | Internal (processor-specific) |
| `STV_HIDDEN` | 2 | Hidden (not exported from shared object) |
| `STV_PROTECTED` | 3 | Protected (exported but not preemptible) |
