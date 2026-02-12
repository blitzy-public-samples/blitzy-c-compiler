# BCC ABI Reference

## Table of Contents

- [1. Overview](#1-overview)
- [2. x86-64 — System V AMD64 ABI](#2-x86-64--system-v-amd64-abi)
  - [2.1 Data Model](#21-data-model)
  - [2.2 Register File](#22-register-file)
  - [2.3 Parameter Passing](#23-parameter-passing)
  - [2.4 Return Values](#24-return-values)
  - [2.5 Callee-Saved and Caller-Saved Registers](#25-callee-saved-and-caller-saved-registers)
  - [2.6 Struct Classification Algorithm](#26-struct-classification-algorithm)
  - [2.7 Red Zone](#27-red-zone)
  - [2.8 Stack Frame Layout](#28-stack-frame-layout)
  - [2.9 Variadic Functions](#29-variadic-functions)
  - [2.10 Implementation Reference](#210-implementation-reference)
- [3. i686 — cdecl / System V i386 ABI](#3-i686--cdecl--system-v-i386-abi)
  - [3.1 Data Model](#31-data-model)
  - [3.2 Register File](#32-register-file)
  - [3.3 Parameter Passing](#33-parameter-passing)
  - [3.4 Return Values](#34-return-values)
  - [3.5 Callee-Saved and Caller-Saved Registers](#35-callee-saved-and-caller-saved-registers)
  - [3.6 Struct Passing](#36-struct-passing)
  - [3.7 Stack Frame Layout](#37-stack-frame-layout)
  - [3.8 x87 Floating-Point Considerations](#38-x87-floating-point-considerations)
  - [3.9 Implementation Reference](#39-implementation-reference)
- [4. AArch64 — AAPCS64 ABI](#4-aarch64--aapcs64-abi)
  - [4.1 Data Model](#41-data-model)
  - [4.2 Register File](#42-register-file)
  - [4.3 Parameter Passing](#43-parameter-passing)
  - [4.4 Return Values](#44-return-values)
  - [4.5 Callee-Saved and Caller-Saved Registers](#45-callee-saved-and-caller-saved-registers)
  - [4.6 HFA and HVA Passing](#46-hfa-and-hva-passing)
  - [4.7 Stack Frame Layout](#47-stack-frame-layout)
  - [4.8 Implementation Reference](#48-implementation-reference)
- [5. RISC-V 64 — LP64D ABI](#5-risc-v-64--lp64d-abi)
  - [5.1 Data Model](#51-data-model)
  - [5.2 Register File](#52-register-file)
  - [5.3 Parameter Passing](#53-parameter-passing)
  - [5.4 Return Values](#54-return-values)
  - [5.5 Callee-Saved and Caller-Saved Registers](#55-callee-saved-and-caller-saved-registers)
  - [5.6 Struct and Floating-Point Passing](#56-struct-and-floating-point-passing)
  - [5.7 Stack Frame Layout](#57-stack-frame-layout)
  - [5.8 Large Immediates and Addressing](#58-large-immediates-and-addressing)
  - [5.9 Implementation Reference](#59-implementation-reference)
- [6. Cross-Architecture Comparison](#6-cross-architecture-comparison)
  - [6.1 Data Models](#61-data-models)
  - [6.2 Argument Passing Summary](#62-argument-passing-summary)
  - [6.3 Return Value Conventions](#63-return-value-conventions)
  - [6.4 Register Counts and Classification](#64-register-counts-and-classification)
  - [6.5 Stack Alignment and Frame Conventions](#65-stack-alignment-and-frame-conventions)
- [7. Type System Bridge](#7-type-system-bridge)
  - [7.1 Dual Type System Overview](#71-dual-type-system-overview)
  - [7.2 C Type to IR Type Mapping](#72-c-type-to-ir-type-mapping)
  - [7.3 IR Type to Register Class Mapping](#73-ir-type-to-register-class-mapping)
  - [7.4 Aggregate Layout and ABI Classification](#74-aggregate-layout-and-abi-classification)

---

## 1. Overview

This document is the definitive reference for calling convention implementations in BCC's
architecture-specific ABI modules (`src/backend/*/abi.rs`). It covers parameter passing, return
value handling, register usage, stack frame layout, and struct/union passing rules for all four
target architectures supported by BCC:

| Architecture | ABI Standard                    | Implementation File              |
|--------------|---------------------------------|----------------------------------|
| x86-64       | System V AMD64 ABI              | `src/backend/x86_64/abi.rs`      |
| i686         | cdecl / System V i386 ABI       | `src/backend/i686/abi.rs`        |
| AArch64      | AAPCS64 (ARM Architecture)      | `src/backend/aarch64/abi.rs`     |
| RISC-V 64    | LP64D (RISC-V ELF psABI)       | `src/backend/riscv64/abi.rs`     |

Each ABI module implements the architecture-specific parameter classification, register assignment,
and struct-passing logic that the code generation driver (`src/backend/generation.rs`) invokes
through the `ArchCodegen` trait (`src/backend/traits.rs`).

The dual type system in `src/common/types.rs` and `src/common/type_builder.rs` provides the
bridge between C language types used by the frontend and machine-level types used by the backend
for ABI-correct code generation.

---

## 2. x86-64 — System V AMD64 ABI

The System V AMD64 ABI is the standard calling convention for 64-bit x86 Linux systems. BCC
implements this ABI for all x86-64 code generation, including the struct classification algorithm
that determines whether aggregate types are passed in registers or on the stack.

### 2.1 Data Model

The x86-64 target uses the **LP64** data model:

| C Type            | Size (bytes) | Alignment (bytes) |
|-------------------|--------------|--------------------|
| `_Bool`           | 1            | 1                  |
| `char`            | 1            | 1                  |
| `short`           | 2            | 2                  |
| `int`             | 4            | 4                  |
| `long`            | 8            | 8                  |
| `long long`       | 8            | 8                  |
| `float`           | 4            | 4                  |
| `double`          | 8            | 8                  |
| `long double`     | 16           | 16                 |
| `_Complex float`  | 8            | 4                  |
| `_Complex double` | 16           | 8                  |
| pointer           | 8            | 8                  |
| `size_t`          | 8            | 8                  |
| `ptrdiff_t`       | 8            | 8                  |
| `wchar_t`         | 4            | 4                  |

Key LP64 characteristic: `int` is 32-bit, `long` and pointers are 64-bit.

### 2.2 Register File

The x86-64 architecture provides 16 general-purpose registers (GPRs) and 16 SSE registers:

**General-Purpose Registers (64-bit):**

| Register | ABI Name        | Purpose                           |
|----------|-----------------|-----------------------------------|
| RAX      | Return value    | Integer return (low half)         |
| RBX      | Callee-saved    | Base register (preserved)         |
| RCX      | Arg 4           | 4th integer argument              |
| RDX      | Arg 3 / Return  | 3rd integer arg / return (high)   |
| RSI      | Arg 2           | 2nd integer argument              |
| RDI      | Arg 1           | 1st integer argument              |
| RBP      | Frame pointer   | Callee-saved frame pointer        |
| RSP      | Stack pointer   | Stack pointer (not allocatable)   |
| R8       | Arg 5           | 5th integer argument              |
| R9       | Arg 6           | 6th integer argument              |
| R10      | Scratch         | Static chain pointer (caller-saved) |
| R11      | Scratch         | Scratch (caller-saved)            |
| R12      | Callee-saved    | Preserved across calls            |
| R13      | Callee-saved    | Preserved across calls            |
| R14      | Callee-saved    | Preserved across calls            |
| R15      | Callee-saved    | Preserved across calls            |

**SSE Registers (128-bit):**

| Register   | Purpose                         |
|------------|----------------------------------|
| XMM0       | 1st FP arg / FP return value    |
| XMM1       | 2nd FP argument                 |
| XMM2       | 3rd FP argument                 |
| XMM3       | 4th FP argument                 |
| XMM4       | 5th FP argument                 |
| XMM5       | 6th FP argument                 |
| XMM6       | 7th FP argument                 |
| XMM7       | 8th FP argument                 |
| XMM8–XMM15 | Scratch (caller-saved)          |

### 2.3 Parameter Passing

Integer and pointer arguments are passed in the following registers, in order:

1. **RDI** — 1st integer/pointer argument
2. **RSI** — 2nd integer/pointer argument
3. **RDX** — 3rd integer/pointer argument
4. **RCX** — 4th integer/pointer argument
5. **R8**  — 5th integer/pointer argument
6. **R9**  — 6th integer/pointer argument

Floating-point arguments are passed in the following registers, in order:

1. **XMM0** — 1st floating-point argument
2. **XMM1** — 2nd floating-point argument
3. **XMM2** — 3rd floating-point argument
4. **XMM3** — 4th floating-point argument
5. **XMM4** — 5th floating-point argument
6. **XMM5** — 6th floating-point argument
7. **XMM6** — 7th floating-point argument
8. **XMM7** — 8th floating-point argument

**Rules:**

- Integer and floating-point argument sequences are tracked independently. A function
  `void f(int a, double b, int c)` passes `a` in RDI, `b` in XMM0, and `c` in RSI.
- Arguments beyond the available registers are passed on the stack, pushed right-to-left.
- Stack arguments are aligned to 8-byte boundaries.
- The stack pointer (RSP) must be 16-byte aligned **before** the `call` instruction. Since `call`
  pushes an 8-byte return address, the callee sees RSP at an 8-byte-aligned (but not 16-byte-aligned)
  value upon entry; the prologue must account for this.
- For variadic functions (`...`), the number of SSE registers used is passed in AL (RAX low byte).

### 2.4 Return Values

| Return Category              | Registers Used       | Notes                                       |
|------------------------------|----------------------|----------------------------------------------|
| Integer ≤ 64 bits            | RAX                  | Zero- or sign-extended as appropriate        |
| Integer 65–128 bits          | RAX (low), RDX (high)| Two-register return                          |
| `float` / `double`           | XMM0                 | Scalar SSE return                            |
| `_Complex float`             | XMM0                 | Real in low 32 bits, imag in high 32 bits    |
| `_Complex double`            | XMM0 (real), XMM1 (imag) | Two-register return                     |
| `long double`                | ST(0)                | x87 FPU top of stack                         |
| `_Complex long double`       | ST(0) (real), ST(1) (imag) | x87 FPU stack                          |
| Struct classified as INTEGER | RAX, optionally RDX  | After struct classification                  |
| Struct classified as SSE     | XMM0, optionally XMM1| After struct classification                  |
| Struct classified as MEMORY  | Caller-allocated      | Hidden pointer in RDI (consumes 1st arg slot)|

When a struct is returned as MEMORY, the caller allocates stack space, passes a hidden pointer
as the first argument (in RDI, shifting all other integer arguments by one register), and the
callee writes the return value through this pointer. RAX returns the same pointer value.

### 2.5 Callee-Saved and Caller-Saved Registers

**Callee-saved** (the called function must preserve these; if used, they must be saved and restored):

- RBX, RBP, R12, R13, R14, R15

**Caller-saved** (the calling function must assume these are clobbered after a call):

- RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11
- XMM0–XMM15 (all SSE registers are caller-saved)
- x87 FPU stack (ST(0)–ST(7)) — caller-saved
- RFLAGS — caller-saved (direction flag DF must be clear on function entry and exit)

### 2.6 Struct Classification Algorithm

The System V AMD64 ABI classifies each 8-byte chunk ("eightbyte") of a struct independently.
This determines how the struct is passed or returned. The classification categories are:

| Class         | Meaning                                                      |
|---------------|--------------------------------------------------------------|
| INTEGER       | Passed in a general-purpose register (RDI, RSI, etc.)       |
| SSE           | Passed in an SSE register (XMM0, etc.)                      |
| SSEUP         | Upper part of a wide SSE value (paired with preceding SSE)  |
| X87           | Passed via x87 FPU (long double real part)                  |
| X87UP         | Upper part of a long double (paired with preceding X87)     |
| COMPLEX_X87   | Complex long double — passed in memory                      |
| MEMORY        | Passed on the stack (caller-allocated for returns)           |
| NO_CLASS      | Zero-width or padding — ignored during merge                |

**Classification procedure for a struct of total size S:**

1. If S > 16 bytes (two eightbytes), classify as MEMORY.
2. If the struct has unaligned fields, classify as MEMORY.
3. For each eightbyte of the struct, classify based on the fields it contains:
   - `_Bool`, `char`, `short`, `int`, `long`, `long long`, pointer → INTEGER
   - `float`, `double` → SSE
   - `long double` → X87 (first eightbyte) + X87UP (second eightbyte)
   - `_Complex long double` → COMPLEX_X87
4. If any eightbyte is classified as MEMORY, the entire struct is MEMORY.
5. If the struct has a `__attribute__((packed))` attribute or `_Atomic` qualifier on a misaligned
   field, classify as MEMORY.
6. Apply post-merger rules:
   - If one eightbyte is X87 and the other is not X87UP, the whole struct is MEMORY.
   - If the size exceeds two eightbytes and the first eightbyte is not SSE, or any other
     eightbyte is not SSEUP, the whole struct is MEMORY.

**Example classifications:**

```
struct { int a; int b; }          → 1 eightbyte: INTEGER          → passed in RDI
struct { int a; float b; }        → 1 eightbyte: INTEGER (int dominates float in same eightbyte)
struct { long a; double b; }      → 2 eightbytes: INTEGER, SSE    → passed in RDI + XMM0
struct { char a[17]; }            → > 16 bytes: MEMORY             → passed on stack
struct { long double x; }         → 2 eightbytes: X87, X87UP      → passed in x87 ST(0)
```

### 2.7 Red Zone

The 128 bytes below RSP (i.e., RSP-128 through RSP-1) constitute the **red zone**. Leaf
functions (functions that do not call other functions) may use this area without adjusting RSP.
This optimization avoids the overhead of a stack frame setup/teardown for small leaf functions.

**Rules:**

- Signal handlers and interrupt handlers must not use the red zone (they may overwrite it).
- The red zone is only valid in user-space code. Kernel code does not use the red zone
  (the Linux kernel compiles with `-mno-red-zone`).
- BCC respects the red zone for regular user-space code but does not rely on it for
  kernel-targeted compilation.

### 2.8 Stack Frame Layout

A typical x86-64 stack frame layout (growing downward toward lower addresses):

```
High addresses
┌──────────────────────────┐
│   Caller's frame          │
├──────────────────────────┤
│   Return address (8 bytes)│  ← pushed by CALL instruction
├──────────────────────────┤
│   Saved RBP (8 bytes)    │  ← if frame pointer is used (pushed by callee)
├──────────────────────────┤  ← RBP points here (if frame pointer is used)
│   Local variables         │
│   Spill slots             │
│   Saved callee-saved regs │
├──────────────────────────┤  ← RSP points here
│   Red zone (128 bytes)    │  ← usable by leaf functions without RSP adjustment
└──────────────────────────┘
Low addresses
```

**Stack alignment rules:**

- RSP must be 16-byte aligned before each `call` instruction.
- The `call` instruction pushes 8 bytes (return address), so RSP is 8-mod-16 at function entry.
- Prologues must adjust RSP to restore 16-byte alignment for the function body.
- Stack allocations (alloca, VLAs) must maintain 16-byte alignment.

**Stack probe requirement (BCC-specific, security mitigation):**

When a function's stack frame exceeds 4096 bytes, BCC generates a probe loop that touches
each page sequentially before the final RSP adjustment. This prevents skipping over stack
guard pages. See `src/backend/x86_64/security.rs` for implementation details.

### 2.9 Variadic Functions

For variadic functions (those declared with `...`):

- Named arguments are passed normally (in registers per the rules above).
- The caller must set AL (the low byte of RAX) to the number of vector (SSE) registers used
  for variable arguments (0 through 8). This is used by the callee's register-save area setup.
- The callee typically saves all argument registers (both integer and SSE) to the **register
  save area** on the stack to support `va_arg` iteration.
- `va_list` is a struct containing:
  - `gp_offset` (unsigned int) — offset into the integer register save area
  - `fp_offset` (unsigned int) — offset into the SSE register save area
  - `overflow_arg_area` (void *) — pointer to stack arguments
  - `reg_save_area` (void *) — pointer to the register save area

### 2.10 Implementation Reference

- **Source file:** `src/backend/x86_64/abi.rs`
- **Register definitions:** `src/backend/x86_64/registers.rs`
- **Code generation:** `src/backend/x86_64/codegen.rs`
- **Security mitigations:** `src/backend/x86_64/security.rs`

---

## 3. i686 — cdecl / System V i386 ABI

The cdecl (System V i386) ABI is the standard calling convention for 32-bit x86 Linux systems.
All arguments are passed on the stack. The x87 FPU is used for floating-point operations.

### 3.1 Data Model

The i686 target uses the **ILP32** data model:

| C Type            | Size (bytes) | Alignment (bytes) |
|-------------------|--------------|--------------------|
| `_Bool`           | 1            | 1                  |
| `char`            | 1            | 1                  |
| `short`           | 2            | 2                  |
| `int`             | 4            | 4                  |
| `long`            | 4            | 4                  |
| `long long`       | 8            | 4                  |
| `float`           | 4            | 4                  |
| `double`          | 8            | 4                  |
| `long double`     | 12           | 4                  |
| `_Complex float`  | 8            | 4                  |
| `_Complex double` | 16           | 4                  |
| pointer           | 4            | 4                  |
| `size_t`          | 4            | 4                  |
| `ptrdiff_t`       | 4            | 4                  |
| `wchar_t`         | 4            | 4                  |

Key ILP32 characteristic: `int`, `long`, and pointers are all 32-bit.

### 3.2 Register File

The i686 architecture provides 8 general-purpose registers (32-bit):

| Register | ABI Name                | Purpose                           |
|----------|-------------------------|-----------------------------------|
| EAX      | Return value / Scratch  | Integer return (low 32 bits)      |
| EBX      | Callee-saved            | Preserved across calls            |
| ECX      | Scratch                 | Caller-saved                      |
| EDX      | Return high / Scratch   | Integer return (high 32 bits for 64-bit) |
| ESI      | Callee-saved            | Preserved across calls            |
| EDI      | Callee-saved            | Preserved across calls            |
| EBP      | Frame pointer           | Callee-saved frame pointer        |
| ESP      | Stack pointer           | Stack pointer (not allocatable)   |

**x87 FPU registers:** ST(0) through ST(7) — 80-bit extended-precision FPU register stack.
Floating-point arguments and return values use the x87 FPU stack.

### 3.3 Parameter Passing

In the cdecl calling convention, **all arguments are passed on the stack**:

- Arguments are pushed in **right-to-left** order (the first argument is at the lowest stack
  address, closest to the top of the stack at the point of the call).
- Each argument occupies at least 4 bytes on the stack (smaller types are padded to 4 bytes).
- `long long` (8 bytes) is pushed as two 4-byte words (low word at lower address).
- `double` (8 bytes) is pushed as two 4-byte words.
- `long double` (12 bytes) is pushed as three 4-byte words.
- Struct arguments are copied onto the stack in their entirety, with padding to 4-byte alignment.

**There are no register arguments in the standard cdecl convention.**

**Stack alignment:**

- The traditional i386 ABI requires only 4-byte stack alignment.
- The modern Linux i386 ABI (as used by GCC with `-mpreferred-stack-boundary=4`) requires
  16-byte stack alignment at `call` instructions for SSE compatibility. BCC supports both
  modes and defaults to 16-byte alignment for compatibility.

### 3.4 Return Values

| Return Category              | Location         | Notes                                       |
|------------------------------|------------------|----------------------------------------------|
| Integer ≤ 32 bits            | EAX              | Zero- or sign-extended                       |
| Integer 33–64 bits           | EAX (low), EDX (high) | `long long` returned in register pair   |
| `float`                      | ST(0)            | x87 FPU top of stack                         |
| `double`                     | ST(0)            | x87 FPU top of stack                         |
| `long double`                | ST(0)            | x87 FPU top of stack (80-bit precision)      |
| Struct ≤ 8 bytes             | EAX, optionally EDX | Implementation-defined; GCC returns small structs in EAX:EDX |
| Struct > 8 bytes             | Caller-allocated  | Hidden pointer as first (stack) argument     |

For structs returned via hidden pointer:

- The caller allocates space for the return value and pushes a pointer to that space as the
  first argument (at the bottom of the argument area on the stack).
- The callee writes the return value through this pointer and returns the pointer in EAX.
- This hidden pointer consumes a stack slot but does not shift other arguments (it is placed
  before the first declared argument).

### 3.5 Callee-Saved and Caller-Saved Registers

**Callee-saved** (must be preserved by the called function):

- EBX, ESI, EDI, EBP

**Caller-saved** (may be clobbered by any function call):

- EAX, ECX, EDX
- x87 FPU stack (ST(0)–ST(7)) — caller-saved
- EFLAGS — caller-saved (direction flag DF must be clear on entry/exit)

### 3.6 Struct Passing

Structs are passed by value on the stack:

- The entire struct is copied onto the stack, respecting the struct's natural alignment
  requirements (padded to 4-byte boundary).
- Very large structs may cause significant stack usage; callers must account for this.
- Bitfield layout follows the System V i386 ABI rules (big-endian bit numbering within
  little-endian byte storage units).

### 3.7 Stack Frame Layout

```
High addresses
┌──────────────────────────┐
│   Caller's frame          │
├──────────────────────────┤
│   Arguments (right-to-left)│  ← arg N, ..., arg 2, arg 1
├──────────────────────────┤
│   Return address (4 bytes)│  ← pushed by CALL instruction
├──────────────────────────┤
│   Saved EBP (4 bytes)    │  ← pushed by callee (if frame pointer used)
├──────────────────────────┤  ← EBP points here
│   Local variables         │
│   Spill slots             │
│   Saved callee-saved regs │
├──────────────────────────┤  ← ESP points here
└──────────────────────────┘
Low addresses
```

**Caller responsibility:** After the call returns, the caller cleans up the arguments from the
stack (cdecl is caller-cleanup). This is typically done by adjusting ESP:
`add esp, <arg_bytes>`.

### 3.8 x87 Floating-Point Considerations

- All floating-point arithmetic on i686 uses the x87 FPU by default.
- The x87 FPU operates with 80-bit extended precision internally, regardless of the C type
  (`float`, `double`, or `long double`).
- Floating-point arguments are pushed onto the stack (not into FPU registers) for function calls.
- Floating-point return values are left in ST(0).
- The FPU control word (precision, rounding mode) must be caller-saved if modified.
- BCC ensures the x87 FPU stack is properly balanced — each function must leave the FPU stack
  in the same state as entry, except for a single return value in ST(0) if applicable.

### 3.9 Implementation Reference

- **Source file:** `src/backend/i686/abi.rs`
- **Register definitions:** `src/backend/i686/registers.rs`
- **Code generation:** `src/backend/i686/codegen.rs`

---

## 4. AArch64 — AAPCS64 ABI

The AAPCS64 (Procedure Call Standard for the Arm 64-bit Architecture) is the calling convention
for 64-bit ARM Linux systems. It provides register-rich parameter passing with dedicated handling
for homogeneous floating-point aggregates (HFAs) and homogeneous vector aggregates (HVAs).

### 4.1 Data Model

The AArch64 target uses the **LP64** data model:

| C Type            | Size (bytes) | Alignment (bytes) |
|-------------------|--------------|--------------------|
| `_Bool`           | 1            | 1                  |
| `char`            | 1            | 1                  |
| `short`           | 2            | 2                  |
| `int`             | 4            | 4                  |
| `long`            | 8            | 8                  |
| `long long`       | 8            | 8                  |
| `float`           | 4            | 4                  |
| `double`          | 8            | 8                  |
| `long double`     | 16           | 16                 |
| `_Complex float`  | 8            | 4                  |
| `_Complex double` | 16           | 8                  |
| pointer           | 8            | 8                  |
| `size_t`          | 8            | 8                  |
| `ptrdiff_t`       | 8            | 8                  |
| `wchar_t`         | 4            | 4                  |

### 4.2 Register File

AArch64 provides 31 general-purpose registers and 32 SIMD/FP registers:

**General-Purpose Registers (64-bit: X0–X30, 32-bit view: W0–W30):**

| Register | ABI Name          | Purpose                              |
|----------|--------------------|--------------------------------------|
| X0       | Arg 1 / Return     | 1st integer arg / integer return     |
| X1       | Arg 2 / Return     | 2nd integer arg / return (high)      |
| X2       | Arg 3              | 3rd integer argument                 |
| X3       | Arg 4              | 4th integer argument                 |
| X4       | Arg 5              | 5th integer argument                 |
| X5       | Arg 6              | 6th integer argument                 |
| X6       | Arg 7              | 7th integer argument                 |
| X7       | Arg 8              | 8th integer argument                 |
| X8       | Indirect result    | Indirect result location register    |
| X9–X15   | Scratch            | Caller-saved temporaries             |
| X16      | IP0                | Intra-procedure-call scratch         |
| X17      | IP1                | Intra-procedure-call scratch         |
| X18      | Platform register  | Platform-reserved (not for general use on Linux) |
| X19–X28  | Callee-saved       | Preserved across calls               |
| X29      | FP (Frame pointer) | Frame pointer (callee-saved)         |
| X30      | LR (Link register) | Return address (callee-saved)        |

**Special registers:**

| Register | Purpose                                          |
|----------|--------------------------------------------------|
| SP       | Stack pointer (separate from X0–X30)             |
| XZR/WZR  | Zero register — reads as 0, writes are discarded |
| PC       | Program counter (not directly addressable)       |
| NZCV     | Condition flags (N, Z, C, V)                     |

**SIMD/FP Registers (128-bit: V0–V31, with sub-register views):**

| Register | ABI Name        | Purpose                             |
|----------|------------------|--------------------------------------|
| V0       | FP Arg 1 / Return| 1st FP/SIMD arg / FP return          |
| V1       | FP Arg 2        | 2nd FP/SIMD argument                 |
| V2       | FP Arg 3        | 3rd FP/SIMD argument                 |
| V3       | FP Arg 4        | 4th FP/SIMD argument                 |
| V4       | FP Arg 5        | 5th FP/SIMD argument                 |
| V5       | FP Arg 6        | 6th FP/SIMD argument                 |
| V6       | FP Arg 7        | 7th FP/SIMD argument                 |
| V7       | FP Arg 8        | 8th FP/SIMD argument                 |
| V8–V15   | Callee-saved     | Lower 64 bits preserved (D8–D15)    |
| V16–V31  | Scratch          | Caller-saved                         |

**Sub-register naming convention:**

| Name   | Bits  | Example      |
|--------|-------|--------------|
| Bn     | 8     | B0 = V0[7:0] |
| Hn     | 16    | H0 = V0[15:0] |
| Sn     | 32    | S0 = V0[31:0] |
| Dn     | 64    | D0 = V0[63:0] |
| Qn     | 128   | Q0 = V0[127:0] |

### 4.3 Parameter Passing

Integer and pointer arguments are passed in registers X0–X7:

1. **X0** — 1st integer/pointer argument
2. **X1** — 2nd integer/pointer argument
3. **X2** — 3rd integer/pointer argument
4. **X3** — 4th integer/pointer argument
5. **X4** — 5th integer/pointer argument
6. **X5** — 6th integer/pointer argument
7. **X6** — 7th integer/pointer argument
8. **X7** — 8th integer/pointer argument

Floating-point and SIMD arguments are passed in registers V0–V7:

1. **V0** — 1st floating-point/SIMD argument (as S0 for float, D0 for double, Q0 for 128-bit)
2. **V1** — 2nd floating-point/SIMD argument
3. **V2** through **V7** — 3rd through 8th floating-point/SIMD arguments

**Rules:**

- Integer and floating-point argument sequences are tracked independently (same as x86-64).
- Arguments that do not fit in registers are passed on the stack.
- Stack arguments are 8-byte aligned (natural alignment, minimum 8 bytes per slot).
- Structs ≤ 16 bytes may be passed in one or two registers.
- Structs > 16 bytes are passed by reference (caller copies to a temporary and passes a pointer).
- HFA/HVA types receive special treatment (see Section 4.6).
- The stack pointer must be 16-byte aligned at all times.

### 4.4 Return Values

| Return Category              | Registers Used       | Notes                                       |
|------------------------------|----------------------|----------------------------------------------|
| Integer ≤ 64 bits            | X0                   | Zero- or sign-extended as appropriate        |
| Integer 65–128 bits          | X0 (low), X1 (high) | Two-register return                          |
| `float`                      | S0 (= V0[31:0])     | Scalar FP return                             |
| `double`                     | D0 (= V0[63:0])     | Scalar FP return                             |
| `long double`                | Q0 (= V0[127:0])    | 128-bit quad-precision                       |
| Struct ≤ 16 bytes            | X0, optionally X1    | Or in V registers if HFA                    |
| HFA (≤ 4 members)            | V0–V3 (consecutive)  | One register per HFA member                 |
| Struct > 16 bytes            | Indirect via X8      | Caller allocates, passes pointer in X8      |

When a struct is returned indirectly:

- The caller allocates memory for the return value.
- The address of this memory is passed in X8 (the indirect result location register).
- The callee writes the return value to the memory pointed to by X8.
- X8 is NOT an argument register — it does not consume an argument slot.

### 4.5 Callee-Saved and Caller-Saved Registers

**Callee-saved** (must be preserved by the called function):

- X19, X20, X21, X22, X23, X24, X25, X26, X27, X28 (10 GPRs)
- X29 (FP — frame pointer)
- X30 (LR — link register)
- V8–V15 (lower 64 bits only; the upper 64 bits of V8–V15 are not preserved)

**Caller-saved** (may be clobbered by any function call):

- X0–X18 (includes argument registers, indirect result register, scratch, and platform register)
- V0–V7 (argument/return registers)
- V16–V31 (scratch SIMD/FP registers)
- NZCV (condition flags)

### 4.6 HFA and HVA Passing

**Homogeneous Floating-point Aggregate (HFA):**

An HFA is a struct (or array) that contains 1 to 4 members, all of the same floating-point type
(`float`, `double`, or `long double`). HFAs receive privileged register allocation:

- An HFA with N members (1 ≤ N ≤ 4) is passed in N consecutive SIMD/FP registers.
- Example: `struct { float x, y, z; }` is an HFA with 3 float members → passed in S0, S1, S2.
- Example: `struct { double a, b; }` is an HFA with 2 double members → passed in D0, D1.
- If insufficient SIMD/FP registers remain, the entire HFA is passed on the stack (not split).

**Homogeneous Vector Aggregate (HVA):**

An HVA is analogous to an HFA but for SIMD vector types. BCC treats these similarly to HFAs,
using consecutive V registers for passing.

**HFA/HVA classification rules:**

1. The struct must contain only floating-point members (no integers, no pointers).
2. All members must be the same base floating-point type.
3. The struct must have between 1 and 4 members (inclusive).
4. Nested structs are flattened for classification purposes.
5. Arrays of floating-point types count as multiple members of the element type.
6. A struct failing any of these rules is NOT an HFA/HVA and follows normal struct passing.

**Return value HFA handling:**

- An HFA return value with N members is returned in V0 through V(N-1).
- Example: A function returning `struct { float x, y; }` returns x in S0 and y in S1.

### 4.7 Stack Frame Layout

```
High addresses
┌──────────────────────────┐
│   Caller's frame          │
├──────────────────────────┤
│   Stack arguments         │  ← args that didn't fit in registers
├──────────────────────────┤
│   Return address (LR)    │  ← saved by callee (STP X29, X30, [SP, #-N]!)
│   Saved FP (X29)         │
├──────────────────────────┤  ← X29 (FP) points here
│   Saved callee-saved regs │  ← X19–X28, D8–D15 as needed
├──────────────────────────┤
│   Local variables         │
│   Spill slots             │
├──────────────────────────┤  ← SP points here
└──────────────────────────┘
Low addresses
```

**Stack alignment:**

- SP must be 16-byte aligned at all times (hardware-enforced on AArch64).
- Store pair (STP) and load pair (LDP) instructions are used for efficient 16-byte aligned
  register saving/restoring.

### 4.8 Implementation Reference

- **Source file:** `src/backend/aarch64/abi.rs`
- **Register definitions:** `src/backend/aarch64/registers.rs`
- **Code generation:** `src/backend/aarch64/codegen.rs`

---

## 5. RISC-V 64 — LP64D ABI

The RISC-V LP64D ABI is the standard calling convention for 64-bit RISC-V Linux systems with
hardware double-precision floating-point support. It is the ABI used by the Linux kernel on
RISC-V 64, making it the critical ABI for BCC's Checkpoint 6 (kernel build and boot).

### 5.1 Data Model

The RISC-V 64 target uses the **LP64** data model with the **LP64D** ABI variant (D for
double-precision hardware FP):

| C Type            | Size (bytes) | Alignment (bytes) |
|-------------------|--------------|--------------------|
| `_Bool`           | 1            | 1                  |
| `char`            | 1            | 1                  |
| `short`           | 2            | 2                  |
| `int`             | 4            | 4                  |
| `long`            | 8            | 8                  |
| `long long`       | 8            | 8                  |
| `float`           | 4            | 4                  |
| `double`          | 8            | 8                  |
| `long double`     | 16           | 16                 |
| `_Complex float`  | 8            | 4                  |
| `_Complex double` | 16           | 8                  |
| pointer           | 8            | 8                  |
| `size_t`          | 8            | 8                  |
| `ptrdiff_t`       | 8            | 8                  |
| `wchar_t`         | 4            | 4                  |

**ISA:** RV64IMAFDC

| Extension | Meaning                              |
|-----------|--------------------------------------|
| I         | Base integer instruction set (RV64I) |
| M         | Integer multiplication and division  |
| A         | Atomic instructions                  |
| F         | Single-precision floating-point      |
| D         | Double-precision floating-point      |
| C         | Compressed instructions (16-bit)     |

### 5.2 Register File

RISC-V 64 provides 32 integer registers and 32 floating-point registers:

**Integer Registers (64-bit):**

| Register | ABI Name | Description                         |
|----------|----------|--------------------------------------|
| x0       | zero     | Hardwired zero (reads 0, writes ignored) |
| x1       | ra       | Return address                       |
| x2       | sp       | Stack pointer                        |
| x3       | gp       | Global pointer                       |
| x4       | tp       | Thread pointer                       |
| x5       | t0       | Temporary / alternate link register  |
| x6       | t1       | Temporary                            |
| x7       | t2       | Temporary                            |
| x8       | s0 / fp  | Saved register / frame pointer       |
| x9       | s1       | Saved register                       |
| x10      | a0       | 1st argument / return value          |
| x11      | a1       | 2nd argument / return value (high)   |
| x12      | a2       | 3rd argument                         |
| x13      | a3       | 4th argument                         |
| x14      | a4       | 5th argument                         |
| x15      | a5       | 6th argument                         |
| x16      | a6       | 7th argument                         |
| x17      | a7       | 8th argument                         |
| x18      | s2       | Saved register                       |
| x19      | s3       | Saved register                       |
| x20      | s4       | Saved register                       |
| x21      | s5       | Saved register                       |
| x22      | s6       | Saved register                       |
| x23      | s7       | Saved register                       |
| x24      | s8       | Saved register                       |
| x25      | s9       | Saved register                       |
| x26      | s10      | Saved register                       |
| x27      | s11      | Saved register                       |
| x28      | t3       | Temporary                            |
| x29      | t4       | Temporary                            |
| x30      | t5       | Temporary                            |
| x31      | t6       | Temporary                            |

**Floating-Point Registers (64-bit double-precision):**

| Register | ABI Name | Description                         |
|----------|----------|--------------------------------------|
| f0       | ft0      | FP temporary                        |
| f1       | ft1      | FP temporary                        |
| f2       | ft2      | FP temporary                        |
| f3       | ft3      | FP temporary                        |
| f4       | ft4      | FP temporary                        |
| f5       | ft5      | FP temporary                        |
| f6       | ft6      | FP temporary                        |
| f7       | ft7      | FP temporary                        |
| f8       | fs0      | FP saved register                   |
| f9       | fs1      | FP saved register                   |
| f10      | fa0      | 1st FP argument / FP return value   |
| f11      | fa1      | 2nd FP argument                     |
| f12      | fa2      | 3rd FP argument                     |
| f13      | fa3      | 4th FP argument                     |
| f14      | fa4      | 5th FP argument                     |
| f15      | fa5      | 6th FP argument                     |
| f16      | fa6      | 7th FP argument                     |
| f17      | fa7      | 8th FP argument                     |
| f18      | fs2      | FP saved register                   |
| f19      | fs3      | FP saved register                   |
| f20      | fs4      | FP saved register                   |
| f21      | fs5      | FP saved register                   |
| f22      | fs6      | FP saved register                   |
| f23      | fs7      | FP saved register                   |
| f24      | fs8      | FP saved register                   |
| f25      | fs9      | FP saved register                   |
| f26      | fs10     | FP saved register                   |
| f27      | fs11     | FP saved register                   |
| f28      | ft8      | FP temporary                        |
| f29      | ft9      | FP temporary                        |
| f30      | ft10     | FP temporary                        |
| f31      | ft11     | FP temporary                        |

### 5.3 Parameter Passing

Integer and pointer arguments are passed in registers a0–a7 (x10–x17):

1. **a0 (x10)** — 1st integer/pointer argument
2. **a1 (x11)** — 2nd integer/pointer argument
3. **a2 (x12)** — 3rd integer/pointer argument
4. **a3 (x13)** — 4th integer/pointer argument
5. **a4 (x14)** — 5th integer/pointer argument
6. **a5 (x15)** — 6th integer/pointer argument
7. **a6 (x16)** — 7th integer/pointer argument
8. **a7 (x17)** — 8th integer/pointer argument

Floating-point arguments are passed in registers fa0–fa7 (f10–f17):

1. **fa0 (f10)** — 1st floating-point argument
2. **fa1 (f11)** — 2nd floating-point argument
3. **fa2 (f12)** — 3rd floating-point argument
4. **fa3 (f13)** — 4th floating-point argument
5. **fa4 (f14)** — 5th floating-point argument
6. **fa5 (f15)** — 6th floating-point argument
7. **fa6 (f16)** — 7th floating-point argument
8. **fa7 (f17)** — 8th floating-point argument

**Rules:**

- Integer and floating-point argument sequences are tracked independently.
- Scalars wider than the register size (e.g., `long double` at 128 bits) are passed in a pair
  of registers if two are available, or partly in a register and partly on the stack, or
  entirely on the stack.
- Structs ≤ 2×XLEN (16 bytes on RV64) may be passed in up to two integer registers.
- Structs > 2×XLEN are passed by reference (caller copies and passes a pointer in an integer
  argument register).
- Stack arguments are aligned to XLEN (8 bytes on RV64).
- The stack pointer must be 16-byte aligned at all times.
- Small integer types (`char`, `short`) are sign- or zero-extended to the full register width
  (XLEN = 64 bits) when passed in registers.

### 5.4 Return Values

| Return Category              | Registers Used       | Notes                                       |
|------------------------------|----------------------|----------------------------------------------|
| Integer ≤ 64 bits            | a0 (x10)             | Sign- or zero-extended to XLEN              |
| Integer 65–128 bits          | a0 (low), a1 (high)  | Two-register return                          |
| `float`                      | fa0 (f10)            | Scalar FP return (NaN-boxed in 64-bit freg) |
| `double`                     | fa0 (f10)            | Scalar FP return                             |
| Struct ≤ 2×XLEN              | a0, optionally a1    | Packed into integer registers                |
| Struct with FP fields        | fa0, fa1, or a0+fa0  | See float-int struct passing rules           |
| Struct > 2×XLEN              | Indirect via a0      | Caller allocates, passes pointer in a0      |

When a struct is returned indirectly:

- The caller allocates memory for the return value.
- The address of this memory is passed as a hidden first argument in a0.
- This consumes the a0 argument slot, shifting all other integer arguments by one register.
- The callee writes the return value through the pointer and returns the address in a0.

### 5.5 Callee-Saved and Caller-Saved Registers

**Callee-saved** (must be preserved by the called function):

- **Integer:** s0–s11 (x8–x9, x18–x27) — 12 saved registers
- **Floating-point:** fs0–fs11 (f8–f9, f18–f27) — 12 saved FP registers
- **ra (x1):** The return address register is callee-saved (the callee must save it if it
  makes further calls)

**Caller-saved** (may be clobbered by any function call):

- **Integer:** t0–t6 (x5–x7, x28–x31) — 7 temporary registers
- **Integer:** a0–a7 (x10–x17) — 8 argument registers
- **Floating-point:** ft0–ft11 (f0–f7, f28–f31) — 12 temporary FP registers
- **Floating-point:** fa0–fa7 (f10–f17) — 8 FP argument registers

### 5.6 Struct and Floating-Point Passing

The RISC-V LP64D ABI has special rules for structs containing floating-point fields, designed to
pass them efficiently in FP registers when possible:

**Rule 1 — Pure FP struct (1 or 2 FP members, ≤ 2×FLEN):**

If a struct contains only 1 or 2 floating-point members (and no integer members), and the total
size does not exceed 2×FLEN (16 bytes for double-precision), the members are passed in FP registers:

- `struct { float x; }` → fa0 (as float)
- `struct { double x; }` → fa0 (as double)
- `struct { float x; float y; }` → fa0, fa1
- `struct { double x; double y; }` → fa0, fa1

**Rule 2 — Mixed int+FP struct (1 int + 1 FP, ≤ 2×XLEN):**

If a struct contains exactly one integer member and one floating-point member, and fits within
2×XLEN (16 bytes), the integer part is passed in an integer register and the FP part in an FP
register:

- `struct { int i; float f; }` → a0 (int part), fa0 (float part)
- `struct { long l; double d; }` → a0 (long part), fa0 (double part)

**Rule 3 — Fallback to integer registers:**

If no FP registers are available, or the struct does not match the patterns above, the struct
is flattened into integer registers (up to 2×XLEN) or passed on the stack.

**Rule 4 — Large structs:**

Structs exceeding 2×XLEN (16 bytes) are passed by reference — the caller copies the struct
to a temporary location and passes a pointer in an integer argument register.

### 5.7 Stack Frame Layout

```
High addresses
┌──────────────────────────┐
│   Caller's frame          │
├──────────────────────────┤
│   Stack arguments         │  ← args that didn't fit in registers
├──────────────────────────┤
│   Return address (ra)     │  ← saved by callee (SD ra, offset(sp))
│   Saved FP (s0)           │  ← if frame pointer is used
├──────────────────────────┤  ← s0 (FP) points here (if used)
│   Saved callee-saved regs │  ← s1–s11, fs0–fs11 as needed
├──────────────────────────┤
│   Local variables         │
│   Spill slots             │
├──────────────────────────┤  ← sp points here
└──────────────────────────┘
Low addresses
```

**Stack alignment:**

- SP must be 16-byte aligned at all times.
- Function prologues adjust SP downward by a 16-byte-aligned frame size.
- The RISC-V C extension (compressed instructions) requires 16-bit instruction alignment but does
  not affect stack alignment requirements.

### 5.8 Large Immediates and Addressing

RISC-V instructions have limited immediate fields, requiring multi-instruction sequences for
large constants and addresses:

| Pattern        | Instructions            | Purpose                            |
|----------------|-------------------------|------------------------------------|
| LUI + ADDI     | `lui rd, hi20; addi rd, rd, lo12` | Load 32-bit constant       |
| AUIPC + ADDI   | `auipc rd, hi20; addi rd, rd, lo12` | PC-relative address (PIC) |
| AUIPC + JALR   | `auipc rd, hi20; jalr ra, lo12(rd)` | PC-relative function call |
| AUIPC + LD     | `auipc rd, hi20; ld rd, lo12(rd)` | PC-relative GOT load (PIC) |

These addressing patterns are critical for PIC code generation and GOT/PLT access in shared
libraries. The `%hi()` and `%lo()` relocations split a 32-bit value across the two instructions.

BCC's RISC-V assembler (`src/backend/riscv64/assembler/`) handles these patterns and emits
the appropriate relocations (`R_RISCV_HI20`, `R_RISCV_LO12_I`, `R_RISCV_LO12_S`,
`R_RISCV_PCREL_HI20`, `R_RISCV_PCREL_LO12_I`, etc.).

**Linker relaxation:** The RISC-V linker may relax multi-instruction sequences into shorter
forms when the target is close enough. BCC's built-in linker (`src/backend/riscv64/linker/`)
supports this optimization.

### 5.9 Implementation Reference

- **Source file:** `src/backend/riscv64/abi.rs`
- **Register definitions:** `src/backend/riscv64/registers.rs`
- **Code generation:** `src/backend/riscv64/codegen.rs`
- **Assembler:** `src/backend/riscv64/assembler/mod.rs`

---

## 6. Cross-Architecture Comparison

### 6.1 Data Models

| Property          | x86-64 (LP64) | i686 (ILP32) | AArch64 (LP64) | RISC-V 64 (LP64) |
|-------------------|----------------|--------------|-----------------|-------------------|
| `char` size       | 1              | 1            | 1               | 1                 |
| `short` size      | 2              | 2            | 2               | 2                 |
| `int` size        | 4              | 4            | 4               | 4                 |
| `long` size       | 8              | 4            | 8               | 8                 |
| `long long` size  | 8              | 8            | 8               | 8                 |
| `pointer` size    | 8              | 4            | 8               | 8                 |
| `float` size      | 4              | 4            | 4               | 4                 |
| `double` size     | 8              | 8            | 8               | 8                 |
| `long double` size| 16             | 12           | 16              | 16                |
| `size_t` size     | 8              | 4            | 8               | 8                 |

### 6.2 Argument Passing Summary

| Property                    | x86-64        | i686            | AArch64       | RISC-V 64       |
|-----------------------------|---------------|-----------------|---------------|-----------------|
| Integer arg registers       | 6 (RDI..R9)  | 0 (all stack)   | 8 (X0–X7)    | 8 (a0–a7)      |
| FP arg registers            | 8 (XMM0–XMM7)| 0 (all stack)   | 8 (V0–V7)    | 8 (fa0–fa7)    |
| Stack arg push order        | Right-to-left | Right-to-left   | Left-to-right | Left-to-right   |
| Stack arg cleanup           | Caller        | Caller (cdecl)  | Caller        | Caller          |
| Variadic special handling   | AL = SSE count| None            | None          | None            |
| Small struct in registers   | ≤ 16 bytes    | No (stack)      | ≤ 16 bytes    | ≤ 16 bytes      |
| HFA/HVA support             | No            | No              | Yes (≤ 4 members) | Partial (≤ 2 FP) |

### 6.3 Return Value Conventions

| Property                    | x86-64           | i686             | AArch64          | RISC-V 64        |
|-----------------------------|------------------|------------------|------------------|------------------|
| Integer return register     | RAX              | EAX              | X0               | a0 (x10)         |
| Integer return (wide)       | RAX + RDX        | EAX + EDX        | X0 + X1          | a0 + a1          |
| FP return register          | XMM0             | ST(0) (x87)      | V0               | fa0 (f10)        |
| Large struct return         | Hidden ptr in RDI| Hidden ptr on stack | Indirect via X8 | Hidden ptr in a0 |
| Hidden ptr consumes arg slot| Yes (shifts args)| Yes (stack slot) | No (X8 separate) | Yes (shifts args)|

### 6.4 Register Counts and Classification

| Property                    | x86-64  | i686    | AArch64 | RISC-V 64 |
|-----------------------------|---------|---------|---------|-----------|
| Total GPRs                  | 16      | 8       | 31      | 32        |
| Callee-saved GPRs           | 6       | 4       | 12      | 13        |
| Caller-saved GPRs           | 10      | 3       | 19      | 15        |
| Total FP/SIMD registers     | 16 (SSE)| 8 (x87) | 32      | 32        |
| Callee-saved FP registers   | 0       | 0       | 8 (low 64 bits) | 12   |
| Caller-saved FP registers   | 16      | 8       | 24      | 20        |
| Zero register               | No      | No      | XZR/WZR | x0 (zero) |

### 6.5 Stack Alignment and Frame Conventions

| Property                    | x86-64       | i686           | AArch64        | RISC-V 64      |
|-----------------------------|--------------|----------------|----------------|----------------|
| Stack alignment at call     | 16-byte      | 4-byte (or 16) | 16-byte        | 16-byte        |
| Red zone                    | 128 bytes    | None           | None           | None           |
| Frame pointer register      | RBP          | EBP            | X29 (FP)       | s0/x8 (fp)     |
| Link register               | Stack (RIP)  | Stack (EIP)    | X30 (LR)       | ra (x1)        |
| Return address storage      | CALL pushes  | CALL pushes    | BL writes LR   | JAL writes ra  |
| Stack growth direction      | Downward     | Downward       | Downward       | Downward       |

---

## 7. Type System Bridge

### 7.1 Dual Type System Overview

BCC maintains a dual type system that bridges the gap between C language-level types and
machine-level register classes:

```
C Source Types           IR Types              Machine Types / Register Classes
(src/common/types.rs)    (src/ir/types.rs)     (src/backend/*/abi.rs)
─────────────────────    ─────────────────     ────────────────────────────────
CType::Int           →   IrType::I32       →   INTEGER class (GPR)
CType::Long          →   IrType::I64       →   INTEGER class (GPR)
CType::Float         →   IrType::F32       →   SSE/FP class (XMM/V/fa)
CType::Double        →   IrType::F64       →   SSE/FP class (XMM/V/fa)
CType::LongDouble    →   IrType::F80       →   X87/MEMORY (arch-dependent)
CType::Pointer       →   IrType::Ptr       →   INTEGER class (GPR)
CType::Struct        →   IrType::Struct     →   Classified per ABI rules
CType::Array         →   IrType::Array      →   MEMORY (decayed to pointer)
CType::Function      →   IrType::Function   →   (not passed by value)
```

The type system flow through the pipeline:

1. **Frontend** (`src/frontend/sema/`): Constructs `CType` values during semantic analysis, using
   `src/common/type_builder.rs` for complex types.
2. **IR Lowering** (`src/ir/lowering/`): Converts `CType` to `IrType` during AST-to-IR
   translation. Type sizes and alignments are target-dependent, queried from `src/common/target.rs`.
3. **Code Generation** (`src/backend/*/abi.rs`): The ABI module maps `IrType` (or the
   original `CType` when ABI classification requires it) to architecture-specific register classes
   for parameter passing, return values, and struct layout.

### 7.2 C Type to IR Type Mapping

| C Type (`CType`)     | IR Type (`IrType`)   | Notes                                      |
|----------------------|----------------------|---------------------------------------------|
| `_Bool`              | `I1`                 | 1-bit boolean, zero-extended to i8 in memory|
| `char`               | `I8`                 | Signedness is target-dependent (signed on x86, AArch64, RISC-V) |
| `signed char`        | `I8`                 | Explicitly signed                           |
| `unsigned char`      | `I8`                 | Zero-extended on load                       |
| `short`              | `I16`                | Sign-extended on load                       |
| `unsigned short`     | `I16`                | Zero-extended on load                       |
| `int`                | `I32`                | All architectures                           |
| `unsigned int`       | `I32`                | Zero-extended on load                       |
| `long`               | `I64` (LP64), `I32` (ILP32) | Target-dependent                   |
| `unsigned long`      | `I64` (LP64), `I32` (ILP32) | Target-dependent                   |
| `long long`          | `I64`                | All architectures                           |
| `unsigned long long` | `I64`                | All architectures                           |
| `__int128`           | `I128`               | GCC extension                               |
| `float`              | `F32`                | IEEE 754 single-precision                   |
| `double`             | `F64`                | IEEE 754 double-precision                   |
| `long double`        | `F80`                | 80-bit extended (x86), 128-bit quad (AArch64, RISC-V) |
| `T *` (pointer)      | `Ptr`                | Opaque pointer, size from target            |
| `T[N]` (array)       | `Array(T_ir, N)`     | Decays to pointer in most expression contexts |
| `struct { ... }`     | `Struct(fields)`     | Field offsets computed with padding/alignment |
| `union { ... }`      | `Struct(max_field)`  | Represented as largest member               |
| `enum`               | `I32`                | Default; may vary with GCC `packed` attribute|
| `void`               | `Void`               | Only valid as function return or pointer target |
| `_Atomic T`          | Same as `T`          | Same IR type; atomic operations are distinct instructions |
| `_Complex float`     | `Struct([F32, F32])` | Lowered as a two-field struct               |
| `_Complex double`    | `Struct([F64, F64])` | Lowered as a two-field struct               |

### 7.3 IR Type to Register Class Mapping

Each architecture's ABI module (`src/backend/*/abi.rs`) maps IR types to register classes:

**x86-64 Register Class Mapping:**

| IR Type        | Register Class | Notes                                     |
|----------------|----------------|-------------------------------------------|
| `I1`–`I64`     | INTEGER        | General-purpose register (RAX, RDI, etc.) |
| `I128`         | INTEGER ×2     | Two GPRs (RAX+RDX for return, two arg regs)|
| `F32`, `F64`   | SSE            | XMM register                              |
| `F80`          | X87            | x87 FPU stack ST(0)                       |
| `Ptr`          | INTEGER        | Same as I64 on x86-64                     |
| `Struct`       | Classified     | Run classification algorithm (Section 2.6) |

**i686 Register Class Mapping:**

| IR Type        | Register Class | Notes                                     |
|----------------|----------------|-------------------------------------------|
| `I1`–`I32`     | INTEGER        | General-purpose register (EAX, etc.)      |
| `I64`          | INTEGER ×2     | Register pair (EAX:EDX)                   |
| `F32`, `F64`   | X87            | x87 FPU stack                             |
| `F80`          | X87            | x87 FPU stack (native 80-bit)             |
| `Ptr`          | INTEGER        | Same as I32 on i686                       |
| `Struct`       | MEMORY         | Always passed on stack                    |

**AArch64 Register Class Mapping:**

| IR Type        | Register Class | Notes                                     |
|----------------|----------------|-------------------------------------------|
| `I1`–`I64`     | INTEGER        | General-purpose register (X0, etc.)       |
| `I128`         | INTEGER ×2     | Register pair (X0+X1)                     |
| `F32`          | FP_S           | Single-precision in Sn sub-register       |
| `F64`          | FP_D           | Double-precision in Dn sub-register       |
| `F80`/`F128`   | FP_Q           | Quad-precision in Qn register             |
| `Ptr`          | INTEGER        | Same as I64 on AArch64                    |
| `Struct(HFA)`  | FP ×N          | 1–4 consecutive V registers               |
| `Struct(other)` | INTEGER / MEMORY | ≤ 16 bytes in X regs, else indirect    |

**RISC-V 64 Register Class Mapping:**

| IR Type        | Register Class | Notes                                     |
|----------------|----------------|-------------------------------------------|
| `I1`–`I64`     | INTEGER        | General-purpose register (a0, etc.)       |
| `I128`         | INTEGER ×2     | Register pair (a0+a1)                     |
| `F32`          | FLOAT          | FP register (fa0, etc.), NaN-boxed        |
| `F64`          | DOUBLE         | FP register (fa0, etc.)                   |
| `Ptr`          | INTEGER        | Same as I64 on RV64                       |
| `Struct(FP)`   | FLOAT/DOUBLE   | 1–2 FP members in FP registers            |
| `Struct(mixed)`| INT + FP       | One int reg + one FP reg                  |
| `Struct(other)` | INTEGER / MEMORY | ≤ 2×XLEN in int regs, else indirect   |

### 7.4 Aggregate Layout and ABI Classification

When computing struct/union layout for ABI purposes, BCC follows these steps:

1. **Compute field offsets and padding** using `src/common/type_builder.rs`:
   - Each field is placed at the next offset that satisfies its alignment requirement.
   - Padding bytes are inserted between fields as needed.
   - The `__attribute__((packed))` attribute suppresses inter-field padding.
   - The `__attribute__((aligned(N)))` attribute overrides the natural alignment.
   - The total struct size is rounded up to its alignment.

2. **Determine ABI classification** using the architecture-specific ABI module:
   - **x86-64:** Run the eightbyte classification algorithm (Section 2.6).
   - **i686:** All structs are passed on the stack (MEMORY class).
   - **AArch64:** Check for HFA/HVA (Section 4.6); otherwise classify by size (≤ 16 bytes
     in registers, > 16 bytes indirect).
   - **RISC-V 64:** Check for FP-only or mixed int+FP struct (Section 5.6); otherwise
     classify by size (≤ 2×XLEN in registers, > 2×XLEN indirect).

3. **Assign registers or stack slots** based on classification:
   - MEMORY-classified aggregates are passed by reference (or on the stack for i686).
   - Register-classified aggregates are packed into the assigned register(s).
   - For return values, large aggregates use the hidden pointer mechanism.

This three-step process is implemented in each `src/backend/*/abi.rs` module and invoked by
the code generation driver (`src/backend/generation.rs`) during function call lowering and
function prologue/epilogue generation.

---

*This document is maintained as part of the BCC project. For implementation details, refer to
the source files referenced in each section. For the overall architecture, see
[docs/architecture.md](architecture.md). For ELF output format details, see
[docs/elf_format.md](elf_format.md).*
