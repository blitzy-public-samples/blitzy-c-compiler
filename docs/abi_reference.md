# ABI Reference

BCC (Blitzy's C Compiler) implements architecture-specific Application Binary Interfaces (ABIs) for correct code generation across all four supported target architectures: x86-64, i686, AArch64, and RISC-V 64. Each ABI defines the rules for function calling conventions, parameter passing, return value handling, register preservation, struct classification and passing, stack frame layout, and alignment requirements that the compiler must honor to produce correct, interoperable object code and executables.

BCC employs a **dual type system** to bridge the gap between the C language level and the machine level. C language types — such as `int`, `struct`, `char *`, and `_Atomic int` — are represented in `src/common/types.rs` and constructed via the builder API in `src/common/type_builder.rs`. During code generation, these C types are mapped to machine-specific register classes and memory layouts by the architecture ABI modules. This mapping is critical for correct parameter classification, struct layout computation, return value placement, and alignment enforcement.

The ABI implementations reside in the following source files:

| Architecture | ABI Standard | Source File |
|:-------------|:-------------|:------------|
| x86-64 | System V AMD64 | `src/backend/x86_64/abi.rs` |
| i686 | cdecl / System V i386 | `src/backend/i686/abi.rs` |
| AArch64 | AAPCS64 | `src/backend/aarch64/abi.rs` |
| RISC-V 64 | RISC-V LP64D | `src/backend/riscv64/abi.rs` |

Target-specific constants (pointer widths, endianness, data models) are centralized in `src/common/target.rs` and queried by all ABI modules during type layout and calling convention decisions.

---

## System V AMD64 ABI (x86-64)

The System V AMD64 ABI is the standard calling convention for 64-bit x86 Linux systems. It is the primary ABI implemented in BCC and serves as the first validation target during backend development.

### Data Model

The x86-64 target uses the **LP64** data model:

| C Type | Size (bytes) | Alignment (bytes) |
|:-------|:-------------|:-------------------|
| `_Bool` | 1 | 1 |
| `char` | 1 | 1 |
| `short` | 2 | 2 |
| `int` | 4 | 4 |
| `long` | 8 | 8 |
| `long long` | 8 | 8 |
| `float` | 4 | 4 |
| `double` | 8 | 8 |
| `long double` | 16 | 16 |
| `_Complex float` | 8 | 4 |
| `_Complex double` | 16 | 8 |
| pointer | 8 | 8 |
| `size_t` | 8 | 8 |
| `ptrdiff_t` | 8 | 8 |

> **Note:** `long double` is 80-bit extended precision (x87 format) stored in 16 bytes with 6 bytes of padding. Software long-double arithmetic is implemented in `src/common/long_double.rs`.

### Register Definitions

Defined in `src/backend/x86_64/registers.rs`.

**16 General-Purpose Registers (GPRs):**

| 64-bit | 32-bit | 16-bit | 8-bit (low) | Purpose |
|:-------|:-------|:-------|:-------------|:--------|
| RAX | EAX | AX | AL | Return value, caller-saved |
| RBX | EBX | BX | BL | Callee-saved |
| RCX | ECX | CX | CL | 4th integer argument, caller-saved |
| RDX | EDX | DX | DL | 3rd integer argument, caller-saved |
| RSI | ESI | SI | SIL | 2nd integer argument, caller-saved |
| RDI | EDI | DI | DIL | 1st integer argument, caller-saved |
| RBP | EBP | BP | BPL | Frame pointer, callee-saved |
| RSP | ESP | SP | SPL | Stack pointer (reserved) |
| R8 | R8D | R8W | R8B | 5th integer argument, caller-saved |
| R9 | R9D | R9W | R9B | 6th integer argument, caller-saved |
| R10 | R10D | R10W | R10B | Caller-saved (static chain pointer) |
| R11 | R11D | R11W | R11B | Caller-saved (scratch) |
| R12 | R12D | R12W | R12B | Callee-saved |
| R13 | R13D | R13W | R13B | Callee-saved |
| R14 | R14D | R14W | R14B | Callee-saved |
| R15 | R15D | R15W | R15B | Callee-saved |

**16 SSE Registers:**

| Register | Width | Purpose |
|:---------|:------|:--------|
| XMM0 | 128-bit | 1st FP argument / FP return value, caller-saved |
| XMM1 | 128-bit | 2nd FP argument / 2nd FP return value, caller-saved |
| XMM2 | 128-bit | 3rd FP argument, caller-saved |
| XMM3 | 128-bit | 4th FP argument, caller-saved |
| XMM4 | 128-bit | 5th FP argument, caller-saved |
| XMM5 | 128-bit | 6th FP argument, caller-saved |
| XMM6 | 128-bit | 7th FP argument, caller-saved |
| XMM7 | 128-bit | 8th FP argument, caller-saved |
| XMM8–XMM15 | 128-bit | Caller-saved (scratch) |

**Callee-Saved Registers:** RBX, RBP, R12, R13, R14, R15

**Caller-Saved Registers:** RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, XMM0–XMM15

### Integer Parameter Passing

Integer and pointer arguments are passed in registers, in the following order:

| Argument Position | Register |
|:------------------|:---------|
| 1st | RDI |
| 2nd | RSI |
| 3rd | RDX |
| 4th | RCX |
| 5th | R8 |
| 6th | R9 |
| 7th and beyond | Stack (right-to-left push order) |

Stack-passed arguments are aligned to 8 bytes. Arguments smaller than 8 bytes are zero-extended or sign-extended to fill the full register or stack slot width.

### Floating-Point Parameter Passing

Floating-point arguments (`float`, `double`) are passed in SSE registers:

| Argument Position | Register |
|:------------------|:---------|
| 1st | XMM0 |
| 2nd | XMM1 |
| 3rd | XMM2 |
| 4th | XMM3 |
| 5th | XMM4 |
| 6th | XMM5 |
| 7th | XMM6 |
| 8th | XMM7 |
| 9th and beyond | Stack |

Integer and floating-point arguments consume registers from their respective register files independently. A function `void f(int a, double b, int c)` passes `a` in RDI, `b` in XMM0, and `c` in RSI.

### Return Values

| Return Type | Register(s) |
|:------------|:------------|
| Integer/pointer (≤ 64-bit) | RAX |
| 128-bit integer | RAX (low 64 bits), RDX (high 64 bits) |
| `float` / `double` | XMM0 |
| `_Complex float` / `_Complex double` | XMM0 (real), XMM1 (imaginary) |
| `long double` | ST(0) via x87 FPU |
| Small struct (≤ 16 bytes, classifiable) | RAX and/or XMM0 per classification |
| Large struct (> 16 bytes or MEMORY class) | Hidden pointer in RDI (caller-allocated) |

### Struct Classification

The System V AMD64 ABI classifies each struct for parameter passing and return using a recursive per-eightword algorithm. Each 8-byte segment ("eightword") of the struct is assigned one of the following classes:

| Class | Description |
|:------|:------------|
| `INTEGER` | Integral types and pointers — passed in GPRs |
| `SSE` | Floating-point types — passed in SSE registers |
| `MEMORY` | Struct too large or complex — passed on the stack via hidden pointer |
| `X87` | `long double` lower 8 bytes — passed on the x87 FPU stack |
| `X87UP` | `long double` upper 8 bytes — companion to X87 |
| `COMPLEX_X87` | `_Complex long double` — passed in memory |
| `NO_CLASS` | Padding or empty — inherits the class of the other field in the eightword |

**Classification Rules:**

1. Structs larger than 4 eightwords (32 bytes) are classified as `MEMORY` and passed via a hidden pointer.
2. Structs containing unaligned fields are classified as `MEMORY`.
3. Each eightword is classified independently by inspecting all fields that overlap it.
4. When two fields share an eightword, their classes are merged:
   - If both are `NO_CLASS`, the result is `NO_CLASS`.
   - If one is `MEMORY`, the result is `MEMORY`.
   - If one is `INTEGER`, the result is `INTEGER`.
   - If one is `X87`, `X87UP`, or `COMPLEX_X87`, the result is `MEMORY`.
   - Otherwise, the result is `SSE`.
5. Post-merge cleanup: if any eightword is `MEMORY`, the entire struct is `MEMORY`. If `X87UP` is not preceded by `X87`, the class becomes `MEMORY`.

**Passing Mechanism:**

- `INTEGER` eightwords consume the next available integer register (RDI, RSI, RDX, RCX, R8, R9).
- `SSE` eightwords consume the next available SSE register (XMM0–XMM7).
- `MEMORY` structs are copied to the stack by the caller, and a hidden pointer is passed in the first available integer register.
- If insufficient registers remain for the entire struct, the whole struct falls back to `MEMORY`.

### Stack Layout

```
High addresses
┌─────────────────────────┐
│  Caller's frame         │
├─────────────────────────┤
│  Return address         │ ← pushed by `call`
├─────────────────────────┤ ← RSP at function entry (16-byte aligned before `call`)
│  Saved RBP (if used)    │
├─────────────────────────┤ ← RBP (frame pointer, debug mode)
│  Local variables        │
│  Spill slots            │
│  Callee-saved registers │
├─────────────────────────┤
│  Outgoing arguments     │ ← 16-byte aligned at next `call`
├─────────────────────────┤ ← RSP (current stack pointer)
│  Red zone (128 bytes)   │ ← usable by leaf functions without RSP adjustment
└─────────────────────────┘
Low addresses
```

**Key Properties:**

- The stack pointer RSP must be 16-byte aligned immediately **before** the `call` instruction. After `call` pushes the 8-byte return address, RSP is misaligned by 8 bytes at function entry.
- The **128-byte red zone** below RSP is reserved for the current function's use. Leaf functions (functions that make no further calls) may use this space for temporaries without adjusting RSP, saving the overhead of `sub rsp` / `add rsp` pairs.
- When `-g` (debug mode) is active, BCC emits a frame pointer prologue (`push rbp; mov rbp, rsp`) to support debugger stack unwinding. In optimized builds, the frame pointer may be omitted.
- Stack guard page probing is required for frames exceeding 4,096 bytes (one page). BCC implements a probe loop that touches each page in sequence to ensure the guard page is not skipped, preventing stack clash vulnerabilities. Stack probing is activated automatically when the stack frame size exceeds 4,096 bytes.

### Variadic Functions

For variadic functions (those using `...` in the parameter list):

- The `AL` register (lower 8 bits of RAX) must contain the number of SSE registers used for variable arguments (0–8). This allows the callee to avoid saving unnecessary SSE registers.
- The callee must save all potential argument registers (both integer and SSE) into a **register save area** at the top of its stack frame. This area is accessed by the `va_arg` macro expansion.
- `va_list` on x86-64 is a struct containing:
  - `gp_offset` (unsigned int): offset to the next available integer register save slot
  - `fp_offset` (unsigned int): offset to the next available SSE register save slot
  - `overflow_arg_area` (void *): pointer to the next stack-passed argument
  - `reg_save_area` (void *): pointer to the register save area

---

## System V i386 ABI (i686)

The System V i386 ABI (cdecl calling convention) is the standard for 32-bit x86 Linux systems. It is notable for its simplicity: all parameters are passed on the stack with no register-based parameter passing.

### Data Model

The i686 target uses the **ILP32** data model:

| C Type | Size (bytes) | Alignment (bytes) |
|:-------|:-------------|:-------------------|
| `_Bool` | 1 | 1 |
| `char` | 1 | 1 |
| `short` | 2 | 2 |
| `int` | 4 | 4 |
| `long` | 4 | 4 |
| `long long` | 8 | 4 |
| `float` | 4 | 4 |
| `double` | 8 | 4 |
| `long double` | 12 | 4 |
| `_Complex float` | 8 | 4 |
| `_Complex double` | 16 | 4 |
| pointer | 4 | 4 |
| `size_t` | 4 | 4 |
| `ptrdiff_t` | 4 | 4 |

> **Note:** `long double` is 80-bit extended precision (x87 format) stored in 12 bytes with 2 bytes of padding. This differs from x86-64 where `long double` occupies 16 bytes.

### Register Definitions

Defined in `src/backend/i686/registers.rs`.

**8 General-Purpose Registers (GPRs):**

| 32-bit | 16-bit | 8-bit (low) | 8-bit (high) | Purpose |
|:-------|:-------|:-------------|:-------------|:--------|
| EAX | AX | AL | AH | Return value, caller-saved |
| EBX | BX | BL | BH | Callee-saved |
| ECX | CX | CL | CH | Caller-saved |
| EDX | DX | DL | DH | Caller-saved, high 32 bits of 64-bit return |
| ESI | SI | — | — | Callee-saved |
| EDI | DI | — | — | Callee-saved |
| EBP | BP | — | — | Frame pointer, callee-saved |
| ESP | SP | — | — | Stack pointer (reserved) |

**x87 FPU Stack:**

| Register | Width | Purpose |
|:---------|:------|:--------|
| ST(0) | 80-bit | FPU stack top — FP return value |
| ST(1) | 80-bit | FPU stack |
| ST(2) | 80-bit | FPU stack |
| ST(3) | 80-bit | FPU stack |
| ST(4) | 80-bit | FPU stack |
| ST(5) | 80-bit | FPU stack |
| ST(6) | 80-bit | FPU stack |
| ST(7) | 80-bit | FPU stack bottom |

**Callee-Saved Registers:** EBX, ESI, EDI, EBP

**Caller-Saved Registers:** EAX, ECX, EDX

### Parameter Passing

All parameters are passed on the **stack** in **right-to-left** push order (pure cdecl convention). There is no register-based parameter passing for general-purpose arguments.

```
; Example: calling foo(1, 2, 3)
push 3        ; 3rd argument (pushed first)
push 2        ; 2nd argument
push 1        ; 1st argument
call foo
add esp, 12   ; caller cleans up the stack (3 args × 4 bytes)
```

**Key Rules:**

- Each stack argument occupies at least 4 bytes (the natural word size).
- Arguments smaller than 4 bytes (e.g., `char`, `short`) are promoted to 4 bytes.
- `long long` (64-bit) arguments occupy 8 bytes (two stack slots), with the low 32 bits at the lower address.
- `double` arguments occupy 8 bytes on the stack.
- The **caller** is responsible for cleaning up the stack after the call returns (unlike stdcall where the callee cleans).

### Return Values

| Return Type | Location |
|:------------|:---------|
| Integer (≤ 32-bit) | EAX |
| 64-bit integer (`long long`) | EDX:EAX (EDX = high 32 bits, EAX = low 32 bits) |
| `float` / `double` / `long double` | ST(0) (x87 FPU stack top) |
| Struct (any size) | Hidden pointer — caller allocates space, passes address as hidden first parameter |

**Struct Return Convention:**

When a function returns a struct, the caller allocates space for the return value and passes a hidden pointer to that space as the first argument (pushed last, appearing at the lowest stack address). The callee copies the struct to the pointed-to location and returns the pointer in EAX.

### Stack Layout

```
High addresses
┌─────────────────────────┐
│  Caller's frame         │
├─────────────────────────┤
│  Arguments (right-to-left) │
├─────────────────────────┤
│  Return address         │ ← pushed by `call`
├─────────────────────────┤ ← ESP at function entry
│  Saved EBP              │
├─────────────────────────┤ ← EBP (frame pointer)
│  Local variables        │
│  Spill slots            │
│  Callee-saved registers │
├─────────────────────────┤ ← ESP (current stack pointer)
└─────────────────────────┘
Low addresses
```

**Key Properties:**

- The stack must be 16-byte aligned at function entry (per modern GCC/ABI convention for SSE compatibility, though i686 cdecl historically required only 4-byte alignment).
- There is **no red zone** on i686 — all local storage requires explicit stack pointer adjustment.
- The frame pointer (EBP) is standard in debug builds (`-g`). In optimized builds, it may be omitted to free EBP as a general-purpose register.

### Floating-Point Arithmetic

The i686 backend uses the **x87 FPU** for all floating-point operations:

- Arithmetic operations (`fadd`, `fsub`, `fmul`, `fdiv`) operate on the x87 stack registers ST(0)–ST(7).
- Function results are returned in ST(0).
- The x87 FPU operates in 80-bit extended precision internally, which can cause subtle precision differences compared to 64-bit SSE-based computation on x86-64.
- BCC does not use SSE instructions on i686 for baseline compatibility.

---

## AAPCS64 ABI (AArch64)

The Arm Architecture Procedure Call Standard for 64-bit (AAPCS64) defines the calling convention for AArch64 Linux systems. It features a large general-purpose register file and a separate SIMD/FP register file, enabling efficient register-based parameter passing.

### Data Model

The AArch64 target uses the **LP64** data model:

| C Type | Size (bytes) | Alignment (bytes) |
|:-------|:-------------|:-------------------|
| `_Bool` | 1 | 1 |
| `char` | 1 | 1 |
| `short` | 2 | 2 |
| `int` | 4 | 4 |
| `long` | 8 | 8 |
| `long long` | 8 | 8 |
| `float` | 4 | 4 |
| `double` | 8 | 8 |
| `long double` | 16 | 16 |
| `_Complex float` | 8 | 4 |
| `_Complex double` | 16 | 8 |
| pointer | 8 | 8 |
| `size_t` | 8 | 8 |
| `ptrdiff_t` | 8 | 8 |

> **Note:** `long double` on AArch64 is 128-bit IEEE 754 quadruple precision, unlike x86-64/i686 which use 80-bit extended precision. BCC's software long-double arithmetic (`src/common/long_double.rs`) must handle this format for AArch64 targets.

### Register Definitions

Defined in `src/backend/aarch64/registers.rs`.

**31 General-Purpose Registers:**

| 64-bit | 32-bit | ABI Role | Saved By |
|:-------|:-------|:---------|:---------|
| X0 | W0 | 1st argument / return value | Caller |
| X1 | W1 | 2nd argument / 2nd return value | Caller |
| X2 | W2 | 3rd argument | Caller |
| X3 | W3 | 4th argument | Caller |
| X4 | W4 | 5th argument | Caller |
| X5 | W5 | 6th argument | Caller |
| X6 | W6 | 7th argument | Caller |
| X7 | W7 | 8th argument | Caller |
| X8 | W8 | Indirect result location register | Caller |
| X9–X15 | W9–W15 | Temporary / scratch | Caller |
| X16 | W16 | Intra-procedure-call scratch (IP0) | Caller |
| X17 | W17 | Intra-procedure-call scratch (IP1) | Caller |
| X18 | W18 | Platform register (reserved on Linux) | Caller |
| X19 | W19 | Callee-saved | Callee |
| X20 | W20 | Callee-saved | Callee |
| X21 | W21 | Callee-saved | Callee |
| X22 | W22 | Callee-saved | Callee |
| X23 | W23 | Callee-saved | Callee |
| X24 | W24 | Callee-saved | Callee |
| X25 | W25 | Callee-saved | Callee |
| X26 | W26 | Callee-saved | Callee |
| X27 | W27 | Callee-saved | Callee |
| X28 | W28 | Callee-saved | Callee |
| X29 | W29 | Frame Pointer (FP) | Callee |
| X30 | W30 | Link Register (LR) | Callee |

**Special Registers:**

| Register | Purpose |
|:---------|:--------|
| SP | Stack Pointer (not a GPR, separate register) |
| XZR / WZR | Zero register (reads as zero, writes discarded) |
| NZCV | Condition flags (Negative, Zero, Carry, oVerflow) |
| PC | Program Counter (not directly accessible) |

**32 SIMD/FP Registers:**

Each SIMD/FP register can be accessed at multiple widths:

| 128-bit | 64-bit | 32-bit | 16-bit | 8-bit | ABI Role |
|:--------|:-------|:-------|:-------|:------|:---------|
| V0 | D0 | S0 | H0 | B0 | 1st FP arg / FP return value, caller-saved |
| V1 | D1 | S1 | H1 | B1 | 2nd FP arg, caller-saved |
| V2 | D2 | S2 | H2 | B2 | 3rd FP arg, caller-saved |
| V3 | D3 | S3 | H3 | B3 | 4th FP arg, caller-saved |
| V4 | D4 | S4 | H4 | B4 | 5th FP arg, caller-saved |
| V5 | D5 | S5 | H5 | B5 | 6th FP arg, caller-saved |
| V6 | D6 | S6 | H6 | B6 | 7th FP arg, caller-saved |
| V7 | D7 | S7 | H7 | B7 | 8th FP arg, caller-saved |
| V8–V15 | D8–D15 | S8–S15 | H8–H15 | B8–B15 | Callee-saved (lower 64 bits only) |
| V16–V31 | D16–D31 | S16–S31 | H16–H31 | B16–B31 | Caller-saved (scratch) |

**Callee-Saved Registers:** X19–X28, X29 (FP), X30 (LR), D8–D15 (lower 64 bits of V8–V15 only)

**Caller-Saved Registers:** X0–X18, D0–D7, D16–D31

### Integer Parameter Passing

Integer and pointer arguments are passed in registers, in the following order:

| Argument Position | Register |
|:------------------|:---------|
| 1st | X0 |
| 2nd | X1 |
| 3rd | X2 |
| 4th | X3 |
| 5th | X4 |
| 6th | X5 |
| 7th | X6 |
| 8th | X7 |
| 9th and beyond | Stack |

- Arguments smaller than 64 bits are zero-extended or sign-extended to 64 bits within the register.
- 128-bit integer arguments (e.g., `__int128`) are passed in a pair of consecutive even-aligned registers (X0:X1, X2:X3, etc.).

### Floating-Point Parameter Passing

Floating-point arguments are passed in SIMD/FP registers:

| Argument Position | Register |
|:------------------|:---------|
| 1st | V0 (as S0 for `float`, D0 for `double`) |
| 2nd | V1 |
| 3rd | V2 |
| 4th | V3 |
| 5th | V4 |
| 6th | V5 |
| 7th | V6 |
| 8th | V7 |
| 9th and beyond | Stack |

Integer and floating-point arguments consume registers from their respective register files independently, exactly as on x86-64.

### Return Values

| Return Type | Register(s) |
|:------------|:------------|
| Integer/pointer (≤ 64-bit) | X0 |
| 128-bit integer | X0 (low 64 bits), X1 (high 64 bits) |
| `float` | S0 |
| `double` | D0 |
| `long double` (128-bit quad) | Q0 |
| Small struct (≤ 16 bytes) | X0 and/or X1, or V0 and/or V1, per classification |
| Large struct (> 16 bytes) | Hidden pointer in X8 (indirect result register) |

> **Note:** Unlike x86-64 which uses the first integer argument register (RDI) for the hidden struct return pointer, AArch64 uses the dedicated **X8** register (indirect result location register). This means X8 is not available for normal parameter passing when returning a large struct.

### HFA/HVA Handling

A **Homogeneous Floating-point Aggregate (HFA)** is a struct or array containing 1 to 4 members, all of the same floating-point type. An **HVA (Homogeneous Vector Aggregate)** follows the same rules for SIMD vector types.

**HFA Classification Rules:**

- All members must be the same base floating-point type (`float`, `double`, or `long double`).
- The struct must contain 1 to 4 such members (no more).
- Nested structs are flattened for classification purposes.

**HFA Passing:**

- HFAs are passed in consecutive SIMD/FP registers (V0–V3 for up to 4 members).
- If insufficient SIMD/FP registers remain, the entire HFA falls back to the stack.
- HFA members are never split between registers and the stack.

**Example:**

```c
struct Point { float x, y; };           // HFA: 2 floats → S0, S1
struct Color { float r, g, b, a; };     // HFA: 4 floats → S0, S1, S2, S3
struct Mixed { float x; int y; };       // NOT an HFA (mixed types) → X0
```

### Stack Layout

```
High addresses
┌─────────────────────────┐
│  Caller's frame         │
├─────────────────────────┤
│  Stack-passed arguments │
├─────────────────────────┤ ← SP at function entry (16-byte aligned)
│  Saved FP (X29) and LR (X30) │
├─────────────────────────┤ ← X29 (Frame Pointer)
│  Callee-saved registers │
│  Local variables        │
│  Spill slots            │
├─────────────────────────┤
│  Outgoing arguments     │
├─────────────────────────┤ ← SP (current, 16-byte aligned)
└─────────────────────────┘
Low addresses
```

**Key Properties:**

- The stack pointer SP must be **16-byte aligned at all times** (not just at call boundaries). This is enforced by the AArch64 hardware — misaligned SP access causes a fault.
- There is **no red zone** on AArch64. All local storage requires explicit SP adjustment.
- Frame Pointer (X29) and Link Register (X30) are saved as a pair at the top of the callee's frame. X29 points to the saved FP/LR pair, forming a frame chain for debugger unwinding.
- All AArch64 instructions are fixed-width 32-bit, which simplifies instruction encoding and decoding but requires multi-instruction sequences for large immediates (e.g., `MOVZ` + `MOVK`, or `ADRP` + `ADD` for PC-relative addressing).

---

## RISC-V LP64D ABI (RISC-V 64)

The RISC-V LP64D ABI is the standard calling convention for 64-bit RISC-V Linux systems with hardware double-precision floating-point. It is the target ABI for BCC's Linux kernel 6.9 build and boot validation (Checkpoint 6).

### Data Model

The RISC-V 64 target uses the **LP64** data model:

| C Type | Size (bytes) | Alignment (bytes) |
|:-------|:-------------|:-------------------|
| `_Bool` | 1 | 1 |
| `char` | 1 | 1 |
| `short` | 2 | 2 |
| `int` | 4 | 4 |
| `long` | 8 | 8 |
| `long long` | 8 | 8 |
| `float` | 4 | 4 |
| `double` | 8 | 8 |
| `long double` | 16 | 16 |
| `_Complex float` | 8 | 4 |
| `_Complex double` | 16 | 8 |
| pointer | 8 | 8 |
| `size_t` | 8 | 8 |
| `ptrdiff_t` | 8 | 8 |

> **Note:** `long double` on RISC-V 64 is 128-bit IEEE 754 quadruple precision, matching AArch64.

### ISA

BCC targets the **RV64IMAFDC** ISA profile:

| Extension | Full Name | Description |
|:----------|:----------|:------------|
| I | Integer | Base 64-bit integer instruction set |
| M | Multiply | Integer multiplication and division |
| A | Atomic | Atomic memory operations (LR/SC, AMO) |
| F | Float | Single-precision floating-point |
| D | Double | Double-precision floating-point |
| C | Compressed | 16-bit compressed instruction encoding |

The "D" extension is the key enabler for the LP64D ABI variant, which passes floating-point arguments in hardware FP registers.

### Register Definitions

Defined in `src/backend/riscv64/registers.rs`.

**32 Integer Registers:**

| Register | ABI Name | Purpose | Saved By |
|:---------|:---------|:--------|:---------|
| x0 | zero | Hardwired zero (reads 0, writes ignored) | — |
| x1 | ra | Return address | Callee |
| x2 | sp | Stack pointer | Callee |
| x3 | gp | Global pointer | — (not allocatable) |
| x4 | tp | Thread pointer | — (not allocatable) |
| x5 | t0 | Temporary | Caller |
| x6 | t1 | Temporary | Caller |
| x7 | t2 | Temporary | Caller |
| x8 | s0 / fp | Saved register / Frame pointer | Callee |
| x9 | s1 | Saved register | Callee |
| x10 | a0 | 1st argument / return value | Caller |
| x11 | a1 | 2nd argument / 2nd return value | Caller |
| x12 | a2 | 3rd argument | Caller |
| x13 | a3 | 4th argument | Caller |
| x14 | a4 | 5th argument | Caller |
| x15 | a5 | 6th argument | Caller |
| x16 | a6 | 7th argument | Caller |
| x17 | a7 | 8th argument | Caller |
| x18 | s2 | Saved register | Callee |
| x19 | s3 | Saved register | Callee |
| x20 | s4 | Saved register | Callee |
| x21 | s5 | Saved register | Callee |
| x22 | s6 | Saved register | Callee |
| x23 | s7 | Saved register | Callee |
| x24 | s8 | Saved register | Callee |
| x25 | s9 | Saved register | Callee |
| x26 | s10 | Saved register | Callee |
| x27 | s11 | Saved register | Callee |
| x28 | t3 | Temporary | Caller |
| x29 | t4 | Temporary | Caller |
| x30 | t5 | Temporary | Caller |
| x31 | t6 | Temporary | Caller |

**32 Floating-Point Registers:**

| Register | ABI Name | Purpose | Saved By |
|:---------|:---------|:--------|:---------|
| f0 | ft0 | FP temporary | Caller |
| f1 | ft1 | FP temporary | Caller |
| f2 | ft2 | FP temporary | Caller |
| f3 | ft3 | FP temporary | Caller |
| f4 | ft4 | FP temporary | Caller |
| f5 | ft5 | FP temporary | Caller |
| f6 | ft6 | FP temporary | Caller |
| f7 | ft7 | FP temporary | Caller |
| f8 | fs0 | FP saved register | Callee |
| f9 | fs1 | FP saved register | Callee |
| f10 | fa0 | 1st FP argument / FP return value | Caller |
| f11 | fa1 | 2nd FP argument / 2nd FP return value | Caller |
| f12 | fa2 | 3rd FP argument | Caller |
| f13 | fa3 | 4th FP argument | Caller |
| f14 | fa4 | 5th FP argument | Caller |
| f15 | fa5 | 6th FP argument | Caller |
| f16 | fa6 | 7th FP argument | Caller |
| f17 | fa7 | 8th FP argument | Caller |
| f18 | fs2 | FP saved register | Callee |
| f19 | fs3 | FP saved register | Callee |
| f20 | fs4 | FP saved register | Callee |
| f21 | fs5 | FP saved register | Callee |
| f22 | fs6 | FP saved register | Callee |
| f23 | fs7 | FP saved register | Callee |
| f24 | fs8 | FP saved register | Callee |
| f25 | fs9 | FP saved register | Callee |
| f26 | fs10 | FP saved register | Callee |
| f27 | fs11 | FP saved register | Callee |
| f28 | ft8 | FP temporary | Caller |
| f29 | ft9 | FP temporary | Caller |
| f30 | ft10 | FP temporary | Caller |
| f31 | ft11 | FP temporary | Caller |

**Callee-Saved Registers:** ra (x1), sp (x2), s0–s11 (x8–x9, x18–x27), fs0–fs11 (f8–f9, f18–f27)

**Caller-Saved Registers:** t0–t6 (x5–x7, x28–x31), a0–a7 (x10–x17), ft0–ft11 (f0–f7, f28–f31), fa0–fa7 (f10–f17)

**Non-Allocatable Registers:** zero (x0), gp (x3), tp (x4)

### Integer Parameter Passing

Integer and pointer arguments are passed in the `a` registers:

| Argument Position | Register |
|:------------------|:---------|
| 1st | a0 (x10) |
| 2nd | a1 (x11) |
| 3rd | a2 (x12) |
| 4th | a3 (x13) |
| 5th | a4 (x14) |
| 6th | a5 (x15) |
| 7th | a6 (x16) |
| 8th | a7 (x17) |
| 9th and beyond | Stack |

- Arguments smaller than 64 bits (XLEN) are sign-extended or zero-extended to fill the register.
- 128-bit arguments are passed in a pair of consecutive even-aligned `a` registers (a0:a1, a2:a3, etc.). If the first register of the pair is odd-numbered, the preceding register is skipped.
- Structs up to 2×XLEN (16 bytes) may be passed in register pairs. Larger structs are passed by reference (pointer in an `a` register).

### Floating-Point Parameter Passing (LP64D)

With the LP64D ABI, floating-point arguments are passed in the `fa` registers:

| Argument Position | Register |
|:------------------|:---------|
| 1st FP | fa0 (f10) |
| 2nd FP | fa1 (f11) |
| 3rd FP | fa2 (f12) |
| 4th FP | fa3 (f13) |
| 5th FP | fa4 (f14) |
| 6th FP | fa5 (f15) |
| 7th FP | fa6 (f16) |
| 8th FP | fa7 (f17) |
| 9th and beyond | Stack |

**Mixed Integer/FP Passing:**

The RISC-V ABI uses **both** integer and FP register files simultaneously. Each file is consumed independently:

```c
// Example: void f(int a, double b, int c, double d)
// a → a0 (x10)     [integer register file]
// b → fa0 (f10)    [FP register file]
// c → a1 (x11)     [integer register file]
// d → fa1 (f11)    [FP register file]
```

**Struct Flattening (LP64D specific):**

Small structs containing one integer and one float field (or two float fields) may be passed in a combination of integer and FP registers:

- A struct with one `int` and one `float` field: `int` → integer register, `float` → FP register.
- A struct with two `float` fields: first field → FP register, second field → FP register.
- A struct with one `double` field: passed in a single FP register.

If FP registers are exhausted, FP arguments fall back to integer registers. If integer registers are also exhausted, arguments spill to the stack.

### Return Values

| Return Type | Register(s) |
|:------------|:------------|
| Integer/pointer (≤ 64-bit) | a0 (x10) |
| 128-bit integer | a0:a1 (x10:x11) |
| `float` | fa0 (f10) |
| `double` | fa0 (f10) |
| Small struct (≤ 2×XLEN) | a0 and/or a1, or fa0 and/or fa1, per flattening rules |
| Large struct (> 2×XLEN) | Hidden pointer in a0 (caller-allocated) |

### Stack Layout

```
High addresses
┌─────────────────────────┐
│  Caller's frame         │
├─────────────────────────┤
│  Stack-passed arguments │
├─────────────────────────┤ ← SP at function entry (16-byte aligned)
│  Saved RA (x1)          │
│  Saved FP/S0 (x8)       │
├─────────────────────────┤ ← S0/FP (Frame Pointer)
│  Callee-saved registers │
│  Local variables        │
│  Spill slots            │
├─────────────────────────┤
│  Outgoing arguments     │
├─────────────────────────┤ ← SP (current, 16-byte aligned)
└─────────────────────────┘
Low addresses
```

**Key Properties:**

- The stack pointer (sp, x2) must be **16-byte aligned** at all times.
- There is **no red zone** on RISC-V. All local storage requires explicit SP adjustment.
- The **global pointer** (gp, x3) is used for GP-relative addressing of global variables within a ±2 KiB range (`lui`/`addi` can be replaced with a single `addi` relative to gp). The linker resolves GP-relative relocations.
- The **thread pointer** (tp, x4) is reserved for Thread-Local Storage (TLS) access. It is set by the runtime and must not be modified by compiled code.
- Large immediates are constructed using `LUI` (Load Upper Immediate) + `ADDI` or `AUIPC` (Add Upper Immediate to PC) + `ADDI` for PC-relative addressing.
- RISC-V supports **linker relaxation**: the linker may shorten multi-instruction sequences (e.g., `AUIPC` + `JALR` → `JAL`) when the target is within range. BCC's built-in linker (`src/backend/riscv64/linker/`) supports this optimization.

---

## Common ABI Concerns

This section documents cross-cutting ABI topics that apply to multiple or all target architectures.

### Struct Passing Summary

The following table summarizes struct passing rules across all four architectures:

| Architecture | Small Struct (in registers) | Large Struct (by reference) | Classification |
|:-------------|:----------------------------|:----------------------------|:---------------|
| x86-64 | ≤ 16 bytes (if classifiable as INTEGER/SSE) | > 16 bytes or MEMORY class | Per-eightword recursive classification |
| i686 | Never (always by reference) | All structs | Hidden pointer as first stack argument |
| AArch64 | ≤ 16 bytes | > 16 bytes | Field inspection, HFA detection |
| RISC-V 64 | ≤ 16 bytes (2×XLEN) | > 16 bytes | Integer/FP flattening rules |

### Alignment Requirements

| Architecture | Stack Alignment | Struct Member Alignment | `max_align_t` |
|:-------------|:----------------|:------------------------|:---------------|
| x86-64 | 16 bytes | Natural alignment | 16 bytes |
| i686 | 16 bytes | Natural alignment (max 4 for most) | 16 bytes |
| AArch64 | 16 bytes (always) | Natural alignment | 16 bytes |
| RISC-V 64 | 16 bytes | Natural alignment | 16 bytes |

The `__attribute__((aligned(N)))` attribute can increase alignment beyond the natural value. The `__attribute__((packed))` attribute can reduce struct member alignment to 1 byte, suppressing padding.

### Endianness

All four BCC target architectures operate in **little-endian** byte order:

| Architecture | Endianness |
|:-------------|:-----------|
| x86-64 | Little-endian |
| i686 | Little-endian |
| AArch64 | Little-endian (BCC configuration) |
| RISC-V 64 | Little-endian |

> **Note:** AArch64 and RISC-V 64 architectures support big-endian configurations, but BCC targets exclusively little-endian Linux systems.

### `_Atomic` Types

The `_Atomic` type qualifier is supported at the storage and representation level:

- An `_Atomic int` has the same size and alignment as `int` (with possible increased alignment for lock-free atomicity — e.g., `_Atomic long long` may be 8-byte aligned on i686 even though `long long` is normally 4-byte aligned).
- Actual atomic load/store/exchange/compare-and-swap operations may delegate to `libatomic` at link time for types that exceed the hardware's native atomic width.
- The compiler emits appropriate memory ordering fences (`memory_order_seq_cst`, `memory_order_acquire`, `memory_order_release`, etc.) as specified by the C11 `<stdatomic.h>` memory model.

### Variadic Functions

Architecture-specific `va_list` representations:

| Architecture | `va_list` Type | Implementation |
|:-------------|:---------------|:---------------|
| x86-64 | Struct (4 fields) | `gp_offset`, `fp_offset`, `overflow_arg_area`, `reg_save_area` |
| i686 | `char *` | Simple pointer to the next stack argument |
| AArch64 | Struct (5 fields) | `__stack`, `__gr_top`, `__vr_top`, `__gr_offs`, `__vr_offs` |
| RISC-V 64 | `char *` | Simple pointer to the next stack argument (like i686) |

**x86-64 `va_list` Detail:**

```c
typedef struct {
    unsigned int gp_offset;       // Offset to next integer register save slot
    unsigned int fp_offset;       // Offset to next FP register save slot
    void *overflow_arg_area;      // Pointer to next stack-passed argument
    void *reg_save_area;          // Pointer to register save area
} va_list[1];
```

**AArch64 `va_list` Detail:**

```c
typedef struct {
    void *__stack;     // Pointer to next stack-passed argument
    void *__gr_top;    // End of GPR save area
    void *__vr_top;    // End of FP/SIMD save area
    int __gr_offs;     // Offset from __gr_top to next GPR argument
    int __vr_offs;     // Offset from __vr_top to next FP/SIMD argument
} va_list;
```

### `long double` Representation

The representation of `long double` varies significantly across architectures:

| Architecture | Format | Size (bytes) | Storage (bytes) | Alignment (bytes) |
|:-------------|:-------|:-------------|:----------------|:-------------------|
| x86-64 | 80-bit x87 extended | 10 | 16 (6 padding) | 16 |
| i686 | 80-bit x87 extended | 10 | 12 (2 padding) | 4 |
| AArch64 | 128-bit IEEE 754 quad | 16 | 16 | 16 |
| RISC-V 64 | 128-bit IEEE 754 quad | 16 | 16 | 16 |

BCC's software long-double arithmetic (`src/common/long_double.rs`) must handle both the 80-bit extended precision format (for x86-64 and i686) and the 128-bit IEEE 754 quadruple precision format (for AArch64 and RISC-V 64), since the zero-dependency mandate forbids use of external math libraries.

### Bitfield Layout

Bitfield layout rules differ subtly across ABIs:

- **All architectures:** Bitfields are allocated from the least significant bit within their storage unit.
- **x86-64 / i686:** Bitfields do not cross their declared type's alignment boundary. A bitfield of type `int` that would cross a 4-byte boundary starts a new storage unit.
- **AArch64 / RISC-V 64:** Similar rules, with the storage unit determined by the bitfield's declared type.
- **`__attribute__((packed))` on bitfield containers:** Eliminates padding between storage units, allowing bitfields to pack tightly.

---

## Implementation Mapping

The following table maps each ABI component to its implementing source file in the BCC codebase:

| ABI Component | Source File | Description |
|:--------------|:------------|:------------|
| System V AMD64 calling convention | `src/backend/x86_64/abi.rs` | Parameter classification, register assignment, struct passing |
| x86-64 register definitions | `src/backend/x86_64/registers.rs` | GPR and SSE register enums, callee/caller-saved sets |
| x86-64 instruction selection | `src/backend/x86_64/codegen.rs` | Machine code generation using ABI constraints |
| x86-64 security mitigations | `src/backend/x86_64/security.rs` | Retpoline, CET/IBT, stack probe (ABI-adjacent) |
| cdecl / System V i386 calling convention | `src/backend/i686/abi.rs` | Stack-based parameter passing, struct return pointer |
| i686 register definitions | `src/backend/i686/registers.rs` | GPR and x87 FPU register enums |
| i686 instruction selection | `src/backend/i686/codegen.rs` | 32-bit machine code generation |
| AAPCS64 calling convention | `src/backend/aarch64/abi.rs` | Register-based passing, HFA/HVA handling |
| AArch64 register definitions | `src/backend/aarch64/registers.rs` | GPR and SIMD/FP register enums |
| AArch64 instruction selection | `src/backend/aarch64/codegen.rs` | A64 format instruction generation |
| RISC-V LP64D calling convention | `src/backend/riscv64/abi.rs` | Integer/FP register passing, struct flattening |
| RISC-V 64 register definitions | `src/backend/riscv64/registers.rs` | Integer and FP register enums with ABI names |
| RISC-V 64 instruction selection | `src/backend/riscv64/codegen.rs` | RV64IMAFDC instruction generation |
| C type definitions | `src/common/types.rs` | `CType` enum, `sizeof`/`alignof` per target |
| Type construction API | `src/common/type_builder.rs` | Builder pattern for struct layout, alignment, flexible arrays |
| Target definitions | `src/common/target.rs` | `Target` enum, pointer widths, endianness, data models |
| IR type system | `src/ir/types.rs` | Bridge between C types and machine types |
| Register allocator | `src/backend/register_allocator.rs` | Register assignment respecting ABI callee/caller-saved sets |
| Code generation driver | `src/backend/generation.rs` | Architecture dispatch, ABI selection based on `--target` flag |
| ArchCodegen trait | `src/backend/traits.rs` | Abstract interface for architecture-specific ABI operations |

### ABI Data Flow

The following diagram illustrates how ABI information flows through the BCC compilation pipeline:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ src/common/       │     │ src/frontend/     │     │ src/ir/           │
│   types.rs        │────▶│   sema/           │────▶│   lowering/       │
│   type_builder.rs │     │   type_checker.rs │     │   expr_lowering.rs│
│   target.rs       │     │                  │     │   decl_lowering.rs│
│                  │     │ (sizeof, alignof, │     │ (alloca sizes,   │
│ (C types, data   │     │  struct layout)   │     │  call lowering)  │
│  model constants)│     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Output: ELF      │     │ src/backend/      │     │ src/backend/      │
│   (.o / exec /   │◀────│   */codegen.rs    │◀────│   */abi.rs        │
│    .so)          │     │   */assembler/    │     │                  │
│                  │     │                  │     │ (param classify, │
│ (correct calling │     │ (instruction     │     │  register assign,│
│  convention in   │     │  selection per   │     │  struct passing) │
│  machine code)   │     │  ABI rules)      │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **Type Definitions** (`src/common/types.rs`, `src/common/target.rs`): C types and target-specific data model constants are defined.
2. **Semantic Analysis** (`src/frontend/sema/`): Type sizes, alignments, and struct layouts are computed using the target's ABI rules.
3. **IR Lowering** (`src/ir/lowering/`): Function calls are lowered using ABI-specific parameter passing rules. Alloca sizes reflect target-specific type sizes.
4. **ABI Classification** (`src/backend/*/abi.rs`): Each architecture's ABI module classifies function parameters and return values, assigning them to registers or stack locations.
5. **Code Generation** (`src/backend/*/codegen.rs`): Machine instructions are emitted according to the ABI's register usage and calling convention rules.
6. **Output** (`src/backend/elf_writer_common.rs`): The final ELF binary contains machine code that correctly implements the target ABI, producing interoperable executables and shared objects.
