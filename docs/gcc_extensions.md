# BCC GCC Extension Manifest

## Overview

This document is the **living manifest** for all GCC extensions supported by the BCC compiler. It serves as the central tracking document for GCC compatibility coverage, recording every supported attribute, language extension, builtin function, and inline assembly feature.

**Purpose:** The Linux kernel and many real-world C codebases depend heavily on GCC-specific extensions beyond the C11 standard. BCC must support a comprehensive set of these extensions to compile the Linux kernel 6.9 and other production C projects. This manifest tracks the implementation status of each extension and is updated during the kernel build phase (Checkpoint 6) when new extensions are discovered.

**Policy (Section 0.7.6):** All GCC extensions encountered during the kernel build **must** be handled gracefully — either fully implemented or diagnosed with a clear error message identifying the unsupported construct. The compiler **MUST NOT** silently miscompile unknown extensions. Any extension discovered during kernel compilation that is not already listed in this manifest is added as an amendment in the [Kernel Build Extension Discovery](#kernel-build-extension-discovery) section.

**Baseline Standard:** C11 (ISO/IEC 9899:2011) with the following GCC extension categories layered on top.

---

## 1. GCC Attributes

BCC supports 22 GCC-style attributes applied via `__attribute__((...))` syntax. Attributes may be attached to functions, variables, types, struct/union fields, and enum values depending on the specific attribute.

**Implementation Files:**

- **Parsing:** `src/frontend/parser/attributes.rs` — `__attribute__((...))` syntax recognition and AST representation
- **Validation:** `src/frontend/sema/attribute_handler.rs` — Semantic validation, constraint checking, and propagation to symbols and types

| # | Attribute | Syntax | Description | Applies To | Status |
|---|-----------|--------|-------------|------------|--------|
| 1 | `aligned` | `__attribute__((aligned(N)))` | Override the minimum alignment of a type or variable to `N` bytes. `N` must be a positive power of two. When applied to a struct/union type, sets the alignment of the entire aggregate. When used without an argument, aligns to the largest alignment supported by the target. | Types, variables, struct fields | Implemented |
| 2 | `packed` | `__attribute__((packed))` | Remove all padding from a struct or union, laying out fields at their natural size boundaries without alignment gaps. When applied to individual fields, packs only that field. Commonly combined with `aligned` for precise memory layout control. | Structs, unions, struct fields | Implemented |
| 3 | `section` | `__attribute__((section("name")))` | Place a function or variable into the named ELF section instead of the default `.text`, `.data`, or `.bss`. The section name must be a valid string literal. Used extensively in the Linux kernel for `__init`, `__exit`, and per-CPU data placement. | Functions, variables | Implemented |
| 4 | `used` | `__attribute__((used))` | Mark a function or variable as used even if no reference to it exists in the translation unit. Prevents the compiler and linker from dead-stripping the symbol. Ensures the symbol appears in the final ELF object. | Functions, variables | Implemented |
| 5 | `unused` | `__attribute__((unused))` | Suppress compiler warnings about an unused function, variable, parameter, label, or typedef. Does not prevent the entity from being emitted — it only silences the diagnostic. | Functions, variables, parameters, labels, typedefs | Implemented |
| 6 | `weak` | `__attribute__((weak))` | Declare a symbol with weak binding (`STB_WEAK` in ELF). A weakly-bound definition can be overridden by a strong definition from another translation unit at link time. If no strong definition exists, the weak definition is used. If no definition exists at all, the symbol resolves to zero/null. | Functions, variables | Implemented |
| 7 | `constructor` | `__attribute__((constructor))` or `__attribute__((constructor(priority)))` | Mark a function to be called automatically before `main()` during program initialization. The function is placed into the `.init_array` ELF section. An optional priority argument (integer) controls execution order — lower priorities execute first. | Functions | Implemented |
| 8 | `destructor` | `__attribute__((destructor))` or `__attribute__((destructor(priority)))` | Mark a function to be called automatically after `main()` returns (or `exit()` is called) during program finalization. The function is placed into the `.fini_array` ELF section. An optional priority argument controls execution order. | Functions | Implemented |
| 9 | `visibility` | `__attribute__((visibility("default"\|"hidden"\|"protected")))` | Control the ELF symbol visibility for shared library builds (`-fPIC -shared`). `"default"` exports the symbol (visible to dynamic linker), `"hidden"` restricts it to the defining shared object, and `"protected"` makes it visible but non-interposable. Maps to `STV_DEFAULT`, `STV_HIDDEN`, and `STV_PROTECTED` in the ELF symbol table. | Functions, variables | Implemented |
| 10 | `deprecated` | `__attribute__((deprecated))` or `__attribute__((deprecated("message")))` | Emit a warning diagnostic when the marked entity is referenced. An optional string argument provides a custom deprecation message included in the diagnostic output. | Functions, variables, types, struct fields, enum values | Implemented |
| 11 | `noreturn` | `__attribute__((noreturn))` | Indicate that a function never returns to its caller (e.g., `exit()`, `abort()`, infinite loops). Enables the compiler to optimize the call site by not generating code after the call. Calling a `noreturn` function and then falling through is undefined behavior. | Functions | Implemented |
| 12 | `noinline` | `__attribute__((noinline))` | Prevent the compiler from inlining this function at any call site, regardless of optimization level. The function is always emitted as a separate callable entity. | Functions | Implemented |
| 13 | `always_inline` | `__attribute__((always_inline))` | Force the compiler to inline this function at every call site. The function must also be declared `inline` or `static inline`. If inlining is not possible (e.g., recursive call, address taken), a diagnostic is emitted. | Functions | Implemented |
| 14 | `cold` | `__attribute__((cold))` | Hint that a function is unlikely to be called during normal execution (e.g., error handlers, panic paths). The compiler may place cold functions in a separate section and optimize branch predictions to favor the non-cold path. | Functions | Implemented |
| 15 | `hot` | `__attribute__((hot))` | Hint that a function is called frequently during normal execution. The compiler may place hot functions in a separate section and optimize for their execution speed. Opposite of `cold`. | Functions | Implemented |
| 16 | `format` | `__attribute__((format(archetype, string_idx, first_to_check)))` | Enable printf/scanf format string validation. `archetype` is `printf`, `scanf`, `strftime`, or `strfmon`. `string_idx` (1-based) identifies the format string parameter. `first_to_check` (1-based) identifies the first variadic argument to validate. Set `first_to_check` to 0 for `vprintf`-style functions. | Functions | Implemented |
| 17 | `format_arg` | `__attribute__((format_arg(string_idx)))` | Indicate that a function parameter at position `string_idx` (1-based) is a format string that is returned (possibly translated) and should be treated as a format string by the caller. Used for `gettext`-style wrapper functions. | Functions | Implemented |
| 18 | `malloc` | `__attribute__((malloc))` | Indicate that the function returns a pointer that does not alias any other pointer accessible to the caller at the point of the call. This enables the compiler to assume the returned pointer is unique, improving alias analysis. | Functions | Implemented |
| 19 | `pure` | `__attribute__((pure))` | Indicate that the function has no observable side effects except its return value, though it may read global memory and its arguments. A `pure` function called with the same arguments and global state will return the same result. The compiler may eliminate redundant calls or reorder them. | Functions | Implemented |
| 20 | `const` | `__attribute__((const))` | Stricter than `pure`: the function has no side effects and does not read any global memory — its return value depends only on its argument values. The compiler may aggressively cache and reorder calls to `const` functions. | Functions | Implemented |
| 21 | `warn_unused_result` | `__attribute__((warn_unused_result))` | Emit a warning if the return value of this function is discarded by the caller. Used for functions where ignoring the return value is likely a bug (e.g., error-returning system calls). | Functions | Implemented |
| 22 | `fallthrough` | `__attribute__((fallthrough))` | Explicitly mark intentional fall-through between `switch` `case` labels to suppress the `-Wimplicit-fallthrough` warning. Must appear as a standalone statement (i.e., `__attribute__((fallthrough));`) immediately before a `case` or `default` label. | Statements (switch cases) | Implemented |

---

## 2. GCC Language Extensions

BCC supports the following GCC language extensions beyond the C11 standard. These are commonly used in the Linux kernel and other systems-level C codebases.

**Implementation Files:**

- **Parsing:** `src/frontend/parser/gcc_extensions.rs` — Extension-specific grammar rules
- **Expressions:** `src/frontend/parser/expressions.rs` — Statement expressions, conditional omission
- **Statements:** `src/frontend/parser/statements.rs` — Computed gotos, case ranges, local labels
- **Types:** `src/frontend/parser/types.rs` — `typeof`, `__extension__`

| # | Extension | Syntax Example | Description | Status |
|---|-----------|---------------|-------------|--------|
| 1 | Statement Expressions | `({ int x = 5; x + 1; })` | A compound statement enclosed in parentheses that evaluates to the value of the last expression in the block. This allows embedding complex multi-statement logic in expression context, commonly used in Linux kernel macros (`min()`, `max()`, `container_of()`). The type of the statement expression is the type of the final expression. Variables declared inside are local to the block. | Implemented |
| 2 | `typeof` / `__typeof__` | `typeof(expr) var;` or `__typeof__(type) var;` | A type specifier that infers the type of an expression or another type at compile time. Produces the type of its operand without evaluating it. Used extensively in kernel macros for type-safe generic operations. Both `typeof` and `__typeof__` spellings are recognized; `__typeof__` is the form usable under `__extension__`. | Implemented |
| 3 | Zero-Length Arrays | `struct s { int n; int arr[0]; };` | An array declared with zero length as the last member of a struct, serving as a flexible boundary for variable-length data appended after the struct. Semantically equivalent to C99 flexible array members (`int arr[];`) but uses the GCC legacy syntax. `sizeof(arr)` is 0 and does not contribute to the struct size. | Implemented |
| 4 | Designated Initializers (Extended) | `struct s x = { .b = 2, .a = 1 };` or `int arr[] = { [5] = 50, [2] = 20 };` | GCC extends C99 designated initializers with support for out-of-order field designation, nested designators (`.field.subfield = val`), array range designators (`[first ... last] = val`), and brace elision. Unspecified members are zero-initialized. Parsing and semantic validation in `src/frontend/sema/initializer.rs`. | Implemented |
| 5 | Computed Gotos | `void *labels[] = { &&L1, &&L2 };` `goto *labels[i];` | The unary `&&` operator applied to a label produces a `void *` value representing the label's address. The `goto *expr` statement performs an indirect jump to the address computed by the expression. Used in the Linux kernel for threaded interpreter dispatch (e.g., BPF). The parser records label address-of expressions and indirect goto targets for control-flow analysis. | Implemented |
| 6 | Case Ranges | `case 1 ... 5:` | A range of consecutive values in a `switch` `case` label, expressed as `case LOW ... HIGH:`, where `LOW` and `HIGH` are integer constant expressions and `LOW <= HIGH`. Equivalent to listing each individual case label from `LOW` to `HIGH` inclusive. The `...` is a GCC extension token separated from the operands by whitespace. | Implemented |
| 7 | Conditional Omission | `x ?: y` | A ternary expression where the middle operand is omitted. Equivalent to `x ? x : y` except that `x` is evaluated only once. If `x` is truthy, the result is `x`; otherwise the result is `y`. The type of the expression follows the usual arithmetic conversions between the types of `x` and `y`. | Implemented |
| 8 | `__extension__` | `__extension__ ({ ... })` | A prefix keyword that suppresses GCC extension warnings for the expression or declaration that follows it. Allows use of GCC extensions in headers that must compile cleanly with `-pedantic`. The keyword has no effect on code generation — it is purely a diagnostic suppression mechanism. | Implemented |
| 9 | Transparent Unions | `__attribute__((transparent_union))` on a union | A union type attributed with `transparent_union` can be passed to a function by value, and the caller may pass any of the union's member types directly without explicit union wrapping. The called function receives the value as if the appropriate union member were initialized. Used in glibc for backwards-compatible API design (`wait()` family). | Implemented |
| 10 | Local Labels | `__label__ L1, L2;` | Declare labels that are scoped to the enclosing block rather than the entire function. Local label declarations must appear at the beginning of a block (after the opening `{`). This allows statement-expression macros to define jump targets without conflicting with labels in the surrounding function. | Implemented |

---

## 3. GCC Builtins

BCC supports approximately 30 GCC builtin functions. These are recognized by name during semantic analysis and handled in one of two ways:

- **Compile-Time Builtins:** Evaluated entirely at compile time; they produce a constant result and are folded during semantic analysis or constant evaluation. Implemented in `src/frontend/sema/builtin_eval.rs`.
- **Runtime Builtins:** Lowered to IR instructions or architecture-specific code sequences during IR lowering. Implemented across `src/ir/lowering/` and architecture-specific codegen modules.

| # | Builtin | Signature | Description | Evaluation | Status |
|---|---------|-----------|-------------|------------|--------|
| 1 | `__builtin_expect` | `long __builtin_expect(long expr, long val)` | Provide a branch prediction hint to the compiler. Returns `expr` unchanged. The hint indicates that `expr` is expected to equal `val` in the common case. The compiler uses this to optimize branch layout (e.g., placing the unlikely path out-of-line). Common wrappers: `likely()` / `unlikely()` macros in the Linux kernel. | Compile-time (hint only; returns `expr`) | Implemented |
| 2 | `__builtin_unreachable` | `void __builtin_unreachable(void)` | Inform the compiler that this point in the code is unreachable. If execution reaches this point, behavior is undefined. Allows the compiler to optimize away branches that lead to unreachable code and to assume that preceding conditions must hold. | Compile-time (control flow marker) | Implemented |
| 3 | `__builtin_constant_p` | `int __builtin_constant_p(expr)` | Return 1 if the argument is a compile-time constant, 0 otherwise. Used in kernel macros to select between compile-time optimized and runtime generic code paths. Evaluation is performed during semantic analysis when the argument is a constant expression; otherwise returns 0. | Compile-time | Implemented |
| 4 | `__builtin_offsetof` | `size_t __builtin_offsetof(type, member)` | Return the byte offset of `member` within `type`. Equivalent to `offsetof()` from `<stddef.h>`. The result is an integer constant expression. Type and member are validated during semantic analysis. Handles nested member access (e.g., `__builtin_offsetof(struct s, field.subfield)`). | Compile-time | Implemented |
| 5 | `__builtin_types_compatible_p` | `int __builtin_types_compatible_p(type1, type2)` | Return 1 if `type1` and `type2` are compatible types (ignoring top-level qualifiers), 0 otherwise. This is a type-level operation — no values are evaluated. Used in kernel `__same_type()` macros and type-dispatch logic. Array decay and function-to-pointer conversions are not applied. | Compile-time | Implemented |
| 6 | `__builtin_choose_expr` | `__builtin_choose_expr(const_expr, expr1, expr2)` | If `const_expr` evaluates to nonzero (at compile time), the result is `expr1`; otherwise `expr2`. The unchosen expression is not type-checked or evaluated. This provides compile-time selection similar to C11 `_Generic` but based on integer constant expressions rather than type matching. | Compile-time | Implemented |
| 7 | `__builtin_clz` | `int __builtin_clz(unsigned int x)` | Count the number of leading zero bits in `x`, starting from the most significant bit. The result is undefined if `x` is 0. For 32-bit integers, `__builtin_clz(1) == 31`. | Runtime (architecture instruction: `lzcnt`/`bsr` on x86, `clz` on AArch64, custom on RISC-V) | Implemented |
| 8 | `__builtin_clzl` | `int __builtin_clzl(unsigned long x)` | Count leading zeros for `unsigned long`. Behaves identically to `__builtin_clz` but operates on `unsigned long` (32-bit on ILP32, 64-bit on LP64). | Runtime | Implemented |
| 9 | `__builtin_clzll` | `int __builtin_clzll(unsigned long long x)` | Count leading zeros for `unsigned long long` (always 64-bit). | Runtime | Implemented |
| 10 | `__builtin_ctz` | `int __builtin_ctz(unsigned int x)` | Count the number of trailing zero bits in `x`, starting from the least significant bit. The result is undefined if `x` is 0. For example, `__builtin_ctz(8) == 3`. | Runtime (architecture instruction: `tzcnt`/`bsf` on x86, `rbit`+`clz` on AArch64, custom on RISC-V) | Implemented |
| 11 | `__builtin_ctzl` | `int __builtin_ctzl(unsigned long x)` | Count trailing zeros for `unsigned long`. | Runtime | Implemented |
| 12 | `__builtin_ctzll` | `int __builtin_ctzll(unsigned long long x)` | Count trailing zeros for `unsigned long long` (always 64-bit). | Runtime | Implemented |
| 13 | `__builtin_popcount` | `int __builtin_popcount(unsigned int x)` | Return the number of set (1) bits in `x` (population count / Hamming weight). For example, `__builtin_popcount(0xFF) == 8`. | Runtime (architecture instruction: `popcnt` on x86 with SSE4.2, software fallback otherwise) | Implemented |
| 14 | `__builtin_popcountl` | `int __builtin_popcountl(unsigned long x)` | Population count for `unsigned long`. | Runtime | Implemented |
| 15 | `__builtin_popcountll` | `int __builtin_popcountll(unsigned long long x)` | Population count for `unsigned long long` (always 64-bit). | Runtime | Implemented |
| 16 | `__builtin_bswap16` | `uint16_t __builtin_bswap16(uint16_t x)` | Reverse the byte order of a 16-bit value. Converts between big-endian and little-endian representations. For example, `__builtin_bswap16(0x1234) == 0x3412`. | Runtime (architecture instruction: `ror` on x86, `rev16` on AArch64) | Implemented |
| 17 | `__builtin_bswap32` | `uint32_t __builtin_bswap32(uint32_t x)` | Reverse the byte order of a 32-bit value. For example, `__builtin_bswap32(0x12345678) == 0x78563412`. | Runtime (architecture instruction: `bswap` on x86, `rev` on AArch64) | Implemented |
| 18 | `__builtin_bswap64` | `uint64_t __builtin_bswap64(uint64_t x)` | Reverse the byte order of a 64-bit value. | Runtime (architecture instruction: `bswap` on x86-64, `rev` on AArch64) | Implemented |
| 19 | `__builtin_ffs` | `int __builtin_ffs(int x)` | Find the position of the first (least significant) set bit in `x`, returning a 1-based index. Returns 0 if `x` is 0. For example, `__builtin_ffs(0x80) == 8`. Equivalent to `ffs()` from `<strings.h>`. | Runtime | Implemented |
| 20 | `__builtin_ffsl` | `int __builtin_ffsl(long x)` | Find first set bit for `long`. | Runtime | Implemented |
| 21 | `__builtin_ffsll` | `int __builtin_ffsll(long long x)` | Find first set bit for `long long` (always 64-bit). | Runtime | Implemented |
| 22 | `__builtin_va_start` | `void __builtin_va_start(va_list ap, last_param)` | Initialize the `va_list` object `ap` for subsequent variadic argument retrieval. `last_param` is the name of the last fixed parameter before the `...` in the function signature. Must be called before any `__builtin_va_arg`. Architecture-specific implementation depends on the calling convention. | Runtime (ABI-specific register save area setup) | Implemented |
| 23 | `__builtin_va_end` | `void __builtin_va_end(va_list ap)` | Clean up the `va_list` object `ap` after variadic argument processing is complete. Each `va_start` must have a matching `va_end` before the function returns. | Runtime (cleanup) | Implemented |
| 24 | `__builtin_va_arg` | `type __builtin_va_arg(va_list ap, type)` | Retrieve the next variadic argument of the specified type from `va_list ap`. The type must match the promoted type of the argument as passed by the caller. Advances the `va_list` internal pointer. | Runtime (ABI-specific argument retrieval) | Implemented |
| 25 | `__builtin_va_copy` | `void __builtin_va_copy(va_list dest, va_list src)` | Copy the state of `src` to `dest`, creating an independent copy of the variadic argument traversal position. `dest` must be cleaned up with `va_end` independently. | Runtime (memory copy of va_list state) | Implemented |
| 26 | `__builtin_frame_address` | `void *__builtin_frame_address(unsigned int level)` | Return the frame pointer of the specified stack frame. Level 0 returns the frame address of the current function. Level 1 returns the frame address of the caller. Higher levels walk the call stack. Results for `level > 0` are not guaranteed to be valid in optimized code. | Runtime (frame pointer register read) | Implemented |
| 27 | `__builtin_return_address` | `void *__builtin_return_address(unsigned int level)` | Return the return address of the specified stack frame. Level 0 returns the return address of the current function (i.e., where it will return to). Level 1 returns the return address of the caller. Higher levels walk the stack. Results for `level > 0` may be invalid in optimized code. | Runtime (return address retrieval from stack/link register) | Implemented |
| 28 | `__builtin_trap` | `void __builtin_trap(void)` | Cause the program to abort abnormally. On most architectures, this emits an illegal instruction (`ud2` on x86, `brk` on AArch64, `ebreak` on RISC-V). The function never returns. | Runtime (illegal instruction emission) | Implemented |
| 29 | `__builtin_assume_aligned` | `void *__builtin_assume_aligned(const void *ptr, size_t align)` | Return `ptr` with the compile-time assumption that it is aligned to at least `align` bytes. The compiler may use this assumption to generate more efficient code (e.g., aligned loads/stores). If `ptr` is not actually aligned at runtime, behavior is undefined. | Compile-time (alignment hint; returns `ptr`) | Implemented |
| 30 | `__builtin_add_overflow` | `bool __builtin_add_overflow(type a, type b, type *result)` | Perform addition of `a + b` and store the result in `*result`. Returns `true` if the operation overflowed (the mathematical result cannot be represented in the result type), `false` otherwise. The result is always written regardless of overflow. Works with any integer type. | Runtime (architecture-specific overflow flag or widening arithmetic) | Implemented |
| 31 | `__builtin_sub_overflow` | `bool __builtin_sub_overflow(type a, type b, type *result)` | Perform subtraction of `a - b` and store the result in `*result`. Returns `true` on overflow, `false` otherwise. Semantics are identical to `__builtin_add_overflow` but for subtraction. | Runtime | Implemented |
| 32 | `__builtin_mul_overflow` | `bool __builtin_mul_overflow(type a, type b, type *result)` | Perform multiplication of `a * b` and store the result in `*result`. Returns `true` on overflow, `false` otherwise. May use widening multiplication on architectures that support it. | Runtime | Implemented |

---

## 4. Inline Assembly Support

BCC provides full inline assembly support using GCC's extended `asm` syntax with AT&T (gas) dialect. This is critical for Linux kernel compilation, which contains thousands of inline assembly statements for architecture-specific operations, memory barriers, atomic operations, and context switching.

**Implementation Files:**

- **Parsing:** `src/frontend/parser/inline_asm.rs` — `asm`/`__asm__` syntax recognition, operand and constraint parsing
- **IR Lowering:** `src/ir/lowering/asm_lowering.rs` — Template string processing, operand binding, clobber handling, `asm goto` target wiring

### 4.1 Basic Syntax

```c
asm volatile (
    "assembly template"
    : output_operands       /* optional */
    : input_operands        /* optional */
    : clobber_list          /* optional */
);
```

Both `asm` and `__asm__` spellings are recognized. The `__asm__` form is usable in strict C11 mode and under `__extension__`.

### 4.2 Operand Constraints

| Constraint | Class | Description |
|-----------|-------|-------------|
| `"r"` | Input | General-purpose register (compiler selects) |
| `"i"` | Input | Immediate integer constant |
| `"n"` | Input | Immediate integer constant with known value at compile time |
| `"m"` | Input/Output | Memory operand (address expression) |
| `"=r"` | Output | Write-only general-purpose register |
| `"=m"` | Output | Write-only memory operand |
| `"+r"` | Input+Output | Read-write general-purpose register |
| `"+m"` | Input+Output | Read-write memory operand |
| `"&r"` | Output (early-clobber) | Output register that is written before all inputs are consumed |
| `"0"`, `"1"`, ... | Input (matching) | Input must be in the same register as the numbered output operand |

Architecture-specific constraints are also recognized and delegated to the corresponding backend assembler module:

- **x86-64/i686:** `"a"` (EAX/RAX), `"b"` (EBX/RBX), `"c"` (ECX/RCX), `"d"` (EDX/RDX), `"S"` (ESI/RSI), `"D"` (EDI/RDI)
- **AArch64:** Register class constraints mapped to GPR (X/W) or FP/SIMD (V) register files
- **RISC-V 64:** Register class constraints mapped to integer (x) or floating-point (f) register files

### 4.3 Clobber Lists

| Clobber | Description |
|---------|-------------|
| `"memory"` | The assembly statement reads or writes memory not listed in the operands. Acts as a compiler memory barrier — the compiler will not reorder memory accesses across this statement. |
| `"cc"` | The assembly statement modifies the condition code / flags register. On x86, this means EFLAGS; on AArch64, this means NZCV; on RISC-V, this is typically implicit. |
| `"register_name"` | The assembly statement overwrites the named register (e.g., `"rax"`, `"x0"`, `"t0"`). The compiler will not place live values in the clobbered register across the asm statement. |

### 4.4 Named Operands

```c
asm ("add %[src], %[dst]"
    : [dst] "+r" (result)
    : [src] "r" (value)
);
```

Operands can be referenced by name (`%[name]`) instead of positional index (`%0`, `%1`). Named operands improve readability for complex inline assembly with many operands. The name is an identifier enclosed in square brackets preceding the constraint string.

### 4.5 `asm goto`

```c
asm goto (
    "cmpb $0, %0\n\t"
    "je %l[error]"
    : /* no outputs */
    : "m" (flag)
    : "cc"
    : error               /* jump labels */
);
/* fall-through path */
return 0;

error:
    return -1;
```

`asm goto` extends inline assembly with jump labels that the assembly template can branch to. The labels appear in a fourth colon-delimited section after the clobber list. Each label name must refer to a C label in the enclosing function. The `%l[name]` syntax references a jump label within the assembly template. The compiler must model all possible control flow paths — both fall-through and jumps to any listed label. `asm goto` statements cannot have output operands (GCC constraint).

### 4.6 `asm volatile`

The `volatile` qualifier on an `asm` statement prevents the compiler from optimizing it away, reordering it relative to other volatile operations, or moving it out of loops. All `asm` statements with side effects (including memory clobbers or output operands) should use `volatile`. Without `volatile`, the compiler may eliminate, duplicate, or move the asm statement if it determines the outputs are unused or the statement has no visible side effects.

### 4.7 Section Directives

```c
asm volatile (
    ".pushsection .data\n\t"
    ".ascii \"hello\"\n\t"
    ".popsection"
);
```

The `.pushsection` and `.popsection` assembler directives within inline assembly allow code or data to be emitted into arbitrary ELF sections from within an inline assembly block. The built-in assembler processes these directives during assembly, switching the output section and restoring it afterward. This is used extensively in the Linux kernel for exception tables, alternative instructions, and static key entries.

---

## 5. Kernel Build Extension Discovery

This section tracks GCC extensions discovered during the Linux kernel 6.9 compilation process (Checkpoint 6) that were not part of the initial manifest above. Each entry documents when the extension was encountered, how it was classified, and its resolution.

### 5.1 Discovery Protocol

When a kernel source file fails to compile, the failure is classified using the following priority order:

1. **Missing GCC Extension** — An unrecognized `__attribute__`, language construct, or syntax extension
2. **Missing Builtin** — An unrecognized `__builtin_*` function call
3. **Inline Assembly Constraint** — An unrecognized or unsupported constraint in an `asm` statement
4. **Preprocessor Issue** — A macro expansion, `#include`, or conditional compilation problem
5. **Codegen Bug** — Correct parsing and semantic analysis but incorrect code generation

### 5.2 Amendment Tracking Format

Each discovered extension is recorded with the following information:

| Field | Description |
|-------|-------------|
| **Extension** | Name or description of the GCC extension |
| **Classification** | Category from the discovery protocol (1–5 above) |
| **Kernel Source** | File path in the kernel tree where the extension was encountered |
| **Kernel Config** | Relevant `CONFIG_*` option that enables the code path (if applicable) |
| **Resolution** | How the extension was handled: `Implemented`, `Diagnosed (error)`, or `Workaround` |
| **Implementation** | BCC source file(s) modified to support the extension |
| **Date** | Date the extension was discovered and resolved |
| **Notes** | Additional context, edge cases, or related extensions |

### 5.3 Discovered Extensions Log

> This section is populated during the kernel build phase. Each entry below represents an extension that was not in the initial manifest and was discovered during Checkpoint 6 execution.

| # | Extension | Classification | Kernel Source | Resolution | Implementation | Date | Notes |
|---|-----------|---------------|---------------|------------|----------------|------|-------|
| — | *(No extensions discovered yet — kernel build phase has not started)* | — | — | — | — | — | — |

### 5.4 Regression Verification

After each extension is implemented during the kernel build phase:

1. The full internal test suite (Checkpoint 3) **must** be re-run to confirm no regressions
2. All previously passing kernel sub-gates must be re-verified
3. The newly compiling kernel file must produce correct object code
4. Any regression found must be resolved before proceeding to the next kernel sub-gate

**Regression Definition:** Any test that passed before the current change and fails after is a regression. Resolution is mandatory before proceeding.

---

## 6. Extension Policy Summary

### 6.1 Handling Unknown Extensions

BCC follows a strict policy for GCC extension encounters:

- **Known and Implemented:** Extension is fully supported — parsing, semantic validation, and code generation produce correct results.
- **Known and Diagnosed:** Extension is recognized but not implemented — the compiler emits a clear, actionable error message identifying the unsupported construct by name and suggesting alternatives where possible.
- **Unknown:** Extension is not recognized at all — the compiler emits a diagnostic error identifying the unrecognized syntax or identifier. The compiler **never** silently ignores or miscompiles an unknown extension.

### 6.2 Silent Miscompilation Prevention

The compiler **MUST NOT** silently miscompile unknown extensions. This is enforced through:

- The parser rejects unrecognized `__attribute__` names with a diagnostic rather than silently ignoring them
- The semantic analyzer validates all attribute arguments and emits errors for malformed or unsupported combinations
- The builtin evaluator rejects unrecognized `__builtin_*` identifiers rather than treating them as ordinary function calls
- The inline assembly constraint validator rejects unsupported constraints rather than generating incorrect register allocations

### 6.3 Extension Coverage Metrics

| Category | Count | Status |
|----------|-------|--------|
| GCC Attributes | 22 | All implemented |
| Language Extensions | 10 | All implemented |
| GCC Builtins | 32 | All implemented |
| Inline Assembly Features | 7 (constraints, clobbers, named operands, asm goto, asm volatile, section directives, matching constraints) | All implemented |
| Kernel-Discovered Extensions | 0 | Pending Checkpoint 6 |

---

## 7. Implementation File Reference

Quick reference mapping extension categories to their primary implementation files:

| Component | Parsing | Semantic Analysis | IR Lowering | Codegen |
|-----------|---------|-------------------|-------------|---------|
| Attributes | `src/frontend/parser/attributes.rs` | `src/frontend/sema/attribute_handler.rs` | — | Architecture backends |
| Language Extensions | `src/frontend/parser/gcc_extensions.rs` | `src/frontend/sema/type_checker.rs` | `src/ir/lowering/expr_lowering.rs`, `src/ir/lowering/stmt_lowering.rs` | — |
| Builtins (compile-time) | — | `src/frontend/sema/builtin_eval.rs` | — | — |
| Builtins (runtime) | — | `src/frontend/sema/builtin_eval.rs` | `src/ir/lowering/expr_lowering.rs` | Architecture backends |
| Inline Assembly | `src/frontend/parser/inline_asm.rs` | — | `src/ir/lowering/asm_lowering.rs` | Architecture assemblers |
| Designated Initializers | `src/frontend/parser/declarations.rs` | `src/frontend/sema/initializer.rs` | `src/ir/lowering/decl_lowering.rs` | — |
| Type Extensions (`typeof`) | `src/frontend/parser/types.rs` | `src/frontend/sema/type_checker.rs` | — | — |
| Computed Gotos | `src/frontend/parser/statements.rs` | `src/frontend/sema/scope.rs` | `src/ir/lowering/stmt_lowering.rs` | Architecture backends |

---

*This manifest is maintained as a living document. Last updated at initial project creation. Updates will be recorded during each kernel build iteration in the Kernel Build Extension Discovery section.*
