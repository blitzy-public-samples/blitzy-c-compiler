# GCC Extension Manifest

BCC (Blitzy's C Compiler) supports a comprehensive set of GCC attributes, language extensions,
builtins, and inline assembly features required to compile the Linux kernel source and other
large-scale C codebases. This document serves as the authoritative reference for every
GCC-compatible extension implemented in BCC.

**Living Document Notice:** This manifest also functions as a status tracker. Extensions
discovered during the Linux kernel 6.9 build phase (§5.4) that are not present in the
initial list below are recorded in the
[Kernel Build Extension Discovery Log](#kernel-build-extension-discovery-log) section as
amendments.

### Extension Handling Policy

The following policy governs GCC extension support in BCC (derived from AAP §0.7.6):

1. **All listed extensions MUST be implemented.** Every attribute, builtin, language extension,
   and inline assembly feature documented in this manifest is a hard requirement.
2. **Unknown extensions must be handled gracefully.** Any GCC extension encountered during the
   kernel build that is not in this initial manifest must either be implemented or diagnosed
   with a clear, actionable error message identifying the unsupported construct.
3. **Silent miscompilation is forbidden.** The compiler MUST NOT silently miscompile unknown
   extensions under any circumstances. An explicit diagnostic is always required.
4. **Kernel build failure classification protocol:** When a kernel source file fails to compile,
   the root cause is classified in this priority order:
   - Missing GCC extension → Missing builtin → Inline assembly constraint issue →
     Preprocessor issue → Code generation bug

### Implementation Map

| Pipeline Stage | Source Location | Responsibility |
|----------------|----------------|----------------|
| Attribute parsing | `src/frontend/parser/attributes.rs` | Syntactic recognition of `__attribute__((...))` |
| Attribute semantics | `src/frontend/sema/attribute_handler.rs` | Semantic validation and propagation |
| Extension parsing | `src/frontend/parser/gcc_extensions.rs` | GCC language extension syntax |
| Builtin evaluation | `src/frontend/sema/builtin_eval.rs` | Compile-time builtin resolution |
| Inline asm parsing | `src/frontend/parser/inline_asm.rs` | AT&T syntax asm statement parsing |
| Inline asm lowering | `src/ir/lowering/asm_lowering.rs` | Constraint binding and IR emission |
| Arch-specific encoding | `src/backend/*/assembler/` | Machine code emission for asm templates |

---

## GCC Attributes

BCC supports the `__attribute__((...))` syntax for attaching metadata and directives to
declarations. Attributes may be applied to functions, variables, types, struct/union fields,
parameters, labels, and statements depending on the specific attribute.

### Syntax

```c
/* Single attribute */
__attribute__((attribute_name))
__attribute__((attribute_name(arguments)))

/* Multiple attributes */
__attribute__((attr1, attr2(arg), attr3))

/* Positional usage */
int x __attribute__((aligned(16)));
void f(void) __attribute__((noreturn));
struct __attribute__((packed)) s { int a; char b; };
```

### Attribute Reference

| # | Attribute | Syntax | Description | Applied To | Status |
|---|-----------|--------|-------------|------------|--------|
| 1 | `aligned` | `__attribute__((aligned(N)))` | Specifies minimum alignment in bytes. `N` must be a positive power of two. When applied to a type, all objects of that type receive the specified alignment. When applied to a struct field, the field is aligned within the struct layout. If `N` is omitted, the target's maximum useful alignment is used. | Variables, types, struct fields | ✅ Implemented |
| 2 | `packed` | `__attribute__((packed))` | Removes inter-member padding from a struct or union, laying out members contiguously with alignment 1. When applied to an individual field, only that field loses its natural alignment padding. Commonly combined with `aligned` for precise layout control. | Struct/union types, struct fields | ✅ Implemented |
| 3 | `section` | `__attribute__((section("name")))` | Places the symbol into the named ELF output section instead of the default (`.text` for functions, `.data`/`.bss` for variables). The section name string is validated for non-empty content. Used extensively in the kernel for `__init`, `__exit`, and per-CPU data sections. | Functions, variables | ✅ Implemented |
| 4 | `used` | `__attribute__((used))` | Marks a symbol as externally reachable even when the compiler detects no internal references. Prevents dead-code elimination from removing the symbol. Emits the symbol in the output object unconditionally. Critical for kernel symbols referenced only from assembly or linker scripts. | Functions, variables | ✅ Implemented |
| 5 | `unused` | `__attribute__((unused))` | Suppresses compiler warnings about an unused entity. Does not change code generation. Applied frequently to function parameters in callback signatures where not all parameters are used in every implementation. | Functions, variables, parameters, labels, types | ✅ Implemented |
| 6 | `weak` | `__attribute__((weak))` | Assigns weak linkage binding to the symbol. A weak symbol may be overridden by a strong (global) definition at link time. If no strong definition exists, the weak definition is used. If no definition exists at all, the symbol resolves to zero/null. Used in the kernel for optional feature hooks. | Functions, variables | ✅ Implemented |
| 7 | `constructor` | `__attribute__((constructor))` or `__attribute__((constructor(priority)))` | Registers a function to be called automatically before `main()` is entered. The function pointer is placed in the `.init_array` ELF section. An optional integer priority argument controls execution order (lower values run first). | Functions | ✅ Implemented |
| 8 | `destructor` | `__attribute__((destructor))` or `__attribute__((destructor(priority)))` | Registers a function to be called automatically after `main()` returns or `exit()` is called. The function pointer is placed in the `.fini_array` ELF section. An optional priority argument controls execution order. | Functions | ✅ Implemented |
| 9 | `visibility` | `__attribute__((visibility("default"\|"hidden"\|"protected")))` | Controls ELF symbol visibility for shared library builds (`-fPIC` / `-shared`). `"default"` — symbol is exported and preemptible. `"hidden"` — symbol is not exported; available only within the shared object. `"protected"` — symbol is exported but not preemptible by other shared objects. | Functions, variables | ✅ Implemented |
| 10 | `deprecated` | `__attribute__((deprecated))` or `__attribute__((deprecated("message")))` | Causes the compiler to emit a warning whenever the marked symbol is referenced. An optional string argument provides a custom deprecation message included in the diagnostic. | Functions, variables, types | ✅ Implemented |
| 11 | `noreturn` | `__attribute__((noreturn))` | Declares that a function never returns to its caller (e.g., `exit()`, `abort()`, kernel `panic()`). Enables the compiler to omit function epilogue code and propagate unreachability information. Calling a `noreturn` function allows dead code elimination of subsequent statements. | Functions | ✅ Implemented |
| 12 | `noinline` | `__attribute__((noinline))` | Prevents the compiler from inlining the function at any call site, regardless of optimization level. Used in the kernel for functions that must maintain a distinct stack frame (e.g., for stack unwinding or instrumentation). | Functions | ✅ Implemented |
| 13 | `always_inline` | `__attribute__((always_inline))` | Forces the compiler to inline the function at every call site. Typically used in conjunction with the `inline` keyword. Failure to inline (e.g., due to recursion or address-taken) should produce a diagnostic. Used in kernel hot paths and architecture-specific helper wrappers. | Functions | ✅ Implemented |
| 14 | `cold` | `__attribute__((cold))` | Marks a function as unlikely to be executed during typical program operation. The compiler may place the function in a separate `.text.cold` section to improve instruction cache locality of hot code. Branch predictions to calls of cold functions are biased toward not-taken. | Functions | ✅ Implemented |
| 15 | `hot` | `__attribute__((hot))` | Marks a function as frequently executed. The compiler may place the function in a `.text.hot` section and optimize it more aggressively. Opposite of `cold`. | Functions | ✅ Implemented |
| 16 | `format` | `__attribute__((format(archetype, string_index, first_to_check)))` | Enables compile-time format string checking for printf-like or scanf-like functions. `archetype` is `printf`, `scanf`, `strftime`, or `strfmon`. `string_index` identifies the format string parameter (1-based). `first_to_check` identifies the first variadic argument to check (0 disables checking). | Functions | ✅ Implemented |
| 17 | `format_arg` | `__attribute__((format_arg(string_index)))` | Marks a function parameter as a format string that is returned (possibly after translation). The `string_index` parameter (1-based) identifies which parameter is the format string. Used for functions like `gettext()` that return a translated format string. | Functions | ✅ Implemented |
| 18 | `malloc` | `__attribute__((malloc))` | Declares that the returned pointer does not alias any other pointer accessible to the caller at the time of the call. Enables alias analysis optimizations. The returned pointer may alias pointers returned by subsequent calls to the same or other `malloc`-attributed functions. | Functions | ✅ Implemented |
| 19 | `pure` | `__attribute__((pure))` | Declares that the function has no observable side effects except through its return value, but may read from global memory or memory pointed to by its arguments. The compiler may eliminate redundant calls with the same arguments if no intervening writes occur. Weaker than `const`. | Functions | ✅ Implemented |
| 20 | `const` | `__attribute__((const))` | Declares that the function has no side effects and does not read any memory except its arguments. Strictly stronger than `pure` — the function result depends only on its parameter values. The compiler may hoist calls out of loops and eliminate redundant calls freely. | Functions | ✅ Implemented |
| 21 | `warn_unused_result` | `__attribute__((warn_unused_result))` | Causes the compiler to emit a warning when the return value of the function is discarded by the caller. Used in the kernel for error-returning functions where ignoring the return value is almost certainly a bug (e.g., `copy_from_user()`). | Functions | ✅ Implemented |
| 22 | `fallthrough` | `__attribute__((fallthrough))` | Statement attribute placed before a case label to indicate intentional fallthrough from the preceding case. Suppresses `-Wimplicit-fallthrough` warnings. Equivalent to the `/* fallthrough */` comment convention but machine-verifiable. | Statements (switch cases) | ✅ Implemented |

> **Note:** Additional attributes may be discovered and added during the Linux kernel 6.9 build
> phase. Any newly implemented attributes will be appended to this table and recorded in the
> [Kernel Build Extension Discovery Log](#kernel-build-extension-discovery-log).

---

## Language Extensions

BCC implements the following GCC language extensions beyond the C11 standard. These extensions
are required for compiling the Linux kernel and other GCC-dependent C codebases.

### 1. Statement Expressions

**Syntax:**
```c
({ statement1; statement2; expression; })
```

**Semantics:** A compound statement enclosed in parentheses produces a value — the value of the
last expression in the block. Variables declared inside the block are scoped to the statement
expression. The type of the statement expression is the type of the final expression.

**Kernel Usage:** Used extensively in kernel macros such as `min()`, `max()`, `container_of()`,
and type-checking wrappers where a macro must evaluate to a value but requires intermediate
local variables.

**Example:**
```c
#define max(a, b) ({       \
    typeof(a) _a = (a);    \
    typeof(b) _b = (b);    \
    _a > _b ? _a : _b;    \
})
```

**Implementation:** `src/frontend/parser/gcc_extensions.rs`, `src/frontend/parser/expressions.rs`

### 2. `typeof` / `__typeof__`

**Syntax:**
```c
typeof(expression)
typeof(type_name)
__typeof__(expression)
__typeof__(type_name)
```

**Semantics:** Yields the type of the operand. When the operand is an expression, the expression
is not evaluated — only its type is determined. Can be used anywhere a type name is valid:
variable declarations, casts, `sizeof`, and other type contexts.

**Kernel Usage:** Fundamental to type-safe kernel macros. Used in `min()`, `max()`,
`READ_ONCE()`, `WRITE_ONCE()`, and hundreds of other macros throughout the kernel tree.

**Example:**
```c
typeof(x) temp = x;       /* declares temp with same type as x */
typeof(int *) ptr;         /* declares ptr as int* */
```

**Implementation:** `src/frontend/parser/types.rs`

### 3. Zero-Length Arrays

**Syntax:**
```c
struct variable_data {
    int header;
    char data[0];  /* zero-length array */
};
```

**Semantics:** An array declared with zero elements at the end of a structure. The array itself
contributes zero bytes to the struct size, but provides a typed pointer to the memory
immediately following the struct. This is the pre-C99 idiom for variable-length structures
(C99 introduced flexible array members `char data[]` which BCC also supports).

**Kernel Usage:** Used throughout the kernel for variable-length data structures, network packet
buffers, and ring buffer entries.

**Implementation:** `src/frontend/parser/gcc_extensions.rs`

### 4. Designated Initializers (Extended)

**Syntax:**
```c
struct point p = { .y = 10, .x = 5 };              /* out-of-order */
int arr[10] = { [3] = 42, [7] = 99 };              /* array index */
struct nested n = { .outer.inner = 1 };             /* nested member */
int matrix[3][3] = { [0][1] = 5, [2][0] = 8 };    /* nested array */
```

**Semantics:** C11 provides basic designated initializers. BCC extends this with full GCC
compatibility: out-of-order field designation, nested member designation (`.field.subfield`),
nested array designation (`[i][j]`), and range designation (`[first ... last] = value`).
Unspecified members are implicitly zero-initialized.

**Kernel Usage:** Used pervasively for initializing complex kernel data structures, file
operations tables, and device driver descriptor arrays.

**Implementation:** `src/frontend/sema/initializer.rs`

### 5. Computed Gotos

**Syntax:**
```c
void *label_addr = &&my_label;   /* address-of-label */
goto *label_addr;                 /* indirect goto */
```

**Semantics:** The `&&label` operator takes the address of a label, returning a `void *` value.
The `goto *expr` statement performs an indirect jump to the address contained in the
expression. The label must be a valid label in the current function scope.

**Kernel Usage:** Used in the BPF interpreter and other kernel bytecode dispatch loops for
high-performance threaded dispatch tables.

**Example:**
```c
static void *dispatch_table[] = { &&op_add, &&op_sub, &&op_halt };
goto *dispatch_table[opcode];
op_add: result += *sp++; goto *dispatch_table[*pc++];
op_sub: result -= *sp++; goto *dispatch_table[*pc++];
op_halt: return result;
```

**Implementation:** `src/frontend/parser/statements.rs`, `src/ir/lowering/stmt_lowering.rs`

### 6. Case Ranges

**Syntax:**
```c
switch (c) {
    case 'a' ... 'z':   /* matches 'a' through 'z' inclusive */
        handle_lower();
        break;
    case '0' ... '9':   /* matches '0' through '9' inclusive */
        handle_digit();
        break;
}
```

**Semantics:** A `case` label with a range matches all values from the first value to the last
value, inclusive. Both endpoints must be integer constant expressions. The low value must be
less than or equal to the high value. The `...` is surrounded by spaces to avoid parsing
ambiguity with floating-point literals.

**Kernel Usage:** Used in character classification switch statements and opcode dispatch.

**Implementation:** `src/frontend/parser/statements.rs`

### 7. Conditional Operand Omission (Elvis Operator)

**Syntax:**
```c
x ?: y    /* equivalent to (x ? x : y) but x is evaluated only once */
```

**Semantics:** In a conditional expression `a ?: b`, if the condition `a` evaluates to a
non-zero (truthy) value, the result is `a` itself; otherwise the result is `b`. The key
difference from writing `a ? a : b` is that `a` is evaluated exactly once, which matters
when `a` has side effects or is expensive to compute.

**Kernel Usage:** Used in kernel macros for default-value patterns.

**Implementation:** `src/frontend/parser/expressions.rs`

### 8. `__extension__`

**Syntax:**
```c
__extension__ typedef unsigned long long uint64_t;
__extension__ ({ /* statement expression */ })
```

**Semantics:** The `__extension__` prefix suppresses compiler warnings about GCC extensions
when compiling in strict conformance mode (`-std=c11` or `-pedantic`). It does not change the
semantics of the following declaration or expression — it only silences diagnostics about
non-standard constructs.

**Kernel Usage:** Used in kernel headers to silence warnings when GCC extensions appear in
contexts that may be compiled with strict conformance flags.

**Implementation:** `src/frontend/parser/gcc_extensions.rs`

### 9. Transparent Unions

**Syntax:**
```c
typedef union __attribute__((transparent_union)) {
    int *ip;
    float *fp;
    void *vp;
} any_ptr;

void process(any_ptr p);
process(my_int_ptr);     /* caller passes int* directly, not union */
```

**Semantics:** A union with the `transparent_union` attribute allows callers to pass any
member type directly as a function argument, without explicitly wrapping it in a union value.
The calling convention uses the representation of the first union member. At the ABI level,
the parameter is passed as though it were the first member type.

**Kernel Usage:** Used in the kernel for system call argument types and backward-compatible
API transitions.

**Implementation:** `src/frontend/parser/gcc_extensions.rs`, `src/frontend/sema/attribute_handler.rs`

### 10. Local Labels

**Syntax:**
```c
{
    __label__ retry, done;
    /* ... */
    retry:
        if (failed) goto retry;
    done:
        return result;
}
```

**Semantics:** `__label__` declares label names whose scope is limited to the enclosing block
(rather than the entire function, as with normal C labels). This is critical for macros that
contain labels — without local labels, expanding the macro more than once in the same function
would produce duplicate label errors.

**Kernel Usage:** Used in kernel macros that contain internal control flow (retry loops,
error handling) to prevent label name collisions when the macro is expanded multiple times.

**Implementation:** `src/frontend/parser/statements.rs`

---

## GCC Builtins

BCC implements approximately 30 GCC builtin functions. These builtins are divided into two
categories: **compile-time builtins** that are fully evaluated during semantic analysis, and
**runtime builtins** that emit architecture-specific machine code or function calls during
code generation.

### Compile-Time Evaluation Builtins

These builtins are resolved entirely at compile time by `src/frontend/sema/builtin_eval.rs`.
They do not generate any runtime code.

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 1 | `__builtin_constant_p` | `int __builtin_constant_p(expr)` | Returns `1` if `expr` can be determined to be a compile-time constant, `0` otherwise. The expression is not evaluated for side effects. Used in kernel macros to select between compile-time-optimized and runtime code paths. | ✅ Implemented |
| 2 | `__builtin_types_compatible_p` | `int __builtin_types_compatible_p(type1, type2)` | Returns `1` if `type1` and `type2` are compatible types (ignoring top-level qualifiers such as `const` and `volatile`), `0` otherwise. Does not evaluate any expression. Used in kernel `BUILD_BUG_ON` and type-checking macros. | ✅ Implemented |
| 3 | `__builtin_choose_expr` | `type __builtin_choose_expr(const_expr, expr1, expr2)` | Compile-time conditional selection. If `const_expr` evaluates to non-zero, the entire expression is replaced by `expr1`; otherwise by `expr2`. Unlike a ternary operator, only the selected branch is type-checked. Similar to C11 `_Generic` but driven by value rather than type. | ✅ Implemented |
| 4 | `__builtin_offsetof` | `size_t __builtin_offsetof(type, member)` | Returns the byte offset of `member` within `type`. Equivalent to the `offsetof` macro from `<stddef.h>`. Handles nested member access (e.g., `__builtin_offsetof(struct s, field.subfield)`). | ✅ Implemented |

### Branch Prediction and Control Flow Builtins

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 5 | `__builtin_expect` | `long __builtin_expect(long expr, long value)` | Branch prediction hint. Returns `expr` unchanged but informs the compiler that `expr` is expected to equal `value`. The compiler may reorder basic blocks to favor the expected path. Used via the kernel `likely()` and `unlikely()` macros. | ✅ Implemented |
| 6 | `__builtin_unreachable` | `void __builtin_unreachable(void)` | Informs the compiler that the current execution path is unreachable. Reaching this point is undefined behavior. Enables dead code elimination and optimization of preceding branches. Used after `switch` defaults that are known to be unreachable. | ✅ Implemented |
| 7 | `__builtin_trap` | `void __builtin_trap(void)` | Generates a trap/fault instruction that terminates the program abnormally. On x86, this emits `ud2`. On AArch64, this emits `.inst 0xde01` (permanent undefined). On RISC-V, this emits `unimp`. | ✅ Implemented |

### Bit Manipulation Builtins

These builtins emit architecture-specific instructions where available or software fallback sequences otherwise.

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 8 | `__builtin_clz` | `int __builtin_clz(unsigned int x)` | Count Leading Zeros — returns the number of leading 0-bits in `x`, starting from the most significant bit. Behavior is undefined if `x` is 0. Emits `bsr`/`lzcnt` on x86, `clz` on AArch64, or software sequence on RISC-V. | ✅ Implemented |
| 9 | `__builtin_clzl` | `int __builtin_clzl(unsigned long x)` | CLZ for `unsigned long` (64-bit on LP64 targets, 32-bit on ILP32). | ✅ Implemented |
| 10 | `__builtin_clzll` | `int __builtin_clzll(unsigned long long x)` | CLZ for `unsigned long long` (always 64-bit). | ✅ Implemented |
| 11 | `__builtin_ctz` | `int __builtin_ctz(unsigned int x)` | Count Trailing Zeros — returns the number of trailing 0-bits in `x`, starting from the least significant bit. Behavior is undefined if `x` is 0. Emits `bsf`/`tzcnt` on x86, `rbit`+`clz` on AArch64. | ✅ Implemented |
| 12 | `__builtin_ctzl` | `int __builtin_ctzl(unsigned long x)` | CTZ for `unsigned long`. | ✅ Implemented |
| 13 | `__builtin_ctzll` | `int __builtin_ctzll(unsigned long long x)` | CTZ for `unsigned long long`. | ✅ Implemented |
| 14 | `__builtin_popcount` | `int __builtin_popcount(unsigned int x)` | Population Count — returns the number of 1-bits in `x`. Emits `popcnt` on x86 (with SSE4.2), software fallback otherwise. | ✅ Implemented |
| 15 | `__builtin_popcountl` | `int __builtin_popcountl(unsigned long x)` | Popcount for `unsigned long`. | ✅ Implemented |
| 16 | `__builtin_popcountll` | `int __builtin_popcountll(unsigned long long x)` | Popcount for `unsigned long long`. | ✅ Implemented |
| 17 | `__builtin_ffs` | `int __builtin_ffs(int x)` | Find First Set — returns the 1-based index of the least significant 1-bit in `x`, or 0 if `x` is 0. Equivalent to `ctz(x) + 1` for non-zero `x`. | ✅ Implemented |
| 18 | `__builtin_ffsl` | `int __builtin_ffsl(long x)` | FFS for `long`. | ✅ Implemented |
| 19 | `__builtin_ffsll` | `int __builtin_ffsll(long long x)` | FFS for `long long`. | ✅ Implemented |

### Byte Swap Builtins

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 20 | `__builtin_bswap16` | `uint16_t __builtin_bswap16(uint16_t x)` | Reverses the byte order of a 16-bit value. Emits `ror` (x86), `rev16` (AArch64), or shift/mask sequence. | ✅ Implemented |
| 21 | `__builtin_bswap32` | `uint32_t __builtin_bswap32(uint32_t x)` | Reverses the byte order of a 32-bit value. Emits `bswap` (x86), `rev` (AArch64). | ✅ Implemented |
| 22 | `__builtin_bswap64` | `uint64_t __builtin_bswap64(uint64_t x)` | Reverses the byte order of a 64-bit value. Emits `bswap` (x86-64), `rev` (AArch64). | ✅ Implemented |

### Variadic Argument Builtins

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 23 | `__builtin_va_start` | `void __builtin_va_start(va_list ap, last_param)` | Initializes the `va_list` object `ap` to point to the first variadic argument following `last_param`. Must be called before any `va_arg` access. Architecture-specific implementation (register save area on x86-64, stack pointer on i686). | ✅ Implemented |
| 24 | `__builtin_va_end` | `void __builtin_va_end(va_list ap)` | Performs cleanup of the `va_list` object. Must be called before the function returns if `va_start` or `va_copy` was used. | ✅ Implemented |
| 25 | `__builtin_va_arg` | `type __builtin_va_arg(va_list ap, type)` | Retrieves the next variadic argument of the specified `type` and advances the `va_list`. Behavior is undefined if the type does not match the actual argument or if `ap` was not initialized. | ✅ Implemented |
| 26 | `__builtin_va_copy` | `void __builtin_va_copy(va_list dest, va_list src)` | Creates an independent copy of `src` in `dest`. Both `dest` and `src` may be used independently after the copy. `dest` must be cleaned up with `va_end`. | ✅ Implemented |

### Stack and Frame Introspection Builtins

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 27 | `__builtin_frame_address` | `void *__builtin_frame_address(unsigned int level)` | Returns the frame pointer of the function at the specified call stack `level`. Level 0 is the current function. Level 1 is the caller. Higher levels walk the frame chain. Behavior is undefined if the specified level exceeds the actual stack depth. | ✅ Implemented |
| 28 | `__builtin_return_address` | `void *__builtin_return_address(unsigned int level)` | Returns the return address of the function at the specified call stack `level`. Level 0 is the return address of the current function. Higher levels walk the frame chain. | ✅ Implemented |

### Alignment Builtin

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 29 | `__builtin_assume_aligned` | `void *__builtin_assume_aligned(const void *ptr, size_t align)` | Returns `ptr` unchanged but asserts to the compiler that `ptr` is aligned to at least `align` bytes. The compiler may use this information for optimization (e.g., selecting aligned load/store instructions). | ✅ Implemented |

### Overflow Arithmetic Builtins

| # | Builtin | Signature | Description | Status |
|---|---------|-----------|-------------|--------|
| 30 | `__builtin_add_overflow` | `bool __builtin_add_overflow(type a, type b, type *result)` | Performs `a + b`, stores the result in `*result`, and returns `true` if the operation overflowed (unsigned wrap or signed overflow). The types of `a`, `b`, and `*result` may differ — all are promoted to a common type for the operation. | ✅ Implemented |
| 31 | `__builtin_sub_overflow` | `bool __builtin_sub_overflow(type a, type b, type *result)` | Performs `a - b` with overflow detection. Stores the result in `*result` and returns `true` on overflow. | ✅ Implemented |
| 32 | `__builtin_mul_overflow` | `bool __builtin_mul_overflow(type a, type b, type *result)` | Performs `a * b` with overflow detection. Stores the result in `*result` and returns `true` on overflow. Uses widening multiplication where available. | ✅ Implemented |

> **Note:** Additional builtins may be discovered during the Linux kernel 6.9 build phase. Any
> newly implemented builtins will be appended to the appropriate category table and recorded in
> the [Kernel Build Extension Discovery Log](#kernel-build-extension-discovery-log).

---

## Inline Assembly

BCC implements full support for GCC-style inline assembly using AT&T syntax. This includes
basic and extended `asm` statements, `asm volatile`, `asm goto`, named operands, and the
critical `.pushsection`/`.popsection` directives used by the Linux kernel for instrumentation.

### Basic Syntax

```c
asm("instruction");                         /* basic asm (no operands) */
__asm__("instruction");                     /* alternate spelling */
asm volatile ("instruction");               /* volatile — not optimized away */
```

### Extended Syntax

```c
asm [volatile] [goto] (
    "assembly template"
    : output_operands       /* optional — comma-separated list */
    : input_operands        /* optional — comma-separated list */
    : clobber_list          /* optional — comma-separated strings */
    : goto_labels           /* optional — only with asm goto */
);
```

**Template String:** The assembly template is a string literal containing AT&T syntax
instructions. Operand references use `%0`, `%1`, etc. (positional) or `%[name]` (named).
Register references use `%%` prefix (e.g., `%%rax`) to distinguish from operand references.

### Operand Constraints

#### Output Operands

| Constraint | Meaning | Example |
|------------|---------|---------|
| `"=r"` | Write-only register | `"=r" (result)` |
| `"=m"` | Write-only memory | `"=m" (global_var)` |
| `"+r"` | Read-write register | `"+r" (counter)` |
| `"+m"` | Read-write memory | `"+m" (shared_var)` |
| `"=&r"` | Early-clobber register (written before inputs consumed) | `"=&r" (temp)` |
| `"=a"` | Specific register: `eax`/`rax` (x86) | `"=a" (low_result)` |
| `"=d"` | Specific register: `edx`/`rdx` (x86) | `"=d" (high_result)` |

#### Input Operands

| Constraint | Meaning | Example |
|------------|---------|---------|
| `"r"` | Any general-purpose register | `"r" (value)` |
| `"m"` | Memory operand | `"m" (mem_var)` |
| `"i"` | Immediate integer operand | `"i" (42)` |
| `"n"` | Known numeric constant (compile-time) | `"n" (CONST_VAL)` |
| `"0"`, `"1"` | Matching constraint (same location as output N) | `"0" (initial_val)` |
| `"a"` | Specific register: `eax`/`rax` (x86) | `"a" (dividend_lo)` |
| `"d"` | Specific register: `edx`/`rdx` (x86) | `"d" (dividend_hi)` |

#### Named Operands

```c
asm ("add %[src], %[dst]"
    : [dst] "+r" (result)
    : [src] "r" (addend));
```

Named operands use `[name]` before the constraint string. In the template, `%[name]`
references the operand by name instead of by positional index. This improves readability
for complex inline assembly with many operands.

### Clobber Lists

Clobber declarations inform the compiler which registers or memory locations are modified
by the assembly instructions beyond the explicit output operands.

| Clobber | Meaning |
|---------|---------|
| `"rax"`, `"rcx"`, `"rdx"`, etc. | Specific register is modified |
| `"memory"` | Assembly reads or writes memory not listed in operands; acts as a compiler memory barrier |
| `"cc"` | Assembly modifies the condition code / flags register (EFLAGS on x86, NZCV on AArch64) |

**Example:**
```c
asm volatile (
    "cpuid"
    : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
    : "a" (function_id), "c" (sub_id)
    : "memory"
);
```

### `asm goto`

`asm goto` extends the inline assembly syntax with jump label targets, allowing the assembly
code to transfer control to C labels in the containing function.

```c
asm goto (
    "cmpxchg %[new], %[ptr]\n\t"
    "jne %l[retry]"
    : /* no outputs allowed with asm goto */
    : [ptr] "m" (*ptr), [new] "r" (new_val), "a" (expected)
    : "memory", "cc"
    : retry                     /* label target */
);
/* fall-through path: cmpxchg succeeded */
return;

retry:
    expected = *ptr;
    goto try_again;
```

**Key Rules:**
- `asm goto` statements may not have output operands (GCC limitation)
- Jump labels are listed in the fourth colon-separated section
- Labels are referenced in the template via `%l[name]` or `%l0`, `%l1`, etc.
- Each label must be a valid C label within the current function scope

### Directives: `.pushsection` / `.popsection`

The `.pushsection` and `.popsection` assembly directives allow inline assembly to emit data
into alternate ELF sections without disrupting the current section context.

```c
asm volatile (
    ".pushsection .data\n\t"
    ".asciz \"trace_marker_string\"\n\t"
    ".popsection"
);
```

**Kernel Usage:** This mechanism is critical for Linux kernel instrumentation frameworks:
- **ftrace:** Inserts `nop` sleds with metadata in `__mcount_loc`
- **static keys / jump labels:** Emits patch-site descriptors in `__jump_table`
- **kprobes:** Records probe points in `__kprobes_text`
- **tracepoints:** Registers trace callsites in `__tracepoints`

The assembler must correctly handle section switching and return to the previous section
context after `.popsection`.

### Architecture-Specific Constraints

Each target architecture supports additional architecture-specific register constraints:

| Architecture | Additional Constraints |
|-------------|----------------------|
| x86-64 | `"a"` (rax), `"b"` (rbx), `"c"` (rcx), `"d"` (rdx), `"S"` (rsi), `"D"` (rdi) |
| i686 | `"a"` (eax), `"b"` (ebx), `"c"` (ecx), `"d"` (edx), `"S"` (esi), `"D"` (edi) |
| AArch64 | `"r"` (any X register), `"w"` (any SIMD/FP V register) |
| RISC-V 64 | `"r"` (any x register), `"f"` (any f register) |

### Implementation Files

| Component | File | Responsibility |
|-----------|------|----------------|
| Parsing | `src/frontend/parser/inline_asm.rs` | Parse `asm`/`__asm__` statements, operand lists, constraints, clobbers, goto labels |
| AST representation | `src/frontend/parser/ast.rs` | `AsmStatement` node with template, operands, clobbers, labels |
| IR lowering | `src/ir/lowering/asm_lowering.rs` | Bind operands to IR values, validate constraints, wire goto targets |
| Machine encoding | `src/backend/*/assembler/` | Architecture-specific instruction encoding for template strings |

---

## Kernel Build Extension Discovery Log

This section tracks GCC extensions discovered during the Linux kernel 6.9 build phase that
were not included in the initial manifest above. Each entry records the extension, the kernel
source file that triggered the discovery, and the implementation status.

The purpose of this log is to maintain a complete record of every extension addition made
during iterative kernel compilation, ensuring no extension is silently dropped or forgotten.

### Failure Classification Protocol

When a kernel source file fails to compile, the root cause is diagnosed using this priority
order:

1. **Missing GCC extension** — An `__attribute__`, language construct, or syntax not yet
   recognized by the parser or semantic analyzer
2. **Missing builtin** — A `__builtin_*` function call not yet implemented
3. **Inline assembly constraint issue** — An unrecognized operand constraint, clobber, or
   template directive
4. **Preprocessor issue** — A macro expansion failure, missing predefined macro, or
   conditional compilation error
5. **Code generation bug** — An internal compiler error during IR lowering, optimization,
   or machine code emission

### Discovery Log

| # | Extension | Kernel Source File | Date Discovered | Implementation Status | Notes |
|---|-----------|-------------------|-----------------|----------------------|-------|
| | *(to be populated during kernel build)* | | | | |

> Entries are added chronologically as kernel compilation surfaces new extension requirements.
> Each entry transitions through statuses: **Discovered** → **Implementing** → **Implemented** → **Validated**.

---

## C11 Baseline Features

In addition to GCC extensions, BCC implements the full C11 (ISO/IEC 9899:2011) feature set
as its language baseline. The following C11 features beyond the C99 standard are supported:

| Feature | Syntax | Description |
|---------|--------|-------------|
| `_Static_assert` | `_Static_assert(expr, "message");` | Compile-time assertion. If `expr` evaluates to zero, compilation fails with the specified message. Evaluated during semantic analysis by `src/frontend/sema/constant_eval.rs`. |
| `_Alignof` | `_Alignof(type)` | Returns the alignment requirement (in bytes) of the specified type. Also available as `__alignof__` (GCC spelling). |
| `_Alignas` | `_Alignas(alignment)` or `_Alignas(type)` | Specifies the alignment requirement for a variable or struct member. The alignment must be a positive power of two and at least as large as the type's natural alignment. |
| `_Noreturn` | `_Noreturn void func(void);` | Function specifier declaring that the function does not return. Semantically equivalent to `__attribute__((noreturn))`. |
| `_Generic` | `_Generic(expr, type1: e1, type2: e2, default: e3)` | Type-generic selection expression. Selects one of the association expressions based on the type of the controlling expression. Exactly one association must match. Used for type-generic macro implementations. |
| `_Atomic` | `_Atomic int counter;` or `_Atomic(type)` | Type qualifier for atomic types. Ensures that operations on the qualified object are atomic with respect to other threads. Storage representation matches the unqualified type with appropriate alignment. Actual atomic operations may delegate to `libatomic` at link time. |
| `_Thread_local` | `_Thread_local int tls_var;` | Storage class specifier for thread-local storage. Each thread receives its own copy of the variable. Also available as `__thread` (GCC spelling). Implemented via TLS segments in the ELF output. |
| `_Complex` | `double _Complex z;` | Complex number type support. Represents a pair of real and imaginary floating-point values. Supports arithmetic operations `+`, `-`, `*`, `/` on complex operands. |
| Anonymous structs/unions | `struct s { union { int x; float f; }; };` | Unnamed struct or union members whose fields are accessible directly through the containing type. Eliminates the need for an intermediate member name. |
| Unicode literals | `u8"UTF-8"`, `u"UTF-16"`, `U"UTF-32"`, `L"wide"` | String and character literal prefixes specifying encoding. `u8` produces UTF-8 encoded `char` arrays. `u` produces `char16_t` arrays. `U` produces `char32_t` arrays. `L` produces `wchar_t` arrays. |

### Predefined C11 Macros

BCC defines the following standard macros to indicate C11 conformance:

| Macro | Value | Description |
|-------|-------|-------------|
| `__STDC__` | `1` | Conforms to the ISO C standard |
| `__STDC_VERSION__` | `201112L` | Conforms to C11 (ISO/IEC 9899:2011) |
| `__STDC_HOSTED__` | `1` | Hosted implementation (has standard library) |
| `__STDC_UTF_16__` | `1` | `char16_t` uses UTF-16 encoding |
| `__STDC_UTF_32__` | `1` | `char32_t` uses UTF-32 encoding |
| `__STDC_NO_ATOMICS__` | *(not defined)* | BCC supports `_Atomic` — this macro is absent |
| `__STDC_NO_COMPLEX__` | *(not defined)* | BCC supports `_Complex` — this macro is absent |
| `__STDC_NO_THREADS__` | *(not defined)* | BCC supports `_Thread_local` — this macro is absent |

---

## Appendix: Extension Status Summary

### Coverage Statistics

| Category | Total | Implemented | Remaining |
|----------|-------|-------------|-----------|
| GCC Attributes | 22 | 22 | 0 |
| Language Extensions | 10 | 10 | 0 |
| GCC Builtins | 32 | 32 | 0 |
| Inline Assembly Features | Full | Full | — |
| C11 Baseline Features | 9 | 9 | 0 |

### Cross-Reference: Implementation Files

| File | Extensions Handled |
|------|--------------------|
| `src/frontend/parser/attributes.rs` | All 22 GCC attributes (parsing) |
| `src/frontend/sema/attribute_handler.rs` | All 22 GCC attributes (semantic validation) |
| `src/frontend/parser/gcc_extensions.rs` | Statement expressions, zero-length arrays, `__extension__`, transparent unions |
| `src/frontend/parser/expressions.rs` | Statement expressions (value), conditional omission (`?:`) |
| `src/frontend/parser/statements.rs` | Computed gotos, case ranges, local labels |
| `src/frontend/parser/types.rs` | `typeof`/`__typeof__`, C11 type qualifiers/specifiers |
| `src/frontend/parser/inline_asm.rs` | Full inline assembly parsing |
| `src/frontend/sema/builtin_eval.rs` | All compile-time builtins (4) |
| `src/frontend/sema/initializer.rs` | Designated initializers (extended) |
| `src/ir/lowering/asm_lowering.rs` | Inline assembly IR lowering |
| `src/ir/lowering/stmt_lowering.rs` | Computed gotos IR lowering |
| `src/backend/*/assembler/` | Architecture-specific asm encoding, runtime builtins |
| `src/frontend/preprocessor/predefined.rs` | C11 predefined macros, architecture macros |
