//! Predefined macro registration module for the BCC (Blitzy C Compiler).
//!
//! This module is responsible for populating the preprocessor's macro table with
//! all predefined macros **before** any user source file is processed. Predefined
//! macros fall into several categories:
//!
//! 1. **Dynamic macros** — `__FILE__`, `__LINE__`, `__DATE__`, `__TIME__`,
//!    `__COUNTER__`: these are marked with `is_predefined = true` and carry an
//!    empty body. The macro expander detects this flag and computes their values
//!    at the point of expansion rather than from a static replacement list.
//!
//! 2. **C11 standard compliance** — `__STDC__`, `__STDC_VERSION__` (201112L),
//!    `__STDC_HOSTED__`, `__STDC_UTF_16__`, `__STDC_UTF_32__`, `__STDC_NO_VLA__`:
//!    static macros that signal conformance features per ISO/IEC 9899:2011.
//!
//! 3. **Architecture-specific** — obtained from `Target::predefined_macros()`,
//!    e.g. `__x86_64__`, `__aarch64__`, `__riscv`, `__riscv_xlen=64`.
//!
//! 4. **Platform** — `__linux__`, `__gnu_linux__`, `__ELF__`, `__unix__` and
//!    their unadorned variants, signalling the Linux/ELF hosted environment.
//!
//! 5. **Compiler identification** — `__BCC__`, `__BCC_VERSION__`, and
//!    GCC-compatibility macros (`__GNUC__=12`, etc.) required for Linux kernel
//!    header processing.
//!
//! 6. **Type size / limit** — `__SIZEOF_*__`, `__INT_MAX__`, `__LONG_MAX__`,
//!    `__LONG_LONG_MAX__`, `__GCC_HAVE_SYNC_COMPARE_AND_SWAP_*` etc.
//!
//! 7. **Command-line defines** — `-D` flags from the CLI, which take precedence
//!    over all predefined macros.
//!
//! # Usage
//!
//! ```ignore
//! use crate::frontend::preprocessor::predefined;
//!
//! let target = Target::X86_64;
//! predefined::register_predefined_macros(&mut pp, &target);
//! predefined::register_cli_defines(&mut pp, &cli_defines);
//! ```
//!
//! # Integration
//!
//! This module is a child of `src/frontend/preprocessor/mod.rs` and accesses the
//! `Preprocessor` struct's public fields (`macros`, `interner`) via `super::`.
//! It depends on `crate::common::target::Target` for architecture-specific macro
//! sets and on `crate::frontend::lexer::token` for token construction.

// ─── Imports from parent module ─────────────────────────────────────────────
use super::{MacroDef, Preprocessor};

// ─── Internal crate imports ─────────────────────────────────────────────────
use crate::common::string_interner::Symbol;
use crate::common::target::{DataModel, Target};
use crate::frontend::lexer::token::{IntegerSuffix, Span, StringPrefix, Token, TokenKind};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// BCC compiler version string, used for `__BCC_VERSION__` and `__VERSION__`.
const BCC_VERSION: &str = "1.0.0";

// ═══════════════════════════════════════════════════════════════════════════
// Token Construction Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Creates an integer literal token at a dummy source location.
///
/// Used to construct the replacement body for integer-valued predefined macros
/// such as `__STDC__` → `1`, `__SIZEOF_INT__` → `4`, etc.
#[inline]
fn int_token(value: u128, suffix: IntegerSuffix) -> Token {
    Token::new(TokenKind::IntegerLiteral { value, suffix }, Span::DUMMY)
}

/// Creates a string literal token at a dummy source location.
///
/// Used to construct the replacement body for string-valued predefined macros
/// such as `__BCC_VERSION__` → `"1.0.0"`, `__VERSION__` → `"BCC 1.0.0"`.
#[inline]
fn str_token(value: &[u8]) -> Token {
    Token::new(
        TokenKind::StringLiteral {
            value: value.to_vec(),
            prefix: StringPrefix::None,
        },
        Span::DUMMY,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Macro Registration Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Registers a dynamic predefined macro with `is_predefined = true` and an
/// empty body.
///
/// Dynamic macros are handled specially by the macro expander: instead of
/// using a static replacement list, the expander generates a fresh value at
/// each expansion point. This mechanism supports `__FILE__` (current file
/// name), `__LINE__` (current line number), `__DATE__`, `__TIME__`, and
/// `__COUNTER__`.
fn define_dynamic(pp: &mut Preprocessor, name: &str) {
    let sym: Symbol = pp.interner.intern(name);
    pp.macros.insert(sym, MacroDef::predefined(sym));
}

/// Registers a static, integer-valued object-like macro.
///
/// Creates a `MacroDef` with a single `IntegerLiteral` token as its body.
/// The `suffix` parameter controls the C type suffix (e.g., `IntegerSuffix::L`
/// for `201112L`).
fn define_int(pp: &mut Preprocessor, name: &str, value: u128, suffix: IntegerSuffix) {
    let sym: Symbol = pp.interner.intern(name);
    let body = vec![int_token(value, suffix)];
    let def = MacroDef::object_like(sym, body, Span::DUMMY);
    pp.macros.insert(sym, def);
}

/// Registers a static, integer-valued object-like macro only if the name
/// is not already present in the macro table.
///
/// Prevents overwriting dynamically-evaluated macros or macros already
/// registered by `Target::predefined_macros()`.
fn define_int_if_absent(pp: &mut Preprocessor, name: &str, value: u128) {
    let sym: Symbol = pp.interner.intern(name);
    if pp.macros.contains_key(&sym) {
        return;
    }
    let body = vec![int_token(value, IntegerSuffix::None)];
    let def = MacroDef::object_like(sym, body, Span::DUMMY);
    pp.macros.insert(sym, def);
}

/// Registers a static, string-valued object-like macro.
///
/// Creates a `MacroDef` with a single `StringLiteral` token as its body.
fn define_str(pp: &mut Preprocessor, name: &str, value: &[u8]) {
    let sym: Symbol = pp.interner.intern(name);
    let body = vec![str_token(value)];
    let def = MacroDef::object_like(sym, body, Span::DUMMY);
    pp.macros.insert(sym, def);
}

/// Registers a static, identifier-valued object-like macro.
///
/// Creates a `MacroDef` whose body consists of `Identifier` tokens. Multi-word
/// values (e.g., `"long unsigned int"`) are split on whitespace into separate
/// tokens so that the parser processes them correctly as distinct keywords.
fn define_ident(pp: &mut Preprocessor, name: &str, value: &str) {
    let name_sym: Symbol = pp.interner.intern(name);
    let mut body: Vec<Token> = Vec::new();
    for word in value.split_whitespace() {
        let val_sym: Symbol = pp.interner.intern(word);
        body.push(Token::new(TokenKind::Identifier(val_sym), Span::DUMMY));
    }
    let def = MacroDef::object_like(name_sym, body, Span::DUMMY);
    pp.macros.insert(name_sym, def);
}

/// Registers a macro from a `(name, value)` string pair, automatically
/// determining the token kind.
///
/// Attempts the following parse order:
/// 1. Integer with C suffix (e.g., `"201112L"` → `IntegerLiteral` with `L`)
/// 2. Plain unsigned integer (e.g., `"4"` → `IntegerLiteral`)
/// 3. Identifier fallback (e.g., `"__ORDER_LITTLE_ENDIAN__"` → `Identifier`)
///
/// An empty value string produces a flag-style macro with an empty body
/// (equivalent to `#define NAME`).
fn define_from_str_pair(pp: &mut Preprocessor, name: &str, value: &str) {
    if value.is_empty() {
        // Flag-style define: `#define NAME` (empty body).
        let sym: Symbol = pp.interner.intern(name);
        let def = MacroDef::object_like(sym, Vec::new(), Span::DUMMY);
        pp.macros.insert(sym, def);
        return;
    }

    // Attempt integer with suffix (e.g., "201112L", "0xFFFFFFFFULL").
    if let Some((num_val, suffix)) = parse_integer_with_suffix(value) {
        define_int(pp, name, num_val, suffix);
        return;
    }

    // Attempt plain unsigned integer.
    if let Ok(n) = value.parse::<u128>() {
        define_int(pp, name, n, IntegerSuffix::None);
        return;
    }

    // Fallback: treat value as an identifier token.
    define_ident(pp, name, value);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer Suffix Parsing
// ═══════════════════════════════════════════════════════════════════════════

/// Parses an integer literal string with an optional C type suffix.
///
/// Recognises the following suffixes (case-insensitive, longest match first):
///
/// | Suffix  | `IntegerSuffix` variant |
/// |---------|------------------------|
/// | `ull`, `llu` | `ULL`           |
/// | `ll`    | `LL`                   |
/// | `ul`, `lu` | `UL`              |
/// | `l`     | `L`                    |
/// | `u`     | `U`                    |
///
/// Returns `Some((value, suffix))` on success, `None` if the string does not
/// end with a recognised suffix or the numeric prefix cannot be parsed.
fn parse_integer_with_suffix(s: &str) -> Option<(u128, IntegerSuffix)> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return None;
    }

    // Determine suffix length and kind by inspecting trailing bytes
    // (case-insensitive). We check the longest suffixes first to avoid
    // mis-parsing "ull" as "u" + "ll".
    let lower: Vec<u8> = bytes.iter().map(|b| b.to_ascii_lowercase()).collect();

    let (suffix_len, suffix) = if len >= 3
        && ((lower[len - 3] == b'u' && lower[len - 2] == b'l' && lower[len - 1] == b'l')
            || (lower[len - 3] == b'l' && lower[len - 2] == b'l' && lower[len - 1] == b'u'))
    {
        (3, IntegerSuffix::ULL)
    } else if len >= 2 && lower[len - 2] == b'l' && lower[len - 1] == b'l' {
        (2, IntegerSuffix::LL)
    } else if len >= 2
        && ((lower[len - 2] == b'u' && lower[len - 1] == b'l')
            || (lower[len - 2] == b'l' && lower[len - 1] == b'u'))
    {
        (2, IntegerSuffix::UL)
    } else if lower[len - 1] == b'l' {
        (1, IntegerSuffix::L)
    } else if lower[len - 1] == b'u' {
        (1, IntegerSuffix::U)
    } else {
        return None;
    };

    let num_part = &s[..len - suffix_len];
    if num_part.is_empty() {
        return None;
    }

    // Support hex (0x), octal (0), and decimal number prefixes.
    let parsed = if num_part.starts_with("0x") || num_part.starts_with("0X") {
        u128::from_str_radix(&num_part[2..], 16).ok()
    } else if num_part.starts_with('0') && num_part.len() > 1 {
        u128::from_str_radix(&num_part[1..], 8).ok()
    } else {
        num_part.parse::<u128>().ok()
    };

    parsed.map(|v| (v, suffix))
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API — Main Registration Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Registers **all** predefined preprocessor macros for the BCC compiler.
///
/// This function must be called during preprocessor initialisation, **before**
/// any source file is processed. It populates the preprocessor's macro table
/// (`pp.macros`) with every predefined macro required for C11 compliance, GCC
/// compatibility, and Linux kernel header processing.
///
/// # Registration Order
///
/// 1. Dynamic macros (`__FILE__`, `__LINE__`, `__DATE__`, `__TIME__`, `__COUNTER__`)
/// 2. Architecture-specific and platform macros from `Target::predefined_macros()`
/// 3. Additional C11 standard macros not covered by the target set
/// 4. Type-size macros not included in the target set
/// 5. Type-limit macros (`__INT_MAX__`, `__LONG_MAX__`, etc.)
/// 6. GCC sync-builtin indicator macros
/// 7. Compiler version identification macros
///
/// Dynamic macros are registered first so that static macros from the target
/// set do not overwrite them (the loop in [`register_target_macros`] skips
/// names already present in the table).
///
/// # Arguments
///
/// * `pp` — Mutable reference to the preprocessor. Accesses `pp.macros` for
///   macro insertion and `pp.interner` for string interning.
/// * `target` — Target architecture, used to obtain architecture-specific
///   predefined macros and type-size constants.
pub fn register_predefined_macros(pp: &mut Preprocessor, target: &Target) {
    // Phase 1 — Dynamic macros: evaluated at expansion point by the expander.
    register_dynamic_macros(pp);

    // Phase 2 — Architecture + platform + standard macros from the target.
    //           This covers __STDC__, __linux__, arch defines, __SIZEOF_*__,
    //           __GNUC__, __BCC__, byte-order macros, etc.
    register_target_macros(pp, target);

    // Phase 3 — Extra C11 standard macros not returned by target.
    register_c11_extra_macros(pp);

    // Phase 4 — Type-size macros not covered by target.predefined_macros().
    register_extra_type_size_macros(pp, target);

    // Phase 5 — Integer type-limit macros (__INT_MAX__, __LONG_MAX__, etc.).
    register_type_limit_macros(pp, target);

    // Phase 6 — GCC __sync CAS indicator macros (for kernel atomics).
    register_gcc_sync_macros(pp);

    // Phase 7 — Compiler version identification strings.
    register_compiler_version_macros(pp);

    // Phase 8 — GCC built-in type compatibility macros.
    //   `__float128` is used in system headers (stddef.h max_align_t) on i386.
    //   Map it to `long double` for compatibility since we don't have a native
    //   128-bit float type.
    register_gcc_type_compat_macros(pp);
}

/// Registers command-line `-D` macro definitions into the preprocessor.
///
/// Each entry in `defines` is a `(name, optional_value)` pair originating
/// from the CLI driver's parsing of `-DFOO` and `-DFOO=bar` flags.
///
/// # Value Interpretation
///
/// | Input                       | Effect                              |
/// |-----------------------------|-------------------------------------|
/// | `("FOO", None)`             | `#define FOO 1`                     |
/// | `("FOO", Some("42"))`       | `#define FOO 42`  (integer literal) |
/// | `("FOO", Some("bar"))`      | `#define FOO bar` (identifier)      |
/// | `("FOO", Some(""))`         | `#define FOO`     (empty body)      |
/// | `("FOO", Some("201112L"))` | `#define FOO 201112L` (long int)    |
///
/// # Precedence
///
/// CLI defines are registered **after** predefined macros and unconditionally
/// overwrite any existing macro with the same name. This allows the user to
/// override predefined macros (e.g., `-D__STDC_VERSION__=199901L`).
///
/// # Arguments
///
/// * `pp` — Mutable reference to the preprocessor.
/// * `defines` — Slice of `(name, optional_value)` pairs from the CLI.
pub fn register_cli_defines(pp: &mut Preprocessor, defines: &[(String, Option<String>)]) {
    for (name, value) in defines {
        let val_str = match value {
            Some(v) => v.as_str(),
            None => "1",
        };

        let sym: Symbol = pp.interner.intern(name);
        let body = tokenize_cli_value(pp, val_str);

        // CLI defines override any existing predefined macro.
        let def = MacroDef::object_like(sym, body, Span::DUMMY);
        pp.macros.insert(sym, def);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Registration Sub-functions
// ═══════════════════════════════════════════════════════════════════════════

/// Registers the five dynamic predefined macros whose values are computed by
/// the macro expander at the point of expansion rather than from a static
/// replacement list.
///
/// | Macro          | Expander Behaviour                                        |
/// |----------------|----------------------------------------------------------|
/// | `__FILE__`     | Current file name as a string literal                    |
/// | `__LINE__`     | Current line number as an integer literal                 |
/// | `__DATE__`     | Compilation date as `"Mmm dd yyyy"` string literal       |
/// | `__TIME__`     | Compilation time as `"hh:mm:ss"` string literal          |
/// | `__COUNTER__`  | Unique integer, incrementing from 0 on each expansion    |
fn register_dynamic_macros(pp: &mut Preprocessor) {
    let dynamic_names: [&str; 5] = [
        "__FILE__",
        "__LINE__",
        "__DATE__",
        "__TIME__",
        "__COUNTER__",
    ];
    for name in &dynamic_names {
        define_dynamic(pp, name);
    }
}

/// Registers architecture-specific, platform, compiler-identification, and
/// type-size macros produced by `Target::predefined_macros()`.
///
/// This step covers the majority of non-dynamic predefined macros:
///
/// - **C standard**: `__STDC__`, `__STDC_VERSION__`, `__STDC_HOSTED__`
/// - **Platform**: `__linux__`, `__gnu_linux__`, `__ELF__`, `__unix__`, etc.
/// - **Architecture**: `__x86_64__`, `__aarch64__`, `__riscv`, `__i386__`, etc.
/// - **Compiler ID**: `__BCC__`, `__GNUC__`, `__GNUC_MINOR__`, `__GNUC_PATCHLEVEL__`
/// - **Type sizes**: `__SIZEOF_INT__`, `__SIZEOF_POINTER__`, `__CHAR_BIT__`, etc.
/// - **Byte order**: `__BYTE_ORDER__`, `__ORDER_LITTLE_ENDIAN__`, etc.
///
/// Macros already in the table (e.g., the dynamic macros from Phase 1) are
/// **not** overwritten — the loop skips any name that already has an entry.
fn register_target_macros(pp: &mut Preprocessor, target: &Target) {
    let macros = target.predefined_macros();
    for (name, value) in macros {
        let sym: Symbol = pp.interner.intern(name);
        // Preserve dynamic macros registered in Phase 1.
        if pp.macros.contains_key(&sym) {
            continue;
        }
        define_from_str_pair(pp, name, value);
    }
}

/// Registers C11 standard macros that are **not** returned by
/// `Target::predefined_macros()` but are required for full C11 conformance.
///
/// | Macro               | Value | Meaning                              |
/// |---------------------|-------|--------------------------------------|
/// | `__STDC_UTF_16__`   | 1     | `char16_t` values are UTF-16         |
/// | `__STDC_UTF_32__`   | 1     | `char32_t` values are UTF-32         |
/// | `__STDC_NO_VLA__`   | 1     | VLAs are not supported (C11 opt-out) |
///
/// Deliberately **not** defined (indicating support):
/// - `__STDC_NO_ATOMICS__` — we support `_Atomic` at storage level
/// - `__STDC_NO_COMPLEX__` — we support `_Complex`
/// - `__STDC_NO_THREADS__` — we support `_Thread_local` at storage level
fn register_c11_extra_macros(pp: &mut Preprocessor) {
    define_int_if_absent(pp, "__STDC_UTF_16__", 1);
    define_int_if_absent(pp, "__STDC_UTF_32__", 1);
    define_int_if_absent(pp, "__STDC_NO_VLA__", 1);
}

/// Registers type-size macros not included in `Target::predefined_macros()`.
///
/// The target's `predefined_macros()` method covers `__SIZEOF_INT__`,
/// `__SIZEOF_LONG__`, `__SIZEOF_POINTER__`, `__SIZEOF_SHORT__`, etc. but does
/// **not** include:
///
/// - `__SIZEOF_LONG_DOUBLE__` — varies by architecture (16/12/8 bytes)
/// - `__SIZEOF_WCHAR_T__` — always 4 on Linux (32-bit signed `int`)
/// - `__SIZEOF_WINT_T__` — matches `unsigned int` (4 bytes)
fn register_extra_type_size_macros(pp: &mut Preprocessor, target: &Target) {
    // __SIZEOF_LONG_DOUBLE__ — varies per architecture:
    //   x86-64: 16 (80-bit x87 padded), i686: 12, AArch64/RISC-V: 8
    let ld_size = target.long_double_size() as u128;
    define_int_if_absent(pp, "__SIZEOF_LONG_DOUBLE__", ld_size);

    // wchar_t on Linux is always 32-bit signed int.
    define_int_if_absent(pp, "__SIZEOF_WCHAR_T__", 4);

    // wint_t matches unsigned int on Linux.
    define_int_if_absent(pp, "__SIZEOF_WINT_T__", 4);

    // __POINTER_WIDTH__ — some code checks this instead of __SIZEOF_POINTER__
    let ptr_width_bits = (target.pointer_width() as u128) * 8;
    define_int_if_absent(pp, "__POINTER_WIDTH__", ptr_width_bits);
}

/// Registers integer type-limit macros for all fundamental C types.
///
/// These are required by `<limits.h>` and by kernel headers that perform
/// compile-time range checks. Values depend on the target data model
/// (LP64 vs ILP32) for `long` and pointer-width types.
fn register_type_limit_macros(pp: &mut Preprocessor, target: &Target) {
    // ── Fixed-width limits (same across all targets) ────────────────────
    define_int(pp, "__SCHAR_MAX__", 127, IntegerSuffix::None);
    define_int(pp, "__SHRT_MAX__", 32767, IntegerSuffix::None);
    define_int(pp, "__INT_MAX__", 2_147_483_647, IntegerSuffix::None);
    define_int(
        pp,
        "__LONG_LONG_MAX__",
        9_223_372_036_854_775_807,
        IntegerSuffix::LL,
    );

    // ── Data-model-dependent limits ─────────────────────────────────────
    // __LONG_MAX__ is 2^63-1 on LP64, 2^31-1 on ILP32.
    let long_max: u128 = if target.long_size() == 8 {
        9_223_372_036_854_775_807
    } else {
        2_147_483_647
    };
    define_int(pp, "__LONG_MAX__", long_max, IntegerSuffix::L);

    // __SIZE_MAX__ depends on pointer width (8→64-bit, 4→32-bit).
    let size_max: u128 = if target.pointer_width() == 8 {
        18_446_744_073_709_551_615 // 2^64 - 1
    } else {
        4_294_967_295 // 2^32 - 1
    };
    // GCC defines __SIZE_MAX__ without suffix; some headers expect this.
    // We use UL suffix for correctness (size_t is unsigned long on LP64).
    define_int(pp, "__SIZE_MAX__", size_max, IntegerSuffix::UL);

    // __PTRDIFF_MAX__ — signed counterpart of size_t width.
    let ptrdiff_max: u128 = if target.pointer_width() == 8 {
        9_223_372_036_854_775_807
    } else {
        2_147_483_647
    };
    define_int(pp, "__PTRDIFF_MAX__", ptrdiff_max, IntegerSuffix::L);

    // __WCHAR_MAX__ — wchar_t is 32-bit signed int on Linux.
    define_int(pp, "__WCHAR_MAX__", 2_147_483_647, IntegerSuffix::None);

    // __WINT_MAX__ — wint_t is unsigned int on Linux.
    define_int(pp, "__WINT_MAX__", 4_294_967_295, IntegerSuffix::U);

    // __INTMAX_MAX__ — intmax_t is long long (always 64-bit).
    define_int(
        pp,
        "__INTMAX_MAX__",
        9_223_372_036_854_775_807,
        IntegerSuffix::LL,
    );

    // __CHAR_UNSIGNED__ — char is signed on Linux x86/ARM/RISC-V.
    // Not defined → char is signed. Define as 0 to be explicit.
    define_int_if_absent(pp, "__CHAR_UNSIGNED__", 0);

    // __WCHAR_TYPE__ — underlying type of wchar_t
    define_ident(pp, "__WCHAR_TYPE__", "int");

    // __WINT_TYPE__ — underlying type of wint_t
    define_ident(pp, "__WINT_TYPE__", "unsigned int");

    // __INTMAX_TYPE__ — underlying type of intmax_t
    match target.data_model() {
        DataModel::LP64 => {
            define_ident(pp, "__INTMAX_TYPE__", "long int");
            define_ident(pp, "__UINTMAX_TYPE__", "long unsigned int");
            define_ident(pp, "__SIZE_TYPE__", "long unsigned int");
            define_ident(pp, "__PTRDIFF_TYPE__", "long int");
            define_ident(pp, "__INT64_TYPE__", "long int");
            define_ident(pp, "__UINT64_TYPE__", "long unsigned int");
        }
        DataModel::ILP32 => {
            define_ident(pp, "__INTMAX_TYPE__", "long long int");
            define_ident(pp, "__UINTMAX_TYPE__", "long long unsigned int");
            define_ident(pp, "__SIZE_TYPE__", "unsigned int");
            define_ident(pp, "__PTRDIFF_TYPE__", "int");
            define_ident(pp, "__INT64_TYPE__", "long long int");
            define_ident(pp, "__UINT64_TYPE__", "long long unsigned int");
        }
    }
}

/// Registers `__GCC_HAVE_SYNC_COMPARE_AND_SWAP_*` macros for widths 1, 2, 4,
/// and 8 bytes.
///
/// These macros indicate that the compiler supports `__sync_*` atomic builtin
/// functions for the specified byte widths. Linux kernel headers check these
/// macros to decide whether to use compiler atomic builtins or fall back to
/// architecture-specific assembly.
///
/// All four BCC target architectures support compare-and-swap at 1-, 2-, 4-,
/// and 8-byte widths:
///
/// | Target     | 1 | 2 | 4 | 8 | Mechanism           |
/// |------------|---|---|---|---|---------------------|
/// | x86-64     | ✓ | ✓ | ✓ | ✓ | LOCK CMPXCHG[8B/16B] |
/// | i686       | ✓ | ✓ | ✓ | ✓ | LOCK CMPXCHG, CMPXCHG8B |
/// | AArch64    | ✓ | ✓ | ✓ | ✓ | LDXR/STXR, LDXP/STXP |
/// | RISC-V 64  | ✓ | ✓ | ✓ | ✓ | LR/SC instructions |
fn register_gcc_sync_macros(pp: &mut Preprocessor) {
    define_int(
        pp,
        "__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1",
        1,
        IntegerSuffix::None,
    );
    define_int(
        pp,
        "__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2",
        1,
        IntegerSuffix::None,
    );
    define_int(
        pp,
        "__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4",
        1,
        IntegerSuffix::None,
    );
    define_int(
        pp,
        "__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8",
        1,
        IntegerSuffix::None,
    );
}

/// Registers compiler version identification macros.
///
/// - `__BCC_VERSION__` — BCC version as a string literal (e.g., `"1.0.0"`).
/// - `__VERSION__` — GCC-compatible version string for code that inspects
///   the compiler's self-reported version (e.g., `"BCC 1.0.0"`).
fn register_compiler_version_macros(pp: &mut Preprocessor) {
    // __BCC_VERSION__ — our own version string.
    define_str(pp, "__BCC_VERSION__", BCC_VERSION.as_bytes());

    // __VERSION__ — GCC-compatible composite version string.
    let version_display = format!("BCC {}", BCC_VERSION);
    define_str(pp, "__VERSION__", version_display.as_bytes());
}

/// Registers GCC built-in type compatibility macros.
///
/// `__float128` is a GCC built-in type used in system headers (e.g., in
/// `stddef.h`'s `max_align_t` definition when `__i386__` is defined).
/// Since we do not implement a native 128-bit floating-point type, we
/// map `__float128` to `long double` for source-level compatibility.
fn register_gcc_type_compat_macros(pp: &mut Preprocessor) {
    // Map __float128 → long double (closest approximation we support).
    define_from_str_pair(pp, "__float128", "long double");
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI Value Tokenisation
// ═══════════════════════════════════════════════════════════════════════════

/// Tokenises a `-D` value string into a `Vec<Token>` for use as a macro body.
///
/// The function attempts to parse the value as:
/// 1. An integer with an optional C suffix (e.g., `"42"`, `"201112L"`)
/// 2. A plain identifier (fallback)
///
/// An empty value string produces an empty token vector (flag-style define).
fn tokenize_cli_value(pp: &mut Preprocessor, value: &str) -> Vec<Token> {
    if value.is_empty() {
        return Vec::new();
    }

    // Try integer with suffix first (handles "42L", "0xFFu", etc.).
    if let Some((n, suffix)) = parse_integer_with_suffix(value) {
        return vec![int_token(n, suffix)];
    }

    // Try plain unsigned integer.
    if let Ok(n) = value.parse::<u128>() {
        return vec![int_token(n, IntegerSuffix::None)];
    }

    // Fallback: intern the value string as an identifier.
    let sym: Symbol = pp.interner.intern(value);
    vec![Token::new(TokenKind::Identifier(sym), Span::DUMMY)]
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_integer_with_suffix ────────────────────────────────────────

    #[test]
    fn parse_plain_integer_no_suffix_returns_none() {
        // No suffix → parse_integer_with_suffix returns None (caller
        // should fall back to plain u128::parse).
        assert!(parse_integer_with_suffix("42").is_none());
        assert!(parse_integer_with_suffix("0").is_none());
    }

    #[test]
    fn parse_suffix_l() {
        let (val, suf) = parse_integer_with_suffix("201112L").unwrap();
        assert_eq!(val, 201112);
        assert_eq!(suf, IntegerSuffix::L);
    }

    #[test]
    fn parse_suffix_l_lowercase() {
        let (val, suf) = parse_integer_with_suffix("100l").unwrap();
        assert_eq!(val, 100);
        assert_eq!(suf, IntegerSuffix::L);
    }

    #[test]
    fn parse_suffix_u() {
        let (val, suf) = parse_integer_with_suffix("255U").unwrap();
        assert_eq!(val, 255);
        assert_eq!(suf, IntegerSuffix::U);
    }

    #[test]
    fn parse_suffix_ul() {
        let (val, suf) = parse_integer_with_suffix("1000UL").unwrap();
        assert_eq!(val, 1000);
        assert_eq!(suf, IntegerSuffix::UL);
    }

    #[test]
    fn parse_suffix_lu() {
        // "LU" is an alternate form of "UL".
        let (val, suf) = parse_integer_with_suffix("500LU").unwrap();
        assert_eq!(val, 500);
        assert_eq!(suf, IntegerSuffix::UL);
    }

    #[test]
    fn parse_suffix_ll() {
        let (val, suf) = parse_integer_with_suffix("9223372036854775807LL").unwrap();
        assert_eq!(val, 9_223_372_036_854_775_807);
        assert_eq!(suf, IntegerSuffix::LL);
    }

    #[test]
    fn parse_suffix_ull() {
        let (val, suf) = parse_integer_with_suffix("18446744073709551615ULL").unwrap();
        assert_eq!(val, 18_446_744_073_709_551_615);
        assert_eq!(suf, IntegerSuffix::ULL);
    }

    #[test]
    fn parse_suffix_llu() {
        let (val, suf) = parse_integer_with_suffix("100LLU").unwrap();
        assert_eq!(val, 100);
        assert_eq!(suf, IntegerSuffix::ULL);
    }

    #[test]
    fn parse_hex_with_suffix() {
        let (val, suf) = parse_integer_with_suffix("0xFFu").unwrap();
        assert_eq!(val, 255);
        assert_eq!(suf, IntegerSuffix::U);
    }

    #[test]
    fn parse_empty_string_returns_none() {
        assert!(parse_integer_with_suffix("").is_none());
    }

    #[test]
    fn parse_suffix_only_returns_none() {
        // "L" with no numeric part → None.
        assert!(parse_integer_with_suffix("L").is_none());
        assert!(parse_integer_with_suffix("ULL").is_none());
    }

    #[test]
    fn parse_non_numeric_returns_none() {
        // "fooL" → numeric part "foo" fails to parse → None.
        assert!(parse_integer_with_suffix("fooL").is_none());
    }

    // ── int_token / str_token helpers ───────────────────────────────────

    #[test]
    fn int_token_creates_correct_kind() {
        let tok = int_token(42, IntegerSuffix::None);
        assert_eq!(
            tok.kind,
            TokenKind::IntegerLiteral {
                value: 42,
                suffix: IntegerSuffix::None
            }
        );
        assert_eq!(tok.span, Span::DUMMY);
    }

    #[test]
    fn int_token_with_long_suffix() {
        let tok = int_token(201112, IntegerSuffix::L);
        assert_eq!(
            tok.kind,
            TokenKind::IntegerLiteral {
                value: 201112,
                suffix: IntegerSuffix::L
            }
        );
    }

    #[test]
    fn str_token_creates_correct_kind() {
        let tok = str_token(b"hello");
        match &tok.kind {
            TokenKind::StringLiteral { value, prefix } => {
                assert_eq!(value, b"hello");
                assert_eq!(*prefix, StringPrefix::None);
            }
            other => panic!("expected StringLiteral, got {:?}", other),
        }
        assert_eq!(tok.span, Span::DUMMY);
    }
}
