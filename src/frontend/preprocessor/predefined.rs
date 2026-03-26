//! Predefined macros for the BCC C compiler.
//!
//! This module implements registration and identification of compiler-predefined
//! macros as defined by C11 §6.10.8 and common GCC extensions.
//!
//! ## Standard C11 Predefined Macros
//!
//! - `__STDC__` — always `1`
//! - `__STDC_VERSION__` — `201112L` (C11)
//! - `__STDC_HOSTED__` — always `1`
//! - `__FILE__` — current source filename (context-dependent magic macro)
//! - `__LINE__` — current source line number (context-dependent magic macro)
//! - `__DATE__` — compilation date, captured once at startup ("Mmm dd yyyy")
//! - `__TIME__` — compilation time, captured once at startup ("hh:mm:ss")
//!
//! ## GCC Extension Predefined Macros
//!
//! - `__COUNTER__` — auto-incrementing integer starting at 0
//! - `__GNUC__`, `__GNUC_MINOR__`, `__GNUC_PATCHLEVEL__` — GCC compatibility
//! - `__SIZEOF_*__` — type size macros (target-dependent)
//! - `__*_TYPE__` — underlying type macros (target-dependent)
//! - Architecture-specific macros via [`Target::predefined_macros()`]
//!
//! ## Magic Macros
//!
//! Some macros (`__FILE__`, `__LINE__`, `__COUNTER__`, `__DATE__`, `__TIME__`)
//! require context-dependent or dynamically-computed expansion.  These are
//! identified via [`is_magic_macro()`] and handled specially by the macro
//! expander during preprocessing rather than undergoing normal object-like
//! macro expansion.
//!
//! ## Zero-Dependency Mandate
//!
//! All date/time computation uses manual Gregorian calendar arithmetic from
//! [`std::time::SystemTime`] — no external date/time crates are permitted.

use super::{MacroDef, MacroKind, PPToken, PPTokenKind};
use crate::common::diagnostics::Span;
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// MagicMacro — context-dependent macro identification
// ---------------------------------------------------------------------------

/// Predefined macros that require context-dependent or dynamically-computed
/// expansion, rather than simple token replacement.
///
/// The macro expander checks [`is_magic_macro()`] before performing normal
/// macro expansion.  For magic macros, the expander generates the appropriate
/// token(s) dynamically based on the current preprocessing context.
///
/// # Variants
///
/// | Variant   | Macro         | Expansion                                |
/// |-----------|---------------|------------------------------------------|
/// | `File`    | `__FILE__`    | Current filename as string literal       |
/// | `Line`    | `__LINE__`    | Current line number as pp-number         |
/// | `Counter` | `__COUNTER__` | Auto-incrementing integer (0, 1, 2, …)   |
/// | `Date`    | `__DATE__`    | Compilation date `"Mmm dd yyyy"`         |
/// | `Time`    | `__TIME__`    | Compilation time `"hh:mm:ss"`            |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MagicMacro {
    /// `__FILE__` — expands to the current source filename as a string literal.
    /// The value changes depending on which file is currently being processed,
    /// including `#line` directive remapping.
    File,
    /// `__LINE__` — expands to the current source line number as a pp-number.
    /// The value changes on every line of every source file.
    Line,
    /// `__COUNTER__` — GCC extension.  Expands to an auto-incrementing integer
    /// starting at 0.  Each expansion increments the global counter by one.
    Counter,
    /// `__DATE__` — expands to the compilation date as a string literal in
    /// `"Mmm dd yyyy"` format (e.g., `"Mar  2 2026"`).  Captured once at
    /// compilation start per C11 §6.10.8 — remains constant throughout.
    Date,
    /// `__TIME__` — expands to the compilation time as a string literal in
    /// `"hh:mm:ss"` format (e.g., `"14:30:05"`).  Captured once at
    /// compilation start per C11 §6.10.8 — remains constant throughout.
    Time,
}

// ---------------------------------------------------------------------------
// is_magic_macro — magic macro detection
// ---------------------------------------------------------------------------

/// Check whether a macro name refers to a magic (context-dependent) macro.
///
/// Returns `Some(MagicMacro)` if the name matches one of the five magic macros,
/// or `None` if it is a regular (non-magic) macro or not a macro at all.
///
/// The macro expander should call this function before attempting normal
/// object-like or function-like macro expansion.  If the result is `Some`,
/// the expander must generate the appropriate token(s) dynamically rather
/// than using the placeholder replacement list stored in the macro definition.
///
/// # Examples
///
/// ```ignore
/// use bcc::frontend::preprocessor::predefined::{is_magic_macro, MagicMacro};
///
/// assert_eq!(is_magic_macro("__FILE__"), Some(MagicMacro::File));
/// assert_eq!(is_magic_macro("__LINE__"), Some(MagicMacro::Line));
/// assert_eq!(is_magic_macro("__COUNTER__"), Some(MagicMacro::Counter));
/// assert_eq!(is_magic_macro("__DATE__"), Some(MagicMacro::Date));
/// assert_eq!(is_magic_macro("__TIME__"), Some(MagicMacro::Time));
/// assert_eq!(is_magic_macro("__STDC__"), None);
/// assert_eq!(is_magic_macro("FOO"), None);
/// ```
pub fn is_magic_macro(name: &str) -> Option<MagicMacro> {
    match name {
        "__FILE__" => Some(MagicMacro::File),
        "__LINE__" => Some(MagicMacro::Line),
        "__COUNTER__" => Some(MagicMacro::Counter),
        "__DATE__" => Some(MagicMacro::Date),
        "__TIME__" => Some(MagicMacro::Time),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Compilation timestamp capture
// ---------------------------------------------------------------------------

/// Capture the compilation date and time, formatted for `__DATE__` and
/// `__TIME__`.
///
/// This function should be called exactly once at compilation start.  Per C11
/// §6.10.8, `__DATE__` and `__TIME__` must remain constant throughout the
/// entire translation unit.
///
/// # Returns
///
/// A tuple `(date_string, time_string)` where:
///
/// - `date_string` is in `"Mmm dd yyyy"` format (e.g., `"Mar  2 2026"`)
///   - Month is a 3-letter abbreviation (Jan, Feb, Mar, …, Dec)
///   - Day is right-justified with a space pad when < 10 (e.g., `" 2"`)
///   - Year is always 4 digits
///   - Total length: always 11 characters
/// - `time_string` is in `"hh:mm:ss"` format (e.g., `"14:30:05"`)
///   - 24-hour format, zero-padded
///   - Total length: always 8 characters
///
/// # Implementation Notes
///
/// Uses manual calendar arithmetic from Unix epoch seconds obtained via
/// [`std::time::SystemTime::now()`] and [`std::time::UNIX_EPOCH`].  No
/// external date/time crates are used — this complies with the zero-
/// dependency mandate.  The Gregorian calendar conversion uses Howard
/// Hinnant's `civil_from_days` algorithm.
pub fn capture_compilation_timestamp() -> (String, String) {
    let total_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (year, month, day, hour, min, sec) = unix_timestamp_to_components(total_secs);

    // C11 §6.10.8 __DATE__ format: "Mmm dd yyyy"
    // Day is right-justified, space-padded if single digit.
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_idx = if (1..=12).contains(&month) {
        (month - 1) as usize
    } else {
        0 // defensive fallback — should never occur with valid timestamps
    };
    let month_name = MONTHS[month_idx];

    let day_str = if day < 10 {
        format!(" {}", day)
    } else {
        format!("{}", day)
    };
    let date_string = format!("{} {} {}", month_name, day_str, year);

    // C11 §6.10.8 __TIME__ format: "hh:mm:ss"
    let time_string = format!("{:02}:{:02}:{:02}", hour, min, sec);

    (date_string, time_string)
}

// ---------------------------------------------------------------------------
// Calendar arithmetic — Unix timestamp to (year, month, day, hour, min, sec)
// ---------------------------------------------------------------------------

/// Convert a Unix timestamp (seconds since 1970-01-01 00:00:00 UTC) into
/// Gregorian calendar date and time-of-day components.
///
/// Returns `(year, month, day, hour, minute, second)` where:
/// - `year`  : full calendar year (e.g., 2026)
/// - `month` : 1–12
/// - `day`   : 1–31
/// - `hour`  : 0–23
/// - `minute`: 0–59
/// - `second`: 0–59
///
/// # Algorithm
///
/// Uses Howard Hinnant's `civil_from_days` algorithm for efficient and
/// correct Gregorian calendar conversion.  The algorithm shifts the epoch
/// to March 1 of year 0, making month/day extraction purely arithmetic
/// (no lookup tables for cumulative month lengths).
fn unix_timestamp_to_components(total_secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    // ---- Time-of-day from remainder after full days ----
    let day_secs = (total_secs % 86_400) as u32;
    let hour = day_secs / 3600;
    let min = (day_secs % 3600) / 60;
    let sec = day_secs % 60;

    // ---- Total days since Unix epoch (1970-01-01) ----
    let days_since_epoch = (total_secs / 86_400) as i64;

    // Shift epoch from 1970-01-01 to 0000-03-01 (proleptic Gregorian).
    // 719 468 = number of days from 0000-03-01 to 1970-01-01.
    let z = days_since_epoch + 719_468;

    // Era: a complete 400-year Gregorian cycle (146 097 days each).
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };

    // Day-of-era: day within the current 400-year cycle [0, 146 096].
    let doe = (z - era * 146_097) as u32;

    // Year-of-era [0, 399].
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;

    // Absolute March-based year (March 1 is day 0 of the year).
    let y = (yoe as i64) + era * 400;

    // Day-of-year [0, 365] (March 1 = day 0).
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);

    // Month-from-March [0, 11]: March=0, Apr=1, …, Feb=11.
    let mp = (5 * doy + 2) / 153;

    // Day-of-month [1, 31].
    let d = doy - (153 * mp + 2) / 5 + 1;

    // Convert March-based month to calendar month [1, 12].
    let m = if mp < 10 { mp + 3 } else { mp - 9 };

    // Adjust year for January and February (months 11-12 in the March-based
    // scheme belong to the next calendar year).
    let y = if m <= 2 { y + 1 } else { y };

    (y as i32, m, d, hour, min, sec)
}

// ---------------------------------------------------------------------------
// Predefined macro registration
// ---------------------------------------------------------------------------

/// Register all compiler-predefined macros for the given target architecture.
///
/// This function is called once during `Preprocessor::new()` to populate the
/// initial macro definition table before any user source is processed.  It
/// registers macros in the following categories:
///
/// 1. **Standard C11 macros** (§6.10.8) — `__STDC__`, `__STDC_VERSION__`,
///    `__STDC_HOSTED__`
/// 2. **Magic macros** — `__FILE__`, `__LINE__`, `__COUNTER__`, `__DATE__`,
///    `__TIME__` (registered with placeholder replacements; the macro expander
///    generates actual values dynamically)
/// 3. **GCC compatibility macros** — `__GNUC__`, `__GNUC_MINOR__`,
///    `__GNUC_PATCHLEVEL__`, `__VERSION__`
/// 4. **Byte order macros** — `__BYTE_ORDER__`, `__ORDER_LITTLE_ENDIAN__`, etc.
/// 5. **sizeof macros** — `__SIZEOF_INT__`, `__SIZEOF_LONG__`, etc.
///    (target-dependent values from [`Target`] methods)
/// 6. **Numeric limit macros** — `__INT_MAX__`, `__LONG_MAX__`, etc.
/// 7. **Type definition macros** — `__SIZE_TYPE__`, `__PTRDIFF_TYPE__`, etc.
/// 8. **Architecture-specific macros** — from [`Target::predefined_macros()`]
pub fn register_predefined_macros(macro_defs: &mut FxHashMap<String, MacroDef>, target: &Target) {
    // -------------------------------------------------------------------
    // 1. Capture compilation timestamp (must happen exactly once)
    // -------------------------------------------------------------------
    let (date_str, time_str) = capture_compilation_timestamp();

    // -------------------------------------------------------------------
    // 2. Standard C11 macros (§6.10.8)
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__STDC__", "1");
    register_object_macro(macro_defs, "__STDC_VERSION__", "201112L");
    register_object_macro(macro_defs, "__STDC_HOSTED__", "1");

    // -------------------------------------------------------------------
    // 3. Magic macros — placeholders for the macro expander.
    //    Registered so that `#ifdef __FILE__` etc. evaluate to true.
    //    The macro expander overrides their replacement at expansion time.
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__FILE__", "\"<built-in>\"");
    register_object_macro(macro_defs, "__LINE__", "0");
    register_object_macro(macro_defs, "__COUNTER__", "0");

    // __DATE__ and __TIME__ — captured values (constant for entire TU).
    let date_literal = format!("\"{}\"", date_str);
    register_object_macro(macro_defs, "__DATE__", &date_literal);
    let time_literal = format!("\"{}\"", time_str);
    register_object_macro(macro_defs, "__TIME__", &time_literal);

    // -------------------------------------------------------------------
    // 4. GCC compatibility version macros
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__GNUC__", "12");
    register_object_macro(macro_defs, "__GNUC_MINOR__", "2");
    register_object_macro(macro_defs, "__GNUC_PATCHLEVEL__", "0");
    register_object_macro(macro_defs, "__VERSION__", "\"BCC 0.1.0 (GCC compatible)\"");
    register_object_macro(macro_defs, "__GCC_HAVE_DWARF2_CFI_ASM", "1");
    register_object_macro(macro_defs, "__GNUC_STDC_INLINE__", "1");

    // -------------------------------------------------------------------
    // 4b. Atomic memory order constants (GCC __atomic builtins)
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__ATOMIC_RELAXED", "0");
    register_object_macro(macro_defs, "__ATOMIC_CONSUME", "1");
    register_object_macro(macro_defs, "__ATOMIC_ACQUIRE", "2");
    register_object_macro(macro_defs, "__ATOMIC_RELEASE", "3");
    register_object_macro(macro_defs, "__ATOMIC_ACQ_REL", "4");
    register_object_macro(macro_defs, "__ATOMIC_SEQ_CST", "5");

    // -------------------------------------------------------------------
    // 5. Byte order macros
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__ORDER_LITTLE_ENDIAN__", "1234");
    register_object_macro(macro_defs, "__ORDER_BIG_ENDIAN__", "4321");
    register_object_macro(macro_defs, "__ORDER_PDP_ENDIAN__", "3412");
    // All four BCC targets are little-endian.
    register_object_macro(macro_defs, "__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__");

    // -------------------------------------------------------------------
    // 6. sizeof macros — fixed across all targets
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__SIZEOF_SHORT__", "2");
    register_object_macro(macro_defs, "__SIZEOF_INT__", "4");
    register_object_macro(macro_defs, "__SIZEOF_FLOAT__", "4");
    register_object_macro(macro_defs, "__SIZEOF_DOUBLE__", "8");
    register_object_macro(macro_defs, "__SIZEOF_LONG_LONG__", "8");
    register_object_macro(macro_defs, "__SIZEOF_WCHAR_T__", "4");
    register_object_macro(macro_defs, "__SIZEOF_WINT_T__", "4");

    // sizeof macros — target-dependent via Target methods
    let long_sz_str = target.long_size().to_string();
    register_object_macro(macro_defs, "__SIZEOF_LONG__", &long_sz_str);

    let ptr_w_str = target.pointer_width().to_string();
    register_object_macro(macro_defs, "__SIZEOF_POINTER__", &ptr_w_str);
    register_object_macro(macro_defs, "__SIZEOF_SIZE_T__", &ptr_w_str);
    register_object_macro(macro_defs, "__SIZEOF_PTRDIFF_T__", &ptr_w_str);

    let ld_sz_str = target.long_double_size().to_string();
    register_object_macro(macro_defs, "__SIZEOF_LONG_DOUBLE__", &ld_sz_str);

    // __int128 support — do NOT define __SIZEOF_INT128__ because BCC does not
    // yet fully implement 128-bit integer arithmetic in code generation.
    // Defining it would cause programs to use __int128 code paths that fail at
    // runtime.  Un-comment the line below once 128-bit codegen is implemented.
    // if target.is_64bit() {
    //     register_object_macro(macro_defs, "__SIZEOF_INT128__", "16");
    // }

    // -------------------------------------------------------------------
    // 7. Numeric limit macros
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__CHAR_BIT__", "8");
    register_object_macro(macro_defs, "__SCHAR_MAX__", "127");
    register_object_macro(macro_defs, "__SHRT_MAX__", "32767");
    register_object_macro(macro_defs, "__INT_MAX__", "2147483647");
    register_object_macro(macro_defs, "__LONG_LONG_MAX__", "9223372036854775807LL");

    // __LONG_MAX__ is target-dependent: 32-bit on i686, 64-bit on LP64.
    if target.is_64bit() {
        register_object_macro(macro_defs, "__LONG_MAX__", "9223372036854775807L");
    } else {
        register_object_macro(macro_defs, "__LONG_MAX__", "2147483647L");
    }

    // -------------------------------------------------------------------
    // 7b. Standard C limit macros (provided directly since BCC does not
    //     support #include_next and the system <limits.h> chain relies on it)
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "CHAR_BIT", "8");
    register_object_macro(macro_defs, "SCHAR_MIN", "(-128)");
    register_object_macro(macro_defs, "SCHAR_MAX", "127");
    register_object_macro(macro_defs, "UCHAR_MAX", "255");
    register_object_macro(macro_defs, "CHAR_MIN", "(-128)");
    register_object_macro(macro_defs, "CHAR_MAX", "127");
    register_object_macro(macro_defs, "SHRT_MIN", "(-32768)");
    register_object_macro(macro_defs, "SHRT_MAX", "32767");
    register_object_macro(macro_defs, "USHRT_MAX", "65535");
    register_object_macro(macro_defs, "INT_MIN", "(-2147483647 - 1)");
    register_object_macro(macro_defs, "INT_MAX", "2147483647");
    register_object_macro(macro_defs, "UINT_MAX", "4294967295U");
    register_object_macro(macro_defs, "LLONG_MIN", "(-9223372036854775807LL - 1)");
    register_object_macro(macro_defs, "LLONG_MAX", "9223372036854775807LL");
    register_object_macro(macro_defs, "ULLONG_MAX", "18446744073709551615ULL");
    if target.is_64bit() {
        register_object_macro(macro_defs, "LONG_MIN", "(-9223372036854775807L - 1)");
        register_object_macro(macro_defs, "LONG_MAX", "9223372036854775807L");
        register_object_macro(macro_defs, "ULONG_MAX", "18446744073709551615UL");
    } else {
        register_object_macro(macro_defs, "LONG_MIN", "(-2147483647L - 1)");
        register_object_macro(macro_defs, "LONG_MAX", "2147483647L");
        register_object_macro(macro_defs, "ULONG_MAX", "4294967295UL");
    }

    // -------------------------------------------------------------------
    // 8. Type definition macros — some target-dependent
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__CHAR_UNSIGNED__", "0");
    register_object_macro(macro_defs, "__WCHAR_TYPE__", "int");
    register_object_macro(macro_defs, "__WINT_TYPE__", "unsigned int");
    // C11 char16_t/char32_t underlying types (used by <stdatomic.h>)
    register_object_macro(macro_defs, "__CHAR16_TYPE__", "unsigned short");
    register_object_macro(macro_defs, "__CHAR32_TYPE__", "unsigned int");
    register_object_macro(macro_defs, "__INT8_TYPE__", "signed char");
    register_object_macro(macro_defs, "__INT16_TYPE__", "short");
    register_object_macro(macro_defs, "__INT32_TYPE__", "int");
    register_object_macro(macro_defs, "__UINT8_TYPE__", "unsigned char");
    register_object_macro(macro_defs, "__UINT16_TYPE__", "unsigned short");
    register_object_macro(macro_defs, "__UINT32_TYPE__", "unsigned int");

    // Target-dependent type macros: LP64 uses `long`, ILP32 uses wider types.
    if target.is_64bit() {
        register_object_macro(macro_defs, "__INT64_TYPE__", "long");
        register_object_macro(macro_defs, "__UINT64_TYPE__", "unsigned long");
        register_object_macro(macro_defs, "__INTMAX_TYPE__", "long");
        register_object_macro(macro_defs, "__UINTMAX_TYPE__", "unsigned long");
        register_object_macro(macro_defs, "__SIZE_TYPE__", "unsigned long");
        register_object_macro(macro_defs, "__PTRDIFF_TYPE__", "long");
        register_object_macro(macro_defs, "__INTPTR_TYPE__", "long");
        register_object_macro(macro_defs, "__UINTPTR_TYPE__", "unsigned long");
    } else {
        register_object_macro(macro_defs, "__INT64_TYPE__", "long long");
        register_object_macro(macro_defs, "__UINT64_TYPE__", "unsigned long long");
        register_object_macro(macro_defs, "__INTMAX_TYPE__", "long long");
        register_object_macro(macro_defs, "__UINTMAX_TYPE__", "unsigned long long");
        register_object_macro(macro_defs, "__SIZE_TYPE__", "unsigned int");
        register_object_macro(macro_defs, "__PTRDIFF_TYPE__", "int");
        register_object_macro(macro_defs, "__INTPTR_TYPE__", "int");
        register_object_macro(macro_defs, "__UINTPTR_TYPE__", "unsigned int");
    }

    // -------------------------------------------------------------------
    // 8b. GCC __INT_LEASTn_TYPE__ / __UINT_LEASTn_TYPE__ /
    //     __INT_FASTn_TYPE__ / __UINT_FASTn_TYPE__ predefined macros.
    //     These are required by <stdint.h> and many GCC torture tests.
    // -------------------------------------------------------------------
    register_object_macro(macro_defs, "__INT_LEAST8_TYPE__", "signed char");
    register_object_macro(macro_defs, "__UINT_LEAST8_TYPE__", "unsigned char");
    register_object_macro(macro_defs, "__INT_LEAST16_TYPE__", "short");
    register_object_macro(macro_defs, "__UINT_LEAST16_TYPE__", "unsigned short");
    register_object_macro(macro_defs, "__INT_LEAST32_TYPE__", "int");
    register_object_macro(macro_defs, "__UINT_LEAST32_TYPE__", "unsigned int");
    register_object_macro(macro_defs, "__INT_FAST8_TYPE__", "signed char");
    register_object_macro(macro_defs, "__UINT_FAST8_TYPE__", "unsigned char");

    if target.is_64bit() {
        register_object_macro(macro_defs, "__INT_LEAST64_TYPE__", "long");
        register_object_macro(macro_defs, "__UINT_LEAST64_TYPE__", "unsigned long");
        register_object_macro(macro_defs, "__INT_FAST16_TYPE__", "long");
        register_object_macro(macro_defs, "__UINT_FAST16_TYPE__", "unsigned long");
        register_object_macro(macro_defs, "__INT_FAST32_TYPE__", "long");
        register_object_macro(macro_defs, "__UINT_FAST32_TYPE__", "unsigned long");
        register_object_macro(macro_defs, "__INT_FAST64_TYPE__", "long");
        register_object_macro(macro_defs, "__UINT_FAST64_TYPE__", "unsigned long");
    } else {
        register_object_macro(macro_defs, "__INT_LEAST64_TYPE__", "long long");
        register_object_macro(macro_defs, "__UINT_LEAST64_TYPE__", "unsigned long long");
        register_object_macro(macro_defs, "__INT_FAST16_TYPE__", "int");
        register_object_macro(macro_defs, "__UINT_FAST16_TYPE__", "unsigned int");
        register_object_macro(macro_defs, "__INT_FAST32_TYPE__", "int");
        register_object_macro(macro_defs, "__UINT_FAST32_TYPE__", "unsigned int");
        register_object_macro(macro_defs, "__INT_FAST64_TYPE__", "long long");
        register_object_macro(macro_defs, "__UINT_FAST64_TYPE__", "unsigned long long");
    }

    // GCC __SIG_ATOMIC_TYPE__ — used by <signal.h>/<stdatomic.h>
    register_object_macro(macro_defs, "__SIG_ATOMIC_TYPE__", "int");

    // -------------------------------------------------------------------
    // 9. Architecture-specific macros from Target::predefined_macros()
    //    Registers __x86_64__, __i386__, __aarch64__, __riscv, plus
    //    __linux__, __unix__, __ELF__ and all arch-specific feature macros.
    // -------------------------------------------------------------------
    for (name, value) in target.predefined_macros() {
        register_object_macro(macro_defs, name, value);
    }

    // -------------------------------------------------------------------
    // 10. GCC-compatible floating-point constant macros
    //     Required by system math headers and real-world C projects.
    //     Values match IEEE 754 double-precision and single-precision.
    // -------------------------------------------------------------------
    // Double-precision (64-bit) constants
    register_object_macro(macro_defs, "__DBL_EPSILON__", "2.2204460492503131e-16");
    register_object_macro(macro_defs, "__DBL_MIN__", "2.2250738585072014e-308");
    register_object_macro(macro_defs, "__DBL_MAX__", "1.7976931348623157e+308");
    register_object_macro(macro_defs, "__DBL_DIG__", "15");
    register_object_macro(macro_defs, "__DBL_MANT_DIG__", "53");
    register_object_macro(macro_defs, "__DBL_MIN_EXP__", "(-1021)");
    register_object_macro(macro_defs, "__DBL_MAX_EXP__", "1024");
    register_object_macro(macro_defs, "__DBL_MIN_10_EXP__", "(-307)");
    register_object_macro(macro_defs, "__DBL_MAX_10_EXP__", "308");
    register_object_macro(macro_defs, "__DBL_HAS_INFINITY__", "1");
    register_object_macro(macro_defs, "__DBL_HAS_QUIET_NAN__", "1");
    register_object_macro(macro_defs, "__DBL_DENORM_MIN__", "4.9406564584124654e-324");
    // Single-precision (32-bit) constants
    register_object_macro(macro_defs, "__FLT_EPSILON__", "1.1920928955078125e-07F");
    register_object_macro(
        macro_defs,
        "__FLT_MIN__",
        "1.17549435082228750796873653722225e-38F",
    );
    register_object_macro(
        macro_defs,
        "__FLT_MAX__",
        "3.40282346638528859811704183484517e+38F",
    );
    register_object_macro(macro_defs, "__FLT_DIG__", "6");
    register_object_macro(macro_defs, "__FLT_MANT_DIG__", "24");
    register_object_macro(macro_defs, "__FLT_MIN_EXP__", "(-125)");
    register_object_macro(macro_defs, "__FLT_MAX_EXP__", "128");
    register_object_macro(macro_defs, "__FLT_MIN_10_EXP__", "(-37)");
    register_object_macro(macro_defs, "__FLT_MAX_10_EXP__", "38");
    register_object_macro(macro_defs, "__FLT_HAS_INFINITY__", "1");
    register_object_macro(macro_defs, "__FLT_HAS_QUIET_NAN__", "1");
    register_object_macro(
        macro_defs,
        "__FLT_DENORM_MIN__",
        "1.40129846432481707092372958328992e-45F",
    );
    // Long-double constants (80-bit extended / 128-bit on some arches)
    register_object_macro(
        macro_defs,
        "__LDBL_EPSILON__",
        "1.08420217248550443401e-19L",
    );
    register_object_macro(macro_defs, "__LDBL_MIN__", "3.36210314311209350626e-4932L");
    register_object_macro(macro_defs, "__LDBL_MAX__", "1.18973149535723176502e+4932L");
    register_object_macro(macro_defs, "__LDBL_DIG__", "18");
    register_object_macro(macro_defs, "__LDBL_MANT_DIG__", "64");
    register_object_macro(macro_defs, "__LDBL_MIN_EXP__", "(-16381)");
    register_object_macro(macro_defs, "__LDBL_MAX_EXP__", "16384");
    // Decimal floating-point radix
    register_object_macro(macro_defs, "__FLT_RADIX__", "2");
    // GCC misc float macros
    register_object_macro(macro_defs, "__GCC_IEC_559__", "2");
    register_object_macro(macro_defs, "__DECIMAL_DIG__", "21");
}

// ---------------------------------------------------------------------------
// Helper: register an object-like predefined macro
// ---------------------------------------------------------------------------

/// Register an object-like predefined macro with the given replacement value.
///
/// The `value` string is tokenized into preprocessing tokens using
/// [`tokenize_value()`].  Most predefined macros have single-token values
/// (a pp-number or identifier), but multi-token values such as
/// `"unsigned long"` are also supported and produce separate tokens with
/// whitespace tokens between words.
///
/// All macros registered by this function are marked as predefined
/// (`is_predefined = true`) and carry a [`Span::dummy()`] since they have
/// no corresponding source file location.
fn register_object_macro(macro_defs: &mut FxHashMap<String, MacroDef>, name: &str, value: &str) {
    let tokens = tokenize_value(value);
    macro_defs.insert(
        name.to_string(),
        MacroDef {
            name: name.to_string(),
            kind: MacroKind::ObjectLike,
            replacement: tokens,
            is_predefined: true,
            definition_span: Span::dummy(),
        },
    );
}

// ---------------------------------------------------------------------------
// Helper: tokenize a macro replacement value string
// ---------------------------------------------------------------------------

/// Tokenize a macro replacement value string into a vector of preprocessing
/// tokens.
///
/// Handles the following value patterns:
///
/// - **Number literals**: `"1"`, `"201112L"`, `"2147483647"`,
///   `"9223372036854775807LL"`
/// - **Single identifiers**: `"int"`, `"long"`, `"__ORDER_LITTLE_ENDIAN__"`
/// - **Multi-word type names**: `"unsigned long"`, `"signed char"`
/// - **String literals**: `"\"BCC 0.1.0 (GCC compatible)\""` (with
///   surrounding quotes)
/// - **Punctuation**: any non-alphanumeric, non-whitespace, non-quote
///   character produces a [`PPTokenKind::Punctuator`] token
///
/// All tokens are created with [`Span::dummy()`] since predefined macros
/// have no corresponding source file location.
fn tokenize_value(value: &str) -> Vec<PPToken> {
    let bytes = value.as_bytes();
    let len = bytes.len();
    let mut tokens: Vec<PPToken> = Vec::new();
    let mut pos: usize = 0;

    while pos < len {
        let b = bytes[pos];

        // ---- Whitespace ----
        if b == b' ' || b == b'\t' {
            let start = pos;
            while pos < len && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
                pos += 1;
            }
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                &value[start..pos],
                Span::dummy(),
            ));
            continue;
        }

        // ---- String literal: starts with '"' ----
        if b == b'"' {
            let start = pos;
            pos += 1; // skip opening quote
            while pos < len && bytes[pos] != b'"' {
                if bytes[pos] == b'\\' && pos + 1 < len {
                    pos += 2; // skip escape sequence
                } else {
                    pos += 1;
                }
            }
            if pos < len {
                pos += 1; // skip closing quote
            }
            tokens.push(PPToken::new(
                PPTokenKind::StringLiteral,
                &value[start..pos],
                Span::dummy(),
            ));
            continue;
        }

        // ---- Pp-number: starts with a digit ----
        if b.is_ascii_digit() {
            let start = pos;
            // pp-number: digit (digit | letter | '.' | {e,E,p,P}{+,-})*
            while pos < len {
                let c = bytes[pos];
                if c.is_ascii_alphanumeric() || c == b'.' || c == b'_' {
                    pos += 1;
                } else if (c == b'+' || c == b'-')
                    && pos > start
                    && matches!(bytes[pos - 1], b'e' | b'E' | b'p' | b'P')
                {
                    pos += 1; // exponent sign
                } else {
                    break;
                }
            }
            tokens.push(PPToken::new(
                PPTokenKind::Number,
                &value[start..pos],
                Span::dummy(),
            ));
            continue;
        }

        // ---- Identifier: starts with letter or underscore ----
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = pos;
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            tokens.push(PPToken::new(
                PPTokenKind::Identifier,
                &value[start..pos],
                Span::dummy(),
            ));
            continue;
        }

        // ---- Punctuator: any other character ----
        tokens.push(PPToken::new(
            PPTokenKind::Punctuator,
            &value[pos..pos + 1],
            Span::dummy(),
        ));
        pos += 1;
    }

    tokens
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- MagicMacro identification ------------------------------------------

    #[test]
    fn magic_macro_file() {
        assert_eq!(is_magic_macro("__FILE__"), Some(MagicMacro::File));
    }

    #[test]
    fn magic_macro_line() {
        assert_eq!(is_magic_macro("__LINE__"), Some(MagicMacro::Line));
    }

    #[test]
    fn magic_macro_counter() {
        assert_eq!(is_magic_macro("__COUNTER__"), Some(MagicMacro::Counter));
    }

    #[test]
    fn magic_macro_date() {
        assert_eq!(is_magic_macro("__DATE__"), Some(MagicMacro::Date));
    }

    #[test]
    fn magic_macro_time() {
        assert_eq!(is_magic_macro("__TIME__"), Some(MagicMacro::Time));
    }

    #[test]
    fn not_magic_macro() {
        assert_eq!(is_magic_macro("__STDC__"), None);
        assert_eq!(is_magic_macro("FOO"), None);
        assert_eq!(is_magic_macro(""), None);
        assert_eq!(is_magic_macro("__GNUC__"), None);
    }

    // -- Calendar arithmetic ------------------------------------------------

    #[test]
    fn unix_epoch_is_1970_01_01() {
        let (y, m, d, h, min, s) = unix_timestamp_to_components(0);
        assert_eq!((y, m, d, h, min, s), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn known_date_2000_01_01() {
        // 2000-01-01 00:00:00 UTC = 946 684 800 seconds since epoch.
        let (y, m, d, h, min, s) = unix_timestamp_to_components(946_684_800);
        assert_eq!((y, m, d, h, min, s), (2000, 1, 1, 0, 0, 0));
    }

    #[test]
    fn known_date_2024_02_29_leap() {
        // 2024-02-29 00:00:00 UTC = 1 709 164 800 seconds since epoch.
        let (y, m, d, _, _, _) = unix_timestamp_to_components(1_709_164_800);
        assert_eq!((y, m, d), (2024, 2, 29));
    }

    #[test]
    fn known_date_2024_12_31() {
        // 2024-12-31 23:59:59 UTC = 1 735 689 599 seconds since epoch.
        let (y, m, d, h, min, s) = unix_timestamp_to_components(1_735_689_599);
        assert_eq!((y, m, d, h, min, s), (2024, 12, 31, 23, 59, 59));
    }

    #[test]
    fn known_date_1970_03_01() {
        // 1970-03-01 00:00:00 UTC = 59 * 86400 = 5 097 600 seconds.
        let (y, m, d, _, _, _) = unix_timestamp_to_components(5_097_600);
        assert_eq!((y, m, d), (1970, 3, 1));
    }

    // -- Timestamp capture format -------------------------------------------

    #[test]
    fn capture_timestamp_date_format() {
        let (date, _time) = capture_compilation_timestamp();
        // Format: "Mmm dd yyyy" — always 11 characters.
        assert_eq!(date.len(), 11, "Date '{}' should be 11 chars", date);
        // Month should be a valid 3-letter abbreviation.
        let month = &date[0..3];
        let valid = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        assert!(valid.contains(&month), "Invalid month in date: '{}'", date);
        // Year should be the last 4 characters and parseable.
        let year = &date[7..11];
        assert!(
            year.parse::<u32>().is_ok(),
            "Invalid year in date: '{}'",
            date
        );
    }

    #[test]
    fn capture_timestamp_time_format() {
        let (_date, time) = capture_compilation_timestamp();
        // Format: "hh:mm:ss" — always 8 characters.
        assert_eq!(time.len(), 8, "Time '{}' should be 8 chars", time);
        assert_eq!(
            time.as_bytes()[2],
            b':',
            "Missing ':' at pos 2 in '{}'",
            time
        );
        assert_eq!(
            time.as_bytes()[5],
            b':',
            "Missing ':' at pos 5 in '{}'",
            time
        );
        assert!(time[0..2].parse::<u32>().is_ok());
        assert!(time[3..5].parse::<u32>().is_ok());
        assert!(time[6..8].parse::<u32>().is_ok());
    }

    // -- tokenize_value -----------------------------------------------------

    #[test]
    fn tokenize_single_number() {
        let tokens = tokenize_value("42");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "42");
    }

    #[test]
    fn tokenize_number_with_suffix() {
        let tokens = tokenize_value("201112L");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "201112L");
    }

    #[test]
    fn tokenize_number_long_long_suffix() {
        let tokens = tokenize_value("9223372036854775807LL");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "9223372036854775807LL");
    }

    #[test]
    fn tokenize_single_identifier() {
        let tokens = tokenize_value("int");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::Identifier);
        assert_eq!(tokens[0].text, "int");
    }

    #[test]
    fn tokenize_underscore_identifier() {
        let tokens = tokenize_value("__ORDER_LITTLE_ENDIAN__");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::Identifier);
        assert_eq!(tokens[0].text, "__ORDER_LITTLE_ENDIAN__");
    }

    #[test]
    fn tokenize_multi_word_type() {
        let tokens = tokenize_value("unsigned long");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].kind, PPTokenKind::Identifier);
        assert_eq!(tokens[0].text, "unsigned");
        assert_eq!(tokens[1].kind, PPTokenKind::Whitespace);
        assert_eq!(tokens[1].text, " ");
        assert_eq!(tokens[2].kind, PPTokenKind::Identifier);
        assert_eq!(tokens[2].text, "long");
    }

    #[test]
    fn tokenize_three_word_type() {
        let tokens = tokenize_value("unsigned long long");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].text, "unsigned");
        assert_eq!(tokens[2].text, "long");
        assert_eq!(tokens[4].text, "long");
    }

    #[test]
    fn tokenize_string_literal() {
        let tokens = tokenize_value("\"hello world\"");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::StringLiteral);
        assert_eq!(tokens[0].text, "\"hello world\"");
    }

    #[test]
    fn tokenize_string_with_escapes() {
        let tokens = tokenize_value("\"line1\\nline2\"");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::StringLiteral);
    }

    #[test]
    fn tokenize_punctuator() {
        let tokens = tokenize_value("(");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, PPTokenKind::Punctuator);
        assert_eq!(tokens[0].text, "(");
    }

    #[test]
    fn tokenize_empty() {
        let tokens = tokenize_value("");
        assert!(tokens.is_empty());
    }

    // -- register_predefined_macros -----------------------------------------

    #[test]
    fn register_has_stdc() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert!(defs.contains_key("__STDC__"));
        let stdc = &defs["__STDC__"];
        assert!(stdc.is_predefined);
        assert_eq!(stdc.replacement.len(), 1);
        assert_eq!(stdc.replacement[0].text, "1");
        assert_eq!(stdc.replacement[0].kind, PPTokenKind::Number);
    }

    #[test]
    fn register_has_stdc_version() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        let v = &defs["__STDC_VERSION__"];
        assert_eq!(v.replacement[0].text, "201112L");
        assert_eq!(v.replacement[0].kind, PPTokenKind::Number);
    }

    #[test]
    fn register_has_stdc_hosted() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        let h = &defs["__STDC_HOSTED__"];
        assert_eq!(h.replacement[0].text, "1");
    }

    #[test]
    fn register_has_all_magic_macros() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert!(defs.contains_key("__FILE__"), "missing __FILE__");
        assert!(defs.contains_key("__LINE__"), "missing __LINE__");
        assert!(defs.contains_key("__COUNTER__"), "missing __COUNTER__");
        assert!(defs.contains_key("__DATE__"), "missing __DATE__");
        assert!(defs.contains_key("__TIME__"), "missing __TIME__");
        assert_eq!(
            defs["__DATE__"].replacement[0].kind,
            PPTokenKind::StringLiteral
        );
        assert_eq!(
            defs["__TIME__"].replacement[0].kind,
            PPTokenKind::StringLiteral
        );
    }

    #[test]
    fn register_arch_x86_64() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert!(defs.contains_key("__x86_64__"));
        assert!(defs.contains_key("__amd64__"));
        assert!(defs.contains_key("__LP64__"));
        assert_eq!(defs["__SIZEOF_POINTER__"].replacement[0].text, "8");
        assert_eq!(defs["__SIZEOF_LONG__"].replacement[0].text, "8");
    }

    #[test]
    fn register_arch_i686() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::I686);
        assert!(defs.contains_key("__i386__"));
        assert!(defs.contains_key("__i686__"));
        assert!(defs.contains_key("__ILP32__"));
        assert_eq!(defs["__SIZEOF_POINTER__"].replacement[0].text, "4");
        assert_eq!(defs["__SIZEOF_LONG__"].replacement[0].text, "4");
    }

    #[test]
    fn register_arch_aarch64() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::AArch64);
        assert!(defs.contains_key("__aarch64__"));
        assert!(defs.contains_key("__ARM_64BIT_STATE"));
        assert_eq!(defs["__SIZEOF_POINTER__"].replacement[0].text, "8");
    }

    #[test]
    fn register_arch_riscv64() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::RiscV64);
        assert!(defs.contains_key("__riscv"));
        assert_eq!(defs["__riscv_xlen"].replacement[0].text, "64");
        assert_eq!(defs["__SIZEOF_POINTER__"].replacement[0].text, "8");
    }

    #[test]
    fn register_type_macros_64bit() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        let st = &defs["__SIZE_TYPE__"];
        let joined: String = st
            .replacement
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(joined, "unsigned long");
    }

    #[test]
    fn register_type_macros_32bit() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::I686);
        let st = &defs["__SIZE_TYPE__"];
        let joined: String = st
            .replacement
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(joined, "unsigned int");
    }

    #[test]
    fn register_gnuc_macros() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert!(defs.contains_key("__GNUC__"));
        assert_eq!(defs["__GNUC__"].replacement[0].text, "12");
        assert!(defs.contains_key("__GNUC_MINOR__"));
        assert_eq!(defs["__GNUC_MINOR__"].replacement[0].text, "2");
        assert!(defs.contains_key("__GNUC_PATCHLEVEL__"));
        assert_eq!(defs["__GNUC_PATCHLEVEL__"].replacement[0].text, "0");
        assert!(defs.contains_key("__VERSION__"));
    }

    #[test]
    fn register_byte_order_macros() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert!(defs.contains_key("__ORDER_LITTLE_ENDIAN__"));
        assert!(defs.contains_key("__ORDER_BIG_ENDIAN__"));
        assert!(defs.contains_key("__ORDER_PDP_ENDIAN__"));
        assert!(defs.contains_key("__BYTE_ORDER__"));
        assert_eq!(
            defs["__BYTE_ORDER__"].replacement[0].kind,
            PPTokenKind::Identifier
        );
        assert_eq!(
            defs["__BYTE_ORDER__"].replacement[0].text,
            "__ORDER_LITTLE_ENDIAN__"
        );
    }

    #[test]
    fn register_linux_platform_macros() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert!(defs.contains_key("__linux__"));
        assert!(defs.contains_key("__unix__"));
        assert!(defs.contains_key("__ELF__"));
    }

    #[test]
    fn register_sizeof_macros_common() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        assert_eq!(defs["__SIZEOF_SHORT__"].replacement[0].text, "2");
        assert_eq!(defs["__SIZEOF_INT__"].replacement[0].text, "4");
        assert_eq!(defs["__SIZEOF_FLOAT__"].replacement[0].text, "4");
        assert_eq!(defs["__SIZEOF_DOUBLE__"].replacement[0].text, "8");
        assert_eq!(defs["__SIZEOF_LONG_LONG__"].replacement[0].text, "8");
        assert_eq!(defs["__CHAR_BIT__"].replacement[0].text, "8");
    }

    #[test]
    fn register_sizeof_long_double_target_dependent() {
        let mut defs64 = FxHashMap::default();
        register_predefined_macros(&mut defs64, &Target::X86_64);
        assert_eq!(defs64["__SIZEOF_LONG_DOUBLE__"].replacement[0].text, "16");

        let mut defs32 = FxHashMap::default();
        register_predefined_macros(&mut defs32, &Target::I686);
        assert_eq!(defs32["__SIZEOF_LONG_DOUBLE__"].replacement[0].text, "12");
    }

    #[test]
    fn all_predefined_macros_are_flagged() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        for (_name, def) in defs.iter() {
            assert!(
                def.is_predefined,
                "Macro '{}' should be is_predefined",
                def.name
            );
            assert!(
                def.definition_span.is_dummy(),
                "Macro '{}' should have dummy span",
                def.name
            );
        }
    }

    #[test]
    fn all_predefined_macros_are_object_like() {
        let mut defs = FxHashMap::default();
        register_predefined_macros(&mut defs, &Target::X86_64);
        for (_name, def) in defs.iter() {
            assert!(
                matches!(def.kind, MacroKind::ObjectLike),
                "Predefined macro '{}' should be ObjectLike",
                def.name
            );
        }
    }

    #[test]
    fn date_time_values_are_constant_format() {
        let (d1, t1) = capture_compilation_timestamp();
        let (d2, t2) = capture_compilation_timestamp();
        assert_eq!(d1.len(), 11);
        assert_eq!(d2.len(), 11);
        assert_eq!(t1.len(), 8);
        assert_eq!(t2.len(), 8);
    }

    #[test]
    fn long_max_64bit_vs_32bit() {
        let mut defs64 = FxHashMap::default();
        register_predefined_macros(&mut defs64, &Target::X86_64);
        assert_eq!(
            defs64["__LONG_MAX__"].replacement[0].text,
            "9223372036854775807L"
        );

        let mut defs32 = FxHashMap::default();
        register_predefined_macros(&mut defs32, &Target::I686);
        assert_eq!(defs32["__LONG_MAX__"].replacement[0].text, "2147483647L");
    }

    #[test]
    fn int64_type_64bit_vs_32bit() {
        let mut defs64 = FxHashMap::default();
        register_predefined_macros(&mut defs64, &Target::X86_64);
        let joined64: String = defs64["__INT64_TYPE__"]
            .replacement
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(joined64, "long");

        let mut defs32 = FxHashMap::default();
        register_predefined_macros(&mut defs32, &Target::I686);
        let joined32: String = defs32["__INT64_TYPE__"]
            .replacement
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(joined32, "long long");
    }
}
