// src/frontend/lexer/token.rs
//
// Token type definitions for the BCC compiler lexer — the most foundational
// type in the entire frontend pipeline.
//
// This module defines:
//
// - `TokenKind` — a comprehensive enumeration of every token the BCC lexer
//   can produce, covering all C11 keywords, C11 special keywords, GCC
//   extension keywords, GCC builtins, identifiers, integer/float/string/
//   character literals, all single- and multi-character operators and
//   punctuators, and special tokens (EOF, error recovery, newline,
//   whitespace).
//
// - `Token` — pairs a `TokenKind` with a source `Span` for location tracking.
//
// - `IntegerSuffix`, `FloatSuffix`, `StringPrefix`, `CharPrefix` — auxiliary
//   enums describing literal encoding and type suffixes.
//
// - `Span` — re-exported from `crate::common::diagnostics` to provide a
//   single import point for consumers. The canonical definition lives in
//   `diagnostics.rs`; this module re-exports it so that `use token::Span`
//   works everywhere in the frontend.
//
// This file is imported by virtually every other frontend module
// (preprocessor, parser, sema) and by IR lowering. It must remain
// self-consistent and free of circular dependencies.

use std::fmt;

// Re-export Span from the canonical definition in diagnostics.rs.
// Per architectural coordination: diagnostics.rs defines the Span struct
// with fields (file_id: u32, start: u32, end: u32), the DUMMY sentinel
// constant, new() constructor, and merge() combiner. All consumers of
// Token access Span through this re-export to avoid duplication.
//
// Usage:
//   Span::new(file_id, start, end)  — construct a span
//   Span::DUMMY                     — sentinel for synthesized/generated tokens
//   Span::merge(a, b)               — combine two spans into covering span
pub use crate::common::diagnostics::Span;

use crate::common::string_interner::Symbol;

// ===========================================================================
// Integer Literal Suffix
// ===========================================================================

/// Suffix applied to integer literal tokens, determining the C type of the
/// constant.
///
/// The lexer parses trailing `u`/`U`, `l`/`L`, and `ll`/`LL` characters
/// from integer literals and stores the normalised suffix here. The semantic
/// analyser uses this to determine the type (`int`, `unsigned int`,
/// `long`, `unsigned long`, `long long`, `unsigned long long`) according
/// to the C11 §6.4.4.1 conversion rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntegerSuffix {
    /// No suffix — the literal's type is determined by its value.
    None,
    /// `u` or `U` — unsigned.
    U,
    /// `l` or `L` — long.
    L,
    /// `ul`, `uL`, `Ul`, `UL` — unsigned long.
    UL,
    /// `ll` or `LL` — long long.
    LL,
    /// `ull`, `uLL`, `Ull`, `ULL` — unsigned long long.
    ULL,
}

impl fmt::Display for IntegerSuffix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegerSuffix::None => Ok(()),
            IntegerSuffix::U => write!(f, "u"),
            IntegerSuffix::L => write!(f, "l"),
            IntegerSuffix::UL => write!(f, "ul"),
            IntegerSuffix::LL => write!(f, "ll"),
            IntegerSuffix::ULL => write!(f, "ull"),
        }
    }
}

// ===========================================================================
// Floating-Point Literal Suffix
// ===========================================================================

/// Suffix applied to floating-point literal tokens, determining the C type
/// of the constant.
///
/// - `None` → `double` (the default for unsuffixed floating-point literals).
/// - `F` → `float`.
/// - `L` → `long double` (80-bit extended or 128-bit, target-dependent).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatSuffix {
    /// No suffix — `double`.
    None,
    /// `f` or `F` — `float`.
    F,
    /// `l` or `L` — `long double`.
    L,
}

impl fmt::Display for FloatSuffix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatSuffix::None => Ok(()),
            FloatSuffix::F => write!(f, "f"),
            FloatSuffix::L => write!(f, "l"),
        }
    }
}

// ===========================================================================
// String Literal Prefix
// ===========================================================================

/// Encoding prefix for string literals, determining the element type and
/// encoding of the string data.
///
/// C11 §6.4.5 defines these prefix forms:
/// - No prefix → `char[]` (execution charset).
/// - `L` → `wchar_t[]` (wide character, typically 32-bit on Linux).
/// - `u8` → `char[]` (UTF-8 encoded).
/// - `u` → `char16_t[]` (UTF-16 encoded).
/// - `U` → `char32_t[]` (UTF-32 encoded).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StringPrefix {
    /// No prefix — byte string in execution charset.
    None,
    /// `L` prefix — wide string (`wchar_t`).
    L,
    /// `u8` prefix — UTF-8 encoded byte string.
    U8,
    /// `u` prefix — UTF-16 encoded string (`char16_t`).
    SmallU,
    /// `U` prefix — UTF-32 encoded string (`char32_t`).
    BigU,
}

impl fmt::Display for StringPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StringPrefix::None => Ok(()),
            StringPrefix::L => write!(f, "L"),
            StringPrefix::U8 => write!(f, "u8"),
            StringPrefix::SmallU => write!(f, "u"),
            StringPrefix::BigU => write!(f, "U"),
        }
    }
}

// ===========================================================================
// Character Literal Prefix
// ===========================================================================

/// Encoding prefix for character literals, determining the type and encoding
/// of the character constant.
///
/// C11 §6.4.4.4 defines these prefix forms:
/// - No prefix → `int` (ordinary character constant).
/// - `L` → `wchar_t` (wide character constant).
/// - `u` → `char16_t` (UTF-16 character constant).
/// - `U` → `char32_t` (UTF-32 character constant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CharPrefix {
    /// No prefix — ordinary character constant (`int`).
    None,
    /// `L` prefix — wide character constant (`wchar_t`).
    L,
    /// `u` prefix — UTF-16 character constant (`char16_t`).
    SmallU,
    /// `U` prefix — UTF-32 character constant (`char32_t`).
    BigU,
}

impl fmt::Display for CharPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CharPrefix::None => Ok(()),
            CharPrefix::L => write!(f, "L"),
            CharPrefix::SmallU => write!(f, "u"),
            CharPrefix::BigU => write!(f, "U"),
        }
    }
}

// ===========================================================================
// TokenKind — comprehensive token enumeration
// ===========================================================================

/// Every distinct token kind that the BCC lexer can produce.
///
/// This enum is the heart of the tokenisation layer. It covers:
///
/// - **34 standard C keywords** (`auto` through `while`).
/// - **10 C11-specific keywords** (`_Alignas` through `_Thread_local`).
/// - **10 GCC extension keywords** (`__attribute__`, `typeof`, etc.).
/// - **27+ GCC builtin keywords** (`__builtin_va_list`, `__builtin_expect`, etc.).
/// - **5 literal forms** (identifier, integer, float, string, character).
/// - **24 single-character operators/punctuators** (`+`, `-`, `*`, etc.).
/// - **24 multi-character operators/punctuators** (`==`, `->`, `...`, etc.).
/// - **4 special tokens** (EOF, error, newline, whitespace).
///
/// Data-carrying variants (`Identifier`, `IntegerLiteral`, `FloatLiteral`,
/// `StringLiteral`, `CharLiteral`) store the parsed value directly in the
/// token. Simple keyword and operator variants carry no data.
///
/// # Comparison
///
/// `TokenKind` derives `PartialEq` for value-level comparison. For
/// discriminant-only comparison (e.g., "is this token any identifier?"),
/// use `Token::is()` which compares `std::mem::discriminant`.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // -----------------------------------------------------------------------
    // C11 Standard Keywords (ISO/IEC 9899:2011 §6.4.1)
    // -----------------------------------------------------------------------

    /// `auto` — storage class specifier.
    Auto,
    /// `break` — loop/switch exit statement.
    Break,
    /// `case` — switch case label.
    Case,
    /// `char` — character type specifier.
    Char,
    /// `const` — type qualifier.
    Const,
    /// `continue` — loop continuation statement.
    Continue,
    /// `default` — switch default label.
    Default,
    /// `do` — do-while loop.
    Do,
    /// `double` — double-precision floating-point type.
    Double,
    /// `else` — conditional else branch.
    Else,
    /// `enum` — enumeration type specifier.
    Enum,
    /// `extern` — external linkage storage class.
    Extern,
    /// `float` — single-precision floating-point type.
    Float,
    /// `for` — for loop.
    For,
    /// `goto` — unconditional jump (also computed goto via GCC extension).
    Goto,
    /// `if` — conditional statement.
    If,
    /// `inline` — function specifier.
    Inline,
    /// `int` — integer type specifier.
    Int,
    /// `long` — type specifier modifier.
    Long,
    /// `register` — storage class specifier (advisory).
    Register,
    /// `restrict` — pointer qualifier (C99/C11).
    Restrict,
    /// `return` — function return statement.
    Return,
    /// `short` — type specifier modifier.
    Short,
    /// `signed` — type specifier modifier.
    Signed,
    /// `sizeof` — size-of operator.
    Sizeof,
    /// `static` — storage class specifier / function-scope persistence.
    Static,
    /// `struct` — structure type specifier.
    Struct,
    /// `switch` — multi-way branch statement.
    Switch,
    /// `typedef` — type alias storage class.
    Typedef,
    /// `union` — union type specifier.
    Union,
    /// `unsigned` — type specifier modifier.
    Unsigned,
    /// `void` — void type specifier.
    Void,
    /// `volatile` — type qualifier.
    Volatile,
    /// `while` — while loop / do-while condition.
    While,

    // -----------------------------------------------------------------------
    // C11-Specific Keywords (§6.4.1, underscore-uppercase forms)
    // -----------------------------------------------------------------------

    /// `_Alignas` — alignment specifier.
    Alignas,
    /// `_Alignof` — alignment query operator.
    Alignof,
    /// `_Atomic` — atomic type qualifier / specifier.
    Atomic,
    /// `_Bool` — boolean type specifier.
    Bool,
    /// `_Complex` — complex number type specifier.
    Complex,
    /// `_Generic` — generic selection expression.
    Generic,
    /// `_Imaginary` — imaginary number type specifier (reserved).
    Imaginary,
    /// `_Noreturn` — function specifier for non-returning functions.
    Noreturn,
    /// `_Static_assert` — compile-time assertion.
    StaticAssert,
    /// `_Thread_local` — thread-local storage class specifier.
    ThreadLocal,

    // -----------------------------------------------------------------------
    // GCC Extension Keywords
    // -----------------------------------------------------------------------

    /// `__attribute__` / `__attribute` — GCC attribute syntax.
    Attribute,
    /// `typeof` / `__typeof__` / `__typeof` — GCC typeof operator.
    TypeofKeyword,
    /// `__extension__` — suppress pedantic warnings for GCC extensions.
    Extension,
    /// `asm` / `__asm__` / `__asm` — inline assembly statement.
    AsmKeyword,
    /// `__volatile__` / `__volatile` — GCC volatile qualifier form
    /// (distinguishable from C `volatile` for parser clarity).
    VolatileGcc,
    /// `__inline__` / `__inline` — GCC inline specifier form.
    InlineGcc,
    /// `__signed__` / `__signed` — GCC signed specifier form.
    SignedGcc,
    /// `__const__` / `__const` — GCC const qualifier form.
    ConstGcc,
    /// `__restrict__` / `__restrict` — GCC restrict qualifier form.
    RestrictGcc,
    /// `__label__` — GCC local label declaration.
    Label,

    // -----------------------------------------------------------------------
    // GCC Builtins — va_list and Variadic Argument Support
    // -----------------------------------------------------------------------

    /// `__builtin_va_list` — variadic argument list type.
    BuiltinVaList,
    /// `__builtin_va_start` — initialise variadic argument traversal.
    BuiltinVaStart,
    /// `__builtin_va_end` — end variadic argument traversal.
    BuiltinVaEnd,
    /// `__builtin_va_arg` — retrieve next variadic argument.
    BuiltinVaArg,
    /// `__builtin_va_copy` — copy variadic argument state.
    BuiltinVaCopy,

    // -----------------------------------------------------------------------
    // GCC Builtins — Type Introspection and Compile-Time Evaluation
    // -----------------------------------------------------------------------

    /// `__builtin_offsetof` — byte offset of a struct member.
    BuiltinOffsetof,
    /// `__builtin_types_compatible_p` — compile-time type compatibility check.
    BuiltinTypesCompatibleP,
    /// `__builtin_choose_expr` — compile-time conditional expression.
    BuiltinChooseExpr,
    /// `__builtin_constant_p` — check if expression is a compile-time constant.
    BuiltinConstantP,

    // -----------------------------------------------------------------------
    // GCC Builtins — Branch Prediction and Control Flow
    // -----------------------------------------------------------------------

    /// `__builtin_expect` — branch prediction hint.
    BuiltinExpect,
    /// `__builtin_unreachable` — mark code path as unreachable.
    BuiltinUnreachable,
    /// `__builtin_trap` — generate a trap instruction (abort).
    BuiltinTrap,

    // -----------------------------------------------------------------------
    // GCC Builtins — Bit Manipulation
    // -----------------------------------------------------------------------

    /// `__builtin_clz` — count leading zeros (undefined for zero input).
    BuiltinClz,
    /// `__builtin_ctz` — count trailing zeros (undefined for zero input).
    BuiltinCtz,
    /// `__builtin_popcount` — count set bits (population count).
    BuiltinPopcount,

    // -----------------------------------------------------------------------
    // GCC Builtins — Byte Swap
    // -----------------------------------------------------------------------

    /// `__builtin_bswap16` — 16-bit byte swap.
    BuiltinBswap16,
    /// `__builtin_bswap32` — 32-bit byte swap.
    BuiltinBswap32,
    /// `__builtin_bswap64` — 64-bit byte swap.
    BuiltinBswap64,

    // -----------------------------------------------------------------------
    // GCC Builtins — Miscellaneous
    // -----------------------------------------------------------------------

    /// `__builtin_ffs` — find first set bit (1-indexed, 0 if input is zero).
    BuiltinFfs,
    /// `__builtin_frame_address` — return address of stack frame.
    BuiltinFrameAddress,
    /// `__builtin_return_address` — return address of calling function.
    BuiltinReturnAddress,
    /// `__builtin_assume_aligned` — pointer alignment hint.
    BuiltinAssumeAligned,

    // -----------------------------------------------------------------------
    // GCC Builtins — Checked Arithmetic (Overflow Detection)
    // -----------------------------------------------------------------------

    /// `__builtin_add_overflow` — addition with overflow detection.
    BuiltinAddOverflow,
    /// `__builtin_sub_overflow` — subtraction with overflow detection.
    BuiltinSubOverflow,
    /// `__builtin_mul_overflow` — multiplication with overflow detection.
    BuiltinMulOverflow,

    // -----------------------------------------------------------------------
    // Identifiers and Literals
    // -----------------------------------------------------------------------

    /// An identifier — interned via `Symbol` for zero-cost comparison.
    ///
    /// The `Symbol` handle is an index into the global `Interner`. Identifiers
    /// include user-defined names, unrecognised keywords, and macro names
    /// after preprocessing.
    Identifier(Symbol),

    /// Integer literal with its parsed value and type suffix.
    ///
    /// The `value` field holds the full-precision unsigned value (up to 128 bits)
    /// as parsed from decimal, hexadecimal, octal, or binary notation. The
    /// `suffix` determines the C type of the constant per C11 §6.4.4.1.
    IntegerLiteral {
        /// Unsigned integer value (up to 128 bits).
        value: u128,
        /// Parsed suffix (None, U, L, UL, LL, ULL).
        suffix: IntegerSuffix,
    },

    /// Floating-point literal with its parsed value and type suffix.
    ///
    /// The `value` field holds the nearest `f64` representation. For `long double`
    /// literals (suffix `L`), the full 80-bit/128-bit precision is handled by
    /// the semantic analyser using `crate::common::long_double`.
    FloatLiteral {
        /// Floating-point value (nearest f64 representation).
        value: f64,
        /// Parsed suffix (None = double, F = float, L = long double).
        suffix: FloatSuffix,
    },

    /// String literal with its raw byte content and encoding prefix.
    ///
    /// The `value` field contains the bytes after escape-sequence processing.
    /// Non-UTF-8 bytes are preserved via PUA encoding for byte-exact fidelity
    /// (required for Linux kernel compilation). The `prefix` determines the
    /// element type and encoding.
    StringLiteral {
        /// Byte content after escape processing.
        value: Vec<u8>,
        /// Encoding prefix (None, L, u8, u, U).
        prefix: StringPrefix,
    },

    /// Character literal with its parsed code-point value and encoding prefix.
    ///
    /// Multi-character constants (e.g., `'ab'`) are implementation-defined;
    /// BCC stores them as the combined value. The `prefix` determines the
    /// result type per C11 §6.4.4.4.
    CharLiteral {
        /// Character code point or combined multi-character value.
        value: u32,
        /// Encoding prefix (None = int, L = wchar_t, u = char16_t, U = char32_t).
        prefix: CharPrefix,
    },

    // -----------------------------------------------------------------------
    // Single-Character Operators and Punctuators
    // -----------------------------------------------------------------------

    /// `+` — addition / unary plus.
    Plus,
    /// `-` — subtraction / unary minus / arrow (when followed by `>`).
    Minus,
    /// `*` — multiplication / dereference / pointer declarator.
    Star,
    /// `/` — division.
    Slash,
    /// `%` — modulo.
    Percent,
    /// `&` — bitwise AND / address-of.
    Ampersand,
    /// `|` — bitwise OR.
    Pipe,
    /// `^` — bitwise XOR.
    Caret,
    /// `~` — bitwise NOT.
    Tilde,
    /// `!` — logical NOT.
    Exclaim,
    /// `<` — less-than / template argument (if extended).
    Less,
    /// `>` — greater-than.
    Greater,
    /// `=` — assignment.
    Assign,
    /// `.` — member access.
    Dot,
    /// `,` — comma operator / separator.
    Comma,
    /// `;` — statement terminator.
    Semicolon,
    /// `:` — label / ternary / bitfield width.
    Colon,
    /// `?` — ternary conditional operator.
    Question,
    /// `(` — left parenthesis.
    LeftParen,
    /// `)` — right parenthesis.
    RightParen,
    /// `[` — left bracket (array subscript).
    LeftBracket,
    /// `]` — right bracket.
    RightBracket,
    /// `{` — left brace (compound statement / initializer).
    LeftBrace,
    /// `}` — right brace.
    RightBrace,

    // -----------------------------------------------------------------------
    // Multi-Character Operators and Punctuators
    // -----------------------------------------------------------------------

    /// `==` — equality comparison.
    EqualEqual,
    /// `!=` — inequality comparison.
    NotEqual,
    /// `<=` — less-than-or-equal comparison.
    LessEqual,
    /// `>=` — greater-than-or-equal comparison.
    GreaterEqual,
    /// `<<` — left shift.
    LeftShift,
    /// `>>` — right shift.
    RightShift,
    /// `->` — member access through pointer.
    Arrow,
    /// `++` — increment (prefix or postfix).
    PlusPlus,
    /// `--` — decrement (prefix or postfix).
    MinusMinus,
    /// `&&` — logical AND.
    AmpAmp,
    /// `||` — logical OR.
    PipePipe,
    /// `+=` — addition assignment.
    PlusAssign,
    /// `-=` — subtraction assignment.
    MinusAssign,
    /// `*=` — multiplication assignment.
    StarAssign,
    /// `/=` — division assignment.
    SlashAssign,
    /// `%=` — modulo assignment.
    PercentAssign,
    /// `&=` — bitwise AND assignment.
    AmpAssign,
    /// `|=` — bitwise OR assignment.
    PipeAssign,
    /// `^=` — bitwise XOR assignment.
    CaretAssign,
    /// `<<=` — left shift assignment.
    LeftShiftAssign,
    /// `>>=` — right shift assignment.
    RightShiftAssign,
    /// `...` — ellipsis (variadic parameter, designator range).
    Ellipsis,
    /// `#` — preprocessor directive prefix / stringification operator.
    Hash,
    /// `##` — preprocessor token-pasting operator.
    HashHash,

    // -----------------------------------------------------------------------
    // Special Tokens
    // -----------------------------------------------------------------------

    /// End of file — signals the lexer has consumed all input.
    Eof,
    /// Error recovery token — produced when the lexer encounters an
    /// unrecognisable character sequence. The diagnostic engine records
    /// the details; this token allows the parser to continue.
    Error,
    /// Newline — significant for preprocessor directive boundary detection.
    /// Stripped from the token stream before the parser sees it.
    Newline,
    /// Whitespace — optionally preserved for preprocessor token spacing
    /// and `#` / `##` operator semantics. Stripped before parsing.
    Whitespace,
}

// ===========================================================================
// TokenKind — utility methods
// ===========================================================================

impl TokenKind {
    /// Returns `true` if this token is any assignment operator.
    ///
    /// Matches: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`.
    #[inline]
    pub fn is_assignment_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Assign
                | TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::PercentAssign
                | TokenKind::AmpAssign
                | TokenKind::PipeAssign
                | TokenKind::CaretAssign
                | TokenKind::LeftShiftAssign
                | TokenKind::RightShiftAssign
        )
    }

    /// Returns `true` if this token is a comparison (relational/equality) operator.
    ///
    /// Matches: `==`, `!=`, `<`, `>`, `<=`, `>=`.
    #[inline]
    pub fn is_comparison_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::EqualEqual
                | TokenKind::NotEqual
                | TokenKind::Less
                | TokenKind::Greater
                | TokenKind::LessEqual
                | TokenKind::GreaterEqual
        )
    }

    /// Returns `true` if this token can appear as a unary prefix operator.
    ///
    /// Matches: `!`, `~`, `-`, `+`, `*`, `&`, `++`, `--`.
    ///
    /// Note: `*` (dereference) and `&` (address-of) are contextually unary;
    /// this method reports their syntactic eligibility, not semantic role.
    #[inline]
    pub fn is_unary_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Exclaim
                | TokenKind::Tilde
                | TokenKind::Minus
                | TokenKind::Plus
                | TokenKind::Star
                | TokenKind::Ampersand
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus
        )
    }

    /// Returns `true` if this token is a type qualifier keyword.
    ///
    /// Matches: `const`, `volatile`, `restrict`, `_Atomic`,
    /// `__const__`, `__volatile__`, `__restrict__`.
    #[inline]
    pub fn is_type_qualifier(&self) -> bool {
        matches!(
            self,
            TokenKind::Const
                | TokenKind::Volatile
                | TokenKind::Restrict
                | TokenKind::Atomic
                | TokenKind::ConstGcc
                | TokenKind::VolatileGcc
                | TokenKind::RestrictGcc
        )
    }

    /// Returns `true` if this token is a storage class specifier keyword.
    ///
    /// Matches: `auto`, `register`, `static`, `extern`, `typedef`, `_Thread_local`.
    #[inline]
    pub fn is_storage_class(&self) -> bool {
        matches!(
            self,
            TokenKind::Auto
                | TokenKind::Register
                | TokenKind::Static
                | TokenKind::Extern
                | TokenKind::Typedef
                | TokenKind::ThreadLocal
        )
    }

    /// Returns the binary operator precedence level for expression parsing.
    ///
    /// Higher values bind more tightly. Returns `None` for tokens that are
    /// not binary operators. Used by the parser's precedence-climbing
    /// algorithm.
    ///
    /// Precedence levels (lowest to highest):
    ///
    /// | Level | Operators                                    |
    /// |-------|----------------------------------------------|
    /// |   1   | `,` (comma)                                  |
    /// |   2   | `=` `+=` `-=` `*=` `/=` `%=` `&=` etc.      |
    /// |   3   | `?:` (ternary conditional)                   |
    /// |   4   | `\|\|` (logical OR)                          |
    /// |   5   | `&&` (logical AND)                           |
    /// |   6   | `\|` (bitwise OR)                            |
    /// |   7   | `^` (bitwise XOR)                            |
    /// |   8   | `&` (bitwise AND)                            |
    /// |   9   | `==` `!=` (equality)                         |
    /// |  10   | `<` `>` `<=` `>=` (relational)               |
    /// |  11   | `<<` `>>` (shift)                            |
    /// |  12   | `+` `-` (additive)                           |
    /// |  13   | `*` `/` `%` (multiplicative)                 |
    pub fn precedence(&self) -> Option<u8> {
        match self {
            // Level 1 — Comma (lowest binary precedence)
            TokenKind::Comma => Some(1),

            // Level 2 — Assignment operators (right-associative)
            TokenKind::Assign
            | TokenKind::PlusAssign
            | TokenKind::MinusAssign
            | TokenKind::StarAssign
            | TokenKind::SlashAssign
            | TokenKind::PercentAssign
            | TokenKind::AmpAssign
            | TokenKind::PipeAssign
            | TokenKind::CaretAssign
            | TokenKind::LeftShiftAssign
            | TokenKind::RightShiftAssign => Some(2),

            // Level 3 — Ternary conditional (right-associative)
            // The parser handles the ?: syntax specially, but precedence
            // is provided for the initial `?` token.
            TokenKind::Question => Some(3),

            // Level 4 — Logical OR
            TokenKind::PipePipe => Some(4),

            // Level 5 — Logical AND
            TokenKind::AmpAmp => Some(5),

            // Level 6 — Bitwise OR
            TokenKind::Pipe => Some(6),

            // Level 7 — Bitwise XOR
            TokenKind::Caret => Some(7),

            // Level 8 — Bitwise AND
            TokenKind::Ampersand => Some(8),

            // Level 9 — Equality
            TokenKind::EqualEqual | TokenKind::NotEqual => Some(9),

            // Level 10 — Relational
            TokenKind::Less
            | TokenKind::Greater
            | TokenKind::LessEqual
            | TokenKind::GreaterEqual => Some(10),

            // Level 11 — Shift
            TokenKind::LeftShift | TokenKind::RightShift => Some(11),

            // Level 12 — Additive
            TokenKind::Plus | TokenKind::Minus => Some(12),

            // Level 13 — Multiplicative (highest binary precedence)
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some(13),

            // Not a binary operator
            _ => None,
        }
    }

    /// Returns `true` if this operator is right-associative.
    ///
    /// In C, the right-associative operators are:
    /// - All assignment operators (`=`, `+=`, `-=`, etc.)
    /// - The ternary conditional operator (`?:`)
    ///
    /// All other binary operators are left-associative.
    #[inline]
    pub fn is_right_associative(&self) -> bool {
        matches!(
            self,
            TokenKind::Assign
                | TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::PercentAssign
                | TokenKind::AmpAssign
                | TokenKind::PipeAssign
                | TokenKind::CaretAssign
                | TokenKind::LeftShiftAssign
                | TokenKind::RightShiftAssign
                | TokenKind::Question
        )
    }

    /// Returns `true` if this token is an identifier.
    #[inline]
    pub fn is_identifier(&self) -> bool {
        matches!(self, TokenKind::Identifier(_))
    }

    /// Returns the `Symbol` handle of an identifier token.
    ///
    /// For non-identifier tokens, returns `Symbol::EMPTY` as a sentinel.
    /// Callers should check `is_identifier()` first if the distinction
    /// matters.
    #[inline]
    pub fn identifier_symbol(&self) -> Symbol {
        match self {
            TokenKind::Identifier(sym) => *sym,
            _ => Symbol::EMPTY,
        }
    }
}

// ===========================================================================
// TokenKind — Display implementation
// ===========================================================================

/// Formats a `TokenKind` as a human-readable string for diagnostic messages.
///
/// - Keywords display as their C spelling (e.g., `auto`, `_Alignas`,
///   `__attribute__`).
/// - Operators display as their symbols (e.g., `==`, `->`).
/// - Literals display as their category names (e.g., `integer literal`).
/// - Special tokens display as descriptive names (e.g., `end of file`).
impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // C11 Standard Keywords
            TokenKind::Auto => write!(f, "auto"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Case => write!(f, "case"),
            TokenKind::Char => write!(f, "char"),
            TokenKind::Const => write!(f, "const"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Default => write!(f, "default"),
            TokenKind::Do => write!(f, "do"),
            TokenKind::Double => write!(f, "double"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::Enum => write!(f, "enum"),
            TokenKind::Extern => write!(f, "extern"),
            TokenKind::Float => write!(f, "float"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Goto => write!(f, "goto"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Inline => write!(f, "inline"),
            TokenKind::Int => write!(f, "int"),
            TokenKind::Long => write!(f, "long"),
            TokenKind::Register => write!(f, "register"),
            TokenKind::Restrict => write!(f, "restrict"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Short => write!(f, "short"),
            TokenKind::Signed => write!(f, "signed"),
            TokenKind::Sizeof => write!(f, "sizeof"),
            TokenKind::Static => write!(f, "static"),
            TokenKind::Struct => write!(f, "struct"),
            TokenKind::Switch => write!(f, "switch"),
            TokenKind::Typedef => write!(f, "typedef"),
            TokenKind::Union => write!(f, "union"),
            TokenKind::Unsigned => write!(f, "unsigned"),
            TokenKind::Void => write!(f, "void"),
            TokenKind::Volatile => write!(f, "volatile"),
            TokenKind::While => write!(f, "while"),

            // C11-Specific Keywords
            TokenKind::Alignas => write!(f, "_Alignas"),
            TokenKind::Alignof => write!(f, "_Alignof"),
            TokenKind::Atomic => write!(f, "_Atomic"),
            TokenKind::Bool => write!(f, "_Bool"),
            TokenKind::Complex => write!(f, "_Complex"),
            TokenKind::Generic => write!(f, "_Generic"),
            TokenKind::Imaginary => write!(f, "_Imaginary"),
            TokenKind::Noreturn => write!(f, "_Noreturn"),
            TokenKind::StaticAssert => write!(f, "_Static_assert"),
            TokenKind::ThreadLocal => write!(f, "_Thread_local"),

            // GCC Extension Keywords
            TokenKind::Attribute => write!(f, "__attribute__"),
            TokenKind::TypeofKeyword => write!(f, "typeof"),
            TokenKind::Extension => write!(f, "__extension__"),
            TokenKind::AsmKeyword => write!(f, "asm"),
            TokenKind::VolatileGcc => write!(f, "__volatile__"),
            TokenKind::InlineGcc => write!(f, "__inline__"),
            TokenKind::SignedGcc => write!(f, "__signed__"),
            TokenKind::ConstGcc => write!(f, "__const__"),
            TokenKind::RestrictGcc => write!(f, "__restrict__"),
            TokenKind::Label => write!(f, "__label__"),

            // GCC Builtins — Variadic
            TokenKind::BuiltinVaList => write!(f, "__builtin_va_list"),
            TokenKind::BuiltinVaStart => write!(f, "__builtin_va_start"),
            TokenKind::BuiltinVaEnd => write!(f, "__builtin_va_end"),
            TokenKind::BuiltinVaArg => write!(f, "__builtin_va_arg"),
            TokenKind::BuiltinVaCopy => write!(f, "__builtin_va_copy"),

            // GCC Builtins — Type Introspection
            TokenKind::BuiltinOffsetof => write!(f, "__builtin_offsetof"),
            TokenKind::BuiltinTypesCompatibleP => write!(f, "__builtin_types_compatible_p"),
            TokenKind::BuiltinChooseExpr => write!(f, "__builtin_choose_expr"),
            TokenKind::BuiltinConstantP => write!(f, "__builtin_constant_p"),

            // GCC Builtins — Branch Prediction and Control Flow
            TokenKind::BuiltinExpect => write!(f, "__builtin_expect"),
            TokenKind::BuiltinUnreachable => write!(f, "__builtin_unreachable"),
            TokenKind::BuiltinTrap => write!(f, "__builtin_trap"),

            // GCC Builtins — Bit Manipulation
            TokenKind::BuiltinClz => write!(f, "__builtin_clz"),
            TokenKind::BuiltinCtz => write!(f, "__builtin_ctz"),
            TokenKind::BuiltinPopcount => write!(f, "__builtin_popcount"),

            // GCC Builtins — Byte Swap
            TokenKind::BuiltinBswap16 => write!(f, "__builtin_bswap16"),
            TokenKind::BuiltinBswap32 => write!(f, "__builtin_bswap32"),
            TokenKind::BuiltinBswap64 => write!(f, "__builtin_bswap64"),

            // GCC Builtins — Miscellaneous
            TokenKind::BuiltinFfs => write!(f, "__builtin_ffs"),
            TokenKind::BuiltinFrameAddress => write!(f, "__builtin_frame_address"),
            TokenKind::BuiltinReturnAddress => write!(f, "__builtin_return_address"),
            TokenKind::BuiltinAssumeAligned => write!(f, "__builtin_assume_aligned"),

            // GCC Builtins — Checked Arithmetic
            TokenKind::BuiltinAddOverflow => write!(f, "__builtin_add_overflow"),
            TokenKind::BuiltinSubOverflow => write!(f, "__builtin_sub_overflow"),
            TokenKind::BuiltinMulOverflow => write!(f, "__builtin_mul_overflow"),

            // Identifiers and Literals — display as category names
            TokenKind::Identifier(sym) => {
                if *sym == Symbol::EMPTY {
                    write!(f, "identifier")
                } else {
                    // Include the symbol index for debug-level diagnostics.
                    // The diagnostic engine resolves the actual string via the
                    // Interner when formatting the final user-facing message.
                    write!(f, "identifier(#{})", sym.as_u32())
                }
            }
            TokenKind::IntegerLiteral { .. } => write!(f, "integer literal"),
            TokenKind::FloatLiteral { .. } => write!(f, "floating-point literal"),
            TokenKind::StringLiteral { .. } => write!(f, "string literal"),
            TokenKind::CharLiteral { .. } => write!(f, "character literal"),

            // Single-Character Operators and Punctuators
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::Ampersand => write!(f, "&"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::Tilde => write!(f, "~"),
            TokenKind::Exclaim => write!(f, "!"),
            TokenKind::Less => write!(f, "<"),
            TokenKind::Greater => write!(f, ">"),
            TokenKind::Assign => write!(f, "="),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),

            // Multi-Character Operators and Punctuators
            TokenKind::EqualEqual => write!(f, "=="),
            TokenKind::NotEqual => write!(f, "!="),
            TokenKind::LessEqual => write!(f, "<="),
            TokenKind::GreaterEqual => write!(f, ">="),
            TokenKind::LeftShift => write!(f, "<<"),
            TokenKind::RightShift => write!(f, ">>"),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::PlusPlus => write!(f, "++"),
            TokenKind::MinusMinus => write!(f, "--"),
            TokenKind::AmpAmp => write!(f, "&&"),
            TokenKind::PipePipe => write!(f, "||"),
            TokenKind::PlusAssign => write!(f, "+="),
            TokenKind::MinusAssign => write!(f, "-="),
            TokenKind::StarAssign => write!(f, "*="),
            TokenKind::SlashAssign => write!(f, "/="),
            TokenKind::PercentAssign => write!(f, "%="),
            TokenKind::AmpAssign => write!(f, "&="),
            TokenKind::PipeAssign => write!(f, "|="),
            TokenKind::CaretAssign => write!(f, "^="),
            TokenKind::LeftShiftAssign => write!(f, "<<="),
            TokenKind::RightShiftAssign => write!(f, ">>="),
            TokenKind::Ellipsis => write!(f, "..."),
            TokenKind::Hash => write!(f, "#"),
            TokenKind::HashHash => write!(f, "##"),

            // Special Tokens
            TokenKind::Eof => write!(f, "end of file"),
            TokenKind::Error => write!(f, "<error>"),
            TokenKind::Newline => write!(f, "newline"),
            TokenKind::Whitespace => write!(f, "whitespace"),
        }
    }
}

// ===========================================================================
// Token — the fundamental unit of the lexer's output
// ===========================================================================

/// A token produced by the lexer, pairing a `TokenKind` with its source
/// location `Span`.
///
/// `Token` is the primary data structure flowing from the lexer to the
/// preprocessor (for directive detection), and from the preprocessor to
/// the parser (for syntax analysis). Every subsequent pipeline stage
/// carries tokens or AST nodes that reference token spans for diagnostic
/// reporting.
///
/// # Layout
///
/// `Token` is intentionally not `Copy` because `TokenKind` may contain
/// heap-allocated data (e.g., `StringLiteral`'s `Vec<u8>`). It is `Clone`
/// for cases where token duplication is needed (e.g., macro expansion).
#[derive(Clone, Debug)]
pub struct Token {
    /// The kind/type of this token (keyword, literal, operator, etc.).
    pub kind: TokenKind,
    /// The source location span of this token.
    pub span: Span,
}

impl Token {
    /// Creates a new token from a kind and source span.
    #[inline]
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }

    /// Creates an EOF token at a dummy source location.
    ///
    /// Convenience factory for signalling end-of-input without a real
    /// source position. Uses `Span::DUMMY` as the sentinel location.
    #[inline]
    pub fn eof() -> Self {
        Token::new(TokenKind::Eof, Span::DUMMY)
    }

    /// Returns `true` if this token's kind matches the given `kind`.
    ///
    /// Comparison is performed at the **discriminant** level, so data-carrying
    /// variants match regardless of their payload. For example:
    ///
    /// ```ignore
    /// // Matches any Identifier, regardless of which Symbol it carries.
    /// token.is(TokenKind::Identifier(Symbol::EMPTY))
    /// ```
    ///
    /// For exact value comparison, use `token.kind == expected_kind`.
    #[inline]
    pub fn is(&self, kind: TokenKind) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(&kind)
    }

    /// Returns `true` if this token is any keyword (C11 standard, C11
    /// special, GCC extension, or GCC builtin).
    ///
    /// Does NOT match identifiers, literals, operators, or special tokens.
    pub fn is_keyword(&self) -> bool {
        matches!(
            self.kind,
            // C11 Standard Keywords
            TokenKind::Auto
                | TokenKind::Break
                | TokenKind::Case
                | TokenKind::Char
                | TokenKind::Const
                | TokenKind::Continue
                | TokenKind::Default
                | TokenKind::Do
                | TokenKind::Double
                | TokenKind::Else
                | TokenKind::Enum
                | TokenKind::Extern
                | TokenKind::Float
                | TokenKind::For
                | TokenKind::Goto
                | TokenKind::If
                | TokenKind::Inline
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Register
                | TokenKind::Restrict
                | TokenKind::Return
                | TokenKind::Short
                | TokenKind::Signed
                | TokenKind::Sizeof
                | TokenKind::Static
                | TokenKind::Struct
                | TokenKind::Switch
                | TokenKind::Typedef
                | TokenKind::Union
                | TokenKind::Unsigned
                | TokenKind::Void
                | TokenKind::Volatile
                | TokenKind::While
                // C11-Specific Keywords
                | TokenKind::Alignas
                | TokenKind::Alignof
                | TokenKind::Atomic
                | TokenKind::Bool
                | TokenKind::Complex
                | TokenKind::Generic
                | TokenKind::Imaginary
                | TokenKind::Noreturn
                | TokenKind::StaticAssert
                | TokenKind::ThreadLocal
                // GCC Extension Keywords
                | TokenKind::Attribute
                | TokenKind::TypeofKeyword
                | TokenKind::Extension
                | TokenKind::AsmKeyword
                | TokenKind::VolatileGcc
                | TokenKind::InlineGcc
                | TokenKind::SignedGcc
                | TokenKind::ConstGcc
                | TokenKind::RestrictGcc
                | TokenKind::Label
                // GCC Builtins
                | TokenKind::BuiltinVaList
                | TokenKind::BuiltinVaStart
                | TokenKind::BuiltinVaEnd
                | TokenKind::BuiltinVaArg
                | TokenKind::BuiltinVaCopy
                | TokenKind::BuiltinOffsetof
                | TokenKind::BuiltinTypesCompatibleP
                | TokenKind::BuiltinChooseExpr
                | TokenKind::BuiltinConstantP
                | TokenKind::BuiltinExpect
                | TokenKind::BuiltinUnreachable
                | TokenKind::BuiltinTrap
                | TokenKind::BuiltinClz
                | TokenKind::BuiltinCtz
                | TokenKind::BuiltinPopcount
                | TokenKind::BuiltinBswap16
                | TokenKind::BuiltinBswap32
                | TokenKind::BuiltinBswap64
                | TokenKind::BuiltinFfs
                | TokenKind::BuiltinFrameAddress
                | TokenKind::BuiltinReturnAddress
                | TokenKind::BuiltinAssumeAligned
                | TokenKind::BuiltinAddOverflow
                | TokenKind::BuiltinSubOverflow
                | TokenKind::BuiltinMulOverflow
        )
    }

    /// Returns `true` if this token is a type-specifier keyword.
    ///
    /// Type specifiers determine the base type in a declaration. This includes
    /// basic types (`int`, `char`, `void`, etc.), composite type introducers
    /// (`struct`, `union`, `enum`), `typeof`/`__typeof__`, and C11 type
    /// keywords (`_Bool`, `_Complex`, `_Atomic`).
    ///
    /// Note: `_Atomic` can serve as both a type specifier (when followed by
    /// parentheses, e.g., `_Atomic(int)`) and a type qualifier (when used
    /// directly, e.g., `_Atomic int`). This method returns `true` for
    /// `_Atomic` in both roles; the parser disambiguates.
    pub fn is_type_specifier(&self) -> bool {
        matches!(
            self.kind,
            // Basic type specifiers
            TokenKind::Void
                | TokenKind::Char
                | TokenKind::Short
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Float
                | TokenKind::Double
                | TokenKind::Signed
                | TokenKind::Unsigned
                // C11 type specifiers
                | TokenKind::Bool
                | TokenKind::Complex
                | TokenKind::Imaginary
                | TokenKind::Atomic
                // Composite type introducers
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
                // GCC typeof (acts as type specifier)
                | TokenKind::TypeofKeyword
                // GCC __signed__ (equivalent to signed)
                | TokenKind::SignedGcc
        )
    }

    /// Returns the source file ID for this token's location.
    ///
    /// Provides direct access to the span's file identifier without
    /// going through `self.span.file_id` at the call site.
    #[inline]
    pub fn file_id(&self) -> u32 {
        self.span.file_id
    }

    /// Returns the inclusive start byte offset of this token in its source file.
    #[inline]
    pub fn start(&self) -> u32 {
        self.span.start
    }

    /// Returns the exclusive end byte offset of this token in its source file.
    #[inline]
    pub fn end(&self) -> u32 {
        self.span.end
    }

    /// Merges the spans of this token and another, returning a span that
    /// covers both source regions.
    ///
    /// Useful for creating spans for compound AST nodes that span multiple
    /// tokens (e.g., a binary expression spanning from the LHS token to
    /// the RHS token).
    #[inline]
    pub fn merge_span(&self, other: &Token) -> Span {
        Span::merge(self.span, other.span)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}
