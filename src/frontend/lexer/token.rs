//! Token type definitions for the BCC C11 lexer.
//!
//! This module defines the complete token type hierarchy consumed by the lexer,
//! parser, and semantic analyzer. It is the **foundational type file** for the
//! entire frontend pipeline.
//!
//! # Contents
//!
//! - [`Token`] — A token produced by the lexer, carrying its [`TokenKind`] and
//!   source [`Span`].
//! - [`TokenKind`] — Discriminated union of all C11 keywords (44), GCC extension
//!   keywords, GCC builtins (~30), operators/punctuators, identifiers, literals,
//!   and special markers (EOF, Error).
//! - [`NumericBase`] — Radix of a numeric literal (Decimal, Hex, Octal, Binary).
//! - [`IntegerSuffix`] — Type suffix on integer literals (None, U, L, UL, LL, ULL).
//! - [`FloatSuffix`] — Type suffix on float literals (None, F, L).
//! - [`StringPrefix`] — Encoding prefix on string/char literals (None, L, U8, U16, U32).
//! - [`lookup_keyword`] — Maps a scanned identifier string to its keyword
//!   [`TokenKind`] variant, or returns `None` for plain identifiers.
//!
//! # Design Decisions
//!
//! - Every GCC extension keyword and builtin is a distinct [`TokenKind`] variant
//!   (not a plain `Identifier`). The parser depends on exact keyword dispatch.
//! - `__volatile__` maps to [`TokenKind::Volatile`] (the standard C type
//!   qualifier), which also serves as an asm qualifier in `asm volatile`.
//! - `__inline__` maps to [`TokenKind::Inline`] (same variant as C `inline`).
//! - `typeof` and `__typeof__` both map to [`TokenKind::Typeof`].
//! - `asm` and `__asm__` both map to [`TokenKind::Asm`].
//! - Identifier values use [`Symbol`] handles for O(1) comparison.
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library (`std`) and internal
//! crate modules (`crate::common::diagnostics`, `crate::common::string_interner`).

use std::fmt;

use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;

// ---------------------------------------------------------------------------
// NumericBase — radix of a numeric literal
// ---------------------------------------------------------------------------

/// Represents the base (radix) of a numeric literal.
///
/// Used in [`TokenKind::IntegerLiteral`] and [`TokenKind::FloatLiteral`]
/// to record how the literal was written in source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericBase {
    /// Decimal (base 10): `42`, `3.14`
    Decimal,
    /// Hexadecimal (base 16): `0xFF`, `0x1.0p+0`
    Hexadecimal,
    /// Octal (base 8): `0755`
    Octal,
    /// Binary (base 2, GCC extension): `0b1010`
    Binary,
}

// ---------------------------------------------------------------------------
// IntegerSuffix — type suffix on integer literals
// ---------------------------------------------------------------------------

/// Integer type suffix on a numeric literal, per C11 §6.4.4.1.
///
/// All valid combinations of `u`/`U` (unsigned) and `l`/`L`/`ll`/`LL` (long)
/// are represented. The suffix is order-independent: `ul` and `lu` both map
/// to [`IntegerSuffix::UL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerSuffix {
    /// No suffix — type determined by value and base.
    None,
    /// `u` or `U` — unsigned.
    U,
    /// `l` or `L` — long.
    L,
    /// `ul`, `uL`, `Ul`, `UL`, `lu`, `lU`, `Lu`, `LU` — unsigned long.
    UL,
    /// `ll` or `LL` — long long.
    LL,
    /// `ull`, `uLL`, `Ull`, `ULL`, `llu`, `llU`, `LLu`, `LLU` — unsigned long long.
    ULL,
}

// ---------------------------------------------------------------------------
// FloatSuffix — type suffix on floating-point literals
// ---------------------------------------------------------------------------

/// Float type suffix on a numeric literal, per C11 §6.4.4.2.
/// GCC extension: `i` / `j` suffix for imaginary constants (e.g. `1.0i`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSuffix {
    /// No suffix — `double`.
    None,
    /// `f` or `F` — `float`.
    F,
    /// `l` or `L` — `long double`.
    L,
    /// `i` or `j` — `_Complex double` imaginary literal (GCC extension).
    I,
    /// `fi` / `if` / `fj` / `jf` — `_Complex float` imaginary literal.
    FI,
    /// `li` / `il` / `lj` / `jl` — `_Complex long double` imaginary literal.
    LI,
}

// ---------------------------------------------------------------------------
// StringPrefix — encoding prefix on string/char literals
// ---------------------------------------------------------------------------

/// Encoding prefix on a string or character literal, per C11 §6.4.5.
///
/// Determines the element type and encoding of the literal:
/// - `None` → `char *` (byte string)
/// - `L` → `wchar_t *` (wide string)
/// - `U8` → `char *` (UTF-8 string, C11)
/// - `U16` → `char16_t *` (UTF-16, C11)
/// - `U32` → `char32_t *` (UTF-32, C11)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringPrefix {
    /// No prefix: `"..."` or `'...'`.
    None,
    /// `L` prefix: `L"..."` or `L'...'`.
    L,
    /// `u8` prefix: `u8"..."` (string literals only in C11).
    U8,
    /// `u` prefix: `u"..."` or `u'...'` (UTF-16).
    U16,
    /// `U` prefix: `U"..."` or `U'...'` (UTF-32).
    U32,
}

// ---------------------------------------------------------------------------
// TokenKind — the discriminated union of all token types
// ---------------------------------------------------------------------------

/// The type/category of a [`Token`] — a discriminated union covering every
/// lexical element in C11 plus GCC extensions.
///
/// Variants are grouped into:
/// 1. **C11 keywords** (44 total)
/// 2. **GCC extension keywords** (`__attribute__`, `typeof`, `asm`, etc.)
/// 3. **GCC builtins** (~30, each with its own variant for parser dispatch)
/// 4. **Identifiers** (interned via [`Symbol`])
/// 5. **Literals** (integer, float, string, character)
/// 6. **Operators** (arithmetic, bitwise, shift, comparison, logical, assignment)
/// 7. **Punctuators** (member access, ternary, comma, semicolon, ellipsis)
/// 8. **Delimiters** (parentheses, brackets, braces)
/// 9. **Special** (EOF, Error)
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // =======================================================================
    // C11 Keywords — Storage class specifiers
    // =======================================================================
    /// `auto`
    Auto,
    /// `extern`
    Extern,
    /// `register`
    Register,
    /// `static`
    Static,
    /// `typedef`
    Typedef,

    // =======================================================================
    // C11 Keywords — Type specifiers
    // =======================================================================
    /// `void`
    Void,
    /// `char`
    Char,
    /// `short`
    Short,
    /// `int`
    Int,
    /// `long`
    Long,
    /// `float`
    Float,
    /// `double`
    Double,
    /// `signed`
    Signed,
    /// `unsigned`
    Unsigned,

    // =======================================================================
    // C11 Keywords — Type qualifiers
    // =======================================================================
    /// `const`
    Const,
    /// `volatile`
    Volatile,
    /// `restrict`
    Restrict,

    // =======================================================================
    // C11 Keywords — Function specifiers and C11-specific
    // =======================================================================
    /// `inline` (also function specifier; `__inline__` maps here too)
    Inline,
    /// `_Bool`
    Bool,
    /// `_Complex`
    Complex,
    /// `_Atomic`
    Atomic,
    /// `_Alignas`
    Alignas,
    /// `_Alignof`
    Alignof,
    /// `_Generic`
    Generic,
    /// `_Noreturn`
    Noreturn,
    /// `_Static_assert`
    StaticAssert,
    /// `_Thread_local`
    ThreadLocal,

    // =======================================================================
    // C11 Keywords — Control flow
    // =======================================================================
    /// `if`
    If,
    /// `else`
    Else,
    /// `switch`
    Switch,
    /// `case`
    Case,
    /// `default`
    Default,
    /// `while`
    While,
    /// `do`
    Do,
    /// `for`
    For,
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// `return`
    Return,
    /// `goto`
    Goto,

    // =======================================================================
    // C11 Keywords — Aggregate types
    // =======================================================================
    /// `struct`
    Struct,
    /// `union`
    Union,
    /// `enum`
    Enum,

    // =======================================================================
    // C11 Keywords — Operators that are keywords
    // =======================================================================
    /// `sizeof`
    Sizeof,

    // =======================================================================
    // GCC Extension Keywords
    // =======================================================================
    /// `__attribute__`
    Attribute,
    /// `typeof` / `__typeof__`
    Typeof,
    /// `__extension__`
    Extension,
    /// `asm` / `__asm__`
    Asm,
    /// Legacy `AsmVolatile` variant — no longer produced by the lexer.
    /// `__volatile__` now maps to [`TokenKind::Volatile`], the standard C
    /// type qualifier, which also serves as an asm qualifier in `asm volatile`.
    /// Kept for exhaustive-match compatibility; dead-code in practice.
    AsmVolatile,
    /// `__label__` (local label declaration in GCC)
    Label,
    /// `__real__` / `__real` — GCC real-part extraction of `_Complex`.
    RealPart,
    /// `__imag__` / `__imag` — GCC imaginary-part extraction of `_Complex`.
    ImagPart,
    /// `__int128` — GCC 128-bit integer type extension.
    Int128Keyword,
    /// `_Float128` / `__float128` — GCC 128-bit floating-point type.
    Float128Keyword,
    /// `_Float16` — IEC 60559 16-bit float (maps to half precision).
    Float16Keyword,
    /// `_Float32` / `_Float32x` — IEC 60559 32-bit float / extended.
    Float32Keyword,
    /// `_Float64` / `_Float64x` — IEC 60559 64-bit float / extended.
    Float64Keyword,
    /// `__auto_type` (GCC auto type inference)
    AutoType,

    // =======================================================================
    // GCC Builtins — each as its own variant for parser dispatch
    // =======================================================================
    /// `__builtin_va_start`
    BuiltinVaStart,
    /// `__builtin_va_end`
    BuiltinVaEnd,
    /// `__builtin_va_arg`
    BuiltinVaArg,
    /// `__builtin_va_copy`
    BuiltinVaCopy,
    /// `__builtin_offsetof`
    BuiltinOffsetof,
    /// `__builtin_types_compatible_p`
    BuiltinTypesCompatibleP,
    /// `__builtin_choose_expr`
    BuiltinChooseExpr,
    /// `__builtin_constant_p`
    BuiltinConstantP,
    /// `__builtin_expect`
    BuiltinExpect,
    /// `__builtin_unreachable`
    BuiltinUnreachable,
    /// `__builtin_trap`
    BuiltinTrap,
    /// `__builtin_clz`
    BuiltinClz,
    /// `__builtin_clzl`
    BuiltinClzl,
    /// `__builtin_clzll`
    BuiltinClzll,
    /// `__builtin_ctz`
    BuiltinCtz,
    /// `__builtin_ctzl`
    BuiltinCtzl,
    /// `__builtin_ctzll`
    BuiltinCtzll,
    /// `__builtin_popcount`
    BuiltinPopcount,
    /// `__builtin_popcountl`
    BuiltinPopcountl,
    /// `__builtin_popcountll`
    BuiltinPopcountll,
    /// `__builtin_bswap16`
    BuiltinBswap16,
    /// `__builtin_bswap32`
    BuiltinBswap32,
    /// `__builtin_bswap64`
    BuiltinBswap64,
    /// `__builtin_ffs`
    BuiltinFfs,
    /// `__builtin_ffsll`
    BuiltinFfsll,
    /// `__builtin_frame_address`
    BuiltinFrameAddress,
    /// `__builtin_return_address`
    BuiltinReturnAddress,
    /// `__builtin_assume_aligned`
    BuiltinAssumeAligned,
    /// `__builtin_add_overflow`
    BuiltinAddOverflow,
    /// `__builtin_sub_overflow`
    BuiltinSubOverflow,
    /// `__builtin_mul_overflow`
    BuiltinMulOverflow,
    /// `__builtin_add_overflow_p`
    BuiltinAddOverflowP,
    /// `__builtin_sub_overflow_p`
    BuiltinSubOverflowP,
    /// `__builtin_mul_overflow_p`
    BuiltinMulOverflowP,
    /// `__builtin_object_size`
    BuiltinObjectSize,
    /// `__builtin_extract_return_addr`
    BuiltinExtractReturnAddr,
    /// `__builtin_prefetch`
    BuiltinPrefetch,

    // =======================================================================
    // Identifiers
    // =======================================================================
    /// An identifier — the [`Symbol`] is an interned handle for O(1) comparison.
    Identifier(Symbol),

    // =======================================================================
    // Literals
    // =======================================================================
    /// An integer literal with its parsed value, type suffix, and numeric base.
    IntegerLiteral {
        /// The numeric value (may overflow for very large constants).
        value: u64,
        /// The type suffix (`u`, `l`, `ll`, `ul`, `ull`, or none).
        suffix: IntegerSuffix,
        /// The radix the literal was written in.
        base: NumericBase,
    },

    /// A floating-point literal with its raw text, type suffix, and numeric base.
    FloatLiteral {
        /// Raw text representation for full precision (parsed later in sema).
        value: String,
        /// The type suffix (`f`, `l`, or none).
        suffix: FloatSuffix,
        /// Decimal or Hexadecimal (hex floats use `p`/`P` exponent).
        base: NumericBase,
    },

    /// A string literal with its processed content and encoding prefix.
    StringLiteral {
        /// Processed string content (escape sequences resolved, PUA preserved).
        value: String,
        /// Encoding prefix (`L`, `u8`, `u`, `U`, or none).
        prefix: StringPrefix,
    },

    /// A character literal with its code-point value and encoding prefix.
    CharLiteral {
        /// The numeric character code point value.
        value: u32,
        /// Encoding prefix (`L`, `u`, `U`, or none).
        prefix: StringPrefix,
    },

    // =======================================================================
    // Operators — Arithmetic
    // =======================================================================
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,

    // =======================================================================
    // Operators — Bitwise
    // =======================================================================
    /// `&`
    Ampersand,
    /// `|`
    Pipe,
    /// `^`
    Caret,
    /// `~`
    Tilde,

    // =======================================================================
    // Operators — Shift
    // =======================================================================
    /// `<<`
    LessLess,
    /// `>>`
    GreaterGreater,

    // =======================================================================
    // Operators — Comparison
    // =======================================================================
    /// `==`
    EqualEqual,
    /// `!=`
    BangEqual,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,

    // =======================================================================
    // Operators — Logical
    // =======================================================================
    /// `&&`
    AmpAmp,
    /// `||`
    PipePipe,
    /// `!`
    Bang,

    // =======================================================================
    // Operators — Assignment
    // =======================================================================
    /// `=`
    Equal,
    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    StarEqual,
    /// `/=`
    SlashEqual,
    /// `%=`
    PercentEqual,
    /// `&=`
    AmpEqual,
    /// `|=`
    PipeEqual,
    /// `^=`
    CaretEqual,
    /// `<<=`
    LessLessEqual,
    /// `>>=`
    GreaterGreaterEqual,

    // =======================================================================
    // Operators — Increment / Decrement
    // =======================================================================
    /// `++`
    PlusPlus,
    /// `--`
    MinusMinus,

    // =======================================================================
    // Punctuators — Member access
    // =======================================================================
    /// `.`
    Dot,
    /// `->`
    Arrow,

    // =======================================================================
    // Punctuators — Ternary, comma, terminator, variadic
    // =======================================================================
    /// `?`
    Question,
    /// `:`
    Colon,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `...` (variadic, case ranges)
    Ellipsis,

    // =======================================================================
    // Preprocessor tokens (may still appear in the token stream)
    // =======================================================================
    /// `#`
    Hash,
    /// `##`
    HashHash,

    // =======================================================================
    // Delimiters
    // =======================================================================
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `{`
    LeftBrace,
    /// `}`
    RightBrace,

    // =======================================================================
    // Special
    // =======================================================================
    /// End of file marker.
    Eof,
    /// Lexer error recovery token — produced when the scanner encounters
    /// an invalid character or malformed construct.
    Error,
}

// ---------------------------------------------------------------------------
// TokenKind — helper methods
// ---------------------------------------------------------------------------

impl TokenKind {
    /// Returns `true` if this token kind is any C11 or GCC extension keyword.
    ///
    /// This includes storage-class specifiers, type specifiers/qualifiers,
    /// control-flow keywords, aggregate-type keywords, `sizeof`, and all GCC
    /// extension keywords and builtins.
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            // C11 keywords — storage class
            TokenKind::Auto
            | TokenKind::Extern
            | TokenKind::Register
            | TokenKind::Static
            | TokenKind::Typedef
            // C11 keywords — type specifiers
            | TokenKind::Void
            | TokenKind::Char
            | TokenKind::Short
            | TokenKind::Int
            | TokenKind::Long
            | TokenKind::Float
            | TokenKind::Double
            | TokenKind::Signed
            | TokenKind::Unsigned
            // C11 keywords — type qualifiers
            | TokenKind::Const
            | TokenKind::Volatile
            | TokenKind::Restrict
            // C11 keywords — function specifiers and C11-specific
            | TokenKind::Inline
            | TokenKind::Bool
            | TokenKind::Complex
            | TokenKind::Atomic
            | TokenKind::Alignas
            | TokenKind::Alignof
            | TokenKind::Generic
            | TokenKind::Noreturn
            | TokenKind::StaticAssert
            | TokenKind::ThreadLocal
            // C11 keywords — control flow
            | TokenKind::If
            | TokenKind::Else
            | TokenKind::Switch
            | TokenKind::Case
            | TokenKind::Default
            | TokenKind::While
            | TokenKind::Do
            | TokenKind::For
            | TokenKind::Break
            | TokenKind::Continue
            | TokenKind::Return
            | TokenKind::Goto
            // C11 keywords — aggregate types
            | TokenKind::Struct
            | TokenKind::Union
            | TokenKind::Enum
            // C11 keywords — operators that are keywords
            | TokenKind::Sizeof
            // GCC extension keywords
            | TokenKind::Attribute
            | TokenKind::Typeof
            | TokenKind::Extension
            | TokenKind::Asm
            | TokenKind::AsmVolatile
            | TokenKind::Label
            | TokenKind::Int128Keyword
            | TokenKind::Float128Keyword
            | TokenKind::Float16Keyword
            | TokenKind::Float32Keyword
            | TokenKind::Float64Keyword
            | TokenKind::AutoType
            // GCC builtins
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
            | TokenKind::BuiltinClzl
            | TokenKind::BuiltinClzll
            | TokenKind::BuiltinCtz
            | TokenKind::BuiltinCtzl
            | TokenKind::BuiltinCtzll
            | TokenKind::BuiltinPopcount
            | TokenKind::BuiltinPopcountl
            | TokenKind::BuiltinPopcountll
            | TokenKind::BuiltinBswap16
            | TokenKind::BuiltinBswap32
            | TokenKind::BuiltinBswap64
            | TokenKind::BuiltinFfs
            | TokenKind::BuiltinFfsll
            | TokenKind::BuiltinFrameAddress
            | TokenKind::BuiltinReturnAddress
            | TokenKind::BuiltinAssumeAligned
            | TokenKind::BuiltinAddOverflow
            | TokenKind::BuiltinSubOverflow
            | TokenKind::BuiltinMulOverflow
            | TokenKind::BuiltinAddOverflowP
            | TokenKind::BuiltinSubOverflowP
            | TokenKind::BuiltinMulOverflowP
            | TokenKind::BuiltinObjectSize
            | TokenKind::BuiltinExtractReturnAddr
            | TokenKind::BuiltinPrefetch
        )
    }

    /// Returns `true` if this token kind is any literal (integer, float,
    /// string, or character).
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            TokenKind::IntegerLiteral { .. }
                | TokenKind::FloatLiteral { .. }
                | TokenKind::StringLiteral { .. }
                | TokenKind::CharLiteral { .. }
        )
    }

    /// Returns `true` if this token kind is any operator or punctuator
    /// (excluding delimiters and special tokens).
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::Ampersand
                | TokenKind::Pipe
                | TokenKind::Caret
                | TokenKind::Tilde
                | TokenKind::LessLess
                | TokenKind::GreaterGreater
                | TokenKind::EqualEqual
                | TokenKind::BangEqual
                | TokenKind::Less
                | TokenKind::Greater
                | TokenKind::LessEqual
                | TokenKind::GreaterEqual
                | TokenKind::AmpAmp
                | TokenKind::PipePipe
                | TokenKind::Bang
                | TokenKind::Equal
                | TokenKind::PlusEqual
                | TokenKind::MinusEqual
                | TokenKind::StarEqual
                | TokenKind::SlashEqual
                | TokenKind::PercentEqual
                | TokenKind::AmpEqual
                | TokenKind::PipeEqual
                | TokenKind::CaretEqual
                | TokenKind::LessLessEqual
                | TokenKind::GreaterGreaterEqual
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus
                | TokenKind::Dot
                | TokenKind::Arrow
                | TokenKind::Question
                | TokenKind::Colon
                | TokenKind::Comma
                | TokenKind::Semicolon
                | TokenKind::Ellipsis
                | TokenKind::Hash
                | TokenKind::HashHash
        )
    }

    /// Returns `true` if this token kind is an assignment operator
    /// (`=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`).
    pub fn is_assignment_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Equal
                | TokenKind::PlusEqual
                | TokenKind::MinusEqual
                | TokenKind::StarEqual
                | TokenKind::SlashEqual
                | TokenKind::PercentEqual
                | TokenKind::AmpEqual
                | TokenKind::PipeEqual
                | TokenKind::CaretEqual
                | TokenKind::LessLessEqual
                | TokenKind::GreaterGreaterEqual
        )
    }

    /// Returns `true` if this token kind is a comparison operator
    /// (`==`, `!=`, `<`, `>`, `<=`, `>=`).
    pub fn is_comparison_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::EqualEqual
                | TokenKind::BangEqual
                | TokenKind::Less
                | TokenKind::Greater
                | TokenKind::LessEqual
                | TokenKind::GreaterEqual
        )
    }

    /// Returns `true` if this token kind is a unary operator
    /// (`+`, `-`, `!`, `~`, `*`, `&`, `++`, `--`).
    pub fn is_unary_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Bang
                | TokenKind::Tilde
                | TokenKind::Star
                | TokenKind::Ampersand
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus
        )
    }

    /// Returns the C source spelling of a keyword token, or `None` if
    /// this is not a keyword variant.
    ///
    /// For GCC extension keywords, the canonical spelling is returned
    /// (e.g., `__attribute__` rather than a hypothetical shorthand).
    pub fn keyword_str(&self) -> Option<&'static str> {
        match self {
            // C11 keywords — storage class
            TokenKind::Auto => Some("auto"),
            TokenKind::Extern => Some("extern"),
            TokenKind::Register => Some("register"),
            TokenKind::Static => Some("static"),
            TokenKind::Typedef => Some("typedef"),
            // C11 keywords — type specifiers
            TokenKind::Void => Some("void"),
            TokenKind::Char => Some("char"),
            TokenKind::Short => Some("short"),
            TokenKind::Int => Some("int"),
            TokenKind::Long => Some("long"),
            TokenKind::Float => Some("float"),
            TokenKind::Double => Some("double"),
            TokenKind::Signed => Some("signed"),
            TokenKind::Unsigned => Some("unsigned"),
            // C11 keywords — type qualifiers
            TokenKind::Const => Some("const"),
            TokenKind::Volatile => Some("volatile"),
            TokenKind::Restrict => Some("restrict"),
            // C11 keywords — function specifiers and C11-specific
            TokenKind::Inline => Some("inline"),
            TokenKind::Bool => Some("_Bool"),
            TokenKind::Complex => Some("_Complex"),
            TokenKind::Atomic => Some("_Atomic"),
            TokenKind::Alignas => Some("_Alignas"),
            TokenKind::Alignof => Some("_Alignof"),
            TokenKind::Generic => Some("_Generic"),
            TokenKind::Noreturn => Some("_Noreturn"),
            TokenKind::StaticAssert => Some("_Static_assert"),
            TokenKind::ThreadLocal => Some("_Thread_local"),
            // C11 keywords — control flow
            TokenKind::If => Some("if"),
            TokenKind::Else => Some("else"),
            TokenKind::Switch => Some("switch"),
            TokenKind::Case => Some("case"),
            TokenKind::Default => Some("default"),
            TokenKind::While => Some("while"),
            TokenKind::Do => Some("do"),
            TokenKind::For => Some("for"),
            TokenKind::Break => Some("break"),
            TokenKind::Continue => Some("continue"),
            TokenKind::Return => Some("return"),
            TokenKind::Goto => Some("goto"),
            // C11 keywords — aggregate types
            TokenKind::Struct => Some("struct"),
            TokenKind::Union => Some("union"),
            TokenKind::Enum => Some("enum"),
            // C11 keywords — operators that are keywords
            TokenKind::Sizeof => Some("sizeof"),
            // GCC extension keywords
            TokenKind::Attribute => Some("__attribute__"),
            TokenKind::Typeof => Some("typeof"),
            TokenKind::Extension => Some("__extension__"),
            TokenKind::RealPart => Some("__real__"),
            TokenKind::ImagPart => Some("__imag__"),
            TokenKind::Asm => Some("asm"),
            TokenKind::AsmVolatile => Some("__volatile__"),
            TokenKind::Label => Some("__label__"),
            TokenKind::Int128Keyword => Some("__int128"),
            TokenKind::Float128Keyword => Some("_Float128"),
            TokenKind::Float16Keyword => Some("_Float16"),
            TokenKind::Float32Keyword => Some("_Float32"),
            TokenKind::Float64Keyword => Some("_Float64"),
            TokenKind::AutoType => Some("__auto_type"),
            // GCC builtins
            TokenKind::BuiltinVaStart => Some("__builtin_va_start"),
            TokenKind::BuiltinVaEnd => Some("__builtin_va_end"),
            TokenKind::BuiltinVaArg => Some("__builtin_va_arg"),
            TokenKind::BuiltinVaCopy => Some("__builtin_va_copy"),
            TokenKind::BuiltinOffsetof => Some("__builtin_offsetof"),
            TokenKind::BuiltinTypesCompatibleP => Some("__builtin_types_compatible_p"),
            TokenKind::BuiltinChooseExpr => Some("__builtin_choose_expr"),
            TokenKind::BuiltinConstantP => Some("__builtin_constant_p"),
            TokenKind::BuiltinExpect => Some("__builtin_expect"),
            TokenKind::BuiltinUnreachable => Some("__builtin_unreachable"),
            TokenKind::BuiltinTrap => Some("__builtin_trap"),
            TokenKind::BuiltinClz => Some("__builtin_clz"),
            TokenKind::BuiltinClzl => Some("__builtin_clzl"),
            TokenKind::BuiltinClzll => Some("__builtin_clzll"),
            TokenKind::BuiltinCtz => Some("__builtin_ctz"),
            TokenKind::BuiltinCtzl => Some("__builtin_ctzl"),
            TokenKind::BuiltinCtzll => Some("__builtin_ctzll"),
            TokenKind::BuiltinPopcount => Some("__builtin_popcount"),
            TokenKind::BuiltinPopcountl => Some("__builtin_popcountl"),
            TokenKind::BuiltinPopcountll => Some("__builtin_popcountll"),
            TokenKind::BuiltinBswap16 => Some("__builtin_bswap16"),
            TokenKind::BuiltinBswap32 => Some("__builtin_bswap32"),
            TokenKind::BuiltinBswap64 => Some("__builtin_bswap64"),
            TokenKind::BuiltinFfs => Some("__builtin_ffs"),
            TokenKind::BuiltinFfsll => Some("__builtin_ffsll"),
            TokenKind::BuiltinFrameAddress => Some("__builtin_frame_address"),
            TokenKind::BuiltinReturnAddress => Some("__builtin_return_address"),
            TokenKind::BuiltinAssumeAligned => Some("__builtin_assume_aligned"),
            TokenKind::BuiltinAddOverflow => Some("__builtin_add_overflow"),
            TokenKind::BuiltinSubOverflow => Some("__builtin_sub_overflow"),
            TokenKind::BuiltinMulOverflow => Some("__builtin_mul_overflow"),
            TokenKind::BuiltinAddOverflowP => Some("__builtin_add_overflow_p"),
            TokenKind::BuiltinSubOverflowP => Some("__builtin_sub_overflow_p"),
            TokenKind::BuiltinMulOverflowP => Some("__builtin_mul_overflow_p"),
            TokenKind::BuiltinObjectSize => Some("__builtin_object_size"),
            TokenKind::BuiltinExtractReturnAddr => Some("__builtin_extract_return_addr"),
            TokenKind::BuiltinPrefetch => Some("__builtin_prefetch"),
            _ => None,
        }
    }

    /// Returns the string representation of an operator or punctuator token,
    /// or `None` if this is not an operator/punctuator variant.
    pub fn punctuator_str(&self) -> Option<&'static str> {
        match self {
            // Arithmetic
            TokenKind::Plus => Some("+"),
            TokenKind::Minus => Some("-"),
            TokenKind::Star => Some("*"),
            TokenKind::Slash => Some("/"),
            TokenKind::Percent => Some("%"),
            // Bitwise
            TokenKind::Ampersand => Some("&"),
            TokenKind::Pipe => Some("|"),
            TokenKind::Caret => Some("^"),
            TokenKind::Tilde => Some("~"),
            // Shift
            TokenKind::LessLess => Some("<<"),
            TokenKind::GreaterGreater => Some(">>"),
            // Comparison
            TokenKind::EqualEqual => Some("=="),
            TokenKind::BangEqual => Some("!="),
            TokenKind::Less => Some("<"),
            TokenKind::Greater => Some(">"),
            TokenKind::LessEqual => Some("<="),
            TokenKind::GreaterEqual => Some(">="),
            // Logical
            TokenKind::AmpAmp => Some("&&"),
            TokenKind::PipePipe => Some("||"),
            TokenKind::Bang => Some("!"),
            // Assignment
            TokenKind::Equal => Some("="),
            TokenKind::PlusEqual => Some("+="),
            TokenKind::MinusEqual => Some("-="),
            TokenKind::StarEqual => Some("*="),
            TokenKind::SlashEqual => Some("/="),
            TokenKind::PercentEqual => Some("%="),
            TokenKind::AmpEqual => Some("&="),
            TokenKind::PipeEqual => Some("|="),
            TokenKind::CaretEqual => Some("^="),
            TokenKind::LessLessEqual => Some("<<="),
            TokenKind::GreaterGreaterEqual => Some(">>="),
            // Increment/Decrement
            TokenKind::PlusPlus => Some("++"),
            TokenKind::MinusMinus => Some("--"),
            // Member access
            TokenKind::Dot => Some("."),
            TokenKind::Arrow => Some("->"),
            // Ternary, comma, terminator
            TokenKind::Question => Some("?"),
            TokenKind::Colon => Some(":"),
            TokenKind::Comma => Some(","),
            TokenKind::Semicolon => Some(";"),
            TokenKind::Ellipsis => Some("..."),
            // Preprocessor
            TokenKind::Hash => Some("#"),
            TokenKind::HashHash => Some("##"),
            // Delimiters
            TokenKind::LeftParen => Some("("),
            TokenKind::RightParen => Some(")"),
            TokenKind::LeftBracket => Some("["),
            TokenKind::RightBracket => Some("]"),
            TokenKind::LeftBrace => Some("{"),
            TokenKind::RightBrace => Some("}"),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Display for TokenKind
// ---------------------------------------------------------------------------

impl fmt::Display for TokenKind {
    /// Formats the token kind for diagnostic messages and debugging output.
    ///
    /// - Keywords display as their C source spelling (e.g., `auto`, `_Bool`).
    /// - Operators display as their symbol (e.g., `+`, `->`).
    /// - Identifiers display as `<identifier>` (symbol index shown in debug).
    /// - Literals display a brief type description with their value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Try keyword spelling first.
        if let Some(kw) = self.keyword_str() {
            return write!(f, "{}", kw);
        }
        // Try punctuator/operator spelling.
        if let Some(punc) = self.punctuator_str() {
            return write!(f, "{}", punc);
        }
        // Handle remaining data-carrying and special variants.
        match self {
            TokenKind::Identifier(sym) => write!(f, "<identifier:{}>", sym.as_u32()),
            TokenKind::IntegerLiteral {
                value,
                suffix,
                base,
            } => {
                let base_str = match base {
                    NumericBase::Decimal => "",
                    NumericBase::Hexadecimal => "0x",
                    NumericBase::Octal => "0",
                    NumericBase::Binary => "0b",
                };
                let suffix_str = match suffix {
                    IntegerSuffix::None => "",
                    IntegerSuffix::U => "u",
                    IntegerSuffix::L => "l",
                    IntegerSuffix::UL => "ul",
                    IntegerSuffix::LL => "ll",
                    IntegerSuffix::ULL => "ull",
                };
                write!(f, "{}{}{}", base_str, value, suffix_str)
            }
            TokenKind::FloatLiteral { value, suffix, .. } => {
                let suffix_str = match suffix {
                    FloatSuffix::None => "",
                    FloatSuffix::F => "f",
                    FloatSuffix::L => "L",
                    FloatSuffix::I => "i",
                    FloatSuffix::FI => "fi",
                    FloatSuffix::LI => "Li",
                };
                write!(f, "{}{}", value, suffix_str)
            }
            TokenKind::StringLiteral { value, prefix } => {
                let prefix_str = match prefix {
                    StringPrefix::None => "",
                    StringPrefix::L => "L",
                    StringPrefix::U8 => "u8",
                    StringPrefix::U16 => "u",
                    StringPrefix::U32 => "U",
                };
                write!(f, "{}\"{}\"", prefix_str, value)
            }
            TokenKind::CharLiteral { value, prefix } => {
                let prefix_str = match prefix {
                    StringPrefix::None => "",
                    StringPrefix::L => "L",
                    StringPrefix::U8 => "u8",
                    StringPrefix::U16 => "u",
                    StringPrefix::U32 => "U",
                };
                // Display printable ASCII chars as characters, others as code points.
                if *value >= 0x20 && *value < 0x7F {
                    write!(
                        f,
                        "{}'{}'",
                        prefix_str,
                        char::from_u32(*value).unwrap_or('?')
                    )
                } else {
                    write!(f, "{}'{}'", prefix_str, value)
                }
            }
            TokenKind::Eof => write!(f, "<eof>"),
            TokenKind::Error => write!(f, "<error>"),
            // All keyword and punctuator variants are handled above; this arm
            // is structurally unreachable but satisfies exhaustiveness.
            _ => write!(f, "<unknown>"),
        }
    }
}

// ---------------------------------------------------------------------------
// Token — a lexer output token
// ---------------------------------------------------------------------------

/// A token produced by the lexer, carrying its kind and source span.
///
/// Every token in the BCC pipeline carries a [`Span`] for diagnostic reporting
/// and error messages throughout the parser, semantic analyzer, and downstream
/// pipeline stages.
///
/// # Examples
///
/// ```ignore
/// let tok = Token::new(TokenKind::Int, Span::new(0, 10, 13));
/// assert!(tok.is(&TokenKind::Int));
/// assert!(!tok.is_eof());
/// ```
#[derive(Debug, Clone)]
pub struct Token {
    /// The type/category of this token.
    pub kind: TokenKind,
    /// Source location span (file_id, start byte offset, end byte offset).
    pub span: Span,
}

impl Token {
    /// Create a new token with the given kind and source span.
    #[inline]
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }

    /// Check whether this token's variant matches `kind`.
    ///
    /// Uses [`std::mem::discriminant`] for tag-only comparison, which means
    /// that for data-carrying variants (e.g., `Identifier`, `IntegerLiteral`)
    /// only the variant tag is compared — the payload is ignored. This is
    /// the most useful semantic for parser pattern matching.
    ///
    /// For exact value comparison, use `self.kind == *kind` (requires
    /// [`PartialEq`]).
    #[inline]
    pub fn is(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(kind)
    }

    /// Shorthand for checking whether this token is the end-of-file marker.
    #[inline]
    pub fn is_eof(&self) -> bool {
        matches!(self.kind, TokenKind::Eof)
    }
}

// ---------------------------------------------------------------------------
// lookup_keyword — keyword string-to-TokenKind mapping
// ---------------------------------------------------------------------------

/// Look up a keyword by its exact C source spelling.
///
/// Returns `Some(TokenKind)` if the string is a recognized C11 keyword or
/// GCC extension keyword/builtin, or `None` if it is a plain identifier.
///
/// The match is **case-sensitive**: `"int"` is a keyword but `"Int"` is not.
///
/// # Implementation Note
///
/// A plain `match` statement is used. The Rust compiler optimizes large
/// `match`-on-`&str` patterns into efficient lookup structures (typically
/// a trie or jump table), so external crate dependencies (phf, lazy_static)
/// are unnecessary.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(lookup_keyword("int"), Some(TokenKind::Int));
/// assert_eq!(lookup_keyword("__attribute__"), Some(TokenKind::Attribute));
/// assert_eq!(lookup_keyword("foo"), None);
/// ```
pub fn lookup_keyword(s: &str) -> Option<TokenKind> {
    match s {
        // =================================================================
        // C11 keywords (44 total)
        // =================================================================
        "auto" => Some(TokenKind::Auto),
        "break" => Some(TokenKind::Break),
        "case" => Some(TokenKind::Case),
        "char" => Some(TokenKind::Char),
        "const" => Some(TokenKind::Const),
        "continue" => Some(TokenKind::Continue),
        "default" => Some(TokenKind::Default),
        "do" => Some(TokenKind::Do),
        "double" => Some(TokenKind::Double),
        "else" => Some(TokenKind::Else),
        "enum" => Some(TokenKind::Enum),
        "extern" => Some(TokenKind::Extern),
        "float" => Some(TokenKind::Float),
        "for" => Some(TokenKind::For),
        "goto" => Some(TokenKind::Goto),
        "if" => Some(TokenKind::If),
        "inline" => Some(TokenKind::Inline),
        "int" => Some(TokenKind::Int),
        "long" => Some(TokenKind::Long),
        "register" => Some(TokenKind::Register),
        "restrict" => Some(TokenKind::Restrict),
        "return" => Some(TokenKind::Return),
        "short" => Some(TokenKind::Short),
        "signed" => Some(TokenKind::Signed),
        "sizeof" => Some(TokenKind::Sizeof),
        "static" => Some(TokenKind::Static),
        "struct" => Some(TokenKind::Struct),
        "switch" => Some(TokenKind::Switch),
        "typedef" => Some(TokenKind::Typedef),
        "union" => Some(TokenKind::Union),
        "unsigned" => Some(TokenKind::Unsigned),
        "void" => Some(TokenKind::Void),
        "volatile" => Some(TokenKind::Volatile),
        "while" => Some(TokenKind::While),
        // C11-specific keywords (underscore-prefixed)
        "_Alignas" => Some(TokenKind::Alignas),
        "_Alignof" | "__alignof__" | "__alignof" => Some(TokenKind::Alignof),
        "_Atomic" => Some(TokenKind::Atomic),
        "_Bool" => Some(TokenKind::Bool),
        "_Complex" | "__complex__" | "__complex" => Some(TokenKind::Complex),
        "_Generic" => Some(TokenKind::Generic),
        "_Noreturn" => Some(TokenKind::Noreturn),
        "_Static_assert" => Some(TokenKind::StaticAssert),
        "_Thread_local" => Some(TokenKind::ThreadLocal),

        // =================================================================
        // GCC extension keywords
        // =================================================================
        "__attribute__" => Some(TokenKind::Attribute),
        "__attribute" => Some(TokenKind::Attribute),
        // `typeof`, `__typeof`, and `__typeof__` all map to the same variant.
        "typeof" => Some(TokenKind::Typeof),
        "__typeof" => Some(TokenKind::Typeof),
        "__typeof__" => Some(TokenKind::Typeof),
        "__extension__" => Some(TokenKind::Extension),
        "__real__" => Some(TokenKind::RealPart),
        "__real" => Some(TokenKind::RealPart),
        "__imag__" => Some(TokenKind::ImagPart),
        "__imag" => Some(TokenKind::ImagPart),
        // `asm` and `__asm__` both map to the same variant.
        "asm" => Some(TokenKind::Asm),
        "__asm__" | "__asm" => Some(TokenKind::Asm),
        // `__volatile__` is the GCC alternate spelling for `volatile`.
        // Treated as the same `Volatile` token so it works both as a type
        // qualifier (`__volatile__ int x;`) and as an asm qualifier
        // (`asm __volatile__("nop")`).
        "__volatile__" => Some(TokenKind::Volatile),
        // `__inline__` maps to the same `Inline` variant as C `inline`.
        "__inline__" => Some(TokenKind::Inline),
        "__label__" => Some(TokenKind::Label),
        "__int128" => Some(TokenKind::Int128Keyword),
        "__int128_t" => Some(TokenKind::Int128Keyword),
        "_Float128" => Some(TokenKind::Float128Keyword),
        "__float128" => Some(TokenKind::Float128Keyword),
        "_Float16" => Some(TokenKind::Float16Keyword),
        "_Float32" => Some(TokenKind::Float32Keyword),
        "_Float32x" => Some(TokenKind::Float32Keyword),
        "_Float64" => Some(TokenKind::Float64Keyword),
        "_Float64x" => Some(TokenKind::Float64Keyword),
        "__auto_type" => Some(TokenKind::AutoType),
        // GCC alternate spellings for C keywords used in system headers
        "__const" => Some(TokenKind::Const),
        "__const__" => Some(TokenKind::Const),
        "__restrict" => Some(TokenKind::Restrict),
        "__restrict__" => Some(TokenKind::Restrict),
        "__volatile" => Some(TokenKind::Volatile),
        "__signed__" => Some(TokenKind::Signed),
        "__inline" => Some(TokenKind::Inline),

        // =================================================================
        // GCC builtins (29 total)
        // =================================================================
        "__builtin_va_start" => Some(TokenKind::BuiltinVaStart),
        "__builtin_va_end" => Some(TokenKind::BuiltinVaEnd),
        "__builtin_va_arg" => Some(TokenKind::BuiltinVaArg),
        "__builtin_va_copy" => Some(TokenKind::BuiltinVaCopy),
        "__builtin_offsetof" => Some(TokenKind::BuiltinOffsetof),
        "__builtin_types_compatible_p" => Some(TokenKind::BuiltinTypesCompatibleP),
        "__builtin_choose_expr" => Some(TokenKind::BuiltinChooseExpr),
        "__builtin_constant_p" => Some(TokenKind::BuiltinConstantP),
        "__builtin_expect" => Some(TokenKind::BuiltinExpect),
        "__builtin_unreachable" => Some(TokenKind::BuiltinUnreachable),
        "__builtin_trap" => Some(TokenKind::BuiltinTrap),
        "__builtin_clz" => Some(TokenKind::BuiltinClz),
        "__builtin_clzl" => Some(TokenKind::BuiltinClzl),
        "__builtin_clzll" => Some(TokenKind::BuiltinClzll),
        "__builtin_ctz" => Some(TokenKind::BuiltinCtz),
        "__builtin_ctzl" => Some(TokenKind::BuiltinCtzl),
        "__builtin_ctzll" => Some(TokenKind::BuiltinCtzll),
        "__builtin_popcount" => Some(TokenKind::BuiltinPopcount),
        "__builtin_popcountl" => Some(TokenKind::BuiltinPopcountl),
        "__builtin_popcountll" => Some(TokenKind::BuiltinPopcountll),
        "__builtin_bswap16" => Some(TokenKind::BuiltinBswap16),
        "__builtin_bswap32" => Some(TokenKind::BuiltinBswap32),
        "__builtin_bswap64" => Some(TokenKind::BuiltinBswap64),
        "__builtin_ffs" => Some(TokenKind::BuiltinFfs),
        "__builtin_ffsll" => Some(TokenKind::BuiltinFfsll),
        "__builtin_frame_address" => Some(TokenKind::BuiltinFrameAddress),
        "__builtin_return_address" => Some(TokenKind::BuiltinReturnAddress),
        "__builtin_assume_aligned" => Some(TokenKind::BuiltinAssumeAligned),
        "__builtin_add_overflow" => Some(TokenKind::BuiltinAddOverflow),
        "__builtin_sub_overflow" => Some(TokenKind::BuiltinSubOverflow),
        "__builtin_mul_overflow" => Some(TokenKind::BuiltinMulOverflow),
        "__builtin_add_overflow_p" => Some(TokenKind::BuiltinAddOverflowP),
        "__builtin_sub_overflow_p" => Some(TokenKind::BuiltinSubOverflowP),
        "__builtin_mul_overflow_p" => Some(TokenKind::BuiltinMulOverflowP),
        "__builtin_object_size" => Some(TokenKind::BuiltinObjectSize),
        "__builtin_extract_return_addr" => Some(TokenKind::BuiltinExtractReturnAddr),
        "__builtin_prefetch" => Some(TokenKind::BuiltinPrefetch),

        // Not a keyword — the lexer should treat it as a plain identifier.
        _ => None,
    }
}
