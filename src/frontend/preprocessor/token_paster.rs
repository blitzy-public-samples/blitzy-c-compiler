// src/frontend/preprocessor/token_paster.rs
//
// Token pasting (##) and stringification (#) operator implementation for
// the BCC preprocessor. Implements ## token concatenation and # stringifying
// with proper whitespace and token-boundary handling per C11 §6.10.3.
//
// Used by macro_expander.rs during function-like macro body substitution.

use crate::common::diagnostics::DiagnosticEngine;
use crate::common::string_interner::{Interner, Symbol};
use crate::frontend::lexer::token::{
    FloatSuffix, IntegerSuffix, Span, StringPrefix, Token, TokenKind,
};

// ===========================================================================
// Internal helpers — token text reconstruction
// ===========================================================================

/// Reconstructs the textual (source-spelling) representation of a token.
///
/// For identifiers the text is resolved from the interner. For literals the
/// text is reconstructed from the parsed value and affixes. For keywords and
/// operators the `Display` impl provides the correct C spelling.
fn token_text(kind: &TokenKind, interner: &Interner) -> String {
    match kind {
        TokenKind::Identifier(sym) => {
            if *sym == Symbol::EMPTY {
                String::new()
            } else {
                interner.resolve(*sym).to_string()
            }
        }

        // Integer — decimal value + suffix (original radix lost at lex time)
        TokenKind::IntegerLiteral { value, suffix } => {
            format!("{}{}", value, suffix)
        }

        // Float — guarantee a decimal point so re-lexing sees a float
        TokenKind::FloatLiteral { value, suffix } => {
            let s = format!("{}", value);
            let s = if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                format!("{}.0", s)
            } else {
                s
            };
            format!("{}{}", s, suffix)
        }

        // String literal — prefix + quotes + escaped bytes
        TokenKind::StringLiteral { value, prefix } => {
            let mut result = format!("{}", prefix);
            result.push('"');
            for &b in value.iter() {
                push_byte_escaped(&mut result, b, '"');
            }
            result.push('"');
            result
        }

        // Character literal — prefix + quotes + escaped code-point
        TokenKind::CharLiteral { value, prefix } => {
            let mut result = format!("{}", prefix);
            result.push('\'');
            if *value <= 0x7F {
                push_byte_escaped(&mut result, *value as u8, '\'');
            } else if *value <= 0xFFFF {
                result.push_str(&format!("\\u{:04x}", value));
            } else {
                result.push_str(&format!("\\U{:08x}", value));
            }
            result.push('\'');
            result
        }

        // Whitespace / newline — collapse to single space
        TokenKind::Whitespace | TokenKind::Newline => " ".to_string(),

        // EOF / Error — empty (no textual representation)
        TokenKind::Eof | TokenKind::Error => String::new(),

        // Everything else (keywords, operators, punctuators) — Display gives
        // the correct C spelling: "auto", "==", "->", etc.
        other => format!("{}", other),
    }
}

/// Appends a byte to `out` with C-style escape-sequence encoding.
/// `quote_char` is the surrounding quote ('"' for strings, '\'' for chars).
fn push_byte_escaped(out: &mut String, byte: u8, quote_char: char) {
    match byte {
        b'\\' => out.push_str("\\\\"),
        b'"' if quote_char == '"' => out.push_str("\\\""),
        b'\'' if quote_char == '\'' => out.push_str("\\'"),
        b'\n' => out.push_str("\\n"),
        b'\r' => out.push_str("\\r"),
        b'\t' => out.push_str("\\t"),
        b'\0' => out.push_str("\\0"),
        0x07 => out.push_str("\\a"),
        0x08 => out.push_str("\\b"),
        0x0C => out.push_str("\\f"),
        0x0B => out.push_str("\\v"),
        0x20..=0x7E => out.push(byte as char),
        _ => {
            // Non-printable / high bytes → hex escape
            out.push_str(&format!("\\x{:02x}", byte));
        }
    }
}

// ===========================================================================
// Internal helpers — character classification
// ===========================================================================

/// Returns `true` if the byte can begin a C identifier (`[a-zA-Z_]`).
#[inline]
fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

/// Returns `true` if the byte can continue a C identifier (`[a-zA-Z0-9_]`).
#[inline]
fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

// ===========================================================================
// Internal helpers — re-lexing concatenated text
// ===========================================================================

/// Attempts to re-lex a concatenated string as a single preprocessing token.
///
/// Returns `Some(TokenKind)` if the text forms a valid single token, or
/// `None` if it does not (indicating an invalid paste).
fn try_relex(text: &str, interner: &mut Interner) -> Option<TokenKind> {
    if text.is_empty() {
        return None;
    }

    // 1. Exact punctuator match
    if let Some(kind) = try_lex_punctuator(text) {
        return Some(kind);
    }

    let bytes = text.as_bytes();
    let first = bytes[0];

    // 2. Identifier / keyword: [a-zA-Z_][a-zA-Z0-9_]*
    if is_ident_start(first) {
        if bytes.iter().all(|&b| is_ident_continue(b)) {
            return Some(keyword_or_ident(text, interner));
        }
        // Starts like an ident but contains invalid chars → not a token
        return None;
    }

    // 3. Number: starts with digit (or dot-digit handled below)
    if first.is_ascii_digit() {
        if let Some(kind) = try_lex_number(text) {
            return Some(kind);
        }
        // If it's a valid pp-number but not a concrete C number, keep it
        // as an identifier (the later phases will diagnose it).
        if is_pp_number(text) {
            return Some(TokenKind::Identifier(interner.intern(text)));
        }
        return None;
    }

    // 4. Dot-digit float literal
    if first == b'.' && bytes.len() > 1 && bytes[1].is_ascii_digit() {
        if let Some(kind) = try_lex_number(text) {
            return Some(kind);
        }
        if is_pp_number(text) {
            return Some(TokenKind::Identifier(interner.intern(text)));
        }
        return None;
    }

    None
}

/// Checks whether `text` is a valid C preprocessing number (C11 §6.4.8).
///
/// A pp-number: starts with a digit or `.digit`, then contains digits,
/// letters, underscores, dots, and `[eEpP][+-]`.
fn is_pp_number(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let first = bytes[0];
    if first == b'.' {
        if bytes.len() < 2 || !bytes[1].is_ascii_digit() {
            return false;
        }
    } else if !first.is_ascii_digit() {
        return false;
    }
    let mut i = 1;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'.' {
            i += 1;
        } else if (b == b'+' || b == b'-') && i > 0 {
            let prev = bytes[i - 1];
            if prev == b'e' || prev == b'E' || prev == b'p' || prev == b'P' {
                i += 1;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

// ===========================================================================
// Internal helpers — punctuator matching
// ===========================================================================

/// Attempts to match `text` as a known C/GCC punctuator (exact match).
fn try_lex_punctuator(text: &str) -> Option<TokenKind> {
    match text {
        // --- three-character ---
        "<<=" => Some(TokenKind::LeftShiftAssign),
        ">>=" => Some(TokenKind::RightShiftAssign),
        "..." => Some(TokenKind::Ellipsis),
        // --- two-character ---
        "==" => Some(TokenKind::EqualEqual),
        "!=" => Some(TokenKind::NotEqual),
        "<=" => Some(TokenKind::LessEqual),
        ">=" => Some(TokenKind::GreaterEqual),
        "<<" => Some(TokenKind::LeftShift),
        ">>" => Some(TokenKind::RightShift),
        "->" => Some(TokenKind::Arrow),
        "++" => Some(TokenKind::PlusPlus),
        "--" => Some(TokenKind::MinusMinus),
        "&&" => Some(TokenKind::AmpAmp),
        "||" => Some(TokenKind::PipePipe),
        "+=" => Some(TokenKind::PlusAssign),
        "-=" => Some(TokenKind::MinusAssign),
        "*=" => Some(TokenKind::StarAssign),
        "/=" => Some(TokenKind::SlashAssign),
        "%=" => Some(TokenKind::PercentAssign),
        "&=" => Some(TokenKind::AmpAssign),
        "|=" => Some(TokenKind::PipeAssign),
        "^=" => Some(TokenKind::CaretAssign),
        "##" => Some(TokenKind::HashHash),
        // --- single-character ---
        "+" => Some(TokenKind::Plus),
        "-" => Some(TokenKind::Minus),
        "*" => Some(TokenKind::Star),
        "/" => Some(TokenKind::Slash),
        "%" => Some(TokenKind::Percent),
        "&" => Some(TokenKind::Ampersand),
        "|" => Some(TokenKind::Pipe),
        "^" => Some(TokenKind::Caret),
        "~" => Some(TokenKind::Tilde),
        "!" => Some(TokenKind::Exclaim),
        "<" => Some(TokenKind::Less),
        ">" => Some(TokenKind::Greater),
        "=" => Some(TokenKind::Assign),
        "." => Some(TokenKind::Dot),
        "," => Some(TokenKind::Comma),
        ";" => Some(TokenKind::Semicolon),
        ":" => Some(TokenKind::Colon),
        "?" => Some(TokenKind::Question),
        "(" => Some(TokenKind::LeftParen),
        ")" => Some(TokenKind::RightParen),
        "[" => Some(TokenKind::LeftBracket),
        "]" => Some(TokenKind::RightBracket),
        "{" => Some(TokenKind::LeftBrace),
        "}" => Some(TokenKind::RightBrace),
        "#" => Some(TokenKind::Hash),
        _ => None,
    }
}

// ===========================================================================
// Internal helpers — keyword matching
// ===========================================================================

/// Returns the keyword `TokenKind` for `text`, or interns it as an Identifier.
fn keyword_or_ident(text: &str, interner: &mut Interner) -> TokenKind {
    if let Some(kw) = try_keyword(text) {
        kw
    } else {
        TokenKind::Identifier(interner.intern(text))
    }
}

/// Checks if `text` matches a C11 keyword or GCC extension keyword.
fn try_keyword(text: &str) -> Option<TokenKind> {
    match text {
        // --- C11 standard keywords ---
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
        // --- C11 special keywords ---
        "_Alignas" => Some(TokenKind::Alignas),
        "_Alignof" => Some(TokenKind::Alignof),
        "_Atomic" => Some(TokenKind::Atomic),
        "_Bool" => Some(TokenKind::Bool),
        "_Complex" => Some(TokenKind::Complex),
        "_Generic" => Some(TokenKind::Generic),
        "_Imaginary" => Some(TokenKind::Imaginary),
        "_Noreturn" => Some(TokenKind::Noreturn),
        "_Static_assert" => Some(TokenKind::StaticAssert),
        "_Thread_local" => Some(TokenKind::ThreadLocal),
        // --- GCC extension keywords ---
        "__attribute__" | "__attribute" => Some(TokenKind::Attribute),
        "typeof" | "__typeof__" | "__typeof" => Some(TokenKind::TypeofKeyword),
        "__extension__" => Some(TokenKind::Extension),
        "asm" | "__asm__" | "__asm" => Some(TokenKind::AsmKeyword),
        "__volatile__" | "__volatile" => Some(TokenKind::VolatileGcc),
        "__inline__" | "__inline" => Some(TokenKind::InlineGcc),
        "__signed__" | "__signed" => Some(TokenKind::SignedGcc),
        "__const__" | "__const" => Some(TokenKind::ConstGcc),
        "__restrict__" | "__restrict" => Some(TokenKind::RestrictGcc),
        "__label__" => Some(TokenKind::Label),
        // --- GCC builtins ---
        "__builtin_va_list" => Some(TokenKind::BuiltinVaList),
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
        "__builtin_ctz" => Some(TokenKind::BuiltinCtz),
        "__builtin_popcount" => Some(TokenKind::BuiltinPopcount),
        "__builtin_bswap16" => Some(TokenKind::BuiltinBswap16),
        "__builtin_bswap32" => Some(TokenKind::BuiltinBswap32),
        "__builtin_bswap64" => Some(TokenKind::BuiltinBswap64),
        "__builtin_ffs" => Some(TokenKind::BuiltinFfs),
        "__builtin_frame_address" => Some(TokenKind::BuiltinFrameAddress),
        "__builtin_return_address" => Some(TokenKind::BuiltinReturnAddress),
        "__builtin_assume_aligned" => Some(TokenKind::BuiltinAssumeAligned),
        "__builtin_add_overflow" => Some(TokenKind::BuiltinAddOverflow),
        "__builtin_sub_overflow" => Some(TokenKind::BuiltinSubOverflow),
        "__builtin_mul_overflow" => Some(TokenKind::BuiltinMulOverflow),
        _ => None,
    }
}

// ===========================================================================
// Internal helpers — number lexing for re-lex
// ===========================================================================

/// Attempts to parse `text` as a numeric literal (integer or float).
fn try_lex_number(text: &str) -> Option<TokenKind> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || (!bytes[0].is_ascii_digit() && bytes[0] != b'.') {
        return None;
    }

    // Hex prefix 0x / 0X
    if bytes.len() > 2 && bytes[0] == b'0' && (bytes[1] == b'x' || bytes[1] == b'X') {
        return try_lex_hex_number(&text[2..]);
    }

    // Binary prefix 0b / 0B
    if bytes.len() > 2 && bytes[0] == b'0' && (bytes[1] == b'b' || bytes[1] == b'B') {
        return try_lex_binary_number(&text[2..]);
    }

    // Float detection: contains '.', 'e', or 'E' (but not hex)
    let has_dot = text.contains('.');
    let has_exp = text.to_ascii_lowercase().contains('e');

    if has_dot || has_exp {
        return try_lex_decimal_float(text);
    }

    // Decimal or octal integer
    try_lex_decimal_or_octal(text)
}

/// Splits integer suffix from digit body.
fn split_integer_suffix(text: &str) -> (&str, IntegerSuffix) {
    let lower = text.to_ascii_lowercase();
    let len = text.len();
    if len >= 3 {
        let tail3 = &lower[len - 3..];
        if tail3 == "ull" || tail3 == "llu" {
            return (&text[..len - 3], IntegerSuffix::ULL);
        }
    }
    if len >= 2 {
        let tail2 = &lower[len - 2..];
        match tail2 {
            "ul" | "lu" => return (&text[..len - 2], IntegerSuffix::UL),
            "ll" => return (&text[..len - 2], IntegerSuffix::LL),
            _ => {}
        }
    }
    if len >= 1 {
        match lower.as_bytes()[len - 1] {
            b'u' => return (&text[..len - 1], IntegerSuffix::U),
            b'l' => return (&text[..len - 1], IntegerSuffix::L),
            _ => {}
        }
    }
    (text, IntegerSuffix::None)
}

/// Splits float suffix from the float body.
fn split_float_suffix(text: &str) -> (&str, FloatSuffix) {
    if text.is_empty() {
        return (text, FloatSuffix::None);
    }
    match text.as_bytes()[text.len() - 1] {
        b'f' | b'F' => (&text[..text.len() - 1], FloatSuffix::F),
        b'l' | b'L' => (&text[..text.len() - 1], FloatSuffix::L),
        _ => (text, FloatSuffix::None),
    }
}

/// Tries to lex a hex number (after '0x' has been consumed).
fn try_lex_hex_number(after_prefix: &str) -> Option<TokenKind> {
    if after_prefix.is_empty() {
        return None;
    }
    // Hex float: contains '.' or 'p'/'P'
    if after_prefix.contains('.')
        || after_prefix.contains('p')
        || after_prefix.contains('P')
    {
        return try_lex_hex_float(after_prefix);
    }
    let (digits, suffix) = split_integer_suffix(after_prefix);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let value = u128::from_str_radix(digits, 16).ok()?;
    Some(TokenKind::IntegerLiteral { value, suffix })
}

/// Tries to lex a binary number (after '0b' has been consumed).
fn try_lex_binary_number(after_prefix: &str) -> Option<TokenKind> {
    if after_prefix.is_empty() {
        return None;
    }
    let (digits, suffix) = split_integer_suffix(after_prefix);
    if digits.is_empty() || !digits.bytes().all(|b| b == b'0' || b == b'1') {
        return None;
    }
    let value = u128::from_str_radix(digits, 2).ok()?;
    Some(TokenKind::IntegerLiteral { value, suffix })
}

/// Tries to lex a decimal or octal integer.
fn try_lex_decimal_or_octal(text: &str) -> Option<TokenKind> {
    let (digits, suffix) = split_integer_suffix(text);
    if digits.is_empty() {
        return None;
    }
    // Octal: starts with '0' and all digits are 0-7
    if digits.starts_with('0') && digits.len() > 1
        && digits.bytes().all(|b| (b'0'..=b'7').contains(&b))
    {
        let value = u128::from_str_radix(digits, 8).ok()?;
        return Some(TokenKind::IntegerLiteral { value, suffix });
    }
    // Decimal
    if digits.bytes().all(|b| b.is_ascii_digit()) {
        let value: u128 = digits.parse().ok()?;
        return Some(TokenKind::IntegerLiteral { value, suffix });
    }
    None
}

/// Tries to lex a decimal float.
fn try_lex_decimal_float(text: &str) -> Option<TokenKind> {
    let (body, suffix) = split_float_suffix(text);
    if body.is_empty() {
        return None;
    }
    let value: f64 = body.parse().ok()?;
    Some(TokenKind::FloatLiteral { value, suffix })
}

/// Tries to lex a hex float (after '0x'; contains 'p'/'P' exponent).
fn try_lex_hex_float(after_prefix: &str) -> Option<TokenKind> {
    let (body, suffix) = split_float_suffix(after_prefix);
    if body.is_empty() {
        return None;
    }
    // Split on 'p'/'P' for the binary exponent (required for hex floats)
    let p_pos = body.find(['p', 'P'])?;
    let mantissa_str = &body[..p_pos];
    let exp_str = &body[p_pos + 1..];

    // Split mantissa on optional dot
    let (int_part, frac_part) = if let Some(dot_pos) = mantissa_str.find('.') {
        (&mantissa_str[..dot_pos], &mantissa_str[dot_pos + 1..])
    } else {
        (mantissa_str, "")
    };

    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.is_empty() && !int_part.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if !frac_part.is_empty() && !frac_part.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    let int_val = if int_part.is_empty() {
        0u64
    } else {
        u64::from_str_radix(int_part, 16).ok()?
    };

    let frac_val = if frac_part.is_empty() {
        0.0f64
    } else {
        let frac_int = u64::from_str_radix(frac_part, 16).ok()?;
        frac_int as f64 / (16.0f64).powi(frac_part.len() as i32)
    };

    let exponent: i32 = if exp_str.is_empty() {
        0
    } else {
        exp_str.parse().ok()?
    };

    let value = (int_val as f64 + frac_val) * (2.0f64).powi(exponent);
    Some(TokenKind::FloatLiteral { value, suffix })
}

// ===========================================================================
// Internal helpers — parameter lookup & operand resolution
// ===========================================================================

/// Returns the parameter index if `kind` is an Identifier matching a param.
fn find_param_index(kind: &TokenKind, params: &[Symbol]) -> Option<usize> {
    if let TokenKind::Identifier(sym) = kind {
        params.iter().position(|p| p.as_u32() == sym.as_u32())
    } else {
        None
    }
}

/// Resolves a token to its paste-operand token list.
///
/// If the token is a macro parameter reference, returns the corresponding
/// *unexpanded* argument tokens (with whitespace stripped, since pasting
/// operates on the token boundaries). If the argument is empty, returns
/// an empty vec (placemarker). Otherwise returns a one-element vec.
fn resolve_paste_operand(tok: &Token, args: &[Vec<Token>], params: &[Symbol]) -> Vec<Token> {
    if let Some(idx) = find_param_index(&tok.kind, params) {
        if idx < args.len() {
            args[idx]
                .iter()
                .filter(|t| {
                    !matches!(t.kind, TokenKind::Whitespace | TokenKind::Newline)
                })
                .cloned()
                .collect()
        } else {
            // Parameter with no corresponding argument → placemarker
            Vec::new()
        }
    } else {
        vec![tok.clone()]
    }
}

/// Pastes two operand token lists at their boundary.
///
/// The last token of `left` is pasted with the first token of `right`.
/// If either operand is empty (placemarker) the other is returned unchanged.
fn paste_operand_pair(
    left: Vec<Token>,
    right: Vec<Token>,
    interner: &mut Interner,
    diag: &mut DiagnosticEngine,
) -> Vec<Token> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }

    let left_last = &left[left.len() - 1];
    let right_first = &right[0];
    let pasted = paste_tokens(left_last, right_first, interner, diag);

    let mut result = Vec::with_capacity(left.len() + right.len());
    result.extend_from_slice(&left[..left.len() - 1]);
    result.push(pasted);
    if right.len() > 1 {
        result.extend_from_slice(&right[1..]);
    }
    result
}

// ===========================================================================
// Public API — Token pasting (## operator)
// ===========================================================================

/// Concatenates the textual representations of `left` and `right` and
/// re-lexes the result as a single preprocessing token.
///
/// If the result is valid, a token with the merged span is returned.
/// Otherwise a diagnostic warning is emitted and the concatenation is
/// returned as an Identifier token (following GCC's recovery behaviour).
///
/// # Common valid pastes
///
/// | Left       | Right     | Result     |
/// |------------|-----------|------------|
/// | `foo`      | `bar`     | `foobar`   |
/// | `x`        | `1`       | `x1`       |
/// | `1`        | `2`       | `12`       |
/// | `<`        | `=`       | `<=`       |
pub fn paste_tokens(
    left: &Token,
    right: &Token,
    interner: &mut Interner,
    diag: &mut DiagnosticEngine,
) -> Token {
    let left_text = token_text(&left.kind, interner);
    let right_text = token_text(&right.kind, interner);
    let combined = format!("{}{}", left_text, right_text);
    let merged_span = Span::merge(left.span, right.span);

    // Attempt to re-lex the concatenated text as a single token
    if let Some(kind) = try_relex(&combined, interner) {
        return Token::new(kind, merged_span);
    }

    // Invalid paste — emit diagnostic warning matching GCC wording
    diag.warning(
        merged_span,
        format!(
            "pasting \"{}\" and \"{}\" does not give a valid preprocessing token",
            left_text, right_text,
        ),
    );

    // Return the concatenation as an Identifier so compilation can continue
    let sym = interner.intern(&combined);
    Token::new(TokenKind::Identifier(sym), merged_span)
}

/// Processes a macro replacement body, resolving all `##` paste operators.
///
/// Each `##` joins its left and right operands (which may be macro
/// parameters, in which case the **unexpanded** argument tokens are used).
/// Chains such as `a ## b ## c` are evaluated left-to-right. An empty
/// macro argument acts as a placemarker and yields the other operand.
pub fn apply_paste_operators(
    tokens: &[Token],
    args: &[Vec<Token>],
    params: &[Symbol],
    interner: &mut Interner,
    diag: &mut DiagnosticEngine,
) -> Vec<Token> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut i: usize = 0;

    while i < tokens.len() {
        // Case 1: ## at the very start of the replacement list — left is
        // a placemarker (empty).
        if matches!(tokens[i].kind, TokenKind::HashHash) {
            let left: Vec<Token> = Vec::new();
            i += 1;
            // skip whitespace between ## and the right operand
            while i < tokens.len()
                && matches!(tokens[i].kind, TokenKind::Whitespace | TokenKind::Newline)
            {
                i += 1;
            }
            if i < tokens.len() {
                let right = resolve_paste_operand(&tokens[i], args, params);
                let pasted = paste_operand_pair(left, right, interner, diag);
                result.extend(pasted);
                i += 1;
            }
            continue;
        }

        // Look ahead (skipping whitespace) for a following ##.
        let mut lookahead = i + 1;
        while lookahead < tokens.len()
            && matches!(
                tokens[lookahead].kind,
                TokenKind::Whitespace | TokenKind::Newline
            )
        {
            lookahead += 1;
        }

        if lookahead < tokens.len() && matches!(tokens[lookahead].kind, TokenKind::HashHash) {
            // Found `token ## ...` — begin a paste chain (left-to-right).
            let mut current = resolve_paste_operand(&tokens[i], args, params);
            i = lookahead + 1; // advance past the ##

            loop {
                // Skip whitespace after ##
                while i < tokens.len()
                    && matches!(
                        tokens[i].kind,
                        TokenKind::Whitespace | TokenKind::Newline
                    )
                {
                    i += 1;
                }

                if i >= tokens.len() {
                    // ## at end of body — right operand is a placemarker
                    current =
                        paste_operand_pair(current, Vec::new(), interner, diag);
                    break;
                }

                let right = resolve_paste_operand(&tokens[i], args, params);
                current = paste_operand_pair(current, right, interner, diag);
                i += 1;

                // Check for another chained ##
                let mut next_look = i;
                while next_look < tokens.len()
                    && matches!(
                        tokens[next_look].kind,
                        TokenKind::Whitespace | TokenKind::Newline
                    )
                {
                    next_look += 1;
                }
                if next_look < tokens.len()
                    && matches!(tokens[next_look].kind, TokenKind::HashHash)
                {
                    // Continue the chain
                    i = next_look + 1;
                } else {
                    break;
                }
            }

            result.extend(current);
        } else {
            // No ## follows — emit the token as-is.
            result.push(tokens[i].clone());
            i += 1;
        }
    }

    result
}

// ===========================================================================
// Public API — Stringification (# operator)
// ===========================================================================

/// Converts a sequence of tokens (a macro argument) into a string literal.
///
/// Per C11 §6.10.3.2:
/// - Leading and trailing whitespace in the argument is removed.
/// - Each run of whitespace between non-whitespace tokens collapses to a
///   single space.
/// - The source spelling of each token is preserved in the resulting string.
///
/// The `value` field of the returned `StringLiteral` contains the post-escape
/// bytes — i.e. the actual content of the string at runtime. Because we
/// reconstruct source spellings with `token_text` (which already renders
/// escape sequences), the bytes stored are exactly the source spelling.
pub fn stringify(tokens: &[Token], interner: &Interner) -> Token {
    let mut buf = String::new();
    let mut first_span: Option<Span> = None;
    let mut last_span: Option<Span> = None;
    let mut needs_space = false;
    let mut has_content = false;

    for tok in tokens {
        // Skip whitespace; mark that a space is pending
        if matches!(tok.kind, TokenKind::Whitespace | TokenKind::Newline) {
            if has_content {
                needs_space = true;
            }
            continue;
        }

        // Track span coverage
        if first_span.is_none() {
            first_span = Some(tok.span);
        }
        last_span = Some(tok.span);

        // Collapse pending whitespace to a single space
        if needs_space && has_content {
            buf.push(' ');
        }
        needs_space = false;
        has_content = true;

        // Append the token's source-spelling text
        buf.push_str(&token_text(&tok.kind, interner));
    }

    let result_span = match (first_span, last_span) {
        (Some(f), Some(l)) => Span::merge(f, l),
        (Some(s), None) | (None, Some(s)) => s,
        (None, None) => Span::DUMMY,
    };

    Token::new(
        TokenKind::StringLiteral {
            value: buf.into_bytes(),
            prefix: StringPrefix::None,
        },
        result_span,
    )
}

/// Processes a macro replacement body, resolving all `#` stringify operators.
///
/// Each `#` followed by a macro parameter name is replaced by a stringified
/// version of the corresponding **unexpanded** argument. A `#` that is *not*
/// followed by a parameter name triggers a diagnostic error.
pub fn apply_stringify_operators(
    tokens: &[Token],
    args: &[Vec<Token>],
    params: &[Symbol],
    interner: &mut Interner,
    diag: &mut DiagnosticEngine,
) -> Vec<Token> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut i: usize = 0;

    while i < tokens.len() {
        if matches!(tokens[i].kind, TokenKind::Hash) {
            let hash_span = tokens[i].span;

            // Look ahead past whitespace for the operand
            let mut next = i + 1;
            while next < tokens.len()
                && matches!(
                    tokens[next].kind,
                    TokenKind::Whitespace | TokenKind::Newline
                )
            {
                next += 1;
            }

            if next < tokens.len() {
                if let Some(param_idx) = find_param_index(&tokens[next].kind, params) {
                    // Stringify the corresponding unexpanded argument
                    let empty: Vec<Token> = Vec::new();
                    let arg_tokens = if param_idx < args.len() {
                        &args[param_idx]
                    } else {
                        &empty
                    };

                    let stringified = stringify(arg_tokens, interner);
                    let merged = Span::merge(hash_span, tokens[next].span);
                    result.push(Token::new(stringified.kind, merged));

                    i = next + 1;
                    continue;
                }
            }

            // '#' not followed by a macro parameter — error
            diag.error(hash_span, "'#' is not followed by a macro parameter");
            result.push(tokens[i].clone());
            i += 1;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }

    result
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a bare Interner + DiagnosticEngine for tests.
    fn test_env() -> (Interner, DiagnosticEngine) {
        (Interner::new(), DiagnosticEngine::new())
    }

    /// Helper: build an identifier token.
    fn ident(name: &str, interner: &mut Interner) -> Token {
        Token::new(
            TokenKind::Identifier(interner.intern(name)),
            Span::DUMMY,
        )
    }

    /// Helper: build an integer literal token.
    fn int_tok(value: u128) -> Token {
        Token::new(
            TokenKind::IntegerLiteral {
                value,
                suffix: IntegerSuffix::None,
            },
            Span::DUMMY,
        )
    }

    /// Helper: build a single-char operator token.
    fn op_tok(kind: TokenKind) -> Token {
        Token::new(kind, Span::DUMMY)
    }

    // --- paste_tokens tests ---

    #[test]
    fn paste_two_identifiers() {
        let (mut int, mut diag) = test_env();
        let left = ident("foo", &mut int);
        let right = ident("bar", &mut int);
        let result = paste_tokens(&left, &right, &mut int, &mut diag);
        assert!(matches!(result.kind, TokenKind::Identifier(s) if int.resolve(s) == "foobar"));
        assert!(!diag.has_errors());
    }

    #[test]
    fn paste_ident_and_number() {
        let (mut int, mut diag) = test_env();
        let left = ident("x", &mut int);
        let right = int_tok(1);
        let result = paste_tokens(&left, &right, &mut int, &mut diag);
        assert!(matches!(result.kind, TokenKind::Identifier(s) if int.resolve(s) == "x1"));
    }

    #[test]
    fn paste_two_numbers() {
        let (mut int, mut diag) = test_env();
        let left = int_tok(1);
        let right = int_tok(2);
        let result = paste_tokens(&left, &right, &mut int, &mut diag);
        assert!(matches!(result.kind, TokenKind::IntegerLiteral { value: 12, .. }));
    }

    #[test]
    fn paste_operators_to_compound() {
        let (mut int, mut diag) = test_env();
        let left = op_tok(TokenKind::Less);
        let right = op_tok(TokenKind::Assign);
        let result = paste_tokens(&left, &right, &mut int, &mut diag);
        assert!(matches!(result.kind, TokenKind::LessEqual));
    }

    #[test]
    fn paste_invalid_emits_warning() {
        let (mut int, mut diag) = test_env();
        let left = Token::new(
            TokenKind::StringLiteral {
                value: b"a".to_vec(),
                prefix: StringPrefix::None,
            },
            Span::DUMMY,
        );
        let right = Token::new(
            TokenKind::StringLiteral {
                value: b"b".to_vec(),
                prefix: StringPrefix::None,
            },
            Span::DUMMY,
        );
        let _result = paste_tokens(&left, &right, &mut int, &mut diag);
        // Pasting two string literals does not give a valid token — warning emitted
        assert!(diag.warning_count() > 0);
    }

    // --- stringify tests ---

    #[test]
    fn stringify_simple_tokens() {
        let (mut int, _diag) = test_env();
        // Include whitespace tokens between content tokens, as the
        // preprocessor would deliver them.
        let tokens = vec![
            ident("hello", &mut int),
            Token::new(TokenKind::Whitespace, Span::DUMMY),
            op_tok(TokenKind::Comma),
            Token::new(TokenKind::Whitespace, Span::DUMMY),
            ident("world", &mut int),
        ];
        let result = stringify(&tokens, &int);
        if let TokenKind::StringLiteral { value, prefix } = &result.kind {
            assert_eq!(std::str::from_utf8(value).unwrap(), "hello , world");
            assert!(matches!(prefix, StringPrefix::None));
        } else {
            panic!("expected StringLiteral");
        }
    }

    #[test]
    fn stringify_adjacent_tokens_no_space() {
        let (mut int, _diag) = test_env();
        // Without whitespace between, tokens are concatenated directly.
        let tokens = vec![
            ident("hello", &mut int),
            op_tok(TokenKind::Comma),
            ident("world", &mut int),
        ];
        let result = stringify(&tokens, &int);
        if let TokenKind::StringLiteral { value, .. } = &result.kind {
            assert_eq!(std::str::from_utf8(value).unwrap(), "hello,world");
        } else {
            panic!("expected StringLiteral");
        }
    }

    #[test]
    fn stringify_empty_produces_empty_string() {
        let (_int, _diag) = test_env();
        let tokens: Vec<Token> = vec![];
        let int = Interner::new();
        let result = stringify(&tokens, &int);
        if let TokenKind::StringLiteral { value, .. } = &result.kind {
            assert!(value.is_empty());
        } else {
            panic!("expected StringLiteral");
        }
    }

    #[test]
    fn stringify_collapses_whitespace() {
        let (mut int, _diag) = test_env();
        let tokens = vec![
            Token::new(TokenKind::Whitespace, Span::DUMMY),
            ident("a", &mut int),
            Token::new(TokenKind::Whitespace, Span::DUMMY),
            Token::new(TokenKind::Whitespace, Span::DUMMY),
            ident("b", &mut int),
            Token::new(TokenKind::Whitespace, Span::DUMMY),
        ];
        let result = stringify(&tokens, &int);
        if let TokenKind::StringLiteral { value, .. } = &result.kind {
            assert_eq!(std::str::from_utf8(value).unwrap(), "a b");
        } else {
            panic!("expected StringLiteral");
        }
    }
}
