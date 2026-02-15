//! BCC — Blitzy's C Compiler
//!
//! CLI entry point for the BCC compilation toolchain. Handles command-line
//! argument parsing for GCC-compatible flags, spawns a worker thread with
//! a 64 MiB stack (required for deeply nested kernel macro expansions),
//! and orchestrates the full compilation pipeline from preprocessing
//! through code generation and linking.
//!
//! # Usage
//!
//! ```text
//! bcc [options] <input files>
//!
//! Options:
//!   --target=<arch>    Target architecture (x86-64, i686, aarch64, riscv64)
//!   -o <file>          Output file path
//!   -c                 Compile to object file only
//!   -S                 Compile to assembly output
//!   -E                 Preprocess only
//!   -g                 Emit DWARF v4 debug information
//!   -O0/-O1/-O2/-O3   Optimization level (default: -O0)
//!   -fPIC              Generate position-independent code
//!   -shared            Produce shared object
//!   -mretpoline        Enable retpoline mitigation (x86-64 only)
//!   -fcf-protection    Enable CET/IBT protection (x86-64 only)
//!   -I<dir>            Add include search path
//!   -D<macro>[=value]  Define preprocessor macro
//!   -L<dir>            Add library search path
//!   -l<lib>            Link against library
//! ```

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

// Library imports — common infrastructure.
use bcc::common::{DiagnosticEngine, Interner, SourceMap, Target as LibTarget};

// Library imports — frontend pipeline (Phases 1–5).
use bcc::frontend::lexer::token::{Token as LexToken, TokenKind};
use bcc::frontend::parser::Parser;
use bcc::frontend::preprocessor::Preprocessor;
use bcc::frontend::sema::SemanticAnalyzer;

// Library imports — IR middle-end (Phases 6–9).
use bcc::ir::lowering::lower_translation_unit;
use bcc::ir::mem2reg::phi_eliminate::eliminate_phis;
use bcc::ir::mem2reg::promote_allocas_to_registers;
use bcc::passes::PassManager;

// Library imports — backend code generation (Phase 10).
use bcc::backend::generation::{generate_code, CodegenConfig, OutputMode as BackendOutputMode};

/// Maximum recursion depth for the parser and macro expander.
/// Enforced to prevent stack overflow on deeply nested kernel constructs
/// (e.g., Linux kernel macro expansions). See Section 0.7.3.
const MAX_RECURSION_DEPTH: usize = 512;

/// Worker thread stack size: 64 MiB (67,108,864 bytes).
/// Required because deeply nested kernel macro expansions can exhaust the
/// default thread stack. Spawned via `std::thread::Builder::stack_size()`.
const WORKER_STACK_SIZE: usize = 64 * 1024 * 1024;

// =============================================================================
// CLI Enums
// =============================================================================

/// Output mode determined by compilation flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Produce a fully linked executable (default behaviour).
    Executable,
    /// Compile to relocatable object file only (`-c` flag).
    Object,
    /// Compile to assembly text output (`-S` flag).
    Assembly,
    /// Preprocess only, output expanded tokens to stdout (`-E` flag).
    Preprocess,
}

/// Optimization level selected via `-O` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization (default). `-O0`
    O0,
    /// Basic optimizations. `-O1`
    O1,
    /// Standard optimizations. `-O2`
    O2,
    /// Aggressive optimizations. `-O3`
    O3,
}

/// Target architecture for code generation.
///
/// This is the CLI-level representation used for argument parsing and
/// display formatting. It delegates to [`bcc::common::Target`] for the
/// actual target resolution logic, ensuring consistency between the CLI
/// and the library's target-handling infrastructure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    /// x86-64 (AMD64) — 64-bit, LP64 data model.
    X86_64,
    /// i686 (IA-32) — 32-bit, ILP32 data model.
    I686,
    /// AArch64 (ARM64) — 64-bit, LP64 data model.
    AArch64,
    /// RISC-V 64 — 64-bit, LP64D ABI.
    RiscV64,
}

impl TargetArch {
    /// Parse a target architecture from a CLI string.
    ///
    /// Delegates to [`bcc::common::Target::from_str()`] for consistent
    /// target name resolution across the CLI and the library, then maps
    /// back to the CLI enum.
    fn from_str(s: &str) -> Option<Self> {
        LibTarget::from_str(s).map(|t| match t {
            LibTarget::X86_64 => TargetArch::X86_64,
            LibTarget::I686 => TargetArch::I686,
            LibTarget::AArch64 => TargetArch::AArch64,
            LibTarget::RiscV64 => TargetArch::RiscV64,
        })
    }

    /// Detect the host architecture at compile time.
    ///
    /// Delegates to [`bcc::common::Target::host_target()`] for consistent
    /// host detection, then maps back to the CLI enum.
    fn host() -> Self {
        let lib_target = LibTarget::host_target();
        match lib_target {
            LibTarget::X86_64 => TargetArch::X86_64,
            LibTarget::I686 => TargetArch::I686,
            LibTarget::AArch64 => TargetArch::AArch64,
            LibTarget::RiscV64 => TargetArch::RiscV64,
        }
    }
}

impl std::fmt::Display for TargetArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetArch::X86_64 => write!(f, "x86-64"),
            TargetArch::I686 => write!(f, "i686"),
            TargetArch::AArch64 => write!(f, "aarch64"),
            TargetArch::RiscV64 => write!(f, "riscv64"),
        }
    }
}

// =============================================================================
// CompilationContext
// =============================================================================

/// Compilation context assembled from parsed command-line arguments.
///
/// Contains all configuration needed to drive the full compilation pipeline.
/// Constructed by [`parse_args()`] and consumed by [`run_compilation()`].
pub struct CompilationContext {
    /// Target architecture for code generation.
    pub target: TargetArch,
    /// Output mode (executable, object, assembly, preprocess-only).
    pub output_mode: OutputMode,
    /// Explicit output file path (`-o`). `None` means derive from input name.
    pub output_path: Option<PathBuf>,
    /// Input source files to compile.
    pub input_files: Vec<PathBuf>,
    /// Optimization level (O0 through O3).
    pub optimization_level: OptimizationLevel,
    /// Whether to emit DWARF v4 debug information (`-g`).
    pub debug_info: bool,
    /// Whether to generate position-independent code (`-fPIC`).
    pub pic: bool,
    /// Whether to produce a shared object / ET_DYN (`-shared`).
    pub shared: bool,
    /// Whether to enable retpoline indirect branch mitigation (`-mretpoline`, x86-64 only).
    pub retpoline: bool,
    /// Whether to enable CET/IBT control-flow protection (`-fcf-protection`, x86-64 only).
    pub cf_protection: bool,
    /// Include search paths from `-I` flags.
    pub include_paths: Vec<PathBuf>,
    /// Preprocessor macro definitions from `-D` flags (name, optional value).
    pub macro_definitions: Vec<(String, Option<String>)>,
    /// Library search paths from `-L` flags.
    pub library_paths: Vec<PathBuf>,
    /// Libraries to link against from `-l` flags.
    pub libraries: Vec<String>,
    /// Maximum recursion depth for parser and macro expander (default: 512).
    pub recursion_depth_limit: usize,
}

// =============================================================================
// Argument Parsing
// =============================================================================

/// Parse command-line arguments into a [`CompilationContext`].
///
/// Implements a hand-rolled GCC-compatible argument parser without any
/// external crate dependencies (per zero-dependency mandate). Supports both
/// attached (`-Ipath`) and detached (`-I path`) argument forms for flags
/// that accept values.
///
/// # Errors
///
/// Returns a descriptive error string if argument parsing fails (e.g.,
/// unknown flags, missing required arguments, invalid target architecture,
/// security flags used with a non-x86-64 target).
pub fn parse_args(args: &[String]) -> Result<CompilationContext, String> {
    let mut ctx = CompilationContext {
        target: TargetArch::host(),
        output_mode: OutputMode::Executable,
        output_path: None,
        input_files: Vec::new(),
        optimization_level: OptimizationLevel::O0,
        debug_info: false,
        pic: false,
        shared: false,
        retpoline: false,
        cf_protection: false,
        include_paths: Vec::new(),
        macro_definitions: Vec::new(),
        library_paths: Vec::new(),
        libraries: Vec::new(),
        recursion_depth_limit: MAX_RECURSION_DEPTH,
    };

    let mut i = 1; // Skip argv[0] (program name).
    while i < args.len() {
        let arg = &args[i];

        // ── Target architecture ───────────────────────────────────────
        if let Some(target_str) = arg.strip_prefix("--target=") {
            ctx.target = TargetArch::from_str(target_str)
                .ok_or_else(|| format!("unknown target architecture: '{}'", target_str))?;
        } else if arg == "--target" {
            i += 1;
            if i >= args.len() {
                return Err("--target requires an argument".to_string());
            }
            ctx.target = TargetArch::from_str(&args[i])
                .ok_or_else(|| format!("unknown target architecture: '{}'", &args[i]))?;
        }
        // ── Output file ───────────────────────────────────────────────
        else if arg == "-o" {
            i += 1;
            if i >= args.len() {
                return Err("-o requires an output file path".to_string());
            }
            ctx.output_path = Some(PathBuf::from(&args[i]));
        }
        // ── Compilation mode flags ────────────────────────────────────
        else if arg == "-c" {
            ctx.output_mode = OutputMode::Object;
        } else if arg == "-S" {
            ctx.output_mode = OutputMode::Assembly;
        } else if arg == "-E" {
            ctx.output_mode = OutputMode::Preprocess;
        }
        // ── Debug info ────────────────────────────────────────────────
        else if arg == "-g" {
            ctx.debug_info = true;
        }
        // ── Optimization levels ───────────────────────────────────────
        else if arg == "-O0" {
            ctx.optimization_level = OptimizationLevel::O0;
        } else if arg == "-O1" || arg == "-O" {
            ctx.optimization_level = OptimizationLevel::O1;
        } else if arg == "-O2" {
            ctx.optimization_level = OptimizationLevel::O2;
        } else if arg == "-O3" {
            ctx.optimization_level = OptimizationLevel::O3;
        }
        // ── PIC / shared ─────────────────────────────────────────────
        else if arg == "-fPIC" || arg == "-fpic" {
            ctx.pic = true;
        } else if arg == "-shared" {
            ctx.shared = true;
            // -shared implies -fPIC for all code in the shared object.
            ctx.pic = true;
        }
        // ── Security mitigations (x86-64 only) ───────────────────────
        else if arg == "-mretpoline" {
            ctx.retpoline = true;
        } else if arg == "-fcf-protection" {
            ctx.cf_protection = true;
        }
        // ── Include paths ─────────────────────────────────────────────
        else if let Some(rest) = arg.strip_prefix("-I") {
            let path = if !rest.is_empty() {
                rest // Attached form: -I/usr/include
            } else {
                i += 1; // Detached form: -I /usr/include
                if i >= args.len() {
                    return Err("-I requires a directory path".to_string());
                }
                &args[i]
            };
            ctx.include_paths.push(PathBuf::from(path));
        }
        // ── Macro definitions ─────────────────────────────────────────
        else if let Some(rest) = arg.strip_prefix("-D") {
            let def = if !rest.is_empty() {
                rest // Attached form: -DFOO=bar
            } else {
                i += 1; // Detached form: -D FOO=bar
                if i >= args.len() {
                    return Err("-D requires a macro definition".to_string());
                }
                &args[i]
            };
            if let Some(eq_pos) = def.find('=') {
                ctx.macro_definitions.push((
                    def[..eq_pos].to_string(),
                    Some(def[eq_pos + 1..].to_string()),
                ));
            } else {
                ctx.macro_definitions.push((def.to_string(), None));
            }
        }
        // ── Library search paths ──────────────────────────────────────
        else if let Some(rest) = arg.strip_prefix("-L") {
            let path = if !rest.is_empty() {
                rest
            } else {
                i += 1;
                if i >= args.len() {
                    return Err("-L requires a directory path".to_string());
                }
                &args[i]
            };
            ctx.library_paths.push(PathBuf::from(path));
        }
        // ── Libraries to link ─────────────────────────────────────────
        else if let Some(rest) = arg.strip_prefix("-l") {
            let lib = if !rest.is_empty() {
                rest
            } else {
                i += 1;
                if i >= args.len() {
                    return Err("-l requires a library name".to_string());
                }
                &args[i]
            };
            ctx.libraries.push(lib.to_string());
        }
        // ── Informational flags ───────────────────────────────────────
        else if arg == "--help" || arg == "-h" {
            print_usage();
            process::exit(0);
        } else if arg == "--version" || arg == "-v" {
            println!("bcc 0.1.0");
            process::exit(0);
        }
        // ── Ignore common GCC flags for compatibility ─────────────────
        else if arg == "-w" || arg == "-pipe" || arg.starts_with("-W") {
            // Silently ignored for GCC command-line compatibility.
        }
        // ── Unknown flag ──────────────────────────────────────────────
        else if arg.starts_with('-') && arg != "-" {
            return Err(format!("unrecognized command-line option: '{}'", arg));
        }
        // ── Positional argument: input source file ────────────────────
        else {
            ctx.input_files.push(PathBuf::from(arg));
        }

        i += 1;
    }

    // ── Post-parse validation ─────────────────────────────────────────
    if ctx.input_files.is_empty() {
        return Err("no input files".to_string());
    }

    // Security mitigation flags are x86-64 only.
    if ctx.retpoline && ctx.target != TargetArch::X86_64 {
        return Err(format!(
            "-mretpoline is only supported for x86-64 target, not {}",
            ctx.target
        ));
    }
    if ctx.cf_protection && ctx.target != TargetArch::X86_64 {
        return Err(format!(
            "-fcf-protection is only supported for x86-64 target, not {}",
            ctx.target
        ));
    }

    Ok(ctx)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Print comprehensive usage information to stderr.
fn print_usage() {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "Usage: bcc [options] <input files>");
    let _ = writeln!(stderr);
    let _ = writeln!(stderr, "Options:");
    let _ = writeln!(
        stderr,
        "  --target=<arch>    Target architecture (x86-64, i686, aarch64, riscv64)"
    );
    let _ = writeln!(stderr, "  -o <file>          Output file path");
    let _ = writeln!(stderr, "  -c                 Compile to object file only");
    let _ = writeln!(stderr, "  -S                 Compile to assembly output");
    let _ = writeln!(stderr, "  -E                 Preprocess only");
    let _ = writeln!(
        stderr,
        "  -g                 Emit DWARF v4 debug information"
    );
    let _ = writeln!(
        stderr,
        "  -O0/-O1/-O2/-O3   Optimization level (default: -O0)"
    );
    let _ = writeln!(
        stderr,
        "  -fPIC              Generate position-independent code"
    );
    let _ = writeln!(stderr, "  -shared            Produce shared object (ET_DYN)");
    let _ = writeln!(
        stderr,
        "  -mretpoline        Enable retpoline mitigation (x86-64 only)"
    );
    let _ = writeln!(
        stderr,
        "  -fcf-protection    Enable CET/IBT protection (x86-64 only)"
    );
    let _ = writeln!(stderr, "  -I<dir>            Add include search path");
    let _ = writeln!(
        stderr,
        "  -D<macro>[=value]  Define preprocessor macro"
    );
    let _ = writeln!(stderr, "  -L<dir>            Add library search path");
    let _ = writeln!(stderr, "  -l<lib>            Link against library");
    let _ = writeln!(stderr, "  --help, -h         Show this help message");
    let _ = writeln!(stderr, "  --version, -v      Show version");
}

/// Derive the default output file path from an input file and output mode.
///
/// For `-c`: `foo.c` → `foo.o`  
/// For `-S`: `foo.c` → `foo.s`  
/// For executable: `foo.c` → `a.out`
fn derive_output_path(input: &Path, mode: OutputMode) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    match mode {
        OutputMode::Executable => PathBuf::from("a.out"),
        OutputMode::Object => PathBuf::from(stem).with_extension("o"),
        OutputMode::Assembly => PathBuf::from(stem).with_extension("s"),
        OutputMode::Preprocess => {
            // -E mode writes to stdout; this path is not used.
            PathBuf::from("-")
        }
    }
}

/// Convert a CLI [`TargetArch`] to the library [`LibTarget`] enum.
///
/// This bridge function translates between the CLI-level target representation
/// and the library's internal target type used throughout the compilation
/// pipeline.
fn to_lib_target(arch: TargetArch) -> LibTarget {
    match arch {
        TargetArch::X86_64 => LibTarget::X86_64,
        TargetArch::I686 => LibTarget::I686,
        TargetArch::AArch64 => LibTarget::AArch64,
        TargetArch::RiscV64 => LibTarget::RiscV64,
    }
}

/// Convert local [`OutputMode`] to backend [`BackendOutputMode`] for
/// the code generation configuration.
fn to_backend_output_mode(mode: OutputMode) -> BackendOutputMode {
    match mode {
        OutputMode::Executable => BackendOutputMode::Executable,
        OutputMode::Object => BackendOutputMode::Object,
        OutputMode::Assembly => BackendOutputMode::Assembly,
        OutputMode::Preprocess => BackendOutputMode::PreprocessOnly,
    }
}

/// Convert [`OptimizationLevel`] to a `u32` for the backend's
/// [`CodegenConfig`].
fn opt_level_to_u32(level: OptimizationLevel) -> u32 {
    match level {
        OptimizationLevel::O0 => 0,
        OptimizationLevel::O1 => 1,
        OptimizationLevel::O2 => 2,
        OptimizationLevel::O3 => 3,
    }
}

/// Configure a preprocessor instance with system paths, user-specified
/// include paths, and macro definitions from the compilation context.
///
/// This shared configuration logic is used by both the preprocess-only
/// path (`-E`) and the full compilation pipeline to ensure consistent
/// preprocessing behaviour.
fn configure_preprocessor(pp: &mut Preprocessor, ctx: &CompilationContext) {
    // ── System include paths ──────────────────────────────────────────
    pp.add_include_path(PathBuf::from("/usr/include"));
    pp.add_include_path(PathBuf::from("/usr/local/include"));

    // Architecture-specific system include paths.
    match ctx.target {
        TargetArch::X86_64 => {
            pp.add_include_path(PathBuf::from("/usr/include/x86_64-linux-gnu"));
        }
        TargetArch::I686 => {
            pp.add_include_path(PathBuf::from("/usr/include/i386-linux-gnu"));
        }
        TargetArch::AArch64 => {
            pp.add_include_path(PathBuf::from("/usr/include/aarch64-linux-gnu"));
        }
        TargetArch::RiscV64 => {
            pp.add_include_path(PathBuf::from("/usr/include/riscv64-linux-gnu"));
        }
    }

    // GCC internal include paths for compiler builtins (stdarg.h,
    // stddef.h, etc.). Scan for the latest installed version.
    if let Ok(entries) = std::fs::read_dir("/usr/lib/gcc/x86_64-linux-gnu/") {
        let mut versions: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        versions.sort();
        if let Some(latest) = versions.last() {
            pp.add_include_path(PathBuf::from(format!(
                "/usr/lib/gcc/x86_64-linux-gnu/{}/include",
                latest
            )));
        }
    }

    // ── User include paths from -I flags ──────────────────────────────
    for path in &ctx.include_paths {
        pp.add_user_include_path(path.clone());
    }

    // ── Macro definitions from -D flags ───────────────────────────────
    for (name, value) in &ctx.macro_definitions {
        if let Some(val) = value {
            pp.add_define(&format!("{}={}", name, val));
        } else {
            pp.add_define(name);
        }
    }

    // Set the maximum recursion depth from the compilation context.
    // This limits both macro expansion depth and include nesting depth,
    // preventing stack overflow on deeply nested kernel constructs.
    pp.max_recursion_depth = ctx.recursion_depth_limit as u32;
}

/// Convert a preprocessor token to its textual representation for `-E` output.
///
/// Handles all token kinds including identifiers (resolved through the
/// string interner), numeric/string/char literals, keywords, operators,
/// and punctuation.  Every [`TokenKind`] variant is covered explicitly so
/// that the compiler will flag any future additions as non-exhaustive.
fn token_to_text(token: &LexToken, interner: &Interner) -> String {
    use bcc::frontend::lexer::token::{CharPrefix, FloatSuffix, StringPrefix};

    match &token.kind {
        // ── Identifiers ────────────────────────────────────────────────
        TokenKind::Identifier(sym) => interner.resolve(*sym).to_string(),

        // ── Numeric literals ───────────────────────────────────────────
        // Integer literals are formatted as their decimal value followed by
        // the parsed suffix (u, l, ul, ll, ull).
        TokenKind::IntegerLiteral { value, suffix } => {
            format!("{}{}", value, suffix)
        }
        // Floating-point literals are formatted in full precision.
        TokenKind::FloatLiteral { value, suffix } => {
            // Use enough decimal precision to round-trip an f64.
            let base = if value.fract() == 0.0 && !value.is_nan() && !value.is_infinite() {
                format!("{:.1}", value)
            } else {
                format!("{}", value)
            };
            match suffix {
                FloatSuffix::None => base,
                FloatSuffix::F => format!("{}f", base),
                FloatSuffix::L => format!("{}l", base),
            }
        }

        // ── String and character literals ───────────────────────────────
        TokenKind::StringLiteral { value, prefix } => {
            // Reconstruct the source-level string literal by escaping
            // non-printable bytes and preserving PUA-encoded bytes.
            let mut s = String::new();
            match prefix {
                StringPrefix::None => {}
                StringPrefix::L => s.push('L'),
                StringPrefix::U8 => { s.push_str("u8"); }
                StringPrefix::SmallU => s.push('u'),
                StringPrefix::BigU => s.push('U'),
            }
            s.push('"');
            for &b in value.iter() {
                match b {
                    b'\\' => s.push_str("\\\\"),
                    b'"' => s.push_str("\\\""),
                    b'\n' => s.push_str("\\n"),
                    b'\r' => s.push_str("\\r"),
                    b'\t' => s.push_str("\\t"),
                    b'\0' => s.push_str("\\0"),
                    0x20..=0x7e => s.push(b as char),
                    _ => {
                        // Non-printable / high bytes: emit hex escape.
                        s.push_str(&format!("\\x{:02x}", b));
                    }
                }
            }
            s.push('"');
            s
        }
        TokenKind::CharLiteral { value, prefix } => {
            let mut s = String::new();
            match prefix {
                CharPrefix::None => {}
                CharPrefix::L => s.push('L'),
                CharPrefix::SmallU => s.push('u'),
                CharPrefix::BigU => s.push('U'),
            }
            s.push('\'');
            let c = *value;
            if c < 0x80 {
                match c as u8 {
                    b'\\' => s.push_str("\\\\"),
                    b'\'' => s.push_str("\\'"),
                    b'\n' => s.push_str("\\n"),
                    b'\r' => s.push_str("\\r"),
                    b'\t' => s.push_str("\\t"),
                    b'\0' => s.push_str("\\0"),
                    0x20..=0x7e => s.push(c as u8 as char),
                    _ => s.push_str(&format!("\\x{:02x}", c)),
                }
            } else {
                // Multi-byte / wide character — emit hex escape.
                s.push_str(&format!("\\x{:02x}", c));
            }
            s.push('\'');
            s
        }

        // ── C11 Standard Keywords ──────────────────────────────────────
        TokenKind::Auto => "auto".into(),
        TokenKind::Break => "break".into(),
        TokenKind::Case => "case".into(),
        TokenKind::Char => "char".into(),
        TokenKind::Const => "const".into(),
        TokenKind::Continue => "continue".into(),
        TokenKind::Default => "default".into(),
        TokenKind::Do => "do".into(),
        TokenKind::Double => "double".into(),
        TokenKind::Else => "else".into(),
        TokenKind::Enum => "enum".into(),
        TokenKind::Extern => "extern".into(),
        TokenKind::Float => "float".into(),
        TokenKind::For => "for".into(),
        TokenKind::Goto => "goto".into(),
        TokenKind::If => "if".into(),
        TokenKind::Inline => "inline".into(),
        TokenKind::Int => "int".into(),
        TokenKind::Long => "long".into(),
        TokenKind::Register => "register".into(),
        TokenKind::Restrict => "restrict".into(),
        TokenKind::Return => "return".into(),
        TokenKind::Short => "short".into(),
        TokenKind::Signed => "signed".into(),
        TokenKind::Sizeof => "sizeof".into(),
        TokenKind::Static => "static".into(),
        TokenKind::Struct => "struct".into(),
        TokenKind::Switch => "switch".into(),
        TokenKind::Typedef => "typedef".into(),
        TokenKind::Union => "union".into(),
        TokenKind::Unsigned => "unsigned".into(),
        TokenKind::Void => "void".into(),
        TokenKind::Volatile => "volatile".into(),
        TokenKind::While => "while".into(),

        // ── C11-Specific Keywords ──────────────────────────────────────
        TokenKind::Alignas => "_Alignas".into(),
        TokenKind::Alignof => "_Alignof".into(),
        TokenKind::Atomic => "_Atomic".into(),
        TokenKind::Bool => "_Bool".into(),
        TokenKind::Complex => "_Complex".into(),
        TokenKind::Generic => "_Generic".into(),
        TokenKind::Imaginary => "_Imaginary".into(),
        TokenKind::Noreturn => "_Noreturn".into(),
        TokenKind::StaticAssert => "_Static_assert".into(),
        TokenKind::ThreadLocal => "_Thread_local".into(),

        // ── GCC Extension Keywords ─────────────────────────────────────
        TokenKind::Attribute => "__attribute__".into(),
        TokenKind::TypeofKeyword => "typeof".into(),
        TokenKind::Extension => "__extension__".into(),
        TokenKind::AsmKeyword => "asm".into(),
        TokenKind::VolatileGcc => "__volatile__".into(),
        TokenKind::InlineGcc => "__inline__".into(),
        TokenKind::SignedGcc => "__signed__".into(),
        TokenKind::ConstGcc => "__const__".into(),
        TokenKind::RestrictGcc => "__restrict__".into(),
        TokenKind::Label => "__label__".into(),

        // ── GCC Builtins — Variadic Argument Support ───────────────────
        TokenKind::BuiltinVaList => "__builtin_va_list".into(),
        TokenKind::BuiltinVaStart => "__builtin_va_start".into(),
        TokenKind::BuiltinVaEnd => "__builtin_va_end".into(),
        TokenKind::BuiltinVaArg => "__builtin_va_arg".into(),
        TokenKind::BuiltinVaCopy => "__builtin_va_copy".into(),

        // ── GCC Builtins — Type Introspection ──────────────────────────
        TokenKind::BuiltinOffsetof => "__builtin_offsetof".into(),
        TokenKind::BuiltinTypesCompatibleP => "__builtin_types_compatible_p".into(),
        TokenKind::BuiltinChooseExpr => "__builtin_choose_expr".into(),
        TokenKind::BuiltinConstantP => "__builtin_constant_p".into(),

        // ── GCC Builtins — Branch Prediction & Control Flow ────────────
        TokenKind::BuiltinExpect => "__builtin_expect".into(),
        TokenKind::BuiltinUnreachable => "__builtin_unreachable".into(),
        TokenKind::BuiltinTrap => "__builtin_trap".into(),

        // ── GCC Builtins — Bit Manipulation ────────────────────────────
        TokenKind::BuiltinClz => "__builtin_clz".into(),
        TokenKind::BuiltinCtz => "__builtin_ctz".into(),
        TokenKind::BuiltinPopcount => "__builtin_popcount".into(),

        // ── GCC Builtins — Byte Swap ───────────────────────────────────
        TokenKind::BuiltinBswap16 => "__builtin_bswap16".into(),
        TokenKind::BuiltinBswap32 => "__builtin_bswap32".into(),
        TokenKind::BuiltinBswap64 => "__builtin_bswap64".into(),

        // ── GCC Builtins — Miscellaneous ───────────────────────────────
        TokenKind::BuiltinFfs => "__builtin_ffs".into(),
        TokenKind::BuiltinFrameAddress => "__builtin_frame_address".into(),
        TokenKind::BuiltinReturnAddress => "__builtin_return_address".into(),
        TokenKind::BuiltinAssumeAligned => "__builtin_assume_aligned".into(),

        // ── GCC Builtins — Checked Arithmetic ──────────────────────────
        TokenKind::BuiltinAddOverflow => "__builtin_add_overflow".into(),
        TokenKind::BuiltinSubOverflow => "__builtin_sub_overflow".into(),
        TokenKind::BuiltinMulOverflow => "__builtin_mul_overflow".into(),

        // ── Single-Character Operators & Punctuators ───────────────────
        TokenKind::Plus => "+".into(),
        TokenKind::Minus => "-".into(),
        TokenKind::Star => "*".into(),
        TokenKind::Slash => "/".into(),
        TokenKind::Percent => "%".into(),
        TokenKind::Ampersand => "&".into(),
        TokenKind::Pipe => "|".into(),
        TokenKind::Caret => "^".into(),
        TokenKind::Tilde => "~".into(),
        TokenKind::Exclaim => "!".into(),
        TokenKind::Less => "<".into(),
        TokenKind::Greater => ">".into(),
        TokenKind::Assign => "=".into(),
        TokenKind::Dot => ".".into(),
        TokenKind::Comma => ",".into(),
        TokenKind::Semicolon => ";".into(),
        TokenKind::Colon => ":".into(),
        TokenKind::Question => "?".into(),
        TokenKind::LeftParen => "(".into(),
        TokenKind::RightParen => ")".into(),
        TokenKind::LeftBracket => "[".into(),
        TokenKind::RightBracket => "]".into(),
        TokenKind::LeftBrace => "{".into(),
        TokenKind::RightBrace => "}".into(),

        // ── Multi-Character Operators & Punctuators ────────────────────
        TokenKind::EqualEqual => "==".into(),
        TokenKind::NotEqual => "!=".into(),
        TokenKind::LessEqual => "<=".into(),
        TokenKind::GreaterEqual => ">=".into(),
        TokenKind::LeftShift => "<<".into(),
        TokenKind::RightShift => ">>".into(),
        TokenKind::Arrow => "->".into(),
        TokenKind::PlusPlus => "++".into(),
        TokenKind::MinusMinus => "--".into(),
        TokenKind::AmpAmp => "&&".into(),
        TokenKind::PipePipe => "||".into(),
        TokenKind::PlusAssign => "+=".into(),
        TokenKind::MinusAssign => "-=".into(),
        TokenKind::StarAssign => "*=".into(),
        TokenKind::SlashAssign => "/=".into(),
        TokenKind::PercentAssign => "%=".into(),
        TokenKind::AmpAssign => "&=".into(),
        TokenKind::PipeAssign => "|=".into(),
        TokenKind::CaretAssign => "^=".into(),
        TokenKind::LeftShiftAssign => "<<=".into(),
        TokenKind::RightShiftAssign => ">>=".into(),
        TokenKind::Ellipsis => "...".into(),
        TokenKind::Hash => "#".into(),
        TokenKind::HashHash => "##".into(),

        // ── Special Tokens ─────────────────────────────────────────────
        // Whitespace, newlines, EOF, and error recovery tokens are handled
        // by the caller (run_preprocess) or produce empty output.
        TokenKind::Eof | TokenKind::Newline | TokenKind::Whitespace => String::new(),
        TokenKind::Error => String::new(),
    }
}

// =============================================================================
// Preprocess-Only Pipeline (-E)
// =============================================================================

/// Run the preprocessing pipeline and emit expanded tokens to stdout.
///
/// This implements the `-E` flag behaviour: run Phases 1–2 of the
/// compilation pipeline (trigraph replacement, line splicing, directive
/// processing, macro expansion) and write the resulting token stream
/// to stdout.
///
/// # Returns
///
/// `true` on successful preprocessing, `false` on any error.
fn run_preprocess(ctx: &CompilationContext, input_path: &Path) -> bool {
    let lib_target = to_lib_target(ctx.target);
    let source_map = SourceMap::new();
    let diagnostics = DiagnosticEngine::new();
    let interner = Interner::new();

    let mut pp = Preprocessor::new(source_map, diagnostics, lib_target, interner);
    configure_preprocessor(&mut pp, ctx);

    match pp.preprocess(input_path) {
        Ok(tokens) => {
            // Write expanded token stream to stdout.
            let stdout = std::io::stdout();
            let mut out = std::io::BufWriter::new(stdout.lock());
            let mut need_space = false;

            for tok in &tokens {
                match &tok.kind {
                    TokenKind::Eof => break,
                    TokenKind::Newline => {
                        let _ = writeln!(out);
                        need_space = false;
                    }
                    TokenKind::Whitespace => {
                        need_space = true;
                    }
                    _ => {
                        let text = token_to_text(tok, &pp.interner);
                        if !text.is_empty() {
                            if need_space {
                                let _ = write!(out, " ");
                            }
                            let _ = write!(out, "{}", text);
                            need_space = true;
                        }
                    }
                }
            }
            // Ensure output ends with a newline.
            let _ = writeln!(out);
            let _ = out.flush();

            // Check for non-fatal errors accumulated during preprocessing.
            if pp.diagnostics.has_errors() {
                pp.diagnostics.print_all(&pp.source_map);
                return false;
            }
            true
        }
        Err(()) => {
            pp.diagnostics.print_all(&pp.source_map);
            false
        }
    }
}

// =============================================================================
// Full Compilation Pipeline (Phases 1–10)
// =============================================================================

/// Compile a single source file through the complete 10-phase pipeline.
///
/// Orchestrates every stage of the BCC compilation pipeline in strict
/// sequential order:
///
/// 1. **Phases 1–2 — Preprocessing:** Trigraph replacement, line splicing,
///    `#include` resolution, macro expansion with paint-marker recursion
///    protection.
/// 2. **Phase 3 — Lexing:** Integrated with preprocessing; the preprocessor
///    directly produces a token stream with PUA-encoded non-UTF-8 bytes.
/// 3. **Phase 4 — Parsing:** Recursive-descent construction of the AST with
///    full GCC extension support (statement expressions, `typeof`, attributes,
///    inline assembly, computed gotos, case ranges).
/// 4. **Phase 5 — Semantic Analysis:** Type checking, scope resolution,
///    constant evaluation, builtin evaluation, attribute validation.
/// 5. **Phase 6 — IR Lowering:** AST → IR with alloca instructions for all
///    local variables (the "alloca" half of alloca-then-promote).
/// 6. **Phase 7 — mem2reg:** SSA construction via dominance-frontier-based
///    promotion of eligible allocas to virtual registers (the "promote" half).
/// 7. **Phase 8 — Optimization Passes:** Constant folding, dead code
///    elimination, CFG simplification, iterated to fixpoint.
/// 8. **Phase 9 — Phi Elimination:** Conversion of SSA phi nodes to parallel
///    copies at predecessor block terminators.
/// 9. **Phase 10 — Code Generation:** Architecture-dispatching instruction
///    selection, register allocation, built-in assembler, built-in linker.
///
/// The pipeline halts on the first error-producing phase, printing all
/// accumulated diagnostics to stderr.
///
/// # Returns
///
/// `true` on successful compilation, `false` on any error.
fn compile_single_file(ctx: &CompilationContext, input_path: &Path) -> bool {
    let lib_target = to_lib_target(ctx.target);

    // Determine the output path for this input file.
    let output_path = ctx
        .output_path
        .clone()
        .unwrap_or_else(|| derive_output_path(input_path, ctx.output_mode));

    // ══════════════════════════════════════════════════════════════════
    // Phases 1–2: Preprocessing
    // ══════════════════════════════════════════════════════════════════
    // Create the shared infrastructure objects owned by the preprocessor.
    // After preprocessing completes, these are passed forward: borrowed
    // by the parser and semantic analyzer, then transferred by value to
    // the IR lowering phase.
    let source_map = SourceMap::new();
    let diagnostics = DiagnosticEngine::new();
    let interner = Interner::new();

    let mut pp = Preprocessor::new(source_map, diagnostics, lib_target, interner);
    configure_preprocessor(&mut pp, ctx);

    let tokens = match pp.preprocess(input_path) {
        Ok(tokens) => tokens,
        Err(()) => {
            pp.diagnostics.print_all(&pp.source_map);
            return false;
        }
    };

    if pp.diagnostics.has_errors() {
        pp.diagnostics.print_all(&pp.source_map);
        return false;
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 3: Lexing (integrated with preprocessing)
    // ══════════════════════════════════════════════════════════════════
    // The preprocessor already produces a fully-tokenised stream. No
    // separate lexer pass is needed — tokens are ready for the parser.

    // ══════════════════════════════════════════════════════════════════
    // Phase 4: Parsing
    // ══════════════════════════════════════════════════════════════════
    // The parser borrows the diagnostics engine, source map, and interner
    // from the preprocessor. We scope the parser to a block so that the
    // borrows end cleanly before we access pp for error reporting.
    let parse_result = {
        let mut parser = Parser::new(
            &tokens,
            &mut pp.diagnostics,
            &pp.source_map,
            &mut pp.interner,
            &lib_target,
        );
        parser.parse_translation_unit()
    };
    // All Parser borrows on pp fields end here.

    let ast = match parse_result {
        Ok(ast) => ast,
        Err(_) => {
            pp.diagnostics.print_all(&pp.source_map);
            return false;
        }
    };

    if pp.diagnostics.has_errors() {
        pp.diagnostics.print_all(&pp.source_map);
        return false;
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 5: Semantic Analysis
    // ══════════════════════════════════════════════════════════════════
    // The semantic analyzer borrows the same shared state from pp.
    // Again scoped to release borrows before error handling.
    let sema_result = {
        let mut sema = SemanticAnalyzer::new(
            &mut pp.diagnostics,
            &pp.source_map,
            &mut pp.interner,
            &lib_target,
        );
        sema.analyze(&ast)
    };
    // All SemanticAnalyzer borrows on pp fields end here.

    let checked_tu = match sema_result {
        Ok(checked) => checked,
        Err(()) => {
            pp.diagnostics.print_all(&pp.source_map);
            return false;
        }
    };

    if pp.diagnostics.has_errors() {
        pp.diagnostics.print_all(&pp.source_map);
        return false;
    }

    // ══════════════════════════════════════════════════════════════════
    // Ownership Transfer: Preprocessor → IR Lowering
    // ══════════════════════════════════════════════════════════════════
    // The IR lowering phase takes ownership of DiagnosticEngine, SourceMap,
    // and Interner. We extract them from the preprocessor using
    // std::mem::replace so that the preprocessor remains in a valid
    // (albeit empty) state for its Drop implementation.
    let diag_owned = std::mem::replace(&mut pp.diagnostics, DiagnosticEngine::new());
    let smap_owned = std::mem::replace(&mut pp.source_map, SourceMap::new());
    let intern_owned = std::mem::replace(&mut pp.interner, Interner::new());
    drop(pp); // Release preprocessor resources cleanly.

    // Derive the module name from the input file name for IR metadata.
    let module_name = input_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("module")
        .to_string();

    // ══════════════════════════════════════════════════════════════════
    // Phase 6: IR Lowering (alloca insertion)
    // ══════════════════════════════════════════════════════════════════
    // Lower the semantically validated AST into BCC's IR representation.
    // Every local variable is initially emitted as an `alloca` instruction
    // in the function entry block — the "alloca" half of the mandated
    // alloca-then-promote SSA construction architecture.
    let lowering_ctx = match lower_translation_unit(
        &checked_tu,
        lib_target,
        diag_owned,
        smap_owned,
        intern_owned,
        module_name,
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "bcc: error: IR lowering failed: {:?}", e);
            return false;
        }
    };

    if lowering_ctx.diagnostics.has_errors() {
        lowering_ctx
            .diagnostics
            .print_all(&lowering_ctx.source_map);
        return false;
    }

    // Extract the IR module and diagnostic infrastructure from the
    // lowering context for subsequent pipeline stages.
    let mut ir_module = lowering_ctx.module;
    let mut diagnostics = lowering_ctx.diagnostics;
    let source_map_for_diag = lowering_ctx.source_map;

    // ══════════════════════════════════════════════════════════════════
    // Phase 7: mem2reg (SSA Construction)
    // ══════════════════════════════════════════════════════════════════
    // Promote eligible alloca instructions to SSA virtual registers using
    // dominance-frontier computation. This is the "promote" half of the
    // alloca-then-promote architecture. Only scalar, non-address-taken
    // allocas are promoted; complex aggregates remain in memory.
    for func in ir_module.functions.iter_mut() {
        promote_allocas_to_registers(func);
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 8: Optimization Passes
    // ══════════════════════════════════════════════════════════════════
    // Run the fixed optimization pipeline: constant folding → dead code
    // elimination → CFG simplification, iterated to a fixpoint. The pass
    // manager operates on all functions in the module.
    let mut pass_manager = PassManager::default_pipeline();
    let _opt_stats = pass_manager.run_on_module(&mut ir_module);

    // ══════════════════════════════════════════════════════════════════
    // Phase 9: Phi Elimination
    // ══════════════════════════════════════════════════════════════════
    // Convert SSA phi nodes to parallel copies placed before predecessor
    // block terminators, then sequentialise copies. This produces a form
    // suitable for register allocation in the backend.
    for func in ir_module.functions.iter_mut() {
        eliminate_phis(func);
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 10: Code Generation → Assembler → Linker
    // ══════════════════════════════════════════════════════════════════
    // Build the codegen configuration from the compilation context and
    // invoke the architecture-dispatching code generation driver. This
    // performs instruction selection, register allocation, machine code
    // emission via the built-in assembler, and (for Executable/shared
    // modes) linking via the built-in linker — all without invoking any
    // external toolchain component.
    let codegen_config = CodegenConfig {
        target: lib_target,
        optimization_level: opt_level_to_u32(ctx.optimization_level),
        debug_info: ctx.debug_info,
        pic: ctx.pic,
        shared: ctx.shared,
        retpoline: ctx.retpoline,
        cf_protection: ctx.cf_protection,
        output_path: output_path.clone(),
        output_mode: to_backend_output_mode(ctx.output_mode),
    };

    match generate_code(&ir_module, &codegen_config, &mut diagnostics) {
        Ok(()) => {
            // Code generation succeeded. Print any non-fatal warnings that
            // were accumulated during the backend phases.
            if diagnostics.has_errors() {
                diagnostics.print_all(&source_map_for_diag);
                return false;
            }
            true
        }
        Err(e) => {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "bcc: error: code generation failed: {}", e);
            diagnostics.print_all(&source_map_for_diag);
            false
        }
    }
}

// =============================================================================
// Compilation Driver
// =============================================================================

/// Run the compilation pipeline for all input files in the context.
///
/// Iterates over each input file, dispatching to either the preprocess-only
/// path (`-E`) or the full 10-phase compilation pipeline. Each file is
/// compiled independently; errors in one file do not prevent compilation
/// of subsequent files (matching GCC behaviour with `-c`).
///
/// # Returns
///
/// `0` on success (all files compiled without errors), `1` on any failure.
pub fn run_compilation(ctx: &CompilationContext) -> i32 {
    let mut had_errors = false;

    for input_path in &ctx.input_files {
        // Validate that the input file exists and is readable.
        if !input_path.exists() {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(
                stderr,
                "bcc: error: no such file or directory: '{}'",
                input_path.display()
            );
            had_errors = true;
            continue;
        }

        // Handle -E mode (preprocess only) — separate fast path that
        // writes expanded tokens to stdout and returns.
        if ctx.output_mode == OutputMode::Preprocess {
            if !run_preprocess(ctx, input_path) {
                had_errors = true;
            }
            continue;
        }

        // Full compilation pipeline (Phases 1–10).
        if !compile_single_file(ctx, input_path) {
            had_errors = true;
        }
    }

    if had_errors {
        1
    } else {
        0
    }
}

// =============================================================================
// Entry Point
// =============================================================================

/// Program entry point.
///
/// Parses command-line arguments and spawns a worker thread with a 64 MiB
/// stack to execute the compilation pipeline. The oversized stack is
/// required because deeply nested kernel macro expansions (particularly
/// in the Linux kernel build) can exhaust the default ~8 MiB thread stack.
///
/// The main thread waits for the worker to complete and propagates its
/// exit code to the process.
fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command-line arguments.
    let ctx = match parse_args(&args) {
        Ok(ctx) => ctx,
        Err(msg) => {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "bcc: error: {}", msg);
            process::exit(1);
        }
    };

    // Spawn the compilation worker thread with a 64 MiB stack.
    // This is mandated by Section 0.7.3 to handle deeply nested kernel
    // macro expansions that would otherwise overflow the default stack.
    let exit_code = std::thread::Builder::new()
        .name("bcc-worker".to_string())
        .stack_size(WORKER_STACK_SIZE)
        .spawn(move || run_compilation(&ctx))
        .expect("bcc: fatal: failed to spawn worker thread with 64 MiB stack")
        .join()
        .expect("bcc: fatal: worker thread panicked");

    process::exit(exit_code);
}
