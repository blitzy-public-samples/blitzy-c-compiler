//! BCC — Blitzy's C Compiler
//!
//! CLI entry point for the BCC compilation toolchain. Handles command-line
//! argument parsing for GCC-compatible flags, spawns a worker thread with
//! a 64 MiB stack (required for deeply nested kernel macro expansions),
//! and orchestrates the full compilation pipeline.
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

// Library imports for the preprocessing pipeline (-E mode).
use bcc::common::{DiagnosticEngine, Interner, SourceMap, Target as LibTarget};
use bcc::frontend::preprocessor::Preprocessor;
use bcc::frontend::lexer::token::{TokenKind, Token as LexToken};

/// Maximum recursion depth for the parser and macro expander.
/// Enforced to prevent stack overflow on deeply nested kernel constructs.
const MAX_RECURSION_DEPTH: usize = 512;

/// Worker thread stack size: 64 MiB (67,108,864 bytes).
/// Required because deeply nested kernel macro expansions can exhaust the
/// default thread stack.
const WORKER_STACK_SIZE: usize = 64 * 1024 * 1024;

/// Output mode determined by compilation flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Produce a linked executable (default behavior).
    Executable,
    /// Compile to object file only (`-c` flag).
    Object,
    /// Compile to assembly text (`-S` flag).
    Assembly,
    /// Preprocess only, output to stdout (`-E` flag).
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
/// Mirrors the library's `Target` enum but defined locally in the binary
/// crate for CLI parsing before library initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    /// x86-64 (AMD64) — 64-bit, LP64 data model
    X86_64,
    /// i686 (IA-32) — 32-bit, ILP32 data model
    I686,
    /// AArch64 (ARM64) — 64-bit, LP64 data model
    AArch64,
    /// RISC-V 64 — 64-bit, LP64D ABI
    RiscV64,
}

impl TargetArch {
    /// Parse a target architecture from a CLI string.
    ///
    /// Accepts: `"x86-64"`, `"x86_64"`, `"i686"`, `"i386"`, `"aarch64"`,
    /// `"arm64"`, `"riscv64"`.
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "x86-64" | "x86_64" => Some(TargetArch::X86_64),
            "i686" | "i386" => Some(TargetArch::I686),
            "aarch64" | "arm64" => Some(TargetArch::AArch64),
            "riscv64" => Some(TargetArch::RiscV64),
            _ => None,
        }
    }

    /// Detect the host architecture at compile time.
    fn host() -> Self {
        if cfg!(target_arch = "x86_64") {
            TargetArch::X86_64
        } else if cfg!(target_arch = "x86") {
            TargetArch::I686
        } else if cfg!(target_arch = "aarch64") {
            TargetArch::AArch64
        } else if cfg!(target_arch = "riscv64") {
            TargetArch::RiscV64
        } else {
            // Default to x86-64 on unsupported host architectures.
            TargetArch::X86_64
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

/// Compilation context built from parsed command-line arguments.
/// Contains all configuration needed to drive the compilation pipeline.
pub struct CompilationContext {
    /// Target architecture for code generation.
    pub target: TargetArch,
    /// Output mode (executable, object, assembly, preprocess-only).
    pub output_mode: OutputMode,
    /// Output file path (None = derive from input file name).
    pub output_path: Option<PathBuf>,
    /// Input source files to compile.
    pub input_files: Vec<PathBuf>,
    /// Optimization level (O0 through O3).
    pub optimization_level: OptimizationLevel,
    /// Whether to emit DWARF v4 debug information (`-g`).
    pub debug_info: bool,
    /// Whether to generate position-independent code (`-fPIC`).
    pub pic: bool,
    /// Whether to produce a shared object (`-shared`).
    pub shared: bool,
    /// Whether to enable retpoline indirect branch mitigation (`-mretpoline`, x86-64 only).
    pub retpoline: bool,
    /// Whether to enable CET/IBT control-flow protection (`-fcf-protection`, x86-64 only).
    pub cf_protection: bool,
    /// Include search paths (`-I`).
    pub include_paths: Vec<PathBuf>,
    /// Preprocessor macro definitions (`-D`). Each entry is (name, optional_value).
    pub macro_definitions: Vec<(String, Option<String>)>,
    /// Library search paths (`-L`).
    pub library_paths: Vec<PathBuf>,
    /// Libraries to link against (`-l`).
    pub libraries: Vec<String>,
    /// Maximum recursion depth for parser and macro expander.
    pub recursion_depth_limit: usize,
}

/// Parse command-line arguments into a [`CompilationContext`].
///
/// Implements a hand-rolled GCC-compatible argument parser without any
/// external crate dependencies. Supports both attached (`-Ipath`) and
/// detached (`-I path`) argument forms for flags that take values.
///
/// # Errors
///
/// Returns a descriptive error string if argument parsing fails (unknown
/// flags, missing required arguments, invalid target, etc.).
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
        } else if arg == "-o" {
            i += 1;
            if i >= args.len() {
                return Err("-o requires an output file path".to_string());
            }
            ctx.output_path = Some(PathBuf::from(&args[i]));
        } else if arg == "-c" {
            ctx.output_mode = OutputMode::Object;
        } else if arg == "-S" {
            ctx.output_mode = OutputMode::Assembly;
        } else if arg == "-E" {
            ctx.output_mode = OutputMode::Preprocess;
        } else if arg == "-g" {
            ctx.debug_info = true;
        } else if arg == "-O0" {
            ctx.optimization_level = OptimizationLevel::O0;
        } else if arg == "-O1" || arg == "-O" {
            ctx.optimization_level = OptimizationLevel::O1;
        } else if arg == "-O2" {
            ctx.optimization_level = OptimizationLevel::O2;
        } else if arg == "-O3" {
            ctx.optimization_level = OptimizationLevel::O3;
        } else if arg == "-fPIC" || arg == "-fpic" {
            ctx.pic = true;
        } else if arg == "-shared" {
            ctx.shared = true;
            // -shared implies -fPIC for all code in the shared object.
            ctx.pic = true;
        } else if arg == "-mretpoline" {
            ctx.retpoline = true;
        } else if arg == "-fcf-protection" {
            ctx.cf_protection = true;
        } else if let Some(rest) = arg.strip_prefix("-I") {
            let path = if !rest.is_empty() {
                // Attached form: -I/usr/include
                rest
            } else {
                // Detached form: -I /usr/include
                i += 1;
                if i >= args.len() {
                    return Err("-I requires a directory path".to_string());
                }
                &args[i]
            };
            ctx.include_paths.push(PathBuf::from(path));
        } else if let Some(rest) = arg.strip_prefix("-D") {
            let def = if !rest.is_empty() {
                // Attached form: -DFOO=bar
                rest
            } else {
                // Detached form: -D FOO=bar
                i += 1;
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
                // -DFOO without value defines FOO as 1 (standard behavior).
                ctx.macro_definitions.push((def.to_string(), None));
            }
        } else if let Some(rest) = arg.strip_prefix("-L") {
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
        } else if let Some(rest) = arg.strip_prefix("-l") {
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
        } else if arg == "--help" || arg == "-h" {
            print_usage();
            process::exit(0);
        } else if arg == "--version" || arg == "-v" {
            println!("bcc 0.1.0");
            process::exit(0);
        } else if arg.starts_with('-') && arg != "-" {
            return Err(format!("unrecognized command-line option: '{}'", arg));
        } else {
            // Positional argument: input source file.
            ctx.input_files.push(PathBuf::from(arg));
        }

        i += 1;
    }

    if ctx.input_files.is_empty() {
        return Err("no input files".to_string());
    }

    // Validate that security mitigation flags are only used with x86-64.
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

/// Print usage information to stderr.
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
    let _ = writeln!(stderr, "  -shared            Produce shared object");
    let _ = writeln!(
        stderr,
        "  -mretpoline        Enable retpoline mitigation (x86-64)"
    );
    let _ = writeln!(
        stderr,
        "  -fcf-protection    Enable CET/IBT protection (x86-64)"
    );
    let _ = writeln!(stderr, "  -I<dir>            Add include search path");
    let _ = writeln!(stderr, "  -D<macro>[=value]  Define preprocessor macro");
    let _ = writeln!(stderr, "  -L<dir>            Add library search path");
    let _ = writeln!(stderr, "  -l<lib>            Link against library");
    let _ = writeln!(stderr, "  --help, -h         Display this help");
    let _ = writeln!(stderr, "  --version, -v      Display version");
}

/// Derive the default output file path from the first input file and output mode.
fn derive_output_path(input: &std::path::Path, mode: OutputMode) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("a");
    match mode {
        OutputMode::Executable => PathBuf::from("a.out"),
        OutputMode::Object => PathBuf::from(format!("{}.o", stem)),
        OutputMode::Assembly => PathBuf::from(format!("{}.s", stem)),
        OutputMode::Preprocess => PathBuf::from("-"), // stdout sentinel
    }
}

/// Convert a CLI `TargetArch` to the library `Target` enum.
///
/// The CLI defines its own `TargetArch` type for argument parsing, while
/// the library uses `bcc::common::Target`. This helper bridges the two.
fn to_lib_target(arch: TargetArch) -> LibTarget {
    match arch {
        TargetArch::X86_64 => LibTarget::X86_64,
        TargetArch::I686 => LibTarget::I686,
        TargetArch::AArch64 => LibTarget::AArch64,
        TargetArch::RiscV64 => LibTarget::RiscV64,
    }
}

/// Convert a preprocessor token to its textual representation for `-E` output.
///
/// The library's `Token::Display` implementation prints category names
/// (e.g., "identifier", "string literal") which is useful for diagnostics
/// but not for preprocessor output. This function reconstructs the actual
/// C source text from the token data.
///
/// # Arguments
///
/// * `token` — The token to convert.
/// * `interner` — The string interner for resolving identifier `Symbol` handles.
fn token_to_text(token: &LexToken, interner: &Interner) -> String {
    match &token.kind {
        // Identifiers — resolve from the interner.
        TokenKind::Identifier(sym) => interner.resolve(*sym).to_string(),

        // Integer literals — reconstruct value + suffix.
        TokenKind::IntegerLiteral { value, suffix } => {
            format!("{}{}", value, suffix)
        }

        // Floating-point literals — reconstruct value + suffix.
        TokenKind::FloatLiteral { value, suffix } => {
            // Use a representation that can round-trip. For values that are
            // whole numbers, ensure a decimal point is present.
            let val_str = if value.fract() == 0.0 && !value.is_infinite() && !value.is_nan() {
                format!("{:.1}", value)
            } else {
                format!("{}", value)
            };
            format!("{}{}", val_str, suffix)
        }

        // String literals — reconstruct with prefix, quotes, and escape sequences.
        TokenKind::StringLiteral { value, prefix } => {
            let mut s = format!("{}\"", prefix);
            for &b in value.iter() {
                match b {
                    b'\\' => s.push_str("\\\\"),
                    b'"' => s.push_str("\\\""),
                    b'\n' => s.push_str("\\n"),
                    b'\r' => s.push_str("\\r"),
                    b'\t' => s.push_str("\\t"),
                    0 => s.push_str("\\0"),
                    0x20..=0x7e => s.push(b as char),
                    _ => {
                        // Non-printable / high bytes — hex escape.
                        s.push_str(&format!("\\x{:02x}", b));
                    }
                }
            }
            s.push('"');
            s
        }

        // Character literals — reconstruct with prefix and quotes.
        TokenKind::CharLiteral { value, prefix } => {
            let ch = *value;
            let prefix_str = format!("{}", prefix);
            if (0x20..0x7f).contains(&ch) && ch != (b'\\' as u32) && ch != (b'\'' as u32) {
                format!("{}'{}'" , prefix_str, char::from_u32(ch).unwrap_or('?'))
            } else {
                match ch {
                    0x0a => format!("{}'\\n'", prefix_str),
                    0x0d => format!("{}'\\r'", prefix_str),
                    0x09 => format!("{}'\\t'", prefix_str),
                    0x00 => format!("{}'\\0'", prefix_str),
                    0x5c => format!("{}'\\\\'" , prefix_str),
                    0x27 => format!("{}'\\''", prefix_str),
                    _ => format!("{}'\\x{:02x}'", prefix_str, ch),
                }
            }
        }

        // EOF — no output text.
        TokenKind::Eof => String::new(),

        // Newline / Whitespace — emit a space to separate tokens.
        TokenKind::Newline => "\n".to_string(),
        TokenKind::Whitespace => " ".to_string(),

        // Error tokens — skip in preprocessor output.
        TokenKind::Error => String::new(),

        // All other tokens (keywords, operators, punctuators, builtins) —
        // the Display implementation produces the correct C source text.
        other => format!("{}", other),
    }
}

/// Run the preprocessing pipeline for `-E` mode.
///
/// Instantiates the preprocessor, processes the input file, and writes
/// the token stream as reconstructed C source text to stdout. Returns
/// `true` on success, `false` on failure.
fn run_preprocess(ctx: &CompilationContext, input_path: &Path) -> bool {
    let lib_target = to_lib_target(ctx.target);
    let source_map = SourceMap::new();
    let diagnostics = DiagnosticEngine::new();
    let interner = Interner::new();

    let mut pp = Preprocessor::new(source_map, diagnostics, lib_target, interner);

    // Add system include paths for standard header resolution.
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

    // Add GCC internal include paths for builtins (stdarg.h, stddef.h, etc.).
    // Search for the latest GCC version available on the system.
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

    // Add user include paths from -I flags.
    for path in &ctx.include_paths {
        pp.add_user_include_path(path.clone());
    }

    // Add macro definitions from -D flags.
    for (name, value) in &ctx.macro_definitions {
        if let Some(val) = value {
            pp.add_define(&format!("{}={}", name, val));
        } else {
            pp.add_define(name);
        }
    }

    // Run preprocessing.
    match pp.preprocess(input_path) {
        Ok(tokens) => {
            // Reconstruct and write the preprocessed token stream to stdout.
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
            // Final newline to ensure clean output.
            let _ = writeln!(out);
            let _ = out.flush();

            // Print any diagnostics (warnings) that occurred during preprocessing.
            if pp.diagnostics.has_errors() {
                pp.diagnostics.print_all(&pp.source_map);
                return false;
            }
            true
        }
        Err(()) => {
            // Print diagnostics from the preprocessor.
            pp.diagnostics.print_all(&pp.source_map);
            false
        }
    }
}

/// Run the compilation pipeline for a single input file.
///
/// Executes the full 10-phase compilation pipeline:
/// - Phase 1-2: Preprocessing (trigraphs, line splicing, macro expansion)
/// - Phase 3: Lexing (tokenization)
/// - Phase 4: Parsing (AST construction)
/// - Phase 5: Semantic analysis (type checking)
/// - Phase 6: IR lowering (alloca insertion)
/// - Phase 7: mem2reg (SSA construction via alloca-then-promote)
/// - Phase 8: Optimization passes
/// - Phase 9: Phi elimination
/// - Phase 10: Code generation → Assembler → Linker
///
/// Returns 0 on success, 1 on compilation failure.
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

        // ── Handle -E mode (preprocess only) ─────────────────────────
        // For -E mode, we invoke the preprocessor directly and write the
        // expanded token stream to stdout. No further pipeline stages are
        // needed. This is fully operational and does not depend on IR or
        // backend modules.
        if ctx.output_mode == OutputMode::Preprocess {
            if !run_preprocess(ctx, input_path) {
                had_errors = true;
            }
            continue;
        }

        // ── Full compilation pipeline (Phases 1–10) ─────────────────
        // Determine the output path for this input file.
        let output = ctx
            .output_path
            .clone()
            .unwrap_or_else(|| derive_output_path(input_path, ctx.output_mode));

        // Read the source file as raw bytes for PUA encoding.
        let source_bytes = match std::fs::read(input_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                let mut stderr = std::io::stderr().lock();
                let _ = writeln!(
                    stderr,
                    "bcc: error: cannot read '{}': {}",
                    input_path.display(),
                    e
                );
                had_errors = true;
                continue;
            }
        };

        // Convert source bytes to a string, handling non-UTF-8 gracefully.
        // Full PUA encoding will be applied once the encoding module is connected.
        let source = String::from_utf8(source_bytes).unwrap_or_else(|e| {
            // Lossy conversion for now — PUA encoding module will handle
            // byte-exact round-tripping once connected.
            String::from_utf8_lossy(e.as_bytes()).into_owned()
        });

        // Validate that the source file is not empty.
        if source.is_empty() {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "bcc: warning: '{}' is empty", input_path.display());
        }

        // The compilation pipeline stages will be invoked here as the
        // frontend, IR, optimization, and backend modules are assembled.
        // Each phase transforms the compilation unit through the pipeline:
        //
        // source text → tokens → AST → typed AST → IR → SSA IR →
        // optimized IR → machine code → ELF object → linked executable

        // For now, verify we can at least read and measure the source file.
        let _source_len = source.len();
        let _output_path = output;

        // Report that compilation for this input cannot yet complete
        // because pipeline modules are still being assembled.
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(
            stderr,
            "bcc: error: compilation pipeline for '{}' (target: {}) is not yet fully connected",
            input_path.display(),
            ctx.target
        );
        had_errors = true;
    }

    if had_errors {
        1
    } else {
        0
    }
}

/// Program entry point.
///
/// Parses command-line arguments and spawns a worker thread with a 64 MiB
/// stack to execute the compilation pipeline. The oversized stack is necessary
/// because deeply nested kernel macro expansions (e.g., Linux kernel build)
/// can exhaust the default thread stack.
fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command-line arguments into a compilation context.
    let ctx = match parse_args(&args) {
        Ok(ctx) => ctx,
        Err(msg) => {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "bcc: error: {}", msg);
            process::exit(1);
        }
    };

    // Spawn a worker thread with 64 MiB stack for handling deep recursion
    // in the parser and macro expander (required per resource constraints).
    let exit_code = std::thread::Builder::new()
        .name("bcc-worker".to_string())
        .stack_size(WORKER_STACK_SIZE)
        .spawn(move || run_compilation(&ctx))
        .expect("bcc: fatal: failed to spawn worker thread with 64 MiB stack")
        .join()
        .expect("bcc: fatal: worker thread panicked");

    process::exit(exit_code);
}
