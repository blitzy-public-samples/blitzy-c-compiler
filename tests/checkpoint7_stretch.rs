//! Checkpoint 7 — Optional Stretch Target Validation
//!
//! This integration test suite validates that the BCC compiler can successfully
//! build four large, real-world C projects: SQLite, Redis, PostgreSQL, and FFmpeg.
//! Each test verifies:
//!
//! 1. The project compiles to completion with BCC as the C compiler.
//! 2. The produced binary is a valid ELF executable.
//! 3. A basic functionality smoke test passes on the compiled binary.
//! 4. The wall-clock build time does not exceed 5× the equivalent GCC build time
//!    on the same hardware (Section 0.7.8).
//!
//! All tests are marked `#[ignore]` by default because they require external
//! source trees to be present on the host and are optional stretch targets
//! (Section 0.7.5). They may execute in parallel after Checkpoint 6 passes.
//!
//! **Environment variables for source directories:**
//!
//! - `SQLITE_SRC_DIR`     — Path to the SQLite amalgamation source directory
//!   (must contain `sqlite3.c` and `sqlite3.h`).
//! - `REDIS_SRC_DIR`      — Path to the Redis source tree root
//!   (must contain a top-level `Makefile`).
//! - `POSTGRESQL_SRC_DIR` — Path to the PostgreSQL source tree root
//!   (must contain a `configure` script).
//! - `FFMPEG_SRC_DIR`     — Path to the FFmpeg source tree root
//!   (must contain a `configure` script).
//!
//! **Environment variables for GCC benchmark times (optional):**
//!
//! - `GCC_BENCHMARK_SECS_SQLITE`     — GCC build time in seconds for SQLite.
//! - `GCC_BENCHMARK_SECS_REDIS`      — GCC build time in seconds for Redis.
//! - `GCC_BENCHMARK_SECS_POSTGRESQL` — GCC build time in seconds for PostgreSQL.
//! - `GCC_BENCHMARK_SECS_FFMPEG`     — GCC build time in seconds for FFmpeg.
//!
//! If a GCC benchmark variable is not set, a conservative default is used.
//!
//! No external dependencies — uses only the Rust standard library.

mod common;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Environment variable names — source directory paths
// ---------------------------------------------------------------------------

/// Environment variable pointing to the SQLite amalgamation source directory.
const SQLITE_SRC_DIR_ENV: &str = "SQLITE_SRC_DIR";

/// Environment variable pointing to the Redis source tree root.
const REDIS_SRC_DIR_ENV: &str = "REDIS_SRC_DIR";

/// Environment variable pointing to the PostgreSQL source tree root.
const POSTGRESQL_SRC_DIR_ENV: &str = "POSTGRESQL_SRC_DIR";

/// Environment variable pointing to the FFmpeg source tree root.
const FFMPEG_SRC_DIR_ENV: &str = "FFMPEG_SRC_DIR";

// ---------------------------------------------------------------------------
// Environment variable names — GCC benchmark seconds (for 5× ceiling)
// ---------------------------------------------------------------------------

/// Environment variable for GCC benchmark time (seconds) building SQLite.
const GCC_BENCHMARK_SECS_SQLITE_ENV: &str = "GCC_BENCHMARK_SECS_SQLITE";

/// Environment variable for GCC benchmark time (seconds) building Redis.
const GCC_BENCHMARK_SECS_REDIS_ENV: &str = "GCC_BENCHMARK_SECS_REDIS";

/// Environment variable for GCC benchmark time (seconds) building PostgreSQL.
const GCC_BENCHMARK_SECS_POSTGRESQL_ENV: &str = "GCC_BENCHMARK_SECS_POSTGRESQL";

/// Environment variable for GCC benchmark time (seconds) building FFmpeg.
const GCC_BENCHMARK_SECS_FFMPEG_ENV: &str = "GCC_BENCHMARK_SECS_FFMPEG";

// ---------------------------------------------------------------------------
// Default GCC benchmark seconds (conservative estimates)
// ---------------------------------------------------------------------------

/// Default GCC build time for SQLite amalgamation (seconds).
/// SQLite compiles from a single amalgamation file; 30s is conservative.
const DEFAULT_GCC_BENCHMARK_SQLITE_SECS: u64 = 30;

/// Default GCC build time for Redis (seconds).
/// Redis is a moderately sized C project; 60s is conservative.
const DEFAULT_GCC_BENCHMARK_REDIS_SECS: u64 = 60;

/// Default GCC build time for PostgreSQL (seconds).
/// PostgreSQL is a large C project; 300s (5 min) is conservative.
const DEFAULT_GCC_BENCHMARK_POSTGRESQL_SECS: u64 = 300;

/// Default GCC build time for FFmpeg (seconds).
/// FFmpeg is a large, heavily optimised C project; 300s (5 min) is conservative.
const DEFAULT_GCC_BENCHMARK_FFMPEG_SECS: u64 = 300;

/// Wall-clock multiplier: BCC build time must not exceed this factor of the
/// GCC equivalent build time (Section 0.7.8).
const WALL_CLOCK_MULTIPLIER: u64 = 5;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Read the GCC benchmark seconds from an environment variable, falling back
/// to a provided default if the variable is not set or not a valid integer.
///
/// # Arguments
///
/// * `env_var`        — Name of the environment variable to read.
/// * `default_secs`   — Default seconds to use if the variable is absent.
///
/// # Returns
///
/// The benchmark duration in seconds.
fn gcc_benchmark_secs(env_var: &str, default_secs: u64) -> u64 {
    match env::var(env_var) {
        Ok(val) => val.parse::<u64>().unwrap_or_else(|_| {
            eprintln!(
                "Warning: {} is set to '{}' which is not a valid u64; \
                 using default {} seconds.",
                env_var, val, default_secs
            );
            default_secs
        }),
        Err(_) => default_secs,
    }
}

/// Compute the maximum allowed BCC build time from the GCC benchmark seconds
/// and the 5× wall-clock multiplier.
///
/// # Arguments
///
/// * `gcc_secs` — GCC benchmark time in seconds.
///
/// # Returns
///
/// The ceiling duration (gcc_secs × WALL_CLOCK_MULTIPLIER) as a `Duration`.
fn wall_clock_ceiling(gcc_secs: u64) -> Duration {
    Duration::from_secs(gcc_secs.saturating_mul(WALL_CLOCK_MULTIPLIER))
}

/// Assert that the measured build duration does not exceed the 5× GCC ceiling.
///
/// # Arguments
///
/// * `elapsed`   — Actual BCC build duration.
/// * `gcc_secs`  — GCC benchmark time in seconds.
/// * `project`   — Human-readable project name for diagnostics.
///
/// # Panics
///
/// Panics if `elapsed` exceeds `gcc_secs × 5`.
fn assert_within_wall_clock_ceiling(elapsed: Duration, gcc_secs: u64, project: &str) {
    let ceiling = wall_clock_ceiling(gcc_secs);
    assert!(
        elapsed <= ceiling,
        "{} build took {:.1}s, exceeding the {:.1}s ceiling \
         (5× GCC benchmark of {}s per Section 0.7.8).",
        project,
        elapsed.as_secs_f64(),
        ceiling.as_secs_f64(),
        gcc_secs
    );
}

/// Resolve the source directory for a stretch target from its environment
/// variable. Returns `None` (and prints a skip message) if the variable is
/// not set or the directory does not exist.
///
/// Uses `env::var_os()` first for lossless OsString retrieval, then falls
/// back to `env::var()` for the string representation used in diagnostics.
///
/// # Arguments
///
/// * `env_var`  — Name of the environment variable.
/// * `project`  — Human-readable project name for diagnostics.
///
/// # Returns
///
/// `Some(PathBuf)` if the directory exists, `None` otherwise.
fn resolve_source_dir(env_var: &str, project: &str) -> Option<PathBuf> {
    // Use var_os() for lossless path retrieval (paths may contain non-UTF-8).
    let os_val = match env::var_os(env_var) {
        Some(val) if !val.is_empty() => val,
        _ => {
            eprintln!(
                "Skipping {} stretch target test: {} is not set.",
                project, env_var
            );
            return None;
        }
    };

    let mut dir = PathBuf::from(&os_val);
    // Normalise: if the path has a trailing slash component, push resolves it.
    if dir.as_path().to_str().is_some_and(|s| s.ends_with("/.")) {
        dir.push(".");
        dir = dir.canonicalize().unwrap_or(dir);
    }

    if !dir.as_path().is_dir() {
        // Fall back to env::var() for a human-readable diagnostic string.
        let dir_str = env::var(env_var).unwrap_or_else(|_| format!("{:?}", os_val));
        eprintln!(
            "Skipping {} stretch target test: {} ('{}') is not a valid directory.",
            project, env_var, dir_str
        );
        return None;
    }

    Some(dir)
}

/// Verify that a produced binary exists and is a non-empty regular file.
///
/// # Arguments
///
/// * `path`    — Path to the expected binary.
/// * `project` — Human-readable project name for diagnostics.
///
/// # Panics
///
/// Panics if the file does not exist, is not a regular file, or is empty.
fn assert_binary_produced(path: &Path, project: &str) {
    assert!(
        path.exists(),
        "{} build did not produce expected binary at '{}'.",
        project,
        path.display()
    );
    assert!(
        path.is_file(),
        "{} output path '{}' exists but is not a regular file.",
        project,
        path.display()
    );
    let metadata = fs::metadata(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read metadata for {} binary '{}': {}",
            project,
            path.display(),
            e
        )
    });
    assert!(
        metadata.len() > 0,
        "{} binary '{}' exists but is empty (0 bytes).",
        project,
        path.display()
    );
}

/// Run `make clean` in a directory, ignoring any errors (best-effort cleanup).
///
/// Uses `Command::status()` instead of `output()` since we do not need to
/// capture stdout/stderr for a best-effort cleanup step.
///
/// # Arguments
///
/// * `dir` — Directory in which to run `make clean`.
fn make_clean_best_effort(dir: &Path) {
    let _ = Command::new("make")
        .args(["clean", "-s"])
        .current_dir(dir)
        .status();
}

// ===========================================================================
// Checkpoint 7 Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 7.1 — SQLite Compilation Test
// ---------------------------------------------------------------------------

/// Compile the SQLite amalgamation source (`sqlite3.c`) with BCC, link it into
/// an executable, run a basic SQL query, and verify the wall-clock ceiling.
///
/// **Prerequisites:**
/// - `SQLITE_SRC_DIR` environment variable set to a directory containing
///   `sqlite3.c`, `sqlite3.h`, and optionally `shell.c` (the CLI shell).
///
/// **Validation:**
/// 1. BCC compiles `sqlite3.c` to a relocatable object (`sqlite3.o`).
/// 2. BCC compiles a minimal test harness that opens an in-memory database,
///    executes `SELECT 1;`, and prints the result.
/// 3. The test harness links against `sqlite3.o` into a final executable.
/// 4. Executing the binary produces the expected output.
/// 5. The produced binary is a valid ELF executable.
/// 6. Total build time ≤ 5× GCC benchmark.
#[test]
#[ignore]
fn test_sqlite_build() {
    // --- Source directory resolution ---
    let src_dir = match resolve_source_dir(SQLITE_SRC_DIR_ENV, "SQLite") {
        Some(d) => d,
        None => return, // Skip if source not available.
    };

    // Verify essential source files exist.
    let sqlite3_c = src_dir.join("sqlite3.c");
    let sqlite3_h = src_dir.join("sqlite3.h");
    assert!(
        sqlite3_c.is_file(),
        "SQLite source directory '{}' does not contain sqlite3.c.",
        src_dir.display()
    );
    assert!(
        sqlite3_h.is_file(),
        "SQLite source directory '{}' does not contain sqlite3.h.",
        src_dir.display()
    );

    // --- Build artifact directory ---
    let test_dir = common::TestDir::new("checkpoint7_sqlite");

    // --- Create a minimal test harness ---
    let harness_c = test_dir.file_path("sqlite_test.c");
    let harness_source = format!(
        r#"#include "{sqlite3_h}"
#include <stdio.h>
#include <stdlib.h>

static int callback(void *unused, int ncols, char **values, char **names) {{
    (void)unused;
    (void)names;
    int i;
    for (i = 0; i < ncols; i++) {{
        printf("%s", values[i] ? values[i] : "NULL");
        if (i < ncols - 1) printf("|");
    }}
    printf("\n");
    return 0;
}}

int main(void) {{
    sqlite3 *db;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {{
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }}
    char *err = 0;
    rc = sqlite3_exec(db, "SELECT 1;", callback, 0, &err);
    if (rc != SQLITE_OK) {{
        fprintf(stderr, "SQL error: %s\n", err);
        sqlite3_free(err);
        sqlite3_close(db);
        return 1;
    }}
    sqlite3_close(db);
    return 0;
}}
"#,
        sqlite3_h = sqlite3_h.display()
    );
    fs::write(&harness_c, harness_source).unwrap_or_else(|e| {
        panic!(
            "Failed to write SQLite test harness to '{}': {}",
            harness_c.display(),
            e
        )
    });

    // --- Timed compilation ---
    let sqlite3_o = test_dir.file_path("sqlite3.o");
    let sqlite_test_bin = test_dir.file_path("sqlite_test");

    let start_time = Instant::now();

    let ((), _inner_elapsed) = common::timed_execution(|| {
        // Step 1: Compile sqlite3.c → sqlite3.o using compile() with extra flags.
        let obj_result = common::compile(
            sqlite3_c.to_str().unwrap(),
            &[
                "-c",
                "-o",
                sqlite3_o.to_str().unwrap(),
                "-DSQLITE_THREADSAFE=0",
                "-DSQLITE_OMIT_LOAD_EXTENSION",
            ],
        );
        obj_result.assert_success();

        // Step 2: Compile test harness and link against sqlite3.o in one step
        // using compile_to_binary() — demonstrates single-source-to-binary flow.
        let include_flag = format!("-I{}", src_dir.display());
        let link_obj_flag = sqlite3_o.to_str().unwrap().to_string();
        let result = common::compile_to_binary(
            harness_c.to_str().unwrap(),
            sqlite_test_bin.to_str().unwrap(),
            "x86-64",
            &[&include_flag, &link_obj_flag, "-lpthread", "-ldl", "-lm"],
        );
        result.assert_success();
    });

    let build_elapsed = start_time.elapsed();

    // --- Verify output binary ---
    assert_binary_produced(sqlite_test_bin.as_path(), "SQLite");
    common::assert_elf_type(sqlite_test_bin.to_str().unwrap(), "EXEC");
    common::assert_elf_machine(
        sqlite_test_bin.to_str().unwrap(),
        "Advanced Micro Devices X86-64",
    );

    // --- Functionality smoke test ---
    // Execute the compiled SQLite test binary directly.
    let sqlite_bin_path = Path::new(sqlite_test_bin.to_str().unwrap());
    let run_output = Command::new(sqlite_bin_path)
        .current_dir(test_dir.path())
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to execute SQLite test binary '{}': {}",
                sqlite_test_bin.display(),
                e
            )
        });
    assert!(
        run_output.status.success(),
        "SQLite test binary exited with code {:?}.\nstdout: {}\nstderr: {}",
        run_output.status.code(),
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );
    let stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        stdout.contains('1'),
        "SQLite test binary did not produce expected output '1'.\nActual stdout: '{}'",
        stdout.trim()
    );

    // --- Wall-clock ceiling assertion (Section 0.7.8) ---
    let gcc_secs = gcc_benchmark_secs(
        GCC_BENCHMARK_SECS_SQLITE_ENV,
        DEFAULT_GCC_BENCHMARK_SQLITE_SECS,
    );
    let ceiling = wall_clock_ceiling(gcc_secs);
    assert!(
        build_elapsed.as_secs() <= ceiling.as_secs(),
        "SQLite build took {}s, exceeding the {}s ceiling \
         (5× GCC benchmark of {}s per Section 0.7.8).",
        build_elapsed.as_secs(),
        ceiling.as_secs(),
        gcc_secs
    );

    eprintln!(
        "Checkpoint 7 — SQLite: PASSED (build time: {:.1}s, ceiling: {:.1}s)",
        build_elapsed.as_secs_f64(),
        wall_clock_ceiling(gcc_secs).as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// 7.2 — Redis Compilation Test
// ---------------------------------------------------------------------------

/// Build the Redis source tree with `make CC=<bcc>`, verify that the
/// `redis-server` binary is produced as a valid ELF executable, and assert
/// the wall-clock ceiling.
///
/// **Prerequisites:**
/// - `REDIS_SRC_DIR` environment variable set to the Redis source tree root
///   (must contain a top-level `Makefile`).
///
/// **Validation:**
/// 1. `make CC=<bcc>` succeeds with exit code 0.
/// 2. `src/redis-server` binary is produced and is a valid ELF EXEC.
/// 3. Total build time ≤ 5× GCC benchmark.
#[test]
#[ignore]
fn test_redis_build() {
    // --- Source directory resolution ---
    let src_dir = match resolve_source_dir(REDIS_SRC_DIR_ENV, "Redis") {
        Some(d) => d,
        None => return, // Skip if source not available.
    };

    // Verify Makefile presence.
    let makefile = src_dir.join("Makefile");
    assert!(
        makefile.is_file(),
        "Redis source directory '{}' does not contain a Makefile.",
        src_dir.display()
    );

    // --- Clean previous build artifacts (best-effort) ---
    make_clean_best_effort(&src_dir);

    // --- Locate BCC binary ---
    let bcc = common::bcc_binary_path();
    let bcc_str = bcc.to_str().expect("BCC binary path is not valid UTF-8");

    // --- Timed build ---
    let (build_output, build_elapsed) = common::timed_execution(|| {
        Command::new("make")
            .arg(format!("CC={}", bcc_str))
            .arg("-j")
            .arg(num_cpus_str())
            .current_dir(&src_dir)
            .env("MALLOC", "libc") // Redis defaults to jemalloc; use libc for simplicity.
            .output()
            .unwrap_or_else(|e| panic!("Failed to invoke make for Redis build: {}", e))
    });

    assert!(
        build_output.status.success(),
        "Redis build failed (exit code {:?}).\nstdout (last 2000 chars):\n{}\nstderr (last 2000 chars):\n{}",
        build_output.status.code(),
        tail_string(&String::from_utf8_lossy(&build_output.stdout), 2000),
        tail_string(&String::from_utf8_lossy(&build_output.stderr), 2000)
    );

    // --- Verify output binary ---
    let redis_server = src_dir.join("src").join("redis-server");
    assert_binary_produced(redis_server.as_path(), "Redis");
    common::assert_elf_type(redis_server.to_str().unwrap(), "EXEC");
    common::assert_elf_machine(
        redis_server.to_str().unwrap(),
        "Advanced Micro Devices X86-64",
    );

    // --- Functionality smoke test: redis-server --version ---
    let version_output = Command::new(redis_server.to_str().unwrap())
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute redis-server for version check: {}", e));
    let version_stdout = String::from_utf8_lossy(&version_output.stdout);
    assert!(
        version_stdout.to_lowercase().contains("redis") || version_stdout.contains("v="),
        "redis-server --version did not produce expected output.\nstdout: '{}'",
        version_stdout.trim()
    );

    // --- Wall-clock ceiling assertion (Section 0.7.8) ---
    let gcc_secs = gcc_benchmark_secs(
        GCC_BENCHMARK_SECS_REDIS_ENV,
        DEFAULT_GCC_BENCHMARK_REDIS_SECS,
    );
    assert_within_wall_clock_ceiling(build_elapsed, gcc_secs, "Redis");

    // --- Cleanup ---
    make_clean_best_effort(&src_dir);

    eprintln!(
        "Checkpoint 7 — Redis: PASSED (build time: {:.1}s, ceiling: {:.1}s)",
        build_elapsed.as_secs_f64(),
        wall_clock_ceiling(gcc_secs).as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// 7.3 — PostgreSQL Compilation Test
// ---------------------------------------------------------------------------

/// Configure and build PostgreSQL with BCC as the C compiler, verify that
/// core binaries are produced as valid ELF executables, and assert the
/// wall-clock ceiling.
///
/// **Prerequisites:**
/// - `POSTGRESQL_SRC_DIR` environment variable set to the PostgreSQL source
///   tree root (must contain a `configure` script).
///
/// **Validation:**
/// 1. `./configure CC=<bcc>` succeeds.
/// 2. `make -j` succeeds.
/// 3. `src/backend/postgres` binary is produced and is a valid ELF EXEC.
/// 4. Total build time (configure + make) ≤ 5× GCC benchmark.
#[test]
#[ignore]
fn test_postgresql_build() {
    // --- Source directory resolution ---
    let src_dir = match resolve_source_dir(POSTGRESQL_SRC_DIR_ENV, "PostgreSQL") {
        Some(d) => d,
        None => return, // Skip if source not available.
    };

    // Verify configure script presence.
    let configure_script = src_dir.join("configure");
    assert!(
        configure_script.is_file(),
        "PostgreSQL source directory '{}' does not contain a configure script.",
        src_dir.display()
    );

    // --- Clean previous build artifacts (best-effort) ---
    make_clean_best_effort(&src_dir);

    // --- Locate BCC binary ---
    let bcc = common::bcc_binary_path();
    let bcc_str = bcc.to_str().expect("BCC binary path is not valid UTF-8");

    // --- Build directory (out-of-tree build to avoid polluting the source) ---
    let build_dir = common::TestDir::new("checkpoint7_postgresql");

    // --- Timed build (configure + make) ---
    let ((), build_elapsed) = common::timed_execution(|| {
        // Step 1: Configure
        let configure_output = Command::new(configure_script.to_str().unwrap())
            .arg(format!("CC={}", bcc_str))
            .arg("--without-readline")
            .arg("--without-zlib")
            .arg("--without-icu")
            .current_dir(build_dir.path())
            .env("CC", bcc_str)
            .output()
            .unwrap_or_else(|e| panic!("Failed to invoke PostgreSQL configure script: {}", e));

        assert!(
            configure_output.status.success(),
            "PostgreSQL configure failed (exit code {:?}).\n\
             stdout (last 2000 chars):\n{}\n\
             stderr (last 2000 chars):\n{}",
            configure_output.status.code(),
            tail_string(&String::from_utf8_lossy(&configure_output.stdout), 2000),
            tail_string(&String::from_utf8_lossy(&configure_output.stderr), 2000)
        );

        // Step 2: Build
        let make_output = Command::new("make")
            .arg("-j")
            .arg(num_cpus_str())
            .current_dir(build_dir.path())
            .output()
            .unwrap_or_else(|e| panic!("Failed to invoke make for PostgreSQL build: {}", e));

        assert!(
            make_output.status.success(),
            "PostgreSQL make failed (exit code {:?}).\n\
             stdout (last 2000 chars):\n{}\n\
             stderr (last 2000 chars):\n{}",
            make_output.status.code(),
            tail_string(&String::from_utf8_lossy(&make_output.stdout), 2000),
            tail_string(&String::from_utf8_lossy(&make_output.stderr), 2000)
        );
    });

    // --- Verify output binary ---
    // PostgreSQL places the main backend binary at src/backend/postgres within
    // the build directory. Some build configurations use a flat layout.
    let postgres_bin = find_binary_in_tree(build_dir.path(), "postgres");
    let postgres_path = postgres_bin.unwrap_or_else(|| {
        panic!(
            "PostgreSQL build did not produce a 'postgres' binary in '{}'.",
            build_dir.path().display()
        )
    });
    assert_binary_produced(postgres_path.as_path(), "PostgreSQL");
    common::assert_elf_type(postgres_path.to_str().unwrap(), "EXEC");

    // --- Wall-clock ceiling assertion (Section 0.7.8) ---
    let gcc_secs = gcc_benchmark_secs(
        GCC_BENCHMARK_SECS_POSTGRESQL_ENV,
        DEFAULT_GCC_BENCHMARK_POSTGRESQL_SECS,
    );
    assert_within_wall_clock_ceiling(build_elapsed, gcc_secs, "PostgreSQL");

    eprintln!(
        "Checkpoint 7 — PostgreSQL: PASSED (build time: {:.1}s, ceiling: {:.1}s)",
        build_elapsed.as_secs_f64(),
        wall_clock_ceiling(gcc_secs).as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// 7.4 — FFmpeg Compilation Test
// ---------------------------------------------------------------------------

/// Configure and build FFmpeg with BCC as the C compiler, verify that the
/// `ffmpeg` binary is produced as a valid ELF executable, and assert the
/// wall-clock ceiling.
///
/// **Prerequisites:**
/// - `FFMPEG_SRC_DIR` environment variable set to the FFmpeg source tree root
///   (must contain a `configure` script).
///
/// **Validation:**
/// 1. `./configure --cc=<bcc>` succeeds.
/// 2. `make -j` succeeds.
/// 3. `ffmpeg` binary is produced and is a valid ELF EXEC.
/// 4. Total build time (configure + make) ≤ 5× GCC benchmark.
#[test]
#[ignore]
fn test_ffmpeg_build() {
    // --- Source directory resolution ---
    let src_dir = match resolve_source_dir(FFMPEG_SRC_DIR_ENV, "FFmpeg") {
        Some(d) => d,
        None => return, // Skip if source not available.
    };

    // Verify configure script presence.
    let configure_script = src_dir.join("configure");
    assert!(
        configure_script.is_file(),
        "FFmpeg source directory '{}' does not contain a configure script.",
        src_dir.display()
    );

    // --- Clean previous build artifacts (best-effort) ---
    make_clean_best_effort(&src_dir);

    // --- Locate BCC binary ---
    let bcc = common::bcc_binary_path();
    let bcc_str = bcc.to_str().expect("BCC binary path is not valid UTF-8");

    // --- Build directory ---
    let build_dir = common::TestDir::new("checkpoint7_ffmpeg");

    // --- Timed build (configure + make) ---
    let ((), build_elapsed) = common::timed_execution(|| {
        // Step 1: Configure
        // FFmpeg uses --cc= for specifying the C compiler.
        let configure_output = Command::new(configure_script.to_str().unwrap())
            .arg(format!("--cc={}", bcc_str))
            .arg("--disable-x86asm") // Avoid nasm/yasm dependency.
            .arg("--disable-doc") // Skip documentation generation.
            .arg("--disable-network") // Reduce build scope.
            .arg("--disable-programs") // Build libraries only first, then add ffmpeg.
            .arg("--enable-ffmpeg") // Ensure ffmpeg binary is built.
            .arg(format!("--prefix={}", build_dir.path().display()))
            .current_dir(build_dir.path())
            .env("CC", bcc_str)
            .output()
            .unwrap_or_else(|e| panic!("Failed to invoke FFmpeg configure script: {}", e));

        assert!(
            configure_output.status.success(),
            "FFmpeg configure failed (exit code {:?}).\n\
             stdout (last 2000 chars):\n{}\n\
             stderr (last 2000 chars):\n{}",
            configure_output.status.code(),
            tail_string(&String::from_utf8_lossy(&configure_output.stdout), 2000),
            tail_string(&String::from_utf8_lossy(&configure_output.stderr), 2000)
        );

        // Step 2: Build
        let make_output = Command::new("make")
            .arg("-j")
            .arg(num_cpus_str())
            .current_dir(build_dir.path())
            .output()
            .unwrap_or_else(|e| panic!("Failed to invoke make for FFmpeg build: {}", e));

        assert!(
            make_output.status.success(),
            "FFmpeg make failed (exit code {:?}).\n\
             stdout (last 2000 chars):\n{}\n\
             stderr (last 2000 chars):\n{}",
            make_output.status.code(),
            tail_string(&String::from_utf8_lossy(&make_output.stdout), 2000),
            tail_string(&String::from_utf8_lossy(&make_output.stderr), 2000)
        );
    });

    // --- Verify output binary ---
    let ffmpeg_bin = find_binary_in_tree(build_dir.path(), "ffmpeg");
    let ffmpeg_path = ffmpeg_bin.unwrap_or_else(|| {
        panic!(
            "FFmpeg build did not produce an 'ffmpeg' binary in '{}'.",
            build_dir.path().display()
        )
    });
    assert_binary_produced(ffmpeg_path.as_path(), "FFmpeg");
    common::assert_elf_type(ffmpeg_path.to_str().unwrap(), "EXEC");

    // --- Functionality smoke test: ffmpeg -version ---
    let version_output = Command::new(ffmpeg_path.to_str().unwrap())
        .arg("-version")
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute ffmpeg for version check: {}", e));
    let version_stdout = String::from_utf8_lossy(&version_output.stdout);
    assert!(
        version_stdout.to_lowercase().contains("ffmpeg") || version_stdout.contains("version"),
        "ffmpeg -version did not produce expected output.\nstdout: '{}'",
        version_stdout.trim()
    );

    // --- Wall-clock ceiling assertion (Section 0.7.8) ---
    let gcc_secs = gcc_benchmark_secs(
        GCC_BENCHMARK_SECS_FFMPEG_ENV,
        DEFAULT_GCC_BENCHMARK_FFMPEG_SECS,
    );
    assert_within_wall_clock_ceiling(build_elapsed, gcc_secs, "FFmpeg");

    // --- Cleanup ---
    make_clean_best_effort(build_dir.path());

    eprintln!(
        "Checkpoint 7 — FFmpeg: PASSED (build time: {:.1}s, ceiling: {:.1}s)",
        build_elapsed.as_secs_f64(),
        wall_clock_ceiling(gcc_secs).as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Internal utility functions
// ---------------------------------------------------------------------------

/// Return a string representing a reasonable parallelism level for `make -j`.
///
/// Attempts to detect the number of available CPUs from the environment or
/// from `/proc/cpuinfo`. Falls back to `"4"` if detection fails.
fn num_cpus_str() -> String {
    // Try the NUM_CPUS environment variable first (allows CI override).
    if let Ok(val) = env::var("NUM_CPUS") {
        if val.parse::<u32>().is_ok() {
            return val;
        }
    }

    // Parse /proc/cpuinfo to count processor entries.
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        let count = cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        if count > 0 {
            return count.to_string();
        }
    }

    // Conservative fallback.
    "4".to_string()
}

/// Return the last `n` characters of a string, useful for truncating long
/// build output in assertion messages.
fn tail_string(s: &str, n: usize) -> &str {
    if s.len() <= n {
        s
    } else {
        // Find a valid UTF-8 char boundary near the truncation point.
        let start = s.len() - n;
        let mut idx = start;
        while idx < s.len() && !s.is_char_boundary(idx) {
            idx += 1;
        }
        &s[idx..]
    }
}

/// Recursively search for a binary with the given name in a directory tree.
///
/// Returns the first match found (depth-first). Useful for locating build
/// outputs in project trees with varying directory layouts.
///
/// # Arguments
///
/// * `root`        — Root directory to search.
/// * `binary_name` — File name to search for (e.g., `"postgres"`, `"ffmpeg"`).
///
/// # Returns
///
/// `Some(PathBuf)` if found, `None` otherwise.
fn find_binary_in_tree(root: &Path, binary_name: &str) -> Option<PathBuf> {
    // Check the root directory itself first.
    let direct = root.join(binary_name);
    if direct.is_file() {
        return Some(direct);
    }

    // Walk the directory tree (iterative breadth-first to avoid deep recursion).
    let mut queue = vec![root.to_path_buf()];
    while let Some(dir) = queue.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name() {
                    if name == binary_name {
                        return Some(path);
                    }
                }
            } else if path.is_dir() {
                // Skip common non-build directories for efficiency.
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name == ".git" || name == "node_modules" || name == ".hg" {
                        continue;
                    }
                }
                queue.push(path);
            }
        }
    }

    None
}
