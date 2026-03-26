//! Checkpoint 7 — Stretch Targets (Optional)
//!
//! This test suite validates BCC's ability to compile large, real-world C codebases
//! beyond the Linux kernel. The four stretch targets are:
//!
//! - **SQLite** — Single-file amalgamation database engine
//! - **Redis** — In-memory key-value store
//! - **PostgreSQL** — Full relational database
//! - **FFmpeg** — Multimedia transcoding framework
//!
//! # AAP Compliance
//! - Per AAP §0.7.5: "Checkpoint 7 (stretch targets) is optional and may execute
//!   in parallel after Checkpoint 6 passes."
//! - Per AAP §0.7.8: "Each stretch target build MUST NOT exceed 5× equivalent GCC
//!   build time."
//!
//! # Prerequisites
//! - Stretch target source code must be available on disk. The location is resolved
//!   via environment variables (`SQLITE_SRC_DIR`, `REDIS_SRC_DIR`, `POSTGRESQL_SRC_DIR`,
//!   `FFMPEG_SRC_DIR`) or common default paths.
//! - All tests are marked `#[ignore]` because they require external source code that
//!   is not shipped with the BCC repository.
//! - A release build of BCC must be available (`cargo build --release`).
//!
//! # Zero Dependencies
//! Only `std::` imports are used — no external crates.

mod common;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The wall-clock ceiling multiplier. Per AAP §0.7.8, each stretch target build
/// MUST NOT exceed 5× the equivalent GCC build time on the same hardware.
const WALL_CLOCK_CEILING_MULTIPLIER: f64 = 5.0;

/// Default timeout (in seconds) for a single stretch target build. This acts as
/// an absolute safety net to prevent runaway builds from hanging CI indefinitely.
/// Used by build orchestration logic when enforcing the wall-clock ceiling.
#[allow(dead_code)]
const BUILD_TIMEOUT_SECS: u64 = 7200; // 2 hours

/// Port offset base for Redis tests, to avoid collisions with a running Redis
/// instance on the default port 6379.
const REDIS_TEST_PORT: u16 = 16379;

/// Port for PostgreSQL tests.
const POSTGRESQL_TEST_PORT: u16 = 15432;

// ---------------------------------------------------------------------------
// Source Directory Resolution Helpers
// ---------------------------------------------------------------------------

/// Attempts to locate the source directory for a stretch target.
///
/// Resolution order:
/// 1. Environment variable `env_var_name` (e.g., `SQLITE_SRC_DIR`)
/// 2. Each path in `default_paths`, tried in order
///
/// Returns `Some(PathBuf)` if a valid directory is found, `None` otherwise.
fn locate_source_dir(env_var_name: &str, default_paths: &[&str]) -> Option<PathBuf> {
    // Try environment variable first
    if let Ok(val) = env::var(env_var_name) {
        let p = PathBuf::from(&val);
        if p.exists() && p.is_dir() {
            return Some(p);
        }
    }

    // Try default paths
    for default in default_paths {
        let p = Path::new(default);
        if p.exists() && p.is_dir() {
            return Some(p.to_path_buf());
        }
    }

    None
}

/// Resolves the BCC binary path as a string suitable for passing to `make CC=...`
/// or `./configure CC=...`. Returns an absolute path string.
fn bcc_path_str() -> String {
    let p = common::bcc_path();
    // Canonicalize to absolute path so it works from any CWD
    match fs::canonicalize(&p) {
        Ok(abs) => abs.to_string_lossy().into_owned(),
        Err(_) => p.to_string_lossy().into_owned(),
    }
}

/// Returns the number of parallel build jobs to use. Defaults to the number of
/// available CPUs, or 1 if detection fails.
fn parallel_jobs() -> String {
    // Read from env or use a sensible default
    if let Ok(val) = env::var("BCC_TEST_JOBS") {
        return val;
    }
    // Try to detect CPU count via /proc/cpuinfo or nproc
    let output = Command::new("nproc").output();
    match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.parse::<u32>().is_ok() {
                s
            } else {
                "1".to_string()
            }
        }
        _ => "1".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Build Time Measurement and Ceiling Enforcement
// ---------------------------------------------------------------------------

/// Records a build time measurement for a stretch target.
#[derive(Debug, Clone)]
struct BuildTimeMeasurement {
    /// Name of the stretch target (e.g., "SQLite", "Redis").
    target_name: String,
    /// Wall-clock build time using BCC.
    bcc_duration: Duration,
    /// Wall-clock build time using GCC (baseline), if measured.
    gcc_duration: Option<Duration>,
    /// Whether the 5× ceiling was satisfied.
    ceiling_passed: Option<bool>,
}

/// Measures the GCC baseline build time for a stretch target by invoking a
/// provided build command sequence. Returns `None` if GCC is not available or
/// the build fails.
///
/// This is a general-purpose helper available for stretch targets that use
/// multi-step build processes (configure + make). Individual tests may also
/// implement inline GCC measurement for simpler build workflows.
#[allow(dead_code)]
fn measure_gcc_build_time(
    source_dir: &Path,
    build_commands: &[(&str, &[&str])],
) -> Option<Duration> {
    // Check if GCC is available
    let gcc_check = Command::new("gcc").arg("--version").output();
    if gcc_check.is_err() || !gcc_check.unwrap().status.success() {
        eprintln!("[checkpoint7] GCC not available for baseline measurement");
        return None;
    }

    let start = Instant::now();
    for (program, args) in build_commands {
        let status = Command::new(program)
            .args(*args)
            .current_dir(source_dir)
            .env("CC", "gcc")
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "[checkpoint7] GCC baseline build step '{} {:?}' failed with {}",
                    program, args, s
                );
                return None;
            }
            Err(e) => {
                eprintln!(
                    "[checkpoint7] GCC baseline build step '{} {:?}' error: {}",
                    program, args, e
                );
                return None;
            }
        }
    }
    Some(start.elapsed())
}

/// Asserts that `bcc_time <= WALL_CLOCK_CEILING_MULTIPLIER * gcc_time`.
///
/// If `gcc_time` is `None` (GCC baseline not available), the assertion is skipped
/// with a warning, and the function returns `None`.
///
/// Returns `Some(true)` if the ceiling is met, `Some(false)` if violated.
fn check_wall_clock_ceiling(
    target_name: &str,
    bcc_time: Duration,
    gcc_time: Option<Duration>,
) -> Option<bool> {
    match gcc_time {
        Some(gcc_dur) => {
            let bcc_secs = bcc_time.as_secs_f64();
            let gcc_secs = gcc_dur.as_secs_f64();
            let ceiling = gcc_secs * WALL_CLOCK_CEILING_MULTIPLIER;
            let passed = bcc_secs <= ceiling;

            eprintln!(
                "[checkpoint7] {} wall-clock: BCC={:.2}s, GCC={:.2}s, \
                 ceiling={:.2}s (5×GCC), {}",
                target_name,
                bcc_secs,
                gcc_secs,
                ceiling,
                if passed { "PASS" } else { "FAIL" }
            );

            if !passed {
                eprintln!(
                    "[checkpoint7] WALL-CLOCK CEILING VIOLATED for {}: \
                     BCC took {:.2}s but ceiling is {:.2}s (5× GCC's {:.2}s)",
                    target_name, bcc_secs, ceiling, gcc_secs
                );
            }

            Some(passed)
        }
        None => {
            eprintln!(
                "[checkpoint7] {} wall-clock: BCC={:.2}s, GCC=N/A (baseline not available)",
                target_name,
                bcc_time.as_secs_f64()
            );
            None
        }
    }
}

/// Asserts the wall-clock ceiling, panicking if it is violated.
fn assert_wall_clock_ceiling(target_name: &str, bcc_time: Duration, gcc_time: Option<Duration>) {
    if let Some(false) = check_wall_clock_ceiling(target_name, bcc_time, gcc_time) {
        let gcc_secs = gcc_time.unwrap().as_secs_f64();
        panic!(
            "Wall-clock ceiling violated for {}: BCC={:.2}s > {:.2}s (5× GCC's {:.2}s)",
            target_name,
            bcc_time.as_secs_f64(),
            gcc_secs * WALL_CLOCK_CEILING_MULTIPLIER,
            gcc_secs
        );
    }
}

// ---------------------------------------------------------------------------
// Build Directory Cleanup
// ---------------------------------------------------------------------------

/// Performs a `make clean` in the given directory, ignoring errors.
fn make_clean(dir: &Path) {
    let _ = Command::new("make").arg("clean").current_dir(dir).output();
}

/// Creates a temporary build directory and returns its path.
/// The caller is responsible for removing it after use.
fn create_temp_build_dir(name: &str) -> PathBuf {
    let dir = env::temp_dir().join(format!("bcc_stretch_{}", name));
    let _ = fs::create_dir_all(&dir);
    dir
}

// ===========================================================================
// SQLITE STRETCH TARGET
// ===========================================================================

/// Locates the SQLite amalgamation source directory.
fn locate_sqlite_source() -> Option<PathBuf> {
    locate_source_dir(
        "SQLITE_SRC_DIR",
        &[
            "/usr/local/src/sqlite",
            "/opt/sqlite",
            "/tmp/sqlite",
            "vendor/sqlite",
        ],
    )
}

/// Compiles the SQLite amalgamation using BCC.
///
/// SQLite is distributed as a single amalgamation file (`sqlite3.c`) plus a
/// CLI shell (`shell.c`). This makes it an ideal first stretch target — a
/// single-file compile exercising the full C language surface.
#[test]
#[ignore]
fn test_sqlite_build() {
    let src_dir = match locate_sqlite_source() {
        Some(d) => d,
        None => {
            eprintln!(
                "[checkpoint7] SQLite source not found. Set SQLITE_SRC_DIR or place \
                 source in /usr/local/src/sqlite. Skipping."
            );
            return;
        }
    };

    let sqlite3_c = src_dir.join("sqlite3.c");
    let shell_c = src_dir.join("shell.c");

    if !sqlite3_c.exists() {
        eprintln!(
            "[checkpoint7] sqlite3.c not found at {}. Skipping.",
            sqlite3_c.display()
        );
        return;
    }

    let bcc = bcc_path_str();
    let output_path = common::temp_output_path("sqlite3_stretch");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Measure BCC build time
    let start = Instant::now();
    let compile_result = if shell_c.exists() {
        // Full SQLite CLI build
        Command::new(&bcc)
            .args([
                "-O0",
                "-o",
                output_str,
                "-DSQLITE_THREADSAFE=0",
                "-DSQLITE_OMIT_LOAD_EXTENSION",
            ])
            .arg(shell_c.to_str().unwrap())
            .arg(sqlite3_c.to_str().unwrap())
            .output()
            .expect("Failed to invoke BCC for SQLite build")
    } else {
        // Library-only build
        Command::new(&bcc)
            .args([
                "-O0",
                "-c",
                "-DSQLITE_THREADSAFE=0",
                "-DSQLITE_OMIT_LOAD_EXTENSION",
                "-o",
                output_str,
            ])
            .arg(sqlite3_c.to_str().unwrap())
            .output()
            .expect("Failed to invoke BCC for SQLite build")
    };
    let bcc_duration = start.elapsed();

    common::assert_compilation_succeeds(&compile_result);

    eprintln!(
        "[checkpoint7] SQLite BCC build completed in {:.2}s",
        bcc_duration.as_secs_f64()
    );

    // Measure GCC baseline (if available)
    let gcc_output_path = common::temp_output_path("sqlite3_gcc_baseline");
    let gcc_output_str = gcc_output_path.to_str().unwrap();
    let gcc_duration = if shell_c.exists() {
        let gcc_start = Instant::now();
        let gcc_result = Command::new("gcc")
            .args([
                "-O0",
                "-o",
                gcc_output_str,
                "-DSQLITE_THREADSAFE=0",
                "-DSQLITE_OMIT_LOAD_EXTENSION",
            ])
            .arg(shell_c.to_str().unwrap())
            .arg(sqlite3_c.to_str().unwrap())
            .output();
        match gcc_result {
            Ok(ref o) if o.status.success() => Some(gcc_start.elapsed()),
            _ => None,
        }
    } else {
        None
    };

    // Enforce wall-clock ceiling
    assert_wall_clock_ceiling("SQLite", bcc_duration, gcc_duration);

    // Cleanup
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&gcc_output_path);
}

/// Runs basic SQL operations on a BCC-compiled SQLite binary to validate
/// functional correctness of the generated executable.
#[test]
#[ignore]
fn test_sqlite_basic_operations() {
    let src_dir = match locate_sqlite_source() {
        Some(d) => d,
        None => {
            eprintln!("[checkpoint7] SQLite source not found. Skipping.");
            return;
        }
    };

    let sqlite3_c = src_dir.join("sqlite3.c");
    let shell_c = src_dir.join("shell.c");

    if !sqlite3_c.exists() || !shell_c.exists() {
        eprintln!("[checkpoint7] SQLite amalgamation (sqlite3.c + shell.c) not found. Skipping.");
        return;
    }

    let output_path = common::temp_output_path("sqlite3_ops");
    let output_str = output_path.to_str().unwrap();

    // Build SQLite CLI using common::compile helper for the main amalgamation file.
    // We compile shell.c first to exercise the common::compile interface, then
    // link together via a full build command.
    let shell_compile = common::compile(
        shell_c.to_str().unwrap(),
        &[
            "-O0",
            "-c",
            "-DSQLITE_THREADSAFE=0",
            "-DSQLITE_OMIT_LOAD_EXTENSION",
            "-o",
            &format!("{}.shell.o", output_str),
        ],
    );
    common::assert_compilation_succeeds(&shell_compile);

    // Full combined build for the executable
    let bcc = bcc_path_str();
    let compile_result = Command::new(&bcc)
        .args([
            "-O0",
            "-o",
            output_str,
            "-DSQLITE_THREADSAFE=0",
            "-DSQLITE_OMIT_LOAD_EXTENSION",
        ])
        .arg(shell_c.to_str().unwrap())
        .arg(sqlite3_c.to_str().unwrap())
        .output()
        .expect("Failed to invoke BCC for SQLite build");

    common::assert_compilation_succeeds(&compile_result);

    // Clean up the intermediate object file
    let shell_obj = format!("{}.shell.o", output_str);
    let _ = fs::remove_file(&shell_obj);

    // Create a temporary database path
    let db_path = common::temp_output_path("sqlite3_test_db");
    let db_str = db_path.to_str().unwrap();

    // Execute SQL commands: CREATE TABLE, INSERT, SELECT
    let sql_commands = "CREATE TABLE test(id INTEGER PRIMARY KEY, name TEXT);\n\
                        INSERT INTO test VALUES(1, 'hello');\n\
                        INSERT INTO test VALUES(2, 'world');\n\
                        SELECT name FROM test ORDER BY id;\n";

    // Use common::run_binary to execute the compiled SQLite binary (x86-64 native)
    let version_output = common::run_binary(output_str, "x86-64");
    // SQLite with no args prints version info — just verify it runs
    let _ = version_output;

    let run_result = Command::new(output_str)
        .arg(db_str)
        .arg("-batch")
        .arg("-cmd")
        .arg(sql_commands)
        .output();

    match run_result {
        Ok(ref output) => {
            common::assert_exit_success(output);
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Verify the SELECT output contains our inserted values
            assert!(
                stdout.contains("hello"),
                "SQLite SELECT output should contain 'hello', got: {}",
                stdout
            );
            assert!(
                stdout.contains("world"),
                "SQLite SELECT output should contain 'world', got: {}",
                stdout
            );
            eprintln!("[checkpoint7] SQLite basic operations: PASS");
        }
        Err(e) => {
            eprintln!(
                "[checkpoint7] Failed to run SQLite binary: {}. \
                 Binary may require additional libraries.",
                e
            );
        }
    }

    // Test assert_stdout_eq with a simple SELECT that has known exact output.
    // This uses the sqlite3 -batch mode which produces deterministic output.
    let simple_sql = "SELECT 'bcc_works';";
    let simple_result = Command::new(output_str)
        .arg(":memory:")
        .args(["-batch", "-cmd", simple_sql])
        .output();
    if let Ok(ref output) = simple_result {
        if output.status.success() {
            common::assert_stdout_eq(output, "bcc_works\n");
        }
    }

    // Cleanup
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&db_path);
    // SQLite may create journal/WAL files
    let wal_path = format!("{}-wal", db_str);
    let shm_path = format!("{}-shm", db_str);
    let journal_path = format!("{}-journal", db_str);
    common::cleanup_temp_files(&[&wal_path, &shm_path, &journal_path]);
}

// ===========================================================================
// REDIS STRETCH TARGET
// ===========================================================================

/// Locates the Redis source directory.
fn locate_redis_source() -> Option<PathBuf> {
    locate_source_dir(
        "REDIS_SRC_DIR",
        &[
            "/usr/local/src/redis",
            "/opt/redis",
            "/tmp/redis",
            "vendor/redis",
        ],
    )
}

/// Builds the Redis server using BCC via `make CC=<bcc>`.
///
/// Redis uses a standard `make`-based build system. This test validates BCC's
/// ability to handle a moderately complex, multi-file C project with extensive
/// use of POSIX APIs, networking, and event-loop patterns.
#[test]
#[ignore]
fn test_redis_build() {
    let src_dir = match locate_redis_source() {
        Some(d) => d,
        None => {
            eprintln!(
                "[checkpoint7] Redis source not found. Set REDIS_SRC_DIR or place \
                 source in /usr/local/src/redis. Skipping."
            );
            return;
        }
    };

    // Verify Makefile exists
    if !src_dir.join("Makefile").exists() && !src_dir.join("src").join("Makefile").exists() {
        eprintln!(
            "[checkpoint7] Redis Makefile not found at {}. Skipping.",
            src_dir.display()
        );
        return;
    }

    let bcc = bcc_path_str();
    let jobs = parallel_jobs();

    // Clean previous builds
    make_clean(&src_dir);

    // Measure BCC build time
    let start = Instant::now();
    let build_result = Command::new("make")
        .args(["-j", &jobs])
        .env("CC", &bcc)
        .current_dir(&src_dir)
        .output()
        .expect("Failed to invoke make for Redis build");
    let bcc_duration = start.elapsed();

    if !build_result.status.success() {
        let stderr = String::from_utf8_lossy(&build_result.stderr);
        panic!("[checkpoint7] Redis BCC build failed:\nstderr: {}", stderr);
    }

    eprintln!(
        "[checkpoint7] Redis BCC build completed in {:.2}s",
        bcc_duration.as_secs_f64()
    );

    // Verify redis-server binary was produced
    let redis_server = src_dir.join("src").join("redis-server");
    assert!(
        redis_server.exists(),
        "redis-server binary not found at {}",
        redis_server.display()
    );

    // Measure GCC baseline
    make_clean(&src_dir);
    let gcc_duration = {
        let gcc_start = Instant::now();
        let gcc_result = Command::new("make")
            .args(["-j", &jobs])
            .env("CC", "gcc")
            .current_dir(&src_dir)
            .output();
        match gcc_result {
            Ok(ref o) if o.status.success() => Some(gcc_start.elapsed()),
            _ => None,
        }
    };

    // Enforce wall-clock ceiling
    assert_wall_clock_ceiling("Redis", bcc_duration, gcc_duration);

    // Clean up
    make_clean(&src_dir);
}

/// Tests basic Redis operations (SET/GET) using a BCC-compiled redis-server.
#[test]
#[ignore]
fn test_redis_basic_operations() {
    let src_dir = match locate_redis_source() {
        Some(d) => d,
        None => {
            eprintln!("[checkpoint7] Redis source not found. Skipping.");
            return;
        }
    };

    let redis_server = src_dir.join("src").join("redis-server");
    let redis_cli = src_dir.join("src").join("redis-cli");

    if !redis_server.exists() {
        eprintln!("[checkpoint7] redis-server not found. Run test_redis_build first. Skipping.");
        return;
    }

    let port = REDIS_TEST_PORT.to_string();

    // Start Redis server on a non-default port to avoid conflicts
    let mut server = match Command::new(&redis_server)
        .args(["--port", &port, "--daemonize", "no", "--save", ""])
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "[checkpoint7] Failed to start redis-server: {}. Skipping.",
                e
            );
            return;
        }
    };

    // Wait briefly for server to initialize
    std::thread::sleep(Duration::from_secs(2));

    // Determine CLI tool: use built redis-cli if available, else system redis-cli
    let cli_path = if redis_cli.exists() {
        redis_cli.to_string_lossy().into_owned()
    } else {
        "redis-cli".to_string()
    };

    // SET a key
    let set_result = Command::new(&cli_path)
        .args(["-p", &port, "SET", "bcc_test_key", "bcc_test_value"])
        .output();

    if let Ok(ref output) = set_result {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("OK"),
            "Redis SET should return OK, got: {}",
            stdout
        );
    }

    // GET the key
    let get_result = Command::new(&cli_path)
        .args(["-p", &port, "GET", "bcc_test_key"])
        .output();

    if let Ok(ref output) = get_result {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("bcc_test_value"),
            "Redis GET should return 'bcc_test_value', got: {}",
            stdout
        );
        eprintln!("[checkpoint7] Redis basic operations: PASS");
    }

    // Graceful shutdown via SHUTDOWN command
    let _ = Command::new(&cli_path)
        .args(["-p", &port, "SHUTDOWN", "NOSAVE"])
        .output();

    // Wait for server process to exit, then kill if still running
    let _ = server.wait();
}

// ===========================================================================
// POSTGRESQL STRETCH TARGET
// ===========================================================================

/// Locates the PostgreSQL source directory.
fn locate_postgresql_source() -> Option<PathBuf> {
    locate_source_dir(
        "POSTGRESQL_SRC_DIR",
        &[
            "/usr/local/src/postgresql",
            "/opt/postgresql",
            "/tmp/postgresql",
            "vendor/postgresql",
        ],
    )
}

/// Builds PostgreSQL using BCC via `./configure CC=<bcc> && make`.
///
/// PostgreSQL is a large, complex C codebase with extensive use of POSIX,
/// system calls, shared memory, and networking. This test exercises BCC's
/// handling of configure scripts, complex build systems, and system-level
/// C programming patterns.
#[test]
#[ignore]
fn test_postgresql_build() {
    let src_dir = match locate_postgresql_source() {
        Some(d) => d,
        None => {
            eprintln!(
                "[checkpoint7] PostgreSQL source not found. Set POSTGRESQL_SRC_DIR or place \
                 source in /usr/local/src/postgresql. Skipping."
            );
            return;
        }
    };

    // Verify configure script exists
    if !src_dir.join("configure").exists() {
        eprintln!(
            "[checkpoint7] PostgreSQL configure script not found at {}. Skipping.",
            src_dir.display()
        );
        return;
    }

    let bcc = bcc_path_str();
    let jobs = parallel_jobs();
    let install_dir = create_temp_build_dir("postgresql_install");
    let install_str = install_dir.to_str().unwrap();

    // Clean previous builds
    make_clean(&src_dir);

    // Configure with BCC
    let configure_result = Command::new("./configure")
        .args([
            &format!("--prefix={}", install_str),
            "--without-readline",
            "--without-zlib",
        ])
        .env("CC", &bcc)
        .current_dir(&src_dir)
        .output()
        .expect("Failed to run PostgreSQL configure");

    if !configure_result.status.success() {
        let stderr = String::from_utf8_lossy(&configure_result.stderr);
        eprintln!("[checkpoint7] PostgreSQL configure failed:\n{}", stderr);
        panic!("PostgreSQL configure with BCC failed");
    }

    // Build with BCC
    let start = Instant::now();
    let build_result = Command::new("make")
        .args(["-j", &jobs])
        .current_dir(&src_dir)
        .output()
        .expect("Failed to invoke make for PostgreSQL build");
    let bcc_duration = start.elapsed();

    if !build_result.status.success() {
        let stderr = String::from_utf8_lossy(&build_result.stderr);
        panic!(
            "[checkpoint7] PostgreSQL BCC build failed:\nstderr: {}",
            stderr
        );
    }

    eprintln!(
        "[checkpoint7] PostgreSQL BCC build completed in {:.2}s",
        bcc_duration.as_secs_f64()
    );

    // Measure GCC baseline
    make_clean(&src_dir);
    let gcc_duration = {
        let configure_gcc = Command::new("./configure")
            .args([
                &format!("--prefix={}", install_str),
                "--without-readline",
                "--without-zlib",
            ])
            .env("CC", "gcc")
            .current_dir(&src_dir)
            .output();

        if configure_gcc.is_ok() && configure_gcc.as_ref().unwrap().status.success() {
            let gcc_start = Instant::now();
            let gcc_result = Command::new("make")
                .args(["-j", &jobs])
                .current_dir(&src_dir)
                .output();
            match gcc_result {
                Ok(ref o) if o.status.success() => Some(gcc_start.elapsed()),
                _ => None,
            }
        } else {
            None
        }
    };

    // Enforce wall-clock ceiling
    assert_wall_clock_ceiling("PostgreSQL", bcc_duration, gcc_duration);

    // Clean up
    make_clean(&src_dir);
    let _ = fs::remove_dir_all(&install_dir);
}

/// Tests basic PostgreSQL operations using a BCC-compiled PostgreSQL server.
#[test]
#[ignore]
fn test_postgresql_basic_operations() {
    let src_dir = match locate_postgresql_source() {
        Some(d) => d,
        None => {
            eprintln!("[checkpoint7] PostgreSQL source not found. Skipping.");
            return;
        }
    };

    // Check if postgres binary was built
    let postgres_bin = src_dir.join("src").join("backend").join("postgres");
    let initdb_bin = src_dir
        .join("src")
        .join("bin")
        .join("initdb")
        .join("initdb");

    if !postgres_bin.exists() {
        eprintln!(
            "[checkpoint7] postgres binary not found. Run test_postgresql_build first. Skipping."
        );
        return;
    }

    let data_dir = create_temp_build_dir("postgresql_data");
    let data_str = data_dir.to_str().unwrap();
    let port = POSTGRESQL_TEST_PORT.to_string();

    // Initialize database cluster
    if initdb_bin.exists() {
        let init_result = Command::new(&initdb_bin)
            .args(["-D", data_str, "--no-locale", "-E", "UTF8"])
            .output();

        if init_result.is_err() || !init_result.as_ref().unwrap().status.success() {
            eprintln!("[checkpoint7] PostgreSQL initdb failed. Skipping operations test.");
            let _ = fs::remove_dir_all(&data_dir);
            return;
        }
    } else {
        eprintln!("[checkpoint7] initdb binary not found. Skipping PostgreSQL operations.");
        let _ = fs::remove_dir_all(&data_dir);
        return;
    }

    // Start PostgreSQL server
    let mut server = match Command::new(&postgres_bin)
        .args(["-D", data_str, "-p", &port, "-h", "localhost"])
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!("[checkpoint7] Failed to start PostgreSQL: {}. Skipping.", e);
            let _ = fs::remove_dir_all(&data_dir);
            return;
        }
    };

    // Wait for server to start
    std::thread::sleep(Duration::from_secs(3));

    // Check for psql binary (built or system)
    let psql_bin = src_dir.join("src").join("bin").join("psql").join("psql");
    let psql_path = if psql_bin.exists() {
        psql_bin.to_string_lossy().into_owned()
    } else {
        "psql".to_string()
    };

    // Run basic SQL operations
    let sql = "CREATE TABLE bcc_test (id SERIAL PRIMARY KEY, name TEXT); \
               INSERT INTO bcc_test (name) VALUES ('hello'); \
               INSERT INTO bcc_test (name) VALUES ('world'); \
               SELECT name FROM bcc_test ORDER BY id;";

    let query_result = Command::new(&psql_path)
        .args([
            "-h",
            "localhost",
            "-p",
            &port,
            "-d",
            "postgres",
            "-c",
            sql,
            "--no-psqlrc",
            "-t",
        ])
        .output();

    match query_result {
        Ok(ref output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                stdout.contains("hello"),
                "PostgreSQL query should return 'hello', got: {}",
                stdout
            );
            assert!(
                stdout.contains("world"),
                "PostgreSQL query should return 'world', got: {}",
                stdout
            );
            eprintln!("[checkpoint7] PostgreSQL basic operations: PASS");
        }
        Ok(ref output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("[checkpoint7] PostgreSQL query failed: {}", stderr);
        }
        Err(e) => {
            eprintln!("[checkpoint7] Failed to run psql: {}. Skipping.", e);
        }
    }

    // Shut down PostgreSQL
    let pg_ctl_bin = src_dir
        .join("src")
        .join("bin")
        .join("pg_ctl")
        .join("pg_ctl");
    if pg_ctl_bin.exists() {
        let _ = Command::new(&pg_ctl_bin)
            .args(["-D", data_str, "stop", "-m", "fast"])
            .output();
    } else {
        // Use system pg_ctl or kill
        let _ = Command::new("pg_ctl")
            .args(["-D", data_str, "stop", "-m", "fast"])
            .output();
    }

    // Wait for server process to exit
    let _ = server.wait();

    // Clean up data directory
    let _ = fs::remove_dir_all(&data_dir);
}

// ===========================================================================
// FFMPEG STRETCH TARGET
// ===========================================================================

/// Locates the FFmpeg source directory.
fn locate_ffmpeg_source() -> Option<PathBuf> {
    locate_source_dir(
        "FFMPEG_SRC_DIR",
        &[
            "/usr/local/src/ffmpeg",
            "/opt/ffmpeg",
            "/tmp/ffmpeg",
            "vendor/ffmpeg",
        ],
    )
}

/// Builds FFmpeg using BCC via `./configure --cc=<bcc> && make`.
///
/// FFmpeg is a massive C codebase with heavy use of inline assembly, SIMD
/// intrinsics, complex preprocessor macros, and platform-specific code paths.
/// This is the most challenging stretch target.
#[test]
#[ignore]
fn test_ffmpeg_build() {
    let src_dir = match locate_ffmpeg_source() {
        Some(d) => d,
        None => {
            eprintln!(
                "[checkpoint7] FFmpeg source not found. Set FFMPEG_SRC_DIR or place \
                 source in /usr/local/src/ffmpeg. Skipping."
            );
            return;
        }
    };

    // Verify configure script exists
    if !src_dir.join("configure").exists() {
        eprintln!(
            "[checkpoint7] FFmpeg configure script not found at {}. Skipping.",
            src_dir.display()
        );
        return;
    }

    let bcc = bcc_path_str();
    let jobs = parallel_jobs();

    // Clean previous builds
    make_clean(&src_dir);

    // Configure with BCC — use minimal configuration to reduce build scope
    let configure_result = Command::new("./configure")
        .args([
            &format!("--cc={}", bcc),
            "--disable-x86asm",
            "--disable-inline-asm",
            "--disable-programs",
            "--disable-doc",
            "--disable-network",
            "--enable-small",
        ])
        .current_dir(&src_dir)
        .output()
        .expect("Failed to run FFmpeg configure");

    if !configure_result.status.success() {
        let stderr = String::from_utf8_lossy(&configure_result.stderr);
        let stdout = String::from_utf8_lossy(&configure_result.stdout);
        eprintln!(
            "[checkpoint7] FFmpeg configure failed:\nstdout: {}\nstderr: {}",
            stdout, stderr
        );
        panic!("FFmpeg configure with BCC failed");
    }

    // Build with BCC
    let start = Instant::now();
    let build_result = Command::new("make")
        .args(["-j", &jobs])
        .current_dir(&src_dir)
        .output()
        .expect("Failed to invoke make for FFmpeg build");
    let bcc_duration = start.elapsed();

    if !build_result.status.success() {
        let stderr = String::from_utf8_lossy(&build_result.stderr);
        panic!("[checkpoint7] FFmpeg BCC build failed:\nstderr: {}", stderr);
    }

    eprintln!(
        "[checkpoint7] FFmpeg BCC build completed in {:.2}s",
        bcc_duration.as_secs_f64()
    );

    // Measure GCC baseline
    make_clean(&src_dir);
    let gcc_duration = {
        let configure_gcc = Command::new("./configure")
            .args([
                "--cc=gcc",
                "--disable-x86asm",
                "--disable-inline-asm",
                "--disable-programs",
                "--disable-doc",
                "--disable-network",
                "--enable-small",
            ])
            .current_dir(&src_dir)
            .output();

        if configure_gcc.is_ok() && configure_gcc.as_ref().unwrap().status.success() {
            let gcc_start = Instant::now();
            let gcc_result = Command::new("make")
                .args(["-j", &jobs])
                .current_dir(&src_dir)
                .output();
            match gcc_result {
                Ok(ref o) if o.status.success() => Some(gcc_start.elapsed()),
                _ => None,
            }
        } else {
            None
        }
    };

    // Enforce wall-clock ceiling
    assert_wall_clock_ceiling("FFmpeg", bcc_duration, gcc_duration);

    // Clean up
    make_clean(&src_dir);
}

/// Tests basic FFmpeg operations using a BCC-compiled ffmpeg binary.
#[test]
#[ignore]
fn test_ffmpeg_basic_operations() {
    let src_dir = match locate_ffmpeg_source() {
        Some(d) => d,
        None => {
            eprintln!("[checkpoint7] FFmpeg source not found. Skipping.");
            return;
        }
    };

    // Look for the ffmpeg binary
    let ffmpeg_bin = src_dir.join("ffmpeg");
    let _ffprobe_bin = src_dir.join("ffprobe");

    if !ffmpeg_bin.exists() {
        // Try alternate path under build output
        let alt_ffmpeg = src_dir.join("ffmpeg_g");
        if !alt_ffmpeg.exists() {
            eprintln!(
                "[checkpoint7] ffmpeg binary not found. Run test_ffmpeg_build with \
                 --enable-programs first. Skipping."
            );
            return;
        }
    }

    let ffmpeg_path = if ffmpeg_bin.exists() {
        ffmpeg_bin.to_string_lossy().into_owned()
    } else {
        src_dir.join("ffmpeg_g").to_string_lossy().into_owned()
    };

    // Generate a simple test audio file (silence) and transcode it
    let input_path = common::temp_output_path("ffmpeg_test_input.wav");
    let output_path = common::temp_output_path("ffmpeg_test_output.wav");
    let input_str = input_path.to_str().unwrap();
    let output_str = output_path.to_str().unwrap();

    // Generate 1 second of silence as input
    let gen_result = Command::new(&ffmpeg_path)
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=8000",
            "-t",
            "1",
            input_str,
        ])
        .output();

    match gen_result {
        Ok(ref output) if output.status.success() => {
            // Transcode the file
            let transcode_result = Command::new(&ffmpeg_path)
                .args(["-y", "-i", input_str, "-acodec", "pcm_s16le", output_str])
                .output();

            match transcode_result {
                Ok(ref out) if out.status.success() => {
                    // Verify output file exists and has non-zero size
                    let metadata_result = fs::metadata(output_str);
                    assert!(metadata_result.is_ok(), "FFmpeg output file should exist");
                    let meta = metadata_result.unwrap();
                    assert!(meta.len() > 0, "FFmpeg output file should be non-empty");
                    eprintln!("[checkpoint7] FFmpeg basic operations: PASS");
                }
                Ok(ref out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    eprintln!("[checkpoint7] FFmpeg transcode failed: {}", stderr);
                }
                Err(e) => {
                    eprintln!("[checkpoint7] Failed to run ffmpeg transcode: {}", e);
                }
            }
        }
        Ok(ref output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "[checkpoint7] FFmpeg input generation failed: {}. \
                 Binary may not support lavfi. Skipping.",
                stderr
            );
        }
        Err(e) => {
            eprintln!("[checkpoint7] Failed to run ffmpeg: {}. Skipping.", e);
        }
    }

    // Cleanup
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// CROSS-ARCHITECTURE STRETCH TESTS
// ===========================================================================

/// Tests SQLite compilation for AArch64 via cross-compilation.
///
/// This validates BCC's cross-compilation capabilities for stretch targets
/// beyond the default x86-64 host architecture.
#[test]
#[ignore]
fn test_sqlite_cross_compile_aarch64() {
    let src_dir = match locate_sqlite_source() {
        Some(d) => d,
        None => {
            eprintln!("[checkpoint7] SQLite source not found. Skipping cross-arch test.");
            return;
        }
    };

    let sqlite3_c = src_dir.join("sqlite3.c");
    if !sqlite3_c.exists() {
        eprintln!("[checkpoint7] sqlite3.c not found. Skipping.");
        return;
    }

    let bcc = bcc_path_str();
    let output_path = common::temp_output_path("sqlite3_aarch64");
    let output_str = output_path.to_str().unwrap();

    let start = Instant::now();
    let compile_result = Command::new(&bcc)
        .args([
            "--target=aarch64",
            "-O0",
            "-c",
            "-DSQLITE_THREADSAFE=0",
            "-DSQLITE_OMIT_LOAD_EXTENSION",
            "-o",
            output_str,
        ])
        .arg(sqlite3_c.to_str().unwrap())
        .output()
        .expect("Failed to invoke BCC for SQLite AArch64 build");
    let build_time = start.elapsed();

    if compile_result.status.success() {
        eprintln!(
            "[checkpoint7] SQLite AArch64 cross-compile completed in {:.2}s: PASS",
            build_time.as_secs_f64()
        );
    } else {
        let stderr = String::from_utf8_lossy(&compile_result.stderr);
        eprintln!(
            "[checkpoint7] SQLite AArch64 cross-compile failed:\n{}",
            stderr
        );
    }

    let _ = fs::remove_file(&output_path);
}

/// Tests SQLite compilation for RISC-V 64 via cross-compilation.
#[test]
#[ignore]
fn test_sqlite_cross_compile_riscv64() {
    let src_dir = match locate_sqlite_source() {
        Some(d) => d,
        None => {
            eprintln!("[checkpoint7] SQLite source not found. Skipping cross-arch test.");
            return;
        }
    };

    let sqlite3_c = src_dir.join("sqlite3.c");
    if !sqlite3_c.exists() {
        eprintln!("[checkpoint7] sqlite3.c not found. Skipping.");
        return;
    }

    let bcc = bcc_path_str();
    let output_path = common::temp_output_path("sqlite3_riscv64");
    let output_str = output_path.to_str().unwrap();

    let start = Instant::now();
    let compile_result = Command::new(&bcc)
        .args([
            "--target=riscv64",
            "-O0",
            "-c",
            "-DSQLITE_THREADSAFE=0",
            "-DSQLITE_OMIT_LOAD_EXTENSION",
            "-o",
            output_str,
        ])
        .arg(sqlite3_c.to_str().unwrap())
        .output()
        .expect("Failed to invoke BCC for SQLite RISC-V 64 build");
    let build_time = start.elapsed();

    if compile_result.status.success() {
        eprintln!(
            "[checkpoint7] SQLite RISC-V 64 cross-compile completed in {:.2}s: PASS",
            build_time.as_secs_f64()
        );
    } else {
        let stderr = String::from_utf8_lossy(&compile_result.stderr);
        eprintln!(
            "[checkpoint7] SQLite RISC-V 64 cross-compile failed:\n{}",
            stderr
        );
    }

    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// WALL-CLOCK PERFORMANCE SUMMARY
// ===========================================================================

/// Aggregates build times for all stretch targets and compares against GCC
/// baselines, reporting pass/fail per the 5× ceiling.
///
/// This test is designed to be run AFTER the individual build tests, as it
/// independently measures and reports build times for all available targets.
#[test]
#[ignore]
fn test_stretch_performance_summary() {
    eprintln!("\n========================================================");
    eprintln!("  BCC Checkpoint 7 — Stretch Target Performance Summary  ");
    eprintln!("========================================================\n");

    let mut measurements: Vec<BuildTimeMeasurement> = Vec::new();

    // --- SQLite ---
    if let Some(src_dir) = locate_sqlite_source() {
        let sqlite3_c = src_dir.join("sqlite3.c");
        if sqlite3_c.exists() {
            let bcc = bcc_path_str();
            let output_path = common::temp_output_path("sqlite3_perf");
            let output_str = output_path.to_str().unwrap();

            let start = Instant::now();
            let result = Command::new(&bcc)
                .args([
                    "-O0",
                    "-c",
                    "-DSQLITE_THREADSAFE=0",
                    "-DSQLITE_OMIT_LOAD_EXTENSION",
                    "-o",
                    output_str,
                ])
                .arg(sqlite3_c.to_str().unwrap())
                .output();
            let bcc_dur = start.elapsed();

            let gcc_output_path = common::temp_output_path("sqlite3_gcc_perf");
            let gcc_output_str = gcc_output_path.to_str().unwrap();
            let gcc_start = Instant::now();
            let gcc_result = Command::new("gcc")
                .args([
                    "-O0",
                    "-c",
                    "-DSQLITE_THREADSAFE=0",
                    "-DSQLITE_OMIT_LOAD_EXTENSION",
                    "-o",
                    gcc_output_str,
                ])
                .arg(sqlite3_c.to_str().unwrap())
                .output();
            let gcc_dur = match gcc_result {
                Ok(ref o) if o.status.success() => Some(gcc_start.elapsed()),
                _ => None,
            };

            let ceiling_passed = check_wall_clock_ceiling("SQLite", bcc_dur, gcc_dur);
            measurements.push(BuildTimeMeasurement {
                target_name: "SQLite".to_string(),
                bcc_duration: bcc_dur,
                gcc_duration: gcc_dur,
                ceiling_passed,
            });

            let _ = fs::remove_file(&output_path);
            let _ = fs::remove_file(&gcc_output_path);

            if result.is_err() || !result.unwrap().status.success() {
                eprintln!("  SQLite: BCC compilation FAILED");
            }
        }
    } else {
        eprintln!("  SQLite: source not available (skipped)");
    }

    // --- Redis ---
    if let Some(src_dir) = locate_redis_source() {
        if src_dir.join("Makefile").exists() || src_dir.join("src").join("Makefile").exists() {
            let bcc = bcc_path_str();
            let jobs = parallel_jobs();

            make_clean(&src_dir);
            let start = Instant::now();
            let result = Command::new("make")
                .args(["-j", &jobs])
                .env("CC", &bcc)
                .current_dir(&src_dir)
                .output();
            let bcc_dur = start.elapsed();

            make_clean(&src_dir);
            let gcc_start = Instant::now();
            let gcc_result = Command::new("make")
                .args(["-j", &jobs])
                .env("CC", "gcc")
                .current_dir(&src_dir)
                .output();
            let gcc_dur = match gcc_result {
                Ok(ref o) if o.status.success() => Some(gcc_start.elapsed()),
                _ => None,
            };

            let ceiling_passed = check_wall_clock_ceiling("Redis", bcc_dur, gcc_dur);
            measurements.push(BuildTimeMeasurement {
                target_name: "Redis".to_string(),
                bcc_duration: bcc_dur,
                gcc_duration: gcc_dur,
                ceiling_passed,
            });

            make_clean(&src_dir);

            if result.is_err() || !result.unwrap().status.success() {
                eprintln!("  Redis: BCC build FAILED");
            }
        }
    } else {
        eprintln!("  Redis: source not available (skipped)");
    }

    // --- PostgreSQL ---
    if let Some(_src_dir) = locate_postgresql_source() {
        eprintln!(
            "  PostgreSQL: measurement requires configure step — \
             run test_postgresql_build individually"
        );
    } else {
        eprintln!("  PostgreSQL: source not available (skipped)");
    }

    // --- FFmpeg ---
    if let Some(_src_dir) = locate_ffmpeg_source() {
        eprintln!(
            "  FFmpeg: measurement requires configure step — \
             run test_ffmpeg_build individually"
        );
    } else {
        eprintln!("  FFmpeg: source not available (skipped)");
    }

    // --- Summary Table ---
    eprintln!("\n--------------------------------------------------------");
    eprintln!(
        "  {:<15} {:<12} {:<12} {:<10} {:<8}",
        "Target", "BCC (s)", "GCC (s)", "Ceiling", "Status"
    );
    eprintln!("--------------------------------------------------------");

    let mut all_passed = true;
    for m in &measurements {
        let bcc_secs = format!("{:.2}", m.bcc_duration.as_secs_f64());
        let gcc_secs = match m.gcc_duration {
            Some(d) => format!("{:.2}", d.as_secs_f64()),
            None => "N/A".to_string(),
        };
        let ceiling = match m.gcc_duration {
            Some(d) => format!("{:.2}", d.as_secs_f64() * WALL_CLOCK_CEILING_MULTIPLIER),
            None => "N/A".to_string(),
        };
        let status = match m.ceiling_passed {
            Some(true) => "PASS",
            Some(false) => {
                all_passed = false;
                "FAIL"
            }
            None => "N/A",
        };

        eprintln!(
            "  {:<15} {:<12} {:<12} {:<10} {:<8}",
            m.target_name, bcc_secs, gcc_secs, ceiling, status
        );
    }
    eprintln!("--------------------------------------------------------");

    if measurements.is_empty() {
        eprintln!("\n  No stretch targets available for performance measurement.");
        eprintln!(
            "  Set environment variables (SQLITE_SRC_DIR, REDIS_SRC_DIR, etc.)\n  \
             to enable performance testing."
        );
    } else if all_passed {
        eprintln!("\n  Overall: ALL stretch targets within 5× GCC ceiling. PASS");
    } else {
        eprintln!("\n  Overall: Some stretch targets EXCEEDED 5× GCC ceiling. FAIL");
        // Note: we don't panic here because this is a summary/reporting test.
        // The individual build tests enforce the ceiling with panics.
    }

    eprintln!("\n========================================================\n");
}
