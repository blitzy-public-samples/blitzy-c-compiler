//! Common infrastructure layer for the BCC compiler.
//!
//! This module provides foundational utilities imported by every other layer
//! (frontend, ir, passes, backend) in the compilation pipeline. Each submodule
//! is designed to be zero-dependency (no external crates), relying solely on
//! the Rust standard library.
//!
//! # Submodules
//!
//! - [`fx_hash`]: Fast non-cryptographic hashing (FxHash) for symbol tables,
//!   scope lookups, and all performance-critical hash-based data structures.
//!   Provides [`FxHashMap`] and [`FxHashSet`] type aliases that replace the
//!   default SipHash with significantly faster Fibonacci hashing for
//!   compiler-typical small string and integer keys.

pub mod fx_hash;
pub mod target;

// Re-export commonly used types for ergonomic access via `crate::common::FxHashMap`
// instead of requiring the longer `crate::common::fx_hash::FxHashMap` path.
pub use fx_hash::{fx_hash_map, fx_hash_map_with_capacity, fx_hash_set};
pub use fx_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};

// Re-export target architecture types for ergonomic access via `crate::common::Target`
// instead of requiring `crate::common::target::Target`.
pub use target::{DataModel, Endianness, Target};
