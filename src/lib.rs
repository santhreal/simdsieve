//! SIMD-accelerated byte pattern pre-filtering.
//!
//! `simdsieve` scans a byte haystack for multiple fixed-string patterns at
//! once and yields verified match offsets through a streaming iterator.
//!
//! # SIMD Prefiltering
//!
//! The engine uses a multi-stage prefiltering approach. First, it extracts
//! candidate offsets by searching for the first 1-4 bytes (the "prefix") of
//! each pattern using SIMD vector instructions. This allows scanning dozens of
//! gigabytes per second because the hardware can compare 32 or 64 bytes
//! simultaneously.
//!
//! Once a prefix hit is found in a SIMD register, the engine performs a
//! lightweight verification of the full pattern at that offset.
//!
//! # Supported Architectures
//!
//! The crate includes specialized backends for different CPU architectures:
//!
//! - **AVX-512 (`x86_64`):** Uses 512-bit ZMM registers for maximum throughput
//!   on modern Intel and AMD processors.
//! - **AVX2 (`x86_64`):** Uses 256-bit YMM registers for broad compatibility
//!   across most `x86_64` hardware.
//! - **NEON (AArch64):** Uses 128-bit vector registers on ARM processors,
//!   optimized for Apple Silicon and Graviton.
//! - **Scalar (Any):** A portable fallback implementation for architectures
//!   without specialized SIMD support.
//!
//! The backend is selected automatically at runtime based on the host CPU's
//! capabilities.
//!
//! # Example
//!
//! ```rust
//! use simdsieve::SimdSieve;
//!
//! let haystack = b"GET /admin HTTP/1.1\r\nHost: example\r\n";
//! let patterns: &[&[u8]] = &[b"GET", b"/admin"];
//!
//! let matches: Vec<usize> = SimdSieve::new(haystack, patterns)
//!     .unwrap()
//!     .collect();
//!
//! assert_eq!(matches, vec![0, 4]);
//! ```

#![warn(missing_docs, clippy::pedantic)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::unreadable_literal,
        clippy::panic,
        clippy::manual_let_else
    )
)]
#![allow(clippy::must_use_candidate, clippy::module_name_repetitions)]

pub mod error;
pub mod fold;
pub mod multi;
#[allow(unsafe_code)]
mod scalar;
#[allow(unsafe_code)]
mod sieve;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;

pub use error::{Result, SimdSieveError};
pub use multi::MultiSieve;
pub use sieve::SimdSieve;
