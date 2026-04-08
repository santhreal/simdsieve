//! Error types for `simdsieve` construction failures.
//!
//! This module defines the error enum used when `SimdSieve` construction
//! fails due to invalid input parameters. All errors occur at construction
//! time; the streaming iterator itself is infallible.

use core::fmt;

/// The primary result type for `SimdSieve` construction.
///
/// This is a type alias for `Result<T, SimdSieveError>`, used as the
/// return type for all fallible construction functions.
pub type Result<T> = core::result::Result<T, SimdSieveError>;

/// Errors that can occur during `SimdSieve` construction.
///
/// These errors indicate invalid input parameters. Once a `SimdSieve` is
/// successfully constructed, iteration is infallible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimdSieveError {
    /// No patterns were provided.
    ///
    /// At least one pattern is required to construct a sieve. An empty
    /// pattern set would result in an iterator that always yields `None`,
    /// which is almost certainly a programming error.
    EmptyPatternSet,
    /// More than 16 patterns were provided.
    ///
    /// The unrolled register layout supports a maximum of 16 simultaneous
    /// patterns. Each additional pattern adds a broadcast + compare + OR
    /// reduction per byte position, and 16 patterns saturate the available
    /// execution ports on most micro-architectures.
    ///
    /// The wrapped value is the number of patterns that were passed.
    PatternLimitExceeded(usize),
}

impl fmt::Display for SimdSieveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPatternSet => {
                write!(f, "sieve construction failed: empty pattern set provided")
            }
            Self::PatternLimitExceeded(c) => write!(
                f,
                "sieve construction failed: provided {c} patterns, but hardware router only supports maximum 16"
            ),
        }
    }
}

impl std::error::Error for SimdSieveError {}
