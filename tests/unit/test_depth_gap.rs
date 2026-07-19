#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Gap tests for simdsieve.
//! These tests highlight implementation gaps where the engine does not align
//! with its own API contract or expected behaviors.

use simdsieve::{MultiSieve, SimdSieveError};

#[test]
fn test_gap_multisieve_empty_patterns_error() {
    // GAP FINDING: The MultiSieve::new documentation explicitly says:
    // "# Errors
    // Returns an error if the pattern set is empty."
    // However, the implementation does not check for empty patterns; it iterates
    // over chunks(16), which yields 0 chunks for an empty array, and successfully
    // returns a MultiSieve with `sieves: vec![]`.
    let haystack = b"hello";
    let result = MultiSieve::new(haystack, &[]);

    // The following assert will fail because result is actually Ok(_), not Err.
    // This highlights an ENGINE GAP where the code violates the documented contract.
    // The engine MUST be fixed to pass this test.
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPatternSet)),
        "Engine failed to return EmptyPatternSet error for MultiSieve as documented"
    );
}
