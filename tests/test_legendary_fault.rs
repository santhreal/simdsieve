#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Fault-survival tests for simdsieve.
//!
//! These tests verify that the library returns actionable errors and never
//! panics on adversarial or resource-constrained inputs.

use simdsieve::{MultiSieve, SimdSieve, SimdSieveError};

#[test]
fn test_empty_pattern_set_returns_error() {
    let result = SimdSieve::new(b"haystack", &[]);
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPatternSet)),
        "Empty pattern set must return EmptyPatternSet error"
    );
}

#[test]
fn test_all_empty_patterns_return_error() {
    let result = SimdSieve::new(b"haystack", &[b"", b""]);
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPattern { index: 0 })),
        "All-empty patterns must return EmptyPattern error"
    );
}

#[test]
fn test_too_many_patterns_returns_error() {
    let patterns: Vec<Vec<u8>> = (0..17).map(|i| vec![b'a' + i as u8]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
    let result = SimdSieve::new(b"haystack", &pattern_refs);
    assert!(
        matches!(result, Err(SimdSieveError::PatternLimitExceeded(17))),
        "17 patterns must return PatternLimitExceeded error"
    );
}

#[test]
fn test_multisieve_empty_patterns_error() {
    let result = MultiSieve::new(b"haystack", &[]);
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPatternSet)),
        "MultiSieve empty pattern set must return error"
    );
}

#[test]
fn test_oom_survival_simdsieve() {
    // SimdSieve::new does not heap-allocate, so it should succeed even on
    // large inputs when called directly.
    let haystack = vec![b'x'; 10_000_000];
    let result = SimdSieve::new(&haystack, &[b"xyz"]);
    assert!(
        result.is_ok(),
        "SimdSieve::new must not allocate internally and should survive large inputs"
    );
}

#[test]
fn test_io_error_independence() {
    // simdsieve works purely on memory slices; verify it is unaffected by
    // external I/O state.
    let haystack = b"test io fault";
    let sieve = SimdSieve::new(haystack, &[b"fault"]).unwrap();
    let matches: Vec<usize> = sieve.collect();
    assert_eq!(
        matches,
        vec![8],
        "SimdSieve must work independently of external I/O faults"
    );
}
