#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Unit tests for simdsieve.

use simdsieve::{MultiSieve, SimdSieve, SimdSieveError};

#[test]
fn test_simdsieve_new_basic() {
    let haystack = b"hello world";
    let sieve = SimdSieve::new(haystack, &[b"hello"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![0]);
}

#[test]
fn test_simdsieve_new_multiple_patterns() {
    let haystack = b"alpha beta gamma";
    let sieve = SimdSieve::new(haystack, &[b"alpha", b"gamma"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![0, 11]);
}

#[test]
fn test_simdsieve_new_empty_patterns_error() {
    let haystack = b"hello";
    let result = SimdSieve::new(haystack, &[]);
    assert!(matches!(result, Err(SimdSieveError::EmptyPatternSet)));
}

#[test]
fn test_simdsieve_new_too_many_patterns_error() {
    let haystack = b"hello";
    let patterns: Vec<&[u8]> = vec![b"a"; 17];
    let result = SimdSieve::new(haystack, &patterns);
    assert!(matches!(
        result,
        Err(SimdSieveError::PatternLimitExceeded(17))
    ));
}

#[test]
fn test_simdsieve_new_case_insensitive_basic() {
    let haystack = b"HeLlO wOrLd";
    let sieve = SimdSieve::new_case_insensitive(haystack, &[b"hello"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![0]);
}

#[test]
fn test_multisieve_new_basic() {
    let haystack = b"a b c d e f g h i j k l m n o p q r s t u v w x y z";
    let patterns: Vec<&[u8]> = vec![
        b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h", b"i", b"j", b"k", b"l", b"m", b"n", b"o",
        b"p", b"q", b"r", b"s", b"t", b"u", b"v", b"w", b"x",
    ];
    let sieve = MultiSieve::new(haystack, &patterns).unwrap();
    let results: Vec<usize> = sieve.candidates().collect();
    assert_eq!(results.len(), 24);
}

#[test]
fn test_simdsieve_iterator_size_hint() {
    let haystack = b"hello world";
    let mut sieve = SimdSieve::new(haystack, &[b"world"]).unwrap();
    let (lower, upper) = sieve.size_hint();
    assert_eq!(lower, 0);
    assert_eq!(upper, Some(11));
    assert_eq!(sieve.next(), Some(6));
}

#[test]
fn test_estimate_match_count_exists() {
    let haystack = vec![b'a'; 256];
    let count = SimdSieve::estimate_match_count(&haystack, &[b"a"], false);
    assert!(count > 0, "Should have counted matches (got {count})");
}
