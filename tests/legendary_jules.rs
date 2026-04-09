#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Legendary test suite for simdsieve by Jules.

use proptest::prelude::*;
use simdsieve::{SimdSieve, SimdSieveError};

// =============================================================================
// PARITY: Naive byte-by-byte scan vs SimdSieve
// =============================================================================

fn naive_find(haystack: &[u8], patterns: &[&[u8]]) -> Vec<usize> {
    let mut results = Vec::new();
    for i in 0..haystack.len() {
        for &pat in patterns {
            if i + pat.len() <= haystack.len() && &haystack[i..i + pat.len()] == pat {
                results.push(i);
                break;
            }
        }
    }
    results
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]
    #[test]
    fn parity_proptest(
        haystack in prop::collection::vec(0u8..=255u8, 0..512),
        patterns in prop::collection::vec(prop::collection::vec(0u8..=255u8, 1..16), 1..16)
    ) {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let sieve_results: Vec<usize> = SimdSieve::new(&haystack, &pattern_refs)
            .expect("Valid patterns should not fail construction")
            .collect();
        let naive_results = naive_find(&haystack, &pattern_refs);
        assert_eq!(sieve_results, naive_results, "Mismatch between SimdSieve and naive search");
    }
}

// =============================================================================
// SHORT INPUT: 0, 1, 15, 16, 31, 32 bytes
// =============================================================================

#[test]
fn short_input_lengths() {
    let lengths = [0, 1, 15, 16, 31, 32];
    for &len in &lengths {
        let haystack = vec![b'x'; len];

        // 1. Pattern that doesn't exist
        let sieve = SimdSieve::new(&haystack, &[b"a"]).unwrap();
        assert_eq!(
            sieve.count(),
            0,
            "Should find 0 matches for non-existent pattern in length {len}"
        );

        // 2. Pattern that is the entire haystack (if len > 0)
        if len > 0 {
            let sieve = SimdSieve::new(&haystack, &[&haystack]).unwrap();
            assert_eq!(
                sieve.collect::<Vec<_>>(),
                vec![0],
                "Should find exactly 1 match for full haystack in length {len}"
            );
        }
    }
}

// =============================================================================
// ALIGNMENT: Patterns at aligned and unaligned positions
// =============================================================================

#[test]
fn alignment_positions() {
    let mut haystack = vec![b'A'; 256];
    let pattern = b"B";

    // Test all offsets from 0 to 128
    for offset in 0..=128 {
        haystack[offset] = b'B';
        let sieve = SimdSieve::new(&haystack, &[pattern]).unwrap();
        let results: Vec<usize> = sieve.collect();
        assert_eq!(
            results,
            vec![offset],
            "Failed to find pattern at offset {offset}"
        );
        haystack[offset] = b'A'; // Reset
    }
}

// =============================================================================
// EMPTY PATTERN: Rejected with error, not panic
// =============================================================================

#[test]
fn empty_pattern_rejected() {
    // Single empty pattern
    let result = SimdSieve::new(b"haystack", &[b""]);
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPatternSet)),
        "Should return EmptyPatternSet"
    );

    // Empty pattern slice
    let result2 = SimdSieve::new(b"haystack", &[]);
    assert!(
        matches!(result2, Err(SimdSieveError::EmptyPatternSet)),
        "Should return EmptyPatternSet"
    );
}

// =============================================================================
// SINGLE BYTE PATTERN: Finds every occurrence
// =============================================================================

#[test]
fn single_byte_pattern_every_occurrence() {
    let haystack = vec![b'X'; 100];
    let sieve = SimdSieve::new(&haystack, &[b"X"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    let expected: Vec<usize> = (0..100).collect();
    assert_eq!(
        results, expected,
        "Failed to find all occurrences of single byte"
    );
}

// =============================================================================
// 8 PATTERNS: All found in crafted input
// =============================================================================

#[test]
fn eight_patterns_found() {
    let patterns: [&[u8]; 8] = [
        b"PAT1", b"PAT2", b"PAT3", b"PAT4", b"PAT5", b"PAT6", b"PAT7", b"PAT8",
    ];

    // Create a haystack with all 8 patterns interspersed
    // Haystack: PAT1...PAT2...PAT3...PAT4...PAT5...PAT6...PAT7...PAT8
    let mut haystack = vec![b'.'; 100];
    for (i, &pat) in patterns.iter().enumerate() {
        let offset = i * 10;
        haystack[offset..offset + pat.len()].copy_from_slice(pat);
    }

    let sieve = SimdSieve::new(&haystack, &patterns).unwrap();
    let results: Vec<usize> = sieve.collect();

    let expected: Vec<usize> = (0..8).map(|i| i * 10).collect();
    assert_eq!(results, expected, "Failed to find all 8 patterns");
}

// Also test max capacity (16 patterns)
#[test]
fn max_capacity_16_patterns_found() {
    let patterns: Vec<Vec<u8>> = (0..16).map(|i| vec![b'A' + i as u8]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

    let mut haystack = vec![b'.'; 200];
    for i in 0..16 {
        let offset = i * 10;
        haystack[offset] = b'A' + i as u8;
    }

    let sieve = SimdSieve::new(&haystack, &pattern_refs).unwrap();
    let results: Vec<usize> = sieve.collect();

    let expected: Vec<usize> = (0..16).map(|i| i * 10).collect();
    assert_eq!(results, expected, "Failed to find all 16 patterns");
}

// =============================================================================
// SCORE DENSITY: estimate_match_count on random data
// =============================================================================

#[test]
fn score_density_random_data() {
    // Generate uniform random data
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let haystack: Vec<u8> = (0..4096).map(|_| rand::Rng::r#gen(&mut rng)).collect();

    // Pattern "A" should have probability 1/256 of matching
    // Over 4096 bytes, expected match count is around 16.
    let count = SimdSieve::estimate_match_count(&haystack, &[b"A"], false).unwrap();

    // It should be reasonably close to 16, let's say between 5 and 35
    assert!(
        (5..=35).contains(&count),
        "Estimated match count {count} is outside expected variance for 1-byte pattern"
    );

    // Pattern "AB" should have probability 1/65536, expected matches: ~0
    let count2 = SimdSieve::estimate_match_count(&haystack, &[b"AB"], false).unwrap();
    assert!(
        count2 <= 5,
        "Estimated match count {count2} is too high for 2-byte pattern"
    );
}
