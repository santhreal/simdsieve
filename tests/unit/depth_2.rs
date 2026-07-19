#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Adversarial test suite for simdsieve.
//!
//! This module contains 30+ tests covering boundary conditions,
//! edge cases, and adversarial inputs designed to stress the
//! SIMD-accelerated matching engine.

use simdsieve::SimdSieve;

// Scalar Parity Tests
// =============================================================================

fn brute_force_find(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut results = Vec::new();
    for i in 0..haystack.len() {
        for &pat in patterns {
            if i + pat.len() <= haystack.len() {
                let candidate = &haystack[i..i + pat.len()];
                let matches = if case_insensitive {
                    candidate
                        .iter()
                        .zip(pat.iter())
                        .all(|(&c, &p)| c.eq_ignore_ascii_case(&p))
                } else {
                    candidate == pat
                };
                if matches {
                    results.push(i);
                    break; // Don't report same position twice
                }
            }
        }
    }
    results
}

#[test]
fn scalar_parity_random_patterns() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        // Generate random haystack (64-256 bytes)
        let haystack_len = rng.gen_range(64..=256);
        let haystack: Vec<u8> = (0..haystack_len)
            .map(|_| rng.gen_range(0..=255u8))
            .collect();

        // Generate 1-8 random patterns (1-8 bytes each)
        let num_patterns = rng.gen_range(1..=8);
        let patterns: Vec<Vec<u8>> = (0..num_patterns)
            .map(|_| {
                let pat_len = rng.gen_range(1..=8);
                (0..pat_len).map(|_| rng.gen_range(b'a'..=b'z')).collect()
            })
            .collect();
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        // Compare simdsieve with brute force
        let sieve = SimdSieve::new(&haystack, &pattern_refs).unwrap();
        let sieve_results: Vec<_> = sieve.collect();
        let brute_results = brute_force_find(&haystack, &pattern_refs, false);

        assert_eq!(
            sieve_results, brute_results,
            "Mismatch for haystack len={haystack_len}, patterns={patterns:?}"
        );
    }
}

#[test]
fn scalar_parity_case_insensitive() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let haystack_len = rng.gen_range(64..=256);
        // Use only ASCII letters for case-insensitive test
        let haystack: Vec<u8> = (0..haystack_len)
            .map(|_| rng.gen_range(b'A'..=b'z'))
            .collect();

        let num_patterns = rng.gen_range(1..=8);
        let patterns: Vec<Vec<u8>> = (0..num_patterns)
            .map(|_| {
                let pat_len = rng.gen_range(1..=8);
                (0..pat_len).map(|_| rng.gen_range(b'a'..=b'z')).collect()
            })
            .collect();
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        let sieve = SimdSieve::new_case_insensitive(&haystack, &pattern_refs).unwrap();
        let sieve_results: Vec<_> = sieve.collect();
        let brute_results = brute_force_find(&haystack, &pattern_refs, true);

        assert_eq!(
            sieve_results, brute_results,
            "CI mismatch for haystack len={haystack_len}, patterns={patterns:?}"
        );
    }
}

// =============================================================================
// Iterator Behavior Tests
// =============================================================================

#[test]
fn iterator_fused_after_none() {
    let haystack = b"abc";
    let mut sieve = SimdSieve::new(haystack, &[b"x"]).unwrap();

    // First call returns None (no matches)
    assert_eq!(sieve.next(), None);

    // Subsequent calls also return None (fused behavior)
    assert_eq!(sieve.next(), None);
    assert_eq!(sieve.next(), None);
}

#[test]
fn iterator_size_hint() {
    let haystack = b"abcdefghij";
    let sieve = SimdSieve::new(haystack, &[b"x"]).unwrap();

    let (low, high) = sieve.size_hint();
    assert_eq!(low, 0);
    assert!(high.is_some());
}

#[test]
fn iterator_collect_empty() {
    let haystack = b"abc";
    let sieve = SimdSieve::new(haystack, &[b"x"]).unwrap();
    let matches: Vec<_> = sieve.collect();
    assert!(matches.is_empty());
}

// =============================================================================
// Construction Error Tests
// =============================================================================

use simdsieve::SimdSieveError;

#[test]
fn error_empty_pattern_set() {
    let result = SimdSieve::new(b"haystack", &[]);
    assert!(matches!(result, Err(SimdSieveError::EmptyPatternSet)));
}

#[test]
fn error_pattern_limit_exceeded() {
    // Limit is 16 patterns: 17 should trigger the error.
    let patterns: Vec<Vec<u8>> = (0..17).map(|i| vec![b'a' + (i % 26)]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
    let result = SimdSieve::new(b"haystack", &pattern_refs);
    assert!(matches!(
        result,
        Err(SimdSieveError::PatternLimitExceeded(17))
    ));
}

#[test]
fn error_exactly_16_patterns_ok() {
    let patterns: Vec<Vec<u8>> = (0..16).map(|i| vec![b'a' + i]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
    let result = SimdSieve::new(b"haystack", &pattern_refs);
    assert!(result.is_ok());
}

// =============================================================================
// Score Density Tests
// =============================================================================

#[test]
fn estimate_match_count_basic() {
    // estimate_match_count counts SIMD-level prefix hits (not tail-processed bytes)
    // Need a longer haystack to trigger block processing
    let haystack = vec![b'a'; 100]; // 100 'a' characters
    let count = SimdSieve::estimate_match_count(&haystack, &[b"a"], false);
    // Should count hits from block processing (64 bytes per scalar block)
    // Exact count depends on backend but should be > 0
    assert!(count > 0, "estimate_match_count should find prefix hits");
}

#[test]
fn estimate_match_count_no_matches() {
    let haystack = b"bbbb";
    let count = SimdSieve::estimate_match_count(haystack, &[b"a"], false);
    assert_eq!(count, 0);
}

#[test]
fn estimate_match_count_prefix_only() {
    // estimate_match_count counts raw prefix hits (first 1–4 bytes).
    // It does not verify that the full pattern fits in the remaining haystack.
    let haystack = b"abcd";
    let count = SimdSieve::estimate_match_count(haystack, &[b"abce"], false);
    assert_eq!(count, 0);
}

#[test]
fn estimate_match_count_long_pattern_edge() {
    // A prefix can match even when the full pattern does not fit at the end
    // of the haystack. estimate_match_count counts these prefix hits.
    let haystack = b"abcde";
    let count = SimdSieve::estimate_match_count(haystack, &[b"abcdef"], false);
    // Prefix "abcd" matches at position 0, but the full 6-byte pattern doesn't fit.
    assert_eq!(count, 1);
}
