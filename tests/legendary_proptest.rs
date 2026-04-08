#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Property-based tests for simdsieve using proptest.
//!
//! These tests verify fundamental properties that should always hold
//! regardless of input values.

use proptest::prelude::*;
use simdsieve::SimdSieve;

// =============================================================================
// Property: No False Positives
// =============================================================================

proptest! {
    /// Every yielded position must actually contain the pattern.
    /// This is the fundamental correctness property.
    #[test]
    fn no_false_positives(
        haystack in proptest::collection::vec(0u8..=255, 0..=1024),
        pattern in proptest::collection::vec(0u8..=255, 1..=8),
    ) {
        prop_assume!(!pattern.is_empty());

        let pat_slice: &[u8] = &pattern;
        let sieve = SimdSieve::new(&haystack, &[pat_slice]).unwrap();

        for pos in sieve {
            prop_assert!(
                pos + pattern.len() <= haystack.len(),
                "Position {} exceeds haystack bounds (len={})",
                pos, haystack.len()
            );
            prop_assert_eq!(
                &haystack[pos..pos + pattern.len()],
                &pattern[..],
                "False positive at position {}",
                pos
            );
        }
    }
}

// =============================================================================
// Property: No False Negatives (for short inputs)
// =============================================================================

/// Brute force find for property testing
fn brute_force_find(haystack: &[u8], pattern: &[u8]) -> Vec<usize> {
    let mut results = Vec::new();
    for i in 0..haystack.len().saturating_sub(pattern.len() - 1) {
        if &haystack[i..i + pattern.len()] == pattern {
            results.push(i);
        }
    }
    results
}

proptest! {
    /// For short inputs, simdsieve finds everything a naive scan would.
    /// We limit haystack size to keep brute force comparison fast.
    #[test]
    fn no_false_negatives_short(
        haystack in proptest::collection::vec(0u8..=255, 0..=256),
        pattern in proptest::collection::vec(0u8..=255, 1..=8),
    ) {
        prop_assume!(!pattern.is_empty());

        let pat_slice: &[u8] = &pattern;
        let sieve = SimdSieve::new(&haystack, &[pat_slice]).unwrap();
        let sieve_results: Vec<_> = sieve.collect();
        let brute_results = brute_force_find(&haystack, &pattern);

        prop_assert!(
            sieve_results == brute_results,
            "Mismatch: sieve found {:?}, brute found {:?}",
            sieve_results, brute_results
        );
    }
}

// =============================================================================
// Property: Case-Insensitive Parity
// =============================================================================

/// Brute force case-insensitive find
fn brute_force_find_ci(haystack: &[u8], pattern: &[u8]) -> Vec<usize> {
    let mut results = Vec::new();
    for i in 0..haystack.len().saturating_sub(pattern.len() - 1) {
        let candidate = &haystack[i..i + pattern.len()];
        let matches = candidate
            .iter()
            .zip(pattern.iter())
            .all(|(&c, &p)| c.eq_ignore_ascii_case(&p));
        if matches {
            results.push(i);
        }
    }
    results
}

proptest! {
    /// Case-insensitive search on haystack = exact search on lowercased haystack
    /// (for ASCII inputs)
    #[test]
    fn case_insensitive_parity(
        haystack_bytes in proptest::collection::vec(0u8..=127u8, 0..=256),
        pattern_bytes in proptest::collection::vec(b'a'..=b'z', 1..=8),
    ) {
        // Lowercase the haystack for ASCII-only test
        let haystack: Vec<u8> = haystack_bytes
            .iter()
            .map(|&b| b.to_ascii_lowercase())
            .collect();
        let pattern: Vec<u8> = pattern_bytes.iter().map(|&b| b.to_ascii_lowercase()).collect();

        let pat_slice: &[u8] = &pattern;

        // CI search should find matches
        let ci_sieve = SimdSieve::new_case_insensitive(&haystack, &[pat_slice]).unwrap();
        let ci_results: Vec<_> = ci_sieve.collect();

        // Exact search on same haystack should find same matches
        let exact_sieve = SimdSieve::new(&haystack, &[pat_slice]).unwrap();
        let exact_results: Vec<_> = exact_sieve.collect();

        prop_assert_eq!(
            ci_results, exact_results,
            "CI and exact should match when haystack is already lowercase"
        );
    }
}

proptest! {
    /// CI search finds same positions as brute force CI search
    #[test]
    fn case_insensitive_vs_brute(
        haystack in proptest::collection::vec(0u8..=127u8, 0..=256),
        pattern in proptest::collection::vec(0u8..=127u8, 1..=8),
    ) {
        prop_assume!(!pattern.is_empty());

        let pat_slice: &[u8] = &pattern;
        let sieve = SimdSieve::new_case_insensitive(&haystack, &[pat_slice]).unwrap();
        let sieve_results: Vec<_> = sieve.collect();
        let brute_results = brute_force_find_ci(&haystack, &pattern);

        prop_assert!(
            sieve_results == brute_results,
            "CI mismatch: sieve found {:?}, brute found {:?}",
            sieve_results, brute_results
        );
    }
}

// =============================================================================
// Property: Fused Iterator
// =============================================================================

proptest! {
    /// After returning None, iterator always returns None
    #[test]
    fn fused_iterator(
        haystack in proptest::collection::vec(0u8..=255, 0..=256),
        pattern in proptest::collection::vec(0u8..=255, 1..=8),
    ) {
        prop_assume!(!pattern.is_empty());

        let pat_slice: &[u8] = &pattern;
        let mut sieve = SimdSieve::new(&haystack, &[pat_slice]).unwrap();

        // Exhaust the iterator
        while sieve.next().is_some() {}

        // After exhaustion, should always return None
        prop_assert!(sieve.next().is_none());
        prop_assert!(sieve.next().is_none());
        prop_assert!(sieve.next().is_none());
    }
}

// =============================================================================
// Property: Multiple Patterns
// =============================================================================

proptest! {
    /// Results from multi-pattern search should be subset of union of
    /// single-pattern searches (may have duplicates removed)
    #[test]
    fn multi_pattern_consistency(
        haystack in proptest::collection::vec(b'a'..=b'z', 0..=256),
        pat1 in proptest::collection::vec(b'a'..=b'z', 1..=4),
        pat2 in proptest::collection::vec(b'a'..=b'z', 1..=4),
    ) {
        prop_assume!(!pat1.is_empty() && !pat2.is_empty());

        let pat1_slice: &[u8] = &pat1;
        let pat2_slice: &[u8] = &pat2;

        // Single pattern results
        let sieve1 = SimdSieve::new(&haystack, &[pat1_slice]).unwrap();
        let results1: std::collections::HashSet<_> = sieve1.collect();

        let sieve2 = SimdSieve::new(&haystack, &[pat2_slice]).unwrap();
        let results2: std::collections::HashSet<_> = sieve2.collect();

        // Multi-pattern results
        let multi_sieve = SimdSieve::new(&haystack, &[pat1_slice, pat2_slice]).unwrap();
        let multi_results: std::collections::HashSet<_> = multi_sieve.collect();

        // Multi results should equal union of singles
        let expected: std::collections::HashSet<_> =
            results1.union(&results2).copied().collect();

        prop_assert_eq!(
            multi_results, expected,
            "Multi-pattern results don't match union of singles"
        );
    }
}

// =============================================================================
// Property: Order Preservation
// =============================================================================

proptest! {
    /// Results should always be yielded in ascending order
    #[test]
    fn results_in_ascending_order(
        haystack in proptest::collection::vec(b'a'..=b'z', 0..=512),
        patterns in proptest::collection::vec(proptest::collection::vec(b'a'..=b'z', 1..=4), 1..=8),
    ) {
        prop_assume!(!patterns.is_empty());

        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let sieve = SimdSieve::new(&haystack, &pattern_refs).unwrap();
        let results: Vec<_> = sieve.collect();

        for i in 1..results.len() {
            prop_assert!(
                results[i] > results[i - 1],
                "Results not in ascending order: {:?}",
                results
            );
        }
    }
}

// =============================================================================
// Property: Determinism
// =============================================================================

proptest! {
    /// Same inputs should always produce same outputs
    #[test]
    fn deterministic_results(
        haystack in proptest::collection::vec(0u8..=255, 0..=256),
        patterns in proptest::collection::vec(proptest::collection::vec(0u8..=255, 1..=4), 1..=4),
    ) {
        prop_assume!(!patterns.is_empty());

        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        let sieve1 = SimdSieve::new(&haystack, &pattern_refs).unwrap();
        let results1: Vec<_> = sieve1.collect();

        let sieve2 = SimdSieve::new(&haystack, &pattern_refs).unwrap();
        let results2: Vec<_> = sieve2.collect();

        prop_assert_eq!(results1, results2, "Non-deterministic results");
    }
}

// =============================================================================
// Property: Empty Pattern Set Error
// =============================================================================

proptest! {
    /// Empty pattern set should always error
    #[test]
    fn empty_pattern_set_errors(haystack in proptest::collection::vec(0u8..=255, 0..=256)) {
        let result = SimdSieve::new(&haystack, &[]);
        prop_assert!(result.is_err());
    }
}

// =============================================================================
// Property: Too Many Patterns Error
// =============================================================================

proptest! {
    /// More than 16 patterns should always error
    #[test]
    fn too_many_patterns_errors(
        haystack in proptest::collection::vec(0u8..=255, 0..=256),
        patterns in proptest::collection::vec(proptest::collection::vec(0u8..=255, 1..=4), 17..=24),
    ) {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let result = SimdSieve::new(&haystack, &pattern_refs);
        prop_assert!(result.is_err());
    }
}
