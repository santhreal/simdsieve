#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_long_pattern_fingerprinting() {
    let haystack = b"xxxContent-Security-Policy: default-src 'none';xxx".to_vec();
    let sieve = SimdSieve::new(
        &haystack,
        &[b"Content-Security-Policy: default-src 'none';"],
    )
    .unwrap();

    let results: Vec<usize> = sieve.collect();

    // Because simdsieve truncates to a 4-byte leading fingerprint natively (e.g. "Cont"),
    // it yields the start offset seamlessly. The upstream DFA catches the hit, evaluates the
    // full string starting at index 3, and passes. False negatives are mathematically impossible.
    assert_eq!(results, vec![3]);
}

#[test]
fn test_density_scoring_algorithm() {
    // Generate a payload filled densely with exact overlapping tokens
    let haystack = b"AAAAAAAABBBBAAAACAA".repeat(10);

    // In typical iteration, returning indices requires 1 branch per match.
    // In Density Scoring, the pure `popcnt` of the folded mask resolves instantly.

    let score = SimdSieve::estimate_match_count(&haystack, &[b"AA", b"BBB"], false);

    // Score reflects total logical independent boolean hits aggregated globally over the blocks.
    assert!(
        score > 0,
        "Threat density accumulator failed to capture mass hits"
    );
}
