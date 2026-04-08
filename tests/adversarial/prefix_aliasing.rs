#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_adversarial_prefix_aliasing_and_overlap() {
    // This is the classic Aho-Corasick nightmare state.
    // If the patterns themselves overlap and alias each other, the math must not double-yield
    // the same offset, and must find all exact starting boundaries across different length targets.

    // We search for: "A", "AA", "AAA", "AAAA"
    // Inside pure "AAAAAAAAAA"
    let haystack = b"AAAAAAAAAA".to_vec();
    let patterns: Vec<&[u8]> = vec![b"A", b"AA", b"AAA", b"AAAA"];

    let sieve = SimdSieve::new(&haystack, &patterns).unwrap();
    let mut results: Vec<usize> = sieve.collect();

    // `simdsieve` aims to find if ANY pattern starts at an index.
    // Because "A", "AA", "AAA", "AAAA" all start at index 0..10,
    // the underlying OR mask should yield exactly 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // exactly ONCE each, despite 4 different patterns triggering simultaneously.
    results.sort_unstable();

    let expected: Vec<usize> = (0..10).collect();
    assert_eq!(
        results, expected,
        "Prefix aliasing failed to merge simultaneous identical-offset matches without duplicates."
    );
}

#[test]
fn test_adversarial_case_insensitive_aliasing() {
    // A mixture of overlapping cases
    let haystack = b"xAaAAx".to_vec();
    let patterns: Vec<&[u8]> = vec![b"Aa", b"aA"];

    let sieve = SimdSieve::new_case_insensitive(&haystack, &patterns).unwrap();
    let mut results: Vec<usize> = sieve.collect();
    results.sort_unstable();

    // Indices:
    // 0: x
    // 1: A
    // 2: a
    // 3: A
    // 4: A
    // 5: x
    // "AA" case insensitive matches expected at 1, 2, 3.
    assert_eq!(results, vec![1, 2, 3]);
}
