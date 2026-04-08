#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Integer overflow probes and extreme boundary tests for simdsieve.

use simdsieve::{MultiSieve, SimdSieve};

#[test]
fn test_overflow_probe_u32_truncation_haystack() {
    // If the engine uses u32 internally for offsets and truncates,
    // a match past u32::MAX would return the wrong index.
    // We cannot reasonably allocate 4GB in CI, but we can simulate a very large offset
    // conceptually if the engine had a bug, by verifying it correctly uses usize.
    // We will test close to the limits we CAN test.
    let mut haystack = vec![b'A'; 1024 * 1024 * 16]; // 16 MB
    // Set a pattern at the very end
    let end_idx = haystack.len() - 4;
    haystack[end_idx..].copy_from_slice(b"ZZZZ");

    let sieve = SimdSieve::new(&haystack, &[b"ZZZZ"]).unwrap();
    let matches: Vec<usize> = sieve.collect();
    assert_eq!(
        matches,
        vec![end_idx],
        "Must correctly identify match at index {end_idx}"
    );
}

#[test]
fn test_overflow_probe_pattern_count_limits() {
    let haystack = b"hello";
    let pattern = b"h";

    // Pattern count at exactly 16 (max for single SimdSieve)
    let patterns_16 = vec![pattern.as_ref(); 16];
    assert!(
        SimdSieve::new(haystack, &patterns_16).is_ok(),
        "SimdSieve should handle up to 16 patterns"
    );

    // Pattern count at exactly 17 (should fail)
    let patterns_17 = vec![pattern.as_ref(); 17];
    assert!(
        SimdSieve::new(haystack, &patterns_17).is_err(),
        "SimdSieve should error on >16 patterns"
    );

    // MultiSieve should handle 17+
    assert!(
        MultiSieve::new(haystack, &patterns_17).is_ok(),
        "MultiSieve should handle >16 patterns"
    );

    // What about 256 patterns? (8 bit boundary)
    let patterns_256 = vec![pattern.as_ref(); 256];
    assert!(
        MultiSieve::new(haystack, &patterns_256).is_ok(),
        "MultiSieve should handle 256 patterns"
    );

    // What about 65536 patterns? (16 bit boundary)
    let patterns_65536 = vec![pattern.as_ref(); 65536];
    assert!(
        MultiSieve::new(haystack, &patterns_65536).is_ok(),
        "MultiSieve should handle 65536 patterns"
    );
}

#[test]
fn test_overflow_probe_match_count() {
    // We create a haystack of just all 'a's and pattern 'a'.
    // We check how the engine handles overlapping/consecutive matches in buffers.
    // 65536 matches (tests 16-bit bounds on any internal match buffer).
    let haystack = vec![b'a'; 65536];
    let sieve = SimdSieve::new(&haystack, &[b"a"]).unwrap();
    let count = sieve.count();
    assert_eq!(
        count, 65536,
        "Must handle exactly 65536 matches without overflowing any internal buffers"
    );
}

#[test]
fn test_adversarial_alternating_patterns() {
    // Tests input designed to maximize hashing collisions or branch mispredictions
    let mut haystack = Vec::with_capacity(1024 * 1024);
    for i in 0..1024 * 1024 {
        haystack.push(if i % 2 == 0 { 0x00 } else { 0xFF });
    }

    let pattern = &[0x00, 0xFF, 0x00, 0xFF];
    let sieve = SimdSieve::new(&haystack, &[pattern]).unwrap();
    let count = sieve.count();
    // length is 1048576, matches start at every even index, up to len - 4
    // 1048576 - 4 = 1048572
    // Even indices from 0 to 1048572 inclusive -> 1048572 / 2 + 1 = 524287
    assert_eq!(
        count, 524287,
        "Must correctly handle alternating max-entropy boundaries"
    );
}

#[test]
fn test_adversarial_hash_collision_simulation() {
    // Patterns that share the same first 3 bytes, testing prefix collisions.
    let patterns: &[&[u8]] = &[
        b"AAA1", b"AAA2", b"AAA3", b"AAA4", b"AAA5", b"AAA6", b"AAA7", b"AAA8", b"AAA9", b"AAAA",
        b"AAAB", b"AAAC",
    ];
    let haystack = b"AAA1 AAA2 AAAX AAAB";
    let matches: Vec<usize> = MultiSieve::new(haystack, patterns)
        .unwrap()
        .candidates()
        .collect();
    // AAA1 at 0
    // AAA2 at 5
    // AAAX at 10 (not a full match, but might be yielded as candidate by MultiSieve since candidates() yields prefix matches! Wait, candidates() yields prefix matches or full matches?
    // Let's verify candidates() returns only verified matches or prefix?
    // Wait, MultiSieve::candidates() yields candidates? No, looking at its code and documentation, "If multiple groups report the same position, that offset is yielded only once."
    // SimdSieve yields verified matches. MultiSieve just wraps SimdSieve and merges them.
    // So they are verified matches! Wait, MultiSieve calls it `candidates()` but actually it yields verified matches if SimdSieve yields verified matches.
    // Let's just collect and check.
    assert_eq!(
        matches,
        vec![0, 5, 15],
        "Should identify proper prefix collisions dynamically"
    );
}

#[test]
fn test_adversarial_empty_and_single_byte_boundaries() {
    let empty_haystack: &[u8] = b"";
    let single_byte_haystack: &[u8] = b"A";
    let single_byte_pattern: &[u8] = b"A";

    // Empty haystack
    let sieve1 = SimdSieve::new(empty_haystack, &[single_byte_pattern])
        .expect("Engine must accept empty haystack");
    assert_eq!(sieve1.count(), 0, "Empty haystack must yield 0 matches");

    // Single byte haystack, match
    let sieve2 = SimdSieve::new(single_byte_haystack, &[single_byte_pattern])
        .expect("Engine must accept single byte haystack");
    assert_eq!(
        sieve2.count(),
        1,
        "Single byte haystack must yield 1 match if pattern matches"
    );

    // Single byte haystack, no match
    let sieve3 = SimdSieve::new(single_byte_haystack, &[b"B"])
        .expect("Engine must accept single byte haystack");
    assert_eq!(
        sieve3.count(),
        0,
        "Single byte haystack must yield 0 matches if pattern mismatches"
    );
}
