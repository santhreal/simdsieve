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

// =============================================================================
// Boundary Tests: Haystack Lengths
// =============================================================================

#[test]
fn boundary_empty_haystack() {
    let sieve = SimdSieve::new(b"", &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 0);
}

#[test]
fn boundary_length_1() {
    let haystack = b"a";
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
}

#[test]
fn boundary_length_31() {
    // 31 bytes: less than one 32-byte AVX2 half-block
    let haystack = &[b'a'; 31];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 31);
}

#[test]
fn boundary_length_32() {
    // Exactly one AVX2 half-block
    let haystack = &[b'a'; 32];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 32);
}

#[test]
fn boundary_length_33() {
    // One byte past AVX2 half-block boundary
    let haystack = &[b'a'; 33];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 33);
}

#[test]
fn boundary_length_63() {
    // One byte short of full AVX2 block
    let haystack = &[b'a'; 63];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 63);
}

#[test]
fn boundary_length_64() {
    // Exactly one AVX2 block / Scalar block
    let haystack = &[b'a'; 64];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 64);
}

#[test]
fn boundary_length_65() {
    // One byte past AVX2 block boundary
    let haystack = &[b'a'; 65];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 65);
}

#[test]
fn boundary_length_127() {
    // One byte short of AVX-512 block
    let haystack = &[b'a'; 127];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 127);
}

#[test]
fn boundary_length_128() {
    // Exactly one AVX-512 block
    let haystack = &[b'a'; 128];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 128);
}

#[test]
fn boundary_length_129() {
    // One byte past AVX-512 block boundary
    let haystack = &[b'a'; 129];
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 129);
}

// =============================================================================
// Position Tests: Match Location
// =============================================================================

#[test]
fn position_match_at_byte_0() {
    let haystack = b"abc def";
    let sieve = SimdSieve::new(haystack, &[b"abc"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
}

#[test]
fn position_match_at_last_byte() {
    let haystack = b"xxx abc";
    let sieve = SimdSieve::new(haystack, &[b"abc"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![4]);
}

#[test]
fn position_spanning_block_boundary_32() {
    // Pattern spans bytes 30-33 (across 32-byte boundary)
    let mut haystack = vec![b'x'; 34];
    haystack[30..34].copy_from_slice(b"abca");
    let sieve = SimdSieve::new(&haystack, &[b"abc"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![30]);
}

#[test]
fn position_spanning_block_boundary_64() {
    // Pattern spans bytes 62-66 (across 64-byte boundary)
    let mut haystack = vec![b'x'; 67];
    haystack[62..66].copy_from_slice(b"abca");
    let sieve = SimdSieve::new(&haystack, &[b"abc"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![62]);
}

#[test]
fn position_spanning_block_boundary_128() {
    // Pattern spans bytes 126-130 (across 128-byte boundary)
    let mut haystack = vec![b'x'; 131];
    haystack[126..130].copy_from_slice(b"abca");
    let sieve = SimdSieve::new(&haystack, &[b"abc"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![126]);
}

// =============================================================================
// Pattern Length Tests
// =============================================================================

#[test]
fn pattern_length_1() {
    let haystack = b"aaaa";
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 4);
}

#[test]
fn pattern_length_2() {
    let haystack = b"ababab";
    let sieve = SimdSieve::new(haystack, &[b"ab"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0, 2, 4]);
}

#[test]
fn pattern_length_3() {
    let haystack = b"abcabcabc";
    let sieve = SimdSieve::new(haystack, &[b"abc"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0, 3, 6]);
}

#[test]
fn pattern_length_4() {
    let haystack = b"abcdabcd";
    let sieve = SimdSieve::new(haystack, &[b"abcd"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0, 4]);
}

#[test]
fn pattern_length_5_plus() {
    // Patterns longer than 4 bytes: only first 4 used for filtering
    let haystack = b"abcdefghij";
    let sieve = SimdSieve::new(haystack, &[b"abcde"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
}

#[test]
fn pattern_longer_than_haystack() {
    let haystack = b"short";
    let sieve = SimdSieve::new(haystack, &[b"this is longer than haystack"]).unwrap();
    assert_eq!(sieve.count(), 0);
}

#[test]
fn pattern_equals_haystack() {
    let haystack = b"exact";
    let sieve = SimdSieve::new(haystack, &[b"exact"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
}

// =============================================================================
// Multi-Pattern Tests
// =============================================================================

#[test]
fn multi_8_patterns_simultaneously() {
    // Create 8 distinct single-byte patterns
    let patterns: Vec<Vec<u8>> = (0..8).map(|i| vec![b'a' + i]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

    let mut haystack = vec![b'x'; 100];
    // Place each pattern at position i*10
    for i in 0..8 {
        haystack[i * 10] = b'a' + i as u8;
    }
    let sieve = SimdSieve::new(&haystack, &pattern_refs).unwrap();
    let matches: Vec<_> = sieve.collect();
    assert_eq!(matches.len(), 8);
}

#[test]
fn multi_8_identical_patterns() {
    // All 8 patterns are the same - should still work
    let patterns: Vec<&[u8]> = vec![b"abc"; 8];
    let haystack = b"xxxabcxxx";
    let sieve = SimdSieve::new(haystack, &patterns).unwrap();
    let matches: Vec<_> = sieve.collect();
    // Should find "abc" at position 3 exactly once
    assert_eq!(matches, vec![3]);
}

#[test]
fn multi_overlapping_patterns() {
    let haystack = b"aaaa";
    let pat1 = b"aa";
    let pat2 = b"aaa";
    let patterns: &[&[u8]] = &[pat1, pat2];
    let sieve = SimdSieve::new(haystack, patterns).unwrap();
    // Should find "aa" at positions 0,1,2 and "aaa" at positions 0,1
    // But iterator should yield unique positions in order
    let matches: Vec<_> = sieve.collect();
    assert_eq!(matches, vec![0, 1, 2]);
}

// =============================================================================
// Density Tests
// =============================================================================

#[test]
fn density_all_positions_match() {
    // Pattern 'a' matches every position in "aaaa"
    let haystack = b"aaaa";
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 4);
}

#[test]
fn density_no_positions_match() {
    let haystack = b"xxxxxxxxx";
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 0);
}

#[test]
fn density_alternating() {
    let haystack = b"abababab";
    let sieve = SimdSieve::new(haystack, &[b"ab"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0, 2, 4, 6]);
}

// =============================================================================
// Case-Insensitive Tests
// =============================================================================

#[test]
fn case_insensitive_ascii_lowercase() {
    // Test lowercase ASCII matching uppercase pattern
    for low in b'a'..=b'z' {
        let upper = low - 0x20;
        let pattern = vec![upper];
        let haystack = vec![low, low, low];
        let sieve = SimdSieve::new_case_insensitive(&haystack, &[&pattern]).unwrap();
        assert_eq!(sieve.count(), 3, "Failed for byte {low}");
    }
}

#[test]
fn case_insensitive_non_ascii_unchanged() {
    // Non-ASCII bytes (0x80-0xFF) should be compared verbatim
    for byte in 0x80u8..=0xFF {
        let haystack = vec![byte, byte];
        let pattern = vec![byte];
        let sieve = SimdSieve::new_case_insensitive(&haystack, &[&pattern]).unwrap();
        assert_eq!(sieve.count(), 2, "Failed for byte {byte:#04x}");
    }
}

#[test]
fn case_insensitive_uppercase_pattern() {
    let haystack = b"Hello World";
    let sieve = SimdSieve::new_case_insensitive(haystack, &[b"HELLO"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
}

#[test]
fn case_insensitive_mixed_case() {
    let haystack = b"HeLLo WoRLd";
    let sieve = SimdSieve::new_case_insensitive(haystack, &[b"hello"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
}

// =============================================================================
// Adversarial Collision Tests
// =============================================================================

#[test]
fn adversarial_prefix_collision_same_4bytes() {
    // Two patterns with identical 4-byte prefix but different 5th byte
    // Only first 4 bytes used for SIMD filtering, so both will be candidates
    let haystack = b"abcdexxxx";
    let pat1 = b"abcdex";
    let pat2 = b"abcdeg";
    let patterns: &[&[u8]] = &[pat1, pat2];
    let sieve = SimdSieve::new(haystack, patterns).unwrap();
    // Only "abcdex" should match at position 0
    let matches: Vec<_> = sieve.collect();
    assert_eq!(matches, vec![0]);
}

#[test]
fn adversarial_prefix_collision_different_5th() {
    // Same 4-byte prefix, different 5th byte - test at boundary
    let haystack = b"xxxabcdeyyy";
    let pat1 = b"abcdex";
    let pat2 = b"abcdeg";
    let patterns: &[&[u8]] = &[pat1, pat2];
    let sieve = SimdSieve::new(haystack, patterns).unwrap();
    let matches: Vec<_> = sieve.collect();
    // Neither matches (haystack has "abcde" not "abcdex")
    assert!(matches.is_empty());
}

#[test]
fn adversarial_repeated_pattern() {
    // Pattern that repeats within itself can cause overlapping matches
    // "abababab" has 8 chars (indices 0-7)
    // "aba" can match at positions 0, 2, 4 (position 6 would need index 8 which is OOB)
    let haystack = b"abababab";
    let sieve = SimdSieve::new(haystack, &[b"aba"]).unwrap();
    assert_eq!(sieve.collect::<Vec<_>>(), vec![0, 2, 4]);
}

#[test]
fn adversarial_prefix_of_another() {
    // One pattern is prefix of another
    let haystack = b"abcd abc";
    let pat1 = b"abc";
    let pat2 = b"abcd";
    let patterns: &[&[u8]] = &[pat1, pat2];
    let sieve = SimdSieve::new(haystack, patterns).unwrap();
    // "abc" at 0, 5; "abcd" at 0
    let matches: Vec<_> = sieve.collect();
    assert_eq!(matches, vec![0, 5]);
}
