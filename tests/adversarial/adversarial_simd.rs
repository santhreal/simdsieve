#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Adversarial and property tests for simdsieve.
//!
//! Covers edge-case lengths, byte values, pattern limits, case-insensitive
//! folding, large inputs, and proptest properties.

use proptest::prelude::*;
use simdsieve::{MultiSieve, SimdSieve, SimdSieveError};

// =============================================================================
// Reference helpers
// =============================================================================

fn brute_force_matches(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut hits = Vec::new();
    for i in 0..haystack.len() {
        for &pat in patterns {
            if pat.is_empty() {
                hits.push(i);
                break;
            }
            if i + pat.len() > haystack.len() {
                continue;
            }
            let matches = if case_insensitive {
                haystack[i..i + pat.len()]
                    .iter()
                    .zip(pat)
                    .all(|(&a, &b)| a.eq_ignore_ascii_case(&b))
            } else {
                &haystack[i..i + pat.len()] == pat
            };
            if matches {
                hits.push(i);
                break;
            }
        }
    }
    hits.sort_unstable();
    hits.dedup();
    hits
}

// =============================================================================
// Adversarial tests
// =============================================================================

#[test]
fn empty_haystack_no_crash() {
    let sieve = SimdSieve::new(b"", &[b"x"]).unwrap();
    let hits: Vec<usize> = sieve.collect();
    assert!(hits.is_empty());
}

#[test]
fn haystack_shorter_than_64_bytes() {
    let haystack = b"this is a test string that is less than sixty-four bytes";
    let patterns: &[&[u8]] = &[b"test", b"sixty"];
    let hits: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    let expected = brute_force_matches(haystack, patterns, false);
    assert_eq!(hits, expected);
}

#[test]
fn haystack_exactly_64_bytes() {
    let mut haystack = vec![b'x'; 64];
    haystack[0..4].copy_from_slice(b"ABCD");
    haystack[30..34].copy_from_slice(b"EFGH");
    haystack[60..64].copy_from_slice(b"IJKL");
    let patterns: &[&[u8]] = &[b"ABCD", b"EFGH", b"IJKL"];
    let hits: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected = brute_force_matches(&haystack, patterns, false);
    assert_eq!(hits, expected);
}

#[test]
fn haystack_exactly_128_bytes() {
    let mut haystack = vec![b'x'; 128];
    haystack[0..4].copy_from_slice(b"ABCD");
    haystack[62..66].copy_from_slice(b"EFGH");
    haystack[124..128].copy_from_slice(b"IJKL");
    let patterns: &[&[u8]] = &[b"ABCD", b"EFGH", b"IJKL"];
    let hits: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected = brute_force_matches(&haystack, patterns, false);
    assert_eq!(hits, expected);
}

#[test]
fn single_byte_pattern_on_uniform_haystack() {
    let haystack = vec![b'A'; 200];
    let hits: Vec<usize> = SimdSieve::new(&haystack, &[b"A"]).unwrap().collect();
    assert_eq!(hits, (0..200).collect::<Vec<usize>>());
}

#[test]
fn pattern_not_in_haystack() {
    let haystack = vec![b'a'; 100];
    let hits: Vec<usize> = SimdSieve::new(&haystack, &[b"xyz"]).unwrap().collect();
    assert!(hits.is_empty());
}

#[test]
fn eight_patterns_max_all_found() {
    let mut haystack = vec![b'_'; 256];
    let patterns: Vec<Vec<u8>> = (0..8)
        .map(|i| {
            let pat = format!("PAT{i}").into_bytes();
            let start = i * 30;
            haystack[start..start + pat.len()].copy_from_slice(&pat);
            pat
        })
        .collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
    let hits: Vec<usize> = SimdSieve::new(&haystack, &pattern_refs).unwrap().collect();
    let expected = brute_force_matches(&haystack, &pattern_refs, false);
    assert_eq!(hits, expected);
}

#[test]
fn seventeen_patterns_errors() {
    let haystack = b"test";
    // Pattern limit was raised from 8 to 16. 17 patterns should exceed it.
    let pattern_bytes: Vec<Vec<u8>> = (0..17).map(|i| format!("p{i}").into_bytes()).collect();
    let patterns: Vec<&[u8]> = pattern_bytes.iter().map(std::vec::Vec::as_slice).collect();
    let result = SimdSieve::new(haystack, &patterns);
    assert!(matches!(
        result,
        Err(SimdSieveError::PatternLimitExceeded(17))
    ));
}

#[test]
fn sixteen_patterns_succeeds() {
    let haystack = b"p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15";
    let pattern_bytes: Vec<Vec<u8>> = (0..16).map(|i| format!("p{i}").into_bytes()).collect();
    let patterns: Vec<&[u8]> = pattern_bytes.iter().map(std::vec::Vec::as_slice).collect();
    let result = SimdSieve::new(haystack, &patterns);
    assert!(result.is_ok(), "16 patterns should be within the limit");
}

#[test]
fn nine_patterns_multisieve_graceful() {
    let mut haystack = vec![b'_'; 256];
    let patterns: Vec<Vec<u8>> = (0..9)
        .map(|i| {
            let pat = format!("PAT{i}").into_bytes();
            let start = i * 25;
            haystack[start..start + pat.len()].copy_from_slice(&pat);
            pat
        })
        .collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
    let hits: Vec<usize> = MultiSieve::new(&haystack, &pattern_refs)
        .unwrap()
        .candidates()
        .collect();
    let expected = brute_force_matches(&haystack, &pattern_refs, false);
    assert_eq!(hits, expected);
}

#[test]
fn case_insensitive_a_matches_a_everywhere() {
    let haystack = vec![b'a'; 100];
    let hits: Vec<usize> = SimdSieve::new_case_insensitive(&haystack, &[b"A"])
        .unwrap()
        .collect();
    assert_eq!(hits, (0..100).collect::<Vec<usize>>());
}

#[test]
fn pattern_with_null_byte() {
    let mut haystack = vec![b'x'; 100];
    haystack[42] = 0x00;
    haystack[43] = 0x00;
    let hits: Vec<usize> = SimdSieve::new(&haystack, &[b"\x00"]).unwrap().collect();
    assert_eq!(hits, vec![42, 43]);
}

#[test]
fn pattern_with_0xff_byte() {
    let mut haystack = vec![b'x'; 100];
    haystack[77] = 0xFF;
    let hits: Vec<usize> = SimdSieve::new(&haystack, &[b"\xFF"]).unwrap().collect();
    assert_eq!(hits, vec![77]);
}

#[test]
fn all_zeros_haystack_and_pattern() {
    let haystack = vec![0u8; 200];
    let hits: Vec<usize> = SimdSieve::new(&haystack, &[b"\x00"]).unwrap().collect();
    assert_eq!(hits, (0..200).collect::<Vec<usize>>());
}

#[test]
fn all_0xff_no_false_positive() {
    let haystack = vec![0xFF; 200];
    let hits: Vec<usize> = SimdSieve::new(&haystack, &[b"\xFE"]).unwrap().collect();
    assert!(hits.is_empty());
}

#[test]
fn avx2_scalar_parity_adversarial() {
    // Adversarial alignment and byte-value distribution.
    let mut haystack = vec![0u8; 1000];
    for (i, b) in haystack.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    haystack[0..4].copy_from_slice(b"ABCD");
    haystack[31..35].copy_from_slice(b"EFGH");
    haystack[63..67].copy_from_slice(b"IJKL");
    haystack[127..131].copy_from_slice(b"MNOP");
    haystack[999] = b'Z';

    let patterns: &[&[u8]] = &[b"ABCD", b"EFGH", b"IJKL", b"MNOP", b"Z"];
    let hits: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected = brute_force_matches(&haystack, patterns, false);
    assert_eq!(hits, expected);
}

#[test]
fn prefix_filter_parity_across_backends() {
    // Patterns <= 4 bytes => prefix == full pattern.
    // Haystack length chosen so all backends process exactly the same block
    // positions with no tail region affecting estimate_match_count.
    let max_len = 4;
    let len = 128 + max_len - 1; // 131
    let mut haystack = vec![0u8; len];
    for (i, b) in haystack.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    haystack[0..4].copy_from_slice(b"ABCD");
    haystack[63..67].copy_from_slice(b"ABCD");
    haystack[127..131].copy_from_slice(b"ABCD");

    let patterns: &[&[u8]] = &[b"ABCD", b"\x3F\x40"];
    let density = SimdSieve::estimate_match_count(&haystack, patterns, false);

    let mut manual = 0u64;
    for i in 0..128 {
        for &pat in patterns {
            let plen = pat.len().min(4);
            if i + plen <= haystack.len() && haystack[i..i + plen] == pat[..plen] {
                manual += 1;
                break;
            }
        }
    }

    assert_eq!(
        density, manual,
        "SIMD prefix-hit density must match scalar reference count"
    );
}

#[test]
#[cfg(not(miri))]
fn large_haystack_completes_without_panic() {
    let size = 10 * 1024 * 1024; // 10 MB
    let mut haystack = vec![b'x'; size];
    let pos = size - 4;
    haystack[pos..pos + 4].copy_from_slice(b"END!");

    let sieve = SimdSieve::new(&haystack, &[b"END!"]).unwrap();
    let hits: Vec<usize> = sieve.collect();
    assert_eq!(hits, vec![pos]);
}

// =============================================================================
// Property tests (proptest)
// =============================================================================

proptest! {
    /// Property 16: every yielded position starts with the pattern prefix.
    #[test]
    fn every_yielded_position_has_matching_prefix(
        haystack in proptest::collection::vec(0u8..=255, 0..=512),
        pattern in proptest::collection::vec(0u8..=255, 1..=16),
    ) {
        let prefix_len = pattern.len().min(4);
        let sieve = SimdSieve::new(&haystack, &[&pattern]).unwrap();
        for pos in sieve {
            prop_assert!(pos + prefix_len <= haystack.len());
            prop_assert_eq!(
                &haystack[pos..pos + prefix_len],
                &pattern[..prefix_len],
                "position {} does not match prefix",
                pos
            );
        }
    }

    /// Property 17: zero false negatives (candidates are a superset of true matches).
    #[test]
    fn zero_false_negatives(
        haystack in proptest::collection::vec(0u8..=255, 0..=512),
        patterns in proptest::collection::vec(proptest::collection::vec(0u8..=255, 1..=8), 1..=8),
    ) {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let sieve_results: Vec<usize> = SimdSieve::new(&haystack, &pattern_refs).unwrap().collect();
        let expected = brute_force_matches(&haystack, &pattern_refs, false);

        for &pos in &expected {
            prop_assert!(
                sieve_results.contains(&pos),
                "false negative at {} (expected {:?}, got {:?})",
                pos, expected, sieve_results
            );
        }
    }

    /// Property 18: cross-backend parity: SimdSieve output equals scalar reference.
    #[test]
    fn cross_backend_parity(
        haystack in proptest::collection::vec(0u8..=255, 0..=512),
        patterns in proptest::collection::vec(proptest::collection::vec(0u8..=255, 1..=8), 1..=8),
    ) {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let sieve_results: Vec<usize> = SimdSieve::new(&haystack, &pattern_refs).unwrap().collect();
        let expected = brute_force_matches(&haystack, &pattern_refs, false);
        prop_assert_eq!(sieve_results, expected);
    }

    /// Property 19: MultiSieve with N patterns == union of N/8 individual SimdSieve runs.
    #[test]
    fn multisieve_equals_union_of_chunks(
        haystack in proptest::collection::vec(0u8..=255, 0..=512),
        patterns in proptest::collection::vec(proptest::collection::vec(0u8..=255, 1..=8), 1..=24),
    ) {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        let multi_results: Vec<usize> = MultiSieve::new(&haystack, &pattern_refs)
            .unwrap()
            .candidates()
            .collect();

        let mut chunk_results = Vec::new();
        for chunk in pattern_refs.chunks(8) {
            let sieve = SimdSieve::new(&haystack, chunk).unwrap();
            chunk_results.extend(sieve);
        }
        chunk_results.sort_unstable();
        chunk_results.dedup();

        prop_assert_eq!(multi_results, chunk_results);
    }

    /// Property 20: candidates are always in sorted ascending order.
    #[test]
    fn candidates_are_sorted_ascending(
        haystack in proptest::collection::vec(0u8..=255, 0..=512),
        patterns in proptest::collection::vec(proptest::collection::vec(0u8..=255, 1..=8), 1..=8),
    ) {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let results: Vec<usize> = SimdSieve::new(&haystack, &pattern_refs)
            .unwrap()
            .collect();

        for i in 1..results.len() {
            prop_assert!(
                results[i] > results[i - 1],
                "results not in ascending order: {:?}",
                results
            );
        }
    }
}
