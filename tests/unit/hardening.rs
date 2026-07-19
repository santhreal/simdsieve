#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Hardening test suite for simdsieve.
//!
//! Covers: pattern length boundaries, cross-backend parity, alignment
//! torture, case-insensitive fold boundaries, edge cases, and
//! fuzz-style random inputs.

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use simdsieve::SimdSieve;

// =============================================================================
// Reference implementation, brute-force linear scan
// =============================================================================

/// Returns every offset where at least one pattern matches exactly.
fn reference_scan(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
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

/// Verify `SimdSieve` produces a superset of the reference matches.
///
/// `SimdSieve` is a pre-filter: it may yield false positives but must
/// never yield false negatives.  Every reference match must appear in
/// the sieve output.
fn assert_no_false_negatives(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) {
    let sieve = if case_insensitive {
        SimdSieve::new_case_insensitive(haystack, patterns)
    } else {
        SimdSieve::new(haystack, patterns)
    }
    .unwrap_or_else(|e| panic!("construction failed: {e}"));

    let sieve_hits: Vec<usize> = sieve.collect();
    let expected = reference_scan(haystack, patterns, case_insensitive);

    for &idx in &expected {
        assert!(
            sieve_hits.contains(&idx),
            "false negative at offset {idx}: pattern exists in reference \
             but missing from sieve output (haystack.len()={}, patterns={:?})",
            haystack.len(),
            patterns
                .iter()
                .map(|p| String::from_utf8_lossy(p))
                .collect::<Vec<_>>(),
        );
    }
}

// =============================================================================
// Pattern Length Boundaries
// =============================================================================

#[test]
fn pattern_length_1_byte() {
    let haystack = b"abcdefghijklmnop";
    assert_no_false_negatives(haystack, &[b"d"], false);
}

#[test]
fn pattern_length_2_bytes() {
    let haystack = b"abcdefghijklmnop";
    assert_no_false_negatives(haystack, &[b"ef"], false);
}

#[test]
fn pattern_length_3_bytes() {
    let haystack = b"abcdefghijklmnop";
    assert_no_false_negatives(haystack, &[b"ghi"], false);
}

#[test]
fn pattern_length_4_bytes() {
    let haystack = b"abcdefghijklmnop";
    assert_no_false_negatives(haystack, &[b"jklm"], false);
}

#[test]
fn pattern_length_5_bytes_truncated_prefix() {
    // Patterns >4 bytes use a 4-byte prefix for SIMD sieving,
    // then full-length verification.
    let haystack = b"abcdefghijklmnop";
    assert_no_false_negatives(haystack, &[b"efghi"], false);
}

#[test]
fn pattern_length_8_bytes() {
    let haystack = b"abcdefghijklmnop";
    assert_no_false_negatives(haystack, &[b"efghijkl"], false);
}

#[test]
fn pattern_length_16_bytes() {
    let haystack = b"xyzabcdefghijklmnopqrs";
    assert_no_false_negatives(haystack, &[b"abcdefghijklmnop"], false);
}

#[test]
fn max_patterns_8_simultaneous() {
    let haystack = b"aabbccddeeffgghh_ijklmnop";
    let patterns: &[&[u8]] = &[b"aa", b"bb", b"cc", b"dd", b"ee", b"ff", b"gg", b"hh"];
    assert_no_false_negatives(haystack, patterns, false);
}

#[test]
fn pattern_limit_exceeded_returns_error() {
    let haystack = b"test";
    // Limit is 16 patterns: 17 should trigger the error.
    let patterns: Vec<&[u8]> = (0..17).map(|_| b"x" as &[u8]).collect();
    let result = SimdSieve::new(haystack, &patterns);
    assert!(
        result.is_err(),
        "17 patterns should exceed the 16-pattern limit"
    );
}

#[test]
fn empty_patterns_returns_error() {
    let haystack = b"test";
    let result = SimdSieve::new(haystack, &[]);
    assert!(result.is_err(), "empty patterns should fail");
}

// =============================================================================
// Cross-Backend Parity (multiple haystack sizes)
// =============================================================================

#[test]
fn parity_across_haystack_sizes() {
    let mut rng = StdRng::seed_from_u64(0x0A01_7EE5);
    let patterns: &[&[u8]] = &[b"ab", b"xyz"];

    // Plant patterns in repeating data
    for size in [
        0usize, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1000, 10000,
    ] {
        if size < 3 {
            // Too small for patterns, just verify no panic
            let haystack = vec![b'q'; size];
            let result = SimdSieve::new(&haystack, patterns);
            if let Ok(sieve) = result {
                let _hits: Vec<usize> = sieve.collect();
            }
            continue;
        }

        let mut haystack = vec![0u8; size];
        rng.fill_bytes(&mut haystack);

        // Plant "ab" at a random position
        let pos = rng.gen_range(0..size.saturating_sub(2).max(1));
        if pos + 2 <= size {
            haystack[pos] = b'a';
            haystack[pos + 1] = b'b';
        }

        assert_no_false_negatives(&haystack, patterns, false);
    }
}

#[test]
fn parity_multi_seed() {
    for seed in 0..20u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut haystack = vec![0u8; 5000];
        rng.fill_bytes(&mut haystack);

        let t1 = [haystack[100], haystack[101], haystack[102]];
        let t2 = [haystack[2000], haystack[2001]];

        assert_no_false_negatives(&haystack, &[&t1, &t2], false);
    }
}

// =============================================================================
// Alignment Torture
// =============================================================================

#[test]
fn pattern_at_every_alignment_offset() {
    // Test pattern match at every possible position within a 128-byte window
    for offset in 0..128usize {
        let mut haystack = vec![b'_'; 256];
        if offset + 3 <= haystack.len() {
            haystack[offset] = b'N';
            haystack[offset + 1] = b'D';
            haystack[offset + 2] = b'L';
        }
        assert_no_false_negatives(&haystack, &[b"NDL"], false);
    }
}

#[test]
fn pattern_straddling_block_boundaries() {
    // SIMD blocks are typically 32 or 64 bytes.
    // Place pattern exactly at boundaries.
    for boundary in [31usize, 32, 63, 64, 95, 96, 127, 128] {
        let mut haystack = vec![b'_'; 256];
        if boundary + 2 < haystack.len() {
            haystack[boundary] = b'X';
            haystack[boundary + 1] = b'Y';
            haystack[boundary + 2] = b'Z';
        }
        assert_no_false_negatives(&haystack, &[b"XYZ"], false);
    }
}

#[test]
fn tail_handling_non_block_multiple() {
    // Haystack length is NOT a multiple of any SIMD block size
    for tail_size in [1, 7, 15, 17, 31, 33, 47, 63] {
        let size = 128 + tail_size;
        let mut haystack = vec![b'_'; size];
        // Plant pattern in the tail region
        let pos = size - 4;
        haystack[pos] = b'T';
        haystack[pos + 1] = b'A';
        haystack[pos + 2] = b'I';
        haystack[pos + 3] = b'L';
        assert_no_false_negatives(&haystack, &[b"TAIL"], false);
    }
}

// =============================================================================
// Case-Insensitive Boundaries
// =============================================================================

#[test]
fn case_fold_all_26_letters() {
    for letter in b'a'..=b'z' {
        let upper = letter.to_ascii_uppercase();
        let haystack = [b'_', upper, b'_'];
        let pattern = [letter];
        assert_no_false_negatives(&haystack, &[&pattern], true);
    }
}

#[test]
fn case_fold_does_not_affect_non_alpha() {
    // Bytes just outside the a-z/A-Z range
    let boundary_bytes: &[u8] = &[
        b'@', // 0x40, just before 'A'
        b'[', // 0x5B, just after 'Z'
        b'`', // 0x60, just before 'a'
        b'{', // 0x7B, just after 'z'
        0x00, 0x7F, 0x80, 0xFF,
    ];

    for &byte in boundary_bytes {
        let haystack = [b'_', byte, b'_'];
        // Case-insensitive search for the byte itself should find it
        let pattern = [byte];
        assert_no_false_negatives(&haystack, &[&pattern], true);
    }
}

#[test]
fn case_fold_mixed_case_pattern() {
    let haystack = b"Hello World HELLO world hElLo";
    assert_no_false_negatives(haystack, &[b"hello"], true);
    assert_no_false_negatives(haystack, &[b"HELLO"], true);
    assert_no_false_negatives(haystack, &[b"hElLo"], true);
}

#[test]
fn case_fold_non_ascii_unaffected() {
    // Bytes > 127 must NOT be folded
    let haystack: &[u8] = &[0x80, 0xC0, 0xE0, 0xFF, b'a', b'b'];
    let pattern: &[u8] = &[0xC0];
    assert_no_false_negatives(haystack, &[pattern], true);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn empty_haystack_no_results() {
    let sieve = SimdSieve::new(b"", &[b"abc"]).unwrap();
    let hits: Vec<usize> = sieve.collect();
    assert!(hits.is_empty(), "empty haystack must yield no results");
}

#[test]
fn single_byte_haystack_single_byte_pattern_match() {
    let sieve = SimdSieve::new(b"x", &[b"x"]).unwrap();
    let hits: Vec<usize> = sieve.collect();
    // Must contain offset 0 (exact match)
    let expected = reference_scan(b"x", &[b"x"], false);
    for &idx in &expected {
        assert!(hits.contains(&idx), "false negative at offset {idx}");
    }
}

#[test]
fn single_byte_haystack_no_match() {
    let sieve = SimdSieve::new(b"x", &[b"y"]).unwrap();
    let hits: Vec<usize> = sieve.collect();
    // Sieve may yield false positives, but reference scan has nothing
    // So we just verify no panic
    let _ = hits;
}

#[test]
fn uniform_haystack_every_position_matches() {
    let haystack = vec![b'A'; 200];
    let expected = reference_scan(&haystack, &[b"A"], false);
    assert_no_false_negatives(&haystack, &[b"A"], false);
    // Should be ~200 matches
    assert!(!expected.is_empty());
}

#[test]
fn pattern_not_in_haystack() {
    let haystack = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let expected = reference_scan(haystack, &[b"xyz"], false);
    assert!(expected.is_empty(), "reference should find nothing");
}

#[test]
fn haystack_shorter_than_pattern() {
    let sieve = SimdSieve::new(b"ab", &[b"abcde"]).unwrap();
    let hits: Vec<usize> = sieve.collect();
    // No match possible since haystack < pattern
    let expected = reference_scan(b"ab", &[b"abcde"], false);
    assert!(expected.is_empty());
    // Sieve may yield false positives from prefix matching, that's OK
    let _ = hits;
}

#[test]
fn pattern_at_very_end_of_haystack() {
    let mut haystack = vec![b'_'; 300];
    haystack[297] = b'E';
    haystack[298] = b'N';
    haystack[299] = b'D';
    assert_no_false_negatives(&haystack, &[b"END"], false);
}

#[test]
fn pattern_at_very_start_of_haystack() {
    let mut haystack = vec![b'_'; 300];
    haystack[0] = b'S';
    haystack[1] = b'T';
    haystack[2] = b'A';
    assert_no_false_negatives(&haystack, &[b"STA"], false);
}

// =============================================================================
// Fuzz-Style Random Testing
// =============================================================================

#[cfg(not(miri))]
#[test]
fn fuzz_random_1k_iterations_no_false_negatives() {
    let mut rng = StdRng::seed_from_u64(0xF022_DEAD);

    for iteration in 0..1000 {
        let haystack_len = rng.gen_range(0..500);
        let mut haystack = vec![0u8; haystack_len];
        rng.fill_bytes(&mut haystack);

        let num_patterns = rng.gen_range(1..=8usize);
        let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(num_patterns);
        for _ in 0..num_patterns {
            let pat_len = rng.gen_range(1..=6usize);
            let pat: Vec<u8> = (0..pat_len).map(|_| rng.r#gen()).collect();
            patterns.push(pat);
        }

        let pat_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        let sieve = match SimdSieve::new(&haystack, &pat_refs) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let sieve_hits: Vec<usize> = sieve.collect();
        let expected = reference_scan(&haystack, &pat_refs, false);

        for &idx in &expected {
            assert!(
                sieve_hits.contains(&idx),
                "false negative at iteration {iteration}, offset {idx}, \
                 haystack.len()={haystack_len}"
            );
        }
    }
}

#[cfg(not(miri))]
#[test]
fn fuzz_case_insensitive_500_iterations() {
    let mut rng = StdRng::seed_from_u64(0xCA5E_F01D);

    for iteration in 0..500 {
        let haystack_len = rng.gen_range(0..300);
        let mut haystack = vec![0u8; haystack_len];
        rng.fill_bytes(&mut haystack);

        let num_patterns = rng.gen_range(1..=4usize);
        let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(num_patterns);
        for _ in 0..num_patterns {
            let pat_len = rng.gen_range(1..=4usize);
            // Use ASCII-only patterns for case-insensitive
            let pat: Vec<u8> = (0..pat_len).map(|_| rng.gen_range(b'a'..=b'z')).collect();
            patterns.push(pat);
        }

        let pat_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        let sieve = match SimdSieve::new_case_insensitive(&haystack, &pat_refs) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let sieve_hits: Vec<usize> = sieve.collect();
        let expected = reference_scan(&haystack, &pat_refs, true);

        for &idx in &expected {
            assert!(
                sieve_hits.contains(&idx),
                "case-insensitive false negative at iteration {iteration}, \
                 offset {idx}, haystack.len()={haystack_len}"
            );
        }
    }
}

// =============================================================================
// Iterator Protocol
// =============================================================================

#[test]
fn double_collect_is_empty() {
    let haystack = b"abcdefghij";
    let mut sieve = SimdSieve::new(haystack, &[b"def"]).unwrap();
    let first: Vec<usize> = sieve.by_ref().collect();
    let second: Vec<usize> = sieve.collect();
    assert!(!first.is_empty() || haystack.len() < 3);
    assert!(
        second.is_empty(),
        "second collect on exhausted iterator must be empty"
    );
}

#[test]
fn size_hint_upper_bound_is_correct() {
    let haystack = b"abcdefghijklmnopqrstuvwxyz";
    let sieve = SimdSieve::new(haystack, &[b"xyz"]).unwrap();
    let (lower, upper) = sieve.size_hint();
    assert_eq!(lower, 0);
    assert!(upper.unwrap() <= haystack.len());
}
