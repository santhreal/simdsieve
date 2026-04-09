#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else,
    clippy::similar_names,
    clippy::needless_pass_by_value
)]

use simdsieve::SimdSieve;

fn reference_scan(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut hits = Vec::new();
    for i in 0..haystack.len().saturating_add(1) {
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

fn assert_matches_reference(
    haystack: &[u8],
    patterns: &[&[u8]],
    case_insensitive: bool,
    test_name: &str,
) {
    let sieve = if case_insensitive {
        SimdSieve::new_case_insensitive(haystack, patterns)
    } else {
        SimdSieve::new(haystack, patterns)
    }
    .unwrap_or_else(|e| panic!("[{test_name}] construction failed: {e:?}"));

    let sieve_hits: Vec<usize> = sieve.collect();
    let expected = reference_scan(haystack, patterns, case_insensitive);

    assert_eq!(
        sieve_hits,
        expected,
        "[{test_name}] mismatch: haystack_len={}, patterns={:?}, sieve={:?}, expected={:?}",
        haystack.len(),
        patterns
            .iter()
            .map(|p| String::from_utf8_lossy(p))
            .collect::<Vec<_>>(),
        sieve_hits,
        expected
    );
}

#[test]
fn test_01_empty_input_empty_pattern() {
    let result = SimdSieve::new(b"", &[b""]);
    assert!(result.is_err());
}

#[test]
fn test_02_empty_input_single_byte() {
    assert_matches_reference(b"", &[b"a"], false, "test_02_empty_input_single_byte");
}

#[test]
fn test_03_input_exactly_64_bytes_no_match() {
    let haystack = vec![b'a'; 64];
    assert_matches_reference(
        &haystack,
        &[b"xyz"],
        false,
        "test_03_input_exactly_64_bytes_no_match",
    );
}

#[test]
fn test_04_input_exactly_64_bytes_match_at_start() {
    let mut haystack = vec![b'a'; 64];
    haystack[0] = b'b';
    assert_matches_reference(
        &haystack,
        &[b"b"],
        false,
        "test_04_input_exactly_64_bytes_match_at_start",
    );
}

#[test]
fn test_05_input_exactly_64_bytes_match_at_end() {
    let mut haystack = vec![b'a'; 64];
    haystack[63] = b'b';
    assert_matches_reference(
        &haystack,
        &[b"b"],
        false,
        "test_05_input_exactly_64_bytes_match_at_end",
    );
}

#[test]
fn test_06_input_exactly_63_bytes_match_at_end() {
    let mut haystack = vec![b'a'; 63];
    haystack[62] = b'b';
    assert_matches_reference(
        &haystack,
        &[b"b"],
        false,
        "test_06_input_exactly_63_bytes_match_at_end",
    );
}

#[test]
fn test_07_input_exactly_63_bytes_match_spanning_end() {
    let mut haystack = vec![b'a'; 63];
    haystack[61] = b'b';
    haystack[62] = b'c';
    assert_matches_reference(
        &haystack,
        &[b"bc"],
        false,
        "test_07_input_exactly_63_bytes_match_spanning_end",
    );
}

#[test]
fn test_08_null_byte_in_haystack() {
    let mut haystack = vec![b'a'; 100];
    haystack[50] = 0x00;
    assert_matches_reference(
        &haystack,
        &[b"a\x00a"],
        false,
        "test_08_null_byte_in_haystack",
    );
}

#[test]
fn test_09_null_byte_in_pattern() {
    let mut haystack = vec![b'a'; 100];
    haystack[20] = 0x00;
    haystack[40] = 0x00;
    assert_matches_reference(&haystack, &[b"\x00"], false, "test_09_null_byte_in_pattern");
}

#[test]
fn test_10_multiple_null_bytes_in_pattern() {
    let mut haystack = vec![b'a'; 100];
    haystack[20] = 0x00;
    haystack[21] = 0x00;
    assert_matches_reference(
        &haystack,
        &[b"\x00\x00"],
        false,
        "test_10_multiple_null_bytes_in_pattern",
    );
}

#[test]
fn test_11_single_byte_pattern_all_matches() {
    let haystack = vec![b'x'; 200];
    assert_matches_reference(
        &haystack,
        &[b"x"],
        false,
        "test_11_single_byte_pattern_all_matches",
    );
}

#[test]
fn test_12_single_byte_pattern_alternating() {
    let mut haystack = vec![b'y'; 200];
    for i in (0..200).step_by(2) {
        haystack[i] = b'x';
    }
    assert_matches_reference(
        &haystack,
        &[b"x"],
        false,
        "test_12_single_byte_pattern_alternating",
    );
}

#[test]
fn test_13_patterns_with_common_prefix_2_bytes() {
    let haystack = b"abcdefg abcdxyz abcpqr";
    assert_matches_reference(
        haystack,
        &[b"abcd", b"abce"],
        false,
        "test_13_patterns_with_common_prefix_2_bytes",
    );
}

#[test]
fn test_14_patterns_with_common_prefix_4_bytes() {
    let haystack = b"1234567 1234abc 1234xyz";
    assert_matches_reference(
        haystack,
        &[b"12345", b"1234a", b"1234x"],
        false,
        "test_14_patterns_with_common_prefix_4_bytes",
    );
}

#[test]
fn test_15_pattern_longer_than_64_bytes() {
    let mut pattern = vec![b'p'; 70];
    pattern[69] = b'q';
    let mut haystack = vec![b'a'; 200];
    haystack[50..120].copy_from_slice(&pattern);
    assert_matches_reference(
        &haystack,
        &[&pattern],
        false,
        "test_15_pattern_longer_than_64_bytes",
    );
}

#[test]
fn test_16_pattern_longer_than_64_bytes_prefix_match_only() {
    let mut pattern = vec![b'p'; 70];
    pattern[69] = b'q';
    let mut haystack = vec![b'a'; 200];
    // Put prefix but not full pattern
    haystack[50..119].copy_from_slice(&pattern[0..69]);
    haystack[119] = b'z';
    assert_matches_reference(
        &haystack,
        &[&pattern],
        false,
        "test_16_pattern_longer_than_64_bytes_prefix_match_only",
    );
}

#[test]
fn test_17_multiple_patterns_longer_than_64_bytes() {
    let pattern1 = vec![b'p'; 70];
    let pattern2 = vec![b'q'; 80];
    let mut haystack = vec![b'a'; 300];
    haystack[10..80].copy_from_slice(&pattern1);
    haystack[100..180].copy_from_slice(&pattern2);
    assert_matches_reference(
        &haystack,
        &[&pattern1, &pattern2],
        false,
        "test_17_multiple_patterns_longer_than_64_bytes",
    );
}

#[test]
fn test_18_estimate_match_count_4kb_boundary_exact() {
    let mut haystack = vec![b'a'; 8192];
    haystack[4095] = b'b'; // Match exactly at the 4096 boundary (index 4095)
    let count = SimdSieve::estimate_match_count(&haystack, &[b"b"], false).unwrap();
    assert_eq!(count, 1, "Should count 1 match within the first 4KB");
}

#[test]
fn test_19_estimate_match_count_4kb_boundary_outside() {
    let mut haystack = vec![b'a'; 8192];
    haystack[4096] = b'b'; // Match exactly outside the 4096 boundary (index 4096)
    let count = SimdSieve::estimate_match_count(&haystack, &[b"b"], false).unwrap();
    assert_eq!(count, 0, "Should count 0 matches since it is beyond 4KB");
}

#[test]
fn test_20_estimate_match_count_multiple_within_4kb() {
    let mut haystack = vec![b'a'; 8192];
    haystack[0] = b'x';
    haystack[2048] = b'x';
    haystack[4095] = b'x';
    haystack[4096] = b'x'; // Outside
    haystack[8191] = b'x'; // Outside
    let count = SimdSieve::estimate_match_count(&haystack, &[b"x"], false).unwrap();
    assert_eq!(count, 3, "Should count exactly 3 matches within first 4KB");
}

#[test]
fn test_21_case_insensitive_common_prefix() {
    let haystack = b"aBcDeFg aBcDxYz";
    assert_matches_reference(
        haystack,
        &[b"abcdef", b"abcdxy"],
        true,
        "test_21_case_insensitive_common_prefix",
    );
}

#[test]
fn test_22_case_insensitive_null_byte() {
    let haystack = b"aBc\x00dEf";
    assert_matches_reference(
        haystack,
        &[b"abc\x00def"],
        true,
        "test_22_case_insensitive_null_byte",
    );
}

#[test]
fn test_23_case_insensitive_long_pattern() {
    let mut pattern = vec![b'P'; 70];
    pattern[69] = b'q';
    let mut haystack = vec![b'a'; 200];
    haystack[50..120].copy_from_slice(&pattern);
    // Mutate haystack to be lowercase
    haystack[50..119].fill(b'p');
    assert_matches_reference(
        &haystack,
        &[&pattern],
        true,
        "test_23_case_insensitive_long_pattern",
    );
}

#[test]
fn test_24_pattern_equals_haystack_64_bytes() {
    let haystack = vec![b'x'; 64];
    assert_matches_reference(
        &haystack,
        &[&haystack],
        false,
        "test_24_pattern_equals_haystack_64_bytes",
    );
}

#[test]
fn test_25_pattern_equals_haystack_63_bytes() {
    let haystack = vec![b'x'; 63];
    assert_matches_reference(
        &haystack,
        &[&haystack],
        false,
        "test_25_pattern_equals_haystack_63_bytes",
    );
}

#[test]
fn test_26_16_patterns_common_prefix() {
    let haystack = b"p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15";
    let pattern_bytes: Vec<Vec<u8>> = (0..16).map(|i| format!("p{i}").into_bytes()).collect();
    let patterns: Vec<&[u8]> = pattern_bytes.iter().map(std::vec::Vec::as_slice).collect();
    assert_matches_reference(
        haystack,
        &patterns,
        false,
        "test_26_16_patterns_common_prefix",
    );
}

#[test]
fn test_27_estimate_match_count_all_matches_4kb() {
    let haystack = vec![b'x'; 8192];
    let count = SimdSieve::estimate_match_count(&haystack, &[b"x"], false).unwrap();
    assert_eq!(count, 4096, "Should count 4096 matches in the first 4KB");
}

#[test]
fn test_28_estimate_match_count_overlapping_4kb() {
    let haystack = vec![b'a'; 8192];
    // "aa" overlaps. estimate_match_count counts prefix hits.
    // The prefix of "aa" is "aa".
    // "aa" appears at indices 0..4095.
    let count = SimdSieve::estimate_match_count(&haystack, &[b"aa"], false).unwrap();
    assert_eq!(
        count, 4095,
        "Should count 4095 prefix hits in the first 4KB"
    );
}

#[test]
fn test_29_estimate_match_count_case_insensitive() {
    let mut haystack = vec![b'a'; 8192];
    haystack[10] = b'A';
    haystack[20] = b'X';
    let count = SimdSieve::estimate_match_count(&haystack, &[b"X"], true).unwrap();
    assert_eq!(count, 1, "Should count 1 match for X (case insensitive)");
}

#[test]
fn test_30_estimate_match_count_empty_haystack() {
    let count = SimdSieve::estimate_match_count(b"", &[b"x"], false).unwrap();
    assert_eq!(count, 0, "Should count 0 matches in empty haystack");
}

#[test]
fn test_31_estimate_match_count_haystack_less_than_4kb() {
    let haystack = vec![b'x'; 100];
    let count = SimdSieve::estimate_match_count(&haystack, &[b"x"], false).unwrap();
    assert_eq!(count, 100, "Should count exactly 100 matches");
}

#[test]
fn test_32_estimate_match_count_multiple_patterns() {
    let mut haystack = vec![b'a'; 8192];
    haystack[100] = b'x';
    haystack[200] = b'y';
    haystack[300] = b'z';
    let count = SimdSieve::estimate_match_count(&haystack, &[b"x", b"y", b"z"], false).unwrap();
    assert_eq!(count, 3, "Should count 3 matches in the first 4KB");
}

#[test]
fn test_33_simd_alignment_boundary_64_bytes() {
    // Offset match at exact alignment block boundaries
    let mut haystack = vec![b'a'; 256];
    haystack[63] = b'x'; // End of block 0
    haystack[64] = b'x'; // Start of block 1
    haystack[127] = b'x'; // End of block 1
    haystack[128] = b'x'; // Start of block 2
    assert_matches_reference(
        &haystack,
        &[b"x"],
        false,
        "test_33_simd_alignment_boundary_64_bytes",
    );
}

#[test]
fn test_34_simd_alignment_boundary_pattern_spanning() {
    // Offset match spanning block boundaries
    let mut haystack = vec![b'a'; 256];
    haystack[63] = b'x';
    haystack[64] = b'y';
    haystack[127] = b'x';
    haystack[128] = b'y';
    assert_matches_reference(
        &haystack,
        &[b"xy"],
        false,
        "test_34_simd_alignment_boundary_pattern_spanning",
    );
}
