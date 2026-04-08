#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::{SimdSieve, SimdSieveError};

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

// Group 1: 8 patterns all identical (dedup behavior)

#[test]
fn test_eight_identical_patterns() {
    let haystack = b"abc abc abc abc";
    let pattern = b"abc";
    let patterns = vec![pattern as &[u8]; 8];
    assert_matches_reference(haystack, &patterns, false, "eight_identical_patterns");
}

#[test]
fn test_eight_identical_patterns_overlapping() {
    let haystack = b"aaaaa";
    let pattern = b"aa";
    let patterns = vec![pattern as &[u8]; 8];
    assert_matches_reference(haystack, &patterns, false, "eight_identical_patterns_overlapping");
}

#[test]
fn test_eight_identical_patterns_empty() {
    let haystack = b"abc";
    let patterns = vec![b"" as &[u8]; 8];
    let result = SimdSieve::new(haystack, &patterns);
    assert!(result.is_err());
}

#[test]
fn test_eight_identical_patterns_case_insensitive() {
    let haystack = b"AbC aBc ABC abc";
    let pattern = b"abc";
    let patterns = vec![pattern as &[u8]; 8];
    assert_matches_reference(haystack, &patterns, true, "eight_identical_patterns_case_insensitive");
}

// Group 2: pattern that is 1 byte (minimum)

#[test]
fn test_one_byte_pattern() {
    let haystack = b"x";
    assert_matches_reference(haystack, &[b"x"], false, "one_byte_pattern");
}

#[test]
fn test_one_byte_pattern_at_end() {
    let haystack = b"hello x";
    assert_matches_reference(haystack, &[b"x"], false, "one_byte_pattern_at_end");
}

#[test]
fn test_one_byte_pattern_at_start() {
    let haystack = b"x hello";
    assert_matches_reference(haystack, &[b"x"], false, "one_byte_pattern_at_start");
}

#[test]
fn test_one_byte_pattern_multiple_times() {
    let haystack = b"xaxbxxcx";
    assert_matches_reference(haystack, &[b"x"], false, "one_byte_pattern_multiple_times");
}

// Group 3: pattern that is 128 bytes (exceeds SIMD register)

#[test]
fn test_128_byte_pattern() {
    let mut haystack = vec![b'a'; 256];
    let pattern = vec![b'b'; 128];
    haystack[50..178].copy_from_slice(&pattern);
    assert_matches_reference(&haystack, &[&pattern], false, "128_byte_pattern");
}

#[test]
fn test_128_byte_pattern_at_start() {
    let mut haystack = vec![b'a'; 256];
    let pattern = vec![b'b'; 128];
    haystack[0..128].copy_from_slice(&pattern);
    assert_matches_reference(&haystack, &[&pattern], false, "128_byte_pattern_at_start");
}

#[test]
fn test_128_byte_pattern_at_end() {
    let mut haystack = vec![b'a'; 256];
    let pattern = vec![b'b'; 128];
    haystack[128..256].copy_from_slice(&pattern);
    assert_matches_reference(&haystack, &[&pattern], false, "128_byte_pattern_at_end");
}

#[test]
fn test_128_byte_pattern_exact_haystack() {
    let pattern = vec![b'b'; 128];
    assert_matches_reference(&pattern, &[&pattern], false, "128_byte_pattern_exact_haystack");
}

// Group 4: input of exactly 64 bytes (one AVX2 register)

#[test]
fn test_exactly_64_bytes_input() {
    let haystack = vec![b'a'; 64];
    assert_matches_reference(&haystack, &[b"a"], false, "exactly_64_bytes_input");
}

#[test]
fn test_exactly_64_bytes_input_no_match() {
    let haystack = vec![b'a'; 64];
    assert_matches_reference(&haystack, &[b"b"], false, "exactly_64_bytes_input_no_match");
}

#[test]
fn test_exactly_64_bytes_input_pattern_matches_all() {
    let haystack = vec![b'a'; 64];
    let pattern = vec![b'a'; 64];
    assert_matches_reference(&haystack, &[&pattern], false, "exactly_64_bytes_input_pattern_matches_all");
}

#[test]
fn test_exactly_64_bytes_input_multiple_patterns() {
    let mut haystack = vec![b'a'; 64];
    haystack[0] = b'1';
    haystack[31] = b'2';
    haystack[63] = b'3';
    assert_matches_reference(&haystack, &[b"1", b"2", b"3"], false, "exactly_64_bytes_input_multiple_patterns");
}

// Group 5: input of 63 bytes (off-by-one)

#[test]
fn test_63_bytes_input() {
    let haystack = vec![b'a'; 63];
    assert_matches_reference(&haystack, &[b"a"], false, "63_bytes_input");
}

#[test]
fn test_63_bytes_input_no_match() {
    let haystack = vec![b'a'; 63];
    assert_matches_reference(&haystack, &[b"b"], false, "63_bytes_input_no_match");
}

#[test]
fn test_63_bytes_input_pattern_matches_all() {
    let haystack = vec![b'a'; 63];
    let pattern = vec![b'a'; 63];
    assert_matches_reference(&haystack, &[&pattern], false, "63_bytes_input_pattern_matches_all");
}

#[test]
fn test_63_bytes_input_multiple_patterns() {
    let mut haystack = vec![b'a'; 63];
    haystack[0] = b'1';
    haystack[31] = b'2';
    haystack[62] = b'3';
    assert_matches_reference(&haystack, &[b"1", b"2", b"3"], false, "63_bytes_input_multiple_patterns");
}

// Group 6: input of 0 bytes

#[test]
fn test_0_bytes_input() {
    let haystack = b"";
    assert_matches_reference(haystack, &[b"a"], false, "0_bytes_input");
}

#[test]
fn test_0_bytes_input_multiple_patterns() {
    let haystack = b"";
    assert_matches_reference(haystack, &[b"a", b"b", b"c"], false, "0_bytes_input_multiple_patterns");
}

#[test]
fn test_0_bytes_input_empty_pattern() {
    let haystack = b"";
    let result = SimdSieve::new(haystack, &[b""]);
    assert!(result.is_err());
}

#[test]
fn test_0_bytes_input_case_insensitive() {
    let haystack = b"";
    assert_matches_reference(haystack, &[b"a"], true, "0_bytes_input_case_insensitive");
}

// Group 7: input of 1 byte matching a 1-byte pattern

#[test]
fn test_1_byte_input_1_byte_pattern() {
    let haystack = b"x";
    assert_matches_reference(haystack, &[b"x"], false, "1_byte_input_1_byte_pattern");
}

#[test]
fn test_1_byte_input_1_byte_pattern_no_match() {
    let haystack = b"x";
    assert_matches_reference(haystack, &[b"y"], false, "1_byte_input_1_byte_pattern_no_match");
}

#[test]
fn test_1_byte_input_1_byte_pattern_case_insensitive() {
    let haystack = b"X";
    assert_matches_reference(haystack, &[b"x"], true, "1_byte_input_1_byte_pattern_case_insensitive");
}

#[test]
fn test_1_byte_input_multiple_patterns() {
    let haystack = b"x";
    assert_matches_reference(haystack, &[b"y", b"x", b"z"], false, "1_byte_input_multiple_patterns");
}

// Group 8: all 256 single-byte patterns simultaneously

#[test]
fn test_all_256_single_byte_patterns() {
    let patterns: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    let haystack = b"test";
    
    // We expect an error due to pattern limit exceeded
    let result = SimdSieve::new(haystack, &pattern_refs);
    assert!(
        matches!(result, Err(SimdSieveError::PatternLimitExceeded(256))),
        "Expected PatternLimitExceeded error"
    );
}

#[test]
fn test_all_256_single_byte_patterns_empty_haystack() {
    let patterns: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    let haystack = b"";
    
    let result = SimdSieve::new(haystack, &pattern_refs);
    assert!(
        matches!(result, Err(SimdSieveError::PatternLimitExceeded(256))),
        "Expected PatternLimitExceeded error"
    );
}

#[test]
fn test_all_256_single_byte_patterns_case_insensitive() {
    let patterns: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    let haystack = b"test";
    
    let result = SimdSieve::new_case_insensitive(haystack, &pattern_refs);
    assert!(
        matches!(result, Err(SimdSieveError::PatternLimitExceeded(256))),
        "Expected PatternLimitExceeded error"
    );
}

// Group 9: score_density at exactly 4096 byte boundary

#[test]
fn test_score_density_4096_boundary() {
    let haystack = vec![b'x'; 4096];
    let count = SimdSieve::estimate_match_count(&haystack, &[b"x"], false);
    assert!(count > 0, "Expected count > 0, got {}", count);
}

#[test]
fn test_score_density_4096_boundary_no_match() {
    let haystack = vec![b'x'; 4096];
    let count = SimdSieve::estimate_match_count(&haystack, &[b"y"], false);
    assert_eq!(count, 0, "Expected count == 0, got {}", count);
}

#[test]
fn test_score_density_4096_boundary_multiple_patterns() {
    let mut haystack = vec![b'x'; 4096];
    haystack[0] = b'y';
    haystack[4095] = b'z';
    let count = SimdSieve::estimate_match_count(&haystack, &[b"y", b"z"], false);
    assert!(count > 0, "Expected count > 0, got {}", count);
}

#[test]
fn test_score_density_4096_boundary_case_insensitive() {
    let haystack = vec![b'X'; 4096];
    let count = SimdSieve::estimate_match_count(&haystack, &[b"x"], true);
    assert!(count > 0, "Expected count > 0, got {}", count);
}

// Group 10: case-insensitive matching for all ASCII letters

#[test]
fn test_case_insensitive_all_ascii_letters_chunk1() {
    let mut haystack = Vec::new();
    let mut patterns = Vec::new();
    for b in b'A'..=b'P' { // 16 letters
        haystack.push(b);
        patterns.push(vec![b.to_ascii_lowercase()]);
    }
    let refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    assert_matches_reference(&haystack, &refs, true, "case_insensitive_all_ascii_letters_chunk1");
}

#[test]
fn test_case_insensitive_all_ascii_letters_chunk2() {
    let mut haystack = Vec::new();
    let mut patterns = Vec::new();
    for b in b'Q'..=b'Z' { // 10 letters
        haystack.push(b);
        patterns.push(vec![b.to_ascii_lowercase()]);
    }
    let refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    assert_matches_reference(&haystack, &refs, true, "case_insensitive_all_ascii_letters_chunk2");
}

#[test]
fn test_case_insensitive_all_ascii_letters_lowercase_haystack() {
    let mut haystack = Vec::new();
    let mut patterns = Vec::new();
    for b in b'a'..=b'p' { // 16 letters
        haystack.push(b);
        patterns.push(vec![b.to_ascii_uppercase()]);
    }
    let refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    assert_matches_reference(&haystack, &refs, true, "case_insensitive_all_ascii_letters_lowercase_haystack");
}

#[test]
fn test_case_insensitive_all_ascii_letters_mixed_haystack() {
    let haystack = b"aBcDeFgHiJkLmNoP";
    let mut patterns = Vec::new();
    for b in b'A'..=b'P' { // 16 letters
        patterns.push(vec![b.to_ascii_lowercase()]);
    }
    let refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    assert_matches_reference(haystack, &refs, true, "case_insensitive_all_ascii_letters_mixed_haystack");
}

