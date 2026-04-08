#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Cross-backend parity and edge-case tests.
//!
//! These tests verify that every supported backend produces identical,
/// correct results for adversarial edge cases.
use simdsieve::{SimdSieve, SimdSieveError};

/// Reference implementation used as ground truth.
fn reference_scan(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut hits = Vec::new();
    for i in 0..=haystack.len() {
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

fn assert_parity(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool, test_name: &str) {
    let sieve = if case_insensitive {
        SimdSieve::new_case_insensitive(haystack, patterns)
    } else {
        SimdSieve::new(haystack, patterns)
    }
    .unwrap_or_else(|e| panic!("[{test_name}] construction failed: {e:?}"));

    let actual: Vec<usize> = sieve.collect();
    let expected = reference_scan(haystack, patterns, case_insensitive);
    assert_eq!(
        actual,
        expected,
        "[{test_name}] mismatch: haystack_len={}, patterns={:?}, actual={actual:?}, expected={expected:?}",
        haystack.len(),
        patterns
            .iter()
            .map(|p| String::from_utf8_lossy(p))
            .collect::<Vec<_>>(),
    );
}

// ============================================================================
// Empty and minimal inputs
// ============================================================================

#[test]
fn empty_haystack_empty_patterns() {
    assert!(matches!(
        SimdSieve::new(b"", &[]),
        Err(SimdSieveError::EmptyPatternSet)
    ));
}

#[test]
fn empty_haystack_with_pattern() {
    assert_parity(b"", &[b"x"], false, "empty_haystack_with_pattern");
}

#[test]
fn one_byte_haystack_match() {
    assert_parity(b"A", &[b"A"], false, "one_byte_haystack_match");
}

#[test]
fn one_byte_haystack_no_match() {
    assert_parity(b"A", &[b"B"], false, "one_byte_haystack_no_match");
}

#[test]
fn one_byte_pattern_at_last_byte() {
    let mut haystack = vec![b'x'; 127];
    haystack[126] = b'Z';
    assert_parity(
        &haystack,
        &[b"Z"],
        false,
        "one_byte_pattern_at_last_byte_127",
    );
}

// ============================================================================
// 64-byte boundary stress
// ============================================================================

#[test]
fn pattern_at_exact_64_boundary() {
    for boundary in [0, 31, 32, 33, 63, 64, 65] {
        let mut haystack = vec![b'x'; 128];
        if boundary + 4 <= haystack.len() {
            haystack[boundary..boundary + 4].copy_from_slice(b"TEST");
        }
        assert_parity(
            &haystack,
            &[b"TEST"],
            false,
            &format!("boundary_{boundary}"),
        );
    }
}

#[test]
fn pattern_spanning_64_boundary() {
    let mut haystack = vec![b'x'; 128];
    // Pattern starts at 62 and ends at 66
    haystack[62..67].copy_from_slice(b"SPANN");
    assert_parity(
        &haystack,
        &[b"SPANN"],
        false,
        "pattern_spanning_64_boundary",
    );
}

#[test]
fn exact_64_byte_haystack() {
    let mut haystack = vec![b'x'; 64];
    haystack[60..64].copy_from_slice(b"END!");
    assert_parity(&haystack, &[b"END!"], false, "exact_64_byte_haystack");
}

#[test]
fn exact_64_byte_all_same() {
    let haystack = vec![b'A'; 64];
    assert_parity(&haystack, &[b"AAAA"], false, "exact_64_byte_all_same");
}

// ============================================================================
// 128-byte boundary stress (covers AVX-512)
// ============================================================================

#[test]
fn pattern_at_exact_128_boundary() {
    for boundary in [126, 127, 128, 129] {
        let mut haystack = vec![b'x'; 256];
        if boundary + 4 <= haystack.len() {
            haystack[boundary..boundary + 4].copy_from_slice(b"TEST");
        }
        assert_parity(
            &haystack,
            &[b"TEST"],
            false,
            &format!("boundary_128_{boundary}"),
        );
    }
}

#[test]
fn exact_128_byte_haystack() {
    let mut haystack = vec![b'x'; 128];
    haystack[124..128].copy_from_slice(b"LAST");
    assert_parity(&haystack, &[b"LAST"], false, "exact_128_byte_haystack");
}

// ============================================================================
// Multi-byte prefixes and pattern lengths
// ============================================================================

#[test]
fn all_prefix_lengths_1_to_8() {
    let haystack = b"abcdefghijklmnopqrstuvwxyz0123456789";
    for len in 1..=8usize {
        let pat = &haystack[10..10 + len];
        assert_parity(haystack, &[pat], false, &format!("prefix_len_{len}"));
    }
}

#[test]
fn pattern_longer_than_four_bytes() {
    let haystack = b"prefix_match_1234567890abcdefghij";
    assert_parity(
        haystack,
        &[b"prefix_match_12345"],
        false,
        "pattern_longer_than_4",
    );
}

// ============================================================================
// Case-insensitive edge cases
// ============================================================================

#[test]
fn case_insensitive_at_boundaries() {
    let mut haystack = vec![b'x'; 128];
    haystack[31] = b'a';
    haystack[63] = b'B';
    haystack[95] = b'c';
    haystack[127] = b'D';
    assert_parity(
        &haystack,
        &[b"a", b"b", b"c", b"d"],
        true,
        "ci_at_boundaries",
    );
}

#[test]
fn case_insensitive_mixed_ascii_non_ascii() {
    let haystack: Vec<u8> = vec![0xC0, b'a', 0xE0, b'B', 0xFF];
    assert_parity(&haystack, &[b"ab"], true, "ci_mixed_ascii_non_ascii");
}

// ============================================================================
// Multiple patterns (up to 16)
// ============================================================================

#[test]
fn sixteen_patterns_at_various_offsets() {
    let mut haystack = vec![b'.'; 256];
    let patterns: Vec<Vec<u8>> = (0..16).map(|i| vec![b'A' + i as u8]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

    for i in 0..16 {
        haystack[i * 15] = b'A' + i as u8;
    }

    assert_parity(
        &haystack,
        &pattern_refs,
        false,
        "sixteen_patterns_various_offsets",
    );
}

#[test]
fn overlapping_patterns_same_prefix() {
    let haystack = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    assert_parity(
        haystack,
        &[b"aaa", b"aaaa", b"aaaaa"],
        false,
        "overlapping_patterns_same_prefix",
    );
}

// ============================================================================
// Binary / adversarial data
// ============================================================================

#[test]
fn all_byte_values_as_single_byte_patterns() {
    for byte in 0u8..=255 {
        let mut haystack = vec![0u8; 64];
        haystack[32] = byte;
        let pattern = vec![byte];
        assert_parity(
            &haystack,
            &[&pattern],
            false,
            &format!("all_bytes_{byte:02X}"),
        );
    }
}

#[test]
fn null_bytes_everywhere() {
    let mut haystack = vec![0xFF; 64];
    haystack[0] = 0x00;
    haystack[31] = 0x00;
    haystack[32] = 0x00;
    haystack[63] = 0x00;
    assert_parity(&haystack, &[b"\x00"], false, "null_bytes_everywhere");
    assert_parity(&haystack, &[b"\x00\xFF"], false, "null_prefix_pair");
}

#[test]
fn max_value_0xff_repeated() {
    let haystack = vec![0xFF; 64];
    assert_parity(
        &haystack,
        &[b"\xFF\xFF\xFF"],
        false,
        "max_value_0xff_repeated",
    );
}

// ============================================================================
// Density and tail processing
// ============================================================================

#[test]
fn pattern_at_very_last_byte_single() {
    let mut haystack = vec![b'x'; 65];
    haystack[64] = b'Z';
    assert_parity(
        &haystack,
        &[b"Z"],
        false,
        "pattern_at_very_last_byte_single",
    );
}

#[test]
fn pattern_at_very_last_byte_multi() {
    let mut haystack = vec![b'x'; 66];
    haystack[63] = b'A';
    haystack[64] = b'B';
    haystack[65] = b'C';
    assert_parity(
        &haystack,
        &[b"ABC"],
        false,
        "pattern_at_very_last_byte_multi",
    );
}

#[test]
fn pattern_exceeds_haystack_length() {
    assert_parity(
        b"short",
        &[b"this_is_longer"],
        false,
        "pattern_exceeds_haystack",
    );
}

#[test]
fn estimate_match_count_parity() {
    // Use a haystack large enough to trigger block processing in every backend
    // (AVX-512 needs at least 128 + tail_req bytes).
    let haystack = b"abababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababab";
    let count = SimdSieve::estimate_match_count(haystack, &[b"ab"], false);
    // estimate_match_count counts prefix hits in SIMD blocks only.
    assert!(
        count > 0,
        "estimate_match_count should find prefix hits in block-processing region"
    );
    assert!(
        count <= haystack.len() as u64,
        "estimate_match_count should not exceed haystack length"
    );
}
