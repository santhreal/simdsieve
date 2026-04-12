#![allow(clippy::panic, clippy::unwrap_used)]

use rand::{RngCore, SeedableRng, rngs::StdRng};
use simdsieve::SimdSieve;

fn reference_scan(haystack: &[u8], patterns: &[&[u8]]) -> Vec<usize> {
    let mut hits = Vec::new();
    for start in 0..haystack.len() {
        if patterns.iter().any(|pattern| {
            start + pattern.len() <= haystack.len()
                && &haystack[start..start + pattern.len()] == *pattern
        }) {
            hits.push(start);
        }
    }
    hits
}

#[test]
fn avx2_vs_scalar_parity_on_random_1mb_input() {
    #[cfg(not(target_arch = "x86_64"))]
    {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
    }

    let mut rng = StdRng::seed_from_u64(0x5EED_FACE_CAFE_BEEF);
    let mut haystack = vec![0_u8; 1024 * 1024];
    rng.fill_bytes(&mut haystack);
    haystack[31_337..31_341].copy_from_slice(b"EDGE");
    haystack[262_144..262_147].copy_from_slice(b"aba");
    let tail_start = haystack.len() - 4;
    haystack[tail_start..].copy_from_slice(b"TAIL");

    let patterns: &[&[u8]] = &[b"EDGE", b"aba", b"TAIL", b"\0\0", b"\xff\x00"];
    let expected = reference_scan(&haystack, patterns);
    let actual: Vec<usize> = SimdSieve::new(&haystack, patterns)
        .expect("simdsieve should build on 1MB random input")
        .collect();

    assert_eq!(
        actual, expected,
        "runtime SIMD path diverged from scalar reference"
    );
}

#[test]
fn pattern_at_exact_end_of_buffer_is_reported() {
    let haystack = b"prefix::needle";
    let patterns: &[&[u8]] = &[b"needle"];

    let actual: Vec<usize> = SimdSieve::new(haystack, patterns)
        .expect("construction should succeed")
        .collect();

    assert_eq!(
        actual,
        vec![8],
        "match at the exact end of the buffer was lost"
    );
}

#[test]
fn overlapping_patterns_surface_every_valid_offset() {
    let haystack = b"aaaaa";
    let patterns: &[&[u8]] = &[b"aaaa", b"aaa"];

    let actual: Vec<usize> = SimdSieve::new(haystack, patterns)
        .expect("construction should succeed")
        .collect();

    assert_eq!(
        actual,
        vec![0, 1, 2],
        "overlapping matches were collapsed incorrectly"
    );
}

#[test]
fn empty_haystack_yields_no_matches() {
    let actual: Vec<usize> = SimdSieve::new(b"", &[b"abc"])
        .expect("empty haystack should be a valid search target")
        .collect();

    assert!(
        actual.is_empty(),
        "empty haystack should never produce matches"
    );
}

#[test]
fn pattern_longer_than_haystack_never_matches() {
    let actual: Vec<usize> = SimdSieve::new(b"tiny", &[b"this is much longer"])
        .expect("construction should succeed for long patterns")
        .collect();

    assert!(
        actual.is_empty(),
        "pattern longer than haystack must not match"
    );
}
