#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use simdsieve::SimdSieve;

#[allow(clippy::if_same_then_else)]
#[test]
fn test_simd_scalar_absolute_parity_multi_pattern() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut haystack = vec![0u8; 100_000];
    rng.fill_bytes(&mut haystack);

    let t1 = [haystack[500], haystack[501], haystack[502]];
    let t2 = [haystack[1500], haystack[1501]];

    let hw_sieve = SimdSieve::new(&haystack, &[&t1, &t2]).unwrap();
    let hw_results: Vec<usize> = hw_sieve.collect();

    let mut expected = Vec::new();
    for i in 0..haystack.len() {
        if i + 3 <= haystack.len() && haystack[i..i + 3] == t1 {
            expected.push(i);
        } else if i + 2 <= haystack.len() && haystack[i..i + 2] == t2 {
            expected.push(i);
        }
    }

    expected.sort_unstable();
    expected.dedup();

    assert_eq!(
        hw_results, expected,
        "SIMD sieve output must match brute-force linear scan"
    );
}

/// Reference brute-force scan shared by the seeded sweeps below. Ground truth
/// is a naive left-to-right window check per pattern; positions are sorted and
/// deduped to match `SimdSieve` output semantics.
fn brute_force_multi(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut hits = Vec::new();
    for i in 0..haystack.len() {
        for &pat in patterns {
            if i + pat.len() > haystack.len() {
                continue;
            }
            let window = &haystack[i..i + pat.len()];
            let matched = if case_insensitive {
                window
                    .iter()
                    .zip(pat)
                    .all(|(&a, &b)| a.eq_ignore_ascii_case(&b))
            } else {
                window == pat
            };
            if matched {
                hits.push(i);
                break;
            }
        }
    }
    hits.sort_unstable();
    hits.dedup();
    hits
}

fn assert_seeded_parity(
    haystack: &[u8],
    patterns: &[&[u8]],
    case_insensitive: bool,
    label: &str,
) {
    let sieve = if case_insensitive {
        SimdSieve::new_case_insensitive(haystack, patterns).unwrap()
    } else {
        SimdSieve::new(haystack, patterns).unwrap()
    };
    let actual: Vec<usize> = sieve.collect();
    let expected = brute_force_multi(haystack, patterns, case_insensitive);
    assert_eq!(
        actual, expected,
        "[{label}] parity mismatch: haystack_len={}, pattern_count={}",
        haystack.len(),
        patterns.len()
    );
}

/// Builds a pattern set that mixes needles guaranteed to hit (substrings
/// copied out of the haystack) with random needles that almost never hit, so
/// both the match and no-match paths are exercised per corpus.
fn mixed_patterns(rng: &mut StdRng, haystack: &[u8], count: usize) -> Vec<Vec<u8>> {
    let mut patterns = Vec::with_capacity(count);
    for i in 0..count {
        let len = 1 + (rng.next_u64() as usize % 12);
        if i % 2 == 0 && haystack.len() >= len {
            let start = rng.next_u64() as usize % (haystack.len() - len + 1);
            patterns.push(haystack[start..start + len].to_vec());
        } else {
            patterns.push((0..len).map(|_| (rng.next_u32() & 0xFF) as u8).collect());
        }
    }
    patterns
}

/// Fixed-seed randomized corpora parity sweep: 22 corpora spanning SIMD lane
/// boundaries (16/32/64/128/256 bytes), 1..=16 patterns each, case-sensitive
/// and case-insensitive. Every corpus must match the brute-force reference
/// exactly. Exists because single-corpus parity tests cannot catch backend
/// divergence that only appears at specific length/lane-count combinations.
#[test]
fn test_seeded_corpora_parity_sweep() {
    let mut rng = StdRng::seed_from_u64(0x5EED_5EED_5EED_0001);
    let sizes = [
        0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512,
        1024, 4096, 65_537,
    ];
    for (round, &size) in sizes.iter().enumerate() {
        let mut haystack = vec![0u8; size];
        rng.fill_bytes(&mut haystack);
        let pattern_count = 1 + (rng.next_u64() as usize % 16);
        let owned = mixed_patterns(&mut rng, &haystack, pattern_count);
        let patterns: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        assert_seeded_parity(&haystack, &patterns, false, "seeded-sweep-sensitive");
        if round % 3 == 0 {
            assert_seeded_parity(&haystack, &patterns, true, "seeded-sweep-insensitive");
        }
    }
}

/// Pathological all-same-byte haystacks at lane-boundary sizes: both the
/// matching byte (every position hits) and a non-matching byte (no position
/// hits) are swept, because degenerate inputs are where SIMD prefix/suffix
/// handling most often diverges from a naive scan.
#[test]
fn test_all_same_byte_haystack_parity_sweep() {
    let mut rng = StdRng::seed_from_u64(0xA115_0A11_5A1E_0002);
    for &size in &[3usize, 16, 31, 32, 64, 128, 256, 1000, 8192] {
        for &byte in &[0x00u8, 0x41, 0x61, 0xFF] {
            let haystack = vec![byte; size];
            let hit_len = (1 + (rng.next_u64() as usize % 8)).min(size.max(1));
            let hit: Vec<u8> = vec![byte; hit_len];
            let miss: Vec<u8> = vec![byte.wrapping_add(1); 3];
            let patterns: Vec<&[u8]> = vec![&hit, &miss];
            assert_seeded_parity(&haystack, &patterns, false, "all-same-byte");
            assert_seeded_parity(&haystack, &patterns, true, "all-same-byte-ci");
        }
    }
}
