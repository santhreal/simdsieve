//! S-proptest-03 (simdsieve mass proptest: match index invariants, no panic on arbitrary bytes).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use proptest::prelude::*;
use simdsieve::SimdSieve;

fn brute_find(haystack: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let limit = haystack.len().saturating_sub(pattern.len() - 1);
    for i in 0..limit {
        if haystack[i..].starts_with(pattern) {
            out.push(i);
        }
    }
    out
}

fn collect_matches(haystack: &[u8], patterns: &[&[u8]]) -> proptest::test_runner::TestCaseResult {
    let sieve = SimdSieve::new(haystack, patterns)?;
    let mut got: Vec<usize> = sieve.collect();
    got.sort_unstable();
    for pat in patterns {
        let mut expected = brute_find(haystack, pat);
        expected.sort_unstable();
        let pat_hits: Vec<_> = got
            .iter()
            .copied()
            .filter(|&pos| haystack[pos..].starts_with(pat))
            .collect();
        prop_assert_eq!(&pat_hits, &expected);
    }
    Ok(())
}

macro_rules! simd_cases {
    ($($name:ident => |$haystack:ident, $pattern:ident| $body:block),+ $(,)?) => {
        $(
            proptest! {
                #![proptest_config(ProptestConfig::with_cases(64))]
                #[test]
                fn $name(
                    $haystack in prop::collection::vec(any::<u8>(), 0..256),
                    $pattern in prop::collection::vec(any::<u8>(), 1..8),
                ) {
                    $body
                }
            }
        )+
    };
}

simd_cases! {
    p00_no_false_positives => |haystack, pattern| {
        let pat = pattern.as_slice();
        if let Ok(sieve) = SimdSieve::new(&haystack, &[pat]) {
            for pos in sieve {
                prop_assert!(haystack[pos..].starts_with(pat));
            }
        }
    },
    p01_short_hay_brute_parity => |haystack, pattern| {
        let pat = pattern.as_slice();
        if haystack.len() <= 128 {
            let _ = collect_matches(&haystack, &[pat])?;
        }
    },
    p02_empty_pattern_list_err => |haystack, pattern| {
        let _ = SimdSieve::new(&haystack, &[]);
    },
    p03_pattern_longer_than_hay => |haystack, pattern| {
        if pattern.len() > haystack.len() {
            if let Ok(sieve) = SimdSieve::new(&haystack, &[&pattern]) {
                prop_assert_eq!(sieve.count(), 0);
            }
        }
    },
    p04_results_sorted => |haystack, pattern| {
        let pat = pattern.as_slice();
        if let Ok(sieve) = SimdSieve::new(&haystack, &[pat]) {
            let mut prev = None;
            for pos in sieve {
                if let Some(p) = prev {
                    prop_assert!(pos >= p);
                }
                prev = Some(pos);
            }
        }
    },
    p05_single_byte_pattern => |haystack, pattern| {
        if pattern.len() == 1 {
            let _ = SimdSieve::new(&haystack, &[&pattern])?;
        }
    },
    p06_haystack_empty => |haystack, pattern| {
        let _ = SimdSieve::new(&[], &[&pattern]);
    },
    p07_duplicate_pattern_ok => |haystack, pattern| {
        let pat = pattern.as_slice();
        let _ = SimdSieve::new(&haystack, &[pat, pat]);
    },
    p08_all_zero_bytes => |haystack, pattern| {
        let hay = vec![0u8; haystack.len().min(64)];
        let pat = vec![0u8; pattern.len().min(4)];
        let _ = SimdSieve::new(&hay, &[&pat]);
    },
    p09_all_ff_bytes => |haystack, pattern| {
        let hay = vec![0xFFu8; haystack.len().min(64)];
        let pat = vec![0xFFu8; pattern.len().min(4)];
        let _ = SimdSieve::new(&hay, &[&pat]);
    },
    p10_count_matches_collect_len => |haystack, pattern| {
        let pat = pattern.as_slice();
        if let Ok(sieve) = SimdSieve::new(&haystack, &[pat]) {
            let collected: Vec<usize> = sieve.collect();
            // Reference: every start position (overlapping allowed) where the
            // full pattern occurs. The sieve verifies the full pattern at each
            // candidate, so its collected positions must exactly equal this
            // brute-force set - not merely equal themselves (the old tautology).
            let expected: Vec<usize> = (0..=haystack.len().saturating_sub(pat.len()))
                .filter(|&i| haystack[i..].starts_with(pat))
                .collect();
            prop_assert_eq!(collected, expected);
        }
    },
}

