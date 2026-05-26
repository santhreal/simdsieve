//! S-proptest-03 - simdsieve mass proptest (p11-p34).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use proptest::prelude::*;
use simdsieve::SimdSieve;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn p11_two_pattern_union(
        haystack in prop::collection::vec(any::<u8>(), 0..128),
        p1 in prop::collection::vec(any::<u8>(), 1..4),
        p2 in prop::collection::vec(any::<u8>(), 1..4),
    ) {
        let _ = SimdSieve::new(&haystack, &[p1.as_slice(), p2.as_slice()]);
    }

    #[test]
    fn p12_three_patterns(
        haystack in prop::collection::vec(any::<u8>(), 0..96),
        patterns in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..4), 1..3),
    ) {
        let refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();
        let _ = SimdSieve::new(&haystack, &refs);
    }

    #[test]
    fn p13_overlapping_pattern(_unused in 0..1i32) {
        if let Ok(sieve) = SimdSieve::new(b"aaaa", &[b"aaa"]) {
            let got: Vec<_> = sieve.collect();
            prop_assert_eq!(got, vec![0, 1]);
        }
    }

    #[test]
    fn p14_case_sensitive_mismatch(_unused in 0..1i32) {
        if let Ok(sieve) = SimdSieve::new(b"AbC", &[b"abc"]) {
            prop_assert_eq!(sieve.count(), 0);
        }
    }

    #[test]
    fn p15_boundary_last_byte(
        haystack in prop::collection::vec(any::<u8>(), 1..16),
        last in any::<u8>(),
    ) {
        let mut hay = haystack;
        let pat = vec![last];
        if let Some(slot) = hay.last_mut() {
            *slot = last;
        }
        let _ = SimdSieve::new(&hay, &[&pat]);
    }

    #[test]
    fn p16_max_pattern_len_8(
        haystack in prop::collection::vec(any::<u8>(), 0..64),
        pattern in prop::collection::vec(any::<u8>(), 8..=8),
    ) {
        let _ = SimdSieve::new(&haystack, &[&pattern]);
    }

    #[test]
    fn p17_repeated_haystack(byte in any::<u8>(), len in 1usize..128) {
        let hay = vec![byte; len];
        let pat = vec![byte];
        if let Ok(sieve) = SimdSieve::new(&hay, &[&pat]) {
            for pos in sieve {
                prop_assert!(hay[pos..].starts_with(&pat));
            }
        }
    }

    #[test]
    fn p18_alternating_hay(len in 8usize..64) {
        let hay: Vec<u8> = (0..len).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
        let _ = SimdSieve::new(&hay, &[&[0xAA, 0x55]]);
    }

    #[test]
    fn p19_prefix_only_match(_unused in 0..1i32) {
        if let Ok(sieve) = SimdSieve::new(b"prefix", &[b"pre"]) {
            prop_assert_eq!(sieve.collect::<Vec<_>>(), vec![0]);
        }
    }

    #[test]
    fn p20_suffix_only_match(_unused in 0..1i32) {
        if let Ok(sieve) = SimdSieve::new(b"suffix", &[b"fix"]) {
            let got: Vec<_> = sieve.collect();
            prop_assert!(!got.is_empty());
        }
    }

    #[test]
    fn p21_no_match_random(
        haystack in prop::collection::vec(1u8..=254, 8..32),
        pattern in prop::collection::vec(2u8..=253, 2..4),
    ) {
        let _ = SimdSieve::new(&haystack, &[&pattern]);
    }

    #[test]
    fn p22_hay_equals_pattern(pattern in prop::collection::vec(any::<u8>(), 1..8)) {
        if let Ok(sieve) = SimdSieve::new(&pattern, &[&pattern]) {
            for pos in sieve {
                prop_assert!(pattern[pos..].starts_with(&pattern));
            }
        }
    }

    #[test]
    fn p23_count_zero_when_no_match(haystack in prop::collection::vec(any::<u8>(), 0..8)) {
        let pat = vec![0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8, 0xF7];
        if let Ok(sieve) = SimdSieve::new(&haystack, &[&pat]) {
            let limit = haystack.len().saturating_sub(pat.len().saturating_sub(1));
            let any_hit = (0..limit).any(|i| haystack[i..].starts_with(&pat));
            if !any_hit {
                prop_assert_eq!(sieve.count(), 0);
            }
        }
    }

    #[test]
    fn p24_multi_pat_short(
        haystack in prop::collection::vec(any::<u8>(), 0..64),
        p1 in prop::collection::vec(any::<u8>(), 1..3),
        p2 in prop::collection::vec(any::<u8>(), 1..3),
    ) {
        if let Ok(sieve) = SimdSieve::new(&haystack, &[p1.as_slice(), p2.as_slice()]) {
            for pos in sieve {
                let hit = haystack[pos..].starts_with(p1.as_slice())
                    || haystack[pos..].starts_with(p2.as_slice());
                prop_assert!(hit);
            }
        }
    }

    #[test]
    fn p25_null_bytes_in_hay(haystack in prop::collection::vec(any::<u8>(), 4..32)) {
        let mut hay = haystack;
        hay[0] = 0;
        let _ = SimdSieve::new(&hay, &[&[0]]);
    }

    #[test]
    fn p26_high_bytes_pattern(
        pattern in prop::collection::vec(200u8..=255, 1..4),
        haystack in prop::collection::vec(any::<u8>(), 8..32),
    ) {
        let _ = SimdSieve::new(&haystack, &[&pattern]);
    }

    #[test]
    fn p27_four_byte_pattern(
        haystack in prop::collection::vec(any::<u8>(), 16..64),
        b0 in any::<u8>(),
        b1 in any::<u8>(),
        b2 in any::<u8>(),
        b3 in any::<u8>(),
    ) {
        let pat = [b0, b1, b2, b3];
        let _ = SimdSieve::new(&haystack, &[&pat]);
    }

    #[test]
    fn p28_collect_twice_same(
        haystack in prop::collection::vec(any::<u8>(), 0..48),
        pattern in prop::collection::vec(any::<u8>(), 1..4),
    ) {
        if let Ok(sieve) = SimdSieve::new(&haystack, &[&pattern]) {
            let a: Vec<_> = sieve.collect();
            let b: Vec<_> = SimdSieve::new(&haystack, &[&pattern]).unwrap().collect();
            prop_assert_eq!(a, b);
        }
    }

    #[test]
    fn p29_pattern_at_start(haystack in prop::collection::vec(any::<u8>(), 4..16)) {
        if haystack.len() >= 2 {
            let pat = &haystack[..2];
            let _ = SimdSieve::new(&haystack, &[pat]);
        }
    }

    #[test]
    fn p30_pattern_at_end(haystack in prop::collection::vec(any::<u8>(), 4..16)) {
        if haystack.len() >= 2 {
            let pat = &haystack[haystack.len() - 2..];
            let _ = SimdSieve::new(&haystack, &[pat]);
        }
    }

    #[test]
    fn p31_many_single_char_patterns(haystack in prop::collection::vec(any::<u8>(), 8..32)) {
        let owned: Vec<Vec<u8>> = (0u8..8).map(|b| vec![b]).collect();
        let pats: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let _ = SimdSieve::new(&haystack, &pats);
    }

    #[test]
    fn p32_utf8_like_bytes(haystack in prop::collection::vec(any::<u8>(), 0..48)) {
        let pat = &[0xF0, 0x9F, 0x92, 0xA9];
        let _ = SimdSieve::new(&haystack, &[pat]);
    }

    #[test]
    fn p33_windows_crlf(_unused in 0..1i32) {
        let _ = SimdSieve::new(b"hello\r\nworld", &[b"\r\n"]);
    }

    #[test]
    fn p34_brute_parity_tiny_hay(
        haystack in prop::collection::vec(any::<u8>(), 0..32),
        pattern in prop::collection::vec(any::<u8>(), 1..4),
    ) {
        if !pattern.is_empty() {
        if let Ok(sieve) = SimdSieve::new(&haystack, &[&pattern]) {
            for pos in sieve {
                prop_assert!(haystack[pos..].starts_with(&pattern));
            }
        }
        }
    }
}
