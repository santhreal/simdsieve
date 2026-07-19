//! Core `SimdSieve` streaming iterator.
//!
//! This module contains the main public API for the simdsieve engine.
//! `SimdSieve` wraps hardware-specific filter backends behind a standard
//! `Iterator<Item = usize>` interface, yielding byte offsets into the
//! haystack where pattern matches begin.

use crate::sieve::dispatch::HardwareTier;

pub(crate) mod collector;
pub(crate) mod compiler;
pub(crate) mod dispatch;

/// A streaming hardware-accelerated iterator that yields byte offsets
/// where the haystack matches one of the supplied patterns.
pub struct SimdSieve<'a> {
    /// Reference to the haystack being searched.
    pub(crate) haystack: &'a [u8],
    /// Current byte offset in the haystack.
    pub(crate) offset: usize,
    /// Full patterns for verification (up to 16).
    /// Unused slots contain empty slices.
    pub(crate) verification_patterns: [&'a [u8]; 16],
    /// Number of valid patterns in `verification_patterns`.
    pub(crate) pattern_count: usize,
    /// Maximum prefix length across all patterns (1–4).
    pub(crate) max_len: usize,
    /// Selected hardware backend.
    pub(crate) tier: HardwareTier,
    /// Bitmask of candidate positions in current half-block.
    /// Bit `i` set means position `mask_base_offset + i` is a candidate.
    pub(crate) current_mask: u64,
    /// Cached second-half mask from dual-pump processing.
    /// Only used by AVX-512 and AVX2 backends.
    pub(crate) next_mask_cache: u64,
    /// Base offset for the current mask.
    pub(crate) mask_base_offset: usize,
    /// Function pointer for verification (exact or case-insensitive).
    pub(crate) verifier: fn(&[u8], &[u8]) -> bool,
}

impl<'a> SimdSieve<'a> {
    /// Counts total prefix-hit positions without full verification.
    ///
    /// This is faster than collecting the iterator because it skips
    /// full-length verification: it counts raw SIMD bitmask popcount.
    /// Use this for density estimation when deciding whether to use
    /// a more expensive verification algorithm.
    ///
    /// # Behavior with >16 patterns
    ///
    /// If more than 16 patterns are provided, this function delegates to
    /// [`crate::MultiSieve`] and counts verified match positions in the first
    /// 4 KB of the haystack. This ensures the function remains infallible
    /// while still providing a useful density signal for large pattern sets.
    ///
    /// # 4 KB window cap
    ///
    /// To keep the call cheap on huge inputs, density estimation is
    /// performed only on the first 4 096 bytes of `haystack`. If your
    /// workload has density that changes dramatically after the header,
    /// scan a representative slice manually instead of relying on this
    /// coarse estimate.
    ///
    /// # Notes
    ///
    /// This counts every position where the first 1–4 bytes of any pattern
    /// match, even if the full pattern is longer than the remaining haystack.
    /// For density estimation on large inputs this edge effect is negligible.
    #[must_use]
    pub fn estimate_match_count(
        haystack: &'a [u8],
        patterns: &[&'a [u8]],
        case_insensitive: bool,
    ) -> u64 {
        if patterns.is_empty() {
            return 0;
        }

        // Density estimation only needs the first 4KB of haystack.
        // This is sufficient for the prefilter decision and avoids
        // scanning huge inputs for a coarse density score.
        let haystack = &haystack[..haystack.len().min(4096)];

        // For pattern sets larger than a single sieve can hold, delegate to
        // MultiSieve and count verified matches instead of raw prefix hits.
        if patterns.len() > crate::MAX_PATTERNS {
            let multi = if case_insensitive {
                crate::MultiSieve::new_case_insensitive(haystack, patterns)
            } else {
                crate::MultiSieve::new(haystack, patterns)
            };
            return match multi {
                Ok(m) => m.candidates().count() as u64,
                Err(error) => fail_safe_density(haystack, &error),
            };
        }

        let sieve_result = if case_insensitive {
            Self::new_case_insensitive(haystack, patterns)
        } else {
            Self::new(haystack, patterns)
        };

        let mut sieve = match sieve_result {
            Ok(sieve) => sieve,
            Err(error) => return fail_safe_density(haystack, &error),
        };

        let mut global_popcnt: u64 = 0;

        while sieve.fetch_next_chunk() {
            if sieve.current_mask != 0 {
                global_popcnt += u64::from(sieve.current_mask.count_ones());
                sieve.current_mask = 0;
            }
            if sieve.next_mask_cache != 0 {
                global_popcnt += u64::from(sieve.next_mask_cache.count_ones());
                sieve.next_mask_cache = 0;
            }
        }

        while sieve.offset + sieve.max_len <= haystack.len() {
            let current_idx = sieve.offset;
            sieve.offset += 1;

            for p_idx in 0..sieve.pattern_count {
                let vp = sieve.verification_patterns[p_idx];
                let prefix_len = vp.len().min(4);
                if (sieve.verifier)(
                    &haystack[current_idx..current_idx + prefix_len],
                    &vp[..prefix_len],
                ) {
                    global_popcnt += 1;
                    break;
                }
            }
        }

        // `<` (not `<=`): at current_idx == haystack.len() the guard
        // `current_idx + prefix_len <= haystack.len()` is always false for the
        // non-empty patterns this sieve accepts (prefix_len >= 1), so the final
        // iteration can never increment the count.
        while sieve.offset < haystack.len() {
            let current_idx = sieve.offset;
            sieve.offset += 1;

            for p_idx in 0..sieve.pattern_count {
                let vp = sieve.verification_patterns[p_idx];
                let prefix_len = vp.len().min(4);
                if current_idx + prefix_len <= haystack.len()
                    && (sieve.verifier)(
                        &haystack[current_idx..current_idx + prefix_len],
                        &vp[..prefix_len],
                    )
                {
                    global_popcnt += 1;
                    break;
                }
            }
        }

        global_popcnt
    }
}

/// Fail-safe density returned when the sieve/`MultiSieve` cannot be built for an
/// estimate.
///
/// A construction failure (an oversized or otherwise invalid pattern set) must
/// NOT be reported as zero density: `estimate_match_count` is a prefilter, and a
/// zero would let a caller decide to SKIP the real scan, silently losing recall
/// (Law-10). Instead we surface the error loudly and return the maximum possible
/// density (one candidate per scanned byte) so the caller always errs toward
/// scanning. Loud + recall-preserving, never a silent zero.
fn fail_safe_density(haystack: &[u8], error: &crate::error::SimdSieveError) -> u64 {
    eprintln!(
        "simdsieve::estimate_match_count: sieve construction failed ({error}); \
         returning max density to force a full scan (fail-safe, not zero)."
    );
    haystack.len() as u64
}

#[cfg(test)]
mod tests {
    use super::SimdSieve;
    use super::dispatch::HardwareTier;

    #[test]
    fn case_insensitive_iterator_keeps_boundary_matches() {
        let mut haystack = vec![0u8; 1024];
        haystack[63] = b'Z';
        haystack[150] = b'z';
        haystack[1023] = b'Z';

        let sieve = SimdSieve::new_case_insensitive(&haystack, &[b"Z"]).unwrap();
        let tier = match &sieve.tier {
            #[cfg(target_arch = "x86_64")]
            HardwareTier::Avx512(_) => "avx512",
            #[cfg(target_arch = "x86_64")]
            HardwareTier::Avx2(_) => "avx2",
            #[cfg(target_arch = "aarch64")]
            HardwareTier::Neon(_) => "neon",
            HardwareTier::Scalar(_) => "scalar",
        };
        let results: Vec<usize> = sieve.collect();
        eprintln!("tier={tier} results={results:?}");
        assert_eq!(results, vec![63, 150, 1023]);
    }
}

#[cfg(test)]
mod sieve_unit_tests {
    use super::*;

    #[test]
    fn empty_haystack_no_matches() {
        let patterns: &[&[u8]] = &[b"test"];
        let matches: Vec<_> = SimdSieve::new(b"", patterns).unwrap().collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn single_pattern_found() {
        let patterns: &[&[u8]] = &[b"GET"];
        let matches: Vec<_> = SimdSieve::new(b"GET /index.html", patterns)
            .unwrap()
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], 0);
    }

    #[test]
    fn multiple_patterns_found() {
        let patterns: &[&[u8]] = &[b"GET", b"POST"];
        let haystack = b"GET /a HTTP/1.1\r\nPOST /b HTTP/1.1\r\n";
        let matches: Vec<_> = SimdSieve::new(haystack, patterns).unwrap().collect();
        assert!(matches.len() >= 2, "both patterns should match");
    }

    #[test]
    fn pattern_at_end_of_haystack() {
        let patterns: &[&[u8]] = &[b"end"];
        let matches: Vec<_> = SimdSieve::new(b"the end", patterns).unwrap().collect();
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn no_match_returns_empty() {
        let patterns: &[&[u8]] = &[b"xyz"];
        let matches: Vec<_> = SimdSieve::new(b"abcdefghijklmnop", patterns)
            .unwrap()
            .collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn estimate_match_count_basic() {
        let patterns: &[&[u8]] = &[b"abc"];
        // Use a haystack large enough to trigger block processing in all backends.
        let haystack = vec![b'x'; 128];
        let count = SimdSieve::estimate_match_count(&haystack, patterns, false);
        // estimate_match_count is infallible; just verify it returns a finite value.
        assert!(
            count <= haystack.len() as u64,
            "estimate should not exceed haystack length"
        );
    }

    #[test]
    fn estimate_match_count_fails_safe_on_invalid_pattern_set() {
        // Regression for sieve/mod.rs:102 (Law-10): a pattern set that fails to
        // build a sieve (here an empty pattern makes SimdSieve::new return
        // EmptyPattern) must NOT be reported as zero density - a zero could let a
        // prefilter skip the real scan and lose recall. It must return the
        // fail-safe MAX density instead.
        let haystack = vec![b'x'; 100];
        let patterns: [&[u8]; 2] = [b"x", b""];
        let count = SimdSieve::estimate_match_count(&haystack, &patterns, false);
        assert_eq!(
            count, 100,
            "invalid pattern set must return max density (haystack len), not 0"
        );
    }

    #[test]
    fn estimate_match_count_fails_safe_on_invalid_many_pattern_set() {
        // Same fail-safe contract on the MultiSieve delegation path (> MAX_PATTERNS).
        let haystack = vec![b'x'; 100];
        let mut patterns: Vec<&[u8]> = (0..=crate::MAX_PATTERNS)
            .map(|_| b"ab".as_slice())
            .collect();
        patterns.push(b""); // empty pattern -> construction error
        let count = SimdSieve::estimate_match_count(&haystack, &patterns, false);
        assert_eq!(
            count, 100,
            "invalid many-pattern set must return max density, not 0"
        );
    }

    #[test]
    fn estimate_match_count_with_many_patterns() {
        // Create 20 distinct single-byte patterns.
        let patterns: Vec<Vec<u8>> = (0..20).map(|i| vec![b'a' + i]).collect();
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();

        let mut haystack = vec![b'x'; 128];
        haystack[10] = b'a';
        haystack[20] = b'b';
        haystack[30] = b'c';

        let count = SimdSieve::estimate_match_count(&haystack, &pattern_refs, false);
        // MultiSieve delegation should still find the three matches.
        assert!(
            count >= 3,
            "estimate_match_count should delegate to MultiSieve and find matches"
        );
    }

    #[test]
    fn binary_data_no_crash() {
        let patterns: &[&[u8]] = &[b"\x00\x01\x02"];
        let data = vec![0u8; 1024];
        let _matches: Vec<_> = SimdSieve::new(&data, patterns).unwrap().collect();
    }

    #[test]
    fn pattern_longer_than_haystack() {
        let patterns: &[&[u8]] = &[b"toolongpattern"];
        let matches: Vec<_> = SimdSieve::new(b"short", patterns).unwrap().collect();
        assert!(matches.is_empty());
    }
}
