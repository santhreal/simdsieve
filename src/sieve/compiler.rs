//! Construction and pattern compilation logic.

use crate::SimdSieve;
#[cfg(target_arch = "x86_64")]
use crate::avx2::Avx2Filter;
#[cfg(target_arch = "x86_64")]
use crate::avx512::Avx512Filter;
use crate::error::{Result, SimdSieveError};
use crate::fold::{verify_case_insensitive, verify_exact};
#[cfg(target_arch = "aarch64")]
use crate::neon::NeonFilter;
use crate::scalar::ScalarFilter;
use crate::sieve::dispatch::HardwareTier;

impl<'a> SimdSieve<'a> {
    /// Creates a new exact-match sieve.
    #[allow(clippy::missing_errors_doc)]
    pub fn new(haystack: &'a [u8], patterns: &[&'a [u8]]) -> Result<Self> {
        Self::build(haystack, patterns, false)
    }

    /// Creates a case-insensitive sieve (ASCII `a`–`z` only).
    #[allow(clippy::missing_errors_doc)]
    pub fn new_case_insensitive(haystack: &'a [u8], patterns: &[&'a [u8]]) -> Result<Self> {
        Self::build(haystack, patterns, true)
    }

    /// Common construction logic for both case-sensitive and case-insensitive modes.
    fn build(haystack: &'a [u8], patterns: &[&'a [u8]], case_insensitive: bool) -> Result<Self> {
        if patterns.is_empty() {
            return Err(SimdSieveError::EmptyPatternSet);
        }
        if patterns.len() > crate::MAX_PATTERNS {
            return Err(SimdSieveError::PatternLimitExceeded(patterns.len()));
        }

        let mut max_len = 0;
        let mut verify_patterns = [&b""[..]; 16];
        let mut count = 0;

        for (i, &p) in patterns.iter().enumerate() {
            if p.is_empty() {
                return Err(SimdSieveError::EmptyPattern { index: i });
            }
            // Deduplicate: a repeated pattern finds exactly the same positions,
            // so verifying it a second time is pure redundant work (extra vector
            // loads + comparisons per candidate). count is bounded by 16, so the
            // linear membership check is negligible.
            if verify_patterns[..count].contains(&p) {
                continue;
            }
            let evaluate_len = if p.len() > 4 { 4 } else { p.len() };
            if evaluate_len > max_len {
                max_len = evaluate_len;
            }
            verify_patterns[count] = p;
            count += 1;
        }
        let filter_patterns = &verify_patterns[..count];
        let verifier = if case_insensitive {
            verify_case_insensitive
        } else {
            verify_exact
        };

        let tier = Self::select_hardware_tier(filter_patterns, case_insensitive);

        Ok(Self {
            haystack,
            offset: 0,
            verification_patterns: verify_patterns,
            pattern_count: count,
            max_len,
            tier: crate::sieve::dispatch::SieveTier::Owned(tier),
            current_mask: 0,
            next_mask_cache: 0,
            mask_base_offset: 0,
            verifier,
        })
    }

    /// Selects the best available hardware tier based on runtime feature detection.
    pub(crate) fn select_hardware_tier(filter_patterns: &[&[u8]], case_insensitive: bool) -> HardwareTier {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return HardwareTier::Avx512(Box::new(unsafe {
                    Avx512Filter::new(filter_patterns, case_insensitive)
                }));
            }
            if std::is_x86_feature_detected!("avx2") {
                return HardwareTier::Avx2(Box::new(unsafe {
                    Avx2Filter::new(filter_patterns, case_insensitive)
                }));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return HardwareTier::Neon(Box::new(unsafe {
                NeonFilter::new(filter_patterns, case_insensitive)
            }));
        }

        // unreachable on aarch64 (the return above always fires), but
        // this is the scalar fallback on every other target. Suppress
        // the unreachable-code lint that only fires on aarch64.
        #[allow(unreachable_code)]
        HardwareTier::Scalar(Box::new(ScalarFilter::new(
            filter_patterns,
            case_insensitive,
        )))
    }
}
/// A compiled pattern sieve ready to scan multiple haystacks efficiently.
///
/// `CompiledSieve` precomputes SIMD vector registers, deduplicates patterns,
/// and selects the optimal hardware backend once during construction.
/// Scanning a haystack via [`CompiledSieve::scan`] avoids all filter compilation
/// and memory allocation.
#[derive(Debug)]
pub struct CompiledSieve {
    /// Owned pattern bytes stored for exact verification.
    patterns: Vec<Vec<u8>>,
    /// Number of deduplicated patterns.
    pattern_count: usize,
    /// Maximum prefix length across all patterns (1–4).
    max_len: usize,
    /// Hardware backend compiled once.
    tier: HardwareTier,
    /// Exact or case-insensitive verifier function.
    verifier: fn(&[u8], &[u8]) -> bool,
}

impl CompiledSieve {
    /// Compiles an exact-match sieve for the provided patterns (up to 16).
    ///
    /// # Errors
    ///
    /// Returns [`SimdSieveError::EmptyPatternSet`] if `patterns` is empty,
    /// [`SimdSieveError::PatternLimitExceeded`] if `patterns.len() > 16`,
    /// or [`SimdSieveError::EmptyPattern`] if any pattern is empty.
    pub fn new(patterns: &[&[u8]]) -> Result<Self> {
        Self::build(patterns, false)
    }

    /// Compiles a case-insensitive sieve for the provided patterns (up to 16).
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`CompiledSieve::new`].
    pub fn new_case_insensitive(patterns: &[&[u8]]) -> Result<Self> {
        Self::build(patterns, true)
    }

    fn build(patterns: &[&[u8]], case_insensitive: bool) -> Result<Self> {
        if patterns.is_empty() {
            return Err(SimdSieveError::EmptyPatternSet);
        }
        if patterns.len() > crate::MAX_PATTERNS {
            return Err(SimdSieveError::PatternLimitExceeded(patterns.len()));
        }

        let mut owned_patterns: Vec<Vec<u8>> = Vec::with_capacity(patterns.len().min(16));
        let mut max_len = 0;

        for (i, &p) in patterns.iter().enumerate() {
            if p.is_empty() {
                return Err(SimdSieveError::EmptyPattern { index: i });
            }
            if owned_patterns.iter().any(|existing| existing.as_slice() == p) {
                continue;
            }
            let eval_len = if p.len() > 4 { 4 } else { p.len() };
            if eval_len > max_len {
                max_len = eval_len;
            }
            owned_patterns.push(p.to_vec());
        }

        let filter_patterns: Vec<&[u8]> = owned_patterns.iter().map(Vec::as_slice).collect();
        let verifier = if case_insensitive {
            verify_case_insensitive
        } else {
            verify_exact
        };

        let tier = SimdSieve::select_hardware_tier(&filter_patterns, case_insensitive);

        Ok(Self {
            pattern_count: owned_patterns.len(),
            patterns: owned_patterns,
            max_len,
            tier,
            verifier,
        })
    }

    /// Scans `haystack` yielding candidate match offsets.
    ///
    /// This method rebinds the haystack to the compiled filter state with zero
    /// memory allocation or SIMD filter rebuild.
    pub fn scan<'a>(&'a self, haystack: &'a [u8]) -> SimdSieve<'a> {
        let mut verify_patterns = [&b""[..]; 16];
        for (i, p) in self.patterns.iter().enumerate() {
            verify_patterns[i] = p.as_slice();
        }

        SimdSieve {
            haystack,
            offset: 0,
            verification_patterns: verify_patterns,
            pattern_count: self.pattern_count,
            max_len: self.max_len,
            tier: crate::sieve::dispatch::SieveTier::Borrowed(&self.tier),
            current_mask: 0,
            next_mask_cache: 0,
            mask_base_offset: 0,
            verifier: self.verifier,
        }
    }

    /// Estimates total prefix hits in the first 4 KB of `haystack`.
    #[must_use]
    pub fn estimate_match_count(&self, haystack: &[u8]) -> u64 {
        if self.pattern_count == 0 {
            return 0;
        }

        let haystack = &haystack[..haystack.len().min(4096)];
        let mut sieve = self.scan(haystack);
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

    /// Returns the number of unique patterns in this compiled sieve.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.pattern_count
    }

    /// Returns the unique patterns compiled in this sieve.
    #[must_use]
    pub fn patterns(&self) -> &[Vec<u8>] {
        &self.patterns
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_sieve_exact_parity_with_simdsieve_new() {
        let haystack = b"alpha beta gamma GET /admin HTTP/1.1 Host: example\r\nPOST /api/v1\r\n";
        let patterns: &[&[u8]] = &[b"GET", b"/admin", b"POST", b"alpha"];

        let compiled = CompiledSieve::new(patterns).expect("compilation failed");
        assert_eq!(compiled.pattern_count(), 4);

        let direct_matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
        let compiled_matches: Vec<usize> = compiled.scan(haystack).collect();

        assert_eq!(compiled_matches, direct_matches);
    }

    #[test]
    fn test_compiled_sieve_zero_rebuild_pointer_identity() {
        let haystack = b"quick brown fox jumps over the lazy dog";
        let patterns: &[&[u8]] = &[b"quick", b"fox", b"dog"];

        let compiled = CompiledSieve::new(patterns).unwrap();

        for _ in 0..10 {
            let sieve = compiled.scan(haystack);
            let tier_ref: &HardwareTier = &sieve.tier;
            let compiled_tier_ref: &HardwareTier = &compiled.tier;
            assert!(
                std::ptr::eq(tier_ref, compiled_tier_ref),
                "scan must borrow the compiled hardware tier directly without rebuilding"
            );
        }
    }

    #[test]
    fn test_compiled_sieve_case_insensitive_parity() {
        let haystack = b"GeT /AdMiN HTTP/1.1\r\n";
        let patterns: &[&[u8]] = &[b"get", b"/admin"];

        let compiled = CompiledSieve::new_case_insensitive(patterns).unwrap();
        let direct_matches: Vec<usize> = SimdSieve::new_case_insensitive(haystack, patterns)
            .unwrap()
            .collect();
        let compiled_matches: Vec<usize> = compiled.scan(haystack).collect();

        assert_eq!(compiled_matches, direct_matches);
    }

    #[test]
    fn test_compiled_sieve_repeated_scans_distinct_haystacks() {
        let compiled = CompiledSieve::new(&[b"foo", b"bar"]).unwrap();

        let haystacks = [
            b"foo bar baz".as_slice(),
            b"no match here".as_slice(),
            b"bar foo bar".as_slice(),
        ];

        let expected_results = [
            vec![0, 4],
            vec![],
            vec![0, 4, 8],
        ];

        for (haystack, expected) in haystacks.iter().zip(expected_results.iter()) {
            let matches: Vec<usize> = compiled.scan(haystack).collect();
            assert_eq!(&matches, expected);
        }
    }

    #[test]
    fn test_compiled_sieve_deduplication() {
        let patterns: &[&[u8]] = &[b"abc", b"def", b"abc", b"def"];
        let compiled = CompiledSieve::new(patterns).unwrap();
        assert_eq!(compiled.pattern_count(), 2);
    }

    #[test]
    fn test_compiled_sieve_estimate_match_count() {
        let haystack = b"GET /index.html HTTP/1.1\r\nHost: test\r\n";
        let patterns: &[&[u8]] = &[b"GET", b"HTTP"];
        let compiled = CompiledSieve::new(patterns).unwrap();

        let direct_estimate = SimdSieve::estimate_match_count(haystack, patterns, false);
        let compiled_estimate = compiled.estimate_match_count(haystack);

        assert_eq!(compiled_estimate, direct_estimate);
    }
}
