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

        for (i, &p) in patterns.iter().enumerate() {
            if p.is_empty() {
                return Err(SimdSieveError::EmptyPattern { index: i });
            }
            let evaluate_len = if p.len() > 4 { 4 } else { p.len() };
            if evaluate_len > max_len {
                max_len = evaluate_len;
            }
            verify_patterns[i] = p;
        }

        let count = patterns.len();
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
            tier,
            current_mask: 0,
            next_mask_cache: 0,
            mask_base_offset: 0,
            verifier,
        })
    }

    /// Selects the best available hardware tier based on runtime feature detection.
    fn select_hardware_tier(filter_patterns: &[&[u8]], case_insensitive: bool) -> HardwareTier {
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

        HardwareTier::Scalar(Box::new(ScalarFilter::new(
            filter_patterns,
            case_insensitive,
        )))
    }
}
