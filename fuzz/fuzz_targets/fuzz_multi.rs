//! Fuzz target: up to 8 random patterns.
//!
//! This fuzzer verifies that simdsieve handles multiple patterns correctly
//! without panicking, including the edge case of exactly 8 patterns.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdsieve::SimdSieve;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // First byte: number of patterns (1-8)
    let num_patterns = (data[0] as usize % 8).max(1);

    // Second byte: base pattern length (1-16)
    let base_pat_len = (data[1] as usize % 16).max(1);

    // Need enough data for all patterns
    let min_needed = 2 + num_patterns * base_pat_len;
    if data.len() < min_needed {
        return;
    }

    // Extract patterns
    let mut patterns: Vec<&[u8]> = Vec::with_capacity(num_patterns);
    let mut offset = 2;

    for i in 0..num_patterns {
        // Vary pattern lengths slightly for diversity
        let pat_len = if i % 2 == 0 {
            base_pat_len
        } else {
            (base_pat_len + 1).min(16)
        };

        if offset + pat_len > data.len() {
            return;
        }

        patterns.push(&data[offset..offset + pat_len]);
        offset += pat_len;
    }

    // Remaining data is haystack
    let haystack = &data[offset..];

    // Test exact match with multiple patterns
    if let Ok(sieve) = SimdSieve::new(haystack, &patterns) {
        for pos in sieve {
            // Verify position is valid
            assert!(
                pos < haystack.len(),
                "Invalid position {} for haystack len {}",
                pos,
                haystack.len()
            );

            // Verify at least one pattern matches at this position
            let mut found_match = false;
            for &pat in &patterns {
                if pos + pat.len() <= haystack.len() {
                    if &haystack[pos..pos + pat.len()] == pat {
                        found_match = true;
                        break;
                    }
                }
            }
            assert!(
                found_match,
                "False positive at position {} - no pattern matches",
                pos
            );
        }
    }

    // Test case-insensitive match
    if let Ok(sieve) = SimdSieve::new_case_insensitive(haystack, &patterns) {
        for pos in sieve {
            assert!(
                pos < haystack.len(),
                "Invalid CI position {} for haystack len {}",
                pos,
                haystack.len()
            );

            // Verify at least one pattern matches case-insensitively
            let mut found_match = false;
            for &pat in &patterns {
                if pos + pat.len() <= haystack.len() {
                    let candidate = &haystack[pos..pos + pat.len()];
                    let matches_ci = candidate
                        .iter()
                        .zip(pat.iter())
                        .all(|(&c, &p)| c.to_ascii_lowercase() == p.to_ascii_lowercase());
                    if matches_ci {
                        found_match = true;
                        break;
                    }
                }
            }
            assert!(
                found_match,
                "CI false positive at position {} - no pattern matches",
                pos
            );
        }
    }

    // Test with exactly 8 patterns (boundary case)
    if num_patterns == 8 {
        // This should succeed
        let result = SimdSieve::new(haystack, &patterns);
        assert!(
            result.is_ok(),
            "8 patterns should be accepted but got error"
        );
    }
});
