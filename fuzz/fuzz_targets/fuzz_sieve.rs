//! Fuzz target: random haystack + single pattern.
//!
//! This fuzzer verifies that simdsieve never panics on arbitrary inputs
//! and that all yielded positions are valid matches.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdsieve::SimdSieve;

fuzz_target!(|data: &[u8]| {
    // Need at least 1 byte for pattern + haystack
    if data.is_empty() {
        return;
    }

    // First byte is pattern length (0-255, but we cap at 32 for practicality)
    let pat_len = (data[0] as usize % 32).max(1).min(data.len() - 1);
    
    if data.len() <= pat_len {
        return;
    }

    let pattern = &data[0..pat_len];
    let haystack = &data[pat_len..];

    // Try exact match
    if let Ok(sieve) = SimdSieve::new(haystack, &[pattern]) {
        for pos in sieve {
            // Verify the position is valid
            assert!(
                pos + pattern.len() <= haystack.len(),
                "Invalid position {} for haystack len {} and pattern len {}",
                pos,
                haystack.len(),
                pattern.len()
            );

            // Verify the match is correct
            assert_eq!(
                &haystack[pos..pos + pattern.len()],
                pattern,
                "False positive at position {}",
                pos
            );
        }
    }

    // Try case-insensitive match
    if let Ok(sieve) = SimdSieve::new_case_insensitive(haystack, &[pattern]) {
        for pos in sieve {
            assert!(
                pos + pattern.len() <= haystack.len(),
                "Invalid CI position {} for haystack len {} and pattern len {}",
                pos,
                haystack.len(),
                pattern.len()
            );

            // Verify case-insensitive match
            let candidate = &haystack[pos..pos + pattern.len()];
            for (&c, &p) in candidate.iter().zip(pattern.iter()) {
                assert_eq!(
                    c.to_ascii_lowercase(),
                    p.to_ascii_lowercase(),
                    "CI false positive at position {}: {:?} vs {:?}",
                    pos,
                    candidate,
                    pattern
                );
            }
        }
    }
});
