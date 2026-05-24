#![no_main]

use libfuzzer_sys::fuzz_target;
use simdsieve::MultiSieve;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    // Use the first 16 bytes as prefixes (1 byte each) for testing the limit
    let mut patterns = Vec::new();
    for i in 0..16 {
        patterns.push(&data[i..i + 1]);
    }

    let haystack = &data[16..];

    // Ensure it doesn't panic and gracefully completes
    if let Ok(sieve) = MultiSieve::new(haystack, &patterns) {
        for _ in sieve.candidates() {
            // just iterate
        }
    }
});