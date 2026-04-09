#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_simd_scalar_mask_parity() {
    for byte in 0u8..=255 {
        let haystack = vec![byte; 256];
        let pattern = [byte];

        let result_sieve: Vec<_> = SimdSieve::new(&haystack, &[&pattern]).unwrap().collect();
        let result_naive: Vec<_> = (0..256).collect();

        assert_eq!(result_sieve, result_naive, "Parity failed for byte {byte}");
    }
}

#[test]
fn estimate_match_count_greater_than_4kb() {
    let mut haystack = vec![0u8; 8192];
    haystack[0..4096].fill(b'a');
    haystack[4096..8192].fill(b'a'); // Fill rest as well
    let pattern = [b'a'];

    // The density score only evaluates up to 4096 bytes.
    let count = SimdSieve::estimate_match_count(&haystack, &[&pattern], false);
    // Since pattern is 1 byte, it checks up to 4096 bytes.
    assert!(
        count <= 4096,
        "Score density should not exceed 4096, got {count}"
    );
}

#[test]
fn prefetch_out_of_bounds_tiny_haystack() {
    let haystack = vec![b'x'; 100];
    let pattern = [b'y'];

    // Should not panic or out of bounds when prefetching
    let results: Vec<_> = SimdSieve::new(&haystack, &[&pattern]).unwrap().collect();
    assert!(results.is_empty());
}

#[test]
fn empty_pattern_list_errors() {
    let haystack = vec![b'a'; 64];
    assert!(SimdSieve::new(&haystack, &[]).is_err());
}

#[test]
fn single_byte_patterns() {
    let haystack = b"hello world";
    let results: Vec<_> = SimdSieve::new(haystack, &[b"o"]).unwrap().collect();
    assert_eq!(results, vec![4, 7]);
}

#[test]
fn patterns_longer_than_haystack() {
    let haystack = b"short";
    let pattern = b"very long pattern";
    let results: Vec<_> = SimdSieve::new(haystack, &[pattern]).unwrap().collect();
    assert!(results.is_empty());
}
