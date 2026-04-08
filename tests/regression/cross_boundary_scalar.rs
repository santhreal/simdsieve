#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_cross_boundary_scalar_miss() {
    // We explicitly test the Scalar fallback tier for a bug where it drops matches
    // that straddle the 64-byte logical block boundary.
    // If we have a target of length 3 ("CVE"), the offset 62 and 63
    // MUST correctly yield a match using the `tail_req` buffer.

    // Create a 100 byte haystack
    let mut haystack = vec![0u8; 100];

    // Insert "CVE" right at the physical boundary (starts at 63, straddles 63, 64, 65)
    haystack[63] = b'C';
    haystack[64] = b'V';
    haystack[65] = b'E';

    let sieve = SimdSieve::new(&haystack, &[b"CVE"]).unwrap();
    let results: Vec<usize> = sieve.collect();

    // If the Scalar fallback is using `0..=64 - p.len`, it will STOP evaluating
    // at index 61. It will silently drop the match at 63 because it doesn't use the tail buffer.
    assert_eq!(
        results,
        vec![63],
        "SCALAR FAIL: Iteration silently dropped cross-boundary match at index 63"
    );
}
