#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use simdsieve::SimdSieve;

#[allow(clippy::if_same_then_else)]
#[test]
fn test_simd_scalar_absolute_parity_multi_pattern() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut haystack = vec![0u8; 100_000];
    rng.fill_bytes(&mut haystack);

    let t1 = [haystack[500], haystack[501], haystack[502]];
    let t2 = [haystack[1500], haystack[1501]];

    let hw_sieve = SimdSieve::new(&haystack, &[&t1, &t2]).unwrap();
    let hw_results: Vec<usize> = hw_sieve.collect();

    let mut expected = Vec::new();
    for i in 0..haystack.len() {
        if i + 3 <= haystack.len() && haystack[i..i + 3] == t1 {
            expected.push(i);
        } else if i + 2 <= haystack.len() && haystack[i..i + 2] == t2 {
            expected.push(i);
        }
    }

    expected.sort_unstable();
    expected.dedup();

    assert_eq!(
        hw_results, expected,
        "SIMD sieve output must match brute-force linear scan"
    );
}
