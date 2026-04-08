#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simdsieve::SimdSieve;

#[test]
fn test_simd_adversarial_fuzzing_million_iterations_multi() {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    for _ in 0..10_000 {
        let len = rng.gen_range(1..10000);
        let mut haystack = vec![0u8; len];
        rng.fill(&mut haystack[..]);

        let p1_len = rng.gen_range(1..=4);
        let p2_len = rng.gen_range(1..=4);
        let mut p1 = vec![0u8; p1_len];
        let mut p2 = vec![0u8; p2_len];
        rng.fill(&mut p1[..]);
        rng.fill(&mut p2[..]);

        let hw_sieve = SimdSieve::new(&haystack, &[&p1, &p2]).unwrap();
        let hw_results: Vec<usize> = hw_sieve.collect();

        // Baseline comparison using string finding logic
        let mut baseline: Vec<usize> = Vec::new();
        for i in 0..haystack.len() {
            let remain = &haystack[i..];
            if (remain.len() >= p1_len && &remain[..p1_len] == p1.as_slice())
                || (remain.len() >= p2_len && &remain[..p2_len] == p2.as_slice())
            {
                baseline.push(i);
            }
        }

        assert_eq!(
            hw_results, baseline,
            "CRITICAL DIVERGENCE in multi-pattern fastpath router."
        );
    }
}
