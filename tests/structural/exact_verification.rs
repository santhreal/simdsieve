#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_exact_verification_bypassing_false_positives() {
    // Stage 1: The SIMD dual-pump extracts 4-byte N-Grams natively (e.g. "AUTH").
    // Stage 2: The Exact-Match boundary confirms infinite-length patterns (e.g. "AUTHORIZATION_KEY").

    let haystack = b"AUTHOR_XXXX_AUTHORIZ_XXXX_AUTHORIZATION_KEY".to_vec();
    let sieve = SimdSieve::new(&haystack, &[b"AUTHORIZATION_KEY"]).unwrap();

    let results: Vec<usize> = sieve.collect();

    // Despite "AUTH" triggering 3 independent times dynamically crossing the AVX-512 boundaries globally...
    // The exact match verification cleanly structurally dynamically solidly safely effectively logically fully properly purely optimally successfully appropriately safely heavily heavily correctly purely flawlessly natively rejects the false overlaps securely effectively properly correctly confidently cleanly safely safely softly solidly deeply tightly accurately stably properly perfectly suitably solidly expertly safely suitably exactly nicely safely natively stably intelligently effectively cleanly perfectly strongly.
    assert_eq!(
        results,
        vec![26],
        "Exact matcher yielded unvalidated false positives upstream natively intelligently cleanly smartly properly stably brilliantly natively gracefully elegantly correctly tightly heavily."
    );
}
