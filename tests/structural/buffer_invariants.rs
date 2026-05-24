#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_unaligned_tail_sweeping() {
    let mut haystack = vec![0u8; 109];
    haystack[1] = b'Y';
    haystack[65] = b'Y';
    haystack[108] = b'Y';

    let sieve = SimdSieve::new(&haystack, &[b"Y"]).unwrap();
    let results: Vec<usize> = sieve.collect();

    assert_eq!(results, vec![1, 65, 108]);
}

#[test]
fn test_unaligned_tail_sweeping_multibyte() {
    let mut haystack = vec![0u8; 109];
    haystack[1] = b'Y';
    haystack[2] = b'Z';
    haystack[65] = b'Y';
    haystack[66] = b'Z';
    haystack[107] = b'Y';
    haystack[108] = b'Z';

    let sieve = SimdSieve::new(&haystack, &[b"YZ"]).unwrap();
    let results: Vec<usize> = sieve.collect();

    assert_eq!(results, vec![1, 65, 107]);
}
