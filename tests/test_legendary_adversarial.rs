#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Adversarial tests for simdsieve.

use simdsieve::SimdSieve;

#[test]
fn test_adversarial_empty_input() {
    let sieve = SimdSieve::new(b"", &[b"a"]).unwrap();
    assert_eq!(sieve.count(), 0);
}

#[test]
fn test_adversarial_null_bytes() {
    let haystack = b"\0\0\0\0\0\0\0\0\0\0a\0\0\0\0";
    let sieve = SimdSieve::new(haystack, &[b"a"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![10]);
}

#[test]
fn test_adversarial_all_0xff() {
    let haystack = vec![0xFF; 128];
    let sieve = SimdSieve::new(&haystack, &[&[0xFF, 0xFF]]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results.len(), 127);
}

#[test]
fn test_adversarial_huge_input() {
    // 1MB+ input
    let mut haystack = vec![0x00; 1024 * 1024 + 100];
    haystack[1024 * 1024] = b'X';
    let sieve = SimdSieve::new(&haystack, &[b"X"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![1024 * 1024]);
}

#[test]
fn test_adversarial_unicode() {
    let haystack = "🦀 rust is awesome 🚀".as_bytes();
    let sieve = SimdSieve::new(haystack, &["🚀".as_bytes()]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results.len(), 1);
    assert_eq!(&haystack[results[0]..], "🚀".as_bytes());
}

#[test]
fn test_adversarial_path_traversal() {
    let haystack = b"GET /../../../../etc/passwd HTTP/1.1";
    // Indexes: "GET /" -> 0,1,2,3,4. ".." -> 5,6. "/" -> 7.
    let sieve = SimdSieve::new(haystack, &[b"../"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![5, 8, 11, 14]);
}

#[test]
fn test_adversarial_boundary_integer_overflow() {
    // We test max_len boundary effects
    let haystack = vec![b'a'; 100];
    let large_pattern = vec![b'a'; 500]; // much larger than 100
    let sieve = SimdSieve::new(&haystack, &[&large_pattern]).unwrap();
    assert_eq!(sieve.count(), 0);
}

#[test]
fn test_adversarial_off_by_one_boundary() {
    let haystack = b"1234567890123456789012345678901X"; // 32 bytes
    let sieve = SimdSieve::new(haystack, &[b"X"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![31]);

    let haystack = b"12345678901234567890123456789012X"; // 33 bytes
    let sieve = SimdSieve::new(haystack, &[b"X"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![32]);
}

#[test]
fn test_adversarial_empty_pattern() {
    let haystack = b"hello";
    // A set containing an empty pattern returns EmptyPattern error.
    // Matching every position is never useful and catastrophic at scale.
    let result = SimdSieve::new(haystack, &[b""]);
    assert!(result.is_err(), "empty-only pattern set should error");
}
