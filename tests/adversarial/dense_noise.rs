#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;

#[test]
fn test_simdsieve_chunk_boundaries_case_sensitive() {
    // First: verify case-sensitive finds exact matches at boundaries
    let mut haystack = vec![0u8; 1024];
    haystack[63] = b'Z';
    haystack[150] = b'Z';
    haystack[1023] = b'Z';

    let sieve = SimdSieve::new(&haystack, &[b"Z"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![63, 150, 1023], "case-sensitive boundary test");
}

#[test]
fn test_simdsieve_chunk_boundaries() {
    let mut haystack = vec![0u8; 1024];
    haystack[63] = b'Z';
    haystack[150] = b'z';
    haystack[1023] = b'Z';

    let sieve = SimdSieve::new_case_insensitive(&haystack, &[b"Z"]).unwrap();
    let results: Vec<usize> = sieve.collect();

    assert_eq!(results, vec![63, 150, 1023]);
}

#[test]
fn test_simdsieve_dense_noise() {
    let haystack: Vec<u8> = (0..5000)
        .map(|i| {
            if i % 256 == b'M' as usize || i % 256 == b'm' as usize {
                b'N'
            } else {
                (i % 256) as u8
            }
        })
        .collect();

    let sieve = SimdSieve::new_case_insensitive(&haystack, &[b"M"]).unwrap();
    let results: Vec<usize> = sieve.collect();

    assert!(
        results.is_empty(),
        "expected zero matches inside dense noise"
    );
}

#[test]
fn test_multibyte_adversarial_overlap() {
    // Looking for "CVE" and "cve" and "cVe" etc using native bitwise OR
    let haystack = b"xxCVxxCvExxCVVVExxcVEVExC".to_vec();

    // "CvE" is at 6
    // "cVE" is at 18
    let sieve = SimdSieve::new_case_insensitive(&haystack, &[b"CVE"]).unwrap();
    let results: Vec<usize> = sieve.collect();
    assert_eq!(results, vec![6, 18]);
}

#[test]
fn test_multi_pattern_parallel_extraction() {
    let haystack = b"xxABCxxdEfxxGHIxxAbCDeF".to_vec();
    // ABC / AbC at 2, 17
    // dEf / DeF at 7, 20
    let sieve = SimdSieve::new_case_insensitive(&haystack, &[b"ABC", b"DEF"]).unwrap();
    let mut results: Vec<usize> = sieve.collect();
    results.sort_unstable();
    assert_eq!(results, vec![2, 7, 17, 20]);
}
