#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::SimdSieve;
use std::sync::Arc;
use std::thread;

#[test]
fn test_massive_thread_contention() {
    let mut haystack = vec![0u8; 100 * 1024 * 1024];

    haystack[1024] = b'T';
    haystack[1025] = b'G';
    haystack[1026] = b'T';
    haystack[50_000_000] = b'Q';
    haystack[50_000_001] = b'Z';
    haystack[50_000_002] = b'Q';
    haystack[99_999_997] = b'T';
    haystack[99_999_998] = b'G';
    haystack[99_999_999] = b'T';

    let shared_haystack = Arc::new(haystack);
    let mut handles = vec![];

    for _ in 0..256 {
        let hay = shared_haystack.clone();
        handles.push(thread::spawn(move || {
            let sieve = SimdSieve::new(&hay, &[b"TGT", b"QZQ"]).unwrap();
            let mut results: Vec<usize> = sieve.collect();
            results.sort_unstable();
            assert_eq!(results, vec![1024, 50_000_000, 99_999_997]);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
