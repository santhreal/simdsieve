#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Concurrent stress tests for simdsieve.

use simdsieve::{MultiSieve, SimdSieve};
use std::sync::Arc;
use std::thread;

#[test]
fn test_concurrent_stress_simdsieve() {
    let haystack = Arc::new(vec![b'A'; 1024 * 1024]); // 1MB haystack
    let patterns = vec![b"AAA".as_ref(), b"AAB".as_ref(), b"BAA".as_ref()];
    let mut handles = vec![];

    for _ in 0..32 {
        let h = Arc::clone(&haystack);
        let pats = patterns.clone();
        handles.push(thread::spawn(move || {
            let sieve = SimdSieve::new(&h, &pats).expect("SimdSieve::new failed concurrently");
            let count = sieve.count();
            assert!(
                count > 0,
                "Concurrent SimdSieve expected to find matches in A-filled haystack"
            );
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_stress_multisieve() {
    let haystack = Arc::new(vec![b'B'; 512 * 1024]);
    // 30 patterns to trigger multiple chunks in MultiSieve
    let patterns: Vec<&[u8]> = std::iter::repeat_n(b"BBB".as_slice(), 30).collect();

    let mut handles = vec![];

    for _ in 0..32 {
        let h = Arc::clone(&haystack);
        let pats = patterns.clone();
        handles.push(thread::spawn(move || {
            let multisieve =
                MultiSieve::new(&h, &pats).expect("MultiSieve::new failed concurrently");
            let count = multisieve.candidates().count();
            assert!(
                count > 0,
                "Concurrent MultiSieve expected to find matches in B-filled haystack"
            );
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_stress_estimate_match_count() {
    let haystack = Arc::new(vec![0xFF; 256 * 1024]);
    let patterns = vec![&[0xFF, 0xFF][..]];
    let mut handles = vec![];

    for _ in 0..32 {
        let h = Arc::clone(&haystack);
        let pats = patterns.clone();
        handles.push(thread::spawn(move || {
            let count = SimdSieve::estimate_match_count(&h, &pats, false).unwrap();
            assert!(
                count > 0,
                "Concurrent estimate_match_count expected >0 matches for 0xFF haystack"
            );
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
