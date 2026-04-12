use simdsieve::SimdSieve;

#[test]
fn test_boundary_hit() {
    let mut haystack = vec![0u8; 64];
    haystack[31] = b'X';
    haystack[32] = b'Y';

    let results: Vec<usize> = SimdSieve::new(&haystack, &[b"XY"]).unwrap().collect();
    assert!(
        results.contains(&31),
        "Pattern XY at position 31 should be found, got {results:?}",
    );
}

#[test]
fn test_boundary_hit_4byte() {
    let mut haystack = vec![0u8; 64];
    haystack[30] = b'A';
    haystack[31] = b'B';
    haystack[32] = b'C';
    haystack[33] = b'D';

    let results: Vec<usize> = SimdSieve::new(&haystack, &[b"ABCD"]).unwrap().collect();
    assert!(
        results.contains(&30),
        "Pattern ABCD at position 30 should be found, got {results:?}",
    );
}

#[test]
fn test_block_end() {
    let mut haystack = vec![0u8; 128];
    haystack[63] = b'X';
    haystack[64] = b'Y';

    let results: Vec<usize> = SimdSieve::new(&haystack, &[b"XY"]).unwrap().collect();
    assert!(
        results.contains(&63),
        "Pattern XY at position 63 should be found, got {results:?}",
    );
}
