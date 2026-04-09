#[test]
fn test_avx2_lane_boundary_31_32() {
    use simdsieve::SimdSieve;

    // 2-byte pattern crossing AVX2 lane boundary at bytes 31-32
    let mut haystack = vec![b'a'; 128];
    haystack[31] = b'X';
    haystack[32] = b'Y';

    let results: Vec<usize> = SimdSieve::new(&haystack, &[b"XY"]).unwrap().collect();
    assert!(
        results.contains(&31),
        "Pattern XY at position 31 should be found, got {:?}",
        results
    );

    // 4-byte pattern starting at byte 30 (crosses 30-33)
    let mut haystack = vec![b'a'; 128];
    haystack[30] = b'A';
    haystack[31] = b'B';
    haystack[32] = b'C';
    haystack[33] = b'D';

    let results: Vec<usize> = SimdSieve::new(&haystack, &[b"ABCD"]).unwrap().collect();
    assert!(
        results.contains(&30),
        "Pattern ABCD at position 30 should be found, got {:?}",
        results
    );

    // 2-byte pattern at 63-64 (crosses 64-byte block boundary)
    let mut haystack = vec![b'a'; 128];
    haystack[63] = b'X';
    haystack[64] = b'Y';

    let results: Vec<usize> = SimdSieve::new(&haystack, &[b"XY"]).unwrap().collect();
    assert!(
        results.contains(&63),
        "Pattern XY at position 63 should be found, got {:?}",
        results
    );
}
