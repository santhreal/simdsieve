//! Adversarial SIMD boundary tests.
use simdsieve::SimdSieve;


#[test]
fn test_split_across_32_byte_avx2_boundary() {
    let mut haystack = vec![b'A'; 64];
    // Place pattern starting at index 30, so it crosses 32
    // "TARGET" is 6 bytes. 30, 31, 32, 33, 34, 35
    haystack[30..36].copy_from_slice(b"TARGET");
    let patterns: &[&[u8]] = &[b"TARGET"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![30]);
}

#[test]
fn test_pattern_at_exact_end() {
    let mut haystack = vec![b'A'; 64];
    haystack[60..64].copy_from_slice(b"END!");
    let patterns: &[&[u8]] = &[b"END!"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![60]);
}

#[test]
fn test_input_shorter_than_simd_lane_width() {
    let haystack = b"SHORT";
    let patterns: &[&[u8]] = &[b"OR"];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![2]);
}

#[test]
fn test_all_zero_input() {
    let haystack = vec![0u8; 128];
    let patterns: &[&[u8]] = &[&[0, 0, 0, 0]];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    // It should match at every possible starting index up to 128 - 4 = 124
    let expected: Vec<usize> = (0..=124).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_all_0xff_input() {
    let haystack = vec![0xffu8; 128];
    let patterns: &[&[u8]] = &[&[0xff, 0xff]];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..=126).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_single_byte_pattern() {
    let mut haystack = vec![0u8; 64];
    haystack[17] = b'X';
    haystack[42] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![17, 42]);
}

#[test]
fn test_multiple_patterns_at_boundary() {
    let mut haystack = vec![b'A'; 64];
    haystack[31..33].copy_from_slice(b"B1");
    haystack[62..64].copy_from_slice(b"B2");
    let patterns: &[&[u8]] = &[b"B1", b"B2"];
    let mut matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    matches.sort_unstable();
    assert_eq!(matches, vec![31, 62]);
}

#[test]
fn test_case_insensitive_boundary() {
    let mut haystack = vec![b'A'; 64];
    haystack[30..34].copy_from_slice(b"TeSt");
    let patterns: &[&[u8]] = &[b"test"];
    let matches: Vec<usize> = SimdSieve::new_case_insensitive(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![30]);
}

#[test]
fn test_pattern_spanning_64_byte_boundary() {
    let mut haystack = vec![b'A'; 128];
    haystack[62..66].copy_from_slice(b"SPAN");
    let patterns: &[&[u8]] = &[b"SPAN"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![62]);
}

#[test]
fn test_pattern_spanning_128_byte_boundary() {
    let mut haystack = vec![b'A'; 256];
    haystack[126..130].copy_from_slice(b"BNDR");
    let patterns: &[&[u8]] = &[b"BNDR"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![126]);
}

#[test]
fn test_empty_haystack() {
    let haystack: &[u8] = &[];
    let patterns: &[&[u8]] = &[b"A"];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    assert!(matches.is_empty());
}

#[test]
fn test_haystack_length_1() {
    let haystack = b"X";
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![0]);
}

#[test]
fn test_pattern_longer_than_haystack() {
    let haystack = b"SHORT";
    let patterns: &[&[u8]] = &[b"SHORTER"];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    assert!(matches.is_empty());
}

#[test]
fn test_overlapping_matches() {
    let haystack = b"AAAA";
    let patterns: &[&[u8]] = &[b"AA"];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![0, 1, 2]);
}

#[test]
fn test_16_single_byte_patterns() {
    let haystack = b"0123456789ABCDEF";
    let patterns: &[&[u8]] = &[
        b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7",
        b"8", b"9", b"A", b"B", b"C", b"D", b"E", b"F"
    ];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..16).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_all_zeros_with_one_0xff_pattern() {
    let haystack = vec![0u8; 128];
    let patterns: &[&[u8]] = &[&[0xff]];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert!(matches.is_empty());
}

#[test]
fn test_all_0xff_with_one_zero_pattern() {
    let haystack = vec![0xffu8; 128];
    let patterns: &[&[u8]] = &[&[0]];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert!(matches.is_empty());
}

#[test]
fn test_boundary_31() {
    let mut haystack = vec![b'A'; 64];
    haystack[31] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![31]);
}

#[test]
fn test_boundary_32() {
    let mut haystack = vec![b'A'; 64];
    haystack[32] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![32]);
}

#[test]
fn test_boundary_63() {
    let mut haystack = vec![b'A'; 128];
    haystack[63] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![63]);
}

#[test]
fn test_boundary_64() {
    let mut haystack = vec![b'A'; 128];
    haystack[64] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![64]);
}

#[test]
fn test_boundary_127() {
    let mut haystack = vec![b'A'; 256];
    haystack[127] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![127]);
}

#[test]
fn test_boundary_128() {
    let mut haystack = vec![b'A'; 256];
    haystack[128] = b'X';
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![128]);
}

#[test]
fn test_exact_lane_width_match() {
    let haystack = vec![b'X'; 32];
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..32).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_exact_lane_width_minus_one_match() {
    let haystack = vec![b'X'; 31];
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..31).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_exact_lane_width_plus_one_match() {
    let haystack = vec![b'X'; 33];
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..33).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_16_byte_pattern() {
    let mut haystack = vec![b'A'; 64];
    let pattern = b"0123456789ABCDEF";
    haystack[20..36].copy_from_slice(pattern);
    let patterns: &[&[u8]] = &[pattern];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![20]);
}

#[test]
fn test_two_patterns_same_length_boundary() {
    let mut haystack = vec![b'A'; 64];
    haystack[30..34].copy_from_slice(b"WXYZ");
    haystack[40..44].copy_from_slice(b"ABCD");
    let patterns: &[&[u8]] = &[b"ABCD", b"WXYZ"];
    let mut matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    matches.sort_unstable();
    assert_eq!(matches, vec![30, 40]);
}

#[test]
fn test_all_0xff_with_one_0xff_single_byte() {
    let haystack = vec![0xffu8; 128];
    let patterns: &[&[u8]] = &[&[0xff]];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..128).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_all_0x00_with_one_0x00_single_byte() {
    let haystack = vec![0x00u8; 128];
    let patterns: &[&[u8]] = &[&[0x00]];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    let expected: Vec<usize> = (0..128).collect();
    assert_eq!(matches, expected);
}

#[test]
fn test_long_pattern_at_end() {
    let mut haystack = vec![b'A'; 64];
    let pattern = b"LONGPATTERN";
    haystack[64 - pattern.len()..64].copy_from_slice(pattern);
    let patterns: &[&[u8]] = &[pattern];
    let matches: Vec<usize> = SimdSieve::new(&haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![64 - pattern.len()]);
}

#[test]
fn test_consecutive_single_byte_matches() {
    let haystack = b"XXX";
    let patterns: &[&[u8]] = &[b"X"];
    let matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    assert_eq!(matches, vec![0, 1, 2]);
}

#[test]
fn test_interleaved_matches() {
    let haystack = b"XAYAZA";
    let patterns: &[&[u8]] = &[b"X", b"Y", b"Z"];
    let mut matches: Vec<usize> = SimdSieve::new(haystack, patterns).unwrap().collect();
    matches.sort_unstable();
    assert_eq!(matches, vec![0, 2, 4]);
}
