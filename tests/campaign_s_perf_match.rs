//! S-perf-match campaign catalog: +40 simdsieve scalar/SIMD smoke cases.

#![allow(clippy::unwrap_used)]

use simdsieve::SimdSieve;

macro_rules! campaign_find {
    ($name:ident, $hay:expr, $pat:expr, $expect:expr) => {
        #[test]
        fn $name() {
            let hay = $hay;
            let pat = $pat;
            let got: Vec<usize> = SimdSieve::new(hay, &[pat]).unwrap().collect();
            assert_eq!(got, $expect);
        }
    };
}

campaign_find!(c00, b"", b"x", vec![]);
campaign_find!(c01, b"a", b"a", vec![0]);
campaign_find!(c02, b"ab", b"b", vec![1]);
campaign_find!(c03, b"aaa", b"aa", vec![0, 1]);
campaign_find!(c04, b"GET /admin", b"GET", vec![0]);
campaign_find!(c05, b"GET /admin", b"/admin", vec![4]);
campaign_find!(c06, b"\x00\x01\x02", b"\x00", vec![0]);
campaign_find!(c07, b"foo bar foo", b"foo", vec![0, 8]);
campaign_find!(c08, b"needle", b"eed", vec![1]);
campaign_find!(c09, b"zzzz", b"z", vec![0, 1, 2, 3]);

#[test]
fn campaign_multi_pattern_union() {
    let hay = b"alpha beta gamma";
    let got: Vec<usize> = SimdSieve::new(hay, &[b"alpha", b"gamma"]).unwrap().collect();
    assert!(got.contains(&0));
    assert!(got.contains(&11));
}

#[test]
fn campaign_no_match_long_hay() {
    let hay = vec![b'x'; 256];
    let got: Vec<usize> = SimdSieve::new(&hay, &[b"needle"]).unwrap().collect();
    assert!(got.is_empty());
}

#[test]
fn campaign_pattern_longer_than_hay() {
    let got: Vec<usize> = SimdSieve::new(b"ab", &[b"abcd"]).unwrap().collect();
    assert!(got.is_empty());
}

#[test]
fn campaign_empty_pattern_list() {
    assert!(SimdSieve::new(b"data", &[]).is_err());
}

#[test]
fn campaign_duplicate_patterns() {
    let got: Vec<usize> = SimdSieve::new(b"aa", &[b"a", b"a"]).unwrap().collect();
    assert_eq!(got, vec![0, 1]);
}

#[test]
fn campaign_case_sensitive() {
    let got: Vec<usize> = SimdSieve::new(b"AbC", &[b"abc"]).unwrap().collect();
    assert!(got.is_empty());
}

#[test]
fn campaign_boundary_last_byte() {
    let got: Vec<usize> = SimdSieve::new(b"xxz", &[b"z"]).unwrap().collect();
    assert_eq!(got, vec![2]);
}

#[test]
fn campaign_overlapping_occurrences() {
    let got: Vec<usize> = SimdSieve::new(b"ababab", &[b"aba"]).unwrap().collect();
    assert_eq!(got, vec![0, 2]);
}

#[test]
fn campaign_binary_blob() {
    let hay = [0u8, 255, 0, 255, 0];
    let got: Vec<usize> = SimdSieve::new(&hay, &[&[255]]).unwrap().collect();
    assert_eq!(got, vec![1, 3]);
}

#[test]
fn campaign_single_byte_many() {
    let hay = b"a".repeat(64);
    let got: Vec<usize> = SimdSieve::new(&hay, &[b"a"]).unwrap().collect();
    assert_eq!(got.len(), 64);
}

#[test]
fn campaign_two_byte_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"hello", &[b"ll"]).unwrap().collect();
    assert_eq!(got, vec![2]);
}

#[test]
fn campaign_three_byte_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"abcdef", &[b"cde"]).unwrap().collect();
    assert_eq!(got, vec![2]);
}

#[test]
fn campaign_four_byte_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"12345678", &[b"5678"]).unwrap().collect();
    assert_eq!(got, vec![4]);
}

#[test]
fn campaign_prefix_at_zero() {
    let got: Vec<usize> = SimdSieve::new(b"prefix", &[b"pre"]).unwrap().collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn campaign_suffix_only() {
    let got: Vec<usize> = SimdSieve::new(b"endsuffix", &[b"suffix"]).unwrap().collect();
    assert_eq!(got, vec![3]);
}

#[test]
fn campaign_whitespace_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"a b c", &[b" "]).unwrap().collect();
    assert_eq!(got, vec![1, 3]);
}

#[test]
fn campaign_newline_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"a\nb\n", &[b"\n"]).unwrap().collect();
    assert_eq!(got, vec![1, 3]);
}

#[test]
fn campaign_tab_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"a\tb", &[b"\t"]).unwrap().collect();
    assert_eq!(got, vec![1]);
}

#[test]
fn campaign_high_bit_pattern() {
    let got: Vec<usize> = SimdSieve::new(b"\x80\x81", &[b"\x80"]).unwrap().collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn campaign_repeated_pattern_overlapping_matches() {
    let got: Vec<usize> = SimdSieve::new(b"----", &[b"--"]).unwrap().collect();
    assert_eq!(got, vec![0, 1, 2]);
}

#[test]
fn campaign_http_method() {
    let hay = b"POST /api HTTP/1.1\r\n";
    let got: Vec<usize> = SimdSieve::new(hay, &[b"POST", b"/api"]).unwrap().collect();
    assert!(got.contains(&0));
    assert!(got.contains(&5));
}

#[test]
fn campaign_json_key() {
    let hay = br#"{"key":"value"}"#;
    let got: Vec<usize> = SimdSieve::new(hay, &[b"\"key\""]).unwrap().collect();
    assert_eq!(got, vec![1]);
}

#[test]
fn campaign_pe_magic() {
    let hay = b"\x4d\x5a\x90\x00";
    let got: Vec<usize> = SimdSieve::new(hay, &[b"MZ"]).unwrap().collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn campaign_elf_magic() {
    let hay = b"\x7fELF";
    let got: Vec<usize> = SimdSieve::new(hay, &[b"\x7fELF"]).unwrap().collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn campaign_zip_local_header() {
    let hay = b"\x50\x4b\x03\x04file";
    let got: Vec<usize> = SimdSieve::new(hay, &[b"\x50\x4b\x03\x04"]).unwrap().collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn campaign_many_short_patterns() {
    let patterns: Vec<Vec<u8>> = (0u8..16).map(|b| vec![b]).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();
    let hay = (0u8..16).collect::<Vec<_>>();
    let got: Vec<usize> = SimdSieve::new(&hay, &pattern_refs).unwrap().collect();
    assert_eq!(got.len(), 16);
}

#[test]
fn campaign_pattern_equals_hay() {
    let got: Vec<usize> = SimdSieve::new(b"exact", &[b"exact"]).unwrap().collect();
    assert_eq!(got, vec![0]);
}

#[test]
fn campaign_interleaved_miss() {
    let got: Vec<usize> = SimdSieve::new(b"abcXdefXghi", &[b"def"]).unwrap().collect();
    assert_eq!(got, vec![4]);
}

#[test]
fn campaign_collect_twice_same() {
    let sieve = SimdSieve::new(b"test test", &[b"test"]).unwrap();
    let a: Vec<_> = sieve.collect();
    let b: Vec<_> = SimdSieve::new(b"test test", &[b"test"]).unwrap().collect();
    assert_eq!(a, b);
}

#[test]
fn campaign_utf8_bytes_literal() {
    let hay = "café".as_bytes();
    let got: Vec<usize> = SimdSieve::new(hay, &[b"\xC3\xA9"]).unwrap().collect();
    assert_eq!(got, vec![3]);
}

#[test]
fn campaign_max_practical_pattern_len() {
    let pat = vec![b'A'; 32];
    let mut hay = pat.clone();
    hay.extend(b"tail");
    let got: Vec<usize> = SimdSieve::new(&hay, &[&pat]).unwrap().collect();
    assert_eq!(got, vec![0]);
}
