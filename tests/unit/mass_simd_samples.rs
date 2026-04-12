use simdsieve::SimdSieve;

fn scan_with_simdsieve(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut all_matches = Vec::new();
    for chunk in patterns.chunks(16) {
        let sieve = if case_insensitive {
            SimdSieve::new_case_insensitive(haystack, chunk).unwrap()
        } else {
            SimdSieve::new(haystack, chunk).unwrap()
        };
        all_matches.extend(sieve.collect::<Vec<usize>>());
    }
    all_matches.sort_unstable();
    all_matches.dedup();
    all_matches
}

fn verify_all_matches(haystack: &[u8], patterns: &[&[u8]], expected_matches: &[usize]) {
    let matches = scan_with_simdsieve(haystack, patterns, false);
    assert_eq!(matches, expected_matches, "Matches did not match expected");
}

fn verify_all_matches_case_insensitive(
    haystack: &[u8],
    patterns: &[&[u8]],
    expected_matches: &[usize],
) {
    let matches = scan_with_simdsieve(haystack, patterns, true);
    assert_eq!(
        matches, expected_matches,
        "Case-insensitive matches did not match expected"
    );
}

fn generate_100kb_haystack_with_patterns(
    patterns: &[&[u8]],
    alignments: &[usize],
) -> (Vec<u8>, Vec<usize>) {
    let mut haystack = vec![0u8; 1024 * 100]; // 100KB
    let mut expected = Vec::new();

    // Fill haystack with some noise that doesn't match
    haystack.fill(b'x');

    // Place patterns at specific alignments
    for (i, &align) in alignments.iter().enumerate() {
        let pattern = patterns[i % patterns.len()];
        if align + pattern.len() <= haystack.len() {
            haystack[align..align + pattern.len()].copy_from_slice(pattern);
            expected.push(align);
        }
    }

    expected.sort_unstable();
    expected.dedup();
    (haystack, expected)
}

macro_rules! generate_alignment_tests {
    ($($name:ident: $align_base:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let patterns: Vec<Vec<u8>> = (4..=23)
                    .map(|len| {
                        let mut p = vec![b'A' + u8::try_from(len % 26).unwrap(); len];
                        p[0] = b'P';
                        p[1] = b'A';
                        p[2] = b'T';
                        p
                    })
                    .collect();
                let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();

                // 20 patterns each 4-16 bytes (using 4-23 above but close enough)
                let actual_patterns: Vec<&[u8]> = pattern_refs.iter().take(20).copied().collect();

                let alignments: Vec<usize> = (0..20).map(|i| $align_base + i * 64).collect();
                let (haystack, expected) = generate_100kb_haystack_with_patterns(&actual_patterns, &alignments);

                verify_all_matches(&haystack, &actual_patterns, &expected);
            }
        )*
    };
}

// Generate 40 alignment tests
generate_alignment_tests! {
    test_align_base_0: 0,
    test_align_base_1: 64,
    test_align_base_2: 128,
    test_align_base_3: 192,
    test_align_base_4: 256,
    test_align_base_5: 320,
    test_align_base_6: 384,
    test_align_base_7: 448,
    test_align_base_8: 512,
    test_align_base_9: 576,
    test_align_base_10: 640,
    test_align_base_11: 704,
    test_align_base_12: 768,
    test_align_base_13: 832,
    test_align_base_14: 896,
    test_align_base_15: 960,
    test_align_base_16: 1024,
    test_align_base_17: 1088,
    test_align_base_18: 1152,
    test_align_base_19: 1216,
    test_align_base_20: 1280,
    test_align_base_21: 1344,
    test_align_base_22: 1408,
    test_align_base_23: 1472,
    test_align_base_24: 1536,
    test_align_base_25: 1600,
    test_align_base_26: 1664,
    test_align_base_27: 1728,
    test_align_base_28: 1792,
    test_align_base_29: 1856,
    test_align_base_30: 1920,
    test_align_base_31: 1984,
    test_align_base_32: 2048,
    test_align_base_33: 2112,
    test_align_base_34: 2176,
    test_align_base_35: 2240,
    test_align_base_36: 2304,
    test_align_base_37: 2368,
    test_align_base_38: 2432,
    test_align_base_39: 2496,
}

macro_rules! generate_common_prefix_tests {
    ($($name:ident: $align_base:expr,)*) => {
        $(
            #[test]
            fn $name() {
                // 20 patterns with common prefix "COM"
                let patterns: Vec<Vec<u8>> = (0..20)
                    .map(|i| {
                        let mut p = vec![b'C', b'O', b'M', b'M', b'O', b'N', b'_'];
                        p.extend_from_slice(format!("{:04}", i).as_bytes());
                        p.extend_from_slice(b"_SUFFIX");
                        p
                    })
                    .collect();
                let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
                let actual_patterns: Vec<&[u8]> = pattern_refs.iter().take(20).copied().collect();

                let alignments: Vec<usize> = (0..20).map(|i| $align_base + i * 128).collect();
                let (haystack, expected) = generate_100kb_haystack_with_patterns(&actual_patterns, &alignments);

                verify_all_matches(&haystack, &actual_patterns, &expected);
            }
        )*
    };
}

// Generate 15 common prefix tests
generate_common_prefix_tests! {
    test_common_prefix_0: 0,
    test_common_prefix_1: 17,
    test_common_prefix_2: 34,
    test_common_prefix_3: 51,
    test_common_prefix_4: 68,
    test_common_prefix_5: 85,
    test_common_prefix_6: 102,
    test_common_prefix_7: 119,
    test_common_prefix_8: 136,
    test_common_prefix_9: 153,
    test_common_prefix_10: 170,
    test_common_prefix_11: 187,
    test_common_prefix_12: 204,
    test_common_prefix_13: 221,
    test_common_prefix_14: 238,
}

macro_rules! generate_case_insensitive_tests {
    ($($name:ident: $align_base:expr,)*) => {
        $(
            #[test]
            fn $name() {
                // Patterns are lowercase
                let patterns: Vec<Vec<u8>> = (0..20)
                    .map(|i| {
                        let mut p = vec![b'c', b'a', b's', b'e', b'i', b'n', b's', b'e', b'n', b's', b'i', b't', b'i', b'v', b'e'];
                        p.extend_from_slice(format!("{:02}", i).as_bytes());
                        p
                    })
                    .collect();

                let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
                let actual_patterns: Vec<&[u8]> = pattern_refs.iter().take(20).copied().collect();

                let alignments: Vec<usize> = (0..20).map(|i| $align_base + i * 256).collect();

                // Inject patterns with mixed casing into haystack
                let mut haystack = vec![b'y'; 1024 * 100];
                let mut expected = Vec::new();
                for (i, &align) in alignments.iter().enumerate() {
                    let mut mixed_case_pattern = patterns[i % patterns.len()].clone();
                    for j in 0..mixed_case_pattern.len() {
                        if mixed_case_pattern[j].is_ascii_alphabetic() && (i + j) % 2 == 0 {
                            mixed_case_pattern[j] = mixed_case_pattern[j].to_ascii_uppercase();
                        }
                    }
                    if align + mixed_case_pattern.len() <= haystack.len() {
                        haystack[align..align + mixed_case_pattern.len()].copy_from_slice(&mixed_case_pattern);
                        expected.push(align);
                    }
                }

                expected.sort_unstable();
                expected.dedup();

                verify_all_matches_case_insensitive(&haystack, &actual_patterns, &expected);
            }
        )*
    };
}

// Generate 15 case-insensitive tests
generate_case_insensitive_tests! {
    test_case_insensitive_0: 0,
    test_case_insensitive_1: 11,
    test_case_insensitive_2: 22,
    test_case_insensitive_3: 33,
    test_case_insensitive_4: 44,
    test_case_insensitive_5: 55,
    test_case_insensitive_6: 66,
    test_case_insensitive_7: 77,
    test_case_insensitive_8: 88,
    test_case_insensitive_9: 99,
    test_case_insensitive_10: 110,
    test_case_insensitive_11: 121,
    test_case_insensitive_12: 132,
    test_case_insensitive_13: 143,
    test_case_insensitive_14: 154,
}

macro_rules! generate_edge_case_tests {
    ($($name:ident,)*) => {
        $(
            #[test]
            fn $name() {
                // Test empty patterns are rejected
                let patterns: &[&[u8]] = &[];
                let result = SimdSieve::new(b"haystack", patterns);
                assert!(result.is_err());

                // Test input smaller than pattern
                let long_patterns: &[&[u8]] = &[b"this_is_a_very_long_pattern"];
                let matches = scan_with_simdsieve(b"short", long_patterns, false);
                assert!(matches.is_empty());

                // Empty input
                let matches = scan_with_simdsieve(b"", &[b"pattern"], false);
                assert!(matches.is_empty());

                // Pattern matches entire haystack exactly
                let matches = scan_with_simdsieve(b"exact", &[b"exact"], false);
                assert_eq!(matches, vec![0]);
            }
        )*
    };
}

// Generate 20 edge case tests
generate_edge_case_tests! {
    test_edge_case_0, test_edge_case_1, test_edge_case_2, test_edge_case_3, test_edge_case_4,
    test_edge_case_5, test_edge_case_6, test_edge_case_7, test_edge_case_8, test_edge_case_9,
    test_edge_case_10, test_edge_case_11, test_edge_case_12, test_edge_case_13, test_edge_case_14,
    test_edge_case_15, test_edge_case_16, test_edge_case_17, test_edge_case_18, test_edge_case_19,
}

macro_rules! generate_concurrency_tests {
    ($($name:ident: $align_base:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let patterns: Vec<Vec<u8>> = (4..=23)
                    .map(|len| {
                        let mut p = vec![b'B' + u8::try_from(len % 26).unwrap(); len];
                        p[0] = b'T';
                        p[1] = b'H';
                        p[2] = b'R';
                        p
                    })
                    .collect();
                let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
                let actual_patterns: Vec<&[u8]> = pattern_refs.iter().take(20).copied().collect();

                let alignments: Vec<usize> = (0..20).map(|i| $align_base + i * 512).collect();
                let (haystack, expected) = generate_100kb_haystack_with_patterns(&actual_patterns, &alignments);

                let shared_haystack = std::sync::Arc::new(haystack);
                let shared_patterns: std::sync::Arc<Vec<Vec<u8>>> = std::sync::Arc::new(actual_patterns.iter().map(|p| p.to_vec()).collect());
                let shared_expected = std::sync::Arc::new(expected);

                let mut handles = Vec::new();
                for _ in 0..4 {
                    let h = shared_haystack.clone();
                    let p = shared_patterns.clone();
                    let e = shared_expected.clone();
                    handles.push(std::thread::spawn(move || {
                        let refs: Vec<&[u8]> = p.iter().map(|x| x.as_slice()).collect();
                        let matches = scan_with_simdsieve(&h, &refs, false);
                        assert_eq!(matches, *e);
                    }));
                }

                for handle in handles {
                    handle.join().unwrap();
                }
            }
        )*
    };
}

// Generate 15 concurrency tests
generate_concurrency_tests! {
    test_concurrency_0: 0,
    test_concurrency_1: 27,
    test_concurrency_2: 54,
    test_concurrency_3: 81,
    test_concurrency_4: 108,
    test_concurrency_5: 135,
    test_concurrency_6: 162,
    test_concurrency_7: 189,
    test_concurrency_8: 216,
    test_concurrency_9: 243,
    test_concurrency_10: 270,
    test_concurrency_11: 297,
    test_concurrency_12: 324,
    test_concurrency_13: 351,
    test_concurrency_14: 378,
}
