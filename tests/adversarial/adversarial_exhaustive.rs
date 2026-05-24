#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Exhaustive adversarial test suite for simdsieve.
//!
//! This test suite provides 40+ rigorous tests covering:
//! - Multi-pattern prefix matching edge cases
//! - SIMD backend correctness verification
//! - Dual-pump processing validation
//! - Runtime backend selection
//! - Adversarial inputs and edge cases
//!
//! Every assertion includes diagnostic messages for failure analysis.

use simdsieve::{SimdSieve, SimdSieveError};

// =============================================================================
// Reference Implementation — Brute-force Linear Scan
// =============================================================================

/// Returns every offset where at least one pattern matches exactly.
/// This is the ground truth for all correctness verification.
fn reference_scan(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut hits = Vec::new();
    for i in 0..haystack.len().saturating_add(1) {
        for &pat in patterns {
            if pat.is_empty() {
                hits.push(i);
                break;
            }
            if i + pat.len() > haystack.len() {
                continue;
            }
            let matches = if case_insensitive {
                haystack[i..i + pat.len()]
                    .iter()
                    .zip(pat)
                    .all(|(&a, &b)| a.eq_ignore_ascii_case(&b))
            } else {
                &haystack[i..i + pat.len()] == pat
            };
            if matches {
                hits.push(i);
                break;
            }
        }
    }
    hits.sort_unstable();
    hits.dedup();
    hits
}

/// Verify `SimdSieve` produces exactly matching results to reference scan.
fn assert_matches_reference(
    haystack: &[u8],
    patterns: &[&[u8]],
    case_insensitive: bool,
    test_name: &str,
) {
    let sieve = if case_insensitive {
        SimdSieve::new_case_insensitive(haystack, patterns)
    } else {
        SimdSieve::new(haystack, patterns)
    }
    .unwrap_or_else(|e| panic!("[{test_name}] construction failed: {e:?}"));

    let sieve_hits: Vec<usize> = sieve.collect();
    let expected = reference_scan(haystack, patterns, case_insensitive);

    assert_eq!(
        sieve_hits,
        expected,
        "[{test_name}] mismatch: haystack_len={}, patterns={:?}, sieve={:?}, expected={:?}",
        haystack.len(),
        patterns
            .iter()
            .map(|p| String::from_utf8_lossy(p))
            .collect::<Vec<_>>(),
        sieve_hits,
        expected
    );
}

// =============================================================================
// Test Group 1: Multi-Pattern Prefix Matching
// =============================================================================

#[test]
fn zero_patterns_returns_error() {
    let result = SimdSieve::new(b"haystack", &[]);
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPatternSet)),
        "zero patterns should return EmptyPatternSet error"
    );
}

#[test]
fn one_pattern_basic() {
    let haystack = b"hello world hello";
    assert_matches_reference(haystack, &[b"hello"], false, "one_pattern_basic");
}

#[test]
fn one_pattern_not_found() {
    let haystack = b"hello world";
    assert_matches_reference(haystack, &[b"xyz"], false, "one_pattern_not_found");
}

#[test]
fn eight_patterns_sieve_boundary() {
    // 8 patterns tests the SIMD register saturation boundary
    let haystack = b"ABCDEFGH_abcdefgh_12345678_XYZxyz!@#";
    let patterns: &[&[u8]] = &[
        b"ABC", b"DEF", b"GHI", b"abc", b"def", b"ghi", b"123", b"XYZ",
    ];
    assert_matches_reference(haystack, patterns, false, "eight_patterns_sieve_boundary");
}

#[test]
fn nine_patterns_exceeds_single_sieve() {
    // 9 patterns - verify we can still construct (limit is 16)
    let haystack = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let patterns: &[&[u8]] = &[
        b"ABC", b"DEF", b"GHI", b"JKL", b"MNO", b"PQR", b"STU", b"VWX", b"YZ",
    ];
    let result = SimdSieve::new(haystack, patterns);
    assert!(result.is_ok(), "9 patterns should succeed (limit is 16)");
    assert_matches_reference(
        haystack,
        patterns,
        false,
        "nine_patterns_exceeds_single_sieve",
    );
}

#[test]
fn sixteen_patterns_at_limit() {
    // Exactly at the pattern limit
    let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(16);
    for i in 0..16 {
        patterns.push(format!("P{i:02}").into_bytes());
    }
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

    let mut haystack = vec![b'_'; 200];
    for (i, pat) in patterns.iter().enumerate() {
        let start = i * 10;
        haystack[start..start + pat.len()].copy_from_slice(pat);
    }

    assert_matches_reference(&haystack, &pattern_refs, false, "sixteen_patterns_at_limit");
}

#[test]
fn seventeen_patterns_exceeds_limit() {
    let patterns: Vec<Vec<u8>> = (0..17).map(|i| format!("p{i}").into_bytes()).collect();
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

    let result = SimdSieve::new(b"haystack", &pattern_refs);
    assert!(
        matches!(result, Err(SimdSieveError::PatternLimitExceeded(17))),
        "17 patterns should exceed limit"
    );
}

#[test]
fn empty_pattern_returns_error() {
    // Empty patterns are rejected immediately to prevent matching every position.
    let haystack = b"ABC";
    let result = SimdSieve::new(haystack, &[b""]);
    assert!(
        matches!(result, Err(SimdSieveError::EmptyPattern { index: 0 })),
        "a set containing an empty pattern should return EmptyPattern error"
    );
}

#[test]
fn one_byte_pattern() {
    let haystack = vec![b'x'; 100];
    assert_matches_reference(&haystack, &[b"x"], false, "one_byte_pattern");
}

#[test]
fn very_long_pattern() {
    // Pattern longer than 4 bytes (prefix limit)
    let pattern = b"this_is_a_very_long_pattern";
    let mut haystack = vec![b'_'; 100];
    haystack[10..10 + pattern.len()].copy_from_slice(pattern);
    assert_matches_reference(&haystack, &[pattern], false, "very_long_pattern");
}

#[test]
fn pattern_that_matches_every_byte() {
    // Pattern 0x00 with all-zero input - every position matches
    let haystack = vec![0x00u8; 50];
    assert_matches_reference(
        &haystack,
        &[b"\x00"],
        false,
        "pattern_that_matches_every_byte",
    );
}

#[test]
fn pattern_that_matches_nothing() {
    // Pattern that definitely doesn't exist
    let haystack = vec![b'a'; 100];
    assert_matches_reference(&haystack, &[b"XYZ"], false, "pattern_that_matches_nothing");
}

#[test]
fn overlapping_prefix_matches() {
    // Patterns that share prefixes
    let haystack = b"abababababababab";
    let patterns: &[&[u8]] = &[b"aba", b"bab", b"abab"];
    assert_matches_reference(haystack, patterns, false, "overlapping_prefix_matches");
}

// =============================================================================
// Test Group 2: SIMD Backend Correctness — Input Sizes
// =============================================================================

#[test]
fn input_size_0_bytes() {
    let haystack: &[u8] = b"";
    assert_matches_reference(haystack, &[b"x"], false, "input_size_0_bytes");
}

#[test]
fn input_size_1_byte() {
    let haystack: &[u8] = b"x";
    assert_matches_reference(haystack, &[b"x"], false, "input_size_1_byte");
}

#[test]
fn input_size_63_bytes() {
    let mut haystack = vec![b'x'; 63];
    haystack[30..33].copy_from_slice(b"ABC");
    assert_matches_reference(&haystack, &[b"ABC"], false, "input_size_63_bytes");
}

#[test]
fn input_size_64_bytes() {
    let mut haystack = vec![b'x'; 64];
    haystack[30..33].copy_from_slice(b"ABC");
    haystack[62..64].copy_from_slice(b"DE");
    assert_matches_reference(&haystack, &[b"ABC", b"DE"], false, "input_size_64_bytes");
}

#[test]
fn input_size_65_bytes() {
    let mut haystack = vec![b'x'; 65];
    haystack[30..33].copy_from_slice(b"ABC");
    haystack[63..65].copy_from_slice(b"DE");
    assert_matches_reference(&haystack, &[b"ABC", b"DE"], false, "input_size_65_bytes");
}

#[test]
fn input_size_127_bytes() {
    let mut haystack = vec![b'x'; 127];
    haystack[30..33].copy_from_slice(b"ABC");
    haystack[100..103].copy_from_slice(b"DEF");
    assert_matches_reference(&haystack, &[b"ABC", b"DEF"], false, "input_size_127_bytes");
}

#[test]
fn input_size_128_bytes() {
    let mut haystack = vec![b'x'; 128];
    haystack[30..33].copy_from_slice(b"ABC");
    haystack[100..103].copy_from_slice(b"DEF");
    haystack[125..128].copy_from_slice(b"GHI");
    assert_matches_reference(
        &haystack,
        &[b"ABC", b"DEF", b"GHI"],
        false,
        "input_size_128_bytes",
    );
}

#[test]
fn input_size_256_bytes() {
    let mut haystack = vec![b'x'; 256];
    haystack[50..53].copy_from_slice(b"AAA");
    haystack[150..153].copy_from_slice(b"BBB");
    haystack[250..253].copy_from_slice(b"CCC");
    assert_matches_reference(
        &haystack,
        &[b"AAA", b"BBB", b"CCC"],
        false,
        "input_size_256_bytes",
    );
}

#[test]
fn input_size_1024_bytes() {
    let mut haystack = vec![b'x'; 1024];
    haystack[100..103].copy_from_slice(b"PAT");
    haystack[500..503].copy_from_slice(b"ERN");
    haystack[1000..1003].copy_from_slice(b"MAT");
    assert_matches_reference(
        &haystack,
        &[b"PAT", b"ERN", b"MAT"],
        false,
        "input_size_1024_bytes",
    );
}

#[test]
#[cfg(not(miri))]
fn input_size_1mb() {
    let mut haystack = vec![b'x'; 1024 * 1024];
    haystack[1000..1004].copy_from_slice(b"ONE!");
    haystack[500000..500004].copy_from_slice(b"TWO!");
    haystack[1048572..1048576].copy_from_slice(b"END!");
    assert_matches_reference(
        &haystack,
        &[b"ONE!", b"TWO!", b"END!"],
        false,
        "input_size_1mb",
    );
}

// =============================================================================
// Test Group 3: Input Alignment
// =============================================================================

fn create_aligned_haystack(size: usize, align: usize) -> Vec<u8> {
    // Over-allocate, slice to an aligned window, then copy into a
    // fresh Vec via `extend_from_slice` to grow capacity to `size`
    // exactly. The result's pointer alignment depends on the
    // allocator, so we ASSERT alignment and retry up to a few
    // times — and if it still doesn't land, panic with a clear
    // message instead of silently producing misaligned data.
    //
    // The previous implementation relied on `.to_vec()` preserving
    // the source's alignment, which is an allocator-quirk that held
    // on Linux glibc but not on Windows msvcrt — Windows nightly CI
    // was failing on `alignment_32_bytes_aligned` as a result.
    for _attempt in 0..16 {
        let v = vec![0u8; size + align * 4];
        let ptr = v.as_ptr() as usize;
        let aligned_ptr = if ptr.is_multiple_of(align) {
            ptr
        } else {
            ptr + (align - ptr % align)
        };
        let offset = aligned_ptr - ptr;
        let aligned_slice = &v[offset..offset + size];
        let mut result = Vec::with_capacity(size);
        result.extend_from_slice(aligned_slice);
        if result.as_ptr() as usize % align == 0 {
            return result;
        }
    }
    panic!(
        "could not obtain a {}-byte-aligned Vec<u8> of size {} after 16 attempts \
         — allocator does not provide sufficient alignment for this test on this \
         platform; skip with `#[cfg(not(target_os = \"windows\"))]` if needed",
        align, size
    );
}

#[test]
fn alignment_32_bytes_aligned() {
    let mut haystack = create_aligned_haystack(256, 32);
    haystack.fill(b'x');
    haystack[100..103].copy_from_slice(b"ABC");
    assert_matches_reference(&haystack, &[b"ABC"], false, "alignment_32_bytes_aligned");
}

#[test]
fn alignment_unaligned_by_1() {
    let mut haystack = vec![b'x'; 256];
    // Shift data by slicing (creates unaligned view)
    let unaligned = &mut haystack[1..];
    unaligned[100..103].copy_from_slice(b"ABC");
    assert_matches_reference(unaligned, &[b"ABC"], false, "alignment_unaligned_by_1");
}

#[test]
fn alignment_unaligned_by_15() {
    let mut haystack = vec![b'x'; 256];
    let unaligned = &mut haystack[15..];
    unaligned[100..103].copy_from_slice(b"ABC");
    assert_matches_reference(unaligned, &[b"ABC"], false, "alignment_unaligned_by_15");
}

#[test]
fn alignment_unaligned_by_31() {
    let mut haystack = vec![b'x'; 256];
    let unaligned = &mut haystack[31..];
    unaligned[100..103].copy_from_slice(b"ABC");
    assert_matches_reference(unaligned, &[b"ABC"], false, "alignment_unaligned_by_31");
}

// =============================================================================
// Test Group 4: Dual-Pump Processing
// =============================================================================

#[test]
fn dual_pump_exact_64_byte_boundary() {
    // AVX2: exactly at 64-byte boundary
    let mut haystack = vec![b'x'; 128];
    haystack[31..34].copy_from_slice(b"ABC"); // Position 31 in first 64-byte block
    haystack[63..66].copy_from_slice(b"DEF"); // Position 63 (end of first block)
    haystack[64..67].copy_from_slice(b"GHI"); // Position 64 (start of second block)
    assert_matches_reference(
        &haystack,
        &[b"ABC", b"DEF", b"GHI"],
        false,
        "dual_pump_exact_64_byte_boundary",
    );
}

#[test]
fn dual_pump_exact_128_byte_boundary() {
    // AVX-512: exactly at 128-byte boundary
    let mut haystack = vec![b'x'; 256];
    haystack[63..66].copy_from_slice(b"ABC"); // Position 63 in first 128-byte block
    haystack[127..130].copy_from_slice(b"DEF"); // Position 127 (end of first block)
    haystack[128..131].copy_from_slice(b"GHI"); // Position 128 (start of second block)
    assert_matches_reference(
        &haystack,
        &[b"ABC", b"DEF", b"GHI"],
        false,
        "dual_pump_exact_128_byte_boundary",
    );
}

#[test]
fn dual_pump_split_at_pump_boundary_32() {
    // AVX2 splits 64-byte block into two 32-byte pumps
    let mut haystack = vec![b'x'; 128];
    haystack[30..33].copy_from_slice(b"ABC"); // Near pump A/B boundary
    haystack[31..34].copy_from_slice(b"DEF"); // At pump A/B boundary
    haystack[32..35].copy_from_slice(b"GHI"); // Just after pump A/B boundary
    assert_matches_reference(
        &haystack,
        &[b"ABC", b"DEF", b"GHI"],
        false,
        "dual_pump_split_at_pump_boundary_32",
    );
}

#[test]
fn dual_pump_split_at_pump_boundary_64() {
    // AVX-512 splits 128-byte block into two 64-byte pumps
    let mut haystack = vec![b'x'; 256];
    haystack[62..65].copy_from_slice(b"ABC"); // Near pump A/B boundary
    haystack[63..66].copy_from_slice(b"DEF"); // At pump A/B boundary
    haystack[64..67].copy_from_slice(b"GHI"); // Just after pump A/B boundary
    assert_matches_reference(
        &haystack,
        &[b"ABC", b"DEF", b"GHI"],
        false,
        "dual_pump_split_at_pump_boundary_64",
    );
}
