//! AVX-512 intrinsics for 128-byte dual-pumped multi-pattern prefix matching.
//!
#![allow(
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::incompatible_msrv
)]
//! This module implements the AVX-512 (512-bit) backend for `x86_64` targets.
//! It processes 128-byte blocks using two 64-byte "pumps" to maximize
//! instruction-level parallelism.
//!
//! # Architecture
//!
//! The AVX-512 backend uses `_mm512_cmpeq_epi8_mask` to compare 64 haystack
//! bytes against a broadcast pattern byte, producing a 64-bit mask directly
//! in a mask register (`k`-register). This eliminates the need for
//! `_mm256_movemask_epi8`-style extraction.
//!
//! ## Dual-Pump Processing
//!
//! Each 128-byte logical block is processed as two 64-byte halves:
//! 1. Load bytes 0-63 (pump A) and bytes 64-127 (pump B)
//! 2. Interleave comparisons across both pumps
//! 3. Return separate 64-bit bitmasks for each half
//!
//! # Safety
//!
//! All `unsafe` blocks in this module require:
//! 1. AVX-512F and AVX-512BW target features are available (verified at construction).
//! 2. Input blocks have sufficient trailing bytes for multi-byte prefixes.
//! 3. Pointer arithmetic stays within allocated slices.
//!
//! Unaligned loads (`_mm512_loadu_si512`) are safe for any valid pointer.

use crate::fold::fold_ascii_lowercase;
use core::arch::x86_64::{
    __m512i, _mm512_cmpeq_epi8_mask, _mm512_cmpge_epi8_mask, _mm512_cmple_epi8_mask,
    _mm512_loadu_si512, _mm512_mask_sub_epi8, _mm512_set1_epi8, _mm512_setzero_si512,
};

/// A single pattern's first 4 prefix bytes, aligned for AVX-512 loads.
///
/// The `#[repr(C, align(64))]` ensures the struct can be loaded
/// efficiently with AVX-512 aligned load instructions if desired.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
struct Avx512Pattern {
    /// First 1-4 prefix bytes (upper-cased when case-insensitive).
    bytes: [u8; 4],
    /// Number of valid prefix bytes (0-4).
    len: usize,
    /// Precomputed 512-bit broadcast vectors for each prefix byte.
    /// Avoids `_mm512_set1_epi8` per pattern per block in the hot loop.
    bcast: [__m512i; 4],
}

/// AVX-512 multi-pattern filter operating on 128-byte blocks.
///
/// Holds up to 8 patterns and produces bitmasks indicating which byte
/// positions in a block match at least one pattern prefix.
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub(crate) struct Avx512Filter {
    /// Loaded patterns (up to 16).
    patterns: [Avx512Pattern; 16],
    /// Number of valid patterns.
    pattern_count: usize,
    /// Maximum prefix length across all patterns (1-4).
    max_len: usize,
    /// Whether to use ASCII case-insensitive matching.
    case_insensitive: bool,
}

impl Avx512Filter {
    /// Builds an AVX-512 filter from up to 8 prefix byte slices.
    ///
    /// Each prefix is truncated to 4 bytes. When `case_insensitive` is
    /// `true`, ASCII `a`-`z` bytes are folded to upper-case.
    ///
    /// # Parameters
    ///
    /// - `prefixes`: Slice of pattern byte slices (max 8, each max 4 bytes).
    /// - `case_insensitive`: Enable ASCII case-insensitive matching.
    #[must_use]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub(crate) unsafe fn new(prefixes: &[&[u8]], case_insensitive: bool) -> Self {
        let mut max_len = 0;
        let count = prefixes.len().min(16);

        // Zero-initialize the array safely to avoid UB from uninitialized padding
        // or array elements when `count` < 16.
        let mut patterns: [Avx512Pattern; 16] = unsafe { core::mem::zeroed() };

        for (i, &slice) in prefixes.iter().take(16).enumerate() {
            let eval_len = slice.len().min(4);
            let mut arr = [0u8; 4];
            for j in 0..eval_len {
                arr[j] = if case_insensitive {
                    fold_ascii_lowercase(slice[j])
                } else {
                    slice[j]
                };
            }
            if eval_len > max_len {
                max_len = eval_len;
            }
            // Precompute broadcast vectors for each prefix byte.
            // SAFETY: `_mm512_setzero_si512()` produces valid all-zero SIMD vectors.
            // Only indices 0..eval_len are used (checked by p.len in hot loops).
            let mut bcast: [__m512i; 4] = [_mm512_setzero_si512(); 4];
            for j in 0..eval_len {
                #[allow(clippy::cast_possible_wrap)]
                {
                    bcast[j] = _mm512_set1_epi8(arr[j] as i8);
                }
            }
            patterns[i] = Avx512Pattern {
                bytes: arr,
                len: eval_len,
                bcast,
            };
        }

        Self {
            patterns,
            pattern_count: count,
            max_len,
            case_insensitive,
        }
    }

    /// Folds ASCII lowercase letters to uppercase in a 512-bit vector.
    ///
    /// Uses AVX-512 mask registers to perform branchless case folding.
    /// The mask is computed by range comparison, then used to selectively
    /// subtract 0x20 from lowercase bytes.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX-512F and AVX-512BW are available.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    unsafe fn ascii_fold_vector(v: __m512i) -> __m512i {
        let lower_a = _mm512_set1_epi8(b'a' as i8);
        let upper_z = _mm512_set1_epi8(b'z' as i8);
        let fold_val = _mm512_set1_epi8(0x20);

        // Create mask where each bit is set if corresponding byte >= 'a'
        let is_ge_a = _mm512_cmpge_epi8_mask(v, lower_a);
        // Create mask where each bit is set if corresponding byte <= 'z'
        let is_le_z = _mm512_cmple_epi8_mask(v, upper_z);
        // Both conditions must be true for lowercase
        let is_alpha = is_ge_a & is_le_z;

        // Subtract 0x20 only where mask is set
        _mm512_mask_sub_epi8(v, is_alpha, v, fold_val)
    }

    /// Scans a 128-byte block, returning per-half bitmasks.
    ///
    /// Returns `(mask_lo, mask_hi)` where bit `i` of `mask_lo` covers
    /// byte positions 0-63 and bit `i` of `mask_hi` covers 64-127.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `block.len() >= 128 + self.max_len.saturating_sub(1)`
    /// - The CPU supports AVX-512F and AVX-512BW
    ///
    /// # Implementation Notes
    ///
    /// Unlike AVX2, AVX-512 produces masks directly in mask registers,
    /// avoiding the need for `movemask` operations. The masks are
    /// combined using bitwise OR across patterns.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    #[must_use]
    pub(crate) unsafe fn check_128byte_block(&self, block: &[u8]) -> (u64, u64) {
        debug_assert!(
            block.len() >= 128 + self.max_len.saturating_sub(1),
            "block lacks trailing buffer"
        );

        let mut folded_mask_a: u64 = 0;
        let mut folded_mask_b: u64 = 0;

        // Load position 0 vectors for both pumps.
        // SAFETY: Caller guarantees block is at least 128 + max_len - 1
        // bytes. _mm512_loadu_si512 performs an unaligned 64-byte load.
        let mut v0_a: __m512i;
        let mut v0_b: __m512i;
        unsafe {
            v0_a = _mm512_loadu_si512(block.as_ptr().cast());
            v0_b = _mm512_loadu_si512(block.as_ptr().add(64).cast());

            if self.case_insensitive {
                v0_a = Self::ascii_fold_vector(v0_a);
                v0_b = Self::ascii_fold_vector(v0_b);
            }
        }

        let mut v1_a = v0_a;
        let mut v1_b = v0_b;
        let mut v2_a = v0_a;
        let mut v2_b = v0_b;
        let mut v3_a = v0_a;
        let mut v3_b = v0_b;

        // Load offset vectors conditionally based on max pattern length.
        if self.max_len > 1 {
            // SAFETY: offset 1 and 65 are within bounds because
            // block.len() >= 128 + max_len - 1 >= 128 + 1 = 129.
            unsafe {
                let mut v_a = _mm512_loadu_si512(block.as_ptr().add(1).cast());
                let mut v_b = _mm512_loadu_si512(block.as_ptr().add(65).cast());
                if self.case_insensitive {
                    v_a = Self::ascii_fold_vector(v_a);
                    v_b = Self::ascii_fold_vector(v_b);
                }
                v1_a = v_a;
                v1_b = v_b;
            }
        }
        if self.max_len > 2 {
            // SAFETY: offset 2 and 66 are within bounds.
            unsafe {
                let mut v_a = _mm512_loadu_si512(block.as_ptr().add(2).cast());
                let mut v_b = _mm512_loadu_si512(block.as_ptr().add(66).cast());
                if self.case_insensitive {
                    v_a = Self::ascii_fold_vector(v_a);
                    v_b = Self::ascii_fold_vector(v_b);
                }
                v2_a = v_a;
                v2_b = v_b;
            }
        }
        if self.max_len > 3 {
            // SAFETY: offset 3 and 67 are within bounds.
            unsafe {
                let mut v_a = _mm512_loadu_si512(block.as_ptr().add(3).cast());
                let mut v_b = _mm512_loadu_si512(block.as_ptr().add(67).cast());
                if self.case_insensitive {
                    v_a = Self::ascii_fold_vector(v_a);
                    v_b = Self::ascii_fold_vector(v_b);
                }
                v3_a = v_a;
                v3_b = v_b;
            }
        }

        // Compare each pattern against all loaded positions.
        // AVX-512 produces masks directly in mask registers.
        for p_idx in 0..self.pattern_count {
            let p = &self.patterns[p_idx];
            let mut pattern_mask_a: u64 = !0;
            let mut pattern_mask_b: u64 = !0;

            if p.len > 0 {
                pattern_mask_a &= _mm512_cmpeq_epi8_mask(v0_a, p.bcast[0]);
                pattern_mask_b &= _mm512_cmpeq_epi8_mask(v0_b, p.bcast[0]);
            }
            if p.len > 1 {
                pattern_mask_a &= _mm512_cmpeq_epi8_mask(v1_a, p.bcast[1]);
                pattern_mask_b &= _mm512_cmpeq_epi8_mask(v1_b, p.bcast[1]);
            }
            if p.len > 2 {
                pattern_mask_a &= _mm512_cmpeq_epi8_mask(v2_a, p.bcast[2]);
                pattern_mask_b &= _mm512_cmpeq_epi8_mask(v2_b, p.bcast[2]);
            }
            if p.len > 3 {
                pattern_mask_a &= _mm512_cmpeq_epi8_mask(v3_a, p.bcast[3]);
                pattern_mask_b &= _mm512_cmpeq_epi8_mask(v3_b, p.bcast[3]);
            }

            folded_mask_a |= pattern_mask_a;
            folded_mask_b |= pattern_mask_b;
        }

        (folded_mask_a, folded_mask_b)
    }

    /// Scans a 64-byte block, returning a single bitmask.
    ///
    /// Bit `i` is set if byte position `i` starts with a matching
    /// pattern prefix.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `block.len() >= 64 + self.max_len.saturating_sub(1)`
    /// - The CPU supports AVX-512F and AVX-512BW
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    #[must_use]
    pub(crate) unsafe fn check_64byte_block(&self, block: &[u8]) -> u64 {
        debug_assert!(
            block.len() >= 64 + self.max_len.saturating_sub(1),
            "block lacks trailing buffer"
        );

        let mut folded_mask: u64 = 0;

        // SAFETY: Caller guarantees sufficient block length.
        // _mm512_loadu_si512 performs an unaligned 64-byte load.
        unsafe {
            let mut v0: __m512i = _mm512_loadu_si512(block.as_ptr().cast());
            if self.case_insensitive {
                v0 = Self::ascii_fold_vector(v0);
            }

            let mut v1 = v0;
            let mut v2 = v0;
            let mut v3 = v0;

            if self.max_len > 1 {
                let mut v = _mm512_loadu_si512(block.as_ptr().add(1).cast());
                if self.case_insensitive {
                    v = Self::ascii_fold_vector(v);
                }
                v1 = v;
            }
            if self.max_len > 2 {
                let mut v = _mm512_loadu_si512(block.as_ptr().add(2).cast());
                if self.case_insensitive {
                    v = Self::ascii_fold_vector(v);
                }
                v2 = v;
            }
            if self.max_len > 3 {
                let mut v = _mm512_loadu_si512(block.as_ptr().add(3).cast());
                if self.case_insensitive {
                    v = Self::ascii_fold_vector(v);
                }
                v3 = v;
            }

            for p_idx in 0..self.pattern_count {
                let p = &self.patterns[p_idx];
                let mut pattern_mask: u64 = !0;
                if p.len > 0 {
                    pattern_mask &= _mm512_cmpeq_epi8_mask(v0, p.bcast[0]);
                }
                if p.len > 1 {
                    pattern_mask &= _mm512_cmpeq_epi8_mask(v1, p.bcast[1]);
                }
                if p.len > 2 {
                    pattern_mask &= _mm512_cmpeq_epi8_mask(v2, p.bcast[2]);
                }
                if p.len > 3 {
                    pattern_mask &= _mm512_cmpeq_epi8_mask(v3, p.bcast[3]);
                }
                folded_mask |= pattern_mask;
            }
        }
        folded_mask
    }
}

#[cfg(test)]
mod tests {
    use super::Avx512Filter;
    use crate::scalar::ScalarFilter;

    #[test]
    fn case_insensitive_masks_cover_half_and_block_boundaries() {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512bw") {
            return;
        }

        let filter = unsafe { Avx512Filter::new(&[b"Z"], true) };
        let mut block = [b'x'; 129];
        block[63] = b'Z';
        block[127] = b'Z';

        let (mask_a, mask_b) = unsafe { filter.check_128byte_block(&block) };
        eprintln!("mask_a={mask_a:064b}");
        eprintln!("mask_b={mask_b:064b}");

        assert_eq!(mask_a & (1_u64 << 63), 1_u64 << 63);
        assert_eq!(mask_b & (1_u64 << 63), 1_u64 << 63);
    }

    #[test]
    fn avx512_128byte_block_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512bw") {
            return;
        }

        let patterns: &[&[u8]] = &[b"ab", b"XY", b"1"];
        let avx512 = unsafe { Avx512Filter::new(patterns, false) };
        let scalar = ScalarFilter::new(patterns, false);

        let mut block = [b'x'; 132];
        block[10] = b'a';
        block[11] = b'b';
        block[67] = b'X';
        block[68] = b'Y';
        block[127] = b'1';

        let (mask_a, mask_b) = unsafe { avx512.check_128byte_block(&block) };
        let scalar_mask_lo = scalar.check_64byte_block(&block);
        let scalar_mask_hi = scalar.check_64byte_block(&block[64..]);

        assert_eq!(
            mask_a, scalar_mask_lo,
            "AVX-512 low 64 bytes must match scalar"
        );
        assert_eq!(
            mask_b, scalar_mask_hi,
            "AVX-512 high 64 bytes must match scalar"
        );
    }

    #[test]
    fn avx512_64byte_block_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512bw") {
            return;
        }

        let patterns: &[&[u8]] = &[b"te", b"ST"];
        let avx512 = unsafe { Avx512Filter::new(patterns, false) };
        let scalar = ScalarFilter::new(patterns, false);

        let mut block = [b'x'; 68];
        block[5] = b't';
        block[6] = b'e';
        block[62] = b'S';
        block[63] = b'T';

        let avx512_mask = unsafe { avx512.check_64byte_block(&block) };
        let scalar_mask = scalar.check_64byte_block(&block);

        assert_eq!(
            avx512_mask, scalar_mask,
            "AVX-512 64-byte block must match scalar backend"
        );
    }

    #[test]
    fn avx512_case_insensitive_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512bw") {
            return;
        }

        let patterns: &[&[u8]] = &[b"Ab", b"z"];
        let avx512 = unsafe { Avx512Filter::new(patterns, true) };
        let scalar = ScalarFilter::new(patterns, true);

        let mut block = [b'x'; 132];
        block[15] = b'a';
        block[16] = b'B';
        block[79] = b'Z';

        let (mask_a, mask_b) = unsafe { avx512.check_128byte_block(&block) };
        let scalar_mask_lo = scalar.check_64byte_block(&block);
        let scalar_mask_hi = scalar.check_64byte_block(&block[64..]);

        assert_eq!(
            mask_a, scalar_mask_lo,
            "AVX-512 CI low 64 bytes must match scalar"
        );
        assert_eq!(
            mask_b, scalar_mask_hi,
            "AVX-512 CI high 64 bytes must match scalar"
        );
    }
}
