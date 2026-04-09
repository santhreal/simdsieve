//! AVX2 intrinsics for 64-byte dual-pumped multi-pattern prefix matching.
//!
#![allow(
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
//! This module implements the AVX2 (256-bit) backend for `x86_64` targets.
//! It processes 64-byte blocks using two 32-byte "pumps" to maximize
//! instruction-level parallelism.
//!
//! # Architecture
//!
//! The AVX2 backend uses `_mm256_cmpeq_epi8` to compare 32 haystack bytes
//! against a broadcast pattern byte in a single instruction. Results are
//! combined using bitmasks from `_mm256_movemask_epi8`.
//!
//! ## Dual-Pump Processing
//!
//! Each 64-byte logical block is processed as two 32-byte halves:
//! 1. Load bytes 0-31 (pump A) and bytes 32-63 (pump B)
//! 2. Interleave comparisons across both pumps
//! 3. Return separate bitmasks for each half
//!
//! This hides load latency by keeping both load ports busy and allowing
//! out-of-order execution to overlap computation.
//!
//! # Safety
//!
//! All `unsafe` blocks in this module require:
//! 1. AVX2 target feature is available (verified at construction).
//! 2. Input blocks have sufficient trailing bytes for multi-byte prefixes.
//! 3. Pointer arithmetic stays within allocated slices.
//!
//! Unaligned loads (`_mm256_loadu_si256`) are safe for any valid pointer.

use crate::fold::fold_ascii_lowercase;
use crate::scalar::{build_mask, pack_word};
use core::arch::x86_64::{
    __m256i, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256,
    _mm256_movemask_epi8, _mm256_set1_epi8, _mm256_sub_epi8,
};

/// A single pattern's prefix, aligned for AVX2 loads.
///
/// The `#[repr(C, align(32))]` ensures the struct can be loaded
/// efficiently with AVX2 aligned load instructions if desired.
///
/// Stores the prefix as a `u32` word (matching `ScalarPattern`) and
/// precomputed AVX2 broadcast vectors so the hot path avoids
/// `_mm256_set1_epi8` per pattern per block.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Avx2Pattern {
    /// Number of valid prefix bytes (0-4).
    len: usize,
    /// Packed pattern word for consistency with scalar backend.
    word: u32,
    /// Mask with 0xFF for each valid byte position.
    mask: u32,
    /// Precomputed broadcast vectors for each prefix byte position.
    bcast: [__m256i; 4],
}

/// AVX2 multi-pattern filter operating on 64-byte blocks.
///
/// Holds up to 8 patterns and produces bitmasks indicating which byte
/// positions in a block match at least one pattern prefix.
#[derive(Clone)]
#[repr(C, align(32))]
pub(crate) struct Avx2Filter {
    /// Loaded patterns (up to 16).
    patterns: [Avx2Pattern; 16],
    /// Number of valid patterns.
    pattern_count: usize,
    /// Maximum prefix length across all patterns (1-4).
    max_len: usize,
    /// Whether to use ASCII case-insensitive matching.
    case_insensitive: bool,
}

impl core::fmt::Debug for Avx2Filter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Avx2Filter")
            .field("pattern_count", &self.pattern_count)
            .field("max_len", &self.max_len)
            .field("case_insensitive", &self.case_insensitive)
            .finish_non_exhaustive()
    }
}

impl Avx2Filter {
    /// Builds the broadcast vectors for a single pattern prefix.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    unsafe fn build_broadcasts(bytes: [u8; 4]) -> [__m256i; 4] {
        [
            _mm256_set1_epi8(bytes[0] as i8),
            _mm256_set1_epi8(bytes[1] as i8),
            _mm256_set1_epi8(bytes[2] as i8),
            _mm256_set1_epi8(bytes[3] as i8),
        ]
    }

    /// Builds an AVX2 filter from up to 8 prefix byte slices.
    ///
    /// Each prefix is truncated to 4 bytes. When `case_insensitive` is
    /// `true`, ASCII `a`-`z` bytes are folded to upper-case.
    ///
    /// # Parameters
    ///
    /// - `prefixes`: Slice of pattern byte slices (max 8, each max 4 bytes).
    /// - `case_insensitive`: Enable ASCII case-insensitive matching.
    ///   Maximum patterns per AVX2 filter. 16 patterns × 4 prefix bytes × 2 pumps
    ///   = 128 comparisons per 64-byte block. Still within AVX2 throughput budget.
    pub(crate) const MAX_PATTERNS: usize = 16;

    /// # Safety
    ///
    /// Caller must ensure AVX2 is available before calling this function.
    pub(crate) unsafe fn new(prefixes: &[&[u8]], case_insensitive: bool) -> Self {
        let mut max_len = 0;
        let count = prefixes.len().min(Self::MAX_PATTERNS);

        // Zero-initialize the array to avoid UB from uninitialized padding or
        // uninitialized elements when count < 16. All-zero is a valid representation
        // for Avx2Pattern, including its __m256i fields.
        let mut patterns: [Avx2Pattern; 16] = unsafe { core::mem::zeroed() };

        for (i, &slice) in prefixes.iter().take(Self::MAX_PATTERNS).enumerate() {
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
            let word = pack_word(arr, eval_len);
            let mask = build_mask(eval_len);
            // SAFETY: AVX2 is required for this module to be used.
            // Caller must have ensured AVX2 is available.
            let bcast = unsafe { Self::build_broadcasts(arr) };
            patterns[i] = Avx2Pattern {
                len: eval_len,
                word,
                mask,
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

    /// Folds ASCII lowercase letters to uppercase in a 256-bit vector.
    ///
    /// Uses the property that ASCII lowercase letters are exactly 0x20
    /// greater than their uppercase counterparts. The comparison uses
    /// range checks against 'a'-1 and 'z' to identify lowercase bytes.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    unsafe fn ascii_fold_vector(v: __m256i) -> __m256i {
        let lower_bound = _mm256_set1_epi8((b'a' - 1) as i8);
        let fold_val = _mm256_set1_epi8(0x20);

        let mask1 = _mm256_cmpgt_epi8(v, lower_bound);
        let upper_limit = _mm256_set1_epi8(b'z' as i8);
        let mask2 = _mm256_cmpgt_epi8(v, upper_limit);

        let v_sub = _mm256_sub_epi8(v, fold_val);
        let is_alpha = core::arch::x86_64::_mm256_andnot_si256(mask2, mask1);

        _mm256_blendv_epi8(v, v_sub, is_alpha)
    }

    /// Scans a 64-byte block, returning per-half bitmasks.
    ///
    /// Returns `(mask_lo, mask_hi)` where bit `i` of `mask_lo` is set
    /// if byte position `i` (0-31) starts with a matching prefix, and
    /// bit `i` of `mask_hi` covers positions 32-63.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `block.len() >= 64 + max_length() - 1` so that offset reads for
    ///   multi-byte prefixes remain in bounds.
    /// - The CPU supports the AVX2 instruction set.
    ///
    /// # Implementation Notes
    ///
    /// The four position-offset vectors (v0, v1, v2, v3) are loaded
    /// conditionally based on `max_len`. This avoids unnecessary loads
    /// for short patterns, improving cache efficiency.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[must_use]
    pub(crate) unsafe fn check_64byte_block(&self, block: &[u8]) -> (u32, u32) {
        debug_assert!(
            block.len() >= 64 + self.max_len.saturating_sub(1),
            "block lacks trailing buffer"
        );

        let mut folded_mask_a: u32 = 0;
        let mut folded_mask_b: u32 = 0;

        // SAFETY: Caller guarantees block is at least 64 + max_len - 1
        // bytes. _mm256_loadu_si256 performs an unaligned 32-byte load
        // which is safe as long as the source range is within the slice.
        unsafe {
            let mut v0_a: __m256i = _mm256_loadu_si256(block.as_ptr().cast());
            let mut v0_b: __m256i = _mm256_loadu_si256(block.as_ptr().add(32).cast());

            if self.case_insensitive {
                v0_a = Self::ascii_fold_vector(v0_a);
                v0_b = Self::ascii_fold_vector(v0_b);
            }

            let mut v1_a = v0_a;
            let mut v1_b = v0_b;
            let mut v2_a = v0_a;
            let mut v2_b = v0_b;
            let mut v3_a = v0_a;
            let mut v3_b = v0_b;

            // Load offset vectors conditionally based on max pattern length.
            // This avoids unnecessary memory accesses for short patterns.
            if self.max_len > 1 {
                // SAFETY: offset 1 and 33 are within bounds because
                // block.len() >= 64 + max_len - 1 >= 64 + 1 = 65.
                let mut v_a = _mm256_loadu_si256(block.as_ptr().add(1).cast());
                let mut v_b = _mm256_loadu_si256(block.as_ptr().add(33).cast());
                if self.case_insensitive {
                    v_a = Self::ascii_fold_vector(v_a);
                    v_b = Self::ascii_fold_vector(v_b);
                }
                v1_a = v_a;
                v1_b = v_b;
            }
            if self.max_len > 2 {
                // SAFETY: offset 2 and 34 are within bounds.
                let mut v_a = _mm256_loadu_si256(block.as_ptr().add(2).cast());
                let mut v_b = _mm256_loadu_si256(block.as_ptr().add(34).cast());
                if self.case_insensitive {
                    v_a = Self::ascii_fold_vector(v_a);
                    v_b = Self::ascii_fold_vector(v_b);
                }
                v2_a = v_a;
                v2_b = v_b;
            }
            if self.max_len > 3 {
                // SAFETY: offset 3 and 35 are within bounds.
                let mut v_a = _mm256_loadu_si256(block.as_ptr().add(3).cast());
                let mut v_b = _mm256_loadu_si256(block.as_ptr().add(35).cast());
                if self.case_insensitive {
                    v_a = Self::ascii_fold_vector(v_a);
                    v_b = Self::ascii_fold_vector(v_b);
                }
                v3_a = v_a;
                v3_b = v_b;
            }

            // Compare each loaded vector position against all pattern prefixes.
            // The mask starts as all 1s and is ANDed with each byte comparison.
            // Only positions matching all prefix bytes survive.
            for p_idx in 0..self.pattern_count {
                let p = &self.patterns[p_idx];
                let mut pattern_mask_a: u32 = !0;
                let mut pattern_mask_b: u32 = !0;

                // Use precomputed broadcast vectors — avoids _mm256_set1_epi8
                // per pattern per block. 32 broadcasts eliminated per call.
                if p.len > 0 {
                    let cmp_a = _mm256_cmpeq_epi8(v0_a, p.bcast[0]);
                    let cmp_b = _mm256_cmpeq_epi8(v0_b, p.bcast[0]);
                    pattern_mask_a &= _mm256_movemask_epi8(cmp_a) as u32;
                    pattern_mask_b &= _mm256_movemask_epi8(cmp_b) as u32;
                }
                if p.len > 1 {
                    let cmp_a = _mm256_cmpeq_epi8(v1_a, p.bcast[1]);
                    let cmp_b = _mm256_cmpeq_epi8(v1_b, p.bcast[1]);
                    pattern_mask_a &= _mm256_movemask_epi8(cmp_a) as u32;
                    pattern_mask_b &= _mm256_movemask_epi8(cmp_b) as u32;
                }
                if p.len > 2 {
                    let cmp_a = _mm256_cmpeq_epi8(v2_a, p.bcast[2]);
                    let cmp_b = _mm256_cmpeq_epi8(v2_b, p.bcast[2]);
                    pattern_mask_a &= _mm256_movemask_epi8(cmp_a) as u32;
                    pattern_mask_b &= _mm256_movemask_epi8(cmp_b) as u32;
                }
                if p.len > 3 {
                    let cmp_a = _mm256_cmpeq_epi8(v3_a, p.bcast[3]);
                    let cmp_b = _mm256_cmpeq_epi8(v3_b, p.bcast[3]);
                    pattern_mask_a &= _mm256_movemask_epi8(cmp_a) as u32;
                    pattern_mask_b &= _mm256_movemask_epi8(cmp_b) as u32;
                }

                // OR across patterns: position matches if it matches ANY pattern.
                folded_mask_a |= pattern_mask_a;
                folded_mask_b |= pattern_mask_b;
            }
        }

        (folded_mask_a, folded_mask_b)
    }

    /// Scans a 32-byte block, returning a single bitmask.
    ///
    /// Bit `i` is set if byte position `i` starts with a matching
    /// pattern prefix.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `block.len() >= 32 + self.max_len.saturating_sub(1)`
    /// - The CPU supports the AVX2 instruction set.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[must_use]
    pub(crate) unsafe fn check_32byte_block(&self, block: &[u8]) -> u32 {
        debug_assert!(
            block.len() >= 32 + self.max_len.saturating_sub(1),
            "block lacks trailing buffer"
        );
        let mut folded_mask: u32 = 0;

        // SAFETY: Caller guarantees sufficient block length for all
        // offset reads. Unaligned loads are safe on AVX2 hardware.
        unsafe {
            let mut v0: __m256i = _mm256_loadu_si256(block.as_ptr().cast());
            if self.case_insensitive {
                v0 = Self::ascii_fold_vector(v0);
            }

            let mut v1 = v0;
            let mut v2 = v0;
            let mut v3 = v0;

            if self.max_len > 1 {
                let mut v = _mm256_loadu_si256(block.as_ptr().add(1).cast());
                if self.case_insensitive {
                    v = Self::ascii_fold_vector(v);
                }
                v1 = v;
            }
            if self.max_len > 2 {
                let mut v = _mm256_loadu_si256(block.as_ptr().add(2).cast());
                if self.case_insensitive {
                    v = Self::ascii_fold_vector(v);
                }
                v2 = v;
            }
            if self.max_len > 3 {
                let mut v = _mm256_loadu_si256(block.as_ptr().add(3).cast());
                if self.case_insensitive {
                    v = Self::ascii_fold_vector(v);
                }
                v3 = v;
            }

            for p_idx in 0..self.pattern_count {
                let p = &self.patterns[p_idx];
                let mut pattern_mask: u32 = !0;
                if p.len > 0 {
                    let cmp = _mm256_cmpeq_epi8(v0, p.bcast[0]);
                    pattern_mask &= _mm256_movemask_epi8(cmp) as u32;
                }
                if p.len > 1 {
                    let cmp = _mm256_cmpeq_epi8(v1, p.bcast[1]);
                    pattern_mask &= _mm256_movemask_epi8(cmp) as u32;
                }
                if p.len > 2 {
                    let cmp = _mm256_cmpeq_epi8(v2, p.bcast[2]);
                    pattern_mask &= _mm256_movemask_epi8(cmp) as u32;
                }
                if p.len > 3 {
                    let cmp = _mm256_cmpeq_epi8(v3, p.bcast[3]);
                    pattern_mask &= _mm256_movemask_epi8(cmp) as u32;
                }
                folded_mask |= pattern_mask;
            }
        }
        folded_mask
    }
}

#[cfg(test)]
mod tests {
    use super::Avx2Filter;
    use crate::scalar::ScalarFilter;

    #[test]
    fn case_insensitive_masks_expose_pump_b_boundary_state() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let filter = unsafe { Avx2Filter::new(&[b"Z"], true) };
        let mut block = [b'x'; 65];
        block[63] = b'Z';

        let (mask_a, mask_b) = unsafe { filter.check_64byte_block(&block) };
        eprintln!("mask_a={mask_a:032b}");
        eprintln!("mask_b={mask_b:032b}");

        assert_eq!(mask_a, 0);
        assert_eq!(mask_b & (1 << 31), 1 << 31);
    }

    #[test]
    fn avx2_64byte_block_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let patterns: &[&[u8]] = &[b"ab", b"XY", b"1"];
        let avx2 = unsafe { Avx2Filter::new(patterns, false) };
        let scalar = ScalarFilter::new(patterns, false);

        let mut block = [b'x'; 68];
        block[10] = b'a';
        block[11] = b'b';
        block[35] = b'X';
        block[36] = b'Y';
        block[63] = b'1';

        let (mask_a, mask_b) = unsafe { avx2.check_64byte_block(&block) };
        let scalar_mask = scalar.check_64byte_block(&block);
        let avx2_mask = u64::from(mask_a) | (u64::from(mask_b) << 32);

        assert_eq!(
            avx2_mask, scalar_mask,
            "AVX2 64-byte block must match scalar backend"
        );
    }

    #[test]
    fn avx2_32byte_block_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let patterns: &[&[u8]] = &[b"te", b"ST"];
        let avx2 = unsafe { Avx2Filter::new(patterns, false) };
        let scalar = ScalarFilter::new(patterns, false);

        // Scalar check_64byte_block needs 64 + max_len - 1 bytes.
        let mut block = [b'x'; 65];
        block[5] = b't';
        block[6] = b'e';
        block[30] = b'S';
        block[31] = b'T';

        let avx2_mask = unsafe { avx2.check_32byte_block(&block) };
        let scalar_mask = scalar.check_64byte_block(&block) as u32;

        assert_eq!(
            avx2_mask, scalar_mask,
            "AVX2 32-byte block must match scalar backend low 32 bits"
        );
    }

    #[test]
    fn avx2_case_insensitive_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let patterns: &[&[u8]] = &[b"Ab", b"z"];
        let avx2 = unsafe { Avx2Filter::new(patterns, true) };
        let scalar = ScalarFilter::new(patterns, true);

        let mut block = [b'x'; 68];
        block[15] = b'a';
        block[16] = b'B';
        block[47] = b'Z';

        let (mask_a, mask_b) = unsafe { avx2.check_64byte_block(&block) };
        let scalar_mask = scalar.check_64byte_block(&block);
        let avx2_mask = u64::from(mask_a) | (u64::from(mask_b) << 32);

        assert_eq!(
            avx2_mask, scalar_mask,
            "AVX2 case-insensitive must match scalar backend"
        );
    }
}
