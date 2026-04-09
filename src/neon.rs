//! NEON intrinsics for multi-pattern prefix matching.
//!
//! This module implements the AArch64 (NEON) backend for ARM processors.

#![allow(
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#![cfg(target_arch = "aarch64")]

use crate::fold::fold_ascii_lowercase;
use crate::scalar::{build_mask, pack_word};

use core::arch::aarch64::*;

/// A single pattern's prefix packed for NEON vector comparison.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct NeonPattern {
    len: usize,
    word: u32,
    mask: u32,
    bcast: [uint8x16_t; 4],
}

/// NEON multi-pattern filter operating on 64-byte blocks.
#[derive(Clone)]
#[repr(C, align(16))]
pub(crate) struct NeonFilter {
    patterns: [NeonPattern; 16],
    pattern_count: usize,
    max_len: usize,
    case_insensitive: bool,
}

impl core::fmt::Debug for NeonFilter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NeonFilter")
            .field("pattern_count", &self.pattern_count)
            .field("max_len", &self.max_len)
            .field("case_insensitive", &self.case_insensitive)
            .finish_non_exhaustive()
    }
}

impl NeonFilter {
    pub(crate) const MAX_PATTERNS: usize = 16;

    /// Builds the broadcast vectors for a single pattern prefix.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn build_broadcasts(bytes: [u8; 4]) -> [uint8x16_t; 4] {
        [
            vdupq_n_u8(bytes[0]),
            vdupq_n_u8(bytes[1]),
            vdupq_n_u8(bytes[2]),
            vdupq_n_u8(bytes[3]),
        ]
    }

    /// Builds a NEON filter from up to 16 prefix byte slices.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available before calling this function.
    pub(crate) unsafe fn new(prefixes: &[&[u8]], case_insensitive: bool) -> Self {
        let mut max_len = 0;
        let count = prefixes.len().min(Self::MAX_PATTERNS);
        let mut patterns: [NeonPattern; 16] = unsafe { core::mem::zeroed() };

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
            let bcast = unsafe { Self::build_broadcasts(arr) };
            patterns[i] = NeonPattern {
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

    /// Folds ASCII lowercase letters to uppercase in a 128-bit vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn ascii_fold_vector(v: uint8x16_t) -> uint8x16_t {
        let lower_bound = vdupq_n_u8(b'a' - 1);
        let upper_limit = vdupq_n_u8(b'z' + 1);
        let fold_val = vdupq_n_u8(0x20);

        let mask1 = vcgtq_u8(v, lower_bound);
        let mask2 = vcltq_u8(v, upper_limit);
        let is_alpha = vandq_u8(mask1, mask2);

        let v_sub = vsubq_u8(v, fold_val);
        vbslq_u8(is_alpha, v_sub, v)
    }

    /// Computes a 16-bit movemask from a NEON vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available.
    #[target_feature(enable = "neon")]
    #[inline(always)]
    unsafe fn neon_movemask(v: uint8x16_t) -> u16 {
        let bit_weights = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let weights = vld1q_u8(bit_weights.as_ptr());
        let tmp = vandq_u8(v, weights);

        let tmp16 = vpaddlq_u8(tmp);
        let tmp32 = vpaddlq_u16(tmp16);
        let tmp64 = vpaddlq_u32(tmp32);

        let lo = vgetq_lane_u64(tmp64, 0);
        let hi = vgetq_lane_u64(tmp64, 1);

        #[allow(clippy::cast_possible_truncation)]
        let mask = (lo as u16) | ((hi as u16) << 8);
        mask
    }

    /// Scans a 64-byte block, returning per-half bitmasks.
    ///
    /// Returns `(mask_lo, mask_hi)` where bit `i` of `mask_lo` covers
    /// byte positions 0-31 and bit `i` of `mask_hi` covers 32-63.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `block.len() >= 64 + self.max_len.saturating_sub(1)`
    /// - The CPU supports NEON instructions.
    #[target_feature(enable = "neon")]
    #[inline]
    #[must_use]
    pub(crate) unsafe fn check_64byte_block(&self, block: &[u8]) -> (u32, u32) {
        debug_assert!(
            block.len() >= 64 + self.max_len.saturating_sub(1),
            "block lacks trailing buffer"
        );

        let mask_a = unsafe { self.check_32byte_block(block) };
        let mask_b = unsafe { self.check_32byte_block(&block[32..]) };
        (mask_a, mask_b)
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
    /// - The CPU supports NEON instructions.
    #[target_feature(enable = "neon")]
    #[inline]
    #[must_use]
    pub(crate) unsafe fn check_32byte_block(&self, block: &[u8]) -> u32 {
        debug_assert!(
            block.len() >= 32 + self.max_len.saturating_sub(1),
            "block lacks trailing buffer"
        );
        let mut folded_mask: u32 = 0;

        unsafe {
            let mut v0_lo: uint8x16_t = vld1q_u8(block.as_ptr());
            let mut v0_hi: uint8x16_t = vld1q_u8(block.as_ptr().add(16));

            if self.case_insensitive {
                v0_lo = Self::ascii_fold_vector(v0_lo);
                v0_hi = Self::ascii_fold_vector(v0_hi);
            }

            let mut v1_lo = v0_lo;
            let mut v1_hi = v0_hi;
            let mut v2_lo = v0_lo;
            let mut v2_hi = v0_hi;
            let mut v3_lo = v0_lo;
            let mut v3_hi = v0_hi;

            if self.max_len > 1 {
                let mut v_lo = vld1q_u8(block.as_ptr().add(1));
                let mut v_hi = vld1q_u8(block.as_ptr().add(17));
                if self.case_insensitive {
                    v_lo = Self::ascii_fold_vector(v_lo);
                    v_hi = Self::ascii_fold_vector(v_hi);
                }
                v1_lo = v_lo;
                v1_hi = v_hi;
            }
            if self.max_len > 2 {
                let mut v_lo = vld1q_u8(block.as_ptr().add(2));
                let mut v_hi = vld1q_u8(block.as_ptr().add(18));
                if self.case_insensitive {
                    v_lo = Self::ascii_fold_vector(v_lo);
                    v_hi = Self::ascii_fold_vector(v_hi);
                }
                v2_lo = v_lo;
                v2_hi = v_hi;
            }
            if self.max_len > 3 {
                let mut v_lo = vld1q_u8(block.as_ptr().add(3));
                let mut v_hi = vld1q_u8(block.as_ptr().add(19));
                if self.case_insensitive {
                    v_lo = Self::ascii_fold_vector(v_lo);
                    v_hi = Self::ascii_fold_vector(v_hi);
                }
                v3_lo = v_lo;
                v3_hi = v_hi;
            }

            for p_idx in 0..self.pattern_count {
                let p = &self.patterns[p_idx];
                let mut p_mask_lo: u32 = !0;
                let mut p_mask_hi: u32 = !0;

                if p.len > 0 {
                    p_mask_lo &= u32::from(Self::neon_movemask(vceqq_u8(v0_lo, p.bcast[0])));
                    p_mask_hi &= u32::from(Self::neon_movemask(vceqq_u8(v0_hi, p.bcast[0])));
                }
                if p.len > 1 {
                    p_mask_lo &= u32::from(Self::neon_movemask(vceqq_u8(v1_lo, p.bcast[1])));
                    p_mask_hi &= u32::from(Self::neon_movemask(vceqq_u8(v1_hi, p.bcast[1])));
                }
                if p.len > 2 {
                    p_mask_lo &= u32::from(Self::neon_movemask(vceqq_u8(v2_lo, p.bcast[2])));
                    p_mask_hi &= u32::from(Self::neon_movemask(vceqq_u8(v2_hi, p.bcast[2])));
                }
                if p.len > 3 {
                    p_mask_lo &= u32::from(Self::neon_movemask(vceqq_u8(v3_lo, p.bcast[3])));
                    p_mask_hi &= u32::from(Self::neon_movemask(vceqq_u8(v3_hi, p.bcast[3])));
                }

                folded_mask |= p_mask_lo | (p_mask_hi << 16);
            }
        }
        folded_mask
    }
}
