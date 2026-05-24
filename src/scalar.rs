//! Multi-pattern scalar fallback for platforms without SIMD support.
//!
//! This module provides a portable backend that produces identical results
//! to the SIMD backends using only standard Rust operations. It uses
//! word-wise `u32` comparison to avoid the O(positions x patterns x `prefix_len`)
//! byte-by-byte inner loop.
//!
//! # Algorithm
//!
//! Each pattern prefix is packed into a `u32` with a corresponding mask:
//! - Pattern "abc" (3 bytes) -> word: `0x00636261`, mask: `0x00FFFFFF`
//! - Pattern "xy" (2 bytes) -> word: `0x00007978`, mask: `0x0000FFFF`
//!
//! For each position in the block:
//! 1. Load 4 bytes from the haystack as a `u32` (little-endian).
//! 2. Optionally fold ASCII lowercase to uppercase.
//! 3. XOR with pattern word, AND with mask.
//! 4. If result is 0, we have a prefix match.
//!
//! This reduces 8 patterns x 4 bytes = 32 comparisons to 8 `u32` operations.

// SAFETY: this module is one of the crate's unsafe surfaces. The hot-path
// unchecked loads in `check_64byte_block` are guarded by an explicit slice-length
// precondition that proves every accessed byte is in bounds.

use crate::fold::{fold_ascii_lowercase, fold_byte};

/// A single pattern's prefix packed into a `u32` word for word-wise comparison.
///
/// Only the first `len` bytes are meaningful. The mask has `0xFF` in each
/// valid byte position and `0x00` in unused positions.
#[derive(Debug, Clone, Copy)]
struct ScalarPattern {
    /// Number of valid bytes in `bytes` (0-4).
    len: usize,
    /// Packed pattern word for word-wise comparison.
    word: u32,
    /// Mask with 0xFF for each valid byte position.
    mask: u32,
}

impl Default for ScalarPattern {
    #[inline]
    fn default() -> Self {
        Self {
            len: 0,
            word: 0,
            mask: 0,
        }
    }
}

/// Portable multi-pattern filter using word-wise comparison.
///
/// This is the fallback backend for platforms without AVX2/AVX-512 support.
/// It produces identical bitmask results to the SIMD backends, enabling
/// seamless cross-backend parity.
///
/// ## Performance
///
/// The naive approach would be O(positions x patterns x `prefix_len`) byte
/// comparisons. Using word-wise comparison, each position is checked against
/// each pattern with a single `u32` masked comparison. For 8 patterns with
/// 4-byte prefixes, this reduces inner loop work from 2048 operations to
/// 512 per 64-byte block.
#[derive(Debug, Clone)]
pub(crate) struct ScalarFilter {
    /// Loaded patterns (up to 16).
    patterns: [ScalarPattern; 16],
    /// Number of valid patterns.
    pattern_count: usize,
    /// Whether to use case-insensitive matching.
    case_insensitive: bool,
}

/// Packs up to 4 bytes into a little-endian `u32` word.
#[inline]
pub(crate) fn pack_word(bytes: [u8; 4], len: usize) -> u32 {
    let mut word = 0u32;
    // Unrolled: compiler will elide branches for constant len values.
    if len >= 1 {
        word |= u32::from(bytes[0]);
    }
    if len >= 2 {
        word |= u32::from(bytes[1]) << 8;
    }
    if len >= 3 {
        word |= u32::from(bytes[2]) << 16;
    }
    if len >= 4 {
        word |= u32::from(bytes[3]) << 24;
    }
    word
}

/// Builds a mask with `0xFF` for each valid byte position.
#[inline]
pub(crate) const fn build_mask(len: usize) -> u32 {
    match len {
        0 => 0,
        1 => 0x0000_00FF,
        2 => 0x0000_FFFF,
        3 => 0x00FF_FFFF,
        _ => 0xFFFF_FFFF,
    }
}

/// Folds ASCII lowercase to uppercase in a packed `u32` word.
///
/// Operates byte-by-byte using the branchless `fold_byte` helper.
/// Non-ASCII bytes and non-letter bytes are unaffected.
#[inline]
fn fold_case_u32(word: u32) -> u32 {
    let bytes = word.to_le_bytes();
    u32::from_le_bytes([
        fold_byte(bytes[0]),
        fold_byte(bytes[1]),
        fold_byte(bytes[2]),
        fold_byte(bytes[3]),
    ])
}

/// Loads up to 4 bytes from `block` at `offset` into a little-endian `u32`.
/// Missing bytes are zero-filled, which is safe because they are masked out
/// by the pattern mask.
#[inline]
fn load_u32_safe(block: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    let remaining = block.len().saturating_sub(offset);
    let to_copy = remaining.min(4);
    bytes[..to_copy].copy_from_slice(&block[offset..offset + to_copy]);
    u32::from_le_bytes(bytes)
}

impl ScalarFilter {
    /// Builds a scalar filter from up to 16 prefix slices.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `prefixes.len() > MAX_PATTERNS`.
    #[must_use]
    pub(crate) fn new(prefixes: &[&[u8]], case_insensitive: bool) -> Self {
        let mut ptrns = [ScalarPattern::default(); 16];
        debug_assert!(
            prefixes.len() <= crate::MAX_PATTERNS,
            "scalar filter given {} prefixes, max is {}",
            prefixes.len(),
            crate::MAX_PATTERNS
        );
        let count = prefixes.len();

        for (i, &slice) in prefixes.iter().take(crate::MAX_PATTERNS).enumerate() {
            let eval_len = slice.len().min(4);
            let mut arr = [0u8; 4];

            for j in 0..eval_len {
                arr[j] = if case_insensitive {
                    fold_ascii_lowercase(slice[j])
                } else {
                    slice[j]
                };
            }
            let word = pack_word(arr, eval_len);
            let mask = build_mask(eval_len);
            ptrns[i] = ScalarPattern {
                len: eval_len,
                word,
                mask,
            };
        }

        Self {
            patterns: ptrns,
            pattern_count: count,
            case_insensitive,
        }
    }

    /// Checks a 64-byte block for prefix matches, returning a bitmask.
    ///
    /// Bit `i` is set if position `i` in the block starts with at least
    /// one of the loaded pattern prefixes.
    ///
    /// Uses word-wise `u32` comparison: each position loads a `u32` from
    /// the block, optionally folds case, masks to the pattern length, and
    /// compares in one operation.
    ///
    /// # Preconditions
    ///
    /// For full multi-byte prefix coverage, the block should contain at least
    /// `64 + max_len - 1` bytes. The function is defensive and skips patterns
    /// or positions that would read out of bounds.
    #[must_use]
    #[inline]
    pub(crate) fn check_64byte_block(&self, block: &[u8]) -> u64 {
        let mut mask: u64 = 0;
        for p_idx in 0..self.pattern_count {
            let p = &self.patterns[p_idx];
            if block.len() < 64 + p.len.saturating_sub(1) {
                continue;
            }

            let pat_word = p.word;
            let pat_mask = p.mask;
            let ci = self.case_insensitive;

            let mut i = 0;
            while i + 6 < block.len() && i < 64 {
                // Safety: i + 6 < block.len() guarantees every accessed index is in bounds.
                let loaded0 = unsafe {
                    u32::from_le_bytes([
                        *block.get_unchecked(i),
                        *block.get_unchecked(i + 1),
                        *block.get_unchecked(i + 2),
                        *block.get_unchecked(i + 3),
                    ])
                };
                let loaded1 = unsafe {
                    u32::from_le_bytes([
                        *block.get_unchecked(i + 1),
                        *block.get_unchecked(i + 2),
                        *block.get_unchecked(i + 3),
                        *block.get_unchecked(i + 4),
                    ])
                };
                let loaded2 = unsafe {
                    u32::from_le_bytes([
                        *block.get_unchecked(i + 2),
                        *block.get_unchecked(i + 3),
                        *block.get_unchecked(i + 4),
                        *block.get_unchecked(i + 5),
                    ])
                };
                let loaded3 = unsafe {
                    u32::from_le_bytes([
                        *block.get_unchecked(i + 3),
                        *block.get_unchecked(i + 4),
                        *block.get_unchecked(i + 5),
                        *block.get_unchecked(i + 6),
                    ])
                };

                let folded0 = if ci { fold_case_u32(loaded0) } else { loaded0 };
                let folded1 = if ci { fold_case_u32(loaded1) } else { loaded1 };
                let folded2 = if ci { fold_case_u32(loaded2) } else { loaded2 };
                let folded3 = if ci { fold_case_u32(loaded3) } else { loaded3 };

                if (folded0 ^ pat_word) & pat_mask == 0 {
                    mask |= 1 << i;
                }
                if (folded1 ^ pat_word) & pat_mask == 0 {
                    mask |= 1 << (i + 1);
                }
                if (folded2 ^ pat_word) & pat_mask == 0 {
                    mask |= 1 << (i + 2);
                }
                if (folded3 ^ pat_word) & pat_mask == 0 {
                    mask |= 1 << (i + 3);
                }

                i += 4;
            }

            while i < block.len().min(64) {
                let loaded = load_u32_safe(block, i);
                let folded = if ci { fold_case_u32(loaded) } else { loaded };
                if (folded ^ pat_word) & pat_mask == 0 {
                    mask |= 1 << i;
                }
                i += 1;
            }
        }
        mask
    }
}
