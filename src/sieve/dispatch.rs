//! Hardware tier routing and block fetching logic.

use crate::SimdSieve;
#[cfg(target_arch = "x86_64")]
use crate::avx2::Avx2Filter;
#[cfg(target_arch = "x86_64")]
use crate::avx512::Avx512Filter;
#[cfg(target_arch = "aarch64")]
use crate::neon::NeonFilter;
use crate::scalar::ScalarFilter;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_MM_HINT_T1, _mm_prefetch};

/// Hardware tier selected at construction time.
pub(crate) enum HardwareTier {
    /// AVX-512 backend: 128-byte blocks, 512-bit vectors.
    #[cfg(target_arch = "x86_64")]
    Avx512(Box<Avx512Filter>),
    /// AVX2 backend: 64-byte blocks, 256-bit vectors.
    #[cfg(target_arch = "x86_64")]
    Avx2(Box<Avx2Filter>),
    /// NEON backend: 64-byte blocks, 128-bit vectors.
    #[cfg(target_arch = "aarch64")]
    Neon(Box<NeonFilter>),
    /// Scalar backend: 64-byte blocks, word-wise comparison.
    Scalar(Box<ScalarFilter>),
}

impl HardwareTier {
    /// Returns the half-block stride for dual-pump tiers.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    pub(crate) fn half_block_stride(&self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx512(_) => 64,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => 32,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => 32,
            Self::Scalar(_) => {
                debug_assert!(false, "scalar backend should never use half_block_stride");
                0
            }
        }
    }
}

impl SimdSieve<'_> {
    /// Advances the scan by one block, filling `current_mask`.
    #[inline]
    #[allow(clippy::too_many_lines)]
    pub(crate) fn fetch_next_chunk(&mut self) -> bool {
        let tail_req = self.max_len.saturating_sub(1);

        self.prefetch_ahead(512);

        match &self.tier {
            #[cfg(target_arch = "x86_64")]
            HardwareTier::Avx512(filter) => {
                while self.offset + 128 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 128 + tail_req];
                    let (mask_a, mask_b) = unsafe { filter.check_128byte_block(chunk) };

                    let base = self.offset;
                    self.offset += 128;

                    if mask_a != 0 || mask_b != 0 {
                        if mask_a != 0 {
                            self.current_mask = mask_a;
                            self.mask_base_offset = base;
                            self.next_mask_cache = mask_b;
                            return true;
                        }
                        self.current_mask = mask_b;
                        self.mask_base_offset = base + 64;
                        return true;
                    }
                    self.prefetch_ahead(512);
                }

                if self.offset + 64 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 64 + tail_req];
                    let mask = unsafe { filter.check_64byte_block(chunk) };

                    let base = self.offset;
                    self.offset += 64;
                    if mask != 0 {
                        self.current_mask = mask;
                        self.mask_base_offset = base;
                        return true;
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            HardwareTier::Avx2(filter) => {
                while self.offset + 64 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 64 + tail_req];
                    let (mask_a, mask_b) = unsafe { filter.check_64byte_block(chunk) };

                    let base = self.offset;
                    self.offset += 64;

                    if mask_a != 0 || mask_b != 0 {
                        if mask_a != 0 {
                            self.current_mask = u64::from(mask_a);
                            self.mask_base_offset = base;
                            self.next_mask_cache = u64::from(mask_b);
                            return true;
                        }
                        self.current_mask = u64::from(mask_b);
                        self.mask_base_offset = base + 32;
                        return true;
                    }
                    self.prefetch_ahead(512);
                }

                if self.offset + 32 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 32 + tail_req];
                    let mask = unsafe { filter.check_32byte_block(chunk) };

                    let base = self.offset;
                    self.offset += 32;
                    if mask != 0 {
                        self.current_mask = u64::from(mask);
                        self.mask_base_offset = base;
                        return true;
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            HardwareTier::Neon(filter) => {
                while self.offset + 64 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 64 + tail_req];
                    let (mask_a, mask_b) = unsafe { filter.check_64byte_block(chunk) };

                    let base = self.offset;
                    self.offset += 64;

                    if mask_a != 0 || mask_b != 0 {
                        if mask_a != 0 {
                            self.current_mask = u64::from(mask_a);
                            self.mask_base_offset = base;
                            self.next_mask_cache = u64::from(mask_b);
                            return true;
                        }
                        self.current_mask = u64::from(mask_b);
                        self.mask_base_offset = base + 32;
                        return true;
                    }
                    self.prefetch_ahead(512);
                }

                if self.offset + 32 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 32 + tail_req];
                    let mask = unsafe { filter.check_32byte_block(chunk) };

                    let base = self.offset;
                    self.offset += 32;
                    if mask != 0 {
                        self.current_mask = u64::from(mask);
                        self.mask_base_offset = base;
                        return true;
                    }
                }
            }
            HardwareTier::Scalar(filter) => {
                while self.offset + 64 + tail_req <= self.haystack.len() {
                    let chunk = &self.haystack[self.offset..self.offset + 64 + tail_req];
                    let mask = filter.check_64byte_block(chunk);

                    let base = self.offset;
                    self.offset += 64;
                    if mask != 0 {
                        self.current_mask = mask;
                        self.mask_base_offset = base;
                        return true;
                    }
                }
            }
        }

        // Note: tail miss alignment sweeping is intentionally omitted because
        // edge matching and offsets are handled transparently by the lengths check.
        false
    }

    /// Issues prefetch hints for upcoming haystack bytes.
    #[inline]
    fn prefetch_ahead(&self, lookahead: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            let base = self.haystack.as_ptr();
            let prefetch_offset = self.offset + lookahead;
            if prefetch_offset < self.haystack.len() {
                unsafe {
                    _mm_prefetch(base.add(prefetch_offset).cast(), _MM_HINT_T1);
                    if prefetch_offset + 64 < self.haystack.len() {
                        _mm_prefetch(base.add(prefetch_offset + 64).cast(), _MM_HINT_T1);
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = lookahead;
    }
}
