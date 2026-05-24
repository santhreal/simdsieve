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
///
/// This enum is marked `#[non_exhaustive]` so that future SIMD backends
/// (e.g., RISC-V V, WASM SIMD128) can be added without breaking downstream
/// match arms.
#[non_exhaustive]
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
    pub(crate) fn half_block_stride(&self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx512(_) => 64,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => 32,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => 32,
            Self::Scalar(_) => {
                unreachable!("scalar backend never sets next_mask_cache")
            }
        }
    }
}

/// Macro for the dual-pump main loop shared by AVX-512, AVX2, and NEON.
macro_rules! dual_pump_main {
    ($self:ident, $check:expr, $block_size:literal, $half_stride:literal) => {
        while $self.offset + $block_size + $self.max_len.saturating_sub(1) <= $self.haystack.len() {
            let chunk = &$self.haystack
                [$self.offset..$self.offset + $block_size + $self.max_len.saturating_sub(1)];
            let (mask_a, mask_b) = $check(chunk);
            let base = $self.offset;
            $self.offset += $block_size;
            if mask_a != 0 || mask_b != 0 {
                if mask_a != 0 {
                    $self.current_mask = mask_a;
                    $self.mask_base_offset = base;
                    $self.next_mask_cache = mask_b;
                } else {
                    $self.current_mask = mask_b;
                    $self.mask_base_offset = base + $half_stride;
                }
                return true;
            }
            $self.prefetch_ahead(512);
        }
    };
}

/// Macro for the single-pump main loop shared by scalar and SIMD tail blocks.
macro_rules! single_pump_block {
    ($self:ident, $check:expr, $block_size:literal) => {
        if $self.offset + $block_size + $self.max_len.saturating_sub(1) <= $self.haystack.len() {
            let chunk = &$self.haystack
                [$self.offset..$self.offset + $block_size + $self.max_len.saturating_sub(1)];
            let mask = $check(chunk);
            let base = $self.offset;
            $self.offset += $block_size;
            if mask != 0 {
                $self.current_mask = mask;
                $self.mask_base_offset = base;
                return true;
            }
        }
    };
}

impl SimdSieve<'_> {
    /// Advances the scan by one block, filling `current_mask`.
    #[inline]
    #[allow(clippy::too_many_lines)]
    pub(crate) fn fetch_next_chunk(&mut self) -> bool {
        self.prefetch_ahead(512);

        match &self.tier {
            #[cfg(target_arch = "x86_64")]
            HardwareTier::Avx512(filter) => {
                dual_pump_main!(
                    self,
                    |chunk| unsafe { filter.check_128byte_block(chunk) },
                    128,
                    64
                );

                single_pump_block!(
                    self,
                    |chunk| unsafe { filter.check_64byte_block(chunk) },
                    64
                );
            }
            #[cfg(target_arch = "x86_64")]
            HardwareTier::Avx2(filter) => {
                dual_pump_main!(
                    self,
                    |chunk| {
                        let (a, b) = unsafe { filter.check_64byte_block(chunk) };
                        (u64::from(a), u64::from(b))
                    },
                    64,
                    32
                );

                single_pump_block!(
                    self,
                    |chunk| u64::from(unsafe { filter.check_32byte_block(chunk) }),
                    32
                );
            }
            #[cfg(target_arch = "aarch64")]
            HardwareTier::Neon(filter) => {
                dual_pump_main!(
                    self,
                    |chunk| {
                        let (a, b) = unsafe { filter.check_64byte_block(chunk) };
                        (u64::from(a), u64::from(b))
                    },
                    64,
                    32
                );

                single_pump_block!(
                    self,
                    |chunk| u64::from(unsafe { filter.check_32byte_block(chunk) }),
                    32
                );
            }
            HardwareTier::Scalar(filter) => {
                while self.offset + 64 + self.max_len.saturating_sub(1) <= self.haystack.len() {
                    let chunk = &self.haystack
                        [self.offset..self.offset + 64 + self.max_len.saturating_sub(1)];
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
        let base = self.haystack.as_ptr();
        let prefetch_offset = self.offset + lookahead;
        if prefetch_offset >= self.haystack.len() {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(base.add(prefetch_offset).cast(), _MM_HINT_T1);
            if prefetch_offset + 64 < self.haystack.len() {
                _mm_prefetch(base.add(prefetch_offset + 64).cast(), _MM_HINT_T1);
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Prefetch using the PRFM instruction via inline assembly.
            // pldl1keep = prefetch for load, L1 cache, keep temporal locality.
            core::arch::asm!(
                "prfm pldl1keep, [{addr}]",
                addr = in(reg) base.add(prefetch_offset),
                options(nostack, preserves_flags)
            );
            if prefetch_offset + 64 < self.haystack.len() {
                core::arch::asm!(
                    "prfm pldl1keep, [{addr}]",
                    addr = in(reg) base.add(prefetch_offset + 64),
                    options(nostack, preserves_flags)
                );
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = lookahead;
        }
    }
}
