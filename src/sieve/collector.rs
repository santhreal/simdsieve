//! Match collection and deduplication.

use crate::SimdSieve;
use core::iter::FusedIterator;

impl Iterator for SimdSieve<'_> {
    type Item = usize;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.haystack.len().saturating_sub(self.offset)))
    }

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Consume bits from current mask first.
            if self.current_mask != 0 {
                let trailing = self.current_mask.trailing_zeros();
                self.current_mask &= self.current_mask - 1;
                let potential_idx = self.mask_base_offset + (trailing as usize);

                let mut exact_match = false;
                for p_idx in 0..self.pattern_count {
                    let vp = self.verification_patterns[p_idx];
                    let remain_len = self.haystack.len() - potential_idx;

                    if remain_len >= vp.len() {
                        let potential_slice =
                            &self.haystack[potential_idx..potential_idx + vp.len()];
                        if (self.verifier)(potential_slice, vp) {
                            exact_match = true;
                            break;
                        }
                    }
                }

                if exact_match {
                    return Some(potential_idx);
                }
                continue;
            }

            // Switch to cached second-half mask if available.
            if self.next_mask_cache != 0 {
                self.current_mask = self.next_mask_cache;
                self.next_mask_cache = 0;
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                {
                    self.mask_base_offset += self.tier.half_block_stride();
                }
                continue;
            }

            // Fetch next block from hardware tier.
            if !self.fetch_next_chunk() {
                break;
            }
        }

        // Byte-by-byte tail scan for remaining positions.
        while self.offset <= self.haystack.len() {
            let current_idx = self.offset;
            let remaining = &self.haystack[self.offset..];
            self.offset += 1;

            for p_idx in 0..self.pattern_count {
                let vp = self.verification_patterns[p_idx];
                if remaining.len() >= vp.len() && (self.verifier)(&remaining[..vp.len()], vp) {
                    return Some(current_idx);
                }
            }
        }

        None
    }
}

impl FusedIterator for SimdSieve<'_> {}
