//! Multi-pass search support for pattern sets larger than eight entries.
//!
//! `MultiSieve` batches arbitrary pattern sets into groups of eight, runs one
//! [`SimdSieve`] per group, then merges the sorted candidate
//! streams into a single sorted, deduplicated iterator.

use crate::{Result, SimdSieve};
use core::cmp::{Ordering, Reverse};
use core::iter::FusedIterator;
use std::collections::BinaryHeap;

/// A multi-pass sieve that supports any number of patterns.
///
/// Internally, patterns are partitioned into groups of at most 16 entries
/// so each group can reuse the existing [`SimdSieve`]
/// implementation. Candidate offsets from every group are then merged with a
/// k-way merge, preserving ascending order and removing duplicates.
///
/// # Errors
///
/// Returns the same construction errors as [`SimdSieve::new`]. In practice,
/// only an empty pattern set can fail because `MultiSieve` never forwards more
/// than eight patterns to a single underlying sieve.
///
/// # Example
///
/// ```
/// use simdsieve::MultiSieve;
///
/// let haystack = b"alpha beta gamma delta";
/// let patterns: &[&[u8]] = &[b"alpha", b"beta", b"gamma", b"delta"];
///
/// let matches: Vec<usize> = MultiSieve::new(haystack, patterns)
///     .unwrap()
///     .candidates()
///     .collect();
///
/// assert_eq!(matches, vec![0, 6, 11, 17]);
/// ```
pub struct MultiSieve<'a> {
    sieves: Vec<SimdSieve<'a>>,
}

impl<'a> MultiSieve<'a> {
    /// Creates a multi-pass sieve from any number of patterns.
    ///
    /// Patterns are grouped into chunks of 16 so each chunk can be searched
    /// by a regular [`SimdSieve`] (AVX2 supports up to 16 patterns per filter).
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern set is empty.
    pub fn new(haystack: &'a [u8], patterns: &[&'a [u8]]) -> Result<Self> {
        if patterns.is_empty() {
            return Err(crate::error::SimdSieveError::EmptyPatternSet);
        }

        #[cfg(debug_assertions)]
        debug_assert!(
            patterns.len() <= 1_000_000,
            "patterns list is extremely large, potential for excessive memory allocation"
        );
        let mut sieves = Vec::with_capacity(patterns.len().div_ceil(16));

        for chunk in patterns.chunks(16) {
            sieves.push(SimdSieve::new(haystack, chunk)?);
        }

        Ok(Self { sieves })
    }

    /// Iterates candidate positions from all pattern groups in sorted order.
    ///
    /// If multiple groups report the same position, that offset is yielded only
    /// once.
    pub fn candidates(self) -> impl Iterator<Item = usize> + 'a {
        MultiCandidates::new(self.sieves)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapEntry {
    position: usize,
    sieve_index: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.position
            .cmp(&other.position)
            .then_with(|| self.sieve_index.cmp(&other.sieve_index))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

enum MergeState {
    Two([Option<usize>; 2]),
    Three([Option<usize>; 3]),
    Heap(BinaryHeap<Reverse<HeapEntry>>),
}

struct MultiCandidates<'a> {
    sieves: Vec<SimdSieve<'a>>,
    state: MergeState,
    last_yielded: Option<usize>,
}

impl<'a> MultiCandidates<'a> {
    fn new(mut sieves: Vec<SimdSieve<'a>>) -> Self {
        let mut current = Vec::with_capacity(sieves.len());
        for sieve in &mut sieves {
            current.push(sieve.next());
        }

        let state = match sieves.len() {
            2 => MergeState::Two([current[0], current[1]]),
            3 => MergeState::Three([current[0], current[1], current[2]]),
            _ => {
                let mut heap = BinaryHeap::with_capacity(sieves.len());
                for (sieve_index, position) in current.into_iter().enumerate() {
                    if let Some(position) = position {
                        heap.push(Reverse(HeapEntry {
                            position,
                            sieve_index,
                        }));
                    }
                }
                MergeState::Heap(heap)
            }
        };

        Self {
            sieves,
            state,
            last_yielded: None,
        }
    }
}

impl Iterator for MultiCandidates<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = match &mut self.state {
                MergeState::Two(vals) => match (vals[0], vals[1]) {
                    (None, None) => None,
                    (Some(a), None) => Some(HeapEntry {
                        position: a,
                        sieve_index: 0,
                    }),
                    (None, Some(b)) => Some(HeapEntry {
                        position: b,
                        sieve_index: 1,
                    }),
                    (Some(a), Some(b)) => {
                        if a <= b {
                            Some(HeapEntry {
                                position: a,
                                sieve_index: 0,
                            })
                        } else {
                            Some(HeapEntry {
                                position: b,
                                sieve_index: 1,
                            })
                        }
                    }
                },
                MergeState::Three(vals) => {
                    let mut best = None;
                    for (sieve_index, position) in vals.iter().enumerate() {
                        if let Some(position) = *position {
                            if best.is_none_or(|(_, best_pos)| position < best_pos) {
                                best = Some((sieve_index, position));
                            }
                        }
                    }
                    best.map(|(sieve_index, position)| HeapEntry {
                        position,
                        sieve_index,
                    })
                }
                MergeState::Heap(heap) => heap.pop().map(|Reverse(e)| e),
            };

            let entry = entry?;

            let next_position = self.sieves[entry.sieve_index].next();
            match &mut self.state {
                MergeState::Two(vals) => vals[entry.sieve_index] = next_position,
                MergeState::Three(vals) => vals[entry.sieve_index] = next_position,
                MergeState::Heap(heap) => {
                    if let Some(position) = next_position {
                        heap.push(Reverse(HeapEntry {
                            position,
                            sieve_index: entry.sieve_index,
                        }));
                    }
                }
            }

            if self.last_yielded == Some(entry.position) {
                continue;
            }

            self.last_yielded = Some(entry.position);
            return Some(entry.position);
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl FusedIterator for MultiCandidates<'_> {}

#[cfg(test)]
mod tests {
    use super::MultiSieve;

    fn naive_matches(haystack: &[u8], patterns: &[&[u8]]) -> Vec<usize> {
        let mut positions = Vec::new();

        for start in 0..=haystack.len() {
            if patterns.iter().any(|pattern| {
                haystack.get(start..start.saturating_add(pattern.len())) == Some(*pattern)
            }) {
                positions.push(start);
            }
        }

        positions
    }

    fn numbered_patterns(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|idx| format!("PATTERN_{idx:03}").into_bytes())
            .collect()
    }

    fn build_refs(patterns: &[Vec<u8>]) -> Vec<&[u8]> {
        patterns.iter().map(Vec::as_slice).collect()
    }

    #[test]
    fn sixteen_patterns_work_correctly() {
        let owned_patterns = numbered_patterns(16);
        let pattern_refs = build_refs(&owned_patterns);
        let haystack = owned_patterns
            .iter()
            .flat_map(|pattern| pattern.iter().copied().chain([b'|']))
            .collect::<Vec<u8>>();

        let actual: Vec<usize> = MultiSieve::new(&haystack, &pattern_refs)
            .unwrap()
            .candidates()
            .collect();

        assert_eq!(actual, naive_matches(&haystack, &pattern_refs));
    }

    #[test]
    fn hundred_patterns_work_correctly() {
        let owned_patterns = numbered_patterns(100);
        let pattern_refs = build_refs(&owned_patterns);
        let haystack = owned_patterns
            .iter()
            .enumerate()
            .flat_map(|(idx, pattern)| {
                pattern
                    .iter()
                    .copied()
                    .chain([b'-', b'0' + (idx % 10) as u8, b'|'])
            })
            .collect::<Vec<u8>>();

        let actual: Vec<usize> = MultiSieve::new(&haystack, &pattern_refs)
            .unwrap()
            .candidates()
            .collect();

        assert_eq!(actual, naive_matches(&haystack, &pattern_refs));
    }

    #[test]
    fn results_match_naive_scan() {
        let haystack = b"aba|secret|hash|aba|needle|secret|hash";
        let patterns: &[&[u8]] = &[
            b"aba",
            b"secret",
            b"hash",
            b"needle",
            b"ret",
            b"ash",
            b"a|s",
            b"ecr",
            b"hash|",
            b"|aba",
            b"needle|secret",
            b"missing",
        ];

        let actual: Vec<usize> = MultiSieve::new(haystack, patterns)
            .unwrap()
            .candidates()
            .collect();

        assert_eq!(actual, naive_matches(haystack, patterns));
    }

    #[test]
    fn positions_are_deduplicated_and_sorted() {
        let haystack = b"token-01 token-02 token-03";
        let patterns: &[&[u8]] = &[
            b"token-01",
            b"token-02",
            b"token-03",
            b"token-01",
            b"token-02",
            b"token-03",
            b"token",
            b"token",
            b"token-01",
            b"token-02",
            b"token-03",
            b"token",
        ];

        let actual: Vec<usize> = MultiSieve::new(haystack, patterns)
            .unwrap()
            .candidates()
            .collect();

        assert_eq!(actual, vec![0, 9, 18]);
    }
}
