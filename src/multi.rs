//! Multi-pass search support for pattern sets larger than 16 entries.
//!
//! `MultiSieve` batches arbitrary pattern sets into groups of 16, runs one
//! [`SimdSieve`] per group, then merges the sorted candidate
//! streams into a single sorted, deduplicated iterator.

use crate::{CompiledSieve, Result, SimdSieve};
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
/// than 16 patterns to a single underlying sieve.
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
#[derive(Debug)]
pub struct MultiSieve<'a> {
    sieves: Vec<SimdSieve<'a>>,
}

impl<'a> MultiSieve<'a> {
    /// Creates a new exact-match multi-pass sieve.
    ///
    /// Patterns are grouped into chunks of 16 so each chunk can be searched
    /// by a regular [`SimdSieve`] (AVX2 supports up to 16 patterns per filter).
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern set is empty.
    pub fn new(haystack: &'a [u8], patterns: &[&'a [u8]]) -> Result<Self> {
        Self::build(haystack, patterns, false)
    }

    /// Creates a case-insensitive multi-pass sieve (ASCII `a`–`z` only).
    ///
    /// Patterns are grouped into chunks of 16 so each chunk can be searched
    /// by a regular [`SimdSieve`].
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern set is empty.
    pub fn new_case_insensitive(haystack: &'a [u8], patterns: &[&'a [u8]]) -> Result<Self> {
        Self::build(haystack, patterns, true)
    }

    fn build(haystack: &'a [u8], patterns: &[&'a [u8]], case_insensitive: bool) -> Result<Self> {
        if patterns.is_empty() {
            return Err(crate::error::SimdSieveError::EmptyPatternSet);
        }
        for (i, &p) in patterns.iter().enumerate() {
            if p.is_empty() {
                return Err(crate::error::SimdSieveError::EmptyPattern { index: i });
            }
        }
        #[cfg(debug_assertions)]
        debug_assert!(
            patterns.len() <= 1_000_000,
            "patterns list is extremely large, potential for excessive memory allocation"
        );

        // Deduplicate patterns before grouping into 16-element chunks: identical
        // patterns would otherwise inflate the number of SimdSieve structures
        // (extra heap allocations) and add redundant work to the k-way position
        // merge. First-occurrence order is preserved so chunk grouping stays
        // deterministic. (Byte-exact dedup only - case-insensitive folding of
        // near-duplicates is intentionally left to per-pattern verification.)
        let mut seen = std::collections::HashSet::with_capacity(patterns.len());
        let mut unique: Vec<&'a [u8]> = Vec::with_capacity(patterns.len());
        for &p in patterns {
            if seen.insert(p) {
                unique.push(p);
            }
        }

        let mut sieves = Vec::with_capacity(unique.len().div_ceil(16));

        for chunk in unique.chunks(16) {
            let sieve = if case_insensitive {
                SimdSieve::new_case_insensitive(haystack, chunk)?
            } else {
                SimdSieve::new(haystack, chunk)?
            };
            sieves.push(sieve);
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
/// A compiled multi-pass sieve for arbitrary pattern sets (>16 entries).
///
/// Patterns are partitioned into 16-element chunks, each compiled into a
/// [`CompiledSieve`]. Calling [`CompiledMultiSieve::scan`] or
/// [`CompiledMultiSieve::candidates`] rebinds a haystack across all chunks
/// with zero filter recompilation.
#[derive(Debug)]
pub struct CompiledMultiSieve {
    sieves: Vec<CompiledSieve>,
}

impl CompiledMultiSieve {
    /// Compiles an exact-match multi-pass sieve for any number of patterns.
    ///
    /// # Errors
    ///
    /// Returns [`SimdSieveError::EmptyPatternSet`] if `patterns` is empty.
    pub fn new(patterns: &[&[u8]]) -> Result<Self> {
        Self::build(patterns, false)
    }

    /// Compiles a case-insensitive multi-pass sieve for any number of patterns.
    ///
    /// # Errors
    ///
    /// Returns [`SimdSieveError::EmptyPatternSet`] if `patterns` is empty.
    pub fn new_case_insensitive(patterns: &[&[u8]]) -> Result<Self> {
        Self::build(patterns, true)
    }

    fn build(patterns: &[&[u8]], case_insensitive: bool) -> Result<Self> {
        if patterns.is_empty() {
            return Err(crate::error::SimdSieveError::EmptyPatternSet);
        }
        for (i, &p) in patterns.iter().enumerate() {
            if p.is_empty() {
                return Err(crate::error::SimdSieveError::EmptyPattern { index: i });
            }
        }
        let mut seen = std::collections::HashSet::with_capacity(patterns.len());
        let mut unique: Vec<&[u8]> = Vec::with_capacity(patterns.len());
        for &p in patterns {
            if seen.insert(p) {
                unique.push(p);
            }
        }

        let mut sieves = Vec::with_capacity(unique.len().div_ceil(16));
        for chunk in unique.chunks(16) {
            let compiled = if case_insensitive {
                CompiledSieve::new_case_insensitive(chunk)?
            } else {
                CompiledSieve::new(chunk)?
            };
            sieves.push(compiled);
        }

        Ok(Self { sieves })
    }

    /// Scans `haystack` using all compiled sieve groups, returning a [`MultiSieve`].
    pub fn scan<'a>(&'a self, haystack: &'a [u8]) -> MultiSieve<'a> {
        MultiSieve {
            sieves: self.sieves.iter().map(|cs| cs.scan(haystack)).collect(),
        }
    }

    /// Scans `haystack` and streams deduplicated, sorted candidate match offsets.
    pub fn candidates<'a>(&'a self, haystack: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
        self.scan(haystack).candidates()
    }

    /// Returns the total number of compiled sieve chunks (>16 patterns use multiple chunks).
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.sieves.len()
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
                        if let Some(position) = *position
                            && best.is_none_or(|(_, best_pos)| position < best_pos)
                        {
                            best = Some((sieve_index, position));
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
    use super::{CompiledMultiSieve, MultiSieve};

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
            .flat_map(|pattern| pattern.iter().copied().chain(*b"|"))
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

    #[test]
    fn case_insensitive_multi_sieve_matches() {
        let haystack = b"Alpha Beta GAMMA delta";
        let patterns: &[&[u8]] = &[b"alpha", b"GAMMA", b"Delta"];

        let actual: Vec<usize> = MultiSieve::new_case_insensitive(haystack, patterns)
            .unwrap()
            .candidates()
            .collect();

        assert_eq!(actual, vec![0, 11, 17]);
    }
    #[test]
    fn test_compiled_multi_sieve_large_pattern_set_parity() {
        let haystack = b"p00 p05 p10 p15 p20 p25 end";
        let pattern_bufs: Vec<Vec<u8>> = (0..25).map(|i| format!("p{i:02}").into_bytes()).collect();
        let pattern_refs: Vec<&[u8]> = pattern_bufs.iter().map(|v| v.as_slice()).collect();

        let compiled_multi = CompiledMultiSieve::new(&pattern_refs).unwrap();
        assert_eq!(compiled_multi.chunk_count(), 2);

        let direct_matches: Vec<usize> = MultiSieve::new(haystack, &pattern_refs)
            .unwrap()
            .candidates()
            .collect();
        let compiled_matches: Vec<usize> = compiled_multi.candidates(haystack).collect();

        assert_eq!(compiled_matches, direct_matches);
    }

    #[test]
    fn test_compiled_multi_sieve_case_insensitive() {
        let haystack = b"P00 P05 P10 P15 P20 P25 END";
        let pattern_bufs: Vec<Vec<u8>> = (0..20).map(|i| format!("p{i:02}").into_bytes()).collect();
        let pattern_refs: Vec<&[u8]> = pattern_bufs.iter().map(|v| v.as_slice()).collect();

        let compiled_multi = CompiledMultiSieve::new_case_insensitive(&pattern_refs).unwrap();

        let direct_matches: Vec<usize> = MultiSieve::new_case_insensitive(haystack, &pattern_refs)
            .unwrap()
            .candidates()
            .collect();
        let compiled_matches: Vec<usize> = compiled_multi.candidates(haystack).collect();

        assert_eq!(compiled_matches, direct_matches);
    }

    #[test]
    fn test_multi_sieve_empty_pattern_preserves_exact_index() {
        use crate::SimdSieveError;

        // 20 patterns with an empty pattern at index 18
        let mut patterns: Vec<&[u8]> = (0..20).map(|i| match i {
            0..=15 => b"abc" as &[u8],
            16 => b"def",
            17 => b"ghi",
            18 => b"",
            _ => b"jkl",
        }).collect();

        let err = MultiSieve::new(b"haystack", &patterns).unwrap_err();
        assert_eq!(err, SimdSieveError::EmptyPattern { index: 18 });

        let err_compiled = CompiledMultiSieve::new(&patterns).unwrap_err();
        assert_eq!(err_compiled, SimdSieveError::EmptyPattern { index: 18 });
    }
}
