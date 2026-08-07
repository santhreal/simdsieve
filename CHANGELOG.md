# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.5] - 2026-08-07

### Fixed

- **Upfront pattern validation in `MultiSieve` and `CompiledMultiSieve`**: Added pre-chunking/pre-dedup pattern validation loop to ensure empty patterns (`b""`) in multi-pattern sets immediately fail-closed with the exact input index in `SimdSieveError::EmptyPattern { index }`.
- **`Debug` trait implementations**: Derived `Debug` for `SimdSieve` and `MultiSieve` structs to enable debug formatting and result unwrapping in tests.
## [0.1.4] - 2026-08-07

### Changed
- Yanked crates.io `0.1.1` (broken aarch64 `inline(always)` + `target_feature` E0658). Main-thread release hygiene.
### Added

- **`CompiledSieve` API**: Added public compile-once pattern sieve for scanning multiple haystacks with zero heap allocation and zero filter recompilation per scan ([#11](https://github.com/santhreal/simdsieve/issues/11)).
- **`CompiledMultiSieve` API**: Added public compile-once multi-pass pattern sieve for arbitrary pattern sets (>16 entries) with zero filter recompilation across haystacks.


## [0.1.2] - 2026-06-27

### Fixed

- **ARM/NEON stable build (E0658)**: the NEON `neon_movemask` helper carried
  `#[inline(always)]` alongside `#[target_feature(enable = "neon")]`, a
  combination that requires the unstable `target_feature_11` feature and so
  failed to compile on stable rustc for `aarch64` targets (E0658). The
  attribute is now plain `#[inline]`, which still hints the optimizer and
  builds on stable. Supersedes the broken `0.1.1` publish, which only ever
  compiled on x86_64.

## [0.1.0] - 2024-03-30

### Added

- Initial release of `simdsieve` crate.
- **Hardware backends**:
  - AVX-512 backend: 128-byte blocks using 512-bit vectors
  - AVX2 backend: 64-byte blocks using 256-bit vectors  
  - Scalar backend: Portable 64-byte blocks using word-wise comparison
- **Runtime feature detection**: Auto-selects optimal backend on x86_64
- **Multi-pattern search**: Up to 16 patterns searched simultaneously
- **Case-insensitive mode**: ASCII-only folding (`a`-`z` to uppercase)
- **Streaming iterator**: `Iterator<Item = usize>` with `FusedIterator` support
- **Zero-allocation iteration**: All state on stack, no heap during search
- **Density estimation**: `estimate_match_count()` for prefix-hit estimation without verification
- **Prefetch hints**: Automatic 512-byte lookahead prefetching on x86_64
- **Comprehensive documentation**:
  - Module-level documentation for all source files
  - Every public item has doc comments with examples
  - SAFETY comments on all unsafe blocks
  - Architecture and extension guide in README
- **Testing**:
  - 30+ adversarial tests covering boundary conditions
  - Property-based tests with proptest
  - Fuzz targets for random input validation
- **Benchmarks**: Criterion-based throughput benchmarks
- **CI/CD**: GitHub Actions with test, clippy, fmt, doc, and MSRV checks

### Performance

- AVX-512: >50 GB/s for single-byte patterns on modern hardware
- AVX2: >25 GB/s for single-byte patterns
- Scalar: >2 GB/s portable fallback

### Safety

- `#[forbid(unsafe_op_in_unsafe_fn)]` at crate level
- All unsafe code confined to backend modules
- Bounds-checked block processing
- Unaligned loads used throughout for memory safety

[Unreleased]: https://github.com/santhreal/simdsieve/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/santhreal/simdsieve/releases/tag/v0.1.2
[0.1.0]: https://github.com/santhreal/simdsieve/releases/tag/v0.1.0
