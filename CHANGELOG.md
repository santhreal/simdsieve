# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-03-30

### Added

- Initial release of `simdsieve` crate.
- **Hardware backends**:
  - AVX-512 backend: 128-byte blocks using 512-bit vectors
  - AVX2 backend: 64-byte blocks using 256-bit vectors  
  - Scalar backend: Portable 64-byte blocks using word-wise comparison
- **Runtime feature detection**: Auto-selects optimal backend on x86_64
- **Multi-pattern search**: Up to 8 patterns searched simultaneously
- **Case-insensitive mode**: ASCII-only folding (`a`-`z` to uppercase)
- **Streaming iterator**: `Iterator<Item = usize>` with `FusedIterator` support
- **Zero-allocation iteration**: All state on stack, no heap during search
- **Density scoring**: `score_density()` for prefix-hit estimation without verification
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

[Unreleased]: https://github.com/santhsecurity/simdsieve/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/santhsecurity/simdsieve/releases/tag/v0.1.0
