# simdsieve: Technical Spec

## Overview

SIMD-accelerated byte pattern pre-filtering.  `simdsieve` scans a byte haystack for multiple fixed-string patterns at once and yields verified match offsets through a streaming iterator.  # SIMD Prefiltering  The engine uses a multi-stage prefiltering approach. First, it extracts candidate offsets by searching for the first 1-4 bytes (the "prefix") of each pattern using SIMD vector instructions. This allows scanning dozens of gigabytes per second because the hardware can compare 32 or 64 bytes simultaneously.  Once a prefix hit is found in a SIMD register, the engine performs a lightweight verification of the full pattern at that offset.  # Supported Architectures  The crate includes specialized backends for different CPU architectures:  - **AVX-512 (`x86_64`):** Uses 512-bit ZMM registers for maximum throughput on modern Intel and AMD processors. - **AVX2 (`x86_64`):** Uses 256-bit YMM registers for broad compatibility across most `x86_64` hardware. - **NEON (AArch64):** Uses 128-bit vector registers on ARM processors, optimized for Apple Silicon and Graviton. - **Scalar (Any):** A portable fallback implementation for architectures without specialized SIMD support.  The backend is selected automatically at runtime based on the host CPU's capabilities.  # Example  ```rust use simdsieve::SimdSieve;  let haystack = b"GET /admin HTTP/1.1\r\nHost: example\r\n"; let patterns: &[&[u8]] = &[b"GET", b"/admin"];  let matches: Vec<usize> = SimdSieve::new(haystack, patterns) .unwrap() .collect();  assert_eq!(matches, vec![0, 4]); ```

## Architecture

The crate is organized into the following public modules:

- `error`
- `fold`
- `multi`

## Guarantees

- `#![forbid(unsafe_code)]` where applicable; see `src/lib.rs` for the exact lint preamble.
- All public types have doc comments.
- Error messages are actionable where applicable.

## Public API Summary

Key entry points are exported from `src/lib.rs`:

- [`SimdSieve`]: Streaming single-pass SIMD candidate iterator for up to 16 patterns.
- [`CompiledSieve`]: Compile-once SIMD filter for up to 16 patterns; zero heap allocation on haystack rebinds (`.scan(haystack)`).
- [`MultiSieve`]: Streaming multi-pass candidate iterator with $k$-way merge for pattern sets larger than 16.
- [`CompiledMultiSieve`]: Compile-once multi-chunk SIMD filter for $>16$ patterns with zero-rebuild haystack rebinding.
- [`SimdSieveError`]: Enumerates pattern validation errors (empty set, empty pattern, limit exceeded).

## Construction Costs & Design Caps

- **16-Pattern Cap**: SIMD filter backends (AVX-512, AVX2, NEON, Scalar) pack up to 16 pattern prefixes into SIMD registers. Pattern sets exceeding 16 are automatically partitioned into 16-element chunks by `MultiSieve` / `CompiledMultiSieve`.
- **Filter Compilation Cost**: `SimdSieve::new` executes CPU feature detection (`std::is_x86_feature_detected!`), pattern deduplication, and heap allocation of boxed backend filters (`Box<Filter>`). `CompiledSieve::new` absorbs this cost up front; subsequent `.scan(haystack)` calls rebind haystack pointers on stack memory without heap allocation.
## Error Handling

- `SimdSieveError`
