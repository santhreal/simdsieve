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

Key entry points are exported from `src/lib.rs` via `pub mod` and `pub use` re-exports.
Consult the module-level documentation in each source file for function signatures and usage examples.

## Error Handling

- `SimdSieveError`
