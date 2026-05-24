# simdsieve — Internal Spec

> This file is gitignored. It exists for agents and internal development. Never committed to public repos.

## Identity
SIMD-accelerated byte-pattern pre-filtering engine — scans haystacks for up to 16 fixed-string patterns simultaneously using AVX-512, AVX2, NEON, or scalar fallback.

## Purpose
Without simdsieve, warpstate and warpscan would spend excessive CPU cycles verifying full patterns against every byte position. simdsieve provides a fast first-pass filter that rejects 99%+ of non-matching data before expensive verification.

## North Star
Match or exceed the prefix-filtering throughput of Intel Hyperscan's literal matcher or ripgrep's SIMD strategies. Legendary means >50 GB/s on AVX-512, zero false positives, seamless runtime backend selection, and a WebAssembly SIMD backend. A senior engineer should see the same attention to micro-architecture as cloudflare/zlib or ackermann/vectorized-search.

## Role in Ecosystem
- **Depends on:** none (zero runtime dependencies)
- **Depended on by:** tools/warpscan (via `simd-prefilter` feature), performance/matching/warpstate/core (via `fused` feature), tools/warpgrep, performance/matching/simdsieve/fuzz
- **Relationship to warpscan:** warpscan uses simdsieve as an optional SIMD prefilter before dispatching to warpstate backends.
- **Standalone value:** YES — excellent standalone crate for anyone needing fast multi-pattern prefix filtering in Rust.

## Invariants
- Every yielded offset is a verified match start (zero false positives).
- Backend selection happens once at construction and is deterministic for the host CPU.
- Block bounds are verified before every SIMD load; unaligned loads are safe for any valid pointer.
- No uninitialized memory is read; SIMD struct arrays are zero-initialized.
- `MAX_PATTERNS = 16` is a hard limit to keep register pressure bounded.

## Boundaries
- Does not handle regex or DFA semantics — fixed-string literals only.
- Does not manage pattern sets larger than 16 patterns — caller must shard.
- Does not provide capture groups or position metadata beyond the match start offset.
- Does not target GPUs — that's warpstate/vyre territory.

## Quality State
- Tests: 33 explicit test targets, 29 inline tests, 68 test files (~130 total)
- Lint preamble: yes (`#![warn(missing_docs, clippy::pedantic)]`, `#![forbid(unsafe_op_in_unsafe_fn)]`, unwrap deny block)
- `#![forbid(unsafe_code)]`: no — justified exception: SIMD intrinsics (`loadu`, `prefetch`, target-feature gated functions) require `unsafe` blocks confined to `avx2`, `avx512`, `neon`, and `scalar` backend modules. `unsafe_op_in_unsafe_fn` is forbidden.
- Doc coverage: ~85% (backends documented, some micro-optimization internals light on docs)
- Known issues: WebAssembly SIMD backend is missing; scalar fallback is slower than ideal on very small (<1 KB) haystacks.
