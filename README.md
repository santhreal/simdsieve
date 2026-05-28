# simdsieve

Part of [Santh](https://santh.dev) - open source Rust security and infrastructure tooling. Follow [@SanthProject](https://x.com/SanthProject) on X.

[![Crates.io](https://img.shields.io/crates/v/simdsieve)](https://crates.io/crates/simdsieve)
[![Docs.rs](https://docs.rs/simdsieve/badge.svg)](https://docs.rs/simdsieve)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A SIMD-accelerated multi-pattern pre-filtering engine with zero-allocation
streaming iteration. Scans byte haystacks for up to 16 patterns simultaneously
using the widest SIMD instructions available.

## What It Does

`simdsieve` is a **first-pass candidate filter** for substring search:

1. **Prefix Extraction**: Takes the first 1-4 bytes of each pattern
2. **SIMD Filtering**: Scans haystack in 32-128 byte blocks, finding positions
   where prefixes match using vectorized comparison
3. **Verification**: Confirms full pattern matches before yielding
4. **Zero False Positives**: Every yielded offset is a verified match start

Use this when you need to quickly filter large data streams for multiple
patterns before applying more expensive analysis.

## Architecture

| Backend  | Instruction Set      | Block Size | Throughput* | Registers |
|----------|----------------------|------------|-------------|-----------|
| AVX-512  | AVX-512F + AVX-512BW | 128 bytes  | >50 GB/s    | 512-bit   |
| AVX2     | AVX2                 | 64 bytes   | >25 GB/s    | 256-bit   |
| NEON     | AArch64 NEON         | 64 bytes   | >15 GB/s    | 128-bit   |
| Scalar   | Portable Rust        | 64 bytes   | >2 GB/s     | 64-bit    |

\* Single-byte pattern on modern x86_64. Multi-byte patterns scale linearly
with prefix length.

### Backend Selection

The optimal backend is auto-selected at construction time:

1. **AVX-512**: If `avx512f` and `avx512bw` CPU features are detected
2. **AVX2**: If `avx2` is detected
3. **NEON**: On `aarch64` targets (guaranteed by the architecture)
4. **Scalar**: Portable fallback for all other platforms

### Dual-Pump Processing

AVX-512, AVX2, and NEON use "dual-pump" processing where each logical block is
split into two halves processed together. This hides load latency by
interleaving independent operations across both load ports.

### Design Limits

- **16 patterns maximum**: Unrolled register layout saturates execution ports
- **4-byte prefix**: First 4 bytes used for SIMD filtering; full pattern for verification
- **ASCII case-insensitive**: Only `a`-`z` folded; non-ASCII compared verbatim

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
simdsieve = "0.1"
```

Basic exact matching:

```rust
use simdsieve::SimdSieve;

let haystack = b"The quick brown fox jumps over the lazy dog";
let patterns = &[b"fox"[..], b"dog"[..]];

let sieve = SimdSieve::new(haystack, patterns).unwrap();
let matches: Vec<usize> = sieve.collect();

assert_eq!(matches, vec![16, 40]); // Positions of "fox" and "dog"
```

Case-insensitive matching:

```rust
use simdsieve::SimdSieve;

let haystack = b"Hello World HELLO";
let patterns = &[b"hello"[..]];

let sieve = SimdSieve::new_case_insensitive(haystack, patterns).unwrap();
let matches: Vec<usize> = sieve.collect();

assert_eq!(matches, vec![0, 12]); // Matches "Hello" and "HELLO"
```

Density estimation (without full verification):

```rust
use simdsieve::SimdSieve;

let haystack = b"aaaaaa";
let patterns = &[b"a"[..]];

let count = SimdSieve::estimate_match_count(haystack, patterns, false);
assert_eq!(count, 6); // One prefix hit at each position
```

### Iterator Behavior

`SimdSieve` implements `Iterator<Item = usize>` and `FusedIterator`:

- Yields match offsets in ascending order
- After returning `None`, all subsequent calls return `None`
- Zero heap allocation during iteration
- Thread-safe: `Send` and `Sync` (immutable iterator)

## Error Handling

Construction can fail with [`SimdSieveError`](crate::error::SimdSieveError):

- `EmptyPatternSet`: No patterns provided
- `PatternLimitExceeded(usize)`: More than 16 patterns provided

```rust
use simdsieve::{SimdSieve, SimdSieveError};

let result = SimdSieve::new(b"haystack", &[]);
assert!(matches!(result, Err(SimdSieveError::EmptyPatternSet)));
```

## Platform Support

| Platform | Backend | Notes |
|----------|---------|-------|
| x86_64   | AVX-512, AVX2, Scalar | Runtime feature detection |
| aarch64  | NEON, Scalar | NEON guaranteed on AArch64 |
| wasm32   | Scalar  | No SIMD backend yet |
| other    | Scalar  | Portable fallback |

## Safety

`simdsieve` contains a small amount of `unsafe` code confined to
platform-specific backend modules. Every `unsafe` block has been audited for
soundness:

- **SIMD loads** use unaligned load intrinsics (`loadu`), safe for any valid
  pointer alignment.
- **Block bounds** are verified before each SIMD call; multi-byte prefixes
  require `block_len >= block_size + max_prefix_len - 1` trailing bytes.
- **Target features** are detected at construction; `#[target_feature]` gates
  prevent executing unsupported instructions.
- **No uninitialized memory** is read; SIMD struct arrays are zero-initialized.
- **`unsafe_op_in_unsafe_fn` is forbidden** at the crate level.

## Performance Tuning

### When to Use This Crate

✅ **Good for:**
- Filtering large streams (>1MB) for multiple patterns
- Finding candidate positions for regex engines
- Log analysis with multiple fixed-string patterns
- Network packet inspection with signature matching

❌ **Not ideal for:**
- Single small haystacks (<1KB)
- Patterns longer than 4 bytes that differ only after position 4
- Pattern sets that share a long common prefix against uniform haystacks
  (this shifts the bottleneck from SIMD filtering to byte-by-byte verification,
  resulting in O(n × pattern_count) work)
- When you need capture groups or regex features

### Prefetching

The engine automatically issues prefetch hints 512 bytes ahead. This value
(8 cache lines) balances:
- Enough lookahead to hide memory latency on modern CPUs
- Not so much that we evict useful data from cache

### Pattern Selection

For best performance:
- Use patterns with distinct first bytes when possible
- Place more frequent patterns earlier in the pattern list
- For case-insensitive mode, use lowercase in patterns

## Benchmarks

Run benchmarks with:

```bash
cargo bench
```

Example throughput on AMD EPYC 9R14 (AVX-512):

| Haystack | Patterns | Throughput |
|----------|----------|------------|
| 1 MB     | 1        | ~55 GB/s   |
| 1 MB     | 4        | ~45 GB/s   |
| 1 MB     | 16       | ~30 GB/s   |

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Areas of interest:

- WebAssembly SIMD backend
- Additional benchmarks and fuzz targets
- New verification strategies

Please ensure:
- `cargo test` passes
- `cargo clippy -- -D warnings` is clean
- `cargo fmt --check` passes
- New code has module-level and item-level documentation
