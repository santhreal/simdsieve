#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Criterion benchmarks for simdsieve.
//!
//! These benchmarks measure throughput at various haystack sizes and
//! pattern counts, comparing exact vs case-insensitive modes.

use criterion::measurement::WallTime;
use criterion::{
    Bencher, BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, criterion_group,
    criterion_main,
};
use simdsieve::SimdSieve;

/// Creates a haystack of the given size with some pattern occurrences.
fn create_haystack(size: usize) -> Vec<u8> {
    let mut haystack = vec![b'x'; size];
    // Sprinkle in some matches every 256 bytes
    for i in (0..size.saturating_sub(4)).step_by(256) {
        haystack[i..i + 4].copy_from_slice(b"test");
    }
    haystack
}

/// Benchmark throughput at different haystack sizes with 1 pattern.
fn bench_throughput_1pattern(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("throughput_1pattern");

    for size in [1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024] {
        let haystack = create_haystack(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(&haystack, &[b"test"]).unwrap();
                    let count = sieve.count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark throughput with 4 patterns.
fn bench_throughput_4patterns(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("throughput_4patterns");

    for size in [1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024] {
        let haystack = create_haystack(size);
        let patterns: Vec<&[u8]> = vec![b"test", b"xxxx", b"aaaa", b"bbbb"];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(&haystack, &patterns).unwrap();
                    let count = sieve.count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark throughput with 8 patterns.
fn bench_throughput_8patterns(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("throughput_8patterns");

    for size in [1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024] {
        let haystack = create_haystack(size);
        let patterns: Vec<&[u8]> = vec![
            b"test", b"xxxx", b"aaaa", b"bbbb", b"cccc", b"dddd", b"eeee", b"ffff",
        ];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(&haystack, &patterns).unwrap();
                    let count = sieve.count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark exact vs case-insensitive comparison.
fn bench_exact_vs_ci(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_vs_ci");
    let size = 1024 * 1024; // 1MB
    let haystack = create_haystack(size);

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("exact", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve = SimdSieve::new(&haystack, &[b"test"]).unwrap();
            let count = sieve.count();
            black_box(count);
        });
    });

    group.bench_function("case_insensitive", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve = SimdSieve::new_case_insensitive(&haystack, &[b"test"]).unwrap();
            let count = sieve.count();
            black_box(count);
        });
    });

    group.finish();
}

/// Benchmark construction cost (filter setup).
fn bench_construction(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("construction");
    let haystack = create_haystack(1024);

    for num_patterns in [1, 4, 8] {
        let patterns: Vec<Vec<u8>> = (0..num_patterns).map(|i| vec![b'a' + i as u8; 4]).collect();
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();

        group.bench_with_input(
            BenchmarkId::new("patterns", num_patterns),
            &num_patterns,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(&haystack, &pattern_refs).unwrap();
                    black_box(sieve);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different pattern lengths.
fn bench_pattern_lengths(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("pattern_lengths");
    let size = 1024 * 1024; // 1MB
    let haystack = create_haystack(size);

    group.throughput(Throughput::Bytes(size as u64));

    for pat_len in [1, 2, 3, 4, 5, 8] {
        let pattern: Vec<u8> = vec![b't'; pat_len];
        group.bench_with_input(
            BenchmarkId::from_parameter(pat_len),
            &pat_len,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(&haystack, &[&pattern]).unwrap();
                    let count = sieve.count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark density scoring vs full iteration.
fn bench_estimate_match_count(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("estimate_match_count");
    let size = 1024 * 1024; // 1MB
    let haystack = create_haystack(size);

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("full_iteration", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve = SimdSieve::new(&haystack, &[b"test"]).unwrap();
            let count = sieve.count() as u64;
            black_box(count);
        });
    });

    group.bench_function("estimate_match_count", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let count = SimdSieve::estimate_match_count(&haystack, &[b"test"], false).unwrap();
            black_box(count);
        });
    });

    group.finish();
}

/// Benchmark worst-case (no matches) vs best-case (all matches).
fn bench_match_density(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("match_density");
    let size = 1024 * 1024; // 1MB

    // Worst case: no matches (all zeros)
    let no_match_haystack = vec![b'x'; size];
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_function("no_matches", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve = SimdSieve::new(&no_match_haystack, &[b"test"]).unwrap();
            let count = sieve.count();
            black_box(count);
        });
    });

    // Best case: single-byte pattern matches everywhere
    let all_match_haystack = vec![b'a'; size];
    group.bench_function("all_match_single_byte", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve = SimdSieve::new(&all_match_haystack, &[b"a"]).unwrap();
            let count = sieve.count();
            black_box(count);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_throughput_1pattern,
    bench_throughput_4patterns,
    bench_throughput_8patterns,
    bench_exact_vs_ci,
    bench_construction,
    bench_pattern_lengths,
    bench_estimate_match_count,
    bench_match_density,
);
criterion_main!(benches);
