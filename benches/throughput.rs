#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
//! Throughput benchmarks for `simdsieve`.

use criterion::measurement::WallTime;
use criterion::{
    Bencher, BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, criterion_group,
    criterion_main,
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use simdsieve::SimdSieve;

fn make_haystack(size: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(0xBE0C_DEAD);
    let mut buf = vec![0u8; size];
    rng.fill_bytes(&mut buf);
    buf
}

fn bench_scan_throughput(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("scan_throughput");

    for size in [64, 1024, 64 * 1024, 256 * 1024, 1024 * 1024] {
        let haystack = make_haystack(size);
        let patterns: &[&[u8]] = &[b"abc", b"xyz", b"qrs", b"mno"];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}B")),
            &size,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(black_box(&haystack), black_box(patterns)).unwrap();
                    let count: usize = sieve.count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

fn bench_varying_patterns(c: &mut Criterion) {
    let mut group: BenchmarkGroup<'_, WallTime> = c.benchmark_group("varying_patterns");
    let haystack = make_haystack(256 * 1024);

    let all_patterns: &[&[u8]] = &[
        b"abc", b"xyz", b"qrs", b"mno", b"def", b"ghi", b"jkl", b"stu",
    ];

    for count in [1, 2, 4, 8] {
        let patterns = &all_patterns[..count];
        group.throughput(Throughput::Bytes(haystack.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}_patterns")),
            &count,
            |b: &mut Bencher<'_>, _| {
                b.iter(|| {
                    let sieve = SimdSieve::new(black_box(&haystack), black_box(patterns)).unwrap();
                    let count: usize = sieve.count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

fn bench_construction(c: &mut Criterion) {
    let haystack = make_haystack(64 * 1024);
    let patterns: &[&[u8]] = &[b"abc", b"xyz", b"qrs", b"mno"];

    c.bench_function("construction", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve = SimdSieve::new(black_box(&haystack), black_box(patterns)).unwrap();
            black_box(sieve);
        });
    });
}

fn bench_case_insensitive(c: &mut Criterion) {
    let haystack = make_haystack(256 * 1024);
    let patterns: &[&[u8]] = &[b"hello", b"world", b"rust", b"simd"];

    c.bench_function("case_insensitive_256k", |b: &mut Bencher<'_>| {
        b.iter(|| {
            let sieve =
                SimdSieve::new_case_insensitive(black_box(&haystack), black_box(patterns)).unwrap();
            let count: usize = sieve.count();
            black_box(count);
        });
    });
}

criterion_group!(
    benches,
    bench_scan_throughput,
    bench_varying_patterns,
    bench_construction,
    bench_case_insensitive,
);
criterion_main!(benches);
