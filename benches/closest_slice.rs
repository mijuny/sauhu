//! Benchmark baseline for closest-slice lookup used by viewport sync.
//!
//! `sauhu::dicom::find_closest_slice` currently does a linear scan. After
//! switching to `partition_point` on a sorted (loc, idx) table, rerunning this
//! bench shows the speedup.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sauhu::dicom::find_closest_slice;

fn synth_locations(n: usize) -> Vec<Option<f64>> {
    (0..n).map(|i| Some(i as f64 * 1.25 - 100.0)).collect()
}

fn bench_closest_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("closest_slice");
    for &n in &[64usize, 256, 1024] {
        let locs = synth_locations(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("linear_{n}"), |b| {
            b.iter(|| find_closest_slice(black_box(&locs), black_box(42.7)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_closest_slice);
criterion_main!(benches);
