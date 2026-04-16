//! Benchmark baseline for scaled pixel min/max computation.
//!
//! Mirrors the two-fold pattern in `src/dicom/image.rs:118-125` so we can
//! measure the gain from merging into a single pass.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn minmax_two_passes(pixels: &[u16], slope: f64, intercept: f64) -> (f64, f64) {
    let min = pixels
        .iter()
        .map(|&p| (p as f64) * slope + intercept)
        .fold(f64::INFINITY, f64::min);
    let max = pixels
        .iter()
        .map(|&p| (p as f64) * slope + intercept)
        .fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// Post-optimization path: one linear pass over u16, rescale the two
/// endpoints once.
fn minmax_single_pass(pixels: &[u16], slope: f64, intercept: f64) -> (f64, f64) {
    let mut iter = pixels.iter().copied();
    let Some(first) = iter.next() else {
        return (f64::INFINITY, f64::NEG_INFINITY);
    };
    let mut min = first;
    let mut max = first;
    for p in iter {
        if p < min {
            min = p;
        } else if p > max {
            max = p;
        }
    }
    let a = (min as f64) * slope + intercept;
    let b = (max as f64) * slope + intercept;
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn synth_pixels(n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n);
    let mut state: u32 = 0xC0DE_BABE;
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        out.push((state >> 16) as u16 & 0x0FFF);
    }
    out
}

fn bench_minmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixel_minmax");
    for &dim in &[256usize, 512, 1024] {
        let pixels = synth_pixels(dim * dim);
        group.throughput(Throughput::Elements((dim * dim) as u64));
        group.bench_function(format!("two_pass_{dim}x{dim}"), |b| {
            b.iter(|| {
                minmax_two_passes(black_box(&pixels), black_box(1.0), black_box(-1024.0))
            })
        });
        group.bench_function(format!("single_pass_{dim}x{dim}"), |b| {
            b.iter(|| {
                minmax_single_pass(black_box(&pixels), black_box(1.0), black_box(-1024.0))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_minmax);
criterion_main!(benches);
