//! Benchmark baseline for auto-window computation.
//!
//! Mirrors `calculate_optimal_window` in `src/ui/viewport.rs` so we can track
//! perf regressions/improvements against the same input data as production.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

/// Legacy implementation: materializes f64 values, sorts, picks 2/98
/// percentiles. Kept as a baseline so we can watch the histogram path
/// pull ahead.
fn calculate_optimal_window_sort(pixels: &[u16], slope: f64, intercept: f64) -> (f64, f64) {
    if pixels.is_empty() {
        return (0.0, 1.0);
    }
    let values: Vec<f64> = pixels
        .iter()
        .map(|&p| (p as f64) * slope + intercept)
        .collect();
    let mut sorted = values;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = sorted.len();
    let p2 = sorted[len * 2 / 100];
    let p98 = sorted[len * 98 / 100];
    let window_width = (p98 - p2).max(1.0);
    let window_center = (p2 + p98) / 2.0;
    (window_center, window_width)
}

/// Post-optimization implementation (mirrors viewport::calculate_optimal_window):
/// bucket u16 pixels directly into a 65536-entry histogram, recover the two
/// percentile raw values, apply rescale once.
fn calculate_optimal_window_histogram(
    pixels: &[u16],
    slope: f64,
    intercept: f64,
) -> (f64, f64) {
    if pixels.is_empty() {
        return (0.0, 1.0);
    }
    const U16_BINS: usize = 1 << 16;
    let mut hist = vec![0u32; U16_BINS];
    for &p in pixels {
        hist[p as usize] += 1;
    }
    let total = pixels.len() as u64;
    let lo_target = total * 2 / 100;
    let hi_target = total * 98 / 100;
    let mut cumulative: u64 = 0;
    let mut p2_raw: u16 = 0;
    let mut p98_raw: u16 = u16::MAX;
    let mut p2_found = false;
    for (bin, &count) in hist.iter().enumerate() {
        if count == 0 {
            continue;
        }
        cumulative += count as u64;
        if !p2_found && cumulative >= lo_target {
            p2_raw = bin as u16;
            p2_found = true;
        }
        if cumulative >= hi_target {
            p98_raw = bin as u16;
            break;
        }
    }
    let p2 = (p2_raw as f64) * slope + intercept;
    let p98 = (p98_raw as f64) * slope + intercept;
    ((p2 + p98) / 2.0, (p98 - p2).max(1.0))
}

fn synth_pixels(n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n);
    let mut state: u32 = 0x1234_5678;
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        out.push((state >> 16) as u16 & 0x0FFF);
    }
    out
}

fn bench_auto_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_window");
    for &dim in &[256usize, 512, 1024] {
        let pixels = synth_pixels(dim * dim);
        group.throughput(Throughput::Elements((dim * dim) as u64));
        group.bench_function(format!("sort_{dim}x{dim}"), |b| {
            b.iter(|| {
                calculate_optimal_window_sort(black_box(&pixels), black_box(1.0), black_box(-1024.0))
            })
        });
        group.bench_function(format!("histogram_{dim}x{dim}"), |b| {
            b.iter(|| {
                calculate_optimal_window_histogram(
                    black_box(&pixels),
                    black_box(1.0),
                    black_box(-1024.0),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_auto_window);
criterion_main!(benches);
