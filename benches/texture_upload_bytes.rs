//! Benchmark baseline for u16 → LE bytes conversion used by `DicomTexture`.
//!
//! Mirrors `src/gpu/texture.rs:70`, where we currently `flat_map + collect`
//! to produce the byte buffer passed to `queue.write_texture`. The post-
//! optimization path uses `bytemuck::cast_slice` with zero allocation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn to_bytes_flat_map_collect(pixels: &[u16]) -> Vec<u8> {
    pixels.iter().flat_map(|&p| p.to_le_bytes()).collect()
}

fn synth_pixels(n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n);
    let mut state: u32 = 0xDEAD_BEEF;
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        out.push((state >> 16) as u16);
    }
    out
}

fn bench_texture_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("texture_upload_bytes");
    for &dim in &[256usize, 512, 1024] {
        let pixels = synth_pixels(dim * dim);
        group.throughput(Throughput::Bytes((dim * dim * 2) as u64));
        group.bench_function(format!("flat_map_{dim}x{dim}"), |b| {
            b.iter(|| to_bytes_flat_map_collect(black_box(&pixels)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_texture_bytes);
criterion_main!(benches);
