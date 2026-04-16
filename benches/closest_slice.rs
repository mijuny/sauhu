//! Benchmark for closest-slice lookup used by viewport sync.
//!
//! `sauhu::dicom::find_closest_slice` is the legacy O(N) linear scan (still
//! pub for any external callers). `SyncInfo::closest_index` uses the cached
//! sorted (loc, idx) table and `partition_point`, which is what the hot
//! viewport-sync loop now calls.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sauhu::dicom::{find_closest_slice, AnatomicalPlane, SeriesInfo, SyncInfo};

fn synth_locations(n: usize) -> Vec<Option<f64>> {
    (0..n).map(|i| Some(i as f64 * 1.25 - 100.0)).collect()
}

fn build_sync_info(locations: Vec<Option<f64>>) -> SyncInfo {
    let info = SeriesInfo {
        series_uid: "bench".into(),
        description: None,
        display_name: "bench".into(),
        modality: Some("CT".into()),
        series_number: Some(1),
        files: Vec::new(),
        frame_of_reference_uid: None,
        study_instance_uid: None,
        orientation: Some(AnatomicalPlane::Axial),
        slice_locations: locations,
    };
    SyncInfo::from_series_info(&info)
}

fn bench_closest_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("closest_slice");
    for &n in &[64usize, 256, 1024] {
        let locs = synth_locations(n);
        let sync_info = build_sync_info(locs.clone());
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("linear_{n}"), |b| {
            b.iter(|| find_closest_slice(black_box(&locs), black_box(42.7)))
        });
        group.bench_function(format!("binary_{n}"), |b| {
            b.iter(|| sync_info.closest_index(black_box(42.7)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_closest_slice);
criterion_main!(benches);
