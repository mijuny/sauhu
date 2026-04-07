# Coregistration Implementation Guide

This document explains the rigid image coregistration implementation in Sauhu, covering the algorithms, coordinate systems, pitfalls encountered, and solutions developed.

## Overview

Coregistration aligns a **source** volume to a **target** volume using rigid transformation (6 DOF: 3 rotation + 3 translation). This enables:
- Comparing images from different imaging sessions
- Fusing information across modalities (MRI + CT)
- Tracking disease progression over time

## Architecture

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/coregistration/pipeline.rs` | Core algorithms: COM, PCA, transform application |
| `src/coregistration/optimizer.rs` | Powell's method optimizer, pyramid schedules |
| `src/coregistration/transform.rs` | RigidTransform, Mat4, coordinate conversions |
| `src/coregistration/metrics.rs` | NCC (Normalized Cross-Correlation) metric |
| `src/coregistration/manager.rs` | UI workflow, GPU orchestration |
| `src/gpu/coregistration.rs` | GPU compute shader resources |
| `src/gpu/shaders/coregistration.wgsl` | WGSL shader for NCC computation |

### Pipeline Flow

```
1. User selects target (Ctrl+click) and source (Ctrl+click again)
2. Volumes are downsampled to pyramid levels (32³ → 64³ → 128³)
3. Initial alignment computed (center-of-mass translation)
4. Multi-resolution optimization (coarse-to-fine)
5. Final transform applied to full-resolution source
6. Target geometry copied to transformed volume (critical for sync)
```

## Coordinate Systems

### Physical Coordinates (mm)

All coregistration computations use physical coordinates in millimeters. This is critical because:
- Volumes may have different voxel sizes
- Rotation must be around the correct center
- Translation units must be consistent

```rust
// VolumeGeometry converts between voxel and physical coordinates
pub struct VolumeGeometry {
    pub dimensions: (usize, usize, usize),  // voxels
    pub spacing: (f64, f64, f64),           // mm per voxel
    pub origin: (f64, f64, f64),            // mm
}

// Voxel to physical
let phys_x = voxel_x * spacing.0;
let phys_y = voxel_y * spacing.1;
let phys_z = voxel_z * spacing.2;
```

### Transform Convention

RigidTransform represents the transformation from **source to target**:
- `transform.translation_x/y/z`: Translation in mm
- `transform.rotation_x/y/z`: Euler angles (XYZ convention) in radians

To apply the transform (resample source aligned to target):
```rust
// For each OUTPUT voxel, find the INPUT location
let inv = transform.to_inverse_matrix();
let src_phys = inv.transform_point(output_phys);
let value = trilinear_sample(source, src_phys);
```

## Initial Alignment

### Center-of-Mass (COM) Pre-alignment

Before optimization, we compute initial translation from center-of-mass:

```rust
pub fn compute_center_of_mass(
    data: &[f32],
    geometry: &VolumeGeometry,
    threshold: f32,
) -> (f64, f64, f64)
```

This calculates the intensity-weighted centroid in physical coordinates:
```
COM = Σ(intensity × position) / Σ(intensity)
```

Initial translation: `target_COM - source_COM`

### PCA Rotation (Disabled)

We implemented PCA-based rotation pre-alignment but disabled it due to eigenvector sign ambiguity:

```rust
// Computes principal axes of intensity distribution
fn compute_principal_axes(...) -> PrincipalAxes;

// Computes rotation to align source axes to target
fn compute_rotation_from_axes(...) -> (f64, f64, f64);
```

**Problem**: Eigenvectors have arbitrary sign. For some volume pairs, this caused -45° rotations that made alignment worse (NCC dropped from 0.35 to 0.02).

**Current approach**: COM-only alignment. The optimizer finds rotation.

## Multi-Resolution Optimization

### Pyramid Schedule

Three resolution levels, coarse-to-fine:

| Level | Resolution | Iterations | Rotation Step | Translation Step |
|-------|------------|------------|---------------|------------------|
| 0 | 32³ | 30 | 9° | 12mm |
| 1 | 64³ | 25 | 5° | 6mm |
| 2 | 128³ | 20 | 2° | 2mm |

```rust
impl PyramidSchedule {
    pub fn balanced() -> Self {
        Self {
            iterations: vec![30, 25, 20],
            rotation_steps: vec![0.15, 0.08, 0.03],    // radians
            translation_steps: vec![12.0, 6.0, 2.0],  // mm
        }
    }
}
```

### Powell's Method Optimizer

Derivative-free optimization suitable for image registration:

1. For each parameter dimension (rx, ry, rz, tx, ty, tz):
   - Search along that direction for better NCC
   - Use line search with step refinement
2. Repeat until convergence (metric improvement < tolerance)
3. Reduce step sizes and continue if not converged

```rust
pub struct PowellOptimizer {
    pub bounds: TransformBounds,
    pub tolerance: f64,        // convergence criterion
    pub initial_steps: [f64; 7],
    pub min_step: f64,
}
```

## GPU Acceleration

### NCC Computation Shader

The similarity metric is computed on GPU for speed:

```wgsl
// coregistration.wgsl
@compute @workgroup_size(8, 8, 8)
fn compute_ncc(...) {
    // For each target voxel:
    // 1. Transform to source coordinates
    // 2. Trilinear sample from source
    // 3. Accumulate sums for NCC formula
}
```

NCC formula:
```
NCC = Σ(T - T̄)(S - S̄) / sqrt(Σ(T - T̄)² × Σ(S - S̄)²)
```

### Performance

- Single NCC evaluation: ~4ms at 64³
- Full optimization (3 levels): ~300-400ms
- Total coregistration time: ~2-3 seconds

## Critical: Geometry After Transform

### The Problem

After `apply_transform_to_volume`:
- Voxel data is resampled to align with target ✓
- Volume metadata (origin, directions) was kept from source ✗

This caused sync failure because slice locations are computed from volume geometry.

### The Solution

Copy target volume's geometry to the transformed volume:

```rust
pub fn apply_transform_to_volume(
    volume: &Volume,
    transform: &RigidTransform,
    target_volume: Option<&Volume>,  // NEW: target for geometry
) -> Volume {
    // ... resample voxel data ...

    // CRITICAL: Copy target geometry for sync
    if let Some(target) = target_volume {
        new_volume.origin = target.origin.clone();
        new_volume.row_direction = target.row_direction.clone();
        new_volume.col_direction = target.col_direction.clone();
        new_volume.slice_direction = target.slice_direction.clone();
        new_volume.acquisition_orientation = target.acquisition_orientation;
        new_volume.frame_of_reference_uid = target.frame_of_reference_uid.clone();
        new_volume.study_instance_uid = target.study_instance_uid.clone();
    }
}
```

This is analogous to the gantry tilt issue in MPR sync - geometry must be consistent for slice location computation to work.

## Reference Line Integration

After coregistration, reference lines must work between target and coregistered source. This requires:

1. **Same frame_of_reference_uid**: Copied from target
2. **Consistent geometry**: Copied from target
3. **Correct col_direction sign**: Must point INFERIOR for head-up display

The col_direction fix (separate from coregistration):
```rust
// z_sign indicates if z_dir points superior (+1) or inferior (-1)
let inferior_dir = if z_sign > 0.0 { z_dir.negate() } else { z_dir };
```

## Pitfalls Encountered

### 1. Voxel vs Physical Coordinates

**Symptom**: Transform worked at low resolution but failed at full resolution
**Cause**: Transform computed in normalized voxel space, applied to physical space
**Fix**: All computations in physical (mm) coordinates

### 2. Initial Alignment Making Things Worse

**Symptom**: NCC dropped from 0.35 to 0.02 after PCA alignment
**Cause**: Eigenvector sign ambiguity causing -45° rotations
**Fix**: Disabled PCA, use COM-only alignment

### 3. Sync Not Working After Coregistration

**Symptom**: Coregistered image didn't sync with target
**Cause**: Volume kept source geometry (origin, directions, UIDs)
**Fix**: Copy target geometry to transformed volume

### 4. Reference Lines Inverted

**Symptom**: Scrolling down moved reference line up
**Cause**: z_dir.negate() wrong when z_dir already points inferior
**Fix**: Check z_sign and use correct direction

## Testing Checklist

When modifying coregistration:

- [ ] COM alignment improves NCC (not makes it worse)
- [ ] Multi-resolution optimization converges
- [ ] Transformed volume syncs with target
- [ ] Reference lines move in correct direction
- [ ] Works for different acquisition orientations
- [ ] Works for different voxel sizes

## Future Work

1. **Affine registration**: Add scale and shear (12 DOF)
2. **Mutual Information**: Better metric for multi-modality (MRI + CT)
3. **Mask support**: Ignore regions outside brain/ROI
4. **Better initial alignment**: Fix PCA sign ambiguity or use moment-based approach
5. **Deformable registration**: For non-rigid changes (tumor growth, atrophy)

## File Reference

| File | Key Functions |
|------|---------------|
| `pipeline.rs` | `apply_transform_to_volume`, `compute_center_of_mass`, `compute_initial_alignment` |
| `optimizer.rs` | `PowellOptimizer::optimize`, `PyramidSchedule::balanced` |
| `transform.rs` | `RigidTransform::to_matrix`, `Mat4::transform_point` |
| `manager.rs` | `CoregistrationManager::start_registration`, GPU orchestration |
| `coregistration.wgsl` | `compute_ncc` shader |

## History

- **Initial implementation**: GPU-accelerated NCC with Powell optimizer
- **Phase 1**: Fixed coordinate system (voxel → physical mm)
- **Phase 2**: Added COM pre-alignment
- **Phase 3**: Improved pyramid schedule (3 levels, more iterations)
- **Phase 4**: Added PCA rotation (later disabled due to sign issues)
- **Geometry fix**: Copy target geometry for correct sync
- **Reference line fix**: Correct col_direction sign for head-up display
