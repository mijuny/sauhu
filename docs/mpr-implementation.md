# MPR Implementation Guide

This document explains the critical details of Multi-Planar Reconstruction (MPR) in Sauhu, including coordinate systems, sync mechanisms, and reference line rendering.

## Overview

MPR allows viewing a 3D volume (e.g., 3D Sagittal T1) in any anatomical plane (Axial, Coronal, Sagittal). The implementation must handle:

1. **Volume resampling** - extracting 2D slices from 3D data
2. **Coordinate geometry** - mapping between volume indices and patient coordinates
3. **Cross-series sync** - ensuring different series show the same anatomical level
4. **Reference lines** - showing slice position across orthogonal views

## Coordinate Systems

### Patient Coordinate System (DICOM LPS)
- **X**: Left to Right (patient's left is positive)
- **Y**: Posterior to Anterior (patient's back is positive)
- **Z**: Inferior to Superior (patient's head is positive)

### Volume Index Space
- **(i, j, k)**: Integer indices into the 3D voxel array
- Mapping to patient coordinates depends on acquisition orientation

### Image Pixel Space
- **(col, row)**: 2D pixel coordinates in a displayed image
- (0, 0) is top-left corner
- Row increases downward, column increases rightward

## Critical: Image Position - Corner vs Center

### The Problem: Gantry Tilt

Brain MRI acquisitions often use **gantry tilt** (typically 15-25 degrees) to align axial slices parallel to the AC-PC line and avoid imaging the eyes.

For a tilted slice:
- `ImagePositionPatient` (DICOM tag 0020,0032) is the **corner** position
- The **center** of the image is at a different Z coordinate

```
Side view of tilted axial slice:

        Corner (IPP) ----> Z_corner
       /
      /  ~20 deg tilt
     /
    Center ----> Z_center (higher Z, more superior)
```

For a 256mm FOV with 20 deg tilt:
```
Z_center = Z_corner + (128mm x sin(20 deg)) ~ Z_corner + 44mm
```

### The Solution: Always Use Center Position for Sync

Both regular series AND MPR must compute slice location at the **image center**, not corner:

```rust
// Regular series (series_utils.rs)
let center_z = corner_z
    + (half_cols * col_spacing * row_dir_z)
    + (half_rows * row_spacing * col_dir_z);

// MPR series (mpr.rs - to_dicom_image and MprSeries::generate)
let center_pos = corner_pos
    .add(&row_direction.scale(half_w * col_spacing))
    .add(&col_direction.scale(half_h * row_spacing));
```

This matters because:

| Scenario | Corner Z | Center Z | Anatomy |
|----------|----------|----------|---------|
| Tilted regular axial slice 30 | 30mm | 74mm | Basal ganglia |
| Untilted MPR axial at same anatomy | 74mm | 74mm | Basal ganglia |

## MPR Slice Generation

### Volume Structure

```rust
pub struct Volume {
    pub data: Vec<u16>,           // 3D voxel data, Z-Y-X order (slice, row, column)
    pub dimensions: (usize, usize, usize),  // (width, height, depth) = (cols, rows, slices)
    pub spacing: (f64, f64, f64), // (col_spacing, row_spacing, slice_spacing)
    pub origin: Vec3,             // Patient coordinate of first sorted slice corner
    pub row_direction: Vec3,      // Unit vector along columns (horizontal in slice)
    pub col_direction: Vec3,      // Unit vector along rows (vertical in slice)
    pub slice_direction: Vec3,    // Slice normal (between-slice direction)
    pub acquisition_orientation: AcquisitionOrientation,
    // ...
}
```

### Axis Direction Mapping

`Volume::get_axis_direction(axis)` returns `(direction, spacing, sign)` for the
volume direction that most closely aligns with a patient axis. The sign indicates
whether the direction is positive (+1) or negative (-1) along that axis. This is
used by both `resample()` and `compute_image_plane()` to ensure consistency.

### Resampling via Reslice Operations

The resample function classifies each target plane as one of three operations
based on the acquisition orientation:

| Operation | Description | Used when |
|-----------|-------------|-----------|
| `Native` | Return original slices unchanged | Target matches acquisition |
| `AlongRows` | Take one row from each slice | Axial from Sag/Cor, etc. |
| `AlongColumns` | Take one column from each slice | Coronal from Sag, Sagittal from Ax, etc. |

### Axis Reversal in Resampling

The resample applies two axis reversals for correct anatomical display:

1. **Z reversal** (`z_sign > 0`): When the volume's Z axis increases toward
   superior, the Z iteration is reversed so pixel row 0 = superior (top of image).

2. **X reversal** (`x_sign > 0`): When the volume's X axis increases toward
   patient left, the X iteration is reversed so pixel column 0 = patient left
   (radiological convention: patient left on screen right is NOT applied;
   instead patient left at column 0).

Both reversals must be matched by `compute_image_plane()` so the ImagePlane
geometry accurately describes each pixel's patient coordinate.

## ImagePlane Geometry (compute_image_plane)

`ReformattedSlice::compute_image_plane()` produces an `ImagePlane` that describes
the 3D patient-coordinate geometry of the displayed MPR image. This is used by
`ImagePlane::intersect()` to compute reference line positions.

### Key invariants

1. **Position** is the top-left pixel (0,0) in patient coordinates.
2. **row_direction** points from pixel (0,0) toward pixel (width,0), matching the
   actual voxel data at each column.
3. **col_direction** points from pixel (0,0) toward pixel (0,height), matching the
   actual voxel data at each row.
4. `pixel_to_patient(px, py) = position + row_direction * px * col_spacing + col_direction * py * row_spacing`
   must return the correct patient coordinate for the voxel displayed at that pixel.

### How axis reversals are handled

For the **Z axis** (affects col_direction in Coronal/Sagittal):
- `z_sign > 0`: Position is shifted to the superior end of the Z range.
  col_direction = `z_dir.negate()` (pointing inferior/downward).
- `z_sign < 0`: Position stays at origin (already superior).
  col_direction = `z_dir` (already points inferior since z_dir.z < 0).

For the **X axis** (affects row_direction in Coronal/Axial):
- `x_sign > 0`: Position is shifted to the far X end by
  `x_dir * (width - 1) * col_spacing`. row_direction = `x_dir.negate()`.
- `x_sign < 0`: No adjustment needed.

## Reference Lines

Reference lines show where the active slice intersects other views.

### Two Code Paths

**1. Geometry-based (`ImagePlane::intersect`)** in `calculate_reference_lines()`:
The primary path. Used for all cross-series reference lines and for MPR-to-original
reference lines. Projects the active plane through 3D patient coordinates and
finds where it crosses the other viewport's image edges. Handles arbitrary
orientations including gantry tilt.

Requires: both viewports have `ImagePlane` with matching `frame_of_reference_uid`,
and the planes are not parallel.

**2. Index-based** in `calculate_mpr_reference_lines()`:
Used only for MPR-to-MPR within the same volume (`Arc::ptr_eq`). Maps the active
slice index directly to a pixel position on the other viewport. Fires only when
both viewports share the same `Arc<Volume>`.

### Rendering: pixel_ratio Correction

The `draw_reference_lines()` function must apply the same `pixel_ratio`
(row_spacing / col_spacing) correction as the image rendering. Without this,
reference lines drift from the anatomy on images with non-square pixels, with
the error increasing from zero at the image center to maximum at the edges.

MPR reformats commonly have non-square pixels when in-plane resolution differs
from slice spacing.

### Reference Line Colors

- **Yellow** (255, 255, 0): Axial plane
- **Cyan** (0, 255, 255): Coronal plane
- **Magenta** (255, 0, 255): Sagittal plane

## Sync Mechanism

### SyncInfo Structure

```rust
pub struct SyncInfo {
    pub frame_of_reference_uid: Option<String>,
    pub study_instance_uid: Option<String>,
    pub orientation: Option<AnatomicalPlane>,
    pub slice_locations: Vec<Option<f64>>,  // CENTER position for each slice
}
```

### Sync Algorithm

Sync matches viewports by orientation (Ax syncs with Ax, Cor with Cor, etc.)
within the same Frame of Reference. When the active viewport scrolls, other
viewports with matching orientation find the closest slice_location.

Cross-orientation sync is intentionally disabled. Use reference lines to see
the spatial relationship between different orientations.

## Common Pitfalls

### 1. Forgetting Gantry Tilt Correction
**Symptom**: Synced series show different anatomical levels (off by 30-50mm)
**Cause**: Using corner position instead of center position
**Fix**: Always compute center position using ImageOrientationPatient

### 2. Resample and ImagePlane Disagree on Pixel Layout
**Symptom**: Reference lines appear at wrong position or inverted
**Cause**: `resample()` reverses an axis for display, but `compute_image_plane()`
does not account for the reversal
**Fix**: Both functions must check `z_sign` and `x_sign` and handle each case
consistently. See the "Axis Reversal in Resampling" section.

### 3. Missing pixel_ratio in Reference Line Drawing
**Symptom**: Reference lines drift from the anatomy toward image edges,
correct only at the image center
**Cause**: `draw_reference_lines()` does not apply `pixel_ratio` stretch
that the rendering functions apply
**Fix**: Multiply Y coordinates and vertical scaling by `pixel_ratio`

### 4. Wrong Pixel Spacing Convention
**Symptom**: Images appear stretched or sync is slightly off
**Cause**: Confusing row_spacing vs col_spacing
**Remember**:
- `PixelSpacing[0]` = row spacing (vertical distance between pixels)
- `PixelSpacing[1]` = column spacing (horizontal distance between pixels)
- Row direction moves along columns (horizontal)
- Col direction moves along rows (vertical)

## File Reference

| File | Responsibility |
|------|----------------|
| `src/dicom/mpr.rs` | Volume, resample, compute_image_plane, MprSeries generation |
| `src/dicom/geometry.rs` | ImagePlane, intersect, pixel_to_patient |
| `src/dicom/series_utils.rs` | Regular series slice_location (with tilt correction) |
| `src/ui/viewport_manager.rs` | Reference line calculation, sync coordination |
| `src/ui/viewport.rs` | Reference line rendering (draw_reference_lines) |

## Testing Checklist

When modifying MPR or sync code, verify:

- [ ] Regular Axial syncs with MPR Axial at same anatomical level
- [ ] Tilted acquisitions (brain MRI) sync correctly
- [ ] Reference lines move in correct direction when scrolling
- [ ] All three planes (Ax/Cor/Sag) show reference lines correctly
- [ ] Reference lines correct at image edges, not just center
- [ ] Works for different acquisition orientations (axial, sagittal, coronal)
- [ ] Coregistered images show correct reference lines

## History

**January 2026**: Initial MPR implementation. Key lessons:
1. Gantry tilt correction is essential for brain MRI sync
2. Both regular series and MPR must use consistent center-based slice locations

**April 2026**: Fixed reference line inversion for z_sign-dependent volumes.
3. Resample pixel layout and ImagePlane geometry must agree on z-axis direction
4. The `z_sign` from `get_axis_direction(PatientAxis::Z)` determines whether
   z-index increases toward superior (> 0) or inferior (< 0)
5. Never hardcode iteration order; always derive from volume geometry

**April 2026**: Fixed cross-series reference line positioning on MPR views.
6. `draw_reference_lines()` must apply `pixel_ratio` to match rendered image scaling
7. `compute_image_plane()` must account for X-axis reversal when `x_sign > 0`
8. These fixes resolved reference lines being wrong on coronal/sagittal MPR
   when viewing cross-series (e.g., T2 axial reference on T1 coronal MPR)
