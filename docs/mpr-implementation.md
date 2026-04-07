# MPR Implementation Guide

This document explains the critical details of Multi-Planar Reconstruction (MPR) in Sauhu, including coordinate systems, sync mechanisms, and common pitfalls.

## Overview

MPR allows viewing a 3D volume (e.g., 3D Sagittal T1) in any anatomical plane (Axial, Coronal, Sagittal). The implementation must handle:

1. **Volume resampling** - extracting 2D slices from 3D data
2. **Coordinate geometry** - mapping between volume indices and patient coordinates
3. **Cross-series sync** - ensuring different series show the same anatomical level
4. **Reference lines** - showing slice position across orthogonal views

## Coordinate Systems

### Patient Coordinate System (DICOM LPS)
- **X**: Left → Right (patient's left is positive)
- **Y**: Posterior → Anterior (patient's back is positive)
- **Z**: Inferior → Superior (patient's head is positive)

### Volume Index Space
- **(i, j, k)**: Integer indices into the 3D voxel array
- Mapping to patient coordinates depends on acquisition orientation

### Image Pixel Space
- **(col, row)**: 2D pixel coordinates in a displayed image
- (0, 0) is top-left corner
- Row increases downward, column increases rightward

## Critical: Image Position - Corner vs Center

**This is the most important concept for sync to work correctly.**

### The Problem: Gantry Tilt

Brain MRI acquisitions often use **gantry tilt** (typically 15-25°) to align axial slices parallel to the AC-PC line and avoid imaging the eyes.

For a tilted slice:
- `ImagePositionPatient` (DICOM tag 0020,0032) is the **corner** position
- The **center** of the image is at a different Z coordinate

```
Side view of tilted axial slice:

        Corner (IPP) ----→ Z_corner
       /
      /  ~20° tilt
     /
    Center ----→ Z_center (higher Z, more superior)
```

For a 256mm FOV with 20° tilt:
```
Z_center = Z_corner + (128mm × sin(20°)) ≈ Z_corner + 44mm
```

### The Solution: Always Use Center Position for Sync

Both regular series AND MPR must compute slice location at the **image center**, not corner:

```rust
// Regular series (series_utils.rs)
let center_z = corner_z
    + (half_cols × col_spacing × row_dir_z)
    + (half_rows × row_spacing × col_dir_z);

// MPR series (mpr.rs)
let center_pos = corner_pos
    .add(&row_direction.scale(half_w × col_spacing))
    .add(&col_direction.scale(half_h × row_spacing));
```

Where:
- `row_dir` = ImageOrientationPatient[0:3] (points along columns, horizontal)
- `col_dir` = ImageOrientationPatient[3:6] (points along rows, vertical)
- For tilted axial: `col_dir_z = sin(tilt_angle)` ≠ 0

### Why This Matters

| Scenario | Corner Z | Center Z | Anatomy |
|----------|----------|----------|---------|
| Tilted regular axial slice 30 | 30mm | 74mm | Basal ganglia |
| Untilted MPR axial at same anatomy | 74mm | 74mm | Basal ganglia |

If we use corner Z for sync:
- Regular series reports Z=30mm for basal ganglia
- MPR at Z=30mm shows inferior temporal lobe (wrong!)

If we use center Z for sync:
- Both report Z=74mm for basal ganglia
- Sync works correctly!

## MPR Slice Generation

### Volume Structure

```rust
pub struct Volume {
    pub data: Vec<u16>,           // 3D voxel data, row-major order
    pub dimensions: (usize, usize, usize),  // (width, height, depth)
    pub spacing: (f64, f64, f64), // Voxel spacing in mm
    pub origin: Vec3D,            // Patient coordinate of voxel (0,0,0)
    pub acquisition_orientation: AcquisitionOrientation,
    // ... direction vectors for each axis
}
```

### Axis Direction Mapping

The volume maps its dimensions to patient axes based on acquisition orientation:

```rust
// For a sagittal-acquired volume:
// - Volume X dimension (width) = slices along patient X (L-R)
// - Volume Y dimension (height) = along patient Z (S-I)
// - Volume Z dimension (depth) = along patient Y (A-P)

fn get_axis_direction(&self, axis: PatientAxis) -> (Vec3D, f64, f64) {
    // Returns (direction_vector, spacing, sign)
    // Maps from volume index to patient coordinate change
}
```

### Resampling Process

To generate an axial MPR slice at index `k`:

1. Determine the slice plane position in patient coordinates
2. For each output pixel (col, row):
   - Compute patient coordinate using ImagePlane geometry
   - Transform to volume index space
   - Sample (trilinear interpolation) from volume data

```rust
// Simplified resampling
for row in 0..height {
    for col in 0..width {
        let patient_pos = corner
            + row_dir.scale(col as f64 * col_spacing)
            + col_dir.scale(row as f64 * row_spacing);

        let (vi, vj, vk) = patient_to_volume_index(patient_pos);
        output[row][col] = trilinear_sample(volume, vi, vj, vk);
    }
}
```

## Sync Mechanism

### SyncInfo Structure

```rust
pub struct SyncInfo {
    pub frame_of_reference_uid: Option<String>,  // Must match for sync
    pub study_instance_uid: Option<String>,      // Should match
    pub orientation: Option<&'static str>,       // "Ax", "Cor", "Sag"
    pub slice_locations: Vec<Option<f64>>,       // CENTER position for each slice
}
```

### Sync Algorithm

```rust
fn sync_to_location(target_z: f64, other_series: &SyncInfo) -> usize {
    // 1. Check frame_of_reference matches
    // 2. Check orientation matches (Ax syncs with Ax, etc.)
    // 3. Find slice with closest slice_location to target_z
    other_series.slice_locations
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a - target_z).abs().partial_cmp(&(b - target_z).abs())
        })
        .map(|(idx, _)| idx)
}
```

## Reference Lines

Reference lines show where the active slice intersects other views.

### Principle

For orthogonal planes, the intersection is a straight line:
- Axial slice intersects Sagittal/Coronal as a **horizontal** line
- Sagittal slice intersects Axial as a **vertical** line
- Coronal slice intersects Axial as a **horizontal** line

### Two Code Paths

**1. Geometry-based (`ImagePlane::intersect`)** in `calculate_reference_lines()`:
Used for cross-volume reference lines (e.g., MPR to original series, coregistered
to target). Projects the active plane through 3D patient coordinates and finds
where it crosses the other viewport's image edges. This is the primary path and
handles arbitrary orientations.

**2. Index-based** in `calculate_mpr_reference_lines()`:
Used for MPR-to-MPR within the same volume (`Arc::ptr_eq`). Maps slice index
directly to pixel position. Only fires when both viewports share the same
`Arc<Volume>`.

### Critical: Resample and ImagePlane Must Agree

The `resample()` function produces pixel data with a specific layout (which end
is at row 0). The `compute_image_plane()` function describes that layout in 3D
patient coordinates. If they disagree, `ImagePlane::intersect()` computes
reference line positions that don't match the displayed anatomy.

The key variable is `z_sign` from `get_axis_direction(PatientAxis::Z)`:

| z_sign | Volume z=0 | Superior at pixel row 0 via | ImagePlane position |
|--------|------------|----------------------------|---------------------|
| > 0 | inferior | `(0..d).rev()` | origin + z_offset (far end) |
| < 0 | superior | forward `(0..d)` | origin (near end) |

Both `resample()` and `compute_image_plane()` check `z_sign` and handle each
case so that pixel row 0 always corresponds to the superior end of the Z range.

### Reference Line Colors

- **Yellow** (255, 255, 0): Axial plane
- **Cyan** (0, 255, 255): Coronal plane
- **Magenta** (255, 0, 255): Sagittal plane

## Common Pitfalls

### 1. Forgetting Gantry Tilt Correction
**Symptom**: Synced series show different anatomical levels (off by 30-50mm)
**Cause**: Using corner position instead of center position
**Fix**: Always compute center position using ImageOrientationPatient

### 2. Reference Line Inversion
**Symptom**: Scrolling UP in axial moves reference line DOWN in coronal/sagittal
**Cause**: `resample()` and `compute_image_plane()` disagree about pixel layout.
The resample hardcodes a z-iteration order, but the ImagePlane derives directions
from the raw volume geometry. When `slice_direction.z < 0`, the two disagree.
**Fix**: Both must check `z_sign` and handle each case consistently. See the
"Resample and ImagePlane Must Agree" section above.

### 3. Wrong Pixel Spacing Convention
**Symptom**: Images appear stretched or sync is slightly off
**Cause**: Confusing row_spacing vs col_spacing
**Remember**:
- `PixelSpacing[0]` = row spacing (vertical distance between pixels)
- `PixelSpacing[1]` = column spacing (horizontal distance between pixels)
- Row direction moves along columns (horizontal)
- Col direction moves along rows (vertical)

### 4. Assuming Untilted Acquisition
**Symptom**: Works for some patients, fails for others
**Cause**: Hardcoding assumptions about slice orientation
**Fix**: Always use ImageOrientationPatient for geometry calculations

## File Reference

| File | Responsibility |
|------|----------------|
| `src/dicom/mpr.rs` | Volume resampling, MPR slice generation, ImagePlane geometry |
| `src/dicom/series_utils.rs` | Regular series slice_location computation (with tilt correction) |
| `src/ui/viewport_manager.rs` | Reference line calculation, sync coordination |
| `src/dicom/spatial.rs` | Geometric utilities, find_closest_slice |

## Testing Checklist

When modifying MPR or sync code, verify:

- [ ] Regular Axial syncs with MPR Axial at same anatomical level
- [ ] Tilted acquisitions (brain MRI) sync correctly
- [ ] Reference lines move in correct direction when scrolling
- [ ] All three planes (Ax/Cor/Sag) sync and show reference lines correctly
- [ ] Works for different acquisition orientations (axial-acquired, sagittal-acquired, etc.)

## History

**January 2026**: Initial MPR implementation. Key lessons:
1. Gantry tilt correction is essential for brain MRI sync
2. Both regular series and MPR must use consistent center-based slice locations

**April 2026**: Fixed reference line inversion. Key lessons:
3. Resample pixel layout and ImagePlane geometry must agree on z-axis direction
4. The `z_sign` from `get_axis_direction(PatientAxis::Z)` determines whether
   z-index increases toward superior (> 0) or inferior (< 0)
5. Never hardcode iteration order; always derive from volume geometry
