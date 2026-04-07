# Changelog

## 2026-04-07 -- Roadmap to 10: Tier 2

### Changed

- **Split database_window.rs**: Broke the 2,139-line monolith into a directory
  module with 4 files: `mod.rs` (orchestration, 282 lines),
  `local_browser.rs` (study listing, 477), `pacs_query.rs` (C-FIND/C-MOVE, 728),
  `anonymize.rs` (PHI stripping, 693).

- **VolumeUpload struct**: Replaced 8-argument `upload_volumes()` in GPU
  coregistration with a `VolumeUpload { data, dims, spacing }` struct that
  pairs volume data with its geometry.

- **GridContext struct**: Replaced 7-argument `show_grid()` in viewport manager
  with a `GridContext` struct bundling active viewport, drops, reference lines,
  and interaction mode.

### Added

- **72 new unit tests** (36 -> 108 total):
  - `dicom/anonymize.rs`: 22 tests for UID generation, name/ID formatting,
    accession numbers, config defaults, PHI tag coverage
  - `dicom/mpr.rs`: 44 tests for volume geometry, trilinear interpolation,
    anatomical plane methods, reslicing, the center-position invariant
    (gantry tilt), and image plane computation
  - `cache.rs`: 11 tests for insert/get, LRU eviction, stats, prefetch

## 2026-04-07 -- Roadmap to 10: Tier 1

### Changed

- **Named tuples**: Replaced `(PathBuf, usize, usize, ViewportId, u64)` with
  `ImageLoadRequest` struct in channels, pending lists, and function signatures.
  Removed `#[allow(clippy::type_complexity)]` suppression.

- **Channel error logging**: Added `tracing::warn` on channel send failures in
  anonymization, retrieval, and quick fetch workflows. Silent `let _ = tx.send()`
  retained only for shutdown-race cases.

- **Dead code cleanup**: Removed blanket `#![allow(dead_code)]` from 38 files.
  Replaced with targeted item-level `#[allow(dead_code)]` on specific items
  with comments explaining why they are kept. WIP modules (fusion, IPC) and
  utility libraries (spatial, db/schema) get module-level allows with comments.

### Added

- `src/app/interaction.rs`: Committed previously untracked file.

## 2026-04-07 21:12 -- Fix MPR reference lines and coregistration geometry

### Fixed

- **MPR reference lines inverted on coronal/sagittal views**: The `resample()`
  function hardcoded `(0..d).rev()` for the Z-axis, assuming volume z-index
  always increases from inferior to superior. When `slice_direction.z < 0`
  (z=0 is most superior), this put inferior at pixel row 0 (top of image).
  The `compute_image_plane()` geometry derived from the raw volume directions
  then described the opposite layout, causing `ImagePlane::intersect()` to
  compute reference line positions that appeared inverted on screen.

  Fix: both `resample()` and `compute_image_plane()` now check `z_sign` from
  `get_axis_direction(PatientAxis::Z)`. The resample only reverses when
  z_sign > 0; `compute_image_plane` places the position and col_direction
  accordingly. Both always produce superior at pixel row 0.

  Same fix applied to X-axis reversal in sagittal acquisition resampling.

- **Coregistration reference lines**: `apply_transform_to_volume()` kept the
  source volume's dimensions and spacing while copying the target's origin and
  directions, creating a geometry mismatch. Fix: resample onto the target's
  exact grid (target dimensions, spacing, origin, and directions) so both
  volumes are dimensionally identical.

### Removed

- `Volume::is_coregistered` field and associated reference line inversion hack
  in `calculate_mpr_reference_lines()`. No longer needed since the geometry is
  now fully consistent.

## 2026-04-07 17:00 -- Anonymous viewing mode

### Added

- **Anonymous viewing mode** (Ctrl+H): Hides all patient identifying information
  (name, ID, sex, age, study description, date) from viewport overlays. Also
  hides the patient sidebar. Press Ctrl+H again to restore.

## 2026-04-07 15:00 -- Refactoring and cleanup

### Changed

- **Type-safe orientations**: Replaced stringly-typed orientation values
  ("Ax", "Cor", "Sag") with `AnatomicalPlane` enum throughout the codebase.
  `detect_orientation()` in both parser and series_utils now returns
  `Option<AnatomicalPlane>`. Eliminates a class of typo-related bugs.

- **O(1) LRU cache**: Replaced `HashMap` + `VecDeque` (O(n) per access via
  `retain()`) with `IndexMap` for true O(1) LRU eviction and promotion.

- **Arc-wrapped images**: `ImageCache` now stores `Arc<DicomImage>`. Viewport
  holds `Arc<DicomImage>` instead of owned copies. Eliminates deep cloning of
  pixel data (512x512 x 2 bytes per scroll event) across cache lookups,
  viewport updates, and MPR image display.

- **SauhuApp decomposition**: Broke the 31-field god struct into 5 sub-systems:
  `MprSystem`, `QuickFetchSystem`, `BackgroundSystem`, `ImageLoadingSystem`,
  `InteractionState`. Top-level fields reduced from 31 to 19.

- **Long function extraction**: Split `check_quick_fetch_results()` (~275 lines)
  into focused handlers (`handle_query_complete`, `handle_series_complete`,
  `handle_retrieve_complete`). Extracted `compute_slice_location()` from
  `group_files_by_series()`.

- **Improved memory estimation**: Cache now accounts for `Vec` capacity overhead
  and metadata string allocations, not just pixel buffer length.

### Fixed

- **Unsafe unwrap/expect calls**: Replaced panicking unwraps in GPU buffer
  mapping (`gpu/coregistration.rs`), MPR volume handling (`app/mpr.rs`),
  fusion rendering (`ui/viewport.rs`), and sidebar logging
  (`ui/patient_sidebar.rs`) with proper error handling or safe patterns.

### Removed

- Dead blocking `set_patient()` method in `PatientSidebar` (replaced by async
  `request_set_patient()` path).
- `short_to_full_orientation()` helper (replaced by `AnatomicalPlane::full_name()`).
- `ImageCache::get_clone()` method (callers use `Arc::clone` via `get()`).

### Internal

- Named constants for coregistration HU thresholds (`CT_BRAIN_HU_MIN/MAX`,
  `CT_BODY_HU_MIN/MAX`) and GPU workgroup sizes (`NCC_WORKGROUP_SIZE`,
  `RESAMPLE_WORKGROUP_SIZE`).
- Added `indexmap` dependency.
- `Volume::from_series()` now accepts `&[impl Borrow<DicomImage>]` to work
  with both `Vec<DicomImage>` and `Vec<Arc<DicomImage>>`.
