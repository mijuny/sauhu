# Sauhu: Roadmap to 10/10

After the third ThePrimeagen x DHH review (9.0/10), these are the items that close the remaining gap.

Previous review items (8.5 -> 9.0): all 10 completed. Named tuples, channel logging, dead code cleanup, database_window split, 72 new tests, VolumeUpload/GridContext structs, consistent error types, GPU resilience, unsafe docs.

## Tier 1: High impact, low effort

### 1. Fix the last `too_many_arguments`

`spatial.rs:143` `compute_circle_roi_stats()` takes 8 args. Bundle the image context:

```rust
pub struct ImagePixels<'a> {
    pub data: &'a [u16],
    pub width: u32,
    pub height: u32,
    pub rescale_slope: f64,
    pub rescale_intercept: f64,
}
```

This was the pattern for items 5-7 last round. One was left behind.

### 2. Log the critical `let _ =` in MPR pipeline

`src/app/mpr.rs` has 9 silent discards. The `MprResult::VolumeFailed` and `MprResult::SeriesFailed` sends (lines 122, 158) are the important ones -- if the channel is closed AND the operation failed, the user sees nothing. Add `tracing::warn` on those. The `Cancel` sends (lines 412, 423) and `BuildVolume`/`GenerateSeries` requests are fine as-is (shutdown races).

### 3. Guard the one production unwrap in `mpr.rs`

Line 291: `images[*idx].image_plane.as_ref().unwrap()`. Every image used for volume construction should have an `image_plane`, but if one doesn't, the entire viewer panics. Filter or skip with a warning instead:

```rust
.filter_map(|(idx, _)| {
    let plane = images[*idx].image_plane.as_ref()?;
    Some(match acquisition_orientation {
        AcquisitionOrientation::Axial | AcquisitionOrientation::Unknown => plane.position.z,
        // ...
    })
})
```

## Tier 2: Medium effort, architectural clarity

### 4. Audit the coregistration module: ship or shelf

The coregistration module (2,400+ lines, 5 files) has:
- 28 `#[allow(dead_code)]` annotations across manager, metrics, optimizer, pipeline
- 7 `Mutex::lock().unwrap()` (standard, but lots of them)
- `pipeline.rs` alone has 21 dead code allows and 234 changed lines this round

**Decision needed:** Is coregistration shipping this cycle?
- **If yes:** Remove dead_code allows by actually calling the API, add tests (0 currently for pipeline/optimizer), document the GPU compute pipeline.
- **If no:** Move to a feature flag (`#[cfg(feature = "coregistration")]`) so it compiles out cleanly. Stops the dead code annotations from accumulating and makes the main binary smaller.

Don't leave it in limbo. WIP modules that never ship are tech debt that discourages contributors and confuses reviewers.

### 5. Same question for fusion (518 lines)

`src/fusion/mod.rs` has a blanket `#![allow(dead_code)]`. The module is 518 lines with a colormap implementation. Either wire it up or gate it behind a feature flag. Small enough that the decision is quick.

### 6. Split `mpr.rs` proactively

At 2,574 lines it's now the largest file in the codebase, bigger than `database_window.rs` was when it got split (2,139). Natural seams:

```
dicom/mpr/
  mod.rs           -- Volume struct, public API, resample()
  volume_build.rs  -- from_series(), slice sorting, spacing calculation
  reslice.rs       -- AlongRows/AlongColumns reslice implementations
  image_plane.rs   -- compute_image_plane(), axis direction logic
  tests.rs         -- the 44 tests (already a natural unit)
```

The test block alone is ~550 lines. Moving it to a submodule file is zero-risk and immediately makes the production code easier to navigate.

## Tier 3: Polish

### 7. `mpr_sandbox.rs` maintenance

2,555 lines for a test/debug binary. Questions:
- Is it tested by CI? (If not, it rots.)
- Does it duplicate logic from `mpr.rs`? (If so, it diverges.)
- Could it be trimmed or split into focused examples?

Not urgent, but it will become a liability if left untouched while `mpr.rs` evolves.

### 8. Remaining `let _ =` audit

37 silent discards total. After fixing the MPR ones (item 2), sweep the rest:
- `src/app/quick_fetch.rs` (5) -- are any of these failure-result sends?
- `src/app/image_loading.rs` (4) -- same question
- `src/dicom/anonymize.rs` (3) -- compliance-adjacent, worth logging

The pattern: `let _ =` on a `send()` that carries an error result should log. `let _ =` on a `send()` that carries a cancel/shutdown signal is fine.

### 9. Tighten coregistration unwraps (if keeping the module)

The 7 `Mutex::lock().unwrap()` in `manager.rs` are idiomatic Rust, but if a panic in any coregistration worker thread poisons the mutex, the next `lock().unwrap()` takes down the whole app. Consider `lock().expect("coregistration task mutex poisoned")` at minimum for debuggability, or handle the `PoisonError` if you want the viewer to survive a coregistration failure gracefully.

## What NOT to do

- Don't add doc comments to internal coregistration/fusion functions until the modules are feature-complete. Documentation on unstable APIs is wasted effort.
- Don't refactor `app/mod.rs` further. At 1,937 lines with 19 fields it's stable and coherent after the subsystem extraction.
- Don't chase the remaining `allow(dead_code)` in `db/schema.rs` or `dicom/image.rs`. Schema fields and image metadata are legitimately used conditionally.
- Don't unit test UI code (`viewport.rs`, `viewport_manager.rs`). Integration tests via the viewer itself are more valuable there.

## Completed (roadmap-fixes branch)

1. **Item 1**: Bundled `compute_circle_roi_stats` args into `ImagePixels` struct. Removed `#[allow(clippy::too_many_arguments)]`.
2. **Item 2**: Added `tracing::warn` on `VolumeFailed`/`SeriesFailed` channel sends in `src/app/mpr.rs`.
3. **Item 3**: Replaced bare `.unwrap()` with `.expect("image_plane guaranteed by earlier check")` in `src/dicom/mpr.rs`. The unwrap was already guarded by a check at line 229, so `expect` documents the invariant.
4. **Item 4**: Coregistration ships. Removed truly dead code (`body_ct()`, `cross_modality()`, `higher_is_better()`). Restored `resample_volume_parallel`/`trilinear_sample` (wrongly identified as dead, actually used internally). Kept all other `dead_code` annotations (internally used API).
5. **Item 5**: Fusion module removed entirely (never worked). Deleted `src/fusion/`, `src/gpu/shaders/fusion.wgsl`, and all references across renderer, viewport, viewport_manager, app/mod.rs, app/mpr.rs, app/coregistration.rs.
6. **Item 6**: Extracted tests (984 lines) from `mpr.rs` into `src/dicom/mpr/tests.rs`. Production code reduced from 2578 to 1594 lines. Full `volume_build`/`reslice`/`image_plane` split deferred (diminishing returns at 1594 lines).
7. **Item 8**: Audited `let _ =` in quick_fetch.rs, image_loading.rs, anonymize.rs. Only `quick_fetch.rs:430` (`upsert_study_for_retrieval` DB write) warranted logging; rest are shutdown-race sends.

## Summary

The gap between 9.0 and 9.5 is items 1-3 (quick fixes) plus the architectural decisions in 4-6. The gap between 9.5 and 10 is executing on those decisions and maintaining the discipline over time.

The hardest item is #4: deciding whether coregistration ships or shelves. Everything else follows from that.
