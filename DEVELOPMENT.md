# Sauhu Development Status

## Current Version: 0.1.0-alpha

## Completed Features

### Phase 1: Core Viewer (COMPLETE except 1.1)

#### DICOM Parsing & Loading
- [x] DICOM file parsing with dicom-rs
- [x] Metadata extraction (patient, study, series info)
- [x] Transfer syntax detection
- [x] Drag & drop loading
- [x] Folder scanning

#### Pixel Data Extraction
- [x] Uncompressed formats (Implicit/Explicit VR, LE/BE)
- [x] JPEG 2000 via OpenJPEG
- [x] JPEG Baseline/Lossless
- [x] JPEG-LS via CharLS
- [x] Proper HU handling for uncompressed CT
- [x] Auto-detection of pre-normalized data
- [x] Fallback handling for edge cases

#### Window/Level
- [x] Mouse-based adjustment (Shift+drag, right-click drag)
- [x] CT presets (keys 1-9)
- [x] Auto-optimal windowing (key 0)
- [x] Sensitivity scaling based on window width

#### Navigation & Display
- [x] Series navigation (arrows, scroll)
- [x] Zoom (Ctrl+drag, scroll wheel)
- [x] Pan (left-drag, middle-drag)
- [x] Fit to window (Space)
- [x] 1:1 pixel display

#### Infrastructure
- [x] SQLite database for persistence
- [x] Configuration system
- [x] Logging with tracing

---

## Phase 1.1: GPU-Accelerated Windowing (NEXT)

**Problem**: Large images (e.g., 1841x1955 CR) cause lag during windowing because we:
1. Process 3.6M pixels on CPU
2. Allocate 14MB per frame
3. Upload 14MB to GPU each frame

**Solution**: GPU-based windowing
- Upload raw u16 pixel data to GPU once
- Apply window/level in fragment shader
- Only update shader uniforms during drag

### Implementation Plan

1. **Add wgpu dependency**
   - Already in project plan
   - Will provide Vulkan/Metal/DX12 abstraction

2. **Create GPU texture from raw pixels**
   - Upload DicomImage.pixels as R16_UINT texture
   - Store rescale_slope/intercept as uniforms

3. **Write window/level shader**
   ```wgsl
   // Fragment shader pseudocode
   @fragment
   fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
       let pixel = textureSample(dicom_texture, sampler, uv).r;
       let hu = pixel * rescale_slope + rescale_intercept;
       let normalized = clamp((hu - wc + ww/2) / ww, 0.0, 1.0);
       return vec4<f32>(normalized, normalized, normalized, 1.0);
   }
   ```

4. **Integrate with egui**
   - Use egui's custom painting or texture callback
   - Update uniforms (WC, WW) on drag
   - No texture re-upload during windowing

### Expected Performance
- Initial load: Same (one-time texture upload)
- Windowing: 60+ FPS (only uniform updates)
- Memory: Reduced (no per-frame allocations)

---

## Phase 2: PACS Connectivity

### Features
- [ ] C-FIND (Query by patient, study, date, modality)
- [ ] C-MOVE (Retrieve to local storage)
- [ ] C-STORE (Send studies)
- [ ] PACS configuration UI
- [ ] Worklist view
- [ ] Background download with progress

### Implementation
- Use dicom-ul crate for DICOM networking
- Async operations with tokio
- Local cache in SQLite

---

## Phase 3: Multi-Viewport & Synchronization

### Features
- [ ] Dynamic viewport layouts (1x1, 1x2, 2x2, NxM)
- [ ] Tab-based workspace management
- [ ] Split viewport horizontal/vertical
- [ ] Drag & drop series to viewports
- [ ] Viewport synchronization (scroll sync by frame of reference)
- [ ] Sync groups & toggle per viewport
- [ ] Viewport maximize/restore
- [ ] Multi-monitor: detachable windows

### Implementation
- Use egui_tiles for dynamic tiling
- Synchronization manager for linked scrolling
- Ring buffer preload per viewport

---

## Phase 4: Smart Windowing

### Features
- [ ] Histogram-based auto-optimal WL
- [ ] ROI-based windowing (draw region -> optimize WL)
- [ ] Modality-aware defaults

---

## Phase 5: Measurements

### Features
- [ ] Distance measurement (line)
- [ ] Area (ellipse, rectangle, freehand)
- [ ] ROI statistics (mean, std, min, max HU)
- [ ] Angle measurements (Cobb, open angle)
- [ ] Measurement persistence in SQLite

### Implementation
- Trait-based tool system for extensibility
- Overlay rendering on viewport

---

## Phase 6: Co-registration

### Features
- [ ] Automatic intensity-based registration
- [ ] Mutual information algorithm
- [ ] Transform matrix storage
- [ ] Visual progress feedback

---

## Technical Decisions

### Why Transfer Syntax-Based Extraction?
CT images need raw pixel values for proper HU calculation. Compressed images get normalized by decoders. Solution: detect transfer syntax and use appropriate extraction path.

### Why GPU Windowing?
Professional DICOM viewers (UkkO, etc.) use GPU for windowing. CPU-based processing can't handle large images (3-4 megapixels) at interactive framerates.

### Why egui?
- Immediate mode simplicity
- Native Wayland support
- Good enough performance for UI
- Easy to extend with custom rendering

---

## File Structure

```
src/
├── main.rs              # Entry point, eframe setup
├── app.rs               # Main application state, event handling
├── config.rs            # TOML configuration
├── dicom/
│   ├── mod.rs           # Module exports
│   ├── parser.rs        # DICOM file reading, metadata
│   └── image.rs         # Pixel extraction, windowing
├── ui/
│   ├── mod.rs
│   └── viewport.rs      # Image display, interaction
└── db/
    ├── mod.rs           # Database connection
    └── schema.rs        # Tables, migrations
```

---

## Testing

### Test Files
- [pydicom test files](https://github.com/pydicom/pydicom/tree/main/tests/data) - 116 DICOM test files
- CT_small.dcm - Uncompressed CT (HU testing)
- 693_J2KI.dcm - JPEG 2000 CT
- RG1_UNCI.dcm - CR chest X-ray (large, MONOCHROME1)

### Current Test Results
- 103/116 files load successfully
- Failures: Non-image DICOM (SR, RT Plan), edge cases (32-bit RLE)
