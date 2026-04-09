# Sauhu Development Status

## Current Version: 0.2.0

## Codebase

~20,000 lines of Rust across two crates (`sauhu` and `sauhu-common`), 117 tests.

## Implemented Features

### DICOM Parsing & Loading
- DICOM file parsing with dicom-rs
- Metadata extraction (patient, study, series info)
- Transfer syntax detection (Implicit/Explicit VR, LE/BE)
- Compressed formats: JPEG 2000, JPEG Baseline/Lossless, JPEG-LS
- HU handling for CT, auto-detection of pre-normalized data
- Drag & drop loading, folder scanning

### GPU-Accelerated Rendering
- wgpu-based rendering pipeline (Vulkan/Metal/DX12)
- Raw pixel data uploaded as GPU texture once
- Window/level applied in fragment shader (no per-frame CPU work)
- Mouse-based W/L adjustment (Shift+drag, right-click drag)
- CT presets (keys 1-9), auto-optimal windowing (key 0)

### Multi-Viewport & Synchronization
- Dynamic viewport layouts (1x1, 1x2, 2x2, NxM)
- Tab-based workspace management
- Drag & drop series to viewports
- Scroll sync by frame of reference
- Cross-series reference lines (geometry-based and index-based)
- Viewport maximize/restore

### MPR (Multi-Planar Reconstruction)
- Volume loading from 3D DICOM series
- Axial, coronal, sagittal slice generation
- Gantry tilt correction
- MPR-to-original sync with correct center position
- Reference lines across MPR planes

### PACS Connectivity
- C-FIND (query by patient, study, date, modality)
- C-MOVE (retrieve to local storage)
- C-STORE (send studies)
- PACS configuration UI
- Parallel retriever (4 workers)
- Quick fetch (Ctrl+G)
- Mixed-modality study handling (filters non-image objects)

### Coregistration
- Automatic intensity-based registration
- Mutual information algorithm
- GPU-accelerated computation
- Transform matrix storage
- Visual progress feedback
- 16 pipeline tests on synthetic volumes

### Other
- SQLite database for study/series persistence
- TOML configuration system
- Anonymization (patient data stripping)
- Measurement tools (distance, ROI)
- Annotations
- Hanging protocols
- IPC server (single-instance, open files from CLI)
- Anonymous viewing mode (Ctrl+H)
- O(1) LRU image cache with `IndexMap`

## Architecture

```
src/
├── main.rs                    # Entry point, eframe setup
├── app/                       # Application state (decomposed into subsystems)
│   ├── mod.rs                 # SauhuApp, event handling
│   ├── background.rs          # DB queries, directory scans, image cache
│   ├── coregistration.rs      # Coregistration UI integration
│   ├── image_loading.rs       # Async file loading thread
│   ├── interaction.rs         # Mouse interaction, measurements
│   ├── mpr.rs                 # MPR pipeline management
│   └── quick_fetch.rs         # Ctrl+G PACS retrieve
├── cache.rs                   # Image cache
├── config.rs                  # TOML configuration
├── coregistration/            # Intensity-based coregistration
│   ├── manager.rs             # Registration orchestration
│   ├── metrics.rs             # Mutual information, NCC
│   ├── optimizer.rs           # Powell optimizer
│   ├── pipeline.rs            # Multi-resolution pipeline
│   └── transform.rs           # Affine transforms
├── db/                        # SQLite persistence
│   ├── mod.rs                 # Database connection
│   └── schema.rs              # Tables, migrations
├── dicom/                     # DICOM parsing & processing
│   ├── anonymize.rs           # Patient data stripping
│   ├── geometry.rs            # Geometric calculations
│   ├── image.rs               # Pixel extraction, windowing
│   ├── mpr/                   # Multi-planar reconstruction
│   ├── parser.rs              # DICOM file reading
│   ├── series_utils.rs        # Series grouping, slice location
│   └── spatial.rs             # Spatial calculations
├── gpu/                       # GPU rendering
│   ├── renderer.rs            # wgpu render pipeline
│   ├── texture.rs             # GPU texture management
│   ├── coregistration.rs      # GPU coregistration compute
│   └── shaders/               # WGSL shaders
├── hanging_protocol.rs        # Display protocol rules
├── ipc.rs                     # Single-instance IPC
├── pacs/                      # Re-exports sauhu-common PACS
│   └── mod.rs
└── ui/                        # User interface
    ├── annotations.rs         # Measurement/annotation overlay
    ├── database_window/       # PACS query/retrieve UI
    │   ├── local_browser.rs   # Local study browser
    │   ├── pacs_query.rs      # PACS search UI
    │   └── anonymize.rs       # Anonymization UI
    ├── patient_sidebar.rs     # Patient/study/series tree
    ├── thumbnail_cache.rs     # Series thumbnail generation
    ├── viewport.rs            # Single viewport rendering
    └── viewport_manager.rs    # Multi-viewport layout & sync

sauhu-common/src/              # Shared crate
├── lib.rs
└── pacs/                      # DICOM networking
    ├── query.rs               # C-FIND
    ├── retrieve.rs            # C-MOVE orchestration
    ├── scp.rs                 # Storage SCP (receiver)
    └── scu.rs                 # Service class user (sender)
```

### Binaries

| Name | Purpose |
|------|---------|
| `sauhu` | Main DICOM viewer |
| `sauhu-agent-cli` | CLI for AI agents to inspect DICOM output (6 commands: view, point, roi, measure, info, debug) |
| `dicom_test` | DICOM parsing test harness |
| `pacs_test` | PACS connectivity test |

## CI

GitHub Actions on every push and pull request:

- `cargo test --workspace --lib` (117 tests, <1s)
- `cargo clippy --workspace -- -D warnings`

Workflow: `.github/workflows/ci.yml`

## Key Dependencies

| Category | Crates |
|----------|--------|
| UI | eframe 0.29, egui 0.29 |
| GPU | wgpu 22 |
| DICOM | dicom 0.9, dicom-pixeldata 0.9, dicom-ul 0.9 |
| Database | rusqlite 0.32 (bundled) |
| Async | tokio 1 |
| Image | image 0.25, rayon 1.10 |

## Build

```bash
cargo build --release           # Build release
cargo run --release             # Run viewer
cargo run --release --bin sauhu-agent-cli -- --help  # Agent CLI
cargo test --workspace --lib    # Run tests
cargo clippy --workspace -- -D warnings  # Lint
```

Release profile: `opt-level=3`, `lto=true`, `codegen-units=1`
