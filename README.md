# Sauhu - Lightning-Fast DICOM Viewer for Linux

> "Sauhu" (Finnish) - a puff of smoke, like the one left behind by Ukko's lightning

All the things a radiologist needs in one fast app. Built in Rust for Linux. Designed for [Omarchy](https://omarchy.com).

**Disclaimer:** Sauhu is not a certified medical device and is not intended for clinical diagnosis. Use at your own risk.

## What It Does

Multi-viewport DICOM viewer with synchronized viewports and reference lines across all views, including coregistered volumes. Query, retrieve, and send studies via PACS (C-FIND, C-MOVE, C-STORE). Reformat volumes into axial, coronal, and sagittal planes with MPR. Coregister volumes from different imaging sessions using GPU-accelerated rigid registration with NCC metric. Measure distances and draw ROIs with calibrated values. CT window presets for brain, subdural, stroke, bone, lung, and more. Three-tier caching (48 GB RAM, 500-texture VRAM, async I/O) keeps scrolling smooth across large studies. Open local DICOM files or connect to any hospital PACS.

## Technical Details

### DICOM Support
- Uncompressed formats (Explicit/Implicit VR, Little/Big Endian)
- JPEG 2000 (OpenJPEG)
- JPEG Baseline/Lossless
- JPEG-LS (CharLS)
- Proper Hounsfield Unit handling for CT

### GPU-Accelerated Rendering
- Real-time window/level adjustment via wgpu shaders
- Smooth 60fps rendering even for large images
- Efficient GPU texture management for multi-viewport

### High-Performance Caching Architecture
Sauhu implements a three-tier caching system designed for radiologist workflow with large datasets:

**1. CPU Image Cache (RAM)**
- LRU cache for decoded DICOM images (default: 48 GB)
- Eliminates disk I/O and JPEG2000/JPEG-LS decoding on repeat views
- Background prefetching of entire series when opened
- 2 parallel worker threads for prefetch decoding

**2. GPU Texture Cache (VRAM)**
- Caches uploaded GPU textures (up to 500 textures)
- Eliminates 80ms+ texture upload time on cached images
- Critical for smooth synchronized multi-viewport scrolling
- LRU eviction with automatic cache management

**3. Async Background Operations**
- All file validation runs in background threads
- Directory scanning is non-blocking
- Database queries are async
- UI remains responsive during all I/O operations

**Performance Characteristics:**
| Operation | First Time | Cached |
|-----------|------------|--------|
| Load image from disk | 50-200ms | - |
| Decode JPEG2000 | 100-300ms | 0ms (RAM cache) |
| GPU texture upload | 50-80ms | 0ms (GPU cache) |
| Scroll (cached) | - | <1ms |

With 8 synced MRI viewports, scrolling remains smooth after initial cache fill.

### Multi-Viewport Support
- Up to 8 simultaneous viewports (1x1, 1x2, 2x2, 2x3, 2x4 layouts)
- Drag & drop series between viewports
- Reference lines showing slice positions across all viewports, including MPR and coregistered volumes
- Synchronized scrolling between related series

### PACS Connectivity
- Full DICOM network protocol support
- C-FIND: Query studies by patient name, ID, accession number, date
- C-MOVE: Retrieve studies to local storage
- Storage SCP: Receive pushed images
- Quick fetch (Ctrl+G): Paste accession number from clipboard

### Patient Database
- SQLite database for local study management
- Patient browser with study hierarchy
- Series thumbnails with async loading
- Smart series naming (anatomy + technique)

### Measurement Tools
- **Length measurement (L)**: Click two points to measure distance
  - Calibrated in mm when PixelSpacing available
  - Falls back to ImagerPixelSpacing for CR/DR/DX modalities
  - Shows "px (no calibration)" when no spacing data
- **Circle ROI (O)**: Click-drag to draw circular region of interest
  - Calculates mean, min, max, standard deviation
  - Shows values in HU for CT images
- **ROI-based windowing (Shift+W)**: Draw ROI to auto-set window/level
- Annotations persist per image (SOP Instance UID)
- Delete all annotations (Shift+D)

### Smart Image Display
- **Smart Zoom (Space)**: Automatically detects anatomy bounds and zooms to fit actual content
  - CT: Threshold at -500 HU (above air)
  - Other modalities: Percentile-based detection
- **Smart Windowing (Shift+A)**: Automatically calculates optimal window/level for non-CT images
- Auto-applies smart fit and smart window when loading new series
- Regular fit to window (Shift+Space)

### MPR (Multi-Planar Reconstruction)
- Reformat 3D volumetric series into different anatomical planes
- **A**: Switch to Axial view (view from feet)
- **C**: Switch to Coronal view (view from front)
- **S**: Switch to Sagittal view (view from side)
- **X**: Return to original acquisition plane
- Uses trilinear interpolation for smooth reformatted images
- Volume built lazily on first MPR request, cached for instant plane switching
- Navigate reformatted slices with arrow keys

### Volume Coregistration
- GPU-accelerated rigid registration (6 DOF) using wgpu compute shaders
- Normalized cross-correlation (NCC) similarity metric
- Multi-resolution pyramid optimization for speed and robustness
- Trilinear resampling of source volume to target space
- Workflow: `Ctrl+R` to enter coregistration mode, click target, click source, view result
- This is a feature rarely found in open-source DICOM viewers

### Window/Level
- Interactive adjustment (W mode or right-click drag)
- CT presets (keys 1-9): Brain, Subdural, Stroke, Bone, Soft Tissue, Lung, Liver, Mediastinum, Abdomen
- Auto-optimal windowing (key 0)
- Scroll mode (Shift+S) for smooth series navigation

### Navigation
- Series browsing (arrow keys, scroll wheel)
- Smooth scroll mode with right-click drag
- Drag & drop file/folder loading
- File picker (Ctrl+O)
- Patient sidebar with series thumbnails

## Quick Start

```bash
# Build
sudo pacman -S cmake       # Arch Linux (or apt install cmake on Debian/Ubuntu)
git clone https://github.com/mijuny/sauhu.git
cd sauhu
cargo build --release

# Open a DICOM file or folder
./target/release/sauhu scan.dcm
./target/release/sauhu /path/to/dicom/folder/
```

That's it. Sauhu opens the files, detects series automatically, and displays them with proper windowing.

## Opening Images

### Local files

Sauhu works with local DICOM files out of the box, no PACS required.

**From the command line:**
```bash
sauhu file.dcm                    # Single file
sauhu /path/to/folder/            # Folder (scanned recursively)
sauhu file1.dcm file2.dcm dir/   # Mix of files and folders
```

**From the GUI:**
- `Ctrl+O` or File > Open File(s) to pick DICOM files
- File > Open Folder to open an entire directory
- Drag and drop files or folders onto the window

Sauhu groups files into series automatically based on DICOM metadata. Multi-series studies get separate entries in the patient sidebar (`B` to toggle).

### PACS (hospital network)

To retrieve images from a hospital PACS, you need to configure a connection. Edit `~/.config/sauhu/config.toml`:

```toml
[pacs]
local_ae_title = "SAUHU"       # AE title for this machine (register with your PACS admin)
local_port = 11113              # Port for incoming C-STORE from PACS

[[pacs.servers]]
name = "Hospital PACS"
ae_title = "YOUR_PACS_AET"     # PACS AE title (ask your PACS admin)
host = "pacs.hospital.local"   # PACS hostname or IP
port = 104                     # PACS port (104 is standard)
```

**What to ask your PACS administrator:**
1. The PACS AE title and hostname/IP
2. Register your AE title (e.g. `SAUHU`) on the PACS so it accepts your queries
3. Ensure your machine can reach the PACS host and that the PACS can connect back to your SCP port (11113) for C-MOVE

**Test the connection:**
```bash
sauhu pacs find --patient-name "TEST*"
```

**Retrieve a study:**
```bash
# Quick fetch in GUI: Ctrl+G, paste accession number
# Or from CLI:
sauhu pacs find --patient-id 12345
sauhu pacs move --study-uid 1.2.3.4.5.6.7
```

### Database

Sauhu stores metadata for opened studies in a local SQLite database at `~/.local/share/sauhu/sauhu.db`. Press `D` to open the database browser, which shows all previously opened studies and allows searching and re-opening them.

## Installation

### Requirements

- Rust 1.70+
- cmake (for JPEG-LS support)
- wgpu-compatible GPU (Vulkan, Metal, or DX12)

### Build

```bash
# Install dependencies (Arch Linux)
sudo pacman -S cmake

# Ubuntu/Debian
# sudo apt install cmake build-essential

# Build
cargo build --release

# Optional: install to PATH
cargo install --path .
```

## Usage

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Scroll` / `Arrow Up/Down` | Navigate images |
| `W` | Toggle windowing mode |
| `Right Drag` | Window/Level (in any mode) |
| `Ctrl + Drag` / `Mouse Wheel` | Zoom |
| `Left Drag` / `Middle Drag` | Pan |
| `Space` | Smart fit to anatomy |
| `Shift+Space` | Regular fit to window |
| `Shift+A` | Smart auto-windowing (non-CT) |
| `0` | Auto-optimal windowing |
| `1-9` | CT presets |
| `L` | Length measurement mode |
| `O` | Circle ROI mode |
| `Shift+W` | ROI-based windowing mode |
| `Shift+D` | Delete all annotations |
| `A` | MPR: Axial view |
| `C` | MPR: Coronal view |
| `S` | MPR: Sagittal view |
| `X` | MPR: Original view |
| `Y` | Toggle synchronized scrolling |
| `Shift+S` | Scroll mode (right-drag to scroll) |
| `D` | Toggle patient database |
| `Ctrl+G` | Quick fetch (accession from clipboard) |
| `Ctrl+O` | Open file |
| `Escape` | Cancel measurement / return to normal |

### CT Window Presets

| Key | Preset | WC/WW |
|-----|--------|-------|
| 1 | Brain | 40/80 |
| 2 | Subdural | 75/215 |
| 3 | Stroke | 40/40 |
| 4 | Bone | 400/1800 |
| 5 | Soft Tissue | 50/400 |
| 6 | Lung | -600/1500 |
| 7 | Liver | 60/160 |
| 8 | Mediastinum | 50/350 |
| 9 | Abdomen | 40/400 |

### Viewport Layouts

| Key | Layout |
|-----|--------|
| F1 | 1x1 (single viewport) |
| F2 | 1x2 (two side-by-side) |
| F3 | 2x2 (four viewports) |
| F4 | 2x3 (six viewports) |
| F5 | 2x4 (eight viewports) |

### Configuration

Settings are stored in `~/.config/sauhu/config.toml`:

```toml
[pacs]
local_ae_title = "SAUHU"
local_port = 11113
storage_dir = "~/.local/share/sauhu/dicom"

[[pacs.servers]]
name = "Hospital PACS"
ae_title = "PACS"
host = "pacs.hospital.local"
port = 104

[shortcuts]
length_measure = "L"
circle_roi = "O"
sync_toggle = "Y"
scroll_mode = "Shift+S"
windowing_mode = "W"
roi_windowing = "Shift+W"
smart_fit = "Space"
regular_fit = "Shift+Space"
delete_annotations = "Shift+D"
smart_window = "Shift+A"
mpr_axial = "A"
mpr_coronal = "C"
mpr_sagittal = "S"
mpr_original = "X"

[cache]
max_memory_gb = 48      # RAM cache limit (default: 48 GB)
prefetch_ahead = 15     # Prefetch images ahead of current
prefetch_behind = 5     # Keep images behind current cached
```

For systems with large RAM (64GB+), increase `max_memory_gb` for better performance with multiple large series.

## Architecture

```
sauhu/
├── src/
│   ├── main.rs                  # Entry point, eframe setup
│   ├── app/                     # Application state (decomposed into subsystems)
│   │   ├── mod.rs               # SauhuApp, event handling
│   │   ├── background.rs        # DB queries, directory scans, image cache
│   │   ├── coregistration.rs    # Coregistration UI integration
│   │   ├── image_loading.rs     # Async file loading thread
│   │   ├── interaction.rs       # Mouse interaction, measurements
│   │   ├── mpr.rs               # MPR pipeline management
│   │   └── quick_fetch.rs       # Ctrl+G PACS retrieve
│   ├── config.rs                # TOML configuration with shortcuts
│   ├── cache.rs                 # O(1) LRU image cache with prefetching
│   ├── ipc.rs                   # Unix socket IPC for external apps
│   ├── hanging_protocol.rs      # Display protocol rules
│   ├── dicom/
│   │   ├── parser.rs            # DICOM file parsing
│   │   ├── image.rs             # Pixel data, ROI stats, anatomy detection
│   │   ├── anonymize.rs         # Patient data stripping
│   │   ├── geometry.rs          # Image planes, reference lines
│   │   ├── spatial.rs           # Spatial calculations
│   │   ├── series_utils.rs      # Series grouping, slice location
│   │   └── mpr/                 # MPR volume construction and resampling
│   ├── ui/
│   │   ├── viewport.rs          # Single viewport rendering
│   │   ├── viewport_manager.rs  # Multi-viewport layout & sync
│   │   ├── annotations.rs       # Measurement data structures
│   │   ├── patient_sidebar.rs   # Series browser
│   │   ├── thumbnail_cache.rs   # Async thumbnail loading
│   │   └── database_window/     # Patient/PACS browser
│   │       ├── local_browser.rs # Local study browser
│   │       ├── pacs_query.rs    # PACS search & retrieve UI
│   │       └── anonymize.rs     # Anonymization UI
│   ├── gpu/
│   │   ├── renderer.rs          # wgpu pipeline + GPU texture cache
│   │   ├── texture.rs           # GPU texture management
│   │   ├── coregistration.rs    # GPU coregistration compute
│   │   └── shaders/             # WGSL shaders
│   ├── coregistration/          # Intensity-based coregistration
│   │   ├── manager.rs           # Registration orchestration
│   │   ├── metrics.rs           # NCC similarity metric
│   │   ├── optimizer.rs         # Powell optimizer
│   │   ├── pipeline.rs          # Multi-resolution pipeline
│   │   └── transform.rs         # Affine transforms
│   ├── pacs/                    # Re-exports sauhu-common
│   └── db/
│       └── schema.rs            # SQLite schema & migrations
├── sauhu-common/src/            # Shared DICOM networking crate
│   └── pacs/
│       ├── query.rs             # C-FIND
│       ├── retrieve.rs          # C-MOVE orchestration
│       ├── scp.rs               # Storage SCP (receiver)
│       └── scu.rs               # Service class user (sender)
```

### Data Flow

```
                    ┌─────────────────────────────────────────────────┐
                    │                  GPU VRAM                        │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │     Texture Cache (500 textures LRU)     │   │
                    │  └──────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                          ▲
                                          │ bind (instant if cached)
                                          │ upload (~80ms if miss)
┌──────────────┐    ┌─────────────────────┴───────────────────────────┐
│              │    │                    RAM                           │
│    Disk      │───▶│  ┌──────────────────────────────────────────┐   │
│   (DICOM)    │    │  │      Image Cache (48 GB LRU)             │   │
│              │    │  │  - Decoded DicomImage structs            │   │
└──────────────┘    │  │  - Prefetch workers (2 threads)          │   │
       │            │  └──────────────────────────────────────────┘   │
       │            └─────────────────────────────────────────────────┘
       │                              ▲
       │ decode (100-300ms)           │ lookup (<1ms)
       │ (JPEG2000, JPEG-LS)          │
       ▼                              │
┌──────────────────────────────────────┘
│  Background Worker Threads
│  - File validation
│  - DICOM decoding
│  - Prefetch scheduling
└──────────────────────────────────────
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Rust |
| UI Framework | egui + eframe |
| GPU Rendering | wgpu |
| DICOM Parsing | dicom-rs |
| Image Decoding | dicom-pixeldata (OpenJPEG, CharLS) |
| DICOM Networking | dicom-ul, dicom-transfer-syntax-registry |
| Database | SQLite (rusqlite) |

## Development Status

### Completed
- [x] DICOM parsing with multi-format codec support
- [x] GPU-accelerated windowing (wgpu)
- [x] Multi-viewport layouts (up to 8 viewports)
- [x] Reference lines between viewports (all views including coregistered)
- [x] Synchronized scrolling
- [x] PACS connectivity (C-FIND, C-MOVE, C-STORE)
- [x] Patient database with SQLite
- [x] Measurement tools (length, circle ROI)
- [x] Smart zoom and smart windowing
- [x] Configurable keyboard shortcuts
- [x] High-performance image caching (RAM + GPU)
- [x] Background prefetching with parallel workers
- [x] Async file operations (no UI freezing)
- [x] IPC integration for external apps (Sanelu)
- [x] MPR (Multi-Planar Reconstruction) with trilinear interpolation
- [x] Volume coregistration (GPU-accelerated rigid registration)
- [x] Study anonymization
- [x] CI (GitHub Actions: tests + clippy)

### Planned
- [ ] KO (Key Object Selection) and SR (Structured Reports) display
- [ ] Image fusion (alpha blend, color overlay, checkerboard)

## IPC Integration

Sauhu provides a Unix socket IPC interface for external applications (like Sanelu dictation app) to control DICOM viewing.

### Socket Location

```
$XDG_RUNTIME_DIR/sauhu.sock
# e.g., /run/user/1000/sauhu.sock
```

### Protocol

JSON-based request/response over Unix socket. Each message is a single line terminated by newline.

### Commands

#### open_study
Opens a study by accession number. Queries configured PACS and retrieves if not cached locally.

Request:
```json
{"command": "open_study", "accession": "12345678"}
```

Response (success):
```json
{"status": "ok", "message": "Opening study: 12345678"}
```

Response (error):
```json
{"status": "error", "message": "No PACS server configured"}
```

#### ping
Check if Sauhu is running.

Request:
```json
{"command": "ping"}
```

Response:
```json
{"status": "ok", "message": "pong"}
```

### Example Client (Bash)

```bash
# Check if Sauhu is running
echo '{"command": "ping"}' | nc -U /run/user/$(id -u)/sauhu.sock

# Open study by accession number
echo '{"command": "open_study", "accession": "12345678"}' | nc -U /run/user/$(id -u)/sauhu.sock
```

### Integration with Sanelu

Sanelu dictation app's "Kuvat" button sends accession number from clipboard to Sauhu via IPC. This enables seamless workflow:

1. Copy accession number from RIS
2. Dictate report in Sanelu
3. Click "Kuvat" to view images in Sauhu

## Acknowledgments

Sauhu is built on excellent open source libraries:

- [dicom-rs](https://github.com/Edd-the-Dev/dicom-rs) by Eduardo Pinho - DICOM parsing, pixel decoding, and networking
- [egui](https://github.com/emilk/egui) by Emil Ernerfeldt - immediate mode GUI
- [wgpu](https://github.com/gfx-rs/wgpu) - cross-platform GPU abstraction
- [OpenJPEG](https://www.openjpeg.org/) and [CharLS](https://github.com/team-charls/charls) - JPEG 2000 and JPEG-LS decoding
- [rusqlite](https://github.com/rusqlite/rusqlite) - SQLite bindings
- [rayon](https://github.com/rayon-rs/rayon) - parallel processing
- [tokio](https://github.com/tokio-rs/tokio) - async runtime

And to the desktop environment that made building this worthwhile:

- [Hyprland](https://hyprland.org) by Vaxry - the Wayland compositor that makes Linux a viable daily-driver workstation
- [Omarchy](https://omarchy.com) - the opinionated Arch desktop that ties it all together

Thank you to all contributors and maintainers, and to the countless open source contributors whose work makes projects like this possible.

## License

Copyright 2025-2026 Mikko Nyman

[PolyForm Noncommercial 1.0.0](LICENSE) - Free for personal, research, educational, and nonprofit use. Not for commercial use.

## Author

Mikko Nyman - Neuroradiologist, Turku University Hospital
[nyman.xyz](https://nyman.xyz)
