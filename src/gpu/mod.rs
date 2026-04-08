//! GPU-accelerated rendering for DICOM images
//!
//! This module provides GPU-based window/level adjustment using wgpu.
//! Benefits:
//! - Raw pixel data uploaded to GPU once (on image load)
//! - Window/level applied in fragment shader
//! - Only uniform buffer updates during windowing (bytes, not megabytes)
//! - 60+ FPS even for large images (3+ megapixels)

pub mod coregistration;
mod renderer;
mod texture;

pub use coregistration::{GpuCoregistration, VolumeUpload};
pub use renderer::{DicomPaintCallback, DicomRenderResources, WindowingUniforms};
