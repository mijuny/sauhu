//! DICOM texture handling for GPU rendering

use crate::dicom::DicomImage;
use eframe::wgpu;
use std::panic;

/// GPU texture holding raw DICOM pixel data
pub struct DicomTexture {
    /// The wgpu texture
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    /// Texture view for binding
    pub view: wgpu::TextureView,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Rescale slope for HU conversion
    #[allow(dead_code)]
    pub rescale_slope: f32,
    /// Rescale intercept for HU conversion
    #[allow(dead_code)]
    pub rescale_intercept: f32,
}

impl DicomTexture {
    /// Create GPU texture from DICOM image
    ///
    /// Uploads raw u16 pixel data to GPU as R16Uint texture.
    /// The shader will apply rescale and windowing.
    ///
    /// Returns `None` if the GPU texture creation or upload fails (e.g., out of VRAM).
    /// The caller should fall back to CPU rendering in that case.
    pub fn from_dicom_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &DicomImage,
    ) -> Option<Self> {
        let width = image.width;
        let height = image.height;

        // Create texture descriptor for R16Uint
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // wgpu panics on allocation failure (OOM, device lost), so catch that
        let texture = match panic::catch_unwind(panic::AssertUnwindSafe(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("DICOM Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        })) {
            Ok(texture) => texture,
            Err(_) => {
                tracing::error!("GPU texture creation failed ({}x{}) - likely out of VRAM", width, height);
                return None;
            }
        };

        // Convert u16 pixels to bytes and upload
        let pixel_bytes: Vec<u8> = image.pixels.iter().flat_map(|&p| p.to_le_bytes()).collect();

        if panic::catch_unwind(panic::AssertUnwindSafe(|| {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixel_bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 2),
                    rows_per_image: Some(height),
                },
                texture_size,
            );
        }))
        .is_err()
        {
            tracing::error!("GPU texture upload failed ({}x{}) - likely out of VRAM", width, height);
            return None;
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Some(Self {
            texture,
            view,
            width,
            height,
            rescale_slope: image.rescale_slope as f32,
            rescale_intercept: image.rescale_intercept as f32,
        })
    }

    /// Get texture dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
