//! DICOM texture handling for GPU rendering
#![allow(dead_code)]

use crate::dicom::DicomImage;
use eframe::wgpu;

/// GPU texture holding raw DICOM pixel data
pub struct DicomTexture {
    /// The wgpu texture
    pub texture: wgpu::Texture,
    /// Texture view for binding
    pub view: wgpu::TextureView,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Rescale slope for HU conversion
    pub rescale_slope: f32,
    /// Rescale intercept for HU conversion
    pub rescale_intercept: f32,
}

impl DicomTexture {
    /// Create GPU texture from DICOM image
    ///
    /// Uploads raw u16 pixel data to GPU as R16Uint texture.
    /// The shader will apply rescale and windowing.
    pub fn from_dicom_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &DicomImage,
    ) -> Self {
        let width = image.width;
        let height = image.height;

        // Create texture descriptor for R16Uint
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DICOM Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // R16Uint allows us to store raw u16 values without normalization
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Convert u16 pixels to bytes and upload
        let pixel_bytes: Vec<u8> = image.pixels.iter().flat_map(|&p| p.to_le_bytes()).collect();

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
                bytes_per_row: Some(width * 2), // 2 bytes per pixel (u16)
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            width,
            height,
            rescale_slope: image.rescale_slope as f32,
            rescale_intercept: image.rescale_intercept as f32,
        }
    }

    /// Get texture dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
