//! GPU renderer for DICOM windowing

use super::texture::DicomTexture;
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, CallbackResources};
use eframe::wgpu;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU64;
use std::path::PathBuf;
use wgpu::util::DeviceExt;

/// Uniform buffer data for window/level shader
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct WindowingUniforms {
    /// Window center (in HU for CT)
    pub window_center: f32,
    /// Window width
    pub window_width: f32,
    /// Rescale slope
    pub rescale_slope: f32,
    /// Rescale intercept
    pub rescale_intercept: f32,
    /// Scale factor for zoom (x, y)
    pub scale: [f32; 2],
    /// Offset for pan (x, y) in NDC
    pub offset: [f32; 2],
    /// Pixel representation (0 = unsigned, 1 = signed)
    pub pixel_representation: u32,
    /// Padding for alignment
    pub _padding: u32,
}

/// Uniform buffer data for fusion shader
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct FusionUniforms {
    pub opacity: f32,
    pub blend_mode: u32,
    pub base_wc: f32,
    pub base_ww: f32,
    pub base_slope: f32,
    pub base_intercept: f32,
    pub base_pixel_rep: u32,
    pub overlay_wc: f32,
    pub overlay_ww: f32,
    pub overlay_slope: f32,
    pub overlay_intercept: f32,
    pub overlay_pixel_rep: u32,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub colormap_index: u32,
    pub checker_size: f32,
    pub threshold: f32,
    pub _padding: f32,
}

/// Maximum number of viewport texture slots
pub const MAX_TEXTURE_SLOTS: usize = 8;

/// Maximum number of textures in the GPU cache
pub const MAX_CACHED_TEXTURES: usize = 500;

/// Cached texture entry with bind group
struct CachedTexture {
    texture: DicomTexture,
}

/// GPU resources for DICOM rendering, stored in CallbackResources
pub struct DicomRenderResources {
    /// Render pipeline
    pub pipeline: wgpu::RenderPipeline,
    /// Bind group layout for creating bind groups
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffers for each viewport slot (separate so each viewport can have different WL)
    pub uniform_buffers: Vec<wgpu::Buffer>,
    /// DICOM textures for each viewport slot (up to 8) - now references cached textures
    pub dicom_textures: Vec<Option<DicomTexture>>,
    /// Bind groups for each viewport slot (texture + uniforms)
    pub bind_groups: Vec<Option<wgpu::BindGroup>>,
    /// Image dimensions for each slot (for aspect ratio calculation)
    pub image_dims: Vec<Option<(u32, u32)>>,
    /// GPU texture cache: path -> cached texture
    texture_cache: HashMap<PathBuf, CachedTexture>,
    /// LRU order for texture cache eviction
    texture_cache_lru: VecDeque<PathBuf>,
    /// Which path is currently bound to each slot
    slot_paths: Vec<Option<PathBuf>>,
    /// Fusion render pipeline (for blending two textures)
    pub fusion_pipeline: wgpu::RenderPipeline,
    /// Fusion bind group layout
    pub fusion_bind_group_layout: wgpu::BindGroupLayout,
    /// Fusion uniform buffers (one per viewport slot)
    pub fusion_uniform_buffers: Vec<wgpu::Buffer>,
    /// Fusion bind groups: (base_texture, overlay_texture, uniforms)
    /// Created dynamically when fusion is active
    pub fusion_bind_groups: Vec<Option<wgpu::BindGroup>>,
}

impl DicomRenderResources {
    /// Create new GPU resources for DICOM rendering
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Windowing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/windowing.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Windowing Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<WindowingUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // DICOM texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Windowing Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Windowing Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Create default uniforms
        let default_uniforms = WindowingUniforms {
            window_center: 40.0,
            window_width: 400.0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            scale: [1.0, 1.0],
            offset: [0.0, 0.0],
            pixel_representation: 0,
            _padding: 0,
        };

        // Create per-slot uniform buffers (each viewport needs its own to have independent WL)
        let mut uniform_buffers = Vec::with_capacity(MAX_TEXTURE_SLOTS);
        for slot in 0..MAX_TEXTURE_SLOTS {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Windowing Uniforms Slot {}", slot)),
                contents: bytemuck::cast_slice(&[default_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            uniform_buffers.push(buffer);
        }

        // Initialize slot vectors
        let mut dicom_textures = Vec::with_capacity(MAX_TEXTURE_SLOTS);
        let mut bind_groups = Vec::with_capacity(MAX_TEXTURE_SLOTS);
        let mut image_dims = Vec::with_capacity(MAX_TEXTURE_SLOTS);
        let mut slot_paths = Vec::with_capacity(MAX_TEXTURE_SLOTS);
        for _ in 0..MAX_TEXTURE_SLOTS {
            dicom_textures.push(None);
            bind_groups.push(None);
            image_dims.push(None);
            slot_paths.push(None);
        }

        // === Fusion pipeline ===
        let fusion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fusion Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fusion.wgsl").into()),
        });

        let fusion_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fusion Bind Group Layout"),
                entries: &[
                    // Fusion uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<FusionUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    // Base DICOM texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Overlay DICOM texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let fusion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fusion Pipeline Layout"),
                bind_group_layouts: &[&fusion_bind_group_layout],
                push_constant_ranges: &[],
            });

        let fusion_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fusion Pipeline"),
            layout: Some(&fusion_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fusion_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fusion_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let default_fusion_uniforms = FusionUniforms {
            opacity: 0.5,
            blend_mode: 0,
            base_wc: 40.0,
            base_ww: 400.0,
            base_slope: 1.0,
            base_intercept: 0.0,
            base_pixel_rep: 0,
            overlay_wc: 40.0,
            overlay_ww: 400.0,
            overlay_slope: 1.0,
            overlay_intercept: 0.0,
            overlay_pixel_rep: 0,
            scale: [1.0, 1.0],
            offset: [0.0, 0.0],
            colormap_index: 0,
            checker_size: 32.0,
            threshold: 0.0,
            _padding: 0.0,
        };

        let mut fusion_uniform_buffers = Vec::with_capacity(MAX_TEXTURE_SLOTS);
        let mut fusion_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(MAX_TEXTURE_SLOTS);
        for slot in 0..MAX_TEXTURE_SLOTS {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Fusion Uniforms Slot {}", slot)),
                contents: bytemuck::cast_slice(&[default_fusion_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            fusion_uniform_buffers.push(buffer);
            fusion_bind_groups.push(None);
        }

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffers,
            dicom_textures,
            bind_groups,
            image_dims,
            texture_cache: HashMap::new(),
            texture_cache_lru: VecDeque::new(),
            slot_paths,
            fusion_pipeline,
            fusion_bind_group_layout,
            fusion_uniform_buffers,
            fusion_bind_groups,
        }
    }

    /// Set DICOM texture for a specific viewport slot (called when new image is loaded)
    #[allow(dead_code)] // GPU rendering API
    pub fn set_texture_for_slot(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        texture: DicomTexture,
    ) {
        if slot >= MAX_TEXTURE_SLOTS {
            tracing::warn!("Invalid texture slot {} (max {})", slot, MAX_TEXTURE_SLOTS);
            return;
        }

        self.image_dims[slot] = Some(texture.dimensions());

        // Create bind group with new texture and per-slot uniform buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Windowing Bind Group Slot {}", slot)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffers[slot].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
            ],
        });

        self.dicom_textures[slot] = Some(texture);
        self.bind_groups[slot] = Some(bind_group);
    }

    /// Set DICOM texture for slot 0 (backwards compatibility)
    #[allow(dead_code)] // GPU rendering API
    pub fn set_texture(&mut self, device: &wgpu::Device, texture: DicomTexture) {
        self.set_texture_for_slot(device, 0, texture);
    }

    /// Clear texture from a specific slot
    #[allow(dead_code)] // GPU rendering API
    pub fn clear_texture_slot(&mut self, slot: usize) {
        if slot < MAX_TEXTURE_SLOTS {
            self.dicom_textures[slot] = None;
            self.bind_groups[slot] = None;
            self.image_dims[slot] = None;
        }
    }

    /// Clear all textures (for patient switching)
    pub fn clear_all_textures(&mut self) {
        for slot in 0..MAX_TEXTURE_SLOTS {
            self.dicom_textures[slot] = None;
            self.bind_groups[slot] = None;
            self.image_dims[slot] = None;
            self.slot_paths[slot] = None;
        }
        // Also clear texture cache when switching patients
        self.texture_cache.clear();
        self.texture_cache_lru.clear();
        tracing::debug!("GPU texture cache cleared");
    }

    /// Check if texture for path is already cached in GPU
    pub fn is_texture_cached(&self, path: &PathBuf) -> bool {
        self.texture_cache.contains_key(path)
    }

    /// Bind cached texture to slot (returns true if found in cache)
    /// This is the fast path - no GPU upload needed
    pub fn bind_cached_texture(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        path: &PathBuf,
    ) -> bool {
        if slot >= MAX_TEXTURE_SLOTS {
            return false;
        }

        // Check if this path is already bound to this slot
        if let Some(ref current_path) = self.slot_paths[slot] {
            if current_path == path {
                // Already bound, nothing to do
                return true;
            }
        }

        // Check if texture is in cache
        if let Some(cached) = self.texture_cache.get(path) {
            // Update LRU order
            self.texture_cache_lru.retain(|p| p != path);
            self.texture_cache_lru.push_back(path.clone());

            // Create bind group with cached texture
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Cached Bind Group Slot {}", slot)),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffers[slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&cached.texture.view),
                    },
                ],
            });

            self.image_dims[slot] = Some(cached.texture.dimensions());
            self.bind_groups[slot] = Some(bind_group);
            self.slot_paths[slot] = Some(path.clone());

            tracing::trace!("GPU cache hit: {:?}", path.file_name().unwrap_or_default());
            return true;
        }

        false
    }

    /// Upload texture to cache only (without binding to slot)
    /// Use this for bulk uploads like MPR series
    pub fn upload_to_cache(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: PathBuf,
        image: &crate::dicom::DicomImage,
    ) {
        // Skip if already cached
        if self.texture_cache.contains_key(&path) {
            return;
        }

        // Evict oldest if cache is full
        while self.texture_cache.len() >= MAX_CACHED_TEXTURES {
            if let Some(old_path) = self.texture_cache_lru.pop_front() {
                self.texture_cache.remove(&old_path);
                tracing::trace!(
                    "GPU cache evicted: {:?}",
                    old_path.file_name().unwrap_or_default()
                );
            }
        }

        // Upload texture
        let texture = DicomTexture::from_dicom_image(device, queue, image);

        // Add to cache (don't bind to any slot)
        self.texture_cache
            .insert(path.clone(), CachedTexture { texture });
        self.texture_cache_lru.push_back(path);

        tracing::trace!("GPU cache: uploaded texture to cache");
    }

    /// Upload and cache texture for path, then bind to slot
    pub fn upload_and_cache_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        path: PathBuf,
        image: &crate::dicom::DicomImage,
    ) {
        if slot >= MAX_TEXTURE_SLOTS {
            return;
        }

        // Upload to cache first
        self.upload_to_cache(device, queue, path.clone(), image);

        // Now bind from cache
        self.bind_cached_texture(device, slot, &path);
    }

    /// Update uniforms for a specific slot (window/level and transform)
    pub fn update_uniforms_for_slot(
        &self,
        queue: &wgpu::Queue,
        slot: usize,
        uniforms: WindowingUniforms,
    ) {
        if slot < MAX_TEXTURE_SLOTS {
            queue.write_buffer(
                &self.uniform_buffers[slot],
                0,
                bytemuck::cast_slice(&[uniforms]),
            );
        }
    }

    /// Update uniforms for slot 0 (backwards compatibility)
    #[allow(dead_code)] // GPU rendering API
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: WindowingUniforms) {
        self.update_uniforms_for_slot(queue, 0, uniforms);
    }

    /// Check if resources have a texture loaded in a specific slot
    #[allow(dead_code)] // GPU rendering API
    pub fn has_texture_in_slot(&self, slot: usize) -> bool {
        slot < MAX_TEXTURE_SLOTS && self.dicom_textures[slot].is_some()
    }

    /// Check if resources have a texture loaded in slot 0 (backwards compatibility)
    #[allow(dead_code)] // GPU rendering API
    pub fn has_texture(&self) -> bool {
        self.has_texture_in_slot(0)
    }

    /// Get bind group for a specific slot
    pub fn get_bind_group(&self, slot: usize) -> Option<&wgpu::BindGroup> {
        if slot < MAX_TEXTURE_SLOTS {
            self.bind_groups[slot].as_ref()
        } else {
            None
        }
    }

    /// Create a fusion bind group for a viewport slot.
    /// Combines the base texture (from `base_slot`) and overlay texture (from `overlay_slot`)
    /// with the fusion uniform buffer for `target_slot`.
    pub fn create_fusion_bind_group(
        &mut self,
        device: &wgpu::Device,
        target_slot: usize,
        base_path: &PathBuf,
        overlay_path: &PathBuf,
    ) -> bool {
        if target_slot >= MAX_TEXTURE_SLOTS {
            return false;
        }

        // Get texture views from cache
        let base_view = self
            .texture_cache
            .get(base_path)
            .map(|ct| &ct.texture.view);
        let overlay_view = self
            .texture_cache
            .get(overlay_path)
            .map(|ct| &ct.texture.view);

        match (base_view, overlay_view) {
            (Some(base_v), Some(overlay_v)) => {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Fusion Bind Group Slot {}", target_slot)),
                    layout: &self.fusion_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.fusion_uniform_buffers[target_slot]
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(base_v),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(overlay_v),
                        },
                    ],
                });
                self.fusion_bind_groups[target_slot] = Some(bind_group);
                true
            }
            _ => {
                tracing::warn!(
                    "Fusion: cannot create bind group for slot {} - missing textures",
                    target_slot
                );
                false
            }
        }
    }

    /// Update fusion uniforms for a specific slot
    pub fn update_fusion_uniforms(
        &self,
        queue: &wgpu::Queue,
        slot: usize,
        uniforms: FusionUniforms,
    ) {
        if slot < MAX_TEXTURE_SLOTS {
            queue.write_buffer(
                &self.fusion_uniform_buffers[slot],
                0,
                bytemuck::cast_slice(&[uniforms]),
            );
        }
    }

    /// Get fusion bind group for a specific slot
    pub fn get_fusion_bind_group(&self, slot: usize) -> Option<&wgpu::BindGroup> {
        if slot < MAX_TEXTURE_SLOTS {
            self.fusion_bind_groups[slot].as_ref()
        } else {
            None
        }
    }

    /// Clear fusion bind group for a slot
    #[allow(dead_code)] // GPU rendering API
    pub fn clear_fusion_bind_group(&mut self, slot: usize) {
        if slot < MAX_TEXTURE_SLOTS {
            self.fusion_bind_groups[slot] = None;
        }
    }
}

/// Paint callback for custom DICOM rendering
pub struct DicomPaintCallback {
    pub uniforms: WindowingUniforms,
    /// Which texture slot to render (0-7)
    pub slot: usize,
}

/// Paint callback for fusion rendering (two textures blended)
pub struct FusionPaintCallback {
    pub uniforms: FusionUniforms,
    /// Which texture slot's fusion bind group to use (0-7)
    pub slot: usize,
}

impl egui_wgpu::CallbackTrait for DicomPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Update uniforms for this slot (each viewport has its own uniform buffer)
        if let Some(resources) = callback_resources.get::<DicomRenderResources>() {
            resources.update_uniforms_for_slot(queue, self.slot, self.uniforms);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(resources) = callback_resources.get::<DicomRenderResources>() {
            if let Some(bind_group) = resources.get_bind_group(self.slot) {
                render_pass.set_pipeline(&resources.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                // Draw fullscreen quad (6 vertices for 2 triangles, generated in vertex shader)
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

impl egui_wgpu::CallbackTrait for FusionPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(resources) = callback_resources.get::<DicomRenderResources>() {
            resources.update_fusion_uniforms(queue, self.slot, self.uniforms);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(resources) = callback_resources.get::<DicomRenderResources>() {
            if let Some(bind_group) = resources.get_fusion_bind_group(self.slot) {
                render_pass.set_pipeline(&resources.fusion_pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}
