//! GPU-accelerated coregistration
//!
//! Uses compute shaders for fast volume resampling and NCC metric calculation.
//!
//! CRITICAL: Transform operates in PHYSICAL (mm) coordinates
//! This ensures transform parameters are independent of resolution level.

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use wgpu::util::DeviceExt;

/// Threads per workgroup for NCC reduction shader
const NCC_WORKGROUP_SIZE: usize = 256;
/// Workgroup size for 3D volume resampling (4x4x4 = 64 threads)
const RESAMPLE_WORKGROUP_SIZE: u32 = 4;

/// Uniforms for transform and resampling
///
/// Transform is applied in physical (mm) coordinates:
/// 1. Target voxel -> physical coords (using target_spacing)
/// 2. Apply inverse transform
/// 3. Physical coords -> source voxel (using source_spacing)
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct TransformUniforms {
    /// 4x4 inverse transform matrix (column-major, f32) - operates in mm
    pub transform: [[f32; 4]; 4],
    /// Target volume dimensions (width, height, depth, padding)
    pub target_dims: [u32; 4],
    /// Source volume dimensions (width, height, depth, padding)
    pub source_dims: [u32; 4],
    /// Target voxel spacing in mm (x, y, z, padding)
    pub target_spacing: [f32; 4],
    /// Source voxel spacing in mm (x, y, z, padding)
    pub source_spacing: [f32; 4],
}

/// Uniforms for NCC calculation
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct NccUniforms {
    /// Volume dimensions
    pub dims: [u32; 4],
    /// Number of voxels
    pub num_voxels: u32,
    /// Padding
    pub _padding: [u32; 3],
}

/// Partial sums from NCC reduction (matches shader struct)
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct NccPartials {
    pub sum_target: f32,
    pub sum_source: f32,
    pub sum_target_sq: f32,
    pub sum_source_sq: f32,
    pub sum_product: f32,
    pub count: u32,
    pub _padding: [u32; 2],
}

/// Volume data for GPU upload (normalized f32 voxels + geometry)
pub struct VolumeUpload<'a> {
    pub data: &'a [f32],
    pub dims: (usize, usize, usize),
    pub spacing: (f64, f64, f64),
}

/// GPU coregistration resources
pub struct GpuCoregistration {
    /// Resample compute pipeline
    resample_pipeline: wgpu::ComputePipeline,
    /// NCC reduction compute pipeline
    ncc_pipeline: wgpu::ComputePipeline,
    /// Bind group layouts
    resample_bind_group_layout: wgpu::BindGroupLayout,
    ncc_bind_group_layout: wgpu::BindGroupLayout,
    /// Target volume buffer (uploaded once)
    target_buffer: Option<wgpu::Buffer>,
    /// Source volume buffer (uploaded once)
    source_buffer: Option<wgpu::Buffer>,
    /// Resampled volume buffer (output of resample, input to NCC)
    resampled_buffer: Option<wgpu::Buffer>,
    /// Transform uniform buffer
    transform_uniform_buffer: wgpu::Buffer,
    /// NCC uniform buffer
    ncc_uniform_buffer: wgpu::Buffer,
    /// NCC partials buffer (output of reduction)
    partials_buffer: Option<wgpu::Buffer>,
    /// Partials staging buffer for CPU readback
    partials_staging_buffer: Option<wgpu::Buffer>,
    /// Cached resample bind group. Rebuilt only when the bound buffers
    /// change (i.e. inside `upload_volumes`). Previously compute_ncc created
    /// it on every optimiser step, which burned noticeable time inside the
    /// Powell line-search inner loop.
    resample_bind_group: Option<wgpu::BindGroup>,
    /// Cached NCC bind group, same lifetime story as the resample one.
    ncc_bind_group: Option<wgpu::BindGroup>,
    /// Target volume dimensions
    target_dims: (usize, usize, usize),
    /// Source volume dimensions
    source_dims: (usize, usize, usize),
    /// Target voxel spacing in mm
    target_spacing: (f64, f64, f64),
    /// Source voxel spacing in mm
    source_spacing: (f64, f64, f64),
    /// Number of workgroups for NCC reduction
    num_workgroups: u32,
}

impl GpuCoregistration {
    /// Create new GPU coregistration resources
    pub fn new(device: &wgpu::Device) -> Self {
        // Load compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Coregistration Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/coregistration.wgsl").into()),
        });

        // Resample bind group layout
        let resample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Resample Bind Group Layout"),
                entries: &[
                    // Transform uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Source volume (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Resampled volume (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // NCC bind group layout
        let ncc_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NCC Bind Group Layout"),
                entries: &[
                    // NCC uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Target volume (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Resampled source (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Partials output (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create resample pipeline
        let resample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Resample Pipeline Layout"),
                bind_group_layouts: &[&resample_bind_group_layout],
                push_constant_ranges: &[],
            });

        let resample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Resample Pipeline"),
            layout: Some(&resample_pipeline_layout),
            module: &shader,
            entry_point: "resample_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create NCC pipeline
        let ncc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NCC Pipeline Layout"),
            bind_group_layouts: &[&ncc_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ncc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NCC Pipeline"),
            layout: Some(&ncc_pipeline_layout),
            module: &shader,
            entry_point: "ncc_reduce",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create uniform buffers
        let transform_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Uniforms"),
            size: std::mem::size_of::<TransformUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ncc_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NCC Uniforms"),
            size: std::mem::size_of::<NccUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            resample_pipeline,
            ncc_pipeline,
            resample_bind_group_layout,
            ncc_bind_group_layout,
            target_buffer: None,
            source_buffer: None,
            resampled_buffer: None,
            transform_uniform_buffer,
            ncc_uniform_buffer,
            partials_buffer: None,
            partials_staging_buffer: None,
            resample_bind_group: None,
            ncc_bind_group: None,
            target_dims: (0, 0, 0),
            source_dims: (0, 0, 0),
            target_spacing: (1.0, 1.0, 1.0),
            source_spacing: (1.0, 1.0, 1.0),
            num_workgroups: 0,
        }
    }

    /// Upload volumes to GPU (call once before optimization loop)
    ///
    /// # Arguments
    /// * `target` - Target volume data (normalized to 0-1)
    /// * `source` - Source volume data (normalized to 0-1)
    /// * `target_dims` - Target volume dimensions (width, height, depth)
    /// * `source_dims` - Source volume dimensions (width, height, depth)
    /// * `target_spacing` - Target voxel spacing in mm
    /// * `source_spacing` - Source voxel spacing in mm
    pub fn upload_volumes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &VolumeUpload,
        source: &VolumeUpload,
    ) {
        let target_voxels = target.dims.0 * target.dims.1 * target.dims.2;
        let source_voxels = source.dims.0 * source.dims.1 * source.dims.2;
        assert_eq!(target.data.len(), target_voxels);
        assert_eq!(source.data.len(), source_voxels);

        self.target_dims = target.dims;
        self.source_dims = source.dims;
        self.target_spacing = target.spacing;
        self.source_spacing = source.spacing;

        // Create target buffer
        self.target_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Target Volume"),
                contents: bytemuck::cast_slice(target.data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );

        // Create source buffer
        self.source_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Source Volume"),
                contents: bytemuck::cast_slice(source.data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );

        // Create resampled buffer (same size as target - we resample source to target grid)
        self.resampled_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resampled Volume"),
            size: (target_voxels * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        self.num_workgroups = target_voxels.div_ceil(NCC_WORKGROUP_SIZE) as u32;

        // Create partials buffer
        let partials_size = self.num_workgroups as usize * std::mem::size_of::<NccPartials>();
        self.partials_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NCC Partials"),
            size: partials_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Create staging buffer for CPU readback
        self.partials_staging_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NCC Partials Staging"),
            size: partials_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Update NCC uniforms (uses target dimensions since we compare in target space)
        let ncc_uniforms = NccUniforms {
            dims: [
                self.target_dims.0 as u32,
                self.target_dims.1 as u32,
                self.target_dims.2 as u32,
                0,
            ],
            num_voxels: target_voxels as u32,
            _padding: [0; 3],
        };
        queue.write_buffer(
            &self.ncc_uniform_buffer,
            0,
            bytemuck::bytes_of(&ncc_uniforms),
        );

        // Build the two compute bind groups once, now that every buffer they
        // reference is in place. compute_ncc runs dozens of times per Powell
        // line-search step, and rebuilding these descriptors per call showed
        // up in profiles on modest CT-MR registrations.
        let target_buffer = self.target_buffer.as_ref().expect("target buffer set above");
        let source_buffer = self.source_buffer.as_ref().expect("source buffer set above");
        let resampled_buffer = self
            .resampled_buffer
            .as_ref()
            .expect("resampled buffer set above");
        let partials_buffer = self
            .partials_buffer
            .as_ref()
            .expect("partials buffer set above");

        self.resample_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resample Bind Group"),
            layout: &self.resample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.transform_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resampled_buffer.as_entire_binding(),
                },
            ],
        }));

        self.ncc_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NCC Bind Group"),
            layout: &self.ncc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ncc_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resampled_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partials_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Compute NCC metric for given transform (synchronous, blocks until result ready)
    pub fn compute_ncc(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
    ) -> f64 {
        // partials + staging are still needed directly for the copy and read;
        // target / source / resampled live inside the cached bind groups now
        // and don't need to be referenced explicitly on the hot path.
        let partials_buffer = self.partials_buffer.as_ref().expect("Volumes not uploaded");
        let staging_buffer = self
            .partials_staging_buffer
            .as_ref()
            .expect("Volumes not uploaded");

        // Update transform uniforms with dimensions and spacing
        let transform_uniforms = TransformUniforms {
            transform: *transform,
            target_dims: [
                self.target_dims.0 as u32,
                self.target_dims.1 as u32,
                self.target_dims.2 as u32,
                0,
            ],
            source_dims: [
                self.source_dims.0 as u32,
                self.source_dims.1 as u32,
                self.source_dims.2 as u32,
                0,
            ],
            target_spacing: [
                self.target_spacing.0 as f32,
                self.target_spacing.1 as f32,
                self.target_spacing.2 as f32,
                0.0,
            ],
            source_spacing: [
                self.source_spacing.0 as f32,
                self.source_spacing.1 as f32,
                self.source_spacing.2 as f32,
                0.0,
            ],
        };
        queue.write_buffer(
            &self.transform_uniform_buffer,
            0,
            bytemuck::bytes_of(&transform_uniforms),
        );

        // Reuse bind groups built once in upload_volumes.
        let resample_bind_group = self
            .resample_bind_group
            .as_ref()
            .expect("Volumes not uploaded");
        let ncc_bind_group = self
            .ncc_bind_group
            .as_ref()
            .expect("Volumes not uploaded");

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Coregistration Encoder"),
        });

        // Dispatch resample compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Resample Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resample_pipeline);
            pass.set_bind_group(0, &resample_bind_group, &[]);

            let dispatch_x = (self.target_dims.0 as u32).div_ceil(RESAMPLE_WORKGROUP_SIZE);
            let dispatch_y = (self.target_dims.1 as u32).div_ceil(RESAMPLE_WORKGROUP_SIZE);
            let dispatch_z = (self.target_dims.2 as u32).div_ceil(RESAMPLE_WORKGROUP_SIZE);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        // Dispatch NCC reduction
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NCC Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ncc_pipeline);
            pass.set_bind_group(0, &ncc_bind_group, &[]);
            pass.dispatch_workgroups(self.num_workgroups, 1, 1);
        }

        // Copy partials to staging buffer
        let partials_size = self.num_workgroups as u64 * std::mem::size_of::<NccPartials>() as u64;
        encoder.copy_buffer_to_buffer(partials_buffer, 0, staging_buffer, 0, partials_size);

        // Submit and wait
        queue.submit(std::iter::once(encoder.finish()));

        // Map staging buffer and read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        if let Err(e) = rx.recv() {
            tracing::error!("GPU buffer map channel error: {}", e);
            return f64::NEG_INFINITY;
        }

        // Read partials and compute final NCC
        let data = buffer_slice.get_mapped_range();
        let partials: &[NccPartials] = bytemuck::cast_slice(&data);

        // Sum up all partial results on CPU
        let mut sum_t = 0.0f64;
        let mut sum_s = 0.0f64;
        let mut sum_t2 = 0.0f64;
        let mut sum_s2 = 0.0f64;
        let mut sum_ts = 0.0f64;
        let mut count = 0u64;

        for p in partials {
            sum_t += p.sum_target as f64;
            sum_s += p.sum_source as f64;
            sum_t2 += p.sum_target_sq as f64;
            sum_s2 += p.sum_source_sq as f64;
            sum_ts += p.sum_product as f64;
            count += p.count as u64;
        }

        drop(data);
        staging_buffer.unmap();

        // Compute NCC
        if count == 0 {
            return 0.0;
        }

        let n = count as f64;
        let mean_t = sum_t / n;
        let mean_s = sum_s / n;

        let var_t = (sum_t2 / n) - (mean_t * mean_t);
        let var_s = (sum_s2 / n) - (mean_s * mean_s);

        let std_t = var_t.sqrt();
        let std_s = var_s.sqrt();

        if std_t < 1e-10 || std_s < 1e-10 {
            return 0.0;
        }

        let covariance = (sum_ts / n) - (mean_t * mean_s);
        let ncc = covariance / (std_t * std_s);

        ncc.clamp(-1.0, 1.0)
    }
}
