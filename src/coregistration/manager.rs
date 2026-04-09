//! Coregistration manager - handles UI state and workflow

use super::optimizer::PowellOptimizer;
use super::pipeline::{
    compute_initial_alignment, RegistrationConfig, RegistrationResult, VolumeGeometry,
};
use super::transform::RigidTransform;
use crate::dicom::Volume;
use crate::gpu::{GpuCoregistration, VolumeUpload};
use pollster::FutureExt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Current coregistration mode
#[derive(Debug, Clone)]
pub enum CoregistrationMode {
    /// Not in coregistration mode
    Inactive,
    /// Waiting for user to select target (fixed reference)
    SelectingTarget,
    /// Target selected, waiting for source selection
    SelectingSource {
        /// Target viewport index
        target_viewport: usize,
    },
    /// Registration in progress
    Running {
        /// Target viewport index
        target_viewport: usize,
        /// Source viewport index
        source_viewport: usize,
    },
    /// Registration completed, waiting for user to accept/reject
    Completed {
        /// Target viewport index
        target_viewport: usize,
        /// Source viewport index
        source_viewport: usize,
        /// Result
        result: Box<RegistrationResult>,
    },
}

impl CoregistrationMode {
    pub fn is_active(&self) -> bool {
        !matches!(self, CoregistrationMode::Inactive)
    }

    pub fn is_selecting(&self) -> bool {
        matches!(
            self,
            CoregistrationMode::SelectingTarget | CoregistrationMode::SelectingSource { .. }
        )
    }

    pub fn is_running(&self) -> bool {
        matches!(self, CoregistrationMode::Running { .. })
    }

    pub fn status_text(&self) -> &'static str {
        match self {
            CoregistrationMode::Inactive => "",
            CoregistrationMode::SelectingTarget => {
                "Coregistration: Click TARGET series (fixed reference)"
            }
            CoregistrationMode::SelectingSource { .. } => {
                "Coregistration: Click SOURCE series (to transform)"
            }
            CoregistrationMode::Running { .. } => "Coregistration: Running...",
            CoregistrationMode::Completed { .. } => "Coregistration: Complete",
        }
    }
}

/// Registration task handle
struct RegistrationTask {
    /// Cancel flag
    cancel: Arc<AtomicBool>,
    /// Progress (0-100)
    progress: Arc<AtomicU32>,
    /// Result when complete
    result: Arc<Mutex<Option<RegistrationResult>>>,
    /// Start time
    started: Instant,
}

/// Coregistration manager
pub struct CoregistrationManager {
    /// Current mode
    mode: CoregistrationMode,
    /// Active registration task
    task: Option<RegistrationTask>,
    /// Last error message (for retry dialog)
    last_error: Option<String>,
}

impl Default for CoregistrationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CoregistrationManager {
    pub fn new() -> Self {
        Self {
            mode: CoregistrationMode::Inactive,
            task: None,
            last_error: None,
        }
    }

    /// Get current mode
    pub fn mode(&self) -> CoregistrationMode {
        self.mode.clone()
    }

    /// Start coregistration mode (called when user presses Ctrl+R)
    pub fn start(&mut self) {
        self.mode = CoregistrationMode::SelectingTarget;
        self.task = None;
        self.last_error = None;
        tracing::info!("Coregistration mode started - select target series");
    }

    /// Cancel coregistration mode
    pub fn cancel(&mut self) {
        if let Some(task) = &self.task {
            task.cancel.store(true, Ordering::Relaxed);
        }
        self.mode = CoregistrationMode::Inactive;
        self.task = None;
        tracing::info!("Coregistration cancelled");
    }

    /// Handle viewport click during selection
    /// Returns true if the click was consumed
    pub fn handle_viewport_click(&mut self, viewport_index: usize) -> bool {
        match self.mode {
            CoregistrationMode::SelectingTarget => {
                self.mode = CoregistrationMode::SelectingSource {
                    target_viewport: viewport_index,
                };
                tracing::info!("Target selected: viewport {}", viewport_index);
                true
            }
            CoregistrationMode::SelectingSource { target_viewport } => {
                if viewport_index == target_viewport {
                    tracing::warn!("Cannot select same viewport as target and source");
                    return true; // Consume click but don't proceed
                }
                // Both selected - ready to start registration
                self.mode = CoregistrationMode::Running {
                    target_viewport,
                    source_viewport: viewport_index,
                };
                tracing::info!(
                    "Source selected: viewport {}. Ready to start registration.",
                    viewport_index
                );
                true
            }
            _ => false, // Not in selection mode
        }
    }

    /// Start the registration process
    /// Returns (target_viewport, source_viewport) if ready to start
    pub fn get_registration_viewports(&self) -> Option<(usize, usize)> {
        match self.mode {
            CoregistrationMode::Running {
                target_viewport,
                source_viewport,
            } => Some((target_viewport, source_viewport)),
            _ => None,
        }
    }

    /// Launch GPU-accelerated registration in background thread
    pub fn start_registration(
        &mut self,
        target_volume: Arc<Volume>,
        source_volume: Arc<Volume>,
        _config: RegistrationConfig,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(AtomicU32::new(0));
        let result: Arc<Mutex<Option<RegistrationResult>>> = Arc::new(Mutex::new(None));

        let task_cancel = Arc::clone(&cancel);
        let task_progress = Arc::clone(&progress);
        let task_result = Arc::clone(&result);

        // Spawn background thread for GPU registration
        thread::spawn(move || {
            let start = Instant::now();
            task_progress.store(5, Ordering::Relaxed);

            // Check for cancellation
            if task_cancel.load(Ordering::Relaxed) {
                let mut result_lock = task_result.lock().unwrap();
                *result_lock = Some(RegistrationResult::Cancelled);
                return;
            }

            // Initialize wgpu in this thread
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .block_on()
            {
                Some(a) => a,
                None => {
                    let mut result_lock = task_result.lock().unwrap();
                    *result_lock = Some(RegistrationResult::Failed {
                        error: "Failed to find GPU adapter".to_string(),
                        partial_transform: None,
                    });
                    return;
                }
            };

            let (device, queue) = match adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Coregistration Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .block_on()
            {
                Ok(dq) => dq,
                Err(e) => {
                    let mut result_lock = task_result.lock().unwrap();
                    *result_lock = Some(RegistrationResult::Failed {
                        error: format!("Failed to create GPU device: {}", e),
                        partial_transform: None,
                    });
                    return;
                }
            };

            task_progress.store(10, Ordering::Relaxed);

            // Preprocess volumes (normalize to 0-1)
            let target_preprocessed = preprocess_volume_mri(&target_volume.data);
            let source_preprocessed = preprocess_volume_mri(&source_volume.data);

            task_progress.store(30, Ordering::Relaxed);

            if task_cancel.load(Ordering::Relaxed) {
                let mut result_lock = task_result.lock().unwrap();
                *result_lock = Some(RegistrationResult::Cancelled);
                return;
            }

            // Three-level pyramid: 32³ → 64³ → 128³ for balanced quality/speed
            // Get volume geometries for computing spacing
            let target_geom = VolumeGeometry::from_volume(&target_volume);
            let source_geom = VolumeGeometry::from_volume(&source_volume);

            // Create GPU coregistration
            let mut gpu_coreg = GpuCoregistration::new(&device);

            // ========== Level 1: Very coarse (32³) - find rough alignment ==========
            let level1_size = 32;
            let level1_dims = (level1_size, level1_size, level1_size);
            let target_level1 =
                downsample_volume(&target_preprocessed, target_volume.dimensions, level1_dims);
            let source_level1 =
                downsample_volume(&source_preprocessed, source_volume.dimensions, level1_dims);
            let target_level1_geom = target_geom.downsampled(level1_dims);
            let source_level1_geom = source_geom.downsampled(level1_dims);

            gpu_coreg.upload_volumes(
                &device,
                &queue,
                &VolumeUpload { data: &target_level1, dims: level1_dims, spacing: target_level1_geom.spacing },
                &VolumeUpload { data: &source_level1, dims: level1_dims, spacing: source_level1_geom.spacing },
            );

            // Compute initial alignment using center-of-mass
            let mut transform = compute_initial_alignment(
                &target_level1,
                &source_level1,
                &target_level1_geom,
                &source_level1_geom,
                0.1,
            );

            // Level 1 optimization: large steps, many iterations
            let level1_optimizer = PowellOptimizer::new(false).with_steps(0.15, 12.0);
            let level1_result = {
                let metric_fn = |t: &RigidTransform| {
                    let inv_matrix = t.to_inverse_matrix();
                    let matrix_f32 = rigid_to_f32_matrix(&inv_matrix);
                    gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
                };
                level1_optimizer.optimize(transform, 30, metric_fn)
            };
            transform = level1_result.transform;
            tracing::info!(
                "Level 1 (32³): metric={:.4}, iter={}",
                level1_result.metric,
                level1_result.iterations
            );

            task_progress.store(40, Ordering::Relaxed);

            if task_cancel.load(Ordering::Relaxed) {
                let mut result_lock = task_result.lock().unwrap();
                *result_lock = Some(RegistrationResult::Cancelled);
                return;
            }

            // ========== Level 2: Medium (64³) - refine alignment ==========
            let level2_size = 64;
            let level2_dims = (level2_size, level2_size, level2_size);
            let target_level2 =
                downsample_volume(&target_preprocessed, target_volume.dimensions, level2_dims);
            let source_level2 =
                downsample_volume(&source_preprocessed, source_volume.dimensions, level2_dims);
            let target_level2_geom = target_geom.downsampled(level2_dims);
            let source_level2_geom = source_geom.downsampled(level2_dims);

            gpu_coreg.upload_volumes(
                &device,
                &queue,
                &VolumeUpload { data: &target_level2, dims: level2_dims, spacing: target_level2_geom.spacing },
                &VolumeUpload { data: &source_level2, dims: level2_dims, spacing: source_level2_geom.spacing },
            );

            let level2_optimizer = PowellOptimizer::new(false).with_steps(0.08, 6.0);
            let level2_result = {
                let metric_fn = |t: &RigidTransform| {
                    let inv_matrix = t.to_inverse_matrix();
                    let matrix_f32 = rigid_to_f32_matrix(&inv_matrix);
                    gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
                };
                level2_optimizer.optimize(transform, 25, metric_fn)
            };
            transform = level2_result.transform;
            tracing::info!(
                "Level 2 (64³): metric={:.4}, iter={}",
                level2_result.metric,
                level2_result.iterations
            );

            task_progress.store(65, Ordering::Relaxed);

            if task_cancel.load(Ordering::Relaxed) {
                let mut result_lock = task_result.lock().unwrap();
                *result_lock = Some(RegistrationResult::Cancelled);
                return;
            }

            // ========== Level 3: Fine (128³) - precise alignment ==========
            let level3_size = 128;
            let level3_dims = (level3_size, level3_size, level3_size);
            let target_level3 =
                downsample_volume(&target_preprocessed, target_volume.dimensions, level3_dims);
            let source_level3 =
                downsample_volume(&source_preprocessed, source_volume.dimensions, level3_dims);
            let target_level3_geom = target_geom.downsampled(level3_dims);
            let source_level3_geom = source_geom.downsampled(level3_dims);

            gpu_coreg.upload_volumes(
                &device,
                &queue,
                &VolumeUpload { data: &target_level3, dims: level3_dims, spacing: target_level3_geom.spacing },
                &VolumeUpload { data: &source_level3, dims: level3_dims, spacing: source_level3_geom.spacing },
            );

            let level3_optimizer = PowellOptimizer::new(false).with_steps(0.03, 2.0);
            let level3_result = {
                let metric_fn = |t: &RigidTransform| {
                    let inv_matrix = t.to_inverse_matrix();
                    let matrix_f32 = rigid_to_f32_matrix(&inv_matrix);
                    gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
                };
                level3_optimizer.optimize(transform, 20, metric_fn)
            };
            transform = level3_result.transform;
            tracing::info!(
                "Level 3 (128³): metric={:.4}, iter={}",
                level3_result.metric,
                level3_result.iterations
            );

            task_progress.store(100, Ordering::Relaxed);

            let elapsed = start.elapsed();
            tracing::info!(
                "GPU coregistration completed: metric={:.4}, time={:?}",
                level3_result.metric,
                elapsed
            );

            // Store result
            let mut result_lock = task_result.lock().unwrap();
            *result_lock = Some(RegistrationResult::Success {
                transform,
                metric: level3_result.metric,
                elapsed,
                level_results: vec![level1_result, level2_result, level3_result],
            });
        });

        self.task = Some(RegistrationTask {
            cancel,
            progress,
            result,
            started: Instant::now(),
        });

        tracing::info!("GPU registration task started");
    }

    /// Check if registration is complete and get result
    pub fn poll_result(&mut self) -> Option<RegistrationResult> {
        if let Some(task) = &self.task {
            if let Ok(mut result_lock) = task.result.try_lock() {
                if let Some(result) = result_lock.take() {
                    tracing::info!("Registration completed in {:?}", task.started.elapsed());
                    return Some(result);
                }
            }
        }
        None
    }

    /// Get current progress (0-100)
    pub fn get_progress(&self) -> u32 {
        self.task
            .as_ref()
            .map(|t| t.progress.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Reset to inactive state
    pub fn reset(&mut self) {
        self.mode = CoregistrationMode::Inactive;
        self.task = None;
        self.last_error = None;
    }

    /// Set error message for retry dialog
    pub fn set_error(&mut self, error: String) {
        self.last_error = Some(error);
    }
}

// Helper functions for GPU coregistration

/// Preprocess MRI volume data - normalize to 0-1 using percentiles
fn preprocess_volume_mri(data: &[u16]) -> Vec<f32> {
    let mut sorted: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p2 = sorted[sorted.len() * 2 / 100];
    let p98 = sorted[sorted.len() * 98 / 100];
    let range = p98 - p2;

    if range > 1e-6 {
        data.iter()
            .map(|&v| ((v as f64 - p2) / range).clamp(0.0, 1.0) as f32)
            .collect()
    } else {
        data.iter().map(|&v| v as f32 / 65535.0).collect()
    }
}

/// Downsample volume to target dimensions
fn downsample_volume(
    data: &[f32],
    src_dims: (usize, usize, usize),
    dst_dims: (usize, usize, usize),
) -> Vec<f32> {
    let (sw, sh, sd) = src_dims;
    let (dw, dh, dd) = dst_dims;

    let scale_x = sw as f64 / dw as f64;
    let scale_y = sh as f64 / dh as f64;
    let scale_z = sd as f64 / dd as f64;

    let mut result = vec![0.0f32; dw * dh * dd];

    for dz in 0..dd {
        for dy in 0..dh {
            for dx in 0..dw {
                // Sample center of destination voxel in source coordinates
                let sx = ((dx as f64 + 0.5) * scale_x).min(sw as f64 - 1.0);
                let sy = ((dy as f64 + 0.5) * scale_y).min(sh as f64 - 1.0);
                let sz = ((dz as f64 + 0.5) * scale_z).min(sd as f64 - 1.0);

                // Nearest neighbor for simplicity
                let src_idx = sx as usize + sy as usize * sw + sz as usize * sw * sh;
                let dst_idx = dx + dy * dw + dz * dw * dh;

                if src_idx < data.len() {
                    result[dst_idx] = data[src_idx];
                }
            }
        }
    }

    result
}

/// Convert Mat4 to f32 array for GPU
fn rigid_to_f32_matrix(m: &super::transform::Mat4) -> [[f32; 4]; 4] {
    [
        [
            m.m[0][0] as f32,
            m.m[0][1] as f32,
            m.m[0][2] as f32,
            m.m[0][3] as f32,
        ],
        [
            m.m[1][0] as f32,
            m.m[1][1] as f32,
            m.m[1][2] as f32,
            m.m[1][3] as f32,
        ],
        [
            m.m[2][0] as f32,
            m.m[2][1] as f32,
            m.m[2][2] as f32,
            m.m[2][3] as f32,
        ],
        [
            m.m[3][0] as f32,
            m.m[3][1] as f32,
            m.m[3][2] as f32,
            m.m[3][3] as f32,
        ],
    ]
}
