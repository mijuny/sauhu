//! Registration pipeline - orchestrates the full coregistration process
//!
//! Handles preprocessing, multi-resolution optimization, and result generation.

use super::metrics::MetricType;
use super::optimizer::{OptimizationResult, PowellOptimizer, PyramidSchedule};
use super::transform::RigidTransform;
use crate::dicom::Volume;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// HU windowing ranges for CT coregistration preprocessing
#[allow(dead_code)] // coregistration preprocessing constants
const CT_BRAIN_HU_MIN: f64 = -500.0; // Bone window: center=500, width=2000
#[allow(dead_code)]
const CT_BRAIN_HU_MAX: f64 = 1500.0;
#[allow(dead_code)]
const CT_BODY_HU_MIN: f64 = -160.0; // Soft tissue window: center=40, width=400
#[allow(dead_code)]
const CT_BODY_HU_MAX: f64 = 240.0;

/// Registration configuration
#[derive(Debug, Clone)]
#[allow(dead_code)] // coregistration module API
pub struct RegistrationConfig {
    /// Modality of target image
    pub target_modality: Modality,
    /// Modality of source image
    pub source_modality: Modality,
    /// Anatomy type (affects preprocessing)
    pub anatomy: Anatomy,
    /// Whether to optimize scale (7 DOF vs 6 DOF)
    pub optimize_scale: bool,
    /// Similarity metric to use
    pub metric: MetricType,
    /// Multi-resolution schedule
    pub schedule: PyramidSchedule,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            target_modality: Modality::Mri,
            source_modality: Modality::Mri,
            anatomy: Anatomy::Brain,
            optimize_scale: false,
            metric: MetricType::Ncc,
            schedule: PyramidSchedule::fast(),
        }
    }
}

impl RegistrationConfig {
    /// Create config for brain CT-CT registration
    #[allow(dead_code)] // coregistration module API
    pub fn brain_ct() -> Self {
        Self {
            target_modality: Modality::CT,
            source_modality: Modality::CT,
            anatomy: Anatomy::Brain,
            optimize_scale: false,
            metric: MetricType::Ncc,
            schedule: PyramidSchedule::fast(),
        }
    }

    /// Create config for brain MRI-MRI registration
    #[allow(dead_code)] // coregistration module API
    pub fn brain_mri() -> Self {
        Self {
            target_modality: Modality::Mri,
            source_modality: Modality::Mri,
            anatomy: Anatomy::Brain,
            optimize_scale: false,
            metric: MetricType::Ncc,
            schedule: PyramidSchedule::fast(),
        }
    }

    /// Create config for body CT-CT registration
    #[allow(dead_code)] // coregistration module API
    pub fn body_ct() -> Self {
        Self {
            target_modality: Modality::CT,
            source_modality: Modality::CT,
            anatomy: Anatomy::Body,
            optimize_scale: false,
            metric: MetricType::Ncc,
            schedule: PyramidSchedule::fast(),
        }
    }

    /// Create config for cross-modality (CT-MRI)
    #[allow(dead_code)] // coregistration module API
    pub fn cross_modality() -> Self {
        Self {
            target_modality: Modality::CT,
            source_modality: Modality::Mri,
            anatomy: Anatomy::Brain,
            optimize_scale: false,
            metric: MetricType::Nmi, // Mutual information for cross-modality
            schedule: PyramidSchedule::fast(),
        }
    }

    /// Detect appropriate config from modality strings
    pub fn detect(target_modality: Option<&str>, source_modality: Option<&str>) -> Self {
        let t_mod = Modality::from_str(target_modality);
        let s_mod = Modality::from_str(source_modality);

        let metric = if t_mod != s_mod {
            MetricType::Nmi // Cross-modality
        } else {
            MetricType::Ncc // Same modality
        };

        Self {
            target_modality: t_mod,
            source_modality: s_mod,
            anatomy: Anatomy::Brain, // Default to brain
            optimize_scale: false,
            metric,
            schedule: PyramidSchedule::fast(),
        }
    }
}

/// Image modality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    CT,
    Mri,
    Other,
}

impl Modality {
    pub fn from_str(s: Option<&str>) -> Self {
        match s.map(|s| s.to_uppercase()).as_deref() {
            Some("CT") => Self::CT,
            Some("MR") | Some("MRI") => Self::Mri,
            _ => Self::Other,
        }
    }
}

/// Anatomy type (affects preprocessing)
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // coregistration module API
pub enum Anatomy {
    Brain,
    Body,
}

/// Volume geometry for coordinate conversions
///
/// Enables conversion between voxel indices and physical (mm) coordinates.
/// Critical for multi-resolution registration where different pyramid levels
/// have different voxel dimensions but the same physical extent.
#[derive(Debug, Clone)]
pub struct VolumeGeometry {
    /// Volume dimensions (width, height, depth) in voxels
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing (col_spacing, row_spacing, slice_spacing) in mm
    pub spacing: (f64, f64, f64),
    /// Physical extent (width, height, depth) in mm
    pub extent: (f64, f64, f64),
}

impl VolumeGeometry {
    /// Create geometry from volume dimensions and spacing
    pub fn new(dimensions: (usize, usize, usize), spacing: (f64, f64, f64)) -> Self {
        let extent = (
            dimensions.0 as f64 * spacing.0,
            dimensions.1 as f64 * spacing.1,
            dimensions.2 as f64 * spacing.2,
        );
        Self {
            dimensions,
            spacing,
            extent,
        }
    }

    /// Create geometry from a Volume
    pub fn from_volume(volume: &Volume) -> Self {
        Self::new(volume.dimensions, volume.spacing)
    }

    /// Create geometry for a downsampled version of this volume
    pub fn downsampled(&self, new_dims: (usize, usize, usize)) -> Self {
        // Maintain the same physical extent
        let new_spacing = (
            self.extent.0 / new_dims.0 as f64,
            self.extent.1 / new_dims.1 as f64,
            self.extent.2 / new_dims.2 as f64,
        );
        Self {
            dimensions: new_dims,
            spacing: new_spacing,
            extent: self.extent,
        }
    }

    /// Convert voxel coordinates to physical (mm) coordinates
    ///
    /// Voxel (0,0,0) maps to physical (0,0,0)
    /// Voxel center is at (i+0.5)*spacing
    #[inline]
    #[allow(dead_code)] // coregistration module API
    pub fn voxel_to_physical(&self, voxel: (f64, f64, f64)) -> (f64, f64, f64) {
        (
            voxel.0 * self.spacing.0,
            voxel.1 * self.spacing.1,
            voxel.2 * self.spacing.2,
        )
    }

    /// Convert physical (mm) coordinates to voxel coordinates
    #[inline]
    #[allow(dead_code)] // coregistration module API
    pub fn physical_to_voxel(&self, physical: (f64, f64, f64)) -> (f64, f64, f64) {
        (
            physical.0 / self.spacing.0,
            physical.1 / self.spacing.1,
            physical.2 / self.spacing.2,
        )
    }

    /// Get the physical center of the volume
    pub fn center(&self) -> (f64, f64, f64) {
        (
            self.extent.0 / 2.0,
            self.extent.1 / 2.0,
            self.extent.2 / 2.0,
        )
    }
}

/// Compute intensity-weighted center of mass in physical coordinates
///
/// Uses a threshold to ignore background voxels. Returns the centroid
/// in mm coordinates relative to the volume origin.
///
/// # Arguments
/// * `data` - Normalized volume data (0-1 range)
/// * `geometry` - Volume geometry for coordinate conversion
/// * `threshold` - Minimum intensity to include (0.0-1.0, typically 0.1)
pub fn compute_center_of_mass(
    data: &[f32],
    geometry: &VolumeGeometry,
    threshold: f32,
) -> (f64, f64, f64) {
    let (w, h, d) = geometry.dimensions;
    let (sx, sy, sz) = geometry.spacing;

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut total_weight = 0.0;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = x + y * w + z * w * h;
                let weight = data[idx];

                if weight > threshold {
                    // Use voxel center position
                    let px = (x as f64 + 0.5) * sx;
                    let py = (y as f64 + 0.5) * sy;
                    let pz = (z as f64 + 0.5) * sz;

                    sum_x += px * weight as f64;
                    sum_y += py * weight as f64;
                    sum_z += pz * weight as f64;
                    total_weight += weight as f64;
                }
            }
        }
    }

    if total_weight > 0.0 {
        (
            sum_x / total_weight,
            sum_y / total_weight,
            sum_z / total_weight,
        )
    } else {
        // Fallback to geometric center if no voxels above threshold
        geometry.center()
    }
}

/// Compute initial alignment transform based on center-of-mass and PCA
///
/// Calculates both the translation (from COM) and rotation (from PCA)
/// needed to align the source volume to the target volume.
///
/// # Arguments
/// * `target_data` - Target volume data (normalized)
/// * `source_data` - Source volume data (normalized)
/// * `target_geom` - Target volume geometry
/// * `source_geom` - Source volume geometry
/// * `threshold` - Intensity threshold for calculations
///
/// # Returns
/// RigidTransform with both translation and rotation for initial alignment
pub fn compute_initial_alignment(
    target_data: &[f32],
    source_data: &[f32],
    target_geom: &VolumeGeometry,
    source_geom: &VolumeGeometry,
    threshold: f32,
) -> super::transform::RigidTransform {
    // Step 1: Compute centers of mass
    let target_com = compute_center_of_mass(target_data, target_geom, threshold);
    let source_com = compute_center_of_mass(source_data, source_geom, threshold);

    // Step 2: Compute principal axes using PCA
    let target_axes = compute_principal_axes(target_data, target_geom, target_com, threshold);
    let source_axes = compute_principal_axes(source_data, source_geom, source_com, threshold);

    // Step 3: Compute rotation to align source axes to target axes
    // NOTE: PCA rotation is currently disabled due to sign ambiguity issues
    // that can cause worse initial alignment. The optimizer will find the rotation.
    let (rx, ry, rz) = compute_rotation_from_axes(&source_axes, &target_axes);

    // Log the computed rotation for debugging, but use zero rotation
    println!(
        "PCA computed rotation: ({:.2}°, {:.2}°, {:.2}°) - DISABLED, using (0, 0, 0)",
        rx.to_degrees(),
        ry.to_degrees(),
        rz.to_degrees()
    );

    // Use zero rotation - let the optimizer find it
    let (rx, ry, rz) = (0.0, 0.0, 0.0);

    // Step 4: Compute translation (after rotation is applied, source COM moves)
    // For small rotations, we can approximate: translate source COM to target COM
    let tx = target_com.0 - source_com.0;
    let ty = target_com.1 - source_com.1;
    let tz = target_com.2 - source_com.2;

    println!(
        "COM alignment: target=({:.1}, {:.1}, {:.1})mm, source=({:.1}, {:.1}, {:.1})mm",
        target_com.0, target_com.1, target_com.2, source_com.0, source_com.1, source_com.2
    );
    println!("Initial translation: ({:.1}, {:.1}, {:.1})mm", tx, ty, tz);

    super::transform::RigidTransform {
        translation_x: tx,
        translation_y: ty,
        translation_z: tz,
        rotation_x: rx,
        rotation_y: ry,
        rotation_z: rz,
        scale: 1.0,
    }
}

/// Principal axes (eigenvectors) of the intensity distribution
/// Stored as column vectors: axes[i] is the i-th principal axis
type PrincipalAxes = [[f64; 3]; 3];

/// Compute principal axes of the intensity distribution using PCA
///
/// Returns 3 orthogonal unit vectors representing the principal directions
/// of the intensity distribution, sorted by decreasing eigenvalue.
fn compute_principal_axes(
    data: &[f32],
    geometry: &VolumeGeometry,
    com: (f64, f64, f64),
    threshold: f32,
) -> PrincipalAxes {
    let (w, h, d) = geometry.dimensions;
    let (sx, sy, sz) = geometry.spacing;

    // Compute covariance matrix elements
    let mut cxx = 0.0;
    let mut cxy = 0.0;
    let mut cxz = 0.0;
    let mut cyy = 0.0;
    let mut cyz = 0.0;
    let mut czz = 0.0;
    let mut total_weight = 0.0;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = x + y * w + z * w * h;
                let weight = data[idx];

                if weight > threshold {
                    // Position relative to COM
                    let px = (x as f64 + 0.5) * sx - com.0;
                    let py = (y as f64 + 0.5) * sy - com.1;
                    let pz = (z as f64 + 0.5) * sz - com.2;

                    let w = weight as f64;
                    cxx += w * px * px;
                    cxy += w * px * py;
                    cxz += w * px * pz;
                    cyy += w * py * py;
                    cyz += w * py * pz;
                    czz += w * pz * pz;
                    total_weight += w;
                }
            }
        }
    }

    if total_weight > 0.0 {
        cxx /= total_weight;
        cxy /= total_weight;
        cxz /= total_weight;
        cyy /= total_weight;
        cyz /= total_weight;
        czz /= total_weight;
    }

    // Covariance matrix
    let cov = [[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]];

    // Compute eigenvectors using power iteration
    eigendecomposition_3x3(&cov)
}

/// Compute eigenvectors of a 3x3 symmetric matrix using power iteration
///
/// Returns eigenvectors as columns, sorted by decreasing eigenvalue
fn eigendecomposition_3x3(matrix: &[[f64; 3]; 3]) -> PrincipalAxes {
    let mut result = [[0.0; 3]; 3];
    let mut m = *matrix;

    let initial_vectors: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (i, &init_v) in initial_vectors.iter().enumerate() {
        // Power iteration to find dominant eigenvector
        let mut v = init_v;

        for _ in 0..50 {
            // Multiply: v = M * v
            let new_v = [
                m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
            ];

            // Normalize
            let norm = (new_v[0] * new_v[0] + new_v[1] * new_v[1] + new_v[2] * new_v[2]).sqrt();
            if norm > 1e-10 {
                v = [new_v[0] / norm, new_v[1] / norm, new_v[2] / norm];
            } else {
                break;
            }
        }

        result[i] = v;

        // Deflate matrix: M = M - lambda * v * v^T
        // where lambda = v^T * M * v
        let mv = [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ];
        let lambda = v[0] * mv[0] + v[1] * mv[1] + v[2] * mv[2];

        for j in 0..3 {
            for k in 0..3 {
                m[j][k] -= lambda * v[j] * v[k];
            }
        }
    }

    // Ensure right-handed coordinate system
    // If det < 0, flip the third axis
    let det = result[0][0] * (result[1][1] * result[2][2] - result[1][2] * result[2][1])
        - result[0][1] * (result[1][0] * result[2][2] - result[1][2] * result[2][0])
        + result[0][2] * (result[1][0] * result[2][1] - result[1][1] * result[2][0]);

    if det < 0.0 {
        result[2] = [-result[2][0], -result[2][1], -result[2][2]];
    }

    result
}

/// Compute Euler angles (XYZ convention) to rotate source axes to target axes
///
/// Returns (rx, ry, rz) in radians
fn compute_rotation_from_axes(source: &PrincipalAxes, target: &PrincipalAxes) -> (f64, f64, f64) {
    // Build rotation matrices from axes (columns are the axes)
    // R_target * R_source^T gives rotation from source to target

    // Source axes as columns of a matrix
    let s = source;
    // Target axes as columns of a matrix
    let t = target;

    // R = T * S^T (rotation matrix that transforms source axes to target axes)
    // Since S is orthonormal, S^T = S^(-1)
    let r = [
        [
            t[0][0] * s[0][0] + t[1][0] * s[1][0] + t[2][0] * s[2][0],
            t[0][0] * s[0][1] + t[1][0] * s[1][1] + t[2][0] * s[2][1],
            t[0][0] * s[0][2] + t[1][0] * s[1][2] + t[2][0] * s[2][2],
        ],
        [
            t[0][1] * s[0][0] + t[1][1] * s[1][0] + t[2][1] * s[2][0],
            t[0][1] * s[0][1] + t[1][1] * s[1][1] + t[2][1] * s[2][1],
            t[0][1] * s[0][2] + t[1][1] * s[1][2] + t[2][1] * s[2][2],
        ],
        [
            t[0][2] * s[0][0] + t[1][2] * s[1][0] + t[2][2] * s[2][0],
            t[0][2] * s[0][1] + t[1][2] * s[1][1] + t[2][2] * s[2][1],
            t[0][2] * s[0][2] + t[1][2] * s[1][2] + t[2][2] * s[2][2],
        ],
    ];

    // Extract Euler angles (XYZ convention) from rotation matrix
    // This matches the convention used in RigidTransform
    rotation_matrix_to_euler_xyz(&r)
}

/// Extract Euler angles (XYZ convention) from a 3x3 rotation matrix
///
/// Returns (rx, ry, rz) in radians
fn rotation_matrix_to_euler_xyz(r: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    // For XYZ Euler angles:
    // R = Rz * Ry * Rx
    //
    // r[0][2] = sin(ry)
    // r[1][2] = -cos(ry) * sin(rx)
    // r[2][2] = cos(ry) * cos(rx)
    // r[0][0] = cos(ry) * cos(rz)
    // r[0][1] = -cos(ry) * sin(rz)

    let ry = r[0][2].clamp(-1.0, 1.0).asin();
    let cy = ry.cos();

    let (rx, rz);
    if cy.abs() > 1e-6 {
        rx = (-r[1][2] / cy).atan2(r[2][2] / cy);
        rz = (-r[0][1] / cy).atan2(r[0][0] / cy);
    } else {
        // Gimbal lock - ry is ±90°
        rx = 0.0;
        rz = r[1][0].atan2(r[1][1]);
    }

    // Clamp to reasonable range for medical imaging (±45°)
    // Large rotations likely indicate axis ambiguity
    let max_rot = std::f64::consts::FRAC_PI_4; // 45°
    let rx = rx.clamp(-max_rot, max_rot);
    let ry = ry.clamp(-max_rot, max_rot);
    let rz = rz.clamp(-max_rot, max_rot);

    (rx, ry, rz)
}

/// Registration progress info
#[derive(Debug, Clone)]
#[allow(dead_code)] // coregistration module API
pub struct RegistrationProgress {
    /// Current level (0 = finest)
    pub level: usize,
    /// Total levels
    pub total_levels: usize,
    /// Current iteration within level
    pub iteration: usize,
    /// Total iterations for current level
    pub total_iterations: usize,
    /// Current metric value
    pub metric: f64,
}

/// Registration result
#[derive(Debug, Clone)]
#[allow(dead_code)] // coregistration module API
pub enum RegistrationResult {
    Success {
        /// Final rigid transform
        transform: RigidTransform,
        /// Final similarity metric value
        metric: f64,
        /// Total time elapsed
        elapsed: Duration,
        /// Per-level results
        level_results: Vec<OptimizationResult>,
    },
    Failed {
        /// Error message
        error: String,
        /// Partial transform (if any progress was made)
        partial_transform: Option<RigidTransform>,
    },
    Cancelled,
}

/// Registration pipeline
#[allow(dead_code)] // coregistration module API
pub struct RegistrationPipeline {
    /// Configuration
    config: RegistrationConfig,
    /// Cancel flag (shared with UI)
    cancel_flag: Arc<AtomicBool>,
    /// Progress (0-100, shared with UI)
    progress: Arc<AtomicU32>,
}

#[allow(dead_code)] // coregistration module API
impl RegistrationPipeline {
    pub fn new(config: RegistrationConfig) -> Self {
        Self {
            config,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Create pipeline with external cancel/progress shared state
    pub fn with_shared_state(
        config: RegistrationConfig,
        cancel_flag: Arc<AtomicBool>,
        progress: Arc<AtomicU32>,
    ) -> Self {
        Self {
            config,
            cancel_flag,
            progress,
        }
    }

    /// Get cancel flag for external cancellation
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }

    /// Get progress counter (0-100)
    pub fn progress(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.progress)
    }

    /// Run registration on CPU with parallel processing
    pub fn run_cpu(&self, target: &Volume, source: &Volume) -> RegistrationResult {
        let start = Instant::now();

        // Check for cancellation
        if self.cancel_flag.load(Ordering::Relaxed) {
            return RegistrationResult::Cancelled;
        }

        // Preprocess volumes
        self.progress.store(5, Ordering::Relaxed);
        tracing::info!("Preprocessing volumes...");
        let target_data = self.preprocess_volume(target, self.config.target_modality);
        let source_data = self.preprocess_volume(source, self.config.source_modality);

        // Build resolution pyramids (only 3 levels for speed)
        self.progress.store(10, Ordering::Relaxed);
        tracing::info!("Building pyramids...");
        let num_levels = self.config.schedule.num_levels();
        let target_pyramid = build_pyramid(&target_data, target.dimensions, num_levels);
        let source_pyramid = build_pyramid(&source_data, source.dimensions, num_levels);

        // Debug: check metric at coarsest level with identity transform
        let coarsest_level = num_levels - 1;
        let coarsest_dims = pyramid_dimensions(target.dimensions, coarsest_level);
        tracing::info!(
            "Coarsest level {} dimensions: {:?} ({} voxels)",
            coarsest_level,
            coarsest_dims,
            coarsest_dims.0 * coarsest_dims.1 * coarsest_dims.2
        );

        let identity_metric = self.config.metric.compute(
            &target_pyramid[coarsest_level],
            &source_pyramid[coarsest_level],
        );
        tracing::info!(
            "Initial metric at coarsest level (no transform): {:.4}",
            identity_metric
        );

        // Multi-resolution optimization (coarse to fine)
        let mut transform = RigidTransform::identity();
        let mut level_results = Vec::new();

        for level in (0..num_levels).rev() {
            if self.cancel_flag.load(Ordering::Relaxed) {
                return RegistrationResult::Cancelled;
            }

            let (iterations, rot_step, trans_step) = self.config.schedule.for_level(level);
            let level_dims = pyramid_dimensions(target.dimensions, level);

            tracing::info!(
                "Level {}: dims={:?}, iterations={}, rot_step={:.3}, trans_step={:.1}",
                level,
                level_dims,
                iterations,
                rot_step,
                trans_step
            );

            let optimizer =
                PowellOptimizer::new(self.config.optimize_scale).with_steps(rot_step, trans_step);

            let target_level = &target_pyramid[level];
            let source_level = &source_pyramid[level];

            // Create metric function for this level (with parallel resampling)
            let metric_fn = |t: &RigidTransform| {
                let resampled =
                    resample_volume_parallel(source_level, level_dims, &t.to_inverse_matrix());
                self.config.metric.compute(target_level, &resampled)
            };

            let result = optimizer.optimize(transform, iterations, metric_fn);
            transform = result.transform;
            level_results.push(result.clone());

            // Update progress (10-90% for optimization)
            let level_progress = 10 + (80 * (num_levels - level) / num_levels) as u32;
            self.progress.store(level_progress, Ordering::Relaxed);

            tracing::info!(
                "Level {} complete: metric={:.4}, iterations={}",
                level,
                result.metric,
                result.iterations
            );
        }

        self.progress.store(100, Ordering::Relaxed);

        let final_metric = level_results.last().map(|r| r.metric).unwrap_or(0.0);

        RegistrationResult::Success {
            transform,
            metric: final_metric,
            elapsed: start.elapsed(),
            level_results,
        }
    }

    /// Preprocess volume data for registration
    fn preprocess_volume(&self, volume: &Volume, modality: Modality) -> Vec<f32> {
        let data = &volume.data;
        let mut result = Vec::with_capacity(data.len());

        match (modality, self.config.anatomy) {
            (Modality::CT, Anatomy::Brain) => {
                for &v in data {
                    let hu = v as f64 * volume.rescale_slope + volume.rescale_intercept;
                    let normalized =
                        ((hu - CT_BRAIN_HU_MIN) / (CT_BRAIN_HU_MAX - CT_BRAIN_HU_MIN)).clamp(0.0, 1.0);
                    result.push(normalized as f32);
                }
            }
            (Modality::CT, Anatomy::Body) => {
                for &v in data {
                    let hu = v as f64 * volume.rescale_slope + volume.rescale_intercept;
                    let normalized =
                        ((hu - CT_BODY_HU_MIN) / (CT_BODY_HU_MAX - CT_BODY_HU_MIN)).clamp(0.0, 1.0);
                    result.push(normalized as f32);
                }
            }
            (Modality::Mri, _) | (Modality::Other, _) => {
                // Normalize to 0-1 range based on percentiles (robust to outliers)
                let mut sorted: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let p2 = sorted[sorted.len() * 2 / 100];
                let p98 = sorted[sorted.len() * 98 / 100];
                let range = p98 - p2;

                if range > 1e-6 {
                    for &v in data {
                        let normalized = ((v as f64 - p2) / range).clamp(0.0, 1.0);
                        result.push(normalized as f32);
                    }
                } else {
                    result = data.iter().map(|&v| v as f32 / 65535.0).collect();
                }
            }
        }

        result
    }
}

/// Apply transform to volume and generate new MPR series
///
/// # Arguments
/// * `volume` - Source volume to transform
/// * `transform` - Rigid transform to apply
/// * `target_volume` - Optional target volume whose geometry will be adopted.
///   This is critical for sync to work correctly - the transformed volume's
///   origin, directions, and frame_of_reference must match the target.
pub fn apply_transform_to_volume(
    volume: &Volume,
    transform: &RigidTransform,
    target_volume: Option<&Volume>,
) -> Volume {
    use super::transform::Vec3 as TransformVec3;

    let (src_width, src_height, src_depth) = volume.dimensions;
    let (src_col_sp, src_row_sp, src_slice_sp) = volume.spacing;

    // When target is available, resample onto the target's grid so that
    // the output volume has identical geometry (dimensions, spacing, origin,
    // directions). This makes ImagePlane::intersect() produce correct
    // reference lines without any special-case hacks.
    let (out_width, out_height, out_depth, out_col_sp, out_row_sp, out_slice_sp) =
        if let Some(target) = target_volume {
            let (tw, th, td) = target.dimensions;
            let (tc, tr, ts) = target.spacing;
            (tw, th, td, tc, tr, ts)
        } else {
            (src_width, src_height, src_depth, src_col_sp, src_row_sp, src_slice_sp)
        };

    // Get the inverse transform matrix (maps output coords to source coords)
    let inv = transform.to_inverse_matrix();

    // Create output data buffer
    let mut output_data = vec![0u16; out_width * out_height * out_depth];

    // For each output voxel, compute its position in source coordinates
    for z in 0..out_depth {
        for y in 0..out_height {
            for x in 0..out_width {
                // Output voxel center in physical coordinates (mm) using output spacing
                let phys_x = (x as f64 + 0.5) * out_col_sp;
                let phys_y = (y as f64 + 0.5) * out_row_sp;
                let phys_z = (z as f64 + 0.5) * out_slice_sp;

                // Apply inverse transform to get source physical coordinates
                let src_phys = inv.transform_point(TransformVec3::new(phys_x, phys_y, phys_z));

                // Convert to source index coordinates using source spacing
                let src_x = src_phys.x / src_col_sp;
                let src_y = src_phys.y / src_row_sp;
                let src_z = src_phys.z / src_slice_sp;

                // Trilinear interpolation from source
                let value = trilinear_sample_u16(
                    &volume.data,
                    (src_width, src_height, src_depth),
                    src_x,
                    src_y,
                    src_z,
                );

                let out_idx = x + y * out_width + z * out_width * out_height;
                output_data[out_idx] = value;
            }
        }
    }

    // Build the output volume. When target is available, adopt its full geometry
    // so that both volumes are dimensionally identical.
    if let Some(target) = target_volume {
        tracing::info!(
            "Coregistration: resampled source ({}x{}x{}, sp={:.2},{:.2},{:.2}) onto target grid ({}x{}x{}, sp={:.2},{:.2},{:.2})",
            src_width, src_height, src_depth, src_col_sp, src_row_sp, src_slice_sp,
            out_width, out_height, out_depth, out_col_sp, out_row_sp, out_slice_sp,
        );

        Volume {
            data: output_data,
            dimensions: target.dimensions,
            spacing: target.spacing,
            origin: target.origin,
            row_direction: target.row_direction,
            col_direction: target.col_direction,
            slice_direction: target.slice_direction,
            rescale_slope: volume.rescale_slope,
            rescale_intercept: volume.rescale_intercept,
            modality: volume.modality.clone(),
            series_description: Some(
                volume
                    .series_description
                    .as_ref()
                    .map(|s| format!("{} (Coregistered)", s))
                    .unwrap_or_else(|| "(Coregistered)".to_string()),
            ),
            acquisition_orientation: target.acquisition_orientation,
            frame_of_reference_uid: target.frame_of_reference_uid.clone(),
            study_instance_uid: target.study_instance_uid.clone(),
            default_window_center: volume.default_window_center,
            default_window_width: volume.default_window_width,
            patient_name: volume.patient_name.clone(),
            patient_id: volume.patient_id.clone(),
            patient_age: volume.patient_age.clone(),
            patient_sex: volume.patient_sex.clone(),
            study_date: volume.study_date.clone(),
            study_description: volume.study_description.clone(),
            original_slice_positions: if !target.original_slice_positions.is_empty() {
                target.original_slice_positions.clone()
            } else {
                volume.original_slice_positions.clone()
            },
            pixel_representation: volume.pixel_representation,
        }
    } else {
        Volume {
            data: output_data,
            dimensions: volume.dimensions,
            spacing: volume.spacing,
            origin: volume.origin,
            row_direction: volume.row_direction,
            col_direction: volume.col_direction,
            slice_direction: volume.slice_direction,
            rescale_slope: volume.rescale_slope,
            rescale_intercept: volume.rescale_intercept,
            modality: volume.modality.clone(),
            series_description: Some(
                volume
                    .series_description
                    .as_ref()
                    .map(|s| format!("{} (Coregistered)", s))
                    .unwrap_or_else(|| "(Coregistered)".to_string()),
            ),
            acquisition_orientation: volume.acquisition_orientation,
            frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
            study_instance_uid: volume.study_instance_uid.clone(),
            default_window_center: volume.default_window_center,
            default_window_width: volume.default_window_width,
            patient_name: volume.patient_name.clone(),
            patient_id: volume.patient_id.clone(),
            patient_age: volume.patient_age.clone(),
            patient_sex: volume.patient_sex.clone(),
            study_date: volume.study_date.clone(),
            study_description: volume.study_description.clone(),
            original_slice_positions: volume.original_slice_positions.clone(),
            pixel_representation: volume.pixel_representation,
        }
    }
}

/// Trilinear interpolation sample from u16 volume data
fn trilinear_sample_u16(data: &[u16], dims: (usize, usize, usize), x: f64, y: f64, z: f64) -> u16 {
    let (w, h, d) = dims;

    // Check bounds
    if x < 0.0 || y < 0.0 || z < 0.0 || x >= w as f64 || y >= h as f64 || z >= d as f64 {
        return 0;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;

    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let z1 = (z0 + 1).min(d - 1);

    let xf = x - x0 as f64;
    let yf = y - y0 as f64;
    let zf = z - z0 as f64;

    // Sample 8 corners
    let c000 = data[x0 + y0 * w + z0 * w * h] as f64;
    let c100 = data[x1 + y0 * w + z0 * w * h] as f64;
    let c010 = data[x0 + y1 * w + z0 * w * h] as f64;
    let c110 = data[x1 + y1 * w + z0 * w * h] as f64;
    let c001 = data[x0 + y0 * w + z1 * w * h] as f64;
    let c101 = data[x1 + y0 * w + z1 * w * h] as f64;
    let c011 = data[x0 + y1 * w + z1 * w * h] as f64;
    let c111 = data[x1 + y1 * w + z1 * w * h] as f64;

    // Trilinear interpolation
    let c00 = c000 * (1.0 - xf) + c100 * xf;
    let c10 = c010 * (1.0 - xf) + c110 * xf;
    let c01 = c001 * (1.0 - xf) + c101 * xf;
    let c11 = c011 * (1.0 - xf) + c111 * xf;

    let c0 = c00 * (1.0 - yf) + c10 * yf;
    let c1 = c01 * (1.0 - yf) + c11 * yf;

    let result = c0 * (1.0 - zf) + c1 * zf;
    result.round().clamp(0.0, 65535.0) as u16
}

/// Build resolution pyramid (CPU version)
#[allow(dead_code)] // coregistration pipeline internals
fn build_pyramid(data: &[f32], dims: (usize, usize, usize), levels: usize) -> Vec<Vec<f32>> {
    let mut pyramid = Vec::with_capacity(levels);

    // Level 0 = original
    pyramid.push(data.to_vec());

    let mut current = data.to_vec();
    let mut current_dims = dims;

    for _ in 1..levels {
        let new_dims = (
            (current_dims.0 / 2).max(1),
            (current_dims.1 / 2).max(1),
            (current_dims.2 / 2).max(1),
        );

        current = downsample_parallel(&current, current_dims, new_dims);
        current_dims = new_dims;
        pyramid.push(current.clone());
    }

    pyramid
}

/// Downsample volume by 2x in each dimension (parallel version)
#[allow(dead_code)] // coregistration pipeline internals
fn downsample_parallel(
    data: &[f32],
    dims: (usize, usize, usize),
    new_dims: (usize, usize, usize),
) -> Vec<f32> {
    let (w, h, _d) = dims;
    let (nw, nh, nd) = new_dims;

    let result: Vec<f32> = (0..nd)
        .into_par_iter()
        .flat_map(|nz| {
            let mut slice = vec![0.0f32; nw * nh];
            for ny in 0..nh {
                for nx in 0..nw {
                    // Sample 2x2x2 block from source
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let sx = (nx * 2 + dx).min(dims.0 - 1);
                                let sy = (ny * 2 + dy).min(dims.1 - 1);
                                let sz = (nz * 2 + dz).min(dims.2 - 1);

                                let idx = sx + sy * w + sz * w * h;
                                if idx < data.len() {
                                    sum += data[idx];
                                    count += 1;
                                }
                            }
                        }
                    }

                    slice[nx + ny * nw] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
            slice
        })
        .collect();

    result
}

/// Get dimensions at pyramid level
#[allow(dead_code)] // coregistration pipeline internals
fn pyramid_dimensions(base: (usize, usize, usize), level: usize) -> (usize, usize, usize) {
    let divisor = 1 << level; // 2^level
    (
        (base.0 / divisor).max(1),
        (base.1 / divisor).max(1),
        (base.2 / divisor).max(1),
    )
}

/// Resample volume with transformation (parallel CPU version)
#[allow(dead_code)] // coregistration pipeline internals
fn resample_volume_parallel(
    source: &[f32],
    dims: (usize, usize, usize),
    transform: &super::transform::Mat4,
) -> Vec<f32> {
    let (w, h, d) = dims;

    // Process slices in parallel
    let result: Vec<f32> = (0..d)
        .into_par_iter()
        .flat_map(|z| {
            let mut slice = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    // Output voxel center
                    let out_pos =
                        super::transform::Vec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);

                    // Transform to source coordinates
                    let src_pos = transform.transform_point(out_pos);

                    // Trilinear interpolation
                    slice[x + y * w] = trilinear_sample(source, dims, src_pos);
                }
            }
            slice
        })
        .collect();

    result
}

/// Trilinear interpolation
#[allow(dead_code)] // coregistration pipeline internals
fn trilinear_sample(data: &[f32], dims: (usize, usize, usize), pos: super::transform::Vec3) -> f32 {
    let (w, h, d) = dims;

    // Handle out of bounds
    if pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 {
        return 0.0;
    }
    if pos.x >= w as f64 || pos.y >= h as f64 || pos.z >= d as f64 {
        return 0.0;
    }

    let x0 = pos.x.floor() as usize;
    let y0 = pos.y.floor() as usize;
    let z0 = pos.z.floor() as usize;

    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let z1 = (z0 + 1).min(d - 1);

    let xf = (pos.x - x0 as f64) as f32;
    let yf = (pos.y - y0 as f64) as f32;
    let zf = (pos.z - z0 as f64) as f32;

    // Sample 8 corners
    let c000 = data[x0 + y0 * w + z0 * w * h];
    let c100 = data[x1 + y0 * w + z0 * w * h];
    let c010 = data[x0 + y1 * w + z0 * w * h];
    let c110 = data[x1 + y1 * w + z0 * w * h];
    let c001 = data[x0 + y0 * w + z1 * w * h];
    let c101 = data[x1 + y0 * w + z1 * w * h];
    let c011 = data[x0 + y1 * w + z1 * w * h];
    let c111 = data[x1 + y1 * w + z1 * w * h];

    // Trilinear interpolation
    let c00 = c000 * (1.0 - xf) + c100 * xf;
    let c10 = c010 * (1.0 - xf) + c110 * xf;
    let c01 = c001 * (1.0 - xf) + c101 * xf;
    let c11 = c011 * (1.0 - xf) + c111 * xf;

    let c0 = c00 * (1.0 - yf) + c10 * yf;
    let c1 = c01 * (1.0 - yf) + c11 * yf;

    c0 * (1.0 - zf) + c1 * zf
}
