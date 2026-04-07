//! Powell's method optimizer for rigid registration
//!
//! Derivative-free optimization suitable for image registration where
//! gradients are expensive or noisy.

use super::transform::{RigidTransform, TransformBounds};

/// Optimization result
#[derive(Debug, Clone)]
#[allow(dead_code)] // coregistration module API
pub struct OptimizationResult {
    /// Final transform
    pub transform: RigidTransform,
    /// Final metric value (higher is better for NCC)
    pub metric: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

/// Powell's method optimizer
pub struct PowellOptimizer {
    /// Parameter bounds
    pub bounds: TransformBounds,
    /// Whether to optimize scale (7 DOF vs 6 DOF)
    pub optimize_scale: bool,
    /// Convergence tolerance for metric improvement
    pub tolerance: f64,
    /// Initial step sizes for each parameter
    pub initial_steps: [f64; 7],
    /// Minimum step size (convergence criterion)
    pub min_step: f64,
}

impl Default for PowellOptimizer {
    fn default() -> Self {
        Self {
            bounds: TransformBounds::default(),
            optimize_scale: false,
            tolerance: 1e-6,
            initial_steps: [
                0.05, // rotation_x (radians, ~3 degrees)
                0.05, // rotation_y
                0.05, // rotation_z
                5.0,  // translation_x (mm)
                5.0,  // translation_y
                5.0,  // translation_z
                0.02, // scale
            ],
            min_step: 1e-4,
        }
    }
}

impl PowellOptimizer {
    /// Create optimizer for given DOF
    pub fn new(optimize_scale: bool) -> Self {
        Self {
            optimize_scale,
            ..Default::default()
        }
    }

    /// Set step sizes appropriate for the resolution level
    pub fn with_steps(mut self, rotation_step: f64, translation_step: f64) -> Self {
        self.initial_steps[0] = rotation_step;
        self.initial_steps[1] = rotation_step;
        self.initial_steps[2] = rotation_step;
        self.initial_steps[3] = translation_step;
        self.initial_steps[4] = translation_step;
        self.initial_steps[5] = translation_step;
        self
    }

    /// Run optimization
    ///
    /// The `metric_fn` takes a transform and returns a similarity metric
    /// (higher is better, e.g., NCC).
    pub fn optimize<F>(
        &self,
        initial: RigidTransform,
        max_iterations: usize,
        metric_fn: F,
    ) -> OptimizationResult
    where
        F: Fn(&RigidTransform) -> f64,
    {
        let n_params = if self.optimize_scale { 7 } else { 6 };

        let mut params = initial.to_params();
        let mut steps = self.initial_steps;
        let mut best_metric = metric_fn(&initial);

        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..max_iterations {
            iterations = iter + 1;
            let metric_before = best_metric;

            // Optimize along each parameter direction
            for (dim, &step) in steps[..n_params].iter().enumerate() {
                let (new_params, new_metric) =
                    self.line_search(&params, dim, step, best_metric, &metric_fn);

                if new_metric > best_metric {
                    params = new_params;
                    best_metric = new_metric;
                }
            }

            // Check convergence
            let improvement = best_metric - metric_before;
            if improvement < self.tolerance && iter > 0 {
                // Reduce step sizes
                let mut all_small = true;
                for step in steps[..n_params].iter_mut() {
                    *step *= 0.5;
                    if *step > self.min_step {
                        all_small = false;
                    }
                }

                if all_small {
                    converged = true;
                    break;
                }
            }
        }

        let mut transform = RigidTransform::from_params(&params);
        self.bounds.clamp(&mut transform);

        OptimizationResult {
            transform,
            metric: best_metric,
            iterations,
            converged,
        }
    }

    /// Line search along one parameter dimension
    fn line_search<F>(
        &self,
        params: &[f64; 7],
        dim: usize,
        step: f64,
        current_metric: f64,
        metric_fn: &F,
    ) -> ([f64; 7], f64)
    where
        F: Fn(&RigidTransform) -> f64,
    {
        let mut best_params = *params;
        let mut best_metric = current_metric;

        // Try positive direction
        let mut test_params = *params;
        test_params[dim] += step;
        let mut transform = RigidTransform::from_params(&test_params);
        self.bounds.clamp(&mut transform);
        test_params = transform.to_params();

        let metric_pos = metric_fn(&transform);
        if metric_pos > best_metric {
            best_params = test_params;
            best_metric = metric_pos;

            // Continue in positive direction (golden section-ish)
            let mut current_step = step;
            for _ in 0..5 {
                current_step *= 1.5;
                test_params[dim] = params[dim] + current_step;
                transform = RigidTransform::from_params(&test_params);
                self.bounds.clamp(&mut transform);
                test_params = transform.to_params();

                let m = metric_fn(&transform);
                if m > best_metric {
                    best_params = test_params;
                    best_metric = m;
                } else {
                    break;
                }
            }
        }

        // Try negative direction
        test_params = *params;
        test_params[dim] -= step;
        transform = RigidTransform::from_params(&test_params);
        self.bounds.clamp(&mut transform);
        test_params = transform.to_params();

        let metric_neg = metric_fn(&transform);
        if metric_neg > best_metric {
            best_params = test_params;
            best_metric = metric_neg;

            // Continue in negative direction
            let mut current_step = step;
            for _ in 0..5 {
                current_step *= 1.5;
                test_params[dim] = params[dim] - current_step;
                transform = RigidTransform::from_params(&test_params);
                self.bounds.clamp(&mut transform);
                test_params = transform.to_params();

                let m = metric_fn(&transform);
                if m > best_metric {
                    best_params = test_params;
                    best_metric = m;
                } else {
                    break;
                }
            }
        }

        (best_params, best_metric)
    }
}

/// Multi-resolution optimization schedule
#[derive(Debug, Clone)]
#[allow(dead_code)] // coregistration module API
pub struct PyramidSchedule {
    /// Number of iterations per level (coarse to fine)
    pub iterations: Vec<usize>,
    /// Rotation step per level (radians)
    pub rotation_steps: Vec<f64>,
    /// Translation step per level (mm)
    pub translation_steps: Vec<f64>,
}

impl Default for PyramidSchedule {
    fn default() -> Self {
        Self {
            // 4 levels: 64³, 128³, 256³, full
            iterations: vec![50, 30, 20, 10],
            rotation_steps: vec![0.1, 0.05, 0.02, 0.01], // ~6°, 3°, 1°, 0.5°
            translation_steps: vec![10.0, 5.0, 2.0, 1.0], // mm
        }
    }
}

impl PyramidSchedule {
    /// Fast schedule for quick preview (2-5 seconds target)
    /// Uses only 2 levels at low resolution with minimal iterations
    /// May not produce optimal alignment - use for previews only
    pub fn fast() -> Self {
        Self {
            // 2 levels only: 64³ and 128³
            iterations: vec![15, 10],
            rotation_steps: vec![0.05, 0.02],  // ~3°, 1°
            translation_steps: vec![5.0, 2.0], // mm
        }
    }

    /// Balanced schedule for good quality with reasonable speed (5-15 seconds)
    /// 3 levels with more iterations for reliable convergence
    #[allow(dead_code)] // coregistration module API
    pub fn balanced() -> Self {
        Self {
            // 3 levels: 32³ → 64³ → 128³
            iterations: vec![30, 25, 20],
            rotation_steps: vec![0.15, 0.08, 0.03], // ~9°, 5°, 2°
            translation_steps: vec![12.0, 6.0, 2.0], // mm (physical)
        }
    }

    /// High quality schedule for best alignment (15-30 seconds)
    /// 4 levels with many iterations for precise sub-voxel alignment
    #[allow(dead_code)] // coregistration module API
    pub fn quality() -> Self {
        Self {
            // 4 levels: 32³ → 64³ → 128³ → 192³
            iterations: vec![40, 30, 25, 15],
            rotation_steps: vec![0.2, 0.1, 0.04, 0.015], // ~12°, 6°, 2°, 1°
            translation_steps: vec![15.0, 8.0, 3.0, 1.0], // mm
        }
    }

    /// Get optimizer settings for a pyramid level (0 = coarsest, higher = finer)
    #[allow(dead_code)] // coregistration module API
    pub fn for_level(&self, level: usize) -> (usize, f64, f64) {
        let idx = level.min(self.iterations.len() - 1);
        (
            self.iterations[idx],
            self.rotation_steps[idx],
            self.translation_steps[idx],
        )
    }

    /// Number of levels
    #[allow(dead_code)] // coregistration module API
    pub fn num_levels(&self) -> usize {
        self.iterations.len()
    }

    /// Get resolution for a pyramid level (coarsest first)
    /// Returns the target cube size for downsampling
    #[allow(dead_code)] // coregistration module API
    pub fn resolution_for_level(&self, level: usize) -> usize {
        // Standard resolutions: 32, 64, 128, 192
        match level {
            0 => 32,
            1 => 64,
            2 => 128,
            3 => 192,
            _ => 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_identity() {
        let optimizer = PowellOptimizer::default();

        // Metric function that peaks at identity transform
        let metric_fn = |t: &RigidTransform| {
            let params = t.to_params();
            // Negative sum of squared parameters (higher = closer to zero)
            -params[0..6].iter().map(|x| x * x).sum::<f64>()
        };

        let initial = RigidTransform::identity();
        let result = optimizer.optimize(initial, 10, metric_fn);

        // Should stay near identity
        assert!(result.transform.rotation_x.abs() < 0.1);
        assert!(result.transform.translation_x.abs() < 1.0);
    }

    #[test]
    fn test_optimizer_finds_offset() {
        let optimizer = PowellOptimizer::default();

        // Metric function that peaks at tx=10
        let metric_fn = |t: &RigidTransform| {
            let target_tx = 10.0;
            -(t.translation_x - target_tx).powi(2)
        };

        let initial = RigidTransform::identity();
        let result = optimizer.optimize(initial, 50, metric_fn);

        assert!((result.transform.translation_x - 10.0).abs() < 1.0);
    }
}
