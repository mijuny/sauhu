//! Similarity metrics for image registration
//!
//! CPU implementations as reference; GPU versions in shaders.

/// Normalized Cross-Correlation (NCC)
///
/// Range: [-1, 1], where 1 = perfect positive correlation
/// Good for same-modality registration (CT-CT, MRI-MRI)
pub fn ncc(target: &[f32], source: &[f32]) -> f64 {
    if target.len() != source.len() || target.is_empty() {
        return 0.0;
    }

    let n = target.len() as f64;

    // Compute means
    let mean_t: f64 = target.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_s: f64 = source.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Compute variances and covariance
    let mut var_t = 0.0;
    let mut var_s = 0.0;
    let mut cov = 0.0;

    for (&t, &s) in target.iter().zip(source.iter()) {
        let dt = t as f64 - mean_t;
        let ds = s as f64 - mean_s;
        var_t += dt * dt;
        var_s += ds * ds;
        cov += dt * ds;
    }

    let std_t = var_t.sqrt();
    let std_s = var_s.sqrt();

    if std_t < 1e-10 || std_s < 1e-10 {
        return 0.0;
    }

    cov / (std_t * std_s)
}

/// Sum of Squared Differences (SSD)
///
/// Lower is better. Simple but sensitive to intensity differences.
pub fn ssd(target: &[f32], source: &[f32]) -> f64 {
    if target.len() != source.len() {
        return f64::MAX;
    }

    target
        .iter()
        .zip(source.iter())
        .map(|(&t, &s)| {
            let diff = t as f64 - s as f64;
            diff * diff
        })
        .sum()
}

/// Normalized SSD (0-1 range, lower is better)
pub fn normalized_ssd(target: &[f32], source: &[f32]) -> f64 {
    if target.len() != source.len() || target.is_empty() {
        return 1.0;
    }

    let n = target.len() as f64;

    // Compute means
    let mean_t: f64 = target.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_s: f64 = source.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Compute variance of target for normalization
    let var_t: f64 = target.iter().map(|&x| (x as f64 - mean_t).powi(2)).sum();

    if var_t < 1e-10 {
        return 1.0;
    }

    let ssd_val: f64 = target
        .iter()
        .zip(source.iter())
        .map(|(&t, &s)| {
            let diff = (t as f64 - mean_t) - (s as f64 - mean_s);
            diff * diff
        })
        .sum();

    (ssd_val / var_t).min(1.0)
}

/// Mutual Information (MI) using histogram binning
///
/// Range: [0, inf), higher is better
/// Good for cross-modality registration (CT-MRI)
pub fn mutual_information(target: &[f32], source: &[f32], num_bins: usize) -> f64 {
    if target.len() != source.len() || target.is_empty() {
        return 0.0;
    }

    let n = target.len() as f64;

    // Find intensity ranges
    let (t_min, t_max) = intensity_range(target);
    let (s_min, s_max) = intensity_range(source);

    if t_max - t_min < 1e-10 || s_max - s_min < 1e-10 {
        return 0.0;
    }

    // Build joint histogram
    let mut joint_hist = vec![0.0f64; num_bins * num_bins];
    let t_scale = (num_bins - 1) as f64 / (t_max - t_min);
    let s_scale = (num_bins - 1) as f64 / (s_max - s_min);

    for (&t, &s) in target.iter().zip(source.iter()) {
        let ti = ((t as f64 - t_min) * t_scale).round() as usize;
        let si = ((s as f64 - s_min) * s_scale).round() as usize;
        let ti = ti.min(num_bins - 1);
        let si = si.min(num_bins - 1);
        joint_hist[ti * num_bins + si] += 1.0;
    }

    // Normalize to probability
    for h in joint_hist.iter_mut() {
        *h /= n;
    }

    // Compute marginal histograms
    let mut hist_t = vec![0.0f64; num_bins];
    let mut hist_s = vec![0.0f64; num_bins];

    for ti in 0..num_bins {
        for si in 0..num_bins {
            let p = joint_hist[ti * num_bins + si];
            hist_t[ti] += p;
            hist_s[si] += p;
        }
    }

    // Compute mutual information
    let mut mi = 0.0;
    for ti in 0..num_bins {
        for si in 0..num_bins {
            let p_joint = joint_hist[ti * num_bins + si];
            let p_t = hist_t[ti];
            let p_s = hist_s[si];

            if p_joint > 1e-10 && p_t > 1e-10 && p_s > 1e-10 {
                mi += p_joint * (p_joint / (p_t * p_s)).ln();
            }
        }
    }

    mi
}

/// Normalized Mutual Information (NMI)
///
/// Range: [1, 2] typically, where 2 = identical images
/// More robust than MI to overlap changes
pub fn normalized_mutual_information(target: &[f32], source: &[f32], num_bins: usize) -> f64 {
    if target.len() != source.len() || target.is_empty() {
        return 1.0;
    }

    let n = target.len() as f64;

    // Find intensity ranges
    let (t_min, t_max) = intensity_range(target);
    let (s_min, s_max) = intensity_range(source);

    if t_max - t_min < 1e-10 || s_max - s_min < 1e-10 {
        return 1.0;
    }

    // Build joint histogram
    let mut joint_hist = vec![0.0f64; num_bins * num_bins];
    let t_scale = (num_bins - 1) as f64 / (t_max - t_min);
    let s_scale = (num_bins - 1) as f64 / (s_max - s_min);

    for (&t, &s) in target.iter().zip(source.iter()) {
        let ti = ((t as f64 - t_min) * t_scale).round() as usize;
        let si = ((s as f64 - s_min) * s_scale).round() as usize;
        let ti = ti.min(num_bins - 1);
        let si = si.min(num_bins - 1);
        joint_hist[ti * num_bins + si] += 1.0;
    }

    // Normalize
    for h in joint_hist.iter_mut() {
        *h /= n;
    }

    // Marginals
    let mut hist_t = vec![0.0f64; num_bins];
    let mut hist_s = vec![0.0f64; num_bins];

    for ti in 0..num_bins {
        for si in 0..num_bins {
            let p = joint_hist[ti * num_bins + si];
            hist_t[ti] += p;
            hist_s[si] += p;
        }
    }

    // Entropies
    let entropy = |hist: &[f64]| -> f64 {
        hist.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    };

    let h_t = entropy(&hist_t);
    let h_s = entropy(&hist_s);

    let mut h_joint = 0.0;
    for &p in joint_hist.iter() {
        if p > 1e-10 {
            h_joint -= p * p.ln();
        }
    }

    // NMI = (H(T) + H(S)) / H(T,S)
    if h_joint > 1e-10 {
        (h_t + h_s) / h_joint
    } else {
        1.0
    }
}

/// Find min and max intensity values
fn intensity_range(data: &[f32]) -> (f64, f64) {
    let mut min = f64::MAX;
    let mut max = f64::MIN;

    for &v in data {
        let v = v as f64;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    (min, max)
}

/// Metric type for registration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Normalized Cross-Correlation (for same modality)
    Ncc,
    /// Sum of Squared Differences (fastest, same modality)
    Ssd,
    /// Mutual Information (for cross-modality)
    MI,
    /// Normalized Mutual Information (cross-modality, more robust)
    Nmi,
}

impl MetricType {
    /// Compute metric value
    pub fn compute(&self, target: &[f32], source: &[f32]) -> f64 {
        match self {
            MetricType::Ncc => ncc(target, source),
            MetricType::Ssd => -normalized_ssd(target, source), // Negate so higher is better
            MetricType::MI => mutual_information(target, source, 64),
            MetricType::Nmi => normalized_mutual_information(target, source, 64),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncc_identical() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let ncc_val = ncc(&a, &a);
        assert!((ncc_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ncc_inverted() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..100).map(|x| (99 - x) as f32).collect();
        let ncc_val = ncc(&a, &b);
        assert!((ncc_val + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ncc_uncorrelated() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // Constant
        let ncc_val = ncc(&a, &b);
        // NCC with constant image should be 0 or NaN-handled
        assert!(ncc_val.abs() < 1e-6 || ncc_val.is_nan());
    }

    #[test]
    fn test_mi_identical() {
        let a: Vec<f32> = (0..1000).map(|x| (x % 64) as f32).collect();
        let mi_val = mutual_information(&a, &a, 64);
        assert!(mi_val > 0.0); // Positive MI for identical images
    }
}
