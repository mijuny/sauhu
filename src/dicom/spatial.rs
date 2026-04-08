//! Spatial calculations for DICOM image geometry
//!
//! This module contains pure functions for geometric calculations used by
//! both the main Sauhu viewer and the Claude sandbox. By keeping these
//! functions separate from UI state, we ensure identical behavior in both.
//!
//! # Design Principle
//!
//! All calculation logic lives here. The viewers (Sauhu UI, Claude sandbox)
//! only handle rendering. This guarantees feature parity.
//!
//! Many functions here are not yet called from the main viewer but form
//! the geometry library for upcoming viewport sync, reference lines,
//! and measurement features.
#![allow(dead_code)]

use super::{ImagePlane, Point2D, ReferenceLine, Vec3};

/// Compute reference line where two image planes intersect
/// Returns None if planes are parallel or don't share Frame of Reference
pub fn compute_reference_line(
    target_plane: &ImagePlane,
    source_plane: &ImagePlane,
) -> Option<ReferenceLine> {
    // Must share coordinate system
    if !target_plane.same_frame_of_reference(source_plane) {
        return None;
    }

    // Parallel planes don't intersect in a line
    if target_plane.is_parallel(source_plane) {
        return None;
    }

    // Use ImagePlane's intersection calculation
    target_plane
        .intersect(source_plane)
        .map(|(start, end)| ReferenceLine::new(start, end))
}

/// Compute where a 3D patient coordinate projects onto an image plane
/// Returns pixel coordinates (column, row) or None if point is behind plane
pub fn compute_point_in_plane(point: &Vec3, plane: &ImagePlane) -> Option<Point2D> {
    // Project point onto plane
    let pixel = plane.patient_to_pixel(point);

    // Check if within image bounds (with some margin for edge cases)
    let (rows, cols) = plane.dimensions;
    let margin = 5.0;

    if pixel.x >= -margin
        && pixel.x < cols as f64 + margin
        && pixel.y >= -margin
        && pixel.y < rows as f64 + margin
    {
        Some(pixel)
    } else {
        None
    }
}

/// Compute distance between two points in mm using pixel spacing
pub fn compute_distance_mm(p1: (f64, f64), p2: (f64, f64), pixel_spacing: (f64, f64)) -> f64 {
    let (row_spacing, col_spacing) = pixel_spacing;
    let dx = (p2.0 - p1.0) * col_spacing;
    let dy = (p2.1 - p1.1) * row_spacing;
    (dx * dx + dy * dy).sqrt()
}

/// Compute 3D patient coordinate from pixel position on a plane
pub fn compute_patient_coord(pixel_x: f64, pixel_y: f64, plane: &ImagePlane) -> Vec3 {
    plane.pixel_to_patient(pixel_x, pixel_y)
}

/// Find the closest slice index for a given slice location
pub fn find_closest_slice(slice_locations: &[Option<f64>], target: f64) -> usize {
    slice_locations
        .iter()
        .enumerate()
        .filter_map(|(i, loc)| loc.map(|l| (i, (l - target).abs())))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute sync slice index in target series for a given source location
pub fn compute_sync_slice(source_location: f64, target_slice_locations: &[Option<f64>]) -> usize {
    find_closest_slice(target_slice_locations, source_location)
}

/// Check if two planes should sync based on Frame of Reference and orientation
pub fn should_planes_sync(
    plane_a_frame_uid: Option<&str>,
    plane_a_orientation: Option<&str>,
    plane_b_frame_uid: Option<&str>,
    plane_b_orientation: Option<&str>,
) -> bool {
    // Must have same Frame of Reference
    match (plane_a_frame_uid, plane_b_frame_uid) {
        (Some(a), Some(b)) if a == b => {}
        _ => return false,
    }

    // Must have same orientation
    matches!((plane_a_orientation, plane_b_orientation), (Some(a), Some(b)) if a == b)
}

/// Compute where a 3D point appears in multiple planes
/// Returns a map of plane index to pixel coordinates
pub fn compute_point_across_planes<'a>(
    point: &Vec3,
    planes: impl Iterator<Item = (usize, &'a ImagePlane)>,
) -> Vec<(usize, Point2D)> {
    planes
        .filter_map(|(idx, plane)| compute_point_in_plane(point, plane).map(|pixel| (idx, pixel)))
        .collect()
}

/// Compute reference lines from one source plane to multiple target planes
pub fn compute_reference_lines_to_planes<'a>(
    source_plane: &ImagePlane,
    target_planes: impl Iterator<Item = (usize, &'a ImagePlane)>,
) -> Vec<(usize, ReferenceLine)> {
    target_planes
        .filter_map(|(idx, target)| {
            compute_reference_line(target, source_plane).map(|line| (idx, line))
        })
        .collect()
}

/// ROI statistics result
#[derive(Debug, Clone, Copy)]
pub struct ROIStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub pixel_count: usize,
}

/// Image pixel data with rescale parameters for ROI statistics
pub struct ImagePixels<'a> {
    pub data: &'a [u16],
    pub width: u32,
    pub height: u32,
    pub rescale_slope: f64,
    pub rescale_intercept: f64,
}

/// Compute statistics for a circular ROI
/// Takes pixel values and their positions, center, and radius in pixels
pub fn compute_circle_roi_stats(
    image: &ImagePixels,
    center_x: f64,
    center_y: f64,
    radius: f64,
) -> Option<ROIStats> {
    if radius <= 0.0 {
        return None;
    }

    let mut values = Vec::new();
    let r2 = radius * radius;

    let x_min = ((center_x - radius).floor() as i32).max(0) as u32;
    let x_max = ((center_x + radius).ceil() as i32).min(image.width as i32 - 1) as u32;
    let y_min = ((center_y - radius).floor() as i32).max(0) as u32;
    let y_max = ((center_y + radius).ceil() as i32).min(image.height as i32 - 1) as u32;

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            if dx * dx + dy * dy <= r2 {
                let idx = (y * image.width + x) as usize;
                if idx < image.data.len() {
                    let raw = image.data[idx] as f64;
                    let value = raw * image.rescale_slope + image.rescale_intercept;
                    values.push(value);
                }
            }
        }
    }

    if values.is_empty() {
        return None;
    }

    let count = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    Some(ROIStats {
        mean,
        min,
        max,
        std_dev,
        pixel_count: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_calculation() {
        // 3-4-5 triangle with 1mm spacing
        let dist = compute_distance_mm((0.0, 0.0), (3.0, 4.0), (1.0, 1.0));
        assert!((dist - 5.0).abs() < 0.001);

        // With non-square pixels
        let dist = compute_distance_mm((0.0, 0.0), (3.0, 4.0), (2.0, 1.0));
        // dx = 3 * 1 = 3, dy = 4 * 2 = 8, dist = sqrt(9 + 64) = sqrt(73)
        assert!((dist - 73.0_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_find_closest_slice() {
        let locations = vec![Some(0.0), Some(5.0), Some(10.0), Some(15.0)];

        assert_eq!(find_closest_slice(&locations, 0.0), 0);
        assert_eq!(find_closest_slice(&locations, 4.0), 1);
        assert_eq!(find_closest_slice(&locations, 7.0), 1);
        assert_eq!(find_closest_slice(&locations, 8.0), 2);
        assert_eq!(find_closest_slice(&locations, 100.0), 3);
    }
}
