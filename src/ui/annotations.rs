//! Annotation and measurement types for DICOM images
#![allow(dead_code)]

use std::collections::HashMap;

/// A point in image coordinates (pixels)
#[derive(Debug, Clone, Copy)]
pub struct AnnotationPoint {
    pub x: f64,
    pub y: f64,
}

impl AnnotationPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Calculate distance to another point in pixels
    pub fn distance_to(&self, other: &AnnotationPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Result of a length measurement
#[derive(Debug, Clone, Copy)]
pub struct DistanceResult {
    /// The distance value
    pub value: f64,
    /// Whether the value is in mm (true) or pixels (false)
    pub is_calibrated: bool,
}

impl std::fmt::Display for DistanceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_calibrated {
            write!(f, "{:.1} mm", self.value)
        } else {
            write!(f, "{:.0} px (no calibration)", self.value)
        }
    }
}

/// A length measurement between two points
#[derive(Debug, Clone)]
pub struct LengthMeasurement {
    /// Unique ID for this measurement
    pub id: u64,
    /// Start point in image coordinates
    pub start: AnnotationPoint,
    /// End point in image coordinates
    pub end: AnnotationPoint,
    /// Calculated distance result
    pub distance: DistanceResult,
}

/// Statistics from a circular ROI
#[derive(Debug, Clone, Copy)]
pub struct ROIStatistics {
    /// Mean value in the ROI
    pub mean: f64,
    /// Minimum value in the ROI
    pub min: f64,
    /// Maximum value in the ROI
    pub max: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of pixels in the ROI
    pub pixel_count: usize,
}

impl ROIStatistics {
    /// Format as multiline string for display
    pub fn to_display_string(self, is_ct: bool) -> String {
        let unit = if is_ct { " HU" } else { "" };
        format!(
            "Mean: {:.1}{}\nSD: {:.1}\nMin: {:.1} Max: {:.1}\nN: {}",
            self.mean, unit, self.std_dev, self.min, self.max, self.pixel_count
        )
    }
}

/// A circular ROI measurement
#[derive(Debug, Clone)]
pub struct CircleROI {
    /// Unique ID for this ROI
    pub id: u64,
    /// Center point in image coordinates
    pub center: AnnotationPoint,
    /// Radius in pixels
    pub radius_px: f64,
    /// Calculated statistics (if available)
    pub stats: Option<ROIStatistics>,
}

/// Types of annotations
#[derive(Debug, Clone)]
pub enum Annotation {
    Length(LengthMeasurement),
    Circle(CircleROI),
}

impl Annotation {
    pub fn id(&self) -> u64 {
        match self {
            Annotation::Length(m) => m.id,
            Annotation::Circle(r) => r.id,
        }
    }
}

/// Current measurement state (what the user is drawing)
#[derive(Debug, Clone, Default)]
pub enum MeasurementState {
    /// Not drawing anything
    #[default]
    Idle,
    /// Drawing a length measurement - first point placed
    DrawingLength { start: AnnotationPoint },
    /// Drawing a circle ROI - center placed, dragging radius
    DrawingCircle { center: AnnotationPoint },
    /// Drawing a circle for ROI-based windowing
    DrawingROIWindow { center: AnnotationPoint },
}

impl MeasurementState {
    pub fn is_idle(&self) -> bool {
        matches!(self, MeasurementState::Idle)
    }

    pub fn is_drawing(&self) -> bool {
        !self.is_idle()
    }
}

/// Storage for annotations, keyed by image identifier (SOP Instance UID or synthetic path)
#[derive(Debug, Default)]
pub struct AnnotationStore {
    /// Map: image key -> Vec<Annotation>
    annotations: HashMap<String, Vec<Annotation>>,
    /// Next annotation ID
    next_id: u64,
}

impl AnnotationStore {
    pub fn new() -> Self {
        Self {
            annotations: HashMap::new(),
            next_id: 1,
        }
    }

    /// Get next unique ID and increment counter
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Add an annotation for a specific image
    pub fn add(&mut self, image_key: &str, annotation: Annotation) {
        self.annotations
            .entry(image_key.to_string())
            .or_default()
            .push(annotation);
    }

    /// Get all annotations for an image
    pub fn get(&self, image_key: &str) -> Option<&Vec<Annotation>> {
        self.annotations.get(image_key)
    }

    /// Remove an annotation by ID
    pub fn remove(&mut self, image_key: &str, annotation_id: u64) {
        if let Some(list) = self.annotations.get_mut(image_key) {
            list.retain(|a| a.id() != annotation_id);
        }
    }

    /// Clear all annotations for an image
    pub fn clear_image(&mut self, image_key: &str) {
        self.annotations.remove(image_key);
    }

    /// Clear all annotations
    pub fn clear_all(&mut self) {
        self.annotations.clear();
    }

    /// Check if an image has any annotations
    pub fn has_annotations(&self, image_key: &str) -> bool {
        self.annotations
            .get(image_key)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }
}
