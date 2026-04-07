//! Multi-Planar Reconstruction (MPR) for 3D volume reformatting
//!
//! Allows viewing volumetric series (e.g., 3D T1 brain MRI) in
//! axial, coronal, or sagittal planes regardless of acquisition orientation.
//!
//! # Orientation Handling
//!
//! The module automatically detects the acquisition orientation (axial, coronal,
//! or sagittal) from the DICOM ImageOrientationPatient tag and applies appropriate
//! transformations when reformatting to other planes.
//!
//! For sagittal acquisitions (common in 3D brain MRI), the output is transposed
//! to maintain correct anatomical orientation:
//! - Axial: X (left-right) horizontal, Y (anterior-posterior) vertical
//! - Coronal: X (left-right) horizontal, Z (superior-inferior) vertical
//! - Sagittal: Y (anterior-posterior) horizontal, Z (superior-inferior) vertical
//!
//! # Usage
//!
//! ```ignore
//! // Build volume from series
//! let volume = Volume::from_series(&images)?;
//!
//! // Create MPR state for a viewport
//! let mut mpr = MprState::new();
//! mpr.set_volume(Arc::new(volume));
//! mpr.set_plane(AnatomicalPlane::Coronal);
//!
//! // Get reformatted slice
//! let slice = mpr.get_slice();
//! ```

use super::{DicomImage, ImagePlane, SyncInfo, Vec3};
use std::sync::Arc;

/// Standard anatomical viewing planes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnatomicalPlane {
    /// Axial/Transverse - view from feet (X-Y plane)
    Axial,
    /// Coronal/Frontal - view from front (X-Z plane)
    Coronal,
    /// Sagittal/Lateral - view from right side (Y-Z plane)
    Sagittal,
    /// Original acquisition plane
    Original,
}

impl AnatomicalPlane {
    /// Short abbreviation used in UI and sync matching
    #[allow(dead_code)]
    pub fn abbrev(&self) -> &'static str {
        match self {
            AnatomicalPlane::Axial => "Ax",
            AnatomicalPlane::Coronal => "Cor",
            AnatomicalPlane::Sagittal => "Sag",
            AnatomicalPlane::Original => "Orig",
        }
    }

    /// Full display name
    pub fn full_name(&self) -> &'static str {
        match self {
            AnatomicalPlane::Axial => "Axial",
            AnatomicalPlane::Coronal => "Coronal",
            AnatomicalPlane::Sagittal => "Sagittal",
            AnatomicalPlane::Original => "Original",
        }
    }

    /// Parse from short abbreviation ("Ax", "Cor", "Sag")
    #[allow(dead_code)]
    pub fn from_abbrev(s: &str) -> Option<Self> {
        match s {
            "Ax" => Some(AnatomicalPlane::Axial),
            "Cor" => Some(AnatomicalPlane::Coronal),
            "Sag" => Some(AnatomicalPlane::Sagittal),
            "Orig" => Some(AnatomicalPlane::Original),
            _ => None,
        }
    }
}

impl std::fmt::Display for AnatomicalPlane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.full_name())
    }
}

/// Patient coordinate axes (LPS convention)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatientAxis {
    /// Left-Right axis (X)
    X,
    /// Anterior-Posterior axis (Y)
    Y,
    /// Superior-Inferior axis (Z)
    Z,
}

/// Detected acquisition orientation of the original series
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionOrientation {
    /// Slices acquired axially (most common - CT, most MRI)
    Axial,
    /// Slices acquired coronally
    Coronal,
    /// Slices acquired sagittally (3D brain acquisitions often)
    Sagittal,
    /// Unknown or oblique
    Unknown,
}

/// Internal reslice operation type
#[derive(Debug, Clone, Copy)]
enum ResliceOperation {
    /// Return native slices (no reslice needed)
    Native,
    /// Take rows from each slice (X-Z plane, like Horos X-reslice)
    AlongRows,
    /// Take columns from each slice (Y-Z plane, like Horos Y-reslice)
    AlongColumns,
}

impl AcquisitionOrientation {
    /// Detect acquisition orientation from the slice normal vector
    /// Normal vector points perpendicular to the slice plane
    pub fn from_normal(normal: &Vec3) -> Self {
        // Find which axis the normal is most aligned with
        let abs_x = normal.x.abs();
        let abs_y = normal.y.abs();
        let abs_z = normal.z.abs();

        // Threshold for "mostly aligned" (cos(30°) ≈ 0.866)
        let threshold = 0.7;

        if abs_z > threshold && abs_z > abs_x && abs_z > abs_y {
            // Normal points mostly S-I (z-axis) → Axial acquisition
            AcquisitionOrientation::Axial
        } else if abs_y > threshold && abs_y > abs_x && abs_y > abs_z {
            // Normal points mostly A-P (y-axis) → Coronal acquisition
            AcquisitionOrientation::Coronal
        } else if abs_x > threshold && abs_x > abs_y && abs_x > abs_z {
            // Normal points mostly L-R (x-axis) → Sagittal acquisition
            AcquisitionOrientation::Sagittal
        } else {
            AcquisitionOrientation::Unknown
        }
    }
}

/// 3D volume constructed from a series of DICOM slices
#[derive(Clone)]
pub struct Volume {
    /// 3D voxel data in Z-Y-X order (slice, row, column)
    /// Index: z * (width * height) + y * width + x
    pub data: Vec<u16>,
    /// Volume dimensions (width, height, depth) = (columns, rows, slices)
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in mm (column_spacing, row_spacing, slice_spacing)
    pub spacing: (f64, f64, f64),
    /// Volume origin in patient coordinates (top-left-front corner)
    pub origin: Vec3,
    /// Unit vector along columns (left to right in slice)
    pub row_direction: Vec3,
    /// Unit vector along rows (top to bottom in slice)
    pub col_direction: Vec3,
    /// Unit vector along slices (slice normal direction)
    pub slice_direction: Vec3,
    /// Rescale slope for HU conversion
    pub rescale_slope: f64,
    /// Rescale intercept for HU conversion
    pub rescale_intercept: f64,
    /// Original modality
    pub modality: Option<String>,
    /// Original series description
    pub series_description: Option<String>,
    /// Detected acquisition orientation
    pub acquisition_orientation: AcquisitionOrientation,
    /// Frame of Reference UID (for sync matching - same coordinate system)
    pub frame_of_reference_uid: Option<String>,
    /// Study Instance UID (for sync matching - same exam)
    pub study_instance_uid: Option<String>,
    /// Default window center from original series
    pub default_window_center: f64,
    /// Default window width from original series
    pub default_window_width: f64,
    // --- Patient/Study Metadata (for MPR image display) ---
    /// Patient name
    pub patient_name: Option<String>,
    /// Patient ID
    pub patient_id: Option<String>,
    /// Patient age
    pub patient_age: Option<String>,
    /// Patient sex
    pub patient_sex: Option<String>,
    /// Study date
    pub study_date: Option<String>,
    /// Study description
    pub study_description: Option<String>,
    /// Original ImagePositionPatient coordinates (for sync with original series)
    /// Indexed by volume z coordinate. Contains the appropriate patient coordinate based on
    /// acquisition orientation (z for axial, y for coronal, x for sagittal).
    pub original_slice_positions: Vec<f64>,
    /// Pixel representation (0 = unsigned, 1 = signed)
    pub pixel_representation: u16,
}

impl Volume {
    /// Construct a 3D volume from a series of DicomImages
    ///
    /// Slices must be:
    /// - From the same Frame of Reference
    /// - Parallel (same orientation)
    /// - Evenly spaced (or close to it)
    pub fn from_series(images: &[impl std::borrow::Borrow<DicomImage>]) -> Option<Self> {
        let images: Vec<&DicomImage> = images.iter().map(|i| i.borrow()).collect();
        if images.len() < 2 {
            tracing::debug!("Volume requires at least 2 slices, got {}", images.len());
            return None;
        }

        // Get image planes from all slices
        let planes: Vec<&ImagePlane> = images
            .iter()
            .filter_map(|img| img.image_plane.as_ref())
            .collect();

        if planes.len() != images.len() {
            tracing::debug!("Not all images have geometry data");
            return None;
        }

        // Verify all slices are parallel (same orientation)
        let first_plane = planes[0];
        for plane in &planes[1..] {
            if !first_plane.is_parallel(plane) {
                tracing::debug!("Slices are not parallel");
                return None;
            }
            if !first_plane.same_frame_of_reference(plane) {
                tracing::debug!("Slices have different Frame of Reference");
                return None;
            }
        }

        // Sort slices by position along the slice normal
        let normal = first_plane.normal;
        let mut slice_indices: Vec<(usize, f64)> = images
            .iter()
            .enumerate()
            .filter_map(|(i, img)| {
                img.image_plane.as_ref().map(|p| {
                    let pos = p.position.dot(&normal);
                    (i, pos)
                })
            })
            .collect();

        slice_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate slice spacing from first two slices
        let slice_spacing = if slice_indices.len() >= 2 {
            (slice_indices[1].1 - slice_indices[0].1).abs()
        } else {
            1.0
        };

        if slice_spacing < 0.001 {
            tracing::debug!("Invalid slice spacing: {}", slice_spacing);
            return None;
        }

        // Get dimensions
        let first_image = &images[0];
        let width = first_image.width as usize;
        let height = first_image.height as usize;
        let depth = images.len();

        // Allocate 3D array
        let total_voxels = width * height * depth;
        let mut data = vec![0u16; total_voxels];

        // Build original slice positions from ImagePositionPatient
        // Use the appropriate coordinate based on acquisition orientation
        // This must match how series_utils.rs computes slice_location for sync to work
        let acquisition_orientation = AcquisitionOrientation::from_normal(&normal);
        let original_slice_positions: Vec<f64> = slice_indices
            .iter()
            .map(|(idx, _)| {
                let pos = &images[*idx].image_plane.as_ref().unwrap().position;
                // Use the coordinate that matches the through-plane direction
                match acquisition_orientation {
                    AcquisitionOrientation::Axial | AcquisitionOrientation::Unknown => pos.z,
                    AcquisitionOrientation::Coronal => pos.y,
                    AcquisitionOrientation::Sagittal => pos.x,
                }
            })
            .collect();

        // Copy slices into volume in sorted order
        for (z, (original_idx, _)) in slice_indices.iter().enumerate() {
            let img = &images[*original_idx];
            let slice_start = z * width * height;

            // Copy row by row (images may have different strides)
            for y in 0..height {
                for x in 0..width {
                    let src_idx = y * width + x;
                    let dst_idx = slice_start + y * width + x;
                    if src_idx < img.pixels.len() {
                        data[dst_idx] = img.pixels[src_idx];
                    }
                }
            }
        }

        // Get origin from first sorted slice
        let first_sorted_idx = slice_indices[0].0;
        let origin_plane = images[first_sorted_idx].image_plane.as_ref()?;

        let pixel_spacing = first_image.pixel_spacing.unwrap_or((1.0, 1.0));

        // Detect acquisition orientation from the slice normal
        let acquisition_orientation = AcquisitionOrientation::from_normal(&normal);

        tracing::info!(
            "Volume created: {}x{}x{} voxels (WxHxD), spacing: {:.3}x{:.3}x{:.3} mm (col/row/slice), acquisition: {:?}",
            width, height, depth,
            pixel_spacing.1, pixel_spacing.0, slice_spacing,
            acquisition_orientation
        );
        tracing::debug!(
            "Volume orientation: row={:.2},{:.2},{:.2} col={:.2},{:.2},{:.2} normal={:.2},{:.2},{:.2}",
            origin_plane.row_direction.x, origin_plane.row_direction.y, origin_plane.row_direction.z,
            origin_plane.col_direction.x, origin_plane.col_direction.y, origin_plane.col_direction.z,
            normal.x, normal.y, normal.z
        );

        // Debug: log first few slice positions
        for (i, (idx, pos)) in slice_indices.iter().take(3).enumerate() {
            tracing::debug!("Volume slice {}: orig_idx={}, pos={:.2}mm", i, idx, pos);
        }

        // Debug: log volume geometry
        tracing::info!(
            "Volume::from_series: origin=({:.1},{:.1},{:.1}), spacing=({:.3},{:.3},{:.3})",
            origin_plane.position.x,
            origin_plane.position.y,
            origin_plane.position.z,
            pixel_spacing.1,
            pixel_spacing.0,
            slice_spacing
        );
        tracing::info!(
            "Volume::from_series: row_dir=({:.3},{:.3},{:.3}), col_dir=({:.3},{:.3},{:.3}), slice_dir=({:.3},{:.3},{:.3})",
            origin_plane.row_direction.x, origin_plane.row_direction.y, origin_plane.row_direction.z,
            origin_plane.col_direction.x, origin_plane.col_direction.y, origin_plane.col_direction.z,
            normal.x, normal.y, normal.z
        );

        Some(Self {
            data,
            dimensions: (width, height, depth),
            spacing: (pixel_spacing.1, pixel_spacing.0, slice_spacing),
            origin: origin_plane.position,
            row_direction: origin_plane.row_direction,
            col_direction: origin_plane.col_direction,
            slice_direction: normal,
            rescale_slope: first_image.rescale_slope,
            rescale_intercept: first_image.rescale_intercept,
            modality: first_image.modality.clone(),
            series_description: first_image.series_description.clone(),
            acquisition_orientation,
            frame_of_reference_uid: origin_plane.frame_of_reference_uid.clone(),
            study_instance_uid: first_image.study_instance_uid.clone(),
            default_window_center: first_image.window_center,
            default_window_width: first_image.window_width,
            // Patient/Study metadata for MPR image display
            patient_name: first_image.patient_name.clone(),
            patient_id: first_image.patient_id.clone(),
            patient_age: first_image.patient_age.clone(),
            patient_sex: first_image.patient_sex.clone(),
            study_date: first_image.study_date.clone(),
            study_description: first_image.study_description.clone(),
            original_slice_positions,
            pixel_representation: first_image.pixel_representation,
        })
    }

    /// Get voxel value at integer coordinates (bounds checked)
    #[inline]
    #[allow(dead_code)]
    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> u16 {
        let (w, h, d) = self.dimensions;
        if x < w && y < h && z < d {
            self.data[z * w * h + y * w + x]
        } else {
            0
        }
    }

    /// Get which volume direction most closely aligns with a patient axis
    /// Returns (direction_vector, spacing, sign) where sign is +1 or -1
    /// indicating if the direction needs to be negated to match positive axis direction
    pub fn get_axis_direction(&self, axis: PatientAxis) -> (Vec3, f64, f64) {
        let (sx, sy, sz) = self.spacing;

        // For each volume direction, check which patient axis it most aligns with
        let dirs = [
            (&self.row_direction, sx, "row"),
            (&self.col_direction, sy, "col"),
            (&self.slice_direction, sz, "slice"),
        ];

        // Find which direction has the largest component along the requested axis
        let mut best_dir = self.row_direction;
        let mut best_spacing = sx;
        let mut best_magnitude = 0.0;
        let mut best_sign = 1.0;

        for (dir, spacing, _name) in dirs {
            let component = match axis {
                PatientAxis::X => dir.x,
                PatientAxis::Y => dir.y,
                PatientAxis::Z => dir.z,
            };
            let magnitude = component.abs();
            if magnitude > best_magnitude {
                best_magnitude = magnitude;
                best_dir = *dir;
                best_spacing = spacing;
                best_sign = component.signum();
            }
        }

        (best_dir, best_spacing, best_sign)
    }

    /// Get the volume extent (min, max) along a patient axis in mm
    #[allow(dead_code)]
    pub fn get_axis_extent(&self, axis: PatientAxis) -> (f64, f64) {
        let (dir, spacing, _sign) = self.get_axis_direction(axis);
        let (w, h, d) = self.dimensions;

        // Calculate which dimension corresponds to this direction
        let count = if (dir.x.abs() - self.row_direction.x.abs()).abs() < 0.01
            && (dir.y.abs() - self.row_direction.y.abs()).abs() < 0.01
        {
            w
        } else if (dir.x.abs() - self.col_direction.x.abs()).abs() < 0.01
            && (dir.y.abs() - self.col_direction.y.abs()).abs() < 0.01
        {
            h
        } else {
            d
        };

        let origin_component = match axis {
            PatientAxis::X => self.origin.x,
            PatientAxis::Y => self.origin.y,
            PatientAxis::Z => self.origin.z,
        };

        let _extent = count as f64 * spacing;

        // Determine min/max based on direction
        let end_component = origin_component
            + dir.scale(count as f64 * spacing).dot(&match axis {
                PatientAxis::X => Vec3::new(1.0, 0.0, 0.0),
                PatientAxis::Y => Vec3::new(0.0, 1.0, 0.0),
                PatientAxis::Z => Vec3::new(0.0, 0.0, 1.0),
            });

        let min = origin_component.min(end_component);
        let max = origin_component.max(end_component);

        (min, max)
    }

    /// Calculate the actual patient coordinate for an MPR slice position
    /// This is needed for sync to work correctly between series
    /// Returns the position along the normal direction (Z for Axial, Y for Coronal, X for Sagittal)
    pub fn calculate_slice_position(&self, plane: AnatomicalPlane, slice_index: usize) -> f64 {
        // Get the direction that corresponds to the slice normal for this plane
        let (normal_axis, _offset_axis) = match plane {
            AnatomicalPlane::Axial | AnatomicalPlane::Original => (PatientAxis::Z, PatientAxis::Z),
            AnatomicalPlane::Coronal => (PatientAxis::Y, PatientAxis::Y),
            AnatomicalPlane::Sagittal => (PatientAxis::X, PatientAxis::X),
        };

        // Get the volume direction that aligns with this patient axis
        let (axis_dir, axis_spacing, _sign) = self.get_axis_direction(normal_axis);

        // Calculate the position component along the normal direction
        // The slice position is: origin_component + slice_index * spacing * direction_component
        let origin_component = match normal_axis {
            PatientAxis::X => self.origin.x,
            PatientAxis::Y => self.origin.y,
            PatientAxis::Z => self.origin.z,
        };

        let dir_component = match normal_axis {
            PatientAxis::X => axis_dir.x,
            PatientAxis::Y => axis_dir.y,
            PatientAxis::Z => axis_dir.z,
        };

        origin_component + slice_index as f64 * axis_spacing * dir_component
    }

    /// Trilinear interpolation for smooth sampling
    #[inline]
    #[allow(dead_code)]
    fn sample_trilinear(&self, x: f64, y: f64, z: f64) -> u16 {
        let (w, h, d) = self.dimensions;

        // Clamp to valid range
        let x = x.clamp(0.0, (w - 1) as f64);
        let y = y.clamp(0.0, (h - 1) as f64);
        let z = z.clamp(0.0, (d - 1) as f64);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;

        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let z1 = (z0 + 1).min(d - 1);

        let xd = x - x0 as f64;
        let yd = y - y0 as f64;
        let zd = z - z0 as f64;

        // Get 8 corner values
        let c000 = self.get_voxel(x0, y0, z0) as f64;
        let c001 = self.get_voxel(x0, y0, z1) as f64;
        let c010 = self.get_voxel(x0, y1, z0) as f64;
        let c011 = self.get_voxel(x0, y1, z1) as f64;
        let c100 = self.get_voxel(x1, y0, z0) as f64;
        let c101 = self.get_voxel(x1, y0, z1) as f64;
        let c110 = self.get_voxel(x1, y1, z0) as f64;
        let c111 = self.get_voxel(x1, y1, z1) as f64;

        // Interpolate along x
        let c00 = c000 * (1.0 - xd) + c100 * xd;
        let c01 = c001 * (1.0 - xd) + c101 * xd;
        let c10 = c010 * (1.0 - xd) + c110 * xd;
        let c11 = c011 * (1.0 - xd) + c111 * xd;

        // Interpolate along y
        let c0 = c00 * (1.0 - yd) + c10 * yd;
        let c1 = c01 * (1.0 - yd) + c11 * yd;

        // Interpolate along z
        let c = c0 * (1.0 - zd) + c1 * zd;

        c.round() as u16
    }

    /// Get volume extent in mm for each dimension
    #[allow(dead_code)]
    pub fn extent_mm(&self) -> (f64, f64, f64) {
        let (w, h, d) = self.dimensions;
        let (sx, sy, sz) = self.spacing;
        (w as f64 * sx, h as f64 * sy, d as f64 * sz)
    }

    /// Get number of slices available for a given anatomical plane
    /// Accounts for acquisition orientation
    pub fn slice_count(&self, plane: AnatomicalPlane) -> usize {
        let (w, h, d) = self.dimensions;

        // Original always returns native slices
        if plane == AnatomicalPlane::Original {
            return d;
        }

        // Map requested plane to actual dimension based on acquisition
        match self.acquisition_orientation {
            AcquisitionOrientation::Axial => {
                // Native = Axial, so: Axial=depth, Coronal=rows, Sagittal=cols
                match plane {
                    AnatomicalPlane::Axial => d,
                    AnatomicalPlane::Coronal => h,
                    AnatomicalPlane::Sagittal => w,
                    AnatomicalPlane::Original => d,
                }
            }
            AcquisitionOrientation::Sagittal => {
                // Native = Sagittal, so: Sagittal=depth
                // For sagittal: rows=Z direction (col_direction has largest Z), cols=Y direction
                // Axial (constant Z) = iterate along rows = h slices
                // Coronal (constant Y) = iterate along cols = w slices
                match plane {
                    AnatomicalPlane::Sagittal => d,
                    AnatomicalPlane::Axial => h,   // Z varies along rows
                    AnatomicalPlane::Coronal => w, // Y varies along columns
                    AnatomicalPlane::Original => d,
                }
            }
            AcquisitionOrientation::Coronal => {
                // Native = Coronal, so: Coronal=depth, Axial=rows, Sagittal=cols
                match plane {
                    AnatomicalPlane::Coronal => d,
                    AnatomicalPlane::Axial => h,
                    AnatomicalPlane::Sagittal => w,
                    AnatomicalPlane::Original => d,
                }
            }
            AcquisitionOrientation::Unknown => {
                // Fallback to axial-like behavior
                match plane {
                    AnatomicalPlane::Axial => d,
                    AnatomicalPlane::Coronal => h,
                    AnatomicalPlane::Sagittal => w,
                    AnatomicalPlane::Original => d,
                }
            }
        }
    }

    /// Determine which reslice operation to use for a target anatomical plane
    /// based on the original acquisition orientation
    fn get_reslice_operation(&self, target_plane: AnatomicalPlane) -> ResliceOperation {
        // Use type aliases to avoid ambiguity between enum variants with same names
        type Acq = AcquisitionOrientation;
        type Plane = AnatomicalPlane;

        match (self.acquisition_orientation, target_plane) {
            // Axial acquisition
            (Acq::Axial, Plane::Axial) => ResliceOperation::Native,
            (Acq::Axial, Plane::Coronal) => ResliceOperation::AlongRows, // X-reslice
            (Acq::Axial, Plane::Sagittal) => ResliceOperation::AlongColumns, // Y-reslice

            // Sagittal acquisition
            // For sagittal: col_direction is Z (superior-inferior), row_direction is Y (anterior-posterior)
            // Axial slices (constant Z) -> iterate along rows = AlongRows
            // Coronal slices (constant Y) -> iterate along cols = AlongColumns
            (Acq::Sagittal, Plane::Sagittal) => ResliceOperation::Native,
            (Acq::Sagittal, Plane::Axial) => ResliceOperation::AlongRows, // Z varies along rows
            (Acq::Sagittal, Plane::Coronal) => ResliceOperation::AlongColumns, // Y varies along columns

            // Coronal acquisition
            (Acq::Coronal, Plane::Coronal) => ResliceOperation::Native,
            (Acq::Coronal, Plane::Axial) => ResliceOperation::AlongRows,
            (Acq::Coronal, Plane::Sagittal) => ResliceOperation::AlongColumns,

            // Unknown - treat like axial
            (Acq::Unknown, Plane::Axial) => ResliceOperation::Native,
            (Acq::Unknown, Plane::Coronal) => ResliceOperation::AlongRows,
            (Acq::Unknown, Plane::Sagittal) => ResliceOperation::AlongColumns,

            // Original always native
            (_, Plane::Original) => ResliceOperation::Native,
        }
    }

    /// Generate a reformatted slice at given index for the specified anatomical plane
    /// Uses direct pixel access (like Horos/Ukko) for reliability
    /// Properly handles different acquisition orientations
    pub fn resample(&self, plane: AnatomicalPlane, slice_index: usize) -> Option<ReformattedSlice> {
        let (w, h, d) = self.dimensions;
        let (sx, sy, sz) = self.spacing;

        // For Original, always return native slices
        if plane == AnatomicalPlane::Original {
            if slice_index >= d {
                return None;
            }
            let slice_start = slice_index * w * h;
            let pixels: Vec<u16> = self.data[slice_start..slice_start + w * h].to_vec();
            return Some(ReformattedSlice {
                pixels,
                width: w as u32,
                height: h as u32,
                // pixel_spacing: (row_spacing, col_spacing) to match DicomImage convention
                pixel_spacing: (sy, sx),
                slice_position_mm: self.calculate_slice_position(plane, slice_index),
                plane,
            });
        }

        // Determine the reslice operation based on acquisition and target plane
        let reslice_op = self.get_reslice_operation(plane);

        tracing::debug!(
            "MPR resample: plane={:?}, slice_index={}, reslice_op={:?}, dims={}x{}x{}",
            plane,
            slice_index,
            reslice_op,
            w,
            h,
            d
        );

        match reslice_op {
            ResliceOperation::Native => {
                // Return native slice (acquisition matches target)
                if slice_index >= d {
                    return None;
                }
                let slice_start = slice_index * w * h;
                let pixels: Vec<u16> = self.data[slice_start..slice_start + w * h].to_vec();
                tracing::debug!("MPR Native: output {}x{} pixels", w, h);
                Some(ReformattedSlice {
                    pixels,
                    width: w as u32,
                    height: h as u32,
                    // pixel_spacing: (row_spacing, col_spacing) to match DicomImage convention
                    pixel_spacing: (sy, sx),
                    slice_position_mm: self.calculate_slice_position(plane, slice_index),
                    plane,
                })
            }

            ResliceOperation::AlongRows => {
                // Take row from each slice, stack them
                if slice_index >= h {
                    return None;
                }
                let row_idx = slice_index;

                match self.acquisition_orientation {
                    AcquisitionOrientation::Sagittal => {
                        // Sagittal -> Axial: output should have width=d (X), height=w (Y)
                        // For sagittal: columns=Y, rows=Z, slices=X
                        // For axial output: horizontal=X (left-right), vertical=Y (anterior at top)
                        let out_width = d;
                        let out_height = w;
                        let mut pixels = Vec::with_capacity(out_width * out_height);

                        // Reverse X if slice_direction.x > 0 (depth increases in +X),
                        // so that patient-left appears on screen-right (radiological convention)
                        let (_, _, x_sign) = self.get_axis_direction(PatientAxis::X);
                        let x_iter: Vec<usize> = if x_sign > 0.0 {
                            (0..d).rev().collect()
                        } else {
                            (0..d).collect()
                        };
                        for col_y in 0..w {
                            for &slice_x in &x_iter {
                                let idx = slice_x * w * h + row_idx * w + col_y;
                                pixels.push(self.data[idx]);
                            }
                        }

                        tracing::debug!(
                            "MPR AlongRows (sagittal->axial): row={}, output {}x{}",
                            row_idx,
                            out_width,
                            out_height
                        );
                        Some(ReformattedSlice {
                            pixels,
                            width: out_width as u32,
                            height: out_height as u32,
                            pixel_spacing: (sx, sz),
                            slice_position_mm: self.calculate_slice_position(plane, slice_index),
                            plane,
                        })
                    }
                    AcquisitionOrientation::Coronal => {
                        // Coronal -> Axial: output width=w (X), height=d (Y)
                        // For coronal: columns=X, rows=Z, slices=Y
                        // For axial output: horizontal=X, vertical=Y (anterior at top)
                        let out_width = w;
                        let out_height = d;
                        let mut pixels = Vec::with_capacity(out_width * out_height);

                        // Y comes from slice index, need to flip for anterior at top
                        for slice_y in 0..d {
                            let slice_start = slice_y * w * h;
                            let row_start = slice_start + row_idx * w;
                            for x in 0..w {
                                pixels.push(self.data[row_start + x]);
                            }
                        }

                        tracing::debug!(
                            "MPR AlongRows (coronal->axial): row={}, output {}x{}",
                            row_idx,
                            out_width,
                            out_height
                        );
                        Some(ReformattedSlice {
                            pixels,
                            width: out_width as u32,
                            height: out_height as u32,
                            pixel_spacing: (sz, sx),
                            slice_position_mm: self.calculate_slice_position(plane, slice_index),
                            plane,
                        })
                    }
                    _ => {
                        // Axial acquisition -> Coronal view (standard case)
                        let out_width = w;
                        let out_height = d;
                        let mut pixels = Vec::with_capacity(out_width * out_height);

                        // Reverse Z if slice_direction.z > 0 (depth increases in +Z = superior),
                        // so that superior appears at pixel row 0 (top of image)
                        let (_, _, z_sign) = self.get_axis_direction(PatientAxis::Z);
                        let z_iter: Vec<usize> = if z_sign > 0.0 {
                            (0..d).rev().collect()
                        } else {
                            (0..d).collect()
                        };
                        for &z in &z_iter {
                            let slice_start = z * w * h;
                            let row_start = slice_start + row_idx * w;
                            for x in 0..w {
                                pixels.push(self.data[row_start + x]);
                            }
                        }

                        tracing::debug!(
                            "MPR AlongRows (axial->coronal): row={}, output {}x{}",
                            row_idx,
                            out_width,
                            out_height
                        );
                        Some(ReformattedSlice {
                            pixels,
                            width: out_width as u32,
                            height: out_height as u32,
                            pixel_spacing: (sz, sx),
                            slice_position_mm: self.calculate_slice_position(plane, slice_index),
                            plane,
                        })
                    }
                }
            }

            ResliceOperation::AlongColumns => {
                // Take column from each slice, stack them
                if slice_index >= w {
                    return None;
                }
                let col_idx = slice_index;

                match self.acquisition_orientation {
                    AcquisitionOrientation::Sagittal => {
                        // Sagittal -> Coronal: output should have width=d (X), height=h (Z)
                        // For sagittal: columns=Y, rows=Z, slices=X
                        // For coronal output: horizontal=X (left-right), vertical=Z (superior at top)
                        let out_width = d;
                        let out_height = h;
                        let mut pixels = Vec::with_capacity(out_width * out_height);

                        let (_, _, x_sign) = self.get_axis_direction(PatientAxis::X);
                        let x_iter: Vec<usize> = if x_sign > 0.0 {
                            (0..d).rev().collect()
                        } else {
                            (0..d).collect()
                        };
                        for row_z in 0..h {
                            for &slice_x in &x_iter {
                                let idx = slice_x * w * h + row_z * w + col_idx;
                                pixels.push(self.data[idx]);
                            }
                        }

                        tracing::debug!(
                            "MPR AlongColumns (sagittal->coronal): col={}, output {}x{}",
                            col_idx,
                            out_width,
                            out_height
                        );
                        Some(ReformattedSlice {
                            pixels,
                            width: out_width as u32,
                            height: out_height as u32,
                            pixel_spacing: (sy, sz),
                            slice_position_mm: self.calculate_slice_position(plane, slice_index),
                            plane,
                        })
                    }
                    AcquisitionOrientation::Coronal => {
                        // Coronal -> Sagittal: output width=d (Y), height=h (Z)
                        // For coronal: columns=X, rows=Z, slices=Y
                        // For sagittal output: horizontal=Y (ant-post), vertical=Z (superior at top)
                        let out_width = d;
                        let out_height = h;
                        let mut pixels = Vec::with_capacity(out_width * out_height);

                        // Z from rows, Y from slices
                        for row_z in 0..h {
                            for slice_y in 0..d {
                                let idx = slice_y * w * h + row_z * w + col_idx;
                                pixels.push(self.data[idx]);
                            }
                        }

                        tracing::debug!(
                            "MPR AlongColumns (coronal->sagittal): col={}, output {}x{}",
                            col_idx,
                            out_width,
                            out_height
                        );
                        Some(ReformattedSlice {
                            pixels,
                            width: out_width as u32,
                            height: out_height as u32,
                            // pixel_spacing = (row_spacing, col_spacing) = (Z_spacing, Y_spacing)
                            pixel_spacing: (sy, sz),
                            slice_position_mm: self.calculate_slice_position(plane, slice_index),
                            plane,
                        })
                    }
                    _ => {
                        // Axial acquisition -> Sagittal view (standard case)
                        let out_width = h;
                        let out_height = d;
                        let mut pixels = Vec::with_capacity(out_width * out_height);

                        // Reverse Z if slice_direction.z > 0, so superior at top
                        let (_, _, z_sign) = self.get_axis_direction(PatientAxis::Z);
                        let z_iter: Vec<usize> = if z_sign > 0.0 {
                            (0..d).rev().collect()
                        } else {
                            (0..d).collect()
                        };
                        for &z in &z_iter {
                            let slice_start = z * w * h;
                            for y in 0..h {
                                let idx = slice_start + y * w + col_idx;
                                pixels.push(self.data[idx]);
                            }
                        }

                        tracing::debug!(
                            "MPR AlongColumns (axial->sagittal): col={}, output {}x{}",
                            col_idx,
                            out_width,
                            out_height
                        );
                        Some(ReformattedSlice {
                            pixels,
                            width: out_width as u32,
                            height: out_height as u32,
                            pixel_spacing: (sz, sy),
                            slice_position_mm: self.calculate_slice_position(plane, slice_index),
                            plane,
                        })
                    }
                }
            }
        }
    }

    /// Find slice index closest to given position in mm
    #[allow(dead_code)]
    pub fn position_to_slice(&self, plane: AnatomicalPlane, position_mm: f64) -> usize {
        let count = self.slice_count(plane);
        let spacing = match plane {
            AnatomicalPlane::Axial | AnatomicalPlane::Original => self.spacing.2,
            AnatomicalPlane::Coronal => self.spacing.1,
            AnatomicalPlane::Sagittal => self.spacing.0,
        };

        let index = (position_mm / spacing).round() as usize;
        index.min(count.saturating_sub(1))
    }

    /// Get position in mm for given slice index
    pub fn slice_to_position(&self, plane: AnatomicalPlane, slice_index: usize) -> f64 {
        let spacing = match plane {
            AnatomicalPlane::Axial | AnatomicalPlane::Original => self.spacing.2,
            AnatomicalPlane::Coronal => self.spacing.1,
            AnatomicalPlane::Sagittal => self.spacing.0,
        };
        slice_index as f64 * spacing
    }
}

/// A reformatted slice from MPR
#[derive(Clone)]
pub struct ReformattedSlice {
    /// Pixel data
    pub pixels: Vec<u16>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel spacing (row_spacing, col_spacing) in mm - matches DicomImage convention
    pub pixel_spacing: (f64, f64),
    /// Slice position along the normal direction (mm)
    pub slice_position_mm: f64,
    /// Which plane this slice represents
    pub plane: AnatomicalPlane,
}

impl ReformattedSlice {
    /// Compute ImagePlane geometry for this MPR slice
    /// This enables reference line calculations between MPR and original images
    ///
    /// The geometry is derived from the volume's actual coordinate system, mapping
    /// patient coordinate axes to the appropriate volume directions regardless of
    /// acquisition orientation.
    ///
    /// The ImagePlane position is at the top-left corner of the slice, with:
    /// - row_direction pointing to the right edge
    /// - col_direction pointing to the bottom edge
    pub fn compute_image_plane(&self, volume: &Volume, slice_index: usize) -> ImagePlane {
        // Get the volume direction that corresponds to each patient axis
        let (x_dir, x_spacing, _x_sign) = volume.get_axis_direction(PatientAxis::X);
        let (y_dir, y_spacing, _y_sign) = volume.get_axis_direction(PatientAxis::Y);
        let (z_dir, z_spacing, z_sign) = volume.get_axis_direction(PatientAxis::Z);

        // Calculate the volume's extent to find corner positions
        // For each plane, we need to position the slice at the correct corner
        //
        // The resample always puts superior at pixel row 0. The ImagePlane must match:
        // - position at the superior end of the Z range (top-left corner)
        // - col_direction pointing inferior (pixel y increases going down)
        //
        // When z_sign > 0: z increases toward superior, so the superior end is at
        //   origin + z_dir * z_extent. col_dir = z_dir.negate() (toward inferior).
        // When z_sign < 0: z decreases toward superior, so origin is already at
        //   the superior end. col_dir = z_dir (which points inferior since z_dir.z < 0).
        let (row_dir, col_dir, normal, position) = match self.plane {
            AnatomicalPlane::Axial | AnatomicalPlane::Original => {
                // Axial: slices perpendicular to Z axis
                // Row direction: along X (left-right)
                // Col direction: along Y (anterior-posterior)
                // Normal: along Z (superior-inferior)
                // Position: origin + slice_offset_along_Z
                let offset = z_dir.scale(slice_index as f64 * z_spacing);
                let pos = volume.origin.add(&offset);
                (x_dir, y_dir, z_dir, pos)
            }
            AnatomicalPlane::Coronal => {
                // Coronal: slices perpendicular to Y axis
                // Row direction: along X (left-right)
                // Col direction: toward inferior (superior at top)
                // Normal: along Y (anterior-posterior)
                let slice_offset = y_dir.scale(slice_index as f64 * y_spacing);

                let (_w, h, d) = volume.dimensions;
                let z_extent = match volume.acquisition_orientation {
                    AcquisitionOrientation::Axial => d as f64 * volume.spacing.2,
                    AcquisitionOrientation::Coronal => h as f64 * volume.spacing.1,
                    AcquisitionOrientation::Sagittal => h as f64 * volume.spacing.1,
                    AcquisitionOrientation::Unknown => d as f64 * volume.spacing.2,
                };

                if z_sign > 0.0 {
                    // z increases toward superior: offset to top, col goes down
                    let z_offset = z_dir.scale(z_extent);
                    let pos = volume.origin.add(&slice_offset).add(&z_offset);
                    (x_dir, z_dir.negate(), y_dir, pos)
                } else {
                    // z decreases toward superior: origin is at top, col = z_dir (goes inferior)
                    let pos = volume.origin.add(&slice_offset);
                    (x_dir, z_dir, y_dir, pos)
                }
            }
            AnatomicalPlane::Sagittal => {
                // Sagittal: slices perpendicular to X axis
                // Row direction: along Y (anterior-posterior)
                // Col direction: toward inferior (superior at top)
                // Normal: along X (left-right)
                let slice_offset = x_dir.scale(slice_index as f64 * x_spacing);

                let (_w, h, d) = volume.dimensions;
                let z_extent = match volume.acquisition_orientation {
                    AcquisitionOrientation::Axial => d as f64 * volume.spacing.2,
                    AcquisitionOrientation::Coronal => h as f64 * volume.spacing.1,
                    AcquisitionOrientation::Sagittal => h as f64 * volume.spacing.1,
                    AcquisitionOrientation::Unknown => d as f64 * volume.spacing.2,
                };

                if z_sign > 0.0 {
                    let z_offset = z_dir.scale(z_extent);
                    let pos = volume.origin.add(&slice_offset).add(&z_offset);
                    (y_dir, z_dir.negate(), x_dir, pos)
                } else {
                    let pos = volume.origin.add(&slice_offset);
                    (y_dir, z_dir, x_dir, pos)
                }
            }
        };

        // Debug: log geometry for first and middle slice
        if slice_index == 0 || slice_index == 100 {
            let (sx, sy, sz) = volume.spacing;
            tracing::info!(
                "MPR ImagePlane {:?} slice {}: pos=({:.1},{:.1},{:.1})",
                self.plane,
                slice_index,
                position.x,
                position.y,
                position.z
            );
            tracing::info!(
                "  Volume: origin=({:.1},{:.1},{:.1}), spacing=({:.3},{:.3},{:.3})",
                volume.origin.x,
                volume.origin.y,
                volume.origin.z,
                sx,
                sy,
                sz
            );
            tracing::info!(
                "  Axis directions: X=({:.3},{:.3},{:.3}), Y=({:.3},{:.3},{:.3}), Z=({:.3},{:.3},{:.3})",
                x_dir.x, x_dir.y, x_dir.z,
                y_dir.x, y_dir.y, y_dir.z,
                z_dir.x, z_dir.y, z_dir.z
            );
            tracing::info!(
                "  Computed: row=({:.3},{:.3},{:.3}), col=({:.3},{:.3},{:.3}), normal=({:.3},{:.3},{:.3})",
                row_dir.x, row_dir.y, row_dir.z,
                col_dir.x, col_dir.y, col_dir.z,
                normal.x, normal.y, normal.z
            );
        }

        ImagePlane {
            position,
            row_direction: row_dir,
            col_direction: col_dir,
            normal,
            pixel_spacing: self.pixel_spacing,
            dimensions: (self.height, self.width),
            frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
        }
    }

    /// Get slice location for sync purposes (distance along normal from origin)
    #[allow(dead_code)]
    pub fn slice_location(&self) -> f64 {
        self.slice_position_mm
    }

    /// Convert to DicomImage for display (reuses existing rendering pipeline)
    pub fn to_dicom_image(&self, volume: &Volume, slice_index: usize) -> DicomImage {
        let image_plane = self.compute_image_plane(volume, slice_index);

        // Compute slice location at IMAGE CENTER for proper sync with regular series
        // The center position better represents the anatomical level for tilted MPR planes
        let (row_sp, col_sp) = self.pixel_spacing;
        let half_w = self.width as f64 / 2.0;
        let half_h = self.height as f64 / 2.0;
        let center_offset_h = image_plane.row_direction.scale(half_w * col_sp);
        let center_offset_v = image_plane.col_direction.scale(half_h * row_sp);
        let center_pos = image_plane
            .position
            .add(&center_offset_h)
            .add(&center_offset_v);

        let slice_location = match self.plane {
            AnatomicalPlane::Axial | AnatomicalPlane::Original => center_pos.z,
            AnatomicalPlane::Coronal => center_pos.y,
            AnatomicalPlane::Sagittal => center_pos.x,
        };

        DicomImage {
            pixels: self.pixels.clone(),
            width: self.width,
            height: self.height,
            pixel_spacing: Some(self.pixel_spacing),
            rescale_slope: volume.rescale_slope,
            rescale_intercept: volume.rescale_intercept,
            window_center: volume.default_window_center,
            window_width: volume.default_window_width,
            modality: volume.modality.clone(),
            min_value: *self.pixels.iter().min().unwrap_or(&0) as f64,
            max_value: *self.pixels.iter().max().unwrap_or(&65535) as f64,
            invert: false,
            image_plane: Some(image_plane),
            sop_instance_uid: None,
            study_instance_uid: volume.study_instance_uid.clone(),
            // Patient/Study metadata from volume
            patient_name: volume.patient_name.clone(),
            patient_id: volume.patient_id.clone(),
            patient_age: volume.patient_age.clone(),
            patient_sex: volume.patient_sex.clone(),
            study_date: volume.study_date.clone(),
            study_description: volume.study_description.clone(),
            series_description: volume.series_description.clone(),
            slice_thickness: None,
            magnetic_field_strength: None,
            repetition_time: None,
            echo_time: None,
            sequence_name: None,
            kvp: None,
            exposure: None,
            slice_location: Some(slice_location),
            pixel_representation: volume.pixel_representation,
        }
    }
}

/// Pre-generated MPR series for GPU caching
pub struct MprSeries {
    /// Plane this series represents
    pub plane: AnatomicalPlane,
    /// Pre-generated DicomImages for each slice (Arc to avoid cloning pixel data)
    pub images: Vec<Arc<DicomImage>>,
    /// Synthetic paths for GPU cache (e.g., "mpr://coronal/0", "mpr://coronal/1")
    pub paths: Vec<std::path::PathBuf>,
    /// Sync info for this MPR series (enables sync with other viewports)
    pub sync_info: SyncInfo,
}

impl MprSeries {
    /// Generate a complete MPR series for a plane
    pub fn generate(volume: &Volume, plane: AnatomicalPlane) -> Option<Self> {
        let slice_count = volume.slice_count(plane);
        if slice_count == 0 {
            return None;
        }

        let mut images = Vec::with_capacity(slice_count);
        let mut paths = Vec::with_capacity(slice_count);
        let mut slice_locations = Vec::with_capacity(slice_count);

        let plane_name = match plane {
            AnatomicalPlane::Axial => "axial",
            AnatomicalPlane::Coronal => "coronal",
            AnatomicalPlane::Sagittal => "sagittal",
            AnatomicalPlane::Original => "original",
        };

        // Map plane to orientation for sync matching
        let orientation: Option<AnatomicalPlane> = match plane {
            AnatomicalPlane::Original => None, // Could detect from volume
            other => Some(other),
        };

        tracing::info!(
            "Generating {} MPR slices for {} plane",
            slice_count,
            plane_name
        );

        // Get spacing for each axis
        let (_sx, _sy, _sz) = volume.spacing;

        // Get axis directions for computing slice positions in patient coordinates
        let (_z_dir, _z_spacing, _) = volume.get_axis_direction(PatientAxis::Z);
        let (_y_dir, _y_spacing, _) = volume.get_axis_direction(PatientAxis::Y);
        let (_x_dir, _x_spacing, _) = volume.get_axis_direction(PatientAxis::X);

        for i in 0..slice_count {
            if let Some(slice) = volume.resample(plane, i) {
                // Compute slice location at IMAGE CENTER for proper sync
                // Regular series slice_location is at corner, but for tilted MPR planes
                // the center position better represents the anatomical level
                let img_plane = slice.compute_image_plane(volume, i);

                let (row_sp, col_sp) = slice.pixel_spacing;
                let half_w = slice.width as f64 / 2.0;
                let half_h = slice.height as f64 / 2.0;
                let center_offset_h = img_plane.row_direction.scale(half_w * col_sp);
                let center_offset_v = img_plane.col_direction.scale(half_h * row_sp);
                let center_pos = img_plane
                    .position
                    .add(&center_offset_h)
                    .add(&center_offset_v);

                let slice_loc = match plane {
                    AnatomicalPlane::Axial | AnatomicalPlane::Original => center_pos.z,
                    AnatomicalPlane::Coronal => center_pos.y,
                    AnatomicalPlane::Sagittal => center_pos.x,
                };
                slice_locations.push(Some(slice_loc));

                // Convert to DicomImage with geometry
                let dicom_image = slice.to_dicom_image(volume, i);

                // Debug: verify sync_info slice_loc matches DicomImage slice_location
                if i == 0 || i == 170 || i == slice_count / 2 {
                    tracing::info!(
                        "MprSeries slice {}: sync_loc={:.2}, img_slice_loc={:?}, slice_position_mm={:.2}",
                        i, slice_loc, dicom_image.slice_location, slice.slice_position_mm
                    );
                }

                images.push(Arc::new(dicom_image));
                // Use synthetic path for GPU cache key
                paths.push(std::path::PathBuf::from(format!(
                    "mpr://{}/{}",
                    plane_name, i
                )));
            }
        }

        tracing::info!(
            "Generated {} MPR images for {} plane",
            images.len(),
            plane_name
        );

        // VERIFY: images and slice_locations are aligned
        for (i, (img, loc)) in images.iter().zip(slice_locations.iter()).enumerate() {
            let img_loc = img.slice_location;
            if let (Some(img_z), Some(sync_z)) = (img_loc, loc) {
                let diff = (img_z - sync_z).abs();
                if diff > 0.01 {
                    tracing::error!(
                        "MprSeries INDEX MISMATCH at {}: images[{}].slice_location={:.3} != slice_locations[{}]={:.3}, diff={:.3}mm",
                        i, i, img_z, i, sync_z, diff
                    );
                }
            }
        }

        // Create sync info for this MPR series
        let sync_info = SyncInfo {
            frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
            study_instance_uid: volume.study_instance_uid.clone(),
            orientation,
            slice_locations,
        };

        // Debug: log sync info and patient info
        tracing::info!(
            "MprSeries::generate: sync_info frame_of_reference={:?}, study_instance_uid={:?}, orientation={:?}",
            sync_info.frame_of_reference_uid, sync_info.study_instance_uid, sync_info.orientation
        );
        tracing::info!(
            "MprSeries::generate: volume patient_name={:?}, patient_id={:?}",
            volume.patient_name,
            volume.patient_id
        );
        if let Some(first_img) = images.first() {
            tracing::info!(
                "MprSeries::generate: first_img patient_name={:?}, patient_id={:?}, image_plane={:?}",
                first_img.patient_name, first_img.patient_id, first_img.image_plane.is_some()
            );
        }

        Some(Self {
            plane,
            images,
            paths,
            sync_info,
        })
    }

    /// Get the number of slices
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Check if empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

/// MPR state for a viewport
#[derive(Clone)]
pub struct MprState {
    /// Current viewing plane
    pub plane: AnatomicalPlane,
    /// Current slice index in the reformatted plane
    pub slice_index: usize,
    /// Constructed volume (shared between views)
    pub volume: Option<Arc<Volume>>,
    /// Cached reformatted slice for current view (legacy, used for on-demand generation)
    cached_slice: Option<ReformattedSlice>,
    /// Cache key (plane, slice_index) to detect when to regenerate
    cache_key: Option<(AnatomicalPlane, usize)>,
    /// Pre-generated MPR series for current plane (for GPU caching)
    pub mpr_series: Option<Arc<MprSeries>>,
}

impl Default for MprState {
    fn default() -> Self {
        Self {
            plane: AnatomicalPlane::Original,
            slice_index: 0,
            volume: None,
            cached_slice: None,
            cache_key: None,
            mpr_series: None,
        }
    }
}

impl MprState {
    /// Create new MPR state
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if MPR mode is active (not viewing original)
    pub fn is_active(&self) -> bool {
        self.plane != AnatomicalPlane::Original && self.volume.is_some()
    }

    /// Set the volume for this viewport
    pub fn set_volume(&mut self, volume: Arc<Volume>) {
        self.volume = Some(volume);
        self.invalidate_cache();
    }

    /// Clear volume and reset to original
    pub fn clear(&mut self) {
        self.volume = None;
        self.plane = AnatomicalPlane::Original;
        self.slice_index = 0;
        self.mpr_series = None;
        self.invalidate_cache();
    }

    /// Switch to a different anatomical plane
    pub fn set_plane(&mut self, plane: AnatomicalPlane) {
        if self.plane != plane {
            // Clear any pre-generated series when switching planes
            self.mpr_series = None;

            // When switching planes, try to maintain approximate position
            if let Some(vol) = &self.volume {
                let _old_position = vol.slice_to_position(self.plane, self.slice_index);
                self.plane = plane;
                // Start at center of new plane if position doesn't map well
                let new_count = vol.slice_count(plane);
                self.slice_index = new_count / 2;
            } else {
                self.plane = plane;
                self.slice_index = 0;
            }
            self.invalidate_cache();
        }
    }

    /// Generate and cache the full MPR series for the current plane
    /// Returns the generated series for GPU upload
    #[allow(dead_code)]
    pub fn generate_series(&mut self) -> Option<Arc<MprSeries>> {
        if let Some(vol) = &self.volume {
            if let Some(series) = MprSeries::generate(vol, self.plane) {
                let series_arc = Arc::new(series);
                self.mpr_series = Some(series_arc.clone());
                return Some(series_arc);
            }
        }
        None
    }

    /// Set pre-generated series (from async worker)
    pub fn set_series(&mut self, series: Arc<MprSeries>) {
        self.mpr_series = Some(series);
    }

    /// Check if a pre-generated series is available for the current plane
    pub fn has_series(&self) -> bool {
        self.mpr_series
            .as_ref()
            .map(|s| s.plane == self.plane)
            .unwrap_or(false)
    }

    /// Get image at current slice index from pre-generated series
    #[allow(dead_code)]
    pub fn get_series_image(&self) -> Option<&Arc<DicomImage>> {
        self.mpr_series
            .as_ref()
            .and_then(|s| s.images.get(self.slice_index))
    }

    /// Get path for current slice (for GPU cache lookup)
    pub fn get_series_path(&self) -> Option<&std::path::PathBuf> {
        self.mpr_series
            .as_ref()
            .and_then(|s| s.paths.get(self.slice_index))
    }

    /// Navigate to a specific slice
    #[allow(dead_code)]
    pub fn set_slice(&mut self, index: usize) {
        if let Some(vol) = &self.volume {
            let max = vol.slice_count(self.plane).saturating_sub(1);
            self.slice_index = index.min(max);
            self.invalidate_cache();
        }
    }

    /// Navigate relative to current slice
    pub fn navigate(&mut self, delta: i32) {
        if let Some(vol) = &self.volume {
            let max = vol.slice_count(self.plane).saturating_sub(1);
            let new_index = (self.slice_index as i32 + delta).clamp(0, max as i32) as usize;
            if new_index != self.slice_index {
                self.slice_index = new_index;
                self.invalidate_cache();
            }
        }
    }

    /// Get current slice count for the active plane
    #[allow(dead_code)]
    pub fn slice_count(&self) -> usize {
        self.volume
            .as_ref()
            .map(|v| v.slice_count(self.plane))
            .unwrap_or(0)
    }

    /// Get the current reformatted slice (cached)
    pub fn get_slice(&mut self) -> Option<&ReformattedSlice> {
        let key = (self.plane, self.slice_index);

        // Check if cache is valid
        if self.cache_key == Some(key) && self.cached_slice.is_some() {
            return self.cached_slice.as_ref();
        }

        // Generate new slice
        if let Some(vol) = &self.volume {
            if let Some(slice) = vol.resample(self.plane, self.slice_index) {
                self.cached_slice = Some(slice);
                self.cache_key = Some(key);
                return self.cached_slice.as_ref();
            }
        }

        None
    }

    /// Invalidate the cached slice
    fn invalidate_cache(&mut self) {
        self.cache_key = None;
        self.cached_slice = None;
    }

    /// Get position info string for display
    pub fn position_info(&self) -> String {
        if let Some(vol) = &self.volume {
            let pos = vol.slice_to_position(self.plane, self.slice_index);
            let count = vol.slice_count(self.plane);
            format!(
                "{} {}/{} ({:.1}mm)",
                self.plane,
                self.slice_index + 1,
                count,
                pos
            )
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Helpers: build minimal Volume instances with known geometry
    // ---------------------------------------------------------------

    /// Create a small axial-acquired volume (4x3x5, spacing 1mm isotropic).
    /// Standard DICOM orientation: row_dir = +X, col_dir = +Y, slice_dir = +Z.
    /// Voxel values: v(x,y,z) = z*100 + y*10 + x  (unique per voxel).
    fn make_axial_volume() -> Volume {
        let (w, h, d) = (4, 3, 5);
        let mut data = vec![0u16; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (z * 100 + y * 10 + x) as u16;
                }
            }
        }
        Volume {
            data,
            dimensions: (w, h, d),
            spacing: (1.0, 1.0, 1.0),
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            col_direction: Vec3::new(0.0, 1.0, 0.0),
            slice_direction: Vec3::new(0.0, 0.0, 1.0),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Axial,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: Some("1.2.3.4".into()),
            default_window_center: 500.0,
            default_window_width: 1000.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: (0..5).map(|z| z as f64).collect(),
            pixel_representation: 0,
        }
    }

    /// Create a sagittal-acquired volume (4x3x5).
    /// Sagittal: row_dir = +Y, col_dir = +Z, slice_dir = +X.
    /// dimensions: (cols=4, rows=3, slices=5) map to (Y, Z, X).
    fn make_sagittal_volume() -> Volume {
        let (w, h, d) = (4, 3, 5);
        let mut data = vec![0u16; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (z * 100 + y * 10 + x) as u16;
                }
            }
        }
        Volume {
            data,
            dimensions: (w, h, d),
            spacing: (1.0, 1.0, 1.0),
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_direction: Vec3::new(0.0, 1.0, 0.0),  // columns along Y
            col_direction: Vec3::new(0.0, 0.0, 1.0),  // rows along Z
            slice_direction: Vec3::new(1.0, 0.0, 0.0), // slices along X
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Sagittal,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: Some("1.2.3.4".into()),
            default_window_center: 500.0,
            default_window_width: 1000.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: (0..5).map(|x| x as f64).collect(),
            pixel_representation: 0,
        }
    }

    /// Create a volume with non-unit spacing and non-zero origin.
    fn make_offset_axial_volume() -> Volume {
        let (w, h, d) = (2, 2, 3);
        let mut data = vec![0u16; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (z * 100 + y * 10 + x) as u16;
                }
            }
        }
        Volume {
            data,
            dimensions: (w, h, d),
            spacing: (0.5, 0.5, 2.0),
            origin: Vec3::new(-10.0, -20.0, 30.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            col_direction: Vec3::new(0.0, 1.0, 0.0),
            slice_direction: Vec3::new(0.0, 0.0, 1.0),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Axial,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: None,
            default_window_center: 0.0,
            default_window_width: 100.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: vec![30.0, 32.0, 34.0],
            pixel_representation: 0,
        }
    }

    // ---------------------------------------------------------------
    // AnatomicalPlane tests
    // ---------------------------------------------------------------

    #[test]
    fn test_anatomical_plane_display() {
        assert_eq!(format!("{}", AnatomicalPlane::Axial), "Axial");
        assert_eq!(format!("{}", AnatomicalPlane::Coronal), "Coronal");
        assert_eq!(format!("{}", AnatomicalPlane::Sagittal), "Sagittal");
        assert_eq!(format!("{}", AnatomicalPlane::Original), "Original");
    }

    #[test]
    fn test_anatomical_plane_abbrev_roundtrip() {
        for plane in &[
            AnatomicalPlane::Axial,
            AnatomicalPlane::Coronal,
            AnatomicalPlane::Sagittal,
            AnatomicalPlane::Original,
        ] {
            let abbrev = plane.abbrev();
            let parsed = AnatomicalPlane::from_abbrev(abbrev);
            assert_eq!(parsed, Some(*plane), "roundtrip failed for {:?}", plane);
        }
        assert_eq!(AnatomicalPlane::from_abbrev("garbage"), None);
    }

    // ---------------------------------------------------------------
    // AcquisitionOrientation::from_normal
    // ---------------------------------------------------------------

    #[test]
    fn test_acquisition_orientation_from_normal() {
        // Pure axis-aligned normals
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(0.0, 0.0, 1.0)),
            AcquisitionOrientation::Axial
        );
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(0.0, 0.0, -1.0)),
            AcquisitionOrientation::Axial
        );
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(0.0, 1.0, 0.0)),
            AcquisitionOrientation::Coronal
        );
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(1.0, 0.0, 0.0)),
            AcquisitionOrientation::Sagittal
        );
    }

    #[test]
    fn test_acquisition_orientation_tilted_axial() {
        // 20-degree gantry tilt around X: normal has Z and Y components
        let tilt_rad = 20.0_f64.to_radians();
        let normal = Vec3::new(0.0, tilt_rad.sin(), tilt_rad.cos());
        // cos(20) ~ 0.94 > 0.7 threshold, still biggest component
        assert_eq!(
            AcquisitionOrientation::from_normal(&normal),
            AcquisitionOrientation::Axial
        );
    }

    #[test]
    fn test_acquisition_orientation_oblique_is_unknown() {
        // All components roughly equal: should be Unknown
        let v = 1.0 / 3.0_f64.sqrt();
        let normal = Vec3::new(v, v, v);
        assert_eq!(
            AcquisitionOrientation::from_normal(&normal),
            AcquisitionOrientation::Unknown
        );
    }

    // ---------------------------------------------------------------
    // Volume::get_voxel
    // ---------------------------------------------------------------

    #[test]
    fn test_get_voxel_in_bounds() {
        let vol = make_axial_volume();
        // v(x,y,z) = z*100 + y*10 + x
        assert_eq!(vol.get_voxel(0, 0, 0), 0);
        assert_eq!(vol.get_voxel(3, 2, 4), 423);
        assert_eq!(vol.get_voxel(1, 1, 2), 211);
    }

    #[test]
    fn test_get_voxel_out_of_bounds_returns_zero() {
        let vol = make_axial_volume();
        assert_eq!(vol.get_voxel(4, 0, 0), 0); // x out of bounds
        assert_eq!(vol.get_voxel(0, 3, 0), 0); // y out of bounds
        assert_eq!(vol.get_voxel(0, 0, 5), 0); // z out of bounds
        assert_eq!(vol.get_voxel(100, 100, 100), 0);
    }

    // ---------------------------------------------------------------
    // Volume::sample_trilinear
    // ---------------------------------------------------------------

    #[test]
    fn test_trilinear_at_integer_coords_matches_get_voxel() {
        let vol = make_axial_volume();
        // At integer positions, trilinear should return the exact voxel value
        assert_eq!(vol.sample_trilinear(0.0, 0.0, 0.0), 0);
        assert_eq!(vol.sample_trilinear(3.0, 2.0, 4.0), 423);
        assert_eq!(vol.sample_trilinear(1.0, 1.0, 2.0), 211);
    }

    #[test]
    fn test_trilinear_midpoint_interpolation() {
        let vol = make_axial_volume();
        // Interpolate halfway between (0,0,0)=0 and (1,0,0)=1 along X
        let val = vol.sample_trilinear(0.5, 0.0, 0.0);
        // Should be (0 + 1) / 2 = 0.5 -> rounds to 1 (or 0 depending on rounding)
        assert!(val == 0 || val == 1, "expected 0 or 1, got {}", val);

        // Halfway between (0,0,0)=0 and (0,1,0)=10 along Y
        let val = vol.sample_trilinear(0.0, 0.5, 0.0);
        assert_eq!(val, 5); // (0+10)/2 = 5
    }

    #[test]
    fn test_trilinear_clamping_at_boundaries() {
        let vol = make_axial_volume();
        // Negative coords should clamp to 0
        assert_eq!(vol.sample_trilinear(-1.0, 0.0, 0.0), vol.get_voxel(0, 0, 0));
        // Beyond max should clamp to last voxel
        assert_eq!(
            vol.sample_trilinear(10.0, 10.0, 10.0),
            vol.get_voxel(3, 2, 4)
        );
    }

    // ---------------------------------------------------------------
    // Volume::extent_mm
    // ---------------------------------------------------------------

    #[test]
    fn test_extent_mm() {
        let vol = make_axial_volume();
        assert_eq!(vol.extent_mm(), (4.0, 3.0, 5.0));

        let vol2 = make_offset_axial_volume();
        assert_eq!(vol2.extent_mm(), (1.0, 1.0, 6.0));
    }

    // ---------------------------------------------------------------
    // Volume::get_axis_direction
    // ---------------------------------------------------------------

    #[test]
    fn test_axis_direction_axial_volume() {
        let vol = make_axial_volume();

        let (dir, spacing, sign) = vol.get_axis_direction(PatientAxis::X);
        assert!((dir.x - 1.0).abs() < 1e-9, "X axis should be row_direction");
        assert!((spacing - 1.0).abs() < 1e-9);
        assert!(sign > 0.0);

        let (dir, spacing, sign) = vol.get_axis_direction(PatientAxis::Y);
        assert!((dir.y - 1.0).abs() < 1e-9, "Y axis should be col_direction");
        assert!((spacing - 1.0).abs() < 1e-9);
        assert!(sign > 0.0);

        let (dir, spacing, sign) = vol.get_axis_direction(PatientAxis::Z);
        assert!((dir.z - 1.0).abs() < 1e-9, "Z axis should be slice_direction");
        assert!((spacing - 1.0).abs() < 1e-9);
        assert!(sign > 0.0);
    }

    #[test]
    fn test_axis_direction_sagittal_volume() {
        let vol = make_sagittal_volume();

        // Sagittal: row=Y, col=Z, slice=X
        let (dir, _, sign) = vol.get_axis_direction(PatientAxis::X);
        assert!((dir.x - 1.0).abs() < 1e-9, "X axis should be slice_direction");
        assert!(sign > 0.0);

        let (dir, _, sign) = vol.get_axis_direction(PatientAxis::Y);
        assert!((dir.y - 1.0).abs() < 1e-9, "Y axis should be row_direction");
        assert!(sign > 0.0);

        let (dir, _, sign) = vol.get_axis_direction(PatientAxis::Z);
        assert!((dir.z - 1.0).abs() < 1e-9, "Z axis should be col_direction");
        assert!(sign > 0.0);
    }

    // ---------------------------------------------------------------
    // Volume::get_axis_extent
    // ---------------------------------------------------------------

    #[test]
    fn test_axis_extent_axial_volume() {
        let vol = make_axial_volume();
        let (min_z, max_z) = vol.get_axis_extent(PatientAxis::Z);
        assert!((min_z - 0.0).abs() < 1e-9);
        assert!((max_z - 5.0).abs() < 1e-9);

        let (min_x, max_x) = vol.get_axis_extent(PatientAxis::X);
        assert!((min_x - 0.0).abs() < 1e-9);
        assert!((max_x - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_extent_with_offset_origin() {
        let vol = make_offset_axial_volume();
        // origin.z = 30, slice_dir = +Z, 3 slices * 2.0mm spacing = 6.0
        let (min_z, max_z) = vol.get_axis_extent(PatientAxis::Z);
        assert!((min_z - 30.0).abs() < 1e-9);
        assert!((max_z - 36.0).abs() < 1e-9);

        // origin.x = -10, row_dir = +X, 2 cols * 0.5mm = 1.0
        let (min_x, max_x) = vol.get_axis_extent(PatientAxis::X);
        assert!((min_x - (-10.0)).abs() < 1e-9);
        assert!((max_x - (-9.0)).abs() < 1e-9);
    }

    // ---------------------------------------------------------------
    // Volume::slice_count
    // ---------------------------------------------------------------

    #[test]
    fn test_slice_count_axial_volume() {
        let vol = make_axial_volume(); // 4x3x5
        assert_eq!(vol.slice_count(AnatomicalPlane::Axial), 5);    // depth
        assert_eq!(vol.slice_count(AnatomicalPlane::Coronal), 3);  // rows
        assert_eq!(vol.slice_count(AnatomicalPlane::Sagittal), 4); // cols
        assert_eq!(vol.slice_count(AnatomicalPlane::Original), 5); // depth
    }

    #[test]
    fn test_slice_count_sagittal_volume() {
        let vol = make_sagittal_volume(); // 4x3x5
        // Sagittal native = depth=5, Axial = rows=3, Coronal = cols=4
        assert_eq!(vol.slice_count(AnatomicalPlane::Sagittal), 5);
        assert_eq!(vol.slice_count(AnatomicalPlane::Axial), 3);
        assert_eq!(vol.slice_count(AnatomicalPlane::Coronal), 4);
    }

    // ---------------------------------------------------------------
    // Volume::calculate_slice_position
    // ---------------------------------------------------------------

    #[test]
    fn test_calculate_slice_position_axial() {
        let vol = make_axial_volume();
        // Axial: normal axis is Z, slice_direction = (0,0,1), spacing = 1.0
        // position = origin.z + slice_index * 1.0 * 1.0
        assert!((vol.calculate_slice_position(AnatomicalPlane::Axial, 0) - 0.0).abs() < 1e-9);
        assert!((vol.calculate_slice_position(AnatomicalPlane::Axial, 3) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_calculate_slice_position_with_offset() {
        let vol = make_offset_axial_volume();
        // origin.z = 30.0, slice spacing = 2.0, slice_dir.z = 1.0
        assert!(
            (vol.calculate_slice_position(AnatomicalPlane::Axial, 0) - 30.0).abs() < 1e-9
        );
        assert!(
            (vol.calculate_slice_position(AnatomicalPlane::Axial, 2) - 34.0).abs() < 1e-9
        );
    }

    #[test]
    fn test_calculate_slice_position_coronal_and_sagittal() {
        let vol = make_axial_volume();
        // Coronal: normal axis Y, col_dir = (0,1,0), spacing = 1.0
        // position = origin.y + slice_index * 1.0 * 1.0
        assert!((vol.calculate_slice_position(AnatomicalPlane::Coronal, 2) - 2.0).abs() < 1e-9);
        // Sagittal: normal axis X, row_dir = (1,0,0), spacing = 1.0
        assert!((vol.calculate_slice_position(AnatomicalPlane::Sagittal, 1) - 1.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------
    // Volume::slice_to_position / position_to_slice
    // ---------------------------------------------------------------

    #[test]
    fn test_slice_to_position_roundtrip() {
        let vol = make_offset_axial_volume();
        // slice_to_position uses raw index * spacing (no origin offset)
        let pos = vol.slice_to_position(AnatomicalPlane::Axial, 1);
        assert!((pos - 2.0).abs() < 1e-9); // 1 * 2.0mm

        let idx = vol.position_to_slice(AnatomicalPlane::Axial, pos);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_position_to_slice_clamps() {
        let vol = make_axial_volume();
        // Very large position should clamp to last slice
        let idx = vol.position_to_slice(AnatomicalPlane::Axial, 999.0);
        assert_eq!(idx, 4); // max index = 5-1 = 4
    }

    // ---------------------------------------------------------------
    // Volume::resample - native slices
    // ---------------------------------------------------------------

    #[test]
    fn test_resample_native_axial_returns_correct_slice() {
        let vol = make_axial_volume();
        let slice = vol.resample(AnatomicalPlane::Axial, 2).unwrap();
        assert_eq!(slice.width, 4);
        assert_eq!(slice.height, 3);
        assert_eq!(slice.pixels.len(), 12);

        // z=2: values should be 200+y*10+x
        assert_eq!(slice.pixels[0], 200);  // (0,0)
        assert_eq!(slice.pixels[1], 201);  // (1,0)
        assert_eq!(slice.pixels[5], 211);  // (1,1)
    }

    #[test]
    fn test_resample_original_matches_native() {
        let vol = make_axial_volume();
        let native = vol.resample(AnatomicalPlane::Axial, 1).unwrap();
        let orig = vol.resample(AnatomicalPlane::Original, 1).unwrap();
        assert_eq!(native.pixels, orig.pixels);
    }

    #[test]
    fn test_resample_out_of_range_returns_none() {
        let vol = make_axial_volume();
        assert!(vol.resample(AnatomicalPlane::Axial, 5).is_none());
        assert!(vol.resample(AnatomicalPlane::Coronal, 3).is_none());
        assert!(vol.resample(AnatomicalPlane::Sagittal, 4).is_none());
    }

    // ---------------------------------------------------------------
    // Volume::resample - resliced (coronal/sagittal from axial)
    // ---------------------------------------------------------------

    #[test]
    fn test_resample_coronal_from_axial_dimensions() {
        let vol = make_axial_volume(); // 4x3x5
        // Coronal from axial: AlongRows -> out_width=4 (w), out_height=5 (d)
        let slice = vol.resample(AnatomicalPlane::Coronal, 1).unwrap();
        assert_eq!(slice.width, 4);
        assert_eq!(slice.height, 5);
    }

    #[test]
    fn test_resample_sagittal_from_axial_dimensions() {
        let vol = make_axial_volume(); // 4x3x5
        // Sagittal from axial: AlongColumns -> out_width=3 (h), out_height=5 (d)
        let slice = vol.resample(AnatomicalPlane::Sagittal, 2).unwrap();
        assert_eq!(slice.width, 3);
        assert_eq!(slice.height, 5);
    }

    #[test]
    fn test_resample_coronal_from_axial_pixel_content() {
        // For a standard axial volume with z_sign > 0, coronal resample reverses Z
        // so superior (high z) is at pixel row 0.
        let vol = make_axial_volume(); // 4x3x5, slice_dir = +Z => z_sign > 0
        // Coronal at row_idx=1: takes row 1 from each slice, then reverses Z order
        let slice = vol.resample(AnatomicalPlane::Coronal, 1).unwrap();
        // z_sign > 0 means z_iter = (0..5).rev() = [4,3,2,1,0]
        // Output row 0 = slice z=4, row 1, cols 0..4 -> values 410,411,412,413
        assert_eq!(slice.pixels[0], 410);
        assert_eq!(slice.pixels[1], 411);
        // Output row 4 = slice z=0, row 1, cols 0..4 -> values 10,11,12,13
        assert_eq!(slice.pixels[4 * 4], 10);
        assert_eq!(slice.pixels[4 * 4 + 3], 13);
    }

    #[test]
    fn test_resample_sagittal_from_axial_pixel_content() {
        let vol = make_axial_volume(); // z_sign > 0 => z_iter reversed
        // Sagittal at col_idx=2: takes column 2 from each slice
        let slice = vol.resample(AnatomicalPlane::Sagittal, 2).unwrap();
        // z_iter = [4,3,2,1,0], output row 0 from z=4
        // For z=4, columns: for each y in 0..3, pixel = data[4*12 + y*4 + 2]
        //   y=0: 402, y=1: 412, y=2: 422
        assert_eq!(slice.pixels[0], 402);
        assert_eq!(slice.pixels[1], 412);
        assert_eq!(slice.pixels[2], 422);
        // Last output row from z=0: y=0: 2, y=1: 12, y=2: 22
        assert_eq!(slice.pixels[4 * 3], 2);
        assert_eq!(slice.pixels[4 * 3 + 1], 12);
    }

    // ---------------------------------------------------------------
    // Volume::resample - resliced from sagittal acquisition
    // ---------------------------------------------------------------

    #[test]
    fn test_resample_axial_from_sagittal_dimensions() {
        let vol = make_sagittal_volume(); // 4x3x5
        // Axial from sagittal: AlongRows -> out_width=5 (d), out_height=4 (w)
        let slice = vol.resample(AnatomicalPlane::Axial, 1).unwrap();
        assert_eq!(slice.width, 5);
        assert_eq!(slice.height, 4);
    }

    #[test]
    fn test_resample_coronal_from_sagittal_dimensions() {
        let vol = make_sagittal_volume(); // 4x3x5
        // Coronal from sagittal: AlongColumns -> out_width=5 (d), out_height=3 (h)
        let slice = vol.resample(AnatomicalPlane::Coronal, 2).unwrap();
        assert_eq!(slice.width, 5);
        assert_eq!(slice.height, 3);
    }

    // ---------------------------------------------------------------
    // get_reslice_operation
    // ---------------------------------------------------------------

    #[test]
    fn test_reslice_operation_native_planes() {
        let axial_vol = make_axial_volume();
        let sag_vol = make_sagittal_volume();

        assert!(matches!(
            axial_vol.get_reslice_operation(AnatomicalPlane::Axial),
            ResliceOperation::Native
        ));
        assert!(matches!(
            sag_vol.get_reslice_operation(AnatomicalPlane::Sagittal),
            ResliceOperation::Native
        ));
        // Original is always Native
        assert!(matches!(
            axial_vol.get_reslice_operation(AnatomicalPlane::Original),
            ResliceOperation::Native
        ));
    }

    #[test]
    fn test_reslice_operation_cross_planes() {
        let vol = make_axial_volume();
        assert!(matches!(
            vol.get_reslice_operation(AnatomicalPlane::Coronal),
            ResliceOperation::AlongRows
        ));
        assert!(matches!(
            vol.get_reslice_operation(AnatomicalPlane::Sagittal),
            ResliceOperation::AlongColumns
        ));
    }

    // ---------------------------------------------------------------
    // ReformattedSlice::compute_image_plane
    // ---------------------------------------------------------------

    #[test]
    fn test_image_plane_axial_position_increases_with_z() {
        let vol = make_axial_volume();
        let s0 = vol.resample(AnatomicalPlane::Axial, 0).unwrap();
        let s3 = vol.resample(AnatomicalPlane::Axial, 3).unwrap();
        let p0 = s0.compute_image_plane(&vol, 0);
        let p3 = s3.compute_image_plane(&vol, 3);
        // slice_direction = +Z, so higher index = higher Z position
        assert!(p3.position.z > p0.position.z);
    }

    #[test]
    fn test_image_plane_coronal_position() {
        let vol = make_axial_volume();
        let s1 = vol.resample(AnatomicalPlane::Coronal, 1).unwrap();
        let plane = s1.compute_image_plane(&vol, 1);
        // Row direction should be along X (left-right)
        assert!((plane.row_direction.x.abs() - 1.0).abs() < 1e-6);
        // Normal should be along Y
        assert!((plane.normal.y.abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_image_plane_sagittal_position() {
        let vol = make_axial_volume();
        let s1 = vol.resample(AnatomicalPlane::Sagittal, 1).unwrap();
        let plane = s1.compute_image_plane(&vol, 1);
        // Row direction should be along Y
        assert!((plane.row_direction.y.abs() - 1.0).abs() < 1e-6);
        // Normal should be along X
        assert!((plane.normal.x.abs() - 1.0).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // Center position invariant (critical for sync, per docs)
    // ---------------------------------------------------------------

    #[test]
    fn test_center_position_calculation_in_to_dicom_image() {
        // Verify that to_dicom_image computes slice_location at IMAGE CENTER,
        // not at the corner. This is the core invariant documented in
        // docs/mpr-implementation.md for correct sync with tilted acquisitions.
        let vol = make_axial_volume();
        let slice = vol.resample(AnatomicalPlane::Axial, 2).unwrap();
        let img = slice.to_dicom_image(&vol, 2);

        let plane = slice.compute_image_plane(&vol, 2);
        let (row_sp, col_sp) = slice.pixel_spacing;
        let half_w = slice.width as f64 / 2.0;
        let half_h = slice.height as f64 / 2.0;
        let expected_center = plane
            .position
            .add(&plane.row_direction.scale(half_w * col_sp))
            .add(&plane.col_direction.scale(half_h * row_sp));

        let expected_z = expected_center.z;
        let actual_z = img.slice_location.unwrap();
        assert!(
            (actual_z - expected_z).abs() < 1e-6,
            "slice_location should be at image center Z: expected {}, got {}",
            expected_z,
            actual_z
        );
    }

    #[test]
    fn test_center_vs_corner_differ_when_tilted() {
        // Simulate a gantry-tilted axial volume where col_direction has a Z component.
        // This verifies that center and corner Z differ, which is the whole reason
        // center-based sync exists.
        let tilt_deg = 20.0_f64;
        let tilt_rad = tilt_deg.to_radians();
        let (w, h, d) = (256, 256, 50);
        let data = vec![100u16; w * h * d];

        let vol = Volume {
            data,
            dimensions: (w, h, d),
            spacing: (1.0, 1.0, 3.0),
            origin: Vec3::new(-128.0, -128.0, -75.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            // Tilted col_direction: Y and Z components
            col_direction: Vec3::new(0.0, tilt_rad.cos(), tilt_rad.sin()),
            slice_direction: Vec3::new(0.0, -tilt_rad.sin(), tilt_rad.cos()),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Axial,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: None,
            default_window_center: 0.0,
            default_window_width: 100.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: (0..50).map(|z| -75.0 + z as f64 * 3.0).collect(),
            pixel_representation: 0,
        };

        let slice = vol.resample(AnatomicalPlane::Axial, 25).unwrap();
        let plane = slice.compute_image_plane(&vol, 25);
        let corner_z = plane.position.z;

        let img = slice.to_dicom_image(&vol, 25);
        let center_z = img.slice_location.unwrap();

        // With 20-degree tilt and 256mm FOV, center should be ~44mm higher than corner
        let diff = (center_z - corner_z).abs();
        assert!(
            diff > 10.0,
            "tilted acquisition should have significant center-vs-corner Z difference, got {:.1}mm",
            diff
        );
    }

    // ---------------------------------------------------------------
    // MprState basic operations
    // ---------------------------------------------------------------

    #[test]
    fn test_mpr_state_default_is_not_active() {
        let state = MprState::new();
        assert!(!state.is_active());
        assert_eq!(state.plane, AnatomicalPlane::Original);
    }

    #[test]
    fn test_mpr_state_set_volume_and_plane() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        assert!(!state.is_active()); // Still Original plane

        state.set_plane(AnatomicalPlane::Coronal);
        assert!(state.is_active());
        // set_plane centers the slice index
        assert_eq!(state.slice_index, 1); // 3/2 = 1
    }

    #[test]
    fn test_mpr_state_navigate() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Axial); // 5 slices, starts at center=2

        state.navigate(1);
        assert_eq!(state.slice_index, 3);
        state.navigate(-10); // should clamp to 0
        assert_eq!(state.slice_index, 0);
        state.navigate(100); // should clamp to 4
        assert_eq!(state.slice_index, 4);
    }

    #[test]
    fn test_mpr_state_clear() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Sagittal);
        assert!(state.is_active());

        state.clear();
        assert!(!state.is_active());
        assert_eq!(state.plane, AnatomicalPlane::Original);
        assert_eq!(state.slice_index, 0);
    }

    #[test]
    fn test_mpr_state_get_slice_caches() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Axial);

        // First call generates slice
        assert!(state.get_slice().is_some());
        // Second call returns cached
        assert!(state.get_slice().is_some());
    }

    // ---------------------------------------------------------------
    // MprState::position_info
    // ---------------------------------------------------------------

    #[test]
    fn test_position_info_format() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Axial);
        let info = state.position_info();
        // Should contain plane name, slice number, count, and mm
        assert!(info.contains("Axial"), "info: {}", info);
        assert!(info.contains("/5"), "info: {}", info);
        assert!(info.contains("mm"), "info: {}", info);
    }
}
