//! Geometry calculations for DICOM image planes
//!
//! Provides 3D geometry for calculating reference line intersections between
//! orthogonal image planes.

use super::DicomFile;

/// 3D vector for geometric calculations
#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[allow(dead_code)]
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len > 1e-10 {
            Vec3::new(self.x / len, self.y / len, self.z / len)
        } else {
            *self
        }
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn scale(&self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn negate(&self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// 2D point in pixel coordinates
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Image plane in patient coordinate system
#[derive(Debug, Clone)]
pub struct ImagePlane {
    /// Top-left corner position in patient coordinates (mm)
    pub position: Vec3,
    /// Unit vector along image rows (left to right)
    pub row_direction: Vec3,
    /// Unit vector along image columns (top to bottom)
    pub col_direction: Vec3,
    /// Normal vector to the plane (row × col)
    pub normal: Vec3,
    /// Pixel spacing in mm (row_spacing, col_spacing)
    pub pixel_spacing: (f64, f64),
    /// Image dimensions (rows, columns) in pixels
    pub dimensions: (u32, u32),
    /// Frame of Reference UID (for checking if planes can be compared)
    pub frame_of_reference_uid: Option<String>,
}

impl ImagePlane {
    /// Create an ImagePlane from DICOM file metadata
    pub fn from_dicom(dcm: &DicomFile) -> Option<Self> {
        let (px, py, pz) = dcm.image_position_patient()?;
        let orientation = dcm.image_orientation_patient()?;
        let pixel_spacing = dcm.pixel_spacing()?;
        let rows = dcm.rows()?;
        let columns = dcm.columns()?;

        let row_direction = Vec3::new(orientation[0], orientation[1], orientation[2]).normalize();
        let col_direction = Vec3::new(orientation[3], orientation[4], orientation[5]).normalize();
        let normal = row_direction.cross(&col_direction).normalize();

        Some(Self {
            position: Vec3::new(px, py, pz),
            row_direction,
            col_direction,
            normal,
            pixel_spacing,
            dimensions: (rows, columns),
            frame_of_reference_uid: dcm.frame_of_reference_uid(),
        })
    }

    /// Check if two planes share the same coordinate system
    pub fn same_frame_of_reference(&self, other: &ImagePlane) -> bool {
        match (&self.frame_of_reference_uid, &other.frame_of_reference_uid) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// Check if two planes are approximately parallel
    pub fn is_parallel(&self, other: &ImagePlane) -> bool {
        self.normal.dot(&other.normal).abs() > 0.99
    }

    /// Get the physical size of the image in mm
    #[allow(dead_code)]
    pub fn physical_size(&self) -> (f64, f64) {
        let width = self.dimensions.1 as f64 * self.pixel_spacing.1;
        let height = self.dimensions.0 as f64 * self.pixel_spacing.0;
        (width, height)
    }

    /// Convert a 3D patient coordinate to 2D pixel coordinates on this plane
    /// Returns None if the point is not on the plane
    #[allow(dead_code)]
    pub fn patient_to_pixel(&self, point: &Vec3) -> Point2D {
        // Vector from plane origin to point
        let v = point.sub(&self.position);

        // Project onto row and column directions
        let col_mm = v.dot(&self.row_direction);
        let row_mm = v.dot(&self.col_direction);

        // Convert to pixels
        let col_px = col_mm / self.pixel_spacing.1;
        let row_px = row_mm / self.pixel_spacing.0;

        Point2D::new(col_px, row_px)
    }

    /// Convert pixel coordinates to 3D patient coordinates
    pub fn pixel_to_patient(&self, px: f64, py: f64) -> Vec3 {
        // px = column (x direction), py = row (y direction)
        let col_offset = self.row_direction.scale(px * self.pixel_spacing.1);
        let row_offset = self.col_direction.scale(py * self.pixel_spacing.0);
        self.position.add(&col_offset).add(&row_offset)
    }

    /// Get the four corners of the image plane in patient coordinates
    #[allow(dead_code)]
    pub fn corners(&self) -> [Vec3; 4] {
        let (rows, cols) = self.dimensions;
        [
            self.pixel_to_patient(0.0, 0.0),                 // Top-left
            self.pixel_to_patient(cols as f64, 0.0),         // Top-right
            self.pixel_to_patient(cols as f64, rows as f64), // Bottom-right
            self.pixel_to_patient(0.0, rows as f64),         // Bottom-left
        ]
    }

    /// Calculate intersection line between this plane and another plane
    /// Returns two points in pixel coordinates of `self` where the other plane intersects
    pub fn intersect(&self, other: &ImagePlane) -> Option<(Point2D, Point2D)> {
        // Can only intersect planes in the same coordinate system
        if !self.same_frame_of_reference(other) {
            return None;
        }

        // Parallel planes don't intersect in a line
        if self.is_parallel(other) {
            return None;
        }

        // The intersection line direction is perpendicular to both normals
        let _line_dir = self.normal.cross(&other.normal).normalize();

        // Find a point on the intersection line
        // We need a point that's on both planes
        // Plane equations: n1 · (p - p1) = 0 and n2 · (p - p2) = 0

        // Use a parametric approach: find where the "other" plane intersects "self"
        // The other plane's equation is: other.normal · (p - other.position) = 0

        // Find intersection points by checking where the line crosses the edges of self's image
        let (rows, cols) = self.dimensions;
        let edges = [
            // Top edge: y=0, x from 0 to cols
            (Point2D::new(0.0, 0.0), Point2D::new(cols as f64, 0.0)),
            // Right edge: x=cols, y from 0 to rows
            (
                Point2D::new(cols as f64, 0.0),
                Point2D::new(cols as f64, rows as f64),
            ),
            // Bottom edge: y=rows, x from cols to 0
            (
                Point2D::new(cols as f64, rows as f64),
                Point2D::new(0.0, rows as f64),
            ),
            // Left edge: x=0, y from rows to 0
            (Point2D::new(0.0, rows as f64), Point2D::new(0.0, 0.0)),
        ];

        let mut intersections: Vec<Point2D> = Vec::new();

        for (start, end) in edges.iter() {
            if let Some(point) = self.edge_plane_intersection(start, end, other) {
                // Check if point is within edge bounds (with small epsilon)
                let min_x = start.x.min(end.x) - 0.5;
                let max_x = start.x.max(end.x) + 0.5;
                let min_y = start.y.min(end.y) - 0.5;
                let max_y = start.y.max(end.y) + 0.5;

                if point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y {
                    intersections.push(point);
                }
            }
        }

        // We need exactly 2 intersection points to draw a line
        if intersections.len() >= 2 {
            Some((intersections[0], intersections[1]))
        } else {
            None
        }
    }

    /// Find where an edge (in pixel coords) intersects another plane
    fn edge_plane_intersection(
        &self,
        start: &Point2D,
        end: &Point2D,
        other: &ImagePlane,
    ) -> Option<Point2D> {
        // Convert edge endpoints to patient coordinates
        let p1 = self.pixel_to_patient(start.x, start.y);
        let p2 = self.pixel_to_patient(end.x, end.y);

        // Edge direction
        let edge_dir = p2.sub(&p1);
        let edge_len = edge_dir.length();
        if edge_len < 1e-10 {
            return None;
        }

        // Plane equation: other.normal · (p - other.position) = 0
        // Line equation: p = p1 + t * edge_dir
        // Substitute: other.normal · (p1 + t * edge_dir - other.position) = 0
        // Solve for t: t = other.normal · (other.position - p1) / (other.normal · edge_dir)

        let denom = other.normal.dot(&edge_dir);
        if denom.abs() < 1e-10 {
            // Edge is parallel to plane
            return None;
        }

        let diff = other.position.sub(&p1);
        let t = other.normal.dot(&diff) / denom;

        if !(0.0..=1.0).contains(&t) {
            // Intersection is outside the edge
            return None;
        }

        // Calculate intersection point in pixel coordinates
        let px = start.x + t * (end.x - start.x);
        let py = start.y + t * (end.y - start.y);

        Some(Point2D::new(px, py))
    }
}

/// Reference line to draw on a viewport
#[derive(Debug, Clone)]
pub struct ReferenceLine {
    /// Start point in pixel coordinates (relative to image)
    pub start: Point2D,
    /// End point in pixel coordinates (relative to image)
    pub end: Point2D,
    /// Color for the line (R, G, B)
    pub color: (u8, u8, u8),
}

impl ReferenceLine {
    pub fn new(start: Point2D, end: Point2D) -> Self {
        Self {
            start,
            end,
            color: (255, 255, 0), // Yellow by default
        }
    }

    pub fn with_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.color = (r, g, b);
        self
    }
}
