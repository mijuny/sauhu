//! Rigid transformation representation and matrix math
//!
//! Supports 6 DOF (rotation + translation) and 7 DOF (+ uniform scale)
#![allow(dead_code)]

use std::f64::consts::PI;

/// 3D vector
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self::default()
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 1e-10 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            *self
        }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

/// 4x4 homogeneous transformation matrix (column-major for GPU compatibility)
#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    /// Column-major storage: m[col][row]
    pub m: [[f64; 4]; 4],
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat4 {
    pub fn identity() -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn from_translation(t: Vec3) -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [t.x, t.y, t.z, 1.0],
            ],
        }
    }

    pub fn from_scale(s: f64) -> Self {
        Self {
            m: [
                [s, 0.0, 0.0, 0.0],
                [0.0, s, 0.0, 0.0],
                [0.0, 0.0, s, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create rotation matrix from axis-angle (Rodrigues formula)
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let axis = axis.normalize();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self {
            m: [
                [t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0.0],
                [t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0.0],
                [t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create rotation matrix from Euler angles (XYZ order)
    pub fn from_euler_xyz(rx: f64, ry: f64, rz: f64) -> Self {
        let cx = rx.cos();
        let sx = rx.sin();
        let cy = ry.cos();
        let sy = ry.sin();
        let cz = rz.cos();
        let sz = rz.sin();

        // Rotation order: Rz * Ry * Rx
        Self {
            m: [
                [cy * cz, cx * sz + sx * sy * cz, sx * sz - cx * sy * cz, 0.0],
                [
                    -cy * sz,
                    cx * cz - sx * sy * sz,
                    sx * cz + cx * sy * sz,
                    0.0,
                ],
                [sy, -sx * cy, cx * cy, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Matrix multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = [[0.0; 4]; 4];

        for (i, result_col) in result.iter_mut().enumerate() {
            for (j, result_elem) in result_col.iter_mut().enumerate() {
                for k in 0..4 {
                    *result_elem += self.m[k][j] * other.m[i][k];
                }
            }
        }

        Self { m: result }
    }

    /// Transform a point (applies full transformation including translation)
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let w = self.m[0][3] * p.x + self.m[1][3] * p.y + self.m[2][3] * p.z + self.m[3][3];
        Vec3 {
            x: (self.m[0][0] * p.x + self.m[1][0] * p.y + self.m[2][0] * p.z + self.m[3][0]) / w,
            y: (self.m[0][1] * p.x + self.m[1][1] * p.y + self.m[2][1] * p.z + self.m[3][1]) / w,
            z: (self.m[0][2] * p.x + self.m[1][2] * p.y + self.m[2][2] * p.z + self.m[3][2]) / w,
        }
    }

    /// Transform a vector (no translation)
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.m[0][0] * v.x + self.m[1][0] * v.y + self.m[2][0] * v.z,
            y: self.m[0][1] * v.x + self.m[1][1] * v.y + self.m[2][1] * v.z,
            z: self.m[0][2] * v.x + self.m[1][2] * v.y + self.m[2][2] * v.z,
        }
    }

    /// Compute matrix inverse (for 4x4 with last row [0,0,0,1])
    pub fn inverse(&self) -> Option<Self> {
        // For rigid transforms, inverse is transpose of rotation + negated translation
        // This is a simplified inverse assuming orthogonal rotation + translation

        // Extract 3x3 rotation part
        let r00 = self.m[0][0];
        let r01 = self.m[0][1];
        let r02 = self.m[0][2];
        let r10 = self.m[1][0];
        let r11 = self.m[1][1];
        let r12 = self.m[1][2];
        let r20 = self.m[2][0];
        let r21 = self.m[2][1];
        let r22 = self.m[2][2];

        // Translation
        let tx = self.m[3][0];
        let ty = self.m[3][1];
        let tz = self.m[3][2];

        // Transpose rotation
        let rt00 = r00;
        let rt01 = r10;
        let rt02 = r20;
        let rt10 = r01;
        let rt11 = r11;
        let rt12 = r21;
        let rt20 = r02;
        let rt21 = r12;
        let rt22 = r22;

        // New translation: -R^T * t
        let new_tx = -(rt00 * tx + rt10 * ty + rt20 * tz);
        let new_ty = -(rt01 * tx + rt11 * ty + rt21 * tz);
        let new_tz = -(rt02 * tx + rt12 * ty + rt22 * tz);

        Some(Self {
            m: [
                [rt00, rt01, rt02, 0.0],
                [rt10, rt11, rt12, 0.0],
                [rt20, rt21, rt22, 0.0],
                [new_tx, new_ty, new_tz, 1.0],
            ],
        })
    }

    /// Convert to f32 array for GPU upload (column-major, flattened)
    pub fn to_f32_array(self) -> [f32; 16] {
        [
            self.m[0][0] as f32,
            self.m[0][1] as f32,
            self.m[0][2] as f32,
            self.m[0][3] as f32,
            self.m[1][0] as f32,
            self.m[1][1] as f32,
            self.m[1][2] as f32,
            self.m[1][3] as f32,
            self.m[2][0] as f32,
            self.m[2][1] as f32,
            self.m[2][2] as f32,
            self.m[2][3] as f32,
            self.m[3][0] as f32,
            self.m[3][1] as f32,
            self.m[3][2] as f32,
            self.m[3][3] as f32,
        ]
    }
}

/// Rigid transformation with optional uniform scale
///
/// Stored as Euler angles + translation + scale for easy optimization.
/// Converted to matrix for actual transformation.
#[derive(Debug, Clone, Copy)]
pub struct RigidTransform {
    /// Rotation around X axis (radians)
    pub rotation_x: f64,
    /// Rotation around Y axis (radians)
    pub rotation_y: f64,
    /// Rotation around Z axis (radians)
    pub rotation_z: f64,
    /// Translation in X (mm)
    pub translation_x: f64,
    /// Translation in Y (mm)
    pub translation_y: f64,
    /// Translation in Z (mm)
    pub translation_z: f64,
    /// Uniform scale (1.0 = no scaling)
    pub scale: f64,
}

impl Default for RigidTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl RigidTransform {
    pub fn identity() -> Self {
        Self {
            rotation_x: 0.0,
            rotation_y: 0.0,
            rotation_z: 0.0,
            translation_x: 0.0,
            translation_y: 0.0,
            translation_z: 0.0,
            scale: 1.0,
        }
    }

    /// Create from parameter array [rx, ry, rz, tx, ty, tz, scale]
    pub fn from_params(params: &[f64; 7]) -> Self {
        Self {
            rotation_x: params[0],
            rotation_y: params[1],
            rotation_z: params[2],
            translation_x: params[3],
            translation_y: params[4],
            translation_z: params[5],
            scale: params[6],
        }
    }

    /// Create from 6-DOF parameter array [rx, ry, rz, tx, ty, tz] (no scale)
    pub fn from_params_6dof(params: &[f64; 6]) -> Self {
        Self {
            rotation_x: params[0],
            rotation_y: params[1],
            rotation_z: params[2],
            translation_x: params[3],
            translation_y: params[4],
            translation_z: params[5],
            scale: 1.0,
        }
    }

    /// Convert to parameter array [rx, ry, rz, tx, ty, tz, scale]
    pub fn to_params(self) -> [f64; 7] {
        [
            self.rotation_x,
            self.rotation_y,
            self.rotation_z,
            self.translation_x,
            self.translation_y,
            self.translation_z,
            self.scale,
        ]
    }

    /// Convert to 6-DOF parameter array [rx, ry, rz, tx, ty, tz]
    pub fn to_params_6dof(self) -> [f64; 6] {
        [
            self.rotation_x,
            self.rotation_y,
            self.rotation_z,
            self.translation_x,
            self.translation_y,
            self.translation_z,
        ]
    }

    /// Convert to 4x4 transformation matrix
    ///
    /// The matrix transforms points from source to target space:
    /// target_point = matrix * source_point
    pub fn to_matrix(self) -> Mat4 {
        let rotation = Mat4::from_euler_xyz(self.rotation_x, self.rotation_y, self.rotation_z);
        let scale = Mat4::from_scale(self.scale);
        let translation = Mat4::from_translation(Vec3::new(
            self.translation_x,
            self.translation_y,
            self.translation_z,
        ));

        // Order: scale -> rotate -> translate
        // For a point p: T * R * S * p
        translation.mul(&rotation.mul(&scale))
    }

    /// Convert to inverse matrix (target to source)
    pub fn to_inverse_matrix(self) -> Mat4 {
        self.to_matrix().inverse().unwrap_or_default()
    }

    /// Get rotation as Vec3 (for display/debugging)
    pub fn rotation_degrees(&self) -> Vec3 {
        Vec3::new(
            self.rotation_x * 180.0 / PI,
            self.rotation_y * 180.0 / PI,
            self.rotation_z * 180.0 / PI,
        )
    }

    /// Get translation as Vec3
    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.translation_x, self.translation_y, self.translation_z)
    }

    /// Compose two transforms: self followed by other
    pub fn compose(&self, other: &Self) -> Self {
        // Convert to matrices, multiply, extract parameters
        // This is approximate due to Euler angle extraction
        let combined = other.to_matrix().mul(&self.to_matrix());

        // Extract translation directly
        let tx = combined.m[3][0];
        let ty = combined.m[3][1];
        let tz = combined.m[3][2];

        // Extract scale (approximate - assumes uniform scale)
        let sx =
            (combined.m[0][0].powi(2) + combined.m[0][1].powi(2) + combined.m[0][2].powi(2)).sqrt();

        // Extract rotation (Euler angles from rotation matrix)
        // Assumes XYZ rotation order
        let (rx, ry, rz) = extract_euler_xyz(&combined, sx);

        Self {
            rotation_x: rx,
            rotation_y: ry,
            rotation_z: rz,
            translation_x: tx,
            translation_y: ty,
            translation_z: tz,
            scale: sx,
        }
    }
}

/// Extract Euler angles (XYZ order) from rotation matrix
fn extract_euler_xyz(m: &Mat4, scale: f64) -> (f64, f64, f64) {
    let s = if scale > 1e-10 { scale } else { 1.0 };

    // Normalized rotation matrix elements
    let r00 = m.m[0][0] / s;
    let r01 = m.m[0][1] / s;
    let _r02 = m.m[0][2] / s;
    let r10 = m.m[1][0] / s;
    let r11 = m.m[1][1] / s;
    let r20 = m.m[2][0] / s;
    let r21 = m.m[2][1] / s;
    let r22 = m.m[2][2] / s;

    // Extract angles
    let ry = r20.asin();

    let (rx, rz) = if ry.cos().abs() > 1e-6 {
        let rx = (-r21).atan2(r22);
        let rz = (-r10).atan2(r00);
        (rx, rz)
    } else {
        // Gimbal lock
        let rx = 0.0;
        let rz = r01.atan2(r11);
        (rx, rz)
    };

    (rx, ry, rz)
}

/// Parameter bounds for optimization
#[derive(Debug, Clone)]
pub struct TransformBounds {
    /// Max rotation in each axis (radians)
    pub max_rotation: f64,
    /// Max translation in each axis (mm)
    pub max_translation: f64,
    /// Scale range [min, max]
    pub scale_range: (f64, f64),
}

impl Default for TransformBounds {
    fn default() -> Self {
        Self {
            max_rotation: 30.0 * PI / 180.0, // 30 degrees
            max_translation: 50.0,           // 50mm
            scale_range: (0.8, 1.2),         // 20% scaling
        }
    }
}

impl TransformBounds {
    /// Clamp transform parameters to bounds
    pub fn clamp(&self, transform: &mut RigidTransform) {
        transform.rotation_x = transform
            .rotation_x
            .clamp(-self.max_rotation, self.max_rotation);
        transform.rotation_y = transform
            .rotation_y
            .clamp(-self.max_rotation, self.max_rotation);
        transform.rotation_z = transform
            .rotation_z
            .clamp(-self.max_rotation, self.max_rotation);

        transform.translation_x = transform
            .translation_x
            .clamp(-self.max_translation, self.max_translation);
        transform.translation_y = transform
            .translation_y
            .clamp(-self.max_translation, self.max_translation);
        transform.translation_z = transform
            .translation_z
            .clamp(-self.max_translation, self.max_translation);

        transform.scale = transform
            .scale
            .clamp(self.scale_range.0, self.scale_range.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let t = RigidTransform::identity();
        let m = t.to_matrix();

        let p = Vec3::new(1.0, 2.0, 3.0);
        let result = m.transform_point(p);

        assert!((result.x - p.x).abs() < 1e-10);
        assert!((result.y - p.y).abs() < 1e-10);
        assert!((result.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_translation() {
        let mut t = RigidTransform::identity();
        t.translation_x = 10.0;
        t.translation_y = 20.0;
        t.translation_z = 30.0;

        let m = t.to_matrix();
        let p = Vec3::new(0.0, 0.0, 0.0);
        let result = m.transform_point(p);

        assert!((result.x - 10.0).abs() < 1e-10);
        assert!((result.y - 20.0).abs() < 1e-10);
        assert!((result.z - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_90_z() {
        let mut t = RigidTransform::identity();
        t.rotation_z = PI / 2.0; // 90 degrees around Z

        let m = t.to_matrix();
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = m.transform_point(p);

        // X axis rotates to Y axis
        assert!((result.x - 0.0).abs() < 1e-10);
        assert!((result.y - 1.0).abs() < 1e-10);
        assert!((result.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let mut t = RigidTransform::identity();
        t.rotation_x = 0.1;
        t.rotation_y = 0.2;
        t.rotation_z = 0.3;
        t.translation_x = 10.0;
        t.translation_y = 20.0;
        t.translation_z = 30.0;

        let m = t.to_matrix();
        let m_inv = m.inverse().unwrap();

        let p = Vec3::new(5.0, 10.0, 15.0);
        let transformed = m.transform_point(p);
        let back = m_inv.transform_point(transformed);

        assert!((back.x - p.x).abs() < 1e-6);
        assert!((back.y - p.y).abs() < 1e-6);
        assert!((back.z - p.z).abs() < 1e-6);
    }
}
