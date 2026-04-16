//! DICOM pixel data extraction and image processing

use super::{spatial::ROIStats, DicomFile, ImagePlane};
use anyhow::{Context, Result};
use dicom_pixeldata::PixelDecoder;
use image::{GrayImage, Luma, RgbaImage};
use std::sync::OnceLock;

/// Decoded DICOM image with pixel data and metadata
#[derive(Clone)]
pub struct DicomImage {
    /// Raw pixel data as u16
    pub pixels: Vec<u16>,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Pixel spacing (row, column) in mm
    pub pixel_spacing: Option<(f64, f64)>,
    /// Rescale slope for Hounsfield conversion
    pub rescale_slope: f64,
    /// Rescale intercept for Hounsfield conversion
    pub rescale_intercept: f64,
    /// Default window center
    pub window_center: f64,
    /// Default window width
    pub window_width: f64,
    /// Modality
    pub modality: Option<String>,
    /// Min pixel value (for windowing)
    #[allow(dead_code)]
    pub min_value: f64,
    /// Max pixel value (for windowing)
    #[allow(dead_code)]
    pub max_value: f64,
    /// Whether to invert display (MONOCHROME1)
    #[allow(dead_code)]
    pub invert: bool,
    /// Image plane geometry for reference line calculations
    pub image_plane: Option<ImagePlane>,
    /// SOP Instance UID (unique identifier for this image)
    pub sop_instance_uid: Option<String>,
    /// Study Instance UID (for sync matching)
    pub study_instance_uid: Option<String>,
    // --- Patient/Study Metadata ---
    /// Patient name
    pub patient_name: Option<String>,
    /// Patient ID
    pub patient_id: Option<String>,
    /// Patient age (e.g., "045Y") or calculated from birth date
    pub patient_age: Option<String>,
    /// Patient sex (M/F/O)
    pub patient_sex: Option<String>,
    /// Study date (YYYYMMDD)
    pub study_date: Option<String>,
    /// Study description
    pub study_description: Option<String>,
    /// Series description
    pub series_description: Option<String>,
    // --- Technical Parameters ---
    /// Slice thickness (mm)
    pub slice_thickness: Option<f64>,
    /// MRI field strength (Tesla)
    pub magnetic_field_strength: Option<f64>,
    /// MRI TR (ms)
    pub repetition_time: Option<f64>,
    /// MRI TE (ms)
    pub echo_time: Option<f64>,
    /// MRI sequence name
    pub sequence_name: Option<String>,
    /// CT/X-ray kVp
    pub kvp: Option<f64>,
    /// CT/X-ray mAs
    pub exposure: Option<i32>,
    /// Slice location from DICOM tag (for sync)
    pub slice_location: Option<f64>,
    /// Pixel representation (0 = unsigned, 1 = signed)
    pub pixel_representation: u16,

    /// Lazily computed anatomy bounding box; cached so repeated smart-fit and
    /// auto-window calls don't rescan the whole pixel buffer. Thread-safe
    /// via `OnceLock` because `DicomImage` instances travel across worker
    /// threads (cache, prefetch, MPR).
    #[doc(hidden)]
    pub(crate) anatomy_bounds_cache: OnceLock<Option<(u32, u32, u32, u32)>>,
    /// Lazily computed content window keyed by the bounds it was computed
    /// for. Stores `(x_min, y_min, x_max, y_max, center, width)`. Only the
    /// first caller's bounds are memoised; mismatched callers recompute but
    /// do not evict, which keeps behaviour identical to the old path.
    #[doc(hidden)]
    pub(crate) content_window_cache: OnceLock<(u32, u32, u32, u32, f64, f64)>,
}

impl DicomImage {
    /// Extract pixel data from a DICOM file
    pub fn from_file(dcm: &DicomFile) -> Result<Self> {
        let width = dcm.columns().context("Missing Columns tag")?;
        let height = dcm.rows().context("Missing Rows tag")?;
        let modality = dcm.modality();
        let is_ct = modality.as_ref().map(|m| m == "CT").unwrap_or(false);
        let is_compressed = dcm.is_compressed();

        tracing::debug!(
            "Loading image {}x{}, modality={:?}, compressed={}, transfer_syntax={}",
            width,
            height,
            modality,
            is_compressed,
            dcm.transfer_syntax()
        );

        // Decide extraction strategy based on compression and photometric interpretation:
        // - Uncompressed: extract raw bytes directly from PixelData (more reliable)
        // - Compressed: use decode_pixel_data() which handles decompression
        let is_monochrome1 = dcm.is_monochrome1();
        let (pixels, min_value, max_value, rescale_slope, rescale_intercept) = if !is_compressed {
            // Uncompressed: extract raw pixel data directly (bypasses decode_pixel_data issues)
            tracing::debug!(
                "Uncompressed {}: extracting raw pixels directly",
                if is_ct { "CT" } else { "non-CT" }
            );
            let (raw_pixels, raw_min, raw_max) =
                extract_raw_pixels_direct(&dcm.object, width, height)?;

            // Get rescale values (CT uses these for HU, others may have them too)
            let slope = dcm.rescale_slope();
            let intercept = dcm.rescale_intercept();

            if is_ct || intercept != 0.0 || slope != 1.0 {
                tracing::debug!("Rescale: slope={}, intercept={}", slope, intercept);
                // raw_min/raw_max already bracket the u16 range; the rescale is
                // monotonic in the pixel value so we only need to transform the
                // two endpoints (swapping when slope is negative) instead of
                // sweeping the whole buffer twice more.
                let a = raw_min * slope + intercept;
                let b = raw_max * slope + intercept;
                let (min, max) = if a <= b { (a, b) } else { (b, a) };
                tracing::debug!("Scaled range: min={:.0}, max={:.0}", min, max);
                (raw_pixels, min, max, slope, intercept)
            } else {
                // No rescale - use raw values directly
                tracing::debug!("Raw pixel range: {:.0}-{:.0}", raw_min, raw_max);
                (raw_pixels, raw_min, raw_max, 1.0, 0.0)
            }
        } else {
            // Compressed: use decode_pixel_data() which handles decompression
            tracing::debug!("Compressed: using decode_pixel_data()");
            let pixel_data = dcm.object.decode_pixel_data().with_context(|| {
                format!(
                    "Failed to decode compressed pixel data (transfer_syntax={})",
                    dcm.transfer_syntax()
                )
            })?;

            if is_monochrome1 {
                // MONOCHROME1: use to_dynamic_image() which handles inversion
                tracing::debug!("MONOCHROME1: using to_dynamic_image()");
                match pixel_data.to_dynamic_image(0) {
                    Ok(img) => {
                        let gray = img.to_luma16();
                        let pixels: Vec<u16> = gray.pixels().map(|p| p.0[0]).collect();
                        let (min_u, max_u) = u16_minmax(&pixels);
                        let min = min_u as f64;
                        let max = max_u as f64;
                        tracing::debug!(
                            "Decompressed to grayscale, pixel range: {:.0}-{:.0}",
                            min,
                            max
                        );
                        (pixels, min, max, 1.0, 0.0)
                    }
                    Err(e) => {
                        tracing::warn!("to_dynamic_image failed: {}, trying raw extraction", e);
                        let (pixels, min, max) = extract_raw_pixels(&pixel_data, width, height)?;
                        (pixels, min, max, 1.0, 0.0)
                    }
                }
            } else {
                // Compressed MONOCHROME2: extract from decoded data
                let (pixels, min, max) = extract_raw_pixels(&pixel_data, width, height)?;
                tracing::debug!("Compressed pixel range: {:.0}-{:.0}", min, max);
                (pixels, min, max, 1.0, 0.0)
            }
        };

        // Get pixel spacing with fallback to ImagerPixelSpacing for X-ray modalities
        let pixel_spacing = dcm.pixel_spacing_with_fallback();

        // Get SOP Instance UID for annotation persistence
        let sop_instance_uid = dcm.sop_instance_uid();

        // Get Study Instance UID for sync matching
        let study_instance_uid = dcm.study_instance_uid();

        // Get window settings from DICOM or calculate from actual value range
        let (window_center, window_width) = match (dcm.window_center(), dcm.window_width()) {
            (Some(wc), Some(ww)) if is_ct && !is_compressed => {
                // For uncompressed CT, DICOM WL is in HU terms - use directly
                tracing::debug!("Using DICOM WL (HU): {}/{}", wc, ww);
                (wc, ww)
            }
            _ => {
                // Calculate from actual pixel/HU range
                let ww = max_value - min_value;
                let wc = (min_value + max_value) / 2.0;
                tracing::debug!("Auto WL from range: {:.0}/{:.0}", wc, ww);
                (wc, ww.max(1.0))
            }
        };

        // Extract image plane geometry for reference lines
        let image_plane = ImagePlane::from_dicom(dcm);

        // Extract patient/study metadata
        let patient_name = dcm.patient_name();
        let patient_id = dcm.patient_id();
        let patient_sex = dcm.patient_sex();
        let study_date = dcm.study_date();
        let study_description = dcm.study_description();
        let series_description = dcm.series_description();

        // Get patient age (prefer direct tag, fallback to calculation from birth date)
        let patient_age = dcm.patient_age().or_else(|| {
            let birth = dcm.patient_birth_date()?;
            let study = dcm.study_date()?;
            calculate_age(&birth, &study)
        });

        // Extract technical parameters (modality-specific)
        let slice_location = dcm.slice_location();
        let slice_thickness = dcm.slice_thickness();
        let magnetic_field_strength = dcm.magnetic_field_strength();
        let repetition_time = dcm.repetition_time();
        let echo_time = dcm.echo_time();
        let sequence_name = dcm.sequence_name();
        let kvp = dcm.kvp();
        let exposure = dcm.exposure();
        let pixel_representation = dcm.pixel_representation();

        Ok(Self {
            pixels,
            width,
            height,
            pixel_spacing,
            rescale_slope,
            rescale_intercept,
            window_center,
            window_width,
            modality,
            min_value,
            max_value,
            invert: false,
            image_plane,
            sop_instance_uid,
            study_instance_uid,
            patient_name,
            patient_id,
            patient_age,
            patient_sex,
            study_date,
            study_description,
            series_description,
            slice_thickness,
            magnetic_field_strength,
            repetition_time,
            echo_time,
            sequence_name,
            kvp,
            exposure,
            slice_location,
            pixel_representation,
            anatomy_bounds_cache: OnceLock::new(),
            content_window_cache: OnceLock::new(),
        })
    }

    /// Apply window/level and convert to 8-bit grayscale
    pub fn apply_window(&self, window_center: f64, window_width: f64) -> GrayImage {
        let mut img = GrayImage::new(self.width, self.height);

        let min_val = window_center - window_width / 2.0;
        let max_val = window_center + window_width / 2.0;
        let is_signed = self.pixel_representation == 1;

        for (i, &pixel) in self.pixels.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;

            // Get raw pixel value, handling signed representation
            let raw_value = if is_signed {
                // Interpret u16 bits as i16 for signed pixel data
                (pixel as i16) as f64
            } else {
                pixel as f64
            };

            // Apply rescale (converts to HU for CT, identity for others)
            let value = raw_value * self.rescale_slope + self.rescale_intercept;

            // Apply window/level
            let normalized = if value <= min_val {
                0.0
            } else if value >= max_val {
                1.0
            } else {
                (value - min_val) / window_width
            };

            let gray = (normalized * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Luma([gray]));
        }

        img
    }

    /// Convert to RGBA for display
    pub fn to_rgba(&self, window_center: f64, window_width: f64) -> RgbaImage {
        let gray = self.apply_window(window_center, window_width);

        // Convert grayscale to RGBA
        let mut rgba = RgbaImage::new(self.width, self.height);
        for (x, y, pixel) in gray.enumerate_pixels() {
            let g = pixel.0[0];
            rgba.put_pixel(x, y, image::Rgba([g, g, g, 255]));
        }
        rgba
    }

    /// Get pixel value at position (raw, not rescaled)
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<u16> {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.pixels.get(idx).copied()
        } else {
            None
        }
    }

    /// Get Hounsfield value at position
    pub fn get_hounsfield(&self, x: u32, y: u32) -> Option<f64> {
        self.get_pixel(x, y).map(|p| {
            // Handle signed pixel representation
            let raw = if self.pixel_representation == 1 {
                (p as i16) as f64
            } else {
                p as f64
            };
            raw * self.rescale_slope + self.rescale_intercept
        })
    }

    /// Calculate statistics for a circular ROI
    /// Returns (mean, min, max, std_dev, pixel_count)
    pub fn calculate_circle_stats(&self, cx: f64, cy: f64, radius: f64) -> Option<ROIStats> {
        if radius <= 0.0 {
            return None;
        }

        let mut values = Vec::new();
        let r2 = radius * radius;

        // Iterate over bounding box of the circle
        let x_min = ((cx - radius).floor() as i32).max(0) as u32;
        let x_max = ((cx + radius).ceil() as i32).min(self.width as i32 - 1) as u32;
        let y_min = ((cy - radius).floor() as i32).max(0) as u32;
        let y_max = ((cy + radius).ceil() as i32).min(self.height as i32 - 1) as u32;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r2 {
                    if let Some(val) = self.get_hounsfield(x, y) {
                        values.push(val);
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

        // Calculate standard deviation
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

    /// Find bounding box of actual anatomy (excluding background)
    /// Returns (x_min, y_min, x_max, y_max) or None if image is empty.
    ///
    /// Pixel data is immutable after decode, so the result is cached in
    /// `anatomy_bounds_cache` and reused on every subsequent call. Before
    /// caching, smart-fit and auto-window would rescan the full buffer and
    /// (for non-CT images) sort a freshly-allocated f64 copy each time.
    pub fn find_anatomy_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        *self
            .anatomy_bounds_cache
            .get_or_init(|| self.compute_anatomy_bounds())
    }

    fn compute_anatomy_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let is_ct = self.modality.as_ref().map(|m| m == "CT").unwrap_or(false);

        // Determine threshold based on modality
        let threshold = if is_ct {
            // CT: threshold at -500 HU (above air, which is ~-1000 HU)
            -500.0
        } else {
            // Other modalities: use percentile-based threshold
            // Find 5th percentile as background threshold
            let mut values: Vec<f64> = self
                .pixels
                .iter()
                .map(|&p| (p as f64) * self.rescale_slope + self.rescale_intercept)
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p5_idx = values.len() * 5 / 100;
            let p95_idx = values.len() * 95 / 100;
            // Threshold is slightly above the 5th percentile
            values[p5_idx] + (values[p95_idx] - values[p5_idx]) * 0.1
        };

        let mut x_min = self.width;
        let mut x_max = 0u32;
        let mut y_min = self.height;
        let mut y_max = 0u32;

        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(val) = self.get_hounsfield(x, y) {
                    if val > threshold {
                        x_min = x_min.min(x);
                        x_max = x_max.max(x);
                        y_min = y_min.min(y);
                        y_max = y_max.max(y);
                    }
                }
            }
        }

        // Check if we found any anatomy
        if x_min >= x_max || y_min >= y_max {
            return None;
        }

        // Add 2% padding
        let padding_x = ((x_max - x_min) as f64 * 0.02) as u32;
        let padding_y = ((y_max - y_min) as f64 * 0.02) as u32;

        let x_min = x_min.saturating_sub(padding_x);
        let y_min = y_min.saturating_sub(padding_y);
        let x_max = (x_max + padding_x).min(self.width - 1);
        let y_max = (y_max + padding_y).min(self.height - 1);

        Some((x_min, y_min, x_max, y_max))
    }

    /// Check if this is a CT image
    pub fn is_ct(&self) -> bool {
        self.modality.as_ref().map(|m| m == "CT").unwrap_or(false)
    }

    /// Calculate optimal window center/width for content within bounds.
    /// Returns `(window_center, window_width)`.
    ///
    /// The result is memoised per-image keyed by the bounds it was computed
    /// for: callers that re-run smart-fit with the same anatomy bounds get
    /// an O(1) cache hit. Callers passing different bounds still recompute
    /// (and do not evict the cached entry).
    pub fn calculate_content_window(
        &self,
        x_min: u32,
        y_min: u32,
        x_max: u32,
        y_max: u32,
    ) -> (f64, f64) {
        if let Some(&(cx0, cy0, cx1, cy1, center, width)) = self.content_window_cache.get() {
            if (cx0, cy0, cx1, cy1) == (x_min, y_min, x_max, y_max) {
                return (center, width);
            }
        }
        let (center, width) = self.compute_content_window(x_min, y_min, x_max, y_max);
        // Only seed the cache once; an earlier caller with different bounds
        // keeps its entry. This matches the previous behaviour of "no cache"
        // for mismatched callers.
        let _ = self
            .content_window_cache
            .set((x_min, y_min, x_max, y_max, center, width));
        (center, width)
    }

    fn compute_content_window(
        &self,
        x_min: u32,
        y_min: u32,
        x_max: u32,
        y_max: u32,
    ) -> (f64, f64) {
        // Collect all pixel values within the anatomy bounds
        let mut values: Vec<f64> = Vec::new();
        for y in y_min..=y_max {
            for x in x_min..=x_max {
                if let Some(val) = self.get_hounsfield(x, y) {
                    values.push(val);
                }
            }
        }

        if values.is_empty() {
            return (self.window_center, self.window_width);
        }

        // Sort for percentile calculation
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use 1st to 99th percentile to exclude extreme outliers
        let p1_idx = (values.len() as f64 * 0.01) as usize;
        let p99_idx = (values.len() as f64 * 0.99) as usize;
        let p1_idx = p1_idx.max(0);
        let p99_idx = p99_idx.min(values.len() - 1);

        let min_val = values[p1_idx];
        let max_val = values[p99_idx];

        // Window width spans the range, center is in the middle
        let window_width = (max_val - min_val).max(1.0);
        let window_center = (min_val + max_val) / 2.0;

        (window_center, window_width)
    }
}

/// Extract raw pixels directly from DICOM object (bypasses decode_pixel_data)
/// This is more reliable for uncompressed transfer syntaxes
fn extract_raw_pixels_direct(
    object: &dicom::object::DefaultDicomObject,
    width: u32,
    height: u32,
) -> Result<(Vec<u16>, f64, f64)> {
    use dicom::dictionary_std::tags;

    let expected_len = (width * height) as usize;

    // Get PixelData element directly
    let pixel_elem = object
        .element(tags::PIXEL_DATA)
        .context("Missing PixelData element")?;

    // Get bits allocated to determine byte layout
    let bits_allocated: u16 = object
        .element(tags::BITS_ALLOCATED)
        .ok()
        .and_then(|e| e.to_int().ok())
        .unwrap_or(16);

    let samples_per_pixel: u16 = object
        .element(tags::SAMPLES_PER_PIXEL)
        .ok()
        .and_then(|e| e.to_int().ok())
        .unwrap_or(1);

    // Extract raw bytes from PixelData
    let data: Vec<u8> = pixel_elem
        .to_bytes()
        .context("Failed to get PixelData bytes")?
        .to_vec();

    tracing::debug!(
        "Direct extraction: {} bytes, bits_allocated={}, samples_per_pixel={}, expected={} pixels",
        data.len(),
        bits_allocated,
        samples_per_pixel,
        expected_len
    );

    let pixels: Vec<u16> = if samples_per_pixel == 1 {
        // Grayscale
        if bits_allocated == 16 || data.len() == expected_len * 2 {
            // 16-bit grayscale
            data.chunks_exact(2)
                .take(expected_len)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect()
        } else if bits_allocated == 8 || data.len() == expected_len {
            // 8-bit grayscale - scale to 16-bit
            data.iter()
                .take(expected_len)
                .map(|&b| (b as u16) << 8)
                .collect()
        } else {
            anyhow::bail!(
                "Unexpected grayscale data: {} bytes for {} pixels, bits_allocated={}",
                data.len(),
                expected_len,
                bits_allocated
            );
        }
    } else if samples_per_pixel == 3 {
        // RGB - convert to grayscale
        if bits_allocated == 8 || data.len() == expected_len * 3 {
            tracing::debug!("Converting 8-bit RGB to grayscale");
            data.chunks_exact(3)
                .take(expected_len)
                .map(|rgb| {
                    let lum =
                        (rgb[0] as u32 * 299 + rgb[1] as u32 * 587 + rgb[2] as u32 * 114) / 1000;
                    (lum as u16) << 8
                })
                .collect()
        } else if bits_allocated == 16 || data.len() == expected_len * 6 {
            tracing::debug!("Converting 16-bit RGB to grayscale");
            data.chunks_exact(6)
                .take(expected_len)
                .map(|rgb| {
                    let r = u16::from_le_bytes([rgb[0], rgb[1]]) as u32;
                    let g = u16::from_le_bytes([rgb[2], rgb[3]]) as u32;
                    let b = u16::from_le_bytes([rgb[4], rgb[5]]) as u32;
                    ((r * 299 + g * 587 + b * 114) / 1000) as u16
                })
                .collect()
        } else {
            anyhow::bail!(
                "Unexpected RGB data: {} bytes for {} pixels, bits_allocated={}",
                data.len(),
                expected_len,
                bits_allocated
            );
        }
    } else {
        anyhow::bail!("Unsupported samples_per_pixel: {}", samples_per_pixel);
    };

    if pixels.len() != expected_len {
        anyhow::bail!(
            "Pixel count mismatch: got {}, expected {}",
            pixels.len(),
            expected_len
        );
    }

    let (min, max) = u16_minmax(&pixels);
    Ok((pixels, min as f64, max as f64))
}

/// Extract raw pixels from decoded pixel data (for compressed images)
fn extract_raw_pixels(
    pixel_data: &dicom_pixeldata::DecodedPixelData,
    width: u32,
    height: u32,
) -> Result<(Vec<u16>, f64, f64)> {
    let expected_len = (width * height) as usize;
    let data = pixel_data.data();

    tracing::debug!(
        "Raw data size: {}, expected: {} pixels",
        data.len(),
        expected_len
    );

    let pixels: Vec<u16> = if data.len() == expected_len * 2 {
        // 16-bit grayscale
        data.chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    } else if data.len() == expected_len {
        // 8-bit grayscale
        data.iter().map(|&b| (b as u16) << 8).collect()
    } else if data.len() == expected_len * 3 {
        // 8-bit RGB - convert to grayscale (luminance)
        tracing::debug!("Converting 8-bit RGB to grayscale");
        data.chunks_exact(3)
            .map(|rgb| {
                let lum = (rgb[0] as u32 * 299 + rgb[1] as u32 * 587 + rgb[2] as u32 * 114) / 1000;
                (lum as u16) << 8
            })
            .collect()
    } else if data.len() == expected_len * 6 {
        // 16-bit RGB - convert to grayscale
        tracing::debug!("Converting 16-bit RGB to grayscale");
        data.chunks_exact(6)
            .map(|rgb| {
                let r = u16::from_le_bytes([rgb[0], rgb[1]]) as u32;
                let g = u16::from_le_bytes([rgb[2], rgb[3]]) as u32;
                let b = u16::from_le_bytes([rgb[4], rgb[5]]) as u32;
                ((r * 299 + g * 587 + b * 114) / 1000) as u16
            })
            .collect()
    } else if data.len() == expected_len * 4 {
        // 32-bit data (e.g., RT dose) - take high 16 bits
        tracing::debug!("Converting 32-bit to 16-bit");
        data.chunks_exact(4)
            .map(|chunk| {
                let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                (val >> 16) as u16
            })
            .collect()
    } else if data.len() > expected_len * 2 {
        // Multi-frame or extra data - just take first frame
        tracing::debug!("Extra data ({}), taking first frame", data.len());
        let frame_size = expected_len * 2;
        if data.len() >= frame_size {
            data[..frame_size]
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect()
        } else {
            // Maybe 8-bit multi-frame
            data[..expected_len]
                .iter()
                .map(|&b| (b as u16) << 8)
                .collect()
        }
    } else {
        anyhow::bail!(
            "Unexpected pixel data size: {} (expected {} pixels, {}x{})",
            data.len(),
            expected_len,
            width,
            height
        );
    };

    let (min, max) = u16_minmax(&pixels);
    Ok((pixels, min as f64, max as f64))
}

/// Single-pass min/max over u16 pixels.
///
/// `iter().min()` + `iter().max()` walks the buffer twice. For a 512^2 slice
/// that is an extra ~260 k reads we pay on every compressed-image decode and
/// direct pixel extract path. Merging into one pass also lets the optimiser
/// keep `min`/`max` in registers.
#[inline]
pub(crate) fn u16_minmax(pixels: &[u16]) -> (u16, u16) {
    let mut iter = pixels.iter().copied();
    let Some(first) = iter.next() else {
        return (0, u16::MAX);
    };
    let mut min = first;
    let mut max = first;
    for p in iter {
        if p < min {
            min = p;
        } else if p > max {
            max = p;
        }
    }
    (min, max)
}

#[cfg(test)]
mod minmax_tests {
    use super::u16_minmax;

    #[test]
    fn empty_returns_full_range_sentinel() {
        assert_eq!(u16_minmax(&[]), (0, u16::MAX));
    }

    #[test]
    fn single_value_is_both_ends() {
        assert_eq!(u16_minmax(&[1234]), (1234, 1234));
    }

    #[test]
    fn matches_naive_minmax() {
        let xs = [500u16, 10, 42, 42, 65000, 1, 7];
        let expected = (
            *xs.iter().min().unwrap(),
            *xs.iter().max().unwrap(),
        );
        assert_eq!(u16_minmax(&xs), expected);
    }
}

#[cfg(test)]
mod cache_tests {
    //! Exercises the OnceLock caches for anatomy bounds and content window.
    //! We can't easily construct a real DicomImage from a raw DICOM file in a
    //! unit test, but the cache logic itself is independent of the decode
    //! path, so we build a DicomImage by hand via the same shape used by
    //! the cache module's test helpers.
    use super::*;

    fn test_image(w: u32, h: u32, pixels: Vec<u16>) -> DicomImage {
        DicomImage {
            pixels,
            width: w,
            height: h,
            pixel_spacing: None,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            window_center: 400.0,
            window_width: 1500.0,
            modality: Some("CT".to_string()),
            min_value: 0.0,
            max_value: 4095.0,
            invert: false,
            image_plane: None,
            sop_instance_uid: None,
            study_instance_uid: None,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            series_description: None,
            slice_thickness: None,
            magnetic_field_strength: None,
            repetition_time: None,
            echo_time: None,
            sequence_name: None,
            kvp: None,
            exposure: None,
            slice_location: None,
            pixel_representation: 0,
            anatomy_bounds_cache: OnceLock::new(),
            content_window_cache: OnceLock::new(),
        }
    }

    #[test]
    fn anatomy_bounds_cached_result_is_stable() {
        // Simple bright-on-dark 8x8 pattern (central 4x4 "anatomy").
        let mut pixels = vec![0u16; 64];
        for y in 2..6 {
            for x in 2..6 {
                pixels[y * 8 + x] = 2000; // > -500 HU after slope=1, intercept=0
            }
        }
        let img = test_image(8, 8, pixels);
        let first = img.find_anatomy_bounds();
        let second = img.find_anatomy_bounds();
        assert_eq!(first, second);
        // Second call must see the cache populated.
        assert!(img.anatomy_bounds_cache.get().is_some());
    }

    #[test]
    fn content_window_caches_only_first_bounds() {
        let img = test_image(4, 4, vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]);
        let (c1, w1) = img.calculate_content_window(0, 0, 3, 3);
        // Same bounds — must hit cache and return identical values.
        let (c2, w2) = img.calculate_content_window(0, 0, 3, 3);
        assert_eq!((c1, w1), (c2, w2));
        assert!(img.content_window_cache.get().is_some());

        // Different bounds recompute but must not evict the cached entry.
        let (_c3, _w3) = img.calculate_content_window(1, 1, 2, 2);
        let cached = img.content_window_cache.get().unwrap();
        assert_eq!((cached.0, cached.1, cached.2, cached.3), (0, 0, 3, 3));
    }
}

/// Preset window/level values for different modalities and tissues
pub struct WindowPreset {
    pub name: &'static str,
    pub center: f64,
    pub width: f64,
}

pub const CT_PRESETS: &[WindowPreset] = &[
    WindowPreset {
        name: "Brain",
        center: 40.0,
        width: 80.0,
    },
    WindowPreset {
        name: "Subdural",
        center: 75.0,
        width: 215.0,
    },
    WindowPreset {
        name: "Stroke",
        center: 40.0,
        width: 40.0,
    },
    WindowPreset {
        name: "Bone",
        center: 400.0,
        width: 1800.0,
    },
    WindowPreset {
        name: "Soft Tissue",
        center: 50.0,
        width: 400.0,
    },
    WindowPreset {
        name: "Lung",
        center: -600.0,
        width: 1500.0,
    },
    WindowPreset {
        name: "Liver",
        center: 60.0,
        width: 160.0,
    },
    WindowPreset {
        name: "Mediastinum",
        center: 50.0,
        width: 350.0,
    },
    WindowPreset {
        name: "Abdomen",
        center: 40.0,
        width: 400.0,
    },
];

/// Calculate age from birth date and study date (both YYYYMMDD format)
fn calculate_age(birth_date: &str, study_date: &str) -> Option<String> {
    // Parse dates (YYYYMMDD)
    if birth_date.len() < 8 || study_date.len() < 8 {
        return None;
    }

    let birth_year: i32 = birth_date[0..4].parse().ok()?;
    let birth_month: i32 = birth_date[4..6].parse().ok()?;
    let birth_day: i32 = birth_date[6..8].parse().ok()?;

    let study_year: i32 = study_date[0..4].parse().ok()?;
    let study_month: i32 = study_date[4..6].parse().ok()?;
    let study_day: i32 = study_date[6..8].parse().ok()?;

    let mut age = study_year - birth_year;
    if study_month < birth_month || (study_month == birth_month && study_day < birth_day) {
        age -= 1;
    }

    if age >= 0 {
        Some(format!("{:03}Y", age))
    } else {
        None
    }
}
