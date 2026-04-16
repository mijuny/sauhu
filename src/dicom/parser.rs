//! DICOM file parsing

use anyhow::{Context, Result};
use dicom::object::{open_file, DefaultDicomObject};
use std::path::Path;

/// Parsed DICOM file with metadata
#[derive(Debug)]
pub struct DicomFile {
    /// The underlying DICOM object
    pub object: DefaultDicomObject,
    /// File path
    pub path: String,
}

impl DicomFile {
    /// Open and parse a DICOM file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let object =
            open_file(&path).with_context(|| format!("Failed to open DICOM file: {}", path_str))?;

        Ok(Self {
            object,
            path: path_str,
        })
    }

    /// Get a string tag value
    pub fn get_string(&self, tag: dicom::core::Tag) -> Option<String> {
        self.object
            .element(tag)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.trim().to_string())
    }

    /// Get Study Instance UID
    pub fn study_instance_uid(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::STUDY_INSTANCE_UID)
    }

    /// Get Series Instance UID
    pub fn series_instance_uid(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::SERIES_INSTANCE_UID)
    }

    /// Get SOP Instance UID
    pub fn sop_instance_uid(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::SOP_INSTANCE_UID)
    }

    /// Get Patient Name
    pub fn patient_name(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::PATIENT_NAME)
    }

    /// Get Patient ID
    pub fn patient_id(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::PATIENT_ID)
    }

    /// Get Study Date
    pub fn study_date(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::STUDY_DATE)
    }

    /// Get Study Description
    pub fn study_description(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::STUDY_DESCRIPTION)
    }

    /// Get Series Description
    pub fn series_description(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::SERIES_DESCRIPTION)
    }

    /// Get Series Number
    pub fn series_number(&self) -> Option<i32> {
        self.object
            .element(dicom::dictionary_std::tags::SERIES_NUMBER)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok())
    }

    /// Get Instance Number (image index in series)
    pub fn instance_number(&self) -> Option<i32> {
        self.object
            .element(dicom::dictionary_std::tags::INSTANCE_NUMBER)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok())
    }

    /// Get Slice Location (position along scan axis in mm)
    pub fn slice_location(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::SLICE_LOCATION)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get Image Position Patient (x, y, z coordinates in mm)
    pub fn image_position_patient(&self) -> Option<(f64, f64, f64)> {
        let elem = self
            .object
            .element(dicom::dictionary_std::tags::IMAGE_POSITION_PATIENT)
            .ok()?;

        let values: Vec<f64> = elem.to_multi_float64().ok()?.into_iter().collect();

        if values.len() >= 3 {
            Some((values[0], values[1], values[2]))
        } else {
            None
        }
    }

    /// Get Image Orientation Patient (direction cosines)
    /// Returns (row_x, row_y, row_z, col_x, col_y, col_z)
    pub fn image_orientation_patient(&self) -> Option<[f64; 6]> {
        let elem = self
            .object
            .element(dicom::dictionary_std::tags::IMAGE_ORIENTATION_PATIENT)
            .ok()?;

        let values: Vec<f64> = elem.to_multi_float64().ok()?.into_iter().collect();

        if values.len() >= 6 {
            Some([
                values[0], values[1], values[2], values[3], values[4], values[5],
            ])
        } else {
            None
        }
    }

    /// Get Frame of Reference UID (for synchronization between series)
    pub fn frame_of_reference_uid(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::FRAME_OF_REFERENCE_UID)
    }

    /// Detect orientation from Image Orientation Patient tag
    pub fn detect_orientation(&self) -> Option<crate::dicom::AnatomicalPlane> {
        use crate::dicom::AnatomicalPlane;

        let iop = self.image_orientation_patient()?;

        // Row and column direction cosines
        let row = [iop[0], iop[1], iop[2]];
        let col = [iop[3], iop[4], iop[5]];

        // Calculate normal vector (cross product of row and col)
        let normal = [
            row[1] * col[2] - row[2] * col[1],
            row[2] * col[0] - row[0] * col[2],
            row[0] * col[1] - row[1] * col[0],
        ];

        // Find dominant axis of normal vector
        let abs_normal = [normal[0].abs(), normal[1].abs(), normal[2].abs()];

        if abs_normal[2] > abs_normal[0] && abs_normal[2] > abs_normal[1] {
            // Z-axis dominant = Axial
            Some(AnatomicalPlane::Axial)
        } else if abs_normal[1] > abs_normal[0] && abs_normal[1] > abs_normal[2] {
            // Y-axis dominant = Coronal
            Some(AnatomicalPlane::Coronal)
        } else {
            // X-axis dominant = Sagittal
            Some(AnatomicalPlane::Sagittal)
        }
    }

    /// Get Modality
    pub fn modality(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::MODALITY)
    }

    /// Get Rows (image height)
    pub fn rows(&self) -> Option<u32> {
        self.object
            .element(dicom::dictionary_std::tags::ROWS)
            .ok()
            .and_then(|e| e.to_int::<u32>().ok())
    }

    /// Get Columns (image width)
    pub fn columns(&self) -> Option<u32> {
        self.object
            .element(dicom::dictionary_std::tags::COLUMNS)
            .ok()
            .and_then(|e| e.to_int::<u32>().ok())
    }

    /// Get Pixel Spacing (row, column in mm)
    pub fn pixel_spacing(&self) -> Option<(f64, f64)> {
        let elem = self
            .object
            .element(dicom::dictionary_std::tags::PIXEL_SPACING)
            .ok()?;

        let values: Vec<f64> = elem.to_multi_float64().ok()?.into_iter().collect();

        if values.len() >= 2 {
            Some((values[0], values[1]))
        } else {
            None
        }
    }

    /// Get Imager Pixel Spacing (row, column in mm) - detector-level spacing for X-ray/CR/DR
    /// This is a fallback when PixelSpacing is not available
    pub fn imager_pixel_spacing(&self) -> Option<(f64, f64)> {
        let elem = self
            .object
            .element(dicom::dictionary_std::tags::IMAGER_PIXEL_SPACING)
            .ok()?;

        let values: Vec<f64> = elem.to_multi_float64().ok()?.into_iter().collect();

        if values.len() >= 2 {
            Some((values[0], values[1]))
        } else {
            None
        }
    }

    /// Get pixel spacing with fallback to imager pixel spacing for X-ray modalities
    pub fn pixel_spacing_with_fallback(&self) -> Option<(f64, f64)> {
        // Try PixelSpacing first (preferred, at patient level)
        if let Some(spacing) = self.pixel_spacing() {
            return Some(spacing);
        }

        // For X-ray modalities (CR, DR, DX), fall back to ImagerPixelSpacing
        let modality = self.modality().unwrap_or_default();
        if matches!(modality.as_str(), "CR" | "DR" | "DX" | "XR") {
            if let Some(spacing) = self.imager_pixel_spacing() {
                tracing::debug!("Using ImagerPixelSpacing as fallback for {}", modality);
                return Some(spacing);
            }
        }

        None
    }

    /// Get Window Center
    pub fn window_center(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::WINDOW_CENTER)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get Window Width
    pub fn window_width(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::WINDOW_WIDTH)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get Rescale Intercept
    pub fn rescale_intercept(&self) -> f64 {
        self.object
            .element(dicom::dictionary_std::tags::RESCALE_INTERCEPT)
            .ok()
            .and_then(|e| e.to_float64().ok())
            .unwrap_or(0.0)
    }

    /// Get Rescale Slope
    pub fn rescale_slope(&self) -> f64 {
        self.object
            .element(dicom::dictionary_std::tags::RESCALE_SLOPE)
            .ok()
            .and_then(|e| e.to_float64().ok())
            .unwrap_or(1.0)
    }

    /// Get Photometric Interpretation
    pub fn photometric_interpretation(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::PHOTOMETRIC_INTERPRETATION)
    }

    /// Check if image is MONOCHROME1 (inverted display)
    pub fn is_monochrome1(&self) -> bool {
        self.photometric_interpretation()
            .map(|s| s == "MONOCHROME1")
            .unwrap_or(false)
    }

    /// Get Pixel Representation (0 = unsigned, 1 = signed)
    pub fn pixel_representation(&self) -> u16 {
        self.object
            .element(dicom::dictionary_std::tags::PIXEL_REPRESENTATION)
            .ok()
            .and_then(|e| e.to_int::<u16>().ok())
            .unwrap_or(0) // Default to unsigned
    }

    /// Get Transfer Syntax UID
    pub fn transfer_syntax(&self) -> String {
        self.object
            .meta()
            .transfer_syntax()
            .trim_end_matches('\0')
            .to_string()
    }

    /// Check if the image uses compressed transfer syntax
    pub fn is_compressed(&self) -> bool {
        let ts = self.transfer_syntax();
        // Uncompressed transfer syntaxes:
        // 1.2.840.10008.1.2   - Implicit VR Little Endian
        // 1.2.840.10008.1.2.1 - Explicit VR Little Endian
        // 1.2.840.10008.1.2.2 - Explicit VR Big Endian
        // All others (1.2.840.10008.1.2.4.*, 1.2.840.10008.1.2.5) are compressed
        !matches!(
            ts.as_str(),
            "1.2.840.10008.1.2" | "1.2.840.10008.1.2.1" | "1.2.840.10008.1.2.2"
        )
    }

    /// Get Patient Age (AS tag like "045Y")
    pub fn patient_age(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::PATIENT_AGE)
    }

    /// Get Patient Birth Date
    pub fn patient_birth_date(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::PATIENT_BIRTH_DATE)
    }

    /// Get Patient Sex
    pub fn patient_sex(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::PATIENT_SEX)
    }

    /// Get Slice Thickness (mm)
    pub fn slice_thickness(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::SLICE_THICKNESS)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get KVP (tube voltage for CT/X-ray)
    pub fn kvp(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::KVP)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get Exposure (mAs for CT/X-ray)
    pub fn exposure(&self) -> Option<i32> {
        self.object
            .element(dicom::dictionary_std::tags::EXPOSURE)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok())
    }

    /// Get MRI Magnetic Field Strength (Tesla)
    pub fn magnetic_field_strength(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::MAGNETIC_FIELD_STRENGTH)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get MRI Repetition Time (TR, ms)
    pub fn repetition_time(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::REPETITION_TIME)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get MRI Echo Time (TE, ms)
    pub fn echo_time(&self) -> Option<f64> {
        self.object
            .element(dicom::dictionary_std::tags::ECHO_TIME)
            .ok()
            .and_then(|e| e.to_float64().ok())
    }

    /// Get MRI Sequence Name
    pub fn sequence_name(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::SEQUENCE_NAME)
    }

    /// Get SOP Class UID
    pub fn sop_class_uid(&self) -> Option<String> {
        self.get_string(dicom::dictionary_std::tags::SOP_CLASS_UID)
    }

    /// Check if this is an image storage SOP class (has displayable pixel data)
    /// Filters out Raw Data, Presentation States, Structured Reports, etc.
    pub fn is_image_storage(&self) -> bool {
        let sop_class = match self.sop_class_uid() {
            Some(uid) => uid,
            None => return false,
        };

        // Image Storage SOP Classes (have pixel data)
        // MR, CT, CR, DX, US, NM, PT, XA, RF, SC, etc.
        let image_sop_classes = [
            "1.2.840.10008.5.1.4.1.1.1",      // CR Image
            "1.2.840.10008.5.1.4.1.1.1.1",    // Digital X-Ray - Presentation
            "1.2.840.10008.5.1.4.1.1.1.1.1",  // Digital X-Ray - Processing
            "1.2.840.10008.5.1.4.1.1.1.2",    // Digital Mammography - Presentation
            "1.2.840.10008.5.1.4.1.1.1.2.1",  // Digital Mammography - Processing
            "1.2.840.10008.5.1.4.1.1.1.3",    // Digital Intra-Oral X-Ray - Presentation
            "1.2.840.10008.5.1.4.1.1.1.3.1",  // Digital Intra-Oral X-Ray - Processing
            "1.2.840.10008.5.1.4.1.1.2",      // CT Image
            "1.2.840.10008.5.1.4.1.1.2.1",    // Enhanced CT
            "1.2.840.10008.5.1.4.1.1.2.2",    // Legacy CT
            "1.2.840.10008.5.1.4.1.1.3.1",    // Ultrasound Multi-frame
            "1.2.840.10008.5.1.4.1.1.4",      // MR Image
            "1.2.840.10008.5.1.4.1.1.4.1",    // Enhanced MR
            "1.2.840.10008.5.1.4.1.1.4.2",    // MR Spectroscopy
            "1.2.840.10008.5.1.4.1.1.4.3",    // Enhanced MR Color
            "1.2.840.10008.5.1.4.1.1.6.1",    // Ultrasound
            "1.2.840.10008.5.1.4.1.1.6.2",    // Enhanced US Volume
            "1.2.840.10008.5.1.4.1.1.7",      // Secondary Capture
            "1.2.840.10008.5.1.4.1.1.7.1",    // Multi-frame Single Bit SC
            "1.2.840.10008.5.1.4.1.1.7.2",    // Multi-frame Grayscale Byte SC
            "1.2.840.10008.5.1.4.1.1.7.3",    // Multi-frame Grayscale Word SC
            "1.2.840.10008.5.1.4.1.1.7.4",    // Multi-frame True Color SC
            "1.2.840.10008.5.1.4.1.1.12.1",   // X-Ray Angiographic
            "1.2.840.10008.5.1.4.1.1.12.1.1", // Enhanced XA
            "1.2.840.10008.5.1.4.1.1.12.2",   // X-Ray Fluoroscopy
            "1.2.840.10008.5.1.4.1.1.12.2.1", // Enhanced XRF
            "1.2.840.10008.5.1.4.1.1.20",     // NM Image
            "1.2.840.10008.5.1.4.1.1.128",    // PET Image
            "1.2.840.10008.5.1.4.1.1.128.1",  // Enhanced PET
            "1.2.840.10008.5.1.4.1.1.130",    // Enhanced PET (legacy)
            "1.2.840.10008.5.1.4.1.1.481.1",  // RT Image
            "1.2.840.10008.5.1.4.1.1.481.2",  // RT Dose
        ];

        // Check if it's a known image SOP class
        if image_sop_classes.iter().any(|&uid| sop_class == uid) {
            return true;
        }

        // Also accept any SOP class that starts with image storage prefixes
        // This catches vendor-specific extensions
        let image_prefixes = [
            "1.2.840.10008.5.1.4.1.1.1",   // X-Ray family
            "1.2.840.10008.5.1.4.1.1.2",   // CT family
            "1.2.840.10008.5.1.4.1.1.3",   // US family
            "1.2.840.10008.5.1.4.1.1.4",   // MR family
            "1.2.840.10008.5.1.4.1.1.6",   // US family
            "1.2.840.10008.5.1.4.1.1.7",   // SC family
            "1.2.840.10008.5.1.4.1.1.12",  // XA/XRF family
            "1.2.840.10008.5.1.4.1.1.20",  // NM
            "1.2.840.10008.5.1.4.1.1.128", // PET
        ];

        if image_prefixes
            .iter()
            .any(|&prefix| sop_class.starts_with(prefix))
        {
            // Exclude known non-image classes within these prefixes
            let non_image_classes = [
                "1.2.840.10008.5.1.4.1.1.66",   // Raw Data Storage
                "1.2.840.10008.5.1.4.1.1.66.1", // Spatial Registration
                "1.2.840.10008.5.1.4.1.1.66.2", // Spatial Fiducials
                "1.2.840.10008.5.1.4.1.1.66.3", // Deformable Spatial Registration
                "1.2.840.10008.5.1.4.1.1.66.4", // Segmentation
                "1.2.840.10008.5.1.4.1.1.66.5", // Surface Segmentation
                "1.2.840.10008.5.1.4.1.1.11.1", // Grayscale Softcopy Presentation State
                "1.2.840.10008.5.1.4.1.1.11.2", // Color Softcopy Presentation State
                "1.2.840.10008.5.1.4.1.1.88",   // SR family
            ];

            return !non_image_classes
                .iter()
                .any(|&uid| sop_class.starts_with(uid));
        }

        false
    }
}

/// Scan a directory for DICOM files.
///
/// Each candidate file is opened via dicom-object to read its meta header.
/// On a 1 k-file study that's 1 k serial file opens on the calling thread;
/// parallelising the opens with rayon gives a roughly Ncpu speedup on spinning
/// / network storage and doesn't hurt on fast SSDs. The order of the returned
/// Vec doesn't matter — `group_files_by_series` re-sorts each series.
pub fn scan_directory<P: AsRef<Path>>(path: P) -> Result<Vec<DicomFile>> {
    use rayon::prelude::*;

    // Walk the directory serially (cheap, dominated by syscall latency); only
    // collect regular files and hand their paths off to rayon for the parse.
    let candidates: Vec<std::path::PathBuf> = walkdir::WalkDir::new(&path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();

    let files: Vec<DicomFile> = candidates
        .into_par_iter()
        .filter_map(|p| DicomFile::open(&p).ok())
        .collect();

    tracing::info!(
        "Scanned {} DICOM files from {:?}",
        files.len(),
        path.as_ref()
    );

    Ok(files)
}
