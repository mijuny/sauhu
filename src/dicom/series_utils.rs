//! Series utilities for grouping, sorting, and naming
#![allow(dead_code)]

use crate::dicom::DicomFile;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Synchronization info for a viewport's series
#[derive(Debug, Clone)]
pub struct SyncInfo {
    pub frame_of_reference_uid: Option<String>,
    pub study_instance_uid: Option<String>,
    pub orientation: Option<&'static str>,
    pub slice_locations: Vec<Option<f64>>,
}

impl SyncInfo {
    pub fn from_series_info(info: &SeriesInfo) -> Self {
        Self {
            frame_of_reference_uid: info.frame_of_reference_uid.clone(),
            study_instance_uid: info.study_instance_uid.clone(),
            orientation: info.orientation,
            slice_locations: info.slice_locations.clone(),
        }
    }
}

/// Information about a series with its sorted files
#[derive(Debug, Clone)]
pub struct SeriesInfo {
    /// Series Instance UID
    pub series_uid: String,
    /// Series description (raw from DICOM)
    pub description: Option<String>,
    /// Parsed/formatted series name
    pub display_name: String,
    /// Modality (MR, CT, etc.)
    pub modality: Option<String>,
    /// Series number
    pub series_number: Option<i32>,
    /// Sorted file paths
    pub files: Vec<PathBuf>,
    /// Frame of Reference UID (for sync grouping)
    pub frame_of_reference_uid: Option<String>,
    /// Study Instance UID (for sync grouping)
    pub study_instance_uid: Option<String>,
    /// Orientation (Ax/Cor/Sag) from DICOM tag
    pub orientation: Option<&'static str>,
    /// Slice locations for each file (same order as files, for sync scrolling)
    pub slice_locations: Vec<Option<f64>>,
}

/// File with sorting key for ordering within a series
struct SortableFile {
    path: PathBuf,
    instance_number: Option<i32>,
    slice_location: Option<f64>,
    /// Full ImagePositionPatient for consistent sync computation
    image_position: Option<(f64, f64, f64)>,
    /// ImageOrientationPatient (row_x, row_y, row_z, col_x, col_y, col_z)
    image_orientation: Option<[f64; 6]>,
    /// PixelSpacing (row_spacing, col_spacing)
    pixel_spacing: Option<(f64, f64)>,
    /// Image dimensions (rows, columns)
    image_dimensions: Option<(u32, u32)>,
    orientation: Option<&'static str>,
}

/// Series metadata collected from first file
struct SeriesMetadata {
    description: Option<String>,
    modality: Option<String>,
    series_number: Option<i32>,
    orientation: Option<&'static str>,
    frame_of_reference_uid: Option<String>,
    study_instance_uid: Option<String>,
}

/// Group DICOM files by series and return sorted SeriesInfo for each
pub fn group_files_by_series(files: Vec<DicomFile>) -> Vec<SeriesInfo> {
    // Group by series UID
    let mut series_map: HashMap<String, Vec<SortableFile>> = HashMap::new();
    let mut series_info_map: HashMap<String, SeriesMetadata> = HashMap::new();
    let mut skipped_non_image = 0;

    for file in files {
        // Skip non-image SOP classes (Raw Data, Presentation States, SR, etc.)
        if !file.is_image_storage() {
            skipped_non_image += 1;
            continue;
        }

        let series_uid = match file.series_instance_uid() {
            Some(uid) => uid,
            None => continue, // Skip files without series UID
        };

        // Detect orientation from DICOM ImageOrientationPatient tag
        let orientation = file.detect_orientation();

        // Store series metadata from first file
        series_info_map
            .entry(series_uid.clone())
            .or_insert_with(|| SeriesMetadata {
                description: file.series_description(),
                modality: file.modality(),
                series_number: file.series_number(),
                orientation,
                frame_of_reference_uid: file.frame_of_reference_uid(),
                study_instance_uid: file.study_instance_uid(),
            });

        // Extract sorting keys and geometry for center position computation
        let sortable = SortableFile {
            path: PathBuf::from(&file.path),
            instance_number: file.instance_number(),
            slice_location: file.slice_location(),
            image_position: file.image_position_patient(),
            image_orientation: file.image_orientation_patient(),
            pixel_spacing: file.pixel_spacing_with_fallback(),
            image_dimensions: match (file.rows(), file.columns()) {
                (Some(r), Some(c)) => Some((r, c)),
                _ => None,
            },
            orientation,
        };

        series_map.entry(series_uid).or_default().push(sortable);
    }

    // Sort files within each series and create SeriesInfo
    let mut result: Vec<SeriesInfo> = series_map
        .into_iter()
        .map(|(series_uid, mut files)| {
            // Sort by instance number, then slice location, then position Z, then filename
            files.sort_by(|a, b| {
                // Primary: instance number
                if let (Some(a_num), Some(b_num)) = (a.instance_number, b.instance_number) {
                    return a_num.cmp(&b_num);
                }
                // Fallback: slice location
                if let (Some(a_loc), Some(b_loc)) = (a.slice_location, b.slice_location) {
                    return a_loc
                        .partial_cmp(&b_loc)
                        .unwrap_or(std::cmp::Ordering::Equal);
                }
                // Fallback: image position Z
                let a_z = a.image_position.map(|(_, _, z)| z);
                let b_z = b.image_position.map(|(_, _, z)| z);
                if let (Some(az), Some(bz)) = (a_z, b_z) {
                    return az.partial_cmp(&bz).unwrap_or(std::cmp::Ordering::Equal);
                }
                // Last resort: filename
                a.path.cmp(&b.path)
            });

            let metadata = series_info_map
                .remove(&series_uid)
                .unwrap_or(SeriesMetadata {
                    description: None,
                    modality: None,
                    series_number: None,
                    orientation: None,
                    frame_of_reference_uid: None,
                    study_instance_uid: None,
                });

            let display_name = parse_series_name_with_orientation(
                metadata.description.as_deref().unwrap_or(""),
                metadata.modality.as_deref(),
                metadata.orientation,
            );

            // Compute slice locations at IMAGE CENTER for proper sync with MPR
            // For gantry-tilted acquisitions, corner position differs from center position.
            // Using center ensures tilted regular series sync correctly with untilted MPR.
            let slice_locations: Vec<Option<f64>> = files
                .iter()
                .map(|f| {
                    // Compute center position if we have all geometry data
                    let center_coord = match (
                        f.image_position,
                        f.image_orientation,
                        f.pixel_spacing,
                        f.image_dimensions,
                    ) {
                        (
                            Some((x, y, z)),
                            Some(iop),
                            Some((row_sp, col_sp)),
                            Some((rows, cols)),
                        ) => {
                            // ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z]
                            // row direction points along columns (horizontal, to the right)
                            // col direction points along rows (vertical, downward)
                            let row_dir = (iop[0], iop[1], iop[2]);
                            let col_dir = (iop[3], iop[4], iop[5]);

                            // Move from corner to center:
                            // - Half columns along row direction (using col_spacing)
                            // - Half rows along col direction (using row_spacing)
                            let half_cols = cols as f64 / 2.0;
                            let half_rows = rows as f64 / 2.0;

                            let center_x =
                                x + half_cols * col_sp * row_dir.0 + half_rows * row_sp * col_dir.0;
                            let center_y =
                                y + half_cols * col_sp * row_dir.1 + half_rows * row_sp * col_dir.1;
                            let center_z =
                                z + half_cols * col_sp * row_dir.2 + half_rows * row_sp * col_dir.2;

                            Some((center_x, center_y, center_z))
                        }
                        _ => None, // Fall back to corner position if geometry incomplete
                    };

                    // Use center coordinate based on orientation, fall back to corner or SliceLocation
                    match (center_coord, f.image_position, f.orientation) {
                        // Center position available - use appropriate coordinate
                        (Some((cx, _, _)), _, Some("Sag")) => Some(cx),
                        (Some((_, cy, _)), _, Some("Cor")) => Some(cy),
                        (Some((_, _, cz)), _, Some("Ax")) => Some(cz),
                        (Some((_, _, cz)), _, _) => Some(cz), // Default to Z for unknown orientation

                        // Fall back to corner position (ImagePositionPatient)
                        (None, Some((x, _, _)), Some("Sag")) => Some(x),
                        (None, Some((_, y, _)), Some("Cor")) => Some(y),
                        (None, Some((_, _, z)), Some("Ax")) => Some(z),
                        (None, Some((_, _, z)), _) => Some(z),

                        // Last resort: DICOM SliceLocation tag
                        (None, None, _) => f.slice_location,
                    }
                })
                .collect();

            SeriesInfo {
                series_uid,
                description: metadata.description,
                display_name,
                modality: metadata.modality,
                series_number: metadata.series_number,
                files: files.into_iter().map(|f| f.path).collect(),
                frame_of_reference_uid: metadata.frame_of_reference_uid,
                study_instance_uid: metadata.study_instance_uid,
                orientation: metadata.orientation,
                slice_locations,
            }
        })
        .collect();

    // Sort series by series number
    result.sort_by(|a, b| match (a.series_number, b.series_number) {
        (Some(a_num), Some(b_num)) => a_num.cmp(&b_num),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.series_uid.cmp(&b.series_uid),
    });

    if skipped_non_image > 0 {
        tracing::debug!(
            "Skipped {} non-image files (Raw Data, PR, SR, etc.)",
            skipped_non_image
        );
    }

    result
}

/// Get the middle file path from a series (for thumbnail)
pub fn get_middle_file(files: &[PathBuf]) -> Option<&Path> {
    if files.is_empty() {
        return None;
    }
    let middle_idx = files.len() / 2;
    files.get(middle_idx).map(|p| p.as_path())
}

/// Parse series description into a readable display name
pub fn parse_series_name(description: &str, modality: Option<&str>) -> String {
    parse_series_name_with_orientation(description, modality, None)
}

/// Parse series description with DICOM orientation fallback
pub fn parse_series_name_with_orientation(
    description: &str,
    modality: Option<&str>,
    dicom_orientation: Option<&str>,
) -> String {
    if description.is_empty() {
        // Even with empty description, try to build name from orientation
        if let Some(orient) = dicom_orientation {
            let orient_full = short_to_full_orientation(orient);
            return format!("{} {}", modality.unwrap_or("Series"), orient_full);
        }
        return modality.unwrap_or("Series").to_string();
    }

    let desc_upper = description.to_uppercase();

    // For CT, just clean up the description
    if modality == Some("CT") {
        return clean_series_name(description);
    }

    // For MR, parse sequence components
    if modality == Some("MR") {
        return parse_mr_sequence_name(&desc_upper, description, dicom_orientation);
    }

    // Default: clean up the description
    clean_series_name(description)
}

/// Parse MR sequence name into readable format
fn parse_mr_sequence_name(
    desc_upper: &str,
    original: &str,
    dicom_orientation: Option<&str>,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Detect weighting
    let weighting = detect_weighting(desc_upper);
    if let Some(w) = weighting {
        parts.push(w.to_string());
    }

    // Detect special sequences (before orientation)
    if desc_upper.contains("FLAIR") {
        parts.push("FLAIR".to_string());
    }
    if (desc_upper.contains("DWI") || desc_upper.contains("DIFF") || desc_upper.contains("EP2D_DIFF"))
        && !parts.iter().any(|p| p == "DWI")
    {
        parts.push("DWI".to_string());
    }
    if desc_upper.contains("ADC") {
        parts.push("ADC".to_string());
    }
    if desc_upper.contains("SWI") {
        parts.push("SWI".to_string());
    }

    // Detect 3D
    if desc_upper.contains("3D")
        || desc_upper.contains("MPRAGE")
        || desc_upper.contains("SPACE")
        || desc_upper.contains("CUBE")
        || desc_upper.contains("VISTA")
        || desc_upper.contains("VIBE")
        || desc_upper.contains("BRAVO")
    {
        parts.push("3D".to_string());
    }

    // Detect fat suppression
    if desc_upper.contains("STIR") {
        parts.push("STIR".to_string());
    } else if desc_upper.contains("SPAIR") || desc_upper.contains("SPIR") {
        parts.push("FS".to_string());
    } else if desc_upper.contains("DIXON") {
        parts.push("Dixon".to_string());
    } else if desc_upper.contains("_FS") || desc_upper.contains("FS_") || desc_upper.ends_with("FS")
    {
        parts.push("FS".to_string());
    }

    // Detect contrast
    if desc_upper.contains("+C")
        || desc_upper.contains("_C_")
        || desc_upper.contains("POST")
        || desc_upper.contains(" GD")
        || desc_upper.contains("_GD")
        || desc_upper.contains("GAD")
    {
        parts.push("+C".to_string());
    }

    // Detect b-value for DWI
    if let Some(b_value) = extract_b_value(desc_upper) {
        parts.push(format!("b{}", b_value));
    }

    // Detect orientation - try from description first, then fall back to DICOM tag
    let orientation = detect_orientation(desc_upper).or(dicom_orientation);

    // Add orientation at the end with full name
    if let Some(o) = orientation {
        parts.push(short_to_full_orientation(o).to_string());
    }

    // If we found components, join them
    if !parts.is_empty() {
        return parts.join(" ");
    }

    // Fallback: clean the original name
    clean_series_name(original)
}

/// Convert short orientation to full name
fn short_to_full_orientation(short: &str) -> &'static str {
    match short {
        "Ax" => "Axial",
        "Cor" => "Coronal",
        "Sag" => "Sagittal",
        _ => "Unknown",
    }
}

/// Detect MR weighting from description
fn detect_weighting(desc: &str) -> Option<&'static str> {
    // Check for specific patterns
    if desc.contains("T1W")
        || desc.contains("T1_")
        || desc.contains("_T1")
        || desc.starts_with("T1")
    {
        return Some("T1");
    }
    if desc.contains("T2*") || desc.contains("T2STAR") {
        return Some("T2*");
    }
    if desc.contains("T2W")
        || desc.contains("T2_")
        || desc.contains("_T2")
        || desc.starts_with("T2")
    {
        return Some("T2");
    }
    if desc.contains("PDW") || desc.contains("PD_") || desc.contains("_PD") {
        return Some("PD");
    }
    None
}

/// Detect orientation from description
fn detect_orientation(desc: &str) -> Option<&'static str> {
    // Axial patterns
    if desc.contains("AX") || desc.contains("TRA") || desc.contains("TRANSV") {
        return Some("Ax");
    }
    // Coronal patterns
    if desc.contains("COR") {
        return Some("Cor");
    }
    // Sagittal patterns - check for various forms
    if desc.contains("SAGITTAL")
        || desc.contains("SAG_")
        || desc.contains("_SAG")
        || desc.contains(" SAG")
        || desc.ends_with("SAG")
        || desc.starts_with("SAG")
    {
        return Some("Sag");
    }
    None
}

/// Extract b-value from DWI description
fn extract_b_value(desc: &str) -> Option<i32> {
    // Look for patterns like "B1000", "b=1000", "b_1000"
    let patterns = ["B", "b=", "b_"];
    for pattern in patterns {
        if let Some(idx) = desc.find(pattern) {
            let start = idx + pattern.len();
            let num_str: String = desc[start..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(val) = num_str.parse::<i32>() {
                if val > 0 && val <= 3000 {
                    return Some(val);
                }
            }
        }
    }
    None
}

/// Clean up a series name by removing underscores and excessive formatting
fn clean_series_name(name: &str) -> String {
    // Replace underscores with spaces
    let cleaned = name.replace('_', " ");
    // Remove multiple spaces
    let cleaned: String = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    // Capitalize first letter of each word
    cleaned
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(|c| c.to_lowercase()))
                    .collect(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mr_names() {
        assert_eq!(parse_series_name("T2_TSE_AX", Some("MR")), "T2 Axial");
        assert_eq!(
            parse_series_name("T1_MPRAGE_SAG", Some("MR")),
            "T1 3D Sagittal"
        );
        assert_eq!(
            parse_series_name("T2_FLAIR_COR", Some("MR")),
            "T2 FLAIR Coronal"
        );
        assert_eq!(
            parse_series_name("ep2d_diff_b1000", Some("MR")),
            "DWI b1000"
        );
        assert_eq!(
            parse_series_name("T1_SE_FS_AX_+C", Some("MR")),
            "T1 FS +C Axial"
        );
    }

    #[test]
    fn test_parse_mr_names_with_dicom_orientation() {
        // When description has no orientation, use DICOM fallback
        assert_eq!(
            parse_series_name_with_orientation("T2_TSE", Some("MR"), Some("Sag")),
            "T2 Sagittal"
        );
        // When description has orientation, it takes precedence
        assert_eq!(
            parse_series_name_with_orientation("T2_TSE_AX", Some("MR"), Some("Sag")),
            "T2 Axial"
        );
    }

    #[test]
    fn test_clean_series_name() {
        assert_eq!(clean_series_name("HEAD_CT_ROUTINE"), "Head Ct Routine");
        assert_eq!(clean_series_name("brain_mri"), "Brain Mri");
    }

    #[test]
    fn test_extract_b_value() {
        assert_eq!(extract_b_value("DWI_B1000"), Some(1000));
        assert_eq!(extract_b_value("ep2d_diff_b=800"), Some(800));
        assert_eq!(extract_b_value("T2_TSE"), None);
    }

    #[test]
    fn test_short_to_full_orientation() {
        assert_eq!(short_to_full_orientation("Ax"), "Axial");
        assert_eq!(short_to_full_orientation("Cor"), "Coronal");
        assert_eq!(short_to_full_orientation("Sag"), "Sagittal");
    }
}
