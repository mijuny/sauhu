//! Series utilities for grouping, sorting, and naming

use crate::dicom::{AnatomicalPlane, DicomFile};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Synchronization info for a viewport's series
#[derive(Debug, Clone)]
pub struct SyncInfo {
    pub frame_of_reference_uid: Option<String>,
    pub study_instance_uid: Option<String>,
    pub orientation: Option<AnatomicalPlane>,
    pub slice_locations: Vec<Option<f64>>,
    /// `(location, original_index)` pairs sorted by location, with `None`
    /// entries filtered out. Cached once at construction so hot viewport-sync
    /// paths can binary-search instead of scanning `slice_locations` each time.
    pub sorted_locations: Vec<(f64, usize)>,
}

impl SyncInfo {
    pub fn from_series_info(info: &SeriesInfo) -> Self {
        let sorted_locations = build_sorted_locations(&info.slice_locations);
        Self {
            frame_of_reference_uid: info.frame_of_reference_uid.clone(),
            study_instance_uid: info.study_instance_uid.clone(),
            orientation: info.orientation,
            slice_locations: info.slice_locations.clone(),
            sorted_locations,
        }
    }

    /// Index of the slice whose location is closest to `target`.
    ///
    /// Uses the cached `sorted_locations` table for an O(log N) lookup; the
    /// old linear scan had to walk every slice on every viewport sync event,
    /// which became visible with 8-viewport layouts + 500-slice series.
    pub fn closest_index(&self, target: f64) -> usize {
        closest_index_sorted(&self.sorted_locations, target)
    }
}

fn build_sorted_locations(slice_locations: &[Option<f64>]) -> Vec<(f64, usize)> {
    let mut sorted: Vec<(f64, usize)> = slice_locations
        .iter()
        .enumerate()
        .filter_map(|(i, loc)| loc.map(|l| (l, i)))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

pub(crate) fn closest_index_sorted(sorted: &[(f64, usize)], target: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    // partition_point returns the first index whose location is strictly > target.
    let upper = sorted.partition_point(|&(loc, _)| loc <= target);
    if upper == 0 {
        return sorted[0].1;
    }
    if upper == sorted.len() {
        return sorted[sorted.len() - 1].1;
    }
    let (lo_loc, lo_idx) = sorted[upper - 1];
    let (hi_loc, hi_idx) = sorted[upper];
    if (target - lo_loc).abs() <= (hi_loc - target).abs() {
        lo_idx
    } else {
        hi_idx
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
    #[allow(dead_code)]
    pub modality: Option<String>,
    /// Series number
    pub series_number: Option<i32>,
    /// Sorted file paths
    pub files: Vec<PathBuf>,
    /// Frame of Reference UID (for sync grouping)
    pub frame_of_reference_uid: Option<String>,
    /// Study Instance UID (for sync grouping)
    pub study_instance_uid: Option<String>,
    /// Orientation from DICOM tag
    pub orientation: Option<AnatomicalPlane>,
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
    orientation: Option<AnatomicalPlane>,
}

/// Series metadata collected from first file
struct SeriesMetadata {
    description: Option<String>,
    modality: Option<String>,
    series_number: Option<i32>,
    orientation: Option<AnatomicalPlane>,
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
            let slice_locations: Vec<Option<f64>> =
                files.iter().map(compute_slice_location).collect();

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

/// Compute the slice location for a file using image center position.
///
/// For gantry-tilted acquisitions, the corner position (ImagePositionPatient)
/// differs from the center position. Using center ensures tilted regular series
/// sync correctly with untilted MPR.
fn compute_slice_location(f: &SortableFile) -> Option<f64> {
    // Compute center position if we have all geometry data
    let center_coord = match (
        f.image_position,
        f.image_orientation,
        f.pixel_spacing,
        f.image_dimensions,
    ) {
        (Some((x, y, z)), Some(iop), Some((row_sp, col_sp)), Some((rows, cols))) => {
            let row_dir = (iop[0], iop[1], iop[2]);
            let col_dir = (iop[3], iop[4], iop[5]);
            let half_cols = cols as f64 / 2.0;
            let half_rows = rows as f64 / 2.0;

            let center_x = x + half_cols * col_sp * row_dir.0 + half_rows * row_sp * col_dir.0;
            let center_y = y + half_cols * col_sp * row_dir.1 + half_rows * row_sp * col_dir.1;
            let center_z = z + half_cols * col_sp * row_dir.2 + half_rows * row_sp * col_dir.2;

            Some((center_x, center_y, center_z))
        }
        _ => None,
    };

    match (center_coord, f.image_position, f.orientation) {
        (Some((cx, _, _)), _, Some(AnatomicalPlane::Sagittal)) => Some(cx),
        (Some((_, cy, _)), _, Some(AnatomicalPlane::Coronal)) => Some(cy),
        (Some((_, _, cz)), _, Some(AnatomicalPlane::Axial)) => Some(cz),
        (Some((_, _, cz)), _, _) => Some(cz),

        (None, Some((x, _, _)), Some(AnatomicalPlane::Sagittal)) => Some(x),
        (None, Some((_, y, _)), Some(AnatomicalPlane::Coronal)) => Some(y),
        (None, Some((_, _, z)), Some(AnatomicalPlane::Axial)) => Some(z),
        (None, Some((_, _, z)), _) => Some(z),

        (None, None, _) => f.slice_location,
    }
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
#[allow(dead_code)]
pub fn parse_series_name(description: &str, modality: Option<&str>) -> String {
    parse_series_name_with_orientation(description, modality, None)
}

/// Parse series description with DICOM orientation fallback
pub fn parse_series_name_with_orientation(
    description: &str,
    modality: Option<&str>,
    dicom_orientation: Option<AnatomicalPlane>,
) -> String {
    if description.is_empty() {
        // Even with empty description, try to build name from orientation
        if let Some(orient) = dicom_orientation {
            return format!("{} {}", modality.unwrap_or("Series"), orient);
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
    dicom_orientation: Option<AnatomicalPlane>,
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
        parts.push(o.full_name().to_string());
    }

    // If we found components, join them
    if !parts.is_empty() {
        return parts.join(" ");
    }

    // Fallback: clean the original name
    clean_series_name(original)
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
fn detect_orientation(desc: &str) -> Option<AnatomicalPlane> {
    // Axial patterns
    if desc.contains("AX") || desc.contains("TRA") || desc.contains("TRANSV") {
        return Some(AnatomicalPlane::Axial);
    }
    // Coronal patterns
    if desc.contains("COR") {
        return Some(AnatomicalPlane::Coronal);
    }
    // Sagittal patterns - check for various forms
    if desc.contains("SAGITTAL")
        || desc.contains("SAG_")
        || desc.contains("_SAG")
        || desc.contains(" SAG")
        || desc.ends_with("SAG")
        || desc.starts_with("SAG")
    {
        return Some(AnatomicalPlane::Sagittal);
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

    fn naive_closest(locations: &[Option<f64>], target: f64) -> usize {
        locations
            .iter()
            .enumerate()
            .filter_map(|(i, loc)| loc.map(|l| (i, (l - target).abs())))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn closest_index_sorted_matches_naive_scan() {
        // A deliberately non-monotonic, sparsely populated table: shuffled
        // locations with a few None gaps, to match the shape we see from
        // multi-orientation series.
        let locations = vec![
            Some(5.0),
            None,
            Some(-3.5),
            Some(12.0),
            Some(2.25),
            None,
            Some(7.75),
            Some(-10.0),
            Some(15.5),
        ];
        let sorted = build_sorted_locations(&locations);

        for target in [-20.0, -3.5, 0.0, 2.24, 2.26, 6.9, 12.0, 13.2, 1000.0] {
            let fast = closest_index_sorted(&sorted, target);
            let slow = naive_closest(&locations, target);
            let fast_loc = locations[fast].expect("fast returned a None slot");
            let slow_loc = locations[slow].expect("slow returned a None slot");
            // Distances must match even if indices tie-break differently.
            assert!(
                (fast_loc - target).abs() == (slow_loc - target).abs(),
                "target {target}: fast idx={fast} (loc={fast_loc}), slow idx={slow} (loc={slow_loc})"
            );
        }
    }

    #[test]
    fn closest_index_sorted_handles_empty_and_all_none() {
        let sorted = build_sorted_locations(&[]);
        assert_eq!(closest_index_sorted(&sorted, 0.0), 0);

        let sorted = build_sorted_locations(&[None, None, None]);
        assert_eq!(closest_index_sorted(&sorted, 42.0), 0);
    }

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
            parse_series_name_with_orientation("T2_TSE", Some("MR"), Some(AnatomicalPlane::Sagittal)),
            "T2 Sagittal"
        );
        // When description has orientation, it takes precedence
        assert_eq!(
            parse_series_name_with_orientation("T2_TSE_AX", Some("MR"), Some(AnatomicalPlane::Sagittal)),
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
    fn test_anatomical_plane_display() {
        assert_eq!(AnatomicalPlane::Axial.full_name(), "Axial");
        assert_eq!(AnatomicalPlane::Coronal.full_name(), "Coronal");
        assert_eq!(AnatomicalPlane::Sagittal.full_name(), "Sagittal");
        assert_eq!(AnatomicalPlane::Axial.abbrev(), "Ax");
        assert_eq!(AnatomicalPlane::from_abbrev("Cor"), Some(AnatomicalPlane::Coronal));
    }
}
