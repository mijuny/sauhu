//! DICOM anonymization module
//!
//! Provides functionality to anonymize DICOM files by removing PHI
//! and replacing UIDs with new compliant ones.

use anyhow::{Context, Result};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::dictionary_std::tags;
use dicom::object::open_file;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use uuid::Uuid;

/// Fire-themed Finnish first names (Tulenkantajat vocabulary)
const FIRE_FIRST_NAMES: &[&str] = &[
    "Kipinä",  // spark
    "Roihu",   // blaze
    "Liekki",  // flame
    "Nuotio",  // campfire
    "Hiillos", // embers
    "Savu",    // smoke
    "Polte",   // burn
    "Tulikki", // fire maiden
    "Palo",    // fire
    "Kekäle",  // firebrand
    "Korento", // torch
    "Valko",   // glow
    "Hehku",   // heat glow
    "Kytö",    // smolder
    "Arina",   // hearth
];

/// Finnish nature-themed surnames
const NATURE_SURNAMES: &[&str] = &[
    "Koivu",  // birch
    "Mänty",  // pine
    "Järvi",  // lake
    "Niemi",  // peninsula
    "Vuori",  // mountain
    "Koski",  // rapids
    "Ranta",  // shore
    "Lehto",  // grove
    "Kallio", // rock
    "Salo",   // wilderness
    "Lampi",  // pond
    "Harju",  // ridge
    "Mäki",   // hill
    "Suo",    // swamp
    "Saari",  // island
];

/// UID generator that maintains consistent mapping
pub struct UidGenerator {
    /// Mapping from original UID to anonymized UID
    uid_map: HashMap<String, String>,
}

impl UidGenerator {
    pub fn new() -> Self {
        Self {
            uid_map: HashMap::new(),
        }
    }

    /// Get or create anonymized UID for an original UID.
    /// Uses 2.25 root with UUID-derived suffix (ITI TF compliant).
    pub fn get_uid(&mut self, original: &str) -> String {
        if let Some(anon) = self.uid_map.get(original) {
            return anon.clone();
        }

        // Generate new UID using UUID converted to decimal
        // Format: 2.25.<128-bit UUID as decimal>
        let uuid = Uuid::new_v4();
        let uuid_bytes = uuid.as_bytes();

        // Convert 128-bit UUID to decimal representation
        let value = u128::from_be_bytes(*uuid_bytes);
        let decimal_str = value.to_string();

        // Construct UID with 2.25 root
        let uid = format!("2.25.{}", decimal_str);

        // DICOM UIDs are max 64 chars - this should always fit
        // (2.25. = 5 chars, max u128 decimal = 39 chars = 44 total)
        debug_assert!(uid.len() <= 64);

        self.uid_map.insert(original.to_string(), uid.clone());
        uid
    }

    /// Get the UID mapping (original -> anonymized)
    #[allow(dead_code)]
    pub fn get_mapping(&self) -> &HashMap<String, String> {
        &self.uid_map
    }
}

impl Default for UidGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for anonymization
#[derive(Debug, Clone)]
pub struct AnonymizeConfig {
    /// Patient ID prefix (e.g., "TULI")
    pub patient_id_prefix: String,
    /// Accession number prefix (e.g., "SAU")
    pub accession_prefix: String,
    /// Whether to keep patient sex
    #[allow(dead_code)]
    pub keep_patient_sex: bool,
}

impl Default for AnonymizeConfig {
    fn default() -> Self {
        Self {
            patient_id_prefix: "TULI".to_string(),
            accession_prefix: "SAU".to_string(),
            keep_patient_sex: true,
        }
    }
}

/// Progress information during anonymization
#[derive(Debug, Clone, Default)]
pub struct AnonymizeProgress {
    pub completed: u32,
    pub total: u32,
    pub current_file: Option<String>,
    pub is_complete: bool,
    #[allow(dead_code)]
    pub error: Option<String>,
}

/// Patient info for anonymization result
#[derive(Debug, Clone)]
pub struct AnonymizedPatientInfo {
    pub new_patient_name: String,
    pub new_patient_id: String,
}

/// Result of anonymization operation
#[derive(Debug)]
pub struct AnonymizeResult {
    pub files_processed: u32,
    pub files_failed: u32,
    /// Mapping of original study UIDs to new UIDs
    pub study_uid_map: HashMap<String, String>,
    /// Mapping of original series UIDs to new UIDs
    pub series_uid_map: HashMap<String, String>,
    /// Mapping of original study UIDs to new accession numbers
    pub accession_map: HashMap<String, String>,
    /// Mapping of original study UIDs to new patient info
    pub study_patient_map: HashMap<String, AnonymizedPatientInfo>,
    /// New patient name used (comma-separated for multiple patients)
    pub new_patient_name: String,
    /// New patient ID used (first patient's ID)
    #[allow(dead_code)]
    pub new_patient_id: String,
}

/// Generate a random fire-themed Finnish name
pub fn generate_random_name() -> String {
    let mut rng = rand::thread_rng();
    let first = FIRE_FIRST_NAMES.choose(&mut rng).unwrap_or(&"Kipinä");
    let last = NATURE_SURNAMES.choose(&mut rng).unwrap_or(&"Koivu");
    format!("{}^{}", last, first) // DICOM format: Family^Given
}

/// Generate a patient ID with prefix and number
pub fn generate_patient_id(prefix: &str, number: u32) -> String {
    format!("{}{:03}", prefix, number)
}

/// Generate an accession number with prefix and number
pub fn generate_accession_number(prefix: &str, number: u32) -> String {
    format!("{}{:05}", prefix, number)
}

/// Tags to completely remove (set to empty)
fn get_tags_to_clear() -> Vec<Tag> {
    vec![
        tags::INSTITUTION_NAME,
        tags::INSTITUTION_ADDRESS,
        tags::REFERRING_PHYSICIAN_NAME,
        tags::PERFORMING_PHYSICIAN_NAME,
        tags::OPERATORS_NAME,
        tags::PATIENT_ADDRESS,
        tags::PATIENT_TELEPHONE_NUMBERS,
        Tag(0x0010, 0x4000), // Patient Comments
        Tag(0x0010, 0x1000), // Other Patient IDs
        Tag(0x0010, 0x1001), // Other Patient Names
        Tag(0x0008, 0x0092), // Referring Physician's Address
        Tag(0x0008, 0x0094), // Referring Physician's Telephone Numbers
        Tag(0x0008, 0x1048), // Physician(s) of Record
        Tag(0x0008, 0x1050), // Performing Physician's Name (already in list)
        Tag(0x0032, 0x1032), // Requesting Physician
        Tag(0x0040, 0x0006), // Scheduled Performing Physician's Name
        // Note: Accession Number is now generated, not cleared
        Tag(0x0020, 0x0010), // Study ID
    ]
}

/// Anonymize a single DICOM file
pub fn anonymize_file(
    path: &Path,
    uid_generator: &mut UidGenerator,
    new_patient_name: &str,
    new_patient_id: &str,
    new_accession_number: &str,
) -> Result<AnonymizedFileInfo> {
    // Open the file
    let file_obj =
        open_file(path).with_context(|| format!("Failed to open DICOM file: {:?}", path))?;

    // Get original UIDs before modification
    let original_study_uid = file_obj
        .element(tags::STUDY_INSTANCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let original_series_uid = file_obj
        .element(tags::SERIES_INSTANCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let original_sop_uid = file_obj
        .element(tags::SOP_INSTANCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let original_frame_of_ref = file_obj
        .element(tags::FRAME_OF_REFERENCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.to_string());

    // Generate new UIDs (consistent for same original UID)
    let new_study_uid = uid_generator.get_uid(&original_study_uid);
    let new_series_uid = uid_generator.get_uid(&original_series_uid);
    let new_sop_uid = uid_generator.get_uid(&original_sop_uid);
    let new_frame_of_ref = original_frame_of_ref
        .as_ref()
        .map(|uid| uid_generator.get_uid(uid));

    // Get transfer syntax and SOP class for file metadata
    let transfer_syntax = file_obj.meta().transfer_syntax().to_string();
    let sop_class_uid = file_obj.meta().media_storage_sop_class_uid().to_string();

    // Convert to in-memory object for modification
    let mut obj = file_obj.into_inner();

    // Replace patient identifying information
    obj.put(DataElement::new(
        tags::PATIENT_NAME,
        VR::PN,
        PrimitiveValue::from(new_patient_name),
    ));
    obj.put(DataElement::new(
        tags::PATIENT_ID,
        VR::LO,
        PrimitiveValue::from(new_patient_id),
    ));
    obj.put(DataElement::new(
        tags::PATIENT_BIRTH_DATE,
        VR::DA,
        PrimitiveValue::from(""),
    ));

    // Replace UIDs
    obj.put(DataElement::new(
        tags::STUDY_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(new_study_uid.as_str()),
    ));
    obj.put(DataElement::new(
        tags::SERIES_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(new_series_uid.as_str()),
    ));
    obj.put(DataElement::new(
        tags::SOP_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(new_sop_uid.as_str()),
    ));

    if let Some(ref new_for) = new_frame_of_ref {
        obj.put(DataElement::new(
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            PrimitiveValue::from(new_for.as_str()),
        ));
    }

    // Set new accession number
    obj.put(DataElement::new(
        tags::ACCESSION_NUMBER,
        VR::SH,
        PrimitiveValue::from(new_accession_number),
    ));

    // Clear sensitive tags (set to empty)
    for tag in get_tags_to_clear() {
        if obj.element(tag).is_ok() {
            // Get the VR of the existing element to preserve it
            let vr = obj.element(tag).map(|e| e.vr()).unwrap_or(VR::LO);
            obj.put(DataElement::new(tag, vr, PrimitiveValue::from("")));
        }
    }

    // Create file object with updated metadata
    let file_obj = obj.with_meta(
        dicom::object::FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(&sop_class_uid)
            .media_storage_sop_instance_uid(&new_sop_uid)
            .transfer_syntax(&transfer_syntax),
    )?;

    // Write back to the same file
    file_obj
        .write_to_file(path)
        .with_context(|| format!("Failed to write anonymized file: {:?}", path))?;

    Ok(AnonymizedFileInfo {
        path: path.to_path_buf(),
        original_study_uid,
        new_study_uid,
        original_series_uid,
        new_series_uid,
        original_sop_uid,
        new_sop_uid,
        new_accession_number: new_accession_number.to_string(),
    })
}

/// Information about an anonymized file
#[derive(Debug)]
#[allow(dead_code)]
pub struct AnonymizedFileInfo {
    pub path: PathBuf,
    pub original_study_uid: String,
    pub new_study_uid: String,
    pub original_series_uid: String,
    pub new_series_uid: String,
    pub original_sop_uid: String,
    pub new_sop_uid: String,
    pub new_accession_number: String,
}

/// Anonymize all DICOM files for a patient
pub fn anonymize_patient(
    file_paths: Vec<PathBuf>,
    config: &AnonymizeConfig,
    patient_number: u32,
    start_accession_number: u32,
    progress_tx: Option<mpsc::Sender<AnonymizeProgress>>,
) -> Result<AnonymizeResult> {
    let mut uid_generator = UidGenerator::new();
    let new_patient_name = generate_random_name();
    let new_patient_id = generate_patient_id(&config.patient_id_prefix, patient_number);

    let total = file_paths.len() as u32;
    let mut files_processed = 0u32;
    let mut files_failed = 0u32;

    // Track study and series UID mappings from actual file processing
    let mut study_uid_map: HashMap<String, String> = HashMap::new();
    let mut series_uid_map: HashMap<String, String> = HashMap::new();

    // Track accession numbers per study (original study UID -> new accession number)
    let mut accession_map: HashMap<String, String> = HashMap::new();
    let mut next_accession = start_accession_number;

    // Send initial progress
    if let Some(ref tx) = progress_tx {
        let _ = tx.send(AnonymizeProgress {
            completed: 0,
            total,
            current_file: None,
            is_complete: false,
            error: None,
        });
    }

    for (i, path) in file_paths.iter().enumerate() {
        // Send progress update
        if let Some(ref tx) = progress_tx {
            let _ = tx.send(AnonymizeProgress {
                completed: i as u32,
                total,
                current_file: Some(
                    path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                ),
                is_complete: false,
                error: None,
            });
        }

        // First, read the file to get its study UID for accession number lookup
        let study_uid = if let Ok(file_obj) = open_file(path) {
            file_obj
                .element(tags::STUDY_INSTANCE_UID)
                .ok()
                .and_then(|e| e.to_str().ok())
                .map(|s| s.to_string())
                .unwrap_or_default()
        } else {
            String::new()
        };

        // Get or create accession number for this study
        let accession_number = if !study_uid.is_empty() {
            accession_map
                .entry(study_uid.clone())
                .or_insert_with(|| {
                    let acc = generate_accession_number(&config.accession_prefix, next_accession);
                    next_accession += 1;
                    acc
                })
                .clone()
        } else {
            generate_accession_number(&config.accession_prefix, next_accession)
        };

        match anonymize_file(
            path,
            &mut uid_generator,
            &new_patient_name,
            &new_patient_id,
            &accession_number,
        ) {
            Ok(info) => {
                files_processed += 1;
                // Collect study and series UID mappings from the actual file
                if !info.original_study_uid.is_empty() {
                    study_uid_map.insert(info.original_study_uid.clone(), info.new_study_uid);
                    // Also store accession mapping
                    accession_map.entry(info.original_study_uid).or_insert_with(|| accession_number.clone());
                }
                if !info.original_series_uid.is_empty() {
                    series_uid_map.insert(info.original_series_uid, info.new_series_uid);
                }
                tracing::debug!("Anonymized: {:?}", path);
            }
            Err(e) => {
                files_failed += 1;
                tracing::warn!("Failed to anonymize {:?}: {}", path, e);
            }
        }
    }

    tracing::info!(
        "Anonymization complete: {} studies, {} series, {} files processed",
        study_uid_map.len(),
        series_uid_map.len(),
        files_processed
    );

    // Build study -> patient mapping (all studies in this call belong to the same patient)
    let patient_info = AnonymizedPatientInfo {
        new_patient_name: new_patient_name.clone(),
        new_patient_id: new_patient_id.clone(),
    };
    let study_patient_map: HashMap<String, AnonymizedPatientInfo> = study_uid_map
        .keys()
        .map(|study_uid| (study_uid.clone(), patient_info.clone()))
        .collect();

    // Send completion
    if let Some(ref tx) = progress_tx {
        let _ = tx.send(AnonymizeProgress {
            completed: total,
            total,
            current_file: None,
            is_complete: true,
            error: if files_failed > 0 {
                Some(format!("{} files failed to anonymize", files_failed))
            } else {
                None
            },
        });
    }

    Ok(AnonymizeResult {
        files_processed,
        files_failed,
        study_uid_map,
        series_uid_map,
        accession_map,
        study_patient_map,
        new_patient_name,
        new_patient_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uid_generator_consistency() {
        let mut gen = UidGenerator::new();
        let uid1 = gen.get_uid("1.2.3.4.5");
        let uid2 = gen.get_uid("1.2.3.4.5");
        assert_eq!(
            uid1, uid2,
            "Same original UID should produce same anonymized UID"
        );

        let uid3 = gen.get_uid("1.2.3.4.6");
        assert_ne!(
            uid1, uid3,
            "Different original UIDs should produce different anonymized UIDs"
        );
    }

    #[test]
    fn test_uid_format() {
        let mut gen = UidGenerator::new();
        let uid = gen.get_uid("original");
        assert!(uid.starts_with("2.25."), "UID should start with 2.25 root");
        assert!(uid.len() <= 64, "UID should be max 64 chars");
    }

    #[test]
    fn test_random_name_generation() {
        let name1 = generate_random_name();
        let name2 = generate_random_name();

        // Names should be in DICOM format (Family^Given)
        assert!(name1.contains('^'), "Name should be in DICOM format");

        // Names could be same by chance, but format should be consistent
        assert!(name1.split('^').count() == 2);
    }

    #[test]
    fn test_patient_id_generation() {
        let id = generate_patient_id("TULI", 1);
        assert_eq!(id, "TULI001");

        let id2 = generate_patient_id("TULI", 42);
        assert_eq!(id2, "TULI042");
    }
}
