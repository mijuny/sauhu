//! Anonymization UI and logic

use super::{DatabaseWindow, PatientInfo};
use crate::db::{
    get_next_anon_accession_number, get_next_anon_patient_number, update_series_uid,
    update_study_accession, update_study_after_anonymization, Study,
};
use crate::dicom::{anonymize_patient, AnonymizeConfig, AnonymizeProgress, AnonymizeResult};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::PathBuf;
use walkdir::WalkDir;

/// Result from anonymization background worker
pub(super) enum AnonymizeProgressResult {
    Progress(AnonymizeProgress),
    Complete(AnonymizeResult),
    Error(String),
}

/// Info about a patient group to be anonymized
#[derive(Clone)]
pub(super) struct AnonymizePatientGroup {
    /// Original patient ID
    #[allow(dead_code)]
    pub(super) patient_id: String,
    /// Original patient name
    pub(super) patient_name: String,
    /// Studies belonging to this patient
    pub(super) studies: Vec<Study>,
    /// Number of DICOM files
    pub(super) num_files: u32,
}

/// State for anonymization operation
#[derive(Default)]
pub(super) enum AnonymizeState {
    #[default]
    Idle,
    /// Showing confirmation dialog for one or more patient groups
    Confirming {
        /// Patient groups to anonymize (grouped by original patient_id)
        patient_groups: Vec<AnonymizePatientGroup>,
        /// Total number of files across all groups
        total_files: u32,
    },
    /// Anonymization in progress
    Running {
        progress: AnonymizeProgress,
        /// Number of patient groups being processed
        num_patients: usize,
    },
    /// Anonymization completed
    #[allow(dead_code)]
    Complete {
        files_processed: u32,
        /// List of new patient names created
        new_names: Vec<String>,
    },
    /// Anonymization failed
    Error { message: String },
}


impl DatabaseWindow {
    /// Show anonymization confirmation dialog and progress
    pub(super) fn show_anonymization_ui(&mut self, ui: &mut egui::Ui, conn: &Connection) {
        // Track state transitions to avoid borrow conflicts
        let mut new_state: Option<AnonymizeState> = None;
        let mut start_anon_groups: Option<Vec<AnonymizePatientGroup>> = None;

        match &self.anonymize_state {
            AnonymizeState::Confirming {
                patient_groups,
                total_files,
            } => {
                ui.separator();

                // Build description based on number of patients
                let description = if patient_groups.len() == 1 {
                    let group = &patient_groups[0];
                    format!(
                        "Anonymize {} ({} studies, {} files)?",
                        group.patient_name,
                        group.studies.len(),
                        total_files
                    )
                } else {
                    format!(
                        "Anonymize {} patients ({} studies, {} files)?",
                        patient_groups.len(),
                        patient_groups
                            .iter()
                            .map(|g| g.studies.len())
                            .sum::<usize>(),
                        total_files
                    )
                };

                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::YELLOW, description);
                });

                // Show patient breakdown for multiple patients
                if patient_groups.len() > 1 {
                    ui.indent("patient_list", |ui| {
                        for group in patient_groups {
                            ui.label(format!(
                                "  {} - {} studies, {} files",
                                group.patient_name,
                                group.studies.len(),
                                group.num_files
                            ));
                        }
                    });
                }

                ui.label("This will permanently remove PHI from all DICOM files and replace UIDs.");
                ui.label("Studies from the same original patient will get the same anonymized ID.");

                ui.horizontal(|ui| {
                    if ui.button("Confirm").clicked() {
                        start_anon_groups = Some(patient_groups.clone());
                    }
                    if ui.button("Cancel").clicked() {
                        new_state = Some(AnonymizeState::Idle);
                    }
                });
            }
            AnonymizeState::Running {
                progress,
                num_patients,
            } => {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.spinner();
                    let patients_text = if *num_patients > 1 {
                        format!(" ({} patients)", num_patients)
                    } else {
                        String::new()
                    };
                    ui.label(format!(
                        "Anonymizing{} {}/{}",
                        patients_text, progress.completed, progress.total
                    ));
                    if let Some(ref file) = progress.current_file {
                        ui.label(format!("({})", file));
                    }
                });
            }
            AnonymizeState::Complete {
                files_processed,
                new_names,
            } => {
                ui.separator();
                let names_text = if new_names.len() == 1 {
                    format!("New name: {}", new_names[0])
                } else {
                    format!("New names: {}", new_names.join(", "))
                };
                ui.colored_label(
                    egui::Color32::GREEN,
                    format!("Anonymized {} files. {}", files_processed, names_text),
                );
                if ui.button("OK").clicked() {
                    new_state = Some(AnonymizeState::Idle);
                }
            }
            AnonymizeState::Error { message } => {
                ui.separator();
                ui.colored_label(egui::Color32::RED, message);
                if ui.button("OK").clicked() {
                    new_state = Some(AnonymizeState::Idle);
                }
            }
            AnonymizeState::Idle => {}
        }

        // Apply state transitions after match (avoids borrow conflicts)
        if let Some(groups) = start_anon_groups {
            self.start_anonymization_for_groups(conn, groups);
        } else if let Some(state) = new_state {
            self.anonymize_state = state;
        }
    }

    /// Start anonymization for multiple patient groups
    pub(super) fn start_anonymization_for_groups(
        &mut self,
        conn: &Connection,
        groups: Vec<AnonymizePatientGroup>,
    ) {
        if groups.is_empty() {
            self.anonymize_state = AnonymizeState::Error {
                message: "No patients to anonymize".to_string(),
            };
            return;
        }

        // Collect all DICOM file paths across all groups
        let mut all_file_paths = Vec::new();
        for group in &groups {
            for study in &group.studies {
                let study_path = PathBuf::from(&study.file_path);
                if study_path.is_dir() {
                    for entry in WalkDir::new(&study_path)
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_type().is_file())
                    {
                        let path = entry.path();
                        if path.extension().map(|e| e == "dcm").unwrap_or(false)
                            || path.extension().is_none()
                        {
                            all_file_paths.push(path.to_path_buf());
                        }
                    }
                }
            }
        }

        if all_file_paths.is_empty() {
            self.anonymize_state = AnonymizeState::Error {
                message: "No DICOM files found".to_string(),
            };
            return;
        }

        // Get next available patient number and accession number
        let start_patient_number = get_next_anon_patient_number(conn, "TULI").unwrap_or(1);
        let start_accession = get_next_anon_accession_number(conn, "SAU").unwrap_or(1);

        let (result_tx, result_rx) = std::sync::mpsc::channel();
        self.anonymize_rx = Some(result_rx);

        let config = AnonymizeConfig::default();
        let total_files = all_file_paths.len() as u32;
        let num_patients = groups.len();

        self.anonymize_state = AnonymizeState::Running {
            progress: AnonymizeProgress {
                completed: 0,
                total: total_files,
                current_file: None,
                is_complete: false,
                error: None,
            },
            num_patients,
        };

        // Spawn background thread
        std::thread::spawn(move || {
            // Create a progress channel that forwards to the result channel
            let (progress_tx, progress_rx) = std::sync::mpsc::channel::<AnonymizeProgress>();
            let result_tx_clone = result_tx.clone();

            // Spawn a thread to forward progress updates
            let forwarder = std::thread::spawn(move || {
                while let Ok(progress) = progress_rx.recv() {
                    if progress.is_complete {
                        break;
                    }
                    let _ = result_tx_clone.send(AnonymizeProgressResult::Progress(progress));
                }
            });

            // Process each patient group with a unique patient number
            // but track accession numbers across all groups
            let mut current_patient_number = start_patient_number;
            let mut current_accession = start_accession;
            let mut combined_result = AnonymizeResult {
                files_processed: 0,
                files_failed: 0,
                study_uid_map: HashMap::new(),
                series_uid_map: HashMap::new(),
                accession_map: HashMap::new(),
                study_patient_map: HashMap::new(),
                new_patient_name: String::new(),
                new_patient_id: String::new(),
            };
            let mut new_patient_names = Vec::new();

            for group in &groups {
                // Collect file paths for this patient group
                let mut group_file_paths = Vec::new();
                for study in &group.studies {
                    let study_path = PathBuf::from(&study.file_path);
                    if study_path.is_dir() {
                        for entry in WalkDir::new(&study_path)
                            .into_iter()
                            .filter_map(|e| e.ok())
                            .filter(|e| e.file_type().is_file())
                        {
                            let path = entry.path();
                            if path.extension().map(|e| e == "dcm").unwrap_or(false)
                                || path.extension().is_none()
                            {
                                group_file_paths.push(path.to_path_buf());
                            }
                        }
                    }
                }

                if group_file_paths.is_empty() {
                    continue;
                }

                // Anonymize this patient group
                match anonymize_patient(
                    group_file_paths,
                    &config,
                    current_patient_number,
                    current_accession,
                    Some(progress_tx.clone()),
                ) {
                    Ok(result) => {
                        // Merge results
                        combined_result.files_processed += result.files_processed;
                        combined_result.files_failed += result.files_failed;
                        combined_result.study_uid_map.extend(result.study_uid_map);
                        combined_result.series_uid_map.extend(result.series_uid_map);
                        combined_result
                            .accession_map
                            .extend(result.accession_map.clone());
                        combined_result
                            .study_patient_map
                            .extend(result.study_patient_map);
                        new_patient_names.push(result.new_patient_name.clone());

                        // Update accession number for next group
                        // (get max accession from this group + 1)
                        if let Some(max_acc) = result
                            .accession_map
                            .values()
                            .filter_map(|acc| {
                                acc.strip_prefix(&config.accession_prefix)
                                    .and_then(|n| n.parse::<u32>().ok())
                            })
                            .max()
                        {
                            current_accession = max_acc + 1;
                        }
                    }
                    Err(e) => {
                        if result_tx.send(AnonymizeProgressResult::Error(format!(
                            "Failed to anonymize patient {}: {}",
                            group.patient_name, e
                        ))).is_err() {
                            tracing::warn!("Anonymize channel closed before error delivered");
                        }
                        let _ = forwarder.join();
                        return;
                    }
                }

                current_patient_number += 1;
            }

            // Store all new patient names for the result
            combined_result.new_patient_name = new_patient_names.join(", ");

            if result_tx.send(AnonymizeProgressResult::Complete(combined_result)).is_err() {
                tracing::warn!("Anonymize channel closed before completion delivered");
            }
            let _ = forwarder.join();
        });
    }

    /// Check for anonymization results
    pub(super) fn check_anonymize_results(&mut self, conn: &Connection) {
        let results: Vec<AnonymizeProgressResult> = if let Some(rx) = &self.anonymize_rx {
            let mut collected = Vec::new();
            while let Ok(result) = rx.try_recv() {
                collected.push(result);
            }
            collected
        } else {
            return;
        };

        // Get current num_patients from state (if running)
        let num_patients = match &self.anonymize_state {
            AnonymizeState::Running { num_patients, .. } => *num_patients,
            _ => 1,
        };

        for result in results {
            match result {
                AnonymizeProgressResult::Progress(progress) => {
                    if progress.is_complete {
                        // Will be handled by Complete message
                    } else {
                        self.anonymize_state = AnonymizeState::Running {
                            progress,
                            num_patients,
                        };
                    }
                }
                AnonymizeProgressResult::Complete(result) => {
                    // Parse new patient names (comma-separated for multiple patients)
                    let new_names: Vec<String> = result
                        .new_patient_name
                        .split(", ")
                        .map(|s| s.to_string())
                        .collect();

                    tracing::info!(
                        "Anonymization result: {} files, {} studies, {} series, {} patients: {}",
                        result.files_processed,
                        result.study_uid_map.len(),
                        result.series_uid_map.len(),
                        new_names.len(),
                        result.new_patient_name
                    );

                    // Update study records using the study_patient_map to get correct patient for each study
                    for (orig_uid, new_uid) in &result.study_uid_map {
                        tracing::info!("Updating study: {} -> {}", orig_uid, new_uid);
                        match crate::db::get_study_id_by_uid(conn, orig_uid) {
                            Ok(Some(id)) => {
                                // Get patient info for this study from the mapping
                                if let Some(patient_info) = result.study_patient_map.get(orig_uid) {
                                    if let Err(e) = update_study_after_anonymization(
                                        conn,
                                        id,
                                        new_uid,
                                        &patient_info.new_patient_name,
                                        &patient_info.new_patient_id,
                                    ) {
                                        tracing::error!("Failed to update study {}: {}", id, e);
                                    } else {
                                        tracing::info!(
                                            "Updated study {} to patient {} ({})",
                                            id,
                                            patient_info.new_patient_name,
                                            patient_info.new_patient_id
                                        );
                                    }
                                } else {
                                    tracing::warn!(
                                        "No patient info found for study UID: {}",
                                        orig_uid
                                    );
                                }
                            }
                            Ok(None) => {
                                tracing::warn!("Study not found in DB for UID: {}", orig_uid);
                            }
                            Err(e) => {
                                tracing::error!("Error looking up study: {}", e);
                            }
                        }
                    }

                    // Update series records
                    for (orig_uid, new_uid) in &result.series_uid_map {
                        if let Err(e) = update_series_uid(conn, orig_uid, new_uid) {
                            tracing::warn!("Failed to update series: {}", e);
                        }
                    }

                    // Update accession numbers
                    for (orig_study_uid, new_accession) in &result.accession_map {
                        match crate::db::get_study_id_by_uid(conn, orig_study_uid) {
                            Ok(Some(id)) => {
                                if let Err(e) = update_study_accession(conn, id, new_accession) {
                                    tracing::warn!(
                                        "Failed to update accession for study {}: {}",
                                        id,
                                        e
                                    );
                                } else {
                                    tracing::info!(
                                        "Updated study {} accession to {}",
                                        id,
                                        new_accession
                                    );
                                }
                            }
                            Ok(None) => {
                                tracing::warn!(
                                    "Study not found for accession update: {}",
                                    orig_study_uid
                                );
                            }
                            Err(e) => {
                                tracing::error!("Error looking up study for accession: {}", e);
                            }
                        }
                    }

                    // Reload study list
                    self.request_load_studies();

                    // Go directly back to Idle (no OK button needed)
                    self.anonymize_state = AnonymizeState::Idle;
                    self.anonymize_rx = None;
                    tracing::info!(
                        "Anonymization complete: {} files, new names: {}",
                        result.files_processed,
                        result.new_patient_name
                    );
                    return;
                }
                AnonymizeProgressResult::Error(message) => {
                    self.anonymize_state = AnonymizeState::Error { message };
                    self.anonymize_rx = None;
                    return;
                }
            }
        }
    }
}

/// Group studies by patient
#[allow(dead_code)]
pub fn group_studies_by_patient(studies: Vec<Study>) -> Vec<PatientInfo> {
    let mut patient_map: HashMap<String, PatientInfo> = HashMap::new();

    for study in studies {
        let patient_id = study
            .patient_id
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());
        let patient_name = study
            .patient_name
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());

        let entry = patient_map
            .entry(patient_id.clone())
            .or_insert_with(|| PatientInfo {
                patient_id: patient_id.clone(),
                patient_name: patient_name.clone(),
                studies: Vec::new(),
            });
        entry.studies.push(study);
    }

    // Sort studies within each patient by date (newest first)
    for patient in patient_map.values_mut() {
        patient.studies.sort_by(|a, b| {
            let date_a = a.study_date.as_deref().unwrap_or("");
            let date_b = b.study_date.as_deref().unwrap_or("");
            date_b.cmp(date_a)
        });
    }

    // Convert to vec and sort by patient name
    let mut patients: Vec<PatientInfo> = patient_map.into_values().collect();
    patients.sort_by(|a, b| a.patient_name.cmp(&b.patient_name));
    patients
}

/// Count DICOM files for a patient's studies
pub(super) fn count_patient_dicom_files(studies: &[Study]) -> u32 {
    let mut count = 0u32;
    for study in studies {
        let study_path = PathBuf::from(&study.file_path);
        if study_path.is_dir() {
            for entry in WalkDir::new(&study_path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                if path.extension().map(|e| e == "dcm").unwrap_or(false)
                    || path.extension().is_none()
                {
                    count += 1;
                }
            }
        }
    }
    count
}
