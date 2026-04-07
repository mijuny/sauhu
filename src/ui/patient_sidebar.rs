//! Patient Sidebar
//!
//! Shows series from the currently opened patient, grouped by study.
//! Each series has a thumbnail preview and smart naming.
#![allow(dead_code)]

use crate::db::{get_series_for_study, Series, Study};
use crate::dicom::{get_middle_file, group_files_by_series, scan_directory, SeriesInfo, SyncInfo};
use crate::ui::{SeriesAddedEvent, ThumbnailCache};
use rusqlite::Connection;
use std::path::PathBuf;

/// Action requested by the sidebar
#[derive(Debug, Clone)]
pub enum SidebarAction {
    /// Load a series with its sorted file paths
    LoadSeries {
        series_uid: String,
        files: Vec<PathBuf>,
        name: String,
        sync_info: Option<SyncInfo>,
    },
}

/// Drag payload for series
#[derive(Debug, Clone)]
pub struct SeriesDragPayload {
    pub series_uid: String,
    pub files: Vec<PathBuf>,
    pub label: String,
    pub sync_info: Option<SyncInfo>,
}

/// Info about a study with its series
struct StudyWithSeries {
    study: Study,
    db_series: Vec<Series>,
    /// Loaded series info with sorted files (from scanning)
    series_info: Vec<SeriesInfo>,
}

/// Pending request to load patient series
pub struct PendingPatientLoad {
    pub patient_id: String,
    pub patient_name: String,
    pub studies: Vec<Study>,
}

/// Patient sidebar showing current patient's series
pub struct PatientSidebar {
    /// Whether the sidebar is visible
    pub visible: bool,
    /// Current patient info
    patient_id: Option<String>,
    patient_name: Option<String>,
    /// Studies with their series
    studies_with_series: Vec<StudyWithSeries>,
    /// Pending action for the app to handle
    pending_action: Option<SidebarAction>,
    /// Thumbnail cache
    thumbnail_cache: ThumbnailCache,
    /// Whether we're loading series
    loading: bool,
    /// Pending patient load request
    pending_load: Option<PendingPatientLoad>,
}

impl Default for PatientSidebar {
    fn default() -> Self {
        Self {
            visible: true,
            patient_id: None,
            patient_name: None,
            studies_with_series: Vec::new(),
            pending_action: None,
            thumbnail_cache: ThumbnailCache::new(),
            loading: false,
            pending_load: None,
        }
    }
}

impl PatientSidebar {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request async loading of patient series (non-blocking)
    pub fn request_set_patient(
        &mut self,
        patient_id: &str,
        patient_name: &str,
        studies: Vec<Study>,
    ) {
        self.patient_id = Some(patient_id.to_string());
        self.patient_name = Some(patient_name.to_string());
        self.thumbnail_cache.clear();
        self.studies_with_series.clear();
        self.loading = true;
        self.pending_load = Some(PendingPatientLoad {
            patient_id: patient_id.to_string(),
            patient_name: patient_name.to_string(),
            studies,
        });
    }

    /// Take pending load request (for app to send to background worker)
    pub fn take_load_request(&mut self) -> Option<PendingPatientLoad> {
        self.pending_load.take()
    }

    /// Whether series are currently loading
    pub fn is_loading(&self) -> bool {
        self.loading
    }

    /// Set the current patient and load their series (blocking - use request_set_patient for async)
    #[allow(dead_code)]
    pub fn set_patient(
        &mut self,
        conn: &Connection,
        patient_id: &str,
        patient_name: &str,
        studies: Vec<Study>,
    ) {
        self.patient_id = Some(patient_id.to_string());
        self.patient_name = Some(patient_name.to_string());

        // Clear thumbnail cache when changing patient
        self.thumbnail_cache.clear();

        // Load series for each study
        self.studies_with_series = studies
            .into_iter()
            .map(|study| {
                let db_series = get_series_for_study(conn, study.id).unwrap_or_default();

                // Scan the study directory and group files by series
                let series_info = match scan_directory(&study.file_path) {
                    Ok(files) => group_files_by_series(files),
                    Err(e) => {
                        tracing::warn!("Failed to scan study directory {}: {}", study.file_path, e);
                        Vec::new()
                    }
                };

                StudyWithSeries {
                    study,
                    db_series,
                    series_info,
                }
            })
            .collect();

        // Sort by date (newest first)
        self.studies_with_series.sort_by(|a, b| {
            let date_a = a.study.study_date.as_deref().unwrap_or("");
            let date_b = b.study.study_date.as_deref().unwrap_or("");
            date_b.cmp(date_a)
        });
    }

    /// Clear the current patient
    pub fn clear(&mut self) {
        self.patient_id = None;
        self.patient_name = None;
        self.studies_with_series.clear();
        self.thumbnail_cache.clear();
    }

    /// Handle loaded series from background worker
    pub fn handle_series_loaded(&mut self, loaded: Vec<crate::app::LoadedStudyWithSeries>) {
        self.loading = false;

        tracing::info!(
            "Sidebar received {} studies from background loader",
            loaded.len()
        );
        for l in &loaded {
            tracing::info!(
                "  Study {} (path={}): {} db_series, {} series_info",
                l.study.id,
                l.study.file_path,
                l.db_series.len(),
                l.series_info.len()
            );
            for s in &l.series_info {
                tracing::info!(
                    "    Series {}: {} ({} files)",
                    s.series_uid,
                    s.display_name,
                    s.files.len()
                );
            }
        }

        // Convert LoadedStudyWithSeries to internal StudyWithSeries
        self.studies_with_series = loaded
            .into_iter()
            .map(|l| StudyWithSeries {
                study: l.study,
                db_series: l.db_series,
                series_info: l.series_info,
            })
            .collect();

        // Sort by date (newest first)
        self.studies_with_series.sort_by(|a, b| {
            let date_a = a.study.study_date.as_deref().unwrap_or("");
            let date_b = b.study.study_date.as_deref().unwrap_or("");
            date_b.cmp(date_a)
        });
    }

    /// Check if a specific study (by database ID) is currently shown in the sidebar
    pub fn is_showing_study(&self, study_id: i64) -> bool {
        self.studies_with_series
            .iter()
            .any(|s| s.study.id == study_id)
    }

    /// Get number of studies in the sidebar
    pub fn study_count(&self) -> usize {
        self.studies_with_series.len()
    }

    /// Add a study directly to the sidebar for incremental retrieval updates
    ///
    /// This bypasses the async loading and adds an empty study that will be
    /// populated with series as they arrive during retrieval.
    pub fn add_study_for_retrieval(&mut self, study: Study) {
        // Check if study already exists
        if self
            .studies_with_series
            .iter()
            .any(|s| s.study.id == study.id)
        {
            return;
        }

        // Set patient info if not already set
        if self.patient_id.is_none() {
            self.patient_id = study.patient_id.clone();
            self.patient_name = study.patient_name.clone();
        }

        // Add the study with empty series list (series will be added incrementally)
        self.studies_with_series.push(StudyWithSeries {
            study,
            db_series: Vec::new(),
            series_info: Vec::new(),
        });

        // Stop loading indicator since we're adding manually
        self.loading = false;

        tracing::info!("Added study to sidebar for incremental retrieval");
    }

    /// Add a series incrementally during retrieval (real-time updates)
    ///
    /// Called when a series completes during PACS retrieval to update the sidebar
    /// without waiting for the full study to finish.
    pub fn add_series_incremental(&mut self, event: &SeriesAddedEvent) {
        // Find the study in our list
        if let Some(study) = self
            .studies_with_series
            .iter_mut()
            .find(|s| s.study.id == event.study_id)
        {
            // Check if series already exists (avoid duplicates)
            let series_exists = study
                .series_info
                .iter()
                .any(|s| s.series_uid == event.series_uid)
                || study
                    .db_series
                    .iter()
                    .any(|s| s.series_instance_uid == event.series_uid);

            if !series_exists {
                // Add to series_info (for display)
                // Note: files will be populated when user expands/clicks the series
                study.series_info.push(SeriesInfo {
                    series_uid: event.series_uid.clone(),
                    description: Some(event.series_description.clone()),
                    display_name: event.series_description.clone(),
                    modality: Some(event.modality.clone()),
                    series_number: None, // Will be filled from DICOM if available
                    files: Vec::new(),   // Files will be loaded when series is expanded
                    frame_of_reference_uid: None,
                    study_instance_uid: None,
                    orientation: None,
                    slice_locations: Vec::new(),
                });

                // Also add to db_series for consistency
                study.db_series.push(Series {
                    id: 0, // Not used for display
                    study_id: event.study_id,
                    series_instance_uid: event.series_uid.clone(),
                    series_number: None,
                    series_description: Some(event.series_description.clone()),
                    modality: Some(event.modality.clone()),
                    num_images: event.num_images as i32,
                    body_part: None,
                    num_frames: 0,
                });

                tracing::debug!(
                    "Added series {} to sidebar incrementally ({} images)",
                    event.series_uid,
                    event.num_images
                );
            }
        }
    }

    /// Add a series from quick fetch retrieval (simplified for single-study case)
    ///
    /// Unlike add_series_incremental which requires a study_id, this method
    /// adds the series to the first study in the sidebar (used by quick fetch).
    pub fn add_series_from_retrieval(
        &mut self,
        series_uid: &str,
        series_number: Option<i32>,
        series_description: &str,
        modality: &str,
        _num_images: u32,
        storage_path: &std::path::PathBuf,
    ) {
        // Find the first study (quick fetch only has one)
        if let Some(study) = self.studies_with_series.first_mut() {
            // Check if series already exists (trim UIDs for comparison - DICOM can have trailing spaces)
            let series_uid_trimmed = series_uid.trim();
            let series_exists = study
                .series_info
                .iter()
                .any(|s| s.series_uid.trim() == series_uid_trimmed);

            if !series_exists {
                // Scan the storage path for files that belong to THIS series
                let files: Vec<std::path::PathBuf> = if storage_path.is_dir() {
                    let mut matching_files = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(storage_path) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let path = entry.path();
                            if !path.extension().map(|ext| ext == "dcm").unwrap_or(false) {
                                continue;
                            }
                            // Check if file belongs to this series
                            if let Ok(file) = std::fs::File::open(&path) {
                                if let Ok(obj) = dicom::object::from_reader(file) {
                                    if let Ok(uid_elem) = obj
                                        .element(dicom::dictionary_std::tags::SERIES_INSTANCE_UID)
                                    {
                                        if let Ok(file_series_uid) = uid_elem.to_str() {
                                            let file_uid_trimmed = file_series_uid.trim();
                                            if file_uid_trimmed == series_uid_trimmed {
                                                matching_files.push(path);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    matching_files
                } else {
                    Vec::new()
                };

                let display_name = if series_description.is_empty() {
                    format!("Series {}", series_number.unwrap_or(0))
                } else {
                    series_description.to_string()
                };

                tracing::debug!(
                    "Added series '{}' to sidebar ({} files)",
                    series_description,
                    files.len()
                );

                study.series_info.push(SeriesInfo {
                    series_uid: series_uid_trimmed.to_string(),
                    description: Some(series_description.to_string()),
                    display_name,
                    modality: Some(modality.to_string()),
                    series_number,
                    files,
                    frame_of_reference_uid: None,
                    study_instance_uid: None,
                    orientation: None,
                    slice_locations: Vec::new(),
                });
            }
        } else {
            tracing::warn!("No study in sidebar to add series to");
        }
    }

    /// Check if a patient is loaded
    pub fn has_patient(&self) -> bool {
        self.patient_id.is_some()
    }

    /// Take pending action (if any)
    pub fn take_action(&mut self) -> Option<SidebarAction> {
        self.pending_action.take()
    }

    /// Show the sidebar
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }

        // Process any pending thumbnail results
        self.thumbnail_cache.process_pending(ctx);

        egui::SidePanel::left("patient_sidebar")
            .default_width(260.0)
            .min_width(180.0)
            .max_width(400.0)
            .resizable(true)
            .show(ctx, |ui| {
                self.show_content(ui);
            });
    }

    fn show_content(&mut self, ui: &mut egui::Ui) {
        // Debug: Log sidebar state (once per 2 seconds approximately)
        use std::sync::Mutex;
        static LAST_LOG: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let should_log = {
            let now = std::time::Instant::now();
            let mut last = LAST_LOG.lock().unwrap();
            if last.is_none() || now.duration_since(last.unwrap()).as_secs() >= 2 {
                *last = Some(now);
                true
            } else {
                false
            }
        };
        if should_log {
            tracing::debug!(
                "Sidebar state: patient={:?}, loading={}, studies={}, series_total={}",
                self.patient_name,
                self.loading,
                self.studies_with_series.len(),
                self.studies_with_series
                    .iter()
                    .map(|s| s.series_info.len())
                    .sum::<usize>()
            );
        }

        // Header with patient name
        ui.horizontal(|ui| {
            if let Some(name) = &self.patient_name {
                ui.heading(name);
            } else {
                ui.heading("No Patient");
            }
        });

        if let Some(id) = &self.patient_id {
            ui.small(format!("ID: {}", id));
        }

        ui.separator();

        if !self.has_patient() {
            ui.colored_label(
                egui::Color32::GRAY,
                "Press D to open Database and select a patient.",
            );
            return;
        }

        // Show loading indicator while series are loading
        if self.loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Loading series...");
            });
            return;
        }

        // Collect actions to perform after rendering
        let mut load_series: Option<(String, Vec<PathBuf>, String, Option<SyncInfo>)> = None;

        // Show studies and their series
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for study_data in &self.studies_with_series {
                    let study = &study_data.study;
                    let series_list = &study_data.series_info;

                    // Study header
                    ui.horizontal(|ui| {
                        // Date
                        if let Some(date) = &study.study_date {
                            ui.strong(format_date(date));
                        }
                        // Modality
                        if let Some(modality) = &study.modality {
                            ui.label(format!("[{}]", modality));
                        }
                    });

                    // Study description
                    if let Some(desc) = &study.study_description {
                        if !desc.is_empty() {
                            ui.small(desc);
                        }
                    }

                    // Series list with thumbnails
                    ui.indent(study.id, |ui| {
                        for series in series_list {
                            let series_label = &series.display_name;
                            let sync_info = Some(SyncInfo::from_series_info(series));

                            // Create draggable series item
                            let payload = SeriesDragPayload {
                                series_uid: series.series_uid.clone(),
                                files: series.files.clone(),
                                label: series_label.clone(),
                                sync_info: sync_info.clone(),
                            };

                            let response = ui.dnd_drag_source(
                                egui::Id::new(("series", &series.series_uid)),
                                payload,
                                |ui| {
                                    ui.horizontal(|ui| {
                                        // Thumbnail (72x72 display size)
                                        let thumb_size = egui::vec2(72.0, 72.0);
                                        if let Some(middle_path) = get_middle_file(&series.files) {
                                            if let Some(texture) = self
                                                .thumbnail_cache
                                                .get_or_request(&series.series_uid, middle_path)
                                            {
                                                ui.add(
                                                    egui::Image::new(texture)
                                                        .fit_to_exact_size(thumb_size),
                                                );
                                            } else {
                                                // Placeholder while loading
                                                let (rect, _) = ui.allocate_exact_size(
                                                    thumb_size,
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().rect_filled(
                                                    rect,
                                                    4.0,
                                                    egui::Color32::from_gray(60),
                                                );
                                                if self
                                                    .thumbnail_cache
                                                    .is_loading(&series.series_uid)
                                                {
                                                    // Show loading indicator
                                                    ui.painter().text(
                                                        rect.center(),
                                                        egui::Align2::CENTER_CENTER,
                                                        "...",
                                                        egui::FontId::default(),
                                                        egui::Color32::GRAY,
                                                    );
                                                }
                                            }
                                        } else {
                                            // No files - empty placeholder
                                            let (rect, _) = ui.allocate_exact_size(
                                                thumb_size,
                                                egui::Sense::hover(),
                                            );
                                            ui.painter().rect_filled(
                                                rect,
                                                4.0,
                                                egui::Color32::from_gray(40),
                                            );
                                        }

                                        // Series info
                                        ui.vertical(|ui| {
                                            // Series name
                                            ui.label(series_label);
                                            // Image count
                                            if !series.files.is_empty() {
                                                ui.small(format!("{} images", series.files.len()));
                                            }
                                        });
                                    });
                                },
                            );

                            // Double-click to load series
                            if response.response.double_clicked() && !series.files.is_empty() {
                                load_series = Some((
                                    series.series_uid.clone(),
                                    series.files.clone(),
                                    series_label.clone(),
                                    sync_info.clone(),
                                ));
                            }
                        }

                        if series_list.is_empty() {
                            ui.small("No series found");
                        }
                    });

                    ui.add_space(8.0);
                }

                if self.studies_with_series.is_empty() {
                    ui.colored_label(egui::Color32::GRAY, "No studies for this patient.");
                }
            });

        // Apply collected actions
        if let Some((series_uid, files, name, sync_info)) = load_series {
            self.pending_action = Some(SidebarAction::LoadSeries {
                series_uid,
                files,
                name,
                sync_info,
            });
        }
    }
}

/// Format DICOM date (YYYYMMDD) for display
fn format_date(date: &str) -> String {
    if date.len() == 8 {
        format!("{}-{}-{}", &date[0..4], &date[4..6], &date[6..8])
    } else {
        date.to_string()
    }
}
