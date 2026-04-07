//! Database Window
//!
//! Browse local patients and query PACS for studies.
#![allow(dead_code)]

use crate::config::Settings;
use crate::db::{
    delete_study, get_all_studies_with_image_count, get_next_anon_accession_number,
    get_next_anon_patient_number, get_study, insert_series, insert_series_from_event,
    update_series_uid, update_study_accession, update_study_after_anonymization,
    upsert_study_for_retrieval, RetrievalStudyInfo, Series, Study, StudySource, StudyWithImages,
};
use crate::dicom::{anonymize_patient, AnonymizeConfig, AnonymizeProgress, AnonymizeResult};
use crate::pacs::{
    AggregateProgress, DicomScp, DicomScu, ParallelRetrieveConfig, ParallelRetriever, QueryParams,
    SeriesCompleteEvent, StudyResult,
};
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use walkdir::WalkDir;

/// Patient info aggregated from studies
#[derive(Debug, Clone)]
pub struct PatientInfo {
    pub patient_id: String,
    pub patient_name: String,
    pub studies: Vec<Study>,
}

/// Action to open a study in the viewer
#[derive(Debug, Clone)]
pub struct OpenStudyAction {
    pub study: Study,
}

/// Sort column for local studies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LocalSortColumn {
    Name,
    Id,
    Accession,
    Modality,
    Description,
    #[default]
    Date,
    Images,
}

/// State for database window
pub struct DatabaseWindow {
    /// Whether the window is open
    pub open: bool,
    /// Application settings
    settings: Arc<Settings>,
    /// Local studies with image counts
    local_studies: Vec<StudyWithImages>,
    /// Search filter for local studies
    local_filter: String,
    /// Selected study IDs for multi-select
    selected_studies: HashSet<i64>,
    /// Anchor study ID for shift-click range selection
    last_selected_id: Option<i64>,
    /// Sort column for local tab
    local_sort_column: LocalSortColumn,
    /// Sort ascending (false = descending, newest first for date)
    local_sort_ascending: bool,
    /// Current tab (Local or PACS)
    current_tab: Tab,
    /// PACS search filters
    pacs_filters: SearchFilters,
    /// PACS query results
    pacs_results: Vec<StudyResult>,
    /// Selected PACS result index
    selected_pacs: Option<usize>,
    /// Selected PACS server id
    selected_server: Option<String>,
    /// Query state
    query_state: QueryState,
    /// Retrieve state
    retrieve_state: RetrieveState,
    /// Channel for query results
    query_rx: Option<mpsc::Receiver<QueryResult>>,
    /// Channel for retrieve progress
    retrieve_rx: Option<mpsc::Receiver<RetrieveResult>>,
    /// Cancel flag for retrieve
    cancel_flag: Option<Arc<AtomicBool>>,
    /// Status message
    status: String,
    /// Pending action (open study)
    pending_action: Option<OpenStudyAction>,
    /// Whether we're loading local studies
    loading_studies: bool,
    /// Whether we need to request a study load
    needs_study_load: bool,
    /// Channel to send series added events to app
    series_added_tx: mpsc::Sender<SeriesAddedEvent>,
    /// Whether to open study in viewer when first series arrives
    open_after_retrieve: bool,
    /// Anonymization state
    anonymize_state: AnonymizeState,
    /// Channel for anonymization progress
    anonymize_rx: Option<mpsc::Receiver<AnonymizeProgressResult>>,
    /// Delete state
    delete_state: DeleteState,
}

/// Result from anonymization background worker
enum AnonymizeProgressResult {
    Progress(AnonymizeProgress),
    Complete(AnonymizeResult),
    Error(String),
}

#[derive(Default, PartialEq)]
enum Tab {
    #[default]
    Local,
    Pacs,
}

/// Search filter fields for PACS
#[derive(Default)]
struct SearchFilters {
    patient_name: String,
    patient_id: String,
    accession: String,
    study_date: String,
    modality: String,
}

#[derive(Default, PartialEq)]
enum QueryState {
    #[default]
    Idle,
    Querying,
}

/// Event emitted when a series is added to the database during retrieval
#[derive(Debug, Clone)]
pub struct SeriesAddedEvent {
    /// Database study ID
    pub study_id: i64,
    /// Series Instance UID
    pub series_uid: String,
    /// Series description
    pub series_description: String,
    /// Modality (CT, MR, etc.)
    pub modality: String,
    /// Number of images
    pub num_images: u32,
    /// Path to series files
    pub storage_path: PathBuf,
}

#[derive(Default)]
#[allow(clippy::large_enum_variant)]
enum RetrieveState {
    #[default]
    Idle,
    Retrieving {
        study_uid: String,
        progress: AggregateProgress,
        /// Study info for incremental DB inserts
        study_info: Option<StudyImportInfo>,
        /// Database study ID (set after first series completes)
        db_study_id: Option<i64>,
    },
    Complete {
        path: PathBuf,
    },
    Error {
        message: String,
    },
}

/// Info about a patient group to be anonymized
#[derive(Clone)]
struct AnonymizePatientGroup {
    /// Original patient ID
    patient_id: String,
    /// Original patient name
    patient_name: String,
    /// Studies belonging to this patient
    studies: Vec<Study>,
    /// Number of DICOM files
    num_files: u32,
}

/// State for anonymization operation
#[derive(Default)]
enum AnonymizeState {
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
    Complete {
        files_processed: u32,
        /// List of new patient names created
        new_names: Vec<String>,
    },
    /// Anonymization failed
    Error { message: String },
}

/// State for delete operation
#[derive(Default, PartialEq)]
enum DeleteState {
    #[default]
    Idle,
    /// Delete failed
    Error { message: String },
}

enum QueryResult {
    Success(Vec<StudyResult>),
    Error(String),
}

enum RetrieveResult {
    Progress(AggregateProgress),
    Complete {
        path: PathBuf,
        study_info: StudyImportInfo,
    },
    Error(String),
}

#[derive(Clone)]
struct StudyImportInfo {
    study_instance_uid: String,
    patient_name: String,
    patient_id: String,
    study_date: String,
    study_description: String,
    modality: String,
    accession_number: String,
}

impl DatabaseWindow {
    pub fn new(settings: Arc<Settings>, series_added_tx: mpsc::Sender<SeriesAddedEvent>) -> Self {
        // Pre-select default PACS server
        let selected_server = settings
            .get_default_pacs_server()
            .map(|(id, _)| id.to_string());
        Self {
            open: false,
            settings,
            local_studies: Vec::new(),
            local_filter: String::new(),
            selected_studies: HashSet::new(),
            last_selected_id: None,
            local_sort_column: LocalSortColumn::default(),
            local_sort_ascending: false, // Default: newest first
            current_tab: Tab::default(),
            pacs_filters: SearchFilters::default(),
            pacs_results: Vec::new(),
            selected_pacs: None,
            selected_server,
            query_state: QueryState::default(),
            retrieve_state: RetrieveState::default(),
            query_rx: None,
            retrieve_rx: None,
            cancel_flag: None,
            status: String::new(),
            pending_action: None,
            loading_studies: false,
            needs_study_load: false,
            series_added_tx,
            open_after_retrieve: false,
            anonymize_state: AnonymizeState::default(),
            anonymize_rx: None,
            delete_state: DeleteState::default(),
        }
    }

    /// Load local studies from database (blocking - use request_load_studies for async)
    pub fn load_local_studies(&mut self, conn: &Connection) {
        match get_all_studies_with_image_count(conn) {
            Ok(studies) => {
                self.local_studies = studies;
            }
            Err(e) => {
                tracing::error!("Failed to load studies: {}", e);
                self.local_studies = Vec::new();
            }
        }
    }

    /// Request async loading of local studies (non-blocking)
    pub fn request_load_studies(&mut self) {
        self.loading_studies = true;
        self.needs_study_load = true;
    }

    /// Check if a study load was requested and should be triggered
    pub fn take_load_request(&mut self) -> bool {
        if self.needs_study_load {
            self.needs_study_load = false;
            true
        } else {
            false
        }
    }

    /// Handle loaded studies from background worker
    pub fn handle_studies_loaded(&mut self, studies: Vec<StudyWithImages>) {
        self.local_studies = studies;
        self.loading_studies = false;
    }

    /// Whether studies are currently loading
    pub fn is_loading_studies(&self) -> bool {
        self.loading_studies
    }

    /// Take pending action (if any)
    pub fn take_action(&mut self) -> Option<OpenStudyAction> {
        self.pending_action.take()
    }

    /// Update retrieve state even when window is closed
    /// This must be called every frame to process incremental series updates
    pub fn update(&mut self, conn: &Connection) {
        self.check_retrieve_results(conn);
        self.check_anonymize_results(conn);
    }

    /// Check if a retrieval is in progress
    pub fn is_retrieving(&self) -> bool {
        matches!(self.retrieve_state, RetrieveState::Retrieving { .. })
    }

    /// Show the database window
    pub fn show(&mut self, ctx: &egui::Context, conn: &Connection) {
        // Check for async results
        self.check_query_results();
        // Note: check_retrieve_results is now called from update() which runs every frame

        let mut open = self.open;
        egui::Window::new("Database")
            .open(&mut open)
            .default_width(800.0)
            .default_height(600.0)
            .resizable(true)
            .show(ctx, |ui| {
                self.show_content(ui, conn);
            });
        // Only overwrite if X button closed window (open becomes false)
        // Don't overwrite if we explicitly closed via double-click (self.open already false)
        self.open = self.open && open;
    }

    fn show_content(&mut self, ui: &mut egui::Ui, conn: &Connection) {
        // Tab bar
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.current_tab == Tab::Local, "Local")
                .clicked()
            {
                self.current_tab = Tab::Local;
            }
            if ui
                .selectable_label(self.current_tab == Tab::Pacs, "PACS Query")
                .clicked()
            {
                self.current_tab = Tab::Pacs;
            }
        });

        ui.separator();

        match self.current_tab {
            Tab::Local => self.show_local_tab(ui, conn),
            Tab::Pacs => self.show_pacs_tab(ui, conn),
        }
    }

    fn show_local_tab(&mut self, ui: &mut egui::Ui, conn: &Connection) {
        // Show dialogs at the TOP if active (before anything else)
        self.show_delete_ui(ui, conn);
        self.show_anonymization_ui(ui, conn);

        // Toolbar
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.text_edit_singleline(&mut self.local_filter);
            if ui.button("Refresh").clicked() && !self.loading_studies {
                self.request_load_studies();
            }
            if self.loading_studies {
                ui.spinner();
            }
        });

        ui.separator();

        // Filter and sort studies
        let filter_lower = self.local_filter.to_lowercase();
        let sort_col = self.local_sort_column;
        let sort_asc = self.local_sort_ascending;

        // Build filtered list with extracted data for sorting
        let mut filtered: Vec<_> = self
            .local_studies
            .iter()
            .enumerate()
            .filter(|(_, sw)| {
                if filter_lower.is_empty() {
                    return true;
                }
                let s = &sw.study;
                s.patient_name
                    .as_ref()
                    .map(|n| n.to_lowercase().contains(&filter_lower))
                    .unwrap_or(false)
                    || s.patient_id
                        .as_ref()
                        .map(|id| id.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
                    || s.accession_number
                        .as_ref()
                        .map(|a| a.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
                    || s.modality
                        .as_ref()
                        .map(|m| m.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
                    || s.study_description
                        .as_ref()
                        .map(|d| d.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
            })
            .map(|(idx, sw)| {
                let s = &sw.study;
                (
                    idx,
                    s.patient_name.clone().unwrap_or_default(),
                    s.patient_id.clone().unwrap_or_default(),
                    s.accession_number.clone().unwrap_or_default(),
                    s.modality.clone().unwrap_or_default(),
                    s.study_description.clone().unwrap_or_default(),
                    s.study_date.clone().unwrap_or_default(),
                    sw.num_images,
                    sw.study.clone(),
                )
            })
            .collect();

        // Sort based on current column and direction
        filtered.sort_by(|a, b| {
            let cmp = match sort_col {
                LocalSortColumn::Name => a.1.cmp(&b.1),
                LocalSortColumn::Id => a.2.cmp(&b.2),
                LocalSortColumn::Accession => a.3.cmp(&b.3),
                LocalSortColumn::Modality => a.4.cmp(&b.4),
                LocalSortColumn::Description => a.5.cmp(&b.5),
                LocalSortColumn::Date => a.6.cmp(&b.6),
                LocalSortColumn::Images => a.7.cmp(&b.7),
            };
            if sort_asc {
                cmp
            } else {
                cmp.reverse()
            }
        });

        // Pre-compute sort indicators
        let sort_indicator = |col: LocalSortColumn| -> &'static str {
            if sort_col == col {
                if sort_asc {
                    " ▲"
                } else {
                    " ▼"
                }
            } else {
                ""
            }
        };
        let name_header = format!("Patient Name{}", sort_indicator(LocalSortColumn::Name));
        let id_header = format!("Patient ID{}", sort_indicator(LocalSortColumn::Id));
        let accession_header = format!("Accession{}", sort_indicator(LocalSortColumn::Accession));
        let modality_header = format!("Modality{}", sort_indicator(LocalSortColumn::Modality));
        let desc_header = format!(
            "Description{}",
            sort_indicator(LocalSortColumn::Description)
        );
        let date_header = format!("Date{}", sort_indicator(LocalSortColumn::Date));
        let images_header = format!("Images{}", sort_indicator(LocalSortColumn::Images));

        // Track which column was clicked
        let mut clicked_column: Option<LocalSortColumn> = None;
        // Track click info: (study_id, filtered_position, modifiers)
        let mut clicked_study_info: Option<(i64, usize, egui::Modifiers)> = None;
        let mut double_clicked_study_id: Option<i64> = None;

        // Build filtered_ids for range selection
        let filtered_ids: Vec<i64> = filtered
            .iter()
            .map(|(_, _, _, _, _, _, _, _, s)| s.id)
            .collect();

        let available_height = ui.available_height() - 40.0;
        egui::ScrollArea::vertical()
            .max_height(available_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("local_studies_table")
                    .num_columns(7)
                    .striped(true)
                    .min_col_width(60.0)
                    .show(ui, |ui| {
                        // Header row with sortable columns
                        if ui.selectable_label(false, &name_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Name);
                        }
                        if ui.selectable_label(false, &id_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Id);
                        }
                        if ui.selectable_label(false, &accession_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Accession);
                        }
                        if ui.selectable_label(false, &modality_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Modality);
                        }
                        if ui.selectable_label(false, &desc_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Description);
                        }
                        if ui.selectable_label(false, &date_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Date);
                        }
                        if ui.selectable_label(false, &images_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Images);
                        }
                        ui.end_row();

                        // Data rows
                        for (
                            filtered_pos,
                            (
                                _,
                                name,
                                id,
                                accession,
                                modality,
                                description,
                                date,
                                num_images,
                                study,
                            ),
                        ) in filtered.iter().enumerate()
                        {
                            let selected = self.selected_studies.contains(&study.id);
                            let formatted_date = format_date(date);

                            let name_resp = ui.selectable_label(selected, name);
                            ui.label(id);
                            ui.label(accession);
                            ui.label(modality);
                            ui.label(description);
                            ui.label(&formatted_date);
                            ui.label(num_images.to_string());
                            ui.end_row();

                            if name_resp.clicked() {
                                let modifiers = ui.input(|i| i.modifiers);
                                clicked_study_info = Some((study.id, filtered_pos, modifiers));
                            }
                            if name_resp.double_clicked() {
                                double_clicked_study_id = Some(study.id);
                            }
                        }
                    });

                if filtered.is_empty() {
                    ui.colored_label(
                        egui::Color32::GRAY,
                        "No local studies. Load DICOM files or retrieve from PACS.",
                    );
                }
            });

        // Handle clicks after the UI code (to avoid borrow conflicts)
        if let Some(col) = clicked_column {
            self.toggle_sort(col);
        }
        if let Some((study_id, filtered_pos, modifiers)) = clicked_study_info {
            self.handle_study_click(study_id, filtered_pos, &filtered_ids, modifiers);
        }
        if let Some(study_id) = double_clicked_study_id {
            // Find the study and open it
            if let Some(sw) = self.local_studies.iter().find(|sw| sw.study.id == study_id) {
                // Select just this study
                self.selected_studies.clear();
                self.selected_studies.insert(study_id);
                self.last_selected_id = Some(study_id);
                self.pending_action = Some(OpenStudyAction {
                    study: sw.study.clone(),
                });
                self.open = false;
            }
        }

        // Handle Delete key
        if ui.input(|i| i.key_pressed(egui::Key::Delete))
            && !self.selected_studies.is_empty()
            && matches!(self.delete_state, DeleteState::Idle)
        {
            let study_ids: Vec<i64> = self.selected_studies.iter().cloned().collect();
            self.execute_delete(conn, study_ids);
        }

        ui.separator();

        // Action buttons
        ui.horizontal(|ui| {
            // Open Study button - only enable for single selection
            let can_open = self.selected_studies.len() == 1;
            if ui
                .add_enabled(can_open, egui::Button::new("Open Study"))
                .clicked()
            {
                if let Some(&study_id) = self.selected_studies.iter().next() {
                    if let Some(sw) = self.local_studies.iter().find(|sw| sw.study.id == study_id) {
                        self.pending_action = Some(OpenStudyAction {
                            study: sw.study.clone(),
                        });
                        self.open = false;
                    }
                }
            }

            // Anonymize button - works with any selection (groups by patient)
            let can_anonymize = !self.selected_studies.is_empty()
                && matches!(self.anonymize_state, AnonymizeState::Idle);
            if ui
                .add_enabled(can_anonymize, egui::Button::new("Anonymize..."))
                .clicked()
            {
                tracing::info!(
                    "Anonymize button clicked with {} studies selected",
                    self.selected_studies.len()
                );

                // Collect selected studies
                let selected: Vec<Study> = self
                    .selected_studies
                    .iter()
                    .filter_map(|&id| {
                        self.local_studies
                            .iter()
                            .find(|sw| sw.study.id == id)
                            .map(|sw| sw.study.clone())
                    })
                    .collect();

                // Group by patient_id
                let mut patient_map: HashMap<String, AnonymizePatientGroup> = HashMap::new();
                for study in selected {
                    let patient_id = study.patient_id.clone().unwrap_or_default();
                    let patient_name = study.patient_name.clone().unwrap_or_default();

                    let group = patient_map.entry(patient_id.clone()).or_insert_with(|| {
                        AnonymizePatientGroup {
                            patient_id,
                            patient_name,
                            studies: Vec::new(),
                            num_files: 0,
                        }
                    });
                    group.studies.push(study);
                }

                // Count files for each group
                let mut patient_groups: Vec<AnonymizePatientGroup> =
                    patient_map.into_values().collect();
                let mut total_files = 0u32;
                for group in &mut patient_groups {
                    group.num_files = count_patient_dicom_files(&group.studies);
                    total_files += group.num_files;
                }

                // Sort by patient name for consistent display
                patient_groups.sort_by(|a, b| a.patient_name.cmp(&b.patient_name));

                tracing::info!(
                    "Grouped into {} patients with {} total files",
                    patient_groups.len(),
                    total_files
                );

                self.anonymize_state = AnonymizeState::Confirming {
                    patient_groups,
                    total_files,
                };
            }

            // Delete button - works with any selection
            let can_delete =
                !self.selected_studies.is_empty() && matches!(self.delete_state, DeleteState::Idle);
            if ui
                .add_enabled(can_delete, egui::Button::new("Delete"))
                .clicked()
            {
                let study_ids: Vec<i64> = self.selected_studies.iter().cloned().collect();
                self.execute_delete(conn, study_ids);
            }
        });
    }

    /// Show anonymization confirmation dialog and progress
    fn show_anonymization_ui(&mut self, ui: &mut egui::Ui, conn: &Connection) {
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

    /// Show delete error if any
    fn show_delete_ui(&mut self, ui: &mut egui::Ui, _conn: &Connection) {
        if let DeleteState::Error { message } = &self.delete_state {
            ui.separator();
            ui.colored_label(egui::Color32::RED, message);
            if ui.button("OK").clicked() {
                self.delete_state = DeleteState::Idle;
            }
        }
    }

    /// Execute delete for the given study IDs
    fn execute_delete(&mut self, conn: &Connection, study_ids: Vec<i64>) {
        let mut errors = Vec::new();

        for study_id in &study_ids {
            // Get study info before deleting
            if let Some(sw) = self
                .local_studies
                .iter()
                .find(|sw| sw.study.id == *study_id)
            {
                let study = &sw.study;

                // Delete files for PACS studies only
                if study.source == StudySource::Pacs {
                    let path = PathBuf::from(&study.file_path);
                    if path.exists() {
                        if let Err(e) = std::fs::remove_dir_all(&path) {
                            tracing::warn!("Failed to delete PACS cache at {:?}: {}", path, e);
                            // Continue anyway - DB delete is more important
                        } else {
                            tracing::info!("Deleted PACS cache at {:?}", path);
                        }
                    }
                }

                // Delete from database (cascade handles series/measurements)
                if let Err(e) = delete_study(conn, *study_id) {
                    errors.push(format!("Failed to delete study {}: {}", study_id, e));
                } else {
                    tracing::info!("Deleted study {} from database", study_id);
                }
            }
        }

        // Clear selection
        self.selected_studies.clear();
        self.last_selected_id = None;

        // Request study list refresh
        self.request_load_studies();

        // Update state
        if errors.is_empty() {
            self.delete_state = DeleteState::Idle;
            tracing::info!("Deleted {} studies", study_ids.len());
        } else {
            self.delete_state = DeleteState::Error {
                message: errors.join("; "),
            };
        }
    }

    /// Start anonymization for multiple patient groups
    fn start_anonymization_for_groups(
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

        let (result_tx, result_rx) = mpsc::channel();
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
            let (progress_tx, progress_rx) = mpsc::channel::<AnonymizeProgress>();
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
                        let _ = result_tx.send(AnonymizeProgressResult::Error(format!(
                            "Failed to anonymize patient {}: {}",
                            group.patient_name, e
                        )));
                        let _ = forwarder.join();
                        return;
                    }
                }

                current_patient_number += 1;
            }

            // Store all new patient names for the result
            combined_result.new_patient_name = new_patient_names.join(", ");

            let _ = result_tx.send(AnonymizeProgressResult::Complete(combined_result));
            let _ = forwarder.join();
        });
    }

    /// Check for anonymization results
    fn check_anonymize_results(&mut self, conn: &Connection) {
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

    /// Toggle sort on a column
    fn toggle_sort(&mut self, column: LocalSortColumn) {
        if self.local_sort_column == column {
            self.local_sort_ascending = !self.local_sort_ascending;
        } else {
            self.local_sort_column = column;
            // Default to descending for date, ascending for others
            self.local_sort_ascending = column != LocalSortColumn::Date;
        }
    }

    /// Handle study click with modifier keys for multi-select
    fn handle_study_click(
        &mut self,
        study_id: i64,
        filtered_pos: usize,
        filtered_ids: &[i64],
        modifiers: egui::Modifiers,
    ) {
        if modifiers.command {
            // Ctrl+Click (or Cmd on Mac): toggle selection
            if !self.selected_studies.remove(&study_id) {
                self.selected_studies.insert(study_id);
            }
            // Don't update anchor for Ctrl+Click
        } else if modifiers.shift {
            // Shift+Click: range selection from anchor
            if let Some(anchor_id) = self.last_selected_id {
                if let Some(anchor_pos) = filtered_ids.iter().position(|&id| id == anchor_id) {
                    self.selected_studies.clear();
                    let (start, end) = (anchor_pos.min(filtered_pos), anchor_pos.max(filtered_pos));
                    for &id in &filtered_ids[start..=end] {
                        self.selected_studies.insert(id);
                    }
                }
            } else {
                // No anchor, just select clicked item
                self.selected_studies.clear();
                self.selected_studies.insert(study_id);
            }
            self.last_selected_id = Some(study_id);
        } else {
            // Plain click: single selection
            self.selected_studies.clear();
            self.selected_studies.insert(study_id);
            self.last_selected_id = Some(study_id);
        }
    }

    fn show_pacs_tab(&mut self, ui: &mut egui::Ui, conn: &Connection) {
        // Server selection
        let servers = self.settings.pacs_servers();
        ui.horizontal(|ui| {
            ui.label("Server:");
            egui::ComboBox::from_id_salt("pacs_server_select")
                .selected_text(
                    self.selected_server
                        .as_ref()
                        .and_then(|id| self.settings.get_pacs_server(id))
                        .map(|s| s.name.as_str())
                        .unwrap_or("Select server..."),
                )
                .show_ui(ui, |ui| {
                    for (id, server) in &servers {
                        let label = format!("{} ({}:{})", server.name, server.host, server.port);
                        if ui
                            .selectable_label(
                                self.selected_server.as_deref() == Some(*id),
                                label,
                            )
                            .clicked()
                        {
                            self.selected_server = Some(id.to_string());
                        }
                    }
                });
        });

        ui.separator();

        // Search filters
        ui.heading("Search Filters");
        egui::Grid::new("search_filters")
            .num_columns(4)
            .spacing([10.0, 6.0])
            .show(ui, |ui| {
                ui.label("Patient Name:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.pacs_filters.patient_name)
                        .hint_text("e.g., Smith* or *John*"),
                );
                ui.label("Patient ID:");
                ui.text_edit_singleline(&mut self.pacs_filters.patient_id);
                ui.end_row();

                ui.label("Accession:");
                ui.text_edit_singleline(&mut self.pacs_filters.accession);
                ui.label("Study Date:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.pacs_filters.study_date)
                        .hint_text("YYYYMMDD"),
                );
                ui.end_row();

                ui.label("Modality:");
                egui::ComboBox::from_id_salt("modality_filter")
                    .selected_text(if self.pacs_filters.modality.is_empty() {
                        "Any"
                    } else {
                        &self.pacs_filters.modality
                    })
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(self.pacs_filters.modality.is_empty(), "Any")
                            .clicked()
                        {
                            self.pacs_filters.modality.clear();
                        }
                        for modality in ["CT", "MR", "CR", "DX", "US", "NM", "PT", "XA"] {
                            if ui
                                .selectable_label(self.pacs_filters.modality == modality, modality)
                                .clicked()
                            {
                                self.pacs_filters.modality = modality.to_string();
                            }
                        }
                    });
                ui.end_row();
            });

        // Query buttons
        ui.horizontal(|ui| {
            let can_query = self.selected_server.is_some() && self.query_state == QueryState::Idle;
            if ui
                .add_enabled(can_query, egui::Button::new("Search"))
                .clicked()
            {
                self.start_query();
            }
            if ui.button("Clear").clicked() {
                self.pacs_filters = SearchFilters::default();
                self.pacs_results.clear();
                self.selected_pacs = None;
                self.status.clear();
            }
            if self.query_state == QueryState::Querying {
                ui.spinner();
                ui.label("Searching...");
            }
        });

        ui.separator();

        // Results table
        ui.heading(format!("Results ({})", self.pacs_results.len()));

        // Track double-click outside closures
        let mut double_clicked_pacs: Option<usize> = None;
        let mut clicked_pacs: Option<usize> = None;

        let available_height = ui.available_height() - 80.0;
        egui::ScrollArea::vertical()
            .max_height(available_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("results_table")
                    .num_columns(6)
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Patient Name");
                        ui.strong("Patient ID");
                        ui.strong("Study Date");
                        ui.strong("Description");
                        ui.strong("Modality");
                        ui.strong("Images");
                        ui.end_row();

                        for (idx, result) in self.pacs_results.iter().enumerate() {
                            let selected = self.selected_pacs == Some(idx);

                            // Make all cells respond to clicks
                            let r1 = ui.selectable_label(selected, &result.patient_name);
                            let r2 = ui.selectable_label(selected, &result.patient_id);
                            let r3 = ui.selectable_label(selected, result.formatted_date());
                            let r4 = ui.selectable_label(selected, &result.study_description);
                            let r5 = ui.selectable_label(selected, &result.modalities);
                            let r6 = ui.selectable_label(
                                selected,
                                result
                                    .num_instances
                                    .map(|n| n.to_string())
                                    .unwrap_or_default(),
                            );
                            ui.end_row();

                            // Check if any cell in the row was clicked or double-clicked
                            if r1.clicked()
                                || r2.clicked()
                                || r3.clicked()
                                || r4.clicked()
                                || r5.clicked()
                                || r6.clicked()
                            {
                                clicked_pacs = Some(idx);
                            }
                            if r1.double_clicked()
                                || r2.double_clicked()
                                || r3.double_clicked()
                                || r4.double_clicked()
                                || r5.double_clicked()
                                || r6.double_clicked()
                            {
                                double_clicked_pacs = Some(idx);
                            }
                        }
                    });

                if self.pacs_results.is_empty() && self.query_state == QueryState::Idle {
                    ui.colored_label(
                        egui::Color32::GRAY,
                        "No results. Enter search criteria and click 'Search'.",
                    );
                }
            });

        // Handle clicks after scroll area (avoid borrow conflicts)
        if let Some(idx) = clicked_pacs {
            self.selected_pacs = Some(idx);
        }

        // Handle double-click: start retrieve and open patient
        if let Some(idx) = double_clicked_pacs {
            tracing::info!("PACS result double-clicked: index {}", idx);
            self.selected_pacs = Some(idx);
            self.open_after_retrieve = true;
            if !matches!(self.retrieve_state, RetrieveState::Retrieving { .. }) {
                tracing::info!("Starting retrieve after double-click");
                self.start_retrieve(conn);
            } else {
                tracing::warn!("Already retrieving, cannot start new retrieve");
            }
        }

        ui.separator();

        // Retrieve button, Query Patient button, and status
        ui.horizontal(|ui| {
            let can_retrieve = self.selected_pacs.is_some()
                && !matches!(self.retrieve_state, RetrieveState::Retrieving { .. });

            if ui
                .add_enabled(can_retrieve, egui::Button::new("Retrieve Study"))
                .clicked()
            {
                self.start_retrieve(conn);
            }

            // Query Patient button - queries all studies for selected patient
            let can_query_patient =
                self.selected_pacs.is_some() && self.query_state == QueryState::Idle;
            if ui
                .add_enabled(can_query_patient, egui::Button::new("Query Patient"))
                .clicked()
            {
                if let Some(idx) = self.selected_pacs {
                    if let Some(result) = self.pacs_results.get(idx) {
                        // Clear filters and set patient ID
                        self.pacs_filters = SearchFilters::default();
                        self.pacs_filters.patient_id = result.patient_id.clone();
                        self.selected_pacs = None;
                        self.start_query();
                    }
                }
            }

            match &self.retrieve_state {
                RetrieveState::Idle => {}
                RetrieveState::Retrieving { progress, .. } => {
                    ui.spinner();
                    // Show series and image progress
                    if progress.total_series > 1 {
                        ui.label(format!(
                            "Retrieving series {}/{} ({} images, {} workers)",
                            progress.completed_series,
                            progress.total_series,
                            progress.completed_images,
                            progress.active_workers
                        ));
                    } else if progress.total_images > 0 {
                        ui.label(format!(
                            "Retrieving... {}/{} ({}%)",
                            progress.completed_images,
                            progress.total_images,
                            progress.percent() as i32
                        ));
                    } else {
                        ui.label(format!(
                            "Retrieving... {} images",
                            progress.completed_images
                        ));
                    }
                    if ui.button("Stop").clicked() {
                        // Signal cancel to retriever (will abort PACS associations)
                        if let Some(flag) = &self.cancel_flag {
                            flag.store(true, Ordering::SeqCst);
                        }
                        self.retrieve_state = RetrieveState::Idle;
                        self.retrieve_rx = None;
                        self.cancel_flag = None;
                        self.status = "Retrieve cancelled".to_string();
                    }
                }
                RetrieveState::Complete { path } => {
                    ui.colored_label(egui::Color32::GREEN, format!("Retrieved to {:?}", path));
                }
                RetrieveState::Error { message } => {
                    ui.colored_label(egui::Color32::RED, message);
                }
            }
        });

        if !self.status.is_empty() {
            ui.label(&self.status);
        }
    }

    fn start_query(&mut self) {
        let Some(server_id) = &self.selected_server else {
            return;
        };
        let Some(server_cfg) = self.settings.get_pacs_server(server_id) else {
            return;
        };

        // Convert config::PacsServer to db::PacsServer for DicomScu
        let server = crate::db::PacsServer {
            id: 0,
            name: server_cfg.name.clone(),
            ae_title: server_cfg.ae_title.clone(),
            host: server_cfg.host.clone(),
            port: server_cfg.port as i32,
            our_ae_title: self.settings.local.ae_title.clone(),
        };

        let (tx, rx) = mpsc::channel();
        self.query_rx = Some(rx);
        self.query_state = QueryState::Querying;
        self.pacs_results.clear();
        self.selected_pacs = None;

        let mut params = QueryParams::new();
        if !self.pacs_filters.patient_name.is_empty() {
            params = params.with_patient_name(&self.pacs_filters.patient_name);
        }
        if !self.pacs_filters.patient_id.is_empty() {
            params = params.with_patient_id(&self.pacs_filters.patient_id);
        }
        if !self.pacs_filters.accession.is_empty() {
            params = params.with_accession(&self.pacs_filters.accession);
        }
        if !self.pacs_filters.study_date.is_empty() {
            params = params.with_date(&self.pacs_filters.study_date);
        }
        if !self.pacs_filters.modality.is_empty() {
            params = params.with_modality(&self.pacs_filters.modality);
        }

        std::thread::spawn(move || {
            let scu = DicomScu::new(server.to_config());
            let result = match scu.find_studies(&params) {
                Ok(results) => QueryResult::Success(results),
                Err(e) => QueryResult::Error(format!("{}", e)),
            };
            let _ = tx.send(result);
        });
    }

    fn check_query_results(&mut self) {
        if let Some(rx) = &self.query_rx {
            if let Ok(result) = rx.try_recv() {
                match result {
                    QueryResult::Success(mut results) => {
                        // Sort by study_date descending (newest first)
                        results.sort_by(|a, b| b.study_date.cmp(&a.study_date));
                        self.status = format!("Found {} studies", results.len());
                        self.pacs_results = results;
                    }
                    QueryResult::Error(e) => {
                        self.status = format!("Query failed: {}", e);
                        tracing::error!("PACS query failed: {}", e);
                    }
                }
                self.query_state = QueryState::Idle;
                self.query_rx = None;
            }
        }
    }

    fn start_retrieve(&mut self, _conn: &Connection) {
        let Some(result_idx) = self.selected_pacs else {
            return;
        };
        let Some(result) = self.pacs_results.get(result_idx) else {
            return;
        };
        let Some(server_id) = &self.selected_server else {
            return;
        };
        let Some(server_cfg) = self.settings.get_pacs_server(server_id) else {
            return;
        };

        let server = crate::db::PacsServer {
            id: 0,
            name: server_cfg.name.clone(),
            ae_title: server_cfg.ae_title.clone(),
            host: server_cfg.host.clone(),
            port: server_cfg.port as i32,
            our_ae_title: self.settings.local.ae_title.clone(),
        };

        let study_uid = result.study_instance_uid.clone();
        let study_info = StudyImportInfo {
            study_instance_uid: result.study_instance_uid.clone(),
            patient_name: result.patient_name.clone(),
            patient_id: result.patient_id.clone(),
            study_date: result.study_date.clone(),
            study_description: result.study_description.clone(),
            modality: result.modalities.clone(),
            accession_number: result.accession_number.clone(),
        };

        let (tx, rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.retrieve_rx = Some(rx);
        self.cancel_flag = Some(cancel_flag.clone());
        self.retrieve_state = RetrieveState::Retrieving {
            study_uid: study_uid.clone(),
            progress: AggregateProgress::new(),
            study_info: Some(study_info.clone()),
            db_study_id: None,
        };

        let storage_path = self.settings.storage_path();
        let scp_port = self.settings.local.port;
        let our_ae_title = self.settings.local.ae_title.clone();
        let parallel_workers = self.settings.pacs.parallel_workers;

        std::thread::spawn(move || {
            let scp = DicomScp::new(&our_ae_title, scp_port, storage_path.clone());

            match scp.start() {
                Ok(_) => {
                    tracing::info!("SCP started on port {}", scp_port);

                    // Use parallel retriever with configured number of workers
                    let config = ParallelRetrieveConfig {
                        max_workers: parallel_workers,
                        stagger_delay_ms: 100,
                    };
                    let retriever = ParallelRetriever::new(server.to_config(), config);

                    let (progress_tx, progress_rx) = mpsc::channel();
                    let study_uid_clone = study_uid.clone();
                    let storage_clone = storage_path.clone();
                    let cancel_for_retriever = cancel_flag.clone();

                    let retrieve_handle = std::thread::spawn(move || {
                        retriever.retrieve_study(
                            &study_uid_clone,
                            &storage_clone,
                            scp_port,
                            progress_tx,
                            cancel_for_retriever,
                        )
                    });

                    // Forward progress to UI
                    loop {
                        match progress_rx.try_recv() {
                            Ok(progress) => {
                                let is_complete = progress.is_complete;
                                let _ = tx.send(RetrieveResult::Progress(progress));
                                if is_complete {
                                    break;
                                }
                            }
                            Err(mpsc::TryRecvError::Disconnected) => break,
                            Err(mpsc::TryRecvError::Empty) => {
                                std::thread::sleep(std::time::Duration::from_millis(50));
                            }
                        }
                    }

                    match retrieve_handle.join() {
                        Ok(Ok(path)) => {
                            if !cancel_flag.load(Ordering::SeqCst) {
                                let _ = tx.send(RetrieveResult::Complete { path, study_info });
                            }
                        }
                        Ok(Err(e)) => {
                            if !cancel_flag.load(Ordering::SeqCst) {
                                let _ = tx.send(RetrieveResult::Error(format!("{}", e)));
                            }
                        }
                        Err(_) => {
                            let _ = tx.send(RetrieveResult::Error(
                                "Retrieve thread panicked".to_string(),
                            ));
                        }
                    }
                    scp.stop();
                }
                Err(e) => {
                    let _ = tx.send(RetrieveResult::Error(format!("Failed to start SCP: {}", e)));
                }
            }
        });
    }

    fn check_retrieve_results(&mut self, conn: &Connection) {
        // Collect results first to avoid borrow conflicts
        let results: Vec<RetrieveResult> = if let Some(rx) = &self.retrieve_rx {
            let mut collected = Vec::new();
            while let Ok(result) = rx.try_recv() {
                collected.push(result);
            }
            collected
        } else {
            return;
        };

        // Process collected results
        if !results.is_empty() {
            tracing::debug!("Processing {} retrieve results", results.len());
        }
        for result in results {
            match result {
                RetrieveResult::Progress(progress) => {
                    // Handle series completion event for incremental updates
                    if let Some(series_event) = progress.newly_completed_series.clone() {
                        tracing::info!(
                            "Series completed during retrieval: {} ({} images)",
                            series_event.series_uid,
                            series_event.num_images
                        );
                        self.handle_series_complete(conn, &series_event);
                    }

                    // Update progress in state
                    if let RetrieveState::Retrieving {
                        study_uid,
                        study_info,
                        db_study_id,
                        ..
                    } = &self.retrieve_state
                    {
                        self.retrieve_state = RetrieveState::Retrieving {
                            study_uid: study_uid.clone(),
                            progress,
                            study_info: study_info.clone(),
                            db_study_id: *db_study_id,
                        };
                    }
                }
                RetrieveResult::Complete {
                    path,
                    study_info: _,
                } => {
                    // Series were inserted incrementally, just finalize
                    tracing::info!("Retrieval complete, series already inserted to database");

                    // Reload local patients to ensure sidebar is up to date
                    self.request_load_studies();

                    self.retrieve_state = RetrieveState::Complete { path };
                    self.retrieve_rx = None;
                    return;
                }
                RetrieveResult::Error(message) => {
                    self.retrieve_state = RetrieveState::Error { message };
                    self.retrieve_rx = None;
                    return;
                }
            }
        }
    }

    /// Handle a series completion event - insert to DB and notify sidebar
    fn handle_series_complete(&mut self, conn: &Connection, event: &SeriesCompleteEvent) {
        // Track if this is the first series (study being created now)
        let is_first_series = !matches!(
            &self.retrieve_state,
            RetrieveState::Retrieving {
                db_study_id: Some(_),
                ..
            }
        );

        // Get or create the study ID
        let study_id = if let RetrieveState::Retrieving {
            db_study_id: Some(id),
            ..
        } = &self.retrieve_state
        {
            *id
        } else if let RetrieveState::Retrieving {
            study_info: Some(info),
            study_uid,
            progress,
            ..
        } = &self.retrieve_state
        {
            // First series - create the study record
            let retrieval_info = RetrievalStudyInfo {
                study_instance_uid: info.study_instance_uid.clone(),
                patient_name: info.patient_name.clone(),
                patient_id: info.patient_id.clone(),
                study_date: info.study_date.clone(),
                study_description: info.study_description.clone(),
                modality: info.modality.clone(),
                accession_number: info.accession_number.clone(),
            };

            match upsert_study_for_retrieval(conn, &retrieval_info, &event.storage_path) {
                Ok(id) => {
                    // Update state with the study ID
                    self.retrieve_state = RetrieveState::Retrieving {
                        study_uid: study_uid.clone(),
                        progress: progress.clone(),
                        study_info: Some(info.clone()),
                        db_study_id: Some(id),
                    };
                    id
                }
                Err(e) => {
                    tracing::error!("Failed to create study record: {}", e);
                    return;
                }
            }
        } else {
            tracing::warn!("Series complete but no study info available");
            return;
        };

        // If this is the first series and we should open the study, do so
        if is_first_series && self.open_after_retrieve {
            self.open_after_retrieve = false; // Reset flag

            // Get the study from DB to open in viewer
            if let Ok(Some(study)) = get_study(conn, study_id) {
                self.pending_action = Some(OpenStudyAction { study });
                // Close the database window
                self.open = false;
                tracing::info!("Opening study after first series arrived");
            }
        }

        // Insert the series
        match insert_series_from_event(
            conn,
            study_id,
            &event.series_uid,
            event.series_number,
            &event.series_description,
            &event.modality,
            event.num_images,
        ) {
            Ok(_) => {
                tracing::info!(
                    "Inserted series {} ({} images) during retrieval",
                    event.series_uid,
                    event.num_images
                );

                // Send event to app/sidebar for real-time update
                let added_event = SeriesAddedEvent {
                    study_id,
                    series_uid: event.series_uid.clone(),
                    series_description: event.series_description.clone(),
                    modality: event.modality.clone(),
                    num_images: event.num_images,
                    storage_path: event.storage_path.clone(),
                };

                if let Err(e) = self.series_added_tx.send(added_event) {
                    tracing::warn!("Failed to send series added event: {}", e);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to insert series {}: {}", event.series_uid, e);
            }
        }
    }
}

/// Group studies by patient
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

/// Format DICOM date (YYYYMMDD) for display
fn format_date(date: &str) -> String {
    if date.len() == 8 {
        format!("{}-{}-{}", &date[0..4], &date[4..6], &date[6..8])
    } else {
        date.to_string()
    }
}

/// Scan a directory for DICOM files and import series into the database
fn import_series_from_path(conn: &Connection, study_id: i64, path: &PathBuf, study_uid: &str) {
    use dicom::dictionary_std::tags;
    use dicom::object::open_file;

    let mut series_map: HashMap<String, SeriesImportInfo> = HashMap::new();

    let scan_dir = if path.is_dir()
        && std::fs::read_dir(path)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
    {
        path.clone()
    } else if let Some(parent) = path.parent() {
        parent.to_path_buf()
    } else {
        path.clone()
    };

    tracing::info!(
        "Scanning for DICOM files in {:?} for study {}",
        scan_dir,
        study_uid
    );

    if let Ok(entries) = std::fs::read_dir(&scan_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let file_path = entry.path();
            if !file_path.is_file() {
                continue;
            }

            if let Some(ext) = file_path.extension() {
                if ext != "dcm" {
                    continue;
                }
            }

            if let Ok(obj) = open_file(&file_path) {
                let file_study_uid = obj
                    .element(tags::STUDY_INSTANCE_UID)
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                if file_study_uid != study_uid {
                    continue;
                }

                if let Some(series_uid) = obj
                    .element(tags::SERIES_INSTANCE_UID)
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.to_string())
                {
                    let entry =
                        series_map
                            .entry(series_uid.clone())
                            .or_insert_with(|| SeriesImportInfo {
                                series_instance_uid: series_uid,
                                series_number: obj
                                    .element(tags::SERIES_NUMBER)
                                    .ok()
                                    .and_then(|e| e.to_int::<i32>().ok()),
                                series_description: obj
                                    .element(tags::SERIES_DESCRIPTION)
                                    .ok()
                                    .and_then(|e| e.to_str().ok())
                                    .map(|s| s.to_string()),
                                modality: obj
                                    .element(tags::MODALITY)
                                    .ok()
                                    .and_then(|e| e.to_str().ok())
                                    .map(|s| s.to_string()),
                                body_part: obj
                                    .element(tags::BODY_PART_EXAMINED)
                                    .ok()
                                    .and_then(|e| e.to_str().ok())
                                    .map(|s| s.to_string()),
                                num_images: 0,
                            });
                    entry.num_images += 1;
                }
            }
        }
    }

    for (_uid, info) in series_map {
        let series = Series {
            id: 0,
            study_id,
            series_instance_uid: info.series_instance_uid,
            series_number: info.series_number,
            series_description: info.series_description,
            modality: info.modality,
            num_images: info.num_images,
            body_part: info.body_part,
            num_frames: 0,
        };

        if let Err(e) = insert_series(conn, &series) {
            tracing::warn!("Failed to import series: {}", e);
        }
    }
}

struct SeriesImportInfo {
    series_instance_uid: String,
    series_number: Option<i32>,
    series_description: Option<String>,
    modality: Option<String>,
    body_part: Option<String>,
    num_images: i32,
}

/// Count DICOM files for a patient's studies
fn count_patient_dicom_files(studies: &[Study]) -> u32 {
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
