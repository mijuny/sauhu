//! PACS query and retrieve tab

use super::{OpenStudyAction, SeriesAddedEvent};
use crate::db::{
    get_study, insert_series_from_event, upsert_study_for_retrieval, RetrievalStudyInfo,
};
use crate::pacs::{
    AggregateProgress, DicomScp, DicomScu, ParallelRetrieveConfig, ParallelRetriever, QueryParams,
    SeriesCompleteEvent,
};
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

#[derive(Default, PartialEq)]
pub(super) enum QueryState {
    #[default]
    Idle,
    Querying,
}

pub(super) enum QueryResult {
    Success(Vec<crate::pacs::StudyResult>),
    Error(String),
}

#[derive(Default)]
#[allow(clippy::large_enum_variant)]
pub(super) enum RetrieveState {
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

pub(super) enum RetrieveResult {
    Progress(AggregateProgress),
    Complete {
        path: PathBuf,
        #[allow(dead_code)]
        study_info: StudyImportInfo,
    },
    Error(String),
}

#[derive(Clone)]
pub(super) struct StudyImportInfo {
    pub(super) study_instance_uid: String,
    pub(super) patient_name: String,
    pub(super) patient_id: String,
    pub(super) study_date: String,
    pub(super) study_description: String,
    pub(super) modality: String,
    pub(super) accession_number: String,
}

impl super::DatabaseWindow {
    pub(super) fn show_pacs_tab(&mut self, ui: &mut egui::Ui, conn: &Connection) {
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
                self.pacs_filters = super::SearchFilters::default();
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
                        self.pacs_filters = super::SearchFilters::default();
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

    pub(super) fn start_query(&mut self) {
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
            if tx.send(result).is_err() {
                tracing::warn!("PACS query channel closed before result delivered");
            }
        });
    }

    pub(super) fn check_query_results(&mut self) {
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

    pub(super) fn start_retrieve(&mut self, _conn: &Connection) {
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
                            if !cancel_flag.load(Ordering::SeqCst)
                                && tx.send(RetrieveResult::Complete { path, study_info }).is_err()
                            {
                                tracing::warn!("Retrieve channel closed before completion delivered");
                            }
                        }
                        Ok(Err(e)) => {
                            if !cancel_flag.load(Ordering::SeqCst)
                                && tx.send(RetrieveResult::Error(format!("{}", e))).is_err()
                            {
                                tracing::warn!("Retrieve channel closed before error delivered");
                            }
                        }
                        Err(_) => {
                            if tx.send(RetrieveResult::Error(
                                "Retrieve thread panicked".to_string(),
                            )).is_err() {
                                tracing::warn!("Retrieve channel closed before panic error delivered");
                            }
                        }
                    }
                    scp.stop();
                }
                Err(e) => {
                    if tx.send(RetrieveResult::Error(format!("Failed to start SCP: {}", e))).is_err() {
                        tracing::warn!("Retrieve channel closed before SCP error delivered");
                    }
                }
            }
        });
    }

    pub(super) fn check_retrieve_results(&mut self, conn: &Connection) {
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
    pub(super) fn handle_series_complete(&mut self, conn: &Connection, event: &SeriesCompleteEvent) {
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
