//! Database Window
//!
//! Browse local patients and query PACS for studies.

mod anonymize;
mod local_browser;
mod pacs_query;

use crate::config::Settings;
use crate::db::StudyWithImages;
use pacs_query::{QueryState, RetrieveState};
use rusqlite::Connection;
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::Arc;

#[allow(unused_imports)] // planned patient grouping in database UI
pub use anonymize::group_studies_by_patient;

use self::anonymize::{AnonymizeProgressResult, AnonymizeState};
use self::local_browser::LocalSortColumn;
use self::pacs_query::{QueryResult, RetrieveResult};

/// Patient info aggregated from studies
#[derive(Debug, Clone)]
#[allow(dead_code)] // planned patient grouping in database UI
pub struct PatientInfo {
    pub patient_id: String,
    pub patient_name: String,
    pub studies: Vec<crate::db::Study>,
}

/// Action to open a study in the viewer
#[derive(Debug, Clone)]
pub struct OpenStudyAction {
    pub study: crate::db::Study,
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
    #[allow(dead_code)]
    pub storage_path: std::path::PathBuf,
}

#[derive(Default, PartialEq)]
pub(super) enum Tab {
    #[default]
    Local,
    Pacs,
}

/// Search filter fields for PACS
#[derive(Default)]
pub(super) struct SearchFilters {
    pub(super) patient_name: String,
    pub(super) patient_id: String,
    pub(super) accession: String,
    pub(super) study_date: String,
    pub(super) modality: String,
}

/// State for delete operation
#[derive(Default, PartialEq)]
pub(super) enum DeleteState {
    #[default]
    Idle,
    /// Delete failed
    Error { message: String },
}

/// State for database window
pub struct DatabaseWindow {
    /// Whether the window is open
    pub open: bool,
    /// Application settings
    pub(in crate::ui::database_window) settings: Arc<Settings>,
    /// Local studies with image counts
    pub(in crate::ui::database_window) local_studies: Vec<StudyWithImages>,
    /// Search filter for local studies
    pub(in crate::ui::database_window) local_filter: String,
    /// Selected study IDs for multi-select
    pub(in crate::ui::database_window) selected_studies: HashSet<i64>,
    /// Anchor study ID for shift-click range selection
    pub(in crate::ui::database_window) last_selected_id: Option<i64>,
    /// Sort column for local tab
    pub(in crate::ui::database_window) local_sort_column: LocalSortColumn,
    /// Sort ascending (false = descending, newest first for date)
    pub(in crate::ui::database_window) local_sort_ascending: bool,
    /// Current tab (Local or PACS)
    pub(in crate::ui::database_window) current_tab: Tab,
    /// PACS search filters
    pub(in crate::ui::database_window) pacs_filters: SearchFilters,
    /// PACS query results
    pub(in crate::ui::database_window) pacs_results: Vec<crate::pacs::StudyResult>,
    /// Selected PACS result index
    pub(in crate::ui::database_window) selected_pacs: Option<usize>,
    /// Selected PACS server id
    pub(in crate::ui::database_window) selected_server: Option<String>,
    /// Query state
    pub(in crate::ui::database_window) query_state: QueryState,
    /// Retrieve state
    pub(in crate::ui::database_window) retrieve_state: RetrieveState,
    /// Channel for query results
    pub(in crate::ui::database_window) query_rx: Option<mpsc::Receiver<QueryResult>>,
    /// Channel for retrieve progress
    pub(in crate::ui::database_window) retrieve_rx: Option<mpsc::Receiver<RetrieveResult>>,
    /// Cancel flag for retrieve
    pub(in crate::ui::database_window) cancel_flag: Option<Arc<AtomicBool>>,
    /// Status message
    pub(in crate::ui::database_window) status: String,
    /// Pending action (open study)
    pub(in crate::ui::database_window) pending_action: Option<OpenStudyAction>,
    /// Whether we're loading local studies
    pub(in crate::ui::database_window) loading_studies: bool,
    /// Whether we need to request a study load
    pub(in crate::ui::database_window) needs_study_load: bool,
    /// Channel to send series added events to app
    pub(in crate::ui::database_window) series_added_tx: mpsc::Sender<SeriesAddedEvent>,
    /// Whether to open study in viewer when first series arrives
    pub(in crate::ui::database_window) open_after_retrieve: bool,
    /// Anonymization state
    pub(in crate::ui::database_window) anonymize_state: AnonymizeState,
    /// Channel for anonymization progress
    pub(in crate::ui::database_window) anonymize_rx: Option<mpsc::Receiver<AnonymizeProgressResult>>,
    /// Delete state
    pub(in crate::ui::database_window) delete_state: DeleteState,
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
    #[allow(dead_code)] // synchronous alternative to request_load_studies
    pub fn load_local_studies(&mut self, conn: &Connection) {
        match crate::db::get_all_studies_with_image_count(conn) {
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
    #[allow(dead_code)] // public API for study loading state
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
}
