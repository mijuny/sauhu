
use crate::db::{get_study, Database};
use crate::dicom::{DicomFile, SyncInfo};
use crate::hanging_protocol::{self, SeriesCandidate};
use crate::ui::ViewportId;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use super::image_loading::ScanContext;
use super::{LoadedStudyWithSeries, SauhuApp};

/// Background loading system state
pub(super) struct BackgroundSystem {
    pub(super) request_tx: Sender<BackgroundRequest>,
    pub(super) result_rx: Receiver<BackgroundResult>,
    pub(super) loading_state: BackgroundLoadingState,
    pub(super) image_cache: crate::cache::ImageCache,
}

/// Request for background operations (to avoid blocking UI)
pub(super) enum BackgroundRequest {
    /// Load local patients from database
    LoadLocalPatients,
    /// Scan a directory for DICOM files
    ScanDirectory { path: PathBuf, context: ScanContext },
    /// Load patient series (DB queries + directory scans)
    LoadPatientSeries {
        patient_id: String,
        patient_name: String,
        studies: Vec<crate::db::Study>,
    },
    /// Validate and load files (dropped files, command line files)
    ValidateFiles {
        paths: Vec<PathBuf>,
        viewport_id: ViewportId,
    },
    /// Scan a directory, group by series, and return SeriesInfo (for quick fetch)
    ScanSeriesDirectory {
        path: PathBuf,
        viewport_id: ViewportId,
    },
}

/// Result from background operations
pub(super) enum BackgroundResult {
    /// Local studies loaded from database
    LocalStudiesLoaded {
        studies: Vec<crate::db::StudyWithImages>,
    },
    /// Directory scan complete - returns only paths (lightweight)
    DirectoryScanComplete {
        path: PathBuf,
        paths: Vec<PathBuf>,
        context: ScanContext,
    },
    /// Patient series loaded
    PatientSeriesLoaded {
        patient_id: String,
        patient_name: String,
        studies_with_series: Vec<LoadedStudyWithSeries>,
    },
    /// Files validated and ready to load
    FilesValidated {
        paths: Vec<PathBuf>,
        viewport_id: ViewportId,
    },
    /// Series directory scanned and grouped (for quick fetch)
    SeriesDirectoryScanned {
        series_info: Vec<crate::dicom::SeriesInfo>,
        viewport_id: ViewportId,
    },
    /// Error occurred
    Error { message: String },
}

/// Loading state for UI feedback
#[derive(Debug, Clone, Default)]
pub(super) enum BackgroundLoadingState {
    #[default]
    Idle,
    LoadingPatients,
    ScanningDirectory {
        #[allow(dead_code)]
        path: String,
    },
    LoadingSeries,
}

impl SauhuApp {
    /// Create background worker for non-blocking database and file operations
    pub(super) fn create_background_worker(
        db: Database,
    ) -> (Sender<BackgroundRequest>, Receiver<BackgroundResult>) {
        let (request_tx, request_rx) = mpsc::channel::<BackgroundRequest>();
        let (result_tx, result_rx) = mpsc::channel::<BackgroundResult>();

        thread::spawn(move || {
            while let Ok(request) = request_rx.recv() {
                let result = match request {
                    BackgroundRequest::LoadLocalPatients => {
                        match crate::db::get_all_studies_with_image_count(&db.conn()) {
                            Ok(studies) => BackgroundResult::LocalStudiesLoaded { studies },
                            Err(e) => BackgroundResult::Error {
                                message: format!("Failed to load studies: {}", e),
                            },
                        }
                    }
                    BackgroundRequest::ScanDirectory { path, context } => {
                        match crate::dicom::scan_directory(&path) {
                            Ok(files) => {
                                // Extract paths immediately - don't send heavy DicomFile objects
                                let paths: Vec<PathBuf> =
                                    files.iter().map(|d| PathBuf::from(&d.path)).collect();
                                // Drop the heavy DicomFile objects here in background thread
                                drop(files);
                                BackgroundResult::DirectoryScanComplete {
                                    path,
                                    paths,
                                    context,
                                }
                            }
                            Err(e) => BackgroundResult::Error {
                                message: format!("Failed to scan directory: {}", e),
                            },
                        }
                    }
                    BackgroundRequest::LoadPatientSeries {
                        patient_id,
                        patient_name,
                        studies,
                    } => {
                        tracing::info!(
                            "Loading series for patient {} ({}) - {} studies",
                            patient_name,
                            patient_id,
                            studies.len()
                        );
                        let studies_with_series: Vec<LoadedStudyWithSeries> = studies
                            .into_iter()
                            .map(|study| {
                                let study_path = std::path::PathBuf::from(&study.file_path);
                                let path_exists = study_path.exists();
                                let is_dir = study_path.is_dir();
                                let file_count = if is_dir {
                                    std::fs::read_dir(&study_path)
                                        .map(|d| d.count())
                                        .unwrap_or(0)
                                } else {
                                    0
                                };
                                tracing::info!(
                                    "Scanning study {} at path: {} (exists={}, is_dir={}, files={})",
                                    study.id,
                                    study.file_path,
                                    path_exists,
                                    is_dir,
                                    file_count
                                );
                                let db_series = crate::db::get_series_for_study(&db.conn(), study.id)
                                    .unwrap_or_default();
                                let series_info = match crate::dicom::scan_directory(&study.file_path) {
                                    Ok(files) => {
                                        let grouped = crate::dicom::group_files_by_series(files);
                                        tracing::info!(
                                            "Found {} files, grouped into {} series",
                                            grouped.iter().map(|s| s.files.len()).sum::<usize>(),
                                            grouped.len()
                                        );
                                        grouped
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Failed to scan study directory {}: {}",
                                            study.file_path,
                                            e
                                        );
                                        Vec::new()
                                    }
                                };
                                tracing::info!(
                                    "Study {} has {} db_series, {} series_info",
                                    study.id,
                                    db_series.len(),
                                    series_info.len()
                                );
                                LoadedStudyWithSeries {
                                    study,
                                    db_series,
                                    series_info,
                                }
                            })
                            .collect();
                        BackgroundResult::PatientSeriesLoaded {
                            patient_id,
                            patient_name,
                            studies_with_series,
                        }
                    }
                    BackgroundRequest::ScanSeriesDirectory { path, viewport_id } => {
                        match crate::dicom::scan_directory(&path) {
                            Ok(files) => {
                                let grouped = crate::dicom::group_files_by_series(files);
                                tracing::info!(
                                    "ScanSeriesDirectory: {} series found in {:?}",
                                    grouped.len(),
                                    path
                                );
                                BackgroundResult::SeriesDirectoryScanned {
                                    series_info: grouped,
                                    viewport_id,
                                }
                            }
                            Err(e) => BackgroundResult::Error {
                                message: format!("Failed to scan series directory: {}", e),
                            },
                        }
                    }
                    BackgroundRequest::ValidateFiles { paths, viewport_id } => {
                        use rayon::prelude::*;

                        // Split directories from single files: directories feed
                        // scan_directory (already parallelised internally),
                        // single-file paths get opened in parallel via rayon so
                        // large drag-and-drop sets don't stall the UI while we
                        // parse headers serially.
                        let (dir_paths, file_paths): (Vec<PathBuf>, Vec<PathBuf>) =
                            paths.into_iter().partition(|p| p.is_dir());

                        let mut dicom_paths: Vec<PathBuf> = Vec::new();
                        for path in dir_paths {
                            if let Ok(dcm_files) = crate::dicom::scan_directory(&path) {
                                dicom_paths.extend(
                                    dcm_files.into_iter().map(|dcm| PathBuf::from(&dcm.path)),
                                );
                            }
                        }
                        let mut validated: Vec<PathBuf> = file_paths
                            .into_par_iter()
                            .filter(|p| p.is_file())
                            .filter_map(|path| {
                                DicomFile::open(&path).ok().and_then(|dcm| {
                                    if dcm.rows().is_some() && dcm.columns().is_some() {
                                        Some(path)
                                    } else {
                                        None
                                    }
                                })
                            })
                            .collect();
                        dicom_paths.append(&mut validated);

                        // Sort by filename for consistent ordering
                        dicom_paths.sort();

                        BackgroundResult::FilesValidated {
                            paths: dicom_paths,
                            viewport_id,
                        }
                    }
                };
                let _ = result_tx.send(result);
            }
        });

        (request_tx, result_rx)
    }

    /// Check for background task results and send pending requests
    pub(super) fn check_background_results(&mut self) {
        // Check if database window needs patient loading
        if self.database_window.take_load_request() {
            self.background.loading_state = BackgroundLoadingState::LoadingPatients;
            let _ = self
                .background.request_tx
                .send(BackgroundRequest::LoadLocalPatients);
        }

        // Check if patient sidebar needs series loading
        if let Some(pending) = self.patient_sidebar.take_load_request() {
            tracing::info!(
                "Sending LoadPatientSeries request for {} ({}) with {} studies",
                pending.patient_name,
                pending.patient_id,
                pending.studies.len()
            );
            for s in &pending.studies {
                tracing::info!("  Study {} path: {}", s.id, s.file_path);
            }
            self.background.loading_state = BackgroundLoadingState::LoadingSeries;
            let _ = self
                .background.request_tx
                .send(BackgroundRequest::LoadPatientSeries {
                    patient_id: pending.patient_id,
                    patient_name: pending.patient_name,
                    studies: pending.studies,
                });
        }

        // Poll for results
        while let Ok(result) = self.background.result_rx.try_recv() {
            match result {
                BackgroundResult::LocalStudiesLoaded { studies } => {
                    self.database_window.handle_studies_loaded(studies);
                    self.background.loading_state = BackgroundLoadingState::Idle;
                }
                BackgroundResult::DirectoryScanComplete {
                    path,
                    paths,
                    context,
                } => {
                    self.handle_directory_scanned(path, paths, context);
                    self.background.loading_state = BackgroundLoadingState::Idle;
                }
                BackgroundResult::PatientSeriesLoaded {
                    patient_id,
                    patient_name,
                    studies_with_series,
                } => {
                    tracing::info!(
                        "Received PatientSeriesLoaded for {} ({}) with {} studies",
                        patient_name,
                        patient_id,
                        studies_with_series.len()
                    );

                    // If a hanging protocol is active, do bulk series assignment
                    if self.protocol_state.is_active() {
                        // Collect all series info for assignment
                        let all_series: Vec<&crate::dicom::SeriesInfo> = studies_with_series
                            .iter()
                            .flat_map(|sws| sws.series_info.iter())
                            .collect();

                        let candidates: Vec<SeriesCandidate> = all_series
                            .iter()
                            .map(|si| SeriesCandidate {
                                series_uid: si.series_uid.clone(),
                                description: si.description.clone().unwrap_or_default(),
                                num_images: si.files.len() as u32,
                            })
                            .collect();

                        let assignments = hanging_protocol::assign_all_series(
                            &mut self.protocol_state,
                            &candidates,
                        );

                        for (slot_index, series_uid) in &assignments {
                            if let Some(si) =
                                all_series.iter().find(|si| si.series_uid == *series_uid)
                            {
                                let sync_info = Some(SyncInfo::from_series_info(si));
                                tracing::info!(
                                    "Protocol: loading '{}' ({} images) to viewport {}",
                                    si.display_name,
                                    si.files.len(),
                                    slot_index
                                );
                                self.load_series_to_viewport(
                                    *slot_index,
                                    si.files.clone(),
                                    &si.display_name,
                                    sync_info,
                                );
                            }
                        }
                    }

                    self.patient_sidebar
                        .handle_series_loaded(studies_with_series);
                    self.background.loading_state = BackgroundLoadingState::Idle;
                    tracing::info!(
                        "Loaded series for patient {} ({})",
                        patient_name,
                        patient_id
                    );
                }
                BackgroundResult::FilesValidated { paths, viewport_id } => {
                    self.handle_files_validated(paths, viewport_id);
                    self.background.loading_state = BackgroundLoadingState::Idle;
                }
                BackgroundResult::SeriesDirectoryScanned {
                    series_info,
                    viewport_id,
                } => {
                    if let Some(si) = series_info.into_iter().max_by_key(|s| s.files.len()) {
                        let sync_info = Some(SyncInfo::from_series_info(&si));
                        tracing::info!(
                            "Quick fetch loading series '{}' ({} images) to viewport {}",
                            si.display_name,
                            si.files.len(),
                            viewport_id
                        );
                        self.load_series_to_viewport(
                            viewport_id,
                            si.files,
                            &si.display_name,
                            sync_info,
                        );
                    }
                    self.background.loading_state = BackgroundLoadingState::Idle;
                }
                BackgroundResult::Error { message } => {
                    tracing::error!("Background task error: {}", message);
                    self.status = format!("Error: {}", message);
                    self.background.loading_state = BackgroundLoadingState::Idle;
                }
            }
        }

        // Check for series added events from retrieval (real-time sidebar updates)
        while let Ok(event) = self.series_added_rx.try_recv() {
            tracing::info!(
                "Received series added event: study_id={}, series={}",
                event.study_id,
                event.series_uid
            );

            // If sidebar doesn't have this study yet, try to add it from the database
            if !self.patient_sidebar.is_showing_study(event.study_id) {
                // Get the study from DB and add it to sidebar
                if let Ok(Some(study)) = get_study(&self.db.conn(), event.study_id) {
                    // Check if this patient is currently being viewed
                    let is_current_patient = self
                        .current_patient
                        .as_ref()
                        .map(|cp| cp.patient_id == study.patient_id.clone().unwrap_or_default())
                        .unwrap_or(false);

                    if is_current_patient {
                        tracing::info!(
                            "Adding study {} to sidebar for incremental updates",
                            event.study_id
                        );
                        self.patient_sidebar.add_study_for_retrieval(study);
                    }
                }
            }

            // Now try to add the series
            if self.patient_sidebar.is_showing_study(event.study_id) {
                self.patient_sidebar.add_series_incremental(&event);
                tracing::info!(
                    "Added series {} to sidebar during retrieval",
                    event.series_uid
                );
            } else {
                tracing::debug!(
                    "Series added but not for current patient: study_id={}",
                    event.study_id
                );
            }
        }
    }

    /// Handle directory scan completion
    pub(super) fn handle_directory_scanned(
        &mut self,
        _path: PathBuf,
        paths: Vec<PathBuf>,
        context: ScanContext,
    ) {
        // Paths are now passed directly (no heavy DicomFile objects)
        match context {
            ScanContext::LoadFolder => {
                self.handle_scanned_paths(paths);
            }
            ScanContext::OpenPatient => {
                // This is handled via LoadPatientSeries now
                self.handle_scanned_paths(paths);
            }
            ScanContext::QuickFetchComplete { accession } => {
                self.status = format!("Loaded study {}", accession);
                self.handle_scanned_paths(paths);
            }
            ScanContext::LoadToViewport { viewport_id } => {
                self.handle_scanned_paths_to_viewport(paths, viewport_id);
            }
        }
    }

    /// Handle scanned paths directly (files already validated during scan)
    pub(super) fn handle_scanned_paths(&mut self, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            self.status = "No valid DICOM files found".to_string();
            return;
        }

        let viewport_id = self.viewport_manager.active_viewport();
        let path_count = paths.len();

        // Prefetch entire series to cache for smooth scrolling
        let t0 = std::time::Instant::now();
        self.background.image_cache.prefetch_series(&paths);
        tracing::info!(
            "prefetch_series({} files) took {:?}",
            path_count,
            t0.elapsed()
        );

        let start_at_middle = self.settings.viewer.start_at_middle_slice;
        self.viewport_manager.load_series(
            viewport_id,
            paths.clone(),
            "Files",
            None,
            start_at_middle,
        );
        // Get the actual start index from the slot
        let start_index = self
            .viewport_manager
            .get_slot(viewport_id)
            .map(|s| s.current_index)
            .unwrap_or(0);
        self.load_image_for_viewport(viewport_id, start_index);

        self.status = format!("Loaded {} images - scroll to navigate", path_count);
    }

    /// Handle scanned paths and load to a specific viewport (for hanging protocol)
    pub(super) fn handle_scanned_paths_to_viewport(&mut self, paths: Vec<PathBuf>, viewport_id: ViewportId) {
        if paths.is_empty() {
            tracing::warn!("No valid DICOM files found for viewport {}", viewport_id);
            return;
        }

        let path_count = paths.len();
        self.background.image_cache.prefetch_series(&paths);

        let start_at_middle = self.settings.viewer.start_at_middle_slice;
        self.viewport_manager
            .load_series(viewport_id, paths, "Files", None, start_at_middle);
        let start_index = self
            .viewport_manager
            .get_slot(viewport_id)
            .map(|s| s.current_index)
            .unwrap_or(0);
        self.load_image_for_viewport(viewport_id, start_index);

        tracing::info!("Loaded {} images to viewport {}", path_count, viewport_id);
    }
}
