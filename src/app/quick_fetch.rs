#![allow(dead_code)]

use crate::db::{get_study, upsert_study_for_retrieval, PacsServer, RetrievalStudyInfo, Study};
use crate::hanging_protocol::{self, SeriesCandidate};
use crate::pacs::{DicomScp, DicomScu, ParallelRetrieveConfig, ParallelRetriever, QueryParams};
use crate::ui::{ViewportId, ViewportLayout};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use super::{CurrentPatient, SauhuApp};

/// State for quick fetch (Ctrl+G) operation
#[derive(Debug, Clone, Default)]
pub(super) enum QuickFetchState {
    #[default]
    Idle,
    Querying {
        accession: String,
    },
    Retrieving {
        accession: String,
        description: String,
        completed: u32,
        first_series_loaded: bool,
    },
    Opening {
        study_id: i64,
    },
    Error {
        message: String,
    },
}

/// Study info for database insertion during quick fetch
#[derive(Debug, Clone)]
pub(super) struct QuickFetchStudyInfo {
    pub study_uid: String,
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
    pub modality: String,
    pub accession_number: String,
}

/// Result from quick fetch background thread
pub(super) enum QuickFetchResult {
    QueryComplete {
        accession: String,
        study_uid: String,
        description: String,
        study_info: QuickFetchStudyInfo,
    },
    QueryNotFound {
        accession: String,
    },
    QueryError {
        message: String,
    },
    RetrieveProgress {
        completed: u32,
    },
    /// A series has completed - can be opened immediately
    SeriesComplete {
        series_uid: String,
        series_number: Option<i32>,
        series_description: String,
        modality: String,
        num_images: u32,
        storage_path: PathBuf,
    },
    RetrieveComplete {
        path: PathBuf,
        accession: String,
        study_info: QuickFetchStudyInfo,
    },
    RetrieveError {
        message: String,
    },
}

impl SauhuApp {
    /// Start quick fetch from clipboard (Ctrl+G)
    pub(super) fn start_quick_fetch(&mut self) {
        // Don't start if already fetching
        if !matches!(
            self.quick_fetch_state,
            QuickFetchState::Idle | QuickFetchState::Error { .. }
        ) {
            return;
        }

        // Read accession number from clipboard
        let accession = match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.get_text() {
                Ok(text) => text.trim().to_string(),
                Err(e) => {
                    self.quick_fetch_state = QuickFetchState::Error {
                        message: format!("Clipboard error: {}", e),
                    };
                    self.status = "Failed to read clipboard".to_string();
                    return;
                }
            },
            Err(e) => {
                self.quick_fetch_state = QuickFetchState::Error {
                    message: format!("Clipboard error: {}", e),
                };
                self.status = "Failed to access clipboard".to_string();
                return;
            }
        };

        self.start_quick_fetch_with_accession(accession);
    }

    /// Start quick fetch with a given accession number (called from IPC or clipboard)
    pub(super) fn start_quick_fetch_with_accession(&mut self, accession: String) {
        // Don't start if already fetching
        if !matches!(
            self.quick_fetch_state,
            QuickFetchState::Idle | QuickFetchState::Error { .. }
        ) {
            tracing::warn!(
                "Quick fetch already in progress, ignoring request for AC={}",
                accession
            );
            return;
        }

        if accession.is_empty() {
            self.quick_fetch_state = QuickFetchState::Error {
                message: "Accession number is empty".to_string(),
            };
            self.status = "Accession number is empty".to_string();
            return;
        }

        // Get default PACS server
        let (_server_id, server_config) = match self.settings.get_default_pacs_server() {
            Some(s) => s,
            None => {
                self.quick_fetch_state = QuickFetchState::Error {
                    message: "No PACS server configured".to_string(),
                };
                self.status = "No PACS server configured".to_string();
                return;
            }
        };

        tracing::info!("Quick fetch: AC={} from {}", accession, server_config.name);
        self.status = format!("Querying {} for AC {}...", server_config.name, accession);
        self.quick_fetch_state = QuickFetchState::Querying {
            accession: accession.clone(),
        };

        // Create channel for results
        let (tx, rx) = mpsc::channel();
        self.quick_fetch_rx = Some(rx);

        // Build PACS server for SCU
        let pacs_server = PacsServer {
            id: 0, // Not from DB, doesn't matter
            name: server_config.name.clone(),
            ae_title: server_config.ae_title.clone(),
            host: server_config.host.clone(),
            port: server_config.port as i32,
            our_ae_title: self.settings.local.ae_title.clone(),
        };

        let storage_path = self.settings.storage_path();
        let scp_port = self.settings.local.port;
        let our_ae_title = self.settings.local.ae_title.clone();

        // Spawn background thread for query + retrieve
        thread::spawn(move || {
            let scu = DicomScu::new(pacs_server.to_config());

            // Query by accession number
            let params = QueryParams::new().with_accession(&accession);
            match scu.find_studies(&params) {
                Ok(results) => {
                    if results.is_empty() {
                        let _ = tx.send(QuickFetchResult::QueryNotFound { accession });
                        return;
                    }

                    // Take first result
                    let study = &results[0];
                    let study_uid = study.study_instance_uid.clone();
                    let description = study.study_description.clone();

                    // Create study info for database insertion
                    let study_info = QuickFetchStudyInfo {
                        study_uid: study.study_instance_uid.clone(),
                        patient_name: study.patient_name.clone(),
                        patient_id: study.patient_id.clone(),
                        study_date: study.study_date.clone(),
                        study_description: study.study_description.clone(),
                        modality: study.modalities.clone(),
                        accession_number: study.accession_number.clone(),
                    };

                    let _ = tx.send(QuickFetchResult::QueryComplete {
                        accession: accession.clone(),
                        study_uid: study_uid.clone(),
                        description: description.clone(),
                        study_info: study_info.clone(),
                    });

                    // Start SCP and retrieve using parallel retriever
                    let scp = DicomScp::new(&our_ae_title, scp_port, storage_path.clone());
                    match scp.start() {
                        Ok(_) => {
                            let (progress_tx, progress_rx) = mpsc::channel();

                            // Use parallel retriever with 4 workers
                            let config = ParallelRetrieveConfig {
                                max_workers: 4,
                                stagger_delay_ms: 100,
                            };
                            let retriever = ParallelRetriever::new(pacs_server.to_config(), config);

                            let study_uid_clone = study_uid.clone();
                            let storage_clone = storage_path.clone();
                            let cancel_flag =
                                std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                            let cancel_for_retriever = cancel_flag.clone();

                            let retrieve_handle = thread::spawn(move || {
                                retriever.retrieve_study(
                                    &study_uid_clone,
                                    &storage_clone,
                                    scp_port,
                                    progress_tx,
                                    cancel_for_retriever,
                                )
                            });

                            // Forward progress updates including series completions
                            loop {
                                match progress_rx.try_recv() {
                                    Ok(progress) => {
                                        // Send progress update
                                        let _ = tx.send(QuickFetchResult::RetrieveProgress {
                                            completed: progress.completed_images,
                                        });

                                        // Send series completion event if one just finished
                                        if let Some(ref series_event) =
                                            progress.newly_completed_series
                                        {
                                            tracing::info!(
                                                "Quick fetch: series complete - {} ({} images)",
                                                series_event.series_description,
                                                series_event.num_images
                                            );
                                            let _ = tx.send(QuickFetchResult::SeriesComplete {
                                                series_uid: series_event.series_uid.clone(),
                                                series_number: series_event.series_number,
                                                series_description: series_event
                                                    .series_description
                                                    .clone(),
                                                modality: series_event.modality.clone(),
                                                num_images: series_event.num_images,
                                                storage_path: series_event.storage_path.clone(),
                                            });
                                        }

                                        if progress.is_complete {
                                            break;
                                        }
                                    }
                                    Err(mpsc::TryRecvError::Disconnected) => break,
                                    Err(mpsc::TryRecvError::Empty) => {
                                        thread::sleep(std::time::Duration::from_millis(50));
                                    }
                                }
                            }

                            match retrieve_handle.join() {
                                Ok(Ok(path)) => {
                                    let _ = tx.send(QuickFetchResult::RetrieveComplete {
                                        path,
                                        accession,
                                        study_info,
                                    });
                                }
                                Ok(Err(e)) => {
                                    let _ = tx.send(QuickFetchResult::RetrieveError {
                                        message: format!("{}", e),
                                    });
                                }
                                Err(_) => {
                                    let _ = tx.send(QuickFetchResult::RetrieveError {
                                        message: "Retrieve thread panicked".to_string(),
                                    });
                                }
                            }

                            scp.stop();
                        }
                        Err(e) => {
                            let _ = tx.send(QuickFetchResult::RetrieveError {
                                message: format!("Failed to start SCP: {}", e),
                            });
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(QuickFetchResult::QueryError {
                        message: format!("{}", e),
                    });
                }
            }
        });
    }

    /// Check for quick fetch results from background thread
    pub(super) fn check_quick_fetch_results(&mut self) {
        // Collect results first to avoid borrow issues
        let results: Vec<QuickFetchResult> = if let Some(rx) = &self.quick_fetch_rx {
            let mut collected = Vec::new();
            while let Ok(result) = rx.try_recv() {
                collected.push(result);
            }
            collected
        } else {
            return;
        };

        let mut completed_info: Option<(PathBuf, QuickFetchStudyInfo)> = None;
        let mut clear_rx = false;
        let mut series_to_load: Vec<(PathBuf, ViewportId)> = Vec::new();

        for result in results {
            match result {
                QuickFetchResult::QueryComplete {
                    accession,
                    description,
                    study_uid,
                    study_info,
                } => {
                    self.quick_fetch_state = QuickFetchState::Retrieving {
                        accession: accession.clone(),
                        description: description.clone(),
                        completed: 0,
                        first_series_loaded: false,
                    };
                    self.status = format!("Retrieving: {} ({})", description, accession);

                    // Store study info for when series complete
                    // Insert study to DB early so series can be added incrementally
                    let storage_path = self.settings.storage_path();
                    let study_dir = storage_path.join(study_uid.replace('.', "_"));
                    let retrieval_info = RetrievalStudyInfo {
                        study_instance_uid: study_info.study_uid.clone(),
                        patient_name: study_info.patient_name.clone(),
                        patient_id: study_info.patient_id.clone(),
                        study_date: study_info.study_date.clone(),
                        study_description: study_info.study_description.clone(),
                        modality: study_info.modality.clone(),
                        accession_number: study_info.accession_number.clone(),
                    };

                    // Insert study to DB and get study for sidebar setup
                    // Do DB work first to avoid borrow issues
                    let study_for_sidebar: Option<Study> = {
                        let conn = self.db.conn();
                        if let Ok(study_id) =
                            upsert_study_for_retrieval(&conn, &retrieval_info, &study_dir)
                        {
                            tracing::info!(
                                "Quick fetch: created study {} in DB for incremental updates",
                                study_id
                            );
                            get_study(&conn, study_id).ok().flatten()
                        } else {
                            None
                        }
                    }; // conn guard dropped here

                    // Now we can call mutable methods
                    if let Some(study) = study_for_sidebar {
                        // Set up patient context and sidebar (but don't load files yet)
                        self.clear_patient();

                        // Match hanging protocol for this study
                        let study_mod = study.modality.as_deref().unwrap_or(&study_info.modality);
                        let study_desc_str = study
                            .study_description
                            .as_deref()
                            .unwrap_or(&study_info.study_description);
                        if let Some(protocol) = hanging_protocol::match_protocol(
                            &self.settings.hanging_protocol,
                            study_mod,
                            study_desc_str,
                        ) {
                            tracing::info!(
                                "Quick fetch: hanging protocol matched: '{}' (layout: {})",
                                protocol.name,
                                protocol.layout
                            );
                            if let Some(layout) = ViewportLayout::from_config_str(&protocol.layout) {
                                self.viewport_manager.set_layout(layout);
                            }
                            self.protocol_state.active_protocol = Some(protocol.clone());
                        }

                        let patient_id = study.patient_id.clone().unwrap_or_default();
                        let patient_name = study.patient_name.clone().unwrap_or_default();
                        self.current_patient = Some(CurrentPatient {
                            patient_id: patient_id.clone(),
                            patient_name: patient_name.clone(),
                            studies: vec![study.clone()],
                        });
                        // Add study to sidebar with empty series (will be populated incrementally)
                        self.patient_sidebar.add_study_for_retrieval(study);
                    }
                }
                QuickFetchResult::QueryNotFound { accession } => {
                    self.quick_fetch_state = QuickFetchState::Error {
                        message: format!("Not found: {}", accession),
                    };
                    self.status = format!("AC {} not found", accession);
                    clear_rx = true;
                }
                QuickFetchResult::QueryError { message } => {
                    self.quick_fetch_state = QuickFetchState::Error {
                        message: message.clone(),
                    };
                    self.status = format!("Query error: {}", message);
                    clear_rx = true;
                }
                QuickFetchResult::RetrieveProgress { completed } => {
                    if let QuickFetchState::Retrieving {
                        ref accession,
                        ref description,
                        first_series_loaded,
                        ..
                    } = self.quick_fetch_state.clone()
                    {
                        self.quick_fetch_state = QuickFetchState::Retrieving {
                            accession: accession.clone(),
                            description: description.clone(),
                            completed,
                            first_series_loaded,
                        };
                        self.status = format!("Retrieving: {} - {} images", description, completed);
                    }
                }
                QuickFetchResult::SeriesComplete {
                    series_uid,
                    series_number,
                    series_description,
                    modality,
                    num_images,
                    storage_path,
                } => {
                    tracing::info!(
                        "Quick fetch: series {} complete ({} images) at {:?}",
                        series_description,
                        num_images,
                        storage_path
                    );

                    // Add series to sidebar
                    self.patient_sidebar.add_series_from_retrieval(
                        &series_uid,
                        series_number,
                        &series_description,
                        &modality,
                        num_images,
                        &storage_path,
                    );

                    // Assign series to viewport via hanging protocol, or load first series as fallback
                    if self.protocol_state.is_active() {
                        let candidate = SeriesCandidate {
                            series_uid: series_uid.clone(),
                            description: series_description.clone(),
                            num_images,
                        };
                        if let Some(slot_index) = hanging_protocol::try_assign_series(
                            &mut self.protocol_state,
                            &candidate,
                        ) {
                            tracing::info!(
                                "Protocol assigned series '{}' to slot {}",
                                series_description,
                                slot_index
                            );
                            series_to_load.push((storage_path, slot_index));
                        }
                    } else {
                        // No protocol - load first series immediately (only once)
                        if let QuickFetchState::Retrieving {
                            ref accession,
                            ref description,
                            completed,
                            first_series_loaded,
                        } = self.quick_fetch_state.clone()
                        {
                            if !first_series_loaded {
                                series_to_load
                                    .push((storage_path, self.viewport_manager.active_viewport()));
                                // Mark that we've loaded the first series
                                self.quick_fetch_state = QuickFetchState::Retrieving {
                                    accession: accession.clone(),
                                    description: description.clone(),
                                    completed,
                                    first_series_loaded: true,
                                };
                            }
                        }
                    }
                }
                QuickFetchResult::RetrieveComplete {
                    path,
                    accession,
                    study_info,
                } => {
                    // Update study path in case it differs from what we expected
                    let retrieval_info = RetrievalStudyInfo {
                        study_instance_uid: study_info.study_uid.clone(),
                        patient_name: study_info.patient_name.clone(),
                        patient_id: study_info.patient_id.clone(),
                        study_date: study_info.study_date.clone(),
                        study_description: study_info.study_description.clone(),
                        modality: study_info.modality.clone(),
                        accession_number: study_info.accession_number.clone(),
                    };
                    let _ = upsert_study_for_retrieval(&self.db.conn(), &retrieval_info, &path);

                    self.status = format!("Retrieved: {}", accession);
                    self.quick_fetch_state = QuickFetchState::Idle;
                    completed_info = Some((path, study_info));
                    clear_rx = true;
                }
                QuickFetchResult::RetrieveError { message } => {
                    self.quick_fetch_state = QuickFetchState::Error {
                        message: message.clone(),
                    };
                    self.status = format!("Retrieve error: {}", message);
                    clear_rx = true;
                }
            }
        }

        if clear_rx {
            self.quick_fetch_rx = None;
        }

        // Load series to their assigned viewports (protocol or first-series fallback)
        // Use ScanSeriesDirectory to get proper series grouping, sorting, and sync info
        for (path, viewport_id) in series_to_load {
            tracing::info!(
                "Quick fetch: scanning series from {:?} for viewport {}",
                path,
                viewport_id
            );
            let _ = self.bg_request_tx.send(
                super::background::BackgroundRequest::ScanSeriesDirectory {
                    path,
                    viewport_id,
                },
            );
        }

        // Handle retrieve completion: rebuild sidebar with proper series info
        if let Some((_path, study_info)) = completed_info {
            tracing::info!("Quick fetch complete - rebuilding sidebar from DICOM files");

            // Get the study from DB (was inserted on QueryComplete)
            let study = {
                let conn = self.db.conn();
                crate::db::get_studies_by_patient_id(&conn, &study_info.patient_id)
                    .unwrap_or_default()
                    .into_iter()
                    .find(|s| s.study_instance_uid == study_info.study_uid)
            };

            if let Some(study) = study {
                // Trigger the same LoadPatientSeries flow as DB open
                // This re-scans files with group_files_by_series for proper
                // series names, sorting, sync info, and filtering
                self.patient_sidebar.request_set_patient(
                    &study_info.patient_id,
                    &study_info.patient_name,
                    vec![study],
                );
            }
        }
    }
}
