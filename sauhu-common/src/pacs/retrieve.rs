//! Parallel series retrieval for faster C-MOVE operations
//!
//! Instead of retrieving an entire study with a single C-MOVE,
//! this module queries series first and retrieves them in parallel.

use crate::PacsServerConfig;
use crate::pacs::{
    AggregateProgress, DicomScu, RetrieveProgress, SeriesCompleteEvent, SeriesResult,
};
use anyhow::Result;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Configuration for parallel retrieval
#[derive(Debug, Clone)]
pub struct ParallelRetrieveConfig {
    /// Maximum concurrent SCU associations (default: 4)
    pub max_workers: usize,
    /// Stagger delay between starting workers (ms)
    pub stagger_delay_ms: u64,
}

impl Default for ParallelRetrieveConfig {
    fn default() -> Self {
        Self {
            max_workers: 4,
            stagger_delay_ms: 100,
        }
    }
}

/// Progress update from a single series worker
#[derive(Debug)]
struct WorkerProgress {
    series_uid: String,
    progress: RetrieveProgress,
    /// Set when series retrieval completes successfully
    completion_event: Option<SeriesCompleteEvent>,
}

/// Parallel retriever that fetches multiple series concurrently
pub struct ParallelRetriever {
    server: PacsServerConfig,
    config: ParallelRetrieveConfig,
}

impl ParallelRetriever {
    /// Create a new parallel retriever
    pub fn new(server: PacsServerConfig, config: ParallelRetrieveConfig) -> Self {
        Self { server, config }
    }

    /// Retrieve a study by fetching series in parallel
    ///
    /// 1. Queries series in the study
    /// 2. Spawns worker threads (up to max_workers)
    /// 3. Workers pull series from queue and retrieve them
    /// 4. Aggregates progress from all workers
    /// 5. Falls back to study-level retrieve if series query fails
    pub fn retrieve_study(
        &self,
        study_uid: &str,
        cache_dir: &std::path::Path,
        scp_port: u16,
        progress_tx: Sender<AggregateProgress>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<PathBuf> {
        tracing::info!(
            "Parallel retrieve study {} with {} workers",
            study_uid,
            self.config.max_workers
        );

        // Create study directory
        let study_dir = cache_dir.join(study_uid.replace('.', "_"));
        std::fs::create_dir_all(&study_dir)?;

        // Query series in the study
        let scu = DicomScu::new(self.server.clone());
        let series_list = match scu.find_series(study_uid) {
            Ok(series) => {
                if series.is_empty() {
                    tracing::warn!("No series found, falling back to study-level retrieve");
                    return self.fallback_study_retrieve(
                        study_uid,
                        cache_dir,
                        scp_port,
                        progress_tx,
                        cancel_flag,
                    );
                }
                series
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to query series: {}, falling back to study-level retrieve",
                    e
                );
                return self.fallback_study_retrieve(
                    study_uid,
                    cache_dir,
                    scp_port,
                    progress_tx,
                    cancel_flag,
                );
            }
        };

        let series_list_len = series_list.len();
        tracing::info!("Found {} series to retrieve", series_list_len);

        // Filter out non-image modalities (SR, KO, PR) that the viewer can't display.
        // Many PACS servers refuse C-MOVE for these SOP classes, and consecutive failures
        // from non-image series can kill all worker threads before image series are processed.
        let image_series: Vec<SeriesResult> = series_list
            .into_iter()
            .filter(|s| {
                let dominated = matches!(
                    s.modality.trim().to_uppercase().as_str(),
                    "SR" | "KO"
                        | "PR"
                        | "SEG"
                        | "REG"
                        | "FID"
                        | "RWV"
                        | "PLAN"
                        | "RTDOSE"
                        | "RTSTRUCT"
                        | "RTPLAN"
                        | "RTRECORD"
                );
                if dominated {
                    tracing::debug!(
                        "Skipping non-image series {} (modality: {})",
                        s.series_instance_uid,
                        s.modality
                    );
                }
                !dominated
            })
            .collect();

        tracing::info!(
            "Retrieving {} image series (skipped {} non-image)",
            image_series.len(),
            series_list_len - image_series.len()
        );

        if image_series.is_empty() {
            tracing::warn!(
                "No image series found after filtering, falling back to study-level retrieve"
            );
            return self.fallback_study_retrieve(
                study_uid,
                cache_dir,
                scp_port,
                progress_tx,
                cancel_flag,
            );
        }

        // Calculate total expected images
        let total_images: u32 = image_series
            .iter()
            .filter_map(|s| s.num_instances)
            .map(|n| n as u32)
            .sum();

        // Initialize aggregate progress
        let mut aggregate = AggregateProgress::new();
        aggregate.total_series = image_series.len() as u32;
        aggregate.total_images = total_images;
        let _ = progress_tx.send(aggregate.clone());

        // Create work queue
        let work_queue: Arc<Mutex<VecDeque<SeriesResult>>> =
            Arc::new(Mutex::new(VecDeque::from(image_series)));

        // Create channel for worker progress updates
        let (worker_tx, worker_rx): (Sender<WorkerProgress>, Receiver<WorkerProgress>) =
            mpsc::channel();

        // Track active workers
        let active_workers = Arc::new(AtomicU32::new(0));
        let completed_series = Arc::new(AtomicU32::new(0));

        // Spawn worker threads
        let num_workers = self.config.max_workers;
        let mut handles = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            let server = self.server.clone();
            let queue = Arc::clone(&work_queue);
            let tx = worker_tx.clone();
            let cancel = Arc::clone(&cancel_flag);
            let active = Arc::clone(&active_workers);
            let completed = Arc::clone(&completed_series);
            let study = study_uid.to_string();
            let dir = cache_dir.to_path_buf();

            // Stagger worker startup to avoid connection storms
            if worker_id > 0 {
                thread::sleep(Duration::from_millis(self.config.stagger_delay_ms));
            }

            let handle = thread::Builder::new()
                .name(format!("retrieve-worker-{}", worker_id))
                .spawn(move || {
                    worker_loop(
                        worker_id, server, queue, &study, &dir, scp_port, tx, cancel, active,
                        completed,
                    )
                })
                .expect("Failed to spawn retrieve worker");

            handles.push(handle);
        }

        // Drop our sender so the channel closes when all workers are done
        drop(worker_tx);

        // Aggregate progress from workers
        let mut series_progress: std::collections::HashMap<String, RetrieveProgress> =
            std::collections::HashMap::new();

        loop {
            // Check for cancellation
            if cancel_flag.load(Ordering::SeqCst) {
                tracing::info!("Parallel retrieve cancelled");
                aggregate.is_complete = true;
                aggregate.error = Some("Cancelled".to_string());
                let _ = progress_tx.send(aggregate.clone());
                break;
            }

            // Receive progress updates (with timeout to check cancel)
            match worker_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(update) => {
                    series_progress.insert(update.series_uid, update.progress);

                    // Aggregate across all series
                    aggregate.completed_images =
                        series_progress.values().map(|p| p.completed).sum();
                    aggregate.failed_images = series_progress.values().map(|p| p.failed).sum();
                    aggregate.completed_series = completed_series.load(Ordering::SeqCst);
                    aggregate.active_workers = active_workers.load(Ordering::SeqCst);

                    // Include completion event if this series just finished
                    aggregate.newly_completed_series = update.completion_event;

                    let _ = progress_tx.send(aggregate.clone());

                    // Clear for next update (completion is one-time event)
                    aggregate.newly_completed_series = None;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Just loop and check cancel
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // All workers done
                    tracing::info!("All workers finished");
                    break;
                }
            }
        }

        // Wait for all workers to complete
        for handle in handles {
            let _ = handle.join();
        }

        // Final progress update
        aggregate.completed_series = completed_series.load(Ordering::SeqCst);
        aggregate.active_workers = 0;
        aggregate.is_complete = true;
        let _ = progress_tx.send(aggregate.clone());

        tracing::info!(
            "Parallel retrieve complete: {}/{} series, {}/{} images",
            aggregate.completed_series,
            aggregate.total_series,
            aggregate.completed_images,
            aggregate.total_images
        );

        Ok(study_dir)
    }

    /// Fallback to single study-level C-MOVE
    fn fallback_study_retrieve(
        &self,
        study_uid: &str,
        cache_dir: &std::path::Path,
        scp_port: u16,
        progress_tx: Sender<AggregateProgress>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<PathBuf> {
        tracing::info!("Using fallback study-level retrieve for {}", study_uid);

        let scu = DicomScu::new(self.server.clone());
        let (series_tx, series_rx) = mpsc::channel::<RetrieveProgress>();

        // Convert RetrieveProgress to AggregateProgress
        let progress_thread = {
            let tx = progress_tx.clone();
            thread::spawn(move || {
                while let Ok(progress) = series_rx.recv() {
                    let mut agg = AggregateProgress::new();
                    agg.total_series = 1;
                    agg.completed_series = if progress.is_complete { 1 } else { 0 };
                    agg.total_images = progress.total();
                    agg.completed_images = progress.completed;
                    agg.failed_images = progress.failed;
                    agg.active_workers = if progress.is_complete { 0 } else { 1 };
                    agg.is_complete = progress.is_complete;
                    agg.error = progress.error.clone();
                    let _ = tx.send(agg);
                }
            })
        };

        let result =
            scu.retrieve_study(study_uid, cache_dir, scp_port, series_tx, Some(cancel_flag));

        // Wait for progress thread
        let _ = progress_thread.join();

        result
    }
}

/// Worker loop: pulls series from queue and retrieves them
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    worker_id: usize,
    server: PacsServerConfig,
    queue: Arc<Mutex<VecDeque<SeriesResult>>>,
    study_uid: &str,
    cache_dir: &std::path::Path,
    scp_port: u16,
    progress_tx: Sender<WorkerProgress>,
    cancel_flag: Arc<AtomicBool>,
    active_workers: Arc<AtomicU32>,
    completed_series: Arc<AtomicU32>,
) {
    let mut consecutive_failures = 0;
    const MAX_CONSECUTIVE_FAILURES: u32 = 3;

    loop {
        // Check cancellation
        if cancel_flag.load(Ordering::SeqCst) {
            tracing::debug!("Worker {} cancelled", worker_id);
            break;
        }

        // Get next series from queue
        let series = {
            let mut queue = match queue.lock() {
                Ok(q) => q,
                Err(_) => break,
            };
            queue.pop_front()
        };

        let series = match series {
            Some(s) => s,
            None => {
                tracing::debug!("Worker {} finished (queue empty)", worker_id);
                break;
            }
        };

        tracing::info!(
            "Worker {} retrieving series {} ({} images)",
            worker_id,
            series.series_instance_uid,
            series.num_instances.unwrap_or(0)
        );

        active_workers.fetch_add(1, Ordering::SeqCst);

        // Create SCU for this retrieval
        let scu = DicomScu::new(server.clone());

        // Create channel for this series' progress
        let (series_progress_tx, series_progress_rx) = mpsc::channel::<RetrieveProgress>();

        // Store series metadata for completion event
        let series_number = series.series_number;
        let series_description = series.series_description.clone();
        let modality = series.modality.clone();

        // Forward progress to main aggregator (without completion event during retrieval)
        let series_uid = series.series_instance_uid.clone();
        let main_tx = progress_tx.clone();
        let forward_thread = {
            let uid = series_uid.clone();
            thread::spawn(move || {
                while let Ok(progress) = series_progress_rx.recv() {
                    let _ = main_tx.send(WorkerProgress {
                        series_uid: uid.clone(),
                        progress,
                        completion_event: None, // Set only on completion below
                    });
                }
            })
        };

        // Retrieve the series
        let result = scu.retrieve_series(
            study_uid,
            &series_uid,
            cache_dir,
            scp_port,
            series_progress_tx,
            Arc::clone(&cancel_flag),
        );

        // Wait for progress forwarder
        let _ = forward_thread.join();

        active_workers.fetch_sub(1, Ordering::SeqCst);

        match result {
            Ok(()) => {
                completed_series.fetch_add(1, Ordering::SeqCst);
                consecutive_failures = 0; // Reset on success

                // Count actual files received for this series
                let series_path = cache_dir.join(study_uid.replace('.', "_"));
                let num_images = count_series_files(&series_path, &series_uid);

                tracing::info!(
                    "Worker {} completed series {} ({} images)",
                    worker_id,
                    series_uid,
                    num_images
                );

                // Send completion event with series metadata
                let completion_event = SeriesCompleteEvent {
                    series_uid: series_uid.clone(),
                    series_number,
                    series_description,
                    modality,
                    num_images,
                    storage_path: series_path,
                };

                // Send final progress with completion event
                let _ = progress_tx.send(WorkerProgress {
                    series_uid: series_uid.clone(),
                    progress: RetrieveProgress {
                        remaining: 0,
                        completed: num_images,
                        failed: 0,
                        warnings: 0,
                        is_complete: true,
                        error: None,
                    },
                    completion_event: Some(completion_event),
                });
            }
            Err(e) => {
                consecutive_failures += 1;
                tracing::warn!(
                    "Worker {} failed series {} ({}/{}): {}",
                    worker_id,
                    series_uid,
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                    e
                );

                // If too many consecutive failures, stop this worker
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    tracing::error!(
                        "Worker {} stopping after {} consecutive failures",
                        worker_id,
                        consecutive_failures
                    );
                    break;
                }

                // Add delay before retry to avoid hammering PACS
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

/// Count DICOM files for a specific series in the study directory
fn count_series_files(study_dir: &std::path::Path, series_uid: &str) -> u32 {
    let mut count = 0u32;

    if let Ok(entries) = std::fs::read_dir(study_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "dcm") {
                // Quick check: read the file and check SeriesInstanceUID
                // For performance, we could cache this or use filename patterns
                if let Ok(file) = std::fs::File::open(&path) {
                    if let Ok(obj) = dicom::object::from_reader(file) {
                        if let Ok(uid) =
                            obj.element(dicom::dictionary_std::tags::SERIES_INSTANCE_UID)
                        {
                            if uid.to_str().is_ok_and(|u| u == series_uid) {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    count
}
