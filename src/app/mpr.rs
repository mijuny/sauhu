
use crate::dicom::{AnatomicalPlane, DicomImage, MprSeries, Volume};
use crate::ui::ViewportId;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use super::SauhuApp;

/// MPR (Multi-Planar Reconstruction) system state
pub(super) struct MprSystem {
    pub(super) pending: Option<(ViewportId, crate::dicom::AnatomicalPlane, usize)>,
    pub(super) request_tx: Sender<MprRequest>,
    pub(super) result_rx: Receiver<MprResult>,
    pub(super) generating: Option<MprGeneratingState>,
}

/// Request for MPR background worker
pub(super) enum MprRequest {
    /// Build 3D volume from DICOM images
    BuildVolume {
        viewport_id: ViewportId,
        images: Vec<std::sync::Arc<DicomImage>>,
        generation: u64,
    },
    /// Generate reformatted slices for a plane
    GenerateSeries {
        viewport_id: ViewportId,
        volume: Arc<Volume>,
        plane: AnatomicalPlane,
        generation: u64,
    },
    /// Cancel current operation
    Cancel,
}

/// Result from MPR background worker
pub(super) enum MprResult {
    /// Volume built successfully
    VolumeBuilt {
        #[allow(dead_code)]
        viewport_id: ViewportId,
        volume: Arc<Volume>,
        generation: u64,
    },
    /// Volume building failed
    VolumeFailed {
        #[allow(dead_code)]
        viewport_id: ViewportId,
        error: String,
        generation: u64,
    },
    /// Progress update during series generation
    #[allow(dead_code)]
    SeriesProgress {
        viewport_id: ViewportId,
        completed: usize,
        total: usize,
        generation: u64,
    },
    /// Series generated successfully
    SeriesGenerated {
        viewport_id: ViewportId,
        plane: AnatomicalPlane,
        series: Arc<MprSeries>,
        generation: u64,
    },
    /// Series generation failed
    SeriesFailed {
        #[allow(dead_code)]
        viewport_id: ViewportId,
        error: String,
        generation: u64,
    },
}

/// State for MPR generation progress (for UI feedback)
pub(super) struct MprGeneratingState {
    pub viewport_id: ViewportId,
    pub plane: AnatomicalPlane,
    pub phase: MprPhase,
    /// Saved window level to restore after MPR completes (WC, WW)
    pub saved_window: (f64, f64),
}

/// Phase of MPR generation
pub(super) enum MprPhase {
    BuildingVolume,
    #[allow(dead_code)]
    GeneratingSlices {
        completed: usize,
        total: usize,
    },
}

impl SauhuApp {
    /// Create MPR worker for background volume building and slice generation
    pub(super) fn create_mpr_worker() -> (Sender<MprRequest>, Receiver<MprResult>) {
        let (request_tx, request_rx) = mpsc::channel::<MprRequest>();
        let (result_tx, result_rx) = mpsc::channel::<MprResult>();

        thread::spawn(move || {
            while let Ok(request) = request_rx.recv() {
                match request {
                    MprRequest::BuildVolume {
                        viewport_id,
                        images,
                        generation,
                    } => {
                        tracing::info!("MPR worker: building volume from {} images", images.len());
                        match Volume::from_series(&images) {
                            Some(vol) => {
                                tracing::info!("MPR worker: volume built successfully");
                                let _ = result_tx.send(MprResult::VolumeBuilt {
                                    viewport_id,
                                    volume: Arc::new(vol),
                                    generation,
                                });
                            }
                            None => {
                                if result_tx.send(MprResult::VolumeFailed {
                                    viewport_id,
                                    error: "Could not build volume (images may not be parallel)"
                                        .to_string(),
                                    generation,
                                }).is_err() {
                                    tracing::warn!("MPR result channel closed while reporting volume failure");
                                }
                            }
                        }
                    }
                    MprRequest::GenerateSeries {
                        viewport_id,
                        volume,
                        plane,
                        generation,
                    } => {
                        let total = volume.slice_count(plane);
                        tracing::info!("MPR worker: generating {} slices for {:?}", total, plane);

                        // Check for cancellation before starting
                        if let Ok(MprRequest::Cancel) = request_rx.try_recv() {
                            tracing::info!("MPR worker: cancelled before generation started");
                            continue;
                        }

                        // Generate series using existing method
                        match MprSeries::generate(&volume, plane) {
                            Some(series) => {
                                tracing::info!("MPR worker: generated {} slices", series.len());
                                let _ = result_tx.send(MprResult::SeriesGenerated {
                                    viewport_id,
                                    plane,
                                    series: Arc::new(series),
                                    generation,
                                });
                            }
                            None => {
                                if result_tx.send(MprResult::SeriesFailed {
                                    viewport_id,
                                    error: "Failed to generate MPR series".to_string(),
                                    generation,
                                }).is_err() {
                                    tracing::warn!("MPR result channel closed while reporting series failure");
                                }
                            }
                        }
                    }
                    MprRequest::Cancel => {
                        // Just drain any pending requests
                        tracing::info!("MPR worker: cancel received");
                    }
                }
            }
        });

        (request_tx, result_rx)
    }

    /// Check if pending MPR request can now be executed (all images cached)
    pub(super) fn check_pending_mpr(&mut self) {
        let Some((viewport_id, plane, total_needed)) = self.mpr.pending else {
            return;
        };

        // Check if we have all images cached for this viewport
        let cached_count = if let Some(slot) = self.viewport_manager.get_slot(viewport_id) {
            slot.series_files
                .iter()
                .filter(|path| self.background.image_cache.contains(path))
                .count()
        } else {
            // Viewport no longer exists, clear pending
            self.mpr.pending = None;
            return;
        };

        // Update status with progress
        if cached_count < total_needed {
            // Still waiting - update status periodically
            self.status = format!(
                "MPR: Waiting for images ({}/{}) - will generate automatically...",
                cached_count, total_needed
            );
            return;
        }

        // All images cached! Clear pending and trigger MPR generation
        tracing::info!(
            "Pending MPR: all {} images now cached, generating {:?}",
            total_needed,
            plane
        );
        self.mpr.pending = None;

        // Make sure this viewport is still active before setting MPR
        if self.viewport_manager.active_viewport() == viewport_id {
            self.set_mpr_plane(plane);
        } else {
            // Viewport is no longer active, user switched - don't auto-generate
            tracing::debug!(
                "Pending MPR cancelled: viewport {} no longer active",
                viewport_id
            );
        }
    }

    /// Check for MPR generation results from background worker
    pub(super) fn check_mpr_results(&mut self) {
        while let Ok(result) = self.mpr.result_rx.try_recv() {
            match result {
                MprResult::VolumeBuilt {
                    viewport_id,
                    volume,
                    generation,
                } => {
                    // Ignore stale results from previous patient
                    if generation != self.image_loading.patient_generation {
                        tracing::debug!(
                            "Ignoring stale VolumeBuilt (gen {} != {})",
                            generation,
                            self.image_loading.patient_generation
                        );
                        continue;
                    }

                    tracing::info!("MPR volume built successfully");

                    // Store volume in viewport's mpr_state
                    if let Some(slot) = self.viewport_manager.get_slot_mut(viewport_id) {
                        slot.mpr_state.set_volume(volume.clone());
                    }

                    // Continue to series generation if we're still generating
                    if let Some(ref state) = self.mpr.generating {
                        if state.viewport_id == viewport_id {
                            let plane = state.plane;
                            let saved_window = state.saved_window;
                            let total = volume.slice_count(plane);

                            // Update state to generating slices (preserve saved window)
                            self.mpr.generating = Some(MprGeneratingState {
                                viewport_id,
                                plane,
                                phase: MprPhase::GeneratingSlices {
                                    completed: 0,
                                    total,
                                },
                                saved_window,
                            });
                            self.status = format!("Generating {} slices...", plane);

                            let _ = self.mpr.request_tx.send(MprRequest::GenerateSeries {
                                viewport_id,
                                volume,
                                plane,
                                generation: self.image_loading.patient_generation,
                            });
                        }
                    }
                }
                MprResult::VolumeFailed {
                    viewport_id: _,
                    error,
                    generation,
                } => {
                    if generation != self.image_loading.patient_generation {
                        continue;
                    }
                    tracing::warn!("MPR volume build failed: {}", error);
                    self.mpr.generating = None;
                    self.status = error;
                }
                MprResult::SeriesProgress {
                    viewport_id,
                    completed,
                    total,
                    generation,
                } => {
                    if generation != self.image_loading.patient_generation {
                        continue;
                    }
                    if let Some(ref mut state) = self.mpr.generating {
                        if state.viewport_id == viewport_id {
                            state.phase = MprPhase::GeneratingSlices { completed, total };
                            self.status = format!("Generating slices ({}/{})...", completed, total);
                        }
                    }
                }
                MprResult::SeriesGenerated {
                    viewport_id,
                    plane,
                    series,
                    generation,
                } => {
                    if generation != self.image_loading.patient_generation {
                        tracing::debug!(
                            "Ignoring stale SeriesGenerated (gen {} != {})",
                            generation,
                            self.image_loading.patient_generation
                        );
                        continue;
                    }

                    tracing::info!(
                        "MPR series generated: {} slices for {:?}",
                        series.len(),
                        plane
                    );

                    // Extract saved window before clearing state
                    let saved_window = self.mpr.generating.as_ref().map(|s| s.saved_window);

                    // Clear generating state
                    self.mpr.generating = None;

                    if let Some(slot) = self.viewport_manager.get_slot_mut(viewport_id) {
                        // Set the plane and series
                        slot.mpr_state.set_plane(plane);
                        slot.mpr_state.set_series(series.clone());

                        // Set sync_info from MPR series (enables sync with other viewports)
                        slot.sync_info = Some(series.sync_info.clone());

                        // Use CPU rendering for MPR
                        slot.viewport.set_use_gpu(false);

                        // Display middle slice (smart zoom will be applied via set_image)
                        let middle_idx = series.len() / 2;
                        slot.mpr_state.slice_index = middle_idx;
                        if let Some(img) = series.images.get(middle_idx) {
                            slot.viewport.set_image(Arc::clone(img));
                        }

                        // Restore saved window level (after set_image which resets to defaults)
                        if let Some((wc, ww)) = saved_window {
                            slot.viewport.set_window_level(wc, ww);
                        }

                        self.status = format!("MPR: {} ({} slices)", plane, series.len());
                    }
                }
                MprResult::SeriesFailed {
                    viewport_id: _,
                    error,
                    generation,
                } => {
                    if generation != self.image_loading.patient_generation {
                        continue;
                    }
                    tracing::warn!("MPR series generation failed: {}", error);
                    self.mpr.generating = None;
                    self.status = error;
                }
            }
        }
    }

    /// Set MPR viewing plane for active viewport
    /// Generates full MPR series and uploads to GPU for smooth scrolling
    pub(super) fn set_mpr_plane(&mut self, plane: crate::dicom::AnatomicalPlane) {
        use crate::dicom::AnatomicalPlane;

        // Extract info before mutable borrow
        let viewport_id = self.viewport_manager.active_viewport();
        let slot = self.viewport_manager.get_active();
        let series_len = slot.series_files.len();
        let has_volume = slot.mpr_state.volume.is_some();
        let existing_volume = slot.mpr_state.volume.clone();
        let current_window = slot.viewport.window_level(); // Save current WC/WW to restore after MPR

        // Need at least 2 images for MPR
        if series_len < 2 {
            self.status = "MPR requires at least 2 images".to_string();
            return;
        }

        // If switching to Original, just clear MPR mode
        if plane == AnatomicalPlane::Original {
            let slot = self.viewport_manager.get_active_mut();
            if slot.mpr_state.is_active() {
                slot.mpr_state.clear();
                // Re-enable GPU rendering for original images
                slot.viewport.set_use_gpu(self.gpu_available);
                // Reload current original image
                let current_idx = slot.current_index;
                if let Some(path) = slot.series_files.get(current_idx) {
                    if let Some(image) = self.background.image_cache.get(path) {
                        slot.viewport.set_image(image);
                    }
                }
                self.status = "MPR: Original view".to_string();
            }
            // Cancel any ongoing MPR generation
            let _ = self.mpr.request_tx.send(MprRequest::Cancel);
            self.mpr.generating = None;
            return;
        }

        // If we're already generating for this plane, ignore duplicate requests
        if let Some(ref state) = self.mpr.generating {
            if state.viewport_id == viewport_id && state.plane == plane {
                return;
            }
            // Different plane requested - cancel current and start new
            let _ = self.mpr.request_tx.send(MprRequest::Cancel);
        }

        // Build volume if not already built
        if !has_volume {
            // Collect paths first (immutable borrow)
            let slot = self.viewport_manager.get_active();
            let series_files: Vec<_> = slot.series_files.clone();

            // Load all images from cache
            let mut images: Vec<std::sync::Arc<crate::dicom::DicomImage>> = Vec::new();
            for path in &series_files {
                if let Some(img) = self.background.image_cache.get(path) {
                    images.push(img);
                }
            }

            // Require ALL images to be cached before building volume
            if images.len() < series_len {
                self.mpr.pending = Some((viewport_id, plane, series_len));
                self.status = format!(
                    "MPR: Waiting for images ({}/{}) - will generate automatically...",
                    images.len(),
                    series_len
                );
                return;
            }

            if images.len() < 2 {
                self.status = "MPR requires at least 2 images".to_string();
                return;
            }

            // Clear pending MPR since we're now processing it
            self.mpr.pending = None;

            // Set generating state and send async request
            self.mpr.generating = Some(MprGeneratingState {
                viewport_id,
                plane,
                phase: MprPhase::BuildingVolume,
                saved_window: current_window,
            });
            self.status = "Building 3D volume...".to_string();

            let _ = self.mpr.request_tx.send(MprRequest::BuildVolume {
                viewport_id,
                images,
                generation: self.image_loading.patient_generation,
            });
            return;
        }

        // Volume exists - send series generation request
        let Some(volume) = existing_volume else {
            tracing::error!("MPR: expected volume to exist but it was None");
            return;
        };

        // Clear pending MPR since we're now processing it
        self.mpr.pending = None;

        // Set generating state
        let total = volume.slice_count(plane);
        self.mpr.generating = Some(MprGeneratingState {
            viewport_id,
            plane,
            phase: MprPhase::GeneratingSlices {
                completed: 0,
                total,
            },
            saved_window: current_window,
        });
        self.status = format!("Generating {} slices...", plane);

        let _ = self.mpr.request_tx.send(MprRequest::GenerateSeries {
            viewport_id,
            volume,
            plane,
            generation: self.image_loading.patient_generation,
        });
    }

}
