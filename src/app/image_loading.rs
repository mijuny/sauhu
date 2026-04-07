#![allow(dead_code)]

use crate::dicom::{DicomImage, SyncInfo};
use crate::gpu::DicomRenderResources;
use crate::ui::ViewportId;
use std::path::PathBuf;

use super::SauhuApp;

/// Result of async image loading
#[allow(clippy::large_enum_variant)]
pub(super) enum LoadResult {
    Success {
        image: DicomImage,
        path: PathBuf,
        patient: String,
        series_desc: String,
        index: usize,
        total: usize,
        viewport_id: ViewportId,
        generation: u64,
    },
    Failed {
        index: usize,
        error: String,
        viewport_id: ViewportId,
        generation: u64,
    },
}

/// Context for what to do after a background scan completes
#[derive(Debug, Clone)]
pub(super) enum ScanContext {
    /// Loading a folder opened by user
    LoadFolder,
    /// Opening a patient from database
    OpenPatient,
    /// Quick fetch completed
    QuickFetchComplete { accession: String },
    /// Load to a specific viewport (hanging protocol assignment)
    LoadToViewport { viewport_id: ViewportId },
}

impl SauhuApp {
    /// Load files from paths into the active viewport (async - validates in background)
    pub(super) fn load_files(&mut self, paths: Vec<PathBuf>) {
        self.load_files_to_viewport(self.viewport_manager.active_viewport(), paths);
    }

    /// Load files from paths into a specific viewport (async - validates in background)
    pub(super) fn load_files_to_viewport(&mut self, viewport_id: ViewportId, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            return;
        }

        self.status = format!("Validating {} file(s)...", paths.len());
        self.bg_loading_state = super::background::BackgroundLoadingState::ScanningDirectory {
            path: "files".to_string(),
        };
        let _ = self
            .bg_request_tx
            .send(super::background::BackgroundRequest::ValidateFiles { paths, viewport_id });
    }

    /// Handle validated files result (called from background result handler)
    pub(super) fn handle_files_validated(&mut self, paths: Vec<PathBuf>, viewport_id: ViewportId) {
        if !paths.is_empty() {
            // Prefetch entire series to cache for smooth scrolling
            self.image_cache.prefetch_series(&paths);

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

            self.status = format!("Loaded {} images - scroll to navigate", paths.len());
        } else {
            self.status = "No valid DICOM files found".to_string();
        }
    }

    /// Load folder (async - uses background worker)
    pub(super) fn load_folder(&mut self, path: &PathBuf) {
        self.status = format!("Scanning {:?}...", path);
        self.bg_loading_state = super::background::BackgroundLoadingState::ScanningDirectory {
            path: path.to_string_lossy().to_string(),
        };
        let _ = self.bg_request_tx.send(super::background::BackgroundRequest::ScanDirectory {
            path: path.clone(),
            context: ScanContext::LoadFolder,
        });
    }

    /// Load folder to a specific viewport (async - for hanging protocol assignment)
    pub(super) fn load_folder_to_viewport(&mut self, path: &PathBuf, viewport_id: ViewportId) {
        self.status = format!("Scanning {:?}...", path);
        let _ = self.bg_request_tx.send(super::background::BackgroundRequest::ScanDirectory {
            path: path.clone(),
            context: ScanContext::LoadToViewport { viewport_id },
        });
    }

    /// Load a series from pre-sorted file list into the active viewport
    pub(super) fn load_series(&mut self, files: Vec<PathBuf>, name: &str, sync_info: Option<SyncInfo>) {
        self.load_series_to_viewport(
            self.viewport_manager.active_viewport(),
            files,
            name,
            sync_info,
        );
    }

    /// Load a series into a specific viewport
    pub(super) fn load_series_to_viewport(
        &mut self,
        viewport_id: ViewportId,
        files: Vec<PathBuf>,
        name: &str,
        sync_info: Option<SyncInfo>,
    ) {
        if files.is_empty() {
            self.status = "No images in series".to_string();
            return;
        }

        tracing::info!(
            "Loading series '{}' with {} images to viewport {}",
            name,
            files.len(),
            viewport_id
        );

        // Prefetch entire series to cache for smooth scrolling
        self.image_cache.prefetch_series(&files);

        // Load series into viewport manager
        let start_at_middle = self.settings.viewer.start_at_middle_slice;
        self.viewport_manager.load_series(
            viewport_id,
            files.clone(),
            name,
            sync_info,
            start_at_middle,
        );
        // Get the actual start index from the slot
        let start_index = self
            .viewport_manager
            .get_slot(viewport_id)
            .map(|s| s.current_index)
            .unwrap_or(0);
        self.load_image_for_viewport(viewport_id, start_index);

        self.status = format!("{} - {} images - scroll to navigate", name, files.len());
    }

    /// Load image at index for a specific viewport
    pub(super) fn load_image_for_viewport(&mut self, viewport_id: ViewportId, index: usize) {
        // Check if this viewport is in MPR mode - if so, update MPR slice instead
        // Extract all needed values before mutable borrow to satisfy borrow checker
        let mpr_data = self
            .viewport_manager
            .get_slot(viewport_id)
            .and_then(|slot| {
                if slot.mpr_state.is_active() && slot.mpr_state.has_series() {
                    slot.mpr_state.mpr_series.as_ref().and_then(|series| {
                        series.images.get(index).map(|img| {
                            // Also get the sync_info slice location for comparison
                            let sync_loc = series
                                .sync_info
                                .slice_locations
                                .get(index)
                                .copied()
                                .flatten();
                            (
                                img.clone(),
                                slot.viewport.window_level(),
                                sync_loc,
                                img.slice_location,
                            )
                        })
                    })
                } else {
                    None
                }
            });

        // Apply MPR image if we have one
        if let Some((image, (wc, ww), sync_loc, img_loc)) = mpr_data {
            if let Some(slot) = self.viewport_manager.get_slot_mut(viewport_id) {
                let old_idx = slot.mpr_state.slice_index;
                tracing::info!("MPR sync APPLY: viewport {} slice {} -> {}, sync_loc={:?}, img_loc={:?}, mpr_series len={}",
                    viewport_id, old_idx, index, sync_loc, img_loc,
                    slot.mpr_state.mpr_series.as_ref().map(|s| s.len()).unwrap_or(0));
                slot.mpr_state.slice_index = index;
                slot.viewport.set_image_keep_view(image.clone());
                slot.viewport.set_window(wc, ww);
                // Verify the displayed image matches
                let displayed_loc = slot.viewport.image().and_then(|i| i.slice_location);
                tracing::info!("MPR sync VERIFY: viewport {} now displaying slice_index={}, image.slice_location={:?}",
                    viewport_id, slot.mpr_state.slice_index, displayed_loc);
            }
            return;
        } else {
            // Debug why MPR data wasn't found
            if let Some(slot) = self.viewport_manager.get_slot(viewport_id) {
                if slot.mpr_state.is_active() {
                    tracing::warn!(
                        "MPR sync failed: viewport {} is_active={} has_series={} index={}",
                        viewport_id,
                        slot.mpr_state.is_active(),
                        slot.mpr_state.has_series(),
                        index
                    );
                }
            }
        }

        // Normal mode: load from series files
        if let Some(slot) = self.viewport_manager.get_slot(viewport_id) {
            if slot.series_files.is_empty() {
                return;
            }

            // Debug: log the slice_location for this index
            let expected_loc = slot
                .sync_info
                .as_ref()
                .and_then(|s| s.slice_locations.get(index).copied().flatten());
            tracing::info!(
                "Normal load: viewport {} index {} expected_slice_loc={:?}",
                viewport_id,
                index,
                expected_loc
            );

            if let Some(path) = slot.series_files.get(index).cloned() {
                let total = slot.series_files.len();

                // Check cache first
                if self.image_cache.contains(&path) {
                    // Queue for processing in update() where we have frame access for GPU texture
                    tracing::debug!("Cache hit for {:?}", path.file_name().unwrap_or_default());
                    self.pending_cache_hits.push((
                        path,
                        index,
                        total,
                        viewport_id,
                        self.patient_generation,
                    ));
                } else {
                    // Cache miss - load from disk
                    tracing::debug!("Cache miss for {:?}", path.file_name().unwrap_or_default());
                    self.loading = true;
                    self.status = format!("Loading [{}/{}]...", index + 1, total);
                    let _ = self.load_sender.send((
                        path,
                        index,
                        total,
                        viewport_id,
                        self.patient_generation,
                    ));
                }
            }
        }
    }

    /// Load current image for active viewport
    pub(super) fn load_current_image(&mut self) {
        let active_id = self.viewport_manager.active_viewport();
        if let Some(slot) = self.viewport_manager.get_slot(active_id) {
            self.load_image_for_viewport(active_id, slot.current_index);
        }
    }

    /// Check for loaded images from background thread
    pub(super) fn check_loaded_images(&mut self, frame: &eframe::Frame) {
        while let Ok(result) = self.load_receiver.try_recv() {
            self.loading = false;
            match result {
                LoadResult::Success {
                    image,
                    path,
                    patient,
                    series_desc,
                    index,
                    total,
                    viewport_id,
                    generation,
                } => {
                    // Discard stale results from previous patient
                    if generation != self.patient_generation {
                        tracing::debug!(
                            "Discarding stale load result (gen {} vs current {})",
                            generation,
                            self.patient_generation
                        );
                        continue;
                    }

                    tracing::info!(
                        "Loaded image for viewport {}: {} - {} ({}/{}) actual_slice_loc={:?}",
                        viewport_id,
                        patient,
                        series_desc,
                        index + 1,
                        total,
                        image.slice_location
                    );

                    // Add to image cache before applying
                    self.image_cache.insert(path.clone(), image.clone());

                    // Upload texture to GPU cache if available
                    if self.gpu_available {
                        if let Some(render_state) = frame.wgpu_render_state() {
                            let mut renderer = render_state.renderer.write();
                            if let Some(resources) = renderer
                                .callback_resources
                                .get_mut::<DicomRenderResources>()
                            {
                                // Try cached texture first (fast path)
                                if !resources.bind_cached_texture(
                                    &render_state.device,
                                    viewport_id,
                                    &path,
                                ) {
                                    // Not in GPU cache, upload and cache
                                    resources.upload_and_cache_texture(
                                        &render_state.device,
                                        &render_state.queue,
                                        viewport_id,
                                        path,
                                        &image,
                                    );
                                }
                            }
                        }
                    }

                    // Check if this is the first image of a new series (needs smart fit)
                    // or scrolling through existing series (preserve view)
                    let is_new_series = self
                        .viewport_manager
                        .get_slot(viewport_id)
                        .map(|s| s.new_series || !s.viewport.has_image())
                        .unwrap_or(true);

                    if is_new_series {
                        // New series - fit to window and use image defaults
                        self.viewport_manager.set_image(viewport_id, image);
                        if let Some(slot) = self.viewport_manager.get_slot_mut(viewport_id) {
                            slot.new_series = false;
                        }
                    } else {
                        // Scrolling through series - preserve zoom, pan, and windowing
                        self.viewport_manager
                            .set_image_keep_view(viewport_id, image);
                    }

                    self.status =
                        format!("{} - {} [{}/{}]", patient, series_desc, index + 1, total);
                }
                LoadResult::Failed {
                    index,
                    error,
                    viewport_id,
                    generation,
                } => {
                    // Discard stale results from previous patient
                    if generation != self.patient_generation {
                        continue;
                    }

                    let total = self
                        .viewport_manager
                        .get_slot(viewport_id)
                        .map(|s| s.series_files.len())
                        .unwrap_or(0);
                    tracing::warn!(
                        "Failed to load image {} for viewport {}: {:?}",
                        index + 1,
                        viewport_id,
                        error
                    );
                    self.status = format!(
                        "Failed [{}/{}]: {} - press arrow to skip",
                        index + 1,
                        total,
                        error
                    );
                }
            }
        }
    }

    /// Process pending cache hits (apply cached images to viewports)
    pub(super) fn process_cache_hits(&mut self, frame: &eframe::Frame) {
        let hits: Vec<_> = self.pending_cache_hits.drain(..).collect();

        for (path, index, total, viewport_id, generation) in hits {
            // Discard stale cache hits from previous patient
            if generation != self.patient_generation {
                tracing::debug!(
                    "Discarding stale cache hit (gen {} vs current {})",
                    generation,
                    self.patient_generation
                );
                continue;
            }

            // Try GPU cache first (fastest path - no image clone needed)
            let mut gpu_cache_hit = false;
            if self.gpu_available {
                if let Some(render_state) = frame.wgpu_render_state() {
                    let mut renderer = render_state.renderer.write();
                    if let Some(resources) = renderer
                        .callback_resources
                        .get_mut::<DicomRenderResources>()
                    {
                        gpu_cache_hit =
                            resources.bind_cached_texture(&render_state.device, viewport_id, &path);
                    }
                }
            }

            // Get image from CPU cache (for viewport state, even if GPU cached)
            if let Some(image) = self.image_cache.get_clone(&path) {
                tracing::debug!(
                    "Applying cached image for viewport {}: ({}/{}) gpu_cached={}",
                    viewport_id,
                    index + 1,
                    total,
                    gpu_cache_hit
                );

                // Upload to GPU cache if not already cached
                if !gpu_cache_hit && self.gpu_available {
                    if let Some(render_state) = frame.wgpu_render_state() {
                        let mut renderer = render_state.renderer.write();
                        if let Some(resources) = renderer
                            .callback_resources
                            .get_mut::<DicomRenderResources>()
                        {
                            resources.upload_and_cache_texture(
                                &render_state.device,
                                &render_state.queue,
                                viewport_id,
                                path,
                                &image,
                            );
                        }
                    }
                }

                // Check if this is the first image of a new series (needs smart fit)
                let is_new_series = self
                    .viewport_manager
                    .get_slot(viewport_id)
                    .map(|s| s.new_series || !s.viewport.has_image())
                    .unwrap_or(true);

                if is_new_series {
                    self.viewport_manager.set_image(viewport_id, image);
                    if let Some(slot) = self.viewport_manager.get_slot_mut(viewport_id) {
                        slot.new_series = false;
                    }
                } else {
                    self.viewport_manager
                        .set_image_keep_view(viewport_id, image);
                }

                // Update status with index (we don't have patient/series info from cache)
                self.status = format!("[{}/{}]", index + 1, total);
            }
        }
    }
}
