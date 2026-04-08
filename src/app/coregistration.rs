
use crate::coregistration::{apply_transform_to_volume, RegistrationConfig, RegistrationResult};
use crate::dicom::{AnatomicalPlane, MprSeries};
use crate::ui::{CoregistrationVisualState, ViewportId};

use super::SauhuApp;

impl SauhuApp {
    /// Check for coregistration results from background worker
    pub(super) fn check_coregistration_results(&mut self) {
        // Update status during registration
        if self.coregistration.mode().is_running() {
            let progress = self.coregistration.get_progress();
            self.status = format!("Coregistration: {}%", progress);
        }

        // Check for completed registration
        if let Some(result) = self.coregistration.poll_result() {
            match &result {
                RegistrationResult::Success {
                    transform,
                    metric,
                    elapsed,
                    ..
                } => {
                    let rot = transform.rotation_degrees();
                    tracing::info!(
                        "Coregistration successful: metric={:.4}, elapsed={:?}, rotation=({:.1}°, {:.1}°, {:.1}°), translation=({:.1}, {:.1}, {:.1})mm",
                        metric, elapsed,
                        rot.x, rot.y, rot.z,
                        transform.translation_x, transform.translation_y, transform.translation_z
                    );
                    self.status = format!(
                        "Coregistration complete: {:.1}% match in {:.1}s",
                        metric * 100.0,
                        elapsed.as_secs_f64()
                    );

                    // Apply transform to source viewport
                    if let Some((target_vp, source_vp)) =
                        self.coregistration.get_registration_viewports()
                    {
                        // Get target volume for geometry copying (critical for sync)
                        let target_volume = self
                            .viewport_manager
                            .get_slot(target_vp)
                            .and_then(|s| s.mpr_state.volume.clone());

                        if let Some(source_slot) = self.viewport_manager.get_slot_mut(source_vp) {
                            if let Some(source_volume) = source_slot.mpr_state.volume.clone() {
                                // Apply transform with target volume's geometry for correct sync
                                tracing::info!("Applying transform to source volume...");
                                let transformed_volume = apply_transform_to_volume(
                                    &source_volume,
                                    transform,
                                    target_volume.as_deref(),
                                );
                                let transformed_arc = std::sync::Arc::new(transformed_volume);

                                // Update the MPR state with the new volume
                                source_slot.mpr_state.set_volume(transformed_arc.clone());

                                // Regenerate MPR series for current plane
                                let plane = source_slot.mpr_state.plane;
                                let current_slice_idx = source_slot.mpr_state.slice_index;

                                if plane != AnatomicalPlane::Original {
                                    if let Some(mpr_series) =
                                        MprSeries::generate(&transformed_arc, plane)
                                    {
                                        let mpr_arc = std::sync::Arc::new(mpr_series);

                                        // Update MPR series
                                        source_slot.mpr_state.mpr_series = Some(mpr_arc.clone());

                                        // FIX 1: Update sync_info from new MPR series
                                        source_slot.sync_info = Some(mpr_arc.sync_info.clone());

                                        // FIX 2: Update displayed image
                                        let max_idx = mpr_arc.len().saturating_sub(1);
                                        let display_idx = current_slice_idx.min(max_idx);
                                        source_slot.mpr_state.slice_index = display_idx;

                                        if let Some(img) = mpr_arc.images.get(display_idx) {
                                            source_slot.viewport.set_image_keep_view(std::sync::Arc::clone(img));
                                        }

                                        tracing::info!(
                                            "Coregistration applied: {} slices, displaying {}, FOR={:?}, STUDY={:?}",
                                            mpr_arc.len(), display_idx,
                                            mpr_arc.sync_info.frame_of_reference_uid,
                                            mpr_arc.sync_info.study_instance_uid
                                        );
                                    }
                                }
                            }
                        }

                    }

                    self.coregistration.reset();
                }
                RegistrationResult::Failed { error, .. } => {
                    tracing::warn!("Coregistration failed: {}", error);
                    self.status = format!("Coregistration failed: {}", error);
                    self.coregistration.set_error(error.clone());
                    self.coregistration.reset();
                }
                RegistrationResult::Cancelled => {
                    tracing::info!("Coregistration cancelled by user");
                    self.status = "Coregistration cancelled".to_string();
                    self.coregistration.reset();
                }
            }
            // Clear any visual state
            self.viewport_manager.clear_coregistration_state();
        }
    }

    /// Handle viewport click during coregistration selection
    pub(super) fn handle_coregistration_click(&mut self, viewport_id: ViewportId) -> bool {
        if !self.coregistration.mode().is_selecting() {
            return false;
        }

        // Handle the click
        if self.coregistration.handle_viewport_click(viewport_id) {
            // Update status
            self.status = self.coregistration.mode().status_text().to_string();

            // Update visual state based on new mode
            self.update_coregistration_visual_state();

            // Check if we're now ready to start registration
            if let Some((target_vp, source_vp)) = self.coregistration.get_registration_viewports() {
                // Get volumes from both viewports
                let target_vol = self
                    .viewport_manager
                    .get_slot(target_vp)
                    .and_then(|s| s.mpr_state.volume.clone());
                let source_vol = self
                    .viewport_manager
                    .get_slot(source_vp)
                    .and_then(|s| s.mpr_state.volume.clone());

                match (target_vol, source_vol) {
                    (Some(target), Some(source)) => {
                        // Detect config based on modality
                        let target_mod = self
                            .viewport_manager
                            .get_slot(target_vp)
                            .and_then(|s| s.viewport.image())
                            .and_then(|img| img.modality.clone());
                        let source_mod = self
                            .viewport_manager
                            .get_slot(source_vp)
                            .and_then(|s| s.viewport.image())
                            .and_then(|img| img.modality.clone());

                        let config = RegistrationConfig::detect(
                            target_mod.as_deref(),
                            source_mod.as_deref(),
                        );

                        tracing::info!(
                            "Starting coregistration: target={}, source={}, metric={:?}",
                            target_vp,
                            source_vp,
                            config.metric
                        );

                        self.coregistration
                            .start_registration(target, source, config);
                        self.status = "Coregistration: Running...".to_string();

                        // Clear visual state while running
                        self.viewport_manager.clear_coregistration_state();
                    }
                    _ => {
                        // Need MPR volumes - volumes not available
                        self.status =
                            "Coregistration requires 3D volumes (use MPR mode first)".to_string();
                        self.coregistration.reset();
                        self.viewport_manager.clear_coregistration_state();
                    }
                }
            }

            return true;
        }

        false
    }

    /// Update viewport visual state based on coregistration mode
    pub(super) fn update_coregistration_visual_state(&mut self) {
        use crate::coregistration::CoregistrationMode;

        match self.coregistration.mode() {
            CoregistrationMode::Inactive => {
                self.viewport_manager.clear_coregistration_state();
            }
            CoregistrationMode::SelectingTarget => {
                self.viewport_manager
                    .set_coregistration_state(CoregistrationVisualState {
                        target: None,
                        selecting: true,
                    });
            }
            CoregistrationMode::SelectingSource { target_viewport } => {
                self.viewport_manager
                    .set_coregistration_state(CoregistrationVisualState {
                        target: Some(target_viewport),
                        selecting: true,
                    });
            }
            CoregistrationMode::Running { .. } | CoregistrationMode::Completed { .. } => {
                self.viewport_manager.clear_coregistration_state();
            }
        }
    }
}
