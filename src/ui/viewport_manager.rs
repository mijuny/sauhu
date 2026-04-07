//! Multi-viewport manager
//!
//! Manages multiple viewports with different layouts and synchronized scrolling.
#![allow(dead_code)]

use crate::dicom::{
    AnatomicalPlane, DicomImage, MprState, PatientAxis, Point2D, ReferenceLine, SyncInfo,
};
use crate::fusion::FusionState;
use crate::ui::{DroppedSeries, InteractionMode, Viewport};
use egui::{Color32, Ui};
use std::path::PathBuf;

/// Maximum number of viewport slots
pub const MAX_VIEWPORTS: usize = 8;

/// Viewport identifier (0-7)
pub type ViewportId = usize;

/// Viewport layout configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportLayout {
    /// Single viewport (V+1)
    Single,
    /// 2 viewports side by side (V+2)
    Horizontal2,
    /// 3 viewports side by side (V+3)
    Horizontal3,
    /// 2x2 grid (V+4)
    Grid2x2,
    /// 2x2 left + 1 tall right (V+5)
    Grid2x2Plus1,
    /// 3x2 grid (V+6)
    Grid3x2,
    /// 4x2 grid (V+8)
    Grid4x2,
}

impl ViewportLayout {
    /// Parse layout from config string
    pub fn from_config_str(s: &str) -> Option<Self> {
        match s {
            "single" => Some(Self::Single),
            "horizontal_2" => Some(Self::Horizontal2),
            "horizontal_3" => Some(Self::Horizontal3),
            "grid_2x2" => Some(Self::Grid2x2),
            "grid_2x2_plus_1" => Some(Self::Grid2x2Plus1),
            "grid_3x2" => Some(Self::Grid3x2),
            "grid_4x2" => Some(Self::Grid4x2),
            _ => None,
        }
    }

    /// Number of viewports for this layout
    pub fn viewport_count(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::Horizontal2 => 2,
            Self::Horizontal3 => 3,
            Self::Grid2x2 => 4,
            Self::Grid2x2Plus1 => 5,
            Self::Grid3x2 => 6,
            Self::Grid4x2 => 8,
        }
    }
}

/// State for a single viewport slot
pub struct ViewportSlot {
    pub id: ViewportId,
    pub viewport: Viewport,
    pub series_files: Vec<PathBuf>,
    pub current_index: usize,
    pub series_name: String,
    pub sync_info: Option<SyncInfo>,
    /// GPU texture slot index
    pub texture_slot: usize,
    /// MPR (Multi-Planar Reconstruction) state
    pub mpr_state: MprState,
    /// True when a new series was just loaded — first image should trigger smart fit
    pub new_series: bool,
    /// Fusion state (for blending with another viewport)
    pub fusion_state: FusionState,
}

impl ViewportSlot {
    pub fn new(id: ViewportId, use_gpu: bool) -> Self {
        Self {
            id,
            viewport: Viewport::with_id(id, use_gpu),
            series_files: Vec::new(),
            current_index: 0,
            series_name: String::new(),
            sync_info: None,
            texture_slot: id,
            mpr_state: MprState::new(),
            new_series: false,
            fusion_state: FusionState::new(),
        }
    }

    pub fn has_series(&self) -> bool {
        !self.series_files.is_empty()
    }

    pub fn current_file(&self) -> Option<&PathBuf> {
        self.series_files.get(self.current_index)
    }

    pub fn image_count(&self) -> usize {
        self.series_files.len()
    }

    /// Check if MPR mode is active
    pub fn is_mpr_active(&self) -> bool {
        self.mpr_state.is_active()
    }

    /// Get current slice count (MPR or original)
    pub fn slice_count(&self) -> usize {
        if self.mpr_state.is_active() {
            self.mpr_state.slice_count()
        } else {
            self.series_files.len()
        }
    }

    /// Get current slice index (MPR or original)
    pub fn current_slice_index(&self) -> usize {
        if self.mpr_state.is_active() {
            self.mpr_state.slice_index
        } else {
            self.current_index
        }
    }

    /// Get a stable key for annotation storage on the current image.
    /// For regular DICOM: SOP Instance UID.
    /// For MPR reformats: synthetic path like "mpr://coronal/42".
    pub fn annotation_key(&self) -> Option<String> {
        // Try SOP Instance UID first (works for regular DICOM)
        if let Some(uid) = self.viewport.sop_instance_uid() {
            return Some(uid.to_string());
        }
        // For MPR: use the synthetic path from mpr_series
        if self.mpr_state.is_active() {
            if let Some(path) = self.mpr_state.get_series_path() {
                return Some(path.to_string_lossy().into_owned());
            }
        }
        // Final fallback: current file path
        self.current_file().map(|p| p.to_string_lossy().into_owned())
    }

    /// Clear MPR state (called when loading new series)
    pub fn clear_mpr(&mut self) {
        self.mpr_state.clear();
    }
}

/// Response from showing a viewport
pub struct ViewportResponse {
    pub clicked: bool,
    /// Whether the mouse cursor is over this viewport
    pub hovered: bool,
    pub dropped_series: Option<DroppedSeries>,
}

impl ViewportResponse {
    fn new() -> Self {
        Self {
            clicked: false,
            hovered: false,
            dropped_series: None,
        }
    }
}

/// Result from ViewportManager::show()
pub struct ViewportShowResult {
    /// Dropped series (viewport_id, series)
    pub dropped_series: Vec<(ViewportId, DroppedSeries)>,
    /// Viewport that was clicked (if any)
    pub clicked_viewport: Option<ViewportId>,
}

/// Coregistration visual state for viewport manager
#[derive(Debug, Clone, Copy, Default)]
pub struct CoregistrationVisualState {
    /// Target viewport (highlighted in green)
    pub target: Option<ViewportId>,
    /// Whether we're selecting (viewports show dashed border)
    pub selecting: bool,
}

/// Manager for multiple viewports
pub struct ViewportManager {
    /// All viewport slots (up to MAX_VIEWPORTS)
    slots: Vec<ViewportSlot>,
    /// Current layout
    layout: ViewportLayout,
    /// Active viewport (receives keyboard input, presets)
    active_viewport: ViewportId,
    /// Whether synchronized scrolling is enabled
    sync_enabled: bool,
    /// Whether GPU rendering is available
    use_gpu: bool,
    /// Whether reference lines are enabled (on by default)
    reference_lines_enabled: bool,
    /// Coregistration visual state
    coregistration_state: CoregistrationVisualState,
}

impl ViewportManager {
    pub fn new(use_gpu: bool) -> Self {
        let mut slots = Vec::with_capacity(MAX_VIEWPORTS);
        for i in 0..MAX_VIEWPORTS {
            slots.push(ViewportSlot::new(i, use_gpu));
        }

        Self {
            slots,
            layout: ViewportLayout::Single,
            active_viewport: 0,
            sync_enabled: false,
            use_gpu,
            reference_lines_enabled: true, // On by default
            coregistration_state: CoregistrationVisualState::default(),
        }
    }

    /// Get current layout
    pub fn layout(&self) -> ViewportLayout {
        self.layout
    }

    /// Set layout
    pub fn set_layout(&mut self, layout: ViewportLayout) {
        self.layout = layout;
        // Ensure active viewport is within bounds
        if self.active_viewport >= layout.viewport_count() {
            self.active_viewport = 0;
        }
    }

    /// Get active viewport ID
    pub fn active_viewport(&self) -> ViewportId {
        self.active_viewport
    }

    /// Set active viewport
    pub fn set_active(&mut self, id: ViewportId) {
        if id < self.layout.viewport_count() {
            self.active_viewport = id;
        }
    }

    /// Set coregistration visual state
    pub fn set_coregistration_state(&mut self, state: CoregistrationVisualState) {
        self.coregistration_state = state;
    }

    /// Clear coregistration visual state
    pub fn clear_coregistration_state(&mut self) {
        self.coregistration_state = CoregistrationVisualState::default();
    }

    /// Get active viewport slot
    pub fn get_active(&self) -> &ViewportSlot {
        &self.slots[self.active_viewport]
    }

    /// Get active viewport slot mutably
    pub fn get_active_mut(&mut self) -> &mut ViewportSlot {
        &mut self.slots[self.active_viewport]
    }

    /// Get viewport slot by ID
    pub fn get_slot(&self, id: ViewportId) -> Option<&ViewportSlot> {
        if id < self.layout.viewport_count() {
            Some(&self.slots[id])
        } else {
            None
        }
    }

    /// Get viewport slot by ID mutably
    pub fn get_slot_mut(&mut self, id: ViewportId) -> Option<&mut ViewportSlot> {
        if id < self.layout.viewport_count() {
            Some(&mut self.slots[id])
        } else {
            None
        }
    }

    /// Whether sync is enabled
    pub fn sync_enabled(&self) -> bool {
        self.sync_enabled
    }

    /// Toggle synchronized scrolling
    pub fn toggle_sync(&mut self) {
        self.sync_enabled = !self.sync_enabled;
    }

    /// Whether reference lines are enabled
    pub fn reference_lines_enabled(&self) -> bool {
        self.reference_lines_enabled
    }

    /// Toggle reference lines
    pub fn toggle_reference_lines(&mut self) {
        self.reference_lines_enabled = !self.reference_lines_enabled;
    }

    /// Load series into a specific viewport
    /// If start_at_middle is true, starts at the middle slice instead of the first
    pub fn load_series(
        &mut self,
        id: ViewportId,
        files: Vec<PathBuf>,
        name: &str,
        sync_info: Option<SyncInfo>,
        start_at_middle: bool,
    ) {
        if let Some(slot) = self.get_slot_mut(id) {
            // Clear any active MPR state when loading a new series
            slot.clear_mpr();
            // Don't clear viewport here - let the old image stay until new one is loaded
            // Clearing mid-frame can cause GPU texture-in-use crashes
            let start_index = if start_at_middle && !files.is_empty() {
                files.len() / 2
            } else {
                0
            };
            slot.series_files = files;
            slot.current_index = start_index;
            slot.series_name = name.to_string();
            slot.sync_info = sync_info;
            slot.new_series = true;
        }
    }

    /// Navigate active viewport and sync others if enabled
    /// Returns list of (viewport_id, new_index) that need image loading
    pub fn navigate(&mut self, delta: i32) -> Vec<(ViewportId, usize)> {
        let active_id = self.active_viewport;
        let mut updates = Vec::new();

        // Navigate active viewport
        {
            let slot = &mut self.slots[active_id];
            if slot.series_files.is_empty() {
                return updates;
            }

            let old_index = slot.current_index;
            let max_index = slot.series_files.len().saturating_sub(1);
            let new_index = (old_index as i32 + delta).clamp(0, max_index as i32) as usize;

            if new_index != old_index {
                slot.current_index = new_index;
                updates.push((active_id, new_index));
            }
        }

        if !self.sync_enabled || updates.is_empty() {
            return updates;
        }

        // Get sync info from active viewport
        let active_sync = match &self.slots[active_id].sync_info {
            Some(info) => info.clone(),
            None => return updates,
        };

        let active_index = self.slots[active_id].current_index;
        let active_slice_loc = active_sync
            .slice_locations
            .get(active_index)
            .copied()
            .flatten();

        tracing::info!(
            "sync_viewports ENTRY: active_id={}, active_index={}, active_slice_loc={:?}, total_locs={}",
            active_id, active_index, active_slice_loc, active_sync.slice_locations.len()
        );

        // Sync other viewports (using indices to avoid borrow issues)
        let slot_count = self.slots.len();
        for i in 0..slot_count {
            if i == active_id || self.slots[i].series_files.is_empty() {
                continue;
            }

            if !should_sync(&active_sync, &self.slots[i].sync_info) {
                continue;
            }

            // Find closest slice in this viewport
            if let Some(target_loc) = active_slice_loc {
                if let Some(ref sync_info) = self.slots[i].sync_info {
                    let closest_idx = find_closest_slice(&sync_info.slice_locations, target_loc);
                    let other_loc = sync_info
                        .slice_locations
                        .get(closest_idx)
                        .copied()
                        .flatten();
                    tracing::info!(
                        "sync_viewports: target_loc={:.1}, closest_idx={}, other_loc={:?}, current_idx={}",
                        target_loc, closest_idx, other_loc, self.slots[i].current_index
                    );
                    if closest_idx != self.slots[i].current_index {
                        self.slots[i].current_index = closest_idx;
                        updates.push((self.slots[i].id, closest_idx));
                        tracing::info!(
                            "sync_viewports: updated viewport {} to index {}",
                            i,
                            closest_idx
                        );
                    }
                }
            } else {
                tracing::info!(
                    "sync_viewports: active_slice_loc is None for viewport {}",
                    active_id
                );
            }
        }

        updates
    }

    /// Sync other viewports to a given slice location (for MPR navigation)
    /// Returns list of (viewport_id, new_index) for viewports that need image loading
    /// This is called after MPR navigation to sync non-MPR viewports to the same location
    pub fn sync_to_location(
        &mut self,
        active_id: ViewportId,
        slice_location: f64,
    ) -> Vec<(ViewportId, usize)> {
        tracing::info!(
            "sync_to_location ENTRY: active_id={}, slice_location={:.2}, sync_enabled={}",
            active_id,
            slice_location,
            self.sync_enabled
        );

        if !self.sync_enabled {
            tracing::info!("sync_to_location: SKIPPED - sync disabled");
            return Vec::new();
        }

        let mut updates = Vec::new();

        // Get sync info from active viewport
        let active_sync = match &self.slots[active_id].sync_info {
            Some(info) => {
                let loc_range = info
                    .slice_locations
                    .iter()
                    .filter_map(|l| *l)
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                        (min.min(v), max.max(v))
                    });
                tracing::info!("sync_to_location: active sync_info: frame_of_ref={:?}, study_uid={:?}, orientation={:?}, loc_range=[{:.1}..{:.1}]",
                    info.frame_of_reference_uid.as_ref().map(|s| &s[s.len().saturating_sub(12)..]),
                    info.study_instance_uid.as_ref().map(|s| &s[s.len().saturating_sub(12)..]),
                    info.orientation,
                    loc_range.0, loc_range.1);
                info.clone()
            }
            None => {
                tracing::info!("sync_to_location: SKIPPED - no sync_info on active viewport");
                return updates;
            }
        };

        // Sync other viewports
        let slot_count = self.slots.len();
        for i in 0..slot_count {
            if i == active_id {
                continue;
            }

            // MPR viewports can now sync - they have proper slice_locations in sync_info
            // The should_sync check below will verify frame of reference and orientation match

            // Skip viewports without series
            if self.slots[i].series_files.is_empty() {
                continue;
            }

            if !should_sync(&active_sync, &self.slots[i].sync_info) {
                tracing::info!(
                    "sync_to_location: SKIPPED viewport {} - should_sync returned false",
                    i
                );
                continue;
            }

            // Find closest slice in this viewport
            if let Some(ref sync_info) = self.slots[i].sync_info {
                let closest_idx = find_closest_slice(&sync_info.slice_locations, slice_location);
                let closest_loc = sync_info
                    .slice_locations
                    .get(closest_idx)
                    .copied()
                    .flatten();
                tracing::info!("sync_to_location: viewport {} target={:.2}, closest_idx={}, closest_loc={:?}, current_idx={}",
                    i, slice_location, closest_idx, closest_loc, self.slots[i].current_index);
                if closest_idx != self.slots[i].current_index {
                    self.slots[i].current_index = closest_idx;
                    updates.push((self.slots[i].id, closest_idx));
                    tracing::info!(
                        "sync_to_location: UPDATED viewport {} to index {}",
                        i,
                        closest_idx
                    );
                }
            }
        }

        tracing::info!("sync_to_location RESULT: {} updates", updates.len());
        updates
    }
}

/// Check if two viewports should sync based on their sync info
fn should_sync(active_info: &SyncInfo, other_info: &Option<SyncInfo>) -> bool {
    let other = match other_info {
        Some(info) => info,
        None => {
            tracing::debug!("should_sync: other has no sync_info");
            return false;
        }
    };

    // Log full sync info comparison for debugging
    tracing::info!(
        "should_sync: comparing active(frame={:?}, study={:?}, orient={:?}, locs={}) vs other(frame={:?}, study={:?}, orient={:?}, locs={})",
        active_info.frame_of_reference_uid.as_ref().map(|s| &s[s.len().saturating_sub(8)..]),
        active_info.study_instance_uid.as_ref().map(|s| &s[s.len().saturating_sub(8)..]),
        active_info.orientation,
        active_info.slice_locations.len(),
        other.frame_of_reference_uid.as_ref().map(|s| &s[s.len().saturating_sub(8)..]),
        other.study_instance_uid.as_ref().map(|s| &s[s.len().saturating_sub(8)..]),
        other.orientation,
        other.slice_locations.len(),
    );

    // Must have same Frame of Reference UID
    match (
        &active_info.frame_of_reference_uid,
        &other.frame_of_reference_uid,
    ) {
        (Some(a), Some(b)) if a == b => {
            tracing::debug!("should_sync: frame_of_reference matches: {}", a);
        }
        _ => {
            tracing::debug!(
                "should_sync: frame_of_reference mismatch: {:?} vs {:?}",
                active_info.frame_of_reference_uid,
                other.frame_of_reference_uid
            );
            return false;
        }
    }

    // Study Instance UID check: must match if both have it, or allow if either is None
    // This enables sync between MPR (which might not have study_instance_uid) and original images
    match (&active_info.study_instance_uid, &other.study_instance_uid) {
        (Some(a), Some(b)) if a != b => {
            tracing::debug!("should_sync: study_instance_uid mismatch: {} vs {}", a, b);
            return false; // Different UIDs = no sync
        }
        // (Some, Some) with equal values = OK, continue to orientation check
        // (Some, None) or (None, Some) or (None, None) = OK, continue
        _ => {
            tracing::debug!(
                "should_sync: study_instance_uid OK: {:?} vs {:?}",
                active_info.study_instance_uid,
                other.study_instance_uid
            );
        }
    }

    // Must have same orientation
    match (active_info.orientation, other.orientation) {
        (Some(a), Some(b)) if a == b => {
            // Log slice location ranges to help debug sync issues
            let active_range = active_info
                .slice_locations
                .iter()
                .filter_map(|l| *l)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                    (min.min(v), max.max(v))
                });
            let other_range = other
                .slice_locations
                .iter()
                .filter_map(|l| *l)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                    (min.min(v), max.max(v))
                });
            tracing::info!(
                "should_sync: YES - orientation={:?}, active_loc_range=[{:.1}..{:.1}], other_loc_range=[{:.1}..{:.1}]",
                a, active_range.0, active_range.1, other_range.0, other_range.1
            );
            true
        }
        _ => {
            tracing::info!(
                "should_sync: NO - orientation mismatch: {:?} vs {:?}",
                active_info.orientation,
                other.orientation
            );
            false
        }
    }
}

impl ViewportManager {
    /// Calculate reference lines from the active viewport to all other viewports
    /// Returns a map of viewport_id -> list of reference lines to draw
    fn calculate_reference_lines(&self) -> Vec<Vec<ReferenceLine>> {
        let mut result: Vec<Vec<ReferenceLine>> = vec![Vec::new(); MAX_VIEWPORTS];

        if !self.reference_lines_enabled {
            return result;
        }

        let active_slot = &self.slots[self.active_viewport];
        let active_is_mpr = active_slot.mpr_state.is_active();

        // If active viewport is in MPR mode, also calculate MPR-to-MPR reference lines
        if active_is_mpr {
            result = self.calculate_mpr_reference_lines();
        }

        // Get the active viewport's image plane (works for both MPR and original images)
        let active_plane = match self.slots[self.active_viewport].viewport.image_plane() {
            Some(plane) => plane,
            None => return result,
        };

        // For each other viewport, calculate intersection using ImagePlane geometry
        let count = self.layout.viewport_count();
        for (i, other_slot) in self.slots[..count].iter().enumerate() {
            if i == self.active_viewport {
                continue; // Don't draw reference line on active viewport
            }

            // Skip MPR-to-MPR if already handled above (same volume)
            if active_is_mpr && other_slot.mpr_state.is_active() {
                let shares_volume =
                    match (&active_slot.mpr_state.volume, &other_slot.mpr_state.volume) {
                        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
                        _ => false,
                    };
                if shares_volume {
                    continue; // Already handled by calculate_mpr_reference_lines
                }
            }

            let other_plane = match other_slot.viewport.image_plane() {
                Some(plane) => plane,
                None => continue,
            };

            // Check if they share the same coordinate system
            if !active_plane.same_frame_of_reference(other_plane) {
                continue;
            }

            // Skip if planes are parallel (same orientation)
            if active_plane.is_parallel(other_plane) {
                continue;
            }

            // Calculate intersection
            if let Some((start, end)) = other_plane.intersect(active_plane) {
                let ref_line = ReferenceLine::new(start, end);
                result[i].push(ref_line);
            }
        }

        result
    }

    /// Calculate reference lines for MPR views (orthogonal planes)
    /// Reference lines are drawn on OTHER viewports to show where the active viewport's
    /// plane intersects them - same behavior as non-MPR reference lines.
    fn calculate_mpr_reference_lines(&self) -> Vec<Vec<ReferenceLine>> {
        let mut result: Vec<Vec<ReferenceLine>> = vec![Vec::new(); MAX_VIEWPORTS];
        let count = self.layout.viewport_count();

        let active_slot = &self.slots[self.active_viewport];
        let active_plane = active_slot.mpr_state.plane;
        let active_index = active_slot.mpr_state.slice_index;

        // Get the volume from the active viewport
        let Some(ref volume) = active_slot.mpr_state.volume else {
            return result;
        };
        let (_, vol_h, _) = volume.dimensions;

        // Bounds check: active_index must be valid for the active plane's slice count
        // Each plane has a different number of slices based on volume dimensions
        let max_slices = volume.slice_count(active_plane);
        if active_index >= max_slices {
            return result;
        }

        // For each other viewport in MPR mode with the same volume
        for (i, other_slot) in self.slots[..count].iter().enumerate() {
            if i == self.active_viewport {
                continue;
            }

            if !other_slot.mpr_state.is_active() {
                continue;
            }

            // Must share the same volume
            let shares_volume = match (&active_slot.mpr_state.volume, &other_slot.mpr_state.volume)
            {
                (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
                _ => false,
            };
            if !shares_volume {
                continue;
            }

            let other_plane = other_slot.mpr_state.plane;

            // Skip if same plane orientation
            if active_plane == other_plane {
                continue;
            }

            // Get the other viewport's image dimensions
            let Some(other_image) = other_slot.viewport.image() else {
                continue;
            };
            let other_w = other_image.width as f64;
            let other_h = other_image.height as f64;

            // Calculate reference line based on plane intersection
            // For orthogonal planes, the intersection is either horizontal or vertical
            let ref_line = match (active_plane, other_plane) {
                // Active Coronal (Y fixed) on Other views
                (AnatomicalPlane::Coronal, AnatomicalPlane::Axial) => {
                    // Coronal intersects Axial as horizontal line at row Y
                    let y = active_index as f64;
                    if y < other_h {
                        Some(
                            ReferenceLine::new(Point2D::new(0.0, y), Point2D::new(other_w, y))
                                .with_color(0, 255, 255),
                        ) // Cyan for coronal
                    } else {
                        None
                    }
                }
                (AnatomicalPlane::Coronal, AnatomicalPlane::Sagittal) => {
                    // Coronal intersects Sagittal as vertical line
                    // Y position maps to X in sagittal (left-right becomes which sagittal slice)
                    let x = vol_h.saturating_sub(1).saturating_sub(active_index) as f64; // Flip for proper orientation
                    if x >= 0.0 && x < other_w {
                        Some(
                            ReferenceLine::new(Point2D::new(x, 0.0), Point2D::new(x, other_h))
                                .with_color(0, 255, 255),
                        )
                    } else {
                        None
                    }
                }

                // Active Sagittal (X fixed) on Other views
                (AnatomicalPlane::Sagittal, AnatomicalPlane::Axial) => {
                    // Sagittal intersects Axial as vertical line at column X
                    let x = active_index as f64;
                    if x < other_w {
                        Some(
                            ReferenceLine::new(Point2D::new(x, 0.0), Point2D::new(x, other_h))
                                .with_color(255, 0, 255),
                        ) // Magenta for sagittal
                    } else {
                        None
                    }
                }
                (AnatomicalPlane::Sagittal, AnatomicalPlane::Coronal) => {
                    // Sagittal intersects Coronal as vertical line
                    let x = active_index as f64;
                    if x < other_w {
                        Some(
                            ReferenceLine::new(Point2D::new(x, 0.0), Point2D::new(x, other_h))
                                .with_color(255, 0, 255),
                        )
                    } else {
                        None
                    }
                }

                // Active Axial (Z fixed) on Other views
                (AnatomicalPlane::Axial, AnatomicalPlane::Coronal)
                | (AnatomicalPlane::Axial, AnatomicalPlane::Sagittal) => {
                    // Axial intersects Coronal/Sagittal as horizontal line
                    // The Y position depends on the volume's Z-axis direction:
                    // - Sagittal/Coronal views display with superior at top (y=0)
                    // - If z_sign > 0: higher slice index = more superior = smaller y
                    // - If z_sign < 0: higher slice index = more inferior = larger y
                    let axial_slice_count =
                        volume.slice_count(crate::dicom::AnatomicalPlane::Axial);
                    let max_index = axial_slice_count.saturating_sub(1).max(1) as f64;

                    // Check z_sign to determine mapping direction
                    let (_, _, z_sign) = volume.get_axis_direction(PatientAxis::Z);

                    // Calculate base y_normalized from z_sign
                    let base_y = if z_sign > 0.0 {
                        1.0 - (active_index as f64 / max_index)
                    } else {
                        active_index as f64 / max_index
                    };

                    // For coregistered volumes, invert the result (geometry mismatch)
                    let y_normalized = if volume.is_coregistered {
                        1.0 - base_y
                    } else {
                        base_y
                    };
                    let y = y_normalized * other_h;

                    if y >= 0.0 && y <= other_h {
                        Some(
                            ReferenceLine::new(Point2D::new(0.0, y), Point2D::new(other_w, y))
                                .with_color(255, 255, 0),
                        ) // Yellow for axial
                    } else {
                        None
                    }
                }

                // Original plane - no crosshairs
                (AnatomicalPlane::Original, _) | (_, AnatomicalPlane::Original) => None,

                // Same plane - skip (handled above)
                _ => None,
            };

            if let Some(line) = ref_line {
                result[i].push(line);
            }
        }

        result
    }

    /// Show all viewports according to current layout
    /// Returns ViewportShowResult with dropped series and clicked viewport
    pub fn show(&mut self, ui: &mut Ui, mode: InteractionMode) -> ViewportShowResult {
        let mut drops = Vec::new();
        let mut clicked: Option<ViewportId> = None;
        let active_id = self.active_viewport;
        let count = self.layout.viewport_count();

        // Calculate reference lines from active viewport to others
        let ref_lines = self.calculate_reference_lines();

        match self.layout {
            ViewportLayout::Single => {
                let resp = self.show_viewport(ui, 0, true, &ref_lines[0], mode);
                if let Some(series) = resp.dropped_series {
                    drops.push((0, series));
                }
                if resp.clicked {
                    clicked = Some(0);
                }
                // Single viewport is always active
            }
            ViewportLayout::Horizontal2 | ViewportLayout::Horizontal3 => {
                let mut new_active: Option<ViewportId> = None;
                let mut clicked_inner: Option<ViewportId> = None;
                ui.columns(count, |cols| {
                    for (i, col) in cols.iter_mut().enumerate() {
                        let resp = self.show_viewport(col, i, i == active_id, &ref_lines[i], mode);
                        if let Some(series) = resp.dropped_series {
                            drops.push((i, series));
                        }
                        if resp.hovered {
                            new_active = Some(i);
                        }
                        if resp.clicked {
                            clicked_inner = Some(i);
                        }
                    }
                });
                if let Some(id) = new_active {
                    self.active_viewport = id;
                }
                clicked = clicked_inner;
            }
            ViewportLayout::Grid2x2 => {
                clicked = self.show_grid(ui, 2, 2, active_id, &mut drops, &ref_lines, mode);
            }
            ViewportLayout::Grid3x2 => {
                clicked = self.show_grid(ui, 3, 2, active_id, &mut drops, &ref_lines, mode);
            }
            ViewportLayout::Grid4x2 => {
                clicked = self.show_grid(ui, 4, 2, active_id, &mut drops, &ref_lines, mode);
            }
            ViewportLayout::Grid2x2Plus1 => {
                // 2x2 grid on left, 1 tall viewport on right
                let available = ui.available_size();
                let left_width = available.x * 0.6;
                let right_width = available.x * 0.4 - 4.0;

                let mut right_hovered = false;
                let mut right_clicked = false;
                let mut grid_clicked: Option<ViewportId> = None;
                ui.horizontal(|ui| {
                    // Left side: 2x2 grid
                    ui.allocate_ui(egui::vec2(left_width, available.y), |ui| {
                        grid_clicked =
                            self.show_grid(ui, 2, 2, active_id, &mut drops, &ref_lines, mode);
                    });

                    ui.add_space(4.0);

                    // Right side: single tall viewport
                    ui.allocate_ui(egui::vec2(right_width, available.y), |ui| {
                        let resp = self.show_viewport(ui, 4, 4 == active_id, &ref_lines[4], mode);
                        if let Some(series) = resp.dropped_series {
                            drops.push((4, series));
                        }
                        if resp.hovered {
                            right_hovered = true;
                        }
                        if resp.clicked {
                            right_clicked = true;
                        }
                    });
                });
                if right_hovered {
                    self.active_viewport = 4;
                }
                clicked = if right_clicked { Some(4) } else { grid_clicked };
            }
        }

        ViewportShowResult {
            dropped_series: drops,
            clicked_viewport: clicked,
        }
    }

    /// Show a grid of viewports
    /// Returns clicked viewport ID if any
    #[allow(clippy::too_many_arguments)]
    fn show_grid(
        &mut self,
        ui: &mut Ui,
        cols: usize,
        rows: usize,
        active_id: ViewportId,
        drops: &mut Vec<(ViewportId, DroppedSeries)>,
        ref_lines: &[Vec<ReferenceLine>],
        mode: InteractionMode,
    ) -> Option<ViewportId> {
        let available = ui.available_size();
        let cell_width = (available.x - (cols as f32 - 1.0) * 4.0) / cols as f32;
        let cell_height = (available.y - (rows as f32 - 1.0) * 4.0) / rows as f32;

        let mut new_active: Option<ViewportId> = None;
        let mut clicked: Option<ViewportId> = None;
        egui::Grid::new("viewport_grid")
            .num_columns(cols)
            .spacing([4.0, 4.0])
            .show(ui, |ui| {
                for row in 0..rows {
                    for col in 0..cols {
                        let id = row * cols + col;
                        ui.allocate_ui(egui::vec2(cell_width, cell_height), |ui| {
                            let resp =
                                self.show_viewport(ui, id, id == active_id, &ref_lines[id], mode);
                            if let Some(series) = resp.dropped_series {
                                drops.push((id, series));
                            }
                            if resp.hovered {
                                new_active = Some(id);
                            }
                            if resp.clicked {
                                clicked = Some(id);
                            }
                        });
                    }
                    ui.end_row();
                }
            });
        if let Some(id) = new_active {
            self.active_viewport = id;
        }
        clicked
    }

    /// Show a single viewport with active indicator
    fn show_viewport(
        &mut self,
        ui: &mut Ui,
        id: ViewportId,
        is_active: bool,
        ref_lines: &[ReferenceLine],
        mode: InteractionMode,
    ) -> ViewportResponse {
        let rect = ui.available_rect_before_wrap();
        let coreg_state = self.coregistration_state;
        let slot = &mut self.slots[id];

        // Draw border based on state
        let is_coreg_target = coreg_state.target == Some(id);
        let is_coreg_selecting = coreg_state.selecting;

        if is_coreg_target {
            // Target viewport: green border
            ui.painter().rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(3.0, Color32::from_rgb(50, 205, 50)), // Lime green
            );
        } else if is_coreg_selecting {
            // Selecting mode: orange dashed border for other viewports
            ui.painter().rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(2.0, Color32::from_rgb(255, 165, 0)), // Orange
            );
        } else if is_active {
            // Normal active viewport: cornflower blue
            ui.painter().rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(2.0, Color32::from_rgb(100, 149, 237)), // Cornflower blue
            );
        }

        let fusion = if slot.fusion_state.is_active() {
            Some(&slot.fusion_state)
        } else {
            None
        };
        let dropped =
            slot.viewport
                .show_with_reference_lines_and_fusion(ui, ref_lines, mode, fusion);

        // Track hover and click
        // Don't consider viewport hovered if mouse is over any egui window (database, sidebar, etc.)
        let hovered = ui.rect_contains_pointer(rect) && !ui.ctx().is_pointer_over_area();
        let clicked = hovered && ui.input(|i| i.pointer.primary_clicked());

        ViewportResponse {
            clicked,
            hovered,
            dropped_series: dropped,
        }
    }

    /// Set image on a viewport (resets zoom - use for new series)
    pub fn set_image(&mut self, id: ViewportId, image: DicomImage) {
        if let Some(slot) = self.get_slot_mut(id) {
            slot.viewport.set_image(image);
        }
    }

    /// Set image on a viewport while preserving zoom/pan (use when scrolling)
    pub fn set_image_keep_view(&mut self, id: ViewportId, image: DicomImage) {
        if let Some(slot) = self.get_slot_mut(id) {
            slot.viewport.set_image_keep_view(image);
        }
    }

    /// Get status text for the status bar
    pub fn status_text(&self) -> String {
        let slot = &self.slots[self.active_viewport];
        if slot.series_files.is_empty() {
            return "No series loaded".to_string();
        }

        let mut flags = Vec::new();
        if self.sync_enabled {
            flags.push("SYNC".to_string());
        }
        if self.reference_lines_enabled {
            flags.push("REF".to_string());
        }
        // Show fusion state in status bar
        if slot.fusion_state.is_active() {
            let fusion_text = slot.fusion_state.status_text();
            if !fusion_text.is_empty() {
                flags.push(fusion_text);
            }
        }
        let flags_str = if flags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", flags.join(" | "))
        };

        format!(
            "{} - {}/{} images{}",
            slot.series_name,
            slot.current_index + 1,
            slot.series_files.len(),
            flags_str
        )
    }

    /// Get window/level from active viewport
    pub fn window_level(&self) -> (f64, f64) {
        self.slots[self.active_viewport].viewport.window_level()
    }

    /// Clear all viewports (for patient switching)
    pub fn clear_all(&mut self) {
        for slot in &mut self.slots {
            slot.viewport.clear();
            slot.series_files.clear();
            slot.current_index = 0;
            slot.series_name.clear();
            slot.sync_info = None;
            slot.new_series = false;
            slot.clear_mpr();
            slot.fusion_state = FusionState::new();
        }
    }

    /// Set sync enabled/disabled directly
    pub fn set_sync_enabled(&mut self, enabled: bool) {
        self.sync_enabled = enabled;
    }

    /// Iterate over viewport slots (immutable)
    pub fn slots(&self) -> impl Iterator<Item = &ViewportSlot> {
        let count = self.layout.viewport_count();
        self.slots[..count].iter()
    }

    /// Iterate over viewport slots (mutable)
    pub fn slots_mut(&mut self) -> impl Iterator<Item = &mut ViewportSlot> {
        let count = self.layout.viewport_count();
        self.slots[..count].iter_mut()
    }
}

/// Find the index with the closest slice location
fn find_closest_slice(slice_locations: &[Option<f64>], target: f64) -> usize {
    slice_locations
        .iter()
        .enumerate()
        .filter_map(|(i, loc)| loc.map(|l| (i, (l - target).abs())))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
