//! Viewport widget for displaying DICOM images
#![allow(dead_code)]

use crate::dicom::{DicomImage, ImagePlane, ReferenceLine, SyncInfo, CT_PRESETS};
use crate::fusion::FusionState;
use crate::gpu::{DicomPaintCallback, FusionPaintCallback, FusionUniforms, WindowingUniforms};
use crate::ui::SeriesDragPayload;
use egui::{Color32, ColorImage, Pos2, Rect, Sense, TextureHandle, TextureOptions, Vec2};
use std::path::PathBuf;
use std::time::Instant;

/// Minimum interval between texture updates during windowing (ms) - CPU fallback only
const TEXTURE_UPDATE_THROTTLE_MS: u128 = 50; // ~20 FPS during drag

/// Mouse interaction mode for viewports
/// Note: Right-click always scrolls (handled at app level), these modes control left-click behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InteractionMode {
    /// Windowing mode (default): left-click = window/level, shift+left = pan
    #[default]
    Windowing,
    /// Pan mode: left-click = pan
    Pan,
    /// Length measurement mode: click two points
    Measure,
    /// Circle ROI mode: click-drag to define circle
    CircleROI,
    /// ROI-based windowing mode: click-drag circle, apply window from stats
    ROIWindowing,
}

impl InteractionMode {
    /// Get display name for status bar
    pub fn display_name(&self) -> &'static str {
        match self {
            InteractionMode::Windowing => "Windowing",
            InteractionMode::Pan => "Pan",
            InteractionMode::Measure => "Length",
            InteractionMode::CircleROI => "Circle ROI",
            InteractionMode::ROIWindowing => "ROI Window",
        }
    }

    /// Get status bar hint for left/right click
    pub fn mouse_hints(&self) -> (&'static str, &'static str) {
        match self {
            InteractionMode::Windowing => ("Window/Level", "Scroll"),
            InteractionMode::Pan => ("Pan", "Scroll"),
            InteractionMode::Measure => ("Set point", "Scroll"),
            InteractionMode::CircleROI => ("Draw ROI", "Scroll"),
            InteractionMode::ROIWindowing => ("Draw ROI", "Scroll"),
        }
    }
}

/// Dropped series info returned from viewport
pub struct DroppedSeries {
    pub files: Vec<PathBuf>,
    pub label: String,
    pub sync_info: Option<SyncInfo>,
}

/// Window/level adjustment state
pub struct WindowingState {
    pub window_center: f64,
    pub window_width: f64,
    pub adjusting_wl: bool,
    pub last_mouse_pos: Option<Pos2>,
}

/// View transform (zoom and pan)
pub struct ViewTransform {
    pub zoom: f32,
    pub pan: Vec2,
    pub is_dragging: bool,
}

/// A single viewport for displaying a DICOM image
pub struct Viewport {
    /// Viewport ID (also used as GPU texture slot)
    id: usize,
    /// Current image
    image: Option<DicomImage>,
    /// Cached texture for display (CPU fallback)
    texture: Option<TextureHandle>,
    /// Texture pending destruction (deferred to avoid use-after-free)
    /// Kept alive until next frame to ensure egui finishes rendering with it
    texture_graveyard: Option<TextureHandle>,
    /// Windowing state (window center/width, adjustment tracking)
    windowing: WindowingState,
    /// View transform (zoom, pan, drag state)
    view: ViewTransform,
    /// Whether texture needs update (CPU fallback)
    needs_update: bool,
    /// Whether to fit to window on next frame
    needs_fit: bool,
    /// Whether to apply smart fit (anatomy bounds + smart window) on next frame
    needs_smart_fit: bool,
    /// Last known viewport size
    last_size: Vec2,
    /// Last rendered rect (for coordinate conversion)
    last_rect: Rect,
    /// Last texture update time (for throttling, CPU fallback)
    last_texture_update: Instant,
    /// Whether GPU rendering is available
    use_gpu: bool,
    /// Rescale values for GPU uniforms
    rescale_slope: f32,
    rescale_intercept: f32,
    /// Pixel representation (0 = unsigned, 1 = signed)
    pixel_representation: u16,
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Viewport {
    pub fn new(use_gpu: bool) -> Self {
        Self::with_id(0, use_gpu)
    }

    /// Create a viewport with a specific ID (for multi-viewport support)
    pub fn with_id(id: usize, use_gpu: bool) -> Self {
        let default_size = Vec2::new(1400.0, 800.0);
        Self {
            id,
            image: None,
            texture: None,
            texture_graveyard: None,
            windowing: WindowingState {
                window_center: 40.0,
                window_width: 400.0,
                adjusting_wl: false,
                last_mouse_pos: None,
            },
            view: ViewTransform {
                zoom: 1.0,
                pan: Vec2::ZERO,
                is_dragging: false,
            },
            needs_update: true,
            needs_fit: false,
            needs_smart_fit: false,
            last_size: default_size,
            last_rect: Rect::from_min_size(Pos2::ZERO, default_size),
            last_texture_update: Instant::now(),
            use_gpu,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            pixel_representation: 0,
        }
    }

    /// Get the viewport ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Set the image to display (resets zoom and pan - use for new series)
    pub fn set_image(&mut self, image: DicomImage) {
        // Clear old texture first to avoid GPU texture-in-use issues
        self.texture = None;
        self.windowing.window_center = image.window_center;
        self.windowing.window_width = image.window_width;
        self.rescale_slope = image.rescale_slope as f32;
        self.rescale_intercept = image.rescale_intercept as f32;
        self.pixel_representation = image.pixel_representation;
        self.image = Some(image);
        self.needs_update = true;
        self.needs_smart_fit = true; // Smart fit + smart window on next frame
        self.view.pan = Vec2::ZERO;
    }

    /// Set the image while preserving current zoom, pan, and windowing (use when scrolling through series)
    pub fn set_image_keep_view(&mut self, image: DicomImage) {
        // Clear old texture first to avoid GPU texture-in-use issues
        self.texture = None;
        // Preserve window center/width - don't reset to image defaults
        self.rescale_slope = image.rescale_slope as f32;
        self.rescale_intercept = image.rescale_intercept as f32;
        self.pixel_representation = image.pixel_representation;
        self.image = Some(image);
        self.needs_update = true;
        // Don't reset needs_fit, zoom, pan, or window/level - preserve current view
    }

    /// Get current window/level
    pub fn window_level(&self) -> (f64, f64) {
        (self.windowing.window_center, self.windowing.window_width)
    }

    /// Set window/level
    pub fn set_window_level(&mut self, center: f64, width: f64) {
        self.windowing.window_center = center;
        self.windowing.window_width = width.max(1.0);
        self.needs_update = true;
    }

    /// Apply a preset window
    pub fn apply_preset(&mut self, preset_index: usize) {
        if let Some(preset) = CT_PRESETS.get(preset_index) {
            self.set_window_level(preset.center, preset.width);
        }
    }

    /// Auto-optimize window/level for the current image
    pub fn auto_window(&mut self) {
        if let Some(ref image) = self.image {
            // Recalculate from image data
            let (center, width) = calculate_optimal_window(
                &image.pixels,
                image.rescale_slope,
                image.rescale_intercept,
            );
            self.set_window_level(center, width);
        }
    }

    /// Reset zoom and pan
    pub fn reset_view(&mut self) {
        self.view.zoom = 1.0;
        self.view.pan = Vec2::ZERO;
    }

    /// Clear the viewport (remove image and texture)
    pub fn clear(&mut self) {
        // Move texture to graveyard instead of dropping immediately
        // This prevents use-after-free when egui has queued render commands
        if let Some(tex) = self.texture.take() {
            self.texture_graveyard = Some(tex);
        }
        self.image = None;
        self.needs_update = false;
        self.needs_fit = false;
        self.needs_smart_fit = false;
    }

    /// Calculate pixel aspect ratio (row_spacing / col_spacing)
    /// Returns > 1.0 if rows are wider than columns (image needs vertical stretch)
    /// Returns < 1.0 if columns are wider than rows (image needs horizontal stretch)
    fn pixel_ratio(&self) -> f32 {
        self.image
            .as_ref()
            .and_then(|img| img.pixel_spacing)
            .map(|(row_spacing, col_spacing)| (row_spacing / col_spacing) as f32)
            .unwrap_or(1.0)
    }

    /// Fit image to window (accounting for non-square pixel spacing)
    pub fn fit_to_window(&mut self, available_size: Vec2) {
        if let Some(ref image) = self.image {
            let ratio = self.pixel_ratio();
            let img_width = image.width as f32;
            // Physical height relative to width = pixel_height * pixel_ratio
            let img_height = image.height as f32 * ratio;

            self.view.zoom = (available_size.x / img_width).min(available_size.y / img_height);
            self.view.pan = Vec2::ZERO;
        }
    }

    /// Smart fit to anatomy bounds (excluding background)
    pub fn smart_fit_to_window(&mut self, available_size: Vec2) {
        if let Some(ref image) = self.image {
            if let Some((x_min, y_min, x_max, y_max)) = image.find_anatomy_bounds() {
                let ratio = self.pixel_ratio();
                let anatomy_width = (x_max - x_min + 1) as f32;
                // Physical height relative to width
                let anatomy_height = (y_max - y_min + 1) as f32 * ratio;

                // Calculate zoom to fit anatomy
                self.view.zoom =
                    (available_size.x / anatomy_width).min(available_size.y / anatomy_height);

                // Calculate pan to center the anatomy
                let anatomy_center_x = (x_min + x_max) as f32 / 2.0;
                let anatomy_center_y = (y_min + y_max) as f32 / 2.0;
                let image_center_x = image.width as f32 / 2.0;
                let image_center_y = image.height as f32 / 2.0;

                // Pan offset to move anatomy center to viewport center
                self.view.pan = Vec2::new(
                    (image_center_x - anatomy_center_x) * self.view.zoom,
                    (image_center_y - anatomy_center_y) * self.view.zoom * ratio,
                );
            } else {
                // Fall back to regular fit if no anatomy bounds found
                self.fit_to_window(available_size);
            }
        }
    }

    /// Set 1:1 pixel display
    pub fn set_one_to_one(&mut self) {
        self.view.zoom = 1.0;
    }

    /// Convert screen coordinates to image pixel coordinates
    pub fn screen_to_image(&self, screen_pos: Pos2, rect: Rect) -> Option<(f64, f64)> {
        let image = self.image.as_ref()?;
        let img_size = Vec2::new(image.width as f32, image.height as f32);
        let scaled_size = img_size * self.view.zoom;

        // Calculate image position in viewport
        let center = rect.center() + self.view.pan;
        let img_left = center.x - scaled_size.x / 2.0;
        let img_top = center.y - scaled_size.y / 2.0;

        // Convert screen to image coordinates
        let img_x = (screen_pos.x - img_left) / self.view.zoom;
        let img_y = (screen_pos.y - img_top) / self.view.zoom;

        // Check bounds
        if img_x >= 0.0 && img_x < image.width as f32 && img_y >= 0.0 && img_y < image.height as f32
        {
            Some((img_x as f64, img_y as f64))
        } else {
            None
        }
    }

    /// Convert image pixel coordinates to screen coordinates
    pub fn image_to_screen(&self, img_x: f64, img_y: f64, rect: Rect) -> Pos2 {
        if let Some(ref image) = self.image {
            let img_size = Vec2::new(image.width as f32, image.height as f32);
            let scaled_size = img_size * self.view.zoom;

            let center = rect.center() + self.view.pan;
            let img_left = center.x - scaled_size.x / 2.0;
            let img_top = center.y - scaled_size.y / 2.0;

            Pos2::new(
                img_left + img_x as f32 * self.view.zoom,
                img_top + img_y as f32 * self.view.zoom,
            )
        } else {
            rect.center()
        }
    }

    /// Calculate distance in mm (or pixels if no calibration)
    pub fn calculate_distance(&self, p1: (f64, f64), p2: (f64, f64)) -> (f64, bool) {
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let pixel_dist = (dx * dx + dy * dy).sqrt();

        if let Some(ref image) = self.image {
            if let Some((row_spacing, col_spacing)) = image.pixel_spacing {
                // Calculate distance in mm using pixel spacing
                let dx_mm = dx * col_spacing;
                let dy_mm = dy * row_spacing;
                let mm_dist = (dx_mm * dx_mm + dy_mm * dy_mm).sqrt();
                return (mm_dist, true);
            }
        }

        // No calibration - return pixel distance
        (pixel_dist, false)
    }

    /// Get pixel spacing if available
    pub fn pixel_spacing(&self) -> Option<(f64, f64)> {
        self.image.as_ref().and_then(|img| img.pixel_spacing)
    }

    /// Get SOP Instance UID if available
    pub fn sop_instance_uid(&self) -> Option<&str> {
        self.image
            .as_ref()
            .and_then(|img| img.sop_instance_uid.as_deref())
    }


    /// Get reference to current image
    pub fn image(&self) -> Option<&crate::dicom::DicomImage> {
        self.image.as_ref()
    }

    /// Get current zoom level
    pub fn zoom(&self) -> f32 {
        self.view.zoom
    }

    /// Get last rendered rect
    pub fn last_rect(&self) -> Rect {
        self.last_rect
    }

    /// Temporarily disable GPU rendering (e.g., for MPR mode where images are generated in memory)
    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        self.use_gpu = use_gpu;
        if !use_gpu {
            // Force texture regeneration on next frame
            self.needs_update = true;
        }
    }

    /// Check if GPU rendering is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.use_gpu
    }

    /// Set window center and width
    pub fn set_window(&mut self, center: f64, width: f64) {
        self.windowing.window_center = center;
        self.windowing.window_width = width.max(1.0);
        self.needs_update = true;
    }

    /// Update texture if needed (CPU fallback)
    fn update_texture_cpu(&mut self, ctx: &egui::Context) {
        if !self.needs_update {
            return;
        }

        // Throttle updates during dragging to keep UI responsive
        if self.view.is_dragging {
            let elapsed = self.last_texture_update.elapsed().as_millis();
            if elapsed < TEXTURE_UPDATE_THROTTLE_MS {
                // Skip this update, but request a repaint to try again
                ctx.request_repaint();
                return;
            }
        }

        if let Some(ref image) = self.image {
            tracing::debug!(
                "CPU: Updating texture for {}x{} image",
                image.width,
                image.height
            );
            let rgba = image.to_rgba(self.windowing.window_center, self.windowing.window_width);
            let size = [image.width as usize, image.height as usize];
            let color_image = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

            self.texture =
                Some(ctx.load_texture("dicom_image", color_image, TextureOptions::LINEAR));
            self.last_texture_update = Instant::now();
            self.needs_update = false;
            tracing::debug!("CPU: Texture updated successfully");
        }
    }

    /// Calculate scale and offset for GPU rendering
    fn calculate_gpu_transform(&self, viewport_size: Vec2) -> ([f32; 2], [f32; 2]) {
        if let Some(ref image) = self.image {
            let ratio = self.pixel_ratio();
            let img_size = Vec2::new(image.width as f32, image.height as f32);
            let scaled_size = img_size * self.view.zoom;

            // Calculate aspect ratio preserving scale in NDC
            // Apply pixel_ratio to Y to account for non-square pixels
            let scale_x = scaled_size.x / viewport_size.x;
            let scale_y = (scaled_size.y * ratio) / viewport_size.y;

            // Pan offset in NDC (normalized device coordinates)
            let offset_x = self.view.pan.x / viewport_size.x * 2.0;
            let offset_y = -self.view.pan.y / viewport_size.y * 2.0; // Y is flipped in NDC

            ([scale_x, scale_y], [offset_x, offset_y])
        } else {
            ([1.0, 1.0], [0.0, 0.0])
        }
    }

    /// Show the viewport UI
    /// Returns series info if a series was dropped on this viewport
    pub fn show(&mut self, ui: &mut egui::Ui, mode: InteractionMode) -> Option<DroppedSeries> {
        // Clear texture graveyard - safe now that previous frame is complete
        self.texture_graveyard = None;

        let available_size = ui.available_size();

        // Re-fit when viewport size changes (e.g. sidebar open/close)
        if self.image.is_some()
            && !self.needs_fit
            && !self.needs_smart_fit
            && ((available_size.x - self.last_size.x).abs() > 1.0
                || (available_size.y - self.last_size.y).abs() > 1.0)
        {
            self.smart_fit_to_window(available_size);
        }

        self.last_size = available_size;

        // Fit to window if needed (after we know the actual size)
        if self.needs_fit && self.image.is_some() {
            self.fit_to_window(available_size);
            self.needs_fit = false;
        }

        // Smart fit (anatomy bounds + smart windowing) if needed
        if self.needs_smart_fit && self.image.is_some() {
            // Smart zoom: fit to anatomy bounds
            self.smart_fit_to_window(available_size);

            // Smart window: for non-CT images, calculate optimal window
            // Extract info before borrowing self mutably
            let smart_window_info = self.image.as_ref().and_then(|image| {
                if !image.is_ct() {
                    image
                        .find_anatomy_bounds()
                        .map(|(x_min, y_min, x_max, y_max)| {
                            image.calculate_content_window(x_min, y_min, x_max, y_max)
                        })
                } else {
                    None
                }
            });

            if let Some((wc, ww)) = smart_window_info {
                self.windowing.window_center = wc;
                self.windowing.window_width = ww;
                self.needs_update = true;
            }

            self.needs_smart_fit = false;
        }

        // Create the viewport area
        let (response, painter) = ui.allocate_painter(available_size, Sense::click_and_drag());
        let rect = response.rect;

        // Check for dropped series
        let dropped_series = response
            .dnd_release_payload::<SeriesDragPayload>()
            .map(|payload| DroppedSeries {
                files: payload.files.clone(),
                label: payload.label.clone(),
                sync_info: payload.sync_info.clone(),
            });

        // Show drop highlight when dragging over
        if response.dnd_hover_payload::<SeriesDragPayload>().is_some() {
            painter.rect_stroke(rect, 4.0, egui::Stroke::new(3.0, Color32::LIGHT_BLUE));
        }

        // Draw background
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        // Render image
        if self.use_gpu && self.image.is_some() {
            // GPU rendering path
            self.render_gpu(ui, rect);
        } else {
            // CPU fallback path
            self.update_texture_cpu(ui.ctx());
            self.render_cpu(&painter, rect);
        }

        // Handle interactions
        self.handle_input(ui, &response, mode);

        // Show info overlay
        self.show_info_overlay(ui, rect);

        dropped_series
    }

    /// Render using GPU pipeline
    fn render_gpu(&self, ui: &mut egui::Ui, rect: Rect) {
        let (scale, offset) = self.calculate_gpu_transform(rect.size());

        let uniforms = WindowingUniforms {
            window_center: self.windowing.window_center as f32,
            window_width: self.windowing.window_width as f32,
            rescale_slope: self.rescale_slope,
            rescale_intercept: self.rescale_intercept,
            scale,
            offset,
            pixel_representation: self.pixel_representation as u32,
            _padding: 0,
        };

        let callback = DicomPaintCallback {
            uniforms,
            slot: self.id,
        };

        // Use egui_wgpu::Callback wrapper for proper integration
        let wgpu_callback = eframe::egui_wgpu::Callback::new_paint_callback(rect, callback);
        ui.painter().add(wgpu_callback);
    }

    /// Render using fusion GPU pipeline (two textures blended)
    pub fn render_fusion(&self, ui: &mut egui::Ui, rect: Rect, fusion: &FusionState) {
        let (scale, offset) = self.calculate_gpu_transform(rect.size());

        let uniforms = FusionUniforms {
            opacity: fusion.opacity,
            blend_mode: fusion.blend_mode.shader_index(),
            base_wc: fusion.base_window_center,
            base_ww: fusion.base_window_width,
            base_slope: fusion.base_rescale_slope,
            base_intercept: fusion.base_rescale_intercept,
            base_pixel_rep: fusion.base_pixel_representation,
            overlay_wc: fusion.overlay_window_center,
            overlay_ww: fusion.overlay_window_width,
            overlay_slope: fusion.overlay_rescale_slope,
            overlay_intercept: fusion.overlay_rescale_intercept,
            overlay_pixel_rep: fusion.overlay_pixel_representation,
            scale,
            offset,
            colormap_index: fusion.colormap_index,
            checker_size: fusion.checker_size,
            threshold: fusion.threshold,
            _padding: 0.0,
        };

        let callback = FusionPaintCallback {
            uniforms,
            slot: self.id,
        };

        let wgpu_callback = eframe::egui_wgpu::Callback::new_paint_callback(rect, callback);
        ui.painter().add(wgpu_callback);
    }

    /// Render using CPU texture (fallback)
    fn render_cpu(&self, painter: &egui::Painter, rect: Rect) {
        if let Some(ref texture) = self.texture {
            if let Some(ref image) = self.image {
                let ratio = self.pixel_ratio();
                let img_size = Vec2::new(image.width as f32, image.height as f32);
                // Apply pixel_ratio to height for non-square pixels
                let scaled_size = Vec2::new(img_size.x * self.view.zoom, img_size.y * self.view.zoom * ratio);

                // Center the image with pan offset
                let center = rect.center() + self.view.pan;
                let img_rect = Rect::from_center_size(center, scaled_size);

                painter.image(
                    texture.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        } else if self.image.is_none() {
            // Show placeholder text
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drag DICOM file here",
                egui::FontId::proportional(18.0),
                Color32::GRAY,
            );
        }
    }

    /// Handle mouse/keyboard input
    fn handle_input(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        mode: InteractionMode,
    ) {
        let ctx = ui.ctx();

        // Get modifier and drag state
        let shift_held = ctx.input(|i| i.modifiers.shift);
        let ctrl_held = ctx.input(|i| i.modifiers.ctrl);
        let dragging_primary = response.dragged_by(egui::PointerButton::Primary);

        // Determine if we should do windowing:
        // - Windowing mode (default): left-click = windowing (unless shift held for pan)
        // - Pan mode: no windowing from left-click
        // - Measurement modes: no windowing (mouse is used for measurement)
        // Note: Right-click scroll is handled at app level, not here
        let windowing = match mode {
            InteractionMode::Windowing => dragging_primary && !shift_held && !ctrl_held,
            InteractionMode::Pan => false,
            InteractionMode::Measure
            | InteractionMode::CircleROI
            | InteractionMode::ROIWindowing => false,
        };

        if windowing {
            self.view.is_dragging = true;
            if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
                if let Some(last_pos) = self.windowing.last_mouse_pos {
                    let delta = pos - last_pos;
                    // Scale sensitivity based on current window width
                    // Larger windows need faster adjustment
                    let sensitivity = (self.windowing.window_width / 100.0).clamp(1.0, 50.0);
                    // Horizontal = window width, Vertical = window center
                    self.windowing.window_width = (self.windowing.window_width + delta.x as f64 * sensitivity).max(1.0);
                    self.windowing.window_center += delta.y as f64 * sensitivity;
                    self.needs_update = true;
                }
                self.windowing.last_mouse_pos = Some(pos);
                self.windowing.adjusting_wl = true;
            }
        } else if self.windowing.adjusting_wl {
            // Drag ended - do final update at full quality
            self.view.is_dragging = false;
            self.needs_update = true; // Force final update
            self.windowing.last_mouse_pos = None;
            self.windowing.adjusting_wl = false;
        }

        // Left-click drag for pan:
        // - Pan mode: always pan with left-click
        // - Windowing mode: shift+left = pan
        // - Never pan with ctrl held (that's zoom)
        let can_pan = match mode {
            InteractionMode::Windowing => shift_held && !ctrl_held, // Shift+left = pan in windowing mode
            InteractionMode::Pan => !ctrl_held,                     // Left-click = pan in pan mode
            InteractionMode::Measure
            | InteractionMode::CircleROI
            | InteractionMode::ROIWindowing => false,
        };
        if dragging_primary && can_pan {
            self.view.pan += response.drag_delta();
        }

        // Middle-click drag also for pan
        if response.dragged_by(egui::PointerButton::Middle) {
            self.view.pan += response.drag_delta();
        }

        // Ctrl + drag for zoom (trackpad friendly)
        if ctrl_held && dragging_primary {
            let delta = response.drag_delta();
            // Vertical drag = zoom
            let zoom_factor = 1.0 + delta.y * 0.01;
            self.view.zoom = (self.view.zoom * zoom_factor).clamp(0.1, 10.0);
        }

        // Scroll for zoom (with Ctrl) or pinch
        let scroll_delta = ctx.input(|i| i.raw_scroll_delta.y);
        if scroll_delta != 0.0 && ctrl_held {
            let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 0.9 };
            self.view.zoom = (self.view.zoom * zoom_factor).clamp(0.1, 10.0);
        }

        // Pinch to zoom (trackpad)
        let zoom_delta = ctx.input(|i| i.zoom_delta());
        if zoom_delta != 1.0 {
            self.view.zoom = (self.view.zoom * zoom_delta).clamp(0.1, 10.0);
        }

        // Note: Window preset keyboard shortcuts (0-9) are handled at the app level
        // to ensure they only affect the active viewport.
    }

    /// Show info overlay (WL, zoom, patient info, technical params)
    fn show_info_overlay(&self, ui: &mut egui::Ui, rect: Rect) {
        let image = match &self.image {
            Some(img) => img,
            None => return,
        };

        // Debug: log patient info once per image change (use static to avoid spam)
        static LAST_LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let img_hash =
            image.pixels.len() as u64 + image.width as u64 * 1000 + image.height as u64 * 1000000;
        let last = LAST_LOGGED.load(std::sync::atomic::Ordering::Relaxed);
        if last != img_hash {
            LAST_LOGGED.store(img_hash, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "Viewport overlay: patient_name={:?}, patient_id={:?}, study_instance_uid={:?}, image_plane={:?}",
                image.patient_name, image.patient_id, image.study_instance_uid, image.image_plane.is_some()
            );
        }

        let painter = ui.painter();
        let font = egui::FontId::monospace(12.0);

        // === TOP LEFT: WL and Zoom ===
        let mode = if self.use_gpu { "GPU" } else { "CPU" };
        let left_text = format!(
            "WC: {:.0} WW: {:.0}\nZoom: {:.0}% [{}]",
            self.windowing.window_center,
            self.windowing.window_width,
            self.view.zoom * 100.0,
            mode
        );
        self.draw_text_with_bg(
            painter,
            rect.left_top() + Vec2::new(10.0, 10.0),
            &left_text,
            &font,
            egui::Align2::LEFT_TOP,
        );

        // === TOP RIGHT: Patient/Study Info ===
        let mut right_lines = Vec::new();
        if let Some(name) = &image.patient_name {
            right_lines.push(name.replace('^', " "));
        }
        // ID, sex, age on same line
        let mut id_line = String::new();
        if let Some(id) = &image.patient_id {
            id_line.push_str(id);
        }
        if let Some(sex) = &image.patient_sex {
            if !id_line.is_empty() {
                id_line.push_str(", ");
            }
            id_line.push_str(sex);
        }
        if let Some(age) = &image.patient_age {
            if !id_line.is_empty() {
                id_line.push_str(", ");
            }
            // Parse age like "045Y" to "45y"
            let age_clean = age
                .trim_start_matches('0')
                .replace('Y', "y")
                .replace('M', "m")
                .replace('D', "d");
            id_line.push_str(&age_clean);
        }
        if !id_line.is_empty() {
            right_lines.push(id_line);
        }
        if let Some(desc) = &image.study_description {
            right_lines.push(desc.clone());
        }
        if let Some(date) = &image.study_date {
            // Format YYYYMMDD to YYYY-MM-DD
            if date.len() >= 8 {
                right_lines.push(format!("{}-{}-{}", &date[0..4], &date[4..6], &date[6..8]));
            }
        }
        if !right_lines.is_empty() {
            let right_text = right_lines.join("\n");
            self.draw_text_with_bg(
                painter,
                rect.right_top() + Vec2::new(-10.0, 10.0),
                &right_text,
                &font,
                egui::Align2::RIGHT_TOP,
            );
        }

        // === BOTTOM RIGHT: Technical Info (modality-specific) ===
        let modality = image.modality.as_deref().unwrap_or("");
        let mut tech_lines = Vec::new();

        match modality {
            "MR" => {
                if let Some(strength) = image.magnetic_field_strength {
                    tech_lines.push(format!("{:.1}T", strength));
                }
                let mut seq_line = String::new();
                if let Some(tr) = image.repetition_time {
                    seq_line.push_str(&format!("TR:{:.0}", tr));
                }
                if let Some(te) = image.echo_time {
                    if !seq_line.is_empty() {
                        seq_line.push(' ');
                    }
                    seq_line.push_str(&format!("TE:{:.0}", te));
                }
                if !seq_line.is_empty() {
                    tech_lines.push(seq_line);
                }
                if let Some(seq) = &image.sequence_name {
                    tech_lines.push(seq.clone());
                }
            }
            "CT" => {
                let mut ct_line = String::new();
                if let Some(kvp) = image.kvp {
                    ct_line.push_str(&format!("{:.0}kV", kvp));
                }
                if let Some(mas) = image.exposure {
                    if !ct_line.is_empty() {
                        ct_line.push(' ');
                    }
                    ct_line.push_str(&format!("{}mAs", mas));
                }
                if !ct_line.is_empty() {
                    tech_lines.push(ct_line);
                }
            }
            "CR" | "DX" | "XR" => {
                let mut xr_line = String::new();
                if let Some(kvp) = image.kvp {
                    xr_line.push_str(&format!("{:.0}kV", kvp));
                }
                if let Some(mas) = image.exposure {
                    if !xr_line.is_empty() {
                        xr_line.push(' ');
                    }
                    xr_line.push_str(&format!("{}mAs", mas));
                }
                if !xr_line.is_empty() {
                    tech_lines.push(xr_line);
                }
            }
            _ => {}
        }

        // Slice thickness for cross-sectional imaging
        if let Some(thickness) = image.slice_thickness {
            tech_lines.push(format!("{:.1}mm", thickness));
        }

        if !tech_lines.is_empty() {
            let tech_text = tech_lines.join("\n");
            self.draw_text_with_bg(
                painter,
                rect.right_bottom() + Vec2::new(-10.0, -10.0),
                &tech_text,
                &font,
                egui::Align2::RIGHT_BOTTOM,
            );
        }
    }

    /// Helper to draw text with semi-transparent background
    fn draw_text_with_bg(
        &self,
        painter: &egui::Painter,
        pos: Pos2,
        text: &str,
        font: &egui::FontId,
        align: egui::Align2,
    ) {
        let galley = painter.layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE);
        let text_size = galley.size();

        // Calculate background rect based on alignment
        let bg_pos = match align {
            egui::Align2::LEFT_TOP => pos,
            egui::Align2::RIGHT_TOP => pos - Vec2::new(text_size.x + 8.0, 0.0),
            egui::Align2::RIGHT_BOTTOM => pos - Vec2::new(text_size.x + 8.0, text_size.y + 8.0),
            egui::Align2::LEFT_BOTTOM => pos - Vec2::new(0.0, text_size.y + 8.0),
            _ => pos,
        };
        let bg_rect = Rect::from_min_size(bg_pos, text_size + Vec2::splat(8.0));
        painter.rect_filled(bg_rect, 4.0, Color32::from_black_alpha(180));

        painter.text(
            bg_pos + Vec2::new(4.0, 4.0),
            egui::Align2::LEFT_TOP,
            text,
            font.clone(),
            Color32::WHITE,
        );
    }

    /// Check if viewport has an image
    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// Get the last known viewport size (for fit to window from keyboard)
    pub fn last_size(&self) -> Vec2 {
        self.last_size
    }

    /// Get the image plane for reference line calculations
    pub fn image_plane(&self) -> Option<&ImagePlane> {
        self.image.as_ref().and_then(|img| img.image_plane.as_ref())
    }

    /// Show the viewport with reference lines (with optional fusion state)
    pub fn show_with_reference_lines_and_fusion(
        &mut self,
        ui: &mut egui::Ui,
        ref_lines: &[ReferenceLine],
        mode: InteractionMode,
        fusion: Option<&FusionState>,
    ) -> Option<DroppedSeries> {
        // Delegate to main implementation with fusion
        self.show_with_reference_lines_impl(ui, ref_lines, mode, fusion)
    }

    /// Show the viewport with reference lines
    pub fn show_with_reference_lines(
        &mut self,
        ui: &mut egui::Ui,
        ref_lines: &[ReferenceLine],
        mode: InteractionMode,
    ) -> Option<DroppedSeries> {
        self.show_with_reference_lines_impl(ui, ref_lines, mode, None)
    }

    /// Internal implementation for showing viewport with optional fusion
    fn show_with_reference_lines_impl(
        &mut self,
        ui: &mut egui::Ui,
        ref_lines: &[ReferenceLine],
        mode: InteractionMode,
        fusion: Option<&FusionState>,
    ) -> Option<DroppedSeries> {
        let available_size = ui.available_size();

        // Re-fit when viewport size changes (e.g. sidebar open/close)
        if self.image.is_some()
            && !self.needs_fit
            && !self.needs_smart_fit
            && ((available_size.x - self.last_size.x).abs() > 1.0
                || (available_size.y - self.last_size.y).abs() > 1.0)
        {
            self.smart_fit_to_window(available_size);
        }

        self.last_size = available_size;

        // Fit to window if needed (after we know the actual size)
        if self.needs_fit && self.image.is_some() {
            self.fit_to_window(available_size);
            self.needs_fit = false;
        }

        // Smart fit (anatomy bounds + smart windowing) if needed
        if self.needs_smart_fit && self.image.is_some() {
            // Smart zoom: fit to anatomy bounds
            self.smart_fit_to_window(available_size);

            // Smart window: for non-CT images, calculate optimal window
            // Extract info before borrowing self mutably
            let smart_window_info = self.image.as_ref().and_then(|image| {
                if !image.is_ct() {
                    image
                        .find_anatomy_bounds()
                        .map(|(x_min, y_min, x_max, y_max)| {
                            image.calculate_content_window(x_min, y_min, x_max, y_max)
                        })
                } else {
                    None
                }
            });

            if let Some((wc, ww)) = smart_window_info {
                self.windowing.window_center = wc;
                self.windowing.window_width = ww;
                self.needs_update = true;
            }

            self.needs_smart_fit = false;
        }

        // Create the viewport area
        let (response, painter) = ui.allocate_painter(available_size, Sense::click_and_drag());
        let rect = response.rect;
        self.last_rect = rect;

        // Check for dropped series
        let dropped_series = response
            .dnd_release_payload::<SeriesDragPayload>()
            .map(|payload| DroppedSeries {
                files: payload.files.clone(),
                label: payload.label.clone(),
                sync_info: payload.sync_info.clone(),
            });

        // Show drop highlight when dragging over
        if response.dnd_hover_payload::<SeriesDragPayload>().is_some() {
            painter.rect_stroke(rect, 4.0, egui::Stroke::new(3.0, Color32::LIGHT_BLUE));
        }

        // Draw background
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        // Render image — fusion mode uses the fusion pipeline if active
        let fusion_active = fusion.map(|f: &FusionState| f.is_active()).unwrap_or(false);
        if fusion_active && self.use_gpu && self.image.is_some() {
            // GPU fusion rendering path (two textures blended)
            self.render_fusion(ui, rect, fusion.unwrap());
        } else if self.use_gpu && self.image.is_some() {
            // GPU rendering path (single texture)
            self.render_gpu(ui, rect);
        } else {
            // CPU fallback path
            self.update_texture_cpu(ui.ctx());
            self.render_cpu(&painter, rect);
        }

        // Draw reference lines on top of the image
        if !ref_lines.is_empty() {
            self.draw_reference_lines(&painter, rect, ref_lines);
        }

        // Handle interactions
        self.handle_input(ui, &response, mode);

        // Show fusion status overlay
        if let Some(f) = fusion {
            if f.is_active() {
                let fusion_text = f.status_text();
                let font = egui::FontId::monospace(12.0);
                self.draw_text_with_bg(
                    &painter,
                    rect.left_bottom() + egui::Vec2::new(10.0, -10.0),
                    &fusion_text,
                    &font,
                    egui::Align2::LEFT_BOTTOM,
                );
            }
        }

        // Show info overlay
        self.show_info_overlay(ui, rect);

        dropped_series
    }

    /// Draw reference lines on the viewport
    fn draw_reference_lines(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        ref_lines: &[ReferenceLine],
    ) {
        let Some(image) = self.image.as_ref() else {
            return;
        };
        let img_size = Vec2::new(image.width as f32, image.height as f32);
        let scaled_size = img_size * self.view.zoom;

        // Calculate image position in viewport
        let center = rect.center() + self.view.pan;
        let img_left = center.x - scaled_size.x / 2.0;
        let img_top = center.y - scaled_size.y / 2.0;

        for ref_line in ref_lines {
            // Convert pixel coordinates to screen coordinates
            let start_screen = Pos2::new(
                img_left + ref_line.start.x as f32 * self.view.zoom,
                img_top + ref_line.start.y as f32 * self.view.zoom,
            );
            let end_screen = Pos2::new(
                img_left + ref_line.end.x as f32 * self.view.zoom,
                img_top + ref_line.end.y as f32 * self.view.zoom,
            );

            // Draw the reference line
            let color = Color32::from_rgb(ref_line.color.0, ref_line.color.1, ref_line.color.2);
            painter.line_segment([start_screen, end_screen], egui::Stroke::new(2.0, color));
        }
    }

    /// Draw a length measurement annotation
    pub fn draw_length_measurement(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        start: (f64, f64),
        end: (f64, f64),
        label: &str,
    ) {
        let start_screen = self.image_to_screen(start.0, start.1, rect);
        let end_screen = self.image_to_screen(end.0, end.1, rect);

        // Draw line
        let color = Color32::YELLOW;
        painter.line_segment([start_screen, end_screen], egui::Stroke::new(2.0, color));

        // Draw endpoints (small circles)
        painter.circle_filled(start_screen, 4.0, color);
        painter.circle_filled(end_screen, 4.0, color);

        // Draw label at midpoint
        let midpoint = Pos2::new(
            (start_screen.x + end_screen.x) / 2.0,
            (start_screen.y + end_screen.y) / 2.0 - 12.0,
        );

        // Background for text
        let font = egui::FontId::monospace(12.0);
        let galley = painter.layout_no_wrap(label.to_string(), font.clone(), color);
        let text_rect = Rect::from_min_size(
            midpoint - Vec2::new(galley.size().x / 2.0 + 4.0, galley.size().y / 2.0 + 2.0),
            galley.size() + Vec2::new(8.0, 4.0),
        );
        painter.rect_filled(text_rect, 4.0, Color32::from_black_alpha(180));
        painter.galley(
            midpoint - Vec2::new(galley.size().x / 2.0, galley.size().y / 2.0),
            galley,
            color,
        );
    }

    /// Draw a circular ROI annotation
    pub fn draw_circle_roi(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        center: (f64, f64),
        radius_px: f64,
        stats_label: Option<&str>,
    ) {
        let center_screen = self.image_to_screen(center.0, center.1, rect);
        let radius_screen = radius_px as f32 * self.view.zoom;

        // Draw circle
        let color = Color32::LIGHT_BLUE;
        painter.circle_stroke(center_screen, radius_screen, egui::Stroke::new(2.0, color));

        // Draw center crosshairs
        let cross_size = 6.0;
        painter.line_segment(
            [
                center_screen - Vec2::new(cross_size, 0.0),
                center_screen + Vec2::new(cross_size, 0.0),
            ],
            egui::Stroke::new(1.5, color),
        );
        painter.line_segment(
            [
                center_screen - Vec2::new(0.0, cross_size),
                center_screen + Vec2::new(0.0, cross_size),
            ],
            egui::Stroke::new(1.5, color),
        );

        // Draw stats label if provided
        if let Some(label) = stats_label {
            let label_pos = Pos2::new(center_screen.x, center_screen.y + radius_screen + 15.0);
            let font = egui::FontId::monospace(11.0);
            let galley = painter.layout_no_wrap(label.to_string(), font.clone(), color);

            // Background for text
            let text_rect = Rect::from_min_size(
                label_pos - Vec2::new(galley.size().x / 2.0 + 4.0, 2.0),
                galley.size() + Vec2::new(8.0, 4.0),
            );
            painter.rect_filled(text_rect, 4.0, Color32::from_black_alpha(180));
            painter.galley(
                label_pos - Vec2::new(galley.size().x / 2.0, 0.0),
                galley,
                color,
            );
        }
    }

    /// Draw a measurement preview (rubber-band while drawing)
    pub fn draw_measurement_preview(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        start: (f64, f64),
        current_screen_pos: Pos2,
        is_circle: bool,
    ) {
        let start_screen = self.image_to_screen(start.0, start.1, rect);

        if is_circle {
            // Circle preview
            let radius = start_screen.distance(current_screen_pos);
            let color = Color32::LIGHT_BLUE.gamma_multiply(0.7);
            painter.circle_stroke(start_screen, radius, egui::Stroke::new(1.5, color));
        } else {
            // Line preview
            let color = Color32::YELLOW.gamma_multiply(0.7);
            painter.line_segment(
                [start_screen, current_screen_pos],
                egui::Stroke::new(1.5, color),
            );

            // Draw endpoint markers
            painter.circle_filled(start_screen, 3.0, color);
            painter.circle_stroke(current_screen_pos, 3.0, egui::Stroke::new(1.0, color));
        }
    }
}

/// Calculate optimal window from pixel data using histogram analysis
fn calculate_optimal_window(pixels: &[u16], slope: f64, intercept: f64) -> (f64, f64) {
    if pixels.is_empty() {
        return (0.0, 1.0);
    }

    // Convert to actual values
    let values: Vec<f64> = pixels
        .iter()
        .map(|&p| (p as f64) * slope + intercept)
        .collect();

    // Calculate percentiles for robust windowing
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    let p2 = sorted[len * 2 / 100]; // 2nd percentile
    let p98 = sorted[len * 98 / 100]; // 98th percentile

    let window_width = (p98 - p2).max(1.0);
    let window_center = (p2 + p98) / 2.0;

    (window_center, window_width)
}
