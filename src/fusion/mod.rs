//! Image Fusion Module
//!
//! Enables overlay of two co-registered series in a single viewport.
//! Supports alpha blending (Phase 1), color overlay (Phase 2), and checkerboard (Phase 3).
#![allow(dead_code)]

pub mod colormap;

/// Blend mode for image fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Linear alpha blend between base and overlay
    #[default]
    Alpha,
    /// Overlay displayed with a colormap (e.g. hot, rainbow)
    ColorOverlay,
    /// Checkerboard pattern for registration QA
    Checkerboard,
}

impl BlendMode {
    /// Display name for UI
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Alpha => "Alpha Blend",
            Self::ColorOverlay => "Color Overlay",
            Self::Checkerboard => "Checkerboard",
        }
    }

    /// GPU blend_mode index for shader uniform
    pub fn shader_index(&self) -> u32 {
        match self {
            Self::Alpha => 0,
            Self::ColorOverlay => 1,
            Self::Checkerboard => 2,
        }
    }

    /// Cycle to next blend mode: Alpha → ColorOverlay → Checkerboard → Alpha
    pub fn next(&self) -> Self {
        match self {
            Self::Alpha => Self::ColorOverlay,
            Self::ColorOverlay => Self::Checkerboard,
            Self::Checkerboard => Self::Alpha,
        }
    }
}

/// Colormap type for Color Overlay mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapType {
    /// Hot colormap: black → red → yellow → white (PET standard)
    #[default]
    Hot,
    /// Rainbow: blue → cyan → green → yellow → red
    Rainbow,
    /// Cool-Warm diverging: blue → white → red
    CoolWarm,
    /// Grayscale (default)
    Grayscale,
}

impl ColormapType {
    /// Display name for UI
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Hot => "Hot",
            Self::Rainbow => "Rainbow",
            Self::CoolWarm => "Cool-Warm",
            Self::Grayscale => "Grayscale",
        }
    }

    /// Shader index for uniform (must match fusion.wgsl apply_colormap)
    pub fn shader_index(&self) -> u32 {
        match self {
            Self::Grayscale => 0,
            Self::Hot => 1,
            Self::Rainbow => 2,
            Self::CoolWarm => 3,
        }
    }

    /// Cycle to next colormap: Hot → Rainbow → CoolWarm → Grayscale → Hot
    pub fn next(&self) -> Self {
        match self {
            Self::Hot => Self::Rainbow,
            Self::Rainbow => Self::CoolWarm,
            Self::CoolWarm => Self::Grayscale,
            Self::Grayscale => Self::Hot,
        }
    }
}

/// State for image fusion in a single viewport
#[derive(Debug, Clone)]
pub struct FusionState {
    /// Whether fusion is currently active
    pub enabled: bool,
    /// Current blend mode
    pub blend_mode: BlendMode,
    /// Overlay opacity (0.0 = only base, 1.0 = only overlay)
    pub opacity: f32,
    /// Overlay threshold (0.0–1.0): below this windowed intensity, overlay is transparent
    pub threshold: f32,
    /// Current colormap for Color Overlay mode
    pub colormap: ColormapType,
    /// Window center for base image
    pub base_window_center: f32,
    /// Window width for base image
    pub base_window_width: f32,
    /// Rescale slope for base image
    pub base_rescale_slope: f32,
    /// Rescale intercept for base image
    pub base_rescale_intercept: f32,
    /// Pixel representation for base image (0=unsigned, 1=signed)
    pub base_pixel_representation: u32,
    /// Window center for overlay image
    pub overlay_window_center: f32,
    /// Window width for overlay image
    pub overlay_window_width: f32,
    /// Rescale slope for overlay image
    pub overlay_rescale_slope: f32,
    /// Rescale intercept for overlay image
    pub overlay_rescale_intercept: f32,
    /// Pixel representation for overlay image (0=unsigned, 1=signed)
    pub overlay_pixel_representation: u32,
    /// Colormap index for shader (derived from colormap field)
    pub colormap_index: u32,
    /// Checkerboard square size in pixels (for Checkerboard mode)
    pub checker_size: f32,
    /// Source viewport ID that provides overlay images
    pub overlay_source_viewport: Option<usize>,
}

impl Default for FusionState {
    fn default() -> Self {
        let colormap = ColormapType::default();
        Self {
            enabled: false,
            blend_mode: BlendMode::Alpha,
            opacity: 0.5,
            threshold: 0.0,
            colormap,
            base_window_center: 40.0,
            base_window_width: 400.0,
            base_rescale_slope: 1.0,
            base_rescale_intercept: 0.0,
            base_pixel_representation: 0,
            overlay_window_center: 40.0,
            overlay_window_width: 400.0,
            overlay_rescale_slope: 1.0,
            overlay_rescale_intercept: 0.0,
            overlay_pixel_representation: 0,
            colormap_index: colormap.shader_index(),
            checker_size: 32.0,
            overlay_source_viewport: None,
        }
    }
}

impl FusionState {
    /// Create new fusion state
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle fusion on/off
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    /// Adjust opacity by delta (clamped to 0.0..=1.0)
    pub fn adjust_opacity(&mut self, delta: f32) {
        self.opacity = (self.opacity + delta).clamp(0.0, 1.0);
    }

    /// Adjust threshold by delta (clamped to 0.0..=1.0)
    pub fn adjust_threshold(&mut self, delta: f32) {
        self.threshold = (self.threshold + delta).clamp(0.0, 1.0);
    }

    /// Reset threshold to 0
    pub fn reset_threshold(&mut self) {
        self.threshold = 0.0;
    }

    /// Cycle blend mode: Alpha → ColorOverlay → Checkerboard → Alpha
    pub fn cycle_blend_mode(&mut self) {
        self.blend_mode = self.blend_mode.next();
    }

    /// Cycle colormap: Hot → Rainbow → CoolWarm → Grayscale → Hot
    pub fn cycle_colormap(&mut self) {
        self.colormap = self.colormap.next();
        self.colormap_index = self.colormap.shader_index();
    }

    /// Set base image windowing parameters
    pub fn set_base_window(&mut self, center: f32, width: f32) {
        self.base_window_center = center;
        self.base_window_width = width.max(1.0);
    }

    /// Set overlay image windowing parameters
    pub fn set_overlay_window(&mut self, center: f32, width: f32) {
        self.overlay_window_center = center;
        self.overlay_window_width = width.max(1.0);
    }

    /// Set base image rescale parameters
    pub fn set_base_rescale(&mut self, slope: f32, intercept: f32, pixel_rep: u32) {
        self.base_rescale_slope = slope;
        self.base_rescale_intercept = intercept;
        self.base_pixel_representation = pixel_rep;
    }

    /// Set overlay image rescale parameters
    pub fn set_overlay_rescale(&mut self, slope: f32, intercept: f32, pixel_rep: u32) {
        self.overlay_rescale_slope = slope;
        self.overlay_rescale_intercept = intercept;
        self.overlay_pixel_representation = pixel_rep;
    }

    /// Check if fusion is active and has an overlay source
    pub fn is_active(&self) -> bool {
        self.enabled && self.overlay_source_viewport.is_some()
    }

    /// Status text for UI overlay
    pub fn status_text(&self) -> String {
        if !self.enabled {
            return String::new();
        }
        match self.blend_mode {
            BlendMode::Alpha => {
                format!(
                    "FUSION: {} {:.0}%",
                    self.blend_mode.display_name(),
                    self.opacity * 100.0
                )
            }
            BlendMode::ColorOverlay => {
                let threshold_str = if self.threshold > 0.0 {
                    format!(" T:{:.0}%", self.threshold * 100.0)
                } else {
                    String::new()
                };
                format!(
                    "FUSION: {} {} {:.0}%{}",
                    self.blend_mode.display_name(),
                    self.colormap.display_name(),
                    self.opacity * 100.0,
                    threshold_str
                )
            }
            BlendMode::Checkerboard => {
                format!(
                    "FUSION: {} {:.0}px",
                    self.blend_mode.display_name(),
                    self.checker_size
                )
            }
        }
    }
}

/// Manager for coordinating fusion across viewports
pub struct FusionManager;

impl FusionManager {
    /// Activate fusion on a target viewport using overlay from a source viewport.
    /// Call this after coregistration completes.
    pub fn activate_fusion(
        fusion_state: &mut FusionState,
        source_viewport_id: usize,
    ) {
        fusion_state.enabled = true;
        fusion_state.blend_mode = BlendMode::Alpha;
        fusion_state.opacity = 0.5;
        fusion_state.threshold = 0.0;
        fusion_state.overlay_source_viewport = Some(source_viewport_id);
        tracing::info!(
            "Fusion activated: overlay from viewport {}",
            source_viewport_id
        );
    }

    /// Deactivate fusion on a viewport
    pub fn deactivate_fusion(fusion_state: &mut FusionState) {
        fusion_state.enabled = false;
        fusion_state.overlay_source_viewport = None;
        tracing::info!("Fusion deactivated");
    }
}
