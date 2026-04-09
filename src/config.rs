//! Configuration file management
//!
//! Settings are stored in `~/.config/sauhu/config.toml`

use crate::hanging_protocol::HangingProtocol;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// PACS server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacsServer {
    /// Display name
    pub name: String,
    /// Remote AE Title
    pub ae_title: String,
    /// Hostname or IP
    pub host: String,
    /// Port number
    pub port: u16,
    /// Whether this is the default PACS for quick fetch
    #[serde(default)]
    pub default: bool,
}

impl Default for PacsServer {
    fn default() -> Self {
        Self {
            name: "PACS".to_string(),
            ae_title: "PACS".to_string(),
            host: "localhost".to_string(),
            port: 104,
            default: true,
        }
    }
}

/// Local DICOM listener configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Our AE Title
    pub ae_title: String,
    /// Port to listen on for incoming DICOM
    pub port: u16,
    /// Storage path for retrieved DICOM files
    pub storage_path: String,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            ae_title: "SAUHU".to_string(),
            port: 11112,
            storage_path: "~/.cache/sauhu/pacs".to_string(),
        }
    }
}

/// Viewer preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewerConfig {
    /// Default window center for CT (if no DICOM value)
    pub default_window_center: f64,
    /// Default window width for CT (if no DICOM value)
    pub default_window_width: f64,
    /// Scroll sensitivity in pixels per image (lower = faster scrolling)
    #[serde(default = "default_scroll_sensitivity")]
    pub scroll_sensitivity: f32,
    /// Start viewing at middle slice instead of first slice
    #[serde(default = "default_start_at_middle")]
    pub start_at_middle_slice: bool,
}

fn default_scroll_sensitivity() -> f32 {
    15.0
}
fn default_start_at_middle() -> bool {
    true
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            default_window_center: 40.0,
            default_window_width: 400.0,
            scroll_sensitivity: default_scroll_sensitivity(),
            start_at_middle_slice: default_start_at_middle(),
        }
    }
}

/// Keyboard shortcut configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortcutConfig {
    /// Length measurement mode
    #[serde(default = "default_length_measure")]
    pub length_measure: String,
    /// Circle ROI mode
    #[serde(default = "default_circle_roi")]
    pub circle_roi: String,
    /// Toggle synchronized scrolling
    #[serde(default = "default_sync_toggle")]
    pub sync_toggle: String,
    /// Scroll mode (right-click drag)
    #[serde(default = "default_scroll_mode")]
    pub scroll_mode: String,
    /// Windowing mode
    #[serde(default = "default_windowing_mode")]
    pub windowing_mode: String,
    /// ROI-based windowing mode
    #[serde(default = "default_roi_windowing")]
    pub roi_windowing: String,
    /// Smart fit to anatomy
    #[serde(default = "default_smart_fit")]
    pub smart_fit: String,
    /// Regular fit to window
    #[serde(default = "default_regular_fit")]
    pub regular_fit: String,
    /// Delete all annotations on current image
    #[serde(default = "default_delete_annotations")]
    pub delete_annotations: String,
    /// Smart auto-window (non-CT only)
    #[serde(default = "default_smart_window")]
    pub smart_window: String,
    /// MPR: Switch to axial plane
    #[serde(default = "default_mpr_axial")]
    pub mpr_axial: String,
    /// MPR: Switch to coronal plane
    #[serde(default = "default_mpr_coronal")]
    pub mpr_coronal: String,
    /// MPR: Switch to sagittal plane
    #[serde(default = "default_mpr_sagittal")]
    pub mpr_sagittal: String,
    /// MPR: Return to original plane
    #[serde(default = "default_mpr_original")]
    pub mpr_original: String,
}

fn default_length_measure() -> String {
    "L".to_string()
}
fn default_circle_roi() -> String {
    "O".to_string()
}
fn default_sync_toggle() -> String {
    "Y".to_string()
} // Changed from S to Y for MPR
fn default_scroll_mode() -> String {
    "Shift+S".to_string()
}
fn default_windowing_mode() -> String {
    "W".to_string()
}
fn default_roi_windowing() -> String {
    "Shift+W".to_string()
}
fn default_smart_fit() -> String {
    "Space".to_string()
}
fn default_regular_fit() -> String {
    "Shift+Space".to_string()
}
fn default_delete_annotations() -> String {
    "Shift+D".to_string()
}
fn default_smart_window() -> String {
    "Shift+A".to_string()
}
fn default_mpr_axial() -> String {
    "A".to_string()
}
fn default_mpr_coronal() -> String {
    "C".to_string()
}
fn default_mpr_sagittal() -> String {
    "S".to_string()
}
fn default_mpr_original() -> String {
    "X".to_string()
}

impl Default for ShortcutConfig {
    fn default() -> Self {
        Self {
            length_measure: default_length_measure(),
            circle_roi: default_circle_roi(),
            sync_toggle: default_sync_toggle(),
            scroll_mode: default_scroll_mode(),
            windowing_mode: default_windowing_mode(),
            roi_windowing: default_roi_windowing(),
            smart_fit: default_smart_fit(),
            regular_fit: default_regular_fit(),
            delete_annotations: default_delete_annotations(),
            smart_window: default_smart_window(),
            mpr_axial: default_mpr_axial(),
            mpr_coronal: default_mpr_coronal(),
            mpr_sagittal: default_mpr_sagittal(),
            mpr_original: default_mpr_original(),
        }
    }
}

impl ShortcutConfig {
    /// Check if a key event matches a shortcut
    pub fn matches(&self, shortcut: &str, key: egui::Key, modifiers: &egui::Modifiers) -> bool {
        parse_shortcut(shortcut).is_some_and(|(expected_key, shift, ctrl, alt)| {
            key == expected_key
                && modifiers.shift == shift
                && modifiers.ctrl == ctrl
                && modifiers.alt == alt
        })
    }
}

/// Parse a shortcut string like "Shift+S" or "L" into (Key, shift, ctrl, alt)
fn parse_shortcut(shortcut: &str) -> Option<(egui::Key, bool, bool, bool)> {
    let parts: Vec<&str> = shortcut.split('+').collect();
    let mut shift = false;
    let mut ctrl = false;
    let mut alt = false;
    let mut key_str = "";

    for part in &parts {
        match part.to_lowercase().as_str() {
            "shift" => shift = true,
            "ctrl" | "control" => ctrl = true,
            "alt" => alt = true,
            _ => key_str = part,
        }
    }

    let key = match key_str.to_uppercase().as_str() {
        "A" => egui::Key::A,
        "B" => egui::Key::B,
        "C" => egui::Key::C,
        "D" => egui::Key::D,
        "E" => egui::Key::E,
        "F" => egui::Key::F,
        "G" => egui::Key::G,
        "H" => egui::Key::H,
        "I" => egui::Key::I,
        "J" => egui::Key::J,
        "K" => egui::Key::K,
        "L" => egui::Key::L,
        "M" => egui::Key::M,
        "N" => egui::Key::N,
        "O" => egui::Key::O,
        "P" => egui::Key::P,
        "Q" => egui::Key::Q,
        "R" => egui::Key::R,
        "S" => egui::Key::S,
        "T" => egui::Key::T,
        "U" => egui::Key::U,
        "V" => egui::Key::V,
        "W" => egui::Key::W,
        "X" => egui::Key::X,
        "Y" => egui::Key::Y,
        "Z" => egui::Key::Z,
        "SPACE" => egui::Key::Space,
        "ESCAPE" | "ESC" => egui::Key::Escape,
        "ENTER" | "RETURN" => egui::Key::Enter,
        _ => return None,
    };

    Some((key, shift, ctrl, alt))
}

/// Viewport layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportConfig {
    /// Layout shortcuts: key number -> layout name
    /// Layouts: "single", "horizontal_2", "horizontal_3", "grid_2x2",
    ///          "grid_2x2_plus_1", "grid_3x2", "grid_4x2"
    #[serde(default = "default_layout_shortcuts")]
    pub layout_shortcuts: HashMap<String, String>,
}

fn default_layout_shortcuts() -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("1".to_string(), "single".to_string());
    map.insert("2".to_string(), "horizontal_2".to_string());
    map.insert("3".to_string(), "horizontal_3".to_string());
    map.insert("4".to_string(), "grid_2x2".to_string());
    map.insert("5".to_string(), "grid_2x2_plus_1".to_string());
    map.insert("6".to_string(), "grid_3x2".to_string());
    map.insert("8".to_string(), "grid_4x2".to_string());
    map
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            layout_shortcuts: default_layout_shortcuts(),
        }
    }
}

impl ViewportConfig {
    /// Get layout name for a key (1-9)
    pub fn get_layout_for_key(&self, key: &str) -> Option<&str> {
        self.layout_shortcuts.get(key).map(|s| s.as_str())
    }
}

/// Image cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache memory in GB (default: 32)
    #[serde(default = "default_cache_max_gb")]
    pub max_memory_gb: u32,
    /// Number of images to prefetch ahead of current
    #[serde(default = "default_prefetch_ahead")]
    pub prefetch_ahead: usize,
    /// Number of images to keep cached behind current
    #[serde(default = "default_prefetch_behind")]
    pub prefetch_behind: usize,
}

fn default_cache_max_gb() -> u32 {
    48
}
fn default_prefetch_ahead() -> usize {
    15
}
fn default_prefetch_behind() -> usize {
    5
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_gb: default_cache_max_gb(),
            prefetch_ahead: default_prefetch_ahead(),
            prefetch_behind: default_prefetch_behind(),
        }
    }
}

/// Main application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Viewer preferences
    #[serde(default)]
    pub viewer: ViewerConfig,
    /// Viewport layout configuration
    #[serde(default)]
    pub viewport: ViewportConfig,
    /// Keyboard shortcuts
    #[serde(default)]
    pub shortcuts: ShortcutConfig,
    /// Local DICOM listener config
    #[serde(default)]
    pub local: LocalConfig,
    /// PACS servers (keyed by identifier)
    #[serde(default)]
    pub pacs: PacsSettings,
    /// Image cache configuration
    #[serde(default)]
    pub cache: CacheConfig,
    /// Hanging protocols for automatic layout and series assignment
    #[serde(default)]
    pub hanging_protocol: Vec<HangingProtocol>,
}

/// PACS settings container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacsSettings {
    /// Map of server id -> server config
    #[serde(default)]
    pub servers: HashMap<String, PacsServer>,
    /// Number of parallel workers for series retrieval (default: 4)
    #[serde(default = "default_parallel_workers")]
    pub parallel_workers: usize,
}

fn default_parallel_workers() -> usize {
    4
}

impl Default for PacsSettings {
    fn default() -> Self {
        Self {
            servers: HashMap::new(),
            parallel_workers: default_parallel_workers(),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        let mut pacs = PacsSettings::default();
        // Add a default PACS server as example
        pacs.servers
            .insert("default".to_string(), PacsServer::default());

        Self {
            viewer: ViewerConfig::default(),
            viewport: ViewportConfig::default(),
            shortcuts: ShortcutConfig::default(),
            local: LocalConfig::default(),
            pacs,
            cache: CacheConfig::default(),
            hanging_protocol: Vec::new(),
        }
    }
}

impl Settings {
    /// Get the config file path
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = directories::ProjectDirs::from("", "", "sauhu")
            .context("Could not determine config directory")?
            .config_dir()
            .to_path_buf();
        Ok(config_dir.join("config.toml"))
    }

    /// Load settings from config file, or create default if not exists
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;

        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("Failed to read config file: {:?}", path))?;

            match toml::from_str::<Settings>(&content) {
                Ok(settings) => {
                    // Validate the loaded settings
                    if let Err(warnings) = settings.validate() {
                        for warning in warnings {
                            tracing::warn!("Config warning: {}", warning);
                        }
                    }
                    tracing::info!("Loaded config from {:?}", path);
                    Ok(settings)
                }
                Err(e) => {
                    // Show helpful error with line number if available
                    let error_msg = format!(
                        "Failed to parse config file {:?}:\n  {}\n\nPlease fix the config file or delete it to reset to defaults.",
                        path, e
                    );
                    Err(anyhow::anyhow!(error_msg))
                }
            }
        } else {
            tracing::info!("Config file not found, creating default at {:?}", path);
            let settings = Settings::default();
            settings.save()?;
            Ok(settings)
        }
    }

    /// Validate settings and return any warnings
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut warnings = Vec::new();

        // Check local listener port
        if self.local.port < 1024 {
            warnings.push(format!(
                "Local port {} requires root privileges. Consider using a port >= 1024.",
                self.local.port
            ));
        }

        // Check PACS servers
        if self.pacs.servers.is_empty() {
            warnings.push("No PACS servers configured.".to_string());
        }

        for (id, server) in &self.pacs.servers {
            if server.ae_title.is_empty() {
                warnings.push(format!("PACS server '{}' has empty AE title.", id));
            }
            if server.host.is_empty() {
                warnings.push(format!("PACS server '{}' has empty host.", id));
            }
            if server.port == 0 {
                warnings.push(format!("PACS server '{}' has invalid port 0.", id));
            }
        }

        if warnings.is_empty() {
            Ok(())
        } else {
            Err(warnings)
        }
    }

    /// Save settings to config file
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let content = toml::to_string_pretty(self).context("Failed to serialize settings")?;

        // Add header comment
        let content_with_header = format!(
            "# Sauhu DICOM Viewer Configuration\n\
             # Edit this file to configure PACS servers and viewer settings.\n\
             # Changes take effect on next application start.\n\n{}",
            content
        );

        fs::write(&path, content_with_header)
            .with_context(|| format!("Failed to write config file: {:?}", path))?;

        tracing::info!("Saved config to {:?}", path);
        Ok(())
    }

    /// Get expanded storage path (resolve ~)
    pub fn storage_path(&self) -> PathBuf {
        let path = &self.local.storage_path;
        if let Some(stripped) = path.strip_prefix("~/") {
            if let Some(home) = directories::BaseDirs::new() {
                return home.home_dir().join(stripped);
            }
        }
        PathBuf::from(path)
    }

    /// Get list of PACS servers
    pub fn pacs_servers(&self) -> Vec<(&str, &PacsServer)> {
        self.pacs
            .servers
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// Get a specific PACS server by id
    pub fn get_pacs_server(&self, id: &str) -> Option<&PacsServer> {
        self.pacs.servers.get(id)
    }

    /// Get the default PACS server (first one marked as default, or first one if none marked)
    pub fn get_default_pacs_server(&self) -> Option<(&str, &PacsServer)> {
        // First try to find one marked as default
        if let Some((id, server)) = self.pacs.servers.iter().find(|(_, s)| s.default) {
            return Some((id.as_str(), server));
        }
        // Fall back to first server if none marked as default
        self.pacs
            .servers
            .iter()
            .next()
            .map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.local.ae_title, "SAUHU");
        assert_eq!(settings.local.port, 11112);
        assert!(settings.pacs.servers.contains_key("default"));
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        let parsed: Settings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.local.ae_title, settings.local.ae_title);
    }
}
