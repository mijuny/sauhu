//! Sauhu Common - Shared DICOM networking types and PACS modules
//!
//! Used by both the sauhu viewer and sauhu-palvelin relay server.

pub mod pacs;

/// PACS server connection configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PacsServerConfig {
    /// Display name
    pub name: String,
    /// Remote AE Title
    pub ae_title: String,
    /// Hostname or IP
    pub host: String,
    /// Port number
    pub port: i32,
    /// Our (calling) AE Title
    pub our_ae_title: String,
}

impl PacsServerConfig {
    pub fn new(
        name: String,
        ae_title: String,
        host: String,
        port: i32,
        our_ae_title: String,
    ) -> Self {
        Self {
            name,
            ae_title,
            host,
            port,
            our_ae_title,
        }
    }
}
