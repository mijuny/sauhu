//! PACS query parameters and result types

use std::fmt;

/// Query level for C-FIND
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryLevel {
    Patient,
    Study,
    Series,
    Image,
}

impl fmt::Display for QueryLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryLevel::Patient => write!(f, "PATIENT"),
            QueryLevel::Study => write!(f, "STUDY"),
            QueryLevel::Series => write!(f, "SERIES"),
            QueryLevel::Image => write!(f, "IMAGE"),
        }
    }
}

/// Query parameters for C-FIND
#[derive(Debug, Clone, Default)]
pub struct QueryParams {
    /// Patient name (wildcard supported: * for any, ? for single char)
    pub patient_name: Option<String>,
    /// Patient ID
    pub patient_id: Option<String>,
    /// Accession number
    pub accession_number: Option<String>,
    /// Study date (YYYYMMDD or range YYYYMMDD-YYYYMMDD)
    pub study_date: Option<String>,
    /// Modality (CT, MR, CR, etc.)
    pub modality: Option<String>,
    /// Study description
    pub study_description: Option<String>,
}

impl QueryParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_patient_name(mut self, name: &str) -> Self {
        self.patient_name = Some(name.to_string());
        self
    }

    pub fn with_patient_id(mut self, id: &str) -> Self {
        self.patient_id = Some(id.to_string());
        self
    }

    pub fn with_accession(mut self, acc: &str) -> Self {
        self.accession_number = Some(acc.to_string());
        self
    }

    pub fn with_date(mut self, date: &str) -> Self {
        self.study_date = Some(date.to_string());
        self
    }

    pub fn with_modality(mut self, modality: &str) -> Self {
        self.modality = Some(modality.to_string());
        self
    }

    /// Check if any search criteria are specified
    pub fn is_empty(&self) -> bool {
        self.patient_name.is_none()
            && self.patient_id.is_none()
            && self.accession_number.is_none()
            && self.study_date.is_none()
            && self.modality.is_none()
            && self.study_description.is_none()
    }
}

/// Study result from C-FIND
#[derive(Debug, Clone)]
pub struct StudyResult {
    /// Study Instance UID (required for C-MOVE)
    pub study_instance_uid: String,
    /// Patient name
    pub patient_name: String,
    /// Patient ID
    pub patient_id: String,
    /// Accession number
    pub accession_number: String,
    /// Study date (YYYYMMDD)
    pub study_date: String,
    /// Study time (HHMMSS)
    pub study_time: String,
    /// Study description
    pub study_description: String,
    /// Modalities in study (may be multiple)
    pub modalities: String,
    /// Number of series
    pub num_series: Option<i32>,
    /// Number of instances
    pub num_instances: Option<i32>,
}

impl StudyResult {
    /// Format study date for display (YYYY-MM-DD)
    pub fn formatted_date(&self) -> String {
        if self.study_date.len() == 8 {
            format!(
                "{}-{}-{}",
                &self.study_date[0..4],
                &self.study_date[4..6],
                &self.study_date[6..8]
            )
        } else {
            self.study_date.clone()
        }
    }
}

/// Series result from C-FIND (when querying at series level)
#[derive(Debug, Clone)]
pub struct SeriesResult {
    /// Series Instance UID
    pub series_instance_uid: String,
    /// Parent Study Instance UID
    pub study_instance_uid: String,
    /// Series number
    pub series_number: Option<i32>,
    /// Series description
    pub series_description: String,
    /// Modality
    pub modality: String,
    /// Body part examined
    pub body_part: Option<String>,
    /// Number of instances in series
    pub num_instances: Option<i32>,
}

/// C-MOVE/retrieve progress
#[derive(Debug, Clone)]
pub struct RetrieveProgress {
    /// Number of instances remaining
    pub remaining: u32,
    /// Number of instances completed
    pub completed: u32,
    /// Number of failures
    pub failed: u32,
    /// Number of warnings
    pub warnings: u32,
    /// Whether the operation is complete
    pub is_complete: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl RetrieveProgress {
    pub fn new() -> Self {
        Self {
            remaining: 0,
            completed: 0,
            failed: 0,
            warnings: 0,
            is_complete: false,
            error: None,
        }
    }

    /// Total number of instances expected
    pub fn total(&self) -> u32 {
        self.remaining + self.completed + self.failed
    }

    /// Progress as percentage (0-100)
    pub fn percent(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            (self.completed as f32 / total as f32) * 100.0
        }
    }
}

impl Default for RetrieveProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Event sent when a series retrieval completes successfully
#[derive(Debug, Clone)]
pub struct SeriesCompleteEvent {
    /// Series Instance UID
    pub series_uid: String,
    /// Series number (for ordering)
    pub series_number: Option<i32>,
    /// Series description
    pub series_description: String,
    /// Modality (CT, MR, etc.)
    pub modality: String,
    /// Number of images actually received
    pub num_images: u32,
    /// Path where series files are stored
    pub storage_path: std::path::PathBuf,
}

/// Aggregated progress for parallel series retrieval
#[derive(Debug, Clone)]
pub struct AggregateProgress {
    /// Total number of series in the study
    pub total_series: u32,
    /// Number of series completed
    pub completed_series: u32,
    /// Total images across all series
    pub total_images: u32,
    /// Completed images across all series
    pub completed_images: u32,
    /// Failed images across all series
    pub failed_images: u32,
    /// Number of active worker threads
    pub active_workers: u32,
    /// Whether the entire operation is complete
    pub is_complete: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Series that just completed (for incremental updates)
    pub newly_completed_series: Option<SeriesCompleteEvent>,
}

impl AggregateProgress {
    pub fn new() -> Self {
        Self {
            total_series: 0,
            completed_series: 0,
            total_images: 0,
            completed_images: 0,
            failed_images: 0,
            active_workers: 0,
            is_complete: false,
            error: None,
            newly_completed_series: None,
        }
    }

    /// Progress as percentage (0-100)
    pub fn percent(&self) -> f32 {
        if self.total_images == 0 {
            0.0
        } else {
            (self.completed_images as f32 / self.total_images as f32) * 100.0
        }
    }
}

impl Default for AggregateProgress {
    fn default() -> Self {
        Self::new()
    }
}
