// DB schema provides full CRUD API; not all functions are used yet
#![allow(dead_code)]

//! Database schema for Sauhu

use anyhow::Result;
use rusqlite::Connection;

/// Current schema version
const SCHEMA_VERSION: i32 = 2;

/// Initialize the database schema
pub fn init(conn: &Connection) -> Result<()> {
    // Create schema version table
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );
        "#,
    )?;

    // Get current version
    let current_version: i32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Run migrations
    if current_version < 1 {
        migrate_v1(conn)?;
    }
    if current_version < 2 {
        migrate_v2(conn)?;
    }

    // Update version
    if current_version < SCHEMA_VERSION {
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?1)",
            [SCHEMA_VERSION],
        )?;
    }

    tracing::debug!("Database schema initialized (version {})", SCHEMA_VERSION);
    Ok(())
}

/// Migration v1: Initial schema
fn migrate_v1(conn: &Connection) -> Result<()> {
    tracing::info!("Running database migration v1");
    conn.execute_batch(
        r#"
        -- Studies table
        CREATE TABLE IF NOT EXISTS studies (
            id INTEGER PRIMARY KEY,
            study_instance_uid TEXT UNIQUE NOT NULL,
            patient_name TEXT,
            patient_id TEXT,
            study_date TEXT,
            study_description TEXT,
            modality TEXT,
            file_path TEXT NOT NULL,
            imported_at TEXT DEFAULT (datetime('now'))
        );

        -- Series table
        CREATE TABLE IF NOT EXISTS series (
            id INTEGER PRIMARY KEY,
            study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
            series_instance_uid TEXT UNIQUE NOT NULL,
            series_number INTEGER,
            series_description TEXT,
            modality TEXT,
            num_images INTEGER DEFAULT 0
        );

        -- Measurements table
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY,
            series_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
            image_index INTEGER NOT NULL,
            tool_type TEXT NOT NULL,
            coordinates TEXT NOT NULL,
            value REAL,
            unit TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_studies_patient_id ON studies(patient_id);
        CREATE INDEX IF NOT EXISTS idx_studies_study_date ON studies(study_date);
        CREATE INDEX IF NOT EXISTS idx_series_study_id ON series(study_id);
        CREATE INDEX IF NOT EXISTS idx_measurements_series_id ON measurements(series_id);
        "#,
    )?;
    Ok(())
}

/// Migration v2: PACS support
fn migrate_v2(conn: &Connection) -> Result<()> {
    tracing::info!("Running database migration v2 (PACS support)");
    conn.execute_batch(
        r#"
        -- PACS server configuration
        CREATE TABLE IF NOT EXISTS pacs_servers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            ae_title TEXT NOT NULL,
            host TEXT NOT NULL,
            port INTEGER NOT NULL DEFAULT 4242,
            our_ae_title TEXT NOT NULL DEFAULT 'SAUHU',
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Add new columns to studies (SQLite requires separate ALTER statements)
        "#,
    )?;

    // Add columns if they don't exist (SQLite doesn't have IF NOT EXISTS for columns)
    add_column_if_missing(conn, "studies", "source", "TEXT DEFAULT 'local'")?;
    add_column_if_missing(conn, "studies", "accession_number", "TEXT")?;
    add_column_if_missing(conn, "studies", "last_accessed", "TEXT")?;

    // Add columns to series
    add_column_if_missing(conn, "series", "body_part", "TEXT")?;
    add_column_if_missing(conn, "series", "num_frames", "INTEGER DEFAULT 0")?;

    // Create index for accession number
    conn.execute_batch(
        r#"
        CREATE INDEX IF NOT EXISTS idx_studies_accession ON studies(accession_number);
        CREATE INDEX IF NOT EXISTS idx_studies_source ON studies(source);
        "#,
    )?;

    Ok(())
}

/// Add a column to a table if it doesn't already exist
fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    // Check if column exists
    let sql = format!("PRAGMA table_info({})", table);
    let mut stmt = conn.prepare(&sql)?;
    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .collect();

    if !columns.contains(&column.to_string()) {
        let alter_sql = format!("ALTER TABLE {} ADD COLUMN {} {}", table, column, definition);
        conn.execute(&alter_sql, [])?;
        tracing::debug!("Added column {}.{}", table, column);
    }
    Ok(())
}

/// PACS server configuration
#[derive(Debug, Clone)]
pub struct PacsServer {
    pub id: i64,
    pub name: String,
    pub ae_title: String,
    pub host: String,
    pub port: i32,
    pub our_ae_title: String,
}

impl PacsServer {
    /// Create a new PACS server config (not yet saved to DB)
    pub fn new(name: String, ae_title: String, host: String, port: i32) -> Self {
        Self {
            id: 0,
            name,
            ae_title,
            host,
            port,
            our_ae_title: "SAUHU".to_string(),
        }
    }

    /// Convert to the common PacsServerConfig used by PACS networking code
    pub fn to_config(&self) -> sauhu_common::PacsServerConfig {
        sauhu_common::PacsServerConfig {
            name: self.name.clone(),
            ae_title: self.ae_title.clone(),
            host: self.host.clone(),
            port: self.port,
            our_ae_title: self.our_ae_title.clone(),
        }
    }
}

/// Study source type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudySource {
    Local,
    Pacs,
}

impl std::fmt::Display for StudySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudySource::Local => write!(f, "local"),
            StudySource::Pacs => write!(f, "pacs"),
        }
    }
}

impl std::str::FromStr for StudySource {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" => Ok(StudySource::Local),
            "pacs" => Ok(StudySource::Pacs),
            _ => Err(()),
        }
    }
}

/// Study record
#[derive(Debug, Clone)]
pub struct Study {
    pub id: i64,
    pub study_instance_uid: String,
    pub patient_name: Option<String>,
    pub patient_id: Option<String>,
    pub study_date: Option<String>,
    pub study_description: Option<String>,
    pub modality: Option<String>,
    pub file_path: String,
    pub source: StudySource,
    pub accession_number: Option<String>,
    pub last_accessed: Option<String>,
}

/// Series record
#[derive(Debug, Clone)]
pub struct Series {
    pub id: i64,
    pub study_id: i64,
    pub series_instance_uid: String,
    pub series_number: Option<i32>,
    pub series_description: Option<String>,
    pub modality: Option<String>,
    pub num_images: i32,
    pub body_part: Option<String>,
    pub num_frames: i32,
}

/// Measurement record
#[derive(Debug, Clone)]
pub struct Measurement {
    pub id: i64,
    pub series_id: i64,
    pub image_index: i32,
    pub tool_type: String,
    pub coordinates: String,
    pub value: Option<f64>,
    pub unit: Option<String>,
}

// ============================================================================
// PACS Server CRUD Operations
// ============================================================================

/// Get all PACS servers
pub fn get_pacs_servers(conn: &Connection) -> Result<Vec<PacsServer>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, ae_title, host, port, our_ae_title FROM pacs_servers ORDER BY name",
    )?;

    let servers = stmt
        .query_map([], |row| {
            Ok(PacsServer {
                id: row.get(0)?,
                name: row.get(1)?,
                ae_title: row.get(2)?,
                host: row.get(3)?,
                port: row.get(4)?,
                our_ae_title: row.get(5)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(servers)
}

/// Get a PACS server by ID
pub fn get_pacs_server(conn: &Connection, id: i64) -> Result<Option<PacsServer>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, ae_title, host, port, our_ae_title FROM pacs_servers WHERE id = ?1",
    )?;

    let server = stmt
        .query_row([id], |row| {
            Ok(PacsServer {
                id: row.get(0)?,
                name: row.get(1)?,
                ae_title: row.get(2)?,
                host: row.get(3)?,
                port: row.get(4)?,
                our_ae_title: row.get(5)?,
            })
        })
        .ok();

    Ok(server)
}

/// Insert a new PACS server
pub fn insert_pacs_server(conn: &Connection, server: &PacsServer) -> Result<i64> {
    conn.execute(
        "INSERT INTO pacs_servers (name, ae_title, host, port, our_ae_title) VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![server.name, server.ae_title, server.host, server.port, server.our_ae_title],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Update a PACS server
pub fn update_pacs_server(conn: &Connection, server: &PacsServer) -> Result<()> {
    conn.execute(
        "UPDATE pacs_servers SET name = ?1, ae_title = ?2, host = ?3, port = ?4, our_ae_title = ?5 WHERE id = ?6",
        rusqlite::params![server.name, server.ae_title, server.host, server.port, server.our_ae_title, server.id],
    )?;
    Ok(())
}

/// Delete a PACS server
pub fn delete_pacs_server(conn: &Connection, id: i64) -> Result<()> {
    conn.execute("DELETE FROM pacs_servers WHERE id = ?1", [id])?;
    Ok(())
}

// ============================================================================
// Study CRUD Operations
// ============================================================================

/// Get all studies, ordered by most recent first
pub fn get_all_studies(conn: &Connection) -> Result<Vec<Study>> {
    let mut stmt = conn.prepare(
        r#"SELECT id, study_instance_uid, patient_name, patient_id, study_date,
                  study_description, modality, file_path, source, accession_number, last_accessed
           FROM studies
           ORDER BY COALESCE(last_accessed, imported_at) DESC"#,
    )?;

    let studies = stmt
        .query_map([], |row| {
            let source_str: String = row
                .get::<_, Option<String>>(8)?
                .unwrap_or_else(|| "local".to_string());
            Ok(Study {
                id: row.get(0)?,
                study_instance_uid: row.get(1)?,
                patient_name: row.get(2)?,
                patient_id: row.get(3)?,
                study_date: row.get(4)?,
                study_description: row.get(5)?,
                modality: row.get(6)?,
                file_path: row.get(7)?,
                source: source_str.parse().unwrap_or(StudySource::Local),
                accession_number: row.get(9)?,
                last_accessed: row.get(10)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(studies)
}

/// Study with aggregated image count from series
#[derive(Debug, Clone)]
pub struct StudyWithImages {
    pub study: Study,
    pub num_images: i32,
}

/// Get all studies with total image count, ordered by most recent first
pub fn get_all_studies_with_image_count(conn: &Connection) -> Result<Vec<StudyWithImages>> {
    let mut stmt = conn.prepare(
        r#"SELECT s.id, s.study_instance_uid, s.patient_name, s.patient_id, s.study_date,
                  s.study_description, s.modality, s.file_path, s.source, s.accession_number,
                  s.last_accessed, COALESCE(SUM(ser.num_images), 0) as total_images
           FROM studies s
           LEFT JOIN series ser ON ser.study_id = s.id
           GROUP BY s.id
           ORDER BY COALESCE(s.last_accessed, s.imported_at) DESC"#,
    )?;

    let studies = stmt
        .query_map([], |row| {
            let source_str: String = row
                .get::<_, Option<String>>(8)?
                .unwrap_or_else(|| "local".to_string());
            Ok(StudyWithImages {
                study: Study {
                    id: row.get(0)?,
                    study_instance_uid: row.get(1)?,
                    patient_name: row.get(2)?,
                    patient_id: row.get(3)?,
                    study_date: row.get(4)?,
                    study_description: row.get(5)?,
                    modality: row.get(6)?,
                    file_path: row.get(7)?,
                    source: source_str.parse().unwrap_or(StudySource::Local),
                    accession_number: row.get(9)?,
                    last_accessed: row.get(10)?,
                },
                num_images: row.get(11)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(studies)
}

/// Get a study by ID
pub fn get_study(conn: &Connection, id: i64) -> Result<Option<Study>> {
    let mut stmt = conn.prepare(
        r#"SELECT id, study_instance_uid, patient_name, patient_id, study_date,
                  study_description, modality, file_path, source, accession_number, last_accessed
           FROM studies WHERE id = ?1"#,
    )?;

    let study = stmt
        .query_row([id], |row| {
            let source_str: String = row
                .get::<_, Option<String>>(8)?
                .unwrap_or_else(|| "local".to_string());
            Ok(Study {
                id: row.get(0)?,
                study_instance_uid: row.get(1)?,
                patient_name: row.get(2)?,
                patient_id: row.get(3)?,
                study_date: row.get(4)?,
                study_description: row.get(5)?,
                modality: row.get(6)?,
                file_path: row.get(7)?,
                source: source_str.parse().unwrap_or(StudySource::Local),
                accession_number: row.get(9)?,
                last_accessed: row.get(10)?,
            })
        })
        .ok();

    Ok(study)
}

/// Insert a new study
pub fn insert_study(conn: &Connection, study: &Study) -> Result<i64> {
    conn.execute(
        r#"INSERT INTO studies (study_instance_uid, patient_name, patient_id, study_date,
                                study_description, modality, file_path, source, accession_number)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
        rusqlite::params![
            study.study_instance_uid,
            study.patient_name,
            study.patient_id,
            study.study_date,
            study.study_description,
            study.modality,
            study.file_path,
            study.source.to_string(),
            study.accession_number,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Update last accessed time for a study
pub fn touch_study(conn: &Connection, id: i64) -> Result<()> {
    conn.execute(
        "UPDATE studies SET last_accessed = datetime('now') WHERE id = ?1",
        [id],
    )?;
    Ok(())
}

/// Delete a study (cascades to series and measurements)
pub fn delete_study(conn: &Connection, id: i64) -> Result<()> {
    conn.execute("DELETE FROM studies WHERE id = ?1", [id])?;
    Ok(())
}

// ============================================================================
// Series CRUD Operations
// ============================================================================

/// Get all series for a study
pub fn get_series_for_study(conn: &Connection, study_id: i64) -> Result<Vec<Series>> {
    let mut stmt = conn.prepare(
        r#"SELECT id, study_id, series_instance_uid, series_number, series_description,
                  modality, num_images, body_part, num_frames
           FROM series
           WHERE study_id = ?1
           ORDER BY series_number"#,
    )?;

    let series = stmt
        .query_map([study_id], |row| {
            Ok(Series {
                id: row.get(0)?,
                study_id: row.get(1)?,
                series_instance_uid: row.get(2)?,
                series_number: row.get(3)?,
                series_description: row.get(4)?,
                modality: row.get(5)?,
                num_images: row.get::<_, Option<i32>>(6)?.unwrap_or(0),
                body_part: row.get(7)?,
                num_frames: row.get::<_, Option<i32>>(8)?.unwrap_or(0),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(series)
}

/// Insert a new series
pub fn insert_series(conn: &Connection, series: &Series) -> Result<i64> {
    conn.execute(
        r#"INSERT INTO series (study_id, series_instance_uid, series_number, series_description,
                               modality, num_images, body_part, num_frames)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
        rusqlite::params![
            series.study_id,
            series.series_instance_uid,
            series.series_number,
            series.series_description,
            series.modality,
            series.num_images,
            series.body_part,
            series.num_frames,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

// ============================================================================
// Incremental Retrieval Support
// ============================================================================

/// Study info from PACS query result (for incremental retrieval)
#[derive(Debug, Clone)]
pub struct RetrievalStudyInfo {
    pub study_instance_uid: String,
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
    pub modality: String,
    pub accession_number: String,
}

/// Get or create a study record for an ongoing retrieval
///
/// Returns the study ID. If the study already exists, returns its ID.
/// If it doesn't exist, creates a new record and returns the new ID.
pub fn upsert_study_for_retrieval(
    conn: &Connection,
    info: &RetrievalStudyInfo,
    file_path: &std::path::Path,
) -> Result<i64> {
    // First, try to find existing study
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM studies WHERE study_instance_uid = ?1",
            [&info.study_instance_uid],
            |row| row.get(0),
        )
        .ok();

    if let Some(id) = existing {
        tracing::debug!(
            "Study {} already exists with ID {}",
            info.study_instance_uid,
            id
        );
        return Ok(id);
    }

    // Create new study record
    conn.execute(
        r#"INSERT INTO studies (study_instance_uid, patient_name, patient_id, study_date,
                                study_description, modality, file_path, source, accession_number)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pacs', ?8)"#,
        rusqlite::params![
            info.study_instance_uid,
            info.patient_name,
            info.patient_id,
            info.study_date,
            info.study_description,
            info.modality,
            file_path.to_string_lossy(),
            info.accession_number,
        ],
    )?;

    let id = conn.last_insert_rowid();
    tracing::info!(
        "Created study record {} for {}",
        id,
        info.study_instance_uid
    );
    Ok(id)
}

/// Insert a series from a completion event (for incremental updates)
///
/// Uses INSERT OR REPLACE to handle duplicate series UIDs gracefully.
pub fn insert_series_from_event(
    conn: &Connection,
    study_id: i64,
    series_uid: &str,
    series_number: Option<i32>,
    series_description: &str,
    modality: &str,
    num_images: u32,
) -> Result<i64> {
    // Use INSERT OR REPLACE to handle re-retrieval of same series
    conn.execute(
        r#"INSERT OR REPLACE INTO series
           (study_id, series_instance_uid, series_number, series_description, modality, num_images, body_part, num_frames)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, 0)"#,
        rusqlite::params![
            study_id,
            series_uid,
            series_number,
            series_description,
            modality,
            num_images as i32,
        ],
    )?;

    let id = conn.last_insert_rowid();
    tracing::debug!(
        "Inserted series {} ({} images) for study {}",
        series_uid,
        num_images,
        study_id
    );
    Ok(id)
}

/// Get study ID by study instance UID
pub fn get_study_id_by_uid(conn: &Connection, study_uid: &str) -> Result<Option<i64>> {
    let id: Option<i64> = conn
        .query_row(
            "SELECT id FROM studies WHERE study_instance_uid = ?1",
            [study_uid],
            |row| row.get(0),
        )
        .ok();
    Ok(id)
}

// ============================================================================
// Anonymization Support
// ============================================================================

/// Update study record after anonymization
pub fn update_study_after_anonymization(
    conn: &Connection,
    study_id: i64,
    new_study_uid: &str,
    new_patient_name: &str,
    new_patient_id: &str,
) -> Result<()> {
    conn.execute(
        r#"UPDATE studies
           SET study_instance_uid = ?1,
               patient_name = ?2,
               patient_id = ?3,
               accession_number = NULL
           WHERE id = ?4"#,
        rusqlite::params![new_study_uid, new_patient_name, new_patient_id, study_id],
    )?;
    tracing::info!(
        "Anonymized study {}: {} -> {}",
        study_id,
        new_patient_name,
        new_patient_id
    );
    Ok(())
}

/// Update series UID after anonymization
pub fn update_series_uid(conn: &Connection, old_uid: &str, new_uid: &str) -> Result<()> {
    conn.execute(
        "UPDATE series SET series_instance_uid = ?1 WHERE series_instance_uid = ?2",
        [new_uid, old_uid],
    )?;
    Ok(())
}

/// Get all file paths for studies belonging to a patient
pub fn get_patient_file_paths(
    conn: &Connection,
    patient_id: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let mut stmt = conn.prepare("SELECT file_path FROM studies WHERE patient_id = ?1")?;

    let paths: Vec<std::path::PathBuf> = stmt
        .query_map([patient_id], |row| {
            let path_str: String = row.get(0)?;
            Ok(std::path::PathBuf::from(path_str))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(paths)
}

/// Get all studies for a patient by patient_id
pub fn get_studies_by_patient_id(conn: &Connection, patient_id: &str) -> Result<Vec<Study>> {
    let mut stmt = conn.prepare(
        r#"SELECT id, study_instance_uid, patient_name, patient_id, study_date,
                  study_description, modality, file_path, source, accession_number, last_accessed
           FROM studies
           WHERE patient_id = ?1
           ORDER BY study_date DESC"#,
    )?;

    let studies = stmt
        .query_map([patient_id], |row| {
            let source_str: String = row
                .get::<_, Option<String>>(8)?
                .unwrap_or_else(|| "local".to_string());
            Ok(Study {
                id: row.get(0)?,
                study_instance_uid: row.get(1)?,
                patient_name: row.get(2)?,
                patient_id: row.get(3)?,
                study_date: row.get(4)?,
                study_description: row.get(5)?,
                modality: row.get(6)?,
                file_path: row.get(7)?,
                source: source_str.parse().unwrap_or(StudySource::Local),
                accession_number: row.get(9)?,
                last_accessed: row.get(10)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(studies)
}

/// Get the next available patient number for anonymization
pub fn get_next_anon_patient_number(conn: &Connection, prefix: &str) -> Result<u32> {
    // Find highest existing number with this prefix
    let pattern = format!("{}%", prefix);
    let max_num: Option<i64> = conn
        .query_row(
            r#"SELECT MAX(CAST(SUBSTR(patient_id, ?1) AS INTEGER))
               FROM studies
               WHERE patient_id LIKE ?2"#,
            rusqlite::params![prefix.len() + 1, pattern],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    Ok(max_num.map(|n| n as u32 + 1).unwrap_or(1))
}

/// Get the next available accession number for anonymization
pub fn get_next_anon_accession_number(conn: &Connection, prefix: &str) -> Result<u32> {
    // Find highest existing number with this prefix
    let pattern = format!("{}%", prefix);
    let max_num: Option<i64> = conn
        .query_row(
            r#"SELECT MAX(CAST(SUBSTR(accession_number, ?1) AS INTEGER))
               FROM studies
               WHERE accession_number LIKE ?2"#,
            rusqlite::params![prefix.len() + 1, pattern],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    Ok(max_num.map(|n| n as u32 + 1).unwrap_or(1))
}

/// Update study accession number after anonymization
pub fn update_study_accession(conn: &Connection, study_id: i64, new_accession: &str) -> Result<()> {
    conn.execute(
        "UPDATE studies SET accession_number = ?1 WHERE id = ?2",
        rusqlite::params![new_accession, study_id],
    )?;
    Ok(())
}
