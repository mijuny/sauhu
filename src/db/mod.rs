//! Database module for Sauhu
//!
//! Handles SQLite database for storing studies, series, and measurements.

mod schema;

use anyhow::Result;
use directories::ProjectDirs;
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub use schema::*;

/// Database handle with thread-safe connection
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    pub path: PathBuf,
}

impl Database {
    /// Open or create the database
    pub fn open() -> Result<Self> {
        let path = Self::default_path()?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&path)?;

        // Initialize schema
        schema::init(&conn)?;

        tracing::debug!("Database opened at {:?}", path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path,
        })
    }

    /// Get the default database path
    fn default_path() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "sauhu", "Sauhu")
            .ok_or_else(|| anyhow::anyhow!("Could not determine project directories"))?;

        let data_dir = proj_dirs.data_dir();
        Ok(data_dir.join("sauhu.db"))
    }

    /// Get direct access to the connection (locks mutex)
    pub fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("Database mutex poisoned")
    }
}
