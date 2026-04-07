//! IPC Server for Sauhu
//!
//! Provides a Unix socket server for external applications (like Sanelu)
//! to communicate with Sauhu.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::thread;

/// Commands that can be sent to Sauhu via IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum IpcCommand {
    /// Open a study by accession number (queries PACS if needed)
    OpenStudy { accession: String },
    /// Check if Sauhu is running (ping)
    Ping,
}

/// Response from Sauhu
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IpcResponse {
    Ok { message: Option<String> },
    Error { message: String },
}

/// Get the IPC socket path
pub fn socket_path() -> PathBuf {
    // Use XDG_RUNTIME_DIR for proper Linux socket placement
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        PathBuf::from(runtime_dir).join("sauhu.sock")
    } else {
        // Fallback to /tmp
        PathBuf::from("/tmp").join(format!("sauhu-{}.sock", unsafe { libc::getuid() }))
    }
}

/// IPC Server that listens for commands
pub struct IpcServer {
    socket_path: PathBuf,
    command_tx: Sender<IpcCommand>,
}

impl IpcServer {
    pub fn new(command_tx: Sender<IpcCommand>) -> Self {
        Self {
            socket_path: socket_path(),
            command_tx,
        }
    }

    /// Start the IPC server in a background thread
    pub fn start(&self) -> std::io::Result<()> {
        let socket_path = self.socket_path.clone();
        let command_tx = self.command_tx.clone();

        // Remove existing socket file if present
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)?;
        }

        // Create the listener
        let listener = UnixListener::bind(&socket_path)?;
        tracing::info!("IPC server listening on {:?}", socket_path);

        // Spawn listener thread
        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let tx = command_tx.clone();
                        thread::spawn(move || {
                            if let Err(e) = handle_client(stream, tx) {
                                tracing::warn!("IPC client error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("IPC accept error: {}", e);
                    }
                }
            }
        });

        Ok(())
    }
}

fn handle_client(mut stream: UnixStream, command_tx: Sender<IpcCommand>) -> std::io::Result<()> {
    let reader = BufReader::new(stream.try_clone()?);

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        tracing::debug!("IPC received: {}", line);

        let response = match serde_json::from_str::<IpcCommand>(&line) {
            Ok(command) => {
                match &command {
                    IpcCommand::Ping => IpcResponse::Ok {
                        message: Some("pong".to_string()),
                    },
                    IpcCommand::OpenStudy { accession } => {
                        // Send command to main app
                        if command_tx.send(command.clone()).is_ok() {
                            IpcResponse::Ok {
                                message: Some(format!("Opening study: {}", accession)),
                            }
                        } else {
                            IpcResponse::Error {
                                message: "Failed to send command to app".to_string(),
                            }
                        }
                    }
                }
            }
            Err(e) => IpcResponse::Error {
                message: format!("Invalid command: {}", e),
            },
        };

        let response_json = serde_json::to_string(&response).unwrap_or_else(|_| {
            r#"{"status":"error","message":"Failed to serialize response"}"#.to_string()
        });

        writeln!(stream, "{}", response_json)?;
        stream.flush()?;
    }

    Ok(())
}

/// IPC Client for sending commands to Sauhu
pub struct IpcClient {
    socket_path: PathBuf,
}

impl IpcClient {
    pub fn new() -> Self {
        Self {
            socket_path: socket_path(),
        }
    }

    /// Check if Sauhu is running
    pub fn is_running(&self) -> bool {
        self.send_command(&IpcCommand::Ping).is_ok()
    }

    /// Send a command to Sauhu
    pub fn send_command(&self, command: &IpcCommand) -> Result<IpcResponse, String> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .map_err(|e| format!("Cannot connect to Sauhu: {}", e))?;

        let command_json = serde_json::to_string(command)
            .map_err(|e| format!("Failed to serialize command: {}", e))?;

        writeln!(stream, "{}", command_json)
            .map_err(|e| format!("Failed to send command: {}", e))?;
        stream
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;

        let mut reader = BufReader::new(stream);
        let mut response_line = String::new();
        reader
            .read_line(&mut response_line)
            .map_err(|e| format!("Failed to read response: {}", e))?;

        serde_json::from_str(&response_line).map_err(|e| format!("Failed to parse response: {}", e))
    }

    /// Open a study by accession number
    pub fn open_study(&self, accession: &str) -> Result<IpcResponse, String> {
        self.send_command(&IpcCommand::OpenStudy {
            accession: accession.to_string(),
        })
    }
}

impl Default for IpcClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_serialization() {
        let cmd = IpcCommand::OpenStudy {
            accession: "12345".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("open_study"));
        assert!(json.contains("12345"));
    }

    #[test]
    fn test_response_serialization() {
        let resp = IpcResponse::Ok {
            message: Some("test".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("ok"));
    }
}
