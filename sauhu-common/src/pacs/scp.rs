//! DICOM Service Class Provider (SCP) implementation
//!
//! Provides C-STORE SCP for receiving images from C-MOVE operations.

use anyhow::{anyhow, Result};
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::dictionary_std::tags;
use dicom::object::{mem::InMemDicomObject, StandardDataDictionary};
use dicom_ul::association::server::ServerAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

/// Transfer syntax UIDs
const IMPLICIT_VR_LE: &str = "1.2.840.10008.1.2";
const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";

/// Progress information for C-STORE operations
#[derive(Debug, Clone, Default)]
pub struct StoreProgress {
    pub received: u32,
    pub failed: u32,
    pub is_complete: bool,
    pub error: Option<String>,
}

impl StoreProgress {
    pub fn new() -> Self {
        Self {
            received: 0,
            failed: 0,
            is_complete: false,
            error: None,
        }
    }
}

/// SCP server for receiving DICOM images
pub struct DicomScp {
    ae_title: String,
    port: u16,
    cache_dir: PathBuf,
    running: Arc<AtomicBool>,
    received_count: Arc<AtomicU32>,
    failed_count: Arc<AtomicU32>,
}

impl DicomScp {
    /// Create a new SCP server
    pub fn new(ae_title: &str, port: u16, cache_dir: PathBuf) -> Self {
        Self {
            ae_title: ae_title.to_string(),
            port,
            cache_dir,
            running: Arc::new(AtomicBool::new(false)),
            received_count: Arc::new(AtomicU32::new(0)),
            failed_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Get the port this SCP is configured for
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Check if the SCP is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get current progress
    pub fn progress(&self) -> StoreProgress {
        StoreProgress {
            received: self.received_count.load(Ordering::SeqCst),
            failed: self.failed_count.load(Ordering::SeqCst),
            is_complete: !self.running.load(Ordering::SeqCst),
            error: None,
        }
    }

    /// Start the SCP server in a background thread
    /// Returns a channel that will receive progress updates
    pub fn start(&self) -> Result<mpsc::Receiver<StoreProgress>> {
        if self.running.load(Ordering::SeqCst) {
            return Err(anyhow!("SCP already running"));
        }

        let (tx, rx) = mpsc::channel();

        // Ensure cache directory exists
        std::fs::create_dir_all(&self.cache_dir)?;

        let ae_title = self.ae_title.clone();
        let port = self.port;
        let cache_dir = self.cache_dir.clone();
        let running = self.running.clone();
        let received_count = self.received_count.clone();
        let failed_count = self.failed_count.clone();

        running.store(true, Ordering::SeqCst);
        received_count.store(0, Ordering::SeqCst);
        failed_count.store(0, Ordering::SeqCst);

        thread::spawn(move || {
            if let Err(e) = run_scp(
                &ae_title,
                port,
                &cache_dir,
                running.clone(),
                received_count.clone(),
                failed_count.clone(),
                tx.clone(),
            ) {
                tracing::error!("SCP error: {}", e);
                let mut progress = StoreProgress::new();
                progress.error = Some(e.to_string());
                progress.is_complete = true;
                let _ = tx.send(progress);
            }
            running.store(false, Ordering::SeqCst);
        });

        Ok(rx)
    }

    /// Stop the SCP server
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

fn run_scp(
    ae_title: &str,
    port: u16,
    cache_dir: &std::path::Path,
    running: Arc<AtomicBool>,
    received_count: Arc<AtomicU32>,
    failed_count: Arc<AtomicU32>,
    progress_tx: mpsc::Sender<StoreProgress>,
) -> Result<()> {
    let addr = format!("0.0.0.0:{}", port);
    let listener =
        TcpListener::bind(&addr).map_err(|e| anyhow!("Failed to bind to {}: {}", addr, e))?;

    // Set non-blocking so we can check the running flag
    listener.set_nonblocking(true)?;

    tracing::info!("DICOM SCP listening on {} (AE: {})", addr, ae_title);

    while running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, peer_addr)) => {
                tracing::info!("Incoming connection from {}", peer_addr);

                // Set stream back to blocking for the association
                stream.set_nonblocking(false)?;

                // Accept any association (promiscuous mode)
                let options = ServerAssociationOptions::new()
                    .ae_title(ae_title)
                    .promiscuous(true);

                match options.establish(stream) {
                    Ok(mut association) => {
                        tracing::debug!("Association established");

                        // Handle C-STORE requests
                        if let Err(e) = handle_association(
                            &mut association,
                            cache_dir,
                            &received_count,
                            &failed_count,
                            &progress_tx,
                        ) {
                            tracing::error!("Association error: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to accept association: {}", e);
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No connection pending, sleep briefly and check running flag
                thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                tracing::error!("Accept error: {}", e);
            }
        }
    }

    tracing::info!("DICOM SCP stopped");

    // Send final progress
    let mut progress = StoreProgress::new();
    progress.received = received_count.load(Ordering::SeqCst);
    progress.failed = failed_count.load(Ordering::SeqCst);
    progress.is_complete = true;
    let _ = progress_tx.send(progress);

    Ok(())
}

fn handle_association(
    association: &mut dicom_ul::association::server::ServerAssociation<TcpStream>,
    cache_dir: &std::path::Path,
    received_count: &AtomicU32,
    failed_count: &AtomicU32,
    progress_tx: &mpsc::Sender<StoreProgress>,
) -> Result<()> {
    loop {
        let pdu = match association.receive() {
            Ok(pdu) => pdu,
            Err(e) => {
                tracing::debug!("Receive ended: {}", e);
                break;
            }
        };

        match pdu {
            Pdu::PData { data } => {
                // Collect all data for this message
                let mut command_data = Vec::new();
                let mut dataset_data = Vec::new();
                let mut pc_id = 0;
                let mut is_command_complete = false;

                for pdv in &data {
                    pc_id = pdv.presentation_context_id;
                    match pdv.value_type {
                        PDataValueType::Command => {
                            command_data.extend_from_slice(&pdv.data);
                            if pdv.is_last {
                                is_command_complete = true;
                            }
                        }
                        PDataValueType::Data => {
                            dataset_data.extend_from_slice(&pdv.data);
                        }
                    }
                }

                // If we have a complete command, process it
                if is_command_complete && !command_data.is_empty() {
                    if let Ok(cmd) = decode_dataset(&command_data) {
                        if let Ok(cmd_field) = cmd.element(tags::COMMAND_FIELD) {
                            if let Ok(field) = cmd_field.to_int::<u16>() {
                                if field == 0x0001 {
                                    // C-STORE-RQ
                                    let msg_id = cmd
                                        .element(tags::MESSAGE_ID)
                                        .ok()
                                        .and_then(|e| e.to_int::<u16>().ok())
                                        .unwrap_or(1);

                                    let affected_sop = cmd
                                        .element(tags::AFFECTED_SOP_CLASS_UID)
                                        .ok()
                                        .and_then(|e| e.to_str().ok())
                                        .map(|s| s.to_string())
                                        .unwrap_or_default();

                                    let sop_instance = cmd
                                        .element(tags::AFFECTED_SOP_INSTANCE_UID)
                                        .ok()
                                        .and_then(|e| e.to_str().ok())
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| format!("unknown_{}", msg_id));

                                    tracing::debug!("Receiving C-STORE for {}", sop_instance);

                                    // Continue receiving data until complete
                                    let mut all_data = dataset_data;
                                    loop {
                                        match association.receive() {
                                            Ok(Pdu::PData { data }) => {
                                                let mut is_last = false;
                                                for pdv in data {
                                                    if pdv.value_type == PDataValueType::Data {
                                                        all_data.extend_from_slice(&pdv.data);
                                                        is_last = pdv.is_last;
                                                    }
                                                }
                                                if is_last {
                                                    break;
                                                }
                                            }
                                            Ok(Pdu::ReleaseRQ) => {
                                                tracing::debug!(
                                                    "Received ReleaseRQ during data transfer"
                                                );
                                                let _ = association.send(&Pdu::ReleaseRP);
                                                return Ok(());
                                            }
                                            Ok(_) => continue,
                                            Err(e) => {
                                                tracing::warn!("Error receiving data: {}", e);
                                                break;
                                            }
                                        }
                                    }

                                    // Get the transfer syntax for this presentation context
                                    let ts = association
                                        .presentation_contexts()
                                        .iter()
                                        .find(|pc| pc.id == pc_id)
                                        .map(|pc| pc.transfer_syntax.as_str())
                                        .unwrap_or(EXPLICIT_VR_LE);

                                    // Save the DICOM file
                                    let status = match save_dicom_file(
                                        &all_data,
                                        ts,
                                        &affected_sop,
                                        &sop_instance,
                                        cache_dir,
                                    ) {
                                        Ok(_) => {
                                            received_count.fetch_add(1, Ordering::SeqCst);
                                            let progress = StoreProgress {
                                                received: received_count.load(Ordering::SeqCst),
                                                failed: failed_count.load(Ordering::SeqCst),
                                                is_complete: false,
                                                error: None,
                                            };
                                            let _ = progress_tx.send(progress);
                                            tracing::info!("Saved: {}", sop_instance);
                                            0x0000 // Success
                                        }
                                        Err(e) => {
                                            failed_count.fetch_add(1, Ordering::SeqCst);
                                            tracing::warn!(
                                                "Failed to save {}: {}",
                                                sop_instance,
                                                e
                                            );
                                            0xA700 // Out of resources
                                        }
                                    };

                                    // Send C-STORE-RSP
                                    let rsp = build_c_store_rsp(msg_id, &affected_sop, status);
                                    let rsp_bytes = encode_command(&rsp)?;

                                    let rsp_pdu = Pdu::PData {
                                        data: vec![PDataValue {
                                            presentation_context_id: pc_id,
                                            value_type: PDataValueType::Command,
                                            is_last: true,
                                            data: rsp_bytes,
                                        }],
                                    };
                                    if let Err(e) = association.send(&rsp_pdu) {
                                        tracing::warn!("Failed to send C-STORE-RSP: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Pdu::ReleaseRQ => {
                tracing::debug!("Received ReleaseRQ");
                let _ = association.send(&Pdu::ReleaseRP);
                break;
            }
            Pdu::AbortRQ { source } => {
                tracing::warn!("Received AbortRQ from {:?}", source);
                break;
            }
            _ => {
                tracing::debug!("Unexpected PDU");
            }
        }
    }

    Ok(())
}

fn save_dicom_file(
    data: &[u8],
    ts_uid: &str,
    _sop_class: &str,
    sop_instance: &str,
    cache_dir: &std::path::Path,
) -> Result<()> {
    use std::io::Cursor;

    let (ts, ts_erased) = match ts_uid {
        EXPLICIT_VR_LE => (
            &dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN,
            dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased(),
        ),
        IMPLICIT_VR_LE => (
            &dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN,
            dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
        ),
        _ => {
            tracing::warn!("Unknown transfer syntax {}, using Explicit VR LE", ts_uid);
            (
                &dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN,
                dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased(),
            )
        }
    };

    let cursor = Cursor::new(data);
    let ds = InMemDicomObject::read_dataset_with_ts(cursor, &ts_erased)?;

    // Get SOP Class UID from dataset or use default
    let sop_class_uid = ds
        .element(tags::SOP_CLASS_UID)
        .ok()
        .and_then(|e| e.to_str().ok())
        .unwrap_or("1.2.840.10008.5.1.4.1.1.2".into())
        .to_string();

    // Get Study Instance UID for organizing files into subdirectories
    let study_uid = ds
        .element(tags::STUDY_INSTANCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown_study".to_string());

    // Create study subdirectory (replace dots with underscores for filesystem)
    let study_dir = cache_dir.join(study_uid.replace('.', "_"));
    std::fs::create_dir_all(&study_dir)?;

    // Create file object by wrapping the parsed dataset with meta info
    let file_obj = ds.with_meta(
        dicom::object::FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(&sop_class_uid)
            .media_storage_sop_instance_uid(sop_instance)
            .transfer_syntax(ts.uid()),
    )?;

    // Generate filename and save in study subdirectory
    let filename = format!("{}.dcm", sop_instance.replace('.', "_"));
    let file_path = study_dir.join(&filename);

    file_obj.write_to_file(&file_path)?;
    tracing::debug!("Saved DICOM file: {:?}", file_path);
    tracing::info!("Stored: {} ({} bytes)", sop_instance, data.len());

    Ok(())
}

fn build_c_store_rsp(
    msg_id: u16,
    sop_class: &str,
    status: u16,
) -> InMemDicomObject<StandardDataDictionary> {
    let mut cmd = InMemDicomObject::new_empty();
    cmd.put(DataElement::new(
        tags::AFFECTED_SOP_CLASS_UID,
        VR::UI,
        PrimitiveValue::from(sop_class),
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_FIELD,
        VR::US,
        PrimitiveValue::from(0x8001u16), // C-STORE-RSP
    ));
    cmd.put(DataElement::new(
        tags::MESSAGE_ID_BEING_RESPONDED_TO,
        VR::US,
        PrimitiveValue::from(msg_id),
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_DATA_SET_TYPE,
        VR::US,
        PrimitiveValue::from(0x0101u16), // No data set
    ));
    cmd.put(DataElement::new(
        tags::STATUS,
        VR::US,
        PrimitiveValue::from(status),
    ));
    cmd
}

fn decode_dataset(data: &[u8]) -> Result<InMemDicomObject<StandardDataDictionary>> {
    use std::io::Cursor;
    let cursor = Cursor::new(data);
    let ds = InMemDicomObject::read_dataset_with_ts(
        cursor,
        &dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
    )?;
    Ok(ds)
}

fn encode_command(ds: &InMemDicomObject<StandardDataDictionary>) -> Result<Vec<u8>> {
    // First, encode the dataset without group length
    let mut data_buf = Vec::new();
    ds.write_dataset_with_ts(
        &mut data_buf,
        &dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
    )?;

    // Calculate length and prepend CommandGroupLength (0000,0000)
    let data_len = data_buf.len() as u32;
    let mut buf = Vec::with_capacity(12 + data_buf.len());

    // Tag (0000,0000) = CommandGroupLength
    buf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    // Length of value (4 bytes for UL)
    buf.extend_from_slice(&4u32.to_le_bytes());
    // Value: length of remaining command data
    buf.extend_from_slice(&data_len.to_le_bytes());
    // Append the actual command data
    buf.extend_from_slice(&data_buf);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_progress() {
        let progress = StoreProgress::new();
        assert_eq!(progress.received, 0);
        assert_eq!(progress.failed, 0);
        assert!(!progress.is_complete);
        assert!(progress.error.is_none());
    }
}
