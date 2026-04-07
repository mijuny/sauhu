//! DICOM Service Class User (SCU) implementation
//!
//! Provides C-FIND and C-MOVE operations for PACS connectivity.
#![allow(dead_code)]

use crate::PacsServerConfig;
use crate::pacs::{QueryParams, RetrieveProgress, SeriesResult, StudyResult};
use anyhow::{anyhow, Result};
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::dictionary_std::tags;
use dicom::object::{mem::InMemDicomObject, StandardDataDictionary};
use dicom_ul::association::client::ClientAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

/// Transfer syntax UIDs
const IMPLICIT_VR_LE: &str = "1.2.840.10008.1.2";
const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";

/// Abstract syntax UIDs
const STUDY_ROOT_FIND: &str = "1.2.840.10008.5.1.4.1.2.2.1";
const STUDY_ROOT_MOVE: &str = "1.2.840.10008.5.1.4.1.2.2.2";
const VERIFICATION_SOP: &str = "1.2.840.10008.1.1";

/// DICOM SCU client
pub struct DicomScu {
    server: PacsServerConfig,
}

impl DicomScu {
    /// Create a new SCU for the given PACS server
    pub fn new(server: PacsServerConfig) -> Self {
        Self { server }
    }

    /// Get the address string for connection
    fn addr(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }

    /// Test connection to PACS (C-ECHO)
    pub fn echo(&self) -> Result<bool> {
        tracing::info!(
            "Testing connection to {} ({})",
            self.server.name,
            self.addr()
        );

        let options = ClientAssociationOptions::new()
            .calling_ae_title(&self.server.our_ae_title)
            .called_ae_title(&self.server.ae_title)
            .with_abstract_syntax(VERIFICATION_SOP);

        let mut association = options
            .establish(self.addr())
            .map_err(|e| anyhow!("Failed to establish association: {}", e))?;

        // Get the presentation context ID for verification
        let pc_id = association
            .presentation_contexts()
            .iter()
            .find(|pc| pc.abstract_syntax == VERIFICATION_SOP)
            .map(|pc| pc.id)
            .ok_or_else(|| anyhow!("Verification SOP not accepted"))?;

        // Build C-ECHO command
        let msg_id = 1u16;
        let cmd = build_c_echo_rq(msg_id);
        let cmd_bytes = encode_command(&cmd)?;

        // Send C-ECHO request
        let pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        };

        association
            .send(&pdu)
            .map_err(|e| anyhow!("Failed to send C-ECHO: {}", e))?;

        // Receive response
        let response = association
            .receive()
            .map_err(|e| anyhow!("Failed to receive C-ECHO response: {}", e))?;

        // Check response
        let success = match response {
            Pdu::PData { data } => {
                if let Some(pdv) = data.first() {
                    if pdv.value_type == PDataValueType::Command {
                        if let Ok(rsp) = decode_dataset(&pdv.data) {
                            get_status(&rsp).unwrap_or(0xFFFF) == 0x0000
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false,
        };

        // Release association
        association
            .release()
            .map_err(|e| anyhow!("Failed to release association: {}", e))?;

        if success {
            tracing::info!("C-ECHO successful");
        } else {
            tracing::warn!("C-ECHO failed");
        }

        Ok(success)
    }

    /// Perform C-FIND query for studies
    pub fn find_studies(&self, params: &QueryParams) -> Result<Vec<StudyResult>> {
        tracing::info!("C-FIND studies from {} ({})", self.server.name, self.addr());

        let options = ClientAssociationOptions::new()
            .calling_ae_title(&self.server.our_ae_title)
            .called_ae_title(&self.server.ae_title)
            .with_presentation_context(STUDY_ROOT_FIND, vec![EXPLICIT_VR_LE, IMPLICIT_VR_LE]);

        let mut association = options
            .establish(self.addr())
            .map_err(|e| anyhow!("Failed to establish association: {}", e))?;

        // Log all negotiated presentation contexts
        for pc in association.presentation_contexts() {
            tracing::debug!(
                "Presentation context {}: {} -> {}",
                pc.id,
                pc.abstract_syntax,
                pc.transfer_syntax
            );
        }

        let (pc_id, transfer_syntax) = {
            let pc = association
                .presentation_contexts()
                .iter()
                .find(|pc| pc.abstract_syntax == STUDY_ROOT_FIND)
                .ok_or_else(|| anyhow!("Study Root Query not accepted"))?;
            (pc.id, pc.transfer_syntax.clone())
        };

        tracing::info!(
            "Using presentation context {}, transfer syntax: {}",
            pc_id,
            transfer_syntax
        );

        // Build and send C-FIND request
        let msg_id = 1u16;
        let query_ds = build_study_query(params);
        let cmd = build_c_find_rq(msg_id, STUDY_ROOT_FIND);
        // Command is always Implicit VR LE
        let cmd_bytes = encode_command(&cmd)?;
        // Data uses negotiated transfer syntax
        let data_bytes = encode_dataset_with_ts(&query_ds, &transfer_syntax)?;

        tracing::debug!("Command bytes: {} bytes", cmd_bytes.len());
        tracing::debug!("Data bytes: {} bytes", data_bytes.len());

        // Send command PDU
        let cmd_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        };
        association
            .send(&cmd_pdu)
            .map_err(|e| anyhow!("Failed to send C-FIND command: {}", e))?;
        tracing::debug!("Command PDU sent");

        // Send data PDU (query)
        let data_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Data,
                is_last: true,
                data: data_bytes,
            }],
        };
        association
            .send(&data_pdu)
            .map_err(|e| anyhow!("Failed to send C-FIND query: {}", e))?;

        tracing::info!("C-FIND request sent, waiting for response...");

        // Clone transfer syntax for use in receive loop
        let ts_uid = transfer_syntax.clone();

        // Receive responses
        let mut results = Vec::new();
        loop {
            let pdu = association
                .receive()
                .map_err(|e| anyhow!("Failed to receive C-FIND response: {}", e))?;

            match pdu {
                Pdu::PData { data } => {
                    tracing::debug!("Received PData with {} value(s)", data.len());
                    for pdv in data {
                        tracing::debug!(
                            "  PDV: pc_id={}, type={:?}, is_last={}, {} bytes",
                            pdv.presentation_context_id,
                            pdv.value_type,
                            pdv.is_last,
                            pdv.data.len()
                        );
                        if pdv.value_type == PDataValueType::Command {
                            // Parse command to check status (commands are always Implicit VR LE)
                            if let Ok(cmd_ds) = decode_dataset(&pdv.data) {
                                let status = get_status(&cmd_ds).unwrap_or(0xFFFF);
                                tracing::info!("C-FIND status: 0x{:04X}", status);

                                if status == 0x0000 {
                                    // Success - no more results
                                    tracing::info!("C-FIND complete, {} results", results.len());
                                    association.release().ok();
                                    return Ok(results);
                                } else if status != 0xFF00 && status != 0xFF01 {
                                    // Error
                                    association.release().ok();
                                    return Err(anyhow!(
                                        "C-FIND failed with status: 0x{:04X}",
                                        status
                                    ));
                                }
                                // 0xFF00/0xFF01 = Pending, more results coming
                            }
                        } else if pdv.value_type == PDataValueType::Data {
                            // Parse result dataset with negotiated transfer syntax
                            if let Ok(ds) = decode_dataset_with_ts(&pdv.data, &ts_uid) {
                                if let Some(result) = parse_study_result(&ds) {
                                    tracing::debug!(
                                        "Found study: {} - {}",
                                        result.patient_name,
                                        result.study_description
                                    );
                                    results.push(result);
                                }
                            } else {
                                tracing::warn!("Failed to decode response dataset");
                            }
                        }
                    }
                }
                Pdu::ReleaseRQ => {
                    tracing::debug!("Received ReleaseRQ");
                    association.send(&Pdu::ReleaseRP).ok();
                    break;
                }
                Pdu::ReleaseRP => {
                    tracing::debug!("Received ReleaseRP");
                    break;
                }
                Pdu::AbortRQ { source } => {
                    tracing::error!("Received Abort from {:?}", source);
                    return Err(anyhow!("Association aborted"));
                }
                Pdu::AssociationRJ(rj) => {
                    tracing::error!("Association rejected: {:?}", rj);
                    return Err(anyhow!("Association rejected"));
                }
                other => {
                    tracing::warn!("Unexpected PDU type received: {:#?}", other);
                    // Continue trying to receive more PDUs
                }
            }
        }

        Ok(results)
    }

    /// Perform C-FIND query for series within a study
    pub fn find_series(&self, study_uid: &str) -> Result<Vec<SeriesResult>> {
        tracing::info!("C-FIND series for study {}", study_uid);

        let options = ClientAssociationOptions::new()
            .calling_ae_title(&self.server.our_ae_title)
            .called_ae_title(&self.server.ae_title)
            .with_presentation_context(STUDY_ROOT_FIND, vec![EXPLICIT_VR_LE, IMPLICIT_VR_LE]);

        let mut association = options
            .establish(self.addr())
            .map_err(|e| anyhow!("Failed to establish association: {}", e))?;

        let (pc_id, transfer_syntax) = {
            let pc = association
                .presentation_contexts()
                .iter()
                .find(|pc| pc.abstract_syntax == STUDY_ROOT_FIND)
                .ok_or_else(|| anyhow!("Study Root Query not accepted"))?;
            (pc.id, pc.transfer_syntax.clone())
        };

        // Build series-level query
        let msg_id = 1u16;
        let query_ds = build_series_query(study_uid);
        let cmd = build_c_find_rq(msg_id, STUDY_ROOT_FIND);
        let cmd_bytes = encode_command(&cmd)?;
        let data_bytes = encode_dataset_with_ts(&query_ds, &transfer_syntax)?;

        // Send command
        let cmd_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        };
        association
            .send(&cmd_pdu)
            .map_err(|e| anyhow!("Failed to send C-FIND command: {}", e))?;

        // Send query data
        let data_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Data,
                is_last: true,
                data: data_bytes,
            }],
        };
        association
            .send(&data_pdu)
            .map_err(|e| anyhow!("Failed to send C-FIND query: {}", e))?;

        let ts_uid = transfer_syntax.clone();

        // Receive responses
        let mut results = Vec::new();
        loop {
            let pdu = association
                .receive()
                .map_err(|e| anyhow!("Failed to receive C-FIND response: {}", e))?;

            match pdu {
                Pdu::PData { data } => {
                    for pdv in data {
                        if pdv.value_type == PDataValueType::Command {
                            if let Ok(cmd_ds) = decode_dataset(&pdv.data) {
                                let status = get_status(&cmd_ds).unwrap_or(0xFFFF);
                                tracing::debug!("C-FIND status: 0x{:04X}", status);
                                if status == 0x0000 {
                                    association.release().ok();
                                    return Ok(results);
                                } else if status != 0xFF00 && status != 0xFF01 {
                                    association.release().ok();
                                    return Err(anyhow!(
                                        "C-FIND failed with status: 0x{:04X}",
                                        status
                                    ));
                                }
                            }
                        } else if pdv.value_type == PDataValueType::Data {
                            if let Ok(ds) = decode_dataset_with_ts(&pdv.data, &ts_uid) {
                                if let Some(result) = parse_series_result(&ds, study_uid) {
                                    results.push(result);
                                }
                            }
                        }
                    }
                }
                Pdu::ReleaseRQ => {
                    association.send(&Pdu::ReleaseRP).ok();
                    break;
                }
                _ => {}
            }
        }

        Ok(results)
    }

    /// Retrieve a study via C-MOVE to local cache
    /// Returns the directory where files were saved
    pub fn retrieve_study(
        &self,
        study_uid: &str,
        cache_dir: &std::path::Path,
        scp_port: u16,
        progress_tx: mpsc::Sender<RetrieveProgress>,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Result<PathBuf> {
        tracing::info!("C-MOVE study {} to {:?}", study_uid, cache_dir);

        // Create study directory
        let study_dir = cache_dir.join(study_uid.replace('.', "_"));
        std::fs::create_dir_all(&study_dir)?;

        // Establish association for C-MOVE (no read timeout - blocking receive)
        let options = ClientAssociationOptions::new()
            .calling_ae_title(&self.server.our_ae_title)
            .called_ae_title(&self.server.ae_title)
            .with_presentation_context(STUDY_ROOT_MOVE, vec![EXPLICIT_VR_LE, IMPLICIT_VR_LE]);

        let mut association = options
            .establish(self.addr())
            .map_err(|e| anyhow!("Failed to establish association: {}", e))?;

        let (pc_id, transfer_syntax) = {
            let pc = association
                .presentation_contexts()
                .iter()
                .find(|pc| pc.abstract_syntax == STUDY_ROOT_MOVE)
                .ok_or_else(|| anyhow!("Study Root Move not accepted"))?;
            (pc.id, pc.transfer_syntax.clone())
        };

        tracing::info!(
            "C-MOVE using presentation context {}, transfer syntax: {}",
            pc_id,
            transfer_syntax
        );

        // Build C-MOVE request
        let msg_id = 1u16;
        let cmd = build_c_move_rq(msg_id, STUDY_ROOT_MOVE, &self.server.our_ae_title);
        let query = build_study_level_retrieve(study_uid);
        let cmd_bytes = encode_command(&cmd)?;
        let data_bytes = encode_dataset_with_ts(&query, &transfer_syntax)?;

        // Send command PDU
        let cmd_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        };
        association
            .send(&cmd_pdu)
            .map_err(|e| anyhow!("Failed to send C-MOVE command: {}", e))?;

        // Send data PDU (query identifier)
        let data_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Data,
                is_last: true,
                data: data_bytes,
            }],
        };
        association
            .send(&data_pdu)
            .map_err(|e| anyhow!("Failed to send C-MOVE query: {}", e))?;

        tracing::info!("C-MOVE request sent, waiting for response...");
        tracing::info!(
            "Move destination: {} (port {})",
            self.server.our_ae_title,
            scp_port
        );

        // Receive C-MOVE responses (status updates)
        // Note: cancel_flag is currently unused - blocking receive prevents checking it
        let _ = cancel_flag; // suppress unused warning
        loop {
            let pdu = association
                .receive()
                .map_err(|e| anyhow!("Failed to receive C-MOVE response: {}", e))?;

            match pdu {
                Pdu::PData { data } => {
                    for pdv in data {
                        if pdv.value_type == PDataValueType::Command {
                            if let Ok(cmd_ds) = decode_dataset(&pdv.data) {
                                let status = get_status(&cmd_ds).unwrap_or(0xFFFF);
                                tracing::debug!("C-MOVE status: 0x{:04X}", status);

                                // Extract sub-operation counts
                                let remaining = cmd_ds
                                    .element(tags::NUMBER_OF_REMAINING_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);
                                let completed = cmd_ds
                                    .element(tags::NUMBER_OF_COMPLETED_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);
                                let failed = cmd_ds
                                    .element(tags::NUMBER_OF_FAILED_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);
                                let warnings = cmd_ds
                                    .element(tags::NUMBER_OF_WARNING_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);

                                tracing::debug!(
                                    "C-MOVE progress: {} completed, {} remaining, {} failed, {} warnings",
                                    completed, remaining, failed, warnings
                                );

                                // Send progress update
                                let mut progress = RetrieveProgress::new();
                                progress.completed = completed;
                                progress.remaining = remaining;
                                progress.failed = failed;
                                progress.warnings = warnings;

                                if status == 0x0000 {
                                    // Success - all operations complete
                                    tracing::info!(
                                        "C-MOVE complete: {} images retrieved",
                                        completed
                                    );
                                    progress.is_complete = true;
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Ok(study_dir);
                                } else if status == 0xFF00 {
                                    // Pending - operations in progress
                                    let _ = progress_tx.send(progress);
                                } else if (status & 0xF000) == 0xA000 {
                                    // Failure
                                    progress.is_complete = true;
                                    progress.error = Some(format!(
                                        "C-MOVE failed with status: 0x{:04X}",
                                        status
                                    ));
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Err(anyhow!(
                                        "C-MOVE failed with status: 0x{:04X}",
                                        status
                                    ));
                                } else if (status & 0xF000) == 0xB000 {
                                    // Warning - some operations failed
                                    tracing::warn!(
                                        "C-MOVE completed with warnings: {} failed",
                                        failed
                                    );
                                    progress.is_complete = true;
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Ok(study_dir);
                                } else if status == 0xFE00 {
                                    // Cancel
                                    progress.is_complete = true;
                                    progress.error = Some("C-MOVE cancelled".to_string());
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Err(anyhow!("C-MOVE cancelled"));
                                }
                            }
                        }
                    }
                }
                Pdu::ReleaseRQ => {
                    tracing::debug!("Received ReleaseRQ");
                    association.send(&Pdu::ReleaseRP).ok();
                    break;
                }
                Pdu::ReleaseRP => {
                    tracing::debug!("Received ReleaseRP");
                    break;
                }
                Pdu::AbortRQ { source } => {
                    tracing::error!("Received Abort from {:?}", source);
                    let mut progress = RetrieveProgress::new();
                    progress.is_complete = true;
                    progress.error = Some("Association aborted".to_string());
                    let _ = progress_tx.send(progress);
                    return Err(anyhow!("Association aborted"));
                }
                other => {
                    tracing::warn!("Unexpected PDU: {:?}", other);
                }
            }
        }

        Ok(study_dir)
    }

    /// Retrieve a single series via C-MOVE to local cache
    /// Used by parallel retriever to fetch individual series
    pub fn retrieve_series(
        &self,
        study_uid: &str,
        series_uid: &str,
        cache_dir: &std::path::Path,
        _scp_port: u16,
        progress_tx: mpsc::Sender<RetrieveProgress>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()> {
        tracing::info!("C-MOVE series {} to {:?}", series_uid, cache_dir);

        // Establish association for C-MOVE
        let options = ClientAssociationOptions::new()
            .calling_ae_title(&self.server.our_ae_title)
            .called_ae_title(&self.server.ae_title)
            .with_presentation_context(STUDY_ROOT_MOVE, vec![EXPLICIT_VR_LE, IMPLICIT_VR_LE]);

        let mut association = options
            .establish(self.addr())
            .map_err(|e| anyhow!("Failed to establish association: {}", e))?;

        let (pc_id, transfer_syntax) = {
            let pc = association
                .presentation_contexts()
                .iter()
                .find(|pc| pc.abstract_syntax == STUDY_ROOT_MOVE)
                .ok_or_else(|| anyhow!("Study Root Move not accepted"))?;
            (pc.id, pc.transfer_syntax.clone())
        };

        tracing::debug!("C-MOVE series using pc {}, ts: {}", pc_id, transfer_syntax);

        // Build C-MOVE request at SERIES level
        let msg_id = 1u16;
        let cmd = build_c_move_rq(msg_id, STUDY_ROOT_MOVE, &self.server.our_ae_title);
        let query = build_series_level_retrieve(study_uid, series_uid);
        let cmd_bytes = encode_command(&cmd)?;
        let data_bytes = encode_dataset_with_ts(&query, &transfer_syntax)?;

        // Send command PDU
        let cmd_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        };
        association
            .send(&cmd_pdu)
            .map_err(|e| anyhow!("Failed to send C-MOVE command: {}", e))?;

        // Send data PDU (query identifier)
        let data_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Data,
                is_last: true,
                data: data_bytes,
            }],
        };
        association
            .send(&data_pdu)
            .map_err(|e| anyhow!("Failed to send C-MOVE query: {}", e))?;

        tracing::debug!("C-MOVE series request sent, waiting for response...");

        // Receive C-MOVE responses (status updates)
        loop {
            // Check cancellation before blocking receive
            if cancel_flag.load(Ordering::SeqCst) {
                tracing::info!("Series retrieve cancelled");
                association.abort().ok();
                return Err(anyhow!("Retrieve cancelled"));
            }

            let pdu = association
                .receive()
                .map_err(|e| anyhow!("Failed to receive C-MOVE response: {}", e))?;

            match pdu {
                Pdu::PData { data } => {
                    for pdv in data {
                        if pdv.value_type == PDataValueType::Command {
                            if let Ok(cmd_ds) = decode_dataset(&pdv.data) {
                                let status = get_status(&cmd_ds).unwrap_or(0xFFFF);
                                tracing::debug!("C-MOVE series status: 0x{:04X}", status);

                                // Extract sub-operation counts
                                let remaining = cmd_ds
                                    .element(tags::NUMBER_OF_REMAINING_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);
                                let completed = cmd_ds
                                    .element(tags::NUMBER_OF_COMPLETED_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);
                                let failed = cmd_ds
                                    .element(tags::NUMBER_OF_FAILED_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);
                                let warnings = cmd_ds
                                    .element(tags::NUMBER_OF_WARNING_SUBOPERATIONS)
                                    .ok()
                                    .and_then(|e| e.to_int::<u32>().ok())
                                    .unwrap_or(0);

                                // Send progress update
                                let mut progress = RetrieveProgress::new();
                                progress.completed = completed;
                                progress.remaining = remaining;
                                progress.failed = failed;
                                progress.warnings = warnings;

                                if status == 0x0000 {
                                    // Success
                                    tracing::info!("C-MOVE series complete: {} images", completed);
                                    progress.is_complete = true;
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Ok(());
                                } else if status == 0xFF00 {
                                    // Pending
                                    let _ = progress_tx.send(progress);
                                } else if (status & 0xF000) == 0xA000 {
                                    // Failure
                                    progress.is_complete = true;
                                    progress.error =
                                        Some(format!("C-MOVE failed: 0x{:04X}", status));
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Err(anyhow!(
                                        "C-MOVE failed with status: 0x{:04X}",
                                        status
                                    ));
                                } else if (status & 0xF000) == 0xB000 {
                                    // Warning - some failed
                                    tracing::warn!(
                                        "C-MOVE series with warnings: {} failed",
                                        failed
                                    );
                                    progress.is_complete = true;
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Ok(());
                                } else if status == 0xFE00 {
                                    // Cancel
                                    progress.is_complete = true;
                                    progress.error = Some("C-MOVE cancelled".to_string());
                                    let _ = progress_tx.send(progress);
                                    association.release().ok();
                                    return Err(anyhow!("C-MOVE cancelled"));
                                }
                            }
                        }
                    }
                }
                Pdu::ReleaseRQ => {
                    association.send(&Pdu::ReleaseRP).ok();
                    break;
                }
                Pdu::ReleaseRP => {
                    break;
                }
                Pdu::AbortRQ { source } => {
                    tracing::error!("Received Abort from {:?}", source);
                    let mut progress = RetrieveProgress::new();
                    progress.is_complete = true;
                    progress.error = Some("Association aborted".to_string());
                    let _ = progress_tx.send(progress);
                    return Err(anyhow!("Association aborted"));
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Store a DICOM file to the PACS server (C-STORE)
    pub fn store(&self, file_path: &std::path::Path) -> Result<()> {
        use dicom::object::open_file;

        tracing::info!(
            "C-STORE {} to {} ({})",
            file_path.display(),
            self.server.name,
            self.addr()
        );

        // Open the DICOM file
        let file_obj =
            open_file(file_path).map_err(|e| anyhow!("Failed to open DICOM file: {}", e))?;

        // Get SOP Class UID and SOP Instance UID from the file
        let sop_class_uid = file_obj
            .element(tags::SOP_CLASS_UID)
            .map_err(|_| anyhow!("Missing SOP Class UID"))?
            .to_str()
            .map_err(|_| anyhow!("Invalid SOP Class UID"))?
            .trim()
            .to_string();

        let sop_instance_uid = file_obj
            .element(tags::SOP_INSTANCE_UID)
            .map_err(|_| anyhow!("Missing SOP Instance UID"))?
            .to_str()
            .map_err(|_| anyhow!("Invalid SOP Instance UID"))?
            .trim()
            .to_string();

        tracing::debug!(
            "SOP Class: {}, Instance: {}",
            sop_class_uid,
            sop_instance_uid
        );

        // Establish association for C-STORE
        // Use a static str slice for the SOP class UID to match the API
        let ts_list: &[&str] = &[EXPLICIT_VR_LE, IMPLICIT_VR_LE];
        let options = ClientAssociationOptions::new()
            .calling_ae_title(&self.server.our_ae_title)
            .called_ae_title(&self.server.ae_title)
            .with_presentation_context(sop_class_uid.as_str(), ts_list.to_vec());

        let mut association = options
            .establish(self.addr())
            .map_err(|e| anyhow!("Failed to establish association: {}", e))?;

        let (pc_id, transfer_syntax) = {
            let pc = association
                .presentation_contexts()
                .iter()
                .find(|pc| pc.abstract_syntax == sop_class_uid)
                .ok_or_else(|| anyhow!("SOP Class {} not accepted", sop_class_uid))?;
            (pc.id, pc.transfer_syntax.clone())
        };

        tracing::debug!(
            "Using presentation context {}, transfer syntax: {}",
            pc_id,
            transfer_syntax
        );

        // Build C-STORE command
        let msg_id = 1u16;
        let cmd = build_c_store_rq(msg_id, &sop_class_uid, &sop_instance_uid);
        let cmd_bytes = encode_command(&cmd)?;

        // Encode the dataset
        let data_bytes = encode_file_dataset(&file_obj, &transfer_syntax)?;

        // Send command PDU
        let cmd_pdu = Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: pc_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        };

        association
            .send(&cmd_pdu)
            .map_err(|e| anyhow!("Failed to send C-STORE command: {}", e))?;

        // Send data PDU - chunk to fit within max PDU size
        // Default max PDU is 16384, PDV header takes ~12 bytes
        // Use 8KB chunks to be safe with any negotiated max PDU
        let chunk_size = 8192;

        for (i, chunk) in data_bytes.chunks(chunk_size).enumerate() {
            let is_last = (i + 1) * chunk_size >= data_bytes.len();
            let data_pdu = Pdu::PData {
                data: vec![PDataValue {
                    presentation_context_id: pc_id,
                    value_type: PDataValueType::Data,
                    is_last,
                    data: chunk.to_vec(),
                }],
            };

            association
                .send(&data_pdu)
                .map_err(|e| anyhow!("Failed to send C-STORE data: {}", e))?;
        }

        tracing::debug!("C-STORE request sent, waiting for response...");

        // Receive response
        let response = association
            .receive()
            .map_err(|e| anyhow!("Failed to receive C-STORE response: {}", e))?;

        let status = match response {
            Pdu::PData { data } => {
                if let Some(pdv) = data.first() {
                    if pdv.value_type == PDataValueType::Command {
                        if let Ok(rsp) = decode_dataset(&pdv.data) {
                            get_status(&rsp).unwrap_or(0xFFFF)
                        } else {
                            0xFFFF
                        }
                    } else {
                        0xFFFF
                    }
                } else {
                    0xFFFF
                }
            }
            _ => 0xFFFF,
        };

        association.release().ok();

        if status == 0x0000 {
            tracing::info!("C-STORE successful for {}", sop_instance_uid);
            Ok(())
        } else {
            Err(anyhow!("C-STORE failed with status: 0x{:04X}", status))
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn build_c_echo_rq(msg_id: u16) -> InMemDicomObject<StandardDataDictionary> {
    let mut cmd = InMemDicomObject::new_empty();
    cmd.put(DataElement::new(
        tags::AFFECTED_SOP_CLASS_UID,
        VR::UI,
        PrimitiveValue::from(VERIFICATION_SOP),
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_FIELD,
        VR::US,
        PrimitiveValue::from(0x0030u16), // C-ECHO-RQ
    ));
    cmd.put(DataElement::new(
        tags::MESSAGE_ID,
        VR::US,
        PrimitiveValue::from(msg_id),
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_DATA_SET_TYPE,
        VR::US,
        PrimitiveValue::from(0x0101u16), // No data set
    ));
    cmd
}

fn build_c_find_rq(msg_id: u16, sop_class: &str) -> InMemDicomObject<StandardDataDictionary> {
    let mut cmd = InMemDicomObject::new_empty();
    cmd.put(DataElement::new(
        tags::AFFECTED_SOP_CLASS_UID,
        VR::UI,
        PrimitiveValue::from(sop_class),
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_FIELD,
        VR::US,
        PrimitiveValue::from(0x0020u16), // C-FIND-RQ
    ));
    cmd.put(DataElement::new(
        tags::MESSAGE_ID,
        VR::US,
        PrimitiveValue::from(msg_id),
    ));
    cmd.put(DataElement::new(
        tags::PRIORITY,
        VR::US,
        PrimitiveValue::from(0x0000u16), // Medium
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_DATA_SET_TYPE,
        VR::US,
        PrimitiveValue::from(0x0000u16), // Data set present
    ));
    cmd
}

fn build_c_move_rq(
    msg_id: u16,
    sop_class: &str,
    move_destination: &str,
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
        PrimitiveValue::from(0x0021u16), // C-MOVE-RQ
    ));
    cmd.put(DataElement::new(
        tags::MESSAGE_ID,
        VR::US,
        PrimitiveValue::from(msg_id),
    ));
    cmd.put(DataElement::new(
        tags::PRIORITY,
        VR::US,
        PrimitiveValue::from(0x0000u16), // Medium
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_DATA_SET_TYPE,
        VR::US,
        PrimitiveValue::from(0x0000u16), // Data set present
    ));
    // Move Destination - padded to 16 chars with spaces
    let padded_dest = format!("{:<16}", move_destination);
    cmd.put(DataElement::new(
        tags::MOVE_DESTINATION,
        VR::AE,
        PrimitiveValue::from(padded_dest.as_str()),
    ));
    cmd
}

fn build_c_store_rq(
    msg_id: u16,
    sop_class: &str,
    sop_instance: &str,
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
        PrimitiveValue::from(0x0001u16), // C-STORE-RQ
    ));
    cmd.put(DataElement::new(
        tags::MESSAGE_ID,
        VR::US,
        PrimitiveValue::from(msg_id),
    ));
    cmd.put(DataElement::new(
        tags::PRIORITY,
        VR::US,
        PrimitiveValue::from(0x0000u16), // Medium
    ));
    cmd.put(DataElement::new(
        tags::COMMAND_DATA_SET_TYPE,
        VR::US,
        PrimitiveValue::from(0x0000u16), // Data set present
    ));
    cmd.put(DataElement::new(
        tags::AFFECTED_SOP_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(sop_instance),
    ));
    cmd
}

fn build_study_level_retrieve(study_uid: &str) -> InMemDicomObject<StandardDataDictionary> {
    let mut ds = InMemDicomObject::new_empty();
    ds.put(DataElement::new(
        tags::QUERY_RETRIEVE_LEVEL,
        VR::CS,
        PrimitiveValue::from("STUDY"),
    ));
    ds.put(DataElement::new(
        tags::STUDY_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(study_uid),
    ));
    ds
}

fn build_series_level_retrieve(
    study_uid: &str,
    series_uid: &str,
) -> InMemDicomObject<StandardDataDictionary> {
    let mut ds = InMemDicomObject::new_empty();
    ds.put(DataElement::new(
        tags::QUERY_RETRIEVE_LEVEL,
        VR::CS,
        PrimitiveValue::from("SERIES"),
    ));
    ds.put(DataElement::new(
        tags::STUDY_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(study_uid),
    ));
    ds.put(DataElement::new(
        tags::SERIES_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(series_uid),
    ));
    ds
}

fn build_study_query(params: &QueryParams) -> InMemDicomObject<StandardDataDictionary> {
    let mut ds = InMemDicomObject::new_empty();

    // Query/Retrieve Level
    ds.put(DataElement::new(
        tags::QUERY_RETRIEVE_LEVEL,
        VR::CS,
        PrimitiveValue::from("STUDY"),
    ));

    // Return keys (always request these)
    ds.put(DataElement::new(
        tags::STUDY_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::PATIENT_NAME,
        VR::PN,
        PrimitiveValue::from(params.patient_name.as_deref().unwrap_or("")),
    ));
    ds.put(DataElement::new(
        tags::PATIENT_ID,
        VR::LO,
        PrimitiveValue::from(params.patient_id.as_deref().unwrap_or("")),
    ));
    ds.put(DataElement::new(
        tags::ACCESSION_NUMBER,
        VR::SH,
        PrimitiveValue::from(params.accession_number.as_deref().unwrap_or("")),
    ));
    ds.put(DataElement::new(
        tags::STUDY_DATE,
        VR::DA,
        PrimitiveValue::from(params.study_date.as_deref().unwrap_or("")),
    ));
    ds.put(DataElement::new(
        tags::STUDY_TIME,
        VR::TM,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::STUDY_DESCRIPTION,
        VR::LO,
        PrimitiveValue::from(params.study_description.as_deref().unwrap_or("")),
    ));
    ds.put(DataElement::new(
        tags::MODALITIES_IN_STUDY,
        VR::CS,
        PrimitiveValue::from(params.modality.as_deref().unwrap_or("")),
    ));
    ds.put(DataElement::new(
        tags::NUMBER_OF_STUDY_RELATED_SERIES,
        VR::IS,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::NUMBER_OF_STUDY_RELATED_INSTANCES,
        VR::IS,
        PrimitiveValue::from(""),
    ));

    ds
}

fn build_series_query(study_uid: &str) -> InMemDicomObject<StandardDataDictionary> {
    let mut ds = InMemDicomObject::new_empty();

    ds.put(DataElement::new(
        tags::QUERY_RETRIEVE_LEVEL,
        VR::CS,
        PrimitiveValue::from("SERIES"),
    ));
    ds.put(DataElement::new(
        tags::STUDY_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(study_uid),
    ));
    ds.put(DataElement::new(
        tags::SERIES_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::SERIES_NUMBER,
        VR::IS,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::SERIES_DESCRIPTION,
        VR::LO,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::MODALITY,
        VR::CS,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::BODY_PART_EXAMINED,
        VR::CS,
        PrimitiveValue::from(""),
    ));
    ds.put(DataElement::new(
        tags::NUMBER_OF_SERIES_RELATED_INSTANCES,
        VR::IS,
        PrimitiveValue::from(""),
    ));

    ds
}

fn encode_dataset(ds: &InMemDicomObject<StandardDataDictionary>) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    ds.write_dataset_with_ts(
        &mut buf,
        &dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
    )?;
    Ok(buf)
}

/// Encode dataset with specific transfer syntax
fn encode_dataset_with_ts(
    ds: &InMemDicomObject<StandardDataDictionary>,
    ts_uid: &str,
) -> Result<Vec<u8>> {
    let ts = match ts_uid {
        EXPLICIT_VR_LE => &dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased(),
        IMPLICIT_VR_LE => &dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
        _ => {
            tracing::warn!("Unknown transfer syntax {}, using Explicit VR LE", ts_uid);
            &dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased()
        }
    };

    let mut buf = Vec::new();
    ds.write_dataset_with_ts(&mut buf, ts)?;
    Ok(buf)
}

/// Encode a DIMSE command with proper CommandGroupLength
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

/// Encode a FileDicomObject's dataset for C-STORE
fn encode_file_dataset(
    file_obj: &dicom::object::FileDicomObject<InMemDicomObject<StandardDataDictionary>>,
    ts_uid: &str,
) -> Result<Vec<u8>> {
    let ts = match ts_uid {
        EXPLICIT_VR_LE => dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased(),
        IMPLICIT_VR_LE => dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
        _ => {
            tracing::warn!("Unknown transfer syntax {}, using Explicit VR LE", ts_uid);
            dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased()
        }
    };

    let mut buf = Vec::new();
    file_obj.write_dataset_with_ts(&mut buf, &ts)?;
    Ok(buf)
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

/// Decode dataset with specific transfer syntax
fn decode_dataset_with_ts(
    data: &[u8],
    ts_uid: &str,
) -> Result<InMemDicomObject<StandardDataDictionary>> {
    use std::io::Cursor;

    let ts = match ts_uid {
        EXPLICIT_VR_LE => &dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased(),
        IMPLICIT_VR_LE => &dicom::transfer_syntax::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased(),
        _ => {
            tracing::warn!(
                "Unknown transfer syntax {} for decoding, trying Explicit VR LE",
                ts_uid
            );
            &dicom::transfer_syntax::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased()
        }
    };

    let cursor = Cursor::new(data);
    let ds = InMemDicomObject::read_dataset_with_ts(cursor, ts)?;
    Ok(ds)
}

fn get_status(cmd: &InMemDicomObject<StandardDataDictionary>) -> Result<u16> {
    cmd.element(tags::STATUS)?
        .to_int::<u16>()
        .map_err(|e| anyhow!("Invalid status: {}", e))
}

fn parse_study_result(ds: &InMemDicomObject<StandardDataDictionary>) -> Option<StudyResult> {
    let study_uid = ds.element(tags::STUDY_INSTANCE_UID).ok()?.to_str().ok()?;

    Some(StudyResult {
        study_instance_uid: study_uid.to_string(),
        patient_name: ds
            .element(tags::PATIENT_NAME)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        patient_id: ds
            .element(tags::PATIENT_ID)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        accession_number: ds
            .element(tags::ACCESSION_NUMBER)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        study_date: ds
            .element(tags::STUDY_DATE)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        study_time: ds
            .element(tags::STUDY_TIME)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        study_description: ds
            .element(tags::STUDY_DESCRIPTION)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        modalities: ds
            .element(tags::MODALITIES_IN_STUDY)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        num_series: ds
            .element(tags::NUMBER_OF_STUDY_RELATED_SERIES)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok()),
        num_instances: ds
            .element(tags::NUMBER_OF_STUDY_RELATED_INSTANCES)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok()),
    })
}

fn parse_series_result(
    ds: &InMemDicomObject<StandardDataDictionary>,
    study_uid: &str,
) -> Option<SeriesResult> {
    let series_uid = ds.element(tags::SERIES_INSTANCE_UID).ok()?.to_str().ok()?;

    Some(SeriesResult {
        series_instance_uid: series_uid.to_string(),
        study_instance_uid: study_uid.to_string(),
        series_number: ds
            .element(tags::SERIES_NUMBER)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok()),
        series_description: ds
            .element(tags::SERIES_DESCRIPTION)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        modality: ds
            .element(tags::MODALITY)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default(),
        body_part: ds
            .element(tags::BODY_PART_EXAMINED)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.to_string()),
        num_instances: ds
            .element(tags::NUMBER_OF_SERIES_RELATED_INSTANCES)
            .ok()
            .and_then(|e| e.to_int::<i32>().ok()),
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_params() {
        let params = QueryParams::new()
            .with_patient_name("Smith*")
            .with_modality("CT");

        assert_eq!(params.patient_name, Some("Smith*".to_string()));
        assert_eq!(params.modality, Some("CT".to_string()));
        assert!(!params.is_empty());
    }

    #[test]
    fn test_empty_query_params() {
        let params = QueryParams::new();
        assert!(params.is_empty());
    }
}
