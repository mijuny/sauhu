#![allow(dead_code, unused_variables, unused_imports, unused_assignments)]
//! Test PACS connectivity with Orthanc
//!
//! Run with: cargo run --bin pacs_test

use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc;

// Re-include the modules we need (since this is a binary, not using lib)
#[path = "../db/mod.rs"]
mod db;

#[path = "../pacs/mod.rs"]
mod pacs;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("pacs_test=info,debug")
        .init();

    println!("=== PACS Test ===\n");

    // Create test server config for local Orthanc
    let server = db::PacsServer {
        id: 0,
        name: "Local Orthanc".to_string(),
        ae_title: "ORTHANC".to_string(),
        host: "localhost".to_string(),
        port: 4242,
        our_ae_title: "SAUHU".to_string(),
    };

    println!(
        "Testing connection to {} ({}:{})...",
        server.name, server.host, server.port
    );

    let scu = pacs::DicomScu::new(server.to_config());

    // Test C-ECHO
    println!("\n1. C-ECHO test:");
    match scu.echo() {
        Ok(true) => println!("   SUCCESS - Connection verified"),
        Ok(false) => println!("   FAILED - Echo returned false"),
        Err(e) => println!("   ERROR - {}", e),
    }

    // Test C-FIND (all studies)
    println!("\n2. C-FIND test (all studies):");
    let params = pacs::QueryParams::new();
    let studies = match scu.find_studies(&params) {
        Ok(results) => {
            println!("   Found {} studies:", results.len());
            for (i, study) in results.iter().enumerate() {
                println!(
                    "   {}. {} - {} - {} - {}",
                    i + 1,
                    study.patient_name,
                    study.patient_id,
                    study.study_description,
                    study.formatted_date()
                );
            }
            results
        }
        Err(e) => {
            println!("   ERROR - {}", e);
            vec![]
        }
    };

    // Test C-FIND with filter
    println!("\n3. C-FIND test (filter by patient name 'Test*'):");
    let params = pacs::QueryParams::new().with_patient_name("Test*");
    match scu.find_studies(&params) {
        Ok(results) => {
            println!("   Found {} studies:", results.len());
            for study in &results {
                println!("   - {} ({})", study.patient_name, study.study_instance_uid);
            }
        }
        Err(e) => println!("   ERROR - {}", e),
    }

    // Test C-MOVE (if we have studies)
    if !studies.is_empty() {
        println!("\n4. C-MOVE test (retrieve first study):");
        let study = &studies[0];
        println!(
            "   Retrieving: {} - {}",
            study.patient_name, study.study_description
        );

        // Start local SCP to receive images
        let cache_dir = PathBuf::from("/tmp/sauhu-pacs-test");
        let scp_port = 11112u16;

        println!("   Starting local SCP on port {}...", scp_port);
        let scp = pacs::DicomScp::new(&server.our_ae_title, scp_port, cache_dir.clone());

        match scp.start() {
            Ok(scp_rx) => {
                println!("   SCP started, initiating C-MOVE...");

                // Start C-MOVE in a thread
                let study_uid = study.study_instance_uid.clone();
                let scu_clone = pacs::DicomScu::new(server.to_config());
                let (move_tx, move_rx) = mpsc::channel();

                let move_handle = std::thread::spawn(move || {
                    scu_clone.retrieve_study(&study_uid, &cache_dir, scp_port, move_tx, None)
                });

                // Monitor progress from both SCP and C-MOVE
                let mut scp_received = 0; // will be assigned later
                let mut move_completed = false;

                loop {
                    // Check SCP progress
                    if let Ok(progress) = scp_rx.try_recv() {
                        scp_received = progress.received;
                        println!("   SCP: {} images received", scp_received);
                        if progress.is_complete {
                            println!("   SCP complete");
                        }
                    }

                    // Check C-MOVE progress
                    if let Ok(progress) = move_rx.try_recv() {
                        println!(
                            "   C-MOVE: {} completed, {} remaining",
                            progress.completed, progress.remaining
                        );
                        if progress.is_complete {
                            move_completed = true;
                            if let Some(err) = &progress.error {
                                println!("   C-MOVE error: {}", err);
                            }
                        }
                    }

                    if move_completed {
                        break;
                    }

                    std::thread::sleep(std::time::Duration::from_millis(100));
                }

                // Wait for move thread
                match move_handle.join() {
                    Ok(Ok(path)) => {
                        println!("   SUCCESS - Images saved to {:?}", path);
                        // List received files
                        if let Ok(entries) = std::fs::read_dir(&path) {
                            let count = entries.filter(|e| e.is_ok()).count();
                            println!("   {} files in directory", count);
                        }
                    }
                    Ok(Err(e)) => println!("   ERROR - C-MOVE failed: {}", e),
                    Err(_) => println!("   ERROR - Thread panicked"),
                }

                // Stop SCP
                scp.stop();
            }
            Err(e) => println!("   ERROR - Failed to start SCP: {}", e),
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
