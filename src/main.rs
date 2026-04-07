//! Sauhu - Lightning-Fast DICOM Viewer for Linux
//!
//! A lean, fast DICOM viewer optimized for radiologist workflow.

mod app;
mod cache;
mod config;
mod coregistration;
mod db;
mod dicom;
mod fusion;
mod gpu;
mod hanging_protocol;
mod ipc;
mod pacs;
mod ui;

use anyhow::Result;
use std::env;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sauhu=debug,wgpu=warn,eframe=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Sauhu DICOM Viewer");

    // Load configuration
    let settings = config::Settings::load()?;
    tracing::info!(
        "Config: local AET={}, port={}, {} PACS server(s)",
        settings.local.ae_title,
        settings.local.port,
        settings.pacs.servers.len()
    );

    // Parse command line arguments
    let args: Vec<String> = env::args().skip(1).collect();

    // Check for CLI subcommands
    if !args.is_empty() {
        match args[0].as_str() {
            "pacs" => return run_pacs_command(&args[1..], &settings),
            "help" | "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
    }

    // GUI mode - collect files/folders to open
    let initial_paths: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();

    if !initial_paths.is_empty() {
        tracing::info!(
            "Will open {} file(s)/folder(s) from command line",
            initial_paths.len()
        );
    }

    // Initialize database
    let db = db::Database::open()?;
    tracing::info!("Database initialized");

    // Start IPC server
    let (ipc_tx, ipc_rx) = std::sync::mpsc::channel();
    let ipc_server = ipc::IpcServer::new(ipc_tx);
    if let Err(e) = ipc_server.start() {
        tracing::warn!("Failed to start IPC server: {}", e);
    }

    // Run the application
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Sauhu - DICOM Viewer")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Sauhu",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(app::SauhuApp::new_with_paths(
                cc,
                db,
                settings,
                initial_paths,
                ipc_rx,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run eframe: {}", e))?;

    Ok(())
}

fn print_help() {
    println!("Sauhu - Lightning-Fast DICOM Viewer for Linux");
    println!();
    println!("USAGE:");
    println!("    sauhu [FILES...]              Open DICOM files/folders in GUI");
    println!("    sauhu pacs <command>          PACS operations (CLI only)");
    println!();
    println!("PACS COMMANDS:");
    println!("    sauhu pacs find [OPTIONS]     Query PACS for studies");
    println!("    sauhu pacs move [OPTIONS]     Retrieve study from PACS");
    println!("    sauhu pacs store <FILES...>   Store DICOM files to PACS");
    println!();
    println!("FIND OPTIONS:");
    println!("    --patient-id <ID>             Filter by patient ID");
    println!("    --patient-name <NAME>         Filter by patient name (wildcards: *)");
    println!("    --study-date <DATE>           Filter by study date (YYYYMMDD)");
    println!("    --modality <MOD>              Filter by modality (CT, MR, etc.)");
    println!();
    println!("MOVE OPTIONS:");
    println!("    --patient-id <ID>             Retrieve by patient ID");
    println!("    --study-uid <UID>             Retrieve by Study Instance UID");
    println!();
    println!("KEYBOARD SHORTCUTS (GUI):");
    println!("    D                             Open Database window");
    println!("    B                             Toggle patient sidebar");
    println!("    1-9                           Window presets");
    println!("    0                             Auto window");
    println!("    Space                         Fit to window");
}

fn run_pacs_command(args: &[String], settings: &config::Settings) -> Result<()> {
    if args.is_empty() {
        println!("Usage: sauhu pacs <find|move|store> [OPTIONS]");
        return Ok(());
    }

    // Get default PACS server
    let server = settings
        .pacs
        .servers
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No PACS servers configured"))?;

    let pacs_server = db::PacsServer {
        id: 0,
        name: server.name.clone(),
        ae_title: server.ae_title.clone(),
        host: server.host.clone(),
        port: server.port as i32,
        our_ae_title: settings.local.ae_title.clone(),
    };

    match args[0].as_str() {
        "find" => run_pacs_find(&args[1..], &pacs_server),
        "move" => run_pacs_move(&args[1..], &pacs_server, settings),
        "store" => run_pacs_store(&args[1..], &pacs_server),
        _ => {
            println!("Unknown PACS command: {}", args[0]);
            println!("Available: find, move, store");
            Ok(())
        }
    }
}

fn run_pacs_find(args: &[String], server: &db::PacsServer) -> Result<()> {
    let mut params = pacs::QueryParams::new();

    // Parse options
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--patient-id" => {
                if i + 1 < args.len() {
                    params = params.with_patient_id(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--patient-name" => {
                if i + 1 < args.len() {
                    params = params.with_patient_name(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--study-date" => {
                if i + 1 < args.len() {
                    params = params.with_date(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--modality" => {
                if i + 1 < args.len() {
                    params = params.with_modality(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    let scu = pacs::DicomScu::new(server.to_config());
    let studies = scu.find_studies(&params)?;

    println!("Found {} studies:", studies.len());
    for study in &studies {
        println!(
            "  {} | {} | {} | {} | {}",
            study.patient_id,
            study.patient_name,
            study.study_date,
            study.modalities,
            study.study_description
        );
    }

    Ok(())
}

fn run_pacs_move(
    args: &[String],
    server: &db::PacsServer,
    settings: &config::Settings,
) -> Result<()> {
    let mut patient_id: Option<String> = None;
    let mut study_uid: Option<String> = None;

    // Parse options
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--patient-id" => {
                if i + 1 < args.len() {
                    patient_id = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--study-uid" => {
                if i + 1 < args.len() {
                    study_uid = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // If patient-id provided, first do C-FIND to get study UID
    let study_uid = if let Some(uid) = study_uid {
        uid
    } else if let Some(pid) = patient_id {
        let scu = pacs::DicomScu::new(server.to_config());
        let params = pacs::QueryParams::new().with_patient_id(&pid);
        let studies = scu.find_studies(&params)?;

        if studies.is_empty() {
            return Err(anyhow::anyhow!("No studies found for patient ID: {}", pid));
        }

        studies[0].study_instance_uid.clone()
    } else {
        return Err(anyhow::anyhow!("Must specify --patient-id or --study-uid"));
    };

    // Start SCP
    let cache_dir = settings.storage_path();
    let scp = pacs::DicomScp::new(
        &settings.local.ae_title,
        settings.local.port,
        cache_dir.clone(),
    );

    let _scp_rx = scp.start()?;
    println!("SCP started on port {}", settings.local.port);

    // C-MOVE
    let scu = pacs::DicomScu::new(server.to_config());
    let (tx, rx) = std::sync::mpsc::channel();

    let result = scu.retrieve_study(&study_uid, &cache_dir, settings.local.port, tx, None);

    // Drain progress messages
    while let Ok(progress) = rx.try_recv() {
        if progress.is_complete {
            if let Some(err) = progress.error {
                println!("Error: {}", err);
            } else {
                println!("Retrieved {} images", progress.completed);
            }
        }
    }

    scp.stop();

    match result {
        Ok(path) => {
            println!("Study saved to: {:?}", path);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

fn run_pacs_store(args: &[String], server: &db::PacsServer) -> Result<()> {
    if args.is_empty() {
        println!("Usage: sauhu pacs store <FILES...>");
        return Ok(());
    }

    let scu = pacs::DicomScu::new(server.to_config());

    for path in args {
        let file_path = PathBuf::from(path);
        if !file_path.exists() {
            println!("File not found: {}", path);
            continue;
        }

        println!("Storing: {}", path);
        match scu.store(&file_path) {
            Ok(_) => println!("  Success"),
            Err(e) => println!("  Error: {}", e),
        }
    }

    Ok(())
}
