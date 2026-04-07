#![allow(dead_code, unused_variables, unused_imports)]
//! CLI tool to test DICOM parsing without GUI
//!
//! Usage: cargo run --bin dicom_test -- <path_to_dicom>

use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_dicom_file_or_folder>", args[0]);
        eprintln!("\nThis tool tests DICOM parsing without the GUI.");
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);

    if !path.exists() {
        eprintln!("Error: Path does not exist: {}", args[1]);
        std::process::exit(1);
    }

    if path.is_file() {
        test_single_file(path);
    } else if path.is_dir() {
        test_directory(path);
    }
}

fn test_single_file(path: &Path) {
    println!("Testing DICOM file: {:?}\n", path);

    match dicom::object::open_file(path) {
        Ok(obj) => {
            println!("SUCCESS: Valid DICOM file!\n");

            // Print transfer syntax
            println!("=== Transfer Syntax ===");
            let meta = obj.meta();
            println!(
                "Transfer Syntax UID: {}",
                meta.transfer_syntax().trim_end_matches('\0')
            );

            // SOP Class
            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::SOP_CLASS_UID) {
                if let Ok(val) = elem.to_str() {
                    println!("SOP Class UID: {}", val);
                }
            }

            // Print metadata
            println!("\n=== DICOM Metadata ===");

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::PATIENT_NAME) {
                if let Ok(val) = elem.to_str() {
                    println!("Patient Name: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::PATIENT_ID) {
                if let Ok(val) = elem.to_str() {
                    println!("Patient ID: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::STUDY_DATE) {
                if let Ok(val) = elem.to_str() {
                    println!("Study Date: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::MODALITY) {
                if let Ok(val) = elem.to_str() {
                    println!("Modality: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::STUDY_DESCRIPTION) {
                if let Ok(val) = elem.to_str() {
                    println!("Study Description: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::SERIES_DESCRIPTION) {
                if let Ok(val) = elem.to_str() {
                    println!("Series Description: {}", val);
                }
            }

            // Image dimensions
            println!("\n=== Image Info ===");

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::ROWS) {
                if let Ok(val) = elem.to_int::<u32>() {
                    println!("Rows: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::COLUMNS) {
                if let Ok(val) = elem.to_int::<u32>() {
                    println!("Columns: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::BITS_ALLOCATED) {
                if let Ok(val) = elem.to_int::<u16>() {
                    println!("Bits Allocated: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::PHOTOMETRIC_INTERPRETATION) {
                if let Ok(val) = elem.to_str() {
                    println!("Photometric Interpretation: {}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::PIXEL_SPACING) {
                if let Ok(vals) = elem.to_multi_float64() {
                    let vals: Vec<f64> = vals.into_iter().collect();
                    if vals.len() >= 2 {
                        println!("Pixel Spacing: {:.3} x {:.3} mm", vals[0], vals[1]);
                    }
                }
            }

            // Window/Level
            println!("\n=== Window/Level ===");

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::WINDOW_CENTER) {
                if let Ok(val) = elem.to_float64() {
                    println!("Window Center: {:.1}", val);
                }
            }

            if let Ok(elem) = obj.element(dicom::dictionary_std::tags::WINDOW_WIDTH) {
                if let Ok(val) = elem.to_float64() {
                    println!("Window Width: {:.1}", val);
                }
            }

            // Try pixel data - both methods
            println!("\n=== Pixel Data ===");

            // Method 1: decode_pixel_data()
            use dicom_pixeldata::PixelDecoder;
            match obj.decode_pixel_data() {
                Ok(pixels) => {
                    println!(
                        "decode_pixel_data() succeeded: {} bytes",
                        pixels.data().len()
                    );
                }
                Err(e) => {
                    println!("decode_pixel_data() failed: {}", e);
                }
            }

            // Method 2: direct extraction
            use dicom::core::Tag;
            let pd_tag = Tag(0x7FE0, 0x0010);
            match obj.element(pd_tag) {
                Ok(elem) => match elem.to_bytes() {
                    Ok(bytes) => {
                        println!(
                            "Direct PixelData extraction succeeded: {} bytes",
                            bytes.len()
                        );
                    }
                    Err(e) => {
                        println!("Direct PixelData to_bytes() failed: {}", e);
                    }
                },
                Err(e) => {
                    println!("Direct PixelData element() failed: {}", e);
                }
            }

            println!("\n=== Test PASSED ===");
        }
        Err(e) => {
            eprintln!("FAILED: Could not open DICOM file: {}", e);
            std::process::exit(1);
        }
    }
}

fn test_directory(path: &Path) {
    println!("Scanning directory: {:?}\n", path);

    let mut total = 0;
    let mut success = 0;
    let mut failed = 0;

    for entry in walkdir::WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            total += 1;

            match dicom::object::open_file(entry.path()) {
                Ok(_) => {
                    success += 1;
                    println!("[OK] {:?}", entry.path());
                }
                Err(_) => {
                    failed += 1;
                    // Don't print failures - might be non-DICOM files
                }
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Total files: {}", total);
    println!("DICOM files: {}", success);
    println!("Non-DICOM files: {}", failed);

    if success > 0 {
        println!("\n=== Test PASSED ===");
    } else {
        println!("\n=== No DICOM files found ===");
    }
}
