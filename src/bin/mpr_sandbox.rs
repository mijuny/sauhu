#![allow(dead_code, unused_variables, unused_imports, unused_assignments)]
//! Claude DICOM Viewer - Sandbox for testing and image analysis
//!
//! A command-line DICOM viewer that outputs PNG images, enabling Claude to:
//! - View and analyze DICOM images
//! - Test MPR/sync/reference line calculations
//! - Compare multiple series side-by-side
//! - Use measurement and ROI tools
//!
//! # Usage
//!
//! ```bash
//! # View a single series with MPR planes
//! mpr_sandbox view /path/to/dicom -o output.png
//!
//! # Compare multiple series with sync
//! mpr_sandbox compare /series1 /series2 --sync -o comparison.png
//!
//! # Mark a point across all viewports
//! mpr_sandbox point /path/to/dicom --xyz -50,30,120 -o point.png
//!
//! # Measure ROI statistics
//! mpr_sandbox roi /path/to/dicom --center 100,150 --radius 10 --slice 15
//! ```

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use std::path::PathBuf;

use sauhu::coregistration::{
    compute_initial_alignment, PowellOptimizer, PyramidSchedule, RegistrationConfig,
    RegistrationPipeline, RegistrationResult, RigidTransform, VolumeGeometry,
};
use sauhu::dicom::{
    compute_circle_roi_stats, compute_distance_mm, compute_point_in_plane, compute_reference_line,
    group_files_by_series, AnatomicalPlane, DicomFile, DicomImage, ImagePlane, MprSeries, Point2D,
    Vec3, Volume,
};
use sauhu::gpu::GpuCoregistration;

/// Window/level presets
#[derive(Debug, Clone, Copy)]
struct WindowPreset {
    name: &'static str,
    center: f64,
    width: f64,
}

const PRESETS: &[WindowPreset] = &[
    WindowPreset {
        name: "brain",
        center: 40.0,
        width: 80.0,
    },
    WindowPreset {
        name: "subdural",
        center: 75.0,
        width: 215.0,
    },
    WindowPreset {
        name: "stroke",
        center: 40.0,
        width: 40.0,
    },
    WindowPreset {
        name: "bone",
        center: 400.0,
        width: 1800.0,
    },
    WindowPreset {
        name: "soft",
        center: 50.0,
        width: 400.0,
    },
    WindowPreset {
        name: "lung",
        center: -600.0,
        width: 1500.0,
    },
    WindowPreset {
        name: "abdomen",
        center: 40.0,
        width: 400.0,
    },
];

fn get_preset(name: &str) -> Option<WindowPreset> {
    PRESETS.iter().find(|p| p.name == name).copied()
}

/// Viewport state for rendering
struct Viewport {
    /// DicomImage to display
    image: Option<DicomImage>,
    /// Window center
    window_center: f64,
    /// Window width
    window_width: f64,
    /// Label for this viewport
    label: String,
    /// Series path (for identification)
    series_path: Option<PathBuf>,
}

impl Viewport {
    fn new(label: &str) -> Self {
        Self {
            image: None,
            window_center: 40.0,
            window_width: 400.0,
            label: label.to_string(),
            series_path: None,
        }
    }

    fn set_image(&mut self, image: DicomImage) {
        self.window_center = image.window_center;
        self.window_width = image.window_width;
        self.image = Some(image);
    }

    fn set_window(&mut self, center: f64, width: f64) {
        self.window_center = center;
        self.window_width = width;
    }

    fn image_plane(&self) -> Option<&ImagePlane> {
        self.image.as_ref().and_then(|img| img.image_plane.as_ref())
    }
}

/// Multi-viewport renderer
struct ViewerState {
    viewports: Vec<Viewport>,
    /// Active viewport index (for reference lines source)
    active: usize,
    /// Output image size per viewport
    viewport_size: (u32, u32),
    /// Points to mark across all viewports (patient coordinates)
    marked_points: Vec<Vec3>,
}

impl ViewerState {
    fn new(num_viewports: usize) -> Self {
        let mut viewports = Vec::with_capacity(num_viewports);
        for i in 0..num_viewports {
            viewports.push(Viewport::new(&format!("Viewport {}", i + 1)));
        }
        Self {
            viewports,
            active: 0,
            viewport_size: (512, 512),
            marked_points: Vec::new(),
        }
    }

    fn add_point(&mut self, point: Vec3) {
        self.marked_points.push(point);
    }

    /// Render all viewports to a single image
    fn render(&self) -> RgbImage {
        let num = self.viewports.len();
        let (vp_w, vp_h) = self.viewport_size;

        // Calculate grid layout
        let (cols, rows) = match num {
            1 => (1, 1),
            2 => (2, 1),
            3 => (3, 1),
            4 => (2, 2),
            5 | 6 => (3, 2),
            _ => (4, 2),
        };

        let img_width = cols * vp_w + (cols - 1) * 4; // 4px gap
        let img_height = rows * vp_h + (rows - 1) * 4;

        let mut output = RgbImage::new(img_width, img_height);

        // Fill with dark gray background
        for pixel in output.pixels_mut() {
            *pixel = Rgb([30, 30, 30]);
        }

        // Render each viewport
        for (i, viewport) in self.viewports.iter().enumerate() {
            let col = i % cols as usize;
            let row = i / cols as usize;
            let x_offset = col as u32 * (vp_w + 4);
            let y_offset = row as u32 * (vp_h + 4);

            self.render_viewport(&mut output, viewport, i, x_offset, y_offset);
        }

        output
    }

    fn render_viewport(
        &self,
        output: &mut RgbImage,
        viewport: &Viewport,
        viewport_idx: usize,
        x_offset: u32,
        y_offset: u32,
    ) {
        let (vp_w, vp_h) = self.viewport_size;

        // Render image if present
        if let Some(ref image) = viewport.image {
            let rendered = self.render_dicom_image(
                image,
                viewport.window_center,
                viewport.window_width,
                vp_w,
                vp_h,
            );

            // Copy to output
            for y in 0..vp_h {
                for x in 0..vp_w {
                    let pixel = rendered.get_pixel(x, y);
                    output.put_pixel(x + x_offset, y + y_offset, *pixel);
                }
            }

            // Calculate scale for drawing overlays
            let scale_x = vp_w as f64 / image.width as f64;
            let scale_y = vp_h as f64 / image.height as f64;
            let scale = scale_x.min(scale_y);
            let offset_x = (vp_w as f64 - image.width as f64 * scale) / 2.0;
            let offset_y = (vp_h as f64 - image.height as f64 * scale) / 2.0;

            // Draw reference lines from active viewport
            if viewport_idx != self.active {
                if let Some(active_plane) = self.viewports[self.active].image_plane() {
                    if let Some(this_plane) = viewport.image_plane() {
                        if let Some(ref_line) = compute_reference_line(this_plane, active_plane) {
                            self.draw_line(
                                output,
                                ref_line.start.x * scale + offset_x + x_offset as f64,
                                ref_line.start.y * scale + offset_y + y_offset as f64,
                                ref_line.end.x * scale + offset_x + x_offset as f64,
                                ref_line.end.y * scale + offset_y + y_offset as f64,
                                Rgb([255, 255, 0]), // Yellow
                            );
                        }
                    }
                }
            }

            // Draw marked points
            if let Some(plane) = viewport.image_plane() {
                for point in &self.marked_points {
                    if let Some(pixel) = compute_point_in_plane(point, plane) {
                        let screen_x = pixel.x * scale + offset_x + x_offset as f64;
                        let screen_y = pixel.y * scale + offset_y + y_offset as f64;
                        self.draw_crosshair(output, screen_x, screen_y, Rgb([0, 255, 255]));
                    }
                }
            }
        } else {
            // No image - fill with black
            for y in 0..vp_h {
                for x in 0..vp_w {
                    output.put_pixel(x + x_offset, y + y_offset, Rgb([0, 0, 0]));
                }
            }
        }

        // Draw viewport border (highlight active)
        let border_color = if viewport_idx == self.active {
            Rgb([100, 149, 237]) // Cornflower blue
        } else {
            Rgb([80, 80, 80])
        };
        self.draw_rect_outline(output, x_offset, y_offset, vp_w, vp_h, border_color);

        // Draw label
        self.draw_label(output, x_offset + 5, y_offset + 5, &viewport.label);
    }

    fn render_dicom_image(
        &self,
        image: &DicomImage,
        window_center: f64,
        window_width: f64,
        target_w: u32,
        target_h: u32,
    ) -> RgbImage {
        let mut output = RgbImage::new(target_w, target_h);

        // Calculate scaling to fit
        let scale_x = target_w as f64 / image.width as f64;
        let scale_y = target_h as f64 / image.height as f64;
        let scale = scale_x.min(scale_y);

        let offset_x = ((target_w as f64 - image.width as f64 * scale) / 2.0) as i32;
        let offset_y = ((target_h as f64 - image.height as f64 * scale) / 2.0) as i32;

        let min_val = window_center - window_width / 2.0;
        let max_val = window_center + window_width / 2.0;

        for ty in 0..target_h {
            for tx in 0..target_w {
                // Map target pixel to source
                let sx = ((tx as i32 - offset_x) as f64 / scale) as i32;
                let sy = ((ty as i32 - offset_y) as f64 / scale) as i32;

                let gray =
                    if sx >= 0 && sx < image.width as i32 && sy >= 0 && sy < image.height as i32 {
                        let idx = (sy as u32 * image.width + sx as u32) as usize;
                        if idx < image.pixels.len() {
                            let raw = image.pixels[idx] as f64;
                            let value = raw * image.rescale_slope + image.rescale_intercept;

                            let normalized = if value <= min_val {
                                0.0
                            } else if value >= max_val {
                                1.0
                            } else {
                                (value - min_val) / window_width
                            };

                            (normalized * 255.0).clamp(0.0, 255.0) as u8
                        } else {
                            0
                        }
                    } else {
                        0
                    };

                output.put_pixel(tx, ty, Rgb([gray, gray, gray]));
            }
        }

        output
    }

    fn draw_line(&self, img: &mut RgbImage, x1: f64, y1: f64, x2: f64, y2: f64, color: Rgb<u8>) {
        // Bresenham's line algorithm
        let dx = (x2 - x1).abs();
        let dy = (y2 - y1).abs();
        let sx = if x1 < x2 { 1.0 } else { -1.0 };
        let sy = if y1 < y2 { 1.0 } else { -1.0 };
        let mut err = dx - dy;

        let mut x = x1;
        let mut y = y1;

        let (w, h) = img.dimensions();

        loop {
            if x >= 0.0 && x < w as f64 && y >= 0.0 && y < h as f64 {
                img.put_pixel(x as u32, y as u32, color);
                // Make line thicker
                if x + 1.0 < w as f64 {
                    img.put_pixel(x as u32 + 1, y as u32, color);
                }
                if y + 1.0 < h as f64 {
                    img.put_pixel(x as u32, y as u32 + 1, color);
                }
            }

            if (x - x2).abs() < 1.0 && (y - y2).abs() < 1.0 {
                break;
            }

            let e2 = 2.0 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }

    fn draw_crosshair(&self, img: &mut RgbImage, cx: f64, cy: f64, color: Rgb<u8>) {
        let (w, h) = img.dimensions();
        let size: i32 = 10;

        // Horizontal line
        for dx in -size..=size {
            let x = cx as i32 + dx;
            let y = cy as i32;
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }

        // Vertical line
        for dy in -size..=size {
            let x = cx as i32;
            let y = cy as i32 + dy;
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }

        // Circle around crosshair
        let radius = 8.0;
        for angle in 0..360 {
            let rad = (angle as f64) * std::f64::consts::PI / 180.0;
            let x = (cx + radius * rad.cos()) as i32;
            let y = (cy + radius * rad.sin()) as i32;
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }

    fn draw_rect_outline(
        &self,
        img: &mut RgbImage,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        color: Rgb<u8>,
    ) {
        let (iw, ih) = img.dimensions();

        // Top and bottom
        for dx in 0..w {
            if x + dx < iw {
                if y < ih {
                    img.put_pixel(x + dx, y, color);
                }
                if y + h - 1 < ih {
                    img.put_pixel(x + dx, y + h - 1, color);
                }
            }
        }

        // Left and right
        for dy in 0..h {
            if y + dy < ih {
                if x < iw {
                    img.put_pixel(x, y + dy, color);
                }
                if x + w - 1 < iw {
                    img.put_pixel(x + w - 1, y + dy, color);
                }
            }
        }
    }

    fn draw_label(&self, img: &mut RgbImage, x: u32, y: u32, _text: &str) {
        // Simple label background
        let (iw, ih) = img.dimensions();
        for dy in 0..14 {
            for dx in 0..100 {
                if x + dx < iw && y + dy < ih {
                    let pixel = img.get_pixel_mut(x + dx, y + dy);
                    // Semi-transparent black background
                    pixel[0] /= 3;
                    pixel[1] /= 3;
                    pixel[2] /= 3;
                }
            }
        }
        // Note: actual text rendering would require a font library
        // For now, labels are shown in terminal output
    }
}

// ============================================================================
// Command implementations
// ============================================================================

fn cmd_view(args: &[String]) -> Result<()> {
    let mut dicom_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("view.png");
    let mut plane = AnatomicalPlane::Axial;
    let mut slice_idx: Option<usize> = None;
    let mut window_preset: Option<String> = None;
    let mut series_idx: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--plane" => {
                if i + 1 < args.len() {
                    plane = match args[i + 1].to_lowercase().as_str() {
                        "axial" | "ax" => AnatomicalPlane::Axial,
                        "coronal" | "cor" => AnatomicalPlane::Coronal,
                        "sagittal" | "sag" => AnatomicalPlane::Sagittal,
                        _ => AnatomicalPlane::Axial,
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--slice" => {
                if i + 1 < args.len() {
                    slice_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--window" => {
                if i + 1 < args.len() {
                    window_preset = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--series" => {
                if i + 1 < args.len() {
                    series_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            path if !path.starts_with('-') => {
                dicom_path = Some(PathBuf::from(path));
                i += 1;
            }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    println!("Loading DICOM from: {}", dicom_path.display());

    // Load and group by series
    let files = load_dicom_dir(&dicom_path)?;
    let series_list = group_files_by_series(files);

    if series_list.is_empty() {
        anyhow::bail!("No series found");
    }

    // Filter to series with enough images for MPR
    let mpr_series: Vec<_> = series_list.iter().filter(|s| s.files.len() >= 10).collect();

    if mpr_series.is_empty() {
        anyhow::bail!("No series with enough images for MPR (need >= 10)");
    }

    // Select series (default to largest)
    let selected = if let Some(idx) = series_idx {
        mpr_series
            .get(idx.saturating_sub(1))
            .copied()
            .context(format!(
                "Series {} not found (have {} series)",
                idx,
                mpr_series.len()
            ))?
    } else {
        // Find largest series suitable for MPR
        mpr_series
            .iter()
            .max_by_key(|s| s.files.len())
            .copied()
            .unwrap()
    };

    println!(
        "Selected series: {} ({} images)",
        selected.display_name,
        selected.files.len()
    );

    // Load images from selected series
    let images = load_images_from_paths(&selected.files)?;

    if images.len() < 2 {
        anyhow::bail!("Need at least 2 images for volume reconstruction");
    }

    println!("Loaded {} images, building volume...", images.len());
    let volume = Volume::from_series(&images).context("Failed to build volume")?;

    println!(
        "Volume: {:?} acquisition, {}x{}x{} voxels",
        volume.acquisition_orientation,
        volume.dimensions.0,
        volume.dimensions.1,
        volume.dimensions.2
    );

    // Create viewer with 3 MPR viewports
    let mut viewer = ViewerState::new(3);

    // Generate MPR series for each plane
    let planes = [
        AnatomicalPlane::Axial,
        AnatomicalPlane::Coronal,
        AnatomicalPlane::Sagittal,
    ];

    for (i, &p) in planes.iter().enumerate() {
        if let Some(series) = MprSeries::generate(&volume, p) {
            let idx = slice_idx.unwrap_or(series.len() / 2);
            let idx = idx.min(series.len().saturating_sub(1));

            if let Some(img) = series.images.get(idx) {
                viewer.viewports[i].set_image(img.clone());
                viewer.viewports[i].label = format!("{:?} {}/{}", p, idx + 1, series.len());

                if let Some(ref preset_name) = window_preset {
                    if let Some(preset) = get_preset(preset_name) {
                        viewer.viewports[i].set_window(preset.center, preset.width);
                    }
                }
            }
        }
    }

    // Set active viewport based on requested plane
    viewer.active = match plane {
        AnatomicalPlane::Axial => 0,
        AnatomicalPlane::Coronal => 1,
        AnatomicalPlane::Sagittal => 2,
        _ => 0,
    };

    // Render and save
    let output = viewer.render();
    output.save(&output_path)?;

    println!("Saved to: {}", output_path.display());
    println!("\nViewport info:");
    for (i, vp) in viewer.viewports.iter().enumerate() {
        let active_marker = if i == viewer.active { " [ACTIVE]" } else { "" };
        println!(
            "  {}: WC={:.0} WW={:.0}{}",
            vp.label, vp.window_center, vp.window_width, active_marker
        );
    }

    Ok(())
}

fn cmd_point(args: &[String]) -> Result<()> {
    let mut dicom_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("point.png");
    let mut point: Option<Vec3> = None;
    let mut window_preset: Option<String> = None;
    let mut series_idx: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--xyz" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 3 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[0].parse::<f64>(),
                            parts[1].parse::<f64>(),
                            parts[2].parse::<f64>(),
                        ) {
                            point = Some(Vec3::new(x, y, z));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--window" => {
                if i + 1 < args.len() {
                    window_preset = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--series" => {
                if i + 1 < args.len() {
                    series_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            path if !path.starts_with('-') => {
                dicom_path = Some(PathBuf::from(path));
                i += 1;
            }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    let point = point.context("No point provided (use --xyz x,y,z)")?;

    println!("Loading DICOM from: {}", dicom_path.display());
    println!(
        "Marking point: ({:.1}, {:.1}, {:.1})",
        point.x, point.y, point.z
    );

    // Load and select series
    let (volume, _series_name) = load_volume_from_dir(&dicom_path, series_idx)?;

    // Create viewer with 3 MPR viewports
    let mut viewer = ViewerState::new(3);
    viewer.add_point(point);

    // For each plane, find the slice that contains this point
    let planes = [
        AnatomicalPlane::Axial,
        AnatomicalPlane::Coronal,
        AnatomicalPlane::Sagittal,
    ];

    for (i, &plane) in planes.iter().enumerate() {
        if let Some(series) = MprSeries::generate(&volume, plane) {
            // Find slice closest to point
            let slice_idx = find_slice_for_point(&point, &series, &volume, plane);

            if let Some(img) = series.images.get(slice_idx) {
                viewer.viewports[i].set_image(img.clone());
                viewer.viewports[i].label =
                    format!("{:?} {}/{}", plane, slice_idx + 1, series.len());

                if let Some(ref preset_name) = window_preset {
                    if let Some(preset) = get_preset(preset_name) {
                        viewer.viewports[i].set_window(preset.center, preset.width);
                    }
                }

                // Print where point appears
                if let Some(plane_geom) = img.image_plane.as_ref() {
                    if let Some(pixel) = compute_point_in_plane(&point, plane_geom) {
                        println!(
                            "  {:?}: slice {}, pixel ({:.1}, {:.1})",
                            plane, slice_idx, pixel.x, pixel.y
                        );
                    }
                }
            }
        }
    }

    // Render and save
    let output = viewer.render();
    output.save(&output_path)?;
    println!("Saved to: {}", output_path.display());

    Ok(())
}

fn cmd_compare(args: &[String]) -> Result<()> {
    let mut series_paths: Vec<PathBuf> = Vec::new();
    let mut output_path = PathBuf::from("compare.png");
    let mut sync_enabled = false;
    let mut slice_idx: Option<usize> = None;
    let mut window_preset: Option<String> = None;
    let mut point: Option<Vec3> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--sync" => {
                sync_enabled = true;
                i += 1;
            }
            "--slice" => {
                if i + 1 < args.len() {
                    slice_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--window" => {
                if i + 1 < args.len() {
                    window_preset = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--xyz" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 3 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[0].parse::<f64>(),
                            parts[1].parse::<f64>(),
                            parts[2].parse::<f64>(),
                        ) {
                            point = Some(Vec3::new(x, y, z));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            path if !path.starts_with('-') => {
                series_paths.push(PathBuf::from(path));
                i += 1;
            }
            _ => i += 1,
        }
    }

    if series_paths.is_empty() {
        anyhow::bail!("No series paths provided");
    }

    println!(
        "Comparing {} series (sync={})",
        series_paths.len(),
        sync_enabled
    );

    let mut viewer = ViewerState::new(series_paths.len());

    if let Some(p) = point {
        viewer.add_point(p);
    }

    for (i, path) in series_paths.iter().enumerate() {
        println!("Loading: {}", path.display());

        let files = load_dicom_dir(path)?;
        let images = load_images(&files)?;

        if images.is_empty() {
            println!("  No images found, skipping");
            continue;
        }

        // Use middle slice or specified slice
        let idx = slice_idx.unwrap_or(images.len() / 2);
        let idx = idx.min(images.len().saturating_sub(1));

        let img = images[idx].clone();
        let series_desc = img
            .series_description
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());

        viewer.viewports[i].set_image(img);
        viewer.viewports[i].label = format!("{} {}/{}", series_desc, idx + 1, images.len());
        viewer.viewports[i].series_path = Some(path.clone());

        if let Some(ref preset_name) = window_preset {
            if let Some(preset) = get_preset(preset_name) {
                viewer.viewports[i].set_window(preset.center, preset.width);
            }
        }

        println!(
            "  Loaded slice {}/{}: {}",
            idx + 1,
            images.len(),
            series_desc
        );
    }

    // Render and save
    let output = viewer.render();
    output.save(&output_path)?;
    println!("Saved to: {}", output_path.display());

    Ok(())
}

fn cmd_roi(args: &[String]) -> Result<()> {
    let mut dicom_path: Option<PathBuf> = None;
    let mut center: Option<(f64, f64)> = None;
    let mut radius = 10.0;
    let mut slice_idx: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--center" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(x), Ok(y)) = (parts[0].parse(), parts[1].parse()) {
                            center = Some((x, y));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--radius" => {
                if i + 1 < args.len() {
                    radius = args[i + 1].parse().unwrap_or(10.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--slice" => {
                if i + 1 < args.len() {
                    slice_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            path if !path.starts_with('-') => {
                dicom_path = Some(PathBuf::from(path));
                i += 1;
            }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    let (cx, cy) = center.context("No center provided (use --center x,y)")?;

    println!("Loading DICOM from: {}", dicom_path.display());

    let files = load_dicom_dir(&dicom_path)?;
    let images = load_images(&files)?;

    let idx = slice_idx.unwrap_or(images.len() / 2);
    let idx = idx.min(images.len().saturating_sub(1));
    let img = &images[idx];

    println!(
        "Slice {}/{}, ROI center=({}, {}), radius={}",
        idx + 1,
        images.len(),
        cx,
        cy,
        radius
    );

    if let Some(stats) = compute_circle_roi_stats(
        &img.pixels,
        img.width,
        img.height,
        cx,
        cy,
        radius,
        img.rescale_slope,
        img.rescale_intercept,
    ) {
        println!("\nROI Statistics:");
        println!("  Mean:   {:.1}", stats.mean);
        println!("  SD:     {:.1}", stats.std_dev);
        println!("  Min:    {:.1}", stats.min);
        println!("  Max:    {:.1}", stats.max);
        println!("  Pixels: {}", stats.pixel_count);

        // Interpret for CT
        if img.modality.as_deref() == Some("CT") {
            println!("\nCT Interpretation:");
            if stats.mean < -500.0 {
                println!("  Density: Air/Lung");
            } else if stats.mean < -50.0 {
                println!("  Density: Fat");
            } else if stats.mean < 20.0 {
                println!("  Density: Water/Fluid");
            } else if stats.mean < 80.0 {
                println!("  Density: Soft Tissue");
            } else if stats.mean < 200.0 {
                println!("  Density: Blood/Contrast");
            } else {
                println!("  Density: Bone/Calcification");
            }
        }
    } else {
        println!("Failed to compute ROI stats (ROI may be outside image)");
    }

    Ok(())
}

fn cmd_measure(args: &[String]) -> Result<()> {
    let mut dicom_path: Option<PathBuf> = None;
    let mut from: Option<(f64, f64)> = None;
    let mut to: Option<(f64, f64)> = None;
    let mut slice_idx: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--from" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(x), Ok(y)) = (parts[0].parse(), parts[1].parse()) {
                            from = Some((x, y));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--to" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(x), Ok(y)) = (parts[0].parse(), parts[1].parse()) {
                            to = Some((x, y));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--slice" => {
                if i + 1 < args.len() {
                    slice_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            path if !path.starts_with('-') => {
                dicom_path = Some(PathBuf::from(path));
                i += 1;
            }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    let from = from.context("No start point (use --from x,y)")?;
    let to = to.context("No end point (use --to x,y)")?;

    let files = load_dicom_dir(&dicom_path)?;
    let images = load_images(&files)?;

    let idx = slice_idx.unwrap_or(images.len() / 2);
    let idx = idx.min(images.len().saturating_sub(1));
    let img = &images[idx];

    println!("Slice {}/{}", idx + 1, images.len());
    println!(
        "From: ({:.1}, {:.1}) to ({:.1}, {:.1})",
        from.0, from.1, to.0, to.1
    );

    if let Some(spacing) = img.pixel_spacing {
        let dist = compute_distance_mm(from, to, spacing);
        println!("\nDistance: {:.1} mm", dist);
        println!("Pixel spacing: {:.3} x {:.3} mm", spacing.0, spacing.1);
    } else {
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;
        let dist = (dx * dx + dy * dy).sqrt();
        println!(
            "\nDistance: {:.1} pixels (no pixel spacing available)",
            dist
        );
    }

    Ok(())
}

fn cmd_info(args: &[String]) -> Result<()> {
    let dicom_path = args
        .first()
        .map(PathBuf::from)
        .context("No DICOM path provided")?;

    println!("Loading DICOM from: {}", dicom_path.display());

    let files = load_dicom_dir(&dicom_path)?;
    println!("Found {} DICOM files", files.len());

    // Group by series
    let series_list = group_files_by_series(files);
    println!("\n{} series found:\n", series_list.len());

    for (i, series) in series_list.iter().enumerate() {
        println!(
            "{}. {} ({} images)",
            i + 1,
            series.display_name,
            series.files.len()
        );
        println!("   Modality: {:?}", series.modality);
        println!("   Orientation: {:?}", series.orientation);
        println!(
            "   Frame of Reference: {:?}",
            series
                .frame_of_reference_uid
                .as_ref()
                .map(|s| &s[s.len().saturating_sub(12)..])
        );

        // Check if suitable for MPR
        if series.files.len() >= 2 {
            let images = load_images_from_paths(&series.files[..2.min(series.files.len())])?;
            if images.len() >= 2 {
                if let (Some(p1), Some(p2)) = (
                    images[0].image_plane.as_ref(),
                    images[1].image_plane.as_ref(),
                ) {
                    if p1.is_parallel(p2) && p1.same_frame_of_reference(p2) {
                        println!("   MPR: Suitable for volume reconstruction");
                    }
                }
            }
        }
        println!();
    }

    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

fn load_dicom_dir(dir: &PathBuf) -> Result<Vec<DicomFile>> {
    let mut files = Vec::new();
    let mut rejected_non_tuli = 0;

    for entry in walkdir::WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Ok(dcm) = DicomFile::open(path) {
                // SAFETY: Only open anonymized test images (Patient ID contains "TULI")
                let patient_id = dcm.patient_id().unwrap_or_default();
                if !patient_id.contains("TULI") {
                    rejected_non_tuli += 1;
                    continue;
                }

                if dcm.is_image_storage() && dcm.rows().is_some() && dcm.columns().is_some() {
                    files.push(dcm);
                }
            }
        }
    }

    if rejected_non_tuli > 0 {
        println!(
            "Safety: Rejected {} non-anonymized files (Patient ID must contain 'TULI')",
            rejected_non_tuli
        );
    }

    Ok(files)
}

fn load_images(files: &[DicomFile]) -> Result<Vec<DicomImage>> {
    let mut images = Vec::new();
    for file in files {
        match DicomImage::from_file(file) {
            Ok(img) => images.push(img),
            Err(e) => tracing::warn!("Failed to load image {}: {}", file.path, e),
        }
    }
    Ok(images)
}

fn load_images_from_paths(paths: &[PathBuf]) -> Result<Vec<DicomImage>> {
    let mut images = Vec::new();
    for path in paths {
        if let Ok(dcm) = DicomFile::open(path) {
            if let Ok(img) = DicomImage::from_file(&dcm) {
                images.push(img);
            }
        }
    }
    Ok(images)
}

fn cmd_debug(args: &[String]) -> Result<()> {
    let mut dicom_path: Option<PathBuf> = None;
    let mut test_point: Option<Vec3> = None;
    let mut series_idx: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--xyz" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 3 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[0].parse::<f64>(),
                            parts[1].parse::<f64>(),
                            parts[2].parse::<f64>(),
                        ) {
                            test_point = Some(Vec3::new(x, y, z));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--series" => {
                if i + 1 < args.len() {
                    series_idx = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            path if !path.starts_with('-') => {
                dicom_path = Some(PathBuf::from(path));
                i += 1;
            }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    println!("=== MPR GEOMETRY DEBUG ===\n");

    // Load volume
    let (volume, series_name) = load_volume_from_dir(&dicom_path, series_idx)?;

    // Print volume geometry
    println!("VOLUME GEOMETRY ({})", series_name);
    println!(
        "  Dimensions: {}x{}x{} (W x H x D)",
        volume.dimensions.0, volume.dimensions.1, volume.dimensions.2
    );
    println!(
        "  Spacing: {:.3} x {:.3} x {:.3} mm",
        volume.spacing.0, volume.spacing.1, volume.spacing.2
    );
    println!(
        "  Origin: ({:.2}, {:.2}, {:.2})",
        volume.origin.x, volume.origin.y, volume.origin.z
    );
    println!(
        "  Row direction:   ({:.4}, {:.4}, {:.4}) [points along image columns]",
        volume.row_direction.x, volume.row_direction.y, volume.row_direction.z
    );
    println!(
        "  Col direction:   ({:.4}, {:.4}, {:.4}) [points along image rows]",
        volume.col_direction.x, volume.col_direction.y, volume.col_direction.z
    );
    println!(
        "  Slice direction: ({:.4}, {:.4}, {:.4}) [normal to slices]",
        volume.slice_direction.x, volume.slice_direction.y, volume.slice_direction.z
    );
    println!("  Acquisition: {:?}", volume.acquisition_orientation);

    // Calculate volume extent
    let (sx, sy, sz) = volume.spacing;
    let (w, h, d) = volume.dimensions;
    let extent_x = w as f64 * sx;
    let extent_y = h as f64 * sy;
    let extent_z = d as f64 * sz;
    println!(
        "  Physical extent: {:.1} x {:.1} x {:.1} mm",
        extent_x, extent_y, extent_z
    );

    // Calculate volume center in patient coordinates
    let center_offset = volume
        .row_direction
        .scale(w as f64 * sx / 2.0)
        .add(&volume.col_direction.scale(h as f64 * sy / 2.0))
        .add(&volume.slice_direction.scale(d as f64 * sz / 2.0));
    let volume_center = volume.origin.add(&center_offset);
    println!(
        "  Volume center: ({:.2}, {:.2}, {:.2})",
        volume_center.x, volume_center.y, volume_center.z
    );

    // Use provided point or volume center
    let test_pt = test_point.unwrap_or(volume_center);
    println!(
        "\nTEST POINT: ({:.2}, {:.2}, {:.2})",
        test_pt.x, test_pt.y, test_pt.z
    );

    // Generate and analyze each MPR plane
    let planes = [
        (AnatomicalPlane::Axial, "AXIAL"),
        (AnatomicalPlane::Coronal, "CORONAL"),
        (AnatomicalPlane::Sagittal, "SAGITTAL"),
    ];

    for (plane, name) in &planes {
        println!("\n--- {} PLANE ---", name);

        if let Some(series) = MprSeries::generate(&volume, *plane) {
            println!("  Generated {} slices", series.len());

            // Show slice location range for sync debugging
            if let (Some(first), Some(last)) = (series.images.first(), series.images.last()) {
                if let (Some(first_loc), Some(last_loc)) =
                    (first.slice_location, last.slice_location)
                {
                    println!(
                        "  Slice location range: {:.2} to {:.2} mm",
                        first_loc, last_loc
                    );
                }
            }

            // Find slice containing test point
            let slice_idx = find_slice_for_point(&test_pt, &series, &volume, *plane);
            println!("  Best slice for point: {}", slice_idx);

            if let Some(img) = series.images.get(slice_idx) {
                println!("  Image size: {}x{}", img.width, img.height);

                if let Some(ref img_plane) = img.image_plane {
                    println!("  ImagePlane geometry:");
                    println!(
                        "    Position: ({:.2}, {:.2}, {:.2})",
                        img_plane.position.x, img_plane.position.y, img_plane.position.z
                    );
                    println!(
                        "    Row dir:  ({:.4}, {:.4}, {:.4})",
                        img_plane.row_direction.x,
                        img_plane.row_direction.y,
                        img_plane.row_direction.z
                    );
                    println!(
                        "    Col dir:  ({:.4}, {:.4}, {:.4})",
                        img_plane.col_direction.x,
                        img_plane.col_direction.y,
                        img_plane.col_direction.z
                    );
                    println!(
                        "    Normal:   ({:.4}, {:.4}, {:.4})",
                        img_plane.normal.x, img_plane.normal.y, img_plane.normal.z
                    );
                    println!(
                        "    Pixel spacing: {:.3} x {:.3} mm",
                        img_plane.pixel_spacing.0, img_plane.pixel_spacing.1
                    );

                    // Project test point to this plane
                    let pixel = img_plane.patient_to_pixel(&test_pt);
                    println!("  Point projection:");
                    println!("    Pixel coords: ({:.2}, {:.2})", pixel.x, pixel.y);
                    // dimensions is (rows, cols), so cols=dimensions.1, rows=dimensions.0
                    let (rows, cols) = img_plane.dimensions;
                    println!(
                        "    In bounds: x=[0..{}] (cols), y=[0..{}] (rows)",
                        cols, rows
                    );

                    let in_bounds = pixel.x >= 0.0
                        && pixel.x < cols as f64
                        && pixel.y >= 0.0
                        && pixel.y < rows as f64;
                    println!(
                        "    Result: {}",
                        if in_bounds {
                            "IN BOUNDS"
                        } else {
                            "OUT OF BOUNDS"
                        }
                    );

                    if !in_bounds {
                        // Debug: calculate step by step
                        println!("  Debug calculation:");
                        let v = test_pt.sub(&img_plane.position);
                        println!(
                            "    Vector from plane origin to point: ({:.2}, {:.2}, {:.2})",
                            v.x, v.y, v.z
                        );
                        let col = v.dot(&img_plane.row_direction) / img_plane.pixel_spacing.1;
                        let row = v.dot(&img_plane.col_direction) / img_plane.pixel_spacing.0;
                        println!("    Dot with row_dir / col_spacing = {:.2} (pixel X)", col);
                        println!("    Dot with col_dir / row_spacing = {:.2} (pixel Y)", row);
                    }
                } else {
                    println!("  WARNING: No ImagePlane computed for this slice!");
                }
            }
        } else {
            println!("  ERROR: Failed to generate MPR series");
        }
    }

    // Also test with original images for comparison
    println!("\n--- ORIGINAL IMAGES (for comparison) ---");
    let files = load_dicom_dir(&dicom_path)?;
    let all_series = group_files_by_series(files);

    for series_info in all_series.iter().take(1) {
        if series_info.files.len() >= 2 {
            let images = load_images_from_paths(&series_info.files[..2])?;
            if let Some(img) = images.first() {
                if let Some(ref img_plane) = img.image_plane {
                    println!("  Original {} image:", series_info.display_name);
                    println!(
                        "    Position: ({:.2}, {:.2}, {:.2})",
                        img_plane.position.x, img_plane.position.y, img_plane.position.z
                    );
                    println!(
                        "    Row dir:  ({:.4}, {:.4}, {:.4})",
                        img_plane.row_direction.x,
                        img_plane.row_direction.y,
                        img_plane.row_direction.z
                    );
                    println!(
                        "    Col dir:  ({:.4}, {:.4}, {:.4})",
                        img_plane.col_direction.x,
                        img_plane.col_direction.y,
                        img_plane.col_direction.z
                    );
                    println!(
                        "    Normal:   ({:.4}, {:.4}, {:.4})",
                        img_plane.normal.x, img_plane.normal.y, img_plane.normal.z
                    );
                }
            }
        }
    }

    Ok(())
}

/// Load a volume from a DICOM directory, selecting appropriate series
fn load_volume_from_dir(dir: &PathBuf, series_idx: Option<usize>) -> Result<(Volume, String)> {
    let files = load_dicom_dir(dir)?;
    let series_list = group_files_by_series(files);

    if series_list.is_empty() {
        anyhow::bail!("No series found");
    }

    // Filter to series with enough images for MPR
    let mpr_series: Vec<_> = series_list.iter().filter(|s| s.files.len() >= 10).collect();

    if mpr_series.is_empty() {
        anyhow::bail!("No series with enough images for MPR (need >= 10)");
    }

    // Select series
    let selected = if let Some(idx) = series_idx {
        mpr_series
            .get(idx.saturating_sub(1))
            .copied()
            .context(format!(
                "Series {} not found (have {} series)",
                idx,
                mpr_series.len()
            ))?
    } else {
        mpr_series
            .iter()
            .max_by_key(|s| s.files.len())
            .copied()
            .unwrap()
    };

    println!(
        "Selected series: {} ({} images)",
        selected.display_name,
        selected.files.len()
    );

    let images = load_images_from_paths(&selected.files)?;
    let volume = Volume::from_series(&images).context("Failed to build volume")?;

    Ok((volume, selected.display_name.clone()))
}

fn find_slice_for_point(
    point: &Vec3,
    series: &MprSeries,
    volume: &Volume,
    plane: AnatomicalPlane,
) -> usize {
    // Find slice whose plane contains (or is closest to) the point
    let count = series.len();
    let mut best_idx = count / 2;
    let mut best_dist = f64::MAX;

    for i in 0..count {
        if let Some(img) = series.images.get(i) {
            if let Some(img_plane) = img.image_plane.as_ref() {
                // Distance from point to plane
                let v = point.sub(&img_plane.position);
                let dist = v.dot(&img_plane.normal).abs();

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
        }
    }

    best_idx
}

fn print_help() {
    println!("Claude DICOM Viewer - Sandbox for testing and image analysis");
    println!();
    println!("USAGE:");
    println!("    mpr_sandbox <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    view <path>      View MPR planes from a DICOM series");
    println!("    point <path>     Mark a 3D point across all planes");
    println!("    compare <paths>  Compare multiple series side-by-side");
    println!("    roi <path>       Calculate ROI statistics");
    println!("    measure <path>   Measure distance between points");
    println!("    info <path>      Show DICOM directory information");
    println!();
    println!("OPTIONS:");
    println!("    -o, --output     Output PNG path (default: <command>.png)");
    println!("    --plane          Plane to view: axial, coronal, sagittal");
    println!("    --slice          Slice index (default: middle)");
    println!("    --window         Window preset: brain, stroke, bone, soft, lung, abdomen");
    println!("    --xyz            3D point in patient coordinates: x,y,z");
    println!("    --center         2D pixel coordinates for ROI: x,y");
    println!("    --radius         ROI radius in pixels (default: 10)");
    println!("    --from/--to      Measurement endpoints: x,y");
    println!("    --sync           Enable sync for multi-series comparison");
    println!();
    println!("EXAMPLES:");
    println!("    mpr_sandbox view ~/dicom/brain -o brain.png --window brain");
    println!("    mpr_sandbox point ~/dicom/brain --xyz -50,30,120 -o point.png");
    println!("    mpr_sandbox compare ~/dwi ~/adc --sync -o stroke.png");
    println!("    mpr_sandbox roi ~/dicom/brain --center 256,256 --radius 15");
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("sauhu=warn".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_help();
        return Ok(());
    }

    match args[0].as_str() {
        "view" => cmd_view(&args[1..]),
        "point" => cmd_point(&args[1..]),
        "compare" => cmd_compare(&args[1..]),
        "roi" => cmd_roi(&args[1..]),
        "measure" => cmd_measure(&args[1..]),
        "info" => cmd_info(&args[1..]),
        "debug" => cmd_debug(&args[1..]),
        "sync" => cmd_sync(&args[1..]),
        "coreg" => cmd_coreg(&args[1..]),
        "coreg-gpu" => cmd_coreg_gpu(&args[1..]),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => {
            println!("Unknown command: {}", args[0]);
            print_help();
            Ok(())
        }
    }
}

/// Coregistration test command
fn cmd_coreg(args: &[String]) -> Result<()> {
    let dicom_path = args
        .first()
        .map(PathBuf::from)
        .context("No DICOM path provided")?;

    // Parse series indices (default: 4 and 10 for two 3D series)
    let target_idx = args
        .iter()
        .position(|a| a == "--target")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let source_idx = args
        .iter()
        .position(|a| a == "--source")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("Loading DICOM from: {:?}", dicom_path);

    // Load series
    let files = load_dicom_dir(&dicom_path)?;
    let series_list = group_files_by_series(files);
    println!("Found {} series", series_list.len());

    if series_list.len() < 2 {
        anyhow::bail!("Need at least 2 series for coregistration");
    }

    // Get target and source series
    let target_series = series_list
        .get(target_idx - 1)
        .context("Target series index out of range")?;
    let source_series = series_list
        .get(source_idx - 1)
        .context("Source series index out of range")?;

    println!(
        "\nTarget series {}: {} ({} images)",
        target_idx,
        target_series.display_name,
        target_series.files.len()
    );
    println!(
        "Source series {}: {} ({} images)",
        source_idx,
        source_series.display_name,
        source_series.files.len()
    );

    // Build volumes
    println!("\nBuilding target volume...");
    let target_images = load_images_from_paths(&target_series.files)?;
    let target_volume = std::sync::Arc::new(
        Volume::from_series(&target_images).context("Failed to build target volume")?,
    );
    println!("  Dimensions: {:?}", target_volume.dimensions);
    println!("  Spacing: {:?}", target_volume.spacing);

    println!("\nBuilding source volume...");
    let source_images = load_images_from_paths(&source_series.files)?;
    let source_volume = std::sync::Arc::new(
        Volume::from_series(&source_images).context("Failed to build source volume")?,
    );
    println!("  Dimensions: {:?}", source_volume.dimensions);
    println!("  Spacing: {:?}", source_volume.spacing);

    // Run coregistration
    println!("\nRunning coregistration...");
    let config = RegistrationConfig::brain_mri();
    println!("  Metric: {:?}", config.metric);
    println!("  Schedule: {} levels", config.schedule.num_levels());

    let start = std::time::Instant::now();
    let pipeline = RegistrationPipeline::new(config);
    let result = pipeline.run_cpu(&target_volume, &source_volume);

    match result {
        RegistrationResult::Success {
            transform,
            metric,
            elapsed,
            level_results,
        } => {
            let rot = transform.rotation_degrees();
            println!("\n=== COREGISTRATION SUCCESS ===");
            println!("  Final metric: {:.4}", metric);
            println!("  Time: {:?} (pipeline: {:?})", start.elapsed(), elapsed);
            println!("  Rotation: ({:.2}°, {:.2}°, {:.2}°)", rot.x, rot.y, rot.z);
            println!(
                "  Translation: ({:.2}, {:.2}, {:.2}) mm",
                transform.translation_x, transform.translation_y, transform.translation_z
            );
            println!("  Scale: {:.4}", transform.scale);

            println!("\n  Per-level results:");
            for (i, r) in level_results.iter().enumerate() {
                println!(
                    "    Level {}: metric={:.4}, iterations={}",
                    i, r.metric, r.iterations
                );
            }
        }
        RegistrationResult::Failed { error, .. } => {
            println!("\n=== COREGISTRATION FAILED ===");
            println!("  Error: {}", error);
        }
        RegistrationResult::Cancelled => {
            println!("\n=== COREGISTRATION CANCELLED ===");
        }
    }

    Ok(())
}

/// GPU-accelerated coregistration test
/// Usage: coreg-gpu <target_path> [source_path] --target <n> --source <n> [--size <n>] [-o output.png]
/// If source_path is omitted, both series are loaded from target_path
fn cmd_coreg_gpu(args: &[String]) -> Result<()> {
    use pollster::FutureExt;

    // Parse paths - first positional arg is target, second (optional) is source
    let mut positional_args: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with('-') {
            i += 2; // Skip flag and its value
        } else {
            positional_args.push(&args[i]);
            i += 1;
        }
    }

    let target_path = positional_args
        .first()
        .map(|s| PathBuf::from(s.as_str()))
        .context("No target DICOM path provided")?;

    let source_path = positional_args
        .get(1)
        .map(|s| PathBuf::from(s.as_str()))
        .unwrap_or_else(|| target_path.clone());

    let cross_study = source_path != target_path;

    // Parse series indices
    let target_idx = args
        .iter()
        .position(|a| a == "--target")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2); // Default to series 2 (common for T1 3D)

    let source_idx = args
        .iter()
        .position(|a| a == "--source")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    // Downsample size (default 64 for speed)
    let downsample = args
        .iter()
        .position(|a| a == "--size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(64usize);

    // Load target study
    println!("Loading target DICOM from: {:?}", target_path);
    let target_files = load_dicom_dir(&target_path)?;
    let target_series_list = group_files_by_series(target_files);
    println!("Found {} series in target study", target_series_list.len());

    // Load source study (same or different)
    let source_series_list = if cross_study {
        println!("\nLoading source DICOM from: {:?}", source_path);
        let source_files = load_dicom_dir(&source_path)?;
        let list = group_files_by_series(source_files);
        println!("Found {} series in source study", list.len());
        list
    } else {
        target_series_list.clone()
    };

    // Get target and source series
    let target_series = target_series_list
        .get(target_idx - 1)
        .context("Target series index out of range")?;
    let source_series = source_series_list
        .get(source_idx - 1)
        .context("Source series index out of range")?;

    println!(
        "\nTarget series {}: {} ({} images)",
        target_idx,
        target_series.display_name,
        target_series.files.len()
    );
    println!(
        "Source series {}: {} ({} images)",
        source_idx,
        source_series.display_name,
        source_series.files.len()
    );

    // Build volumes
    println!("\nBuilding target volume...");
    let target_images = load_images_from_paths(&target_series.files)?;
    let target_volume =
        Volume::from_series(&target_images).context("Failed to build target volume")?;
    println!("  Dimensions: {:?}", target_volume.dimensions);

    println!("\nBuilding source volume...");
    let source_images = load_images_from_paths(&source_series.files)?;
    let source_volume =
        Volume::from_series(&source_images).context("Failed to build source volume")?;
    println!("  Dimensions: {:?}", source_volume.dimensions);

    // Initialize wgpu
    println!("\nInitializing GPU...");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .block_on()
        .context("Failed to find GPU adapter")?;

    println!("  Adapter: {:?}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Coregistration Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .block_on()
        .context("Failed to create device")?;

    // Preprocess and downsample volumes
    println!("\nPreprocessing volumes to {}³...", downsample);
    let preprocess_start = std::time::Instant::now();

    let target_preprocessed = preprocess_volume_mri(&target_volume.data);
    let source_preprocessed = preprocess_volume_mri(&source_volume.data);

    let target_downsampled = downsample_volume(
        &target_preprocessed,
        target_volume.dimensions,
        (downsample, downsample, downsample),
    );
    let source_downsampled = downsample_volume(
        &source_preprocessed,
        source_volume.dimensions,
        (downsample, downsample, downsample),
    );

    println!("  Preprocessing took: {:?}", preprocess_start.elapsed());

    // Compute spacing for downsampled volumes (preserves physical extent)
    let target_ds_dims = (downsample, downsample, downsample);
    let source_ds_dims = (downsample, downsample, downsample);
    let target_ds_spacing = compute_downsampled_spacing(
        target_volume.dimensions,
        target_volume.spacing,
        target_ds_dims,
    );
    let source_ds_spacing = compute_downsampled_spacing(
        source_volume.dimensions,
        source_volume.spacing,
        source_ds_dims,
    );

    println!(
        "  Target spacing: {:.2}x{:.2}x{:.2} mm",
        target_ds_spacing.0, target_ds_spacing.1, target_ds_spacing.2
    );
    println!(
        "  Source spacing: {:.2}x{:.2}x{:.2} mm",
        source_ds_spacing.0, source_ds_spacing.1, source_ds_spacing.2
    );

    // Create GPU coregistration resources
    println!("\nUploading to GPU...");
    let upload_start = std::time::Instant::now();
    let mut gpu_coreg = GpuCoregistration::new(&device);
    gpu_coreg.upload_volumes(
        &device,
        &queue,
        &target_downsampled,
        &source_downsampled,
        target_ds_dims,
        source_ds_dims,
        target_ds_spacing,
        source_ds_spacing,
    );
    println!("  Upload took: {:?}", upload_start.elapsed());

    // Test initial NCC with identity transform
    let identity_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    println!("\nTesting GPU NCC...");
    let ncc_start = std::time::Instant::now();
    let initial_ncc = gpu_coreg.compute_ncc(&device, &queue, &identity_matrix);
    println!("  Initial NCC (identity): {:.4}", initial_ncc);
    println!("  Single NCC took: {:?}", ncc_start.elapsed());

    // Check for pyramid mode (two-level: 64→128)
    let use_pyramid = args.iter().any(|a| a == "--pyramid");

    // Compute initial alignment using center-of-mass
    println!("\nComputing initial alignment (center-of-mass)...");
    let target_geom = VolumeGeometry::new(target_ds_dims, target_ds_spacing);
    let source_geom = VolumeGeometry::new(source_ds_dims, source_ds_spacing);
    let initial_transform = compute_initial_alignment(
        &target_downsampled,
        &source_downsampled,
        &target_geom,
        &source_geom,
        0.1, // 10% threshold
    );
    println!(
        "  Initial translation: ({:.1}, {:.1}, {:.1}) mm",
        initial_transform.translation_x,
        initial_transform.translation_y,
        initial_transform.translation_z
    );

    // Test NCC with initial alignment
    let initial_matrix = initial_transform.to_inverse_matrix();
    let initial_matrix_f32 = [
        [
            initial_matrix.m[0][0] as f32,
            initial_matrix.m[0][1] as f32,
            initial_matrix.m[0][2] as f32,
            initial_matrix.m[0][3] as f32,
        ],
        [
            initial_matrix.m[1][0] as f32,
            initial_matrix.m[1][1] as f32,
            initial_matrix.m[1][2] as f32,
            initial_matrix.m[1][3] as f32,
        ],
        [
            initial_matrix.m[2][0] as f32,
            initial_matrix.m[2][1] as f32,
            initial_matrix.m[2][2] as f32,
            initial_matrix.m[2][3] as f32,
        ],
        [
            initial_matrix.m[3][0] as f32,
            initial_matrix.m[3][1] as f32,
            initial_matrix.m[3][2] as f32,
            initial_matrix.m[3][3] as f32,
        ],
    ];
    let initial_aligned_ncc = gpu_coreg.compute_ncc(&device, &queue, &initial_matrix_f32);
    println!("  NCC after initial alignment: {:.4}", initial_aligned_ncc);

    // Run optimization loop
    println!("\nRunning GPU-accelerated optimization...");
    let opt_start = std::time::Instant::now();

    let mut transform = initial_transform; // Start from initial alignment, not identity
    let mut final_metric = 0.0;
    let mut total_iterations = 0;

    if use_pyramid && downsample >= 128 {
        // Three-level pyramid: 32³ → 64³ → 128³
        println!("  Using three-level pyramid (32³ → 64³ → 128³)...");

        // Level 1: Very coarse (32³)
        let level1_size = 32;
        let level1_dims = (level1_size, level1_size, level1_size);
        let target_level1 =
            downsample_volume(&target_preprocessed, target_volume.dimensions, level1_dims);
        let source_level1 =
            downsample_volume(&source_preprocessed, source_volume.dimensions, level1_dims);
        let target_level1_spacing = compute_downsampled_spacing(
            target_volume.dimensions,
            target_volume.spacing,
            level1_dims,
        );
        let source_level1_spacing = compute_downsampled_spacing(
            source_volume.dimensions,
            source_volume.spacing,
            level1_dims,
        );

        gpu_coreg.upload_volumes(
            &device,
            &queue,
            &target_level1,
            &source_level1,
            level1_dims,
            level1_dims,
            target_level1_spacing,
            source_level1_spacing,
        );

        let level1_optimizer = PowellOptimizer::new(false).with_steps(0.15, 12.0);
        let level1_result = {
            let metric_fn = |t: &RigidTransform| {
                let inv_matrix = t.to_inverse_matrix();
                let matrix_f32 = [
                    [
                        inv_matrix.m[0][0] as f32,
                        inv_matrix.m[0][1] as f32,
                        inv_matrix.m[0][2] as f32,
                        inv_matrix.m[0][3] as f32,
                    ],
                    [
                        inv_matrix.m[1][0] as f32,
                        inv_matrix.m[1][1] as f32,
                        inv_matrix.m[1][2] as f32,
                        inv_matrix.m[1][3] as f32,
                    ],
                    [
                        inv_matrix.m[2][0] as f32,
                        inv_matrix.m[2][1] as f32,
                        inv_matrix.m[2][2] as f32,
                        inv_matrix.m[2][3] as f32,
                    ],
                    [
                        inv_matrix.m[3][0] as f32,
                        inv_matrix.m[3][1] as f32,
                        inv_matrix.m[3][2] as f32,
                        inv_matrix.m[3][3] as f32,
                    ],
                ];
                gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
            };
            level1_optimizer.optimize(transform, 30, metric_fn)
        };
        transform = level1_result.transform;
        total_iterations += level1_result.iterations;
        println!(
            "    Level 1 (32³): metric={:.4}, iterations={}",
            level1_result.metric, level1_result.iterations
        );

        // Level 2: Medium (64³)
        let level2_size = 64;
        let level2_dims = (level2_size, level2_size, level2_size);
        let target_level2 =
            downsample_volume(&target_preprocessed, target_volume.dimensions, level2_dims);
        let source_level2 =
            downsample_volume(&source_preprocessed, source_volume.dimensions, level2_dims);
        let target_level2_spacing = compute_downsampled_spacing(
            target_volume.dimensions,
            target_volume.spacing,
            level2_dims,
        );
        let source_level2_spacing = compute_downsampled_spacing(
            source_volume.dimensions,
            source_volume.spacing,
            level2_dims,
        );

        gpu_coreg.upload_volumes(
            &device,
            &queue,
            &target_level2,
            &source_level2,
            level2_dims,
            level2_dims,
            target_level2_spacing,
            source_level2_spacing,
        );

        let level2_optimizer = PowellOptimizer::new(false).with_steps(0.08, 6.0);
        let level2_result = {
            let metric_fn = |t: &RigidTransform| {
                let inv_matrix = t.to_inverse_matrix();
                let matrix_f32 = [
                    [
                        inv_matrix.m[0][0] as f32,
                        inv_matrix.m[0][1] as f32,
                        inv_matrix.m[0][2] as f32,
                        inv_matrix.m[0][3] as f32,
                    ],
                    [
                        inv_matrix.m[1][0] as f32,
                        inv_matrix.m[1][1] as f32,
                        inv_matrix.m[1][2] as f32,
                        inv_matrix.m[1][3] as f32,
                    ],
                    [
                        inv_matrix.m[2][0] as f32,
                        inv_matrix.m[2][1] as f32,
                        inv_matrix.m[2][2] as f32,
                        inv_matrix.m[2][3] as f32,
                    ],
                    [
                        inv_matrix.m[3][0] as f32,
                        inv_matrix.m[3][1] as f32,
                        inv_matrix.m[3][2] as f32,
                        inv_matrix.m[3][3] as f32,
                    ],
                ];
                gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
            };
            level2_optimizer.optimize(transform, 25, metric_fn)
        };
        transform = level2_result.transform;
        total_iterations += level2_result.iterations;
        println!(
            "    Level 2 (64³): metric={:.4}, iterations={}",
            level2_result.metric, level2_result.iterations
        );

        // Level 3: Fine (128³)
        gpu_coreg.upload_volumes(
            &device,
            &queue,
            &target_downsampled,
            &source_downsampled,
            target_ds_dims,
            source_ds_dims,
            target_ds_spacing,
            source_ds_spacing,
        );

        let level3_optimizer = PowellOptimizer::new(false).with_steps(0.03, 2.0);
        let level3_result = {
            let metric_fn = |t: &RigidTransform| {
                let inv_matrix = t.to_inverse_matrix();
                let matrix_f32 = [
                    [
                        inv_matrix.m[0][0] as f32,
                        inv_matrix.m[0][1] as f32,
                        inv_matrix.m[0][2] as f32,
                        inv_matrix.m[0][3] as f32,
                    ],
                    [
                        inv_matrix.m[1][0] as f32,
                        inv_matrix.m[1][1] as f32,
                        inv_matrix.m[1][2] as f32,
                        inv_matrix.m[1][3] as f32,
                    ],
                    [
                        inv_matrix.m[2][0] as f32,
                        inv_matrix.m[2][1] as f32,
                        inv_matrix.m[2][2] as f32,
                        inv_matrix.m[2][3] as f32,
                    ],
                    [
                        inv_matrix.m[3][0] as f32,
                        inv_matrix.m[3][1] as f32,
                        inv_matrix.m[3][2] as f32,
                        inv_matrix.m[3][3] as f32,
                    ],
                ];
                gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
            };
            level3_optimizer.optimize(transform, 20, metric_fn)
        };
        transform = level3_result.transform;
        final_metric = level3_result.metric;
        total_iterations += level3_result.iterations;
        println!(
            "    Level 3 (128³): metric={:.4}, iterations={}",
            level3_result.metric, level3_result.iterations
        );
    } else {
        // Single level optimization
        let schedule = PyramidSchedule::fast();
        let (iterations, rot_step, trans_step) = schedule.for_level(0);

        let optimizer = PowellOptimizer::new(false).with_steps(rot_step, trans_step);

        let metric_fn = |t: &RigidTransform| {
            let inv_matrix = t.to_inverse_matrix();
            let matrix_f32 = [
                [
                    inv_matrix.m[0][0] as f32,
                    inv_matrix.m[0][1] as f32,
                    inv_matrix.m[0][2] as f32,
                    inv_matrix.m[0][3] as f32,
                ],
                [
                    inv_matrix.m[1][0] as f32,
                    inv_matrix.m[1][1] as f32,
                    inv_matrix.m[1][2] as f32,
                    inv_matrix.m[1][3] as f32,
                ],
                [
                    inv_matrix.m[2][0] as f32,
                    inv_matrix.m[2][1] as f32,
                    inv_matrix.m[2][2] as f32,
                    inv_matrix.m[2][3] as f32,
                ],
                [
                    inv_matrix.m[3][0] as f32,
                    inv_matrix.m[3][1] as f32,
                    inv_matrix.m[3][2] as f32,
                    inv_matrix.m[3][3] as f32,
                ],
            ];
            gpu_coreg.compute_ncc(&device, &queue, &matrix_f32)
        };

        let result = optimizer.optimize(transform, iterations, metric_fn);
        transform = result.transform;
        final_metric = result.metric;
        total_iterations = result.iterations;
    }

    let opt_elapsed = opt_start.elapsed();
    let rot = transform.rotation_degrees();

    println!("\n=== GPU COREGISTRATION RESULT ===");
    println!("  Final metric: {:.4}", final_metric);
    println!("  Optimization time: {:?}", opt_elapsed);
    println!("  Iterations: {}", total_iterations);
    println!("  Rotation: ({:.2}°, {:.2}°, {:.2}°)", rot.x, rot.y, rot.z);
    println!(
        "  Translation: ({:.2}, {:.2}, {:.2}) mm",
        transform.translation_x, transform.translation_y, transform.translation_z
    );

    // Count metric evaluations
    let single_ncc_time = ncc_start.elapsed();
    let estimated_evals = opt_elapsed.as_secs_f64() / single_ncc_time.as_secs_f64();
    println!("  Estimated metric evaluations: {:.0}", estimated_evals);

    // Generate visual output if requested
    let output_path = args
        .iter()
        .position(|a| a == "-o" || a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    if let Some(output) = output_path {
        println!("\nGenerating visual comparison...");

        // Generate MPR axial from both volumes
        let target_mpr = MprSeries::generate(&target_volume, AnatomicalPlane::Axial)
            .context("Failed to generate target MPR")?;
        let source_mpr = MprSeries::generate(&source_volume, AnatomicalPlane::Axial)
            .context("Failed to generate source MPR")?;

        // Pick middle slice based on similar anatomical position
        let target_mid = target_mpr.len() / 2;
        let source_mid = source_mpr.len() / 2;
        let target_img = &target_mpr.images[target_mid];
        let source_img = &source_mpr.images[source_mid];

        // Use larger dimensions for canvas
        let canvas_w = target_img.width.max(source_img.width) as u32;
        let canvas_h = target_img.height.max(source_img.height) as u32;
        let mut comparison = RgbImage::new(canvas_w * 2, canvas_h);

        // Get window/level for MRI (use auto-windowing based on image percentiles)
        let target_sorted: Vec<u16> = {
            let mut v = target_img.pixels.clone();
            v.sort();
            v
        };
        let source_sorted: Vec<u16> = {
            let mut v = source_img.pixels.clone();
            v.sort();
            v
        };
        let t_p2 = target_sorted[target_sorted.len() * 2 / 100] as f64;
        let t_p98 = target_sorted[target_sorted.len() * 98 / 100] as f64;
        let s_p2 = source_sorted[source_sorted.len() * 2 / 100] as f64;
        let s_p98 = source_sorted[source_sorted.len() * 98 / 100] as f64;

        // Draw target (left) - centered if smaller than canvas
        let t_w = target_img.width as u32;
        let t_h = target_img.height as u32;
        let t_offset_x = (canvas_w - t_w) / 2;
        let t_offset_y = (canvas_h - t_h) / 2;
        for y in 0..t_h {
            for x in 0..t_w {
                let idx = (x + y * t_w) as usize;
                let val = target_img.pixels.get(idx).copied().unwrap_or(0) as f64;
                let normalized = ((val - t_p2) / (t_p98 - t_p2).max(1.0)).clamp(0.0, 1.0);
                let gray = (normalized * 255.0) as u8;
                comparison.put_pixel(x + t_offset_x, y + t_offset_y, Rgb([gray, gray, gray]));
            }
        }

        // Draw source (right) - centered if smaller than canvas
        let s_w = source_img.width as u32;
        let s_h = source_img.height as u32;
        let s_offset_x = canvas_w + (canvas_w - s_w) / 2;
        let s_offset_y = (canvas_h - s_h) / 2;
        for y in 0..s_h {
            for x in 0..s_w {
                let idx = (x + y * s_w) as usize;
                let val = source_img.pixels.get(idx).copied().unwrap_or(0) as f64;
                let normalized = ((val - s_p2) / (s_p98 - s_p2).max(1.0)).clamp(0.0, 1.0);
                let gray = (normalized * 255.0) as u8;
                comparison.put_pixel(x + s_offset_x, y + s_offset_y, Rgb([gray, gray, gray]));
            }
        }

        comparison.save(&output)?;
        println!("  Saved comparison to: {}", output.display());
        println!("  Layout: Target (2026) | Source (2014)");
        println!(
            "  Transform found: rot=({:.1}°,{:.1}°,{:.1}°) trans=({:.1},{:.1},{:.1})mm",
            rot.x,
            rot.y,
            rot.z,
            transform.translation_x,
            transform.translation_y,
            transform.translation_z
        );
    }

    Ok(())
}

/// Preprocess MRI volume (normalize to 0-1 using percentiles)
fn preprocess_volume_mri(data: &[u16]) -> Vec<f32> {
    let mut sorted: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p2 = sorted[sorted.len() * 2 / 100];
    let p98 = sorted[sorted.len() * 98 / 100];
    let range = p98 - p2;

    if range > 1e-6 {
        data.iter()
            .map(|&v| ((v as f64 - p2) / range).clamp(0.0, 1.0) as f32)
            .collect()
    } else {
        data.iter().map(|&v| v as f32 / 65535.0).collect()
    }
}

/// Compute spacing for a downsampled volume that preserves physical extent
fn compute_downsampled_spacing(
    src_dims: (usize, usize, usize),
    src_spacing: (f64, f64, f64),
    dst_dims: (usize, usize, usize),
) -> (f64, f64, f64) {
    // Physical extent = original dims * original spacing
    // New spacing = physical extent / new dims
    (
        (src_dims.0 as f64 * src_spacing.0) / dst_dims.0 as f64,
        (src_dims.1 as f64 * src_spacing.1) / dst_dims.1 as f64,
        (src_dims.2 as f64 * src_spacing.2) / dst_dims.2 as f64,
    )
}

/// Downsample volume to target dimensions
fn downsample_volume(
    data: &[f32],
    src_dims: (usize, usize, usize),
    dst_dims: (usize, usize, usize),
) -> Vec<f32> {
    let (sw, sh, sd) = src_dims;
    let (dw, dh, dd) = dst_dims;

    let scale_x = sw as f64 / dw as f64;
    let scale_y = sh as f64 / dh as f64;
    let scale_z = sd as f64 / dd as f64;

    let mut result = vec![0.0f32; dw * dh * dd];

    for dz in 0..dd {
        for dy in 0..dh {
            for dx in 0..dw {
                // Sample center of destination voxel in source coordinates
                let sx = ((dx as f64 + 0.5) * scale_x).min(sw as f64 - 1.0);
                let sy = ((dy as f64 + 0.5) * scale_y).min(sh as f64 - 1.0);
                let sz = ((dz as f64 + 0.5) * scale_z).min(sd as f64 - 1.0);

                // Nearest neighbor for simplicity
                let src_idx = sx as usize + sy as usize * sw + sz as usize * sw * sh;
                let dst_idx = dx + dy * dw + dz * dw * dh;

                if src_idx < data.len() {
                    result[dst_idx] = data[src_idx];
                }
            }
        }
    }

    result
}

/// Test sync between two series - simulates scrolling one and syncing the other
fn cmd_sync(args: &[String]) -> Result<()> {
    let dicom_path = args
        .first()
        .map(PathBuf::from)
        .context("No DICOM path provided")?;

    // Parse series indices (default: 2 for T2 Axial, 9 for 3D FLAIR)
    let series1_idx = args
        .iter()
        .position(|a| a == "--series1")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let series2_idx = args
        .iter()
        .position(|a| a == "--series2")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(9);

    let output_dir = args
        .iter()
        .position(|a| a == "-o" || a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/sync_test"));

    std::fs::create_dir_all(&output_dir)?;

    println!("=== SYNC TEST ===");
    println!("Path: {}", dicom_path.display());
    println!("Series 1 (scroll): index {}", series1_idx);
    println!("Series 2 (sync target): index {}", series2_idx);
    println!("Output: {}", output_dir.display());
    println!();

    // Load both volumes
    let (volume1, name1) = load_volume_from_dir(&dicom_path, Some(series1_idx))?;
    let (volume2, name2) = load_volume_from_dir(&dicom_path, Some(series2_idx))?;

    println!(
        "Series 1: {} ({} slices)",
        name1,
        volume1.slice_count(AnatomicalPlane::Original)
    );
    println!(
        "Series 2: {} ({} slices)",
        name2,
        volume2.slice_count(AnatomicalPlane::Original)
    );

    // Generate MPR Axial for series 2 (simulating user switching to MPR mode)
    let mpr_axial = MprSeries::generate(&volume2, AnatomicalPlane::Axial)
        .context("Failed to generate MPR Axial")?;

    println!("\nSeries 2 MPR Axial: {} slices", mpr_axial.len());

    // Get slice locations for both
    let series1_locations: Vec<Option<f64>> = (0..volume1.slice_count(AnatomicalPlane::Original))
        .map(|i| Some(volume1.calculate_slice_position(AnatomicalPlane::Original, i)))
        .collect();

    let mpr_locations: Vec<Option<f64>> = mpr_axial
        .images
        .iter()
        .map(|img| img.slice_location)
        .collect();

    println!("\nSlice location ranges:");
    if let (Some(first), Some(last)) = (
        series1_locations.first().and_then(|x| *x),
        series1_locations.last().and_then(|x| *x),
    ) {
        println!("  Series 1 ({}): {:.1} to {:.1} mm", name1, first, last);
    }
    if let (Some(first), Some(last)) = (
        mpr_locations.first().and_then(|x| *x),
        mpr_locations.last().and_then(|x| *x),
    ) {
        println!("  Series 2 MPR Axial: {:.1} to {:.1} mm", first, last);
    }

    // Test sync at several positions
    println!("\n=== SYNC SIMULATION ===");
    println!("Scrolling Series 1, showing where Series 2 MPR would sync to:\n");

    let test_slices = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        33.min(series1_locations.len() - 1),
    ];

    // Generate original series for series 1 (using Volume's resample)
    let series1_original = MprSeries::generate(&volume1, AnatomicalPlane::Original)
        .context("Failed to generate original series")?;

    for &slice_idx in &test_slices {
        if slice_idx >= series1_locations.len() {
            continue;
        }

        if let Some(slice_loc) = series1_locations[slice_idx] {
            // Find closest slice in MPR series (same algorithm as Sauhu)
            let closest_mpr_idx = find_closest_slice_by_location(&mpr_locations, slice_loc);
            let mpr_loc = mpr_locations.get(closest_mpr_idx).and_then(|x| *x);

            println!(
                "  {} slice {} (Z={:.1}mm) -> MPR Axial slice {} (Z={:.1}mm)",
                name1,
                slice_idx,
                slice_loc,
                closest_mpr_idx,
                mpr_loc.unwrap_or(0.0)
            );

            // Create side-by-side image
            let mut viewer = ViewerState::new(2);

            // Get original series image from volume
            if let Some(img) = series1_original.images.get(slice_idx) {
                viewer.viewports[0].set_image(img.clone());
                viewer.viewports[0].label =
                    format!("{} {}/{}", name1, slice_idx + 1, series1_original.len());
            }

            // Load MPR Axial image at synced position
            if let Some(mpr_img) = mpr_axial.images.get(closest_mpr_idx) {
                viewer.viewports[1].set_image(mpr_img.clone());
                viewer.viewports[1].label =
                    format!("MPR Axial {}/{}", closest_mpr_idx + 1, mpr_axial.len());
            }

            // Save comparison image
            let output = viewer.render();
            let filename = format!("sync_slice_{:03}.png", slice_idx);
            output.save(output_dir.join(&filename))?;
        }
    }

    println!(
        "\nSaved {} comparison images to {}",
        test_slices.len(),
        output_dir.display()
    );
    Ok(())
}

/// Find closest slice index by slice location (same algorithm as Sauhu sync)
fn find_closest_slice_by_location(locations: &[Option<f64>], target: f64) -> usize {
    locations
        .iter()
        .enumerate()
        .filter_map(|(i, loc)| loc.map(|l| (i, (l - target).abs())))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
