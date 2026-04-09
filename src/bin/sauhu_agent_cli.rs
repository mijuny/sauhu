#![allow(dead_code, unused_variables, unused_imports)]
//! Sauhu Agent CLI - DICOM inspection tool for AI agents
//!
//! Renders DICOM images to PNG files, enabling AI agents to visually inspect
//! MPR slices, reference lines, measurements, and geometry.
//!
//! # Usage
//!
//! ```bash
//! sauhu-agent-cli view /path/to/dicom -o output.png
//! sauhu-agent-cli point /path/to/dicom --xyz -50,30,120 -o point.png
//! sauhu-agent-cli roi /path/to/dicom --center 100,150 --radius 10
//! sauhu-agent-cli info /path/to/dicom
//! sauhu-agent-cli debug /path/to/dicom --xyz -50,30,120
//! ```

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use std::path::PathBuf;
use std::sync::Arc;

use sauhu::dicom::{
    compute_circle_roi_stats, compute_distance_mm, compute_point_in_plane, compute_reference_line,
    group_files_by_series, AnatomicalPlane, DicomFile, DicomImage, ImagePixels, ImagePlane,
    MprSeries, Point2D, Vec3, Volume,
};

/// Window/level presets
#[derive(Debug, Clone, Copy)]
struct WindowPreset {
    name: &'static str,
    center: f64,
    width: f64,
}

const PRESETS: &[WindowPreset] = &[
    WindowPreset { name: "brain", center: 40.0, width: 80.0 },
    WindowPreset { name: "subdural", center: 75.0, width: 215.0 },
    WindowPreset { name: "stroke", center: 40.0, width: 40.0 },
    WindowPreset { name: "bone", center: 400.0, width: 1800.0 },
    WindowPreset { name: "soft", center: 50.0, width: 400.0 },
    WindowPreset { name: "lung", center: -600.0, width: 1500.0 },
    WindowPreset { name: "abdomen", center: 40.0, width: 400.0 },
];

fn get_preset(name: &str) -> Option<WindowPreset> {
    PRESETS.iter().find(|p| p.name == name).copied()
}

// ============================================================================
// PNG rendering (software renderer for headless output)
// ============================================================================

struct Viewport {
    image: Option<Arc<DicomImage>>,
    window_center: f64,
    window_width: f64,
    label: String,
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

    fn set_image(&mut self, image: Arc<DicomImage>) {
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

struct ViewerState {
    viewports: Vec<Viewport>,
    active: usize,
    viewport_size: (u32, u32),
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

    fn render(&self) -> RgbImage {
        let num = self.viewports.len();
        let (vp_w, vp_h) = self.viewport_size;

        let (cols, rows) = match num {
            1 => (1, 1),
            2 => (2, 1),
            3 => (3, 1),
            4 => (2, 2),
            5 | 6 => (3, 2),
            _ => (4, 2),
        };

        let img_width = cols * vp_w + (cols - 1) * 4;
        let img_height = rows * vp_h + (rows - 1) * 4;

        let mut output = RgbImage::new(img_width, img_height);
        for pixel in output.pixels_mut() {
            *pixel = Rgb([30, 30, 30]);
        }

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

        if let Some(ref image) = viewport.image {
            let rendered = self.render_dicom_image(
                image, viewport.window_center, viewport.window_width, vp_w, vp_h,
            );

            for y in 0..vp_h {
                for x in 0..vp_w {
                    let pixel = rendered.get_pixel(x, y);
                    output.put_pixel(x + x_offset, y + y_offset, *pixel);
                }
            }

            let scale_x = vp_w as f64 / image.width as f64;
            let scale_y = vp_h as f64 / image.height as f64;
            let scale = scale_x.min(scale_y);
            let offset_x = (vp_w as f64 - image.width as f64 * scale) / 2.0;
            let offset_y = (vp_h as f64 - image.height as f64 * scale) / 2.0;

            // Reference lines from active viewport
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
                                Rgb([255, 255, 0]),
                            );
                        }
                    }
                }
            }

            // Marked points
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
            for y in 0..vp_h {
                for x in 0..vp_w {
                    output.put_pixel(x + x_offset, y + y_offset, Rgb([0, 0, 0]));
                }
            }
        }

        let border_color = if viewport_idx == self.active {
            Rgb([100, 149, 237])
        } else {
            Rgb([80, 80, 80])
        };
        self.draw_rect_outline(output, x_offset, y_offset, vp_w, vp_h, border_color);
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

        let scale_x = target_w as f64 / image.width as f64;
        let scale_y = target_h as f64 / image.height as f64;
        let scale = scale_x.min(scale_y);

        let offset_x = ((target_w as f64 - image.width as f64 * scale) / 2.0) as i32;
        let offset_y = ((target_h as f64 - image.height as f64 * scale) / 2.0) as i32;

        let min_val = window_center - window_width / 2.0;
        let max_val = window_center + window_width / 2.0;

        for ty in 0..target_h {
            for tx in 0..target_w {
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

        for dx in -size..=size {
            let x = cx as i32 + dx;
            let y = cy as i32;
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
        for dy in -size..=size {
            let x = cx as i32;
            let y = cy as i32 + dy;
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }

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

    fn draw_rect_outline(&self, img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>) {
        let (iw, ih) = img.dimensions();
        for dx in 0..w {
            if x + dx < iw {
                if y < ih { img.put_pixel(x + dx, y, color); }
                if y + h - 1 < ih { img.put_pixel(x + dx, y + h - 1, color); }
            }
        }
        for dy in 0..h {
            if y + dy < ih {
                if x < iw { img.put_pixel(x, y + dy, color); }
                if x + w - 1 < iw { img.put_pixel(x + w - 1, y + dy, color); }
            }
        }
    }

    fn draw_label(&self, img: &mut RgbImage, x: u32, y: u32, _text: &str) {
        let (iw, ih) = img.dimensions();
        for dy in 0..14 {
            for dx in 0..100 {
                if x + dx < iw && y + dy < ih {
                    let pixel = img.get_pixel_mut(x + dx, y + dy);
                    pixel[0] /= 3;
                    pixel[1] /= 3;
                    pixel[2] /= 3;
                }
            }
        }
    }
}

// ============================================================================
// Commands
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
                if i + 1 < args.len() { output_path = PathBuf::from(&args[i + 1]); i += 2; }
                else { i += 1; }
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
                } else { i += 1; }
            }
            "--slice" => {
                if i + 1 < args.len() { slice_idx = args[i + 1].parse().ok(); i += 2; }
                else { i += 1; }
            }
            "--window" => {
                if i + 1 < args.len() { window_preset = Some(args[i + 1].clone()); i += 2; }
                else { i += 1; }
            }
            "--series" => {
                if i + 1 < args.len() { series_idx = args[i + 1].parse().ok(); i += 2; }
                else { i += 1; }
            }
            path if !path.starts_with('-') => { dicom_path = Some(PathBuf::from(path)); i += 1; }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    println!("Loading DICOM from: {}", dicom_path.display());

    let (volume, series_name) = load_volume_from_dir(&dicom_path, series_idx)?;

    println!(
        "Volume: {:?} acquisition, {}x{}x{} voxels",
        volume.acquisition_orientation,
        volume.dimensions.0, volume.dimensions.1, volume.dimensions.2
    );

    let mut viewer = ViewerState::new(3);
    let planes = [AnatomicalPlane::Axial, AnatomicalPlane::Coronal, AnatomicalPlane::Sagittal];

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

    viewer.active = match plane {
        AnatomicalPlane::Axial => 0,
        AnatomicalPlane::Coronal => 1,
        AnatomicalPlane::Sagittal => 2,
        _ => 0,
    };

    let output = viewer.render();
    output.save(&output_path)?;

    println!("Saved to: {}", output_path.display());
    println!("\nViewport info:");
    for (i, vp) in viewer.viewports.iter().enumerate() {
        let active_marker = if i == viewer.active { " [ACTIVE]" } else { "" };
        println!("  {}: WC={:.0} WW={:.0}{}", vp.label, vp.window_center, vp.window_width, active_marker);
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
                if i + 1 < args.len() { output_path = PathBuf::from(&args[i + 1]); i += 2; }
                else { i += 1; }
            }
            "--xyz" => {
                if i + 1 < args.len() {
                    let parts: Vec<&str> = args[i + 1].split(',').collect();
                    if parts.len() == 3 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[0].parse::<f64>(), parts[1].parse::<f64>(), parts[2].parse::<f64>(),
                        ) {
                            point = Some(Vec3::new(x, y, z));
                        }
                    }
                    i += 2;
                } else { i += 1; }
            }
            "--window" => {
                if i + 1 < args.len() { window_preset = Some(args[i + 1].clone()); i += 2; }
                else { i += 1; }
            }
            "--series" => {
                if i + 1 < args.len() { series_idx = args[i + 1].parse().ok(); i += 2; }
                else { i += 1; }
            }
            path if !path.starts_with('-') => { dicom_path = Some(PathBuf::from(path)); i += 1; }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    let point = point.context("No point provided (use --xyz x,y,z)")?;

    println!("Loading DICOM from: {}", dicom_path.display());
    println!("Marking point: ({:.1}, {:.1}, {:.1})", point.x, point.y, point.z);

    let (volume, _series_name) = load_volume_from_dir(&dicom_path, series_idx)?;

    let mut viewer = ViewerState::new(3);
    viewer.add_point(point);

    let planes = [AnatomicalPlane::Axial, AnatomicalPlane::Coronal, AnatomicalPlane::Sagittal];

    for (i, &plane) in planes.iter().enumerate() {
        if let Some(series) = MprSeries::generate(&volume, plane) {
            let slice_idx = find_slice_for_point(&point, &series, &volume, plane);

            if let Some(img) = series.images.get(slice_idx) {
                viewer.viewports[i].set_image(img.clone());
                viewer.viewports[i].label = format!("{:?} {}/{}", plane, slice_idx + 1, series.len());

                if let Some(ref preset_name) = window_preset {
                    if let Some(preset) = get_preset(preset_name) {
                        viewer.viewports[i].set_window(preset.center, preset.width);
                    }
                }

                if let Some(plane_geom) = img.image_plane.as_ref() {
                    if let Some(pixel) = compute_point_in_plane(&point, plane_geom) {
                        println!("  {:?}: slice {}, pixel ({:.1}, {:.1})", plane, slice_idx, pixel.x, pixel.y);
                    }
                }
            }
        }
    }

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
                } else { i += 1; }
            }
            "--radius" => {
                if i + 1 < args.len() { radius = args[i + 1].parse().unwrap_or(10.0); i += 2; }
                else { i += 1; }
            }
            "--slice" => {
                if i + 1 < args.len() { slice_idx = args[i + 1].parse().ok(); i += 2; }
                else { i += 1; }
            }
            path if !path.starts_with('-') => { dicom_path = Some(PathBuf::from(path)); i += 1; }
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

    println!("Slice {}/{}, ROI center=({}, {}), radius={}", idx + 1, images.len(), cx, cy, radius);

    let image_pixels = ImagePixels {
        data: &img.pixels,
        width: img.width,
        height: img.height,
        rescale_slope: img.rescale_slope,
        rescale_intercept: img.rescale_intercept,
    };
    if let Some(stats) = compute_circle_roi_stats(&image_pixels, cx, cy, radius) {
        println!("\nROI Statistics:");
        println!("  Mean:   {:.1}", stats.mean);
        println!("  SD:     {:.1}", stats.std_dev);
        println!("  Min:    {:.1}", stats.min);
        println!("  Max:    {:.1}", stats.max);
        println!("  Pixels: {}", stats.pixel_count);

        if img.modality.as_deref() == Some("CT") {
            println!("\nCT Interpretation:");
            if stats.mean < -500.0 { println!("  Density: Air/Lung"); }
            else if stats.mean < -50.0 { println!("  Density: Fat"); }
            else if stats.mean < 20.0 { println!("  Density: Water/Fluid"); }
            else if stats.mean < 80.0 { println!("  Density: Soft Tissue"); }
            else if stats.mean < 200.0 { println!("  Density: Blood/Contrast"); }
            else { println!("  Density: Bone/Calcification"); }
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
                } else { i += 1; }
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
                } else { i += 1; }
            }
            "--slice" => {
                if i + 1 < args.len() { slice_idx = args[i + 1].parse().ok(); i += 2; }
                else { i += 1; }
            }
            path if !path.starts_with('-') => { dicom_path = Some(PathBuf::from(path)); i += 1; }
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
    println!("From: ({:.1}, {:.1}) to ({:.1}, {:.1})", from.0, from.1, to.0, to.1);

    if let Some(spacing) = img.pixel_spacing {
        let dist = compute_distance_mm(from, to, spacing);
        println!("\nDistance: {:.1} mm", dist);
        println!("Pixel spacing: {:.3} x {:.3} mm", spacing.0, spacing.1);
    } else {
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;
        let dist = (dx * dx + dy * dy).sqrt();
        println!("\nDistance: {:.1} pixels (no pixel spacing available)", dist);
    }

    Ok(())
}

fn cmd_info(args: &[String]) -> Result<()> {
    let dicom_path = args.first().map(PathBuf::from).context("No DICOM path provided")?;

    println!("Loading DICOM from: {}", dicom_path.display());

    let files = load_dicom_dir(&dicom_path)?;
    println!("Found {} DICOM files", files.len());

    let series_list = group_files_by_series(files);
    println!("\n{} series found:\n", series_list.len());

    for (i, series) in series_list.iter().enumerate() {
        println!("{}. {} ({} images)", i + 1, series.display_name, series.files.len());
        println!("   Modality: {:?}", series.modality);
        println!("   Orientation: {:?}", series.orientation);
        println!(
            "   Frame of Reference: {:?}",
            series.frame_of_reference_uid.as_ref().map(|s| &s[s.len().saturating_sub(12)..])
        );

        if series.files.len() >= 2 {
            let images = load_images_from_paths(&series.files[..2.min(series.files.len())])?;
            if images.len() >= 2 {
                if let (Some(p1), Some(p2)) = (images[0].image_plane.as_ref(), images[1].image_plane.as_ref()) {
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
                            parts[0].parse::<f64>(), parts[1].parse::<f64>(), parts[2].parse::<f64>(),
                        ) {
                            test_point = Some(Vec3::new(x, y, z));
                        }
                    }
                    i += 2;
                } else { i += 1; }
            }
            "--series" => {
                if i + 1 < args.len() { series_idx = args[i + 1].parse().ok(); i += 2; }
                else { i += 1; }
            }
            path if !path.starts_with('-') => { dicom_path = Some(PathBuf::from(path)); i += 1; }
            _ => i += 1,
        }
    }

    let dicom_path = dicom_path.context("No DICOM path provided")?;
    println!("=== MPR GEOMETRY DEBUG ===\n");

    let (volume, series_name) = load_volume_from_dir(&dicom_path, series_idx)?;

    println!("VOLUME GEOMETRY ({})", series_name);
    println!("  Dimensions: {}x{}x{} (W x H x D)", volume.dimensions.0, volume.dimensions.1, volume.dimensions.2);
    println!("  Spacing: {:.3} x {:.3} x {:.3} mm", volume.spacing.0, volume.spacing.1, volume.spacing.2);
    println!("  Origin: ({:.2}, {:.2}, {:.2})", volume.origin.x, volume.origin.y, volume.origin.z);
    println!("  Row direction:   ({:.4}, {:.4}, {:.4})", volume.row_direction.x, volume.row_direction.y, volume.row_direction.z);
    println!("  Col direction:   ({:.4}, {:.4}, {:.4})", volume.col_direction.x, volume.col_direction.y, volume.col_direction.z);
    println!("  Slice direction: ({:.4}, {:.4}, {:.4})", volume.slice_direction.x, volume.slice_direction.y, volume.slice_direction.z);
    println!("  Acquisition: {:?}", volume.acquisition_orientation);

    let (sx, sy, sz) = volume.spacing;
    let (w, h, d) = volume.dimensions;
    let extent_x = w as f64 * sx;
    let extent_y = h as f64 * sy;
    let extent_z = d as f64 * sz;
    println!("  Physical extent: {:.1} x {:.1} x {:.1} mm", extent_x, extent_y, extent_z);

    let center_offset = volume.row_direction.scale(w as f64 * sx / 2.0)
        .add(&volume.col_direction.scale(h as f64 * sy / 2.0))
        .add(&volume.slice_direction.scale(d as f64 * sz / 2.0));
    let volume_center = volume.origin.add(&center_offset);
    println!("  Volume center: ({:.2}, {:.2}, {:.2})", volume_center.x, volume_center.y, volume_center.z);

    let test_pt = test_point.unwrap_or(volume_center);
    println!("\nTEST POINT: ({:.2}, {:.2}, {:.2})", test_pt.x, test_pt.y, test_pt.z);

    let planes = [
        (AnatomicalPlane::Axial, "AXIAL"),
        (AnatomicalPlane::Coronal, "CORONAL"),
        (AnatomicalPlane::Sagittal, "SAGITTAL"),
    ];

    for (plane, name) in &planes {
        println!("\n--- {} PLANE ---", name);

        if let Some(series) = MprSeries::generate(&volume, *plane) {
            println!("  Generated {} slices", series.len());

            if let (Some(first), Some(last)) = (series.images.first(), series.images.last()) {
                if let (Some(first_loc), Some(last_loc)) = (first.slice_location, last.slice_location) {
                    println!("  Slice location range: {:.2} to {:.2} mm", first_loc, last_loc);
                }
            }

            let slice_idx = find_slice_for_point(&test_pt, &series, &volume, *plane);
            println!("  Best slice for point: {}", slice_idx);

            if let Some(img) = series.images.get(slice_idx) {
                println!("  Image size: {}x{}", img.width, img.height);

                if let Some(ref img_plane) = img.image_plane {
                    println!("  ImagePlane geometry:");
                    println!("    Position: ({:.2}, {:.2}, {:.2})", img_plane.position.x, img_plane.position.y, img_plane.position.z);
                    println!("    Row dir:  ({:.4}, {:.4}, {:.4})", img_plane.row_direction.x, img_plane.row_direction.y, img_plane.row_direction.z);
                    println!("    Col dir:  ({:.4}, {:.4}, {:.4})", img_plane.col_direction.x, img_plane.col_direction.y, img_plane.col_direction.z);
                    println!("    Normal:   ({:.4}, {:.4}, {:.4})", img_plane.normal.x, img_plane.normal.y, img_plane.normal.z);
                    println!("    Pixel spacing: {:.3} x {:.3} mm", img_plane.pixel_spacing.0, img_plane.pixel_spacing.1);

                    let pixel = img_plane.patient_to_pixel(&test_pt);
                    println!("  Point projection:");
                    println!("    Pixel coords: ({:.2}, {:.2})", pixel.x, pixel.y);
                    let (rows, cols) = img_plane.dimensions;
                    println!("    In bounds: x=[0..{}] (cols), y=[0..{}] (rows)", cols, rows);

                    let in_bounds = pixel.x >= 0.0 && pixel.x < cols as f64 && pixel.y >= 0.0 && pixel.y < rows as f64;
                    println!("    Result: {}", if in_bounds { "IN BOUNDS" } else { "OUT OF BOUNDS" });

                    if !in_bounds {
                        println!("  Debug calculation:");
                        let v = test_pt.sub(&img_plane.position);
                        println!("    Vector from plane origin to point: ({:.2}, {:.2}, {:.2})", v.x, v.y, v.z);
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

    // Compare with original images
    println!("\n--- ORIGINAL IMAGES (for comparison) ---");
    let files = load_dicom_dir(&dicom_path)?;
    let all_series = group_files_by_series(files);

    for series_info in all_series.iter().take(1) {
        if series_info.files.len() >= 2 {
            let images = load_images_from_paths(&series_info.files[..2])?;
            if let Some(img) = images.first() {
                if let Some(ref img_plane) = img.image_plane {
                    println!("  Original {} image:", series_info.display_name);
                    println!("    Position: ({:.2}, {:.2}, {:.2})", img_plane.position.x, img_plane.position.y, img_plane.position.z);
                    println!("    Row dir:  ({:.4}, {:.4}, {:.4})", img_plane.row_direction.x, img_plane.row_direction.y, img_plane.row_direction.z);
                    println!("    Col dir:  ({:.4}, {:.4}, {:.4})", img_plane.col_direction.x, img_plane.col_direction.y, img_plane.col_direction.z);
                    println!("    Normal:   ({:.4}, {:.4}, {:.4})", img_plane.normal.x, img_plane.normal.y, img_plane.normal.z);
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

fn load_dicom_dir(dir: &PathBuf) -> Result<Vec<DicomFile>> {
    let mut files = Vec::new();
    let mut rejected_non_tuli = 0;

    for entry in walkdir::WalkDir::new(dir).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Ok(dcm) = DicomFile::open(path) {
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
        println!("Safety: Rejected {} non-anonymized files (Patient ID must contain 'TULI')", rejected_non_tuli);
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

fn load_volume_from_dir(dir: &PathBuf, series_idx: Option<usize>) -> Result<(Volume, String)> {
    let files = load_dicom_dir(dir)?;
    let series_list = group_files_by_series(files);

    if series_list.is_empty() {
        anyhow::bail!("No series found");
    }

    let mpr_series: Vec<_> = series_list.iter().filter(|s| s.files.len() >= 10).collect();

    if mpr_series.is_empty() {
        anyhow::bail!("No series with enough images for MPR (need >= 10)");
    }

    let selected = if let Some(idx) = series_idx {
        mpr_series.get(idx.saturating_sub(1)).copied().context(format!(
            "Series {} not found (have {} series)", idx, mpr_series.len()
        ))?
    } else {
        mpr_series.iter().max_by_key(|s| s.files.len()).copied().unwrap()
    };

    println!("Selected series: {} ({} images)", selected.display_name, selected.files.len());

    let images = load_images_from_paths(&selected.files)?;
    let volume = Volume::from_series(&images).context("Failed to build volume")?;

    Ok((volume, selected.display_name.clone()))
}

fn find_slice_for_point(point: &Vec3, series: &MprSeries, _volume: &Volume, _plane: AnatomicalPlane) -> usize {
    let count = series.len();
    let mut best_idx = count / 2;
    let mut best_dist = f64::MAX;

    for i in 0..count {
        if let Some(img) = series.images.get(i) {
            if let Some(img_plane) = img.image_plane.as_ref() {
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
    println!("Sauhu Agent CLI - DICOM inspection tool for AI agents");
    println!();
    println!("USAGE:");
    println!("    sauhu-agent-cli <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    view <path>      Render MPR planes to PNG");
    println!("    point <path>     Mark a 3D point across all planes");
    println!("    roi <path>       Calculate ROI statistics");
    println!("    measure <path>   Measure distance between points");
    println!("    info <path>      Show DICOM directory information");
    println!("    debug <path>     Debug MPR geometry");
    println!();
    println!("OPTIONS:");
    println!("    -o, --output     Output PNG path (default: <command>.png)");
    println!("    --plane          Plane to view: axial, coronal, sagittal");
    println!("    --slice          Slice index (default: middle)");
    println!("    --window         Window preset: brain, stroke, bone, soft, lung, abdomen");
    println!("    --series         Series index (default: largest)");
    println!("    --xyz            3D point in patient coordinates: x,y,z");
    println!("    --center         2D pixel coordinates for ROI: x,y");
    println!("    --radius         ROI radius in pixels (default: 10)");
    println!("    --from/--to      Measurement endpoints: x,y");
    println!();
    println!("EXAMPLES:");
    println!("    sauhu-agent-cli view ~/dicom/brain -o brain.png --window brain");
    println!("    sauhu-agent-cli point ~/dicom/brain --xyz -50,30,120 -o point.png");
    println!("    sauhu-agent-cli roi ~/dicom/brain --center 256,256 --radius 15");
    println!("    sauhu-agent-cli debug ~/dicom/brain --xyz -50,30,120");
}

fn main() -> Result<()> {
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
        "roi" => cmd_roi(&args[1..]),
        "measure" => cmd_measure(&args[1..]),
        "info" => cmd_info(&args[1..]),
        "debug" => cmd_debug(&args[1..]),
        "help" | "--help" | "-h" => { print_help(); Ok(()) }
        _ => { println!("Unknown command: {}", args[0]); print_help(); Ok(()) }
    }
}
