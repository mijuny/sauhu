//! Thumbnail cache for series previews
#![allow(dead_code)]

use crate::dicom::{DicomFile, DicomImage};
use egui::{ColorImage, TextureHandle, TextureOptions};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

/// Thumbnail size in pixels
const THUMBNAIL_SIZE: u32 = 96;

/// Cached thumbnail with metadata
struct CachedThumbnail {
    texture: TextureHandle,
    #[allow(dead_code)]
    source_path: PathBuf,
}

/// Result from thumbnail generation thread
struct ThumbnailResult {
    series_uid: String,
    source_path: PathBuf,
    image: Option<ColorImage>,
}

/// Cache for series thumbnail images
pub struct ThumbnailCache {
    /// Map: series_instance_uid -> cached thumbnail
    thumbnails: HashMap<String, CachedThumbnail>,
    /// Series UIDs currently being loaded
    pending: HashSet<String>,
    /// Channel for receiving generated thumbnails
    receiver: Receiver<ThumbnailResult>,
    /// Channel for sending thumbnail requests
    sender: Sender<(String, PathBuf)>,
    /// Flag to clear on next frame (deferred to avoid GPU texture-in-use errors)
    clear_pending: bool,
}

impl ThumbnailCache {
    pub fn new() -> Self {
        let (request_tx, request_rx) = channel::<(String, PathBuf)>();
        let (result_tx, result_rx) = channel::<ThumbnailResult>();

        // Spawn background thread for thumbnail generation
        thread::spawn(move || {
            while let Ok((series_uid, path)) = request_rx.recv() {
                let image = generate_thumbnail(&path);
                let _ = result_tx.send(ThumbnailResult {
                    series_uid,
                    source_path: path,
                    image,
                });
            }
        });

        Self {
            thumbnails: HashMap::new(),
            pending: HashSet::new(),
            receiver: result_rx,
            sender: request_tx,
            clear_pending: false,
        }
    }

    /// Get thumbnail for a series, or request it if not cached
    pub fn get_or_request(
        &mut self,
        series_uid: &str,
        middle_file: &Path,
    ) -> Option<&TextureHandle> {
        // Don't return or request thumbnails if a clear is pending
        if self.clear_pending {
            return None;
        }

        // Check if already cached
        if self.thumbnails.contains_key(series_uid) {
            return self.thumbnails.get(series_uid).map(|c| &c.texture);
        }

        // Check if already pending
        if self.pending.contains(series_uid) {
            return None;
        }

        // Request thumbnail generation
        self.pending.insert(series_uid.to_string());
        let _ = self
            .sender
            .send((series_uid.to_string(), middle_file.to_path_buf()));

        None
    }

    /// Process pending thumbnail results and register textures
    pub fn process_pending(&mut self, ctx: &egui::Context) {
        // Handle deferred clearing (do this at the start of a new frame, not mid-frame)
        if self.clear_pending {
            self.thumbnails.clear();
            self.pending.clear();
            self.clear_pending = false;
            // Drain any pending results from the old patient
            while self.receiver.try_recv().is_ok() {}
        }

        // Process all available results
        while let Ok(result) = self.receiver.try_recv() {
            self.pending.remove(&result.series_uid);

            if let Some(image) = result.image {
                let texture = ctx.load_texture(
                    format!("thumbnail_{}", result.series_uid),
                    image,
                    TextureOptions::LINEAR,
                );

                self.thumbnails.insert(
                    result.series_uid,
                    CachedThumbnail {
                        texture,
                        source_path: result.source_path,
                    },
                );

                // Request repaint to show new thumbnail
                ctx.request_repaint();
            }
        }
    }

    /// Clear all cached thumbnails (deferred to next frame to avoid GPU texture-in-use errors)
    pub fn clear(&mut self) {
        self.clear_pending = true;
    }

    /// Check if a series has a cached thumbnail
    pub fn has_thumbnail(&self, series_uid: &str) -> bool {
        !self.clear_pending && self.thumbnails.contains_key(series_uid)
    }

    /// Check if a series thumbnail is being loaded
    pub fn is_loading(&self, series_uid: &str) -> bool {
        !self.clear_pending && self.pending.contains(series_uid)
    }
}

impl Default for ThumbnailCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a thumbnail from a DICOM file
fn generate_thumbnail(path: &Path) -> Option<ColorImage> {
    // Load DICOM file
    let dcm = DicomFile::open(path).ok()?;
    let image = DicomImage::from_file(&dcm).ok()?;

    // Apply default window/level
    let gray = image.apply_window(image.window_center, image.window_width);

    // Resize to thumbnail size using simple sampling
    let thumb = resize_image(&gray, THUMBNAIL_SIZE, THUMBNAIL_SIZE);

    // Convert to egui ColorImage
    let pixels: Vec<egui::Color32> = thumb
        .pixels()
        .map(|p| {
            let g = p.0[0];
            egui::Color32::from_rgb(g, g, g)
        })
        .collect();

    Some(ColorImage {
        size: [THUMBNAIL_SIZE as usize, THUMBNAIL_SIZE as usize],
        pixels,
    })
}

/// Simple nearest-neighbor resize for grayscale image
fn resize_image(src: &image::GrayImage, new_width: u32, new_height: u32) -> image::GrayImage {
    let src_width = src.width();
    let src_height = src.height();

    let mut dst = image::GrayImage::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x * src_width / new_width).min(src_width - 1);
            let src_y = (y * src_height / new_height).min(src_height - 1);
            dst.put_pixel(x, y, *src.get_pixel(src_x, src_y));
        }
    }

    dst
}
