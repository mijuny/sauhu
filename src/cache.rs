//! Image caching and prefetching for smooth scrolling
//!
//! Provides an LRU cache for decoded DICOM images with background prefetching.
//! Images are stored as `Arc<DicomImage>` to avoid cloning pixel data.

use crate::dicom::{DicomFile, DicomImage};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// Estimate memory usage of a DicomImage in bytes
fn estimate_memory(image: &DicomImage) -> usize {
    std::mem::size_of::<DicomImage>()
        + image.pixels.capacity() * 2 // Vec capacity, not len
        + 256 // estimate for metadata String fields
}

/// Cached image entry with memory tracking
struct CachedEntry {
    image: Arc<DicomImage>,
    memory_bytes: usize,
}

/// Request for prefetch worker
struct PrefetchRequest {
    path: PathBuf,
}

/// Result from prefetch worker
struct PrefetchResult {
    path: PathBuf,
    image: Option<DicomImage>,
}

/// LRU Image Cache with memory-based eviction and background prefetching
///
/// Uses IndexMap for O(1) LRU operations (insertion-order iteration + O(1) removal).
/// Images are stored as `Arc<DicomImage>` so callers get cheap reference-counted clones.
pub struct ImageCache {
    /// Cached images by file path (insertion order = LRU order, oldest first)
    entries: IndexMap<PathBuf, CachedEntry>,
    /// Current memory usage in bytes
    current_memory: usize,
    /// Maximum memory limit in bytes
    max_memory: usize,
    /// Prefetch request sender
    prefetch_tx: Sender<PrefetchRequest>,
    /// Prefetch result receiver
    prefetch_rx: Receiver<PrefetchResult>,
    /// Paths currently being prefetched (avoid duplicates)
    pending_prefetch: HashSet<PathBuf>,
    /// Prefetch window: images ahead of current
    prefetch_ahead: usize,
    /// Prefetch window: images behind current
    prefetch_behind: usize,
}

impl ImageCache {
    /// Create new cache with memory limit in bytes
    #[allow(dead_code)] // cache API
    pub fn new(max_memory_bytes: usize) -> Self {
        Self::with_prefetch_window(max_memory_bytes, 5, 2)
    }

    /// Create new cache with custom prefetch window
    pub fn with_prefetch_window(
        max_memory_bytes: usize,
        prefetch_ahead: usize,
        prefetch_behind: usize,
    ) -> Self {
        let (prefetch_tx, prefetch_request_rx) = channel::<PrefetchRequest>();
        let (prefetch_result_tx, prefetch_rx) = channel::<PrefetchResult>();

        // Spawn prefetch worker threads
        Self::spawn_prefetch_workers(prefetch_request_rx, prefetch_result_tx);

        Self {
            entries: IndexMap::new(),
            current_memory: 0,
            max_memory: max_memory_bytes,
            prefetch_tx,
            prefetch_rx,
            pending_prefetch: HashSet::new(),
            prefetch_ahead,
            prefetch_behind,
        }
    }

    /// Spawn background workers for prefetching (2 workers for parallel decoding)
    fn spawn_prefetch_workers(
        request_rx: Receiver<PrefetchRequest>,
        result_tx: Sender<PrefetchResult>,
    ) {
        let request_rx = Arc::new(Mutex::new(request_rx));

        // 2 workers - good balance, don't saturate CPU
        for worker_id in 0..2 {
            let rx = Arc::clone(&request_rx);
            let tx = result_tx.clone();

            thread::Builder::new()
                .name(format!("prefetch-worker-{}", worker_id))
                .spawn(move || {
                    loop {
                        let request = {
                            let guard = match rx.lock() {
                                Ok(g) => g,
                                Err(_) => break,
                            };
                            guard.recv()
                        };

                        match request {
                            Ok(req) => {
                                let image = Self::load_image(&req.path);
                                let _ = tx.send(PrefetchResult {
                                    path: req.path,
                                    image,
                                });
                            }
                            Err(_) => break, // Channel closed
                        }
                    }
                })
                .expect("Failed to spawn prefetch worker");
        }
    }

    /// Load an image from disk (called in worker thread)
    fn load_image(path: &Path) -> Option<DicomImage> {
        match DicomFile::open(path) {
            Ok(dcm) => match DicomImage::from_file(&dcm) {
                Ok(image) => Some(image),
                Err(e) => {
                    tracing::debug!("Prefetch decode failed for {:?}: {}", path, e);
                    None
                }
            },
            Err(e) => {
                tracing::debug!("Prefetch open failed for {:?}: {}", path, e);
                None
            }
        }
    }

    /// Get image from cache, returning an Arc (cheap clone, no pixel copy)
    pub fn get(&mut self, path: &Path) -> Option<Arc<DicomImage>> {
        let key = path.to_path_buf();

        // Remove and re-insert to move to end (most recent)
        if let Some(entry) = self.entries.shift_remove(&key) {
            let image = Arc::clone(&entry.image);
            self.entries.insert(key, entry);
            return Some(image);
        }

        None
    }

    /// Check if image is in cache
    pub fn contains(&self, path: &Path) -> bool {
        self.entries.contains_key(path)
    }

    /// Insert image into cache (evicts oldest if needed)
    pub fn insert(&mut self, path: PathBuf, image: DicomImage) {
        let memory = estimate_memory(&image);

        // Evict until we have space
        while self.current_memory + memory > self.max_memory && !self.entries.is_empty() {
            self.evict_oldest();
        }

        // Remove from pending if it was prefetched
        self.pending_prefetch.remove(&path);

        // Update if already exists
        if let Some(old) = self.entries.shift_remove(&path) {
            self.current_memory -= old.memory_bytes;
        }

        // Insert new entry
        self.current_memory += memory;
        self.entries.insert(
            path,
            CachedEntry {
                image: Arc::new(image),
                memory_bytes: memory,
            },
        );
    }

    /// Evict oldest entry from cache
    fn evict_oldest(&mut self) {
        if let Some((key, entry)) = self.entries.shift_remove_index(0) {
            self.current_memory -= entry.memory_bytes;
            tracing::trace!(
                "Cache evicted: {:?} (freed {} KB)",
                key.file_name().unwrap_or_default(),
                entry.memory_bytes / 1024
            );
        }
    }

    /// Request prefetch of images around current position (for navigation)
    pub fn prefetch(&mut self, series_files: &[PathBuf], current_index: usize) {
        if series_files.is_empty() {
            return;
        }

        // Define prefetch window
        let start = current_index.saturating_sub(self.prefetch_behind);
        let end = (current_index + self.prefetch_ahead + 1).min(series_files.len());

        // Queue images for prefetch (closest first for better UX)
        let mut to_prefetch: Vec<(usize, &PathBuf)> = Vec::new();

        for (i, path) in series_files.iter().enumerate().take(end).skip(start) {
            // Skip if already cached or pending
            if self.entries.contains_key(path) || self.pending_prefetch.contains(path) {
                continue;
            }

            to_prefetch.push((i, path));
        }

        // Sort by distance from current (closest first)
        to_prefetch.sort_by_key(|(i, _)| (*i as i32 - current_index as i32).abs());

        // Queue for prefetch
        for (_, path) in to_prefetch {
            self.pending_prefetch.insert(path.clone());
            let _ = self
                .prefetch_tx
                .send(PrefetchRequest { path: path.clone() });
        }
    }

    /// Prefetch entire series (call when series is first opened)
    pub fn prefetch_series(&mut self, series_files: &[PathBuf]) {
        if series_files.is_empty() {
            return;
        }

        tracing::info!("Prefetching entire series: {} images", series_files.len());

        for path in series_files {
            // Skip if already cached or pending
            if self.entries.contains_key(path) || self.pending_prefetch.contains(path) {
                continue;
            }

            self.pending_prefetch.insert(path.clone());
            let _ = self
                .prefetch_tx
                .send(PrefetchRequest { path: path.clone() });
        }
    }

    /// Process completed prefetch results (call from main thread each frame)
    pub fn process_prefetch_results(&mut self) {
        while let Ok(result) = self.prefetch_rx.try_recv() {
            self.pending_prefetch.remove(&result.path);

            if let Some(image) = result.image {
                // Only insert if not already cached (avoid race)
                if !self.entries.contains_key(&result.path) {
                    self.insert(result.path, image);
                }
            }
        }
    }

    /// Clear all cached images (call when switching patients)
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_memory = 0;
        self.pending_prefetch.clear();
        tracing::debug!("Image cache cleared");
    }

    /// Get cache statistics for debugging
    #[allow(dead_code)] // cache API
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            memory_mb: self.current_memory / (1024 * 1024),
            max_memory_mb: self.max_memory / (1024 * 1024),
            pending_prefetch: self.pending_prefetch.len(),
        }
    }
}

/// Cache statistics for debugging/monitoring
#[derive(Debug, Clone)]
#[allow(dead_code)] // returned by ImageCache::stats()
pub struct CacheStats {
    pub entries: usize,
    pub memory_mb: usize,
    pub max_memory_mb: usize,
    pub pending_prefetch: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache: {} images, {}/{} MB, {} pending",
            self.entries, self.memory_mb, self.max_memory_mb, self.pending_prefetch
        )
    }
}
