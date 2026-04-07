//! Main application state for Sauhu

mod background;
mod coregistration;
mod image_loading;
mod interaction;
mod mpr;
mod quick_fetch;

use crate::cache::ImageCache;
use crate::config::Settings;
use crate::coregistration::CoregistrationManager;
use crate::db::Database;
use crate::dicom::{DicomFile, DicomImage};
use crate::gpu::DicomRenderResources;
use crate::hanging_protocol::{self, ProtocolState};
use crate::ipc::IpcCommand;
use crate::ui::{
    AnnotationStore, DatabaseWindow, InteractionMode, MeasurementState, OpenStudyAction, PatientSidebar,
    SeriesAddedEvent, SidebarAction, ViewportLayout, ViewportManager,
    ViewportShowResult,
};
use egui::DroppedFile;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use background::{BackgroundLoadingState, BackgroundSystem};
use image_loading::{ImageLoadRequest, ImageLoadingSystem, LoadResult};
use interaction::InteractionState;
use mpr::{MprRequest, MprSystem};
use quick_fetch::{QuickFetchState, QuickFetchSystem};

/// Loaded study with series info (result of background loading)
#[derive(Debug, Clone)]
pub struct LoadedStudyWithSeries {
    pub study: crate::db::Study,
    pub db_series: Vec<crate::db::Series>,
    pub series_info: Vec<crate::dicom::SeriesInfo>,
}

/// Current patient context
#[derive(Debug, Clone)]
pub struct CurrentPatient {
    pub patient_id: String,
    #[allow(dead_code)]
    pub patient_name: String,
    pub studies: Vec<crate::db::Study>,
}

/// Main application state
pub struct SauhuApp {
    /// Application settings (from config file)
    settings: Arc<Settings>,
    /// Database handle
    db: Database,
    /// Multi-viewport manager
    viewport_manager: ViewportManager,
    /// Status message
    status: String,
    /// Image loading system (file loading, async channels, generation tracking)
    image_loading: ImageLoadingSystem,
    /// Whether GPU rendering is available
    gpu_available: bool,
    /// Database window (patient browser + PACS query)
    database_window: DatabaseWindow,
    /// Current patient being viewed
    current_patient: Option<CurrentPatient>,
    /// Patient sidebar (shows current patient's series)
    patient_sidebar: PatientSidebar,
    /// Quick fetch system (Ctrl+G PACS retrieve)
    quick_fetch: QuickFetchSystem,
    /// User interaction state (mode, measurements, annotations, mouse tracking)
    interaction: InteractionState,
    /// IPC command receiver (for external apps like Sanelu)
    ipc_rx: Option<Receiver<IpcCommand>>,
    /// Background loading system (DB queries, directory scans, image cache)
    background: BackgroundSystem,
    /// Pending cache hits to process (need frame access for GPU texture upload)
    pending_cache_hits: Vec<ImageLoadRequest>,
    /// Flag to request GPU texture cache clear (processed safely during render)
    gpu_cache_needs_clear: bool,
    /// MPR (Multi-Planar Reconstruction) system
    mpr: MprSystem,
    /// Channel for receiving series added events from retrieval
    series_added_rx: Receiver<SeriesAddedEvent>,
    /// Coregistration manager (handles Ctrl+R workflow)
    coregistration: CoregistrationManager,
    /// Hanging protocol state (auto layout and series assignment)
    protocol_state: ProtocolState,
    /// Whether the toolbar is visible (default: true, toggle with F1)
    toolbar_visible: bool,
}

impl SauhuApp {
    /// Create channels for async image loading
    fn create_loader_channels() -> (Sender<ImageLoadRequest>, Receiver<LoadResult>) {
        let (request_tx, request_rx) = mpsc::channel::<ImageLoadRequest>();
        let (result_tx, result_rx) = mpsc::channel::<LoadResult>();

        // Spawn loader thread
        thread::spawn(move || {
            while let Ok(req) = request_rx.recv() {
                let ImageLoadRequest { path, index, total, viewport_id, generation } = req;
                let result = DicomFile::open(&path)
                    .and_then(|dcm| {
                        DicomImage::from_file(&dcm)
                            .map(|image| {
                                let patient = dcm.patient_name().unwrap_or_default();
                                let series_desc = dcm.series_description().unwrap_or_default();
                                LoadResult::Success {
                                    image,
                                    path: path.clone(),
                                    patient,
                                    series_desc,
                                    index,
                                    total,
                                    viewport_id,
                                    generation,
                                }
                            })
                            .map_err(|e| anyhow::anyhow!("{}", e))
                    })
                    .unwrap_or_else(|e| LoadResult::Failed {
                        index,
                        error: e.to_string(),
                        viewport_id,
                        generation,
                    });
                let _ = result_tx.send(result);
            }
        });

        (request_tx, result_rx)
    }

    /// Initialize GPU resources if wgpu is available
    fn init_gpu(cc: &eframe::CreationContext<'_>) -> bool {
        if let Some(render_state) = cc.wgpu_render_state.as_ref() {
            let device = &render_state.device;
            let target_format = render_state.target_format;

            tracing::info!("Initializing GPU rendering with format {:?}", target_format);

            let resources = DicomRenderResources::new(device, target_format);

            // Store resources in callback resources
            render_state
                .renderer
                .write()
                .callback_resources
                .insert(resources);

            tracing::info!("GPU rendering initialized successfully");
            true
        } else {
            tracing::warn!("wgpu not available, falling back to CPU rendering");
            false
        }
    }

    /// Create new application instance
    #[allow(dead_code)]
    pub fn new(cc: &eframe::CreationContext<'_>, db: Database, settings: Settings) -> Self {
        let gpu_available = Self::init_gpu(cc);
        let (load_sender, load_receiver) = Self::create_loader_channels();
        let (bg_request_tx, bg_result_rx) = Self::create_background_worker(db.clone());
        let (mpr_request_tx, mpr_result_rx) = Self::create_mpr_worker();
        let (series_added_tx, series_added_rx) = mpsc::channel();
        let settings = Arc::new(settings);

        // Initialize image cache from config
        let cache_bytes = (settings.cache.max_memory_gb as usize) * 1024 * 1024 * 1024;
        let image_cache = ImageCache::with_prefetch_window(
            cache_bytes,
            settings.cache.prefetch_ahead,
            settings.cache.prefetch_behind,
        );

        Self {
            settings: settings.clone(),
            db,
            viewport_manager: ViewportManager::new(gpu_available),
            status: "Ready - Press D to open Database".to_string(),
            image_loading: ImageLoadingSystem {
                pending_files: None,
                load_receiver,
                load_sender,
                patient_generation: 0,
                loading: false,
            },
            gpu_available,
            database_window: DatabaseWindow::new(settings, series_added_tx),
            current_patient: None,
            patient_sidebar: PatientSidebar::new(),
            quick_fetch: QuickFetchSystem {
                state: QuickFetchState::default(),
                rx: None,
            },
            interaction: InteractionState {
                mode: InteractionMode::default(),
                scroll_drag_accumulator: 0.0,
                scroll_drag_last_y: None,
                annotation_store: AnnotationStore::new(),
                measurement_state: MeasurementState::default(),
                current_mouse_pos: None,
                active_viewport_rect: None,
            },
            ipc_rx: None,
            background: BackgroundSystem {
                request_tx: bg_request_tx,
                result_rx: bg_result_rx,
                loading_state: BackgroundLoadingState::default(),
                image_cache,
            },
            pending_cache_hits: Vec::new(),
            gpu_cache_needs_clear: false,
            mpr: MprSystem {
                pending: None,
                request_tx: mpr_request_tx,
                result_rx: mpr_result_rx,
                generating: None,
            },
            series_added_rx,
            coregistration: CoregistrationManager::new(),
            protocol_state: ProtocolState::default(),
            toolbar_visible: true,
        }
    }

    /// Create new application instance with initial paths to load
    pub fn new_with_paths(
        cc: &eframe::CreationContext<'_>,
        db: Database,
        settings: Settings,
        paths: Vec<PathBuf>,
        ipc_rx: Receiver<IpcCommand>,
    ) -> Self {
        let gpu_available = Self::init_gpu(cc);
        let pending = if paths.is_empty() { None } else { Some(paths) };
        let (load_sender, load_receiver) = Self::create_loader_channels();
        let (bg_request_tx, bg_result_rx) = Self::create_background_worker(db.clone());
        let (mpr_request_tx, mpr_result_rx) = Self::create_mpr_worker();
        let (series_added_tx, series_added_rx) = mpsc::channel();
        let settings = Arc::new(settings);

        // Initialize image cache from config
        let cache_bytes = (settings.cache.max_memory_gb as usize) * 1024 * 1024 * 1024;
        let image_cache = ImageCache::with_prefetch_window(
            cache_bytes,
            settings.cache.prefetch_ahead,
            settings.cache.prefetch_behind,
        );

        Self {
            settings: settings.clone(),
            db,
            viewport_manager: ViewportManager::new(gpu_available),
            status: "Ready - Press D to open Database".to_string(),
            image_loading: ImageLoadingSystem {
                pending_files: pending,
                load_receiver,
                load_sender,
                patient_generation: 0,
                loading: false,
            },
            gpu_available,
            database_window: DatabaseWindow::new(settings, series_added_tx),
            current_patient: None,
            patient_sidebar: PatientSidebar::new(),
            quick_fetch: QuickFetchSystem {
                state: QuickFetchState::default(),
                rx: None,
            },
            interaction: InteractionState {
                mode: InteractionMode::default(),
                scroll_drag_accumulator: 0.0,
                scroll_drag_last_y: None,
                annotation_store: AnnotationStore::new(),
                measurement_state: MeasurementState::default(),
                current_mouse_pos: None,
                active_viewport_rect: None,
            },
            ipc_rx: Some(ipc_rx),
            background: BackgroundSystem {
                request_tx: bg_request_tx,
                result_rx: bg_result_rx,
                loading_state: BackgroundLoadingState::default(),
                image_cache,
            },
            pending_cache_hits: Vec::new(),
            gpu_cache_needs_clear: false,
            mpr: MprSystem {
                pending: None,
                request_tx: mpr_request_tx,
                result_rx: mpr_result_rx,
                generating: None,
            },
            series_added_rx,
            coregistration: CoregistrationManager::new(),
            protocol_state: ProtocolState::default(),
            toolbar_visible: true,
        }
    }

    /// Open file dialog
    fn open_file_dialog(&mut self) {
        let files = rfd::FileDialog::new()
            .add_filter("DICOM", &["dcm", "DCM", "dicom", "DICOM"])
            .add_filter("All files", &["*"])
            .set_title("Open DICOM file(s)")
            .pick_files();

        if let Some(paths) = files {
            self.image_loading.pending_files = Some(paths);
        }
    }

    /// Open folder dialog
    fn open_folder_dialog(&mut self) {
        let folder = rfd::FileDialog::new()
            .set_title("Open DICOM folder")
            .pick_folder();

        if let Some(path) = folder {
            self.load_folder(&path);
        }
    }

    /// Handle dropped files
    fn handle_dropped_files(&mut self, files: &[DroppedFile]) {
        tracing::info!("Received {} dropped files", files.len());

        let mut paths: Vec<PathBuf> = Vec::new();

        for file in files {
            tracing::debug!("Dropped file: {:?}, path: {:?}", file.name, file.path);

            if let Some(path) = &file.path {
                paths.push(path.clone());
            } else if !file.name.is_empty() {
                // On some platforms, we only get the name, not the path
                tracing::warn!("Dropped file has no path, only name: {}", file.name);
            }
        }

        if !paths.is_empty() {
            self.load_files(paths);
        } else {
            self.status = "Drop failed - use File -> Open instead".to_string();
        }
    }

    /// Navigate to next/previous image in active viewport (with sync support)
    fn navigate(&mut self, delta: i32) {
        // Check if MPR mode is active - need to extract info before mutable borrow
        let mpr_info = {
            let slot = self.viewport_manager.get_active();
            if slot.mpr_state.is_active() && slot.mpr_state.has_series() {
                Some((
                    slot.mpr_state.mpr_series.clone(),
                    slot.viewport.window_level(),
                ))
            } else if slot.mpr_state.is_active() {
                // Legacy fallback: no pre-generated series
                Some((None, slot.viewport.window_level()))
            } else {
                None
            }
        };

        // Handle MPR navigation
        if let Some((series_opt, (wc, ww))) = mpr_info {
            let active_id = self.viewport_manager.active_viewport();

            let slot = self.viewport_manager.get_active_mut();
            slot.mpr_state.navigate(delta);
            let new_index = slot.mpr_state.slice_index;

            if let Some(series) = series_opt {
                // Use pre-generated series - CPU rendering for simplicity
                if let Some(image) = series.images.get(new_index) {
                    slot.viewport.set_image_keep_view(Arc::clone(image));
                    slot.viewport.set_window(wc, ww);
                }
            } else {
                // Legacy fallback: generate slice on-demand
                let volume = slot.mpr_state.volume.clone();
                if let (Some(slice), Some(vol)) = (slot.mpr_state.get_slice(), volume.as_ref()) {
                    let mpr_image = slice.to_dicom_image(vol, new_index);
                    slot.viewport.set_image(Arc::new(mpr_image));
                    slot.viewport.set_window(wc, ww);
                }
            }
            self.status = slot.mpr_state.position_info();

            // Sync other viewports to the MPR slice location
            let slice_loc = self
                .viewport_manager
                .get_active()
                .mpr_state
                .mpr_series
                .as_ref()
                .and_then(|s| s.sync_info.slice_locations.get(new_index).copied())
                .flatten();

            tracing::info!(
                "MPR navigate: active_id={}, new_index={}, slice_loc={:?}",
                active_id,
                new_index,
                slice_loc
            );

            if let Some(loc) = slice_loc {
                let sync_updates = self.viewport_manager.sync_to_location(active_id, loc);
                tracing::info!("MPR navigate sync_updates: {:?}", sync_updates);
                // Load images for synced viewports
                for (viewport_id, new_idx) in &sync_updates {
                    self.load_image_for_viewport(*viewport_id, *new_idx);
                }
            } else {
                tracing::warn!("MPR navigate: no slice_loc for index {}", new_index);
            }
            return;
        }

        // Normal navigation using viewport manager (handles sync)
        let updates = self.viewport_manager.navigate(delta);

        // Load images for all viewports that need updating
        for (viewport_id, new_index) in &updates {
            self.load_image_for_viewport(*viewport_id, *new_index);
        }

        // Trigger prefetch for all updated viewports
        for (viewport_id, new_index) in updates {
            if let Some(slot) = self.viewport_manager.get_slot(viewport_id) {
                self.background.image_cache.prefetch(&slot.series_files, new_index);
            }
        }
    }

    /// Show the main UI
    fn show_main_ui(&mut self, ctx: &egui::Context) {
        // MPR action collected from toolbar or keyboard, executed after input handling
        let mut mpr_action: Option<crate::dicom::AnatomicalPlane> = None;

        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open File(s)... (Ctrl+O)").clicked() {
                        self.open_file_dialog();
                        ui.close_menu();
                    }
                    if ui.button("Open Folder...").clicked() {
                        self.open_folder_dialog();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    let toolbar_label = if self.toolbar_visible {
                        "Hide Toolbar (F1)"
                    } else {
                        "Show Toolbar (F1)"
                    };
                    if ui.button(toolbar_label).clicked() {
                        self.toolbar_visible = !self.toolbar_visible;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Database... (D)").clicked() {
                        self.database_window.open = !self.database_window.open;
                        if self.database_window.open {
                            self.database_window.request_load_studies();
                        }
                        ui.close_menu();
                    }
                    let sidebar_label = if self.patient_sidebar.visible {
                        "Hide Sidebar (B)"
                    } else {
                        "Show Sidebar (B)"
                    };
                    if ui.button(sidebar_label).clicked() {
                        self.patient_sidebar.visible = !self.patient_sidebar.visible;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Layout", |ui| {
                        if ui.button("Single (V+1)").clicked() {
                            self.viewport_manager.set_layout(ViewportLayout::Single);
                            ui.close_menu();
                        }
                        if ui.button("2 Horizontal (V+2)").clicked() {
                            self.viewport_manager
                                .set_layout(ViewportLayout::Horizontal2);
                            ui.close_menu();
                        }
                        if ui.button("3 Horizontal (V+3)").clicked() {
                            self.viewport_manager
                                .set_layout(ViewportLayout::Horizontal3);
                            ui.close_menu();
                        }
                        if ui.button("2x2 Grid (V+4)").clicked() {
                            self.viewport_manager.set_layout(ViewportLayout::Grid2x2);
                            ui.close_menu();
                        }
                        if ui.button("2x2 + 1 Right (V+5)").clicked() {
                            self.viewport_manager
                                .set_layout(ViewportLayout::Grid2x2Plus1);
                            ui.close_menu();
                        }
                        if ui.button("3x2 Grid (V+6)").clicked() {
                            self.viewport_manager.set_layout(ViewportLayout::Grid3x2);
                            ui.close_menu();
                        }
                        if ui.button("4x2 Grid (V+8)").clicked() {
                            self.viewport_manager.set_layout(ViewportLayout::Grid4x2);
                            ui.close_menu();
                        }
                    });
                    let sync_label = if self.viewport_manager.sync_enabled() {
                        "Disable Sync (S)"
                    } else {
                        "Enable Sync (S)"
                    };
                    if ui.button(sync_label).clicked() {
                        self.viewport_manager.toggle_sync();
                        ui.close_menu();
                    }
                    let ref_lines_label = if self.viewport_manager.reference_lines_enabled() {
                        "Disable Reference Lines (R)"
                    } else {
                        "Enable Reference Lines (R)"
                    };
                    if ui.button(ref_lines_label).clicked() {
                        self.viewport_manager.toggle_reference_lines();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Fit to Window (Space)").clicked() {
                        self.viewport_manager
                            .get_active_mut()
                            .viewport
                            .fit_to_window(ui.available_size());
                        ui.close_menu();
                    }
                    if ui.button("1:1 Pixel (Shift+0)").clicked() {
                        self.viewport_manager
                            .get_active_mut()
                            .viewport
                            .set_one_to_one();
                        ui.close_menu();
                    }
                    if ui.button("Reset View").clicked() {
                        self.viewport_manager.get_active_mut().viewport.reset_view();
                        ui.close_menu();
                    }
                });

                ui.menu_button("Window", |ui| {
                    if ui.button("Auto (0)").clicked() {
                        self.viewport_manager
                            .get_active_mut()
                            .viewport
                            .auto_window();
                        ui.close_menu();
                    }
                    ui.separator();
                    for (idx, preset) in crate::dicom::CT_PRESETS.iter().enumerate() {
                        let label = format!(
                            "{} ({}) - {:.0}/{:.0}",
                            preset.name,
                            idx + 1,
                            preset.center,
                            preset.width
                        );
                        if ui.button(label).clicked() {
                            self.viewport_manager
                                .get_active_mut()
                                .viewport
                                .apply_preset(idx);
                            ui.close_menu();
                        }
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.status);
                });
            });
        });

        // Toolbar (toggle with F1)
        if self.toolbar_visible {
            egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().button_padding = egui::vec2(6.0, 3.0);
                    ui.spacing_mut().item_spacing.x = 2.0;

                    // === Tool modes ===
                    let mode = self.interaction.mode;
                    if ui
                        .selectable_label(mode == InteractionMode::Windowing, "☀ W/L")
                        .on_hover_text("Window/Level (W)")
                        .clicked()
                    {
                        self.interaction.mode = InteractionMode::Windowing;
                    }
                    if ui
                        .selectable_label(mode == InteractionMode::Pan, "✋ Pan")
                        .on_hover_text("Pan (Shift+drag)")
                        .clicked()
                    {
                        self.interaction.mode = InteractionMode::Pan;
                    }
                    if ui
                        .selectable_label(mode == InteractionMode::Measure, "📏 Length")
                        .on_hover_text("Length measurement (L)")
                        .clicked()
                    {
                        self.interaction.mode = InteractionMode::Measure;
                    }
                    if ui
                        .selectable_label(mode == InteractionMode::CircleROI, "⊙ ROI")
                        .on_hover_text("Circle ROI (O)")
                        .clicked()
                    {
                        self.interaction.mode = InteractionMode::CircleROI;
                    }
                    if ui
                        .selectable_label(mode == InteractionMode::ROIWindowing, "⊙☀ ROI W/L")
                        .on_hover_text("ROI Windowing (Shift+W)")
                        .clicked()
                    {
                        self.interaction.mode = InteractionMode::ROIWindowing;
                    }

                    ui.separator();

                    // === MPR planes ===
                    ui.label("MPR:");
                    let active_slot = self.viewport_manager.get_active();
                    let active_plane = if active_slot.mpr_state.is_active() {
                        Some(active_slot.mpr_state.plane)
                    } else {
                        None
                    };
                    if ui
                        .selectable_label(
                            active_plane == Some(crate::dicom::AnatomicalPlane::Axial),
                            "Ax",
                        )
                        .on_hover_text("Axial (A)")
                        .clicked()
                    {
                        mpr_action = Some(crate::dicom::AnatomicalPlane::Axial);
                    }
                    if ui
                        .selectable_label(
                            active_plane == Some(crate::dicom::AnatomicalPlane::Coronal),
                            "Cor",
                        )
                        .on_hover_text("Coronal (C)")
                        .clicked()
                    {
                        mpr_action = Some(crate::dicom::AnatomicalPlane::Coronal);
                    }
                    if ui
                        .selectable_label(
                            active_plane == Some(crate::dicom::AnatomicalPlane::Sagittal),
                            "Sag",
                        )
                        .on_hover_text("Sagittal (S)")
                        .clicked()
                    {
                        mpr_action = Some(crate::dicom::AnatomicalPlane::Sagittal);
                    }
                    if ui
                        .selectable_label(
                            active_plane == Some(crate::dicom::AnatomicalPlane::Original),
                            "Orig",
                        )
                        .on_hover_text("Original series (X)")
                        .clicked()
                    {
                        mpr_action = Some(crate::dicom::AnatomicalPlane::Original);
                    }

                    ui.separator();

                    // === Layout ===
                    let layout = self.viewport_manager.layout();
                    if ui
                        .selectable_label(layout == ViewportLayout::Single, "▣ 1")
                        .on_hover_text("Single viewport (Alt+1)")
                        .clicked()
                    {
                        self.viewport_manager.set_layout(ViewportLayout::Single);
                    }
                    if ui
                        .selectable_label(layout == ViewportLayout::Horizontal2, "◫ 1×2")
                        .on_hover_text("2 side by side (Alt+2)")
                        .clicked()
                    {
                        self.viewport_manager
                            .set_layout(ViewportLayout::Horizontal2);
                    }
                    if ui
                        .selectable_label(layout == ViewportLayout::Grid2x2, "⊞ 2×2")
                        .on_hover_text("2×2 grid (Alt+4)")
                        .clicked()
                    {
                        self.viewport_manager.set_layout(ViewportLayout::Grid2x2);
                    }
                    if ui
                        .selectable_label(layout == ViewportLayout::Grid3x2, "▦ 3×2")
                        .on_hover_text("3×2 grid (Alt+6)")
                        .clicked()
                    {
                        self.viewport_manager.set_layout(ViewportLayout::Grid3x2);
                    }

                    ui.separator();

                    // === View toggles ===
                    let sync_on = self.viewport_manager.sync_enabled();
                    if ui
                        .selectable_label(sync_on, "↕ Sync")
                        .on_hover_text("Toggle sync (Y)")
                        .clicked()
                    {
                        self.viewport_manager.toggle_sync();
                    }
                    let ref_on = self.viewport_manager.reference_lines_enabled();
                    if ui
                        .selectable_label(ref_on, "⫼ Ref")
                        .on_hover_text("Reference lines (R)")
                        .clicked()
                    {
                        self.viewport_manager.toggle_reference_lines();
                    }

                    ui.separator();

                    // === Actions ===
                    if ui
                        .button("⟳ Coreg")
                        .on_hover_text("Coregistration (Ctrl+R)")
                        .clicked()
                    {
                        if self.coregistration.mode().is_active() {
                            self.coregistration.cancel();
                            self.status = "Coregistration cancelled".to_string();
                            self.viewport_manager.clear_coregistration_state();
                        } else {
                            self.coregistration.start();
                            self.status = self.coregistration.mode().status_text().to_string();
                            self.update_coregistration_visual_state();
                        }
                    }
                    if ui
                        .button("🗄 DB")
                        .on_hover_text("Database browser (D)")
                        .clicked()
                    {
                        self.database_window.open = !self.database_window.open;
                        if self.database_window.open {
                            self.database_window.request_load_studies();
                        }
                    }
                });
            });
        }

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Show viewport manager status
                ui.label(self.viewport_manager.status_text());
                ui.separator();
                let (wc, ww) = self.viewport_manager.window_level();
                ui.label(format!("WC: {:.0} WW: {:.0}", wc, ww));
                ui.separator();

                // Show current interaction mode and mouse button functions
                let mode_name = self.interaction.mode.display_name();
                let (left_fn, right_fn) = self.interaction.mode.mouse_hints();
                ui.label(format!("[{}] L: {} | R: {}", mode_name, left_fn, right_fn));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("Sauhu v0.1.0");
                });
            });
        });

        // Patient sidebar (shows current patient's series)
        self.patient_sidebar.show(ctx);

        // Handle patient sidebar actions
        if let Some(action) = self.patient_sidebar.take_action() {
            match action {
                SidebarAction::LoadSeries {
                    files,
                    name,
                    sync_info,
                    ..
                } => {
                    self.protocol_state.user_overridden = true;
                    self.load_series(files, &name, sync_info);
                }
            }
        }

        // Main viewport area - show all viewports according to current layout
        let mut viewport_result: Option<ViewportShowResult> = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            viewport_result = Some(self.viewport_manager.show(ui, self.interaction.mode));
        });

        // Draw annotations on the active viewport
        self.draw_annotations(ctx);

        // Handle viewport interactions
        if let Some(result) = viewport_result {
            // Handle drag & drop of series from sidebar to specific viewports
            for (viewport_id, series) in result.dropped_series {
                self.protocol_state.user_overridden = true;
                self.load_series_to_viewport(
                    viewport_id,
                    series.files,
                    &series.label,
                    series.sync_info,
                );
            }

            // Handle coregistration viewport clicks
            if let Some(clicked_id) = result.clicked_viewport {
                self.handle_coregistration_click(clicked_id);
            }
        }

        // Update database window (processes retrieval even when closed)
        self.database_window.update(&self.db.conn());

        // Database window UI
        if self.database_window.open {
            self.database_window.show(ctx, &self.db.conn());
        }

        // Handle open study action from database window
        if let Some(action) = self.database_window.take_action() {
            self.open_study(action);
        }

        // MPR keyboard shortcuts
        let shortcuts = &self.settings.shortcuts;
        let kbd_mpr = ctx.input(|i| {
            if shortcuts.matches(&shortcuts.mpr_axial, egui::Key::A, &i.modifiers)
                && i.key_pressed(egui::Key::A)
                && !i.modifiers.shift
            {
                Some(crate::dicom::AnatomicalPlane::Axial)
            } else if shortcuts.matches(&shortcuts.mpr_coronal, egui::Key::C, &i.modifiers)
                && i.key_pressed(egui::Key::C)
            {
                Some(crate::dicom::AnatomicalPlane::Coronal)
            } else if shortcuts.matches(&shortcuts.mpr_sagittal, egui::Key::S, &i.modifiers)
                && i.key_pressed(egui::Key::S)
                && !i.modifiers.shift
            {
                Some(crate::dicom::AnatomicalPlane::Sagittal)
            } else if shortcuts.matches(&shortcuts.mpr_original, egui::Key::X, &i.modifiers)
                && i.key_pressed(egui::Key::X)
            {
                Some(crate::dicom::AnatomicalPlane::Original)
            } else {
                None
            }
        });
        if let Some(plane) = kbd_mpr {
            mpr_action = Some(plane);
        }

        // Execute MPR action if any (from toolbar or keyboard)
        if let Some(plane) = mpr_action {
            self.set_mpr_plane(plane);
        }
    }

    /// Clear all state when switching patients
    fn clear_patient(&mut self) {
        tracing::info!("Clearing patient state");

        // Increment generation to invalidate all pending async operations
        self.image_loading.patient_generation += 1;
        tracing::debug!(
            "Patient generation incremented to {}",
            self.image_loading.patient_generation
        );

        // Clear pending cache hits (now stale)
        self.pending_cache_hits.clear();

        // Clear pending MPR request
        self.mpr.pending = None;

        // Cancel any ongoing MPR generation
        let _ = self.mpr.request_tx.send(MprRequest::Cancel);
        self.mpr.generating = None;

        // Drain stale load results (best effort - loader thread may still be working)
        while self.image_loading.load_receiver.try_recv().is_ok() {}

        // Drain stale MPR results
        while self.mpr.result_rx.try_recv().is_ok() {}

        // Clear viewports (GPU textures will be replaced when new images load)
        self.viewport_manager.clear_all();

        // Request GPU texture cache clear (processed safely during render)
        self.gpu_cache_needs_clear = true;

        // Clear patient sidebar
        self.patient_sidebar.clear();

        // Clear annotations
        self.interaction.annotation_store.clear_all();

        // Reset measurement state
        self.interaction.measurement_state = MeasurementState::default();

        // Reset interaction mode
        self.interaction.mode = InteractionMode::default();

        // Clear image cache (don't keep old patient's images in memory)
        self.background.image_cache.clear();

        // Reset hanging protocol state
        self.protocol_state.reset();
    }

    /// Draw annotations on the active viewport
    fn draw_annotations(&self, ctx: &egui::Context) {
        use crate::ui::Annotation;

        let slot = self.viewport_manager.get_active();
        let viewport = &slot.viewport;
        let rect = viewport.last_rect();

        // Skip if rect is too small (not rendered yet)
        if rect.width() < 10.0 || rect.height() < 10.0 {
            return;
        }

        // Get painter for overlay
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("annotations"),
        ));

        // Draw stored annotations for current image
        if let Some(key) = slot.annotation_key() {
            if let Some(annotations) = self.interaction.annotation_store.get(&key) {
                for annotation in annotations {
                    match annotation {
                        Annotation::Length(m) => {
                            let label = m.distance.to_string();
                            viewport.draw_length_measurement(
                                &painter,
                                rect,
                                (m.start.x, m.start.y),
                                (m.end.x, m.end.y),
                                &label,
                            );
                        }
                        Annotation::Circle(roi) => {
                            let is_ct = viewport.image().map(|i| i.is_ct()).unwrap_or(false);
                            let label = roi.stats.as_ref().map(|s| s.to_display_string(is_ct));
                            viewport.draw_circle_roi(
                                &painter,
                                rect,
                                (roi.center.x, roi.center.y),
                                roi.radius_px,
                                label.as_deref(),
                            );
                        }
                    }
                }
            }
        }

        // Draw measurement preview (rubber-band while drawing)
        if let Some(mouse_pos) = self.interaction.current_mouse_pos {
            match &self.interaction.measurement_state {
                crate::ui::MeasurementState::DrawingLength { start } => {
                    viewport.draw_measurement_preview(
                        &painter,
                        rect,
                        (start.x, start.y),
                        mouse_pos,
                        false, // is_circle = false
                    );
                }
                crate::ui::MeasurementState::DrawingCircle { center }
                | crate::ui::MeasurementState::DrawingROIWindow { center } => {
                    viewport.draw_measurement_preview(
                        &painter,
                        rect,
                        (center.x, center.y),
                        mouse_pos,
                        true, // is_circle = true
                    );
                }
                _ => {}
            }
        }
    }

    /// Handle measurement input (clicks for length, drag for circles)
    fn handle_measurement_input(&mut self, ctx: &egui::Context) {
        use crate::ui::{
            Annotation, AnnotationPoint, CircleROI, DistanceResult, LengthMeasurement,
            ROIStatistics,
        };

        // Don't process measurement input if mouse is over any egui window (database, sidebar, etc.)
        if ctx.is_pointer_over_area() {
            return;
        }

        // Track mouse position for preview
        self.interaction.current_mouse_pos = ctx.input(|i| i.pointer.hover_pos());

        // Get active viewport info
        let _active_id = self.viewport_manager.active_viewport();
        let slot = self.viewport_manager.get_active();
        let viewport_rect = slot.viewport.last_rect();
        self.interaction.active_viewport_rect = Some(viewport_rect);

        // Check if mouse is over active viewport
        let mouse_over_viewport = self
            .interaction.current_mouse_pos
            .map(|pos| viewport_rect.contains(pos))
            .unwrap_or(false);

        if !mouse_over_viewport {
            return;
        }

        // Handle mouse input
        let left_clicked = ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Primary));
        let _left_pressed = ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary));
        let _left_released = ctx.input(|i| i.pointer.button_released(egui::PointerButton::Primary));

        match self.interaction.mode {
            InteractionMode::Measure => {
                // Length measurement uses clicks (two discrete points)
                if left_clicked {
                    if let Some(screen_pos) = self.interaction.current_mouse_pos {
                        let slot = self.viewport_manager.get_active();
                        if let Some(img_pos) =
                            slot.viewport.screen_to_image(screen_pos, viewport_rect)
                        {
                            match &self.interaction.measurement_state {
                                MeasurementState::Idle => {
                                    // First click - start measurement
                                    self.interaction.measurement_state = MeasurementState::DrawingLength {
                                        start: AnnotationPoint::new(img_pos.0, img_pos.1),
                                    };
                                }
                                MeasurementState::DrawingLength { start } => {
                                    // Second click - complete measurement
                                    let end = AnnotationPoint::new(img_pos.0, img_pos.1);
                                    let (dist_value, is_calibrated) = slot
                                        .viewport
                                        .calculate_distance((start.x, start.y), (end.x, end.y));

                                    let measurement = LengthMeasurement {
                                        id: self.interaction.annotation_store.next_id(),
                                        start: *start,
                                        end,
                                        distance: DistanceResult {
                                            value: dist_value,
                                            is_calibrated,
                                        },
                                    };

                                    // Store annotation
                                    if let Some(key) = slot.annotation_key() {
                                        self.interaction.annotation_store
                                            .add(&key, Annotation::Length(measurement));
                                    }

                                    self.interaction.measurement_state = MeasurementState::Idle;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            InteractionMode::CircleROI | InteractionMode::ROIWindowing => {
                // Circle ROI uses click-click (like length measurement)
                // First click sets center, second click sets radius
                if left_clicked {
                    if let Some(screen_pos) = self.interaction.current_mouse_pos {
                        let slot = self.viewport_manager.get_active();
                        if let Some(img_pos) =
                            slot.viewport.screen_to_image(screen_pos, viewport_rect)
                        {
                            match &self.interaction.measurement_state {
                                MeasurementState::Idle => {
                                    // First click - set center
                                    if self.interaction.mode == InteractionMode::CircleROI {
                                        self.interaction.measurement_state = MeasurementState::DrawingCircle {
                                            center: AnnotationPoint::new(img_pos.0, img_pos.1),
                                        };
                                    } else {
                                        self.interaction.measurement_state =
                                            MeasurementState::DrawingROIWindow {
                                                center: AnnotationPoint::new(img_pos.0, img_pos.1),
                                            };
                                    }
                                }
                                MeasurementState::DrawingCircle { center }
                                | MeasurementState::DrawingROIWindow { center } => {
                                    // Second click - set radius and complete
                                    let edge = AnnotationPoint::new(img_pos.0, img_pos.1);
                                    let radius_px = center.distance_to(&edge);

                                    if radius_px > 2.0 {
                                        // Calculate stats from image
                                        let stats =
                                            slot.viewport.image().and_then(|img| {
                                                img.calculate_circle_stats(
                                                    center.x, center.y, radius_px,
                                                )
                                                .map(|s| ROIStatistics {
                                                    mean: s.mean,
                                                    min: s.min,
                                                    max: s.max,
                                                    std_dev: s.std_dev,
                                                    pixel_count: s.pixel_count,
                                                })
                                            });

                                        if matches!(
                                            self.interaction.measurement_state,
                                            MeasurementState::DrawingCircle { .. }
                                        ) {
                                            // Store ROI annotation
                                            let roi = CircleROI {
                                                id: self.interaction.annotation_store.next_id(),
                                                center: *center,
                                                radius_px,
                                                stats,
                                            };

                                            if let Some(key) = slot.annotation_key() {
                                                self.interaction.annotation_store
                                                    .add(&key, Annotation::Circle(roi));
                                            }
                                        } else {
                                            // ROI-based windowing - apply window from stats
                                            if let Some(stats) = stats {
                                                let margin = (stats.max - stats.min) * 0.1;
                                                let new_ww =
                                                    (stats.max - stats.min + margin * 2.0).max(1.0);
                                                let new_wc = (stats.min + stats.max) / 2.0;

                                                // Apply to active viewport
                                                let slot = self.viewport_manager.get_active_mut();
                                                slot.viewport.set_window(new_wc, new_ww);
                                            }
                                        }
                                    }

                                    self.interaction.measurement_state = MeasurementState::Idle;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Open a study for viewing (loads all studies for that patient in sidebar)
    fn open_study(&mut self, action: OpenStudyAction) {
        let study = &action.study;
        let patient_id = study.patient_id.clone().unwrap_or_default();
        let patient_name = study.patient_name.clone().unwrap_or_default();

        // Fetch all studies for this patient from DB
        let all_studies = crate::db::get_studies_by_patient_id(&self.db.conn(), &patient_id)
            .unwrap_or_else(|_| vec![study.clone()]);

        tracing::info!(
            "Opening study for patient: {} ({}) with {} total studies",
            patient_name,
            patient_id,
            all_studies.len()
        );

        // Clear previous patient state
        self.clear_patient();

        // Match hanging protocol for this study
        let study_modality = study.modality.as_deref().unwrap_or("");
        let study_desc = study.study_description.as_deref().unwrap_or("");
        if let Some(protocol) = hanging_protocol::match_protocol(
            &self.settings.hanging_protocol,
            study_modality,
            study_desc,
        ) {
            tracing::info!(
                "Hanging protocol matched: '{}' (layout: {})",
                protocol.name,
                protocol.layout
            );
            if let Some(layout) = ViewportLayout::from_config_str(&protocol.layout) {
                self.viewport_manager.set_layout(layout);
            }
            self.protocol_state.active_protocol = Some(protocol.clone());
        }

        // Resolve correct paths for all studies (SCP may save files in different subdirectory)
        let mut corrected_studies = all_studies;
        let mut load_path: Option<PathBuf> = None;

        // Find the selected study's path first for loading
        let selected_study_uid = study.study_instance_uid.clone();

        for study in &mut corrected_studies {
            let path = PathBuf::from(&study.file_path);
            tracing::info!("Checking study path: {:?}", path);

            let resolved_path = if path.is_dir() {
                // Check if directory has any files
                let has_files = std::fs::read_dir(&path)
                    .map(|mut d| d.next().is_some())
                    .unwrap_or(false);

                if has_files {
                    tracing::info!("  Path has files, using as-is");
                    Some(path)
                } else if let Some(parent) = path.parent() {
                    // Directory is empty - search sibling directories for matching study
                    // This handles the case where SCP saved files to a slightly different path
                    tracing::info!("  Directory empty, searching siblings in {:?}", parent);

                    let mut found_path: Option<PathBuf> = None;
                    if let Ok(entries) = std::fs::read_dir(parent) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let entry_path = entry.path();
                            if entry_path.is_dir() && entry_path != path {
                                // Check if this directory has DICOM files
                                let has_dcm = std::fs::read_dir(&entry_path)
                                    .map(|d| {
                                        d.filter_map(|e| e.ok()).any(|e| {
                                            e.path()
                                                .extension()
                                                .map(|ext| ext == "dcm")
                                                .unwrap_or(false)
                                        })
                                    })
                                    .unwrap_or(false);

                                if has_dcm {
                                    tracing::info!(
                                        "  Found DICOM files in sibling: {:?}",
                                        entry_path
                                    );
                                    found_path = Some(entry_path);
                                    break;
                                }
                            }
                        }
                    }

                    if let Some(ref fp) = found_path {
                        study.file_path = fp.to_string_lossy().to_string();
                    }
                    found_path
                } else {
                    None
                }
            } else if path.exists() {
                // Single file
                Some(path)
            } else if let Some(parent) = path.parent() {
                // Path doesn't exist - search parent for study directories
                tracing::info!("  Path doesn't exist, searching in parent: {:?}", parent);

                let mut found_path: Option<PathBuf> = None;
                if parent.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(parent) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let entry_path = entry.path();
                            if entry_path.is_dir() {
                                let has_dcm = std::fs::read_dir(&entry_path)
                                    .map(|d| {
                                        d.filter_map(|e| e.ok()).any(|e| {
                                            e.path()
                                                .extension()
                                                .map(|ext| ext == "dcm")
                                                .unwrap_or(false)
                                        })
                                    })
                                    .unwrap_or(false);

                                if has_dcm {
                                    tracing::info!("  Found DICOM files in: {:?}", entry_path);
                                    found_path = Some(entry_path);
                                    break;
                                }
                            }
                        }
                    }
                }

                if let Some(ref fp) = found_path {
                    study.file_path = fp.to_string_lossy().to_string();
                }
                found_path
            } else {
                None
            };

            // Use first valid path for loading
            if load_path.is_none() {
                load_path = resolved_path;
            }
        }

        // Set current patient context with corrected paths
        self.current_patient = Some(CurrentPatient {
            patient_id: patient_id.clone(),
            patient_name: patient_name.clone(),
            studies: corrected_studies.clone(),
        });

        // Request async loading of patient series with corrected paths
        self.patient_sidebar
            .request_set_patient(&patient_id, &patient_name, corrected_studies);

        // Load the selected study — skip if a hanging protocol is active
        // (protocol assignment in PatientSeriesLoaded will load series to correct viewports)
        if !self.protocol_state.is_active() {
            let study_path = self
                .current_patient
                .as_ref()
                .and_then(|p| {
                    p.studies
                        .iter()
                        .find(|s| s.study_instance_uid == selected_study_uid)
                })
                .map(|s| PathBuf::from(&s.file_path))
                .or(load_path);

            if let Some(path) = study_path {
                if path.is_dir() {
                    self.load_folder(&path);
                } else {
                    self.load_files(vec![path]);
                }
            } else {
                self.status = "No valid study paths found".to_string();
            }
        }

        self.status = format!("Opened: {} ({})", patient_name, patient_id);
    }

    /// Check for IPC commands from external applications (like Sanelu)
    fn check_ipc_commands(&mut self) {
        // Collect commands first to avoid borrow issues
        let commands: Vec<IpcCommand> = if let Some(ref rx) = self.ipc_rx {
            let mut collected = Vec::new();
            while let Ok(command) = rx.try_recv() {
                collected.push(command);
            }
            collected
        } else {
            return;
        };

        // Process collected commands
        for command in commands {
            match command {
                IpcCommand::OpenStudy { accession } => {
                    tracing::info!("IPC: Opening study with accession {}", accession);
                    self.start_quick_fetch_with_accession(accession);
                }
                IpcCommand::Ping => {
                    // Already handled in IPC server, nothing to do here
                }
            }
        }
    }
}

impl eframe::App for SauhuApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let frame_start = std::time::Instant::now();
        let threshold_ms = 16; // Flag anything over 16ms (60fps budget)

        // Process prefetch results from background threads
        let t0 = std::time::Instant::now();
        self.background.image_cache.process_prefetch_results();
        if t0.elapsed().as_millis() > threshold_ms as u128 {
            tracing::warn!("SLOW: process_prefetch_results {:?}", t0.elapsed());
        }

        // Process pending cache hits (apply cached images to viewports)
        let t0 = std::time::Instant::now();
        self.process_cache_hits(frame);
        if t0.elapsed().as_millis() > threshold_ms as u128 {
            tracing::warn!("SLOW: process_cache_hits {:?}", t0.elapsed());
        }

        // Clear GPU texture cache if requested (e.g., when switching patients)
        if self.gpu_cache_needs_clear {
            self.gpu_cache_needs_clear = false;
            if let Some(render_state) = frame.wgpu_render_state() {
                let mut renderer = render_state.renderer.write();
                if let Some(resources) = renderer
                    .callback_resources
                    .get_mut::<DicomRenderResources>()
                {
                    resources.clear_all_textures();
                    tracing::info!("GPU texture cache cleared for patient switch");
                }
            }
        }

        // Check for loaded images from background thread
        let t0 = std::time::Instant::now();
        self.check_loaded_images(frame);
        if t0.elapsed().as_millis() > threshold_ms as u128 {
            tracing::warn!("SLOW: check_loaded_images {:?}", t0.elapsed());
        }

        // Check if pending MPR can now be executed (images are cached)
        self.check_pending_mpr();

        // Check for MPR generation results from background worker
        self.check_mpr_results();

        // Check for coregistration results
        self.check_coregistration_results();

        // Update fusion bind groups (must happen after image loading, before rendering)
        self.update_fusion_bind_groups(frame);

        // Check for quick fetch results
        self.check_quick_fetch_results();

        // Check for IPC commands from external apps
        self.check_ipc_commands();

        // Check for background task results and send pending requests
        let t0 = std::time::Instant::now();
        self.check_background_results();
        if t0.elapsed().as_millis() > threshold_ms as u128 {
            tracing::warn!("SLOW: check_background_results {:?}", t0.elapsed());
        }

        // Keep repainting while loading or fetching or generating MPR or coregistering or retrieving
        if self.image_loading.loading
            || self.quick_fetch.rx.is_some()
            || !matches!(self.background.loading_state, BackgroundLoadingState::Idle)
            || !self.pending_cache_hits.is_empty()
            || self.mpr.generating.is_some()
            || self.coregistration.mode().is_running()
            || self.database_window.is_retrieving()
        {
            ctx.request_repaint();
        }

        // Handle pending file dialog results
        if let Some(paths) = self.image_loading.pending_files.take() {
            self.load_files(paths);
        }

        // Handle dropped files
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                self.handle_dropped_files(&i.raw.dropped_files);
            }

            // Also check for hovered files (drag in progress)
            if !i.raw.hovered_files.is_empty() {
                self.status = format!("Drop {} file(s) to open", i.raw.hovered_files.len());
            }
        });

        // Handle Ctrl+O for open file dialog
        ctx.input(|i| {
            if i.modifiers.ctrl && i.key_pressed(egui::Key::O) {
                self.open_file_dialog();
            }
            // D to toggle Database window (patient browser + PACS query)
            // (but not Shift+D which deletes annotations)
            if i.key_pressed(egui::Key::D) && !i.modifiers.ctrl && !i.modifiers.shift {
                self.database_window.open = !self.database_window.open;
                if self.database_window.open {
                    self.database_window.request_load_studies();
                }
            }
            // B: cycle blend mode if fusion active, otherwise toggle patient sidebar
            if i.key_pressed(egui::Key::B) && !i.modifiers.ctrl {
                let slot = self.viewport_manager.get_active_mut();
                if slot.fusion_state.is_active() {
                    slot.fusion_state.cycle_blend_mode();
                    self.status = format!(
                        "Blend: {}",
                        slot.fusion_state.blend_mode.display_name()
                    );
                } else {
                    self.patient_sidebar.visible = !self.patient_sidebar.visible;
                }
            }
            // F1 to toggle toolbar
            if i.key_pressed(egui::Key::F1) {
                self.toolbar_visible = !self.toolbar_visible;
            }
            // Ctrl+H to toggle PHI (patient info) visibility
            if i.modifiers.ctrl && i.key_pressed(egui::Key::H) {
                self.viewport_manager.toggle_phi_hidden();
                if self.viewport_manager.phi_hidden() {
                    self.status = "PHI hidden".to_string();
                    self.patient_sidebar.visible = false;
                } else {
                    self.status = "PHI visible".to_string();
                }
            }
            // Ctrl+G for quick fetch from clipboard
            if i.modifiers.ctrl && i.key_pressed(egui::Key::G) {
                self.start_quick_fetch();
            }
            // R to toggle reference lines
            if i.key_pressed(egui::Key::R) && !i.modifiers.ctrl && !i.modifiers.alt {
                self.viewport_manager.toggle_reference_lines();
            }
            // Ctrl+R for coregistration mode
            if i.modifiers.ctrl && i.key_pressed(egui::Key::R) {
                if self.coregistration.mode().is_active() {
                    self.coregistration.cancel();
                    self.status = "Coregistration cancelled".to_string();
                    self.viewport_manager.clear_coregistration_state();
                } else {
                    self.coregistration.start();
                    self.status = self.coregistration.mode().status_text().to_string();
                    self.update_coregistration_visual_state();
                }
            }
            // Escape to cancel coregistration
            if i.key_pressed(egui::Key::Escape) && self.coregistration.mode().is_active() {
                self.coregistration.cancel();
                self.status = "Coregistration cancelled".to_string();
                self.viewport_manager.clear_coregistration_state();
            }
            // F: Toggle fusion on/off on active viewport
            if i.key_pressed(egui::Key::F) && !i.modifiers.ctrl && !i.modifiers.alt {
                let slot = &mut self.viewport_manager.get_active_mut();
                if slot.fusion_state.overlay_source_viewport.is_some() {
                    slot.fusion_state.toggle();
                    if slot.fusion_state.enabled {
                        self.status = format!(
                            "Fusion ON — {:.0}%",
                            slot.fusion_state.opacity * 100.0
                        );
                    } else {
                        self.status = "Fusion OFF".to_string();
                    }
                }
            }
            // M: Cycle colormap in fusion mode (Hot → Rainbow → Cool-Warm → Grayscale)
            if i.key_pressed(egui::Key::M) && !i.modifiers.ctrl {
                let slot = self.viewport_manager.get_active_mut();
                if slot.fusion_state.is_active() {
                    slot.fusion_state.cycle_colormap();
                    self.status = format!(
                        "Colormap: {}",
                        slot.fusion_state.colormap.display_name()
                    );
                }
            }
            // T: Reset threshold in fusion mode
            if i.key_pressed(egui::Key::T) && !i.modifiers.ctrl && !i.modifiers.shift {
                let slot = self.viewport_manager.get_active_mut();
                if slot.fusion_state.is_active() {
                    slot.fusion_state.reset_threshold();
                    self.status = "Threshold reset to 0%".to_string();
                }
            }
        });

        // Handle V+number for layout shortcuts (configurable)
        ctx.input(|i| {
            if i.key_pressed(egui::Key::V) {
                // V is pressed, now check for numbers
            }
            // Check for V modifier + number (V held while pressing number)
            // For simplicity, we'll use V then number in quick succession
            // or just use the number keys when V was recently pressed
            for (key, num) in [
                (egui::Key::Num1, "1"),
                (egui::Key::Num2, "2"),
                (egui::Key::Num3, "3"),
                (egui::Key::Num4, "4"),
                (egui::Key::Num5, "5"),
                (egui::Key::Num6, "6"),
                (egui::Key::Num7, "7"),
                (egui::Key::Num8, "8"),
            ] {
                if i.key_pressed(key) && i.modifiers.alt {
                    // Alt+Number for layout shortcuts
                    if let Some(layout_name) = self.settings.viewport.get_layout_for_key(num) {
                        if let Some(layout) = ViewportLayout::from_config_str(layout_name) {
                            self.viewport_manager.set_layout(layout);
                        }
                    }
                }
            }
        });

        // Handle scroll for image navigation (only when mouse is over a viewport, not over UI)
        // NOTE: is_pointer_over_area() calls ctx.input() internally, so it must be
        // called OUTSIDE ctx.input() closures to avoid deadlock.
        let pointer_over_ui = ctx.is_pointer_over_area();
        ctx.input(|i| {
            let scroll = i.raw_scroll_delta.y;
            if scroll != 0.0 && !pointer_over_ui {
                if i.modifiers.ctrl && i.modifiers.shift {
                    // Ctrl+Shift+scroll: adjust fusion threshold (5% steps)
                    let slot = self.viewport_manager.get_active_mut();
                    if slot.fusion_state.is_active() {
                        let delta = if scroll > 0.0 { 0.05 } else { -0.05 };
                        slot.fusion_state.adjust_threshold(delta);
                        self.status = format!(
                            "Threshold: {:.0}%",
                            slot.fusion_state.threshold * 100.0
                        );
                    }
                } else if i.modifiers.ctrl {
                    // Ctrl+scroll: adjust fusion opacity if fusion is active
                    let slot = self.viewport_manager.get_active_mut();
                    if slot.fusion_state.is_active() {
                        let delta = if scroll > 0.0 { 0.05 } else { -0.05 };
                        slot.fusion_state.adjust_opacity(delta);
                        self.status = format!(
                            "Fusion opacity: {:.0}%",
                            slot.fusion_state.opacity * 100.0
                        );
                    }
                } else {
                    // Normal scroll: navigate images
                    let mouse_pos = i.pointer.hover_pos();
                    let mouse_over_viewport = mouse_pos
                        .map(|pos| {
                            self.viewport_manager
                                .get_active()
                                .viewport
                                .last_rect()
                                .contains(pos)
                        })
                        .unwrap_or(false);

                    if mouse_over_viewport {
                        let delta = if scroll > 0.0 { -1 } else { 1 };
                        self.navigate(delta);
                    }
                }
            }
        });

        // Handle keyboard shortcuts
        ctx.input(|i| {
            // Arrow keys for navigation
            if i.key_pressed(egui::Key::ArrowUp) || i.key_pressed(egui::Key::ArrowLeft) {
                self.navigate(-1);
            }
            if i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::ArrowRight) {
                self.navigate(1);
            }

            // Page up/down for faster navigation
            if i.key_pressed(egui::Key::PageUp) {
                self.navigate(-10);
            }
            if i.key_pressed(egui::Key::PageDown) {
                self.navigate(10);
            }

            // Home/End for first/last image in active viewport
            if i.key_pressed(egui::Key::Home) {
                let active_id = self.viewport_manager.active_viewport();
                if let Some(slot) = self.viewport_manager.get_slot_mut(active_id) {
                    slot.current_index = 0;
                }
                self.load_current_image();
            }
            if i.key_pressed(egui::Key::End) {
                let active_id = self.viewport_manager.active_viewport();
                if let Some(slot) = self.viewport_manager.get_slot_mut(active_id) {
                    if !slot.series_files.is_empty() {
                        slot.current_index = slot.series_files.len() - 1;
                    }
                }
                self.load_current_image();
            }
        });

        // Handle window preset shortcuts (only affects active viewport)
        ctx.input(|i| {
            // Only process if not in a text field and Alt is not pressed (Alt+num is for layouts)
            if !i.modifiers.alt {
                // Number keys 1-9 for presets
                for (idx, key) in [
                    egui::Key::Num1,
                    egui::Key::Num2,
                    egui::Key::Num3,
                    egui::Key::Num4,
                    egui::Key::Num5,
                    egui::Key::Num6,
                    egui::Key::Num7,
                    egui::Key::Num8,
                    egui::Key::Num9,
                ]
                .iter()
                .enumerate()
                {
                    if i.key_pressed(*key) {
                        self.viewport_manager
                            .get_active_mut()
                            .viewport
                            .apply_preset(idx);
                    }
                }

                // 0 for auto window, Shift+0 for 1:1 pixel
                if i.key_pressed(egui::Key::Num0) {
                    if i.modifiers.shift {
                        self.viewport_manager
                            .get_active_mut()
                            .viewport
                            .set_one_to_one();
                    } else {
                        self.viewport_manager
                            .get_active_mut()
                            .viewport
                            .auto_window();
                    }
                }

                // Space for fit to window (smart or regular based on shift)
                if i.key_pressed(egui::Key::Space) {
                    let slot = self.viewport_manager.get_active_mut();
                    let size = slot.viewport.last_size();
                    if i.modifiers.shift {
                        // Shift+Space: regular fit to window
                        slot.viewport.fit_to_window(size);
                    } else {
                        // Space: smart fit to anatomy
                        slot.viewport.smart_fit_to_window(size);
                    }
                }
            }

            // Escape cancels measurement and returns to default mode (Windowing)
            if i.key_pressed(egui::Key::Escape) {
                self.interaction.measurement_state = MeasurementState::Idle;
                self.interaction.mode = InteractionMode::Windowing;
            }

            // Mode shortcuts (using configurable shortcuts)
            let shortcuts = &self.settings.shortcuts;

            // L: Length measurement mode
            if shortcuts.matches(&shortcuts.length_measure, egui::Key::L, &i.modifiers)
                && i.key_pressed(egui::Key::L)
            {
                self.interaction.mode = match self.interaction.mode {
                    InteractionMode::Measure => InteractionMode::Windowing,
                    _ => {
                        self.interaction.measurement_state = MeasurementState::Idle;
                        InteractionMode::Measure
                    }
                };
            }

            // O: Circle ROI mode
            if shortcuts.matches(&shortcuts.circle_roi, egui::Key::O, &i.modifiers)
                && i.key_pressed(egui::Key::O)
            {
                self.interaction.mode = match self.interaction.mode {
                    InteractionMode::CircleROI => InteractionMode::Windowing,
                    _ => {
                        self.interaction.measurement_state = MeasurementState::Idle;
                        InteractionMode::CircleROI
                    }
                };
            }

            // Y: Toggle sync (configurable, moved from S for MPR)
            if shortcuts.matches(&shortcuts.sync_toggle, egui::Key::Y, &i.modifiers)
                && i.key_pressed(egui::Key::Y)
            {
                self.viewport_manager.toggle_sync();
            }

            // W: Toggle between Windowing and Pan modes (no shift)
            if i.key_pressed(egui::Key::W) && !i.modifiers.shift {
                self.interaction.mode = match self.interaction.mode {
                    InteractionMode::Windowing => InteractionMode::Pan,
                    InteractionMode::Pan => InteractionMode::Windowing,
                    _ => InteractionMode::Windowing,
                };
            }

            // Shift+W: ROI-based windowing mode
            if i.key_pressed(egui::Key::W) && i.modifiers.shift {
                self.interaction.mode = match self.interaction.mode {
                    InteractionMode::ROIWindowing => InteractionMode::Windowing,
                    _ => {
                        self.interaction.measurement_state = MeasurementState::Idle;
                        InteractionMode::ROIWindowing
                    }
                };
            }

            // Shift+D: Delete all annotations on current image
            if shortcuts.matches(&shortcuts.delete_annotations, egui::Key::D, &i.modifiers)
                && i.key_pressed(egui::Key::D)
            {
                let slot = self.viewport_manager.get_active();
                if let Some(key) = slot.annotation_key() {
                    self.interaction.annotation_store.clear_image(&key);
                    self.status = "Annotations cleared".to_string();
                }
            }

            // Shift+A: Smart auto-windowing (non-CT)
            if shortcuts.matches(&shortcuts.smart_window, egui::Key::A, &i.modifiers)
                && i.key_pressed(egui::Key::A)
                && i.modifiers.shift
            // Only with Shift
            {
                let slot = self.viewport_manager.get_active_mut();
                if let Some(image) = slot.viewport.image() {
                    if image.is_ct() {
                        self.status = "Smart windowing not for CT - use presets (1-5)".to_string();
                    } else {
                        // Use anatomy bounds to find optimal window
                        if let Some((x_min, y_min, x_max, y_max)) = image.find_anatomy_bounds() {
                            // Calculate window from content area
                            let (wc, ww) =
                                image.calculate_content_window(x_min, y_min, x_max, y_max);
                            slot.viewport.set_window(wc, ww);
                            self.status = format!("Smart window: C={:.0} W={:.0}", wc, ww);
                        } else {
                            self.status = "Could not detect anatomy bounds".to_string();
                        }
                    }
                }
            }
        });

        // Handle right-click drag for smooth scrolling (only when not over UI)
        // Reuse pointer_over_ui from above (already computed outside ctx.input())
        ctx.input(|i| {
            let right_down = i.pointer.button_down(egui::PointerButton::Secondary);
            if right_down && !pointer_over_ui {
                if let Some(pos) = i.pointer.hover_pos() {
                    if let Some(last_y) = self.interaction.scroll_drag_last_y {
                        let delta_y = pos.y - last_y;
                        // Accumulate scroll - negative delta (drag up) = scroll forward
                        self.interaction.scroll_drag_accumulator += delta_y;
                        // Threshold for triggering scroll (pixels per image) - configurable
                        let threshold = self.settings.viewer.scroll_sensitivity;
                        while self.interaction.scroll_drag_accumulator >= threshold {
                            self.navigate(1);
                            self.interaction.scroll_drag_accumulator -= threshold;
                        }
                        while self.interaction.scroll_drag_accumulator <= -threshold {
                            self.navigate(-1);
                            self.interaction.scroll_drag_accumulator += threshold;
                        }
                    }
                    self.interaction.scroll_drag_last_y = Some(pos.y);
                }
            } else {
                // Reset when button released or when over UI
                self.interaction.scroll_drag_last_y = None;
                self.interaction.scroll_drag_accumulator = 0.0;
            }
        });

        // Handle measurement input (clicks for length, drag for circles)
        if matches!(
            self.interaction.mode,
            InteractionMode::Measure | InteractionMode::CircleROI | InteractionMode::ROIWindowing
        ) {
            self.handle_measurement_input(ctx);
            // Keep repainting during measurement for preview
            if self.interaction.measurement_state.is_drawing() {
                ctx.request_repaint();
            }
        }

        // Show the UI
        let t0 = std::time::Instant::now();
        self.show_main_ui(ctx);
        if t0.elapsed().as_millis() > 16 {
            tracing::warn!("SLOW: show_main_ui {:?}", t0.elapsed());
        }

        // Log total frame time if slow
        let total_frame = frame_start.elapsed();
        if total_frame.as_millis() > 50 {
            tracing::error!("VERY SLOW FRAME: {:?}", total_frame);
        } else if total_frame.as_millis() > 16 {
            tracing::warn!("SLOW FRAME: {:?}", total_frame);
        }
    }
}
