//! Local study browser tab

use super::anonymize::{AnonymizePatientGroup, AnonymizeState};
use super::{DeleteState, OpenStudyAction};
use crate::db::{delete_study, StudySource};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::PathBuf;

/// Sort column for local studies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum LocalSortColumn {
    Name,
    Id,
    Accession,
    Modality,
    Description,
    #[default]
    Date,
    Images,
}

impl super::DatabaseWindow {
    pub(super) fn show_local_tab(&mut self, ui: &mut egui::Ui, conn: &Connection) {
        // Show dialogs at the TOP if active (before anything else)
        self.show_delete_ui(ui, conn);
        self.show_anonymization_ui(ui, conn);

        // Toolbar
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.text_edit_singleline(&mut self.local_filter);
            if ui.button("Refresh").clicked() && !self.loading_studies {
                self.request_load_studies();
            }
            if self.loading_studies {
                ui.spinner();
            }
        });

        ui.separator();

        // Filter and sort studies
        let filter_lower = self.local_filter.to_lowercase();
        let sort_col = self.local_sort_column;
        let sort_asc = self.local_sort_ascending;

        // Build filtered list with extracted data for sorting
        let mut filtered: Vec<_> = self
            .local_studies
            .iter()
            .enumerate()
            .filter(|(_, sw)| {
                if filter_lower.is_empty() {
                    return true;
                }
                let s = &sw.study;
                s.patient_name
                    .as_ref()
                    .map(|n| n.to_lowercase().contains(&filter_lower))
                    .unwrap_or(false)
                    || s.patient_id
                        .as_ref()
                        .map(|id| id.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
                    || s.accession_number
                        .as_ref()
                        .map(|a| a.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
                    || s.modality
                        .as_ref()
                        .map(|m| m.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
                    || s.study_description
                        .as_ref()
                        .map(|d| d.to_lowercase().contains(&filter_lower))
                        .unwrap_or(false)
            })
            .map(|(idx, sw)| {
                let s = &sw.study;
                (
                    idx,
                    s.patient_name.clone().unwrap_or_default(),
                    s.patient_id.clone().unwrap_or_default(),
                    s.accession_number.clone().unwrap_or_default(),
                    s.modality.clone().unwrap_or_default(),
                    s.study_description.clone().unwrap_or_default(),
                    s.study_date.clone().unwrap_or_default(),
                    sw.num_images,
                    sw.study.clone(),
                )
            })
            .collect();

        // Sort based on current column and direction
        filtered.sort_by(|a, b| {
            let cmp = match sort_col {
                LocalSortColumn::Name => a.1.cmp(&b.1),
                LocalSortColumn::Id => a.2.cmp(&b.2),
                LocalSortColumn::Accession => a.3.cmp(&b.3),
                LocalSortColumn::Modality => a.4.cmp(&b.4),
                LocalSortColumn::Description => a.5.cmp(&b.5),
                LocalSortColumn::Date => a.6.cmp(&b.6),
                LocalSortColumn::Images => a.7.cmp(&b.7),
            };
            if sort_asc {
                cmp
            } else {
                cmp.reverse()
            }
        });

        // Pre-compute sort indicators
        let sort_indicator = |col: LocalSortColumn| -> &'static str {
            if sort_col == col {
                if sort_asc {
                    " ▲"
                } else {
                    " ▼"
                }
            } else {
                ""
            }
        };
        let name_header = format!("Patient Name{}", sort_indicator(LocalSortColumn::Name));
        let id_header = format!("Patient ID{}", sort_indicator(LocalSortColumn::Id));
        let accession_header = format!("Accession{}", sort_indicator(LocalSortColumn::Accession));
        let modality_header = format!("Modality{}", sort_indicator(LocalSortColumn::Modality));
        let desc_header = format!(
            "Description{}",
            sort_indicator(LocalSortColumn::Description)
        );
        let date_header = format!("Date{}", sort_indicator(LocalSortColumn::Date));
        let images_header = format!("Images{}", sort_indicator(LocalSortColumn::Images));

        // Track which column was clicked
        let mut clicked_column: Option<LocalSortColumn> = None;
        // Track click info: (study_id, filtered_position, modifiers)
        let mut clicked_study_info: Option<(i64, usize, egui::Modifiers)> = None;
        let mut double_clicked_study_id: Option<i64> = None;

        // Build filtered_ids for range selection
        let filtered_ids: Vec<i64> = filtered
            .iter()
            .map(|(_, _, _, _, _, _, _, _, s)| s.id)
            .collect();

        let available_height = ui.available_height() - 40.0;
        egui::ScrollArea::vertical()
            .max_height(available_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("local_studies_table")
                    .num_columns(7)
                    .striped(true)
                    .min_col_width(60.0)
                    .show(ui, |ui| {
                        // Header row with sortable columns
                        if ui.selectable_label(false, &name_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Name);
                        }
                        if ui.selectable_label(false, &id_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Id);
                        }
                        if ui.selectable_label(false, &accession_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Accession);
                        }
                        if ui.selectable_label(false, &modality_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Modality);
                        }
                        if ui.selectable_label(false, &desc_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Description);
                        }
                        if ui.selectable_label(false, &date_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Date);
                        }
                        if ui.selectable_label(false, &images_header).clicked() {
                            clicked_column = Some(LocalSortColumn::Images);
                        }
                        ui.end_row();

                        // Data rows
                        for (
                            filtered_pos,
                            (
                                _,
                                name,
                                id,
                                accession,
                                modality,
                                description,
                                date,
                                num_images,
                                study,
                            ),
                        ) in filtered.iter().enumerate()
                        {
                            let selected = self.selected_studies.contains(&study.id);
                            let formatted_date = format_date(date);

                            let name_resp = ui.selectable_label(selected, name);
                            ui.label(id);
                            ui.label(accession);
                            ui.label(modality);
                            ui.label(description);
                            ui.label(&formatted_date);
                            ui.label(num_images.to_string());
                            ui.end_row();

                            if name_resp.clicked() {
                                let modifiers = ui.input(|i| i.modifiers);
                                clicked_study_info = Some((study.id, filtered_pos, modifiers));
                            }
                            if name_resp.double_clicked() {
                                double_clicked_study_id = Some(study.id);
                            }
                        }
                    });

                if filtered.is_empty() {
                    ui.colored_label(
                        egui::Color32::GRAY,
                        "No local studies. Load DICOM files or retrieve from PACS.",
                    );
                }
            });

        // Handle clicks after the UI code (to avoid borrow conflicts)
        if let Some(col) = clicked_column {
            self.toggle_sort(col);
        }
        if let Some((study_id, filtered_pos, modifiers)) = clicked_study_info {
            self.handle_study_click(study_id, filtered_pos, &filtered_ids, modifiers);
        }
        if let Some(study_id) = double_clicked_study_id {
            // Find the study and open it
            if let Some(sw) = self.local_studies.iter().find(|sw| sw.study.id == study_id) {
                // Select just this study
                self.selected_studies.clear();
                self.selected_studies.insert(study_id);
                self.last_selected_id = Some(study_id);
                self.pending_action = Some(OpenStudyAction {
                    study: sw.study.clone(),
                });
                self.open = false;
            }
        }

        // Handle Delete key
        if ui.input(|i| i.key_pressed(egui::Key::Delete))
            && !self.selected_studies.is_empty()
            && matches!(self.delete_state, DeleteState::Idle)
        {
            let study_ids: Vec<i64> = self.selected_studies.iter().cloned().collect();
            self.execute_delete(conn, study_ids);
        }

        ui.separator();

        // Action buttons
        ui.horizontal(|ui| {
            // Open Study button - only enable for single selection
            let can_open = self.selected_studies.len() == 1;
            if ui
                .add_enabled(can_open, egui::Button::new("Open Study"))
                .clicked()
            {
                if let Some(&study_id) = self.selected_studies.iter().next() {
                    if let Some(sw) = self.local_studies.iter().find(|sw| sw.study.id == study_id) {
                        self.pending_action = Some(OpenStudyAction {
                            study: sw.study.clone(),
                        });
                        self.open = false;
                    }
                }
            }

            // Anonymize button - works with any selection (groups by patient)
            let can_anonymize = !self.selected_studies.is_empty()
                && matches!(self.anonymize_state, AnonymizeState::Idle);
            if ui
                .add_enabled(can_anonymize, egui::Button::new("Anonymize..."))
                .clicked()
            {
                tracing::info!(
                    "Anonymize button clicked with {} studies selected",
                    self.selected_studies.len()
                );

                // Collect selected studies
                let selected: Vec<crate::db::Study> = self
                    .selected_studies
                    .iter()
                    .filter_map(|&id| {
                        self.local_studies
                            .iter()
                            .find(|sw| sw.study.id == id)
                            .map(|sw| sw.study.clone())
                    })
                    .collect();

                // Group by patient_id
                let mut patient_map: HashMap<String, AnonymizePatientGroup> = HashMap::new();
                for study in selected {
                    let patient_id = study.patient_id.clone().unwrap_or_default();
                    let patient_name = study.patient_name.clone().unwrap_or_default();

                    let group = patient_map.entry(patient_id.clone()).or_insert_with(|| {
                        AnonymizePatientGroup {
                            patient_id,
                            patient_name,
                            studies: Vec::new(),
                            num_files: 0,
                        }
                    });
                    group.studies.push(study);
                }

                // Count files for each group
                let mut patient_groups: Vec<AnonymizePatientGroup> =
                    patient_map.into_values().collect();
                let mut total_files = 0u32;
                for group in &mut patient_groups {
                    group.num_files = super::anonymize::count_patient_dicom_files(&group.studies);
                    total_files += group.num_files;
                }

                // Sort by patient name for consistent display
                patient_groups.sort_by(|a, b| a.patient_name.cmp(&b.patient_name));

                tracing::info!(
                    "Grouped into {} patients with {} total files",
                    patient_groups.len(),
                    total_files
                );

                self.anonymize_state = AnonymizeState::Confirming {
                    patient_groups,
                    total_files,
                };
            }

            // Delete button - works with any selection
            let can_delete =
                !self.selected_studies.is_empty() && matches!(self.delete_state, DeleteState::Idle);
            if ui
                .add_enabled(can_delete, egui::Button::new("Delete"))
                .clicked()
            {
                let study_ids: Vec<i64> = self.selected_studies.iter().cloned().collect();
                self.execute_delete(conn, study_ids);
            }
        });
    }

    /// Toggle sort on a column
    pub(super) fn toggle_sort(&mut self, column: LocalSortColumn) {
        if self.local_sort_column == column {
            self.local_sort_ascending = !self.local_sort_ascending;
        } else {
            self.local_sort_column = column;
            // Default to descending for date, ascending for others
            self.local_sort_ascending = column != LocalSortColumn::Date;
        }
    }

    /// Handle study click with modifier keys for multi-select
    pub(super) fn handle_study_click(
        &mut self,
        study_id: i64,
        filtered_pos: usize,
        filtered_ids: &[i64],
        modifiers: egui::Modifiers,
    ) {
        if modifiers.command {
            // Ctrl+Click (or Cmd on Mac): toggle selection
            if !self.selected_studies.remove(&study_id) {
                self.selected_studies.insert(study_id);
            }
            // Don't update anchor for Ctrl+Click
        } else if modifiers.shift {
            // Shift+Click: range selection from anchor
            if let Some(anchor_id) = self.last_selected_id {
                if let Some(anchor_pos) = filtered_ids.iter().position(|&id| id == anchor_id) {
                    self.selected_studies.clear();
                    let (start, end) = (anchor_pos.min(filtered_pos), anchor_pos.max(filtered_pos));
                    for &id in &filtered_ids[start..=end] {
                        self.selected_studies.insert(id);
                    }
                }
            } else {
                // No anchor, just select clicked item
                self.selected_studies.clear();
                self.selected_studies.insert(study_id);
            }
            self.last_selected_id = Some(study_id);
        } else {
            // Plain click: single selection
            self.selected_studies.clear();
            self.selected_studies.insert(study_id);
            self.last_selected_id = Some(study_id);
        }
    }

    /// Show delete error if any
    pub(super) fn show_delete_ui(&mut self, ui: &mut egui::Ui, _conn: &Connection) {
        if let DeleteState::Error { message } = &self.delete_state {
            ui.separator();
            ui.colored_label(egui::Color32::RED, message);
            if ui.button("OK").clicked() {
                self.delete_state = DeleteState::Idle;
            }
        }
    }

    /// Execute delete for the given study IDs
    pub(super) fn execute_delete(&mut self, conn: &Connection, study_ids: Vec<i64>) {
        let mut errors = Vec::new();

        for study_id in &study_ids {
            // Get study info before deleting
            if let Some(sw) = self
                .local_studies
                .iter()
                .find(|sw| sw.study.id == *study_id)
            {
                let study = &sw.study;

                // Delete files for PACS studies only
                if study.source == StudySource::Pacs {
                    let path = PathBuf::from(&study.file_path);
                    if path.exists() {
                        if let Err(e) = std::fs::remove_dir_all(&path) {
                            tracing::warn!("Failed to delete PACS cache at {:?}: {}", path, e);
                            // Continue anyway - DB delete is more important
                        } else {
                            tracing::info!("Deleted PACS cache at {:?}", path);
                        }
                    }
                }

                // Delete from database (cascade handles series/measurements)
                if let Err(e) = delete_study(conn, *study_id) {
                    errors.push(format!("Failed to delete study {}: {}", study_id, e));
                } else {
                    tracing::info!("Deleted study {} from database", study_id);
                }
            }
        }

        // Clear selection
        self.selected_studies.clear();
        self.last_selected_id = None;

        // Request study list refresh
        self.request_load_studies();

        // Update state
        if errors.is_empty() {
            self.delete_state = DeleteState::Idle;
            tracing::info!("Deleted {} studies", study_ids.len());
        } else {
            self.delete_state = DeleteState::Error {
                message: errors.join("; "),
            };
        }
    }
}

/// Format DICOM date (YYYYMMDD) for display
pub(super) fn format_date(date: &str) -> String {
    if date.len() == 8 {
        format!("{}-{}-{}", &date[0..4], &date[4..6], &date[6..8])
    } else {
        date.to_string()
    }
}
