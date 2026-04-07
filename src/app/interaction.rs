use crate::ui::{AnnotationStore, InteractionMode, MeasurementState};

pub(super) struct InteractionState {
    pub(super) mode: InteractionMode,
    pub(super) scroll_drag_accumulator: f32,
    pub(super) scroll_drag_last_y: Option<f32>,
    pub(super) annotation_store: AnnotationStore,
    pub(super) measurement_state: MeasurementState,
    pub(super) current_mouse_pos: Option<egui::Pos2>,
    pub(super) active_viewport_rect: Option<egui::Rect>,
}
