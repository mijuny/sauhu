//! DICOM module for Sauhu
//!
//! Handles DICOM file parsing, pixel data extraction, and metadata.
#![allow(unused_imports)]

mod anonymize;
mod geometry;
mod image;
mod mpr;
mod parser;
mod series_utils;
mod spatial;

pub use anonymize::*;
pub use geometry::*;
pub use image::*;
pub use mpr::*;
pub use parser::*;
pub use series_utils::*;
pub use spatial::{
    compute_circle_roi_stats, compute_distance_mm, compute_patient_coord,
    compute_point_across_planes, compute_point_in_plane, compute_reference_line,
    compute_reference_lines_to_planes, compute_sync_slice, find_closest_slice,
    should_planes_sync,
};
