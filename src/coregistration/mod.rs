//! Coregistration module for aligning images from different imaging sessions
//!
//! Provides fast rigid registration (6-7 DOF) using multi-resolution optimization.
//!
//! ## Usage
//!
//! ```ignore
//! use coregistration::{RegistrationPipeline, RegistrationConfig};
//!
//! let config = RegistrationConfig::brain_mri();
//! let pipeline = RegistrationPipeline::new(config);
//!
//! let result = pipeline.run_cpu(&target_volume, &source_volume);
//! ```
//!
//! ## Workflow
//!
//! 1. User presses Ctrl+R to enter coregistration mode
//! 2. User clicks on target series (fixed reference)
//! 3. User clicks on source series (to be transformed)
//! 4. Registration runs in background with progress indicator
//! 5. Source viewport is updated with coregistered series
#![allow(unused_imports)]
#![allow(dead_code)]

mod manager;
mod metrics;
mod optimizer;
mod pipeline;
mod transform;

pub use manager::{CoregistrationManager, CoregistrationMode};
pub use optimizer::{PowellOptimizer, PyramidSchedule};
pub use pipeline::{
    apply_transform_to_volume, compute_initial_alignment, RegistrationConfig, RegistrationPipeline,
    RegistrationResult, VolumeGeometry,
};
pub use transform::RigidTransform;
