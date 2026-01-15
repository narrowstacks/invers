//! Tone curve and tone mapping algorithms
//!
//! This module contains functions for applying tone curves to converted images.
//! Different curve types are available for various aesthetic looks including
//! S-curves, asymmetric film-like curves, and log-based cinematic curves.

mod apply;
mod curves;

#[cfg(test)]
mod tests;

// Re-export public API
pub use apply::apply_tone_curve;
pub use curves::{apply_asymmetric_curve, apply_log_curve, apply_s_curve, apply_s_curve_point};

/// Prevent values from ever hitting absolute black/white while retaining full range.
pub(crate) const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
pub(crate) const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

/// Threshold for switching to parallel processing (100k pixels * 3 channels)
pub(crate) const PARALLEL_THRESHOLD: usize = 300_000;

#[inline]
pub(crate) fn clamp_to_working_range(value: f32) -> f32 {
    value.clamp(WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING)
}
