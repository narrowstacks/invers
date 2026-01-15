//! Image processing pipeline
//!
//! Core pipeline for negative-to-positive conversion.
//!
//! This module is organized into submodules:
//! - `base_estimation`: Film base detection and measurement
//! - `inversion`: Negative-to-positive conversion algorithms
//! - `tone_mapping`: Tone curves and contrast adjustments
//! - `legacy`: Original Invers pipeline implementation
//! - `research`: Density-balance-first pipeline for eliminating color crossover
//! - `helpers`: Utility functions for color matrices, stats, and range enforcement

mod base_estimation;
mod helpers;
mod inversion;
mod legacy;
mod research;
mod tone_mapping;

#[cfg(test)]
mod tests;

// Re-export public items from submodules
pub use base_estimation::{
    estimate_base, estimate_base_from_border, estimate_base_from_histogram,
    estimate_base_from_manual_roi, estimate_base_from_regions, BaseRoiCandidate,
};
pub use helpers::{apply_color_matrix, compute_stats, enforce_working_range};
pub use inversion::{apply_reciprocal_inversion, invert_negative};
pub use tone_mapping::{
    apply_asymmetric_curve, apply_log_curve, apply_s_curve, apply_s_curve_point, apply_tone_curve,
};

// Re-export pub(crate) items for internal use
pub(crate) use helpers::{apply_scan_profile_fused, clamp_to_working_range};
pub(crate) use legacy::process_image_legacy;
pub(crate) use research::{apply_film_base_white_balance, process_image_research};

use crate::decoders::DecodedImage;
use crate::models::{ConvertOptions, PipelineMode};

#[cfg(feature = "gpu")]
use crate::gpu;

/// Prevent values from ever hitting absolute black/white while retaining full range.
pub const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
pub const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

/// Result of the processing pipeline
pub struct ProcessedImage {
    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,

    /// Processed linear RGB data (f32)
    pub data: Vec<f32>,

    /// Number of channels
    pub channels: u8,

    /// Whether to export as grayscale (single channel)
    /// Set true for B&W images to save space (1 channel instead of 3)
    pub export_as_grayscale: bool,
}

/// Execute the full processing pipeline
///
/// Routes to the appropriate pipeline based on `options.pipeline_mode`:
/// - Legacy: Original Invers pipeline with multiple inversion modes
/// - Research: Density-balance-first pipeline for eliminating color crossover
/// - CbStyle: Curve-based pipeline inspired by Negative Lab Pro
pub fn process_image(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    match options.pipeline_mode {
        PipelineMode::Legacy => process_image_legacy(image, options),
        PipelineMode::Research => process_image_research(image, options),
        PipelineMode::CbStyle => {
            #[cfg(feature = "gpu")]
            if options.use_gpu && gpu::is_gpu_available() {
                if options.debug {
                    if let Some(info) = gpu::gpu_info() {
                        eprintln!("[DEBUG] Using GPU acceleration (CB pipeline): {}", info);
                    }
                }

                match gpu::process_image_cb_gpu(&image, options) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        eprintln!(
                            "[WARN] GPU CB processing failed, falling back to CPU: {}",
                            e
                        );
                    }
                }
            }

            #[cfg(feature = "gpu")]
            if options.use_gpu && !gpu::is_gpu_available() && options.debug {
                eprintln!("[DEBUG] GPU requested but not available, using CPU");
            }

            crate::cb_pipeline::process_image_cb(image, options)
        }
    }
}
