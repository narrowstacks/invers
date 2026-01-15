//! Film base estimation algorithms
//!
//! This module contains functions for estimating the film base (orange mask)
//! from scanned color negative images. The base estimation is critical for
//! accurate negative-to-positive conversion.

mod analysis;
mod extraction;
mod methods;

#[cfg(test)]
mod tests;

// Re-export public API
pub use methods::{
    estimate_base, estimate_base_from_border, estimate_base_from_histogram,
    estimate_base_from_manual_roi, estimate_base_from_regions,
};

use crate::decoders::DecodedImage;

// ============================================================================
// Constants
// ============================================================================

/// Minimum fraction of ROI pixels to sample for base estimation (1%).
/// Ensures we have enough samples even for very small ROIs.
pub(crate) const MIN_BASE_SAMPLE_FRACTION: f32 = 0.01;

/// Maximum fraction of ROI pixels to sample for base estimation (30%).
/// Limits sampling to avoid including image content.
pub(crate) const MAX_BASE_SAMPLE_FRACTION: f32 = 0.30;

/// Minimum brightness threshold for valid film base candidates (0.25).
/// Film base should be relatively bright; darker regions are likely image content.
pub(crate) const BASE_VALIDATION_MIN_BRIGHTNESS: f32 = 0.25;

/// Maximum noise threshold for valid film base candidates (0.15).
/// Clean film base should have low variance; high noise indicates image content.
pub(crate) const BASE_VALIDATION_MAX_NOISE: f32 = 0.15;

// ============================================================================
// Structs
// ============================================================================

/// Candidate region of interest (ROI) for film base estimation.
///
/// Represents a potential source region for sampling film base color,
/// typically from image borders or manually specified areas.
#[derive(Clone, Copy, Debug)]
pub struct BaseRoiCandidate {
    /// ROI rectangle as (x, y, width, height) in pixels.
    pub rect: (u32, u32, u32, u32),
    /// Average brightness of the region (0.0-1.0), weighted for orange mask detection.
    pub brightness: f32,
    /// Human-readable label for the region source (e.g., "top", "left", "manual").
    pub label: &'static str,
}

impl BaseRoiCandidate {
    pub fn new(rect: (u32, u32, u32, u32), brightness: f32, label: &'static str) -> Self {
        Self {
            rect,
            brightness,
            label,
        }
    }

    pub fn from_manual_roi(image: &DecodedImage, rect: (u32, u32, u32, u32)) -> Self {
        let brightness =
            extraction::sample_region_brightness(image, rect.0, rect.1, rect.2, rect.3);
        Self {
            rect,
            brightness,
            label: "manual",
        }
    }
}

/// Results from filtering base pixels to remove invalid samples.
///
/// When sampling film base, we need to exclude pixels that are:
/// - Clipped (near-white, indicating scanner saturation)
/// - Too dark (likely image content, not film base)
/// - Lacking color variation (not matching orange mask characteristics)
pub(crate) struct FilteredBasePixels {
    /// Valid pixels that passed all filtering criteria.
    pub pixels: Vec<[f32; 3]>,
    /// Ratio of pixels rejected due to clipping (0.0-1.0).
    /// High values (>0.5) indicate possible overexposed film base.
    pub clipped_ratio: f32,
    /// Ratio of pixels rejected as too dark (0.0-1.0).
    /// High values indicate the region may contain image content.
    pub dark_ratio: f32,
}
