//! Type definitions for parameter testing infrastructure
//!
//! Contains the core data structures used for testing parameter combinations
//! and optimizing conversion results.

use crate::config;
use crate::decoders::{decode_image, DecodedImage};
use crate::models::{BaseEstimation, BaseSamplingMode, InversionMode, ShadowLiftMode};
use crate::pipeline::estimate_base;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Parameter set for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterTest {
    // Auto-levels settings
    pub enable_auto_levels: bool,
    pub clip_percent: f32,

    // Color neutralization
    pub enable_auto_color: bool,
    pub auto_color_strength: f32,
    pub auto_color_min_gain: f32,
    pub auto_color_max_gain: f32,
    pub auto_color_max_divergence: f32,

    // Base estimation
    pub base_brightest_percent: f32,
    pub base_sampling_mode: BaseSamplingMode,

    // Inversion
    pub inversion_mode: InversionMode,

    // Shadow/highlight adjustments
    pub shadow_lift_mode: ShadowLiftMode,
    pub shadow_lift_value: f32,
    pub highlight_compression: f32,

    // Auto exposure
    pub enable_auto_exposure: bool,
    pub auto_exposure_target_median: f32,
    pub auto_exposure_strength: f32,
    pub auto_exposure_min_gain: f32,
    pub auto_exposure_max_gain: f32,

    // Tone curve
    pub tone_curve_strength: f32,
    pub skip_tone_curve: bool,

    // Exposure
    pub exposure_compensation: f32,
}

impl Default for ParameterTest {
    fn default() -> Self {
        let handle = config::pipeline_config_handle();

        if let Some(defaults) = handle.config.testing.parameter_test_defaults.clone() {
            let mut defaults = defaults;
            defaults.sanitize();
            return defaults.into();
        }

        let mut defaults = config::ParameterTestDefaults::default();
        defaults.sanitize();
        defaults.into()
    }
}

/// Test result with scoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub params: ParameterTest,
    pub mae_r: f32,
    pub mae_g: f32,
    pub mae_b: f32,
    pub exposure_ratio: f32,
    pub color_shift: [f32; 3],
    pub contrast_ratio: f32,
    pub overall_score: f32,        // Lower is better
    pub base_estimation: [f32; 3], // RGB base values used
}

/// Pre-loaded test context to avoid redundant image loading and base estimation
/// This dramatically improves performance for grid/adaptive searches
pub struct PreloadedTestContext {
    /// Original negative image (pre-decoded)
    pub original: DecodedImage,
    /// Reference image to compare against (pre-decoded)
    pub reference: DecodedImage,
    /// Pre-computed base estimation from original
    pub base_estimation: BaseEstimation,
}

impl PreloadedTestContext {
    /// Create a new preloaded context by loading images and computing base estimation once
    pub fn new<P: AsRef<Path>>(original_path: P, reference_path: P) -> Result<Self, String> {
        eprintln!("[PRELOAD] Loading original negative...");
        let original = decode_image(original_path)?;
        eprintln!(
            "[PRELOAD] Original: {}x{}, {} channels",
            original.width, original.height, original.channels
        );

        eprintln!("[PRELOAD] Loading reference image...");
        let reference = decode_image(reference_path)?;
        eprintln!(
            "[PRELOAD] Reference: {}x{}, {} channels",
            reference.width, reference.height, reference.channels
        );

        eprintln!("[PRELOAD] Computing base estimation...");
        let base_estimation = estimate_base(&original, None, None, None)?;
        eprintln!(
            "[PRELOAD] Base RGB: [{:.4}, {:.4}, {:.4}]",
            base_estimation.medians[0], base_estimation.medians[1], base_estimation.medians[2]
        );

        Ok(Self {
            original,
            reference,
            base_estimation,
        })
    }
}

/// Parameter grid for exhaustive search
#[derive(Debug, Clone)]
pub struct ParameterGrid {
    pub auto_levels: Vec<bool>,
    pub clip_percent: Vec<f32>,
    pub auto_color: Vec<bool>,
    pub auto_color_strength: Vec<f32>,
    pub auto_color_min_gain: Vec<f32>,
    pub auto_color_max_gain: Vec<f32>,
    pub auto_color_max_divergence: Vec<f32>,
    pub base_brightest_percent: Vec<f32>,
    pub base_sampling_mode: Vec<BaseSamplingMode>,
    pub inversion_mode: Vec<InversionMode>,
    pub shadow_lift_mode: Vec<ShadowLiftMode>,
    pub shadow_lift_value: Vec<f32>,
    pub tone_curve_strength: Vec<f32>,
    pub exposure_compensation: Vec<f32>,
}

impl Default for ParameterGrid {
    fn default() -> Self {
        let handle = config::pipeline_config_handle();
        if let Some(values) = handle.config.testing.default_grid.clone() {
            let mut values = values;
            values.sanitize_with(config::TestingGridValues::default_grid());
            return values.into();
        }

        config::TestingGridValues::default_grid().into()
    }
}

impl ParameterGrid {
    /// Create a minimal grid for quick testing (12 combinations)
    pub fn minimal() -> Self {
        let handle = config::pipeline_config_handle();
        if let Some(values) = handle.config.testing.minimal_grid.clone() {
            let mut values = values;
            values.sanitize_with(config::TestingGridValues::minimal_grid());
            return values.into();
        }

        config::TestingGridValues::minimal_grid().into()
    }

    /// Create a comprehensive grid for thorough testing (~200 combinations)
    pub fn comprehensive() -> Self {
        let handle = config::pipeline_config_handle();
        if let Some(values) = handle.config.testing.comprehensive_grid.clone() {
            let mut values = values;
            values.sanitize_with(config::TestingGridValues::comprehensive_grid());
            return values.into();
        }

        config::TestingGridValues::comprehensive_grid().into()
    }
}
