//! Parameter testing and optimization infrastructure
//!
//! Provides tools for testing different parameter combinations to optimize
//! conversion results against reference images.

use crate::config;
use crate::decoders::{decode_image, DecodedImage};
use crate::diagnostics::{compare_conversions, DiagnosticReport};
use crate::models::{
    BaseEstimation, BaseSamplingMode, ConvertOptions, InversionMode, OutputFormat, ShadowLiftMode,
};
use crate::pipeline::{estimate_base, process_image};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

/// Run a single parameter test
pub fn run_parameter_test<P: AsRef<Path>>(
    original_path: P,
    reference_path: P,
    params: &ParameterTest,
    save_path: Option<P>,
) -> Result<TestResult, String> {
    // Load original negative
    let original = decode_image(original_path)?;

    // Load reference (third-party) conversion
    let reference = decode_image(reference_path)?;

    // Estimate base from original
    let base_estimation = estimate_base(&original, None, None, None)?;
    let base_rgb = base_estimation.medians;

    // Build conversion options from test parameters
    let options = ConvertOptions {
        input_paths: vec![],
        output_dir: std::path::PathBuf::from("."),
        output_format: OutputFormat::Tiff16,
        working_colorspace: "linear-rec2020".to_string(),
        bit_depth_policy: crate::models::BitDepthPolicy::Force16Bit,
        film_preset: None,
        scan_profile: None,
        base_estimation: Some(base_estimation),
        num_threads: None,
        skip_tone_curve: params.skip_tone_curve,
        skip_color_matrix: true, // Skip for testing
        exposure_compensation: params.exposure_compensation,
        debug: false,
        enable_auto_levels: params.enable_auto_levels,
        auto_levels_clip_percent: params.clip_percent,
        preserve_headroom: false,
        enable_auto_color: params.enable_auto_color,
        auto_color_strength: params.auto_color_strength,
        auto_color_min_gain: params.auto_color_min_gain,
        auto_color_max_gain: params.auto_color_max_gain,
        auto_color_max_divergence: params.auto_color_max_divergence,
        base_brightest_percent: params.base_brightest_percent,
        base_sampling_mode: params.base_sampling_mode,
        base_estimation_method: crate::models::BaseEstimationMethod::default(),
        auto_levels_mode: crate::models::AutoLevelsMode::default(),
        inversion_mode: params.inversion_mode,
        shadow_lift_mode: params.shadow_lift_mode,
        shadow_lift_value: params.shadow_lift_value,
        highlight_compression: params.highlight_compression,
        enable_auto_exposure: params.enable_auto_exposure,
        auto_exposure_target_median: params.auto_exposure_target_median,
        auto_exposure_strength: params.auto_exposure_strength,
        auto_exposure_min_gain: params.auto_exposure_min_gain,
        auto_exposure_max_gain: params.auto_exposure_max_gain,
        no_clip: false,
        enable_auto_wb: false,
        auto_wb_strength: 1.0,
        auto_wb_mode: crate::models::AutoWbMode::default(),
        use_gpu: false,
        // Research pipeline options (use defaults for legacy testing)
        pipeline_mode: crate::models::PipelineMode::Legacy,
        density_balance: None,
        neutral_point: None,
        density_balance_red: None,
        density_balance_blue: None,
        tone_curve_override: None,
        // Curves-based pipeline options
        cb_options: None,
    };

    // Process with our pipeline
    let processed = process_image(original, &options)?;

    // Save processed image if requested
    if let Some(path) = save_path {
        use crate::exporters::export_tiff16;
        export_tiff16(&processed, path.as_ref(), None)?;
    }

    // Compare against reference
    let report = compare_conversions(&processed, &reference)?;

    // Calculate overall score (weighted combination of errors)
    let overall_score = calculate_score(&report);

    Ok(TestResult {
        params: params.clone(),
        mae_r: report.difference_stats[0].mean,
        mae_g: report.difference_stats[1].mean,
        mae_b: report.difference_stats[2].mean,
        exposure_ratio: report.exposure_ratio,
        color_shift: report.color_shift,
        contrast_ratio: calculate_contrast_ratio(&report),
        overall_score,
        base_estimation: base_rgb,
    })
}

/// Run a parameter test using preloaded context (much faster for grid searches)
/// This avoids redundant image loading and base estimation
pub fn run_parameter_test_preloaded(
    ctx: &PreloadedTestContext,
    params: &ParameterTest,
) -> Result<TestResult, String> {
    let base_rgb = ctx.base_estimation.medians;

    // Build conversion options from test parameters
    let options = ConvertOptions {
        input_paths: vec![],
        output_dir: std::path::PathBuf::from("."),
        output_format: OutputFormat::Tiff16,
        working_colorspace: "linear-rec2020".to_string(),
        bit_depth_policy: crate::models::BitDepthPolicy::Force16Bit,
        film_preset: None,
        scan_profile: None,
        base_estimation: Some(ctx.base_estimation.clone()),
        num_threads: None,
        skip_tone_curve: params.skip_tone_curve,
        skip_color_matrix: true, // Skip for testing
        exposure_compensation: params.exposure_compensation,
        debug: false,
        enable_auto_levels: params.enable_auto_levels,
        auto_levels_clip_percent: params.clip_percent,
        preserve_headroom: false,
        enable_auto_color: params.enable_auto_color,
        auto_color_strength: params.auto_color_strength,
        auto_color_min_gain: params.auto_color_min_gain,
        auto_color_max_gain: params.auto_color_max_gain,
        auto_color_max_divergence: params.auto_color_max_divergence,
        base_brightest_percent: params.base_brightest_percent,
        base_sampling_mode: params.base_sampling_mode,
        base_estimation_method: crate::models::BaseEstimationMethod::default(),
        auto_levels_mode: crate::models::AutoLevelsMode::default(),
        inversion_mode: params.inversion_mode,
        shadow_lift_mode: params.shadow_lift_mode,
        shadow_lift_value: params.shadow_lift_value,
        highlight_compression: params.highlight_compression,
        enable_auto_exposure: params.enable_auto_exposure,
        auto_exposure_target_median: params.auto_exposure_target_median,
        auto_exposure_strength: params.auto_exposure_strength,
        auto_exposure_min_gain: params.auto_exposure_min_gain,
        auto_exposure_max_gain: params.auto_exposure_max_gain,
        no_clip: false,
        enable_auto_wb: false,
        auto_wb_strength: 1.0,
        auto_wb_mode: crate::models::AutoWbMode::default(),
        use_gpu: false,
        // Research pipeline options (use defaults for legacy testing)
        pipeline_mode: crate::models::PipelineMode::Legacy,
        density_balance: None,
        neutral_point: None,
        density_balance_red: None,
        density_balance_blue: None,
        tone_curve_override: None,
        // Curves-based pipeline options
        cb_options: None,
    };

    // Process with our pipeline - clone original since process_image consumes it
    let processed = process_image(ctx.original.clone(), &options)?;

    // Compare against reference
    let report = compare_conversions(&processed, &ctx.reference)?;

    // Calculate overall score (weighted combination of errors)
    let overall_score = calculate_score(&report);

    Ok(TestResult {
        params: params.clone(),
        mae_r: report.difference_stats[0].mean,
        mae_g: report.difference_stats[1].mean,
        mae_b: report.difference_stats[2].mean,
        exposure_ratio: report.exposure_ratio,
        color_shift: report.color_shift,
        contrast_ratio: calculate_contrast_ratio(&report),
        overall_score,
        base_estimation: base_rgb,
    })
}

/// Calculate contrast ratio from diagnostic report
fn calculate_contrast_ratio(report: &DiagnosticReport) -> f32 {
    let our_contrast = report.our_stats.iter().map(|s| s.max - s.min).sum::<f32>() / 3.0;
    let tp_contrast = report
        .third_party_stats
        .iter()
        .map(|s| s.max - s.min)
        .sum::<f32>()
        / 3.0;

    if our_contrast > 0.0 {
        tp_contrast / our_contrast
    } else {
        1.0
    }
}

impl From<config::ParameterTestDefaults> for ParameterTest {
    fn from(defaults: config::ParameterTestDefaults) -> Self {
        Self {
            enable_auto_levels: defaults.enable_auto_levels,
            clip_percent: defaults.clip_percent,
            enable_auto_color: defaults.enable_auto_color,
            auto_color_strength: defaults.auto_color_strength,
            auto_color_min_gain: defaults.auto_color_min_gain,
            auto_color_max_gain: defaults.auto_color_max_gain,
            auto_color_max_divergence: defaults.auto_color_max_divergence,
            base_brightest_percent: defaults.base_brightest_percent,
            base_sampling_mode: defaults.base_sampling_mode,
            inversion_mode: defaults.inversion_mode,
            shadow_lift_mode: defaults.shadow_lift_mode,
            shadow_lift_value: defaults.shadow_lift_value,
            highlight_compression: defaults.highlight_compression,
            enable_auto_exposure: defaults.enable_auto_exposure,
            auto_exposure_target_median: defaults.auto_exposure_target_median,
            auto_exposure_strength: defaults.auto_exposure_strength,
            auto_exposure_min_gain: defaults.auto_exposure_min_gain,
            auto_exposure_max_gain: defaults.auto_exposure_max_gain,
            tone_curve_strength: defaults.tone_curve_strength,
            skip_tone_curve: defaults.skip_tone_curve,
            exposure_compensation: defaults.exposure_compensation,
        }
    }
}

impl From<config::TestingGridValues> for ParameterGrid {
    fn from(values: config::TestingGridValues) -> Self {
        Self {
            auto_levels: values.auto_levels,
            clip_percent: values.clip_percent,
            auto_color: values.auto_color,
            auto_color_strength: values.auto_color_strength,
            auto_color_min_gain: values.auto_color_min_gain,
            auto_color_max_gain: values.auto_color_max_gain,
            auto_color_max_divergence: values.auto_color_max_divergence,
            base_brightest_percent: values.base_brightest_percent,
            base_sampling_mode: values.base_sampling_mode,
            inversion_mode: values.inversion_mode,
            shadow_lift_mode: values.shadow_lift_mode,
            shadow_lift_value: values.shadow_lift_value,
            tone_curve_strength: values.tone_curve_strength,
            exposure_compensation: values.exposure_compensation,
        }
    }
}

/// Calculate overall score from diagnostic report
/// Lower scores are better (closer to reference)
fn calculate_score(report: &DiagnosticReport) -> f32 {
    // Weight different metrics
    let mae_weight = 1.0;
    let exposure_weight = 2.0;
    let color_shift_weight = 2.0;
    let contrast_weight = 0.5;

    // Average MAE across channels
    let mae_avg = (report.difference_stats[0].mean
        + report.difference_stats[1].mean
        + report.difference_stats[2].mean)
        / 3.0;

    // Exposure error (deviation from 1.0)
    let exposure_error = (report.exposure_ratio - 1.0).abs();

    // Color shift magnitude
    let color_shift_mag = (report.color_shift[0].powi(2)
        + report.color_shift[1].powi(2)
        + report.color_shift[2].powi(2))
    .sqrt();

    // Contrast error
    let contrast_ratio = calculate_contrast_ratio(report);
    let contrast_error = (contrast_ratio - 1.0).abs();

    // Weighted sum
    mae_avg * mae_weight
        + exposure_error * exposure_weight
        + color_shift_mag * color_shift_weight
        + contrast_error * contrast_weight
}

/// Run parameter grid search
pub fn run_parameter_grid_search<P: AsRef<Path>>(
    original_path: P,
    reference_path: P,
    grid: &ParameterGrid,
) -> Result<Vec<TestResult>, String> {
    let original_path = original_path.as_ref();
    let reference_path = reference_path.as_ref();

    let mut results = Vec::new();
    let mut total_tests = 0;

    // Calculate total number of tests
    let total_combinations = grid.auto_levels.len()
        * grid.clip_percent.len()
        * grid.auto_color.len()
        * grid.auto_color_strength.len()
        * grid.auto_color_min_gain.len()
        * grid.auto_color_max_gain.len()
        * grid.base_brightest_percent.len()
        * grid.base_sampling_mode.len()
        * grid.inversion_mode.len()
        * grid.shadow_lift_mode.len()
        * grid.shadow_lift_value.len()
        * grid.tone_curve_strength.len()
        * grid.exposure_compensation.len();

    eprintln!(
        "[GRID SEARCH] Testing {} parameter combinations...",
        total_combinations
    );

    // Iterate through all combinations
    for &auto_levels in &grid.auto_levels {
        for &clip_percent in &grid.clip_percent {
            for &auto_color in &grid.auto_color {
                for &auto_color_strength in &grid.auto_color_strength {
                    for &auto_color_min_gain in &grid.auto_color_min_gain {
                        for &auto_color_max_gain in &grid.auto_color_max_gain {
                            for &base_brightest_percent in &grid.base_brightest_percent {
                                for &base_sampling_mode in &grid.base_sampling_mode {
                                    for &inversion_mode in &grid.inversion_mode {
                                        for &shadow_lift_mode in &grid.shadow_lift_mode {
                                            for &shadow_lift_value in &grid.shadow_lift_value {
                                                for &tone_curve_strength in
                                                    &grid.tone_curve_strength
                                                {
                                                    for &exposure_compensation in
                                                        &grid.exposure_compensation
                                                    {
                                                        total_tests += 1;

                                                        let params = ParameterTest {
                                                            enable_auto_levels: auto_levels,
                                                            clip_percent,
                                                            enable_auto_color: auto_color,
                                                            auto_color_strength,
                                                            auto_color_min_gain,
                                                            auto_color_max_gain,
                                                            auto_color_max_divergence: grid
                                                                .auto_color_max_divergence
                                                                .first()
                                                                .copied()
                                                                .unwrap_or(0.15),
                                                            base_brightest_percent,
                                                            base_sampling_mode,
                                                            inversion_mode,
                                                            shadow_lift_mode,
                                                            shadow_lift_value,
                                                            highlight_compression: 1.0,
                                                            enable_auto_exposure: true,
                                                            auto_exposure_target_median: 0.25,
                                                            auto_exposure_strength: 1.0,
                                                            auto_exposure_min_gain: 0.6,
                                                            auto_exposure_max_gain: 1.4,
                                                            tone_curve_strength,
                                                            skip_tone_curve: false,
                                                            exposure_compensation,
                                                        };

                                                        match run_parameter_test(
                                                            original_path,
                                                            reference_path,
                                                            &params,
                                                            None::<&Path>,
                                                        ) {
                                                            Ok(result) => {
                                                                if total_tests % 10 == 0 {
                                                                    eprintln!(
                                                                "[GRID SEARCH] Progress: {}/{}",
                                                                total_tests, total_combinations
                                                            );
                                                                }
                                                                results.push(result);
                                                            }
                                                            Err(e) => {
                                                                eprintln!(
                                                            "[GRID SEARCH] Test {} failed: {}",
                                                            total_tests, e
                                                        );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by overall score (best first)
    results.sort_by(|a, b| {
        a.overall_score
            .partial_cmp(&b.overall_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!(
        "[GRID SEARCH] Complete! Tested {} combinations",
        total_tests
    );

    Ok(results)
}

/// Print test result summary
pub fn print_test_result(result: &TestResult, rank: usize) {
    println!("\n{:=<80}", "");
    println!(
        "PARAMETER SET #{}: (Score: {:.4})",
        rank, result.overall_score
    );
    println!("{:-<80}", "");

    println!("\nBase Estimation:");
    println!(
        "  RGB Base Values:    [{:.4}, {:.4}, {:.4}]",
        result.base_estimation[0], result.base_estimation[1], result.base_estimation[2]
    );

    println!("\nParameters:");
    println!(
        "  Auto Levels:        {} (clip: {:.1}%)",
        result.params.enable_auto_levels, result.params.clip_percent
    );
    println!(
        "  Auto Color:         {} (strength: {:.2}, range: {:.2}-{:.2})",
        result.params.enable_auto_color,
        result.params.auto_color_strength,
        result.params.auto_color_min_gain,
        result.params.auto_color_max_gain
    );
    println!(
        "  Base Sampling:      {:?} (top {:.0}%)",
        result.params.base_sampling_mode, result.params.base_brightest_percent
    );
    println!("  Inversion Mode:     {:?}", result.params.inversion_mode);
    println!(
        "  Shadow Lift:        {:?} (target: {:.3})",
        result.params.shadow_lift_mode, result.params.shadow_lift_value
    );
    println!(
        "  Tone Curve:         {} (strength: {:.2})",
        !result.params.skip_tone_curve, result.params.tone_curve_strength
    );
    println!(
        "  Auto Exposure:      {} (target: {:.2}, strength: {:.2}, range: {:.2}-{:.2})",
        result.params.enable_auto_exposure,
        result.params.auto_exposure_target_median,
        result.params.auto_exposure_strength,
        result.params.auto_exposure_min_gain,
        result.params.auto_exposure_max_gain
    );
    println!(
        "  Exposure:           {:.2}x",
        result.params.exposure_compensation
    );

    println!("\nResults:");
    println!("  Exposure Ratio:     {:.4}x", result.exposure_ratio);
    println!(
        "  Color Shift (RGB):  [{:+.4}, {:+.4}, {:+.4}]",
        result.color_shift[0], result.color_shift[1], result.color_shift[2]
    );
    println!(
        "  Mean Abs Error:     R:{:.4}  G:{:.4}  B:{:.4}",
        result.mae_r, result.mae_g, result.mae_b
    );
    println!("  Contrast Ratio:     {:.4}x", result.contrast_ratio);

    // Performance indicators
    let exposure_ok = (result.exposure_ratio - 1.0).abs() < 0.05;
    let color_ok = result.color_shift.iter().all(|&x| x.abs() < 0.02);
    let mae_ok = (result.mae_r + result.mae_g + result.mae_b) / 3.0 < 0.10;

    println!("\nQuality Indicators:");
    println!(
        "  Exposure Match:     {}",
        if exposure_ok {
            "✓ GOOD"
        } else {
            "⚠ NEEDS WORK"
        }
    );
    println!(
        "  Color Balance:      {}",
        if color_ok {
            "✓ GOOD"
        } else {
            "⚠ NEEDS WORK"
        }
    );
    println!(
        "  Overall Accuracy:   {}",
        if mae_ok { "✓ GOOD" } else { "⚠ NEEDS WORK" }
    );
}

/// Build all parameter combinations from a grid
fn build_parameter_combinations(grid: &ParameterGrid) -> Vec<ParameterTest> {
    let mut combinations = Vec::new();

    for &auto_levels in &grid.auto_levels {
        for &clip_percent in &grid.clip_percent {
            for &auto_color in &grid.auto_color {
                for &auto_color_strength in &grid.auto_color_strength {
                    for &auto_color_min_gain in &grid.auto_color_min_gain {
                        for &auto_color_max_gain in &grid.auto_color_max_gain {
                            for &base_brightest_percent in &grid.base_brightest_percent {
                                for &base_sampling_mode in &grid.base_sampling_mode {
                                    for &inversion_mode in &grid.inversion_mode {
                                        for &shadow_lift_mode in &grid.shadow_lift_mode {
                                            for &shadow_lift_value in &grid.shadow_lift_value {
                                                for &tone_curve_strength in
                                                    &grid.tone_curve_strength
                                                {
                                                    for &exposure_compensation in
                                                        &grid.exposure_compensation
                                                    {
                                                        combinations.push(ParameterTest {
                                                            enable_auto_levels: auto_levels,
                                                            clip_percent,
                                                            enable_auto_color: auto_color,
                                                            auto_color_strength,
                                                            auto_color_min_gain,
                                                            auto_color_max_gain,
                                                            auto_color_max_divergence: grid
                                                                .auto_color_max_divergence
                                                                .first()
                                                                .copied()
                                                                .unwrap_or(0.15),
                                                            base_brightest_percent,
                                                            base_sampling_mode,
                                                            inversion_mode,
                                                            shadow_lift_mode,
                                                            shadow_lift_value,
                                                            highlight_compression: 1.0,
                                                            enable_auto_exposure: true,
                                                            auto_exposure_target_median: 0.25,
                                                            auto_exposure_strength: 1.0,
                                                            auto_exposure_min_gain: 0.6,
                                                            auto_exposure_max_gain: 1.4,
                                                            tone_curve_strength,
                                                            skip_tone_curve: false,
                                                            exposure_compensation,
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    combinations
}

/// Run parameter grid search in parallel using Rayon with preloaded images
/// Much faster than the old version - images are loaded once and shared
pub fn run_parameter_grid_search_parallel<P: AsRef<Path>>(
    original_path: P,
    reference_path: P,
    grid: &ParameterGrid,
    target_score: Option<f32>,
) -> Result<Vec<TestResult>, String> {
    // OPTIMIZATION: Preload images and base estimation ONCE
    let ctx = Arc::new(PreloadedTestContext::new(original_path, reference_path)?);

    // Build all parameter combinations
    let combinations = build_parameter_combinations(grid);

    let total_combinations = combinations.len();
    eprintln!(
        "[PARALLEL GRID SEARCH] Testing {} parameter combinations...",
        total_combinations
    );

    if let Some(target) = target_score {
        eprintln!(
            "[PARALLEL GRID SEARCH] Target score: {:.4} (will stop early if reached)",
            target
        );
    }

    // Shared counter for progress reporting
    let completed = Arc::new(AtomicUsize::new(0));
    let best_score = Arc::new(std::sync::Mutex::new(f32::MAX));

    // Process in parallel with Rayon using preloaded context
    let results: Vec<TestResult> = combinations
        .par_iter()
        .filter_map(|params| {
            // Check if we've hit target score (early termination)
            if let Some(target) = target_score {
                if let Ok(current_best) = best_score.lock() {
                    if *current_best <= target {
                        return None; // Skip remaining tests
                    }
                }
            }

            // Use preloaded context - no more redundant I/O!
            match run_parameter_test_preloaded(&ctx, params) {
                Ok(result) => {
                    // Update best score
                    if let Ok(mut current_best) = best_score.lock() {
                        if result.overall_score < *current_best {
                            *current_best = result.overall_score;
                            eprintln!(
                                "[PARALLEL GRID SEARCH] New best score: {:.4}",
                                result.overall_score
                            );
                        }
                    }

                    // Progress reporting
                    let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(10) || count == total_combinations {
                        eprintln!(
                            "[PARALLEL GRID SEARCH] Progress: {}/{}",
                            count, total_combinations
                        );
                    }

                    Some(result)
                }
                Err(e) => {
                    eprintln!("[PARALLEL GRID SEARCH] Test failed: {}", e);
                    completed.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .collect();

    let final_count = completed.load(Ordering::Relaxed);
    eprintln!(
        "[PARALLEL GRID SEARCH] Complete! Tested {} combinations",
        final_count
    );

    // Sort by overall score (best first)
    let mut sorted_results = results;
    sorted_results.sort_by(|a, b| {
        a.overall_score
            .partial_cmp(&b.overall_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(sorted_results)
}

/// Run parameter grid search using a preloaded context (internal helper)
/// This avoids loading images multiple times in adaptive search
fn run_grid_search_with_context(
    ctx: &Arc<PreloadedTestContext>,
    grid: &ParameterGrid,
    target_score: Option<f32>,
) -> Result<Vec<TestResult>, String> {
    let combinations = build_parameter_combinations(grid);
    let total_combinations = combinations.len();

    eprintln!(
        "[GRID SEARCH] Testing {} parameter combinations...",
        total_combinations
    );

    let completed = Arc::new(AtomicUsize::new(0));
    let best_score = Arc::new(std::sync::Mutex::new(f32::MAX));

    let results: Vec<TestResult> = combinations
        .par_iter()
        .filter_map(|params| {
            if let Some(target) = target_score {
                if let Ok(current_best) = best_score.lock() {
                    if *current_best <= target {
                        return None;
                    }
                }
            }

            match run_parameter_test_preloaded(ctx, params) {
                Ok(result) => {
                    if let Ok(mut current_best) = best_score.lock() {
                        if result.overall_score < *current_best {
                            *current_best = result.overall_score;
                            eprintln!("[GRID SEARCH] New best score: {:.4}", result.overall_score);
                        }
                    }

                    let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(10) || count == total_combinations {
                        eprintln!("[GRID SEARCH] Progress: {}/{}", count, total_combinations);
                    }

                    Some(result)
                }
                Err(e) => {
                    eprintln!("[GRID SEARCH] Test failed: {}", e);
                    completed.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .collect();

    let mut sorted_results = results;
    sorted_results.sort_by(|a, b| {
        a.overall_score
            .partial_cmp(&b.overall_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(sorted_results)
}

/// Adaptive grid search - starts coarse, refines around best results
/// More efficient than exhaustive search for finding optimal parameters
/// OPTIMIZED: Loads images once and reuses across all iterations
pub fn run_adaptive_grid_search<P: AsRef<Path>>(
    original_path: P,
    reference_path: P,
    target_score: f32,
    max_iterations: usize,
) -> Result<Vec<TestResult>, String> {
    eprintln!("[ADAPTIVE SEARCH] Starting adaptive parameter search");
    eprintln!(
        "[ADAPTIVE SEARCH] Target score: {:.4}, Max iterations: {}",
        target_score, max_iterations
    );

    // OPTIMIZATION: Load images ONCE for all iterations
    let ctx = Arc::new(PreloadedTestContext::new(original_path, reference_path)?);

    let mut all_results = Vec::new();
    let mut best_params = ParameterTest::default();
    let mut best_score = f32::MAX;

    // Phase 1: Coarse grid search
    eprintln!("\n[ADAPTIVE SEARCH] Phase 1: Coarse grid search");
    let coarse_grid = ParameterGrid {
        auto_levels: vec![true],
        clip_percent: vec![0.5, 1.0, 2.0],
        auto_color: vec![true, false],
        auto_color_strength: vec![0.6, 0.8],
        auto_color_min_gain: vec![0.7],
        auto_color_max_gain: vec![1.3],
        auto_color_max_divergence: vec![0.15],
        base_brightest_percent: vec![10.0, 20.0],
        base_sampling_mode: vec![BaseSamplingMode::Median],
        inversion_mode: vec![InversionMode::Linear],
        shadow_lift_mode: vec![ShadowLiftMode::Percentile],
        shadow_lift_value: vec![0.02],
        tone_curve_strength: vec![0.4, 0.6, 0.8],
        exposure_compensation: vec![1.0],
    };

    let mut results = run_grid_search_with_context(&ctx, &coarse_grid, Some(target_score))?;

    if let Some(best) = results.first() {
        best_score = best.overall_score;
        best_params = best.params.clone();
        eprintln!(
            "[ADAPTIVE SEARCH] Phase 1 complete. Best score: {:.4}",
            best_score
        );
        all_results.append(&mut results);

        if best_score <= target_score {
            eprintln!("[ADAPTIVE SEARCH] Target score reached in Phase 1!");
            all_results.sort_by(|a, b| {
                a.overall_score
                    .partial_cmp(&b.overall_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(all_results);
        }
    }

    // Phase 2: Refine around best results
    for iteration in 1..=max_iterations {
        eprintln!(
            "\n[ADAPTIVE SEARCH] Phase 2, Iteration {}/{}",
            iteration, max_iterations
        );
        eprintln!(
            "[ADAPTIVE SEARCH] Refining around: clip={:.2}, tone={:.2}",
            best_params.clip_percent, best_params.tone_curve_strength
        );

        // Create refined grid around best parameters
        let refined_grid = ParameterGrid {
            auto_levels: vec![best_params.enable_auto_levels],
            clip_percent: vec![
                (best_params.clip_percent * 0.8).max(0.1),
                best_params.clip_percent,
                (best_params.clip_percent * 1.2).min(10.0),
            ],
            auto_color: vec![best_params.enable_auto_color],
            auto_color_strength: vec![best_params.auto_color_strength],
            auto_color_min_gain: vec![
                (best_params.auto_color_min_gain * 0.9).clamp(0.5, 0.9),
                best_params.auto_color_min_gain,
                (best_params.auto_color_min_gain * 1.1).clamp(0.5, 1.0),
            ],
            auto_color_max_gain: vec![
                (best_params.auto_color_max_gain * 0.9).clamp(1.0, 1.5),
                best_params.auto_color_max_gain,
                (best_params.auto_color_max_gain * 1.1).clamp(1.0, 1.6),
            ],
            auto_color_max_divergence: vec![best_params.auto_color_max_divergence],
            base_brightest_percent: vec![
                (best_params.base_brightest_percent - 2.0).max(5.0),
                best_params.base_brightest_percent,
                (best_params.base_brightest_percent + 2.0).min(30.0),
            ],
            base_sampling_mode: vec![best_params.base_sampling_mode],
            inversion_mode: vec![best_params.inversion_mode],
            shadow_lift_mode: vec![best_params.shadow_lift_mode],
            shadow_lift_value: vec![best_params.shadow_lift_value],
            tone_curve_strength: vec![
                (best_params.tone_curve_strength - 0.1).max(0.0),
                best_params.tone_curve_strength,
                (best_params.tone_curve_strength + 0.1).min(1.0),
            ],
            exposure_compensation: vec![best_params.exposure_compensation],
        };

        let mut iteration_results =
            run_grid_search_with_context(&ctx, &refined_grid, Some(target_score))?;

        if let Some(best) = iteration_results.first() {
            if best.overall_score < best_score {
                let improvement = best_score - best.overall_score;
                best_score = best.overall_score;
                best_params = best.params.clone();
                eprintln!(
                    "[ADAPTIVE SEARCH] Improved! New best: {:.4} (Δ {:.4})",
                    best_score, improvement
                );

                if best_score <= target_score {
                    eprintln!("[ADAPTIVE SEARCH] Target score reached!");
                    all_results.append(&mut iteration_results);
                    all_results.sort_by(|a, b| {
                        a.overall_score
                            .partial_cmp(&b.overall_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    return Ok(all_results);
                }
            } else {
                eprintln!(
                    "[ADAPTIVE SEARCH] No improvement (score: {:.4})",
                    best.overall_score
                );
                // Convergence - no improvement, stop early
                if iteration > 2 {
                    eprintln!("[ADAPTIVE SEARCH] Converged after {} iterations", iteration);
                    break;
                }
            }
        }

        all_results.append(&mut iteration_results);
    }

    eprintln!(
        "\n[ADAPTIVE SEARCH] Complete! Final best score: {:.4}",
        best_score
    );

    // Sort and deduplicate
    all_results.sort_by(|a, b| {
        a.overall_score
            .partial_cmp(&b.overall_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.dedup_by(|a, b| {
        a.params.clip_percent == b.params.clip_percent
            && a.params.tone_curve_strength == b.params.tone_curve_strength
            && a.params.enable_auto_color == b.params.enable_auto_color
    });

    Ok(all_results)
}
