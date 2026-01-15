//! Test runner functions
//!
//! Functions for running individual parameter tests.

use crate::decoders::decode_image;
use crate::diagnostics::compare_conversions;
use crate::models::{ConvertOptions, OutputFormat};
use crate::pipeline::{estimate_base, process_image};
use crate::testing::scoring::{calculate_contrast_ratio, calculate_score};
use crate::testing::types::{ParameterTest, PreloadedTestContext, TestResult};
use std::path::Path;

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
