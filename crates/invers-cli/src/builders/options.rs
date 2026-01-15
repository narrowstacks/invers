//! ConvertOptions builder functions.

use std::path::PathBuf;

/// Build a ConvertOptions struct with all options including GPU control
///
/// This is the canonical function for building ConvertOptions with all
/// pipeline defaults, making it reusable across CLI and GUI applications.
#[allow(clippy::too_many_arguments)]
pub fn build_convert_options_full_with_gpu(
    input: PathBuf,
    output_dir: PathBuf,
    export: &str,
    colorspace: String,
    base_estimation: Option<invers_core::models::BaseEstimation>,
    film_preset: Option<invers_core::models::FilmPreset>,
    scan_profile: Option<invers_core::models::ScanProfile>,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    inversion_mode: Option<invers_core::models::InversionMode>,
    no_auto_levels: bool,
    preserve_headroom: bool,
    no_clip: bool,
    auto_wb: bool,
    auto_wb_strength: f32,
    debug: bool,
    use_gpu: bool,
) -> Result<invers_core::models::ConvertOptions, String> {
    let config_handle = invers_core::config::pipeline_config_handle();
    let defaults = config_handle.config.defaults.clone();

    // Parse output format
    let output_format = match export {
        "tiff16" | "tiff" => invers_core::models::OutputFormat::Tiff16,
        "dng" => invers_core::models::OutputFormat::LinearDng,
        _ => return Err(format!("Unknown export format: {}", export)),
    };

    // Use provided inversion mode, or scan profile preference, or fall back to config default
    let inversion_mode = inversion_mode
        .or_else(|| {
            scan_profile
                .as_ref()
                .and_then(|sp| sp.preferred_inversion_mode)
        })
        .unwrap_or(defaults.inversion_mode);

    // Auto-levels: disabled if --no-auto-levels is set
    let enable_auto_levels = !no_auto_levels && defaults.enable_auto_levels;

    Ok(invers_core::models::ConvertOptions {
        input_paths: vec![input],
        output_dir,
        output_format,
        working_colorspace: colorspace,
        bit_depth_policy: invers_core::models::BitDepthPolicy::Force16Bit,
        film_preset,
        scan_profile,
        base_estimation,
        num_threads: None,
        skip_tone_curve: no_tonecurve || defaults.skip_tone_curve,
        skip_color_matrix: no_colormatrix || defaults.skip_color_matrix,
        exposure_compensation: defaults.exposure_compensation * exposure,
        debug,
        enable_auto_levels,
        auto_levels_clip_percent: defaults.auto_levels_clip_percent,
        preserve_headroom: preserve_headroom || defaults.preserve_headroom,
        enable_auto_color: defaults.enable_auto_color,
        auto_color_strength: defaults.auto_color_strength,
        auto_color_min_gain: defaults.auto_color_min_gain,
        auto_color_max_gain: defaults.auto_color_max_gain,
        auto_color_max_divergence: defaults.auto_color_max_divergence,
        base_brightest_percent: defaults.base_brightest_percent,
        base_sampling_mode: defaults.base_sampling_mode,
        base_estimation_method: invers_core::models::BaseEstimationMethod::default(),
        auto_levels_mode: invers_core::models::AutoLevelsMode::default(),
        inversion_mode,
        shadow_lift_mode: defaults.shadow_lift_mode,
        shadow_lift_value: defaults.shadow_lift_value,
        highlight_compression: defaults.highlight_compression,
        enable_auto_exposure: defaults.enable_auto_exposure,
        auto_exposure_target_median: defaults.auto_exposure_target_median,
        auto_exposure_strength: defaults.auto_exposure_strength,
        auto_exposure_min_gain: defaults.auto_exposure_min_gain,
        auto_exposure_max_gain: defaults.auto_exposure_max_gain,
        no_clip,
        enable_auto_wb: auto_wb,
        auto_wb_strength,
        auto_wb_mode: invers_core::models::AutoWbMode::default(),
        use_gpu,
        // Research pipeline options (defaults for now, CLI args coming soon)
        pipeline_mode: invers_core::models::PipelineMode::Legacy,
        density_balance: None,
        neutral_point: None,
        density_balance_red: None,
        density_balance_blue: None,
        tone_curve_override: None,
        // CB-style pipeline options
        cb_options: None,
    })
}
