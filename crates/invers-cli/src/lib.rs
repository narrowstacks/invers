//! Shared utilities for invers-cli
//!
//! This module provides reusable functions and utilities that can be
//! shared between the CLI and GUI applications.

use std::path::{Path, PathBuf};

/// Parse base RGB values in format "R,G,B"
///
/// # Arguments
/// * `base_str` - A string in format "R,G,B" with values 0.0-1.0
///
/// # Returns
/// An array of [R, G, B] as f32 values
pub fn parse_base_rgb(base_str: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = base_str.split(',').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Base must be in format R,G,B (e.g., 0.48,0.50,0.30), got: {}",
            base_str
        ));
    }

    let r = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("Invalid red value: {}", parts[0]))?;
    let g = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("Invalid green value: {}", parts[1]))?;
    let b = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("Invalid blue value: {}", parts[2]))?;

    // Validate range
    for (val, name) in [(r, "Red"), (g, "Green"), (b, "Blue")] {
        if val <= 0.0 || val > 1.0 {
            return Err(format!(
                "{} value {} must be in range (0.0, 1.0]",
                name, val
            ));
        }
    }

    Ok([r, g, b])
}

/// Parse ROI string in format "x,y,width,height"
///
/// # Arguments
/// * `roi_str` - A string in format "x,y,width,height"
///
/// # Returns
/// A tuple of (x, y, width, height) as u32 values
pub fn parse_roi(roi_str: &str) -> Result<(u32, u32, u32, u32), String> {
    let parts: Vec<&str> = roi_str.split(',').collect();
    if parts.len() != 4 {
        return Err(format!(
            "ROI must be in format x,y,width,height, got: {}",
            roi_str
        ));
    }

    let x = parts[0]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid x coordinate: {}", parts[0]))?;
    let y = parts[1]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid y coordinate: {}", parts[1]))?;
    let width = parts[2]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid width: {}", parts[2]))?;
    let height = parts[3]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid height: {}", parts[3]))?;

    Ok((x, y, width, height))
}

/// Determine output path based on input, output dir, and export format
///
/// # Arguments
/// * `input` - Input file path
/// * `out` - Optional output directory or file path
/// * `export` - Export format ("tiff16" or "dng")
///
/// # Returns
/// The full output path for the converted image
pub fn determine_output_path(
    input: &Path,
    out: &Option<PathBuf>,
    export: &str,
) -> Result<PathBuf, String> {
    let extension = match export {
        "tiff16" | "tiff" => "tif",
        "dng" => "dng",
        _ => "tif",
    };

    if let Some(out_path) = out {
        // If out is a directory, use input filename with new extension
        if out_path.is_dir() {
            let filename = input
                .file_stem()
                .ok_or("Invalid input filename")?
                .to_string_lossy();
            Ok(out_path.join(format!("{}_positive.{}", filename, extension)))
        } else {
            // Use the specified path as-is
            Ok(out_path.clone())
        }
    } else {
        // Use input directory with modified filename
        let filename = input
            .file_stem()
            .ok_or("Invalid input filename")?
            .to_string_lossy();
        let parent = input.parent().unwrap_or(std::path::Path::new("."));
        Ok(parent.join(format!("{}_positive.{}", filename, extension)))
    }
}

/// Parse inversion mode from string
///
/// Supported values:
/// - "mask-aware" / "mask" (default): Orange mask-aware inversion for color negative film
/// - "linear": Simple (base - negative) / base inversion
/// - "log" / "logarithmic": Density-based inversion
/// - "divide-blend" / "divide": Photoshop-style divide blend mode
/// - "bw" / "blackandwhite" / "grayscale": Simple B&W inversion with headroom
pub fn parse_inversion_mode(
    mode_str: Option<&str>,
) -> Result<Option<invers_core::models::InversionMode>, String> {
    match mode_str {
        None => Ok(None), // Use default from config
        Some(s) => match s.to_lowercase().as_str() {
            "mask-aware" | "mask" | "maskaware" => {
                Ok(Some(invers_core::models::InversionMode::MaskAware))
            }
            "linear" => Ok(Some(invers_core::models::InversionMode::Linear)),
            "log" | "logarithmic" => Ok(Some(invers_core::models::InversionMode::Logarithmic)),
            "divide-blend" | "divide" => {
                Ok(Some(invers_core::models::InversionMode::DivideBlend))
            }
            "bw" | "blackandwhite" | "black-and-white" | "grayscale" | "mono" => {
                Ok(Some(invers_core::models::InversionMode::BlackAndWhite))
            }
            _ => Err(format!(
                "Unknown inversion mode: '{}'. Valid options: mask-aware (default), linear, log, divide-blend, bw",
                s
            )),
        },
    }
}

/// Build a ConvertOptions struct from common parameters
///
/// This function centralizes the logic for building ConvertOptions with all
/// pipeline defaults, making it reusable across CLI and GUI applications.
#[allow(clippy::too_many_arguments)]
pub fn build_convert_options(
    input: PathBuf,
    output_dir: PathBuf,
    export: &str,
    colorspace: String,
    base_estimation: Option<invers_core::models::BaseEstimation>,
    film_preset: Option<invers_core::models::FilmPreset>,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    debug: bool,
) -> Result<invers_core::models::ConvertOptions, String> {
    build_convert_options_with_inversion(
        input,
        output_dir,
        export,
        colorspace,
        base_estimation,
        film_preset,
        no_tonecurve,
        no_colormatrix,
        exposure,
        None,
        debug,
    )
}

/// Build a ConvertOptions struct with explicit inversion mode override
#[allow(clippy::too_many_arguments)]
pub fn build_convert_options_with_inversion(
    input: PathBuf,
    output_dir: PathBuf,
    export: &str,
    colorspace: String,
    base_estimation: Option<invers_core::models::BaseEstimation>,
    film_preset: Option<invers_core::models::FilmPreset>,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    inversion_mode: Option<invers_core::models::InversionMode>,
    debug: bool,
) -> Result<invers_core::models::ConvertOptions, String> {
    build_convert_options_full(
        input,
        output_dir,
        export,
        colorspace,
        base_estimation,
        film_preset,
        None, // scan_profile
        no_tonecurve,
        no_colormatrix,
        exposure,
        inversion_mode,
        false, // no_auto_levels
        false, // preserve_headroom
        false, // no_clip
        false, // auto_wb
        debug,
    )
}

/// Build a ConvertOptions struct with all options
#[allow(clippy::too_many_arguments)]
pub fn build_convert_options_full(
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
    debug: bool,
) -> Result<invers_core::models::ConvertOptions, String> {
    build_convert_options_full_with_gpu(
        input,
        output_dir,
        export,
        colorspace,
        base_estimation,
        film_preset,
        scan_profile,
        no_tonecurve,
        no_colormatrix,
        exposure,
        inversion_mode,
        no_auto_levels,
        preserve_headroom,
        no_clip,
        auto_wb,
        1.0, // auto_wb_strength: default full strength
        debug,
        true, // use_gpu: default true (will fallback if unavailable)
    )
}

/// Build a ConvertOptions struct with all options including GPU control
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
        use_gpu,
    })
}
