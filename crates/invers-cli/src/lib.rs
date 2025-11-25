//! Shared utilities for invers-cli
//!
//! This module provides reusable functions and utilities that can be
//! shared between the CLI and GUI applications.

use std::path::{Path, PathBuf};

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

/// Build a ConvertOptions struct from common parameters
///
/// This function centralizes the logic for building ConvertOptions with all
/// pipeline defaults, making it reusable across CLI and GUI applications.
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
    let config_handle = invers_core::config::pipeline_config_handle();
    let defaults = config_handle.config.defaults.clone();

    // Parse output format
    let output_format = match export {
        "tiff16" | "tiff" => invers_core::models::OutputFormat::Tiff16,
        "dng" => invers_core::models::OutputFormat::LinearDng,
        _ => return Err(format!("Unknown export format: {}", export)),
    };

    Ok(invers_core::models::ConvertOptions {
        input_paths: vec![input],
        output_dir,
        output_format,
        working_colorspace: colorspace,
        bit_depth_policy: invers_core::models::BitDepthPolicy::Force16Bit,
        film_preset,
        scan_profile: None,
        base_estimation,
        num_threads: None,
        skip_tone_curve: no_tonecurve || defaults.skip_tone_curve,
        skip_color_matrix: no_colormatrix || defaults.skip_color_matrix,
        exposure_compensation: defaults.exposure_compensation * exposure,
        debug,
        enable_auto_levels: defaults.enable_auto_levels,
        auto_levels_clip_percent: defaults.auto_levels_clip_percent,
        enable_auto_color: defaults.enable_auto_color,
        auto_color_strength: defaults.auto_color_strength,
        auto_color_min_gain: defaults.auto_color_min_gain,
        auto_color_max_gain: defaults.auto_color_max_gain,
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
    })
}
