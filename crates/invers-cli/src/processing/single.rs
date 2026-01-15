//! Single image processing functions.

use std::path::{Path, PathBuf};

use invers_core::decoders::DecodedImage;
use invers_core::models::{
    AutoWbMode, BaseEstimation, FilmPreset, InversionMode, MaskProfile, PipelineMode, ScanProfile,
    ToneCurveParams,
};

use crate::builders::{build_cb_options, build_convert_options_full_with_gpu};
use crate::parsers::{parse_inversion_mode, parse_neutral_roi, parse_pipeline_mode};
use crate::types::ProcessingParams;

/// Create a BaseEstimation from manual RGB values.
pub fn make_base_from_rgb(rgb: [f32; 3]) -> BaseEstimation {
    let mask_profile = MaskProfile::from_base_medians(&rgb);
    BaseEstimation {
        roi: None,
        medians: rgb,
        noise_stats: None,
        auto_estimated: false,
        mask_profile: Some(mask_profile),
    }
}

/// Process a single decoded image with full option handling.
///
/// This encapsulates:
/// - B&W auto-detection
/// - ConvertOptions building
/// - Pipeline mode application
/// - CB options application
/// - Debug output
/// - Processing and exporting
#[allow(clippy::too_many_arguments)]
pub fn process_single_image(
    decoded: DecodedImage,
    base_estimation: BaseEstimation,
    film_preset: Option<FilmPreset>,
    scan_profile: Option<ScanProfile>,
    output_path: &Path,
    params: &ProcessingParams,
) -> Result<PathBuf, String> {
    // Parse inversion mode if specified
    let inversion_mode = parse_inversion_mode(params.inversion.as_deref())?;

    // Auto-detect B&W mode for grayscale/monochrome images (unless user specified a mode)
    let effective_inversion_mode =
        if inversion_mode.is_none() && (decoded.source_is_grayscale || decoded.is_monochrome) {
            if !params.silent {
                println!("  Detected B&W image, using BlackAndWhite inversion mode");
            }
            Some(InversionMode::BlackAndWhite)
        } else {
            inversion_mode
        };

    // GPU is enabled by default, --cpu forces CPU-only processing
    let use_gpu = !params.cpu_only;

    let output_dir = output_path.parent().unwrap_or(Path::new(".")).to_path_buf();

    // Derive white balance settings from the unified WhiteBalance preset
    // Debug args can still override via auto_wb/auto_wb_strength/auto_wb_mode
    let wb_settings = params.white_balance.to_settings();
    let effective_auto_wb = if params.auto_wb != wb_settings.enabled {
        // Debug arg explicitly overrode white balance
        params.auto_wb
    } else {
        wb_settings.enabled
    };
    let effective_wb_strength = if (params.auto_wb_strength - wb_settings.strength).abs() > 0.001 {
        // Debug arg explicitly overrode strength
        params.auto_wb_strength
    } else {
        wb_settings.strength
    };

    // Build conversion options using shared utility
    let mut options = build_convert_options_full_with_gpu(
        output_path.to_path_buf(), // input path (used for metadata, not critical here)
        output_dir,
        &params.export,
        "linear-rec2020".to_string(),
        Some(base_estimation),
        film_preset,
        scan_profile,
        params.no_tonecurve,
        params.no_colormatrix,
        params.exposure,
        effective_inversion_mode,
        false, // no_auto_levels
        false, // preserve_headroom
        false, // no_clip
        effective_auto_wb,
        effective_wb_strength,
        params.debug,
        use_gpu,
    )?;

    // Apply research pipeline options
    options.pipeline_mode = parse_pipeline_mode(&params.pipeline)?;
    options.density_balance_red = params.db_red;
    options.density_balance_blue = params.db_blue;
    options.neutral_point = parse_neutral_roi(&params.neutral_roi)?;

    // Apply tone curve override if specified
    if let Some(tc_type) = &params.tone_curve {
        options.tone_curve_override = Some(ToneCurveParams {
            curve_type: tc_type.clone(),
            strength: 0.25,
            ..Default::default()
        });
    }

    // Apply auto-WB mode from white balance preset or debug override
    let effective_wb_mode =
        if params.auto_wb_mode != "avg" && params.auto_wb_mode != wb_settings.mode {
            // Debug arg explicitly overrode mode
            &params.auto_wb_mode
        } else {
            wb_settings.mode
        };
    options.auto_wb_mode = match effective_wb_mode.to_lowercase().as_str() {
        "avg" | "average" | "grayworld" => AutoWbMode::Average,
        "pct" | "percentile" | "whitepatch" => AutoWbMode::Percentile,
        _ => AutoWbMode::GrayPixel,
    };

    // Print research pipeline info if using research mode
    if !params.silent && options.pipeline_mode == PipelineMode::Research {
        println!("Using RESEARCH pipeline (density balance before inversion)");
        if let Some(red) = params.db_red {
            println!("  Density balance red: {:.3}", red);
        }
        if let Some(blue) = params.db_blue {
            println!("  Density balance blue: {:.3}", blue);
        }
        if options.neutral_point.is_some() {
            println!("  Neutral ROI: specified");
        }
    }

    // Apply CB pipeline options
    if options.pipeline_mode == PipelineMode::CbStyle {
        let cb_opts = build_cb_options(
            params.cb_tone.as_deref(),
            params.cb_lut.as_deref(),
            params.cb_color.as_deref(),
            params.cb_film.as_deref(),
            params.cb_wb.as_deref(),
        )?;
        if !params.silent {
            println!("Using CB pipeline (curve-based processing)");
            println!("  Tone profile: {:?}", cb_opts.tone_profile);
            println!("  Enhanced profile: {:?}", cb_opts.enhanced_profile);
            println!("  Color model: {:?}", cb_opts.color_model);
            println!("  Film character: {:?}", cb_opts.film_character);
            println!("  WB preset: {:?}", cb_opts.wb_preset);
        }
        options.cb_options = Some(cb_opts);
    }

    // Debug output: show pipeline configuration
    if params.debug {
        eprintln!("\n[debug] Pipeline configuration:");
        eprintln!("  Inversion mode: {:?}", options.inversion_mode);
        eprintln!(
            "  Auto levels: {} (clip: {:.2}%)",
            options.enable_auto_levels, options.auto_levels_clip_percent
        );
        eprintln!("  Preserve headroom: {}", options.preserve_headroom);
        eprintln!(
            "  Auto color: {} (strength: {:.2})",
            options.enable_auto_color, options.auto_color_strength
        );
        eprintln!(
            "  Auto color gain: [{:.2}, {:.2}]",
            options.auto_color_min_gain, options.auto_color_max_gain
        );
        eprintln!(
            "  Auto WB: {} (strength: {:.2})",
            options.enable_auto_wb, options.auto_wb_strength
        );
        eprintln!(
            "  Auto exposure: {} (target: {:.2}, strength: {:.2})",
            options.enable_auto_exposure,
            options.auto_exposure_target_median,
            options.auto_exposure_strength
        );
        eprintln!(
            "  Exposure compensation: {:.2}",
            options.exposure_compensation
        );
        eprintln!(
            "  Shadow lift: {:?} (value: {:.3})",
            options.shadow_lift_mode, options.shadow_lift_value
        );
        eprintln!(
            "  Highlight compression: {:.2}",
            options.highlight_compression
        );
        eprintln!("  Skip tone curve: {}", options.skip_tone_curve);
        eprintln!("  Skip color matrix: {}", options.skip_color_matrix);
        eprintln!(
            "  Base sampling: {:?} (brightest: {:.1}%)",
            options.base_sampling_mode, options.base_brightest_percent
        );
        eprintln!("  GPU processing: {}", options.use_gpu);
        eprintln!();
    }

    // Check if we're using B&W mode (auto-detected or user-specified)
    let using_bw_mode = effective_inversion_mode == Some(InversionMode::BlackAndWhite);

    // For B&W images/mode, disable color-specific operations
    if decoded.source_is_grayscale || decoded.is_monochrome || using_bw_mode {
        options.enable_auto_color = false;
        options.enable_auto_wb = false;
        options.skip_color_matrix = true;
    }

    // Process image
    if !params.silent {
        println!("Processing image...");
    }
    let mut processed = invers_core::pipeline::process_image(decoded, &options)?;

    // Force grayscale export if B&W mode was used
    if using_bw_mode {
        processed.export_as_grayscale = true;
    }

    // Export
    if !params.silent {
        let format_str = if options.output_format == invers_core::models::OutputFormat::Tiff16 {
            if processed.export_as_grayscale {
                "TIFF16 (grayscale)"
            } else {
                "TIFF16"
            }
        } else {
            "DNG"
        };
        println!("Exporting to {}...", format_str);
    }

    match options.output_format {
        invers_core::models::OutputFormat::Tiff16 => {
            invers_core::exporters::export_tiff16(&processed, output_path, None)?;
        }
        invers_core::models::OutputFormat::LinearDng => {
            return Err("Linear DNG export not yet implemented".to_string());
        }
    }

    Ok(output_path.to_path_buf())
}
