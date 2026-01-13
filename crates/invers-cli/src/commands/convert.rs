use std::path::PathBuf;
use std::time::Instant;

use invers_cli::{determine_output_path, parse_base_rgb, process_single_image, ProcessingParams};

/// Execute the convert command for a single image.
///
/// Converts a film negative image to a positive, applying the full processing
/// pipeline including base estimation, inversion, tone curves, and color correction.
///
/// The function handles:
/// - Image decoding (TIFF, PNG, RAW formats)
/// - Automatic or manual base estimation
/// - B&W auto-detection (switches to BlackAndWhite inversion mode)
/// - Pipeline mode selection (Legacy, Research, or CB)
/// - Film preset and scan profile application
/// - Export to TIFF16 or Linear DNG
///
/// # Returns
/// Returns `Ok(())` on success, or an error message describing the failure.
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)] // Some params only used in debug builds
pub fn cmd_convert(
    input: PathBuf,
    out: Option<PathBuf>,
    preset: Option<PathBuf>,
    scan_profile_path: Option<PathBuf>,
    export: String,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    inversion: Option<String>,
    base: Option<String>,
    silent: bool,
    cpu_only: bool,
    auto_wb: bool,
    auto_wb_strength: f32,
    auto_wb_mode: String,
    tone_curve: Option<String>,
    verbose: bool,
    debug: bool,
    // Research pipeline options
    pipeline: String,
    db_red: Option<f32>,
    db_blue: Option<f32>,
    neutral_roi: Option<String>,
    // CB pipeline options
    cb_tone: Option<String>,
    cb_lut: Option<String>,
    cb_color: Option<String>,
    cb_film: Option<String>,
    cb_wb: Option<String>,
) -> Result<(), String> {
    let start_time = Instant::now();

    // Set verbose mode for core library and log config source
    if verbose || debug {
        invers_core::config::set_verbose(true);
        invers_core::config::log_config_usage();

        // Display loaded config settings
        let handle = invers_core::config::pipeline_config_handle();
        let defaults = &handle.config.defaults;
        println!("Pipeline config:");
        println!("  inversion_mode: {:?}", defaults.inversion_mode);
        println!("  enable_auto_levels: {}", defaults.enable_auto_levels);
        println!("  preserve_headroom: {}", defaults.preserve_headroom);
        println!(
            "  exposure_compensation: {}",
            defaults.exposure_compensation
        );
        println!("  enable_auto_color: {}", defaults.enable_auto_color);
        println!("  skip_tone_curve: {}", defaults.skip_tone_curve);
        println!("  enable_auto_exposure: {}", defaults.enable_auto_exposure);
        #[cfg(debug_assertions)]
        {
            println!(
                "  auto_levels_clip_percent: {}",
                defaults.auto_levels_clip_percent
            );
            println!("  auto_color_strength: {}", defaults.auto_color_strength);
            println!(
                "  highlight_compression: {}",
                defaults.highlight_compression
            );
            println!(
                "  auto_exposure_strength: {}",
                defaults.auto_exposure_strength
            );
        }
        println!();
    }

    if !silent {
        println!("Converting {} to positive...", input.display());
    }

    // Decode input image
    if !silent {
        println!("Decoding image...");
    }
    let decoded = invers_core::decoders::decode_image(&input)?;
    if !silent {
        let color_mode = if decoded.source_is_grayscale {
            "grayscale"
        } else if decoded.is_monochrome {
            "RGB (monochrome)"
        } else {
            "RGB"
        };
        println!(
            "  Image: {}x{}, {} channels ({})",
            decoded.width, decoded.height, decoded.channels, color_mode
        );
    }

    // Load scan profile if provided
    let scan_profile = if let Some(profile_path) = scan_profile_path {
        if !silent {
            println!("Loading scan profile from {}...", profile_path.display());
        }
        let profile = invers_core::presets::load_scan_profile(&profile_path)?;
        if !silent {
            println!("  Profile: {} ({})", profile.name, profile.source_type);
            if let Some(ref hsl) = profile.hsl_adjustments {
                if hsl.has_adjustments() {
                    println!("  HSL adjustments: enabled");
                }
            }
            if let Some(gamma) = profile.default_gamma {
                println!(
                    "  Default gamma: [{:.2}, {:.2}, {:.2}]",
                    gamma[0], gamma[1], gamma[2]
                );
            }
        }
        Some(profile)
    } else {
        None
    };

    // Parse manual base values or auto-estimate using default method
    let base_estimation = if let Some(base_str) = base {
        // Parse manual base values (R,G,B format)
        let base_rgb = parse_base_rgb(&base_str)?;
        if !silent {
            println!("Using manual base values...");
            println!(
                "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
                base_rgb[0], base_rgb[1], base_rgb[2]
            );
        }
        // Auto-detect mask profile from manual base values
        let mask_profile = invers_core::models::MaskProfile::from_base_medians(&base_rgb);
        invers_core::models::BaseEstimation {
            roi: None,
            medians: base_rgb,
            noise_stats: None,
            auto_estimated: false,
            mask_profile: Some(mask_profile),
        }
    } else {
        // Auto-estimate base using default regions method
        if !silent {
            println!("Estimating film base...");
        }
        let estimation = invers_core::pipeline::estimate_base(
            &decoded,
            None, // No ROI - use auto-detection
            Some(invers_core::models::BaseEstimationMethod::Regions),
            None, // Default border percent
        )?;
        if !silent {
            println!(
                "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
                estimation.medians[0], estimation.medians[1], estimation.medians[2]
            );
        }
        estimation
    };

    // Load film preset if provided
    let film_preset = if let Some(preset_path) = preset {
        if !silent {
            println!("Loading film preset from {}...", preset_path.display());
        }
        Some(invers_core::presets::load_film_preset(&preset_path)?)
    } else {
        None
    };

    // Prepare output path
    let output_path = determine_output_path(&input, &out, &export)?;
    if !silent {
        println!("Output: {}", output_path.display());
    }

    // Build processing params
    let params = ProcessingParams {
        export: export.clone(),
        exposure,
        cpu_only,
        silent,
        verbose,
        debug,
        pipeline: pipeline.clone(),
        db_red,
        db_blue,
        neutral_roi: neutral_roi.clone(),
        cb_tone: cb_tone.clone(),
        cb_lut: cb_lut.clone(),
        cb_color: cb_color.clone(),
        cb_film: cb_film.clone(),
        cb_wb: cb_wb.clone(),
        no_tonecurve,
        no_colormatrix,
        inversion: inversion.clone(),
        auto_wb,
        auto_wb_strength,
        auto_wb_mode: auto_wb_mode.clone(),
        tone_curve: tone_curve.clone(),
    };

    // Process using shared function
    process_single_image(
        decoded,
        base_estimation,
        film_preset,
        scan_profile,
        &output_path,
        &params,
    )?;

    let elapsed = start_time.elapsed();
    if !silent {
        println!(
            "Done! Positive image saved to: {} ({:.2}s)",
            output_path.display(),
            elapsed.as_secs_f64()
        );
    } else {
        println!("{}", output_path.display());
    }
    Ok(())
}
