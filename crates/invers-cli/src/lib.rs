//! Shared utilities for invers-cli
//!
//! This module provides reusable functions and utilities that can be
//! shared between the CLI and GUI applications.

use std::path::{Path, PathBuf};

/// White balance preset for CLI interface.
///
/// This provides a unified, user-friendly interface for white balance settings,
/// consolidating the various auto_wb flags into a single enum.
#[derive(Clone, Debug, Default, clap::ValueEnum)]
pub enum WhiteBalance {
    /// Auto white balance using average/gray world assumption (default)
    #[default]
    Auto,
    /// No white balance adjustment
    None,
    /// Neutral/gray world assumption with full strength
    Neutral,
    /// Warmer tones (reduced blue, slight red boost)
    Warm,
    /// Cooler tones (reduced red, slight blue boost)
    Cool,
}

/// White balance settings derived from the unified WhiteBalance preset.
#[derive(Clone, Debug)]
pub struct WhiteBalanceSettings {
    /// Whether auto white balance is enabled
    pub enabled: bool,
    /// Strength of the white balance correction (0.0-1.0)
    pub strength: f32,
    /// The auto WB mode to use
    pub mode: &'static str,
    /// Color temperature bias (positive = warmer, negative = cooler)
    pub temperature_bias: f32,
}

impl WhiteBalance {
    /// Convert the unified WhiteBalance preset to internal settings.
    ///
    /// Returns settings for:
    /// - `Auto`: Enable auto WB with strength 0.5, average mode
    /// - `None`: Disable auto WB
    /// - `Neutral`: Enable auto WB with "gray" mode for neutral tones
    /// - `Warm`: Enable auto WB with warm bias (reduce blue)
    /// - `Cool`: Enable auto WB with cool bias (reduce red)
    pub fn to_settings(&self) -> WhiteBalanceSettings {
        match self {
            WhiteBalance::Auto => WhiteBalanceSettings {
                enabled: true,
                strength: 0.5,
                mode: "avg",
                temperature_bias: 0.0,
            },
            WhiteBalance::None => WhiteBalanceSettings {
                enabled: false,
                strength: 0.0,
                mode: "avg",
                temperature_bias: 0.0,
            },
            WhiteBalance::Neutral => WhiteBalanceSettings {
                enabled: true,
                strength: 1.0,
                mode: "gray",
                temperature_bias: 0.0,
            },
            WhiteBalance::Warm => WhiteBalanceSettings {
                enabled: true,
                strength: 0.5,
                mode: "avg",
                temperature_bias: 500.0, // Positive = warmer (shift toward yellow/red)
            },
            WhiteBalance::Cool => WhiteBalanceSettings {
                enabled: true,
                strength: 0.5,
                mode: "avg",
                temperature_bias: -500.0, // Negative = cooler (shift toward blue)
            },
        }
    }
}

use invers_core::decoders::DecodedImage;
use invers_core::models::{
    AutoWbMode, BaseEstimation, FilmPreset, InversionMode, MaskProfile, PipelineMode, ScanProfile,
    ToneCurveParams,
};

/// Parameters for processing a single image.
/// Used by both convert and batch commands to avoid duplication.
#[derive(Clone)]
pub struct ProcessingParams {
    // Basic options
    pub export: String,
    pub exposure: f32,
    pub cpu_only: bool,
    pub silent: bool,
    pub verbose: bool,
    pub debug: bool,

    // White balance (user-facing unified interface)
    pub white_balance: WhiteBalance,

    // Pipeline options
    pub pipeline: String,
    pub db_red: Option<f32>,
    pub db_blue: Option<f32>,
    pub neutral_roi: Option<String>,

    // CB options
    pub cb_tone: Option<String>,
    pub cb_lut: Option<String>,
    pub cb_color: Option<String>,
    pub cb_film: Option<String>,
    pub cb_wb: Option<String>,

    // Debug options (always present, release builds use defaults)
    pub no_tonecurve: bool,
    pub no_colormatrix: bool,
    pub inversion: Option<String>,
    pub auto_wb: bool,
    pub auto_wb_strength: f32,
    pub auto_wb_mode: String,
    pub tone_curve: Option<String>,
}

impl Default for ProcessingParams {
    fn default() -> Self {
        Self {
            export: "tiff16".to_string(),
            exposure: 1.0,
            cpu_only: false,
            silent: false,
            verbose: false,
            debug: false,
            white_balance: WhiteBalance::Auto,
            pipeline: "legacy".to_string(),
            db_red: None,
            db_blue: None,
            neutral_roi: None,
            cb_tone: None,
            cb_lut: None,
            cb_color: None,
            cb_film: None,
            cb_wb: None,
            no_tonecurve: false,
            no_colormatrix: false,
            inversion: None,
            auto_wb: true,
            auto_wb_strength: 1.0,
            auto_wb_mode: "avg".to_string(),
            tone_curve: None,
        }
    }
}

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
    let effective_wb_mode = if params.auto_wb_mode != "avg" && params.auto_wb_mode != wb_settings.mode {
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

/// Parse pipeline mode from string
///
/// Supported values:
/// - "legacy" (default): Original Invers pipeline
/// - "research": Research-based pipeline with density balance before inversion
/// - "cb": CB-style curve-based pipeline
pub fn parse_pipeline_mode(mode_str: &str) -> Result<invers_core::models::PipelineMode, String> {
    match mode_str.to_lowercase().as_str() {
        "legacy" | "default" | "" => Ok(invers_core::models::PipelineMode::Legacy),
        "research" | "new" | "density" => Ok(invers_core::models::PipelineMode::Research),
        "cb" | "cbstyle" | "cb-style" => Ok(invers_core::models::PipelineMode::CbStyle),
        _ => Err(format!(
            "Unknown pipeline mode: '{}'. Valid options: legacy (default), research, cb",
            mode_str
        )),
    }
}

/// Parse CB tone profile from string
pub fn parse_cb_tone_profile(
    profile_str: Option<&str>,
) -> Result<invers_core::models::CbToneProfile, String> {
    match profile_str {
        None => Ok(invers_core::models::CbToneProfile::default()),
        Some(s) => match s.to_lowercase().replace('-', "_").as_str() {
            "standard" | "std" | "" => Ok(invers_core::models::CbToneProfile::Standard),
            "linear" => Ok(invers_core::models::CbToneProfile::Linear),
            "linear_gamma" | "lineargamma" => Ok(invers_core::models::CbToneProfile::LinearGamma),
            "linear_flat" | "linearflat" => Ok(invers_core::models::CbToneProfile::LinearFlat),
            "linear_deep" | "lineardeep" => Ok(invers_core::models::CbToneProfile::LinearDeep),
            "logarithmic" | "log" => Ok(invers_core::models::CbToneProfile::Logarithmic),
            "logarithmic_rich" | "log_rich" | "logrich" => {
                Ok(invers_core::models::CbToneProfile::LogarithmicRich)
            }
            "logarithmic_flat" | "log_flat" | "logflat" => {
                Ok(invers_core::models::CbToneProfile::LogarithmicFlat)
            }
            "all_soft" | "allsoft" | "soft" => Ok(invers_core::models::CbToneProfile::AllSoft),
            "all_hard" | "allhard" | "hard" => Ok(invers_core::models::CbToneProfile::AllHard),
            "highlight_hard" | "highlighthard" => {
                Ok(invers_core::models::CbToneProfile::HighlightHard)
            }
            "highlight_soft" | "highlightsoft" => {
                Ok(invers_core::models::CbToneProfile::HighlightSoft)
            }
            "shadow_hard" | "shadowhard" => Ok(invers_core::models::CbToneProfile::ShadowHard),
            "shadow_soft" | "shadowsoft" => Ok(invers_core::models::CbToneProfile::ShadowSoft),
            "autotone" | "auto" | "auto_tone" => Ok(invers_core::models::CbToneProfile::AutoTone),
            _ => Err(format!(
                "Unknown CB tone profile: '{}'. Valid: standard, linear, linear-gamma, \
                 linear-flat, linear-deep, log, log-rich, log-flat, soft, hard, \
                 highlight-hard, highlight-soft, shadow-hard, shadow-soft, auto",
                s
            )),
        },
    }
}

/// Parse CB enhanced profile (LUT) from string
pub fn parse_cb_enhanced_profile(
    profile_str: Option<&str>,
) -> Result<invers_core::models::CbEnhancedProfile, String> {
    match profile_str {
        None => Ok(invers_core::models::CbEnhancedProfile::default()),
        Some(s) => match s.to_lowercase().as_str() {
            "none" | "" => Ok(invers_core::models::CbEnhancedProfile::None),
            "natural" => Ok(invers_core::models::CbEnhancedProfile::Natural),
            "frontier" => Ok(invers_core::models::CbEnhancedProfile::Frontier),
            "crystal" => Ok(invers_core::models::CbEnhancedProfile::Crystal),
            "pakon" => Ok(invers_core::models::CbEnhancedProfile::Pakon),
            _ => Err(format!(
                "Unknown CB enhanced profile: '{}'. Valid: none, natural, frontier, crystal, pakon",
                s
            )),
        },
    }
}

/// Parse CB color model from string
pub fn parse_cb_color_model(
    model_str: Option<&str>,
) -> Result<invers_core::models::CbColorModel, String> {
    match model_str {
        None => Ok(invers_core::models::CbColorModel::default()),
        Some(s) => match s.to_lowercase().as_str() {
            "none" => Ok(invers_core::models::CbColorModel::None),
            "basic" | "" => Ok(invers_core::models::CbColorModel::Basic),
            "frontier" => Ok(invers_core::models::CbColorModel::Frontier),
            "noritsu" => Ok(invers_core::models::CbColorModel::Noritsu),
            "bw" | "blackandwhite" | "black_and_white" | "mono" => {
                Ok(invers_core::models::CbColorModel::BlackAndWhite)
            }
            _ => Err(format!(
                "Unknown CB color model: '{}'. Valid: none, basic, frontier, noritsu, bw",
                s
            )),
        },
    }
}

/// Parse CB film character from string
pub fn parse_cb_film_character(
    character_str: Option<&str>,
) -> Result<invers_core::models::CbFilmCharacter, String> {
    match character_str {
        None => Ok(invers_core::models::CbFilmCharacter::default()),
        Some(s) => match s.to_lowercase().replace('-', "_").as_str() {
            "none" => Ok(invers_core::models::CbFilmCharacter::None),
            "generic" | "genericcolor" | "generic_color" => {
                Ok(invers_core::models::CbFilmCharacter::GenericColor)
            }
            "kodak" | "portra" | "" => Ok(invers_core::models::CbFilmCharacter::Kodak),
            "fuji" | "fujifilm" => Ok(invers_core::models::CbFilmCharacter::Fuji),
            "cinestill_50d" | "cinestill50d" | "50d" => {
                Ok(invers_core::models::CbFilmCharacter::Cinestill50D)
            }
            "cinestill_800t" | "cinestill800t" | "800t" => {
                Ok(invers_core::models::CbFilmCharacter::Cinestill800T)
            }
            _ => Err(format!(
                "Unknown CB film character: '{}'. Valid: none, generic, kodak, fuji, \
                 cinestill-50d, cinestill-800t",
                s
            )),
        },
    }
}

/// Parse CB white balance preset from string
pub fn parse_cb_wb_preset(wb_str: Option<&str>) -> Result<invers_core::models::CbWbPreset, String> {
    match wb_str {
        None => Ok(invers_core::models::CbWbPreset::default()),
        Some(s) => match s.to_lowercase().replace('-', "_").as_str() {
            "none" => Ok(invers_core::models::CbWbPreset::None),
            "auto" | "autocolor" | "auto_color" | "avg" | "auto_avg" | "" => {
                Ok(invers_core::models::CbWbPreset::AutoColor)
            }
            "neutral" | "autoneutral" | "auto_neutral" => {
                Ok(invers_core::models::CbWbPreset::AutoNeutral)
            }
            "warm" | "autowarm" | "auto_warm" => Ok(invers_core::models::CbWbPreset::AutoWarm),
            "cool" | "autocool" | "auto_cool" => Ok(invers_core::models::CbWbPreset::AutoCool),
            "mix" | "automix" | "auto_mix" => Ok(invers_core::models::CbWbPreset::AutoMix),
            "standard" | "std" => Ok(invers_core::models::CbWbPreset::Standard),
            "kodak" => Ok(invers_core::models::CbWbPreset::Kodak),
            "fuji" | "fujifilm" => Ok(invers_core::models::CbWbPreset::Fuji),
            "cine_t" | "cinet" | "cinestill_t" | "tungsten" => {
                Ok(invers_core::models::CbWbPreset::CineT)
            }
            "cine_d" | "cined" | "cinestill_d" | "daylight" => {
                Ok(invers_core::models::CbWbPreset::CineD)
            }
            "custom" => Ok(invers_core::models::CbWbPreset::Custom),
            _ => Err(format!(
                "Unknown CB WB preset: '{}'. Valid: none, auto (default), neutral, warm, cool, \
                 mix, standard, kodak, fuji, cine-t, cine-d, custom",
                s
            )),
        },
    }
}

/// Build CbOptions from parsed command line arguments
pub fn build_cb_options(
    tone_profile: Option<&str>,
    enhanced_profile: Option<&str>,
    color_model: Option<&str>,
    film_character: Option<&str>,
    wb_preset: Option<&str>,
) -> Result<invers_core::models::CbOptions, String> {
    let tone = parse_cb_tone_profile(tone_profile)?;
    let lut = parse_cb_enhanced_profile(enhanced_profile)?;
    let color = parse_cb_color_model(color_model)?;
    let film = parse_cb_film_character(film_character)?;
    let wb = parse_cb_wb_preset(wb_preset)?;

    let mut opts = invers_core::models::CbOptions::from_presets(
        tone,
        lut,
        color,
        film,
        invers_core::models::CbEnginePreset::V3_1,
    );
    opts.wb_preset = wb;
    Ok(opts)
}

/// Parse neutral point ROI for density balance calculation
///
/// # Arguments
/// * `roi_str` - Optional ROI string in format "x,y,width,height"
///
/// # Returns
/// A NeutralPointSample with the parsed ROI, or None if not provided
pub fn parse_neutral_roi(
    roi_str: &Option<String>,
) -> Result<Option<invers_core::models::NeutralPointSample>, String> {
    match roi_str {
        Some(s) if !s.is_empty() => {
            let roi = parse_roi(s)?;
            Ok(Some(invers_core::models::NeutralPointSample {
                roi: Some(roi),
                neutral_rgb: [0.0, 0.0, 0.0], // Will be sampled during processing
                auto_detected: false,
            }))
        }
        _ => Ok(None),
    }
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

/// Supported image extensions for batch processing
const SUPPORTED_EXTENSIONS: &[&str] = &["tif", "tiff", "png", "dng"];

/// Expand a list of inputs (files and directories) into a list of image files.
///
/// Directories are scanned for supported image files (.tif, .tiff, .png, .dng).
/// If `recursive` is true, subdirectories are also scanned.
pub fn expand_inputs(inputs: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    for input in inputs {
        if input.is_dir() {
            collect_images_from_dir(input, recursive, &mut files)?;
        } else if input.is_file() {
            files.push(input.clone());
        } else {
            return Err(format!("Path not found: {}", input.display()));
        }
    }

    // Sort for consistent ordering
    files.sort();
    Ok(files)
}

/// Recursively collect image files from a directory.
fn collect_images_from_dir(
    dir: &Path,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Error reading directory entry: {}", e))?;
        let path = entry.path();

        if path.is_dir() && recursive {
            collect_images_from_dir(&path, recursive, files)?;
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}
