use clap::{Parser, Subcommand};
use invers_cli::{
    build_convert_options, build_convert_options_full_with_gpu, determine_output_path,
    parse_base_rgb, parse_inversion_mode, parse_roi,
};
use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "invers")]
#[command(version, about = "Film negative to positive converter", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert negative image(s) to positive
    Convert {
        /// Input file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output directory or file path
        #[arg(short, long, value_name = "PATH")]
        out: Option<PathBuf>,

        /// Film preset file
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// Scan profile file
        #[arg(short, long, value_name = "FILE")]
        scan_profile: Option<PathBuf>,

        /// Export format (tiff16 or dng)
        #[arg(long, value_name = "FORMAT", default_value = "tiff16")]
        export: String,

        /// Skip tone curve application
        #[arg(long)]
        no_tonecurve: bool,

        /// Skip color matrix correction
        #[arg(long)]
        no_colormatrix: bool,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        exposure: f32,

        /// Inversion mode: "mask-aware" (default), "linear", "log", or "divide-blend"
        #[arg(long, value_name = "MODE")]
        inversion: Option<String>,

        /// Manual base RGB values (comma-separated: R,G,B)
        /// Use 'invers analyze' to determine these values, then reuse across a roll
        #[arg(long, value_name = "R,G,B")]
        base: Option<String>,

        /// Suppress non-essential output (timing, progress messages)
        #[arg(long)]
        silent: bool,

        /// Force CPU-only processing (GPU is used by default when available)
        #[arg(long)]
        cpu: bool,

        /// Enable automatic white balance correction
        #[arg(long)]
        auto_wb: bool,

        /// Strength of auto white balance correction (0.0-1.0, default 1.0)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        auto_wb_strength: f32,

        /// Enable verbose output (config loading, processing details)
        #[arg(short, long)]
        verbose: bool,

        /// Enable debug output (detailed pipeline parameters)
        #[arg(long)]
        debug: bool,
    },

    /// Analyze image and estimate film base color
    ///
    /// Use this command to inspect an image and determine base RGB values
    /// that can be reused across multiple frames from the same roll of film.
    Analyze {
        /// Input file to analyze
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// ROI for base estimation (x,y,width,height)
        #[arg(long, value_name = "X,Y,W,H")]
        roi: Option<String>,

        /// Base estimation method: "regions" (default) or "border"
        #[arg(long, value_name = "METHOD", default_value = "regions")]
        base_method: String,

        /// Border percentage for "border" base method (1-25%, default: 5)
        #[arg(long, value_name = "PERCENT", default_value = "5.0")]
        border_percent: f32,

        /// Output as JSON (machine-readable)
        #[arg(long)]
        json: bool,

        /// Save analysis to file
        #[arg(short, long, value_name = "FILE")]
        save: Option<PathBuf>,

        /// Show detailed analysis output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Batch process multiple files with shared settings
    Batch {
        /// Input files or pattern
        #[arg(value_name = "INPUTS")]
        inputs: Vec<PathBuf>,

        /// Base estimation file
        #[arg(long, value_name = "FILE")]
        base_from: Option<PathBuf>,

        /// Film preset file
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// Export format (tiff16 or dng)
        #[arg(long, value_name = "FORMAT", default_value = "tiff16")]
        export: String,

        /// Output directory
        #[arg(short, long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Number of parallel threads
        #[arg(short = 'j', long, value_name = "N")]
        threads: Option<usize>,

        /// Suppress non-essential output (timing, progress messages)
        #[arg(long)]
        silent: bool,

        /// Enable verbose output (base estimation details, config loading)
        #[arg(short, long)]
        verbose: bool,

        /// Force CPU-only processing (GPU is used by default when available)
        #[arg(long)]
        cpu: bool,
    },

    /// Manage film presets
    Preset {
        #[command(subcommand)]
        action: PresetAction,
    },

    /// Initialize user configuration directory with default presets
    ///
    /// Copies default configuration and preset files to ~/invers/.
    /// Safe to run multiple times - won't overwrite existing files.
    Init {
        /// Force overwrite of existing files
        #[arg(long)]
        force: bool,
    },

    /// Diagnose and compare our conversion with third-party software
    Diagnose {
        /// Original negative image
        #[arg(value_name = "ORIGINAL")]
        original: PathBuf,

        /// Third-party converted image to compare against
        #[arg(value_name = "THIRD_PARTY")]
        third_party: PathBuf,

        /// Film preset file (optional)
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// ROI for base estimation (x,y,width,height)
        #[arg(long, value_name = "X,Y,W,H")]
        roi: Option<String>,

        /// Output directory for diagnostic images
        #[arg(short, long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Skip tone curve application
        #[arg(long)]
        no_tonecurve: bool,

        /// Skip color matrix correction
        #[arg(long)]
        no_colormatrix: bool,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        exposure: f32,

        /// Enable debug output
        #[arg(long)]
        debug: bool,
    },

    /// Test and optimize parameters against reference conversion
    TestParams {
        /// Original negative image
        #[arg(value_name = "ORIGINAL")]
        original: PathBuf,

        /// Reference conversion to match
        #[arg(value_name = "REFERENCE")]
        reference: PathBuf,

        /// Run grid search over parameter space
        #[arg(long)]
        grid: bool,

        /// Use parallel grid search (much faster for large grids)
        #[arg(long)]
        parallel: bool,

        /// Use adaptive search (coarse->fine refinement, most efficient)
        #[arg(long)]
        adaptive: bool,

        /// Target score for adaptive search (stops when reached)
        #[arg(long, value_name = "FLOAT", default_value = "0.05")]
        target_score: f32,

        /// Maximum refinement iterations for adaptive search
        #[arg(long, value_name = "N", default_value = "5")]
        max_iterations: usize,

        /// Number of top results to show
        #[arg(long, value_name = "N", default_value = "5")]
        top: usize,

        /// Output JSON file for results
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Save test conversion output for visual comparison
        #[arg(long, value_name = "FILE")]
        save_output: Option<PathBuf>,

        /// Test specific clip percent value (skip grid search)
        #[arg(long, value_name = "FLOAT")]
        clip_percent: Option<f32>,

        /// Test specific tone curve strength (skip grid search)
        #[arg(long, value_name = "FLOAT")]
        tone_strength: Option<f32>,

        /// Test specific exposure compensation (skip grid search)
        #[arg(long, value_name = "FLOAT")]
        exposure: Option<f32>,
    },
}

#[derive(Subcommand)]
enum PresetAction {
    /// List available presets
    List {
        /// Directory to list presets from
        #[arg(short, long, value_name = "DIR")]
        dir: Option<PathBuf>,
    },

    /// Show details of a preset
    Show {
        /// Preset name or file path
        preset: String,
    },

    /// Create a new preset template
    Create {
        /// Output file path
        output: PathBuf,

        /// Preset name
        #[arg(short, long)]
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Convert {
            input,
            out,
            preset,
            scan_profile,
            export,
            no_tonecurve,
            no_colormatrix,
            exposure,
            inversion,
            base,
            silent,
            cpu,
            auto_wb,
            auto_wb_strength,
            verbose,
            debug,
        } => cmd_convert(
            input,
            out,
            preset,
            scan_profile,
            export,
            no_tonecurve,
            no_colormatrix,
            exposure,
            inversion,
            base,
            silent,
            cpu,
            auto_wb,
            auto_wb_strength,
            verbose,
            debug,
        ),

        Commands::Analyze {
            input,
            roi,
            base_method,
            border_percent,
            json,
            save,
            verbose,
        } => cmd_analyze(input, roi, base_method, border_percent, json, save, verbose),

        Commands::Batch {
            inputs,
            base_from,
            preset,
            export,
            out,
            threads,
            silent,
            verbose,
            cpu,
        } => cmd_batch(inputs, base_from, preset, export, out, threads, silent, verbose, cpu),

        Commands::Preset { action } => match action {
            PresetAction::List { dir } => cmd_preset_list(dir),
            PresetAction::Show { preset } => cmd_preset_show(preset),
            PresetAction::Create { output, name } => cmd_preset_create(output, name),
        },

        Commands::Init { force } => cmd_init(force),

        Commands::Diagnose {
            original,
            third_party,
            preset,
            roi,
            out,
            no_tonecurve,
            no_colormatrix,
            exposure,
            debug,
        } => cmd_diagnose(
            original,
            third_party,
            preset,
            roi,
            out,
            no_tonecurve,
            no_colormatrix,
            exposure,
            debug,
        ),

        Commands::TestParams {
            original,
            reference,
            grid,
            parallel,
            adaptive,
            target_score,
            max_iterations,
            top,
            output,
            save_output,
            clip_percent,
            tone_strength,
            exposure,
        } => cmd_test_params(
            original,
            reference,
            grid,
            parallel,
            adaptive,
            target_score,
            max_iterations,
            top,
            output,
            save_output,
            clip_percent,
            tone_strength,
            exposure,
        ),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_convert(
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
    verbose: bool,
    debug: bool,
) -> Result<(), String> {
    let start_time = Instant::now();

    // Set verbose mode for core library and log config source
    if verbose || debug {
        invers_core::config::set_verbose(true);
        invers_core::config::log_config_usage();
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
                println!("  Default gamma: [{:.2}, {:.2}, {:.2}]", gamma[0], gamma[1], gamma[2]);
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

    let output_dir = output_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();

    // Parse inversion mode if specified
    let inversion_mode = parse_inversion_mode(inversion.as_deref())?;
    if !silent {
        if let Some(mode) = &inversion_mode {
            println!("Using inversion mode: {:?}", mode);
        }
    }

    // GPU is enabled by default, --cpu forces CPU-only processing
    let use_gpu = !cpu_only;

    // Auto-detect B&W mode for grayscale/monochrome images (unless user specified a mode)
    let effective_inversion_mode = if inversion_mode.is_none()
        && (decoded.source_is_grayscale || decoded.is_monochrome)
    {
        if !silent {
            println!("  Detected B&W image, using BlackAndWhite inversion mode");
        }
        Some(invers_core::models::InversionMode::BlackAndWhite)
    } else {
        inversion_mode
    };

    // Build conversion options using shared utility (with sensible defaults)
    let mut options = build_convert_options_full_with_gpu(
        input.clone(),
        output_dir,
        &export,
        "linear-rec2020".to_string(), // Default colorspace
        Some(base_estimation),
        film_preset,
        scan_profile,
        no_tonecurve,
        no_colormatrix,
        exposure,
        effective_inversion_mode,
        false, // no_auto_levels - use default (enabled)
        false, // preserve_headroom - use default
        false, // no_clip - use default
        auto_wb,
        auto_wb_strength,
        debug,
        use_gpu,
    )?;

    // Debug output: show pipeline configuration
    if debug {
        eprintln!("\n[debug] Pipeline configuration:");
        eprintln!("  Inversion mode: {:?}", options.inversion_mode);
        eprintln!("  Auto levels: {} (clip: {:.2}%)", options.enable_auto_levels, options.auto_levels_clip_percent);
        eprintln!("  Preserve headroom: {}", options.preserve_headroom);
        eprintln!("  Auto color: {} (strength: {:.2})", options.enable_auto_color, options.auto_color_strength);
        eprintln!("  Auto color gain: [{:.2}, {:.2}]", options.auto_color_min_gain, options.auto_color_max_gain);
        eprintln!("  Auto WB: {} (strength: {:.2})", options.enable_auto_wb, options.auto_wb_strength);
        eprintln!("  Auto exposure: {} (target: {:.2}, strength: {:.2})",
            options.enable_auto_exposure, options.auto_exposure_target_median, options.auto_exposure_strength);
        eprintln!("  Exposure compensation: {:.2}", options.exposure_compensation);
        eprintln!("  Shadow lift: {:?} (value: {:.3})", options.shadow_lift_mode, options.shadow_lift_value);
        eprintln!("  Highlight compression: {:.2}", options.highlight_compression);
        eprintln!("  Skip tone curve: {}", options.skip_tone_curve);
        eprintln!("  Skip color matrix: {}", options.skip_color_matrix);
        eprintln!("  Base sampling: {:?} (brightest: {:.1}%)", options.base_sampling_mode, options.base_brightest_percent);
        eprintln!("  GPU processing: {}", options.use_gpu);
        eprintln!();
    }

    // Check if we're using B&W mode (auto-detected or user-specified)
    let using_bw_mode = effective_inversion_mode
        == Some(invers_core::models::InversionMode::BlackAndWhite);

    // For B&W images/mode, disable color-specific operations
    if decoded.source_is_grayscale || decoded.is_monochrome || using_bw_mode {
        options.enable_auto_color = false;
        options.enable_auto_wb = false;
        options.skip_color_matrix = true;
    }

    // Process image
    if !silent {
        println!("Processing image...");
    }
    let mut processed = invers_core::pipeline::process_image(decoded, &options)?;

    // Force grayscale export if B&W mode was used (even on RGB source)
    if using_bw_mode {
        processed.export_as_grayscale = true;
    }

    // Export
    if !silent {
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
            invers_core::exporters::export_tiff16(&processed, &output_path, None)?;
        }
        invers_core::models::OutputFormat::LinearDng => {
            return Err("Linear DNG export not yet implemented for M1".to_string());
        }
    }

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

/// Analysis result structure for JSON output
#[derive(Serialize)]
struct AnalysisResult {
    file: String,
    dimensions: [u32; 2],
    channels: u8,
    base_estimation: BaseEstimationResult,
    channel_stats: ChannelStats,
}

#[derive(Serialize)]
struct BaseEstimationResult {
    method: String,
    medians: [f32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    noise_stats: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    roi: Option<(u32, u32, u32, u32)>,
}

#[derive(Serialize)]
struct ChannelStats {
    red: ChannelStat,
    green: ChannelStat,
    blue: ChannelStat,
}

#[derive(Serialize)]
struct ChannelStat {
    min: f32,
    max: f32,
    mean: f32,
}

fn compute_channel_stats(decoded: &invers_core::decoders::DecodedImage) -> ChannelStats {
    let pixels = &decoded.data;
    let channels = decoded.channels as usize;

    let mut r_min = f32::MAX;
    let mut r_max = f32::MIN;
    let mut r_sum = 0.0f64;
    let mut g_min = f32::MAX;
    let mut g_max = f32::MIN;
    let mut g_sum = 0.0f64;
    let mut b_min = f32::MAX;
    let mut b_max = f32::MIN;
    let mut b_sum = 0.0f64;

    let pixel_count = pixels.len() / channels;

    for i in 0..pixel_count {
        let r = pixels[i * channels];
        let g = pixels[i * channels + 1];
        let b = pixels[i * channels + 2];

        r_min = r_min.min(r);
        r_max = r_max.max(r);
        r_sum += r as f64;

        g_min = g_min.min(g);
        g_max = g_max.max(g);
        g_sum += g as f64;

        b_min = b_min.min(b);
        b_max = b_max.max(b);
        b_sum += b as f64;
    }

    let count = pixel_count as f64;

    ChannelStats {
        red: ChannelStat {
            min: r_min,
            max: r_max,
            mean: (r_sum / count) as f32,
        },
        green: ChannelStat {
            min: g_min,
            max: g_max,
            mean: (g_sum / count) as f32,
        },
        blue: ChannelStat {
            min: b_min,
            max: b_max,
            mean: (b_sum / count) as f32,
        },
    }
}

fn cmd_analyze(
    input: PathBuf,
    roi: Option<String>,
    base_method: String,
    border_percent: f32,
    json_output: bool,
    save: Option<PathBuf>,
    verbose: bool,
) -> Result<(), String> {
    // Decode input image
    let decoded = invers_core::decoders::decode_image(&input)?;

    // Parse ROI if provided
    let roi_rect = if let Some(roi_str) = &roi {
        Some(parse_roi(roi_str)?)
    } else {
        None
    };

    // Parse base estimation method
    let method = match base_method.to_lowercase().as_str() {
        "border" => invers_core::models::BaseEstimationMethod::Border,
        "regions" | _ => invers_core::models::BaseEstimationMethod::Regions,
    };

    // Estimate base
    let base_estimation = invers_core::pipeline::estimate_base(
        &decoded,
        roi_rect,
        Some(method),
        Some(border_percent),
    )?;

    // Compute channel statistics
    let channel_stats = compute_channel_stats(&decoded);

    // Build analysis result
    let result = AnalysisResult {
        file: input.display().to_string(),
        dimensions: [decoded.width, decoded.height],
        channels: decoded.channels,
        base_estimation: BaseEstimationResult {
            method: base_method.clone(),
            medians: base_estimation.medians,
            noise_stats: base_estimation.noise_stats,
            roi: base_estimation.roi,
        },
        channel_stats,
    };

    if json_output {
        // JSON output
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| format!("Failed to serialize analysis: {}", e))?;
        println!("{}", json);
    } else {
        // Human-readable output
        println!("Analyzing: {}\n", input.display());

        println!("Image Info:");
        println!("  Dimensions: {}x{}", decoded.width, decoded.height);
        println!("  Channels: {}", decoded.channels);

        println!("\nFilm Base Estimation:");
        println!("  Method: {}", base_method);
        println!(
            "  Base RGB: [{:.4}, {:.4}, {:.4}]",
            base_estimation.medians[0], base_estimation.medians[1], base_estimation.medians[2]
        );
        if let Some(noise) = &base_estimation.noise_stats {
            println!(
                "  Noise RGB: [{:.4}, {:.4}, {:.4}]",
                noise[0], noise[1], noise[2]
            );
        }
        if let Some(roi) = base_estimation.roi {
            println!("  ROI: ({}, {}, {}, {})", roi.0, roi.1, roi.2, roi.3);
        }

        if verbose {
            println!("\nChannel Statistics:");
            println!(
                "  Red:   min={:.4}, max={:.4}, mean={:.4}",
                result.channel_stats.red.min,
                result.channel_stats.red.max,
                result.channel_stats.red.mean
            );
            println!(
                "  Green: min={:.4}, max={:.4}, mean={:.4}",
                result.channel_stats.green.min,
                result.channel_stats.green.max,
                result.channel_stats.green.mean
            );
            println!(
                "  Blue:  min={:.4}, max={:.4}, mean={:.4}",
                result.channel_stats.blue.min,
                result.channel_stats.blue.max,
                result.channel_stats.blue.mean
            );
        }

        println!("\nUsage:");
        println!(
            "  invers convert {} --base {:.4},{:.4},{:.4}",
            input.display(),
            base_estimation.medians[0],
            base_estimation.medians[1],
            base_estimation.medians[2]
        );
    }

    // Save if requested
    if let Some(save_path) = save {
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| format!("Failed to serialize analysis: {}", e))?;
        std::fs::write(&save_path, &json)
            .map_err(|e| format!("Failed to write analysis file: {}", e))?;
        if !json_output {
            println!("\nAnalysis saved to: {}", save_path.display());
        }
    }

    Ok(())
}

fn cmd_batch(
    inputs: Vec<PathBuf>,
    base_from: Option<PathBuf>,
    preset: Option<PathBuf>,
    export: String,
    out: Option<PathBuf>,
    threads: Option<usize>,
    silent: bool,
    verbose: bool,
    cpu_only: bool,
) -> Result<(), String> {
    let batch_start = Instant::now();

    // Set verbose mode for core library
    invers_core::config::set_verbose(verbose);
    invers_core::config::log_config_usage();

    if inputs.is_empty() {
        return Err("No input files specified".to_string());
    }

    // Configure thread pool if specified
    if let Some(num_threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|e| format!("Failed to configure thread pool: {}", e))?;
        if !silent {
            println!("Using {} threads for parallel processing", num_threads);
        }
    }

    // Determine output directory
    let output_dir = out.clone().unwrap_or_else(|| PathBuf::from("."));
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    // Load shared base estimation if provided
    let shared_base = if let Some(base_path) = base_from {
        if !silent {
            println!("Loading base estimation from {}...", base_path.display());
        }
        let json = std::fs::read_to_string(&base_path)
            .map_err(|e| format!("Failed to read base estimation file: {}", e))?;
        let base: invers_core::models::BaseEstimation = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse base estimation: {}", e))?;
        if !silent {
            println!(
                "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
                base.medians[0], base.medians[1], base.medians[2]
            );
        }
        Some(base)
    } else {
        None
    };

    // Load shared film preset if provided
    let film_preset = if let Some(preset_path) = &preset {
        if !silent {
            println!("Loading film preset from {}...", preset_path.display());
        }
        Some(invers_core::presets::load_film_preset(preset_path)?)
    } else {
        None
    };

    if !silent {
        println!("\nProcessing {} files in parallel...\n", inputs.len());
    }

    // Progress tracking
    let processed_count = AtomicUsize::new(0);
    let total_files = inputs.len();

    // Process files in parallel, returning both path and timing
    let results: Vec<Result<(PathBuf, f64), String>> = inputs
        .par_iter()
        .map(|input| {
            let file_start = Instant::now();

            // Decode image
            let decoded = invers_core::decoders::decode_image(input)?;

            // Estimate or use shared base
            let base_estimation = if let Some(ref base) = shared_base {
                base.clone()
            } else {
                invers_core::pipeline::estimate_base(&decoded, None, None, None)?
            };

            // Build output path using shared utility
            let output_path = determine_output_path(input, &out, &export)?;

            let output_dir_for_options = output_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf();

            // GPU is enabled by default, --cpu forces CPU-only processing
            let use_gpu = !cpu_only;

            // Build conversion options using shared utility
            let options = build_convert_options_full_with_gpu(
                input.clone(),
                output_dir_for_options,
                &export,
                "linear-rec2020".to_string(),
                Some(base_estimation),
                film_preset.clone(),
                None,  // scan_profile
                false, // no_tonecurve
                false, // no_colormatrix
                1.0,   // exposure
                None,  // inversion_mode
                false, // no_auto_levels
                false, // preserve_headroom
                false, // no_clip
                false, // auto_wb
                1.0,   // auto_wb_strength
                false, // debug
                use_gpu,
            )?;

            // Process image
            let processed = invers_core::pipeline::process_image(decoded, &options)?;

            // Export
            match options.output_format {
                invers_core::models::OutputFormat::Tiff16 => {
                    invers_core::exporters::export_tiff16(&processed, &output_path, None)?;
                }
                invers_core::models::OutputFormat::LinearDng => {
                    return Err("Linear DNG export not yet implemented".to_string());
                }
            }

            let file_elapsed = file_start.elapsed().as_secs_f64();

            // Update progress
            let count = processed_count.fetch_add(1, Ordering::SeqCst) + 1;
            if !silent {
                println!(
                    "[{}/{}] {} -> {} ({:.2}s)",
                    count,
                    total_files,
                    input.display(),
                    output_path.display(),
                    file_elapsed
                );
            } else {
                println!("{}", output_path.display());
            }

            Ok((output_path, file_elapsed))
        })
        .collect();

    // Summarize results
    let mut success_count = 0;
    let mut errors: Vec<(PathBuf, String)> = Vec::new();

    for (input, result) in inputs.iter().zip(results.iter()) {
        match result {
            Ok(_) => success_count += 1,
            Err(e) => {
                errors.push((input.clone(), e.clone()));
            }
        }
    }

    let batch_elapsed = batch_start.elapsed();

    if !silent {
        println!("\n========================================");
        println!("BATCH PROCESSING COMPLETE");
        println!("========================================");
        println!("  Successful: {}", success_count);
        println!("  Failed:     {}", errors.len());
        println!("  Output dir: {}", output_dir.display());
        println!("  Total time: {:.2}s", batch_elapsed.as_secs_f64());
        if success_count > 0 {
            println!(
                "  Avg time:   {:.2}s per file",
                batch_elapsed.as_secs_f64() / success_count as f64
            );
        }

        if !errors.is_empty() {
            println!("\nErrors:");
            for (path, error) in &errors {
                println!("  {}: {}", path.display(), error);
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("{} files failed to process", errors.len()))
    }
}

fn cmd_preset_list(_dir: Option<PathBuf>) -> Result<(), String> {
    let dir = _dir.unwrap_or_else(|| {
        invers_core::presets::get_presets_dir().unwrap_or_else(|_| PathBuf::from("profiles/film"))
    });

    println!("Listing presets in: {}", dir.display());
    match invers_core::presets::list_film_presets(&dir) {
        Ok(presets) => {
            if presets.is_empty() {
                println!("No presets found.");
            } else {
                for preset in presets {
                    println!("  {}", preset);
                }
            }
            Ok(())
        }
        Err(e) => Err(format!("Failed to list presets: {}", e)),
    }
}

fn cmd_preset_show(preset: String) -> Result<(), String> {
    println!("Loading preset: {}", preset);

    // Try to load as file first
    let preset_path = PathBuf::from(&preset);
    let preset_obj = if preset_path.exists() {
        invers_core::presets::load_film_preset(&preset_path)?
    } else {
        // Try to find it in the presets directory
        let dir = invers_core::presets::get_presets_dir()
            .unwrap_or_else(|_| PathBuf::from("profiles/film"));
        let full_path = dir.join(format!("{}.yml", preset));
        invers_core::presets::load_film_preset(&full_path)?
    };

    println!("\nPreset: {}", preset_obj.name);
    println!(
        "Base Offsets (RGB): [{:.6}, {:.6}, {:.6}]",
        preset_obj.base_offsets[0], preset_obj.base_offsets[1], preset_obj.base_offsets[2]
    );

    println!("\nTone Curve:");
    println!("  Type: {}", preset_obj.tone_curve.curve_type);
    println!("  Strength: {:.6}", preset_obj.tone_curve.strength);

    println!("\nColor Matrix (3x3):");
    for row in &preset_obj.color_matrix {
        println!("  [{:.6}, {:.6}, {:.6}]", row[0], row[1], row[2]);
    }

    if let Some(notes) = &preset_obj.notes {
        println!("\nNotes: {}", notes);
    }

    println!();
    Ok(())
}

fn cmd_preset_create(output: PathBuf, name: String) -> Result<(), String> {
    println!("Creating new preset: {}", name);

    // Create a default preset
    let preset = invers_core::models::FilmPreset {
        name: name.clone(),
        base_offsets: [0.0, 0.0, 0.0],
        color_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        tone_curve: invers_core::models::ToneCurveParams {
            curve_type: "neutral".to_string(),
            strength: 0.5,
            toe_strength: 0.4,
            shoulder_strength: 0.3,
            toe_length: 0.25,
            shoulder_start: 0.75,
            params: std::collections::HashMap::new(),
        },
        notes: Some(format!("Film preset: {}", name)),
    };

    // Serialize to YAML
    let yaml_str =
        serde_yaml::to_string(&preset).map_err(|e| format!("Failed to serialize preset: {}", e))?;

    // Write to file
    std::fs::write(&output, yaml_str).map_err(|e| format!("Failed to write preset file: {}", e))?;

    println!("Preset created: {}", output.display());
    println!("You can now edit this file to customize the parameters.");
    println!();

    Ok(())
}

fn cmd_init(force: bool) -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "Could not determine home directory")?;
    let invers_dir = PathBuf::from(&home).join("invers");

    println!("Initializing invers configuration in: {}", invers_dir.display());
    println!();

    // Look for default presets in common Homebrew locations
    let share_locations = [
        PathBuf::from("/opt/homebrew/opt/invers/share/invers"),  // Apple Silicon
        PathBuf::from("/usr/local/opt/invers/share/invers"),     // Intel Mac
        PathBuf::from("/home/linuxbrew/.linuxbrew/opt/invers/share/invers"), // Linux
    ];

    let share_dir = share_locations
        .iter()
        .find(|p| p.exists())
        .ok_or_else(|| {
            "Could not find invers share directory. Make sure invers is installed via Homebrew.".to_string()
        })?;

    println!("Found default presets in: {}", share_dir.display());
    println!();

    // Create directory structure
    let presets_film_dir = invers_dir.join("presets/film");
    let presets_scan_dir = invers_dir.join("presets/scan");

    std::fs::create_dir_all(&presets_film_dir)
        .map_err(|e| format!("Failed to create film presets directory: {}", e))?;
    std::fs::create_dir_all(&presets_scan_dir)
        .map_err(|e| format!("Failed to create scan presets directory: {}", e))?;

    // Copy pipeline_defaults.yml
    let src_config = share_dir.join("config/pipeline_defaults.yml");
    let dst_config = invers_dir.join("pipeline_defaults.yml");

    if src_config.exists() {
        if !dst_config.exists() || force {
            std::fs::copy(&src_config, &dst_config)
                .map_err(|e| format!("Failed to copy pipeline_defaults.yml: {}", e))?;
            println!("  Copied: pipeline_defaults.yml");
        } else {
            println!("  Skipped: pipeline_defaults.yml (already exists, use --force to overwrite)");
        }
    }

    // Copy film presets
    let src_film = share_dir.join("profiles/film");
    if src_film.exists() {
        copy_dir_contents(&src_film, &presets_film_dir, force, "  ")?;
    }

    // Copy scan profiles
    let src_scan = share_dir.join("profiles/scan");
    if src_scan.exists() {
        copy_dir_contents(&src_scan, &presets_scan_dir, force, "  ")?;
    }

    println!();
    println!("Initialization complete!");
    println!();
    println!("Configuration files are now in:");
    println!("  ~/invers/pipeline_defaults.yml  - Pipeline processing defaults");
    println!("  ~/invers/presets/film/          - Film preset profiles");
    println!("  ~/invers/presets/scan/          - Scanner profiles");

    Ok(())
}

fn copy_dir_contents(src: &PathBuf, dst: &PathBuf, force: bool, indent: &str) -> Result<(), String> {
    let entries = std::fs::read_dir(src)
        .map_err(|e| format!("Failed to read directory {}: {}", src.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let src_path = entry.path();
        let file_name = src_path.file_name().unwrap();
        let dst_path = dst.join(file_name);

        if src_path.is_dir() {
            std::fs::create_dir_all(&dst_path)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
            copy_dir_contents(&src_path, &dst_path, force, indent)?;
        } else if !dst_path.exists() || force {
            std::fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("Failed to copy {}: {}", src_path.display(), e))?;
            println!("{}Copied: {}", indent, file_name.to_string_lossy());
        } else {
            println!("{}Skipped: {} (exists)", indent, file_name.to_string_lossy());
        }
    }

    Ok(())
}

fn cmd_diagnose(
    original: PathBuf,
    third_party: PathBuf,
    preset: Option<PathBuf>,
    roi: Option<String>,
    out: Option<PathBuf>,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    debug: bool,
) -> Result<(), String> {
    invers_core::config::log_config_usage();

    println!("========================================");
    println!("INVERS DIAGNOSTIC COMPARISON");
    println!("========================================\n");

    // Step 1: Decode original negative
    println!("1. Loading original negative: {}", original.display());
    let decoded_original = invers_core::decoders::decode_image(&original)?;
    println!(
        "   Dimensions: {}x{}, {} channels",
        decoded_original.width, decoded_original.height, decoded_original.channels
    );

    // Step 2: Decode third-party conversion
    println!(
        "\n2. Loading third-party conversion: {}",
        third_party.display()
    );
    let decoded_third_party = invers_core::decoders::decode_image(&third_party)?;
    println!(
        "   Dimensions: {}x{}, {} channels",
        decoded_third_party.width, decoded_third_party.height, decoded_third_party.channels
    );

    // Check if dimensions match
    if decoded_original.width != decoded_third_party.width
        || decoded_original.height != decoded_third_party.height
    {
        eprintln!("\n⚠ WARNING: Image dimensions don't match!");
        eprintln!(
            "   Original: {}x{}, Third-party: {}x{}",
            decoded_original.width,
            decoded_original.height,
            decoded_third_party.width,
            decoded_third_party.height
        );
        eprintln!("   Comparison may not be accurate.");
    }

    // Step 3: Estimate base from original
    println!("\n3. Estimating film base from original negative...");
    let roi_rect = if let Some(roi_str) = roi {
        Some(parse_roi(&roi_str)?)
    } else {
        None
    };
    let base_estimation = invers_core::pipeline::estimate_base(&decoded_original, roi_rect, None, None)?;
    println!(
        "   Base RGB: [{:.6}, {:.6}, {:.6}]",
        base_estimation.medians[0], base_estimation.medians[1], base_estimation.medians[2]
    );

    // Step 4: Load film preset if provided
    let film_preset = if let Some(preset_path) = preset {
        println!("\n4. Loading film preset: {}", preset_path.display());
        Some(invers_core::presets::load_film_preset(&preset_path)?)
    } else {
        println!("\n4. No film preset specified, using default processing");
        None
    };

    // Step 5: Process with our pipeline
    println!("\n5. Processing with our pipeline...");
    let options = build_convert_options(
        original.clone(),
        std::path::PathBuf::from("."),
        "tiff16",
        "linear-rec2020".to_string(),
        Some(base_estimation),
        film_preset,
        no_tonecurve,
        no_colormatrix,
        exposure,
        debug,
    )?;

    let our_processed = invers_core::pipeline::process_image(decoded_original, &options)?;
    println!("   Processing complete!");

    // Step 6: Compare conversions
    println!("\n6. Performing diagnostic comparison...");
    let report =
        invers_core::diagnostics::compare_conversions(&our_processed, &decoded_third_party)?;

    // Step 7: Print report
    println!();
    invers_core::diagnostics::print_report(&report);

    // Step 8: Save diagnostic images
    let output_dir = out.unwrap_or_else(|| std::path::PathBuf::from("."));
    println!("\n7. Saving diagnostic images to: {}", output_dir.display());

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    invers_core::diagnostics::save_diagnostic_images(
        &our_processed,
        &decoded_third_party,
        &output_dir,
    )?;

    println!("\n========================================");
    println!("DIAGNOSTIC COMPLETE");
    println!("========================================");

    Ok(())
}

fn cmd_test_params(
    original: PathBuf,
    reference: PathBuf,
    grid: bool,
    parallel: bool,
    adaptive: bool,
    target_score: f32,
    max_iterations: usize,
    top: usize,
    output: Option<PathBuf>,
    save_output: Option<PathBuf>,
    clip_percent: Option<f32>,
    tone_strength: Option<f32>,
    exposure: Option<f32>,
) -> Result<(), String> {
    invers_core::config::log_config_usage();
    println!("========================================");
    println!("PARAMETER TESTING & OPTIMIZATION");
    println!("========================================\n");

    println!("Original:  {}", original.display());
    println!("Reference: {}", reference.display());
    println!();

    let results = if adaptive {
        // Adaptive search (most efficient)
        println!("Running adaptive parameter search...\n");
        invers_core::testing::run_adaptive_grid_search(
            &original,
            &reference,
            target_score,
            max_iterations,
        )?
    } else if grid {
        // Grid search (exhaustive or parallel)
        let grid_config = invers_core::testing::ParameterGrid::default();

        if parallel {
            println!("Running parallel grid search...\n");
            invers_core::testing::run_parameter_grid_search_parallel(
                &original,
                &reference,
                &grid_config,
                None,
            )?
        } else {
            println!("Running sequential grid search...\n");
            invers_core::testing::run_parameter_grid_search(&original, &reference, &grid_config)?
        }
    } else if clip_percent.is_some() || tone_strength.is_some() || exposure.is_some() {
        // Single test with specific parameters
        println!("Testing specific parameters...\n");

        let mut params = invers_core::testing::ParameterTest::default();

        if let Some(clip) = clip_percent {
            params.clip_percent = clip;
        }
        if let Some(strength) = tone_strength {
            params.tone_curve_strength = strength;
        }
        if let Some(exp) = exposure {
            params.exposure_compensation = exp;
        }

        let result = invers_core::testing::run_parameter_test(
            &original,
            &reference,
            &params,
            save_output.as_ref(),
        )?;
        vec![result]
    } else {
        // Default: test current default parameters
        println!("Testing default parameters...\n");

        let params = invers_core::testing::ParameterTest::default();
        let result = invers_core::testing::run_parameter_test(
            &original,
            &reference,
            &params,
            save_output.as_ref(),
        )?;
        vec![result]
    };

    // Print top N results
    let num_to_show = top.min(results.len());
    println!("\n{:=<80}\n", "");
    println!("TOP {} RESULTS", num_to_show);
    println!("{:=<80}", "");

    for (i, result) in results.iter().take(num_to_show).enumerate() {
        invers_core::testing::print_test_result(result, i + 1);
    }

    // Save to JSON if requested
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| format!("Failed to serialize results: {}", e))?;
        std::fs::write(&output_path, json)
            .map_err(|e| format!("Failed to write output file: {}", e))?;
        println!("\n{:=<80}", "");
        println!("Results saved to: {}", output_path.display());
        println!("Total combinations tested: {}", results.len());
    }

    println!("\n{:=<80}", "");
    println!("TESTING COMPLETE");
    println!("{:=<80}\n", "");

    Ok(())
}
