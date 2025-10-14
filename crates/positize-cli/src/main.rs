use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "positize")]
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

        /// Output directory
        #[arg(short, long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Film preset file
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// Scan profile file
        #[arg(short, long, value_name = "FILE")]
        scan_profile: Option<PathBuf>,

        /// ROI for base estimation (x,y,width,height)
        #[arg(long, value_name = "X,Y,W,H")]
        roi: Option<String>,

        /// Export format (tiff16 or dng)
        #[arg(long, value_name = "FORMAT", default_value = "tiff16")]
        export: String,

        /// Working colorspace
        #[arg(long, value_name = "COLORSPACE", default_value = "linear-rec2020")]
        colorspace: String,

        /// Number of parallel threads
        #[arg(short = 'j', long, value_name = "N")]
        threads: Option<usize>,

        /// Skip tone curve application
        #[arg(long)]
        no_tonecurve: bool,

        /// Skip color matrix correction
        #[arg(long)]
        no_colormatrix: bool,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        exposure: f32,

        /// Enable debug output showing intermediate statistics
        #[arg(long)]
        debug: bool,
    },

    /// Analyze and estimate film base from ROI
    AnalyzeBase {
        /// Input file
        input: PathBuf,

        /// ROI for base estimation (x,y,width,height)
        #[arg(long, value_name = "X,Y,W,H")]
        roi: Option<String>,

        /// Save base estimation to file
        #[arg(short, long, value_name = "FILE")]
        save: Option<PathBuf>,

        /// Use auto-estimation heuristic
        #[arg(long)]
        auto: bool,
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
    },

    /// Manage film presets
    Preset {
        #[command(subcommand)]
        action: PresetAction,
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

        /// Reference conversion to match (e.g., Grain2Pixel output)
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
            roi,
            export,
            colorspace,
            threads,
            no_tonecurve,
            no_colormatrix,
            exposure,
            debug,
        } => cmd_convert(
            input,
            out,
            preset,
            scan_profile,
            roi,
            export,
            colorspace,
            threads,
            no_tonecurve,
            no_colormatrix,
            exposure,
            debug,
        ),

        Commands::AnalyzeBase {
            input,
            roi,
            save,
            auto,
        } => cmd_analyze_base(input, roi, save, auto),

        Commands::Batch {
            inputs,
            base_from,
            preset,
            export,
            out,
            threads,
        } => cmd_batch(inputs, base_from, preset, export, out, threads),

        Commands::Preset { action } => match action {
            PresetAction::List { dir } => cmd_preset_list(dir),
            PresetAction::Show { preset } => cmd_preset_show(preset),
            PresetAction::Create { output, name } => cmd_preset_create(output, name),
        },

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
    _scan_profile: Option<PathBuf>,
    roi: Option<String>,
    export: String,
    colorspace: String,
    _threads: Option<usize>,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    debug: bool,
) -> Result<(), String> {
    positize_core::config::log_config_usage();
    let config_handle = positize_core::config::pipeline_config_handle();
    let defaults = config_handle.config.defaults.clone();

    println!("Converting {} to positive...", input.display());

    // Decode input image
    println!("Decoding image...");
    let decoded = positize_core::decoders::decode_image(&input)?;
    println!(
        "  Image: {}x{}, {} channels",
        decoded.width, decoded.height, decoded.channels
    );

    // Parse ROI if provided
    let roi_rect = if let Some(roi_str) = roi {
        Some(parse_roi(&roi_str)?)
    } else {
        None
    };

    // Estimate or load base
    println!("Estimating film base...");
    let base_estimation = positize_core::pipeline::estimate_base(&decoded, roi_rect)?;
    println!(
        "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
        base_estimation.medians[0], base_estimation.medians[1], base_estimation.medians[2]
    );
    if let Some(noise) = &base_estimation.noise_stats {
        println!(
            "  Noise (RGB): [{:.4}, {:.4}, {:.4}]",
            noise[0], noise[1], noise[2]
        );
    }

    // Load film preset if provided
    let film_preset = if let Some(preset_path) = preset {
        println!("Loading film preset from {}...", preset_path.display());
        Some(positize_core::presets::load_film_preset(&preset_path)?)
    } else {
        None
    };

    // Prepare output path
    let output_path = determine_output_path(&input, &out, &export)?;
    println!("Output: {}", output_path.display());

    // Parse output format
    let output_format = match export.as_str() {
        "tiff16" | "tiff" => positize_core::models::OutputFormat::Tiff16,
        "dng" => positize_core::models::OutputFormat::LinearDng,
        _ => return Err(format!("Unknown export format: {}", export)),
    };

    // Build conversion options
    let options = positize_core::models::ConvertOptions {
        input_paths: vec![input.clone()],
        output_dir: output_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf(),
        output_format,
        working_colorspace: colorspace.clone(),
        bit_depth_policy: positize_core::models::BitDepthPolicy::Force16Bit,
        film_preset,
        scan_profile: None,
        base_estimation: Some(base_estimation),
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
    };

    // Process image
    println!("Processing image...");
    let processed = positize_core::pipeline::process_image(decoded, &options)?;

    // Export
    println!(
        "Exporting to {}...",
        if output_format == positize_core::models::OutputFormat::Tiff16 {
            "TIFF16"
        } else {
            "DNG"
        }
    );
    match output_format {
        positize_core::models::OutputFormat::Tiff16 => {
            positize_core::exporters::export_tiff16(&processed, &output_path, None)?;
        }
        positize_core::models::OutputFormat::LinearDng => {
            return Err("Linear DNG export not yet implemented for M1".to_string());
        }
    }

    println!("Done! Positive image saved to: {}", output_path.display());
    Ok(())
}

fn cmd_analyze_base(
    input: PathBuf,
    roi: Option<String>,
    save: Option<PathBuf>,
    auto: bool,
) -> Result<(), String> {
    println!("Analyzing film base from {}...", input.display());

    // Decode input image
    let decoded = positize_core::decoders::decode_image(&input)?;
    println!(
        "Image: {}x{}, {} channels",
        decoded.width, decoded.height, decoded.channels
    );

    // Parse ROI if provided, or use auto if specified
    let roi_rect = if auto {
        println!("Using automatic base estimation...");
        None
    } else if let Some(roi_str) = roi {
        Some(parse_roi(&roi_str)?)
    } else {
        return Err("Either --roi or --auto must be specified".to_string());
    };

    // Estimate base
    let base_estimation = positize_core::pipeline::estimate_base(&decoded, roi_rect)?;

    // Print results
    println!("\nFilm Base Estimation:");
    if let Some(roi) = base_estimation.roi {
        println!("  ROI: ({}, {}, {}, {})", roi.0, roi.1, roi.2, roi.3);
    }
    println!(
        "  Medians (RGB): [{:.6}, {:.6}, {:.6}]",
        base_estimation.medians[0], base_estimation.medians[1], base_estimation.medians[2]
    );
    if let Some(noise) = &base_estimation.noise_stats {
        println!(
            "  Noise (RGB): [{:.6}, {:.6}, {:.6}]",
            noise[0], noise[1], noise[2]
        );
    }
    println!("  Auto-estimated: {}", base_estimation.auto_estimated);

    // Save if requested
    if let Some(save_path) = save {
        let json = serde_json::to_string_pretty(&base_estimation)
            .map_err(|e| format!("Failed to serialize base estimation: {}", e))?;
        std::fs::write(&save_path, json)
            .map_err(|e| format!("Failed to write base estimation file: {}", e))?;
        println!("\nBase estimation saved to: {}", save_path.display());
    }

    Ok(())
}

/// Parse ROI string in format "x,y,width,height"
fn parse_roi(roi_str: &str) -> Result<(u32, u32, u32, u32), String> {
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
fn determine_output_path(
    input: &PathBuf,
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

fn cmd_batch(
    _inputs: Vec<PathBuf>,
    _base_from: Option<PathBuf>,
    _preset: Option<PathBuf>,
    _export: String,
    _out: Option<PathBuf>,
    _threads: Option<usize>,
) -> Result<(), String> {
    println!("Batch command - not yet implemented");
    Ok(())
}

fn cmd_preset_list(_dir: Option<PathBuf>) -> Result<(), String> {
    let dir = _dir.unwrap_or_else(|| {
        positize_core::presets::get_presets_dir().unwrap_or_else(|_| PathBuf::from("profiles/film"))
    });

    println!("Listing presets in: {}", dir.display());
    match positize_core::presets::list_film_presets(&dir) {
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

fn cmd_preset_show(_preset: String) -> Result<(), String> {
    println!("Show preset command - not yet implemented");
    Ok(())
}

fn cmd_preset_create(_output: PathBuf, _name: String) -> Result<(), String> {
    println!("Create preset command - not yet implemented");
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
    positize_core::config::log_config_usage();
    let config_handle = positize_core::config::pipeline_config_handle();
    let defaults = config_handle.config.defaults.clone();

    println!("========================================");
    println!("POSITIZE DIAGNOSTIC COMPARISON");
    println!("========================================\n");

    // Step 1: Decode original negative
    println!("1. Loading original negative: {}", original.display());
    let decoded_original = positize_core::decoders::decode_image(&original)?;
    println!(
        "   Dimensions: {}x{}, {} channels",
        decoded_original.width, decoded_original.height, decoded_original.channels
    );

    // Step 2: Decode third-party conversion
    println!(
        "\n2. Loading third-party conversion: {}",
        third_party.display()
    );
    let decoded_third_party = positize_core::decoders::decode_image(&third_party)?;
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
    let base_estimation = positize_core::pipeline::estimate_base(&decoded_original, roi_rect)?;
    println!(
        "   Base RGB: [{:.6}, {:.6}, {:.6}]",
        base_estimation.medians[0], base_estimation.medians[1], base_estimation.medians[2]
    );

    // Step 4: Load film preset if provided
    let film_preset = if let Some(preset_path) = preset {
        println!("\n4. Loading film preset: {}", preset_path.display());
        Some(positize_core::presets::load_film_preset(&preset_path)?)
    } else {
        println!("\n4. No film preset specified, using default processing");
        None
    };

    // Step 5: Process with our pipeline
    println!("\n5. Processing with our pipeline...");
    let options = positize_core::models::ConvertOptions {
        input_paths: vec![original.clone()],
        output_dir: std::path::PathBuf::from("."),
        output_format: positize_core::models::OutputFormat::Tiff16,
        working_colorspace: "linear-rec2020".to_string(),
        bit_depth_policy: positize_core::models::BitDepthPolicy::Force16Bit,
        film_preset,
        scan_profile: None,
        base_estimation: Some(base_estimation),
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
    };

    let our_processed = positize_core::pipeline::process_image(decoded_original, &options)?;
    println!("   Processing complete!");

    // Step 6: Compare conversions
    println!("\n6. Performing diagnostic comparison...");
    let report =
        positize_core::diagnostics::compare_conversions(&our_processed, &decoded_third_party)?;

    // Step 7: Print report
    println!();
    positize_core::diagnostics::print_report(&report);

    // Step 8: Save diagnostic images
    let output_dir = out.unwrap_or_else(|| std::path::PathBuf::from("."));
    println!("\n7. Saving diagnostic images to: {}", output_dir.display());

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    positize_core::diagnostics::save_diagnostic_images(
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
    positize_core::config::log_config_usage();
    println!("========================================");
    println!("PARAMETER TESTING & OPTIMIZATION");
    println!("========================================\n");

    println!("Original:  {}", original.display());
    println!("Reference: {}", reference.display());
    println!();

    let results = if adaptive {
        // Adaptive search (most efficient)
        println!("Running adaptive parameter search...\n");
        positize_core::testing::run_adaptive_grid_search(
            &original,
            &reference,
            target_score,
            max_iterations,
        )?
    } else if grid {
        // Grid search (exhaustive or parallel)
        let grid_config = positize_core::testing::ParameterGrid::default();

        if parallel {
            println!("Running parallel grid search...\n");
            positize_core::testing::run_parameter_grid_search_parallel(
                &original,
                &reference,
                &grid_config,
                None,
            )?
        } else {
            println!("Running sequential grid search...\n");
            positize_core::testing::run_parameter_grid_search(&original, &reference, &grid_config)?
        }
    } else if clip_percent.is_some() || tone_strength.is_some() || exposure.is_some() {
        // Single test with specific parameters
        println!("Testing specific parameters...\n");

        let mut params = positize_core::testing::ParameterTest::default();

        if let Some(clip) = clip_percent {
            params.clip_percent = clip;
        }
        if let Some(strength) = tone_strength {
            params.tone_curve_strength = strength;
        }
        if let Some(exp) = exposure {
            params.exposure_compensation = exp;
        }

        let result = positize_core::testing::run_parameter_test(
            &original,
            &reference,
            &params,
            save_output.as_ref(),
        )?;
        vec![result]
    } else {
        // Default: test current default parameters
        println!("Testing default parameters...\n");

        let params = positize_core::testing::ParameterTest::default();
        let result = positize_core::testing::run_parameter_test(
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
        positize_core::testing::print_test_result(result, i + 1);
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
