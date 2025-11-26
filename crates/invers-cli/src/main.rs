use clap::{Parser, Subcommand};
use invers_cli::{
    build_convert_options, build_convert_options_full, build_convert_options_with_inversion,
    determine_output_path, parse_base_rgb, parse_inversion_mode, parse_roi,
};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

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

        /// Base estimation method: "regions" (default) or "border"
        #[arg(long, value_name = "METHOD", default_value = "regions")]
        base_method: String,

        /// Border percentage for "border" base method (1-25%, default: 5)
        #[arg(long, value_name = "PERCENT", default_value = "5.0")]
        border_percent: f32,

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

        /// Inversion mode: "linear" (default), "log", or "divide-blend"
        #[arg(long, value_name = "MODE")]
        inversion: Option<String>,

        /// Manual base RGB values (comma-separated: R,G,B)
        /// Overrides automatic base estimation
        #[arg(long, value_name = "R,G,B")]
        base: Option<String>,

        /// Skip auto-levels (histogram stretching)
        #[arg(long)]
        no_auto_levels: bool,

        /// Preserve shadow/highlight headroom (don't stretch to full 0-1 range)
        /// Sets output range to approximately 0.005-0.98
        #[arg(long)]
        preserve_headroom: bool,

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
            roi,
            base_method,
            border_percent,
            export,
            colorspace,
            threads,
            no_tonecurve,
            no_colormatrix,
            exposure,
            inversion,
            base,
            no_auto_levels,
            preserve_headroom,
            debug,
        } => cmd_convert(
            input,
            out,
            preset,
            scan_profile,
            roi,
            base_method,
            border_percent,
            export,
            colorspace,
            threads,
            no_tonecurve,
            no_colormatrix,
            exposure,
            inversion,
            base,
            no_auto_levels,
            preserve_headroom,
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
    scan_profile_path: Option<PathBuf>,
    roi: Option<String>,
    base_method: String,
    border_percent: f32,
    export: String,
    colorspace: String,
    _threads: Option<usize>,
    no_tonecurve: bool,
    no_colormatrix: bool,
    exposure: f32,
    inversion: Option<String>,
    base: Option<String>,
    no_auto_levels: bool,
    preserve_headroom: bool,
    debug: bool,
) -> Result<(), String> {
    invers_core::config::log_config_usage();

    println!("Converting {} to positive...", input.display());

    // Decode input image
    println!("Decoding image...");
    let decoded = invers_core::decoders::decode_image(&input)?;
    println!(
        "  Image: {}x{}, {} channels",
        decoded.width, decoded.height, decoded.channels
    );

    // Load scan profile if provided
    let scan_profile = if let Some(profile_path) = scan_profile_path {
        println!("Loading scan profile from {}...", profile_path.display());
        let profile = invers_core::presets::load_scan_profile(&profile_path)?;
        println!("  Profile: {} ({})", profile.name, profile.source_type);
        if let Some(ref hsl) = profile.hsl_adjustments {
            if hsl.has_adjustments() {
                println!("  HSL adjustments: enabled");
            }
        }
        if let Some(gamma) = profile.default_gamma {
            println!("  Default gamma: [{:.2}, {:.2}, {:.2}]", gamma[0], gamma[1], gamma[2]);
        }
        Some(profile)
    } else {
        None
    };

    // Parse manual base values or estimate
    let base_estimation = if let Some(base_str) = base {
        // Parse manual base values (R,G,B format)
        let base_rgb = parse_base_rgb(&base_str)?;
        println!("Using manual base values...");
        println!(
            "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
            base_rgb[0], base_rgb[1], base_rgb[2]
        );
        invers_core::models::BaseEstimation {
            roi: None,
            medians: base_rgb,
            noise_stats: None,
            auto_estimated: false,
        }
    } else {
        // Parse ROI if provided
        let roi_rect = if let Some(roi_str) = roi {
            Some(parse_roi(&roi_str)?)
        } else {
            None
        };

        // Parse base estimation method
        let method = match base_method.to_lowercase().as_str() {
            "border" => Some(invers_core::models::BaseEstimationMethod::Border),
            "regions" | _ => Some(invers_core::models::BaseEstimationMethod::Regions),
        };

        // Estimate base
        println!("Estimating film base...");
        let estimation = invers_core::pipeline::estimate_base(
            &decoded,
            roi_rect,
            method,
            Some(border_percent),
        )?;
        println!(
            "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
            estimation.medians[0], estimation.medians[1], estimation.medians[2]
        );
        if let Some(noise) = &estimation.noise_stats {
            println!(
                "  Noise (RGB): [{:.4}, {:.4}, {:.4}]",
                noise[0], noise[1], noise[2]
            );
        }
        estimation
    };

    // Load film preset if provided
    let film_preset = if let Some(preset_path) = preset {
        println!("Loading film preset from {}...", preset_path.display());
        Some(invers_core::presets::load_film_preset(&preset_path)?)
    } else {
        None
    };

    // Prepare output path
    let output_path = determine_output_path(&input, &out, &export)?;
    println!("Output: {}", output_path.display());

    let output_dir = output_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();

    // Parse inversion mode if specified
    let inversion_mode = parse_inversion_mode(inversion.as_deref())?;
    if let Some(mode) = &inversion_mode {
        println!("Using inversion mode: {:?}", mode);
    }

    if no_auto_levels {
        println!("Auto-levels disabled");
    }
    if preserve_headroom {
        println!("Preserving shadow/highlight headroom");
    }

    // Build conversion options using shared utility
    let options = build_convert_options_full(
        input.clone(),
        output_dir,
        &export,
        colorspace,
        Some(base_estimation),
        film_preset,
        scan_profile,
        no_tonecurve,
        no_colormatrix,
        exposure,
        inversion_mode,
        no_auto_levels,
        preserve_headroom,
        debug,
    )?;

    // Process image
    println!("Processing image...");
    let processed = invers_core::pipeline::process_image(decoded, &options)?;

    // Export
    println!(
        "Exporting to {}...",
        if options.output_format == invers_core::models::OutputFormat::Tiff16 {
            "TIFF16"
        } else {
            "DNG"
        }
    );
    match options.output_format {
        invers_core::models::OutputFormat::Tiff16 => {
            invers_core::exporters::export_tiff16(&processed, &output_path, None)?;
        }
        invers_core::models::OutputFormat::LinearDng => {
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
    let decoded = invers_core::decoders::decode_image(&input)?;
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

    // Estimate base (use default regions method for analyze-base command)
    let base_estimation = invers_core::pipeline::estimate_base(&decoded, roi_rect, None, None)?;

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

fn cmd_batch(
    inputs: Vec<PathBuf>,
    base_from: Option<PathBuf>,
    preset: Option<PathBuf>,
    export: String,
    out: Option<PathBuf>,
    threads: Option<usize>,
) -> Result<(), String> {
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
        println!("Using {} threads for parallel processing", num_threads);
    }

    // Determine output directory
    let output_dir = out.clone().unwrap_or_else(|| PathBuf::from("."));
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    // Load shared base estimation if provided
    let shared_base = if let Some(base_path) = base_from {
        println!("Loading base estimation from {}...", base_path.display());
        let json = std::fs::read_to_string(&base_path)
            .map_err(|e| format!("Failed to read base estimation file: {}", e))?;
        let base: invers_core::models::BaseEstimation = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse base estimation: {}", e))?;
        println!(
            "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
            base.medians[0], base.medians[1], base.medians[2]
        );
        Some(base)
    } else {
        None
    };

    // Load shared film preset if provided
    let film_preset = if let Some(preset_path) = &preset {
        println!("Loading film preset from {}...", preset_path.display());
        Some(invers_core::presets::load_film_preset(preset_path)?)
    } else {
        None
    };

    println!("\nProcessing {} files in parallel...\n", inputs.len());

    // Progress tracking
    let processed_count = AtomicUsize::new(0);
    let total_files = inputs.len();

    // Process files in parallel
    let results: Vec<Result<PathBuf, String>> = inputs
        .par_iter()
        .map(|input| {
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

            // Build conversion options using shared utility
            let options = build_convert_options(
                input.clone(),
                output_dir_for_options,
                &export,
                "linear-rec2020".to_string(),
                Some(base_estimation),
                film_preset.clone(),
                false,
                false,
                1.0,
                false,
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

            // Update progress
            let count = processed_count.fetch_add(1, Ordering::SeqCst) + 1;
            println!(
                "[{}/{}] Processed: {} -> {}",
                count,
                total_files,
                input.display(),
                output_path.display()
            );

            Ok(output_path)
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

    println!("\n========================================");
    println!("BATCH PROCESSING COMPLETE");
    println!("========================================");
    println!("  Successful: {}", success_count);
    println!("  Failed:     {}", errors.len());
    println!("  Output dir: {}", output_dir.display());

    if !errors.is_empty() {
        println!("\nErrors:");
        for (path, error) in &errors {
            println!("  {}: {}", path.display(), error);
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
