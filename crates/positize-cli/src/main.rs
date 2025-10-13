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
        } => cmd_convert(input, out, preset, scan_profile, roi, export, colorspace, threads, no_tonecurve, no_colormatrix, exposure, debug),

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
        output_dir: output_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf(),
        output_format,
        working_colorspace: colorspace.clone(),
        bit_depth_policy: positize_core::models::BitDepthPolicy::Force16Bit,
        film_preset,
        scan_profile: None,
        base_estimation: Some(base_estimation),
        num_threads: None,
        skip_tone_curve: no_tonecurve,
        skip_color_matrix: no_colormatrix,
        exposure_compensation: exposure,
        debug,
    };

    // Process image
    println!("Processing image...");
    let processed = positize_core::pipeline::process_image(decoded, &options)?;

    // Export
    println!("Exporting to {}...", if output_format == positize_core::models::OutputFormat::Tiff16 { "TIFF16" } else { "DNG" });
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
        positize_core::presets::get_presets_dir()
            .unwrap_or_else(|_| PathBuf::from("profiles/film"))
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
