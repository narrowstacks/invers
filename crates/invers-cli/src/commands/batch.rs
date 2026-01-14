use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use invers_cli::{
    determine_output_path, expand_inputs, make_base_from_rgb, parse_base_rgb, process_single_image,
    ProcessingParams, WhiteBalance,
};

use invers_core::models::BaseEstimation;

/// Strategy for determining base estimation in batch mode
enum BaseStrategy {
    /// Load from JSON file (highest priority)
    FromFile(BaseEstimation),
    /// Manual RGB values from command line
    Manual(BaseEstimation),
    /// Estimate from first image, share with all (default)
    FirstImage,
    /// Estimate each image independently (opt-in)
    PerImage,
}

/// Execute the batch processing command.
///
/// Processes multiple negative images with shared settings, optionally using
/// a common base estimation across all files (same-roll mode).
///
/// # Base Estimation Priority
/// 1. `--base-from`: Load from JSON file (highest priority)
/// 2. `--base`: Manual RGB values from command line
/// 3. First-image estimation (default): Share base from first image
/// 4. `--per-image`: Estimate each image independently (opt-in)
///
/// # Returns
/// Returns `Ok(())` if all files processed successfully, or an error message
/// indicating how many files failed.
#[allow(clippy::too_many_arguments)]
pub fn cmd_batch(
    inputs: Vec<PathBuf>,
    recursive: bool,
    base_from: Option<PathBuf>,
    base: Option<String>,
    per_image: bool,
    export: String,
    white_balance: WhiteBalance,
    exposure: f32,
    out: Option<PathBuf>,
    threads: Option<usize>,
    silent: bool,
    verbose: bool,
    cpu_only: bool,
    dry_run: bool,
    // Debug options
    no_tonecurve: bool,
    no_colormatrix: bool,
    inversion: Option<String>,
    auto_wb: bool,
    auto_wb_strength: f32,
    auto_wb_mode: String,
    tone_curve: Option<String>,
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
    // B&W conversion flag
    bw: bool,
) -> Result<(), String> {
    let batch_start = Instant::now();

    // Set verbose mode for core library
    invers_core::config::set_verbose(verbose);
    if verbose {
        invers_core::config::log_config_usage();
    }

    if inputs.is_empty() {
        return Err("No input files or directories specified".to_string());
    }

    // Expand directories to file lists
    let inputs = expand_inputs(&inputs, recursive)?;

    if inputs.is_empty() {
        return Err(
            "No supported image files found (supported: .tif, .tiff, .png, .dng)".to_string(),
        );
    }

    if !silent {
        println!("Found {} image files to process", inputs.len());
    }

    // Determine base estimation strategy description for messaging
    let strategy_message = if let Some(ref base_str) = base {
        let base_rgb = parse_base_rgb(base_str)?;
        format!(
            "Using manual base values: [{:.4}, {:.4}, {:.4}]",
            base_rgb[0], base_rgb[1], base_rgb[2]
        )
    } else if per_image {
        "Using per-image base estimation (each image analyzed separately)".to_string()
    } else if base_from.is_some() {
        "Using base estimation from file".to_string()
    } else {
        "Using same-roll mode: base will be estimated from first image".to_string()
    };

    if !silent {
        println!("{}", strategy_message);
    }

    // Handle dry-run mode
    if dry_run {
        println!("\n=== DRY RUN MODE ===");
        println!("Files that would be processed ({}):", inputs.len());
        for input in &inputs {
            println!("  {}", input.display());
        }
        println!("\nBase estimation strategy: {}", strategy_message);
        println!("No files were processed.");
        return Ok(());
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

    // Determine base estimation strategy
    // Priority: --base-from > --base > first-image (default) > --per-image
    let base_strategy = if let Some(base_path) = base_from {
        // Priority 1: Load from JSON file
        if !silent {
            println!("Loading base estimation from {}...", base_path.display());
        }
        let json = std::fs::read_to_string(&base_path)
            .map_err(|e| format!("Failed to read base estimation file: {}", e))?;
        let base_est: BaseEstimation = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse base estimation: {}", e))?;
        if !silent {
            println!(
                "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
                base_est.medians[0], base_est.medians[1], base_est.medians[2]
            );
        }
        BaseStrategy::FromFile(base_est)
    } else if let Some(base_str) = base {
        // Priority 2: Manual base values
        let base_rgb = parse_base_rgb(&base_str)?;
        if !silent {
            println!("Using manual base values...");
            println!(
                "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
                base_rgb[0], base_rgb[1], base_rgb[2]
            );
        }
        BaseStrategy::Manual(make_base_from_rgb(base_rgb))
    } else if per_image {
        // Priority 4: Per-image estimation (opt-in)
        if !silent {
            println!("Using per-image base estimation (each file estimated independently)");
        }
        BaseStrategy::PerImage
    } else {
        // Priority 3: First-image estimation (default for same-roll processing)
        if !silent {
            println!("Using same-roll mode: estimating base from first image");
        }
        BaseStrategy::FirstImage
    };

    // For non-per-image strategies, determine shared base before parallel processing
    let shared_base: Option<BaseEstimation> = match &base_strategy {
        BaseStrategy::FromFile(base) => Some(base.clone()),
        BaseStrategy::Manual(base) => Some(base.clone()),
        BaseStrategy::FirstImage => {
            // Decode first image and estimate base
            let first_input = inputs.first().ok_or("No input files for base estimation")?;

            if !silent {
                println!(
                    "Estimating base from first image: {}...",
                    first_input.display()
                );
            }

            let first_decoded = invers_core::decoders::decode_image(first_input)?;
            let estimation = invers_core::pipeline::estimate_base(
                &first_decoded,
                None,
                Some(invers_core::models::BaseEstimationMethod::Regions),
                None,
            )?;

            if !silent {
                println!(
                    "  Base (RGB): [{:.4}, {:.4}, {:.4}]",
                    estimation.medians[0], estimation.medians[1], estimation.medians[2]
                );
                println!(
                    "  This base will be shared across all {} images\n",
                    inputs.len()
                );
            }

            Some(estimation)
        }
        BaseStrategy::PerImage => None, // Each image will estimate its own
    };

    // Override inversion mode if --bw flag is set
    let effective_inversion = if bw {
        Some("bw".to_string())
    } else {
        inversion.clone()
    };

    // Build shared processing params (reused for all images)
    let params = ProcessingParams {
        export: export.clone(),
        exposure,
        cpu_only,
        silent: true, // Suppress per-image output in batch mode
        verbose,
        debug,
        white_balance,
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
        inversion: effective_inversion,
        auto_wb,
        auto_wb_strength,
        auto_wb_mode: auto_wb_mode.clone(),
        tone_curve: tone_curve.clone(),
    };

    if !silent {
        println!("\nProcessing {} files in parallel...\n", inputs.len());
    }

    // Progress tracking
    let processed_count = AtomicUsize::new(0);
    let total_files = inputs.len();
    let processing_start = Instant::now();
    // Track cumulative time for averaging (used for ETA calculation)
    let cumulative_time = Mutex::new(0.0f64);

    // Process files in parallel
    let results: Vec<Result<(PathBuf, f64), String>> = inputs
        .par_iter()
        .map(|input| {
            let file_start = Instant::now();

            // Decode image
            let decoded = invers_core::decoders::decode_image(input)?;

            // Determine base estimation for this image
            let base_estimation = if let Some(ref base) = shared_base {
                base.clone()
            } else {
                // Per-image estimation
                invers_core::pipeline::estimate_base(
                    &decoded,
                    None,
                    Some(invers_core::models::BaseEstimationMethod::Regions),
                    None,
                )?
            };

            // Build output path
            let output_path = determine_output_path(input, &out, &export)?;

            // Process using shared function
            process_single_image(
                decoded,
                base_estimation,
                None, // No film preset
                None, // No scan profile
                &output_path,
                &params,
            )?;

            let file_elapsed = file_start.elapsed().as_secs_f64();

            // Update progress
            let count = processed_count.fetch_add(1, Ordering::SeqCst) + 1;

            // Update cumulative time for ETA calculation
            {
                let mut cum = cumulative_time.lock().unwrap();
                *cum += file_elapsed;
            }

            if !silent {
                // Calculate estimated time remaining based on wall-clock time
                let wall_elapsed = processing_start.elapsed().as_secs_f64();
                let remaining_files = total_files - count;
                let avg_time_per_file = wall_elapsed / count as f64;
                let est_remaining_secs = avg_time_per_file * remaining_files as f64;

                // Format estimated remaining time
                let est_remaining_str = format_duration(est_remaining_secs);

                // Get just the filename for cleaner output
                let filename = input
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| input.display().to_string());

                println!(
                    "[{}/{}] processing {} - {:.1}s (est. {} remaining)",
                    count, total_files, filename, file_elapsed, est_remaining_str
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

/// Format a duration in seconds to a human-readable string.
///
/// Examples: "2.3s", "1m 30s", "4m 30s"
fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else {
        let minutes = (seconds / 60.0).floor() as u64;
        let remaining_secs = (seconds % 60.0).round() as u64;
        if remaining_secs == 0 {
            format!("{}m", minutes)
        } else {
            format!("{}m {}s", minutes, remaining_secs)
        }
    }
}
