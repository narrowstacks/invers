//! Debug-only commands for diagnostics and parameter testing.
//!
//! This module is only compiled in debug builds.

use std::path::PathBuf;

use invers_cli::{build_convert_options_full_with_gpu, parse_roi};

#[allow(clippy::too_many_arguments)]
pub fn cmd_diagnose(
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
    let base_estimation =
        invers_core::pipeline::estimate_base(&decoded_original, roi_rect, None, None)?;
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
    let options = build_convert_options_full_with_gpu(
        original.clone(),
        std::path::PathBuf::from("."),
        "tiff16",
        "linear-rec2020".to_string(),
        Some(base_estimation),
        film_preset,
        None, // scan_profile
        no_tonecurve,
        no_colormatrix,
        exposure,
        None,  // inversion_mode
        false, // no_auto_levels
        false, // preserve_headroom
        false, // no_clip
        true,  // auto_wb
        1.0,   // auto_wb_strength
        debug,
        true, // use_gpu
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

#[allow(clippy::too_many_arguments)]
pub fn cmd_test_params(
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
