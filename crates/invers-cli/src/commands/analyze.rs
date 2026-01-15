use serde::Serialize;
use std::path::PathBuf;

use invers_cli::parse_roi;

/// Analysis result structure for JSON output.
///
/// Contains all metadata and analysis results for a single image,
/// serializable to JSON for machine-readable output.
#[derive(Serialize)]
pub struct AnalysisResult {
    pub file: String,
    pub dimensions: [u32; 2],
    pub channels: u8,
    pub base_estimation: BaseEstimationResult,
    pub channel_stats: ChannelStats,
}

/// Film base estimation result for JSON output.
#[derive(Serialize)]
pub struct BaseEstimationResult {
    pub method: String,
    pub medians: [f32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_stats: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roi: Option<(u32, u32, u32, u32)>,
}

/// Per-channel (RGB) statistics for an image.
#[derive(Serialize)]
pub struct ChannelStats {
    pub red: ChannelStat,
    pub green: ChannelStat,
    pub blue: ChannelStat,
}

/// Statistics for a single color channel.
#[derive(Serialize)]
pub struct ChannelStat {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

/// Compute min, max, and mean for each RGB channel in a decoded image.
pub fn compute_channel_stats(decoded: &invers_core::decoders::DecodedImage) -> ChannelStats {
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

/// Execute the analyze command to inspect an image and estimate film base color.
///
/// Analyzes a negative image to determine base RGB values that can be reused
/// across multiple frames from the same roll of film. Output can be displayed
/// as human-readable text or saved as JSON for batch processing.
pub fn cmd_analyze(
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
        _ => invers_core::models::BaseEstimationMethod::Regions,
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
