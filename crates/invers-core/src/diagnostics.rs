//! Diagnostic tools for comparing image conversions
//!
//! Provides comprehensive comparison between our conversion pipeline
//! and third-party software conversions.

use crate::decoders::DecodedImage;
use crate::exporters::export_tiff16;
use crate::pipeline::ProcessedImage;
use std::path::Path;

/// Statistics for a single channel
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub percentiles: Vec<(u8, f32)>, // (percentile, value)
}

/// Histogram data for a channel (256 bins)
#[derive(Debug, Clone)]
pub struct Histogram {
    pub bins: Vec<u32>,
    pub bin_edges: Vec<f32>,
}

/// Complete diagnostic comparison result
#[derive(Debug)]
pub struct DiagnosticReport {
    pub our_stats: [ChannelStats; 3],
    pub third_party_stats: [ChannelStats; 3],
    pub difference_stats: [ChannelStats; 3],
    pub our_histograms: [Histogram; 3],
    pub third_party_histograms: [Histogram; 3],
    pub color_shift: [f32; 3], // Average offset per channel
    pub exposure_ratio: f32,   // Overall brightness ratio (third_party / ours)
}

/// Compute comprehensive statistics for an image
pub fn compute_statistics(data: &[f32], channels: u8) -> [ChannelStats; 3] {
    if channels != 3 {
        panic!("Only 3-channel RGB images supported");
    }

    let mut channel_data: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // Separate channels
    for pixel in data.chunks_exact(3) {
        channel_data[0].push(pixel[0]);
        channel_data[1].push(pixel[1]);
        channel_data[2].push(pixel[2]);
    }

    // Compute stats for each channel
    [
        compute_channel_stats(&channel_data[0]),
        compute_channel_stats(&channel_data[1]),
        compute_channel_stats(&channel_data[2]),
    ]
}

/// Compute statistics for a single channel
fn compute_channel_stats(data: &[f32]) -> ChannelStats {
    if data.is_empty() {
        return ChannelStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            percentiles: vec![],
        };
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let median = sorted[sorted.len() / 2];

    // Compute standard deviation
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();

    // Compute percentiles
    let percentiles = vec![1, 5, 25, 50, 75, 95, 99]
        .into_iter()
        .map(|p| {
            let idx = ((p as f32 / 100.0) * (sorted.len() - 1) as f32).round() as usize;
            (p, sorted[idx])
        })
        .collect();

    ChannelStats {
        min,
        max,
        mean,
        median,
        std_dev,
        percentiles,
    }
}

/// Generate histogram for an image
pub fn compute_histograms(data: &[f32], channels: u8, num_bins: usize) -> [Histogram; 3] {
    if channels != 3 {
        panic!("Only 3-channel RGB images supported");
    }

    let mut channel_data: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // Separate channels
    for pixel in data.chunks_exact(3) {
        channel_data[0].push(pixel[0]);
        channel_data[1].push(pixel[1]);
        channel_data[2].push(pixel[2]);
    }

    [
        compute_channel_histogram(&channel_data[0], num_bins),
        compute_channel_histogram(&channel_data[1], num_bins),
        compute_channel_histogram(&channel_data[2], num_bins),
    ]
}

/// Generate histogram for a single channel
fn compute_channel_histogram(data: &[f32], num_bins: usize) -> Histogram {
    let mut bins = vec![0u32; num_bins];
    let bin_edges: Vec<f32> = (0..=num_bins).map(|i| i as f32 / num_bins as f32).collect();

    for &value in data {
        let clamped = value.clamp(0.0, 1.0);
        let bin_idx = ((clamped * (num_bins - 1) as f32) as usize).min(num_bins - 1);
        bins[bin_idx] += 1;
    }

    Histogram { bins, bin_edges }
}

/// Create a difference map between two images
/// Returns absolute differences as a new image
pub fn create_difference_map(
    img1: &[f32],
    img2: &[f32],
    _width: u32,
    _height: u32,
) -> Result<Vec<f32>, String> {
    if img1.len() != img2.len() {
        return Err(format!(
            "Image size mismatch: {} vs {}",
            img1.len(),
            img2.len()
        ));
    }

    let diff: Vec<f32> = img1
        .iter()
        .zip(img2.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    Ok(diff)
}

/// Extract sample patches from different brightness regions
/// Returns (shadow_patch, midtone_patch, highlight_patch) as flat RGB arrays
pub fn extract_sample_patches(
    data: &[f32],
    width: u32,
    height: u32,
    patch_size: u32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    // Find representative regions
    let shadow_center = find_representative_pixel(data, width, height, 0.1, 0.3)?;
    let midtone_center = find_representative_pixel(data, width, height, 0.4, 0.6)?;
    let highlight_center = find_representative_pixel(data, width, height, 0.7, 0.9)?;

    let shadow_patch = extract_patch(data, width, height, shadow_center, patch_size)?;
    let midtone_patch = extract_patch(data, width, height, midtone_center, patch_size)?;
    let highlight_patch = extract_patch(data, width, height, highlight_center, patch_size)?;

    Ok((shadow_patch, midtone_patch, highlight_patch))
}

/// Find a representative pixel in a brightness range
fn find_representative_pixel(
    data: &[f32],
    width: u32,
    height: u32,
    min_brightness: f32,
    max_brightness: f32,
) -> Result<(u32, u32), String> {
    // Sample a grid to find suitable region
    let step = 50;
    let mut candidates = Vec::new();

    for y in (step..height.saturating_sub(step)).step_by(step as usize) {
        for x in (step..width.saturating_sub(step)).step_by(step as usize) {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < data.len() {
                let brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3.0;
                if brightness >= min_brightness && brightness <= max_brightness {
                    candidates.push((x, y, brightness));
                }
            }
        }
    }

    if candidates.is_empty() {
        // Fallback to center of image
        return Ok((width / 2, height / 2));
    }

    // Sort by closeness to middle of range
    let target = (min_brightness + max_brightness) / 2.0;
    candidates.sort_by(|a, b| {
        let dist_a = (a.2 - target).abs();
        let dist_b = (b.2 - target).abs();
        dist_a.partial_cmp(&dist_b).unwrap()
    });

    Ok((candidates[0].0, candidates[0].1))
}

/// Extract a square patch centered at (x, y)
fn extract_patch(
    data: &[f32],
    width: u32,
    height: u32,
    center: (u32, u32),
    patch_size: u32,
) -> Result<Vec<f32>, String> {
    let half_size = patch_size / 2;
    let (cx, cy) = center;

    let x_start = cx.saturating_sub(half_size);
    let y_start = cy.saturating_sub(half_size);
    let x_end = (cx + half_size).min(width);
    let y_end = (cy + half_size).min(height);

    let mut patch = Vec::new();

    for y in y_start..y_end {
        for x in x_start..x_end {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < data.len() {
                patch.push(data[idx]);
                patch.push(data[idx + 1]);
                patch.push(data[idx + 2]);
            }
        }
    }

    Ok(patch)
}

/// Perform comprehensive diagnostic comparison
pub fn compare_conversions(
    our_image: &ProcessedImage,
    third_party: &DecodedImage,
) -> Result<DiagnosticReport, String> {
    // Ensure dimensions match
    if our_image.width != third_party.width || our_image.height != third_party.height {
        return Err(format!(
            "Image dimensions mismatch: ours {}x{}, third-party {}x{}",
            our_image.width, our_image.height, third_party.width, third_party.height
        ));
    }

    // Compute statistics
    let our_stats = compute_statistics(&our_image.data, our_image.channels);
    let third_party_stats = compute_statistics(&third_party.data, third_party.channels);

    // Compute difference statistics
    let diff_data = create_difference_map(
        &our_image.data,
        &third_party.data,
        our_image.width,
        our_image.height,
    )?;
    let difference_stats = compute_statistics(&diff_data, 3);

    // Generate histograms
    let our_histograms = compute_histograms(&our_image.data, our_image.channels, 256);
    let third_party_histograms = compute_histograms(&third_party.data, third_party.channels, 256);

    // Compute color shift (average difference per channel)
    let mut color_shift = [0.0f32; 3];
    for i in 0..3 {
        color_shift[i] = third_party_stats[i].mean - our_stats[i].mean;
    }

    // Compute overall exposure ratio
    let our_avg_brightness = (our_stats[0].mean + our_stats[1].mean + our_stats[2].mean) / 3.0;
    let third_party_avg_brightness =
        (third_party_stats[0].mean + third_party_stats[1].mean + third_party_stats[2].mean) / 3.0;
    let exposure_ratio = if our_avg_brightness > 0.0001 {
        third_party_avg_brightness / our_avg_brightness
    } else {
        1.0
    };

    Ok(DiagnosticReport {
        our_stats,
        third_party_stats,
        difference_stats,
        our_histograms,
        third_party_histograms,
        color_shift,
        exposure_ratio,
    })
}

/// Print diagnostic report to console
pub fn print_report(report: &DiagnosticReport) {
    println!("\n{}\n", "=".repeat(80));
    println!("DIAGNOSTIC COMPARISON REPORT");
    println!("{}\n", "=".repeat(80));

    let channel_names = ["Red", "Green", "Blue"];

    // Statistics comparison
    println!("STATISTICS COMPARISON");
    println!("{}\n", "-".repeat(80));

    for (i, name) in channel_names.iter().enumerate() {
        println!("{} Channel:", name);
        println!("                    Ours        Third-Party   Difference");
        println!(
            "  Min:            {:<12.6} {:<12.6} {:<12.6}",
            report.our_stats[i].min,
            report.third_party_stats[i].min,
            report.difference_stats[i].min
        );
        println!(
            "  Max:            {:<12.6} {:<12.6} {:<12.6}",
            report.our_stats[i].max,
            report.third_party_stats[i].max,
            report.difference_stats[i].max
        );
        println!(
            "  Mean:           {:<12.6} {:<12.6} {:<12.6}",
            report.our_stats[i].mean,
            report.third_party_stats[i].mean,
            report.difference_stats[i].mean
        );
        println!(
            "  Median:         {:<12.6} {:<12.6} {:<12.6}",
            report.our_stats[i].median,
            report.third_party_stats[i].median,
            report.difference_stats[i].median
        );
        println!(
            "  Std Dev:        {:<12.6} {:<12.6}",
            report.our_stats[i].std_dev, report.third_party_stats[i].std_dev
        );
        println!();
    }

    // Percentiles
    println!("PERCENTILE COMPARISON");
    println!("{}\n", "-".repeat(80));

    for (i, name) in channel_names.iter().enumerate() {
        println!("{} Channel:", name);
        println!("  Percentile      Ours        Third-Party   Difference");
        for (p, our_val) in &report.our_stats[i].percentiles {
            let tp_val = report.third_party_stats[i]
                .percentiles
                .iter()
                .find(|(tp_p, _)| tp_p == p)
                .map(|(_, v)| v)
                .unwrap_or(&0.0);
            println!(
                "  {:3}%:           {:<12.6} {:<12.6} {:+.6}",
                p,
                our_val,
                tp_val,
                tp_val - our_val
            );
        }
        println!();
    }

    // Overall analysis
    println!("OVERALL ANALYSIS");
    println!("{}\n", "-".repeat(80));
    println!(
        "  Color Shift (RGB):      [{:+.6}, {:+.6}, {:+.6}]",
        report.color_shift[0], report.color_shift[1], report.color_shift[2]
    );
    println!("  Exposure Ratio:         {:.4}x", report.exposure_ratio);
    println!("  Mean Absolute Error:");
    println!(
        "    Red:                  {:.6}",
        report.difference_stats[0].mean
    );
    println!(
        "    Green:                {:.6}",
        report.difference_stats[1].mean
    );
    println!(
        "    Blue:                 {:.6}",
        report.difference_stats[2].mean
    );
    println!();

    // Diagnosis
    println!("DIAGNOSIS");
    println!("{}\n", "-".repeat(80));

    // Check for exposure differences
    if (report.exposure_ratio - 1.0).abs() > 0.05 {
        println!("  ⚠ EXPOSURE DIFFERENCE:");
        if report.exposure_ratio > 1.0 {
            println!(
                "    Third-party conversion is {:.1}% BRIGHTER than ours",
                (report.exposure_ratio - 1.0) * 100.0
            );
        } else {
            println!(
                "    Third-party conversion is {:.1}% DARKER than ours",
                (1.0 - report.exposure_ratio) * 100.0
            );
        }
        println!();
    }

    // Check for color shifts
    let max_color_shift = report
        .color_shift
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max_color_shift > 0.02 {
        println!("  ⚠ COLOR SHIFT DETECTED:");
        for (i, name) in channel_names.iter().enumerate() {
            if report.color_shift[i].abs() > 0.02 {
                if report.color_shift[i] > 0.0 {
                    println!(
                        "    {} channel: +{:.3} (third-party has MORE {})",
                        name,
                        report.color_shift[i],
                        name.to_lowercase()
                    );
                } else {
                    println!(
                        "    {} channel: {:.3} (third-party has LESS {})",
                        name,
                        report.color_shift[i],
                        name.to_lowercase()
                    );
                }
            }
        }
        println!();
    }

    // Check for tone curve differences
    let our_contrast = report.our_stats.iter().map(|s| s.max - s.min).sum::<f32>() / 3.0;
    let tp_contrast = report
        .third_party_stats
        .iter()
        .map(|s| s.max - s.min)
        .sum::<f32>()
        / 3.0;
    let contrast_ratio = tp_contrast / our_contrast;

    if (contrast_ratio - 1.0).abs() > 0.05 {
        println!("  ⚠ CONTRAST DIFFERENCE:");
        if contrast_ratio > 1.0 {
            println!(
                "    Third-party has {:.1}% MORE contrast",
                (contrast_ratio - 1.0) * 100.0
            );
        } else {
            println!(
                "    Third-party has {:.1}% LESS contrast",
                (1.0 - contrast_ratio) * 100.0
            );
        }
        println!("    This suggests different tone curves are being applied.");
        println!();
    }

    println!("{}\n", "=".repeat(80));
}

/// Save diagnostic visualizations
pub fn save_diagnostic_images<P: AsRef<Path>>(
    our_image: &ProcessedImage,
    third_party: &DecodedImage,
    output_dir: P,
) -> Result<(), String> {
    let output_dir = output_dir.as_ref();

    // Create difference map
    let diff_data = create_difference_map(
        &our_image.data,
        &third_party.data,
        our_image.width,
        our_image.height,
    )?;

    // Scale difference map for visibility (multiply by 5 to make small differences visible)
    let scaled_diff: Vec<f32> = diff_data.iter().map(|&x| (x * 5.0).min(1.0)).collect();

    let diff_image = ProcessedImage {
        width: our_image.width,
        height: our_image.height,
        data: scaled_diff,
        channels: 3,
    };

    // Save images
    export_tiff16(our_image, output_dir.join("our_conversion.tif"), None)?;

    let tp_as_processed = ProcessedImage {
        width: third_party.width,
        height: third_party.height,
        data: third_party.data.clone(),
        channels: third_party.channels,
    };
    export_tiff16(
        &tp_as_processed,
        output_dir.join("third_party_conversion.tif"),
        None,
    )?;

    export_tiff16(&diff_image, output_dir.join("difference_map_5x.tif"), None)?;

    println!("Diagnostic images saved to:");
    println!("  - {}", output_dir.join("our_conversion.tif").display());
    println!(
        "  - {}",
        output_dir.join("third_party_conversion.tif").display()
    );
    println!("  - {}", output_dir.join("difference_map_5x.tif").display());

    Ok(())
}
