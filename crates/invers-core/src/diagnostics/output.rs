//! Report formatting and output functions

use std::path::Path;

use crate::decoders::DecodedImage;
use crate::exporters::export_tiff16;
use crate::pipeline::ProcessedImage;

use super::compare::create_difference_map;
use super::DiagnosticReport;

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
    print_diagnosis(report);

    println!("{}\n", "=".repeat(80));
}

/// Print diagnostic analysis section
fn print_diagnosis(report: &DiagnosticReport) {
    let channel_names = ["Red", "Green", "Blue"];

    println!("DIAGNOSIS");
    println!("{}\n", "-".repeat(80));

    // Check for exposure differences
    if (report.exposure_ratio - 1.0).abs() > 0.05 {
        println!("  * EXPOSURE DIFFERENCE:");
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
        println!("  * COLOR SHIFT DETECTED:");
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
        println!("  * CONTRAST DIFFERENCE:");
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
        export_as_grayscale: false,
    };

    // Save images
    export_tiff16(our_image, output_dir.join("our_conversion.tif"), None)?;

    let tp_as_processed = ProcessedImage {
        width: third_party.width,
        height: third_party.height,
        data: third_party.data.clone(),
        channels: third_party.channels,
        export_as_grayscale: false,
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
