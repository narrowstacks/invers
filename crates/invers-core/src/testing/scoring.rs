//! Scoring functions for parameter testing
//!
//! Contains functions for calculating test scores and quality metrics.

use crate::diagnostics::DiagnosticReport;

/// Calculate contrast ratio from diagnostic report
pub(crate) fn calculate_contrast_ratio(report: &DiagnosticReport) -> f32 {
    let our_contrast = report.our_stats.iter().map(|s| s.max - s.min).sum::<f32>() / 3.0;
    let tp_contrast = report
        .third_party_stats
        .iter()
        .map(|s| s.max - s.min)
        .sum::<f32>()
        / 3.0;

    if our_contrast > 0.0 {
        tp_contrast / our_contrast
    } else {
        1.0
    }
}

/// Calculate overall score from diagnostic report
/// Lower scores are better (closer to reference)
pub(crate) fn calculate_score(report: &DiagnosticReport) -> f32 {
    // Weight different metrics
    let mae_weight = 1.0;
    let exposure_weight = 2.0;
    let color_shift_weight = 2.0;
    let contrast_weight = 0.5;

    // Average MAE across channels
    let mae_avg = (report.difference_stats[0].mean
        + report.difference_stats[1].mean
        + report.difference_stats[2].mean)
        / 3.0;

    // Exposure error (deviation from 1.0)
    let exposure_error = (report.exposure_ratio - 1.0).abs();

    // Color shift magnitude
    let color_shift_mag = (report.color_shift[0].powi(2)
        + report.color_shift[1].powi(2)
        + report.color_shift[2].powi(2))
    .sqrt();

    // Contrast error
    let contrast_ratio = calculate_contrast_ratio(report);
    let contrast_error = (contrast_ratio - 1.0).abs();

    // Weighted sum
    mae_avg * mae_weight
        + exposure_error * exposure_weight
        + color_shift_mag * color_shift_weight
        + contrast_error * contrast_weight
}
