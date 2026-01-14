//! Curves-based processing pipeline
//!
//! Implements a curve-based negative-to-positive conversion pipeline inspired by
//! Negative Lab Pro's algorithms. This pipeline uses per-channel histogram analysis,
//! multiple white balance methods, and sigmoid-based tone adjustments.
//!
//! Key features:
//! - Per-channel histogram analysis to find white/black points
//! - 5 different white balance application methods
//! - 3 white balance tonality modes
//! - Sigmoid contrast via tanh/atanh
//! - Shadow/highlight toning
//! - Configurable processing order (color-first vs tones-first)

mod histogram;
mod layers;
mod white_balance;

// Re-export public items
pub use histogram::analyze_histogram;
pub use white_balance::{
    analyze_wb_points, calculate_gamma_balance, calculate_linear_balance, calculate_wb_gamma,
    calculate_wb_offsets, calculate_wb_preset_offsets, AnalyzedWbPoints,
};

use crate::decoders::DecodedImage;
use crate::models::{CbHistogramAnalysis, CbLayerOrder, CbOptions, CbWbPreset, ConvertOptions};
use crate::pipeline::ProcessedImage;

// Note: compute_channel_histogram and find_clip_points are used internally by analyze_histogram
use layers::{
    apply_blacks_layer, apply_color_gamma_layer, apply_contrast_layer, apply_exposure_layer,
    apply_gamma_layer, apply_highlight_toning, apply_highlights_layer, apply_shadow_toning,
    apply_shadows_layer, apply_wb_pixel, apply_whites_layer,
};

// ============================================================
// Math Utilities
// ============================================================

/// Compute atanh (inverse hyperbolic tangent)
/// atanh(x) = 0.5 * ln((1+x)/(1-x))
#[inline]
fn atanh(x: f32) -> f32 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

/// Compute log base b of x: log_b(x) = ln(x) / ln(b)
#[inline]
pub(crate) fn logb(base: f32, x: f32) -> f32 {
    if base <= 0.0 || base == 1.0 || x <= 0.0 {
        return 1.0;
    }
    x.ln() / base.ln()
}

/// Clamp value to 0.0-1.0 range
#[inline]
pub(crate) fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ============================================================
// Sigmoid Contrast Functions (tanh-based)
// ============================================================

/// Core sigmoid function using tanh
/// sig(contrast, midpoint, x) = tanh(0.5 * contrast * (x - midpoint))
#[inline]
fn sigmoid_core(contrast: f32, midpoint: f32, x: f32) -> f32 {
    (0.5 * contrast * (x - midpoint)).tanh()
}

/// Apply sigmoidal contrast
/// Normalizes the tanh output to 0-1 range
pub(crate) fn apply_sigmoid(contrast: f32, midpoint: f32, x: f32, range: f32) -> f32 {
    let r = if range != 0.0 { range } else { 1.0 };
    let sig_0 = sigmoid_core(contrast, midpoint, 0.0);
    let sig_1 = sigmoid_core(contrast, midpoint, 1.0);
    let sig_x = sigmoid_core(contrast, midpoint, x / r);

    ((sig_x - sig_0) / (sig_1 - sig_0)) * r
}

/// Inverse sigmoidal contrast
pub(crate) fn apply_inverse_sigmoid(contrast: f32, midpoint: f32, x: f32, range: f32) -> f32 {
    let r = if range != 0.0 { range } else { 1.0 };
    let sig_0 = sigmoid_core(contrast, midpoint, 0.0);
    let sig_1 = sigmoid_core(contrast, midpoint, 1.0);

    // Calculate the argument for atanh
    let mut arg = (sig_1 - sig_0) * x / r + sig_0;

    // Clamp to valid atanh range (-1, 1)
    arg = arg.clamp(-0.9999, 0.9999);

    (midpoint + 2.0 / contrast * atanh(arg)) * r
}

// ============================================================
// Main Inversion Function
// ============================================================

/// Invert negative to positive using per-channel curve mapping
fn invert_with_curves(
    data: &mut [f32],
    channels: u8,
    analysis: &CbHistogramAnalysis,
    is_negative: bool,
) {
    let ch = channels as usize;

    // Calculate per-channel scale factors
    // For negatives: white_point is the brightest (film base), black_point is the darkest (exposed)
    // Range should be white_point - black_point (positive value)
    let r_range = (analysis.red.white_point - analysis.red.black_point).max(1.0);
    let g_range = (analysis.green.white_point - analysis.green.black_point).max(1.0);
    let b_range = (analysis.blue.white_point - analysis.blue.black_point).max(1.0);

    for pixel in data.chunks_exact_mut(ch) {
        // Convert to 0-255 scale, map through curve, back to 0-1
        let r_255 = pixel[0] * 255.0;
        let g_255 = pixel[1] * 255.0;
        let b_255 = pixel[2] * 255.0;

        // Normalize to 0-1 within the detected range
        // Maps [black_point, white_point] -> [0, 1]
        // black_point (darkest in negative = brightest in positive) maps to 0
        // white_point (brightest in negative = darkest in positive) maps to 1
        let r_norm = ((r_255 - analysis.red.black_point) / r_range).clamp(0.0, 1.0);
        let g_norm = ((g_255 - analysis.green.black_point) / g_range).clamp(0.0, 1.0);
        let b_norm = ((b_255 - analysis.blue.black_point) / b_range).clamp(0.0, 1.0);

        // Invert if negative
        // After normalization: 0 = darkest negative (brightest positive), 1 = brightest negative (darkest positive)
        // After inversion: 0 -> 1 (bright), 1 -> 0 (dark)
        if is_negative {
            pixel[0] = 1.0 - r_norm;
            pixel[1] = 1.0 - g_norm;
            pixel[2] = 1.0 - b_norm;
        } else {
            pixel[0] = r_norm;
            pixel[1] = g_norm;
            pixel[2] = b_norm;
        }
    }
}

// ============================================================
// Tonal Adjustments
// ============================================================

/// Apply all tonal adjustments (brightness, exposure, contrast, highlights, shadows, blacks, whites)
fn apply_tonal_adjustments(
    data: &mut [f32],
    ch: usize,
    cb: &CbOptions,
    brightness_gamma: f32,
    exposure_factor: f32,
    debug: bool,
) {
    // Exposure
    if cb.exposure != 0.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_exposure_layer(pixel, exposure_factor);
        }
    }

    // Brightness (gamma)
    if cb.brightness != 0.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_gamma_layer(pixel, brightness_gamma);
        }
    }

    // Contrast
    if cb.contrast.abs() >= 1.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_contrast_layer(pixel, cb.contrast);
        }

        if debug {
            let (min, max, mean) = compute_stats(data);
            eprintln!(
                "[CB] After contrast - min: {:.4}, max: {:.4}, mean: {:.4}",
                min, max, mean
            );
        }
    }

    // Highlights
    if cb.highlights.abs() >= 1.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_highlights_layer(pixel, cb.highlights);
        }
    }

    // Shadows
    if cb.shadows.abs() > 0.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_shadows_layer(pixel, cb.shadows);
        }
    }

    // Blacks
    if cb.blacks.abs() >= 1.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_blacks_layer(pixel, cb.blacks, cb.shadow_range);
        }
    }

    // Whites
    if cb.whites.abs() >= 1.0 {
        for pixel in data.chunks_exact_mut(ch) {
            apply_whites_layer(pixel, cb.whites, cb.highlight_range);
        }
    }
}

// ============================================================
// Main Pipeline Entry Point
// ============================================================

/// Process image using CB-style (curves-based) pipeline
///
/// # Arguments
/// * `image` - Decoded input image
/// * `options` - Conversion options (must have pipeline_mode = CbStyle)
///
/// # Returns
/// * `ProcessedImage` - Processed positive image
pub fn process_image_cb(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    // Get CB options (use defaults if not provided)
    let cb = options.cb_options.clone().unwrap_or_default();

    // Extract image data
    let DecodedImage {
        width,
        height,
        channels,
        mut data,
        source_is_grayscale,
        is_monochrome,
        ..
    } = image;

    let export_as_grayscale = source_is_grayscale || is_monochrome;

    if channels != 3 {
        return Err(format!(
            "CB pipeline requires 3-channel RGB, got {}",
            channels
        ));
    }

    if options.debug {
        eprintln!("[CB] Starting Curves-based pipeline processing");
        eprintln!(
            "[CB] Image size: {}x{}, {} channels",
            width, height, channels
        );
    }

    // Optional: normalize film base if we have an estimation
    if let Some(base_estimation) = options.base_estimation.as_ref() {
        if options.debug {
            eprintln!("[CB] Applying film base white balance");
        }
        crate::pipeline::apply_film_base_white_balance(&mut data, base_estimation, options)?;
    } else if options.debug {
        eprintln!("[CB] Film base WB skipped (no base estimation)");
    }

    // Step 1: Histogram analysis to find per-channel white/black points
    let analysis = analyze_histogram(&data, channels, cb.white_threshold, cb.black_threshold);

    if options.debug {
        eprintln!(
            "[CB] Histogram analysis - R: [{:.1}-{:.1}], G: [{:.1}-{:.1}], B: [{:.1}-{:.1}]",
            analysis.red.white_point,
            analysis.red.black_point,
            analysis.green.white_point,
            analysis.green.black_point,
            analysis.blue.white_point,
            analysis.blue.black_point
        );
        eprintln!(
            "[CB] Mean points - R: {:.3}, G: {:.3}, B: {:.3}",
            analysis.red.mean_point, analysis.green.mean_point, analysis.blue.mean_point
        );
    }

    // Step 2: Invert using per-channel curves
    // For negatives, we invert; for positives, we just normalize
    let is_negative = true; // TODO: make this configurable
    invert_with_curves(&mut data, channels, &analysis, is_negative);

    if options.debug {
        let (min, max, mean) = compute_stats(&data);
        eprintln!(
            "[CB] After inversion - min: {:.4}, max: {:.4}, mean: {:.4}",
            min, max, mean
        );
    }

    // Step 2b: Apply Curves-based white balance based on preset selection
    // This analyzes the positive image and applies balance based on the selected preset
    if cb.wb_preset != CbWbPreset::None {
        // Analyze the positive image to get WB points (neutral, warm, cool)
        let wb_points = analyze_wb_points(&data, channels);

        if options.debug {
            eprintln!(
                "[CB] WB points - neutral: [{:.3}, {:.3}, {:.3}]",
                wb_points.neutral[0], wb_points.neutral[1], wb_points.neutral[2]
            );
            eprintln!(
                "[CB] WB points - warm: [{:.3}, {:.3}, {:.3}]",
                wb_points.warm[0], wb_points.warm[1], wb_points.warm[2]
            );
            eprintln!(
                "[CB] WB points - cool: [{:.3}, {:.3}, {:.3}]",
                wb_points.cool[0], wb_points.cool[1], wb_points.cool[2]
            );
        }

        // Calculate WB offsets based on preset
        let auto_wb_offsets =
            calculate_wb_preset_offsets(cb.wb_preset, &wb_points, cb.film_character);

        if options.debug {
            eprintln!(
                "[CB] WB preset: {:?}, offsets - R: {:.3}, G: {:.3}, B: {:.3}",
                cb.wb_preset, auto_wb_offsets[0], auto_wb_offsets[1], auto_wb_offsets[2]
            );
        }

        // Apply WB offsets with strength control
        let strength = options.auto_wb_strength;
        let ch = channels as usize;
        for pixel in data.chunks_exact_mut(ch) {
            for (i, value) in pixel.iter_mut().enumerate() {
                // Scale offset from 0-255 range to 0-1 range
                let offset = auto_wb_offsets[i] / 255.0 * strength;
                *value = (*value + offset).clamp(0.0, 1.0);
            }
        }

        if options.debug {
            let (min, max, mean) = compute_stats(&data);
            eprintln!(
                "[CB] After WB ({:?}) - min: {:.4}, max: {:.4}, mean: {:.4}",
                cb.wb_preset, min, max, mean
            );
        }
    } else if options.enable_auto_wb {
        use crate::models::AutoWbMode;

        let multipliers = match options.auto_wb_mode {
            AutoWbMode::GrayPixel => crate::auto_adjust::auto_white_balance(
                &mut data,
                channels,
                options.auto_wb_strength,
            ),
            AutoWbMode::Average => crate::auto_adjust::auto_white_balance_avg(
                &mut data,
                channels,
                options.auto_wb_strength,
            ),
            AutoWbMode::Percentile => crate::auto_adjust::auto_white_balance_percentile(
                &mut data,
                channels,
                options.auto_wb_strength,
                98.0,
            ),
        };

        if options.debug {
            eprintln!(
                "[CB] After auto-WB ({:?}) - multipliers: R={:.4}, G={:.4}, B={:.4}",
                options.auto_wb_mode, multipliers[0], multipliers[1], multipliers[2]
            );
            let (min, max, mean) = compute_stats(&data);
            eprintln!("[CB]   min: {:.4}, max: {:.4}, mean: {:.4}", min, max, mean);
        }
    }

    // Calculate derived values for tone adjustments
    let brightness_gamma = 1.0 / (1.0 + cb.brightness * 0.02);
    let exposure_factor = 1.0 / (1.0 + cb.exposure * 0.02);
    let color_offsets = [
        1.0 - cb.cyan * 0.01,
        1.0 - cb.tint * 0.01,
        1.0 - cb.temp * 0.01,
    ];

    // Calculate WB parameters
    let wb_offsets = calculate_wb_offsets(cb.wb_temp, cb.wb_tint, cb.wb_tonality);
    let wb_gamma = calculate_wb_gamma(cb.wb_temp, cb.wb_tint, cb.wb_tonality);

    // Step 3: Apply processing layers based on layer order
    let ch = channels as usize;

    match cb.layer_order {
        CbLayerOrder::ColorFirst => {
            // Apply color/WB first
            if cb.wb_temp != 0.0 || cb.wb_tint != 0.0 {
                for pixel in data.chunks_exact_mut(ch) {
                    apply_wb_pixel(pixel, &wb_offsets, &wb_gamma, cb.wb_method);
                }

                if options.debug {
                    let (min, max, mean) = compute_stats(&data);
                    eprintln!(
                        "[CB] After WB (color first) - min: {:.4}, max: {:.4}, mean: {:.4}",
                        min, max, mean
                    );
                }
            }

            // Apply color gamma
            for pixel in data.chunks_exact_mut(ch) {
                apply_color_gamma_layer(pixel, color_offsets);
            }

            // Apply tonal adjustments
            apply_tonal_adjustments(
                &mut data,
                ch,
                &cb,
                brightness_gamma,
                exposure_factor,
                options.debug,
            );
        }
        CbLayerOrder::TonesFirst => {
            // Apply tonal adjustments first
            apply_tonal_adjustments(
                &mut data,
                ch,
                &cb,
                brightness_gamma,
                exposure_factor,
                options.debug,
            );

            // Then apply color/WB
            if cb.wb_temp != 0.0 || cb.wb_tint != 0.0 {
                for pixel in data.chunks_exact_mut(ch) {
                    apply_wb_pixel(pixel, &wb_offsets, &wb_gamma, cb.wb_method);
                }

                if options.debug {
                    let (min, max, mean) = compute_stats(&data);
                    eprintln!(
                        "[CB] After WB (tones first) - min: {:.4}, max: {:.4}, mean: {:.4}",
                        min, max, mean
                    );
                }
            }

            // Apply color gamma
            for pixel in data.chunks_exact_mut(ch) {
                apply_color_gamma_layer(pixel, color_offsets);
            }
        }
    }

    // Step 4: Apply shadow/highlight toning
    let shadow_colors = [cb.shadow_cyan, cb.shadow_tint, cb.shadow_temp];
    let highlight_colors = [cb.highlight_cyan, cb.highlight_tint, cb.highlight_temp];

    if shadow_colors.iter().any(|&c| c != 0.0) {
        for pixel in data.chunks_exact_mut(ch) {
            apply_shadow_toning(pixel, shadow_colors, cb.shadow_range);
        }

        if options.debug {
            let (min, max, mean) = compute_stats(&data);
            eprintln!(
                "[CB] After shadow toning - min: {:.4}, max: {:.4}, mean: {:.4}",
                min, max, mean
            );
        }
    }

    if highlight_colors.iter().any(|&c| c != 0.0) {
        for pixel in data.chunks_exact_mut(ch) {
            apply_highlight_toning(pixel, highlight_colors, cb.highlight_range);
        }

        if options.debug {
            let (min, max, mean) = compute_stats(&data);
            eprintln!(
                "[CB] After highlight toning - min: {:.4}, max: {:.4}, mean: {:.4}",
                min, max, mean
            );
        }
    }

    // Step 5: Final clamp to valid range
    for value in data.iter_mut() {
        *value = value.clamp(0.0, 1.0);
    }

    if options.debug {
        let (min, max, mean) = compute_stats(&data);
        eprintln!(
            "[CB] Final output - min: {:.4}, max: {:.4}, mean: {:.4}",
            min, max, mean
        );
    }

    Ok(ProcessedImage {
        width,
        height,
        data,
        channels,
        export_as_grayscale,
    })
}

// ============================================================
// Debug Utilities
// ============================================================

/// Compute basic statistics for debugging
fn compute_stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0;

    for &v in data {
        min = min.min(v);
        max = max.max(v);
        sum += v;
    }

    (min, max, sum / data.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_contrast() {
        // Test that sigmoid at midpoint returns midpoint
        let result = apply_sigmoid(2.0, 0.5, 0.5, 1.0);
        assert!((result - 0.5).abs() < 0.01);

        // Test that sigmoid increases contrast
        let low = apply_sigmoid(2.0, 0.5, 0.3, 1.0);
        let high = apply_sigmoid(2.0, 0.5, 0.7, 1.0);
        assert!(low < 0.3);
        assert!(high > 0.7);
    }

    #[test]
    fn test_inverse_sigmoid() {
        // Test that inverse sigmoid reverses sigmoid
        let original = 0.3;
        let sigmoid_result = apply_sigmoid(2.0, 0.5, original, 1.0);
        let recovered = apply_inverse_sigmoid(2.0, 0.5, sigmoid_result, 1.0);
        assert!((recovered - original).abs() < 0.01);
    }

    #[test]
    fn test_clamp01() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(1.5), 1.0);
    }

    #[test]
    fn test_logb() {
        // log_2(8) = 3
        assert!((logb(2.0, 8.0) - 3.0).abs() < 0.001);
        // log_10(100) = 2
        assert!((logb(10.0, 100.0) - 2.0).abs() < 0.001);
        // Edge cases
        assert_eq!(logb(0.0, 10.0), 1.0);
        assert_eq!(logb(1.0, 10.0), 1.0);
        assert_eq!(logb(10.0, 0.0), 1.0);
    }

    #[test]
    fn test_compute_stats() {
        let data = vec![0.0, 0.5, 1.0, 0.25, 0.75];
        let (min, max, mean) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 1.0);
        assert!((mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_empty() {
        let data: Vec<f32> = vec![];
        let (min, max, mean) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
    }
}
