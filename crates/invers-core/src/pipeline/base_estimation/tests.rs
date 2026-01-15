//! Tests for film base estimation

use super::analysis::{
    compute_channel_medians_from_brightest, compute_median, compute_noise_stats, is_likely_bw,
    validate_base_candidate,
};
use super::estimate_base_from_histogram;
use super::extraction::{
    extract_border_pixels, extract_roi_pixels, filter_valid_base_pixels, sample_region_brightness,
};
use super::methods::estimate_base;
use super::BaseRoiCandidate;
use crate::decoders::DecodedImage;
use crate::models::BaseEstimationMethod;

/// Create a test image with uniform color
fn create_uniform_image(width: u32, height: u32, color: [f32; 3]) -> DecodedImage {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        data.extend_from_slice(&color);
    }
    DecodedImage {
        width,
        height,
        data,
        channels: 3,
        black_level: None,
        white_level: None,
        color_matrix: None,
        source_is_grayscale: false,
        is_monochrome: false,
    }
}

/// Create a test image with a gradient from one color to another
fn create_gradient_image(width: u32, height: u32, start: [f32; 3], end: [f32; 3]) -> DecodedImage {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        let t = i as f32 / (pixel_count - 1) as f32;
        data.push(start[0] + t * (end[0] - start[0]));
        data.push(start[1] + t * (end[1] - start[1]));
        data.push(start[2] + t * (end[2] - start[2]));
    }
    DecodedImage {
        width,
        height,
        data,
        channels: 3,
        black_level: None,
        white_level: None,
        color_matrix: None,
        source_is_grayscale: false,
        is_monochrome: false,
    }
}

/// Create a test image with a specific border color and center color
fn create_border_image(
    width: u32,
    height: u32,
    border_color: [f32; 3],
    center_color: [f32; 3],
    border_size: u32,
) -> DecodedImage {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * 3);

    for y in 0..height {
        for x in 0..width {
            let is_border = x < border_size
                || x >= width - border_size
                || y < border_size
                || y >= height - border_size;
            let color = if is_border {
                border_color
            } else {
                center_color
            };
            data.extend_from_slice(&color);
        }
    }

    DecodedImage {
        width,
        height,
        data,
        channels: 3,
        black_level: None,
        white_level: None,
        color_matrix: None,
        source_is_grayscale: false,
        is_monochrome: false,
    }
}

// ========================================================================
// BaseRoiCandidate Tests
// ========================================================================

#[test]
fn test_base_roi_candidate_new() {
    let candidate = BaseRoiCandidate::new((10, 20, 100, 50), 0.75, "test");
    assert_eq!(candidate.rect, (10, 20, 100, 50));
    assert!((candidate.brightness - 0.75).abs() < 0.001);
    assert_eq!(candidate.label, "test");
}

#[test]
fn test_base_roi_candidate_from_manual_roi() {
    // Create a uniform orange image (typical film base color)
    let image = create_uniform_image(100, 100, [0.8, 0.6, 0.3]);
    let candidate = BaseRoiCandidate::from_manual_roi(&image, (10, 10, 50, 50));

    assert_eq!(candidate.rect, (10, 10, 50, 50));
    assert_eq!(candidate.label, "manual");
    // Brightness should be weighted average: 0.5*0.8 + 0.4*0.6 + 0.1*0.3 = 0.67
    assert!(
        (candidate.brightness - 0.67).abs() < 0.01,
        "Expected brightness ~0.67, got {}",
        candidate.brightness
    );
}

// ========================================================================
// estimate_base Tests
// ========================================================================

#[test]
fn test_estimate_base_with_manual_roi() {
    // Orange film base color
    let image = create_uniform_image(200, 200, [0.7, 0.5, 0.3]);
    let result = estimate_base(&image, Some((50, 50, 100, 100)), None, None);

    assert!(result.is_ok());
    let base = result.unwrap();
    assert!(!base.auto_estimated); // Manual ROI means not auto-estimated
    assert_eq!(base.roi, Some((50, 50, 100, 100)));

    // Medians should be close to the uniform color
    assert!(
        (base.medians[0] - 0.7).abs() < 0.05,
        "R median should be ~0.7, got {}",
        base.medians[0]
    );
}

#[test]
fn test_estimate_base_manual_roi_out_of_bounds() {
    let image = create_uniform_image(100, 100, [0.7, 0.5, 0.3]);
    // ROI extends beyond image bounds
    let result = estimate_base(&image, Some((50, 50, 100, 100)), None, None);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("out of bounds"));
}

#[test]
fn test_estimate_base_regions_method() {
    // Create image with bright orange border (film base) and dark center
    let image = create_border_image(400, 400, [0.8, 0.6, 0.3], [0.2, 0.15, 0.1], 50);

    let result = estimate_base(&image, None, Some(BaseEstimationMethod::Regions), None);

    assert!(result.is_ok());
    let base = result.unwrap();
    assert!(base.auto_estimated);

    // Should detect the orange border as film base
    assert!(
        base.medians[0] > 0.5,
        "R should be high (orange base), got {}",
        base.medians[0]
    );
}

#[test]
fn test_estimate_base_border_method() {
    let image = create_border_image(400, 400, [0.8, 0.6, 0.3], [0.2, 0.15, 0.1], 40);

    let result = estimate_base(&image, None, Some(BaseEstimationMethod::Border), Some(10.0));

    assert!(result.is_ok());
    let base = result.unwrap();
    assert!(base.auto_estimated);
    assert!(base.roi.is_none()); // Border method doesn't set specific ROI
}

#[test]
fn test_estimate_base_histogram_method() {
    // Create gradient image with peak in bright orange region
    let image = create_gradient_image(200, 200, [0.3, 0.2, 0.1], [0.9, 0.7, 0.4]);

    let result = estimate_base(&image, None, Some(BaseEstimationMethod::Histogram), None);

    assert!(result.is_ok());
    let base = result.unwrap();
    assert!(base.auto_estimated);
    assert!(base.roi.is_none());
    assert!(base.noise_stats.is_none()); // Histogram method doesn't compute noise
}

// ========================================================================
// estimate_base_from_histogram Tests
// ========================================================================

#[test]
fn test_histogram_estimation_typical_negative() {
    // Typical color negative: peaks in 0.3-0.9 range with R > G > B
    let mut data = Vec::new();
    // Most pixels clustered around typical film base
    for _ in 0..8000 {
        data.extend_from_slice(&[0.75, 0.55, 0.35]);
    }
    // Some image content (darker)
    for _ in 0..2000 {
        data.extend_from_slice(&[0.4, 0.3, 0.2]);
    }

    let image = DecodedImage {
        width: 100,
        height: 100,
        data,
        channels: 3,
        black_level: None,
        white_level: None,
        color_matrix: None,
        source_is_grayscale: false,
        is_monochrome: false,
    };

    let result = estimate_base_from_histogram(&image);
    assert!(result.is_ok());

    let base = result.unwrap();
    // Should detect the peak around the film base values
    assert!(
        base.medians[0] > base.medians[1],
        "R should be > G for color negative"
    );
    assert!(
        base.medians[1] > base.medians[2],
        "G should be > B for color negative"
    );
}

#[test]
fn test_histogram_estimation_insufficient_pixels() {
    // Very small image with few valid pixels
    let image = create_uniform_image(10, 10, [0.1, 0.1, 0.1]); // All dark, outside valid range
    let result = estimate_base_from_histogram(&image);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Insufficient valid pixels"));
}

// ========================================================================
// Filter and Validation Tests
// ========================================================================

#[test]
fn test_filter_valid_base_pixels_removes_clipped() {
    let pixels = vec![
        [0.5, 0.4, 0.3],    // Valid (has color variation)
        [0.99, 0.98, 0.97], // Clipped (near white)
        [0.6, 0.5, 0.4],    // Valid (has color variation)
        [0.01, 0.01, 0.01], // Dark
    ];

    let filtered = filter_valid_base_pixels(pixels);

    // Pixels with color variation pass; clipped and dark are removed
    // Both [0.5, 0.4, 0.3] and [0.6, 0.5, 0.4] have >5% variation and are not clipped/dark
    assert_eq!(filtered.pixels.len(), 2);
    assert!(filtered.clipped_ratio > 0.0, "Some should be clipped");
    assert!(filtered.dark_ratio > 0.0, "Some should be dark");
}

#[test]
fn test_filter_valid_base_pixels_removes_dark() {
    let pixels = vec![
        [0.5, 0.4, 0.3],    // Valid
        [0.02, 0.02, 0.02], // Too dark
    ];

    let filtered = filter_valid_base_pixels(pixels);

    assert!(filtered.dark_ratio > 0.0);
}

#[test]
fn test_validate_base_candidate_valid_orange_mask() {
    // Valid orange mask: R > G > B, bright, low noise
    let medians = [0.8, 0.6, 0.35];
    let noise = [0.02, 0.02, 0.02];
    let brightness = 0.65;

    let (valid, _reason) = validate_base_candidate(&medians, &noise, brightness);
    assert!(valid, "Valid orange mask should pass validation");
}

#[test]
fn test_validate_base_candidate_too_dark() {
    let medians = [0.5, 0.4, 0.3];
    let noise = [0.02, 0.02, 0.02];
    let brightness = 0.15; // Below threshold

    let (valid, reason) = validate_base_candidate(&medians, &noise, brightness);
    assert!(!valid);
    assert!(reason.contains("brightness"));
}

#[test]
fn test_validate_base_candidate_too_noisy() {
    let medians = [0.8, 0.6, 0.35];
    let noise = [0.3, 0.3, 0.3]; // Very noisy
    let brightness = 0.65;

    let (valid, reason) = validate_base_candidate(&medians, &noise, brightness);
    assert!(!valid);
    assert!(reason.contains("noise"));
}

#[test]
fn test_validate_base_candidate_wrong_channel_order() {
    // Wrong order: R < G (not orange mask)
    // Use values with significant color variation (not B&W)
    // But with R/G and G/B ratios in valid ranges, just wrong ordering
    // R/G = 0.55/0.7 = 0.786 (in 0.70-2.20 range)
    // G/B = 0.7/0.55 = 1.27 (in 1.00-2.50 range)
    // But R < G, so should fail channel ordering check
    let medians = [0.55, 0.70, 0.55]; // R < G (not R >= G >= B)
    let noise = [0.02, 0.02, 0.02];
    let brightness = 0.55;

    let (valid, reason) = validate_base_candidate(&medians, &noise, brightness);
    assert!(!valid, "Should fail validation: {}", reason);
    // Should fail on channel ordering since R < G
    assert!(
        reason.contains("channel ordering"),
        "Should mention channel ordering: {}",
        reason
    );
}

#[test]
fn test_validate_base_candidate_bw_film() {
    // B&W film: all channels similar
    let medians = [0.7, 0.69, 0.71];
    let noise = [0.02, 0.02, 0.02];
    let brightness = 0.7;

    let (valid, reason) = validate_base_candidate(&medians, &noise, brightness);
    assert!(valid, "B&W film should be detected and accepted");
    assert!(reason.contains("B&W"));
}

// ========================================================================
// is_likely_bw Tests
// ========================================================================

#[test]
fn test_is_likely_bw_grayscale() {
    let medians = [0.5, 0.5, 0.5];
    assert!(is_likely_bw(&medians));
}

#[test]
fn test_is_likely_bw_near_grayscale() {
    // Within 15% deviation
    let medians = [0.52, 0.48, 0.50];
    assert!(is_likely_bw(&medians));
}

#[test]
fn test_is_likely_bw_color() {
    let medians = [0.8, 0.5, 0.3]; // Clear color difference
    assert!(!is_likely_bw(&medians));
}

// ========================================================================
// Median Computation Tests
// ========================================================================

#[test]
fn test_compute_median_odd_length() {
    let mut values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let median = compute_median(&mut values);
    assert!((median - 3.0).abs() < 0.001, "Median of [1,2,3,4,5] = 3");
}

#[test]
fn test_compute_median_even_length() {
    let mut values = vec![1.0, 2.0, 3.0, 4.0];
    let median = compute_median(&mut values);
    assert!(
        (median - 2.5).abs() < 0.001,
        "Median of [1,2,3,4] = 2.5, got {}",
        median
    );
}

#[test]
fn test_compute_median_empty() {
    let mut values: Vec<f32> = vec![];
    let median = compute_median(&mut values);
    assert_eq!(median, 0.0);
}

#[test]
fn test_compute_median_single_value() {
    let mut values = vec![5.0];
    let median = compute_median(&mut values);
    assert_eq!(median, 5.0);
}

// ========================================================================
// Noise Stats Tests
// ========================================================================

#[test]
fn test_compute_noise_stats_uniform() {
    // All identical pixels = zero noise
    let pixels = vec![[0.5, 0.5, 0.5]; 100];
    let medians = [0.5, 0.5, 0.5];
    let noise = compute_noise_stats(&pixels, &medians);

    for &n in &noise {
        assert!(n < 0.001, "Uniform pixels should have ~0 noise");
    }
}

#[test]
fn test_compute_noise_stats_varied() {
    // Varied pixels = some noise
    let pixels = vec![[0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.5, 0.5, 0.5]];
    let medians = [0.5, 0.5, 0.5];
    let noise = compute_noise_stats(&pixels, &medians);

    for &n in &noise {
        assert!(n > 0.0, "Varied pixels should have some noise");
    }
}

#[test]
fn test_compute_noise_stats_empty() {
    let pixels: Vec<[f32; 3]> = vec![];
    let medians = [0.5, 0.5, 0.5];
    let noise = compute_noise_stats(&pixels, &medians);

    assert_eq!(noise, [0.0, 0.0, 0.0]);
}

// ========================================================================
// extract_roi_pixels Tests
// ========================================================================

#[test]
fn test_extract_roi_pixels_full_image() {
    let image = create_uniform_image(10, 10, [0.5, 0.5, 0.5]);
    let pixels = extract_roi_pixels(&image, 0, 0, 10, 10);

    assert_eq!(pixels.len(), 100);
    for p in &pixels {
        assert!((p[0] - 0.5).abs() < 0.001);
    }
}

#[test]
fn test_extract_roi_pixels_partial() {
    let image = create_uniform_image(10, 10, [0.5, 0.5, 0.5]);
    let pixels = extract_roi_pixels(&image, 2, 2, 3, 3);

    assert_eq!(pixels.len(), 9); // 3x3 = 9 pixels
}

#[test]
fn test_extract_roi_pixels_clamped_to_bounds() {
    let image = create_uniform_image(10, 10, [0.5, 0.5, 0.5]);
    // ROI extends beyond image
    let pixels = extract_roi_pixels(&image, 8, 8, 5, 5);

    // Should be clamped to 2x2 = 4 pixels
    assert_eq!(pixels.len(), 4);
}

// ========================================================================
// extract_border_pixels Tests
// ========================================================================

#[test]
fn test_extract_border_pixels() {
    let image = create_uniform_image(100, 100, [0.5, 0.5, 0.5]);
    let border_pixels = extract_border_pixels(&image, 10.0); // 10% border

    // Border should have pixels from outer 10%
    assert!(!border_pixels.is_empty());
    // With 10% border on 100x100: border is 10 pixels on each side
    // Total pixels = 100*100 = 10000
    // Inner region = 80*80 = 6400
    // Border = 10000 - 6400 = 3600
    assert!(
        border_pixels.len() > 3000 && border_pixels.len() < 4000,
        "Expected ~3600 border pixels, got {}",
        border_pixels.len()
    );
}

// ========================================================================
// sample_region_brightness Tests
// ========================================================================

#[test]
fn test_sample_region_brightness_uniform() {
    let image = create_uniform_image(100, 100, [0.8, 0.6, 0.4]);
    let brightness = sample_region_brightness(&image, 0, 0, 100, 100);

    // Expected: 0.5*0.8 + 0.4*0.6 + 0.1*0.4 = 0.68
    assert!(
        (brightness - 0.68).abs() < 0.01,
        "Expected brightness ~0.68, got {}",
        brightness
    );
}

#[test]
fn test_sample_region_brightness_empty_region() {
    let image = create_uniform_image(10, 10, [0.5, 0.5, 0.5]);
    let brightness = sample_region_brightness(&image, 0, 0, 0, 0);

    assert_eq!(brightness, 0.0);
}

// ========================================================================
// compute_channel_medians_from_brightest Tests
// ========================================================================

#[test]
fn test_channel_medians_from_brightest() {
    // Create pixels with varying brightness
    let pixels = vec![
        [0.9, 0.7, 0.5], // Brightest (sum = 2.1)
        [0.8, 0.6, 0.4], // Second brightest (sum = 1.8)
        [0.3, 0.2, 0.1], // Dark
        [0.4, 0.3, 0.2], // Medium
    ];

    // Get medians from top 2 brightest
    let medians = compute_channel_medians_from_brightest(&pixels, 2);

    // Should be median of [0.9, 0.8], [0.7, 0.6], [0.5, 0.4]
    assert!(
        (medians[0] - 0.85).abs() < 0.01,
        "R median expected ~0.85, got {}",
        medians[0]
    );
}

#[test]
fn test_channel_medians_empty_pixels() {
    let pixels: Vec<[f32; 3]> = vec![];
    let medians = compute_channel_medians_from_brightest(&pixels, 10);
    assert_eq!(medians, [0.0, 0.0, 0.0]);
}
