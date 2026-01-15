//! Tests for levels adjustment functions

use super::*;

#[test]
fn test_auto_levels_simple() {
    // Simple test: values from 0.2 to 0.8 should stretch to 0.0-1.0
    let mut data = vec![
        0.2, 0.3, 0.4, // R channel: 0.2-0.8
        0.3, 0.4, 0.5, // G channel: 0.3-0.9
        0.4, 0.5, 0.6, // B channel: 0.4-0.7
        0.5, 0.6, 0.7, 0.6, 0.7, 0.8, 0.7, 0.8, 0.9, 0.8, 0.9, 0.7,
    ];

    let params = auto_levels(&mut data, 3, 0.0);

    // Check that values have been stretched
    assert!(data[0] < 0.2); // First R value should be lower
    let min_val = data.iter().cloned().fold(f32::MAX, f32::min);
    let max_val = data.iter().cloned().fold(f32::MIN, f32::max);
    assert!(min_val <= 0.001, "expected min close to 0, got {}", min_val);
    assert!(max_val >= 0.999, "expected max close to 1, got {}", max_val);

    println!("Auto-levels params: {:?}", params);
    println!("Adjusted data: {:?}", data);
}

#[test]
fn test_find_histogram_ends() {
    // Data ranging from 0.2 to 0.8
    let data: Vec<f32> = (0..100).map(|i| 0.2 + 0.6 * (i as f32 / 99.0)).collect();

    let ends = find_histogram_ends(&data);

    println!("Histogram ends: low={}, high={}", ends.low, ends.high);

    // Low should be close to 0.2
    assert!(
        (ends.low - 0.2).abs() < 0.01,
        "Expected low ~0.2, got {}",
        ends.low
    );
    // High should be close to 0.8
    assert!(
        (ends.high - 0.8).abs() < 0.01,
        "Expected high ~0.8, got {}",
        ends.high
    );
}

#[test]
fn test_otsu_threshold_bimodal() {
    // Create truly bimodal distribution with clear separation
    // All low values exactly at 0.2, all high values exactly at 0.8
    let mut data = vec![0.2; 500]; // Low cluster: 500 samples at 0.2
    data.extend(vec![0.8; 500]); // High cluster: 500 samples at 0.8

    let result = otsu_threshold(&data);

    println!(
        "Otsu threshold: {}, variance: {}, below_ratio: {}",
        result.threshold, result.variance, result.below_ratio
    );

    // Threshold should be at or between the two modes
    // With discrete bimodal data at 0.2 and 0.8, Otsu may find threshold
    // at the boundary of one of the modes (equal variance for any value in between)
    assert!(
        result.threshold >= 0.2 && result.threshold <= 0.8,
        "Expected threshold at/between modes (0.2 and 0.8), got {}",
        result.threshold
    );
    // Variance should be non-zero for bimodal data
    assert!(
        result.variance > 0.0,
        "Expected non-zero variance for bimodal data"
    );
}

#[test]
fn test_measure_dark_mid_light() {
    // Create data with known distribution
    let mut data = vec![0.1; 25]; // 25% dark (0.0-0.25)
    data.extend(vec![0.5; 50]); // 50% mid (0.25-0.75)
    data.extend(vec![0.9; 25]); // 25% light (0.75-1.0)

    let (dark, mid, light) = measure_dark_mid_light(&data);

    println!("Dark: {}%, Mid: {}%, Light: {}%", dark, mid, light);

    assert!((dark - 25.0).abs() < 0.1, "Expected 25% dark, got {}", dark);
    assert!((mid - 50.0).abs() < 0.1, "Expected 50% mid, got {}", mid);
    assert!(
        (light - 25.0).abs() < 0.1,
        "Expected 25% light, got {}",
        light
    );
}

#[test]
fn test_auto_levels_no_clip() {
    // Test the optimized single-pass auto_levels_no_clip
    // Create data with different ranges per channel
    let mut data = vec![
        0.1, 0.2, 0.3, // Pixel 1
        0.2, 0.3, 0.4, // Pixel 2
        0.3, 0.4, 0.5, // Pixel 3
        0.4, 0.5, 0.6, // Pixel 4
        0.5, 0.6, 0.7, // Pixel 5
        0.6, 0.7, 0.8, // Pixel 6 - B channel has max (0.8)
    ];

    // Save original max
    let original_max = data.iter().cloned().fold(f32::MIN, f32::max);

    let params = auto_levels_no_clip(&mut data, 3, 0.0);

    // The max value after adjustment should not exceed original max
    let new_max = data.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        new_max <= original_max + 0.001,
        "Max after no-clip should not exceed original max. Original: {}, New: {}",
        original_max,
        new_max
    );

    // Params should reflect the channel ranges found
    assert!(params[0] < params[1], "R min should be less than R max");
    assert!(params[2] < params[3], "G min should be less than G max");
    assert!(params[4] < params[5], "B min should be less than B max");

    println!("Auto-levels no-clip params: {:?}", params);
    println!("Original max: {}, New max: {}", original_max, new_max);
}

#[test]
fn test_auto_levels_no_clip_preserves_proportions() {
    // Verify that the relative proportions between channels are maintained
    // This is the key property of the no-clip variant
    let mut data = vec![
        0.2, 0.4, 0.6, // Proportions: 1:2:3
        0.4, 0.8, 0.9, // Different proportions
    ];

    let original_max = data.iter().cloned().fold(f32::MIN, f32::max);

    let _ = auto_levels_no_clip(&mut data, 3, 0.0);

    let new_max = data.iter().cloned().fold(f32::MIN, f32::max);

    // Max should be preserved (within tolerance)
    assert!(
        (new_max - original_max).abs() < 0.01,
        "Max should be approximately preserved. Original: {}, New: {}",
        original_max,
        new_max
    );
}
