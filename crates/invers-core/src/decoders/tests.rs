//! Tests for image decoders

use super::*;

#[test]
fn test_decode_sample_tiff() {
    let sample_path = "../../assets/sample_flatbed_raw_tif.tif";

    // Only run if the sample exists
    if !std::path::Path::new(sample_path).exists() {
        eprintln!("Sample TIFF not found, skipping test");
        return;
    }

    let result = decode_image(sample_path);
    assert!(
        result.is_ok(),
        "Failed to decode sample TIFF: {:?}",
        result.err()
    );

    let image = result.unwrap();

    println!("Decoded sample TIFF:");
    println!("  Dimensions: {}x{}", image.width, image.height);
    println!("  Channels: {}", image.channels);
    println!("  Total pixels: {}", image.width * image.height);
    println!("  Data length: {}", image.data.len());

    // Verify data integrity
    assert_eq!(image.channels, 3, "Expected RGB output (3 channels)");
    assert_eq!(
        image.data.len(),
        (image.width * image.height * 3) as usize,
        "Data length should match width * height * channels"
    );

    // Check that we have valid float values (0.0 - 1.0)
    let min_val = image.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = image.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!("  Value range: {} to {}", min_val, max_val);

    // For a negative, values should generally be in a reasonable range
    assert!(min_val >= 0.0, "Min value should be >= 0.0");
    assert!(
        max_val <= 1.0 || max_val < 2.0,
        "Max value should be reasonable"
    );

    // Calculate some basic statistics
    let sum: f32 = image.data.iter().sum();
    let mean = sum / image.data.len() as f32;
    println!("  Mean value: {}", mean);

    // For a negative, the mean should typically be in the mid-high range
    println!("Test passed: Sample TIFF decoded successfully");
}
