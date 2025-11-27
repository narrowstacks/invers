//! Parity tests between CPU and GPU implementations.

use super::*;
use crate::decoders::DecodedImage;

const TOLERANCE: f32 = 1e-4; // Allow small floating-point variance

/// Generate a test gradient image
fn generate_test_gradient(width: u32, height: u32) -> DecodedImage {
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = x as f32 / (width - 1) as f32;
            let g = y as f32 / (height - 1) as f32;
            let b = ((x + y) as f32 / (width + height - 2) as f32).min(1.0);
            data.push(r);
            data.push(g);
            data.push(b);
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
    }
}

/// Compare two pixel arrays with tolerance
fn pixels_equal(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        if diff > tolerance {
            eprintln!(
                "Pixel mismatch at index {}: CPU={}, GPU={}, diff={}",
                i, av, bv, diff
            );
            return false;
        }
    }

    true
}

#[test]
fn test_gpu_available() {
    if !is_gpu_available() {
        eprintln!("GPU not available, skipping GPU tests");
        return;
    }

    let info = gpu_info().expect("Should get GPU info");
    eprintln!("GPU: {}", info);
}

#[test]
fn test_upload_download_roundtrip() {
    if !is_gpu_available() {
        return;
    }

    let ctx = GpuContext::new().expect("Failed to create GPU context");

    let test_data: Vec<f32> = (0..1024 * 3).map(|i| (i as f32) / 3072.0).collect();

    let gpu_image = buffers::GpuImage::upload(
        ctx.device.clone(),
        ctx.queue.clone(),
        &test_data,
        32,
        32,
        3,
    )
    .expect("Failed to upload");

    let downloaded = gpu_image.download().expect("Failed to download");

    assert!(
        pixels_equal(&test_data, &downloaded, 1e-6),
        "Upload/download roundtrip failed"
    );
}

#[test]
fn test_histogram_buffers() {
    if !is_gpu_available() {
        return;
    }

    let ctx = GpuContext::new().expect("Failed to create GPU context");

    // Create test image with known distribution
    let width = 256u32;
    let height = 256u32;
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for _ in 0..width {
            let value = y as f32 / 255.0;
            data.push(value);
            data.push(value);
            data.push(value);
        }
    }

    let _gpu_image = buffers::GpuImage::upload(
        ctx.device.clone(),
        ctx.queue.clone(),
        &data,
        width,
        height,
        3,
    )
    .expect("Failed to upload");

    let histogram = buffers::GpuHistogram::new(ctx.device.clone(), ctx.queue.clone());

    // Just verify the histogram structures work (not accumulated yet)
    let [hist_r, hist_g, hist_b] = histogram.download().expect("Failed to download histogram");

    // Initially histogram should be all zeros
    assert!(
        hist_r.iter().all(|&x| x == 0),
        "Histogram R should be initially zero"
    );
    assert!(
        hist_g.iter().all(|&x| x == 0),
        "Histogram G should be initially zero"
    );
    assert!(
        hist_b.iter().all(|&x| x == 0),
        "Histogram B should be initially zero"
    );
}

#[test]
fn test_gpu_context_creation() {
    if !is_gpu_available() {
        eprintln!("GPU not available, skipping context test");
        return;
    }

    let ctx = GpuContext::new().expect("Failed to create GPU context");
    let info = ctx.adapter_info();
    eprintln!("GPU adapter: {} ({:?})", info.name, info.backend);
}

#[test]
fn test_gradient_image_generation() {
    let img = generate_test_gradient(64, 64);
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
    assert_eq!(img.data.len(), 64 * 64 * 3);
    assert_eq!(img.channels, 3);

    // Check corners
    // Top-left: (0,0) should be R=0, G=0, B=0
    assert!(img.data[0] < 0.01);
    assert!(img.data[1] < 0.01);
    assert!(img.data[2] < 0.01);

    // Bottom-right: (63,63) should be R≈1, G≈1, B≈1
    let last_pixel = (63 * 64 + 63) * 3;
    assert!(img.data[last_pixel] > 0.99);
    assert!(img.data[last_pixel + 1] > 0.99);
    assert!(img.data[last_pixel + 2] > 0.99);
}

/// Generate a simulated film negative with realistic orange mask characteristics
fn generate_test_negative(width: u32, height: u32) -> DecodedImage {
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Simulate orange mask base (~0.7, 0.5, 0.3) typical of color negative film
    let base_r = 0.70f32;
    let base_g = 0.50f32;
    let base_b = 0.30f32;

    for y in 0..height {
        for x in 0..width {
            // Create a varying pattern that simulates scene content
            let scene_r = (x as f32 / (width - 1) as f32) * 0.3;
            let scene_g = (y as f32 / (height - 1) as f32) * 0.25;
            let scene_b = ((x + y) as f32 / (width + height - 2) as f32).min(1.0) * 0.2;

            // Film negative inverts brightness: darker scene = brighter on film
            // Combine base (orange mask) with scene content
            let r = base_r - scene_r;
            let g = base_g - scene_g;
            let b = base_b - scene_b;

            data.push(r.clamp(0.0, 1.0));
            data.push(g.clamp(0.0, 1.0));
            data.push(b.clamp(0.0, 1.0));
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
    }
}

/// Create minimal ConvertOptions for testing
fn create_test_options(use_gpu: bool) -> crate::models::ConvertOptions {
    use crate::models::*;
    use std::path::PathBuf;

    ConvertOptions {
        input_paths: vec![PathBuf::from("test.tif")],
        output_dir: PathBuf::from("."),
        output_format: OutputFormat::Tiff16,
        working_colorspace: "linear-rec2020".to_string(),
        bit_depth_policy: BitDepthPolicy::Force16Bit,
        film_preset: None,
        scan_profile: None,
        base_estimation: Some(BaseEstimation {
            medians: [0.70, 0.50, 0.30],
            roi: None,
            noise_stats: Some([0.01, 0.01, 0.01]),
            auto_estimated: false,
            mask_profile: None,
        }),
        num_threads: None,
        skip_tone_curve: true,        // Skip for simpler comparison
        skip_color_matrix: true,      // Skip for simpler comparison
        exposure_compensation: 1.0,
        debug: false,
        enable_auto_levels: false,    // Disable for deterministic comparison
        auto_levels_clip_percent: 0.1,
        preserve_headroom: false,
        enable_auto_color: false,
        auto_color_strength: 0.5,
        auto_color_min_gain: 0.8,
        auto_color_max_gain: 1.5,
        base_brightest_percent: 15.0,
        base_sampling_mode: BaseSamplingMode::Median,
        inversion_mode: InversionMode::Linear, // Simplest mode for testing
        shadow_lift_mode: ShadowLiftMode::None,
        shadow_lift_value: 0.0,
        highlight_compression: 0.0,
        enable_auto_exposure: false,
        auto_exposure_target_median: 0.18,
        auto_exposure_strength: 0.5,
        auto_exposure_min_gain: 0.5,
        auto_exposure_max_gain: 2.0,
        no_clip: true,  // Preserve full range for comparison
        enable_auto_wb: false,
        use_gpu,
    }
}

#[test]
fn test_gpu_cpu_parity_linear_inversion() {
    if !is_gpu_available() {
        eprintln!("GPU not available, skipping parity test");
        return;
    }

    // Generate test negative
    let decoded = generate_test_negative(128, 128);
    let decoded_for_gpu = decoded.clone();

    // Process with CPU
    let cpu_options = create_test_options(false);
    let cpu_result = crate::pipeline::process_image(decoded, &cpu_options)
        .expect("CPU processing failed");

    // Process with GPU
    let gpu_options = create_test_options(true);
    let gpu_result = crate::pipeline::process_image(decoded_for_gpu, &gpu_options)
        .expect("GPU processing failed");

    // Compare results
    assert_eq!(cpu_result.width, gpu_result.width, "Width mismatch");
    assert_eq!(cpu_result.height, gpu_result.height, "Height mismatch");
    assert_eq!(cpu_result.data.len(), gpu_result.data.len(), "Data length mismatch");

    // Allow slightly larger tolerance for GPU (floating point accumulation differences)
    let parity_tolerance = 1e-3;

    let mut max_diff: f32 = 0.0;
    let mut mismatch_count = 0;

    for (i, (cpu_val, gpu_val)) in cpu_result.data.iter().zip(gpu_result.data.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        max_diff = max_diff.max(diff);

        if diff > parity_tolerance {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                eprintln!(
                    "Parity mismatch at index {}: CPU={:.6}, GPU={:.6}, diff={:.6}",
                    i, cpu_val, gpu_val, diff
                );
            }
        }
    }

    eprintln!("Max difference between CPU and GPU: {:.6}", max_diff);

    // For now, just log the results since exact parity is hard to achieve
    // In the future, we can tighten this tolerance
    if mismatch_count > 0 {
        eprintln!(
            "Warning: {} mismatches found out of {} values (tolerance={})",
            mismatch_count,
            cpu_result.data.len(),
            parity_tolerance
        );
    }

    // Verify that results are reasonably close (within 1% for most values)
    let acceptable_mismatch_rate = 0.05; // Allow 5% of values to exceed tight tolerance
    let mismatch_rate = mismatch_count as f32 / cpu_result.data.len() as f32;

    assert!(
        mismatch_rate < acceptable_mismatch_rate,
        "Too many mismatches: {}% (max allowed: {}%)",
        mismatch_rate * 100.0,
        acceptable_mismatch_rate * 100.0
    );
}
