//! Comparison functions for diagnostic analysis

use crate::decoders::DecodedImage;
use crate::pipeline::ProcessedImage;

use super::stats::{compute_histograms, compute_statistics};
use super::DiagnosticReport;

/// RGB sample patches (shadow, midtone, highlight) as flat f32 arrays
pub type RgbSamplePatches = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Create a difference map between two images
/// Returns absolute differences as a new image
/// Optimized for cache efficiency
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

    // Pre-allocate and use direct iteration for better performance
    let mut diff = Vec::with_capacity(img1.len());
    for i in 0..img1.len() {
        diff.push((img1[i] - img2[i]).abs());
    }

    Ok(diff)
}

/// Extract sample patches from different brightness regions
/// Returns (shadow_patch, midtone_patch, highlight_patch) as flat RGB arrays
pub fn extract_sample_patches(
    data: &[f32],
    width: u32,
    height: u32,
    patch_size: u32,
) -> Result<RgbSamplePatches, String> {
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
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
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

    Ok(DiagnosticReport::new(
        our_stats,
        third_party_stats,
        difference_stats,
        our_histograms,
        third_party_histograms,
        color_shift,
        exposure_ratio,
    ))
}
