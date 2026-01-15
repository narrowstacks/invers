//! Pixel extraction functions for film base estimation
//!
//! Contains functions for extracting and filtering pixel samples:
//! - ROI pixel extraction
//! - Border pixel extraction
//! - Pixel filtering for base estimation
//! - Region brightness sampling

use crate::decoders::DecodedImage;
use crate::verbose_println;

use super::{BaseRoiCandidate, FilteredBasePixels};

/// Filter out clipped/extreme pixels that are not valid film base
///
/// Film base should be bright but NOT clipped white. This filters:
/// - Near-white clipped pixels (all channels > 0.95)
/// - Very dark pixels (all channels < 0.05)
/// - Bright grayscale pixels without color variation (not orange mask)
pub(crate) fn filter_valid_base_pixels(pixels: Vec<[f32; 3]>) -> FilteredBasePixels {
    let total = pixels.len() as f32;
    let mut clipped_count = 0usize;
    let mut dark_count = 0usize;

    let valid_pixels: Vec<[f32; 3]> = pixels
        .into_iter()
        .filter(|p| {
            let [r, g, b] = *p;
            let max_val = r.max(g).max(b);
            let min_val = r.min(g).min(b);

            // Exclude near-white clipped pixels (any channel > 0.98 or all > 0.90)
            // More aggressive threshold to catch partially clipped pixels
            if max_val > 0.98 || min_val > 0.90 {
                clipped_count += 1;
                return false;
            }

            // Exclude very dark pixels (all channels < 0.05)
            if max_val < 0.05 {
                dark_count += 1;
                return false;
            }

            // For bright pixels (max > 0.7), require some color variation
            // (orange mask has R > G > B, not all equal)
            if max_val > 0.7 {
                let range = max_val - min_val;
                // Require at least 5% variation between channels for bright pixels
                if range < 0.05 {
                    return false;
                }
            }

            true
        })
        .collect();

    FilteredBasePixels {
        pixels: valid_pixels,
        clipped_ratio: if total > 0.0 {
            clipped_count as f32 / total
        } else {
            0.0
        },
        dark_ratio: if total > 0.0 {
            dark_count as f32 / total
        } else {
            0.0
        },
    }
}

/// Extract pixels from a region of interest with pre-allocation
pub(crate) fn extract_roi_pixels(
    image: &DecodedImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Vec<[f32; 3]> {
    // Pre-allocate with exact capacity
    let capacity = (width * height) as usize;
    let mut pixels = Vec::with_capacity(capacity);

    for row in y..(y + height) {
        let row_start = (row * image.width + x) as usize * 3;
        let row_pixels = ((x + width).min(image.width) - x) as usize;
        let row_end = row_start + row_pixels * 3;

        if row_end <= image.data.len() {
            // Process entire row at once for better cache locality
            for pixel in image.data[row_start..row_end].chunks_exact(3) {
                pixels.push([pixel[0], pixel[1], pixel[2]]);
            }
        }
    }

    pixels
}

/// Extract pixels from the outer border region of an image
///
/// The border is defined as the outer N% of the image dimensions, forming
/// a rectangular frame. A pixel is included if:
/// - x < border_width OR x >= (width - border_width), OR
/// - y < border_height OR y >= (height - border_height)
///
/// This is useful for sampling film base from full-frame scans that include
/// the rebate/sprocket area around the edges.
pub(crate) fn extract_border_pixels(image: &DecodedImage, border_percent: f32) -> Vec<[f32; 3]> {
    let border_percent = border_percent.clamp(1.0, 25.0);
    let border_width = ((image.width as f32) * border_percent / 100.0).ceil() as u32;
    let border_height = ((image.height as f32) * border_percent / 100.0).ceil() as u32;

    // Calculate border area for pre-allocation
    // Border area = total area - inner area
    let inner_width = image.width.saturating_sub(2 * border_width);
    let inner_height = image.height.saturating_sub(2 * border_height);
    let total_area = image.width as usize * image.height as usize;
    let inner_area = inner_width as usize * inner_height as usize;
    let border_area = total_area.saturating_sub(inner_area);

    let mut pixels = Vec::with_capacity(border_area);

    let right_edge = image.width.saturating_sub(border_width);
    let bottom_edge = image.height.saturating_sub(border_height);

    for y in 0..image.height {
        let is_vertical_border = y < border_height || y >= bottom_edge;

        for x in 0..image.width {
            let is_horizontal_border = x < border_width || x >= right_edge;

            // Include pixel if it's in the border region
            if is_vertical_border || is_horizontal_border {
                let idx = ((y * image.width + x) * 3) as usize;
                if idx + 2 < image.data.len() {
                    pixels.push([image.data[idx], image.data[idx + 1], image.data[idx + 2]]);
                }
            }
        }
    }

    pixels
}

/// Estimate film base ROI candidates automatically using heuristics.
/// Order is determined later based on brightness.
pub(crate) fn estimate_base_roi_candidates(image: &DecodedImage) -> Vec<BaseRoiCandidate> {
    let border_width = (image.width / 20).clamp(50, 200);
    let border_height = (image.height / 20).clamp(50, 200);

    let candidates = [
        (
            sample_region_brightness(image, 0, 0, image.width, border_height),
            (
                border_width,
                0,
                image.width - 2 * border_width,
                border_height,
            ),
            "top",
        ),
        (
            sample_region_brightness(
                image,
                0,
                image.height.saturating_sub(border_height),
                image.width,
                border_height,
            ),
            (
                border_width,
                image.height.saturating_sub(border_height),
                image.width - 2 * border_width,
                border_height,
            ),
            "bottom",
        ),
        (
            sample_region_brightness(image, 0, 0, border_width, image.height),
            (
                0,
                border_height,
                border_width,
                image.height - 2 * border_height,
            ),
            "left",
        ),
        (
            sample_region_brightness(
                image,
                image.width.saturating_sub(border_width),
                0,
                border_width,
                image.height,
            ),
            (
                image.width.saturating_sub(border_width),
                border_height,
                border_width,
                image.height - 2 * border_height,
            ),
            "right",
        ),
    ];

    verbose_println!(
        "[BASE] Auto-detection brightnesses: top={:.4}, bottom={:.4}, left={:.4}, right={:.4}",
        candidates[0].0,
        candidates[1].0,
        candidates[2].0,
        candidates[3].0
    );

    // Create candidates from borders
    let mut result: Vec<BaseRoiCandidate> = candidates
        .iter()
        .map(|(brightness, rect, label)| BaseRoiCandidate::new(*rect, *brightness, label))
        .collect();

    // Add center region as a fallback candidate (for full-frame images)
    // Use a smaller center sample (10% of image) for potential sky/highlight areas
    let center_width = image.width / 10;
    let center_height = image.height / 10;
    let center_x = (image.width - center_width) / 2;
    let center_y = (image.height - center_height) / 2;

    if center_width > 20 && center_height > 20 {
        let center_brightness =
            sample_region_brightness(image, center_x, center_y, center_width, center_height);
        verbose_println!("[BASE] Center region brightness: {:.4}", center_brightness);
        result.push(BaseRoiCandidate::new(
            (center_x, center_y, center_width, center_height),
            center_brightness,
            "center",
        ));
    }

    result
}

/// Sample the average brightness of a region with color-aware weighting
/// For color negatives, film base is orange (high R+G, lower B)
/// Weight channels to detect orange mask: 0.5*R + 0.4*G + 0.1*B
/// Optimized to use single-pass accumulation
pub(crate) fn sample_region_brightness(
    image: &DecodedImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> f32 {
    let x_end = (x + width).min(image.width);
    let y_end = (y + height).min(image.height);

    let mut sum = 0.0;
    let mut count = 0;

    // Single pass with bounds checking
    for row in y..y_end {
        let row_start = (row * image.width + x) as usize * 3;
        let row_end = (row * image.width + x_end) as usize * 3;

        if row_end <= image.data.len() {
            // Process entire row at once for better cache locality
            for pixel in image.data[row_start..row_end].chunks_exact(3) {
                // Weight for orange mask detection (prioritizes R+G over B)
                sum += 0.5 * pixel[0] + 0.4 * pixel[1] + 0.1 * pixel[2];
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}
