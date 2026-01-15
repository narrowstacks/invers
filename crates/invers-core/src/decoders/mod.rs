//! Image decoders for various formats
//!
//! Support for RAW, TIFF, and PNG file formats.

mod png;
mod raw;
mod tiff;

#[cfg(test)]
mod tests;

use std::path::Path;

/// Decoded image data
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Image width in pixels
    pub width: u32,

    /// Image height in pixels
    pub height: u32,

    /// Linear RGB data (f32, 0.0-1.0 range)
    pub data: Vec<f32>,

    /// Number of channels (typically 3 for RGB)
    pub channels: u8,

    /// Black level from source
    pub black_level: Option<f32>,

    /// White level from source
    pub white_level: Option<f32>,

    /// Color matrix from camera (if available)
    pub color_matrix: Option<[[f32; 3]; 3]>,

    /// Whether the source image was grayscale (1 channel)
    /// This is true when the file format specified grayscale, even if
    /// the data has been expanded to RGB for processing
    pub source_is_grayscale: bool,

    /// Whether the image is effectively B&W (no meaningful color information)
    /// This is detected by analyzing color variance across pixels.
    /// Can be true even for RGB images that contain no color (e.g., B&W scans saved as RGB)
    pub is_monochrome: bool,
}

/// Decode an image from a file path
pub fn decode_image<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .ok_or_else(|| "No file extension found".to_string())?;

    match extension.as_str() {
        "tif" | "tiff" => tiff::decode_tiff(path),
        "png" => png::decode_png(path),
        // RAW formats (via invers-raw/LibRaw)
        ext if invers_raw::is_raw_extension(ext) => raw::decode_raw(path),
        _ => Err(format!("Unsupported file format: {}", extension)),
    }
}

/// Detect if an RGB image is effectively monochrome (B&W).
///
/// Analyzes color variance by sampling pixels across the image.
/// Returns true if the R, G, B channels are nearly identical for most pixels,
/// indicating a grayscale image stored in RGB format.
///
/// This uses a sampling approach for efficiency on large images.
pub(crate) fn detect_monochrome(data: &[f32], width: u32, height: u32) -> bool {
    let pixel_count = (width * height) as usize;

    // For small images, check all pixels; for large images, sample
    let sample_count = if pixel_count < 10000 {
        pixel_count
    } else {
        // Sample ~10000 pixels spread across the image
        10000
    };

    let step = pixel_count / sample_count;
    let mut color_diff_count = 0;

    // Threshold for considering channels "different"
    // In 0.0-1.0 range, 0.02 is about 5 in 8-bit or 1300 in 16-bit terms
    const CHANNEL_DIFF_THRESHOLD: f32 = 0.02;

    for i in 0..sample_count {
        let pixel_idx = i * step;
        let data_idx = pixel_idx * 3;

        if data_idx + 2 >= data.len() {
            break;
        }

        let r = data[data_idx];
        let g = data[data_idx + 1];
        let b = data[data_idx + 2];

        // Check if channels differ significantly
        let rg_diff = (r - g).abs();
        let rb_diff = (r - b).abs();
        let gb_diff = (g - b).abs();

        if rg_diff > CHANNEL_DIFF_THRESHOLD
            || rb_diff > CHANNEL_DIFF_THRESHOLD
            || gb_diff > CHANNEL_DIFF_THRESHOLD
        {
            color_diff_count += 1;
        }
    }

    // If less than 1% of sampled pixels have color, consider it monochrome
    let color_ratio = color_diff_count as f32 / sample_count as f32;
    color_ratio < 0.01
}
