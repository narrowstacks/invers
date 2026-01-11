//! RAW file decoding using LibRaw
//!
//! This crate isolates the rsraw/rsraw-sys dependencies to avoid rebuilding
//! LibRaw bindings when the post-processing pipeline changes.

use std::path::Path;

/// Decoded RAW image data
#[derive(Debug, Clone)]
pub struct DecodedRaw {
    /// Image width in pixels
    pub width: u32,

    /// Image height in pixels
    pub height: u32,

    /// Linear RGB data (f32, 0.0-1.0 range)
    pub data: Vec<f32>,

    /// Number of channels (always 3 for RGB output)
    pub channels: u8,

    /// Black level from source (if available)
    pub black_level: Option<f32>,

    /// White level from source (if available)
    pub white_level: Option<f32>,

    /// Color matrix from camera (if available)
    pub color_matrix: Option<[[f32; 3]; 3]>,
}

/// List of supported RAW file extensions
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "nef", "nrw", "arw", "raf", "rw2", "orf", "pef", "dng", "3fr", "fff", "iiq",
    "rwl", "raw",
];

/// Check if a file extension is a supported RAW format
pub fn is_raw_extension(ext: &str) -> bool {
    RAW_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Decode a RAW file using rsraw (LibRaw wrapper)
pub fn decode_raw<P: AsRef<Path>>(path: P) -> Result<DecodedRaw, String> {
    use rsraw::{RawImage, BIT_DEPTH_16};
    use std::convert::AsMut;

    // Read file into buffer
    let data =
        std::fs::read(path.as_ref()).map_err(|e| format!("Failed to read RAW file: {}", e))?;

    // Open RAW file
    let mut raw = RawImage::open(&data).map_err(|e| format!("Failed to open RAW file: {:?}", e))?;

    // Configure LibRaw processing parameters via low-level access
    // SAFETY: rsraw provides safe AsMut access to libraw_data_t
    {
        let libraw_data: &mut rsraw_sys::libraw_data_t = raw.as_mut();
        // Use AHD demosaic (best quality for film scanning)
        // 0 = linear, 1 = VNG, 2 = PPG, 3 = AHD
        libraw_data.params.user_qual = 3;
        // Disable automatic brightness adjustment (we handle this in pipeline)
        libraw_data.params.no_auto_bright = 1;
        // Use camera white balance if available
        libraw_data.params.use_camera_wb = 1;
    }

    // Unpack the RAW data (modifies raw in place)
    raw.unpack()
        .map_err(|e| format!("Failed to unpack RAW data: {:?}", e))?;

    // Process to 16-bit output (best quality for film scanning)
    let processed = raw
        .process::<BIT_DEPTH_16>()
        .map_err(|e| format!("Failed to process RAW: {:?}", e))?;

    let width = processed.width();
    let height = processed.height();
    let channels = processed.colors() as u8;

    // Get raw pixel data (ProcessedImage<BIT_DEPTH_16> derefs to &[u16])
    let pixel_data: &[u16] = &processed;
    let data = convert_raw_u16_to_f32_rgb(pixel_data, width, height, channels)?;

    Ok(DecodedRaw {
        width,
        height,
        data,
        channels: 3,
        // rsraw's high-level API doesn't expose black/white levels or color matrices
        // These would need to be accessed via rsraw-sys for low-level LibRaw access
        black_level: None,
        white_level: None,
        color_matrix: None,
    })
}

/// Convert RAW pixel data (16-bit u16 slice) to f32 linear RGB (0.0-1.0)
/// Uses parallel processing via rayon for large images
fn convert_raw_u16_to_f32_rgb(
    pixel_data: &[u16],
    width: u32,
    height: u32,
    channels: u8,
) -> Result<Vec<f32>, String> {
    use rayon::prelude::*;

    let pixel_count = (width * height) as usize;
    let expected_len = pixel_count * channels as usize;

    if pixel_data.len() < expected_len {
        return Err(format!(
            "RAW buffer size mismatch: expected at least {}, got {}",
            expected_len,
            pixel_data.len()
        ));
    }

    let rgb_data = if channels == 3 {
        // RGB data: 3 u16 values per pixel - parallel conversion
        pixel_data[..expected_len]
            .par_chunks_exact(3)
            .flat_map(|pixel| {
                [
                    pixel[0] as f32 / 65535.0,
                    pixel[1] as f32 / 65535.0,
                    pixel[2] as f32 / 65535.0,
                ]
            })
            .collect()
    } else if channels == 4 {
        // RGBA data: drop alpha channel - parallel conversion
        pixel_data[..pixel_count * 4]
            .par_chunks_exact(4)
            .flat_map(|pixel| {
                [
                    pixel[0] as f32 / 65535.0,
                    pixel[1] as f32 / 65535.0,
                    pixel[2] as f32 / 65535.0,
                ]
            })
            .collect()
    } else if channels == 1 {
        // Grayscale: expand to RGB - parallel conversion
        pixel_data[..pixel_count]
            .par_iter()
            .flat_map(|&gray| {
                let val = gray as f32 / 65535.0;
                [val, val, val]
            })
            .collect()
    } else {
        return Err(format!("Unexpected RAW channel count: {}", channels));
    };

    Ok(rgb_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_raw_extension() {
        assert!(is_raw_extension("cr2"));
        assert!(is_raw_extension("CR2"));
        assert!(is_raw_extension("nef"));
        assert!(is_raw_extension("dng"));
        assert!(!is_raw_extension("tiff"));
        assert!(!is_raw_extension("png"));
        assert!(!is_raw_extension("jpg"));
    }
}
