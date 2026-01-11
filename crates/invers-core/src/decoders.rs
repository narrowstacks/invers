//! Image decoders for various formats
//!
//! Support for RAW, TIFF, and PNG file formats.

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
        "tif" | "tiff" => decode_tiff(path),
        "png" => decode_png(path),
        // RAW formats (via invers-raw/LibRaw)
        ext if invers_raw::is_raw_extension(ext) => decode_raw(path),
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
fn detect_monochrome(data: &[f32], width: u32, height: u32) -> bool {
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

/// Decode a TIFF file
fn decode_tiff<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    use std::fs::File;
    use std::io::BufReader;
    use tiff::decoder::Limits;

    let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open TIFF file: {}", e))?;

    // Configure limits for large film scans (up to 1GB uncompressed)
    let mut limits = Limits::default();
    limits.decoding_buffer_size = 1024 * 1024 * 1024; // 1GB
    limits.ifd_value_size = 1024 * 1024 * 1024;
    limits.intermediate_buffer_size = 1024 * 1024 * 1024;

    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(file))
        .map_err(|e| format!("Failed to create TIFF decoder: {}", e))?
        .with_limits(limits);

    // Get image dimensions
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| format!("Failed to get TIFF dimensions: {}", e))?;

    // Get color type
    let color_type = decoder
        .colortype()
        .map_err(|e| format!("Failed to get TIFF color type: {}", e))?;

    // Read the image data
    let image_data = decoder
        .read_image()
        .map_err(|e| format!("Failed to read TIFF image data: {}", e))?;

    // Track if source was grayscale
    let source_is_grayscale = matches!(color_type, tiff::ColorType::Gray(_));

    // Convert to f32 linear RGB based on bit depth and color type
    // Uses generic decode_tiff_buffer for all numeric types
    let (data, channels) = match image_data {
        tiff::decoder::DecodingResult::U8(buf) => {
            decode_tiff_buffer(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::U16(buf) => {
            decode_tiff_buffer(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::U32(buf) => {
            decode_tiff_buffer(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::U64(buf) => {
            decode_tiff_buffer(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::F32(buf) => {
            decode_tiff_buffer(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::F64(buf) => {
            decode_tiff_buffer(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::F16(buf) => {
            // Convert f16 to f32 first, then use generic decoder
            let f32_buf: Vec<f32> = buf.iter().map(|&v| v.to_f32()).collect();
            decode_tiff_buffer(&f32_buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::I8(_)
        | tiff::decoder::DecodingResult::I16(_)
        | tiff::decoder::DecodingResult::I32(_)
        | tiff::decoder::DecodingResult::I64(_) => {
            return Err("Signed integer TIFF formats not supported".to_string());
        }
    };

    // Detect if image is monochrome (grayscale source or RGB with no color variance)
    let is_monochrome = source_is_grayscale || detect_monochrome(&data, width, height);

    Ok(DecodedImage {
        width,
        height,
        data,
        channels,
        black_level: None,  // TODO: Extract from TIFF tags if present
        white_level: None,  // TODO: Extract from TIFF tags if present
        color_matrix: None, // Typically not in TIFF
        source_is_grayscale,
        is_monochrome,
    })
}

// =============================================================================
// Generic TIFF value trait and decoder to eliminate code duplication
// =============================================================================

/// Trait for TIFF sample types that can be normalized to f32
trait TiffValue: Copy {
    /// Normalize this value to f32 in range [0.0, 1.0]
    fn to_normalized_f32(self) -> f32;
}

impl TiffValue for u8 {
    #[inline]
    fn to_normalized_f32(self) -> f32 {
        self as f32 / 255.0
    }
}

impl TiffValue for u16 {
    #[inline]
    fn to_normalized_f32(self) -> f32 {
        self as f32 / 65535.0
    }
}

impl TiffValue for u32 {
    #[inline]
    fn to_normalized_f32(self) -> f32 {
        self as f32 / u32::MAX as f32
    }
}

impl TiffValue for u64 {
    #[inline]
    fn to_normalized_f32(self) -> f32 {
        self as f32 / u64::MAX as f32
    }
}

impl TiffValue for f32 {
    #[inline]
    fn to_normalized_f32(self) -> f32 {
        self
    }
}

impl TiffValue for f64 {
    #[inline]
    fn to_normalized_f32(self) -> f32 {
        self as f32
    }
}

/// Generic TIFF buffer decoder - handles all numeric types
fn decode_tiff_buffer<T: TiffValue>(
    buf: &[T],
    width: u32,
    height: u32,
    color_type: tiff::ColorType,
) -> Result<(Vec<f32>, u8), String> {
    let channels = match color_type {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        tiff::ColorType::CMYK(_) => return Err("CMYK color type not supported".to_string()),
        tiff::ColorType::YCbCr(_) => return Err("YCbCr color type not supported yet".to_string()),
        tiff::ColorType::Palette(_) => return Err("Palette color type not supported".to_string()),
        _ => return Err(format!("Unknown TIFF color type: {:?}", color_type)),
    };

    let expected_len = (width * height * channels as u32) as usize;
    if buf.len() != expected_len {
        return Err(format!(
            "TIFF buffer size mismatch: expected {}, got {}",
            expected_len,
            buf.len()
        ));
    }

    match channels {
        1 => {
            // Grayscale: expand to RGB with pre-allocation
            let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
            for &val in buf {
                let gray = val.to_normalized_f32();
                rgb_data.push(gray);
                rgb_data.push(gray);
                rgb_data.push(gray);
            }
            Ok((rgb_data, 3))
        }
        4 => {
            // RGBA: drop alpha channel with pre-allocation
            let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
            for rgba in buf.chunks_exact(4) {
                rgb_data.push(rgba[0].to_normalized_f32());
                rgb_data.push(rgba[1].to_normalized_f32());
                rgb_data.push(rgba[2].to_normalized_f32());
            }
            Ok((rgb_data, 3))
        }
        _ => {
            // RGB: direct conversion
            let data: Vec<f32> = buf.iter().map(|&v| v.to_normalized_f32()).collect();
            Ok((data, channels))
        }
    }
}

/// Decode a PNG file
fn decode_png<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open PNG file: {}", e))?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Failed to read PNG info: {}", e))?;

    let info = reader.info();
    let width = info.width;
    let height = info.height;
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;

    // Allocate buffer for image data
    let buffer_size = reader
        .output_buffer_size()
        .ok_or_else(|| "Failed to determine PNG buffer size".to_string())?;
    let mut buf = vec![0u8; buffer_size];
    let frame_info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to read PNG frame: {}", e))?;

    // Get the actual bytes used
    let bytes = &buf[..frame_info.buffer_size()];

    // Track if source was grayscale
    let source_is_grayscale = matches!(
        color_type,
        png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha
    );

    // Convert to f32 linear RGB
    let (data, channels) = match (color_type, bit_depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            decode_png_gray8(bytes, width, height)?
        }
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
            decode_png_gray16(bytes, width, height)?
        }
        (png::ColorType::Rgb, png::BitDepth::Eight) => decode_png_rgb8(bytes, width, height)?,
        (png::ColorType::Rgb, png::BitDepth::Sixteen) => decode_png_rgb16(bytes, width, height)?,
        (png::ColorType::Rgba, png::BitDepth::Eight) => decode_png_rgba8(bytes, width, height)?,
        (png::ColorType::Rgba, png::BitDepth::Sixteen) => decode_png_rgba16(bytes, width, height)?,
        (png::ColorType::GrayscaleAlpha, _) => {
            return Err("Grayscale+Alpha PNG not yet supported".to_string());
        }
        (png::ColorType::Indexed, _) => {
            return Err("Indexed PNG not supported".to_string());
        }
        _ => {
            return Err(format!(
                "Unsupported PNG format: {:?} with bit depth {:?}",
                color_type, bit_depth
            ));
        }
    };

    // Detect if image is monochrome (grayscale source or RGB with no color variance)
    let is_monochrome = source_is_grayscale || detect_monochrome(&data, width, height);

    Ok(DecodedImage {
        width,
        height,
        data,
        channels,
        black_level: None,
        white_level: None,
        color_matrix: None,
        source_is_grayscale,
        is_monochrome,
    })
}

/// Decode 8-bit grayscale PNG with pre-allocation
fn decode_png_gray8(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // Pre-allocate for RGB output
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

    // Convert to f32 and expand to RGB
    for &gray in bytes {
        let val = gray as f32 / 255.0;
        rgb_data.push(val);
        rgb_data.push(val);
        rgb_data.push(val);
    }

    Ok((rgb_data, 3))
}

/// Decode 16-bit grayscale PNG with pre-allocation
fn decode_png_gray16(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 2) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // Pre-allocate for RGB output
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

    // PNG 16-bit is big-endian
    for chunk in bytes.chunks_exact(2) {
        let gray16 = u16::from_be_bytes([chunk[0], chunk[1]]);
        let val = gray16 as f32 / 65535.0;
        rgb_data.push(val);
        rgb_data.push(val);
        rgb_data.push(val);
    }

    Ok((rgb_data, 3))
}

/// Decode 8-bit RGB PNG
fn decode_png_rgb8(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 3) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    let data: Vec<f32> = bytes.iter().map(|&v| v as f32 / 255.0).collect();
    Ok((data, 3))
}

/// Decode 16-bit RGB PNG
fn decode_png_rgb16(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 3 * 2) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // PNG 16-bit is big-endian
    let data: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let val16 = u16::from_be_bytes([chunk[0], chunk[1]]);
            val16 as f32 / 65535.0
        })
        .collect();

    Ok((data, 3))
}

/// Decode 8-bit RGBA PNG (drop alpha) with pre-allocation
fn decode_png_rgba8(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 4) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // Pre-allocate for RGB output
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

    // Drop alpha, keep RGB
    for rgba in bytes.chunks_exact(4) {
        rgb_data.push(rgba[0] as f32 / 255.0);
        rgb_data.push(rgba[1] as f32 / 255.0);
        rgb_data.push(rgba[2] as f32 / 255.0);
    }

    Ok((rgb_data, 3))
}

/// Decode 16-bit RGBA PNG (drop alpha) with pre-allocation
fn decode_png_rgba16(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 4 * 2) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // Pre-allocate for RGB output
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

    // PNG 16-bit is big-endian, drop alpha
    for rgba in bytes.chunks_exact(8) {
        let r = u16::from_be_bytes([rgba[0], rgba[1]]);
        let g = u16::from_be_bytes([rgba[2], rgba[3]]);
        let b = u16::from_be_bytes([rgba[4], rgba[5]]);
        // Skip alpha at rgba[6], rgba[7]
        rgb_data.push(r as f32 / 65535.0);
        rgb_data.push(g as f32 / 65535.0);
        rgb_data.push(b as f32 / 65535.0);
    }

    Ok((rgb_data, 3))
}

/// Decode a RAW file using invers-raw (LibRaw wrapper)
fn decode_raw<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    let raw = invers_raw::decode_raw(path)?;

    let is_monochrome = detect_monochrome(&raw.data, raw.width, raw.height);

    Ok(DecodedImage {
        width: raw.width,
        height: raw.height,
        data: raw.data,
        channels: raw.channels,
        black_level: raw.black_level,
        white_level: raw.white_level,
        color_matrix: raw.color_matrix,
        source_is_grayscale: false,
        is_monochrome,
    })
}

#[cfg(test)]
mod tests {
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
}
