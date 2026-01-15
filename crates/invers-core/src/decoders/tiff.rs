//! TIFF image decoder

use std::path::Path;

use super::{detect_monochrome, DecodedImage};

/// Decode a TIFF file
pub(crate) fn decode_tiff<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
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
