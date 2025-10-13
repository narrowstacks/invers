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
        // RAW formats would go here (CR2, CR3, NEF, ARW, etc.)
        // "cr2" | "cr3" | "nef" | "arw" => decode_raw(path),
        _ => Err(format!("Unsupported file format: {}", extension)),
    }
}

/// Decode a TIFF file
fn decode_tiff<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open TIFF file: {}", e))?;
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(file))
        .map_err(|e| format!("Failed to create TIFF decoder: {}", e))?;

    // Get image dimensions
    let (width, height) = decoder.dimensions()
        .map_err(|e| format!("Failed to get TIFF dimensions: {}", e))?;

    // Get color type
    let color_type = decoder.colortype()
        .map_err(|e| format!("Failed to get TIFF color type: {}", e))?;

    // Read the image data
    let image_data = decoder.read_image()
        .map_err(|e| format!("Failed to read TIFF image data: {}", e))?;

    // Convert to f32 linear RGB based on bit depth and color type
    let (data, channels) = match image_data {
        tiff::decoder::DecodingResult::U8(buf) => {
            decode_tiff_u8(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::U16(buf) => {
            decode_tiff_u16(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::U32(buf) => {
            decode_tiff_u32(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::U64(buf) => {
            decode_tiff_u64(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::F32(buf) => {
            decode_tiff_f32(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::F64(buf) => {
            decode_tiff_f64(&buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::F16(buf) => {
            // Convert f16 to f32
            let f32_buf: Vec<f32> = buf.iter().map(|&v| v.to_f32()).collect();
            decode_tiff_f32(&f32_buf, width, height, color_type)?
        }
        tiff::decoder::DecodingResult::I8(_)
        | tiff::decoder::DecodingResult::I16(_)
        | tiff::decoder::DecodingResult::I32(_)
        | tiff::decoder::DecodingResult::I64(_) => {
            return Err("Signed integer TIFF formats not supported".to_string());
        }
    };

    Ok(DecodedImage {
        width,
        height,
        data,
        channels,
        black_level: None,  // TODO: Extract from TIFF tags if present
        white_level: None,  // TODO: Extract from TIFF tags if present
        color_matrix: None, // Typically not in TIFF
    })
}

/// Decode u8 TIFF data to f32 linear RGB
fn decode_tiff_u8(
    buf: &[u8],
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

    // Convert u8 to f32 linear
    // Assuming u8 is sRGB-encoded for 8-bit images, but since we want linear,
    // we'll just normalize to 0-1. For negatives, this should be fine.
    let data: Vec<f32> = buf.iter().map(|&v| v as f32 / 255.0).collect();

    // If grayscale, expand to RGB
    if channels == 1 {
        let rgb_data: Vec<f32> = data
            .iter()
            .flat_map(|&gray| [gray, gray, gray])
            .collect();
        Ok((rgb_data, 3))
    } else if channels == 4 {
        // RGBA: drop alpha channel, keep RGB
        let rgb_data: Vec<f32> = data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
        Ok((rgb_data, 3))
    } else {
        Ok((data, channels))
    }
}

/// Decode u16 TIFF data to f32 linear RGB
fn decode_tiff_u16(
    buf: &[u16],
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

    // Convert u16 to f32 linear (16-bit is typically already linear)
    let data: Vec<f32> = buf.iter().map(|&v| v as f32 / 65535.0).collect();

    // If grayscale, expand to RGB
    if channels == 1 {
        let rgb_data: Vec<f32> = data
            .iter()
            .flat_map(|&gray| [gray, gray, gray])
            .collect();
        Ok((rgb_data, 3))
    } else if channels == 4 {
        // RGBA: drop alpha channel, keep RGB
        let rgb_data: Vec<f32> = data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
        Ok((rgb_data, 3))
    } else {
        Ok((data, channels))
    }
}

/// Decode u32 TIFF data to f32 linear RGB
fn decode_tiff_u32(
    buf: &[u32],
    width: u32,
    height: u32,
    color_type: tiff::ColorType,
) -> Result<(Vec<f32>, u8), String> {
    let channels = match color_type {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        _ => return Err(format!("Unsupported TIFF color type for u32: {:?}", color_type)),
    };

    let expected_len = (width * height * channels as u32) as usize;
    if buf.len() != expected_len {
        return Err(format!(
            "TIFF buffer size mismatch: expected {}, got {}",
            expected_len,
            buf.len()
        ));
    }

    // Convert u32 to f32 linear
    let data: Vec<f32> = buf.iter().map(|&v| v as f32 / u32::MAX as f32).collect();

    // Handle grayscale and RGBA like u16
    if channels == 1 {
        let rgb_data: Vec<f32> = data
            .iter()
            .flat_map(|&gray| [gray, gray, gray])
            .collect();
        Ok((rgb_data, 3))
    } else if channels == 4 {
        let rgb_data: Vec<f32> = data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
        Ok((rgb_data, 3))
    } else {
        Ok((data, channels))
    }
}

/// Decode u64 TIFF data to f32 linear RGB
fn decode_tiff_u64(
    buf: &[u64],
    width: u32,
    height: u32,
    color_type: tiff::ColorType,
) -> Result<(Vec<f32>, u8), String> {
    let channels = match color_type {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        _ => return Err(format!("Unsupported color type for u64: {:?}", color_type)),
    };

    let expected_len = (width * height * channels as u32) as usize;
    if buf.len() != expected_len {
        return Err(format!(
            "TIFF buffer size mismatch: expected {}, got {}",
            expected_len,
            buf.len()
        ));
    }

    // Convert u64 to f32 linear (with potential precision loss)
    let data: Vec<f32> = buf.iter().map(|&v| v as f32 / u64::MAX as f32).collect();

    if channels == 1 {
        let rgb_data: Vec<f32> = data
            .iter()
            .flat_map(|&gray| [gray, gray, gray])
            .collect();
        Ok((rgb_data, 3))
    } else if channels == 4 {
        let rgb_data: Vec<f32> = data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
        Ok((rgb_data, 3))
    } else {
        Ok((data, channels))
    }
}

/// Decode f32 TIFF data to f32 linear RGB
fn decode_tiff_f32(
    buf: &[f32],
    width: u32,
    height: u32,
    color_type: tiff::ColorType,
) -> Result<(Vec<f32>, u8), String> {
    let channels = match color_type {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        _ => return Err(format!("Unsupported color type for f32: {:?}", color_type)),
    };

    let expected_len = (width * height * channels as u32) as usize;
    if buf.len() != expected_len {
        return Err(format!(
            "TIFF buffer size mismatch: expected {}, got {}",
            expected_len,
            buf.len()
        ));
    }

    // Already f32, just clone
    let data = buf.to_vec();

    if channels == 1 {
        let rgb_data: Vec<f32> = data
            .iter()
            .flat_map(|&gray| [gray, gray, gray])
            .collect();
        Ok((rgb_data, 3))
    } else if channels == 4 {
        let rgb_data: Vec<f32> = data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
        Ok((rgb_data, 3))
    } else {
        Ok((data, channels))
    }
}

/// Decode f64 TIFF data to f32 linear RGB
fn decode_tiff_f64(
    buf: &[f64],
    width: u32,
    height: u32,
    color_type: tiff::ColorType,
) -> Result<(Vec<f32>, u8), String> {
    let channels = match color_type {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        _ => return Err(format!("Unsupported color type for f64: {:?}", color_type)),
    };

    let expected_len = (width * height * channels as u32) as usize;
    if buf.len() != expected_len {
        return Err(format!(
            "TIFF buffer size mismatch: expected {}, got {}",
            expected_len,
            buf.len()
        ));
    }

    // Convert f64 to f32
    let data: Vec<f32> = buf.iter().map(|&v| v as f32).collect();

    if channels == 1 {
        let rgb_data: Vec<f32> = data
            .iter()
            .flat_map(|&gray| [gray, gray, gray])
            .collect();
        Ok((rgb_data, 3))
    } else if channels == 4 {
        let rgb_data: Vec<f32> = data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
        Ok((rgb_data, 3))
    } else {
        Ok((data, channels))
    }
}

/// Decode a PNG file
fn decode_png<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open PNG file: {}", e))?;
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
    let buffer_size = reader.output_buffer_size()
        .ok_or_else(|| "Failed to determine PNG buffer size".to_string())?;
    let mut buf = vec![0u8; buffer_size];
    let frame_info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to read PNG frame: {}", e))?;

    // Get the actual bytes used
    let bytes = &buf[..frame_info.buffer_size()];

    // Convert to f32 linear RGB
    let (data, channels) = match (color_type, bit_depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            decode_png_gray8(bytes, width, height)?
        }
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
            decode_png_gray16(bytes, width, height)?
        }
        (png::ColorType::Rgb, png::BitDepth::Eight) => {
            decode_png_rgb8(bytes, width, height)?
        }
        (png::ColorType::Rgb, png::BitDepth::Sixteen) => {
            decode_png_rgb16(bytes, width, height)?
        }
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            decode_png_rgba8(bytes, width, height)?
        }
        (png::ColorType::Rgba, png::BitDepth::Sixteen) => {
            decode_png_rgba16(bytes, width, height)?
        }
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

    Ok(DecodedImage {
        width,
        height,
        data,
        channels,
        black_level: None,
        white_level: None,
        color_matrix: None,
    })
}

/// Decode 8-bit grayscale PNG
fn decode_png_gray8(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // Convert to f32 and expand to RGB
    let rgb_data: Vec<f32> = bytes
        .iter()
        .flat_map(|&gray| {
            let val = gray as f32 / 255.0;
            [val, val, val]
        })
        .collect();

    Ok((rgb_data, 3))
}

/// Decode 16-bit grayscale PNG
fn decode_png_gray16(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 2) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // PNG 16-bit is big-endian
    let rgb_data: Vec<f32> = bytes
        .chunks_exact(2)
        .flat_map(|chunk| {
            let gray16 = u16::from_be_bytes([chunk[0], chunk[1]]);
            let val = gray16 as f32 / 65535.0;
            [val, val, val]
        })
        .collect();

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

/// Decode 8-bit RGBA PNG (drop alpha)
fn decode_png_rgba8(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 4) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // Drop alpha, keep RGB
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|rgba| {
            [
                rgba[0] as f32 / 255.0,
                rgba[1] as f32 / 255.0,
                rgba[2] as f32 / 255.0,
            ]
        })
        .collect();

    Ok((data, 3))
}

/// Decode 16-bit RGBA PNG (drop alpha)
fn decode_png_rgba16(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u8), String> {
    let expected_len = (width * height * 4 * 2) as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "PNG buffer size mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }

    // PNG 16-bit is big-endian, drop alpha
    let data: Vec<f32> = bytes
        .chunks_exact(8)
        .flat_map(|rgba| {
            let r = u16::from_be_bytes([rgba[0], rgba[1]]);
            let g = u16::from_be_bytes([rgba[2], rgba[3]]);
            let b = u16::from_be_bytes([rgba[4], rgba[5]]);
            // Skip alpha at rgba[6], rgba[7]
            [
                r as f32 / 65535.0,
                g as f32 / 65535.0,
                b as f32 / 65535.0,
            ]
        })
        .collect();

    Ok((data, 3))
}

/// Decode a RAW file using libraw
#[allow(dead_code)]
fn decode_raw<P: AsRef<Path>>(_path: P) -> Result<DecodedImage, String> {
    // TODO: Implement RAW decoding via libraw FFI
    Err("RAW decoding not yet implemented".to_string())
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
        assert!(result.is_ok(), "Failed to decode sample TIFF: {:?}", result.err());

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
        assert!(max_val <= 1.0 || max_val < 2.0, "Max value should be reasonable");

        // Calculate some basic statistics
        let sum: f32 = image.data.iter().sum();
        let mean = sum / image.data.len() as f32;
        println!("  Mean value: {}", mean);

        // For a negative, the mean should typically be in the mid-high range
        println!("Test passed: Sample TIFF decoded successfully");
    }
}
