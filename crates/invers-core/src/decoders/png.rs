//! PNG image decoder

use std::path::Path;

use super::{detect_monochrome, DecodedImage};

/// Decode a PNG file
pub(crate) fn decode_png<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
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
