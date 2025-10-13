//! Image exporters for various output formats
//!
//! Export processed images to TIFF, DNG, and other formats.

use crate::pipeline::ProcessedImage;
use std::path::Path;

/// Export a processed image to TIFF format
pub fn export_tiff16<P: AsRef<Path>>(
    image: &ProcessedImage,
    path: P,
    icc_profile: Option<&[u8]>,
) -> Result<(), String> {
    use std::fs::File;
    use std::io::BufWriter;

    if image.channels != 3 {
        return Err(format!(
            "TIFF export only supports 3-channel RGB, got {} channels",
            image.channels
        ));
    }

    // Convert f32 (0.0-1.0) to u16 (0-65535)
    let u16_data: Vec<u16> = image
        .data
        .iter()
        .map(|&v| {
            let clamped = v.clamp(0.0, 1.0);
            (clamped * 65535.0).round() as u16
        })
        .collect();

    // Create output file
    let file = File::create(path.as_ref())
        .map_err(|e| format!("Failed to create TIFF file: {}", e))?;
    let writer = BufWriter::new(file);

    // Create TIFF encoder
    let mut encoder = tiff::encoder::TiffEncoder::new(writer)
        .map_err(|e| format!("Failed to create TIFF encoder: {}", e))?;

    // Write the image as 16-bit RGB
    encoder
        .write_image::<tiff::encoder::colortype::RGB16>(
            image.width,
            image.height,
            &u16_data,
        )
        .map_err(|e| format!("Failed to write TIFF image: {}", e))?;

    // TODO: Embed ICC profile if provided
    // The tiff crate doesn't currently expose easy ICC profile embedding
    // This would require writing custom TIFF tags
    let _ = icc_profile;

    Ok(())
}

/// Export a processed image to Linear DNG format
pub fn export_linear_dng<P: AsRef<Path>>(
    image: &ProcessedImage,
    path: P,
    _metadata: &DngMetadata,
) -> Result<(), String> {
    let _path = path.as_ref();

    // TODO: Implement Linear DNG export
    // - Convert to 16-bit linear
    // - Write TIFF structure with DNG tags
    // - Include essential DNG metadata
    // - Preserve camera color matrices if available

    let _ = image; // suppress unused warning

    Err("Linear DNG export not yet implemented".to_string())
}

/// Metadata for DNG export
#[derive(Debug, Clone)]
pub struct DngMetadata {
    /// Camera make
    pub make: Option<String>,

    /// Camera model
    pub model: Option<String>,

    /// Black level
    pub black_level: Option<f32>,

    /// White level
    pub white_level: Option<f32>,

    /// As-shot neutral (white balance)
    pub as_shot_neutral: Option<[f32; 3]>,

    /// Color matrix
    pub color_matrix: Option<[[f32; 3]; 3]>,
}

impl Default for DngMetadata {
    fn default() -> Self {
        Self {
            make: None,
            model: None,
            black_level: None,
            white_level: None,
            as_shot_neutral: None,
            color_matrix: None,
        }
    }
}
