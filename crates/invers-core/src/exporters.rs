//! Image exporters for various output formats
//!
//! Export processed images to TIFF, DNG, and other formats.

use crate::pipeline::ProcessedImage;
use std::path::Path;

/// Export a processed image to TIFF format
///
/// Note: This exports the image data as-is (typically linear light).
/// The image data should have already gone through the full processing pipeline
/// including tone curves, color matrices, and exposure adjustments.
///
/// TIFF viewers that support linear workflows will display this correctly.
/// For sRGB output, the pipeline should apply sRGB tone curve before export.
///
/// If `image.export_as_grayscale` is true, exports as single-channel grayscale
/// (using the first channel, or averaging RGB for B&W images).
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

    // Create output file
    let file =
        File::create(path.as_ref()).map_err(|e| format!("Failed to create TIFF file: {}", e))?;
    let writer = BufWriter::new(file);

    // Create TIFF encoder
    let mut encoder = tiff::encoder::TiffEncoder::new(writer)
        .map_err(|e| format!("Failed to create TIFF encoder: {}", e))?;

    if image.export_as_grayscale {
        // Export as grayscale - average RGB channels (they should be nearly identical for B&W)
        let u16_data: Vec<u16> = image
            .data
            .chunks_exact(3)
            .map(|rgb| {
                let avg = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                let clamped = avg.clamp(0.0, 1.0);
                (clamped * 65535.0).round() as u16
            })
            .collect();

        // Write the image as 16-bit Grayscale
        encoder
            .write_image::<tiff::encoder::colortype::Gray16>(image.width, image.height, &u16_data)
            .map_err(|e| format!("Failed to write grayscale TIFF image: {}", e))?;
    } else {
        // Convert f32 (0.0-1.0) to u16 (0-65535)
        // This is a simple linear scaling - the data should already be
        // in the correct colorspace and tone-mapped by the pipeline
        let u16_data: Vec<u16> = image
            .data
            .iter()
            .map(|&v| {
                let clamped = v.clamp(0.0, 1.0);
                (clamped * 65535.0).round() as u16
            })
            .collect();

        // Write the image as 16-bit RGB
        encoder
            .write_image::<tiff::encoder::colortype::RGB16>(image.width, image.height, &u16_data)
            .map_err(|e| format!("Failed to write TIFF image: {}", e))?;
    }

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
#[derive(Debug, Clone, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_test_image(width: u32, height: u32, channels: u8, grayscale: bool) -> ProcessedImage {
        let pixel_count = (width * height) as usize;
        let data = vec![0.5; pixel_count * channels as usize];
        ProcessedImage {
            width,
            height,
            data,
            channels,
            export_as_grayscale: grayscale,
        }
    }

    // ========================================================================
    // DngMetadata Tests
    // ========================================================================

    #[test]
    fn test_dng_metadata_default() {
        let metadata = DngMetadata::default();

        assert!(metadata.make.is_none());
        assert!(metadata.model.is_none());
        assert!(metadata.black_level.is_none());
        assert!(metadata.white_level.is_none());
        assert!(metadata.as_shot_neutral.is_none());
        assert!(metadata.color_matrix.is_none());
    }

    #[test]
    fn test_dng_metadata_with_values() {
        let metadata = DngMetadata {
            make: Some("Nikon".to_string()),
            model: Some("D850".to_string()),
            black_level: Some(150.0),
            white_level: Some(16383.0),
            as_shot_neutral: Some([0.5, 1.0, 0.7]),
            color_matrix: Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        };

        assert_eq!(metadata.make, Some("Nikon".to_string()));
        assert_eq!(metadata.model, Some("D850".to_string()));
        assert!((metadata.black_level.unwrap() - 150.0).abs() < 0.001);
    }

    // ========================================================================
    // export_tiff16 Tests
    // ========================================================================

    #[test]
    fn test_export_tiff16_wrong_channels() {
        let image = create_test_image(10, 10, 4, false); // 4 channels, not supported
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.tiff");

        let result = export_tiff16(&image, &path, None);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("only supports 3-channel RGB"));
    }

    #[test]
    fn test_export_tiff16_rgb_success() {
        let image = create_test_image(10, 10, 3, false);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_rgb.tiff");

        let result = export_tiff16(&image, &path, None);

        assert!(result.is_ok(), "RGB export should succeed: {:?}", result);
        assert!(path.exists(), "TIFF file should exist");

        // Verify file is not empty
        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0, "TIFF file should not be empty");
    }

    #[test]
    fn test_export_tiff16_grayscale_success() {
        let image = create_test_image(10, 10, 3, true); // export_as_grayscale = true
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_gray.tiff");

        let result = export_tiff16(&image, &path, None);

        assert!(
            result.is_ok(),
            "Grayscale export should succeed: {:?}",
            result
        );
        assert!(path.exists(), "TIFF file should exist");
    }

    #[test]
    fn test_export_tiff16_clamps_values() {
        // Create image with out-of-range values
        let mut image = create_test_image(2, 2, 3, false);
        image.data = vec![
            -0.5, 1.5, 0.5, // Out of range values
            0.0, 1.0, 0.5, // Edge values
            0.25, 0.75, 0.5, // Normal values
            2.0, -1.0, 0.5, // More out of range
        ];

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_clamp.tiff");

        let result = export_tiff16(&image, &path, None);

        // Should succeed - values are clamped internally
        assert!(
            result.is_ok(),
            "Export with out-of-range values should clamp: {:?}",
            result
        );
    }

    #[test]
    fn test_export_tiff16_invalid_path() {
        let image = create_test_image(10, 10, 3, false);
        // Try to write to a directory that doesn't exist
        let path = "/nonexistent/directory/test.tiff";

        let result = export_tiff16(&image, path, None);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to create TIFF file"));
    }

    // ========================================================================
    // export_linear_dng Tests
    // ========================================================================

    #[test]
    fn test_export_linear_dng_not_implemented() {
        let image = create_test_image(10, 10, 3, false);
        let metadata = DngMetadata::default();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.dng");

        let result = export_linear_dng(&image, &path, &metadata);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not yet implemented"));
    }
}
