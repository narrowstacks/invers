//! Colorspace conversions and XYZ color space transforms

use super::lab::{get_rgb_to_xyz_matrix, get_xyz_to_rgb_matrix};

/// Colorspace identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colorspace {
    /// Linear sRGB
    LinearSRGB,

    /// Linear Rec.2020
    LinearRec2020,

    /// Linear ProPhoto RGB
    LinearProPhoto,

    /// Linear Display P3
    LinearDisplayP3,
}

impl Colorspace {
    /// Get the colorspace name as a string
    pub fn as_str(&self) -> &str {
        match self {
            Self::LinearSRGB => "Linear sRGB",
            Self::LinearRec2020 => "Linear Rec.2020",
            Self::LinearProPhoto => "Linear ProPhoto RGB",
            Self::LinearDisplayP3 => "Linear Display P3",
        }
    }
}

impl std::str::FromStr for Colorspace {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linear srgb" | "srgb" | "linear-srgb" => Ok(Self::LinearSRGB),
            "linear rec.2020" | "rec2020" | "linear-rec2020" => Ok(Self::LinearRec2020),
            "linear prophoto rgb" | "prophoto" | "linear-prophoto" => Ok(Self::LinearProPhoto),
            "linear display p3" | "displayp3" | "linear-displayp3" => Ok(Self::LinearDisplayP3),
            _ => Err(format!("Unknown colorspace: {}", s)),
        }
    }
}

/// Transform image data from one colorspace to another
///
/// Performs RGB -> XYZ -> RGB transformation using the appropriate matrices.
/// All colorspaces are D65-adapted so no chromatic adaptation is needed.
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data (3 values per pixel)
/// * `from` - Source colorspace
/// * `to` - Target colorspace
///
/// # Returns
/// Ok(()) if successful, or Err with message if transformation fails
pub fn transform_colorspace(
    data: &mut [f32],
    from: Colorspace,
    to: Colorspace,
) -> Result<(), String> {
    // No-op if source and destination are the same
    if from == to {
        return Ok(());
    }

    // Get transformation matrices
    let rgb_to_xyz = get_rgb_to_xyz_matrix(from);
    let xyz_to_rgb = get_xyz_to_rgb_matrix(to);

    // Precompute combined matrix for efficiency: combined = xyz_to_rgb * rgb_to_xyz
    let combined = multiply_3x3_matrices(xyz_to_rgb, rgb_to_xyz);

    // Apply transformation to each pixel
    const PARALLEL_THRESHOLD: usize = 100_000;
    let num_pixels = data.len() / 3;

    if num_pixels >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        data.par_chunks_exact_mut(3).for_each(|pixel| {
            apply_3x3_matrix_to_pixel(pixel, &combined);
        });
    } else {
        for pixel in data.chunks_exact_mut(3) {
            apply_3x3_matrix_to_pixel(pixel, &combined);
        }
    }

    Ok(())
}

/// Multiply two 3x3 matrices: result = a * b
#[inline]
fn multiply_3x3_matrices(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut result = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}

/// Apply a 3x3 matrix to a single pixel in-place
#[inline]
fn apply_3x3_matrix_to_pixel(pixel: &mut [f32], matrix: &[[f32; 3]; 3]) {
    let r = pixel[0];
    let g = pixel[1];
    let b = pixel[2];
    pixel[0] = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b;
    pixel[1] = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b;
    pixel[2] = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b;
}
