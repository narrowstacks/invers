//! Color management and transformations
//!
//! Provides colorspace conversions (RGB <-> HSL, RGB <-> LAB), ICC profile handling,
//! and working colorspace transformations.

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
/// Performs RGB → XYZ → RGB transformation using the appropriate matrices.
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

// =============================================================================
// RGB <-> HSL Conversions
// =============================================================================

/// HSL color representation
/// - H (hue): 0.0-360.0 degrees
/// - S (saturation): 0.0-1.0
/// - L (lightness): 0.0-1.0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hsl {
    pub h: f32,
    pub s: f32,
    pub l: f32,
}

/// Convert RGB to HSL
///
/// Input: RGB values in range 0.0-1.0
/// Output: HSL where H is 0.0-360.0, S and L are 0.0-1.0
#[inline]
pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> Hsl {
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Lightness
    let l = (max + min) / 2.0;

    // Achromatic case
    if delta < 1e-6 {
        return Hsl { h: 0.0, s: 0.0, l };
    }

    // Saturation
    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    // Hue
    let h = if (max - r).abs() < 1e-6 {
        let mut h = (g - b) / delta;
        if g < b {
            h += 6.0;
        }
        h * 60.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / delta + 2.0) * 60.0
    } else {
        ((r - g) / delta + 4.0) * 60.0
    };

    Hsl { h: h % 360.0, s, l }
}

/// Convert HSL to RGB
///
/// Input: HSL where H is 0.0-360.0, S and L are 0.0-1.0
/// Output: RGB values in range 0.0-1.0
#[inline]
pub fn hsl_to_rgb(hsl: Hsl) -> (f32, f32, f32) {
    let Hsl { h, s, l } = hsl;
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);

    // Achromatic case
    if s < 1e-6 {
        return (l, l, l);
    }

    let h = h % 360.0;
    let h = if h < 0.0 { h + 360.0 } else { h };

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    (r, g, b)
}

/// Helper function for HSL to RGB conversion
#[inline]
fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }

    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

/// Convert RGB array to HSL in place (for batch processing)
/// Data is interleaved RGB triplets
pub fn rgb_array_to_hsl(data: &[f32]) -> Vec<Hsl> {
    data.chunks_exact(3)
        .map(|rgb| rgb_to_hsl(rgb[0], rgb[1], rgb[2]))
        .collect()
}

/// Convert HSL array back to RGB (for batch processing)
/// Returns interleaved RGB triplets
pub fn hsl_array_to_rgb(hsl_data: &[Hsl]) -> Vec<f32> {
    let mut result = Vec::with_capacity(hsl_data.len() * 3);
    for hsl in hsl_data {
        let (r, g, b) = hsl_to_rgb(*hsl);
        result.push(r);
        result.push(g);
        result.push(b);
    }
    result
}

// =============================================================================
// RGB <-> LAB Conversions (D65 illuminant, sRGB primaries)
// =============================================================================

/// LAB color representation (CIE L*a*b*)
/// - L: 0.0-100.0 (lightness)
/// - a: approximately -128 to +128 (green-red axis)
/// - b: approximately -128 to +128 (blue-yellow axis)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

/// D65 standard illuminant reference white point
const D65_X: f32 = 0.95047;
const D65_Y: f32 = 1.00000;
const D65_Z: f32 = 1.08883;

/// sRGB to XYZ matrix (D65)
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.119_192, 0.9503041],
];

/// XYZ to sRGB matrix (D65)
const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.969_266, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

/// Rec.2020 to XYZ matrix (D65)
/// Source: ITU-R BT.2020 specification
const REC2020_TO_XYZ: [[f32; 3]; 3] = [
    [0.6370, 0.1446, 0.1689],
    [0.2627, 0.6780, 0.0593],
    [0.0000, 0.0281, 1.0610],
];

/// XYZ to Rec.2020 matrix (D65)
/// Inverse of REC2020_TO_XYZ
const XYZ_TO_REC2020: [[f32; 3]; 3] = [
    [1.7167, -0.3557, -0.2534],
    [-0.6667, 1.6165, 0.0158],
    [0.0176, -0.0428, 0.9421],
];

/// ProPhoto RGB to XYZ matrix (D50, adapted to D65)
#[allow(clippy::excessive_precision)]
const PROPHOTO_TO_XYZ: [[f32; 3]; 3] = [
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
];

/// XYZ to ProPhoto RGB matrix (D65 adapted to D50)
const XYZ_TO_PROPHOTO: [[f32; 3]; 3] = [
    [1.3459433, -0.2556075, -0.0511118],
    [-0.5445989, 1.5081673, 0.0205351],
    [0.0000000, 0.0000000, 1.2118128],
];

/// Display P3 to XYZ matrix (D65)
const DISPLAYP3_TO_XYZ: [[f32; 3]; 3] = [
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
];

/// XYZ to Display P3 matrix (D65)
#[allow(clippy::excessive_precision)]
const XYZ_TO_DISPLAYP3: [[f32; 3]; 3] = [
    [2.4934969, -0.9313836, -0.4027108],
    [-0.8294890, 1.7626641, 0.0236247],
    [0.0358458, -0.0761724, 0.9568845],
];

/// Get the RGB to XYZ matrix for a colorspace
fn get_rgb_to_xyz_matrix(colorspace: Colorspace) -> &'static [[f32; 3]; 3] {
    match colorspace {
        Colorspace::LinearSRGB => &SRGB_TO_XYZ,
        Colorspace::LinearRec2020 => &REC2020_TO_XYZ,
        Colorspace::LinearProPhoto => &PROPHOTO_TO_XYZ,
        Colorspace::LinearDisplayP3 => &DISPLAYP3_TO_XYZ,
    }
}

/// Get the XYZ to RGB matrix for a colorspace
fn get_xyz_to_rgb_matrix(colorspace: Colorspace) -> &'static [[f32; 3]; 3] {
    match colorspace {
        Colorspace::LinearSRGB => &XYZ_TO_SRGB,
        Colorspace::LinearRec2020 => &XYZ_TO_REC2020,
        Colorspace::LinearProPhoto => &XYZ_TO_PROPHOTO,
        Colorspace::LinearDisplayP3 => &XYZ_TO_DISPLAYP3,
    }
}

/// Convert linear RGB to XYZ (D65) with explicit colorspace
#[inline]
fn linear_rgb_to_xyz_with_colorspace(
    r: f32,
    g: f32,
    b: f32,
    colorspace: Colorspace,
) -> (f32, f32, f32) {
    let m = get_rgb_to_xyz_matrix(colorspace);
    let x = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    let y = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    let z = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    (x, y, z)
}

/// Convert XYZ to linear RGB (D65) with explicit colorspace
#[inline]
fn xyz_to_linear_rgb_with_colorspace(
    x: f32,
    y: f32,
    z: f32,
    colorspace: Colorspace,
) -> (f32, f32, f32) {
    let m = get_xyz_to_rgb_matrix(colorspace);
    let r = m[0][0] * x + m[0][1] * y + m[0][2] * z;
    let g = m[1][0] * x + m[1][1] * y + m[1][2] * z;
    let b = m[2][0] * x + m[2][1] * y + m[2][2] * z;
    (r, g, b)
}

/// LAB f(t) function
#[inline]
fn lab_f(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA_CUBED: f32 = DELTA * DELTA * DELTA; // ~0.008856

    if t > DELTA_CUBED {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// LAB f^-1(t) inverse function
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;

    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Convert linear RGB to CIE LAB (D65 illuminant)
///
/// IMPORTANT: This function assumes sRGB input. For other colorspaces,
/// use `rgb_to_lab_with_colorspace` instead.
///
/// Input: Linear RGB values in range 0.0-1.0
/// Output: LAB where L is 0-100, a and b are approximately -128 to +128
#[inline]
pub fn rgb_to_lab(r: f32, g: f32, b: f32) -> Lab {
    rgb_to_lab_with_colorspace(r, g, b, Colorspace::LinearSRGB)
}

/// Convert linear RGB to CIE LAB (D65 illuminant) with explicit colorspace
///
/// Use this function when working in a non-sRGB colorspace like Rec.2020.
///
/// Input: Linear RGB values in range 0.0-1.0
/// Output: LAB where L is 0-100, a and b are approximately -128 to +128
#[inline]
pub fn rgb_to_lab_with_colorspace(r: f32, g: f32, b: f32, colorspace: Colorspace) -> Lab {
    let r = r.max(0.0);
    let g = g.max(0.0);
    let b = b.max(0.0);

    // RGB to XYZ using the correct colorspace matrix
    let (x, y, z) = linear_rgb_to_xyz_with_colorspace(r, g, b, colorspace);

    // Normalize by reference white
    let xn = x / D65_X;
    let yn = y / D65_Y;
    let zn = z / D65_Z;

    // Apply LAB f function
    let fx = lab_f(xn);
    let fy = lab_f(yn);
    let fz = lab_f(zn);

    // Calculate LAB
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    Lab { l, a, b }
}

/// Convert CIE LAB to linear RGB (D65 illuminant)
///
/// IMPORTANT: This function returns sRGB output. For other colorspaces,
/// use `lab_to_rgb_with_colorspace` instead.
///
/// Input: LAB where L is 0-100, a and b are approximately -128 to +128
/// Output: Linear RGB values (may be outside 0.0-1.0 for out-of-gamut colors)
#[inline]
pub fn lab_to_rgb(lab: Lab) -> (f32, f32, f32) {
    lab_to_rgb_with_colorspace(lab, Colorspace::LinearSRGB)
}

/// Convert CIE LAB to linear RGB (D65 illuminant) with explicit colorspace
///
/// Use this function when working in a non-sRGB colorspace like Rec.2020.
///
/// Input: LAB where L is 0-100, a and b are approximately -128 to +128
/// Output: Linear RGB values (may be outside 0.0-1.0 for out-of-gamut colors)
#[inline]
pub fn lab_to_rgb_with_colorspace(lab: Lab, colorspace: Colorspace) -> (f32, f32, f32) {
    let Lab { l, a, b } = lab;

    // LAB to XYZ
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let x = D65_X * lab_f_inv(fx);
    let y = D65_Y * lab_f_inv(fy);
    let z = D65_Z * lab_f_inv(fz);

    // XYZ to RGB using the correct colorspace matrix
    xyz_to_linear_rgb_with_colorspace(x, y, z, colorspace)
}

/// Convert RGB array to LAB (for batch processing)
/// Data is interleaved RGB triplets
pub fn rgb_array_to_lab(data: &[f32]) -> Vec<Lab> {
    data.chunks_exact(3)
        .map(|rgb| rgb_to_lab(rgb[0], rgb[1], rgb[2]))
        .collect()
}

/// Convert LAB array back to RGB (for batch processing)
/// Returns interleaved RGB triplets
pub fn lab_array_to_rgb(lab_data: &[Lab]) -> Vec<f32> {
    let mut result = Vec::with_capacity(lab_data.len() * 3);
    for lab in lab_data {
        let (r, g, b) = lab_to_rgb(*lab);
        result.push(r);
        result.push(g);
        result.push(b);
    }
    result
}

// =============================================================================
// 8-Color HSL Adjustments (Camera Raw style)
// =============================================================================

/// Get the color range index (0-7) and blend weight for a given hue
///
/// The 8 color ranges with their approximate hue centers:
/// - R (Reds): 0° (wraps around 330°-30°)
/// - O (Oranges): 30°
/// - Y (Yellows): 60°
/// - G (Greens): 120°
/// - A (Aquas): 180°
/// - B (Blues): 240°
/// - P (Purples): 285°
/// - M (Magentas): 315°
///
/// Returns (primary_index, secondary_index, blend_factor)
/// where blend_factor indicates how much to blend from primary to secondary
#[inline]
fn get_color_range_weights(hue: f32) -> (usize, usize, f32) {
    // Hue centers for 8 colors (non-uniform spacing like Camera Raw)
    const CENTERS: [f32; 8] = [
        0.0,   // R - Reds
        30.0,  // O - Oranges
        60.0,  // Y - Yellows
        120.0, // G - Greens
        180.0, // A - Aquas
        240.0, // B - Blues
        285.0, // P - Purples
        315.0, // M - Magentas
    ];

    // Normalize hue to 0-360
    let h = ((hue % 360.0) + 360.0) % 360.0;

    // Find the two nearest color centers
    let mut min_dist = f32::MAX;
    let mut primary = 0usize;

    for (i, &center) in CENTERS.iter().enumerate() {
        let dist = hue_distance(h, center);
        if dist < min_dist {
            min_dist = dist;
            primary = i;
        }
    }

    // Find secondary (next nearest)
    let mut second_dist = f32::MAX;
    let mut secondary = (primary + 1) % 8;

    for (i, &center) in CENTERS.iter().enumerate() {
        if i != primary {
            let dist = hue_distance(h, center);
            if dist < second_dist {
                second_dist = dist;
                secondary = i;
            }
        }
    }

    // Calculate blend factor based on distance to primary
    // At the center of primary, blend = 0; at the boundary, blend approaches 0.5
    let total_dist = min_dist + second_dist;
    let blend = if total_dist > 0.0 {
        (min_dist / total_dist).clamp(0.0, 0.5)
    } else {
        0.0
    };

    (primary, secondary, blend)
}

/// Calculate distance between two hues (handling wrap-around)
#[inline]
fn hue_distance(h1: f32, h2: f32) -> f32 {
    let diff = (h1 - h2).abs();
    diff.min(360.0 - diff)
}

/// Apply 8-color HSL adjustments to RGB data in place
///
/// This is the Camera Raw style HSL adjustment that affects
/// specific color ranges independently.
pub fn apply_hsl_adjustments(data: &mut [f32], adjustments: &crate::models::HslAdjustments) {
    if !adjustments.has_adjustments() {
        return;
    }

    for pixel in data.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);

        // Convert to HSL
        let hsl = rgb_to_hsl(r, g, b);

        // Skip achromatic pixels (no meaningful hue)
        if hsl.s < 0.01 {
            continue;
        }

        // Get color range weights
        let (primary, secondary, blend) = get_color_range_weights(hsl.h);

        // Calculate weighted adjustments
        let hue_adj = adjustments.hue[primary] * (1.0 - blend) + adjustments.hue[secondary] * blend;
        let sat_adj = adjustments.saturation[primary] * (1.0 - blend)
            + adjustments.saturation[secondary] * blend;
        let lum_adj = adjustments.luminance[primary] * (1.0 - blend)
            + adjustments.luminance[secondary] * blend;

        // Apply adjustments (values are -100 to +100, need to scale)
        let mut new_hsl = hsl;

        // Hue shift (100 = 30° shift)
        new_hsl.h = (new_hsl.h + hue_adj * 0.3) % 360.0;
        if new_hsl.h < 0.0 {
            new_hsl.h += 360.0;
        }

        // Saturation adjustment (multiplicative)
        let sat_factor = 1.0 + sat_adj / 100.0;
        new_hsl.s = (new_hsl.s * sat_factor).clamp(0.0, 1.0);

        // Luminance adjustment (additive)
        new_hsl.l = (new_hsl.l + lum_adj / 200.0).clamp(0.0, 1.0);

        // Convert back to RGB
        let (new_r, new_g, new_b) = hsl_to_rgb(new_hsl);
        pixel[0] = new_r;
        pixel[1] = new_g;
        pixel[2] = new_b;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_hsl_roundtrip() {
        let test_cases = [
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Green
            (0.0, 0.0, 1.0), // Blue
            (1.0, 1.0, 1.0), // White
            (0.0, 0.0, 0.0), // Black
            (0.5, 0.5, 0.5), // Gray
            (1.0, 0.5, 0.0), // Orange
            (0.5, 0.0, 0.5), // Purple
        ];

        for (r, g, b) in test_cases {
            let hsl = rgb_to_hsl(r, g, b);
            let (r2, g2, b2) = hsl_to_rgb(hsl);

            assert!(
                (r - r2).abs() < 1e-5,
                "R mismatch for ({}, {}, {}): {} vs {}",
                r,
                g,
                b,
                r,
                r2
            );
            assert!(
                (g - g2).abs() < 1e-5,
                "G mismatch for ({}, {}, {}): {} vs {}",
                r,
                g,
                b,
                g,
                g2
            );
            assert!(
                (b - b2).abs() < 1e-5,
                "B mismatch for ({}, {}, {}): {} vs {}",
                r,
                g,
                b,
                b,
                b2
            );
        }
    }

    #[test]
    fn test_hsl_values() {
        // Red should be H=0, S=1, L=0.5
        let hsl = rgb_to_hsl(1.0, 0.0, 0.0);
        assert!((hsl.h - 0.0).abs() < 1e-5);
        assert!((hsl.s - 1.0).abs() < 1e-5);
        assert!((hsl.l - 0.5).abs() < 1e-5);

        // Green should be H=120, S=1, L=0.5
        let hsl = rgb_to_hsl(0.0, 1.0, 0.0);
        assert!((hsl.h - 120.0).abs() < 1e-5);
        assert!((hsl.s - 1.0).abs() < 1e-5);

        // Blue should be H=240, S=1, L=0.5
        let hsl = rgb_to_hsl(0.0, 0.0, 1.0);
        assert!((hsl.h - 240.0).abs() < 1e-5);
        assert!((hsl.s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_lab_roundtrip() {
        let test_cases = [
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Green
            (0.0, 0.0, 1.0), // Blue
            (1.0, 1.0, 1.0), // White
            (0.5, 0.5, 0.5), // Gray
            (0.8, 0.4, 0.2), // Orange-ish
        ];

        for (r, g, b) in test_cases {
            let lab = rgb_to_lab(r, g, b);
            let (r2, g2, b2) = lab_to_rgb(lab);

            // LAB roundtrip may have slightly more error due to matrix operations
            assert!(
                (r - r2).abs() < 1e-4,
                "R mismatch for ({}, {}, {}): {} vs {}",
                r,
                g,
                b,
                r,
                r2
            );
            assert!(
                (g - g2).abs() < 1e-4,
                "G mismatch for ({}, {}, {}): {} vs {}",
                r,
                g,
                b,
                g,
                g2
            );
            assert!(
                (b - b2).abs() < 1e-4,
                "B mismatch for ({}, {}, {}): {} vs {}",
                r,
                g,
                b,
                b,
                b2
            );
        }
    }

    #[test]
    fn test_lab_values() {
        // White should be L=100, a=0, b=0
        let lab = rgb_to_lab(1.0, 1.0, 1.0);
        assert!((lab.l - 100.0).abs() < 0.1);
        assert!(lab.a.abs() < 0.1);
        assert!(lab.b.abs() < 0.1);

        // Black should be L=0, a=0, b=0
        let lab = rgb_to_lab(0.0, 0.0, 0.0);
        assert!(lab.l.abs() < 0.1);
        assert!(lab.a.abs() < 0.1);
        assert!(lab.b.abs() < 0.1);

        // Gray should have a=0, b=0
        let lab = rgb_to_lab(0.5, 0.5, 0.5);
        assert!(lab.a.abs() < 0.1);
        assert!(lab.b.abs() < 0.1);
    }
}
