//! LAB (CIE L*a*b*) color space conversions and utilities

use super::conversions::Colorspace;

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
pub(crate) const D65_X: f32 = 0.95047;
pub(crate) const D65_Y: f32 = 1.00000;
pub(crate) const D65_Z: f32 = 1.08883;

/// sRGB to XYZ matrix (D65)
pub(crate) const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.119_192, 0.9503041],
];

/// XYZ to sRGB matrix (D65)
pub(crate) const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.969_266, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

/// Rec.2020 to XYZ matrix (D65)
/// Source: ITU-R BT.2020 specification
pub(crate) const REC2020_TO_XYZ: [[f32; 3]; 3] = [
    [0.6370, 0.1446, 0.1689],
    [0.2627, 0.6780, 0.0593],
    [0.0000, 0.0281, 1.0610],
];

/// XYZ to Rec.2020 matrix (D65)
/// Inverse of REC2020_TO_XYZ
pub(crate) const XYZ_TO_REC2020: [[f32; 3]; 3] = [
    [1.7167, -0.3557, -0.2534],
    [-0.6667, 1.6165, 0.0158],
    [0.0176, -0.0428, 0.9421],
];

/// ProPhoto RGB to XYZ matrix (D50, adapted to D65)
#[allow(clippy::excessive_precision)]
pub(crate) const PROPHOTO_TO_XYZ: [[f32; 3]; 3] = [
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
];

/// XYZ to ProPhoto RGB matrix (D65 adapted to D50)
pub(crate) const XYZ_TO_PROPHOTO: [[f32; 3]; 3] = [
    [1.3459433, -0.2556075, -0.0511118],
    [-0.5445989, 1.5081673, 0.0205351],
    [0.0000000, 0.0000000, 1.2118128],
];

/// Display P3 to XYZ matrix (D65)
pub(crate) const DISPLAYP3_TO_XYZ: [[f32; 3]; 3] = [
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
];

/// XYZ to Display P3 matrix (D65)
#[allow(clippy::excessive_precision)]
pub(crate) const XYZ_TO_DISPLAYP3: [[f32; 3]; 3] = [
    [2.4934969, -0.9313836, -0.4027108],
    [-0.8294890, 1.7626641, 0.0236247],
    [0.0358458, -0.0761724, 0.9568845],
];

/// Get the RGB to XYZ matrix for a colorspace
pub(crate) fn get_rgb_to_xyz_matrix(colorspace: Colorspace) -> &'static [[f32; 3]; 3] {
    match colorspace {
        Colorspace::LinearSRGB => &SRGB_TO_XYZ,
        Colorspace::LinearRec2020 => &REC2020_TO_XYZ,
        Colorspace::LinearProPhoto => &PROPHOTO_TO_XYZ,
        Colorspace::LinearDisplayP3 => &DISPLAYP3_TO_XYZ,
    }
}

/// Get the XYZ to RGB matrix for a colorspace
pub(crate) fn get_xyz_to_rgb_matrix(colorspace: Colorspace) -> &'static [[f32; 3]; 3] {
    match colorspace {
        Colorspace::LinearSRGB => &XYZ_TO_SRGB,
        Colorspace::LinearRec2020 => &XYZ_TO_REC2020,
        Colorspace::LinearProPhoto => &XYZ_TO_PROPHOTO,
        Colorspace::LinearDisplayP3 => &XYZ_TO_DISPLAYP3,
    }
}

/// Convert linear RGB to XYZ (D65) with explicit colorspace
#[inline]
pub(crate) fn linear_rgb_to_xyz_with_colorspace(
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
pub(crate) fn xyz_to_linear_rgb_with_colorspace(
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
