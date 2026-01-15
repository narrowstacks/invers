//! HSL (Hue-Saturation-Lightness) color space conversions and utilities

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
// 8-Color HSL Adjustments (Camera Raw style)
// =============================================================================

/// Get the color range index (0-7) and blend weight for a given hue
///
/// The 8 color ranges with their approximate hue centers:
/// - R (Reds): 0 degrees (wraps around 330-30 degrees)
/// - O (Oranges): 30 degrees
/// - Y (Yellows): 60 degrees
/// - G (Greens): 120 degrees
/// - A (Aquas): 180 degrees
/// - B (Blues): 240 degrees
/// - P (Purples): 285 degrees
/// - M (Magentas): 315 degrees
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

        // Hue shift (100 = 30 degree shift)
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
