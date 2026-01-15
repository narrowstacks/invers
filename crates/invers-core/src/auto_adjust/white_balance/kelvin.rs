//! Temperature-based white balance conversion
//!
//! Provides conversion from color temperature in Kelvin to RGB multipliers
//! and functions for applying white balance based on temperature and tint.

use crate::auto_adjust::parallel::parallel_for_each_chunk_mut;

/// Convert color temperature in Kelvin to RGB multipliers
///
/// Based on Tanner Helland's algorithm which approximates the Planckian locus.
/// Reference: https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
///
/// # Arguments
/// * `kelvin` - Color temperature in Kelvin (1000-40000)
///
/// # Returns
/// RGB multipliers normalized to green channel = 1.0
#[allow(clippy::excessive_precision)] // Published constants from Tanner Helland algorithm
pub fn kelvin_to_rgb_multipliers(kelvin: f32) -> [f32; 3] {
    // Clamp temperature to valid range
    let temp = (kelvin / 100.0).clamp(10.0, 400.0);

    // Calculate RGB values using polynomial approximation
    let (r, g, b) = if temp <= 66.0 {
        // For temperatures <= 6600K
        let r = 255.0;
        let g = 99.4708025861 * temp.ln() - 161.1195681661;
        let b = if temp <= 19.0 {
            0.0
        } else {
            138.5177312231 * (temp - 10.0).ln() - 305.0447927307
        };
        (r, g.clamp(0.0, 255.0), b.clamp(0.0, 255.0))
    } else {
        // For temperatures > 6600K
        let r = 329.698727446 * (temp - 60.0).powf(-0.1332047592);
        let g = 288.1221695283 * (temp - 60.0).powf(-0.0755148492);
        let b = 255.0;
        (r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), b)
    };

    // Normalize to 0-1 range
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;

    // Convert to multipliers (normalize to green)
    // To correct from temperature T to neutral (D65 ~6500K), we need
    // to apply the inverse of what that temperature produces
    let g_ref = g.max(0.001);
    [g_ref / r.max(0.001), 1.0, g_ref / b.max(0.001)]
}

/// Apply white balance from temperature and tint
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data
/// * `channels` - Number of channels (must be 3)
/// * `temperature` - Color temperature in Kelvin (e.g., 5500 for daylight)
/// * `tint` - Green-magenta tint adjustment (-100 to +100, 0 = neutral)
///
/// # Returns
/// The RGB multipliers that were applied
pub fn apply_white_balance_from_temperature(
    data: &mut [f32],
    channels: u8,
    temperature: f32,
    tint: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("apply_white_balance_from_temperature only supports 3-channel RGB images");
    }

    // Get base multipliers from temperature
    let mut multipliers = kelvin_to_rgb_multipliers(temperature);

    // Apply tint adjustment (affects green-magenta axis)
    // Positive tint = more green, negative = more magenta
    let tint_factor = 1.0 + tint / 200.0; // ±0.5 adjustment range
    multipliers[1] *= tint_factor;

    // Apply multipliers to the image
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] *= multipliers[0];
        pixel[1] *= multipliers[1];
        pixel[2] *= multipliers[2];
    });

    multipliers
}
