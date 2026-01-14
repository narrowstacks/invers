//! White balance calculations and application for the CB pipeline.
//!
//! Contains functions for calculating and applying white balance offsets
//! using various methods and tonality modes.

use super::logb;
use crate::models::{CbFilmCharacter, CbWbPreset, CbWbTonality};

/// Temp/tint multiplier (CB uses a global multiplier)
const TEMP_MULTIPLIER: f32 = 1.0;

// ============================================================
// White Balance Offset Calculations
// ============================================================

/// Calculate linear balance offsets from a neutral point
/// Returns per-channel offsets [R, G, B] to neutralize the color
pub fn calculate_linear_balance(
    neutral_rgb: [f32; 3],
    reference_channel: Option<usize>,
) -> [f32; 3] {
    let avg = if let Some(ch) = reference_channel {
        neutral_rgb[ch]
    } else {
        (neutral_rgb[0] + neutral_rgb[1] + neutral_rgb[2]) / 3.0
    };

    if neutral_rgb[0] == 0.0 && neutral_rgb[1] == 0.0 && neutral_rgb[2] == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // Return offsets scaled to 0-255 range (CB convention)
    [
        (avg - neutral_rgb[0]) * 255.0,
        (avg - neutral_rgb[1]) * 255.0,
        (avg - neutral_rgb[2]) * 255.0,
    ]
}

/// Calculate gamma balance exponents from a neutral point
/// Returns per-channel gamma exponents [R, G, B]
pub fn calculate_gamma_balance(
    neutral_rgb: [f32; 3],
    reference_channel: Option<usize>,
) -> [f32; 3] {
    let reference = if let Some(ch) = reference_channel {
        neutral_rgb[ch]
    } else {
        (neutral_rgb[0] + neutral_rgb[1] + neutral_rgb[2]) / 3.0
    };

    if neutral_rgb[0] == 0.0 && neutral_rgb[1] == 0.0 && neutral_rgb[2] == 0.0 {
        return [1.0, 1.0, 1.0];
    }

    // gamma = 1 / log_ref(channel_value)
    [
        1.0 / logb(reference, neutral_rgb[0].max(0.001)),
        1.0 / logb(reference, neutral_rgb[1].max(0.001)),
        1.0 / logb(reference, neutral_rgb[2].max(0.001)),
    ]
}

/// Calculate WB offsets based on tonality mode
pub fn calculate_wb_offsets(wb_temp: f32, wb_tint: f32, tonality: CbWbTonality) -> [f32; 3] {
    // Scale temp/tint from -100..100 to internal range
    let temp_scaled = -wb_temp / 255.0 / TEMP_MULTIPLIER;
    let tint_scaled = -wb_tint / 255.0 / TEMP_MULTIPLIER;

    match tonality {
        CbWbTonality::NeutralDensity => [
            -temp_scaled / 2.0 - tint_scaled / 2.0,
            -temp_scaled / 2.0 + tint_scaled / 2.0,
            temp_scaled / 2.0 - tint_scaled / 2.0,
        ],
        CbWbTonality::SubtractDensity => [-temp_scaled - tint_scaled, -temp_scaled, -tint_scaled],
        CbWbTonality::TempTintDensity => [-tint_scaled / 2.0, 0.0, temp_scaled - tint_scaled / 2.0],
    }
}

/// Calculate gamma multipliers for gamma-based WB methods
pub fn calculate_wb_gamma(wb_temp: f32, wb_tint: f32, tonality: CbWbTonality) -> [f32; 3] {
    let temp_factor = (wb_temp / 255.0 / TEMP_MULTIPLIER).clamp(-0.99, 0.99);
    let tint_factor = (wb_tint / 255.0 / TEMP_MULTIPLIER).clamp(-0.99, 0.99);

    match tonality {
        CbWbTonality::NeutralDensity => [
            1.0 / (1.0 - temp_factor * 0.5) / (1.0 - tint_factor * 0.5),
            1.0 / (1.0 - temp_factor * 0.5) * (1.0 - tint_factor * 0.5),
            (1.0 - temp_factor * 0.5) / (1.0 - tint_factor * 0.5),
        ],
        CbWbTonality::SubtractDensity => [
            1.0 / (1.0 - temp_factor) / (1.0 - tint_factor),
            1.0 / (1.0 - temp_factor),
            1.0 / (1.0 - tint_factor),
        ],
        CbWbTonality::TempTintDensity => [1.0, 1.0 - tint_factor, 1.0 - temp_factor],
    }
}

// ============================================================
// WB Point Analysis
// ============================================================

/// Analyzed white balance points for different presets.
/// Similar to CB's smartColorNeutral, smartColorWarming, smartColorCooling.
#[derive(Debug, Clone)]
pub struct AnalyzedWbPoints {
    /// Neutral point - most balanced (green-referenced)
    pub neutral: [f32; 3],
    /// Warm point - shifted toward warmer tones
    pub warm: [f32; 3],
    /// Cool point - shifted toward cooler tones
    pub cool: [f32; 3],
}

/// Analyze the image to compute WB points for different presets.
///
/// Returns neutral, warm, and cool variants based on the image histogram.
pub fn analyze_wb_points(data: &[f32], channels: u8) -> AnalyzedWbPoints {
    let ch = channels as usize;
    let pixel_count = data.len() / ch;

    if pixel_count == 0 {
        return AnalyzedWbPoints {
            neutral: [0.5, 0.5, 0.5],
            warm: [0.5, 0.5, 0.5],
            cool: [0.5, 0.5, 0.5],
        };
    }

    // Calculate channel averages (for neutral point)
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;

    // Also track weighted sums for shadow/highlight biased points
    let mut r_shadow_sum = 0.0f64;
    let mut g_shadow_sum = 0.0f64;
    let mut b_shadow_sum = 0.0f64;
    let mut shadow_weight = 0.0f64;

    let mut r_highlight_sum = 0.0f64;
    let mut g_highlight_sum = 0.0f64;
    let mut b_highlight_sum = 0.0f64;
    let mut highlight_weight = 0.0f64;

    for pixel in data.chunks(ch) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let lum = (r + g + b) / 3.0;

        r_sum += r;
        g_sum += g;
        b_sum += b;

        // Shadow-weighted (warmer look - shadows influence more)
        let shadow_w = (1.0 - lum).powi(2);
        r_shadow_sum += r * shadow_w;
        g_shadow_sum += g * shadow_w;
        b_shadow_sum += b * shadow_w;
        shadow_weight += shadow_w;

        // Highlight-weighted (cooler look - highlights influence more)
        let highlight_w = lum.powi(2);
        r_highlight_sum += r * highlight_w;
        g_highlight_sum += g * highlight_w;
        b_highlight_sum += b * highlight_w;
        highlight_weight += highlight_w;
    }

    let count = pixel_count as f64;

    // Neutral: simple average
    let neutral = [
        (r_sum / count) as f32,
        (g_sum / count) as f32,
        (b_sum / count) as f32,
    ];

    // Warm: shadow-weighted average (emphasizes warmer shadows)
    let warm = if shadow_weight > 0.0 {
        [
            (r_shadow_sum / shadow_weight) as f32,
            (g_shadow_sum / shadow_weight) as f32,
            (b_shadow_sum / shadow_weight) as f32,
        ]
    } else {
        neutral
    };

    // Cool: highlight-weighted average (emphasizes cooler highlights)
    let cool = if highlight_weight > 0.0 {
        [
            (r_highlight_sum / highlight_weight) as f32,
            (g_highlight_sum / highlight_weight) as f32,
            (b_highlight_sum / highlight_weight) as f32,
        ]
    } else {
        neutral
    };

    AnalyzedWbPoints {
        neutral,
        warm,
        cool,
    }
}

// ============================================================
// WB Preset Application
// ============================================================

/// Apply white balance based on the selected preset.
///
/// Returns the WB offsets to apply (in 0-255 scale like CB).
pub fn calculate_wb_preset_offsets(
    wb_preset: CbWbPreset,
    wb_points: &AnalyzedWbPoints,
    _film_character: CbFilmCharacter,
) -> [f32; 3] {
    match wb_preset {
        CbWbPreset::None => [0.0, 0.0, 0.0],

        CbWbPreset::AutoColor | CbWbPreset::AutoNeutral => {
            // Use neutral point with green reference
            calculate_linear_balance(wb_points.neutral, Some(1))
        }

        CbWbPreset::AutoWarm => {
            // To make image warmer, balance using the cool point as reference
            // This shifts colors away from cool (blue) towards warm (red/yellow)
            calculate_linear_balance(wb_points.cool, Some(1))
        }

        CbWbPreset::AutoCool => {
            // To make image cooler, balance using the warm point as reference
            // This shifts colors away from warm (red/yellow) towards cool (blue)
            calculate_linear_balance(wb_points.warm, Some(1))
        }

        CbWbPreset::AutoMix => {
            // Mix of neutral and warm
            let neutral_offsets = calculate_linear_balance(wb_points.neutral, Some(1));
            let warm_offsets = calculate_linear_balance(wb_points.warm, Some(1));
            [
                (neutral_offsets[0] + warm_offsets[0]) / 2.0,
                (neutral_offsets[1] + warm_offsets[1]) / 2.0,
                (neutral_offsets[2] + warm_offsets[2]) / 2.0,
            ]
        }

        CbWbPreset::Standard => {
            // Generic color with neutral balance
            let base_offsets = calculate_linear_balance(wb_points.neutral, Some(1));
            apply_film_character_to_offsets(base_offsets, CbFilmCharacter::GenericColor)
        }

        CbWbPreset::Kodak => {
            let base_offsets = calculate_linear_balance(wb_points.neutral, Some(1));
            apply_film_character_to_offsets(base_offsets, CbFilmCharacter::Kodak)
        }

        CbWbPreset::Fuji => {
            let base_offsets = calculate_linear_balance(wb_points.neutral, Some(1));
            apply_film_character_to_offsets(base_offsets, CbFilmCharacter::Fuji)
        }

        CbWbPreset::CineT => {
            let base_offsets = calculate_linear_balance(wb_points.neutral, Some(1));
            apply_film_character_to_offsets(base_offsets, CbFilmCharacter::Cinestill800T)
        }

        CbWbPreset::CineD => {
            let base_offsets = calculate_linear_balance(wb_points.neutral, Some(1));
            apply_film_character_to_offsets(base_offsets, CbFilmCharacter::Cinestill50D)
        }

        CbWbPreset::Custom => {
            // Custom uses wb_temp/wb_tint directly, not computed here
            [0.0, 0.0, 0.0]
        }
    }
}

/// Apply film character adjustments to WB offsets.
pub fn apply_film_character_to_offsets(offsets: [f32; 3], character: CbFilmCharacter) -> [f32; 3] {
    let (cyan_adj, magenta_adj, yellow_adj) = character.cmy_adjustments();

    // CMY adjustments map to RGB offsets:
    // Cyan affects Red (negative)
    // Magenta affects Green (negative)
    // Yellow affects Blue (negative)
    [
        offsets[0] - cyan_adj as f32,
        offsets[1] - magenta_adj as f32,
        offsets[2] - yellow_adj as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_balance() {
        // Test that equal values produce gamma of 1
        let gamma = calculate_gamma_balance([0.5, 0.5, 0.5], None);
        assert!((gamma[0] - 1.0).abs() < 0.01);
        assert!((gamma[1] - 1.0).abs() < 0.01);
        assert!((gamma[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_balance() {
        // Test that equal values produce zero offsets
        let offsets = calculate_linear_balance([0.5, 0.5, 0.5], None);
        assert!((offsets[0]).abs() < 0.01);
        assert!((offsets[1]).abs() < 0.01);
        assert!((offsets[2]).abs() < 0.01);
    }

    #[test]
    fn test_linear_balance_with_reference() {
        // Test with green reference channel
        let offsets = calculate_linear_balance([0.3, 0.5, 0.7], Some(1));
        // Green is reference, so its offset should be 0
        assert!((offsets[1]).abs() < 0.01);
        // Red should get positive offset (0.5 - 0.3) * 255
        assert!(offsets[0] > 0.0);
        // Blue should get negative offset (0.5 - 0.7) * 255
        assert!(offsets[2] < 0.0);
    }

    #[test]
    fn test_analyze_wb_points() {
        // Create uniform gray image
        let data = vec![0.5f32; 300]; // 100 pixels, 3 channels each
        let points = analyze_wb_points(&data, 3);

        assert!((points.neutral[0] - 0.5).abs() < 0.01);
        assert!((points.neutral[1] - 0.5).abs() < 0.01);
        assert!((points.neutral[2] - 0.5).abs() < 0.01);
    }
}
