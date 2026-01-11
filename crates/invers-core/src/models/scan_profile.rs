//! Scan profile types for capture source characteristics.

use serde::{Deserialize, Serialize};

use super::convert_options::InversionMode;

/// Scan profile defining capture source characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanProfile {
    /// Name of the scan profile
    pub name: String,

    /// Capture source type (e.g., "dslr", "mirrorless", "flatbed")
    pub source_type: String,

    /// Typical white level for this source
    pub white_level: Option<f32>,

    /// Typical black level for this source
    pub black_level: Option<f32>,

    /// Demosaic hints (for RAW sources)
    pub demosaic_hints: Option<DemosaicHints>,

    /// White balance hints
    pub white_balance_hints: Option<WhiteBalanceHints>,

    /// HSL adjustments specific to this scanner/source
    /// Used to compensate for scanner color characteristics
    #[serde(default)]
    pub hsl_adjustments: Option<HslAdjustments>,

    /// Default per-channel gamma values [R, G, B]
    /// Applied during levels adjustment
    #[serde(default)]
    pub default_gamma: Option<[f32; 3]>,

    /// Preferred inversion mode for this scanner type
    #[serde(default)]
    pub preferred_inversion_mode: Option<InversionMode>,
}

/// Demosaic processing hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemosaicHints {
    /// Demosaic algorithm preference
    pub algorithm: String,

    /// Quality vs speed preference (0.0 = fast, 1.0 = quality)
    pub quality: f32,
}

impl Default for DemosaicHints {
    fn default() -> Self {
        Self {
            algorithm: "ahd".to_string(),
            quality: 0.5,
        }
    }
}

/// White balance processing hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteBalanceHints {
    /// Auto white balance preference
    pub auto: bool,

    /// Manual color temperature (if not auto)
    pub temperature: Option<f32>,

    /// Tint adjustment
    pub tint: Option<f32>,
}

impl Default for WhiteBalanceHints {
    fn default() -> Self {
        Self {
            auto: true,
            temperature: None,
            tint: None,
        }
    }
}

/// 8-color HSL adjustments (Camera Raw style)
///
/// Adjusts Hue, Saturation, and Luminance for 8 color ranges:
/// - R (Reds): Hue ~0-30, 330-360
/// - O (Oranges): Hue ~30-60
/// - Y (Yellows): Hue ~60-90
/// - G (Greens): Hue ~90-150
/// - A (Aquas/Cyans): Hue ~150-210
/// - B (Blues): Hue ~210-270
/// - P (Purples): Hue ~270-300
/// - M (Magentas): Hue ~300-330
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HslAdjustments {
    /// Hue shifts for each of the 8 color ranges (-100 to +100)
    /// Order: [R, O, Y, G, A, B, P, M]
    #[serde(default)]
    pub hue: [f32; 8],

    /// Saturation adjustments for each of the 8 color ranges (-100 to +100)
    /// Order: [R, O, Y, G, A, B, P, M]
    #[serde(default)]
    pub saturation: [f32; 8],

    /// Luminance adjustments for each of the 8 color ranges (-100 to +100)
    /// Order: [R, O, Y, G, A, B, P, M]
    #[serde(default)]
    pub luminance: [f32; 8],
}

impl Default for HslAdjustments {
    fn default() -> Self {
        Self {
            hue: [0.0; 8],
            saturation: [0.0; 8],
            luminance: [0.0; 8],
        }
    }
}

impl HslAdjustments {
    /// Check if any adjustments are non-zero
    pub fn has_adjustments(&self) -> bool {
        self.hue.iter().any(|&v| v.abs() > 0.001)
            || self.saturation.iter().any(|&v| v.abs() > 0.001)
            || self.luminance.iter().any(|&v| v.abs() > 0.001)
    }

    /// Color range indices
    pub const REDS: usize = 0;
    pub const ORANGES: usize = 1;
    pub const YELLOWS: usize = 2;
    pub const GREENS: usize = 3;
    pub const AQUAS: usize = 4;
    pub const BLUES: usize = 5;
    pub const PURPLES: usize = 6;
    pub const MAGENTAS: usize = 7;
}

/// Orange mask profile for color negative film
///
/// The orange mask exists because real-world dyes have impurities that cause
/// cross-channel contamination. Film manufacturers add colored dye couplers
/// to compensate, creating the characteristic orange mask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskProfile {
    /// Magenta dye impurity level (blue absorption)
    /// Typical range: 0.3-0.5 for most color negative films
    /// Higher values indicate stronger orange mask (more blue absorbed by magenta layer)
    pub magenta_impurity: f32,

    /// Cyan dye impurity level (green absorption)
    /// Typical range: 0.2-0.4 for most color negative films
    /// Higher values indicate stronger mask contribution from cyan layer
    pub cyan_impurity: f32,

    /// Mask correction strength (0.0-1.0)
    /// 0.0 = no correction (equivalent to linear inversion)
    /// 1.0 = full correction based on calculated impurities
    pub correction_strength: f32,
}

impl Default for MaskProfile {
    fn default() -> Self {
        Self {
            magenta_impurity: 0.50, // Observable notebook default
            cyan_impurity: 0.30,    // Observable notebook default
            correction_strength: 1.0,
        }
    }
}

impl MaskProfile {
    /// Create a mask profile from base color ratios
    ///
    /// Uses the Observable notebook defaults for dye impurities (0.5 magenta, 0.3 cyan)
    /// but scales the correction_strength based on how strongly the base exhibits
    /// the orange mask characteristic (R > G > B pattern).
    ///
    /// The dye impurity values are properties of the film chemistry and are relatively
    /// consistent across film stocks. What varies is how much of the mask is present
    /// in any given scan (affected by exposure, development, scanner characteristics).
    pub fn from_base_medians(medians: &[f32; 3]) -> Self {
        let r = medians[0].max(0.001);
        let g = medians[1].max(0.001);
        let b = medians[2].max(0.001);

        // Use standard dye impurity values from Observable notebook
        // These represent typical color negative film chemistry
        let magenta_impurity = 0.50; // Magenta dye's blue absorption
        let cyan_impurity = 0.30; // Cyan dye's green absorption

        // Calculate how "orange" the base is to scale correction strength
        // A perfect orange mask has R > G > B with specific ratios
        // If the base is more neutral (R ≈ G ≈ B), reduce correction

        // Measure the "orange-ness": how much the base deviates from neutral gray
        // For a typical orange mask: R/G ≈ 1.4-1.8, G/B ≈ 1.1-1.4
        let rg_ratio = r / g;
        let gb_ratio = g / b;

        // Calculate a correction strength based on how orange the base is
        // Scale from 0.0 (neutral gray) to 1.0 (strong orange mask)
        // Typical color negative: R/G ≈ 1.6, G/B ≈ 1.2
        // We want full correction around these values, less for more extreme or neutral bases

        let rg_score = if rg_ratio > 1.0 {
            // Orange base should have R > G, ratio typically 1.3-2.0
            // Map 1.0-2.0 range to 0.0-1.0 score
            ((rg_ratio - 1.0) / 1.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let gb_score = if gb_ratio > 1.0 {
            // Orange base should have G > B, ratio typically 1.1-1.5
            // Map 1.0-1.5 range to 0.0-1.0 score
            ((gb_ratio - 1.0) / 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Combine scores - both need to be present for a true orange mask
        // Use geometric mean to require both conditions
        let orange_score = (rg_score * gb_score).sqrt();

        // Scale correction strength: don't go above 0.7 to avoid overcorrection
        // The Observable notebook's "correction" slider defaults to 0 and goes to 1
        // In practice, values around 0.5-0.7 work well
        let correction_strength = (orange_score * 0.7).clamp(0.0, 0.7);

        Self {
            magenta_impurity,
            cyan_impurity,
            correction_strength,
        }
    }

    /// Calculate the shadow floor values for each channel based on mask impurities
    ///
    /// Returns (red_floor, green_floor, blue_floor)
    /// - Red floor is always 0 (yellow coupler doesn't add extra absorption to red)
    /// - Green floor compensates for cyan's green absorption
    /// - Blue floor compensates for magenta's blue absorption
    pub fn calculate_shadow_floors(&self) -> (f32, f32, f32) {
        let green_floor =
            self.correction_strength * (self.cyan_impurity / (1.0 + self.cyan_impurity));
        let blue_floor =
            self.correction_strength * (self.magenta_impurity / (1.0 + self.magenta_impurity));

        (0.0, green_floor, blue_floor)
    }
}
